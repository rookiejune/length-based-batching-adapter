from __future__ import annotations

import unittest
import json
import logging
import tempfile
import warnings
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from lba import AdaptiveConfig, LBA
from lba._distributed_cost import (
    PlanMetadata,
    RecordMetadata,
    cost_window_stats,
    match_cost_block,
)
from lba.config import LBAConfig
from lba.distributed import (
    DistributedBatchCoordinator,
)
from lba._distributed_flush import (
    DistributedFlushPlanner,
    split_plans_to_count,
)
from lba._records import BatchPlan, PlanReason, SampleRecord
from lba.source import build_batch_loader

try:
    from lightning.pytorch.utilities.data import _update_dataloader
except ImportError:
    _update_dataloader = None


def collate_indexed_lengths(
    samples: list[tuple[int, int]],
) -> dict[str, torch.Tensor]:
    return {
        "indices": torch.tensor([sample[0] for sample in samples]),
        "lengths": torch.tensor(
            [sample[1] for sample in samples], dtype=torch.float32
        ).unsqueeze(1),
    }


def sample_length(sample: int | tuple[int, int]) -> int:
    if isinstance(sample, tuple):
        return sample[1]
    return sample


def quadratic_cost(max_length: int, batch_size: int) -> int:
    return max_length * max_length * batch_size


class LengthDataset(Dataset[tuple[int, int]]):
    def __init__(self) -> None:
        self.lengths = [100, 1, 100, 1, 100, 1, 100, 1]

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, index: int) -> tuple[int, int]:
        return index, self.lengths[index]


class PartialLengthDataset(LengthDataset):
    def __init__(self) -> None:
        self.lengths = [100, 1] * 8


class ThroughputLengthDataset(LengthDataset):
    def __init__(self) -> None:
        self.lengths = [4, 5, 4, 5, 5, 5, 6, 5]


def build_ddp_loader(
    case: str,
    rank: int,
    world_size: int,
    output_path: Path,
) -> LBA:
    lba_kwargs = {
        "len_fn": sample_length,
        "max_padded_length": 15 if case == "throughput" else 100,
        "max_padding_ratio": 0.0,
        "max_cache_samples": 1024 if case == "throughput" else 1,
        "prefetch_batches": 2,
        "planner_mode": "throughput" if case == "throughput" else "quality",
        "max_candidate_windows": 1 if case == "throughput" else None,
        "limited_search_fallback_after": 8 if case == "throughput" else None,
        "limited_search_fallback_pool_size": (
            1024 if case == "throughput" else None
        ),
        "spill_dir": output_path / "shared-spill",
        "log_dir": output_path / f"logs-{case}-rank{rank}",
    }
    if case in (
        "cost",
        "global_cost",
        "global_cost_partial",
        "global_cost_limit",
    ):
        lba_kwargs.pop("max_padded_length")
        lba_kwargs.update(
            {
                "cost_fn": quadratic_cost,
                "max_batch_cost": 20_000,
                "max_cache_samples": 1024,
            }
        )
        if case == "cost":
            lba_kwargs["cost_window_batches"] = 2
        else:
            lba_kwargs["distributed_cost_window_batches"] = (
                3 if case == "global_cost_partial" else 2
            )
        if case == "global_cost_limit":
            lba_kwargs["max_batches"] = 1
    if case == "global_cost_flush":
        lba_kwargs.update(
            {
                "distributed_cost_window_batches": 2,
                "max_cache_samples": 1024,
            }
        )
    if case in (
        "map",
        "cost",
        "global_cost",
        "global_cost_partial",
        "global_cost_flush",
        "global_cost_limit",
        "throughput",
    ):
        dataset = (
            PartialLengthDataset()
            if case == "global_cost_partial"
            else ThroughputLengthDataset()
            if case == "throughput"
            else LengthDataset()
        )
        loader = LBA(
            dataset,
            batch_size=2,
            collate_fn=collate_indexed_lengths,
            num_workers=0,
            **lba_kwargs,
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        if _update_dataloader is None:
            raise RuntimeError("Lightning is required for the map-style DDP smoke test.")
        return _update_dataloader(loader, sampler)
    raise ValueError(f"Unknown DDP smoke case: {case}")


def run_ddp_smoke_worker(
    rank: int,
    world_size: int,
    init_file: str,
    output_dir: str,
    case: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    try:
        output_path = Path(output_dir)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loader = build_ddp_loader(case, rank, world_size, output_path)

        model = DistributedDataParallel(nn.Linear(1, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        steps = 0
        batch_sizes: list[int] = []
        padded_lengths: list[int] = []
        estimated_costs: list[int] = []
        sample_indices: list[int] = []
        for batch in loader:
            if isinstance(batch, dict):
                sample_indices.extend(int(index) for index in batch["indices"].tolist())
                model_batch = batch["lengths"]
            else:
                model_batch = batch
            optimizer.zero_grad(set_to_none=True)
            loss = model(model_batch).sum()
            loss.backward()
            optimizer.step()
            steps += 1
            batch_sizes.append(int(model_batch.shape[0]))
            padded_lengths.append(
                int(model_batch.max().item()) * int(model_batch.shape[0])
            )
            if case in (
                "cost",
                "global_cost",
                "global_cost_partial",
                "global_cost_limit",
            ):
                estimated_costs.append(
                    quadratic_cost(
                        int(model_batch.max().item()),
                        int(model_batch.shape[0]),
                    )
                )
            else:
                estimated_costs.append(padded_lengths[-1])

        dist.barrier()
        spill_dir = output_path / "shared-spill" / f"rank-{rank:05d}"
        events = [
            json.loads(line)
            for line in Path(loader.log_event_path).read_text().splitlines()
        ]
        result = {
            "rank": rank,
            "steps": steps,
            "batch_sizes": batch_sizes,
            "padded_lengths": padded_lengths,
            "estimated_costs": estimated_costs,
            "sample_indices": sample_indices,
            "spill_dir_exists": spill_dir.exists(),
            "no_ready_calls": loader.last_planner_stats.no_ready_call_count,
            "distributed_cost_blocks": sum(
                event["event"] == "distributed_cost_block" for event in events
            ),
            "distributed_cost_block_events": [
                event for event in events if event["event"] == "distributed_cost_block"
            ],
            "source_uses_injected_sampler": (
                case
                not in (
                    "map",
                    "cost",
                    "global_cost",
                    "global_cost_partial",
                    "global_cost_flush",
                    "global_cost_limit",
                    "throughput",
                )
                or loader._source_loader.batch_sampler.sampler is loader.sampler
            ),
        }
        (output_path / f"{case}-rank{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def run_cost_window_mismatch_worker(
    rank: int,
    world_size: int,
    init_file: str,
    output_dir: str,
    mismatch: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    try:
        dataset = LengthDataset()
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        loader = LBA(
            dataset,
            len_fn=sample_length,
            batch_size=2,
            sampler=sampler,
            collate_fn=collate_indexed_lengths,
            max_padded_length=200,
            cost_window_batches=(rank + 1 if mismatch == "local" else 1),
            distributed_cost_window_batches=(
                2 if mismatch == "distributed" and rank != 0 else None
            ),
            max_batches=(rank if mismatch == "max_batches" else None),
            prefetch_batches=(2 if mismatch == "max_batches" else 0),
            log_dir=Path(output_dir) / f"logs-window-mismatch-rank{rank}",
        )
        try:
            list(loader)
        except RuntimeError as error:
            result = {"error": str(error)}
        else:
            result = {"error": None}
        (Path(output_dir) / f"window-mismatch-{mismatch}-rank{rank}.json").write_text(
            json.dumps(result)
        )
    finally:
        dist.destroy_process_group()


class DistributedCoordinatorTest(unittest.TestCase):
    def test_uses_index_flush_only_when_all_ranks_have_indices(self) -> None:
        coordinator = DistributedBatchCoordinator(
            dataloader=DataLoader(["a"], batch_size=1),
            config=LBAConfig(),
            logger=None,
        )
        records = [SampleRecord("a", 1, 0, index=0)]

        with patch.object(
            coordinator,
            "_distributed_int_reduce",
            return_value=0,
        ):
            self.assertFalse(coordinator._all_ranks_have_record_indices(records))

        with patch.object(
            coordinator,
            "_distributed_int_reduce",
            return_value=1,
        ):
            self.assertTrue(coordinator._all_ranks_have_record_indices(records))

    def test_disables_index_flush_without_dataloader(self) -> None:
        coordinator = DistributedBatchCoordinator(
            dataloader=None,
            config=LBAConfig(),
            logger=None,
        )
        records = [SampleRecord("a", 1, 0, index=0)]

        self.assertFalse(coordinator._all_ranks_have_record_indices(records))

    def test_detects_when_all_ranks_reach_batch_limit(self) -> None:
        coordinator = DistributedBatchCoordinator(
            dataloader=None,
            config=LBAConfig(),
            logger=None,
        )

        with patch.object(
            coordinator,
            "_distributed_int_reduce",
            return_value=2,
        ):
            with patch.object(coordinator, "_world_size", return_value=2):
                self.assertTrue(coordinator.all_ranks_reached_batch_limit(True))

    def test_rejects_mismatched_distributed_cost_budget(self) -> None:
        coordinator = DistributedBatchCoordinator(
            dataloader=None,
            config=LBAConfig(
                cost_fn=quadratic_cost,
                max_batch_cost=32,
            ),
            logger=None,
        )

        with patch.object(
            coordinator,
            "_distributed_int_reduce",
            side_effect=[32, 16],
        ):
            with self.assertRaisesRegex(RuntimeError, "identical max_batch_cost"):
                coordinator.sync_max_batch_cost(32)

    def test_rejects_mismatched_distributed_cost_window(self) -> None:
        coordinator = DistributedBatchCoordinator(
            dataloader=None,
            config=LBAConfig(),
            logger=None,
        )

        with patch.object(
            coordinator,
            "_distributed_ints_reduce",
            side_effect=[(1, 4, 0, -1), (1, 0, 0, -1)],
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "identical distributed_cost_window_batches",
            ):
                coordinator.validate_iteration_configuration(
                    cost_window_batches=1,
                    distributed_cost_window_batches=None,
                    max_batches=None,
                )

    def test_accepts_matching_distributed_cost_window(self) -> None:
        coordinator = DistributedBatchCoordinator(
            dataloader=None,
            config=LBAConfig(distributed_cost_window_batches=4),
            logger=None,
        )

        with patch.object(
            coordinator,
            "_distributed_ints_reduce",
            side_effect=[(1, 4, 0, -1), (1, 4, 0, -1)],
        ):
            coordinator.validate_iteration_configuration(
                cost_window_batches=1,
                distributed_cost_window_batches=4,
                max_batches=None,
            )

    def test_rejects_mismatched_adaptive_cost_window_enabled(self) -> None:
        coordinator = DistributedBatchCoordinator(
            dataloader=None,
            config=LBAConfig(),
            logger=None,
        )

        with patch.object(
            coordinator,
            "_distributed_ints_reduce",
            side_effect=[(1, 0, 1, -1), (1, 0, 0, -1)],
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "identical adaptive distributed cost-window",
            ):
                coordinator.validate_iteration_configuration(
                    cost_window_batches=1,
                    distributed_cost_window_batches=None,
                    adaptive_distributed_cost_window_enabled=True,
                    max_batches=None,
                )

    def test_rejects_mismatched_adaptive_cost_window(self) -> None:
        coordinator = DistributedBatchCoordinator(
            dataloader=None,
            config=LBAConfig(
                adaptive=AdaptiveConfig(distributed_cost_window_batches=None)
            ),
            logger=None,
        )

        def gather(output, local_fields, *, group):
            self.assertIsNone(group)
            output[0] = local_fields
            output[1] = {
                **local_fields,
                "high_spread_ratio": 0.5,
            }

        with (
            patch.object(coordinator, "_world_size", return_value=2),
            patch.object(coordinator, "_metadata_process_group", return_value=None),
            patch(
                "lba.distributed.dist.all_gather_object",
                side_effect=gather,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "identical adaptive configuration",
            ):
                coordinator.validate_adaptive_configuration()

    def test_accepts_matching_adaptive_cost_window(self) -> None:
        coordinator = DistributedBatchCoordinator(
            dataloader=None,
            config=LBAConfig(
                adaptive=AdaptiveConfig(distributed_cost_window_batches=None)
            ),
            logger=None,
        )

        def gather(output, local_fields, *, group):
            self.assertIsNone(group)
            output[0] = local_fields
            output[1] = local_fields

        with (
            patch.object(coordinator, "_world_size", return_value=2),
            patch.object(coordinator, "_metadata_process_group", return_value=None),
            patch(
                "lba.distributed.dist.all_gather_object",
                side_effect=gather,
            ),
        ):
            coordinator.validate_adaptive_configuration()

    def test_cost_window_stats_measure_matching_improvement(self) -> None:
        gathered = [
            [
                PlanMetadata(
                    records=(RecordMetadata(index=0, length=100, arrival_id=0),),
                    reason=PlanReason.PLANNED,
                    estimated_cost=100,
                ),
                PlanMetadata(
                    records=(RecordMetadata(index=1, length=100, arrival_id=1),),
                    reason=PlanReason.PLANNED,
                    estimated_cost=100,
                ),
            ],
            [
                PlanMetadata(
                    records=(RecordMetadata(index=2, length=2, arrival_id=2),),
                    reason=PlanReason.PLANNED,
                    estimated_cost=2,
                ),
                PlanMetadata(
                    records=(RecordMetadata(index=3, length=2, arrival_id=3),),
                    reason=PlanReason.PLANNED,
                    estimated_cost=2,
                ),
            ],
        ]
        assigned = match_cost_block(gathered, step_offset=0)

        stats = cost_window_stats(gathered, assigned)

        self.assertEqual(stats.block_size, 2)
        self.assertEqual(stats.source_mean_step_spread, 98)
        self.assertEqual(stats.matched_mean_step_spread, 0)
        self.assertEqual(stats.improvement_ratio, 1.0)
        self.assertEqual(stats.remote_plan_count, 2)

    def test_rejects_mismatched_local_cost_window_and_max_batches(self) -> None:
        coordinator = DistributedBatchCoordinator(
            dataloader=None,
            config=LBAConfig(),
            logger=None,
        )

        with patch.object(
            coordinator,
            "_distributed_ints_reduce",
            side_effect=[(2, 0, 0, -1), (1, 0, 0, -1)],
        ):
            with self.assertRaisesRegex(RuntimeError, "identical cost_window_batches"):
                coordinator.validate_iteration_configuration(
                    cost_window_batches=1,
                    distributed_cost_window_batches=None,
                    max_batches=None,
                )

        with patch.object(
            coordinator,
            "_distributed_ints_reduce",
            side_effect=[(1, 0, 0, 8), (1, 0, 0, 4)],
        ):
            with self.assertRaisesRegex(RuntimeError, "identical max_batches"):
                coordinator.validate_iteration_configuration(
                    cost_window_batches=1,
                    distributed_cost_window_batches=None,
                    max_batches=4,
                )

    def test_cost_matching_gathers_before_rejecting_missing_indices(self) -> None:
        coordinator = DistributedBatchCoordinator(
            dataloader=DataLoader(["a"], batch_size=1),
            config=LBAConfig(distributed_cost_window_batches=2),
            logger=None,
        )
        plan = BatchPlan(
            records=(SampleRecord("a", 1, 0, index=None),),
            raw_length_sum=1,
            padded_length=1,
            padding_length=0,
            padding_ratio=0.0,
            reason=PlanReason.PLANNED,
            estimated_cost=1,
        )

        def gather(output, local_metadata, *, group):
            self.assertIsNone(group)
            output[0] = local_metadata

        with (
            patch.object(coordinator, "_world_size", return_value=1),
            patch.object(coordinator, "_rank", return_value=0),
            patch.object(coordinator, "_metadata_process_group", return_value=None),
            patch(
                "lba.distributed.dist.all_gather_object",
                side_effect=gather,
            ) as all_gather,
        ):
            with self.assertRaisesRegex(RuntimeError, "map-style sample indices"):
                coordinator.match_cost_plans([plan], step_offset=0)

        all_gather.assert_called_once()

    def test_materializes_remote_cost_plan_from_metadata(self) -> None:
        coordinator = DistributedBatchCoordinator(
            dataloader=DataLoader(["x", "aa", "bbbbb"], batch_size=1),
            config=LBAConfig(distributed_cost_window_batches=2),
            logger=None,
            len_fn=len,
        )
        metadata = PlanMetadata(
            records=(
                RecordMetadata(index=1, length=2, arrival_id=4),
                RecordMetadata(index=2, length=5, arrival_id=5),
            ),
            reason=PlanReason.OVERSIZED,
            estimated_cost=77,
        )

        plan = coordinator._materialize_cost_plan(metadata)

        self.assertEqual(plan.samples, [1, 2])
        self.assertEqual(plan.raw_length_sum, 7)
        self.assertEqual(plan.padded_length, 10)
        self.assertEqual(plan.padding_length, 3)
        self.assertEqual(plan.padding_ratio, 0.3)
        self.assertEqual(plan.reason, PlanReason.OVERSIZED)
        self.assertEqual(plan.estimated_cost, 77)

    def test_rejects_materialized_sample_length_changes(self) -> None:
        coordinator = DistributedBatchCoordinator(
            dataloader=DataLoader(["x", "aaa"], batch_size=1),
            config=LBAConfig(distributed_cost_window_batches=2),
            logger=None,
            len_fn=len,
        )
        metadata = PlanMetadata(
            records=(RecordMetadata(index=1, length=2, arrival_id=0),),
            reason=PlanReason.PLANNED,
            estimated_cost=2,
        )

        plan = coordinator._materialize_cost_plan(metadata)

        with self.assertRaisesRegex(RuntimeError, "changed effective length"):
            next(iter(build_batch_loader(coordinator.dataloader, [plan], len)))

    def test_custom_cost_final_flush_recomputes_split_cost(self) -> None:
        flush_planner = DistributedFlushPlanner(
            config=LBAConfig(
                cost_fn=quadratic_cost,
                max_batch_cost=32,
                drop_last_flush=False,
            ),
            logger=None,
            event_writer=None,
        )
        records = [
            SampleRecord(str(index), 4, index)
            for index in range(4)
        ]

        rank_plans = [
            flush_planner.assigned_plans(
                records,
                32,
                rank=rank,
                world_size=2,
            )
            for rank in range(2)
        ]

        plans = [plan for rank in rank_plans for plan in rank]
        self.assertEqual(
            sorted(sample for plan in plans for sample in plan.samples),
            ["0", "1", "2", "3"],
        )
        self.assertTrue(all(plan.estimated_cost == 32 for plan in plans))

    def test_spill_dir_is_scoped_by_rank_when_distributed(self) -> None:
        coordinator = DistributedBatchCoordinator(
            dataloader=None,
            config=LBAConfig(spill_dir="storage/spill"),
            logger=None,
        )

        with (
            patch.object(DistributedBatchCoordinator, "is_initialized", return_value=True),
            patch("lba.distributed.dist.get_rank", return_value=3),
        ):
            self.assertEqual(
                coordinator.spill_dir_for_rank(),
                Path("storage/spill") / "rank-00003",
            )

    def test_spill_dir_is_unchanged_without_distributed(self) -> None:
        coordinator = DistributedBatchCoordinator(
            dataloader=None,
            config=LBAConfig(spill_dir="storage/spill"),
            logger=None,
        )

        with patch.object(
            DistributedBatchCoordinator,
            "is_initialized",
            return_value=False,
        ):
            self.assertEqual(coordinator.spill_dir_for_rank(), "storage/spill")

    def test_uses_gloo_metadata_group_with_nccl_default_group(self) -> None:
        coordinator = DistributedBatchCoordinator(
            dataloader=None,
            config=LBAConfig(),
            logger=None,
        )
        metadata_group = object()

        with (
            patch.object(DistributedBatchCoordinator, "is_initialized", return_value=True),
            patch("lba.distributed.dist.get_backend", return_value="nccl"),
            patch("lba.distributed.dist.is_gloo_available", return_value=True),
            patch("lba.distributed.dist.new_group", return_value=metadata_group) as new_group,
            patch("lba.distributed.dist.get_rank", return_value=0),
            patch("lba.distributed.dist.get_world_size", return_value=1),
        ):
            self.assertIs(coordinator._metadata_process_group(), metadata_group)
            self.assertEqual(
                coordinator._distributed_tensor_device(metadata_group),
                torch.device("cpu"),
            )

        new_group.assert_called_once_with(backend="gloo")

    def test_gloo_default_group_is_reused_without_background_prefetch(self) -> None:
        coordinator = DistributedBatchCoordinator(
            dataloader=None,
            config=LBAConfig(),
            logger=None,
        )

        with (
            patch.object(DistributedBatchCoordinator, "is_initialized", return_value=True),
            patch("lba.distributed.dist.get_backend", return_value="gloo"),
            patch("lba.distributed.dist.new_group") as new_group,
        ):
            self.assertIsNone(coordinator._metadata_process_group())

        new_group.assert_not_called()

    def test_background_prefetch_uses_isolated_gloo_metadata_group(self) -> None:
        coordinator = DistributedBatchCoordinator(
            dataloader=None,
            config=LBAConfig(),
            logger=None,
        )
        metadata_group = object()

        with (
            patch.object(DistributedBatchCoordinator, "is_initialized", return_value=True),
            patch("lba.distributed.dist.get_backend", return_value="gloo"),
            patch("lba.distributed.dist.is_gloo_available", return_value=True),
            patch("lba.distributed.dist.new_group", return_value=metadata_group) as new_group,
            patch("lba.distributed.dist.get_rank", return_value=0),
            patch("lba.distributed.dist.get_world_size", return_value=1),
        ):
            coordinator.prepare_for_background_iteration()

        self.assertTrue(coordinator.use_isolated_metadata_group)
        self.assertIs(coordinator._metadata_group, metadata_group)
        new_group.assert_called_once_with(backend="gloo")

    def test_splits_local_plans_to_match_distributed_step_count(self) -> None:
        records = (
            SampleRecord("a", 1, 0),
            SampleRecord("b", 1, 1),
            SampleRecord("c", 1, 2),
            SampleRecord("d", 1, 3),
        )
        plan = BatchPlan(
            records=records,
            raw_length_sum=4,
            padded_length=4,
            padding_length=0,
            padding_ratio=0.0,
            reason=PlanReason.PLANNED,
        )

        split_plans = split_plans_to_count([plan], 4)

        self.assertEqual(
            [len(split_plan.records) for split_plan in split_plans],
            [1, 1, 1, 1],
        )
        self.assertEqual(
            [
                record.sample
                for split_plan in split_plans
                for record in split_plan.records
            ],
            ["a", "b", "c", "d"],
        )

    def test_drops_unsplittable_tail_by_default(self) -> None:
        flush_planner = DistributedFlushPlanner(
            config=LBAConfig(drop_last_flush=True),
            logger=logging.getLogger("lba.test.drop_tail"),
            event_writer=None,
        )
        records = (
            SampleRecord("a", 1, 0),
            SampleRecord("b", 1, 1),
            SampleRecord("c", 1, 2),
        )
        plans = [
            BatchPlan(
                records=(record,),
                raw_length_sum=record.length,
                padded_length=record.length,
                padding_length=0,
                padding_ratio=0.0,
                reason=PlanReason.PLANNED,
            )
            for record in records
        ]

        with self.assertWarnsRegex(UserWarning, "dropped 1 final flush"):
            kept_plans = flush_planner.drop_tail(plans, world_size=2)

        self.assertEqual(
            [
                record.sample
                for plan in kept_plans
                for record in plan.records
            ],
            ["a", "b"],
        )

    def test_rejects_unsplittable_tail_when_drop_last_flush_is_false(self) -> None:
        flush_planner = DistributedFlushPlanner(
            config=LBAConfig(drop_last_flush=False),
            logger=logging.getLogger("lba.test.keep_tail"),
            event_writer=None,
        )
        record = SampleRecord("a", 1, 0)
        plan = BatchPlan(
            records=(record,),
            raw_length_sum=1,
            padded_length=1,
            padding_length=0,
            padding_ratio=0.0,
            reason=PlanReason.PLANNED,
        )

        with self.assertRaisesRegex(RuntimeError, "could not create enough"):
            flush_planner.drop_tail([plan], world_size=2)

    @unittest.skipUnless(
        dist.is_available()
        and dist.is_gloo_available()
        and _update_dataloader is not None,
        "torch.distributed gloo or Lightning is unavailable",
    )
    def test_two_rank_ddp_smoke_matches_steps(self) -> None:
        for case in (
            "map",
            "throughput",
            "cost",
            "global_cost",
            "global_cost_partial",
            "global_cost_flush",
            "global_cost_limit",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmpdir:
                world_size = 2
                init_file = str(Path(tmpdir) / "dist-init")
                mp.start_processes(
                    run_ddp_smoke_worker,
                    args=(world_size, init_file, tmpdir, case),
                    nprocs=world_size,
                    join=True,
                    start_method="spawn",
                )
                results = [
                    json.loads((Path(tmpdir) / f"{case}-rank{rank}.json").read_text())
                    for rank in range(world_size)
                ]

            self.assertEqual(len({result["steps"] for result in results}), 1)
            if case == "throughput":
                self.assertEqual({result["steps"] for result in results}, {3})
            if case not in (
                "throughput",
                "cost",
                "global_cost",
                "global_cost_partial",
                "global_cost_flush",
                "global_cost_limit",
            ):
                self.assertEqual({result["steps"] for result in results}, {4})
            if case in ("cost", "global_cost"):
                self.assertEqual({result["steps"] for result in results}, {2})
            if case == "global_cost_partial":
                self.assertEqual({result["steps"] for result in results}, {4})
            if case == "global_cost_flush":
                self.assertEqual({result["steps"] for result in results}, {3})
            if case == "global_cost_limit":
                self.assertEqual({result["steps"] for result in results}, {1})
            self.assertTrue(all(result["no_ready_calls"] == 0 for result in results))
            self.assertEqual(
                sum(sum(result["batch_sizes"]) for result in results),
                16
                if case == "global_cost_partial"
                else 4
                if case == "global_cost_limit"
                else 8,
            )
            budget = 20_000 if case in (
                "cost",
                "global_cost",
                "global_cost_partial",
                "global_cost_limit",
            ) else (
                15 if case == "throughput" else 100
            )
            self.assertTrue(
                all(
                    estimated_cost <= budget
                    for result in results
                    for estimated_cost in result["estimated_costs"]
                )
            )
            self.assertTrue(all(result["spill_dir_exists"] for result in results))
            self.assertTrue(
                all(result["source_uses_injected_sampler"] for result in results)
            )
            if case in (
                "map",
                "cost",
                "global_cost",
                "global_cost_partial",
                "global_cost_flush",
                "global_cost_limit",
                "throughput",
            ):
                rank_indices = [
                    set(result["sample_indices"])
                    for result in results
                ]
                self.assertFalse(rank_indices[0] & rank_indices[1])
                expected_size = (
                    len(PartialLengthDataset())
                    if case == "global_cost_partial"
                    else len(ThroughputLengthDataset())
                    if case == "throughput"
                    else len(LengthDataset())
                )
                if case == "global_cost_limit":
                    self.assertEqual(rank_indices[0] | rank_indices[1], {0, 2, 4, 6})
                else:
                    self.assertEqual(
                        rank_indices[0] | rank_indices[1],
                        set(range(expected_size)),
                    )
            if case == "global_cost":
                self.assertEqual(
                    results[0]["estimated_costs"],
                    results[1]["estimated_costs"],
                )
                self.assertEqual(results[0]["estimated_costs"], [20_000, 2])
                self.assertTrue(
                    all(result["distributed_cost_blocks"] == 1 for result in results)
                )
                self.assertEqual(
                    [
                        result["distributed_cost_block_events"][0]["remote_records"]
                        for result in results
                    ],
                    [2, 2],
                )
            if case == "global_cost_partial":
                self.assertEqual(
                    results[0]["estimated_costs"],
                    [20_000, 2, 2, 2],
                )
                self.assertEqual(
                    results[1]["estimated_costs"],
                    [20_000, 20_000, 2, 20_000],
                )
                self.assertTrue(
                    all(result["distributed_cost_blocks"] == 2 for result in results)
                )
            if case == "global_cost_flush":
                self.assertEqual(
                    results[0]["estimated_costs"],
                    results[1]["estimated_costs"],
                )
                self.assertEqual(results[0]["estimated_costs"], [100, 2, 100])
                self.assertTrue(
                    all(result["distributed_cost_blocks"] == 1 for result in results)
                )
            if case == "global_cost_limit":
                self.assertEqual(
                    results[0]["estimated_costs"],
                    results[1]["estimated_costs"],
                )
                self.assertEqual(results[0]["estimated_costs"], [20_000])
                self.assertTrue(
                    all(result["distributed_cost_blocks"] == 1 for result in results)
                )

    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(),
        "torch.distributed gloo is unavailable",
    )
    def test_mismatched_cost_window_fails_on_every_rank(self) -> None:
        for mismatch, error_text in (
            ("distributed", "identical distributed_cost_window_batches"),
            ("local", "identical cost_window_batches"),
            ("max_batches", "identical max_batches"),
        ):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as tmpdir:
                world_size = 2
                init_file = str(Path(tmpdir) / "dist-init")
                mp.start_processes(
                    run_cost_window_mismatch_worker,
                    args=(world_size, init_file, tmpdir, mismatch),
                    nprocs=world_size,
                    join=True,
                    start_method="spawn",
                )
                results = [
                    json.loads(
                        (
                            Path(tmpdir)
                            / f"window-mismatch-{mismatch}-rank{rank}.json"
                        ).read_text()
                    )
                    for rank in range(world_size)
                ]

            self.assertTrue(
                all(
                    result["error"] is not None
                    and error_text in result["error"]
                    for result in results
                )
            )


if __name__ == "__main__":
    unittest.main()
