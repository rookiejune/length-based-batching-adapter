import unittest
import json
import logging
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset

from lba import LBA
from lba.config import LBAConfig
from lba.distributed import (
    DistributedBatchCoordinator,
    DistributedFlushPlanner,
    split_plans_to_count,
)
from lba.types import BatchPlan, PlanReason, SampleRecord


def collate_lengths(samples: list[int]) -> torch.Tensor:
    return torch.tensor(samples, dtype=torch.float32).unsqueeze(1)


def sample_length(sample: int) -> int:
    return sample


class LengthDataset(Dataset[int]):
    def __init__(self) -> None:
        self.lengths = [100, 1, 100, 1, 100, 1, 100, 1]

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, index: int) -> int:
        return self.lengths[index]


class RankIterableDataset(IterableDataset[int]):
    def __init__(self) -> None:
        self.samples_by_rank = (
            [100, 100, 100, 100],
            [1, 1, 1, 1],
        )

    def __iter__(self):
        yield from self.samples_by_rank[dist.get_rank()]


def build_ddp_loader(case: str, rank: int, world_size: int) -> DataLoader:
    if case == "map":
        dataset = LengthDataset()
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        return DataLoader(
            dataset,
            batch_size=2,
            sampler=sampler,
            collate_fn=collate_lengths,
            num_workers=0,
        )
    if case == "iterable":
        return DataLoader(
            RankIterableDataset(),
            batch_size=2,
            collate_fn=collate_lengths,
            num_workers=0,
        )
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
    )
    try:
        output_path = Path(output_dir)
        base_loader = build_ddp_loader(case, rank, world_size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loader = LBA(
                base_loader,
                len_fn=sample_length,
                max_padded_length=100,
                max_padding_ratio=0.0,
                max_cache_samples=1,
                prefetch_batches=2,
                spill_dir=output_path / "shared-spill",
                log_dir=output_path / f"logs-{case}-rank{rank}",
            )

        model = DistributedDataParallel(nn.Linear(1, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        steps = 0
        batch_sizes: list[int] = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch).sum()
            loss.backward()
            optimizer.step()
            steps += 1
            batch_sizes.append(int(batch.shape[0]))

        dist.barrier()
        spill_dir = output_path / "shared-spill" / f"rank-{rank:05d}"
        result = {
            "rank": rank,
            "steps": steps,
            "batch_sizes": batch_sizes,
            "spill_dir_exists": spill_dir.exists(),
        }
        (output_path / f"{case}-rank{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


class DistributedCoordinatorTest(unittest.TestCase):
    def test_uses_index_flush_only_when_all_ranks_have_indices(self) -> None:
        coordinator = DistributedBatchCoordinator(
            dataloader=None,
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
        dist.is_available() and dist.is_gloo_available(),
        "torch.distributed gloo is unavailable",
    )
    def test_two_rank_ddp_smoke_matches_steps(self) -> None:
        for case in ("map", "iterable"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmpdir:
                world_size = 2
                init_file = str(Path(tmpdir) / "dist-init")
                mp.start_processes(
                    run_ddp_smoke_worker,
                    args=(world_size, init_file, tmpdir, case),
                    nprocs=world_size,
                    join=True,
                    start_method="fork",
                )
                results = [
                    json.loads((Path(tmpdir) / f"{case}-rank{rank}.json").read_text())
                    for rank in range(world_size)
                ]

            self.assertEqual({result["steps"] for result in results}, {4})
            self.assertTrue(all(result["spill_dir_exists"] for result in results))


if __name__ == "__main__":
    unittest.main()
