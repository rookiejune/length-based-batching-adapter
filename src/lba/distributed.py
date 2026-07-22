"""Distributed helpers for length-based batching."""

from __future__ import annotations

import logging
import operator
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from ._cost import BatchCost
from ._api_types import EventWriter, LengthFn
from ._distributed_cost import (
    PlanMetadata,
    cost_window_stats,
    match_cost_block,
    plan_metadata,
)
from ._distributed_flush import (
    DistributedFlushPlanner,
    largest_splittable_plan_index,
    make_batch_plan,
    record_count,
    split_plans_to_count,
)
from .config import LBAConfig
from .adaptive import CostWindowStats, adaptive_config_fields
from ._records import BatchPlan, SampleRecord


class DistributedBatchCoordinator:
    """Coordinate DDP-only synchronization and final flush planning."""

    def __init__(
        self,
        dataloader: Optional[DataLoader],
        config: LBAConfig,
        logger: Optional[logging.Logger],
        event_writer: Optional[EventWriter] = None,
        len_fn: Optional[LengthFn] = None,
        *,
        use_isolated_metadata_group: bool = False,
    ) -> None:
        self.dataloader = dataloader
        self.config = config
        self.logger = logger
        self.event_writer = event_writer
        self.len_fn = len_fn
        self.flush_planner = DistributedFlushPlanner(config, logger, event_writer)
        self.use_isolated_metadata_group = use_isolated_metadata_group
        self._metadata_group: Optional[dist.ProcessGroup] = None

    @staticmethod
    def is_initialized() -> bool:
        return dist.is_available() and dist.is_initialized()

    def validate_source_batch_presence(
        self, local_has_batch: int, *, context: str
    ) -> None:
        present_count = self._distributed_int_reduce(
            local_has_batch,
            dist.ReduceOp.SUM,
        )
        if present_count not in (0, self._world_size()):
            raise RuntimeError(
                "LBA distributed mode requires every rank to receive the same "
                f"number of source DataLoader batches; mismatch during {context}."
            )

    def sync_max_padded_length(self, local_value: int) -> int:
        max_value = self._distributed_int_reduce(local_value, dist.ReduceOp.MAX)
        if (
            self.config.max_padded_length is not None
            and self._distributed_int_reduce(local_value, dist.ReduceOp.MIN) != max_value
        ):
            raise RuntimeError(
                "LBA distributed mode requires identical explicit "
                "max_padded_length on every rank."
            )
        if self.config.max_padded_length is None and local_value != max_value:
            rank = self._rank()
            world_size = self._world_size()
            if self.logger is not None:
                self.logger.info(
                    "lba distributed: rank=%s/%s using shared max_padded_length=%s "
                    "local_value=%s",
                    rank,
                    world_size,
                    max_value,
                    local_value,
                )
            self._write_event(
                "distributed_max_padded_length",
                {
                    "rank": rank,
                    "world_size": world_size,
                    "local_value": local_value,
                    "shared_value": max_value,
                },
            )
        return max_value

    def all_ranks_reached_batch_limit(self, local_done: bool) -> bool:
        done_count = self._distributed_int_reduce(
            int(local_done),
            dist.ReduceOp.SUM,
        )
        return done_count == self._world_size()

    def sync_max_batch_cost(self, local_value: int) -> int:
        max_value = self._distributed_int_reduce(local_value, dist.ReduceOp.MAX)
        min_value = self._distributed_int_reduce(local_value, dist.ReduceOp.MIN)
        if min_value != max_value:
            raise RuntimeError(
                "LBA distributed mode requires identical max_batch_cost "
                "on every rank."
            )
        return max_value

    def validate_iteration_configuration(
        self,
        *,
        cost_window_batches: int,
        distributed_cost_window_batches: Optional[int],
        max_batches: Optional[int],
        adaptive_distributed_cost_window_enabled: bool = False,
    ) -> None:
        values = (
            cost_window_batches,
            (
                distributed_cost_window_batches
                if distributed_cost_window_batches is not None
                else 0
            ),
            int(adaptive_distributed_cost_window_enabled),
            max_batches if max_batches is not None else -1,
        )
        max_values = self._distributed_ints_reduce(values, dist.ReduceOp.MAX)
        min_values = self._distributed_ints_reduce(values, dist.ReduceOp.MIN)
        if min_values[0] != max_values[0]:
            raise RuntimeError(
                "LBA distributed mode requires identical cost_window_batches "
                "on every rank."
            )
        if min_values[1] != max_values[1]:
            raise RuntimeError(
                "LBA distributed mode requires identical "
                "distributed_cost_window_batches on every rank."
            )
        if min_values[2] != max_values[2]:
            raise RuntimeError(
                "LBA distributed mode requires identical "
                "adaptive distributed cost-window state on every rank."
            )
        if min_values[3] != max_values[3]:
            raise RuntimeError(
                "LBA distributed mode requires identical max_batches on every rank."
            )

    def validate_adaptive_configuration(self) -> None:
        local_fields = adaptive_config_fields(self.config.adaptive)
        gathered_fields: list[Optional[dict[str, object]]] = [
            None for _ in range(self._world_size())
        ]
        dist.all_gather_object(
            gathered_fields,
            local_fields,
            group=self._metadata_process_group(),
        )
        if any(fields != local_fields for fields in gathered_fields):
            raise RuntimeError(
                "LBA distributed mode requires identical "
                "adaptive configuration on every rank."
            )

    def prepare_for_background_iteration(self) -> None:
        """Prepare metadata collectives for producer-thread iteration."""

        if not self.is_initialized():
            return
        self.use_isolated_metadata_group = True
        self._metadata_process_group()

    def flush_plans(
        self, local_records: list[SampleRecord], *, max_batch_cost: int
    ) -> list[BatchPlan]:
        if not self._all_ranks_have_record_indices(local_records):
            raise RuntimeError(
                "LBA distributed flush requires map-style sample indices."
            )
        plans = self._index_flush_plans(local_records, max_batch_cost)
        self._write_flush_event("index_metadata", local_records, plans)
        return plans

    def match_cost_plans(
        self,
        local_plans: list[BatchPlan],
        *,
        step_offset: int,
    ) -> list[BatchPlan]:
        assigned_plans, _ = self.match_cost_plans_with_stats(
            local_plans,
            step_offset=step_offset,
        )
        return assigned_plans

    def match_cost_plans_with_stats(
        self,
        local_plans: list[BatchPlan],
        *,
        step_offset: int,
    ) -> tuple[list[BatchPlan], CostWindowStats]:
        local_metadata = [plan_metadata(plan) for plan in local_plans]
        gathered_metadata: list[list[PlanMetadata]] = [
            [] for _ in range(self._world_size())
        ]
        dist.all_gather_object(
            gathered_metadata,
            local_metadata,
            group=self._metadata_process_group(),
        )

        rank = self._rank()
        assigned_by_rank = match_cost_block(
            gathered_metadata,
            step_offset=step_offset,
        )
        stats = cost_window_stats(gathered_metadata, assigned_by_rank)
        assigned_refs = assigned_by_rank[rank]
        assigned_plans: list[BatchPlan] = []
        remote_plan_count = 0
        remote_record_count = 0
        for ref in assigned_refs:
            if ref.source_rank == rank:
                local_plan = local_plans[ref.source_position]
                if plan_metadata(local_plan) != ref.metadata:
                    raise RuntimeError(
                        "LBA distributed cost matching received inconsistent "
                        "local plan metadata."
                    )
                assigned_plans.append(local_plan)
                continue

            remote_plan_count += 1
            remote_record_count += len(ref.metadata.records)
            assigned_plans.append(self._materialize_cost_plan(ref.metadata))

        self._write_event(
            "distributed_cost_block",
            {
                "rank": rank,
                "world_size": self._world_size(),
                "step_offset": step_offset,
                "block_size": len(local_plans),
                "remote_batches": remote_plan_count,
                "remote_records": remote_record_count,
                "source_mean_step_spread": stats.source_mean_step_spread,
                "matched_mean_step_spread": stats.matched_mean_step_spread,
                "source_spread_ratio": stats.source_spread_ratio,
                "improvement_ratio": stats.improvement_ratio,
            },
        )
        return assigned_plans, stats

    def spill_dir_for_rank(self) -> Optional[Union[Path, str]]:
        if self.config.spill_dir is None:
            return None
        if not self.is_initialized():
            return self.config.spill_dir
        return Path(self.config.spill_dir) / f"rank-{dist.get_rank():05d}"

    def _index_flush_plans(
        self, local_records: list[SampleRecord], max_batch_cost: int
    ) -> list[BatchPlan]:
        batch_cost = BatchCost(max_batch_cost, self.config.cost_fn)
        local_metadata = [
            (self._require_record_index(record), record.length)
            for record in local_records
        ]
        gathered_metadata: list[list[tuple[int, int]]] = [
            [] for _ in range(self._world_size())
        ]
        dist.all_gather_object(
            gathered_metadata,
            local_metadata,
            group=self._metadata_process_group(),
        )

        global_records = self._records_from_index_metadata(gathered_metadata)
        assigned_plans = self.flush_planner.assigned_plans(
            global_records,
            max_batch_cost,
            rank=self._rank(),
            world_size=self._world_size(),
        )
        return [
            self._materialize_index_plan(plan, batch_cost)
            for plan in assigned_plans
        ]

    @staticmethod
    def _records_have_indices(records: Iterable[SampleRecord]) -> bool:
        return all(record.index is not None for record in records)

    def _all_ranks_have_record_indices(
        self, local_records: Iterable[SampleRecord]
    ) -> bool:
        if self.dataloader is None:
            return False
        local_has_indices = int(self._records_have_indices(local_records))
        min_has_indices = self._distributed_int_reduce(
            local_has_indices,
            dist.ReduceOp.MIN,
        )
        return bool(min_has_indices)

    @staticmethod
    def _require_record_index(record: SampleRecord) -> int:
        if record.index is None:
            raise RuntimeError("LBA expected a sample index for distributed flush.")
        return record.index

    @staticmethod
    def _records_from_index_metadata(
        gathered_metadata: Iterable[Iterable[tuple[int, int]]],
    ) -> list[SampleRecord]:
        global_records: list[SampleRecord] = []
        for rank_metadata in gathered_metadata:
            for sample_index, length in rank_metadata:
                global_records.append(
                    SampleRecord(
                        sample=sample_index,
                        length=length,
                        arrival_id=len(global_records),
                        index=sample_index,
                    )
                )
        return global_records

    def _materialize_index_plan(
        self,
        plan: BatchPlan,
        batch_cost: BatchCost,
    ) -> BatchPlan:
        if self.dataloader is None:
            raise RuntimeError("LBA cannot materialize indexed samples without a dataloader.")
        records = [
            self._materialize_record(
                self._require_record_index(record),
                length=record.length,
                arrival_id=record.arrival_id,
            )
            for record in plan.records
        ]
        return make_batch_plan(
            records,
            plan.reason,
            batch_cost=batch_cost,
        )

    def _materialize_cost_plan(self, metadata: PlanMetadata) -> BatchPlan:
        if self.dataloader is None:
            raise RuntimeError(
                "LBA cannot materialize distributed cost plans without a dataloader."
            )

        records: list[SampleRecord] = []
        for record_metadata in metadata.records:
            if record_metadata.index is None:
                raise RuntimeError(
                    "LBA distributed cost matching requires map-style sample indices."
                )
            records.append(
                self._materialize_record(
                    record_metadata.index,
                    length=record_metadata.length,
                    arrival_id=record_metadata.arrival_id,
                )
            )

        raw_length_sum = sum(record.length for record in records)
        padded_length = max(record.length for record in records) * len(records)
        padding_length = padded_length - raw_length_sum
        return BatchPlan(
            records=tuple(records),
            raw_length_sum=raw_length_sum,
            padded_length=padded_length,
            padding_length=padding_length,
            padding_ratio=(
                padding_length / padded_length if padded_length else 0.0
            ),
            reason=metadata.reason,
            estimated_cost=metadata.estimated_cost,
        )

    def _materialize_record(
        self,
        index: int,
        *,
        length: int,
        arrival_id: int,
    ) -> SampleRecord:
        if self.dataloader is None:
            raise RuntimeError(
                "LBA cannot materialize indexed samples without a dataloader."
            )
        return SampleRecord(
            sample=index,
            length=length,
            arrival_id=arrival_id,
            index=index,
        )

    def _distributed_int_reduce(self, value: int, op: dist.ReduceOp) -> int:
        return self._distributed_ints_reduce((value,), op)[0]

    def _distributed_ints_reduce(
        self,
        values: Sequence[int],
        op: dist.ReduceOp,
    ) -> tuple[int, ...]:
        group = self._metadata_process_group()
        device = self._distributed_tensor_device(group)
        tensor = torch.tensor(values, dtype=torch.long, device=device)
        dist.all_reduce(tensor, op=op, group=group)
        return tuple(int(value) for value in tensor.tolist())

    def _write_event(self, event: str, fields: dict[str, object]) -> None:
        if self.event_writer is None:
            return
        self.event_writer.write(event, fields)

    def _write_flush_event(
        self,
        mode: str,
        local_records: list[SampleRecord],
        plans: list[BatchPlan],
    ) -> None:
        if not self.is_initialized():
            return
        self._write_event(
            "distributed_flush",
            {
                "mode": mode,
                "rank": self._rank(),
                "world_size": self._world_size(),
                "local_records": len(local_records),
                "assigned_batches": len(plans),
                "assigned_records": record_count(plans),
            },
        )

    def _metadata_process_group(self) -> Optional[dist.ProcessGroup]:
        if not self.is_initialized():
            return None
        default_backend = str(dist.get_backend()).lower()
        if "nccl" not in default_backend and not self.use_isolated_metadata_group:
            return None
        if not dist.is_gloo_available():
            raise RuntimeError(
                "LBA distributed metadata synchronization requires the gloo "
                "backend when the default process group uses NCCL or when "
                "distributed prefetch runs metadata collectives in a producer "
                "thread."
            )
        if self._metadata_group is None:
            self._metadata_group = dist.new_group(backend="gloo")
            if self.logger is not None:
                self.logger.info(
                    "lba distributed: using gloo metadata process group for "
                    "CPU metadata collectives alongside %s default process group",
                    default_backend,
                )
            self._write_event(
                "distributed_metadata_group",
                {
                    "default_backend": default_backend,
                    "metadata_backend": "gloo",
                    "rank": self._rank(),
                    "world_size": self._world_size(),
                    "background_iteration": self.use_isolated_metadata_group,
                },
            )
        return self._metadata_group

    def _rank(self) -> int:
        if self._metadata_group is None:
            return dist.get_rank()
        return dist.get_rank(self._metadata_group)

    def _world_size(self) -> int:
        if self._metadata_group is None:
            return dist.get_world_size()
        return dist.get_world_size(self._metadata_group)

    @staticmethod
    def _distributed_tensor_device(
        group: Optional[dist.ProcessGroup],
    ) -> torch.device:
        if group is not None:
            return torch.device("cpu")
        backend = str(dist.get_backend()).lower()
        if "nccl" not in backend:
            return torch.device("cpu")
        if not torch.cuda.is_available():
            raise RuntimeError("LBA distributed mode with NCCL requires CUDA.")
        return torch.device("cuda", torch.cuda.current_device())


__all__ = [
    "DistributedBatchCoordinator",
    "DistributedFlushPlanner",
    "largest_splittable_plan_index",
    "make_batch_plan",
    "record_count",
    "split_plans_to_count",
]
