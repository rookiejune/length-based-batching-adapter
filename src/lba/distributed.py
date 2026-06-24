"""Distributed helpers for length-based batching."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .config import LBAConfig
from .planner import BatchPlanner
from .types import BatchPlan, SampleRecord


class DistributedBatchCoordinator:
    """Coordinate DDP-only synchronization and final flush planning."""

    def __init__(
        self,
        dataloader: DataLoader,
        config: LBAConfig,
        logger: logging.Logger,
    ) -> None:
        self.dataloader = dataloader
        self.config = config
        self.logger = logger

    @staticmethod
    def is_initialized() -> bool:
        return dist.is_available() and dist.is_initialized()

    def validate_source_batch_presence(
        self, local_has_batch: int, *, context: str
    ) -> None:
        min_has_batch, max_has_batch = self._distributed_int_min_max(local_has_batch)
        if min_has_batch != max_has_batch:
            raise RuntimeError(
                "LBA distributed mode requires every rank to receive the same "
                f"number of source DataLoader batches; mismatch during {context}."
            )

    def sync_max_padded_length(self, local_value: int) -> int:
        min_value, max_value = self._distributed_int_min_max(local_value)
        if self.config.max_padded_length is not None and min_value != max_value:
            raise RuntimeError(
                "LBA distributed mode requires identical explicit "
                "max_padded_length on every rank."
            )
        if self.config.max_padded_length is None and local_value != max_value:
            self.logger.info(
                "using distributed max_padded_length=%s from local_value=%s",
                max_value,
                local_value,
            )
        return max_value

    def flush_plans(
        self, local_records: list[SampleRecord], *, max_padded_length: int
    ) -> list[BatchPlan]:
        if self._all_ranks_have_record_indices(local_records):
            return self._index_flush_plans(local_records, max_padded_length)
        return self._object_flush_plans(local_records, max_padded_length)

    def spill_dir_for_rank(self) -> Path | str | None:
        if self.config.spill_dir is None:
            return None
        if not self.is_initialized():
            return self.config.spill_dir
        return Path(self.config.spill_dir) / f"rank-{dist.get_rank():05d}"

    def _index_flush_plans(
        self, local_records: list[SampleRecord], max_padded_length: int
    ) -> list[BatchPlan]:
        local_metadata = [
            (self._require_record_index(record), record.length)
            for record in local_records
        ]
        gathered_metadata: list[list[tuple[int, int]]] = [
            [] for _ in range(dist.get_world_size())
        ]
        dist.all_gather_object(gathered_metadata, local_metadata)

        global_records = self._records_from_index_metadata(gathered_metadata)
        assigned_plans = self._assigned_global_flush_plans(
            global_records,
            max_padded_length,
        )
        return [self._materialize_index_plan(plan) for plan in assigned_plans]

    def _object_flush_plans(
        self, local_records: list[SampleRecord], max_padded_length: int
    ) -> list[BatchPlan]:
        gathered_records: list[list[SampleRecord]] = [
            [] for _ in range(dist.get_world_size())
        ]
        dist.all_gather_object(gathered_records, local_records)

        global_records = self._reassign_arrival_ids(gathered_records)
        return self._assigned_global_flush_plans(global_records, max_padded_length)

    def _assigned_global_flush_plans(
        self, global_records: list[SampleRecord], max_padded_length: int
    ) -> list[BatchPlan]:
        global_plans = self._plan_global_flush_records(
            global_records,
            max_padded_length,
        )
        target_count = self._round_up_to_world_size(len(global_plans))
        if target_count > self._record_count(global_plans):
            global_plans = self._drop_last_flush_plans(global_plans)
            target_count = len(global_plans)
        global_plans = self.split_plans_to_count(global_plans, target_count)

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return [
            plan
            for index, plan in enumerate(global_plans)
            if index % world_size == rank
        ]

    @staticmethod
    def _records_have_indices(records: Iterable[SampleRecord]) -> bool:
        return all(record.index is not None for record in records)

    def _all_ranks_have_record_indices(
        self, local_records: Iterable[SampleRecord]
    ) -> bool:
        local_has_indices = int(self._records_have_indices(local_records))
        min_has_indices, _ = self._distributed_int_min_max(local_has_indices)
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

    def _materialize_index_plan(self, plan: BatchPlan) -> BatchPlan:
        records = [
            SampleRecord(
                sample=self.dataloader.dataset[self._require_record_index(record)],
                length=record.length,
                arrival_id=record.arrival_id,
                index=record.index,
            )
            for record in plan.records
        ]
        return self.make_batch_plan(records, plan.reason)

    @staticmethod
    def _reassign_arrival_ids(
        gathered_records: Iterable[Iterable[SampleRecord]],
    ) -> list[SampleRecord]:
        global_records: list[SampleRecord] = []
        for rank_records in gathered_records:
            for record in rank_records:
                global_records.append(
                    SampleRecord(
                        sample=record.sample,
                        length=record.length,
                        arrival_id=len(global_records),
                        index=record.index,
                    )
                )
        return global_records

    def _plan_global_flush_records(
        self, records: list[SampleRecord], max_padded_length: int
    ) -> list[BatchPlan]:
        if not records:
            return []
        planner = BatchPlanner(
            max_padded_length=max_padded_length,
            max_cache_samples=max(len(records), 1),
            max_padding_ratio=self.config.max_padding_ratio,
            spill_dir=None,
            logger=self.logger,
        )
        try:
            planner.add_records(records, allow_spill=False)
            return list(planner.flush())
        finally:
            planner.close()

    @staticmethod
    def _round_up_to_world_size(value: int) -> int:
        world_size = dist.get_world_size()
        remainder = value % world_size
        if remainder == 0:
            return value
        return value + world_size - remainder

    def _drop_last_flush_plans(self, plans: list[BatchPlan]) -> list[BatchPlan]:
        world_size = dist.get_world_size()
        keep_count = len(plans) - len(plans) % world_size
        if keep_count == len(plans):
            return plans
        if keep_count == 0:
            dropped_record_count = self._record_count(plans)
            self._handle_dropped_flush_records(dropped_record_count)
            return []

        kept_plans = plans[:keep_count]
        dropped_record_count = self._record_count(plans[keep_count:])
        self._handle_dropped_flush_records(dropped_record_count)
        return kept_plans

    def _handle_dropped_flush_records(self, dropped_record_count: int) -> None:
        if not self.config.drop_last_flush:
            raise RuntimeError(
                "LBA distributed flush could not create enough non-empty batches "
                "for every rank."
            )

        message = (
            "LBA distributed mode dropped "
            f"{dropped_record_count} final flush sample(s) because they could not "
            "form a non-empty batch on every rank."
        )
        warnings.warn(message, stacklevel=3)
        self.logger.warning(message)

    @staticmethod
    def _record_count(plans: Iterable[BatchPlan]) -> int:
        return sum(len(plan.records) for plan in plans)

    @staticmethod
    def split_plans_to_count(
        plans: list[BatchPlan], target_count: int
    ) -> list[BatchPlan]:
        if target_count < len(plans):
            raise RuntimeError(
                "LBA distributed batch synchronization received an invalid target."
            )
        total_record_count = sum(len(plan.records) for plan in plans)
        if target_count > total_record_count:
            raise RuntimeError(
                "LBA distributed flush could not create enough non-empty batches "
                "for every rank."
            )

        split_plans = list(plans)
        while len(split_plans) < target_count:
            split_index = DistributedBatchCoordinator._largest_splittable_plan_index(
                split_plans
            )
            if split_index is None:
                raise RuntimeError(
                    "LBA distributed mode could not split local batches enough to "
                    "match the number of steps on every rank. Make sure every rank "
                    "uses equally sized source DataLoader batches."
                )

            plan = split_plans.pop(split_index)
            midpoint = len(plan.records) // 2
            left_plan = DistributedBatchCoordinator.make_batch_plan(
                plan.records[:midpoint],
                plan.reason,
            )
            right_plan = DistributedBatchCoordinator.make_batch_plan(
                plan.records[midpoint:],
                plan.reason,
            )
            split_plans[split_index:split_index] = [left_plan, right_plan]

        return split_plans

    @staticmethod
    def _largest_splittable_plan_index(plans: list[BatchPlan]) -> int | None:
        best_index: int | None = None
        best_size = 1
        for index, plan in enumerate(plans):
            plan_size = len(plan.records)
            if plan_size > best_size:
                best_size = plan_size
                best_index = index
        return best_index

    @staticmethod
    def make_batch_plan(records: Iterable[SampleRecord], reason: str) -> BatchPlan:
        ordered_records = tuple(sorted(records, key=lambda record: record.arrival_id))
        raw_length_sum = sum(record.length for record in ordered_records)
        padded_length = max(record.length for record in ordered_records) * len(
            ordered_records
        )
        padding_length = padded_length - raw_length_sum
        padding_ratio = padding_length / padded_length if padded_length else 0.0
        return BatchPlan(
            records=ordered_records,
            raw_length_sum=raw_length_sum,
            padded_length=padded_length,
            padding_length=padding_length,
            padding_ratio=padding_ratio,
            reason=reason,
        )

    @staticmethod
    def _distributed_int_min_max(value: int) -> tuple[int, int]:
        device = DistributedBatchCoordinator._distributed_tensor_device()
        min_tensor = torch.tensor(value, dtype=torch.long, device=device)
        max_tensor = torch.tensor(value, dtype=torch.long, device=device)
        dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
        return int(min_tensor.item()), int(max_tensor.item())

    @staticmethod
    def _distributed_tensor_device() -> torch.device:
        backend = str(dist.get_backend()).lower()
        if "nccl" not in backend:
            return torch.device("cpu")
        if not torch.cuda.is_available():
            raise RuntimeError("LBA distributed mode with NCCL requires CUDA.")
        return torch.device("cuda", torch.cuda.current_device())
