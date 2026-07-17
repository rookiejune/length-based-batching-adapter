"""Distributed helpers for length-based batching."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from ._distributed_flush import (
    DistributedFlushPlanner,
    largest_splittable_plan_index,
    make_batch_plan,
    record_count,
    split_plans_to_count,
)
from .config import LBAConfig
from ._api_types import EventWriter
from ._records import BatchPlan, SampleRecord


class DistributedBatchCoordinator:
    """Coordinate DDP-only synchronization and final flush planning."""

    def __init__(
        self,
        dataloader: Optional[DataLoader],
        config: LBAConfig,
        logger: Optional[logging.Logger],
        event_writer: Optional[EventWriter] = None,
    ) -> None:
        self.dataloader = dataloader
        self.config = config
        self.logger = logger
        self.event_writer = event_writer
        self.flush_planner = DistributedFlushPlanner(config, logger, event_writer)
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

    def flush_plans(
        self, local_records: list[SampleRecord], *, max_padded_length: int
    ) -> list[BatchPlan]:
        if self._all_ranks_have_record_indices(local_records):
            plans = self._index_flush_plans(local_records, max_padded_length)
            self._write_flush_event("index_metadata", local_records, plans)
            return plans
        plans = self._object_flush_plans(local_records, max_padded_length)
        self._write_flush_event("object_gather", local_records, plans)
        return plans

    def spill_dir_for_rank(self) -> Optional[Union[Path, str]]:
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
            max_padded_length,
            rank=self._rank(),
            world_size=self._world_size(),
        )
        return [self._materialize_index_plan(plan) for plan in assigned_plans]

    def _object_flush_plans(
        self, local_records: list[SampleRecord], max_padded_length: int
    ) -> list[BatchPlan]:
        gathered_records: list[list[SampleRecord]] = [
            [] for _ in range(self._world_size())
        ]
        dist.all_gather_object(
            gathered_records,
            local_records,
            group=self._metadata_process_group(),
        )

        global_records = self._reassign_arrival_ids(gathered_records)
        return self.flush_planner.assigned_plans(
            global_records,
            max_padded_length,
            rank=self._rank(),
            world_size=self._world_size(),
        )

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

    def _materialize_index_plan(self, plan: BatchPlan) -> BatchPlan:
        if self.dataloader is None:
            raise RuntimeError("LBA cannot materialize indexed samples without a dataloader.")
        records = [
            SampleRecord(
                sample=self.dataloader.dataset[self._require_record_index(record)],
                length=record.length,
                arrival_id=record.arrival_id,
                index=record.index,
            )
            for record in plan.records
        ]
        return make_batch_plan(records, plan.reason)

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

    def _distributed_int_reduce(self, value: int, op: dist.ReduceOp) -> int:
        group = self._metadata_process_group()
        device = self._distributed_tensor_device(group)
        tensor = torch.tensor(value, dtype=torch.long, device=device)
        dist.all_reduce(tensor, op=op, group=group)
        return int(tensor.item())

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
        if "nccl" not in str(dist.get_backend()).lower():
            return None
        if not dist.is_gloo_available():
            raise RuntimeError(
                "LBA distributed metadata synchronization requires the gloo "
                "backend when the default process group uses NCCL."
            )
        if self._metadata_group is None:
            self._metadata_group = dist.new_group(backend="gloo")
            if self.logger is not None:
                self.logger.info(
                    "lba distributed: using gloo metadata process group "
                    "alongside NCCL default process group"
                )
            self._write_event(
                "distributed_metadata_group",
                {
                    "default_backend": str(dist.get_backend()),
                    "metadata_backend": "gloo",
                    "rank": self._rank(),
                    "world_size": self._world_size(),
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
