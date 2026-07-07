"""Final-flush planning helpers for distributed LBA iteration."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from typing import Any

from .config import LBAConfig
from .planner import BatchPlanner
from ._records import BatchPlan, PlanReason, SampleRecord


class DistributedFlushPlanner:
    """Plan and assign final flush batches after metadata gathering."""

    def __init__(
        self,
        config: LBAConfig,
        logger: logging.Logger | None,
        event_writer: Any | None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.event_writer = event_writer

    def assigned_plans(
        self,
        records: list[SampleRecord],
        max_padded_length: int,
        *,
        rank: int,
        world_size: int,
    ) -> list[BatchPlan]:
        plans = self._plan_records(records, max_padded_length)
        target_count = self._round_up_to_world_size(len(plans), world_size)
        if target_count > record_count(plans):
            plans = self.drop_tail(plans, world_size)
            target_count = len(plans)
        plans = split_plans_to_count(plans, target_count)
        return [plan for index, plan in enumerate(plans) if index % world_size == rank]

    def _plan_records(
        self, records: list[SampleRecord], max_padded_length: int
    ) -> list[BatchPlan]:
        if not records:
            return []
        planner = BatchPlanner(
            max_padded_length=max_padded_length,
            max_cache_samples=max(len(records), 1),
            max_padding_ratio=self.config.max_padding_ratio,
            max_candidate_windows=self.config.candidate_window_limit,
            limited_search_fallback_after=(
                self.config.limited_search_fallback_after_limit
            ),
            limited_search_fallback_pool_size=(
                self.config.limited_search_fallback_pool_limit
            ),
            spill_dir=None,
            logger=self.logger,
            event_writer=self.event_writer,
        )
        try:
            planner.add_records(records, allow_spill=False)
            return list(planner.flush())
        finally:
            planner.close()

    def drop_tail(self, plans: list[BatchPlan], world_size: int) -> list[BatchPlan]:
        keep_count = len(plans) - len(plans) % world_size
        if keep_count == len(plans):
            return plans
        if keep_count == 0:
            dropped_record_count = record_count(plans)
            self._handle_dropped_records(dropped_record_count, world_size)
            return []

        kept_plans = plans[:keep_count]
        dropped_record_count = record_count(plans[keep_count:])
        self._handle_dropped_records(dropped_record_count, world_size)
        return kept_plans

    def _handle_dropped_records(
        self, dropped_record_count: int, world_size: int
    ) -> None:
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
        if self.logger is not None:
            self.logger.warning(
                "%s Set drop_last_flush=False to fail instead.",
                message,
            )
        self._write_event(
            "distributed_drop_last_flush",
            {
                "dropped_records": dropped_record_count,
                "world_size": world_size,
                "drop_last_flush": self.config.drop_last_flush,
            },
        )

    def _write_event(self, event: str, fields: dict[str, object]) -> None:
        if self.event_writer is None:
            return
        self.event_writer.write(event, fields)

    @staticmethod
    def _round_up_to_world_size(value: int, world_size: int) -> int:
        remainder = value % world_size
        if remainder == 0:
            return value
        return value + world_size - remainder


def record_count(plans: Iterable[BatchPlan]) -> int:
    return sum(len(plan.records) for plan in plans)


def split_plans_to_count(plans: list[BatchPlan], target_count: int) -> list[BatchPlan]:
    if target_count < len(plans):
        raise RuntimeError(
            "LBA distributed batch synchronization received an invalid target."
        )
    total_record_count = record_count(plans)
    if target_count > total_record_count:
        raise RuntimeError(
            "LBA distributed flush could not create enough non-empty batches "
            "for every rank."
        )

    split_plans = list(plans)
    while len(split_plans) < target_count:
        split_index = largest_splittable_plan_index(split_plans)
        if split_index is None:
            raise RuntimeError(
                "LBA distributed mode could not split local batches enough to "
                "match the number of steps on every rank. Make sure every rank "
                "uses equally sized source DataLoader batches."
            )

        plan = split_plans.pop(split_index)
        midpoint = len(plan.records) // 2
        left_plan = make_batch_plan(plan.records[:midpoint], plan.reason)
        right_plan = make_batch_plan(plan.records[midpoint:], plan.reason)
        split_plans[split_index:split_index] = [left_plan, right_plan]

    return split_plans


def largest_splittable_plan_index(plans: list[BatchPlan]) -> int | None:
    best_index: int | None = None
    best_size = 1
    for index, plan in enumerate(plans):
        plan_size = len(plan.records)
        if plan_size > best_size:
            best_size = plan_size
            best_index = index
    return best_index


def make_batch_plan(records: Iterable[SampleRecord], reason: PlanReason) -> BatchPlan:
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
