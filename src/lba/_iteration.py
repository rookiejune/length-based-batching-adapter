"""One iteration of the length-based batching pipeline."""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterable, Iterator
from typing import Any, Optional

from ._api_types import CollateFn, EventWriter
from ._records import BatchPlan, LengthRecord, PlanReason, SampleRecord
from ._run_reporter import RunReporter
from .budget import BatchSizeSource, BudgetResolver
from .config import LBAConfig
from .distributed import DistributedBatchCoordinator
from .metrics import PaddingStats, PlannerStats
from .planner import BatchPlanner


class Iteration:
    """Own planner state and coordination for one adapter iteration."""

    def __init__(
        self,
        config: LBAConfig,
        records: Iterator[list[LengthRecord]],
        collate_fn: CollateFn,
        budget_source: BatchSizeSource,
        distributed: DistributedBatchCoordinator,
        reporter: RunReporter,
        logger: logging.Logger,
        event_writer: EventWriter,
        *,
        max_batches: Optional[int],
    ) -> None:
        self.config = config
        self.records = records
        self.collate_fn = collate_fn
        self.budget_source = budget_source
        self.distributed = distributed
        self.reporter = reporter
        self.logger = logger
        self.event_writer = event_writer
        self.max_batches = max_batches
        self.max_padded_length: Optional[int] = None
        self.planner_stats = PlannerStats()

    def run(self, *, distributed: bool) -> Generator[Any, None, None]:
        if self.max_batches == 0:
            return

        resolver = BudgetResolver(self.config, self.budget_source)
        before = PaddingStats()
        after = PaddingStats()
        warmup_batches = self.collect_warmup_batches(
            resolver,
            before,
            distributed=distributed,
        )
        self.max_padded_length = self.resolve_max_padded_length(
            resolver,
            warmup_batches,
            distributed=distributed,
        )
        planner = self.build_planner(self.max_padded_length)

        try:
            yield from self.iter_plans(
                planner,
                warmup_batches,
                before,
                after,
                distributed=distributed,
            )
        finally:
            self.planner_stats = planner.stats
            self.reporter.log_summary(
                before,
                after,
                planner.stats,
                max_padded_length=self.max_padded_length,
            )
            planner.close()

    def collect_warmup_batches(
        self,
        resolver: BudgetResolver,
        before: PaddingStats,
        *,
        distributed: bool,
    ) -> list[list[LengthRecord]]:
        warmup_batches: list[list[LengthRecord]] = []
        if self.config.max_padded_length is not None:
            return warmup_batches

        for warmup_index in range(resolver.warmup_batch_count()):
            has_batch, records = self.next_source_batch(
                distributed=distributed,
                context=f"warmup batch {warmup_index}",
            )
            if not has_batch:
                break
            before.add_length_records(records)
            warmup_batches.append(records)
        return warmup_batches

    def resolve_max_padded_length(
        self,
        resolver: BudgetResolver,
        warmup_batches: Iterable[Iterable[LengthRecord]],
        *,
        distributed: bool,
    ) -> int:
        value = resolver.resolve(self.flatten_records(warmup_batches))
        if distributed:
            return self.distributed.sync_max_padded_length(value)
        return value

    def build_planner(self, max_padded_length: int) -> BatchPlanner:
        return BatchPlanner(
            max_padded_length=max_padded_length,
            max_cache_samples=self.config.max_cache_samples,
            max_padding_ratio=self.config.max_padding_ratio,
            max_candidate_windows=self.config.candidate_window_limit,
            limited_search_fallback_after=(
                self.config.limited_search_fallback_after_limit
            ),
            limited_search_fallback_pool_size=(
                self.config.limited_search_fallback_pool_limit
            ),
            spill_dir=self.distributed.spill_dir_for_rank(),
            logger=self.logger,
            event_writer=self.event_writer,
        )

    def iter_plans(
        self,
        planner: BatchPlanner,
        warmup_batches: Iterable[Iterable[LengthRecord]],
        before: PaddingStats,
        after: PaddingStats,
        *,
        distributed: bool,
    ) -> Generator[Any, None, None]:
        arrival_id = 0
        yielded_batches = 0
        for records in self.iter_record_groups(
            warmup_batches,
            before,
            distributed=distributed,
        ):
            if not self.batch_limit_reached(yielded_batches):
                sample_records, arrival_id = self.assign_arrival_ids(records, arrival_id)
                plans = self.plans_after_add(planner, sample_records)
                for batch in self.collate_plans(plans, after):
                    yield batch
                    yielded_batches += 1
                    if self.batch_limit_reached(yielded_batches):
                        break

            if self.batch_limit_reached(yielded_batches):
                if not distributed:
                    return
                if self.distributed.all_ranks_reached_batch_limit(True):
                    return
            elif distributed and self.max_batches is not None:
                if self.distributed.all_ranks_reached_batch_limit(False):
                    return

        if distributed:
            final_plans = self.distributed.flush_plans(
                planner.drain_records(),
                max_padded_length=self.require_max_padded_length(),
            )
        else:
            final_plans = planner.flush()
        for batch in self.collate_plans(final_plans, after):
            yield batch
            yielded_batches += 1
            if self.batch_limit_reached(yielded_batches):
                return

    def batch_limit_reached(self, yielded_batches: int) -> bool:
        return self.max_batches is not None and yielded_batches >= self.max_batches

    def iter_record_groups(
        self,
        warmup_batches: Iterable[Iterable[LengthRecord]],
        before: PaddingStats,
        *,
        distributed: bool,
    ) -> Generator[Iterable[LengthRecord], None, None]:
        if distributed:
            yield from warmup_batches
        else:
            warmup_records = self.flatten_records(warmup_batches)
            if warmup_records:
                yield warmup_records

        while True:
            has_batch, records = self.next_source_batch(
                distributed=distributed,
                context="source batch",
            )
            if not has_batch:
                break
            before.add_length_records(records)
            yield records

    @staticmethod
    def plans_after_add(
        planner: BatchPlanner,
        records: list[SampleRecord],
    ) -> list[BatchPlan]:
        planner.add_records(records)
        plan = planner.pop_ready()
        if plan is None:
            return []
        return [plan]

    def collate_plans(
        self,
        plans: Iterable[BatchPlan],
        after: PaddingStats,
    ) -> Generator[Any, None, None]:
        for plan in plans:
            after.add_plan(plan)
            if plan.reason == PlanReason.OVERSIZED:
                self.reporter.warn_oversized_sample(
                    plan.records[0],
                    max_padded_length=self.max_padded_length,
                )
            yield self.collate_fn(plan.samples)

    @staticmethod
    def flatten_records(
        batches: Iterable[Iterable[LengthRecord]],
    ) -> list[LengthRecord]:
        return [record for records in batches for record in records]

    @staticmethod
    def assign_arrival_ids(
        records: Iterable[LengthRecord],
        next_arrival_id: int,
    ) -> tuple[list[SampleRecord], int]:
        sample_records: list[SampleRecord] = []
        for record in records:
            sample_records.append(
                SampleRecord(
                    sample=record.sample,
                    length=record.length,
                    arrival_id=next_arrival_id,
                    index=record.index,
                )
            )
            next_arrival_id += 1
        return sample_records, next_arrival_id

    def next_source_batch(
        self,
        *,
        distributed: bool,
        context: str,
    ) -> tuple[bool, list[LengthRecord]]:
        try:
            records = list(next(self.records))
            local_has_batch = 1
        except StopIteration:
            records = []
            local_has_batch = 0

        if not distributed:
            return bool(local_has_batch), records

        self.distributed.validate_source_batch_presence(
            local_has_batch,
            context=context,
        )
        return bool(local_has_batch), records

    def require_max_padded_length(self) -> int:
        if self.max_padded_length is None:
            raise RuntimeError("LBA has no active max_padded_length for flushing.")
        return self.max_padded_length
