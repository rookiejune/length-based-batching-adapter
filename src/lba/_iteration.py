"""One iteration of the length-based batching pipeline."""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterable, Iterator
from typing import Any, Optional

from ._api_types import CollateFn, EventWriter
from ._pin_memory import pin_batch, pin_memory_enabled
from ._records import BatchPlan, LengthRecord, PlanReason, SampleRecord
from ._run_reporter import RunReporter
from .adaptive import AdaptiveState
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
        pin_memory: bool,
        pin_memory_device: Optional[str],
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
        self.pin_memory = pin_memory_enabled(
            requested=pin_memory,
            device=pin_memory_device,
        )
        self.pin_memory_device = pin_memory_device
        self.max_padded_length: Optional[int] = None
        self.max_batch_cost: Optional[int] = None
        self.planner_stats = PlannerStats()
        self.adaptive = (
            AdaptiveState(config.adaptive)
            if config.adaptive is not None
            else None
        )

    def run(self, *, distributed: bool) -> Generator[Any, None, None]:
        if self.max_batches == 0:
            return

        before = PaddingStats()
        after = PaddingStats()
        if self.config.uses_custom_cost:
            warmup_batches: list[list[LengthRecord]] = []
            self.max_batch_cost = self.resolve_max_batch_cost(
                distributed=distributed,
            )
        else:
            resolver = BudgetResolver(self.config, self.budget_source)
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
            self.max_batch_cost = self.max_padded_length
        planner = self.build_planner(self.require_max_batch_cost())

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
                max_batch_cost=self.max_batch_cost,
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

    def resolve_max_batch_cost(self, *, distributed: bool) -> int:
        value = self.config.max_batch_cost
        if value is None:
            raise RuntimeError("LBA custom cost mode has no max_batch_cost.")
        if distributed:
            return self.distributed.sync_max_batch_cost(value)
        return value

    def build_planner(self, max_batch_cost: int) -> BatchPlanner:
        return BatchPlanner(
            max_padded_length=self.max_padded_length,
            cost_fn=self.config.cost_fn,
            max_batch_cost=max_batch_cost,
            max_cache_samples=self.config.max_cache_samples,
            max_padding_ratio=self.current_max_padding_ratio(),
            max_candidate_windows=self.current_candidate_window_limit(),
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

    def current_max_padding_ratio(self) -> float:
        if self.adaptive is None or self.adaptive.max_padding_ratio is None:
            return self.config.max_padding_ratio
        return float(self.adaptive.max_padding_ratio)

    def current_candidate_window_limit(self) -> Optional[int]:
        if self.adaptive is None or self.adaptive.max_candidate_windows is None:
            return self.config.candidate_window_limit
        return int(self.adaptive.max_candidate_windows)

    def apply_adaptive_update(
        self,
        update: Optional[dict[str, object] | list[dict[str, object]]],
        planner: BatchPlanner,
    ) -> None:
        if update is None:
            return
        if isinstance(update, list):
            for item in update:
                self.apply_adaptive_update(item, planner)
            return
        if update["knob"] == "max_padding_ratio":
            planner.max_padding_ratio = float(update["new_value"])
        elif update["knob"] == "max_candidate_windows":
            planner.max_candidate_windows = int(update["new_value"])
        self.event_writer.write("adaptive_planner_update", update)

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
        pending_plans: list[BatchPlan] = []
        for records in self.iter_record_groups(
            warmup_batches,
            before,
            distributed=distributed,
        ):
            if not self.batch_limit_reached(yielded_batches):
                sample_records, arrival_id = self.assign_arrival_ids(records, arrival_id)
                plans = self.plans_after_add(
                    planner,
                    sample_records,
                    require_plan=distributed,
                )
                pending_plans.extend(plans)
                while self.steady_window_ready(
                    pending_plans,
                    distributed=distributed,
                ):
                    window = self.take_steady_window(
                        pending_plans,
                        distributed=distributed,
                        force=False,
                        step_offset=yielded_batches,
                    )
                    for batch in self.collate_plans(window, after):
                        yield batch
                        yielded_batches += 1
                        if self.batch_limit_reached(yielded_batches):
                            break
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

        pending_window = self.take_steady_window(
            pending_plans,
            distributed=distributed,
            force=True,
            step_offset=yielded_batches,
        )
        for batch in self.collate_plans(pending_window, after):
            yield batch
            yielded_batches += 1
            if self.batch_limit_reached(yielded_batches):
                return

        if distributed:
            final_plans = self.distributed.flush_plans(
                planner.drain_records(),
                max_batch_cost=self.require_max_batch_cost(),
            )
        else:
            final_plans = planner.flush()
        for batch in self.collate_plans(
            self.iter_cost_windows(final_plans),
            after,
        ):
            yield batch
            yielded_batches += 1
            if self.batch_limit_reached(yielded_batches):
                return

    def batch_limit_reached(self, yielded_batches: int) -> bool:
        return self.max_batches is not None and yielded_batches >= self.max_batches

    def steady_window_ready(
        self,
        plans: list[BatchPlan],
        *,
        distributed: bool,
    ) -> bool:
        return len(plans) >= self.steady_window_size(distributed=distributed)

    def steady_window_size(self, *, distributed: bool) -> int:
        if (
            distributed
            and self.adaptive is not None
            and self.adaptive.distributed_cost_window_batches is not None
        ):
            return self.adaptive.distributed_cost_window_batches
        distributed_window = self.config.distributed_cost_window_batches
        if distributed and distributed_window is not None:
            return distributed_window
        return self.config.cost_window_batches

    def take_steady_window(
        self,
        plans: list[BatchPlan],
        *,
        distributed: bool,
        force: bool,
        step_offset: int,
    ) -> list[BatchPlan]:
        if not plans:
            return []
        window_size = self.steady_window_size(distributed=distributed)
        if not force and len(plans) < window_size:
            return []
        take_count = len(plans) if force else window_size
        window = plans[:take_count]
        del plans[:take_count]

        if (
            distributed
            and self.config.distributed_cost_window_batches is not None
        ):
            return self.distributed.match_cost_plans(
                window,
                step_offset=step_offset,
            )
        if (
            distributed
            and self.adaptive is not None
            and self.adaptive.distributed_cost_window_batches is not None
        ):
            old_window_batches = self.adaptive.distributed_cost_window_batches
            matched_window, stats = self.distributed.match_cost_plans_with_stats(
                window,
                step_offset=step_offset,
            )
            update = self.adaptive.update_cost_window(stats)
            new_window_batches = self.adaptive.distributed_cost_window_batches
            self.event_writer.write(
                "adaptive_distributed_cost_window",
                {
                    "step_offset": step_offset,
                    "old_window_batches": old_window_batches,
                    "new_window_batches": new_window_batches,
                    "block_size": stats.block_size,
                    "mean_cost": stats.mean_cost,
                    "source_mean_step_spread": stats.source_mean_step_spread,
                    "matched_mean_step_spread": stats.matched_mean_step_spread,
                    "source_spread_ratio": stats.source_spread_ratio,
                    "improvement_ratio": stats.improvement_ratio,
                    "remote_batches": stats.remote_plan_count,
                    "remote_records": stats.remote_record_count,
                    "update": update,
                },
            )
            return matched_window
        return self.order_cost_window(window)

    def cost_window_ready(self, plans: list[BatchPlan]) -> bool:
        return len(plans) >= self.config.cost_window_batches

    def take_cost_window(
        self,
        plans: list[BatchPlan],
        *,
        force: bool,
    ) -> list[BatchPlan]:
        if not plans:
            return []
        window_size = self.config.cost_window_batches
        if not force and len(plans) < window_size:
            return []
        take_count = len(plans) if force else window_size
        window = plans[:take_count]
        del plans[:take_count]
        return self.order_cost_window(window)

    def iter_cost_windows(
        self,
        plans: Iterable[BatchPlan],
    ) -> Generator[BatchPlan, None, None]:
        pending: list[BatchPlan] = []
        for plan in plans:
            pending.append(plan)
            if self.cost_window_ready(pending):
                yield from self.take_cost_window(pending, force=False)
        yield from self.take_cost_window(pending, force=True)

    @staticmethod
    def order_cost_window(plans: Iterable[BatchPlan]) -> list[BatchPlan]:
        return sorted(
            plans,
            key=lambda plan: (
                -(
                    plan.estimated_cost
                    if plan.estimated_cost is not None
                    else plan.padded_length
                ),
                plan.records[0].arrival_id,
            ),
        )

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

    def plans_after_add(
        self,
        planner: BatchPlanner,
        records: list[SampleRecord],
        *,
        require_plan: bool = False,
    ) -> list[BatchPlan]:
        if require_plan and not records:
            raise RuntimeError(
                "LBA distributed mode requires non-empty source batches."
            )
        planner.add_records(records)
        plan = planner.pop_required() if require_plan else planner.pop_ready()
        if plan is None:
            if self.adaptive is not None:
                self.apply_adaptive_update(
                    self.adaptive.feedback_for_missing_plan(),
                    planner,
                )
            return []
        if self.adaptive is not None:
            self.apply_adaptive_update(
                self.adaptive.feedback_for_plan(plan),
                planner,
            )
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
                    max_batch_cost=self.max_batch_cost,
                    estimated_cost=plan.estimated_cost,
                )
            batch = self.collate_fn(plan.samples)
            yield pin_batch(
                batch,
                enabled=self.pin_memory,
                device=self.pin_memory_device,
            )

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
            records = next(self.records)
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

    def require_max_batch_cost(self) -> int:
        if self.max_batch_cost is None:
            raise RuntimeError("LBA has no active batch cost budget.")
        return self.max_batch_cost
