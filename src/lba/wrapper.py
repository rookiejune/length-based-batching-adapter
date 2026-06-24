"""Top-level dataloader adapter."""

from __future__ import annotations

import warnings
from collections.abc import Generator, Iterable, Iterator
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from .config import DEFAULT_PREFETCH_BATCHES, LBAConfig, PlannerMode
from .distributed import DistributedBatchCoordinator
from .estimator import LengthBudgetResolver
from .logging_utils import JsonlEventWriter, create_run_logger, event_log_path_for
from .metrics import PaddingStats, PlannerStats, padding_ratio_reduction
from .planner import BatchPlanner
from .prefetch import prefetch_iterator
from .source import build_source_loader
from .types import BatchPlan, LengthFn, LengthRecord, SampleRecord


class LengthBatchingAdapter:
    """Wrap a dataloader and prepare length-based dynamic batches."""

    def __init__(
        self,
        dataloader: DataLoader,
        *,
        len_fn: LengthFn,
        max_padded_length: int | None = None,
        warmup_batches: int | None = None,
        max_cache_samples: int = 8192,
        max_padding_ratio: float = 0.05,
        prefetch_batches: int = DEFAULT_PREFETCH_BATCHES,
        planner_mode: PlannerMode = "quality",
        max_candidate_windows: int | None = None,
        drop_last_flush: bool = True,
        spill_dir: str | Path | None = None,
        log_dir: str | Path | None = None,
    ) -> None:
        if len_fn is None:
            raise TypeError("len_fn is required.")

        self.dataloader = dataloader
        self.len_fn = len_fn
        self.original_collate_fn = dataloader.collate_fn
        self.config = LBAConfig(
            max_padded_length=max_padded_length,
            warmup_batches=warmup_batches,
            max_cache_samples=max_cache_samples,
            max_padding_ratio=max_padding_ratio,
            prefetch_batches=prefetch_batches,
            planner_mode=planner_mode,
            max_candidate_windows=max_candidate_windows,
            drop_last_flush=drop_last_flush,
            spill_dir=spill_dir,
            log_dir=log_dir,
        )
        self.logger, self.log_path = create_run_logger(log_dir)
        self.log_event_path = event_log_path_for(self.log_path)
        self.event_writer = JsonlEventWriter(self.log_event_path)
        self._distributed = DistributedBatchCoordinator(
            dataloader,
            self.config,
            self.logger,
            self.event_writer,
        )
        self._active_max_padded_length: int | None = None
        self.last_planner_stats = PlannerStats()

        warnings.warn(
            f"LBA log file: {self.log_path}; structured events: {self.log_event_path}",
            stacklevel=2,
        )
        self.logger.info(
            "lba run: log=%s events=%s",
            self.log_path,
            self.log_event_path,
        )
        self.event_writer.write(
            "run_start",
            {
                "log_path": str(self.log_path),
                "event_path": str(self.log_event_path),
                "config": self._config_event_fields(),
            },
        )
        if max_padded_length is not None:
            warnings.warn(
                "max_padded_length is set explicitly and overrides warmup inference.",
                stacklevel=2,
            )
            self.logger.warning(
                "lba config: explicit max_padded_length=%s overrides warmup inference",
                max_padded_length,
            )
            self.event_writer.write(
                "config_warning",
                {
                    "reason": "explicit_max_padded_length",
                    "max_padded_length": max_padded_length,
                },
            )

    @property
    def max_padded_length(self) -> int | None:
        return self.config.max_padded_length

    def __iter__(self) -> Iterator[Any]:
        if DistributedBatchCoordinator.is_initialized():
            if self.config.prefetch_batches > 0:
                self.logger.info(
                    "disabled LBA prefetch for torch.distributed iteration"
                )
            return self._iter_distributed_sync()

        iterator = self._iter_sync()
        if self.config.prefetch_batches > 0:
            return prefetch_iterator(iterator, self.config.prefetch_batches)
        return iterator

    def _iter_sync(self) -> Generator[Any, None, None]:
        return self._iter_planned(distributed=False)

    def _iter_distributed_sync(self) -> Generator[Any, None, None]:
        return self._iter_planned(distributed=True)

    def _iter_planned(self, *, distributed: bool) -> Generator[Any, None, None]:
        record_loader = build_source_loader(self.dataloader, self.len_fn)
        length_record_iter = iter(record_loader)
        resolver = LengthBudgetResolver(self.config, self.dataloader)
        before_padding_stats = PaddingStats()
        after_padding_stats = PaddingStats()
        warmup_batches = self._collect_warmup_batches(
            length_record_iter,
            resolver,
            before_padding_stats,
            distributed=distributed,
        )
        resolved_max_padded_length = self._resolve_iteration_max_padded_length(
            resolver,
            warmup_batches,
            distributed=distributed,
        )
        self._active_max_padded_length = resolved_max_padded_length
        planner = self._build_planner(resolved_max_padded_length)

        try:
            yield from self._iter_plans(
                planner,
                warmup_batches,
                length_record_iter,
                before_padding_stats,
                after_padding_stats,
                distributed=distributed,
            )
        finally:
            self.last_planner_stats = planner.stats
            self._log_run_summary(
                before_padding_stats,
                after_padding_stats,
                planner.stats,
            )
            planner.close()

    def _collect_warmup_batches(
        self,
        length_record_iter: Iterator[list[LengthRecord]],
        resolver: LengthBudgetResolver,
        before_padding_stats: PaddingStats,
        *,
        distributed: bool,
    ) -> list[list[LengthRecord]]:
        warmup_batches: list[list[LengthRecord]] = []
        if self.config.max_padded_length is not None:
            return warmup_batches

        for warmup_index in range(resolver.warmup_batch_count()):
            has_batch, length_records = self._next_source_batch(
                length_record_iter,
                distributed=distributed,
                context=f"warmup batch {warmup_index}",
            )
            if not has_batch:
                break
            before_padding_stats.add_length_records(length_records)
            warmup_batches.append(length_records)
        return warmup_batches

    def _resolve_iteration_max_padded_length(
        self,
        resolver: LengthBudgetResolver,
        warmup_batches: Iterable[Iterable[LengthRecord]],
        *,
        distributed: bool,
    ) -> int:
        warmup_length_records = self._flatten_length_records(warmup_batches)
        resolved_max_padded_length = resolver.resolve(warmup_length_records)
        if distributed:
            return self._distributed.sync_max_padded_length(resolved_max_padded_length)
        return resolved_max_padded_length

    def _build_planner(self, max_padded_length: int) -> BatchPlanner:
        return BatchPlanner(
            max_padded_length=max_padded_length,
            max_cache_samples=self.config.max_cache_samples,
            max_padding_ratio=self.config.max_padding_ratio,
            max_candidate_windows=self.config.candidate_window_limit,
            spill_dir=self._distributed.spill_dir_for_rank(),
            logger=self.logger,
            event_writer=self.event_writer,
        )

    def _iter_plans(
        self,
        planner: BatchPlanner,
        warmup_batches: Iterable[Iterable[LengthRecord]],
        length_record_iter: Iterator[list[LengthRecord]],
        before_padding_stats: PaddingStats,
        after_padding_stats: PaddingStats,
        *,
        distributed: bool,
    ) -> Generator[Any, None, None]:
        arrival_id = 0
        for length_records in self._iter_length_record_groups(
            warmup_batches,
            length_record_iter,
            before_padding_stats,
            distributed=distributed,
        ):
            sample_records, arrival_id = self._assign_arrival_ids(
                length_records,
                arrival_id,
            )
            plans = self._plans_after_add(
                planner,
                sample_records,
            )
            yield from self._collate_plans(plans, after_padding_stats)

        if distributed:
            yield from self._collate_plans(
                self._distributed_flush_plans(planner),
                after_padding_stats,
            )
        else:
            yield from self._collate_plans(planner.flush(), after_padding_stats)

    def _iter_length_record_groups(
        self,
        warmup_batches: Iterable[Iterable[LengthRecord]],
        length_record_iter: Iterator[list[LengthRecord]],
        before_padding_stats: PaddingStats,
        *,
        distributed: bool,
    ) -> Generator[Iterable[LengthRecord], None, None]:
        if distributed:
            yield from warmup_batches
        else:
            warmup_length_records = self._flatten_length_records(warmup_batches)
            if warmup_length_records:
                yield warmup_length_records

        yield from self._remaining_source_batches(
            length_record_iter,
            before_padding_stats,
            distributed=distributed,
        )

    def _remaining_source_batches(
        self,
        length_record_iter: Iterator[list[LengthRecord]],
        before_padding_stats: PaddingStats,
        *,
        distributed: bool,
    ) -> Generator[list[LengthRecord], None, None]:
        while True:
            has_batch, length_records = self._next_source_batch(
                length_record_iter,
                distributed=distributed,
                context="source batch",
            )
            if not has_batch:
                break
            before_padding_stats.add_length_records(length_records)
            yield length_records

    def _plans_after_add(
        self,
        planner: BatchPlanner,
        sample_records: list[SampleRecord],
    ) -> list[BatchPlan]:
        planner.add_records(sample_records)
        plan = planner.pop_ready()
        if plan is None:
            return []
        return [plan]

    def _collate_plans(
        self, plans: Iterable[BatchPlan], after_padding_stats: PaddingStats
    ) -> Generator[Any, None, None]:
        for plan in plans:
            yield self._collate_recorded_plan(plan, after_padding_stats)

    @staticmethod
    def _flatten_length_records(
        length_record_batches: Iterable[Iterable[LengthRecord]],
    ) -> list[LengthRecord]:
        return [
            record
            for length_records in length_record_batches
            for record in length_records
        ]

    def _assign_arrival_ids(
        self, length_records: Iterable[LengthRecord], next_arrival_id: int
    ) -> tuple[list[SampleRecord], int]:
        sample_records: list[SampleRecord] = []
        for length_record in length_records:
            sample_records.append(
                SampleRecord(
                    sample=length_record.sample,
                    length=length_record.length,
                    arrival_id=next_arrival_id,
                    index=length_record.index,
                )
            )
            next_arrival_id += 1
        return sample_records, next_arrival_id

    def _next_source_batch(
        self,
        length_record_iter: Iterator[list[LengthRecord]],
        *,
        distributed: bool,
        context: str,
    ) -> tuple[bool, list[LengthRecord]]:
        try:
            length_records = list(next(length_record_iter))
            local_has_batch = 1
        except StopIteration:
            length_records = []
            local_has_batch = 0

        if not distributed:
            return bool(local_has_batch), length_records

        self._distributed.validate_source_batch_presence(
            local_has_batch,
            context=context,
        )
        return bool(local_has_batch), length_records

    def _distributed_flush_plans(self, planner: BatchPlanner) -> list[BatchPlan]:
        return self._distributed.flush_plans(
            planner.drain_records(),
            max_padded_length=self._require_active_max_padded_length(),
        )

    def _require_active_max_padded_length(self) -> int:
        if self._active_max_padded_length is None:
            raise RuntimeError("LBA has no active max_padded_length for flushing.")
        return self._active_max_padded_length

    def _collate_recorded_plan(
        self, plan: BatchPlan, after_padding_stats: PaddingStats
    ) -> Any:
        after_padding_stats.add_plan(plan)
        return self._collate_plan(plan)

    def _collate_plan(self, plan: BatchPlan) -> Any:
        if plan.reason == "oversized":
            oversized_sample = plan.records[0].sample
            active_max_padded_length = self._active_max_padded_length
            sample_index = plan.records[0].index
            sample_type = type(oversized_sample).__name__
            warnings.warn(
                f"LBA oversized sample length={plan.records[0].length} "
                f"max_padded_length={active_max_padded_length} "
                "was emitted as a singleton batch.",
                stacklevel=2,
            )
            self.logger.warning(
                "lba health: oversized sample length=%s budget=%s index=%s "
                "sample_type=%s action=emitted_singleton",
                plan.records[0].length,
                active_max_padded_length,
                self._format_optional(sample_index),
                sample_type,
            )
            self.event_writer.write(
                "oversized_sample",
                {
                    "length": plan.records[0].length,
                    "max_padded_length": active_max_padded_length,
                    "index": sample_index,
                    "sample_type": sample_type,
                },
            )
        return self.original_collate_fn(plan.samples)

    def _log_run_summary(
        self,
        before_padding_stats: PaddingStats,
        after_padding_stats: PaddingStats,
        planner_stats: PlannerStats,
    ) -> None:
        reduction = padding_ratio_reduction(before_padding_stats, after_padding_stats)
        saved_padding_length = (
            before_padding_stats.padding_length_sum
            - after_padding_stats.padding_length_sum
        )
        self.logger.info(
            "lba summary: padding %s -> %s (%s reduction) saved_padding=%s "
            "batches=%s->%s samples=%s",
            self._format_percent_value(before_padding_stats.global_padding_ratio),
            self._format_percent_value(after_padding_stats.global_padding_ratio),
            self._format_percent_value(reduction),
            self._format_signed_int(saved_padding_length),
            before_padding_stats.batch_count,
            after_padding_stats.batch_count,
            after_padding_stats.sample_count,
        )
        self.logger.info(
            "lba planner: total=%s pop_ready_avg=%sms sort_avg=%sms "
            "paths=fast:%s/full:%s/flush:%s max_cache=%s",
            self._format_seconds(planner_stats.planner_time_seconds),
            self._format_milliseconds(planner_stats.average_pop_ready_time_ms),
            self._format_milliseconds(planner_stats.average_sort_time_ms),
            planner_stats.fast_path_batch_count,
            planner_stats.full_search_batch_count,
            planner_stats.flush_search_batch_count,
            planner_stats.max_cache_size_seen,
        )
        self.logger.info(
            "lba health: oversized=%s spill_events=%s spilled_records=%s "
            "no_ready=%s other_batches=%s event_log=%s",
            after_padding_stats.oversized_batch_count,
            planner_stats.spill_event_count,
            planner_stats.spilled_record_count,
            planner_stats.no_ready_call_count,
            after_padding_stats.other_batch_count,
            self.log_event_path,
        )
        self.event_writer.write(
            "summary",
            {
                "max_padded_length": self._active_max_padded_length,
                "padding": {
                    "before": self._padding_event_fields(before_padding_stats),
                    "after": self._padding_event_fields(after_padding_stats),
                    "padding_ratio_reduction": reduction,
                    "saved_padding_length": saved_padding_length,
                    "saved_padded_length": (
                        before_padding_stats.padded_length_sum
                        - after_padding_stats.padded_length_sum
                    ),
                },
                "planner": self._planner_event_fields(planner_stats),
                "health": {
                    "oversized_batches": after_padding_stats.oversized_batch_count,
                    "planner_oversized_batches": planner_stats.oversized_batch_count,
                    "other_batches": after_padding_stats.other_batch_count,
                    "spill_events": planner_stats.spill_event_count,
                    "spilled_records": planner_stats.spilled_record_count,
                    "no_ready_calls": planner_stats.no_ready_call_count,
                },
            },
        )

    def _config_event_fields(self) -> dict[str, object]:
        return {
            "max_padded_length": self.config.max_padded_length,
            "warmup_batches": self.config.warmup_batches,
            "max_cache_samples": self.config.max_cache_samples,
            "max_padding_ratio": self.config.max_padding_ratio,
            "prefetch_batches": self.config.prefetch_batches,
            "planner_mode": self.config.planner_mode,
            "max_candidate_windows": self.config.max_candidate_windows,
            "candidate_window_limit": self.config.candidate_window_limit,
            "drop_last_flush": self.config.drop_last_flush,
            "spill_dir": self._path_or_none(self.config.spill_dir),
            "log_dir": self._path_or_none(self.config.log_dir),
        }

    def _padding_event_fields(self, stats: PaddingStats) -> dict[str, object]:
        return {
            "batch_count": stats.batch_count,
            "sample_count": stats.sample_count,
            "raw_length_sum": stats.raw_length_sum,
            "padded_length_sum": stats.padded_length_sum,
            "padding_length_sum": stats.padding_length_sum,
            "padding_ratio": stats.global_padding_ratio,
            "mean_batch_padding_ratio": stats.mean_batch_padding_ratio,
            "planned_batch_count": stats.planned_batch_count,
            "oversized_batch_count": stats.oversized_batch_count,
            "other_batch_count": stats.other_batch_count,
        }

    def _planner_event_fields(self, stats: PlannerStats) -> dict[str, object]:
        return {
            "planner_time_seconds": stats.planner_time_seconds,
            "sort_time_seconds": stats.sort_time_seconds,
            "sort_calls": stats.sort_call_count,
            "average_sort_time_ms": stats.average_sort_time_ms,
            "pop_ready_time_seconds": stats.pop_ready_time_seconds,
            "pop_ready_calls": stats.pop_ready_call_count,
            "average_pop_ready_time_ms": stats.average_pop_ready_time_ms,
            "candidate_window_checks": stats.candidate_window_checks,
            "average_candidate_window_checks": (
                stats.average_candidate_window_checks
            ),
            "max_candidate_window_checks": stats.max_candidate_window_checks,
            "records_sorted_total": stats.records_sorted_total,
            "max_cache_size_seen": stats.max_cache_size_seen,
            "spill_events": stats.spill_event_count,
            "spilled_records": stats.spilled_record_count,
            "paths": {
                "fast_path": {
                    "batches": stats.fast_path_batch_count,
                    "time_seconds": stats.fast_path_time_seconds,
                    "candidate_window_checks": (
                        stats.fast_path_candidate_window_checks
                    ),
                },
                "full_search": {
                    "batches": stats.full_search_batch_count,
                    "time_seconds": stats.full_search_time_seconds,
                    "candidate_window_checks": (
                        stats.full_search_candidate_window_checks
                    ),
                },
                "flush_search": {
                    "batches": stats.flush_search_batch_count,
                    "time_seconds": stats.flush_search_time_seconds,
                    "candidate_window_checks": (
                        stats.flush_search_candidate_window_checks
                    ),
                },
                "oversized": {
                    "batches": stats.oversized_batch_count,
                    "time_seconds": stats.oversized_time_seconds,
                },
                "no_ready": {
                    "calls": stats.no_ready_call_count,
                    "time_seconds": stats.no_ready_time_seconds,
                },
            },
        }

    @staticmethod
    def _format_percent_value(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value * 100:.2f}%"

    @staticmethod
    def _format_seconds(value: float) -> str:
        if value < 1:
            return f"{value * 1000:.3f}ms"
        return f"{value:.3f}s"

    @staticmethod
    def _format_signed_int(value: int) -> str:
        if value > 0:
            return f"+{value}"
        return str(value)

    @staticmethod
    def _format_optional(value: object | None) -> str:
        if value is None:
            return "n/a"
        return str(value)

    @staticmethod
    def _path_or_none(value: str | Path | None) -> str | None:
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _format_milliseconds(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.3f}"

LBA = LengthBatchingAdapter
