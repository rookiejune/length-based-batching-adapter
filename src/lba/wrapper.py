"""Top-level dataloader adapter."""

from __future__ import annotations

import warnings
from collections.abc import Generator, Iterable, Iterator
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from .budget import BudgetResolver
from .config import DEFAULT_PREFETCH_BATCHES, LBAConfig, PlannerMode
from .distributed import DistributedBatchCoordinator
from .logging_utils import (
    JsonlEventWriter,
    RunReporter,
    create_run_logger,
    event_log_path_for,
)
from .metrics import PaddingStats, PlannerStats
from .planner import BatchPlanner
from .prefetch import prefetch_iterator
from .source import build_source_loader
from .types import BatchPlan, LengthFn, LengthRecord, PlanReason, SampleRecord


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
        limited_search_fallback_after: int | None = None,
        limited_search_fallback_pool_size: int | None = None,
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
            limited_search_fallback_after=limited_search_fallback_after,
            limited_search_fallback_pool_size=limited_search_fallback_pool_size,
            drop_last_flush=drop_last_flush,
            spill_dir=spill_dir,
            log_dir=log_dir,
        )
        self.logger, self.log_path = create_run_logger(log_dir)
        self.log_event_path = event_log_path_for(self.log_path)
        self.event_writer = JsonlEventWriter(self.log_event_path)
        self.reporter = RunReporter(
            self.logger,
            self.event_writer,
            self.log_event_path,
        )
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
        resolver = BudgetResolver(self.config, self.dataloader)
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
            self.reporter.log_summary(
                before_padding_stats,
                after_padding_stats,
                planner.stats,
                max_padded_length=self._active_max_padded_length,
            )
            planner.close()

    def _collect_warmup_batches(
        self,
        length_record_iter: Iterator[list[LengthRecord]],
        resolver: BudgetResolver,
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
        resolver: BudgetResolver,
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
            limited_search_fallback_after=(
                self.config.limited_search_fallback_after_limit
            ),
            limited_search_fallback_pool_size=(
                self.config.limited_search_fallback_pool_limit
            ),
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
        if plan.reason == PlanReason.OVERSIZED:
            self.reporter.warn_oversized_sample(
                plan.records[0],
                max_padded_length=self._active_max_padded_length,
            )
        return self.original_collate_fn(plan.samples)

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
            "limited_search_fallback_after": (
                self.config.limited_search_fallback_after
            ),
            "limited_search_fallback_after_limit": (
                self.config.limited_search_fallback_after_limit
            ),
            "limited_search_fallback_pool_size": (
                self.config.limited_search_fallback_pool_size
            ),
            "limited_search_fallback_pool_limit": (
                self.config.limited_search_fallback_pool_limit
            ),
            "drop_last_flush": self.config.drop_last_flush,
            "spill_dir": self._path_or_none(self.config.spill_dir),
            "log_dir": self._path_or_none(self.config.log_dir),
        }

    @staticmethod
    def _path_or_none(value: str | Path | None) -> str | None:
        if value is None:
            return None
        return str(value)

LBA = LengthBatchingAdapter
