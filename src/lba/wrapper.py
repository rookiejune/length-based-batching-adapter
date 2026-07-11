"""Top-level dataloader adapter."""

from __future__ import annotations

from collections.abc import Generator, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Optional, Union

from torch.utils.data import DataLoader

from ._adapter_logging import AdapterRunLogger
from .budget import BatchSizeSource, BudgetResolver
from .config import DEFAULT_PREFETCH_BATCHES, LBAConfig, PlannerMode
from .distributed import DistributedBatchCoordinator
from .metrics import PaddingStats, PlannerStats
from .planner import BatchPlanner
from .prefetch import prefetch_iterator
from .source import build_source_loader, iter_length_record_batches
from ._api_types import CollateFn, LengthFn
from ._records import (
    BatchPlan,
    LengthRecord,
    PlanReason,
    SampleRecord,
)


class _BatchSizeSourceView:
    def __init__(self, batch_size: Optional[int]) -> None:
        self.batch_size = batch_size


class LengthBatchingAdapter:
    """Wrap a dataloader and prepare length-based dynamic batches."""

    def __init__(
        self,
        dataloader: DataLoader,
        *,
        len_fn: LengthFn,
        max_padded_length: Optional[int] = None,
        warmup_batches: Optional[int] = None,
        max_cache_samples: int = 8192,
        max_padding_ratio: float = 0.05,
        prefetch_batches: int = DEFAULT_PREFETCH_BATCHES,
        planner_mode: PlannerMode = "quality",
        max_candidate_windows: Optional[int] = None,
        limited_search_fallback_after: Optional[int] = None,
        limited_search_fallback_pool_size: Optional[int] = None,
        drop_last_flush: bool = True,
        max_batches: Optional[int] = None,
        spill_dir: Optional[Union[str, Path]] = None,
        log_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        if len_fn is None:
            raise TypeError("len_fn is required.")

        self.dataloader = dataloader
        self.len_fn = len_fn
        self.original_collate_fn = dataloader.collate_fn
        self._budget_source: BatchSizeSource = dataloader
        self._init_common(
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
            max_batches=max_batches,
            spill_dir=spill_dir,
            log_dir=log_dir,
            distributed_dataloader=dataloader,
        )

    def _init_common(
        self,
        *,
        max_padded_length: Optional[int],
        warmup_batches: Optional[int],
        max_cache_samples: int,
        max_padding_ratio: float,
        prefetch_batches: int,
        planner_mode: PlannerMode,
        max_candidate_windows: Optional[int],
        limited_search_fallback_after: Optional[int],
        limited_search_fallback_pool_size: Optional[int],
        drop_last_flush: bool,
        max_batches: Optional[int],
        spill_dir: Optional[Union[str, Path]],
        log_dir: Optional[Union[str, Path]],
        distributed_dataloader: Optional[DataLoader],
    ) -> None:
        if max_batches is not None and max_batches < 0:
            raise ValueError("max_batches must be non-negative.")
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
        run_logger = AdapterRunLogger(
            config=self.config,
            max_batches=max_batches,
            log_dir=log_dir,
        )
        self.logger = run_logger.logger
        self.log_path = run_logger.log_path
        self.log_event_path = run_logger.log_event_path
        self.event_writer = run_logger.event_writer
        self.reporter = run_logger.reporter
        self._distributed = DistributedBatchCoordinator(
            distributed_dataloader,
            self.config,
            self.logger,
            self.event_writer,
        )
        self._active_max_padded_length: Optional[int] = None
        self._max_batches = max_batches
        self.last_planner_stats = PlannerStats()

    @property
    def max_padded_length(self) -> Optional[int]:
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
        if self._max_batches == 0:
            return

        length_record_iter = self._length_record_iter()
        resolver = BudgetResolver(self.config, self._budget_source)
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

    def _length_record_iter(self) -> Iterator[list[LengthRecord]]:
        record_loader = build_source_loader(self.dataloader, self.len_fn)
        return iter(record_loader)

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
        yielded_batches = 0
        for length_records in self._iter_length_record_groups(
            warmup_batches,
            length_record_iter,
            before_padding_stats,
            distributed=distributed,
        ):
            if not self._batch_limit_reached(yielded_batches):
                sample_records, arrival_id = self._assign_arrival_ids(
                    length_records,
                    arrival_id,
                )
                plans = self._plans_after_add(
                    planner,
                    sample_records,
                )
                for batch in self._collate_plans(plans, after_padding_stats):
                    yield batch
                    yielded_batches += 1
                    if self._batch_limit_reached(yielded_batches):
                        break

            if self._batch_limit_reached(yielded_batches):
                if not distributed:
                    return
                if self._distributed.all_ranks_reached_batch_limit(True):
                    return
            elif distributed and self._max_batches is not None:
                if self._distributed.all_ranks_reached_batch_limit(False):
                    return

        if distributed:
            final_plans = self._distributed_flush_plans(planner)
        else:
            final_plans = planner.flush()
        for batch in self._collate_plans(final_plans, after_padding_stats):
            yield batch
            yielded_batches += 1
            if self._batch_limit_reached(yielded_batches):
                return

    def _batch_limit_reached(self, yielded_batches: int) -> bool:
        return self._max_batches is not None and yielded_batches >= self._max_batches

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

LBA = LengthBatchingAdapter


class IterableLengthBatchingAdapter(LengthBatchingAdapter):
    """Prepare dynamic batches from an iterable that already yields sample batches."""

    def __init__(
        self,
        source_batches: Iterable[Sequence[Any]],
        *,
        collate_fn: CollateFn,
        len_fn: LengthFn,
        batch_size: Optional[int] = None,
        max_padded_length: Optional[int] = None,
        warmup_batches: Optional[int] = None,
        max_cache_samples: int = 8192,
        max_padding_ratio: float = 0.05,
        prefetch_batches: int = DEFAULT_PREFETCH_BATCHES,
        planner_mode: PlannerMode = "quality",
        max_candidate_windows: Optional[int] = None,
        limited_search_fallback_after: Optional[int] = None,
        limited_search_fallback_pool_size: Optional[int] = None,
        drop_last_flush: bool = True,
        max_batches: Optional[int] = None,
        spill_dir: Optional[Union[str, Path]] = None,
        log_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        if len_fn is None:
            raise TypeError("len_fn is required.")
        if collate_fn is None:
            raise TypeError("collate_fn is required.")

        self.source_batches = source_batches
        self.dataloader = None
        self.len_fn = len_fn
        self.original_collate_fn = collate_fn
        self._budget_source = _BatchSizeSourceView(batch_size)
        self._init_common(
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
            max_batches=max_batches,
            spill_dir=spill_dir,
            log_dir=log_dir,
            distributed_dataloader=None,
        )

    def _length_record_iter(self) -> Iterator[list[LengthRecord]]:
        return iter_length_record_batches(self.source_batches, self.len_fn)


IterableLBA = IterableLengthBatchingAdapter
