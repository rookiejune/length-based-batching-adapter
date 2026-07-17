"""Public adapters for length-based batching."""

from __future__ import annotations

from collections.abc import Generator, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Optional, Union

from torch.utils.data import DataLoader

from ._adapter_logging import AdapterRunLogger
from ._api_types import CollateFn, LengthFn
from ._iteration import Iteration
from ._records import LengthRecord
from .budget import BatchSizeSource
from .config import DEFAULT_PREFETCH_BATCHES, LBAConfig, PlannerMode
from .distributed import DistributedBatchCoordinator
from .metrics import PlannerStats
from .prefetch import prefetch_iterator
from .source import build_source_loader, iter_length_record_batches


class _BatchSizeSource:
    def __init__(self, batch_size: Optional[int]) -> None:
        self.batch_size = batch_size


class _Adapter:
    def __init__(
        self,
        *,
        collate_fn: CollateFn,
        budget_source: BatchSizeSource,
        distributed_dataloader: Optional[DataLoader],
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
    ) -> None:
        if max_batches is not None and max_batches < 0:
            raise ValueError("max_batches must be non-negative.")

        self.collate_fn = collate_fn
        self._budget_source = budget_source
        self._max_batches = max_batches
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
        self.last_planner_stats = PlannerStats()

    @property
    def max_padded_length(self) -> Optional[int]:
        return self.config.max_padded_length

    def __iter__(self) -> Iterator[Any]:
        distributed = DistributedBatchCoordinator.is_initialized()
        iterator = self._run(distributed=distributed)

        if distributed:
            if self.config.prefetch_batches > 0:
                self.logger.info(
                    "disabled LBA prefetch for torch.distributed iteration"
                )
            return iterator
        if self.config.prefetch_batches > 0:
            return prefetch_iterator(iterator, self.config.prefetch_batches)
        return iterator

    def _run(
        self,
        *,
        distributed: bool,
    ) -> Generator[Any, None, None]:
        if self._max_batches == 0:
            return

        iteration = Iteration(
            self.config,
            self._records(),
            self.collate_fn,
            self._budget_source,
            self._distributed,
            self.reporter,
            self.logger,
            self.event_writer,
            max_batches=self._max_batches,
        )
        try:
            yield from iteration.run(distributed=distributed)
        finally:
            self.last_planner_stats = iteration.planner_stats

    def _records(self) -> Iterator[list[LengthRecord]]:
        raise NotImplementedError


class LengthBatchingAdapter(_Adapter):
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
        super().__init__(
            collate_fn=dataloader.collate_fn,
            budget_source=dataloader,
            distributed_dataloader=dataloader,
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
        )

    def _records(self) -> Iterator[list[LengthRecord]]:
        return iter(build_source_loader(self.dataloader, self.len_fn))


LBA = LengthBatchingAdapter


class IterableLengthBatchingAdapter(_Adapter):
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
        super().__init__(
            collate_fn=collate_fn,
            budget_source=_BatchSizeSource(batch_size),
            distributed_dataloader=None,
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
        )

    def _records(self) -> Iterator[list[LengthRecord]]:
        return iter_length_record_batches(self.source_batches, self.len_fn)


IterableLBA = IterableLengthBatchingAdapter
