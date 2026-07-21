"""Public length-based batching dataloader."""

from __future__ import annotations

import logging
import weakref
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any, Optional, Union

from torch.utils.data import DataLoader, Dataset

from ._adapter_logging import AdapterRunLogger
from ._api_types import EventWriter, LengthFn
from ._iteration import Iteration
from ._records import LengthRecord
from ._run_reporter import RunReporter
from .config import DEFAULT_PREFETCH_BATCHES, LBAConfig, PlannerMode
from .distributed import DistributedBatchCoordinator
from .metrics import PlannerStats
from .prefetch import prefetch_iterator
from .source import build_source_loader


class LBA(DataLoader[Any]):
    """Load samples and emit length-based dynamic batches."""

    len_fn: LengthFn
    max_padded_length: Optional[int]
    warmup_batches: Optional[int]
    max_cache_samples: int
    max_padding_ratio: float
    prefetch_batches: int
    planner_mode: PlannerMode
    max_candidate_windows: Optional[int]
    limited_search_fallback_after: Optional[int]
    limited_search_fallback_pool_size: Optional[int]
    drop_last_flush: bool
    max_batches: Optional[int]
    spill_dir: Optional[Union[str, Path]]
    log_dir: Optional[Union[str, Path]]
    config: LBAConfig
    last_planner_stats: PlannerStats
    last_max_padded_length: Optional[int]

    def __init__(
        self,
        dataset: Dataset[Any],
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
        **dataloader_kwargs: Any,
    ) -> None:
        if len_fn is None:
            raise TypeError("len_fn is required.")
        if isinstance(dataset, DataLoader):
            raise TypeError(
                "LBA expects a dataset; pass DataLoader options directly to LBA."
            )
        if max_batches is not None and max_batches < 0:
            raise ValueError("max_batches must be non-negative.")

        config = LBAConfig(
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
        super().__init__(dataset, **dataloader_kwargs)

        self.len_fn = len_fn
        # Lightning reconstructs DataLoader subclasses from public attributes.
        self.max_padded_length = max_padded_length
        self.warmup_batches = warmup_batches
        self.max_cache_samples = max_cache_samples
        self.max_padding_ratio = max_padding_ratio
        self.prefetch_batches = prefetch_batches
        self.planner_mode = planner_mode
        self.max_candidate_windows = max_candidate_windows
        self.limited_search_fallback_after = limited_search_fallback_after
        self.limited_search_fallback_pool_size = limited_search_fallback_pool_size
        self.drop_last_flush = drop_last_flush
        self.max_batches = max_batches
        self.spill_dir = spill_dir
        self.log_dir = log_dir

        self.config = config
        self.last_planner_stats = PlannerStats()
        self.last_max_padded_length = None
        self._source_loader: Optional[DataLoader[Any]] = None
        self._run_logger: Optional[AdapterRunLogger] = None
        self._logger_finalizer: Optional[weakref.finalize[Any]] = None
        self._distributed: Optional[DistributedBatchCoordinator] = None

    def __len__(self) -> int:
        raise TypeError("LBA output batch count is dynamic and unavailable.")

    def __iter__(self) -> Iterator[Any]:
        run_logger = self._ensure_run_logger()
        distributed = DistributedBatchCoordinator.is_initialized()
        if self.max_batches == 0:
            return iter(())

        records = self._records()
        iterator = self._run(records, run_logger=run_logger, distributed=distributed)

        if self.config.prefetch_batches > 0:
            if distributed:
                self._ensure_distributed(
                    run_logger,
                ).prepare_for_background_iteration()
            return prefetch_iterator(iterator, self.config.prefetch_batches)
        return iterator

    @property
    def logger(self) -> logging.Logger:
        return self._ensure_run_logger().logger

    @property
    def log_path(self) -> Path:
        return self._ensure_run_logger().log_path

    @property
    def log_event_path(self) -> Path:
        return self._ensure_run_logger().log_event_path

    @property
    def event_writer(self) -> EventWriter:
        return self._ensure_run_logger().event_writer

    @property
    def reporter(self) -> RunReporter:
        return self._ensure_run_logger().reporter

    def _run(
        self,
        records: Iterator[list[LengthRecord]],
        *,
        run_logger: AdapterRunLogger,
        distributed: bool,
    ) -> Generator[Any, None, None]:
        iteration = Iteration(
            self.config,
            records,
            self.collate_fn,
            self,
            self._ensure_distributed(run_logger),
            run_logger.reporter,
            run_logger.logger,
            run_logger.event_writer,
            max_batches=self.max_batches,
            pin_memory=self.pin_memory,
            pin_memory_device=getattr(self, "pin_memory_device", "") or None,
        )
        try:
            yield from iteration.run(distributed=distributed)
        finally:
            self.last_planner_stats = iteration.planner_stats
            self.last_max_padded_length = iteration.max_padded_length

    def _records(self) -> Iterator[list[LengthRecord]]:
        if self._source_loader is None:
            self._source_loader = build_source_loader(self, self.len_fn)
        return iter(self._source_loader)

    def _ensure_run_logger(self) -> AdapterRunLogger:
        if self._run_logger is None:
            run_logger = AdapterRunLogger(
                config=self.config,
                max_batches=self.max_batches,
                log_dir=self.log_dir,
            )
            self._run_logger = run_logger
            self._logger_finalizer = weakref.finalize(self, run_logger.close)
        return self._run_logger

    def _ensure_distributed(
        self, run_logger: AdapterRunLogger
    ) -> DistributedBatchCoordinator:
        if self._distributed is None:
            self._distributed = DistributedBatchCoordinator(
                self,
                self.config,
                run_logger.logger,
                run_logger.event_writer,
            )
        return self._distributed


__all__ = ["LBA"]
