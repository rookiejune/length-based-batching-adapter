"""Adapter run logging setup and configuration event fields."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional, Union

from .config import LBAConfig
from ._log_events import JsonlEventWriter
from ._log_files import create_run_logger, event_log_path_for
from ._run_reporter import RunReporter


class AdapterRunLogger:
    """Logging objects and run-start events for one adapter instance."""

    def __init__(
        self,
        *,
        config: LBAConfig,
        max_batches: Optional[int],
        log_dir: Optional[Union[str, Path]],
    ) -> None:
        self.logger, self.log_path = create_run_logger(log_dir)
        self.log_event_path = event_log_path_for(self.log_path)
        self.event_writer = JsonlEventWriter(self.log_event_path)
        self.reporter = RunReporter(
            self.logger,
            self.event_writer,
            self.log_event_path,
        )

        warnings.warn(
            f"LBA log file: {self.log_path}; structured events: {self.log_event_path}",
            stacklevel=3,
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
                "config": config_event_fields(config, max_batches=max_batches),
            },
        )
        if config.max_padded_length is not None:
            warnings.warn(
                "max_padded_length is set explicitly and overrides warmup inference.",
                stacklevel=3,
            )
            self.logger.warning(
                "lba config: explicit max_padded_length=%s overrides warmup inference",
                config.max_padded_length,
            )
            self.event_writer.write(
                "config_warning",
                {
                    "reason": "explicit_max_padded_length",
                    "max_padded_length": config.max_padded_length,
                },
            )


def config_event_fields(
    config: LBAConfig,
    *,
    max_batches: Optional[int],
) -> dict[str, object]:
    return {
        "max_padded_length": config.max_padded_length,
        "warmup_batches": config.warmup_batches,
        "max_cache_samples": config.max_cache_samples,
        "max_padding_ratio": config.max_padding_ratio,
        "prefetch_batches": config.prefetch_batches,
        "planner_mode": config.planner_mode,
        "max_candidate_windows": config.max_candidate_windows,
        "candidate_window_limit": config.candidate_window_limit,
        "limited_search_fallback_after": config.limited_search_fallback_after,
        "limited_search_fallback_after_limit": (
            config.limited_search_fallback_after_limit
        ),
        "limited_search_fallback_pool_size": config.limited_search_fallback_pool_size,
        "limited_search_fallback_pool_limit": config.limited_search_fallback_pool_limit,
        "drop_last_flush": config.drop_last_flush,
        "max_batches": max_batches,
        "spill_dir": path_or_none(config.spill_dir),
        "log_dir": path_or_none(config.log_dir),
    }


def path_or_none(value: Optional[Union[str, Path]]) -> Optional[str]:
    if value is None:
        return None
    return str(value)
