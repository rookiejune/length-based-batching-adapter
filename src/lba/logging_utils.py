"""Logging helpers for LBA."""

from __future__ import annotations

import json
import logging
import os
import warnings
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from time import strftime
from typing import Any

from .metrics import PaddingStats, PlannerStats, padding_ratio_reduction
from .types import SampleRecord


def default_log_dir(cwd: Path | None = None) -> Path:
    """Return the default LBA log directory."""

    if cwd is not None:
        return cwd / ".lba" / "logs"
    return Path.home() / ".lba" / "logs"


def create_run_logger(log_dir: str | Path | None = None) -> tuple[logging.Logger, Path]:
    """Create a per-run file logger."""

    directory = Path(log_dir) if log_dir is not None else default_log_dir()
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / f"lba-{strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.log"

    logger = logging.getLogger(f"lba.{id(log_path)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger, log_path


def event_log_path_for(log_path: Path) -> Path:
    """Return the structured-event path next to a human log file."""

    return log_path.with_suffix(".jsonl")


class JsonlEventWriter:
    """Write structured LBA events as one JSON object per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, fields: Mapping[str, Any]) -> None:
        payload = {
            "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as file:
            json.dump(payload, file, sort_keys=True)
            file.write("\n")


class RunReporter:
    """Write human-readable and structured events for one adapter run."""

    def __init__(
        self,
        logger: logging.Logger,
        event_writer: JsonlEventWriter,
        event_log_path: Path,
    ) -> None:
        self.logger = logger
        self.event_writer = event_writer
        self.event_log_path = event_log_path

    def warn_oversized_sample(
        self,
        record: SampleRecord,
        *,
        max_padded_length: int | None,
    ) -> None:
        sample_type = type(record.sample).__name__
        warnings.warn(
            f"LBA oversized sample length={record.length} "
            f"max_padded_length={max_padded_length} "
            "was emitted as a singleton batch.",
            stacklevel=2,
        )
        self.logger.warning(
            "lba health: oversized sample length=%s budget=%s index=%s "
            "sample_type=%s action=emitted_singleton",
            record.length,
            max_padded_length,
            _format_optional(record.index),
            sample_type,
        )
        self.event_writer.write(
            "oversized_sample",
            {
                "length": record.length,
                "max_padded_length": max_padded_length,
                "index": record.index,
                "sample_type": sample_type,
            },
        )

    def log_summary(
        self,
        before: PaddingStats,
        after: PaddingStats,
        planner: PlannerStats,
        *,
        max_padded_length: int | None,
    ) -> None:
        reduction = padding_ratio_reduction(before, after)
        saved_padding_length = before.padding_length_sum - after.padding_length_sum
        self.logger.info(
            "lba summary: padding %s -> %s (%s reduction) saved_padding=%s "
            "batches=%s->%s samples=%s",
            _format_percent_value(before.global_padding_ratio),
            _format_percent_value(after.global_padding_ratio),
            _format_percent_value(reduction),
            _format_signed_int(saved_padding_length),
            before.batch_count,
            after.batch_count,
            after.sample_count,
        )
        self.logger.info(
            "lba planner: total=%s pop_ready_avg=%sms sort_avg=%sms "
            "paths=fast:%s/full:%s/flush:%s max_cache=%s",
            _format_seconds(planner.planner_time_seconds),
            _format_milliseconds(planner.average_pop_ready_time_ms),
            _format_milliseconds(planner.average_sort_time_ms),
            planner.fast_path_batch_count,
            planner.full_search_batch_count,
            planner.flush_search_batch_count,
            planner.max_cache_size_seen,
        )
        self.logger.info(
            "lba health: oversized=%s spill_events=%s spilled_records=%s "
            "no_ready=%s other_batches=%s event_log=%s",
            after.oversized_batch_count,
            planner.spill_event_count,
            planner.spilled_record_count,
            planner.no_ready_call_count,
            after.other_batch_count,
            self.event_log_path,
        )
        self.event_writer.write(
            "summary",
            {
                "max_padded_length": max_padded_length,
                "padding": {
                    "before": padding_event_fields(before),
                    "after": padding_event_fields(after),
                    "padding_ratio_reduction": reduction,
                    "saved_padding_length": saved_padding_length,
                    "saved_padded_length": (
                        before.padded_length_sum - after.padded_length_sum
                    ),
                },
                "planner": planner_event_fields(planner),
                "health": {
                    "oversized_batches": after.oversized_batch_count,
                    "planner_oversized_batches": planner.oversized_batch_count,
                    "other_batches": after.other_batch_count,
                    "spill_events": planner.spill_event_count,
                    "spilled_records": planner.spilled_record_count,
                    "no_ready_calls": planner.no_ready_call_count,
                },
            },
        )


def padding_event_fields(stats: PaddingStats) -> dict[str, object]:
    """Return structured event fields for padding metrics."""

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


def planner_event_fields(stats: PlannerStats) -> dict[str, object]:
    """Return structured event fields for planner metrics."""

    return {
        "planner_time_seconds": stats.planner_time_seconds,
        "sort_time_seconds": stats.sort_time_seconds,
        "sort_calls": stats.sort_call_count,
        "average_sort_time_ms": stats.average_sort_time_ms,
        "pop_ready_time_seconds": stats.pop_ready_time_seconds,
        "pop_ready_calls": stats.pop_ready_call_count,
        "average_pop_ready_time_ms": stats.average_pop_ready_time_ms,
        "candidate_window_checks": stats.candidate_window_checks,
        "average_candidate_window_checks": stats.average_candidate_window_checks,
        "max_candidate_window_checks": stats.max_candidate_window_checks,
        "records_sorted_total": stats.records_sorted_total,
        "max_cache_size_seen": stats.max_cache_size_seen,
        "spill_events": stats.spill_event_count,
        "spilled_records": stats.spilled_record_count,
        "paths": {
            "fast_path": {
                "batches": stats.fast_path_batch_count,
                "time_seconds": stats.fast_path_time_seconds,
                "candidate_window_checks": stats.fast_path_candidate_window_checks,
            },
            "full_search": {
                "batches": stats.full_search_batch_count,
                "time_seconds": stats.full_search_time_seconds,
                "candidate_window_checks": stats.full_search_candidate_window_checks,
            },
            "flush_search": {
                "batches": stats.flush_search_batch_count,
                "time_seconds": stats.flush_search_time_seconds,
                "candidate_window_checks": stats.flush_search_candidate_window_checks,
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


def _format_percent_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _format_seconds(value: float) -> str:
    if value < 1:
        return f"{value * 1000:.3f}ms"
    return f"{value:.3f}s"


def _format_signed_int(value: int) -> str:
    if value > 0:
        return f"+{value}"
    return str(value)


def _format_optional(value: object | None) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _format_milliseconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"
