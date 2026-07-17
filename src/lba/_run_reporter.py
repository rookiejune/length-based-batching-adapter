"""Human-readable run reporting for LBA."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

from ._log_events import JsonlEventWriter, padding_event_fields, planner_event_fields
from .metrics import PaddingStats, PlannerStats, padding_ratio_reduction
from ._records import SampleRecord


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
        max_padded_length: Optional[int],
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
        max_padded_length: Optional[int],
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
            "paths=fast:%s/fallback:%s/flush:%s max_cache=%s",
            _format_seconds(planner.planner_time_seconds),
            _format_milliseconds(planner.average_pop_ready_time_ms),
            _format_milliseconds(planner.average_sort_time_ms),
            planner.fast_path_batch_count,
            planner.fallback_search_batch_count,
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


def _format_percent_value(value: Optional[float]) -> str:
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


def _format_optional(value: Optional[object]) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _format_milliseconds(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"
