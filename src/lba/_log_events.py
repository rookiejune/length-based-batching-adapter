"""Structured event payload helpers for LBA logging."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import PaddingStats, PlannerStats


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
