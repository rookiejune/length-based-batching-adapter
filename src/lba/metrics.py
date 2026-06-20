"""Runtime metrics collected while LBA plans batches."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from .types import BatchPlan, LengthRecord


@dataclass
class PaddingStats:
    """Aggregate padding ratios across a sequence of batches."""

    # Number of non-empty batches included in this aggregate.
    batch_count: int = 0

    # Number of raw samples represented by those batches.
    sample_count: int = 0

    # Sum of effective sample lengths before padding.
    raw_length_sum: int = 0

    # Sum of padded token slots, computed as max_length * batch_size per batch.
    padded_length_sum: int = 0

    # Sum of padded token slots that do not correspond to raw sample content.
    padding_length_sum: int = 0

    # Sum of per-batch padding ratios, used for the arithmetic mean.
    batch_padding_ratio_sum: float = 0.0

    # Number of emitted LBA batches created by the normal planner path.
    planned_batch_count: int = 0

    # Number of singleton batches emitted because one sample exceeded the budget.
    oversized_batch_count: int = 0

    # Catch-all for future or custom plan reasons.
    other_batch_count: int = 0

    def add_lengths(self, lengths: Iterable[int]) -> None:
        batch_lengths = list(lengths)
        if not batch_lengths:
            return

        raw_length_sum = sum(batch_lengths)
        padded_length = max(batch_lengths) * len(batch_lengths)
        padding_length = padded_length - raw_length_sum
        padding_ratio = padding_length / padded_length if padded_length else 0.0
        self.add_batch(
            sample_count=len(batch_lengths),
            raw_length_sum=raw_length_sum,
            padded_length=padded_length,
            padding_length=padding_length,
            padding_ratio=padding_ratio,
        )

    def add_length_records(self, records: Iterable[LengthRecord]) -> None:
        self.add_lengths(record.length for record in records)

    def add_plan(self, plan: BatchPlan) -> None:
        self.add_batch(
            sample_count=len(plan.records),
            raw_length_sum=plan.raw_length_sum,
            padded_length=plan.padded_length,
            padding_length=plan.padding_length,
            padding_ratio=plan.padding_ratio,
        )
        if plan.reason == "planned":
            self.planned_batch_count += 1
        elif plan.reason == "oversized":
            self.oversized_batch_count += 1
        else:
            self.other_batch_count += 1

    def add_batch(
        self,
        *,
        sample_count: int,
        raw_length_sum: int,
        padded_length: int,
        padding_length: int,
        padding_ratio: float,
    ) -> None:
        self.batch_count += 1
        self.sample_count += sample_count
        self.raw_length_sum += raw_length_sum
        self.padded_length_sum += padded_length
        self.padding_length_sum += padding_length
        self.batch_padding_ratio_sum += padding_ratio

    @property
    def global_padding_ratio(self) -> float | None:
        if self.padded_length_sum <= 0:
            return None
        return self.padding_length_sum / self.padded_length_sum

    @property
    def mean_batch_padding_ratio(self) -> float | None:
        if self.batch_count <= 0:
            return None
        return self.batch_padding_ratio_sum / self.batch_count


@dataclass
class PlannerStats:
    """Timing and spill counters owned by the planner."""

    sort_call_count: int = 0
    records_sorted_total: int = 0
    sort_time_seconds: float = 0.0
    max_cache_size_seen: int = 0
    spill_event_count: int = 0
    spilled_record_count: int = 0
    pop_ready_call_count: int = 0
    pop_ready_time_seconds: float = 0.0
    candidate_window_checks: int = 0
    max_candidate_window_checks: int = 0
    fast_path_batch_count: int = 0
    full_search_batch_count: int = 0
    flush_search_batch_count: int = 0
    oversized_batch_count: int = 0
    no_ready_call_count: int = 0

    def record_sort(self, *, sorted_record_count: int, elapsed_seconds: float) -> None:
        self.sort_call_count += 1
        self.records_sorted_total += sorted_record_count
        self.sort_time_seconds += elapsed_seconds
        self.max_cache_size_seen = max(self.max_cache_size_seen, sorted_record_count)

    def record_spill(self, spilled_record_count: int) -> None:
        self.spill_event_count += 1
        self.spilled_record_count += spilled_record_count

    def record_pop_ready(
        self,
        *,
        elapsed_seconds: float,
        candidate_window_checks: int,
        source: Literal[
            "fast_path",
            "full_search",
            "flush_search",
            "oversized",
            "no_ready",
        ],
    ) -> None:
        self.pop_ready_call_count += 1
        self.pop_ready_time_seconds += elapsed_seconds
        self.candidate_window_checks += candidate_window_checks
        self.max_candidate_window_checks = max(
            self.max_candidate_window_checks,
            candidate_window_checks,
        )

        if source == "fast_path":
            self.fast_path_batch_count += 1
        elif source == "full_search":
            self.full_search_batch_count += 1
        elif source == "flush_search":
            self.flush_search_batch_count += 1
        elif source == "oversized":
            self.oversized_batch_count += 1
        elif source == "no_ready":
            self.no_ready_call_count += 1
        else:
            raise ValueError(f"Unknown pop_ready source: {source}")

    @property
    def average_sort_time_ms(self) -> float | None:
        if self.sort_call_count <= 0:
            return None
        return self.sort_time_seconds * 1000 / self.sort_call_count

    @property
    def average_pop_ready_time_ms(self) -> float | None:
        if self.pop_ready_call_count <= 0:
            return None
        return self.pop_ready_time_seconds * 1000 / self.pop_ready_call_count

    @property
    def average_candidate_window_checks(self) -> float | None:
        if self.pop_ready_call_count <= 0:
            return None
        return self.candidate_window_checks / self.pop_ready_call_count


def padding_ratio_reduction(before: PaddingStats, after: PaddingStats) -> float | None:
    before_ratio = before.global_padding_ratio
    after_ratio = after.global_padding_ratio
    if before_ratio is None or after_ratio is None or before_ratio <= 0:
        return None
    return (before_ratio - after_ratio) / before_ratio
