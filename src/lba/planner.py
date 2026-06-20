"""Batch planning and cache management for LBA."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

from .candidates import (
    BatchCandidate,
    CandidateSearchResult,
    find_best_candidate,
    find_threshold_candidate,
)
from .metrics import PlannerStats
from .spill import SpillStore
from .types import BatchPlan, SampleRecord


class BatchPlanner:
    """Plan dynamic batches from length-sorted sample records."""

    def __init__(
        self,
        max_padded_length: int,
        *,
        max_cache_samples: int = 8192,
        max_padding_ratio: float = 0.05,
        spill_dir: str | Path | None = None,
        logger: Any | None = None,
    ) -> None:
        if max_padded_length <= 0:
            raise ValueError("max_padded_length must be a positive integer.")
        if max_cache_samples <= 0:
            raise ValueError("max_cache_samples must be a positive integer.")
        if not 0 <= max_padding_ratio <= 1:
            raise ValueError("max_padding_ratio must be between 0 and 1.")

        self.max_padded_length = max_padded_length
        self.max_cache_samples = max_cache_samples
        self.max_padding_ratio = max_padding_ratio
        self.spill_store = SpillStore(spill_dir)
        self.logger = logger
        self.stats = PlannerStats()

        self._sorted_records: list[SampleRecord] = []
        self._prefix_lengths: list[int] = [0]
        self._prefix_needs_refresh = False
        self._recent_arrival_ids: set[int] = set()

    def add_records(self, records: Iterable[SampleRecord], *, allow_spill: bool = True) -> None:
        new_records = list(records)
        if not new_records:
            return

        self._sorted_records.extend(new_records)
        sort_started_at = time.perf_counter()
        self._sorted_records.sort(key=lambda record: (record.length, record.arrival_id))
        self.stats.record_sort(
            sorted_record_count=len(self._sorted_records),
            elapsed_seconds=time.perf_counter() - sort_started_at,
        )
        self._prefix_needs_refresh = True
        self._recent_arrival_ids = {record.arrival_id for record in new_records}

        if allow_spill:
            self._spill_overflow()

    def pop_ready(self, *, flush: bool = False) -> BatchPlan | None:
        started_at = time.perf_counter()
        inspected_count = 0
        source: Literal[
            "fast_path",
            "full_search",
            "flush_search",
            "oversized",
            "no_ready",
        ] = "no_ready"

        try:
            if not self._sorted_records:
                return None

            oversized = self._find_oversized()
            if oversized is not None:
                source = "oversized"
                return self._remove_records([oversized], reason="oversized")

            threshold_result = self._find_threshold_candidate(ignore_recent=flush)
            inspected_count += threshold_result.inspected_count
            if threshold_result.candidate is not None:
                source = "flush_search" if flush else "fast_path"
                return self._remove_candidate(
                    threshold_result.candidate,
                    reason="planned",
                )

            best_result = self._find_best_candidate()
            inspected_count += best_result.inspected_count
            if best_result.candidate is None:
                return None

            source = "flush_search" if flush else "full_search"
            return self._remove_candidate(best_result.candidate, reason="planned")
        finally:
            self.stats.record_pop_ready(
                elapsed_seconds=time.perf_counter() - started_at,
                candidate_window_checks=inspected_count,
                source=source,
            )

    def flush(self) -> Iterator[BatchPlan]:
        while self._sorted_records:
            plan = self.pop_ready(flush=True)
            if plan is None:
                break
            yield plan

        for shard in self.spill_store.read_shards():
            self.add_records(shard, allow_spill=False)
            while self._sorted_records:
                plan = self.pop_ready(flush=True)
                if plan is None:
                    break
                yield plan

    def drain_records(self) -> list[SampleRecord]:
        records = list(self._sorted_records)
        self._sorted_records = []
        self._prefix_lengths = [0]
        self._prefix_needs_refresh = False
        self._recent_arrival_ids.clear()

        for shard in self.spill_store.drain_shards():
            records.extend(shard)

        return sorted(records, key=lambda record: record.arrival_id)

    def close(self) -> None:
        self.spill_store.cleanup()

    def _find_oversized(self) -> SampleRecord | None:
        if self._sorted_records[-1].length <= self.max_padded_length:
            return None
        for record in self._sorted_records:
            if record.length > self.max_padded_length:
                return record
        return None

    def _find_threshold_candidate(
        self, *, ignore_recent: bool
    ) -> CandidateSearchResult:
        self._ensure_prefix_lengths()
        recent_arrival_ids = frozenset() if ignore_recent else self._recent_arrival_ids
        return find_threshold_candidate(
            self._sorted_records,
            self._prefix_lengths,
            max_padded_length=self.max_padded_length,
            max_padding_ratio=self.max_padding_ratio,
            recent_arrival_ids=recent_arrival_ids,
        )

    def _find_best_candidate(self) -> CandidateSearchResult:
        self._ensure_prefix_lengths()
        return find_best_candidate(
            self._sorted_records,
            self._prefix_lengths,
            max_padded_length=self.max_padded_length,
            max_padding_ratio=self.max_padding_ratio,
        )

    def _remove_candidate(self, candidate: BatchCandidate, *, reason: str) -> BatchPlan:
        records = list(
            self._sorted_records[candidate.start_index : candidate.end_index + 1]
        )
        return self._remove_records(
            records,
            reason=reason,
            raw_length_sum=candidate.total_raw_length,
            padded_length=candidate.total_padded_length,
            padding_length=candidate.total_padding_length,
            padding_ratio=candidate.padding_ratio,
        )

    def _remove_records(
        self,
        records: Sequence[SampleRecord],
        *,
        reason: str,
        raw_length_sum: int | None = None,
        padded_length: int | None = None,
        padding_length: int | None = None,
        padding_ratio: float | None = None,
    ) -> BatchPlan:
        record_ids_to_remove = {record.arrival_id for record in records}
        arrival_ordered_records = tuple(
            sorted(records, key=lambda record: record.arrival_id)
        )
        self._sorted_records = [
            record
            for record in self._sorted_records
            if record.arrival_id not in record_ids_to_remove
        ]
        self._prefix_needs_refresh = True
        self._recent_arrival_ids.difference_update(record_ids_to_remove)

        if raw_length_sum is None:
            raw_length_sum = sum(record.length for record in arrival_ordered_records)
        if padded_length is None:
            max_length = max(record.length for record in arrival_ordered_records)
            padded_length = max_length * len(arrival_ordered_records)
        if padding_length is None:
            padding_length = padded_length - raw_length_sum
        if padding_ratio is None:
            padding_ratio = padding_length / padded_length if padded_length else 0.0

        return BatchPlan(
            records=arrival_ordered_records,
            raw_length_sum=raw_length_sum,
            padded_length=padded_length,
            padding_length=padding_length,
            padding_ratio=padding_ratio,
            reason=reason,
        )

    def _spill_overflow(self) -> None:
        overflow_count = len(self._sorted_records) - self.max_cache_samples
        if overflow_count <= 0:
            return

        records_by_arrival = sorted(
            self._sorted_records,
            key=lambda record: record.arrival_id,
        )
        spill_records = records_by_arrival[:overflow_count]
        spilled_arrival_ids = {record.arrival_id for record in spill_records}
        self.spill_store.write(spill_records)
        self.stats.record_spill(len(spill_records))
        self._sorted_records = [
            record
            for record in self._sorted_records
            if record.arrival_id not in spilled_arrival_ids
        ]
        self._prefix_needs_refresh = True
        self._recent_arrival_ids.difference_update(spilled_arrival_ids)
        if self.logger is not None:
            self.logger.info("spilled %s records to disk", len(spill_records))

    def _ensure_prefix_lengths(self) -> None:
        if not self._prefix_needs_refresh:
            return
        prefix_lengths = [0]
        for record in self._sorted_records:
            prefix_lengths.append(prefix_lengths[-1] + record.length)
        self._prefix_lengths = prefix_lengths
        self._prefix_needs_refresh = False
