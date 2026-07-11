"""Batch planning and cache management for LBA."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Sequence
from heapq import merge
from pathlib import Path
from typing import Any, Literal, Optional, Union

from ._candidate_index import BatchCandidate, CandidateIndex
from ._candidate_search import (
    CandidateSearchResult,
    find_best_candidate,
    find_threshold_candidate,
)
from .metrics import PlannerStats
from .spill import SpillStore
from ._records import BatchPlan, PlanReason, SampleRecord


def _length_sort_key(record: SampleRecord) -> tuple[int, int]:
    return (record.length, record.arrival_id)


class BatchPlanner:
    """Plan dynamic batches from length-sorted sample records."""

    def __init__(
        self,
        max_padded_length: int,
        *,
        max_cache_samples: int = 8192,
        max_padding_ratio: float = 0.05,
        max_candidate_windows: Optional[int] = None,
        limited_search_fallback_after: Optional[int] = None,
        limited_search_fallback_pool_size: Optional[int] = None,
        spill_dir: Optional[Union[str, Path]] = None,
        logger: Optional[Any] = None,
        event_writer: Optional[Any] = None,
    ) -> None:
        if max_padded_length <= 0:
            raise ValueError("max_padded_length must be a positive integer.")
        if max_cache_samples <= 0:
            raise ValueError("max_cache_samples must be a positive integer.")
        if not 0 <= max_padding_ratio <= 1:
            raise ValueError("max_padding_ratio must be between 0 and 1.")
        if max_candidate_windows is not None and max_candidate_windows <= 0:
            raise ValueError("max_candidate_windows must be a positive integer.")
        if (
            limited_search_fallback_after is not None
            and limited_search_fallback_after <= 0
        ):
            raise ValueError(
                "limited_search_fallback_after must be a positive integer."
            )
        if (
            limited_search_fallback_pool_size is not None
            and limited_search_fallback_pool_size <= 0
        ):
            raise ValueError(
                "limited_search_fallback_pool_size must be a positive integer."
            )

        self.max_padded_length = max_padded_length
        self.max_cache_samples = max_cache_samples
        self.max_padding_ratio = max_padding_ratio
        self.max_candidate_windows = max_candidate_windows
        self.limited_search_fallback_after = limited_search_fallback_after
        self.limited_search_fallback_pool_size = limited_search_fallback_pool_size
        self.spill_store = SpillStore(spill_dir)
        self.logger = logger
        self.event_writer = event_writer
        self.stats = PlannerStats()

        self._sorted_records: list[SampleRecord] = []
        self._candidate_index: Optional[CandidateIndex] = None
        self._candidate_indexes_need_refresh = False
        self._recent_arrival_ids: set[int] = set()
        self._limited_search_miss_count = 0

    def add_records(self, records: Iterable[SampleRecord], *, allow_spill: bool = True) -> None:
        new_records = list(records)
        if not new_records:
            return

        sort_started_at = time.perf_counter()
        sorted_new_records = sorted(new_records, key=_length_sort_key)
        self._sorted_records = list(
            merge(self._sorted_records, sorted_new_records, key=_length_sort_key)
        )
        self.stats.record_sort(
            sorted_record_count=len(self._sorted_records),
            elapsed_seconds=time.perf_counter() - sort_started_at,
        )
        self._candidate_indexes_need_refresh = True
        self._recent_arrival_ids = {record.arrival_id for record in new_records}

        if allow_spill:
            self._spill_overflow()

    def pop_ready(self, *, flush: bool = False) -> Optional[BatchPlan]:
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
                return self._remove_records([oversized], reason=PlanReason.OVERSIZED)

            threshold_result = self._find_threshold_candidate(ignore_recent=flush)
            inspected_count += threshold_result.inspected_count
            if threshold_result.candidate is not None:
                source = "flush_search" if flush else "fast_path"
                return self._remove_candidate(
                    threshold_result.candidate,
                    reason=PlanReason.PLANNED,
                )

            if self._should_defer_limited_search_miss(flush=flush):
                self._limited_search_miss_count += 1
                return None

            best_result = self._find_best_candidate()
            inspected_count += best_result.inspected_count
            if best_result.candidate is None:
                return None

            source = "flush_search" if flush else "full_search"
            return self._remove_candidate(best_result.candidate, reason=PlanReason.PLANNED)
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

        for shard in self.spill_store.drain_shards():
            self.add_records(shard, allow_spill=False)
            while self._sorted_records:
                plan = self.pop_ready(flush=True)
                if plan is None:
                    break
                yield plan

    def drain_records(self) -> list[SampleRecord]:
        records = list(self._sorted_records)
        self._sorted_records = []
        self._candidate_index = None
        self._candidate_indexes_need_refresh = False
        self._recent_arrival_ids.clear()
        self._limited_search_miss_count = 0

        for shard in self.spill_store.drain_shards():
            records.extend(shard)

        return sorted(records, key=lambda record: record.arrival_id)

    def close(self) -> None:
        self.spill_store.cleanup()

    def _find_oversized(self) -> Optional[SampleRecord]:
        if self._sorted_records[-1].length <= self.max_padded_length:
            return None
        for record in self._sorted_records:
            if record.length > self.max_padded_length:
                return record
        return None

    def _find_threshold_candidate(
        self, *, ignore_recent: bool
    ) -> CandidateSearchResult:
        if (
            self._uses_limited_search(flush=ignore_recent)
            and not self._recent_arrival_ids
        ):
            return CandidateSearchResult(None, 0)

        index = self._ensure_candidate_index()
        recent_arrival_ids = frozenset() if ignore_recent else self._recent_arrival_ids
        return find_threshold_candidate(
            index,
            max_padded_length=self.max_padded_length,
            max_padding_ratio=self.max_padding_ratio,
            recent_arrival_ids=recent_arrival_ids,
            max_candidate_windows=self._threshold_candidate_window_limit(
                ignore_recent=ignore_recent,
            ),
        )

    def _find_best_candidate(self) -> CandidateSearchResult:
        index = self._ensure_candidate_index()
        return find_best_candidate(
            index,
            max_padded_length=self.max_padded_length,
            max_padding_ratio=self.max_padding_ratio,
        )

    def _uses_limited_search(self, *, flush: bool) -> bool:
        return not flush and self.max_candidate_windows is not None

    def _threshold_candidate_window_limit(
        self, *, ignore_recent: bool
    ) -> Optional[int]:
        if ignore_recent:
            return None
        if self._fallback_pool_limit_reached():
            return None
        return self.max_candidate_windows

    def _should_defer_limited_search_miss(self, *, flush: bool) -> bool:
        if not self._uses_limited_search(flush=flush):
            return False
        if self._fallback_after_limit_reached():
            return False
        if self._fallback_pool_limit_reached():
            return False
        return True

    def _fallback_after_limit_reached(self) -> bool:
        if self.limited_search_fallback_after is None:
            return False
        return self._limited_search_miss_count + 1 >= self.limited_search_fallback_after

    def _fallback_pool_limit_reached(self) -> bool:
        if self.limited_search_fallback_pool_size is None:
            return False
        return len(self._sorted_records) >= self.limited_search_fallback_pool_size

    def _remove_candidate(
        self, candidate: BatchCandidate, *, reason: PlanReason
    ) -> BatchPlan:
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
        reason: PlanReason,
        raw_length_sum: Optional[int] = None,
        padded_length: Optional[int] = None,
        padding_length: Optional[int] = None,
        padding_ratio: Optional[float] = None,
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
        self._candidate_indexes_need_refresh = True
        self._recent_arrival_ids.difference_update(record_ids_to_remove)
        self._limited_search_miss_count = 0

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
        self._candidate_indexes_need_refresh = True
        self._recent_arrival_ids.difference_update(spilled_arrival_ids)
        if self.logger is not None:
            self.logger.warning(
                "lba health: spilled records=%s cache_limit=%s spill_dir=%s "
                "action=increase max_cache_samples or set a faster spill_dir",
                len(spill_records),
                self.max_cache_samples,
                self.spill_store.root,
            )
        if self.event_writer is not None:
            self.event_writer.write(
                "spill",
                {
                    "records": len(spill_records),
                    "max_cache_samples": self.max_cache_samples,
                    "spill_dir": str(self.spill_store.root),
                },
            )

    def _ensure_candidate_index(self) -> CandidateIndex:
        if not self._candidate_indexes_need_refresh and self._candidate_index is not None:
            return self._candidate_index

        self._candidate_index = CandidateIndex.from_records(self._sorted_records)
        self._candidate_indexes_need_refresh = False
        return self._candidate_index
