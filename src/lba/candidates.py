"""Candidate batch selection helpers for LBA."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import ceil
from typing import AbstractSet

from .types import SampleRecord


class ArrivalIdRangeMin:
    """Range-min index for arrival ids in the current length-sorted pool."""

    def __init__(self, arrival_ids: Sequence[int]) -> None:
        if not arrival_ids:
            raise ValueError("arrival_ids must not be empty.")

        size = 1
        while size < len(arrival_ids):
            size *= 2

        empty_value = max(arrival_ids) + 1
        values = [empty_value] * (2 * size)
        values[size : size + len(arrival_ids)] = arrival_ids
        for index in range(size - 1, 0, -1):
            values[index] = min(values[index * 2], values[index * 2 + 1])

        self._length = len(arrival_ids)
        self._size = size
        self._values = values

    @classmethod
    def from_records(cls, records: Sequence[SampleRecord]) -> ArrivalIdRangeMin:
        return cls([record.arrival_id for record in records])

    def range_min(self, start_index: int, end_index: int) -> int:
        if (
            start_index < 0
            or end_index < start_index
            or end_index >= self._length
        ):
            raise ValueError("Invalid range-min query.")

        left = start_index + self._size
        right = end_index + self._size
        best = self._values[left]
        while left <= right:
            if left % 2 == 1:
                best = min(best, self._values[left])
                left += 1
            if right % 2 == 0:
                best = min(best, self._values[right])
                right -= 1
            left //= 2
            right //= 2
        return best


@dataclass(frozen=True)
class BatchCandidate:
    """A contiguous length-sorted window that can become a dynamic batch."""

    start_index: int
    end_index: int
    total_raw_length: int
    total_padded_length: int
    total_padding_length: int
    padding_ratio: float
    earliest_arrival_id: int

    @property
    def record_count(self) -> int:
        return self.end_index - self.start_index + 1


@dataclass(frozen=True)
class CandidateIndex:
    """Indexed view of the current length-sorted record pool."""

    records: Sequence[SampleRecord]
    prefix_lengths: Sequence[int]
    sorted_lengths: Sequence[int]
    arrival_id_range_min: ArrivalIdRangeMin | None

    @classmethod
    def from_records(cls, records: Sequence[SampleRecord]) -> CandidateIndex:
        prefix_lengths = [0]
        sorted_lengths: list[int] = []
        for record in records:
            sorted_lengths.append(record.length)
            prefix_lengths.append(prefix_lengths[-1] + record.length)
        return cls(
            records=records,
            prefix_lengths=prefix_lengths,
            sorted_lengths=sorted_lengths,
            arrival_id_range_min=(
                ArrivalIdRangeMin.from_records(records) if records else None
            ),
        )

    def recent_indices(self, recent_arrival_ids: AbstractSet[int]) -> list[int]:
        return [
            index
            for index, record in enumerate(self.records)
            if record.arrival_id in recent_arrival_ids
        ]

    def make_candidate(self, start_index: int, end_index: int) -> BatchCandidate:
        if self.arrival_id_range_min is None:
            raise RuntimeError("Candidate index has no records.")
        total_raw_length = self.prefix_lengths[end_index + 1] - self.prefix_lengths[
            start_index
        ]
        longest_length = self.records[end_index].length
        record_count = end_index - start_index + 1
        total_padded_length = longest_length * record_count
        total_padding_length = total_padded_length - total_raw_length
        padding_ratio = (
            total_padding_length / total_padded_length if total_padded_length else 0.0
        )
        return BatchCandidate(
            start_index=start_index,
            end_index=end_index,
            total_raw_length=total_raw_length,
            total_padded_length=total_padded_length,
            total_padding_length=total_padding_length,
            padding_ratio=padding_ratio,
            earliest_arrival_id=self.arrival_id_range_min.range_min(
                start_index,
                end_index,
            ),
        )


@dataclass(frozen=True)
class CandidateSearchResult:
    """Candidate search result plus the amount of window work performed."""

    candidate: BatchCandidate | None
    inspected_count: int


def find_threshold_candidate(
    index: CandidateIndex,
    *,
    max_padded_length: int,
    max_padding_ratio: float,
    recent_arrival_ids: AbstractSet[int],
    max_candidate_windows: int | None = None,
) -> CandidateSearchResult:
    """Find the best candidate that satisfies the configured padding threshold."""

    if recent_arrival_ids:
        recent_indices = index.recent_indices(recent_arrival_ids)
        candidates = iter_recent_batch_candidates(
            index,
            max_padded_length=max_padded_length,
            max_padding_ratio=max_padding_ratio,
            recent_indices=recent_indices,
            max_candidate_windows=max_candidate_windows,
        )
    else:
        candidates = iter_batch_candidates(
            index,
            max_padded_length=max_padded_length,
            max_padding_ratio=max_padding_ratio,
        )

    inspected_count = 0
    best_candidate: BatchCandidate | None = None
    for candidate in candidates:
        inspected_count += 1
        if candidate.padding_ratio > max_padding_ratio:
            continue
        if best_candidate is None or threshold_candidate_key(
            candidate
        ) < threshold_candidate_key(best_candidate):
            best_candidate = candidate

    return CandidateSearchResult(best_candidate, inspected_count)


def find_best_candidate(
    index: CandidateIndex,
    *,
    max_padded_length: int,
    max_padding_ratio: float,
) -> CandidateSearchResult:
    """Find the lowest-padding candidate when no threshold candidate is ready."""

    candidates = iter_batch_candidates(
        index,
        max_padded_length=max_padded_length,
        max_padding_ratio=max_padding_ratio,
    )

    inspected_count = 0
    best_candidate: BatchCandidate | None = None
    best_singleton: BatchCandidate | None = None
    for candidate in candidates:
        inspected_count += 1
        if candidate.record_count > 1:
            if best_candidate is None or best_candidate_key(
                candidate
            ) < best_candidate_key(best_candidate):
                best_candidate = candidate
        elif best_singleton is None or best_candidate_key(candidate) < best_candidate_key(
            best_singleton
        ):
            best_singleton = candidate

    return CandidateSearchResult(best_candidate or best_singleton, inspected_count)


def iter_batch_candidates(
    index: CandidateIndex,
    *,
    max_padded_length: int,
    max_padding_ratio: float,
) -> Iterator[BatchCandidate]:
    """Yield candidate windows ending at each length-sorted record."""

    for end_index, longest_record in enumerate(index.records):
        if longest_record.length <= 0:
            continue

        max_record_count = max_padded_length // longest_record.length
        if max_record_count <= 0:
            continue

        widest_start_index = max(0, end_index - max_record_count + 1)
        yield index.make_candidate(widest_start_index, end_index)

        min_length_for_ratio = ceil(longest_record.length * (1 - max_padding_ratio))
        tight_start_index = bisect_left(
            index.sorted_lengths,
            min_length_for_ratio,
            widest_start_index,
            end_index + 1,
        )
        if tight_start_index <= end_index and tight_start_index != widest_start_index:
            yield index.make_candidate(tight_start_index, end_index)


def iter_recent_batch_candidates(
    index: CandidateIndex,
    *,
    max_padded_length: int,
    max_padding_ratio: float,
    recent_indices: Sequence[int],
    max_candidate_windows: int | None = None,
) -> Iterator[BatchCandidate]:
    """Yield candidate windows that contain at least one recent record."""

    if max_candidate_windows is not None and max_candidate_windows <= 0:
        raise ValueError("max_candidate_windows must be a positive integer.")

    seen_windows: set[tuple[int, int]] = set()
    yielded_count = 0
    for recent_index in recent_indices:
        for end_index in range(recent_index, len(index.records)):
            longest_record = index.records[end_index]
            if longest_record.length <= 0:
                continue

            max_record_count = max_padded_length // longest_record.length
            if max_record_count <= 0:
                break

            widest_start_index = max(0, end_index - max_record_count + 1)
            if widest_start_index > recent_index:
                break

            for candidate in _yield_recent_candidate_once(
                index,
                widest_start_index,
                end_index,
                recent_index,
                seen_windows,
            ):
                yield candidate
                yielded_count += 1
                if (
                    max_candidate_windows is not None
                    and yielded_count >= max_candidate_windows
                ):
                    return

            min_length_for_ratio = ceil(longest_record.length * (1 - max_padding_ratio))
            tight_start_index = bisect_left(
                index.sorted_lengths,
                min_length_for_ratio,
                widest_start_index,
                end_index + 1,
            )
            if tight_start_index <= recent_index and tight_start_index != widest_start_index:
                for candidate in _yield_recent_candidate_once(
                    index,
                    tight_start_index,
                    end_index,
                    recent_index,
                    seen_windows,
                ):
                    yield candidate
                    yielded_count += 1
                    if (
                        max_candidate_windows is not None
                        and yielded_count >= max_candidate_windows
                    ):
                        return


def _yield_recent_candidate_once(
    index: CandidateIndex,
    start_index: int,
    end_index: int,
    recent_index: int,
    seen_windows: set[tuple[int, int]],
) -> Iterator[BatchCandidate]:
    if not start_index <= recent_index <= end_index:
        return

    window_key = (start_index, end_index)
    if window_key in seen_windows:
        return

    seen_windows.add(window_key)
    yield index.make_candidate(start_index, end_index)


def threshold_candidate_key(candidate: BatchCandidate) -> tuple[int, float, int, int]:
    return (
        -candidate.total_padded_length,
        candidate.padding_ratio,
        candidate.total_padding_length,
        candidate.earliest_arrival_id,
    )


def best_candidate_key(candidate: BatchCandidate) -> tuple[float, int, int, int]:
    return (
        candidate.padding_ratio,
        candidate.total_padding_length,
        -candidate.total_padded_length,
        candidate.earliest_arrival_id,
    )
