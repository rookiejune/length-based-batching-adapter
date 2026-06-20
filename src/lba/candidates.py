"""Candidate batch selection helpers for LBA."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import ceil
from typing import AbstractSet

from .types import SampleRecord


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
class CandidateSearchResult:
    """Candidate search result plus the amount of window work performed."""

    candidate: BatchCandidate | None
    inspected_count: int


def find_threshold_candidate(
    records: Sequence[SampleRecord],
    prefix_lengths: Sequence[int],
    *,
    max_padded_length: int,
    max_padding_ratio: float,
    recent_arrival_ids: AbstractSet[int],
) -> CandidateSearchResult:
    """Find the best candidate that satisfies the configured padding threshold."""

    sorted_lengths = [record.length for record in records]
    if recent_arrival_ids:
        recent_indices = recent_record_indices(records, recent_arrival_ids)
        candidates = iter_recent_batch_candidates(
            records,
            prefix_lengths,
            max_padded_length=max_padded_length,
            max_padding_ratio=max_padding_ratio,
            recent_indices=recent_indices,
            sorted_lengths=sorted_lengths,
        )
    else:
        candidates = iter_batch_candidates(
            records,
            prefix_lengths,
            max_padded_length=max_padded_length,
            max_padding_ratio=max_padding_ratio,
            sorted_lengths=sorted_lengths,
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
    records: Sequence[SampleRecord],
    prefix_lengths: Sequence[int],
    *,
    max_padded_length: int,
    max_padding_ratio: float,
) -> CandidateSearchResult:
    """Find the lowest-padding candidate when no threshold candidate is ready."""

    sorted_lengths = [record.length for record in records]
    candidates = iter_batch_candidates(
        records,
        prefix_lengths,
        max_padded_length=max_padded_length,
        max_padding_ratio=max_padding_ratio,
        sorted_lengths=sorted_lengths,
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
    records: Sequence[SampleRecord],
    prefix_lengths: Sequence[int],
    *,
    max_padded_length: int,
    max_padding_ratio: float,
    sorted_lengths: Sequence[int] | None = None,
) -> Iterator[BatchCandidate]:
    """Yield candidate windows ending at each length-sorted record."""

    if sorted_lengths is None:
        sorted_lengths = [record.length for record in records]
    for end_index, longest_record in enumerate(records):
        if longest_record.length <= 0:
            continue

        max_record_count = max_padded_length // longest_record.length
        if max_record_count <= 0:
            continue

        widest_start_index = max(0, end_index - max_record_count + 1)
        yield make_batch_candidate(
            records,
            prefix_lengths,
            widest_start_index,
            end_index,
        )

        min_length_for_ratio = ceil(longest_record.length * (1 - max_padding_ratio))
        tight_start_index = bisect_left(
            sorted_lengths,
            min_length_for_ratio,
            widest_start_index,
            end_index + 1,
        )
        if tight_start_index <= end_index and tight_start_index != widest_start_index:
            yield make_batch_candidate(
                records,
                prefix_lengths,
                tight_start_index,
                end_index,
            )


def iter_recent_batch_candidates(
    records: Sequence[SampleRecord],
    prefix_lengths: Sequence[int],
    *,
    max_padded_length: int,
    max_padding_ratio: float,
    recent_indices: Sequence[int],
    sorted_lengths: Sequence[int] | None = None,
) -> Iterator[BatchCandidate]:
    """Yield candidate windows that contain at least one recent record."""

    if sorted_lengths is None:
        sorted_lengths = [record.length for record in records]

    seen_windows: set[tuple[int, int]] = set()
    for recent_index in recent_indices:
        for end_index in range(recent_index, len(records)):
            longest_record = records[end_index]
            if longest_record.length <= 0:
                continue

            max_record_count = max_padded_length // longest_record.length
            if max_record_count <= 0:
                break

            widest_start_index = max(0, end_index - max_record_count + 1)
            if widest_start_index > recent_index:
                break

            yield from _yield_recent_candidate_once(
                records,
                prefix_lengths,
                widest_start_index,
                end_index,
                recent_index,
                seen_windows,
            )

            min_length_for_ratio = ceil(longest_record.length * (1 - max_padding_ratio))
            tight_start_index = bisect_left(
                sorted_lengths,
                min_length_for_ratio,
                widest_start_index,
                end_index + 1,
            )
            if tight_start_index <= recent_index and tight_start_index != widest_start_index:
                yield from _yield_recent_candidate_once(
                    records,
                    prefix_lengths,
                    tight_start_index,
                    end_index,
                    recent_index,
                    seen_windows,
                )


def _yield_recent_candidate_once(
    records: Sequence[SampleRecord],
    prefix_lengths: Sequence[int],
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
    yield make_batch_candidate(records, prefix_lengths, start_index, end_index)


def make_batch_candidate(
    records: Sequence[SampleRecord],
    prefix_lengths: Sequence[int],
    start_index: int,
    end_index: int,
) -> BatchCandidate:
    total_raw_length = prefix_lengths[end_index + 1] - prefix_lengths[start_index]
    longest_length = records[end_index].length
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
        earliest_arrival_id=min(
            records[index].arrival_id for index in range(start_index, end_index + 1)
        ),
    )


def recent_record_indices(
    records: Sequence[SampleRecord],
    recent_arrival_ids: AbstractSet[int],
) -> list[int]:
    return [
        index
        for index, record in enumerate(records)
        if record.arrival_id in recent_arrival_ids
    ]


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
