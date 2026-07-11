"""Candidate-window search strategies for the batch planner."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import ceil
from typing import AbstractSet, Optional

from ._candidate_index import BatchCandidate, CandidateIndex


@dataclass(frozen=True)
class CandidateSearchResult:
    """Candidate search result plus the amount of window work performed."""

    candidate: Optional[BatchCandidate]
    inspected_count: int


def find_threshold_candidate(
    index: CandidateIndex,
    *,
    max_padded_length: int,
    max_padding_ratio: float,
    recent_arrival_ids: AbstractSet[int],
    max_candidate_windows: Optional[int] = None,
) -> CandidateSearchResult:
    """Find the best candidate that satisfies the configured padding threshold."""

    if recent_arrival_ids:
        recent_indices = index.recent_indices(recent_arrival_ids)
        candidate_windows = iter_recent_batch_candidate_windows(
            index,
            max_padded_length=max_padded_length,
            max_padding_ratio=max_padding_ratio,
            recent_indices=recent_indices,
            max_candidate_windows=max_candidate_windows,
        )
    else:
        candidate_windows = iter_batch_candidate_windows(
            index,
            max_padded_length=max_padded_length,
            max_padding_ratio=max_padding_ratio,
        )

    inspected_count = 0
    best_window: Optional[tuple[int, int]] = None
    best_candidate: Optional[BatchCandidate] = None
    best_key: Optional[tuple[int, float, int]] = None
    for start_index, end_index in candidate_windows:
        inspected_count += 1
        (
            _total_raw_length,
            total_padded_length,
            total_padding_length,
            padding_ratio,
        ) = index.candidate_lengths(start_index, end_index)
        if padding_ratio > max_padding_ratio:
            continue

        candidate_key = (-total_padded_length, padding_ratio, total_padding_length)
        if best_key is None or candidate_key < best_key:
            best_window = (start_index, end_index)
            best_candidate = None
            best_key = candidate_key
            continue

        if candidate_key == best_key and best_window is not None:
            if best_candidate is None:
                best_candidate = index.make_candidate_with_scanned_arrivals(*best_window)
            candidate = index.make_candidate_with_scanned_arrivals(start_index, end_index)
            if candidate.earliest_arrival_id < best_candidate.earliest_arrival_id:
                best_window = (start_index, end_index)
                best_candidate = candidate

    if best_window is None:
        return CandidateSearchResult(None, inspected_count)

    if best_candidate is None:
        best_candidate = index.make_candidate_with_scanned_arrivals(*best_window)

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
    best_candidate: Optional[BatchCandidate] = None
    best_singleton: Optional[BatchCandidate] = None
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


def iter_batch_candidate_windows(
    index: CandidateIndex,
    *,
    max_padded_length: int,
    max_padding_ratio: float,
) -> Iterator[tuple[int, int]]:
    """Yield representative candidate windows ending at each length-sorted record."""

    for end_index, longest_record in enumerate(index.records):
        if longest_record.length <= 0:
            continue

        max_record_count = max_padded_length // longest_record.length
        if max_record_count <= 0:
            continue

        widest_start_index = end_index - max_record_count + 1
        if widest_start_index < 0:
            widest_start_index = 0
        yield from _iter_representative_start_indices(
            index,
            widest_start_index=widest_start_index,
            end_index=end_index,
            max_padding_ratio=max_padding_ratio,
        )


def iter_recent_batch_candidate_windows(
    index: CandidateIndex,
    *,
    max_padded_length: int,
    max_padding_ratio: float,
    recent_indices: Sequence[int],
    max_candidate_windows: Optional[int] = None,
) -> Iterator[tuple[int, int]]:
    """Yield candidate window bounds that contain at least one recent record."""

    if max_candidate_windows is not None and max_candidate_windows <= 0:
        raise ValueError("max_candidate_windows must be a positive integer.")

    if max_candidate_windows is None:
        yield from _iter_unlimited_recent_batch_candidate_windows(
            index,
            max_padded_length=max_padded_length,
            max_padding_ratio=max_padding_ratio,
            recent_indices=recent_indices,
        )
        return

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

            widest_start_index = end_index - max_record_count + 1
            if widest_start_index < 0:
                widest_start_index = 0
            if widest_start_index > recent_index:
                break

            for start_index in _representative_start_indices(
                index,
                widest_start_index=widest_start_index,
                end_index=end_index,
                max_padding_ratio=max_padding_ratio,
            ):
                if start_index > recent_index:
                    continue
                window_key = (start_index, end_index)
                if window_key not in seen_windows:
                    seen_windows.add(window_key)
                    yield window_key
                    yielded_count += 1
                    if yielded_count >= max_candidate_windows:
                        return


def iter_batch_candidates(
    index: CandidateIndex,
    *,
    max_padded_length: int,
    max_padding_ratio: float,
) -> Iterator[BatchCandidate]:
    """Yield candidate windows ending at each length-sorted record."""

    for start_index, end_index in iter_batch_candidate_windows(
        index,
        max_padded_length=max_padded_length,
        max_padding_ratio=max_padding_ratio,
    ):
        yield index.make_candidate(start_index, end_index)


def iter_recent_batch_candidates(
    index: CandidateIndex,
    *,
    max_padded_length: int,
    max_padding_ratio: float,
    recent_indices: Sequence[int],
    max_candidate_windows: Optional[int] = None,
) -> Iterator[BatchCandidate]:
    """Yield candidate windows that contain at least one recent record."""

    for start_index, end_index in iter_recent_batch_candidate_windows(
        index,
        max_padded_length=max_padded_length,
        max_padding_ratio=max_padding_ratio,
        recent_indices=recent_indices,
        max_candidate_windows=max_candidate_windows,
    ):
        yield index.make_candidate(start_index, end_index)


def _iter_unlimited_recent_batch_candidate_windows(
    index: CandidateIndex,
    *,
    max_padded_length: int,
    max_padding_ratio: float,
    recent_indices: Sequence[int],
) -> Iterator[tuple[int, int]]:
    recent_counts = _recent_prefix_counts(len(index.records), recent_indices)
    for end_index, longest_record in enumerate(index.records):
        if longest_record.length <= 0:
            continue

        max_record_count = max_padded_length // longest_record.length
        if max_record_count <= 0:
            continue

        widest_start_index = end_index - max_record_count + 1
        if widest_start_index < 0:
            widest_start_index = 0
        for start_index in _representative_start_indices(
            index,
            widest_start_index=widest_start_index,
            end_index=end_index,
            max_padding_ratio=max_padding_ratio,
        ):
            if recent_counts[end_index + 1] > recent_counts[start_index]:
                yield start_index, end_index


def _iter_representative_start_indices(
    index: CandidateIndex,
    *,
    widest_start_index: int,
    end_index: int,
    max_padding_ratio: float,
) -> Iterator[tuple[int, int]]:
    for start_index in _representative_start_indices(
        index,
        widest_start_index=widest_start_index,
        end_index=end_index,
        max_padding_ratio=max_padding_ratio,
    ):
        yield start_index, end_index


def _representative_start_indices(
    index: CandidateIndex,
    *,
    widest_start_index: int,
    end_index: int,
    max_padding_ratio: float,
) -> Iterator[int]:
    yielded: set[int] = set()

    def yield_once(start_index: int) -> Iterator[int]:
        if widest_start_index <= start_index <= end_index and start_index not in yielded:
            yielded.add(start_index)
            yield start_index

    yield from yield_once(widest_start_index)

    longest_length = index.records[end_index].length
    min_length_for_ratio = ceil(longest_length * (1 - max_padding_ratio))
    tight_start_index = bisect_left(
        index.sorted_lengths,
        min_length_for_ratio,
        widest_start_index,
        end_index + 1,
    )
    yield from yield_once(tight_start_index)

    if tight_start_index > widest_start_index:
        yield from yield_once(tight_start_index - 1)
    yield from yield_once(end_index - 1)
    yield from yield_once(end_index)


def _recent_prefix_counts(record_count: int, recent_indices: Sequence[int]) -> list[int]:
    prefix_counts = [0] * (record_count + 1)
    for recent_index in recent_indices:
        prefix_counts[recent_index + 1] = 1
    for index in range(record_count):
        prefix_counts[index + 1] += prefix_counts[index]
    return prefix_counts


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
