"""Candidate-window search strategies for the batch planner."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import ceil
from typing import AbstractSet

from ._candidate_index import BatchCandidate, CandidateIndex


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
