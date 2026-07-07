"""Compatibility exports for candidate batch selection helpers."""

from __future__ import annotations

from ._candidate_index import ArrivalIdRangeMin, BatchCandidate, CandidateIndex
from ._candidate_search import (
    CandidateSearchResult,
    best_candidate_key,
    find_best_candidate,
    find_threshold_candidate,
    iter_batch_candidates,
    iter_recent_batch_candidates,
    threshold_candidate_key,
)

__all__ = [
    "ArrivalIdRangeMin",
    "BatchCandidate",
    "CandidateIndex",
    "CandidateSearchResult",
    "best_candidate_key",
    "find_best_candidate",
    "find_threshold_candidate",
    "iter_batch_candidates",
    "iter_recent_batch_candidates",
    "threshold_candidate_key",
]
