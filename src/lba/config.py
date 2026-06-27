"""Configuration objects for LBA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


DEFAULT_PREFETCH_BATCHES = 4
DEFAULT_THROUGHPUT_MAX_CANDIDATE_WINDOWS = 256
DEFAULT_THROUGHPUT_FALLBACK_AFTER = 8
DEFAULT_THROUGHPUT_FALLBACK_POOL_SIZE = 1024
PlannerMode = Literal["quality", "throughput"]


@dataclass(frozen=True)
class LBAConfig:
    """User-facing configuration normalized for the adapter."""

    max_padded_length: int | None = None
    warmup_batches: int | None = None
    max_cache_samples: int = 8192
    max_padding_ratio: float = 0.05
    prefetch_batches: int = DEFAULT_PREFETCH_BATCHES
    planner_mode: PlannerMode = "quality"
    max_candidate_windows: int | None = None
    limited_search_fallback_after: int | None = None
    limited_search_fallback_pool_size: int | None = None
    drop_last_flush: bool = True
    spill_dir: str | Path | None = None
    log_dir: str | Path | None = None

    def __post_init__(self) -> None:
        if self.max_padded_length is not None and self.max_padded_length <= 0:
            raise ValueError("max_padded_length must be a positive integer.")
        if self.warmup_batches is not None and self.warmup_batches <= 0:
            raise ValueError("warmup_batches must be a positive integer.")
        if self.max_cache_samples <= 0:
            raise ValueError("max_cache_samples must be a positive integer.")
        if not 0 <= self.max_padding_ratio <= 1:
            raise ValueError("max_padding_ratio must be between 0 and 1.")
        if self.prefetch_batches < 0:
            raise ValueError("prefetch_batches must be greater than or equal to 0.")
        if self.planner_mode not in ("quality", "throughput"):
            raise ValueError("planner_mode must be 'quality' or 'throughput'.")
        if self.max_candidate_windows is not None and self.max_candidate_windows <= 0:
            raise ValueError("max_candidate_windows must be a positive integer.")
        if (
            self.limited_search_fallback_after is not None
            and self.limited_search_fallback_after <= 0
        ):
            raise ValueError(
                "limited_search_fallback_after must be a positive integer."
            )
        if (
            self.limited_search_fallback_pool_size is not None
            and self.limited_search_fallback_pool_size <= 0
        ):
            raise ValueError(
                "limited_search_fallback_pool_size must be a positive integer."
            )
        if not isinstance(self.drop_last_flush, bool):
            raise TypeError("drop_last_flush must be a boolean.")

    @property
    def candidate_window_limit(self) -> int | None:
        if self.max_candidate_windows is not None:
            return self.max_candidate_windows
        if self.planner_mode == "throughput":
            return DEFAULT_THROUGHPUT_MAX_CANDIDATE_WINDOWS
        return None

    @property
    def limited_search_fallback_after_limit(self) -> int | None:
        if self.limited_search_fallback_after is not None:
            return self.limited_search_fallback_after
        if self.planner_mode == "throughput":
            return DEFAULT_THROUGHPUT_FALLBACK_AFTER
        return None

    @property
    def limited_search_fallback_pool_limit(self) -> int | None:
        if self.limited_search_fallback_pool_size is not None:
            return self.limited_search_fallback_pool_size
        if self.planner_mode == "throughput":
            return min(
                self.max_cache_samples,
                DEFAULT_THROUGHPUT_FALLBACK_POOL_SIZE,
            )
        return None
