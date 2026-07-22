"""Configuration objects for LBA."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

from ._api_types import CostFn
from .adaptive import AdaptiveConfig


DEFAULT_PREFETCH_BATCHES = 4
DEFAULT_THROUGHPUT_MAX_CANDIDATE_WINDOWS = 256
DEFAULT_THROUGHPUT_FALLBACK_AFTER = 8
DEFAULT_THROUGHPUT_FALLBACK_POOL_SIZE = 1024
PlannerMode = Literal["quality", "throughput", "latency"]


@dataclass(frozen=True)
class LBAConfig:
    """User-facing configuration normalized for the adapter."""

    max_padded_length: Optional[int] = None
    warmup_batches: Optional[int] = None
    cost_fn: Optional[CostFn] = None
    max_batch_cost: Optional[int] = None
    cost_window_batches: int = 1
    distributed_cost_window_batches: Optional[int] = None
    adaptive: Optional[AdaptiveConfig] = None
    max_cache_samples: int = 8192
    max_padding_ratio: float = 0.05
    prefetch_batches: int = DEFAULT_PREFETCH_BATCHES
    planner_mode: PlannerMode = "latency"
    max_candidate_windows: Optional[int] = None
    limited_search_fallback_after: Optional[int] = None
    limited_search_fallback_pool_size: Optional[int] = None
    drop_last_flush: bool = True
    spill_dir: Optional[Union[str, Path]] = None
    log_dir: Optional[Union[str, Path]] = None

    def __post_init__(self) -> None:
        if self.max_padded_length is not None and self.max_padded_length <= 0:
            raise ValueError("max_padded_length must be a positive integer.")
        if self.warmup_batches is not None and self.warmup_batches <= 0:
            raise ValueError("warmup_batches must be a positive integer.")
        if self.cost_fn is not None and not callable(self.cost_fn):
            raise TypeError("cost_fn must be callable.")
        if self.max_batch_cost is not None and self.max_batch_cost <= 0:
            raise ValueError("max_batch_cost must be a positive integer.")
        if self.cost_fn is None and self.max_batch_cost is not None:
            raise ValueError("max_batch_cost requires cost_fn.")
        if self.cost_fn is not None and self.max_batch_cost is None:
            raise ValueError("cost_fn requires max_batch_cost.")
        if self.cost_fn is not None and self.max_padded_length is not None:
            raise ValueError(
                "cost_fn and max_padded_length define overlapping batch budgets."
            )
        if self.cost_fn is not None and self.warmup_batches is not None:
            raise ValueError("warmup_batches is unavailable with cost_fn.")
        if isinstance(self.cost_window_batches, bool):
            raise TypeError("cost_window_batches must be an integer.")
        try:
            cost_window_batches = operator.index(self.cost_window_batches)
        except TypeError as error:
            raise TypeError("cost_window_batches must be an integer.") from error
        if cost_window_batches <= 0:
            raise ValueError("cost_window_batches must be a positive integer.")
        object.__setattr__(self, "cost_window_batches", cost_window_batches)
        if self.distributed_cost_window_batches is not None:
            if isinstance(self.distributed_cost_window_batches, bool):
                raise TypeError(
                    "distributed_cost_window_batches must be an integer."
                )
            try:
                distributed_cost_window_batches = operator.index(
                    self.distributed_cost_window_batches
                )
            except TypeError as error:
                raise TypeError(
                    "distributed_cost_window_batches must be an integer."
                ) from error
            if distributed_cost_window_batches < 2:
                raise ValueError(
                    "distributed_cost_window_batches must be at least 2."
                )
            object.__setattr__(
                self,
                "distributed_cost_window_batches",
                distributed_cost_window_batches,
            )
            if cost_window_batches > 1:
                raise ValueError(
                    "distributed_cost_window_batches and cost_window_batches > 1 "
                    "are mutually exclusive."
                )
        if self.adaptive is not None:
            if not isinstance(self.adaptive, AdaptiveConfig):
                raise TypeError("adaptive must be an AdaptiveConfig.")
            if (
                self.adaptive.adjusts_distributed_cost_window
                and self.distributed_cost_window_batches is not None
            ):
                raise ValueError(
                    "adaptive distributed_cost_window_batches and "
                    "distributed_cost_window_batches are mutually exclusive."
                )
            if (
                self.adaptive.adjusts_distributed_cost_window
                and cost_window_batches > 1
            ):
                raise ValueError(
                    "adaptive distributed_cost_window_batches and "
                    "cost_window_batches > 1 are mutually exclusive."
                )
        if self.max_cache_samples <= 0:
            raise ValueError("max_cache_samples must be a positive integer.")
        if not 0 <= self.max_padding_ratio <= 1:
            raise ValueError("max_padding_ratio must be between 0 and 1.")
        if self.prefetch_batches < 0:
            raise ValueError("prefetch_batches must be greater than or equal to 0.")
        if self.planner_mode not in ("quality", "throughput", "latency"):
            raise ValueError(
                "planner_mode must be 'quality', 'throughput', or 'latency'."
            )
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
    def candidate_window_limit(self) -> Optional[int]:
        if self.max_candidate_windows is not None:
            return self.max_candidate_windows
        if self.planner_mode in ("throughput", "latency"):
            return DEFAULT_THROUGHPUT_MAX_CANDIDATE_WINDOWS
        return None

    @property
    def uses_custom_cost(self) -> bool:
        return self.cost_fn is not None

    @property
    def limited_search_fallback_after_limit(self) -> Optional[int]:
        if self.limited_search_fallback_after is not None:
            return self.limited_search_fallback_after
        if self.planner_mode == "throughput":
            return DEFAULT_THROUGHPUT_FALLBACK_AFTER
        return None

    @property
    def limited_search_fallback_pool_limit(self) -> Optional[int]:
        if self.limited_search_fallback_pool_size is not None:
            return self.limited_search_fallback_pool_size
        if self.planner_mode == "throughput":
            return min(
                self.max_cache_samples,
                DEFAULT_THROUGHPUT_FALLBACK_POOL_SIZE,
            )
        return None

    @property
    def defer_limited_search_miss(self) -> bool:
        return self.planner_mode == "throughput"
