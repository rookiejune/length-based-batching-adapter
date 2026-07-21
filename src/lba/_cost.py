"""Batch-cost evaluation and budget search."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Optional

from ._api_types import CostFn


@dataclass(frozen=True)
class BatchCost:
    """Evaluate a monotone padded-batch cost under one budget."""

    budget: int
    cost_fn: Optional[CostFn] = None

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError("Batch cost budget must be a positive integer.")
        if self.cost_fn is not None and not callable(self.cost_fn):
            raise TypeError("cost_fn must be callable.")

    def estimate(self, max_length: int, batch_size: int) -> int:
        if max_length <= 0:
            raise ValueError("Batch max length must be a positive integer.")
        if batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        if self.cost_fn is None:
            return max_length * batch_size

        try:
            value = operator.index(self.cost_fn(max_length, batch_size))
        except TypeError as error:
            raise TypeError("cost_fn must return a positive integer.") from error
        if value <= 0:
            raise ValueError("cost_fn must return a positive integer.")
        return value

    def max_batch_size(self, max_length: int, available: int) -> int:
        """Return the largest feasible size, assuming cost is monotone in size."""

        if available <= 0:
            return 0
        if max_length <= 0:
            raise ValueError("Batch max length must be a positive integer.")
        if self.cost_fn is None:
            return min(available, self.budget // max_length)
        if self.estimate(max_length, 1) > self.budget:
            return 0
        if self.estimate(max_length, available) <= self.budget:
            return available

        left = 1
        right = available
        best = 1
        while left <= right:
            middle = (left + right) // 2
            if self.estimate(max_length, middle) <= self.budget:
                best = middle
                left = middle + 1
            else:
                right = middle - 1
        return best
