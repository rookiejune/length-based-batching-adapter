"""Length-budget resolution for LBA."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from .config import LBAConfig
from .types import LengthRecord


class BatchSizeSource(Protocol):
    batch_size: int | None


class BudgetResolver:
    """Resolve max_padded_length before dynamic batching starts."""

    def __init__(self, config: LBAConfig, dataloader: BatchSizeSource) -> None:
        self.config = config
        self.dataloader = dataloader

    def warmup_batch_count(self) -> int:
        if self.config.warmup_batches is not None:
            return self.config.warmup_batches

        batch_size = self.dataloader.batch_size
        if isinstance(batch_size, int) and batch_size > 0:
            return min(batch_size, 32)
        return 1

    def resolve(self, warmup_records: Sequence[LengthRecord]) -> int:
        if self.config.max_padded_length is not None:
            return self.config.max_padded_length
        if not warmup_records:
            raise ValueError("Cannot infer max_padded_length without warmup samples.")

        effective_batch_size = self.dataloader.batch_size
        if not isinstance(effective_batch_size, int) or effective_batch_size <= 0:
            effective_batch_size = len(warmup_records)

        average_sample_length = sum(record.length for record in warmup_records) / len(
            warmup_records
        )
        return max(1, math.ceil(average_sample_length * effective_batch_size))
