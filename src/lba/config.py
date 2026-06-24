"""Configuration objects for LBA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_PREFETCH_BATCHES = 4


@dataclass(frozen=True)
class LBAConfig:
    """User-facing configuration normalized for the adapter."""

    max_padded_length: int | None = None
    warmup_batches: int | None = None
    max_cache_samples: int = 8192
    max_padding_ratio: float = 0.05
    prefetch_batches: int = DEFAULT_PREFETCH_BATCHES
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
        if not isinstance(self.drop_last_flush, bool):
            raise TypeError("drop_last_flush must be a boolean.")
