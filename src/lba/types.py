"""Shared internal types for LBA."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

LengthFn = Callable[[Any], int]
CollateFn = Callable[[list[Any]], Any]


class PlanReason(str, Enum):
    """Reason a planned batch exists."""

    def _generate_next_value_(name: str, start: int, count: int, last_values: list[str]) -> str:
        return name.lower()

    PLANNED = auto()
    OVERSIZED = auto()


@dataclass(frozen=True)
class LengthRecord:
    """A raw sample with its effective length before global arrival order exists."""

    sample: Any
    length: int
    index: int | None = None


@dataclass(frozen=True)
class SampleRecord:
    """A raw sample plus the metadata LBA needs to plan batches."""

    sample: Any
    length: int
    arrival_id: int
    index: int | None = None


@dataclass(frozen=True)
class BatchPlan:
    """A planned dynamic batch before the original collate function runs."""

    records: Sequence[SampleRecord]
    raw_length_sum: int
    padded_length: int
    padding_length: int
    padding_ratio: float
    reason: PlanReason

    @property
    def samples(self) -> list[Any]:
        return [record.sample for record in self.records]
