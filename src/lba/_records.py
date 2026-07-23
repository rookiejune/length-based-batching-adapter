"""Record and plan types shared across LBA internals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ._cost import BatchCost


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
    index: Optional[int] = None


@dataclass(frozen=True)
class SampleRecord:
    """A raw sample plus the metadata LBA needs to plan batches."""

    sample: Any
    length: int
    arrival_id: int
    index: Optional[int] = None
    materialized: bool = True


@dataclass(frozen=True)
class BatchPlan:
    """A planned dynamic batch before the original collate function runs."""

    records: Sequence[SampleRecord]
    raw_length_sum: int
    padded_length: int
    padding_length: int
    padding_ratio: float
    reason: PlanReason
    estimated_cost: Optional[int] = None

    @property
    def samples(self) -> list[Any]:
        return [record.sample for record in self.records]


def make_batch_plan(
    records: Sequence[SampleRecord],
    reason: PlanReason,
    *,
    batch_cost: Optional[BatchCost] = None,
    raw_length_sum: Optional[int] = None,
    padded_length: Optional[int] = None,
    padding_length: Optional[int] = None,
    padding_ratio: Optional[float] = None,
    estimated_cost: Optional[int] = None,
) -> BatchPlan:
    """Build a plan from records and optional precomputed shape metrics."""

    ordered_records = tuple(sorted(records, key=lambda record: record.arrival_id))
    if not ordered_records:
        raise ValueError("BatchPlan requires at least one record.")

    if raw_length_sum is None:
        raw_length_sum = sum(record.length for record in ordered_records)
    if padded_length is None:
        max_length = max(record.length for record in ordered_records)
        padded_length = max_length * len(ordered_records)
    if padding_length is None:
        padding_length = padded_length - raw_length_sum
    if padding_ratio is None:
        padding_ratio = padding_length / padded_length if padded_length else 0.0
    if estimated_cost is None:
        if batch_cost is not None:
            max_length = max(record.length for record in ordered_records)
            estimated_cost = batch_cost.estimate(
                max_length,
                len(ordered_records),
            )
        else:
            estimated_cost = padded_length

    return BatchPlan(
        records=ordered_records,
        raw_length_sum=raw_length_sum,
        padded_length=padded_length,
        padding_length=padding_length,
        padding_ratio=padding_ratio,
        reason=reason,
        estimated_cost=estimated_cost,
    )
