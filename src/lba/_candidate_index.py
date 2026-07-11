"""Indexed views over the length-sorted planner pool."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import AbstractSet, Optional

from ._records import SampleRecord


class ArrivalIdRangeMin:
    """Range-min index for arrival ids in the current length-sorted pool."""

    def __init__(self, arrival_ids: Sequence[int]) -> None:
        if not arrival_ids:
            raise ValueError("arrival_ids must not be empty.")

        size = 1
        while size < len(arrival_ids):
            size *= 2

        empty_value = max(arrival_ids) + 1
        values = [empty_value] * (2 * size)
        values[size : size + len(arrival_ids)] = arrival_ids
        for index in range(size - 1, 0, -1):
            values[index] = min(values[index * 2], values[index * 2 + 1])

        self._length = len(arrival_ids)
        self._size = size
        self._values = values

    @classmethod
    def from_records(cls, records: Sequence[SampleRecord]) -> ArrivalIdRangeMin:
        return cls([record.arrival_id for record in records])

    def range_min(self, start_index: int, end_index: int) -> int:
        if (
            start_index < 0
            or end_index < start_index
            or end_index >= self._length
        ):
            raise ValueError("Invalid range-min query.")

        left = start_index + self._size
        right = end_index + self._size
        best = self._values[left]
        while left <= right:
            if left % 2 == 1:
                best = min(best, self._values[left])
                left += 1
            if right % 2 == 0:
                best = min(best, self._values[right])
                right -= 1
            left //= 2
            right //= 2
        return best


@dataclass(frozen=True)
class BatchCandidate:
    """A contiguous length-sorted window that can become a dynamic batch."""

    start_index: int
    end_index: int
    total_raw_length: int
    total_padded_length: int
    total_padding_length: int
    padding_ratio: float
    earliest_arrival_id: int

    @property
    def record_count(self) -> int:
        return self.end_index - self.start_index + 1


@dataclass(frozen=True)
class CandidateIndex:
    """Indexed view of the current length-sorted record pool."""

    records: Sequence[SampleRecord]
    prefix_lengths: Sequence[int]
    sorted_lengths: Sequence[int]
    _arrival_id_range_min: Optional[ArrivalIdRangeMin] = None

    @classmethod
    def from_records(cls, records: Sequence[SampleRecord]) -> CandidateIndex:
        prefix_lengths = [0]
        sorted_lengths: list[int] = []
        for record in records:
            sorted_lengths.append(record.length)
            prefix_lengths.append(prefix_lengths[-1] + record.length)
        return cls(
            records=records,
            prefix_lengths=prefix_lengths,
            sorted_lengths=sorted_lengths,
        )

    def recent_indices(self, recent_arrival_ids: AbstractSet[int]) -> list[int]:
        return [
            index
            for index, record in enumerate(self.records)
            if record.arrival_id in recent_arrival_ids
        ]

    def make_candidate(self, start_index: int, end_index: int) -> BatchCandidate:
        if not self.records:
            raise RuntimeError("Candidate index has no records.")
        total_raw_length, total_padded_length, total_padding_length, padding_ratio = (
            self.candidate_lengths(start_index, end_index)
        )
        return BatchCandidate(
            start_index=start_index,
            end_index=end_index,
            total_raw_length=total_raw_length,
            total_padded_length=total_padded_length,
            total_padding_length=total_padding_length,
            padding_ratio=padding_ratio,
            earliest_arrival_id=self.arrival_id_range_min.range_min(
                start_index, end_index
            ),
        )

    @property
    def arrival_id_range_min(self) -> ArrivalIdRangeMin:
        if self._arrival_id_range_min is None:
            object.__setattr__(
                self,
                "_arrival_id_range_min",
                ArrivalIdRangeMin.from_records(self.records),
            )
        return self._arrival_id_range_min

    def candidate_lengths(
        self, start_index: int, end_index: int
    ) -> tuple[int, int, int, float]:
        total_raw_length = self.prefix_lengths[end_index + 1] - self.prefix_lengths[
            start_index
        ]
        longest_length = self.records[end_index].length
        record_count = end_index - start_index + 1
        total_padded_length = longest_length * record_count
        total_padding_length = total_padded_length - total_raw_length
        padding_ratio = (
            total_padding_length / total_padded_length if total_padded_length else 0.0
        )
        return (
            total_raw_length,
            total_padded_length,
            total_padding_length,
            padding_ratio,
        )

    def make_candidate_with_scanned_arrivals(
        self, start_index: int, end_index: int
    ) -> BatchCandidate:
        total_raw_length, total_padded_length, total_padding_length, padding_ratio = (
            self.candidate_lengths(start_index, end_index)
        )
        return BatchCandidate(
            start_index=start_index,
            end_index=end_index,
            total_raw_length=total_raw_length,
            total_padded_length=total_padded_length,
            total_padding_length=total_padding_length,
            padding_ratio=padding_ratio,
            earliest_arrival_id=min(
                record.arrival_id for record in self.records[start_index : end_index + 1]
            ),
        )
