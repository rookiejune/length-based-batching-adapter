"""Length-record construction for source sample batches."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

from torch.utils.data import Dataset

from ._api_types import LengthFn
from ._records import BatchPlan, LengthRecord


@dataclass(frozen=True)
class IndexedSample:
    index: int
    sample: Any


class IndexedSampleDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        object.__setattr__(self, "dataset", dataset)

    def __getattr__(self, name: str) -> Any:
        try:
            dataset = object.__getattribute__(self, "dataset")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(dataset, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "dataset":
            object.__setattr__(self, name, value)
            return
        setattr(self.dataset, name, value)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> IndexedSample:
        return IndexedSample(index=index, sample=self.dataset[index])

    def __getitems__(self, indices: list[int]) -> list[IndexedSample]:
        batch_getter = getattr(self.dataset, "__getitems__", None)
        if batch_getter is None:
            samples = [self.dataset[index] for index in indices]
        else:
            samples = batch_getter(indices)
        if len(samples) != len(indices):
            raise RuntimeError(
                "Dataset __getitems__ must return one sample for every requested index."
            )
        return [
            IndexedSample(index=index, sample=sample)
            for index, sample in zip(indices, samples)
        ]


class RecordCollator:
    """Collate raw samples into lightweight index/length records."""

    def __init__(self, len_fn: LengthFn) -> None:
        self.len_fn = len_fn

    def __call__(self, samples: list[Any]) -> list[LengthRecord]:
        length_records: list[LengthRecord] = []
        for sample in samples:
            if not isinstance(sample, IndexedSample):
                raise RuntimeError("LBA source records require map-style sample indices.")

            sample_length = operator.index(self.len_fn(sample.sample))
            if sample_length <= 0:
                raise ValueError("len_fn must return a positive integer.")
            length_records.append(
                LengthRecord(
                    sample=sample.index,
                    length=sample_length,
                    index=sample.index,
                )
            )
        return length_records


class PlanBatchSampler:
    """Yield dynamic index batches from planned metadata."""

    def __init__(self, plans: Sequence[BatchPlan]) -> None:
        self.index_batches = [
            [self._require_index(record.index) for record in plan.records]
            for plan in plans
        ]

    def __iter__(self):
        yield from self.index_batches

    def __len__(self) -> int:
        return len(self.index_batches)

    @staticmethod
    def _require_index(index: Optional[int]) -> int:
        if index is None:
            raise RuntimeError("LBA dynamic batch materialization requires sample indices.")
        return index


class PlanCollator:
    """Validate materialized samples and delegate to the user collate function."""

    def __init__(
        self,
        len_fn: LengthFn,
        collate_fn,
        expected_lengths: dict[int, int],
    ) -> None:
        self.len_fn = len_fn
        self.collate_fn = collate_fn
        self.expected_lengths = expected_lengths

    def __call__(self, samples: list[Any]):
        raw_samples: list[Any] = []
        for sample in samples:
            if not isinstance(sample, IndexedSample):
                raise RuntimeError(
                    "LBA materialized batches require map-style sample indices."
                )
            expected_length = self.expected_lengths[sample.index]
            materialized_length = operator.index(self.len_fn(sample.sample))
            if materialized_length <= 0:
                raise RuntimeError(
                    f"LBA materialized dataset index {sample.index} has "
                    "non-positive effective length."
                )
            if materialized_length != expected_length:
                raise RuntimeError(
                    "LBA materialized dataset index changed effective length: "
                    f"index={sample.index} expected={expected_length} "
                    f"actual={materialized_length}."
                )
            raw_samples.append(sample.sample)
        return self.collate_fn(raw_samples)


def expected_plan_lengths(plans: Sequence[BatchPlan]) -> dict[int, int]:
    expected: dict[int, int] = {}
    for plan in plans:
        for record in plan.records:
            index = PlanBatchSampler._require_index(record.index)
            previous = expected.get(index)
            if previous is not None and previous != record.length:
                raise RuntimeError(
                    "LBA source metadata changed effective length for repeated "
                    f"dataset index: index={index} expected={previous} "
                    f"actual={record.length}."
                )
            expected[index] = record.length
    return expected
