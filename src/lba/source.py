"""Source-loader construction for reading and materializing length records."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

from torch.utils.data import DataLoader, Dataset

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


def build_source_loader(dataloader: DataLoader, len_fn: LengthFn) -> DataLoader:
    """Build a loader that yields lists of LengthRecord."""

    collate_fn = RecordCollator(len_fn)
    loader_kwargs = _build_map_loader_kwargs(dataloader, collate_fn)
    return DataLoader(IndexedSampleDataset(dataloader.dataset), **loader_kwargs)


def build_batch_loader(
    dataloader: DataLoader,
    plans: list[BatchPlan],
    len_fn: LengthFn,
) -> DataLoader:
    """Build a loader that materializes planned index batches."""

    collate_fn = PlanCollator(
        len_fn,
        dataloader.collate_fn,
        expected_plan_lengths(plans),
    )
    loader_kwargs = _build_common_loader_kwargs(dataloader, collate_fn)
    loader_kwargs["batch_sampler"] = PlanBatchSampler(plans)
    return DataLoader(IndexedSampleDataset(dataloader.dataset), **loader_kwargs)


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


def _build_map_loader_kwargs(
    dataloader: DataLoader, collate_fn: RecordCollator
) -> dict[str, Any]:
    loader_kwargs = _build_common_loader_kwargs(dataloader, collate_fn)
    loader_kwargs["batch_sampler"] = dataloader.batch_sampler
    return loader_kwargs


def _build_common_loader_kwargs(
    dataloader: DataLoader, collate_fn: RecordCollator | PlanCollator
) -> dict[str, Any]:
    loader_kwargs: dict[str, Any] = {
        "num_workers": dataloader.num_workers,
        "collate_fn": collate_fn,
        # The final collated batch is pinned after planning, not these records.
        "pin_memory": False,
        "timeout": dataloader.timeout,
        "worker_init_fn": dataloader.worker_init_fn,
        "persistent_workers": dataloader.persistent_workers,
    }

    if dataloader.multiprocessing_context is not None:
        loader_kwargs["multiprocessing_context"] = dataloader.multiprocessing_context

    if dataloader.generator is not None:
        loader_kwargs["generator"] = dataloader.generator

    if dataloader.num_workers > 0 and dataloader.prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = dataloader.prefetch_factor

    if hasattr(dataloader, "in_order"):
        loader_kwargs["in_order"] = dataloader.in_order

    return loader_kwargs


__all__ = [
    "IndexedSample",
    "IndexedSampleDataset",
    "PlanBatchSampler",
    "PlanCollator",
    "RecordCollator",
    "build_batch_loader",
    "build_source_loader",
]
