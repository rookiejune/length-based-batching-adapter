"""Source-loader construction for reading length records."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from ._source_records import (
    IndexedSample,
    IndexedSampleDataset,
    PlanBatchSampler,
    PlanCollator,
    RecordCollator,
    expected_plan_lengths,
)
from ._api_types import LengthFn
from ._records import BatchPlan


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


def _build_map_loader_kwargs(
    dataloader: DataLoader, collate_fn: RecordCollator
) -> dict[str, Any]:
    loader_kwargs = _build_common_loader_kwargs(dataloader, collate_fn)
    loader_kwargs["batch_sampler"] = dataloader.batch_sampler
    return loader_kwargs


def _build_common_loader_kwargs(
    dataloader: DataLoader, collate_fn: RecordCollator
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
