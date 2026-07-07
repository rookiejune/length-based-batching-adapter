"""Source-loader construction for reading length records."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader, IterableDataset

from ._source_records import (
    IndexedSample,
    IndexedSampleDataset,
    RecordCollator,
    iter_length_record_batches,
)
from .types import LengthFn


def build_source_loader(dataloader: DataLoader, len_fn: LengthFn) -> DataLoader:
    """Build a loader that yields lists of LengthRecord."""

    collate_fn = RecordCollator(len_fn)
    if isinstance(dataloader.dataset, IterableDataset):
        loader_kwargs = _build_iterable_loader_kwargs(dataloader, collate_fn)
    else:
        loader_kwargs = _build_map_loader_kwargs(dataloader, collate_fn)

    dataset = dataloader.dataset
    if not isinstance(dataset, IterableDataset):
        dataset = IndexedSampleDataset(dataset)
    return DataLoader(dataset, **loader_kwargs)


def _build_map_loader_kwargs(
    dataloader: DataLoader, collate_fn: RecordCollator
) -> dict[str, Any]:
    loader_kwargs = _build_common_loader_kwargs(dataloader, collate_fn)
    loader_kwargs["batch_sampler"] = dataloader.batch_sampler
    return loader_kwargs


def _build_iterable_loader_kwargs(
    dataloader: DataLoader, collate_fn: RecordCollator
) -> dict[str, Any]:
    if dataloader.batch_size is None:
        raise ValueError(
            "LBA requires a batched DataLoader when wrapping an IterableDataset."
        )

    loader_kwargs = _build_common_loader_kwargs(dataloader, collate_fn)
    loader_kwargs["batch_size"] = dataloader.batch_size
    loader_kwargs["drop_last"] = dataloader.drop_last
    return loader_kwargs


def _build_common_loader_kwargs(
    dataloader: DataLoader, collate_fn: RecordCollator
) -> dict[str, Any]:
    loader_kwargs: dict[str, Any] = {
        "num_workers": dataloader.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": dataloader.pin_memory,
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

    if dataloader.pin_memory_device:
        loader_kwargs["pin_memory_device"] = dataloader.pin_memory_device

    return loader_kwargs


__all__ = [
    "IndexedSample",
    "IndexedSampleDataset",
    "RecordCollator",
    "build_source_loader",
    "iter_length_record_batches",
]
