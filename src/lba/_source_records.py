"""Length-record construction for source sample batches."""

from __future__ import annotations

import operator
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from torch.utils.data import Dataset

from ._api_types import LengthFn
from ._records import LengthRecord


@dataclass(frozen=True)
class IndexedSample:
    index: int
    sample: Any


class IndexedSampleDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> IndexedSample:
        return IndexedSample(index=index, sample=self.dataset[index])


class RecordCollator:
    """Collate raw samples into records with lengths."""

    def __init__(self, len_fn: LengthFn) -> None:
        self.len_fn = len_fn

    def __call__(self, samples: list[Any]) -> list[LengthRecord]:
        length_records: list[LengthRecord] = []
        for sample in samples:
            raw_sample = sample
            sample_index: Optional[int] = None
            if isinstance(sample, IndexedSample):
                raw_sample = sample.sample
                sample_index = sample.index

            sample_length = operator.index(self.len_fn(raw_sample))
            if sample_length <= 0:
                raise ValueError("len_fn must return a positive integer.")
            length_records.append(
                LengthRecord(
                    sample=raw_sample,
                    length=sample_length,
                    index=sample_index,
                )
            )
        return length_records


def iter_length_record_batches(
    source_batches: Iterable[Sequence[Any]], len_fn: LengthFn
) -> Iterator[list[LengthRecord]]:
    """Yield length records from an iterable that already produces sample batches."""

    collate_fn = RecordCollator(len_fn)
    for samples in source_batches:
        yield collate_fn(list(samples))
