"""Disk spill storage for LBA planner overflow records."""

from __future__ import annotations

import pickle
import tempfile
from collections import deque
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Optional, Union

from ._records import SampleRecord


class SpillStore:
    """Persist overflow records into ordered pickle shards."""

    def __init__(
        self,
        spill_dir: Optional[Union[str, Path]] = None,
        shard_size: int = 10_000,
    ) -> None:
        if shard_size <= 0:
            raise ValueError("shard_size must be a positive integer.")

        self.shard_size = shard_size
        self._tempdir: Optional[tempfile.TemporaryDirectory] = None
        if spill_dir is None:
            self._tempdir = tempfile.TemporaryDirectory(prefix="lba-spill-")
            self.root = Path(self._tempdir.name)
        else:
            self.root = Path(spill_dir)
            self.root.mkdir(parents=True, exist_ok=True)

        self._shard_paths: deque[Path] = deque()
        self._shard_record_counts: deque[int] = deque()
        self._next_shard_index = 0

    def write(self, records: Sequence[SampleRecord]) -> None:
        start_index = 0
        while start_index < len(records):
            if (
                not self._shard_paths
                or self._shard_record_counts[-1] >= self.shard_size
            ):
                self._start_shard()

            remaining_capacity = self.shard_size - self._shard_record_counts[-1]
            end_index = min(start_index + remaining_capacity, len(records))
            with self._shard_paths[-1].open("ab") as file:
                for record_index in range(start_index, end_index):
                    pickle.dump(
                        records[record_index],
                        file,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
            self._shard_record_counts[-1] += end_index - start_index
            start_index = end_index

    def read_shards(self) -> Iterator[list[SampleRecord]]:
        for shard_path in self._shard_paths:
            yield self._read_shard(shard_path)

    def drain_shards(self) -> Iterator[list[SampleRecord]]:
        while self._shard_paths:
            shard_path = self._shard_paths[0]
            yield self._read_shard(shard_path)
            shard_path.unlink(missing_ok=True)
            self._shard_paths.popleft()
            self._shard_record_counts.popleft()

    def drain_records(self) -> Iterator[SampleRecord]:
        while self._shard_paths:
            shard_path = self._shard_paths[0]
            yield from self._read_records(shard_path)
            shard_path.unlink(missing_ok=True)
            self._shard_paths.popleft()
            self._shard_record_counts.popleft()

    @property
    def has_shards(self) -> bool:
        return bool(self._shard_paths)

    @property
    def shard_count(self) -> int:
        return len(self._shard_paths)

    def cleanup(self) -> None:
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None
        else:
            for shard_path in self._shard_paths:
                shard_path.unlink(missing_ok=True)
        self._shard_paths.clear()
        self._shard_record_counts.clear()

    def _start_shard(self) -> None:
        while True:
            shard_path = self.root / f"spill-{self._next_shard_index:06d}.pkl"
            self._next_shard_index += 1
            if not shard_path.exists():
                break
        self._shard_paths.append(shard_path)
        self._shard_record_counts.append(0)

    @staticmethod
    def _read_shard(shard_path: Path) -> list[SampleRecord]:
        return list(SpillStore._read_records(shard_path))

    @staticmethod
    def _read_records(shard_path: Path) -> Iterator[SampleRecord]:
        with shard_path.open("rb") as file:
            while True:
                try:
                    yield pickle.load(file)
                except EOFError:
                    return

    def __del__(self) -> None:
        self.cleanup()
