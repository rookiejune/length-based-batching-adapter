"""Disk spill storage for LBA planner overflow records."""

from __future__ import annotations

import pickle
import tempfile
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

        self._shard_paths: list[Path] = []

    def write(self, records: Sequence[SampleRecord]) -> None:
        for start_index in range(0, len(records), self.shard_size):
            shard_records = list(records[start_index : start_index + self.shard_size])
            shard_path = self.root / f"spill-{len(self._shard_paths):06d}.pkl"
            with shard_path.open("wb") as file:
                pickle.dump(shard_records, file, protocol=pickle.HIGHEST_PROTOCOL)
            self._shard_paths.append(shard_path)

    def read_shards(self) -> Iterator[list[SampleRecord]]:
        for shard_path in self._shard_paths:
            with shard_path.open("rb") as file:
                yield pickle.load(file)

    def drain_shards(self) -> Iterator[list[SampleRecord]]:
        shard_paths = list(self._shard_paths)
        self._shard_paths.clear()
        for shard_path in shard_paths:
            with shard_path.open("rb") as file:
                yield pickle.load(file)
            shard_path.unlink(missing_ok=True)

    @property
    def has_shards(self) -> bool:
        return bool(self._shard_paths)

    def cleanup(self) -> None:
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None
            return

        for shard_path in self._shard_paths:
            shard_path.unlink(missing_ok=True)
        self._shard_paths.clear()

    def __del__(self) -> None:
        self.cleanup()
