"""Callable type aliases for LBA public entry points."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol


LengthFn = Callable[[Any], int]
CostFn = Callable[[int, int], int]
CollateFn = Callable[[list[Any]], Any]


class EventWriter(Protocol):
    def write(self, event: str, fields: Mapping[str, Any]) -> None: ...
