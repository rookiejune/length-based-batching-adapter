"""Callable type aliases for LBA public entry points."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


LengthFn = Callable[[Any], int]
CollateFn = Callable[[list[Any]], Any]
