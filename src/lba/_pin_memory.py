"""Pin final collated batches using the wrapped DataLoader contract."""

from __future__ import annotations

import warnings
from typing import Any, Optional

import torch
from torch.utils.data._utils.pin_memory import pin_memory


def pin_memory_enabled(*, requested: bool, device: Optional[str]) -> bool:
    if not requested:
        return False
    if device:
        return True

    accelerator = getattr(torch, "accelerator", None)
    if accelerator is None:
        available = torch.cuda.is_available()
        accelerator_type = "cuda" if available else None
    else:
        available = accelerator.is_available()
        current = accelerator.current_accelerator() if available else None
        accelerator_type = current.type if current is not None else None

    if not available:
        warnings.warn(
            "LBA pin_memory=True has no available accelerator; final batches "
            "will not use pinned memory.",
            stacklevel=3,
        )
        return False
    if accelerator_type == "mps":
        warnings.warn(
            "LBA pin_memory=True is not supported by the MPS backend; final "
            "batches will not use pinned memory.",
            stacklevel=3,
        )
        return False
    return True


def pin_batch(batch: Any, *, enabled: bool, device: Optional[str]) -> Any:
    if not enabled:
        return batch
    if device:
        return pin_memory(batch, device)
    return pin_memory(batch)
