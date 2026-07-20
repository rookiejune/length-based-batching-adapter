"""Small DDP smoke script for LBA.

This intentionally creates different length distributions on each rank so a
DDP run exposes rank-local dynamic-batch count mismatches.
"""

from __future__ import annotations

import os
import tempfile

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DistributedSampler

from lba import LBA


class LengthDataset(Dataset[int]):
    def __init__(self) -> None:
        self.lengths = [100, 1, 100, 1, 100, 1, 100, 1]

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, index: int) -> int:
        return self.lengths[index]


def collate_lengths(samples: list[int]) -> torch.Tensor:
    return torch.tensor(samples, dtype=torch.float32).unsqueeze(1)


def sample_length(sample: int) -> int:
    return sample


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dataset = LengthDataset()
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    loader = LBA(
        dataset,
        len_fn=sample_length,
        batch_size=2,
        sampler=sampler,
        collate_fn=collate_lengths,
        num_workers=0,
        max_padded_length=100,
        max_padding_ratio=0.0,
        log_dir=tempfile.mkdtemp(prefix=f"lba-ddp-rank{dist.get_rank()}-"),
    )

    model = DistributedDataParallel(
        nn.Linear(1, 1).to(device),
        device_ids=[local_rank],
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    step_count = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch.to(device)).sum()
        loss.backward()
        optimizer.step()
        print(
            f"rank={dist.get_rank()} step={step_count} batch_size={len(batch)}",
            flush=True,
        )
        step_count += 1

    print(f"rank={dist.get_rank()} finished steps={step_count}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
