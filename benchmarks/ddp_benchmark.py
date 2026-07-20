"""Distributed benchmark for baseline DataLoader vs LBA."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from lba import LBA
from lba.budget import BudgetResolver
from lba.config import DEFAULT_PREFETCH_BATCHES
from lba.metrics import PlannerStats


class SyntheticLengthDataset(Dataset[int]):
    def __init__(self, size: int, seed: int, max_length: int) -> None:
        rng = random.Random(seed)
        self.lengths = [
            min(max_length, max(1, int(rng.lognormvariate(3.2, 1.0))))
            for _ in range(size)
        ]

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, index: int) -> int:
        return self.lengths[index]


class HuggingFaceTextDataset(Dataset[str]):
    def __init__(
        self,
        name: str,
        config: Optional[str],
        split: str,
        text_field: str,
        limit: int,
    ) -> None:
        from datasets import load_dataset

        dataset = load_dataset(name, config, split=split)
        texts: list[str] = []
        for row in dataset:
            text = str(row[text_field]).strip()
            if not text:
                continue
            texts.append(text)
            if len(texts) >= limit:
                break
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> str:
        return self.texts[index]


class TextLineDataset(Dataset[str]):
    def __init__(self, path: Path, limit: Optional[int]) -> None:
        self.path = path
        self.offsets: list[int] = []
        self._file: Optional[BinaryIO] = None
        self._file_pid: Optional[int] = None

        with path.open("rb") as file:
            while True:
                offset = file.tell()
                line = file.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
                    if limit is not None and len(self.offsets) >= limit:
                        break

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> str:
        pid = os.getpid()
        if self._file is None or self._file_pid != pid:
            if self._file is not None:
                self._file.close()
            self._file = self.path.open("rb")
            self._file_pid = pid
        self._file.seek(self.offsets[index])
        return self._file.readline().decode("utf-8", errors="replace")

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        state["_file_pid"] = None
        return state


class TokenWorkModel(nn.Module):
    def __init__(self, compute_iters: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.compute_iters = compute_iters

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        values = tokens
        for _ in range(self.compute_iters):
            values = values * self.scale
        return values.mean() * self.scale


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    dataset: str
    repeat_index: int
    run_position: int
    world_size: int
    dataset_size: int
    batch_size: int
    num_workers: int
    max_padded_length: Optional[int]
    warmup_batches: Optional[int]
    max_cache_samples: int
    max_padding_ratio: Optional[float]
    prefetch_batches: int
    drop_last_flush: bool
    compute_iters: int
    simulate_step_sec: float
    elapsed_sec: float
    time_to_first_batch_sec: float
    loader_wait_sec_sum: float
    step_compute_sec_sum: float
    samples: int
    batches: int
    steps_per_rank: float
    mean_batch_size: float
    samples_per_sec: float
    raw_tokens_per_sec: float
    padded_tokens_per_sec: float
    raw_length_sum: int
    padded_length_sum: int
    padding_length_sum: int
    padding_ratio: float
    planner_sort_time_sec: float
    planner_sort_calls: int
    planner_records_sorted_total: int
    planner_max_cache_size: int
    planner_spill_events: int
    planner_spilled_records: int
    planner_pop_ready_time_sec: float
    planner_pop_ready_calls: int
    planner_avg_pop_ready_ms: float
    planner_candidate_window_checks: int
    planner_avg_candidate_window_checks: float
    planner_max_candidate_window_checks: int
    planner_fast_path_batches: int
    planner_fallback_search_batches: int
    planner_flush_search_batches: int
    planner_oversized_batches: int
    planner_no_ready_calls: int
    planner_fast_path_time_sec: float
    planner_fallback_search_time_sec: float
    planner_flush_search_time_sec: float
    planner_oversized_time_sec: float
    planner_no_ready_time_sec: float
    planner_fast_path_candidate_window_checks: int
    planner_fallback_search_candidate_window_checks: int
    planner_flush_search_candidate_window_checks: int
    planner_mode: str
    max_candidate_windows: Optional[int]
    limited_search_fallback_after: Optional[int]
    limited_search_fallback_pool_size: Optional[int]


def sample_length(sample: Any) -> int:
    if isinstance(sample, int):
        return sample
    if isinstance(sample, str):
        return max(1, len(sample.split()))
    raise TypeError(f"Unsupported sample type: {type(sample)!r}")


def metric_collate(samples: list[Any]) -> dict[str, Any]:
    lengths = [sample_length(sample) for sample in samples]
    max_length = max(lengths)
    raw_length = sum(lengths)
    padded_length = max_length * len(samples)
    return {
        "samples": len(samples),
        "raw_length": raw_length,
        "padded_length": padded_length,
        "padding_length": padded_length - raw_length,
        "tokens": torch.ones((len(samples), max_length), dtype=torch.float32),
    }


def build_loader(
    name: str,
    dataset: Dataset,
    args: argparse.Namespace,
) -> Any:
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    if name == "baseline":
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=metric_collate,
            pin_memory=args.pin_memory,
        )
    if name == "lba":
        return LBA(
            dataset,
            len_fn=sample_length,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=metric_collate,
            pin_memory=args.pin_memory,
            max_padded_length=args.max_padded_length,
            warmup_batches=args.warmup_batches,
            max_cache_samples=args.max_cache_samples,
            max_padding_ratio=args.max_padding_ratio,
            prefetch_batches=args.prefetch_batches,
            planner_mode=args.planner_mode,
            max_candidate_windows=args.max_candidate_windows,
            limited_search_fallback_after=args.limited_search_fallback_after,
            limited_search_fallback_pool_size=args.limited_search_fallback_pool_size,
            drop_last_flush=args.drop_last_flush,
            log_dir=args.log_dir,
        )
    raise ValueError(f"Unknown loader name: {name}")


def planner_stats_from_loader(loader: Any) -> PlannerStats:
    if isinstance(loader, LBA):
        return loader.last_planner_stats
    return PlannerStats()


def resolved_max_padded_length(loader: Any) -> Optional[int]:
    if not isinstance(loader, LBA):
        return None
    if loader.last_max_padded_length is None:
        raise RuntimeError("LBA benchmark run did not resolve max_padded_length.")
    return loader.last_max_padded_length


def effective_warmup_batches(loader: Any) -> Optional[int]:
    if not isinstance(loader, LBA):
        return None
    if loader.config.max_padded_length is not None:
        return 0
    return BudgetResolver(loader.config, loader).warmup_batch_count()


def run_loader(
    name: str,
    dataset_name: str,
    dataset_size: int,
    loader: Any,
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    *,
    repeat_index: int = 0,
    run_position: int = 0,
    simulate_step_sec: Optional[float] = None,
) -> Optional[BenchmarkResult]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    step_delay = (
        args.simulate_step_sec
        if simulate_step_sec is None
        else simulate_step_sec
    )
    dist.barrier()
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    first_batch_time: Optional[float] = None
    loader_wait_sec = 0.0
    step_compute_sec = 0.0
    samples = 0
    batches = 0
    raw_length_sum = 0
    padded_length_sum = 0
    padding_length_sum = 0

    iterator = iter(loader)
    while True:
        wait_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        wait_end = time.perf_counter()
        loader_wait_sec += wait_end - wait_start
        if first_batch_time is None:
            first_batch_time = wait_end - start

        tokens = batch["tokens"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = model(tokens)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        if step_delay > 0:
            time.sleep(step_delay)
        step_compute_sec += time.perf_counter() - wait_end

        batches += 1
        samples += batch["samples"]
        raw_length_sum += batch["raw_length"]
        padded_length_sum += batch["padded_length"]
        padding_length_sum += batch["padding_length"]

    torch.cuda.synchronize(device)
    elapsed_sec = time.perf_counter() - start
    planner_stats = planner_stats_from_loader(loader)

    sum_values = torch.tensor(
        [
            loader_wait_sec,
            step_compute_sec,
            float(samples),
            float(batches),
            float(raw_length_sum),
            float(padded_length_sum),
            float(padding_length_sum),
            planner_stats.sort_time_seconds,
            float(planner_stats.sort_call_count),
            planner_stats.pop_ready_time_seconds,
            float(planner_stats.pop_ready_call_count),
            float(planner_stats.candidate_window_checks),
            float(planner_stats.fast_path_batch_count),
            float(planner_stats.fallback_search_batch_count),
            float(planner_stats.flush_search_batch_count),
            float(planner_stats.oversized_batch_count),
            float(planner_stats.no_ready_call_count),
            planner_stats.fast_path_time_seconds,
            planner_stats.fallback_search_time_seconds,
            planner_stats.flush_search_time_seconds,
            planner_stats.oversized_time_seconds,
            planner_stats.no_ready_time_seconds,
            float(planner_stats.fast_path_candidate_window_checks),
            float(planner_stats.fallback_search_candidate_window_checks),
            float(planner_stats.flush_search_candidate_window_checks),
            float(planner_stats.records_sorted_total),
            float(planner_stats.spill_event_count),
            float(planner_stats.spilled_record_count),
        ],
        dtype=torch.float64,
        device=device,
    )
    max_values = torch.tensor(
        [
            elapsed_sec,
            first_batch_time or 0.0,
            float(planner_stats.max_candidate_window_checks),
            float(planner_stats.max_cache_size_seen),
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(sum_values, op=dist.ReduceOp.SUM)
    dist.all_reduce(max_values, op=dist.ReduceOp.MAX)

    if rank != 0:
        return None

    total_samples = int(sum_values[2].item())
    total_batches = int(sum_values[3].item())
    total_raw_length = int(sum_values[4].item())
    total_padded_length = int(sum_values[5].item())
    total_padding_length = int(sum_values[6].item())
    total_planner_sort_time = float(sum_values[7].item())
    total_planner_sort_calls = int(sum_values[8].item())
    total_planner_pop_ready_time = float(sum_values[9].item())
    total_planner_pop_ready_calls = int(sum_values[10].item())
    total_candidate_window_checks = int(sum_values[11].item())
    total_fast_path_batches = int(sum_values[12].item())
    total_fallback_search_batches = int(sum_values[13].item())
    total_flush_search_batches = int(sum_values[14].item())
    total_oversized_batches = int(sum_values[15].item())
    total_no_ready_calls = int(sum_values[16].item())
    total_fast_path_time = float(sum_values[17].item())
    total_fallback_search_time = float(sum_values[18].item())
    total_flush_search_time = float(sum_values[19].item())
    total_oversized_time = float(sum_values[20].item())
    total_no_ready_time = float(sum_values[21].item())
    total_fast_path_candidate_window_checks = int(sum_values[22].item())
    total_fallback_search_candidate_window_checks = int(sum_values[23].item())
    total_flush_search_candidate_window_checks = int(sum_values[24].item())
    total_records_sorted = int(sum_values[25].item())
    total_spill_events = int(sum_values[26].item())
    total_spilled_records = int(sum_values[27].item())
    max_elapsed = float(max_values[0].item())
    max_candidate_window_checks = int(max_values[2].item())
    max_cache_size = int(max_values[3].item())
    padding_ratio = (
        total_padding_length / total_padded_length if total_padded_length else 0.0
    )
    config = loader.config if isinstance(loader, LBA) else None
    return BenchmarkResult(
        name=name,
        dataset=dataset_name,
        repeat_index=repeat_index,
        run_position=run_position,
        world_size=world_size,
        dataset_size=dataset_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_padded_length=resolved_max_padded_length(loader),
        warmup_batches=effective_warmup_batches(loader),
        max_cache_samples=config.max_cache_samples if config is not None else 0,
        max_padding_ratio=config.max_padding_ratio if config is not None else None,
        prefetch_batches=0,
        drop_last_flush=config.drop_last_flush if config is not None else False,
        compute_iters=args.compute_iters,
        simulate_step_sec=step_delay,
        elapsed_sec=max_elapsed,
        time_to_first_batch_sec=float(max_values[1].item()),
        loader_wait_sec_sum=float(sum_values[0].item()),
        step_compute_sec_sum=float(sum_values[1].item()),
        samples=total_samples,
        batches=total_batches,
        steps_per_rank=total_batches / world_size if world_size else 0.0,
        mean_batch_size=total_samples / total_batches if total_batches else 0.0,
        samples_per_sec=total_samples / max_elapsed if max_elapsed else 0.0,
        raw_tokens_per_sec=total_raw_length / max_elapsed if max_elapsed else 0.0,
        padded_tokens_per_sec=(
            total_padded_length / max_elapsed if max_elapsed else 0.0
        ),
        raw_length_sum=total_raw_length,
        padded_length_sum=total_padded_length,
        padding_length_sum=total_padding_length,
        padding_ratio=padding_ratio,
        planner_sort_time_sec=total_planner_sort_time,
        planner_sort_calls=total_planner_sort_calls,
        planner_records_sorted_total=total_records_sorted,
        planner_max_cache_size=max_cache_size,
        planner_spill_events=total_spill_events,
        planner_spilled_records=total_spilled_records,
        planner_pop_ready_time_sec=total_planner_pop_ready_time,
        planner_pop_ready_calls=total_planner_pop_ready_calls,
        planner_avg_pop_ready_ms=(
            total_planner_pop_ready_time * 1000 / total_planner_pop_ready_calls
            if total_planner_pop_ready_calls
            else 0.0
        ),
        planner_candidate_window_checks=total_candidate_window_checks,
        planner_avg_candidate_window_checks=(
            total_candidate_window_checks / total_planner_pop_ready_calls
            if total_planner_pop_ready_calls
            else 0.0
        ),
        planner_max_candidate_window_checks=max_candidate_window_checks,
        planner_fast_path_batches=total_fast_path_batches,
        planner_fallback_search_batches=total_fallback_search_batches,
        planner_flush_search_batches=total_flush_search_batches,
        planner_oversized_batches=total_oversized_batches,
        planner_no_ready_calls=total_no_ready_calls,
        planner_fast_path_time_sec=total_fast_path_time,
        planner_fallback_search_time_sec=total_fallback_search_time,
        planner_flush_search_time_sec=total_flush_search_time,
        planner_oversized_time_sec=total_oversized_time,
        planner_no_ready_time_sec=total_no_ready_time,
        planner_fast_path_candidate_window_checks=(
            total_fast_path_candidate_window_checks
        ),
        planner_fallback_search_candidate_window_checks=(
            total_fallback_search_candidate_window_checks
        ),
        planner_flush_search_candidate_window_checks=(
            total_flush_search_candidate_window_checks
        ),
        planner_mode=config.planner_mode if config is not None else "baseline",
        max_candidate_windows=(
            config.candidate_window_limit if config is not None else None
        ),
        limited_search_fallback_after=(
            config.limited_search_fallback_after_limit
            if config is not None
            else None
        ),
        limited_search_fallback_pool_size=(
            config.limited_search_fallback_pool_limit
            if config is not None
            else None
        ),
    )


def write_csv(path: Path, rows: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def loader_order(run_index: int, run_order: str) -> tuple[str, str]:
    if run_order == "baseline-first":
        return "baseline", "lba"
    if run_order == "lba-first":
        return "lba", "baseline"
    if run_order == "alternate":
        if run_index % 2 == 0:
            return "baseline", "lba"
        return "lba", "baseline"
    raise ValueError(f"Unknown run order: {run_order}")


def validate_workload(
    rows: list[BenchmarkResult], *, allow_sample_drop: bool = False
) -> None:
    by_name = {row.name: row for row in rows}
    baseline = by_name["baseline"]
    lba = by_name["lba"]
    if (
        baseline.samples == lba.samples
        and baseline.raw_length_sum == lba.raw_length_sum
    ):
        return

    message = (
        "Benchmark workloads differ: "
        f"baseline samples/raw_length={baseline.samples}/{baseline.raw_length_sum}, "
        f"LBA={lba.samples}/{lba.raw_length_sum}."
    )
    if allow_sample_drop:
        warnings.warn(message, stacklevel=2)
        return
    raise RuntimeError(message)


def run_pair(
    dataset_name: str,
    dataset_size: int,
    dataset: Dataset,
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    *,
    repeat_index: int,
    order_index: int,
    measured: bool,
) -> list[BenchmarkResult]:
    rows: list[BenchmarkResult] = []
    for run_position, name in enumerate(loader_order(order_index, args.run_order)):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"LBA log file:")
            warnings.filterwarnings(
                "ignore",
                message=r"max_padded_length is set explicitly",
            )
            loader = build_loader(name, dataset, args)
            result = run_loader(
                name,
                dataset_name,
                dataset_size,
                loader,
                model,
                optimizer,
                device,
                args,
                repeat_index=repeat_index,
                run_position=run_position,
                simulate_step_sec=args.simulate_step_sec if measured else 0.0,
            )
        if result is not None:
            rows.append(result)
    return rows


def validate_run_args(args: argparse.Namespace) -> None:
    if args.repeats <= 0:
        raise ValueError("--repeats must be a positive integer.")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative.")


def build_dataset(args: argparse.Namespace) -> tuple[str, Dataset]:
    if args.dataset == "synthetic":
        return (
            "synthetic",
            SyntheticLengthDataset(args.size, args.seed, args.max_length),
        )
    if args.dataset == "text-file":
        if args.text_file is None:
            raise ValueError("--text-file is required for text-file dataset.")
        path = Path(args.text_file)
        return path.name, TextLineDataset(path, args.size)
    if args.dataset == "hf":
        dataset_name = args.hf_name
        if args.hf_config is not None:
            dataset_name = f"{dataset_name}/{args.hf_config}"
        return (
            dataset_name,
            HuggingFaceTextDataset(
                args.hf_name,
                args.hf_config,
                args.hf_split,
                args.text_field,
                args.size,
            ),
        )
    raise ValueError(f"Unknown dataset: {args.dataset}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "text-file", "hf"],
        default="synthetic",
    )
    parser.add_argument("--text-file")
    parser.add_argument("--size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--hf-name", default="wikitext")
    parser.add_argument("--hf-config", default="wikitext-2-raw-v1")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-padded-length", type=int)
    parser.add_argument("--warmup-batches", type=int)
    parser.add_argument("--max-cache-samples", type=int, default=8192)
    parser.add_argument("--max-padding-ratio", type=float, default=0.05)
    parser.add_argument("--prefetch-batches", type=int, default=DEFAULT_PREFETCH_BATCHES)
    parser.add_argument("--planner-mode", choices=["quality", "throughput"], default="quality")
    parser.add_argument("--max-candidate-windows", type=int)
    parser.add_argument("--limited-search-fallback-after", type=int)
    parser.add_argument("--limited-search-fallback-pool-size", type=int)
    parser.add_argument("--compute-iters", type=int, default=4)
    parser.add_argument("--simulate-step-sec", type=float, default=0.0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--drop-last-flush",
        action="store_true",
        help="Allow LBA to drop a DDP final-flush tail; strict conservation is default.",
    )
    parser.add_argument("--log-dir", default="outputs/lba_logs")
    parser.add_argument("--output", default="outputs/ddp_benchmark.csv")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument(
        "--run-order",
        choices=["alternate", "baseline-first", "lba-first"],
        default="alternate",
    )
    args = parser.parse_args()
    validate_run_args(args)

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dataset_name, dataset = build_dataset(args)
    dataset_size = len(dataset)
    model = DistributedDataParallel(
        TokenWorkModel(args.compute_iters).to(device),
        device_ids=[local_rank],
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    rows: list[BenchmarkResult] = []

    for warmup_index in range(args.warmup_runs):
        run_pair(
            dataset_name,
            dataset_size,
            dataset,
            model,
            optimizer,
            device,
            args,
            repeat_index=warmup_index,
            order_index=warmup_index,
            measured=False,
        )
    for repeat_index in range(args.repeats):
        rows.extend(
            run_pair(
                dataset_name,
                dataset_size,
                dataset,
                model,
                optimizer,
                device,
                args,
                repeat_index=repeat_index,
                order_index=args.warmup_runs + repeat_index,
                measured=True,
            )
        )

    rank = dist.get_rank()
    validation_error: Optional[RuntimeError] = None
    if rank == 0:
        try:
            for repeat_index in range(args.repeats):
                validate_workload(
                    [row for row in rows if row.repeat_index == repeat_index],
                    allow_sample_drop=args.drop_last_flush,
                )
        except RuntimeError as error:
            validation_error = error

    dist.barrier()
    dist.destroy_process_group()

    if validation_error is not None:
        raise validation_error
    if rank == 0:
        output_path = Path(args.output)
        write_csv(output_path, rows)
        print(json.dumps([asdict(row) for row in rows], indent=2))
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
