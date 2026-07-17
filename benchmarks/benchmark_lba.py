"""Benchmark LBA batching on variable-length text samples."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from torch.utils.data import DataLoader, Dataset

from lba import LBA
from lba.config import DEFAULT_PREFETCH_BATCHES
from lba.metrics import PlannerStats


def sample_length(sample: str) -> int:
    return max(1, len(sample.split()))


class SyntheticTextDataset(Dataset):
    def __init__(self, size: int, seed: int) -> None:
        rng = random.Random(seed)
        self.lengths = [
            max(1, int(rng.lognormvariate(3.2, 1.0)))
            for _ in range(size)
        ]

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, index: int) -> str:
        return "x " * self.lengths[index]


class TextLineDataset(Dataset):
    def __init__(self, path: Path, limit: Optional[int] = None) -> None:
        self.path = path
        self.offsets: list[int] = []
        self._file = None

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
        if self._file is None:
            self._file = self.path.open("rb")
        self._file.seek(self.offsets[index])
        return self._file.readline().decode("utf-8", errors="replace")

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        return state


class HuggingFaceTextDataset(Dataset):
    def __init__(
        self,
        name: str,
        config: Optional[str],
        split: str,
        text_field: str,
        limit: Optional[int],
    ) -> None:
        from datasets import load_dataset

        dataset = load_dataset(name, config, split=split)
        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))
        self.dataset = dataset
        self.text_field = text_field

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> str:
        return str(self.dataset[index][self.text_field])


@dataclass
class BenchmarkResult:
    name: str
    dataset: str
    samples: int
    batches: int
    elapsed_sec: float
    time_to_first_batch_sec: float
    loader_wait_sec: float
    loader_wait_per_batch_sec: float
    simulated_gpu_sec: float
    samples_per_sec: float
    raw_length_sum: int
    padded_length_sum: int
    padding_length_sum: int
    padding_ratio: float
    planner_sort_time_sec: float
    planner_sort_calls: int
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


def metric_collate(samples: list[str]) -> dict[str, Any]:
    lengths = [sample_length(sample) for sample in samples]
    max_length = max(lengths)
    padded_length = max_length * len(lengths)
    raw_length = sum(lengths)
    return {
        "samples": len(samples),
        "lengths": lengths,
        "raw_length": raw_length,
        "padded_length": padded_length,
        "padding_length": padded_length - raw_length,
    }


def consume(
    name: str,
    dataset_name: str,
    loader: Any,
    *,
    simulated_gpu_sec: float = 0.0,
) -> BenchmarkResult:
    start = time.perf_counter()
    first_batch_time: Optional[float] = None
    loader_wait_sec = 0.0
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
        now = time.perf_counter()
        loader_wait_sec += now - wait_start
        if first_batch_time is None:
            first_batch_time = now - start
        batches += 1
        samples += batch["samples"]
        raw_length_sum += batch["raw_length"]
        padded_length_sum += batch["padded_length"]
        padding_length_sum += batch["padding_length"]
        if simulated_gpu_sec > 0:
            time.sleep(simulated_gpu_sec)

    elapsed = time.perf_counter() - start
    padding_ratio = padding_length_sum / padded_length_sum if padded_length_sum else 0.0
    planner_stats = planner_stats_from_loader(loader)
    return BenchmarkResult(
        name=name,
        dataset=dataset_name,
        samples=samples,
        batches=batches,
        elapsed_sec=elapsed,
        time_to_first_batch_sec=first_batch_time or 0.0,
        loader_wait_sec=loader_wait_sec,
        loader_wait_per_batch_sec=loader_wait_sec / batches if batches else 0.0,
        simulated_gpu_sec=simulated_gpu_sec,
        samples_per_sec=samples / elapsed if elapsed else 0.0,
        raw_length_sum=raw_length_sum,
        padded_length_sum=padded_length_sum,
        padding_length_sum=padding_length_sum,
        padding_ratio=padding_ratio,
        planner_sort_time_sec=planner_stats.sort_time_seconds,
        planner_sort_calls=planner_stats.sort_call_count,
        planner_pop_ready_time_sec=planner_stats.pop_ready_time_seconds,
        planner_pop_ready_calls=planner_stats.pop_ready_call_count,
        planner_avg_pop_ready_ms=planner_stats.average_pop_ready_time_ms or 0.0,
        planner_candidate_window_checks=planner_stats.candidate_window_checks,
        planner_avg_candidate_window_checks=(
            planner_stats.average_candidate_window_checks or 0.0
        ),
        planner_max_candidate_window_checks=planner_stats.max_candidate_window_checks,
        planner_fast_path_batches=planner_stats.fast_path_batch_count,
        planner_fallback_search_batches=planner_stats.fallback_search_batch_count,
        planner_flush_search_batches=planner_stats.flush_search_batch_count,
        planner_oversized_batches=planner_stats.oversized_batch_count,
        planner_no_ready_calls=planner_stats.no_ready_call_count,
        planner_fast_path_time_sec=planner_stats.fast_path_time_seconds,
        planner_fallback_search_time_sec=planner_stats.fallback_search_time_seconds,
        planner_flush_search_time_sec=planner_stats.flush_search_time_seconds,
        planner_oversized_time_sec=planner_stats.oversized_time_seconds,
        planner_no_ready_time_sec=planner_stats.no_ready_time_seconds,
        planner_fast_path_candidate_window_checks=(
            planner_stats.fast_path_candidate_window_checks
        ),
        planner_fallback_search_candidate_window_checks=(
            planner_stats.fallback_search_candidate_window_checks
        ),
        planner_flush_search_candidate_window_checks=(
            planner_stats.flush_search_candidate_window_checks
        ),
        planner_mode=getattr(loader, "config", None).planner_mode
        if isinstance(loader, LBA)
        else "baseline",
        max_candidate_windows=getattr(loader, "config", None).candidate_window_limit
        if isinstance(loader, LBA)
        else None,
        limited_search_fallback_after=(
            getattr(loader, "config", None).limited_search_fallback_after_limit
            if isinstance(loader, LBA)
            else None
        ),
        limited_search_fallback_pool_size=(
            getattr(loader, "config", None).limited_search_fallback_pool_limit
            if isinstance(loader, LBA)
            else None
        ),
    )


def planner_stats_from_loader(loader: Any) -> PlannerStats:
    if isinstance(loader, LBA):
        return loader.last_planner_stats
    return PlannerStats()


def build_dataset(args: argparse.Namespace) -> tuple[str, Dataset]:
    if args.dataset == "synthetic":
        return "synthetic", SyntheticTextDataset(args.limit, args.seed)
    if args.dataset == "text-file":
        if args.text_file is None:
            raise ValueError("--text-file is required for text-file dataset.")
        path = Path(args.text_file)
        return path.name, TextLineDataset(path, args.limit)
    if args.dataset == "hf":
        return (
            args.hf_name,
            HuggingFaceTextDataset(
                args.hf_name,
                args.hf_config,
                args.hf_split,
                args.text_field,
                args.limit,
            ),
        )
    raise ValueError(f"Unknown dataset: {args.dataset}")


def write_results(path: Path, rows: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synthetic", "text-file", "hf"], default="synthetic")
    parser.add_argument("--text-file")
    parser.add_argument("--hf-name", default="wikitext")
    parser.add_argument("--hf-config", default="wikitext-2-raw-v1")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--limit", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=123)
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
    parser.add_argument("--simulate-gpu-sec", type=float, default=0.0)
    parser.add_argument("--log-dir", default="outputs/lba_logs")
    parser.add_argument("--output", default="outputs/lba_benchmark.csv")
    parser.add_argument("--show-warnings", action="store_true")
    args = parser.parse_args()

    dataset_name, dataset = build_dataset(args)
    baseline_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=metric_collate,
    )
    lba_source_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=metric_collate,
    )
    warning_context = contextlib.nullcontext()
    if not args.show_warnings:
        warning_context = warnings.catch_warnings()

    with warning_context:
        if not args.show_warnings:
            warnings.simplefilter("ignore")
        lba_loader = LBA(
            lba_source_loader,
            len_fn=sample_length,
            max_padded_length=args.max_padded_length,
            warmup_batches=args.warmup_batches,
            max_cache_samples=args.max_cache_samples,
            max_padding_ratio=args.max_padding_ratio,
            prefetch_batches=args.prefetch_batches,
            planner_mode=args.planner_mode,
            max_candidate_windows=args.max_candidate_windows,
            limited_search_fallback_after=args.limited_search_fallback_after,
            limited_search_fallback_pool_size=args.limited_search_fallback_pool_size,
            log_dir=args.log_dir,
        )
        rows = [
            consume(
                "baseline",
                dataset_name,
                baseline_loader,
                simulated_gpu_sec=args.simulate_gpu_sec,
            ),
            consume(
                "lba",
                dataset_name,
                lba_loader,
                simulated_gpu_sec=args.simulate_gpu_sec,
            ),
        ]
    write_results(Path(args.output), rows)
    print(json.dumps([asdict(row) for row in rows], indent=2))


if __name__ == "__main__":
    main()
