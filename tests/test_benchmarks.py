import argparse
import csv
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import SequentialSampler

from benchmarks import benchmark_lba, ddp_benchmark


def benchmark_args(tmpdir: str, **overrides):
    values = {
        "batch_size": 4,
        "num_workers": 0,
        "max_padded_length": None,
        "warmup_batches": 1,
        "max_cache_samples": 8,
        "max_padding_ratio": 0.05,
        "prefetch_batches": 0,
        "distributed_cost_window_batches": None,
        "planner_mode": "quality",
        "max_candidate_windows": None,
        "limited_search_fallback_after": None,
        "limited_search_fallback_pool_size": None,
        "simulate_gpu_sec": 0.0,
        "simulate_step_sec": 0.0,
        "compute_iters": 0,
        "rank_compute_iters": None,
        "rank_simulate_step_sec": None,
        "pin_memory": False,
        "drop_last_flush": False,
        "log_dir": tmpdir,
        "run_order": "alternate",
        "show_warnings": False,
        "repeats": 1,
        "warmup_runs": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class BenchmarkTest(unittest.TestCase):
    def test_text_line_dataset_reopens_reader_after_process_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.txt"
            path.write_text("one\ntwo words\n")

            for module in (benchmark_lba, ddp_benchmark):
                with self.subTest(module=module.__name__):
                    dataset = module.TextLineDataset(path, limit=None)
                    with patch.object(module.os, "getpid", return_value=os.getpid()):
                        self.assertEqual(dataset[0], "one\n")
                    first_reader = dataset._file
                    assert first_reader is not None

                    with patch.object(
                        module.os,
                        "getpid",
                        return_value=os.getpid() + 1,
                    ):
                        self.assertEqual(dataset[1], "two words\n")

                    self.assertTrue(first_reader.closed)
                    self.assertIsNot(dataset._file, first_reader)

    def test_single_process_run_records_effective_config_and_spill_stats(self) -> None:
        dataset = ["x " * (index % 4 + 1) for index in range(64)]
        with tempfile.TemporaryDirectory() as tmpdir:
            args = benchmark_args(tmpdir)
            rows = benchmark_lba.run_pair(
                "synthetic",
                dataset,
                args,
                repeat_index=0,
                order_index=0,
                measured=True,
            )
            output_path = Path(tmpdir) / "results.csv"
            benchmark_lba.write_results(output_path, rows)
            with output_path.open() as file:
                csv_rows = list(csv.DictReader(file))

        by_name = {row.name: row for row in rows}
        lba = by_name["lba"]
        self.assertEqual([row.name for row in rows], ["baseline", "lba"])
        self.assertGreater(lba.max_padded_length, 0)
        self.assertEqual(lba.warmup_batches, 1)
        self.assertEqual(lba.max_cache_samples, 8)
        self.assertGreater(lba.planner_records_sorted_total, 0)
        self.assertGreater(lba.planner_max_cache_size, 0)
        self.assertIn("planner_spill_events", csv_rows[0])
        self.assertIn("planner_spilled_records", csv_rows[0])

    def test_run_order_alternates(self) -> None:
        for module in (benchmark_lba, ddp_benchmark):
            with self.subTest(module=module.__name__):
                self.assertEqual(
                    module.loader_order(0, "alternate"),
                    ("baseline", "lba"),
                )
                self.assertEqual(
                    module.loader_order(1, "alternate"),
                    ("lba", "baseline"),
                )

    def test_workload_validation_is_strict_unless_drop_is_explicit(self) -> None:
        baseline = SimpleNamespace(name="baseline", samples=8, raw_length_sum=20)
        lba = SimpleNamespace(name="lba", samples=7, raw_length_sum=19)

        for module in (benchmark_lba, ddp_benchmark):
            with self.subTest(module=module.__name__):
                with self.assertRaisesRegex(RuntimeError, "workloads differ"):
                    module.validate_workload([baseline, lba])
                with self.assertWarnsRegex(UserWarning, "workloads differ"):
                    module.validate_workload(
                        [baseline, lba],
                        allow_sample_drop=True,
                    )

    def test_ddp_loader_defaults_to_strict_flush_and_resolved_planner_limits(self) -> None:
        dataset = ddp_benchmark.SyntheticLengthDataset(8, seed=123, max_length=32)
        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            args = benchmark_args(
                tmpdir,
                planner_mode="throughput",
                prefetch_batches=4,
                distributed_cost_window_batches=8,
            )
            with patch.object(
                ddp_benchmark,
                "DistributedSampler",
                return_value=SequentialSampler(dataset),
            ):
                loader = ddp_benchmark.build_loader("lba", dataset, args)

        self.assertFalse(loader.config.drop_last_flush)
        self.assertEqual(loader.config.candidate_window_limit, 256)
        self.assertEqual(loader.config.limited_search_fallback_after_limit, 8)
        self.assertEqual(loader.config.limited_search_fallback_pool_limit, 8)
        self.assertEqual(loader.config.distributed_cost_window_batches, 8)

    def test_ddp_result_uses_effective_config_and_aggregates_spill_stats(self) -> None:
        dataset = ddp_benchmark.SyntheticLengthDataset(8, seed=123, max_length=32)
        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            args = benchmark_args(
                tmpdir,
                planner_mode="throughput",
                prefetch_batches=4,
                max_padded_length=128,
            )
            model = ddp_benchmark.TokenWorkModel(compute_iters=0)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            with (
                patch.object(
                    ddp_benchmark,
                    "DistributedSampler",
                    return_value=SequentialSampler(dataset),
                ),
                patch.object(ddp_benchmark.dist, "get_rank", return_value=0),
                patch.object(ddp_benchmark.dist, "get_world_size", return_value=1),
                patch.object(ddp_benchmark.dist, "barrier"),
                patch.object(ddp_benchmark.dist, "all_reduce"),
                patch.object(ddp_benchmark.torch.cuda, "synchronize"),
            ):
                loader = ddp_benchmark.build_loader("lba", dataset, args)
                result = ddp_benchmark.run_loader(
                    "lba",
                    "synthetic",
                    len(dataset),
                    loader,
                    model,
                    optimizer,
                    torch.device("cpu"),
                    args,
                )

        self.assertIsNotNone(result)
        self.assertEqual(result.max_padded_length, 128)
        self.assertEqual(result.warmup_batches, 0)
        self.assertEqual(result.prefetch_batches, 4)
        self.assertIsNone(result.distributed_cost_window_batches)
        self.assertFalse(result.drop_last_flush)
        self.assertEqual(result.max_candidate_windows, 256)
        self.assertEqual(result.limited_search_fallback_after, 8)
        self.assertEqual(result.limited_search_fallback_pool_size, 8)
        self.assertGreater(result.planner_records_sorted_total, 0)
        self.assertGreater(result.planner_max_cache_size, 0)
        self.assertEqual(result.planner_spill_events, 0)
        self.assertEqual(result.planner_spilled_records, 0)
        self.assertEqual(result.rank_compute_iters_min, 0)
        self.assertEqual(result.rank_compute_iters_max, 0)
        self.assertEqual(result.step_compute_sec_spread, 0.0)

    def test_ddp_rank_profile_parsing(self) -> None:
        args = benchmark_args(
            "unused",
            compute_iters=4,
            simulate_step_sec=0.01,
            rank_compute_iters="4,16",
            rank_simulate_step_sec="0.0,0.02",
        )

        self.assertEqual(
            ddp_benchmark.local_compute_iters(args, rank=1, world_size=2),
            16,
        )
        self.assertEqual(
            ddp_benchmark.local_step_delay(args, rank=1, world_size=2),
            0.02,
        )
        with self.assertRaisesRegex(ValueError, "one value per rank"):
            ddp_benchmark.local_compute_iters(args, rank=0, world_size=3)

    def test_ddp_run_pair_does_not_hide_unexpected_warnings(self) -> None:
        args = benchmark_args("unused")

        def warn_and_return(*_args, **_kwargs):
            warnings.warn("final flush dropped records")
            return None

        with (
            patch.object(ddp_benchmark, "build_loader", return_value=object()),
            patch.object(ddp_benchmark, "run_loader", side_effect=warn_and_return),
            self.assertWarnsRegex(UserWarning, "final flush dropped records"),
        ):
            ddp_benchmark.run_pair(
                "synthetic",
                1,
                object(),
                object(),
                object(),
                object(),
                args,
                repeat_index=0,
                order_index=0,
                measured=True,
            )

    def test_rejects_invalid_repeat_counts(self) -> None:
        for module in (benchmark_lba, ddp_benchmark):
            with self.subTest(module=module.__name__):
                with self.assertRaisesRegex(ValueError, "--repeats"):
                    module.validate_run_args(
                        SimpleNamespace(repeats=0, warmup_runs=0)
                    )
                with self.assertRaisesRegex(ValueError, "--warmup-runs"):
                    module.validate_run_args(
                        SimpleNamespace(repeats=1, warmup_runs=-1)
                    )


if __name__ == "__main__":
    unittest.main()
