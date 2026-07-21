import gc
import json
import os
import tempfile
import threading
import unittest
import warnings
from pathlib import Path
from unittest import mock

import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, IterableDataset

from lba import LBA
from lba._iteration import Iteration
from lba.config import DEFAULT_PREFETCH_BATCHES
from lba.planner import BatchPlanner
from lba.prefetch import prefetch_iterator
from lba.source import build_source_loader


def identity_collate(samples):
    return samples


class SequenceIterableDataset(IterableDataset):
    def __init__(self, samples):
        self.samples = samples

    def __iter__(self):
        yield from self.samples


class PidDataset(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> int:
        return os.getpid()


def one_length(sample: int) -> int:
    return 1


class LBATest(unittest.TestCase):
    def test_constructor_keeps_inputs(self) -> None:
        dataset = [[0] * 5, [1] * 5]

        def len_fn(sample):
            return len(sample)

        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter = LBA(
                dataset,
                len_fn=len_fn,
                batch_size=2,
                collate_fn=identity_collate,
                max_padded_length=128,
                log_dir=tmpdir,
            )

        self.assertIs(adapter.dataset, dataset)
        self.assertIs(adapter.len_fn, len_fn)
        self.assertIs(adapter.collate_fn, identity_collate)
        self.assertEqual(adapter.batch_size, 2)
        self.assertEqual(adapter.max_padded_length, 128)
        self.assertEqual(adapter.config.prefetch_batches, DEFAULT_PREFETCH_BATCHES)
        self.assertEqual(adapter.config.planner_mode, "quality")
        self.assertIsNone(adapter.config.candidate_window_limit)

    def test_constructor_keeps_throughput_planner_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter = LBA(
                [[0]],
                len_fn=len,
                batch_size=1,
                collate_fn=identity_collate,
                max_padded_length=10,
                planner_mode="throughput",
                max_candidate_windows=128,
                log_dir=tmpdir,
            )

        self.assertEqual(adapter.config.planner_mode, "throughput")
        self.assertEqual(adapter.config.candidate_window_limit, 128)

    def test_constructor_does_not_create_run_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = LBA(
                [[0]],
                len_fn=len,
                batch_size=1,
                max_padded_length=1,
                log_dir=tmpdir,
            )

            self.assertIsNone(adapter._run_logger)
            self.assertEqual(list(Path(tmpdir).iterdir()), [])

    def test_length_is_unavailable_for_dynamic_batches(self) -> None:
        adapter = LBA([[0]], len_fn=len, batch_size=1, max_padded_length=1)

        with self.assertRaisesRegex(TypeError, "dynamic and unavailable"):
            len(adapter)

    def test_iterates_dynamic_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter = LBA(
                [[0] * 5, [1] * 5, [2] * 4, [3] * 4],
                len_fn=len,
                batch_size=2,
                collate_fn=identity_collate,
                max_padded_length=10,
                max_padding_ratio=0.0,
                log_dir=tmpdir,
            )
            batches = list(adapter)

        self.assertEqual([len(batch) for batch in batches], [2, 2])
        self.assertEqual([len(sample) for sample in batches[0]], [5, 5])
        self.assertGreater(adapter.last_planner_stats.pop_ready_call_count, 0)
        self.assertEqual(adapter.last_max_padded_length, 10)

    def test_prefetch_iterates_dynamic_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter = LBA(
                [[0] * 5, [1] * 5, [2] * 4, [3] * 4],
                len_fn=len,
                batch_size=2,
                collate_fn=identity_collate,
                max_padded_length=10,
                max_padding_ratio=0.0,
                prefetch_batches=2,
                log_dir=tmpdir,
            )
            batches = list(adapter)

        self.assertEqual([len(batch) for batch in batches], [2, 2])
        self.assertEqual([len(sample) for sample in batches[0]], [5, 5])

    def test_distributed_prefetch_uses_background_iterator(self) -> None:
        coordinator = mock.Mock()

        def prefetched(iterator, max_batches):
            self.assertGreater(max_batches, 0)
            return iter([("prefetched", max_batches)])

        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter = LBA(
                [[0] * 5, [1] * 5],
                len_fn=len,
                batch_size=2,
                collate_fn=identity_collate,
                max_padded_length=10,
                prefetch_batches=2,
                log_dir=tmpdir,
            )

            with (
                mock.patch(
                    "lba.loader.DistributedBatchCoordinator.is_initialized",
                    return_value=True,
                ),
                mock.patch.object(
                    adapter,
                    "_ensure_distributed",
                    return_value=coordinator,
                ),
                mock.patch(
                    "lba.loader.prefetch_iterator",
                    side_effect=prefetched,
                ) as prefetch,
            ):
                batches = list(adapter)

        self.assertEqual(batches, [("prefetched", 2)])
        coordinator.prepare_for_background_iteration.assert_called_once_with()
        prefetch.assert_called_once()

    def test_prefetch_close_while_source_is_running_does_not_raise(self) -> None:
        entered_second = threading.Event()
        release_second = threading.Event()
        closed = threading.Event()

        def source():
            try:
                yield "first"
                entered_second.set()
                release_second.wait(timeout=5)
                yield "second"
            finally:
                closed.set()

        iterator = prefetch_iterator(source(), 1)

        self.assertEqual(next(iterator), "first")
        self.assertTrue(entered_second.wait(timeout=2))
        try:
            with self.assertWarnsRegex(RuntimeWarning, "producer is still blocked"):
                iterator.close()
        finally:
            release_second.set()

        self.assertTrue(closed.wait(timeout=2))

    def test_prefetch_propagates_source_error(self) -> None:
        def source():
            yield "first"
            raise ValueError("source failed")

        iterator = prefetch_iterator(source(), 1)

        self.assertEqual(next(iterator), "first")
        with self.assertRaisesRegex(ValueError, "source failed"):
            next(iterator)

    def test_source_loader_starts_on_calling_thread_and_is_reused(self) -> None:
        built_on: list[threading.Thread] = []

        def tracked_build(dataloader, len_fn):
            built_on.append(threading.current_thread())
            return build_source_loader(dataloader, len_fn)

        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings(), mock.patch(
            "lba.loader.build_source_loader",
            side_effect=tracked_build,
        ) as build:
            warnings.simplefilter("ignore")
            adapter = LBA(
                [[0], [1]],
                len_fn=len,
                batch_size=1,
                collate_fn=identity_collate,
                max_padded_length=1,
                prefetch_batches=1,
                log_dir=tmpdir,
            )

            first = list(adapter)
            source_loader = adapter._source_loader
            second = list(adapter)

        self.assertEqual(first, second)
        self.assertEqual(build.call_count, 1)
        self.assertEqual(built_on, [threading.current_thread()])
        self.assertIs(adapter._source_loader, source_loader)

    def test_prefetch_reuses_spawn_persistent_worker_across_iterations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter = LBA(
                PidDataset(),
                len_fn=one_length,
                batch_size=4,
                num_workers=1,
                persistent_workers=True,
                multiprocessing_context="spawn",
                collate_fn=identity_collate,
                max_padded_length=4,
                prefetch_batches=1,
                log_dir=tmpdir,
            )

            first_pids = {pid for batch in adapter for pid in batch}
            second_pids = {pid for batch in adapter for pid in batch}
            del adapter
            gc.collect()

        self.assertEqual(len(first_pids), 1)
        self.assertEqual(first_pids, second_pids)

    def test_pins_final_collated_batches(self) -> None:
        def mark_pinned(batch, *, enabled, device):
            self.assertTrue(enabled)
            self.assertIsNone(device)
            return ("pinned", batch)

        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            with mock.patch(
                "lba._iteration.pin_memory_enabled", return_value=True
            ), mock.patch(
                "lba._iteration.pin_batch",
                side_effect=mark_pinned,
            ) as pin:
                warnings.simplefilter("ignore")
                adapter = LBA(
                    [[0], [1]],
                    len_fn=len,
                    batch_size=2,
                    collate_fn=identity_collate,
                    pin_memory=True,
                    max_padded_length=2,
                    prefetch_batches=0,
                    log_dir=tmpdir,
                )

                batches = list(adapter)

        self.assertEqual(batches, [("pinned", [[0], [1]])])
        pin.assert_called_once()

    def test_adapter_releases_log_handler_when_collected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter = LBA(
                [[0]],
                len_fn=len,
                batch_size=1,
                max_padded_length=1,
                prefetch_batches=0,
                log_dir=tmpdir,
            )
            list(adapter)
            handler = adapter.logger.handlers[0]

            del adapter
            gc.collect()

            self.assertIsNone(handler.stream)

    def test_iterates_iterable_dataset(self) -> None:
        dataset = SequenceIterableDataset(
            [[0] * 5, [1] * 5, [2] * 4, [3] * 4],
        )

        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter = LBA(
                dataset,
                len_fn=len,
                batch_size=2,
                collate_fn=identity_collate,
                max_padded_length=10,
                max_padding_ratio=0.0,
                prefetch_batches=0,
                log_dir=tmpdir,
            )
            batches = list(adapter)

        self.assertEqual([len(batch) for batch in batches], [2, 2])
        self.assertEqual([len(sample) for sample in batches[0]], [5, 5])

    def test_rejects_legacy_dataloader_wrapper_api(self) -> None:
        dataloader = DataLoader([[0]], batch_size=1)

        with self.assertRaisesRegex(TypeError, "expects a dataset"):
            LBA(dataloader, len_fn=len)

    def test_max_batches_drops_remaining_cache(self) -> None:
        samples = [
            [0] * 5,
            [1] * 5,
            [2] * 4,
            [3] * 4,
            [4] * 4,
            [5] * 4,
        ]

        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter = LBA(
                samples,
                len_fn=len,
                batch_size=2,
                collate_fn=identity_collate,
                max_batches=1,
                max_padded_length=10,
                max_padding_ratio=0.0,
                prefetch_batches=0,
                log_dir=tmpdir,
            )
            batches = list(adapter)

        self.assertEqual(len(batches), 1)
        self.assertEqual([len(sample) for sample in batches[0]], [5, 5])

    def test_zero_max_batches_does_not_build_source_loader(self) -> None:
        dataset = SequenceIterableDataset([[0], [1]])

        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter = LBA(
                dataset,
                len_fn=len,
                batch_size=None,
                collate_fn=identity_collate,
                max_batches=0,
                max_padded_length=10,
                prefetch_batches=0,
                log_dir=tmpdir,
            )

            self.assertEqual(list(adapter), [])

    def test_rejects_unbatched_iterable_dataset(self) -> None:
        dataset = SequenceIterableDataset([[0], [1]])

        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter = LBA(
                dataset,
                len_fn=len,
                batch_size=None,
                collate_fn=identity_collate,
                max_padded_length=10,
                prefetch_batches=0,
                log_dir=tmpdir,
            )

            with self.assertRaisesRegex(ValueError, "batch_size"):
                list(adapter)

    def test_logs_human_summary_and_structured_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter = LBA(
                [[0] * 5, [1], [2] * 4, [3] * 4],
                len_fn=len,
                batch_size=2,
                collate_fn=identity_collate,
                max_padded_length=10,
                prefetch_batches=0,
                log_dir=tmpdir,
            )
            list(adapter)
            log_text = Path(adapter.log_path).read_text()
            event_text = Path(adapter.log_event_path).read_text()

        self.assertIn("lba summary: padding", log_text)
        self.assertIn("lba planner: total=", log_text)
        self.assertIn("lba health: oversized=", log_text)
        events = [json.loads(line) for line in event_text.splitlines()]
        summary = next(event for event in events if event["event"] == "summary")
        self.assertIn("padding_ratio_reduction", summary["padding"])
        self.assertIn("before", summary["padding"])
        self.assertIn("after", summary["padding"])
        self.assertIn("candidate_window_checks", summary["planner"])
        self.assertIn("paths", summary["planner"])
        run_start = next(event for event in events if event["event"] == "run_start")
        self.assertEqual(run_start["config"]["planner_mode"], "quality")
        self.assertIsNone(run_start["config"]["candidate_window_limit"])

    def test_oversized_log_omits_sample_repr(self) -> None:
        oversized_sample = [0] * 20
        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter = LBA(
                [oversized_sample],
                len_fn=len,
                batch_size=1,
                collate_fn=identity_collate,
                max_padded_length=10,
                prefetch_batches=0,
                log_dir=tmpdir,
            )
            list(adapter)
            log_text = Path(adapter.log_path).read_text()
            event_text = Path(adapter.log_event_path).read_text()

        self.assertIn("lba health: oversized sample", log_text)
        self.assertIn("sample_type=list", log_text)
        self.assertNotIn(repr(oversized_sample), log_text)
        events = [json.loads(line) for line in event_text.splitlines()]
        oversized_event = next(
            event for event in events if event["event"] == "oversized_sample"
        )
        self.assertEqual(oversized_event["length"], 20)

    def test_rejects_negative_prefetch_batches(self) -> None:
        with self.assertRaises(ValueError), tempfile.TemporaryDirectory() as tmpdir:
            LBA(
                [[0]],
                len_fn=len,
                batch_size=1,
                collate_fn=identity_collate,
                max_padded_length=10,
                prefetch_batches=-1,
                log_dir=tmpdir,
            )

    def test_rejects_negative_max_batches(self) -> None:
        with self.assertRaises(ValueError), tempfile.TemporaryDirectory() as tmpdir:
            LBA(
                [[0]],
                len_fn=len,
                batch_size=1,
                collate_fn=identity_collate,
                max_batches=-1,
                max_padded_length=10,
                log_dir=tmpdir,
            )

    def test_required_plan_rejects_empty_distributed_source_batch(self) -> None:
        planner = BatchPlanner(max_padded_length=10)

        with self.assertRaisesRegex(RuntimeError, "non-empty source batches"):
            Iteration.plans_after_add(planner, [], require_plan=True)

    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(),
        "torch.distributed gloo is unavailable",
    )
    def test_iterates_when_process_group_is_initialized(self) -> None:
        if dist.is_initialized():
            self.skipTest("process group is already initialized")

        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            init_path = Path(tmpdir) / "dist-init"
            dist.init_process_group(
                "gloo",
                init_method=f"file://{init_path}",
                rank=0,
                world_size=1,
            )
            try:
                adapter = LBA(
                    [[0] * 5, [1] * 5, [2] * 4, [3] * 4],
                    len_fn=len,
                    batch_size=2,
                    collate_fn=identity_collate,
                    max_padded_length=10,
                    max_padding_ratio=0.0,
                    log_dir=tmpdir,
                )
                batches = list(adapter)
            finally:
                dist.destroy_process_group()

        self.assertEqual([len(batch) for batch in batches], [2, 2])


if __name__ == "__main__":
    unittest.main()
