import unittest
import tempfile
import warnings
from pathlib import Path

import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset

from lba import LBA
from lba.config import DEFAULT_PREFETCH_BATCHES


def identity_collate(samples):
    return samples


class SequenceIterableDataset(IterableDataset):
    def __init__(self, samples):
        self.samples = samples

    def __iter__(self):
        yield from self.samples


class WrapperSkeletonTest(unittest.TestCase):
    def test_constructor_keeps_inputs(self) -> None:
        dataloader = DataLoader(
            [[0] * 5, [1] * 5],
            batch_size=2,
            collate_fn=identity_collate,
        )

        def len_fn(sample):
            return len(sample)

        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter = LBA(dataloader, len_fn=len_fn, max_padded_length=128, log_dir=tmpdir)

        self.assertIs(adapter.dataloader, dataloader)
        self.assertIs(adapter.len_fn, len_fn)
        self.assertEqual(adapter.max_padded_length, 128)
        self.assertEqual(adapter.config.prefetch_batches, DEFAULT_PREFETCH_BATCHES)

    def test_iterates_dynamic_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter = LBA(
                DataLoader(
                    [[0] * 5, [1] * 5, [2] * 4, [3] * 4],
                    batch_size=2,
                    collate_fn=identity_collate,
                ),
                len_fn=len,
                max_padded_length=10,
                max_padding_ratio=0.0,
                log_dir=tmpdir,
            )
            batches = list(adapter)

        self.assertEqual([len(batch) for batch in batches], [2, 2])
        self.assertEqual([len(sample) for sample in batches[0]], [5, 5])

    def test_prefetch_iterates_dynamic_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter = LBA(
                DataLoader(
                    [[0] * 5, [1] * 5, [2] * 4, [3] * 4],
                    batch_size=2,
                    collate_fn=identity_collate,
                ),
                len_fn=len,
                max_padded_length=10,
                max_padding_ratio=0.0,
                prefetch_batches=2,
                log_dir=tmpdir,
            )
            batches = list(adapter)

        self.assertEqual([len(batch) for batch in batches], [2, 2])
        self.assertEqual([len(sample) for sample in batches[0]], [5, 5])

    def test_iterates_iterable_dataset(self) -> None:
        dataset = SequenceIterableDataset(
            [[0] * 5, [1] * 5, [2] * 4, [3] * 4],
        )

        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter = LBA(
                DataLoader(
                    dataset,
                    batch_size=2,
                    collate_fn=identity_collate,
                ),
                len_fn=len,
                max_padded_length=10,
                max_padding_ratio=0.0,
                prefetch_batches=0,
                log_dir=tmpdir,
            )
            batches = list(adapter)

        self.assertEqual([len(batch) for batch in batches], [2, 2])
        self.assertEqual([len(sample) for sample in batches[0]], [5, 5])

    def test_rejects_unbatched_iterable_dataset(self) -> None:
        dataset = SequenceIterableDataset([[0], [1]])

        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter = LBA(
                DataLoader(
                    dataset,
                    batch_size=None,
                    collate_fn=identity_collate,
                ),
                len_fn=len,
                max_padded_length=10,
                prefetch_batches=0,
                log_dir=tmpdir,
            )

            with self.assertRaisesRegex(ValueError, "batched DataLoader"):
                list(adapter)

    def test_logs_padding_and_planner_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter = LBA(
                DataLoader(
                    [[0] * 5, [1], [2] * 4, [3] * 4],
                    batch_size=2,
                    collate_fn=identity_collate,
                ),
                len_fn=len,
                max_padded_length=10,
                prefetch_batches=0,
                log_dir=tmpdir,
            )
            list(adapter)
            log_text = Path(adapter.log_path).read_text()

        self.assertIn("LBA summary padding", log_text)
        self.assertIn("before_padding_ratio=", log_text)
        self.assertIn("after_padding_ratio=", log_text)
        self.assertIn("padding_ratio_reduction=", log_text)
        self.assertIn("LBA summary planner", log_text)
        self.assertIn("sort_time_seconds=", log_text)
        self.assertIn("pop_ready_time_seconds=", log_text)
        self.assertIn("candidate_window_checks=", log_text)
        self.assertIn("fast_path_batches=", log_text)

    def test_rejects_negative_prefetch_batches(self) -> None:
        with self.assertRaises(ValueError), tempfile.TemporaryDirectory() as tmpdir:
            LBA(
                DataLoader([[0]], batch_size=1, collate_fn=identity_collate),
                len_fn=len,
                max_padded_length=10,
                prefetch_batches=-1,
                log_dir=tmpdir,
            )

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
                    DataLoader(
                        [[0] * 5, [1] * 5, [2] * 4, [3] * 4],
                        batch_size=2,
                        collate_fn=identity_collate,
                    ),
                    len_fn=len,
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
