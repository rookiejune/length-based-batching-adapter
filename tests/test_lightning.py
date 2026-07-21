import tempfile
import unittest
import warnings
from pathlib import Path

import torch
from torch.utils.data import DistributedSampler

from lba import LBA

try:
    from lightning import pytorch as pl
    from lightning.fabric.utilities.data import _set_sampler_epoch
    from lightning.pytorch.utilities.data import _update_dataloader
except ImportError:
    pl = None
    _set_sampler_epoch = None
    _update_dataloader = None


def identity_collate(samples):
    return samples


def unit_length(sample) -> int:
    return 1


def quadratic_cost(max_length: int, batch_size: int) -> int:
    return max_length * max_length * batch_size


if pl is not None:

    class ScalarModel(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.Linear(1, 1)

        def training_step(self, batch, batch_idx):
            del batch_idx
            return self.layer(batch.float()).sum()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)


class RecordingDistributedSampler(DistributedSampler):
    def __init__(self, dataset, **kwargs) -> None:
        super().__init__(dataset, **kwargs)
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)
        super().set_epoch(epoch)


@unittest.skipUnless(_update_dataloader is not None, "Lightning is unavailable")
class LightningDataLoaderTest(unittest.TestCase):
    def test_trainer_runs_with_explicit_step_budget(self) -> None:
        dataset = [torch.tensor([index]) for index in range(8)]
        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loader = LBA(
                dataset,
                len_fn=unit_length,
                batch_size=2,
                max_padded_length=2,
                prefetch_batches=0,
                log_dir=tmpdir,
            )
            trainer = pl.Trainer(
                accelerator="cpu",
                devices=1,
                max_steps=2,
                max_epochs=-1,
                logger=False,
                enable_checkpointing=False,
                enable_model_summary=False,
                enable_progress_bar=False,
                num_sanity_val_steps=0,
            )

            trainer.fit(ScalarModel(), train_dataloaders=loader)

        self.assertEqual(trainer.global_step, 2)

    def test_injected_sampler_reaches_source_loader(self) -> None:
        dataset = [[index] for index in range(8)]
        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loader = LBA(
                dataset,
                len_fn=len,
                batch_size=2,
                collate_fn=identity_collate,
                max_padded_length=2,
                max_padding_ratio=0.0,
                prefetch_batches=0,
                log_dir=tmpdir,
            )
            sampler = DistributedSampler(
                dataset,
                num_replicas=2,
                rank=0,
                shuffle=False,
            )

            updated = _update_dataloader(loader, sampler)

            self.assertIsInstance(updated, LBA)
            self.assertIs(updated.dataset, dataset)
            self.assertIs(updated.sampler, sampler)
            self.assertEqual(updated.config.max_padding_ratio, 0.0)
            self.assertIsNone(updated._run_logger)
            self.assertEqual(list(Path(tmpdir).iterdir()), [])

            list(updated)

        self.assertIsNotNone(updated._source_loader)
        self.assertIs(updated._source_loader.batch_sampler.sampler, sampler)

    def test_reconstruction_preserves_custom_cost_options(self) -> None:
        dataset = [[index] for index in range(8)]
        loader = LBA(
            dataset,
            len_fn=len,
            cost_fn=quadratic_cost,
            max_batch_cost=8,
            cost_window_batches=4,
            batch_size=2,
            prefetch_batches=0,
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=2,
            rank=0,
            shuffle=False,
        )

        updated = _update_dataloader(loader, sampler)

        self.assertIs(updated.cost_fn, quadratic_cost)
        self.assertEqual(updated.max_batch_cost, 8)
        self.assertEqual(updated.cost_window_batches, 4)
        self.assertIs(updated.config.cost_fn, quadratic_cost)

    def test_lightning_epoch_hook_advances_injected_sampler(self) -> None:
        dataset = [[index] for index in range(8)]
        loader = LBA(
            dataset,
            len_fn=len,
            batch_size=2,
            max_padded_length=2,
        )
        sampler = RecordingDistributedSampler(
            dataset,
            num_replicas=2,
            rank=0,
            shuffle=True,
        )
        updated = _update_dataloader(loader, sampler)

        _set_sampler_epoch(updated, 7)

        self.assertEqual(sampler.epochs, [7])

    def test_distributed_sampler_padding_can_duplicate_indices(self) -> None:
        dataset = list(range(5))
        rank_indices = [
            set(
                DistributedSampler(
                    dataset,
                    num_replicas=2,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )
            )
            for rank in range(2)
        ]

        self.assertEqual(rank_indices[0] | rank_indices[1], set(range(5)))
        self.assertTrue(rank_indices[0] & rank_indices[1])


if __name__ == "__main__":
    unittest.main()
