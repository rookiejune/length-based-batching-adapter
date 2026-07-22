import pickle
import unittest

import warnings

from torch.utils.data import DataLoader, Dataset

from lba.source import IndexedSampleDataset, build_source_loader


class BatchedDataset(Dataset):
    def __init__(self) -> None:
        self.item_calls = 0
        self.items_calls = 0
        self.offset = 0

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> list[int]:
        self.item_calls += 1
        return [index + self.offset]

    def __getitems__(self, indices: list[int]) -> list[list[int]]:
        self.items_calls += 1
        return [[index + self.offset] for index in indices]


class SourceLoaderTest(unittest.TestCase):
    def test_map_source_records_keep_dataset_indices(self) -> None:
        loader = DataLoader(
            [[0], [1, 1], [2, 2, 2]],
            batch_size=2,
        )

        records = next(iter(build_source_loader(loader, len)))

        self.assertEqual([record.sample for record in records], [0, 1])
        self.assertEqual([record.length for record in records], [1, 2])
        self.assertEqual([record.index for record in records], [0, 1])

    def test_map_source_preserves_batched_dataset_fetch(self) -> None:
        dataset = BatchedDataset()
        loader = DataLoader(dataset, batch_size=4)

        records = next(iter(build_source_loader(loader, len)))

        self.assertEqual(len(records), 4)
        self.assertEqual(dataset.items_calls, 1)
        self.assertEqual(dataset.item_calls, 0)

    def test_indexed_dataset_forwards_worker_state_to_wrapped_dataset(self) -> None:
        dataset = BatchedDataset()
        wrapped = IndexedSampleDataset(dataset)

        wrapped.offset = 7

        self.assertEqual(dataset.offset, 7)
        self.assertEqual(wrapped.offset, 7)
        self.assertEqual(wrapped[0].sample, [7])

    def test_indexed_dataset_supports_spawn_serialization(self) -> None:
        wrapped = IndexedSampleDataset(BatchedDataset())

        restored = pickle.loads(pickle.dumps(wrapped))

        self.assertEqual(restored[0].sample, [0])

    def test_source_loader_defers_pinning_until_final_collate(self) -> None:
        loader = DataLoader([[0]], batch_size=1, pin_memory=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            source = build_source_loader(loader, len)

        self.assertFalse(source.pin_memory)

    def test_source_loader_preserves_in_order_when_supported(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loader = DataLoader([[0]], batch_size=1, in_order=False)
            source = build_source_loader(loader, len)

        self.assertFalse(source.in_order)


if __name__ == "__main__":
    unittest.main()
