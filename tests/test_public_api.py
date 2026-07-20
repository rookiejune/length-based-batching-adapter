import unittest

from torch.utils.data import DataLoader

import lba
from lba import LBA


class PublicApiTest(unittest.TestCase):
    def test_lba_is_the_only_public_loader(self) -> None:
        self.assertEqual(lba.__all__, ["LBA"])
        self.assertFalse(hasattr(lba, "LengthBatchingAdapter"))
        self.assertFalse(hasattr(lba, "IterableLBA"))

    def test_lba_is_a_dataloader(self) -> None:
        self.assertTrue(issubclass(LBA, DataLoader))

    def test_major_version_matches_breaking_loader_api(self) -> None:
        self.assertEqual(lba.__version__, "2.0.0")


if __name__ == "__main__":
    unittest.main()
