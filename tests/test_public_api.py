import unittest

from lba import (
    IterableLBA,
    IterableLengthBatchingAdapter,
    LBA,
    LengthBatchingAdapter,
)


class PublicApiTest(unittest.TestCase):
    def test_short_alias_points_to_main_adapter(self) -> None:
        self.assertIs(LBA, LengthBatchingAdapter)

    def test_iterable_alias_points_to_iterable_adapter(self) -> None:
        self.assertIs(IterableLBA, IterableLengthBatchingAdapter)


if __name__ == "__main__":
    unittest.main()
