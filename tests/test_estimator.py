import unittest

from torch.utils.data import DataLoader

from lba.config import LBAConfig
from lba.estimator import LengthBudgetResolver
from lba.types import LengthRecord


class EstimatorSkeletonTest(unittest.TestCase):
    def test_config_defaults_to_conservative_padding_ratio(self) -> None:
        self.assertEqual(LBAConfig().max_padding_ratio, 0.05)
        self.assertTrue(LBAConfig().drop_last_flush)

    def test_resolver_keeps_config(self) -> None:
        config = LBAConfig(max_padded_length=256)
        loader = DataLoader([1, 2, 3, 4], batch_size=4)
        resolver = LengthBudgetResolver(config, loader)

        self.assertIs(resolver.config, config)

    def test_resolver_infers_from_warmup_records(self) -> None:
        config = LBAConfig()
        loader = DataLoader([1, 2, 3, 4], batch_size=4)
        resolver = LengthBudgetResolver(config, loader)

        budget = resolver.resolve(
            [
                LengthRecord("a", 2),
                LengthRecord("b", 4),
            ]
        )

        self.assertEqual(budget, 12)


if __name__ == "__main__":
    unittest.main()
