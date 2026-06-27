import unittest

from torch.utils.data import DataLoader

from lba.budget import BudgetResolver
from lba.config import (
    DEFAULT_THROUGHPUT_FALLBACK_AFTER,
    DEFAULT_THROUGHPUT_FALLBACK_POOL_SIZE,
    DEFAULT_THROUGHPUT_MAX_CANDIDATE_WINDOWS,
    LBAConfig,
)
from lba.estimator import LengthBudgetResolver
from lba.types import LengthRecord


class BudgetResolverTest(unittest.TestCase):
    def test_config_defaults_to_conservative_padding_ratio(self) -> None:
        self.assertEqual(LBAConfig().max_padding_ratio, 0.05)
        self.assertEqual(LBAConfig().planner_mode, "quality")
        self.assertIsNone(LBAConfig().candidate_window_limit)
        self.assertTrue(LBAConfig().drop_last_flush)

    def test_throughput_mode_defaults_to_limited_candidate_windows(self) -> None:
        config = LBAConfig(planner_mode="throughput")

        self.assertEqual(
            config.candidate_window_limit,
            DEFAULT_THROUGHPUT_MAX_CANDIDATE_WINDOWS,
        )
        self.assertEqual(
            config.limited_search_fallback_after_limit,
            DEFAULT_THROUGHPUT_FALLBACK_AFTER,
        )
        self.assertEqual(
            config.limited_search_fallback_pool_limit,
            DEFAULT_THROUGHPUT_FALLBACK_POOL_SIZE,
        )

    def test_explicit_candidate_window_limit_overrides_mode_default(self) -> None:
        config = LBAConfig(planner_mode="throughput", max_candidate_windows=128)

        self.assertEqual(config.candidate_window_limit, 128)

    def test_rejects_invalid_planner_options(self) -> None:
        with self.assertRaises(ValueError):
            LBAConfig(planner_mode="fast")
        with self.assertRaises(ValueError):
            LBAConfig(max_candidate_windows=0)
        with self.assertRaises(ValueError):
            LBAConfig(limited_search_fallback_after=0)
        with self.assertRaises(ValueError):
            LBAConfig(limited_search_fallback_pool_size=0)

    def test_resolver_keeps_config(self) -> None:
        config = LBAConfig(max_padded_length=256)
        loader = DataLoader([1, 2, 3, 4], batch_size=4)
        resolver = BudgetResolver(config, loader)

        self.assertIs(resolver.config, config)

    def test_resolver_infers_from_warmup_records(self) -> None:
        config = LBAConfig()
        loader = DataLoader([1, 2, 3, 4], batch_size=4)
        resolver = BudgetResolver(config, loader)

        budget = resolver.resolve(
            [
                LengthRecord("a", 2),
                LengthRecord("b", 4),
            ]
        )

        self.assertEqual(budget, 12)

    def test_old_resolver_name_is_compatible(self) -> None:
        self.assertIs(LengthBudgetResolver, BudgetResolver)


if __name__ == "__main__":
    unittest.main()
