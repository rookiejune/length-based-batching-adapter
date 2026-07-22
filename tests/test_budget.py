import unittest

from torch.utils.data import DataLoader

from lba.budget import BudgetResolver
from lba.config import (
    DEFAULT_THROUGHPUT_FALLBACK_AFTER,
    DEFAULT_THROUGHPUT_FALLBACK_POOL_SIZE,
    DEFAULT_THROUGHPUT_MAX_CANDIDATE_WINDOWS,
    LBAConfig,
)
from lba.adaptive import AdaptiveConfig
from lba.estimator import LengthBudgetResolver
from lba.types import LengthRecord


def quadratic_cost(max_length: int, batch_size: int) -> int:
    return max_length * max_length * batch_size


class BudgetResolverTest(unittest.TestCase):
    def test_config_defaults_to_conservative_padding_ratio(self) -> None:
        self.assertEqual(LBAConfig().max_padding_ratio, 0.05)
        self.assertEqual(LBAConfig().planner_mode, "quality")
        self.assertIsNone(LBAConfig().candidate_window_limit)
        self.assertIsNone(LBAConfig().distributed_cost_window_batches)
        self.assertIsNone(LBAConfig().adaptive)
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

    def test_custom_cost_config_requires_one_explicit_budget(self) -> None:
        config = LBAConfig(
            cost_fn=quadratic_cost,
            max_batch_cost=1024,
            cost_window_batches=4,
        )

        self.assertTrue(config.uses_custom_cost)
        self.assertEqual(config.max_batch_cost, 1024)
        self.assertEqual(config.cost_window_batches, 4)

        with self.assertRaisesRegex(ValueError, "requires max_batch_cost"):
            LBAConfig(cost_fn=quadratic_cost)
        with self.assertRaisesRegex(ValueError, "requires cost_fn"):
            LBAConfig(max_batch_cost=1024)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            LBAConfig(
                max_padded_length=128,
                cost_fn=quadratic_cost,
                max_batch_cost=1024,
            )
        with self.assertRaisesRegex(ValueError, "warmup_batches"):
            LBAConfig(
                warmup_batches=2,
                cost_fn=quadratic_cost,
                max_batch_cost=1024,
            )
        with self.assertRaisesRegex(ValueError, "cost_window_batches"):
            LBAConfig(cost_window_batches=0)
        for invalid_value in (True, 1.5):
            with self.subTest(value=invalid_value), self.assertRaisesRegex(
                TypeError,
                "cost_window_batches",
            ):
                LBAConfig(cost_window_batches=invalid_value)

    def test_distributed_cost_window_validation(self) -> None:
        config = LBAConfig(distributed_cost_window_batches=4)

        self.assertEqual(config.distributed_cost_window_batches, 4)

        for invalid_value in (0, 1, -1):
            with self.subTest(value=invalid_value), self.assertRaisesRegex(
                ValueError,
                "distributed_cost_window_batches",
            ):
                LBAConfig(distributed_cost_window_batches=invalid_value)
        for invalid_value in (True, 2.5):
            with self.subTest(value=invalid_value), self.assertRaisesRegex(
                TypeError,
                "distributed_cost_window_batches",
            ):
                LBAConfig(distributed_cost_window_batches=invalid_value)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            LBAConfig(
                cost_window_batches=2,
                distributed_cost_window_batches=2,
            )

    def test_adaptive_validation(self) -> None:
        adaptive = AdaptiveConfig(distributed_cost_window_batches=None)
        config = LBAConfig(adaptive=adaptive)

        self.assertIs(config.adaptive, adaptive)

        with self.assertRaisesRegex(TypeError, "AdaptiveConfig"):
            LBAConfig(adaptive=object())
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            LBAConfig(
                distributed_cost_window_batches=2,
                adaptive=adaptive,
            )
        with self.assertRaisesRegex(ValueError, "cost_window_batches"):
            LBAConfig(
                cost_window_batches=2,
                adaptive=adaptive,
            )

    def test_adaptive_padding_only_allows_local_cost_window(self) -> None:
        config = LBAConfig(
            adaptive=AdaptiveConfig(),
            cost_window_batches=2,
        )

        self.assertEqual(config.cost_window_batches, 2)

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
