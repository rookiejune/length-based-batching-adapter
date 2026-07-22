import unittest

from lba.adaptive import (
    DISABLED,
    AdaptiveConfig,
    AdaptiveState,
    adaptive_config_fields,
)
from lba._distributed_cost import CostWindowStats
from lba._records import BatchPlan, PlanReason, SampleRecord


def make_plan(padding_ratio: float) -> BatchPlan:
    padded_length = 100
    padding_length = int(padded_length * padding_ratio)
    return BatchPlan(
        records=(SampleRecord(sample="x", length=100 - padding_length, arrival_id=0),),
        raw_length_sum=padded_length - padding_length,
        padded_length=padded_length,
        padding_length=padding_length,
        padding_ratio=padding_ratio,
        reason=PlanReason.PLANNED,
        estimated_cost=padded_length,
    )


def cost_stats(
    *,
    spread_ratio: float,
    improvement_ratio: float,
) -> CostWindowStats:
    return CostWindowStats(
        block_size=2,
        mean_cost=100.0,
        source_mean_step_spread=spread_ratio * 100,
        matched_mean_step_spread=(1 - improvement_ratio) * spread_ratio * 100,
        source_spread_ratio=spread_ratio,
        improvement_ratio=improvement_ratio,
        remote_plan_count=0,
        remote_record_count=0,
    )


class AdaptiveConfigTest(unittest.TestCase):
    def test_default_auto_adjusts_only_padding_ratio(self) -> None:
        config = AdaptiveConfig()
        state = AdaptiveState(config)

        self.assertTrue(config.adjusts_max_padding_ratio)
        self.assertFalse(config.adjusts_distributed_cost_window)
        self.assertFalse(config.adjusts_max_candidate_windows)
        self.assertEqual(state.max_padding_ratio, 0.05)
        self.assertIsNone(state.distributed_cost_window_batches)
        self.assertIsNone(state.max_candidate_windows)

    def test_none_means_auto_for_each_enabled_knob(self) -> None:
        config = AdaptiveConfig(
            max_padding_ratio=None,
            distributed_cost_window_batches=None,
            max_candidate_windows=None,
        )
        state = AdaptiveState(config)

        self.assertEqual(state.max_padding_ratio, 0.05)
        self.assertEqual(state.distributed_cost_window_batches, 4)
        self.assertEqual(state.max_candidate_windows, 128)

    def test_concrete_values_are_fixed_starting_values(self) -> None:
        config = AdaptiveConfig(
            max_padding_ratio=0.075,
            distributed_cost_window_batches=8,
            max_candidate_windows=256,
        )
        state = AdaptiveState(config)

        self.assertEqual(state.max_padding_ratio, 0.075)
        self.assertEqual(state.distributed_cost_window_batches, 8)
        self.assertEqual(state.max_candidate_windows, 256)

    def test_disabled_fields_are_logged_distinctly_from_auto(self) -> None:
        fields = adaptive_config_fields(
            AdaptiveConfig(
                max_padding_ratio=DISABLED,
                distributed_cost_window_batches=None,
            )
        )

        self.assertEqual(fields["max_padding_ratio"], "disabled")
        self.assertEqual(fields["distributed_cost_window_batches"], "auto")

    def test_validates_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_padding_ratio_values"):
            AdaptiveConfig(max_padding_ratio_values=())
        with self.assertRaisesRegex(ValueError, "at least 2"):
            AdaptiveConfig(distributed_cost_window_values=(1,))
        with self.assertRaisesRegex(TypeError, "max_candidate_windows"):
            AdaptiveConfig(max_candidate_windows=True)
        with self.assertRaisesRegex(ValueError, "present"):
            AdaptiveConfig(max_padding_ratio=0.2)
        with self.assertRaisesRegex(ValueError, "low_padding_ratio"):
            AdaptiveConfig(low_padding_ratio=0.9, high_padding_ratio=0.5)


class AdaptiveStateTest(unittest.TestCase):
    def test_missing_plan_loosens_padding_ratio(self) -> None:
        state = AdaptiveState(AdaptiveConfig())

        updates = state.feedback_for_missing_plan()

        self.assertEqual(state.max_padding_ratio, 0.075)
        update = updates[0]
        self.assertEqual(update["knob"], "max_padding_ratio")
        self.assertEqual(update["reason"], "no_ready")

    def test_low_padding_streak_tightens_padding_ratio(self) -> None:
        state = AdaptiveState(AdaptiveConfig(padding_patience=2))

        self.assertEqual(state.feedback_for_plan(make_plan(0.01)), [])
        updates = state.feedback_for_plan(make_plan(0.01))

        self.assertEqual(state.max_padding_ratio, 0.025)
        update = updates[0]
        self.assertEqual(update["reason"], "low_padding_streak")

    def test_fallback_plan_over_threshold_loosens_padding_ratio(self) -> None:
        state = AdaptiveState(AdaptiveConfig())

        updates = state.feedback_for_plan(make_plan(0.08))

        self.assertEqual(state.max_padding_ratio, 0.075)
        update = updates[0]
        self.assertEqual(update["reason"], "fallback_exceeded_threshold")

    def test_concrete_values_are_not_adjusted(self) -> None:
        state = AdaptiveState(
            AdaptiveConfig(
                max_padding_ratio=0.075,
                distributed_cost_window_batches=4,
                max_candidate_windows=128,
            )
        )

        self.assertEqual(state.feedback_for_missing_plan(), [])
        self.assertEqual(state.feedback_for_plan(make_plan(0.12)), [])
        update = state.update_cost_window(
            cost_stats(spread_ratio=0.4, improvement_ratio=0.8)
        )

        self.assertIsNone(update)
        self.assertEqual(state.max_padding_ratio, 0.075)
        self.assertEqual(state.distributed_cost_window_batches, 4)
        self.assertEqual(state.max_candidate_windows, 128)

    def test_missing_plan_increases_candidate_window_when_enabled(self) -> None:
        state = AdaptiveState(AdaptiveConfig(max_candidate_windows=None))

        updates = state.feedback_for_missing_plan()

        self.assertEqual(state.max_candidate_windows, 256)
        self.assertTrue(
            any(update["knob"] == "max_candidate_windows" for update in updates)
        )

    def test_cost_window_increases_when_matching_helps_high_spread(self) -> None:
        state = AdaptiveState(
            AdaptiveConfig(distributed_cost_window_batches=None)
        )

        update = state.update_cost_window(
            cost_stats(spread_ratio=0.4, improvement_ratio=0.8)
        )

        self.assertEqual(state.distributed_cost_window_batches, 8)
        self.assertEqual(update["knob"], "distributed_cost_window_batches")

    def test_cost_window_decreases_when_matching_does_not_help(self) -> None:
        state = AdaptiveState(
            AdaptiveConfig(distributed_cost_window_batches=4)
        )

        update = state.update_cost_window(
            cost_stats(spread_ratio=0.4, improvement_ratio=0.1)
        )

        self.assertEqual(state.distributed_cost_window_batches, 4)
        self.assertIsNone(update)

    def test_auto_cost_window_decreases_when_matching_does_not_help(self) -> None:
        state = AdaptiveState(
            AdaptiveConfig(distributed_cost_window_batches=None)
        )

        update = state.update_cost_window(
            cost_stats(spread_ratio=0.4, improvement_ratio=0.1)
        )

        self.assertEqual(state.distributed_cost_window_batches, 2)
        self.assertEqual(update["reason"], "cost_spread")


if __name__ == "__main__":
    unittest.main()
