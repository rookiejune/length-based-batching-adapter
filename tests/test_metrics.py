import unittest

from lba.metrics import PaddingStats, PlannerStats, padding_ratio_reduction
from lba.types import BatchPlan, PlanReason, SampleRecord


class MetricsTest(unittest.TestCase):
    def test_padding_stats_aggregate_global_and_batch_ratios(self) -> None:
        stats = PaddingStats()

        stats.add_lengths([5, 1])
        stats.add_lengths([4, 4])

        self.assertEqual(stats.batch_count, 2)
        self.assertEqual(stats.sample_count, 4)
        self.assertEqual(stats.raw_length_sum, 14)
        self.assertEqual(stats.padded_length_sum, 18)
        self.assertEqual(stats.padding_length_sum, 4)
        self.assertAlmostEqual(stats.global_padding_ratio, 4 / 18)
        self.assertAlmostEqual(stats.mean_batch_padding_ratio, 0.2)

    def test_padding_stats_count_plan_reasons(self) -> None:
        stats = PaddingStats()

        stats.add_plan(
            BatchPlan(
                records=(SampleRecord("a", 12, 0),),
                raw_length_sum=12,
                padded_length=12,
                padding_length=0,
                padding_ratio=0.0,
                reason=PlanReason.OVERSIZED,
            )
        )

        self.assertEqual(stats.oversized_batch_count, 1)
        self.assertEqual(stats.planned_batch_count, 0)

    def test_padding_ratio_reduction(self) -> None:
        before = PaddingStats()
        after = PaddingStats()
        before.add_lengths([5, 1])
        after.add_lengths([5])
        after.add_lengths([1])

        self.assertAlmostEqual(padding_ratio_reduction(before, after), 1.0)

    def test_planner_stats_record_timing_and_spill(self) -> None:
        stats = PlannerStats()

        stats.record_sort(sorted_record_count=10, elapsed_seconds=0.002)
        stats.record_sort(sorted_record_count=4, elapsed_seconds=0.001)
        stats.record_pop_ready(
            elapsed_seconds=0.004,
            candidate_window_checks=7,
            source="fast_path",
        )
        stats.record_pop_ready(
            elapsed_seconds=0.002,
            candidate_window_checks=3,
            source="flush_search",
        )
        stats.record_spill(3)

        self.assertEqual(stats.sort_call_count, 2)
        self.assertEqual(stats.records_sorted_total, 14)
        self.assertAlmostEqual(stats.sort_time_seconds, 0.003)
        self.assertAlmostEqual(stats.average_sort_time_ms, 1.5)
        self.assertEqual(stats.pop_ready_call_count, 2)
        self.assertAlmostEqual(stats.pop_ready_time_seconds, 0.006)
        self.assertAlmostEqual(stats.average_pop_ready_time_ms, 3.0)
        self.assertEqual(stats.candidate_window_checks, 10)
        self.assertEqual(stats.max_candidate_window_checks, 7)
        self.assertAlmostEqual(stats.average_candidate_window_checks, 5.0)
        self.assertEqual(stats.fast_path_batch_count, 1)
        self.assertEqual(stats.flush_search_batch_count, 1)
        self.assertAlmostEqual(stats.fast_path_time_seconds, 0.004)
        self.assertAlmostEqual(stats.flush_search_time_seconds, 0.002)
        self.assertEqual(stats.fast_path_candidate_window_checks, 7)
        self.assertEqual(stats.flush_search_candidate_window_checks, 3)
        self.assertEqual(stats.max_cache_size_seen, 10)
        self.assertEqual(stats.spill_event_count, 1)
        self.assertEqual(stats.spilled_record_count, 3)


if __name__ == "__main__":
    unittest.main()
