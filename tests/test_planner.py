import tempfile
import unittest
from pathlib import Path

from lba.planner import BatchPlanner
from lba.types import PlanReason, SampleRecord


class BatchPlannerTest(unittest.TestCase):
    def test_rejects_invalid_max_padded_length(self) -> None:
        with self.assertRaises(ValueError):
            BatchPlanner(0)

    def test_selects_zero_padding_window(self) -> None:
        planner = BatchPlanner(max_padded_length=15, max_padding_ratio=0.0)
        planner.add_records(
            [
                SampleRecord("a", 5, 0),
                SampleRecord("b", 5, 1),
                SampleRecord("c", 5, 2),
                SampleRecord("d", 9, 3),
            ]
        )

        plan = planner.pop_ready()

        self.assertIsNotNone(plan)
        self.assertEqual([record.sample for record in plan.records], ["a", "b", "c"])
        self.assertEqual(plan.padded_length, 15)
        self.assertEqual(plan.padding_length, 0)

    def test_oversized_sample_is_singleton(self) -> None:
        planner = BatchPlanner(max_padded_length=10)
        planner.add_records([SampleRecord("long", 12, 0)])

        plan = planner.pop_ready()

        self.assertIsNotNone(plan)
        self.assertEqual(plan.reason, PlanReason.OVERSIZED)
        self.assertEqual([record.sample for record in plan.records], ["long"])
        self.assertEqual(planner.stats.oversized_batch_count, 1)
        self.assertEqual(planner.stats.pop_ready_call_count, 1)

    def test_records_fast_path_search_stats(self) -> None:
        planner = BatchPlanner(max_padded_length=10, max_padding_ratio=0.0)
        planner.add_records(
            [
                SampleRecord("a", 5, 0),
                SampleRecord("b", 5, 1),
            ]
        )

        plan = planner.pop_ready()

        self.assertIsNotNone(plan)
        self.assertEqual([record.sample for record in plan.records], ["a", "b"])
        self.assertEqual(planner.stats.fast_path_batch_count, 1)
        self.assertEqual(planner.stats.pop_ready_call_count, 1)
        self.assertGreater(planner.stats.pop_ready_time_seconds, 0.0)
        self.assertGreater(planner.stats.candidate_window_checks, 0)

    def test_limited_search_defers_fallback_search(self) -> None:
        planner = BatchPlanner(
            max_padded_length=15,
            max_padding_ratio=0.0,
            max_candidate_windows=1,
            limited_search_fallback_after=2,
        )
        planner.add_records(
            [
                SampleRecord("a", 4, 0),
                SampleRecord("b", 5, 1),
            ]
        )
        planner.add_records([SampleRecord("c", 5, 2)])

        plan = planner.pop_ready()
        first_pop_max_checks = planner.stats.max_candidate_window_checks

        self.assertIsNone(plan)
        self.assertEqual(planner.stats.no_ready_call_count, 1)
        self.assertEqual(planner.stats.fallback_search_batch_count, 0)
        self.assertEqual(first_pop_max_checks, 1)

    def test_limited_search_falls_back_after_repeated_misses(self) -> None:
        planner = BatchPlanner(
            max_padded_length=15,
            max_padding_ratio=0.0,
            max_candidate_windows=1,
            limited_search_fallback_after=2,
        )
        planner.add_records(
            [
                SampleRecord("a", 4, 0),
                SampleRecord("b", 5, 1),
            ]
        )
        planner.add_records([SampleRecord("c", 5, 2)])

        first_plan = planner.pop_ready()
        second_plan = planner.pop_ready()

        self.assertIsNone(first_plan)
        self.assertIsNotNone(second_plan)
        self.assertEqual(planner.stats.no_ready_call_count, 1)
        self.assertEqual(planner.stats.fallback_search_batch_count, 1)
        self.assertEqual([record.sample for record in second_plan.records], ["b", "c"])

    def test_limited_search_uncaps_threshold_search_when_pool_is_too_large(self) -> None:
        planner = BatchPlanner(
            max_padded_length=15,
            max_padding_ratio=0.0,
            max_candidate_windows=1,
            limited_search_fallback_pool_size=3,
        )
        planner.add_records(
            [
                SampleRecord("a", 4, 0),
                SampleRecord("b", 5, 1),
            ]
        )
        planner.add_records([SampleRecord("c", 5, 2)])

        plan = planner.pop_ready()

        self.assertIsNotNone(plan)
        self.assertEqual(planner.stats.fast_path_batch_count, 1)
        self.assertEqual(planner.stats.fallback_search_batch_count, 0)
        self.assertCountEqual(
            [record.sample for record in plan.records],
            ["b", "c"],
        )
        self.assertGreater(planner.stats.max_candidate_window_checks, 1)

    def test_add_records_merges_new_records_into_sorted_pool(self) -> None:
        planner = BatchPlanner(max_padded_length=1, max_padding_ratio=0.0)
        planner.add_records(
            [
                SampleRecord("a", 8, 0),
                SampleRecord("b", 2, 1),
            ]
        )
        planner.add_records(
            [
                SampleRecord("c", 5, 2),
                SampleRecord("d", 2, 3),
            ]
        )

        plans = list(planner.flush())

        self.assertEqual(
            [sample for plan in plans for sample in plan.samples],
            ["b", "d", "c", "a"],
        )
        self.assertEqual(planner.stats.sort_call_count, 2)

    def test_records_flush_search_stats(self) -> None:
        planner = BatchPlanner(max_padded_length=10, max_padding_ratio=0.0)
        planner.add_records(
            [
                SampleRecord("a", 5, 0),
                SampleRecord("b", 5, 1),
                SampleRecord("c", 5, 2),
            ]
        )

        plans = list(planner.flush())

        self.assertEqual([sample for plan in plans for sample in plan.samples], ["a", "b", "c"])
        self.assertEqual(planner.stats.flush_search_batch_count, 2)
        self.assertEqual(planner.stats.pop_ready_call_count, 2)
        self.assertGreater(planner.stats.candidate_window_checks, 0)

    def test_spills_and_flushes_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = BatchPlanner(max_padded_length=10, max_cache_samples=2, spill_dir=tmpdir)
            planner.add_records(
                [
                    SampleRecord("a", 5, 0),
                    SampleRecord("b", 5, 1),
                    SampleRecord("c", 5, 2),
                ]
            )

            samples = [sample for plan in planner.flush() for sample in plan.samples]

        self.assertCountEqual(samples, ["a", "b", "c"])

    def test_flush_drains_spill_shards_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = BatchPlanner(
                max_padded_length=10,
                max_cache_samples=1,
                spill_dir=tmpdir,
            )
            planner.add_records(
                [
                    SampleRecord("a", 5, 0),
                    SampleRecord("b", 5, 1),
                ]
            )

            first_samples = [sample for plan in planner.flush() for sample in plan.samples]
            second_samples = [sample for plan in planner.flush() for sample in plan.samples]
            spill_paths = list(Path(tmpdir).glob("*.pkl"))

        self.assertCountEqual(first_samples, ["a", "b"])
        self.assertEqual(second_samples, [])
        self.assertEqual(spill_paths, [])

    def test_drains_records_from_memory_and_spill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = BatchPlanner(
                max_padded_length=10,
                max_cache_samples=2,
                spill_dir=tmpdir,
            )
            planner.add_records(
                [
                    SampleRecord("a", 5, 0),
                    SampleRecord("b", 5, 1),
                    SampleRecord("c", 5, 2),
                ]
            )

            drained_samples = [record.sample for record in planner.drain_records()]
            flushed_samples = [
                sample for plan in planner.flush() for sample in plan.samples
            ]

        self.assertEqual(drained_samples, ["a", "b", "c"])
        self.assertEqual(flushed_samples, [])


if __name__ == "__main__":
    unittest.main()
