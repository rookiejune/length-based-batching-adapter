import unittest
import tempfile

from lba.planner import BatchPlanner
from lba.types import SampleRecord


class PlannerSkeletonTest(unittest.TestCase):
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
        self.assertEqual(plan.reason, "oversized")
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
