import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lba import spill as spill_module
from lba.planner import BatchPlanner
from lba.types import PlanReason, SampleRecord


def quadratic_cost(max_length: int, batch_size: int) -> int:
    return max_length * max_length * batch_size


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

    def test_custom_cost_limits_batch_size_and_records_cost(self) -> None:
        planner = BatchPlanner(
            cost_fn=quadratic_cost,
            max_batch_cost=32,
            max_padding_ratio=0.0,
        )
        planner.add_records(
            [
                SampleRecord("a", 4, 0),
                SampleRecord("b", 4, 1),
                SampleRecord("c", 4, 2),
            ]
        )

        plan = planner.pop_ready()

        self.assertIsNotNone(plan)
        self.assertEqual(plan.samples, ["a", "b"])
        self.assertEqual(plan.padded_length, 8)
        self.assertEqual(plan.estimated_cost, 32)

    def test_custom_cost_oversized_sample_is_singleton(self) -> None:
        planner = BatchPlanner(
            cost_fn=quadratic_cost,
            max_batch_cost=32,
        )
        planner.add_records([SampleRecord("long", 6, 0)])

        plan = planner.pop_ready()

        self.assertIsNotNone(plan)
        self.assertEqual(plan.reason, PlanReason.OVERSIZED)
        self.assertEqual(plan.estimated_cost, 36)

    def test_custom_cost_return_must_be_positive_integer(self) -> None:
        planner = BatchPlanner(
            cost_fn=lambda _max_length, _batch_size: 0,
            max_batch_cost=32,
        )
        planner.add_records([SampleRecord("a", 4, 0)])

        with self.assertRaisesRegex(ValueError, "positive integer"):
            planner.pop_ready()

    def test_oversized_search_keeps_shortest_then_arrival_order(self) -> None:
        planner = BatchPlanner(max_padded_length=10)
        planner.add_records(
            [
                SampleRecord("long-late", 12, 3),
                SampleRecord("long-early", 12, 1),
                SampleRecord("longest", 20, 0),
                SampleRecord("ready", 5, 2),
            ]
        )

        plan = planner.pop_ready()

        self.assertEqual(plan.samples, ["long-early"])

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

    def test_limited_search_can_fallback_immediately_for_latency_mode(self) -> None:
        planner = BatchPlanner(
            max_padded_length=15,
            max_padding_ratio=0.0,
            max_candidate_windows=1,
            limited_search_fallback_after=8,
            defer_limited_search_miss=False,
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
        self.assertEqual(planner.stats.no_ready_call_count, 0)
        self.assertEqual(planner.stats.fallback_search_batch_count, 1)
        self.assertEqual([record.sample for record in plan.records], ["b", "c"])

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

    def test_required_search_does_not_defer_limited_miss(self) -> None:
        planner = BatchPlanner(
            max_padded_length=15,
            max_padding_ratio=0.0,
            max_candidate_windows=1,
            limited_search_fallback_after=8,
            limited_search_fallback_pool_size=1024,
        )
        planner.add_records(
            [
                SampleRecord("old", 4, 0),
                SampleRecord("first-new", 4, 1),
            ]
        )
        planner.pop_required()
        planner.add_records([SampleRecord("second-new", 5, 2)])

        plan = planner.pop_required()

        self.assertEqual(plan.samples, ["first-new", "second-new"])
        self.assertEqual(planner.stats.no_ready_call_count, 0)
        self.assertEqual(planner.stats.fallback_search_batch_count, 1)

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

    def test_small_spills_share_a_shard_and_flush_in_full_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = BatchPlanner(
                max_padded_length=10,
                max_cache_samples=2,
                max_padding_ratio=0.0,
                spill_dir=tmpdir,
            )
            for arrival_id in range(6):
                planner.add_records([SampleRecord(str(arrival_id), 5, arrival_id)])

            max_cache_size_before_flush = planner.stats.max_cache_size_seen
            shard_count_before_flush = planner.spill_store.shard_count
            plans = list(planner.flush())
            spill_paths = list(Path(tmpdir).glob("*.pkl"))

        self.assertEqual(shard_count_before_flush, 1)
        self.assertEqual([len(plan.records) for plan in plans], [2, 2, 2])
        self.assertTrue(all(plan.padding_length == 0 for plan in plans))
        self.assertCountEqual(
            [sample for plan in plans for sample in plan.samples],
            [str(index) for index in range(6)],
        )
        self.assertEqual(
            planner.stats.max_cache_size_seen,
            max_cache_size_before_flush,
        )
        self.assertEqual(spill_paths, [])

    def test_spill_record_drain_loads_one_record_at_a_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = BatchPlanner(
                max_padded_length=10,
                max_cache_samples=2,
                spill_dir=tmpdir,
            )
            for arrival_id in range(6):
                planner.add_records([SampleRecord(str(arrival_id), 5, arrival_id)])

            original_load = spill_module.pickle.load
            with mock.patch.object(
                spill_module.pickle,
                "load",
                wraps=original_load,
            ) as load:
                records = planner.spill_store.drain_records()
                first_record = next(records)
                self.assertEqual(load.call_count, 1)
                records.close()

            planner.close()

        self.assertEqual(first_record.arrival_id, 0)

    def test_flush_combines_records_from_multiple_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = BatchPlanner(
                max_padded_length=20,
                max_cache_samples=4,
                max_padding_ratio=0.0,
                spill_dir=tmpdir,
            )
            planner.spill_store.shard_size = 1
            for arrival_id in range(8):
                planner.add_records([SampleRecord(str(arrival_id), 5, arrival_id)])

            max_cache_size_before_flush = planner.stats.max_cache_size_seen
            shard_count_before_flush = planner.spill_store.shard_count
            plans = list(planner.flush())

        self.assertEqual(shard_count_before_flush, 4)
        self.assertEqual([len(plan.records) for plan in plans], [4, 4])
        self.assertCountEqual(
            [sample for plan in plans for sample in plan.samples],
            [str(index) for index in range(8)],
        )
        self.assertEqual(
            planner.stats.max_cache_size_seen,
            max_cache_size_before_flush,
        )

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
            for arrival_id, sample in enumerate("abcdef"):
                planner.add_records([SampleRecord(sample, 5, arrival_id)])

            shard_count_before_drain = planner.spill_store.shard_count
            drained_samples = [record.sample for record in planner.drain_records()]
            flushed_samples = [
                sample for plan in planner.flush() for sample in plan.samples
            ]
            spill_paths = list(Path(tmpdir).glob("*.pkl"))

        self.assertEqual(shard_count_before_drain, 1)
        self.assertEqual(drained_samples, list("abcdef"))
        self.assertEqual(flushed_samples, [])
        self.assertEqual(spill_paths, [])

    def test_close_cleans_current_shards_from_explicit_directory(self) -> None:
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
            self.assertTrue(list(Path(tmpdir).glob("*.pkl")))

            planner.close()

            self.assertTrue(Path(tmpdir).is_dir())
            self.assertEqual(list(Path(tmpdir).glob("*.pkl")), [])

    def test_seeded_random_plans_preserve_samples_and_budget(self) -> None:
        rng = random.Random(20260719)
        for case_index in range(100):
            record_count = rng.randint(1, 80)
            max_padded_length = rng.randint(1, 200)
            max_cache_samples = rng.choice((1, 4, 16, 64))
            max_candidate_windows = rng.choice((None, 1, 5, 32))
            planner = BatchPlanner(
                max_padded_length=max_padded_length,
                max_cache_samples=max_cache_samples,
                max_padding_ratio=rng.choice((0.0, 0.05, 0.2)),
                max_candidate_windows=max_candidate_windows,
                limited_search_fallback_after=(
                    3 if max_candidate_windows is not None else None
                ),
                limited_search_fallback_pool_size=(
                    max_cache_samples if max_candidate_windows is not None else None
                ),
            )
            records = [
                SampleRecord(index, rng.randint(1, 250), index)
                for index in range(record_count)
            ]
            plans = []
            next_index = 0
            try:
                while next_index < record_count:
                    chunk_size = rng.randint(1, 8)
                    planner.add_records(records[next_index : next_index + chunk_size])
                    next_index += chunk_size
                    plan = planner.pop_ready()
                    if plan is not None:
                        plans.append(plan)
                plans.extend(planner.flush())
            finally:
                planner.close()

            with self.subTest(case_index=case_index):
                self.assertEqual(
                    sorted(sample for plan in plans for sample in plan.samples),
                    list(range(record_count)),
                )
                for plan in plans:
                    if plan.reason == PlanReason.OVERSIZED:
                        self.assertEqual(len(plan.records), 1)
                        self.assertGreater(
                            plan.records[0].length,
                            max_padded_length,
                        )
                    else:
                        self.assertLessEqual(
                            plan.padded_length,
                            max_padded_length,
                        )


if __name__ == "__main__":
    unittest.main()
