import random
import unittest
from bisect import bisect_left
from math import ceil
from unittest.mock import patch

from lba._candidate_index import ArrivalIdRangeMin, CandidateIndex
from lba._candidate_search import (
    best_candidate_key,
    find_best_candidate,
    find_threshold_candidate,
    iter_batch_candidates,
    iter_recent_batch_candidates,
    threshold_candidate_key,
)
from lba._cost import BatchCost
from lba._records import SampleRecord


def quadratic_cost(max_length: int, batch_size: int) -> int:
    return max_length * max_length * batch_size


def _reference_candidate_windows(
    records: list[SampleRecord],
    *,
    max_padded_length: int,
    max_padding_ratio: float,
) -> list[tuple[int, int]]:
    sorted_lengths = [record.length for record in records]
    windows: list[tuple[int, int]] = []
    for end_index, longest_record in enumerate(records):
        if longest_record.length <= 0:
            continue

        max_record_count = max_padded_length // longest_record.length
        if max_record_count <= 0:
            continue

        widest_start_index = max(0, end_index - max_record_count + 1)
        min_length_for_ratio = ceil(
            longest_record.length * (1 - max_padding_ratio)
        )
        tight_start_index = bisect_left(
            sorted_lengths,
            min_length_for_ratio,
            widest_start_index,
            end_index + 1,
        )
        seen: set[int] = set()
        for start_index in (
            widest_start_index,
            tight_start_index,
            tight_start_index - 1,
            end_index - 1,
            end_index,
        ):
            if (
                widest_start_index <= start_index <= end_index
                and start_index not in seen
            ):
                seen.add(start_index)
                windows.append((start_index, end_index))
    return windows


class CandidateSearchTest(unittest.TestCase):
    def test_arrival_id_range_min_queries_windows(self) -> None:
        records = [
            SampleRecord("a", 2, 20),
            SampleRecord("b", 3, 10),
            SampleRecord("c", 5, 30),
            SampleRecord("d", 8, 5),
        ]
        range_min = ArrivalIdRangeMin.from_records(records)

        self.assertEqual(range_min.range_min(0, 0), 20)
        self.assertEqual(range_min.range_min(0, 2), 10)
        self.assertEqual(range_min.range_min(2, 3), 5)
        with self.assertRaises(ValueError):
            range_min.range_min(2, 4)

    def test_recent_threshold_search_matches_full_recent_filter(self) -> None:
        records = [
            SampleRecord("a", 1, 0),
            SampleRecord("b", 2, 1),
            SampleRecord("c", 2, 2),
            SampleRecord("d", 3, 3),
            SampleRecord("e", 5, 4),
            SampleRecord("f", 5, 5),
            SampleRecord("g", 6, 6),
            SampleRecord("h", 9, 7),
        ]
        index = CandidateIndex.from_records(records)
        recent_arrival_ids = {2, 5}

        result = find_threshold_candidate(
            index,
            max_padded_length=18,
            max_padding_ratio=0.2,
            recent_arrival_ids=recent_arrival_ids,
        )
        full_recent_candidates = [
            candidate
            for candidate in iter_batch_candidates(
                index,
                max_padded_length=18,
                max_padding_ratio=0.2,
            )
            if candidate.padding_ratio <= 0.2
            and any(
                record.arrival_id in recent_arrival_ids
                for record in records[
                    candidate.start_index : candidate.end_index + 1
                ]
            )
        ]
        expected = min(full_recent_candidates, key=threshold_candidate_key)
        full_candidate_count = len(
            list(
                iter_batch_candidates(
                    index,
                    max_padded_length=18,
                    max_padding_ratio=0.2,
                )
            )
        )

        self.assertIsNotNone(result.candidate)
        self.assertEqual(result.candidate.start_index, expected.start_index)
        self.assertEqual(result.candidate.end_index, expected.end_index)
        self.assertLess(result.inspected_count, full_candidate_count)

    def test_recent_indices_uses_length_sorted_order(self) -> None:
        records = [
            SampleRecord("a", 1, 10),
            SampleRecord("b", 2, 4),
            SampleRecord("c", 3, 8),
            SampleRecord("d", 5, 6),
        ]
        index = CandidateIndex.from_records(records)

        self.assertEqual(index.recent_indices({6, 10, 999}), [0, 3])

    def test_custom_cost_controls_candidate_size_and_estimate(self) -> None:
        records = [
            SampleRecord("a", 4, 0),
            SampleRecord("b", 4, 1),
            SampleRecord("c", 4, 2),
        ]
        index = CandidateIndex.from_records(records)
        batch_cost = BatchCost(32, quadratic_cost)

        candidates = list(
            iter_batch_candidates(
                index,
                batch_cost=batch_cost,
                max_padding_ratio=0.0,
            )
        )

        self.assertTrue(all(candidate.record_count <= 2 for candidate in candidates))
        self.assertTrue(all(candidate.estimated_cost <= 32 for candidate in candidates))
        self.assertIn(32, [candidate.estimated_cost for candidate in candidates])

    def test_recent_threshold_search_can_limit_candidate_windows(self) -> None:
        records = [
            SampleRecord("a", 10, 0),
            SampleRecord("b", 10, 1),
            SampleRecord("c", 10, 2),
        ]
        index = CandidateIndex.from_records(records)

        result = find_threshold_candidate(
            index,
            max_padded_length=30,
            max_padding_ratio=0.0,
            recent_arrival_ids={0},
            max_candidate_windows=1,
        )

        self.assertIsNotNone(result.candidate)
        self.assertEqual(result.candidate.start_index, 0)
        self.assertEqual(result.candidate.end_index, 0)
        self.assertEqual(result.inspected_count, 1)

    def test_equal_length_threshold_ties_use_range_min(self) -> None:
        arrival_ids = [10, 11, 12, 13, 14, 15, 16, 0, 1, 2]
        records = [
            SampleRecord(str(arrival_id), 1, arrival_id)
            for arrival_id in arrival_ids
        ]
        index = CandidateIndex.from_records(records)

        with patch.object(
            CandidateIndex,
            "make_candidate_with_scanned_arrivals",
            side_effect=AssertionError("tie search must use range-min"),
        ):
            result = find_threshold_candidate(
                index,
                max_padded_length=3,
                max_padding_ratio=0.0,
                recent_arrival_ids=frozenset(),
            )

        self.assertIsNotNone(result.candidate)
        self.assertEqual(result.candidate.start_index, 5)
        self.assertEqual(result.candidate.end_index, 7)
        self.assertEqual(result.candidate.earliest_arrival_id, 0)
        self.assertIsNotNone(index._arrival_id_range_min)

    def test_unique_threshold_winner_keeps_scanned_arrival_path(self) -> None:
        index = CandidateIndex.from_records([SampleRecord("a", 3, 7)])

        with patch.object(
            CandidateIndex,
            "make_candidate",
            side_effect=AssertionError("unique winner must not build range-min"),
        ):
            result = find_threshold_candidate(
                index,
                max_padded_length=3,
                max_padding_ratio=0.0,
                recent_arrival_ids=frozenset(),
            )

        self.assertIsNotNone(result.candidate)
        self.assertEqual(result.candidate.earliest_arrival_id, 7)
        self.assertIsNone(index._arrival_id_range_min)

    def test_unlimited_recent_candidates_match_full_recent_filter(self) -> None:
        records = [
            SampleRecord("a", 1, 0),
            SampleRecord("b", 2, 1),
            SampleRecord("c", 3, 2),
            SampleRecord("d", 5, 3),
            SampleRecord("e", 8, 4),
            SampleRecord("f", 13, 5),
        ]
        index = CandidateIndex.from_records(records)
        recent_indices = [1, 4]

        recent_candidates = {
            (candidate.start_index, candidate.end_index)
            for candidate in iter_recent_batch_candidates(
                index,
                max_padded_length=24,
                max_padding_ratio=0.25,
                recent_indices=recent_indices,
            )
        }
        expected_candidates = {
            (candidate.start_index, candidate.end_index)
            for candidate in iter_batch_candidates(
                index,
                max_padded_length=24,
                max_padding_ratio=0.25,
            )
            if any(
                candidate.start_index <= recent_index <= candidate.end_index
                for recent_index in recent_indices
            )
        }

        self.assertEqual(recent_candidates, expected_candidates)

    def test_random_candidate_windows_match_reference_and_recent_filter(self) -> None:
        rng = random.Random(12345)
        padding_ratios = [0.0, 0.05, 0.2, 0.5, 1.0]

        for case_index in range(200):
            record_count = rng.randint(1, 64)
            lengths = sorted(
                rng.randint(1, 64) for _ in range(record_count)
            )
            arrival_ids = list(range(record_count))
            rng.shuffle(arrival_ids)
            records = [
                SampleRecord(str(arrival_id), length, arrival_id)
                for length, arrival_id in zip(lengths, arrival_ids)
            ]
            index = CandidateIndex.from_records(records)
            max_padded_length = rng.randint(1, 512)
            max_padding_ratio = rng.choice(padding_ratios)
            expected_windows = _reference_candidate_windows(
                records,
                max_padded_length=max_padded_length,
                max_padding_ratio=max_padding_ratio,
            )
            actual_windows = [
                (candidate.start_index, candidate.end_index)
                for candidate in iter_batch_candidates(
                    index,
                    max_padded_length=max_padded_length,
                    max_padding_ratio=max_padding_ratio,
                )
            ]

            recent_indices = rng.sample(
                range(record_count),
                rng.randint(0, record_count),
            )
            rng.shuffle(recent_indices)
            expected_recent_windows = [
                (start_index, end_index)
                for start_index, end_index in expected_windows
                if any(
                    start_index <= recent_index <= end_index
                    for recent_index in recent_indices
                )
            ]
            actual_recent_windows = [
                (candidate.start_index, candidate.end_index)
                for candidate in iter_recent_batch_candidates(
                    index,
                    max_padded_length=max_padded_length,
                    max_padding_ratio=max_padding_ratio,
                    recent_indices=recent_indices,
                )
            ]

            with self.subTest(case_index=case_index):
                self.assertEqual(actual_windows, expected_windows)
                self.assertEqual(actual_recent_windows, expected_recent_windows)
                self.assertEqual(
                    len(actual_recent_windows),
                    len(set(actual_recent_windows)),
                )

    def test_recent_candidate_limit_must_be_positive(self) -> None:
        records = [SampleRecord("a", 1, 0)]

        with self.assertRaises(ValueError):
            list(
                iter_recent_batch_candidates(
                    CandidateIndex.from_records(records),
                    max_padded_length=1,
                    max_padding_ratio=0.0,
                    recent_indices=[0],
                    max_candidate_windows=0,
                )
            )

    def test_best_search_matches_full_scan(self) -> None:
        records = [
            SampleRecord("a", 1, 0),
            SampleRecord("b", 4, 1),
            SampleRecord("c", 5, 2),
            SampleRecord("d", 8, 3),
        ]
        index = CandidateIndex.from_records(records)

        result = find_best_candidate(
            index,
            max_padded_length=12,
            max_padding_ratio=0.1,
        )
        candidates = list(
            iter_batch_candidates(
                index,
                max_padded_length=12,
                max_padding_ratio=0.1,
            )
        )
        multi_record_candidates = [
            candidate for candidate in candidates if candidate.record_count > 1
        ]
        expected = min(multi_record_candidates or candidates, key=best_candidate_key)

        self.assertIsNotNone(result.candidate)
        self.assertEqual(result.candidate.start_index, expected.start_index)
        self.assertEqual(result.candidate.end_index, expected.end_index)
        self.assertEqual(result.inspected_count, len(candidates))

    def test_threshold_search_covers_middle_windows(self) -> None:
        records = [
            SampleRecord("a", 1, 0),
            SampleRecord("b", 1, 1),
            SampleRecord("c", 2, 2),
        ]
        index = CandidateIndex.from_records(records)

        result = find_threshold_candidate(
            index,
            max_padded_length=6,
            max_padding_ratio=0.3,
            recent_arrival_ids=frozenset(),
        )

        self.assertIsNotNone(result.candidate)
        self.assertEqual(result.candidate.start_index, 1)
        self.assertEqual(result.candidate.end_index, 2)

    def test_best_search_covers_middle_windows(self) -> None:
        records = [
            SampleRecord("a", 1, 0),
            SampleRecord("b", 2, 1),
            SampleRecord("c", 3, 2),
        ]
        index = CandidateIndex.from_records(records)

        result = find_best_candidate(
            index,
            max_padded_length=9,
            max_padding_ratio=0.05,
        )

        self.assertIsNotNone(result.candidate)
        self.assertEqual(result.candidate.start_index, 1)
        self.assertEqual(result.candidate.end_index, 2)


if __name__ == "__main__":
    unittest.main()
