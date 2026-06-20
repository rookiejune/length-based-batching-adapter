import unittest

from lba.candidates import (
    best_candidate_key,
    find_best_candidate,
    find_threshold_candidate,
    iter_batch_candidates,
    threshold_candidate_key,
)
from lba.types import SampleRecord


def prefix_lengths(records: list[SampleRecord]) -> list[int]:
    lengths = [0]
    for record in records:
        lengths.append(lengths[-1] + record.length)
    return lengths


class CandidateSearchTest(unittest.TestCase):
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
        prefixes = prefix_lengths(records)
        recent_arrival_ids = {2, 5}

        result = find_threshold_candidate(
            records,
            prefixes,
            max_padded_length=18,
            max_padding_ratio=0.2,
            recent_arrival_ids=recent_arrival_ids,
        )
        full_recent_candidates = [
            candidate
            for candidate in iter_batch_candidates(
                records,
                prefixes,
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
                    records,
                    prefixes,
                    max_padded_length=18,
                    max_padding_ratio=0.2,
                )
            )
        )

        self.assertIsNotNone(result.candidate)
        self.assertEqual(result.candidate.start_index, expected.start_index)
        self.assertEqual(result.candidate.end_index, expected.end_index)
        self.assertLess(result.inspected_count, full_candidate_count)

    def test_best_search_matches_full_scan(self) -> None:
        records = [
            SampleRecord("a", 1, 0),
            SampleRecord("b", 4, 1),
            SampleRecord("c", 5, 2),
            SampleRecord("d", 8, 3),
        ]
        prefixes = prefix_lengths(records)

        result = find_best_candidate(
            records,
            prefixes,
            max_padded_length=12,
            max_padding_ratio=0.1,
        )
        candidates = list(
            iter_batch_candidates(
                records,
                prefixes,
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


if __name__ == "__main__":
    unittest.main()
