from __future__ import annotations

from collections import Counter
import unittest

from lba._distributed_cost import (
    PlanMetadata,
    PlanRef,
    RecordMetadata,
    match_cost_block,
    plan_metadata,
)
from lba._records import BatchPlan, PlanReason, SampleRecord


def make_metadata(
    cost: int,
    index: int | None,
    *,
    position: int = 0,
    length: int = 1,
) -> PlanMetadata:
    return PlanMetadata(
        records=(
            RecordMetadata(
                index=index,
                length=length,
                arrival_id=position,
            ),
        ),
        reason=PlanReason.PLANNED,
        estimated_cost=cost,
    )


def ref_keys(plans: tuple[tuple[PlanRef, ...], ...]) -> list[tuple[int, int]]:
    return [
        (ref.source_rank, ref.source_position)
        for rank_plans in plans
        for ref in rank_plans
    ]


class DistributedCostMatcherTest(unittest.TestCase):
    def test_groups_adjacent_cost_quantiles_and_rotates_targets(self) -> None:
        gathered = (
            tuple(make_metadata(cost, index=cost) for cost in (100, 80, 60)),
            tuple(make_metadata(cost, index=cost) for cost in (90, 70, 50)),
        )

        assigned = match_cost_block(gathered)

        self.assertEqual(
            [
                [ref.metadata.estimated_cost for ref in rank_plans]
                for rank_plans in assigned
            ],
            [[100, 70, 60], [90, 80, 50]],
        )
        self.assertEqual(
            [
                [(ref.source_rank, ref.source_position) for ref in rank_plans]
                for rank_plans in assigned
            ],
            [[(0, 0), (1, 1), (0, 2)], [(1, 0), (0, 1), (1, 2)]],
        )

    def test_step_offset_rotates_the_next_block_instead_of_resetting(self) -> None:
        gathered = (
            tuple(make_metadata(cost, index=cost) for cost in (100, 80, 60)),
            tuple(make_metadata(cost, index=cost) for cost in (90, 70, 50)),
        )

        first = match_cost_block(gathered, step_offset=0)
        second = match_cost_block(gathered, step_offset=3)

        self.assertEqual(
            [
                [ref.metadata.estimated_cost for ref in rank_plans]
                for rank_plans in first
            ],
            [[100, 70, 60], [90, 80, 50]],
        )
        self.assertEqual(
            [
                [ref.metadata.estimated_cost for ref in rank_plans]
                for rank_plans in second
            ],
            [[90, 80, 50], [100, 70, 60]],
        )

    def test_equal_cost_ties_are_deterministic_by_source_and_position(self) -> None:
        gathered = (
            (make_metadata(10, 0, position=0), make_metadata(10, 1, position=1)),
            (make_metadata(10, 2, position=0), make_metadata(10, 3, position=1)),
        )

        first = match_cost_block(gathered)
        second = match_cost_block(gathered)

        self.assertEqual(first, second)
        self.assertEqual(
            [
                [(ref.source_rank, ref.source_position) for ref in rank_plans]
                for rank_plans in first
            ],
            [[(0, 0), (1, 1)], [(0, 1), (1, 0)]],
        )

    def test_each_input_ref_is_assigned_to_exactly_one_target(self) -> None:
        gathered = tuple(
            tuple(
                make_metadata(
                    cost=100 - (rank * 10 + position),
                    index=rank * 10 + position,
                    position=position,
                )
                for position in range(2)
            )
            for rank in range(3)
        )

        assigned = match_cost_block(gathered)
        assigned_keys = ref_keys(assigned)
        expected_keys = [(rank, position) for rank in range(3) for position in range(2)]

        self.assertEqual([len(rank_plans) for rank_plans in assigned], [2, 2, 2])
        self.assertEqual(Counter(assigned_keys), Counter(expected_keys))
        self.assertTrue(all(count == 1 for count in Counter(assigned_keys).values()))

    def test_world_three_rotation_assigns_each_cost_position_once(self) -> None:
        gathered = (
            tuple(make_metadata(cost, index=cost) for cost in (100, 90, 80)),
            tuple(make_metadata(cost, index=cost) for cost in (70, 60, 50)),
            tuple(make_metadata(cost, index=cost) for cost in (40, 30, 20)),
        )

        assigned = match_cost_block(gathered)

        self.assertEqual(
            [
                [ref.metadata.estimated_cost for ref in rank_plans]
                for rank_plans in assigned
            ],
            [[100, 50, 30], [90, 70, 20], [80, 60, 40]],
        )

    def test_plan_metadata_excludes_samples_and_keeps_plan_fields(self) -> None:
        plan = BatchPlan(
            records=(SampleRecord("payload", 7, 4, index=12),),
            raw_length_sum=7,
            padded_length=7,
            padding_length=0,
            padding_ratio=0.0,
            reason=PlanReason.OVERSIZED,
            estimated_cost=None,
        )

        metadata = plan_metadata(plan)

        self.assertEqual(metadata.reason, PlanReason.OVERSIZED)
        self.assertEqual(metadata.estimated_cost, 7)
        self.assertEqual(
            metadata.records,
            (RecordMetadata(index=12, length=7, arrival_id=4),),
        )
        self.assertFalse(hasattr(metadata.records[0], "sample"))

    def test_rejects_mismatched_block_sizes(self) -> None:
        gathered = (
            (make_metadata(10, 0), make_metadata(9, 1)),
            (make_metadata(8, 2),),
        )

        with self.assertRaisesRegex(RuntimeError, "same plan block size"):
            match_cost_block(gathered)

    def test_rejects_empty_gathered_ranks_and_empty_blocks(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "at least one rank"):
            match_cost_block(())

        with self.assertRaisesRegex(RuntimeError, "empty block"):
            match_cost_block(((), ()))

        empty_plan = PlanMetadata(
            records=(),
            reason=PlanReason.PLANNED,
            estimated_cost=1,
        )
        with self.assertRaisesRegex(RuntimeError, "empty plan"):
            match_cost_block(((empty_plan,),))

    def test_rejects_missing_index(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "map-style sample indices"):
            match_cost_block(((make_metadata(10, None),),))

    def test_rejects_non_positive_estimated_cost(self) -> None:
        for cost in (0, -1):
            with self.subTest(cost=cost):
                with self.assertRaisesRegex(RuntimeError, "positive estimated costs"):
                    match_cost_block(((make_metadata(cost, 0),),))

    def test_rejects_non_positive_record_length(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "positive sample lengths"):
            match_cost_block(((make_metadata(10, 0, length=0),),))

    def test_rejects_negative_step_offset(self) -> None:
        with self.assertRaisesRegex(ValueError, "step_offset"):
            match_cost_block(((make_metadata(10, 0),),), step_offset=-1)


if __name__ == "__main__":
    unittest.main()
