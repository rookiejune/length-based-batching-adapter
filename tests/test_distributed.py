import unittest

from lba.distributed import DistributedBatchCoordinator
from lba.types import BatchPlan, SampleRecord


class DistributedCoordinatorTest(unittest.TestCase):
    def test_splits_local_plans_to_match_distributed_step_count(self) -> None:
        records = (
            SampleRecord("a", 1, 0),
            SampleRecord("b", 1, 1),
            SampleRecord("c", 1, 2),
            SampleRecord("d", 1, 3),
        )
        plan = BatchPlan(
            records=records,
            raw_length_sum=4,
            padded_length=4,
            padding_length=0,
            padding_ratio=0.0,
            reason="planned",
        )

        split_plans = DistributedBatchCoordinator.split_plans_to_count([plan], 4)

        self.assertEqual(
            [len(split_plan.records) for split_plan in split_plans],
            [1, 1, 1, 1],
        )
        self.assertEqual(
            [
                record.sample
                for split_plan in split_plans
                for record in split_plan.records
            ],
            ["a", "b", "c", "d"],
        )


if __name__ == "__main__":
    unittest.main()
