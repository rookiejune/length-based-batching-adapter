import tempfile
import unittest
from pathlib import Path

from lba.logging_utils import (
    JsonlEventWriter,
    RunReporter,
    create_run_logger,
    default_log_dir,
    event_log_path_for,
    padding_event_fields,
    planner_event_fields,
)
from lba.metrics import PaddingStats, PlannerStats
from lba.types import SampleRecord


class LoggingUtilsTest(unittest.TestCase):
    def test_default_log_dir(self) -> None:
        self.assertEqual(default_log_dir(Path("/tmp/project")), Path("/tmp/project/.lba/logs"))

    def test_create_run_logger(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger, path = create_run_logger(tmpdir)
            logger.info("hello")

        self.assertTrue(path.name.startswith("lba-"))
        self.assertEqual(event_log_path_for(path), path.with_suffix(".jsonl"))

    def test_create_run_logger_uses_unique_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _first_logger, first_path = create_run_logger(tmpdir)
            _second_logger, second_path = create_run_logger(tmpdir)

        self.assertNotEqual(first_path, second_path)

    def test_jsonl_event_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.jsonl"
            writer = JsonlEventWriter(path)
            writer.write("summary", {"padding": {"after": 0.1}})
            text = path.read_text()

        self.assertIn('"event": "summary"', text)
        self.assertIn('"padding": {"after": 0.1}', text)

    def test_metric_event_fields(self) -> None:
        padding_stats = PaddingStats()
        padding_stats.add_lengths([5, 1])
        planner_stats = PlannerStats()
        planner_stats.record_pop_ready(
            elapsed_seconds=0.001,
            candidate_window_checks=3,
            source="fast_path",
        )

        padding_fields = padding_event_fields(padding_stats)
        planner_fields = planner_event_fields(planner_stats)

        self.assertEqual(padding_fields["batch_count"], 1)
        self.assertEqual(padding_fields["sample_count"], 2)
        self.assertEqual(planner_fields["candidate_window_checks"], 3)
        self.assertEqual(planner_fields["paths"]["fast_path"]["batches"], 1)

    def test_run_reporter_writes_oversized_event_without_sample_repr(self) -> None:
        sample = [0] * 20
        with tempfile.TemporaryDirectory() as tmpdir:
            logger, log_path = create_run_logger(tmpdir)
            event_path = event_log_path_for(log_path)
            reporter = RunReporter(logger, JsonlEventWriter(event_path), event_path)

            with self.assertWarnsRegex(UserWarning, "oversized sample length=20"):
                reporter.warn_oversized_sample(
                    SampleRecord(sample=sample, length=20, arrival_id=0, index=7),
                    max_padded_length=10,
                )
            log_text = log_path.read_text()
            event_text = event_path.read_text()

        self.assertIn("sample_type=list", log_text)
        self.assertNotIn(repr(sample), log_text)
        self.assertIn('"event": "oversized_sample"', event_text)


if __name__ == "__main__":
    unittest.main()
