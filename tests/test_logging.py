from __future__ import annotations

import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

from twen.training.logging import (
    JsonlEventLogger,
    JsonlMetricLogger,
    RankZeroSessionFile,
    ThroughputTracker,
    TrainingProgress,
    TrainingTelemetryTracker,
    exception_fields,
)


class JsonlMetricLoggerTest(unittest.TestCase):
    def test_strict_json_rejects_reserved_and_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            logger = JsonlMetricLogger(Path(directory) / "metrics.jsonl")
            with self.assertRaisesRegex(ValueError, "reserved"):
                logger.log(1, {"step": 99})
            with self.assertRaises(ValueError):
                logger.log(1, {"loss": math.nan})
            with self.assertRaises(TypeError):
                logger.log(1, {"opaque": object()})
            self.assertEqual(logger.last_step, -1)

    def test_reconcile_discards_steps_newer_than_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            logger = JsonlMetricLogger(path)
            for step in (1, 2, 3):
                self.assertTrue(logger.log(step, {"loss": float(step)}))
            logger.reconcile(1)
            self.assertEqual(logger.last_step, 1)
            self.assertTrue(logger.log(2, {"loss": 20.0}))
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([item["step"] for item in records], [1, 2])
            self.assertEqual(records[-1]["loss"], 20.0)

    def test_reconcile_removes_a_torn_tail_before_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text('{"step": 1, "loss": 1.0}\n{"step": 2')
            logger = JsonlMetricLogger(path)
            self.assertEqual(logger.last_step, 1)
            logger.reconcile(1)
            self.assertTrue(logger.log(2, {"loss": 2.0}))
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([item["step"] for item in records], [1, 2])

    def test_zero_step_checkpoint_authenticates_an_empty_metrics_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            logger = JsonlMetricLogger(Path(directory) / "metrics.jsonl")
            binding = logger.snapshot_prefix(through_step=0, committed_tokens=0)
            self.assertEqual(binding["record_count"], 0)
            self.assertEqual(binding["prefix_size_bytes"], 0)
            self.assertEqual(
                binding["prefix_sha256"],
                "e3b0c44298fc1c149afbf4c8996fb924"
                "27ae41e4649b934ca495991b7852b855",
            )


class StructuredTrainingLogTest(unittest.TestCase):
    def test_event_log_has_session_and_utc_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            logger = JsonlEventLogger(path, session_id="session-test")
            logger.log("train_start", {"world_size": 1})
            record = json.loads(path.read_text())
            self.assertEqual(record["event"], "train_start")
            self.assertEqual(record["session_id"], "session-test")
            self.assertTrue(record["timestamp_utc"].endswith("+00:00"))

    def test_exception_fields_include_full_traceback(self) -> None:
        try:
            raise RuntimeError("failure details")
        except RuntimeError as error:
            fields = exception_fields(error)
        self.assertEqual(fields["error_type"], "RuntimeError")
        self.assertEqual(fields["error"], "failure details")
        self.assertIn("Traceback (most recent call last)", fields["traceback"])
        self.assertIn('raise RuntimeError("failure details")', fields["traceback"])
        self.assertIn("RuntimeError: failure details", fields["traceback"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            logger = JsonlEventLogger(path, session_id="failure-session")
            logger.log("train_failed", fields)
            record = json.loads(path.read_text())
        self.assertEqual(record["traceback"], fields["traceback"])

    def test_rank_zero_session_file_preserves_pid_and_marks_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank0-session.json"
            marker = RankZeroSessionFile(
                path,
                session_id="session-test",
                fields={"run_id": "run-test", "stage": "dense-oracle"},
            )
            running = json.loads(path.read_text())
            self.assertEqual(running["session_id"], "session-test")
            self.assertEqual(running["pid"], os.getpid())
            self.assertEqual(running["status"], "running")
            self.assertIsNone(running["ended_at_utc"])

            marker.finish("completed", {"step": 7, "tokens": 4096})
            completed = json.loads(path.read_text())
            self.assertEqual(completed["session_id"], "session-test")
            self.assertEqual(completed["pid"], os.getpid())
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["step"], 7)
            self.assertEqual(completed["tokens"], 4096)
            self.assertIsNotNone(completed["ended_at_utc"])

    def test_throughput_tracker_handles_resume_and_ema(self) -> None:
        tracker = ThroughputTracker(total_tokens=1_000, initial_tokens=200, ema_alpha=0.5)
        first = tracker.observe(300, 2.0)
        second = tracker.observe(500, 2.0)
        self.assertEqual(first["tokens_this_step"], 100)
        self.assertEqual(first["tokens_per_second"], 50.0)
        self.assertEqual(second["tokens_per_second_ema"], 75.0)
        self.assertAlmostEqual(float(second["eta_seconds"]), 500 / 75)
        self.assertEqual(second["progress_percent"], 50.0)

    def test_training_telemetry_separates_compute_and_wall_clock_rates(self) -> None:
        ticks = iter((10.0, 14.0, 20.0))
        tracker = TrainingTelemetryTracker(
            total_tokens=1_000,
            initial_tokens=200,
            ema_alpha=0.5,
            clock=lambda: next(ticks),
        )
        first = tracker.observe(300, 2.0)
        second = tracker.observe(500, 2.0)

        self.assertEqual(first["tokens_this_step"], 100)
        self.assertEqual(first["compute_step_seconds"], 2.0)
        self.assertEqual(first["compute_tokens_per_second"], 50.0)
        self.assertEqual(first["wall_clock_step_seconds"], 4.0)
        self.assertEqual(first["wall_clock_tokens_per_second"], 25.0)
        self.assertEqual(first["step_seconds"], first["compute_step_seconds"])
        self.assertEqual(first["tokens_per_second"], first["compute_tokens_per_second"])
        self.assertEqual(second["compute_tokens_per_second_ema"], 75.0)
        self.assertAlmostEqual(
            float(second["wall_clock_tokens_per_second_ema"]),
            (25.0 + 200 / 6.0) / 2.0,
        )
        self.assertGreater(
            float(second["wall_clock_eta_seconds"]),
            float(second["compute_eta_seconds"]),
        )

    def test_progress_auto_is_silent_for_non_tty_and_always_renders(self) -> None:
        auto_stream = io.StringIO()
        auto = TrainingProgress(total_tokens=100, mode="auto", stream=auto_stream)
        self.assertFalse(auto.enabled)
        auto.update(10, {"loss": 1.0})
        auto.close()
        self.assertEqual(auto_stream.getvalue(), "")

        forced_stream = io.StringIO()
        forced = TrainingProgress(total_tokens=100, mode="always", stream=forced_stream)
        self.assertTrue(forced.enabled)
        forced.update(
            10,
            {
                "loss": 1.25,
                "ntp_loss": 1.0,
                "mtp_loss": 0.95,
                "teacher_kd_loss": 0.9,
                "anchor_kl_loss": 0.8,
                "hidden_alignment_loss": 0.7,
                "dense_oracle_loss": 0.6,
                "router_supervision_loss": 0.5,
                "load_balance_loss": 0.4,
                "router_z_loss": 0.3,
                "grad_norm": 0.2,
                "lr": 2.0e-4,
                "compute_tokens_per_second_ema": 50.0,
                "wall_clock_tokens_per_second_ema": 42.0,
                "wall_clock_eta_seconds": 61.0,
                "gpu_peak_allocated_gib": 3.0,
                "top_k": 2,
            },
        )
        forced.close()
        rendered = forced_stream.getvalue()
        for label in (
            "loss",
            "ntp",
            "mtp",
            "kd",
            "anchor",
            "hidden",
            "oracle",
            "r-sup",
            "balance",
            "r-z",
            "grad",
            "lr",
            "compute tok/s",
            "wall tok/s",
            "ETA",
            "peak GiB",
            "top-k",
        ):
            self.assertIn(label, rendered)


if __name__ == "__main__":
    unittest.main()
