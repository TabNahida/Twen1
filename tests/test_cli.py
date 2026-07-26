from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from twen.cli import _enable_expandable_segments_allocator, build_parser, main
from twen.kd import KDGenerationStopped, _raise_if_stopped, generate_teacher_kd
from twen.progress import TaskProgress


class CliContractTest(unittest.TestCase):
    def test_expandable_allocator_preserves_options_and_overrides_false(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PYTORCH_ALLOC_CONF": (
                    "max_split_size_mb:128,expandable_segments:False,"
                    "garbage_collection_threshold:0.8"
                )
            },
            clear=False,
        ):
            _enable_expandable_segments_allocator()
            self.assertEqual(
                os.environ["PYTORCH_ALLOC_CONF"],
                "max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True",
            )

    def test_expandable_allocator_does_not_duplicate_existing_true(self) -> None:
        with patch.dict(
            os.environ,
            {"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
            clear=False,
        ):
            _enable_expandable_segments_allocator()
            self.assertEqual(
                os.environ["PYTORCH_ALLOC_CONF"],
                "expandable_segments:True",
            )

    def test_task_progress_never_writes_machine_readable_stdout(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            contextlib.redirect_stdout(stdout),
            TaskProgress(
                total=2,
                description="fixture",
                unit="item",
                mode="always",
                stream=stderr,
            ) as progress,
        ):
            progress.update(2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("fixture", stderr.getvalue())

    def test_download_commands_default_to_huggingface_proxy_fallback(self) -> None:
        parser = build_parser()
        download_set = parser.parse_args(
            ["download", "set", "--spec", "lock.json", "--output", "models/local"]
        )
        lock_model = parser.parse_args(
            [
                "download",
                "lock-model",
                "--provider",
                "huggingface",
                "--model-id",
                "Qwen/example",
                "--output",
                "lock.json",
            ]
        )
        self.assertEqual(download_set.network_policy, "fallback")
        self.assertEqual(lock_model.network_policy, "fallback")

    def test_kd_default_uses_stable_rtx5090_logits_chunk(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "data",
                "generate-kd",
                "--prepared-manifest",
                "prepared.json",
                "--output",
                "kd",
                "--teacher",
                "teacher",
                "--teacher-model-id",
                "Qwen/example",
                "--teacher-revision",
                "a" * 40,
                "--teacher-manifest-sha256",
                "b" * 64,
                "--tokenizer-manifest-sha256",
                "c" * 64,
            ]
        )
        self.assertEqual(args.logits_chunk_tokens, 64)

    def test_materialize_cooldown_parser_defaults_to_real_materialization(self) -> None:
        args = build_parser().parse_args(
            [
                "data",
                "materialize-cooldown",
                "--prepared-manifest",
                "prepared.json",
                "--kd-manifest",
                "kd.json",
                "--selection-policy",
                "policy.json",
                "--output",
                "cooldown",
                "--required-cooldown-tokens",
                "50000000",
            ]
        )
        self.assertEqual(args.required_cooldown_tokens, 50_000_000)
        self.assertFalse(args.dry_run)

    def test_materialize_cooldown_rejects_non_positive_token_counts(self) -> None:
        required = [
            "data",
            "materialize-cooldown",
            "--prepared-manifest",
            "prepared.json",
            "--kd-manifest",
            "kd.json",
            "--selection-policy",
            "policy.json",
            "--output",
            "cooldown",
            "--required-cooldown-tokens",
        ]
        for value in ("0", "-1", "not-an-integer"):
            with (
                self.subTest(value=value),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                build_parser().parse_args([*required, value])
            self.assertEqual(raised.exception.code, 2)

    def test_materialize_cooldown_calls_materializer_and_prints_json(self) -> None:
        expected = {"ok": True, "dry_run": True, "selected_shards": 7}
        stdout = io.StringIO()
        with (
            patch(
                "twen.data.materialize_quality_cooldown_view",
                return_value=expected,
                create=True,
            ) as materialize,
            contextlib.redirect_stdout(stdout),
        ):
            code = main(
                [
                    "data",
                    "materialize-cooldown",
                    "--prepared-manifest",
                    "prepared.json",
                    "--kd-manifest",
                    "kd.json",
                    "--selection-policy",
                    "policy.json",
                    "--output",
                    "cooldown",
                    "--required-cooldown-tokens",
                    "50000000",
                    "--dry-run",
                ]
            )
        self.assertEqual(code, 0)
        materialize.assert_called_once_with(
            prepared_manifest_path="prepared.json",
            kd_manifest_path="kd.json",
            selection_policy_path="policy.json",
            output_root="cooldown",
            required_cooldown_tokens=50_000_000,
            dry_run=True,
        )
        self.assertEqual(json.loads(stdout.getvalue()), expected)

    def test_calibration_collect_input_modes_are_mutually_exclusive(self) -> None:
        parser = build_parser()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "calibrate",
                    "collect",
                    "--config",
                    "config.yaml",
                    "--input",
                    "tokens.safetensors",
                    "--prepared-manifest",
                    "manifest.json",
                    "--output",
                    "calibration",
                ]
            )

    def test_calibration_ridge_defaults_are_optimized_for_device(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "calibrate",
                "ridge",
                "--config",
                "config.yaml",
                "--student-activations",
                "student",
                "--donor-activations",
                "donor",
                "--output",
                "adapters.safetensors",
            ]
        )
        self.assertEqual(args.ridge_dtype, "auto")
        self.assertEqual(args.ridge_batch_samples, 1024)

    def test_calibration_success_writes_machine_readable_artifact_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "layer-map.json"
            output.write_text("{}\n", encoding="utf-8")
            stdout = io.StringIO()
            with (
                patch("twen.calibration.run_calibration_command", return_value=0),
                contextlib.redirect_stdout(stdout),
            ):
                code = main(
                    [
                        "calibrate",
                        "layer-map",
                        "--config",
                        "config.yaml",
                        "--student-activations",
                        "student",
                        "--donor-activations",
                        "donor",
                        "--output",
                        str(output),
                        "--progress",
                        "never",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["action"], "layer-map")
            self.assertEqual(payload["artifact"]["path"], str(output.resolve()))
            self.assertEqual(len(payload["artifact"]["sha256"]), 64)

    def test_checkpoint_request_signals_only_active_rank_zero_pid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "rank0-session.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "session_id": "session-fixture",
                        "pid": 43210,
                        "hostname": "host-fixture",
                        "status": "running",
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with (
                patch("platform.node", return_value="host-fixture"),
                patch("twen.cli.os.kill") as kill,
                contextlib.redirect_stdout(stdout),
            ):
                code = main(
                    [
                        "checkpoint",
                        "request",
                        "--run-dir",
                        str(run_dir),
                        "--action",
                        "save",
                    ]
                )
            self.assertEqual(code, 0)
            kill.assert_called_once_with(43210, signal.SIGUSR1)
            self.assertEqual(json.loads(stdout.getvalue())["session_id"], "session-fixture")

    def test_checkpoint_request_refuses_a_completed_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "rank0-session.json").write_text(
                json.dumps({"pid": 43210, "hostname": "host", "status": "completed"}),
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with (
                patch("twen.cli.os.kill") as kill,
                contextlib.redirect_stderr(stderr),
            ):
                code = main(["checkpoint", "request", "--run-dir", str(run_dir)])
            self.assertEqual(code, 2)
            kill.assert_not_called()
            self.assertIn("not running", stderr.getvalue())

    def test_kd_stop_returns_75_and_structured_stdout(self) -> None:
        stopped = KDGenerationStopped(
            stop_file="artifacts/data/base-kd/STOP",
            worker_index=0,
            num_workers=1,
            completed_shards=("shard-000000",),
            assigned_shards=2,
        )
        stdout = io.StringIO()
        with (
            patch("twen.kd.generate_teacher_kd", side_effect=stopped),
            contextlib.redirect_stdout(stdout),
        ):
            code = main(
                [
                    "data",
                    "generate-kd",
                    "--prepared-manifest",
                    "prepared.json",
                    "--output",
                    "kd",
                    "--teacher",
                    "teacher",
                    "--teacher-model-id",
                    "Qwen/example",
                    "--teacher-revision",
                    "a" * 40,
                    "--teacher-manifest-sha256",
                    "b" * 64,
                    "--tokenizer-manifest-sha256",
                    "c" * 64,
                    "--progress",
                    "never",
                ]
            )
        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 75)
        self.assertFalse(payload["ok"])
        self.assertTrue(payload["stopped"])
        self.assertEqual(payload["completed_count"], 1)
        self.assertEqual(payload["assigned_count"], 2)
        self.assertEqual(payload["completed_shards"], ["shard-000000"])

    def test_kd_stop_remains_visible_to_every_worker_until_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / "STOP"
            stop.write_text("stop\n", encoding="utf-8")
            with self.assertRaises(KDGenerationStopped) as raised:
                _raise_if_stopped(
                    str(stop),
                    worker_index=1,
                    num_workers=4,
                    assigned_shards=3,
                )
            self.assertTrue(stop.exists())
            self.assertEqual(raised.exception.worker_index, 1)
            self.assertEqual(raised.exception.assigned_shards, 3)

    def test_kd_preexisting_stop_short_circuits_expensive_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / "STOP"
            stop.touch()
            with self.assertRaises(KDGenerationStopped):
                generate_teacher_kd(
                    prepared_manifest=str(Path(directory) / "missing-prepared.json"),
                    output_root=str(Path(directory) / "kd"),
                    teacher_path=str(Path(directory) / "missing-teacher"),
                    teacher_model_id="Qwen/example",
                    teacher_revision="a" * 40,
                    teacher_manifest_sha256="b" * 64,
                    tokenizer_manifest_sha256="c" * 64,
                    stop_file=str(stop),
                    progress="never",
                )


if __name__ == "__main__":
    unittest.main()
