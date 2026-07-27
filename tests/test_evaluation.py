from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from twen.evaluation import (
    _acceptance_metrics,
    _canonical_sha256,
    _load_evaluation_baseline,
    _nll_sum,
    _normalize_evaluation_device,
    _read_progress,
    _validate_inference_checkpoint_lineage,
    evaluate_nll,
)
from twen.io.locking import FileLock, FileLockTimeout
from twen.utils import sha256_file


class EvaluationCpuTest(unittest.TestCase):
    def test_cuda_default_is_normalized_to_an_explicit_device(self) -> None:
        self.assertEqual(_normalize_evaluation_device("cuda"), "cuda:0")
        self.assertEqual(_normalize_evaluation_device("cuda:1"), "cuda:1")
        self.assertEqual(_normalize_evaluation_device("cpu"), "cpu")

    def test_evaluate_nll_holds_lock_for_entire_worker_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation"
            expected = {"ok": True}

            def worker(**kwargs: object) -> dict[str, bool]:
                self.assertEqual(kwargs["output_dir"], output.resolve())
                with self.assertRaises(FileLockTimeout):
                    FileLock(output / ".eval.lock", timeout_seconds=0).acquire()
                return expected

            with patch(
                "twen.evaluation._evaluate_nll_while_locked",
                side_effect=worker,
            ):
                result = evaluate_nll(
                    config_path="unused",
                    checkpoint_path="unused",
                    prepared_manifest_path="unused",
                    prepared_manifest_sha256="a" * 64,
                    output_dir=str(output),
                )
            self.assertEqual(result, expected)
            with FileLock(output / ".eval.lock", timeout_seconds=0):
                pass

    def test_evaluate_nll_releases_lock_after_worker_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation"
            with (
                patch(
                    "twen.evaluation._evaluate_nll_while_locked",
                    side_effect=RuntimeError("simulated worker failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "simulated worker failure"),
            ):
                evaluate_nll(
                    config_path="unused",
                    checkpoint_path="unused",
                    prepared_manifest_path="unused",
                    prepared_manifest_sha256="a" * 64,
                    output_dir=str(output),
                )
            with FileLock(output / ".eval.lock", timeout_seconds=0):
                pass

    def test_evaluate_nll_binds_resolved_output_across_symlink_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            output_link = root / "output"
            output_link.symlink_to(first, target_is_directory=True)

            def worker(**kwargs: object) -> dict[str, bool]:
                output_link.unlink()
                output_link.symlink_to(second, target_is_directory=True)
                bound_output = kwargs["output_dir"]
                self.assertEqual(bound_output, first.resolve())
                self.assertIsInstance(bound_output, Path)
                assert isinstance(bound_output, Path)
                (bound_output / "BOUND").write_text("first\n", encoding="utf-8")
                with self.assertRaises(FileLockTimeout):
                    FileLock(bound_output / ".eval.lock", timeout_seconds=0).acquire()
                return {"ok": True}

            with patch(
                "twen.evaluation._evaluate_nll_while_locked",
                side_effect=worker,
            ):
                evaluate_nll(
                    config_path="unused",
                    checkpoint_path="unused",
                    prepared_manifest_path="unused",
                    prepared_manifest_sha256="a" * 64,
                    output_dir=str(output_link),
                )
            self.assertTrue((first / "BOUND").is_file())
            self.assertFalse((second / "BOUND").exists())
            self.assertFalse((second / ".eval.lock").exists())

    def test_plan_exists_and_lock_is_held_before_model_and_checkpoint_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "evaluation"
            prepared_path = root / "prepared.json"
            prepared_path.write_text("{}\n", encoding="utf-8")
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "COMPLETE").write_text("fixed\n", encoding="utf-8")
            config = SimpleNamespace(
                run_id="base-v1",
                track="base",
                stage="dense-oracle",
                runtime=SimpleNamespace(bf16=False),
                sources=SimpleNamespace(
                    tokenizer=SimpleNamespace(manifest_sha256="tokenizer"),
                ),
                architecture=SimpleNamespace(
                    expert_initialization="donor",
                    top_k=2,
                ),
                data=SimpleNamespace(global_batch_tokens=4096),
                checkpoint=SimpleNamespace(output_dir=str(root / "runs")),
            )
            report = SimpleNamespace(
                config_fingerprint="config",
                data_fingerprint="data",
            )
            prepared = SimpleNamespace(
                tokenizer_sha256="tokenizer",
                dataset_fingerprint="dataset",
                generator_source_sha256="generator",
            )
            metadata = {
                "global_step": 7,
                "committed_tokens": 28_672,
                "kind": "periodic",
                "tag": None,
            }
            events: list[str] = []

            def assert_plan_and_lock(label: str) -> None:
                plan_path = output / "PLAN.json"
                self.assertTrue(plan_path.is_file())
                self.assertEqual(json.loads(plan_path.read_text())["device_type"], "cpu")
                with self.assertRaises(FileLockTimeout):
                    FileLock(output / ".eval.lock", timeout_seconds=0).acquire()
                events.append(label)

            def build_model(*_args: object, **_kwargs: object) -> SimpleNamespace:
                assert_plan_and_lock("build")
                return SimpleNamespace(model=object(), transfer_modules=[])

            def load_checkpoint(*_args: object, **_kwargs: object) -> SimpleNamespace:
                assert_plan_and_lock("load")
                return SimpleNamespace(path=checkpoint, metadata=metadata)

            manager = SimpleNamespace(load=load_checkpoint)
            role_result = {
                "mean_nll": 2.0,
                "perplexity": math.exp(2.0),
                "predicted_tokens": 16,
                "sequences": 1,
            }
            with (
                patch("twen.evaluation.enforce_offline_environment"),
                patch("twen.evaluation.load_train_config", return_value=config),
                patch("twen.evaluation.run_inference_preflight", return_value=report),
                patch(
                    "twen.evaluation.validate_prepared_corpus_for_inference",
                    return_value=prepared,
                ),
                patch(
                    "twen.evaluation._inspect_inference_evaluation_checkpoint",
                    return_value=(manager, checkpoint, metadata, {"mode": "test"}),
                ),
                patch(
                    "twen.evaluation._dense_control_fingerprint",
                    return_value="control",
                ),
                patch("twen.training.builder.build_transfer_model", side_effect=build_model),
                patch(
                    "twen.training.stateful.TrainableModelState", side_effect=lambda model: model
                ),
                patch("twen.evaluation._evaluate_role", return_value=role_result),
                patch("twen.evaluation._acceptance_metrics", return_value={}),
            ):
                result = evaluate_nll(
                    config_path="config.yaml",
                    checkpoint_path=str(checkpoint),
                    prepared_manifest_path=str(prepared_path),
                    prepared_manifest_sha256=sha256_file(prepared_path),
                    output_dir=str(output),
                    roles=("candidate",),
                    device="cpu",
                )
            self.assertEqual(events, ["build", "load"])
            self.assertEqual(result["roles"]["candidate"]["predicted_tokens"], 16)
            plan = json.loads((output / "PLAN.json").read_text(encoding="utf-8"))
            self.assertEqual(
                plan["prepared_manifest_expected_sha256"],
                sha256_file(prepared_path),
            )
            self.assertEqual(plan["prepared_manifest_identity_source"], "caller_pinned")

    def test_evaluate_nll_rejects_a_manifest_that_differs_from_external_pin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared_path = root / "prepared.json"
            prepared_path.write_text("{}\n", encoding="utf-8")
            with (
                patch("twen.evaluation.enforce_offline_environment"),
                patch("twen.evaluation.load_train_config", return_value=object()),
                patch(
                    "twen.evaluation.run_inference_preflight",
                    return_value=object(),
                ),
                patch(
                    "twen.evaluation.validate_prepared_corpus_for_inference"
                ) as validate,
                self.assertRaisesRegex(
                    ValueError,
                    "differs from --prepared-manifest-sha256",
                ),
            ):
                evaluate_nll(
                    config_path="config.yaml",
                    checkpoint_path="unused",
                    prepared_manifest_path=str(prepared_path),
                    prepared_manifest_sha256="0" * 64,
                    output_dir=str(root / "evaluation"),
                    device="cpu",
                )
            validate.assert_not_called()

    def test_inference_lineage_allows_recorded_source_tree_drift_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "resolved_config.yaml"
            config_path.write_text(
                "losses:\n  ntp: 1.0\n  teacher_kd: 1.0\n",
                encoding="utf-8",
            )
            source = lambda digest: SimpleNamespace(manifest_sha256=digest)  # noqa: E731
            config = SimpleNamespace(
                run_id="base-v1",
                stage="dense-oracle",
                sources=SimpleNamespace(
                    backbone=source("a" * 64),
                    donor=source("b" * 64),
                    teacher=source("b" * 64),
                    tokenizer=source("a" * 64),
                    folded_experts_sha256=None,
                ),
                data=SimpleNamespace(
                    manifest_sha256="c" * 64,
                    teacher_kd_manifest_sha256="d" * 64,
                    global_batch_tokens=262144,
                ),
                architecture=SimpleNamespace(top_k=2),
            )
            report = SimpleNamespace(
                calibration_fingerprints=(("layer_map", "e" * 64),),
                source_tree_sha256="new-tree",
                config_fingerprint="new-critical",
            )
            metadata = {
                "critical_fingerprint": "saved-critical",
                "extra": {
                    "source_manifests": {
                        "backbone": "a" * 64,
                        "donor": "b" * 64,
                        "teacher": "b" * 64,
                        "tokenizer": "a" * 64,
                        "folded_experts": None,
                    },
                    "calibration_artifacts": {"layer_map": "e" * 64},
                    "data_manifest_sha256": "c" * 64,
                    "teacher_kd_manifest_sha256": "d" * 64,
                    "source_tree_sha256": "old-tree",
                },
                "trainer_state": {
                    "run_id": "base-v1",
                    "stage": "dense-oracle",
                    "global_batch_tokens": 262144,
                    "loss_weights": {"ntp": 1.0, "teacher_kd": 1.0},
                    "top_k": None,
                },
            }
            identity = _validate_inference_checkpoint_lineage(
                config_path=config_path,
                config=config,
                report=report,
                metadata=metadata,
            )
            self.assertFalse(identity["source_tree_match"])
            self.assertFalse(identity["exact_training_fingerprint_match"])
            self.assertIn("not_exact_resume", identity["mode"])

            metadata["extra"]["source_manifests"]["backbone"] = "different"
            with self.assertRaisesRegex(ValueError, "source manifests"):
                _validate_inference_checkpoint_lineage(
                    config_path=config_path,
                    config=config,
                    report=report,
                    metadata=metadata,
                )

    def test_nll_sum_uses_causal_shift_and_label_mask(self) -> None:
        logits = torch.zeros((1, 3, 2), dtype=torch.float32)
        labels = torch.tensor([[0, 1, -100]])
        loss, tokens = _nll_sum(logits, labels)
        self.assertEqual(tokens, 1)
        self.assertAlmostEqual(float(loss), math.log(2.0), places=6)

    def test_acceptance_metrics_encode_dense_and_sparse_gates(self) -> None:
        roles = {
            "shared": {"mean_nll": 3.0},
            "candidate": {"mean_nll": 2.0},
            "teacher": {"mean_nll": 1.0},
            "dense-oracle": {"mean_nll": 1.9},
        }
        dense = _acceptance_metrics("dense-oracle", roles)
        self.assertAlmostEqual(dense["teacher_gap_closed_fraction"], 0.5)
        self.assertTrue(dense["dense_gap_gate_pass"])
        stage_b = {
            "roles": {
                "shared": {"mean_nll": 3.0},
                "candidate": {"mean_nll": 1.9},
            }
        }
        sparse = _acceptance_metrics("sparse", roles, dense_baseline=stage_b)
        self.assertAlmostEqual(
            sparse["dense_oracle_gain_retained_fraction"],
            1.0 / 1.1,
        )
        self.assertTrue(sparse["sparse_retention_gate_pass"])
        random = {
            "roles": {
                "shared": {"mean_nll": 3.0},
                "candidate": {"mean_nll": 2.4},
            }
        }
        compared = _acceptance_metrics(
            "dense-oracle",
            roles,
            random_baseline=random,
        )
        self.assertAlmostEqual(compared["donor_over_random_nll_improvement"], 0.4)
        self.assertTrue(compared["donor_beats_random_control"])

    def test_corrupt_microbatch_progress_restarts_from_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            path.write_text("{torn")
            self.assertEqual(
                _read_progress(path, fingerprint="f", sequence_count=4),
                {"next_sequence": 0, "nll_sum": 0.0, "predicted_tokens": 0},
            )
            self.assertFalse(path.exists())

    def test_baseline_authenticates_stage_corpus_checkpoint_and_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {
                "schema_version": 1,
                "kind": "twen_nll_evaluation_plan",
                "prepared_manifest_sha256": "a" * 64,
                "prepared_dataset_fingerprint": "dataset-v1",
                "checkpoint": str(root / "checkpoint"),
                "checkpoint_complete_sha256": "b" * 64,
                "config_fingerprint": "c" * 64,
                "checkpoint_state": {
                    "global_step": 10,
                    "committed_tokens": 100,
                    "kind": "milestone",
                    "tag": "complete",
                },
                "batch_size": 1,
                "device_type": "cuda",
                "dtype": "bfloat16",
            }
            plan["plan_fingerprint"] = _canonical_sha256(plan)
            plan_path = root / "PLAN.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            manifest = {
                "schema_version": 1,
                "kind": "twen_nll_evaluation",
                "plan_sha256": sha256_file(plan_path),
                "plan_fingerprint": plan["plan_fingerprint"],
                "track": "base",
                "stage": "dense-oracle",
                "expert_initialization": "donor",
                "dense_control_fingerprint": "control-v1",
                "checkpoint_state": plan["checkpoint_state"],
                "roles": {
                    "candidate": {"mean_nll": 2.0, "predicted_tokens": 10},
                    "shared": {"mean_nll": 3.0, "predicted_tokens": 10},
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (root / "COMPLETE").write_text(
                f"{sha256_file(manifest_path)}\n",
                encoding="ascii",
            )
            payload, identity = _load_evaluation_baseline(
                manifest_path,
                expected_track="base",
                expected_stage="dense-oracle",
                expected_expert_initialization="donor",
                prepared_manifest_sha256="a" * 64,
                prepared_dataset_fingerprint="dataset-v1",
                expected_batch_size=1,
                expected_device_type="cuda",
                expected_dtype="bfloat16",
                expected_control_fingerprint="control-v1",
                expected_checkpoint=str(root / "checkpoint"),
                expected_checkpoint_complete_sha256="b" * 64,
                expected_config_fingerprint="c" * 64,
            )
            self.assertEqual(payload["roles"]["candidate"]["mean_nll"], 2.0)
            self.assertEqual(identity["sha256"], sha256_file(manifest_path))
            with self.assertRaisesRegex(ValueError, "checkpoint COMPLETE"):
                _load_evaluation_baseline(
                    manifest_path,
                    expected_track="base",
                    expected_stage="dense-oracle",
                    expected_expert_initialization="donor",
                    prepared_manifest_sha256="a" * 64,
                    prepared_dataset_fingerprint="dataset-v1",
                    expected_batch_size=1,
                    expected_device_type="cuda",
                    expected_dtype="bfloat16",
                    expected_checkpoint_complete_sha256="different",
                )


if __name__ == "__main__":
    unittest.main()
