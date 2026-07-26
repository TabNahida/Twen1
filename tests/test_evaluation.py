from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from twen.evaluation import (
    _acceptance_metrics,
    _canonical_sha256,
    _load_evaluation_baseline,
    _nll_sum,
    _normalize_evaluation_device,
    _read_progress,
    _validate_inference_checkpoint_lineage,
)
from twen.utils import sha256_file


class EvaluationCpuTest(unittest.TestCase):
    def test_cuda_default_is_normalized_to_an_explicit_device(self) -> None:
        self.assertEqual(_normalize_evaluation_device("cuda"), "cuda:0")
        self.assertEqual(_normalize_evaluation_device("cuda:1"), "cuda:1")
        self.assertEqual(_normalize_evaluation_device("cpu"), "cpu")

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
