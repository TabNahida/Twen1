from __future__ import annotations

import dataclasses
import json
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from twen.preflight import (
    BatchGeometry,
    DataGovernanceStatus,
    PreflightReport,
    TrainingPreflightError,
    _canonical_config_digest,
    _listener_family,
    _preflight_report_from_payload,
    _preflight_report_payload,
    _prepared_data_governance,
    _qwen35_rope_buffer_bytes,
    _validate_calibration_contract,
    _validate_peer_preflight_requests,
)
from twen.training.engine import _data_governance_log_fields
from twen.utils import atomic_write_json, sha256_file


class CalibrationPreflightContractTest(unittest.TestCase):
    def test_qwen35_rope_buffer_inventory_prefers_nested_partial_factor(self) -> None:
        self.assertEqual(
            _qwen35_rope_buffer_bytes(
                {
                    "head_dim": 256,
                    "partial_rotary_factor": None,
                    "rope_parameters": {"partial_rotary_factor": 0.25},
                }
            ),
            256,
        )
        self.assertEqual(
            _qwen35_rope_buffer_bytes({"head_dim": 256, "partial_rotary_factor": 0.5}),
            512,
        )
        with self.assertRaisesRegex(TrainingPreflightError, "RoPE dimensions"):
            _qwen35_rope_buffer_bytes({"head_dim": 0})

    def test_rank_zero_preflight_report_round_trips_without_cuda(self) -> None:
        report = PreflightReport(
            config_fingerprint="a" * 64,
            data_fingerprint="b" * 64,
            source_tree_sha256="d" * 64,
            batch=BatchGeometry(4, 4096, 16, 262144),
            checked_paths=("model", "data"),
            calibration_fingerprints=(("layer_map", "c" * 64),),
            data_governance=DataGovernanceStatus(
                lineage_kind="authenticated_extracted_corpus",
                research_only=True,
                ready_for_training=False,
                pending_audits=("near_dedup", "pii"),
                warning="RESEARCH-ONLY DATA: fixture warning",
            ),
            teacher_cpu_shadow_bytes=8_625_611_776,
            teacher_gpu_stage_bytes=8_625_611_776,
            activation_checkpoint_layer_count=4,
            hidden_alignment_activation_checkpoint_layer_count=24,
            activation_checkpoint_layer_indices=(0, 8, 15, 23),
            hidden_alignment_activation_checkpoint_layer_indices=tuple(range(24)),
            dense_transfer_execution="differentiable_folded",
            dense_transfer_checkpoint_layer_count=20,
            hidden_alignment_dense_transfer_checkpoint_layer_count=0,
            dense_transfer_token_checkpoint_layer_indices=tuple(
                layer for layer in range(24) if layer not in {0, 8, 15, 23}
            ),
            hidden_alignment_dense_transfer_token_checkpoint_layer_indices=(),
            quality_cooldown_enabled=True,
            quality_cooldown_start_tokens=450_000_000,
            quality_cooldown_dataset_fingerprint="e" * 64,
            quality_cooldown_sequence_count=12_500,
            quality_cooldown_token_count=50_000_000,
            quality_cooldown_selected_shard_ids=("shard-1", "shard-7"),
            quality_cooldown_source_mix_token_counts=(
                ("math", 20_000_000),
                ("science", 30_000_000),
            ),
        )
        payload = _preflight_report_payload(report)
        self.assertEqual(_preflight_report_from_payload(payload), report)
        self.assertEqual(payload["source_tree_sha256"], "d" * 64)
        self.assertEqual(payload["teacher_cpu_shadow_bytes"], 8_625_611_776)
        self.assertEqual(payload["activation_checkpoint_layer_count"], 4)
        self.assertEqual(payload["hidden_alignment_activation_checkpoint_layer_count"], 24)
        self.assertEqual(payload["activation_checkpoint_layer_indices"], [0, 8, 15, 23])
        self.assertEqual(payload["dense_transfer_execution"], "differentiable_folded")
        self.assertEqual(payload["dense_transfer_checkpoint_layer_count"], 20)
        self.assertEqual(payload["hidden_alignment_dense_transfer_checkpoint_layer_count"], 0)
        self.assertEqual(
            payload["dense_transfer_token_checkpoint_layer_indices"],
            [layer for layer in range(24) if layer not in {0, 8, 15, 23}],
        )
        self.assertEqual(
            payload["quality_cooldown"],
            {
                "enabled": True,
                "start_tokens": 450_000_000,
                "dataset_fingerprint": "e" * 64,
                "sequence_count": 12_500,
                "token_count": 50_000_000,
                "selected_shard_ids": ["shard-1", "shard-7"],
                "source_mix_token_counts": [
                    ["math", 20_000_000],
                    ["science", 30_000_000],
                ],
            },
        )
        self.assertTrue(payload["data_governance"]["research_only"])
        self.assertEqual(
            payload["data_governance"]["pending_audits"],
            ["near_dedup", "pii"],
        )
        invalid = dict(payload)
        invalid.pop("source_tree_sha256")
        with self.assertRaisesRegex(TrainingPreflightError, "source tree SHA256"):
            _preflight_report_from_payload(invalid)
        invalid = {**payload, "activation_checkpoint_layer_count": True}
        with self.assertRaisesRegex(TrainingPreflightError, "checkpoint layer count"):
            _preflight_report_from_payload(invalid)
        invalid = {**payload, "dense_transfer_checkpoint_layer_count": True}
        with self.assertRaisesRegex(TrainingPreflightError, "checkpoint layer count"):
            _preflight_report_from_payload(invalid)
        invalid = {
            **payload,
            "hidden_alignment_dense_transfer_checkpoint_layer_count": 1,
            "hidden_alignment_dense_transfer_token_checkpoint_layer_indices": [0],
        }
        with self.assertRaisesRegex(TrainingPreflightError, "nested outer/inner"):
            _preflight_report_from_payload(invalid)
        legacy_payload = dict(payload)
        for field in (
            "hidden_alignment_activation_checkpoint_layer_count",
            "dense_transfer_checkpoint_layer_count",
            "hidden_alignment_dense_transfer_checkpoint_layer_count",
        ):
            legacy_payload.pop(field)
        legacy = _preflight_report_from_payload(legacy_payload)
        self.assertIsNone(legacy.hidden_alignment_activation_checkpoint_layer_count)
        self.assertIsNone(legacy.dense_transfer_checkpoint_layer_count)
        self.assertIsNone(legacy.hidden_alignment_dense_transfer_checkpoint_layer_count)

    def test_research_only_governance_is_visible_but_non_blocking(self) -> None:
        prepared = SimpleNamespace(
            lineage={
                "kind": "authenticated_extracted_corpus",
                "research_only": True,
                "ready_for_training": False,
                "pending_audits": ["near_dedup", "pii"],
            }
        )
        status = _prepared_data_governance(prepared)
        self.assertTrue(status.research_only)
        self.assertFalse(status.ready_for_training)
        self.assertEqual(status.pending_audits, ("near_dedup", "pii"))
        self.assertIn("RESEARCH-ONLY DATA", status.warning or "")
        report = PreflightReport(
            config_fingerprint="a" * 64,
            data_fingerprint="b" * 64,
            source_tree_sha256="d" * 64,
            batch=BatchGeometry(1, 4096, 64, 262144),
            checked_paths=("data",),
            data_governance=status,
        )
        self.assertEqual(
            _data_governance_log_fields(report),
            {"data_governance": dataclasses.asdict(status)},
        )

    def test_ready_data_cannot_retain_pending_audits(self) -> None:
        prepared = SimpleNamespace(
            lineage={
                "kind": "authenticated_extracted_corpus",
                "research_only": False,
                "ready_for_training": True,
                "pending_audits": ["near_dedup"],
            }
        )
        with self.assertRaisesRegex(TrainingPreflightError, "audits remain pending"):
            _prepared_data_governance(prepared)

    def test_fully_audited_data_has_no_governance_warning(self) -> None:
        prepared = SimpleNamespace(
            lineage={
                "kind": "authenticated_extracted_corpus",
                "research_only": False,
                "ready_for_training": True,
                "pending_audits": [],
            }
        )
        status = _prepared_data_governance(prepared)
        self.assertTrue(status.ready_for_training)
        self.assertFalse(status.research_only)
        self.assertIsNone(status.warning)

    def test_coordinated_preflight_requires_matching_rank_config_and_node_scan(self) -> None:
        config = SimpleNamespace(canonical_dict=lambda: {"run_id": "same"})
        report = PreflightReport(
            config_fingerprint="a" * 64,
            data_fingerprint="b" * 64,
            source_tree_sha256="d" * 64,
            batch=BatchGeometry(3, 8, 2, 48),
            checked_paths=("model",),
        )
        payload = _preflight_report_payload(report)
        digest = _canonical_config_digest(config)
        requests = {
            1: {
                "config_digest": digest,
                "node_id": "node-0",
                "local_rank": 1,
                "local_report": None,
                "local_error": None,
            },
            2: {
                "config_digest": digest,
                "node_id": "node-1",
                "local_rank": 0,
                "local_report": payload,
                "local_error": None,
            },
        }
        _validate_peer_preflight_requests(
            config=config,
            report=report,
            world_size=3,
            rank_zero_node="node-0",
            requests=requests,
        )
        requests[2] = {
            **requests[2],
            "local_report": {**payload, "source_tree_sha256": "e" * 64},
        }
        with self.assertRaisesRegex(TrainingPreflightError, "different local artifacts"):
            _validate_peer_preflight_requests(
                config=config,
                report=report,
                world_size=3,
                rank_zero_node="node-0",
                requests=requests,
            )
        requests[2] = {**requests[2], "local_report": payload}
        requests[1] = {**requests[1], "config_digest": "different"}
        with self.assertRaisesRegex(TrainingPreflightError, "different complete"):
            _validate_peer_preflight_requests(
                config=config,
                report=report,
                world_size=3,
                rank_zero_node="node-0",
                requests=requests,
            )

    def test_preflight_listener_matches_ipv4_and_ipv6_master_addresses(self) -> None:
        self.assertEqual(_listener_family("127.0.0.1"), socket.AF_INET)
        if socket.has_ipv6:
            self.assertEqual(_listener_family("::1"), socket.AF_INET6)

    def _artifacts(self, root: Path):
        backbone = SimpleNamespace(manifest_sha256="a" * 64, local_path="backbone")
        donor = SimpleNamespace(manifest_sha256="b" * 64, local_path="donor")
        config = SimpleNamespace(
            track="base",
            architecture=SimpleNamespace(
                student_layers=2,
                donor_layers=3,
                student_hidden_size=2,
                donor_hidden_size=3,
                donor_intermediate_size=4,
                num_experts=2,
                expert_intermediate_size=2,
            ),
            sources=SimpleNamespace(backbone=backbone, donor=donor),
        )
        layer_map = root / "layer_map.json"
        atomic_write_json(
            layer_map,
            {
                "schema_version": 2,
                "kind": "monotonic_same_type_cka",
                "student_to_donor": [0, 2],
                "pairs": [
                    {
                        "student_layer": 0,
                        "donor_layer": 0,
                        "layer_type": "linear",
                        "cka": 0.5,
                    },
                    {
                        "student_layer": 1,
                        "donor_layer": 2,
                        "layer_type": "full",
                        "cka": 0.5,
                    },
                ],
                "lineage": {
                    "student_model": {"manifest_sha256": "a" * 64},
                    "donor_model": {"manifest_sha256": "b" * 64},
                },
            },
        )
        channel_map = root / "channel_map.json"
        atomic_write_json(
            channel_map,
            {
                "schema_version": 2,
                "kind": "channel_partition_map",
                "lineage": {
                    "layer_map_sha256": sha256_file(layer_map),
                    "donor_source": {"manifest_sha256": "b" * 64},
                },
                "layers": {
                    str(layer): {
                        "donor_layer": donor_layer,
                        "indices": [[0, 1], [2, 3]],
                        "num_channels": 4,
                        "num_experts": 2,
                        "expert_size": 2,
                    }
                    for layer, donor_layer in enumerate((0, 2))
                },
            },
        )
        adapter = root / "adapters.safetensors"
        save_file(
            {
                **{f"layers.{layer}.A": torch.zeros(3, 2) for layer in range(2)},
                **{f"layers.{layer}.B": torch.zeros(2, 3) for layer in range(2)},
            },
            adapter,
        )
        atomic_write_json(
            adapter.with_suffix(".json"),
            {
                "schema_version": 2,
                "kind": "bidirectional_ridge_adapters",
                "artifact": {"sha256": sha256_file(adapter)},
                "layer_map": {"sha256": sha256_file(layer_map)},
                "track": "base",
                "model_sources": {
                    "student": {"manifest_sha256": "a" * 64},
                    "donor": {"manifest_sha256": "b" * 64},
                },
            },
        )
        audit = SimpleNamespace(
            student_layer_types=("linear", "full"),
            donor_layer_types=("linear", "linear", "full"),
        )
        return config, layer_map, channel_map, adapter, audit

    def test_full_contract_accepts_exact_map_partition_and_adapter_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            values = self._artifacts(Path(directory))
            config, layer_map, channel_map, adapter, audit = values
            with patch("twen.modeling.audit_source_configs", return_value=audit):
                extras = _validate_calibration_contract(
                    config,
                    layer_map,
                    channel_map,
                    adapter,
                )
            self.assertEqual(extras[0][0], "adapter_init_metadata")

    def test_contract_rejects_duplicate_channel_indices_before_model_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            values = self._artifacts(Path(directory))
            config, layer_map, channel_map, adapter, audit = values
            payload = json.loads(channel_map.read_text())
            payload["layers"]["0"]["indices"] = [[0, 1], [1, 3]]
            atomic_write_json(channel_map, payload)
            with (
                patch("twen.modeling.audit_source_configs", return_value=audit),
                self.assertRaisesRegex(
                    TrainingPreflightError,
                    "complete channel partition",
                ),
            ):
                _validate_calibration_contract(
                    config,
                    layer_map,
                    channel_map,
                    adapter,
                )


if __name__ == "__main__":
    unittest.main()
