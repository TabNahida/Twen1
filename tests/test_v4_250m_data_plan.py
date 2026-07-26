from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import pytest
import yaml

from twen.config import ConfigError, load_train_config
from twen.data.sources import load_base_data_recipe, load_resolved_source_lock

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_RECIPE = ROOT / "locks/base-data-sources-v4-primary.json"
COOLDOWN_RECIPE = ROOT / "locks/base-data-sources-v4-cooldown.json"
PRIMARY_LOCK = ROOT / "locks/base-data-sources-v4-primary.resolved.json"
COOLDOWN_LOCK = ROOT / "locks/base-data-sources-v4-cooldown.resolved.json"
ATTESTATION = ROOT / "locks/base-data-sources-v4-250m.capacity-attestation.json"
READINESS = ROOT / "locks/base-dense-v4-250m-pilot.readiness.json"
BLOCKED_CONFIG = ROOT / "configs/base/dense-v4-250m-pilot.blocked.yaml"
CALIBRATION_CONFIG = ROOT / "configs/base/dense-v4-13m-low-lr-calibration.yaml"
V3_FINAL_COMPLETE_SHA256 = "3a21a50e35de74ecd0ff5b8f00aa29ed6c83f746fc2cf97d4da6b0536262b6c7"


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_v4_250m_recipes_are_valid_remote_bound_and_phase_specific() -> None:
    primary = load_base_data_recipe(PRIMARY_RECIPE)
    cooldown = load_base_data_recipe(COOLDOWN_RECIPE)
    assert primary.profiles == {"materialization": 225_270_000}
    assert cooldown.profiles == {"materialization": 25_270_000}
    assert sum(int(source.mix_basis_points or 0) for source in primary.sources) == 10_000
    assert sum(int(source.mix_basis_points or 0) for source in cooldown.sources) == 10_000
    assert load_resolved_source_lock(PRIMARY_LOCK, primary).sources
    assert load_resolved_source_lock(COOLDOWN_LOCK, cooldown).sources

    primary_by_id = {source.source_id: source for source in primary.sources}
    cooldown_by_id = {source.source_id: source for source in cooldown.sources}
    assert "science_pes2o_open" not in primary_by_id
    assert "science_pes2o_open" not in cooldown_by_id
    assert primary_by_id["science_arxiv_open_permissive"].locked_files[0].path == (
        "arxiv-papers-0005.json.gz"
    )
    assert cooldown_by_id["science_arxiv_open_permissive"].locked_files[0].path == (
        "arxiv-papers-0006.json.gz"
    )
    for source_id in set(primary_by_id) & set(cooldown_by_id):
        primary_files = {item.path for item in primary_by_id[source_id].locked_files}
        cooldown_files = {item.path for item in cooldown_by_id[source_id].locked_files}
        assert primary_files.isdisjoint(cooldown_files)


def test_v4_250m_raw_quotas_use_per_source_retention_and_safety() -> None:
    for path in (PRIMARY_RECIPE, COOLDOWN_RECIPE):
        value = _json(path)
        contract = value["stage_contract"]
        assert isinstance(contract, dict)
        rows = contract["sources"]
        assert isinstance(rows, list)
        assert int(contract["raw_materialization_tokens"]) == sum(
            int(row["planned_raw_tokens"]) for row in rows
        )
        for row in rows:
            clean = int(row["required_clean_tokens"])
            retention = float(row["governance_retention_rate"])
            expected = math.ceil(clean / retention * 1.10 / 1000.0) * 1000
            assert int(row["planned_raw_tokens"]) == expected
            assert row["passed"] is False
        by_id = {str(row["source_id"]): row for row in rows}
        assert float(
            by_id["science_arxiv_open_permissive"]["governance_retention_rate"]
        ) == pytest.approx(0.4534)


def test_v4_250m_capacity_attestation_is_complete_but_blocked() -> None:
    value = _json(ATTESTATION)
    config_sha256 = hashlib.sha256(BLOCKED_CONFIG.read_bytes()).hexdigest()
    assert value["config"] == {
        "contains_pending_identity_sentinels": True,
        "path": "configs/base/dense-v4-250m-pilot.blocked.yaml",
        "sha256": config_sha256,
    }
    assert value["status"] == "blocked_pending_governed_preparation_and_launch_gates"
    assert value["launch_enabled"] is False
    assert value["authorizes_training"] is False
    contract = value["training_contract"]
    assert contract == {
        "adapter_lr": 3e-5,
        "adapter_optimizer": "muon",
        "allow_corpus_reuse": False,
        "anchor_kl_weight": 0.0,
        "complete_tail_batch_required": True,
        "cooldown_tokens": 25_000_000,
        "cooldown_start_tokens": 225_000_000,
        "data_mode": "prepared-text",
        "dense_oracle_weight": 0.0,
        "global_batch_tokens": 262_144,
        "gradient_accumulation_steps": 64,
        "hidden_alignment_weight": 0.0,
        "lora_lr": 3e-5,
        "lr_schedule": "cosine",
        "max_tokens": 250_000_000,
        "micro_batch_size": 1,
        "min_lr_ratio": 0.1,
        "mtp_weight": 0.1,
        "native_mtp_head_frozen": True,
        "ntp_weight": 1.0,
        "objective": "base_text_ntp_plus_native_mtp_no_9b_logits",
        "primary_tokens": 225_000_000,
        "scale_lr": 3e-6,
        "scale_optimizer": "adamw",
        "sequence_length": 4096,
        "source_mix_algorithm": "token-deficit-corrected-source-mix-bp-v2",
        "stage": "dense-oracle",
        "teacher_kd_weight": 0.0,
        "teacher_logits_kd": False,
        "track": "base",
        "warmup_tokens": 10_000_000,
        "world_size": 1,
    }
    stages = value["stages"]
    assert stages["primary"]["required_prepared_tokens"] == 225_262_144
    assert stages["primary"]["required_prepared_samples"] == 54_996
    assert stages["cooldown"]["required_prepared_tokens"] == 25_262_144
    assert stages["cooldown"]["required_prepared_samples"] == 6_168
    for stage in stages.values():
        assert stage["resolved_lock"]["passed"] is True
        assert stage["prepared_identity"]["manifest_sha256"] is None
        assert stage["prepared_identity"]["dataset_fingerprint"] is None
        assert stage["prepared_identity"]["source_map_sha256"] is None
        assert stage["prepared_identity"]["passed"] is False
        assert all(row["passed"] is False for row in stage["per_source_capacity"])
    assert all(gate["passed"] is False for gate in value["phase_disjointness"].values())
    assert value["overall"]["passed"] is False


def test_v4_250m_training_config_cannot_be_launched_with_pending_identity() -> None:
    raw = yaml.safe_load(BLOCKED_CONFIG.read_text(encoding="utf-8"))
    assert raw["data"]["allow_corpus_reuse"] is False
    assert raw["data"]["global_batch_tokens"] == 262_144
    assert raw["data"]["max_sequence_length"] == 4096
    assert raw["data"]["micro_batch_size"] == 1
    assert (
        raw["data"]["global_batch_tokens"]
        // (raw["data"]["max_sequence_length"] * raw["data"]["micro_batch_size"])
        == 64
    )
    assert raw["data"]["quality_cooldown_start_tokens"] == 225_000_000
    assert raw["checkpoint"]["every_steps"] == 50
    assert raw["optimizer"]["max_tokens"] == 250_000_000
    assert raw["optimizer"]["adapter_optimizer"] == "muon"
    assert raw["optimizer"]["adapter_lr"] == pytest.approx(3e-5)
    assert raw["optimizer"]["lora_lr"] == pytest.approx(3e-5)
    assert raw["optimizer"]["scale_lr"] == pytest.approx(3e-6)
    assert raw["optimizer"]["warmup_tokens"] == 10_000_000
    assert raw["optimizer"]["lr_schedule"] == "cosine"
    assert raw["optimizer"]["min_lr_ratio"] == pytest.approx(0.1)
    assert raw["losses"]["ntp"] == pytest.approx(1.0)
    assert raw["losses"]["mtp"] == pytest.approx(0.1)
    for name in ("teacher_kd", "anchor_kl", "hidden_alignment", "dense_oracle"):
        assert raw["losses"][name] == pytest.approx(0.0)
    assert str(raw["data"]["manifest_sha256"]).startswith("PENDING_")
    with pytest.raises(ConfigError, match="manifest_sha256"):
        load_train_config(BLOCKED_CONFIG)


def test_v4_250m_readiness_binds_current_blocked_config() -> None:
    readiness = _json(READINESS)
    assert readiness["config_path"] == ("configs/base/dense-v4-250m-pilot.blocked.yaml")
    assert readiness["config_sha256"] == hashlib.sha256(BLOCKED_CONFIG.read_bytes()).hexdigest()

    readiness = _json(READINESS)
    assert readiness["launch_enabled"] is False
    assert readiness["training_started"] is False
    assert readiness["resolved_lock_identities"]["primary"]["passed"] is True
    assert readiness["resolved_lock_identities"]["cooldown"]["passed"] is True
    assert readiness["fork_from"].endswith("step-000000001912-milestone-complete")
    contract = readiness["contract"]
    assert contract["primary_tokens"] == 225_000_000
    assert contract["cooldown_tokens"] == 25_000_000
    assert contract["sequence_length"] == 4096
    assert contract["world_size"] == 1
    assert contract["micro_batch_size"] == 1
    assert contract["gradient_accumulation_steps"] == 64
    assert contract["global_batch_tokens"] == 262_144
    assert contract["adapter_optimizer"] == "muon"
    assert contract["scale_optimizer"] == "adamw"
    assert contract["adapter_lr"] == pytest.approx(3e-5)
    assert contract["lora_lr"] == pytest.approx(3e-5)
    assert contract["scale_lr"] == pytest.approx(3e-6)
    assert contract["warmup_tokens"] == 10_000_000
    assert contract["lr_schedule"] == "cosine"
    assert contract["min_lr_ratio"] == pytest.approx(0.1)
    assert contract["ntp_weight"] == pytest.approx(1.0)
    assert contract["mtp_weight"] == pytest.approx(0.1)
    assert contract["native_mtp_head_frozen"] is True
    assert contract["teacher_logits_kd"] is False
    for name in (
        "teacher_kd_weight",
        "anchor_kl_weight",
        "hidden_alignment_weight",
        "dense_oracle_weight",
    ):
        assert contract[name] == pytest.approx(0.0)
    assert (
        readiness["fork_policy"]["required_checkpoint_complete_sha256"] == V3_FINAL_COMPLETE_SHA256
    )
    assert readiness["fork_policy"]["forbidden_warm_starts"] == [
        "runs/base-dense-v4-16m-smoke",
        "runs/base-dense-v4-13m-low-lr-calibration",
    ]
    calibration = readiness["calibration_gate"]
    assert calibration["required"] is True
    assert calibration["passed"] is False
    assert calibration["authorizes_training"] is False
    assert calibration["config"] == {
        "path": "configs/base/dense-v4-13m-low-lr-calibration.yaml",
        "sha256": hashlib.sha256(CALIBRATION_CONFIG.read_bytes()).hexdigest(),
    }
    assert calibration["required_fork_checkpoint"]["complete_sha256"] == V3_FINAL_COMPLETE_SHA256
    assert calibration["required_candidate_checkpoints"] == {
        "final_milestone_required": True,
        "global_steps": [40, 50],
        "same_frozen_v3_validation_contract": True,
    }
    thresholds = calibration["hard_thresholds"]
    assert thresholds["best_aggregate_nll_lte"] == pytest.approx(2.3766688031972105)
    assert thresholds["final_aggregate_nll_lte"] == pytest.approx(2.3766688031972105)
    assert thresholds["chinese_source_nll_lte"] == pytest.approx(3.656194313354557)
    assert thresholds["final_scale_relative_l2_lte"] == pytest.approx(0.05)
    assert thresholds["reused_sequences_eq"] == 0
    assert thresholds["reused_tokens_eq"] == 0
    assert thresholds["all_reference_epochs_eq"] == 0
    assert thresholds["clip_fraction_eq"] == pytest.approx(0.0)
    assert calibration["observed"] is None
    assert all(
        value is None
        for identity in calibration["required_authenticated_evidence"].values()
        for value in identity.values()
    )
    formal_validation = readiness["formal_validation_gate"]
    assert formal_validation["required"] is True
    assert formal_validation["passed"] is False
    assert formal_validation["authorizes_training"] is False
    assert (
        formal_validation["v3_final_frozen_validation_baseline"]["checkpoint_complete_sha256"]
        == V3_FINAL_COMPLETE_SHA256
    )
    assert formal_validation["train_validation_union_disjointness"][
        "near_duplicate_threshold"
    ] == pytest.approx(0.8)
    evaluation = readiness["pause_evaluation_policy"]
    assert evaluation["checkpoint_every_steps"] == 50
    assert evaluation["pause_at_committed_tokens"] == [
        13_000_000,
        26_000_000,
        52_000_000,
        105_000_000,
        157_000_000,
        210_000_000,
        223_000_000,
        236_000_000,
        250_000_000,
    ]
    assert evaluation["enforcement"] == "external_governed_controller"
    assert evaluation["controller_implemented"] is False
    assert evaluation["current_launch_command_auto_pauses"] is False
    assert evaluation["current_launch_command_runs_validation"] is False
    assert "science_arxiv_open_permissive" in (evaluation["required_additional_validation_sources"])
    assert evaluation["hard_stop"]["scale_relative_l2_gt"] == pytest.approx(0.05)
    assert readiness["launch_command_capabilities"] == {
        "automatically_enforces_post_launch_hard_stops": False,
        "automatically_pauses_at_policy_thresholds": False,
        "automatically_runs_checkpoint_validation": False,
        "current_blocked_config_rejects_training": True,
        "starts_training_when_explicitly_invoked": False,
    }
    assert readiness["launch_command_after_all_gates_pass"] is None
    assert (
        readiness["launch_command_status"]
        == "pending_final_config_authorization_and_controller"
    )
    assert readiness["blockers"]
    assert any("13M low-LR calibration" in blocker for blocker in readiness["blockers"])
    assert any("formal train/validation" in blocker for blocker in readiness["blockers"])
    assert any("external governed pause" in blocker for blocker in readiness["blockers"])
    assert any("final launch config" in blocker for blocker in readiness["blockers"])


def test_v4_250m_generated_recipe_and_config_bytes_are_reproducible() -> None:
    subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "scripts/prepare_v4_250m_data_plan.py"),
            "--check",
        ],
        cwd=ROOT,
        check=True,
    )
