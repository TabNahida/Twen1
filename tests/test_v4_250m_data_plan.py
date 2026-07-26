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
    assert value["status"] == "blocked_pending_materialization"
    assert value["launch_enabled"] is False
    contract = value["training_contract"]
    assert contract == {
        "allow_corpus_reuse": False,
        "complete_tail_batch_required": True,
        "cooldown_start_tokens": 225_000_000,
        "global_batch_tokens": 262_144,
        "max_tokens": 250_000_000,
        "sequence_length": 4096,
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
    assert readiness["contract"]["adapter_lr"] == pytest.approx(3e-5)
    assert readiness["contract"]["lora_lr"] == pytest.approx(3e-5)
    assert readiness["contract"]["scale_lr"] == pytest.approx(3e-6)
    assert readiness["contract"]["warmup_tokens"] == 10_000_000
    assert readiness["fork_policy"]["forbidden_warm_starts"] == [
        "runs/base-dense-v4-16m-smoke",
        "runs/base-dense-v4-13m-low-lr-calibration",
    ]
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
    assert "science_arxiv_open_permissive" in (evaluation["required_additional_validation_sources"])
    assert evaluation["hard_stop"]["scale_relative_l2_gt"] == pytest.approx(0.05)
    assert readiness["blockers"]


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
