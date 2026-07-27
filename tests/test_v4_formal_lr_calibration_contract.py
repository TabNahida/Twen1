from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from twen.config import load_train_config

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = ROOT / "configs/base/dense-v4-13m-low-lr-calibration.yaml"
CALIBRATION_CONFIG = ROOT / "configs/base/dense-v4-13m-formal-lr-calibration.yaml"
DASHBOARD = ROOT / "configs/web/dashboard.json"
SOURCE_READINESS = ROOT / "locks/base-dense-v4-250m-pilot.readiness.json"
CALIBRATION_READINESS = (
    ROOT / "locks/base-dense-v4-250m-pilot-formal-lr-calibration.readiness.json"
)

SOURCE_RUN_ID = "base-dense-v4-13m-low-lr-calibration"
CALIBRATION_RUN_ID = "base-dense-v4-13m-formal-lr-calibration"
V3_FINAL = "runs/base-dense-v3-500m/step-000000001912-milestone-complete"
SOURCE_CONFIG_SHA256 = "15ce9dbf68643b6abbcbc687a698f2994e2200587c442974903b07613a43109d"
CALIBRATION_CONFIG_SHA256 = (
    "13fdaea6c21b2e070246a6836be33718137ec1df7563232ba2d7a15dbf7536e9"
)
SOURCE_READINESS_SHA256 = (
    "7823a9695e8a20ee2cfadf73160dc8a1fcd954187c63929aca5d542c6df9946d"
)
CALIBRATION_READINESS_SHA256 = (
    "c2e16630f88fa649fcf93336b3610602f88693123dca464f66f9ee57868a002c"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_formal_lr_calibration_is_an_exact_minimal_config_derivation() -> None:
    source = _yaml(SOURCE_CONFIG)
    candidate = _yaml(CALIBRATION_CONFIG)
    expected = copy.deepcopy(source)
    expected["checkpoint"]["output_dir"] = f"runs/{CALIBRATION_RUN_ID}"
    expected["optimizer"]["adapter_lr"] = 3e-5
    expected["optimizer"]["lora_lr"] = 3e-5
    expected["optimizer"]["scale_lr"] = 3e-6
    expected["optimizer"]["warmup_tokens"] = 10_000_000
    expected["run_id"] = CALIBRATION_RUN_ID

    assert candidate == expected
    assert _sha256(SOURCE_CONFIG) == SOURCE_CONFIG_SHA256
    assert _sha256(CALIBRATION_CONFIG) == CALIBRATION_CONFIG_SHA256

    loaded = load_train_config(CALIBRATION_CONFIG)
    assert loaded.run_id == CALIBRATION_RUN_ID
    assert loaded.checkpoint.output_dir == f"runs/{CALIBRATION_RUN_ID}"
    assert loaded.optimizer.max_tokens == 13_000_000
    assert loaded.optimizer.adapter_lr == pytest.approx(3e-5)
    assert loaded.optimizer.lora_lr == pytest.approx(3e-5)
    assert loaded.optimizer.scale_lr == pytest.approx(3e-6)
    assert loaded.optimizer.warmup_tokens == 10_000_000
    assert loaded.checkpoint.every_steps == 10


def test_formal_lr_calibration_dashboard_profile_is_unique_and_closed() -> None:
    dashboard = _json(DASHBOARD)
    profiles = dashboard["profiles"]
    assert isinstance(profiles, list)
    matches = [
        profile
        for profile in profiles
        if isinstance(profile, dict) and profile.get("id") == CALIBRATION_RUN_ID
    ]
    assert matches == [
        {
            "config": "configs/base/dense-v4-13m-formal-lr-calibration.yaml",
            "config_sha256": CALIBRATION_CONFIG_SHA256,
            "fork_from": V3_FINAL,
            "id": CALIBRATION_RUN_ID,
            "label": "Base Dense v4 13M formal-LR calibration (disabled template)",
            "launch_enabled": False,
            "launch_kind": "direct_train",
            "resume": "none",
        }
    ]
    assert all(
        isinstance(profile, dict) and profile.get("launch_enabled") is False
        for profile in profiles
    )


def test_formal_lr_readiness_is_minimal_fail_closed_derivation() -> None:
    source = _json(SOURCE_READINESS)
    candidate = _json(CALIBRATION_READINESS)
    expected = copy.deepcopy(source)
    expected["calibration_gate"]["config"] = {
        "path": "configs/base/dense-v4-13m-formal-lr-calibration.yaml",
        "sha256": CALIBRATION_CONFIG_SHA256,
    }
    expected["fork_policy"]["forbidden_warm_starts"].append(
        f"runs/{CALIBRATION_RUN_ID}"
    )

    assert candidate == expected
    assert _sha256(SOURCE_READINESS) == SOURCE_READINESS_SHA256
    assert _sha256(CALIBRATION_READINESS) == CALIBRATION_READINESS_SHA256
    assert candidate["launch_enabled"] is False
    assert candidate["authorizes_training"] is False
    assert candidate["training_started"] is False
    assert candidate["launch_command_after_all_gates_pass"] is None
    assert candidate["calibration_gate"]["passed"] is False
    assert candidate["calibration_gate"]["authorizes_training"] is False


def test_calibration_run_is_isolated_and_matches_formal_lr_contract() -> None:
    source = _yaml(SOURCE_CONFIG)
    candidate = _yaml(CALIBRATION_CONFIG)
    readiness = _json(CALIBRATION_READINESS)
    formal = readiness["contract"]

    assert candidate["run_id"] != source["run_id"]
    assert candidate["checkpoint"]["output_dir"] != source["checkpoint"]["output_dir"]
    assert readiness["fork_from"] == V3_FINAL
    assert readiness["calibration_gate"]["required_fork_checkpoint"]["path"] == V3_FINAL
    assert readiness["calibration_gate"]["required_fork_checkpoint"][
        "reset_optimizer_and_scheduler"
    ] is True
    assert readiness["fork_policy"]["required_model_only_checkpoint"] == V3_FINAL
    assert readiness["fork_policy"]["reset_optimizer_and_scheduler"] is True
    assert readiness["fork_policy"]["forbidden_warm_starts"] == [
        "runs/base-dense-v4-16m-smoke",
        f"runs/{SOURCE_RUN_ID}",
        f"runs/{CALIBRATION_RUN_ID}",
    ]

    optimizer = candidate["optimizer"]
    for name in ("adapter_lr", "lora_lr", "scale_lr", "warmup_tokens"):
        assert optimizer[name] == pytest.approx(formal[name])
    assert optimizer["lr_schedule"] == formal["lr_schedule"] == "cosine"
    assert optimizer["min_lr_ratio"] == pytest.approx(formal["min_lr_ratio"])
    assert candidate["data"]["global_batch_tokens"] == formal["global_batch_tokens"]
    assert candidate["data"]["micro_batch_size"] == formal["micro_batch_size"]
    assert candidate["data"]["max_sequence_length"] == formal["sequence_length"]
    assert candidate["losses"]["ntp"] == pytest.approx(formal["ntp_weight"])
    assert candidate["losses"]["mtp"] == pytest.approx(formal["mtp_weight"])
    assert candidate["losses"]["teacher_kd"] == pytest.approx(
        formal["teacher_kd_weight"]
    )
    assert readiness["calibration_gate"]["required_candidate_checkpoints"][
        "global_steps"
    ] == [40, 50]
