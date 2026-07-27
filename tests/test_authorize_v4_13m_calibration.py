from __future__ import annotations

import hashlib
import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/authorize_v4_13m_calibration.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_test_authorize_v4_13m_calibration",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


admission = _load_script()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(
    tmp_path: Path,
) -> tuple[Namespace, dict[str, Any], Path, Path]:
    project = tmp_path / "project"
    config = project / "configs/base/dense-v4-13m-low-lr-calibration.yaml"
    config.parent.mkdir(parents=True)
    config.write_bytes((ROOT / "configs/base/dense-v4-13m-low-lr-calibration.yaml").read_bytes())
    monitor_config = project / "configs/base/dense-v3-500m.yaml"
    monitor_config.write_bytes((ROOT / "configs/base/dense-v3-500m.yaml").read_bytes())
    dashboard = project / "configs/web/dashboard.json"
    dashboard_value = {
        "schema_version": 1,
        "project_root": "../..",
        "state_dir": ".twen/dashboard",
        "profiles": [
            {
                "id": admission.PROFILE_ID,
                "label": "blocked",
                "config": ("configs/base/dense-v4-13m-low-lr-calibration.yaml"),
                "config_sha256": _sha(config),
                "fork_from": ("runs/base-dense-v3-500m/step-000000001912-milestone-complete"),
                "launch_enabled": False,
                "resume": "none",
            },
            {
                "id": "base-dense-v3-500m",
                "label": "monitor only",
                "config": "configs/base/dense-v3-500m.yaml",
                "config_sha256": _sha(monitor_config),
                "fork_from": None,
                "launch_enabled": False,
                "resume": "none",
            },
        ],
    }
    _write_json(dashboard, dashboard_value)
    output_parent = project / "artifacts/evidence"
    output_parent.mkdir(parents=True)
    output = output_parent / "calibration-admission"
    expected_ack = "ACCEPT V4 WIKIPEDIA LICENSE " + "f" * 64
    profile_contract = {
        "id": admission.PROFILE_ID,
        "label": admission.AUTHORIZED_LABEL,
        "config": ("configs/base/dense-v4-13m-low-lr-calibration.yaml"),
        "config_sha256": _sha(config),
        "fork_from": ("runs/base-dense-v3-500m/step-000000001912-milestone-complete"),
        "resume": "none",
        "launch_kind": "direct_train",
        "launch_enabled": True,
        "run_dir": str(project / "runs/base-dense-v4-13m-low-lr-calibration"),
        "required_start_confirmation": ("START base-dense-v4-13m-low-lr-calibration"),
    }
    authenticated_inputs = {
        "formal_closure": {
            "path": str(project / "closure"),
            "manifest_sha256": "1" * 64,
            "complete_sha256": "2" * 64,
            "bundle_fingerprint": "3" * 64,
        },
        "licence_contract": {
            "contract": {"source_id": "chinese_wikipedia_zh_20231101"},
            "contract_fingerprint": "f" * 64,
            "required_acknowledgement": expected_ack,
        },
        "calibration_config": admission._identity(config),
        "formal_primary_prepared_manifest": {
            "path": str(project / "prepared/manifest.json"),
            "size": 1,
            "sha256": "4" * 64,
            "dataset_fingerprint": "5" * 64,
            "source_map_sha256": "6" * 64,
        },
        "fork_checkpoint": {
            "path": str(project / "fork"),
            "manifest_sha256": "7" * 64,
            "complete_sha256": "8" * 64,
            "metadata": {"global_step": 1912},
        },
        "dashboard_before": admission._identity(dashboard),
        "source_identity": {
            "authorizer": admission._identity(SCRIPT),
            "release_verifier": admission._identity(Path(admission.publisher.__file__)),
            "twen_source_tree_sha256": "9" * 64,
        },
        "authorized_profile_contract": profile_contract,
    }
    snapshot = {
        "project_root": project,
        "dashboard_path": dashboard,
        "dashboard": dashboard_value,
        "profile_id": admission.PROFILE_ID,
        "expected_acknowledgement": expected_ack,
        "authenticated_inputs": authenticated_inputs,
        "input_fingerprint": admission._canonical_sha256(authenticated_inputs),
    }
    args = Namespace(
        action="publish",
        closure=project / "closure",
        dashboard_config=dashboard,
        profile_id=admission.PROFILE_ID,
        acknowledgement=expected_ack,
        output=output,
    )
    return args, snapshot, dashboard, output


def test_exact_acknowledgement_is_byte_for_byte() -> None:
    expected = "ACCEPT V4 WIKIPEDIA LICENSE " + "f" * 64
    admission._require_exact_acknowledgement(expected, expected)
    with pytest.raises(
        admission.CalibrationAdmissionError,
        match="must equal",
    ):
        admission._require_exact_acknowledgement(expected + " ", expected)


def test_json_reader_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    _write_json(target, {"schema_version": 1})
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(
        admission.CalibrationAdmissionError,
        match="symlink",
    ):
        admission._read_json(link, label="fixture")


def test_publisher_accepts_all_expected_pending_config_differences() -> None:
    expected = {
        "authenticated config retains missing or PENDING data identities",
        ("config phase-disjointness identity differs from closed capacity evidence"),
        "cooldown config identities differ from closed capacity evidence",
        "primary config identities differ from closed capacity evidence",
    }
    assert expected == admission.publisher.PENDING_FORMAL_CONFIG_ISSUES


def test_publish_writes_immutable_bundle_then_enables_only_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, snapshot, dashboard, output = _fixture(tmp_path)
    dashboard_before = dashboard.read_bytes()
    monkeypatch.setattr(
        admission,
        "_authenticate_inputs",
        lambda _args: snapshot,
    )

    result = admission.publish_admission(args)

    assert result["publication_performed"] is True
    assert result["authorizes_calibration_launch"] is True
    assert result["authorizes_formal_training"] is False
    assert result["training_started"] is False
    assert result["dashboard_profile_changed"] is False
    assert result["required_start_confirmation"] == ("START base-dense-v4-13m-low-lr-calibration")
    assert {path.name for path in output.iterdir()} == {
        admission.ADMISSION_NAME,
        admission.DASHBOARD_NAME,
        admission.MANIFEST_NAME,
        admission.COMPLETE_NAME,
    }
    manifest = json.loads((output / admission.MANIFEST_NAME).read_text(encoding="utf-8"))
    complete = json.loads((output / admission.COMPLETE_NAME).read_text(encoding="utf-8"))
    recorded = json.loads((output / admission.ADMISSION_NAME).read_text(encoding="utf-8"))
    assert complete["manifest_sha256"] == _sha(output / admission.MANIFEST_NAME)
    assert complete["bundle_fingerprint"] == manifest["bundle_fingerprint"]
    assert recorded["acknowledgement"] == args.acknowledgement
    assert recorded["status"] == "accepted_explicit_user_acknowledgement"
    assert recorded["authorizes_formal_training"] is False
    assert recorded["training_started"] is False
    assert manifest["files"][admission.DASHBOARD_NAME]["sha256"] == _sha(
        output / admission.DASHBOARD_NAME
    )

    assert dashboard.read_bytes() == dashboard_before
    dashboard_value = json.loads((output / admission.DASHBOARD_NAME).read_text(encoding="utf-8"))
    launchable = [profile for profile in dashboard_value["profiles"] if profile["launch_enabled"]]
    assert [profile["id"] for profile in launchable] == [admission.PROFILE_ID]
    selected = launchable[0]
    assert selected["label"] == admission.AUTHORIZED_LABEL
    assert selected["calibration_admission"]["path"] == ("artifacts/evidence/calibration-admission")
    assert selected["calibration_admission"]["admission_sha256"] == _sha(
        output / admission.ADMISSION_NAME
    )
    assert selected["calibration_admission"]["required_start_confirmation"] == (
        "START base-dense-v4-13m-low-lr-calibration"
    )
    assert not (tmp_path / "project/runs/base-dense-v4-13m-low-lr-calibration").exists()

    verified = admission.verify_admission(args)
    assert verified["verified"] is True
    assert verified["dashboard_config"] == str(output / admission.DASHBOARD_NAME)


def test_publish_rejects_an_existing_calibration_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, snapshot, dashboard, _output = _fixture(tmp_path)
    run_dir = Path(
        snapshot["authenticated_inputs"]["authorized_profile_contract"]["run_dir"]
    )
    run_dir.mkdir(parents=True)
    before = dashboard.read_bytes()
    monkeypatch.setattr(
        admission,
        "_authenticate_inputs",
        lambda _args: snapshot,
    )

    with pytest.raises(
        admission.CalibrationAdmissionError,
        match="calibration run directory already exists",
    ):
        admission.publish_admission(args)

    assert dashboard.read_bytes() == before


def test_verify_remains_available_after_calibration_run_directory_is_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, snapshot, _dashboard, _output = _fixture(tmp_path)
    monkeypatch.setattr(
        admission,
        "_authenticate_inputs",
        lambda _args: snapshot,
    )
    admission.publish_admission(args)
    run_dir = Path(
        snapshot["authenticated_inputs"]["authorized_profile_contract"]["run_dir"]
    )
    run_dir.mkdir(parents=True)

    verified = admission.verify_admission(args)

    assert verified["verified"] is True
    assert verified["training_started"] is False


def test_existing_output_is_fail_closed_and_does_not_enable_dashboard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, snapshot, dashboard, output = _fixture(tmp_path)
    output.mkdir()
    before = dashboard.read_bytes()
    monkeypatch.setattr(
        admission,
        "_authenticate_inputs",
        lambda _args: snapshot,
    )

    with pytest.raises(
        admission.CalibrationAdmissionError,
        match="already exists",
    ):
        admission.publish_admission(args)

    assert dashboard.read_bytes() == before
    assert list(output.iterdir()) == []


def test_dangling_output_symlink_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, snapshot, dashboard, output = _fixture(tmp_path)
    output.symlink_to(output.parent / "missing")
    before = dashboard.read_bytes()
    monkeypatch.setattr(
        admission,
        "_authenticate_inputs",
        lambda _args: snapshot,
    )

    with pytest.raises(
        admission.CalibrationAdmissionError,
        match="symlink",
    ):
        admission.publish_admission(args)

    assert dashboard.read_bytes() == before
    assert output.is_symlink()


def test_verify_rejects_mutated_authorized_dashboard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, snapshot, _dashboard, output = _fixture(tmp_path)
    monkeypatch.setattr(
        admission,
        "_authenticate_inputs",
        lambda _args: snapshot,
    )
    admission.publish_admission(args)
    authorized_dashboard = output / admission.DASHBOARD_NAME
    authorized_dashboard.write_text(
        authorized_dashboard.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )

    with pytest.raises(
        admission.CalibrationAdmissionError,
        match="file identity differs",
    ):
        admission.verify_admission(args)
