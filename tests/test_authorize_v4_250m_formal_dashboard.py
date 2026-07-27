from __future__ import annotations

import hashlib
import importlib.util
import json
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/authorize_v4_250m_formal_dashboard.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "authorize_v4_250m_formal_dashboard_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


admission = _load_script()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _fake_dashboard_loader(path: Path) -> SimpleNamespace:
    raw = json.loads(path.read_text(encoding="utf-8"))
    profiles = []
    for value in raw["profiles"]:
        plan_id = value.get("governed_plan_id")
        launch_kind = value.get("launch_kind", "direct_train")
        start_confirmation = (
            f"RUN {plan_id}"
            if launch_kind == "governed_v4"
            else f"START {value['id']}"
        )
        profiles.append(
            SimpleNamespace(
                profile_id=value["id"],
                launch_enabled=value["launch_enabled"],
                launch_kind=launch_kind,
                governed_plan_id=plan_id,
                start_confirmation=start_confirmation,
            )
        )
    project_root = (path.parent / raw["project_root"]).resolve()
    return SimpleNamespace(
        project_root=project_root,
        profiles=tuple(profiles),
    )


def _resign_dashboard_bundle(output: Path) -> None:
    dashboard_path = output / admission.DASHBOARD_NAME
    manifest_path = output / admission.MANIFEST_NAME
    complete_path = output / admission.COMPLETE_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dashboard_identity = manifest["files"][admission.DASHBOARD_NAME]
    dashboard_identity["size"] = dashboard_path.stat().st_size
    dashboard_identity["sha256"] = _sha(dashboard_path)
    manifest["authorized_dashboard"]["sha256"] = _sha(dashboard_path)
    manifest.pop("bundle_fingerprint")
    manifest["bundle_fingerprint"] = admission._canonical_sha256(manifest)
    _write_json(manifest_path, manifest)
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["manifest_sha256"] = _sha(manifest_path)
    complete["bundle_fingerprint"] = manifest["bundle_fingerprint"]
    _write_json(complete_path, complete)


def _fixture(
    tmp_path: Path,
) -> tuple[Namespace, dict[str, Any], Path, Path]:
    project = tmp_path / "project"
    output_parent = project / "locks"
    output_parent.mkdir(parents=True)
    output = output_parent / "formal-dashboard-admission"
    run_dir = project / admission.EXPECTED_RUN_DIR
    state_path = project / admission.EXPECTED_STATE_PATH
    plan_id = "a" * 64
    release_fingerprint = "b" * 64
    bundle_fingerprint = "c" * 64
    readiness_fingerprint = "d" * 64
    required_run_ack = f"RUN {plan_id}"
    config = project / "release" / admission.FINAL_CONFIG_NAME
    readiness = project / "release" / admission.FINAL_READINESS_NAME
    controller = project / "scripts/govern_v4_training.py"
    config.parent.mkdir(parents=True)
    controller.parent.mkdir(parents=True)
    config.write_text("run_id: base-dense-v4-250m-pilot\n", encoding="utf-8")
    readiness.write_text('{"authorizes_training":true}\n', encoding="utf-8")
    controller.write_text("# governed fixture\n", encoding="utf-8")
    current_dashboard_path = project / "current-admission/dashboard.json"
    current_dashboard = {
        "schema_version": 1,
        "project_root": "../..",
        "state_dir": ".twen/dashboard",
        "profiles": [
            {
                "id": admission.CALIBRATION_PROFILE_ID,
                "label": "calibration admission",
                "config": "configs/calibration.yaml",
                "config_sha256": "1" * 64,
                "fork_from": "runs/v3/final",
                "launch_enabled": True,
                "resume": "none",
                "calibration_admission": {
                    "admission_fingerprint": "2" * 64,
                },
            },
            {
                "id": "base-dense-v3-500m",
                "label": "monitor only",
                "config": "configs/v3.yaml",
                "config_sha256": "3" * 64,
                "fork_from": None,
                "launch_enabled": False,
                "resume": "none",
            },
        ],
    }
    _write_json(current_dashboard_path, current_dashboard)
    profile_contract = {
        "id": admission.PROFILE_ID,
        "label": admission.AUTHORIZED_LABEL,
        "config": f"release/{admission.FINAL_CONFIG_NAME}",
        "config_sha256": _sha(config),
        "fork_from": None,
        "launch_enabled": True,
        "launch_kind": "governed_v4",
        "resume": "none",
        "governed_controller": "scripts/govern_v4_training.py",
        "governed_controller_sha256": _sha(controller),
        "governed_readiness": f"release/{admission.FINAL_READINESS_NAME}",
        "governed_readiness_sha256": _sha(readiness),
        "governed_state": admission.EXPECTED_STATE_PATH,
        "governed_plan_id": plan_id,
        "run_dir": str(run_dir),
        "required_run_ack": required_run_ack,
    }
    authenticated_inputs = {
        "formal_release": {
            "root": str(config.parent),
            "manifest": {
                "path": str(config.parent / admission.MANIFEST_NAME),
                "sha256": "4" * 64,
            },
            "complete": {
                "path": str(config.parent / admission.COMPLETE_NAME),
                "sha256": "5" * 64,
            },
            "config": admission._identity(config),
            "readiness": admission._identity(readiness),
            "release_fingerprint": release_fingerprint,
            "bundle_fingerprint": bundle_fingerprint,
            "readiness_fingerprint": readiness_fingerprint,
            "governed_plan_id": plan_id,
            "required_run_ack": required_run_ack,
            "governed_controller": admission._identity(controller),
            "authorizes_training": True,
            "training_started": False,
        },
        "current_calibration_admission": {
            "root": str(current_dashboard_path.parent),
            "dashboard": admission._identity(current_dashboard_path),
            "admission_fingerprint": "6" * 64,
            "bundle_fingerprint": "7" * 64,
            "authorizes_formal_training": False,
            "training_started": False,
        },
        "source_identity": {
            "formal_dashboard_authorizer": admission._identity(SCRIPT),
            "calibration_admission_verifier": admission._identity(
                Path(admission.calibration.__file__)
            ),
            "governed_controller": admission._identity(controller),
            "source_tree": {
                "path": str(project / "src/twen"),
                "sha256": "8" * 64,
            },
            "dependency_lock": {
                "path": str(project / "uv.lock"),
                "sha256": "9" * 64,
            },
        },
        "authorized_profile_contract": profile_contract,
        "training_authorization": {
            "source": "formal_release_bundle",
            "release_fingerprint": release_fingerprint,
            "bundle_fingerprint": bundle_fingerprint,
            "readiness_fingerprint": readiness_fingerprint,
            "authorizes_training": True,
        },
    }
    snapshot = {
        "project_root": project,
        "dashboard": current_dashboard,
        "profile_id": admission.PROFILE_ID,
        "run_dir": run_dir,
        "state_path": state_path,
        "required_run_ack": required_run_ack,
        "authenticated_inputs": authenticated_inputs,
        "input_fingerprint": admission._canonical_sha256(
            authenticated_inputs
        ),
    }
    args = Namespace(
        action="publish",
        release=config.parent,
        calibration_admission=current_dashboard_path.parent,
        profile_id=admission.PROFILE_ID,
        run_ack=required_run_ack,
        output=output,
    )
    return args, snapshot, current_dashboard_path, output


def test_exact_run_acknowledgement_is_byte_for_byte() -> None:
    expected = "RUN " + "a" * 64
    admission._require_exact_run_ack(expected, expected)
    for observed in (None, expected + " ", expected.lower()):
        with pytest.raises(
            admission.FormalDashboardAdmissionError,
            match="must equal",
        ):
            admission._require_exact_run_ack(observed, expected)


def test_json_reader_and_bundle_inventory_reject_symlinks(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    _write_json(target, {"schema_version": 1})
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(
        admission.FormalDashboardAdmissionError,
        match="symlink",
    ):
        admission._read_json(link, label="fixture")

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "payload").symlink_to(target)
    with pytest.raises(
        admission.FormalDashboardAdmissionError,
        match="symlink",
    ):
        admission._strict_bundle_files(
            bundle,
            expected_files={"payload"},
            label="fixture bundle",
        )


def test_plan_rejects_non_release_training_authorization(
    tmp_path: Path,
) -> None:
    _args, snapshot, _dashboard, output = _fixture(tmp_path)
    snapshot["authenticated_inputs"]["training_authorization"][
        "source"
    ] = "calibration_admission"

    with pytest.raises(
        admission.FormalDashboardAdmissionError,
        match="solely from the release",
    ):
        admission._build_plan(snapshot, output=output)


def test_publish_is_atomic_and_opens_only_the_governed_formal_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, snapshot, current_dashboard_path, output = _fixture(tmp_path)
    current_before = current_dashboard_path.read_bytes()
    monkeypatch.setattr(
        admission,
        "_authenticate_inputs",
        lambda _args: snapshot,
    )
    monkeypatch.setattr(
        admission,
        "load_dashboard_settings",
        _fake_dashboard_loader,
    )

    result = admission.publish_admission(args)

    assert result["publication_performed"] is True
    assert result["required_run_ack"] == f"RUN {'a' * 64}"
    assert result["governed_plan_id"] == "a" * 64
    assert result["authorizes_training"] is True
    assert result["training_authorization_source"] == "formal_release_bundle"
    assert result["training_started"] is False
    assert current_dashboard_path.read_bytes() == current_before
    assert {path.name for path in output.iterdir()} == {
        admission.ADMISSION_NAME,
        admission.DASHBOARD_NAME,
        admission.MANIFEST_NAME,
        admission.COMPLETE_NAME,
    }
    recorded = json.loads(
        (output / admission.ADMISSION_NAME).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (output / admission.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    complete = json.loads(
        (output / admission.COMPLETE_NAME).read_text(encoding="utf-8")
    )
    dashboard = json.loads(
        (output / admission.DASHBOARD_NAME).read_text(encoding="utf-8")
    )
    assert recorded["acknowledgement"] == f"RUN {'a' * 64}"
    assert recorded["training_authorization_source"] == "formal_release_bundle"
    assert recorded["training_started"] is False
    assert manifest["authorizes_training"] is True
    assert manifest["training_authorization_source"] == "formal_release_bundle"
    assert manifest["training_started"] is False
    assert complete["manifest_sha256"] == _sha(
        output / admission.MANIFEST_NAME
    )
    assert complete["training_started"] is False
    launchable = [
        profile
        for profile in dashboard["profiles"]
        if profile["launch_enabled"]
    ]
    assert [profile["id"] for profile in launchable] == [
        admission.PROFILE_ID
    ]
    formal = launchable[0]
    assert formal["launch_kind"] == "governed_v4"
    assert formal["governed_plan_id"] == "a" * 64
    inline = formal["formal_dashboard_admission"]
    assert inline["required_run_ack"] == f"RUN {'a' * 64}"
    assert inline["training_authorization_source"] == "formal_release_bundle"
    assert inline["training_started"] is False
    calibration_profile = next(
        profile
        for profile in dashboard["profiles"]
        if profile["id"] == admission.CALIBRATION_PROFILE_ID
    )
    assert calibration_profile["launch_enabled"] is False
    assert not snapshot["run_dir"].exists()
    assert not snapshot["state_path"].exists()

    verified = admission.verify_admission(args)
    assert verified["verified"] is True
    assert verified["dashboard_config"] == str(
        output / admission.DASHBOARD_NAME
    )


def test_publish_requires_exact_run_ack_without_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, snapshot, _current_dashboard, output = _fixture(tmp_path)
    args.run_ack = f"RUN {'f' * 64}"
    monkeypatch.setattr(
        admission,
        "_authenticate_inputs",
        lambda _args: snapshot,
    )

    with pytest.raises(
        admission.FormalDashboardAdmissionError,
        match="must equal",
    ):
        admission.publish_admission(args)

    assert not output.exists()


def test_publish_rejects_existing_output_and_dangling_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, snapshot, _current_dashboard, output = _fixture(tmp_path)
    monkeypatch.setattr(
        admission,
        "_authenticate_inputs",
        lambda _args: snapshot,
    )
    output.mkdir()
    with pytest.raises(
        admission.FormalDashboardAdmissionError,
        match="already exists",
    ):
        admission.publish_admission(args)
    output.rmdir()
    output.symlink_to(output.parent / "missing")
    with pytest.raises(
        admission.FormalDashboardAdmissionError,
        match="symlink",
    ):
        admission.publish_admission(args)
    assert output.is_symlink()


def test_atomic_noreplace_rejects_destination_that_appeared(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()

    with pytest.raises(
        admission.FormalDashboardAdmissionError,
        match="appeared during publication",
    ):
        admission._rename_directory_noreplace(source, destination)

    assert source.is_dir()
    assert destination.is_dir()


def test_publish_detects_input_toctou_and_leaves_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, snapshot, _current_dashboard, output = _fixture(tmp_path)
    changed = deepcopy(snapshot)
    changed["input_fingerprint"] = "f" * 64
    snapshots = iter((snapshot, changed))
    monkeypatch.setattr(
        admission,
        "_authenticate_inputs",
        lambda _args: next(snapshots),
    )
    monkeypatch.setattr(
        admission,
        "load_dashboard_settings",
        _fake_dashboard_loader,
    )

    with pytest.raises(
        admission.FormalDashboardAdmissionError,
        match="inputs changed during publication",
    ):
        admission.publish_admission(args)

    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.incomplete-*"))


def test_verify_rejects_resigned_dashboard_derivation_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, snapshot, _current_dashboard, output = _fixture(tmp_path)
    monkeypatch.setattr(
        admission,
        "_authenticate_inputs",
        lambda _args: snapshot,
    )
    monkeypatch.setattr(
        admission,
        "load_dashboard_settings",
        _fake_dashboard_loader,
    )
    admission.publish_admission(args)
    dashboard_path = output / admission.DASHBOARD_NAME
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    dashboard["state_dir"] = "tampered-state"
    _write_json(dashboard_path, dashboard)
    _resign_dashboard_bundle(output)

    with pytest.raises(
        admission.FormalDashboardAdmissionError,
        match="not derived exactly",
    ):
        admission.verify_admission(args)


def test_publish_rejects_preexisting_formal_runtime_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, snapshot, _current_dashboard, output = _fixture(tmp_path)
    snapshot["state_path"].parent.mkdir(parents=True)
    snapshot["state_path"].write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        admission,
        "_authenticate_inputs",
        lambda _args: snapshot,
    )

    with pytest.raises(
        admission.FormalDashboardAdmissionError,
        match="controller state already exists",
    ):
        admission.publish_admission(args)

    assert not output.exists()


def test_verify_remains_available_after_formal_run_directory_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, snapshot, _current_dashboard, _output = _fixture(tmp_path)
    monkeypatch.setattr(
        admission,
        "_authenticate_inputs",
        lambda _args: snapshot,
    )
    monkeypatch.setattr(
        admission,
        "load_dashboard_settings",
        _fake_dashboard_loader,
    )
    admission.publish_admission(args)
    snapshot["run_dir"].mkdir(parents=True)
    snapshot["state_path"].parent.mkdir(parents=True, exist_ok=True)
    snapshot["state_path"].write_text("{}\n", encoding="utf-8")

    verified = admission.verify_admission(args)

    assert verified["verified"] is True
    assert verified["training_started"] is False
