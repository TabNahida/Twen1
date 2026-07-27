from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from twen import governed

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_CONFIG = ROOT / "configs/base/dense-v4-250m-pilot.blocked.yaml"
READINESS_TEMPLATE = ROOT / "locks/base-dense-v4-250m-pilot.readiness.json"
CALIBRATION_CONFIG = ROOT / "configs/base/dense-v4-13m-low-lr-calibration.yaml"


def _load_script() -> ModuleType:
    path = ROOT / "scripts/publish_v4_250m_release.py"
    spec = importlib.util.spec_from_file_location(
        "publish_v4_250m_release",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = _load_script()
REAL_RECOMPUTE_CALIBRATION_CLAIMS = release._recompute_calibration_claims


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": _sha(path),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _runtime_checkpoint(
    path: Path,
    *,
    step: int = 50,
    committed_tokens: int = 13_107_200,
) -> dict[str, Any]:
    metadata = {
        "checkpoint_schema_version": 2,
        "checkpoint_id": path.name,
        "saved_world_size": 1,
        "global_step": step,
        "committed_tokens": committed_tokens,
        "kind": "milestone",
        "tag": "complete",
        "run_id": "calibration-fixture",
        "stage": "dense-oracle",
        "trainer_state": {"committed_tokens": committed_tokens},
        "data_cursor": {"global_token_index": committed_tokens},
        "backend": "dcp",
        "critical_fingerprint": "1" * 64,
        "data_fingerprint": "2" * 64,
    }
    _write_json(path / "metadata.json", metadata)
    runtime = path / "runtime/rank-00000.pkl"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_bytes(b"authenticated runtime fixture\n")
    state = path / "state/fixture.distcp"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_bytes(b"authenticated state fixture\n")
    manifest = {
        "algorithm": "sha256",
        "version": 1,
        "files": {
            str(item.relative_to(path)): _sha(item)
            for item in (path / "metadata.json", runtime, state)
        },
    }
    _write_json(path / "manifest.json", manifest)
    (path / "COMPLETE").write_text(
        _sha(path / "manifest.json") + "\n",
        encoding="ascii",
    )
    return governed.authenticate_checkpoint(path)


def _pending_license_gate() -> dict[str, Any]:
    acknowledgement = (
        "ACCEPT V4 WIKIPEDIA LICENSE "
        f"{governed.FORMAL_V4_WIKIPEDIA_LICENSE_FINGERPRINT}"
    )
    return {
        "required": True,
        "status": "pending_explicit_user_acceptance",
        "contract": copy.deepcopy(governed.FORMAL_V4_WIKIPEDIA_LICENSE_CONTRACT),
        "contract_fingerprint": (
            governed.FORMAL_V4_WIKIPEDIA_LICENSE_FINGERPRINT
        ),
        "required_acknowledgement": acknowledgement,
        "observed_acknowledgement": None,
        "passed": False,
        "authorizes_training": False,
    }


def _fake_readiness() -> dict[str, Any]:
    readiness = _json(READINESS_TEMPLATE)
    readiness.update(
        {
            "project_root": str(ROOT.resolve()),
            "config_path": str(BLOCKED_CONFIG.resolve()),
            "config_sha256": _sha(BLOCKED_CONFIG),
            "status": "blocked_for_release_test",
            "blockers": [
                "blocked config identities have not been released",
                "calibration has not been authorized",
            ],
            "launch_enabled": False,
            "authorizes_training": False,
            "training_started": False,
            "wikipedia_license_gate": _pending_license_gate(),
            "chinese_semantic_quality_gate": {
                "required": True,
                "status": "passed_authenticated_chinese_semantic_quality_audit",
                "source_id": governed.FORMAL_V4_CHINESE_SEMANTIC_SOURCE_ID,
                "required_bundle": copy.deepcopy(
                    governed.FORMAL_V4_CHINESE_SEMANTIC_REQUIRED_BUNDLE
                ),
                "required_gates": copy.deepcopy(
                    governed.FORMAL_V4_CHINESE_SEMANTIC_REQUIRED_GATES
                ),
                "observed": {"fixture": True},
                "passed": True,
                "authorizes_training": False,
            },
        }
    )
    formal = readiness["formal_validation_gate"]
    formal.update(
        {
            "status": (
                "passed_authenticated_governed_disjointness_and_v3_baseline"
            ),
            "passed": True,
            "authorizes_training": False,
        }
    )
    calibration = readiness["calibration_gate"]
    calibration.update(
        {
            "config": {
                "path": str(CALIBRATION_CONFIG.resolve()),
                "sha256": _sha(CALIBRATION_CONFIG),
            },
            "observed": None,
            "passed": False,
            "authorizes_training": False,
        }
    )
    controller = ROOT / "scripts/govern_v4_training.py"
    readiness["governed_controller"] = {
        "path": str(controller.resolve()),
        "sha256": _sha(controller),
        "twen_source_tree_sha256": governed.twen_source_tree_sha256(
            ROOT / "src/twen"
        ),
        "implemented": True,
    }
    return readiness


def _fake_snapshot(tmp_path: Path) -> dict[str, Any]:
    primary = tmp_path / "evidence/primary/manifest.json"
    cooldown = tmp_path / "evidence/cooldown/manifest.json"
    phase = tmp_path / "evidence/phase/attestation.json"
    for path in (primary, cooldown, phase):
        _write_json(path, {"fixture": f"{path.parent.name}/{path.name}"})
    primary_binding = {
        "prepared_manifest_path": str(primary.resolve()),
        "prepared_manifest_sha256": _sha(primary),
        "prepared_dataset_fingerprint": "1" * 64,
        "source_map_sha256": "2" * 64,
    }
    cooldown_binding = {
        "prepared_manifest_path": str(cooldown.resolve()),
        "prepared_manifest_sha256": _sha(cooldown),
        "prepared_dataset_fingerprint": "3" * 64,
        "source_map_sha256": "4" * 64,
    }
    phase_binding = {
        "path": str(phase.resolve()),
        "sha256": _sha(phase),
        "attestation_fingerprint": "5" * 64,
    }
    readiness = _fake_readiness()
    capacity_path = tmp_path / "closure/capacity-attestation.json"
    readiness_path = tmp_path / "closure/readiness.json"
    _write_json(capacity_path, {"fixture": "capacity"})
    readiness["capacity_attestation"] = {
        **_file_identity(capacity_path),
        "attestation_fingerprint": "6" * 64,
        "passed": True,
        "authorizes_training": False,
    }
    _write_json(readiness_path, readiness)
    closure_manifest = tmp_path / "closure/MANIFEST.json"
    closure_complete = tmp_path / "closure/COMPLETE"
    closure_manifest_value: dict[str, Any] = {
        "schema_version": 1,
        "kind": release.CLOSURE_BUNDLE_KIND,
        "closure": {"fixture": "authenticated formal closure"},
        "inputs": {"fixture": "authenticated inputs"},
        "files": {
            path.name: {
                "path": path.name,
                "size": path.stat().st_size,
                "sha256": _sha(path),
            }
            for path in (capacity_path, readiness_path)
        },
        "launch_enabled": False,
        "authorizes_training": False,
        "training_started": False,
    }
    closure_manifest_value["bundle_fingerprint"] = (
        release._canonical_sha256(closure_manifest_value)
    )
    _write_json(closure_manifest, closure_manifest_value)
    _write_json(
        closure_complete,
        {
            "schema_version": 1,
            "kind": release.CLOSURE_COMPLETE_KIND,
            "manifest": "MANIFEST.json",
            "manifest_sha256": _sha(closure_manifest),
            "bundle_fingerprint": closure_manifest_value[
                "bundle_fingerprint"
            ],
            "launch_enabled": False,
            "authorizes_training": False,
            "training_started": False,
        },
    )
    blocked_config = yaml.safe_load(BLOCKED_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(blocked_config, dict)
    capacity_bindings = {
        "path": str(capacity_path.resolve()),
        "sha256": _sha(capacity_path),
        "training_contract": copy.deepcopy(
            governed.FORMAL_V4_ATTESTED_CONTRACT
        ),
        "phases": {
            "primary": primary_binding,
            "cooldown": cooldown_binding,
        },
        "source_mix_basis_points": {
            "primary": copy.deepcopy(governed.FORMAL_V4_PRIMARY_SOURCE_MIX),
            "cooldown": copy.deepcopy(governed.FORMAL_V4_COOLDOWN_SOURCE_MIX),
        },
        "phase_disjointness_attestation": phase_binding,
    }
    closure_binding = {
        "path": str((tmp_path / "closure").resolve()),
        "manifest_sha256": _sha(closure_manifest),
        "complete_sha256": _sha(closure_complete),
        "bundle_fingerprint": closure_manifest_value[
            "bundle_fingerprint"
        ],
    }
    evidence = {
        name: _write_report_bundle(
            tmp_path / f"calibration-{name}",
            kind=release.CALIBRATION_REPORT_KINDS[name],
            claims={},
            payload_name=(
                "summary.json"
                if name == "checkpoint_validation_bundle"
                else "analysis.json"
            ),
        )
        for name in release.REPORT_BUNDLE_NAMES
    }
    checkpoint_path = (
        tmp_path / "step-000000000050-milestone-complete"
    )
    authenticated_checkpoint = _runtime_checkpoint(
        checkpoint_path,
    )
    checkpoint_metadata = authenticated_checkpoint["metadata"]
    public_checkpoint = {
        "path": authenticated_checkpoint["path"],
        "manifest_sha256": authenticated_checkpoint["manifest_sha256"],
        "complete_sha256": authenticated_checkpoint["complete_sha256"],
        "global_step": checkpoint_metadata["global_step"],
        "committed_tokens": checkpoint_metadata["committed_tokens"],
        "kind": checkpoint_metadata["kind"],
        "tag": checkpoint_metadata["tag"],
    }
    evaluation = {
        "passed": True,
        "observed_steps": [checkpoint_metadata["global_step"]],
        "final_global_step": checkpoint_metadata["global_step"],
        "final_committed_tokens": checkpoint_metadata[
            "committed_tokens"
        ],
        "hard_thresholds": copy.deepcopy(
            readiness["calibration_gate"]["hard_thresholds"]
        ),
    }
    calibration_attestation = tmp_path / "calibration/attestation.json"
    calibration_attestation_value: dict[str, Any] = {
        "schema_version": 1,
        "kind": release.CALIBRATION_KIND,
        "status": (
            "passed_authenticated_quality_gate_but_does_not_authorize_formal_training"
        ),
        "attestor": {
            "path": str(
                (
                    ROOT
                    / "scripts/attest_v4_13m_calibration_release.py"
                ).resolve()
            ),
            "sha256": _sha(
                ROOT / "scripts/attest_v4_13m_calibration_release.py"
            ),
        },
        "formal_closure": closure_binding,
        "calibration_gate_contract_fingerprint": (
            release._canonical_sha256(readiness["calibration_gate"])
        ),
        "calibration_config": _file_identity(CALIBRATION_CONFIG),
        "evidence": evidence,
        "claims": {},
        "candidate_checkpoints": [public_checkpoint],
        "final_checkpoint": public_checkpoint,
        "passed": True,
        "authorizes_training": False,
        "training_started": False,
    }
    calibration_attestation_value["attestation_fingerprint"] = (
        release._canonical_sha256(calibration_attestation_value)
    )
    _write_json(calibration_attestation, calibration_attestation_value)
    calibration_complete = tmp_path / "calibration/COMPLETE"
    _write_json(
        calibration_complete,
        {
            "schema_version": 1,
            "kind": release.CALIBRATION_COMPLETE_KIND,
            "attestation": calibration_attestation.name,
            "attestation_sha256": _sha(calibration_attestation),
            "attestation_fingerprint": calibration_attestation_value[
                "attestation_fingerprint"
            ],
            "passed": True,
            "authorizes_training": False,
            "training_started": False,
        },
    )
    calibration = {
        **_file_identity(calibration_attestation),
        "complete": _file_identity(calibration_complete),
        "attestation_fingerprint": calibration_attestation_value[
            "attestation_fingerprint"
        ],
        "calibration_config": _file_identity(CALIBRATION_CONFIG),
        "evidence": evidence,
        "claims": {
            "values": {"fixture": True},
            "sources": {"fixture": True},
        },
        "candidate_checkpoints": [public_checkpoint],
        "final_checkpoint": public_checkpoint,
        "evaluation": evaluation,
        "passed": True,
        "authorizes_training": False,
    }
    return {
        "closure": {
            "root": tmp_path / "closure",
            "binding": closure_binding,
            "capacity_path": capacity_path,
            "readiness_path": readiness_path,
            "readiness": readiness,
            "project_root": ROOT.resolve(),
            "blocked_config_path": BLOCKED_CONFIG,
            "blocked_config": blocked_config,
            "governed": {
                "capacity": capacity_bindings,
                "formal": {
                    "validation_phases": {
                        "primary": {"fixture": True},
                        "cooldown": {"fixture": True},
                    },
                    "formal_phase_train_disjointness": phase_binding,
                },
                "release": {
                    "wikipedia_license": {
                        "contract_fingerprint": (
                            governed.FORMAL_V4_WIKIPEDIA_LICENSE_FINGERPRINT
                        ),
                        "passed": True,
                    },
                    "chinese_semantic_quality": {
                        "passed": True,
                        "authorizes_training": True,
                    },
                },
            },
        },
        "calibration": calibration,
    }


def _args(tmp_path: Path, *, output: Path | None = None) -> Namespace:
    return Namespace(
        action="plan",
        closure=tmp_path / "unused-closure",
        calibration_attestation=tmp_path / "unused-calibration.json",
        output=output or tmp_path / "release",
        authorize_ack=None,
        wikipedia_license_ack=None,
    )


def _install_fake_authentication(
    monkeypatch: pytest.MonkeyPatch,
    snapshot: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        release,
        "_authenticate_release_inputs",
        lambda _args: copy.deepcopy(snapshot),
    )


def _structural_closure(tmp_path: Path) -> Path:
    root = tmp_path / "structural-closure"
    root.mkdir()
    inputs = {"fixture": "authenticated inputs"}
    closure_identity = {
        "schema_version": 1,
        "kind": release.CLOSURE_KIND,
        "input_fingerprint": release._canonical_sha256(inputs),
        "closure_source_sha256": _sha(
            ROOT / "scripts/close_v4_250m_formal_evidence.py"
        ),
        "twen_source_tree_sha256": governed.twen_source_tree_sha256(
            ROOT / "src/twen"
        ),
        "launch_enabled": False,
        "authorizes_training": False,
        "training_started": False,
    }
    capacity: dict[str, Any] = {
        "schema_version": 1,
        "kind": release.CAPACITY_KIND,
        "training_contract": copy.deepcopy(
            governed.FORMAL_V4_ATTESTED_CONTRACT
        ),
        "overall": {"passed": True},
        "config": {
            "path": str(BLOCKED_CONFIG.resolve()),
            "sha256": _sha(BLOCKED_CONFIG),
            "contains_pending_identity_sentinels": True,
        },
        "closure": closure_identity,
        "launch_enabled": False,
        "authorizes_training": False,
        "training_started": False,
    }
    capacity["attestation_fingerprint"] = release._canonical_sha256(capacity)
    capacity_path = root / "capacity-attestation.json"
    _write_json(capacity_path, capacity)
    readiness = _fake_readiness()
    readiness.update(
        {
            "closure": closure_identity,
            "capacity_attestation": {
                **_file_identity(capacity_path),
                "attestation_fingerprint": capacity[
                    "attestation_fingerprint"
                ],
                "passed": True,
                "authorizes_training": False,
            },
        }
    )
    readiness.pop("readiness_fingerprint", None)
    readiness["readiness_fingerprint"] = release._canonical_sha256(readiness)
    readiness_path = root / "readiness.json"
    _write_json(readiness_path, readiness)
    files = {
        path.name: {
            "path": path.name,
            "size": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in (capacity_path, readiness_path)
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": release.CLOSURE_BUNDLE_KIND,
        "closure": closure_identity,
        "inputs": inputs,
        "files": files,
        "launch_enabled": False,
        "authorizes_training": False,
        "training_started": False,
    }
    manifest["bundle_fingerprint"] = release._canonical_sha256(manifest)
    manifest_path = root / "MANIFEST.json"
    _write_json(manifest_path, manifest)
    complete = {
        "schema_version": 1,
        "kind": release.CLOSURE_COMPLETE_KIND,
        "manifest": "MANIFEST.json",
        "manifest_sha256": _sha(manifest_path),
        "bundle_fingerprint": manifest["bundle_fingerprint"],
        "launch_enabled": False,
        "authorizes_training": False,
        "training_started": False,
    }
    _write_json(root / "COMPLETE", complete)
    return root


def test_closure_structure_rejects_any_payload_tamper(
    tmp_path: Path,
) -> None:
    root = _structural_closure(tmp_path)
    snapshot = release._authenticate_closure_structure(root)
    assert snapshot["binding"]["bundle_fingerprint"] == _json(
        root / "MANIFEST.json"
    )["bundle_fingerprint"]
    capacity_path = root / "capacity-attestation.json"
    capacity_path.write_text(
        capacity_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(release.ReleaseError, match="payload identity differs"):
        release._authenticate_closure_structure(root)


def test_plan_is_read_only_and_emits_exact_canonical_acknowledgements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _fake_snapshot(tmp_path)
    _install_fake_authentication(monkeypatch, snapshot)
    output = tmp_path / "formal-release"
    args = _args(tmp_path, output=output)
    before = {
        str(path.relative_to(tmp_path))
        for path in tmp_path.rglob("*")
    }
    plan = release.build_release_plan(args)
    after = {
        str(path.relative_to(tmp_path))
        for path in tmp_path.rglob("*")
    }
    assert before == after
    assert not output.exists()
    assert (
        release._canonical_sha256(plan["release_contract"])
        == plan["release_fingerprint"]
    )
    assert plan["required_authorization"] == (
        f"AUTHORIZE V4 {plan['release_fingerprint']}"
    )
    assert plan["required_wikipedia_license_acknowledgement"] == (
        "ACCEPT V4 WIKIPEDIA LICENSE "
        f"{governed.FORMAL_V4_WIKIPEDIA_LICENSE_FINGERPRINT}"
    )
    assert plan["publication_performed"] is False
    assert plan["training_started"] is False
    assert plan["web_profile_changed"] is False


@pytest.mark.parametrize("wrong_gate", ["authorization", "license"])
def test_publish_wrong_or_missing_ack_is_zero_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wrong_gate: str,
) -> None:
    snapshot = _fake_snapshot(tmp_path)
    _install_fake_authentication(monkeypatch, snapshot)
    args = _args(tmp_path)
    plan = release.build_release_plan(args)
    args.action = "publish"
    args.authorize_ack = plan["required_authorization"]
    args.wikipedia_license_ack = plan[
        "required_wikipedia_license_acknowledgement"
    ]
    if wrong_gate == "authorization":
        args.authorize_ack += " "
    else:
        args.wikipedia_license_ack = None
    before = {
        str(path.relative_to(tmp_path))
        for path in tmp_path.rglob("*")
    }
    with pytest.raises(release.ReleaseError, match="must equal"):
        release.publish_release(args)
    after = {
        str(path.relative_to(tmp_path))
        for path in tmp_path.rglob("*")
    }
    assert before == after
    assert not args.output.exists()


def test_publish_never_creates_an_unreviewed_output_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _fake_snapshot(tmp_path)
    _install_fake_authentication(monkeypatch, snapshot)
    output = tmp_path / "missing-parent/release"
    args = _args(tmp_path, output=output)
    plan = release.build_release_plan(args)
    args.action = "publish"
    args.authorize_ack = plan["required_authorization"]
    args.wikipedia_license_ack = plan[
        "required_wikipedia_license_acknowledgement"
    ]
    with pytest.raises(release.ReleaseError, match="parent must already"):
        release.publish_release(args)
    assert not output.parent.exists()


def _publish_fake_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Namespace, dict[str, Any], dict[str, Any]]:
    snapshot = _fake_snapshot(tmp_path)
    _install_fake_authentication(monkeypatch, snapshot)
    args = _args(tmp_path)
    plan = release.build_release_plan(args)
    args.action = "publish"
    args.authorize_ack = plan["required_authorization"]
    args.wikipedia_license_ack = plan[
        "required_wikipedia_license_acknowledgement"
    ]
    result = release.publish_release(args)
    return args, plan, result


def test_publish_atomically_writes_no_pending_final_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, plan, result = _publish_fake_release(tmp_path, monkeypatch)
    assert result["publication_performed"] is True
    assert set(path.name for path in args.output.iterdir()) == {
        release.FINAL_CONFIG_NAME,
        release.FINAL_READINESS_NAME,
        release.MANIFEST_NAME,
        release.COMPLETE_NAME,
    }
    for path in args.output.iterdir():
        assert b"PENDING" not in path.read_bytes()

    config_path = args.output / release.FINAL_CONFIG_NAME
    readiness_path = args.output / release.FINAL_READINESS_NAME
    manifest_path = args.output / release.MANIFEST_NAME
    complete_path = args.output / release.COMPLETE_NAME
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    readiness = _json(readiness_path)
    manifest = _json(manifest_path)
    complete = _json(complete_path)
    assert (
        governed._normalized_formal_config_sha256(config)
        == governed.FORMAL_V4_NORMALIZED_CONFIG_SHA256
    )
    assert readiness["blockers"] == []
    assert readiness["launch_enabled"] is True
    assert readiness["authorizes_training"] is True
    assert readiness["training_started"] is False
    assert readiness["calibration_gate"]["passed"] is True
    assert readiness["calibration_gate"]["authorizes_training"] is True
    assert (
        readiness["chinese_semantic_quality_gate"]["status"]
        == "passed_authenticated_chinese_semantic_quality_audit"
    )
    assert (
        readiness["chinese_semantic_quality_gate"]["authorizes_training"]
        is True
    )
    assert readiness["formal_validation_gate"]["authorizes_training"] is True
    license_gate = readiness["wikipedia_license_gate"]
    assert (
        license_gate["status"] == "accepted_explicit_user_acknowledgement"
    )
    assert (
        license_gate["observed_acknowledgement"]
        == plan["required_wikipedia_license_acknowledgement"]
    )
    assert license_gate["passed"] is True
    assert license_gate["authorizes_training"] is True
    assert readiness["release"]["release_fingerprint"] == plan["release_fingerprint"]
    unsigned_readiness = {
        key: value
        for key, value in readiness.items()
        if key != "readiness_fingerprint"
    }
    assert (
        readiness["readiness_fingerprint"]
        == release._canonical_sha256(unsigned_readiness)
    )
    unsigned_manifest = {
        key: value
        for key, value in manifest.items()
        if key != "bundle_fingerprint"
    }
    assert manifest["bundle_fingerprint"] == release._canonical_sha256(
        unsigned_manifest
    )
    assert complete["manifest_sha256"] == _sha(manifest_path)
    assert complete["release_fingerprint"] == plan["release_fingerprint"]
    assert complete["training_started"] is False


def test_publish_rejects_existing_complete_or_partial_output_without_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _plan, _result = _publish_fake_release(tmp_path, monkeypatch)
    first_hashes = {
        path.name: _sha(path) for path in args.output.iterdir() if path.is_file()
    }
    with pytest.raises(release.ReleaseError, match="already exists"):
        release.publish_release(args)
    assert first_hashes == {
        path.name: _sha(path) for path in args.output.iterdir() if path.is_file()
    }

    partial = tmp_path / "partial-release"
    partial.mkdir()
    sentinel = partial / "user-file"
    sentinel.write_text("preserve\n", encoding="utf-8")
    partial_args = copy.copy(args)
    partial_args.output = partial
    with pytest.raises(release.ReleaseError, match="already exists"):
        release.publish_release(partial_args)
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"
    assert list(partial.iterdir()) == [sentinel]


def test_publish_tamper_during_staging_leaves_no_release_or_partial_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _fake_snapshot(tmp_path)
    calls = 0

    def authenticate(_args: Namespace) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        value = copy.deepcopy(snapshot)
        if calls == 2:
            value["calibration"]["sha256"] = "f" * 64
        return value

    monkeypatch.setattr(release, "_authenticate_release_inputs", authenticate)
    args = _args(tmp_path)
    initial_plan = release._build_release_plan_from_snapshot(
        snapshot,
        output=args.output.resolve(),
    )
    args.action = "publish"
    args.authorize_ack = initial_plan["required_authorization"]
    args.wikipedia_license_ack = initial_plan[
        "required_wikipedia_license_acknowledgement"
    ]
    with pytest.raises(release.ReleaseError, match="changed during publication"):
        release.publish_release(args)
    assert calls == 2
    assert not args.output.exists()
    assert not list(args.output.parent.glob(f".{args.output.name}.incomplete-*"))


def test_atomic_directory_install_never_replaces_even_an_empty_racing_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "release").write_text("complete\n", encoding="utf-8")
    with pytest.raises(release.ReleaseError, match="appeared during publication"):
        release._rename_directory_noreplace(source, destination)
    assert (source / "release").read_text(encoding="utf-8") == "complete\n"
    assert list(destination.iterdir()) == []


def test_atomic_directory_install_fails_closed_without_renameat2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "payload").write_text("keep\n", encoding="utf-8")
    unavailable = type("_NoRenameAt2", (), {})()
    monkeypatch.setattr(
        release.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: unavailable,
    )
    with pytest.raises(release.ReleaseError, match="refusing racy fallback"):
        release._rename_directory_noreplace(source, destination)
    assert (source / "payload").read_text(encoding="utf-8") == "keep\n"
    assert not destination.exists()


def _write_report_bundle(
    root: Path,
    *,
    kind: str,
    claims: dict[str, Any],
    payload_name: str,
) -> dict[str, Any]:
    facts = root / payload_name
    _write_json(facts, {"claims": claims})
    report_name = next(
        name
        for name, expected_kind in release.CALIBRATION_REPORT_KINDS.items()
        if kind == expected_kind
    )
    producer = ROOT / "scripts" / release.CALIBRATION_REPORT_PRODUCERS[
        report_name
    ]
    manifest = {
        "schema_version": 1,
        "kind": kind,
        "bundle_producer": _file_identity(producer),
        "files": {
            payload_name: {
                "path": payload_name,
                "size": facts.stat().st_size,
                "sha256": _sha(facts),
            }
        },
    }
    if report_name == "training_report_bundle":
        manifest.update(
            {
                "run_id": "fixture",
                "source_run_dir": str((root / "run").resolve()),
                "source_inputs": {},
                "source_terminal_checkpoint": {},
                "source_fork_checkpoint": {},
                "release_gate": copy.deepcopy(claims),
            }
        )
    elif report_name == "checkpoint_validation_bundle":
        manifest.update(
            {
                "inputs_sha256": "0" * 64,
                "inputs": {},
                "selection": {},
                "release_gate": copy.deepcopy(claims),
            }
        )
    else:
        manifest.update(
            {
                "input_fingerprint": "0" * 64,
                "inputs": {},
                "measurement_script": _file_identity(
                    ROOT / "scripts/audit_dense_checkpoint_drift.py"
                ),
                "release_gate": copy.deepcopy(claims),
                "passed": True,
                "authorizes_training": False,
                "training_started": False,
            }
        )
    manifest_path = root / "MANIFEST.json"
    _write_json(manifest_path, manifest)
    complete = root / "COMPLETE"
    complete.write_text(_sha(manifest_path) + "\n", encoding="ascii")
    return {
        "path": str(root.resolve()),
        "manifest_sha256": _sha(manifest_path),
        "complete_sha256": _sha(complete),
        "manifest_kind": kind,
        "complete_kind": None,
    }


def _calibration_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, Any]]:
    readiness = _fake_readiness()
    gate = readiness["calibration_gate"]
    closure_root = tmp_path / "formal-closure"
    closure_root.mkdir()
    closure_binding = {
        "path": str(closure_root.resolve()),
        "manifest_sha256": "1" * 64,
        "complete_sha256": "2" * 64,
        "bundle_fingerprint": "3" * 64,
    }
    closure = {
        "binding": closure_binding,
        "calibration_gate_contract_fingerprint": release._canonical_sha256(gate),
        "project_root": ROOT.resolve(),
        "readiness": readiness,
    }
    thresholds = gate["hard_thresholds"]
    values: dict[str, Any] = {
        "reference_epoch_max": thresholds["all_reference_epochs_eq"],
        "reused_sequences": thresholds["reused_sequences_eq"],
        "reused_tokens": thresholds["reused_tokens_eq"],
        "required_metrics_finite": {
            name: True for name in thresholds["all_required_metrics_finite"]
        },
        "clip_fraction": thresholds["clip_fraction_eq"],
        "best_aggregate_nll": thresholds["best_aggregate_nll_lte"],
        "final_aggregate_nll": thresholds["final_aggregate_nll_lte"],
        "chinese_source_nll": thresholds["chinese_source_nll_lte"],
        "final_scale_relative_l2": thresholds[
            "final_scale_relative_l2_lte"
        ],
        "candidate_global_steps": [40, 50],
        "same_frozen_v3_validation_contract": True,
        "fork_checkpoint_complete_sha256": gate["required_fork_checkpoint"][
            "complete_sha256"
        ],
    }
    by_bundle = {
        "training_report_bundle": {
            name: values[name]
            for name in (
                "reference_epoch_max",
                "reused_sequences",
                "reused_tokens",
                "required_metrics_finite",
                "clip_fraction",
                "fork_checkpoint_complete_sha256",
            )
        },
        "checkpoint_validation_bundle": {
            name: values[name]
            for name in (
                "best_aggregate_nll",
                "final_aggregate_nll",
                "chinese_source_nll",
                "candidate_global_steps",
                "same_frozen_v3_validation_contract",
            )
        },
        "checkpoint_drift_audit_bundle": {
            "final_scale_relative_l2": values["final_scale_relative_l2"]
        },
    }
    evidence = {
        name: _write_report_bundle(
            tmp_path / name,
            kind=release.CALIBRATION_REPORT_KINDS[name],
            claims=claims,
            payload_name=(
                "summary.json"
                if name == "checkpoint_validation_bundle"
                else "analysis.json"
            ),
        )
        for name, claims in by_bundle.items()
    }
    claims: dict[str, Any] = {}
    for bundle_name, rows in by_bundle.items():
        for name, value in rows.items():
            claims[name] = {
                "value": value,
                "evidence": {
                    "bundle": bundle_name,
                    "path": release.CLAIM_EVIDENCE_POLICY[name][1],
                    "json_pointer": f"/claims/{name}",
                },
            }

    calibration = yaml.safe_load(CALIBRATION_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(calibration, dict)
    data = calibration["data"]
    max_tokens = calibration["optimizer"]["max_tokens"]
    committed_tokens = max_tokens + 107_200
    closure["governed"] = {
        "capacity": {
            "phases": {
                "primary": {
                    "prepared_manifest_path": str(
                        (ROOT / data["manifest_path"]).resolve()
                    ),
                    "prepared_manifest_sha256": data["manifest_sha256"],
                    "source_map_sha256": data["source_map_sha256"],
                }
            }
        }
    }
    closure["capacity"] = {
        "stages": {
            "primary": {
                "prepared_identity": {
                    "available_unique_tokens": 20_000_000
                }
            }
        }
    }

    def checkpoint_binding(
        raw: Any,
        *,
        project_root: Path,
        label: str,
    ) -> dict[str, Any]:
        del project_root, label
        binding = dict(raw)
        step = int(Path(binding["path"]).name.removeprefix("step-"))
        final = step == 50
        return {
            **binding,
            "path": str(Path(binding["path"]).resolve()),
            "metadata": {
                "global_step": step,
                "committed_tokens": committed_tokens if final else step * 262_144,
                "kind": "milestone" if final else "periodic",
                "tag": "complete" if final else None,
                "run_id": calibration["run_id"],
                "extra": {
                    "data_manifest_sha256": data["manifest_sha256"],
                    "source_mix": {
                        "source_map_sha256": data["source_map_sha256"]
                    },
                },
            },
        }

    monkeypatch.setattr(
        release,
        "_authenticate_checkpoint_binding",
        checkpoint_binding,
    )
    monkeypatch.setattr(
        release,
        "_recompute_calibration_claims",
        lambda *_args, **_kwargs: copy.deepcopy(values),
    )
    checkpoint_bindings = [
        {
            "path": str((tmp_path / f"step-{step}").resolve()),
            "manifest_sha256": str(step % 10) * 64,
            "complete_sha256": chr(97 + index) * 64,
        }
        for index, step in enumerate((40, 50))
    ]
    attestation = {
        "schema_version": 1,
        "kind": release.CALIBRATION_KIND,
        "status": (
            "passed_authenticated_quality_gate_but_does_not_authorize_formal_training"
        ),
        "attestor": {
            "path": str(
                (
                    ROOT / "scripts/attest_v4_13m_calibration_release.py"
                ).resolve()
            ),
            "sha256": _sha(
                ROOT / "scripts/attest_v4_13m_calibration_release.py"
            ),
        },
        "formal_closure": closure_binding,
        "calibration_gate_contract_fingerprint": (
            closure["calibration_gate_contract_fingerprint"]
        ),
        "calibration_config": _file_identity(CALIBRATION_CONFIG),
        "evidence": evidence,
        "claims": claims,
        "candidate_checkpoints": checkpoint_bindings,
        "final_checkpoint": checkpoint_bindings[-1],
        "passed": True,
        "authorizes_training": False,
        "training_started": False,
    }
    attestation["attestation_fingerprint"] = release._canonical_sha256(
        attestation
    )
    path = tmp_path / "calibration-attestation/attestation.json"
    _write_json(path, attestation)
    complete = {
        "schema_version": 1,
        "kind": release.CALIBRATION_COMPLETE_KIND,
        "attestation": path.name,
        "attestation_sha256": _sha(path),
        "attestation_fingerprint": attestation["attestation_fingerprint"],
        "passed": True,
        "authorizes_training": False,
        "training_started": False,
    }
    _write_json(path.with_name("COMPLETE"), complete)
    return path, closure


def test_calibration_attestation_recomputes_every_hard_gate_from_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, closure = _calibration_fixture(tmp_path, monkeypatch)
    result = release._authenticate_calibration_attestation(
        path,
        closure=closure,
    )
    assert result["passed"] is True
    assert result["authorizes_training"] is False
    assert result["evaluation"]["observed_steps"] == [40, 50]
    assert result["evaluation"]["final_global_step"] == 50

    facts = tmp_path / "checkpoint_validation_bundle/summary.json"
    value = _json(facts)
    value["claims"]["final_aggregate_nll"] = 999.0
    _write_json(facts, value)
    with pytest.raises(release.ReleaseError, match="payload identity differs"):
        release._authenticate_calibration_attestation(path, closure=closure)


def test_source_bound_claims_only_reports_are_rejected_without_test_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, closure = _calibration_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        release,
        "_recompute_calibration_claims",
        REAL_RECOMPUTE_CALIBRATION_CLAIMS,
    )
    with pytest.raises(
        release.ReleaseError,
        match=r"training report analysis\.run|training evidence recomputation",
    ):
        release._authenticate_calibration_attestation(path, closure=closure)


def test_final_release_shape_is_accepted_by_source_bound_governed_planner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _release_plan, _result = _publish_fake_release(tmp_path, monkeypatch)
    readiness_path = args.output / release.FINAL_READINESS_NAME
    readiness = _json(readiness_path)
    config = yaml.safe_load(
        (args.output / release.FINAL_CONFIG_NAME).read_text(encoding="utf-8")
    )
    capacity = _fake_snapshot(tmp_path / "planner-fixture")["closure"][
        "governed"
    ]["capacity"]
    # Use the identities actually written into the published config.
    primary = capacity["phases"]["primary"]
    cooldown = capacity["phases"]["cooldown"]
    primary.update(
        {
            "prepared_manifest_path": config["data"]["manifest_path"],
            "prepared_manifest_sha256": config["data"]["manifest_sha256"],
            "source_map_sha256": config["data"]["source_map_sha256"],
        }
    )
    cooldown.update(
        {
            "prepared_manifest_path": config["data"][
                "quality_cooldown_manifest_path"
            ],
            "prepared_manifest_sha256": config["data"][
                "quality_cooldown_manifest_sha256"
            ],
        }
    )
    phase = {
        "path": config["data"]["phase_disjointness_attestation_path"],
        "sha256": config["data"]["phase_disjointness_attestation_sha256"],
        "attestation_fingerprint": "5" * 64,
    }
    capacity["phase_disjointness_attestation"] = phase
    formal = {
        "validation_phases": {
            "primary": {"fixture": True},
            "cooldown": {"fixture": True},
        },
        "formal_phase_train_disjointness": phase,
    }
    release_bindings = {
        "wikipedia_license": {
            "passed": True,
            "authorizes_training": True,
        },
        "chinese_semantic_quality": {
            "passed": True,
            "authorizes_training": True,
        },
    }
    monkeypatch.setattr(
        governed,
        "_capacity_source_map_bindings",
        lambda _readiness, _root: (copy.deepcopy(capacity), []),
    )
    monkeypatch.setattr(
        governed,
        "_formal_bindings",
        lambda _readiness, _root: (copy.deepcopy(formal), []),
    )
    monkeypatch.setattr(
        governed,
        "_release_gate_bindings",
        lambda _readiness, _root: (copy.deepcopy(release_bindings), []),
    )
    plan = governed.build_governed_plan(readiness_path)
    assert plan["launch_enabled"] is True
    assert plan["readiness_issues"] == []
    assert readiness["blockers"] == []
    assert readiness["calibration_gate"]["authorizes_training"] is True
    assert readiness["formal_validation_gate"]["authorizes_training"] is True
