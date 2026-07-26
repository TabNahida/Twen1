from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
CAPACITY_TEMPLATE = ROOT / "locks/base-data-sources-v4-250m.capacity-attestation.json"
READINESS_TEMPLATE = ROOT / "locks/base-dense-v4-250m-pilot.readiness.json"


def _load_script() -> ModuleType:
    path = ROOT / "scripts/close_v4_250m_formal_evidence.py"
    spec = importlib.util.spec_from_file_location(
        "close_v4_250m_formal_evidence",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


closure = _load_script()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _fake_identity(path: str, digit: str = "a") -> dict[str, object]:
    return {
        "path": path,
        "size": 123,
        "sha256": digit * 64,
    }


def _fake_phase(
    phase: str,
    capacity: dict[str, object],
) -> dict[str, object]:
    stage = capacity["stages"][phase]  # type: ignore[index]
    rows = stage["per_source_capacity"]
    source_tokens = {
        str(row["source_id"]): int(row["required_clean_tokens"]) + 4096
        for row in rows
    }
    validation_tokens = {source_id: 4096 for source_id in source_tokens}
    weights = {
        str(row["source_id"]): int(row["mix_basis_points"])
        for row in rows
    }
    train_tokens = sum(source_tokens.values())
    train_samples = int(stage["required_prepared_samples"]) + 1
    prefix = f"/fixture/{phase}"
    return {
        "phase": phase,
        "extracted": {
            "manifest_path": f"{prefix}/corpus-manifest.json",
            "manifest_sha256": "1" * 64,
            "corpus_fingerprint": "2" * 64,
            "complete": _fake_identity(f"{prefix}/COMPLETE", "3"),
        },
        "audit": {
            "path": f"{prefix}/audit/attestation.json",
            "sha256": "4" * 64,
            "attestation_fingerprint": "5" * 64,
            "ready_for_training": True,
        },
        "train": SimpleNamespace(token_count=train_tokens, sequence_count=train_samples),
        "train_source_map": SimpleNamespace(source_mix_weights=weights),
        "train_prepared": {
            "manifest_path": f"{prefix}/train/manifest.json",
            "manifest_sha256": "6" * 64,
            "dataset_fingerprint": "7" * 64,
            "source_map_sha256": "8" * 64,
            "token_count": train_tokens,
            "sequence_count": train_samples,
            "available_unique_tokens": train_tokens,
            "available_unique_samples": train_samples,
            "sequence_length": 4096,
        },
        "source_tokens": source_tokens,
        "validation": SimpleNamespace(
            token_count=sum(validation_tokens.values()),
            sequence_count=len(validation_tokens),
        ),
        "validation_source_map": SimpleNamespace(source_mix_weights=weights),
        "validation_prepared": {
            "manifest_path": f"{prefix}/validation/manifest.json",
            "manifest_sha256": "9" * 64,
            "dataset_fingerprint": "a" * 64,
            "source_map_sha256": "b" * 64,
            "token_count": sum(validation_tokens.values()),
            "sequence_count": len(validation_tokens),
            "available_unique_tokens": sum(validation_tokens.values()),
            "available_unique_samples": len(validation_tokens),
            "sequence_length": 4096,
        },
        "validation_source_tokens": validation_tokens,
        "attribution": _fake_identity(f"{prefix}/attribution-files.txt", "c"),
    }


def _passing_gate(algorithm: str, *, near: bool = False) -> dict[str, object]:
    value: dict[str, object] = {
        "algorithm": algorithm,
        "matches": 0,
        "passed": True,
    }
    if near:
        value["estimated_jaccard_threshold"] = 0.8
    return value


def _fake_snapshot() -> dict[str, object]:
    capacity = _json(CAPACITY_TEMPLATE)
    readiness = _json(READINESS_TEMPLATE)
    checkpoint_sha = readiness["fork_policy"]["required_checkpoint_complete_sha256"]  # type: ignore[index]
    return {
        "capacity_template": capacity,
        "capacity_template_identity": _fake_identity(str(CAPACITY_TEMPLATE), "d"),
        "readiness_template": readiness,
        "readiness_template_identity": _fake_identity(str(READINESS_TEMPLATE), "e"),
        "blocked_config": _fake_identity(
            str(ROOT / "configs/base/dense-v4-250m-pilot.blocked.yaml"),
            "f",
        ),
        "phases": {
            phase: _fake_phase(phase, capacity)
            for phase in ("primary", "cooldown")
        },
        "phase_disjointness": {
            "identity": _fake_identity("/fixture/phase/attestation.json", "1"),
            "attestation_fingerprint": "2" * 64,
            "gates": {
                "stable_id_exact": _passing_gate(
                    "source-scoped-authenticated-stable-id-intersection-v1"
                ),
                "normalized_text_exact": _passing_gate(
                    "unicode-nfkc-whitespace-sha256-intersection-v1"
                ),
                "near_duplicate": _passing_gate(
                    "lexical-5gram-one-permutation-minhash-lsh-v1",
                    near=True,
                ),
            },
        },
        "formal_validation_disjointness": {
            "identity": _fake_identity("/fixture/formal/attestation.json", "3"),
            "attestation_fingerprint": "4" * 64,
            "near_duplicate_threshold": 0.8,
            "gates": {
                "train_validation_stable_id": _passing_gate(
                    "source-scoped-authenticated-stable-id-intersection-v1"
                ),
                "train_validation_normalized_exact": _passing_gate(
                    "unicode-nfkc-whitespace-sha256-intersection-v1"
                ),
                "train_validation_near_duplicate": _passing_gate(
                    "lexical-5gram-one-permutation-minhash-lsh-v1",
                    near=True,
                ),
            },
            "passed": True,
        },
        "formal_baseline": {
            "root": "/fixture/formal-baseline",
            "summary": _fake_identity("/fixture/formal-baseline/summary.json", "5"),
            "manifest": _fake_identity("/fixture/formal-baseline/MANIFEST.json", "6"),
            "complete": _fake_identity("/fixture/formal-baseline/COMPLETE", "7"),
            "checkpoint_complete_sha256": checkpoint_sha,
            "gate": {
                "passed": True,
                "authorizes_training": False,
                "training_started_by_summarizer": False,
            },
        },
        "closure_source_sha256": "8" * 64,
        "twen_source_tree_sha256": "9" * 64,
    }


def _closure_identity() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": closure.CLOSURE_KIND,
        "input_fingerprint": "a" * 64,
        "closure_source_sha256": "b" * 64,
        "twen_source_tree_sha256": "c" * 64,
        "launch_enabled": False,
        "authorizes_training": False,
        "training_started": False,
    }


def test_closed_outputs_pass_data_and_formal_gates_but_never_authorize_launch(
    tmp_path: Path,
) -> None:
    snapshot = _fake_snapshot()
    capacity = closure._closed_capacity(
        snapshot,
        closure_identity=_closure_identity(),
    )
    assert capacity["overall"]["passed"] is True
    assert capacity["launch_enabled"] is False
    assert capacity["authorizes_training"] is False
    assert capacity["training_started"] is False
    assert (
        capacity["phase_disjointness"]["stable_id_exact"]["algorithm"]
        == "source-scoped-authenticated-stable-id-intersection-v1"
    )
    assert set(capacity["phase_disjointness"]) == {
        "stable_id_exact",
        "normalized_text_exact",
        "near_duplicate",
    }
    assert capacity["config"]["contains_pending_identity_sentinels"] is True
    assert all(stage["passed"] is True for stage in capacity["stages"].values())

    capacity_path = tmp_path / "capacity-attestation.json"
    capacity_path.write_text(json.dumps(capacity), encoding="utf-8")
    readiness = closure._closed_readiness(
        snapshot,
        closure_identity=_closure_identity(),
        capacity_path=capacity_path,
        capacity_identity={
            **_fake_identity(str(capacity_path), "d"),
            "passed": True,
            "authorizes_training": False,
        },
    )
    assert readiness["formal_validation_gate"]["passed"] is True
    assert readiness["formal_validation_gate"]["authorizes_training"] is False
    assert readiness["calibration_gate"]["passed"] is False
    assert readiness["pause_evaluation_policy"]["controller_implemented"] is False
    assert readiness["launch_enabled"] is False
    assert readiness["authorizes_training"] is False
    assert readiness["launch_command_after_all_gates_pass"] is None
    assert any("13M low-LR calibration" in item for item in readiness["blockers"])
    assert any("external governed pause" in item for item in readiness["blockers"])
    assert not any("formal train/validation" in item for item in readiness["blockers"])


def test_capacity_underfill_fails_before_any_output() -> None:
    snapshot = _fake_snapshot()
    stage = snapshot["capacity_template"]["stages"]["primary"]  # type: ignore[index]
    source = stage["per_source_capacity"][0]
    source_id = source["source_id"]
    snapshot["phases"]["primary"]["source_tokens"][source_id] = (  # type: ignore[index]
        int(source["required_clean_tokens"]) - 1
    )
    with pytest.raises(closure.ClosureError, match="governed capacity underfill"):
        closure._closed_capacity(
            snapshot,
            closure_identity=_closure_identity(),
        )


def test_atomic_bundle_reauthenticates_and_rejects_changed_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = _fake_snapshot()
    args = Namespace(output=tmp_path / "closed")
    monkeypatch.setattr(closure, "_authenticate_inputs", lambda _args: snapshot)
    result = closure.close_formal_evidence(args)
    output = Path(result["output"])
    assert output.is_dir()
    manifest = _json(output / "MANIFEST.json")
    complete = _json(output / "COMPLETE")
    assert manifest["launch_enabled"] is False
    assert manifest["authorizes_training"] is False
    assert complete["manifest_sha256"] == hashlib.sha256(
        (output / "MANIFEST.json").read_bytes()
    ).hexdigest()
    readiness = _json(output / "readiness.json")
    assert readiness["launch_enabled"] is False
    assert readiness["launch_command_after_all_gates_pass"] is None

    changed = copy.deepcopy(snapshot)
    changed["formal_baseline"]["summary"]["sha256"] = "0" * 64  # type: ignore[index]
    calls = iter((snapshot, changed))
    monkeypatch.setattr(closure, "_authenticate_inputs", lambda _args: next(calls))
    changed_output = tmp_path / "changed"
    with pytest.raises(closure.ClosureError, match="changed during publication"):
        closure.close_formal_evidence(Namespace(output=changed_output))
    assert not changed_output.exists()
    assert not list(tmp_path.glob(".changed.incomplete-*"))


def _bundle_identity(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_formal_baseline_bundle_is_fully_hashed_and_rebuilt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = _fake_snapshot()
    phases = snapshot["phases"]
    root = tmp_path / "baseline"
    (root / "charts").mkdir(parents=True)
    summary = {
        "schema_version": 1,
        "kind": closure.FORMAL_BASELINE_KIND,
        "gate": {
            "passed": True,
            "authorizes_training": False,
            "training_started_by_summarizer": False,
        },
        "phases": [
            {
                "phase": phase,
                "prepared": {
                    "path": phases[phase]["validation_prepared"]["manifest_path"],
                    "sha256": phases[phase]["validation_prepared"]["manifest_sha256"],
                    "dataset_fingerprint": phases[phase]["validation_prepared"][
                        "dataset_fingerprint"
                    ],
                    "token_count": phases[phase]["validation_prepared"][
                        "available_unique_tokens"
                    ],
                    "sequence_count": phases[phase]["validation_prepared"][
                        "available_unique_samples"
                    ],
                },
                "evaluation": {
                    "root": f"/fixture/evaluation/{phase}",
                },
            }
            for phase in ("primary", "cooldown")
        ],
        "legacy_six_source_baseline": {
            "identity": {
                "summary": _fake_identity("/fixture/legacy/summary.json", "1"),
            },
            "checkpoint_artifact": {
                "complete_sha256": "2" * 64,
            },
        },
    }
    (root / "summary.json").write_text(
        json.dumps(summary, sort_keys=True),
        encoding="utf-8",
    )
    (root / "REPORT.zh-CN.md").write_text("report\n", encoding="utf-8")
    (root / "charts/formal-source-nll.svg").write_text("<svg/>\n", encoding="utf-8")
    (root / "charts/formal-source-tokens.svg").write_text("<svg/>\n", encoding="utf-8")
    files = {
        relative: _bundle_identity(root, relative)
        for relative in (
            "summary.json",
            "REPORT.zh-CN.md",
            "charts/formal-source-nll.svg",
            "charts/formal-source-tokens.svg",
        )
    }
    manifest = {
        "schema_version": 1,
        "kind": closure.FORMAL_BASELINE_BUNDLE_KIND,
        "gate": summary["gate"],
        "files": files,
    }
    (root / "MANIFEST.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    (root / "COMPLETE").write_text(
        hashlib.sha256((root / "MANIFEST.json").read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )
    monkeypatch.setattr(
        closure,
        "_load_script",
        lambda *_args: SimpleNamespace(build_summary=lambda **_kwargs: summary),
    )
    value = closure._validate_formal_baseline_bundle(
        root,
        phases=phases,
        validation_disjointness_path=tmp_path / "disjointness.json",
    )
    assert value["checkpoint_complete_sha256"] == "2" * 64

    (root / "charts/formal-source-nll.svg").write_text("<svg>tampered</svg>\n", encoding="utf-8")
    with pytest.raises(closure.ClosureError, match="payload identity differs"):
        closure._validate_formal_baseline_bundle(
            root,
            phases=phases,
            validation_disjointness_path=tmp_path / "disjointness.json",
        )


def test_checked_in_templates_are_still_pending_and_fail_closed() -> None:
    capacity, readiness, _config = closure._require_template_policy(
        CAPACITY_TEMPLATE,
        READINESS_TEMPLATE,
    )
    assert capacity["launch_enabled"] is False
    assert (
        capacity["phase_disjointness"]["stable_id_exact"]["algorithm"]
        == "source-scoped-authenticated-stable-id-intersection-v1"
    )
    assert readiness["launch_enabled"] is False
    assert readiness["launch_command_after_all_gates_pass"] is None


def test_planning_generator_uses_source_scoped_stable_id_contract() -> None:
    path = ROOT / "scripts/prepare_v4_250m_data_plan.py"
    spec = importlib.util.spec_from_file_location("prepare_v4_250m_data_plan", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generated = module._make_capacity_attestation()
    assert (
        generated["phase_disjointness"]["stable_id_exact"]["algorithm"]
        == "source-scoped-authenticated-stable-id-intersection-v1"
    )


@pytest.mark.parametrize(
    ("target", "field", "value"),
    (
        ("capacity", "launch_enabled", True),
        ("capacity", "authorizes_training", True),
        ("readiness", "launch_enabled", True),
        ("readiness", "launch_command_after_all_gates_pass", "train now"),
    ),
)
def test_template_policy_rejects_any_launch_enablement(
    tmp_path: Path,
    target: str,
    field: str,
    value: object,
) -> None:
    capacity = _json(CAPACITY_TEMPLATE)
    readiness = _json(READINESS_TEMPLATE)
    capacity_path = tmp_path / "capacity.json"
    readiness_path = tmp_path / "readiness.json"
    readiness["required_capacity_attestation"] = str(capacity_path)
    if target == "capacity":
        capacity[field] = value
    else:
        readiness[field] = value
    capacity_path.write_text(json.dumps(capacity), encoding="utf-8")
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
    with pytest.raises(closure.ClosureError):
        closure._require_template_policy(capacity_path, readiness_path)
