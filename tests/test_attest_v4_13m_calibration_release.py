from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
READINESS = ROOT / "locks/base-dense-v4-250m-pilot.readiness.json"
CALIBRATION_CONFIG = ROOT / "configs/base/dense-v4-13m-low-lr-calibration.yaml"


def _load_script() -> ModuleType:
    path = ROOT / "scripts/attest_v4_13m_calibration_release.py"
    spec = importlib.util.spec_from_file_location(
        "attest_v4_13m_calibration_release",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


attest = _load_script()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _report(
    root: Path,
    *,
    kind: str,
    claims: dict[str, Any],
    payload_name: str,
) -> None:
    facts = root / payload_name
    _write_json(facts, {"claims": claims})
    report_name = next(
        name
        for name, expected_kind in attest.EXPECTED_REPORT_KINDS.items()
        if kind == expected_kind
    )
    producer = (
        ROOT
        / "scripts"
        / attest.publisher.CALIBRATION_REPORT_PRODUCERS[report_name]
    )
    manifest = {
        "schema_version": 1,
        "kind": kind,
        "bundle_producer": {
            "path": str(producer.resolve()),
            "size": producer.stat().st_size,
            "sha256": _sha(producer),
        },
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
        auditor = ROOT / "scripts/audit_dense_checkpoint_drift.py"
        manifest.update(
            {
                "input_fingerprint": "0" * 64,
                "inputs": {},
                "measurement_script": {
                    "path": str(auditor.resolve()),
                    "size": auditor.stat().st_size,
                    "sha256": _sha(auditor),
                },
                "release_gate": copy.deepcopy(claims),
                "passed": True,
                "authorizes_training": False,
                "training_started": False,
            }
        )
    manifest_path = root / "MANIFEST.json"
    _write_json(manifest_path, manifest)
    (root / "COMPLETE").write_text(
        _sha(manifest_path) + "\n",
        encoding="ascii",
    )


def _checkpoint(path: Path) -> None:
    _write_json(path / "manifest.json", {"fixture": path.name})
    (path / "COMPLETE").write_text(
        _sha(path / "manifest.json") + "\n",
        encoding="ascii",
    )


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Namespace, dict[str, Any]]:
    readiness = _json(READINESS)
    gate = copy.deepcopy(readiness["calibration_gate"])
    gate["config"] = {
        "path": str(CALIBRATION_CONFIG.resolve()),
        "sha256": _sha(CALIBRATION_CONFIG),
    }
    calibration = yaml.safe_load(CALIBRATION_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(calibration, dict)
    data = calibration["data"]
    closure = {
        "binding": {
            "path": str((tmp_path / "closure").resolve()),
            "manifest_sha256": "1" * 64,
            "complete_sha256": "2" * 64,
            "bundle_fingerprint": "3" * 64,
        },
        "calibration_gate_contract_fingerprint": (
            attest.publisher._canonical_sha256(gate)
        ),
        "project_root": ROOT.resolve(),
        "readiness": {"calibration_gate": gate},
        "governed": {
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
        },
        "capacity": {
            "stages": {
                "primary": {
                    "prepared_identity": {
                        "available_unique_tokens": 20_000_000
                    }
                }
            }
        },
    }
    monkeypatch.setattr(
        attest,
        "_authenticate_closure",
        lambda _path: copy.deepcopy(closure),
    )

    thresholds = gate["hard_thresholds"]
    values = {
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
    assignments = {
        "training_report_bundle": (
            "reference_epoch_max",
            "reused_sequences",
            "reused_tokens",
            "required_metrics_finite",
            "clip_fraction",
            "fork_checkpoint_complete_sha256",
        ),
        "checkpoint_validation_bundle": (
            "best_aggregate_nll",
            "final_aggregate_nll",
            "chinese_source_nll",
            "candidate_global_steps",
            "same_frozen_v3_validation_contract",
        ),
        "checkpoint_drift_audit_bundle": (
            "final_scale_relative_l2",
        ),
    }
    report_paths: dict[str, Path] = {}
    locations: dict[str, Any] = {}
    for name, claim_names in assignments.items():
        root = tmp_path / name
        _report(
            root,
            kind=attest.EXPECTED_REPORT_KINDS[name],
            claims={claim_name: values[claim_name] for claim_name in claim_names},
            payload_name=(
                "summary.json"
                if name == "checkpoint_validation_bundle"
                else "analysis.json"
            ),
        )
        report_paths[name] = root
        for claim_name in claim_names:
            locations[claim_name] = {
                "bundle": name,
                "path": attest.publisher.CLAIM_EVIDENCE_POLICY[claim_name][1],
                "json_pointer": f"/claims/{claim_name}",
            }
    spec: dict[str, Any] = {
        "schema_version": 1,
        "kind": attest.SPEC_KIND,
        "claims": locations,
    }
    spec["spec_fingerprint"] = attest._canonical_sha256(spec)
    spec_path = tmp_path / "claim-locations.json"
    _write_json(spec_path, spec)

    step40 = tmp_path / "step-40"
    step50 = tmp_path / "step-50"
    _checkpoint(step40)
    _checkpoint(step50)
    final_tokens = calibration["optimizer"]["max_tokens"] + 107_200

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
                "committed_tokens": final_tokens if final else 10_485_760,
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
        attest.publisher,
        "_authenticate_checkpoint_binding",
        checkpoint_binding,
    )
    monkeypatch.setattr(
        attest.publisher,
        "_recompute_calibration_claims",
        lambda *_args, **_kwargs: copy.deepcopy(values),
    )
    args = Namespace(
        closure=tmp_path / "closure",
        training_report=report_paths["training_report_bundle"],
        checkpoint_validation_report=report_paths[
            "checkpoint_validation_bundle"
        ],
        checkpoint_drift_report=report_paths[
            "checkpoint_drift_audit_bundle"
        ],
        candidate_checkpoint=[step40, step50],
        final_checkpoint=step50,
        claim_locations=spec_path,
        output=tmp_path / "calibration-closure",
    )
    return args, closure


def test_claim_location_spec_is_canonical_and_contains_locations_only(
    tmp_path: Path,
) -> None:
    claims = {
        name: {
            "bundle": attest.publisher.CLAIM_EVIDENCE_POLICY[name][0],
            "path": attest.publisher.CLAIM_EVIDENCE_POLICY[name][1],
            "json_pointer": f"/release/{name}",
        }
        for name in attest.publisher.CLAIM_NAMES
    }
    value: dict[str, Any] = {
        "schema_version": 1,
        "kind": attest.SPEC_KIND,
        "claims": claims,
    }
    value["spec_fingerprint"] = attest._canonical_sha256(value)
    path = tmp_path / "locations.json"
    _write_json(path, value)
    assert attest._claim_locations(path) == claims
    value["claims"]["reused_tokens"]["json_pointer"] = "/tampered"
    _write_json(path, value)
    with pytest.raises(
        attest.CalibrationClosureError,
        match="fingerprint",
    ):
        attest._claim_locations(path)


def test_production_attestor_closes_reports_and_consumer_recomputes_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, closure = _fixture(tmp_path, monkeypatch)
    result = attest.close_calibration_evidence(args)
    assert result["passed"] is True
    assert result["authorizes_training"] is False
    assert result["training_started"] is False
    assert set(path.name for path in args.output.iterdir()) == {
        "attestation.json",
        "COMPLETE",
    }
    attestation = _json(args.output / "attestation.json")
    complete = _json(args.output / "COMPLETE")
    assert attestation["attestor"] == {
        "path": str((ROOT / "scripts/attest_v4_13m_calibration_release.py").resolve()),
        "sha256": _sha(
            ROOT / "scripts/attest_v4_13m_calibration_release.py"
        ),
    }
    assert complete["attestation_sha256"] == _sha(
        args.output / "attestation.json"
    )
    # Re-run the exact production consumer independently.
    authenticated = attest.publisher._authenticate_calibration_attestation(
        args.output / "attestation.json",
        closure=closure,
    )
    assert authenticated["evaluation"]["observed_steps"] == [40, 50]
    assert authenticated["evaluation"]["passed"] is True


def test_attestor_rejects_existing_or_midflight_changed_output_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _closure = _fixture(tmp_path, monkeypatch)
    attest.close_calibration_evidence(args)
    before = {
        path.name: _sha(path) for path in args.output.iterdir() if path.is_file()
    }
    with pytest.raises(
        attest.CalibrationClosureError,
        match="already exists",
    ):
        attest.close_calibration_evidence(args)
    assert before == {
        path.name: _sha(path) for path in args.output.iterdir() if path.is_file()
    }

    changed_args = copy.copy(args)
    changed_args.output = tmp_path / "changed-output"
    original = attest._build_attestation
    calls = 0

    def changed(current_args: Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal calls
        calls += 1
        value, closure = original(current_args)
        if calls == 2:
            value["claims"]["reused_tokens"]["value"] = 1
            value["attestation_fingerprint"] = attest._canonical_sha256(
                {
                    key: item
                    for key, item in value.items()
                    if key != "attestation_fingerprint"
                }
            )
        return value, closure

    monkeypatch.setattr(attest, "_build_attestation", changed)
    with pytest.raises(
        attest.CalibrationClosureError,
        match="changed during closure",
    ):
        attest.close_calibration_evidence(changed_args)
    assert calls == 2
    assert not changed_args.output.exists()
    assert not list(
        changed_args.output.parent.glob(
            f".{changed_args.output.name}.incomplete-*"
        )
    )


def test_attestor_fsyncs_staging_before_rename_and_parent_after(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _closure = _fixture(tmp_path, monkeypatch)
    real_fsync = os.fsync
    fsynced: list[Path] = []

    def track_fsync(descriptor: int) -> None:
        fsynced.append(
            Path(os.readlink(f"/proc/self/fd/{descriptor}")).resolve()
        )
        real_fsync(descriptor)

    monkeypatch.setattr(attest.os, "fsync", track_fsync)
    real_rename = attest.publisher._rename_directory_noreplace

    def checked_rename(source: Path, destination: Path) -> None:
        assert source.resolve() in fsynced
        real_rename(source, destination)

    monkeypatch.setattr(
        attest.publisher,
        "_rename_directory_noreplace",
        checked_rename,
    )
    attest.close_calibration_evidence(args)
    assert args.output.parent.resolve() in fsynced
