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

ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> ModuleType:
    path = ROOT / "scripts/generate_v4_calibration_drift_bundle.py"
    spec = importlib.util.spec_from_file_location(
        "generate_v4_calibration_drift_bundle",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bundle = _load_script()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _checkpoint(
    root: Path,
    *,
    step: int,
    tokens: int,
    kind: str,
    tag: str | None,
) -> dict[str, Any]:
    _write_json(root / "manifest.json", {"fixture": root.name})
    (root / "COMPLETE").write_text(
        _sha(root / "manifest.json") + "\n",
        encoding="ascii",
    )
    return {
        "path": str(root.resolve()),
        "manifest_sha256": _sha(root / "manifest.json"),
        "complete_sha256": _sha(root / "COMPLETE"),
        "metadata": {
            "global_step": step,
            "committed_tokens": tokens,
            "kind": kind,
            "tag": tag,
            "run_id": "base-dense-v4-13m-low-lr-calibration",
        },
    }


def _snapshot(
    tmp_path: Path,
    *,
    final_scale_relative_l2: float = 0.04,
) -> dict[str, Any]:
    closure_root = tmp_path / "formal-closure"
    _write_json(closure_root / "MANIFEST.json", {"fixture": "closure"})
    _write_json(closure_root / "COMPLETE", {"fixture": "complete"})
    config = tmp_path / "calibration.yaml"
    config.write_text("run_id: calibration-fixture\n", encoding="utf-8")
    baseline = _checkpoint(
        tmp_path / "baseline",
        step=1_912,
        tokens=500_009_962,
        kind="milestone",
        tag="complete",
    )
    step40 = _checkpoint(
        tmp_path / "step-40",
        step=40,
        tokens=10_485_760,
        kind="periodic",
        tag=None,
    )
    step50 = _checkpoint(
        tmp_path / "step-50",
        step=50,
        tokens=13_107_200,
        kind="milestone",
        tag="complete",
    )
    rows = [
        {
            **bundle._public_checkpoint(step40),
            "adapter_relative_l2": 0.02,
            "scale_relative_l2": 0.03,
        },
        {
            **bundle._public_checkpoint(step50),
            "adapter_relative_l2": 0.025,
            "scale_relative_l2": final_scale_relative_l2,
        },
    ]
    passed = final_scale_relative_l2 <= 0.05
    analysis = {
        "schema_version": bundle.SCHEMA_VERSION,
        "kind": bundle.ANALYSIS_KIND,
        "execution": {
            "device": "cpu",
            "cuda_initialized": False,
            "model_built": False,
            "optimizer_created": False,
        },
        "baseline": {
            key: baseline[key]
            for key in ("path", "manifest_sha256", "complete_sha256")
        },
        "inventory": {
            "model_tensor_count": 49,
            "adapter_tensor_count": 48,
            "adapter_element_count": 201_326_592,
            "scale_tensor_count": 1,
            "scale_element_count": 24,
        },
        "candidates": [
            {
                key: row[key]
                for key in ("path", "manifest_sha256", "complete_sha256")
            }
            | {
                "adapter": {"relative_l2": row["adapter_relative_l2"]},
                "scale": {"relative_l2": row["scale_relative_l2"]},
            }
            for row in rows
        ],
        "release_gate": {
            "final_checkpoint": bundle._public_checkpoint(step50),
            "final_scale_relative_l2": final_scale_relative_l2,
            "final_scale_relative_l2_lte": 0.05,
            "passed": passed,
            "authorizes_training": False,
        },
    }
    return {
        "closure": {
            "binding": {
                "path": str(closure_root.resolve()),
                "manifest_sha256": _sha(closure_root / "MANIFEST.json"),
                "complete_sha256": _sha(closure_root / "COMPLETE"),
                "bundle_fingerprint": "1" * 64,
            }
        },
        "baseline": baseline,
        "candidates": [step40, step50],
        "final_checkpoint": step50,
        "lineage": {
            "calibration_config": bundle._identity(config),
            "candidate_global_steps": [40, 50],
            "final_scale_relative_l2_threshold": 0.05,
        },
        "analysis": analysis,
        "rows": rows,
        "final_scale_relative_l2": final_scale_relative_l2,
        "threshold": 0.05,
        "passed": passed,
        "auditor": bundle._identity(
            ROOT / "scripts/audit_dense_checkpoint_drift.py"
        ),
        "bundler": bundle._identity(
            ROOT / "scripts/generate_v4_calibration_drift_bundle.py"
        ),
    }


def _args(tmp_path: Path, snapshot: dict[str, Any]) -> Namespace:
    return Namespace(
        closure=tmp_path / "formal-closure",
        candidate=[
            Path(row["path"]) for row in snapshot["candidates"]
        ],
        final_checkpoint=Path(snapshot["final_checkpoint"]["path"]),
        output=tmp_path / "drift-bundle",
    )


def _install_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    snapshot: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        bundle,
        "_snapshot",
        lambda _args: copy.deepcopy(snapshot),
    )
    expected = bundle._canonical_sha256(bundle._input_contract(snapshot))

    def reauthenticate(
        _args: Namespace,
        *,
        expected: dict[str, Any],
    ) -> None:
        assert bundle._canonical_sha256(expected) == expected_fingerprint

    expected_fingerprint = expected
    monkeypatch.setattr(bundle, "_reauthenticate_inputs", reauthenticate)


def _binding(output: Path) -> dict[str, Any]:
    return {
        "path": str(output.resolve()),
        "manifest_sha256": _sha(output / "MANIFEST.json"),
        "complete_sha256": _sha(output / "COMPLETE"),
        "manifest_kind": bundle.BUNDLE_KIND,
        "complete_kind": None,
    }


def test_generate_pass_bundle_report_manifest_and_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(tmp_path)
    args = _args(tmp_path, snapshot)
    _install_snapshot(monkeypatch, snapshot)

    result = bundle.generate_bundle(args)

    assert result["passed"] is True
    assert result["authorizes_training"] is False
    assert result["training_started"] is False
    assert {path.name for path in args.output.iterdir()} == {
        "analysis.json",
        "REPORT.zh-CN.md",
        "MANIFEST.json",
        "COMPLETE",
    }
    analysis = json.loads(
        (args.output / "analysis.json").read_text(encoding="utf-8")
    )
    assert analysis["release_gate"]["final_scale_relative_l2"] == 0.04
    assert analysis["release_gate"]["passed"] is True
    report = (args.output / "REPORT.zh-CN.md").read_text(encoding="utf-8")
    assert "| 40 |" in report
    assert "| 50 |" in report
    assert "0.04000000" in report
    assert "结论: **PASS**" in report
    manifest = json.loads(
        (args.output / "MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["kind"] == bundle.BUNDLE_KIND
    assert manifest["release_gate"] == analysis["release_gate"]
    assert manifest["passed"] is True
    assert set(manifest["files"]) == {"analysis.json", "REPORT.zh-CN.md"}
    for relative, identity in manifest["files"].items():
        payload = args.output / relative
        assert identity == {
            "path": relative,
            "size": payload.stat().st_size,
            "sha256": _sha(payload),
        }
    assert (args.output / "COMPLETE").read_text(encoding="ascii").strip() == (
        _sha(args.output / "MANIFEST.json")
    )
    authenticated = bundle.publisher._authenticate_report_bundle(
        _binding(args.output),
        project_root=tmp_path,
        label="drift bundle",
    )
    assert set(authenticated["payloads"]) == {
        "analysis.json",
        "REPORT.zh-CN.md",
    }
    assert not list(tmp_path.glob(".drift-bundle.incomplete-*"))


def test_failed_scale_gate_still_emits_a_fail_closed_audit_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(tmp_path, final_scale_relative_l2=0.050001)
    args = _args(tmp_path, snapshot)
    _install_snapshot(monkeypatch, snapshot)

    result = bundle.generate_bundle(args)

    assert result["passed"] is False
    analysis = json.loads(
        (args.output / "analysis.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (args.output / "MANIFEST.json").read_text(encoding="utf-8")
    )
    report = (args.output / "REPORT.zh-CN.md").read_text(encoding="utf-8")
    assert analysis["release_gate"]["passed"] is False
    assert manifest["passed"] is False
    assert manifest["authorizes_training"] is False
    assert "结论: **FAIL**" in report


def test_raw_audit_rejects_tampered_checkpoint_identity_and_computes_gate(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    raw = copy.deepcopy(snapshot["analysis"])
    raw.pop("release_gate")

    validated = bundle._validate_raw_audit(
        raw,
        baseline=snapshot["baseline"],
        candidates=snapshot["candidates"],
        final_checkpoint=snapshot["final_checkpoint"],
        threshold=0.05,
    )
    assert validated["passed"] is True
    assert validated["analysis"]["release_gate"]["passed"] is True

    tampered = copy.deepcopy(raw)
    tampered["candidates"][0]["manifest_sha256"] = "0" * 64
    with pytest.raises(bundle.DriftBundleError, match="identity differs"):
        bundle._validate_raw_audit(
            tampered,
            baseline=snapshot["baseline"],
            candidates=snapshot["candidates"],
            final_checkpoint=snapshot["final_checkpoint"],
            threshold=0.05,
        )

    failed = copy.deepcopy(raw)
    failed["candidates"][-1]["scale"]["relative_l2"] = 0.06
    validated_failed = bundle._validate_raw_audit(
        failed,
        baseline=snapshot["baseline"],
        candidates=snapshot["candidates"],
        final_checkpoint=snapshot["final_checkpoint"],
        threshold=0.05,
    )
    assert validated_failed["passed"] is False


def test_published_payload_tamper_is_rejected_by_production_consumer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(tmp_path)
    args = _args(tmp_path, snapshot)
    _install_snapshot(monkeypatch, snapshot)
    bundle.generate_bundle(args)
    binding = _binding(args.output)
    analysis_path = args.output / "analysis.json"
    analysis_path.write_bytes(analysis_path.read_bytes() + b" ")

    with pytest.raises(
        bundle.publisher.ReleaseError,
        match="payload identity differs",
    ):
        bundle.publisher._authenticate_report_bundle(
            binding,
            project_root=tmp_path,
            label="drift bundle",
        )


@pytest.mark.parametrize("partial", [False, True])
def test_existing_complete_or_partial_output_is_never_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    partial: bool,
) -> None:
    snapshot = _snapshot(tmp_path)
    args = _args(tmp_path, snapshot)
    _install_snapshot(monkeypatch, snapshot)
    args.output.mkdir()
    sentinel = args.output / ("MANIFEST.json" if partial else "sentinel")
    sentinel.write_text("keep\n", encoding="utf-8")

    with pytest.raises(bundle.DriftBundleError, match="already exists"):
        bundle.generate_bundle(args)

    assert sentinel.read_text(encoding="utf-8") == "keep\n"
    assert {path.name for path in args.output.iterdir()} == {sentinel.name}


def test_midflight_input_change_and_output_race_leave_no_partial_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(tmp_path)
    args = _args(tmp_path, snapshot)
    monkeypatch.setattr(
        bundle,
        "_snapshot",
        lambda _args: copy.deepcopy(snapshot),
    )
    monkeypatch.setattr(
        bundle,
        "_reauthenticate_inputs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            bundle.DriftBundleError("inputs changed")
        ),
    )

    with pytest.raises(bundle.DriftBundleError, match="inputs changed"):
        bundle.generate_bundle(args)

    assert not args.output.exists()
    assert not list(tmp_path.glob(".drift-bundle.incomplete-*"))

    def create_racing_output(
        _args: Namespace,
        *,
        expected: dict[str, Any],
    ) -> None:
        assert expected
        args.output.mkdir()
        (args.output / "sentinel").write_text("racer\n", encoding="utf-8")

    monkeypatch.setattr(bundle, "_reauthenticate_inputs", create_racing_output)
    with pytest.raises(bundle.DriftBundleError, match="appeared"):
        bundle.generate_bundle(args)

    assert (args.output / "sentinel").read_text(encoding="utf-8") == "racer\n"
    assert not list(tmp_path.glob(".drift-bundle.incomplete-*"))


def test_drift_bundle_fsyncs_staging_before_rename_and_parent_after(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(tmp_path)
    args = _args(tmp_path, snapshot)
    _install_snapshot(monkeypatch, snapshot)
    real_fsync = os.fsync
    fsynced: list[Path] = []

    def track_fsync(descriptor: int) -> None:
        fsynced.append(
            Path(os.readlink(f"/proc/self/fd/{descriptor}")).resolve()
        )
        real_fsync(descriptor)

    monkeypatch.setattr(bundle.os, "fsync", track_fsync)
    real_rename = bundle.publisher._rename_directory_noreplace

    def checked_rename(source: Path, destination: Path) -> None:
        assert source.resolve() in fsynced
        real_rename(source, destination)

    monkeypatch.setattr(
        bundle.publisher,
        "_rename_directory_noreplace",
        checked_rename,
    )
    bundle.generate_bundle(args)
    assert args.output.parent.resolve() in fsynced
