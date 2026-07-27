#!/usr/bin/env python3
"""Generate the authenticated v4 13M calibration drift report bundle.

This is a reporting-only CPU command.  It invokes the existing
``audit_dense_checkpoint_drift.py`` implementation for already completed
checkpoints, fully authenticates every checkpoint and the formal closure, and
atomically emits:

* ``analysis.json`` with the raw tensor audit plus a source-bound release gate;
* ``REPORT.zh-CN.md`` with checkpoint, Adapter, scale, and 5% gate tables;
* ``MANIFEST.json`` authenticating both payloads; and
* ``COMPLETE`` authenticating the manifest.

It never initializes CUDA, creates an optimizer, starts calibration, or starts
formal training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from twen.utils import sha256_file

SCHEMA_VERSION = 1
BUNDLE_KIND = "twen_dense_optimizer_drift_audit_bundle"
ANALYSIS_KIND = "twen_dense_checkpoint_trainable_drift_audit"
ANALYSIS_NAME = "analysis.json"
REPORT_NAME = "REPORT.zh-CN.md"
MANIFEST_NAME = "MANIFEST.json"
COMPLETE_NAME = "COMPLETE"
ROOT = Path(__file__).resolve().parents[1]


class DriftBundleError(ValueError):
    """The drift bundle cannot be safely generated."""


def _load_script(filename: str, module_name: str) -> ModuleType:
    path = Path(__file__).resolve().with_name(filename)
    source_before = path.read_bytes()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise DriftBundleError(f"cannot load reporting dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if path.read_bytes() != source_before:
        raise DriftBundleError(f"reporting dependency changed while loading: {path}")
    return module


publisher = _load_script(
    "publish_v4_250m_release.py",
    "_twen_v4_release_publisher_for_drift_bundle",
)
drift_auditor = _load_script(
    "audit_dense_checkpoint_drift.py",
    "_twen_v4_calibration_drift_auditor",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=Path,
        action="append",
        required=True,
        help=(
            "Completed calibration checkpoint; repeat in strictly increasing "
            "global-step order and include the final milestone"
        ),
    )
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise DriftBundleError(f"evidence file is missing or a symlink: {resolved}")
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _checkpoint_binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "manifest_sha256": sha256_file(resolved / "manifest.json"),
        "complete_sha256": sha256_file(resolved / "COMPLETE"),
    }


def _authenticate_checkpoint(path: Path, *, project_root: Path, label: str) -> dict[str, Any]:
    try:
        return publisher._authenticate_checkpoint_binding(
            _checkpoint_binding(path),
            project_root=project_root,
            label=label,
        )
    except (publisher.ReleaseError, OSError, ValueError) as exc:
        raise DriftBundleError(f"{label} authentication failed: {exc}") from exc


def _authenticate_closure(path: Path) -> dict[str, Any]:
    try:
        closure = publisher._authenticate_closure_structure(path)
        closure["governed"] = publisher._authenticate_governed_closure_gates(
            closure
        )
    except (publisher.ReleaseError, OSError, ValueError) as exc:
        raise DriftBundleError(f"formal closure authentication failed: {exc}") from exc
    return closure


def _finite(value: Any, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DriftBundleError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise DriftBundleError(f"{label} must be finite")
    return result


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DriftBundleError(f"{label} must be an object")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DriftBundleError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _public_checkpoint(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _mapping(row.get("metadata"), label="checkpoint metadata")
    return {
        "path": row["path"],
        "manifest_sha256": row["manifest_sha256"],
        "complete_sha256": row["complete_sha256"],
        "global_step": metadata.get("global_step"),
        "committed_tokens": metadata.get("committed_tokens"),
        "kind": metadata.get("kind"),
        "tag": metadata.get("tag"),
    }


def _validate_checkpoint_lineage(
    closure: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    final_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    readiness = _mapping(closure.get("readiness"), label="closed readiness")
    gate = _mapping(readiness.get("calibration_gate"), label="calibration gate")
    required_fork = _mapping(
        gate.get("required_fork_checkpoint"),
        label="calibration required fork",
    )
    project_root = closure["project_root"]
    required_fork_path = publisher._resolve_path(
        required_fork.get("path"),
        base=project_root,
        label="calibration required fork path",
    )
    if (
        Path(str(baseline.get("path"))).resolve() != required_fork_path
        or baseline.get("complete_sha256") != required_fork.get("complete_sha256")
    ):
        raise DriftBundleError("drift baseline is not the required v3-final fork")
    gate_config = _mapping(gate.get("config"), label="calibration gate config")
    config_path = publisher._resolve_path(
        gate_config.get("path"),
        base=project_root,
        label="calibration config path",
    )
    if gate_config.get("sha256") != sha256_file(config_path):
        raise DriftBundleError("calibration config identity differs from closure")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DriftBundleError(f"cannot read calibration config: {exc}") from exc
    config = dict(_mapping(config, label="calibration config"))
    data = _mapping(config.get("data"), label="calibration config.data")
    required_candidates = _mapping(
        gate.get("required_candidate_checkpoints"),
        label="required calibration candidates",
    )
    required_steps_raw = required_candidates.get("global_steps")
    if (
        not isinstance(required_steps_raw, list)
        or required_candidates.get("final_milestone_required") is not True
    ):
        raise DriftBundleError("calibration candidate checkpoint policy is invalid")
    required_steps = {
        _integer(step, label="required candidate step", minimum=1)
        for step in required_steps_raw
    }
    steps: list[int] = []
    previous_tokens = -1
    for index, row in enumerate(candidates):
        metadata = _mapping(
            row.get("metadata"),
            label=f"candidate {index} metadata",
        )
        extra = _mapping(metadata.get("extra"), label=f"candidate {index} extra")
        source_mix = _mapping(
            extra.get("source_mix"),
            label=f"candidate {index} source_mix",
        )
        step = _integer(
            metadata.get("global_step"),
            label=f"candidate {index} global_step",
            minimum=1,
        )
        tokens = _integer(
            metadata.get("committed_tokens"),
            label=f"candidate {index} committed_tokens",
            minimum=1,
        )
        if (
            step in steps
            or (steps and step <= steps[-1])
            or tokens <= previous_tokens
            or metadata.get("run_id") != config.get("run_id")
            or extra.get("data_manifest_sha256") != data.get("manifest_sha256")
            or source_mix.get("source_map_sha256") != data.get("source_map_sha256")
        ):
            raise DriftBundleError("calibration candidate checkpoint lineage/order differs")
        steps.append(step)
        previous_tokens = tokens
    if not required_steps.issubset(steps):
        raise DriftBundleError("drift candidates omit a required calibration checkpoint")
    if (
        not candidates
        or candidates[-1]["path"] != final_checkpoint["path"]
        or candidates[-1]["manifest_sha256"]
        != final_checkpoint["manifest_sha256"]
        or candidates[-1]["complete_sha256"]
        != final_checkpoint["complete_sha256"]
    ):
        raise DriftBundleError("final checkpoint is not the last drift candidate")
    final_metadata = _mapping(
        final_checkpoint.get("metadata"),
        label="final checkpoint metadata",
    )
    if (
        final_metadata.get("kind") != "milestone"
        or final_metadata.get("tag") != "complete"
    ):
        raise DriftBundleError("final drift checkpoint is not the complete milestone")
    threshold = _finite(
        _mapping(
            gate.get("hard_thresholds"),
            label="calibration hard thresholds",
        ).get("final_scale_relative_l2_lte"),
        label="final scale relative-L2 threshold",
        minimum=0.0,
    )
    return {
        "calibration_config": _identity(config_path),
        "candidate_global_steps": steps,
        "final_scale_relative_l2_threshold": threshold,
    }


def _validate_raw_audit(
    raw: Any,
    *,
    baseline: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    final_checkpoint: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any]:
    audit = dict(_mapping(raw, label="raw drift audit"))
    execution = _mapping(audit.get("execution"), label="raw drift execution")
    if (
        audit.get("schema_version") != SCHEMA_VERSION
        or audit.get("kind") != ANALYSIS_KIND
        or execution.get("device") != "cpu"
        or execution.get("cuda_initialized") is not False
        or execution.get("model_built") is not False
        or execution.get("optimizer_created") is not False
    ):
        raise DriftBundleError("raw drift audit execution contract is invalid")
    raw_baseline = _mapping(audit.get("baseline"), label="raw drift baseline")
    if any(
        raw_baseline.get(field) != baseline.get(field)
        for field in ("path", "manifest_sha256", "complete_sha256")
    ):
        raise DriftBundleError("raw drift audit binds another baseline")
    raw_candidates = audit.get("candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != len(candidates):
        raise DriftBundleError("raw drift candidate inventory differs")
    rows: list[dict[str, Any]] = []
    final_scale_relative_l2: float | None = None
    for index, (raw_row, authenticated) in enumerate(
        zip(raw_candidates, candidates, strict=True)
    ):
        row = _mapping(raw_row, label=f"raw drift candidate {index}")
        if any(
            row.get(field) != authenticated.get(field)
            for field in ("path", "manifest_sha256", "complete_sha256")
        ):
            raise DriftBundleError(f"raw drift candidate {index} identity differs")
        adapter = _mapping(row.get("adapter"), label=f"candidate {index} adapter")
        scale = _mapping(row.get("scale"), label=f"candidate {index} scale")
        adapter_relative = _finite(
            adapter.get("relative_l2"),
            label=f"candidate {index} adapter relative-L2",
            minimum=0.0,
        )
        scale_relative = _finite(
            scale.get("relative_l2"),
            label=f"candidate {index} scale relative-L2",
            minimum=0.0,
        )
        public = _public_checkpoint(authenticated)
        rows.append(
            {
                **public,
                "adapter_relative_l2": adapter_relative,
                "scale_relative_l2": scale_relative,
            }
        )
        if authenticated["path"] == final_checkpoint["path"]:
            final_scale_relative_l2 = scale_relative
    if final_scale_relative_l2 is None:
        raise DriftBundleError("raw drift audit has no final checkpoint row")
    passed = final_scale_relative_l2 <= threshold
    audit["release_gate"] = {
        "final_checkpoint": _public_checkpoint(final_checkpoint),
        "final_scale_relative_l2": final_scale_relative_l2,
        "final_scale_relative_l2_lte": threshold,
        "passed": passed,
        "authorizes_training": False,
    }
    return {
        "analysis": audit,
        "rows": rows,
        "final_scale_relative_l2": final_scale_relative_l2,
        "threshold": threshold,
        "passed": passed,
    }


def _snapshot(args: argparse.Namespace) -> dict[str, Any]:
    closure = _authenticate_closure(args.closure)
    project_root = closure["project_root"]
    gate = _mapping(
        closure["readiness"].get("calibration_gate"),
        label="calibration gate",
    )
    required_fork = _mapping(
        gate.get("required_fork_checkpoint"),
        label="calibration required fork",
    )
    baseline_path = publisher._resolve_path(
        required_fork.get("path"),
        base=project_root,
        label="calibration required fork path",
    )
    baseline = _authenticate_checkpoint(
        baseline_path,
        project_root=project_root,
        label="v3-final drift baseline",
    )
    if not args.candidate:
        raise DriftBundleError("at least one drift candidate is required")
    candidates = [
        _authenticate_checkpoint(
            path,
            project_root=project_root,
            label=f"drift candidate {index}",
        )
        for index, path in enumerate(args.candidate)
    ]
    final_checkpoint = _authenticate_checkpoint(
        args.final_checkpoint,
        project_root=project_root,
        label="final drift checkpoint",
    )
    lineage = _validate_checkpoint_lineage(
        closure,
        baseline=baseline,
        candidates=candidates,
        final_checkpoint=final_checkpoint,
    )
    try:
        raw = drift_auditor.audit(
            Path(baseline["path"]),
            [Path(row["path"]) for row in candidates],
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise DriftBundleError(f"CPU drift audit failed: {exc}") from exc
    validated = _validate_raw_audit(
        raw,
        baseline=baseline,
        candidates=candidates,
        final_checkpoint=final_checkpoint,
        threshold=lineage["final_scale_relative_l2_threshold"],
    )
    return {
        "closure": closure,
        "baseline": baseline,
        "candidates": candidates,
        "final_checkpoint": final_checkpoint,
        "lineage": lineage,
        **validated,
        "auditor": _identity(ROOT / "scripts/audit_dense_checkpoint_drift.py"),
        "bundler": _identity(Path(__file__).resolve()),
    }


def _input_contract(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "formal_closure": copy.deepcopy(snapshot["closure"]["binding"]),
        "calibration_config": copy.deepcopy(
            snapshot["lineage"]["calibration_config"]
        ),
        "baseline": _public_checkpoint(snapshot["baseline"]),
        "candidates": [
            _public_checkpoint(row) for row in snapshot["candidates"]
        ],
        "final_checkpoint": _public_checkpoint(snapshot["final_checkpoint"]),
        "auditor": copy.deepcopy(snapshot["auditor"]),
        "bundler": copy.deepcopy(snapshot["bundler"]),
    }


def _reauthenticate_inputs(
    args: argparse.Namespace,
    *,
    expected: Mapping[str, Any],
) -> None:
    closure = _authenticate_closure(args.closure)
    project_root = closure["project_root"]
    baseline = _authenticate_checkpoint(
        Path(str(expected["baseline"]["path"])),
        project_root=project_root,
        label="v3-final drift baseline",
    )
    candidates = [
        _authenticate_checkpoint(
            path,
            project_root=project_root,
            label=f"drift candidate {index}",
        )
        for index, path in enumerate(args.candidate)
    ]
    final_checkpoint = _authenticate_checkpoint(
        args.final_checkpoint,
        project_root=project_root,
        label="final drift checkpoint",
    )
    current = {
        "formal_closure": copy.deepcopy(closure["binding"]),
        "calibration_config": _identity(
            Path(str(expected["calibration_config"]["path"]))
        ),
        "baseline": _public_checkpoint(baseline),
        "candidates": [_public_checkpoint(row) for row in candidates],
        "final_checkpoint": _public_checkpoint(final_checkpoint),
        "auditor": _identity(ROOT / "scripts/audit_dense_checkpoint_drift.py"),
        "bundler": _identity(Path(__file__).resolve()),
    }
    if _canonical_sha256(current) != _canonical_sha256(expected):
        raise DriftBundleError("drift bundle inputs changed during publication")


def _report(snapshot: Mapping[str, Any]) -> str:
    gate = "PASS" if snapshot["passed"] else "FAIL"
    rows = [
        "| step | checkpoint | Adapter relative-L2 | scale relative-L2 |",
        "|---:|---|---:|---:|",
    ]
    for row in snapshot["rows"]:
        rows.append(
            "| "
            f"{row['global_step']} | `{row['path']}` | "
            f"{row['adapter_relative_l2']:.8f} | "
            f"{row['scale_relative_l2']:.8f} |"
        )
    inputs = _input_contract(snapshot)
    return "\n".join(
        [
            "# v4 13M calibration Adapter/scale 漂移审计",
            "",
            "本报告只分析已完成 checkpoint; 执行设备为 CPU, 未构建模型、未创建优化器、未启动训练。",
            "",
            *rows,
            "",
            "## 5% scale gate",
            "",
            (
                f"- 末 checkpoint scale relative-L2: "
                f"`{snapshot['final_scale_relative_l2']:.8f}`"
            ),
            f"- 上限: `{snapshot['threshold']:.8f}`",
            f"- 结论: **{gate}**",
            "",
            "## 输入身份",
            "",
            f"- formal closure MANIFEST SHA256: `{inputs['formal_closure']['manifest_sha256']}`",
            f"- calibration config SHA256: `{inputs['calibration_config']['sha256']}`",
            f"- v3 baseline manifest SHA256: `{inputs['baseline']['manifest_sha256']}`",
            f"- final checkpoint manifest SHA256: `{inputs['final_checkpoint']['manifest_sha256']}`",
            f"- drift auditor SHA256: `{inputs['auditor']['sha256']}`",
            f"- bundle producer SHA256: `{inputs['bundler']['sha256']}`",
            "",
        ]
    )


def _write_bytes(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes(
        path,
        (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )


def generate_bundle(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise DriftBundleError(f"drift bundle output already exists: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise DriftBundleError(
            f"drift bundle parent must already be a real directory: {output.parent}"
        )
    snapshot = _snapshot(args)
    input_contract = _input_contract(snapshot)
    input_fingerprint = _canonical_sha256(input_contract)
    with publisher._directory_lock(output.parent):
        if output.exists():
            raise DriftBundleError(f"drift bundle output already exists: {output}")
        work = Path(
            tempfile.mkdtemp(
                prefix=f".{output.name}.incomplete-",
                dir=output.parent,
            )
        )
        try:
            analysis_path = work / ANALYSIS_NAME
            report_path = work / REPORT_NAME
            manifest_path = work / MANIFEST_NAME
            complete_path = work / COMPLETE_NAME
            _write_json(analysis_path, snapshot["analysis"])
            _write_bytes(report_path, _report(snapshot).encode("utf-8"))
            files = {
                path.name: {
                    "path": path.name,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in (analysis_path, report_path)
            }
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "kind": BUNDLE_KIND,
                "input_fingerprint": input_fingerprint,
                "inputs": input_contract,
                "measurement_script": copy.deepcopy(snapshot["auditor"]),
                "bundle_producer": copy.deepcopy(snapshot["bundler"]),
                "release_gate": copy.deepcopy(
                    snapshot["analysis"]["release_gate"]
                ),
                "files": files,
                "passed": snapshot["passed"],
                "authorizes_training": False,
                "training_started": False,
            }
            _write_json(manifest_path, manifest)
            _write_bytes(
                complete_path,
                (sha256_file(manifest_path) + "\n").encode("ascii"),
            )
            _reauthenticate_inputs(args, expected=input_contract)
            work_fd = os.open(work, os.O_RDONLY)
            try:
                os.fsync(work_fd)
            finally:
                os.close(work_fd)
            if output.exists():
                raise DriftBundleError(
                    f"drift bundle output appeared during publication: {output}"
                )
            try:
                publisher._rename_directory_noreplace(work, output)
            except publisher.ReleaseError as exc:
                raise DriftBundleError(str(exc)) from exc
            parent_fd = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
    return {
        "output": str(output),
        "analysis": str(output / ANALYSIS_NAME),
        "report": str(output / REPORT_NAME),
        "manifest": str(output / MANIFEST_NAME),
        "complete": str(output / COMPLETE_NAME),
        "input_fingerprint": input_fingerprint,
        "passed": snapshot["passed"],
        "authorizes_training": False,
        "training_started": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = generate_bundle(args)
    except (
        DriftBundleError,
        publisher.ReleaseError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
