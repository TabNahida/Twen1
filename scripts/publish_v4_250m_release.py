#!/usr/bin/env python3
"""Plan or atomically publish the formal v4 250M release.

``plan`` is strictly read-only.  It authenticates the closed data/formal
evidence bundle and a separately completed 13M calibration attestation, then
prints a canonical release fingerprint and the exact acknowledgements required
for publication.

``publish`` repeats the same authentication, requires both exact
acknowledgements, re-authenticates every input after staging, and atomically
installs one new release directory.  It never starts calibration, training, or
the Web service and never changes the Web launch allow-list.

The calibration attestation is deliberately explicit because calibration
evidence does not yet have a project-wide closure schema.  Its schema is
validated by :func:`_authenticate_calibration_attestation`: it binds the formal
closure, the immutable calibration config, three inventoried report bundles,
fully authenticated candidate/final checkpoints, and JSON-pointer-backed
observations for every hard gate.  A bare ``passed: true`` is never sufficient.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import yaml

from twen.config import load_train_config
from twen.governed import (
    FORMAL_V4_ATTESTED_CONTRACT,
    FORMAL_V4_DYNAMIC_CONFIG_IDENTITY_FIELDS,
    FORMAL_V4_NORMALIZED_CONFIG_SHA256,
    FORMAL_V4_WIKIPEDIA_LICENSE_CONTRACT,
    FORMAL_V4_WIKIPEDIA_LICENSE_FINGERPRINT,
    GovernedControllerError,
    _capacity_source_map_bindings,
    _formal_bindings,
    _normalized_formal_config_sha256,
    _release_gate_bindings,
    authenticate_checkpoint,
    build_governed_plan,
)
from twen.source_identity import twen_source_tree_sha256
from twen.utils import sha256_file

SCHEMA_VERSION = 1
PLAN_KIND = "twen_v4_250m_formal_release_plan"
RELEASE_KIND = "twen_v4_250m_formal_release"
RELEASE_BUNDLE_KIND = "twen_v4_250m_formal_release_bundle"
RELEASE_COMPLETE_KIND = "twen_v4_250m_formal_release_complete"
CALIBRATION_KIND = "twen_v4_13m_calibration_release_attestation"
CALIBRATION_COMPLETE_KIND = (
    "twen_v4_13m_calibration_release_attestation_complete"
)
CLOSURE_BUNDLE_KIND = "twen_v4_250m_formal_evidence_closure_bundle"
CLOSURE_COMPLETE_KIND = "twen_v4_250m_formal_evidence_closure_complete"
CLOSURE_KIND = "twen_v4_250m_formal_evidence_closure"
CAPACITY_KIND = "twen_v4_250m_capacity_attestation"
READINESS_KIND = "twen_v4_250m_pilot_readiness"
PENDING_FORMAL_CONFIG_ISSUES = frozenset(
    {
        "authenticated config retains missing or PENDING data identities",
        "config phase-disjointness identity differs from closed capacity evidence",
        "cooldown config identities differ from closed capacity evidence",
        "primary config identities differ from closed capacity evidence",
    }
)

PHASES = ("primary", "cooldown")
REPORT_BUNDLE_NAMES = (
    "training_report_bundle",
    "checkpoint_validation_bundle",
    "checkpoint_drift_audit_bundle",
)
CALIBRATION_REPORT_KINDS = {
    "training_report_bundle": "twen_dense_training_analysis_bundle",
    "checkpoint_validation_bundle": "twen_v4_checkpoint_validation_sweep_bundle",
    "checkpoint_drift_audit_bundle": "twen_dense_optimizer_drift_audit_bundle",
}
CALIBRATION_REPORT_PRODUCERS = {
    "training_report_bundle": "analyze_dense_training.py",
    "checkpoint_validation_bundle": "summarize_v4_checkpoint_validation.py",
    "checkpoint_drift_audit_bundle": (
        "generate_v4_calibration_drift_bundle.py"
    ),
}
CALIBRATION_REPORT_MANIFEST_KEYS = {
    "training_report_bundle": {
        "schema_version",
        "kind",
        "bundle_producer",
        "run_id",
        "source_run_dir",
        "source_inputs",
        "source_terminal_checkpoint",
        "source_fork_checkpoint",
        "release_gate",
        "files",
    },
    "checkpoint_validation_bundle": {
        "schema_version",
        "kind",
        "bundle_producer",
        "inputs_sha256",
        "inputs",
        "selection",
        "release_gate",
        "files",
    },
    "checkpoint_drift_audit_bundle": {
        "schema_version",
        "kind",
        "input_fingerprint",
        "inputs",
        "measurement_script",
        "bundle_producer",
        "release_gate",
        "files",
        "passed",
        "authorizes_training",
        "training_started",
    },
}
TRAINING_STATIC_INPUT_NAMES = (
    "metrics",
    "telemetry",
    "events",
    "resolved_config",
    "rank0_session",
    "latest",
)
# This is duplicated intentionally.  A validation producer edit must not be
# able to redefine the frozen baseline that the independent release consumer
# accepts.
FROZEN_V3_VALIDATION_CONTRACT = {
    "prepared_manifest_sha256": (
        "4d3877bfaf4e01551d41913e11d37db08c6097b513ac94ae4a63b8a640b68e7f"
    ),
    "prepared_dataset_fingerprint": (
        "6839d5aefb3b5b8f960e7da1b54bd40c2882bcb915954e727f2480252d4cdc79"
    ),
    "baseline_run_id": "base-dense-v3-500m",
    "baseline_checkpoint_state": {
        "global_step": 1912,
        "committed_tokens": 500_009_962,
        "kind": "milestone",
        "tag": "complete",
    },
    "baseline_manifest_sha256": (
        "97b9f59d968fef1aa3a9a0234cac542391151645650969283cd449ac0056dd3f"
    ),
    "baseline_complete_sha256": (
        "6d55deb1be53149338189ccc34fdebd180452145ccaedb645db3edd812407cad"
    ),
    "baseline_plan_sha256": (
        "e2f903a91d01f5d550582b4bb7733cad333829693e8462d6a0e789ef6e49f2e7"
    ),
}
CLAIM_NAMES = (
    "reference_epoch_max",
    "reused_sequences",
    "reused_tokens",
    "required_metrics_finite",
    "clip_fraction",
    "best_aggregate_nll",
    "final_aggregate_nll",
    "chinese_source_nll",
    "final_scale_relative_l2",
    "candidate_global_steps",
    "same_frozen_v3_validation_contract",
    "fork_checkpoint_complete_sha256",
)
CLAIM_EVIDENCE_POLICY = {
    "reference_epoch_max": ("training_report_bundle", "analysis.json"),
    "reused_sequences": ("training_report_bundle", "analysis.json"),
    "reused_tokens": ("training_report_bundle", "analysis.json"),
    "required_metrics_finite": ("training_report_bundle", "analysis.json"),
    "clip_fraction": ("training_report_bundle", "analysis.json"),
    "best_aggregate_nll": ("checkpoint_validation_bundle", "summary.json"),
    "final_aggregate_nll": ("checkpoint_validation_bundle", "summary.json"),
    "chinese_source_nll": ("checkpoint_validation_bundle", "summary.json"),
    "final_scale_relative_l2": (
        "checkpoint_drift_audit_bundle",
        "analysis.json",
    ),
    "candidate_global_steps": ("checkpoint_validation_bundle", "summary.json"),
    "same_frozen_v3_validation_contract": (
        "checkpoint_validation_bundle",
        "summary.json",
    ),
    "fork_checkpoint_complete_sha256": (
        "training_report_bundle",
        "analysis.json",
    ),
}
FINAL_CONFIG_NAME = "dense-v4-250m-pilot.yaml"
FINAL_READINESS_NAME = "readiness.json"
MANIFEST_NAME = "MANIFEST.json"
COMPLETE_NAME = "COMPLETE"
ROOT = Path(__file__).resolve().parents[1]


class ReleaseError(ValueError):
    """The release cannot be safely planned or published."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("plan", "publish"), required=True)
    parser.add_argument(
        "--closure",
        type=Path,
        required=True,
        help="Completed output directory from close_v4_250m_formal_evidence.py",
    )
    parser.add_argument(
        "--calibration-attestation",
        type=Path,
        required=True,
        help=(
            "Passing twen_v4_13m_calibration_release_attestation JSON; an "
            "authenticating sibling COMPLETE is mandatory"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New release directory; it must not already exist",
    )
    parser.add_argument(
        "--authorize-ack",
        help="publish only: exact `AUTHORIZE V4 <release-fingerprint>`",
    )
    parser.add_argument(
        "--wikipedia-license-ack",
        help=(
            "publish only: exact `ACCEPT V4 WIKIPEDIA LICENSE "
            "<contract-fingerprint>`"
        ),
    )
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


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseError(f"{label} must be a JSON object: {path}")
    return value


def _sha256_string(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReleaseError(f"{label} must be a lowercase 64-digit SHA256")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReleaseError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _finite(value: Any, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReleaseError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ReleaseError(f"{label} must be finite")
    return result


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseError(f"{label} must be an object")
    return value


def _resolve_path(
    value: Any,
    *,
    base: Path,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ReleaseError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ReleaseError(f"authenticated file is missing or a symlink: {resolved}")
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _same_identity(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    try:
        return (
            Path(str(actual.get("path"))).resolve()
            == Path(str(expected.get("path"))).resolve()
            and actual.get("size") == expected.get("size")
            and actual.get("sha256") == expected.get("sha256")
        )
    except (OSError, ValueError):
        return False


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        raise ReleaseError(
            f"{label} key inventory differs: expected {sorted(expected)}, "
            f"got {sorted(str(key) for key in value)}"
        )


def _contains_pending(value: Any) -> bool:
    if isinstance(value, str):
        return "PENDING" in value
    if isinstance(value, Mapping):
        return any(
            _contains_pending(key) or _contains_pending(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_pending(item) for item in value)
    return False


def _closure_binding(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(root),
        "manifest_sha256": sha256_file(root / MANIFEST_NAME),
        "complete_sha256": sha256_file(root / COMPLETE_NAME),
        "bundle_fingerprint": manifest["bundle_fingerprint"],
    }


def _authenticate_closure_structure(root: Path) -> dict[str, Any]:
    closure_root = root.expanduser().resolve()
    if not closure_root.is_dir() or closure_root.is_symlink():
        raise ReleaseError(f"formal closure directory is missing or a symlink: {closure_root}")
    expected_files = {
        "capacity-attestation.json",
        "readiness.json",
        MANIFEST_NAME,
        COMPLETE_NAME,
    }
    observed_files = {
        str(path.relative_to(closure_root)).replace("\\", "/")
        for path in closure_root.rglob("*")
        if path.is_file()
    }
    if observed_files != expected_files:
        raise ReleaseError("formal closure bundle file inventory differs")
    if any(path.is_symlink() for path in closure_root.rglob("*")):
        raise ReleaseError("formal closure bundle contains a symlink")

    manifest_path = closure_root / MANIFEST_NAME
    complete_path = closure_root / COMPLETE_NAME
    manifest = _read_json(manifest_path, label="formal closure MANIFEST")
    complete = _read_json(complete_path, label="formal closure COMPLETE")
    unsigned_manifest = {
        key: value for key, value in manifest.items() if key != "bundle_fingerprint"
    }
    bundle_fingerprint = _sha256_string(
        manifest.get("bundle_fingerprint"),
        label="formal closure bundle_fingerprint",
    )
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != CLOSURE_BUNDLE_KIND
        or manifest.get("launch_enabled") is not False
        or manifest.get("authorizes_training") is not False
        or manifest.get("training_started") is not False
        or _canonical_sha256(unsigned_manifest) != bundle_fingerprint
    ):
        raise ReleaseError("formal closure MANIFEST contract is invalid")
    if (
        complete.get("schema_version") != SCHEMA_VERSION
        or complete.get("kind") != CLOSURE_COMPLETE_KIND
        or complete.get("manifest") != MANIFEST_NAME
        or complete.get("manifest_sha256") != sha256_file(manifest_path)
        or complete.get("bundle_fingerprint") != bundle_fingerprint
        or complete.get("launch_enabled") is not False
        or complete.get("authorizes_training") is not False
        or complete.get("training_started") is not False
    ):
        raise ReleaseError("formal closure COMPLETE does not authenticate MANIFEST")

    files = _mapping(manifest.get("files"), label="formal closure MANIFEST.files")
    if set(files) != {"capacity-attestation.json", "readiness.json"}:
        raise ReleaseError("formal closure payload inventory differs")
    for relative, raw_identity in files.items():
        identity = _mapping(
            raw_identity,
            label=f"formal closure MANIFEST.files.{relative}",
        )
        path = closure_root / relative
        if (
            identity.get("path") != relative
            or identity.get("size") != path.stat().st_size
            or identity.get("sha256") != sha256_file(path)
        ):
            raise ReleaseError(f"formal closure payload identity differs: {relative}")

    closure = _mapping(manifest.get("closure"), label="formal closure identity")
    inputs = _mapping(manifest.get("inputs"), label="formal closure inputs")
    current_closer = ROOT / "scripts/close_v4_250m_formal_evidence.py"
    if (
        closure.get("schema_version") != SCHEMA_VERSION
        or closure.get("kind") != CLOSURE_KIND
        or closure.get("input_fingerprint") != _canonical_sha256(inputs)
        or closure.get("closure_source_sha256") != sha256_file(current_closer)
        or closure.get("twen_source_tree_sha256")
        != twen_source_tree_sha256(ROOT / "src/twen")
        or closure.get("launch_enabled") is not False
        or closure.get("authorizes_training") is not False
        or closure.get("training_started") is not False
    ):
        raise ReleaseError("formal closure source/input identity is stale")

    capacity_path = closure_root / "capacity-attestation.json"
    readiness_path = closure_root / "readiness.json"
    capacity = _read_json(capacity_path, label="closed capacity attestation")
    readiness = _read_json(readiness_path, label="closed readiness")
    capacity_unsigned = {
        key: value for key, value in capacity.items() if key != "attestation_fingerprint"
    }
    readiness_unsigned = {
        key: value for key, value in readiness.items() if key != "readiness_fingerprint"
    }
    if (
        capacity.get("schema_version") != SCHEMA_VERSION
        or capacity.get("kind") != CAPACITY_KIND
        or capacity.get("attestation_fingerprint")
        != _canonical_sha256(capacity_unsigned)
        or capacity.get("overall", {}).get("passed") is not True
        or capacity.get("launch_enabled") is not False
        or capacity.get("authorizes_training") is not False
        or capacity.get("training_started") is not False
    ):
        raise ReleaseError("closed capacity attestation is invalid")
    if (
        readiness.get("schema_version") != SCHEMA_VERSION
        or readiness.get("kind") != READINESS_KIND
        or readiness.get("readiness_fingerprint")
        != _canonical_sha256(readiness_unsigned)
        or readiness.get("launch_enabled") is not False
        or readiness.get("authorizes_training") is not False
        or readiness.get("training_started") is not False
        or readiness.get("blockers") == []
    ):
        raise ReleaseError("closed readiness is not fail-closed")
    if capacity.get("closure") != closure or readiness.get("closure") != closure:
        raise ReleaseError("closure payloads bind another closure identity")
    if _canonical_sha256(capacity.get("training_contract")) != _canonical_sha256(
        FORMAL_V4_ATTESTED_CONTRACT
    ):
        raise ReleaseError("closed capacity training contract differs")
    if _canonical_sha256(readiness.get("contract")) != _canonical_sha256(
        FORMAL_V4_ATTESTED_CONTRACT
    ):
        raise ReleaseError("closed readiness training contract differs")

    project_root = _resolve_path(
        readiness.get("project_root"),
        base=readiness_path.parent,
        label="closed readiness.project_root",
    )
    if project_root != ROOT.resolve():
        raise ReleaseError("closed readiness project_root differs from this checkout")
    controller = _mapping(
        readiness.get("governed_controller"),
        label="closed readiness.governed_controller",
    )
    controller_path = ROOT / "scripts/govern_v4_training.py"
    if (
        controller.get("implemented") is not True
        or _resolve_path(
            controller.get("path"),
            base=project_root,
            label="closed readiness.governed_controller.path",
        )
        != controller_path
        or controller.get("sha256") != sha256_file(controller_path)
        or controller.get("twen_source_tree_sha256")
        != twen_source_tree_sha256(ROOT / "src/twen")
    ):
        raise ReleaseError("closed readiness governed-controller identity is stale")

    capacity_binding = _mapping(
        readiness.get("capacity_attestation"),
        label="closed readiness.capacity_attestation",
    )
    actual_capacity_identity = _identity(capacity_path)
    if (
        not _same_identity(capacity_binding, actual_capacity_identity)
        or capacity_binding.get("attestation_fingerprint")
        != capacity.get("attestation_fingerprint")
        or capacity_binding.get("passed") is not True
        or capacity_binding.get("authorizes_training") is not False
    ):
        raise ReleaseError("closed readiness capacity binding differs")

    blocked_config_path = _resolve_path(
        readiness.get("config_path"),
        base=project_root,
        label="closed readiness.config_path",
    )
    if (
        not blocked_config_path.is_file()
        or readiness.get("config_sha256") != sha256_file(blocked_config_path)
        or capacity.get("config", {}).get("sha256")
        != readiness.get("config_sha256")
        or _resolve_path(
            capacity.get("config", {}).get("path"),
            base=project_root,
            label="closed capacity.config.path",
        )
        != blocked_config_path
        or capacity.get("config", {}).get("contains_pending_identity_sentinels")
        is not True
    ):
        raise ReleaseError("blocked config identity differs from closure")
    try:
        blocked_config = yaml.safe_load(blocked_config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ReleaseError(f"cannot read blocked formal config: {exc}") from exc
    blocked_config = dict(_mapping(blocked_config, label="blocked formal config"))
    data = _mapping(blocked_config.get("data"), label="blocked formal config.data")
    if any(
        not isinstance(data.get(field), str) or "PENDING" not in data[field]
        for field in FORMAL_V4_DYNAMIC_CONFIG_IDENTITY_FIELDS
    ):
        raise ReleaseError("blocked formal config does not retain every exact PENDING field")
    try:
        normalized = _normalized_formal_config_sha256(blocked_config)
    except GovernedControllerError as exc:
        raise ReleaseError(str(exc)) from exc
    if normalized != FORMAL_V4_NORMALIZED_CONFIG_SHA256:
        raise ReleaseError("blocked formal config semantics differ from source policy")

    calibration_gate = _mapping(
        readiness.get("calibration_gate"),
        label="closed readiness.calibration_gate",
    )
    semantic_gate = _mapping(
        readiness.get("chinese_semantic_quality_gate"),
        label="closed readiness.chinese_semantic_quality_gate",
    )
    formal_gate = _mapping(
        readiness.get("formal_validation_gate"),
        label="closed readiness.formal_validation_gate",
    )
    license_gate = _mapping(
        readiness.get("wikipedia_license_gate"),
        label="closed readiness.wikipedia_license_gate",
    )
    expected_license_ack = (
        f"ACCEPT V4 WIKIPEDIA LICENSE {FORMAL_V4_WIKIPEDIA_LICENSE_FINGERPRINT}"
    )
    if (
        calibration_gate.get("required") is not True
        or calibration_gate.get("passed") is not False
        or calibration_gate.get("authorizes_training") is not False
        or calibration_gate.get("observed") is not None
    ):
        raise ReleaseError("closed calibration gate is not exactly pending")
    if (
        semantic_gate.get("status")
        != "passed_authenticated_chinese_semantic_quality_audit"
        or semantic_gate.get("passed") is not True
        or semantic_gate.get("authorizes_training") is not False
        or not isinstance(semantic_gate.get("observed"), Mapping)
    ):
        raise ReleaseError("closed Chinese semantic gate has not passed")
    if (
        formal_gate.get("status")
        != "passed_authenticated_governed_disjointness_and_v3_baseline"
        or formal_gate.get("passed") is not True
        or formal_gate.get("authorizes_training") is not False
    ):
        raise ReleaseError("closed formal-validation gate has not passed")
    if (
        license_gate.get("required") is not True
        or license_gate.get("status") != "pending_explicit_user_acceptance"
        or _canonical_sha256(license_gate.get("contract"))
        != _canonical_sha256(FORMAL_V4_WIKIPEDIA_LICENSE_CONTRACT)
        or license_gate.get("contract_fingerprint")
        != FORMAL_V4_WIKIPEDIA_LICENSE_FINGERPRINT
        or license_gate.get("required_acknowledgement") != expected_license_ack
        or license_gate.get("observed_acknowledgement") is not None
        or license_gate.get("passed") is not False
        or license_gate.get("authorizes_training") is not False
    ):
        raise ReleaseError("closed Wikipedia license gate differs from pending contract")

    return {
        "root": closure_root,
        "manifest": manifest,
        "complete": complete,
        "closure": dict(closure),
        "binding": _closure_binding(closure_root, manifest),
        "capacity_path": capacity_path,
        "capacity": capacity,
        "readiness_path": readiness_path,
        "readiness": readiness,
        "project_root": project_root,
        "blocked_config_path": blocked_config_path,
        "blocked_config": blocked_config,
        "calibration_gate_contract_fingerprint": _canonical_sha256(calibration_gate),
    }


def _authenticate_governed_closure_gates(
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    readiness = snapshot["readiness"]
    project_root = snapshot["project_root"]
    try:
        capacity_bindings, capacity_issues = _capacity_source_map_bindings(
            readiness,
            project_root,
        )
        formal_bindings, formal_issues = _formal_bindings(readiness, project_root)
        release_probe = copy.deepcopy(readiness)
        release_probe["chinese_semantic_quality_gate"]["authorizes_training"] = True
        license_gate = release_probe["wikipedia_license_gate"]
        expected_license_ack = license_gate["required_acknowledgement"]
        license_gate.update(
            {
                "status": "accepted_explicit_user_acknowledgement",
                "observed_acknowledgement": expected_license_ack,
                "passed": True,
                "authorizes_training": True,
            }
        )
        release_bindings, release_issues = _release_gate_bindings(
            release_probe,
            project_root,
        )
    except (GovernedControllerError, KeyError, TypeError) as exc:
        raise ReleaseError(f"governed closure authentication failed: {exc}") from exc
    issues = [*capacity_issues, *formal_issues, *release_issues]
    if issues:
        raise ReleaseError(
            "governed closure authentication failed: " + "; ".join(sorted(set(issues)))
        )
    try:
        blocked_plan = build_governed_plan(snapshot["readiness_path"])
    except GovernedControllerError as exc:
        raise ReleaseError(
            f"closed readiness violates source-bound governed policy: {exc}"
        ) from exc
    actual_blocked_issues = set(blocked_plan.get("readiness_issues", []))
    allowed_blocked_issues = {
        str(item)
        for item in readiness.get("blockers", [])
        if isinstance(item, str) and item
    } | {
        "readiness launch_enabled is not true",
        "readiness authorizes_training is not true",
        "calibration_gate does not authorize training",
        "formal_validation_gate does not authorize training",
        (
            "wikipedia_license_gate does not contain the exact accepted "
            "acknowledgement"
        ),
        "chinese_semantic_quality_gate does not authorize training",
    } | set(PENDING_FORMAL_CONFIG_ISSUES)
    required_blocked_issues = {
        "readiness launch_enabled is not true",
        "readiness authorizes_training is not true",
        "calibration_gate does not authorize training",
        "formal_validation_gate does not authorize training",
        (
            "wikipedia_license_gate does not contain the exact accepted "
            "acknowledgement"
        ),
        "chinese_semantic_quality_gate does not authorize training",
    } | set(PENDING_FORMAL_CONFIG_ISSUES)
    if (
        not required_blocked_issues.issubset(actual_blocked_issues)
        or not actual_blocked_issues.issubset(allowed_blocked_issues)
        or blocked_plan.get("launch_enabled") is not False
    ):
        unexpected = sorted(actual_blocked_issues - allowed_blocked_issues)
        missing = sorted(required_blocked_issues - actual_blocked_issues)
        raise ReleaseError(
            "closed readiness has unexpected source-bound governed issues: "
            f"unexpected={unexpected}, missing_expected={missing}"
        )
    if set(capacity_bindings.get("phases", {})) != set(PHASES):
        raise ReleaseError("closed capacity does not bind primary and cooldown")
    if set(formal_bindings.get("validation_phases", {})) != set(PHASES):
        raise ReleaseError("formal validation does not bind primary and cooldown")
    release_quality = release_bindings.get("chinese_semantic_quality")
    release_license = release_bindings.get("wikipedia_license")
    if (
        not isinstance(release_quality, Mapping)
        or release_quality.get("passed") is not True
        or not isinstance(release_license, Mapping)
        or release_license.get("contract_fingerprint")
        != FORMAL_V4_WIKIPEDIA_LICENSE_FINGERPRINT
    ):
        raise ReleaseError("source-bound semantic/license gates did not authenticate")
    semantic_inputs = _mapping(
        release_quality.get("inputs"),
        label="source-bound semantic inputs",
    )
    capacity_phases = _mapping(
        capacity_bindings.get("phases"),
        label="closed capacity phases",
    )
    for phase in PHASES:
        semantic_input = _mapping(
            semantic_inputs.get(phase),
            label=f"source-bound semantic {phase} input",
        )
        capacity_phase = _mapping(
            capacity_phases.get(phase),
            label=f"closed capacity {phase}",
        )
        if (
            Path(str(semantic_input.get("path"))).resolve()
            != Path(
                str(capacity_phase.get("extracted_manifest_path"))
            ).resolve()
            or semantic_input.get("sha256")
            != capacity_phase.get("extracted_manifest_sha256")
            or semantic_input.get("corpus_fingerprint")
            != capacity_phase.get("extracted_corpus_fingerprint")
        ):
            raise ReleaseError(
                f"source-bound semantic {phase} input differs from closed "
                "capacity extracted identity"
            )
    return {
        "capacity": capacity_bindings,
        "formal": formal_bindings,
        "release": release_bindings,
    }


def _safe_bundle_path(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ReleaseError(f"{label} must be a non-empty relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or str(pure) in {".", ""}:
        raise ReleaseError(f"{label} is unsafe")
    path = (root / Path(*pure.parts)).resolve()
    if root.resolve() not in path.parents:
        raise ReleaseError(f"{label} escapes its bundle")
    return path


def _authenticate_complete(
    complete_path: Path,
    *,
    manifest_sha256: str,
    expected_kind: Any,
    label: str,
) -> None:
    payload = complete_path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        try:
            marker = payload.decode("ascii", errors="strict").strip()
        except UnicodeDecodeError as exc:
            raise ReleaseError(
                f"{label} COMPLETE is neither typed JSON nor an ASCII digest"
            ) from exc
        if marker != manifest_sha256:
            raise ReleaseError(
                f"{label} COMPLETE does not authenticate MANIFEST"
            ) from None
        if expected_kind is not None:
            raise ReleaseError(f"{label} requires a typed JSON COMPLETE") from None
        return
    if not isinstance(value, Mapping):
        raise ReleaseError(f"{label} COMPLETE must be an object")
    if (
        value.get("manifest_sha256") != manifest_sha256
        or (
            value.get("manifest") is not None
            and value.get("manifest") != MANIFEST_NAME
        )
        or (
            expected_kind is not None
            and value.get("kind") != expected_kind
        )
    ):
        raise ReleaseError(f"{label} COMPLETE does not authenticate MANIFEST")


def _load_reporting_script(
    filename: str,
    *,
    module_name: str,
    label: str,
) -> tuple[ModuleType, dict[str, Any]]:
    path = (ROOT / "scripts" / filename).resolve()
    identity_before = _identity(path)
    source_before = path.read_bytes()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ReleaseError(f"cannot load {label} producer: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise ReleaseError(f"cannot load {label} producer: {exc}") from exc
    if path.read_bytes() != source_before or not _same_identity(
        _identity(path),
        identity_before,
    ):
        raise ReleaseError(f"{label} producer changed while loading")
    return module, identity_before


def _report_name_for_kind(kind: Any) -> str:
    matches = [
        name
        for name, expected_kind in CALIBRATION_REPORT_KINDS.items()
        if kind == expected_kind
    ]
    if len(matches) != 1:
        raise ReleaseError("calibration report kind is not source-bound")
    return matches[0]


def _authenticate_report_bundle(
    raw_binding: Any,
    *,
    project_root: Path,
    label: str,
    report_name: str | None = None,
) -> dict[str, Any]:
    binding = _mapping(raw_binding, label=label)
    _require_exact_keys(
        binding,
        {
            "path",
            "manifest_sha256",
            "complete_sha256",
            "manifest_kind",
            "complete_kind",
        },
        label=label,
    )
    root = _resolve_path(binding.get("path"), base=project_root, label=f"{label}.path")
    if not root.is_dir() or root.is_symlink():
        raise ReleaseError(f"{label} directory is missing or a symlink")
    manifest_path = root / MANIFEST_NAME
    complete_path = root / COMPLETE_NAME
    if not manifest_path.is_file() or not complete_path.is_file():
        raise ReleaseError(f"{label} is incomplete")
    manifest_sha = sha256_file(manifest_path)
    complete_sha = sha256_file(complete_path)
    if (
        manifest_sha
        != _sha256_string(
            binding.get("manifest_sha256"),
            label=f"{label}.manifest_sha256",
        )
        or complete_sha
        != _sha256_string(
            binding.get("complete_sha256"),
            label=f"{label}.complete_sha256",
        )
    ):
        raise ReleaseError(f"{label} MANIFEST/COMPLETE identity differs")
    manifest = _read_json(manifest_path, label=f"{label} MANIFEST")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != binding.get("manifest_kind")
    ):
        raise ReleaseError(f"{label} MANIFEST kind/schema differs")
    resolved_report_name = report_name or _report_name_for_kind(
        manifest.get("kind")
    )
    if (
        resolved_report_name not in CALIBRATION_REPORT_KINDS
        or manifest.get("kind")
        != CALIBRATION_REPORT_KINDS[resolved_report_name]
    ):
        raise ReleaseError(f"{label} report-name/kind policy differs")
    _require_exact_keys(
        manifest,
        CALIBRATION_REPORT_MANIFEST_KEYS[resolved_report_name],
        label=f"{label} MANIFEST",
    )
    producer = _mapping(
        manifest.get("bundle_producer"),
        label=f"{label} MANIFEST.bundle_producer",
    )
    _require_exact_keys(
        producer,
        {"path", "size", "sha256"},
        label=f"{label} MANIFEST.bundle_producer",
    )
    expected_producer = _identity(
        ROOT
        / "scripts"
        / CALIBRATION_REPORT_PRODUCERS[resolved_report_name]
    )
    if not _same_identity(producer, expected_producer):
        raise ReleaseError(f"{label} producer identity is stale")
    complete_kind = binding.get("complete_kind")
    if complete_kind is not None and (
        not isinstance(complete_kind, str) or not complete_kind
    ):
        raise ReleaseError(f"{label}.complete_kind must be null or non-empty")
    _authenticate_complete(
        complete_path,
        manifest_sha256=manifest_sha,
        expected_kind=complete_kind,
        label=label,
    )
    files = _mapping(manifest.get("files"), label=f"{label} MANIFEST.files")
    if not files:
        raise ReleaseError(f"{label} MANIFEST has no payload inventory")
    payloads: dict[str, Path] = {}
    for relative, raw_identity in files.items():
        if not isinstance(relative, str):
            raise ReleaseError(f"{label} MANIFEST has a non-string path")
        identity = _mapping(
            raw_identity,
            label=f"{label} MANIFEST.files.{relative}",
        )
        _require_exact_keys(
            identity,
            {"path", "size", "sha256"},
            label=f"{label} MANIFEST.files.{relative}",
        )
        path = _safe_bundle_path(root, relative, label=f"{label}.files.{relative}")
        expected_path = identity.get("path")
        if expected_path != relative:
            raise ReleaseError(f"{label} inventory path differs: {relative}")
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size
            != _integer(
                identity.get("size"),
                label=f"{label}.files.{relative}.size",
            )
            or sha256_file(path)
            != _sha256_string(
                identity.get("sha256"),
                label=f"{label}.files.{relative}.sha256",
            )
        ):
            raise ReleaseError(f"{label} payload identity differs: {relative}")
        payloads[relative] = path
    observed = {
        str(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*")
        if path.is_file()
    }
    expected = set(payloads) | {MANIFEST_NAME, COMPLETE_NAME}
    if observed != expected or any(path.is_symlink() for path in root.rglob("*")):
        raise ReleaseError(f"{label} contains uninventoried files or symlinks")
    return {
        "path": str(root),
        "manifest_sha256": manifest_sha,
        "complete_sha256": complete_sha,
        "manifest_kind": binding["manifest_kind"],
        "complete_kind": complete_kind,
        "report_name": resolved_report_name,
        "manifest": manifest,
        "payloads": payloads,
    }


def _json_pointer(value: Any, pointer: Any, *, label: str) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ReleaseError(f"{label} must be a non-root RFC 6901 JSON pointer")
    current = value
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if token not in current:
                raise ReleaseError(f"{label} does not exist")
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit() or int(token) >= len(current):
                raise ReleaseError(f"{label} list index does not exist")
            current = current[int(token)]
        else:
            raise ReleaseError(f"{label} traverses a scalar")
    return current


def _authenticate_claims(
    raw_claims: Any,
    *,
    bundles: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    claims = _mapping(raw_claims, label="calibration claims")
    if set(claims) != set(CLAIM_NAMES):
        raise ReleaseError("calibration claim inventory differs")
    values: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for name in CLAIM_NAMES:
        claim = _mapping(claims[name], label=f"calibration claims.{name}")
        _require_exact_keys(claim, {"value", "evidence"}, label=f"claims.{name}")
        evidence = _mapping(
            claim.get("evidence"),
            label=f"calibration claims.{name}.evidence",
        )
        _require_exact_keys(
            evidence,
            {"bundle", "path", "json_pointer"},
            label=f"claims.{name}.evidence",
        )
        bundle_name = evidence.get("bundle")
        if bundle_name not in bundles:
            raise ReleaseError(f"claims.{name} references an unknown bundle")
        required_bundle, required_path = CLAIM_EVIDENCE_POLICY[name]
        if (
            bundle_name != required_bundle
            or evidence.get("path") != required_path
        ):
            raise ReleaseError(
                f"claims.{name} evidence location differs from source policy"
            )
        bundle = bundles[str(bundle_name)]
        relative = evidence.get("path")
        if not isinstance(relative, str) or relative not in bundle["payloads"]:
            raise ReleaseError(f"claims.{name} references an uninventoried payload")
        path = bundle["payloads"][relative]
        payload = _read_json(path, label=f"claims.{name} evidence")
        observed = _json_pointer(
            payload,
            evidence.get("json_pointer"),
            label=f"claims.{name}.json_pointer",
        )
        if _canonical_sha256(observed) != _canonical_sha256(claim.get("value")):
            raise ReleaseError(f"claims.{name} value differs from authenticated evidence")
        values[name] = copy.deepcopy(observed)
        sources[name] = dict(evidence)
    return {"values": values, "sources": sources}


def _authenticate_checkpoint_binding(
    raw_binding: Any,
    *,
    project_root: Path,
    label: str,
) -> dict[str, Any]:
    binding = _mapping(raw_binding, label=label)
    _require_exact_keys(
        binding,
        {"path", "manifest_sha256", "complete_sha256"},
        label=label,
    )
    path = _resolve_path(binding.get("path"), base=project_root, label=f"{label}.path")
    try:
        authenticated = authenticate_checkpoint(path)
    except (GovernedControllerError, OSError, ValueError) as exc:
        raise ReleaseError(f"{label} failed full checkpoint authentication: {exc}") from exc
    if (
        authenticated.get("manifest_sha256")
        != _sha256_string(
            binding.get("manifest_sha256"),
            label=f"{label}.manifest_sha256",
        )
        or authenticated.get("complete_sha256")
        != _sha256_string(
            binding.get("complete_sha256"),
            label=f"{label}.complete_sha256",
        )
    ):
        raise ReleaseError(f"{label} identity differs")
    metadata = _mapping(authenticated.get("metadata"), label=f"{label}.metadata")
    return {
        "path": str(path),
        "manifest_sha256": authenticated["manifest_sha256"],
        "complete_sha256": authenticated["complete_sha256"],
        "metadata": dict(metadata),
    }


def _checkpoint_binding_from_path(
    path: Path,
    *,
    project_root: Path,
    label: str,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return _authenticate_checkpoint_binding(
        {
            "path": str(resolved),
            "manifest_sha256": sha256_file(resolved / "manifest.json"),
            "complete_sha256": sha256_file(resolved / COMPLETE_NAME),
        },
        project_root=project_root,
        label=label,
    )


def _same_checkpoint_identity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    try:
        return (
            Path(str(left.get("path"))).resolve()
            == Path(str(right.get("path"))).resolve()
            and left.get("manifest_sha256") == right.get("manifest_sha256")
            and left.get("complete_sha256") == right.get("complete_sha256")
        )
    except (OSError, ValueError):
        return False


def _reauthenticate_file_identities(
    raw: Any,
    *,
    label: str,
) -> None:
    identities = _mapping(raw, label=label)
    if not identities:
        raise ReleaseError(f"{label} is empty")
    for name, raw_identity in identities.items():
        identity = _mapping(
            raw_identity,
            label=f"{label}.{name}",
        )
        _require_exact_keys(
            identity,
            {"path", "size", "sha256"},
            label=f"{label}.{name}",
        )
        current = _identity(
            _resolve_path(
                identity.get("path"),
                base=ROOT,
                label=f"{label}.{name}.path",
            )
        )
        if not _same_identity(identity, current):
            raise ReleaseError(f"{label}.{name} changed during verification")


def _training_manifest_contract(
    analysis: Mapping[str, Any],
    *,
    producer: Mapping[str, Any],
) -> dict[str, Any]:
    run = _mapping(analysis.get("run"), label="training analysis.run")
    inputs = _mapping(
        analysis.get("inputs"),
        label="training analysis.inputs",
    )
    terminal = _mapping(
        analysis.get("terminal_validation"),
        label="training analysis.terminal_validation",
    )
    gate = _mapping(
        analysis.get("release_gate"),
        label="training analysis.release_gate",
    )
    fork = _mapping(
        gate.get("fork_checkpoint"),
        label="training analysis.release_gate.fork_checkpoint",
    )
    manifest = _mapping(
        terminal.get("manifest"),
        label="training terminal manifest identity",
    )
    complete = _mapping(
        terminal.get("complete_marker"),
        label="training terminal COMPLETE identity",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CALIBRATION_REPORT_KINDS["training_report_bundle"],
        "bundle_producer": copy.deepcopy(dict(producer)),
        "run_id": run.get("run_id"),
        "source_run_dir": run.get("run_dir"),
        "source_inputs": {
            name: copy.deepcopy(inputs[name])
            for name in TRAINING_STATIC_INPUT_NAMES
        },
        "source_terminal_checkpoint": {
            "path": terminal.get("checkpoint"),
            "manifest_sha256": manifest.get("sha256"),
            "complete_sha256": complete.get("sha256"),
        },
        "source_fork_checkpoint": {
            "path": fork.get("path"),
            "manifest_sha256": fork.get("manifest_sha256"),
            "complete_sha256": fork.get("complete_sha256"),
        },
        "release_gate": copy.deepcopy(dict(gate)),
    }


def _recompute_training_evidence(
    bundle: Mapping[str, Any],
    *,
    project_root: Path,
    gate: Mapping[str, Any],
    calibration_config_canonical: Mapping[str, Any],
    final_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    module, producer_identity = _load_reporting_script(
        CALIBRATION_REPORT_PRODUCERS["training_report_bundle"],
        module_name="_twen_v4_release_training_recompute",
        label="training report",
    )
    manifest = _mapping(
        bundle.get("manifest"),
        label="training report MANIFEST",
    )
    analysis_path = bundle["payloads"].get("analysis.json")
    if analysis_path is None:
        raise ReleaseError("training report omits analysis.json")
    supplied = _read_json(analysis_path, label="training report analysis")
    run = _mapping(supplied.get("run"), label="training report analysis.run")
    run_dir = _resolve_path(
        run.get("run_dir"),
        base=project_root,
        label="training report run_dir",
    )
    if (
        not run_dir.is_dir()
        or run_dir.is_symlink()
        or (
            run_dir != project_root
            and project_root not in run_dir.parents
        )
    ):
        raise ReleaseError("training report run_dir is outside the project")
    try:
        recomputed = module.analyze_dense_training(run_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ReleaseError(f"training evidence recomputation failed: {exc}") from exc
    supplied_gate = _mapping(
        supplied.get("release_gate"),
        label="supplied training release gate",
    )
    recomputed_gate = _mapping(
        recomputed.get("release_gate"),
        label="recomputed training release gate",
    )
    if _canonical_sha256(recomputed_gate) != _canonical_sha256(supplied_gate):
        raise ReleaseError(
            "training release gate differs from analyze_dense_training recomputation"
        )
    supplied_inputs = _mapping(
        supplied.get("inputs"),
        label="supplied training inputs",
    )
    for name in TRAINING_STATIC_INPUT_NAMES:
        if _canonical_sha256(supplied_inputs.get(name)) != _canonical_sha256(
            recomputed["inputs"].get(name)
        ):
            raise ReleaseError(
                f"training static input identity differs after recomputation: {name}"
            )
    if (
        _canonical_sha256(supplied.get("run"))
        != _canonical_sha256(recomputed.get("run"))
        or _canonical_sha256(supplied.get("terminal_validation"))
        != _canonical_sha256(recomputed.get("terminal_validation"))
    ):
        raise ReleaseError(
            "training run/terminal facts differ from source recomputation"
        )
    expected_manifest = _training_manifest_contract(
        recomputed,
        producer=producer_identity,
    )
    observed_without_files = {
        key: value for key, value in manifest.items() if key != "files"
    }
    if _canonical_sha256(observed_without_files) != _canonical_sha256(
        expected_manifest
    ):
        raise ReleaseError("training report MANIFEST source bindings differ")

    resolved_config = _mapping(
        recomputed["inputs"].get("resolved_config"),
        label="training report resolved-config identity",
    )
    config_path = _resolve_path(
        resolved_config.get("path"),
        base=project_root,
        label="training resolved config path",
    )
    try:
        observed_config = load_train_config(config_path).canonical_dict()
    except (OSError, ValueError) as exc:
        raise ReleaseError(
            f"cannot parse training resolved config: {exc}"
        ) from exc
    if _canonical_sha256(observed_config) != _canonical_sha256(
        calibration_config_canonical
    ):
        raise ReleaseError(
            "training resolved config differs from the closed calibration config"
        )

    terminal = _mapping(
        recomputed.get("terminal_validation"),
        label="recomputed training terminal validation",
    )
    recomputed_terminal = {
        "path": terminal.get("checkpoint"),
        "manifest_sha256": _mapping(
            terminal.get("manifest"),
            label="recomputed terminal manifest",
        ).get("sha256"),
        "complete_sha256": _mapping(
            terminal.get("complete_marker"),
            label="recomputed terminal COMPLETE",
        ).get("sha256"),
    }
    if not _same_checkpoint_identity(recomputed_terminal, final_checkpoint):
        raise ReleaseError(
            "training report terminal checkpoint differs from supplied final checkpoint"
        )
    required_fork = _mapping(
        gate.get("required_fork_checkpoint"),
        label="calibration required fork checkpoint",
    )
    required_fork_path = _resolve_path(
        required_fork.get("path"),
        base=project_root,
        label="calibration required fork checkpoint path",
    )
    authenticated_fork = _checkpoint_binding_from_path(
        required_fork_path,
        project_root=project_root,
        label="calibration required fork checkpoint",
    )
    recomputed_fork = _mapping(
        recomputed_gate.get("fork_checkpoint"),
        label="recomputed training fork checkpoint",
    )
    if (
        required_fork.get("complete_sha256")
        != authenticated_fork["complete_sha256"]
        or not _same_checkpoint_identity(recomputed_fork, authenticated_fork)
    ):
        raise ReleaseError(
            "training report fork checkpoint differs from the closed v3 fork"
        )
    source_binding = _mapping(
        recomputed_gate.get("source_binding"),
        label="recomputed training source binding",
    )
    expected_source_binding = {
        "terminal_checkpoint_manifest_sha256": recomputed_terminal[
            "manifest_sha256"
        ],
        "terminal_checkpoint_complete_sha256": recomputed_terminal[
            "complete_sha256"
        ],
        "metrics_sha256": recomputed["inputs"]["metrics"]["sha256"],
        "events_sha256": recomputed["inputs"]["events"]["sha256"],
        "resolved_config_sha256": recomputed["inputs"]["resolved_config"][
            "sha256"
        ],
    }
    if _canonical_sha256(source_binding) != _canonical_sha256(
        expected_source_binding
    ):
        raise ReleaseError("training release gate source binding is incomplete")

    _reauthenticate_file_identities(
        {
            name: recomputed["inputs"][name]
            for name in TRAINING_STATIC_INPUT_NAMES
        },
        label="training recomputation inputs",
    )
    final_after = _checkpoint_binding_from_path(
        Path(str(final_checkpoint["path"])),
        project_root=project_root,
        label="training final checkpoint after recomputation",
    )
    fork_after = _checkpoint_binding_from_path(
        required_fork_path,
        project_root=project_root,
        label="training fork checkpoint after recomputation",
    )
    if (
        not _same_checkpoint_identity(final_after, final_checkpoint)
        or not _same_checkpoint_identity(fork_after, authenticated_fork)
        or not _same_identity(
            _identity(
                ROOT
                / "scripts"
                / CALIBRATION_REPORT_PRODUCERS[
                    "training_report_bundle"
                ]
            ),
            producer_identity,
        )
    ):
        raise ReleaseError("training evidence inputs changed during recomputation")
    return dict(recomputed_gate)


def _validation_manifest_inputs(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = _mapping(
        summary.get("baseline"),
        label="validation summary baseline",
    )
    candidates = summary.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ReleaseError("validation summary candidate inventory is empty")
    return {
        "prepared_manifest": copy.deepcopy(summary["prepared_manifest"]),
        "baseline_evaluation": copy.deepcopy(baseline["evaluation"]),
        "candidate_evaluations": {
            str(_mapping(row, label="validation candidate").get("label")): (
                copy.deepcopy(
                    _mapping(
                        row,
                        label="validation candidate",
                    )["evaluation"]
                )
            )
            for row in candidates
        },
    }


def _recompute_validation_evidence(
    bundle: Mapping[str, Any],
    *,
    project_root: Path,
    gate: Mapping[str, Any],
    candidate_checkpoints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    module, producer_identity = _load_reporting_script(
        CALIBRATION_REPORT_PRODUCERS["checkpoint_validation_bundle"],
        module_name="_twen_v4_release_validation_recompute",
        label="checkpoint validation report",
    )
    if _canonical_sha256(
        getattr(module, "FROZEN_V3_VALIDATION_CONTRACT", None)
    ) != _canonical_sha256(FROZEN_V3_VALIDATION_CONTRACT):
        raise ReleaseError("validation producer frozen-v3 contract differs")
    summary_path = bundle["payloads"].get("summary.json")
    if summary_path is None:
        raise ReleaseError("checkpoint validation report omits summary.json")
    supplied = _read_json(summary_path, label="checkpoint validation summary")
    prepared = _mapping(
        supplied.get("prepared_manifest"),
        label="checkpoint validation prepared manifest",
    )
    prepared_path = _resolve_path(
        prepared.get("path"),
        base=project_root,
        label="checkpoint validation prepared manifest path",
    )
    baseline = _mapping(
        supplied.get("baseline"),
        label="checkpoint validation baseline",
    )
    baseline_evaluation = _mapping(
        baseline.get("evaluation"),
        label="checkpoint validation baseline evaluation",
    )
    baseline_root = _resolve_path(
        baseline_evaluation.get("root"),
        base=project_root,
        label="checkpoint validation baseline root",
    )
    candidates_raw = supplied.get("candidates")
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise ReleaseError("checkpoint validation candidate inventory is empty")
    candidate_paths: list[tuple[str, Path]] = []
    for index, raw in enumerate(candidates_raw):
        row = _mapping(raw, label=f"checkpoint validation candidate {index}")
        label = row.get("label")
        if not isinstance(label, str) or not label:
            raise ReleaseError("checkpoint validation candidate label is invalid")
        evaluation = _mapping(
            row.get("evaluation"),
            label=f"checkpoint validation candidate {label} evaluation",
        )
        candidate_paths.append(
            (
                label,
                _resolve_path(
                    evaluation.get("root"),
                    base=project_root,
                    label=f"checkpoint validation candidate {label} root",
                ),
            )
        )
    baseline_label = baseline.get("label")
    if not isinstance(baseline_label, str) or not baseline_label:
        raise ReleaseError("checkpoint validation baseline label is invalid")
    try:
        recomputed = module.build_summary(
            prepared_manifest=prepared_path,
            baseline_path=baseline_root,
            baseline_label=baseline_label,
            candidate_paths=candidate_paths,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ReleaseError(
            f"checkpoint validation recomputation failed: {exc}"
        ) from exc
    if _canonical_sha256(recomputed) != _canonical_sha256(supplied):
        raise ReleaseError(
            "checkpoint validation report differs from build_summary recomputation"
        )
    manifest = _mapping(
        bundle.get("manifest"),
        label="checkpoint validation MANIFEST",
    )
    expected_inputs = _validation_manifest_inputs(recomputed)
    expected_manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": CALIBRATION_REPORT_KINDS[
            "checkpoint_validation_bundle"
        ],
        "bundle_producer": producer_identity,
        "inputs_sha256": recomputed["inputs_sha256"],
        "inputs": expected_inputs,
        "selection": recomputed["selection"],
        "release_gate": recomputed["release_gate"],
    }
    observed_without_files = {
        key: value for key, value in manifest.items() if key != "files"
    }
    if _canonical_sha256(observed_without_files) != _canonical_sha256(
        expected_manifest
    ):
        raise ReleaseError(
            "checkpoint validation MANIFEST source bindings differ"
        )
    frozen = FROZEN_V3_VALIDATION_CONTRACT
    if (
        prepared.get("sha256") != frozen["prepared_manifest_sha256"]
        or prepared.get("dataset_fingerprint")
        != frozen["prepared_dataset_fingerprint"]
        or baseline.get("run_id") != frozen["baseline_run_id"]
        or baseline.get("checkpoint_state")
        != frozen["baseline_checkpoint_state"]
        or baseline_evaluation.get("manifest", {}).get("sha256")
        != frozen["baseline_manifest_sha256"]
        or baseline_evaluation.get("complete", {}).get("sha256")
        != frozen["baseline_complete_sha256"]
        or baseline_evaluation.get("plan", {}).get("sha256")
        != frozen["baseline_plan_sha256"]
        or recomputed["release_gate"].get(
            "same_frozen_v3_validation_contract"
        )
        is not True
    ):
        raise ReleaseError(
            "checkpoint validation does not bind the formal frozen-v3 identity"
        )

    required_fork = _mapping(
        gate.get("required_fork_checkpoint"),
        label="calibration required fork checkpoint",
    )
    required_fork_path = _resolve_path(
        required_fork.get("path"),
        base=project_root,
        label="calibration required fork path",
    )
    baseline_checkpoint = _mapping(
        baseline.get("checkpoint"),
        label="validation baseline checkpoint binding",
    )
    authenticated_fork = _checkpoint_binding_from_path(
        required_fork_path,
        project_root=project_root,
        label="validation frozen-v3 checkpoint",
    )
    if (
        required_fork.get("complete_sha256")
        != authenticated_fork["complete_sha256"]
        or not _same_checkpoint_identity(
            baseline_checkpoint,
            authenticated_fork,
        )
    ):
        raise ReleaseError(
            "validation baseline evaluation checkpoint differs from the closed fork"
        )

    by_step = {
        _integer(
            row["metadata"].get("global_step"),
            label="supplied calibration candidate global_step",
            minimum=1,
        ): row
        for row in candidate_checkpoints
    }
    if len(by_step) != len(candidate_checkpoints):
        raise ReleaseError("supplied calibration candidate steps are not unique")
    seen_steps: list[int] = []
    for raw in candidates_raw:
        row = _mapping(raw, label="validation candidate")
        state = _mapping(
            row.get("checkpoint_state"),
            label="validation candidate checkpoint state",
        )
        step = _integer(
            state.get("global_step"),
            label="validation candidate global_step",
            minimum=1,
        )
        checkpoint = _mapping(
            row.get("checkpoint"),
            label="validation candidate checkpoint binding",
        )
        supplied_checkpoint = by_step.get(step)
        if supplied_checkpoint is None or not _same_checkpoint_identity(
            checkpoint,
            supplied_checkpoint,
        ):
            raise ReleaseError(
                "validation evaluation checkpoint differs from supplied candidates"
            )
        metadata = _mapping(
            supplied_checkpoint.get("metadata"),
            label="supplied validation candidate metadata",
        )
        expected_state = {
            "global_step": metadata.get("global_step"),
            "committed_tokens": metadata.get("committed_tokens"),
            "kind": metadata.get("kind"),
            "tag": metadata.get("tag"),
        }
        if _canonical_sha256(state) != _canonical_sha256(expected_state):
            raise ReleaseError(
                "validation evaluation checkpoint state differs from supplied candidate"
            )
        seen_steps.append(step)
    if len(seen_steps) != len(by_step) or sorted(seen_steps) != sorted(by_step):
        raise ReleaseError(
            "validation evaluation candidates do not exactly cover supplied checkpoints"
        )

    # A second full build re-authenticates every evaluation result and prepared
    # shard after the first computation, closing the input TOCTOU window.
    try:
        after = module.build_summary(
            prepared_manifest=prepared_path,
            baseline_path=baseline_root,
            baseline_label=baseline_label,
            candidate_paths=candidate_paths,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ReleaseError(
            f"checkpoint validation reauthentication failed: {exc}"
        ) from exc
    if (
        _canonical_sha256(after) != _canonical_sha256(recomputed)
        or not _same_identity(
            _identity(
                ROOT
                / "scripts"
                / CALIBRATION_REPORT_PRODUCERS[
                    "checkpoint_validation_bundle"
                ]
            ),
            producer_identity,
        )
    ):
        raise ReleaseError(
            "checkpoint validation inputs changed during recomputation"
        )
    return dict(
        _mapping(
            recomputed.get("release_gate"),
            label="recomputed checkpoint validation release gate",
        )
    )


def _recompute_drift_evidence(
    bundle: Mapping[str, Any],
    *,
    closure: Mapping[str, Any],
    candidate_checkpoints: Sequence[Mapping[str, Any]],
    final_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    module, producer_identity = _load_reporting_script(
        CALIBRATION_REPORT_PRODUCERS["checkpoint_drift_audit_bundle"],
        module_name="_twen_v4_release_drift_recompute",
        label="checkpoint drift report",
    )
    manifest = _mapping(
        bundle.get("manifest"),
        label="checkpoint drift MANIFEST",
    )
    analysis_path = bundle["payloads"].get("analysis.json")
    if analysis_path is None:
        raise ReleaseError("checkpoint drift report omits analysis.json")
    supplied = _read_json(analysis_path, label="checkpoint drift analysis")
    args = argparse.Namespace(
        closure=Path(str(closure["binding"]["path"])),
        candidate=[Path(str(row["path"])) for row in candidate_checkpoints],
        final_checkpoint=Path(str(final_checkpoint["path"])),
        output=Path(str(bundle["path"])),
    )
    try:
        snapshot = module._snapshot(args)
        input_contract = module._input_contract(snapshot)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ReleaseError(f"checkpoint drift recomputation failed: {exc}") from exc
    if _canonical_sha256(snapshot["analysis"]) != _canonical_sha256(supplied):
        raise ReleaseError(
            "checkpoint drift report differs from tensor-auditor recomputation"
        )
    if (
        _canonical_sha256(manifest.get("inputs"))
        != _canonical_sha256(input_contract)
        or manifest.get("input_fingerprint")
        != _canonical_sha256(input_contract)
        or _canonical_sha256(manifest.get("release_gate"))
        != _canonical_sha256(snapshot["analysis"]["release_gate"])
        or manifest.get("passed") is not snapshot["passed"]
        or manifest.get("authorizes_training") is not False
        or manifest.get("training_started") is not False
        or not _same_identity(
            _mapping(
                manifest.get("bundle_producer"),
                label="checkpoint drift bundle producer",
            ),
            producer_identity,
        )
        or _canonical_sha256(manifest.get("measurement_script"))
        != _canonical_sha256(snapshot["auditor"])
    ):
        raise ReleaseError("checkpoint drift MANIFEST source bindings differ")
    if _canonical_sha256(snapshot["closure"]["binding"]) != _canonical_sha256(
        closure["binding"]
    ):
        raise ReleaseError("checkpoint drift report binds another formal closure")
    recomputed_candidates = snapshot["candidates"]
    if len(recomputed_candidates) != len(candidate_checkpoints) or any(
        not _same_checkpoint_identity(recomputed, supplied_checkpoint)
        for recomputed, supplied_checkpoint in zip(
            recomputed_candidates,
            candidate_checkpoints,
            strict=True,
        )
    ):
        raise ReleaseError(
            "checkpoint drift candidates differ from supplied calibration checkpoints"
        )
    if not _same_checkpoint_identity(
        snapshot["final_checkpoint"],
        final_checkpoint,
    ):
        raise ReleaseError(
            "checkpoint drift final checkpoint differs from supplied final checkpoint"
        )
    try:
        module._reauthenticate_inputs(args, expected=input_contract)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ReleaseError(
            f"checkpoint drift input reauthentication failed: {exc}"
        ) from exc
    if not _same_identity(
        _identity(
            ROOT
            / "scripts"
            / CALIBRATION_REPORT_PRODUCERS[
                "checkpoint_drift_audit_bundle"
            ]
        ),
        producer_identity,
    ):
        raise ReleaseError("checkpoint drift producer changed during recomputation")
    return dict(
        _mapping(
            snapshot["analysis"].get("release_gate"),
            label="recomputed checkpoint drift release gate",
        )
    )


def _recompute_calibration_claims(
    bundles: Mapping[str, Mapping[str, Any]],
    *,
    closure: Mapping[str, Any],
    gate: Mapping[str, Any],
    calibration_config_canonical: Mapping[str, Any],
    candidate_checkpoints: Sequence[Mapping[str, Any]],
    final_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    project_root = closure["project_root"]
    training = _recompute_training_evidence(
        bundles["training_report_bundle"],
        project_root=project_root,
        gate=gate,
        calibration_config_canonical=calibration_config_canonical,
        final_checkpoint=final_checkpoint,
    )
    validation = _recompute_validation_evidence(
        bundles["checkpoint_validation_bundle"],
        project_root=project_root,
        gate=gate,
        candidate_checkpoints=candidate_checkpoints,
    )
    drift = _recompute_drift_evidence(
        bundles["checkpoint_drift_audit_bundle"],
        closure=closure,
        candidate_checkpoints=candidate_checkpoints,
        final_checkpoint=final_checkpoint,
    )
    return {
        "reference_epoch_max": training.get("reference_epoch_max"),
        "reused_sequences": training.get("reused_sequences"),
        "reused_tokens": training.get("reused_tokens"),
        "required_metrics_finite": training.get("required_metrics_finite"),
        "clip_fraction": training.get("clip_fraction"),
        "best_aggregate_nll": validation.get("best_aggregate_nll"),
        "final_aggregate_nll": validation.get("final_aggregate_nll"),
        "chinese_source_nll": validation.get("chinese_source_nll"),
        "final_scale_relative_l2": drift.get("final_scale_relative_l2"),
        "candidate_global_steps": validation.get("candidate_global_steps"),
        "same_frozen_v3_validation_contract": validation.get(
            "same_frozen_v3_validation_contract"
        ),
        "fork_checkpoint_complete_sha256": training.get(
            "fork_checkpoint_complete_sha256"
        ),
    }


def _evaluate_calibration_claims(
    claims: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    final_checkpoint: Mapping[str, Any],
    calibration_config: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = _mapping(gate.get("hard_thresholds"), label="calibration hard_thresholds")
    required_steps_raw = _mapping(
        gate.get("required_candidate_checkpoints"),
        label="calibration required_candidate_checkpoints",
    )
    if (
        required_steps_raw.get("final_milestone_required") is not True
        or required_steps_raw.get("same_frozen_v3_validation_contract") is not True
    ):
        raise ReleaseError("calibration candidate-checkpoint contract is invalid")
    raw_global_steps = required_steps_raw.get("global_steps")
    if not isinstance(raw_global_steps, list):
        raise ReleaseError("required calibration checkpoint steps are invalid")
    required_steps = [
        _integer(step, label="required calibration checkpoint step", minimum=1)
        for step in raw_global_steps
    ]
    observed_steps_raw = claims.get("candidate_global_steps")
    if not isinstance(observed_steps_raw, list):
        raise ReleaseError("candidate_global_steps must be a list")
    observed_steps = [
        _integer(step, label="candidate_global_steps item", minimum=1)
        for step in observed_steps_raw
    ]
    bound_steps = [
        _integer(
            row["metadata"].get("global_step"),
            label="candidate checkpoint global_step",
            minimum=1,
        )
        for row in checkpoint_rows
    ]
    if (
        observed_steps != sorted(set(observed_steps))
        or bound_steps != observed_steps
        or not set(required_steps).issubset(observed_steps)
    ):
        raise ReleaseError("candidate checkpoint steps do not satisfy the calibration contract")

    final_metadata = _mapping(
        final_checkpoint.get("metadata"),
        label="final calibration checkpoint metadata",
    )
    final_step = _integer(
        final_metadata.get("global_step"),
        label="final calibration checkpoint global_step",
        minimum=1,
    )
    if (
        final_step != observed_steps[-1]
        or final_metadata.get("kind") != "milestone"
        or final_metadata.get("tag") != "complete"
    ):
        raise ReleaseError("final calibration checkpoint is not the terminal milestone")

    optimizer = _mapping(
        calibration_config.get("optimizer"),
        label="calibration config.optimizer",
    )
    data = _mapping(calibration_config.get("data"), label="calibration config.data")
    max_tokens = _integer(
        optimizer.get("max_tokens"),
        label="calibration config.optimizer.max_tokens",
        minimum=1,
    )
    global_batch = _integer(
        data.get("global_batch_tokens"),
        label="calibration config.data.global_batch_tokens",
        minimum=1,
    )
    committed_tokens = _integer(
        final_metadata.get("committed_tokens"),
        label="final calibration checkpoint committed_tokens",
        minimum=1,
    )
    if not max_tokens <= committed_tokens < max_tokens + global_batch:
        raise ReleaseError("final calibration checkpoint token count is not a complete tail batch")
    if final_metadata.get("run_id") != calibration_config.get("run_id"):
        raise ReleaseError("final calibration checkpoint run_id differs from config")
    extra = _mapping(
        final_metadata.get("extra"),
        label="final calibration checkpoint extra",
    )
    if extra.get("data_manifest_sha256") != data.get("manifest_sha256"):
        raise ReleaseError("final calibration checkpoint data identity differs from config")
    source_mix = _mapping(
        extra.get("source_mix"),
        label="final calibration checkpoint source_mix",
    )
    if source_mix.get("source_map_sha256") != data.get("source_map_sha256"):
        raise ReleaseError("final calibration checkpoint source map differs from config")
    previous_tokens = -1
    for index, row in enumerate(checkpoint_rows):
        metadata = _mapping(
            row.get("metadata"),
            label=f"calibration candidate checkpoint {index} metadata",
        )
        candidate_extra = _mapping(
            metadata.get("extra"),
            label=f"calibration candidate checkpoint {index} extra",
        )
        candidate_mix = _mapping(
            candidate_extra.get("source_mix"),
            label=f"calibration candidate checkpoint {index} source_mix",
        )
        tokens = _integer(
            metadata.get("committed_tokens"),
            label=f"calibration candidate checkpoint {index} committed_tokens",
            minimum=1,
        )
        if (
            tokens <= previous_tokens
            or metadata.get("run_id") != calibration_config.get("run_id")
            or candidate_extra.get("data_manifest_sha256")
            != data.get("manifest_sha256")
            or candidate_mix.get("source_map_sha256")
            != data.get("source_map_sha256")
        ):
            raise ReleaseError(
                "calibration candidate checkpoint lineage/order differs from config"
            )
        previous_tokens = tokens

    expected_metrics = thresholds.get("all_required_metrics_finite")
    finite_metrics = claims.get("required_metrics_finite")
    if (
        not isinstance(expected_metrics, list)
        or not all(isinstance(item, str) and item for item in expected_metrics)
        or not isinstance(finite_metrics, Mapping)
        or set(finite_metrics) != set(expected_metrics)
        or any(value is not True for value in finite_metrics.values())
    ):
        raise ReleaseError("required calibration metric-finiteness gate failed")
    exact_integer_claims = (
        ("reference_epoch_max", "all_reference_epochs_eq"),
        ("reused_sequences", "reused_sequences_eq"),
        ("reused_tokens", "reused_tokens_eq"),
    )
    for claim_name, threshold_name in exact_integer_claims:
        if _integer(claims.get(claim_name), label=claim_name) != _integer(
            thresholds.get(threshold_name),
            label=f"hard_thresholds.{threshold_name}",
        ):
            raise ReleaseError(f"calibration hard gate failed: {claim_name}")
    if _finite(claims.get("clip_fraction"), label="clip_fraction", minimum=0.0) != _finite(
        thresholds.get("clip_fraction_eq"),
        label="hard_thresholds.clip_fraction_eq",
        minimum=0.0,
    ):
        raise ReleaseError("calibration hard gate failed: clip_fraction")
    upper_bound_claims = (
        ("best_aggregate_nll", "best_aggregate_nll_lte"),
        ("final_aggregate_nll", "final_aggregate_nll_lte"),
        ("chinese_source_nll", "chinese_source_nll_lte"),
        ("final_scale_relative_l2", "final_scale_relative_l2_lte"),
    )
    for claim_name, threshold_name in upper_bound_claims:
        observed = _finite(claims.get(claim_name), label=claim_name, minimum=0.0)
        limit = _finite(
            thresholds.get(threshold_name),
            label=f"hard_thresholds.{threshold_name}",
            minimum=0.0,
        )
        if observed > limit:
            raise ReleaseError(
                f"calibration hard gate failed: {claim_name} {observed} > {limit}"
            )
    if claims.get("same_frozen_v3_validation_contract") is not True:
        raise ReleaseError("calibration validation did not use the frozen v3 contract")
    if thresholds.get("authenticated_checkpoint_manifest_and_complete_required") is not True:
        raise ReleaseError("calibration checkpoint-authentication gate is not mandatory")
    required_fork = _mapping(
        gate.get("required_fork_checkpoint"),
        label="calibration required_fork_checkpoint",
    )
    if (
        claims.get("fork_checkpoint_complete_sha256")
        != required_fork.get("complete_sha256")
    ):
        raise ReleaseError("calibration fork checkpoint differs from the v3-final contract")
    return {
        "passed": True,
        "observed_steps": observed_steps,
        "final_global_step": final_step,
        "final_committed_tokens": committed_tokens,
        "hard_thresholds": copy.deepcopy(thresholds),
    }


def _authenticate_calibration_attestation(
    path: Path,
    *,
    closure: Mapping[str, Any],
) -> dict[str, Any]:
    attestation_path = path.expanduser().resolve()
    if not attestation_path.is_file() or attestation_path.is_symlink():
        raise ReleaseError(f"calibration attestation is missing or a symlink: {attestation_path}")
    complete_path = attestation_path.with_name(COMPLETE_NAME)
    if not complete_path.is_file() or complete_path.is_symlink():
        raise ReleaseError("calibration attestation has no sibling COMPLETE")
    attestation = _read_json(attestation_path, label="calibration attestation")
    complete = _read_json(complete_path, label="calibration attestation COMPLETE")
    unsigned = {
        key: value
        for key, value in attestation.items()
        if key != "attestation_fingerprint"
    }
    fingerprint = _sha256_string(
        attestation.get("attestation_fingerprint"),
        label="calibration attestation_fingerprint",
    )
    if (
        attestation.get("schema_version") != SCHEMA_VERSION
        or attestation.get("kind") != CALIBRATION_KIND
        or attestation.get("status")
        != "passed_authenticated_quality_gate_but_does_not_authorize_formal_training"
        or attestation.get("passed") is not True
        or attestation.get("authorizes_training") is not False
        or attestation.get("training_started") is not False
        or _canonical_sha256(unsigned) != fingerprint
    ):
        raise ReleaseError("calibration attestation contract/fingerprint is invalid")
    if (
        complete.get("schema_version") != SCHEMA_VERSION
        or complete.get("kind") != CALIBRATION_COMPLETE_KIND
        or complete.get("attestation") != attestation_path.name
        or complete.get("attestation_sha256") != sha256_file(attestation_path)
        or complete.get("attestation_fingerprint") != fingerprint
        or complete.get("passed") is not True
        or complete.get("authorizes_training") is not False
        or complete.get("training_started") is not False
    ):
        raise ReleaseError("calibration COMPLETE does not authenticate attestation")

    attestor = _mapping(
        attestation.get("attestor"),
        label="calibration attestation.attestor",
    )
    expected_attestor = ROOT / "scripts/attest_v4_13m_calibration_release.py"
    if (
        _resolve_path(
            attestor.get("path"),
            base=ROOT,
            label="calibration attestation.attestor.path",
        )
        != expected_attestor
        or not expected_attestor.is_file()
        or attestor.get("sha256") != sha256_file(expected_attestor)
    ):
        raise ReleaseError("calibration attestation producer identity is stale")
    if _canonical_sha256(attestation.get("formal_closure")) != _canonical_sha256(
        closure["binding"]
    ):
        raise ReleaseError("calibration attestation binds another formal closure")
    if (
        attestation.get("calibration_gate_contract_fingerprint")
        != closure["calibration_gate_contract_fingerprint"]
    ):
        raise ReleaseError("calibration attestation binds another calibration gate")
    project_root = closure["project_root"]
    gate = _mapping(
        closure["readiness"].get("calibration_gate"),
        label="closed calibration gate",
    )
    gate_config = _mapping(gate.get("config"), label="closed calibration gate.config")
    config_path = _resolve_path(
        gate_config.get("path"),
        base=project_root,
        label="closed calibration config path",
    )
    config_identity = _identity(config_path)
    attested_config = _mapping(
        attestation.get("calibration_config"),
        label="calibration attestation.calibration_config",
    )
    if (
        gate_config.get("sha256") != config_identity["sha256"]
        or not _same_identity(attested_config, config_identity)
    ):
        raise ReleaseError("calibration config identity differs from closed gate")
    try:
        calibration_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ReleaseError(f"cannot read calibration config: {exc}") from exc
    calibration_config = dict(
        _mapping(calibration_config, label="calibration config")
    )
    # Parsing through the production loader rejects malformed or semantically
    # incomplete training configs without starting any work.
    try:
        loaded_calibration_config = load_train_config(config_path)
    except (OSError, ValueError) as exc:
        raise ReleaseError(f"calibration config fails production parsing: {exc}") from exc
    calibration_data = _mapping(
        calibration_config.get("data"),
        label="calibration config.data",
    )
    governed_closure = _mapping(
        closure.get("governed"),
        label="formal closure governed bindings",
    )
    closed_capacity = _mapping(
        governed_closure.get("capacity"),
        label="formal closure governed capacity",
    )
    phases = _mapping(
        closed_capacity.get("phases"),
        label="formal closure governed capacity phases",
    )
    formal_primary = _mapping(
        phases.get("primary"),
        label="formal closure governed primary",
    )
    raw_capacity = _mapping(
        closure.get("capacity"),
        label="formal closure capacity attestation",
    )
    raw_stages = _mapping(
        raw_capacity.get("stages"),
        label="formal closure capacity stages",
    )
    raw_primary = _mapping(
        raw_stages.get("primary"),
        label="formal closure primary capacity",
    )
    prepared_capacity = _mapping(
        raw_primary.get("prepared_identity"),
        label="formal closure primary prepared capacity",
    )
    config_manifest_path = _resolve_path(
        calibration_data.get("manifest_path"),
        base=project_root,
        label="calibration config.data.manifest_path",
    )
    formal_manifest_path = _resolve_path(
        formal_primary.get("prepared_manifest_path"),
        base=project_root,
        label="formal primary prepared manifest",
    )
    calibration_max_tokens = _integer(
        _mapping(
            calibration_config.get("optimizer"),
            label="calibration config.optimizer",
        ).get("max_tokens"),
        label="calibration config.optimizer.max_tokens",
        minimum=1,
    )
    calibration_global_batch = _integer(
        calibration_data.get("global_batch_tokens"),
        label="calibration config.data.global_batch_tokens",
        minimum=1,
    )
    available_unique_tokens = _integer(
        prepared_capacity.get("available_unique_tokens"),
        label="formal primary available_unique_tokens",
        minimum=1,
    )
    required_without_wrap = (
        (calibration_max_tokens + calibration_global_batch - 1)
        // calibration_global_batch
    ) * calibration_global_batch
    if (
        config_manifest_path != formal_manifest_path
        or calibration_data.get("manifest_sha256")
        != formal_primary.get("prepared_manifest_sha256")
        or calibration_data.get("source_map_sha256")
        != formal_primary.get("source_map_sha256")
        or calibration_data.get("allow_corpus_reuse") is not False
        or available_unique_tokens < required_without_wrap
    ):
        raise ReleaseError(
            "calibration config is not bound to sufficient no-wrap formal primary data"
        )

    raw_evidence = _mapping(
        attestation.get("evidence"),
        label="calibration attestation.evidence",
    )
    if set(raw_evidence) != set(REPORT_BUNDLE_NAMES):
        raise ReleaseError("calibration evidence bundle inventory differs")
    for name in REPORT_BUNDLE_NAMES:
        binding = _mapping(
            raw_evidence[name],
            label=f"calibration attestation.evidence.{name}",
        )
        if (
            binding.get("manifest_kind") != CALIBRATION_REPORT_KINDS[name]
            or binding.get("complete_kind") is not None
        ):
            raise ReleaseError(
                f"calibration evidence.{name} differs from source-bound report policy"
            )
    bundles = {
        name: _authenticate_report_bundle(
            raw_evidence[name],
            project_root=project_root,
            label=f"calibration evidence.{name}",
            report_name=name,
        )
        for name in REPORT_BUNDLE_NAMES
    }

    candidates_raw = attestation.get("candidate_checkpoints")
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise ReleaseError("calibration candidate checkpoint inventory is empty")
    candidate_checkpoints = [
        _authenticate_checkpoint_binding(
            item,
            project_root=project_root,
            label=f"calibration candidate_checkpoints[{index}]",
        )
        for index, item in enumerate(candidates_raw)
    ]
    steps = [
        _integer(
            row["metadata"].get("global_step"),
            label="calibration candidate global_step",
            minimum=1,
        )
        for row in candidate_checkpoints
    ]
    if steps != sorted(set(steps)):
        raise ReleaseError("calibration candidate checkpoints must be step-sorted and unique")
    final_checkpoint = _authenticate_checkpoint_binding(
        attestation.get("final_checkpoint"),
        project_root=project_root,
        label="calibration final_checkpoint",
    )
    if (
        candidate_checkpoints[-1]["path"] != final_checkpoint["path"]
        or candidate_checkpoints[-1]["manifest_sha256"]
        != final_checkpoint["manifest_sha256"]
        or candidate_checkpoints[-1]["complete_sha256"]
        != final_checkpoint["complete_sha256"]
    ):
        raise ReleaseError("final calibration checkpoint is not the last candidate")
    recomputed_claims = _recompute_calibration_claims(
        bundles,
        closure=closure,
        gate=gate,
        calibration_config_canonical=(
            loaded_calibration_config.canonical_dict()
        ),
        candidate_checkpoints=candidate_checkpoints,
        final_checkpoint=final_checkpoint,
    )
    claim_result = _authenticate_claims(
        attestation.get("claims"),
        bundles=bundles,
    )
    for name in CLAIM_NAMES:
        if _canonical_sha256(claim_result["values"][name]) != _canonical_sha256(
            recomputed_claims[name]
        ):
            raise ReleaseError(
                f"claims.{name} differs from independent source recomputation"
            )
    # Re-authenticate the report containers after every raw input has been
    # recomputed.  This detects a report swap during CPU validation.
    for name in REPORT_BUNDLE_NAMES:
        after = _authenticate_report_bundle(
            raw_evidence[name],
            project_root=project_root,
            label=f"calibration evidence.{name} after recomputation",
            report_name=name,
        )
        if (
            after["manifest_sha256"] != bundles[name]["manifest_sha256"]
            or after["complete_sha256"] != bundles[name]["complete_sha256"]
        ):
            raise ReleaseError(
                f"calibration evidence.{name} changed during recomputation"
            )
    evaluation = _evaluate_calibration_claims(
        recomputed_claims,
        gate=gate,
        checkpoint_rows=candidate_checkpoints,
        final_checkpoint=final_checkpoint,
        calibration_config=calibration_config,
    )
    public_bundles = {
        name: {
            key: value
            for key, value in bundle.items()
            if key not in {"manifest", "payloads"}
        }
        for name, bundle in bundles.items()
    }
    public_checkpoints = [
        {
            "path": row["path"],
            "manifest_sha256": row["manifest_sha256"],
            "complete_sha256": row["complete_sha256"],
            "global_step": row["metadata"].get("global_step"),
            "committed_tokens": row["metadata"].get("committed_tokens"),
            "kind": row["metadata"].get("kind"),
            "tag": row["metadata"].get("tag"),
        }
        for row in candidate_checkpoints
    ]
    return {
        "path": str(attestation_path),
        "size": attestation_path.stat().st_size,
        "sha256": sha256_file(attestation_path),
        "complete": _identity(complete_path),
        "attestation_fingerprint": fingerprint,
        "calibration_config": config_identity,
        "evidence": public_bundles,
        "claims": {
            **copy.deepcopy(claim_result),
            "recomputed_values": copy.deepcopy(recomputed_claims),
        },
        "candidate_checkpoints": public_checkpoints,
        "final_checkpoint": public_checkpoints[-1],
        "evaluation": evaluation,
        "passed": True,
        "authorizes_training": False,
    }


def _authenticate_release_inputs(args: argparse.Namespace) -> dict[str, Any]:
    closure = _authenticate_closure_structure(args.closure)
    governed = _authenticate_governed_closure_gates(closure)
    closure["governed"] = governed
    calibration = _authenticate_calibration_attestation(
        args.calibration_attestation,
        closure=closure,
    )
    return {"closure": closure, "calibration": calibration}


def _final_config(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(snapshot["closure"]["blocked_config"])
    data = _mapping(config.get("data"), label="final config.data")
    if not isinstance(data, dict):
        raise ReleaseError("final config data section is immutable")
    capacity = snapshot["closure"]["governed"]["capacity"]
    phases = _mapping(capacity.get("phases"), label="closed capacity phases")
    primary = _mapping(phases.get("primary"), label="closed capacity primary")
    cooldown = _mapping(phases.get("cooldown"), label="closed capacity cooldown")
    phase = _mapping(
        capacity.get("phase_disjointness_attestation"),
        label="closed phase-disjointness attestation",
    )
    data.update(
        {
            "manifest_path": primary["prepared_manifest_path"],
            "manifest_sha256": primary["prepared_manifest_sha256"],
            "source_map_sha256": primary["source_map_sha256"],
            "quality_cooldown_manifest_path": cooldown["prepared_manifest_path"],
            "quality_cooldown_manifest_sha256": cooldown[
                "prepared_manifest_sha256"
            ],
            "phase_disjointness_attestation_path": phase["path"],
            "phase_disjointness_attestation_sha256": phase["sha256"],
        }
    )
    if _contains_pending(config):
        raise ReleaseError("derived final config still contains a PENDING sentinel")
    try:
        normalized = _normalized_formal_config_sha256(config)
    except GovernedControllerError as exc:
        raise ReleaseError(str(exc)) from exc
    if normalized != FORMAL_V4_NORMALIZED_CONFIG_SHA256:
        raise ReleaseError("derived final config differs from the source-bound policy")
    return config


def _yaml_bytes(value: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        dict(value),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    ).encode("utf-8")


def _release_contract(
    snapshot: Mapping[str, Any],
    *,
    output: Path,
    config_sha256: str,
) -> dict[str, Any]:
    closure = snapshot["closure"]
    calibration = snapshot["calibration"]
    dependency = ROOT / "uv.lock"
    if not dependency.is_file():
        dependency = ROOT / "pyproject.toml"
    contract = {
        "schema_version": SCHEMA_VERSION,
        "kind": RELEASE_KIND,
        "project_root": str(closure["project_root"]),
        "output": {
            "path": str(output),
            "config_path": str(output / FINAL_CONFIG_NAME),
            "readiness_path": str(output / FINAL_READINESS_NAME),
            "manifest_path": str(output / MANIFEST_NAME),
            "complete_path": str(output / COMPLETE_NAME),
        },
        "publisher": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "source_tree": {
            "path": str((ROOT / "src/twen").resolve()),
            "sha256": twen_source_tree_sha256(ROOT / "src/twen"),
        },
        "dependency_lock": _identity(dependency),
        "formal_closure": copy.deepcopy(closure["binding"]),
        "capacity_attestation": _identity(closure["capacity_path"]),
        "closed_readiness": _identity(closure["readiness_path"]),
        "calibration_attestation": {
            key: copy.deepcopy(calibration[key])
            for key in (
                "path",
                "size",
                "sha256",
                "complete",
                "attestation_fingerprint",
                "calibration_config",
                "evidence",
                "candidate_checkpoints",
                "final_checkpoint",
                "evaluation",
            )
        },
        "gates": {
            "capacity": True,
            "calibration": True,
            "chinese_semantic_quality": True,
            "formal_validation": True,
            "wikipedia_license_contract_authenticated": True,
            "wikipedia_license_acceptance_required_at_publish": True,
        },
        "wikipedia_license": {
            "contract": copy.deepcopy(FORMAL_V4_WIKIPEDIA_LICENSE_CONTRACT),
            "contract_fingerprint": FORMAL_V4_WIKIPEDIA_LICENSE_FINGERPRINT,
            "required_acknowledgement": (
                "ACCEPT V4 WIKIPEDIA LICENSE "
                f"{FORMAL_V4_WIKIPEDIA_LICENSE_FINGERPRINT}"
            ),
        },
        "final_config": {
            "sha256": config_sha256,
            "normalized_semantic_sha256": FORMAL_V4_NORMALIZED_CONFIG_SHA256,
        },
        "launch_enabled_after_publish": True,
        "authorizes_training_after_publish": True,
        "training_started": False,
        "web_profile_changed": False,
    }
    if _contains_pending(contract):
        raise ReleaseError("release contract contains a PENDING sentinel")
    return contract


def _build_release_plan_from_snapshot(
    snapshot: Mapping[str, Any],
    *,
    output: Path,
) -> dict[str, Any]:
    final_config = _final_config(snapshot)
    config_bytes = _yaml_bytes(final_config)
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    contract = _release_contract(
        snapshot,
        output=output,
        config_sha256=config_sha,
    )
    fingerprint = _canonical_sha256(contract)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": PLAN_KIND,
        "release_contract": contract,
        "release_fingerprint": fingerprint,
        "required_authorization": f"AUTHORIZE V4 {fingerprint}",
        "required_wikipedia_license_acknowledgement": contract[
            "wikipedia_license"
        ]["required_acknowledgement"],
        "publication_performed": False,
        "launch_enabled_after_publish": True,
        "authorizes_training_after_publish": True,
        "training_started": False,
        "web_profile_changed": False,
        "_final_config": final_config,
        "_final_config_bytes": config_bytes,
    }


def build_release_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Authenticate all inputs and return the read-only release plan."""

    output = args.output.expanduser().resolve()
    if output.exists():
        raise ReleaseError(f"release output already exists: {output}")
    snapshot = _authenticate_release_inputs(args)
    return _build_release_plan_from_snapshot(snapshot, output=output)


def _public_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in plan.items()
        if not key.startswith("_")
    }


def _final_readiness(
    snapshot: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    config_path: Path,
    config_size: int,
    config_sha256: str,
) -> dict[str, Any]:
    readiness = copy.deepcopy(snapshot["closure"]["readiness"])
    calibration = snapshot["calibration"]
    release_fingerprint = plan["release_fingerprint"]
    expected_license_ack = plan["required_wikipedia_license_acknowledgement"]
    calibration_gate = readiness["calibration_gate"]
    calibration_gate.update(
        {
            "status": "passed_authenticated_13m_low_lr_calibration",
            "required_authenticated_evidence": {
                **copy.deepcopy(calibration["evidence"]),
                "final_checkpoint": copy.deepcopy(calibration["final_checkpoint"]),
            },
            "observed": {
                "attestation": {
                    "path": calibration["path"],
                    "size": calibration["size"],
                    "sha256": calibration["sha256"],
                    "attestation_fingerprint": calibration[
                        "attestation_fingerprint"
                    ],
                },
                "complete": copy.deepcopy(calibration["complete"]),
                "claims": copy.deepcopy(calibration["claims"]),
                "candidate_checkpoints": copy.deepcopy(
                    calibration["candidate_checkpoints"]
                ),
                "evaluation": copy.deepcopy(calibration["evaluation"]),
                "passed": True,
                "authorizes_training": False,
            },
            "passed": True,
            "authorizes_training": True,
        }
    )
    semantic = readiness["chinese_semantic_quality_gate"]
    semantic["authorizes_training"] = True
    formal = readiness["formal_validation_gate"]
    formal["authorizes_training"] = True
    license_gate = readiness["wikipedia_license_gate"]
    license_gate.update(
        {
            "status": "accepted_explicit_user_acknowledgement",
            "observed_acknowledgement": expected_license_ack,
            "passed": True,
            "authorizes_training": True,
        }
    )
    readiness.update(
        {
            "status": "authorized_for_governed_v4_250m_launch",
            "config_path": str(config_path),
            "config_sha256": config_sha256,
            "config_identity": {
                "path": str(config_path),
                "size": config_size,
                "sha256": config_sha256,
                "normalized_semantic_sha256": (
                    FORMAL_V4_NORMALIZED_CONFIG_SHA256
                ),
            },
            "blockers": [],
            "launch_enabled": True,
            "authorizes_training": True,
            "training_started": False,
            "launch_command_after_all_gates_pass": None,
            "launch_command_status": (
                "release_authorized_but_governed_RUN_plan_ack_still_required"
            ),
            "release": {
                "kind": RELEASE_KIND,
                "release_fingerprint": release_fingerprint,
                "publisher": copy.deepcopy(
                    plan["release_contract"]["publisher"]
                ),
                "formal_release_acknowledgement": plan[
                    "required_authorization"
                ],
                "wikipedia_license_acknowledgement": expected_license_ack,
                "web_profile_changed": False,
                "training_started": False,
            },
        }
    )
    readiness.pop("readiness_fingerprint", None)
    if _contains_pending(readiness):
        raise ReleaseError("derived final readiness still contains a PENDING sentinel")
    readiness["readiness_fingerprint"] = _canonical_sha256(readiness)
    return readiness


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


def _same_release_plan(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return _canonical_sha256(_public_plan(first)) == _canonical_sha256(
        _public_plan(second)
    )


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically install a directory without replacing even an empty target."""

    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,  # AT_FDCWD
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,  # RENAME_NOREPLACE
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise ReleaseError(
                f"release output appeared during publication: {destination}"
            )
        if error_number in {errno.ENOSYS, errno.EINVAL}:
            raise ReleaseError(
                "atomic no-replace directory installation is unsupported; "
                "refusing racy fallback"
            )
        raise OSError(
            error_number,
            os.strerror(error_number),
            str(destination),
        )
    raise ReleaseError(
        "renameat2(RENAME_NOREPLACE) is unavailable; refusing racy fallback"
    )


@contextmanager
def _directory_lock(path: Path, *, timeout_seconds: float = 300.0) -> Any:
    """Advisory-lock an existing output parent without creating a lock file."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise ReleaseError(
                        f"timed out waiting for release parent lock: {path}"
                    ) from exc
                time.sleep(0.1)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def publish_release(args: argparse.Namespace) -> dict[str, Any]:
    """Require exact ACKs, re-authenticate, and atomically publish one bundle."""

    output = args.output.expanduser().resolve()
    if output.exists():
        raise ReleaseError(f"release output already exists: {output}")
    snapshot = _authenticate_release_inputs(args)
    plan = _build_release_plan_from_snapshot(snapshot, output=output)
    expected_authorization = plan["required_authorization"]
    expected_license = plan["required_wikipedia_license_acknowledgement"]
    if args.authorize_ack != expected_authorization:
        raise ReleaseError(
            f"explicit authorization must equal {expected_authorization!r}"
        )
    if args.wikipedia_license_ack != expected_license:
        raise ReleaseError(
            f"Wikipedia license acknowledgement must equal {expected_license!r}"
        )

    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ReleaseError(
            f"release output parent must already be a real directory: {output.parent}"
        )
    with _directory_lock(output.parent):
        if output.exists():
            raise ReleaseError(f"release output already exists: {output}")
        work = Path(
            tempfile.mkdtemp(
                prefix=f".{output.name}.incomplete-",
                dir=output.parent,
            )
        )
        try:
            config_path = work / FINAL_CONFIG_NAME
            readiness_path = work / FINAL_READINESS_NAME
            manifest_path = work / MANIFEST_NAME
            complete_path = work / COMPLETE_NAME
            config_bytes = plan["_final_config_bytes"]
            _write_bytes(config_path, config_bytes)
            final_readiness = _final_readiness(
                snapshot,
                plan=plan,
                config_path=output / FINAL_CONFIG_NAME,
                config_size=len(config_bytes),
                config_sha256=sha256_file(config_path),
            )
            _write_json(readiness_path, final_readiness)
            files = {
                FINAL_CONFIG_NAME: {
                    "path": FINAL_CONFIG_NAME,
                    "size": config_path.stat().st_size,
                    "sha256": sha256_file(config_path),
                },
                FINAL_READINESS_NAME: {
                    "path": FINAL_READINESS_NAME,
                    "size": readiness_path.stat().st_size,
                    "sha256": sha256_file(readiness_path),
                },
            }
            manifest: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "kind": RELEASE_BUNDLE_KIND,
                "release_fingerprint": plan["release_fingerprint"],
                "release_contract": copy.deepcopy(plan["release_contract"]),
                "acknowledgements": {
                    "formal_release": expected_authorization,
                    "wikipedia_license": expected_license,
                },
                "files": files,
                "launch_enabled": True,
                "authorizes_training": True,
                "training_started": False,
                "web_profile_changed": False,
            }
            if _contains_pending(manifest):
                raise ReleaseError("derived release MANIFEST contains a PENDING sentinel")
            manifest["bundle_fingerprint"] = _canonical_sha256(manifest)
            _write_json(manifest_path, manifest)
            complete = {
                "schema_version": SCHEMA_VERSION,
                "kind": RELEASE_COMPLETE_KIND,
                "manifest": MANIFEST_NAME,
                "manifest_sha256": sha256_file(manifest_path),
                "bundle_fingerprint": manifest["bundle_fingerprint"],
                "release_fingerprint": plan["release_fingerprint"],
                "launch_enabled": True,
                "authorizes_training": True,
                "training_started": False,
                "web_profile_changed": False,
            }
            _write_json(complete_path, complete)
            for path in (config_path, readiness_path, manifest_path, complete_path):
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            directory_fd = os.open(work, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

            second_snapshot = _authenticate_release_inputs(args)
            second_plan = _build_release_plan_from_snapshot(
                second_snapshot,
                output=output,
            )
            if not _same_release_plan(plan, second_plan):
                raise ReleaseError("release inputs changed during publication")
            if output.exists():
                raise ReleaseError(f"release output appeared during publication: {output}")
            _rename_directory_noreplace(work, output)
            parent_fd = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
    return {
        **_public_plan(plan),
        "publication_performed": True,
        "output": str(output),
        "config": str(output / FINAL_CONFIG_NAME),
        "readiness": str(output / FINAL_READINESS_NAME),
        "manifest": str(output / MANIFEST_NAME),
        "complete": str(output / COMPLETE_NAME),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = (
            build_release_plan(args)
            if args.action == "plan"
            else publish_release(args)
        )
    except (ReleaseError, GovernedControllerError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_public_plan(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
