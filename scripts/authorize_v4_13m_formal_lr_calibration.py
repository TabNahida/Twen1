#!/usr/bin/env python3
"""Plan or publish the pre-training admission for v4 13M formal-LR calibration.

The formal evidence closure deliberately keeps the Wikipedia licence gate
pending and immutable.  This command records the user's exact acknowledgement
in a separate, source-bound admission bundle and only then enables the single
13M formal-LR calibration profile in the Dashboard allow-list.

Neither ``plan`` nor ``publish`` starts training.  A launch still requires the
independent Dashboard confirmation printed by this command.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from twen.source_identity import twen_source_tree_sha256
from twen.utils import sha256_file
from twen.web import load_dashboard_settings

SCHEMA_VERSION = 1
PROFILE_ID = "base-dense-v4-13m-formal-lr-calibration"
ADMISSION_KIND = "twen_v4_13m_formal_lr_calibration_admission"
BUNDLE_KIND = "twen_v4_13m_formal_lr_calibration_admission_bundle"
COMPLETE_KIND = "twen_v4_13m_formal_lr_calibration_admission_complete"
ADMISSION_NAME = "admission.json"
DASHBOARD_NAME = "dashboard.json"
MANIFEST_NAME = "MANIFEST.json"
COMPLETE_NAME = "COMPLETE"
AUTHORIZED_LABEL = (
    "Base Dense v4 13M formal-LR calibration (license accepted; awaiting explicit START)"
)
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CLOSURE_MANIFEST_SHA256 = (
    "1d78cde930f7407e443e21ae14870b7075ac7dddb2bcc070034200bb4f69b470"
)
EXPECTED_CONFIG_PATH = "configs/base/dense-v4-13m-formal-lr-calibration.yaml"
EXPECTED_CONFIG_SHA256 = "13fdaea6c21b2e070246a6836be33718137ec1df7563232ba2d7a15dbf7536e9"
EXPECTED_PREPARED_PATH = (
    "artifacts/data/base-v4-250m-primary-r2-semantic-excluded-closed-train-pass-001/manifest.json"
)
EXPECTED_PREPARED_SHA256 = "cf1d837e2130e1d5a045f151eddae5fb20250b44f037676c933b2c6ccfe75af8"
EXPECTED_SOURCE_MAP_SHA256 = "d6620197c785464885461738727c320d8046c513b535090c7412615292c50efe"
EXPECTED_FORK_PATH = "runs/base-dense-v3-500m/step-000000001912-milestone-complete"
EXPECTED_FORK_COMPLETE_SHA256 = "3a21a50e35de74ecd0ff5b8f00aa29ed6c83f746fc2cf97d4da6b0536262b6c7"
EXPECTED_RUN_DIR = "runs/base-dense-v4-13m-formal-lr-calibration"
EXPECTED_DASHBOARD_TEMPLATE = "configs/web/dashboard.json"
SOURCE_BOUND_FORBIDDEN_WARM_STARTS = (
    "runs/base-dense-v4-16m-smoke",
    "runs/base-dense-v4-13m-low-lr-calibration",
)
EXPECTED_FORMAL_LR_FORBIDDEN_WARM_STARTS = (
    *SOURCE_BOUND_FORBIDDEN_WARM_STARTS,
    EXPECTED_RUN_DIR,
)


class CalibrationAdmissionError(ValueError):
    """The calibration admission cannot be authenticated or published."""


def _load_publisher() -> tuple[ModuleType, str]:
    path = Path(__file__).resolve().with_name("publish_v4_250m_release.py")
    source_before = path.read_bytes()
    spec = importlib.util.spec_from_file_location(
        "_twen_v4_release_publisher_for_formal_lr_calibration_admission",
        path,
    )
    if spec is None or spec.loader is None:
        raise CalibrationAdmissionError(f"cannot load release verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if path.read_bytes() != source_before:
        raise CalibrationAdmissionError(
            "release verifier changed while calibration admission loaded"
        )
    return module, hashlib.sha256(source_before).hexdigest()


publisher, PUBLISHER_SOURCE_SHA256_AT_IMPORT = _load_publisher()
AUTHORIZER_SOURCE_SHA256_AT_IMPORT = sha256_file(Path(__file__).resolve())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "publish", "verify"))
    parser.add_argument("--closure", type=Path, required=True)
    parser.add_argument("--dashboard-config", type=Path, required=True)
    parser.add_argument("--profile-id", default=PROFILE_ID)
    parser.add_argument("--acknowledgement", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CalibrationAdmissionError(f"{label} must be an object")
    return value


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise CalibrationAdmissionError(f"{label} is missing or a symlink: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise CalibrationAdmissionError(f"{label} is missing or a symlink: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationAdmissionError(f"cannot read {label} JSON {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise CalibrationAdmissionError(f"{label} must be a JSON object")
    return value


def _identity(path: Path) -> dict[str, Any]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise CalibrationAdmissionError(f"authenticated file is missing or a symlink: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise CalibrationAdmissionError(f"authenticated file is missing or a symlink: {resolved}")
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _reject_direct_symlink(path: Path, *, label: str) -> None:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise CalibrationAdmissionError(f"{label} must not be a symlink: {expanded}")


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path.expanduser()))


def _strict_bundle_inventory(root: Path, *, expected_files: set[str]) -> None:
    entries = list(root.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise CalibrationAdmissionError(
            "calibration admission bundle contains a symlink or non-file entry"
        )
    if {entry.name for entry in entries} != expected_files:
        raise CalibrationAdmissionError("calibration admission bundle file inventory differs")


def _require_sources_unchanged() -> None:
    if (
        sha256_file(Path(__file__).resolve()) != AUTHORIZER_SOURCE_SHA256_AT_IMPORT
        or sha256_file(Path(publisher.__file__).resolve()) != PUBLISHER_SOURCE_SHA256_AT_IMPORT
    ):
        raise CalibrationAdmissionError("authorizer or release verifier changed after import")


def _inside(root: Path, path: Path, *, label: str) -> Path:
    resolved_root = root.expanduser().resolve()
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise CalibrationAdmissionError(
            f"{label} must remain inside {resolved_root}: {resolved}"
        ) from exc
    return resolved


def _require_exact_acknowledgement(observed: str, expected: str) -> None:
    if observed != expected:
        raise CalibrationAdmissionError(f"explicit acknowledgement must equal {expected!r}")


def _dashboard_project_root(
    dashboard_path: Path,
    raw_dashboard: Mapping[str, Any],
) -> Path:
    value = raw_dashboard.get("project_root", ".")
    if not isinstance(value, str) or not value:
        raise CalibrationAdmissionError("dashboard project_root must be a non-empty string")
    return (dashboard_path.parent / value).resolve()


def _authenticate_formal_lr_governed_gates(
    closure: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    readiness = _mapping(closure.get("readiness"), label="closed readiness")
    fork_policy = _mapping(
        readiness.get("fork_policy"),
        label="closed readiness.fork_policy",
    )
    if fork_policy.get("forbidden_warm_starts") != list(EXPECTED_FORMAL_LR_FORBIDDEN_WARM_STARTS):
        raise CalibrationAdmissionError(
            "formal-LR closure forbidden warm starts differ from the fixed contract"
        )

    source_bound_readiness = copy.deepcopy(dict(readiness))
    source_bound_fork_policy = _mapping(
        source_bound_readiness.get("fork_policy"),
        label="source-bound readiness.fork_policy",
    )
    source_bound_fork_policy["forbidden_warm_starts"] = list(SOURCE_BOUND_FORBIDDEN_WARM_STARTS)
    if "readiness_fingerprint" in source_bound_readiness:
        unsigned = {
            key: value
            for key, value in source_bound_readiness.items()
            if key != "readiness_fingerprint"
        }
        source_bound_readiness["readiness_fingerprint"] = _canonical_sha256(unsigned)

    with tempfile.TemporaryDirectory(prefix=".twen-formal-lr-governed-audit-") as temporary:
        readiness_path = Path(temporary) / "readiness.json"
        readiness_path.write_text(
            json.dumps(
                source_bound_readiness,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        source_bound_closure = dict(closure)
        source_bound_closure["readiness"] = source_bound_readiness
        source_bound_closure["readiness_path"] = readiness_path
        try:
            governed = publisher._authenticate_governed_closure_gates(source_bound_closure)
        except (publisher.ReleaseError, OSError, ValueError) as exc:
            raise CalibrationAdmissionError(
                f"formal closure governed-gate authentication failed: {exc}"
            ) from exc
    return governed, copy.deepcopy(dict(fork_policy))


def _authenticate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    _require_sources_unchanged()
    if args.profile_id != PROFILE_ID:
        raise CalibrationAdmissionError(f"profile_id must equal {PROFILE_ID!r}")
    _reject_direct_symlink(args.closure, label="formal closure")
    _reject_direct_symlink(
        args.dashboard_config,
        label="dashboard config",
    )
    try:
        closure = publisher._authenticate_closure_structure(args.closure)
    except (publisher.ReleaseError, OSError, ValueError) as exc:
        raise CalibrationAdmissionError(f"formal closure authentication failed: {exc}") from exc
    governed, formal_lr_fork_policy = _authenticate_formal_lr_governed_gates(closure)

    readiness = _mapping(closure.get("readiness"), label="closed readiness")
    project_root = Path(str(closure.get("project_root"))).resolve()
    if project_root != ROOT.resolve():
        raise CalibrationAdmissionError("formal closure belongs to another project checkout")
    if closure.get("binding", {}).get("manifest_sha256") != EXPECTED_CLOSURE_MANIFEST_SHA256:
        raise CalibrationAdmissionError(
            "formal closure is not the approved semantic-excluded identity"
        )
    expected_dashboard_path = (project_root / EXPECTED_DASHBOARD_TEMPLATE).resolve()
    if args.dashboard_config.expanduser().resolve() != expected_dashboard_path:
        raise CalibrationAdmissionError("dashboard config is not the approved launch template")

    licence_gate = _mapping(
        readiness.get("wikipedia_license_gate"),
        label="closed wikipedia_license_gate",
    )
    expected_acknowledgement = (
        f"ACCEPT V4 WIKIPEDIA LICENSE {publisher.FORMAL_V4_WIKIPEDIA_LICENSE_FINGERPRINT}"
    )
    if (
        licence_gate.get("required_acknowledgement") != expected_acknowledgement
        or licence_gate.get("contract_fingerprint")
        != publisher.FORMAL_V4_WIKIPEDIA_LICENSE_FINGERPRINT
        or licence_gate.get("status") != "pending_explicit_user_acceptance"
        or licence_gate.get("observed_acknowledgement") is not None
        or licence_gate.get("passed") is not False
        or licence_gate.get("authorizes_training") is not False
    ):
        raise CalibrationAdmissionError(
            "closed Wikipedia licence gate differs from the pending contract"
        )
    _require_exact_acknowledgement(
        args.acknowledgement,
        expected_acknowledgement,
    )

    calibration_gate = _mapping(
        readiness.get("calibration_gate"),
        label="closed calibration_gate",
    )
    calibration_config = _mapping(
        calibration_gate.get("config"),
        label="closed calibration_gate.config",
    )
    raw_config_path = calibration_config.get("path")
    config_entry = (
        Path(raw_config_path).expanduser() if isinstance(raw_config_path, str) else Path("")
    )
    if not config_entry.is_absolute():
        config_entry = project_root / config_entry
    _reject_direct_symlink(
        config_entry,
        label="calibration config",
    )
    config_path = publisher._resolve_path(
        raw_config_path,
        base=project_root,
        label="calibration config",
    )
    config_identity = _identity(config_path)
    if (
        config_path != (project_root / EXPECTED_CONFIG_PATH).resolve()
        or config_identity["sha256"] != EXPECTED_CONFIG_SHA256
        or calibration_config.get("sha256") != EXPECTED_CONFIG_SHA256
    ):
        raise CalibrationAdmissionError(
            "calibration config identity differs from the formal closure"
        )
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise CalibrationAdmissionError(f"cannot read calibration config: {exc}") from exc
    config = _mapping(raw_config, label="calibration config")
    if config.get("run_id") != args.profile_id:
        raise CalibrationAdmissionError(
            "calibration config run_id differs from the selected profile"
        )
    data = _mapping(config.get("data"), label="calibration config.data")
    checkpoint = _mapping(
        config.get("checkpoint"),
        label="calibration config.checkpoint",
    )
    optimizer = _mapping(
        config.get("optimizer"),
        label="calibration config.optimizer",
    )
    if (
        optimizer.get("max_tokens") != 13_000_000
        or data.get("allow_corpus_reuse") is not False
        or data.get("global_batch_tokens") != 262_144
        or data.get("micro_batch_size") != 1
        or checkpoint.get("output_dir") != EXPECTED_RUN_DIR
    ):
        raise CalibrationAdmissionError(
            "calibration config no longer matches the bounded 13M contract"
        )

    capacity = _mapping(governed.get("capacity"), label="governed capacity")
    phases = _mapping(capacity.get("phases"), label="governed capacity.phases")
    primary = _mapping(phases.get("primary"), label="governed primary phase")
    raw_prepared_path = data.get("manifest_path")
    prepared_entry = (
        Path(raw_prepared_path).expanduser() if isinstance(raw_prepared_path, str) else Path("")
    )
    if not prepared_entry.is_absolute():
        prepared_entry = project_root / prepared_entry
    _reject_direct_symlink(
        prepared_entry,
        label="calibration prepared manifest",
    )
    prepared_path = publisher._resolve_path(
        raw_prepared_path,
        base=project_root,
        label="calibration prepared manifest",
    )
    prepared_identity = _identity(prepared_path)
    if (
        prepared_path != (project_root / EXPECTED_PREPARED_PATH).resolve()
        or prepared_identity["sha256"] != EXPECTED_PREPARED_SHA256
        or data.get("manifest_sha256") != EXPECTED_PREPARED_SHA256
        or data.get("source_map_sha256") != EXPECTED_SOURCE_MAP_SHA256
        or prepared_path != Path(str(primary.get("prepared_manifest_path"))).resolve()
        or prepared_identity["sha256"] != primary.get("prepared_manifest_sha256")
        or data.get("manifest_sha256") != primary.get("prepared_manifest_sha256")
        or data.get("source_map_sha256") != primary.get("source_map_sha256")
    ):
        raise CalibrationAdmissionError(
            "calibration data does not bind the closed formal primary phase"
        )

    required_fork = _mapping(
        calibration_gate.get("required_fork_checkpoint"),
        label="closed calibration required fork",
    )
    raw_fork_path = required_fork.get("path")
    fork_entry = Path(raw_fork_path).expanduser() if isinstance(raw_fork_path, str) else Path("")
    if not fork_entry.is_absolute():
        fork_entry = project_root / fork_entry
    _reject_direct_symlink(
        fork_entry,
        label="calibration fork checkpoint",
    )
    fork_path = publisher._resolve_path(
        raw_fork_path,
        base=project_root,
        label="calibration fork checkpoint",
    )
    try:
        fork_checkpoint = publisher._checkpoint_binding_from_path(
            fork_path,
            project_root=project_root,
            label="calibration fork checkpoint",
        )
    except (publisher.ReleaseError, OSError, ValueError) as exc:
        raise CalibrationAdmissionError(
            f"calibration fork checkpoint authentication failed: {exc}"
        ) from exc
    if (
        fork_path != (project_root / EXPECTED_FORK_PATH).resolve()
        or fork_checkpoint.get("complete_sha256") != EXPECTED_FORK_COMPLETE_SHA256
        or required_fork.get("complete_sha256") != EXPECTED_FORK_COMPLETE_SHA256
        or fork_checkpoint.get("complete_sha256") != required_fork.get("complete_sha256")
        or required_fork.get("model_only") is not True
        or required_fork.get("reset_optimizer_and_scheduler") is not True
    ):
        raise CalibrationAdmissionError(
            "calibration fork checkpoint differs from the model-only contract"
        )

    dashboard_path = args.dashboard_config.expanduser().resolve()
    dashboard = _read_json(dashboard_path, label="dashboard config")
    if dashboard.get("schema_version") != SCHEMA_VERSION:
        raise CalibrationAdmissionError("dashboard config schema_version must equal 1")
    if _dashboard_project_root(dashboard_path, dashboard) != project_root:
        raise CalibrationAdmissionError("dashboard config belongs to another project checkout")
    raw_profiles = dashboard.get("profiles")
    if not isinstance(raw_profiles, list):
        raise CalibrationAdmissionError("dashboard config has no profiles list")
    selected = [
        item
        for item in raw_profiles
        if isinstance(item, Mapping) and item.get("id") == args.profile_id
    ]
    if len(selected) != 1:
        raise CalibrationAdmissionError(
            "dashboard must contain exactly one selected calibration profile"
        )
    profile = selected[0]
    if any(
        isinstance(item, Mapping) and item.get("launch_enabled") is True for item in raw_profiles
    ):
        raise CalibrationAdmissionError("dashboard already contains a launch-enabled profile")
    raw_profile_config = profile.get("config")
    profile_config_entry = (
        Path(raw_profile_config).expanduser() if isinstance(raw_profile_config, str) else Path("")
    )
    if not profile_config_entry.is_absolute():
        profile_config_entry = project_root / profile_config_entry
    _reject_direct_symlink(
        profile_config_entry,
        label="dashboard calibration config",
    )
    raw_profile_fork = profile.get("fork_from")
    profile_fork_entry = (
        Path(raw_profile_fork).expanduser() if isinstance(raw_profile_fork, str) else Path("")
    )
    if not profile_fork_entry.is_absolute():
        profile_fork_entry = project_root / profile_fork_entry
    _reject_direct_symlink(
        profile_fork_entry,
        label="dashboard calibration fork",
    )
    profile_config = publisher._resolve_path(
        raw_profile_config,
        base=project_root,
        label="dashboard calibration config",
    )
    profile_fork = publisher._resolve_path(
        raw_profile_fork,
        base=project_root,
        label="dashboard calibration fork",
    )
    if (
        profile.get("launch_enabled") is not False
        or profile.get("launch_kind", "direct_train") != "direct_train"
        or profile.get("resume") != "none"
        or profile_config != config_path
        or profile.get("config_sha256") != config_identity["sha256"]
        or profile_fork != fork_path
    ):
        raise CalibrationAdmissionError(
            "dashboard calibration profile differs from the closed contract"
        )
    if "calibration_admission" in profile:
        raise CalibrationAdmissionError(
            "dashboard calibration profile already declares an admission"
        )
    settings = load_dashboard_settings(dashboard_path)
    loaded_profile = settings.profile(PROFILE_ID)
    expected_run_dir = (project_root / EXPECTED_RUN_DIR).resolve()
    if (
        loaded_profile.stage != "dense-oracle"
        or loaded_profile.run_id != PROFILE_ID
        or loaded_profile.run_dir != expected_run_dir
        or loaded_profile.launch_enabled is not False
    ):
        raise CalibrationAdmissionError(
            "loaded Dashboard profile differs from the fixed calibration run"
        )

    output_dir_value = checkpoint.get("output_dir")
    if not isinstance(output_dir_value, str) or not output_dir_value:
        raise CalibrationAdmissionError("calibration checkpoint.output_dir is invalid")
    run_dir = _inside(
        project_root,
        project_root / output_dir_value,
        label="calibration run directory",
    )
    if run_dir != expected_run_dir:
        raise CalibrationAdmissionError("calibration run directory differs from the fixed contract")
    run_dir_entry = project_root / output_dir_value
    _reject_direct_symlink(
        run_dir_entry,
        label="calibration run directory",
    )
    if _lexists(run_dir) and not run_dir.is_dir():
        raise CalibrationAdmissionError(
            f"calibration run directory exists but is not a directory: {run_dir}"
        )

    dashboard_identity = _identity(dashboard_path)
    source_identity = {
        "authorizer": {
            **_identity(Path(__file__).resolve()),
            "import_sha256": AUTHORIZER_SOURCE_SHA256_AT_IMPORT,
        },
        "release_verifier": {
            **_identity(Path(publisher.__file__).resolve()),
            "import_sha256": PUBLISHER_SOURCE_SHA256_AT_IMPORT,
        },
        "twen_source_tree_sha256": twen_source_tree_sha256(project_root / "src/twen"),
    }
    profile_contract = {
        "id": args.profile_id,
        "label": AUTHORIZED_LABEL,
        "config": profile.get("config"),
        "config_sha256": config_identity["sha256"],
        "fork_from": profile.get("fork_from"),
        "resume": "none",
        "launch_kind": "direct_train",
        "launch_enabled": True,
        "run_dir": str(run_dir),
        "required_start_confirmation": f"START {args.profile_id}",
    }
    authenticated_inputs = {
        "formal_closure": copy.deepcopy(closure["binding"]),
        "formal_lr_fork_policy": formal_lr_fork_policy,
        "licence_contract": {
            "contract": copy.deepcopy(licence_gate.get("contract")),
            "contract_fingerprint": licence_gate.get("contract_fingerprint"),
            "required_acknowledgement": expected_acknowledgement,
        },
        "calibration_config": config_identity,
        "formal_primary_prepared_manifest": {
            **prepared_identity,
            "dataset_fingerprint": primary.get("prepared_dataset_fingerprint"),
            "source_map_sha256": primary.get("source_map_sha256"),
        },
        "fork_checkpoint": fork_checkpoint,
        "dashboard_before": dashboard_identity,
        "source_identity": source_identity,
        "authorized_profile_contract": profile_contract,
    }
    _require_sources_unchanged()
    return {
        "project_root": project_root,
        "dashboard_path": dashboard_path,
        "dashboard": copy.deepcopy(dashboard),
        "profile_id": args.profile_id,
        "expected_acknowledgement": expected_acknowledgement,
        "authenticated_inputs": authenticated_inputs,
        "input_fingerprint": _canonical_sha256(authenticated_inputs),
    }


def _require_calibration_not_started(snapshot: Mapping[str, Any]) -> None:
    authenticated_inputs = _mapping(
        snapshot.get("authenticated_inputs"),
        label="authenticated calibration inputs",
    )
    profile = _mapping(
        authenticated_inputs.get("authorized_profile_contract"),
        label="authorized calibration profile",
    )
    raw_run_dir = profile.get("run_dir")
    if not isinstance(raw_run_dir, str) or not raw_run_dir:
        raise CalibrationAdmissionError("authorized calibration run directory is invalid")
    run_dir = Path(raw_run_dir)
    if _lexists(run_dir):
        raise CalibrationAdmissionError(f"calibration run directory already exists: {run_dir}")


def _build_plan(
    snapshot: Mapping[str, Any],
    *,
    output: Path,
) -> dict[str, Any]:
    contract = {
        "schema_version": SCHEMA_VERSION,
        "kind": ADMISSION_KIND,
        "input_fingerprint": snapshot["input_fingerprint"],
        "authenticated_inputs": copy.deepcopy(snapshot["authenticated_inputs"]),
        "acknowledgement": snapshot["expected_acknowledgement"],
        "output": str(output),
        "authorizes_calibration_launch": True,
        "authorizes_formal_training": False,
        "training_started": False,
    }
    admission_fingerprint = _canonical_sha256(contract)
    profile_contract = snapshot["authenticated_inputs"]["authorized_profile_contract"]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "twen_v4_13m_formal_lr_calibration_admission_plan",
        "admission_fingerprint": admission_fingerprint,
        "input_fingerprint": snapshot["input_fingerprint"],
        "output": str(output),
        "dashboard_config": str(snapshot["dashboard_path"]),
        "required_start_confirmation": profile_contract["required_start_confirmation"],
        "authorizes_calibration_launch": True,
        "authorizes_formal_training": False,
        "training_started": False,
        "_contract": contract,
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    snapshot = _authenticate_inputs(args)
    _require_calibration_not_started(snapshot)
    raw_output = args.output.expanduser()
    _reject_direct_symlink(
        raw_output,
        label="calibration admission output",
    )
    _reject_direct_symlink(
        raw_output.parent,
        label="calibration admission output parent",
    )
    output = _inside(
        Path(str(snapshot["project_root"])),
        raw_output.resolve(),
        label="calibration admission output",
    )
    if _lexists(output):
        raise CalibrationAdmissionError(f"calibration admission output already exists: {output}")
    return _build_plan(snapshot, output=output)


def _public_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value) for key, value in plan.items() if not str(key).startswith("_")
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    publisher._write_json(path, value)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _dashboard_with_admission(
    snapshot: Mapping[str, Any],
    *,
    output: Path,
    admission_sha256: str,
    admission_fingerprint: str,
) -> dict[str, Any]:
    dashboard = copy.deepcopy(snapshot["dashboard"])
    project_root = Path(str(snapshot["project_root"])).resolve()
    dashboard["project_root"] = os.path.relpath(project_root, output)
    profile_id = str(snapshot["profile_id"])
    relative_output = output.relative_to(project_root).as_posix()
    selected = 0
    for profile in dashboard["profiles"]:
        if not isinstance(profile, dict):
            continue
        if profile.get("id") == profile_id:
            selected += 1
            profile["label"] = AUTHORIZED_LABEL
            profile["launch_enabled"] = True
            profile["calibration_admission"] = {
                "path": relative_output,
                "admission_sha256": admission_sha256,
                "admission_fingerprint": admission_fingerprint,
                "required_start_confirmation": f"START {profile_id}",
                "authorizes_formal_training": False,
                "training_started": False,
            }
        elif profile.get("launch_enabled") is not False:
            raise CalibrationAdmissionError(
                "a non-calibration dashboard profile became launch-enabled"
            )
    if selected != 1:
        raise CalibrationAdmissionError("dashboard selected calibration profile count changed")
    return dashboard


def _write_dashboard_snapshot(
    dashboard_path: Path,
    dashboard: Mapping[str, Any],
) -> str:
    payload = (
        json.dumps(
            dashboard,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    publisher._write_bytes(dashboard_path, payload)
    settings = load_dashboard_settings(dashboard_path)
    launchable = [profile.profile_id for profile in settings.profiles if profile.launch_enabled]
    if launchable != [PROFILE_ID]:
        raise CalibrationAdmissionError(
            "derived dashboard does not enable exactly the calibration profile"
        )
    return hashlib.sha256(payload).hexdigest()


def publish_admission(args: argparse.Namespace) -> dict[str, Any]:
    snapshot = _authenticate_inputs(args)
    _require_calibration_not_started(snapshot)
    raw_output = args.output.expanduser()
    _reject_direct_symlink(
        raw_output,
        label="calibration admission output",
    )
    _reject_direct_symlink(
        raw_output.parent,
        label="calibration admission output parent",
    )
    output = _inside(
        Path(str(snapshot["project_root"])),
        raw_output.resolve(),
        label="calibration admission output",
    )
    if _lexists(output):
        raise CalibrationAdmissionError(f"calibration admission output already exists: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise CalibrationAdmissionError(
            "calibration admission output parent must already be a real directory"
        )
    plan = _build_plan(snapshot, output=output)

    work: Path | None = None
    with publisher._directory_lock(output.parent):
        if _lexists(output):
            raise CalibrationAdmissionError(
                f"calibration admission output already exists: {output}"
            )
        work = Path(
            tempfile.mkdtemp(
                prefix=f".{output.name}.incomplete-",
                dir=output.parent,
            )
        )
        try:
            accepted_at = datetime.now(UTC).isoformat()
            admission = {
                **copy.deepcopy(plan["_contract"]),
                "status": "accepted_explicit_user_acknowledgement",
                "accepted_at": accepted_at,
                "admission_fingerprint": plan["admission_fingerprint"],
            }
            admission_path = work / ADMISSION_NAME
            dashboard_bundle_path = work / DASHBOARD_NAME
            manifest_path = work / MANIFEST_NAME
            complete_path = work / COMPLETE_NAME
            _write_json(admission_path, admission)
            derived_dashboard = _dashboard_with_admission(
                snapshot,
                output=output,
                admission_sha256=sha256_file(admission_path),
                admission_fingerprint=plan["admission_fingerprint"],
            )
            dashboard_sha256 = _write_dashboard_snapshot(
                dashboard_bundle_path,
                derived_dashboard,
            )
            manifest: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "kind": BUNDLE_KIND,
                "admission_fingerprint": plan["admission_fingerprint"],
                "input_fingerprint": plan["input_fingerprint"],
                "files": {
                    ADMISSION_NAME: {
                        "path": ADMISSION_NAME,
                        "size": admission_path.stat().st_size,
                        "sha256": sha256_file(admission_path),
                    },
                    DASHBOARD_NAME: {
                        "path": DASHBOARD_NAME,
                        "size": dashboard_bundle_path.stat().st_size,
                        "sha256": sha256_file(dashboard_bundle_path),
                    },
                },
                "authorized_dashboard": {
                    "path": DASHBOARD_NAME,
                    "sha256": dashboard_sha256,
                },
                "authorizes_calibration_launch": True,
                "authorizes_formal_training": False,
                "training_started": False,
                "dashboard_profile_change_authorized": True,
            }
            manifest["bundle_fingerprint"] = _canonical_sha256(manifest)
            _write_json(manifest_path, manifest)
            complete = {
                "schema_version": SCHEMA_VERSION,
                "kind": COMPLETE_KIND,
                "manifest": MANIFEST_NAME,
                "manifest_sha256": sha256_file(manifest_path),
                "bundle_fingerprint": manifest["bundle_fingerprint"],
                "admission_fingerprint": plan["admission_fingerprint"],
                "authorizes_calibration_launch": True,
                "authorizes_formal_training": False,
                "training_started": False,
                "dashboard_profile_change_authorized": True,
            }
            _write_json(complete_path, complete)
            for path in (
                admission_path,
                dashboard_bundle_path,
                manifest_path,
                complete_path,
            ):
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            _fsync_directory(work)

            second_snapshot = _authenticate_inputs(args)
            _require_calibration_not_started(second_snapshot)
            if second_snapshot["input_fingerprint"] != snapshot["input_fingerprint"]:
                raise CalibrationAdmissionError(
                    "calibration admission inputs changed during publication"
                )
            if _lexists(output):
                raise CalibrationAdmissionError(
                    "calibration admission output appeared during publication"
                )
            publisher._rename_directory_noreplace(work, output)
            work = None
            _fsync_directory(output.parent)

            final_manifest = output / MANIFEST_NAME
            final_complete = output / COMPLETE_NAME
            published_manifest = _read_json(
                final_manifest,
                label="published calibration admission MANIFEST",
            )
            published_complete = _read_json(
                final_complete,
                label="published calibration admission COMPLETE",
            )
            if (
                published_manifest.get("admission_fingerprint") != plan["admission_fingerprint"]
                or published_complete.get("manifest_sha256") != sha256_file(final_manifest)
                or published_complete.get("bundle_fingerprint")
                != published_manifest.get("bundle_fingerprint")
            ):
                raise CalibrationAdmissionError(
                    "published calibration admission bundle failed verification"
                )
            return {
                **_public_plan(plan),
                "publication_performed": True,
                "accepted_at": accepted_at,
                "admission": str(output / ADMISSION_NAME),
                "manifest": str(final_manifest),
                "manifest_sha256": sha256_file(final_manifest),
                "complete": str(final_complete),
                "complete_sha256": sha256_file(final_complete),
                "dashboard_snapshot": str(output / DASHBOARD_NAME),
                "dashboard_config": str(output / DASHBOARD_NAME),
                "dashboard_config_sha256": dashboard_sha256,
                "dashboard_profile_changed": False,
            }
        except BaseException:
            if work is not None:
                shutil.rmtree(work, ignore_errors=True)
            raise


def verify_admission(args: argparse.Namespace) -> dict[str, Any]:
    snapshot = _authenticate_inputs(args)
    raw_output = args.output.expanduser()
    _reject_direct_symlink(
        raw_output,
        label="calibration admission output",
    )
    output = _inside(
        Path(str(snapshot["project_root"])),
        raw_output.resolve(),
        label="calibration admission output",
    )
    if not output.is_dir() or output.is_symlink():
        raise CalibrationAdmissionError(
            f"calibration admission output is missing or a symlink: {output}"
        )
    expected_files = {
        ADMISSION_NAME,
        DASHBOARD_NAME,
        MANIFEST_NAME,
        COMPLETE_NAME,
    }
    _strict_bundle_inventory(output, expected_files=expected_files)

    plan = _build_plan(snapshot, output=output)
    admission_path = output / ADMISSION_NAME
    dashboard_path = output / DASHBOARD_NAME
    manifest_path = output / MANIFEST_NAME
    complete_path = output / COMPLETE_NAME
    recorded = _read_json(
        admission_path,
        label="calibration admission",
    )
    expected_record_keys = set(plan["_contract"]) | {
        "status",
        "accepted_at",
        "admission_fingerprint",
    }
    if (
        set(recorded) != expected_record_keys
        or any(recorded.get(key) != value for key, value in plan["_contract"].items())
        or recorded.get("status") != "accepted_explicit_user_acknowledgement"
        or recorded.get("admission_fingerprint") != plan["admission_fingerprint"]
    ):
        raise CalibrationAdmissionError(
            "calibration admission payload differs from the approved plan"
        )
    try:
        accepted_at = datetime.fromisoformat(str(recorded["accepted_at"]))
    except ValueError as exc:
        raise CalibrationAdmissionError("calibration admission accepted_at is invalid") from exc
    if accepted_at.utcoffset() is None:
        raise CalibrationAdmissionError("calibration admission accepted_at must be timezone-aware")

    manifest = _read_json(
        manifest_path,
        label="calibration admission MANIFEST",
    )
    unsigned_manifest = {
        key: value for key, value in manifest.items() if key != "bundle_fingerprint"
    }
    files = _mapping(
        manifest.get("files"),
        label="calibration admission MANIFEST.files",
    )
    expected_manifest_keys = {
        "schema_version",
        "kind",
        "admission_fingerprint",
        "input_fingerprint",
        "files",
        "authorized_dashboard",
        "authorizes_calibration_launch",
        "authorizes_formal_training",
        "training_started",
        "dashboard_profile_change_authorized",
        "bundle_fingerprint",
    }
    if (
        set(manifest) != expected_manifest_keys
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != BUNDLE_KIND
        or manifest.get("admission_fingerprint") != plan["admission_fingerprint"]
        or manifest.get("input_fingerprint") != plan["input_fingerprint"]
        or manifest.get("bundle_fingerprint") != _canonical_sha256(unsigned_manifest)
        or manifest.get("authorizes_calibration_launch") is not True
        or manifest.get("authorizes_formal_training") is not False
        or manifest.get("training_started") is not False
        or manifest.get("dashboard_profile_change_authorized") is not True
        or set(files) != {ADMISSION_NAME, DASHBOARD_NAME}
    ):
        raise CalibrationAdmissionError("calibration admission MANIFEST contract is invalid")
    for name in (ADMISSION_NAME, DASHBOARD_NAME):
        identity = _mapping(
            files.get(name),
            label=f"calibration admission MANIFEST.files.{name}",
        )
        path = output / name
        if (
            identity.get("path") != name
            or identity.get("size") != path.stat().st_size
            or identity.get("sha256") != sha256_file(path)
        ):
            raise CalibrationAdmissionError(f"calibration admission file identity differs: {name}")
    authorized_dashboard = _mapping(
        manifest.get("authorized_dashboard"),
        label="calibration admission authorized_dashboard",
    )
    if authorized_dashboard.get("path") != DASHBOARD_NAME or authorized_dashboard.get(
        "sha256"
    ) != sha256_file(dashboard_path):
        raise CalibrationAdmissionError("authorized Dashboard identity differs")

    complete = _read_json(
        complete_path,
        label="calibration admission COMPLETE",
    )
    expected_complete_keys = {
        "schema_version",
        "kind",
        "manifest",
        "manifest_sha256",
        "bundle_fingerprint",
        "admission_fingerprint",
        "authorizes_calibration_launch",
        "authorizes_formal_training",
        "training_started",
        "dashboard_profile_change_authorized",
    }
    if (
        set(complete) != expected_complete_keys
        or complete.get("schema_version") != SCHEMA_VERSION
        or complete.get("kind") != COMPLETE_KIND
        or complete.get("manifest") != MANIFEST_NAME
        or complete.get("manifest_sha256") != sha256_file(manifest_path)
        or complete.get("bundle_fingerprint") != manifest.get("bundle_fingerprint")
        or complete.get("admission_fingerprint") != plan["admission_fingerprint"]
        or complete.get("authorizes_calibration_launch") is not True
        or complete.get("authorizes_formal_training") is not False
        or complete.get("training_started") is not False
        or complete.get("dashboard_profile_change_authorized") is not True
    ):
        raise CalibrationAdmissionError("calibration admission COMPLETE contract is invalid")

    dashboard = _read_json(
        dashboard_path,
        label="authorized Dashboard",
    )
    expected_dashboard = _dashboard_with_admission(
        snapshot,
        output=output,
        admission_sha256=sha256_file(admission_path),
        admission_fingerprint=plan["admission_fingerprint"],
    )
    if dashboard != expected_dashboard:
        raise CalibrationAdmissionError(
            "authorized Dashboard was not derived exactly from authenticated inputs"
        )
    settings = load_dashboard_settings(dashboard_path)
    launchable = [profile for profile in settings.profiles if profile.launch_enabled]
    if (
        len(launchable) != 1
        or launchable[0].profile_id != PROFILE_ID
        or launchable[0].start_confirmation != f"START {PROFILE_ID}"
    ):
        raise CalibrationAdmissionError("authorized Dashboard launch allow-list differs")
    raw_profiles = dashboard.get("profiles")
    selected = [
        profile
        for profile in raw_profiles
        if isinstance(profile, Mapping) and profile.get("id") == PROFILE_ID
    ]
    if len(selected) != 1:
        raise CalibrationAdmissionError("authorized Dashboard calibration profile count differs")
    inline = _mapping(
        selected[0].get("calibration_admission"),
        label="authorized Dashboard calibration_admission",
    )
    expected_relative_output = output.relative_to(
        Path(str(snapshot["project_root"])).resolve()
    ).as_posix()
    if (
        inline.get("path") != expected_relative_output
        or inline.get("admission_sha256") != sha256_file(admission_path)
        or inline.get("admission_fingerprint") != plan["admission_fingerprint"]
        or inline.get("required_start_confirmation") != f"START {PROFILE_ID}"
        or inline.get("authorizes_formal_training") is not False
        or inline.get("training_started") is not False
    ):
        raise CalibrationAdmissionError("authorized Dashboard admission binding differs")
    _require_sources_unchanged()
    return {
        **_public_plan(plan),
        "verified": True,
        "accepted_at": recorded["accepted_at"],
        "admission": str(admission_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "complete": str(complete_path),
        "complete_sha256": sha256_file(complete_path),
        "dashboard_config": str(dashboard_path),
        "dashboard_config_sha256": sha256_file(dashboard_path),
        "dashboard_profile_changed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "plan":
            result = build_plan(args)
        elif args.action == "publish":
            result = publish_admission(args)
        else:
            result = verify_admission(args)
    except (
        CalibrationAdmissionError,
        OSError,
        ValueError,
        publisher.ReleaseError,
    ) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2
    print(
        json.dumps(
            _public_plan(result),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
