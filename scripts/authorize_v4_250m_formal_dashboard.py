#!/usr/bin/env python3
"""Plan, publish, or verify the v4 250M formal Dashboard admission.

The formal release publisher intentionally does not modify the Web allow-list.
This independent command authenticates that release and the existing four-file
13M calibration admission, closes every prior Dashboard profile, and writes a
new atomic bundle whose only launchable profile is the governed 250M run.

Publication requires the exact ``RUN <plan-id>`` acknowledgement.  It records
that authorization but never starts training.  The generated bundle can be
verified after the governed run directory exists, which keeps service restarts
fail-closed without confusing publication provenance with runtime state.
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
import os
import shutil
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from twen.governed import (
    GovernedControllerError,
    authorize_run,
    build_governed_plan,
    expected_run_ack,
)
from twen.utils import sha256_file
from twen.web import DashboardError, load_dashboard_settings

SCHEMA_VERSION = 1
PROFILE_ID = "base-dense-v4-250m-pilot"
CALIBRATION_PROFILE_ID = "base-dense-v4-13m-low-lr-calibration"
ADMISSION_KIND = "twen_v4_250m_formal_dashboard_admission"
PLAN_KIND = "twen_v4_250m_formal_dashboard_admission_plan"
BUNDLE_KIND = "twen_v4_250m_formal_dashboard_admission_bundle"
COMPLETE_KIND = "twen_v4_250m_formal_dashboard_admission_complete"
ADMISSION_NAME = "admission.json"
DASHBOARD_NAME = "dashboard.json"
MANIFEST_NAME = "MANIFEST.json"
COMPLETE_NAME = "COMPLETE"
FINAL_CONFIG_NAME = "dense-v4-250m-pilot.yaml"
FINAL_READINESS_NAME = "readiness.json"
AUTHORIZED_LABEL = "Base Dense v4 250M (formal governed release)"
EXPECTED_RUN_DIR = "runs/base-dense-v4-250m-pilot"
EXPECTED_STATE_PATH = (
    "runs/.base-dense-v4-250m-pilot.governed/controller-state.json"
)
EXPECTED_CALIBRATION_ADMISSION_PATH = (
    "locks/base-dense-v4-13m-calibration-admission-pass-002"
)
ROOT = Path(__file__).resolve().parents[1]


class FormalDashboardAdmissionError(ValueError):
    """The formal Dashboard admission cannot be authenticated or published."""


def _load_calibration_authorizer() -> tuple[ModuleType, str]:
    path = Path(__file__).resolve().with_name(
        "authorize_v4_13m_calibration.py"
    )
    source_before = path.read_bytes()
    spec = importlib.util.spec_from_file_location(
        "_twen_v4_calibration_authorizer_for_formal_dashboard",
        path,
    )
    if spec is None or spec.loader is None:
        raise FormalDashboardAdmissionError(
            f"cannot load calibration admission verifier: {path}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if path.read_bytes() != source_before:
        raise FormalDashboardAdmissionError(
            "calibration admission verifier changed while it was loaded"
        )
    return module, hashlib.sha256(source_before).hexdigest()


calibration, CALIBRATION_AUTHORIZER_SHA256_AT_IMPORT = (
    _load_calibration_authorizer()
)
AUTHORIZER_SHA256_AT_IMPORT = sha256_file(Path(__file__).resolve())
GOVERNED_CONTROLLER_PATH_AT_IMPORT = (
    ROOT / "scripts/govern_v4_training.py"
).resolve()
GOVERNED_CONTROLLER_SHA256_AT_IMPORT = sha256_file(
    GOVERNED_CONTROLLER_PATH_AT_IMPORT
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "publish", "verify"))
    parser.add_argument(
        "--release",
        type=Path,
        required=True,
        help="Final four-file formal release bundle",
    )
    parser.add_argument(
        "--calibration-admission",
        type=Path,
        default=Path(EXPECTED_CALIBRATION_ADMISSION_PATH),
        help="Existing authenticated 13M calibration admission bundle",
    )
    parser.add_argument("--profile-id", default=PROFILE_ID)
    parser.add_argument(
        "--run-ack",
        help="publish only: exact `RUN <plan-id>` acknowledgement",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New formal Dashboard admission directory",
    )
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
        raise FormalDashboardAdmissionError(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        raise FormalDashboardAdmissionError(f"{label} key inventory differs")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise FormalDashboardAdmissionError(
            f"{label} is missing or a symlink: {expanded}"
        )
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise FormalDashboardAdmissionError(
            f"{label} is missing or a symlink: {resolved}"
        )
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormalDashboardAdmissionError(
            f"cannot read {label} JSON {resolved}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise FormalDashboardAdmissionError(f"{label} must be a JSON object")
    return value


def _identity(path: Path) -> dict[str, Any]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise FormalDashboardAdmissionError(
            f"authenticated file is missing or a symlink: {expanded}"
        )
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise FormalDashboardAdmissionError(
            f"authenticated file is missing or a symlink: {resolved}"
        )
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _reject_direct_symlink(path: Path, *, label: str) -> None:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise FormalDashboardAdmissionError(
            f"{label} must not be a symlink: {expanded}"
        )


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path.expanduser()))


def _inside(root: Path, path: Path, *, label: str) -> Path:
    resolved_root = root.expanduser().resolve()
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise FormalDashboardAdmissionError(
            f"{label} must remain inside {resolved_root}: {resolved}"
        ) from exc
    return resolved


def _project_relative(project_root: Path, path: Path, *, label: str) -> str:
    resolved = _inside(project_root, path, label=label)
    return resolved.relative_to(project_root.resolve()).as_posix()


def _strict_bundle_files(
    root: Path,
    *,
    expected_files: set[str],
    label: str,
) -> Path:
    expanded = root.expanduser()
    if expanded.is_symlink():
        raise FormalDashboardAdmissionError(
            f"{label} is missing or a symlink: {expanded}"
        )
    resolved = expanded.resolve()
    if not resolved.is_dir():
        raise FormalDashboardAdmissionError(
            f"{label} is missing or a symlink: {resolved}"
        )
    entries = list(resolved.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise FormalDashboardAdmissionError(f"{label} contains a symlink")
    observed_files = {
        path.relative_to(resolved).as_posix()
        for path in entries
        if path.is_file()
    }
    if observed_files != expected_files:
        raise FormalDashboardAdmissionError(
            f"{label} file inventory differs"
        )
    if any(path.is_dir() for path in entries):
        raise FormalDashboardAdmissionError(
            f"{label} must not contain subdirectories"
        )
    return resolved


def _require_sources_unchanged() -> None:
    calibration_path = Path(calibration.__file__).resolve()
    controller_path = (ROOT / "scripts/govern_v4_training.py").resolve()
    if (
        sha256_file(Path(__file__).resolve()) != AUTHORIZER_SHA256_AT_IMPORT
        or sha256_file(calibration_path)
        != CALIBRATION_AUTHORIZER_SHA256_AT_IMPORT
        or controller_path != GOVERNED_CONTROLLER_PATH_AT_IMPORT
        or sha256_file(controller_path)
        != GOVERNED_CONTROLLER_SHA256_AT_IMPORT
    ):
        raise FormalDashboardAdmissionError(
            "formal authorizer, calibration verifier, or governed controller "
            "changed after import"
        )


def _current_calibration_admission(
    admission_root: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    _reject_direct_symlink(
        admission_root,
        label="calibration admission",
    )
    expected_root = (
        project_root / EXPECTED_CALIBRATION_ADMISSION_PATH
    ).resolve()
    root = _inside(
        project_root,
        admission_root,
        label="calibration admission",
    )
    if root != expected_root:
        raise FormalDashboardAdmissionError(
            "calibration admission is not the current fixed bundle"
        )
    _strict_bundle_files(
        root,
        expected_files={
            calibration.ADMISSION_NAME,
            calibration.DASHBOARD_NAME,
            calibration.MANIFEST_NAME,
            calibration.COMPLETE_NAME,
        },
        label="calibration admission bundle",
    )
    recorded = _read_json(
        root / calibration.ADMISSION_NAME,
        label="calibration admission",
    )
    authenticated = _mapping(
        recorded.get("authenticated_inputs"),
        label="calibration admission authenticated_inputs",
    )
    closure = _mapping(
        authenticated.get("formal_closure"),
        label="calibration admission formal_closure",
    )
    dashboard_before = _mapping(
        authenticated.get("dashboard_before"),
        label="calibration admission dashboard_before",
    )
    closure_path = closure.get("path")
    dashboard_path = dashboard_before.get("path")
    acknowledgement = recorded.get("acknowledgement")
    if not all(
        isinstance(value, str) and value
        for value in (closure_path, dashboard_path, acknowledgement)
    ):
        raise FormalDashboardAdmissionError(
            "calibration admission verifier inputs are incomplete"
        )
    verify_args = argparse.Namespace(
        action="verify",
        closure=Path(str(closure_path)),
        dashboard_config=Path(str(dashboard_path)),
        profile_id=CALIBRATION_PROFILE_ID,
        acknowledgement=str(acknowledgement),
        output=root,
    )
    try:
        verified = calibration.verify_admission(verify_args)
    except (
        calibration.CalibrationAdmissionError,
        calibration.publisher.ReleaseError,
        OSError,
        ValueError,
    ) as exc:
        raise FormalDashboardAdmissionError(
            f"calibration admission authentication failed: {exc}"
        ) from exc

    admission_path = root / calibration.ADMISSION_NAME
    dashboard_bundle_path = root / calibration.DASHBOARD_NAME
    manifest_path = root / calibration.MANIFEST_NAME
    complete_path = root / calibration.COMPLETE_NAME
    manifest = _read_json(
        manifest_path,
        label="calibration admission MANIFEST",
    )
    complete = _read_json(
        complete_path,
        label="calibration admission COMPLETE",
    )
    dashboard = _read_json(
        dashboard_bundle_path,
        label="calibration admission Dashboard",
    )
    admission_fingerprint = recorded.get("admission_fingerprint")
    if (
        verified.get("verified") is not True
        or verified.get("admission_fingerprint") != admission_fingerprint
        or verified.get("dashboard_config")
        != str(dashboard_bundle_path.resolve())
        or recorded.get("kind") != calibration.ADMISSION_KIND
        or recorded.get("authorizes_formal_training") is not False
        or recorded.get("training_started") is not False
        or manifest.get("kind") != calibration.BUNDLE_KIND
        or manifest.get("admission_fingerprint") != admission_fingerprint
        or manifest.get("authorizes_formal_training") is not False
        or manifest.get("training_started") is not False
        or complete.get("kind") != calibration.COMPLETE_KIND
        or complete.get("admission_fingerprint") != admission_fingerprint
        or complete.get("authorizes_formal_training") is not False
        or complete.get("training_started") is not False
    ):
        raise FormalDashboardAdmissionError(
            "calibration admission authorization boundary differs"
        )
    try:
        settings = load_dashboard_settings(dashboard_bundle_path)
    except DashboardError as exc:
        raise FormalDashboardAdmissionError(
            f"calibration Dashboard is invalid: {exc}"
        ) from exc
    launchable = [
        profile.profile_id
        for profile in settings.profiles
        if profile.launch_enabled
    ]
    if (
        settings.project_root != project_root.resolve()
        or launchable != [CALIBRATION_PROFILE_ID]
    ):
        raise FormalDashboardAdmissionError(
            "current calibration Dashboard allow-list differs"
        )
    return {
        "root": str(root),
        "admission": _identity(admission_path),
        "dashboard": _identity(dashboard_bundle_path),
        "manifest": _identity(manifest_path),
        "complete": _identity(complete_path),
        "admission_fingerprint": admission_fingerprint,
        "bundle_fingerprint": manifest.get("bundle_fingerprint"),
        "authorizes_formal_training": False,
        "training_started": False,
        "_dashboard": dashboard,
    }


def _formal_release(
    release_root: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    _reject_direct_symlink(release_root, label="formal release")
    root = _inside(project_root, release_root, label="formal release")
    _strict_bundle_files(
        root,
        expected_files={
            FINAL_CONFIG_NAME,
            FINAL_READINESS_NAME,
            MANIFEST_NAME,
            COMPLETE_NAME,
        },
        label="formal release bundle",
    )
    readiness_path = root / FINAL_READINESS_NAME
    config_path = root / FINAL_CONFIG_NAME
    try:
        plan = build_governed_plan(readiness_path)
        required_ack = expected_run_ack(plan)
        authorize_run(plan, required_ack)
    except (GovernedControllerError, OSError, ValueError) as exc:
        raise FormalDashboardAdmissionError(
            f"formal release authentication failed: {exc}"
        ) from exc

    release_bundle = _mapping(
        plan.get("release_bundle"),
        label="governed plan release_bundle",
    )
    run = _mapping(plan.get("run"), label="governed plan run")
    config = _mapping(plan.get("config"), label="governed plan config")
    readiness = _mapping(
        plan.get("readiness"),
        label="governed plan readiness",
    )
    source_tree = _mapping(
        plan.get("source_tree"),
        label="governed plan source_tree",
    )
    dependency_lock = _mapping(
        plan.get("dependency_lock"),
        label="governed plan dependency_lock",
    )
    controller_path = (
        project_root / "scripts/govern_v4_training.py"
    ).resolve()
    controllers = [
        item
        for item in plan.get("controller_sources", [])
        if isinstance(item, Mapping)
        and Path(str(item.get("path"))).resolve() == controller_path
        and item.get("sha256") == sha256_file(controller_path)
    ]
    expected_run_dir = (project_root / EXPECTED_RUN_DIR).resolve()
    if (
        plan.get("launch_enabled") is not True
        or plan.get("readiness_issues") != []
        or release_bundle.get("root") != str(root)
        or Path(str(config.get("path"))).resolve() != config_path
        or config.get("sha256") != sha256_file(config_path)
        or Path(str(readiness.get("path"))).resolve() != readiness_path
        or readiness.get("sha256") != sha256_file(readiness_path)
        or run.get("run_id") != PROFILE_ID
        or run.get("stage") != "dense-oracle"
        or Path(str(run.get("output_dir"))).resolve() != expected_run_dir
        or len(controllers) != 1
    ):
        raise FormalDashboardAdmissionError(
            "governed formal release plan differs from the fixed 250M profile"
        )

    manifest = _read_json(
        root / MANIFEST_NAME,
        label="formal release MANIFEST",
    )
    complete = _read_json(
        root / COMPLETE_NAME,
        label="formal release COMPLETE",
    )
    readiness_value = _read_json(
        readiness_path,
        label="formal release readiness",
    )
    authorization_values = (
        manifest.get("authorizes_training"),
        complete.get("authorizes_training"),
        readiness_value.get("authorizes_training"),
    )
    if (
        authorization_values != (True, True, True)
        or manifest.get("training_started") is not False
        or complete.get("training_started") is not False
        or readiness_value.get("training_started") is not False
    ):
        raise FormalDashboardAdmissionError(
            "formal release is not the sole unstarted training authorization"
        )
    controller_identity = _identity(controller_path)
    return {
        "root": str(root),
        "manifest": _identity(root / MANIFEST_NAME),
        "complete": _identity(root / COMPLETE_NAME),
        "config": _identity(config_path),
        "readiness": _identity(readiness_path),
        "release_fingerprint": release_bundle.get("release_fingerprint"),
        "bundle_fingerprint": release_bundle.get("bundle_fingerprint"),
        "readiness_fingerprint": release_bundle.get(
            "readiness_fingerprint"
        ),
        "governed_plan_id": plan.get("plan_id"),
        "required_run_ack": required_ack,
        "governed_controller": controller_identity,
        "source_tree": copy.deepcopy(source_tree),
        "dependency_lock": copy.deepcopy(dependency_lock),
        "release_authentication": copy.deepcopy(release_bundle),
        "authorizes_training": True,
        "training_started": False,
        "_plan": plan,
    }


def _authenticate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    _require_sources_unchanged()
    if args.profile_id != PROFILE_ID:
        raise FormalDashboardAdmissionError(
            f"profile_id must equal {PROFILE_ID!r}"
        )
    project_root = ROOT.resolve()
    release = _formal_release(
        args.release.expanduser(),
        project_root=project_root,
    )
    current = _current_calibration_admission(
        args.calibration_admission.expanduser(),
        project_root=project_root,
    )
    dashboard = copy.deepcopy(current.pop("_dashboard"))
    raw_profiles = dashboard.get("profiles")
    if not isinstance(raw_profiles, list):
        raise FormalDashboardAdmissionError(
            "current Dashboard has no profiles list"
        )
    if any(
        isinstance(item, Mapping) and item.get("id") == PROFILE_ID
        for item in raw_profiles
    ):
        raise FormalDashboardAdmissionError(
            "current Dashboard already contains the formal profile"
        )
    plan = _mapping(release.get("_plan"), label="formal governed plan")
    release_public = {
        key: copy.deepcopy(value)
        for key, value in release.items()
        if not str(key).startswith("_")
    }
    profile_contract = {
        "id": PROFILE_ID,
        "label": AUTHORIZED_LABEL,
        "config": _project_relative(
            project_root,
            Path(str(release["config"]["path"])),
            label="formal config",
        ),
        "config_sha256": release["config"]["sha256"],
        "fork_from": None,
        "launch_enabled": True,
        "launch_kind": "governed_v4",
        "resume": "none",
        "governed_controller": _project_relative(
            project_root,
            Path(str(release["governed_controller"]["path"])),
            label="governed controller",
        ),
        "governed_controller_sha256": release[
            "governed_controller"
        ]["sha256"],
        "governed_readiness": _project_relative(
            project_root,
            Path(str(release["readiness"]["path"])),
            label="formal readiness",
        ),
        "governed_readiness_sha256": release["readiness"]["sha256"],
        "governed_state": EXPECTED_STATE_PATH,
        "governed_plan_id": plan["plan_id"],
        "run_dir": str((project_root / EXPECTED_RUN_DIR).resolve()),
        "required_run_ack": release["required_run_ack"],
    }
    authenticated_inputs = {
        "formal_release": release_public,
        "current_calibration_admission": copy.deepcopy(current),
        "source_identity": {
            "formal_dashboard_authorizer": {
                **_identity(Path(__file__).resolve()),
                "import_sha256": AUTHORIZER_SHA256_AT_IMPORT,
            },
            "calibration_admission_verifier": {
                **_identity(Path(calibration.__file__).resolve()),
                "import_sha256": (
                    CALIBRATION_AUTHORIZER_SHA256_AT_IMPORT
                ),
            },
            "governed_controller": copy.deepcopy(
                release["governed_controller"]
            ),
            "source_tree": copy.deepcopy(release["source_tree"]),
            "dependency_lock": copy.deepcopy(release["dependency_lock"]),
        },
        "authorized_profile_contract": profile_contract,
        "training_authorization": {
            "source": "formal_release_bundle",
            "release_fingerprint": release["release_fingerprint"],
            "bundle_fingerprint": release["bundle_fingerprint"],
            "readiness_fingerprint": release["readiness_fingerprint"],
            "authorizes_training": release["authorizes_training"],
        },
    }
    _require_sources_unchanged()
    return {
        "project_root": project_root,
        "dashboard": dashboard,
        "profile_id": PROFILE_ID,
        "run_dir": Path(profile_contract["run_dir"]),
        "state_path": (
            project_root / EXPECTED_STATE_PATH
        ).resolve(),
        "required_run_ack": release["required_run_ack"],
        "authenticated_inputs": authenticated_inputs,
        "input_fingerprint": _canonical_sha256(authenticated_inputs),
    }


def _require_formal_not_started(snapshot: Mapping[str, Any]) -> None:
    for path, label in (
        (Path(str(snapshot["run_dir"])), "formal run directory"),
        (Path(str(snapshot["state_path"])), "formal controller state"),
    ):
        _reject_direct_symlink(path, label=label)
        if _lexists(path):
            raise FormalDashboardAdmissionError(
                f"{label} already exists: {path}"
            )


def _resolve_output(
    snapshot: Mapping[str, Any],
    raw_output: Path,
    *,
    must_exist: bool,
) -> Path:
    _reject_direct_symlink(
        raw_output,
        label="formal Dashboard admission output",
    )
    output = _inside(
        Path(str(snapshot["project_root"])),
        raw_output.resolve(),
        label="formal Dashboard admission output",
    )
    if must_exist:
        if not output.is_dir() or output.is_symlink():
            raise FormalDashboardAdmissionError(
                f"formal Dashboard admission output is missing or a symlink: "
                f"{output}"
            )
    elif _lexists(output):
        raise FormalDashboardAdmissionError(
            f"formal Dashboard admission output already exists: {output}"
        )
    return output


def _build_plan(
    snapshot: Mapping[str, Any],
    *,
    output: Path,
) -> dict[str, Any]:
    authorization = _mapping(
        snapshot["authenticated_inputs"].get("training_authorization"),
        label="formal release training authorization",
    )
    if (
        authorization.get("source") != "formal_release_bundle"
        or authorization.get("authorizes_training") is not True
    ):
        raise FormalDashboardAdmissionError(
            "training authorization does not originate solely from the release"
        )
    contract = {
        "schema_version": SCHEMA_VERSION,
        "kind": ADMISSION_KIND,
        "input_fingerprint": snapshot["input_fingerprint"],
        "authenticated_inputs": copy.deepcopy(
            snapshot["authenticated_inputs"]
        ),
        "acknowledgement": snapshot["required_run_ack"],
        "output": str(output),
        "authorizes_training": authorization["authorizes_training"],
        "training_authorization_source": "formal_release_bundle",
        "training_started": False,
        "dashboard_profile_change_authorized": True,
    }
    admission_fingerprint = _canonical_sha256(contract)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": PLAN_KIND,
        "admission_fingerprint": admission_fingerprint,
        "input_fingerprint": snapshot["input_fingerprint"],
        "output": str(output),
        "required_run_ack": snapshot["required_run_ack"],
        "governed_plan_id": snapshot["authenticated_inputs"][
            "authorized_profile_contract"
        ]["governed_plan_id"],
        "authorizes_training": True,
        "training_authorization_source": "formal_release_bundle",
        "training_started": False,
        "dashboard_profile_change_authorized": True,
        "_contract": contract,
    }


def _public_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in plan.items()
        if not str(key).startswith("_")
    }


def _require_exact_run_ack(
    observed: str | None,
    expected: str,
) -> None:
    if observed != expected:
        raise FormalDashboardAdmissionError(
            f"explicit acknowledgement must equal {expected!r}"
        )


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    snapshot = _authenticate_inputs(args)
    _require_formal_not_started(snapshot)
    output = _resolve_output(snapshot, args.output.expanduser(), must_exist=False)
    if args.run_ack is not None:
        _require_exact_run_ack(args.run_ack, snapshot["required_run_ack"])
    return _build_plan(snapshot, output=output)


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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _directory_lock(
    path: Path,
    *,
    timeout_seconds: float = 300.0,
) -> Iterator[None]:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            try:
                fcntl.flock(
                    descriptor,
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise FormalDashboardAdmissionError(
                        "timed out waiting for formal admission parent lock"
                    ) from exc
                time.sleep(0.1)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _rename_directory_noreplace(
    source: Path,
    destination: Path,
) -> None:
    renameat2 = getattr(
        ctypes.CDLL(None, use_errno=True),
        "renameat2",
        None,
    )
    if renameat2 is None:
        raise FormalDashboardAdmissionError(
            "renameat2(RENAME_NOREPLACE) is unavailable"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FormalDashboardAdmissionError(
            f"formal Dashboard admission output appeared during publication: "
            f"{destination}"
        )
    if error_number in {errno.ENOSYS, errno.EINVAL}:
        raise FormalDashboardAdmissionError(
            "atomic no-replace directory installation is unsupported"
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(destination),
    )


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
    profiles = dashboard.get("profiles")
    if not isinstance(profiles, list):
        raise FormalDashboardAdmissionError(
            "current Dashboard profiles changed"
        )
    if any(
        isinstance(profile, Mapping) and profile.get("id") == PROFILE_ID
        for profile in profiles
    ):
        raise FormalDashboardAdmissionError(
            "current Dashboard already contains the formal profile"
        )
    for profile in profiles:
        if not isinstance(profile, dict):
            raise FormalDashboardAdmissionError(
                "current Dashboard profile is invalid"
            )
        profile["launch_enabled"] = False
    formal_profile = copy.deepcopy(
        snapshot["authenticated_inputs"]["authorized_profile_contract"]
    )
    formal_profile.pop("run_dir", None)
    formal_profile.pop("required_run_ack", None)
    formal_profile["formal_dashboard_admission"] = {
        "path": output.relative_to(project_root).as_posix(),
        "admission_sha256": admission_sha256,
        "admission_fingerprint": admission_fingerprint,
        "release_fingerprint": snapshot["authenticated_inputs"][
            "training_authorization"
        ]["release_fingerprint"],
        "governed_plan_id": formal_profile["governed_plan_id"],
        "required_run_ack": snapshot["required_run_ack"],
        "authorizes_training": True,
        "training_authorization_source": "formal_release_bundle",
        "training_started": False,
    }
    profiles.append(formal_profile)
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
    _write_bytes(dashboard_path, payload)
    try:
        settings = load_dashboard_settings(dashboard_path)
    except DashboardError as exc:
        raise FormalDashboardAdmissionError(
            f"derived formal Dashboard is invalid: {exc}"
        ) from exc
    launchable = [
        profile
        for profile in settings.profiles
        if profile.launch_enabled
    ]
    if (
        len(launchable) != 1
        or launchable[0].profile_id != PROFILE_ID
        or launchable[0].launch_kind != "governed_v4"
        or launchable[0].start_confirmation
        != f"RUN {launchable[0].governed_plan_id}"
    ):
        raise FormalDashboardAdmissionError(
            "derived Dashboard does not enable exactly the governed formal "
            "profile"
        )
    return hashlib.sha256(payload).hexdigest()


def _verify_bundle(
    snapshot: Mapping[str, Any],
    *,
    bundle_root: Path,
    logical_output: Path,
) -> dict[str, Any]:
    root = _strict_bundle_files(
        bundle_root,
        expected_files={
            ADMISSION_NAME,
            DASHBOARD_NAME,
            MANIFEST_NAME,
            COMPLETE_NAME,
        },
        label="formal Dashboard admission bundle",
    )
    plan = _build_plan(snapshot, output=logical_output)
    admission_path = root / ADMISSION_NAME
    dashboard_path = root / DASHBOARD_NAME
    manifest_path = root / MANIFEST_NAME
    complete_path = root / COMPLETE_NAME
    recorded = _read_json(
        admission_path,
        label="formal Dashboard admission",
    )
    expected_record_keys = set(plan["_contract"]) | {
        "status",
        "accepted_at",
        "admission_fingerprint",
    }
    if (
        set(recorded) != expected_record_keys
        or any(
            recorded.get(key) != value
            for key, value in plan["_contract"].items()
        )
        or recorded.get("status")
        != "accepted_exact_governed_run_acknowledgement"
        or recorded.get("admission_fingerprint")
        != plan["admission_fingerprint"]
    ):
        raise FormalDashboardAdmissionError(
            "formal Dashboard admission payload differs from the plan"
        )
    try:
        accepted_at = datetime.fromisoformat(str(recorded["accepted_at"]))
    except ValueError as exc:
        raise FormalDashboardAdmissionError(
            "formal Dashboard admission accepted_at is invalid"
        ) from exc
    if accepted_at.utcoffset() is None:
        raise FormalDashboardAdmissionError(
            "formal Dashboard admission accepted_at must be timezone-aware"
        )

    manifest = _read_json(
        manifest_path,
        label="formal Dashboard admission MANIFEST",
    )
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "kind",
            "admission_fingerprint",
            "input_fingerprint",
            "files",
            "authorized_dashboard",
            "authorizes_training",
            "training_authorization_source",
            "training_started",
            "dashboard_profile_change_authorized",
            "bundle_fingerprint",
        },
        label="formal Dashboard admission MANIFEST",
    )
    unsigned_manifest = {
        key: value
        for key, value in manifest.items()
        if key != "bundle_fingerprint"
    }
    files = _mapping(
        manifest.get("files"),
        label="formal Dashboard admission MANIFEST.files",
    )
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != BUNDLE_KIND
        or manifest.get("admission_fingerprint")
        != plan["admission_fingerprint"]
        or manifest.get("input_fingerprint")
        != plan["input_fingerprint"]
        or manifest.get("bundle_fingerprint")
        != _canonical_sha256(unsigned_manifest)
        or manifest.get("authorizes_training") is not True
        or manifest.get("training_authorization_source")
        != "formal_release_bundle"
        or manifest.get("training_started") is not False
        or manifest.get("dashboard_profile_change_authorized") is not True
        or set(files) != {ADMISSION_NAME, DASHBOARD_NAME}
    ):
        raise FormalDashboardAdmissionError(
            "formal Dashboard admission MANIFEST contract is invalid"
        )
    for name in (ADMISSION_NAME, DASHBOARD_NAME):
        identity = _mapping(
            files.get(name),
            label=f"formal Dashboard admission MANIFEST.files.{name}",
        )
        _require_exact_keys(
            identity,
            {"path", "size", "sha256"},
            label=f"formal Dashboard admission MANIFEST.files.{name}",
        )
        path = root / name
        if (
            identity.get("path") != name
            or identity.get("size") != path.stat().st_size
            or identity.get("sha256") != sha256_file(path)
        ):
            raise FormalDashboardAdmissionError(
                f"formal Dashboard admission file identity differs: {name}"
            )
    authorized_dashboard = _mapping(
        manifest.get("authorized_dashboard"),
        label="formal Dashboard admission authorized_dashboard",
    )
    _require_exact_keys(
        authorized_dashboard,
        {"path", "sha256"},
        label="formal Dashboard admission authorized_dashboard",
    )
    if (
        authorized_dashboard.get("path") != DASHBOARD_NAME
        or authorized_dashboard.get("sha256")
        != sha256_file(dashboard_path)
    ):
        raise FormalDashboardAdmissionError(
            "authorized formal Dashboard identity differs"
        )

    complete = _read_json(
        complete_path,
        label="formal Dashboard admission COMPLETE",
    )
    _require_exact_keys(
        complete,
        {
            "schema_version",
            "kind",
            "manifest",
            "manifest_sha256",
            "bundle_fingerprint",
            "admission_fingerprint",
            "authorizes_training",
            "training_authorization_source",
            "training_started",
            "dashboard_profile_change_authorized",
        },
        label="formal Dashboard admission COMPLETE",
    )
    if (
        complete.get("schema_version") != SCHEMA_VERSION
        or complete.get("kind") != COMPLETE_KIND
        or complete.get("manifest") != MANIFEST_NAME
        or complete.get("manifest_sha256") != sha256_file(manifest_path)
        or complete.get("bundle_fingerprint")
        != manifest.get("bundle_fingerprint")
        or complete.get("admission_fingerprint")
        != plan["admission_fingerprint"]
        or complete.get("authorizes_training") is not True
        or complete.get("training_authorization_source")
        != "formal_release_bundle"
        or complete.get("training_started") is not False
        or complete.get("dashboard_profile_change_authorized") is not True
    ):
        raise FormalDashboardAdmissionError(
            "formal Dashboard admission COMPLETE contract is invalid"
        )

    dashboard = _read_json(
        dashboard_path,
        label="authorized formal Dashboard",
    )
    expected_dashboard = _dashboard_with_admission(
        snapshot,
        output=logical_output,
        admission_sha256=sha256_file(admission_path),
        admission_fingerprint=plan["admission_fingerprint"],
    )
    if _canonical_bytes(dashboard) != _canonical_bytes(
        expected_dashboard
    ):
        raise FormalDashboardAdmissionError(
            "authorized formal Dashboard was not derived exactly from the "
            "authenticated calibration admission"
        )
    try:
        settings = load_dashboard_settings(dashboard_path)
    except DashboardError as exc:
        raise FormalDashboardAdmissionError(
            f"authorized formal Dashboard is invalid: {exc}"
        ) from exc
    launchable = [
        profile
        for profile in settings.profiles
        if profile.launch_enabled
    ]
    governed_profiles = [
        profile
        for profile in settings.profiles
        if profile.launch_kind == "governed_v4"
    ]
    if (
        settings.project_root
        != Path(str(snapshot["project_root"])).resolve()
        or len(launchable) != 1
        or launchable[0].profile_id != PROFILE_ID
        or launchable[0].launch_kind != "governed_v4"
        or launchable[0].governed_plan_id
        != plan["governed_plan_id"]
        or launchable[0].start_confirmation != plan["required_run_ack"]
        or len(governed_profiles) != 1
        or governed_profiles[0].profile_id != PROFILE_ID
    ):
        raise FormalDashboardAdmissionError(
            "authorized formal Dashboard launch allow-list differs"
        )
    raw_profiles = dashboard.get("profiles")
    if not isinstance(raw_profiles, list):
        raise FormalDashboardAdmissionError(
            "authorized formal Dashboard profiles are invalid"
        )
    selected = [
        profile
        for profile in raw_profiles
        if isinstance(profile, Mapping)
        and profile.get("id") == PROFILE_ID
    ]
    if len(selected) != 1 or any(
        isinstance(profile, Mapping)
        and profile.get("id") != PROFILE_ID
        and profile.get("launch_enabled") is not False
        for profile in raw_profiles
    ):
        raise FormalDashboardAdmissionError(
            "authorized formal Dashboard profile inventory differs"
        )
    raw_profile = selected[0]
    expected_profile = snapshot["authenticated_inputs"][
        "authorized_profile_contract"
    ]
    for key, value in expected_profile.items():
        if key not in {"run_dir", "required_run_ack"} and raw_profile.get(
            key
        ) != value:
            raise FormalDashboardAdmissionError(
                f"authorized formal Dashboard profile differs: {key}"
            )
    inline = _mapping(
        raw_profile.get("formal_dashboard_admission"),
        label="authorized formal Dashboard inline admission",
    )
    _require_exact_keys(
        inline,
        {
            "path",
            "admission_sha256",
            "admission_fingerprint",
            "release_fingerprint",
            "governed_plan_id",
            "required_run_ack",
            "authorizes_training",
            "training_authorization_source",
            "training_started",
        },
        label="authorized formal Dashboard inline admission",
    )
    expected_relative_output = logical_output.relative_to(
        Path(str(snapshot["project_root"])).resolve()
    ).as_posix()
    authorization = snapshot["authenticated_inputs"][
        "training_authorization"
    ]
    if (
        inline.get("path") != expected_relative_output
        or inline.get("admission_sha256") != sha256_file(admission_path)
        or inline.get("admission_fingerprint")
        != plan["admission_fingerprint"]
        or inline.get("release_fingerprint")
        != authorization["release_fingerprint"]
        or inline.get("governed_plan_id") != plan["governed_plan_id"]
        or inline.get("required_run_ack") != plan["required_run_ack"]
        or inline.get("authorizes_training") is not True
        or inline.get("training_authorization_source")
        != "formal_release_bundle"
        or inline.get("training_started") is not False
    ):
        raise FormalDashboardAdmissionError(
            "authorized formal Dashboard admission binding differs"
        )
    _require_sources_unchanged()
    return {
        **_public_plan(plan),
        "verified": True,
        "accepted_at": recorded["accepted_at"],
        "admission": str(logical_output / ADMISSION_NAME),
        "manifest": str(logical_output / MANIFEST_NAME),
        "manifest_sha256": sha256_file(manifest_path),
        "complete": str(logical_output / COMPLETE_NAME),
        "complete_sha256": sha256_file(complete_path),
        "dashboard_config": str(logical_output / DASHBOARD_NAME),
        "dashboard_config_sha256": sha256_file(dashboard_path),
    }


def publish_admission(args: argparse.Namespace) -> dict[str, Any]:
    snapshot = _authenticate_inputs(args)
    _require_formal_not_started(snapshot)
    _require_exact_run_ack(args.run_ack, snapshot["required_run_ack"])
    raw_output = args.output.expanduser()
    _reject_direct_symlink(
        raw_output.parent,
        label="formal Dashboard admission output parent",
    )
    output = _resolve_output(snapshot, raw_output, must_exist=False)
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise FormalDashboardAdmissionError(
            "formal Dashboard admission output parent must already be a "
            "real directory"
        )
    plan = _build_plan(snapshot, output=output)

    work: Path | None = None
    with _directory_lock(output.parent):
        if _lexists(output):
            raise FormalDashboardAdmissionError(
                f"formal Dashboard admission output already exists: {output}"
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
                "status": "accepted_exact_governed_run_acknowledgement",
                "accepted_at": accepted_at,
                "admission_fingerprint": plan["admission_fingerprint"],
            }
            admission_path = work / ADMISSION_NAME
            dashboard_path = work / DASHBOARD_NAME
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
                dashboard_path,
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
                        "size": dashboard_path.stat().st_size,
                        "sha256": sha256_file(dashboard_path),
                    },
                },
                "authorized_dashboard": {
                    "path": DASHBOARD_NAME,
                    "sha256": dashboard_sha256,
                },
                "authorizes_training": True,
                "training_authorization_source": "formal_release_bundle",
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
                "authorizes_training": True,
                "training_authorization_source": "formal_release_bundle",
                "training_started": False,
                "dashboard_profile_change_authorized": True,
            }
            _write_json(complete_path, complete)
            for path in (
                admission_path,
                dashboard_path,
                manifest_path,
                complete_path,
            ):
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            _fsync_directory(work)
            _verify_bundle(
                snapshot,
                bundle_root=work,
                logical_output=output,
            )

            second_snapshot = _authenticate_inputs(args)
            _require_formal_not_started(second_snapshot)
            if (
                second_snapshot["input_fingerprint"]
                != snapshot["input_fingerprint"]
            ):
                raise FormalDashboardAdmissionError(
                    "formal Dashboard admission inputs changed during "
                    "publication"
                )
            if _lexists(output):
                raise FormalDashboardAdmissionError(
                    "formal Dashboard admission output appeared during "
                    "publication"
                )
            _rename_directory_noreplace(work, output)
            work = None
            _fsync_directory(output.parent)
            return {
                **_public_plan(plan),
                "publication_performed": True,
                "accepted_at": accepted_at,
                "admission": str(output / ADMISSION_NAME),
                "manifest": str(output / MANIFEST_NAME),
                "manifest_sha256": sha256_file(output / MANIFEST_NAME),
                "complete": str(output / COMPLETE_NAME),
                "complete_sha256": sha256_file(output / COMPLETE_NAME),
                "dashboard_config": str(output / DASHBOARD_NAME),
                "dashboard_config_sha256": dashboard_sha256,
            }
        except BaseException:
            if work is not None:
                shutil.rmtree(work, ignore_errors=True)
            raise


def verify_admission(args: argparse.Namespace) -> dict[str, Any]:
    snapshot = _authenticate_inputs(args)
    if args.run_ack is not None:
        _require_exact_run_ack(args.run_ack, snapshot["required_run_ack"])
    output = _resolve_output(
        snapshot,
        args.output.expanduser(),
        must_exist=True,
    )
    return _verify_bundle(
        snapshot,
        bundle_root=output,
        logical_output=output,
    )


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
        DashboardError,
        FormalDashboardAdmissionError,
        GovernedControllerError,
        OSError,
        ValueError,
        calibration.CalibrationAdmissionError,
        calibration.publisher.ReleaseError,
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
