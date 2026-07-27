#!/usr/bin/env python3
"""Close v4 13M calibration evidence without authorizing formal training.

This reporting-only command consumes an already completed calibration run.  It
does not import the training engine, initialize CUDA, or start work.  It:

* authenticates the formal v4 data/formal closure;
* authenticates the training, frozen-validation, and drift report bundles
  against source-bound report kinds;
* fully authenticates every explicitly supplied checkpoint;
* reads every release claim from an inventoried JSON payload via an RFC 6901
  pointer;
* asks the publisher's production verifier to recompute every hard gate; and
* atomically writes ``attestation.json`` plus ``COMPLETE``.

The claim-location spec contains locations only, never trusted result values.
Its exact schema is::

    {
      "schema_version": 1,
      "kind": "twen_v4_13m_calibration_claim_locations",
      "claims": {
        "<required-claim-name>": {
          "bundle": "training_report_bundle | checkpoint_validation_bundle |
                     checkpoint_drift_audit_bundle",
          "path": "<inventoried JSON payload>",
          "json_pointer": "/rfc/6901/pointer"
        }
      },
      "spec_fingerprint": "<canonical SHA256 excluding this field>"
    }

Publication remains a separate step and requires the user's exact formal and
Wikipedia acknowledgements.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from twen.utils import sha256_file

SCHEMA_VERSION = 1
SPEC_KIND = "twen_v4_13m_calibration_claim_locations"
OUTPUT_NAME = "attestation.json"
COMPLETE_NAME = "COMPLETE"
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REPORT_KINDS = {
    "training_report_bundle": "twen_dense_training_analysis_bundle",
    "checkpoint_validation_bundle": "twen_v4_checkpoint_validation_sweep_bundle",
    "checkpoint_drift_audit_bundle": "twen_dense_optimizer_drift_audit_bundle",
}


class CalibrationClosureError(ValueError):
    """Calibration evidence cannot be safely closed."""


def _load_publisher() -> ModuleType:
    path = Path(__file__).resolve().with_name("publish_v4_250m_release.py")
    source_before = path.read_bytes()
    spec = importlib.util.spec_from_file_location(
        "_twen_v4_release_publisher_for_calibration_closure",
        path,
    )
    if spec is None or spec.loader is None:
        raise CalibrationClosureError(f"cannot load release verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if path.read_bytes() != source_before:
        raise CalibrationClosureError(
            "release verifier changed while calibration closure loaded"
        )
    return module


publisher = _load_publisher()
if EXPECTED_REPORT_KINDS != publisher.CALIBRATION_REPORT_KINDS:
    raise CalibrationClosureError(
        "calibration attestor/report publisher source policies differ"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--checkpoint-validation-report", type=Path, required=True)
    parser.add_argument("--checkpoint-drift-report", type=Path, required=True)
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        action="append",
        required=True,
        help=(
            "Fully authenticated calibration checkpoint; repeat in strictly "
            "increasing global-step order and include the final milestone"
        ),
    )
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--claim-locations", type=Path, required=True)
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


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationClosureError(
            f"cannot read {label} JSON {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise CalibrationClosureError(f"{label} must be a JSON object: {path}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise CalibrationClosureError(
            f"evidence file is missing or a symlink: {resolved}"
        )
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _report_binding(root: Path, *, expected_kind: str) -> dict[str, Any]:
    resolved = root.expanduser().resolve()
    manifest = resolved / "MANIFEST.json"
    complete = resolved / "COMPLETE"
    value = _read_json(manifest, label=f"{expected_kind} MANIFEST")
    if value.get("schema_version") != 1 or value.get("kind") != expected_kind:
        raise CalibrationClosureError(
            f"report bundle is not the source-bound kind {expected_kind}: {resolved}"
        )
    return {
        "path": str(resolved),
        "manifest_sha256": sha256_file(manifest),
        "complete_sha256": sha256_file(complete),
        "manifest_kind": expected_kind,
        # The three source-bound report producers use the plain MANIFEST
        # digest marker.  A typed replacement requires a reviewed source edit.
        "complete_kind": None,
    }


def _checkpoint_binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "manifest_sha256": sha256_file(resolved / "manifest.json"),
        "complete_sha256": sha256_file(resolved / "COMPLETE"),
    }


def _claim_locations(path: Path) -> dict[str, dict[str, str]]:
    value = _read_json(path.expanduser().resolve(), label="claim-location spec")
    unsigned = {
        key: item for key, item in value.items() if key != "spec_fingerprint"
    }
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != SPEC_KIND
        or value.get("spec_fingerprint") != _canonical_sha256(unsigned)
    ):
        raise CalibrationClosureError(
            "claim-location spec kind/schema/fingerprint is invalid"
        )
    raw_claims = value.get("claims")
    if not isinstance(raw_claims, Mapping):
        raise CalibrationClosureError("claim-location spec has no claims object")
    if set(raw_claims) != set(publisher.CLAIM_NAMES):
        raise CalibrationClosureError("claim-location inventory differs")
    claims: dict[str, dict[str, str]] = {}
    for name in publisher.CLAIM_NAMES:
        location = raw_claims[name]
        if not isinstance(location, Mapping) or set(location) != {
            "bundle",
            "path",
            "json_pointer",
        }:
            raise CalibrationClosureError(
                f"claim location has an invalid schema: {name}"
            )
        bundle = location.get("bundle")
        relative = location.get("path")
        pointer = location.get("json_pointer")
        if (
            bundle not in EXPECTED_REPORT_KINDS
            or not isinstance(relative, str)
            or not relative
            or not isinstance(pointer, str)
            or not pointer.startswith("/")
        ):
            raise CalibrationClosureError(
                f"claim location is invalid: {name}"
            )
        required_bundle, required_path = publisher.CLAIM_EVIDENCE_POLICY[name]
        if bundle != required_bundle or relative != required_path:
            raise CalibrationClosureError(
                f"claim location differs from source policy: {name}"
            )
        claims[name] = {
            "bundle": str(bundle),
            "path": relative,
            "json_pointer": pointer,
        }
    return claims


def _authenticate_closure(path: Path) -> dict[str, Any]:
    try:
        closure = publisher._authenticate_closure_structure(path)
        closure["governed"] = publisher._authenticate_governed_closure_gates(
            closure
        )
    except (publisher.ReleaseError, OSError, ValueError) as exc:
        raise CalibrationClosureError(
            f"formal closure authentication failed: {exc}"
        ) from exc
    return closure


def _build_attestation(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    closure = _authenticate_closure(args.closure)
    raw_roots = {
        "training_report_bundle": args.training_report,
        "checkpoint_validation_bundle": args.checkpoint_validation_report,
        "checkpoint_drift_audit_bundle": args.checkpoint_drift_report,
    }
    evidence = {
        name: _report_binding(
            raw_roots[name],
            expected_kind=EXPECTED_REPORT_KINDS[name],
        )
        for name in publisher.REPORT_BUNDLE_NAMES
    }
    # Authenticate complete inventories before extracting any value.
    authenticated_bundles = {
        name: publisher._authenticate_report_bundle(
            evidence[name],
            project_root=closure["project_root"],
            label=f"calibration evidence.{name}",
            report_name=name,
        )
        for name in publisher.REPORT_BUNDLE_NAMES
    }
    locations = _claim_locations(args.claim_locations)
    claims: dict[str, Any] = {}
    for name in publisher.CLAIM_NAMES:
        location = locations[name]
        bundle = authenticated_bundles[location["bundle"]]
        relative = location["path"]
        if relative not in bundle["payloads"]:
            raise CalibrationClosureError(
                f"claim {name} references an uninventoried payload"
            )
        payload = _read_json(
            bundle["payloads"][relative],
            label=f"claim {name} payload",
        )
        try:
            observed = publisher._json_pointer(
                payload,
                location["json_pointer"],
                label=f"claim {name} JSON pointer",
            )
        except publisher.ReleaseError as exc:
            raise CalibrationClosureError(str(exc)) from exc
        claims[name] = {
            "value": copy.deepcopy(observed),
            "evidence": copy.deepcopy(location),
        }

    candidate_bindings = [
        _checkpoint_binding(path) for path in args.candidate_checkpoint
    ]
    final_binding = _checkpoint_binding(args.final_checkpoint)
    calibration_gate = closure["readiness"]["calibration_gate"]
    calibration_config_path = publisher._resolve_path(
        calibration_gate["config"]["path"],
        base=closure["project_root"],
        label="calibration config",
    )
    attestation: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": publisher.CALIBRATION_KIND,
        "status": (
            "passed_authenticated_quality_gate_but_does_not_authorize_formal_training"
        ),
        "attestor": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "formal_closure": copy.deepcopy(closure["binding"]),
        "calibration_gate_contract_fingerprint": closure[
            "calibration_gate_contract_fingerprint"
        ],
        "calibration_config": _identity(calibration_config_path),
        "evidence": evidence,
        "claims": claims,
        "candidate_checkpoints": candidate_bindings,
        "final_checkpoint": final_binding,
        "passed": True,
        "authorizes_training": False,
        "training_started": False,
    }
    attestation["attestation_fingerprint"] = _canonical_sha256(attestation)
    return attestation, closure


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


def _public_result(output: Path, attestation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "output": str(output),
        "attestation": str(output / OUTPUT_NAME),
        "complete": str(output / COMPLETE_NAME),
        "attestation_fingerprint": attestation["attestation_fingerprint"],
        "passed": True,
        "authorizes_training": False,
        "training_started": False,
    }


def close_calibration_evidence(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise CalibrationClosureError(
            f"calibration closure output already exists: {output}"
        )
    first, closure = _build_attestation(args)
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise CalibrationClosureError(
            "calibration closure output parent must already be a real "
            f"directory: {output.parent}"
        )
    with publisher._directory_lock(output.parent):
        if output.exists():
            raise CalibrationClosureError(
                f"calibration closure output already exists: {output}"
            )
        work = Path(
            tempfile.mkdtemp(
                prefix=f".{output.name}.incomplete-",
                dir=output.parent,
            )
        )
        try:
            attestation_path = work / OUTPUT_NAME
            complete_path = work / COMPLETE_NAME
            _write_json(attestation_path, first)
            complete = {
                "schema_version": SCHEMA_VERSION,
                "kind": publisher.CALIBRATION_COMPLETE_KIND,
                "attestation": OUTPUT_NAME,
                "attestation_sha256": sha256_file(attestation_path),
                "attestation_fingerprint": first["attestation_fingerprint"],
                "passed": True,
                "authorizes_training": False,
                "training_started": False,
            }
            _write_json(complete_path, complete)

            # Run the exact consumer that formal publication uses.
            try:
                publisher._authenticate_calibration_attestation(
                    attestation_path,
                    closure=closure,
                )
            except (publisher.ReleaseError, OSError, ValueError) as exc:
                raise CalibrationClosureError(
                    f"generated calibration attestation failed release validation: {exc}"
                ) from exc
            second, _second_closure = _build_attestation(args)
            if _canonical_sha256(first) != _canonical_sha256(second):
                raise CalibrationClosureError(
                    "calibration evidence changed during closure"
                )
            work_fd = os.open(work, os.O_RDONLY)
            try:
                os.fsync(work_fd)
            finally:
                os.close(work_fd)
            if output.exists():
                raise CalibrationClosureError(
                    f"calibration closure output appeared during publication: {output}"
                )
            try:
                publisher._rename_directory_noreplace(work, output)
            except publisher.ReleaseError as exc:
                raise CalibrationClosureError(str(exc)) from exc
            parent_fd = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
    return _public_result(output, first)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = close_calibration_evidence(args)
    except (
        CalibrationClosureError,
        publisher.ReleaseError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
