#!/usr/bin/env python3
"""Close authenticated v4 250M data/formal evidence without authorizing training.

The command is deliberately reporting-only.  It revalidates both finalized
phase corpora, their passing audit attestations, governed train and validation
prepared manifests, phase separation, formal train/validation separation, and
the v3-final formal baseline report.  It then atomically writes independent
capacity/readiness records while keeping the blocked config and all launch
controls unchanged.

It never loads a model, touches CUDA, starts calibration, or starts training.
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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from twen.data.audits import validate_base_audit_attestation
from twen.data.cursor import AuthenticatedSourceMap
from twen.data.prepared import validate_prepared_corpus
from twen.data.sources import validate_extracted_base_corpus
from twen.io.download import sha256_file
from twen.io.locking import FileLock
from twen.source_identity import twen_source_tree_sha256
from twen.utils import atomic_write_json

SCHEMA_VERSION = 1
BUNDLE_KIND = "twen_v4_250m_formal_evidence_closure_bundle"
COMPLETE_KIND = "twen_v4_250m_formal_evidence_closure_complete"
CLOSURE_KIND = "twen_v4_250m_formal_evidence_closure"
CAPACITY_KIND = "twen_v4_250m_capacity_attestation"
READINESS_KIND = "twen_v4_250m_pilot_readiness"
FORMAL_BASELINE_KIND = "twen_v4_formal_frozen_validation_baseline"
FORMAL_BASELINE_BUNDLE_KIND = "twen_v4_formal_frozen_validation_baseline_bundle"
PHASES = ("primary", "cooldown")
REQUIRED_NEAR_DUPLICATE_THRESHOLD = 0.8
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPACITY_TEMPLATE = ROOT / "locks/base-data-sources-v4-250m.capacity-attestation.json"
DEFAULT_READINESS_TEMPLATE = ROOT / "locks/base-dense-v4-250m-pilot.readiness.json"


class ClosureError(ValueError):
    """One or more formal evidence closure conditions did not pass."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-template", type=Path, default=DEFAULT_CAPACITY_TEMPLATE)
    parser.add_argument("--readiness-template", type=Path, default=DEFAULT_READINESS_TEMPLATE)
    for phase in PHASES:
        parser.add_argument(f"--{phase}-extracted", type=Path, required=True)
        parser.add_argument(f"--{phase}-audit", type=Path, required=True)
        parser.add_argument(f"--{phase}-train-prepared", type=Path, required=True)
        parser.add_argument(f"--{phase}-validation-prepared", type=Path, required=True)
    parser.add_argument("--phase-disjointness-attestation", type=Path, required=True)
    parser.add_argument("--formal-validation-disjointness-attestation", type=Path, required=True)
    parser.add_argument("--formal-baseline-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClosureError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ClosureError(f"{label} must be a JSON object: {path}")
    return value


def _identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ClosureError(f"evidence file does not exist: {resolved}")
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _same_file_identity(actual: Mapping[str, object], expected: Mapping[str, object]) -> bool:
    try:
        actual_path = Path(str(actual.get("path"))).resolve()
        expected_path = Path(str(expected.get("path"))).resolve()
    except (OSError, ValueError):
        return False
    return (
        actual_path == expected_path
        and actual.get("size") == expected.get("size")
        and actual.get("sha256") == expected.get("sha256")
    )


def _load_script(filename: str, module_name: str) -> ModuleType:
    path = Path(__file__).resolve().with_name(filename)
    source_before = path.read_bytes()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ClosureError(f"cannot load evidence validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if path.read_bytes() != source_before:
        raise ClosureError(f"evidence validator changed while loading: {path}")
    return module


def _resolve_repo_path(raw: object, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ClosureError(f"{label} must be a non-empty path")
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _require_template_policy(
    capacity_path: Path,
    readiness_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, object]]:
    capacity_file = capacity_path.resolve()
    readiness_file = readiness_path.resolve()
    capacity = _read_json(capacity_file, label="capacity template")
    readiness = _read_json(readiness_file, label="readiness template")
    if (
        capacity.get("schema_version") != SCHEMA_VERSION
        or capacity.get("kind") != CAPACITY_KIND
        or capacity.get("launch_enabled") is not False
        or capacity.get("authorizes_training") is not False
        or capacity.get("training_started") is not False
    ):
        raise ClosureError("capacity template is not the launch-disabled v4 contract")
    if (
        readiness.get("schema_version") != SCHEMA_VERSION
        or readiness.get("kind") != READINESS_KIND
        or readiness.get("launch_enabled") is not False
        or readiness.get("training_started") is not False
        or readiness.get("launch_command_after_all_gates_pass") is not None
    ):
        raise ClosureError("readiness template is not launch-disabled")
    config = capacity.get("config")
    if not isinstance(config, Mapping) or config.get("contains_pending_identity_sentinels") is not True:
        raise ClosureError("capacity template does not bind the blocked PENDING config")
    config_path = _resolve_repo_path(config.get("path"), label="capacity.config.path")
    config_identity = _identity(config_path)
    if config.get("sha256") != config_identity["sha256"]:
        raise ClosureError("capacity template blocked-config SHA256 is stale")
    config_bytes = config_path.read_bytes()
    required_sentinels = (
        b"PENDING_PRIMARY_PREPARED_MANIFEST",
        b"PENDING_COOLDOWN_PREPARED_MANIFEST",
        b"PENDING_PRIMARY_COOLDOWN_PHASE_DISJOINTNESS_ATTESTATION",
    )
    if not all(sentinel in config_bytes for sentinel in required_sentinels):
        raise ClosureError("blocked config no longer contains every required PENDING sentinel")
    if (
        readiness.get("config_path") != config.get("path")
        or readiness.get("config_sha256") != config_identity["sha256"]
    ):
        raise ClosureError("capacity/readiness templates bind different blocked configs")
    required_capacity = _resolve_repo_path(
        readiness.get("required_capacity_attestation"),
        label="readiness.required_capacity_attestation",
    )
    if required_capacity != capacity_file:
        raise ClosureError("readiness template does not bind the supplied capacity template")
    calibration = readiness.get("calibration_gate")
    if (
        not isinstance(calibration, Mapping)
        or calibration.get("required") is not True
        or calibration.get("passed") is not False
        or calibration.get("authorizes_training") is not False
        or calibration.get("observed") is not None
    ):
        raise ClosureError("readiness calibration gate is not pending and fail-closed")
    pause = readiness.get("pause_evaluation_policy")
    if (
        not isinstance(pause, Mapping)
        or pause.get("enforcement") != "external_governed_controller"
        or pause.get("controller_implemented") is not False
        or pause.get("current_launch_command_auto_pauses") is not False
        or pause.get("current_launch_command_runs_validation") is not False
    ):
        raise ClosureError("readiness pause/evaluation controller is not pending")
    capabilities = readiness.get("launch_command_capabilities")
    if (
        not isinstance(capabilities, Mapping)
        or capabilities.get("current_blocked_config_rejects_training") is not True
        or any(
            capabilities.get(field) is not False
            for field in (
                "starts_training_when_explicitly_invoked",
                "automatically_pauses_at_policy_thresholds",
                "automatically_runs_checkpoint_validation",
                "automatically_enforces_post_launch_hard_stops",
            )
        )
    ):
        raise ClosureError("readiness launch capabilities are not fail-closed")
    return capacity, readiness, config_identity


def _governed_prepared(
    path: Path,
    *,
    role: str,
    extracted_path: Path,
    extracted_sha256: str,
    audit_path: Path,
    audit_sha256: str,
) -> tuple[Any, AuthenticatedSourceMap, dict[str, object], dict[str, int]]:
    manifest_path = path.resolve()
    prepared = validate_prepared_corpus(manifest_path)
    lineage = prepared.lineage
    bound_as = "candidate" if role == "train" else "frozen_validation"
    audit_lineage = lineage.get("audit_attestation") if isinstance(lineage, Mapping) else None
    if (
        not isinstance(lineage, Mapping)
        or lineage.get("kind") != "authenticated_extracted_corpus"
        or lineage.get("role") != role
        or lineage.get("ready_for_training") is not True
        or lineage.get("research_only") is not False
        or lineage.get("pending_audits") != []
        or Path(str(lineage.get("extracted_manifest_path"))).resolve() != extracted_path
        or lineage.get("extracted_manifest_sha256") != extracted_sha256
        or not isinstance(audit_lineage, Mapping)
        or Path(str(audit_lineage.get("path"))).resolve() != audit_path
        or audit_lineage.get("sha256") != audit_sha256
        or audit_lineage.get("bound_as") != bound_as
        or audit_lineage.get("ready_for_training") is not True
    ):
        raise ClosureError(f"{role} prepared manifest is outside governed phase lineage")
    source_map = AuthenticatedSourceMap.from_prepared_manifest(prepared)
    entries = {entry.shard_id: entry for entry in prepared.shards}
    source_tokens = {
        source_id: sum(
            entries[shard.shard_id].token_count
            for shard in source_map.shards_for_source(source_id)
        )
        for source_id in source_map.source_ids
    }
    if sum(source_tokens.values()) != prepared.token_count:
        raise ClosureError(f"{role} prepared source-map token inventory differs")
    identity = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_fingerprint": prepared.dataset_fingerprint,
        "source_map_sha256": source_map.fingerprint,
        "token_count": prepared.token_count,
        "sequence_count": prepared.sequence_count,
        "available_unique_tokens": prepared.token_count,
        "available_unique_samples": prepared.sequence_count,
        "sequence_length": prepared.sequence_length,
    }
    return prepared, source_map, identity, source_tokens


def _phase_evidence(
    *,
    phase: str,
    extracted_path: Path,
    audit_path: Path,
    train_prepared_path: Path,
    validation_prepared_path: Path,
) -> dict[str, Any]:
    extracted_file = extracted_path.resolve()
    audit_file = audit_path.resolve()
    validate_extracted_base_corpus(extracted_file, verify_hashes=True)
    extracted = _read_json(extracted_file, label=f"{phase} extracted manifest")
    extracted_sha = sha256_file(extracted_file)
    if (
        extracted.get("kind") != "twen_extracted_base_jsonl_corpus"
        or extracted.get("ready_for_data_prepare") is not True
    ):
        raise ClosureError(f"{phase} extracted corpus is not prepare-ready")
    audit = validate_base_audit_attestation(audit_file)
    candidate = audit.get("candidate")
    frozen = audit.get("frozen_validation")
    if (
        audit.get("ready_for_training") is not True
        or not isinstance(candidate, Mapping)
        or not isinstance(frozen, Mapping)
        or candidate.get("role") != "train"
        or frozen.get("role") != "validation"
        or Path(str(candidate.get("manifest_path"))).resolve() != extracted_file
        or Path(str(frozen.get("manifest_path"))).resolve() != extracted_file
        or candidate.get("manifest_sha256") != extracted_sha
        or frozen.get("manifest_sha256") != extracted_sha
    ):
        raise ClosureError(f"{phase} audit does not pass for both roles of one finalized corpus")
    audit_sha = sha256_file(audit_file)
    train, train_map, train_identity, source_tokens = _governed_prepared(
        train_prepared_path,
        role="train",
        extracted_path=extracted_file,
        extracted_sha256=extracted_sha,
        audit_path=audit_file,
        audit_sha256=audit_sha,
    )
    validation, validation_map, validation_identity, validation_source_tokens = (
        _governed_prepared(
            validation_prepared_path,
            role="validation",
            extracted_path=extracted_file,
            extracted_sha256=extracted_sha,
            audit_path=audit_file,
            audit_sha256=audit_sha,
        )
    )
    if set(train_map.source_ids) != set(validation_map.source_ids):
        raise ClosureError(f"{phase} train/validation source coverage differs")
    if any(tokens <= 0 for tokens in validation_source_tokens.values()):
        raise ClosureError(f"{phase} validation contains an empty source")
    license_audit = extracted.get("license_audit")
    attribution = (
        license_audit.get("attribution_inventory")
        if isinstance(license_audit, Mapping)
        else None
    )
    if (
        not isinstance(license_audit, Mapping)
        or license_audit.get("complete") is not True
        or not isinstance(attribution, Mapping)
    ):
        raise ClosureError(f"{phase} extracted corpus has no complete attribution inventory")
    raw_attribution_path = attribution.get("path")
    if (
        not isinstance(raw_attribution_path, str)
        or not raw_attribution_path
        or Path(raw_attribution_path).is_absolute()
        or ".." in Path(raw_attribution_path).parts
    ):
        raise ClosureError(f"{phase} attribution inventory path is unsafe")
    attribution_path = extracted_file.parent / raw_attribution_path
    attribution_identity = _identity(attribution_path)
    if (
        attribution.get("size") != attribution_identity["size"]
        or attribution.get("sha256") != attribution_identity["sha256"]
    ):
        raise ClosureError(f"{phase} attribution inventory identity differs")
    complete_identity = _identity(extracted_file.parent / "COMPLETE")
    return {
        "phase": phase,
        "extracted": {
            "manifest_path": str(extracted_file),
            "manifest_sha256": extracted_sha,
            "corpus_fingerprint": extracted.get("corpus_fingerprint"),
            "complete": complete_identity,
        },
        "audit": {
            "path": str(audit_file),
            "sha256": audit_sha,
            "attestation_fingerprint": audit.get("attestation_fingerprint"),
            "ready_for_training": True,
        },
        "train": train,
        "train_source_map": train_map,
        "train_prepared": train_identity,
        "source_tokens": source_tokens,
        "validation": validation,
        "validation_source_map": validation_map,
        "validation_prepared": validation_identity,
        "validation_source_tokens": validation_source_tokens,
        "attribution": attribution_identity,
    }


def _validate_formal_baseline_bundle(
    root: Path,
    *,
    phases: Mapping[str, Mapping[str, Any]],
    validation_disjointness_path: Path,
) -> dict[str, Any]:
    bundle = root.resolve()
    if not bundle.is_dir():
        raise ClosureError(f"formal baseline bundle does not exist: {bundle}")
    summary_path = bundle / "summary.json"
    manifest_path = bundle / "MANIFEST.json"
    complete_path = bundle / "COMPLETE"
    manifest = _read_json(manifest_path, label="formal baseline MANIFEST")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != FORMAL_BASELINE_BUNDLE_KIND
    ):
        raise ClosureError("formal baseline bundle kind/schema differs")
    if complete_path.read_text(encoding="ascii").strip() != sha256_file(manifest_path):
        raise ClosureError("formal baseline COMPLETE does not authenticate MANIFEST")
    files = manifest.get("files")
    required_files = {
        "summary.json",
        "REPORT.zh-CN.md",
        "charts/formal-source-nll.svg",
        "charts/formal-source-tokens.svg",
    }
    if not isinstance(files, Mapping) or set(files) != required_files:
        raise ClosureError("formal baseline bundle payload inventory differs")
    for relative, raw_identity in files.items():
        if not isinstance(raw_identity, Mapping):
            raise ClosureError(f"formal baseline file identity is invalid: {relative}")
        actual = _identity(bundle / relative)
        if (
            raw_identity.get("path") != relative
            or raw_identity.get("size") != actual["size"]
            or raw_identity.get("sha256") != actual["sha256"]
        ):
            raise ClosureError(f"formal baseline payload identity differs: {relative}")
    summary = _read_json(summary_path, label="formal baseline summary")
    gate = summary.get("gate")
    if (
        summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("kind") != FORMAL_BASELINE_KIND
        or not isinstance(gate, Mapping)
        or gate.get("passed") is not True
        or gate.get("authorizes_training") is not False
        or gate.get("training_started_by_summarizer") is not False
        or manifest.get("gate") != gate
    ):
        raise ClosureError("formal baseline gate is incomplete or training-authorizing")
    phase_rows = summary.get("phases")
    if not isinstance(phase_rows, list):
        raise ClosureError("formal baseline phase inventory is invalid")
    rows_by_phase = {
        str(row.get("phase")): row for row in phase_rows if isinstance(row, Mapping)
    }
    if set(rows_by_phase) != set(PHASES):
        raise ClosureError("formal baseline does not contain exactly both phases")
    for phase in PHASES:
        expected = phases[phase]["validation_prepared"]
        actual = rows_by_phase[phase].get("prepared")
        if not isinstance(actual, Mapping):
            raise ClosureError(f"formal baseline has no {phase} prepared identity")
        for field in (
            "manifest_sha256",
            "dataset_fingerprint",
            "token_count",
            "sequence_count",
        ):
            actual_field = "sha256" if field == "manifest_sha256" else field
            if actual.get(actual_field) != expected.get(field):
                raise ClosureError(f"formal baseline {phase} prepared {field} differs")
        if Path(str(actual.get("path"))).resolve() != Path(
            str(expected["manifest_path"])
        ).resolve():
            raise ClosureError(f"formal baseline {phase} prepared path differs")
    formal = _load_script(
        "summarize_v4_formal_validation.py",
        "_twen_v4_formal_closure_reporter",
    )
    legacy = summary.get("legacy_six_source_baseline")
    legacy_identity = legacy.get("identity") if isinstance(legacy, Mapping) else None
    legacy_summary = (
        legacy_identity.get("summary") if isinstance(legacy_identity, Mapping) else None
    )
    if not isinstance(legacy_summary, Mapping):
        raise ClosureError("formal baseline has no authenticated legacy summary")
    evaluation_roots: dict[str, Path] = {}
    for phase in PHASES:
        evaluation = rows_by_phase[phase].get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise ClosureError(f"formal baseline has no {phase} evaluation identity")
        evaluation_roots[phase] = Path(str(evaluation.get("root"))).resolve()
    rebuilt = formal.build_summary(
        primary_prepared=Path(str(phases["primary"]["validation_prepared"]["manifest_path"])),
        primary_evaluation=evaluation_roots["primary"],
        cooldown_prepared=Path(str(phases["cooldown"]["validation_prepared"]["manifest_path"])),
        cooldown_evaluation=evaluation_roots["cooldown"],
        validation_disjointness_attestation=validation_disjointness_path,
        legacy_summary=Path(str(legacy_summary.get("path"))),
    )
    if rebuilt != summary:
        raise ClosureError("formal baseline summary differs from current authenticated inputs")
    legacy_baseline = summary.get("legacy_six_source_baseline")
    checkpoint = (
        legacy_baseline.get("checkpoint_artifact")
        if isinstance(legacy_baseline, Mapping)
        else None
    )
    if not isinstance(checkpoint, Mapping):
        raise ClosureError("formal baseline has no v3-final checkpoint artifact")
    return {
        "root": str(bundle),
        "summary": _identity(summary_path),
        "manifest": _identity(manifest_path),
        "complete": _identity(complete_path),
        "checkpoint_complete_sha256": checkpoint.get("complete_sha256"),
        "gate": copy.deepcopy(gate),
    }


def _explicit_paths(args: argparse.Namespace, phase: str) -> dict[str, Path]:
    return {
        "extracted": getattr(args, f"{phase}_extracted"),
        "audit": getattr(args, f"{phase}_audit"),
        "train_prepared": getattr(args, f"{phase}_train_prepared"),
        "validation_prepared": getattr(args, f"{phase}_validation_prepared"),
    }


def _authenticate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    capacity, readiness, blocked_config = _require_template_policy(
        args.capacity_template,
        args.readiness_template,
    )
    phases = {
        phase: _phase_evidence(phase=phase, **{
            f"{name}_path": path for name, path in _explicit_paths(args, phase).items()
        })
        for phase in PHASES
    }
    formal_disjoint = _load_script(
        "attest_v4_formal_validation_disjointness.py",
        "_twen_v4_formal_closure_disjointness",
    )
    formal_path = args.formal_validation_disjointness_attestation.resolve()
    try:
        formal_value = formal_disjoint.validate_formal_validation_disjointness_attestation(
            formal_path
        )
    except (OSError, ValueError) as exc:
        raise ClosureError(f"formal validation disjointness failed authentication: {exc}") from exc
    formal_phases = formal_value.get("phases")
    if not isinstance(formal_phases, Mapping):
        raise ClosureError("formal validation disjointness has no phase identities")
    for phase in PHASES:
        identity = formal_phases.get(phase)
        evidence = phases[phase]
        if (
            not isinstance(identity, Mapping)
            or Path(str(identity.get("manifest_path"))).resolve()
            != Path(str(evidence["extracted"]["manifest_path"])).resolve()
            or identity.get("manifest_sha256") != evidence["extracted"]["manifest_sha256"]
            or Path(str(identity.get("audit_attestation_path"))).resolve()
            != Path(str(evidence["audit"]["path"])).resolve()
            or identity.get("audit_attestation_sha256") != evidence["audit"]["sha256"]
        ):
            raise ClosureError(f"formal validation disjointness differs from explicit {phase} inputs")
        for role, field in (
            ("train", "prepared"),
            ("validation", "validation_prepared"),
        ):
            prepared_identity = identity.get(field)
            expected = evidence[f"{role}_prepared"]
            if (
                not isinstance(prepared_identity, Mapping)
                or Path(str(prepared_identity.get("manifest_path"))).resolve()
                != Path(str(expected["manifest_path"])).resolve()
                or prepared_identity.get("manifest_sha256") != expected["manifest_sha256"]
                or prepared_identity.get("dataset_fingerprint") != expected["dataset_fingerprint"]
            ):
                raise ClosureError(
                    f"formal validation disjointness differs from {phase} {role} prepared"
                )
    phase_identity = formal_value.get("phase_train_disjointness")
    phase_path = args.phase_disjointness_attestation.resolve()
    if (
        not isinstance(phase_identity, Mapping)
        or Path(str(phase_identity.get("path"))).resolve() != phase_path
        or phase_identity.get("sha256") != sha256_file(phase_path)
    ):
        raise ClosureError("formal validation disjointness binds another phase attestation")
    phase_value = _read_json(phase_path, label="phase disjointness attestation")
    phase_gates = phase_value.get("gates")
    if (
        phase_value.get("passed") is not True
        or not isinstance(phase_gates, Mapping)
        or set(phase_gates)
        != {"stable_id_exact", "normalized_text_exact", "near_duplicate"}
    ):
        raise ClosureError("phase disjointness is not passing")
    baseline = _validate_formal_baseline_bundle(
        args.formal_baseline_bundle,
        phases=phases,
        validation_disjointness_path=formal_path,
    )
    expected_checkpoint_sha = (
        readiness.get("formal_validation_gate", {})
        .get("v3_final_frozen_validation_baseline", {})
        .get("checkpoint_complete_sha256")
    )
    if (
        not isinstance(expected_checkpoint_sha, str)
        or baseline["checkpoint_complete_sha256"] != expected_checkpoint_sha
        or readiness.get("fork_policy", {}).get("required_checkpoint_complete_sha256")
        != expected_checkpoint_sha
    ):
        raise ClosureError("formal baseline checkpoint is not the required v3 final")
    return {
        "capacity_template": capacity,
        "capacity_template_identity": _identity(args.capacity_template),
        "readiness_template": readiness,
        "readiness_template_identity": _identity(args.readiness_template),
        "blocked_config": blocked_config,
        "phases": phases,
        "phase_disjointness": {
            "identity": _identity(phase_path),
            "attestation_fingerprint": phase_value.get("attestation_fingerprint"),
            "gates": copy.deepcopy(phase_gates),
        },
        "formal_validation_disjointness": {
            "identity": _identity(formal_path),
            "attestation_fingerprint": formal_value.get("attestation_fingerprint"),
            "near_duplicate_threshold": formal_value.get("near_duplicate_threshold"),
            "gates": copy.deepcopy(formal_value.get("gates")),
            "passed": formal_value.get("passed"),
        },
        "formal_baseline": baseline,
        "closure_source_sha256": sha256_file(Path(__file__).resolve()),
        "twen_source_tree_sha256": twen_source_tree_sha256(),
    }


def _public_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    phases: dict[str, Any] = {}
    for phase in PHASES:
        evidence = snapshot["phases"][phase]
        phases[phase] = {
            key: copy.deepcopy(evidence[key])
            for key in (
                "phase",
                "extracted",
                "audit",
                "train_prepared",
                "source_tokens",
                "validation_prepared",
                "validation_source_tokens",
                "attribution",
            )
        }
    return {
        "capacity_template_identity": copy.deepcopy(snapshot["capacity_template_identity"]),
        "readiness_template_identity": copy.deepcopy(snapshot["readiness_template_identity"]),
        "blocked_config": copy.deepcopy(snapshot["blocked_config"]),
        "phases": phases,
        "phase_disjointness": copy.deepcopy(snapshot["phase_disjointness"]),
        "formal_validation_disjointness": copy.deepcopy(
            snapshot["formal_validation_disjointness"]
        ),
        "formal_baseline": copy.deepcopy(snapshot["formal_baseline"]),
        "closure_source_sha256": snapshot["closure_source_sha256"],
        "twen_source_tree_sha256": snapshot["twen_source_tree_sha256"],
    }


def _closed_capacity(
    snapshot: Mapping[str, Any],
    *,
    closure_identity: Mapping[str, object],
) -> dict[str, Any]:
    result = copy.deepcopy(snapshot["capacity_template"])
    stages = result.get("stages")
    if not isinstance(stages, dict) or set(stages) != set(PHASES):
        raise ClosureError("capacity template phase inventory differs")
    total_available = 0
    for phase in PHASES:
        stage = stages[phase]
        evidence = snapshot["phases"][phase]
        if not isinstance(stage, dict):
            raise ClosureError(f"capacity template {phase} stage is invalid")
        rows = stage.get("per_source_capacity")
        if not isinstance(rows, list) or not rows:
            raise ClosureError(f"capacity template {phase} source inventory is invalid")
        source_tokens = evidence["source_tokens"]
        expected_sources = {
            str(row.get("source_id"))
            for row in rows
            if isinstance(row, Mapping)
        }
        if (
            len(expected_sources) != len(rows)
            or set(source_tokens) != expected_sources
            or set(evidence["validation_source_tokens"]) != expected_sources
        ):
            raise ClosureError(f"{phase} governed source coverage differs from capacity contract")
        expected_weights = {
            str(row["source_id"]): int(row["mix_basis_points"])
            for row in rows
            if isinstance(row, Mapping)
        }
        if evidence["train_source_map"].source_mix_weights != expected_weights:
            raise ClosureError(f"{phase} prepared source mix differs from capacity contract")
        closed_rows: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ClosureError(f"{phase} capacity source row is invalid")
            closed = dict(row)
            source_id = str(closed["source_id"])
            required = int(closed["required_clean_tokens"])
            available = int(source_tokens[source_id])
            if available < required:
                raise ClosureError(
                    f"{phase}/{source_id} governed capacity underfill: "
                    f"{available} < {required}"
                )
            closed.update(
                {
                    "available_governed_unique_tokens": available,
                    "margin_tokens": available - required,
                    "passed": True,
                }
            )
            closed_rows.append(closed)
        train_identity = copy.deepcopy(evidence["train_prepared"])
        required_tokens = int(stage["required_prepared_tokens"])
        required_samples = int(stage["required_prepared_samples"])
        available_tokens = int(train_identity["available_unique_tokens"])
        available_samples = int(train_identity["available_unique_samples"])
        if available_tokens < required_tokens or available_samples < required_samples:
            raise ClosureError(
                f"{phase} prepared capacity underfill: "
                f"{available_tokens}/{available_samples} < "
                f"{required_tokens}/{required_samples}"
            )
        train_identity.update(
            {
                "margin_tokens": available_tokens - required_tokens,
                "passed": True,
            }
        )
        stage.update(
            {
                "extracted_identity": copy.deepcopy(evidence["extracted"]),
                "prepared_identity": train_identity,
                "validation_prepared_identity": copy.deepcopy(
                    evidence["validation_prepared"]
                ),
                "per_source_capacity": closed_rows,
                "quality_audit": {
                    "audit_attestation_path": evidence["audit"]["path"],
                    "audit_attestation_sha256": evidence["audit"]["sha256"],
                    "attestation_fingerprint": evidence["audit"][
                        "attestation_fingerprint"
                    ],
                    "passed": True,
                },
                "license_audit": {
                    "materialized_attribution_manifest_path": evidence["attribution"][
                        "path"
                    ],
                    "materialized_attribution_manifest_sha256": evidence["attribution"][
                        "sha256"
                    ],
                    "passed": True,
                },
                "passed": True,
            }
        )
        total_available += available_tokens
    phase_evidence = snapshot["phase_disjointness"]
    gates = phase_evidence["gates"]
    result["phase_disjointness"] = {
        name: {
            "algorithm": gates[name]["algorithm"],
            **(
                {
                    "threshold": gates[name]["estimated_jaccard_threshold"],
                }
                if name == "near_duplicate"
                else {}
            ),
            "result": gates[name]["matches"],
            "passed": gates[name]["passed"] is True and gates[name]["matches"] == 0,
        }
        for name in ("stable_id_exact", "normalized_text_exact", "near_duplicate")
    }
    if not all(gate["passed"] for gate in result["phase_disjointness"].values()):
        raise ClosureError("phase disjointness gates did not close")
    result["phase_disjointness_attestation"] = {
        **copy.deepcopy(phase_evidence["identity"]),
        "attestation_fingerprint": phase_evidence["attestation_fingerprint"],
        "passed": True,
    }
    required_overall = int(result["overall"]["required_clean_tokens"])
    if total_available < required_overall:
        raise ClosureError("overall governed prepared capacity is insufficient")
    result["overall"] = {
        "required_clean_tokens": required_overall,
        "available_clean_tokens": total_available,
        "margin_tokens": total_available - required_overall,
        "passed": True,
    }
    result.update(
        {
            "status": "data_and_formal_evidence_closed_launch_blocked",
            "launch_enabled": False,
            "authorizes_training": False,
            "training_started": False,
            "closure": copy.deepcopy(closure_identity),
        }
    )
    unsigned = dict(result)
    result["attestation_fingerprint"] = _canonical_sha256(unsigned)
    return result


def _remaining_blockers(readiness: Mapping[str, Any]) -> list[str]:
    blockers = readiness.get("blockers")
    if not isinstance(blockers, list) or not all(isinstance(item, str) for item in blockers):
        raise ClosureError("readiness blocker inventory is invalid")
    retained_fragments = (
        "PENDING sentinels",
        "13M low-LR calibration",
        "final launch config",
        "Chinese semantic conversion noise",
    )
    retained = [
        blocker
        for blocker in blockers
        if any(fragment in blocker for fragment in retained_fragments)
    ]
    if len(retained) != len(retained_fragments):
        raise ClosureError("readiness template lacks one or more post-closure blockers")
    return [
        (
            "blocked config intentionally retains PENDING data identities; "
            "no final launch config or authorization has been generated"
            if "PENDING sentinels" in blocker
            else blocker
        )
        for blocker in retained
    ]


def _closed_readiness(
    snapshot: Mapping[str, Any],
    *,
    closure_identity: Mapping[str, object],
    capacity_path: Path,
    capacity_identity: Mapping[str, object],
) -> dict[str, Any]:
    result = copy.deepcopy(snapshot["readiness_template"])
    formal = snapshot["formal_validation_disjointness"]
    if formal.get("passed") is not True:
        raise ClosureError("formal validation disjointness did not pass")
    gates = formal.get("gates")
    if not isinstance(gates, Mapping):
        raise ClosureError("formal validation disjointness gate inventory is invalid")
    stable = gates.get("train_validation_stable_id")
    exact = gates.get("train_validation_normalized_exact")
    near = gates.get("train_validation_near_duplicate")
    if not all(isinstance(item, Mapping) for item in (stable, exact, near)):
        raise ClosureError("formal train/validation union gates are incomplete")
    baseline = snapshot["formal_baseline"]
    formal_gate = result.get("formal_validation_gate")
    if not isinstance(formal_gate, dict):
        raise ClosureError("readiness formal validation gate is invalid")
    formal_gate.update(
        {
            "status": "passed_authenticated_governed_disjointness_and_v3_baseline",
            "passed": True,
            "authorizes_training": False,
            "train_validation_union_disjointness": {
                "attestation_path": formal["identity"]["path"],
                "attestation_sha256": formal["identity"]["sha256"],
                "attestation_fingerprint": formal["attestation_fingerprint"],
                "stable_id_exact_passed": (
                    stable.get("passed") is True and stable.get("matches") == 0
                ),
                "normalized_text_exact_passed": (
                    exact.get("passed") is True and exact.get("matches") == 0
                ),
                "near_duplicate_passed": (
                    near.get("passed") is True and near.get("matches") == 0
                ),
                "near_duplicate_threshold": formal["near_duplicate_threshold"],
            },
            "v3_final_frozen_validation_baseline": {
                "bundle_path": baseline["root"],
                "summary_sha256": baseline["summary"]["sha256"],
                "manifest_sha256": baseline["manifest"]["sha256"],
                "complete_sha256": baseline["complete"]["sha256"],
                "checkpoint_complete_sha256": baseline[
                    "checkpoint_complete_sha256"
                ],
                "passed": True,
            },
        }
    )
    if not all(
        formal_gate["train_validation_union_disjointness"][field] is True
        for field in (
            "stable_id_exact_passed",
            "normalized_text_exact_passed",
            "near_duplicate_passed",
        )
    ):
        raise ClosureError("formal train/validation union gates did not close")
    pause = result.get("pause_evaluation_policy")
    capabilities = result.get("launch_command_capabilities")
    if not isinstance(pause, dict) or not isinstance(capabilities, dict):
        raise ClosureError("readiness controller capability contracts are invalid")
    pause.update(
        {
            "controller_implemented": True,
            "current_launch_command_auto_pauses": False,
            "current_launch_command_runs_validation": False,
        }
    )
    capabilities.update(
        {
            "starts_training_when_explicitly_invoked": False,
            "automatically_pauses_at_policy_thresholds": True,
            "automatically_runs_checkpoint_validation": True,
            "automatically_enforces_post_launch_hard_stops": True,
        }
    )
    result.update(
        {
            "status": (
                "blocked_pending_calibration_manual_review_and_final_authorization"
            ),
            "project_root": str(ROOT.resolve()),
            "required_capacity_attestation": str(capacity_path.resolve()),
            "capacity_attestation": copy.deepcopy(capacity_identity),
            "blockers": _remaining_blockers(result),
            "launch_enabled": False,
            "authorizes_training": False,
            "training_started": False,
            "launch_command_after_all_gates_pass": None,
            "launch_command_status": (
                "pending_final_config_calibration_manual_review_and_authorization"
            ),
            "governed_controller": {
                "path": str((ROOT / "scripts/govern_v4_training.py").resolve()),
                "sha256": sha256_file(ROOT / "scripts/govern_v4_training.py"),
                "twen_source_tree_sha256": twen_source_tree_sha256(),
                "implemented": True,
            },
            "closure": copy.deepcopy(closure_identity),
        }
    )
    calibration = result["calibration_gate"]
    if (
        calibration.get("passed") is not False
        or calibration.get("authorizes_training") is not False
        or pause.get("controller_implemented") is not True
    ):
        raise ClosureError("closure attempted to bypass calibration gates")
    unsigned = dict(result)
    result["readiness_fingerprint"] = _canonical_sha256(unsigned)
    return result


def _write_bundle(
    snapshot: Mapping[str, Any],
    output_root: Path,
    *,
    input_fingerprint: str,
    before_publish: Callable[[], None],
) -> dict[str, Any]:
    output = output_root.resolve()
    if output.exists():
        raise ClosureError(f"closure output already exists; choose a new directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{output.name}.incomplete-", dir=output.parent))
    closure_identity = {
        "schema_version": SCHEMA_VERSION,
        "kind": CLOSURE_KIND,
        "input_fingerprint": input_fingerprint,
        "closure_source_sha256": snapshot["closure_source_sha256"],
        "twen_source_tree_sha256": snapshot["twen_source_tree_sha256"],
        "launch_enabled": False,
        "authorizes_training": False,
        "training_started": False,
    }
    try:
        capacity_path = work / "capacity-attestation.json"
        readiness_path = work / "readiness.json"
        capacity = _closed_capacity(snapshot, closure_identity=closure_identity)
        atomic_write_json(capacity_path, capacity)
        final_capacity_path = output / capacity_path.name
        capacity_identity = {
            "path": str(final_capacity_path),
            "size": capacity_path.stat().st_size,
            "sha256": sha256_file(capacity_path),
            "attestation_fingerprint": capacity["attestation_fingerprint"],
            "passed": True,
            "authorizes_training": False,
        }
        readiness = _closed_readiness(
            snapshot,
            closure_identity=closure_identity,
            capacity_path=final_capacity_path,
            capacity_identity=capacity_identity,
        )
        atomic_write_json(readiness_path, readiness)
        files = {
            path.name: {
                "path": path.name,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (capacity_path, readiness_path)
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": BUNDLE_KIND,
            "closure": closure_identity,
            "inputs": _public_snapshot(snapshot),
            "files": files,
            "launch_enabled": False,
            "authorizes_training": False,
            "training_started": False,
        }
        manifest["bundle_fingerprint"] = _canonical_sha256(manifest)
        manifest_path = work / "MANIFEST.json"
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(
            work / "COMPLETE",
            {
                "schema_version": SCHEMA_VERSION,
                "kind": COMPLETE_KIND,
                "manifest": manifest_path.name,
                "manifest_sha256": sha256_file(manifest_path),
                "bundle_fingerprint": manifest["bundle_fingerprint"],
                "launch_enabled": False,
                "authorizes_training": False,
                "training_started": False,
            },
        )
        before_publish()
        if output.exists():
            raise ClosureError(f"closure output appeared during publication: {output}")
        os.replace(work, output)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return {
        "output": str(output),
        "capacity": str(output / "capacity-attestation.json"),
        "readiness": str(output / "readiness.json"),
        "manifest": str(output / "MANIFEST.json"),
        "complete": str(output / "COMPLETE"),
        "input_fingerprint": input_fingerprint,
        "launch_enabled": False,
        "authorizes_training": False,
        "training_started": False,
    }


def close_formal_evidence(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise ClosureError(f"closure output already exists; choose a new directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.closure.lock"
    with FileLock(lock_path, timeout_seconds=300.0):
        if output.exists():
            raise ClosureError(
                f"closure output already exists; choose a new directory: {output}"
            )
        first = _authenticate_inputs(args)
        input_fingerprint = _canonical_sha256(_public_snapshot(first))

        def reverify() -> None:
            second = _authenticate_inputs(args)
            if _canonical_sha256(_public_snapshot(second)) != input_fingerprint:
                raise ClosureError("formal closure inputs changed during publication")

        return _write_bundle(
            first,
            output,
            input_fingerprint=input_fingerprint,
            before_publish=reverify,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = close_formal_evidence(args)
    except (ClosureError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
