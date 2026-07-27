#!/usr/bin/env python3
"""Close authenticated v4 250M data/formal evidence without authorizing training.

The command is deliberately reporting-only.  It revalidates both finalized
phase corpora, their passing audit attestations, governed train and validation
prepared manifests, phase separation, formal train/validation separation, and
the v3-final formal baseline report.  It also authenticates the separate
Chinese semantic-quality bundle against the exact phase manifests.  It then
atomically writes independent capacity/readiness records while keeping the
blocked config and all launch controls unchanged.

It never loads a model, touches CUDA, starts calibration, or starts training.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
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

import yaml

from twen.data.audits import validate_base_audit_attestation
from twen.data.cursor import AuthenticatedSourceMap
from twen.data.prepared import validate_prepared_corpus
from twen.data.sources import (
    DataSourceError,
    load_base_data_recipe,
    validate_extracted_base_corpus,
)
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
CHINESE_SEMANTIC_ATTESTATION_KIND = "twen_v4_chinese_semantic_noise_attestation"
CHINESE_SEMANTIC_MANIFEST_KIND = "twen_v4_chinese_semantic_noise_bundle"
CHINESE_SEMANTIC_COMPLETE_KIND = "twen_v4_chinese_semantic_noise_complete"
CHINESE_SEMANTIC_SOURCE_ID = "chinese_wikipedia_zh_20231101"
CHINESE_SEMANTIC_SCANNER_POLICY = {
    "conversion_markers": [
        "da_nian_ye_substitution",
        "engine_substitution",
        "brand_command_substitution",
        "view_character_expansion",
        "cannot_substitution",
        "appeared_substitution",
    ],
    "high_precision_conversion_documents_eq": 0,
    "malformed_punctuation_documents_eq": 0,
    "manual_unacceptable_samples_eq": 0,
    "risk_samples_per_phase_gte": 32,
    "control_samples_per_phase_gte": 32,
    "reviewer_placeholder_forbidden": True,
    "reviewed_at_timezone_aware_iso8601_required": True,
}
WIKIPEDIA_REPO_ID = "wikimedia/wikipedia"
WIKIPEDIA_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
WIKIPEDIA_LICENSE = "CC-BY-SA-3.0 AND GFDL"
WIKIPEDIA_ATTRIBUTION_FIELDS = ("id", "url", "title")
WIKIPEDIA_EXCEPTION_ID = "formal-v4-wikipedia-zh-20231101-share-alike-r1"
WIKIPEDIA_LICENSE_SCOPE = (
    "Chinese Wikipedia snapshot admitted only by the formal-v4 source-specific "
    "share-alike exception; page attribution is mandatory."
)
WIKIPEDIA_EXCEPTION_PRECEDENCE = (
    "source-specific exceptions override excluded families only for the "
    "listed immutable source identity; final launch authorization remains separate"
)
WIKIPEDIA_EXCLUDED_LICENSE_FAMILIES = (
    "cc-by-sa",
    "gfdl",
    "gpl",
    "agpl",
    "lgpl",
    "mpl",
    "unknown",
    "no-license",
)
FORMAL_V4_PENDING_CONFIG_IDENTITIES = {
    "manifest_path": "PENDING_PRIMARY_PREPARED_MANIFEST",
    "manifest_sha256": "PENDING_PRIMARY_PREPARED_MANIFEST_SHA256",
    "source_map_sha256": "PENDING_PRIMARY_SOURCE_MAP_SHA256",
    "quality_cooldown_manifest_path": "PENDING_COOLDOWN_PREPARED_MANIFEST",
    "quality_cooldown_manifest_sha256": "PENDING_COOLDOWN_PREPARED_MANIFEST_SHA256",
    "phase_disjointness_attestation_path": (
        "PENDING_PRIMARY_COOLDOWN_PHASE_DISJOINTNESS_ATTESTATION"
    ),
    "phase_disjointness_attestation_sha256": (
        "PENDING_PRIMARY_COOLDOWN_PHASE_DISJOINTNESS_ATTESTATION_SHA256"
    ),
}
PHASES = ("primary", "cooldown")
REQUIRED_NEAR_DUPLICATE_THRESHOLD = 0.8
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPACITY_TEMPLATE = ROOT / "locks/base-data-sources-v4-250m.capacity-attestation.json"
DEFAULT_READINESS_TEMPLATE = ROOT / "locks/base-dense-v4-250m-pilot.readiness.json"


class ClosureError(ValueError):
    """One or more formal evidence closure conditions did not pass."""


class _ValidationSourceMap:
    __slots__ = ("fingerprint", "source_ids")

    def __init__(self, *, fingerprint: str, source_ids: tuple[str, ...]) -> None:
        self.fingerprint = fingerprint
        self.source_ids = source_ids


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
    parser.add_argument("--chinese-semantic-audit", type=Path, required=True)
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
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    if path.read_bytes() != source_before:
        raise ClosureError(f"evidence validator changed while loading: {path}")
    return module


def _resolve_repo_path(raw: object, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ClosureError(f"{label} must be a non-empty path")
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _pending_wikipedia_license_contract() -> dict[str, object]:
    return {
        "source_id": CHINESE_SEMANTIC_SOURCE_ID,
        "repo_id": WIKIPEDIA_REPO_ID,
        "revision": WIKIPEDIA_REVISION,
        "declared_license": WIKIPEDIA_LICENSE,
        "scope": "formal-v4-250m-primary-and-cooldown-only",
        "attribution_fields": list(WIKIPEDIA_ATTRIBUTION_FIELDS),
        "obligations": [
            "retain the generated attribution manifests with the training evidence",
            "document the CC-BY-SA-3.0/GFDL source in model and data reports",
            "perform a separate final review of model/distribution compliance",
        ],
    }


def _require_pending_wikipedia_license_gate(readiness: Mapping[str, Any]) -> None:
    gate = readiness.get("wikipedia_license_gate")
    contract = _pending_wikipedia_license_contract()
    fingerprint = _canonical_sha256(contract)
    if (
        not isinstance(gate, Mapping)
        or gate.get("required") is not True
        or gate.get("status") != "pending_explicit_user_acceptance"
        or gate.get("contract") != contract
        or gate.get("contract_fingerprint") != fingerprint
        or gate.get("required_acknowledgement")
        != f"ACCEPT V4 WIKIPEDIA LICENSE {fingerprint}"
        or gate.get("observed_acknowledgement") is not None
        or gate.get("passed") is not False
        or gate.get("authorizes_training") is not False
    ):
        raise ClosureError(
            "readiness Wikipedia license gate is not the exact pending contract"
        )


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
    if (
        not isinstance(config, Mapping)
        or config.get("contains_pending_identity_sentinels") is not True
    ):
        raise ClosureError("capacity template does not bind the blocked PENDING config")
    config_path = _resolve_repo_path(config.get("path"), label="capacity.config.path")
    config_identity = _identity(config_path)
    if config.get("sha256") != config_identity["sha256"]:
        raise ClosureError("capacity template blocked-config SHA256 is stale")
    try:
        blocked_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ClosureError(f"blocked config YAML could not be read: {exc}") from exc
    data = blocked_config.get("data") if isinstance(blocked_config, Mapping) else None
    if (
        not isinstance(data, Mapping)
        or any(
            data.get(field) != expected
            for field, expected in FORMAL_V4_PENDING_CONFIG_IDENTITIES.items()
        )
    ):
        raise ClosureError(
            "blocked config dynamic identity differs from the exact PENDING contract"
        )
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
    chinese_quality = readiness.get("chinese_semantic_quality_gate")
    required_bundle = (
        chinese_quality.get("required_bundle") if isinstance(chinese_quality, Mapping) else None
    )
    required_quality_gates = (
        chinese_quality.get("required_gates") if isinstance(chinese_quality, Mapping) else None
    )
    if (
        not isinstance(chinese_quality, Mapping)
        or chinese_quality.get("required") is not True
        or chinese_quality.get("status") != "pending_authenticated_chinese_semantic_quality_audit"
        or chinese_quality.get("source_id") != CHINESE_SEMANTIC_SOURCE_ID
        or chinese_quality.get("passed") is not False
        or chinese_quality.get("authorizes_training") is not False
        or chinese_quality.get("observed") is not None
        or not isinstance(required_bundle, Mapping)
        or required_bundle.get("manifest_kind") != CHINESE_SEMANTIC_MANIFEST_KIND
        or required_bundle.get("complete_kind") != CHINESE_SEMANTIC_COMPLETE_KIND
        or required_bundle.get("attestation_kind") != CHINESE_SEMANTIC_ATTESTATION_KIND
        or required_quality_gates
        != {
            "all_selected_shards_authenticated": True,
            "complete_streaming_scan": True,
            "high_precision_conversion_documents_eq": 0,
            "malformed_punctuation_documents_eq": 0,
            "risk_samples_per_phase_gte": 32,
            "control_samples_per_phase_gte": 32,
            "reviewer_placeholder_forbidden": True,
            "reviewed_at_timezone_aware_iso8601_required": True,
            "manual_review_passed": True,
        }
    ):
        raise ClosureError("readiness Chinese semantic quality gate is not pending and fail-closed")
    _require_pending_wikipedia_license_gate(readiness)
    pause = readiness.get("pause_evaluation_policy")
    if (
        not isinstance(pause, Mapping)
        or pause.get("enforcement") != "external_governed_controller"
        or pause.get("controller_implemented") is not True
        or pause.get("current_launch_command_auto_pauses") is not False
        or pause.get("current_launch_command_runs_validation") is not False
    ):
        raise ClosureError("readiness pause/evaluation controller contract differs")
    capabilities = readiness.get("launch_command_capabilities")
    if (
        not isinstance(capabilities, Mapping)
        or capabilities.get("current_blocked_config_rejects_training") is not True
        or capabilities.get("starts_training_when_explicitly_invoked") is not False
        or any(
            capabilities.get(field) is not True
            for field in (
                "automatically_pauses_at_policy_thresholds",
                "automatically_runs_checkpoint_validation",
                "automatically_enforces_post_launch_hard_stops",
            )
        )
    ):
        raise ClosureError("readiness launch capabilities are not fail-closed")
    controller = readiness.get("governed_controller")
    controller_path = ROOT / "scripts/govern_v4_training.py"
    if (
        not isinstance(controller, Mapping)
        or controller.get("implemented") is not True
        or _resolve_repo_path(
            controller.get("path"),
            label="readiness.governed_controller.path",
        )
        != controller_path.resolve()
        or controller.get("sha256") != sha256_file(controller_path)
        or controller.get("twen_source_tree_sha256") != twen_source_tree_sha256()
    ):
        raise ClosureError("readiness governed controller identity differs")
    return capacity, readiness, config_identity


def _governed_prepared(
    path: Path,
    *,
    role: str,
    extracted_path: Path,
    extracted_sha256: str,
    audit_path: Path,
    audit_sha256: str,
) -> tuple[Any, Any, dict[str, object], dict[str, int]]:
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
    if role == "train":
        source_map = AuthenticatedSourceMap.from_prepared_manifest(prepared)
        entries = {entry.shard_id: entry for entry in prepared.shards}
        source_tokens = {
            source_id: sum(
                entries[shard.shard_id].token_count
                for shard in source_map.shards_for_source(source_id)
            )
            for source_id in source_map.source_ids
        }
    else:
        source_map, source_tokens = _validation_source_inventory(prepared)
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


def _validation_source_inventory(
    prepared: Any,
) -> tuple[_ValidationSourceMap, dict[str, int]]:
    lineage = getattr(prepared, "lineage", None)
    contract = lineage.get("data_contract") if isinstance(lineage, Mapping) else None
    source_map = contract.get("source_map") if isinstance(contract, Mapping) else None
    roles = source_map.get("roles") if isinstance(source_map, Mapping) else None
    validation = roles.get("validation") if isinstance(roles, Mapping) else None
    fingerprint = source_map.get("fingerprint") if isinstance(source_map, Mapping) else None
    if (
        not isinstance(source_map, Mapping)
        or source_map.get("algorithm") != "authenticated-extracted-output-map-v1"
        or not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
        or not isinstance(validation, list)
        or not validation
    ):
        raise ClosureError("validation prepared source-map contract is invalid")
    unsigned = dict(source_map)
    unsigned.pop("fingerprint", None)
    if _canonical_sha256(unsigned) != fingerprint:
        raise ClosureError("validation prepared source-map fingerprint mismatch")

    lineage_files = lineage.get("source_files")
    if not isinstance(lineage_files, list):
        raise ClosureError("validation prepared source file inventory is invalid")
    expected_files: set[tuple[str, str, int]] = set()
    for index, row in enumerate(lineage_files):
        if not isinstance(row, Mapping):
            raise ClosureError(f"validation source_files[{index}] is invalid")
        path = row.get("path")
        digest = row.get("sha256")
        size = row.get("size")
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or not isinstance(digest, str)
            or len(digest) != 64
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ClosureError(f"validation source_files[{index}] identity is invalid")
        expected_files.add((path.replace("\\", "/"), digest, size))

    mapped_files: set[tuple[str, str, int]] = set()
    source_by_file: dict[tuple[str, str], str] = {}
    for index, row in enumerate(validation):
        if not isinstance(row, Mapping):
            raise ClosureError(f"validation source_map[{index}] is invalid")
        source_id = row.get("source_id")
        path = row.get("path")
        digest = row.get("sha256")
        size = row.get("size")
        if (
            not isinstance(source_id, str)
            or not source_id
            or not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or not isinstance(digest, str)
            or len(digest) != 64
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ClosureError(f"validation source_map[{index}] identity is invalid")
        normalized = path.replace("\\", "/")
        identity = (normalized, digest, size)
        key = (normalized, digest)
        if identity in mapped_files or key in source_by_file:
            raise ClosureError("validation source-map repeats a source file")
        mapped_files.add(identity)
        source_by_file[key] = source_id
    if mapped_files != expected_files:
        raise ClosureError("validation source-map differs from prepared source files")

    source_tokens: dict[str, int] = {}
    for shard in prepared.shards:
        normalized_source = str(shard.source_path).replace("\\", "/")
        candidates = {
            source_id
            for (path, digest), source_id in source_by_file.items()
            if shard.source_sha256 == digest
            and (normalized_source == path or normalized_source.endswith(f"/{path}"))
        }
        if len(candidates) != 1:
            raise ClosureError(
                f"validation prepared shard has ambiguous source ownership: {shard.shard_id}"
            )
        source_id = next(iter(candidates))
        source_tokens[source_id] = source_tokens.get(source_id, 0) + int(shard.token_count)
    if sum(source_tokens.values()) != prepared.token_count:
        raise ClosureError("validation prepared source-map token inventory differs")
    return (
        _ValidationSourceMap(
            fingerprint=fingerprint,
            source_ids=tuple(sorted(source_tokens)),
        ),
        source_tokens,
    )


def _expected_schema_v2_wikipedia_exception_contract(
    phase: str,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ClosureError(f"unsupported formal data phase: {phase!r}")
    attribution_fields = list(WIKIPEDIA_ATTRIBUTION_FIELDS)
    return {
        "recipe_schema_version": 2,
        "recipe_kind": "twen_base_data_source_recipe_v2",
        "recipe_schema_status": "stable",
        "license_policy": {
            "attribution_manifest_required": True,
            "excluded_families": list(WIKIPEDIA_EXCLUDED_LICENSE_FAMILIES),
            "source_specific_share_alike_exceptions": [
                {
                    "exception_id": WIKIPEDIA_EXCEPTION_ID,
                    "source_id": CHINESE_SEMANTIC_SOURCE_ID,
                    "declared_license": WIKIPEDIA_LICENSE,
                    "scope": f"formal-v4-250m-{phase}-only",
                    "overrides_excluded_families": ["cc-by-sa", "gfdl"],
                    "attribution_manifest_required": True,
                    "attribution_fields": attribution_fields,
                    "authorizes_training": False,
                }
            ],
            "exception_precedence": WIKIPEDIA_EXCEPTION_PRECEDENCE,
        },
        "source": {
            "source_id": CHINESE_SEMANTIC_SOURCE_ID,
            "repo_id": WIKIPEDIA_REPO_ID,
            "revision": WIKIPEDIA_REVISION,
            "license_declaration": WIKIPEDIA_LICENSE,
            "license_scope": WIKIPEDIA_LICENSE_SCOPE,
            "attribution_fields": attribution_fields,
            "share_alike_exception": {
                "exception_id": WIKIPEDIA_EXCEPTION_ID,
                "declared_license": WIKIPEDIA_LICENSE,
                "scope": "formal-v4-250m-primary-and-cooldown-only",
                "materialization_allowed": True,
                "attribution_manifest_required": True,
                "attribution_fields": attribution_fields,
                "authorizes_training": False,
            },
        },
        "attribution_contract": {
            "manifest_required": True,
            "source_id": CHINESE_SEMANTIC_SOURCE_ID,
            "fields": attribution_fields,
            "retain_with_prepared_corpus": True,
            "required_before_launch": True,
            "authorizes_training": False,
        },
    }


def _require_native_schema_v2_wikipedia_exception(
    recipe: Mapping[str, Any],
    *,
    phase: str,
) -> str:
    """Authenticate the one schema-v2 share-alike exception by canonical JSON.

    ``load_base_data_recipe`` intentionally ignores unknown extension fields.
    Formal v4 therefore validates this policy-bearing extension separately,
    including its exact key inventory and JSON value types.  The exception can
    authorize materialization with attribution, but it can never authorize
    training; the explicit user acknowledgement remains a separate gate.
    """

    sources = recipe.get("sources")
    license_policy = recipe.get("license_policy")
    if not isinstance(sources, list) or not isinstance(license_policy, Mapping):
        raise ClosureError(
            f"{phase} Wikipedia recipe/license/attribution contract differs: "
            "schema-v2 exception containers are invalid"
        )
    exception_sources = [
        source
        for source in sources
        if isinstance(source, Mapping) and "share_alike_exception" in source
    ]
    if len(exception_sources) != 1:
        raise ClosureError(
            f"{phase} Wikipedia recipe/license/attribution contract differs: "
            "share-alike exception source inventory is not singular"
        )
    source = exception_sources[0]
    actual_contract = {
        "recipe_schema_version": recipe.get("schema_version"),
        "recipe_kind": recipe.get("kind"),
        "recipe_schema_status": recipe.get("schema_status"),
        "license_policy": {
            "attribution_manifest_required": license_policy.get(
                "attribution_manifest_required"
            ),
            "excluded_families": license_policy.get("excluded_families"),
            "source_specific_share_alike_exceptions": license_policy.get(
                "source_specific_share_alike_exceptions"
            ),
            "exception_precedence": license_policy.get("exception_precedence"),
        },
        "source": {
            "source_id": source.get("source_id"),
            "repo_id": source.get("repo_id"),
            "revision": source.get("revision"),
            "license_declaration": source.get("license_declaration"),
            "license_scope": source.get("license_scope"),
            "attribution_fields": source.get("attribution_fields"),
            "share_alike_exception": source.get("share_alike_exception"),
        },
        "attribution_contract": recipe.get("attribution_contract"),
    }
    expected_contract = _expected_schema_v2_wikipedia_exception_contract(phase)
    actual_fingerprint = _canonical_sha256(actual_contract)
    expected_fingerprint = _canonical_sha256(expected_contract)
    if actual_fingerprint != expected_fingerprint:
        raise ClosureError(
            f"{phase} Wikipedia recipe/license/attribution contract differs: "
            "schema-v2 share-alike exception fingerprint mismatch"
        )
    return expected_fingerprint


def _require_phase_recipe_contract(
    extracted: Mapping[str, Any],
    *,
    phase: str,
    capacity_stage: Mapping[str, Any],
) -> dict[str, Any]:
    recipe_binding = capacity_stage.get("recipe")
    resolved_binding = capacity_stage.get("resolved_lock")
    if not isinstance(recipe_binding, Mapping) or not isinstance(
        resolved_binding, Mapping
    ):
        raise ClosureError(f"{phase} capacity recipe/resolved-lock binding is invalid")
    recipe_path = _resolve_repo_path(
        recipe_binding.get("path"),
        label=f"capacity.{phase}.recipe.path",
    )
    resolved_path = _resolve_repo_path(
        resolved_binding.get("path"),
        label=f"capacity.{phase}.resolved_lock.path",
    )
    recipe_identity = _identity(recipe_path)
    resolved_identity = _identity(resolved_path)
    recipe = _read_json(recipe_path, label=f"{phase} recipe")
    resolved = _read_json(resolved_path, label=f"{phase} resolved lock")
    _require_native_schema_v2_wikipedia_exception(recipe, phase=phase)
    try:
        parsed_recipe = load_base_data_recipe(recipe_path)
        parsed_recipe.require_runnable(f"formal v4 {phase} closure")
    except (DataSourceError, OSError) as exc:
        raise ClosureError(
            f"{phase} schema-v2 recipe failed native validation: {exc}"
        ) from exc
    recipe_id = recipe_binding.get("recipe_id")
    recipe_sha = recipe_binding.get("sha256")
    resolved_sha = resolved_binding.get("sha256")
    resolved_audit = resolved.get("materialization_audit")
    if (
        not isinstance(recipe_id, str)
        or recipe.get("recipe_id") != recipe_id
        or parsed_recipe.schema_version != 2
        or parsed_recipe.recipe_id != recipe_id
        or parsed_recipe.sha256 != recipe_identity["sha256"]
        or recipe_identity["sha256"] != recipe_sha
        or resolved_identity["sha256"] != resolved_sha
        or resolved_binding.get("passed") is not True
        or resolved_binding.get("remote_identity_verification")
        != "verified_against_hub_metadata"
        or resolved.get("recipe_id") != recipe_id
        or resolved.get("recipe_sha256") != recipe_sha
        or not isinstance(resolved_audit, Mapping)
        or resolved_audit.get("complete") is not True
        or resolved_audit.get("remote_identity_verification")
        != "verified_against_hub_metadata"
    ):
        raise ClosureError(
            f"{phase} capacity recipe/resolved-lock identity is stale or incomplete"
        )
    if (
        extracted.get("recipe_id") != recipe_id
        or extracted.get("recipe_sha256") != recipe_sha
        or extracted.get("resolved_source_lock_sha256") != resolved_sha
    ):
        raise ClosureError(
            f"{phase} extracted corpus recipe/resolved-lock identity differs "
            "from the capacity contract"
        )

    sources = recipe.get("sources")
    resolved_sources = resolved.get("sources")
    if not isinstance(sources, list) or not isinstance(resolved_sources, list):
        raise ClosureError(f"{phase} recipe/resolved source inventory is invalid")
    wikipedia_sources = [
        source
        for source in sources
        if isinstance(source, Mapping)
        and source.get("source_id") == CHINESE_SEMANTIC_SOURCE_ID
    ]
    resolved_wikipedia = [
        source
        for source in resolved_sources
        if isinstance(source, Mapping)
        and source.get("source_id") == CHINESE_SEMANTIC_SOURCE_ID
    ]
    if len(wikipedia_sources) != 1 or len(resolved_wikipedia) != 1:
        raise ClosureError(f"{phase} Wikipedia source inventory is not singular")
    wikipedia = wikipedia_sources[0]
    resolved_source = resolved_wikipedia[0]
    expected_attribution_contract = {
        "manifest_required": True,
        "source_id": CHINESE_SEMANTIC_SOURCE_ID,
        "fields": list(WIKIPEDIA_ATTRIBUTION_FIELDS),
        "retain_with_prepared_corpus": True,
        "required_before_launch": True,
        "authorizes_training": False,
    }
    expected_source_exception = {
        "exception_id": WIKIPEDIA_EXCEPTION_ID,
        "declared_license": WIKIPEDIA_LICENSE,
        "scope": "formal-v4-250m-primary-and-cooldown-only",
        "materialization_allowed": True,
        "attribution_manifest_required": True,
        "attribution_fields": list(WIKIPEDIA_ATTRIBUTION_FIELDS),
        "authorizes_training": False,
    }
    expected_policy_exception = {
        "exception_id": WIKIPEDIA_EXCEPTION_ID,
        "source_id": CHINESE_SEMANTIC_SOURCE_ID,
        "declared_license": WIKIPEDIA_LICENSE,
        "scope": f"formal-v4-250m-{phase}-only",
        "overrides_excluded_families": ["cc-by-sa", "gfdl"],
        "attribution_manifest_required": True,
        "attribution_fields": list(WIKIPEDIA_ATTRIBUTION_FIELDS),
        "authorizes_training": False,
    }
    license_policy = recipe.get("license_policy")
    locked_files = wikipedia.get("locked_files")
    resolved_files = resolved_source.get("files")
    if (
        recipe.get("attribution_contract") != expected_attribution_contract
        or not isinstance(license_policy, Mapping)
        or license_policy.get("source_specific_share_alike_exceptions")
        != [expected_policy_exception]
        or wikipedia.get("repo_id") != WIKIPEDIA_REPO_ID
        or wikipedia.get("revision") != WIKIPEDIA_REVISION
        or wikipedia.get("license_declaration") != WIKIPEDIA_LICENSE
        or wikipedia.get("license_scope") != WIKIPEDIA_LICENSE_SCOPE
        or wikipedia.get("attribution_fields")
        != list(WIKIPEDIA_ATTRIBUTION_FIELDS)
        or wikipedia.get("share_alike_exception") != expected_source_exception
        or not isinstance(locked_files, list)
        or not locked_files
        or resolved_source.get("repo_id") != WIKIPEDIA_REPO_ID
        or resolved_source.get("revision") != WIKIPEDIA_REVISION
        or not isinstance(resolved_files, list)
        or [
            {
                "path": item.get("path"),
                "size": item.get("size"),
                "sha256": item.get("sha256"),
            }
            for item in resolved_files
            if isinstance(item, Mapping)
        ]
        != locked_files
    ):
        raise ClosureError(
            f"{phase} Wikipedia recipe/license/attribution contract differs"
        )
    return recipe


def _wikipedia_attribution_evidence(
    extracted_file: Path,
    extracted: Mapping[str, Any],
) -> dict[str, Any]:
    license_audit = extracted.get("license_audit")
    license_sources = (
        license_audit.get("sources") if isinstance(license_audit, Mapping) else None
    )
    if not isinstance(license_sources, list):
        raise ClosureError("extracted Wikipedia license audit has no source inventory")
    wikipedia_license_rows = [
        row
        for row in license_sources
        if isinstance(row, Mapping)
        and row.get("source_id") == CHINESE_SEMANTIC_SOURCE_ID
    ]
    if (
        len(wikipedia_license_rows) != 1
        or wikipedia_license_rows[0].get("declaration") != WIKIPEDIA_LICENSE
        or wikipedia_license_rows[0].get("scope") != WIKIPEDIA_LICENSE_SCOPE
        or wikipedia_license_rows[0].get("field") is not None
        or wikipedia_license_rows[0].get("allowlist") != []
    ):
        raise ClosureError("extracted Wikipedia license audit differs from recipe")

    sources = extracted.get("sources")
    if not isinstance(sources, list):
        raise ClosureError("extracted Wikipedia source statistics are invalid")
    wikipedia_sources = [
        row
        for row in sources
        if isinstance(row, Mapping)
        and row.get("source_id") == CHINESE_SEMANTIC_SOURCE_ID
    ]
    if len(wikipedia_sources) != 1:
        raise ClosureError("extracted Wikipedia source statistics are not singular")
    source = wikipedia_sources[0]
    train_rows = source.get("train_rows")
    validation_rows = source.get("validation_rows")
    if (
        isinstance(train_rows, bool)
        or not isinstance(train_rows, int)
        or train_rows < 1
        or isinstance(validation_rows, bool)
        or not isinstance(validation_rows, int)
        or validation_rows < 1
    ):
        raise ClosureError("extracted Wikipedia source row counts are invalid")
    expected_rows = train_rows + validation_rows

    raw_inventory = extracted.get("attribution_files")
    if not isinstance(raw_inventory, list) or not raw_inventory:
        raise ClosureError("extracted Wikipedia attribution inventory is empty")
    authenticated_rows = 0
    root = extracted_file.parent.resolve()
    for index, raw_entry in enumerate(raw_inventory):
        if not isinstance(raw_entry, Mapping):
            raise ClosureError(f"attribution_files[{index}] is invalid")
        relative = raw_entry.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ClosureError(f"attribution_files[{index}].path is unsafe")
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ClosureError(f"attribution_files[{index}].path escapes its corpus")
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        raise ClosureError(
                            f"empty attribution row: {relative}:{line_number}"
                        )
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ClosureError(
                            f"invalid attribution JSON: {relative}:{line_number}"
                        ) from exc
                    if not isinstance(row, Mapping):
                        raise ClosureError(
                            f"invalid attribution object: {relative}:{line_number}"
                        )
                    if row.get("source_id") != CHINESE_SEMANTIC_SOURCE_ID:
                        continue
                    if (
                        any(
                            not isinstance(row.get(field), str)
                            or not str(row[field]).strip()
                            for field in WIKIPEDIA_ATTRIBUTION_FIELDS
                        )
                        or row.get("repo_id") != WIKIPEDIA_REPO_ID
                        or row.get("revision") != WIKIPEDIA_REVISION
                        or row.get("source_license") != WIKIPEDIA_LICENSE
                    ):
                        raise ClosureError(
                            "Wikipedia attribution row violates the immutable contract: "
                            f"{relative}:{line_number}"
                        )
                    authenticated_rows += 1
        except OSError as exc:
            raise ClosureError(f"cannot read attribution inventory file: {path}") from exc
    if authenticated_rows != expected_rows:
        raise ClosureError(
            "Wikipedia attribution coverage differs from extracted source rows: "
            f"{authenticated_rows} != {expected_rows}"
        )
    return {
        "source_id": CHINESE_SEMANTIC_SOURCE_ID,
        "required_fields": list(WIKIPEDIA_ATTRIBUTION_FIELDS),
        "authenticated_rows": authenticated_rows,
        "expected_rows": expected_rows,
        "inventory_fingerprint": _canonical_sha256(raw_inventory),
        "passed": True,
        "authorizes_training": False,
    }


def _phase_evidence(
    *,
    phase: str,
    capacity_stage: Mapping[str, Any],
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
    _require_phase_recipe_contract(
        extracted,
        phase=phase,
        capacity_stage=capacity_stage,
    )
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
    validation, validation_map, validation_identity, validation_source_tokens = _governed_prepared(
        validation_prepared_path,
        role="validation",
        extracted_path=extracted_file,
        extracted_sha256=extracted_sha,
        audit_path=audit_file,
        audit_sha256=audit_sha,
    )
    if set(train_map.source_ids) != set(validation_map.source_ids):
        raise ClosureError(f"{phase} train/validation source coverage differs")
    if any(tokens <= 0 for tokens in validation_source_tokens.values()):
        raise ClosureError(f"{phase} validation contains an empty source")
    license_audit = extracted.get("license_audit")
    attribution = (
        license_audit.get("attribution_inventory") if isinstance(license_audit, Mapping) else None
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
    attribution_identity["wikipedia_contract"] = _wikipedia_attribution_evidence(
        extracted_file,
        extracted,
    )
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


def _validate_chinese_semantic_audit_bundle(
    root: Path,
    *,
    phases: Mapping[str, Mapping[str, Any]],
    minimum_risk_samples_per_phase: int,
    minimum_control_samples_per_phase: int,
) -> dict[str, Any]:
    bundle = root.resolve()
    if not bundle.is_dir():
        raise ClosureError(f"Chinese semantic audit bundle does not exist: {bundle}")
    manifest_path = bundle / "MANIFEST.json"
    complete_path = bundle / "COMPLETE"
    attestation_path = bundle / "attestation.json"
    manifest = _read_json(manifest_path, label="Chinese semantic audit MANIFEST")
    complete = _read_json(complete_path, label="Chinese semantic audit COMPLETE")
    attestation = _read_json(
        attestation_path,
        label="Chinese semantic audit attestation",
    )

    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != CHINESE_SEMANTIC_MANIFEST_KIND
        or manifest.get("passed") is not True
        or manifest.get("authorizes_training") is not False
    ):
        raise ClosureError("Chinese semantic audit MANIFEST gate differs")
    if (
        complete.get("schema_version") != SCHEMA_VERSION
        or complete.get("kind") != CHINESE_SEMANTIC_COMPLETE_KIND
        or complete.get("manifest") != "MANIFEST.json"
        or complete.get("manifest_sha256") != sha256_file(manifest_path)
        or complete.get("passed") is not True
        or complete.get("authorizes_training") is not False
    ):
        raise ClosureError("Chinese semantic audit COMPLETE does not authenticate MANIFEST")

    files = manifest.get("files")
    required_files = {
        "attestation.json",
        "manual-review-template.json",
        "samples.jsonl",
    }
    if not isinstance(files, Mapping) or set(files) != required_files:
        raise ClosureError("Chinese semantic audit payload inventory differs")
    for relative, raw_identity in files.items():
        if not isinstance(raw_identity, Mapping):
            raise ClosureError(f"Chinese semantic audit file identity is invalid: {relative}")
        actual = _identity(bundle / relative)
        if (
            raw_identity.get("size") != actual["size"]
            or raw_identity.get("sha256") != actual["sha256"]
        ):
            raise ClosureError(f"Chinese semantic audit payload identity differs: {relative}")

    fingerprint = attestation.get("attestation_fingerprint")
    unsigned_attestation = dict(attestation)
    unsigned_attestation.pop("attestation_fingerprint", None)
    scanner = attestation.get("scanner")
    scanner_path = ROOT / "scripts/audit_v4_chinese_semantic_noise.py"
    scanner_value_path = scanner.get("path") if isinstance(scanner, Mapping) else None
    if (
        not isinstance(scanner, Mapping)
        or not isinstance(scanner_value_path, str)
        or Path(scanner_value_path).resolve() != scanner_path.resolve()
        or scanner.get("sha256") != sha256_file(scanner_path)
        or scanner.get("policy") != CHINESE_SEMANTIC_SCANNER_POLICY
    ):
        raise ClosureError(
            "Chinese semantic audit scanner source identity or policy differs"
        )
    if (
        attestation.get("schema_version") != SCHEMA_VERSION
        or attestation.get("kind") != CHINESE_SEMANTIC_ATTESTATION_KIND
        or attestation.get("source_id") != CHINESE_SEMANTIC_SOURCE_ID
        or attestation.get("passed") is not True
        or attestation.get("authorizes_training") is not False
        or attestation.get("status") != "passed_quality_gate_but_does_not_authorize_training"
        or not isinstance(fingerprint, str)
        or _canonical_sha256(unsigned_attestation) != fingerprint
        or manifest.get("attestation_fingerprint") != fingerprint
        or manifest.get("created_at") != attestation.get("created_at")
    ):
        raise ClosureError("Chinese semantic audit attestation is not a passing bound record")

    inputs = attestation.get("inputs")
    phase_statistics = attestation.get("phases")
    if (
        not isinstance(inputs, Mapping)
        or set(inputs) != set(PHASES)
        or not isinstance(phase_statistics, Mapping)
        or set(phase_statistics) != set(PHASES)
    ):
        raise ClosureError("Chinese semantic audit phase inventory differs")

    samples = attestation.get("samples")
    if not isinstance(samples, Mapping):
        raise ClosureError("Chinese semantic audit sample inventory is invalid")
    risk_sample_size = samples.get("risk_samples_per_phase")
    control_sample_size = samples.get("control_samples_per_phase")
    if (
        isinstance(minimum_risk_samples_per_phase, bool)
        or not isinstance(minimum_risk_samples_per_phase, int)
        or minimum_risk_samples_per_phase < 1
        or isinstance(minimum_control_samples_per_phase, bool)
        or not isinstance(minimum_control_samples_per_phase, int)
        or minimum_control_samples_per_phase < 1
        or isinstance(risk_sample_size, bool)
        or not isinstance(risk_sample_size, int)
        or isinstance(control_sample_size, bool)
        or not isinstance(control_sample_size, int)
    ):
        raise ClosureError("Chinese semantic audit sample quotas are invalid")
    if (
        risk_sample_size < minimum_risk_samples_per_phase
        or control_sample_size < minimum_control_samples_per_phase
    ):
        raise ClosureError(
            "Chinese semantic audit sample quotas are below authenticated "
            "readiness minimum"
        )

    scanner_module = _load_script(
        "audit_v4_chinese_semantic_noise.py",
        "_twen_v4_chinese_semantic_noise",
    )
    scanner_sha_before = sha256_file(scanner_path)
    try:
        recomputed = scanner_module.recompute_scan(
            primary_manifest=Path(
                str(phases["primary"]["extracted"]["manifest_path"])
            ),
            cooldown_manifest=Path(
                str(phases["cooldown"]["extracted"]["manifest_path"])
            ),
            source_id=CHINESE_SEMANTIC_SOURCE_ID,
            risk_sample_size=risk_sample_size,
            control_sample_size=control_sample_size,
        )
    except (OSError, ValueError) as exc:
        raise ClosureError(
            f"Chinese semantic audit deterministic recomputation failed: {exc}"
        ) from exc
    if (
        sha256_file(scanner_path) != scanner_sha_before
        or scanner_sha_before != scanner.get("sha256")
    ):
        raise ClosureError("Chinese semantic audit scanner changed during recomputation")
    if (
        _canonical_sha256(recomputed.get("inputs"))
        != _canonical_sha256(inputs)
        or _canonical_sha256(recomputed.get("phases"))
        != _canonical_sha256(phase_statistics)
    ):
        raise ClosureError(
            "Chinese semantic audit statistics differ from deterministic recomputation"
        )

    recomputed_samples = recomputed.get("samples")
    recomputed_payload = recomputed.get("samples_payload")
    if (
        not isinstance(recomputed_samples, list)
        or not isinstance(recomputed_payload, bytes)
        or (bundle / "samples.jsonl").read_bytes() != recomputed_payload
        or samples.get("size") != len(recomputed_payload)
        or samples.get("sha256") != recomputed.get("samples_sha256")
        or samples.get("count") != len(recomputed_samples)
    ):
        raise ClosureError(
            "Chinese semantic audit samples differ from deterministic recomputation"
        )
    expected_template = scanner_module._manual_template(
        samples_sha256=str(recomputed["samples_sha256"]),
        samples=recomputed_samples,
    )
    actual_template = _read_json(
        bundle / "manual-review-template.json",
        label="Chinese semantic manual-review template",
    )
    if _canonical_sha256(actual_template) != _canonical_sha256(expected_template):
        raise ClosureError(
            "Chinese semantic manual-review template differs from recomputed samples"
        )

    manual = attestation.get("manual_review")
    if not isinstance(manual, Mapping) or not isinstance(manual.get("path"), str):
        raise ClosureError("Chinese semantic audit manual review identity is invalid")
    try:
        recomputed_manual = scanner_module._load_manual_decisions(
            Path(str(manual["path"])).resolve(),
            sample_sha256=str(recomputed["samples_sha256"]),
            samples=recomputed_samples,
        )
    except (OSError, ValueError) as exc:
        raise ClosureError(
            f"Chinese semantic manual review failed recomputation: {exc}"
        ) from exc
    recomputed_manual["status"] = (
        "passed" if recomputed_manual.get("passed") is True else "failed"
    )
    if (
        _canonical_sha256(recomputed_manual) != _canonical_sha256(manual)
        or recomputed_manual.get("passed") is not True
    ):
        raise ClosureError(
            "Chinese semantic manual review differs from recomputed samples"
        )

    total_samples = 0
    for phase in PHASES:
        raw_identity = inputs.get(phase)
        expected = phases[phase]["extracted"]
        expected_path = Path(str(expected["manifest_path"])).resolve()
        actual_manifest = _identity(expected_path)
        if (
            not isinstance(raw_identity, Mapping)
            or not isinstance(raw_identity.get("path"), str)
            or Path(str(raw_identity["path"])).resolve() != expected_path
            or raw_identity.get("size") != actual_manifest["size"]
            or raw_identity.get("sha256") != expected["manifest_sha256"]
            or raw_identity.get("corpus_fingerprint") != expected["corpus_fingerprint"]
            or isinstance(raw_identity.get("shard_count"), bool)
            or not isinstance(raw_identity.get("shard_count"), int)
            or int(raw_identity["shard_count"]) < 1
        ):
            raise ClosureError(f"Chinese semantic audit differs from explicit {phase} manifest")
        statistics = phase_statistics.get(phase)
        if not isinstance(statistics, Mapping):
            raise ClosureError(f"Chinese semantic audit {phase} statistics are invalid")
        marker_documents = statistics.get("marker_documents")
        documents = statistics.get("documents")
        authenticated_shards = statistics.get("authenticated_shards")
        sample_count = statistics.get("sample_count")
        if (
            isinstance(documents, bool)
            or not isinstance(documents, int)
            or documents < 1
            or isinstance(authenticated_shards, bool)
            or not isinstance(authenticated_shards, int)
            or authenticated_shards != raw_identity["shard_count"]
            or isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count < 1
            or statistics.get("high_precision_conversion_documents") != 0
            or not isinstance(marker_documents, Mapping)
            or marker_documents.get("malformed_punctuation", 0) != 0
        ):
            raise ClosureError(f"Chinese semantic audit {phase} statistical quality gate failed")
        total_samples += sample_count

    sample_file = files["samples.jsonl"]
    if (
        not isinstance(samples, Mapping)
        or samples.get("path") != "samples.jsonl"
        or samples.get("size") != sample_file.get("size")
        or samples.get("sha256") != sample_file.get("sha256")
        or samples.get("count") != total_samples
    ):
        raise ClosureError("Chinese semantic audit sample inventory differs")

    manual_identity = _identity(Path(str(manual["path"])))
    if (
        manual.get("size") != manual_identity["size"]
        or manual.get("sha256") != manual_identity["sha256"]
        or not isinstance(manual.get("reviewer"), str)
        or not str(manual["reviewer"]).strip()
        or not isinstance(manual.get("reviewed_at"), str)
        or not str(manual["reviewed_at"]).strip()
        or manual.get("reviewed_samples") != total_samples
        or manual.get("unacceptable_samples") != 0
        or manual.get("unacceptable_rate") != 0
        or manual.get("passed") is not True
        or manual.get("status") != "passed"
    ):
        raise ClosureError("Chinese semantic audit manual review gate failed")

    gates = attestation.get("gates")
    expected_gates = {
        "all_selected_shards_authenticated": True,
        "complete_streaming_scan": True,
        "high_precision_conversion_documents": 0,
        "high_precision_conversion_passed": True,
        "malformed_punctuation_documents": 0,
        "malformed_punctuation_passed": True,
        "manual_review_passed": True,
    }
    if not isinstance(gates, Mapping) or dict(gates) != expected_gates:
        raise ClosureError("Chinese semantic audit statistical/manual gates did not all pass")

    return {
        "root": str(bundle),
        "source_id": CHINESE_SEMANTIC_SOURCE_ID,
        "manifest": _identity(manifest_path),
        "complete": _identity(complete_path),
        "attestation": _identity(attestation_path),
        "attestation_fingerprint": fingerprint,
        "scanner": copy.deepcopy(scanner),
        "inputs": copy.deepcopy(inputs),
        "phases": copy.deepcopy(phase_statistics),
        "gates": copy.deepcopy(gates),
        "manual_review": copy.deepcopy(manual),
        "passed": True,
        "authorizes_training": False,
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
    rows_by_phase = {str(row.get("phase")): row for row in phase_rows if isinstance(row, Mapping)}
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
        if (
            Path(str(actual.get("path"))).resolve()
            != Path(str(expected["manifest_path"])).resolve()
        ):
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
        legacy_baseline.get("checkpoint_artifact") if isinstance(legacy_baseline, Mapping) else None
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
    capacity_stages = capacity.get("stages")
    if not isinstance(capacity_stages, Mapping) or set(capacity_stages) != set(PHASES):
        raise ClosureError("capacity template phase inventory differs")
    phases = {
        phase: _phase_evidence(
            phase=phase,
            capacity_stage=capacity_stages[phase],
            **{f"{name}_path": path for name, path in _explicit_paths(args, phase).items()},
        )
        for phase in PHASES
    }
    chinese_quality = readiness["chinese_semantic_quality_gate"]
    required_quality_gates = chinese_quality["required_gates"]
    chinese_semantic_audit = _validate_chinese_semantic_audit_bundle(
        args.chinese_semantic_audit,
        phases=phases,
        minimum_risk_samples_per_phase=required_quality_gates[
            "risk_samples_per_phase_gte"
        ],
        minimum_control_samples_per_phase=required_quality_gates[
            "control_samples_per_phase_gte"
        ],
    )
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
            raise ClosureError(
                f"formal validation disjointness differs from explicit {phase} inputs"
            )
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
        or set(phase_gates) != {"stable_id_exact", "normalized_text_exact", "near_duplicate"}
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
        "chinese_semantic_audit": chinese_semantic_audit,
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
        "formal_validation_disjointness": copy.deepcopy(snapshot["formal_validation_disjointness"]),
        "chinese_semantic_audit": copy.deepcopy(snapshot["chinese_semantic_audit"]),
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
        expected_sources = {str(row.get("source_id")) for row in rows if isinstance(row, Mapping)}
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
                    f"{phase}/{source_id} governed capacity underfill: {available} < {required}"
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
                "validation_prepared_identity": copy.deepcopy(evidence["validation_prepared"]),
                "per_source_capacity": closed_rows,
                "quality_audit": {
                    "audit_attestation_path": evidence["audit"]["path"],
                    "audit_attestation_sha256": evidence["audit"]["sha256"],
                    "attestation_fingerprint": evidence["audit"]["attestation_fingerprint"],
                    "passed": True,
                },
                "license_audit": {
                    "materialized_attribution_manifest_path": evidence["attribution"]["path"],
                    "materialized_attribution_manifest_sha256": evidence["attribution"]["sha256"],
                    "wikipedia_attribution_contract": copy.deepcopy(
                        evidence["attribution"]["wikipedia_contract"]
                    ),
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
        "Wikipedia CC-BY-SA-3.0/GFDL",
        "final launch config",
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
    semantic_audit = snapshot["chinese_semantic_audit"]
    semantic_gate = result.get("chinese_semantic_quality_gate")
    if (
        not isinstance(semantic_gate, dict)
        or semantic_audit.get("passed") is not True
        or semantic_audit.get("authorizes_training") is not False
        or semantic_audit.get("source_id") != CHINESE_SEMANTIC_SOURCE_ID
    ):
        raise ClosureError("readiness Chinese semantic quality gate is invalid")
    semantic_gate.update(
        {
            "status": "passed_authenticated_chinese_semantic_quality_audit",
            "observed": copy.deepcopy(semantic_audit),
            "passed": True,
            "authorizes_training": False,
        }
    )
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
                "near_duplicate_passed": (near.get("passed") is True and near.get("matches") == 0),
                "near_duplicate_threshold": formal["near_duplicate_threshold"],
            },
            "v3_final_frozen_validation_baseline": {
                "bundle_path": baseline["root"],
                "summary_sha256": baseline["summary"]["sha256"],
                "manifest_sha256": baseline["manifest"]["sha256"],
                "complete_sha256": baseline["complete"]["sha256"],
                "checkpoint_complete_sha256": baseline["checkpoint_complete_sha256"],
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
    if (
        pause.get("enforcement") != "external_governed_controller"
        or pause.get("controller_implemented") is not True
        or pause.get("current_launch_command_auto_pauses") is not False
        or pause.get("current_launch_command_runs_validation") is not False
        or capabilities.get("starts_training_when_explicitly_invoked") is not False
        or any(
            capabilities.get(field) is not True
            for field in (
                "automatically_pauses_at_policy_thresholds",
                "automatically_runs_checkpoint_validation",
                "automatically_enforces_post_launch_hard_stops",
            )
        )
    ):
        raise ClosureError("readiness governed controller capability contract differs")
    result.update(
        {
            "status": ("blocked_pending_calibration_license_acceptance_and_final_authorization"),
            "project_root": str(ROOT.resolve()),
            "required_capacity_attestation": str(capacity_path.resolve()),
            "capacity_attestation": copy.deepcopy(capacity_identity),
            "blockers": _remaining_blockers(result),
            "launch_enabled": False,
            "authorizes_training": False,
            "training_started": False,
            "launch_command_after_all_gates_pass": None,
            "launch_command_status": (
                "pending_final_config_calibration_license_acceptance_and_authorization"
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
        or semantic_gate.get("passed") is not True
        or semantic_gate.get("authorizes_training") is not False
        or pause.get("controller_implemented") is not True
    ):
        raise ClosureError("closure attempted to bypass a remaining launch gate")
    unsigned = dict(result)
    result["readiness_fingerprint"] = _canonical_sha256(unsigned)
    return result


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory or fail closed if no-replace is absent."""

    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise ClosureError(
            "atomic directory no-replace is unavailable on this platform"
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
        raise ClosureError(
            f"closure output appeared during publication: {destination}"
        )
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
        raise ClosureError(
            "filesystem does not support atomic directory no-replace"
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(destination),
    )


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
        for path in work.iterdir():
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        _fsync_directory(work)
        before_publish()
        if output.exists():
            raise ClosureError(f"closure output appeared during publication: {output}")
        _rename_directory_noreplace(work, output)
        _fsync_directory(output.parent)
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
            raise ClosureError(f"closure output already exists; choose a new directory: {output}")
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
