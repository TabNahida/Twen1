#!/usr/bin/env python3
"""Materialize one authenticated v4 semantic-noise exclusion projection.

This command is deliberately separate from both the semantic scanner and the
generic Base-corpus audit.  It accepts an externally pinned canonical
exclusion ledger, rebuilds the selected phase's complete ledger from the exact
parent corpus, proves a one-to-one corpus/attribution join, removes only the
recomputed train rows, and byte-preserves validation.

The resulting extracted corpus is *not* training-ready.  It requires a new,
independent Base audit and never copies or extends the parent audit's rejection
ledger.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, BinaryIO

from twen.data import audits as audit_module
from twen.data import phase_exclusion as phase_helpers
from twen.data.audits import DataAuditError, _normalize_text
from twen.data.sources import validate_extracted_base_corpus
from twen.utils import atomic_write_json, atomic_write_text, sha256_file

SCHEMA_VERSION = 1
ALGORITHM = "authenticated-chinese-semantic-exclusion-projection-v1"
ATTESTATION_KIND = "twen_v4_chinese_semantic_exclusion_attestation"
COMPLETE_KIND = "twen_v4_chinese_semantic_exclusion_complete"
OUTPUT_LEDGER_NAME = "semantic-exclusion-ledger.jsonl"
ATTESTATION_NAME = "semantic-exclusion-attestation.json"
SOURCE_MAP_SIDECAR = "source-map.json"
LICENSE_AUDIT_SIDECAR = "license-audit.json"
PHASES = frozenset({"primary", "cooldown"})

SEMANTIC_SCANNER_PATH = Path(__file__).with_name("audit_v4_chinese_semantic_noise.py")
NORMALIZER_SOURCE_PATH = Path(audit_module.__file__).resolve()
PHASE_HELPER_SOURCE_PATH = Path(phase_helpers.__file__).resolve()

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LEDGER_FIELDS = frozenset(
    {
        "phase",
        "source_id",
        "stable_id",
        "path",
        "line_number",
        "shard_sha256",
        "text_sha256",
        "normalized_text_sha256",
        "token_count",
        "token_count_field",
        "attribution_path",
        "attribution_line_number",
        "attribution_sha256",
        "reasons",
        "reason_occurrences",
        "ledger_entry_fingerprint",
    }
)
_MANIFEST_IDENTITY_FIELDS = (
    "recipe_id",
    "recipe_sha256",
    "resolved_source_lock_sha256",
    "tokenizer_manifest_sha256",
    "extractor_source_sha256",
    "profile",
    "sources",
    "train_files",
    "validation_files",
    "attribution_files",
    "file_lists",
    "source_map",
    "source_mix",
    "format_audit",
    "license_audit",
    "materialization_audit",
)
_BASE_REAUDIT_GATE_NAMES = frozenset(
    {
        "cross_source_exact_dedup",
        "cross_source_near_dedup",
        "deterministic_content_quality_scan",
        "full_contextual_pii_scan",
        "project_benchmark_13gram_scan",
        "train_vs_frozen_validation_exact_dedup",
    }
)
_AUDIT_PROJECTION_ALGORITHM = "authenticated-deletion-monotonic-parent-audit-projection-v1"
_PENDING_BASE_REAUDIT_STATUS = "pending_independent_reaudit_after_chinese_semantic_exclusion"


@dataclass(frozen=True, slots=True)
class _Scanner:
    module: ModuleType
    path: Path
    sha256: str
    conversion_markers: tuple[tuple[str, Any], ...]
    malformed_punctuation: Any


@dataclass(frozen=True, slots=True)
class _Ledger:
    path: Path
    size: int
    sha256: str
    documents: int
    tokens: int
    rows: tuple[dict[str, Any], ...]
    phase_rows: tuple[dict[str, Any], ...]
    phase_tokens: int

    @property
    def identity(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "size": self.size,
            "sha256": self.sha256,
            "documents": self.documents,
            "tokens": self.tokens,
        }


def _project_parent_audits(
    raw_parent_audits: object,
    raw_base_audit_gates: object,
) -> tuple[dict[str, str], dict[str, object]]:
    """Project parent governance through an authenticated deletion-only transform.

    Removing exact, authenticated train rows cannot invalidate immutable source
    identities, license/provenance decisions, declared source filters, or an
    already-passing cross-phase exclusion.  The six content-dependent Base audit
    gates are deliberately invalidated and must be rerun on the exact output.
    """

    if not isinstance(raw_parent_audits, Mapping) or not raw_parent_audits:
        raise DataAuditError("parent audit status inventory is missing")
    parent: dict[str, str] = {}
    for name, status in raw_parent_audits.items():
        if not isinstance(name, str) or not name or not isinstance(status, str) or not status:
            raise DataAuditError("parent audit status inventory is invalid")
        parent[name] = status
    if not isinstance(raw_base_audit_gates, Mapping) or not raw_base_audit_gates:
        raise DataAuditError("authenticated Base audit gate inventory is missing")
    base_gate_statuses: dict[str, str] = {}
    for name, gate in raw_base_audit_gates.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(gate, Mapping)
            or not isinstance(gate.get("status"), str)
            or not gate["status"]
        ):
            raise DataAuditError("authenticated Base audit gate inventory is invalid")
        base_gate_statuses[name] = str(gate["status"])
    dynamic_gate_names = frozenset(base_gate_statuses)
    if dynamic_gate_names != _BASE_REAUDIT_GATE_NAMES:
        raise DataAuditError(
            "authenticated Base audit gate inventory differs from the semantic "
            "exclusion projection contract"
        )
    unexpected_pending = sorted(
        name
        for name, status in parent.items()
        if name not in dynamic_gate_names and status.lower().startswith("pending")
    )
    if unexpected_pending:
        raise DataAuditError(
            "parent has unresolved non-Base audit statuses that a semantic "
            f"deletion cannot close: {unexpected_pending}"
        )

    output = dict(parent)
    invalidated: dict[str, dict[str, str | None]] = {}
    for name in sorted(dynamic_gate_names):
        invalidated[name] = {
            "parent_status": parent.get(name),
            "authenticated_base_audit_status": base_gate_statuses[name],
            "output_status": _PENDING_BASE_REAUDIT_STATUS,
        }
        output[name] = _PENDING_BASE_REAUDIT_STATUS
    added = {
        "chinese_semantic_exclusion": ("complete_authenticated_canonical_projection"),
        "validation_byte_preservation": "complete_sha256_identical",
    }
    output.update(added)
    preserved = {
        name: status for name, status in sorted(parent.items()) if name not in dynamic_gate_names
    }
    projection: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": _AUDIT_PROJECTION_ALGORITHM,
        "transform": "authenticated_train_row_deletion_only",
        "parent_status_inventory_sha256": _canonical_sha256(parent),
        "authenticated_base_audit_gate_statuses": dict(sorted(base_gate_statuses.items())),
        "preserved_parent_statuses": preserved,
        "invalidated_for_independent_base_reaudit": invalidated,
        "added_output_statuses": added,
    }
    return dict(sorted(output.items())), projection


@dataclass(frozen=True, slots=True)
class _AttributionBinding:
    record: Any
    inventory_sha256: str


_AttributionBucket = _AttributionBinding | deque[_AttributionBinding]
_AttributionQueues = dict[
    tuple[str, str],
    dict[str, _AttributionBucket],
]


@dataclass(slots=True)
class _Projection:
    train_files: list[dict[str, object]]
    validation_files: list[dict[str, object]]
    attribution_files: list[dict[str, object]]
    retained_rows: dict[str, dict[str, int]]
    retained_tokens: dict[str, dict[str, int]]
    excluded_rows: dict[str, int]
    excluded_tokens: dict[str, int]
    excluded_attribution_locations: set[tuple[str, int]]
    rebuilt_ledger_rows: list[dict[str, Any]]
    documents_before: dict[str, int]
    tokens_before: dict[str, int]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _manifest_fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DataAuditError(f"{label} must be a lowercase SHA256")
    return value


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataAuditError(f"{label} must be a non-negative integer")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    result = _require_nonnegative_int(value, label=label)
    if result <= 0:
        raise DataAuditError(f"{label} must be positive")
    return result


def _safe_relative(value: object, *, label: str) -> str:
    return phase_helpers._safe_relative(value, label)


def _file_identity(path: Path, *, relative: str) -> dict[str, object]:
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _inventory_identities(
    phase: Any,
    *,
    role: str,
) -> dict[str, tuple[int, str]]:
    raw_inventory = phase.value.get(f"{role}_files")
    if not isinstance(raw_inventory, list) or not raw_inventory:
        raise DataAuditError(f"parent {role} inventory is missing")
    result: dict[str, tuple[int, str]] = {}
    for index, raw in enumerate(raw_inventory):
        if not isinstance(raw, Mapping):
            raise DataAuditError(f"{role}_files[{index}] is invalid")
        relative = _safe_relative(
            raw.get("path"),
            label=f"{role}_files[{index}].path",
        )
        if relative in result:
            raise DataAuditError(f"parent {role} inventory repeats a path")
        result[relative] = (
            _require_nonnegative_int(
                raw.get("size"),
                label=f"{role}_files[{index}].size",
            ),
            _require_sha256(
                raw.get("sha256"),
                label=f"{role}_files[{index}].sha256",
            ),
        )
    return result


def _verify_consumed_stream(
    *,
    role: str,
    relative: str,
    expected_size: int,
    expected_sha256: str,
    consumed_size: int,
    consumed_sha256: str,
) -> None:
    if consumed_size != expected_size or consumed_sha256 != expected_sha256:
        raise DataAuditError(
            f"parent {role} consumed stream identity differs from manifest inventory: {relative}"
        )


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataAuditError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DataAuditError(f"{label} must be a JSON object: {path}")
    return value


def _source_identity_snapshot() -> dict[str, str]:
    return {
        "materializer_source_sha256": sha256_file(Path(__file__).resolve()),
        "semantic_scanner_source_sha256": sha256_file(SEMANTIC_SCANNER_PATH.resolve()),
        "normalizer_source_sha256": sha256_file(NORMALIZER_SOURCE_PATH),
        "phase_helper_source_sha256": sha256_file(PHASE_HELPER_SOURCE_PATH),
    }


def _load_scanner(*, expected_sha256: str) -> _Scanner:
    path = SEMANTIC_SCANNER_PATH.resolve()
    if SEMANTIC_SCANNER_PATH.is_symlink() or not path.is_file():
        raise DataAuditError("semantic scanner must be a regular non-symlink file")
    payload = path.read_bytes()
    digest = _sha256_bytes(payload)
    if digest != _require_sha256(
        expected_sha256,
        label="expected semantic scanner SHA256",
    ):
        raise DataAuditError("semantic scanner external pin mismatch")
    if path.read_bytes() != payload:
        raise DataAuditError("semantic scanner changed before exact-payload execution")
    module_name = f"_twen_semantic_scanner_{digest}"
    module = ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    module.__loader__ = None
    module.__spec__ = None
    module.__cached__ = None
    code = compile(payload, str(path), "exec", dont_inherit=True)
    missing = object()
    previous_module = sys.modules.get(module_name, missing)
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
    finally:
        if previous_module is missing:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
    if path.read_bytes() != payload:
        raise DataAuditError("semantic scanner changed during exact-payload execution")
    raw_markers = getattr(module, "_CONVERSION_MARKERS", None)
    malformed = getattr(module, "_MALFORMED_PUNCTUATION", None)
    if not isinstance(raw_markers, tuple) or not raw_markers:
        raise DataAuditError("semantic scanner conversion marker inventory is invalid")
    markers: list[tuple[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_markers):
        if (
            not isinstance(raw, tuple)
            or len(raw) != 2
            or not isinstance(raw[0], str)
            or not raw[0]
            or raw[0] in names
            or not callable(getattr(raw[1], "findall", None))
        ):
            raise DataAuditError(f"semantic scanner conversion marker {index} is invalid")
        names.add(raw[0])
        markers.append((raw[0], raw[1]))
    if "malformed_punctuation" in names or not callable(getattr(malformed, "findall", None)):
        raise DataAuditError("semantic scanner malformed-punctuation detector is invalid")
    return _Scanner(
        module=module,
        path=path,
        sha256=digest,
        conversion_markers=tuple(markers),
        malformed_punctuation=malformed,
    )


def _semantic_reasons(
    text: str,
    *,
    scanner: _Scanner,
) -> tuple[list[str], dict[str, int]]:
    occurrences: dict[str, int] = {}
    for name, pattern in scanner.conversion_markers:
        count = len(pattern.findall(text))
        if count:
            occurrences[name] = count
    malformed_count = len(scanner.malformed_punctuation.findall(text))
    if malformed_count:
        occurrences["malformed_punctuation"] = malformed_count
    reasons = sorted(occurrences)
    return reasons, {name: occurrences[name] for name in reasons}


def _ledger_sort_key(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row["phase"]),
        str(row["path"]),
        int(row["line_number"]),
        str(row["stable_id"]),
    )


def _validate_ledger_row(
    value: object,
    *,
    index: int,
    allowed_source_id: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _LEDGER_FIELDS:
        raise DataAuditError(f"exclusion ledger row {index} has an invalid schema")
    phase = value.get("phase")
    source_id = value.get("source_id")
    if phase not in PHASES or source_id != allowed_source_id:
        raise DataAuditError(f"exclusion ledger row {index} has an invalid phase/source_id")
    for field in (
        "stable_id",
        "shard_sha256",
        "text_sha256",
        "normalized_text_sha256",
        "attribution_sha256",
        "ledger_entry_fingerprint",
    ):
        _require_sha256(value.get(field), label=f"ledger[{index}].{field}")
    for field in ("path", "attribution_path"):
        value[field] = _safe_relative(
            value.get(field),
            label=f"ledger[{index}].{field}",
        )
    for field in ("line_number", "attribution_line_number", "token_count"):
        value[field] = _require_positive_int(
            value.get(field),
            label=f"ledger[{index}].{field}",
        )
    if value.get("token_count_field") != "token_count_with_eos":
        raise DataAuditError(f"ledger[{index}].token_count_field must be token_count_with_eos")
    reasons = value.get("reasons")
    reason_occurrences = value.get("reason_occurrences")
    if (
        not isinstance(reasons, list)
        or not reasons
        or reasons != sorted(set(reasons))
        or not all(isinstance(item, str) and item for item in reasons)
        or not isinstance(reason_occurrences, dict)
        or list(reason_occurrences) != sorted(reason_occurrences)
        or set(reason_occurrences) != set(reasons)
    ):
        raise DataAuditError(f"ledger[{index}] reason inventory is invalid")
    for name, count in reason_occurrences.items():
        _require_positive_int(count, label=f"ledger[{index}].reason_occurrences.{name}")
    unsigned = dict(value)
    fingerprint = unsigned.pop("ledger_entry_fingerprint")
    if fingerprint != _canonical_sha256(unsigned):
        raise DataAuditError(f"ledger[{index}] entry fingerprint mismatch")
    return value


def _load_canonical_ledger(
    path: Path,
    *,
    expected_sha256: str,
    expected_phase: str,
    source_id: str,
    expected_phase_documents: int,
    expected_phase_tokens: int,
) -> _Ledger:
    raw_path = path.expanduser()
    if raw_path.is_symlink():
        raise DataAuditError("canonical exclusion ledger must not be a symlink")
    ledger_path = raw_path.resolve()
    if not ledger_path.is_file():
        raise DataAuditError(f"canonical exclusion ledger is missing: {ledger_path}")
    payload = ledger_path.read_bytes()
    digest = _sha256_bytes(payload)
    if digest != _require_sha256(
        expected_sha256,
        label="expected canonical exclusion ledger SHA256",
    ):
        raise DataAuditError("canonical exclusion ledger external pin mismatch")
    rows: list[dict[str, Any]] = []
    previous_key: tuple[str, str, int, str] | None = None
    locations: set[tuple[str, str, int]] = set()
    attribution_locations: set[tuple[str, str, int]] = set()
    stable_keys: set[tuple[str, str, str]] = set()
    offset = 0
    for index, raw_line in enumerate(payload.splitlines(keepends=True), start=1):
        offset += len(raw_line)
        if not raw_line.endswith(b"\n") or not raw_line.strip():
            raise DataAuditError(f"canonical exclusion ledger row {index} is blank or unterminated")
        try:
            value = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DataAuditError(f"canonical exclusion ledger row {index} is invalid JSON") from exc
        row = _validate_ledger_row(
            value,
            index=index,
            allowed_source_id=source_id,
        )
        canonical_line = _canonical_bytes(row) + b"\n"
        if canonical_line != raw_line:
            raise DataAuditError(f"canonical exclusion ledger row {index} is not canonical JSONL")
        key = _ledger_sort_key(row)
        if previous_key is not None and key <= previous_key:
            raise DataAuditError("canonical exclusion ledger is unsorted or duplicated")
        previous_key = key
        location = (str(row["phase"]), str(row["path"]), int(row["line_number"]))
        attribution_location = (
            str(row["phase"]),
            str(row["attribution_path"]),
            int(row["attribution_line_number"]),
        )
        stable_key = (
            str(row["phase"]),
            str(row["source_id"]),
            str(row["stable_id"]),
        )
        if (
            location in locations
            or attribution_location in attribution_locations
            or stable_key in stable_keys
        ):
            raise DataAuditError(
                "canonical exclusion ledger contains a duplicate document, "
                "attribution, or stable identity"
            )
        locations.add(location)
        attribution_locations.add(attribution_location)
        stable_keys.add(stable_key)
        rows.append(row)
    if offset != len(payload) or not rows:
        raise DataAuditError("canonical exclusion ledger is empty or incompletely parsed")
    phase_rows = tuple(row for row in rows if row["phase"] == expected_phase)
    phase_tokens = sum(int(row["token_count"]) for row in phase_rows)
    if len(phase_rows) != _require_nonnegative_int(
        expected_phase_documents,
        label="expected exclusion count",
    ):
        raise DataAuditError("canonical exclusion ledger phase document count differs")
    if phase_tokens != _require_nonnegative_int(
        expected_phase_tokens,
        label="expected excluded tokens",
    ):
        raise DataAuditError("canonical exclusion ledger phase token count differs")
    return _Ledger(
        path=ledger_path,
        size=len(payload),
        sha256=digest,
        documents=len(rows),
        tokens=sum(int(row["token_count"]) for row in rows),
        rows=tuple(rows),
        phase_rows=phase_rows,
        phase_tokens=phase_tokens,
    )


def _authenticate_input(
    *,
    parent_manifest: Path,
    base_audit: Path,
    expected_parent_manifest_sha256: str,
    expected_base_audit_sha256: str,
) -> Any:
    phase = phase_helpers._authenticate_phase_input(parent_manifest, base_audit)
    expected_manifest_sha256 = _require_sha256(
        expected_parent_manifest_sha256,
        label="expected parent manifest SHA256",
    )
    if phase.manifest_sha256 != expected_manifest_sha256:
        raise DataAuditError("parent manifest external pin mismatch")
    if phase.audit_sha256 != _require_sha256(
        expected_base_audit_sha256,
        label="expected base audit SHA256",
    ):
        raise DataAuditError("base audit external pin mismatch")
    try:
        manifest_payload = phase.manifest_path.read_bytes()
    except OSError as exc:
        raise DataAuditError(
            f"cannot consume authenticated parent manifest: {phase.manifest_path}"
        ) from exc
    if _sha256_bytes(manifest_payload) != expected_manifest_sha256:
        raise DataAuditError("parent manifest consumed payload differs from authenticated identity")
    try:
        authenticated_value = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataAuditError("authenticated parent manifest payload is not valid JSON") from exc
    if not isinstance(authenticated_value, dict):
        raise DataAuditError("authenticated parent manifest payload must be a JSON object")
    if phase.value != authenticated_value:
        raise DataAuditError("parent manifest value changed between authentication and consumption")
    return replace(phase, value=authenticated_value)


def _attribution_queues(
    phase: Any,
) -> _AttributionQueues:
    identity_by_path = _inventory_identities(phase, role="attribution")
    queues: _AttributionQueues = {}
    for path, relative in phase_helpers._attribution_inventory(phase):
        inventory_identity = identity_by_path.get(relative)
        if inventory_identity is None:
            raise DataAuditError("attribution helper inventory differs from manifest")
        inventory_size, inventory_sha = inventory_identity
        stream_sha256 = hashlib.sha256()
        consumed_size = 0
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                stream_sha256.update(raw)
                consumed_size += len(raw)
                if not raw.strip():
                    raise DataAuditError(f"blank attribution JSONL row at {path}:{line_number}")
                record = phase_helpers._parse_attribution(
                    raw,
                    path=path,
                    relative=relative,
                    line_number=line_number,
                )
                owner_key = (record.split, record.source_id)
                binding = _AttributionBinding(
                    record=record,
                    inventory_sha256=inventory_sha,
                )
                owner_queues = queues.setdefault(owner_key, {})
                bucket = owner_queues.get(record.text_sha256)
                if bucket is None:
                    owner_queues[record.text_sha256] = binding
                elif isinstance(bucket, deque):
                    bucket.append(binding)
                else:
                    owner_queues[record.text_sha256] = deque((bucket, binding))
        _verify_consumed_stream(
            role="attribution",
            relative=relative,
            expected_size=inventory_size,
            expected_sha256=inventory_sha,
            consumed_size=consumed_size,
            consumed_sha256=stream_sha256.hexdigest(),
        )
    return queues


def _pop_attribution(
    queues: _AttributionQueues,
    *,
    role: str,
    source_id: str,
    text_sha256: str,
) -> _AttributionBinding | None:
    owner_key = (role, source_id)
    owner_queues = queues.get(owner_key)
    if owner_queues is None:
        return None
    bucket = owner_queues.get(text_sha256)
    if bucket is None:
        return None
    if isinstance(bucket, deque):
        binding = bucket.popleft()
        if len(bucket) == 1:
            owner_queues[text_sha256] = bucket[0]
    else:
        binding = bucket
        del owner_queues[text_sha256]
    if not owner_queues:
        del queues[owner_key]
    return binding


def _remaining_attribution(queues: _AttributionQueues) -> int:
    return sum(
        len(bucket) if isinstance(bucket, deque) else 1
        for owner_queues in queues.values()
        for bucket in owner_queues.values()
    )


def _parse_document(raw: bytes, *, path: Path, line_number: int) -> str:
    return phase_helpers._parse_document_text(raw, path, line_number)


def _build_ledger_row(
    *,
    phase_name: str,
    source_id: str,
    relative: str,
    line_number: int,
    shard_sha256: str,
    text: str,
    normalized_sha256: str,
    attribution: _AttributionBinding,
    reasons: list[str],
    occurrences: dict[str, int],
) -> dict[str, Any]:
    record = attribution.record
    value: dict[str, Any] = {
        "phase": phase_name,
        "source_id": source_id,
        "stable_id": record.stable_id,
        "path": relative,
        "line_number": line_number,
        "shard_sha256": shard_sha256,
        "text_sha256": _sha256_bytes(text.encode("utf-8")),
        "normalized_text_sha256": normalized_sha256,
        "token_count": record.token_count,
        "token_count_field": "token_count_with_eos",
        "attribution_path": record.relative,
        "attribution_line_number": record.line_number,
        "attribution_sha256": attribution.inventory_sha256,
        "reasons": reasons,
        "reason_occurrences": occurrences,
    }
    value["ledger_entry_fingerprint"] = _canonical_sha256(value)
    return value


def _open_output(path: Path) -> BinaryIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    return os.fdopen(descriptor, "wb", closefd=True)


def _copy_documents(
    *,
    phase: Any,
    phase_name: str,
    source_id: str,
    scanner: _Scanner,
    work: Path,
    role: str,
    owners: Mapping[str, str],
    categories: Mapping[str, str],
    queues: _AttributionQueues,
    excluded_attribution_locations: set[tuple[str, int]],
    rebuilt_rows: list[dict[str, Any]],
    retained_rows: dict[str, int],
    retained_tokens: dict[str, int],
    excluded_rows: dict[str, int],
    excluded_tokens: dict[str, int],
    documents_before: dict[str, int],
    tokens_before: dict[str, int],
) -> list[dict[str, object]]:
    raw_inventory = phase.value.get(f"{role}_files")
    if not isinstance(raw_inventory, list):
        raise DataAuditError(f"parent {role} inventory is invalid")
    result: list[dict[str, object]] = []
    for index, raw_entry in enumerate(raw_inventory):
        if not isinstance(raw_entry, Mapping):
            raise DataAuditError(f"parent {role}_files[{index}] is invalid")
        relative = _safe_relative(
            raw_entry.get("path"),
            label=f"{role}_files[{index}].path",
        )
        owner = owners.get(relative)
        category = categories.get(str(owner))
        if owner is None or category is None:
            raise DataAuditError(f"source ownership/category is missing for {role} file {relative}")
        shard_sha = _require_sha256(
            raw_entry.get("sha256"),
            label=f"{role}_files[{index}].sha256",
        )
        shard_size = _require_nonnegative_int(
            raw_entry.get("size"),
            label=f"{role}_files[{index}].size",
        )
        source = phase_helpers._owned_regular_file(
            phase.manifest_path.parent,
            relative,
            f"parent {role} file",
        )
        output = work / relative
        code = category == "code" or "code" in owner.casefold()
        stream_sha256 = hashlib.sha256()
        consumed_size = 0
        with source.open("rb") as source_handle, _open_output(output) as output_handle:
            for line_number, raw in enumerate(source_handle, start=1):
                stream_sha256.update(raw)
                consumed_size += len(raw)
                if not raw.strip():
                    raise DataAuditError(f"blank corpus JSONL row at {source}:{line_number}")
                text = _parse_document(raw, path=source, line_number=line_number)
                normalized = _normalize_text(
                    text,
                    code=False if owner == source_id else code,
                )
                normalized_sha = _sha256_bytes(normalized.encode("utf-8"))
                binding = _pop_attribution(
                    queues,
                    role=role,
                    source_id=owner,
                    text_sha256=normalized_sha,
                )
                if binding is None:
                    raise DataAuditError(
                        "parent attribution does not cover corpus document exactly: "
                        f"{relative}:{line_number}"
                    )
                record = binding.record
                documents_before[owner] += 1
                tokens_before[owner] += record.token_count
                reasons: list[str] = []
                occurrences: dict[str, int] = {}
                if role == "train" and owner == source_id:
                    reasons, occurrences = _semantic_reasons(text, scanner=scanner)
                if reasons:
                    location = (record.relative, record.line_number)
                    if location in excluded_attribution_locations:
                        raise DataAuditError(
                            "semantic exclusion maps multiple documents to one attribution"
                        )
                    excluded_attribution_locations.add(location)
                    excluded_rows[owner] += 1
                    excluded_tokens[owner] += record.token_count
                    rebuilt_rows.append(
                        _build_ledger_row(
                            phase_name=phase_name,
                            source_id=owner,
                            relative=relative,
                            line_number=line_number,
                            shard_sha256=shard_sha,
                            text=text,
                            normalized_sha256=normalized_sha,
                            attribution=binding,
                            reasons=reasons,
                            occurrences=occurrences,
                        )
                    )
                    continue
                output_handle.write(raw)
                retained_rows[owner] += 1
                retained_tokens[owner] += record.token_count
            output_handle.flush()
            os.fsync(output_handle.fileno())
        _verify_consumed_stream(
            role=role,
            relative=relative,
            expected_size=shard_size,
            expected_sha256=shard_sha,
            consumed_size=consumed_size,
            consumed_sha256=stream_sha256.hexdigest(),
        )
        if role == "validation" and (
            output.stat().st_size != source.stat().st_size or sha256_file(output) != shard_sha
        ):
            raise DataAuditError(f"validation was not byte-preserved: {relative}")
        result.append(_file_identity(output, relative=relative))
    return result


def _write_filtered_attribution(
    *,
    phase: Any,
    work: Path,
    excluded_locations: set[tuple[str, int]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    excluded_written = 0
    identity_by_path = _inventory_identities(phase, role="attribution")
    for source, relative in phase_helpers._attribution_inventory(phase):
        inventory_identity = identity_by_path.get(relative)
        if inventory_identity is None:
            raise DataAuditError("attribution helper inventory differs from manifest")
        inventory_size, inventory_sha = inventory_identity
        output = work / relative
        stream_sha256 = hashlib.sha256()
        consumed_size = 0
        with source.open("rb") as source_handle, _open_output(output) as output_handle:
            for line_number, raw in enumerate(source_handle, start=1):
                stream_sha256.update(raw)
                consumed_size += len(raw)
                record = phase_helpers._parse_attribution(
                    raw,
                    path=source,
                    relative=relative,
                    line_number=line_number,
                )
                if record.split == "train" and (relative, line_number) in excluded_locations:
                    excluded_written += 1
                    continue
                output_handle.write(raw)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        _verify_consumed_stream(
            role="attribution",
            relative=relative,
            expected_size=inventory_size,
            expected_sha256=inventory_sha,
            consumed_size=consumed_size,
            consumed_sha256=stream_sha256.hexdigest(),
        )
        result.append(_file_identity(output, relative=relative))
    if excluded_written != len(excluded_locations):
        raise DataAuditError(
            "filtered attribution did not remove every semantic exclusion exactly once"
        )
    return result


def _project_phase(
    *,
    phase: Any,
    phase_name: str,
    source_id: str,
    scanner: _Scanner,
    work: Path,
) -> _Projection:
    owners = phase_helpers._source_owners(phase.value)
    categories = phase_helpers._source_categories(phase.value)
    if source_id not in categories or source_id not in set(owners["train"].values()):
        raise DataAuditError(f"semantic source_id is absent from parent train data: {source_id}")
    queues = _attribution_queues(phase)
    retained_rows: dict[str, dict[str, int]] = {
        "train": defaultdict(int),
        "validation": defaultdict(int),
    }
    retained_tokens: dict[str, dict[str, int]] = {
        "train": defaultdict(int),
        "validation": defaultdict(int),
    }
    excluded_rows: dict[str, int] = defaultdict(int)
    excluded_tokens: dict[str, int] = defaultdict(int)
    documents_before: dict[str, int] = defaultdict(int)
    tokens_before: dict[str, int] = defaultdict(int)
    excluded_locations: set[tuple[str, int]] = set()
    rebuilt_rows: list[dict[str, Any]] = []
    train_files = _copy_documents(
        phase=phase,
        phase_name=phase_name,
        source_id=source_id,
        scanner=scanner,
        work=work,
        role="train",
        owners=owners["train"],
        categories=categories,
        queues=queues,
        excluded_attribution_locations=excluded_locations,
        rebuilt_rows=rebuilt_rows,
        retained_rows=retained_rows["train"],
        retained_tokens=retained_tokens["train"],
        excluded_rows=excluded_rows,
        excluded_tokens=excluded_tokens,
        documents_before=documents_before,
        tokens_before=tokens_before,
    )
    validation_files = _copy_documents(
        phase=phase,
        phase_name=phase_name,
        source_id=source_id,
        scanner=scanner,
        work=work,
        role="validation",
        owners=owners["validation"],
        categories=categories,
        queues=queues,
        excluded_attribution_locations=excluded_locations,
        rebuilt_rows=rebuilt_rows,
        retained_rows=retained_rows["validation"],
        retained_tokens=retained_tokens["validation"],
        excluded_rows=excluded_rows,
        excluded_tokens=excluded_tokens,
        documents_before=documents_before,
        tokens_before=tokens_before,
    )
    unmatched = _remaining_attribution(queues)
    if unmatched:
        raise DataAuditError(
            f"parent attribution has no exact corpus document for {unmatched} rows"
        )
    rebuilt_rows.sort(key=_ledger_sort_key)
    attribution_files = _write_filtered_attribution(
        phase=phase,
        work=work,
        excluded_locations=excluded_locations,
    )
    return _Projection(
        train_files=train_files,
        validation_files=validation_files,
        attribution_files=attribution_files,
        retained_rows={role: dict(values) for role, values in retained_rows.items()},
        retained_tokens={role: dict(values) for role, values in retained_tokens.items()},
        excluded_rows=dict(excluded_rows),
        excluded_tokens=dict(excluded_tokens),
        excluded_attribution_locations=excluded_locations,
        rebuilt_ledger_rows=rebuilt_rows,
        documents_before=dict(documents_before),
        tokens_before=dict(tokens_before),
    )


def _compare_rebuilt_ledger(
    rebuilt: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
) -> None:
    if list(rebuilt) == list(expected):
        return
    rebuilt_by_location = {(row["path"], row["line_number"]): row for row in rebuilt}
    expected_by_location = {(row["path"], row["line_number"]): row for row in expected}
    omitted = sorted(set(rebuilt_by_location) - set(expected_by_location))
    extra = sorted(set(expected_by_location) - set(rebuilt_by_location))
    stale = sorted(
        location
        for location in set(rebuilt_by_location) & set(expected_by_location)
        if rebuilt_by_location[location] != expected_by_location[location]
    )
    raise DataAuditError(
        "canonical exclusion ledger differs from complete scanner rebuild: "
        f"omitted={len(omitted)}, extra={len(extra)}, stale_or_field_mismatch={len(stale)}"
    )


def _validate_parent_accounting(phase: Any, projection: _Projection) -> None:
    raw_sources = phase.value.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise DataAuditError("parent source accounting is invalid")
    observed_ids: set[str] = set()
    train_total = 0
    validation_total = 0
    train_documents = 0
    validation_documents = 0
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, Mapping):
            raise DataAuditError(f"parent sources[{index}] is invalid")
        source_id = raw.get("source_id")
        if not isinstance(source_id, str) or not source_id or source_id in observed_ids:
            raise DataAuditError(f"parent sources[{index}] source_id is invalid")
        observed_ids.add(source_id)
        train_tokens = projection.retained_tokens["train"].get(
            source_id, 0
        ) + projection.excluded_tokens.get(source_id, 0)
        train_rows = projection.retained_rows["train"].get(
            source_id, 0
        ) + projection.excluded_rows.get(source_id, 0)
        validation_tokens = projection.retained_tokens["validation"].get(source_id, 0)
        validation_rows = projection.retained_rows["validation"].get(source_id, 0)
        if (
            raw.get("actual_train_tokens") != train_tokens
            or raw.get("actual_validation_tokens") != validation_tokens
            or raw.get("train_rows") != train_rows
            or raw.get("validation_rows") != validation_rows
        ):
            raise DataAuditError(
                f"parent source accounting differs from corpus/attribution join: {source_id}"
            )
        train_total += train_tokens
        validation_total += validation_tokens
        train_documents += train_rows
        validation_documents += validation_rows
    if (
        phase.value.get("actual_train_tokens") != train_total
        or phase.value.get("actual_validation_tokens") != validation_total
        or phase.value.get("actual_train_documents") != train_documents
        or phase.value.get("actual_validation_documents") != validation_documents
    ):
        raise DataAuditError("parent aggregate accounting differs from authenticated rows")


def _capacity_report(
    *,
    projection: _Projection,
    required_source_tokens: Mapping[str, int],
    required_aggregate_tokens: int,
    formal_training_tokens: int,
    global_batch_tokens: int,
) -> dict[str, Any]:
    retained = projection.retained_tokens["train"]
    actual_sources = set(retained)
    required_sources = set(required_source_tokens)
    if required_sources != actual_sources:
        raise DataAuditError(
            "required per-source capacity pins differ from parent source inventory "
            f"(missing={sorted(actual_sources - required_sources)}, "
            f"extra={sorted(required_sources - actual_sources)})"
        )
    source_rows: list[dict[str, Any]] = []
    for source_id in sorted(actual_sources):
        required = _require_nonnegative_int(
            required_source_tokens[source_id],
            label=f"required_source_tokens.{source_id}",
        )
        actual = retained[source_id]
        if actual < required:
            raise DataAuditError(
                f"semantic exclusion violates required source capacity: {source_id}"
            )
        source_rows.append(
            {
                "source_id": source_id,
                "required_train_tokens": required,
                "retained_train_tokens": actual,
                "surplus_tokens": actual - required,
                "passed": True,
            }
        )
    aggregate_required = _require_nonnegative_int(
        required_aggregate_tokens,
        label="required aggregate tokens",
    )
    training_required = _require_nonnegative_int(
        formal_training_tokens,
        label="formal training tokens",
    )
    batch_required = _require_positive_int(
        global_batch_tokens,
        label="global batch tokens",
    )
    aggregate_actual = sum(retained.values())
    training_plus_batch = training_required + batch_required
    if aggregate_actual < aggregate_required:
        raise DataAuditError("semantic exclusion violates aggregate materialization capacity")
    if aggregate_actual < training_plus_batch:
        raise DataAuditError(
            "semantic exclusion violates formal-training plus one-global-batch capacity"
        )
    return {
        "sources": source_rows,
        "all_per_source_passed": True,
        "aggregate": {
            "retained_train_tokens": aggregate_actual,
            "required_aggregate_tokens": aggregate_required,
            "aggregate_surplus_tokens": aggregate_actual - aggregate_required,
            "formal_training_tokens": training_required,
            "global_batch_tokens": batch_required,
            "formal_training_plus_global_batch_tokens": training_plus_batch,
            "formal_training_plus_global_batch_surplus_tokens": (
                aggregate_actual - training_plus_batch
            ),
            "passed": True,
        },
    }


def _source_map(
    *,
    parent: Mapping[str, Any],
    projection: _Projection,
    owners: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    parent_map = parent.get("source_map")
    if not isinstance(parent_map, Mapping):
        raise DataAuditError("parent source_map is invalid")
    roles = {
        "train": [
            {
                "source_id": owners["train"][str(entry["path"])],
                **entry,
            }
            for entry in projection.train_files
        ],
        "validation": [
            {
                "source_id": owners["validation"][str(entry["path"])],
                **entry,
            }
            for entry in projection.validation_files
        ],
    }
    unsigned = {
        "schema_version": parent_map.get("schema_version"),
        "algorithm": parent_map.get("algorithm"),
        "roles": roles,
    }
    return {**unsigned, "fingerprint": _manifest_fingerprint(unsigned)}


def _source_mix(
    *,
    parent: Mapping[str, Any],
    profile: str,
    projection: _Projection,
) -> dict[str, Any]:
    raw = parent.get("source_mix")
    if not isinstance(raw, Mapping) or not isinstance(raw.get("sources"), list):
        raise DataAuditError("parent source_mix is invalid")
    sources: list[dict[str, Any]] = []
    for index, value in enumerate(raw["sources"]):
        if not isinstance(value, Mapping):
            raise DataAuditError(f"parent source_mix.sources[{index}] is invalid")
        copied = deepcopy(dict(value))
        source_id = copied.get("source_id")
        if not isinstance(source_id, str):
            raise DataAuditError(f"parent source_mix.sources[{index}] has no source_id")
        copied["actual_train_tokens"] = projection.retained_tokens["train"].get(
            source_id,
            0,
        )
        sources.append(copied)
    unsigned = {
        "schema_version": raw.get("schema_version"),
        "algorithm": raw.get("algorithm"),
        "unit": raw.get("unit"),
        "basis_points_total": raw.get("basis_points_total"),
        "profile": profile,
        "sources": sources,
    }
    return {**unsigned, "fingerprint": _manifest_fingerprint(unsigned)}


def _rebuilt_sources(
    *,
    parent: Mapping[str, Any],
    source_map: Mapping[str, Any],
    projection: _Projection,
) -> list[dict[str, Any]]:
    raw_sources = parent.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise DataAuditError("parent sources are invalid")
    roles = source_map["roles"]
    result: list[dict[str, Any]] = []
    for raw in raw_sources:
        if not isinstance(raw, Mapping):
            raise DataAuditError("parent source entry is invalid")
        source_id = raw.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise DataAuditError("parent source_id is invalid")
        copied = deepcopy(dict(raw))
        copied["actual_train_tokens"] = projection.retained_tokens["train"].get(
            source_id,
            0,
        )
        copied["actual_validation_tokens"] = projection.retained_tokens["validation"].get(
            source_id, 0
        )
        copied["train_rows"] = projection.retained_rows["train"].get(source_id, 0)
        copied["validation_rows"] = projection.retained_rows["validation"].get(
            source_id,
            0,
        )
        outputs = [
            {
                "path": entry["path"],
                "size": entry["size"],
                "sha256": entry["sha256"],
            }
            for role in ("train", "validation")
            for entry in roles[role]
            if entry["source_id"] == source_id
        ]
        copied["chunks"] = [
            {
                "shard_id": "chinese-semantic-excluded",
                "outputs": outputs,
                "statistics": {
                    "excluded_train_rows": projection.excluded_rows.get(
                        source_id,
                        0,
                    ),
                    "excluded_train_tokens": projection.excluded_tokens.get(
                        source_id,
                        0,
                    ),
                },
            }
        ]
        result.append(copied)
    return result


def _write_canonical_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with _open_output(path) as handle:
        for row in rows:
            handle.write(_canonical_bytes(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_file_lists(
    work: Path,
    inventories: Mapping[str, list[dict[str, object]]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for role in ("train", "validation", "attribution"):
        path = work / f"{role}-files.txt"
        atomic_write_text(
            path,
            "".join(f"{entry['path']}\n" for entry in inventories[role]),
        )
        result[role] = _file_identity(path, relative=path.name)
    return result


def _write_json_sidecar(path: Path, value: Mapping[str, Any]) -> dict[str, object]:
    atomic_write_json(path, value)
    return _file_identity(path, relative=path.name)


def _output_path(value: str | Path, *, phase: Any) -> Path:
    requested = Path(value).expanduser()
    if requested.is_symlink():
        raise DataAuditError("semantic exclusion output must not be a symlink")
    output = requested.absolute()
    if output.exists():
        raise DataAuditError(f"semantic exclusion output already exists: {output}")
    try:
        output.resolve(strict=False).relative_to(phase.manifest_path.parent)
    except ValueError:
        pass
    else:
        raise DataAuditError("semantic exclusion output overlaps its parent corpus")
    return output


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        _fsync_directory(directory)
    _fsync_directory(root)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise DataAuditError(
            "atomic directory no-replace is unavailable; refusing racy publication"
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
        raise DataAuditError(
            f"semantic exclusion output appeared during publication: {destination}"
        )
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
        raise DataAuditError("filesystem lacks atomic directory no-replace; refusing publication")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _parse_required_source_tokens(values: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, value in enumerate(values):
        source_id, separator, raw_tokens = value.partition("=")
        if not separator or not source_id or source_id in result or not raw_tokens.isdigit():
            raise DataAuditError(
                f"--required-source-token[{index}] must be unique SOURCE_ID=TOKENS"
            )
        result[source_id] = int(raw_tokens)
    if not result:
        raise DataAuditError("at least one --required-source-token is required")
    return result


def _verify_output_sidecars(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    identities: Mapping[str, Any],
) -> None:
    expected = {
        "source_map": (SOURCE_MAP_SIDECAR, manifest.get("source_map")),
        "license_audit": (
            LICENSE_AUDIT_SIDECAR,
            manifest.get("license_audit"),
        ),
    }
    if set(identities) != set(expected):
        raise DataAuditError("semantic exclusion sidecar inventory is invalid")
    for name, (expected_name, expected_value) in expected.items():
        raw = identities[name]
        if not isinstance(raw, Mapping) or raw.get("path") != expected_name:
            raise DataAuditError(f"semantic exclusion {name} sidecar identity is invalid")
        path = phase_helpers._owned_regular_file(
            root,
            expected_name,
            f"semantic exclusion {name} sidecar",
        )
        if (
            raw.get("size") != path.stat().st_size
            or raw.get("sha256") != sha256_file(path)
            or _json_object(path, label=f"{name} sidecar") != expected_value
        ):
            raise DataAuditError(f"semantic exclusion {name} sidecar differs")


def validate_semantic_exclusion_output(
    manifest_path: str | Path,
) -> Mapping[str, Any]:
    """Authenticate a completed semantic-exclusion projection."""

    manifest_file = Path(manifest_path).resolve()
    report = validate_extracted_base_corpus(manifest_file, verify_hashes=True)
    manifest = _json_object(manifest_file, label="semantic exclusion manifest")
    if (
        report.get("ready_for_training") is not False
        or manifest.get("requires_independent_reaudit") is not True
        or manifest.get("authorizes_training") is not False
    ):
        raise DataAuditError(
            "semantic exclusion output must remain blocked pending independent reaudit"
        )
    root = manifest_file.parent
    marker = _json_object(root / "COMPLETE", label="semantic exclusion COMPLETE")
    if (
        marker.get("semantic_exclusion_kind") != COMPLETE_KIND
        or marker.get("requires_independent_reaudit") is not True
        or marker.get("ready_for_training") is not False
        or marker.get("authorizes_training") is not False
    ):
        raise DataAuditError("semantic exclusion COMPLETE contract is invalid")
    raw_attestation = marker.get("semantic_exclusion_attestation")
    raw_ledger = marker.get("semantic_exclusion_ledger")
    raw_sidecars = marker.get("semantic_exclusion_sidecars")
    if (
        not isinstance(raw_attestation, Mapping)
        or not isinstance(raw_ledger, Mapping)
        or not isinstance(raw_sidecars, Mapping)
    ):
        raise DataAuditError("semantic exclusion COMPLETE identities are incomplete")
    attestation_path = phase_helpers._owned_regular_file(
        root,
        raw_attestation.get("path"),
        "semantic exclusion attestation",
    )
    ledger_path = phase_helpers._owned_regular_file(
        root,
        raw_ledger.get("path"),
        "semantic exclusion ledger",
    )
    for identity, path, label in (
        (raw_attestation, attestation_path, "attestation"),
        (raw_ledger, ledger_path, "ledger"),
    ):
        if identity.get("size") != path.stat().st_size or identity.get("sha256") != sha256_file(
            path
        ):
            raise DataAuditError(f"semantic exclusion {label} identity mismatch")
    _verify_output_sidecars(
        root=root,
        manifest=manifest,
        identities=raw_sidecars,
    )
    attestation = _json_object(
        attestation_path,
        label="semantic exclusion attestation",
    )
    fingerprint = attestation.get("attestation_fingerprint")
    unsigned = dict(attestation)
    unsigned.pop("attestation_fingerprint", None)
    if (
        attestation.get("schema_version") != SCHEMA_VERSION
        or attestation.get("kind") != ATTESTATION_KIND
        or attestation.get("algorithm") != ALGORITHM
        or attestation.get("passed") is not True
        or attestation.get("validation_byte_preserved") is not True
        or attestation.get("requires_independent_reaudit") is not True
        or attestation.get("ready_for_training") is not False
        or attestation.get("authorizes_training") is not False
        or fingerprint != _canonical_sha256(unsigned)
        or raw_attestation.get("attestation_fingerprint") != fingerprint
    ):
        raise DataAuditError("semantic exclusion attestation is invalid")
    source_identity = attestation.get("source_identity")
    if (
        not isinstance(source_identity, Mapping)
        or dict(source_identity) != _source_identity_snapshot()
        or source_identity.get("normalizer_source_sha256") != audit_module.AUDIT_SOURCE_SHA256
    ):
        raise DataAuditError("semantic exclusion source identity is stale")
    scanner = _load_scanner(expected_sha256=str(source_identity["semantic_scanner_source_sha256"]))
    if scanner.sha256 != source_identity["semantic_scanner_source_sha256"]:
        raise DataAuditError("semantic exclusion scanner identity differs")
    inputs = attestation.get("inputs")
    expected = attestation.get("expected")
    if not isinstance(inputs, Mapping) or not isinstance(expected, Mapping):
        raise DataAuditError("semantic exclusion attestation inputs are incomplete")
    parent = _authenticate_input(
        parent_manifest=Path(str(inputs.get("parent_manifest_path"))),
        base_audit=Path(str(inputs.get("base_audit_path"))),
        expected_parent_manifest_sha256=str(expected.get("parent_manifest_sha256")),
        expected_base_audit_sha256=str(expected.get("base_audit_sha256")),
    )
    if parent.identity != inputs.get("parent_identity"):
        raise DataAuditError("semantic exclusion parent input identity changed")
    phase_name = expected.get("phase")
    source_id = expected.get("source_id")
    if phase_name not in PHASES or not isinstance(source_id, str):
        raise DataAuditError("semantic exclusion expected phase/source is invalid")
    input_ledger = _load_canonical_ledger(
        Path(str(inputs.get("canonical_ledger_path"))),
        expected_sha256=str(expected.get("canonical_ledger_sha256")),
        expected_phase=str(phase_name),
        source_id=source_id,
        expected_phase_documents=int(expected.get("excluded_documents", -1)),
        expected_phase_tokens=int(expected.get("excluded_tokens", -1)),
    )
    if input_ledger.identity != inputs.get("canonical_ledger_identity"):
        raise DataAuditError("semantic exclusion canonical ledger identity changed")
    expected_output_payload = b"".join(
        _canonical_bytes(row) + b"\n" for row in input_ledger.phase_rows
    )
    if ledger_path.read_bytes() != expected_output_payload:
        raise DataAuditError("semantic exclusion output ledger differs from canonical phase ledger")
    output = attestation.get("output")
    if (
        not isinstance(output, Mapping)
        or output.get("manifest") != manifest_file.name
        or output.get("manifest_sha256") != report["manifest_sha256"]
        or output.get("corpus_fingerprint") != report["corpus_fingerprint"]
        or output.get("ledger") != raw_ledger
        or output.get("sidecars") != raw_sidecars
    ):
        raise DataAuditError("semantic exclusion attestation output identity differs")
    parent_validation = parent.value.get("validation_files")
    if manifest.get("validation_files") != parent_validation:
        raise DataAuditError("semantic exclusion validation inventory is not byte-identical")
    projection = manifest.get("materialization_audit", {}).get("chinese_semantic_exclusion")
    if projection != attestation.get("projection"):
        raise DataAuditError("semantic exclusion manifest projection differs")
    expected_audits, expected_audit_projection = _project_parent_audits(
        parent.value.get("audits"),
        parent.audit.get("gates"),
    )
    if (
        manifest.get("audits") != expected_audits
        or not isinstance(projection, Mapping)
        or projection.get("audit_projection") != expected_audit_projection
    ):
        raise DataAuditError("semantic exclusion parent audit projection differs")
    return attestation


def materialize_semantic_exclusions(
    *,
    parent_manifest: str | Path,
    base_audit: str | Path,
    canonical_ledger: str | Path,
    phase: str,
    source_id: str,
    expected_parent_manifest_sha256: str,
    expected_base_audit_sha256: str,
    expected_ledger_sha256: str,
    expected_exclusion_count: int,
    expected_excluded_tokens: int,
    expected_scanner_sha256: str,
    expected_normalizer_sha256: str,
    required_source_tokens: Mapping[str, int],
    required_aggregate_tokens: int,
    formal_training_tokens: int,
    global_batch_tokens: int,
    output_root: str | Path,
) -> Path:
    """Create one immutable, non-training-ready semantic exclusion corpus."""

    if phase not in PHASES:
        raise DataAuditError(f"phase must be one of {sorted(PHASES)}")
    if not isinstance(source_id, str) or not source_id:
        raise DataAuditError("source_id must be non-empty")
    source_start = _source_identity_snapshot()
    if source_start["normalizer_source_sha256"] != _require_sha256(
        expected_normalizer_sha256,
        label="expected normalizer SHA256",
    ):
        raise DataAuditError("normalizer external pin mismatch")
    if source_start["normalizer_source_sha256"] != audit_module.AUDIT_SOURCE_SHA256:
        raise DataAuditError("loaded normalizer does not match its source bytes")
    scanner = _load_scanner(expected_sha256=expected_scanner_sha256)
    if scanner.sha256 != source_start["semantic_scanner_source_sha256"]:
        raise DataAuditError("semantic scanner snapshot differs from loaded module")
    parent = _authenticate_input(
        parent_manifest=Path(parent_manifest),
        base_audit=Path(base_audit),
        expected_parent_manifest_sha256=expected_parent_manifest_sha256,
        expected_base_audit_sha256=expected_base_audit_sha256,
    )
    ledger = _load_canonical_ledger(
        Path(canonical_ledger),
        expected_sha256=expected_ledger_sha256,
        expected_phase=phase,
        source_id=source_id,
        expected_phase_documents=expected_exclusion_count,
        expected_phase_tokens=expected_excluded_tokens,
    )
    output = _output_path(output_root, phase=parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    _fsync_directory(output.parent)
    work = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.semantic-exclusion-",
            dir=output.parent,
        )
    )
    try:
        projection = _project_phase(
            phase=parent,
            phase_name=phase,
            source_id=source_id,
            scanner=scanner,
            work=work,
        )
        _validate_parent_accounting(parent, projection)
        _compare_rebuilt_ledger(
            projection.rebuilt_ledger_rows,
            ledger.phase_rows,
        )
        if (
            len(projection.rebuilt_ledger_rows) != expected_exclusion_count
            or sum(int(row["token_count"]) for row in projection.rebuilt_ledger_rows)
            != expected_excluded_tokens
            or projection.excluded_rows.get(source_id, 0) != expected_exclusion_count
            or projection.excluded_tokens.get(source_id, 0) != expected_excluded_tokens
        ):
            raise DataAuditError("semantic exclusion scanner/accounting pins differ")
        capacity = _capacity_report(
            projection=projection,
            required_source_tokens=required_source_tokens,
            required_aggregate_tokens=required_aggregate_tokens,
            formal_training_tokens=formal_training_tokens,
            global_batch_tokens=global_batch_tokens,
        )
        inventories = {
            "train": projection.train_files,
            "validation": projection.validation_files,
            "attribution": projection.attribution_files,
        }
        for inventory in inventories.values():
            inventory.sort(key=lambda row: str(row["path"]))
        file_lists = _write_file_lists(work, inventories)
        output_ledger_path = work / OUTPUT_LEDGER_NAME
        _write_canonical_rows(
            output_ledger_path,
            projection.rebuilt_ledger_rows,
        )
        output_ledger_identity = _file_identity(
            output_ledger_path,
            relative=OUTPUT_LEDGER_NAME,
        )
        output_ledger_identity.update(
            {
                "documents": expected_exclusion_count,
                "tokens": expected_excluded_tokens,
            }
        )
        owners = phase_helpers._source_owners(parent.value)
        source_map = _source_map(
            parent=parent.value,
            projection=projection,
            owners=owners,
        )
        profile = f"{parent.value.get('profile')}-chinese-semantic-excluded"
        source_mix = _source_mix(
            parent=parent.value,
            profile=profile,
            projection=projection,
        )
        sources = _rebuilt_sources(
            parent=parent.value,
            source_map=source_map,
            projection=projection,
        )
        expected_contract = {
            "phase": phase,
            "source_id": source_id,
            "parent_manifest_sha256": expected_parent_manifest_sha256,
            "base_audit_sha256": expected_base_audit_sha256,
            "canonical_ledger_sha256": expected_ledger_sha256,
            "excluded_documents": expected_exclusion_count,
            "excluded_tokens": expected_excluded_tokens,
            "semantic_scanner_source_sha256": expected_scanner_sha256,
            "normalizer_source_sha256": expected_normalizer_sha256,
            "required_source_tokens": dict(sorted(required_source_tokens.items())),
            "required_aggregate_tokens": required_aggregate_tokens,
            "formal_training_tokens": formal_training_tokens,
            "global_batch_tokens": global_batch_tokens,
        }
        audits, audit_projection = _project_parent_audits(
            parent.value.get("audits"),
            parent.audit.get("gates"),
        )
        projection_identity = {
            "schema_version": SCHEMA_VERSION,
            "algorithm": ALGORITHM,
            "source_identity": source_start,
            "parent_identity": parent.identity,
            "canonical_ledger": ledger.identity,
            "expected": expected_contract,
            "output_ledger": output_ledger_identity,
            "excluded_train_documents": expected_exclusion_count,
            "excluded_train_tokens": expected_excluded_tokens,
            "validation_byte_preserved": True,
            "audit_projection": audit_projection,
            "requires_independent_reaudit": True,
            "authorizes_training": False,
        }
        parent_format = parent.value.get("format_audit")
        parent_license = parent.value.get("license_audit")
        if not isinstance(parent_format, Mapping) or not isinstance(
            parent_license,
            Mapping,
        ):
            raise DataAuditError("parent format/license audit is invalid")
        format_audit = deepcopy(dict(parent_format))
        format_audit.update(
            {
                "complete": True,
                "filtered_outputs": source_map["roles"],
                "chinese_semantic_exclusion": projection_identity,
            }
        )
        license_audit = deepcopy(dict(parent_license))
        license_audit.update(
            {
                "complete": True,
                "parent_attribution_inventory": parent_license.get("attribution_inventory"),
                "attribution_inventory": file_lists["attribution"],
                "chinese_semantic_exclusion": projection_identity,
            }
        )
        sidecars = {
            "source_map": _write_json_sidecar(
                work / SOURCE_MAP_SIDECAR,
                source_map,
            ),
            "license_audit": _write_json_sidecar(
                work / LICENSE_AUDIT_SIDECAR,
                license_audit,
            ),
        }
        materialization_audit = {
            "complete": True,
            "network_policy": "offline-authenticated-semantic-exclusion",
            "method": ALGORITHM,
            "chinese_semantic_exclusion": projection_identity,
            "sources": [
                {
                    "source_id": item["source_id"],
                    "retained_train_rows": projection.retained_rows["train"].get(
                        item["source_id"], 0
                    ),
                    "retained_train_tokens": projection.retained_tokens["train"].get(
                        item["source_id"], 0
                    ),
                    "excluded_train_rows": projection.excluded_rows.get(
                        item["source_id"],
                        0,
                    ),
                    "excluded_train_tokens": projection.excluded_tokens.get(
                        item["source_id"],
                        0,
                    ),
                }
                for item in sources
            ],
        }
        identity = {
            "recipe_id": parent.value.get("recipe_id"),
            "recipe_sha256": parent.value.get("recipe_sha256"),
            "resolved_source_lock_sha256": parent.value.get("resolved_source_lock_sha256"),
            "tokenizer_manifest_sha256": parent.value.get("tokenizer_manifest_sha256"),
            "extractor_source_sha256": source_start["materializer_source_sha256"],
            "profile": profile,
            "sources": sources,
            "train_files": inventories["train"],
            "validation_files": inventories["validation"],
            "attribution_files": inventories["attribution"],
            "file_lists": file_lists,
            "source_map": source_map,
            "source_mix": source_mix,
            "format_audit": format_audit,
            "license_audit": license_audit,
            "materialization_audit": materialization_audit,
        }
        corpus_fingerprint = _manifest_fingerprint(identity)
        manifest_value = {
            "schema_version": SCHEMA_VERSION,
            "kind": "twen_extracted_base_jsonl_corpus",
            **identity,
            "corpus_fingerprint": corpus_fingerprint,
            "actual_train_tokens": sum(projection.retained_tokens["train"].values()),
            "actual_validation_tokens": sum(projection.retained_tokens["validation"].values()),
            "actual_train_documents": sum(projection.retained_rows["train"].values()),
            "actual_validation_documents": sum(projection.retained_rows["validation"].values()),
            # Parent audit rejection accounting remains historical and is not
            # forged into the independent semantic-exclusion ledger.
            "rejected_train_documents": parent.value.get(
                "rejected_train_documents",
                0,
            ),
            "rejected_validation_documents": parent.value.get(
                "rejected_validation_documents",
                0,
            ),
            "rejection_reasons": deepcopy(parent.value.get("rejection_reasons", {})),
            "semantic_excluded_train_documents": expected_exclusion_count,
            "semantic_excluded_train_tokens": expected_excluded_tokens,
            "network_policy": "offline-authenticated-semantic-exclusion",
            "audits": audits,
            "ready_for_data_prepare": True,
            "ready_for_training": False,
            "requires_independent_reaudit": True,
            "authorizes_training": False,
        }
        manifest_path = work / "corpus-manifest.json"
        atomic_write_json(manifest_path, manifest_value)
        attestation_value = {
            "schema_version": SCHEMA_VERSION,
            "kind": ATTESTATION_KIND,
            "algorithm": ALGORITHM,
            "source_identity": source_start,
            "inputs": {
                "parent_manifest_path": str(parent.manifest_path),
                "base_audit_path": str(parent.audit_path),
                "parent_identity": parent.identity,
                "canonical_ledger_path": str(ledger.path),
                "canonical_ledger_identity": ledger.identity,
            },
            "expected": expected_contract,
            "projection": projection_identity,
            "capacity": capacity,
            "metrics": {
                "excluded_train_documents": expected_exclusion_count,
                "excluded_train_tokens": expected_excluded_tokens,
                "retained_train_documents": manifest_value["actual_train_documents"],
                "retained_train_tokens": manifest_value["actual_train_tokens"],
                "validation_documents": manifest_value["actual_validation_documents"],
                "validation_tokens": manifest_value["actual_validation_tokens"],
            },
            "output": {
                "manifest": manifest_path.name,
                "manifest_sha256": sha256_file(manifest_path),
                "corpus_fingerprint": corpus_fingerprint,
                "ledger": output_ledger_identity,
                "sidecars": sidecars,
            },
            "validation_byte_preserved": True,
            "requires_independent_reaudit": True,
            "ready_for_training": False,
            "authorizes_training": False,
            "passed": True,
        }
        attestation_value["attestation_fingerprint"] = _canonical_sha256(attestation_value)
        attestation_path = work / ATTESTATION_NAME
        atomic_write_json(attestation_path, attestation_value)
        attestation_identity = _file_identity(
            attestation_path,
            relative=ATTESTATION_NAME,
        )
        attestation_identity["attestation_fingerprint"] = attestation_value[
            "attestation_fingerprint"
        ]
        atomic_write_json(
            work / "COMPLETE",
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "twen_extracted_base_jsonl_complete",
                "corpus_fingerprint": corpus_fingerprint,
                "manifest": manifest_path.name,
                "manifest_sha256": sha256_file(manifest_path),
                "file_lists": file_lists,
                "ready_for_training": False,
                "requires_independent_reaudit": True,
                "authorizes_training": False,
                "semantic_exclusion_kind": COMPLETE_KIND,
                "semantic_exclusion_attestation": attestation_identity,
                "semantic_exclusion_ledger": output_ledger_identity,
                "semantic_exclusion_sidecars": sidecars,
            },
        )
        source_end = _source_identity_snapshot()
        scanner_end = _load_scanner(expected_sha256=expected_scanner_sha256)
        parent_end = _authenticate_input(
            parent_manifest=Path(parent_manifest),
            base_audit=Path(base_audit),
            expected_parent_manifest_sha256=expected_parent_manifest_sha256,
            expected_base_audit_sha256=expected_base_audit_sha256,
        )
        ledger_end = _load_canonical_ledger(
            Path(canonical_ledger),
            expected_sha256=expected_ledger_sha256,
            expected_phase=phase,
            source_id=source_id,
            expected_phase_documents=expected_exclusion_count,
            expected_phase_tokens=expected_excluded_tokens,
        )
        if (
            source_end != source_start
            or scanner_end.sha256 != scanner.sha256
            or parent_end.identity != parent.identity
            or ledger_end.identity != ledger.identity
        ):
            raise DataAuditError(
                "semantic exclusion source or input identity changed during materialization"
            )
        validate_semantic_exclusion_output(manifest_path)
        _fsync_tree(work)
        _rename_directory_noreplace(work, output)
        _fsync_directory(output.parent)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return output / "corpus-manifest.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", required=True, type=Path)
    parser.add_argument("--base-audit", required=True, type=Path)
    parser.add_argument("--canonical-ledger", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--expected-parent-manifest-sha256", required=True)
    parser.add_argument("--expected-base-audit-sha256", required=True)
    parser.add_argument("--expected-ledger-sha256", required=True)
    parser.add_argument("--expected-exclusion-count", required=True, type=int)
    parser.add_argument("--expected-excluded-tokens", required=True, type=int)
    parser.add_argument("--expected-scanner-sha256", required=True)
    parser.add_argument("--expected-normalizer-sha256", required=True)
    parser.add_argument(
        "--required-source-token",
        action="append",
        default=[],
        metavar="SOURCE_ID=TOKENS",
    )
    parser.add_argument("--required-aggregate-tokens", required=True, type=int)
    parser.add_argument("--formal-training-tokens", required=True, type=int)
    parser.add_argument("--global-batch-tokens", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        required_source_tokens = _parse_required_source_tokens(args.required_source_token)
        manifest = materialize_semantic_exclusions(
            parent_manifest=args.parent_manifest,
            base_audit=args.base_audit,
            canonical_ledger=args.canonical_ledger,
            phase=args.phase,
            source_id=args.source_id,
            expected_parent_manifest_sha256=args.expected_parent_manifest_sha256,
            expected_base_audit_sha256=args.expected_base_audit_sha256,
            expected_ledger_sha256=args.expected_ledger_sha256,
            expected_exclusion_count=args.expected_exclusion_count,
            expected_excluded_tokens=args.expected_excluded_tokens,
            expected_scanner_sha256=args.expected_scanner_sha256,
            expected_normalizer_sha256=args.expected_normalizer_sha256,
            required_source_tokens=required_source_tokens,
            required_aggregate_tokens=args.required_aggregate_tokens,
            formal_training_tokens=args.formal_training_tokens,
            global_batch_tokens=args.global_batch_tokens,
            output_root=args.output,
        )
        attestation = validate_semantic_exclusion_output(manifest)
    except (DataAuditError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "attestation_fingerprint": attestation["attestation_fingerprint"],
                "excluded_documents": attestation["metrics"]["excluded_train_documents"],
                "excluded_tokens": attestation["metrics"]["excluded_train_tokens"],
                "ready_for_training": False,
                "requires_independent_reaudit": True,
                "authorizes_training": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
