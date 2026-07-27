"""Authenticated train-phase set difference for extracted Base corpora.

The materializer removes every cooldown train document whose source-scoped
stable split key also occurs in the primary train corpus.  Validation bytes are
copied unchanged.  Both inputs must be complete extracted corpora with passing
``audit-base`` attestations that bind the exact manifest bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict, deque
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ..utils import atomic_write_json, atomic_write_text, sha256_file
from .audits import DataAuditError, validate_base_audit_attestation
from .sources import validate_extracted_base_corpus

PHASE_EXCLUSION_SCHEMA_VERSION = 1
PHASE_EXCLUSION_ALGORITHM = "source-scoped-authenticated-stable-id-set-difference-v1"
PHASE_EXCLUSION_ATTESTATION_KIND = "twen_phase_exclusion_attestation"
PHASE_EXCLUSION_LEDGER_KIND = "twen_phase_exclusion_ledger"
PHASE_EXCLUSION_COMPLETE_KIND = "twen_phase_excluded_corpus_complete"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_FIELDS = (
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


@dataclass(frozen=True, slots=True)
class _PhaseInput:
    manifest_path: Path
    manifest_sha256: str
    corpus_fingerprint: str
    complete_sha256: str
    value: Mapping[str, Any]
    audit_path: Path
    audit_sha256: str
    audit_fingerprint: str
    audit_complete_sha256: str
    audit: Mapping[str, Any]

    @property
    def identity(self) -> dict[str, object]:
        return {
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "corpus_fingerprint": self.corpus_fingerprint,
            "complete_sha256": self.complete_sha256,
            "audit_attestation_path": str(self.audit_path),
            "audit_attestation_sha256": self.audit_sha256,
            "audit_attestation_fingerprint": self.audit_fingerprint,
            "audit_complete_sha256": self.audit_complete_sha256,
        }


@dataclass(frozen=True, slots=True)
class _Attribution:
    relative: str
    line_number: int
    source_id: str
    stable_id: str
    text_sha256: str
    token_count: int
    split: str

    @property
    def stable_key(self) -> tuple[str, str]:
        return self.source_id, self.stable_id

    @property
    def document_key(self) -> tuple[str, str]:
        return self.source_id, self.text_sha256


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _current_source_sha256() -> str:
    return sha256_file(Path(__file__))


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataAuditError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise DataAuditError(f"{label} must be an object: {path}")
    return value


def _safe_relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DataAuditError(f"{label} must be a non-empty relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or relative.as_posix() == "." or ".." in relative.parts:
        raise DataAuditError(f"unsafe {label}: {value!r}")
    return relative.as_posix()


def _owned_regular_file(root: Path, relative: object, label: str) -> Path:
    relative_text = _safe_relative(relative, label)
    lexical = root / relative_text
    cursor = root
    for part in PurePosixPath(relative_text).parts:
        cursor /= part
        if cursor.is_symlink():
            raise DataAuditError(f"{label} must not be a symlink: {cursor}")
    resolved = lexical.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise DataAuditError(f"{label} escapes its authenticated root") from error
    if not resolved.is_file():
        raise DataAuditError(f"missing {label}: {resolved}")
    return resolved


def _file_identity(path: Path, *, relative: str) -> dict[str, object]:
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _verify_extracted_owned_files(
    manifest_path: Path,
    value: Mapping[str, Any],
) -> None:
    root = manifest_path.parent
    _owned_regular_file(root, "COMPLETE", "extracted COMPLETE")
    for role in ("train", "validation", "attribution"):
        inventory = value.get(f"{role}_files")
        if not isinstance(inventory, list):
            raise DataAuditError(f"extracted {role} inventory is invalid")
        for index, raw in enumerate(inventory):
            if not isinstance(raw, Mapping):
                raise DataAuditError(f"extracted {role}[{index}] is invalid")
            _owned_regular_file(
                root,
                raw.get("path"),
                f"extracted {role}[{index}].path",
            )
        _owned_regular_file(root, f"{role}-files.txt", f"{role} file-list")


def _verify_audit_owned_files(
    audit_path: Path,
    value: Mapping[str, Any],
) -> None:
    root = audit_path.parent
    _owned_regular_file(root, "COMPLETE", "audit COMPLETE")
    for field in ("findings", "rejection_ledger"):
        identity = value.get(field)
        if not isinstance(identity, Mapping):
            raise DataAuditError(f"audit {field} identity is invalid")
        _owned_regular_file(root, identity.get("path"), f"audit {field}")


def _bound_audit_role(
    audit: Mapping[str, Any],
    field: str,
    *,
    role: str,
    manifest_path: Path,
    manifest_sha256: str,
    corpus_fingerprint: str,
) -> None:
    identity = audit.get(field)
    if (
        not isinstance(identity, Mapping)
        or identity.get("role") != role
        or identity.get("manifest_sha256") != manifest_sha256
        or identity.get("corpus_fingerprint") != corpus_fingerprint
        or Path(str(identity.get("manifest_path"))).resolve() != manifest_path
    ):
        raise DataAuditError(f"audit {field} does not bind the exact extracted {role} manifest")


def _authenticate_phase_input(
    manifest_path: str | Path,
    audit_path: str | Path,
) -> _PhaseInput:
    raw_manifest_path = Path(manifest_path).expanduser()
    raw_audit_path = Path(audit_path).expanduser()
    if raw_manifest_path.is_symlink():
        raise DataAuditError("extracted manifest must not be a symlink")
    if raw_audit_path.is_symlink():
        raise DataAuditError("audit attestation must not be a symlink")
    manifest = raw_manifest_path.resolve()
    audit_file = raw_audit_path.resolve()

    try:
        report = validate_extracted_base_corpus(manifest, verify_hashes=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise DataAuditError(f"invalid extracted phase input: {error}") from error
    value = _json_object(manifest, "extracted manifest")
    _verify_extracted_owned_files(manifest, value)
    audit = validate_base_audit_attestation(audit_file)
    _verify_audit_owned_files(audit_file, audit)
    if audit.get("ready_for_training") is not True:
        raise DataAuditError(f"audit is not ready_for_training: {audit_file}")

    manifest_sha = str(report["manifest_sha256"])
    fingerprint = str(report["corpus_fingerprint"])
    _bound_audit_role(
        audit,
        "candidate",
        role="train",
        manifest_path=manifest,
        manifest_sha256=manifest_sha,
        corpus_fingerprint=fingerprint,
    )
    _bound_audit_role(
        audit,
        "frozen_validation",
        role="validation",
        manifest_path=manifest,
        manifest_sha256=manifest_sha,
        corpus_fingerprint=fingerprint,
    )
    complete = manifest.parent / "COMPLETE"
    audit_complete = audit_file.parent / "COMPLETE"
    return _PhaseInput(
        manifest_path=manifest,
        manifest_sha256=manifest_sha,
        corpus_fingerprint=fingerprint,
        complete_sha256=sha256_file(complete),
        value=value,
        audit_path=audit_file,
        audit_sha256=sha256_file(audit_file),
        audit_fingerprint=str(audit["attestation_fingerprint"]),
        audit_complete_sha256=sha256_file(audit_complete),
        audit=audit,
    )


def _attribution_inventory(phase: _PhaseInput) -> tuple[tuple[Path, str], ...]:
    raw_inventory = phase.value.get("attribution_files")
    if not isinstance(raw_inventory, list) or not raw_inventory:
        raise DataAuditError("phase exclusion requires attribution files")
    result: list[tuple[Path, str]] = []
    for index, raw in enumerate(raw_inventory):
        if not isinstance(raw, Mapping):
            raise DataAuditError(f"attribution_files[{index}] is invalid")
        relative = _safe_relative(
            raw.get("path"),
            f"attribution_files[{index}].path",
        )
        result.append(
            (
                _owned_regular_file(
                    phase.manifest_path.parent,
                    relative,
                    f"attribution_files[{index}]",
                ),
                relative,
            )
        )
    return tuple(result)


def _parse_attribution(
    raw: bytes,
    *,
    path: Path,
    relative: str,
    line_number: int,
) -> _Attribution:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DataAuditError(f"invalid attribution JSONL at {path}:{line_number}") from error
    if not isinstance(value, Mapping):
        raise DataAuditError(f"invalid attribution row at {path}:{line_number}")
    source_id = value.get("source_id")
    stable_id = value.get("stable_id")
    text_sha = value.get("text_sha256")
    token_count = value.get("token_count_with_eos")
    split = value.get("split")
    if (
        not isinstance(source_id, str)
        or not source_id
        or not isinstance(stable_id, str)
        or _SHA256.fullmatch(stable_id) is None
        or not isinstance(text_sha, str)
        or _SHA256.fullmatch(text_sha) is None
        or isinstance(token_count, bool)
        or not isinstance(token_count, int)
        or token_count <= 0
        or split not in {"train", "validation"}
    ):
        raise DataAuditError(f"invalid attribution identity at {path}:{line_number}")
    return _Attribution(
        relative=relative,
        line_number=line_number,
        source_id=source_id,
        stable_id=stable_id,
        text_sha256=text_sha,
        token_count=token_count,
        split=str(split),
    )


def _iter_attribution(phase: _PhaseInput) -> Iterator[_Attribution]:
    for path, relative in _attribution_inventory(phase):
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    raise DataAuditError(f"blank attribution JSONL row at {path}:{line_number}")
                yield _parse_attribution(
                    raw,
                    path=path,
                    relative=relative,
                    line_number=line_number,
                )


def _source_owners(
    value: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    source_map = value.get("source_map")
    roles = source_map.get("roles") if isinstance(source_map, Mapping) else None
    if not isinstance(roles, Mapping):
        raise DataAuditError("cooldown source_map is invalid")
    result: dict[str, dict[str, str]] = {"train": {}, "validation": {}}
    for role in result:
        entries = roles.get(role)
        if not isinstance(entries, list):
            raise DataAuditError(f"cooldown source_map.{role} is invalid")
        for index, raw in enumerate(entries):
            if not isinstance(raw, Mapping):
                raise DataAuditError(f"source_map.{role}[{index}] is invalid")
            relative = _safe_relative(
                raw.get("path"),
                f"source_map.{role}[{index}].path",
            )
            source_id = raw.get("source_id")
            if not isinstance(source_id, str) or not source_id or relative in result[role]:
                raise DataAuditError(f"source_map.{role}[{index}] owner is invalid or duplicate")
            result[role][relative] = source_id
    return result


def _parse_document_text(raw: bytes, path: Path, line_number: int) -> str:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DataAuditError(f"invalid corpus JSONL at {path}:{line_number}") from error
    text = value.get("text") if isinstance(value, Mapping) else None
    if not isinstance(text, str):
        raise DataAuditError(f"missing corpus text at {path}:{line_number}")
    return text


def _copy_and_match_documents(
    *,
    phase: _PhaseInput,
    work: Path,
    role: str,
    owners: Mapping[str, str],
    attribution: Mapping[tuple[str, str], deque[_Attribution]],
    decisions: Mapping[tuple[str, str], bool],
    excluded_locations: set[tuple[str, int]],
    ledger_handle: Any,
) -> tuple[list[dict[str, object]], dict[str, int], dict[str, int], int]:
    raw_inventory = phase.value.get(f"{role}_files")
    if not isinstance(raw_inventory, list):
        raise DataAuditError(f"cooldown {role} inventory is invalid")
    output_inventory: list[dict[str, object]] = []
    source_rows: dict[str, int] = defaultdict(int)
    source_tokens: dict[str, int] = defaultdict(int)
    excluded_documents = 0
    for index, raw_entry in enumerate(raw_inventory):
        if not isinstance(raw_entry, Mapping):
            raise DataAuditError(f"cooldown {role}_files[{index}] is invalid")
        relative = _safe_relative(
            raw_entry.get("path"),
            f"{role}_files[{index}].path",
        )
        source_id = owners.get(relative)
        if source_id is None:
            raise DataAuditError(f"source_map has no owner for {role} file: {relative}")
        source = _owned_regular_file(
            phase.manifest_path.parent,
            relative,
            f"cooldown {role} file",
        )
        output = work / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as source_handle, output.open("wb") as output_handle:
            for line_number, raw_line in enumerate(source_handle, start=1):
                if not raw_line.strip():
                    raise DataAuditError(f"blank corpus JSONL row at {source}:{line_number}")
                text = _parse_document_text(raw_line, source, line_number)
                text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                document_key = (source_id, text_sha)
                queue = attribution.get(document_key)
                if queue is None or not queue:
                    raise DataAuditError(
                        "cooldown attribution does not cover corpus document exactly: "
                        f"{relative}:{line_number}"
                    )
                record = queue.popleft()
                excluded = decisions[record.stable_key] if role == "train" else False
                if excluded:
                    excluded_documents += 1
                    excluded_locations.add((record.relative, record.line_number))
                    ledger_handle.write(
                        json.dumps(
                            {
                                "schema_version": PHASE_EXCLUSION_SCHEMA_VERSION,
                                "kind": PHASE_EXCLUSION_LEDGER_KIND,
                                "algorithm": PHASE_EXCLUSION_ALGORITHM,
                                "reason": "primary_train_stable_key_intersection",
                                "source_id": record.source_id,
                                "stable_id": record.stable_id,
                                "text_sha256": record.text_sha256,
                                "token_count_with_eos": record.token_count,
                                "cooldown_document": {
                                    "path": relative,
                                    "line": line_number,
                                },
                                "cooldown_attribution": {
                                    "path": record.relative,
                                    "line": record.line_number,
                                },
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    continue
                output_handle.write(raw_line)
                source_rows[source_id] += 1
                source_tokens[source_id] += record.token_count
        if role == "validation" and (
            output.stat().st_size != source.stat().st_size
            or sha256_file(output) != sha256_file(source)
        ):
            raise DataAuditError(f"validation was not byte-preserved: {relative}")
        output_inventory.append(_file_identity(output, relative=relative))
    return (
        output_inventory,
        dict(source_rows),
        dict(source_tokens),
        excluded_documents,
    )


def _write_filtered_attribution(
    phase: _PhaseInput,
    work: Path,
    excluded_locations: set[tuple[str, int]],
) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for source, relative in _attribution_inventory(phase):
        output = work / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as source_handle, output.open("wb") as output_handle:
            for line_number, raw in enumerate(source_handle, start=1):
                record = _parse_attribution(
                    raw,
                    path=source,
                    relative=relative,
                    line_number=line_number,
                )
                if record.split == "train" and (relative, line_number) in excluded_locations:
                    continue
                output_handle.write(raw)
        inventory.append(_file_identity(output, relative=relative))
    return inventory


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


def _rebuilt_sources(
    parent: Mapping[str, Any],
    *,
    source_map_roles: Mapping[str, list[dict[str, object]]],
    source_rows: Mapping[str, Mapping[str, int]],
    source_tokens: Mapping[str, Mapping[str, int]],
    excluded_tokens: Mapping[str, int],
    excluded_rows: Mapping[str, int],
) -> list[dict[str, object]]:
    raw_sources = parent.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise DataAuditError("cooldown source inventory is invalid")
    result: list[dict[str, object]] = []
    for raw_source in raw_sources:
        if not isinstance(raw_source, Mapping):
            raise DataAuditError("cooldown source entry is invalid")
        source_id = raw_source.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise DataAuditError("cooldown source_id is invalid")
        copied = deepcopy(dict(raw_source))
        copied["actual_train_tokens"] = source_tokens["train"].get(source_id, 0)
        copied["actual_validation_tokens"] = source_tokens["validation"].get(source_id, 0)
        copied["train_rows"] = source_rows["train"].get(source_id, 0)
        copied["validation_rows"] = source_rows["validation"].get(source_id, 0)
        outputs = [
            {
                "path": entry["path"],
                "size": entry["size"],
                "sha256": entry["sha256"],
            }
            for role in ("train", "validation")
            for entry in source_map_roles[role]
            if entry["source_id"] == source_id
        ]
        copied["chunks"] = [
            {
                "shard_id": "phase-excluded",
                "outputs": outputs,
                "statistics": {
                    "excluded_train_rows": excluded_rows.get(source_id, 0),
                    "excluded_train_tokens": excluded_tokens.get(source_id, 0),
                },
            }
        ]
        result.append(copied)
    return result


def _phase_input_snapshot(
    primary_manifest: str | Path,
    primary_audit: str | Path,
    cooldown_manifest: str | Path,
    cooldown_audit: str | Path,
) -> tuple[_PhaseInput, _PhaseInput]:
    primary = _authenticate_phase_input(primary_manifest, primary_audit)
    cooldown = _authenticate_phase_input(cooldown_manifest, cooldown_audit)
    if primary.manifest_sha256 == cooldown.manifest_sha256:
        raise DataAuditError("primary and cooldown manifests must be distinct")
    if primary.value.get("tokenizer_manifest_sha256") != cooldown.value.get(
        "tokenizer_manifest_sha256"
    ):
        raise DataAuditError("primary and cooldown tokenizer identities differ")
    return primary, cooldown


def _validate_phase_exclusion_output(manifest_path: Path) -> Mapping[str, Any]:
    report = validate_extracted_base_corpus(manifest_path, verify_hashes=True)
    if report.get("ready_for_training") is not False:
        raise DataAuditError("phase-excluded corpus requires an independent audit before training")
    root = manifest_path.parent
    manifest_value = _json_object(manifest_path, "phase-excluded manifest")
    marker = _json_object(root / "COMPLETE", "phase-exclusion COMPLETE")
    if marker.get("phase_exclusion_kind") != PHASE_EXCLUSION_COMPLETE_KIND:
        raise DataAuditError("phase-exclusion COMPLETE kind is invalid")
    raw_attestation = marker.get("phase_exclusion_attestation")
    raw_ledger = marker.get("phase_exclusion_ledger")
    raw_sidecars = marker.get("phase_exclusion_sidecars")
    if (
        not isinstance(raw_attestation, Mapping)
        or not isinstance(raw_ledger, Mapping)
        or not isinstance(raw_sidecars, Mapping)
    ):
        raise DataAuditError("phase-exclusion COMPLETE identities are missing")
    attestation_relative = _safe_relative(
        raw_attestation.get("path"),
        "phase_exclusion_attestation.path",
    )
    ledger_relative = _safe_relative(
        raw_ledger.get("path"),
        "phase_exclusion_ledger.path",
    )
    attestation_path = _owned_regular_file(
        root,
        attestation_relative,
        "phase-exclusion attestation",
    )
    ledger_path = _owned_regular_file(
        root,
        ledger_relative,
        "phase-exclusion ledger",
    )
    for identity, path, label in (
        (raw_attestation, attestation_path, "attestation"),
        (raw_ledger, ledger_path, "ledger"),
    ):
        if identity.get("size") != path.stat().st_size or identity.get("sha256") != sha256_file(
            path
        ):
            raise DataAuditError(f"phase-exclusion {label} identity mismatch")
    expected_sidecars = {
        "source_map": ("source-map.json", manifest_value.get("source_map")),
        "license_audit": ("license-audit.json", manifest_value.get("license_audit")),
    }
    if set(raw_sidecars) != {
        "source_map",
        "license_audit",
        "attribution_file_list",
    } or raw_sidecars.get("attribution_file_list") != manifest_value.get("file_lists", {}).get(
        "attribution"
    ):
        raise DataAuditError("phase-exclusion sidecar inventory is invalid")
    for name, (expected_path, expected_value) in expected_sidecars.items():
        identity = raw_sidecars.get(name)
        if not isinstance(identity, Mapping):
            raise DataAuditError(f"phase-exclusion {name} sidecar identity is missing")
        relative = _safe_relative(
            identity.get("path"),
            f"phase_exclusion_sidecars.{name}.path",
        )
        if relative != expected_path:
            raise DataAuditError(f"phase-exclusion {name} sidecar path is invalid")
        sidecar = _owned_regular_file(root, relative, f"phase-exclusion {name} sidecar")
        if (
            sidecar.stat().st_size != identity.get("size")
            or sha256_file(sidecar) != identity.get("sha256")
            or _json_object(sidecar, f"phase-exclusion {name} sidecar") != expected_value
        ):
            raise DataAuditError(f"phase-exclusion {name} sidecar identity mismatch")
    attestation = _json_object(attestation_path, "phase-exclusion attestation")
    fingerprint = attestation.get("attestation_fingerprint")
    unsigned = dict(attestation)
    unsigned.pop("attestation_fingerprint", None)
    if (
        attestation.get("schema_version") != PHASE_EXCLUSION_SCHEMA_VERSION
        or attestation.get("kind") != PHASE_EXCLUSION_ATTESTATION_KIND
        or attestation.get("algorithm") != PHASE_EXCLUSION_ALGORITHM
        or attestation.get("passed") is not True
        or attestation.get("requires_independent_audit") is not True
        or attestation.get("ready_for_training") is not False
        or attestation.get("source_sha256") != _current_source_sha256()
        or fingerprint != _canonical_sha256(unsigned)
        or raw_attestation.get("attestation_fingerprint") != fingerprint
    ):
        raise DataAuditError("phase-exclusion attestation is invalid")
    output = attestation.get("output")
    if (
        not isinstance(output, Mapping)
        or output.get("manifest") != manifest_path.name
        or output.get("manifest_sha256") != report["manifest_sha256"]
        or output.get("corpus_fingerprint") != report["corpus_fingerprint"]
        or output.get("ledger") != raw_ledger
        or output.get("sidecars") != raw_sidecars
    ):
        raise DataAuditError("phase-exclusion attestation output identity mismatch")
    return attestation


def validate_phase_exclusion_output(
    manifest_path: str | Path,
) -> Mapping[str, Any]:
    """Authenticate a published phase-excluded extracted corpus."""

    return _validate_phase_exclusion_output(Path(manifest_path).resolve())


def materialize_phase_excluded_cooldown(
    *,
    primary_manifest: str | Path,
    primary_audit: str | Path,
    cooldown_manifest: str | Path,
    cooldown_audit: str | Path,
    output_root: str | Path,
) -> Path:
    """Atomically publish ``cooldown train - primary train stable keys``."""

    source_sha_before = _current_source_sha256()
    primary, cooldown = _phase_input_snapshot(
        primary_manifest,
        primary_audit,
        cooldown_manifest,
        cooldown_audit,
    )
    input_identities = {
        "primary": primary.identity,
        "cooldown": cooldown.identity,
    }

    primary_stable_keys: set[tuple[str, str]] = set()
    primary_train_rows = 0
    primary_duplicate_rows = 0
    for record in _iter_attribution(primary):
        if record.split != "train":
            continue
        primary_train_rows += 1
        if record.stable_key in primary_stable_keys:
            primary_duplicate_rows += 1
        primary_stable_keys.add(record.stable_key)

    cooldown_records: dict[tuple[str, str], deque[_Attribution]] = defaultdict(deque)
    cooldown_stable_keys: set[tuple[str, str]] = set()
    cooldown_train_rows = 0
    cooldown_duplicate_rows = 0
    validation_rows = 0
    for record in _iter_attribution(cooldown):
        cooldown_records[record.document_key].append(record)
        if record.split == "train":
            cooldown_train_rows += 1
            if record.stable_key in cooldown_stable_keys:
                cooldown_duplicate_rows += 1
            cooldown_stable_keys.add(record.stable_key)
        else:
            validation_rows += 1
    intersection = primary_stable_keys & cooldown_stable_keys
    decisions = {key: key in intersection for key in cooldown_stable_keys}

    # Identical text hashes have no file/row locator in the attribution schema.
    # They are safe to consume as a multiset only when every occurrence has the
    # same decision; otherwise no implementation can prove which byte row owns
    # the excluded stable key.
    for document_key, records in cooldown_records.items():
        train_decisions = {
            decisions[record.stable_key] for record in records if record.split == "train"
        }
        splits = {record.split for record in records}
        if len(splits) != 1 or len(train_decisions) > 1:
            raise DataAuditError(
                "ambiguous cooldown attribution for duplicate "
                f"(source_id,text_sha256): {document_key}"
            )

    requested_root = Path(output_root).expanduser()
    if requested_root.is_symlink():
        raise DataAuditError("phase-exclusion output must not be a symlink")
    root = requested_root.absolute()
    if root.exists():
        raise DataAuditError(f"phase-exclusion output already exists: {root}")
    for phase in (primary, cooldown):
        phase_root = phase.manifest_path.parent
        try:
            root.resolve(strict=False).relative_to(phase_root)
        except ValueError:
            pass
        else:
            raise DataAuditError("phase-exclusion output overlaps an input corpus")
    root.parent.mkdir(parents=True, exist_ok=True)
    work = Path(
        tempfile.mkdtemp(
            prefix=f".{root.name}.phase-exclusion-",
            dir=root.parent,
        )
    )
    try:
        owners = _source_owners(cooldown.value)
        excluded_locations: set[tuple[str, int]] = set()
        ledger_path = work / "phase-exclusion-ledger.jsonl"
        with ledger_path.open("w", encoding="utf-8") as ledger:
            (
                train_files,
                train_source_rows,
                train_source_tokens,
                excluded_documents,
            ) = _copy_and_match_documents(
                phase=cooldown,
                work=work,
                role="train",
                owners=owners["train"],
                attribution=cooldown_records,
                decisions=decisions,
                excluded_locations=excluded_locations,
                ledger_handle=ledger,
            )
            (
                validation_files,
                validation_source_rows,
                validation_source_tokens,
                validation_excluded,
            ) = _copy_and_match_documents(
                phase=cooldown,
                work=work,
                role="validation",
                owners=owners["validation"],
                attribution=cooldown_records,
                decisions=decisions,
                excluded_locations=excluded_locations,
                ledger_handle=ledger,
            )
        if validation_excluded:
            raise DataAuditError("phase exclusion attempted to remove validation rows")
        unmatched = sum(len(records) for records in cooldown_records.values())
        if unmatched:
            raise DataAuditError(
                f"cooldown attribution has no exact corpus document for {unmatched} rows"
            )
        if excluded_documents != len(excluded_locations):
            raise DataAuditError("excluded document/attribution coverage is not one-to-one")

        attribution_files = _write_filtered_attribution(
            cooldown,
            work,
            excluded_locations,
        )
        inventories = {
            "train": train_files,
            "validation": validation_files,
            "attribution": attribution_files,
        }
        for inventory in inventories.values():
            inventory.sort(key=lambda item: str(item["path"]))
        file_lists = _write_file_lists(work, inventories)
        ledger_identity = _file_identity(
            ledger_path,
            relative=ledger_path.name,
        )

        excluded_source_rows: dict[str, int] = defaultdict(int)
        excluded_source_tokens: dict[str, int] = defaultdict(int)
        for record in _iter_attribution(cooldown):
            if (record.relative, record.line_number) in excluded_locations:
                excluded_source_rows[record.source_id] += 1
                excluded_source_tokens[record.source_id] += record.token_count

        source_map_roles: dict[str, list[dict[str, object]]] = {
            "train": [
                {"source_id": owners["train"][str(entry["path"])], **entry} for entry in train_files
            ],
            "validation": [
                {"source_id": owners["validation"][str(entry["path"])], **entry}
                for entry in validation_files
            ],
        }
        parent_source_map = cooldown.value.get("source_map")
        if not isinstance(parent_source_map, Mapping):
            raise DataAuditError("cooldown source_map is invalid")
        source_map_unsigned = {
            "schema_version": parent_source_map.get("schema_version"),
            "algorithm": parent_source_map.get("algorithm"),
            "roles": source_map_roles,
        }
        source_map = {
            **source_map_unsigned,
            "fingerprint": _canonical_sha256(source_map_unsigned),
        }

        profile = f"{cooldown.value.get('profile')}-phase-excluded"
        parent_source_mix = cooldown.value.get("source_mix")
        if not isinstance(parent_source_mix, Mapping) or not isinstance(
            parent_source_mix.get("sources"), list
        ):
            raise DataAuditError("cooldown source_mix is invalid")
        source_mix_sources: list[dict[str, object]] = []
        for raw in parent_source_mix["sources"]:
            if not isinstance(raw, Mapping):
                raise DataAuditError("cooldown source_mix entry is invalid")
            copied = deepcopy(dict(raw))
            source_id = str(copied.get("source_id"))
            copied["actual_train_tokens"] = train_source_tokens.get(source_id, 0)
            source_mix_sources.append(copied)
        source_mix_unsigned = {
            "schema_version": parent_source_mix.get("schema_version"),
            "algorithm": parent_source_mix.get("algorithm"),
            "unit": parent_source_mix.get("unit"),
            "basis_points_total": parent_source_mix.get("basis_points_total"),
            "profile": profile,
            "sources": source_mix_sources,
        }
        source_mix = {
            **source_mix_unsigned,
            "fingerprint": _canonical_sha256(source_mix_unsigned),
        }

        projection = {
            "schema_version": PHASE_EXCLUSION_SCHEMA_VERSION,
            "algorithm": PHASE_EXCLUSION_ALGORITHM,
            "source_sha256": source_sha_before,
            "primary": primary.identity,
            "cooldown": cooldown.identity,
            "ledger": ledger_identity,
            "intersecting_stable_keys": len(intersection),
            "excluded_train_documents": excluded_documents,
        }
        parent_format = cooldown.value.get("format_audit")
        parent_license = cooldown.value.get("license_audit")
        if not isinstance(parent_format, Mapping) or not isinstance(parent_license, Mapping):
            raise DataAuditError("cooldown format/license audits are invalid")
        format_audit = deepcopy(dict(parent_format))
        format_audit.update(
            {
                "complete": True,
                "phase_exclusion": projection,
                "filtered_outputs": source_map_roles,
            }
        )
        license_audit = deepcopy(dict(parent_license))
        license_audit.update(
            {
                "complete": True,
                "parent_attribution_inventory": parent_license.get("attribution_inventory"),
                "attribution_inventory": file_lists["attribution"],
                "phase_exclusion": projection,
            }
        )
        source_map_sidecar = work / "source-map.json"
        license_audit_sidecar = work / "license-audit.json"
        atomic_write_json(source_map_sidecar, source_map)
        atomic_write_json(license_audit_sidecar, license_audit)
        phase_sidecars = {
            "source_map": _file_identity(
                source_map_sidecar,
                relative=source_map_sidecar.name,
            ),
            "license_audit": _file_identity(
                license_audit_sidecar,
                relative=license_audit_sidecar.name,
            ),
            "attribution_file_list": file_lists["attribution"],
        }
        materialization_audit = {
            "complete": True,
            "network_policy": "offline-authenticated-phase-exclusion",
            "method": PHASE_EXCLUSION_ALGORITHM,
            "phase_exclusion": projection,
            "sidecars": phase_sidecars,
            "sources": [
                {
                    "source_id": source_id,
                    "retained_train_rows": train_source_rows.get(source_id, 0),
                    "retained_train_tokens": train_source_tokens.get(source_id, 0),
                    "excluded_train_rows": excluded_source_rows.get(source_id, 0),
                    "excluded_train_tokens": excluded_source_tokens.get(source_id, 0),
                }
                for source_id in sorted(
                    {
                        *train_source_rows,
                        *excluded_source_rows,
                        *owners["train"].values(),
                    }
                )
            ],
        }
        source_rows = {
            "train": train_source_rows,
            "validation": validation_source_rows,
        }
        source_tokens = {
            "train": train_source_tokens,
            "validation": validation_source_tokens,
        }
        sources = _rebuilt_sources(
            cooldown.value,
            source_map_roles=source_map_roles,
            source_rows=source_rows,
            source_tokens=source_tokens,
            excluded_tokens=excluded_source_tokens,
            excluded_rows=excluded_source_rows,
        )
        identity = {
            "recipe_id": cooldown.value.get("recipe_id"),
            "recipe_sha256": cooldown.value.get("recipe_sha256"),
            "resolved_source_lock_sha256": cooldown.value.get("resolved_source_lock_sha256"),
            "tokenizer_manifest_sha256": cooldown.value.get("tokenizer_manifest_sha256"),
            "extractor_source_sha256": source_sha_before,
            "profile": profile,
            "sources": sources,
            "train_files": train_files,
            "validation_files": validation_files,
            "attribution_files": attribution_files,
            "file_lists": file_lists,
            "source_map": source_map,
            "source_mix": source_mix,
            "format_audit": format_audit,
            "license_audit": license_audit,
            "materialization_audit": materialization_audit,
        }
        corpus_fingerprint = _canonical_sha256(identity)
        audits = deepcopy(
            dict(cooldown.value.get("audits"))
            if isinstance(cooldown.value.get("audits"), Mapping)
            else {}
        )
        gates = cooldown.audit.get("gates")
        if not isinstance(gates, Mapping):
            raise DataAuditError("cooldown audit gates are invalid")
        for name, gate in gates.items():
            if not isinstance(name, str) or not isinstance(gate, Mapping):
                raise DataAuditError("cooldown audit gate is invalid")
            status = gate.get("status")
            if not isinstance(status, str) or not status:
                raise DataAuditError("cooldown audit gate status is invalid")
            audits[name] = "pending_reaudit_phase_excluded_output"
        audits.update(
            {
                "cross_phase_stable_id_exclusion": ("complete_source_scoped_set_difference"),
                "validation_byte_preservation": "complete_sha256_identical",
            }
        )
        parent_reasons = cooldown.value.get("rejection_reasons")
        rejection_reasons = (
            deepcopy(dict(parent_reasons)) if isinstance(parent_reasons, Mapping) else {}
        )
        rejection_reasons["primary_phase_stable_id_intersection"] = excluded_documents
        manifest_value = {
            "schema_version": 1,
            "kind": "twen_extracted_base_jsonl_corpus",
            **identity,
            "corpus_fingerprint": corpus_fingerprint,
            "actual_train_tokens": sum(train_source_tokens.values()),
            "actual_validation_tokens": sum(validation_source_tokens.values()),
            "actual_train_documents": sum(train_source_rows.values()),
            "actual_validation_documents": sum(validation_source_rows.values()),
            "rejected_train_documents": int(cooldown.value.get("rejected_train_documents") or 0)
            + excluded_documents,
            "rejected_validation_documents": int(
                cooldown.value.get("rejected_validation_documents") or 0
            ),
            "rejection_reasons": rejection_reasons,
            "network_policy": "offline-authenticated-phase-exclusion",
            "audits": audits,
            "ready_for_data_prepare": True,
            "ready_for_training": False,
        }
        manifest = work / "corpus-manifest.json"
        atomic_write_json(manifest, manifest_value)

        metrics = {
            "primary_train_attribution_rows": primary_train_rows,
            "primary_unique_stable_keys": len(primary_stable_keys),
            "primary_duplicate_stable_key_rows": primary_duplicate_rows,
            "cooldown_train_attribution_rows": cooldown_train_rows,
            "cooldown_unique_stable_keys": len(cooldown_stable_keys),
            "cooldown_duplicate_stable_key_rows": cooldown_duplicate_rows,
            "intersecting_stable_keys": len(intersection),
            "excluded_cooldown_train_documents": excluded_documents,
            "retained_cooldown_train_documents": sum(train_source_rows.values()),
            "validation_documents": validation_rows,
            "retained_train_tokens": sum(train_source_tokens.values()),
            "excluded_train_tokens": sum(excluded_source_tokens.values()),
            "validation_tokens": sum(validation_source_tokens.values()),
        }
        attestation_value = {
            "schema_version": PHASE_EXCLUSION_SCHEMA_VERSION,
            "kind": PHASE_EXCLUSION_ATTESTATION_KIND,
            "algorithm": PHASE_EXCLUSION_ALGORITHM,
            "source_sha256": source_sha_before,
            "inputs": input_identities,
            "output": {
                "manifest": manifest.name,
                "manifest_sha256": sha256_file(manifest),
                "corpus_fingerprint": corpus_fingerprint,
                "ledger": ledger_identity,
                "sidecars": phase_sidecars,
            },
            "validation_byte_preserved": True,
            "requires_independent_audit": True,
            "ready_for_training": False,
            "metrics": metrics,
            "passed": True,
        }
        attestation_value["attestation_fingerprint"] = _canonical_sha256(attestation_value)
        attestation_path = work / "phase-exclusion-attestation.json"
        atomic_write_json(attestation_path, attestation_value)
        attestation_identity = _file_identity(
            attestation_path,
            relative=attestation_path.name,
        )
        attestation_identity["attestation_fingerprint"] = attestation_value[
            "attestation_fingerprint"
        ]
        atomic_write_json(
            work / "COMPLETE",
            {
                "schema_version": 1,
                "kind": "twen_extracted_base_jsonl_complete",
                "corpus_fingerprint": corpus_fingerprint,
                "manifest": manifest.name,
                "manifest_sha256": sha256_file(manifest),
                "file_lists": file_lists,
                "ready_for_training": False,
                "phase_exclusion_kind": PHASE_EXCLUSION_COMPLETE_KIND,
                "phase_exclusion_attestation": attestation_identity,
                "phase_exclusion_ledger": ledger_identity,
                "phase_exclusion_sidecars": phase_sidecars,
            },
        )

        source_sha_after = _current_source_sha256()
        primary_after, cooldown_after = _phase_input_snapshot(
            primary_manifest,
            primary_audit,
            cooldown_manifest,
            cooldown_audit,
        )
        if source_sha_after != source_sha_before:
            raise DataAuditError("phase-exclusion source changed during materialization")
        if {
            "primary": primary_after.identity,
            "cooldown": cooldown_after.identity,
        } != input_identities:
            raise DataAuditError("phase-exclusion input identity changed during materialization")
        _validate_phase_exclusion_output(manifest)
        os.replace(work, root)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return root / "corpus-manifest.json"


__all__ = [
    "PHASE_EXCLUSION_ALGORITHM",
    "PHASE_EXCLUSION_ATTESTATION_KIND",
    "PHASE_EXCLUSION_SCHEMA_VERSION",
    "materialize_phase_excluded_cooldown",
    "validate_phase_exclusion_output",
]
