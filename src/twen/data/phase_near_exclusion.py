"""Authenticated exhaustive near-duplicate exclusion for Base cooldown phases.

This is intentionally separate from :mod:`twen.data.phase_exclusion`: existing
stable-ID exclusion attestations bind that module's exact source bytes.  The
materializer here consumes a separately pinned, failed phase-disjointness
attestation whose near-duplicate examples are demonstrably exhaustive, verifies
every recorded pair against the authenticated train bytes, and removes only the
recorded cooldown ``(path, line)`` locations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
from collections import defaultdict, deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..source_identity import twen_source_tree_sha256
from ..utils import atomic_write_json, sha256_file
from .audits import (
    DataAuditError,
    _lexical_tokens,
    _normalize_text,
    _one_permutation_signature,
    _signature_similarity,
)
from .cursor import AuthenticatedSourceMap
from .phase_exclusion import (
    _Attribution,
    _canonical_sha256,
    _file_identity,
    _iter_attribution,
    _owned_regular_file,
    _parse_document_text,
    _phase_input_snapshot,
    _PhaseInput,
    _rebuilt_sources,
    _safe_relative,
    _source_categories,
    _source_owners,
    _write_file_lists,
    _write_filtered_attribution,
)
from .prepared import validate_prepared_corpus
from .sources import validate_extracted_base_corpus

PHASE_NEAR_EXCLUSION_SCHEMA_VERSION = 1
PHASE_NEAR_EXCLUSION_ALGORITHM = (
    "authenticated-exhaustive-phase-near-duplicate-locator-exclusion-v1"
)
PHASE_NEAR_EXCLUSION_ATTESTATION_KIND = "twen_phase_near_exclusion_attestation"
PHASE_NEAR_EXCLUSION_LEDGER_KIND = "twen_phase_near_exclusion_ledger"
PHASE_NEAR_EXCLUSION_COMPLETE_KIND = "twen_phase_near_excluded_corpus_complete"

_PHASE_DISJOINTNESS_KIND = "twen_v4_phase_disjointness_attestation"
_PHASE_DISJOINTNESS_COMPLETE_KIND = "twen_v4_phase_disjointness_complete"
_STABLE_ID_ALGORITHM = "source-scoped-authenticated-stable-id-intersection-v1"
_NORMALIZED_EXACT_ALGORITHM = "unicode-nfkc-whitespace-sha256-intersection-v1"
_NEAR_DUPLICATE_ALGORITHM = "lexical-5gram-one-permutation-minhash-lsh-v1"
_NEAR_DUPLICATE_THRESHOLD = 0.8
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class _DocumentLocator:
    source_id: str
    path: str
    line: int
    normalized_text_sha256: str

    @property
    def location(self) -> tuple[str, int]:
        return self.path, self.line

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "path": self.path,
            "line": self.line,
            "normalized_text_sha256": self.normalized_text_sha256,
        }


@dataclass(frozen=True, slots=True)
class _NearMatch:
    primary: _DocumentLocator
    cooldown: _DocumentLocator
    estimated_jaccard: float

    def to_dict(self) -> dict[str, object]:
        return {
            "primary": self.primary.to_dict(),
            "cooldown": self.cooldown.to_dict(),
            "estimated_jaccard": self.estimated_jaccard,
        }


@dataclass(frozen=True, slots=True)
class _FailedPhaseEvidence:
    path: Path
    sha256: str
    complete_sha256: str
    attestation_fingerprint: str
    scanner_source_sha256: str
    scanner_source_tree_sha256: str
    expected_near_matches: int
    primary_identity: Mapping[str, Any]
    cooldown_identity: Mapping[str, Any]
    matches: tuple[_NearMatch, ...]

    @property
    def identity(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "complete_sha256": self.complete_sha256,
            "attestation_fingerprint": self.attestation_fingerprint,
            "scanner_source_sha256": self.scanner_source_sha256,
            "scanner_source_tree_sha256": self.scanner_source_tree_sha256,
            "expected_near_matches": self.expected_near_matches,
        }


def _current_source_sha256() -> str:
    return sha256_file(Path(__file__))


def _scanner_path() -> Path:
    return Path(__file__).resolve().parents[3] / "scripts/attest_v4_phase_disjointness.py"


def _normalized_text_sha256(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", " ".join(text.split()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _required_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DataAuditError(f"{label} must be lowercase SHA256")
    return value


def _prepared_phase_identity(
    phase: _PhaseInput,
    prepared_path: str | Path,
) -> dict[str, object]:
    raw_path = Path(prepared_path).expanduser()
    if raw_path.is_symlink():
        raise DataAuditError("phase prepared manifest must not be a symlink")
    path = raw_path.resolve()
    try:
        prepared = validate_prepared_corpus(path)
    except (OSError, RuntimeError, ValueError) as error:
        raise DataAuditError(f"invalid phase prepared corpus: {path}: {error}") from error
    lineage = prepared.lineage
    if (
        not isinstance(lineage, Mapping)
        or lineage.get("ready_for_training") is not True
        or lineage.get("research_only") is True
        or lineage.get("pending_audits")
    ):
        raise DataAuditError(f"phase prepared corpus is not fully training-ready: {path}")
    try:
        source_map = AuthenticatedSourceMap.from_prepared_manifest(prepared)
    except (OSError, TypeError, ValueError) as error:
        raise DataAuditError(
            f"phase prepared corpus has no authenticated train source-map: {path}"
        ) from error
    if source_map.extracted_manifest_sha256 != phase.manifest_sha256:
        raise DataAuditError(
            "phase prepared source-map does not bind the extracted manifest"
        )
    return {
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "dataset_fingerprint": prepared.dataset_fingerprint,
        "source_map_sha256": source_map.fingerprint,
    }


def _scanner_phase_identity(
    phase: _PhaseInput,
    prepared_path: str | Path,
) -> dict[str, object]:
    return {
        "manifest_path": str(phase.manifest_path),
        "manifest_sha256": phase.manifest_sha256,
        "corpus_fingerprint": phase.corpus_fingerprint,
        "audit_attestation_path": str(phase.audit_path),
        "audit_attestation_sha256": phase.audit_sha256,
        "audit_attestation_fingerprint": phase.audit_fingerprint,
        "prepared": _prepared_phase_identity(phase, prepared_path),
    }


def _authenticated_phase_pair(
    *,
    primary_manifest: str | Path,
    primary_audit: str | Path,
    primary_prepared: str | Path,
    cooldown_manifest: str | Path,
    cooldown_audit: str | Path,
    cooldown_prepared: str | Path,
) -> tuple[_PhaseInput, _PhaseInput, dict[str, object], dict[str, object]]:
    primary, cooldown = _phase_input_snapshot(
        primary_manifest,
        primary_audit,
        cooldown_manifest,
        cooldown_audit,
    )
    return (
        primary,
        cooldown,
        _scanner_phase_identity(primary, primary_prepared),
        _scanner_phase_identity(cooldown, cooldown_prepared),
    )


def _parse_locator(value: object, label: str) -> _DocumentLocator:
    if not isinstance(value, Mapping) or set(value) != {
        "source_id",
        "path",
        "line",
        "normalized_text_sha256",
    }:
        raise DataAuditError(f"{label} locator has an invalid schema")
    source_id = value.get("source_id")
    line = value.get("line")
    if (
        not isinstance(source_id, str)
        or not source_id
        or isinstance(line, bool)
        or not isinstance(line, int)
        or line <= 0
    ):
        raise DataAuditError(f"{label} locator identity is invalid")
    return _DocumentLocator(
        source_id=source_id,
        path=_safe_relative(value.get("path"), f"{label}.path"),
        line=line,
        normalized_text_sha256=_required_sha256(
            value.get("normalized_text_sha256"),
            f"{label}.normalized_text_sha256",
        ),
    )


def _authenticate_failed_phase_attestation(
    *,
    attestation_path: str | Path,
    expected_attestation_sha256: str,
    expected_near_matches: int,
    primary_identity: Mapping[str, Any],
    cooldown_identity: Mapping[str, Any],
) -> _FailedPhaseEvidence:
    if (
        isinstance(expected_near_matches, bool)
        or not isinstance(expected_near_matches, int)
        or expected_near_matches <= 0
    ):
        raise DataAuditError("expected near-duplicate match count must be positive")
    expected_sha = _required_sha256(
        expected_attestation_sha256,
        "expected phase-disjointness attestation SHA256",
    )
    raw_path = Path(attestation_path).expanduser()
    if raw_path.is_symlink():
        raise DataAuditError("phase-disjointness attestation must not be a symlink")
    path = raw_path.resolve()
    complete_path = path.parent / "COMPLETE"
    if complete_path.is_symlink():
        raise DataAuditError("phase-disjointness COMPLETE must not be a symlink")
    try:
        raw = path.read_bytes()
        complete_raw = complete_path.read_bytes()
        value = json.loads(raw)
        complete = json.loads(complete_raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DataAuditError("cannot read failed phase-disjointness evidence") from error
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise DataAuditError("phase-disjointness attestation differs from its external pin")
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("kind") != _PHASE_DISJOINTNESS_KIND
    ):
        raise DataAuditError("unsupported phase-disjointness attestation")
    fingerprint = _required_sha256(
        value.get("attestation_fingerprint"),
        "phase-disjointness attestation fingerprint",
    )
    unsigned = dict(value)
    unsigned.pop("attestation_fingerprint", None)
    if _canonical_sha256(unsigned) != fingerprint:
        raise DataAuditError("phase-disjointness attestation fingerprint mismatch")
    if (
        not isinstance(complete, Mapping)
        or complete.get("schema_version") != 1
        or complete.get("kind") != _PHASE_DISJOINTNESS_COMPLETE_KIND
        or complete.get("attestation") != path.name
        or complete.get("attestation_sha256") != actual_sha
        or complete.get("attestation_fingerprint") != fingerprint
        or complete.get("passed") is not False
    ):
        raise DataAuditError("phase-disjointness COMPLETE does not bind the failed evidence")

    scanner_source_sha = _required_sha256(
        value.get("scanner_source_sha256"),
        "phase-disjointness scanner source SHA256",
    )
    scanner_source_tree_sha = _required_sha256(
        value.get("scanner_source_tree_sha256"),
        "phase-disjointness scanner source-tree SHA256",
    )
    scanner = _scanner_path()
    if not scanner.is_file() or sha256_file(scanner) != scanner_source_sha:
        raise DataAuditError("phase-disjointness scanner source changed")
    if twen_source_tree_sha256() != scanner_source_tree_sha:
        raise DataAuditError("phase-disjointness scanner source tree changed")
    if value.get("primary") != primary_identity or value.get("cooldown") != cooldown_identity:
        raise DataAuditError("phase-disjointness inputs differ from authenticated phases")
    if (
        value.get("scope") != "authenticated_train_inventories_only"
        or value.get("stores_raw_text") is not False
        or value.get("passed") is not False
    ):
        raise DataAuditError("failed phase-disjointness scope/status is invalid")

    gates = value.get("gates")
    metrics = value.get("metrics")
    examples = value.get("examples")
    if (
        not isinstance(gates, Mapping)
        or set(gates) != {"stable_id_exact", "normalized_text_exact", "near_duplicate"}
        or not isinstance(metrics, Mapping)
        or not isinstance(examples, Mapping)
        or set(examples) != {"stable_id_exact", "normalized_text_exact", "near_duplicate"}
    ):
        raise DataAuditError("failed phase-disjointness gate/example inventory is invalid")
    stable_gate = gates["stable_id_exact"]
    normalized_gate = gates["normalized_text_exact"]
    near_gate = gates["near_duplicate"]
    if (
        not isinstance(stable_gate, Mapping)
        or stable_gate.get("algorithm") != _STABLE_ID_ALGORITHM
        or stable_gate.get("matches") != 0
        or stable_gate.get("passed") is not True
        or metrics.get("stable_id_exact_matches") != 0
        or examples.get("stable_id_exact") != []
        or not isinstance(normalized_gate, Mapping)
        or normalized_gate.get("algorithm") != _NORMALIZED_EXACT_ALGORITHM
        or normalized_gate.get("matches") != 0
        or normalized_gate.get("passed") is not True
        or metrics.get("normalized_text_exact_matches") != 0
        or examples.get("normalized_text_exact") != []
    ):
        raise DataAuditError(
            "near exclusion requires zero stable-ID and normalized-exact matches"
        )
    raw_near_examples = examples.get("near_duplicate")
    if (
        not isinstance(near_gate, Mapping)
        or near_gate.get("algorithm") != _NEAR_DUPLICATE_ALGORITHM
        or near_gate.get("estimated_jaccard_threshold") != _NEAR_DUPLICATE_THRESHOLD
        or near_gate.get("matches") != expected_near_matches
        or near_gate.get("passed") is not False
        or metrics.get("near_duplicate_matches") != expected_near_matches
        or not isinstance(raw_near_examples, list)
        or len(raw_near_examples) != expected_near_matches
    ):
        raise DataAuditError(
            "phase-disjointness near examples are truncated or have the wrong count"
        )

    matches: list[_NearMatch] = []
    cooldown_locations: set[tuple[str, int]] = set()
    for index, raw_match in enumerate(raw_near_examples):
        if not isinstance(raw_match, Mapping) or set(raw_match) != {
            "primary",
            "cooldown",
            "estimated_jaccard",
        }:
            raise DataAuditError(f"near-duplicate example[{index}] schema is invalid")
        similarity = raw_match.get("estimated_jaccard")
        if (
            isinstance(similarity, bool)
            or not isinstance(similarity, (int, float))
            or not math.isfinite(float(similarity))
            or not _NEAR_DUPLICATE_THRESHOLD <= float(similarity) <= 1.0
        ):
            raise DataAuditError(f"near-duplicate example[{index}] similarity is invalid")
        match = _NearMatch(
            primary=_parse_locator(raw_match.get("primary"), f"example[{index}].primary"),
            cooldown=_parse_locator(raw_match.get("cooldown"), f"example[{index}].cooldown"),
            estimated_jaccard=float(similarity),
        )
        if match.cooldown.location in cooldown_locations:
            raise DataAuditError("near-duplicate examples repeat a cooldown path/line")
        cooldown_locations.add(match.cooldown.location)
        matches.append(match)

    if path.read_bytes() != raw or complete_path.read_bytes() != complete_raw:
        raise DataAuditError("phase-disjointness evidence changed during authentication")
    return _FailedPhaseEvidence(
        path=path,
        sha256=actual_sha,
        complete_sha256=hashlib.sha256(complete_raw).hexdigest(),
        attestation_fingerprint=fingerprint,
        scanner_source_sha256=scanner_source_sha,
        scanner_source_tree_sha256=scanner_source_tree_sha,
        expected_near_matches=expected_near_matches,
        primary_identity=dict(primary_identity),
        cooldown_identity=dict(cooldown_identity),
        matches=tuple(matches),
    )


def _load_locator_texts(
    phase: _PhaseInput,
    locators: set[_DocumentLocator],
    *,
    label: str,
) -> dict[_DocumentLocator, str]:
    owners = _source_owners(phase.value)["train"]
    requested: dict[str, dict[int, _DocumentLocator]] = defaultdict(dict)
    for locator in locators:
        owner = owners.get(locator.path)
        if owner != locator.source_id:
            raise DataAuditError(
                f"{label} locator is outside its authenticated train inventory: "
                f"{locator.path}:{locator.line}"
            )
        if locator.line in requested[locator.path]:
            raise DataAuditError(f"{label} locator path/line is ambiguous")
        requested[locator.path][locator.line] = locator

    result: dict[_DocumentLocator, str] = {}
    for relative, lines in sorted(requested.items()):
        path = _owned_regular_file(
            phase.manifest_path.parent,
            relative,
            f"{label} train locator file",
        )
        maximum = max(lines)
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                locator = lines.get(line_number)
                if locator is not None:
                    text = _parse_document_text(raw, path, line_number)
                    if _normalized_text_sha256(text) != locator.normalized_text_sha256:
                        raise DataAuditError(
                            f"{label} locator normalized SHA mismatch: "
                            f"{relative}:{line_number}"
                        )
                    result[locator] = text
                if line_number >= maximum:
                    break
    if len(result) != len(locators):
        raise DataAuditError(f"{label} near-duplicate locators are incomplete")
    return result


def _recompute_near_matches(
    primary: _PhaseInput,
    cooldown: _PhaseInput,
    matches: tuple[_NearMatch, ...],
) -> tuple[_NearMatch, ...]:
    primary_text = _load_locator_texts(
        primary,
        {match.primary for match in matches},
        label="primary",
    )
    cooldown_text = _load_locator_texts(
        cooldown,
        {match.cooldown for match in matches},
        label="cooldown",
    )
    verified: list[_NearMatch] = []
    for match in matches:
        primary_signature = _one_permutation_signature(
            _lexical_tokens(primary_text[match.primary]),
            match.primary.normalized_text_sha256,
        )
        cooldown_signature = _one_permutation_signature(
            _lexical_tokens(cooldown_text[match.cooldown]),
            match.cooldown.normalized_text_sha256,
        )
        similarity = _signature_similarity(primary_signature, cooldown_signature)
        if (
            similarity != match.estimated_jaccard
            or similarity < _NEAR_DUPLICATE_THRESHOLD
        ):
            raise DataAuditError(
                "near-duplicate similarity differs from the failed phase attestation: "
                f"{match.cooldown.path}:{match.cooldown.line}"
            )
        verified.append(
            _NearMatch(
                primary=match.primary,
                cooldown=match.cooldown,
                estimated_jaccard=similarity,
            )
        )
    if len(verified) != len(matches):
        raise DataAuditError("not every near-duplicate pair was recomputed")
    return tuple(verified)


def _copy_documents_excluding_near_locations(
    *,
    phase: _PhaseInput,
    work: Path,
    role: str,
    owners: Mapping[str, str],
    source_categories: Mapping[str, str],
    attribution: Mapping[tuple[str, str], deque[_Attribution]],
    targets: Mapping[tuple[str, int], _NearMatch],
    excluded_attribution_locations: set[tuple[str, int]],
    selection_states: Mapping[tuple[str, str], set[bool]],
    ledger_handle: Any,
) -> tuple[list[dict[str, object]], dict[str, int], dict[str, int], int, set[tuple[str, int]]]:
    raw_inventory = phase.value.get(f"{role}_files")
    if not isinstance(raw_inventory, list):
        raise DataAuditError(f"cooldown {role} inventory is invalid")
    output_inventory: list[dict[str, object]] = []
    source_rows: dict[str, int] = defaultdict(int)
    source_tokens: dict[str, int] = defaultdict(int)
    excluded_documents = 0
    seen_targets: set[tuple[str, int]] = set()
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
        category = source_categories.get(source_id)
        if category is None:
            raise DataAuditError(f"source inventory has no category for {source_id}")
        code = category == "code" or "code" in source_id.casefold()
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
                    raise DataAuditError(
                        f"blank corpus JSONL row at {source}:{line_number}"
                    )
                text = _parse_document_text(raw_line, source, line_number)
                normalized = _normalize_text(text, code=code)
                text_sha = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                document_key = (source_id, text_sha)
                queue = attribution.get(document_key)
                if queue is None or not queue:
                    raise DataAuditError(
                        "cooldown attribution does not cover corpus document exactly: "
                        f"{relative}:{line_number}"
                    )
                record = queue.popleft()
                if record.split != role:
                    raise DataAuditError(
                        "cooldown attribution split differs from corpus inventory: "
                        f"{relative}:{line_number}"
                    )
                location = (relative, line_number)
                match = targets.get(location) if role == "train" else None
                selected = match is not None
                selection_states[document_key].add(selected)
                if selected:
                    assert match is not None
                    if (
                        match.cooldown.source_id != source_id
                        or match.cooldown.normalized_text_sha256
                        != _normalized_text_sha256(text)
                    ):
                        raise DataAuditError(
                            "near-duplicate target differs from cooldown train bytes"
                        )
                    excluded_documents += 1
                    seen_targets.add(location)
                    excluded_attribution_locations.add(
                        (record.relative, record.line_number)
                    )
                    ledger_handle.write(
                        json.dumps(
                            {
                                "schema_version": PHASE_NEAR_EXCLUSION_SCHEMA_VERSION,
                                "kind": PHASE_NEAR_EXCLUSION_LEDGER_KIND,
                                "algorithm": PHASE_NEAR_EXCLUSION_ALGORITHM,
                                "reason": "authenticated_cross_phase_near_duplicate",
                                "primary": match.primary.to_dict(),
                                "cooldown": match.cooldown.to_dict(),
                                "estimated_jaccard": match.estimated_jaccard,
                                "cooldown_attribution": {
                                    "path": record.relative,
                                    "line": record.line_number,
                                    "stable_id": record.stable_id,
                                    "text_sha256": record.text_sha256,
                                    "token_count_with_eos": record.token_count,
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
        seen_targets,
    )


def _validate_unambiguous_selection_states(
    selection_states: Mapping[tuple[str, str], set[bool]],
) -> None:
    if any(len(states) != 1 for states in selection_states.values()):
        raise DataAuditError(
            "partial duplicate-text exclusion is ambiguous without document locators "
            "in attribution"
        )


def _output_attestation(manifest_path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    report = validate_extracted_base_corpus(manifest_path, verify_hashes=True)
    if report.get("ready_for_training") is not False:
        raise DataAuditError("near-excluded corpus requires an independent audit")
    root = manifest_path.parent
    try:
        marker = json.loads((root / "COMPLETE").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataAuditError("near-exclusion output has no valid COMPLETE") from error
    if (
        not isinstance(marker, Mapping)
        or marker.get("phase_near_exclusion_kind")
        != PHASE_NEAR_EXCLUSION_COMPLETE_KIND
        or marker.get("ready_for_training") is not False
    ):
        raise DataAuditError("near-exclusion COMPLETE contract is invalid")
    attestation_identity = marker.get("phase_near_exclusion_attestation")
    ledger_identity = marker.get("phase_near_exclusion_ledger")
    if not isinstance(attestation_identity, Mapping) or not isinstance(
        ledger_identity,
        Mapping,
    ):
        raise DataAuditError("near-exclusion COMPLETE identities are missing")
    attestation_path = _owned_regular_file(
        root,
        attestation_identity.get("path"),
        "near-exclusion attestation",
    )
    ledger_path = _owned_regular_file(
        root,
        ledger_identity.get("path"),
        "near-exclusion ledger",
    )
    for identity, path, label in (
        (attestation_identity, attestation_path, "attestation"),
        (ledger_identity, ledger_path, "ledger"),
    ):
        if (
            identity.get("size") != path.stat().st_size
            or identity.get("sha256") != sha256_file(path)
        ):
            raise DataAuditError(f"near-exclusion {label} identity mismatch")
    try:
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataAuditError("near-exclusion attestation is invalid") from error
    if not isinstance(attestation, dict):
        raise DataAuditError("near-exclusion attestation must be an object")
    fingerprint = attestation.get("attestation_fingerprint")
    unsigned = dict(attestation)
    unsigned.pop("attestation_fingerprint", None)
    metrics = attestation.get("metrics")
    if (
        attestation.get("schema_version") != PHASE_NEAR_EXCLUSION_SCHEMA_VERSION
        or attestation.get("kind") != PHASE_NEAR_EXCLUSION_ATTESTATION_KIND
        or attestation.get("algorithm") != PHASE_NEAR_EXCLUSION_ALGORITHM
        or attestation.get("source_sha256") != _current_source_sha256()
        or attestation.get("source_tree_sha256") != twen_source_tree_sha256()
        or attestation.get("passed") is not True
        or attestation.get("requires_independent_audit") is not True
        or attestation.get("ready_for_training") is not False
        or not isinstance(fingerprint, str)
        or fingerprint != _canonical_sha256(unsigned)
        or attestation_identity.get("attestation_fingerprint") != fingerprint
        or not isinstance(metrics, Mapping)
        or metrics.get("excluded_cooldown_train_documents")
        != metrics.get("recomputed_near_duplicate_pairs")
    ):
        raise DataAuditError("near-exclusion attestation contract is invalid")
    output = attestation.get("output")
    if (
        not isinstance(output, Mapping)
        or output.get("manifest") != manifest_path.name
        or output.get("manifest_sha256") != report["manifest_sha256"]
        or output.get("corpus_fingerprint") != report["corpus_fingerprint"]
        or output.get("ledger") != ledger_identity
    ):
        raise DataAuditError("near-exclusion attestation output identity mismatch")
    ledger_rows = 0
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise DataAuditError(
                    f"invalid near-exclusion ledger row {line_number}"
                ) from error
            if (
                not isinstance(row, Mapping)
                or row.get("kind") != PHASE_NEAR_EXCLUSION_LEDGER_KIND
                or row.get("algorithm") != PHASE_NEAR_EXCLUSION_ALGORITHM
                or "text" in row
            ):
                raise DataAuditError("near-exclusion ledger contract is invalid")
            ledger_rows += 1
    if ledger_rows != metrics["excluded_cooldown_train_documents"]:
        raise DataAuditError("near-exclusion ledger coverage is incomplete")
    return attestation, marker


def validate_phase_near_exclusion_output(
    manifest_path: str | Path,
) -> Mapping[str, Any]:
    """Authenticate a published exhaustive near-excluded extracted corpus."""

    attestation, _ = _output_attestation(Path(manifest_path).resolve())
    return attestation


def materialize_phase_near_excluded_cooldown(
    *,
    primary_manifest: str | Path,
    primary_audit: str | Path,
    primary_prepared: str | Path,
    cooldown_manifest: str | Path,
    cooldown_audit: str | Path,
    cooldown_prepared: str | Path,
    phase_attestation: str | Path,
    expected_phase_attestation_sha256: str,
    expected_near_matches: int,
    output_root: str | Path,
) -> Path:
    """Atomically remove every exhaustively attested cooldown near duplicate."""

    source_sha_before = _current_source_sha256()
    source_tree_before = twen_source_tree_sha256()
    (
        primary,
        cooldown,
        primary_identity,
        cooldown_identity,
    ) = _authenticated_phase_pair(
        primary_manifest=primary_manifest,
        primary_audit=primary_audit,
        primary_prepared=primary_prepared,
        cooldown_manifest=cooldown_manifest,
        cooldown_audit=cooldown_audit,
        cooldown_prepared=cooldown_prepared,
    )
    evidence = _authenticate_failed_phase_attestation(
        attestation_path=phase_attestation,
        expected_attestation_sha256=expected_phase_attestation_sha256,
        expected_near_matches=expected_near_matches,
        primary_identity=primary_identity,
        cooldown_identity=cooldown_identity,
    )
    recomputed = _recompute_near_matches(primary, cooldown, evidence.matches)
    recomputed_payload = [match.to_dict() for match in recomputed]
    recomputed_fingerprint = _canonical_sha256(recomputed_payload)
    targets = {match.cooldown.location: match for match in recomputed}
    if len(targets) != expected_near_matches:
        raise DataAuditError("near-duplicate cooldown target coverage is not one-to-one")

    requested_root = Path(output_root).expanduser()
    if requested_root.is_symlink():
        raise DataAuditError("near-exclusion output must not be a symlink")
    root = requested_root.absolute()
    if root.exists():
        raise DataAuditError(f"near-exclusion output already exists: {root}")
    for phase in (primary, cooldown):
        try:
            root.resolve(strict=False).relative_to(phase.manifest_path.parent)
        except ValueError:
            pass
        else:
            raise DataAuditError("near-exclusion output overlaps an input corpus")
    root.parent.mkdir(parents=True, exist_ok=True)
    work = Path(
        tempfile.mkdtemp(
            prefix=f".{root.name}.phase-near-exclusion-",
            dir=root.parent,
        )
    )
    try:
        owners = _source_owners(cooldown.value)
        source_categories = _source_categories(cooldown.value)
        cooldown_records: dict[tuple[str, str], deque[_Attribution]] = defaultdict(deque)
        for record in _iter_attribution(cooldown):
            cooldown_records[record.document_key].append(record)
        excluded_attribution_locations: set[tuple[str, int]] = set()
        selection_states: dict[tuple[str, str], set[bool]] = defaultdict(set)
        ledger_path = work / "phase-near-exclusion-ledger.jsonl"
        with ledger_path.open("w", encoding="utf-8") as ledger:
            (
                train_files,
                train_source_rows,
                train_source_tokens,
                excluded_documents,
                seen_targets,
            ) = _copy_documents_excluding_near_locations(
                phase=cooldown,
                work=work,
                role="train",
                owners=owners["train"],
                source_categories=source_categories,
                attribution=cooldown_records,
                targets=targets,
                excluded_attribution_locations=excluded_attribution_locations,
                selection_states=selection_states,
                ledger_handle=ledger,
            )
            (
                validation_files,
                validation_source_rows,
                validation_source_tokens,
                validation_excluded,
                validation_targets,
            ) = _copy_documents_excluding_near_locations(
                phase=cooldown,
                work=work,
                role="validation",
                owners=owners["validation"],
                source_categories=source_categories,
                attribution=cooldown_records,
                targets=targets,
                excluded_attribution_locations=excluded_attribution_locations,
                selection_states=selection_states,
                ledger_handle=ledger,
            )
        if (
            validation_excluded
            or validation_targets
            or seen_targets != set(targets)
            or excluded_documents != expected_near_matches
            or len(excluded_attribution_locations) != expected_near_matches
        ):
            raise DataAuditError("near-exclusion document coverage is incomplete")
        _validate_unambiguous_selection_states(selection_states)
        unmatched = sum(len(records) for records in cooldown_records.values())
        if unmatched:
            raise DataAuditError(
                f"cooldown attribution has no exact corpus document for {unmatched} rows"
            )

        attribution_files = _write_filtered_attribution(
            cooldown,
            work,
            excluded_attribution_locations,
        )
        inventories = {
            "train": train_files,
            "validation": validation_files,
            "attribution": attribution_files,
        }
        for inventory in inventories.values():
            inventory.sort(key=lambda item: str(item["path"]))
        file_lists = _write_file_lists(work, inventories)
        ledger_identity = _file_identity(ledger_path, relative=ledger_path.name)

        excluded_source_rows: dict[str, int] = defaultdict(int)
        excluded_source_tokens: dict[str, int] = defaultdict(int)
        for record in _iter_attribution(cooldown):
            if (record.relative, record.line_number) in excluded_attribution_locations:
                excluded_source_rows[record.source_id] += 1
                excluded_source_tokens[record.source_id] += record.token_count

        source_map_roles: dict[str, list[dict[str, object]]] = {
            "train": [
                {"source_id": owners["train"][str(entry["path"])], **entry}
                for entry in train_files
            ],
            "validation": [
                {"source_id": owners["validation"][str(entry["path"])], **entry}
                for entry in validation_files
            ],
        }
        parent_source_map = cooldown.value.get("source_map")
        parent_source_mix = cooldown.value.get("source_mix")
        if (
            not isinstance(parent_source_map, Mapping)
            or not isinstance(parent_source_mix, Mapping)
            or not isinstance(parent_source_mix.get("sources"), list)
        ):
            raise DataAuditError("cooldown source-map/source-mix is invalid")
        source_map_unsigned = {
            "schema_version": parent_source_map.get("schema_version"),
            "algorithm": parent_source_map.get("algorithm"),
            "roles": source_map_roles,
        }
        source_map = {
            **source_map_unsigned,
            "fingerprint": _canonical_sha256(source_map_unsigned),
        }
        profile = f"{cooldown.value.get('profile')}-near-excluded"
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
            "schema_version": PHASE_NEAR_EXCLUSION_SCHEMA_VERSION,
            "algorithm": PHASE_NEAR_EXCLUSION_ALGORITHM,
            "source_sha256": source_sha_before,
            "source_tree_sha256": source_tree_before,
            "failed_phase_attestation": evidence.identity,
            "recomputed_matches_fingerprint": recomputed_fingerprint,
            "recomputed_near_duplicate_pairs": len(recomputed),
            "excluded_train_documents": excluded_documents,
            "ledger": ledger_identity,
        }
        parent_format = cooldown.value.get("format_audit")
        parent_license = cooldown.value.get("license_audit")
        if not isinstance(parent_format, Mapping) or not isinstance(
            parent_license,
            Mapping,
        ):
            raise DataAuditError("cooldown format/license audits are invalid")
        format_audit = deepcopy(dict(parent_format))
        format_audit.update(
            {
                "complete": True,
                "phase_near_exclusion": projection,
                "filtered_outputs": source_map_roles,
            }
        )
        license_audit = deepcopy(dict(parent_license))
        license_audit.update(
            {
                "complete": True,
                "parent_attribution_inventory": parent_license.get(
                    "attribution_inventory"
                ),
                "attribution_inventory": file_lists["attribution"],
                "phase_near_exclusion": projection,
            }
        )
        source_map_sidecar = work / "source-map.json"
        license_audit_sidecar = work / "license-audit.json"
        atomic_write_json(source_map_sidecar, source_map)
        atomic_write_json(license_audit_sidecar, license_audit)
        sidecars = {
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
            "network_policy": "offline-authenticated-phase-near-exclusion",
            "method": PHASE_NEAR_EXCLUSION_ALGORITHM,
            "phase_near_exclusion": projection,
            "sidecars": sidecars,
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
        sources = _rebuilt_sources(
            cooldown.value,
            source_map_roles=source_map_roles,
            source_rows={
                "train": train_source_rows,
                "validation": validation_source_rows,
            },
            source_tokens={
                "train": train_source_tokens,
                "validation": validation_source_tokens,
            },
            excluded_tokens=excluded_source_tokens,
            excluded_rows=excluded_source_rows,
        )
        identity = {
            "recipe_id": cooldown.value.get("recipe_id"),
            "recipe_sha256": cooldown.value.get("recipe_sha256"),
            "resolved_source_lock_sha256": cooldown.value.get(
                "resolved_source_lock_sha256"
            ),
            "tokenizer_manifest_sha256": cooldown.value.get(
                "tokenizer_manifest_sha256"
            ),
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
        parent_audits = cooldown.value.get("audits")
        audits = (
            deepcopy(dict(parent_audits))
            if isinstance(parent_audits, Mapping)
            else {}
        )
        for name in tuple(audits):
            audits[name] = "pending_reaudit_phase_near_excluded_output"
        gates = cooldown.audit.get("gates")
        if not isinstance(gates, Mapping):
            raise DataAuditError("cooldown audit gates are invalid")
        for name in gates:
            if not isinstance(name, str):
                raise DataAuditError("cooldown audit gate name is invalid")
            audits[name] = "pending_reaudit_phase_near_excluded_output"
        audits.update(
            {
                "cross_phase_near_duplicate_exclusion": (
                    "complete_exhaustive_attested_locator_projection"
                ),
                "validation_byte_preservation": "complete_sha256_identical",
            }
        )
        parent_reasons = cooldown.value.get("rejection_reasons")
        rejection_reasons = (
            deepcopy(dict(parent_reasons))
            if isinstance(parent_reasons, Mapping)
            else {}
        )
        rejection_reasons["cross_phase_near_duplicate"] = excluded_documents
        manifest_value = {
            "schema_version": 1,
            "kind": "twen_extracted_base_jsonl_corpus",
            **identity,
            "corpus_fingerprint": corpus_fingerprint,
            "actual_train_tokens": sum(train_source_tokens.values()),
            "actual_validation_tokens": sum(validation_source_tokens.values()),
            "actual_train_documents": sum(train_source_rows.values()),
            "actual_validation_documents": sum(validation_source_rows.values()),
            "rejected_train_documents": int(
                cooldown.value.get("rejected_train_documents") or 0
            )
            + excluded_documents,
            "rejected_validation_documents": int(
                cooldown.value.get("rejected_validation_documents") or 0
            ),
            "rejection_reasons": rejection_reasons,
            "network_policy": "offline-authenticated-phase-near-exclusion",
            "audits": audits,
            "ready_for_data_prepare": True,
            "ready_for_training": False,
        }
        manifest = work / "corpus-manifest.json"
        atomic_write_json(manifest, manifest_value)

        metrics = {
            "expected_near_duplicate_matches": expected_near_matches,
            "recomputed_near_duplicate_pairs": len(recomputed),
            "excluded_cooldown_train_documents": excluded_documents,
            "retained_cooldown_train_documents": sum(train_source_rows.values()),
            "retained_train_tokens": sum(train_source_tokens.values()),
            "excluded_train_tokens": sum(excluded_source_tokens.values()),
            "validation_documents": sum(validation_source_rows.values()),
            "validation_tokens": sum(validation_source_tokens.values()),
        }
        attestation_value = {
            "schema_version": PHASE_NEAR_EXCLUSION_SCHEMA_VERSION,
            "kind": PHASE_NEAR_EXCLUSION_ATTESTATION_KIND,
            "algorithm": PHASE_NEAR_EXCLUSION_ALGORITHM,
            "source_sha256": source_sha_before,
            "source_tree_sha256": source_tree_before,
            "inputs": {
                "primary": primary_identity,
                "cooldown": cooldown_identity,
                "failed_phase_attestation": evidence.identity,
            },
            "output": {
                "manifest": manifest.name,
                "manifest_sha256": sha256_file(manifest),
                "corpus_fingerprint": corpus_fingerprint,
                "ledger": ledger_identity,
                "sidecars": sidecars,
            },
            "recomputed_matches_fingerprint": recomputed_fingerprint,
            "validation_byte_preserved": True,
            "requires_independent_audit": True,
            "ready_for_training": False,
            "metrics": metrics,
            "passed": True,
        }
        attestation_value["attestation_fingerprint"] = _canonical_sha256(
            attestation_value
        )
        attestation_path = work / "phase-near-exclusion-attestation.json"
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
                "phase_near_exclusion_kind": PHASE_NEAR_EXCLUSION_COMPLETE_KIND,
                "phase_near_exclusion_attestation": attestation_identity,
                "phase_near_exclusion_ledger": ledger_identity,
                "phase_near_exclusion_sidecars": sidecars,
            },
        )

        (
            primary_after,
            cooldown_after,
            primary_identity_after,
            cooldown_identity_after,
        ) = _authenticated_phase_pair(
            primary_manifest=primary_manifest,
            primary_audit=primary_audit,
            primary_prepared=primary_prepared,
            cooldown_manifest=cooldown_manifest,
            cooldown_audit=cooldown_audit,
            cooldown_prepared=cooldown_prepared,
        )
        evidence_after = _authenticate_failed_phase_attestation(
            attestation_path=phase_attestation,
            expected_attestation_sha256=expected_phase_attestation_sha256,
            expected_near_matches=expected_near_matches,
            primary_identity=primary_identity_after,
            cooldown_identity=cooldown_identity_after,
        )
        recomputed_after = _recompute_near_matches(
            primary_after,
            cooldown_after,
            evidence_after.matches,
        )
        if (
            primary_identity_after != primary_identity
            or cooldown_identity_after != cooldown_identity
            or evidence_after.identity != evidence.identity
            or recomputed_after != recomputed
            or _current_source_sha256() != source_sha_before
            or twen_source_tree_sha256() != source_tree_before
        ):
            raise DataAuditError("near-exclusion inputs/source changed during materialization")
        _output_attestation(manifest)
        os.replace(work, root)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return root / "corpus-manifest.json"


__all__ = [
    "PHASE_NEAR_EXCLUSION_ALGORITHM",
    "PHASE_NEAR_EXCLUSION_ATTESTATION_KIND",
    "PHASE_NEAR_EXCLUSION_SCHEMA_VERSION",
    "materialize_phase_near_excluded_cooldown",
    "validate_phase_near_exclusion_output",
]
