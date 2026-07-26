"""Authenticated quota refill for audited Base JSONL corpora.

The normal extractor deliberately stops when the immutable recipe quotas are
reached.  A governance audit can remove documents afterwards, so this module
extends a completed raw corpus *after* its last per-source cursor without
changing the recipe, source lock, profile, or original chunk fingerprint.

Refill is intentionally split into two immutable transactions:

* a SHA-bound plan derived from a complete audit rejection ledger and the
  attribution ledgers; and
* a new raw lineage which hard-links every original committed chunk and only
  appends newly streamed chunks.

No function in this module prepares KD data, constructs an optimizer, or runs
training.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from ..io.download import DownloadManager
from ..io.locking import FileLock
from ..io.proxy import ProxySettings
from ..progress import TaskProgress
from ..utils import atomic_write_json, sha256_file
from .audits import validate_base_audit_attestation
from .shards import ShardTransaction, is_shard_complete, read_complete_marker
from .sources import (
    EXTRACTED_CONTRACT_IDENTITY_KEYS,
    BaseDataRecipe,
    CorpusBuildStopped,
    DataSourceError,
    HfRangeFileFactory,
    ResolvedSource,
    ResolvedSourceLock,
    RowCursor,
    SourceProgress,
    SourceRecipe,
    _JsonlChunkWriter,
    _load_seen_hashes,
    _load_source_progress,
    _pipeline_fingerprint,
    _read_chunk,
    _reset_chunk_outputs,
    _source_fingerprint,
    _target_reached,
    _write_corpus_manifest,
    iter_local_jsonl_rows,
    iter_remote_parquet_rows,
    load_base_data_recipe,
    load_resolved_source_lock,
    materialize_jsonl_gzip_artifact,
    validate_extracted_base_corpus,
)

REFILL_PLAN_SCHEMA_VERSION = 1
REFILL_PLAN_KIND = "twen_base_corpus_refill_plan"
REFILL_PLAN_COMPLETE_KIND = "twen_base_corpus_refill_plan_complete"
REFILL_LINEAGE_SCHEMA_VERSION = 1
REFILL_LINEAGE_KIND = "twen_base_corpus_refill_lineage"
REFILL_HARDLINK_INVENTORY_KIND = "twen_base_corpus_refill_hardlink_inventory"
REFILL_NEW_CHUNK_INVENTORY_KIND = "twen_base_corpus_refill_new_chunk_inventory"
REFILL_FINALIZATION_STATE_KIND = "twen_base_corpus_refill_finalization_state"
DEFAULT_CLEAN_GUARD_RATIO = 0.02
DEFAULT_SURVIVAL_GUARD_POINTS = 0.01
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
REFILL_SOURCE_SHA256 = sha256_file(Path(__file__))


class RefillError(DataSourceError):
    """A refill input, plan, or immutable lineage failed authentication."""


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _pretty_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RefillError(f"invalid or missing {label}: {path}") from error
    if not isinstance(value, dict):
        raise RefillError(f"{label} must be a JSON object: {path}")
    return value


def _identity(path: Path, *, relative: str | None = None) -> dict[str, object]:
    return {
        "path": relative if relative is not None else str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _normalized_sha(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value.lower()):
        raise RefillError(f"{field} must be a 64-digit SHA256")
    return value.lower()


def _safe_relative(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RefillError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() == "." or ".." in path.parts:
        raise RefillError(f"unsafe {field}: {value!r}")
    return path.as_posix()


def _validated_manifest(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    manifest = Path(path).resolve()
    validate_extracted_base_corpus(manifest, verify_hashes=True)
    raw = manifest.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):  # pragma: no cover - authenticated above
        raise RefillError("extracted corpus manifest must be an object")
    return manifest, value, hashlib.sha256(raw).hexdigest()


def _source_mapping(value: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    raw_sources = value.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise RefillError("extracted corpus source summaries are missing")
    result: dict[str, Mapping[str, object]] = {}
    for raw in raw_sources:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("source_id"), str):
            raise RefillError("invalid extracted corpus source summary")
        source_id = str(raw["source_id"])
        if source_id in result:
            raise RefillError(f"duplicate extracted source summary: {source_id}")
        result[source_id] = raw
    return result


def _attribution_rows(
    manifest: Path,
    value: Mapping[str, object],
    *,
    role: str | None = None,
) -> Iterator[Mapping[str, object]]:
    raw_inventory = value.get("attribution_files")
    if not isinstance(raw_inventory, list) or not raw_inventory:
        raise RefillError(f"corpus has no attribution ledger: {manifest}")
    for index, raw in enumerate(raw_inventory):
        if not isinstance(raw, Mapping):
            raise RefillError(f"invalid attribution inventory entry {index}")
        relative = _safe_relative(raw.get("path"), f"attribution_files[{index}].path")
        path = manifest.parent / relative
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RefillError(
                        f"invalid attribution JSONL at {path}:{line_number}"
                    ) from error
                if not isinstance(row, Mapping):
                    raise RefillError(f"invalid attribution row at {path}:{line_number}")
                split = row.get("split")
                if split not in {"train", "validation"}:
                    raise RefillError(f"invalid attribution split at {path}:{line_number}")
                if role is None or split == role:
                    yield row


def _attribution_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    role = row.get("split")
    source_id = row.get("source_id")
    text_sha = row.get("text_sha256")
    if (
        role not in {"train", "validation"}
        or not isinstance(source_id, str)
        or not source_id
        or not isinstance(text_sha, str)
        or not _SHA256.fullmatch(text_sha.lower())
    ):
        raise RefillError("attribution row has invalid role/source/text identity")
    return str(role), source_id, text_sha.lower()


def _attribution_token_count(row: Mapping[str, object]) -> int:
    value = row.get("token_count_with_eos")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RefillError("attribution token_count_with_eos must be positive")
    return value


def _validation_digest(
    manifest: Path,
    value: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    digests: dict[str, Any] = {}
    counts: dict[str, int] = {}
    tokens: dict[str, int] = {}
    seen: set[tuple[str, str, str]] = set()
    for row in _attribution_rows(manifest, value, role="validation"):
        key = _attribution_key(row)
        if key in seen:
            raise RefillError(f"duplicate validation attribution identity: {key}")
        seen.add(key)
        source_id = key[1]
        token_count = _attribution_token_count(row)
        digest = digests.setdefault(source_id, hashlib.sha256())
        digest.update(
            json.dumps(
                [key[2], token_count],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        counts[source_id] = counts.get(source_id, 0) + 1
        tokens[source_id] = tokens.get(source_id, 0) + token_count
    return {
        source_id: {
            "documents": counts[source_id],
            "tokens": tokens[source_id],
            "ordered_identity_sha256": digest.hexdigest(),
        }
        for source_id, digest in sorted(digests.items())
    }


def corpus_tokens_by_source(manifest_path: str | Path) -> dict[str, object]:
    """Authenticate and independently total train/validation attribution tokens."""

    manifest, value, manifest_sha = _validated_manifest(manifest_path)
    sources = _source_mapping(value)
    totals = {
        source_id: {
            "train_tokens": 0,
            "validation_tokens": 0,
            "train_documents": 0,
            "validation_documents": 0,
        }
        for source_id in sources
    }
    for row in _attribution_rows(manifest, value):
        key = _attribution_key(row)
        if key[1] not in totals:
            raise RefillError(f"attribution references unknown source: {key[1]}")
        token_count = _attribution_token_count(row)
        totals[key[1]][f"{key[0]}_tokens"] += token_count
        totals[key[1]][f"{key[0]}_documents"] += 1
    train_total = sum(int(item["train_tokens"]) for item in totals.values())
    validation_total = sum(int(item["validation_tokens"]) for item in totals.values())
    manifest_train = value.get("actual_train_tokens")
    manifest_validation = value.get("actual_validation_tokens")
    if manifest_train is not None and int(manifest_train) != train_total:
        raise RefillError(
            f"manifest/attribution train token mismatch: {manifest_train} != {train_total}"
        )
    if manifest_validation is not None and int(manifest_validation) != validation_total:
        raise RefillError(
            "manifest/attribution validation token mismatch: "
            f"{manifest_validation} != {validation_total}"
        )
    return {
        "manifest": _identity(manifest),
        "manifest_sha256": manifest_sha,
        "corpus_fingerprint": value.get("corpus_fingerprint"),
        "sources": totals,
        "train_tokens": train_total,
        "validation_tokens": validation_total,
    }


def _rejection_documents(
    attestation_path: Path,
    attestation: Mapping[str, object],
) -> tuple[dict[tuple[str, str, str], dict[str, object]], dict[str, int]]:
    raw_identity = attestation.get("rejection_ledger")
    if not isinstance(raw_identity, Mapping):
        raise RefillError("attestation has no rejection ledger")
    relative = _safe_relative(raw_identity.get("path"), "rejection_ledger.path")
    ledger = attestation_path.parent / relative
    documents: dict[tuple[str, str, str], dict[str, object]] = {}
    event_counts: dict[str, int] = {}
    with ledger.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RefillError(
                    f"invalid rejection ledger JSONL at {ledger}:{line_number}"
                ) from error
            if not isinstance(value, Mapping) or not isinstance(value.get("document"), Mapping):
                raise RefillError(f"invalid rejection event at {ledger}:{line_number}")
            gate = value.get("gate")
            document = value["document"]
            role = document.get("role")
            source_id = document.get("source_id")
            text_sha = document.get("text_sha256")
            path = document.get("path")
            source_line = document.get("line")
            if (
                not isinstance(gate, str)
                or role not in {"train", "validation"}
                or not isinstance(source_id, str)
                or not source_id
                or not isinstance(text_sha, str)
                or not _SHA256.fullmatch(text_sha.lower())
                or not isinstance(path, str)
                or not path
                or isinstance(source_line, bool)
                or not isinstance(source_line, int)
                or source_line <= 0
            ):
                raise RefillError(f"invalid rejection document at {ledger}:{line_number}")
            key = (str(role), source_id, text_sha.lower())
            location = (path, source_line)
            existing = documents.get(key)
            if existing is None:
                documents[key] = {
                    "role": role,
                    "source_id": source_id,
                    "text_sha256": text_sha.lower(),
                    "path": path,
                    "line": source_line,
                    "gates": [gate],
                }
            else:
                if (existing["path"], existing["line"]) != location:
                    raise RefillError(
                        "one rejected content identity maps to multiple document locations"
                    )
                gates = existing["gates"]
                assert isinstance(gates, list)
                if gate not in gates:
                    gates.append(gate)
            event_counts[gate] = event_counts.get(gate, 0) + 1
    return documents, event_counts


def _resolve_rejected_tokens(
    *,
    documents: Mapping[tuple[str, str, str], Mapping[str, object]],
    candidate_manifest: Path,
    candidate_value: Mapping[str, object],
    frozen_manifest: Path,
    frozen_value: Mapping[str, object],
) -> dict[tuple[str, str, str], int]:
    unresolved = set(documents)
    result: dict[tuple[str, str, str], int] = {}
    for role, manifest, value in (
        ("train", candidate_manifest, candidate_value),
        ("validation", frozen_manifest, frozen_value),
    ):
        for row in _attribution_rows(manifest, value, role=role):
            key = _attribution_key(row)
            if key not in unresolved:
                continue
            token_count = _attribution_token_count(row)
            if key in result:
                raise RefillError(f"rejected attribution identity is ambiguous: {key}")
            result[key] = token_count
            unresolved.remove(key)
    if unresolved:
        preview = sorted(unresolved)[:3]
        raise RefillError(
            f"rejection ledger documents are absent from authenticated attribution: {preview}"
        )
    return result


def _chunk_identity_contract(
    manifest: Path,
    value: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for source_id, summary in _source_mapping(value).items():
        raw_chunks = summary.get("chunks")
        if not isinstance(raw_chunks, list) or not raw_chunks:
            raise RefillError(f"raw source has no committed chunks: {source_id}")
        fingerprints: set[str] = set()
        source_fingerprints: set[str] = set()
        chunk_directories: list[str] = []
        for index, raw_chunk in enumerate(raw_chunks):
            if not isinstance(raw_chunk, Mapping) or not isinstance(raw_chunk.get("outputs"), list):
                raise RefillError(f"invalid raw chunk summary: {source_id}[{index}]")
            output_paths = [
                str(item.get("path"))
                for item in raw_chunk["outputs"]
                if isinstance(item, Mapping) and isinstance(item.get("path"), str)
            ]
            if not output_paths:
                raise RefillError(f"raw chunk has no outputs: {source_id}[{index}]")
            parents = {PurePosixPath(path).parent.as_posix() for path in output_paths}
            if len(parents) != 1:
                raise RefillError(f"raw chunk outputs span directories: {source_id}[{index}]")
            relative_directory = next(iter(parents))
            directory = manifest.parent / relative_directory
            if not is_shard_complete(directory, verify_hashes=True):
                raise RefillError(f"raw chunk is not complete: {directory}")
            marker = read_complete_marker(directory)
            fingerprint = _normalized_sha(marker.get("fingerprint"), "chunk.fingerprint")
            source_fingerprint = _normalized_sha(
                marker.get("source_fingerprint"), "chunk.source_fingerprint"
            )
            fingerprints.add(fingerprint)
            source_fingerprints.add(source_fingerprint)
            chunk_directories.append(relative_directory)
        if len(fingerprints) != 1 or len(source_fingerprints) != 1:
            raise RefillError(f"raw chunks have mixed fingerprints: {source_id}")
        next_file = summary.get("next_file_index")
        next_row = summary.get("next_row_index")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in (next_file, next_row)
        ):
            raise RefillError(f"raw source cursor is invalid: {source_id}")
        result[source_id] = {
            "chunk_count": len(raw_chunks),
            "chunk_directories": chunk_directories,
            "pipeline_fingerprint": next(iter(fingerprints)),
            "source_fingerprint": next(iter(source_fingerprints)),
            "cursor": {"file_index": next_file, "row_index": next_row},
        }
    return result


def _guarded_target(
    *,
    role: str,
    quota: int,
    observed_raw: int,
    observed_clean: int,
    clean_guard_ratio: float,
    survival_guard_points: float,
    zero_clean_fallback: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if quota <= 0 or observed_raw <= 0 or not 0 <= observed_clean <= observed_raw:
        raise RefillError("invalid quota/raw/clean token accounting")
    if role not in {"train", "validation"}:
        raise RefillError(f"invalid refill target role: {role!r}")
    observed_survival = observed_clean / observed_raw
    clean_target_with_guard = math.ceil(quota * (1.0 + clean_guard_ratio))
    if observed_clean == 0:
        if not isinstance(zero_clean_fallback, Mapping):
            raise RefillError(
                "observed survival is zero and no same-source fallback evidence was provided"
            )
        fallback_role = zero_clean_fallback.get("role")
        fallback_raw = zero_clean_fallback.get("observed_raw_tokens")
        fallback_clean = zero_clean_fallback.get("observed_clean_tokens")
        if (
            fallback_role not in {"train", "validation"}
            or fallback_role == role
            or isinstance(fallback_raw, bool)
            or not isinstance(fallback_raw, int)
            or fallback_raw <= 0
            or isinstance(fallback_clean, bool)
            or not isinstance(fallback_clean, int)
            or not 0 < fallback_clean <= fallback_raw
        ):
            raise RefillError("invalid zero-clean fallback survival evidence")
        planning_survival = fallback_clean / fallback_raw
        guarded_survival = planning_survival - survival_guard_points
        if guarded_survival <= 0:
            raise RefillError(
                "fallback survival is too low for the requested survival guard: "
                f"{planning_survival:.8f} - {survival_guard_points:.8f}"
            )
        additional_raw = math.ceil(clean_target_with_guard / guarded_survival)
        runtime_target = observed_raw + additional_raw
        survival_evidence: dict[str, object] = {
            "mode": "same-source-role-fallback-for-zero-clean",
            "target_role": role,
            "evidence_role": fallback_role,
            "observed_raw_tokens": fallback_raw,
            "observed_clean_tokens": fallback_clean,
            "observed_survival": planning_survival,
        }
        formula = (
            "runtime_raw_target=observed_raw+ceil("
            "ceil(quota*(1+clean_guard_ratio))/"
            "(fallback_observed_survival-survival_guard_points))"
        )
    else:
        planning_survival = observed_survival
        guarded_survival = planning_survival - survival_guard_points
        if guarded_survival <= 0:
            raise RefillError(
                "observed survival is too low for the requested survival guard: "
                f"{planning_survival:.8f} - {survival_guard_points:.8f}"
            )
        runtime_target = max(
            observed_raw,
            math.ceil(clean_target_with_guard / guarded_survival),
        )
        additional_raw = runtime_target - observed_raw
        survival_evidence = {
            "mode": "target-role-observed",
            "target_role": role,
            "evidence_role": role,
            "observed_raw_tokens": observed_raw,
            "observed_clean_tokens": observed_clean,
            "observed_survival": observed_survival,
        }
        formula = (
            "runtime_raw_target=max(observed_raw,ceil("
            "ceil(quota*(1+clean_guard_ratio))/(observed_survival-survival_guard_points)))"
        )
    return {
        "role": role,
        "quota_tokens": quota,
        "observed_raw_tokens": observed_raw,
        "observed_clean_tokens": observed_clean,
        "observed_rejected_tokens": observed_raw - observed_clean,
        "observed_survival": observed_survival,
        "survival_evidence": survival_evidence,
        "planning_survival": planning_survival,
        "clean_guard_ratio": clean_guard_ratio,
        "clean_target_with_guard_tokens": clean_target_with_guard,
        "survival_guard_points": survival_guard_points,
        "guarded_survival": guarded_survival,
        "runtime_raw_target_tokens": runtime_target,
        "additional_raw_tokens": additional_raw,
        "formula": formula,
    }


def _derive_refill_semantics(
    *,
    recipe: BaseDataRecipe,
    profile: str,
    base_manifest: Path,
    base_value: Mapping[str, object],
    clean_accounting: Mapping[str, object],
    attestation_path: Path,
    attestation: Mapping[str, object],
    frozen_manifest: Path,
    frozen_value: Mapping[str, object],
    clean_guard_ratio: float,
    survival_guard_points: float,
) -> dict[str, object]:
    recipe_sources = {source.source_id: source for source in recipe.sources}
    base_sources = _source_mapping(base_value)
    clean_sources = clean_accounting.get("sources")
    if not isinstance(clean_sources, Mapping):
        raise RefillError("materialized token accounting has no sources")
    if set(recipe_sources) != set(base_sources) or set(recipe_sources) != set(clean_sources):
        raise RefillError("recipe/base/materialized source sets differ")

    documents, event_counts = _rejection_documents(attestation_path, attestation)
    rejected_tokens = _resolve_rejected_tokens(
        documents=documents,
        candidate_manifest=base_manifest,
        candidate_value=base_value,
        frozen_manifest=frozen_manifest,
        frozen_value=frozen_value,
    )
    rejected_by_source: dict[str, dict[str, int]] = {
        source_id: {
            "train_documents": 0,
            "train_tokens": 0,
            "validation_documents": 0,
            "validation_tokens": 0,
        }
        for source_id in recipe_sources
    }
    for key, token_count in rejected_tokens.items():
        role, source_id, _ = key
        if source_id not in rejected_by_source:
            raise RefillError(f"rejection references source outside recipe: {source_id}")
        rejected_by_source[source_id][f"{role}_documents"] += 1
        rejected_by_source[source_id][f"{role}_tokens"] += token_count

    chunk_contract = _chunk_identity_contract(base_manifest, base_value)
    planned_sources: list[dict[str, object]] = []
    runtime_train_total = 0
    runtime_validation_total = 0
    for source in recipe.sources:
        raw_summary = base_sources[source.source_id]
        clean_summary = clean_sources[source.source_id]
        if not isinstance(clean_summary, Mapping):
            raise RefillError(
                f"materialized source token accounting is invalid: {source.source_id}"
            )
        raw_train = int(raw_summary.get("actual_train_tokens", 0))
        raw_validation = int(raw_summary.get("actual_validation_tokens", 0))
        clean_train = int(clean_summary.get("train_tokens", 0))
        clean_validation = int(clean_summary.get("validation_tokens", 0))
        unique = rejected_by_source[source.source_id]
        if raw_train - unique["train_tokens"] != clean_train:
            raise RefillError(
                f"{source.source_id} train materialization does not equal raw minus unique "
                "rejection tokens"
            )
        if raw_validation - unique["validation_tokens"] != clean_validation:
            raise RefillError(
                f"{source.source_id} validation materialization does not equal raw minus "
                "unique rejection tokens"
            )
        train_plan = _guarded_target(
            role="train",
            quota=int(source.train_token_quotas[profile]),
            observed_raw=raw_train,
            observed_clean=clean_train,
            clean_guard_ratio=clean_guard_ratio,
            survival_guard_points=survival_guard_points,
        )
        validation_plan = _guarded_target(
            role="validation",
            quota=source.validation_token_quota,
            observed_raw=raw_validation,
            observed_clean=clean_validation,
            clean_guard_ratio=clean_guard_ratio,
            survival_guard_points=survival_guard_points,
            zero_clean_fallback={
                "role": "train",
                "observed_raw_tokens": raw_train,
                "observed_clean_tokens": clean_train,
            },
        )
        runtime_train_total += int(train_plan["runtime_raw_target_tokens"])
        runtime_validation_total += int(validation_plan["runtime_raw_target_tokens"])
        planned_sources.append(
            {
                "source_id": source.source_id,
                "cursor": chunk_contract[source.source_id]["cursor"],
                "original_chunk_count": chunk_contract[source.source_id]["chunk_count"],
                "original_pipeline_fingerprint": chunk_contract[source.source_id][
                    "pipeline_fingerprint"
                ],
                "original_source_fingerprint": chunk_contract[source.source_id][
                    "source_fingerprint"
                ],
                "unique_rejections": unique,
                "train": train_plan,
                "validation": validation_plan,
            }
        )
    return {
        "rejection_event_counts": event_counts,
        "unique_rejection_documents": len(documents),
        "sources": planned_sources,
        "runtime_targets": {
            "train_tokens": runtime_train_total,
            "validation_tokens": runtime_validation_total,
        },
        "original_quotas": {
            "train_tokens": recipe.profiles[profile],
            "validation_tokens": recipe.validation_tokens,
        },
    }


def _validate_materialized_projection(
    *,
    materialized_value: Mapping[str, object],
    attestation_path: Path,
    attestation: Mapping[str, object],
    base_manifest_sha256: str,
    frozen_manifest_sha256: str,
) -> None:
    expected_projection = {
        "method": "complete-audit-rejection-ledger-projection-v1",
        "parent_candidate_manifest_sha256": base_manifest_sha256,
        "parent_frozen_validation_manifest_sha256": frozen_manifest_sha256,
        "audit_attestation_sha256": sha256_file(attestation_path),
        "audit_attestation_fingerprint": attestation.get("attestation_fingerprint"),
        "rejection_ledger": attestation.get("rejection_ledger"),
    }
    for audit_name in ("format_audit", "license_audit"):
        audit = materialized_value.get(audit_name)
        if not isinstance(audit, Mapping) or audit.get("projection") != expected_projection:
            raise RefillError(
                f"materialized corpus {audit_name} does not bind the refill audit projection"
            )
    materialization = materialized_value.get("materialization_audit")
    if not isinstance(materialization, Mapping):
        raise RefillError("materialized corpus has no authenticated materialization audit")
    for field, expected in expected_projection.items():
        if materialization.get(field) != expected:
            raise RefillError(
                "materialized corpus materialization audit does not bind the refill "
                f"projection field: {field}"
            )
    if (
        materialization.get("complete") is not True
        or materialization.get("network_policy") != "offline-audit-materialization"
    ):
        raise RefillError("materialized corpus audit projection is incomplete")


def create_refill_plan(
    *,
    audit_attestation_path: str | Path,
    base_raw_manifest_path: str | Path,
    materialized_manifest_path: str | Path,
    recipe_path: str | Path,
    output_root: str | Path,
    clean_guard_ratio: float = DEFAULT_CLEAN_GUARD_RATIO,
    survival_guard_points: float = DEFAULT_SURVIVAL_GUARD_POINTS,
) -> Path:
    """Create a non-overwriting, SHA-authenticated per-source refill plan."""

    if not math.isfinite(clean_guard_ratio) or clean_guard_ratio < 0.02:
        raise RefillError("clean_guard_ratio must be finite and at least 0.02")
    if not math.isfinite(survival_guard_points) or survival_guard_points < 0.01:
        raise RefillError("survival_guard_points must be finite and at least 0.01")
    if survival_guard_points >= 1.0:
        raise RefillError("survival_guard_points must be below 1")

    attestation_path = Path(audit_attestation_path).resolve()
    attestation = dict(validate_base_audit_attestation(attestation_path))
    base_manifest, base_value, base_sha = _validated_manifest(base_raw_manifest_path)
    clean_accounting = corpus_tokens_by_source(materialized_manifest_path)
    materialized_manifest = Path(materialized_manifest_path).resolve()
    materialized_value = _load_object(materialized_manifest, "materialized manifest")
    candidate_identity = attestation.get("candidate")
    frozen_identity = attestation.get("frozen_validation")
    if not isinstance(candidate_identity, Mapping) or not isinstance(frozen_identity, Mapping):
        raise RefillError("audit attestation corpus identities are invalid")
    if (
        Path(str(candidate_identity.get("manifest_path"))).resolve() != base_manifest
        or candidate_identity.get("manifest_sha256") != base_sha
    ):
        raise RefillError("audit attestation does not bind the base raw manifest")
    frozen_manifest, frozen_value, frozen_sha = _validated_manifest(
        str(frozen_identity.get("manifest_path"))
    )
    if frozen_identity.get("manifest_sha256") != frozen_sha:
        raise RefillError("audit frozen validation manifest SHA mismatch")
    if base_value.get("tokenizer_manifest_sha256") != frozen_value.get("tokenizer_manifest_sha256"):
        raise RefillError("base/frozen tokenizer identities differ")
    if base_value.get("recipe_sha256") != frozen_value.get("recipe_sha256"):
        raise RefillError("base/frozen recipe identities differ")
    base_validation_digest = _validation_digest(base_manifest, base_value)
    frozen_validation_digest = _validation_digest(frozen_manifest, frozen_value)
    if base_validation_digest != frozen_validation_digest:
        raise RefillError(
            "base raw validation is not attribution-identical to the audit frozen validation"
        )

    recipe = load_base_data_recipe(recipe_path)
    if base_value.get("recipe_sha256") != recipe.sha256:
        raise RefillError("base raw manifest does not bind the refill recipe bytes")
    profile = base_value.get("profile")
    if profile not in recipe.profiles:
        raise RefillError(f"base raw profile is not refillable: {profile!r}")
    if materialized_value.get("recipe_sha256") != recipe.sha256 or materialized_value.get(
        "tokenizer_manifest_sha256"
    ) != base_value.get("tokenizer_manifest_sha256"):
        raise RefillError("materialized corpus lineage differs from base raw corpus")
    _validate_materialized_projection(
        materialized_value=materialized_value,
        attestation_path=attestation_path,
        attestation=attestation,
        base_manifest_sha256=base_sha,
        frozen_manifest_sha256=frozen_sha,
    )
    semantics = _derive_refill_semantics(
        recipe=recipe,
        profile=str(profile),
        base_manifest=base_manifest,
        base_value=base_value,
        clean_accounting=clean_accounting,
        attestation_path=attestation_path,
        attestation=attestation,
        frozen_manifest=frozen_manifest,
        frozen_value=frozen_value,
        clean_guard_ratio=clean_guard_ratio,
        survival_guard_points=survival_guard_points,
    )

    attestation_identity = _identity(attestation_path)
    base_identity = _identity(base_manifest)
    materialized_identity = _identity(materialized_manifest)
    payload: dict[str, object] = {
        "schema_version": REFILL_PLAN_SCHEMA_VERSION,
        "kind": REFILL_PLAN_KIND,
        "refill_source_sha256": REFILL_SOURCE_SHA256,
        "policy": {
            "clean_guard_ratio": clean_guard_ratio,
            "minimum_clean_guard_ratio": DEFAULT_CLEAN_GUARD_RATIO,
            "survival_guard_points": survival_guard_points,
            "minimum_survival_guard_points": DEFAULT_SURVIVAL_GUARD_POINTS,
            "rejection_accounting": "unique-(role,source_id,text_sha256)-via-attribution-v1",
            "rejected_documents_are_not_counted_as_clean": True,
        },
        "audit_attestation": attestation_identity,
        "a0_attestation_sha256": attestation_identity["sha256"],
        "rejection_ledger": dict(attestation["rejection_ledger"]),
        "rejection_event_counts": semantics["rejection_event_counts"],
        "unique_rejection_documents": semantics["unique_rejection_documents"],
        "base_raw_manifest": base_identity,
        "original_raw_manifest_sha256": base_identity["sha256"],
        "base_raw_corpus_fingerprint": base_value.get("corpus_fingerprint"),
        "materialized_manifest": materialized_identity,
        "frozen_validation_manifest": _identity(frozen_manifest),
        "base_frozen_validation_equivalence": {
            "passed": True,
            "algorithm": "ordered-(text_sha256,token_count)-per-source-v1",
            "sources": base_validation_digest,
        },
        "recipe": _identity(Path(recipe_path).resolve()),
        "recipe_id": recipe.recipe_id,
        "recipe_sha256": recipe.sha256,
        "resolved_source_lock_sha256": base_value.get("resolved_source_lock_sha256"),
        "tokenizer_manifest_sha256": base_value.get("tokenizer_manifest_sha256"),
        "profile": profile,
        "sources": semantics["sources"],
        "runtime_targets": semantics["runtime_targets"],
        "original_quotas": semantics["original_quotas"],
        "network_policy": "fallback (Hugging Face direct first, configured proxy second)",
        "training_started": False,
        "gpu_kd_started": False,
    }
    payload["plan_fingerprint"] = _canonical_sha256(payload)
    root = Path(output_root).resolve()
    if root.exists():
        raise RefillError(f"refill plan output already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{root.name}.incomplete-", dir=root.parent))
    try:
        plan = work / "plan.json"
        atomic_write_json(plan, payload)
        atomic_write_json(
            work / "COMPLETE",
            {
                "schema_version": REFILL_PLAN_SCHEMA_VERSION,
                "kind": REFILL_PLAN_COMPLETE_KIND,
                "plan": plan.name,
                "plan_sha256": sha256_file(plan),
                "plan_fingerprint": payload["plan_fingerprint"],
                "audit_attestation_sha256": attestation_identity["sha256"],
                "base_raw_manifest_sha256": base_identity["sha256"],
                "materialized_manifest_sha256": materialized_identity["sha256"],
                "frozen_validation_manifest_sha256": frozen_sha,
                "training_started": False,
                "gpu_kd_started": False,
            },
        )
        os.replace(work, root)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return root / "plan.json"


def validate_refill_plan(path: str | Path) -> dict[str, Any]:
    plan = Path(path).resolve()
    value = _load_object(plan, "refill plan")
    if (
        value.get("schema_version") != REFILL_PLAN_SCHEMA_VERSION
        or value.get("kind") != REFILL_PLAN_KIND
    ):
        raise RefillError("unsupported refill plan schema/kind")
    if value.get("refill_source_sha256") != REFILL_SOURCE_SHA256:
        raise RefillError("refill implementation changed; regenerate the plan")
    fingerprint = _normalized_sha(value.get("plan_fingerprint"), "plan_fingerprint")
    fingerprint_payload = dict(value)
    fingerprint_payload.pop("plan_fingerprint")
    if _canonical_sha256(fingerprint_payload) != fingerprint:
        raise RefillError("refill plan fingerprint mismatch")
    complete = _load_object(plan.parent / "COMPLETE", "refill plan COMPLETE")
    if (
        complete.get("schema_version") != REFILL_PLAN_SCHEMA_VERSION
        or complete.get("kind") != REFILL_PLAN_COMPLETE_KIND
        or complete.get("plan") != plan.name
        or complete.get("plan_sha256") != sha256_file(plan)
        or complete.get("plan_fingerprint") != fingerprint
        or complete.get("training_started") is not False
        or complete.get("gpu_kd_started") is not False
    ):
        raise RefillError("refill plan COMPLETE binding mismatch")
    for name, complete_name in (
        ("audit_attestation", "audit_attestation_sha256"),
        ("base_raw_manifest", "base_raw_manifest_sha256"),
        ("materialized_manifest", "materialized_manifest_sha256"),
        ("frozen_validation_manifest", "frozen_validation_manifest_sha256"),
    ):
        identity = value.get(name)
        if not isinstance(identity, Mapping):
            raise RefillError(f"refill plan {name} identity is invalid")
        source = Path(str(identity.get("path"))).resolve()
        expected = _normalized_sha(identity.get("sha256"), f"{name}.sha256")
        if not source.is_file() or source.stat().st_size != identity.get("size"):
            raise RefillError(f"refill plan {name} is missing/size-mismatched")
        if sha256_file(source) != expected or complete.get(complete_name) != expected:
            raise RefillError(f"refill plan {name} SHA binding mismatch")
        if name == "audit_attestation":
            validate_base_audit_attestation(source)
        else:
            _validated_manifest(source)
    audit_identity = value["audit_attestation"]
    base_identity = value["base_raw_manifest"]
    materialized_identity = value["materialized_manifest"]
    frozen_identity = value["frozen_validation_manifest"]
    assert isinstance(audit_identity, Mapping)
    assert isinstance(base_identity, Mapping)
    assert isinstance(materialized_identity, Mapping)
    assert isinstance(frozen_identity, Mapping)
    recipe_identity = value.get("recipe")
    if not isinstance(recipe_identity, Mapping):
        raise RefillError("refill plan recipe identity is invalid")
    recipe_path = Path(str(recipe_identity.get("path"))).resolve()
    recipe_sha = _normalized_sha(recipe_identity.get("sha256"), "recipe.sha256")
    if (
        not recipe_path.is_file()
        or recipe_path.stat().st_size != recipe_identity.get("size")
        or sha256_file(recipe_path) != recipe_sha
    ):
        raise RefillError("refill plan recipe is missing or identity-mismatched")
    recipe = load_base_data_recipe(recipe_path)
    attestation = validate_base_audit_attestation(str(audit_identity["path"]))
    attestation_candidate = attestation.get("candidate")
    attestation_frozen = attestation.get("frozen_validation")
    if not isinstance(attestation_candidate, Mapping) or not isinstance(
        attestation_frozen, Mapping
    ):
        raise RefillError("audit attestation corpus identities are invalid")
    if Path(str(attestation_candidate.get("manifest_path"))).resolve() != Path(
        str(base_identity.get("path"))
    ).resolve() or attestation_candidate.get("manifest_sha256") != base_identity.get("sha256"):
        raise RefillError("audit candidate identity differs from refill plan base raw")
    if Path(str(attestation_frozen.get("manifest_path"))).resolve() != Path(
        str(frozen_identity.get("path"))
    ).resolve() or attestation_frozen.get("manifest_sha256") != frozen_identity.get("sha256"):
        raise RefillError("audit frozen validation identity differs from refill plan")
    base_manifest, base_value, _ = _validated_manifest(str(base_identity["path"]))
    materialized_manifest, materialized_value, _ = _validated_manifest(
        str(materialized_identity["path"])
    )
    frozen_manifest, frozen_value, _ = _validated_manifest(str(frozen_identity["path"]))
    base_digest = _validation_digest(base_manifest, base_value)
    frozen_digest = _validation_digest(frozen_manifest, frozen_value)
    equivalence = value.get("base_frozen_validation_equivalence")
    if (
        base_digest != frozen_digest
        or not isinstance(equivalence, Mapping)
        or equivalence.get("passed") is not True
        or equivalence.get("algorithm") != "ordered-(text_sha256,token_count)-per-source-v1"
        or equivalence.get("sources") != base_digest
    ):
        raise RefillError("base/frozen validation ordered attribution digest mismatch")
    policy = value.get("policy")
    if not isinstance(policy, Mapping):
        raise RefillError("refill plan policy is invalid")
    raw_clean_guard = policy.get("clean_guard_ratio")
    raw_survival_guard = policy.get("survival_guard_points")
    if (
        isinstance(raw_clean_guard, bool)
        or not isinstance(raw_clean_guard, (int, float))
        or not math.isfinite(raw_clean_guard)
    ):
        raise RefillError("refill plan clean guard is invalid")
    if (
        isinstance(raw_survival_guard, bool)
        or not isinstance(raw_survival_guard, (int, float))
        or not math.isfinite(raw_survival_guard)
    ):
        raise RefillError("refill plan survival guard is invalid")
    clean_guard_ratio = float(raw_clean_guard)
    survival_guard_points = float(raw_survival_guard)
    if clean_guard_ratio < DEFAULT_CLEAN_GUARD_RATIO:
        raise RefillError("refill plan clean guard is below the policy minimum")
    if survival_guard_points < DEFAULT_SURVIVAL_GUARD_POINTS:
        raise RefillError("refill plan survival guard is below the policy minimum")
    if survival_guard_points >= 1.0:
        raise RefillError("refill plan survival guard must be below one")
    expected_policy = {
        "clean_guard_ratio": clean_guard_ratio,
        "minimum_clean_guard_ratio": DEFAULT_CLEAN_GUARD_RATIO,
        "survival_guard_points": survival_guard_points,
        "minimum_survival_guard_points": DEFAULT_SURVIVAL_GUARD_POINTS,
        "rejection_accounting": "unique-(role,source_id,text_sha256)-via-attribution-v1",
        "rejected_documents_are_not_counted_as_clean": True,
    }
    if dict(policy) != expected_policy:
        raise RefillError("refill plan policy semantics changed")

    profile = base_value.get("profile")
    if not isinstance(profile, str) or profile not in recipe.profiles:
        raise RefillError("base raw profile is not refillable")
    if (
        value.get("a0_attestation_sha256") != audit_identity.get("sha256")
        or value.get("original_raw_manifest_sha256") != base_identity.get("sha256")
        or value.get("base_raw_corpus_fingerprint") != base_value.get("corpus_fingerprint")
        or value.get("recipe_id") != recipe.recipe_id
        or value.get("recipe_sha256") != recipe.sha256
        or recipe_sha != recipe.sha256
        or value.get("resolved_source_lock_sha256") != base_value.get("resolved_source_lock_sha256")
        or value.get("tokenizer_manifest_sha256") != base_value.get("tokenizer_manifest_sha256")
        or value.get("profile") != profile
        or value.get("rejection_ledger") != attestation.get("rejection_ledger")
        or value.get("network_policy")
        != "fallback (Hugging Face direct first, configured proxy second)"
        or value.get("training_started") is not False
        or value.get("gpu_kd_started") is not False
    ):
        raise RefillError("refill plan authenticated lineage semantics changed")
    if (
        base_value.get("recipe_sha256") != recipe.sha256
        or frozen_value.get("recipe_sha256") != recipe.sha256
        or materialized_value.get("recipe_sha256") != recipe.sha256
        or frozen_value.get("tokenizer_manifest_sha256")
        != base_value.get("tokenizer_manifest_sha256")
        or materialized_value.get("tokenizer_manifest_sha256")
        != base_value.get("tokenizer_manifest_sha256")
    ):
        raise RefillError("refill plan corpus lineage differs from recipe/base raw corpus")
    _validate_materialized_projection(
        materialized_value=materialized_value,
        attestation_path=Path(str(audit_identity["path"])).resolve(),
        attestation=attestation,
        base_manifest_sha256=str(base_identity["sha256"]),
        frozen_manifest_sha256=str(frozen_identity["sha256"]),
    )

    clean_accounting = corpus_tokens_by_source(materialized_manifest)
    semantics = _derive_refill_semantics(
        recipe=recipe,
        profile=profile,
        base_manifest=base_manifest,
        base_value=base_value,
        clean_accounting=clean_accounting,
        attestation_path=Path(str(audit_identity["path"])).resolve(),
        attestation=attestation,
        frozen_manifest=frozen_manifest,
        frozen_value=frozen_value,
        clean_guard_ratio=clean_guard_ratio,
        survival_guard_points=survival_guard_points,
    )
    for field in (
        "rejection_event_counts",
        "unique_rejection_documents",
        "sources",
        "runtime_targets",
        "original_quotas",
    ):
        if value.get(field) != semantics[field]:
            raise RefillError(f"refill plan derived {field} semantics changed")
    return value


def _hardlink_original_chunks(
    *,
    plan_path: Path,
    plan: Mapping[str, object],
    output_root: Path,
) -> dict[str, object]:
    base_identity = plan.get("base_raw_manifest")
    if not isinstance(base_identity, Mapping):
        raise RefillError("plan base raw identity is invalid")
    base_manifest, base_value, _ = _validated_manifest(str(base_identity.get("path")))
    source_contracts = _chunk_identity_contract(base_manifest, base_value)
    raw_plan_sources = plan.get("sources")
    if not isinstance(raw_plan_sources, list):
        raise RefillError("plan source targets are invalid")
    plan_sources = {
        str(item.get("source_id")): item
        for item in raw_plan_sources
        if isinstance(item, Mapping) and isinstance(item.get("source_id"), str)
    }
    if set(plan_sources) != set(source_contracts):
        raise RefillError("plan/base raw source sets differ")

    linked_files: list[dict[str, object]] = []
    for source_id in sorted(source_contracts):
        contract = source_contracts[source_id]
        planned = plan_sources[source_id]
        if (
            planned.get("original_chunk_count") != contract["chunk_count"]
            or planned.get("original_pipeline_fingerprint") != contract["pipeline_fingerprint"]
            or planned.get("original_source_fingerprint") != contract["source_fingerprint"]
            or planned.get("cursor") != contract["cursor"]
        ):
            raise RefillError(f"plan raw chunk/cursor contract changed: {source_id}")
        directories = contract["chunk_directories"]
        assert isinstance(directories, list)
        for relative_directory in directories:
            source_directory = base_manifest.parent / str(relative_directory)
            linked_directory = output_root / str(relative_directory)
            linked_directory.mkdir(parents=True, exist_ok=False)
            for source_file in sorted(source_directory.iterdir()):
                if not source_file.is_file():
                    raise RefillError(f"unexpected non-file in committed chunk: {source_file}")
                linked_file = linked_directory / source_file.name
                os.link(source_file, linked_file)
                source_stat = source_file.stat()
                linked_stat = linked_file.stat()
                if (
                    source_stat.st_dev != linked_stat.st_dev
                    or source_stat.st_ino != linked_stat.st_ino
                ):
                    raise RefillError(f"refill chunk was not hard-linked: {linked_file}")
                linked_files.append(
                    {
                        "source_path": str(source_file.resolve()),
                        "path": linked_file.relative_to(output_root).as_posix(),
                        "size": linked_stat.st_size,
                        "sha256": sha256_file(linked_file),
                        "device": linked_stat.st_dev,
                        "inode": linked_stat.st_ino,
                    }
                )
    inventory: dict[str, object] = {
        "schema_version": REFILL_LINEAGE_SCHEMA_VERSION,
        "kind": REFILL_HARDLINK_INVENTORY_KIND,
        "plan": _identity(plan_path),
        "base_raw_manifest": _identity(base_manifest),
        "files": linked_files,
        "file_count": len(linked_files),
        "all_share_inode_with_original": True,
    }
    inventory["inventory_fingerprint"] = _canonical_sha256(inventory)
    atomic_write_json(output_root / "hardlink-inventory.json", inventory)
    return inventory


def _validate_hardlink_inventory(root: Path, plan_path: Path) -> dict[str, Any]:
    inventory_path = root / "hardlink-inventory.json"
    value = _load_object(inventory_path, "hardlink inventory")
    if (
        value.get("schema_version") != REFILL_LINEAGE_SCHEMA_VERSION
        or value.get("kind") != REFILL_HARDLINK_INVENTORY_KIND
    ):
        raise RefillError("unsupported hardlink inventory")
    fingerprint = _normalized_sha(
        value.get("inventory_fingerprint"), "hardlink.inventory_fingerprint"
    )
    payload = dict(value)
    payload.pop("inventory_fingerprint")
    if _canonical_sha256(payload) != fingerprint:
        raise RefillError("hardlink inventory fingerprint mismatch")
    plan_identity = value.get("plan")
    if not isinstance(plan_identity, Mapping) or (
        Path(str(plan_identity.get("path"))).resolve() != plan_path
        or plan_identity.get("sha256") != sha256_file(plan_path)
    ):
        raise RefillError("hardlink inventory plan binding mismatch")
    files = value.get("files")
    if not isinstance(files, list) or value.get("file_count") != len(files):
        raise RefillError("hardlink inventory file count is invalid")
    for raw in files:
        if not isinstance(raw, Mapping):
            raise RefillError("hardlink inventory entry is invalid")
        relative = _safe_relative(raw.get("path"), "hardlink.path")
        linked = root / relative
        original = Path(str(raw.get("source_path"))).resolve()
        if not linked.is_file() or not original.is_file():
            raise RefillError(f"hardlink inventory file is missing: {relative}")
        linked_stat = linked.stat()
        original_stat = original.stat()
        if (
            linked_stat.st_size != raw.get("size")
            or sha256_file(linked) != raw.get("sha256")
            or linked_stat.st_dev != original_stat.st_dev
            or linked_stat.st_ino != original_stat.st_ino
            or linked_stat.st_dev != raw.get("device")
            or linked_stat.st_ino != raw.get("inode")
        ):
            raise RefillError(f"hardlink identity changed: {relative}")
    return value


def _initialize_lineage(root: Path, plan_path: Path, plan: Mapping[str, object]) -> None:
    if root.exists():
        lineage = _load_object(root / "REFILL_LINEAGE.json", "refill lineage marker")
        if (
            lineage.get("schema_version") != REFILL_LINEAGE_SCHEMA_VERSION
            or lineage.get("kind") != REFILL_LINEAGE_KIND
            or lineage.get("plan_sha256") != sha256_file(plan_path)
            or Path(str(lineage.get("plan_path"))).resolve() != plan_path
        ):
            raise RefillError(f"existing refill output belongs to another plan: {root}")
        hardlink_identity = lineage.get("hardlink_inventory")
        hardlink_path = root / "hardlink-inventory.json"
        if (
            not isinstance(hardlink_identity, Mapping)
            or hardlink_identity.get("path") != hardlink_path.name
            or hardlink_identity.get("size") != hardlink_path.stat().st_size
            or hardlink_identity.get("sha256") != sha256_file(hardlink_path)
        ):
            raise RefillError("existing refill hardlink inventory binding changed")
        _validate_hardlink_inventory(root, plan_path)
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{root.name}.initialize-", dir=root.parent))
    try:
        inventory = _hardlink_original_chunks(
            plan_path=plan_path,
            plan=plan,
            output_root=work,
        )
        inventory_path = work / "hardlink-inventory.json"
        atomic_write_json(
            work / "REFILL_LINEAGE.json",
            {
                "schema_version": REFILL_LINEAGE_SCHEMA_VERSION,
                "kind": REFILL_LINEAGE_KIND,
                "plan_path": str(plan_path),
                "plan_sha256": sha256_file(plan_path),
                "plan_fingerprint": plan.get("plan_fingerprint"),
                "base_raw_manifest": plan.get("base_raw_manifest"),
                "hardlink_inventory": _identity(
                    inventory_path,
                    relative="hardlink-inventory.json",
                ),
                "hardlink_inventory_fingerprint": inventory["inventory_fingerprint"],
                "immutable_original_chunks": True,
                "rejected_documents_are_only_reaudited_never_counted_as_clean": True,
                "training_started": False,
                "gpu_kd_started": False,
            },
        )
        os.replace(work, root)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise


def _build_source_to_runtime_targets(
    *,
    source: SourceRecipe,
    resolved: ResolvedSource,
    recipe: BaseDataRecipe,
    resolved_lock: ResolvedSourceLock,
    profile: str,
    tokenizer_sha256: str,
    tokenizer: Any,
    output_root: Path,
    row_iterator: Any,
    seen_hashes: set[str],
    progress_bar: TaskProgress,
    stop_file: Path | None,
    train_target: int,
    validation_target: int,
    expected_pipeline_fingerprint: str,
    expected_source_fingerprint: str,
) -> tuple[SourceProgress, list[dict[str, object]]]:
    source_root = output_root / "extracted" / source.source_id
    pipeline_fingerprint = _pipeline_fingerprint(
        recipe,
        resolved_lock,
        source,
        profile=profile,
        tokenizer_sha256=tokenizer_sha256,
    )
    source_fingerprint = _source_fingerprint(resolved)
    if pipeline_fingerprint != expected_pipeline_fingerprint:
        raise RefillError(
            f"original chunk pipeline fingerprint is no longer reproducible: {source.source_id}"
        )
    if source_fingerprint != expected_source_fingerprint:
        raise RefillError(f"resolved source fingerprint changed: {source.source_id}")
    progress, chunks = _load_source_progress(
        source_root,
        pipeline_fingerprint=pipeline_fingerprint,
        source_fingerprint=source_fingerprint,
    )
    while not _target_reached(
        progress,
        train_target=train_target,
        validation_target=validation_target,
    ):
        if stop_file is not None and stop_file.exists():
            raise CorpusBuildStopped(
                stop_file,
                progress.train_tokens + progress.validation_tokens,
            )
        shard_id = f"chunk-{progress.next_chunk:06d}"
        with ShardTransaction(
            source_root,
            shard_id,
            fingerprint=pipeline_fingerprint,
            source_fingerprint=source_fingerprint,
        ) as transaction:
            if transaction.complete:
                progress, chunks = _load_source_progress(
                    source_root,
                    pipeline_fingerprint=pipeline_fingerprint,
                    source_fingerprint=source_fingerprint,
                )
                continue
            _reset_chunk_outputs(transaction.work_directory)
            writer = _JsonlChunkWriter(transaction.work_directory)
            try:
                next_cursor, stats, exhausted = _read_chunk(
                    source=source,
                    resolved=resolved,
                    recipe=recipe,
                    progress=progress,
                    train_target=train_target,
                    validation_target=validation_target,
                    tokenizer=tokenizer,
                    row_iterator=row_iterator,
                    writer=writer,
                    seen_hashes=seen_hashes,
                )
                known_hashes = writer.close()
            except BaseException:
                writer.abort()
                raise
            if stats["rows_scanned"] == 0 and exhausted:
                raise RefillError(
                    f"{source.source_id} exhausted before refill targets: "
                    f"train {progress.train_tokens}/{train_target}, "
                    f"validation {progress.validation_tokens}/{validation_target}"
                )
            chunk_summary = {
                "schema_version": 1,
                "source_id": source.source_id,
                "profile": profile,
                "start_file_index": progress.cursor.file_index,
                "start_row_index": progress.cursor.row_index,
                "next_file_index": next_cursor.file_index,
                "next_row_index": next_cursor.row_index,
                **stats,
            }
            atomic_write_json(transaction.work_directory / "chunk.json", chunk_summary)
            known_hashes["chunk.json"] = sha256_file(transaction.work_directory / "chunk.json")
            final = transaction.commit(chunk_summary, known_sha256=known_hashes)
        progress.next_chunk += 1
        progress.cursor = next_cursor
        progress.train_tokens += int(stats["train_tokens"])
        progress.validation_tokens += int(stats["validation_tokens"])
        progress.train_rows += int(stats["train_rows"])
        progress.validation_rows += int(stats["validation_rows"])
        marker = read_complete_marker(final)
        chunks.append({"path": final.as_posix(), "marker": marker})
        progress_bar.update(int(stats["train_tokens"]) + int(stats["validation_tokens"]))
        progress_bar.set_postfix(
            {
                "source": source.source_id,
                "train": progress.train_tokens,
                "val": progress.validation_tokens,
                "file": progress.cursor.file_index,
            }
        )
        if exhausted and not _target_reached(
            progress,
            train_target=train_target,
            validation_target=validation_target,
        ):
            raise RefillError(f"{source.source_id} exhausted before reaching refill targets")
    return progress, chunks


def _new_chunk_inventory(
    *,
    root: Path,
    plan: Mapping[str, object],
) -> dict[str, object]:
    raw_sources = plan.get("sources")
    if not isinstance(raw_sources, list):
        raise RefillError("plan sources are invalid")
    chunks: list[dict[str, object]] = []
    for raw in raw_sources:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("source_id"), str):
            raise RefillError("plan source entry is invalid")
        source_id = str(raw["source_id"])
        initial_count = int(raw.get("original_chunk_count", -1))
        source_root = root / "extracted" / source_id
        directories = sorted(
            path
            for path in source_root.glob("chunk-[0-9][0-9][0-9][0-9][0-9][0-9]")
            if path.is_dir()
        )
        for directory in directories[initial_count:]:
            if not is_shard_complete(directory, verify_hashes=True):
                raise RefillError(f"new refill chunk is incomplete: {directory}")
            marker_path = directory / "COMPLETE"
            marker = read_complete_marker(directory)
            if marker.get("fingerprint") != raw.get("original_pipeline_fingerprint"):
                raise RefillError(f"new chunk fingerprint differs: {directory}")
            if marker.get("source_fingerprint") != raw.get("original_source_fingerprint"):
                raise RefillError(f"new chunk source fingerprint differs: {directory}")
            chunks.append(
                {
                    "source_id": source_id,
                    "shard_id": directory.name,
                    "complete": _identity(
                        marker_path,
                        relative=marker_path.relative_to(root).as_posix(),
                    ),
                    "outputs": marker.get("outputs"),
                }
            )
    value: dict[str, object] = {
        "schema_version": REFILL_LINEAGE_SCHEMA_VERSION,
        "kind": REFILL_NEW_CHUNK_INVENTORY_KIND,
        "plan_fingerprint": plan.get("plan_fingerprint"),
        "chunks": chunks,
        "chunk_count": len(chunks),
    }
    value["inventory_fingerprint"] = _canonical_sha256(value)
    atomic_write_json(root / "new-chunk-inventory.json", value)
    return value


def _plan_source_targets(plan: Mapping[str, object]) -> dict[str, dict[str, Any]]:
    raw_sources = plan.get("sources")
    if not isinstance(raw_sources, list):
        raise RefillError("plan sources are invalid")
    result: dict[str, dict[str, Any]] = {}
    for raw in raw_sources:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("source_id"), str):
            raise RefillError("plan source entry is invalid")
        source_id = str(raw["source_id"])
        if source_id in result:
            raise RefillError(f"duplicate plan source: {source_id}")
        train = raw.get("train")
        validation = raw.get("validation")
        if not isinstance(train, Mapping) or not isinstance(validation, Mapping):
            raise RefillError(f"plan source role targets are invalid: {source_id}")
        result[source_id] = {
            "train_target": int(train.get("runtime_raw_target_tokens", 0)),
            "validation_target": int(validation.get("runtime_raw_target_tokens", 0)),
            "pipeline_fingerprint": str(raw.get("original_pipeline_fingerprint")),
            "source_fingerprint": str(raw.get("original_source_fingerprint")),
            "original_chunk_count": int(raw.get("original_chunk_count", -1)),
            "cursor": raw.get("cursor"),
        }
        if result[source_id]["train_target"] <= 0 or result[source_id]["validation_target"] <= 0:
            raise RefillError(f"plan runtime target is not positive: {source_id}")
    return result


def _validate_manifest_payload_without_complete(
    manifest: Path,
    value: Mapping[str, object],
) -> str:
    """Authenticate a manifest while allowing its COMPLETE SHA to be stale.

    A crash can occur after the lineage-bearing manifest rename and before the
    matching COMPLETE rename.  The normal validator correctly rejects that
    pair, so recovery independently reauthenticates every inventory byte and
    the corpus fingerprint before completing the second rename.
    """

    if value.get("schema_version") != 1 or value.get("kind") != (
        "twen_extracted_base_jsonl_corpus"
    ):
        raise RefillError("refill intermediate manifest schema/kind mismatch")
    root = manifest.parent
    inventories = {
        "train": value.get("train_files"),
        "validation": value.get("validation_files"),
        "attribution": value.get("attribution_files"),
    }
    seen: set[str] = set()
    for role, inventory in inventories.items():
        if not isinstance(inventory, list):
            raise RefillError(f"refill intermediate {role} inventory is invalid")
        for raw in inventory:
            if not isinstance(raw, Mapping):
                raise RefillError(f"refill intermediate {role} entry is invalid")
            relative = _safe_relative(raw.get("path"), f"{role}.path")
            if relative in seen:
                raise RefillError(f"duplicate refill intermediate output: {relative}")
            seen.add(relative)
            output = root / relative
            if (
                not output.is_file()
                or output.stat().st_size != raw.get("size")
                or sha256_file(output) != raw.get("sha256")
            ):
                raise RefillError(f"refill intermediate output identity changed: {relative}")
    file_lists = value.get("file_lists")
    if not isinstance(file_lists, Mapping) or set(file_lists) != set(inventories):
        raise RefillError("refill intermediate file-list inventory is invalid")
    expected_names = {
        "train": "train-files.txt",
        "validation": "validation-files.txt",
        "attribution": "attribution-files.txt",
    }
    for role, inventory in inventories.items():
        assert isinstance(inventory, list)
        identity = file_lists.get(role)
        if not isinstance(identity, Mapping):
            raise RefillError(f"refill intermediate {role} file-list identity is invalid")
        relative = _safe_relative(identity.get("path"), f"file_lists.{role}.path")
        if relative != expected_names[role]:
            raise RefillError(f"unexpected refill intermediate file list: {relative}")
        sidecar = root / relative
        expected = "".join(f"{item['path']}\n" for item in inventory)
        if (
            not sidecar.is_file()
            or sidecar.read_text(encoding="utf-8") != expected
            or sidecar.stat().st_size != identity.get("size")
            or sha256_file(sidecar) != identity.get("sha256")
        ):
            raise RefillError(f"refill intermediate file list changed: {relative}")
    identity_keys = (
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
    )
    contract_presence = [name in value for name in EXTRACTED_CONTRACT_IDENTITY_KEYS]
    if any(contract_presence) and not all(contract_presence):
        raise RefillError("refill intermediate has a partial data-contract audit")
    if all(contract_presence):
        identity_keys = (*identity_keys, *EXTRACTED_CONTRACT_IDENTITY_KEYS)
    identity = {name: value.get(name) for name in identity_keys}
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if value.get("corpus_fingerprint") != fingerprint:
        raise RefillError("refill intermediate corpus fingerprint mismatch")
    raw_sources = value.get("sources")
    if not isinstance(raw_sources, list):
        raise RefillError("refill intermediate source summaries are invalid")
    train_total = sum(
        int(source.get("actual_train_tokens", 0))
        for source in raw_sources
        if isinstance(source, Mapping)
    )
    validation_total = sum(
        int(source.get("actual_validation_tokens", 0))
        for source in raw_sources
        if isinstance(source, Mapping)
    )
    if (
        len(raw_sources) != sum(isinstance(source, Mapping) for source in raw_sources)
        or int(value.get("actual_train_tokens", -1)) != train_total
        or int(value.get("actual_validation_tokens", -1)) != validation_total
    ):
        raise RefillError("refill intermediate source/token totals mismatch")
    return hashlib.sha256(_pretty_json_bytes(value)).hexdigest()


def _validate_complete_payload_core(
    complete: Mapping[str, object],
    manifest: Path,
    manifest_value: Mapping[str, object],
) -> None:
    if (
        complete.get("schema_version") != 1
        or complete.get("kind") != "twen_extracted_base_jsonl_complete"
        or complete.get("manifest") != manifest.name
        or complete.get("corpus_fingerprint") != manifest_value.get("corpus_fingerprint")
        or complete.get("file_lists") != manifest_value.get("file_lists")
        or complete.get("ready_for_training") != manifest_value.get("ready_for_training")
    ):
        raise RefillError("refill intermediate COMPLETE core binding mismatch")


def _manifest_refill_lineage(
    *,
    plan_file: Path,
    plan: Mapping[str, object],
    hardlink_inventory_path: Path,
    hardlink_inventory: Mapping[str, object],
    new_inventory_path: Path,
    new_inventory: Mapping[str, object],
    network_policy: object,
) -> dict[str, object]:
    return {
        "schema_version": REFILL_LINEAGE_SCHEMA_VERSION,
        "kind": REFILL_LINEAGE_KIND,
        "plan": _identity(plan_file),
        "plan_fingerprint": plan.get("plan_fingerprint"),
        "base_raw_manifest": plan.get("base_raw_manifest"),
        "runtime_targets": plan.get("runtime_targets"),
        "hardlink_inventory": _identity(hardlink_inventory_path),
        "hardlink_inventory_fingerprint": hardlink_inventory.get("inventory_fingerprint"),
        "new_chunk_inventory": _identity(new_inventory_path),
        "new_chunk_inventory_fingerprint": new_inventory.get("inventory_fingerprint"),
        "continued_after_per_source_manifest_cursor": True,
        "original_recipe_and_chunk_fingerprint_unchanged": True,
        "seen_hashes_include_original_raw_corpus": True,
        "network_policy": network_policy,
        "training_started": False,
        "gpu_kd_started": False,
    }


def _complete_refill_lineage(
    *,
    plan_file: Path,
    plan: Mapping[str, object],
    hardlink_inventory_path: Path,
    hardlink_inventory: Mapping[str, object],
    new_inventory_path: Path,
    new_inventory: Mapping[str, object],
) -> dict[str, object]:
    return {
        "plan": _identity(plan_file),
        "plan_fingerprint": plan.get("plan_fingerprint"),
        "hardlink_inventory": _identity(hardlink_inventory_path),
        "hardlink_inventory_fingerprint": hardlink_inventory.get("inventory_fingerprint"),
        "new_chunk_inventory": _identity(new_inventory_path),
        "new_chunk_inventory_fingerprint": new_inventory.get("inventory_fingerprint"),
        "runtime_targets": plan.get("runtime_targets"),
        "training_started": False,
        "gpu_kd_started": False,
    }


def _finalize_refill_lineage(
    *,
    root: Path,
    plan_file: Path,
    plan: Mapping[str, object],
    fault_inject: Callable[[str], None] | None = None,
) -> Path:
    """Idempotently convert an ordinary extracted pair into a refill pair."""

    manifest = root / "corpus-manifest.json"
    complete_path = root / "COMPLETE"
    if not manifest.is_file() or not complete_path.is_file():
        raise RefillError("refill finalization requires manifest and COMPLETE")
    hardlink_inventory = _validate_hardlink_inventory(root, plan_file)
    new_inventory = _new_chunk_inventory(root=root, plan=plan)
    hardlink_inventory_path = root / "hardlink-inventory.json"
    new_inventory_path = root / "new-chunk-inventory.json"
    current_manifest = _load_object(manifest, "refill intermediate manifest")
    current_complete = _load_object(complete_path, "refill intermediate COMPLETE")
    raw_lineage = current_manifest.get("refill_lineage")
    if raw_lineage is not None and not isinstance(raw_lineage, Mapping):
        raise RefillError("refill intermediate lineage is invalid")
    ordinary_manifest = dict(current_manifest)
    ordinary_manifest.pop("refill_lineage", None)
    ordinary_manifest_sha = _validate_manifest_payload_without_complete(
        manifest,
        ordinary_manifest,
    )
    _validate_complete_payload_core(
        current_complete,
        manifest,
        ordinary_manifest,
    )
    targets = _plan_source_targets(plan)
    sources = _source_mapping(ordinary_manifest)
    if set(sources) != set(targets):
        raise RefillError("refill finalization manifest/plan source sets differ")
    for source_id, target in targets.items():
        source = sources[source_id]
        if int(source.get("actual_train_tokens", 0)) < target["train_target"]:
            raise RefillError(f"refill finalization train target not reached: {source_id}")
        if int(source.get("actual_validation_tokens", 0)) < target["validation_target"]:
            raise RefillError(f"refill finalization validation target not reached: {source_id}")

    manifest_lineage = _manifest_refill_lineage(
        plan_file=plan_file,
        plan=plan,
        hardlink_inventory_path=hardlink_inventory_path,
        hardlink_inventory=hardlink_inventory,
        new_inventory_path=new_inventory_path,
        new_inventory=new_inventory,
        network_policy=ordinary_manifest.get("network_policy"),
    )
    final_manifest = {**ordinary_manifest, "refill_lineage": manifest_lineage}
    final_manifest_sha = hashlib.sha256(_pretty_json_bytes(final_manifest)).hexdigest()
    ordinary_complete = dict(current_complete)
    ordinary_complete.pop("refill_lineage", None)
    ordinary_complete["manifest_sha256"] = ordinary_manifest_sha
    final_complete = {
        **ordinary_complete,
        "manifest_sha256": final_manifest_sha,
        "refill_lineage": _complete_refill_lineage(
            plan_file=plan_file,
            plan=plan,
            hardlink_inventory_path=hardlink_inventory_path,
            hardlink_inventory=hardlink_inventory,
            new_inventory_path=new_inventory_path,
            new_inventory=new_inventory,
        ),
    }
    if raw_lineage is None:
        if current_complete != ordinary_complete:
            raise RefillError("ordinary refill intermediate COMPLETE changed")
    else:
        if current_manifest != final_manifest:
            raise RefillError("lineage-bearing refill manifest differs from expected final bytes")
        if current_complete not in (ordinary_complete, final_complete):
            raise RefillError("lineage-bearing refill COMPLETE is neither ordinary nor final")

    state: dict[str, object] = {
        "schema_version": REFILL_LINEAGE_SCHEMA_VERSION,
        "kind": REFILL_FINALIZATION_STATE_KIND,
        "plan": _identity(plan_file),
        "plan_fingerprint": plan.get("plan_fingerprint"),
        "ordinary_manifest_sha256": ordinary_manifest_sha,
        "final_manifest_sha256": final_manifest_sha,
        "ordinary_complete_sha256": hashlib.sha256(
            _pretty_json_bytes(ordinary_complete)
        ).hexdigest(),
        "final_complete_sha256": hashlib.sha256(_pretty_json_bytes(final_complete)).hexdigest(),
        "hardlink_inventory": _identity(hardlink_inventory_path),
        "new_chunk_inventory": _identity(new_inventory_path),
    }
    state["state_fingerprint"] = _canonical_sha256(state)
    state_path = root / "REFILL_FINALIZATION.json"
    if state_path.exists():
        existing_state = _load_object(state_path, "refill finalization state")
        if existing_state != state:
            raise RefillError("refill finalization state differs from authenticated recovery")
    else:
        atomic_write_json(state_path, state)

    if raw_lineage is None:
        if fault_inject is not None:
            fault_inject("ordinary_pair_authenticated")
        atomic_write_json(manifest, final_manifest)
    if fault_inject is not None:
        fault_inject("lineage_manifest_written")
    if current_complete != final_complete:
        atomic_write_json(complete_path, final_complete)
    validate_refill_lineage(manifest)
    state_path.unlink(missing_ok=True)
    return manifest


def build_refill_lineage(
    *,
    plan_path: str | Path,
    resolved_lock_path: str | Path,
    output_root: str | Path,
    tokenizer_path: str | Path,
    tokenizer_manifest_sha256: str,
    network_policy: str = "fallback",
    proxy_url: str | None = None,
    token: str | None = None,
    range_block_size: int = 8 * 1024 * 1024,
    stop_file: str | Path | None = None,
    progress: str = "auto",
    _tokenizer: Any | None = None,
    _row_iterator: Any | None = None,
    _finalization_fault: Callable[[str], None] | None = None,
) -> Path:
    """Hard-link a raw parent and append Range-streamed chunks to plan targets."""

    plan_file = Path(plan_path).resolve()
    plan = validate_refill_plan(plan_file)
    recipe_identity = plan.get("recipe")
    if not isinstance(recipe_identity, Mapping):
        raise RefillError("plan recipe identity is invalid")
    recipe_path = Path(str(recipe_identity.get("path"))).resolve()
    if sha256_file(recipe_path) != recipe_identity.get("sha256"):
        raise RefillError("refill recipe bytes changed")
    recipe = load_base_data_recipe(recipe_path)
    profile = str(plan.get("profile"))
    if profile not in recipe.profiles:
        raise RefillError(f"plan profile is invalid: {profile!r}")
    tokenizer_sha = _normalized_sha(tokenizer_manifest_sha256, "tokenizer_manifest_sha256")
    if tokenizer_sha != plan.get("tokenizer_manifest_sha256"):
        raise RefillError("runtime tokenizer manifest SHA differs from refill plan")
    resolved_lock = load_resolved_source_lock(resolved_lock_path, recipe)
    if resolved_lock.sha256 != plan.get("resolved_source_lock_sha256"):
        raise RefillError("runtime resolved source lock differs from refill plan")
    if _tokenizer is None:
        from ..io.offline import verify_local_download_directory

        verify_local_download_directory(
            tokenizer_path,
            expected_manifest_sha256=tokenizer_sha,
        )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    else:
        tokenizer = _tokenizer
    if getattr(tokenizer, "eos_token_id", None) is None:
        raise RefillError("tokenizer must define eos_token_id")

    root = Path(output_root).resolve()
    with FileLock(root.parent / f".{root.name}.refill.lock", timeout_seconds=300.0):
        _initialize_lineage(root, plan_file, plan)
        if (root / "corpus-manifest.json").is_file() and (root / "COMPLETE").is_file():
            return _finalize_refill_lineage(
                root=root,
                plan_file=plan_file,
                plan=plan,
                fault_inject=_finalization_fault,
            )
        _validate_hardlink_inventory(root, plan_file)
        targets = _plan_source_targets(plan)
        if set(targets) != {source.source_id for source in recipe.sources}:
            raise RefillError("plan/recipe source sets differ")
        seen_hashes = _load_seen_hashes(root / "extracted", recipe.sources)
        resolved_by_id = {item.source_id: item for item in resolved_lock.sources}

        if _row_iterator is None:
            proxy_settings = ProxySettings.from_environment(proxy_url=proxy_url)
            file_factory = HfRangeFileFactory(
                network_policy=network_policy,
                proxy_settings=proxy_settings,
                token=token,
                block_size=range_block_size,
            )
            download_manager = DownloadManager(
                network_policy=network_policy,
                proxy_settings=proxy_settings,
            )
            resolved_for_url: dict[str, tuple[SourceRecipe, ResolvedSource]] = {}
            derived_for_url: dict[str, Any] = {}
            for source_recipe in recipe.sources:
                resolved_source = resolved_by_id[source_recipe.source_id]
                for resolved_file in resolved_source.files:
                    if resolved_file.url in resolved_for_url:
                        raise RefillError(f"duplicate resolved source URL: {resolved_file.url}")
                    resolved_for_url[resolved_file.url] = (
                        source_recipe,
                        resolved_source,
                    )
            base_identity = plan.get("base_raw_manifest")
            if not isinstance(base_identity, Mapping):
                raise RefillError("plan base raw identity is invalid")
            base_cache_root = (
                Path(str(base_identity.get("path"))).resolve().parent / ".source-cache"
            )

            def row_iterator(
                artifact: Any,
                start_row: int,
                columns: Sequence[str],
            ) -> Iterator[tuple[int, Mapping[str, object]]]:
                if artifact.storage_format == "parquet":
                    return iter_remote_parquet_rows(
                        artifact,
                        start_row,
                        columns,
                        file_factory=file_factory,
                    )
                if artifact.storage_format != "jsonl_gzip":
                    raise RefillError(
                        f"unsupported resolved storage format: {artifact.storage_format}"
                    )
                source_recipe, resolved_source = resolved_for_url[artifact.url]
                derived = derived_for_url.get(artifact.url)
                if derived is None:
                    derived = materialize_jsonl_gzip_artifact(
                        artifact,
                        source_id=source_recipe.source_id,
                        repo_id=resolved_source.repo_id,
                        revision=resolved_source.revision,
                        cache_root=base_cache_root,
                        manager=download_manager,
                        token=token,
                    )
                    derived_for_url[artifact.url] = derived
                return iter_local_jsonl_rows(
                    derived.path,
                    start_row,
                    columns,
                )

        else:
            file_factory = None
            download_manager = None
            row_iterator = _row_iterator

        initial_tokens = 0
        for source in recipe.sources:
            resolved = resolved_by_id[source.source_id]
            target = targets[source.source_id]
            source_progress, _ = _load_source_progress(
                root / "extracted" / source.source_id,
                pipeline_fingerprint=target["pipeline_fingerprint"],
                source_fingerprint=target["source_fingerprint"],
            )
            expected_cursor = target["cursor"]
            if source_progress.next_chunk < target["original_chunk_count"]:
                raise RefillError(f"hard-linked chunk prefix is incomplete: {source.source_id}")
            if source_progress.next_chunk == target["original_chunk_count"] and (
                not isinstance(expected_cursor, Mapping)
                or source_progress.cursor
                != RowCursor(
                    int(expected_cursor.get("file_index", -1)),
                    int(expected_cursor.get("row_index", -1)),
                )
            ):
                raise RefillError(f"hard-linked source cursor changed: {source.source_id}")
            initial_tokens += source_progress.train_tokens + source_progress.validation_tokens
        target_total = sum(
            int(item["train_target"]) + int(item["validation_target"]) for item in targets.values()
        )
        stop = Path(stop_file).resolve() if stop_file is not None else None
        results: list[
            tuple[SourceRecipe, ResolvedSource, SourceProgress, list[dict[str, object]]]
        ] = []
        with TaskProgress(
            total=max(target_total, initial_tokens),
            initial=initial_tokens,
            description=f"base-{profile}-refill",
            unit="tok",
            unit_scale=True,
            mode=progress,
        ) as progress_bar:
            for source in recipe.sources:
                resolved = resolved_by_id[source.source_id]
                target = targets[source.source_id]
                source_progress, chunks = _build_source_to_runtime_targets(
                    source=source,
                    resolved=resolved,
                    recipe=recipe,
                    resolved_lock=resolved_lock,
                    profile=profile,
                    tokenizer_sha256=tokenizer_sha,
                    tokenizer=tokenizer,
                    output_root=root,
                    row_iterator=row_iterator,
                    seen_hashes=seen_hashes,
                    progress_bar=progress_bar,
                    stop_file=stop,
                    train_target=target["train_target"],
                    validation_target=target["validation_target"],
                    expected_pipeline_fingerprint=target["pipeline_fingerprint"],
                    expected_source_fingerprint=target["source_fingerprint"],
                )
                results.append((source, resolved, source_progress, chunks))
        if file_factory is None:
            effective_policy = network_policy
        elif file_factory.effective_network_policy == "proxy-fallback" or (
            download_manager is not None
            and download_manager.effective_network_policy == "proxy-fallback"
        ):
            effective_policy = "proxy-fallback"
        else:
            effective_policy = file_factory.effective_network_policy
        _write_corpus_manifest(
            output_root=root,
            recipe=recipe,
            resolved_lock=resolved_lock,
            tokenizer_sha256=tokenizer_sha,
            profile=profile,
            source_results=results,
            network_policy=effective_policy,
        )
        return _finalize_refill_lineage(
            root=root,
            plan_file=plan_file,
            plan=plan,
            fault_inject=_finalization_fault,
        )


def validate_refill_lineage(manifest_path: str | Path) -> dict[str, object]:
    manifest, value, manifest_sha = _validated_manifest(manifest_path)
    root = manifest.parent
    lineage = value.get("refill_lineage")
    if (
        not isinstance(lineage, Mapping)
        or lineage.get("schema_version") != REFILL_LINEAGE_SCHEMA_VERSION
        or lineage.get("kind") != REFILL_LINEAGE_KIND
    ):
        raise RefillError("manifest has no valid refill lineage")
    plan_identity = lineage.get("plan")
    if not isinstance(plan_identity, Mapping):
        raise RefillError("refill lineage plan identity is invalid")
    plan_path = Path(str(plan_identity.get("path"))).resolve()
    if (
        not plan_path.is_file()
        or plan_path.stat().st_size != plan_identity.get("size")
        or sha256_file(plan_path) != plan_identity.get("sha256")
    ):
        raise RefillError("refill lineage plan identity changed")
    plan = validate_refill_plan(plan_path)
    if lineage.get("plan_fingerprint") != plan.get("plan_fingerprint"):
        raise RefillError("refill lineage plan fingerprint mismatch")
    hardlinks = _validate_hardlink_inventory(root, plan_path)
    hardlink_path = root / "hardlink-inventory.json"
    new_path = root / "new-chunk-inventory.json"
    new_inventory = _load_object(new_path, "new chunk inventory")
    if (
        new_inventory.get("schema_version") != REFILL_LINEAGE_SCHEMA_VERSION
        or new_inventory.get("kind") != REFILL_NEW_CHUNK_INVENTORY_KIND
    ):
        raise RefillError("new chunk inventory schema/kind mismatch")
    new_fingerprint = _normalized_sha(
        new_inventory.get("inventory_fingerprint"), "new_chunk.inventory_fingerprint"
    )
    new_payload = dict(new_inventory)
    new_payload.pop("inventory_fingerprint")
    if _canonical_sha256(new_payload) != new_fingerprint:
        raise RefillError("new chunk inventory fingerprint mismatch")
    for raw in new_inventory.get("chunks", []):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("complete"), Mapping):
            raise RefillError("new chunk inventory entry is invalid")
        identity = raw["complete"]
        relative = _safe_relative(identity.get("path"), "new_chunk.complete.path")
        complete_path = root / relative
        chunk_directory = complete_path.parent
        if (
            not complete_path.is_file()
            or complete_path.stat().st_size != identity.get("size")
            or sha256_file(complete_path) != identity.get("sha256")
            or not is_shard_complete(chunk_directory, verify_hashes=True)
        ):
            raise RefillError(f"new chunk COMPLETE identity changed: {relative}")
    for name, path, fingerprint_value in (
        (
            "hardlink_inventory",
            hardlink_path,
            hardlinks.get("inventory_fingerprint"),
        ),
        ("new_chunk_inventory", new_path, new_fingerprint),
    ):
        identity = lineage.get(name)
        if (
            not isinstance(identity, Mapping)
            or identity.get("size") != path.stat().st_size
            or identity.get("sha256") != sha256_file(path)
            or lineage.get(f"{name}_fingerprint") != fingerprint_value
        ):
            raise RefillError(f"manifest refill lineage {name} binding mismatch")
    targets = _plan_source_targets(plan)
    sources = _source_mapping(value)
    if set(sources) != set(targets):
        raise RefillError("refill manifest/plan source sets differ")
    for source_id, target in targets.items():
        source = sources[source_id]
        if int(source.get("actual_train_tokens", 0)) < target["train_target"]:
            raise RefillError(f"refill train target not reached: {source_id}")
        if int(source.get("actual_validation_tokens", 0)) < target["validation_target"]:
            raise RefillError(f"refill validation target not reached: {source_id}")
    complete = _load_object(root / "COMPLETE", "refill corpus COMPLETE")
    complete_lineage = complete.get("refill_lineage")
    if (
        complete.get("manifest_sha256") != manifest_sha
        or not isinstance(complete_lineage, Mapping)
        or complete_lineage.get("plan_fingerprint") != plan.get("plan_fingerprint")
        or complete_lineage.get("runtime_targets") != plan.get("runtime_targets")
        or complete_lineage.get("hardlink_inventory_fingerprint")
        != hardlinks.get("inventory_fingerprint")
        or complete_lineage.get("new_chunk_inventory_fingerprint") != new_fingerprint
    ):
        raise RefillError("refill corpus COMPLETE lineage binding mismatch")
    return {
        "ok": True,
        "manifest": _identity(manifest),
        "plan": _identity(plan_path),
        "plan_fingerprint": plan.get("plan_fingerprint"),
        "hardlink_inventory": _identity(hardlink_path),
        "new_chunk_inventory": _identity(new_path),
        "runtime_targets": plan.get("runtime_targets"),
        "train_tokens": value.get("actual_train_tokens"),
        "validation_tokens": value.get("actual_validation_tokens"),
        "network_policy": value.get("network_policy"),
        "training_started": False,
        "gpu_kd_started": False,
    }


__all__ = [
    "DEFAULT_CLEAN_GUARD_RATIO",
    "DEFAULT_SURVIVAL_GUARD_POINTS",
    "REFILL_LINEAGE_KIND",
    "REFILL_PLAN_KIND",
    "REFILL_SOURCE_SHA256",
    "RefillError",
    "build_refill_lineage",
    "corpus_tokens_by_source",
    "create_refill_plan",
    "validate_refill_lineage",
    "validate_refill_plan",
]
