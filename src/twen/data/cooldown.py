"""Fail-closed contract for a quality-cooldown prepared/KD subset view.

The view selects whole authenticated shards from the primary corpus. Tensor
files may be hard-linked for storage efficiency, but both corpus manifests and
the per-shard KD manifests remain independent identities with re-based global
ranges and a distinct dataset fingerprint.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from ..io.locking import FileLock
from ..utils import atomic_write_json, sha256_file
from .prepared import (
    PREPARED_SHARD_MANIFEST,
    PREPARED_TENSORS,
    PreparedCorpusManifest,
    PreparedShardEntry,
    _local_prepared_manifest,
    _prepared_dataset_fingerprint,
    _prepared_pipeline_fingerprint,
    validate_prepared_corpus,
)
from .teacher_kd import (
    KD_MANIFEST_FILENAME,
    KD_TENSORS_FILENAME,
    TeacherKDCorpusManifest,
    TeacherKDShardEntry,
    read_kd_manifest,
    validate_kd_corpus_coverage,
    validate_kd_corpus_manifest,
)

QUALITY_COOLDOWN_SCHEMA_VERSION = 1
QUALITY_COOLDOWN_KIND = "authenticated_prepared_kd_subset_view"
QUALITY_COOLDOWN_POLICY_KIND = "twen_quality_cooldown_selection_policy"
QUALITY_COOLDOWN_BUNDLE_KIND = "twen_quality_cooldown_subset_bundle"
QUALITY_COOLDOWN_COMPLETE_KIND = "twen_quality_cooldown_subset_complete"


@dataclass(frozen=True, slots=True)
class QualityCooldownSummary:
    selection_policy_id: str
    parent_dataset_fingerprint: str
    cooldown_dataset_fingerprint: str
    selected_shard_ids: tuple[str, ...]
    source_mix_token_counts: tuple[tuple[str, int], ...]
    sequence_count: int
    token_count: int


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _prepared_payload(entry: PreparedShardEntry) -> tuple[object, ...]:
    return (
        entry.source_path,
        entry.source_sha256,
        entry.tensors_sha256,
        entry.sequence_count,
        entry.token_count,
    )


def _kd_payload(entry: TeacherKDShardEntry) -> tuple[object, ...]:
    return (
        entry.source_tensors_sha256,
        entry.tensors_sha256,
        entry.sequence_count,
        entry.token_count,
    )


def _authenticated_source_map(
    corpus: PreparedCorpusManifest,
) -> dict[str, str] | None:
    """Resolve prepared source paths to IDs from extracted-corpus lineage once."""

    lineage = corpus.lineage
    if lineage is None:
        return None
    if not isinstance(lineage, Mapping) or lineage.get("kind") != "authenticated_extracted_corpus":
        raise ValueError("quality cooldown requires authenticated extracted-corpus lineage")
    extracted_manifest = lineage.get("extracted_manifest_path")
    source_files = lineage.get("source_files")
    if not isinstance(extracted_manifest, str) or not isinstance(source_files, list):
        raise ValueError("quality cooldown parent source lineage is incomplete")
    extracted_root = Path(extracted_manifest).expanduser().resolve().parent
    relative_by_path: dict[str, PurePosixPath] = {}
    for raw in source_files:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("path"), str):
            raise ValueError("quality cooldown parent source inventory is invalid")
        relative = PurePosixPath(str(raw["path"]))
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2:
            raise ValueError("quality cooldown parent source path is invalid")
        relative_by_path[str((extracted_root / relative).resolve())] = relative

    # The extracted manifest carries an exact output-path -> source-ID map,
    # including for audit-filtered paths whose filename alone is ambiguous.
    manifest_path = Path(extracted_manifest).expanduser().resolve()
    expected_manifest_sha = lineage.get("extracted_manifest_sha256")
    if (
        not isinstance(expected_manifest_sha, str)
        or not manifest_path.is_file()
        or sha256_file(manifest_path) != expected_manifest_sha
    ):
        raise ValueError("quality cooldown parent extracted manifest changed")
    extracted = _load_json_object(manifest_path, "authenticated extracted manifest")
    ids_by_relative: dict[str, set[str]] = {}
    raw_sources = extracted.get("sources")
    if isinstance(raw_sources, list):
        for raw_source in raw_sources:
            if not isinstance(raw_source, Mapping):
                continue
            source_id = raw_source.get("source_id")
            chunks = raw_source.get("chunks")
            if not isinstance(source_id, str) or not source_id or not isinstance(chunks, list):
                continue
            for chunk in chunks:
                if not isinstance(chunk, Mapping) or not isinstance(chunk.get("outputs"), list):
                    continue
                for output in chunk["outputs"]:
                    if isinstance(output, Mapping) and isinstance(output.get("path"), str):
                        ids_by_relative.setdefault(str(output["path"]), set()).add(source_id)

    resolved: dict[str, str] = {}
    for source_path, relative in relative_by_path.items():
        matching_ids = ids_by_relative.get(relative.as_posix(), set())
        if len(matching_ids) == 1:
            resolved[source_path] = next(iter(matching_ids))
            continue
        if isinstance(raw_sources, list) and raw_sources:
            raise ValueError(
                f"quality cooldown source ID mapping is missing or ambiguous: {relative}"
            )
        parts = relative.parts
        resolved[source_path] = (
            parts[1] if parts[0] == "extracted" and len(parts) >= 2 else parts[0]
        )
    return resolved


def validate_quality_cooldown_subset(
    primary_prepared: PreparedCorpusManifest,
    primary_kd: TeacherKDCorpusManifest,
    cooldown_prepared: PreparedCorpusManifest,
    cooldown_kd: TeacherKDCorpusManifest,
    *,
    primary_prepared_manifest_sha256: str,
    primary_kd_manifest_sha256: str,
    required_cooldown_tokens: int,
) -> QualityCooldownSummary:
    """Authenticate an ordered whole-shard view against the primary tensors.

    This function assumes the normal prepared/KD validators already checked
    every referenced file and exact prepared/KD coverage for both corpora.
    """

    if required_cooldown_tokens <= 0:
        raise ValueError("required cooldown token count must be positive")
    if cooldown_prepared.token_count < required_cooldown_tokens:
        raise ValueError(
            "quality cooldown corpus is too small: "
            f"requires {required_cooldown_tokens}, has {cooldown_prepared.token_count}"
        )
    if cooldown_prepared.dataset_fingerprint == primary_prepared.dataset_fingerprint:
        raise ValueError("quality cooldown must have an independent dataset fingerprint")
    if len(cooldown_prepared.shards) >= len(primary_prepared.shards):
        raise ValueError("quality cooldown must be a strict whole-shard subset")
    if (
        cooldown_prepared.generator_source_sha256 != primary_prepared.generator_source_sha256
        or cooldown_prepared.tokenizer_sha256 != primary_prepared.tokenizer_sha256
        or cooldown_prepared.sequence_length != primary_prepared.sequence_length
        or cooldown_prepared.text_field != primary_prepared.text_field
    ):
        raise ValueError("quality cooldown prepared semantics differ from primary")

    lineage = _require_mapping(cooldown_prepared.lineage, "cooldown prepared lineage")
    if (
        lineage.get("ready_for_training") is not True
        or lineage.get("research_only") is not False
        or lineage.get("pending_audits") != []
    ):
        raise ValueError("quality cooldown prepared lineage is not training-ready")
    contract = _require_mapping(
        lineage.get("quality_cooldown"), "quality_cooldown lineage contract"
    )
    if (
        type(contract.get("schema_version")) is not int
        or contract.get("schema_version") != QUALITY_COOLDOWN_SCHEMA_VERSION
        or contract.get("kind") != QUALITY_COOLDOWN_KIND
        or contract.get("eligible") is not True
    ):
        raise ValueError("unsupported or unapproved quality_cooldown lineage contract")
    expected_contract_fields = {
        "schema_version",
        "kind",
        "eligible",
        "parent_prepared_manifest_sha256",
        "parent_kd_manifest_sha256",
        "parent_dataset_fingerprint",
        "selection_policy_id",
        "selection_policy_sha256",
        "selection_basis",
        "required_cooldown_tokens",
        "ordered_parent_shard_ids",
        "shard_source_ids",
        "source_mix_token_counts",
    }
    if set(contract) != expected_contract_fields:
        raise ValueError("quality cooldown lineage fields differ from the locked schema")
    if contract.get("parent_prepared_manifest_sha256") != primary_prepared_manifest_sha256:
        raise ValueError("quality cooldown does not bind the primary prepared manifest")
    if contract.get("parent_kd_manifest_sha256") != primary_kd_manifest_sha256:
        raise ValueError("quality cooldown does not bind the primary KD manifest")
    if contract.get("parent_dataset_fingerprint") != primary_prepared.dataset_fingerprint:
        raise ValueError("quality cooldown parent dataset fingerprint changed")
    policy_id = contract.get("selection_policy_id")
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("quality cooldown selection_policy_id is required")
    if (
        not isinstance(contract.get("selection_basis"), str)
        or not str(contract["selection_basis"]).strip()
    ):
        raise ValueError("quality cooldown selection_basis is required")
    if contract.get("required_cooldown_tokens") != required_cooldown_tokens:
        raise ValueError("quality cooldown lineage token requirement changed")

    selected_ids = tuple(entry.shard_id for entry in cooldown_prepared.shards)
    if contract.get("ordered_parent_shard_ids") != list(selected_ids):
        raise ValueError("quality cooldown ordered parent shard list changed")
    primary_entries = {entry.shard_id: entry for entry in primary_prepared.shards}
    authenticated_sources = _authenticated_source_map(primary_prepared)
    for entry in cooldown_prepared.shards:
        parent = primary_entries.get(entry.shard_id)
        if parent is None or _prepared_payload(entry) != _prepared_payload(parent):
            raise ValueError(
                f"quality cooldown prepared shard is not an exact parent tensor: {entry.shard_id}"
            )

    raw_sources = _require_mapping(
        contract.get("shard_source_ids"), "quality cooldown shard_source_ids"
    )
    if set(raw_sources) != set(selected_ids):
        raise ValueError("quality cooldown shard_source_ids do not cover selected shards")
    computed_mix: dict[str, int] = {}
    for entry in cooldown_prepared.shards:
        source_id = raw_sources.get(entry.shard_id)
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"quality cooldown source ID is invalid: {entry.shard_id}")
        parent_source_path = str(
            Path(primary_entries[entry.shard_id].source_path).expanduser().resolve()
        )
        source_matches = (
            source_id in PurePosixPath(entry.source_path).parts
            if authenticated_sources is None
            else authenticated_sources.get(parent_source_path) == source_id
        )
        if not source_matches:
            raise ValueError(f"quality cooldown source ID is not authenticated: {entry.shard_id}")
        computed_mix[source_id] = computed_mix.get(source_id, 0) + entry.token_count
    raw_mix = _require_mapping(
        contract.get("source_mix_token_counts"),
        "quality cooldown source_mix_token_counts",
    )
    declared_mix: dict[str, int] = {}
    for source_id, value in raw_mix.items():
        if (
            not isinstance(source_id, str)
            or not source_id
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError("quality cooldown source mix entries must be positive integers")
        declared_mix[source_id] = value
    if declared_mix != computed_mix:
        raise ValueError("quality cooldown source mix does not match selected shard tokens")

    primary_kd_entries = {entry.source_shard_id: entry for entry in primary_kd.shards}
    cooldown_kd_entries = {entry.source_shard_id: entry for entry in cooldown_kd.shards}
    if tuple(cooldown_kd_entries) != selected_ids:
        raise ValueError("quality cooldown KD shard order differs from prepared view")
    for shard_id in selected_ids:
        parent = primary_kd_entries.get(shard_id)
        selected = cooldown_kd_entries[shard_id]
        if parent is None or _kd_payload(selected) != _kd_payload(parent):
            raise ValueError(
                f"quality cooldown KD shard is not the authenticated parent tensor: {shard_id}"
            )
    primary_teacher = (
        primary_kd.teacher_model_id,
        primary_kd.teacher_revision,
        primary_kd.teacher_model_sha256,
        primary_kd.generator_source_sha256,
        primary_kd.tokenizer_sha256,
        primary_kd.temperature,
        primary_kd.top_k,
        primary_kd.topk_logits_are_raw,
        primary_kd.normalization_definition,
    )
    cooldown_teacher = (
        cooldown_kd.teacher_model_id,
        cooldown_kd.teacher_revision,
        cooldown_kd.teacher_model_sha256,
        cooldown_kd.generator_source_sha256,
        cooldown_kd.tokenizer_sha256,
        cooldown_kd.temperature,
        cooldown_kd.top_k,
        cooldown_kd.topk_logits_are_raw,
        cooldown_kd.normalization_definition,
    )
    if cooldown_teacher != primary_teacher:
        raise ValueError("quality cooldown KD teacher/top-k semantics differ from primary")
    if cooldown_kd.dataset_fingerprint != cooldown_prepared.dataset_fingerprint:
        raise ValueError("quality cooldown prepared/KD dataset fingerprint mismatch")

    return QualityCooldownSummary(
        selection_policy_id=policy_id,
        parent_dataset_fingerprint=primary_prepared.dataset_fingerprint,
        cooldown_dataset_fingerprint=cooldown_prepared.dataset_fingerprint,
        selected_shard_ids=selected_ids,
        source_mix_token_counts=tuple(sorted(computed_mix.items())),
        sequence_count=cooldown_prepared.sequence_count,
        token_count=cooldown_prepared.token_count,
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_values_equal(first: object, second: object) -> bool:
    try:
        return _canonical_json(first) == _canonical_json(second)
    except (TypeError, ValueError):
        return False


def _canonical_sha256(value: object) -> str:
    import hashlib

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_identity(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    target = path.resolve()
    display = (
        target.relative_to(relative_to.resolve()).as_posix()
        if relative_to is not None
        else str(target)
    )
    return {
        "path": display,
        "size": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid or missing {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _validate_policy(
    path: Path,
    *,
    primary_prepared_sha256: str,
    primary_kd_sha256: str,
    required_cooldown_tokens: int,
) -> dict[str, Any]:
    policy = _load_json_object(path, "quality cooldown selection policy")
    expected_fields = {
        "schema_version",
        "kind",
        "policy_id",
        "approved_for_quality_cooldown",
        "selection_basis",
        "parent_prepared_manifest_sha256",
        "parent_kd_manifest_sha256",
        "required_cooldown_tokens",
        "ordered_shards",
        "declared_source_mix_token_counts",
    }
    if set(policy) != expected_fields:
        raise ValueError(
            "quality cooldown policy fields differ from the locked schema: "
            f"expected {sorted(expected_fields)}, got {sorted(policy)}"
        )
    if (
        type(policy.get("schema_version")) is not int
        or policy.get("schema_version") != QUALITY_COOLDOWN_SCHEMA_VERSION
        or policy.get("kind") != QUALITY_COOLDOWN_POLICY_KIND
        or policy.get("approved_for_quality_cooldown") is not True
    ):
        raise ValueError("quality cooldown selection policy is not explicitly approved")
    for field in ("policy_id", "selection_basis"):
        if not isinstance(policy.get(field), str) or not policy[field].strip():
            raise ValueError(f"quality cooldown policy {field} is required")
    if policy.get("parent_prepared_manifest_sha256") != primary_prepared_sha256:
        raise ValueError("quality cooldown policy does not bind the primary prepared manifest")
    if policy.get("parent_kd_manifest_sha256") != primary_kd_sha256:
        raise ValueError("quality cooldown policy does not bind the primary KD manifest")
    policy_required_tokens = policy.get("required_cooldown_tokens")
    if (
        isinstance(policy_required_tokens, bool)
        or not isinstance(policy_required_tokens, int)
        or policy_required_tokens <= 0
    ):
        raise ValueError("quality cooldown policy token requirement must be positive")
    if policy_required_tokens != required_cooldown_tokens:
        raise ValueError("quality cooldown policy token requirement differs from the CLI")
    raw_shards = policy.get("ordered_shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("quality cooldown policy ordered_shards cannot be empty")
    normalized_shards: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_shards):
        if not isinstance(raw, dict) or set(raw) != {"shard_id", "source_id"}:
            raise ValueError(f"quality cooldown ordered_shards[{index}] is invalid")
        shard_id = raw.get("shard_id")
        source_id = raw.get("source_id")
        if (
            not isinstance(shard_id, str)
            or not shard_id
            or not isinstance(source_id, str)
            or not source_id
        ):
            raise ValueError(f"quality cooldown ordered_shards[{index}] is invalid")
        if shard_id in seen:
            raise ValueError(f"quality cooldown policy repeats parent shard: {shard_id}")
        seen.add(shard_id)
        normalized_shards.append({"shard_id": shard_id, "source_id": source_id})
    raw_mix = policy.get("declared_source_mix_token_counts")
    if not isinstance(raw_mix, dict) or not raw_mix:
        raise ValueError("quality cooldown policy source mix cannot be empty")
    normalized_mix: dict[str, int] = {}
    for source_id, tokens in raw_mix.items():
        if (
            not isinstance(source_id, str)
            or not source_id
            or isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens <= 0
        ):
            raise ValueError("quality cooldown policy source mix is invalid")
        normalized_mix[source_id] = tokens
    return {
        **policy,
        "policy_id": policy["policy_id"].strip(),
        "selection_basis": policy["selection_basis"].strip(),
        "ordered_shards": normalized_shards,
        "declared_source_mix_token_counts": normalized_mix,
    }


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return first == second or first in second.parents or second in first.parents


def _assert_safe_output(
    output: Path,
    *,
    prepared_manifest: Path,
    kd_manifest: Path,
    policy_path: Path,
) -> None:
    if not output.name or output == output.parent:
        raise ValueError("quality cooldown output root is unsafe")
    for label, source_root in (
        ("primary prepared", prepared_manifest.parent),
        ("primary KD", kd_manifest.parent),
    ):
        if _paths_overlap(output, source_root):
            raise ValueError(
                f"quality cooldown output must not overlap {label} source: {source_root}"
            )
    if output == policy_path.resolve() or output in policy_path.resolve().parents:
        raise ValueError("quality cooldown output must not contain the selection policy")
    lock_path = output.parent / f".{output.name}.materialize-cooldown.lock"
    protected_inputs = {
        "primary prepared manifest": prepared_manifest,
        "primary KD manifest": kd_manifest,
        "selection policy": policy_path,
    }
    for label, protected in protected_inputs.items():
        same_path = lock_path.resolve() == protected.resolve()
        same_inode = lock_path.exists() and protected.exists() and lock_path.samefile(protected)
        if same_path or same_inode:
            raise ValueError(f"quality cooldown lock path collides with {label}")
    if lock_path.is_symlink():
        raise ValueError("quality cooldown lock path must not be a symlink")
    if lock_path.exists():
        if not lock_path.is_file() or lock_path.stat().st_size > 128:
            raise ValueError("existing quality cooldown lock path is not a lock file")
        try:
            lock_payload = lock_path.read_text(encoding="ascii")
        except (OSError, UnicodeError) as error:
            raise ValueError("existing quality cooldown lock file is invalid") from error
        if re.fullmatch(r"pid=[0-9]+\n", lock_payload) is None:
            raise ValueError("existing quality cooldown lock file is invalid")


def _ensure_hardlink(
    source: Path,
    destination: Path,
    expected_sha256: str,
    *,
    allow_create: bool = True,
) -> dict[str, object]:
    source = source.resolve()
    if not source.is_file() or sha256_file(source) != expected_sha256:
        raise ValueError(f"authenticated source tensor changed: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError(f"hardlink destination must not be a symlink: {destination}")
    if destination.exists():
        if not destination.is_file():
            raise ValueError(f"hardlink destination is not a file: {destination}")
    elif allow_create:
        try:
            os.link(source, destination)
        except OSError as error:
            raise ValueError(
                "quality cooldown tensors require same-filesystem hardlinks; "
                f"cannot link {source} -> {destination}: {error}"
            ) from error
    else:
        raise ValueError(f"quality cooldown hardlink is missing: {destination}")
    source_stat = source.stat()
    destination_stat = destination.stat()
    if (
        source_stat.st_dev != destination_stat.st_dev
        or source_stat.st_ino != destination_stat.st_ino
    ):
        raise ValueError(f"cooldown tensor is not a hardlink to its parent: {destination}")
    if destination_stat.st_size != source_stat.st_size:
        raise ValueError(f"cooldown hardlink size differs from parent: {destination}")
    if sha256_file(destination) != expected_sha256:
        raise ValueError(f"cooldown hardlink SHA256 differs from parent: {destination}")
    with destination.open("rb") as handle:
        os.fsync(handle.fileno())
    return {
        "source": str(source),
        "destination": str(destination),
        "size": destination_stat.st_size,
        "sha256": expected_sha256,
        "device": destination_stat.st_dev,
        "inode": destination_stat.st_ino,
        "same_inode": True,
    }


def _write_or_verify_json(path: Path, payload: Mapping[str, object], label: str) -> None:
    """Write one deterministic JSON artifact, or authenticate a resumed copy."""

    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    if path.exists():
        if not _json_values_equal(_load_json_object(path, label), payload):
            raise ValueError(f"existing {label} differs from the materialization plan: {path}")
        return
    atomic_write_json(path, payload)


def _shard_complete_payload(
    directory: Path,
    *,
    shard_id: str,
    fingerprint: str,
    files: tuple[Path, ...],
    metadata: Mapping[str, object],
) -> dict[str, object]:
    expected_paths = {path.resolve() for path in files}
    unexpected = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if (path.is_file() and path.name != "COMPLETE" and path.resolve() not in expected_paths)
        or path.is_dir()
    }
    if unexpected:
        raise ValueError(f"unexpected files in cooldown shard {directory}: {sorted(unexpected)}")
    outputs = [
        {
            "path": path.name,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(files)
    ]
    return {
        "schema_version": 1,
        "shard_id": shard_id,
        "fingerprint": fingerprint,
        "source_fingerprint": metadata.get("parent_tensor_sha256"),
        "outputs": outputs,
        "metadata": dict(metadata),
    }


def _write_shard_complete(
    directory: Path,
    *,
    shard_id: str,
    fingerprint: str,
    files: tuple[Path, ...],
    metadata: Mapping[str, object],
) -> None:
    marker = _shard_complete_payload(
        directory,
        shard_id=shard_id,
        fingerprint=fingerprint,
        files=files,
        metadata=metadata,
    )
    _write_or_verify_json(directory / "COMPLETE", marker, "cooldown shard COMPLETE")


def _validate_shard_complete(
    directory: Path,
    *,
    shard_id: str,
    fingerprint: str,
    files: tuple[Path, ...],
    metadata: Mapping[str, object],
) -> None:
    marker_path = directory / "COMPLETE"
    if marker_path.is_symlink():
        raise ValueError(f"cooldown shard COMPLETE must not be a symlink: {marker_path}")
    expected = _shard_complete_payload(
        directory,
        shard_id=shard_id,
        fingerprint=fingerprint,
        files=files,
        metadata=metadata,
    )
    marker = _load_json_object(marker_path, "cooldown shard COMPLETE")
    if type(marker.get("schema_version")) is not int or not _json_values_equal(marker, expected):
        raise ValueError(f"cooldown shard COMPLETE metadata changed: {directory}")


def _prepare_plan(
    *,
    primary_prepared: PreparedCorpusManifest,
    primary_kd: TeacherKDCorpusManifest,
    policy: Mapping[str, Any],
    policy_sha256: str,
    primary_prepared_sha256: str,
    primary_kd_sha256: str,
    required_cooldown_tokens: int,
) -> dict[str, Any]:
    if isinstance(primary_prepared.lineage, Mapping) and isinstance(
        primary_prepared.lineage.get("quality_cooldown"), Mapping
    ):
        raise ValueError("nested quality cooldown views are forbidden")
    if (
        not isinstance(primary_prepared.lineage, Mapping)
        or primary_prepared.lineage.get("ready_for_training") is not True
        or primary_prepared.lineage.get("research_only") is not False
        or primary_prepared.lineage.get("pending_audits") != []
    ):
        raise ValueError("primary prepared corpus is not training-ready")
    primary_entries = {entry.shard_id: entry for entry in primary_prepared.shards}
    primary_kd_entries = {entry.source_shard_id: entry for entry in primary_kd.shards}
    authenticated_sources = _authenticated_source_map(primary_prepared)
    if authenticated_sources is None:
        raise ValueError("quality cooldown policy requires authenticated source IDs")
    ordered: list[tuple[PreparedShardEntry, TeacherKDShardEntry, str]] = []
    computed_mix: dict[str, int] = {}
    for item in policy["ordered_shards"]:
        shard_id = item["shard_id"]
        source_id = item["source_id"]
        prepared_entry = primary_entries.get(shard_id)
        kd_entry = primary_kd_entries.get(shard_id)
        if prepared_entry is None or kd_entry is None:
            raise ValueError(f"selection policy names unknown parent shard: {shard_id}")
        prepared_source_path = str(Path(prepared_entry.source_path).expanduser().resolve())
        if authenticated_sources.get(prepared_source_path) != source_id:
            raise ValueError(f"selection policy source_id is not authenticated: {shard_id}")
        if _kd_payload(kd_entry)[:1] != (prepared_entry.tensors_sha256,):
            raise ValueError(f"primary prepared/KD tensor lineage mismatch: {shard_id}")
        computed_mix[source_id] = computed_mix.get(source_id, 0) + prepared_entry.token_count
        ordered.append((prepared_entry, kd_entry, source_id))
    if len(ordered) >= len(primary_prepared.shards):
        raise ValueError("quality cooldown selection must be a strict parent shard subset")
    token_count = sum(item[0].token_count for item in ordered)
    if token_count < required_cooldown_tokens:
        raise ValueError(
            f"quality cooldown selection has {token_count} tokens; "
            f"requires at least {required_cooldown_tokens}"
        )
    if computed_mix != policy["declared_source_mix_token_counts"]:
        raise ValueError("selection policy declared source mix does not match selected shards")
    plan_identity = {
        "schema_version": QUALITY_COOLDOWN_SCHEMA_VERSION,
        "kind": QUALITY_COOLDOWN_BUNDLE_KIND,
        "policy_id": policy["policy_id"],
        "policy_sha256": policy_sha256,
        "parent_prepared_manifest_sha256": primary_prepared_sha256,
        "parent_kd_manifest_sha256": primary_kd_sha256,
        "parent_dataset_fingerprint": primary_prepared.dataset_fingerprint,
        "required_cooldown_tokens": required_cooldown_tokens,
        "ordered_shards": [
            {
                "shard_id": prepared.shard_id,
                "source_id": source_id,
                "prepared_tensor_sha256": prepared.tensors_sha256,
                "kd_tensor_sha256": kd.tensors_sha256,
                "sequence_count": prepared.sequence_count,
                "token_count": prepared.token_count,
            }
            for prepared, kd, source_id in ordered
        ],
        "source_mix_token_counts": computed_mix,
        "sequence_count": sum(item[0].sequence_count for item in ordered),
        "token_count": token_count,
    }
    return {
        "fingerprint": _canonical_sha256(plan_identity),
        "identity": plan_identity,
        "ordered": ordered,
        "source_mix": computed_mix,
    }


def _build_manifests(
    *,
    primary_prepared: PreparedCorpusManifest,
    primary_kd: TeacherKDCorpusManifest,
    plan: Mapping[str, Any],
    policy: Mapping[str, Any],
    policy_sha256: str,
    primary_prepared_sha256: str,
    primary_kd_sha256: str,
) -> tuple[PreparedCorpusManifest, TeacherKDCorpusManifest]:
    lineage = json.loads(json.dumps(primary_prepared.lineage))
    assert isinstance(lineage, dict)
    ordered = plan["ordered"]
    lineage["quality_cooldown"] = {
        "schema_version": QUALITY_COOLDOWN_SCHEMA_VERSION,
        "kind": QUALITY_COOLDOWN_KIND,
        "eligible": True,
        "parent_prepared_manifest_sha256": primary_prepared_sha256,
        "parent_kd_manifest_sha256": primary_kd_sha256,
        "parent_dataset_fingerprint": primary_prepared.dataset_fingerprint,
        "selection_policy_id": policy["policy_id"],
        "selection_policy_sha256": policy_sha256,
        "selection_basis": policy["selection_basis"],
        "required_cooldown_tokens": policy["required_cooldown_tokens"],
        "ordered_parent_shard_ids": [item[0].shard_id for item in ordered],
        "shard_source_ids": {item[0].shard_id: item[2] for item in ordered},
        "source_mix_token_counts": dict(plan["source_mix"]),
    }
    pipeline_fingerprint = _prepared_pipeline_fingerprint(
        [(Path(item[0].source_path), item[0].source_sha256) for item in ordered],
        tokenizer_sha256=primary_prepared.tokenizer_sha256,
        sequence_length=primary_prepared.sequence_length,
        text_field=primary_prepared.text_field,
        generator_source_sha256=primary_prepared.generator_source_sha256,
        lineage=lineage,
    )
    sample_cursor = 0
    token_cursor = 0
    prepared_entries: list[PreparedShardEntry] = []
    kd_entries: list[TeacherKDShardEntry] = []
    for parent_prepared, parent_kd, _source_id in ordered:
        prepared_entry = replace(
            parent_prepared,
            path=f"shards/{parent_prepared.shard_id}",
            global_sample_start=sample_cursor,
            global_sample_end=sample_cursor + parent_prepared.sequence_count,
            global_token_start=token_cursor,
            global_token_end=token_cursor + parent_prepared.token_count,
        )
        kd_entry = replace(
            parent_kd,
            path=f"shards/{parent_prepared.shard_id}",
            global_sample_start=prepared_entry.global_sample_start,
            global_sample_end=prepared_entry.global_sample_end,
            global_token_start=prepared_entry.global_token_start,
            global_token_end=prepared_entry.global_token_end,
        )
        prepared_entries.append(prepared_entry)
        kd_entries.append(kd_entry)
        sample_cursor = prepared_entry.global_sample_end
        token_cursor = prepared_entry.global_token_end
    dataset_fingerprint = _prepared_dataset_fingerprint(
        pipeline_fingerprint=pipeline_fingerprint,
        generator_source_sha256=primary_prepared.generator_source_sha256,
        tokenizer_sha256=primary_prepared.tokenizer_sha256,
        sequence_length=primary_prepared.sequence_length,
        text_field=primary_prepared.text_field,
        shards=prepared_entries,
        lineage=lineage,
    )
    prepared = PreparedCorpusManifest(
        dataset_fingerprint=dataset_fingerprint,
        pipeline_fingerprint=pipeline_fingerprint,
        generator_source_sha256=primary_prepared.generator_source_sha256,
        tokenizer_sha256=primary_prepared.tokenizer_sha256,
        sequence_length=primary_prepared.sequence_length,
        text_field=primary_prepared.text_field,
        shards=tuple(prepared_entries),
        lineage=lineage,
    )
    kd = replace(
        primary_kd,
        dataset_fingerprint=dataset_fingerprint,
        shards=tuple(kd_entries),
    )
    validate_kd_corpus_coverage(kd, prepared)
    return prepared, kd


def _bundle_tree_inventory(
    prepared: PreparedCorpusManifest,
    kd: TeacherKDCorpusManifest,
) -> tuple[set[str], set[str]]:
    expected_files = {
        "STATE.json",
        "manifest.json",
        "COMPLETE",
        "prepared/manifest.json",
        "kd/manifest.json",
    }
    for entry in prepared.shards:
        root = PurePosixPath("prepared") / entry.path
        expected_files.update(
            {
                (root / PREPARED_TENSORS).as_posix(),
                (root / PREPARED_SHARD_MANIFEST).as_posix(),
                (root / "COMPLETE").as_posix(),
            }
        )
    for entry in kd.shards:
        root = PurePosixPath("kd") / entry.path
        expected_files.update(
            {
                (root / KD_TENSORS_FILENAME).as_posix(),
                (root / KD_MANIFEST_FILENAME).as_posix(),
                (root / "COMPLETE").as_posix(),
            }
        )
    expected_directories = {
        parent.as_posix()
        for relative in expected_files
        for parent in PurePosixPath(relative).parents
        if parent.as_posix() != "."
    }
    return expected_files, expected_directories


def _actual_tree_inventory(output: Path) -> tuple[set[str], set[str]]:
    symlinks = [
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_symlink()
    ]
    if symlinks:
        raise ValueError(f"quality cooldown output tree contains symlinks: {sorted(symlinks)}")
    actual_files = {
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    }
    actual_directories = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_dir() and not path.is_symlink()
    }
    return actual_files, actual_directories


def _assert_exact_bundle_tree(
    output: Path,
    prepared: PreparedCorpusManifest,
    kd: TeacherKDCorpusManifest,
) -> None:
    expected_files, expected_directories = _bundle_tree_inventory(prepared, kd)
    actual_files, actual_directories = _actual_tree_inventory(output)
    if actual_files != expected_files or actual_directories != expected_directories:
        raise ValueError(
            "quality cooldown output tree differs from the closed inventory: "
            f"unexpected files={sorted(actual_files - expected_files)}, "
            f"missing files={sorted(expected_files - actual_files)}, "
            f"unexpected directories={sorted(actual_directories - expected_directories)}, "
            f"missing directories={sorted(expected_directories - actual_directories)}"
        )


def _assert_resumable_staging_tree(
    staging: Path,
    prepared: PreparedCorpusManifest,
    kd: TeacherKDCorpusManifest,
) -> None:
    expected_files, expected_directories = _bundle_tree_inventory(prepared, kd)
    # A root bundle without its root COMPLETE marker is an ambiguous late
    # publication failure.  Refuse it instead of overwriting audit metadata.
    expected_files -= {"manifest.json", "COMPLETE"}
    actual_files, actual_directories = _actual_tree_inventory(staging)
    if not actual_files <= expected_files or not actual_directories <= expected_directories:
        raise ValueError("existing cooldown staging tree contains unplanned artifacts")


def _publish_staging(staging: Path, output: Path) -> None:
    os.replace(staging, output)
    directory_fd = os.open(
        output.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _validate_published_bundle(
    output: Path,
    *,
    primary_prepared_path: Path,
    primary_kd_path: Path,
    policy_path: Path,
    required_cooldown_tokens: int,
) -> dict[str, object]:
    bundle_path = output / "manifest.json"
    complete_path = output / "COMPLETE"
    bundle = _load_json_object(bundle_path, "quality cooldown bundle")
    complete = _load_json_object(complete_path, "quality cooldown COMPLETE")
    if (
        type(bundle.get("schema_version")) is not int
        or bundle.get("schema_version") != QUALITY_COOLDOWN_SCHEMA_VERSION
        or bundle.get("kind") != QUALITY_COOLDOWN_BUNDLE_KIND
        or bundle.get("status") != "complete"
        or type(complete.get("schema_version")) is not int
        or complete.get("schema_version") != QUALITY_COOLDOWN_SCHEMA_VERSION
        or complete.get("kind") != QUALITY_COOLDOWN_COMPLETE_KIND
        or complete.get("status") != "complete"
    ):
        raise ValueError("quality cooldown output bundle schema is invalid")
    expected_inputs = {
        "primary_prepared": _file_identity(primary_prepared_path),
        "primary_kd": _file_identity(primary_kd_path),
        "selection_policy": _file_identity(policy_path),
    }
    if not _json_values_equal(bundle.get("inputs"), expected_inputs):
        raise ValueError("quality cooldown output no longer binds its source inputs")
    if bundle.get("required_cooldown_tokens") != required_cooldown_tokens:
        raise ValueError("quality cooldown output token requirement changed")
    if not _json_values_equal(
        complete.get("manifest"),
        _file_identity(bundle_path, relative_to=output),
    ):
        raise ValueError("quality cooldown COMPLETE does not bind the bundle manifest")
    if (
        set(complete)
        != {
            "schema_version",
            "kind",
            "status",
            "manifest",
            "training_started",
            "gpu_kd_started",
        }
        or complete.get("training_started") is not False
        or complete.get("gpu_kd_started") is not False
    ):
        raise ValueError("quality cooldown COMPLETE fields differ from the locked schema")
    prepared_path = output / "prepared/manifest.json"
    kd_path = output / "kd/manifest.json"
    outputs = bundle.get("outputs")
    expected_outputs = {
        "prepared": _file_identity(prepared_path, relative_to=output),
        "teacher_kd": _file_identity(kd_path, relative_to=output),
    }
    if not isinstance(outputs, dict) or not _json_values_equal(outputs, expected_outputs):
        raise ValueError("quality cooldown output manifest identities changed")
    primary_prepared = validate_prepared_corpus(primary_prepared_path)
    declared_primary_kd = TeacherKDCorpusManifest.from_dict(
        _load_json_object(primary_kd_path, "primary KD corpus manifest")
    )
    primary_kd = validate_kd_corpus_manifest(
        primary_kd_path,
        expected_temperature=declared_primary_kd.temperature,
    )
    if bundle.get("temperature") != primary_kd.temperature:
        raise ValueError("quality cooldown bundle KD temperature changed")
    validate_kd_corpus_coverage(primary_kd, primary_prepared)
    cooldown_prepared = validate_prepared_corpus(prepared_path)
    cooldown_kd = validate_kd_corpus_manifest(
        kd_path,
        expected_temperature=primary_kd.temperature,
    )
    validate_kd_corpus_coverage(cooldown_kd, cooldown_prepared)
    summary = validate_quality_cooldown_subset(
        primary_prepared,
        primary_kd,
        cooldown_prepared,
        cooldown_kd,
        primary_prepared_manifest_sha256=sha256_file(primary_prepared_path),
        primary_kd_manifest_sha256=sha256_file(primary_kd_path),
        required_cooldown_tokens=required_cooldown_tokens,
    )
    policy = _validate_policy(
        policy_path,
        primary_prepared_sha256=sha256_file(primary_prepared_path),
        primary_kd_sha256=sha256_file(primary_kd_path),
        required_cooldown_tokens=required_cooldown_tokens,
    )
    policy_sha256 = sha256_file(policy_path)
    plan = _prepare_plan(
        primary_prepared=primary_prepared,
        primary_kd=primary_kd,
        policy=policy,
        policy_sha256=policy_sha256,
        primary_prepared_sha256=sha256_file(primary_prepared_path),
        primary_kd_sha256=sha256_file(primary_kd_path),
        required_cooldown_tokens=required_cooldown_tokens,
    )
    expected_prepared, expected_kd = _build_manifests(
        primary_prepared=primary_prepared,
        primary_kd=primary_kd,
        plan=plan,
        policy=policy,
        policy_sha256=policy_sha256,
        primary_prepared_sha256=sha256_file(primary_prepared_path),
        primary_kd_sha256=sha256_file(primary_kd_path),
    )
    if cooldown_prepared != expected_prepared:
        raise ValueError("quality cooldown prepared manifest differs from its locked policy")
    expected_kd = replace(
        expected_kd,
        shards=tuple(
            replace(
                entry,
                manifest_sha256=sha256_file(kd_path.parent / entry.path / KD_MANIFEST_FILENAME),
            )
            for entry in expected_kd.shards
        ),
    )
    if cooldown_kd != expected_kd:
        raise ValueError("quality cooldown KD manifest differs from its locked policy")
    primary_prepared_by_id = {entry.shard_id: entry for entry in primary_prepared.shards}
    primary_kd_by_id = {entry.source_shard_id: entry for entry in primary_kd.shards}
    for prepared_entry, kd_entry in zip(
        cooldown_prepared.shards,
        cooldown_kd.shards,
        strict=True,
    ):
        parent_prepared = primary_prepared_by_id[prepared_entry.shard_id]
        parent_kd = primary_kd_by_id[prepared_entry.shard_id]
        prepared_directory = prepared_path.parent / prepared_entry.path
        kd_directory = kd_path.parent / kd_entry.path
        _validate_shard_complete(
            prepared_directory,
            shard_id=prepared_entry.shard_id,
            fingerprint=str(plan["fingerprint"]),
            files=(
                prepared_directory / PREPARED_TENSORS,
                prepared_directory / PREPARED_SHARD_MANIFEST,
            ),
            metadata={
                "kind": "quality_cooldown_prepared_hardlink_view",
                "parent_tensor_sha256": parent_prepared.tensors_sha256,
            },
        )
        _validate_shard_complete(
            kd_directory,
            shard_id=prepared_entry.shard_id,
            fingerprint=str(plan["fingerprint"]),
            files=(
                kd_directory / KD_TENSORS_FILENAME,
                kd_directory / KD_MANIFEST_FILENAME,
            ),
            metadata={
                "kind": "quality_cooldown_kd_hardlink_view",
                "parent_tensor_sha256": parent_kd.tensors_sha256,
            },
        )
    expected_scalar_contract = {
        "schema_version": QUALITY_COOLDOWN_SCHEMA_VERSION,
        "kind": QUALITY_COOLDOWN_BUNDLE_KIND,
        "status": "complete",
        "fingerprint": plan["fingerprint"],
        "policy_id": policy["policy_id"],
        "required_cooldown_tokens": required_cooldown_tokens,
        "temperature": primary_kd.temperature,
        "dataset_fingerprint": expected_prepared.dataset_fingerprint,
        "ordered_shard_ids": [entry.shard_id for entry in expected_prepared.shards],
        "source_mix_token_counts": dict(plan["source_mix"]),
        "sequence_count": expected_prepared.sequence_count,
        "token_count": expected_prepared.token_count,
        "training_started": False,
        "gpu_kd_started": False,
    }
    for field, expected in expected_scalar_contract.items():
        if not _json_values_equal(bundle.get(field), expected):
            raise ValueError(f"quality cooldown bundle field changed: {field}")
    expected_bundle_fields = set(expected_scalar_contract) | {
        "inputs",
        "outputs",
        "hardlinks",
    }
    if set(bundle) != expected_bundle_fields:
        raise ValueError("quality cooldown bundle fields differ from the locked schema")
    expected_state = {
        "schema_version": QUALITY_COOLDOWN_SCHEMA_VERSION,
        "kind": QUALITY_COOLDOWN_BUNDLE_KIND,
        "fingerprint": plan["fingerprint"],
        "inputs": expected_inputs,
    }
    state = _load_json_object(output / "STATE.json", "quality cooldown state")
    if type(state.get("schema_version")) is not int or not _json_values_equal(
        state, expected_state
    ):
        raise ValueError("quality cooldown state differs from the locked plan")
    hardlinks = bundle.get("hardlinks")
    if not isinstance(hardlinks, list) or len(hardlinks) != 2 * len(summary.selected_shard_ids):
        raise ValueError("quality cooldown hardlink inventory is incomplete")
    expected_hardlinks: list[tuple[Path, str, str]] = []
    for prepared_entry, kd_entry in zip(cooldown_prepared.shards, cooldown_kd.shards, strict=True):
        parent_prepared = primary_prepared_by_id[prepared_entry.shard_id]
        parent_kd = primary_kd_by_id[prepared_entry.shard_id]
        expected_hardlinks.extend(
            (
                (
                    primary_prepared_path.parent / parent_prepared.path / PREPARED_TENSORS,
                    (PurePosixPath("prepared") / prepared_entry.path / PREPARED_TENSORS).as_posix(),
                    parent_prepared.tensors_sha256,
                ),
                (
                    primary_kd_path.parent / parent_kd.path / KD_TENSORS_FILENAME,
                    (PurePosixPath("kd") / kd_entry.path / KD_TENSORS_FILENAME).as_posix(),
                    parent_kd.tensors_sha256,
                ),
            )
        )
    for item, (expected_source, expected_destination, expected_sha256) in zip(
        hardlinks, expected_hardlinks, strict=True
    ):
        if not isinstance(item, dict):
            raise ValueError("quality cooldown hardlink inventory is invalid")
        if set(item) != {
            "source",
            "destination",
            "size",
            "sha256",
            "device",
            "inode",
            "same_inode",
        }:
            raise ValueError("quality cooldown hardlink fields are invalid")
        source = Path(str(item.get("source", ""))).resolve()
        raw_destination = Path(str(item.get("destination", "")))
        if (
            source != expected_source.resolve()
            or raw_destination.is_absolute()
            or raw_destination.as_posix() != expected_destination
            or item.get("sha256") != expected_sha256
        ):
            raise ValueError("quality cooldown hardlink mapping changed")
        destination = (output / raw_destination).resolve()
        try:
            destination.relative_to(output.resolve())
        except ValueError as error:
            raise ValueError("quality cooldown hardlink escapes output root") from error
        actual = _ensure_hardlink(
            source,
            destination,
            expected_sha256,
            allow_create=False,
        )
        if (
            actual["size"] != item.get("size")
            or actual["device"] != item.get("device")
            or actual["inode"] != item.get("inode")
            or item.get("same_inode") is not True
        ):
            raise ValueError("quality cooldown hardlink identity changed")
    _assert_exact_bundle_tree(output, cooldown_prepared, cooldown_kd)
    return {
        "ok": True,
        "dry_run": False,
        "skipped_existing": True,
        "output_root": str(output),
        "prepared_manifest": str(prepared_path),
        "teacher_kd_manifest": str(kd_path),
        "bundle_manifest": str(bundle_path),
        "complete": str(complete_path),
        "fingerprint": plan["fingerprint"],
        "selection_policy_id": summary.selection_policy_id,
        "dataset_fingerprint": summary.cooldown_dataset_fingerprint,
        "sequence_count": summary.sequence_count,
        "token_count": summary.token_count,
        "selected_shard_ids": list(summary.selected_shard_ids),
        "source_mix_token_counts": dict(summary.source_mix_token_counts),
        "training_started": False,
        "gpu_kd_started": False,
    }


def materialize_quality_cooldown_view(
    *,
    prepared_manifest_path: str | Path,
    kd_manifest_path: str | Path,
    selection_policy_path: str | Path,
    output_root: str | Path,
    required_cooldown_tokens: int,
    dry_run: bool = False,
) -> dict[str, object]:
    """Plan or atomically publish a hard-linked whole-shard cooldown view."""

    if (
        isinstance(required_cooldown_tokens, bool)
        or not isinstance(required_cooldown_tokens, int)
        or required_cooldown_tokens <= 0
    ):
        raise ValueError("required_cooldown_tokens must be a positive integer")
    prepared_path = Path(prepared_manifest_path).expanduser().resolve()
    kd_path = Path(kd_manifest_path).expanduser().resolve()
    policy_path = Path(selection_policy_path).expanduser().resolve()
    requested_output = Path(output_root).expanduser()
    if requested_output.is_symlink():
        raise ValueError("quality cooldown output root must not be a symlink")
    output = requested_output.resolve()
    _assert_safe_output(
        output,
        prepared_manifest=prepared_path,
        kd_manifest=kd_path,
        policy_path=policy_path,
    )
    primary_prepared_sha = sha256_file(prepared_path)
    primary_kd_sha = sha256_file(kd_path)
    primary_prepared = validate_prepared_corpus(prepared_path)
    # Learn the declared temperature from the corpus object itself.  Never
    # follow an unauthenticated shard path merely to discover this value: a
    # forged corpus entry could otherwise make pre-validation read outside the
    # corpus root.
    declared_primary_kd = TeacherKDCorpusManifest.from_dict(
        _load_json_object(kd_path, "primary KD corpus manifest")
    )
    primary_kd = validate_kd_corpus_manifest(
        kd_path,
        expected_temperature=declared_primary_kd.temperature,
    )
    validate_kd_corpus_coverage(primary_kd, primary_prepared)
    policy = _validate_policy(
        policy_path,
        primary_prepared_sha256=primary_prepared_sha,
        primary_kd_sha256=primary_kd_sha,
        required_cooldown_tokens=required_cooldown_tokens,
    )
    policy_sha = sha256_file(policy_path)
    plan = _prepare_plan(
        primary_prepared=primary_prepared,
        primary_kd=primary_kd,
        policy=policy,
        policy_sha256=policy_sha,
        primary_prepared_sha256=primary_prepared_sha,
        primary_kd_sha256=primary_kd_sha,
        required_cooldown_tokens=required_cooldown_tokens,
    )
    prepared, kd = _build_manifests(
        primary_prepared=primary_prepared,
        primary_kd=primary_kd,
        plan=plan,
        policy=policy,
        policy_sha256=policy_sha,
        primary_prepared_sha256=primary_prepared_sha,
        primary_kd_sha256=primary_kd_sha,
    )
    planned = {
        "ok": True,
        "dry_run": bool(dry_run),
        "skipped_existing": False,
        "output_root": str(output),
        "fingerprint": plan["fingerprint"],
        "selection_policy_id": policy["policy_id"],
        "dataset_fingerprint": prepared.dataset_fingerprint,
        "sequence_count": prepared.sequence_count,
        "token_count": prepared.token_count,
        "selected_shard_ids": [entry.shard_id for entry in prepared.shards],
        "source_mix_token_counts": dict(plan["source_mix"]),
        "would_hardlink_files": 2 * len(prepared.shards),
        "training_started": False,
        "gpu_kd_started": False,
    }
    if dry_run:
        return planned

    lock = FileLock(output.parent / f".{output.name}.materialize-cooldown.lock")
    with lock:
        if output.exists():
            if not output.is_dir():
                raise ValueError(f"quality cooldown output exists and is not a directory: {output}")
            return _validate_published_bundle(
                output,
                primary_prepared_path=prepared_path,
                primary_kd_path=kd_path,
                policy_path=policy_path,
                required_cooldown_tokens=required_cooldown_tokens,
            )
        staging = output.with_name(f".{output.name}.incomplete")
        staging_existed = staging.exists()
        if staging.is_symlink():
            raise ValueError(f"quality cooldown staging path must not be a symlink: {staging}")
        if staging_existed and not staging.is_dir():
            raise ValueError(f"quality cooldown staging path is not a directory: {staging}")
        if not staging_existed:
            staging.mkdir(parents=True)
        state_path = staging / "STATE.json"
        expected_state = {
            "schema_version": QUALITY_COOLDOWN_SCHEMA_VERSION,
            "kind": QUALITY_COOLDOWN_BUNDLE_KIND,
            "fingerprint": plan["fingerprint"],
            "inputs": {
                "primary_prepared": _file_identity(prepared_path),
                "primary_kd": _file_identity(kd_path),
                "selection_policy": _file_identity(policy_path),
            },
        }
        if state_path.exists():
            if _load_json_object(state_path, "quality cooldown staging state") != expected_state:
                raise ValueError("existing cooldown staging output belongs to another plan")
        elif staging_existed:
            raise ValueError("existing cooldown staging output has no authenticated state")
        else:
            atomic_write_json(state_path, expected_state)

        if (staging / "COMPLETE").exists():
            _validate_published_bundle(
                staging,
                primary_prepared_path=prepared_path,
                primary_kd_path=kd_path,
                policy_path=policy_path,
                required_cooldown_tokens=required_cooldown_tokens,
            )
            _publish_staging(staging, output)
            result = _validate_published_bundle(
                output,
                primary_prepared_path=prepared_path,
                primary_kd_path=kd_path,
                policy_path=policy_path,
                required_cooldown_tokens=required_cooldown_tokens,
            )
            result["skipped_existing"] = False
            return result
        _assert_resumable_staging_tree(staging, prepared, kd)

        prepared_root = staging / "prepared"
        kd_root = staging / "kd"
        hardlinks: list[dict[str, object]] = []
        parent_prepared_by_id = {entry.shard_id: entry for entry in primary_prepared.shards}
        parent_kd_by_id = {entry.source_shard_id: entry for entry in primary_kd.shards}
        for prepared_entry, kd_entry in zip(prepared.shards, kd.shards, strict=True):
            parent_prepared = parent_prepared_by_id[prepared_entry.shard_id]
            parent_kd = parent_kd_by_id[prepared_entry.shard_id]
            source_prepared_dir = prepared_path.parent / parent_prepared.path
            destination_prepared_dir = prepared_root / prepared_entry.path
            prepared_link = _ensure_hardlink(
                source_prepared_dir / PREPARED_TENSORS,
                destination_prepared_dir / PREPARED_TENSORS,
                prepared_entry.tensors_sha256,
            )
            local_prepared = _local_prepared_manifest(
                shard_id=prepared_entry.shard_id,
                source_path=prepared_entry.source_path,
                source_sha256=prepared_entry.source_sha256,
                sequence_count=prepared_entry.sequence_count,
                token_count=prepared_entry.token_count,
                tensors_sha256=prepared_entry.tensors_sha256,
                pipeline_fingerprint=prepared.pipeline_fingerprint,
                generator_source_sha256=prepared.generator_source_sha256,
                tokenizer_sha256=prepared.tokenizer_sha256,
                sequence_length=prepared.sequence_length,
                text_field=prepared.text_field,
            )
            _write_or_verify_json(
                destination_prepared_dir / PREPARED_SHARD_MANIFEST,
                local_prepared,
                "cooldown local prepared manifest",
            )
            _write_shard_complete(
                destination_prepared_dir,
                shard_id=prepared_entry.shard_id,
                fingerprint=plan["fingerprint"],
                files=(
                    destination_prepared_dir / PREPARED_TENSORS,
                    destination_prepared_dir / PREPARED_SHARD_MANIFEST,
                ),
                metadata={
                    "kind": "quality_cooldown_prepared_hardlink_view",
                    "parent_tensor_sha256": parent_prepared.tensors_sha256,
                },
            )
            prepared_link["destination"] = (
                Path(str(prepared_link["destination"])).relative_to(staging).as_posix()
            )
            hardlinks.append(prepared_link)

            source_kd_dir = kd_path.parent / parent_kd.path
            destination_kd_dir = kd_root / kd_entry.path
            kd_link = _ensure_hardlink(
                source_kd_dir / KD_TENSORS_FILENAME,
                destination_kd_dir / KD_TENSORS_FILENAME,
                kd_entry.tensors_sha256,
            )
            parent_local_kd = read_kd_manifest(source_kd_dir)
            local_kd = replace(
                parent_local_kd,
                dataset_fingerprint=prepared.dataset_fingerprint,
                global_sample_start=kd_entry.global_sample_start,
                global_sample_end=kd_entry.global_sample_end,
                global_token_start=kd_entry.global_token_start,
                global_token_end=kd_entry.global_token_end,
            )
            _write_or_verify_json(
                destination_kd_dir / KD_MANIFEST_FILENAME,
                local_kd.to_dict(),
                "cooldown local KD manifest",
            )
            _write_shard_complete(
                destination_kd_dir,
                shard_id=prepared_entry.shard_id,
                fingerprint=plan["fingerprint"],
                files=(
                    destination_kd_dir / KD_TENSORS_FILENAME,
                    destination_kd_dir / KD_MANIFEST_FILENAME,
                ),
                metadata={
                    "kind": "quality_cooldown_kd_hardlink_view",
                    "parent_tensor_sha256": parent_kd.tensors_sha256,
                },
            )
            kd_link["destination"] = (
                Path(str(kd_link["destination"])).relative_to(staging).as_posix()
            )
            hardlinks.append(kd_link)

        prepared_manifest = prepared_root / "manifest.json"
        kd_manifest = kd_root / "manifest.json"
        _write_or_verify_json(
            prepared_manifest,
            prepared.to_dict(),
            "cooldown prepared corpus manifest",
        )
        kd_entries_with_manifest_sha = tuple(
            replace(
                entry,
                manifest_sha256=sha256_file(kd_root / entry.path / KD_MANIFEST_FILENAME),
            )
            for entry in kd.shards
        )
        kd = replace(kd, shards=kd_entries_with_manifest_sha)
        _write_or_verify_json(
            kd_manifest,
            kd.to_dict(),
            "cooldown KD corpus manifest",
        )
        bundle_path = staging / "manifest.json"
        bundle = {
            "schema_version": QUALITY_COOLDOWN_SCHEMA_VERSION,
            "kind": QUALITY_COOLDOWN_BUNDLE_KIND,
            "status": "complete",
            "fingerprint": plan["fingerprint"],
            "policy_id": policy["policy_id"],
            "required_cooldown_tokens": required_cooldown_tokens,
            "temperature": primary_kd.temperature,
            "inputs": expected_state["inputs"],
            "outputs": {
                "prepared": _file_identity(prepared_manifest, relative_to=staging),
                "teacher_kd": _file_identity(kd_manifest, relative_to=staging),
            },
            "dataset_fingerprint": prepared.dataset_fingerprint,
            "ordered_shard_ids": [entry.shard_id for entry in prepared.shards],
            "source_mix_token_counts": dict(plan["source_mix"]),
            "sequence_count": prepared.sequence_count,
            "token_count": prepared.token_count,
            "hardlinks": hardlinks,
            "training_started": False,
            "gpu_kd_started": False,
        }
        _write_or_verify_json(bundle_path, bundle, "quality cooldown bundle")
        _write_or_verify_json(
            staging / "COMPLETE",
            {
                "schema_version": QUALITY_COOLDOWN_SCHEMA_VERSION,
                "kind": QUALITY_COOLDOWN_COMPLETE_KIND,
                "status": "complete",
                "manifest": _file_identity(bundle_path, relative_to=staging),
                "training_started": False,
                "gpu_kd_started": False,
            },
            "quality cooldown COMPLETE",
        )
        # Validate from the staging root before its single atomic publication.
        _validate_published_bundle(
            staging,
            primary_prepared_path=prepared_path,
            primary_kd_path=kd_path,
            policy_path=policy_path,
            required_cooldown_tokens=required_cooldown_tokens,
        )
        _publish_staging(staging, output)
        result = _validate_published_bundle(
            output,
            primary_prepared_path=prepared_path,
            primary_kd_path=kd_path,
            policy_path=policy_path,
            required_cooldown_tokens=required_cooldown_tokens,
        )
        result["skipped_existing"] = False
        return result
