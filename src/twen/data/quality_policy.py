"""Deterministic, fail-closed Base-v2 quality-cooldown policy generation.

The generator only selects authenticated whole prepared/KD shards.  It never
materializes tensors, runs teacher inference, imports torch, or starts training.
Dry planning is the default; publishing requires an explicit approval flag.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..io.locking import FileLock
from ..utils import atomic_write_json, atomic_write_text, sha256_file
from .cooldown import (
    QUALITY_COOLDOWN_POLICY_KIND,
    QUALITY_COOLDOWN_SCHEMA_VERSION,
    _authenticated_source_map,
)
from .prepared import PreparedCorpusManifest, PreparedShardEntry, validate_prepared_corpus
from .teacher_kd import (
    TeacherKDCorpusManifest,
    validate_kd_corpus_coverage,
    validate_kd_corpus_manifest,
)

QUALITY_POLICY_AUDIT_KIND = "twen_quality_cooldown_selection_audit"
QUALITY_POLICY_BUNDLE_KIND = "twen_quality_cooldown_policy_bundle"
QUALITY_POLICY_COMPLETE_KIND = "twen_quality_cooldown_policy_complete"
QUALITY_POLICY_SCHEMA_VERSION = 1
QUALITY_POLICY_FILENAME = "quality-cooldown-policy.json"
QUALITY_POLICY_AUDIT_FILENAME = "AUDIT.json"
QUALITY_POLICY_REPORT_FILENAME = "REPORT.md"
QUALITY_POLICY_MANIFEST_FILENAME = "MANIFEST.json"
QUALITY_POLICY_COMPLETE_FILENAME = "COMPLETE"
DEFAULT_QUALITY_POLICY_ID = "base-v2-50m-quality-cooldown-v1"
DEFAULT_QUALITY_POLICY_SEED = "twen-base-v2-quality-cooldown-seed-v1"
DEFAULT_QUALITY_SOURCE_TOKEN_TARGETS: tuple[tuple[str, int], ...] = (
    ("english_fineweb_edu_dedup", 15_000_000),
    ("math_finemath_4plus", 15_000_000),
    ("code_github_clean_allowlisted", 7_500_000),
    ("chinese_fineweb2_cmn_hani", 5_000_000),
    ("science_cosmopedia_openstax", 5_000_000),
    ("science_cosmopedia_stanford", 2_500_000),
)
DEFAULT_QUALITY_COOLDOWN_TOKENS = sum(target for _, target in DEFAULT_QUALITY_SOURCE_TOKEN_TARGETS)
_SAFE_POLICY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_SELECTION_SEED = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SHARD_IDENTITY_FIELDS = (
    "parent_dataset_fingerprint",
    "shard_id",
    "source_id",
    "source_sha256",
    "tensors_sha256",
    "sequence_count",
    "token_count",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _normalize_targets(
    targets: Sequence[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    if not targets:
        raise ValueError("quality cooldown source targets cannot be empty")
    normalized: list[tuple[str, int]] = []
    seen: set[str] = set()
    for index, item in enumerate(targets):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"quality cooldown source target {index} is invalid")
        source_id, token_target = item
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"quality cooldown source target {index} has no source_id")
        if source_id in seen:
            raise ValueError(f"duplicate quality cooldown source target: {source_id}")
        if isinstance(token_target, bool) or not isinstance(token_target, int) or token_target <= 0:
            raise ValueError(f"quality cooldown token target must be positive: {source_id}")
        seen.add(source_id)
        normalized.append((source_id, token_target))
    return tuple(normalized)


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _paths_overlap(first: Path, second: Path) -> bool:
    left = first.resolve()
    right = second.resolve()
    return left == right or left in right.parents or right in left.parents


def _assert_safe_output(
    output: Path,
    *,
    prepared_manifest: Path,
    kd_manifest: Path,
) -> None:
    if not output.name or output == output.parent:
        raise ValueError("quality cooldown policy output path is unsafe")
    if _path_lexists(output):
        raise ValueError(
            f"quality cooldown policy output already exists; refusing overwrite: {output}"
        )
    for label, source in (
        ("prepared", prepared_manifest.parent),
        ("KD", kd_manifest.parent),
    ):
        if _paths_overlap(output, source):
            raise ValueError(f"quality cooldown policy output overlaps {label} inputs: {source}")


def _manifest_identity(path: Path, expected_sha256: str) -> dict[str, object]:
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": expected_sha256,
    }


def _shard_identity(
    entry: PreparedShardEntry,
    *,
    source_id: str,
    parent_dataset_fingerprint: str,
) -> dict[str, object]:
    return {
        "parent_dataset_fingerprint": parent_dataset_fingerprint,
        "shard_id": entry.shard_id,
        "source_id": source_id,
        "source_sha256": entry.source_sha256,
        "tensors_sha256": entry.tensors_sha256,
        "sequence_count": entry.sequence_count,
        "token_count": entry.token_count,
    }


def _ordering_hash(
    identity: Mapping[str, object],
    *,
    seed: str,
    scope: str,
) -> str:
    return _canonical_sha256(
        {
            "version": "sha256-canonical-shard-identity-v1",
            "seed": seed,
            "scope": scope,
            "shard_identity": dict(identity),
        }
    )


def _read_declared_kd_manifest(path: Path) -> TeacherKDCorpusManifest:
    return TeacherKDCorpusManifest.from_dict(_load_json_object(path, "parent KD manifest"))


def _authenticate_inputs(
    prepared_path: Path,
    kd_path: Path,
) -> tuple[
    PreparedCorpusManifest,
    TeacherKDCorpusManifest,
    dict[str, str],
    dict[str, object],
]:
    if prepared_path.is_symlink() or kd_path.is_symlink():
        raise ValueError("quality cooldown parent manifests must not be symlinks")
    try:
        prepared_sha = sha256_file(prepared_path)
        kd_sha = sha256_file(kd_path)
    except OSError as error:
        raise ValueError("quality cooldown parent prepared/KD manifest is missing") from error

    prepared = validate_prepared_corpus(prepared_path)
    declared_kd = _read_declared_kd_manifest(kd_path)
    kd = validate_kd_corpus_manifest(
        kd_path,
        expected_temperature=declared_kd.temperature,
    )
    validate_kd_corpus_coverage(kd, prepared)
    if sha256_file(prepared_path) != prepared_sha or sha256_file(kd_path) != kd_sha:
        raise ValueError("quality cooldown parent manifest SHA changed during validation")

    lineage = prepared.lineage
    if (
        not isinstance(lineage, Mapping)
        or lineage.get("kind") != "authenticated_extracted_corpus"
        or lineage.get("role") != "train"
        or lineage.get("ready_for_training") is not True
        or lineage.get("research_only") is not False
        or lineage.get("pending_audits") != []
        or isinstance(lineage.get("quality_cooldown"), Mapping)
    ):
        raise ValueError("quality cooldown parent prepared lineage is not authenticated train data")
    authenticated_sources = _authenticated_source_map(prepared)
    if authenticated_sources is None:
        raise ValueError("quality cooldown parent has no authenticated source-ID mapping")

    source_by_shard: dict[str, str] = {}
    seen_prepared: set[str] = set()
    for entry in prepared.shards:
        if entry.shard_id in seen_prepared:
            raise ValueError(f"duplicate parent prepared shard: {entry.shard_id}")
        seen_prepared.add(entry.shard_id)
        source_path = str(Path(entry.source_path).expanduser().resolve())
        source_id = authenticated_sources.get(source_path)
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"prepared shard source_id is not authenticated: {entry.shard_id}")
        source_by_shard[entry.shard_id] = source_id

    seen_kd: set[str] = set()
    for entry in kd.shards:
        if entry.source_shard_id in seen_kd:
            raise ValueError(f"duplicate parent KD shard: {entry.source_shard_id}")
        seen_kd.add(entry.source_shard_id)
    if seen_kd != seen_prepared:
        raise ValueError("parent prepared/KD shard identities differ")

    extracted_path_value = lineage.get("extracted_manifest_path")
    extracted_sha = lineage.get("extracted_manifest_sha256")
    if not isinstance(extracted_path_value, str) or not isinstance(extracted_sha, str):
        raise ValueError("quality cooldown extracted lineage identity is incomplete")
    extracted_path = Path(extracted_path_value).expanduser().resolve()
    if sha256_file(extracted_path) != extracted_sha:
        raise ValueError("quality cooldown extracted lineage manifest SHA changed")
    inputs = {
        "prepared_manifest": _manifest_identity(prepared_path, prepared_sha),
        "kd_manifest": _manifest_identity(kd_path, kd_sha),
        "extracted_manifest": _manifest_identity(extracted_path, extracted_sha),
        "parent_dataset_fingerprint": prepared.dataset_fingerprint,
        "parent_prepared_shard_count": len(prepared.shards),
        "parent_prepared_token_count": prepared.token_count,
        "parent_kd_shard_count": len(kd.shards),
        "parent_kd_token_count": kd.token_count,
    }
    return prepared, kd, source_by_shard, inputs


def _source_rows(
    *,
    targets: tuple[tuple[str, int], ...],
    prepared: PreparedCorpusManifest,
    source_by_shard: Mapping[str, str],
    selected: Sequence[dict[str, Any]],
) -> list[dict[str, object]]:
    selected_by_source: dict[str, list[dict[str, Any]]] = {
        source_id: [] for source_id, _ in targets
    }
    for item in selected:
        selected_by_source[item["source_id"]].append(item)
    rows: list[dict[str, object]] = []
    required_tokens = sum(target for _, target in targets)
    actual_tokens = sum(int(item["token_count"]) for item in selected)
    for source_id, target in targets:
        candidates = [
            entry for entry in prepared.shards if source_by_shard[entry.shard_id] == source_id
        ]
        chosen = selected_by_source[source_id]
        selected_tokens = sum(int(item["token_count"]) for item in chosen)
        rows.append(
            {
                "source_id": source_id,
                "target_tokens": target,
                "candidate_tokens": sum(entry.token_count for entry in candidates),
                "selected_tokens": selected_tokens,
                "overshoot_tokens": selected_tokens - target,
                "target_share_percent": target / required_tokens * 100.0,
                "selected_share_percent": selected_tokens / actual_tokens * 100.0,
                "candidate_shard_count": len(candidates),
                "selected_shard_count": len(chosen),
            }
        )
    return rows


def _select_shards(
    prepared: PreparedCorpusManifest,
    source_by_shard: Mapping[str, str],
    *,
    targets: tuple[tuple[str, int], ...],
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
    selected: list[dict[str, Any]] = []
    for source_id, token_target in targets:
        ranked: list[dict[str, Any]] = []
        for entry in prepared.shards:
            if source_by_shard[entry.shard_id] != source_id:
                continue
            identity = _shard_identity(
                entry,
                source_id=source_id,
                parent_dataset_fingerprint=prepared.dataset_fingerprint,
            )
            ranked.append(
                {
                    "entry": entry,
                    "identity": identity,
                    "identity_sha256": _canonical_sha256(identity),
                    "source_order_sha256": _ordering_hash(
                        identity,
                        seed=seed,
                        scope=f"source:{source_id}",
                    ),
                }
            )
        candidate_tokens = sum(item["entry"].token_count for item in ranked)
        if candidate_tokens < token_target:
            raise ValueError(
                f"quality cooldown source {source_id} has {candidate_tokens} tokens; "
                f"requires at least {token_target}"
            )
        ranked.sort(key=lambda item: (item["source_order_sha256"], item["entry"].shard_id))
        accumulated = 0
        for source_rank, item in enumerate(ranked, start=1):
            if accumulated >= token_target:
                break
            entry = item["entry"]
            global_order_sha = _ordering_hash(
                item["identity"],
                seed=seed,
                scope="global",
            )
            selected.append(
                {
                    "shard_id": entry.shard_id,
                    "source_id": source_id,
                    "sequence_count": entry.sequence_count,
                    "token_count": entry.token_count,
                    "identity_sha256": item["identity_sha256"],
                    "source_rank": source_rank,
                    "source_order_sha256": item["source_order_sha256"],
                    "global_order_sha256": global_order_sha,
                }
            )
            accumulated += entry.token_count

    selected_ids = [item["shard_id"] for item in selected]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("quality cooldown selection contains duplicate parent shards")
    if len(selected) >= len(prepared.shards):
        raise ValueError("quality cooldown selection must be a strict parent shard subset")
    selected.sort(key=lambda item: (item["global_order_sha256"], item["shard_id"]))
    for index, item in enumerate(selected):
        item["global_order"] = index
    rows = _source_rows(
        targets=targets,
        prepared=prepared,
        source_by_shard=source_by_shard,
        selected=selected,
    )
    return selected, rows


def _build_policy(
    *,
    policy_id: str,
    approved: bool,
    selection_basis: str,
    inputs: Mapping[str, object],
    required_tokens: int,
    selected: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    prepared_identity = inputs["prepared_manifest"]
    kd_identity = inputs["kd_manifest"]
    assert isinstance(prepared_identity, Mapping)
    assert isinstance(kd_identity, Mapping)
    return {
        "schema_version": QUALITY_COOLDOWN_SCHEMA_VERSION,
        "kind": QUALITY_COOLDOWN_POLICY_KIND,
        "policy_id": policy_id,
        "approved_for_quality_cooldown": approved,
        "selection_basis": selection_basis,
        "parent_prepared_manifest_sha256": prepared_identity["sha256"],
        "parent_kd_manifest_sha256": kd_identity["sha256"],
        "required_cooldown_tokens": required_tokens,
        "ordered_shards": [
            {"shard_id": item["shard_id"], "source_id": item["source_id"]} for item in selected
        ],
        "declared_source_mix_token_counts": {
            str(row["source_id"]): row["selected_tokens"] for row in source_rows
        },
    }


def _build_audit(
    *,
    policy_id: str,
    approved: bool,
    seed: str,
    selection_basis: str,
    inputs: Mapping[str, object],
    targets: tuple[tuple[str, int], ...],
    selected: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    required_tokens = sum(target for _, target in targets)
    actual_tokens = sum(int(item["token_count"]) for item in selected)
    selected_sequence_count = sum(int(item["sequence_count"]) for item in selected)
    selection_contract = {
        "policy_id": policy_id,
        "seed": seed,
        "inputs": dict(inputs),
        "source_targets": [
            {"source_id": source_id, "target_tokens": target} for source_id, target in targets
        ],
        "selected_shards": [dict(item) for item in selected],
    }
    return {
        "schema_version": QUALITY_POLICY_SCHEMA_VERSION,
        "kind": QUALITY_POLICY_AUDIT_KIND,
        "policy_id": policy_id,
        "approved_for_quality_cooldown": approved,
        "selection_plan_sha256": _canonical_sha256(selection_contract),
        "selection_basis": selection_basis,
        "selection_rule": {
            "version": "per-source-hash-quota-then-global-hash-v1",
            "seed": seed,
            "whole_shard": True,
            "per_source_order": "ascending SHA256 of canonical shard identity with source scope",
            "per_source_stop": "first ranked whole-shard prefix reaching target_tokens",
            "global_order": "ascending SHA256 of canonical shard identity with global scope",
            "tie_breaker": "ascending shard_id",
            "shard_identity_fields": list(_SHARD_IDENTITY_FIELDS),
        },
        "inputs": dict(inputs),
        "required_cooldown_tokens": required_tokens,
        "actual_cooldown_tokens": actual_tokens,
        "overshoot_tokens": actual_tokens - required_tokens,
        "selected_sequence_count": selected_sequence_count,
        "selected_shard_count": len(selected),
        "source_results": [dict(row) for row in source_rows],
        "selected_shards": [dict(item) for item in selected],
        "checks": {
            "prepared_manifest_fully_validated": True,
            "kd_manifest_fully_validated": True,
            "prepared_kd_exact_coverage": True,
            "extracted_lineage_authenticated": True,
            "parent_manifest_sha_stable": True,
            "source_quotas_met": True,
            "parent_shards_unique": True,
            "selection_is_strict_whole_shard_subset": True,
        },
        "training_started": False,
        "teacher_kd_started": False,
    }


def _render_report(audit: Mapping[str, object]) -> str:
    source_results = audit["source_results"]
    assert isinstance(source_results, list)
    approved = audit["approved_for_quality_cooldown"] is True
    lines = [
        "# Base v2 quality-cooldown selection audit",
        "",
        f"- Policy ID: `{audit['policy_id']}`",
        f"- Status: **{'approved and published' if approved else 'dry plan only'}**",
        f"- Selection plan SHA256: `{audit['selection_plan_sha256']}`",
        f"- Required tokens: {int(audit['required_cooldown_tokens']):,}",
        f"- Actual whole-shard tokens: {int(audit['actual_cooldown_tokens']):,}",
        f"- Overall overshoot: {int(audit['overshoot_tokens']):,}",
        f"- Selected shards: {int(audit['selected_shard_count']):,}",
        "- Training started: **false**; teacher KD started: **false**.",
        "",
        "## Source quotas",
        "",
        "| source | target | actual | overshoot | actual share | shards |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for raw in source_results:
        assert isinstance(raw, Mapping)
        lines.append(
            "| {source} | {target:,} | {actual:,} | {overshoot:,} | "
            "{share:.4f}% | {shards:,} |".format(
                source=raw["source_id"],
                target=int(raw["target_tokens"]),
                actual=int(raw["selected_tokens"]),
                overshoot=int(raw["overshoot_tokens"]),
                share=float(raw["selected_share_percent"]),
                shards=int(raw["selected_shard_count"]),
            )
        )
    rule = audit["selection_rule"]
    assert isinstance(rule, Mapping)
    lines.extend(
        [
            "",
            "## Locked rule",
            "",
            f"- Version: `{rule['version']}`",
            f"- Seed: `{rule['seed']}`",
            "- Each source independently sorts authenticated shard identities by SHA256, "
            "then takes the shortest whole-shard prefix meeting that source quota.",
            "- The union is sorted once more by a separate global-scope SHA256.",
            "- Whole-shard overshoot is retained; no shard is split or reweighted.",
            "",
            "`MANIFEST.json` authenticates the policy, this audit JSON, and this report. "
            "`COMPLETE` binds the manifest. This operation does not run KD or training.",
            "",
        ]
    )
    return "\n".join(lines)


def _assert_input_shas_stable(inputs: Mapping[str, object]) -> None:
    for label in ("prepared_manifest", "kd_manifest", "extracted_manifest"):
        raw = inputs[label]
        if not isinstance(raw, Mapping):
            raise ValueError(f"quality cooldown input identity is invalid: {label}")
        path = Path(str(raw["path"]))
        if path.is_symlink() or not path.is_file() or sha256_file(path) != raw["sha256"]:
            raise ValueError(f"quality cooldown input SHA changed before publication: {label}")


def _file_identity(path: Path, *, relative_to: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _publish_bundle(
    output: Path,
    *,
    policy: Mapping[str, object],
    audit: Mapping[str, object],
    report: str,
    inputs: Mapping[str, object],
) -> dict[str, object]:
    lock_path = output.parent / f".{output.name}.quality-policy.lock"
    staging = output.parent / f".{output.name}.quality-policy.incomplete"
    if lock_path.is_symlink() or staging.is_symlink():
        raise ValueError("quality cooldown policy lock/staging path must not be a symlink")
    with FileLock(lock_path):
        _assert_input_shas_stable(inputs)
        if _path_lexists(output):
            raise ValueError(
                f"quality cooldown policy output already exists; refusing overwrite: {output}"
            )
        if _path_lexists(staging):
            raise ValueError(f"quality cooldown policy staging path already exists: {staging}")
        staging.mkdir(mode=0o700)
        policy_path = staging / QUALITY_POLICY_FILENAME
        audit_path = staging / QUALITY_POLICY_AUDIT_FILENAME
        report_path = staging / QUALITY_POLICY_REPORT_FILENAME
        atomic_write_json(policy_path, dict(policy))
        atomic_write_json(audit_path, dict(audit))
        atomic_write_text(report_path, report)
        files = [
            _file_identity(path, relative_to=staging)
            for path in (policy_path, audit_path, report_path)
        ]
        manifest = {
            "schema_version": QUALITY_POLICY_SCHEMA_VERSION,
            "kind": QUALITY_POLICY_BUNDLE_KIND,
            "policy_id": policy["policy_id"],
            "selection_plan_sha256": audit["selection_plan_sha256"],
            "approved_for_quality_cooldown": True,
            "files": files,
            "training_started": False,
            "teacher_kd_started": False,
        }
        manifest_path = staging / QUALITY_POLICY_MANIFEST_FILENAME
        atomic_write_json(manifest_path, manifest)
        complete = {
            "schema_version": QUALITY_POLICY_SCHEMA_VERSION,
            "kind": QUALITY_POLICY_COMPLETE_KIND,
            "policy_id": policy["policy_id"],
            "manifest": QUALITY_POLICY_MANIFEST_FILENAME,
            "manifest_sha256": sha256_file(manifest_path),
            "approved_for_quality_cooldown": True,
            "training_started": False,
            "teacher_kd_started": False,
        }
        atomic_write_json(staging / QUALITY_POLICY_COMPLETE_FILENAME, complete)
        expected_files = {
            QUALITY_POLICY_FILENAME,
            QUALITY_POLICY_AUDIT_FILENAME,
            QUALITY_POLICY_REPORT_FILENAME,
            QUALITY_POLICY_MANIFEST_FILENAME,
            QUALITY_POLICY_COMPLETE_FILENAME,
        }
        actual_files = {path.name for path in staging.iterdir() if path.is_file()}
        if actual_files != expected_files or any(path.is_dir() for path in staging.iterdir()):
            raise ValueError("quality cooldown policy staging tree differs from closed schema")
        _assert_input_shas_stable(inputs)
        os.rename(staging, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return {
        "policy": _file_identity(output / QUALITY_POLICY_FILENAME, relative_to=output),
        "audit": _file_identity(output / QUALITY_POLICY_AUDIT_FILENAME, relative_to=output),
        "report": _file_identity(output / QUALITY_POLICY_REPORT_FILENAME, relative_to=output),
        "manifest": _file_identity(
            output / QUALITY_POLICY_MANIFEST_FILENAME,
            relative_to=output,
        ),
        "complete": _file_identity(
            output / QUALITY_POLICY_COMPLETE_FILENAME,
            relative_to=output,
        ),
    }


def generate_quality_cooldown_policy(
    *,
    prepared_manifest_path: str | Path,
    kd_manifest_path: str | Path,
    output_root: str | Path,
    approve: bool = False,
    policy_id: str = DEFAULT_QUALITY_POLICY_ID,
    seed: str = DEFAULT_QUALITY_POLICY_SEED,
    source_token_targets: Sequence[tuple[str, int]] = DEFAULT_QUALITY_SOURCE_TOKEN_TARGETS,
) -> dict[str, object]:
    """Plan or explicitly publish a deterministic Base-v2 cooldown policy bundle."""

    if not isinstance(approve, bool):
        raise ValueError("quality cooldown approve flag must be boolean")
    if not isinstance(policy_id, str) or _SAFE_POLICY_ID.fullmatch(policy_id) is None:
        raise ValueError("quality cooldown policy_id is invalid")
    if not isinstance(seed, str) or _SAFE_SELECTION_SEED.fullmatch(seed) is None:
        raise ValueError("quality cooldown selection seed is invalid")
    targets = _normalize_targets(source_token_targets)
    prepared_path = Path(prepared_manifest_path).expanduser().resolve()
    kd_path = Path(kd_manifest_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    _assert_safe_output(
        output,
        prepared_manifest=prepared_path,
        kd_manifest=kd_path,
    )
    prepared, _kd, source_by_shard, inputs = _authenticate_inputs(prepared_path, kd_path)
    selected, source_rows = _select_shards(
        prepared,
        source_by_shard,
        targets=targets,
        seed=seed,
    )
    selection_basis = (
        "authenticated whole-shard per-source minimum quotas; fixed-seed SHA256 source "
        "ranking followed by fixed-seed SHA256 global ordering"
    )
    required_tokens = sum(target for _, target in targets)
    policy = _build_policy(
        policy_id=policy_id,
        approved=approve,
        selection_basis=selection_basis,
        inputs=inputs,
        required_tokens=required_tokens,
        selected=selected,
        source_rows=source_rows,
    )
    audit = _build_audit(
        policy_id=policy_id,
        approved=approve,
        seed=seed,
        selection_basis=selection_basis,
        inputs=inputs,
        targets=targets,
        selected=selected,
        source_rows=source_rows,
    )
    report = _render_report(audit)
    _assert_input_shas_stable(inputs)
    published_files: dict[str, object] | None = None
    if approve:
        published_files = _publish_bundle(
            output,
            policy=policy,
            audit=audit,
            report=report,
            inputs=inputs,
        )
    return {
        "ok": True,
        "dry_run": not approve,
        "approved_for_quality_cooldown": approve,
        "output": str(output),
        "policy": policy,
        "audit": audit,
        "published_files": published_files,
        "training_started": False,
        "teacher_kd_started": False,
    }


__all__ = [
    "DEFAULT_QUALITY_COOLDOWN_TOKENS",
    "DEFAULT_QUALITY_POLICY_ID",
    "DEFAULT_QUALITY_POLICY_SEED",
    "DEFAULT_QUALITY_SOURCE_TOKEN_TARGETS",
    "QUALITY_POLICY_AUDIT_FILENAME",
    "QUALITY_POLICY_COMPLETE_FILENAME",
    "QUALITY_POLICY_FILENAME",
    "QUALITY_POLICY_MANIFEST_FILENAME",
    "QUALITY_POLICY_REPORT_FILENAME",
    "generate_quality_cooldown_policy",
]
