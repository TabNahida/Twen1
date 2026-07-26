#!/usr/bin/env python3
# ruff: noqa: RUF001
"""Build an authenticated, read-only frozen-validation checkpoint sweep.

The script does not evaluate a model or mutate any evaluation input.  It
authenticates one frozen prepared-text manifest, a completed baseline
evaluation, and one or more completed candidate evaluations.  It then compares
the candidate role's token-weighted NLL overall and by prepared-data source.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
EVALUATION_SCHEMA_VERSION = 1
PREPARED_KINDS = frozenset({"twen_prepared_text"})
EVALUATION_KIND = "twen_nll_evaluation"
PLAN_KIND = "twen_nll_evaluation_plan"
TARGET_ROLE = "candidate"
COLORS = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#7c3aed",
    "#d97706",
    "#0891b2",
)


class SweepError(ValueError):
    """An input is incomplete, unauthenticated, or not directly comparable."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-manifest",
        required=True,
        type=Path,
        help="frozen prepared-text validation manifest",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        type=Path,
        help="completed baseline evaluation root (or its manifest.json)",
    )
    parser.add_argument(
        "--baseline-label",
        default="v3",
        help="label used for the baseline in the report (default: v3)",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="LABEL=EVALUATION",
        help="completed candidate evaluation root; repeat for every checkpoint",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="replace an existing output only if its MANIFEST/COMPLETE chain is valid",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise SweepError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SweepError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SweepError(f"{label} must be a JSON object: {path}")
    return value


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SweepError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _finite(value: Any, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SweepError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        suffix = " and non-negative" if minimum == 0.0 else ""
        raise SweepError(f"{label} must be finite{suffix}")
    return result


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SweepError(f"{label} must be a non-empty string")
    return value


def _sha256_string(value: Any, *, label: str) -> str:
    result = _nonempty_string(value, label=label).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise SweepError(f"{label} must be a 64-digit SHA256")
    return result


def _assert_close(actual: float, expected: float, *, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-6):
        raise SweepError(f"{label} is inconsistent: {actual!r} != {expected!r}")


def _identity(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    rendered = (
        path.relative_to(relative_to).as_posix() if relative_to is not None else str(resolved)
    )
    return {
        "path": rendered,
        "sha256": _sha256(path),
        "size": path.stat().st_size,
    }


def _source_id(source_path: str) -> str:
    normalized = source_path.replace("\\", "/")
    marker = "/extracted/"
    if marker in normalized:
        value = normalized.split(marker, 1)[1].split("/", 1)[0]
        if value:
            return value
    parts = [part for part in Path(normalized).parts if part not in {"", "/"}]
    if len(parts) >= 3:
        return parts[-3]
    return "unknown"


def _safe_child(root: Path, value: Any, *, label: str) -> Path:
    raw = _nonempty_string(value, label=label)
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise SweepError(f"{label} must be a safe relative path")
    root_resolved = root.resolve()
    result = (root / relative).resolve()
    if result != root_resolved and root_resolved not in result.parents:
        raise SweepError(f"{label} escapes its evaluation root")
    return result


def _parse_labeled_paths(values: Sequence[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for raw in values:
        label, separator, path = raw.partition("=")
        label = label.strip()
        path = path.strip()
        if not separator or not label or not path:
            raise SweepError("--candidate entries must use LABEL=EVALUATION")
        if label in seen:
            raise SweepError(f"duplicate candidate label: {label!r}")
        seen.add(label)
        result.append((label, Path(path)))
    if not result:
        raise SweepError("at least one --candidate is required")
    return result


def _load_prepared_manifest(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise SweepError(f"prepared manifest does not exist: {path}")
    payload = _read_json(path, label="prepared manifest")
    if payload.get("kind") not in PREPARED_KINDS:
        raise SweepError(f"unsupported prepared manifest kind: {payload.get('kind')!r}")
    _integer(payload.get("schema_version"), label="prepared.schema_version", minimum=1)
    dataset_fingerprint = _nonempty_string(
        payload.get("dataset_fingerprint"),
        label="prepared.dataset_fingerprint",
    )
    token_count = _integer(payload.get("token_count"), label="prepared.token_count", minimum=1)
    sequence_count = _integer(
        payload.get("sequence_count"),
        label="prepared.sequence_count",
        minimum=1,
    )
    entries = payload.get("shards")
    if not isinstance(entries, list) or not entries:
        raise SweepError("prepared.shards must be a non-empty list")

    shards: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    summed_tokens = 0
    summed_sequences = 0
    for index, raw in enumerate(entries):
        if not isinstance(raw, Mapping):
            raise SweepError(f"prepared.shards[{index}] must be an object")
        shard_id = _nonempty_string(
            raw.get("shard_id"),
            label=f"prepared.shards[{index}].shard_id",
        )
        if shard_id in seen_ids:
            raise SweepError(f"prepared manifest has duplicate shard {shard_id!r}")
        seen_ids.add(shard_id)
        shard_tokens = _integer(
            raw.get("token_count"),
            label=f"prepared.shards[{index}].token_count",
            minimum=1,
        )
        shard_sequences = _integer(
            raw.get("sequence_count"),
            label=f"prepared.shards[{index}].sequence_count",
            minimum=1,
        )
        tensors_sha256 = _nonempty_string(
            raw.get("tensors_sha256"),
            label=f"prepared.shards[{index}].tensors_sha256",
        )
        source_path = _nonempty_string(
            raw.get("source_path"),
            label=f"prepared.shards[{index}].source_path",
        )
        shards.append(
            {
                "index": index,
                "shard_id": shard_id,
                "token_count": shard_tokens,
                "sequence_count": shard_sequences,
                "tensors_sha256": tensors_sha256,
                "source_path": source_path,
                "source": _source_id(source_path),
            }
        )
        summed_tokens += shard_tokens
        summed_sequences += shard_sequences
    if summed_tokens != token_count or summed_sequences != sequence_count:
        raise SweepError("prepared top-level token/sequence totals do not match its shards")
    return {
        "path": path,
        "sha256": _sha256(path),
        "dataset_fingerprint": dataset_fingerprint,
        "token_count": token_count,
        "sequence_count": sequence_count,
        "shards": shards,
    }


def _evaluation_root(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_file():
        if resolved.name != "manifest.json":
            raise SweepError(f"evaluation file must be manifest.json: {resolved}")
        return resolved.parent
    if not resolved.is_dir():
        raise SweepError(f"evaluation does not exist: {resolved}")
    return resolved


def _validate_result(
    result: Mapping[str, Any],
    *,
    role: str,
    shard: Mapping[str, Any],
    expected_fingerprint: str,
    label: str,
) -> dict[str, Any]:
    if (
        result.get("schema_version") != EVALUATION_SCHEMA_VERSION
        or result.get("kind") != "nll_shard_result"
    ):
        raise SweepError(f"{label} has unsupported shard-result kind/schema")
    if result.get("role") != role or result.get("source_shard_id") != shard["shard_id"]:
        raise SweepError(f"{label} role/source-shard lineage mismatch")
    if result.get("fingerprint") != expected_fingerprint:
        raise SweepError(f"{label} fingerprint is inconsistent with PLAN and prepared shard")
    sequences = _integer(result.get("sequences"), label=f"{label}.sequences", minimum=1)
    if sequences != shard["sequence_count"]:
        raise SweepError(f"{label} sequence count differs from prepared manifest")
    predicted_tokens = _integer(
        result.get("predicted_tokens"),
        label=f"{label}.predicted_tokens",
        minimum=1,
    )
    nll_sum = _finite(result.get("nll_sum"), label=f"{label}.nll_sum", minimum=0.0)
    mean_nll = _finite(result.get("mean_nll"), label=f"{label}.mean_nll", minimum=0.0)
    _assert_close(
        mean_nll,
        nll_sum / predicted_tokens,
        label=f"{label}.mean_nll",
    )
    return {
        "sequences": sequences,
        "predicted_tokens": predicted_tokens,
        "nll_sum": nll_sum,
        "mean_nll": mean_nll,
    }


def _load_role(
    root: Path,
    *,
    role: str,
    raw: Any,
    plan_fingerprint: str,
    prepared: Mapping[str, Any],
    evaluation_label: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise SweepError(f"{evaluation_label}.roles.{role} must be an object")
    entries = raw.get("shards")
    if not isinstance(entries, list) or not entries:
        raise SweepError(f"{evaluation_label}.roles.{role}.shards must be non-empty")
    prepared_by_id = {str(entry["shard_id"]): entry for entry in prepared["shards"]}
    results: dict[str, dict[str, Any]] = {}
    ordered_results: list[dict[str, Any]] = []
    role_fingerprint = _canonical_sha256({"plan_fingerprint": plan_fingerprint, "role": role})

    for index, raw_entry in enumerate(entries):
        entry_label = f"{evaluation_label}.roles.{role}.shards[{index}]"
        if not isinstance(raw_entry, Mapping):
            raise SweepError(f"{entry_label} must be an object")
        shard_id = _nonempty_string(
            raw_entry.get("source_shard_id"),
            label=f"{entry_label}.source_shard_id",
        )
        if shard_id in results:
            raise SweepError(f"{evaluation_label}.{role} repeats shard {shard_id}")
        shard = prepared_by_id.get(shard_id)
        if shard is None:
            raise SweepError(f"{evaluation_label}.{role} contains unknown shard {shard_id}")
        shard_dir = _safe_child(
            root,
            raw_entry.get("path"),
            label=f"{entry_label}.path",
        )
        complete_path = shard_dir / "COMPLETE"
        if not complete_path.is_file():
            raise SweepError(f"{entry_label} is incomplete: {complete_path}")
        complete_sha256 = _sha256(complete_path)
        if raw_entry.get("complete_sha256") != complete_sha256:
            raise SweepError(f"{entry_label} COMPLETE hash mismatch")
        marker = _read_json(complete_path, label=f"{entry_label} COMPLETE")
        expected_shard_fingerprint = _canonical_sha256(
            {
                "role_fingerprint": role_fingerprint,
                "source_shard_id": shard_id,
                "source_tensors_sha256": shard["tensors_sha256"],
                "sequence_count": shard["sequence_count"],
                "token_count": shard["token_count"],
            }
        )
        if (
            marker.get("schema_version") != EVALUATION_SCHEMA_VERSION
            or marker.get("shard_id") != shard_id
            or marker.get("fingerprint") != expected_shard_fingerprint
            or marker.get("source_fingerprint") != shard["tensors_sha256"]
        ):
            raise SweepError(f"{entry_label} COMPLETE lineage mismatch")
        metadata = marker.get("metadata")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("kind") != "nll_shard_result"
            or metadata.get("role") != role
        ):
            raise SweepError(f"{entry_label} COMPLETE metadata mismatch")
        outputs = marker.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            raise SweepError(f"{entry_label} COMPLETE has no output inventory")
        if raw_entry.get("outputs") != outputs:
            raise SweepError(f"{entry_label} manifest/COMPLETE output inventories differ")
        seen_outputs: set[str] = set()
        for output_index, output in enumerate(outputs):
            output_label = f"{entry_label}.outputs[{output_index}]"
            if not isinstance(output, Mapping):
                raise SweepError(f"{output_label} must be an object")
            output_path = _safe_child(
                shard_dir,
                output.get("path"),
                label=f"{output_label}.path",
            )
            relative_name = output_path.relative_to(shard_dir.resolve()).as_posix()
            if relative_name in seen_outputs:
                raise SweepError(f"{entry_label} repeats output {relative_name!r}")
            seen_outputs.add(relative_name)
            if not output_path.is_file():
                raise SweepError(f"{output_label} is missing: {output_path}")
            if output_path.stat().st_size != _integer(
                output.get("size"), label=f"{output_label}.size"
            ) or _sha256(output_path) != _nonempty_string(
                output.get("sha256"),
                label=f"{output_label}.sha256",
            ):
                raise SweepError(f"{output_label} hash/size mismatch")
        if "result.json" not in seen_outputs:
            raise SweepError(f"{entry_label} does not inventory result.json")
        result_path = shard_dir / "result.json"
        result = _validate_result(
            _read_json(result_path, label=f"{entry_label} result"),
            role=role,
            shard=shard,
            expected_fingerprint=expected_shard_fingerprint,
            label=f"{entry_label}.result",
        )
        if metadata.get("predicted_tokens") != result["predicted_tokens"]:
            raise SweepError(f"{entry_label} COMPLETE predicted-token metadata mismatch")
        result.update(
            {
                "shard_id": shard_id,
                "source": shard["source"],
                "prepared_index": shard["index"],
                "input_tokens": shard["token_count"],
                "complete_sha256": complete_sha256,
            }
        )
        results[shard_id] = result
        ordered_results.append(result)

    expected_ids = set(prepared_by_id)
    if set(results) != expected_ids:
        missing = sorted(expected_ids - set(results))
        extra = sorted(set(results) - expected_ids)
        raise SweepError(
            f"{evaluation_label}.{role} does not cover the frozen manifest "
            f"(missing={missing}, extra={extra})"
        )
    ordered_results.sort(key=lambda row: int(row["prepared_index"]))
    sequences = sum(int(row["sequences"]) for row in ordered_results)
    predicted_tokens = sum(int(row["predicted_tokens"]) for row in ordered_results)
    nll_sum = sum(float(row["nll_sum"]) for row in ordered_results)
    mean_nll = nll_sum / predicted_tokens
    declared_sequences = _integer(
        raw.get("sequences"),
        label=f"{evaluation_label}.roles.{role}.sequences",
        minimum=1,
    )
    declared_tokens = _integer(
        raw.get("predicted_tokens"),
        label=f"{evaluation_label}.roles.{role}.predicted_tokens",
        minimum=1,
    )
    declared_nll_sum = _finite(
        raw.get("nll_sum"),
        label=f"{evaluation_label}.roles.{role}.nll_sum",
        minimum=0.0,
    )
    declared_mean_nll = _finite(
        raw.get("mean_nll"),
        label=f"{evaluation_label}.roles.{role}.mean_nll",
        minimum=0.0,
    )
    if declared_sequences != sequences or declared_tokens != predicted_tokens:
        raise SweepError(f"{evaluation_label}.{role} shard counters do not match role totals")
    _assert_close(
        declared_nll_sum,
        nll_sum,
        label=f"{evaluation_label}.roles.{role}.nll_sum",
    )
    _assert_close(
        declared_mean_nll,
        mean_nll,
        label=f"{evaluation_label}.roles.{role}.mean_nll",
    )
    perplexity = math.exp(mean_nll) if mean_nll < 700 else None
    declared_perplexity = raw.get("perplexity")
    if perplexity is None:
        if declared_perplexity is not None:
            raise SweepError(f"{evaluation_label}.{role}.perplexity must be null")
    else:
        _assert_close(
            _finite(
                declared_perplexity,
                label=f"{evaluation_label}.roles.{role}.perplexity",
                minimum=0.0,
            ),
            perplexity,
            label=f"{evaluation_label}.roles.{role}.perplexity",
        )
    return {
        "role": role,
        "sequences": sequences,
        "predicted_tokens": predicted_tokens,
        "nll_sum": nll_sum,
        "mean_nll": mean_nll,
        "perplexity": perplexity,
        "shards": ordered_results,
    }


def _load_completed_evaluation(
    path: Path,
    *,
    prepared: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    root = _evaluation_root(path)
    manifest_path = root / "manifest.json"
    plan_path = root / "PLAN.json"
    complete_path = root / "COMPLETE"
    if not manifest_path.is_file() or not plan_path.is_file() or not complete_path.is_file():
        raise SweepError(f"{label} evaluation is incomplete: {root}")
    manifest_sha256 = _sha256(manifest_path)
    try:
        authenticated_sha256 = complete_path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise SweepError(f"cannot read {label} COMPLETE: {exc}") from exc
    if authenticated_sha256 != manifest_sha256:
        raise SweepError(f"{label} COMPLETE does not authenticate manifest.json")
    manifest = _read_json(manifest_path, label=f"{label} manifest")
    plan = _read_json(plan_path, label=f"{label} PLAN")
    if (
        manifest.get("schema_version") != EVALUATION_SCHEMA_VERSION
        or manifest.get("kind") != EVALUATION_KIND
    ):
        raise SweepError(f"{label} manifest kind/schema is unsupported")
    if plan.get("schema_version") != EVALUATION_SCHEMA_VERSION or plan.get("kind") != PLAN_KIND:
        raise SweepError(f"{label} PLAN kind/schema is unsupported")
    plan_sha256 = _sha256(plan_path)
    plan_fingerprint = _nonempty_string(
        plan.get("plan_fingerprint"),
        label=f"{label}.PLAN.plan_fingerprint",
    )
    unsigned_plan = {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    if _canonical_sha256(unsigned_plan) != plan_fingerprint:
        raise SweepError(f"{label} PLAN fingerprint is invalid")
    if (
        manifest.get("plan_sha256") != plan_sha256
        or manifest.get("plan_fingerprint") != plan_fingerprint
    ):
        raise SweepError(f"{label} manifest/PLAN lineage is invalid")
    raw_plan_path = manifest.get("plan_path")
    if isinstance(raw_plan_path, str) and Path(raw_plan_path).resolve() != plan_path.resolve():
        raise SweepError(f"{label} manifest plan_path does not identify PLAN.json")
    if (
        plan.get("prepared_manifest_sha256") != prepared["sha256"]
        or plan.get("prepared_dataset_fingerprint") != prepared["dataset_fingerprint"]
    ):
        raise SweepError(f"{label} did not evaluate the supplied frozen prepared manifest")
    checkpoint_state = manifest.get("checkpoint_state")
    if not isinstance(checkpoint_state, Mapping) or checkpoint_state != plan.get(
        "checkpoint_state"
    ):
        raise SweepError(f"{label} checkpoint_state is not authenticated by PLAN")
    global_step = _integer(
        checkpoint_state.get("global_step"),
        label=f"{label}.checkpoint_state.global_step",
    )
    committed_tokens = _integer(
        checkpoint_state.get("committed_tokens"),
        label=f"{label}.checkpoint_state.committed_tokens",
    )
    if checkpoint_state.get("kind") not in {"periodic", "interrupt", "milestone"}:
        raise SweepError(f"{label} checkpoint kind is invalid")
    device_type = _nonempty_string(
        plan.get("device_type"),
        label=f"{label}.PLAN.device_type",
    )
    dtype = _nonempty_string(plan.get("dtype"), label=f"{label}.PLAN.dtype")
    batch_size = _integer(
        plan.get("batch_size"),
        label=f"{label}.PLAN.batch_size",
        minimum=1,
    )
    runtime = plan.get("runtime")
    if not isinstance(runtime, Mapping) or manifest.get("runtime") != runtime:
        raise SweepError(f"{label} runtime is not identical in manifest and PLAN")
    config_fingerprint = _sha256_string(
        plan.get("config_fingerprint"),
        label=f"{label}.PLAN.config_fingerprint",
    )
    inference_lineage = plan.get("checkpoint_inference_lineage")
    if not isinstance(inference_lineage, Mapping):
        raise SweepError(f"{label}.PLAN.checkpoint_inference_lineage must be an object")
    lineage_hashes = {
        field: _sha256_string(
            inference_lineage.get(field),
            label=f"{label}.PLAN.checkpoint_inference_lineage.{field}",
        )
        for field in (
            "archived_config_sha256",
            "saved_critical_fingerprint",
            "current_preflight_fingerprint",
            "saved_source_tree_sha256",
            "current_source_tree_sha256",
        )
    }
    if lineage_hashes["current_preflight_fingerprint"] != config_fingerprint:
        raise SweepError(
            f"{label} PLAN config fingerprint differs from checkpoint inference lineage"
        )
    lineage_booleans: dict[str, bool] = {}
    for field in ("exact_training_fingerprint_match", "source_tree_match"):
        value = inference_lineage.get(field)
        if not isinstance(value, bool):
            raise SweepError(f"{label}.PLAN.checkpoint_inference_lineage.{field} must be boolean")
        lineage_booleans[field] = value
    lineage_mode = _nonempty_string(
        inference_lineage.get("mode"),
        label=f"{label}.PLAN.checkpoint_inference_lineage.mode",
    )
    track = _nonempty_string(manifest.get("track"), label=f"{label}.manifest.track")
    stage = _nonempty_string(manifest.get("stage"), label=f"{label}.manifest.stage")
    expert_initialization = _nonempty_string(
        manifest.get("expert_initialization"),
        label=f"{label}.manifest.expert_initialization",
    )
    if expert_initialization != plan.get("expert_initialization"):
        raise SweepError(f"{label} expert initialization differs in manifest and PLAN")
    plan_roles = plan.get("roles")
    manifest_roles = manifest.get("roles")
    if (
        not isinstance(plan_roles, list)
        or not plan_roles
        or not all(isinstance(role, str) and role for role in plan_roles)
        or len(set(plan_roles)) != len(plan_roles)
        or not isinstance(manifest_roles, Mapping)
        or set(plan_roles) != set(manifest_roles)
        or TARGET_ROLE not in manifest_roles
    ):
        raise SweepError(f"{label} role inventory differs between manifest and PLAN")

    roles: dict[str, dict[str, Any]] = {}
    for role in plan_roles:
        roles[role] = _load_role(
            root,
            role=role,
            raw=manifest_roles[role],
            plan_fingerprint=plan_fingerprint,
            prepared=prepared,
            evaluation_label=label,
        )
    role_token_counts = {int(value["predicted_tokens"]) for value in roles.values()}
    if len(role_token_counts) != 1:
        raise SweepError(f"{label} roles processed different token counts")
    return {
        "label": label,
        "root": root,
        "manifest": manifest,
        "plan": plan,
        "identity": {
            "root": str(root),
            "manifest": _identity(manifest_path),
            "complete": _identity(complete_path),
            "plan": _identity(plan_path),
        },
        "comparison_contract": {
            "track": track,
            "stage": stage,
            "expert_initialization": expert_initialization,
            "prepared_manifest_sha256": plan.get("prepared_manifest_sha256"),
            "prepared_dataset_fingerprint": plan.get("prepared_dataset_fingerprint"),
            "batch_size": batch_size,
            "device_type": device_type,
            "dtype": dtype,
        },
        "checkpoint_state": {
            "global_step": global_step,
            "committed_tokens": committed_tokens,
            "kind": checkpoint_state.get("kind"),
            "tag": checkpoint_state.get("tag"),
        },
        "run_id": manifest.get("run_id"),
        "harness": {
            "config_fingerprint": config_fingerprint,
            "runtime": json.loads(json.dumps(runtime, sort_keys=True)),
            "checkpoint_inference_lineage": {
                "mode": lineage_mode,
                **lineage_hashes,
                **lineage_booleans,
            },
        },
        "roles": roles,
    }


def _source_aggregates(role: Mapping[str, Any]) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "shards": 0,
            "sequences": 0,
            "input_tokens": 0,
            "predicted_tokens": 0,
            "nll_sum": 0.0,
        }
    )
    for shard in role["shards"]:
        source = str(shard["source"])
        aggregate = totals[source]
        aggregate["shards"] += 1
        aggregate["sequences"] += int(shard["sequences"])
        aggregate["input_tokens"] += int(shard["input_tokens"])
        aggregate["predicted_tokens"] += int(shard["predicted_tokens"])
        aggregate["nll_sum"] += float(shard["nll_sum"])
    result: list[dict[str, Any]] = []
    for source, aggregate in sorted(totals.items()):
        mean_nll = float(aggregate["nll_sum"]) / int(aggregate["predicted_tokens"])
        result.append(
            {
                "source": source,
                **aggregate,
                "mean_nll": mean_nll,
                "perplexity": math.exp(mean_nll) if mean_nll < 700 else None,
            }
        )
    return result


def _public_evaluation(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    candidate = evaluation["roles"][TARGET_ROLE]
    return {
        "label": evaluation["label"],
        "run_id": evaluation["run_id"],
        "checkpoint_state": evaluation["checkpoint_state"],
        "evaluation": evaluation["identity"],
        "evaluation_harness": evaluation["harness"],
        "overall": {
            key: candidate[key]
            for key in (
                "sequences",
                "predicted_tokens",
                "nll_sum",
                "mean_nll",
                "perplexity",
            )
        },
        "sources": _source_aggregates(candidate),
    }


def _compare_candidate(
    evaluation: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    public = _public_evaluation(evaluation)
    baseline_role = baseline["roles"][TARGET_ROLE]
    candidate_role = evaluation["roles"][TARGET_ROLE]
    delta = float(candidate_role["mean_nll"]) - float(baseline_role["mean_nll"])
    public["overall"].update(
        {
            "baseline_mean_nll": baseline_role["mean_nll"],
            "nll_delta_candidate_minus_baseline": delta,
            "nll_relative_delta_percent": (delta / float(baseline_role["mean_nll"]) * 100.0),
            "beats_baseline": delta < 0.0,
        }
    )
    baseline_sources = {str(row["source"]): row for row in _source_aggregates(baseline_role)}
    compared_sources: list[dict[str, Any]] = []
    for row in public["sources"]:
        source = str(row["source"])
        baseline_row = baseline_sources[source]
        source_delta = float(row["mean_nll"]) - float(baseline_row["mean_nll"])
        compared_sources.append(
            {
                **row,
                "baseline_mean_nll": baseline_row["mean_nll"],
                "nll_delta_candidate_minus_baseline": source_delta,
                "nll_relative_delta_percent": (
                    source_delta / float(baseline_row["mean_nll"]) * 100.0
                ),
                "beats_baseline": source_delta < 0.0,
            }
        )
    public["sources"] = compared_sources
    return public


def _same_comparison_contract(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    if candidate["comparison_contract"] != baseline["comparison_contract"]:
        raise SweepError(
            f"{candidate['label']} differs from baseline in "
            "track/stage/expert initialization/prepared corpus/batch/device/dtype"
        )
    baseline_role = baseline["roles"][TARGET_ROLE]
    candidate_role = candidate["roles"][TARGET_ROLE]
    if candidate_role["predicted_tokens"] != baseline_role["predicted_tokens"]:
        raise SweepError(f"{candidate['label']} candidate role processed a different token count")
    baseline_shards = {str(row["shard_id"]): row for row in baseline_role["shards"]}
    for candidate_shard in candidate_role["shards"]:
        shard_id = str(candidate_shard["shard_id"])
        baseline_shard = baseline_shards[shard_id]
        if (
            candidate_shard["predicted_tokens"] != baseline_shard["predicted_tokens"]
            or candidate_shard["sequences"] != baseline_shard["sequences"]
        ):
            raise SweepError(
                f"{candidate['label']} shard {shard_id} processed different "
                "candidate-role tokens/sequences"
            )


def build_summary(
    *,
    prepared_manifest: Path,
    baseline_path: Path,
    baseline_label: str,
    candidate_paths: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    baseline_label = baseline_label.strip()
    if not baseline_label:
        raise SweepError("baseline label must not be empty")
    if baseline_label in {label for label, _ in candidate_paths}:
        raise SweepError("baseline label must differ from every candidate label")
    prepared = _load_prepared_manifest(prepared_manifest)
    baseline = _load_completed_evaluation(
        baseline_path,
        prepared=prepared,
        label=baseline_label,
    )
    candidates = [
        _load_completed_evaluation(path, prepared=prepared, label=label)
        for label, path in candidate_paths
    ]
    for candidate in candidates:
        _same_comparison_contract(baseline, candidate)
    public_baseline = _public_evaluation(baseline)
    public_candidates = [
        _compare_candidate(candidate, baseline=baseline) for candidate in candidates
    ]
    ranking = sorted(
        (
            {
                "rank": 0,
                "label": row["label"],
                "checkpoint_step": row["checkpoint_state"]["global_step"],
                "committed_tokens": row["checkpoint_state"]["committed_tokens"],
                "mean_nll": row["overall"]["mean_nll"],
                "nll_delta_candidate_minus_baseline": row["overall"][
                    "nll_delta_candidate_minus_baseline"
                ],
            }
            for row in public_candidates
        ),
        key=lambda row: (float(row["mean_nll"]), str(row["label"])),
    )
    for index, row in enumerate(ranking, 1):
        row["rank"] = index
    prepared_identity = {
        "path": str(prepared["path"]),
        "sha256": prepared["sha256"],
        "dataset_fingerprint": prepared["dataset_fingerprint"],
        "token_count": prepared["token_count"],
        "sequence_count": prepared["sequence_count"],
        "shard_count": len(prepared["shards"]),
        "sources": sorted({str(row["source"]) for row in prepared["shards"]}),
    }
    input_identity = {
        "prepared_manifest": prepared_identity,
        "baseline_evaluation": baseline["identity"],
        "candidate_evaluations": {
            str(candidate["label"]): candidate["identity"] for candidate in candidates
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "twen_v4_checkpoint_frozen_validation_sweep",
        "training_started_by_summarizer": False,
        "inputs_mutated_by_summarizer": False,
        "target_role": TARGET_ROLE,
        "delta_definition": "candidate_mean_nll - baseline_mean_nll; lower is better",
        "comparison_contract": baseline["comparison_contract"],
        "prepared_manifest": prepared_identity,
        "inputs_sha256": _canonical_sha256(input_identity),
        "baseline": public_baseline,
        "candidates": public_candidates,
        "ranking": ranking,
        "selection": {
            "criterion": "lowest authenticated candidate-role mean NLL",
            "label": ranking[0]["label"],
            "checkpoint_step": ranking[0]["checkpoint_step"],
            "committed_tokens": ranking[0]["committed_tokens"],
            "mean_nll": ranking[0]["mean_nll"],
            "nll_delta_candidate_minus_baseline": ranking[0]["nll_delta_candidate_minus_baseline"],
        },
        "scope": {
            "supports": [
                "same-corpus candidate-role NLL checkpoint ranking",
                "token-weighted per-source NLL comparison",
            ],
            "does_not_establish": [
                "training-data quality or absence of data replay",
                "optimizer/LR correctness",
                "generation quality",
                "causal attribution to any single training change",
            ],
        },
    }


def _format_number(value: Any, digits: int = 6) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    number = float(value)
    if not math.isfinite(number):
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{number:.{digits}f}"


def _delta_svg(
    *,
    title: str,
    groups: Sequence[str],
    series: Mapping[str, Sequence[float]],
    y_label: str,
) -> str:
    if not groups or not series:
        raise SweepError(f"chart {title!r} has no data")
    if any(len(values) != len(groups) for values in series.values()):
        raise SweepError(f"chart {title!r} has inconsistent series lengths")
    values = [float(value) for rows in series.values() for value in rows]
    if not values or any(not math.isfinite(value) for value in values):
        raise SweepError(f"chart {title!r} contains non-finite values")
    low = min(0.0, min(values))
    high = max(0.0, max(values))
    pad = max(abs(low) * 0.2, 0.01) if math.isclose(low, high) else max((high - low) * 0.12, 1e-5)
    y_min = low - pad
    y_max = high + pad
    width = max(1100, 260 + len(groups) * max(115, 38 * len(series)))
    height = 720
    left, right, top, bottom = 120.0, float(width - 45), 85.0, 565.0

    def sy(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    zero_y = sy(0.0)
    group_width = (right - left) / len(groups)
    bar_span = group_width * 0.72
    bar_width = bar_span / len(series)
    parts = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            "<style>text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#111827}"
            ".grid{stroke:#e5e7eb;stroke-width:1}.axis{stroke:#374151;stroke-width:1.4}"
            ".title{font-size:24px;font-weight:700}.tick{font-size:12px;fill:#4b5563}"
            ".label{font-size:14px;font-weight:600}.legend{font-size:13px}</style>"
        ),
        f'<text class="title" x="{left}" y="38">{html.escape(title)}</text>',
    ]
    for index in range(7):
        fraction = index / 6
        y = bottom - fraction * (bottom - top)
        value = y_min + fraction * (y_max - y_min)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}"/>')
        parts.append(
            f'<text class="tick" x="{left - 10}" y="{y + 4:.2f}" '
            f'text-anchor="end">{value:+.5f}</text>'
        )
    parts.extend(
        (
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
            (
                f'<line x1="{left}" y1="{zero_y:.2f}" x2="{right}" y2="{zero_y:.2f}" '
                'stroke="#111827" stroke-width="2"/>'
            ),
            (
                f'<text class="label" x="28" y="{(top + bottom) / 2:.2f}" '
                f'text-anchor="middle" transform="rotate(-90 28 {(top + bottom) / 2:.2f})">'
                f"{html.escape(y_label)}</text>"
            ),
        )
    )
    series_items = list(series.items())
    for group_index, group in enumerate(groups):
        center = left + (group_index + 0.5) * group_width
        start = center - bar_span / 2
        for series_index, (_name, rows) in enumerate(series_items):
            value = float(rows[group_index])
            x = start + series_index * bar_width + bar_width * 0.08
            visible_width = bar_width * 0.84
            value_y = sy(value)
            y = min(value_y, zero_y)
            bar_height = max(abs(zero_y - value_y), 1.0)
            color = COLORS[series_index % len(COLORS)]
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{visible_width:.2f}" '
                f'height="{bar_height:.2f}" fill="{color}" opacity="0.88"/>'
            )
            text_y = value_y - 7 if value >= 0 else value_y + 16
            parts.append(
                f'<text class="tick" x="{x + visible_width / 2:.2f}" y="{text_y:.2f}" '
                f'text-anchor="middle">{value:+.5f}</text>'
            )
        parts.append(
            f'<text class="tick" x="{center:.2f}" y="592" text-anchor="end" '
            f'transform="rotate(-28 {center:.2f} 592)">'
            f"{html.escape(group)}</text>"
        )
    legend_x = left
    for index, (name, _values) in enumerate(series_items):
        color = COLORS[index % len(COLORS)]
        parts.append(f'<rect x="{legend_x:.2f}" y="660" width="18" height="12" fill="{color}"/>')
        parts.append(
            f'<text class="legend" x="{legend_x + 25:.2f}" y="671">{html.escape(name)}</text>'
        )
        legend_x += max(150, len(name) * 8 + 55)
    parts.append(
        f'<text class="tick" x="{right}" y="704" text-anchor="end">'
        "Δ = candidate − baseline；负值更好</text>"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n' + "\n".join(parts) + "\n</svg>\n"
    )


def _markdown(summary: Mapping[str, Any]) -> str:
    baseline = summary["baseline"]
    candidates = summary["candidates"]
    selection = summary["selection"]
    contract = summary["comparison_contract"]
    overall_rows = []
    for candidate in candidates:
        overall = candidate["overall"]
        state = candidate["checkpoint_state"]
        overall_rows.append(
            "| {label} | {step} | {tokens} | {nll} | {delta} | {relative} | {result} |".format(
                label=candidate["label"],
                step=_format_number(state["global_step"], 0),
                tokens=_format_number(state["committed_tokens"], 0),
                nll=_format_number(overall["mean_nll"]),
                delta=f"{float(overall['nll_delta_candidate_minus_baseline']):+.6f}",
                relative=f"{float(overall['nll_relative_delta_percent']):+.4f}%",
                result="优于 baseline" if overall["beats_baseline"] else "未优于 baseline",
            )
        )
    source_names = [row["source"] for row in baseline["sources"]]
    candidate_sources = {
        candidate["label"]: {row["source"]: row for row in candidate["sources"]}
        for candidate in candidates
    }
    source_header = (
        "| Source | Baseline NLL | "
        + " | ".join(f"{candidate['label']} NLL / Δ" for candidate in candidates)
        + " |"
    )
    source_separator = "|---|---:|" + "".join("---:|" for _ in candidates)
    source_rows = []
    baseline_sources = {row["source"]: row for row in baseline["sources"]}
    for source in source_names:
        cells = []
        for candidate in candidates:
            row = candidate_sources[candidate["label"]][source]
            cells.append(
                f"{float(row['mean_nll']):.6f} / "
                f"{float(row['nll_delta_candidate_minus_baseline']):+.6f}"
            )
        source_rows.append(
            f"| {source} | {float(baseline_sources[source]['mean_nll']):.6f} | "
            + " | ".join(cells)
            + " |"
        )
    evidence_rows = []
    harness_rows = []
    for item in [baseline, *candidates]:
        identity = item["evaluation"]
        evidence_rows.append(
            f"| {item['label']} | `{identity['manifest']['sha256']}` | "
            f"`{identity['plan']['sha256']}` | `{identity['complete']['sha256']}` |"
        )
        harness = item["evaluation_harness"]
        lineage = harness["checkpoint_inference_lineage"]
        harness_rows.append(
            f"| {item['label']} | `{harness['config_fingerprint']}` | "
            f"`{lineage['archived_config_sha256']}` | "
            f"`{lineage['saved_source_tree_sha256']}` | "
            f"`{lineage['current_source_tree_sha256']}` | "
            f"`{str(lineage['exact_training_fingerprint_match']).lower()}` |"
        )
    return f"""# v4 checkpoint frozen-validation sweep

## 结论

在 {len(candidates)} 个已完成且认证通过的 candidate checkpoint 中，按 candidate-role
token-weighted mean NLL 排名，当前最优是 **{selection["label"]}**（step
{selection["checkpoint_step"]}，NLL {_format_number(selection["mean_nll"])}，相对
{baseline["label"]} 的 Δ 为
{float(selection["nll_delta_candidate_minus_baseline"]):+.6f}）。

本文统一使用 `Δ = candidate NLL - baseline NLL`；因此负值表示改善，正值表示退化。

![Overall NLL delta](charts/overall-nll-delta.svg)

## 严格可比性与认证

- frozen prepared manifest SHA：`{summary["prepared_manifest"]["sha256"]}`
- dataset fingerprint：`{summary["prepared_manifest"]["dataset_fingerprint"]}`
- candidate role 预测 token：{_format_number(baseline["overall"]["predicted_tokens"], 0)}
- numerics：device `{contract["device_type"]}`，dtype `{contract["dtype"]}`，batch size `{contract["batch_size"]}`
- 每个 evaluation 的顶层 `COMPLETE -> manifest.json -> PLAN.json` 均已校验。
- 每个 role/shard 的 `COMPLETE`、输出 hash/size、canonical fingerprint、prepared shard
  覆盖与聚合计数均已校验。
- 除总 token 一致外，还逐 shard 校验 candidate-role token 与 sequence 数完全一致。

## Overall NLL

Baseline `{baseline["label"]}`：NLL {_format_number(baseline["overall"]["mean_nll"])}，
perplexity {_format_number(baseline["overall"]["perplexity"])}。

| Candidate | Step | Committed tokens | Mean NLL | Δ vs baseline | Relative Δ | 判定 |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(overall_rows)}

## 按 source 的 token-weighted NLL

source 聚合以 `nll_sum / predicted_tokens` 计算，不对 shard mean 做简单平均。

{source_header}
{source_separator}
{chr(10).join(source_rows)}

![Per-source NLL delta](charts/per-source-nll-delta.svg)

## 证据身份

| Label | manifest SHA256 | PLAN SHA256 | COMPLETE file SHA256 |
|---|---|---|---|
{chr(10).join(evidence_rows)}

## 评测 harness 身份

下列字段来自 fingerprint 已验证的 immutable `PLAN.json`。`saved source tree` 是
checkpoint 训练时保存的源码身份，`evaluation source tree` 是执行该次只读前向评测的
源码身份；二者不同会被如实记录，但不会被误写为 exact match。

| Label | Config / preflight fingerprint | Archived config SHA256 | Saved source tree | Evaluation source tree | Exact training fingerprint |
|---|---|---|---|---|---|
{chr(10).join(harness_rows)}

## 解释边界

该 sweep 只支持同一 frozen validation corpus 上的 checkpoint NLL 排名与按 source
比较。它不能单独证明训练数据没有回绕或污染、学习率/优化器设置正确、生成质量良好，
也不能把差异因果归因到某一个训练改动；这些结论需要独立训练与数据审计证据。

`summary.json` 保存精确数值和完整输入身份；`MANIFEST.json` 认证报告与图表，
`COMPLETE` 再认证 `MANIFEST.json`。
"""


def _verify_existing_bundle(root: Path) -> None:
    manifest_path = root / "MANIFEST.json"
    complete_path = root / "COMPLETE"
    if not root.is_dir() or not manifest_path.is_file() or not complete_path.is_file():
        raise SweepError(f"refusing to replace incomplete output bundle: {root}")
    try:
        expected = complete_path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise SweepError(f"cannot read existing output COMPLETE: {exc}") from exc
    if expected != _sha256(manifest_path):
        raise SweepError(f"refusing to replace unauthenticated output bundle: {root}")
    manifest = _read_json(manifest_path, label="existing output MANIFEST")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != "twen_v4_checkpoint_validation_sweep_bundle"
    ):
        raise SweepError(f"refusing to replace unsupported output bundle: {root}")


def _tree_identity(root: Path) -> dict[str, tuple[str, int]]:
    return {
        path.relative_to(root).as_posix(): (_sha256(path), path.stat().st_size)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def write_bundle(
    summary: Mapping[str, Any],
    output: Path,
    *,
    replace_existing: bool = False,
) -> dict[str, Any]:
    output = output.resolve()
    staging = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    if staging.exists():
        raise SweepError(f"stale staging directory exists: {staging}")
    (staging / "charts").mkdir(parents=True)
    try:
        summary_path = staging / "summary.json"
        report_path = staging / "REPORT.zh-CN.md"
        overall_chart = staging / "charts" / "overall-nll-delta.svg"
        source_chart = staging / "charts" / "per-source-nll-delta.svg"
        _atomic_write(summary_path, _json_text(summary))
        _atomic_write(report_path, _markdown(summary))
        _atomic_write(
            overall_chart,
            _delta_svg(
                title="Overall candidate NLL delta vs baseline",
                groups=[str(row["label"]) for row in summary["candidates"]],
                series={
                    "overall Δ": [
                        float(row["overall"]["nll_delta_candidate_minus_baseline"])
                        for row in summary["candidates"]
                    ]
                },
                y_label="Mean NLL delta",
            ),
        )
        source_names = [str(row["source"]) for row in summary["baseline"]["sources"]]
        _atomic_write(
            source_chart,
            _delta_svg(
                title="Per-source candidate NLL delta vs baseline",
                groups=source_names,
                series={
                    str(candidate["label"]): [
                        float(row["nll_delta_candidate_minus_baseline"])
                        for row in candidate["sources"]
                    ]
                    for candidate in summary["candidates"]
                },
                y_label="Token-weighted mean NLL delta",
            ),
        )
        payloads = (summary_path, report_path, overall_chart, source_chart)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "twen_v4_checkpoint_validation_sweep_bundle",
            "inputs_sha256": summary["inputs_sha256"],
            "selection": summary["selection"],
            "files": {
                path.relative_to(staging).as_posix(): _identity(
                    path,
                    relative_to=staging,
                )
                for path in payloads
            },
        }
        manifest_path = staging / "MANIFEST.json"
        _atomic_write(manifest_path, _json_text(manifest))
        _atomic_write(staging / "COMPLETE", _sha256(manifest_path) + "\n")
        if output.exists():
            if output.is_dir() and _tree_identity(output) == _tree_identity(staging):
                shutil.rmtree(staging)
            elif not replace_existing:
                raise SweepError(
                    f"output already exists with different content: {output}; "
                    "use --replace-existing only for an authenticated old bundle"
                )
            else:
                _verify_existing_bundle(output)
                backup = output.with_name(f".{output.name}.replaced-{os.getpid()}")
                if backup.exists():
                    raise SweepError(f"stale replacement backup exists: {backup}")
                os.replace(output, backup)
                try:
                    os.replace(staging, output)
                except BaseException:
                    os.replace(backup, output)
                    raise
                shutil.rmtree(backup)
        else:
            os.replace(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "output": str(output),
        "summary": str(output / "summary.json"),
        "report_zh_cn": str(output / "REPORT.zh-CN.md"),
        "manifest": str(output / "MANIFEST.json"),
        "manifest_sha256": _sha256(output / "MANIFEST.json"),
        "complete": str(output / "COMPLETE"),
        "selection": summary["selection"],
    }


def generate(
    *,
    prepared_manifest: Path,
    baseline_path: Path,
    baseline_label: str,
    candidate_paths: Sequence[tuple[str, Path]],
    output: Path,
    replace_existing: bool = False,
) -> dict[str, Any]:
    summary = build_summary(
        prepared_manifest=prepared_manifest,
        baseline_path=baseline_path,
        baseline_label=baseline_label,
        candidate_paths=candidate_paths,
    )
    return write_bundle(summary, output, replace_existing=replace_existing)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = generate(
            prepared_manifest=args.prepared_manifest,
            baseline_path=args.baseline,
            baseline_label=args.baseline_label,
            candidate_paths=_parse_labeled_paths(args.candidate),
            output=args.output,
            replace_existing=args.replace_existing,
        )
    except SweepError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_json_text(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
