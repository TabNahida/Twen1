#!/usr/bin/env python3
"""Analyze a completed dense training run without mutating the run directory.

The analyzer authenticates the terminal checkpoint, replays the deterministic
data cursor to recover optimizer-batch source composition, separates raw from
source-adjusted loss trends, and emits one JSON summary plus a Chinese Markdown
report.  Both outputs are written with fsync + os.replace.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import html
import json
import math
import os
import re
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCHEMA_VERSION = 2
KIND = "twen_dense_training_analysis"
GIB = 1024**3
METRIC_COMPONENTS = (
    "loss",
    "ntp",
    "teacher_kd",
    "mtp",
    "anchor_kl",
    "hidden_alignment",
    "grad_norm",
)
LOSS_COMPONENTS = (
    "ntp",
    "teacher_kd",
    "anchor_kl",
    "hidden_alignment",
    "mtp",
    "dense_oracle",
    "router_supervision",
    "load_balance",
    "router_z",
)
SOURCE_SUFFIX = re.compile(r"[-_]\d{6}$")
SOURCE_TOKENS_STEP_PREFIX = "source_tokens_this_step/"
SOURCE_TOKENS_TOTAL_PREFIX = "source_tokens/"
DASHBOARD_GPU_ARCHIVE_RELATIVE = "raw/dashboard-gpu-telemetry-last-session.jsonl"
DASHBOARD_GPU_ARCHIVE_SERIALIZATION = "canonical-jsonl-sort-keys-v1"
DASHBOARD_GPU_CAPTURE_SCHEMA_VERSION = 1
DASHBOARD_GPU_SNAPSHOT_ATTEMPTS = 4
DASHBOARD_GPU_INPUTS = (
    ("dashboard_gpu_telemetry_rotated", "gpu-telemetry.jsonl.1"),
    ("dashboard_gpu_telemetry_current", "gpu-telemetry.jsonl"),
)
DASHBOARD_GPU_FIELDS = (
    "power_draw_w",
    "power_limit_w",
    "gpu_utilization_percent",
    "memory_utilization_percent",
    "vram_used_mib",
    "temperature_c",
)
DASHBOARD_GPU_SELECTION_CONDITION = (
    "window_started_at_utc >= session_start and window_ended_at_utc <= session_end"
)
RELEASE_REQUIRED_METRIC_SOURCES = {
    "loss": ("loss",),
    "ntp": ("ntp",),
    "mtp": ("mtp",),
    "grad_norm": ("grad_norm",),
    "nominal_lr": ("lr/adapters", "lr"),
    "adjusted_lr": ("lr_adjusted/adapters",),
}


class AnalysisError(ValueError):
    """Raised when a completed run cannot be authenticated or analyzed."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "output directory (writes analysis.json and REPORT.zh-CN.md), or an explicit .json path"
        ),
    )
    return parser


def _json_text(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _canonical_jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, Mapping):
            raise AnalysisError(f"canonical JSONL row {index} must be an object")
        try:
            line = json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise AnalysisError(f"canonical JSONL row {index} is not serializable") from exc
        lines.append(f"{line}\n")
    return "".join(lines)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _output_paths(output: Path) -> tuple[Path, Path]:
    resolved = output.expanduser().resolve()
    if resolved.suffix.lower() == ".json":
        return resolved, resolved.with_name(f"{resolved.stem}.zh-CN.md")
    return resolved / "analysis.json", resolved / "REPORT.zh-CN.md"


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _stable_file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise AnalysisError(f"input must be a regular non-symlink file: {path}")
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    after = resolved.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise AnalysisError(f"input changed while hashing: {resolved}")
    return {
        "path": str(resolved),
        "size": before.st_size,
        "sha256": digest.hexdigest(),
    }


def _read_stable_bytes(path: Path) -> tuple[bytes, dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise AnalysisError(f"input must be a regular non-symlink file: {path}")
    for _attempt in range(3):
        before = resolved.stat()
        with resolved.open("rb") as handle:
            payload = handle.read()
        after = resolved.stat()
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if before_identity == after_identity and len(payload) == before.st_size:
            return payload, {
                "path": str(resolved),
                "size": before.st_size,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
    raise AnalysisError(f"input changed repeatedly while reading: {resolved}")


def _json_object(payload: bytes, path: Path) -> dict[str, Any]:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise AnalysisError(f"expected JSON object: {path}")
    return value


def _jsonl_rows(
    payload: bytes,
    path: Path,
    *,
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise AnalysisError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    if not rows and not allow_empty:
        raise AnalysisError(f"empty JSONL input: {path}")
    return rows


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise AnalysisError(f"{label} must be numeric, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise AnalysisError(f"{label} must be finite")
    return number


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnalysisError(f"{label} must be an integer")
    return value


def _parse_timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise AnalysisError(f"{label} must be an ISO timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AnalysisError(f"invalid {label}: {value!r}") from exc


def _project_root(run_dir: Path) -> Path:
    for candidate in (run_dir, *run_dir.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    if run_dir.parent.name == "runs":
        return run_dir.parent.parent
    return Path.cwd().resolve()


def _resolve_input_path(value: str, *, run_dir: Path, project_root: Path) -> Path:
    raw = Path(value).expanduser()
    candidates = (
        (raw,) if raw.is_absolute() else (project_root / raw, Path.cwd() / raw, run_dir / raw)
    )
    for candidate in candidates:
        if candidate.is_file() or candidate.is_dir():
            return candidate.resolve()
    raise AnalysisError(f"configured input does not exist: {value}")


def _resolve_terminal_checkpoint(
    value: str,
    *,
    run_dir: Path,
    project_root: Path,
) -> Path:
    raw = Path(value).expanduser()
    if not raw.name or ".." in raw.parts:
        raise AnalysisError(f"unsafe terminal checkpoint path: {value}")
    candidates = (
        (raw,) if raw.is_absolute() else (project_root / raw, Path.cwd() / raw, run_dir / raw)
    )
    for candidate in candidates:
        if candidate.is_symlink():
            raise AnalysisError("terminal checkpoint must not be a symlink")
        if not candidate.is_dir():
            continue
        parent = candidate.parent.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        if parent != run_dir or resolved.parent != run_dir:
            raise AnalysisError("terminal checkpoint must be a direct child of run-dir")
        return resolved
    raise AnalysisError(f"terminal checkpoint does not exist: {value}")


def _source_label(source_path: Any) -> str:
    if not isinstance(source_path, str) or not source_path:
        raise AnalysisError("manifest shard source_path must be a non-empty string")
    name = Path(source_path.replace("\\", "/")).name
    for suffix in (".jsonl", ".parquet", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    label = SOURCE_SUFFIX.sub("", name)
    return label or "unknown"


def _stable_integer(*parts: object) -> int:
    digest = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "little")


def _affine_permutation(index: int, size: int, seed: int, epoch: int) -> int:
    if size <= 0:
        raise AnalysisError("cursor permutation size must be positive")
    if size == 1:
        return 0
    multiplier = _stable_integer("multiplier", seed, epoch) % size
    if multiplier == 0:
        multiplier = 1
    while math.gcd(multiplier, size) != 1:
        multiplier = (multiplier + 1) % size
        if multiplier == 0:
            multiplier = 1
    offset = _stable_integer("offset", seed, epoch) % size
    return (multiplier * index + offset) % size


class _PhaseCursor:
    def __init__(
        self,
        shard_ids: Sequence[str],
        shard_lengths: Sequence[int],
        *,
        seed: int,
    ) -> None:
        if not shard_ids or len(shard_ids) != len(shard_lengths):
            raise AnalysisError("cursor shard IDs and lengths must be aligned")
        if len(set(shard_ids)) != len(shard_ids):
            raise AnalysisError("cursor shard IDs must be unique")
        if any(length <= 0 for length in shard_lengths):
            raise AnalysisError("cursor shard lengths must be positive")
        self.shard_ids = tuple(shard_ids)
        self.shard_lengths = tuple(shard_lengths)
        self.seed = seed
        self.size = sum(shard_lengths)
        self.next_sample = 0
        self.committed_tokens = 0
        self._plans: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}

    def _plan(self, epoch: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        cached = self._plans.get(epoch)
        if cached is not None:
            return cached
        count = len(self.shard_ids)
        order = tuple(
            _affine_permutation(position, count, self.seed, epoch) for position in range(count)
        )
        total = 0
        ends: list[int] = []
        for shard_index in order:
            total += self.shard_lengths[shard_index]
            ends.append(total)
        result = order, tuple(ends)
        self._plans[epoch] = result
        return result

    def shard_at(self, position: int) -> str:
        epoch, epoch_position = divmod(position, self.size)
        order, ends = self._plan(epoch)
        logical_shard = bisect.bisect_right(ends, epoch_position)
        return self.shard_ids[order[logical_shard]]

    def plan(self, count: int) -> tuple[str, ...]:
        if count <= 0:
            raise AnalysisError("global batch sample count must be positive")
        return tuple(self.shard_at(self.next_sample + offset) for offset in range(count))

    def commit(self, *, samples: int, tokens: int) -> None:
        if samples <= 0 or tokens <= 0:
            raise AnalysisError("cursor commit counters must be positive")
        self.next_sample += samples
        self.committed_tokens += tokens


class _SourceReplayCursor:
    def __init__(
        self,
        primary: _PhaseCursor,
        cooldown: _PhaseCursor | None,
        *,
        cooldown_start_tokens: int | None,
    ) -> None:
        if (cooldown is None) != (cooldown_start_tokens is None):
            raise AnalysisError("cooldown cursor and threshold must be configured together")
        self.primary = primary
        self.cooldown = cooldown
        self.cooldown_start_tokens = cooldown_start_tokens
        self.committed_tokens = 0
        self.next_sample = 0

    @property
    def phase(self) -> str:
        if (
            self.cooldown is not None
            and self.cooldown_start_tokens is not None
            and self.committed_tokens >= self.cooldown_start_tokens
        ):
            return "cooldown"
        return "primary"

    @property
    def active(self) -> _PhaseCursor:
        if self.phase == "cooldown":
            if self.cooldown is None:
                raise AnalysisError("cooldown phase has no configured cursor")
            return self.cooldown
        return self.primary

    def plan(self, count: int) -> tuple[str, ...]:
        return self.active.plan(count)

    def commit(self, *, samples: int, tokens: int) -> None:
        self.active.commit(samples=samples, tokens=tokens)
        self.next_sample += samples
        self.committed_tokens += tokens


def _manifest_layout(
    manifest: Mapping[str, Any],
) -> tuple[list[str], list[int], dict[str, str]]:
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise AnalysisError("prepared manifest must contain non-empty shards")
    shard_ids: list[str] = []
    lengths: list[int] = []
    source_by_shard: dict[str, str] = {}
    for index, raw in enumerate(shards):
        if not isinstance(raw, Mapping):
            raise AnalysisError(f"manifest shard {index} must be an object")
        shard_id = raw.get("shard_id")
        if not isinstance(shard_id, str) or not shard_id:
            raise AnalysisError(f"manifest shard {index} has invalid shard_id")
        length = _integer(raw.get("sequence_count"), label=f"shard {shard_id} sequence_count")
        if length <= 0:
            raise AnalysisError(f"shard {shard_id} sequence_count must be positive")
        shard_ids.append(shard_id)
        lengths.append(length)
        source_by_shard[shard_id] = _source_label(raw.get("source_path"))
    if len(set(shard_ids)) != len(shard_ids):
        raise AnalysisError("manifest shard IDs must be unique")
    return shard_ids, lengths, source_by_shard


def _event_batch_contract(events: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    starts = [row for row in events if row.get("event") == "session_start"]
    if not starts:
        raise AnalysisError("events.jsonl has no session_start batch contract")
    contracts: list[dict[str, int]] = []
    for index, row in enumerate(starts):
        contract = {
            "gradient_accumulation_steps": _integer(
                row.get("gradient_accumulation_steps"),
                label=f"session_start[{index}].gradient_accumulation_steps",
            ),
            "micro_batch_size": _integer(
                row.get("micro_batch_size"),
                label=f"session_start[{index}].micro_batch_size",
            ),
            "world_size": _integer(
                row.get("world_size"),
                label=f"session_start[{index}].world_size",
            ),
        }
        contracts.append(contract)
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise AnalysisError("session_start batch contracts disagree across resumes")
    contract = contracts[0]
    contract["global_batch_samples"] = (
        contract["gradient_accumulation_steps"]
        * contract["micro_batch_size"]
        * contract["world_size"]
    )
    return contract


def _configured_loss_weights(config: Mapping[str, Any]) -> dict[str, float]:
    losses = config.get("losses")
    if not isinstance(losses, Mapping):
        raise AnalysisError("resolved config losses must be an object")
    weights: dict[str, float] = {}
    for name in LOSS_COMPONENTS:
        raw = losses.get(name, 0.0)
        weights[name] = _finite(raw, label=f"losses.{name}")
    return weights


def _active_loss_components(config: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(name for name, weight in _configured_loss_weights(config).items() if weight != 0.0)


def _available_metric_components(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    return tuple(key for key in METRIC_COMPONENTS if all(row.get(key) is not None for row in rows))


def _validate_series(
    metrics: Sequence[Mapping[str, Any]],
    telemetry: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if len(metrics) != len(telemetry):
        raise AnalysisError("metrics and telemetry row counts differ")
    active_components = _active_loss_components(config)
    required_metric_fields = ("loss", *active_components, "grad_norm", "lr")
    previous_tokens = 0
    for index, (metric, sample) in enumerate(zip(metrics, telemetry, strict=True), 1):
        step = _integer(metric.get("step"), label=f"metrics[{index}].step")
        if step != index:
            raise AnalysisError(f"metrics steps are not contiguous at row {index}")
        tokens = _integer(metric.get("tokens"), label=f"metrics[{index}].tokens")
        tokens_this_step = _integer(
            metric.get("tokens_this_step"), label=f"metrics[{index}].tokens_this_step"
        )
        if tokens <= previous_tokens or tokens - previous_tokens != tokens_this_step:
            raise AnalysisError(f"invalid token commit sequence at step {step}")
        previous_tokens = tokens
        if sample.get("step") != step or sample.get("tokens") != tokens:
            raise AnalysisError(f"metrics/telemetry mismatch at step {step}")
        for key in required_metric_fields:
            _finite(metric.get(key), label=f"metrics[{step}].{key}")
        for key in (
            "compute_step_seconds",
            "wall_clock_step_seconds",
            "gpu_peak_allocated_gib",
            "gpu_peak_reserved_gib",
        ):
            _finite(sample.get(key), label=f"telemetry[{step}].{key}")
    return {
        "metrics_points": len(metrics),
        "telemetry_points": len(telemetry),
        "step_range": [1, len(metrics)],
        "final_tokens": previous_tokens,
        "steps_contiguous": True,
        "tokens_strictly_increasing": True,
        "tokens_this_step_exact": True,
        "metrics_telemetry_step_token_match": True,
        "all_required_values_finite": True,
        "active_loss_components": list(active_components),
        "required_metric_fields": list(required_metric_fields),
        "available_metric_components": list(_available_metric_components(metrics)),
    }


def _loss_formula_validation(
    metrics: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    weights = _configured_loss_weights(config)
    errors: list[float] = []
    reconstructed = 0
    for row in metrics:
        total = 0.0
        usable = True
        for name, weight in weights.items():
            if weight == 0:
                continue
            raw = row.get(name)
            if raw is None:
                usable = False
                break
            total += weight * _finite(raw, label=f"metrics.{name}")
        if usable:
            reconstructed += 1
            errors.append(abs(total - _finite(row.get("loss"), label="metrics.loss")))
    if reconstructed != len(metrics):
        raise AnalysisError("not every loss row can be reconstructed from configured weights")
    maximum = max(errors, default=0.0)
    if maximum > 1e-8:
        raise AnalysisError(f"logged loss formula mismatch: max_abs_error={maximum}")
    return {
        "formula": " + ".join(
            f"{weight:g}*{name}" for name, weight in weights.items() if weight != 0
        ),
        "weights": weights,
        "rows_reconstructed": reconstructed,
        "max_abs_error": maximum,
        "passed": True,
    }


def _checkpoint_validation(
    *,
    run_dir: Path,
    project_root: Path,
    events: Sequence[Mapping[str, Any]],
    rank_session: Mapping[str, Any],
    latest_payload: bytes,
    last_metric: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if rank_session.get("status") != "completed" or not rank_session.get("ended_at_utc"):
        raise AnalysisError("rank0-session.json is not terminal completed")
    completes = [row for row in events if row.get("event") == "train_complete"]
    if len(completes) != 1:
        raise AnalysisError("events.jsonl must contain exactly one train_complete")
    complete = completes[0]
    step = _integer(complete.get("step"), label="train_complete.step")
    tokens = _integer(complete.get("tokens"), label="train_complete.tokens")
    if step != last_metric.get("step") or tokens != last_metric.get("tokens"):
        raise AnalysisError("train_complete does not match final metric")
    checkpoint_value = complete.get("checkpoint")
    if not isinstance(checkpoint_value, str):
        raise AnalysisError("train_complete checkpoint path is missing")
    checkpoint = _resolve_terminal_checkpoint(
        checkpoint_value, run_dir=run_dir, project_root=project_root
    )
    if not checkpoint.is_dir():
        raise AnalysisError("terminal checkpoint must be a non-symlink directory")
    manifest_path = checkpoint / "manifest.json"
    complete_path = checkpoint / "COMPLETE"
    metadata_path = checkpoint / "metadata.json"
    manifest_bytes, manifest_identity = _read_stable_bytes(manifest_path)
    marker_bytes, marker_identity = _read_stable_bytes(complete_path)
    metadata_bytes, metadata_identity = _read_stable_bytes(metadata_path)
    manifest = _json_object(manifest_bytes, manifest_path)
    metadata = _json_object(metadata_bytes, metadata_path)
    marker = marker_bytes.decode("utf-8").strip()
    if marker != manifest_identity["sha256"]:
        raise AnalysisError("terminal checkpoint COMPLETE does not authenticate manifest.json")
    if manifest.get("algorithm") != "sha256" or manifest.get("version") != 1:
        raise AnalysisError("terminal checkpoint manifest contract mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise AnalysisError("terminal checkpoint manifest has no files")
    authenticated: dict[str, Any] = {}
    for relative, expected in sorted(files.items()):
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise AnalysisError("terminal checkpoint file manifest is malformed")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise AnalysisError(f"unsafe checkpoint manifest path: {relative}")
        unresolved_candidate = checkpoint / relative_path
        current = checkpoint
        for part in relative_path.parts:
            current /= part
            if current.is_symlink():
                raise AnalysisError(f"checkpoint manifest path contains symlink: {relative}")
        candidate = unresolved_candidate.resolve(strict=True)
        if not _is_within(candidate, checkpoint):
            raise AnalysisError(f"checkpoint file escapes root: {relative}")
        identity = _stable_file_identity(unresolved_candidate)
        if identity["sha256"] != expected:
            raise AnalysisError(f"terminal checkpoint file hash mismatch: {relative}")
        authenticated[relative] = {
            "size": identity["size"],
            "sha256": identity["sha256"],
        }
    if metadata.get("global_step") != step or metadata.get("committed_tokens") != tokens:
        raise AnalysisError("terminal checkpoint metadata does not match train_complete")
    if metadata.get("kind") != "milestone" or metadata.get("tag") != "complete":
        raise AnalysisError("terminal checkpoint is not milestone-complete")
    if metadata.get("run_id") != config.get("run_id") or metadata.get("stage") != config.get(
        "stage"
    ):
        raise AnalysisError("terminal checkpoint run identity mismatch")
    latest = latest_payload.decode("utf-8").strip()
    if latest != checkpoint.name:
        raise AnalysisError("run latest pointer does not name terminal checkpoint")
    max_tokens = _integer(
        config.get("optimizer", {}).get("max_tokens"),
        label="optimizer.max_tokens",
    )
    if tokens < max_tokens:
        raise AnalysisError("terminal run stopped before optimizer.max_tokens")
    return {
        "passed": True,
        "rank0_status": rank_session["status"],
        "rank0_ended_at_utc": rank_session["ended_at_utc"],
        "train_complete": dict(complete),
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "manifest": manifest_identity,
        "complete_marker": marker_identity,
        "metadata": metadata_identity,
        "authenticated_payload_files": authenticated,
        "authenticated_payload_bytes": sum(item["size"] for item in authenticated.values()),
        "metadata_state": {
            "global_step": metadata["global_step"],
            "committed_tokens": metadata["committed_tokens"],
            "kind": metadata["kind"],
            "tag": metadata["tag"],
            "run_id": metadata["run_id"],
            "stage": metadata["stage"],
        },
        "latest_pointer_matches": True,
        "max_tokens_reached": True,
        "_metadata": metadata,
    }


def _values(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([_finite(row.get(key), label=key) for row in rows], dtype=np.float64)


def _summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = _values(rows, key)
    return {
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _rolling_summary(
    rows: Sequence[Mapping[str, Any]], key: str, window: int
) -> dict[str, Any] | None:
    if len(rows) < window:
        return None
    values = _values(rows, key)
    rolling = np.convolve(values, np.ones(window, dtype=np.float64) / window, mode="valid")
    minimum = int(np.argmin(rolling))
    maximum = int(np.argmax(rolling))
    return {
        "window_steps": window,
        "first": float(rolling[0]),
        "last": float(rolling[-1]),
        "delta": float(rolling[-1] - rolling[0]),
        "min": float(rolling[minimum]),
        "min_end_step": rows[minimum + window - 1]["step"],
        "max": float(rolling[maximum]),
        "max_end_step": rows[maximum + window - 1]["step"],
        "rolling_sample_sd": (float(rolling.std(ddof=1)) if len(rolling) > 1 else 0.0),
    }


def _hac_covariance(
    design: np.ndarray,
    residuals: np.ndarray,
    *,
    lag: int,
) -> tuple[np.ndarray, int]:
    inverse = np.linalg.inv(design.T @ design)
    scores = design * residuals[:, None]
    meat = scores.T @ scores
    effective_lag = min(lag, len(design) - 1)
    for offset in range(1, effective_lag + 1):
        weight = 1.0 - offset / (effective_lag + 1.0)
        gamma = scores[offset:].T @ scores[:-offset]
        meat += weight * (gamma + gamma.T)
    return inverse @ meat @ inverse, effective_lag


def _regression(
    design: np.ndarray,
    values: np.ndarray,
    *,
    coefficient_names: Sequence[str],
    hac_lag: int,
) -> dict[str, Any]:
    if design.ndim != 2 or len(design) != len(values):
        raise AnalysisError("regression design dimensions are invalid")
    if design.shape[1] != len(coefficient_names):
        raise AnalysisError("regression coefficient names do not align")
    rank = int(np.linalg.matrix_rank(design))
    if rank != design.shape[1]:
        raise AnalysisError(
            f"regression design is rank deficient: rank={rank}, width={design.shape[1]}"
        )
    beta = np.linalg.lstsq(design, values, rcond=None)[0]
    residuals = values - design @ beta
    covariance, effective_lag = _hac_covariance(design, residuals, lag=hac_lag)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    total = float(np.sum((values - values.mean()) ** 2))
    residual_sum = float(residuals @ residuals)
    coefficients = {}
    for name, estimate, standard_error in zip(
        coefficient_names, beta, standard_errors, strict=True
    ):
        coefficients[name] = {
            "estimate": float(estimate),
            "hac_standard_error": float(standard_error),
            "confidence_95": [
                float(estimate - 1.96 * standard_error),
                float(estimate + 1.96 * standard_error),
            ],
        }
    return {
        "rows": len(values),
        "rank": rank,
        "width": design.shape[1],
        "hac_lag": effective_lag,
        "coefficients": coefficients,
        "r_squared": 1.0 - residual_sum / total if total > 0 else None,
        "raw_sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "residual_sample_sd": (float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0),
    }


def _phase_rows(
    metrics: Sequence[Mapping[str, Any]],
    *,
    warmup_tokens: int,
) -> tuple[dict[str, list[Mapping[str, Any]]], list[Mapping[str, Any]]]:
    if warmup_tokens < 0:
        raise AnalysisError("optimizer.warmup_tokens must be non-negative")
    learning_rates = [_finite(row.get("lr"), label="metrics.lr") for row in metrics]
    peak = max(learning_rates)
    peak_tolerance = max(abs(peak) * 1e-12, 1e-18)
    warmup = [
        row
        for row in metrics
        if _integer(row.get("tokens"), label="metrics.tokens") <= warmup_tokens
    ]
    post_warmup_primary = [
        row
        for row in metrics
        if row.get("data_phase", "primary") == "primary"
        and _integer(row.get("tokens"), label="metrics.tokens") > warmup_tokens
    ]
    if not post_warmup_primary:
        raise AnalysisError("run has no post-warmup primary metrics")
    post_warmup_primary_steps = {
        _integer(row.get("step"), label="metrics.step") for row in post_warmup_primary
    }
    stable = [
        row
        for row, value in zip(metrics, learning_rates, strict=True)
        if _integer(row.get("step"), label="metrics.step") in post_warmup_primary_steps
        and abs(value - peak) <= peak_tolerance
    ]
    decay = [
        row
        for row, value in zip(metrics, learning_rates, strict=True)
        if _integer(row.get("step"), label="metrics.step") in post_warmup_primary_steps
        and value < peak - peak_tolerance
    ]
    cooldown = [row for row in metrics if row.get("data_phase") == "cooldown"]
    primary = [row for row in metrics if row.get("data_phase", "primary") == "primary"]
    return {
        "warmup": warmup,
        "primary_stable": stable,
        "primary_decay": decay,
        "post_warmup_primary": post_warmup_primary,
        "cooldown": cooldown,
        "primary": primary,
    }, post_warmup_primary


def _phase_statistics(
    metrics: Sequence[Mapping[str, Any]], *, warmup_tokens: int
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    phases, analysis_phase = _phase_rows(metrics, warmup_tokens=warmup_tokens)
    result: dict[str, Any] = {}
    available = _available_metric_components(metrics)
    for name in ("warmup", "primary_stable", "primary_decay", "cooldown"):
        rows = phases[name]
        if not rows:
            result[name] = None
            continue
        result[name] = {
            "points": len(rows),
            "steps": [rows[0]["step"], rows[-1]["step"]],
            "tokens": [rows[0]["tokens"], rows[-1]["tokens"]],
            "applied_lr": [
                _finite(rows[0].get("lr"), label="lr"),
                _finite(rows[-1].get("lr"), label="lr"),
            ],
            "metrics": {key: _summary(rows, key) for key in available},
            "rolling": {
                f"window_{window}": {key: _rolling_summary(rows, key, window) for key in available}
                for window in (100, 152)
            },
        }
    result["analysis_phase"] = {
        "kind": "post_warmup_primary",
        "points": len(analysis_phase),
        "steps": [analysis_phase[0]["step"], analysis_phase[-1]["step"]],
        "tokens": [analysis_phase[0]["tokens"], analysis_phase[-1]["tokens"]],
    }
    return result, analysis_phase


def _training_window_analysis(
    metrics: Sequence[Mapping[str, Any]],
    *,
    requested_window_steps: int = 10,
) -> dict[str, Any]:
    window_steps = min(requested_window_steps, max(1, len(metrics) // 2))
    first = metrics[:window_steps]
    last = metrics[-window_steps:]
    available = _available_metric_components(metrics)
    comparisons: dict[str, Any] = {}
    for key in available:
        first_mean = float(_values(first, key).mean())
        last_mean = float(_values(last, key).mean())
        comparisons[key] = {
            "first_mean": first_mean,
            "last_mean": last_mean,
            "delta": last_mean - first_mean,
            "percent_delta": ((last_mean / first_mean - 1.0) * 100 if first_mean != 0.0 else None),
        }
    return {
        "window_steps": window_steps,
        "first_steps": [first[0]["step"], first[-1]["step"]],
        "last_steps": [last[0]["step"], last[-1]["step"]],
        "metrics": comparisons,
    }


def _load_manifest(
    path: Path,
    *,
    identities: dict[str, Any],
    identity_key: str,
) -> dict[str, Any]:
    payload, identity = _read_stable_bytes(path)
    identities[identity_key] = identity
    return _json_object(payload, path)


def _source_mix_event_contract(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    starts = [row for row in events if row.get("event") == "session_start"]
    if not starts:
        raise AnalysisError("events.jsonl has no session_start source-mix contract")

    def normalize(
        raw: Mapping[str, Any],
        *,
        label: str,
    ) -> dict[str, Any]:
        basis_points = raw.get("effective_basis_points")
        if not isinstance(basis_points, Mapping):
            basis_points = raw.get("basis_points")
        normalized_basis_points: dict[str, int] = {}
        if isinstance(basis_points, Mapping):
            for source, raw_weight in basis_points.items():
                if not isinstance(source, str) or not source:
                    raise AnalysisError(
                        f"{label} source-mix basis-point keys must be non-empty strings"
                    )
                value = _integer(
                    raw_weight,
                    label=f"{label}.source_mix_basis_points.{source}",
                )
                if value <= 0:
                    raise AnalysisError(f"{label} source-mix basis points must be positive")
                normalized_basis_points[source] = value
        enabled = raw.get("enabled")
        if not isinstance(enabled, bool):
            raise AnalysisError(f"{label} source-mix enabled flag must be boolean")
        if enabled and sum(normalized_basis_points.values()) != 10_000:
            raise AnalysisError(f"{label} source-mix basis points must sum to 10000")
        if not enabled and normalized_basis_points:
            raise AnalysisError(f"{label} disabled source mix contains weights")
        return {
            "enabled": enabled,
            "algorithm": raw.get("algorithm"),
            "basis_points": normalized_basis_points,
            "dataset_fingerprint": raw.get("dataset_fingerprint"),
            "source_map_sha256": raw.get("source_map_sha256"),
        }

    contracts: list[dict[str, Any]] = []
    for row in starts:
        nested = row.get("source_mix")
        if not isinstance(nested, Mapping):
            nested = {}
        primary_raw = {
            "enabled": row.get("source_mix_enabled", nested.get("enabled")),
            "algorithm": row.get(
                "source_mix_algorithm",
                nested.get("algorithm"),
            ),
            "effective_basis_points": row.get(
                "source_mix_effective_basis_points",
                nested.get("effective_basis_points"),
            ),
            "basis_points": row.get(
                "source_mix_basis_points",
                nested.get("basis_points"),
            ),
            "dataset_fingerprint": row.get(
                "source_mix_dataset_fingerprint",
                nested.get("dataset_fingerprint"),
            ),
            "source_map_sha256": row.get(
                "source_map_sha256",
                nested.get("source_map_sha256"),
            ),
        }
        primary = normalize(primary_raw, label="primary")
        raw_phases = nested.get("phases")
        if raw_phases is None:
            phases = {"primary": primary}
        else:
            if (
                not isinstance(raw_phases, Mapping)
                or set(raw_phases) != {"primary", "cooldown"}
                or not all(isinstance(value, Mapping) for value in raw_phases.values())
            ):
                raise AnalysisError(
                    "session_start source-mix phases must contain primary and cooldown"
                )
            phases = {
                phase: normalize(
                    raw_phase,  # type: ignore[arg-type]
                    label=phase,
                )
                for phase, raw_phase in raw_phases.items()
            }
            if phases["primary"] != primary:
                raise AnalysisError("flat and phase-specific primary source-mix contracts disagree")
            if not phases["cooldown"]["enabled"]:
                raise AnalysisError("phase-specific cooldown source-mix contract is disabled")
        cooldown_start = nested.get("cooldown_start_tokens")
        if cooldown_start is not None:
            cooldown_start = _integer(
                cooldown_start,
                label="source_mix.cooldown_start_tokens",
            )
            if cooldown_start <= 0:
                raise AnalysisError("source_mix.cooldown_start_tokens must be positive")
        if "cooldown" in phases and cooldown_start is None:
            raise AnalysisError("phase-specific source mix is missing cooldown_start_tokens")
        contracts.append(
            {
                **primary,
                "phases": phases,
                "cooldown_start_tokens": cooldown_start,
            }
        )
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise AnalysisError("session_start source-mix contracts disagree across resumes")
    contract = contracts[0]
    if contract["enabled"] is False:
        raise AnalysisError("logged source-token metrics disagree with source_mix_enabled=false")
    return contract


def _manifest_source_capacity(
    manifest: Mapping[str, Any],
    *,
    sources: Sequence[str],
) -> dict[str, Any]:
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise AnalysisError("prepared manifest must contain non-empty shards")
    sample_capacity = {source: 0 for source in sources}
    token_capacity = {source: 0 for source in sources}
    token_capacity_complete = True
    unmapped_shards: list[str] = []
    for index, raw in enumerate(shards):
        if not isinstance(raw, Mapping):
            raise AnalysisError(f"manifest shard {index} must be an object")
        shard_id = raw.get("shard_id")
        if not isinstance(shard_id, str) or not shard_id:
            raise AnalysisError(f"manifest shard {index} has invalid shard_id")
        source_path = raw.get("source_path")
        if not isinstance(source_path, str) or not source_path:
            raise AnalysisError(f"manifest shard {shard_id} has invalid source_path")
        path_parts = tuple(part for part in source_path.replace("\\", "/").split("/") if part)
        label = _source_label(source_path)
        candidates = [source for source in sources if source == label or source in path_parts]
        if len(candidates) != 1:
            unmapped_shards.append(shard_id)
            continue
        source = candidates[0]
        count = _integer(
            raw.get("sequence_count"),
            label=f"manifest shard {shard_id} sequence_count",
        )
        if count <= 0:
            raise AnalysisError(f"manifest shard {shard_id} sequence_count must be positive")
        sample_capacity[source] += count
        raw_tokens = raw.get("token_count")
        if raw_tokens is None:
            token_capacity_complete = False
        else:
            tokens = _integer(raw_tokens, label=f"manifest shard {shard_id} token_count")
            if tokens <= 0:
                raise AnalysisError(f"manifest shard {shard_id} token_count must be positive")
            token_capacity[source] += tokens
    mapped = not unmapped_shards
    return {
        "source_mapping_complete": mapped,
        "unmapped_shards": unmapped_shards,
        "samples_by_source": sample_capacity if mapped else None,
        "total_samples": sum(sample_capacity.values()) if mapped else None,
        "tokens_by_source": (token_capacity if mapped and token_capacity_complete else None),
        "total_tokens": (
            sum(token_capacity.values()) if mapped and token_capacity_complete else None
        ),
    }


def _combined_manifest_source_capacity(
    phase_inputs: Mapping[
        str,
        tuple[Mapping[str, Any], Sequence[str]],
    ],
    *,
    sources: Sequence[str],
) -> dict[str, Any]:
    """Combine independently authenticated phase capacities by source ID."""

    phase_capacities = {
        phase: _manifest_source_capacity(manifest, sources=phase_sources)
        for phase, (manifest, phase_sources) in phase_inputs.items()
    }
    complete = all(
        bool(capacity["source_mapping_complete"]) for capacity in phase_capacities.values()
    )
    token_complete = complete and all(
        capacity["tokens_by_source"] is not None for capacity in phase_capacities.values()
    )
    samples = dict.fromkeys(sources, 0)
    tokens = dict.fromkeys(sources, 0)
    if complete:
        for capacity in phase_capacities.values():
            phase_samples = capacity["samples_by_source"]
            assert isinstance(phase_samples, Mapping)
            for source, value in phase_samples.items():
                samples[source] += int(value)
            if token_complete:
                phase_tokens = capacity["tokens_by_source"]
                assert isinstance(phase_tokens, Mapping)
                for source, value in phase_tokens.items():
                    tokens[source] += int(value)
    return {
        "source_mapping_complete": complete,
        "unmapped_shards": sorted(
            {
                f"{phase}:{shard_id}"
                for phase, capacity in phase_capacities.items()
                for shard_id in capacity["unmapped_shards"]
            }
        ),
        "samples_by_source": samples if complete else None,
        "total_samples": sum(samples.values()) if complete else None,
        "tokens_by_source": tokens if token_complete else None,
        "total_tokens": sum(tokens.values()) if token_complete else None,
        "phases": phase_capacities,
    }


def _logged_source_token_replay(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    cooldown_manifest: Mapping[str, Any] | None,
    cooldown_manifest_path: Path | None,
    cooldown_start_tokens: int | None,
    metrics: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    batch = _event_batch_contract(events)
    source_mix = _source_mix_event_contract(events)
    phase_contracts = {
        phase: contract for phase, contract in source_mix["phases"].items() if contract["enabled"]
    }
    logged_cooldown_start = source_mix.get("cooldown_start_tokens")
    if logged_cooldown_start != cooldown_start_tokens:
        raise AnalysisError("logged/configured source-mix cooldown thresholds disagree")
    if ("cooldown" in phase_contracts) != (cooldown_manifest is not None):
        raise AnalysisError("phase-specific source-mix contract and cooldown manifest disagree")
    observed_sources = sorted(
        {
            key[len(SOURCE_TOKENS_STEP_PREFIX) :]
            for row in metrics
            for key in row
            if key.startswith(SOURCE_TOKENS_STEP_PREFIX)
        }
    )
    if not observed_sources or any(not source for source in observed_sources):
        raise AnalysisError("logged source-token metrics have invalid source IDs")
    declared_sources = sorted(
        {source for contract in phase_contracts.values() for source in contract["basis_points"]}
    )
    if declared_sources and declared_sources != observed_sources:
        raise AnalysisError("logged source-token IDs disagree with the session source-mix contract")
    sources = declared_sources or observed_sources
    fractions: list[dict[str, float]] = []
    committed = {source: 0 for source in sources}
    phase_committed = {phase: {source: 0 for source in sources} for phase in phase_contracts}
    phase_token_totals = dict.fromkeys(phase_contracts, 0)
    pure_by_phase: Counter[str] = Counter()
    rows_by_phase: Counter[str] = Counter()
    for row in metrics:
        step = _integer(row.get("step"), label="metrics.step")
        phase = str(row.get("data_phase", "primary"))
        phase_contract = phase_contracts.get(phase)
        if phase_contract is None:
            raise AnalysisError(f"metrics step {step} uses undeclared source-mix phase {phase!r}")
        step_keys = {
            key[len(SOURCE_TOKENS_STEP_PREFIX) :]
            for key in row
            if key.startswith(SOURCE_TOKENS_STEP_PREFIX)
        }
        total_keys = {
            key[len(SOURCE_TOKENS_TOTAL_PREFIX) :]
            for key in row
            if key.startswith(SOURCE_TOKENS_TOTAL_PREFIX)
            and not key.startswith(SOURCE_TOKENS_STEP_PREFIX)
        }
        if step_keys != set(sources) or total_keys != set(sources):
            raise AnalysisError(f"incomplete source-token ledger fields at metrics step {step}")
        tokens_this_step = _integer(
            row.get("tokens_this_step"), label=f"metrics[{step}].tokens_this_step"
        )
        step_tokens: dict[str, int] = {}
        for source in sources:
            value = _integer(
                row.get(f"{SOURCE_TOKENS_STEP_PREFIX}{source}"),
                label=f"metrics[{step}].{SOURCE_TOKENS_STEP_PREFIX}{source}",
            )
            if value < 0:
                raise AnalysisError("source tokens must be non-negative")
            if source not in phase_contract["basis_points"] and value != 0:
                raise AnalysisError(
                    f"metrics step {step} assigns tokens to source {source!r} "
                    f"outside phase {phase!r}"
                )
            step_tokens[source] = value
            committed[source] += value
            phase_committed[phase][source] += value
            logged_total = _integer(
                row.get(f"{SOURCE_TOKENS_TOTAL_PREFIX}{source}"),
                label=f"metrics[{step}].{SOURCE_TOKENS_TOTAL_PREFIX}{source}",
            )
            if logged_total != committed[source]:
                raise AnalysisError(
                    f"cumulative source-token ledger mismatch at step {step}: {source}"
                )
        if sum(step_tokens.values()) != tokens_this_step:
            raise AnalysisError(
                f"source-token ledger does not sum to tokens_this_step at step {step}"
            )
        phase_token_totals[phase] += tokens_this_step
        rows_by_phase[phase] += 1
        if sum(value > 0 for value in step_tokens.values()) == 1:
            pure_by_phase[phase] += 1
        fractions.append({source: step_tokens[source] / tokens_this_step for source in sources})
    final_tokens = sum(committed.values())
    expected_token_mass = {
        source: math.fsum(
            phase_token_totals[phase]
            * phase_contracts[phase]["basis_points"].get(source, 0)
            / 10_000
            for phase in phase_contracts
        )
        for source in sources
    }
    expected_basis_points = {
        source: expected_token_mass[source] / final_tokens * 10_000 for source in sources
    }
    observed_basis_points = {
        source: committed[source] / final_tokens * 10_000 for source in sources
    }
    phase_token_mix: dict[str, dict[str, Any]] = {}
    for phase, total in phase_token_totals.items():
        if total <= 0:
            continue
        expected = {
            source: float(phase_contracts[phase]["basis_points"].get(source, 0))
            for source in sources
        }
        observed = {source: phase_committed[phase][source] / total * 10_000 for source in sources}
        phase_token_mix[phase] = {
            "committed_tokens": total,
            "committed_tokens_by_source": phase_committed[phase],
            "expected_basis_points": expected,
            "observed_basis_points": observed,
            "deviation_basis_points": {
                source: observed[source] - expected[source] for source in sources
            },
        }
    phase_manifests: dict[
        str,
        tuple[Mapping[str, Any], Sequence[str]],
    ] = {
        "primary": (
            manifest,
            sorted(phase_contracts["primary"]["basis_points"]),
        )
    }
    if cooldown_manifest is not None:
        phase_manifests["cooldown"] = (
            cooldown_manifest,
            sorted(phase_contracts["cooldown"]["basis_points"]),
        )
    return {
        "contract": {
            "algorithm": source_mix["algorithm"],
            "composition_source": "logged_source_tokens_this_step",
            "composition_unit": "valid_tokens",
            "global_batch_samples": batch["global_batch_samples"],
            "gradient_accumulation_steps": batch["gradient_accumulation_steps"],
            "micro_batch_size": batch["micro_batch_size"],
            "world_size": batch["world_size"],
            "cooldown_start_tokens": cooldown_start_tokens,
            "primary_manifest": str(manifest_path),
            "quality_cooldown_manifest": (
                str(cooldown_manifest_path) if cooldown_manifest_path is not None else None
            ),
            "dataset_fingerprint": source_mix["dataset_fingerprint"],
            "source_map_sha256": source_mix["source_map_sha256"],
            "expected_basis_points": expected_basis_points,
            "phase_contracts": phase_contracts,
        },
        "validation": {
            "all_phase_predictions_match": True,
            "phase_source_contracts_exact": True,
            "source_token_fields_complete": True,
            "source_tokens_sum_to_step_tokens": True,
            "cumulative_source_tokens_exact": True,
            "final_samples": len(metrics) * batch["global_batch_samples"],
            "final_tokens": final_tokens,
            "final_phase": str(metrics[-1].get("data_phase", "primary")),
        },
        "sources": sources,
        "phase_purity": {
            phase: {
                "rows": rows_by_phase[phase],
                "pure_rows": pure_by_phase[phase],
                "pure_fraction": pure_by_phase[phase] / rows_by_phase[phase],
            }
            for phase in sorted(rows_by_phase)
        },
        "token_mix": {
            "committed_tokens_by_source": committed,
            "observed_basis_points": observed_basis_points,
            "deviation_basis_points": {
                source: observed_basis_points[source] - expected_basis_points[source]
                for source in sources
            },
            "phases": phase_token_mix,
        },
        "manifest_capacity": _combined_manifest_source_capacity(
            phase_manifests,
            sources=sources,
        ),
        "_fractions": fractions,
    }


def _source_replay(
    *,
    config: Mapping[str, Any],
    run_dir: Path,
    project_root: Path,
    metrics: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    identities: dict[str, Any],
) -> dict[str, Any]:
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise AnalysisError("resolved config data must be an object")
    primary_value = data.get("manifest_path")
    if not isinstance(primary_value, str):
        raise AnalysisError("data.manifest_path is required for source replay")
    primary_path = _resolve_input_path(primary_value, run_dir=run_dir, project_root=project_root)
    primary_manifest = _load_manifest(
        primary_path, identities=identities, identity_key="primary_manifest"
    )
    configured_primary_sha = data.get("manifest_sha256")
    if (
        isinstance(configured_primary_sha, str)
        and identities["primary_manifest"]["sha256"] != configured_primary_sha
    ):
        raise AnalysisError("primary manifest SHA does not match resolved config")
    cooldown_value = data.get("quality_cooldown_manifest_path")
    cooldown_start = data.get("quality_cooldown_start_tokens")
    cooldown_manifest: Mapping[str, Any] | None = None
    cooldown_path: Path | None = None
    if cooldown_value is not None or cooldown_start is not None:
        if not isinstance(cooldown_value, str):
            raise AnalysisError("quality cooldown manifest path is incomplete")
        cooldown_start_tokens = _integer(
            cooldown_start,
            label="data.quality_cooldown_start_tokens",
        )
        cooldown_path = _resolve_input_path(
            cooldown_value,
            run_dir=run_dir,
            project_root=project_root,
        )
        cooldown_manifest = _load_manifest(
            cooldown_path,
            identities=identities,
            identity_key="quality_cooldown_manifest",
        )
        configured_cooldown_sha = data.get("quality_cooldown_manifest_sha256")
        if (
            isinstance(configured_cooldown_sha, str)
            and identities["quality_cooldown_manifest"]["sha256"] != configured_cooldown_sha
        ):
            raise AnalysisError("cooldown manifest SHA does not match resolved config")
    else:
        cooldown_start_tokens = None
    has_logged_source_tokens = any(key.startswith(SOURCE_TOKENS_STEP_PREFIX) for key in metrics[0])
    if has_logged_source_tokens:
        if any(
            not any(key.startswith(SOURCE_TOKENS_STEP_PREFIX) for key in row) for row in metrics
        ):
            raise AnalysisError("source-token metrics disappear within the run")
        return _logged_source_token_replay(
            manifest=primary_manifest,
            manifest_path=primary_path,
            cooldown_manifest=cooldown_manifest,
            cooldown_manifest_path=cooldown_path,
            cooldown_start_tokens=cooldown_start_tokens,
            metrics=metrics,
            events=events,
        )
    primary_ids, primary_lengths, primary_sources = _manifest_layout(primary_manifest)
    cooldown_cursor: _PhaseCursor | None = None
    cooldown_sources: dict[str, str] = {}
    if cooldown_manifest is not None:
        cooldown_ids, cooldown_lengths, cooldown_sources = _manifest_layout(cooldown_manifest)
        cooldown_cursor = _PhaseCursor(
            cooldown_ids,
            cooldown_lengths,
            seed=_integer(data.get("shuffle_seed"), label="data.shuffle_seed"),
        )
    seed = _integer(data.get("shuffle_seed"), label="data.shuffle_seed")
    replay = _SourceReplayCursor(
        _PhaseCursor(primary_ids, primary_lengths, seed=seed),
        cooldown_cursor,
        cooldown_start_tokens=cooldown_start_tokens,
    )
    batch = _event_batch_contract(events)
    fractions: list[dict[str, float]] = []
    phase_matches = True
    pure_by_phase: Counter[str] = Counter()
    rows_by_phase: Counter[str] = Counter()
    for row in metrics:
        phase = replay.phase
        rows_by_phase[phase] += 1
        source_map = cooldown_sources if phase == "cooldown" else primary_sources
        shard_ids = replay.plan(batch["global_batch_samples"])
        counts = Counter(source_map[shard_id] for shard_id in shard_ids)
        if max(counts.values()) == batch["global_batch_samples"]:
            pure_by_phase[phase] += 1
        fractions.append(
            {
                source: count / batch["global_batch_samples"]
                for source, count in sorted(counts.items())
            }
        )
        phase_matches &= row.get("data_phase", "primary") == phase
        replay.commit(
            samples=batch["global_batch_samples"],
            tokens=_integer(row.get("tokens_this_step"), label="tokens_this_step"),
        )
    if not phase_matches:
        raise AnalysisError("source replay phase does not match metrics data_phase")
    sources = sorted({source for row in fractions for source in row})
    return {
        "contract": {
            "algorithm": "shard-local-affine-v1",
            "seed": seed,
            "global_batch_samples": batch["global_batch_samples"],
            "gradient_accumulation_steps": batch["gradient_accumulation_steps"],
            "micro_batch_size": batch["micro_batch_size"],
            "world_size": batch["world_size"],
            "cooldown_start_tokens": cooldown_start_tokens,
            "primary_manifest": str(primary_path),
            "quality_cooldown_manifest": (
                str(cooldown_path) if cooldown_path is not None else None
            ),
            "source_label": (
                "basename(source_path), strip .jsonl/.parquet/.json and trailing [-_]NNNNNN"
            ),
            "per_row_order": (
                "plan_global_batch -> count source fractions -> commit(tokens_this_step)"
            ),
        },
        "validation": {
            "all_phase_predictions_match": True,
            "final_samples": replay.next_sample,
            "final_tokens": replay.committed_tokens,
            "final_phase": replay.phase,
        },
        "sources": sources,
        "phase_purity": {
            phase: {
                "rows": rows_by_phase[phase],
                "pure_rows": pure_by_phase[phase],
                "pure_fraction": pure_by_phase[phase] / rows_by_phase[phase],
            }
            for phase in sorted(rows_by_phase)
        },
        "_fractions": fractions,
    }


def _source_fixed_effects(
    *,
    metrics: Sequence[Mapping[str, Any]],
    analysis_phase: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
    grad_clip_norm: float,
) -> dict[str, Any]:
    fractions = replay["_fractions"]
    sources = replay["sources"]
    index_by_step = {
        _integer(row.get("step"), label="step"): index for index, row in enumerate(metrics)
    }
    selected_indices = [index_by_step[row["step"]] for row in analysis_phase]
    source_matrix = np.asarray(
        [[fractions[index].get(source, 0.0) for source in sources] for index in selected_indices],
        dtype=np.float64,
    )
    token_axis = _values(analysis_phase, "tokens") / 100_000_000.0
    centered = token_axis - token_axis.mean()
    available = _available_metric_components(analysis_phase)
    logged_token_mix = (
        replay["contract"].get("composition_source") == "logged_source_tokens_this_step"
    )
    reference_source: str | None = None
    modeled_source_indices: set[int] = set()
    if logged_token_mix:
        mean_fractions = source_matrix.mean(axis=0)
        active_indices = [
            index for index, mean_fraction in enumerate(mean_fractions) if mean_fraction > 0.0
        ]
        if not active_indices:
            raise AnalysisError("analysis phase has no active logged source")
        reference_index = max(
            active_indices,
            key=lambda index: mean_fractions[index],
        )
        reference_source = sources[reference_index]
        centered_sources = source_matrix - mean_fractions[None, :]
        # Cooldown-only sources are identically zero in the primary analysis
        # phase. Exact source schedules can also make some active fractions
        # constant or linearly dependent on time. Retain only columns that add
        # identifiable information beyond the intercept and common token trend.
        rank_probe = np.column_stack(
            [
                np.ones(len(source_matrix), dtype=np.float64),
                centered,
            ]
        )
        current_rank = int(np.linalg.matrix_rank(rank_probe))
        modeled_indices: list[int] = []
        for index in active_indices:
            if index == reference_index:
                continue
            candidate = np.column_stack([rank_probe, centered_sources[:, index]])
            candidate_rank = int(np.linalg.matrix_rank(candidate))
            if candidate_rank > current_rank:
                modeled_indices.append(index)
                modeled_source_indices.add(index)
                rank_probe = candidate
                current_rank = candidate_rank
        source_design = np.column_stack(
            [
                np.ones(len(source_matrix), dtype=np.float64),
                centered_sources[:, modeled_indices],
            ]
        )
        source_coefficient_names = ["mean_mixture_intercept"] + [
            f"source_delta_vs_{reference_source}:{sources[index]}" for index in modeled_indices
        ]
        design = np.column_stack([source_design, centered])
        coefficient_names = [*source_coefficient_names, "tokens_per_100m"]
        design_description = (
            "X=[intercept, centered source-token fractions excluding reference "
            f"{reference_source}, centered(tokens/1e8)]; OLS with "
            "Newey-West/Bartlett lag 50"
        )
        model_kind = "mixed_batch_centered_source_fractions"
    else:
        source_design = source_matrix
        source_coefficient_names = [f"source:{source}" for source in sources]
        design = np.column_stack([source_matrix, centered])
        coefficient_names = [*source_coefficient_names, "tokens_per_100m"]
        design_description = (
            "X=[all source fractions, centered(tokens/1e8)], no intercept; "
            "OLS with Newey-West/Bartlett lag 50"
        )
        model_kind = "replayed_batch_source_fixed_effects"
    regressions = {
        key: _regression(
            design,
            _values(analysis_phase, key),
            coefficient_names=coefficient_names,
            hac_lag=50,
        )
        for key in available
    }
    raw_trend_design = np.column_stack([np.ones(len(centered), dtype=np.float64), centered])
    raw_trends = {
        key: _regression(
            raw_trend_design,
            _values(analysis_phase, key),
            coefficient_names=("intercept_at_mid_token", "tokens_per_100m"),
            hac_lag=50,
        )
        for key in available
    }
    source_only_loss = _regression(
        source_design,
        _values(analysis_phase, "loss"),
        coefficient_names=source_coefficient_names,
        hac_lag=50,
    )
    interaction_metrics = tuple(
        key for key in ("loss", "ntp", "teacher_kd", "mtp", "anchor_kl") if key in available
    )
    if logged_token_mix:
        source_specific = {}
        source_interaction_identifiability = {
            "available": False,
            "rank": None,
            "width": None,
            "fallback": (
                "Each optimizer batch mixes nearly the same source proportions. "
                "The report therefore estimates one source-adjusted common token "
                "slope and source-composition deltas, not per-source loss curves."
            ),
        }
    else:
        interactions = np.column_stack([source_matrix, source_matrix * centered[:, None]])
        interaction_names = [f"source:{source}" for source in sources] + [
            f"source_slope_per_100m:{source}" for source in sources
        ]
        interaction_rank = int(np.linalg.matrix_rank(interactions))
        interaction_width = interactions.shape[1]
    if not logged_token_mix and interaction_rank == interaction_width:
        source_specific = {
            key: _regression(
                interactions,
                _values(analysis_phase, key),
                coefficient_names=interaction_names,
                hac_lag=50,
            )
            for key in interaction_metrics
        }
        source_interaction_identifiability = {
            "available": True,
            "rank": interaction_rank,
            "width": interaction_width,
            "fallback": None,
        }
    elif not logged_token_mix:
        source_specific = {}
        source_interaction_identifiability = {
            "available": False,
            "rank": interaction_rank,
            "width": interaction_width,
            "fallback": (
                "Report source levels from the common-slope model and use its "
                "shared tokens_per_100m coefficient; per-source slopes are not "
                "identifiable from this optimizer-batch mixture."
            ),
        }
    source_rows: dict[str, Any] = {}
    for source_index, source in enumerate(sources):
        pure_positions = [
            position
            for position, index in enumerate(selected_indices)
            if fractions[index].get(source, 0.0) == 1.0
        ]
        grad_norms = np.asarray(
            [
                _finite(analysis_phase[position].get("grad_norm"), label="grad_norm")
                for position in pure_positions
            ],
            dtype=np.float64,
        )
        item: dict[str, Any] = {
            "sample_fraction": float(source_matrix[:, source_index].mean()),
            "equivalent_rows": float(source_matrix[:, source_index].sum()),
            "pure_rows": len(pure_positions),
            "mid_token": round(float(token_axis.mean() * 100_000_000)),
        }
        if len(grad_norms):
            item["pure_grad_norm"] = {
                "mean": float(grad_norms.mean()),
                "configured_threshold": grad_clip_norm,
                "clip_gt_configured_fraction": float(np.mean(grad_norms > grad_clip_norm)),
                "clip_gt_1_fraction": float(np.mean(grad_norms > 1.0)),
                "clip_gt_2_fraction": float(np.mean(grad_norms > 2.0)),
                "clip_gt_5_fraction": float(np.mean(grad_norms > 5.0)),
            }
        for key in interaction_metrics:
            if logged_token_mix:
                item[f"{key}_at_mid_token"] = None
                if source == reference_source:
                    effect = 0.0
                elif source_index not in modeled_source_indices:
                    effect = None
                else:
                    coefficient = f"source_delta_vs_{reference_source}:{source}"
                    effect = regressions[key]["coefficients"][coefficient]["estimate"]
                item[f"{key}_effect_vs_reference_at_mid_token"] = effect
                item[f"{key}_slope_per_100m"] = regressions[key]["coefficients"]["tokens_per_100m"][
                    "estimate"
                ]
                item[f"{key}_slope_kind"] = "source_adjusted_common_slope"
            elif key in source_specific:
                item[f"{key}_at_mid_token"] = source_specific[key]["coefficients"][
                    f"source:{source}"
                ]["estimate"]
                item[f"{key}_slope_per_100m"] = source_specific[key]["coefficients"][
                    f"source_slope_per_100m:{source}"
                ]["estimate"]
                item[f"{key}_slope_kind"] = "source_specific"
            else:
                item[f"{key}_at_mid_token"] = regressions[key]["coefficients"][f"source:{source}"][
                    "estimate"
                ]
                item[f"{key}_slope_per_100m"] = regressions[key]["coefficients"]["tokens_per_100m"][
                    "estimate"
                ]
                item[f"{key}_slope_kind"] = "common_slope_fallback"
        source_rows[source] = item
    loss_values = _values(analysis_phase, "loss")
    centered_loss = loss_values - loss_values.mean()
    acf = {}
    for lag in (1, 20, 50, 75, 76, 77, 100, 152, 228):
        if len(centered_loss) > lag:
            acf[str(lag)] = _pearson(
                centered_loss[:-lag],
                centered_loss[lag:],
            )
    return {
        "phase": {
            "kind": "post_warmup_primary",
            "includes": ["primary_stable", "primary_decay"],
            "excludes": ["warmup", "cooldown"],
            "points": len(analysis_phase),
            "steps": [analysis_phase[0]["step"], analysis_phase[-1]["step"]],
            "tokens": [analysis_phase[0]["tokens"], analysis_phase[-1]["tokens"]],
        },
        "model_kind": model_kind,
        "composition_unit": replay["contract"].get("composition_unit", "optimizer_batch_samples"),
        "reference_source": reference_source,
        "design": design_description,
        "source_only_loss": source_only_loss,
        "raw_trends": raw_trends,
        "common_slopes": regressions,
        "source_interactions": source_specific,
        "source_interaction_identifiability": source_interaction_identifiability,
        "sources": source_rows,
        "raw_loss_autocorrelation": acf,
    }


def _cooldown_analysis(
    *,
    metrics: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
) -> dict[str, Any] | None:
    cooldown_indices = [
        index for index, row in enumerate(metrics) if row.get("data_phase") == "cooldown"
    ]
    if not cooldown_indices:
        return None
    primary = [row for row in metrics if row.get("data_phase", "primary") == "primary"]
    cooldown = [metrics[index] for index in cooldown_indices]
    boundary_window = min(20, len(primary), len(cooldown))
    boundary: dict[str, Any] = {"window_steps": boundary_window, "metrics": {}}
    for key in ("loss", "ntp", "teacher_kd", "mtp", "anchor_kl", "grad_norm", "lr"):
        if key not in primary[-1] or key not in cooldown[0]:
            continue
        left = _values(primary[-boundary_window:], key)
        right = _values(cooldown[:boundary_window], key)
        delta = float(right.mean() - left.mean())
        boundary["metrics"][key] = {
            "last_primary_mean": float(left.mean()),
            "first_cooldown_mean": float(right.mean()),
            "delta": delta,
            "percent_delta_vs_primary": (
                delta / float(left.mean()) * 100 if float(left.mean()) != 0 else None
            ),
        }
    sources = replay["sources"]
    fractions = replay["_fractions"]
    cooldown_start_tokens = replay["contract"].get("cooldown_start_tokens")
    if not isinstance(cooldown_start_tokens, int):
        raise AnalysisError("cooldown replay contract has no token threshold")
    source_matrix = np.asarray(
        [[fractions[index].get(source, 0.0) for source in sources] for index in cooldown_indices],
        dtype=np.float64,
    )
    token_axis = (_values(cooldown, "tokens") - float(cooldown_start_tokens)) / 10_000_000.0
    lr_axis = _values(cooldown, "lr") / 0.0001
    available = _available_metric_components(cooldown)

    def conditioned(axis: np.ndarray, axis_name: str) -> dict[str, Any]:
        centered_axis = axis - axis.mean()
        design_columns = [centered_axis]
        current = centered_axis[:, None]
        current_rank = int(np.linalg.matrix_rank(current))
        modeled_sources: list[str] = []
        for source_index, source in enumerate(sources):
            column = source_matrix[:, source_index]
            if not np.any(column > 0):
                continue
            candidate = np.column_stack([current, column])
            candidate_rank = int(np.linalg.matrix_rank(candidate))
            if candidate_rank > current_rank:
                design_columns.insert(-1, column)
                current = candidate
                current_rank = candidate_rank
                modeled_sources.append(source)
        design = np.column_stack(design_columns)
        names = [f"source:{source}" for source in modeled_sources] + [axis_name]
        return {
            key: _regression(
                design,
                _values(cooldown, key),
                coefficient_names=names,
                hac_lag=20,
            )
            for key in available
        }

    return {
        "points": len(cooldown),
        "steps": [cooldown[0]["step"], cooldown[-1]["step"]],
        "tokens": [cooldown[0]["tokens"], cooldown[-1]["tokens"]],
        "applied_lr": [
            _finite(cooldown[0].get("lr"), label="lr"),
            _finite(cooldown[-1].get("lr"), label="lr"),
        ],
        "raw_loss_lr_pearson": _pearson(
            _values(cooldown, "loss"),
            _values(cooldown, "lr"),
        ),
        "boundary": boundary,
        "source_adjusted_tokens_per_10m": conditioned(token_axis, "tokens_per_10m"),
        "source_adjusted_lr_per_1e4": conditioned(lr_axis, "adapter_lr_per_1e-4"),
        "identifiability": (
            "cooldown 内 LR 是 token 的确定性非线性函数; 在 "
            f"{cooldown_start_tokens:,}-token 边界, prepared/KD shard 集合与 "
            "LR schedule 同时变化, 因此这些系数不能识别 LR 的因果效应。"
        ),
    }


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        return None
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = math.sqrt(
        float(left_centered @ left_centered) * float(right_centered @ right_centered)
    )
    if denominator == 0:
        return None
    return float((left_centered @ right_centered) / denominator)


def _configured_grad_clip(config: Mapping[str, Any]) -> float:
    optimizer = config.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise AnalysisError("resolved config optimizer must be an object")
    threshold = _finite(
        optimizer.get("grad_clip_norm"),
        label="optimizer.grad_clip_norm",
    )
    if threshold <= 0:
        raise AnalysisError("optimizer.grad_clip_norm must be positive")
    return threshold


def _clip_statistics(
    metrics: Sequence[Mapping[str, Any]],
    phase_statistics: Mapping[str, Any],
    *,
    configured_threshold: float,
    warmup_tokens: int,
) -> dict[str, Any]:
    phases, _ = _phase_rows(metrics, warmup_tokens=warmup_tokens)
    result: dict[str, Any] = {"configured_threshold": configured_threshold}
    for name in ("warmup", "primary_stable", "primary_decay", "cooldown"):
        rows = phases[name]
        if not rows:
            result[name] = None
            continue
        values = _values(rows, "grad_norm")
        result[name] = {
            "points": len(values),
            "gt_configured_fraction": float(np.mean(values > configured_threshold)),
            "gt_1_fraction": float(np.mean(values > 1.0)),
            "gt_2_fraction": float(np.mean(values > 2.0)),
            "gt_5_fraction": float(np.mean(values > 5.0)),
        }
    result["phase_reference"] = phase_statistics["analysis_phase"]
    return result


def _performance_segment(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "points": 0,
            "tokens": 0,
            "aggregate_compute_tokens_per_second": None,
            "aggregate_active_wall_tokens_per_second": None,
            "sum_compute_seconds": 0.0,
            "sum_active_wall_seconds": 0.0,
            "wall_over_compute_percent": None,
        }
    tokens = sum(_integer(row.get("tokens_this_step"), label="tokens_this_step") for row in rows)
    compute = sum(
        _finite(row.get("compute_step_seconds"), label="compute_step_seconds") for row in rows
    )
    wall = sum(
        _finite(row.get("wall_clock_step_seconds"), label="wall_clock_step_seconds") for row in rows
    )
    result = {
        "points": len(rows),
        "tokens": tokens,
        "aggregate_compute_tokens_per_second": tokens / compute,
        "aggregate_active_wall_tokens_per_second": tokens / wall,
        "sum_compute_seconds": compute,
        "sum_active_wall_seconds": wall,
        "wall_over_compute_percent": (wall / compute - 1.0) * 100,
    }
    for key in (
        "compute_tokens_per_second",
        "wall_clock_tokens_per_second",
        "compute_step_seconds",
        "wall_clock_step_seconds",
        "data_wait_fraction",
        "gpu_peak_allocated_gib",
        "gpu_peak_reserved_gib",
    ):
        if key in rows[0]:
            result[key] = _summary(rows, key)
    return result


def _performance_analysis(
    telemetry: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    ordinary = [row for row in telemetry if not bool(row.get("hidden_alignment_step"))]
    alignment = [row for row in telemetry if bool(row.get("hidden_alignment_step"))]
    cooldown_ordinary = [row for row in ordinary if row.get("data_phase") == "cooldown"]
    primary_ordinary = [row for row in ordinary if row.get("data_phase", "primary") == "primary"]
    segments = {
        "all": _performance_segment(telemetry),
        "ordinary": _performance_segment(ordinary),
        "primary_ordinary": _performance_segment(primary_ordinary),
        "cooldown_ordinary": _performance_segment(cooldown_ordinary),
    }
    if alignment:
        segments["alignment"] = _performance_segment(alignment)
    total_memory_values = [
        _integer(row.get("gpu_total_memory_bytes"), label="gpu_total_memory_bytes")
        for row in events
        if row.get("event") == "session_start"
    ]
    gpu_total = total_memory_values[0] if total_memory_values else None
    if total_memory_values and any(value != gpu_total for value in total_memory_values):
        raise AnalysisError("GPU total memory changed across sessions")
    peak_reserved = max(
        _finite(row.get("gpu_peak_reserved_gib"), label="gpu_peak_reserved_gib")
        for row in telemetry
    )
    peak_allocated = max(
        _finite(row.get("gpu_peak_allocated_gib"), label="gpu_peak_allocated_gib")
        for row in telemetry
    )
    power_fields = sorted(
        {
            key
            for row in telemetry
            for key in row
            if any(
                token in key.lower() for token in ("power", "watt", "utilization", "temperature")
            )
        }
    )
    output = {
        "segments": segments,
        "alignment": {
            "points": len(alignment),
            "fraction_steps": len(alignment) / len(telemetry),
            "step_interval": (
                sorted({right["step"] - left["step"] for left, right in pairwise(alignment)})
                if len(alignment) > 1
                else []
            ),
        },
        "memory": {
            "gpu_total_bytes": gpu_total,
            "gpu_total_gib": gpu_total / GIB if gpu_total is not None else None,
            "peak_allocated_gib": peak_allocated,
            "peak_reserved_gib": peak_reserved,
            "reserved_headroom_gib": (
                gpu_total / GIB - peak_reserved if gpu_total is not None else None
            ),
        },
        "power": {
            "telemetry_fields": power_fields,
            "time_series_available": bool(power_fields),
            "note": (
                "No power/utilization/temperature time-series fields are present."
                if not power_fields
                else None
            ),
        },
    }
    if alignment and ordinary:
        median_ordinary = float(np.median(_values(ordinary, "compute_step_seconds")))
        actual = sum(
            _finite(row.get("compute_step_seconds"), label="compute_step_seconds")
            for row in telemetry
        )
        ideal = (
            sum(
                _finite(row.get("compute_step_seconds"), label="compute_step_seconds")
                for row in ordinary
            )
            + len(alignment) * median_ordinary
        )
        output["alignment"]["estimated_compute_time_penalty_percent"] = (actual / ideal - 1.0) * 100
    return output


def _nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    if not values or not 0 < percentile <= 1:
        raise AnalysisError("nearest-rank percentile inputs are invalid")
    ordered = sorted(values)
    index = math.ceil(percentile * len(ordered)) - 1
    return ordered[index]


def _dashboard_gpu_capture(
    selected: Sequence[tuple[datetime, datetime, str, Mapping[str, Any]]],
    *,
    snapshot_input_keys: Sequence[str],
) -> dict[str, Any]:
    rows = [dict(row) for _started, _ended, _identity_key, row in selected]
    source_input_keys = [identity_key for _started, _ended, identity_key, _row in selected]
    payload = _canonical_jsonl_text(rows)
    encoded = payload.encode("utf-8")
    return {
        "captured_buckets": rows,
        "raw_capture": {
            "schema_version": DASHBOARD_GPU_CAPTURE_SCHEMA_VERSION,
            "bundle_path": DASHBOARD_GPU_ARCHIVE_RELATIVE,
            "encoding": "utf-8",
            "serialization": DASHBOARD_GPU_ARCHIVE_SERIALIZATION,
            "row_count": len(rows),
            "snapshot_input_keys": list(snapshot_input_keys),
            "source_input_keys_by_row": source_input_keys,
            "size": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        },
    }


def _dashboard_gpu_file_state(path: Path) -> tuple[int, int, int, int, int, int] | None:
    try:
        value = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise AnalysisError(
            f"Dashboard GPU telemetry input must be a regular non-symlink file: {path}"
        )
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
    )


def _dashboard_gpu_joint_snapshot(
    dashboard_dir: Path,
) -> list[tuple[str, Path, bytes, dict[str, Any]]]:
    candidates = tuple(
        (identity_key, dashboard_dir / filename) for identity_key, filename in DASHBOARD_GPU_INPUTS
    )
    last_race: str | None = None
    for attempt in range(1, DASHBOARD_GPU_SNAPSHOT_ATTEMPTS + 1):
        before = {
            identity_key: _dashboard_gpu_file_state(path) for identity_key, path in candidates
        }
        captured: list[tuple[str, Path, bytes, dict[str, Any]]] = []
        pair_raced = False
        try:
            for identity_key, path in candidates:
                state = before[identity_key]
                if state is None:
                    continue
                if _dashboard_gpu_file_state(path) != state:
                    pair_raced = True
                    break
                payload, identity = _read_stable_bytes(path)
                if _dashboard_gpu_file_state(path) != state:
                    pair_raced = True
                    break
                captured.append((identity_key, path, payload, identity))
        except FileNotFoundError:
            last_race = f"attempt {attempt}: input disappeared while opening"
            continue
        except AnalysisError as exc:
            if "changed repeatedly while reading" not in str(exc):
                raise
            last_race = f"attempt {attempt}: {exc}"
            continue
        if pair_raced:
            last_race = f"attempt {attempt}: input changed around individual read"
            continue
        after = {identity_key: _dashboard_gpu_file_state(path) for identity_key, path in candidates}
        if before != after:
            last_race = f"attempt {attempt}: rotated/created/changed across pair read"
            continue
        if any(
            identity["size"] != before[identity_key][2] for identity_key, _, _, identity in captured
        ):
            last_race = f"attempt {attempt}: captured identity differs from pair state"
            continue
        return captured
    detail = f" ({last_race})" if last_race is not None else ""
    raise AnalysisError(
        "Dashboard GPU telemetry pair did not stabilize after "
        f"{DASHBOARD_GPU_SNAPSHOT_ATTEMPTS} attempts{detail}"
    )


def _dashboard_gpu_scope(
    rank_session: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "last_rank0_session_only",
        "session_id": rank_session.get("session_id"),
        "started_at_utc": rank_session.get("started_at_utc"),
        "ended_at_utc": rank_session.get("ended_at_utc"),
        "caveat": (
            "Dashboard telemetry is filtered to the last rank0 session and does not "
            "represent earlier resumed sessions or the entire run."
        ),
    }


def _dashboard_scope_datetimes(
    scope: Mapping[str, Any],
) -> tuple[datetime, datetime]:
    session_start = _parse_timestamp(
        scope.get("started_at_utc"),
        label="rank0-session.started_at_utc",
    )
    session_end = _parse_timestamp(
        scope.get("ended_at_utc"),
        label="rank0-session.ended_at_utc",
    )
    if session_start.utcoffset() is None or session_end.utcoffset() is None:
        raise AnalysisError("rank0 session timestamps must be timezone-aware")
    if session_end <= session_start:
        raise AnalysisError("rank0 session end must be after its start")
    return session_start, session_end


def _dashboard_gpu_row_window(
    row: Mapping[str, Any],
    *,
    label: str,
) -> tuple[datetime, datetime]:
    if row.get("kind") != "twen_gpu_telemetry_aggregate":
        raise AnalysisError(f"{label} has unsupported GPU telemetry kind")
    started = _parse_timestamp(
        row.get("window_started_at_utc"),
        label=f"{label}.window_started_at_utc",
    )
    ended = _parse_timestamp(
        row.get("window_ended_at_utc"),
        label=f"{label}.window_ended_at_utc",
    )
    if started.utcoffset() is None or ended.utcoffset() is None:
        raise AnalysisError("GPU telemetry timestamps must be timezone-aware")
    if ended < started:
        raise AnalysisError("GPU telemetry window end must not precede its start")
    if ended == started:
        samples = _integer(
            row.get("sample_count"),
            label=f"{label}.sample_count",
        )
        if samples != 1:
            raise AnalysisError("zero-duration GPU telemetry windows must contain one sample")
    return started, ended


def _dashboard_gpu_aggregation_contract() -> dict[str, str]:
    return {
        "weighted_mean": (
            "sum(bucket field mean * available_sample_count) / sum(available_sample_count)"
        ),
        "bucket_mean_nearest_rank_p95": ("nearest-rank ceil(0.95 * available_bucket_count) - 1"),
    }


def _dashboard_gpu_summary(
    selected: Sequence[tuple[datetime, datetime, str, Mapping[str, Any]]],
    *,
    scope: Mapping[str, Any],
    snapshot_input_keys: Sequence[str],
) -> dict[str, Any]:
    session_start, session_end = _dashboard_scope_datetimes(scope)
    allowed_keys = [identity_key for identity_key, _filename in DASHBOARD_GPU_INPUTS]
    expected_snapshot_order = [
        identity_key for identity_key in allowed_keys if identity_key in snapshot_input_keys
    ]
    if list(snapshot_input_keys) != expected_snapshot_order or len(set(snapshot_input_keys)) != len(
        snapshot_input_keys
    ):
        raise AnalysisError("Dashboard GPU telemetry snapshot inventory is invalid")
    rows_by_input: Counter[str] = Counter({identity_key: 0 for identity_key in snapshot_input_keys})
    if any(identity_key not in rows_by_input for _start, _end, identity_key, _row in selected):
        raise AnalysisError("Dashboard GPU telemetry selected row lacks snapshot provenance")
    deterministic = sorted(selected, key=lambda item: (item[0], item[1], item[2]))
    if list(selected) != deterministic:
        raise AnalysisError("Dashboard GPU telemetry selected rows are not deterministic")
    selection = {
        "condition": DASHBOARD_GPU_SELECTION_CONDITION,
        "selected_bucket_count": len(selected),
        "rows_by_input": {},
        "first_window_started_at_utc": None,
        "last_window_ended_at_utc": None,
    }
    if not selected:
        selection["rows_by_input"] = dict(sorted(rows_by_input.items()))
        return {
            "available": False,
            "scope": dict(scope),
            "selection": selection,
            "samples": None,
            "coverage": None,
            "aggregation": _dashboard_gpu_aggregation_contract(),
            "fields": None,
            "note": (
                "Dashboard GPU telemetry files are absent."
                if not snapshot_input_keys
                else "No complete Dashboard GPU telemetry buckets fall within the last session."
            ),
        }

    sample_count = 0
    available_count = 0
    unavailable_count = 0
    sampled_seconds = 0.0
    weighted_sums = {field: 0.0 for field in DASHBOARD_GPU_FIELDS}
    bucket_means: dict[str, list[float]] = {field: [] for field in DASHBOARD_GPU_FIELDS}
    minima: dict[str, list[float]] = {field: [] for field in DASHBOARD_GPU_FIELDS}
    maxima: dict[str, list[float]] = {field: [] for field in DASHBOARD_GPU_FIELDS}
    raw_intervals: set[int] = set()
    for index, (started, ended, identity_key, row) in enumerate(selected, 1):
        checked_start, checked_end = _dashboard_gpu_row_window(
            row,
            label=f"captured Dashboard GPU telemetry row {index}",
        )
        if (
            checked_start != started
            or checked_end != ended
            or started < session_start
            or ended > session_end
        ):
            raise AnalysisError("captured Dashboard GPU telemetry row is outside its scope")
        rows_by_input[identity_key] += 1
        samples = _integer(row.get("sample_count"), label="GPU telemetry sample_count")
        available = _integer(
            row.get("available_sample_count"),
            label="GPU telemetry available_sample_count",
        )
        unavailable = _integer(
            row.get("unavailable_sample_count"),
            label="GPU telemetry unavailable_sample_count",
        )
        if min(samples, available, unavailable) < 0 or samples != available + unavailable:
            raise AnalysisError("GPU telemetry sample counts are inconsistent")
        interval = _integer(
            row.get("raw_sample_interval_ms"),
            label="GPU telemetry raw_sample_interval_ms",
        )
        if interval <= 0:
            raise AnalysisError("GPU telemetry raw sample interval must be positive")
        raw_intervals.add(interval)
        sample_count += samples
        available_count += available
        unavailable_count += unavailable
        sampled_seconds += samples * interval / 1000.0
        if available == 0:
            continue
        raw_fields = row.get("fields")
        if not isinstance(raw_fields, Mapping):
            raise AnalysisError("GPU telemetry fields must be an object")
        for field in DASHBOARD_GPU_FIELDS:
            summary = raw_fields.get(field)
            if not isinstance(summary, Mapping):
                raise AnalysisError(f"GPU telemetry field is missing: {field}")
            mean = _finite(summary.get("mean"), label=f"GPU telemetry {field}.mean")
            minimum = _finite(summary.get("min"), label=f"GPU telemetry {field}.min")
            maximum = _finite(summary.get("max"), label=f"GPU telemetry {field}.max")
            if not minimum <= mean <= maximum:
                raise AnalysisError(f"GPU telemetry {field} summary ordering is invalid")
            weighted_sums[field] += mean * available
            bucket_means[field].append(mean)
            minima[field].append(minimum)
            maxima[field].append(maximum)
    if available_count <= 0:
        raise AnalysisError("selected GPU telemetry has no available samples")
    first_start = selected[0][0]
    last_end = selected[-1][1]
    covered_seconds = sum(
        (ended - started).total_seconds() for started, ended, _key, _row in selected
    )
    internal_gap_seconds = 0.0
    for left, right in pairwise(selected):
        gap = (right[0] - left[1]).total_seconds()
        if gap < 0:
            raise AnalysisError("selected GPU telemetry windows overlap")
        internal_gap_seconds += gap
    session_seconds = (session_end - session_start).total_seconds()
    selection.update(
        {
            "rows_by_input": dict(sorted(rows_by_input.items())),
            "first_window_started_at_utc": first_start.isoformat(),
            "last_window_ended_at_utc": last_end.isoformat(),
        }
    )
    return {
        "available": True,
        "scope": dict(scope),
        "selection": selection,
        "samples": {
            "sample_count": sample_count,
            "available_sample_count": available_count,
            "unavailable_sample_count": unavailable_count,
            "raw_sample_interval_ms_values": sorted(raw_intervals),
        },
        "coverage": {
            "session_duration_seconds": session_seconds,
            "sampled_seconds": sampled_seconds,
            "coverage_fraction_of_session": sampled_seconds / session_seconds,
            "bucket_duration_seconds": covered_seconds,
            "bucket_duration_fraction_of_session": covered_seconds / session_seconds,
            "leading_gap_seconds": (first_start - session_start).total_seconds(),
            "internal_gap_seconds": internal_gap_seconds,
            "trailing_gap_seconds": (session_end - last_end).total_seconds(),
            "reaches_session_end": last_end == session_end,
            "gap_definition": (
                "Internal gaps are wall-clock gaps between adjacent aggregate windows, "
                "including normal collector spacing."
            ),
        },
        "aggregation": _dashboard_gpu_aggregation_contract(),
        "fields": {
            field: {
                "weighted_mean": weighted_sums[field] / available_count,
                "bucket_mean_nearest_rank_p95": _nearest_rank_percentile(
                    bucket_means[field],
                    0.95,
                ),
                "min": min(minima[field]),
                "max": max(maxima[field]),
            }
            for field in DASHBOARD_GPU_FIELDS
        },
        "note": None,
    }


def _dashboard_gpu_telemetry(
    *,
    project_root: Path,
    rank_session: Mapping[str, Any],
    identities: dict[str, Any],
) -> dict[str, Any]:
    dashboard_dir = project_root / ".twen" / "dashboard"
    snapshots = _dashboard_gpu_joint_snapshot(dashboard_dir)
    for identity_key, _filename in DASHBOARD_GPU_INPUTS:
        identities.pop(identity_key, None)
    for identity_key, _path, _payload, identity in snapshots:
        identities[identity_key] = identity
    scope = _dashboard_gpu_scope(rank_session)
    session_start, session_end = _dashboard_scope_datetimes(scope)
    selected: list[tuple[datetime, datetime, str, Mapping[str, Any]]] = []
    for identity_key, path, payload, _identity in snapshots:
        rows = _jsonl_rows(payload, path, allow_empty=True)
        for index, row in enumerate(rows, 1):
            started, ended = _dashboard_gpu_row_window(
                row,
                label=f"{identity_key}[{index}]",
            )
            if started >= session_start and ended <= session_end:
                selected.append((started, ended, identity_key, row))
    selected.sort(key=lambda item: (item[0], item[1], item[2]))
    snapshot_input_keys = [identity_key for identity_key, _path, _payload, _identity in snapshots]
    return {
        **_dashboard_gpu_summary(
            selected,
            scope=scope,
            snapshot_input_keys=snapshot_input_keys,
        ),
        **_dashboard_gpu_capture(
            selected,
            snapshot_input_keys=snapshot_input_keys,
        ),
    }


def _lr_dose(config: Mapping[str, Any], metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    optimizer = config.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise AnalysisError("resolved config optimizer must be an object")
    maximum_tokens = _integer(optimizer.get("max_tokens"), label="optimizer.max_tokens")
    warmup_tokens = _integer(optimizer.get("warmup_tokens"), label="optimizer.warmup_tokens")
    minimum_ratio = _finite(optimizer.get("min_lr_ratio", 0.1), label="optimizer.min_lr_ratio")
    peak = max(_finite(row.get("lr"), label="lr") for row in metrics)
    if peak <= 0:
        raise AnalysisError("observed learning-rate peak must be positive")
    actual_equivalent = sum(
        _finite(row.get("lr"), label="lr")
        / peak
        * _integer(row.get("tokens_this_step"), label="tokens_this_step")
        for row in metrics
    )
    cosine_area = warmup_tokens / 2 + (maximum_tokens - warmup_tokens) * (1 + minimum_ratio) / 2
    schedule = optimizer.get("lr_schedule", "cosine")
    if schedule == "warmup-stable-decay":
        decay_tokens = _integer(optimizer.get("decay_tokens"), label="optimizer.decay_tokens")
        configured_area = (
            warmup_tokens / 2
            + (maximum_tokens - warmup_tokens - decay_tokens)
            + decay_tokens * (1 + minimum_ratio) / 2
        )
    else:
        decay_tokens = None
        configured_area = cosine_area
    proposed_decay = maximum_tokens // 2
    proposed_area = (
        warmup_tokens / 2
        + (maximum_tokens - warmup_tokens - proposed_decay)
        + proposed_decay * (1 + minimum_ratio) / 2
    )
    peak_rates = {
        name: optimizer.get(name)
        for name in ("adapter_lr", "lora_lr", "scale_lr", "router_lr")
        if isinstance(optimizer.get(name), (int, float))
        and not isinstance(optimizer.get(name), bool)
    }
    proposed_rates = {name: float(value) * 0.9 for name, value in peak_rates.items()}
    observed_lr_fields = tuple(
        key
        for key in (
            "lr",
            "lr/adapters",
            "lr/scale",
            "lr_adjusted/adapters",
            "lr_adjustment_factor/adapters",
        )
        if all(row.get(key) is not None for row in metrics)
    )
    observed_learning_rates = {key: _summary(metrics, key) for key in observed_lr_fields}
    adapter_optimizer = optimizer.get("adapter_optimizer", "adamw")
    muon = None
    if adapter_optimizer == "muon":
        muon = {
            "adjust_lr_fn": optimizer.get("muon_adjust_lr_fn"),
            "momentum": optimizer.get("muon_momentum"),
            "nesterov": optimizer.get("muon_nesterov"),
            "ns_steps": optimizer.get("muon_ns_steps"),
            "configured_nominal_adapter_lr": optimizer.get("adapter_lr"),
            "observed_nominal_adapter_lr_peak": (
                observed_learning_rates.get("lr/adapters", {}).get("max")
            ),
            "observed_adjusted_adapter_lr_peak": (
                observed_learning_rates.get("lr_adjusted/adapters", {}).get("max")
            ),
            "observed_adjustment_factor": (
                observed_learning_rates.get("lr_adjustment_factor/adapters", {}).get("max")
            ),
        }
    return {
        "optimizer": {
            "adapter_optimizer": adapter_optimizer,
            "muon": muon,
            "observed_learning_rates": observed_learning_rates,
        },
        "configured": {
            "schedule": schedule,
            "max_tokens": maximum_tokens,
            "warmup_tokens": warmup_tokens,
            "decay_tokens": decay_tokens,
            "min_lr_ratio": minimum_ratio,
            "peak_rates": peak_rates,
            "analytical_peak_equivalent_tokens": configured_area,
            "analytical_average_ratio": configured_area / maximum_tokens,
            "observed_peak_equivalent_tokens": actual_equivalent,
            "observed_average_ratio": actual_equivalent
            / sum(
                _integer(row.get("tokens_this_step"), label="tokens_this_step") for row in metrics
            ),
        },
        "full_cosine": {
            "same_peak_equivalent_tokens": cosine_area,
            "same_peak_relative_to_configured": cosine_area / configured_area,
            "peak_minus_10pct_relative_to_configured": 0.9 * cosine_area / configured_area,
            "peak_minus_25pct_relative_to_configured": 0.75 * cosine_area / configured_area,
        },
        "same_budget_recommendation": {
            "peak_rates_minus_10pct": proposed_rates,
            "lr_schedule": "warmup-stable-decay",
            "warmup_tokens": warmup_tokens,
            "decay_tokens": proposed_decay,
            "min_lr_ratio": minimum_ratio,
            "relative_dose_to_configured": 0.9 * proposed_area / configured_area,
            "reason": (
                "Avoid combining a large peak reduction with full-run cosine when "
                "source-adjusted stable loss is still decreasing."
            ),
        },
    }


def _nonnegative_integer_map(value: Any, *, label: str) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise AnalysisError(f"{label} must be an object")
    result: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise AnalysisError(f"{label} keys must be non-empty strings")
        item = _integer(raw_value, label=f"{label}.{raw_key}")
        if item < 0:
            raise AnalysisError(f"{label}.{raw_key} must be non-negative")
        result[raw_key] = item
    return result


def _data_consumption_analysis(
    *,
    config: Mapping[str, Any],
    replay: Mapping[str, Any],
    checkpoint_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    optimizer = config.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise AnalysisError("resolved config optimizer must be an object")
    max_tokens = _integer(optimizer.get("max_tokens"), label="optimizer.max_tokens")
    final_tokens = _integer(
        replay["validation"].get("final_tokens"),
        label="source replay final_tokens",
    )
    final_samples = _integer(
        replay["validation"].get("final_samples"),
        label="source replay final_samples",
    )
    cursor = checkpoint_metadata.get("data_cursor")
    cursor_extra: Mapping[str, Any] = {}
    if isinstance(cursor, Mapping) and isinstance(cursor.get("extra"), Mapping):
        cursor_extra = cursor["extra"]
    committed_samples = _nonnegative_integer_map(
        cursor_extra.get("committed_samples_by_source"),
        label="checkpoint committed_samples_by_source",
    )
    committed_tokens = _nonnegative_integer_map(
        cursor_extra.get("committed_tokens_by_source"),
        label="checkpoint committed_tokens_by_source",
    )
    logged_mix = replay.get("token_mix")
    if isinstance(logged_mix, Mapping):
        logged_tokens = _nonnegative_integer_map(
            logged_mix.get("committed_tokens_by_source"),
            label="logged committed_tokens_by_source",
        )
        if committed_tokens is not None and logged_tokens != committed_tokens:
            raise AnalysisError("logged source-token totals disagree with the checkpoint cursor")
        committed_tokens = logged_tokens
    if committed_samples is not None and sum(committed_samples.values()) != final_samples:
        raise AnalysisError("checkpoint source sample totals disagree with the final sample count")
    if committed_tokens is not None and sum(committed_tokens.values()) != final_tokens:
        raise AnalysisError("checkpoint source token totals disagree with the final token count")
    capacity = replay.get("manifest_capacity")
    capacity_samples = None
    capacity_tokens = None
    if isinstance(capacity, Mapping):
        capacity_samples = _nonnegative_integer_map(
            capacity.get("samples_by_source"),
            label="manifest samples_by_source",
        )
        capacity_tokens = _nonnegative_integer_map(
            capacity.get("tokens_by_source"),
            label="manifest tokens_by_source",
        )
    repeated_samples_by_source = None
    repeated_tokens_by_source = None
    if committed_samples is not None and capacity_samples is not None:
        if set(committed_samples) != set(capacity_samples):
            raise AnalysisError("checkpoint and manifest source IDs disagree for sample capacity")
        repeated_samples_by_source = {
            source: max(0, committed_samples[source] - capacity_samples[source])
            for source in committed_samples
        }
    if committed_tokens is not None and capacity_tokens is not None:
        if set(committed_tokens) != set(capacity_tokens):
            raise AnalysisError("checkpoint and manifest source IDs disagree for token capacity")
        repeated_tokens_by_source = {
            source: max(0, committed_tokens[source] - capacity_tokens[source])
            for source in committed_tokens
        }
    repeated_samples = (
        sum(repeated_samples_by_source.values()) if repeated_samples_by_source is not None else None
    )
    repeated_tokens = (
        sum(repeated_tokens_by_source.values()) if repeated_tokens_by_source is not None else None
    )
    total_capacity_samples = (
        sum(capacity_samples.values()) if capacity_samples is not None else None
    )
    total_capacity_tokens = sum(capacity_tokens.values()) if capacity_tokens is not None else None
    return {
        "max_tokens": max_tokens,
        "terminal_tokens": final_tokens,
        "max_tokens_overshoot": final_tokens - max_tokens,
        "terminal_samples": final_samples,
        "manifest_capacity": {
            "samples": total_capacity_samples,
            "tokens": total_capacity_tokens,
            "tokens_minus_max_tokens": (
                total_capacity_tokens - max_tokens if total_capacity_tokens is not None else None
            ),
        },
        "committed_samples_by_source": committed_samples,
        "committed_tokens_by_source": committed_tokens,
        "repeated_samples_by_source": repeated_samples_by_source,
        "repeated_tokens_by_source": repeated_tokens_by_source,
        "repeated_samples": repeated_samples,
        "repeated_tokens": repeated_tokens,
        "repeated_sample_fraction": (
            repeated_samples / final_samples
            if repeated_samples is not None and final_samples
            else None
        ),
        "repeated_token_fraction": (
            repeated_tokens / final_tokens if repeated_tokens is not None and final_tokens else None
        ),
        "source_wrap_detected": (
            bool(repeated_samples or repeated_tokens)
            if repeated_samples is not None or repeated_tokens is not None
            else None
        ),
        "admission": {
            "manifest_covers_max_tokens": (
                total_capacity_tokens >= max_tokens if total_capacity_tokens is not None else None
            ),
            "completed_without_source_wrap": (
                not bool(repeated_samples or repeated_tokens)
                if repeated_samples is not None or repeated_tokens is not None
                else None
            ),
        },
    }


def _cursor_release_observation(
    checkpoint_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive no-reuse claims from the authenticated terminal cursor.

    Older non-source-mix runs do not carry an authenticated source map.  Those
    reports remain usable for descriptive analysis, but their release claims
    stay unavailable instead of being guessed from configured capacity.
    """

    cursor = checkpoint_metadata.get("data_cursor")
    if not isinstance(cursor, Mapping):
        return {
            "reference_epochs": None,
            "reference_epoch_max": None,
            "reused_sequences": None,
            "reused_tokens": None,
        }
    raw_epoch = cursor.get("epoch")
    top_epoch = (
        raw_epoch
        if isinstance(raw_epoch, int) and not isinstance(raw_epoch, bool) and raw_epoch >= 0
        else None
    )
    extra = cursor.get("extra")
    if not isinstance(extra, Mapping):
        return {
            "reference_epochs": ([top_epoch] if top_epoch is not None else None),
            "reference_epoch_max": top_epoch,
            "reused_sequences": None,
            "reused_tokens": None,
        }
    if extra.get("kind") in {
        "deterministic-source-mix-quality-cooldown",
        "deterministic-source-mix-cooldown",
    }:
        raw_leaves = (extra.get("primary_cursor"), extra.get("cooldown_cursor"))
        if not all(isinstance(item, Mapping) for item in raw_leaves):
            return {
                "reference_epochs": ([top_epoch] if top_epoch is not None else None),
                "reference_epoch_max": top_epoch,
                "reused_sequences": None,
                "reused_tokens": None,
            }
        leaves = list(raw_leaves)
    elif isinstance(extra.get("source_map"), Mapping):
        leaves = [extra]
    else:
        return {
            "reference_epochs": ([top_epoch] if top_epoch is not None else None),
            "reference_epoch_max": top_epoch,
            "reused_sequences": None,
            "reused_tokens": None,
        }

    reference_epochs = [top_epoch] if top_epoch is not None else []
    reused_sequences = 0
    reused_tokens = 0
    for leaf_index, raw_leaf in enumerate(leaves):
        assert isinstance(raw_leaf, Mapping)
        source_map = raw_leaf.get("source_map")
        committed = raw_leaf.get("committed_samples_by_source")
        if not isinstance(source_map, Mapping) or not isinstance(committed, Mapping):
            raise AnalysisError(
                f"terminal source-mix cursor[{leaf_index}] lacks source-map counters"
            )
        sequence_length = _integer(
            source_map.get("sequence_length"),
            label=f"terminal cursor[{leaf_index}].source_map.sequence_length",
        )
        if sequence_length <= 0:
            raise AnalysisError("terminal source-map sequence length must be positive")
        shards = source_map.get("shards")
        if not isinstance(shards, list) or not shards:
            raise AnalysisError(
                f"terminal source-mix cursor[{leaf_index}] has no source-map shards"
            )
        capacities: dict[str, int] = {}
        for shard_index, raw_shard in enumerate(shards):
            if not isinstance(raw_shard, Mapping):
                raise AnalysisError(
                    f"terminal cursor[{leaf_index}].shards[{shard_index}] is invalid"
                )
            source_id = raw_shard.get("source_id")
            if not isinstance(source_id, str) or not source_id:
                raise AnalysisError("terminal source-map shard has an invalid source ID")
            capacity = _integer(
                raw_shard.get("sequence_count"),
                label=(f"terminal cursor[{leaf_index}].shards[{shard_index}].sequence_count"),
            )
            if capacity <= 0:
                raise AnalysisError("terminal source-map shard capacity must be positive")
            capacities[source_id] = capacities.get(source_id, 0) + capacity
        if set(committed) != set(capacities):
            raise AnalysisError(
                f"terminal cursor[{leaf_index}] committed/source-map inventories differ"
            )
        for source_id in sorted(capacities):
            count = _integer(
                committed[source_id],
                label=f"terminal cursor[{leaf_index}].committed.{source_id}",
            )
            if count < 0:
                raise AnalysisError("terminal source committed count must be non-negative")
            capacity = capacities[source_id]
            if count:
                reference_epochs.append((count - 1) // capacity)
            overflow = max(count - capacity, 0)
            reused_sequences += overflow
            reused_tokens += overflow * sequence_length
    if not reference_epochs:
        reference_epochs.append(0)
    return {
        "reference_epochs": reference_epochs,
        "reference_epoch_max": max(reference_epochs),
        "reused_sequences": reused_sequences,
        "reused_tokens": reused_tokens,
    }


def _fork_checkpoint_binding(
    events: Sequence[Mapping[str, Any]],
    *,
    run_dir: Path,
    project_root: Path,
) -> dict[str, Any] | None:
    """Authenticate the model-only fork sentinel named by the initialization log."""

    initialized = [row for row in events if row.get("event") == "initialized"]
    if not initialized:
        return None
    if len(initialized) != 1:
        raise AnalysisError("events.jsonl must contain at most one initialized event")
    fork_value = initialized[0].get("fork_from")
    if fork_value is None:
        return None
    if not isinstance(fork_value, str) or not fork_value:
        raise AnalysisError("initialized fork_from must be a non-empty path")
    raw = Path(fork_value).expanduser()
    if ".." in raw.parts:
        raise AnalysisError("initialized fork checkpoint path is unsafe")
    candidates = (
        (raw,) if raw.is_absolute() else (project_root / raw, Path.cwd() / raw, run_dir / raw)
    )
    fork: Path | None = None
    for candidate in candidates:
        if candidate.is_symlink():
            raise AnalysisError("initialized fork checkpoint must not be a symlink")
        if candidate.is_dir():
            fork = candidate.resolve(strict=True)
            break
    if fork is None:
        raise AnalysisError("initialized fork checkpoint does not exist")
    manifest_path = fork / "manifest.json"
    complete_path = fork / "COMPLETE"
    manifest_bytes, manifest_identity = _read_stable_bytes(manifest_path)
    marker_bytes, complete_identity = _read_stable_bytes(complete_path)
    manifest = _json_object(manifest_bytes, manifest_path)
    if (
        manifest.get("algorithm") != "sha256"
        or manifest.get("version") != 1
        or not isinstance(manifest.get("files"), Mapping)
        or not manifest["files"]
    ):
        raise AnalysisError("initialized fork checkpoint manifest is invalid")
    if marker_bytes.decode("ascii").strip() != manifest_identity["sha256"]:
        raise AnalysisError(
            "initialized fork checkpoint COMPLETE does not authenticate manifest.json"
        )
    return {
        "path": str(fork),
        "manifest_sha256": manifest_identity["sha256"],
        "complete_sha256": complete_identity["sha256"],
    }


def _release_metric_finiteness(
    metrics: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, bool], dict[str, str | None]]:
    finite: dict[str, bool] = {}
    sources: dict[str, str | None] = {}
    for claim_name, candidates in RELEASE_REQUIRED_METRIC_SOURCES.items():
        source = next(
            (
                field
                for field in candidates
                if all(field in row and row.get(field) is not None for row in metrics)
            ),
            None,
        )
        sources[claim_name] = source
        if source is None:
            finite[claim_name] = False
            continue
        try:
            for row in metrics:
                _finite(row.get(source), label=f"release metric {source}")
        except AnalysisError:
            finite[claim_name] = False
        else:
            finite[claim_name] = True
    return finite, sources


def _release_gate_observation(
    *,
    metrics: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    checkpoint_metadata: Mapping[str, Any],
    terminal: Mapping[str, Any],
    identities: Mapping[str, Any],
    run_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    cursor = _cursor_release_observation(checkpoint_metadata)
    finite_metrics, metric_sources = _release_metric_finiteness(metrics)
    grad_clip_norm = _configured_grad_clip(config)
    clipped = sum(
        _finite(row.get("grad_norm"), label="release metric grad_norm") > grad_clip_norm
        for row in metrics
    )
    fork = _fork_checkpoint_binding(
        events,
        run_dir=run_dir,
        project_root=project_root,
    )
    return {
        **cursor,
        "required_metrics_finite": finite_metrics,
        "required_metric_sources": metric_sources,
        "clip_fraction": clipped / len(metrics),
        "clip_threshold": grad_clip_norm,
        "fork_checkpoint_complete_sha256": (fork["complete_sha256"] if fork is not None else None),
        "fork_checkpoint": fork,
        "source_binding": {
            "terminal_checkpoint_manifest_sha256": terminal["manifest"]["sha256"],
            "terminal_checkpoint_complete_sha256": terminal["complete_marker"]["sha256"],
            "metrics_sha256": identities["metrics"]["sha256"],
            "events_sha256": identities["events"]["sha256"],
            "resolved_config_sha256": identities["resolved_config"]["sha256"],
        },
    }


def _input_bundle(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    files = {
        "metrics": run_dir / "metrics.jsonl",
        "telemetry": run_dir / "telemetry.jsonl",
        "events": run_dir / "events.jsonl",
        "resolved_config": run_dir / "resolved_config.yaml",
        "rank0_session": run_dir / "rank0-session.json",
        "latest": run_dir / "latest",
    }
    payloads: dict[str, bytes] = {}
    identities: dict[str, Any] = {}
    for key, path in files.items():
        payloads[key], identities[key] = _read_stable_bytes(path)
    config = yaml.safe_load(payloads["resolved_config"])
    if not isinstance(config, dict):
        raise AnalysisError("resolved_config.yaml root must be an object")
    return {
        "metrics": _jsonl_rows(payloads["metrics"], files["metrics"]),
        "telemetry": _jsonl_rows(payloads["telemetry"], files["telemetry"]),
        "events": _jsonl_rows(payloads["events"], files["events"]),
        "config": config,
        "rank0_session": _json_object(payloads["rank0_session"], files["rank0_session"]),
        "latest_payload": payloads["latest"],
    }, identities


def analyze_dense_training(run_dir: Path) -> dict[str, Any]:
    resolved_run = run_dir.expanduser().resolve(strict=True)
    if resolved_run.is_symlink() or not resolved_run.is_dir():
        raise AnalysisError("run-dir must be a non-symlink directory")
    project_root = _project_root(resolved_run)
    bundle, identities = _input_bundle(resolved_run)
    metrics = bundle["metrics"]
    telemetry = bundle["telemetry"]
    events = bundle["events"]
    config = bundle["config"]
    integrity = _validate_series(metrics, telemetry, config)
    formula = _loss_formula_validation(metrics, config)
    terminal = _checkpoint_validation(
        run_dir=resolved_run,
        project_root=project_root,
        events=events,
        rank_session=bundle["rank0_session"],
        latest_payload=bundle["latest_payload"],
        last_metric=metrics[-1],
        config=config,
    )
    checkpoint_metadata = terminal.pop("_metadata")
    replay = _source_replay(
        config=config,
        run_dir=resolved_run,
        project_root=project_root,
        metrics=metrics,
        events=events,
        identities=identities,
    )
    grad_clip_norm = _configured_grad_clip(config)
    warmup_tokens = _integer(
        config.get("optimizer", {}).get("warmup_tokens"),
        label="optimizer.warmup_tokens",
    )
    phases, analysis_phase = _phase_statistics(
        metrics,
        warmup_tokens=warmup_tokens,
    )
    training_windows = _training_window_analysis(metrics)
    source_fixed_effects = _source_fixed_effects(
        metrics=metrics,
        analysis_phase=analysis_phase,
        replay=replay,
        grad_clip_norm=grad_clip_norm,
    )
    cooldown = _cooldown_analysis(metrics=metrics, replay=replay)
    clipping = _clip_statistics(
        metrics,
        phases,
        configured_threshold=grad_clip_norm,
        warmup_tokens=warmup_tokens,
    )
    performance = _performance_analysis(telemetry, events)
    dashboard_gpu = _dashboard_gpu_telemetry(
        project_root=project_root,
        rank_session=bundle["rank0_session"],
        identities=identities,
    )
    performance["dashboard_gpu_last_session"] = dashboard_gpu
    performance["power"]["dashboard_last_session_available"] = dashboard_gpu["available"]
    if dashboard_gpu["available"] and not performance["power"]["time_series_available"]:
        performance["power"]["note"] = (
            "Training-step telemetry has no power fields; authenticated Dashboard "
            "aggregates are reported separately for the last rank0 session only."
        )
    lr_dose = _lr_dose(config, metrics)
    data_consumption = _data_consumption_analysis(
        config=config,
        replay=replay,
        checkpoint_metadata=checkpoint_metadata,
    )
    release_gate = _release_gate_observation(
        metrics=metrics,
        events=events,
        config=config,
        checkpoint_metadata=checkpoint_metadata,
        terminal=terminal,
        identities=identities,
        run_dir=resolved_run,
        project_root=project_root,
    )
    metadata_cursor = checkpoint_metadata.get("data_cursor", {})
    metadata_extra = (
        metadata_cursor.get("extra", {}) if isinstance(metadata_cursor, Mapping) else {}
    )
    if isinstance(metadata_cursor, Mapping):
        if metadata_cursor.get("global_sample_index") != replay["validation"]["final_samples"]:
            raise AnalysisError("replayed final sample count does not match checkpoint cursor")
        if metadata_cursor.get("global_token_index") != replay["validation"]["final_tokens"]:
            raise AnalysisError("replayed final tokens do not match checkpoint cursor")
    integrity["source_replay_matches_checkpoint_cursor"] = True
    integrity["checkpoint_cursor_kind"] = (
        metadata_extra.get("kind") if isinstance(metadata_extra, Mapping) else None
    )
    replay_fractions = replay.pop("_fractions")
    del replay_fractions
    completion_event = terminal["train_complete"]
    cooldown_start_tokens = config.get("data", {}).get("quality_cooldown_start_tokens")
    logged_source_mix = (
        replay["contract"].get("composition_source") == "logged_source_tokens_this_step"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "run": {
            "run_dir": str(resolved_run),
            "run_id": config.get("run_id"),
            "track": config.get("track"),
            "stage": config.get("stage"),
            "terminal_step": metrics[-1]["step"],
            "terminal_tokens": metrics[-1]["tokens"],
            "terminal_timestamp_utc": completion_event.get("timestamp_utc"),
        },
        "inputs": identities,
        "terminal_validation": terminal,
        "integrity": integrity,
        "loss_formula": formula,
        "phases": phases,
        "training_windows": training_windows,
        "source_replay": replay,
        "source_adjusted": source_fixed_effects,
        "cooldown_lr_separation": cooldown,
        "clipping": clipping,
        "performance": performance,
        "lr_dose": lr_dose,
        "data_consumption": data_consumption,
        "release_gate": release_gate,
        "interpretation": {
            "raw_loss_plateau": (
                "Every optimizer batch mixes sources at nearly fixed token ratios; "
                "raw loss still contains finite-batch content noise, but it is not "
                "confounded by the source-pure batch ordering used by earlier runs."
                if logged_source_mix
                else "Raw optimizer-step loss is strongly confounded by source-pure "
                "batches and source-order autocorrelation."
            ),
            "cooldown": (
                "The data bundle and LR change at the same "
                f"{cooldown_start_tokens:,}-token boundary; tail loss cannot identify "
                "a causal LR benefit."
                if isinstance(cooldown_start_tokens, int)
                else "No quality cooldown boundary is configured."
            ),
            "next_version_priority": (
                "Require enough unique prepared tokens for the complete formal run "
                "and reject any source cursor wrap before interpreting quality."
                if data_consumption["source_wrap_detected"]
                else "Retain logged source-token accounting and validate the terminal "
                "checkpoint on the frozen held-out set before scaling the budget."
            ),
        },
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.3f}M"
    if abs(number) >= 10_000:
        return f"{number:,.0f}"
    return f"{number:.{digits}f}"


def _fmt_percent(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}%}"


def _svg_number(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.1f}k"
    if absolute != 0 and absolute < 0.001:
        return f"{value:.1e}"
    if absolute < 10:
        return f"{value:.3f}"
    return f"{value:.1f}"


def _downsample_xy(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    maximum_points: int = 1_500,
) -> tuple[list[float], list[float]]:
    if len(x_values) != len(y_values):
        raise AnalysisError("chart x/y lengths differ")
    if len(x_values) <= maximum_points:
        return list(x_values), list(y_values)
    indices = np.linspace(0, len(x_values) - 1, maximum_points, dtype=np.int64)
    return (
        [float(x_values[index]) for index in indices],
        [float(y_values[index]) for index in indices],
    )


def _moving_average_xy(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    window: int,
) -> tuple[list[float], list[float]]:
    if window <= 1 or len(y_values) < window:
        return list(x_values), list(y_values)
    values = np.asarray(y_values, dtype=np.float64)
    rolling = np.convolve(
        values,
        np.ones(window, dtype=np.float64) / window,
        mode="valid",
    )
    return list(x_values[window - 1 :]), [float(value) for value in rolling]


def _svg_line_chart(
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[Mapping[str, Any]],
    y_floor: float | None = None,
    y_ceiling: float | None = None,
) -> str:
    width = 1_320
    height = 700
    left = 104.0
    right = 36.0
    legend_columns = 3
    legend_rows = max(1, math.ceil(len(series) / legend_columns))
    top = 82.0 + 25.0 * legend_rows
    bottom = 78.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    normalized: list[dict[str, Any]] = []
    all_x: list[float] = []
    all_y: list[float] = []
    for item in series:
        raw_x = [float(value) for value in item["x"]]
        raw_y = [float(value) for value in item["y"]]
        finite = [
            (x_value, y_value)
            for x_value, y_value in zip(raw_x, raw_y, strict=True)
            if math.isfinite(x_value) and math.isfinite(y_value)
        ]
        if not finite:
            continue
        x_values, y_values = zip(*finite, strict=True)
        sampled_x, sampled_y = _downsample_xy(x_values, y_values)
        normalized.append(
            {
                **item,
                "x": sampled_x,
                "y": sampled_y,
            }
        )
        all_x.extend(sampled_x)
        all_y.extend(sampled_y)
    if not normalized:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="260" '
            'viewBox="0 0 1320 260">'
            '<rect width="1320" height="260" fill="#ffffff"/>'
            f'<text x="660" y="92" text-anchor="middle" font-size="26" '
            f'font-family="sans-serif">{html.escape(title)}</text>'
            '<text x="660" y="155" text-anchor="middle" font-size="18" '
            'fill="#64748b" font-family="sans-serif">No authenticated samples available</text>'
            "</svg>\n"
        )
    x_min = min(all_x)
    x_max = max(all_x)
    if x_max <= x_min:
        x_max = x_min + 1.0
    y_min = min(all_y) if y_floor is None else y_floor
    y_max = max(all_y) if y_ceiling is None else y_ceiling
    if y_ceiling is None and y_floor is None:
        padding = max((y_max - y_min) * 0.08, abs(y_max) * 0.01, 1e-12)
        y_min -= padding
        y_max += padding
    elif y_ceiling is None:
        y_max += max((y_max - y_min) * 0.08, abs(y_max) * 0.01, 1e-12)
    elif y_floor is None:
        y_min -= max((y_max - y_min) * 0.08, abs(y_max) * 0.01, 1e-12)
    if y_max <= y_min:
        y_max = y_min + 1.0

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    output = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="38" text-anchor="middle" '
        f'font-family="sans-serif" font-size="25" font-weight="600" fill="#0f172a">'
        f"{html.escape(title)}</text>",
    ]
    for index, item in enumerate(normalized):
        column = index % legend_columns
        row = index // legend_columns
        x = left + column * 380
        y = 70 + row * 24
        color = str(item.get("color", "#2563eb"))
        output.extend(
            [
                f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + 28:.1f}" y2="{y:.1f}" '
                f'stroke="{html.escape(color)}" stroke-width="3"/>',
                f'<text x="{x + 36:.1f}" y="{y + 5:.1f}" font-family="sans-serif" '
                f'font-size="14" fill="#334155">{html.escape(str(item["label"]))}</text>',
            ]
        )
    for tick in range(6):
        fraction = tick / 5
        x_value = x_min + fraction * (x_max - x_min)
        x = sx(x_value)
        output.extend(
            [
                f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" '
                f'y2="{top + plot_height:.1f}" stroke="#e2e8f0" stroke-width="1"/>',
                f'<text x="{x:.1f}" y="{top + plot_height + 27:.1f}" '
                f'text-anchor="middle" font-family="sans-serif" font-size="13" '
                f'fill="#475569">{html.escape(_svg_number(x_value))}</text>',
            ]
        )
        y_value = y_min + fraction * (y_max - y_min)
        y = sy(y_value)
        output.extend(
            [
                f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + plot_width:.1f}" '
                f'y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>',
                f'<text x="{left - 12:.1f}" y="{y + 5:.1f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="13" fill="#475569">'
                f"{html.escape(_svg_number(y_value))}</text>",
            ]
        )
    output.extend(
        [
            f'<line x1="{left:.1f}" y1="{top + plot_height:.1f}" '
            f'x2="{left + plot_width:.1f}" y2="{top + plot_height:.1f}" '
            'stroke="#334155" stroke-width="1.5"/>',
            f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" '
            f'y2="{top + plot_height:.1f}" stroke="#334155" stroke-width="1.5"/>',
            f'<text x="{left + plot_width / 2:.1f}" y="{height - 20:.1f}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="15" '
            f'fill="#334155">{html.escape(x_label)}</text>',
            f'<text x="24" y="{top + plot_height / 2:.1f}" text-anchor="middle" '
            f'transform="rotate(-90 24 {top + plot_height / 2:.1f})" '
            f'font-family="sans-serif" font-size="15" fill="#334155">'
            f"{html.escape(y_label)}</text>",
        ]
    )
    for item in normalized:
        points = " ".join(
            f"{sx(x_value):.2f},{sy(y_value):.2f}"
            for x_value, y_value in zip(item["x"], item["y"], strict=True)
        )
        dash = item.get("dash")
        dash_attribute = f' stroke-dasharray="{html.escape(str(dash))}"' if dash else ""
        output.append(
            f'<polyline points="{points}" fill="none" '
            f'stroke="{html.escape(str(item.get("color", "#2563eb")))}" '
            f'stroke-width="{float(item.get("width", 2.0)):.2f}" '
            f'opacity="{float(item.get("opacity", 1.0)):.3f}"'
            f"{dash_attribute}/>"
        )
    output.append("</svg>")
    return "\n".join(output) + "\n"


def _svg_source_mix(analysis: Mapping[str, Any]) -> str:
    replay = analysis["source_replay"]
    sources = replay["sources"]
    token_mix = replay.get("token_mix")
    if not isinstance(token_mix, Mapping):
        return _svg_line_chart(
            title="Source token mix",
            x_label="",
            y_label="",
            series=[],
        )
    expected = replay["contract"].get("expected_basis_points", {})
    observed = token_mix["observed_basis_points"]
    width = 1_500
    row_height = 48
    top = 105
    bottom = 65
    left = 410
    right = 45
    height = top + bottom + row_height * len(sources)
    chart_width = width - left - right
    maximum = max(
        [
            *(float(expected[source]) for source in sources),
            *(float(observed[source]) for source in sources),
        ]
    )
    maximum = max(1_000.0, math.ceil(maximum / 500.0) * 500.0)

    def sx(value: float) -> float:
        return left + value / maximum * chart_width

    output = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="36" text-anchor="middle" '
        'font-family="sans-serif" font-size="25" font-weight="600" '
        'fill="#0f172a">Source token mix: target vs committed</text>',
        '<rect x="420" y="58" width="24" height="10" fill="#94a3b8"/>',
        '<text x="452" y="68" font-family="sans-serif" font-size="14" fill="#334155">target</text>',
        '<rect x="530" y="58" width="24" height="10" fill="#2563eb"/>',
        '<text x="562" y="68" font-family="sans-serif" font-size="14" '
        'fill="#334155">committed</text>',
    ]
    for tick in range(7):
        value = maximum * tick / 6
        x = sx(value)
        output.extend(
            [
                f'<line x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" '
                f'y2="{height - bottom}" stroke="#e2e8f0" stroke-width="1"/>',
                f'<text x="{x:.1f}" y="{height - 28}" text-anchor="middle" '
                'font-family="sans-serif" font-size="13" fill="#475569">'
                f"{value / 100:.1f}%</text>",
            ]
        )
    for index, source in enumerate(sources):
        y = top + index * row_height
        target_value = float(expected[source])
        observed_value = float(observed[source])
        output.extend(
            [
                f'<text x="{left - 15}" y="{y + 19}" text-anchor="end" '
                'font-family="monospace" font-size="13" fill="#334155">'
                f"{html.escape(source)}</text>",
                f'<rect x="{left}" y="{y + 3}" width="{sx(target_value) - left:.2f}" '
                'height="12" rx="2" fill="#94a3b8"/>',
                f'<rect x="{left}" y="{y + 21}" width="{sx(observed_value) - left:.2f}" '
                'height="12" rx="2" fill="#2563eb"/>',
            ]
        )
    output.append("</svg>")
    return "\n".join(output) + "\n"


def _authenticated_plot_rows(
    *,
    run_dir: Path,
    analysis: Mapping[str, Any],
    name: str,
) -> list[dict[str, Any]]:
    path = run_dir / f"{name}.jsonl"
    payload, identity = _read_stable_bytes(path)
    expected = analysis["inputs"].get(name)
    if not isinstance(expected, Mapping) or identity["sha256"] != expected.get("sha256"):
        raise AnalysisError(f"{name}.jsonl changed after analysis")
    return _jsonl_rows(payload, path)


def _captured_dashboard_payload(
    analysis: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    dashboard = analysis["performance"]["dashboard_gpu_last_session"]
    if not isinstance(dashboard, Mapping):
        raise AnalysisError("Dashboard GPU telemetry analysis must be an object")
    raw_rows = dashboard.get("captured_buckets")
    if not isinstance(raw_rows, list):
        raise AnalysisError("Dashboard GPU telemetry capture rows must be a list")
    if any(not isinstance(row, Mapping) for row in raw_rows):
        raise AnalysisError("Dashboard GPU telemetry capture rows must be objects")
    payload = _canonical_jsonl_text(raw_rows)
    encoded = payload.encode("utf-8")
    capture = dashboard.get("raw_capture")
    if not isinstance(capture, Mapping):
        raise AnalysisError("Dashboard GPU telemetry capture identity is missing")
    expected_capture_keys = {
        "schema_version",
        "bundle_path",
        "encoding",
        "serialization",
        "row_count",
        "snapshot_input_keys",
        "source_input_keys_by_row",
        "size",
        "sha256",
    }
    if set(capture) != expected_capture_keys:
        raise AnalysisError("Dashboard GPU telemetry capture schema differs")
    expected_capture = {
        "schema_version": DASHBOARD_GPU_CAPTURE_SCHEMA_VERSION,
        "bundle_path": DASHBOARD_GPU_ARCHIVE_RELATIVE,
        "encoding": "utf-8",
        "serialization": DASHBOARD_GPU_ARCHIVE_SERIALIZATION,
        "row_count": len(raw_rows),
        "size": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
    for key, expected in expected_capture.items():
        if capture.get(key) != expected:
            raise AnalysisError(f"Dashboard GPU telemetry capture {key} mismatch")
    snapshot_keys = capture.get("snapshot_input_keys")
    source_keys = capture.get("source_input_keys_by_row")
    if (
        not isinstance(snapshot_keys, list)
        or any(not isinstance(key, str) for key in snapshot_keys)
        or not isinstance(source_keys, list)
        or len(source_keys) != len(raw_rows)
        or any(not isinstance(key, str) for key in source_keys)
    ):
        raise AnalysisError("Dashboard GPU telemetry capture provenance is invalid")
    inputs = analysis.get("inputs")
    if not isinstance(inputs, Mapping):
        raise AnalysisError("analysis inputs must be an object")
    allowed_keys = [identity_key for identity_key, _filename in DASHBOARD_GPU_INPUTS]
    observed_snapshot_keys = [key for key in allowed_keys if key in inputs]
    if snapshot_keys != observed_snapshot_keys or any(
        key not in snapshot_keys for key in source_keys
    ):
        raise AnalysisError("Dashboard GPU telemetry capture snapshot binding differs")
    for key in snapshot_keys:
        identity = inputs.get(key)
        if (
            not isinstance(identity, Mapping)
            or not isinstance(identity.get("path"), str)
            or not isinstance(identity.get("size"), int)
            or isinstance(identity.get("size"), bool)
            or identity["size"] < 0
            or not isinstance(identity.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]) is None
        ):
            raise AnalysisError(
                f"Dashboard GPU telemetry capture source identity is invalid: {key}"
            )
    scope = dashboard.get("scope")
    if not isinstance(scope, Mapping):
        raise AnalysisError("Dashboard GPU telemetry scope must be an object")
    expected_scope_keys = {
        "kind",
        "session_id",
        "started_at_utc",
        "ended_at_utc",
        "caveat",
    }
    if (
        set(scope) != expected_scope_keys
        or scope.get("kind") != "last_rank0_session_only"
        or scope.get("caveat")
        != (
            "Dashboard telemetry is filtered to the last rank0 session and does not "
            "represent earlier resumed sessions or the entire run."
        )
    ):
        raise AnalysisError("Dashboard GPU telemetry capture scope schema differs")
    selected: list[tuple[datetime, datetime, str, Mapping[str, Any]]] = []
    for index, (raw_row, source_key) in enumerate(
        zip(raw_rows, source_keys, strict=True),
        1,
    ):
        row = dict(raw_row)
        started, ended = _dashboard_gpu_row_window(
            row,
            label=f"Dashboard GPU telemetry capture row {index}",
        )
        selected.append((started, ended, source_key, row))
    recomputed = _dashboard_gpu_summary(
        selected,
        scope=scope,
        snapshot_input_keys=snapshot_keys,
    )
    observed = {
        key: value
        for key, value in dashboard.items()
        if key not in {"captured_buckets", "raw_capture"}
    }
    if _json_text(observed) != _json_text(recomputed):
        raise AnalysisError("Dashboard GPU telemetry derived summary differs from captured rows")
    copied_rows = _jsonl_rows(encoded, Path(DASHBOARD_GPU_ARCHIVE_RELATIVE)) if encoded else []
    return payload, copied_rows


def _dashboard_plot_rows(
    analysis: Mapping[str, Any],
    captured_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    dashboard = analysis["performance"]["dashboard_gpu_last_session"]
    if not dashboard.get("available"):
        return []
    rows: list[dict[str, Any]] = []
    for row in captured_rows:
        row_start = _parse_timestamp(
            row.get("window_started_at_utc"),
            label="GPU telemetry window start",
        )
        row_end = _parse_timestamp(
            row.get("window_ended_at_utc"),
            label="GPU telemetry window end",
        )
        rows.append({**row, "_midpoint": row_start + (row_end - row_start) / 2})
    rows.sort(key=lambda row: row["_midpoint"])
    return rows


def _chart_payloads(
    *,
    analysis: Mapping[str, Any],
    run_dir: Path,
    dashboard_captured_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    metrics = _authenticated_plot_rows(
        run_dir=run_dir,
        analysis=analysis,
        name="metrics",
    )
    telemetry = _authenticated_plot_rows(
        run_dir=run_dir,
        analysis=analysis,
        name="telemetry",
    )
    tokens_m = [float(row["tokens"]) / 1_000_000 for row in metrics]
    colors = {
        "loss": "#0f172a",
        "ntp": "#2563eb",
        "mtp": "#dc2626",
    }
    window = max(3, min(100, len(metrics) // 10))
    loss_series: list[dict[str, Any]] = []
    for key in ("loss", "ntp", "mtp"):
        if not all(row.get(key) is not None for row in metrics):
            continue
        values = [float(row[key]) for row in metrics]
        smooth_x, smooth_y = _moving_average_xy(
            tokens_m,
            values,
            window=window,
        )
        loss_series.extend(
            [
                {
                    "label": f"{key} raw",
                    "x": tokens_m,
                    "y": values,
                    "color": colors[key],
                    "width": 1.2,
                    "opacity": 0.25,
                },
                {
                    "label": f"{key} MA({window})",
                    "x": smooth_x,
                    "y": smooth_y,
                    "color": colors[key],
                    "width": 3.0,
                    "opacity": 1.0,
                },
            ]
        )
    charts = {
        "charts/training_loss.svg": _svg_line_chart(
            title=f"{analysis['run']['run_id']} — loss and moving averages",
            x_label="Committed tokens (millions)",
            y_label="Loss",
            series=loss_series,
        )
    }
    lr_series = []
    for key, label, color in (
        ("lr/adapters", "Adapter nominal LR", "#2563eb"),
        ("lr_adjusted/adapters", "Adapter shape-adjusted LR", "#dc2626"),
        ("lr/scale", "Scale AdamW LR", "#16a34a"),
    ):
        if all(row.get(key) is not None for row in metrics):
            lr_series.append(
                {
                    "label": label,
                    "x": tokens_m,
                    "y": [float(row[key]) for row in metrics],
                    "color": color,
                    "width": 2.5,
                }
            )
    if not lr_series:
        lr_series.append(
            {
                "label": "Applied LR",
                "x": tokens_m,
                "y": [float(row["lr"]) for row in metrics],
                "color": "#2563eb",
                "width": 2.5,
            }
        )
    charts["charts/learning_rate.svg"] = _svg_line_chart(
        title="Nominal, adjusted and scale learning rates",
        x_label="Committed tokens (millions)",
        y_label="Learning rate",
        series=lr_series,
        y_floor=0.0,
    )
    telemetry_tokens_m = [float(row["tokens"]) / 1_000_000 for row in telemetry]
    charts["charts/throughput.svg"] = _svg_line_chart(
        title="Training throughput",
        x_label="Committed tokens (millions)",
        y_label="Tokens / second",
        series=[
            {
                "label": "compute",
                "x": telemetry_tokens_m,
                "y": [float(row["compute_tokens_per_second"]) for row in telemetry],
                "color": "#2563eb",
                "width": 2.2,
            },
            {
                "label": "active wall",
                "x": telemetry_tokens_m,
                "y": [float(row["wall_clock_tokens_per_second"]) for row in telemetry],
                "color": "#f97316",
                "width": 2.2,
            },
        ],
        y_floor=0.0,
    )
    clip = float(analysis["clipping"]["configured_threshold"])
    charts["charts/gradient_norm.svg"] = _svg_line_chart(
        title="Gradient norm and clipping threshold",
        x_label="Committed tokens (millions)",
        y_label="Gradient norm",
        series=[
            {
                "label": "grad norm",
                "x": tokens_m,
                "y": [float(row["grad_norm"]) for row in metrics],
                "color": "#7c3aed",
                "width": 2.2,
            },
            {
                "label": f"clip={clip:g}",
                "x": [tokens_m[0], tokens_m[-1]],
                "y": [clip, clip],
                "color": "#dc2626",
                "width": 2.0,
                "dash": "8 6",
            },
        ],
        y_floor=0.0,
    )
    charts["charts/gpu_memory.svg"] = _svg_line_chart(
        title="CUDA peak memory by optimizer step",
        x_label="Committed tokens (millions)",
        y_label="GiB",
        series=[
            {
                "label": "peak allocated",
                "x": telemetry_tokens_m,
                "y": [float(row["gpu_peak_allocated_gib"]) for row in telemetry],
                "color": "#2563eb",
                "width": 2.3,
            },
            {
                "label": "peak reserved",
                "x": telemetry_tokens_m,
                "y": [float(row["gpu_peak_reserved_gib"]) for row in telemetry],
                "color": "#f97316",
                "width": 2.3,
            },
        ],
        y_floor=0.0,
    )
    dashboard_rows = _dashboard_plot_rows(analysis, dashboard_captured_rows)
    session_start = None
    dashboard = analysis["performance"]["dashboard_gpu_last_session"]
    if dashboard.get("available"):
        session_start = _parse_timestamp(
            dashboard["scope"]["started_at_utc"],
            label="dashboard session start",
        )

    def dashboard_series(field: str) -> tuple[list[float], list[float]]:
        x_values: list[float] = []
        y_values: list[float] = []
        for row in dashboard_rows:
            fields = row.get("fields")
            if not isinstance(fields, Mapping):
                continue
            summary = fields.get(field)
            if not isinstance(summary, Mapping):
                continue
            if session_start is None:
                raise AnalysisError("Dashboard GPU telemetry session start is missing")
            x_values.append((row["_midpoint"] - session_start).total_seconds() / 60)
            y_values.append(_finite(summary.get("mean"), label=f"dashboard {field}"))
        return x_values, y_values

    util_x, utilization = dashboard_series("gpu_utilization_percent")
    power_x, power = dashboard_series("power_draw_w")
    charts["charts/gpu_utilization.svg"] = _svg_line_chart(
        title="Dashboard GPU utilization",
        x_label="Minutes since final rank0 session start",
        y_label="GPU utilization (%)",
        series=[
            {
                "label": "bucket mean utilization",
                "x": util_x,
                "y": utilization,
                "color": "#16a34a",
                "width": 2.0,
            }
        ],
        y_floor=0.0,
        y_ceiling=100.0,
    )
    charts["charts/gpu_power.svg"] = _svg_line_chart(
        title="Dashboard GPU power",
        x_label="Minutes since final rank0 session start",
        y_label="Power (W)",
        series=[
            {
                "label": "bucket mean power",
                "x": power_x,
                "y": power,
                "color": "#dc2626",
                "width": 2.0,
            }
        ],
        y_floor=0.0,
    )
    charts["charts/source_token_mix.svg"] = _svg_source_mix(analysis)
    return charts


def _markdown(analysis: Mapping[str, Any], json_name: str) -> str:
    run = analysis["run"]
    terminal = analysis["terminal_validation"]
    integrity = analysis["integrity"]
    formula = analysis["loss_formula"]
    phases = analysis["phases"]
    windows = analysis["training_windows"]
    replay = analysis["source_replay"]
    source_adjusted = analysis["source_adjusted"]
    cooldown = analysis["cooldown_lr_separation"]
    performance = analysis["performance"]
    lr_dose = analysis["lr_dose"]
    consumption = analysis["data_consumption"]
    mixed_token_batches = (
        replay["contract"].get("composition_source") == "logged_source_tokens_this_step"
    )
    lines = [
        "# Dense 训练终态分析",
        "",
        f"- run: `{run['run_id']}`",
        f"- 终态: step `{run['terminal_step']}` / `{run['terminal_tokens']:,}` tokens",
        f"- checkpoint: `{terminal['checkpoint_name']}`",
        f"- 完整机器可读统计: `{json_name}`",
        "",
        "## 终态与输入认证",
        "",
        "metrics/telemetry step 连续、token 严格递增、逐行匹配; 最终 milestone "
        "checkpoint 的 manifest、COMPLETE、metadata 及全部 payload SHA256 已复算通过。",
        f"本 run 的 active loss 只有 "
        f"`{', '.join(integrity['active_loss_components'])}`, 逐步公式 "
        f"`loss = {formula['formula']}` 在 `{formula['rows_reconstructed']}` 行上的 "
        f"最大绝对误差为 `{formula['max_abs_error']:.3g}`; 零权重的 teacher KD、"
        "anchor KL 和 hidden alignment 不再被错误要求出现在 metrics 中。",
        "",
        "## 阶段",
        "",
        "| 阶段 | 点数 | step | token | applied LR |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("warmup", "warmup"),
        ("primary_stable", "primary stable"),
        ("primary_decay", "primary cosine decay"),
        ("cooldown", "cooldown"),
    ):
        phase = phases.get(key)
        if phase is None:
            continue
        lines.append(
            f"| {label} | {phase['points']} | {phase['steps'][0]}-{phase['steps'][1]} "
            f"| {_fmt(phase['tokens'][0])}-{_fmt(phase['tokens'][1])} "
            f"| {phase['applied_lr'][0]:.3e}→{phase['applied_lr'][1]:.3e} |"
        )
    lines += [
        "",
        f"## 首尾 {windows['window_steps']} step 对比",
        "",
        "| 指标 | 首窗口均值 | 尾窗口均值 | 变化 |",
        "|---|---:|---:|---:|",
    ]
    for key in ("loss", "ntp", "mtp", "grad_norm"):
        values = windows["metrics"].get(key)
        if values is None:
            continue
        lines.append(
            f"| {key} | {_fmt(values['first_mean'], 6)} "
            f"| {_fmt(values['last_mean'], 6)} "
            f"| {values['percent_delta']:+.2f}% |"
        )
    loss_regression = source_adjusted["common_slopes"]["loss"]
    loss_slope = loss_regression["coefficients"]["tokens_per_100m"]
    raw_loss_slope = source_adjusted["raw_trends"]["loss"]["coefficients"]["tokens_per_100m"]
    source_only_r2 = source_adjusted["source_only_loss"]["r_squared"]
    lines += [
        "",
        "## Source-adjusted 学习趋势",
        "",
        "正式回归窗口固定为 warmup 后的全部 primary batch, 包含 stable 与 cosine decay, "
        "排除 warmup 和 quality cooldown; 因此不同 decay 长度的 run 使用同一数据阶段口径。",
        "",
        f"raw loss slope 为 `{raw_loss_slope['estimate']:.5f}/100M tokens`; "
        f"source composition 单独解释窗口内 raw loss 方差的 "
        f"`{_fmt_percent(source_only_r2)}`。控制 source 后, loss slope 为 "
        f"`{loss_slope['estimate']:.5f}/100M tokens`, "
        f"HAC95 CI `[{loss_slope['confidence_95'][0]:.5f}, "
        f"{loss_slope['confidence_95'][1]:.5f}]`。",
    ]
    if mixed_token_batches:
        lines += [
            "",
            f"这里每个 optimizer batch 都按接近固定的 token 比例混合 "
            f"{len(replay['sources'])} 个来源, "
            "因此 batch loss 不能拆成可信的逐来源 NLL; 报告只给混合比例和共同趋势, "
            "不把极小的比例抖动外推成逐来源 loss。",
            "",
            "| Source | 目标占比 | 实际占比 | 偏差(bp) | committed tokens |",
            "|---|---:|---:|---:|---:|",
        ]
        expected = replay["contract"]["expected_basis_points"]
        token_mix = replay["token_mix"]
        for source in replay["sources"]:
            observed = token_mix["observed_basis_points"][source]
            deviation = token_mix["deviation_basis_points"][source]
            committed = token_mix["committed_tokens_by_source"][source]
            lines.append(
                f"| {source} | {expected[source] / 100:.2f}% "
                f"| {observed / 100:.2f}% | {deviation:+.2f} "
                f"| {committed:,} |"
            )
    else:
        lines += [
            "",
            "| Source | 样本占比 | 中点 loss | loss slope/100M | 纯 batch 超裁剪阈值 |",
            "|---|---:|---:|---:|---:|",
        ]
        for source, row in source_adjusted["sources"].items():
            clip = row.get("pure_grad_norm", {}).get("clip_gt_configured_fraction")
            lines.append(
                f"| {source} | {row['sample_fraction']:.2%} "
                f"| {_fmt(row.get('loss_at_mid_token'))} "
                f"| {_fmt(row.get('loss_slope_per_100m'), 5)} "
                f"| {_fmt_percent(clip)} |"
            )
    capacity = consumption["manifest_capacity"]
    lines += [
        "",
        "## 数据容量与回绕",
        "",
        f"- 配置预算 / 实际提交: `{consumption['max_tokens']:,}` / "
        f"`{consumption['terminal_tokens']:,}` tokens, 末 batch overshoot "
        f"`{consumption['max_tokens_overshoot']:,}` tokens。",
    ]
    if capacity["tokens"] is not None:
        lines.append(
            f"- prepared unique capacity: `{capacity['tokens']:,}` tokens / "
            f"`{capacity['samples']:,}` sequences; 相对配置预算只多 "
            f"`{capacity['tokens_minus_max_tokens']:,}` tokens。"
        )
    if consumption["source_wrap_detected"] is not None:
        repeat_detail = (
            f"`{consumption['repeated_samples']:,}` sequences"
            if consumption["repeated_samples"] is not None
            else "`n/a` sequences"
        )
        if consumption["repeated_tokens"] is not None:
            repeat_detail += (
                f" / `{consumption['repeated_tokens']:,}` tokens "
                f"(`{consumption['repeated_token_fraction']:.2%}` of committed tokens)"
            )
        lines += [
            f"- source cursor wrap: `{consumption['source_wrap_detected']}`; "
            f"重复 {repeat_detail}。",
            "- 这不影响 smoke 的数值/性能门, 但该 manifest 不能作为正式长训数据。"
            "正式 profile 必须在启动前硬拒绝任何容量不足或第二 epoch。",
        ]
    if cooldown is not None:
        boundary = cooldown["boundary"]["metrics"]
        lines += [
            "",
            "## Cooldown 与 LR",
            "",
            f"边界窗口中 LR 变化 `{boundary['lr']['percent_delta_vs_primary']:.2f}%`, "
            f"但 loss 变化 `{boundary['loss']['percent_delta_vs_primary']:.2f}%`; "
            f"完整 cooldown raw corr(loss, LR)=`{_fmt(cooldown['raw_loss_lr_pearson'])}`。",
            "",
            cooldown["identifiability"],
        ]
    optimizer = lr_dose["optimizer"]
    lines += ["", "## 优化器与学习率", ""]
    if optimizer["muon"] is not None:
        muon = optimizer["muon"]
        lines += [
            f"- Adapter optimizer: `Muon`; adjust LR: `{muon['adjust_lr_fn']}`。",
            f"- nominal Adapter peak: "
            f"`{muon['observed_nominal_adapter_lr_peak']:.6g}`; "
            f"shape-adjusted peak: `{muon['observed_adjusted_adapter_lr_peak']:.6g}`; "
            f"adjustment factor: `{muon['observed_adjustment_factor']:.3f}x`。",
            f"- scale AdamW peak: "
            f"`{optimizer['observed_learning_rates']['lr/scale']['max']:.6g}`。",
            "- 因此 `1e-4` 只是 Muon nominal LR, 不能直接当成 AdamW `1e-4` 来判断更新幅度。",
        ]
    else:
        lines.append(
            f"- Adapter optimizer: `{optimizer['adapter_optimizer']}`; "
            f"observed peak LR `{lr_dose['configured']['peak_rates'].get('adapter_lr')}`。"
        )
    ordinary = performance["segments"]["ordinary"]
    alignment = performance["segments"].get("alignment")
    lines += [
        "",
        "## 性能与显存",
        "",
        f"- ordinary compute: `{ordinary['aggregate_compute_tokens_per_second']:.1f} tok/s`",
        f"- ordinary active-wall: "
        f"`{ordinary['aggregate_active_wall_tokens_per_second']:.1f} tok/s`",
    ]
    if alignment is not None:
        lines.append(
            f"- alignment compute: `{alignment['aggregate_compute_tokens_per_second']:.1f} tok/s`"
        )
    memory = performance["memory"]
    lines += [
        f"- peak allocated/reserved: `{memory['peak_allocated_gib']:.3f}` / "
        f"`{memory['peak_reserved_gib']:.3f} GiB`",
        f"- reserved headroom: `{_fmt(memory['reserved_headroom_gib'], 3)} GiB`",
    ]
    dashboard_gpu = performance["dashboard_gpu_last_session"]
    dashboard_capture = dashboard_gpu["raw_capture"]
    lines += ["", "## Dashboard GPU telemetry", ""]
    if dashboard_gpu["available"]:
        power = dashboard_gpu["fields"]["power_draw_w"]
        utilization = dashboard_gpu["fields"]["gpu_utilization_percent"]
        vram = dashboard_gpu["fields"]["vram_used_mib"]
        temperature = dashboard_gpu["fields"]["temperature_c"]
        coverage = dashboard_gpu["coverage"]
        selection = dashboard_gpu["selection"]
        samples = dashboard_gpu["samples"]
        lines += [
            f"- 范围: 最后一次 rank0 session `{dashboard_gpu['scope']['session_id']}`",
            f"- bucket/sample: `{selection['selected_bucket_count']:,}` / "
            f"`{samples['available_sample_count']:,}` available",
            f"- power weighted mean / bucket-mean p95 / max: "
            f"`{power['weighted_mean']:.2f}` / "
            f"`{power['bucket_mean_nearest_rank_p95']:.2f}` / "
            f"`{power['max']:.2f} W`",
            f"- GPU utilization weighted mean / max: "
            f"`{utilization['weighted_mean']:.2f}%` / `{utilization['max']:.0f}%`",
            f"- VRAM weighted mean / max: `{vram['weighted_mean']:.1f}` / `{vram['max']:.0f} MiB`",
            f"- temperature weighted mean / max: `{temperature['weighted_mean']:.2f}` / "
            f"`{temperature['max']:.0f} °C`",
            f"- sample coverage: `{coverage['coverage_fraction_of_session']:.2%}`; "
            f"leading/internal/trailing gap: `{coverage['leading_gap_seconds']:.2f}` / "
            f"`{coverage['internal_gap_seconds']:.2f}` / "
            f"`{coverage['trailing_gap_seconds']:.2f} s`",
            "- internal gap 包含 aggregate window 之间的正常 collector spacing。",
            f"- first/last window: `{selection['first_window_started_at_utc']}` / "
            f"`{selection['last_window_ended_at_utc']}`",
            f"- immutable raw archive: `{dashboard_capture['bundle_path']}` "
            f"(`{dashboard_capture['row_count']:,}` buckets, "
            f"SHA256 `{dashboard_capture['sha256']}`)",
            "",
            dashboard_gpu["scope"]["caveat"],
        ]
    else:
        lines += [
            dashboard_gpu["note"],
            f"Immutable raw archive `{dashboard_capture['bundle_path']}` is empty "
            f"(SHA256 `{dashboard_capture['sha256']}`).",
        ]
    lines += [
        "",
        "## 图表",
        "",
        "### Loss、NTP 与 MTP (含移动平均)",
        "",
        "![Loss、NTP 与 MTP](charts/training_loss.svg)",
        "",
        "### Nominal / shape-adjusted learning rate",
        "",
        "![Learning rate](charts/learning_rate.svg)",
        "",
        "### 吞吐",
        "",
        "![Throughput](charts/throughput.svg)",
        "",
        "### Gradient norm",
        "",
        "![Gradient norm](charts/gradient_norm.svg)",
        "",
        "### CUDA 显存",
        "",
        "![CUDA memory](charts/gpu_memory.svg)",
        "",
        "### GPU utilization 与 power",
        "",
        "![GPU utilization](charts/gpu_utilization.svg)",
        "",
        "![GPU power](charts/gpu_power.svg)",
        "",
        "### Source token mix",
        "",
        "![Source token mix](charts/source_token_mix.svg)",
    ]
    v4_run = "v4" in str(run.get("run_id", "")).lower()
    if v4_run:
        clipping = analysis["clipping"]
        clipped = sum(
            phase["points"] * phase["gt_configured_fraction"]
            for phase in (
                clipping.get("warmup"),
                clipping.get("primary_stable"),
                clipping.get("primary_decay"),
                clipping.get("cooldown"),
            )
            if phase is not None
        )
        conclusion = [
            "",
            "## v4 smoke 结论与 250M gate",
            "",
            f"- 数值门通过: 全部 required metrics finite, loss 公式精确, "
            f"超过 grad clip `{clipping['configured_threshold']}` 的 step 数为 "
            f"`{int(clipped)}`。",
            f"- 首尾窗口 loss 变化 "
            f"`{windows['metrics']['loss']['percent_delta']:+.2f}%`, NTP 变化 "
            f"`{windows['metrics']['ntp']['percent_delta']:+.2f}%`; 只有 "
            f"`{run['terminal_step']}` 个 optimizer steps, "
            "这是方向性 smoke 证据, 不能替代 frozen held-out validation。",
            f"- 性能门通过: active-wall `{ordinary['aggregate_active_wall_tokens_per_second']:.1f}` "
            f"tok/s, reserved headroom `{performance['memory']['reserved_headroom_gib']:.3f}` GiB。",
            "- 数据 admission 未通过: 本次发生 source wrap; 在扩充并锁定足量 unique "
            "prepared text、修复末 batch 预算策略之前, 不应从该 manifest 启动 250M。",
            "- 纯文本 objective 已是 `1.0*NTP + 0.1*MTP`, 没有 teacher logits KD、"
            "anchor KL 或 hidden alignment。",
        ]
    else:
        conclusion = [
            "",
            "## 后续版本建议 (v4)",
            "",
            "在 optimizer batch 内按目标比例混合 source, 并保留逐步 source-token "
            "账本; Muon pilot 需要同时核验 nominal/adjusted LR、数值恢复与 held-out "
            "validation, 不能只比较 optimizer 名称。",
        ]
    lines += [
        *conclusion,
        "",
        "全部机器可读统计及输入 SHA256 见 JSON; 本目录的 `MANIFEST.json` "
        "记录报告与图表哈希, `COMPLETE` 认证 manifest。",
        "",
    ]
    return "\n".join(lines)


def write_analysis(analysis: Mapping[str, Any], *, output: Path, run_dir: Path) -> dict[str, Any]:
    if analysis.get("schema_version") != SCHEMA_VERSION or analysis.get("kind") != KIND:
        raise AnalysisError(
            "analysis schema is not current; rerun analyze_dense_training against the "
            "authenticated run (legacy reports are never backfilled from live telemetry)"
        )
    json_path, markdown_path = _output_paths(output)
    resolved_run = run_dir.expanduser().resolve(strict=True)
    output_root = json_path.parent
    dashboard_raw_payload, dashboard_captured_rows = _captured_dashboard_payload(analysis)
    chart_payloads = _chart_payloads(
        analysis=analysis,
        run_dir=resolved_run,
        dashboard_captured_rows=dashboard_captured_rows,
    )
    chart_paths = {relative: output_root / relative for relative in chart_payloads}
    dashboard_raw_path = output_root / DASHBOARD_GPU_ARCHIVE_RELATIVE
    manifest_path = output_root / "MANIFEST.json"
    complete_path = output_root / "COMPLETE"
    destinations = [
        json_path,
        markdown_path,
        dashboard_raw_path,
        manifest_path,
        complete_path,
        *chart_paths.values(),
    ]
    if any(_is_within(path, resolved_run) for path in destinations):
        raise AnalysisError("analysis outputs must not be written inside run-dir")
    json_payload = _json_text(analysis)
    markdown_payload = _markdown(analysis, json_path.name)
    _atomic_write_text(dashboard_raw_path, dashboard_raw_payload)
    _atomic_write_text(json_path, json_payload)
    for relative, payload in chart_payloads.items():
        _atomic_write_text(chart_paths[relative], payload)
    _atomic_write_text(markdown_path, markdown_payload)
    payload_paths = [
        json_path,
        markdown_path,
        dashboard_raw_path,
        *chart_paths.values(),
    ]
    files = {}
    for path in sorted(payload_paths):
        identity = _stable_file_identity(path)
        files[str(path.relative_to(output_root))] = {
            "path": str(path.relative_to(output_root)),
            "size": identity["size"],
            "sha256": identity["sha256"],
        }
    dashboard_raw_relative = str(dashboard_raw_path.relative_to(output_root))
    dashboard_capture = analysis["performance"]["dashboard_gpu_last_session"]["raw_capture"]
    if (
        files[dashboard_raw_relative]["size"] != dashboard_capture["size"]
        or files[dashboard_raw_relative]["sha256"] != dashboard_capture["sha256"]
    ):
        raise AnalysisError("archived Dashboard GPU telemetry identity mismatch")
    terminal = analysis["terminal_validation"]
    release_gate = analysis["release_gate"]
    fork = release_gate["fork_checkpoint"]
    source_fork_checkpoint = (
        {
            "path": fork["path"],
            "manifest_sha256": fork["manifest_sha256"],
            "complete_sha256": fork["complete_sha256"],
        }
        if isinstance(fork, Mapping)
        else None
    )
    manifest = {
        "schema_version": 1,
        "kind": "twen_dense_training_analysis_bundle",
        "bundle_producer": _stable_file_identity(Path(__file__).resolve()),
        "run_id": analysis["run"].get("run_id"),
        "source_run_dir": analysis["run"]["run_dir"],
        "source_inputs": {
            name: analysis["inputs"][name]
            for name in (
                "metrics",
                "telemetry",
                "events",
                "resolved_config",
                "rank0_session",
                "latest",
            )
        },
        "source_terminal_checkpoint": {
            "path": terminal["checkpoint"],
            "manifest_sha256": terminal["manifest"]["sha256"],
            "complete_sha256": terminal["complete_marker"]["sha256"],
        },
        "source_fork_checkpoint": source_fork_checkpoint,
        "release_gate": release_gate,
        "files": files,
    }
    _atomic_write_text(manifest_path, _json_text(manifest))
    manifest_identity = _stable_file_identity(manifest_path)
    _atomic_write_text(complete_path, f"{manifest_identity['sha256']}\n")
    return {
        "json": str(json_path),
        "markdown_zh_cn": str(markdown_path),
        "charts": {
            Path(relative).stem: str(path) for relative, path in sorted(chart_paths.items())
        },
        "raw": {
            "dashboard_gpu_last_session": str(dashboard_raw_path),
        },
        "manifest": str(manifest_path),
        "complete": str(complete_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    analysis = analyze_dense_training(args.run_dir)
    outputs = write_analysis(analysis, output=args.output, run_dir=args.run_dir)
    print(_json_text(outputs), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
