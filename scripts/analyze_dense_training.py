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
import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCHEMA_VERSION = 1
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
            "output directory (writes analysis.json and REPORT.zh-CN.md), "
            "or an explicit .json path"
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


def _jsonl_rows(payload: bytes, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise AnalysisError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    if not rows:
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
            _affine_permutation(position, count, self.seed, epoch)
            for position in range(count)
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


def _validate_series(
    metrics: Sequence[Mapping[str, Any]],
    telemetry: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(metrics) != len(telemetry):
        raise AnalysisError("metrics and telemetry row counts differ")
    previous_tokens = 0
    formula_errors: list[float] = []
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
        for key in ("loss", "ntp", "teacher_kd", "anchor_kl", "grad_norm", "lr"):
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
        "formula_errors": formula_errors,
    }


def _loss_formula_validation(
    metrics: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    losses = config.get("losses")
    if not isinstance(losses, Mapping):
        raise AnalysisError("resolved config losses must be an object")
    weights: dict[str, float] = {}
    for name in LOSS_COMPONENTS:
        weight = losses.get(name, 0.0)
        if isinstance(weight, (int, float)) and not isinstance(weight, bool):
            weights[name] = _finite(weight, label=f"losses.{name}")
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
        "rolling_sample_sd": (
            float(rolling.std(ddof=1)) if len(rolling) > 1 else 0.0
        ),
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
        "residual_sample_sd": (
            float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0
        ),
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
    available = [key for key in METRIC_COMPONENTS if key in metrics[0]]
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
                f"window_{window}": {
                    key: _rolling_summary(rows, key, window) for key in available
                }
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


def _load_manifest(
    path: Path,
    *,
    identities: dict[str, Any],
    identity_key: str,
) -> dict[str, Any]:
    payload, identity = _read_stable_bytes(path)
    identities[identity_key] = identity
    return _json_object(payload, path)


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
    primary_path = _resolve_input_path(
        primary_value, run_dir=run_dir, project_root=project_root
    )
    primary_manifest = _load_manifest(
        primary_path, identities=identities, identity_key="primary_manifest"
    )
    configured_primary_sha = data.get("manifest_sha256")
    if (
        isinstance(configured_primary_sha, str)
        and identities["primary_manifest"]["sha256"] != configured_primary_sha
    ):
        raise AnalysisError("primary manifest SHA does not match resolved config")
    primary_ids, primary_lengths, primary_sources = _manifest_layout(primary_manifest)
    cooldown_value = data.get("quality_cooldown_manifest_path")
    cooldown_start = data.get("quality_cooldown_start_tokens")
    cooldown_cursor: _PhaseCursor | None = None
    cooldown_sources: dict[str, str] = {}
    cooldown_path: Path | None = None
    if cooldown_value is not None or cooldown_start is not None:
        if not isinstance(cooldown_value, str):
            raise AnalysisError("quality cooldown manifest path is incomplete")
        cooldown_start_tokens = _integer(
            cooldown_start, label="data.quality_cooldown_start_tokens"
        )
        cooldown_path = _resolve_input_path(
            cooldown_value, run_dir=run_dir, project_root=project_root
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
        cooldown_ids, cooldown_lengths, cooldown_sources = _manifest_layout(
            cooldown_manifest
        )
        cooldown_cursor = _PhaseCursor(
            cooldown_ids,
            cooldown_lengths,
            seed=_integer(data.get("shuffle_seed"), label="data.shuffle_seed"),
        )
    else:
        cooldown_start_tokens = None
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
                "basename(source_path), strip .jsonl/.parquet/.json and trailing "
                "[-_]NNNNNN"
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
        "_cursor": replay,
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
        [
            [fractions[index].get(source, 0.0) for source in sources]
            for index in selected_indices
        ],
        dtype=np.float64,
    )
    token_axis = _values(analysis_phase, "tokens") / 100_000_000.0
    centered = token_axis - token_axis.mean()
    design = np.column_stack([source_matrix, centered])
    coefficient_names = [f"source:{source}" for source in sources] + [
        "tokens_per_100m"
    ]
    available = [key for key in METRIC_COMPONENTS if key in analysis_phase[0]]
    regressions = {
        key: _regression(
            design,
            _values(analysis_phase, key),
            coefficient_names=coefficient_names,
            hac_lag=50,
        )
        for key in available
    }
    source_only_loss = _regression(
        source_matrix,
        _values(analysis_phase, "loss"),
        coefficient_names=[f"source:{source}" for source in sources],
        hac_lag=50,
    )
    interactions = np.column_stack([source_matrix, source_matrix * centered[:, None]])
    interaction_names = [f"source:{source}" for source in sources] + [
        f"source_slope_per_100m:{source}" for source in sources
    ]
    source_specific = {
        key: _regression(
            interactions,
            _values(analysis_phase, key),
            coefficient_names=interaction_names,
            hac_lag=50,
        )
        for key in ("loss", "ntp", "teacher_kd", "mtp", "anchor_kl")
        if key in analysis_phase[0]
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
        for key, regression in source_specific.items():
            item[f"{key}_at_mid_token"] = regression["coefficients"][
                f"source:{source}"
            ]["estimate"]
            item[f"{key}_slope_per_100m"] = regression["coefficients"][
                f"source_slope_per_100m:{source}"
            ]["estimate"]
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
        "design": (
            "X=[all source fractions, centered(tokens/1e8)], no intercept; "
            "OLS with Newey-West/Bartlett lag 50"
        ),
        "source_only_loss": source_only_loss,
        "common_slopes": regressions,
        "source_interactions": source_specific,
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
        [
            [fractions[index].get(source, 0.0) for source in sources]
            for index in cooldown_indices
        ],
        dtype=np.float64,
    )
    token_axis = (
        _values(cooldown, "tokens") - float(cooldown_start_tokens)
    ) / 10_000_000.0
    lr_axis = _values(cooldown, "lr") / 0.0001
    available = [key for key in METRIC_COMPONENTS if key in cooldown[0]]

    def conditioned(axis: np.ndarray, axis_name: str) -> dict[str, Any]:
        design = np.column_stack([source_matrix, axis - axis.mean()])
        names = [f"source:{source}" for source in sources] + [axis_name]
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
        "source_adjusted_tokens_per_10m": conditioned(
            token_axis, "tokens_per_10m"
        ),
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
        _finite(row.get("compute_step_seconds"), label="compute_step_seconds")
        for row in rows
    )
    wall = sum(
        _finite(row.get("wall_clock_step_seconds"), label="wall_clock_step_seconds")
        for row in rows
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
    cooldown_ordinary = [
        row
        for row in ordinary
        if row.get("data_phase") == "cooldown"
    ]
    primary_ordinary = [
        row
        for row in ordinary
        if row.get("data_phase", "primary") == "primary"
    ]
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
            if any(token in key.lower() for token in ("power", "watt", "utilization", "temperature"))
        }
    )
    output = {
        "segments": segments,
        "alignment": {
            "points": len(alignment),
            "fraction_steps": len(alignment) / len(telemetry),
            "step_interval": (
                sorted(
                    {
                        right["step"] - left["step"]
                        for left, right in pairwise(alignment)
                    }
                )
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
        ideal = sum(
            _finite(row.get("compute_step_seconds"), label="compute_step_seconds")
            for row in ordinary
        ) + len(alignment) * median_ordinary
        output["alignment"]["estimated_compute_time_penalty_percent"] = (
            actual / ideal - 1.0
        ) * 100
    return output


def _nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    if not values or not 0 < percentile <= 1:
        raise AnalysisError("nearest-rank percentile inputs are invalid")
    ordered = sorted(values)
    index = math.ceil(percentile * len(ordered)) - 1
    return ordered[index]


def _dashboard_gpu_telemetry(
    *,
    project_root: Path,
    rank_session: Mapping[str, Any],
    identities: dict[str, Any],
) -> dict[str, Any]:
    dashboard_dir = project_root / ".twen" / "dashboard"
    candidates = (
        ("dashboard_gpu_telemetry_rotated", dashboard_dir / "gpu-telemetry.jsonl.1"),
        ("dashboard_gpu_telemetry_current", dashboard_dir / "gpu-telemetry.jsonl"),
    )
    file_rows: list[tuple[str, list[dict[str, Any]]]] = []
    for identity_key, path in candidates:
        if not path.exists():
            continue
        payload, identity = _read_stable_bytes(path)
        identities[identity_key] = identity
        file_rows.append((identity_key, _jsonl_rows(payload, path)))
    scope = {
        "kind": "last_rank0_session_only",
        "session_id": rank_session.get("session_id"),
        "started_at_utc": rank_session.get("started_at_utc"),
        "ended_at_utc": rank_session.get("ended_at_utc"),
        "caveat": (
            "Dashboard telemetry is filtered to the last rank0 session and does not "
            "represent earlier resumed sessions or the entire run."
        ),
    }
    if not file_rows:
        return {
            "available": False,
            "scope": scope,
            "note": "Dashboard GPU telemetry files are absent.",
        }
    session_start = _parse_timestamp(
        rank_session.get("started_at_utc"),
        label="rank0-session.started_at_utc",
    )
    session_end = _parse_timestamp(
        rank_session.get("ended_at_utc"),
        label="rank0-session.ended_at_utc",
    )
    if session_start.utcoffset() is None or session_end.utcoffset() is None:
        raise AnalysisError("rank0 session timestamps must be timezone-aware")
    if session_end <= session_start:
        raise AnalysisError("rank0 session end must be after its start")
    selected: list[tuple[datetime, datetime, str, dict[str, Any]]] = []
    rows_by_input: Counter[str] = Counter(
        {identity_key: 0 for identity_key, _rows in file_rows}
    )
    for identity_key, rows in file_rows:
        for index, row in enumerate(rows, 1):
            if row.get("kind") != "twen_gpu_telemetry_aggregate":
                raise AnalysisError(
                    f"{identity_key}[{index}] has unsupported GPU telemetry kind"
                )
            started = _parse_timestamp(
                row.get("window_started_at_utc"),
                label=f"{identity_key}[{index}].window_started_at_utc",
            )
            ended = _parse_timestamp(
                row.get("window_ended_at_utc"),
                label=f"{identity_key}[{index}].window_ended_at_utc",
            )
            if started.utcoffset() is None or ended.utcoffset() is None:
                raise AnalysisError("GPU telemetry timestamps must be timezone-aware")
            if ended < started:
                raise AnalysisError("GPU telemetry window end must not precede its start")
            if ended == started:
                samples = _integer(
                    row.get("sample_count"),
                    label=f"{identity_key}[{index}].sample_count",
                )
                if samples != 1:
                    raise AnalysisError(
                        "zero-duration GPU telemetry windows must contain one sample"
                    )
            if started >= session_start and ended <= session_end:
                selected.append((started, ended, identity_key, row))
                rows_by_input[identity_key] += 1
    if not selected:
        return {
            "available": False,
            "scope": scope,
            "selected_bucket_count": 0,
            "rows_by_input": dict(rows_by_input),
            "note": "No complete Dashboard GPU telemetry buckets fall within the last session.",
        }
    selected.sort(key=lambda item: (item[0], item[1]))
    sample_count = 0
    available_count = 0
    unavailable_count = 0
    sampled_seconds = 0.0
    fields = (
        "power_draw_w",
        "power_limit_w",
        "gpu_utilization_percent",
        "memory_utilization_percent",
        "vram_used_mib",
        "temperature_c",
    )
    weighted_sums = {field: 0.0 for field in fields}
    bucket_means: dict[str, list[float]] = {field: [] for field in fields}
    minima: dict[str, list[float]] = {field: [] for field in fields}
    maxima: dict[str, list[float]] = {field: [] for field in fields}
    raw_intervals: set[int] = set()
    for _started, _ended, _identity_key, row in selected:
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
        for field in fields:
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
    summaries = {
        field: {
            "weighted_mean": weighted_sums[field] / available_count,
            "bucket_mean_nearest_rank_p95": _nearest_rank_percentile(
                bucket_means[field],
                0.95,
            ),
            "min": min(minima[field]),
            "max": max(maxima[field]),
        }
        for field in fields
    }
    return {
        "available": True,
        "scope": scope,
        "selection": {
            "condition": (
                "window_started_at_utc >= session_start and "
                "window_ended_at_utc <= session_end"
            ),
            "selected_bucket_count": len(selected),
            "rows_by_input": dict(sorted(rows_by_input.items())),
            "first_window_started_at_utc": first_start.isoformat(),
            "last_window_ended_at_utc": last_end.isoformat(),
        },
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
        "aggregation": {
            "weighted_mean": (
                "sum(bucket field mean * available_sample_count) / "
                "sum(available_sample_count)"
            ),
            "bucket_mean_nearest_rank_p95": (
                "nearest-rank ceil(0.95 * available_bucket_count) - 1"
            ),
        },
        "fields": summaries,
    }


def _lr_dose(config: Mapping[str, Any], metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    optimizer = config.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise AnalysisError("resolved config optimizer must be an object")
    maximum_tokens = _integer(optimizer.get("max_tokens"), label="optimizer.max_tokens")
    warmup_tokens = _integer(
        optimizer.get("warmup_tokens"), label="optimizer.warmup_tokens"
    )
    minimum_ratio = _finite(
        optimizer.get("min_lr_ratio", 0.1), label="optimizer.min_lr_ratio"
    )
    peak = max(_finite(row.get("lr"), label="lr") for row in metrics)
    actual_equivalent = sum(
        _finite(row.get("lr"), label="lr")
        / peak
        * _integer(row.get("tokens_this_step"), label="tokens_this_step")
        for row in metrics
    )
    cosine_area = warmup_tokens / 2 + (maximum_tokens - warmup_tokens) * (
        1 + minimum_ratio
    ) / 2
    schedule = optimizer.get("lr_schedule", "cosine")
    if schedule == "warmup-stable-decay":
        decay_tokens = _integer(
            optimizer.get("decay_tokens"), label="optimizer.decay_tokens"
        )
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
    proposed_rates = {
        name: float(value) * 0.9 for name, value in peak_rates.items()
    }
    return {
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
            "observed_average_ratio": actual_equivalent / sum(
                _integer(row.get("tokens_this_step"), label="tokens_this_step")
                for row in metrics
            ),
        },
        "full_cosine": {
            "same_peak_equivalent_tokens": cosine_area,
            "same_peak_relative_to_configured": cosine_area / configured_area,
            "peak_minus_10pct_relative_to_configured": 0.9
            * cosine_area
            / configured_area,
            "peak_minus_25pct_relative_to_configured": 0.75
            * cosine_area
            / configured_area,
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
    integrity = _validate_series(metrics, telemetry)
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
    replay_cursor = replay.pop("_cursor")
    grad_clip_norm = _configured_grad_clip(config)
    warmup_tokens = _integer(
        config.get("optimizer", {}).get("warmup_tokens"),
        label="optimizer.warmup_tokens",
    )
    phases, analysis_phase = _phase_statistics(
        metrics,
        warmup_tokens=warmup_tokens,
    )
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
    metadata_cursor = checkpoint_metadata.get("data_cursor", {})
    metadata_extra = (
        metadata_cursor.get("extra", {}) if isinstance(metadata_cursor, Mapping) else {}
    )
    if isinstance(metadata_cursor, Mapping):
        if metadata_cursor.get("global_sample_index") != replay_cursor.next_sample:
            raise AnalysisError("replayed final sample count does not match checkpoint cursor")
        if metadata_cursor.get("global_token_index") != replay_cursor.committed_tokens:
            raise AnalysisError("replayed final tokens do not match checkpoint cursor")
    integrity["source_replay_matches_checkpoint_cursor"] = True
    integrity["checkpoint_cursor_kind"] = (
        metadata_extra.get("kind") if isinstance(metadata_extra, Mapping) else None
    )
    replay_fractions = replay.pop("_fractions")
    del replay_fractions
    completion_event = terminal["train_complete"]
    cooldown_start_tokens = config.get("data", {}).get("quality_cooldown_start_tokens")
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
        "source_replay": replay,
        "source_adjusted": source_fixed_effects,
        "cooldown_lr_separation": cooldown,
        "clipping": clipping,
        "performance": performance,
        "lr_dose": lr_dose,
        "interpretation": {
            "raw_loss_plateau": (
                "Raw optimizer-step loss is strongly confounded by source-pure batches "
                "and source-order autocorrelation."
            ),
            "cooldown": (
                "The data bundle and LR change at the same "
                f"{cooldown_start_tokens:,}-token boundary; tail loss cannot identify "
                "a causal LR benefit."
                if isinstance(cooldown_start_tokens, int)
                else "No quality cooldown boundary is configured."
            ),
            "next_version_priority": (
                "Mix sources within optimizer batches and retain source-conditioned "
                "metrics before making a large LR reduction."
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


def _markdown(analysis: Mapping[str, Any], json_name: str) -> str:
    run = analysis["run"]
    terminal = analysis["terminal_validation"]
    phases = analysis["phases"]
    source_adjusted = analysis["source_adjusted"]
    cooldown = analysis["cooldown_lr_separation"]
    performance = analysis["performance"]
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
    loss_regression = source_adjusted["common_slopes"]["loss"]
    loss_slope = loss_regression["coefficients"]["tokens_per_100m"]
    source_only_r2 = source_adjusted["source_only_loss"]["r_squared"]
    lines += [
        "",
        "## Source-adjusted 学习趋势",
        "",
        "正式回归窗口固定为 warmup 后的全部 primary batch, 包含 stable 与 cosine decay, "
        "排除 warmup 和 quality cooldown; 因此不同 decay 长度的 run 使用同一数据阶段口径。",
        "",
        f"source composition 单独解释 raw loss 方差的 `{_fmt_percent(source_only_r2)}`。"
        f"控制 source 后, loss slope 为 `{loss_slope['estimate']:.5f}/100M tokens`, "
        f"HAC95 CI `[{loss_slope['confidence_95'][0]:.5f}, "
        f"{loss_slope['confidence_95'][1]:.5f}]`。",
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
            f"- alignment compute: "
            f"`{alignment['aggregate_compute_tokens_per_second']:.1f} tok/s`"
        )
    memory = performance["memory"]
    lines += [
        f"- peak allocated/reserved: `{memory['peak_allocated_gib']:.3f}` / "
        f"`{memory['peak_reserved_gib']:.3f} GiB`",
        f"- reserved headroom: `{_fmt(memory['reserved_headroom_gib'], 3)} GiB`",
    ]
    dashboard_gpu = performance["dashboard_gpu_last_session"]
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
            f"- VRAM weighted mean / max: `{vram['weighted_mean']:.1f}` / "
            f"`{vram['max']:.0f} MiB`",
            f"- temperature weighted mean / max: `{temperature['weighted_mean']:.2f}` / "
            f"`{temperature['max']:.0f} °C`",
            f"- sample coverage: `{coverage['coverage_fraction_of_session']:.2%}`; "
            f"leading/internal/trailing gap: `{coverage['leading_gap_seconds']:.2f}` / "
            f"`{coverage['internal_gap_seconds']:.2f}` / "
            f"`{coverage['trailing_gap_seconds']:.2f} s`",
            "- internal gap 包含 aggregate window 之间的正常 collector spacing。",
            f"- first/last window: `{selection['first_window_started_at_utc']}` / "
            f"`{selection['last_window_ended_at_utc']}`",
            "",
            dashboard_gpu["scope"]["caveat"],
        ]
    else:
        lines.append(dashboard_gpu["note"])
    lines += [
        "",
        "## 后续版本建议 (v4)",
        "",
        "v3 已经使用 `1.8e-4` Adapter 峰值和 250M-token cosine decay; 因此不能再把 "
        "“降峰值 10% / 扩展到 250M decay”写成下一轮建议。v4 先在 optimizer batch 内"
        "按目标比例混合 source, 并记录 source-conditioned NTP/MTP/grad norm。",
        "",
        "按当前 v4 设计, 二维 Adapter 使用 Muon (`match_rms_adamw`) 的 nominal peak "
        "`1.0e-4`, 一维 scale 使用 AdamW `3.0e-4`, 5M-token warmup 后做全程 cosine, "
        "min LR ratio `0.1`。Muon 的矩阵正交更新与 AdamW 不同, 这是一轮 optimizer + "
        "objective 的联合 pilot, 不能把 nominal LR 当作 v3 AdamW LR 的单变量 A/B。",
        "",
        "v4 纯文本路径关闭 teacher logits KD、anchor KL 和 hidden alignment, 保留冻结的 "
        "9B donor FFN 与 Qwen3.5 原生 MTP; 先用 20M token 验证数值、吞吐、显存、Muon "
        "step 开销和精确恢复, 再进入更长预算。",
        "",
        "本报告不包含图表; 全部统计及输入 SHA256 见 JSON。",
        "",
    ]
    return "\n".join(lines)


def write_analysis(
    analysis: Mapping[str, Any], *, output: Path, run_dir: Path
) -> dict[str, str]:
    json_path, markdown_path = _output_paths(output)
    resolved_run = run_dir.expanduser().resolve(strict=True)
    if _is_within(json_path, resolved_run) or _is_within(markdown_path, resolved_run):
        raise AnalysisError("analysis outputs must not be written inside run-dir")
    json_payload = _json_text(analysis)
    markdown_payload = _markdown(analysis, json_path.name)
    _atomic_write_text(json_path, json_payload)
    _atomic_write_text(markdown_path, markdown_payload)
    return {"json": str(json_path), "markdown_zh_cn": str(markdown_path)}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    analysis = analyze_dense_training(args.run_dir)
    outputs = write_analysis(analysis, output=args.output, run_dir=args.run_dir)
    print(_json_text(outputs), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
