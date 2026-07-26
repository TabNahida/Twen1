#!/usr/bin/env python3
"""Compare RTX 5090 full-graph dense benchmarks across microbatch sizes.

The report consumes benchmark JSON emitted by ``benchmark_full_dense_graph.py``
and the sibling ``.power.csv`` files sampled during each candidate case.  It never
imports CUDA, creates an optimizer, or mutates a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GIB = 1024**3
DEFAULT_ORDINARY_B1 = Path(
    "artifacts/benchmarks/rtx5090-base-dense-mtp-sdpa-batch1-ordinary-ac0.json"
)
DEFAULT_ALIGNMENT_B1 = Path(
    "artifacts/benchmarks/rtx5090-base-dense-mtp-sdpa-batch1-alignment-ac24.json"
)
DEFAULT_OUTPUT_PREFIX = Path("artifacts/benchmarks/rtx5090-base-dense-utilization-report")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinary-b1", type=Path, default=DEFAULT_ORDINARY_B1)
    parser.add_argument("--alignment-b1", type=Path, default=DEFAULT_ALIGNMENT_B1)
    parser.add_argument(
        "--ordinary-candidate",
        "--ordinary-batch2",
        dest="ordinary_candidates",
        type=Path,
        action="append",
        required=True,
        help="repeat once per independent ordinary candidate AC case JSON",
    )
    parser.add_argument(
        "--alignment-candidate",
        "--alignment-batch2",
        dest="alignment_candidates",
        type=Path,
        action="append",
        required=True,
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="flat output prefix; writes PREFIX.json/.md and PREFIX-*.svg",
    )
    output_group.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="optional directory bundle used by fixtures or downstream packaging",
    )
    parser.add_argument("--ordinary-b1-ac", type=int, default=0)
    parser.add_argument(
        "--nsys-summary",
        type=Path,
        default=None,
        help="optional sanitized Nsight kernel-acceptance summary JSON",
    )
    parser.add_argument("--ordinary-weight", type=float, default=0.95)
    parser.add_argument("--global-batch-tokens", type=int, default=262144)
    parser.add_argument("--minimum-headroom-gib", type=float, default=3.0)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.ordinary_weight < 1.0:
        raise ValueError("ordinary-weight must be strictly between zero and one")
    if args.global_batch_tokens <= 0:
        raise ValueError("global-batch-tokens must be positive")
    if args.minimum_headroom_gib < 0 or not math.isfinite(args.minimum_headroom_gib):
        raise ValueError("minimum-headroom-gib must be finite and non-negative")
    if len(set(args.ordinary_candidates)) != len(args.ordinary_candidates):
        raise ValueError("ordinary candidate paths must be unique")
    if len(set(args.alignment_candidates)) != len(args.alignment_candidates):
        raise ValueError("alignment candidate paths must be unique")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _bundle_file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"benchmark root must be an object: {resolved}")
    return value


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True


def _validated_nsys_summary(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = _load_json(resolved)
    capture = payload.get("capture", {})
    security = payload.get("security", {})
    acceptance = payload.get("acceptance", {})
    kernel_time = payload.get("kernel_time", {})
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(payload.get("schema_version") == 1, "schema_version must be 1")
    require(
        payload.get("kind") == "twen_sanitized_nsys_kernel_acceptance",
        "kind mismatch",
    )
    require(payload.get("publishable_raw_profile") is False, "raw profile must not publish")
    require(isinstance(capture, Mapping), "capture must be an object")
    require(isinstance(security, Mapping), "security must be an object")
    require(isinstance(acceptance, Mapping), "acceptance must be an object")
    require(isinstance(kernel_time, Mapping), "kernel_time must be an object")
    if isinstance(capture, Mapping):
        require(capture.get("batch_size") == 1, "capture batch_size must be 1")
        require(
            capture.get("activation_checkpoint_layer_count") == 0,
            "capture activation checkpoint layer count must be 0",
        )
        require(capture.get("cuda_profiler_api") is True, "CUDA profiler API gate failed")
        require(capture.get("torch_profiler_enabled") is False, "torch profiler must be off")
        for name in ("benchmark", "nsys_report", "sqlite"):
            identity = capture.get(name)
            require(isinstance(identity, Mapping), f"capture.{name} must be an object")
            if not isinstance(identity, Mapping):
                continue
            raw_path = identity.get("path")
            require(
                isinstance(raw_path, str) and bool(raw_path) and Path(raw_path).name == raw_path,
                f"capture.{name}.path must be a sanitized file name",
            )
            require(
                isinstance(identity.get("size"), int)
                and not isinstance(identity.get("size"), bool)
                and int(identity["size"]) > 0,
                f"capture.{name}.size must be positive",
            )
            require(
                isinstance(identity.get("sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", str(identity.get("sha256"))) is not None,
                f"capture.{name}.sha256 is invalid",
            )
    if isinstance(security, Mapping):
        require(security.get("clean_environment_allowlist") is True, "allowlist gate failed")
        require(
            security.get("captured_environment_values_exported") is False,
            "captured environment values must not be exported",
        )
        require(
            security.get("sensitive_environment_variable_name_matches") == 0,
            "sensitive environment name matches must be zero",
        )
        require(
            security.get("raw_profile_in_report_bundle") is False,
            "raw profile must not enter report bundle",
        )
    if isinstance(acceptance, Mapping):
        require(acceptance.get("ok") is True, "kernel acceptance must pass")
        require(acceptance.get("production_shape") is True, "production shape gate failed")
        require(
            acceptance.get("mtp_attention_implementation") == "sdpa",
            "MTP attention must be sdpa",
        )
        require(
            acceptance.get("gradient_tensors_present") == 72,
            "present gradients must be 72",
        )
        require(acceptance.get("gradient_tensors_missing") == 0, "gradients are missing")
        require(
            acceptance.get("gradient_tensors_nonfinite") == 0,
            "gradients must all be finite",
        )
        require(acceptance.get("optimizer_created") is False, "optimizer must not exist")
        require(acceptance.get("optimizer_steps") == 0, "optimizer steps must be zero")
        require(
            isinstance(acceptance.get("flash_forward_instances"), int)
            and int(acceptance["flash_forward_instances"]) > 0,
            "flash forward instances must be positive",
        )
        require(
            isinstance(acceptance.get("flash_backward_instances"), int)
            and int(acceptance["flash_backward_instances"]) > 0,
            "flash backward instances must be positive",
        )
        require(acceptance.get("eager_softmax_instances") == 0, "eager softmax must be absent")
    if isinstance(kernel_time, Mapping):
        total = kernel_time.get("total_cuda_kernel_time_ms")
        require(
            isinstance(total, (int, float))
            and not isinstance(total, bool)
            and math.isfinite(float(total))
            and float(total) > 0.0,
            "total CUDA kernel time must be positive and finite",
        )
        for name in (
            "dense_gemm",
            "compiled_triton_reductions",
            "fla_recurrent_attention",
            "mtp_sdpa_flash",
        ):
            group = kernel_time.get(name)
            require(isinstance(group, Mapping), f"kernel_time.{name} must be an object")
            if not isinstance(group, Mapping):
                continue
            require(
                isinstance(group.get("instances"), int)
                and not isinstance(group.get("instances"), bool)
                and int(group["instances"]) > 0,
                f"kernel_time.{name}.instances must be positive",
            )
            for field in ("time_ms", "time_percent"):
                value = group.get(field)
                upper = 100.0 if field == "time_percent" else math.inf
                require(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and 0.0 <= float(value) <= upper,
                    f"kernel_time.{name}.{field} is invalid",
                )
    require(_all_finite(payload), "summary contains non-finite values")
    if failures:
        raise ValueError("Nsight summary failed strict acceptance: " + "; ".join(failures))
    return {
        "source": {
            "path": resolved.name,
            "size": resolved.stat().st_size,
            "sha256": _sha256(resolved),
        },
        "summary": payload,
    }


def _mean(summary: Mapping[str, Any], section: str, name: str) -> float | None:
    try:
        value = float(summary[section][name]["mean"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(math.ceil(len(ordered) * fraction) - 1, 0)
    return ordered[index]


def _sample_stats(
    rows: Sequence[Mapping[str, float]], names: Sequence[str] | None = None
) -> dict[str, dict[str, float]]:
    selected_names = sorted({name for row in rows for name in row}) if names is None else names
    stats: dict[str, dict[str, float]] = {}
    for name in selected_names:
        values = [row[name] for row in rows if name in row]
        if values:
            stats[name] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p95": _percentile(values, 0.95),
                "max": max(values),
                "min": min(values),
            }
    return stats


def _power_csv_path(benchmark_path: Path) -> Path:
    return benchmark_path.with_suffix(".power.csv")


def _power_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    rows: list[dict[str, float]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, float] = {}
            for name in (
                "power_draw_w",
                "power_limit_w",
                "utilization_gpu_percent",
                "utilization_memory_percent",
                "clocks_sm_mhz",
                "memory_used_mib",
                "memory_free_mib",
                "temperature_gpu_c",
            ):
                try:
                    value = float(str(raw.get(name, "")).strip())
                except ValueError:
                    continue
                if math.isfinite(value):
                    row[name] = value
            if row:
                rows.append(row)
    if not rows:
        return {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "sample_count": 0,
            "stats": {},
            "active_window": {
                "criterion": "first_to_last_gpu_utilization_at_least_threshold_inclusive",
                "utilization_gpu_threshold_percent": 50.0,
                "sample_range": None,
                "sample_count": 0,
                "stats": {},
            },
        }
    active_indices = [
        index
        for index, row in enumerate(rows)
        if row.get("utilization_gpu_percent", -math.inf) >= 50.0
    ]
    if active_indices:
        first_active = active_indices[0]
        last_active = active_indices[-1]
        active_rows = rows[first_active : last_active + 1]
        active_range: dict[str, int] | None = {
            "first_index": first_active,
            "last_index": last_active,
        }
    else:
        active_rows = []
        active_range = None
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "sample_count": len(rows),
        "stats": _sample_stats(rows),
        "active_window": {
            "criterion": "first_to_last_gpu_utilization_at_least_threshold_inclusive",
            "utilization_gpu_threshold_percent": 50.0,
            "sample_range": active_range,
            "sample_count": len(active_rows),
            "stats": _sample_stats(
                active_rows,
                names=(
                    "power_draw_w",
                    "utilization_gpu_percent",
                    "utilization_memory_percent",
                    "clocks_sm_mhz",
                ),
            ),
        },
    }


def _infer_ac(payload: Mapping[str, Any], path: Path) -> int | None:
    runtime = payload.get("runtime", {})
    if isinstance(runtime, Mapping):
        value = runtime.get("activation_checkpoint_layer_count")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    match = re.search(r"(?:^|[-_])ac(\d+)(?:[-_.]|$)", path.name)
    return int(match.group(1)) if match else None


def _case_health(case: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    health = case.get("health")
    if isinstance(health, Mapping):
        present = health.get("present_gradient_tensor_counts", [])
        present_counts = [int(value) for value in present if isinstance(value, int)]
        return {
            "ok": bool(health.get("ok")),
            "loss_finite": bool(health.get("loss_finite")),
            "gradients_finite": bool(health.get("gradients_finite")),
            "missing_gradient_tensors": int(health.get("maximum_missing_gradient_tensors", 0)),
            "nonfinite_gradient_tensors": int(health.get("maximum_nonfinite_gradient_tensors", 0)),
            "present_gradient_tensor_counts": present_counts,
        }
    samples = case.get("samples", payload.get("samples", []))
    if not isinstance(samples, list) or not samples:
        return {
            "ok": False,
            "loss_finite": False,
            "gradients_finite": False,
            "missing_gradient_tensors": None,
            "nonfinite_gradient_tensors": None,
            "present_gradient_tensor_counts": [],
        }
    gradients = [sample.get("gradients", {}) for sample in samples]
    return {
        "ok": all(bool(sample.get("ok")) for sample in samples),
        "loss_finite": all(bool(sample.get("loss_finite")) for sample in samples),
        "gradients_finite": all(bool(item.get("finite")) for item in gradients),
        "missing_gradient_tensors": max(int(item.get("missing_tensors", 0)) for item in gradients),
        "nonfinite_gradient_tensors": max(
            int(item.get("nonfinite_tensors", 0)) for item in gradients
        ),
        "present_gradient_tensor_counts": sorted(
            {int(item.get("present_tensors", 0)) for item in gradients}
        ),
    }


def _minimum_estimated_headroom(
    case: Mapping[str, Any], payload: Mapping[str, Any], summary: Mapping[str, Any]
) -> int | None:
    candidates: list[int] = []
    memory = summary.get("memory_worst_case", {})
    if isinstance(memory, Mapping):
        value = memory.get("minimum_free_after_bytes")
        if isinstance(value, int):
            candidates.append(value)
    samples = case.get("samples", payload.get("samples", []))
    if isinstance(samples, list):
        for sample in samples:
            if not isinstance(sample, Mapping):
                continue
            sample_memory = sample.get("memory", {})
            if not isinstance(sample_memory, Mapping):
                continue
            value = sample_memory.get("estimated_free_at_peak_reserved_bytes")
            if isinstance(value, int):
                candidates.append(value)
    return min(candidates) if candidates else None


def _benchmark_row(
    *,
    path: Path,
    payload: Mapping[str, Any],
    case: Mapping[str, Any],
    mode: str,
    ac_layers: int | None,
    power: dict[str, Any] | None,
    global_batch_tokens: int,
    candidate: bool,
) -> dict[str, Any]:
    summary = case.get("summary", payload.get("summary", {}))
    if not isinstance(summary, Mapping):
        summary = {}
    batch = payload.get("batch", {})
    batch_size = int(batch.get("batch_size", 0)) if isinstance(batch, Mapping) else 0
    logical_tokens = int(batch.get("logical_tokens", 0)) if isinstance(batch, Mapping) else 0
    health = _case_health(case, payload)
    gpu_tps = _mean(summary, "throughput", "logical_tokens_per_second_gpu")
    wall_tps = _mean(summary, "throughput", "logical_tokens_per_second_wall")
    graph_wall_seconds = _mean(summary, "timing_seconds", "total_wall_seconds")
    graph_gpu_seconds = _mean(summary, "timing_seconds", "total_gpu_seconds")
    timing_breakdown = {
        name: _mean(summary, "timing_seconds", name)
        for name in (
            "anchor_forward_seconds",
            "student_streaming_forward_seconds",
            "teacher_hidden_alignment_seconds",
            "forward_seconds",
            "backward_seconds",
            "total_gpu_seconds",
            "total_wall_seconds",
        )
    }
    wall_gpu_bubble_seconds = (
        max(graph_wall_seconds - graph_gpu_seconds, 0.0)
        if graph_wall_seconds is not None and graph_gpu_seconds is not None
        else None
    )
    wall_gpu_bubble_fraction = (
        wall_gpu_bubble_seconds / graph_wall_seconds
        if wall_gpu_bubble_seconds is not None and graph_wall_seconds
        else None
    )
    transition = 0.0
    if mode == "alignment":
        transition = (
            _mean(summary, "timing_seconds", "teacher_cpu_offload_stage_seconds") or 0.0
        ) + (_mean(summary, "timing_seconds", "teacher_cpu_offload_restore_seconds") or 0.0)
    accumulation_microbatches = (
        global_batch_tokens // logical_tokens
        if logical_tokens > 0 and global_batch_tokens % logical_tokens == 0
        else None
    )
    production_tps = wall_tps or gpu_tps
    if production_tps and mode == "alignment" and transition > 0 and accumulation_microbatches:
        base_seconds = (
            graph_wall_seconds
            if graph_wall_seconds is not None
            else logical_tokens / production_tps
        )
        production_tps = logical_tokens / (base_seconds + transition / accumulation_microbatches)
    memory = summary.get("memory_worst_case", {})
    if not isinstance(memory, Mapping):
        memory = {}
    failure_memory = payload.get("cuda_failure_memory", {})
    if not isinstance(failure_memory, Mapping):
        failure_memory = {}
    peak_allocated = memory.get("peak_allocated_bytes", failure_memory.get("peak_allocated_bytes"))
    peak_reserved = memory.get("peak_reserved_bytes", failure_memory.get("peak_reserved_bytes"))
    headroom = _minimum_estimated_headroom(case, payload, summary)
    if headroom is None:
        raw_free = failure_memory.get("free_bytes")
        headroom = raw_free if isinstance(raw_free, int) else None
    loss_weights = payload.get("loss_weights", {})
    mtp_weight = loss_weights.get("mtp") if isinstance(loss_weights, Mapping) else None
    mtp = payload.get("mtp", {})
    mtp_attention_implementation = (
        mtp.get("attention_implementation") if isinstance(mtp, Mapping) else None
    )
    reserve = payload.get("optimizer_state_reserve", {})
    reserve_gib = reserve.get("requested_gib") if isinstance(reserve, Mapping) else None
    teacher_offload = payload.get("teacher_cpu_offload", {})
    teacher_offload_enabled = (
        bool(teacher_offload.get("enabled")) if isinstance(teacher_offload, Mapping) else False
    )
    error = payload.get("error")
    error_type = error.get("type") if isinstance(error, Mapping) else None
    error_message = error.get("message") if isinstance(error, Mapping) else None
    graph = payload.get("graph", {})
    experimental_execution = payload.get("experimental_execution")
    production_execution_enabled = bool(
        not isinstance(experimental_execution, Mapping)
        or experimental_execution.get("production_enabled") is True
    )
    mode_matches = bool(
        isinstance(graph, Mapping)
        and bool(graph.get("online_hidden_alignment")) == (mode == "alignment")
        and graph.get("parameter_update") is False
    )
    no_optimizer = (
        bool(payload.get("no_optimizer_created"))
        and bool(payload.get("no_optimizer_steps"))
        and int(payload.get("optimizer_step_calls", -1)) == 0
    )
    gradients_72 = health["present_gradient_tensor_counts"] == [72]
    power_stats = power.get("stats", {}) if isinstance(power, Mapping) else {}
    power_valid = bool(
        isinstance(power, Mapping)
        and int(power.get("sample_count", 0)) > 0
        and isinstance(power_stats, Mapping)
        and power_stats.get("power_draw_w")
        and power_stats.get("utilization_gpu_percent")
    )
    accepted = bool(
        payload.get("ok")
        and payload.get("production_acceptance")
        and case.get("ok", True)
        and mode_matches
        and health["ok"]
        and health["loss_finite"]
        and health["gradients_finite"]
        and health["missing_gradient_tensors"] == 0
        and health["nonfinite_gradient_tensors"] == 0
        and gradients_72
        and no_optimizer
    )
    accepted = bool(
        accepted
        and production_execution_enabled
        and mtp_weight is not None
        and math.isclose(float(mtp_weight), 0.1)
        and mtp_attention_implementation == "sdpa"
        and reserve_gib is not None
        and math.isclose(float(reserve_gib), 1.5)
        and teacher_offload_enabled
        and power_valid
    )
    capacity_failure = bool(
        error_type == "OutOfMemoryError"
        or (
            isinstance(error_message, str)
            and "device not ready" in error_message.lower()
            and failure_memory.get("free_bytes") == 0
        )
    )
    status = (
        "ok"
        if accepted
        else (
            "capacity-failure"
            if capacity_failure
            else "experimental-not-production"
            if not production_execution_enabled
            else "failed"
        )
    )
    physical_free_gib = None
    if isinstance(power_stats, Mapping):
        free_stats = power_stats.get("memory_free_mib", {})
        if isinstance(free_stats, Mapping) and free_stats.get("min") is not None:
            physical_free_gib = float(free_stats["min"]) / 1024.0
    active_window = power.get("active_window", {}) if isinstance(power, Mapping) else {}
    active_stats = active_window.get("stats", {}) if isinstance(active_window, Mapping) else {}
    active_power = active_stats.get("power_draw_w", {}) if isinstance(active_stats, Mapping) else {}
    active_power_mean = (
        float(active_power["mean"])
        if isinstance(active_power, Mapping)
        and active_power.get("mean") is not None
        and float(active_power["mean"]) > 0.0
        else None
    )
    estimated_active_tokens_per_joule = (
        float(production_tps) / active_power_mean
        if production_tps is not None and active_power_mean is not None
        else None
    )
    return {
        "label": f"b{batch_size}-{mode}-ac{ac_layers}",
        "mode": mode,
        "batch_size": batch_size,
        "logical_tokens": logical_tokens,
        "activation_checkpoint_layer_count": ac_layers,
        "status": status,
        "accepted": accepted,
        "benchmark_ok": bool(payload.get("ok") and case.get("ok", True)),
        "production_acceptance": bool(payload.get("production_acceptance")),
        "gpu_tokens_per_second": gpu_tps,
        "wall_tokens_per_second": wall_tps,
        "graph_wall_seconds": graph_wall_seconds,
        "graph_gpu_seconds": graph_gpu_seconds,
        "timing_breakdown_seconds": timing_breakdown,
        "wall_gpu_bubble_seconds": wall_gpu_bubble_seconds,
        "wall_gpu_bubble_fraction": wall_gpu_bubble_fraction,
        "production_tokens_per_second": production_tps,
        "teacher_transition_seconds_per_stage": transition,
        "production_accumulation_microbatches": accumulation_microbatches,
        "peak_allocated_gib": float(peak_allocated) / GIB
        if isinstance(peak_allocated, int)
        else None,
        "peak_reserved_gib": float(peak_reserved) / GIB if isinstance(peak_reserved, int) else None,
        "minimum_estimated_headroom_gib": float(headroom) / GIB if headroom is not None else None,
        "minimum_nvml_physical_free_gib": physical_free_gib,
        "health": health,
        "no_optimizer_created_or_stepped": no_optimizer,
        "optimizer_state_reserve_gib": reserve_gib,
        "mtp_loss_weight": mtp_weight,
        "mtp_attention_implementation": mtp_attention_implementation,
        "teacher_cpu_offload": teacher_offload_enabled,
        "experimental_execution": experimental_execution,
        "production_execution_enabled": production_execution_enabled,
        "candidate": candidate,
        "estimated_active_tokens_per_joule": estimated_active_tokens_per_joule,
        "power": power,
        "error": {"type": error_type, "message": error_message} if error_type else None,
        "source": {"path": str(path.resolve()), "sha256": _sha256(path)},
    }


def _single_row(
    path: Path,
    *,
    mode: str,
    global_batch_tokens: int,
    require_power: bool,
    candidate: bool,
) -> dict[str, Any]:
    payload = _load_json(path)
    power = _power_summary(_power_csv_path(path)) if require_power else None
    return _benchmark_row(
        path=path,
        payload=payload,
        case=payload,
        mode=mode,
        ac_layers=_infer_ac(payload, path),
        power=power,
        global_batch_tokens=global_batch_tokens,
        candidate=candidate,
    )


def _b1_ordinary_row(path: Path, *, ac_layers: int, global_batch_tokens: int) -> dict[str, Any]:
    payload = _load_json(path)
    cases = payload.get("cases")
    if not isinstance(cases, list):
        if _infer_ac(payload, path) != ac_layers:
            raise ValueError(f"batch-1 ordinary benchmark does not contain AC{ac_layers}: {path}")
        case = payload
    else:
        matches = [
            item
            for item in cases
            if isinstance(item, Mapping)
            and item.get("activation_checkpoint_layer_count") == ac_layers
        ]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one batch-1 AC{ac_layers} case in {path}")
        case = matches[0]
    return _benchmark_row(
        path=path,
        payload=payload,
        case=case,
        mode="ordinary",
        ac_layers=ac_layers,
        power=_power_summary(_power_csv_path(path)),
        global_batch_tokens=global_batch_tokens,
        candidate=False,
    )


def _weighted(ordinary: Mapping[str, Any], alignment: Mapping[str, Any], weight: float) -> float:
    ordinary_tps = float(ordinary["production_tokens_per_second"])
    alignment_tps = float(alignment["production_tokens_per_second"])
    return 1.0 / (weight / ordinary_tps + (1.0 - weight) / alignment_tps)


def _safe_candidate(row: Mapping[str, Any], minimum_headroom_gib: float) -> bool:
    headrooms = [
        float(value)
        for value in (
            row.get("minimum_estimated_headroom_gib"),
            row.get("minimum_nvml_physical_free_gib"),
        )
        if value is not None
    ]
    return bool(row.get("accepted") and headrooms and min(headrooms) >= minimum_headroom_gib)


def _source_provenance() -> dict[str, Any]:
    from twen.source_identity import twen_source_tree_sha256

    root = Path(__file__).resolve().parents[1]
    status_process = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    commit_process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    status = status_process.stdout
    paths = {
        "mtp": root / "src/twen/modeling/mtp.py",
        "benchmark": root / "scripts/benchmark_full_dense_graph.py",
        "report_generator": Path(__file__).resolve(),
        "semantic_ab": root / "scripts/compare_mtp_attention_backends.py",
    }
    return {
        "git_commit": commit_process.stdout.strip() if commit_process.returncode == 0 else None,
        "git_dirty": bool(status.strip()) or status_process.returncode != 0,
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "git_status_line_count": len(status.splitlines()),
        "twen_source_tree_sha256": twen_source_tree_sha256(),
        "files": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()
        },
    }


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    return f"{float(value):,.{digits}f}"


def _svg_document(body: str, *, width: int = 1200, height: int = 680) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        "<style>text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#172033}"
        ".title{font-size:24px;font-weight:700}.axis{font-size:12px;fill:#526078}"
        ".label{font-size:13px}.value{font-size:12px;font-weight:600}</style>\n"
        f"{body}\n</svg>\n"
    )


def _bar_panel(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    title: str,
    x: int,
    y: int,
    width: int,
    height: int,
    color: str,
) -> str:
    values = [float(row[key]) if row.get(key) is not None else 0.0 for row in rows]
    maximum = max(values, default=1.0) or 1.0
    gap = 8
    bar_height = max((height - 40 - gap * max(len(rows) - 1, 0)) / max(len(rows), 1), 8)
    label_width = 160
    plot_width = width - label_width - 90
    parts = [f'<text class="title" x="{x}" y="{y + 24}">{html.escape(title)}</text>']
    for index, (row, value) in enumerate(zip(rows, values, strict=True)):
        top = y + 42 + index * (bar_height + gap)
        length = plot_width * value / maximum
        label = html.escape(str(row["label"]))
        parts.append(
            f'<text class="label" x="{x}" y="{top + bar_height * 0.72:.1f}">{label}</text>'
        )
        parts.append(
            f'<rect x="{x + label_width}" y="{top:.1f}" width="{length:.1f}" '
            f'height="{bar_height:.1f}" rx="3" fill="{color}" opacity="0.86"/>'
        )
        parts.append(
            f'<text class="value" x="{x + label_width + length + 7:.1f}" '
            f'y="{top + bar_height * 0.72:.1f}">{value:,.1f}</text>'
        )
    return "\n".join(parts)


def _throughput_memory_svg(rows: Sequence[Mapping[str, Any]]) -> str:
    usable = [row for row in rows if row.get("production_tokens_per_second") is not None]
    body = [
        '<rect width="1200" height="680" fill="#f7f9fc"/>',
        _bar_panel(
            usable,
            key="production_tokens_per_second",
            title="Production-adjusted throughput (tok/s)",
            x=35,
            y=18,
            width=1130,
            height=300,
            color="#3478f6",
        ),
        _bar_panel(
            usable,
            key="peak_reserved_gib",
            title="Peak reserved memory (GiB)",
            x=35,
            y=350,
            width=1130,
            height=285,
            color="#13a47a",
        ),
    ]
    return _svg_document("\n".join(body))


def _power_svg(rows: Sequence[Mapping[str, Any]]) -> str:
    power_rows = []
    for row in rows:
        power = row.get("power")
        stats = power.get("stats", {}) if isinstance(power, Mapping) else {}
        draw = stats.get("power_draw_w", {}) if isinstance(stats, Mapping) else {}
        if isinstance(draw, Mapping) and draw:
            power_rows.append(
                {
                    "label": row["label"],
                    "median": draw.get("median"),
                    "p95": draw.get("p95"),
                    "max": draw.get("max"),
                }
            )
    body = ['<rect width="1200" height="680" fill="#f7f9fc"/>']
    for position, (key, title, color) in enumerate(
        (
            ("median", "Median power (W)", "#805ad5"),
            ("p95", "P95 power (W)", "#d97706"),
            ("max", "Maximum sampled power (W)", "#dc3f52"),
        )
    ):
        body.append(
            _bar_panel(
                power_rows,
                key=key,
                title=title,
                x=35 + position * 385,
                y=25,
                width=365,
                height=600,
                color=color,
            )
        )
    return _svg_document("\n".join(body))


def _utilization_svg(rows: Sequence[Mapping[str, Any]]) -> str:
    panels = (
        ("utilization_gpu_percent", "p95", "GPU utilization P95 (%)", "#3478f6"),
        ("utilization_memory_percent", "p95", "Memory utilization P95 (%)", "#13a47a"),
        ("clocks_sm_mhz", "p95", "SM clock P95 (MHz)", "#805ad5"),
    )
    body = ['<rect width="1200" height="680" fill="#f7f9fc"/>']
    for position, (metric, statistic, title, color) in enumerate(panels):
        panel_rows = []
        for row in rows:
            power = row.get("power")
            stats = power.get("stats", {}) if isinstance(power, Mapping) else {}
            values = stats.get(metric, {}) if isinstance(stats, Mapping) else {}
            if isinstance(values, Mapping) and values.get(statistic) is not None:
                panel_rows.append({"label": row["label"], "value": values[statistic]})
        body.append(
            _bar_panel(
                panel_rows,
                key="value",
                title=title,
                x=35 + position * 385,
                y=25,
                width=365,
                height=600,
                color=color,
            )
        )
    return _svg_document("\n".join(body))


def _markdown(
    report: Mapping[str, Any],
    *,
    throughput_image: str = "throughput-memory.svg",
    power_image: str = "power.svg",
    utilization_image: str = "utilization.svg",
) -> str:
    rows = report["rows"]
    lines = [
        "# RTX 5090 Base Dense microbatch 利用率验收",
        "",
        f"生成时间: `{report['created_at']}`。本报告只运行完整 forward/backward graph, "
        "基准程序没有构造 optimizer, 也没有执行参数更新。",
        "",
        f"![吞吐与显存]({throughput_image})",
        "",
        f"![功耗采样]({power_image})",
        "",
        f"![GPU 利用率采样]({utilization_image})",
        "",
        "## 单 case 结果",
        "",
        "| case | 状态 | GPU-event/wall tok/s | production tok/s | step wall ms | forward/backward ms | wall-GPU bubble ms (%) | peak A/R GiB | PyTorch free GiB | NVML physical free GiB | gradients |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        gradient_counts = row["health"]["present_gradient_tensor_counts"]
        timing = row["timing_breakdown_seconds"]
        bubble = row["wall_gpu_bubble_seconds"]
        bubble_fraction = row["wall_gpu_bubble_fraction"]
        lines.append(
            "| {label} | {status} | {gpu}/{wall} | {production} | {step} | {forward}/{backward} | "
            "{bubble} ({bubble_percent}) | {allocated}/{reserved} | {free} | {nvml_free} | {gradients} |".format(
                label=row["label"],
                status=row["status"],
                gpu=_fmt(row["gpu_tokens_per_second"]),
                wall=_fmt(row["wall_tokens_per_second"]),
                production=_fmt(row["production_tokens_per_second"]),
                step=_fmt(
                    timing["total_wall_seconds"] * 1000
                    if timing["total_wall_seconds"] is not None
                    else None,
                    2,
                ),
                forward=_fmt(
                    timing["forward_seconds"] * 1000
                    if timing["forward_seconds"] is not None
                    else None,
                    2,
                ),
                backward=_fmt(
                    timing["backward_seconds"] * 1000
                    if timing["backward_seconds"] is not None
                    else None,
                    2,
                ),
                bubble=_fmt(bubble * 1000 if bubble is not None else None, 3),
                bubble_percent=(
                    f"{bubble_fraction * 100:.3f}%" if bubble_fraction is not None else "n/a"
                ),
                allocated=_fmt(row["peak_allocated_gib"], 2),
                reserved=_fmt(row["peak_reserved_gib"], 2),
                free=_fmt(row["minimum_estimated_headroom_gib"], 2),
                nvml_free=_fmt(row["minimum_nvml_physical_free_gib"], 2),
                gradients=gradient_counts or "n/a",
            )
        )
    lines.extend(
        [
            "",
            "## 250 ms 功耗与利用率采样",
            "",
            "| case | samples whole/active (active range) | power mean whole/active W | GPU util mean whole/active % | SM clock mean whole/active MHz | estimated active tok/J |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        power = row.get("power")
        if not isinstance(power, Mapping):
            continue
        stats = power.get("stats", {})
        if not isinstance(stats, Mapping):
            continue
        active_window = power.get("active_window", {})
        active_stats = active_window.get("stats", {}) if isinstance(active_window, Mapping) else {}

        def mean_pair(
            name: str,
            stats: Mapping[str, Any] = stats,
            active_stats: Mapping[str, Any] = active_stats,
        ) -> str:
            whole = stats.get(name, {})
            active = active_stats.get(name, {}) if isinstance(active_stats, Mapping) else {}
            whole_mean = whole.get("mean") if isinstance(whole, Mapping) else None
            active_mean = active.get("mean") if isinstance(active, Mapping) else None
            return f"{_fmt(whole_mean)}/{_fmt(active_mean)}"

        active_range = (
            active_window.get("sample_range") if isinstance(active_window, Mapping) else None
        )
        range_text = (
            f"{active_range['first_index']}..{active_range['last_index']}"
            if isinstance(active_range, Mapping)
            else "n/a"
        )
        active_count = (
            active_window.get("sample_count") if isinstance(active_window, Mapping) else 0
        )

        lines.append(
            f"| {row['label']} | {power['sample_count']}/{active_count} ({range_text}) | "
            f"{mean_pair('power_draw_w')} | {mean_pair('utilization_gpu_percent')} | "
            f"{mean_pair('clocks_sm_mhz')} | "
            f"{_fmt(row['estimated_active_tokens_per_joule'], 3)} |"
        )
    kernel_profile = report.get("kernel_profile")
    if isinstance(kernel_profile, Mapping):
        summary = kernel_profile["summary"]
        kernel_time = summary["kernel_time"]
        acceptance = summary["acceptance"]
        security = summary["security"]
        lines.extend(
            [
                "",
                "## Nsight Systems kernel 验收",
                "",
                "| kernel group | CUDA kernel time % | instances |",
                "|---|---:|---:|",
                f"| dense GEMM | {_fmt(kernel_time['dense_gemm']['time_percent'], 2)} | "
                f"{kernel_time['dense_gemm']['instances']} |",
                "| compiled Triton reductions | "
                f"{_fmt(kernel_time['compiled_triton_reductions']['time_percent'], 2)} | "
                f"{kernel_time['compiled_triton_reductions']['instances']} |",
                f"| FLA recurrent attention | "
                f"{_fmt(kernel_time['fla_recurrent_attention']['time_percent'], 2)} | "
                f"{kernel_time['fla_recurrent_attention']['instances']} |",
                f"| MTP SDPA flash | {_fmt(kernel_time['mtp_sdpa_flash']['time_percent'], 2)} | "
                f"{kernel_time['mtp_sdpa_flash']['instances']} |",
                "",
                "安全验收: `passed`; sensitive environment name matches="
                f"`{security['sensitive_environment_variable_name_matches']}`, raw profile in bundle="
                f"`{str(security['raw_profile_in_report_bundle']).lower()}`, flash forward/backward="
                f"`{acceptance['flash_forward_instances']}/{acceptance['flash_backward_instances']}`, "
                f"eager softmax=`{acceptance['eager_softmax_instances']}`。",
            ]
        )
    recommendation = report["recommendation"]
    lines.extend(
        [
            "",
            "## 95% ordinary / 5% alignment 加权比较",
            "",
            "| batch | ordinary case | alignment case | production-adjusted tok/s | 相对 batch-1 | 验收 |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for item in report["batch_comparisons"]:
        relative = item["relative_to_batch1_percent"]
        relative_text = (
            "baseline"
            if item["batch_size"] == 1
            else (f"{relative:+.2f}%" if relative is not None else "n/a")
        )
        lines.append(
            f"| batch-{item['batch_size']} | {item['ordinary_case'] or 'n/a'} | "
            f"{item['alignment_case'] or 'n/a'} | {_fmt(item['tokens_per_second'])} | "
            f"{relative_text} | {str(item['accepted']).lower()} |"
        )
    lines.extend(
        [
            "",
            "这里使用 harmonic mixture: 每类 batch 的 token 数除以各自耗时后再按 95/5 合并。"
            "alignment 的 CPU→GPU stage/restore 按固定 global batch 内的 accumulation microbatch 数摊销, "
            "不是把一次传输完整计入每个 microbatch。",
            "active window 取 GPU utilization 首次与末次达到 50% 的样本之间的闭区间, "
            "中间的低利用率与 bubble 样本全部保留。estimated active tok/J 是 production-adjusted "
            "wall tok/s 除以 active-window 平均板卡功耗的估算值, 不是逐 token 积分能耗。",
            "",
            "## 结论",
            "",
            "- 推荐 microbatch: `{}`。".format(
                f"batch-{recommendation['batch_size']}"
                if recommendation["batch_size"] is not None
                else "n/a"
            ),
            f"- 推荐 ordinary 档位: `{recommendation['ordinary_case']}`; 选择门槛为保守可用显存不低于 "
            f"{report['minimum_headroom_gib']:.2f} GiB, 并在满足门槛的 case 中取吞吐最高者。",
            f"- alignment 档位: `{recommendation['alignment_case']}`。",
            "- production config: `micro_batch_size={micro_batch_size}`, "
            "`activation_checkpointing={activation_checkpointing}`, "
            "`activation_checkpointing_on_alignment_only={alignment_only}`, "
            "`activation_checkpoint_layer_count={layer_count}`。".format(
                micro_batch_size=(recommendation["production_config"] or {}).get(
                    "micro_batch_size", "n/a"
                ),
                activation_checkpointing=str(
                    (recommendation["production_config"] or {}).get(
                        "activation_checkpointing", "n/a"
                    )
                ).lower(),
                alignment_only=str(
                    (recommendation["production_config"] or {}).get(
                        "activation_checkpointing_on_alignment_only", "n/a"
                    )
                ).lower(),
                layer_count=json.dumps(
                    (recommendation["production_config"] or {}).get(
                        "activation_checkpoint_layer_count"
                    )
                ),
            ),
            f"- 是否存在通过完整验收的安全档位: `{str(report['accepted']).lower()}`。每个通过的 case 都要求 "
            "MTP=0.1、MTP attention=sdpa、teacher offload、1.5 GiB moment 等价 reserve、72/72 finite gradients、"
            "finite losses、零 optimizer 创建/step, 以及非空功耗采样。",
            "- 功耗和 tok/J 只是诊断量, 不参与选档; 低于 600 W 不等同于 GPU 空闲。"
            "最终选择仍由吞吐、完整验收契约与显存安全余量决定。",
            "",
            "## 可复核性",
            "",
            "JSON 报告记录了每个 benchmark/power 文件的绝对路径与 SHA256。任何 OOM case 都保留"
            "错误类型、CUDA 峰值和同期功耗 CSV, 不会覆盖前一 case 的成功结果。",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    kernel_profile = (
        _validated_nsys_summary(args.nsys_summary) if args.nsys_summary is not None else None
    )
    b1_ordinary = _b1_ordinary_row(
        args.ordinary_b1,
        ac_layers=args.ordinary_b1_ac,
        global_batch_tokens=args.global_batch_tokens,
    )
    b1_alignment = _single_row(
        args.alignment_b1,
        mode="alignment",
        global_batch_tokens=args.global_batch_tokens,
        require_power=True,
        candidate=False,
    )
    candidate_ordinary = [
        _single_row(
            path,
            mode="ordinary",
            global_batch_tokens=args.global_batch_tokens,
            require_power=True,
            candidate=True,
        )
        for path in args.ordinary_candidates
    ]
    candidate_ordinary.sort(
        key=lambda row: (
            row["batch_size"],
            -(
                row["activation_checkpoint_layer_count"]
                if row["activation_checkpoint_layer_count"] is not None
                else -1
            ),
            str(row["source"]["path"]),
        )
    )
    candidate_alignment = [
        _single_row(
            path,
            mode="alignment",
            global_batch_tokens=args.global_batch_tokens,
            require_power=True,
            candidate=True,
        )
        for path in args.alignment_candidates
    ]
    candidate_alignment.sort(key=lambda row: (row["batch_size"], row["label"]))
    ordinary_rows = [b1_ordinary, *candidate_ordinary]
    alignment_rows = [b1_alignment, *candidate_alignment]
    batch_comparisons: list[dict[str, Any]] = []
    for batch_size in sorted({row["batch_size"] for row in (*ordinary_rows, *alignment_rows)}):
        safe_ordinary = [
            row
            for row in ordinary_rows
            if row["batch_size"] == batch_size and _safe_candidate(row, args.minimum_headroom_gib)
        ]
        safe_alignment = [
            row
            for row in alignment_rows
            if row["batch_size"] == batch_size and _safe_candidate(row, args.minimum_headroom_gib)
        ]
        selected_ordinary = (
            max(
                safe_ordinary,
                key=lambda row: float(row["production_tokens_per_second"] or 0.0),
            )
            if safe_ordinary
            else None
        )
        selected_alignment = (
            max(
                safe_alignment,
                key=lambda row: float(row["production_tokens_per_second"] or 0.0),
            )
            if safe_alignment
            else None
        )
        weighted = (
            _weighted(selected_ordinary, selected_alignment, args.ordinary_weight)
            if selected_ordinary is not None and selected_alignment is not None
            else None
        )
        batch_comparisons.append(
            {
                "batch_size": batch_size,
                "accepted": weighted is not None,
                "ordinary_case": selected_ordinary["label"] if selected_ordinary else None,
                "alignment_case": selected_alignment["label"] if selected_alignment else None,
                "tokens_per_second": weighted,
                "relative_to_batch1_percent": None,
            }
        )
    batch1 = next((item for item in batch_comparisons if item["batch_size"] == 1), None)
    batch1_weighted = (
        float(batch1["tokens_per_second"])
        if batch1 is not None and batch1["tokens_per_second"] is not None
        else None
    )
    if batch1_weighted is not None:
        for item in batch_comparisons:
            weighted = item["tokens_per_second"]
            item["relative_to_batch1_percent"] = (
                (float(weighted) / batch1_weighted - 1.0) * 100.0 if weighted is not None else None
            )
    accepted_batches = [item for item in batch_comparisons if item["accepted"]]
    recommended_batch = (
        max(accepted_batches, key=lambda item: float(item["tokens_per_second"]))
        if accepted_batches
        else None
    )
    batch2 = next((item for item in batch_comparisons if item["batch_size"] == 2), None)
    rows = [b1_ordinary, b1_alignment, *candidate_ordinary, *candidate_alignment]
    if recommended_batch is not None:
        selected_ordinary = next(
            row for row in rows if row["label"] == recommended_batch["ordinary_case"]
        )
        selected_alignment = next(
            row for row in rows if row["label"] == recommended_batch["alignment_case"]
        )
        ordinary_ac = selected_ordinary["activation_checkpoint_layer_count"]
        alignment_ac = selected_alignment["activation_checkpoint_layer_count"]
        production_config = {
            "micro_batch_size": recommended_batch["batch_size"],
            "activation_checkpointing": bool((ordinary_ac or 0) or (alignment_ac or 0)),
            "activation_checkpointing_on_alignment_only": alignment_ac == 24,
            "activation_checkpoint_layer_count": ordinary_ac if ordinary_ac else None,
        }
    else:
        production_config = None
    recommendation = {
        "batch_size": recommended_batch["batch_size"] if recommended_batch else None,
        "ordinary_case": recommended_batch["ordinary_case"] if recommended_batch else None,
        "alignment_case": recommended_batch["alignment_case"] if recommended_batch else None,
        "tokens_per_second": recommended_batch["tokens_per_second"] if recommended_batch else None,
        "production_config": production_config,
    }
    return {
        "schema_version": 2,
        "kind": "rtx5090_base_dense_utilization_report",
        "created_at": datetime.now(UTC).isoformat(),
        "read_only_report_generation": True,
        "no_optimizer_created_by_report": True,
        "ordinary_weight": args.ordinary_weight,
        "alignment_weight": 1.0 - args.ordinary_weight,
        "global_batch_tokens": args.global_batch_tokens,
        "minimum_headroom_gib": args.minimum_headroom_gib,
        "source_provenance": _source_provenance(),
        "kernel_profile": kernel_profile,
        "legacy_mtp_benchmarks": {
            "status": "INVALID_SUPERSEDED",
            "reason": (
                "_attn_implementation=None skipped causal-mask construction and then "
                "fell back to eager attention, causing bidirectional leakage"
            ),
            "v1_training_affected": False,
            "v1_reason": "base-dense-v1 used losses.mtp=0.0",
        },
        "rows": rows,
        "recommendation": recommendation,
        "batch_comparisons": batch_comparisons,
        "weighted_comparison": {
            "method": "harmonic_token_time_mixture_with_alignment_transition_amortization",
            "batch1_tokens_per_second": batch1_weighted,
            "batch2_tokens_per_second": batch2["tokens_per_second"] if batch2 else None,
            "batch2_relative_change_percent": (
                batch2["relative_to_batch1_percent"] if batch2 else None
            ),
        },
        "accepted": recommended_batch is not None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(args)
    from twen.utils import atomic_write_text

    if args.output_dir is not None:
        output = args.output_dir.expanduser().resolve()
        paths = {
            "report_json": output / "report.json",
            "report_markdown": output / "report.md",
            "throughput_memory_svg": output / "throughput-memory.svg",
            "power_svg": output / "power.svg",
            "utilization_svg": output / "utilization.svg",
            "manifest": output / "MANIFEST.json",
            "complete": output / "COMPLETE",
        }
        reported_output = {"output_dir": str(output)}
    else:
        prefix = args.output_prefix.expanduser().resolve()
        paths = {
            "report_json": prefix.with_name(prefix.name + ".json"),
            "report_markdown": prefix.with_name(prefix.name + ".md"),
            "throughput_memory_svg": prefix.with_name(prefix.name + "-throughput-memory.svg"),
            "power_svg": prefix.with_name(prefix.name + "-power.svg"),
            "utilization_svg": prefix.with_name(prefix.name + "-utilization.svg"),
            "manifest": prefix.with_name(prefix.name + ".MANIFEST.json"),
            "complete": prefix.with_name(prefix.name + ".COMPLETE"),
        }
        reported_output = {"output_prefix": str(prefix)}
    atomic_write_text(
        paths["report_json"],
        _json_text(report),
    )
    atomic_write_text(
        paths["report_markdown"],
        _markdown(
            report,
            throughput_image=paths["throughput_memory_svg"].name,
            power_image=paths["power_svg"].name,
            utilization_image=paths["utilization_svg"].name,
        ),
    )
    atomic_write_text(paths["throughput_memory_svg"], _throughput_memory_svg(report["rows"]))
    atomic_write_text(paths["power_svg"], _power_svg(report["rows"]))
    atomic_write_text(paths["utilization_svg"], _utilization_svg(report["rows"]))

    payload_keys = (
        "report_json",
        "report_markdown",
        "throughput_memory_svg",
        "power_svg",
        "utilization_svg",
    )
    source_provenance_sha256 = _canonical_json_sha256(report["source_provenance"])
    manifest = {
        "schema_version": 1,
        "kind": "twen_rtx5090_base_dense_utilization_report_bundle",
        "accepted": report["accepted"],
        "recommendation": report["recommendation"],
        "source_provenance_sha256": source_provenance_sha256,
        "files": {name: _bundle_file_identity(paths[name]) for name in payload_keys},
    }
    atomic_write_text(paths["manifest"], _json_text(manifest))
    complete = {
        "schema_version": 1,
        "kind": "twen_rtx5090_base_dense_utilization_report_complete",
        "manifest": paths["manifest"].name,
        "manifest_sha256": _sha256(paths["manifest"]),
        "report": {
            "path": paths["report_json"].name,
            "sha256": _sha256(paths["report_json"]),
        },
        "accepted": report["accepted"],
        "recommendation": report["recommendation"],
        "source_provenance_sha256": source_provenance_sha256,
    }
    atomic_write_text(paths["complete"], _json_text(complete))
    print(json.dumps({"ok": report["accepted"], **reported_output}, sort_keys=True))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
