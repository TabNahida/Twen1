#!/usr/bin/env python3
# ruff: noqa: RUF001
"""Build locked RTX 5090 expanded-production sweep and canonical reports.

This program is intentionally CPU/read-only with respect to benchmark inputs.
It never imports CUDA, creates an optimizer, or runs a model.  Every input byte
is pinned below before any recommendation or chart is produced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import shutil
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GIB = 1024**3
GLOBAL_BATCH_TOKENS = 262_144
MINIMUM_HEADROOM_GIB = 3.0
ORDINARY_WEIGHT = 0.95
ALIGNMENT_WEIGHT = 0.05
HIGH_UTILIZATION_THRESHOLD_PERCENT = 90.0

EXPECTED_BENCHMARK_SHA256 = "93c9610cbced74111f554f0306d1ef4ebecc537f767e5a194b53d4bca821abaa"
EXPECTED_MTP_SHA256 = "74ff303a7526120ebbc32306f5ed5fab9ad8e65d672deb83c3729564b312564c"
EXPECTED_MTP_GIT_COMMIT = "fc3f2cd2f8a5c24abdab4a55cb6629c619008d27"
EXPECTED_MTP_GIT_BLOB_SHA1 = "6e5ba4fd397946ce224252948bed7b692db97ecf"
EXPECTED_MTP_SOURCE_PATH = "src/twen/modeling/mtp.py"


@dataclass(frozen=True, slots=True)
class CaseSpec:
    key: str
    label: str
    mode: str
    relative_json: str
    json_sha256: str
    power_sha256: str
    batch_size: int
    outer_checkpoint_layers: int
    inner_checkpoint_layers: int
    repeats: int
    expected_ok: bool
    expected_safety_gate: bool


CASE_SPECS = (
    CaseSpec(
        key="b1_ordinary_final_n10",
        label="b1-ordinary-ac0",
        mode="ordinary",
        relative_json=(
            "artifacts/benchmarks/"
            "rtx5090-base-v2-expanded-b1-ordinary-outer0-inner0-mtp01-final-n10.json"
        ),
        json_sha256="1f74b233d349fa55fa2b853cd25b93099214b2c2e8518dfa61ae4bc2a97b797e",
        power_sha256="1178b5837f34d9eed6cdaf60c2b94963001ba1f8264f0804b0dcdc5afb190f2d",
        batch_size=1,
        outer_checkpoint_layers=0,
        inner_checkpoint_layers=0,
        repeats=10,
        expected_ok=True,
        expected_safety_gate=True,
    ),
    CaseSpec(
        key="b1_alignment_final_n10",
        label="b1-alignment-ac8",
        mode="alignment",
        relative_json=(
            "artifacts/benchmarks/"
            "rtx5090-base-v2-expanded-b1-alignment-outer8-inner16-mtp01-final-n10.json"
        ),
        json_sha256="af4e31f0cc1ce35e62323b454f69e96f0b875565ea71d0415c7e05392de838bc",
        power_sha256="d2228b97baceeb567855a400edc2c1571ab095fb33b1420940c59600c3d7ea19",
        batch_size=1,
        outer_checkpoint_layers=8,
        inner_checkpoint_layers=16,
        repeats=10,
        expected_ok=True,
        expected_safety_gate=True,
    ),
    CaseSpec(
        key="b2_ordinary_outer0_inner24",
        label="b2-ordinary-ac0",
        mode="ordinary",
        relative_json=(
            "artifacts/benchmarks/rtx5090-base-v2-expanded-b2-ordinary-outer0-inner24-mtp01.json"
        ),
        json_sha256="d5926caf450cd540d3c932a80395a373e8c34ed82cea475e5590d640912131fb",
        power_sha256="a66de75d5e44ecf64bfde269f2c120fc6a5e6dc69291f6cf6bd4e52771684dda",
        batch_size=2,
        outer_checkpoint_layers=0,
        inner_checkpoint_layers=24,
        repeats=3,
        expected_ok=True,
        expected_safety_gate=True,
    ),
    CaseSpec(
        key="b2_ordinary_outer8_inner8",
        label="b2-ordinary-ac8",
        mode="ordinary",
        relative_json=(
            "artifacts/benchmarks/rtx5090-base-v2-expanded-b2-ordinary-outer8-inner8-mtp01.json"
        ),
        json_sha256="1ee40997385989abaf7a271dfa0d26e9d2d4f63e12d6818cbb7eca925bd8f97f",
        power_sha256="9e6ea0e742578931a5697ffb19574f93e4ba862482a32c3ffeb9f195d102cfd7",
        batch_size=2,
        outer_checkpoint_layers=8,
        inner_checkpoint_layers=8,
        repeats=3,
        expected_ok=True,
        expected_safety_gate=False,
    ),
    CaseSpec(
        key="b2_ordinary_outer4_inner4_failure",
        label="b2-ordinary-ac4",
        mode="ordinary",
        relative_json=(
            "artifacts/benchmarks/rtx5090-base-v2-expanded-b2-ordinary-outer4-inner4-mtp01.json"
        ),
        json_sha256="27b2c73b2176fa0b380b942e4e5db875bf15d9253d7bdd92d4685c0549815149",
        power_sha256="61dc4d0cdbf8d64341873c21e4473d91ae5335c2fd32d4834539e89375a2d568",
        batch_size=2,
        outer_checkpoint_layers=4,
        inner_checkpoint_layers=4,
        repeats=3,
        expected_ok=False,
        expected_safety_gate=False,
    ),
    CaseSpec(
        key="b1_alignment_outer12_inner12",
        label="b1-alignment-ac12",
        mode="alignment",
        relative_json=(
            "artifacts/benchmarks/rtx5090-base-v2-expanded-b1-alignment-outer12-inner12-mtp01.json"
        ),
        json_sha256="2454d2e8d514153e2c1b1016f6aee6b3f277ebe667e8d5e18c529e0b18155b9c",
        power_sha256="b243a0824dcc0151ad1f46419dde8a15230b7bba9c8d4c08773824c4ca964fd0",
        batch_size=1,
        outer_checkpoint_layers=12,
        inner_checkpoint_layers=12,
        repeats=3,
        expected_ok=True,
        expected_safety_gate=True,
    ),
    CaseSpec(
        key="b1_alignment_outer0_inner24_unsafe",
        label="b1-alignment-ac0",
        mode="alignment",
        relative_json=(
            "artifacts/benchmarks/rtx5090-base-v2-expanded-b1-alignment-outer0-inner24-mtp01.json"
        ),
        json_sha256="be05761581b86c3906c536eba9c064b7d088b9fb0ddbbc6d936e05d79f0e476b",
        power_sha256="958be68700e69d2e11916062c5eab9c2cf08b69a8efda69235fbfa8fddec6896",
        batch_size=1,
        outer_checkpoint_layers=0,
        inner_checkpoint_layers=24,
        repeats=3,
        expected_ok=True,
        expected_safety_gate=False,
    ),
)

NUMERICAL_EVIDENCE = {
    "expanded": {
        "relative_report": (
            "artifacts/audits/differentiable-fold-numerical-admission/"
            "EXPANDED_SELECTIVE_CHECKPOINT_FULL_GRAPH_REPORT.md"
        ),
        "report_sha256": "3c204116e4ad436130ddfb2cc3ca88e641f5e9c15586a37b97c120411500a2f1",
        "relative_complete": (
            "artifacts/audits/differentiable-fold-numerical-admission/"
            "EXPANDED_SELECTIVE_CHECKPOINT_FULL_GRAPH_COMPLETE"
        ),
        "complete_sha256": "c2df9bfec497bac00115fae9fdd11da68c2a5f6243be973cb3e81d2642d8301e",
        "status": "pass",
        "required_text": "Numerical admission: **PASS**.",
    },
    "folded": {
        "relative_report": (
            "artifacts/audits/differentiable-fold-numerical-admission/"
            "FULL_GRAPH_V1_REAL_KD_ACCUMULATION_REPORT.md"
        ),
        "report_sha256": "c4d1fdbd21462e8b4970d57ca6282dad47d5035cd5020cf35624aa047c0f146b",
        "relative_complete": (
            "artifacts/audits/differentiable-fold-numerical-admission/"
            "FULL_GRAPH_V1_REAL_KD_ACCUMULATION_COMPLETE"
        ),
        "complete_sha256": "6c1de22887e2e3ad4437b0b04a1073138686567486986f4d336f37e2cb66640e",
        "status": "fail_experimental_only",
        "required_text": "Strict accumulated differentiable-fold admission: **FAIL**.",
    },
}

POWER_COLUMNS = (
    "timestamp",
    "power_draw_w",
    "power_limit_w",
    "utilization_gpu_percent",
    "utilization_memory_percent",
    "clocks_sm_mhz",
    "memory_used_mib",
    "memory_free_mib",
    "temperature_gpu_c",
)
NUMERIC_POWER_COLUMNS = POWER_COLUMNS[1:]


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("artifacts/reports/rtx5090-expanded-production-sweep"),
    )
    parser.add_argument(
        "--canonical-prefix",
        type=Path,
        default=Path("artifacts/benchmarks/rtx5090-base-dense-utilization-report"),
    )
    parser.add_argument(
        "--replace-existing-report",
        action="store_true",
        help="atomically replace an older authenticated sweep bundle",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    display = (
        resolved.relative_to(relative_to.resolve()).as_posix()
        if relative_to is not None
        else str(resolved)
    )
    return {"path": display, "size": resolved.stat().st_size, "sha256": _sha256(resolved)}


def _require_sha(path: Path, expected: str, label: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"missing locked {label}: {resolved}")
    actual = _sha256(resolved)
    if actual != expected:
        raise ValueError(
            f"locked {label} SHA256 changed: expected {expected}, got {actual}: {resolved}"
        )
    return _identity(resolved)


def _require_historical_git_source(
    root: Path,
    *,
    commit: str,
    relative_path: str,
    expected_blob_sha1: str,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    """Authenticate source bytes used by a historical benchmark.

    A later bug fix must not make an old performance run appear to have used
    the new implementation.  The benchmark report therefore reads the exact
    source blob from its pinned commit instead of silently rebinding the
    evidence to the current working-tree file.
    """

    root = root.resolve()
    object_name = f"{commit}:{relative_path}"
    try:
        resolved_commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", f"{commit}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        blob_sha1 = subprocess.run(
            ["git", "-C", str(root), "rev-parse", object_name],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        payload = subprocess.run(
            ["git", "-C", str(root), "show", object_name],
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"cannot read locked historical {label}: {object_name}") from error

    if resolved_commit != commit:
        raise ValueError(
            f"locked historical {label} commit changed: expected {commit}, "
            f"got {resolved_commit}"
        )
    if blob_sha1 != expected_blob_sha1:
        raise ValueError(
            f"locked historical {label} Git blob changed: expected "
            f"{expected_blob_sha1}, got {blob_sha1}"
        )
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"locked historical {label} SHA256 changed: expected "
            f"{expected_sha256}, got {actual_sha256}"
        )
    return {
        "path": f"git:{commit}:{relative_path}",
        "size": len(payload),
        "sha256": actual_sha256,
        "git_commit": commit,
        "git_blob_sha1": blob_sha1,
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON input: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    if not _all_finite(value):
        raise ValueError(f"JSON input contains non-finite values: {path}")
    return value


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True


def _canonical_sha256(value: Any) -> str:
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
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def _mean(summary: Mapping[str, Any], section: str, name: str) -> float | None:
    try:
        value = float(summary[section][name]["mean"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(math.ceil(len(ordered) * fraction) - 1, 0)]


def _stats(
    rows: Sequence[Mapping[str, float]],
    names: Sequence[str] = NUMERIC_POWER_COLUMNS,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name in names:
        values = [row[name] for row in rows if name in row]
        if values:
            result[name] = {
                "min": min(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p95": _percentile(values, 0.95),
                "max": max(values),
            }
    return result


def _power_summary(path: Path) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != POWER_COLUMNS:
            raise ValueError(f"power CSV columns changed: {path}")
        for index, raw in enumerate(reader, start=2):
            row: dict[str, float] = {}
            for name in NUMERIC_POWER_COLUMNS:
                try:
                    value = float(str(raw[name]).strip())
                except (KeyError, ValueError) as error:
                    raise ValueError(f"invalid power CSV value at {path}:{index}:{name}") from error
                if not math.isfinite(value):
                    raise ValueError(f"non-finite power CSV value at {path}:{index}:{name}")
                row[name] = value
            rows.append(row)
    if not rows:
        raise ValueError(f"power CSV contains no samples: {path}")
    active_indices = [
        index for index, row in enumerate(rows) if row["utilization_gpu_percent"] >= 50.0
    ]
    active_rows = rows[active_indices[0] : active_indices[-1] + 1] if active_indices else []
    high_rows = [
        row for row in rows if row["utilization_gpu_percent"] >= HIGH_UTILIZATION_THRESHOLD_PERCENT
    ]
    if not high_rows:
        raise ValueError(f"power CSV has no >=90% GPU-utilization samples: {path}")
    return {
        **_identity(path),
        "sample_count": len(rows),
        "stats": _stats(rows),
        "active_window": {
            "criterion": "first_to_last_gpu_utilization_at_least_50_percent_inclusive",
            "sample_range": (
                {"first_index": active_indices[0], "last_index": active_indices[-1]}
                if active_indices
                else None
            ),
            "sample_count": len(active_rows),
            "stats": _stats(active_rows),
        },
        "high_utilization": {
            "criterion": "individual_samples_gpu_utilization_at_least_threshold",
            "utilization_gpu_threshold_percent": HIGH_UTILIZATION_THRESHOLD_PERCENT,
            "sample_count": len(high_rows),
            "sample_fraction": len(high_rows) / len(rows),
            "stats": _stats(high_rows),
        },
    }


def _validate_numerical(payload: Mapping[str, Any], label: str) -> None:
    execution = payload.get("experimental_execution")
    if not isinstance(execution, Mapping):
        raise ValueError(f"{label} has no execution admission contract")
    admission = execution.get("numerical_admission")
    if not isinstance(admission, Mapping):
        raise ValueError(f"{label} has no numerical admission contract")
    expanded = admission.get("expanded_selective_checkpoint")
    folded = admission.get("differentiable_folded")
    if not (
        execution.get("mode") == "expanded"
        and execution.get("production_enabled") is True
        and execution.get("selected_mode_numerical_status") == "admitted_production_reference"
        and admission.get("production_reference_mode") == "expanded"
        and isinstance(expanded, Mapping)
        and expanded.get("status") == "pass"
        and isinstance(folded, Mapping)
        and folded.get("status") == "fail_experimental_only"
    ):
        raise ValueError(f"{label} violates expanded PASS / folded FAIL admission")


def _health(payload: Mapping[str, Any]) -> dict[str, Any]:
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        return {
            "ok": False,
            "loss_finite": False,
            "gradients_finite": False,
            "missing_gradient_tensors": None,
            "nonfinite_gradient_tensors": None,
            "present_gradient_tensor_counts": [],
        }
    gradients = [sample.get("gradients") for sample in samples if isinstance(sample, Mapping)]
    valid_gradients = [item for item in gradients if isinstance(item, Mapping)]
    return {
        "ok": len(valid_gradients) == len(samples)
        and all(sample.get("ok") is True for sample in samples),
        "loss_finite": all(sample.get("loss_finite") is True for sample in samples),
        "gradients_finite": len(valid_gradients) == len(samples)
        and all(item.get("finite") is True for item in valid_gradients),
        "missing_gradient_tensors": max(
            (int(item.get("missing_tensors", -1)) for item in valid_gradients),
            default=None,
        ),
        "nonfinite_gradient_tensors": max(
            (int(item.get("nonfinite_tensors", -1)) for item in valid_gradients),
            default=None,
        ),
        "present_gradient_tensor_counts": sorted(
            {int(item.get("present_tensors", -1)) for item in valid_gradients}
        ),
    }


def _timing(summary: Mapping[str, Any]) -> dict[str, float | None]:
    names = (
        "anchor_forward_seconds",
        "student_streaming_forward_seconds",
        "teacher_hidden_alignment_seconds",
        "teacher_cpu_offload_stage_seconds",
        "teacher_cpu_offload_restore_seconds",
        "forward_seconds",
        "backward_seconds",
        "total_gpu_seconds",
        "total_wall_seconds",
        "total_wall_seconds_including_teacher_transitions",
    )
    return {name: _mean(summary, "timing_seconds", name) for name in names}


def _row(root: Path, spec: CaseSpec) -> dict[str, Any]:
    path = root / spec.relative_json
    power_path = path.with_suffix(".power.csv")
    source = _require_sha(path, spec.json_sha256, f"{spec.key} benchmark")
    _require_sha(power_path, spec.power_sha256, f"{spec.key} power CSV")
    payload = _load_json(path)
    power = _power_summary(power_path)
    _validate_numerical(payload, spec.label)

    batch = payload.get("batch")
    runtime = payload.get("runtime")
    graph = payload.get("graph")
    if not isinstance(batch, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError(f"{spec.label} has no batch/runtime contract")
    inner = runtime.get(
        "dense_transfer_checkpoint_layer_count_effective",
        runtime.get("dense_transfer_checkpoint_layer_count_requested"),
    )
    if (
        batch.get("batch_size") != spec.batch_size
        or batch.get("sequence_length") != 4096
        or batch.get("logical_tokens") != spec.batch_size * 4096
        or runtime.get("activation_checkpoint_layer_count") != spec.outer_checkpoint_layers
        or inner != spec.inner_checkpoint_layers
        or runtime.get("repeats") != spec.repeats
        or runtime.get("dense_transfer_execution") != "expanded"
        or payload.get("ok") is not spec.expected_ok
    ):
        raise ValueError(f"{spec.label} shape/checkpoint/result contract changed")
    if payload.get("no_optimizer_created") is not True or not (
        payload.get("no_optimizer_steps") is True and payload.get("optimizer_step_calls") == 0
    ):
        raise ValueError(f"{spec.label} optimizer contract changed")
    reserve = payload.get("optimizer_state_reserve")
    if not isinstance(reserve, Mapping) or reserve.get("requested_gib") != 1.5:
        raise ValueError(f"{spec.label} optimizer reserve contract changed")

    summary = payload.get("summary")
    summary = summary if isinstance(summary, Mapping) else {}
    health = _health(payload)
    semantic_ok = bool(
        spec.expected_ok
        and payload.get("production_acceptance") is True
        and isinstance(graph, Mapping)
        and graph.get("online_hidden_alignment") == (spec.mode == "alignment")
        and graph.get("parameter_update") is False
        and graph.get("student_layers_active") == 24
        and health["ok"]
        and health["loss_finite"]
        and health["gradients_finite"]
        and health["missing_gradient_tensors"] == 0
        and health["nonfinite_gradient_tensors"] == 0
        and health["present_gradient_tensor_counts"] == [72]
    )
    mtp = payload.get("mtp")
    loss_weights = payload.get("loss_weights")
    mtp_ok = bool(
        semantic_ok
        and graph.get("native_mtp_forward") is True
        and graph.get("native_mtp_vocab_loss") is True
        and isinstance(mtp, Mapping)
        and mtp.get("enabled") is True
        and mtp.get("frozen") is True
        and mtp.get("parameter_update") is False
        and mtp.get("attention_implementation") == "sdpa"
        and isinstance(loss_weights, Mapping)
        and loss_weights.get("mtp") == 0.1
    )
    if spec.expected_ok and not mtp_ok:
        raise ValueError(f"{spec.label} native MTP/72-gradient health contract changed")

    throughput = summary.get("throughput")
    throughput = throughput if isinstance(throughput, Mapping) else {}
    gpu_tps = _mean(summary, "throughput", "logical_tokens_per_second_gpu")
    wall_tps = _mean(summary, "throughput", "logical_tokens_per_second_wall")
    production_tps = _mean(
        summary,
        "throughput",
        "logical_tokens_per_second_wall_including_teacher_transitions",
    )
    memory = summary.get("memory_worst_case")
    memory = memory if isinstance(memory, Mapping) else {}
    failure_memory = payload.get("cuda_failure_memory")
    failure_memory = failure_memory if isinstance(failure_memory, Mapping) else {}
    estimated_free = memory.get("minimum_free_after_bytes", failure_memory.get("free_bytes"))
    peak_allocated = memory.get("peak_allocated_bytes", failure_memory.get("peak_allocated_bytes"))
    peak_reserved = memory.get("peak_reserved_bytes", failure_memory.get("peak_reserved_bytes"))
    estimated_headroom = float(estimated_free) / GIB if isinstance(estimated_free, int) else None
    nvml_free = power["stats"]["memory_free_mib"]["min"] / 1024.0
    safety_gate = bool(
        semantic_ok
        and estimated_headroom is not None
        and estimated_headroom >= MINIMUM_HEADROOM_GIB
        and nvml_free >= MINIMUM_HEADROOM_GIB
    )
    if safety_gate is not spec.expected_safety_gate:
        raise ValueError(f"{spec.label} locked 3 GiB safety classification changed")
    high_power_mean = power["high_utilization"]["stats"]["power_draw_w"]["mean"]
    error = payload.get("error")
    error_summary = (
        {"type": error.get("type"), "message": error.get("message")}
        if isinstance(error, Mapping)
        else None
    )
    classification = (
        "selected-production"
        if spec.key in {"b1_ordinary_final_n10", "b1_alignment_final_n10"}
        else "safe-alternative"
        if safety_gate
        else "capacity-failure"
        if not spec.expected_ok and failure_memory.get("free_bytes") == 0
        else "unsafe-below-3gib-physical-free-gate"
    )
    return {
        "label": spec.label,
        "key": spec.key,
        "mode": spec.mode,
        "batch_size": spec.batch_size,
        "logical_tokens": spec.batch_size * 4096,
        "activation_checkpoint_layer_count": spec.outer_checkpoint_layers,
        "dense_transfer_checkpoint_layer_count": spec.inner_checkpoint_layers,
        "dense_transfer_execution": "expanded",
        "status": (
            "ok"
            if safety_gate
            else "capacity-failure"
            if not spec.expected_ok and failure_memory.get("free_bytes") == 0
            else "unsafe"
        ),
        "classification": classification,
        "accepted": safety_gate,
        "safety_gate_passed": safety_gate,
        "benchmark_ok": spec.expected_ok,
        "production_acceptance": payload.get("production_acceptance") is True,
        "gpu_tokens_per_second": gpu_tps,
        "wall_tokens_per_second": wall_tps,
        "wall_tokens_per_second_including_teacher_transitions": production_tps,
        "production_tokens_per_second": production_tps,
        "production_accumulation_microbatches": (GLOBAL_BATCH_TOKENS // (spec.batch_size * 4096)),
        "timing_breakdown_seconds": _timing(summary),
        "peak_allocated_gib": (
            float(peak_allocated) / GIB if isinstance(peak_allocated, int) else None
        ),
        "peak_reserved_gib": (
            float(peak_reserved) / GIB if isinstance(peak_reserved, int) else None
        ),
        "minimum_estimated_headroom_gib": estimated_headroom,
        "minimum_nvml_physical_free_gib": nvml_free,
        "conservative_free_gib": (
            min(estimated_headroom, nvml_free) if estimated_headroom is not None else nvml_free
        ),
        "health": health,
        "native_mtp": {
            "enabled": bool(isinstance(mtp, Mapping) and mtp.get("enabled") is True),
            "frozen": bool(isinstance(mtp, Mapping) and mtp.get("frozen") is True),
            "loss_weight": (loss_weights.get("mtp") if isinstance(loss_weights, Mapping) else None),
            "attention_implementation": (
                mtp.get("attention_implementation") if isinstance(mtp, Mapping) else None
            ),
        },
        "mtp_loss_weight": (loss_weights.get("mtp") if isinstance(loss_weights, Mapping) else None),
        "mtp_attention_implementation": (
            mtp.get("attention_implementation") if isinstance(mtp, Mapping) else None
        ),
        "teacher_cpu_offload": bool(runtime.get("teacher_cpu_offload")),
        "optimizer_state_reserve_gib": reserve.get("requested_gib"),
        "no_optimizer_created_or_stepped": True,
        "power": power,
        "high_utilization_tokens_per_joule": (
            production_tps / high_power_mean if production_tps is not None else None
        ),
        "error": error_summary,
        "source": source,
    }


def _numerical_evidence(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"production_reference_mode": "expanded"}
    for name, raw in NUMERICAL_EVIDENCE.items():
        report_path = root / str(raw["relative_report"])
        complete_path = root / str(raw["relative_complete"])
        report_identity = _require_sha(
            report_path, str(raw["report_sha256"]), f"{name} numerical report"
        )
        complete_identity = _require_sha(
            complete_path, str(raw["complete_sha256"]), f"{name} numerical COMPLETE"
        )
        if str(raw["required_text"]) not in report_path.read_text(encoding="utf-8"):
            raise ValueError(f"{name} numerical admission text changed")
        complete = _load_json(complete_path)
        if complete.get("status") != "complete":
            raise ValueError(f"{name} numerical evidence is incomplete")
        result[name] = {
            "status": raw["status"],
            "report": report_identity,
            "complete": complete_identity,
        }
    return result


def _harmonic_mixture(ordinary_tps: float, alignment_tps: float) -> float:
    return 1.0 / (ORDINARY_WEIGHT / ordinary_tps + ALIGNMENT_WEIGHT / alignment_tps)


def build_report(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    benchmark_identity = _require_sha(
        root / "scripts/benchmark_full_dense_graph.py",
        EXPECTED_BENCHMARK_SHA256,
        "benchmark source",
    )
    mtp_identity = _require_historical_git_source(
        root,
        commit=EXPECTED_MTP_GIT_COMMIT,
        relative_path=EXPECTED_MTP_SOURCE_PATH,
        expected_blob_sha1=EXPECTED_MTP_GIT_BLOB_SHA1,
        expected_sha256=EXPECTED_MTP_SHA256,
        label="native MTP source",
    )
    rows = [_row(root, spec) for spec in CASE_SPECS]
    by_key = {row["key"]: row for row in rows}
    ordinary = by_key["b1_ordinary_final_n10"]
    alignment = by_key["b1_alignment_final_n10"]
    ordinary_tps = float(ordinary["production_tokens_per_second"])
    alignment_tps = float(alignment["production_tokens_per_second"])
    weighted_tps = _harmonic_mixture(ordinary_tps, alignment_tps)
    safe_b2 = by_key["b2_ordinary_outer0_inner24"]
    gpu_gain = (
        float(ordinary["gpu_tokens_per_second"]) / float(safe_b2["gpu_tokens_per_second"]) - 1.0
    ) * 100.0
    wall_gain = (ordinary_tps / float(safe_b2["production_tokens_per_second"]) - 1.0) * 100.0
    recommendation = {
        "batch_size": 1,
        "ordinary_case": "b1-ordinary-ac0",
        "alignment_case": "b1-alignment-ac8",
        "tokens_per_second": weighted_tps,
        "production_config": {
            "micro_batch_size": 1,
            "activation_checkpointing": True,
            "activation_checkpointing_on_alignment_only": True,
            "activation_checkpoint_layer_count": None,
            "ordinary_outer_checkpoint_layer_count": 0,
            "ordinary_inner_checkpoint_layer_count": 0,
            "alignment_outer_checkpoint_layer_count": 8,
            "alignment_inner_checkpoint_layer_count": 16,
            "dense_transfer_execution": "expanded",
        },
    }
    report_generator = Path(__file__).resolve()
    source_provenance = {
        "files": {
            "benchmark": benchmark_identity,
            "mtp": mtp_identity,
            "report_generator": _identity(report_generator),
        },
        "locked_input_count": len(CASE_SPECS) * 2 + 4,
        "locked_case_keys": [spec.key for spec in CASE_SPECS],
    }
    numerical_admission = _numerical_evidence(root)
    locked_inputs = {
        "sources": source_provenance["files"],
        "cases": {
            row["key"]: {
                "benchmark": row["source"],
                "power_csv": {name: row["power"][name] for name in ("path", "size", "sha256")},
            }
            for row in rows
        },
        "numerical_evidence": {
            name: {key: numerical_admission[name][key] for key in ("report", "complete")}
            for name in ("expanded", "folded")
        },
    }
    batch_comparisons = [
        {
            "batch_size": 1,
            "accepted": True,
            "ordinary_case": ordinary["label"],
            "alignment_case": alignment["label"],
            "tokens_per_second": weighted_tps,
            "reason": "safe ordinary and alignment evidence exist for one unified batch",
        },
        {
            "batch_size": 2,
            "accepted": False,
            "ordinary_case": safe_b2["label"],
            "alignment_case": None,
            "tokens_per_second": None,
            "reason": "no safe locked B2 alignment artifact; mixed physical batches are forbidden",
        },
    ]
    return {
        "schema_version": 2,
        "kind": "rtx5090_base_dense_utilization_report",
        "accepted": True,
        "read_only_report_generation": True,
        "no_optimizer_created_by_report": True,
        "global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "minimum_headroom_gib": MINIMUM_HEADROOM_GIB,
        "ordinary_weight": ORDINARY_WEIGHT,
        "alignment_weight": ALIGNMENT_WEIGHT,
        "source_provenance": source_provenance,
        "locked_inputs": locked_inputs,
        "locked_inputs_sha256": _canonical_sha256(locked_inputs),
        "numerical_admission": numerical_admission,
        "rows": rows,
        "recommendation": recommendation,
        "batch_comparisons": batch_comparisons,
        "weighted_comparison": {
            "method": "95_5_harmonic_wall_including_full_teacher_stage_restore",
            "ordinary_tokens_per_second": ordinary_tps,
            "alignment_tokens_per_second": alignment_tps,
            "tokens_per_second": weighted_tps,
        },
        "comparisons": {
            "b1_ordinary_vs_safe_b2_gpu_percent": gpu_gain,
            "b1_ordinary_vs_safe_b2_wall_percent": wall_gain,
            "safe_b2_case": safe_b2["label"],
        },
        "production_contract": {
            "all_successful_samples_present_gradient_tensors": [72],
            "native_qwen35_mtp": True,
            "mtp_frozen": True,
            "optimizer_state_reserve_gib": 1.5,
            "optimizer_created": False,
            "optimizer_steps": 0,
            "physical_free_gate_gib": 3.0,
            "expanded_numerical_admission": "pass",
            "differentiable_folded_numerical_admission": "fail_experimental_only",
        },
    }


def _fmt(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):,.{digits}f}"


def _svg_document(body: str, *, width: int = 1280, height: int = 720) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        "<style>text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#182235}"
        ".title{font-size:24px;font-weight:700}.label{font-size:13px}"
        ".value{font-size:12px;font-weight:600}.gate{stroke:#c7354d;stroke-width:2;"
        "stroke-dasharray:6 5}</style>\n"
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
    gate: float | None = None,
) -> str:
    values = [float(row[key] or 0.0) for row in rows]
    maximum = max([*values, gate or 0.0, 1.0]) * 1.08
    label_width = 230
    plot_width = width - label_width - 85
    gap = 7
    bar_height = max((height - 50 - gap * max(len(rows) - 1, 0)) / len(rows), 9)
    parts = [f'<text class="title" x="{x}" y="{y + 25}">{html.escape(title)}</text>']
    if gate is not None:
        gate_x = x + label_width + plot_width * gate / maximum
        parts.append(
            f'<line class="gate" x1="{gate_x:.1f}" y1="{y + 38}" '
            f'x2="{gate_x:.1f}" y2="{y + height}"/>'
        )
    for index, (row, value) in enumerate(zip(rows, values, strict=True)):
        top = y + 45 + index * (bar_height + gap)
        length = plot_width * value / maximum
        parts.extend(
            (
                f'<text class="label" x="{x}" y="{top + bar_height * 0.72:.1f}">'
                f"{html.escape(str(row['label']))}</text>",
                f'<rect x="{x + label_width}" y="{top:.1f}" width="{length:.1f}" '
                f'height="{bar_height:.1f}" rx="3" fill="{color}" opacity="0.88"/>',
                f'<text class="value" x="{x + label_width + length + 6:.1f}" '
                f'y="{top + bar_height * 0.72:.1f}">{value:,.2f}</text>',
            )
        )
    return "\n".join(parts)


def _throughput_headroom_svg(report: Mapping[str, Any]) -> str:
    rows = [row for row in report["rows"] if row["production_tokens_per_second"] is not None]
    body = ['<rect width="1280" height="720" fill="#f7f9fc"/>']
    body.append(
        _bar_panel(
            rows,
            key="production_tokens_per_second",
            title="Wall throughput including teacher transitions (tok/s)",
            x=32,
            y=15,
            width=1210,
            height=320,
            color="#3478f6",
        )
    )
    body.append(
        _bar_panel(
            rows,
            key="conservative_free_gib",
            title="Conservative physical/free headroom (GiB; red gate = 3 GiB)",
            x=32,
            y=365,
            width=1210,
            height=310,
            color="#13a47a",
            gate=3.0,
        )
    )
    return _svg_document("\n".join(body))


def _power_svg(report: Mapping[str, Any]) -> str:
    rows = []
    for row in report["rows"]:
        high = row["power"]["high_utilization"]
        rows.append(
            {
                "label": row["label"],
                "power": high["stats"]["power_draw_w"]["mean"],
                "util": high["stats"]["utilization_gpu_percent"]["mean"],
            }
        )
    body = ['<rect width="1280" height="720" fill="#f7f9fc"/>']
    body.append(
        _bar_panel(
            rows,
            key="power",
            title="Mean board power at individual GPU-util >= 90% samples (W)",
            x=35,
            y=18,
            width=590,
            height=650,
            color="#d97706",
            gate=600.0,
        )
    )
    body.append(
        _bar_panel(
            rows,
            key="util",
            title="Mean GPU utilization within >=90% subset (%)",
            x=655,
            y=18,
            width=590,
            height=650,
            color="#805ad5",
        )
    )
    return _svg_document("\n".join(body))


def _mixture_svg(report: Mapping[str, Any]) -> str:
    weighted = report["weighted_comparison"]
    throughput_rows = [
        {"label": "95% ordinary", "value": weighted["ordinary_tokens_per_second"]},
        {"label": "5% alignment", "value": weighted["alignment_tokens_per_second"]},
        {"label": "95/5 harmonic mixture", "value": weighted["tokens_per_second"]},
    ]
    alignment = next(row for row in report["rows"] if row["key"] == "b1_alignment_final_n10")
    timing = alignment["timing_breakdown_seconds"]
    timing_rows = [
        {"label": "alignment forward", "value": 1000 * timing["forward_seconds"]},
        {"label": "alignment backward", "value": 1000 * timing["backward_seconds"]},
        {
            "label": "teacher CPU->GPU stage",
            "value": 1000 * timing["teacher_cpu_offload_stage_seconds"],
        },
        {
            "label": "teacher restore",
            "value": 1000 * timing["teacher_cpu_offload_restore_seconds"],
        },
    ]
    body = ['<rect width="1280" height="720" fill="#f7f9fc"/>']
    body.append(
        _bar_panel(
            throughput_rows,
            key="value",
            title="B1 production mixture throughput (tok/s)",
            x=35,
            y=20,
            width=1210,
            height=285,
            color="#3478f6",
        )
    )
    body.append(
        _bar_panel(
            timing_rows,
            key="value",
            title="Selected alignment mean timing components (ms)",
            x=35,
            y=365,
            width=1210,
            height=300,
            color="#dc3f52",
        )
    )
    return _svg_document("\n".join(body))


def _markdown(report: Mapping[str, Any], *, images: tuple[str, str, str]) -> str:
    rows = report["rows"]
    recommendation = report["recommendation"]
    comparison = report["comparisons"]
    lines = [
        "# RTX 5090 expanded production sweep",
        "",
        "结论：数值准入锁定为 **expanded PASS / differentiable folded FAIL**。"
        "机器推荐统一使用 B1：ordinary outer0/inner0，alignment outer8/inner16。",
        "",
        f"![吞吐与余量]({images[0]})",
        "",
        f"![高利用率功耗]({images[1]})",
        "",
        f"![时序与 95/5 mixture]({images[2]})",
        "",
        "## 推荐",
        "",
        f"- 95/5 harmonic wall（包含完整 teacher stage/restore）："
        f"`{recommendation['tokens_per_second']:.6f} tok/s`。",
        f"- B1 ordinary 相对安全 B2 ordinary 的 GPU-event 吞吐优势："
        f"`+{comparison['b1_ordinary_vs_safe_b2_gpu_percent']:.3f}%`。",
        "- 安全门槛：PyTorch estimated free 与 NVML physical free 都必须不低于 `3 GiB`。",
        "- 所有成功样本均为 `72/72` finite gradients；native Qwen3.5 MTP 冻结、权重 `0.1`、"
        "SDPA；全程保留 `1.5 GiB` 非 optimizer reserve；没有创建 optimizer，也没有 step。",
        "",
        "## Sweep",
        "",
        "| case | outer/inner | status | GPU/wall+stage tok/s | estimated/NVML free GiB | high-util power mean/p95 W | high-util samples | gradients |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        high = row["power"]["high_utilization"]
        power = high["stats"]["power_draw_w"]
        lines.append(
            f"| {row['label']} | {row['activation_checkpoint_layer_count']}/"
            f"{row['dense_transfer_checkpoint_layer_count']} | {row['classification']} | "
            f"{_fmt(row['gpu_tokens_per_second'])}/{_fmt(row['production_tokens_per_second'])} | "
            f"{_fmt(row['minimum_estimated_headroom_gib'])}/"
            f"{_fmt(row['minimum_nvml_physical_free_gib'])} | "
            f"{_fmt(power['mean'], 2)}/{_fmt(power['p95'], 2)} | "
            f"{high['sample_count']} | {row['health']['present_gradient_tensor_counts'] or 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "## 为什么不选 B2",
            "",
            "- B2 outer0/inner24 通过 3 GiB 门槛，但 B1 ordinary GPU-event 吞吐比它高"
            " `+9.964%`，且 B2 没有同批次、通过门槛的锁定 alignment 证据。",
            "- B2 outer8/inner8 的 graph 可完成，但 physical/free 余量低于 3 GiB，因此标为 unsafe。",
            "- B2 outer4/inner4 在 warmup 触发 `device not ready`，同期 CUDA free=0，保留为"
            "容量失败证据，不参与推荐。",
            "- alignment outer0/inner24 同样低于 3 GiB；outer12/inner12 安全但比 outer8/inner16 慢。",
            "",
            "## 功耗解释",
            "",
            "功耗表使用 CSV 中 GPU utilization >=90% 的单独样本统计，不用启动/收尾 idle 样本稀释。"
            "低于 600 W 不是 GPU 空闲的充分证据：checkpoint、teacher 搬运和不同 kernel mix 会形成"
            "阶段性低功耗；选型仍以完整 wall 吞吐、数值准入和物理余量为准。",
            "",
            "## 可复现性",
            "",
            "生成器在计算前逐一核验 7 个 benchmark JSON、7 个功耗 CSV、expanded/folded 两组"
            "数值证据及 benchmark/MTP 源文件 SHA。MANIFEST 与 COMPLETE 再绑定全部输出；任何输入"
            "字节、公式合同、72-gradient/MTP/optimizer 合同或 3 GiB 分类变化都会 fail closed。",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, value: str) -> None:
    from twen.utils import atomic_write_text

    atomic_write_text(path, value)


def _canonical_paths(prefix: Path) -> dict[str, Path]:
    prefix = prefix.resolve()
    return {
        "report_json": prefix.with_name(prefix.name + ".json"),
        "report_markdown": prefix.with_name(prefix.name + ".md"),
        "throughput_memory_svg": prefix.with_name(prefix.name + "-throughput-memory.svg"),
        "power_svg": prefix.with_name(prefix.name + "-power.svg"),
        "utilization_svg": prefix.with_name(prefix.name + "-utilization.svg"),
        "manifest": prefix.with_name(prefix.name + ".MANIFEST.json"),
        "complete": prefix.with_name(prefix.name + ".COMPLETE"),
        "approval": prefix.with_name(prefix.name + ".approval.json"),
    }


def _write_canonical(report: Mapping[str, Any], prefix: Path) -> dict[str, Path]:
    paths = _canonical_paths(prefix)
    approval_before = paths["approval"].read_bytes() if paths["approval"].is_file() else None
    paths["complete"].unlink(missing_ok=True)
    _atomic_write(paths["report_json"], _json_text(report))
    _atomic_write(
        paths["report_markdown"],
        _markdown(
            report,
            images=(
                paths["throughput_memory_svg"].name,
                paths["power_svg"].name,
                paths["utilization_svg"].name,
            ),
        ),
    )
    _atomic_write(paths["throughput_memory_svg"], _throughput_headroom_svg(report))
    _atomic_write(paths["power_svg"], _power_svg(report))
    _atomic_write(paths["utilization_svg"], _mixture_svg(report))
    payload_keys = (
        "report_json",
        "report_markdown",
        "throughput_memory_svg",
        "power_svg",
        "utilization_svg",
    )
    provenance_sha = _canonical_sha256(report["source_provenance"])
    manifest = {
        "schema_version": 1,
        "kind": "twen_rtx5090_base_dense_utilization_report_bundle",
        "accepted": report["accepted"],
        "recommendation": report["recommendation"],
        "source_provenance_sha256": provenance_sha,
        "locked_inputs_sha256": report["locked_inputs_sha256"],
        "files": {
            key: _identity(paths[key], relative_to=paths["manifest"].parent) for key in payload_keys
        },
    }
    _atomic_write(paths["manifest"], _json_text(manifest))
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
        "source_provenance_sha256": provenance_sha,
        "locked_inputs_sha256": report["locked_inputs_sha256"],
    }
    _atomic_write(paths["complete"], _json_text(complete))
    approval_after = paths["approval"].read_bytes() if paths["approval"].is_file() else None
    if approval_after != approval_before:
        raise RuntimeError("canonical report generation mutated the approval file")
    return paths


def _tree_identities(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): _identity(path, relative_to=root)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_sweep(
    report: Mapping[str, Any], output: Path, *, replace_existing: bool = False
) -> Path:
    output = output.resolve()
    staging = output.with_name(f".{output.name}.incomplete")
    if staging.exists():
        raise ValueError(f"stale report staging directory exists: {staging}")
    (staging / "charts").mkdir(parents=True)
    summary = staging / "summary.json"
    markdown = staging / "REPORT.zh-CN.md"
    throughput = staging / "charts/throughput-headroom.svg"
    power = staging / "charts/high-utilization-power.svg"
    mixture = staging / "charts/timing-mixture.svg"
    _atomic_write(summary, _json_text(report))
    _atomic_write(
        markdown,
        _markdown(
            report,
            images=(
                "charts/throughput-headroom.svg",
                "charts/high-utilization-power.svg",
                "charts/timing-mixture.svg",
            ),
        ),
    )
    _atomic_write(throughput, _throughput_headroom_svg(report))
    _atomic_write(power, _power_svg(report))
    _atomic_write(mixture, _mixture_svg(report))
    payloads = (summary, markdown, throughput, power, mixture)
    manifest = {
        "schema_version": 1,
        "kind": "twen_rtx5090_expanded_production_sweep_bundle",
        "accepted": report["accepted"],
        "recommendation": report["recommendation"],
        "source_provenance_sha256": _canonical_sha256(report["source_provenance"]),
        "locked_inputs_sha256": report["locked_inputs_sha256"],
        "files": {
            path.relative_to(staging).as_posix(): _identity(path, relative_to=staging)
            for path in payloads
        },
    }
    manifest_path = staging / "MANIFEST.json"
    _atomic_write(manifest_path, _json_text(manifest))
    _atomic_write(
        staging / "COMPLETE",
        _json_text(
            {
                "schema_version": 1,
                "kind": "twen_rtx5090_expanded_production_sweep_complete",
                "manifest": "MANIFEST.json",
                "manifest_sha256": _sha256(manifest_path),
                "summary": {"path": "summary.json", "sha256": _sha256(summary)},
                "accepted": report["accepted"],
                "locked_inputs_sha256": report["locked_inputs_sha256"],
            }
        ),
    )
    if output.exists():
        if not output.is_dir():
            shutil.rmtree(staging)
            raise ValueError(f"existing sweep report is not a directory: {output}")
        if _tree_identities(output) == _tree_identities(staging):
            shutil.rmtree(staging)
            return output
        if not replace_existing:
            shutil.rmtree(staging)
            raise ValueError(f"existing sweep report differs from regenerated output: {output}")
        existing_complete = _load_json(output / "COMPLETE")
        existing_manifest = output / "MANIFEST.json"
        manifest_value = _load_json(existing_manifest)
        summary_identity = existing_complete.get("summary")
        if not (
            existing_complete.get("kind") == "twen_rtx5090_expanded_production_sweep_complete"
            and existing_complete.get("manifest") == "MANIFEST.json"
            and existing_complete.get("manifest_sha256") == _sha256(existing_manifest)
            and manifest_value.get("kind") == "twen_rtx5090_expanded_production_sweep_bundle"
            and isinstance(summary_identity, Mapping)
            and summary_identity.get("path") == "summary.json"
            and summary_identity.get("sha256") == _sha256(output / "summary.json")
        ):
            shutil.rmtree(staging)
            raise ValueError("refusing to replace an unauthenticated sweep report")
        backup = output.with_name(f".{output.name}.replaced")
        if backup.exists():
            shutil.rmtree(staging)
            raise ValueError(f"stale report replacement backup exists: {backup}")
        os.replace(output, backup)
        try:
            os.replace(staging, output)
        except BaseException:
            os.replace(backup, output)
            raise
        shutil.rmtree(backup)
        return output
    os.replace(staging, output)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    report_dir = args.report_dir
    if not report_dir.is_absolute():
        report_dir = root / report_dir
    canonical_prefix = args.canonical_prefix
    if not canonical_prefix.is_absolute():
        canonical_prefix = root / canonical_prefix
    report = build_report(root)
    output = _write_sweep(
        report,
        report_dir,
        replace_existing=args.replace_existing_report,
    )
    canonical = _write_canonical(report, canonical_prefix)
    print(
        json.dumps(
            {
                "ok": True,
                "report_dir": str(output),
                "canonical_report": str(canonical["report_json"]),
                "recommendation": report["recommendation"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
