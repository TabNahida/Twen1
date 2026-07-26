#!/usr/bin/env python3
"""Benchmark the complete single-GPU Base dense graph without an optimizer.

The production-shaped default is one 4096-token sequence through all 24
student layers with a shared 9B hidden-alignment teacher.  ``--batch-size``
can probe larger single-GPU microbatches without changing the production
sequence or top-k shapes.  The benchmark executes the shared-only anchor pass,
streaming NTP/KD/anchor vocabulary losses, optional native MTP and online
hidden alignment, and backward.  Hidden alignment can be disabled while the
shared 9B teacher/donor remains resident to model an ordinary (non-alignment)
training batch.  No optimizer is constructed and no parameter update is
possible in this program.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

DEFAULT_BATCH_SIZE = 1
PRODUCTION_SHAPE = {
    "sequence_length": 4096,
    "student_hidden_size": 1024,
    "student_intermediate_size": 3584,
    "student_layers": 24,
    "donor_hidden_size": 4096,
    "donor_intermediate_size": 12288,
    "donor_layers": 32,
    "vocabulary_size": 248320,
    "experts": 8,
    "expert_intermediate_size": 1536,
    "teacher_top_k": 64,
}
GIB = 1024**3
MIB = 1024**2
PROFILE_SUMMARY_ROW_LIMIT = 100
GPU_TELEMETRY_COLUMNS = (
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
DENSE_TRANSFER_EXECUTION_MODES = ("expanded", "differentiable_folded")
DENSE_TRANSFER_NUMERICAL_ADMISSION = {
    "production_reference_mode": "expanded",
    "expanded_selective_checkpoint": {
        "status": "pass",
        "semantics": (
            "historical expanded formula; selective checkpointing may change only "
            "recomputation and activation retention, and is admitted by bitwise isolated "
            "CUDA evidence plus full-graph loss and pre-locked group-vector gates"
        ),
        "evidence": (
            "artifacts/audits/differentiable-fold-numerical-admission/"
            "EXPANDED_SELECTIVE_CHECKPOINT_FULL_GRAPH_REPORT.md"
        ),
    },
    "differentiable_folded": {
        "status": "fail_experimental_only",
        "semantics": (
            "BF16 reassociation changes the optimization trajectory; the locked four-real-KD "
            "accumulation scale gate failed and cannot be waived for throughput"
        ),
        "evidence": (
            "artifacts/audits/differentiable-fold-numerical-admission/"
            "FULL_GRAPH_V1_REAL_KD_ACCUMULATION_REPORT.md"
        ),
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone",
        default="artifacts/models/qwen3.5-0.8b-base",
        help="local Qwen3.5-0.8B-Base checkpoint",
    )
    parser.add_argument(
        "--teacher",
        default="artifacts/models/qwen3.5-9b-base",
        help="local Qwen3.5-9B-Base text teacher/donor checkpoint",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--cuda-home",
        default=os.environ.get("CUDA_HOME", "/usr/local/cuda"),
        help="compiler/header-coherent CUDA toolkit used by TileLang JIT",
    )
    parser.add_argument(
        "--fla-backend",
        choices=("triton", "tilelang"),
        default="triton",
        help=(
            "FLA backward backend; Triton is the production RTX 5090 default because "
            "the TileLang T=4096 kernel is not alignment-safe"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "single-GPU microbatch size; values above 1 remain production-shaped "
            "when sequence-length=4096 and teacher-top-k=64"
        ),
    )
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--teacher-top-k", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--loss-chunk-tokens", type=int, default=128)
    parser.add_argument(
        "--loss-checkpoint-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compile-streaming-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compile the static per-chunk CUDA vocabulary reductions",
    )
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    checkpoint_count_group = parser.add_mutually_exclusive_group()
    checkpoint_count_group.add_argument(
        "--activation-checkpoint-layer-count",
        type=int,
        default=None,
        metavar="N",
        help=(
            "number of evenly spaced student decoder layers to checkpoint (0..24); "
            "defaults to 24 with --activation-checkpointing and 0 with "
            "--no-activation-checkpointing"
        ),
    )
    checkpoint_count_group.add_argument(
        "--activation-checkpoint-layer-counts",
        default=None,
        metavar="N1,N2,...",
        help=(
            "run multiple checkpoint-layer counts in one process while reusing the "
            "loaded student and 9B teacher/donor (for example 16,18,20,24)"
        ),
    )
    parser.add_argument(
        "--dense-transfer-execution",
        choices=DENSE_TRANSFER_EXECUTION_MODES,
        default="expanded",
        help=(
            "dense transfer branch implementation; expanded is the numerically admitted "
            "production reference and differentiable_folded is experimental-only"
        ),
    )
    parser.add_argument(
        "--dense-transfer-checkpoint-layer-count",
        type=int,
        default=0,
        metavar="N",
        help=(
            "number of deterministic evenly spaced inner transfer checkpoints selected "
            "from each case's outer-checkpoint complement (0..24; capped by availability)"
        ),
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--torch-profile-trace",
        default=None,
        help=(
            "optional PyTorch Chrome trace path; profiles one extra complete "
            "iteration that is excluded from benchmark samples"
        ),
    )
    parser.add_argument(
        "--cuda-profiler-api",
        action="store_true",
        help=(
            "surround the extra profiling iteration with cudaProfilerStart/Stop "
            "for nsys --capture-range=cudaProfilerApi"
        ),
    )
    parser.add_argument(
        "--gpu-telemetry-output",
        default=None,
        help=(
            "optional CSV sampled by a read-only nvidia-smi child while the single "
            "warmup/measurement case runs; activation-checkpoint sweeps are rejected"
        ),
    )
    parser.add_argument(
        "--gpu-telemetry-interval-ms",
        type=int,
        default=250,
        help="nvidia-smi sampling interval for --gpu-telemetry-output",
    )
    parser.add_argument(
        "--nvidia-smi",
        default=None,
        help="optional explicit nvidia-smi executable (WSL fallback is auto-detected)",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--adapter-init-std", type=float, default=0.01)
    parser.add_argument("--branch-scale", type=float, default=0.01)
    parser.add_argument("--ntp-weight", type=float, default=1.0)
    parser.add_argument("--teacher-kd-weight", type=float, default=1.0)
    parser.add_argument("--anchor-kl-weight", type=float, default=0.1)
    parser.add_argument(
        "--hidden-alignment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "execute the 9B teacher forward and all-layer hidden objective; disable "
            "to model the ordinary 95%% batch while keeping the teacher/donor resident"
        ),
    )
    parser.add_argument("--hidden-alignment-weight", type=float, default=0.1)
    parser.add_argument(
        "--teacher-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "keep teacher-only state on CPU and stage it to this single GPU only "
            "for hidden-alignment iterations; shared donor projections stay on CUDA"
        ),
    )
    parser.add_argument(
        "--mtp-loss-weight",
        type=float,
        default=0.0,
        help="enable the frozen native Qwen3.5 MTP block with this loss weight",
    )
    parser.add_argument(
        "--optimizer-state-reserve-gib",
        type=float,
        default=1.5,
        help=(
            "CUDA memory to allocate and touch for projected Adam moments; "
            "this is raw reserve storage, not an optimizer"
        ),
    )
    parser.add_argument(
        "--reserve-chunk-mib",
        type=int,
        default=256,
        help="maximum allocation size for the raw optimizer-state reserve",
    )
    parser.add_argument(
        "--temporary-root",
        default="artifacts/benchmarks",
        help="parent directory for automatically deleted mapping/adapter artifacts",
    )
    parser.add_argument("--output", default=None, help="optional machine-readable JSON path")
    parser.add_argument(
        "--allow-other-gpu",
        action="store_true",
        help="allow execution on a CUDA GPU whose name does not contain 5090",
    )
    parser.add_argument(
        "--allow-non-production-shape",
        action="store_true",
        help="allow sequence/top-k overrides for diagnostics; results are marked non-production",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate checkpoint metadata and print the graph contract without importing CUDA",
    )
    return parser


def _evenly_spaced_layer_indices(count: int, *, total_layers: int) -> tuple[int, ...]:
    """Select deterministic layer indices spanning the complete decoder depth."""

    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError("activation-checkpoint-layer-count must be an integer")
    if isinstance(total_layers, bool) or not isinstance(total_layers, int) or total_layers <= 0:
        raise ValueError("total_layers must be a positive integer")
    if not 0 <= count <= total_layers:
        raise ValueError(f"activation-checkpoint-layer-count must be in [0, {total_layers}]")
    if count == 0:
        return ()
    if count == 1:
        return (total_layers // 2,)
    # Integer nearest-neighbour sampling of inclusive endpoints is stable
    # across Python/numpy versions and guarantees unique indices for count <= L.
    denominator = count - 1
    indices = tuple(
        (2 * position * (total_layers - 1) + denominator) // (2 * denominator)
        for position in range(count)
    )
    if len(set(indices)) != count:  # pragma: no cover - protected by count <= total_layers
        raise RuntimeError(f"selective checkpoint indices are not unique: {indices}")
    return indices


def _activation_checkpoint_indices_for_count(
    enabled: bool,
    count: int,
) -> tuple[int, ...]:
    if not isinstance(enabled, bool):
        raise ValueError("activation-checkpointing must be a boolean")
    indices = _evenly_spaced_layer_indices(
        count,
        total_layers=PRODUCTION_SHAPE["student_layers"],
    )
    if not enabled and indices:
        raise ValueError(
            "--no-activation-checkpointing requires "
            "--activation-checkpoint-layer-count=0 (or no count override)"
        )
    return indices


def _parse_activation_checkpoint_layer_counts(raw: str) -> tuple[int, ...]:
    if not isinstance(raw, str):
        raise ValueError("activation-checkpoint-layer-counts must be a comma-separated string")
    fields = raw.split(",")
    if not fields or any(not field.strip() for field in fields):
        raise ValueError("activation-checkpoint-layer-counts must contain comma-separated integers")
    try:
        counts = tuple(int(field.strip()) for field in fields)
    except ValueError as error:
        raise ValueError(
            "activation-checkpoint-layer-counts must contain comma-separated integers"
        ) from error
    if len(set(counts)) != len(counts):
        raise ValueError("activation-checkpoint-layer-counts must not contain duplicates")
    return counts


def _activation_checkpoint_layer_cases(
    args: argparse.Namespace,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    enabled = args.activation_checkpointing
    sweep_raw = args.activation_checkpoint_layer_counts
    single_count = args.activation_checkpoint_layer_count
    if sweep_raw is not None and single_count is not None:
        raise ValueError(
            "activation-checkpoint-layer-count and activation-checkpoint-layer-counts "
            "are mutually exclusive"
        )
    if sweep_raw is None:
        if single_count is None:
            count = PRODUCTION_SHAPE["student_layers"] if enabled else 0
        else:
            if isinstance(single_count, bool) or not isinstance(single_count, int):
                raise ValueError("activation-checkpoint-layer-count must be an integer")
            count = single_count
        return ((count, _activation_checkpoint_indices_for_count(enabled, count)),)

    counts = _parse_activation_checkpoint_layer_counts(sweep_raw)
    return tuple(
        (count, _activation_checkpoint_indices_for_count(enabled, count)) for count in counts
    )


def _activation_checkpoint_layer_indices(args: argparse.Namespace) -> tuple[int, ...]:
    cases = _activation_checkpoint_layer_cases(args)
    if args.activation_checkpoint_layer_counts is not None:
        raise ValueError("activation checkpoint sweep has no single layer-index selection")
    return cases[0][1]


def _dense_transfer_checkpoint_layer_indices(
    requested_count: int,
    *,
    outer_checkpoint_layer_indices: Sequence[int],
    total_layers: int = PRODUCTION_SHAPE["student_layers"],
) -> tuple[int, ...]:
    """Select inner checkpoints evenly from the outer-checkpoint complement."""

    if isinstance(requested_count, bool) or not isinstance(requested_count, int):
        raise ValueError("dense-transfer-checkpoint-layer-count must be an integer")
    if not 0 <= requested_count <= total_layers:
        raise ValueError(f"dense-transfer-checkpoint-layer-count must be in [0, {total_layers}]")
    outer = tuple(outer_checkpoint_layer_indices)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in outer):
        raise ValueError("outer checkpoint layer indices must be integers")
    if tuple(sorted(set(outer))) != outer:
        raise ValueError("outer checkpoint layer indices must be sorted and unique")
    if any(index < 0 or index >= total_layers for index in outer):
        raise ValueError(f"outer checkpoint layer indices must be in [0, {total_layers})")
    outer_set = set(outer)
    available = tuple(index for index in range(total_layers) if index not in outer_set)
    effective_count = min(requested_count, len(available))
    if effective_count == 0:
        return ()
    selected_positions = _evenly_spaced_layer_indices(
        effective_count,
        total_layers=len(available),
    )
    selected = tuple(available[position] for position in selected_positions)
    if outer_set.intersection(selected):  # pragma: no cover - selected from complement
        raise RuntimeError("nested outer/inner dense transfer checkpointing is forbidden")
    return selected


def _dense_transfer_case_contract(
    args: argparse.Namespace,
    *,
    outer_checkpoint_layer_indices: Sequence[int],
) -> dict[str, Any]:
    requested = args.dense_transfer_checkpoint_layer_count
    inner = _dense_transfer_checkpoint_layer_indices(
        requested,
        outer_checkpoint_layer_indices=outer_checkpoint_layer_indices,
    )
    outer = tuple(outer_checkpoint_layer_indices)
    return {
        "dense_transfer_execution": args.dense_transfer_execution,
        "dense_transfer_checkpoint_layer_count_requested": requested,
        "dense_transfer_checkpoint_layer_count_effective": len(inner),
        "dense_transfer_checkpoint_layer_indices": list(inner),
        "dense_transfer_outer_inner_disjoint": not bool(set(outer).intersection(inner)),
        "dense_transfer_checkpoint_selection_policy": (
            "deterministic_evenly_spaced_outer_complement"
        ),
    }


def _dense_transfer_execution_contract(execution_mode: str) -> dict[str, Any]:
    if execution_mode not in DENSE_TRANSFER_EXECUTION_MODES:
        raise ValueError("dense-transfer-execution must be 'expanded' or 'differentiable_folded'")
    production_enabled = execution_mode == "expanded"
    return {
        "mode": execution_mode,
        "production_enabled": production_enabled,
        "selected_mode_numerical_status": (
            "admitted_production_reference" if production_enabled else "failed_experimental_only"
        ),
        "numerical_admission": dict(DENSE_TRANSFER_NUMERICAL_ADMISSION),
    }


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "teacher_top_k": args.teacher_top_k,
        "loss_chunk_tokens": args.loss_chunk_tokens,
        "repeats": args.repeats,
        "reserve_chunk_mib": args.reserve_chunk_mib,
    }
    invalid = [name for name, value in positive.items() if isinstance(value, bool) or value <= 0]
    if invalid:
        raise ValueError(f"positive benchmark arguments required: {invalid}")
    if isinstance(args.warmup, bool) or args.warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if (
        isinstance(args.dense_transfer_checkpoint_layer_count, bool)
        or not isinstance(args.dense_transfer_checkpoint_layer_count, int)
        or not 0 <= args.dense_transfer_checkpoint_layer_count <= PRODUCTION_SHAPE["student_layers"]
    ):
        raise ValueError("dense-transfer-checkpoint-layer-count must be in [0, 24]")
    _dense_transfer_execution_contract(args.dense_transfer_execution)
    if isinstance(args.gpu_telemetry_interval_ms, bool) or args.gpu_telemetry_interval_ms <= 0:
        raise ValueError("gpu-telemetry-interval-ms must be a positive integer")
    for name in (
        "temperature",
        "adapter_init_std",
        "branch_scale",
        "ntp_weight",
        "teacher_kd_weight",
        "anchor_kl_weight",
        "hidden_alignment_weight",
        "mtp_loss_weight",
        "optimizer_state_reserve_gib",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if args.temperature <= 0 or args.adapter_init_std <= 0 or args.branch_scale <= 0:
        raise ValueError("temperature, adapter-init-std, and branch-scale must be positive")
    if args.teacher_top_k >= PRODUCTION_SHAPE["vocabulary_size"]:
        raise ValueError("teacher-top-k must be smaller than the vocabulary")
    if not isinstance(args.hidden_alignment, bool):
        raise ValueError("hidden-alignment must be a boolean")
    if not isinstance(args.teacher_cpu_offload, bool):
        raise ValueError("teacher-cpu-offload must be a boolean")
    if args.teacher_cpu_offload and not args.device.startswith("cuda"):
        raise ValueError("teacher-cpu-offload requires exactly one CUDA device")
    _activation_checkpoint_layer_cases(args)
    if args.mtp_loss_weight > 0 and args.sequence_length < 3:
        raise ValueError("mtp-loss-weight>0 requires sequence-length>=3 for an L-2 target")
    shape_override = (
        args.sequence_length != PRODUCTION_SHAPE["sequence_length"]
        or args.teacher_top_k != PRODUCTION_SHAPE["teacher_top_k"]
    )
    if shape_override and not args.allow_non_production_shape:
        raise ValueError(
            "sequence-length=4096 and teacher-top-k=64 are required; "
            "pass --allow-non-production-shape only for diagnostics"
        )
    if not args.device.startswith("cuda") and not args.dry_run:
        raise ValueError("the complete dense graph benchmark requires a CUDA device")
    if args.dry_run and (args.torch_profile_trace or args.cuda_profiler_api):
        raise ValueError("profiling options require a CUDA benchmark, not --dry-run")
    if args.activation_checkpoint_layer_counts is not None and (
        args.torch_profile_trace or args.cuda_profiler_api
    ):
        raise ValueError("profiling options do not support activation checkpoint sweeps")
    if args.gpu_telemetry_output and args.activation_checkpoint_layer_counts is not None:
        raise ValueError("gpu telemetry requires one activation-checkpoint case per process")
    if args.gpu_telemetry_output and args.dry_run:
        raise ValueError("gpu telemetry requires a CUDA benchmark, not --dry-run")


def _read_text_config(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    text = value.get("text_config", value)
    if not isinstance(text, dict):
        raise ValueError(f"invalid text_config in {path}")
    return text


def _checkpoint_shape_report(
    backbone: str | Path,
    teacher: str | Path,
    *,
    require_mtp: bool = False,
) -> dict[str, Any]:
    student = _read_text_config(backbone)
    donor = _read_text_config(teacher)
    actual = {
        "student_hidden_size": student.get("hidden_size"),
        "student_intermediate_size": student.get("intermediate_size"),
        "student_layers": student.get("num_hidden_layers"),
        "donor_hidden_size": donor.get("hidden_size"),
        "donor_intermediate_size": donor.get("intermediate_size"),
        "donor_layers": donor.get("num_hidden_layers"),
        "student_vocabulary_size": student.get("vocab_size"),
        "donor_vocabulary_size": donor.get("vocab_size"),
        "student_model_type": student.get("model_type"),
        "donor_model_type": donor.get("model_type"),
        "student_tied_embeddings": bool(student.get("tie_word_embeddings")),
        "student_mtp_num_hidden_layers": student.get("mtp_num_hidden_layers"),
        "student_mtp_use_dedicated_embeddings": student.get("mtp_use_dedicated_embeddings"),
    }
    expected_pairs = {
        "student_hidden_size": PRODUCTION_SHAPE["student_hidden_size"],
        "student_intermediate_size": PRODUCTION_SHAPE["student_intermediate_size"],
        "student_layers": PRODUCTION_SHAPE["student_layers"],
        "donor_hidden_size": PRODUCTION_SHAPE["donor_hidden_size"],
        "donor_intermediate_size": PRODUCTION_SHAPE["donor_intermediate_size"],
        "donor_layers": PRODUCTION_SHAPE["donor_layers"],
        "student_vocabulary_size": PRODUCTION_SHAPE["vocabulary_size"],
        "donor_vocabulary_size": PRODUCTION_SHAPE["vocabulary_size"],
    }
    mismatches = {
        name: {"expected": expected, "actual": actual[name]}
        for name, expected in expected_pairs.items()
        if actual[name] != expected
    }
    if actual["student_model_type"] != "qwen3_5_text":
        mismatches["student_model_type"] = {
            "expected": "qwen3_5_text",
            "actual": actual["student_model_type"],
        }
    if actual["donor_model_type"] != "qwen3_5_text":
        mismatches["donor_model_type"] = {
            "expected": "qwen3_5_text",
            "actual": actual["donor_model_type"],
        }
    if require_mtp:
        mtp_expected = {
            "student_mtp_num_hidden_layers": 1,
            "student_mtp_use_dedicated_embeddings": False,
        }
        for name, expected in mtp_expected.items():
            if actual[name] != expected:
                mismatches[name] = {"expected": expected, "actual": actual[name]}
    return {"ok": not mismatches, "actual": actual, "mismatches": mismatches}


def _base_contract(args: argparse.Namespace, shape_report: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint_cases = _activation_checkpoint_layer_cases(args)
    dense_transfer_cases = tuple(
        _dense_transfer_case_contract(
            args,
            outer_checkpoint_layer_indices=checkpoint_layer_indices,
        )
        for _, checkpoint_layer_indices in checkpoint_cases
    )
    sweep_enabled = args.activation_checkpoint_layer_counts is not None
    checkpoint_layer_indices = checkpoint_cases[0][1] if not sweep_enabled else None
    dense_transfer_case = dense_transfer_cases[0] if not sweep_enabled else None
    mtp_enabled = args.mtp_loss_weight > 0
    production_shape = bool(
        shape_report["ok"]
        and args.sequence_length == PRODUCTION_SHAPE["sequence_length"]
        and args.teacher_top_k == PRODUCTION_SHAPE["teacher_top_k"]
    )
    contract = {
        "benchmark": "base_dense_24_layer_full_graph",
        "track": "base",
        "stage": "dense-oracle",
        "production_shape": production_shape,
        "dry_run": bool(args.dry_run),
        "no_optimizer_created": True,
        "no_optimizer_steps": True,
        "optimizer_step_calls": 0,
        "experimental_execution": _dense_transfer_execution_contract(args.dense_transfer_execution),
        "batch": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "logical_tokens": args.batch_size * args.sequence_length,
        },
        "graph": {
            "student_layers_active": PRODUCTION_SHAPE["student_layers"],
            "anchor_shared_only_forward": True,
            "student_dense_forward": True,
            "streaming_ntp": True,
            "streaming_teacher_kd": True,
            "streaming_anchor_kl": True,
            "native_mtp_forward": mtp_enabled,
            "native_mtp_vocab_loss": mtp_enabled,
            "native_mtp_parameter_update": False,
            "online_hidden_alignment": bool(args.hidden_alignment),
            "teacher_hidden_forward": bool(args.hidden_alignment),
            "teacher_donor_resident": True,
            "teacher_cpu_offload": bool(args.teacher_cpu_offload),
            "backward": True,
            "parameter_update": False,
        },
        "runtime": {
            "activation_checkpointing": (
                any(bool(indices) for _, indices in checkpoint_cases)
                if sweep_enabled
                else bool(checkpoint_layer_indices)
            ),
            "activation_checkpointing_requested": bool(args.activation_checkpointing),
            "teacher_cpu_offload": bool(args.teacher_cpu_offload),
            "activation_checkpoint_layer_count": (
                None if checkpoint_layer_indices is None else len(checkpoint_layer_indices)
            ),
            "activation_checkpoint_layer_indices": (
                None if checkpoint_layer_indices is None else list(checkpoint_layer_indices)
            ),
            "dense_transfer_execution": args.dense_transfer_execution,
            "dense_transfer_checkpoint_layer_count_requested": (
                args.dense_transfer_checkpoint_layer_count
            ),
            "dense_transfer_checkpoint_layer_count_effective": (
                None
                if dense_transfer_case is None
                else dense_transfer_case["dense_transfer_checkpoint_layer_count_effective"]
            ),
            "dense_transfer_checkpoint_layer_indices": (
                None
                if dense_transfer_case is None
                else dense_transfer_case["dense_transfer_checkpoint_layer_indices"]
            ),
            "dense_transfer_outer_inner_disjoint": (
                None
                if dense_transfer_case is None
                else dense_transfer_case["dense_transfer_outer_inner_disjoint"]
            ),
            "dense_transfer_checkpoint_selection_policy": (
                "deterministic_evenly_spaced_outer_complement"
            ),
            "loss_chunk_tokens": args.loss_chunk_tokens,
            "loss_checkpoint_chunks": bool(args.loss_checkpoint_chunks),
            "compile_streaming_loss": bool(args.compile_streaming_loss),
            "bf16": True,
            "allow_tf32": True,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "fla_backend": args.fla_backend,
            "fla_tilelang_env": "0" if args.fla_backend == "triton" else "1",
        },
        "loss_weights": {
            "ntp": args.ntp_weight,
            "teacher_kd": args.teacher_kd_weight,
            "anchor_kl": args.anchor_kl_weight,
            "hidden_alignment": args.hidden_alignment_weight,
            "mtp": args.mtp_loss_weight,
            "temperature": args.temperature,
            "teacher_top_k": args.teacher_top_k,
        },
        "temporary_initialization": {
            "layer_map": "student layer i -> donor layer i for i in [0, 24)",
            "channel_map": "contiguous 8 x 1536 complete donor-FFN partition",
            "adapter_distribution": "independent deterministic BF16 normal",
            "adapter_init_std": args.adapter_init_std,
            "adapter_seed": args.seed,
            "branch_scale": args.branch_scale,
            "artifacts_deleted_after_build": True,
        },
        "mtp": {
            "enabled": mtp_enabled,
            "loss_weight": args.mtp_loss_weight,
            "source_role": "backbone" if mtp_enabled else None,
            "frozen": True if mtp_enabled else None,
            "parameter_update": False,
        },
        "teacher_cpu_offload": {
            "enabled": bool(args.teacher_cpu_offload),
            "scope": "single_gpu_dense_donor_alias_split",
            "teacher_load_device": "cpu" if args.teacher_cpu_offload else args.device,
            "shared_donor_projections_resident_on_cuda": True,
            "teacher_only_residency_between_iterations": (
                "cpu" if args.teacher_cpu_offload else "cuda"
            ),
            "stage_policy": (
                "per_hidden_alignment_iteration"
                if args.teacher_cpu_offload and args.hidden_alignment
                else "never"
            ),
            "transitions_in_gpu_iteration_timing": False,
        },
        "optimizer_state_reserve": {
            "requested_gib": args.optimizer_state_reserve_gib,
            "requested_bytes": round(args.optimizer_state_reserve_gib * GIB),
            "chunk_mib": args.reserve_chunk_mib,
            "cuda_tensor_dtype": "uint8",
            "touched": not args.dry_run and args.optimizer_state_reserve_gib > 0,
            "is_optimizer": False,
        },
        "gpu_telemetry": {
            "enabled": bool(args.gpu_telemetry_output),
            "output": (
                str(Path(args.gpu_telemetry_output).expanduser().resolve())
                if args.gpu_telemetry_output
                else None
            ),
            "interval_ms": args.gpu_telemetry_interval_ms,
            "read_only": True,
            "scope": "warmup_and_measurement_case",
            "sampler": "nvidia-smi" if args.gpu_telemetry_output else None,
        },
        "checkpoints": {
            "backbone": str(Path(args.backbone).resolve()),
            "teacher_and_donor": str(Path(args.teacher).resolve()),
            "shape_validation": dict(shape_report),
        },
    }
    if sweep_enabled:
        contract["runtime"]["activation_checkpoint_layer_counts"] = [
            count for count, _ in checkpoint_cases
        ]
        contract["runtime"]["activation_checkpoint_case_indices"] = [
            list(indices) for _, indices in checkpoint_cases
        ]
        contract["sweep"] = {
            "enabled": True,
            "axis": "activation_checkpoint_layer_count",
            "case_count": len(checkpoint_cases),
            "model_loads_per_cuda_run": 1,
            "shared_teacher_donor_across_cases": True,
            "shared_optimizer_state_reserve_across_cases": True,
            "independent_warmup_and_repeats": True,
        }
        contract["cases"] = []
        for position, ((count, indices), dense_case) in enumerate(
            zip(checkpoint_cases, dense_transfer_cases, strict=True),
            start=1,
        ):
            contract["cases"].append(
                {
                    "case": position,
                    "activation_checkpoint_layer_count": count,
                    "activation_checkpoint_layer_indices": list(indices),
                    **dense_case,
                }
            )
    return contract


def _write_temporary_calibration(
    root: Path,
    *,
    student_layers: int,
    student_hidden: int,
    donor_hidden: int,
    donor_intermediate: int,
    experts: int,
    init_std: float,
    seed: int,
    torch: Any,
) -> dict[str, Path]:
    """Write production-builder inputs with deterministic lightweight semantics."""

    from safetensors.torch import save_file

    if donor_intermediate % experts:
        raise ValueError("experts must evenly partition donor_intermediate")
    root.mkdir(parents=True, exist_ok=True)
    layer_map = root / "layer_map.json"
    channel_map = root / "channel_map.json"
    adapters_path = root / "adapters.safetensors"
    layer_map.write_text(
        json.dumps({"student_to_donor": list(range(student_layers))}),
        encoding="utf-8",
    )
    channel_map.write_text(
        json.dumps(
            {
                "indices": [
                    list(range(start, start + donor_intermediate // experts))
                    for start in range(0, donor_intermediate, donor_intermediate // experts)
                ]
            }
        ),
        encoding="utf-8",
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensors: dict[str, Any] = {}
    for layer in range(student_layers):
        tensors[f"layers.{layer}.A"] = (
            torch.randn(
                (donor_hidden, student_hidden),
                generator=generator,
                dtype=torch.bfloat16,
            )
            * init_std
        ).contiguous()
        tensors[f"layers.{layer}.B"] = (
            torch.randn(
                (student_hidden, donor_hidden),
                generator=generator,
                dtype=torch.bfloat16,
            )
            * init_std
        ).contiguous()
    save_file(tensors, adapters_path)
    del tensors
    return {
        "layer_map": layer_map,
        "channel_map": channel_map,
        "adapters": adapters_path,
    }


def _builder_config(args: argparse.Namespace, artifacts: Mapping[str, Path]) -> Any:
    architecture = SimpleNamespace(
        student_hidden_size=PRODUCTION_SHAPE["student_hidden_size"],
        student_intermediate_size=PRODUCTION_SHAPE["student_intermediate_size"],
        student_layers=PRODUCTION_SHAPE["student_layers"],
        donor_hidden_size=PRODUCTION_SHAPE["donor_hidden_size"],
        donor_intermediate_size=PRODUCTION_SHAPE["donor_intermediate_size"],
        donor_layers=PRODUCTION_SHAPE["donor_layers"],
        num_experts=PRODUCTION_SHAPE["experts"],
        expert_intermediate_size=PRODUCTION_SHAPE["expert_intermediate_size"],
        expert_initialization="donor",
        random_expert_seed=args.seed,
        layer_map_path=str(artifacts["layer_map"]),
        channel_map_path=str(artifacts["channel_map"]),
        adapter_init_path=str(artifacts["adapters"]),
        active_layers=lambda: tuple(range(PRODUCTION_SHAPE["student_layers"])),
    )
    optimizer = SimpleNamespace(
        adapter_lr=2e-4,
        router_lr=1e-3,
        lora_lr=2e-4,
        scale_lr=1e-3,
        weight_decay=0.01,
    )
    return SimpleNamespace(
        stage="dense-oracle",
        architecture=architecture,
        optimizer=optimizer,
        runtime=SimpleNamespace(
            dense_transfer_execution=args.dense_transfer_execution,
        ),
        losses=SimpleNamespace(mtp=args.mtp_loss_weight),
        sources=SimpleNamespace(
            backbone=SimpleNamespace(local_path=str(Path(args.backbone).resolve())),
            donor=SimpleNamespace(local_path=str(Path(args.teacher).resolve())),
        ),
    )


def _set_branch_scale(modules: Sequence[Any], value: float, *, torch: Any) -> None:
    with torch.no_grad():
        for module in modules:
            module.transfer_mlp.branch_scale.fill_(value)


def _storage_alias(left: Any, right: Any) -> bool:
    return bool(
        left.device == right.device
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        and left.storage_offset() == right.storage_offset()
        and tuple(left.shape) == tuple(right.shape)
    )


def _alias_report(built: Any, teacher: Any) -> dict[str, Any]:
    projections = ("gate_proj", "up_proj", "down_proj")
    transfer_names = ("gate_weight", "up_weight", "down_weight")
    object_aliases = 0
    storage_aliases = 0
    expected = len(built.transfer_modules) * len(projections)
    unique_shared_bytes = 0
    seen_storages: set[tuple[str, int]] = set()
    for student_layer, module in zip(
        built.student_layer_indices,
        built.transfer_modules,
        strict=True,
    ):
        teacher_mlp = teacher.layers[student_layer].mlp
        transfer = module.transfer_mlp
        for teacher_name, transfer_name in zip(projections, transfer_names, strict=True):
            teacher_weight = getattr(teacher_mlp, teacher_name).weight
            transfer_weight = getattr(transfer, transfer_name)
            object_aliases += int(teacher_weight is transfer_weight)
            shared = _storage_alias(teacher_weight, transfer_weight)
            storage_aliases += int(shared)
            key = (str(teacher_weight.device), teacher_weight.untyped_storage().data_ptr())
            if shared and key not in seen_storages:
                seen_storages.add(key)
                unique_shared_bytes += teacher_weight.untyped_storage().nbytes()
    embedding = built.model.model.embed_tokens.weight
    head = built.model.lm_head.weight
    return {
        "donor_teacher_shared_flag": bool(built.donor_teacher_shared),
        "expected_projection_aliases": expected,
        "parameter_object_aliases": object_aliases,
        "storage_aliases": storage_aliases,
        "all_donor_projection_aliases_exact": bool(
            built.donor_teacher_shared
            and object_aliases == expected
            and storage_aliases == expected
        ),
        "unique_shared_projection_storage_bytes": unique_shared_bytes,
        "student_embedding_lm_head_parameter_alias": embedding is head,
        "student_embedding_lm_head_storage_alias": _storage_alias(embedding, head),
    }


def _allocate_optimizer_state_reserve(
    requested_bytes: int,
    *,
    chunk_bytes: int,
    torch: Any,
    device: Any,
) -> tuple[list[Any], dict[str, Any]]:
    """Allocate and touch raw CUDA storage while deliberately creating no optimizer."""

    if requested_bytes < 0 or chunk_bytes <= 0:
        raise ValueError("reserve bytes must be non-negative and chunk bytes positive")
    chunks: list[Any] = []
    remaining = requested_bytes
    while remaining:
        size = min(remaining, chunk_bytes)
        value = torch.empty((size,), dtype=torch.uint8, device=device)
        value.zero_()
        chunks.append(value)
        remaining -= size
    allocated = sum(int(value.numel() * value.element_size()) for value in chunks)
    return chunks, {
        "allocated_bytes": allocated,
        "allocation_count": len(chunks),
        "touched": bool(chunks),
        "is_optimizer": False,
    }


def _synthetic_batch(
    args: argparse.Namespace, *, vocabulary_size: int, torch: Any, device: Any
) -> Any:
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 17)
    input_ids = torch.randint(
        0,
        vocabulary_size,
        (args.batch_size, args.sequence_length),
        generator=generator,
        dtype=torch.long,
    ).to(device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    token_position = torch.arange(
        args.sequence_length,
        device=device,
        dtype=torch.long,
    ).view(1, args.sequence_length)
    offsets = torch.arange(args.teacher_top_k, device=device, dtype=torch.long)
    base = (input_ids + token_position * 131).remainder(vocabulary_size)
    teacher_indices = (base.unsqueeze(-1) + offsets).remainder(vocabulary_size)
    scores = torch.linspace(
        2.0,
        -2.0,
        args.teacher_top_k,
        dtype=torch.float32,
        device=device,
    )
    teacher_topk_logits = (
        scores.to(dtype=torch.bfloat16)
        .view(1, 1, -1)
        .expand(
            args.batch_size,
            args.sequence_length,
            -1,
        )
    )
    top_logsumexp = torch.logsumexp(scores / args.temperature, dim=-1)
    teacher_logsumexp = (top_logsumexp - math.log(0.9)).expand(
        args.batch_size,
        args.sequence_length,
    )
    teacher_tail_logprob = torch.full(
        (args.batch_size, args.sequence_length),
        math.log(0.1),
        dtype=torch.float32,
        device=device,
    )
    return SimpleNamespace(
        input_ids=input_ids,
        labels=input_ids.clone(),
        attention_mask=attention_mask,
        topk_indices=teacher_indices,
        topk_logits=teacher_topk_logits,
        teacher_logsumexp=teacher_logsumexp,
        teacher_tail_logprob=teacher_tail_logprob,
        temperature=args.temperature,
    )


def _set_transfer_enabled(modules: Sequence[Any], enabled: bool) -> None:
    for module in modules:
        module.set_transfer_enabled(enabled)


def _configure_dense_transfer_execution(
    transfer_modules: Sequence[Any],
    student_layer_indices: Sequence[int],
    *,
    execution_mode: str,
    outer_checkpoint_layer_indices: Sequence[int],
    inner_checkpoint_layer_indices: Sequence[int],
) -> dict[str, Any]:
    """Configure every dense transfer module and verify its observable state."""

    _dense_transfer_execution_contract(execution_mode)
    modules = tuple(transfer_modules)
    layers = tuple(student_layer_indices)
    if len(modules) != len(layers):
        raise RuntimeError("transfer module and student-layer mappings have different lengths")
    if tuple(sorted(set(layers))) != layers:
        raise RuntimeError("student transfer layer indices must be sorted and unique")
    outer = tuple(outer_checkpoint_layer_indices)
    inner = tuple(inner_checkpoint_layer_indices)
    if tuple(sorted(set(inner))) != inner:
        raise ValueError("inner dense transfer checkpoint indices must be sorted and unique")
    if set(outer).intersection(inner):
        raise RuntimeError("nested outer/inner dense transfer checkpointing is forbidden")
    unknown_inner = sorted(set(inner).difference(layers))
    if unknown_inner:
        raise RuntimeError(f"inner checkpoint layers do not have transfer modules: {unknown_inner}")

    selected = set(inner)
    actual_inner: list[int] = []
    actual_modes: set[str] = set()
    for layer, module in zip(layers, modules, strict=True):
        configure = getattr(module, "configure_transfer_execution", None)
        if not callable(configure):
            raise RuntimeError(f"dense transfer module at layer {layer} lacks execution controls")
        configure(
            execution_mode=execution_mode,
            checkpoint_token_branch=layer in selected,
        )
        transfer = getattr(module, "transfer_mlp", None)
        actual_mode = getattr(transfer, "execution_mode", None)
        actual_checkpoint = getattr(transfer, "checkpoint_token_branch", None)
        if actual_mode != execution_mode:
            raise RuntimeError(
                "dense transfer execution mismatch at layer "
                f"{layer}: expected={execution_mode!r}, actual={actual_mode!r}"
            )
        expected_checkpoint = layer in selected
        if not isinstance(actual_checkpoint, bool) or actual_checkpoint != expected_checkpoint:
            raise RuntimeError(
                "dense transfer checkpoint mismatch at layer "
                f"{layer}: expected={expected_checkpoint}, actual={actual_checkpoint!r}"
            )
        actual_modes.add(str(actual_mode))
        if actual_checkpoint:
            actual_inner.append(int(layer))

    actual_inner_tuple = tuple(actual_inner)
    if actual_inner_tuple != inner:
        raise RuntimeError(
            f"dense transfer checkpoint state mismatch: expected={inner}, "
            f"actual={actual_inner_tuple}"
        )
    if actual_modes != {execution_mode}:
        raise RuntimeError(
            f"dense transfer execution states are inconsistent: {sorted(actual_modes)}"
        )
    return {
        "module_count": len(modules),
        "actual_execution_modes": sorted(actual_modes),
        "actual_checkpoint_layer_indices": list(actual_inner_tuple),
        "outer_inner_disjoint": not bool(set(outer).intersection(actual_inner_tuple)),
        "all_modules_match_requested_state": True,
    }


def _configure_selective_activation_checkpointing(
    raw_model: Any,
    layer_indices: Sequence[int],
) -> None:
    """Enable checkpointing on exactly the selected Qwen decoder layers."""

    expected_layers = PRODUCTION_SHAPE["student_layers"]
    layers = tuple(getattr(getattr(raw_model, "model", None), "layers", ()))
    if len(layers) != expected_layers:
        raise RuntimeError(f"expected {expected_layers} Qwen decoder layers, found {len(layers)}")
    selected = tuple(layer_indices)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in selected):
        raise ValueError("activation checkpoint layer indices must be integers")
    if tuple(sorted(set(selected))) != selected:
        raise ValueError("activation checkpoint layer indices must be sorted and unique")
    if any(index < 0 or index >= expected_layers for index in selected):
        raise ValueError(f"activation checkpoint layer indices must be in [0, {expected_layers})")

    initialized = bool(
        getattr(raw_model, "_twen_selective_activation_checkpointing_initialized", False)
    )
    checkpoint_runtime_enabled = bool(
        getattr(raw_model, "_twen_selective_activation_checkpointing_enabled", False)
    )
    if selected and not checkpoint_runtime_enabled:
        # Let Transformers install its version-correct non-reentrant checkpoint
        # callable and one input-grad hook, then narrow the decoder-layer flags.
        raw_model.gradient_checkpointing_enable()
    elif not selected and (checkpoint_runtime_enabled or not initialized):
        raw_model.gradient_checkpointing_disable()
        disable_input_require_grads = getattr(raw_model, "disable_input_require_grads", None)
        if callable(disable_input_require_grads):
            disable_input_require_grads()
    raw_model._twen_selective_activation_checkpointing_initialized = True
    raw_model._twen_selective_activation_checkpointing_enabled = bool(selected)
    selected_set = set(selected)
    for index, layer in enumerate(layers):
        if not hasattr(layer, "gradient_checkpointing"):
            raise RuntimeError(f"Qwen decoder layer {index} lacks gradient checkpoint support")
        layer.gradient_checkpointing = index in selected_set

    actual = tuple(
        index
        for index, layer in enumerate(layers)
        if bool(getattr(layer, "gradient_checkpointing", False))
    )
    if actual != selected:
        raise RuntimeError(
            f"selective activation checkpointing mismatch: expected={selected}, actual={actual}"
        )


def _gradient_health(model: Any, *, torch: Any) -> dict[str, Any]:
    present = 0
    missing: list[str] = []
    nonfinite: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        present += 1
        if not bool(torch.isfinite(parameter.grad).all().item()):
            nonfinite.append(name)
    return {
        "finite": not nonfinite and present > 0,
        "present_tensors": present,
        "missing_tensors": len(missing),
        "missing_names": missing[:10],
        "nonfinite_tensors": len(nonfinite),
        "nonfinite_names": nonfinite[:10],
    }


def _event(*, torch: Any) -> Any:
    return torch.cuda.Event(enable_timing=True)


def _resolve_nvidia_smi(explicit: str | None) -> Path:
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise FileNotFoundError(f"nvidia-smi executable is absent: {candidate}")
        return candidate
    discovered = shutil.which("nvidia-smi")
    if discovered:
        return Path(discovered).resolve()
    wsl_fallback = Path("/usr/lib/wsl/lib/nvidia-smi")
    if wsl_fallback.is_file() and os.access(wsl_fallback, os.X_OK):
        return wsl_fallback
    raise FileNotFoundError("nvidia-smi was not found in PATH or at /usr/lib/wsl/lib/nvidia-smi")


@contextmanager
def _gpu_telemetry_sampler(args: argparse.Namespace) -> Iterator[dict[str, Any] | None]:
    """Sample read-only GPU telemetry and always reap the looping child process."""

    if not args.gpu_telemetry_output:
        yield None
        return

    output = Path(args.gpu_telemetry_output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    nonce = f"{os.getpid()}-{time.monotonic_ns()}"
    partial = output.with_name(f".{output.name}.{nonce}.partial")
    stderr_path = output.with_name(f".{output.name}.{nonce}.stderr")
    nvidia_smi = _resolve_nvidia_smi(args.nvidia_smi)
    query = (
        "timestamp,power.draw,power.limit,utilization.gpu,utilization.memory,"
        "clocks.sm,memory.used,memory.free,temperature.gpu"
    )
    command = [
        str(nvidia_smi),
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
        f"--loop-ms={args.gpu_telemetry_interval_ms}",
    ]
    report: dict[str, Any] = {
        "enabled": True,
        "output": str(output),
        "interval_ms": args.gpu_telemetry_interval_ms,
        "read_only": True,
        "columns": list(GPU_TELEMETRY_COLUMNS),
        "executable": str(nvidia_smi),
        "command": command,
        "command_uses_shell": False,
        "sampler_pid": None,
        "sample_count": 0,
        "cleanup": None,
    }
    process: subprocess.Popen[str] | None = None
    terminate_sent = False
    kill_sent = False
    stderr_text = ""
    try:
        with (
            partial.open("w", encoding="utf-8", newline="") as output_handle,
            stderr_path.open("w", encoding="utf-8") as stderr_handle,
        ):
            output_handle.write(",".join(GPU_TELEMETRY_COLUMNS) + "\n")
            output_handle.flush()
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=output_handle,
                stderr=stderr_handle,
                text=True,
                shell=False,
            )
            report["sampler_pid"] = process.pid
            # Require at least one sampling opportunity before any CUDA graph starts.
            time.sleep(max(0.35, args.gpu_telemetry_interval_ms / 1000.0 * 1.5))
            if process.poll() is not None:
                stderr_handle.flush()
                stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
                raise RuntimeError(
                    f"nvidia-smi telemetry sampler exited early with {process.returncode}: "
                    f"{stderr_text.strip()}"
                )
            yield report
            if process.poll() is not None:
                stderr_handle.flush()
                stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
                raise RuntimeError(
                    f"nvidia-smi telemetry sampler exited during the graph with "
                    f"{process.returncode}: {stderr_text.strip()}"
                )
    finally:
        if process is not None and process.poll() is None:
            terminate_sent = True
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                kill_sent = True
                process.kill()
                process.wait(timeout=5)
        if stderr_path.is_file() and not stderr_text:
            stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        if partial.is_file():
            # The CSV is useful OOM evidence too, so commit it even if the graph raises.
            with partial.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(partial, output)
            directory_fd = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            with output.open(encoding="utf-8") as handle:
                report["sample_count"] = max(sum(1 for _ in handle) - 1, 0)
        report["cleanup"] = {
            "pid": process.pid if process is not None else None,
            "terminate_sent": terminate_sent,
            "kill_sent": kill_sent,
            "returncode": process.returncode if process is not None else None,
            "reaped": bool(process is None or process.poll() is not None),
        }
        if stderr_text.strip():
            report["stderr"] = stderr_text.strip()
        stderr_path.unlink(missing_ok=True)


@contextmanager
def _cuda_profiler_api_range(*, enabled: bool, torch: Any) -> Iterator[None]:
    """Expose one exact GPU iteration to Nsight Systems capture-range mode."""

    if not enabled:
        yield
        return
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    try:
        yield
    finally:
        cudart.cudaProfilerStop()


def _execute_timed_graph(
    args: argparse.Namespace,
    *,
    built: Any,
    train_model: Any,
    teacher: Any,
    batch: Any,
    torch: Any,
    device: Any,
    cuda_profiler_api: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    """Execute only the GPU graph whose events define benchmark throughput."""

    from twen.training.engine import _hidden_alignment_loss

    iteration_start = _event(torch=torch)
    anchor_end = _event(torch=torch)
    student_end = _event(torch=torch)
    teacher_end = _event(torch=torch)
    backward_end = _event(torch=torch)
    with _cuda_profiler_api_range(enabled=cuda_profiler_api, torch=torch):
        iteration_start.record()
        wall_started = time.perf_counter()

        with torch.profiler.record_function("twen/benchmark/anchor"):
            _set_transfer_enabled(built.transfer_modules, False)
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    anchor_hidden_states = train_model(
                        input_ids=batch.input_ids,
                        attention_mask=batch.attention_mask,
                        anchor_only=True,
                    )["anchor_hidden_states"]
            finally:
                _set_transfer_enabled(built.transfer_modules, True)
            anchor_end.record()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.profiler.record_function("twen/benchmark/student"):
                outputs = train_model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    labels=batch.labels,
                    teacher_indices=batch.topk_indices,
                    teacher_topk_logits=batch.topk_logits,
                    teacher_logsumexp=batch.teacher_logsumexp,
                    teacher_tail_logprob=batch.teacher_tail_logprob,
                    temperature=batch.temperature,
                    anchor_hidden_states=anchor_hidden_states,
                    output_hidden_states=args.hidden_alignment,
                )
                ntp = outputs["ntp"]
                teacher_kd = outputs["teacher_kd"]
                anchor_kl = outputs["anchor_kl"]
                if anchor_kl is None:
                    raise RuntimeError("streaming graph omitted the requested anchor KL")
                loss = (
                    args.ntp_weight * ntp
                    + args.teacher_kd_weight * teacher_kd
                    + args.anchor_kl_weight * anchor_kl
                )
                mtp = outputs["mtp"]
                if args.mtp_loss_weight > 0:
                    if mtp is None:
                        raise RuntimeError("streaming graph omitted the requested native MTP loss")
                    loss = loss + args.mtp_loss_weight * mtp
                elif mtp is not None:
                    raise RuntimeError("streaming graph unexpectedly executed native MTP")
                student_end.record()

            teacher_outputs = None
            hidden_alignment = None
            if args.hidden_alignment:
                with torch.profiler.record_function("twen/benchmark/teacher"):
                    with torch.no_grad():
                        teacher_outputs = teacher(
                            input_ids=batch.input_ids,
                            attention_mask=batch.attention_mask,
                            use_cache=False,
                            output_hidden_states=True,
                        )
                    hidden_alignment = _hidden_alignment_loss(
                        outputs["hidden_states"],
                        teacher_outputs.hidden_states,
                        built.transfer_modules,
                        tuple(range(PRODUCTION_SHAPE["student_layers"])),
                        built.student_layer_indices,
                        batch.attention_mask,
                    )
                    loss = loss + args.hidden_alignment_weight * hidden_alignment
            teacher_end.record()

        with torch.profiler.record_function("twen/benchmark/backward"):
            loss.backward()
            backward_end.record()
            backward_end.synchronize()
    wall_seconds = time.perf_counter() - wall_started

    timing = {
        "anchor_forward_seconds": iteration_start.elapsed_time(anchor_end) / 1000.0,
        "student_streaming_forward_seconds": anchor_end.elapsed_time(student_end) / 1000.0,
        "teacher_hidden_alignment_seconds": student_end.elapsed_time(teacher_end) / 1000.0,
        "backward_seconds": teacher_end.elapsed_time(backward_end) / 1000.0,
        "total_gpu_seconds": iteration_start.elapsed_time(backward_end) / 1000.0,
        "total_wall_seconds": wall_seconds,
    }
    timing["forward_seconds"] = (
        timing["anchor_forward_seconds"]
        + timing["student_streaming_forward_seconds"]
        + timing["teacher_hidden_alignment_seconds"]
    )
    losses = {
        "total": float(loss.detach()),
        "ntp": float(ntp.detach()),
        "teacher_kd": float(teacher_kd.detach()),
        "anchor_kl": float(anchor_kl.detach()),
    }
    if mtp is not None:
        losses["mtp"] = float(mtp.detach())
    if hidden_alignment is not None:
        losses["hidden_alignment"] = float(hidden_alignment.detach())
    del teacher_outputs, outputs, anchor_hidden_states, hidden_alignment, mtp
    del anchor_kl, teacher_kd, ntp
    del loss
    return timing, losses


def _execute_iteration(
    args: argparse.Namespace,
    *,
    built: Any,
    train_model: Any,
    teacher: Any,
    batch: Any,
    torch: Any,
    device: Any,
    teacher_offload: Any | None = None,
    cuda_profiler_api: bool = False,
) -> dict[str, Any]:
    offload_enabled = teacher_offload is not None
    if offload_enabled != bool(args.teacher_cpu_offload):
        raise RuntimeError(
            "teacher CPU offload manager does not match --teacher-cpu-offload: "
            f"enabled={args.teacher_cpu_offload}, manager={offload_enabled}"
        )
    if offload_enabled and teacher_offload.is_staged:
        raise RuntimeError("teacher CPU offload manager entered an iteration already staged")

    built.model.zero_grad(set_to_none=True)
    gc.collect()
    stage_transition = None
    restore_transition = None
    if offload_enabled and args.hidden_alignment:
        stage_transition = teacher_offload.stage()
    try:
        torch.cuda.synchronize(device)
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))
        free_before, total_memory = map(int, torch.cuda.mem_get_info(device))
        torch.cuda.reset_peak_memory_stats(device)
        timing, losses = _execute_timed_graph(
            args,
            built=built,
            train_model=train_model,
            teacher=teacher,
            batch=batch,
            torch=torch,
            device=device,
            cuda_profiler_api=cuda_profiler_api,
        )
    finally:
        if offload_enabled and teacher_offload.is_staged:
            restore_transition = teacher_offload.restore()

    if offload_enabled and teacher_offload.is_staged:
        raise RuntimeError("teacher CPU offload manager failed to restore split residency")
    logical_tokens = args.batch_size * args.sequence_length
    throughput = {
        "logical_tokens_per_second_gpu": logical_tokens / timing["total_gpu_seconds"],
        "logical_tokens_per_second_wall": logical_tokens / timing["total_wall_seconds"],
    }
    transition_seconds = 0.0
    if offload_enabled:
        stage_seconds = float(stage_transition.seconds) if stage_transition is not None else 0.0
        restore_seconds = (
            float(restore_transition.seconds) if restore_transition is not None else 0.0
        )
        transition_seconds = stage_seconds + restore_seconds
        timing["teacher_cpu_offload_stage_seconds"] = stage_seconds
        timing["teacher_cpu_offload_restore_seconds"] = restore_seconds
        timing["total_wall_seconds_including_teacher_transitions"] = (
            timing["total_wall_seconds"] + transition_seconds
        )
        throughput["logical_tokens_per_second_wall_including_teacher_transitions"] = (
            logical_tokens / timing["total_wall_seconds_including_teacher_transitions"]
        )

    loss_finite = all(math.isfinite(value) for value in losses.values())
    gradients = _gradient_health(built.model, torch=torch)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    free_after, _ = map(int, torch.cuda.mem_get_info(device))
    external_or_driver_used = max(total_memory - free_before - baseline_reserved, 0)
    memory = {
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_delta_allocated_bytes": max(peak_allocated - baseline_allocated, 0),
        "free_before_bytes": free_before,
        "free_after_bytes": free_after,
        "device_total_bytes": total_memory,
        "external_or_driver_used_before_bytes": external_or_driver_used,
        "theoretical_headroom_from_peak_allocated_bytes": max(total_memory - peak_allocated, 0),
        "estimated_free_at_peak_reserved_bytes": max(
            total_memory - external_or_driver_used - peak_reserved, 0
        ),
    }
    result = {
        "ok": bool(loss_finite and gradients["finite"] and not gradients["missing_tensors"]),
        "timing": timing,
        "throughput": throughput,
        "memory": memory,
        "losses": losses,
        "loss_finite": loss_finite,
        "gradients": gradients,
    }
    if offload_enabled:
        result["teacher_cpu_offload"] = {
            "enabled": True,
            "staged_for_hidden_alignment": stage_transition is not None,
            "transitions_in_gpu_iteration_timing": False,
            "parameter_bytes": int(teacher_offload.parameter_bytes),
            "buffer_bytes": int(teacher_offload.buffer_bytes),
            "staged_bytes": int(teacher_offload.staged_bytes),
            "stage": (
                dataclasses.asdict(stage_transition) if stage_transition is not None else None
            ),
            "restore": (
                dataclasses.asdict(restore_transition) if restore_transition is not None else None
            ),
            "transition_seconds": transition_seconds,
            "memory_before_gpu_iteration": {
                "allocated_bytes": baseline_allocated,
                "reserved_bytes": baseline_reserved,
                "free_bytes": free_before,
                "teacher_staged": stage_transition is not None,
            },
            "memory_after_restore": {
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "free_bytes": free_after,
                "teacher_staged": False,
            },
        }
    return result


def _profile_metric(event: Any, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def _jsonable_profile_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable_profile_value(item) for key, item in value.items()}
    if isinstance(value, Sequence):
        return [_jsonable_profile_value(item) for item in value]
    return str(value)


def _write_profile_summaries(profiler: Any, trace_path: Path) -> dict[str, str]:
    averages = profiler.key_averages(group_by_input_shape=True)
    summary_text_path = trace_path.with_suffix(".summary.txt")
    summary_json_path = trace_path.with_suffix(".summary.json")
    table = averages.table(
        sort_by="self_cuda_time_total",
        row_limit=PROFILE_SUMMARY_ROW_LIMIT,
    )
    summary_text_path.write_text(
        "# sort_by=self_cuda_time_total "
        f"row_limit={PROFILE_SUMMARY_ROW_LIMIT} group_by_input_shape=true\n{table}\n",
        encoding="utf-8",
    )
    rows = [
        {
            "name": str(getattr(event, "key", getattr(event, "name", "<unknown>"))),
            "calls": int(getattr(event, "count", 0)),
            "cpu": _profile_metric(event, "cpu_time_total"),
            "cuda": _profile_metric(event, "device_time_total", "cuda_time_total"),
            "self_cuda": _profile_metric(
                event,
                "self_device_time_total",
                "self_cuda_time_total",
            ),
            "input_shapes": _jsonable_profile_value(getattr(event, "input_shapes", [])),
        }
        for event in averages
    ]
    rows.sort(key=lambda row: float(row["self_cuda"]), reverse=True)
    rows = rows[:PROFILE_SUMMARY_ROW_LIMIT]
    summary_json_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sort_by": "self_cuda_time_total",
                "time_unit": "microseconds",
                "row_limit": PROFILE_SUMMARY_ROW_LIMIT,
                "group_by_input_shape": True,
                "events": rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "summary_text": str(summary_text_path),
        "summary_json": str(summary_json_path),
    }


def _run_profiling_iteration(
    args: argparse.Namespace,
    *,
    built: Any,
    train_model: Any,
    teacher: Any,
    teacher_offload: Any | None = None,
    batch: Any,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    """Run one health-checked iteration that is never a benchmark sample."""

    if not args.torch_profile_trace and not args.cuda_profiler_api:
        raise ValueError("a profiling iteration requires at least one profiling option")
    trace_path = (
        Path(args.torch_profile_trace).expanduser().resolve() if args.torch_profile_trace else None
    )
    profile_artifacts = {
        "summary_text": None,
        "summary_json": None,
    }
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with torch.profiler.profile(
            activities=(
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            acc_events=True,
        ) as profiler:
            result = _execute_iteration(
                args,
                built=built,
                train_model=train_model,
                teacher=teacher,
                teacher_offload=teacher_offload,
                batch=batch,
                torch=torch,
                device=device,
                cuda_profiler_api=bool(args.cuda_profiler_api),
            )
        profiler.export_chrome_trace(str(trace_path))
        if not trace_path.is_file():
            raise RuntimeError(f"PyTorch profiler did not create {trace_path}")
        profile_artifacts = _write_profile_summaries(profiler, trace_path)
    else:
        result = _execute_iteration(
            args,
            built=built,
            train_model=train_model,
            teacher=teacher,
            teacher_offload=teacher_offload,
            batch=batch,
            torch=torch,
            device=device,
            cuda_profiler_api=True,
        )
    if not result["ok"]:
        raise RuntimeError(f"profiling graph health check failed: {result}")
    return {
        "separate_complete_iteration": True,
        "included_in_benchmark_samples": False,
        "torch_chrome_trace": str(trace_path) if trace_path is not None else None,
        "torch_profile_summary": profile_artifacts["summary_text"],
        "torch_profile_summary_json": profile_artifacts["summary_json"],
        "cuda_profiler_api": bool(args.cuda_profiler_api),
        "nsys_capture_range": "cudaProfilerApi" if args.cuda_profiler_api else None,
        "record_function_ranges": [
            "twen/benchmark/anchor",
            "twen/benchmark/student",
            "twen/benchmark/teacher",
            "twen/benchmark/backward",
        ],
        "iteration_health": {
            "ok": bool(result["ok"]),
            "loss_finite": bool(result["loss_finite"]),
            "gradients": result["gradients"],
        },
    }


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stddev": statistics.pstdev(values),
    }


def _summarize_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    timing_names = tuple(samples[0]["timing"])
    throughput_names = tuple(samples[0]["throughput"])
    return {
        "timing_seconds": {
            name: _summary([float(sample["timing"][name]) for sample in samples])
            for name in timing_names
        },
        "throughput": {
            name: _summary([float(sample["throughput"][name]) for sample in samples])
            for name in throughput_names
        },
        "memory_worst_case": {
            "peak_allocated_bytes": max(
                int(sample["memory"]["peak_allocated_bytes"]) for sample in samples
            ),
            "peak_reserved_bytes": max(
                int(sample["memory"]["peak_reserved_bytes"]) for sample in samples
            ),
            "minimum_free_after_bytes": min(
                int(sample["memory"]["free_after_bytes"]) for sample in samples
            ),
        },
    }


def _case_health(
    warmup_results: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    iterations = (*warmup_results, *samples)
    if not samples:
        raise ValueError("a benchmark case requires at least one measurement sample")
    present_counts = sorted({int(result["gradients"]["present_tensors"]) for result in iterations})
    return {
        "ok": all(bool(result["ok"]) for result in iterations),
        "warmup_iterations": len(warmup_results),
        "measurement_iterations": len(samples),
        "warmup_ok": all(bool(result["ok"]) for result in warmup_results),
        "measurements_ok": all(bool(result["ok"]) for result in samples),
        "loss_finite": all(bool(result["loss_finite"]) for result in iterations),
        "gradients_finite": all(bool(result["gradients"]["finite"]) for result in iterations),
        "maximum_missing_gradient_tensors": max(
            int(result["gradients"]["missing_tensors"]) for result in iterations
        ),
        "maximum_nonfinite_gradient_tensors": max(
            int(result["gradients"]["nonfinite_tensors"]) for result in iterations
        ),
        "present_gradient_tensor_counts": present_counts,
        "consistent_present_gradient_tensor_count": len(present_counts) == 1,
    }


def _run_activation_checkpoint_case(
    args: argparse.Namespace,
    *,
    case_number: int,
    case_count: int,
    checkpoint_layer_count: int,
    checkpoint_layer_indices: Sequence[int],
    built: Any,
    train_model: Any,
    teacher: Any,
    teacher_offload: Any | None = None,
    batch: Any,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    """Run one independently prepared case against already resident model objects."""

    if case_number <= 0 or case_count <= 0 or case_number > case_count:
        raise ValueError("invalid activation checkpoint case position")
    if (
        isinstance(checkpoint_layer_count, bool)
        or not isinstance(checkpoint_layer_count, int)
        or checkpoint_layer_count != len(checkpoint_layer_indices)
    ):
        raise ValueError("checkpoint layer count must match the selected layer indices")
    dense_transfer_case = _dense_transfer_case_contract(
        args,
        outer_checkpoint_layer_indices=checkpoint_layer_indices,
    )
    print(
        f"case {case_number}/{case_count}: activation checkpoint layers="
        f"{checkpoint_layer_count} indices={list(checkpoint_layer_indices)}; "
        f"dense transfer execution={args.dense_transfer_execution} inner checkpoints="
        f"{dense_transfer_case['dense_transfer_checkpoint_layer_indices']}",
        file=sys.stderr,
        flush=True,
    )
    _configure_selective_activation_checkpointing(built.model, checkpoint_layer_indices)
    dense_transfer_state = _configure_dense_transfer_execution(
        built.transfer_modules,
        built.student_layer_indices,
        execution_mode=args.dense_transfer_execution,
        outer_checkpoint_layer_indices=checkpoint_layer_indices,
        inner_checkpoint_layer_indices=dense_transfer_case[
            "dense_transfer_checkpoint_layer_indices"
        ],
    )
    built.model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    memory_before_case = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "free_bytes": int(torch.cuda.mem_get_info(device)[0]),
    }

    warmup_results: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    try:
        for iteration in range(args.warmup):
            print(
                f"case {case_number}/{case_count} warmup {iteration + 1}/{args.warmup}...",
                file=sys.stderr,
                flush=True,
            )
            warmup_result = _execute_iteration(
                args,
                built=built,
                train_model=train_model,
                teacher=teacher,
                teacher_offload=teacher_offload,
                batch=batch,
                torch=torch,
                device=device,
            )
            warmup_results.append(warmup_result)
            if not warmup_result["ok"]:
                raise RuntimeError(f"warmup graph health check failed: {warmup_result}")

        for iteration in range(args.repeats):
            print(
                f"case {case_number}/{case_count} measurement {iteration + 1}/{args.repeats}...",
                file=sys.stderr,
                flush=True,
            )
            sample = _execute_iteration(
                args,
                built=built,
                train_model=train_model,
                teacher=teacher,
                teacher_offload=teacher_offload,
                batch=batch,
                torch=torch,
                device=device,
            )
            sample["repeat"] = iteration + 1
            samples.append(sample)
    finally:
        built.model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

    summary = _summarize_samples(samples)
    health = _case_health(warmup_results, samples)
    return {
        "case": case_number,
        "ok": health["ok"],
        "activation_checkpoint_layer_count": checkpoint_layer_count,
        "activation_checkpoint_layer_indices": list(checkpoint_layer_indices),
        **dense_transfer_case,
        "dense_transfer_actual_state": dense_transfer_state,
        "memory_before_case": memory_before_case,
        "memory_after_cleanup": {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "free_bytes": int(torch.cuda.mem_get_info(device)[0]),
        },
        "samples": samples,
        "summary": summary,
        "health": health,
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _cuda_toolchain_report(cuda_home: Path) -> dict[str, Any]:
    checker = Path(__file__).with_name("check_cuda_toolchain.py")
    process = subprocess.run(
        [sys.executable, str(checker), "--cuda-home", str(cuda_home)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    try:
        report = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"CUDA toolchain checker returned invalid JSON: {process.stdout!r}"
        ) from error
    if process.returncode != 0 or not report.get("ok"):
        raise RuntimeError(f"incoherent CUDA compiler/headers: {report}")
    return report


def _run_benchmark(args: argparse.Namespace, shape_report: Mapping[str, Any]) -> dict[str, Any]:
    cuda_home = Path(args.cuda_home).expanduser().resolve()
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        raise FileNotFoundError(f"CUDA compiler is absent: {nvcc}")
    # TileLang resolves CUDA_HOME exactly once at import time.  Set every
    # compiler selector before importing torch/transformers/FLA so its JIT
    # cannot combine the cu130 runtime wheel with independently upgraded
    # nvidia-cuda-nvcc/CCCL wheels from an optional serving environment.
    os.environ["CUDA_HOME"] = str(cuda_home)
    os.environ["CUDA_PATH"] = str(cuda_home)
    os.environ["CUDACXX"] = str(nvcc)
    os.environ["PATH"] = f"{cuda_home / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ["FLA_TILELANG"] = "0" if args.fla_backend == "triton" else "1"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    toolchain_report = _cuda_toolchain_report(cuda_home)

    import torch

    from twen.model_loading import freeze_module, load_qwen35_text_model
    from twen.training.builder import build_transfer_model
    from twen.training.streaming import StreamingLossCausalLM
    from twen.training.teacher_offload import TeacherCPUOffloadManager

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --dry-run for metadata validation")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    if "5090" not in gpu_name and not args.allow_other_gpu:
        raise RuntimeError(
            f"expected an RTX 5090, found {gpu_name!r}; pass --allow-other-gpu to override"
        )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    contract = _base_contract(args, shape_report)
    checkpoint_cases = _activation_checkpoint_layer_cases(args)
    sweep_enabled = args.activation_checkpoint_layer_counts is not None
    contract["dry_run"] = False
    contract["environment"] = {
        "created_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
        "git_commit": _git_commit(),
        "cuda_home": str(cuda_home),
        "cuda_compiler": str(nvcc),
        "cuda_toolchain": toolchain_report["toolchain"],
        "fla_backend": args.fla_backend,
        "fla_tilelang_env": os.environ["FLA_TILELANG"],
    }
    initial_free, device_total = map(int, torch.cuda.mem_get_info(device))
    contract["memory_at_start"] = {
        "free_bytes": initial_free,
        "device_total_bytes": device_total,
        "non_twen_resident_bytes": max(device_total - initial_free, 0),
    }
    torch.cuda.reset_peak_memory_stats(device)

    print("loading frozen 9B text teacher/donor...", file=sys.stderr, flush=True)
    load_started = time.perf_counter()
    teacher_load_device = "cpu" if args.teacher_cpu_offload else str(device)
    teacher = load_qwen35_text_model(
        args.teacher,
        dtype=torch.bfloat16,
        device=teacher_load_device,
    )
    freeze_module(teacher)
    torch.cuda.synchronize(device)
    contract["teacher_load_seconds"] = time.perf_counter() - load_started
    contract["memory_after_teacher"] = {
        "teacher_load_device": teacher_load_device,
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }

    temporary_parent = Path(args.temporary_root)
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="full-dense-graph-", dir=temporary_parent) as raw_temp:
        print("creating temporary sequential maps and small A/B initialization...", file=sys.stderr)
        artifact_started = time.perf_counter()
        artifacts = _write_temporary_calibration(
            Path(raw_temp),
            student_layers=PRODUCTION_SHAPE["student_layers"],
            student_hidden=PRODUCTION_SHAPE["student_hidden_size"],
            donor_hidden=PRODUCTION_SHAPE["donor_hidden_size"],
            donor_intermediate=PRODUCTION_SHAPE["donor_intermediate_size"],
            experts=PRODUCTION_SHAPE["experts"],
            init_std=args.adapter_init_std,
            seed=args.seed,
            torch=torch,
        )
        contract["temporary_initialization"]["artifact_generation_seconds"] = (
            time.perf_counter() - artifact_started
        )
        print("loading 0.8B student and injecting all 24 shared donor branches...", file=sys.stderr)
        build_started = time.perf_counter()
        built = build_transfer_model(
            _builder_config(args, artifacts),
            device=str(device),
            dtype=torch.bfloat16,
            donor_text_model=teacher,
        )
        _set_branch_scale(built.transfer_modules, args.branch_scale, torch=torch)
        torch.cuda.synchronize(device)
        contract["student_dense_build_seconds"] = time.perf_counter() - build_started
    gc.collect()

    if len(built.transfer_modules) != PRODUCTION_SHAPE["student_layers"]:
        raise RuntimeError(
            f"expected 24 active transfer modules, found {len(built.transfer_modules)}"
        )
    aliases = _alias_report(built, teacher)
    contract["shared_aliases"] = aliases
    if not aliases["all_donor_projection_aliases_exact"]:
        raise RuntimeError("donor/teacher projection aliases are not exact")
    if not aliases["student_embedding_lm_head_storage_alias"]:
        raise RuntimeError("student embedding and LM head are not tied")

    teacher_offload = (
        TeacherCPUOffloadManager.from_transfer_modules(
            teacher,
            built.transfer_modules,
            target_device=device,
        )
        if args.teacher_cpu_offload
        else None
    )
    if teacher_offload is not None and teacher_offload.is_staged:
        raise RuntimeError("teacher CPU offload manager must start in split residency")
    contract["teacher_cpu_offload"].update(
        {
            "manager_created": teacher_offload is not None,
            "parameter_bytes": (
                int(teacher_offload.parameter_bytes) if teacher_offload is not None else 0
            ),
            "buffer_bytes": (
                int(teacher_offload.buffer_bytes) if teacher_offload is not None else 0
            ),
            "staged_bytes": (
                int(teacher_offload.staged_bytes) if teacher_offload is not None else 0
            ),
            "initial_is_staged": (
                bool(teacher_offload.is_staged) if teacher_offload is not None else None
            ),
            "resident_shared_projection_bytes": int(
                aliases["unique_shared_projection_storage_bytes"]
            ),
            "resident_shared_projection_aliases": int(aliases["storage_aliases"]),
        }
    )

    raw_model = built.model
    raw_model.config.use_cache = False
    mtp_enabled = args.mtp_loss_weight > 0
    if (built.mtp is not None) != mtp_enabled:
        raise RuntimeError(
            "native MTP build state does not match --mtp-loss-weight: "
            f"enabled={mtp_enabled}, loaded={built.mtp is not None}"
        )
    mtp_parameters = (
        sum(int(parameter.numel()) for parameter in built.mtp.parameters())
        if built.mtp is not None
        else 0
    )
    mtp_trainable_parameters = (
        sum(
            int(parameter.numel())
            for parameter in built.mtp.parameters()
            if parameter.requires_grad
        )
        if built.mtp is not None
        else 0
    )
    if mtp_trainable_parameters:
        raise RuntimeError("native MTP source parameters must remain frozen")
    mtp_attention_implementation = (
        getattr(built.mtp.config, "_attn_implementation", None) if built.mtp is not None else None
    )
    if mtp_enabled and mtp_attention_implementation != "sdpa":
        raise RuntimeError(
            "production native MTP benchmark requires SDPA attention; "
            f"got {mtp_attention_implementation!r}"
        )
    contract["mtp"].update(
        {
            "loaded": built.mtp is not None,
            "parameters": mtp_parameters,
            "trainable_parameters": mtp_trainable_parameters,
            "attention_implementation": mtp_attention_implementation,
        }
    )
    train_model = StreamingLossCausalLM(
        raw_model,
        chunk_tokens=args.loss_chunk_tokens,
        checkpoint_chunks=args.loss_checkpoint_chunks,
        compile_loss=args.compile_streaming_loss,
        mtp=built.mtp,
    ).train()
    teacher.eval()
    contract["memory_after_graph_build"] = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "construction_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "construction_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "teacher_cpu_offload_split_residency": teacher_offload is not None,
        "teacher_cpu_offload_staged_bytes": 0,
        "teacher_cpu_shadow_bytes": (
            int(teacher_offload.staged_bytes) if teacher_offload is not None else 0
        ),
    }

    print(
        f"allocating and touching {args.optimizer_state_reserve_gib:.3f} GiB raw CUDA reserve...",
        file=sys.stderr,
        flush=True,
    )
    reserve_chunks, reserve_report = _allocate_optimizer_state_reserve(
        round(args.optimizer_state_reserve_gib * GIB),
        chunk_bytes=args.reserve_chunk_mib * MIB,
        torch=torch,
        device=device,
    )
    torch.cuda.synchronize(device)
    contract["optimizer_state_reserve"].update(reserve_report)
    contract["optimizer_state_reserve"]["resident_during_all_iterations"] = True

    batch = _synthetic_batch(
        args,
        vocabulary_size=PRODUCTION_SHAPE["vocabulary_size"],
        torch=torch,
        device=device,
    )
    contract["memory_before_iterations"] = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "free_bytes": int(torch.cuda.mem_get_info(device)[0]),
    }

    case_results = []
    with _gpu_telemetry_sampler(args) as gpu_telemetry:
        for case_number, (checkpoint_layer_count, checkpoint_layer_indices) in enumerate(
            checkpoint_cases,
            start=1,
        ):
            case_results.append(
                _run_activation_checkpoint_case(
                    args,
                    case_number=case_number,
                    case_count=len(checkpoint_cases),
                    checkpoint_layer_count=checkpoint_layer_count,
                    checkpoint_layer_indices=checkpoint_layer_indices,
                    built=built,
                    train_model=train_model,
                    teacher=teacher,
                    teacher_offload=teacher_offload,
                    batch=batch,
                    torch=torch,
                    device=device,
                )
            )
    if gpu_telemetry is not None:
        contract["gpu_telemetry"].update(gpu_telemetry)

    profiling = None
    if args.torch_profile_trace or args.cuda_profiler_api:
        print("separate profiling iteration...", file=sys.stderr, flush=True)
        profiling = _run_profiling_iteration(
            args,
            built=built,
            train_model=train_model,
            teacher=teacher,
            teacher_offload=teacher_offload,
            batch=batch,
            torch=torch,
            device=device,
        )

    if sweep_enabled:
        contract["cases"] = case_results
        contract["sweep"]["completed_case_count"] = len(case_results)
        contract["sweep"]["all_cases_healthy"] = all(
            bool(case["health"]["ok"]) for case in case_results
        )
    else:
        contract["samples"] = case_results[0]["samples"]
        contract["summary"] = case_results[0]["summary"]
        contract["dense_transfer_actual_state"] = case_results[0]["dense_transfer_actual_state"]
    if profiling is not None:
        contract["profiling"] = profiling
    contract["ok"] = bool(
        aliases["all_donor_projection_aliases_exact"]
        and aliases["student_embedding_lm_head_storage_alias"]
        and reserve_report["allocated_bytes"] == round(args.optimizer_state_reserve_gib * GIB)
        and all(bool(case["health"]["ok"]) for case in case_results)
    )
    contract["production_acceptance"] = bool(
        contract["production_shape"]
        and contract["ok"]
        and contract["experimental_execution"]["production_enabled"]
    )
    # Keep both the raw reserve and all model objects live until every peak and
    # health metric above has been sampled.  This assertion prevents an
    # accidental refactor from silently dropping the simulated moment storage.
    contract["optimizer_state_reserve"]["live_chunk_count_at_completion"] = len(reserve_chunks)
    return contract


def _render_and_write(result: Mapping[str, Any], output: str | None) -> None:
    rendered = (
        json.dumps(
            dict(result),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    if output:
        from twen.utils import atomic_write_text

        atomic_write_text(output, rendered)
    print(rendered, end="")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_args(args)
        shape_report = _checkpoint_shape_report(
            args.backbone,
            args.teacher,
            require_mtp=args.mtp_loss_weight > 0,
        )
        if not shape_report["ok"]:
            raise ValueError(f"checkpoint shape mismatch: {shape_report['mismatches']}")
        if args.dry_run:
            result = _base_contract(args, shape_report)
            result["ok"] = True
            result["production_acceptance"] = bool(
                result["production_shape"]
                and result["experimental_execution"]["production_enabled"]
            )
        else:
            result = _run_benchmark(args, shape_report)
        _render_and_write(result, args.output)
        return 0 if result["ok"] else 2
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        failure: dict[str, Any] = {
            "benchmark": "base_dense_24_layer_full_graph",
            "ok": False,
            "dry_run": bool(args.dry_run),
            "no_optimizer_created": True,
            "no_optimizer_steps": True,
            "optimizer_step_calls": 0,
            "experimental_execution": _dense_transfer_execution_contract(
                args.dense_transfer_execution
            ),
            "batch": {
                "batch_size": args.batch_size,
                "sequence_length": args.sequence_length,
                "logical_tokens": args.batch_size * args.sequence_length,
            },
            "runtime": {
                "activation_checkpoint_layer_count": (
                    args.activation_checkpoint_layer_count
                    if args.activation_checkpoint_layer_count is not None
                    else (
                        PRODUCTION_SHAPE["student_layers"] if args.activation_checkpointing else 0
                    )
                ),
                "warmup": args.warmup,
                "repeats": args.repeats,
                "teacher_cpu_offload": bool(args.teacher_cpu_offload),
                "dense_transfer_execution": args.dense_transfer_execution,
                "dense_transfer_checkpoint_layer_count_requested": (
                    args.dense_transfer_checkpoint_layer_count
                ),
            },
            "graph": {
                "online_hidden_alignment": bool(args.hidden_alignment),
                "parameter_update": False,
            },
            "loss_weights": {"mtp": args.mtp_loss_weight},
            "teacher_cpu_offload": {"enabled": bool(args.teacher_cpu_offload)},
            "fla_backend": args.fla_backend,
            "fla_tilelang_env": "0" if args.fla_backend == "triton" else "1",
            "optimizer_state_reserve": {
                "requested_gib": args.optimizer_state_reserve_gib,
                "requested_bytes": round(args.optimizer_state_reserve_gib * GIB),
                "is_optimizer": False,
                "allocation_status": "unknown_after_failure",
            },
            "gpu_telemetry": {
                "enabled": bool(args.gpu_telemetry_output),
                "output": (
                    str(Path(args.gpu_telemetry_output).expanduser().resolve())
                    if args.gpu_telemetry_output
                    else None
                ),
                "interval_ms": args.gpu_telemetry_interval_ms,
                "read_only": True,
            },
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.is_initialized():
                device = torch.device(args.device)
                failure["cuda_failure_memory"] = {
                    "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                    "free_bytes": int(torch.cuda.mem_get_info(device)[0]),
                }
        except Exception:
            pass
        _render_and_write(failure, args.output)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
