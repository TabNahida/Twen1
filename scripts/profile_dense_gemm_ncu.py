#!/usr/bin/env python3
"""Run one production-shaped BF16 GEMM for targeted Nsight Compute capture.

The script deliberately uses explicit ``torch.mm`` calls matching the dense
donor gate/up forward, input-gradient, or weight-gradient orientations.  It
does not construct an optimizer, run a model, or update any parameter.  Warmup
executes before ``cudaProfilerStart`` so NCU can capture exactly one steady
kernel with ``--profile-from-start off``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import UTC, datetime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operation",
        choices=("forward", "input-grad", "weight-grad"),
        default="forward",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--donor-hidden-size", type=int, default=4096)
    parser.add_argument("--donor-intermediate-size", type=int, default=12288)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate(args: argparse.Namespace) -> None:
    for name in (
        "batch_size",
        "sequence_length",
        "donor_hidden_size",
        "donor_intermediate_size",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(args.warmup, bool) or not isinstance(args.warmup, int) or args.warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if not args.device.startswith("cuda") and not args.dry_run:
        raise ValueError("a CUDA device is required unless --dry-run is used")


def _contract(args: argparse.Namespace) -> dict[str, object]:
    tokens = args.batch_size * args.sequence_length
    h = args.donor_hidden_size
    m = args.donor_intermediate_size
    shapes = {
        "forward": {"left": [tokens, h], "right": [h, m], "output": [tokens, m]},
        "input-grad": {
            "left": [tokens, m],
            "right": [m, h],
            "output": [tokens, h],
        },
        "weight-grad": {
            "left": [m, tokens],
            "right": [tokens, h],
            "output": [m, h],
        },
    }
    selected = shapes[args.operation]
    flop = 2 * selected["output"][0] * selected["output"][1] * selected["left"][1]
    return {
        "schema_version": 1,
        "kind": "twen_targeted_dense_gemm_ncu_probe",
        "operation": args.operation,
        "dtype": "bfloat16",
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "logical_tokens": tokens,
        "shapes": selected,
        "nominal_flop": flop,
        "no_optimizer_created": True,
        "optimizer_steps": 0,
        "parameter_updates": False,
        "cuda_profiler_api_range": True,
        "dry_run": bool(args.dry_run),
    }


def _run(args: argparse.Namespace, contract: dict[str, object]) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --dry-run for the shape contract")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    tokens = args.batch_size * args.sequence_length
    h = args.donor_hidden_size
    m = args.donor_intermediate_size

    if args.operation == "forward":
        left = torch.randn((tokens, h), device=device, dtype=torch.bfloat16)
        stored_weight = torch.randn((m, h), device=device, dtype=torch.bfloat16)

        def operation():
            return torch.mm(left, stored_weight.t())

    elif args.operation == "input-grad":
        left = torch.randn((tokens, m), device=device, dtype=torch.bfloat16)
        stored_weight = torch.randn((m, h), device=device, dtype=torch.bfloat16)

        def operation():
            return torch.mm(left, stored_weight)

    else:
        grad_output = torch.randn((tokens, m), device=device, dtype=torch.bfloat16)
        activations = torch.randn((tokens, h), device=device, dtype=torch.bfloat16)

        def operation():
            return torch.mm(grad_output.t(), activations)

    result = None
    for _ in range(args.warmup):
        result = operation()
    torch.cuda.synchronize(device)
    free_before, total = map(int, torch.cuda.mem_get_info(device))
    torch.cuda.reset_peak_memory_stats(device)
    started = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)
    torch.cuda.cudart().cudaProfilerStart()
    wall_started = time.perf_counter()
    started.record()
    result = operation()
    ended.record()
    ended.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    torch.cuda.cudart().cudaProfilerStop()
    if result is None or not bool(torch.isfinite(result).all().item()):
        raise RuntimeError("representative GEMM produced a non-finite result")
    gpu_seconds = started.elapsed_time(ended) / 1000.0
    nominal_flop = int(contract["nominal_flop"])
    free_after, _ = map(int, torch.cuda.mem_get_info(device))
    return {
        **contract,
        "dry_run": False,
        "created_at": datetime.now(UTC).isoformat(),
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "measurement": {
            "gpu_seconds": gpu_seconds,
            "wall_seconds": wall_seconds,
            "nominal_tflop_per_second": nominal_flop / gpu_seconds / 1e12,
            "finite": True,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "free_before_bytes": free_before,
            "free_after_bytes": free_after,
            "device_total_bytes": total,
        },
    }


def main() -> int:
    args = _parser().parse_args()
    _validate(args)
    contract = _contract(args)
    result = contract if args.dry_run else _run(args, contract)
    if not all(
        not isinstance(value, float) or math.isfinite(value)
        for value in result.get("measurement", {}).values()
    ):
        raise RuntimeError("measurement contains a non-finite value")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
