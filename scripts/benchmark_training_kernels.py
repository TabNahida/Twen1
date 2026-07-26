#!/usr/bin/env python3
"""CUDA forward/backward microbenchmarks for Twen hot paths; never creates an optimizer."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import subprocess
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dense-tokens", type=int, default=64)
    parser.add_argument("--sparse-tokens", type=int, default=128)
    parser.add_argument("--loss-tokens", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument("--teacher-top-k", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--sparse-top-k",
        type=int,
        nargs="+",
        default=(8, 4, 2),
        help="top-k values to compare in the sparse forward/backward benchmark",
    )
    parser.add_argument(
        "--loss-chunk-sizes",
        type=int,
        nargs="+",
        default=(128, 256, 512),
        help="token chunk sizes for the combined CE+KD+anchor benchmark",
    )
    parser.add_argument(
        "--loss-checkpoint-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--anchor-weight", type=float, default=0.1)
    parser.add_argument("--dense-output-relative-l2-tolerance", type=float, default=0.01)
    parser.add_argument("--dense-gradient-relative-l2-tolerance", type=float, default=0.05)
    parser.add_argument("--skip-sparse", action="store_true")
    parser.add_argument("--skip-combined-loss", action="store_true")
    parser.add_argument("--output", default=None, help="optional JSON result path")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "dense_tokens": args.dense_tokens,
        "sparse_tokens": args.sparse_tokens,
        "loss_tokens": args.loss_tokens,
        "vocab_size": args.vocab_size,
        "teacher_top_k": args.teacher_top_k,
        "iterations": args.iterations,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"positive benchmark arguments required: {invalid}")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(args.anchor_weight) or args.anchor_weight < 0:
        raise ValueError("anchor weight must be finite and non-negative")
    if args.teacher_top_k >= args.vocab_size:
        raise ValueError("teacher_top_k must be smaller than vocab_size")
    if any(value <= 0 for value in args.loss_chunk_sizes):
        raise ValueError("loss chunk sizes must be positive")
    if any(value not in {2, 4, 8} for value in args.sparse_top_k):
        raise ValueError("this production-shape sparse benchmark supports top-k 2, 4, or 8")
    if len(set(args.sparse_top_k)) != len(args.sparse_top_k):
        raise ValueError("sparse top-k values must not contain duplicates")
    for name in (
        "dense_output_relative_l2_tolerance",
        "dense_gradient_relative_l2_tolerance",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


def _event_seconds(function: Callable[[], Any], *, torch: Any) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / 1000.0


def _timed_many(
    functions: Mapping[str, Callable[[], Any]],
    *,
    torch: Any,
    device: str,
    warmup: int,
    iterations: int,
) -> dict[str, dict[str, float | int]]:
    """Interleave variants and use CUDA events so launch asynchrony cannot skew results."""

    labels = tuple(functions)
    for _ in range(warmup):
        for label in labels:
            functions[label]()
    torch.cuda.synchronize(device)

    samples: dict[str, list[float]] = {label: [] for label in labels}
    for iteration in range(iterations):
        order = labels if iteration % 2 == 0 else tuple(reversed(labels))
        for label in order:
            samples[label].append(_event_seconds(functions[label], torch=torch))
    return {
        label: {
            "mean_seconds": statistics.mean(values),
            "std_seconds": statistics.pstdev(values),
            "min_seconds": min(values),
            "max_seconds": max(values),
            "iterations": len(values),
        }
        for label, values in samples.items()
    }


def _measure_peak(
    function: Callable[[], Any],
    cleanup: Callable[[], None],
    *,
    torch: Any,
    device: str,
) -> dict[str, float]:
    cleanup()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    function()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    allocated = torch.cuda.memory_allocated(device)
    cleanup()
    gib = float(1024**3)
    return {
        "baseline_allocated_gib": baseline / gib,
        "peak_allocated_gib": peak / gib,
        "peak_delta_gib": max(peak - baseline, 0) / gib,
        "post_step_allocated_gib": allocated / gib,
    }


def _tensor_error(actual: Any, reference: Any) -> dict[str, float]:
    actual_fp32 = actual.detach().float()
    reference_fp32 = reference.detach().float()
    difference = actual_fp32 - reference_fp32
    relative_l2 = difference.norm() / reference_fp32.norm().clamp_min(1e-12)
    max_abs = difference.abs().max()
    relative_to_scale = max_abs / reference_fp32.abs().max().clamp_min(1e-12)
    return {
        "relative_l2": float(relative_l2),
        "max_abs": float(max_abs),
        "max_relative_to_reference_scale": float(relative_to_scale),
    }


def _dense_capture(
    module: Any,
    hidden_value: Any,
    *,
    torch: Any,
    sliced_reference: bool,
) -> tuple[Any, dict[str, Any]]:
    module.zero_grad(set_to_none=True)
    hidden = hidden_value.detach().clone().requires_grad_(True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = module(hidden, return_expert_outputs=sliced_reference)
        output = result[0] if sliced_reference else result
        objective = output.float().square().mean()
    objective.backward()
    gradients = {
        "input": hidden.grad.detach().clone(),
        **{
            name: parameter.grad.detach().clone()
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        },
    }
    return output.detach().clone(), gradients


def _dense_benchmark(args: argparse.Namespace, torch: Any) -> dict[str, Any]:
    from twen.modeling import DenseTransferMLP, TransferAdapters

    device = args.device
    small, donor, intermediate, experts = 1024, 4096, 12288, 8
    generator = torch.Generator(device=device).manual_seed(3407)

    def random_weight(shape: tuple[int, ...], scale: float, dtype: Any) -> Any:
        value = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        value.mul_(scale)
        return value

    gate = random_weight((intermediate, donor), donor**-0.5, torch.bfloat16)
    up = random_weight((intermediate, donor), donor**-0.5, torch.bfloat16)
    down = random_weight((donor, intermediate), intermediate**-0.5, torch.bfloat16)
    partition = torch.arange(intermediate, device=device).reshape(experts, -1)
    adapters = TransferAdapters(small, donor, device=device, dtype=torch.float32)
    module = DenseTransferMLP(
        gate,
        up,
        down,
        partition,
        adapters=adapters,
        branch_scale=1.0,
    ).train()
    hidden_value = random_weight((1, args.dense_tokens, small), small**-0.5, torch.bfloat16)

    fast_output, fast_gradients = _dense_capture(
        module,
        hidden_value,
        torch=torch,
        sliced_reference=False,
    )
    reference_output, reference_gradients = _dense_capture(
        module,
        hidden_value,
        torch=torch,
        sliced_reference=True,
    )
    output_error = _tensor_error(fast_output, reference_output)
    gradient_errors = {
        name: _tensor_error(fast_gradients[name], reference_gradients[name])
        for name in fast_gradients
    }
    gradient_relative_l2 = [value["relative_l2"] for value in gradient_errors.values()]
    max_gradient_relative_l2 = max(gradient_relative_l2)
    validation_passed = bool(
        math.isfinite(output_error["relative_l2"])
        and all(math.isfinite(value) for value in gradient_relative_l2)
        and output_error["relative_l2"] <= args.dense_output_relative_l2_tolerance
        and max_gradient_relative_l2 <= args.dense_gradient_relative_l2_tolerance
    )
    del fast_output, reference_output, fast_gradients, reference_gradients

    timing_hidden = hidden_value.detach().clone().requires_grad_(True)

    def cleanup() -> None:
        module.zero_grad(set_to_none=True)
        timing_hidden.grad = None
        module.last_aux = None

    def run(sliced_reference: bool) -> None:
        cleanup()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = module(timing_hidden, return_expert_outputs=sliced_reference)
            output = result[0] if sliced_reference else result
            objective = output.float().square().mean()
        objective.backward()

    variants = {
        "vectorized_full_ffn": lambda: run(False),
        "sliced_expert_sum": lambda: run(True),
    }
    timings = _timed_many(
        variants,
        torch=torch,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    memory = {
        label: _measure_peak(function, cleanup, torch=torch, device=device)
        for label, function in variants.items()
    }
    fast_seconds = float(timings["vectorized_full_ffn"]["mean_seconds"])
    reference_seconds = float(timings["sliced_expert_sum"]["mean_seconds"])
    return {
        "tokens": args.dense_tokens,
        "timing": timings,
        "memory": memory,
        "forward_backward_speedup": reference_seconds / fast_seconds,
        "input_tokens_per_second": args.dense_tokens / fast_seconds,
        "bf16_equivalence": {
            "passed": validation_passed,
            "output": output_error,
            "gradients": gradient_errors,
            "max_gradient_relative_l2": max_gradient_relative_l2,
            "output_relative_l2_tolerance": args.dense_output_relative_l2_tolerance,
            "gradient_relative_l2_tolerance": args.dense_gradient_relative_l2_tolerance,
        },
    }


def _sparse_benchmark(args: argparse.Namespace, torch: Any) -> dict[str, Any]:
    from twen.modeling import SparseTransferMLP

    device = args.device
    hidden_size, intermediate, experts, rank = 1024, 1536, 8, 16
    generator = torch.Generator(device=device).manual_seed(4407)

    def random_weight(shape: tuple[int, ...], scale: float, dtype: Any) -> Any:
        value = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        value.mul_(scale)
        return value

    gate = random_weight((experts, intermediate, hidden_size), hidden_size**-0.5, torch.bfloat16)
    up = random_weight((experts, intermediate, hidden_size), hidden_size**-0.5, torch.bfloat16)
    down = random_weight((experts, hidden_size, intermediate), intermediate**-0.5, torch.bfloat16)
    router = random_weight((experts, hidden_size), hidden_size**-0.5, torch.float32)
    module = SparseTransferMLP(
        torch.nn.Identity(),
        gate,
        up,
        down,
        router,
        top_k=experts,
        lora_rank=rank,
        lora_alpha=float(rank),
        lora_trainable_dtype=torch.float32,
        branch_scale=1.0,
    ).train()
    with torch.no_grad():
        for name, parameter in module.experts.named_parameters():
            if name.endswith("lora_b"):
                parameter.normal_(std=0.001, generator=generator)

    hidden = random_weight(
        (1, args.sparse_tokens, hidden_size), hidden_size**-0.5, torch.bfloat16
    ).requires_grad_(True)

    def cleanup() -> None:
        module.clear_aux()
        module.zero_grad(set_to_none=True)
        hidden.grad = None

    def make_run(top_k: int) -> Callable[[], None]:
        def run() -> None:
            cleanup()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = module(hidden, top_k=top_k)
                objective = output.float().square().mean()
            objective.backward()
            module.clear_aux()

        return run

    variants = {f"top_{top_k}": make_run(top_k) for top_k in args.sparse_top_k}
    timings = _timed_many(
        variants,
        torch=torch,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    memory = {
        label: _measure_peak(function, cleanup, torch=torch, device=device)
        for label, function in variants.items()
    }
    results: dict[str, Any] = {}
    for top_k in args.sparse_top_k:
        label = f"top_{top_k}"
        seconds = float(timings[label]["mean_seconds"])
        results[label] = {
            "dispatch": (
                "vectorized_full_expert" if top_k == experts else "dynamic_nonzero_dispatch"
            ),
            "timing": timings[label],
            "memory": memory[label],
            "input_tokens_per_second": args.sparse_tokens / seconds,
        }
    return {
        "tokens": args.sparse_tokens,
        "experts": experts,
        "results": results,
    }


def _combined_loss_benchmark(args: argparse.Namespace, torch: Any) -> dict[str, Any]:
    from twen.training.losses import (
        bucketed_topk_kl,
        causal_language_model_loss,
        masked_full_kl,
    )

    device = args.device
    tokens = args.loss_tokens
    vocabulary = args.vocab_size
    top_k = args.teacher_top_k
    generator = torch.Generator(device=device).manual_seed(5407)
    student = torch.randn(
        (1, tokens, vocabulary),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).requires_grad_(True)
    reference = torch.randn(
        (1, tokens, vocabulary),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    labels = torch.randint(
        0,
        vocabulary,
        (1, tokens),
        generator=generator,
        device=device,
    )
    offsets = torch.arange(tokens, device=device, dtype=torch.long).unsqueeze(-1) * 104729
    teacher_indices = (
        (offsets + torch.arange(top_k, device=device, dtype=torch.long).unsqueeze(0))
        .remainder(vocabulary)
        .unsqueeze(0)
    )
    teacher_topk_logits = torch.randn(
        (1, tokens, top_k),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    scaled_teacher_topk = teacher_topk_logits.float() / args.temperature
    teacher_logsumexp = torch.logsumexp(scaled_teacher_topk, dim=-1) - math.log(0.8)
    teacher_tail_logprob = torch.full(
        (1, tokens), math.log(0.2), device=device, dtype=torch.float32
    )
    mask = torch.ones((1, tokens), device=device, dtype=torch.long)

    def cleanup() -> None:
        student.grad = None

    def make_run(chunk_tokens: int) -> Callable[[], None]:
        def run() -> None:
            cleanup()
            ntp = causal_language_model_loss(
                student,
                labels,
                chunk_tokens=chunk_tokens,
                checkpoint_chunks=args.loss_checkpoint_chunks,
            )
            kd = bucketed_topk_kl(
                student,
                teacher_indices,
                teacher_topk_logits,
                teacher_logsumexp,
                teacher_tail_logprob,
                temperature=args.temperature,
                mask=mask,
                chunk_tokens=chunk_tokens,
                checkpoint_chunks=args.loss_checkpoint_chunks,
            )
            anchor = masked_full_kl(
                student,
                reference,
                mask,
                chunk_tokens=chunk_tokens,
                checkpoint_chunks=args.loss_checkpoint_chunks,
            )
            (ntp + kd + args.anchor_weight * anchor).backward()

        return run

    variants = {
        f"chunk_{chunk_tokens}": make_run(chunk_tokens) for chunk_tokens in args.loss_chunk_sizes
    }
    timings = _timed_many(
        variants,
        torch=torch,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    memory = {
        label: _measure_peak(function, cleanup, torch=torch, device=device)
        for label, function in variants.items()
    }
    results: dict[str, Any] = {}
    for chunk_tokens in args.loss_chunk_sizes:
        label = f"chunk_{chunk_tokens}"
        seconds = float(timings[label]["mean_seconds"])
        results[label] = {
            "timing": timings[label],
            "memory": memory[label],
            "input_tokens_per_second": tokens / seconds,
        }
    return {
        "tokens": tokens,
        "vocab_size": vocabulary,
        "teacher_top_k": top_k,
        "temperature": args.temperature,
        "anchor_weight": args.anchor_weight,
        "checkpoint_chunks": args.loss_checkpoint_chunks,
        "results": results,
    }


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


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


def _release_cuda(torch: Any) -> None:
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    args = _parser().parse_args()
    _validate_args(args)
    import torch

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA device")
    requested_device = args.device
    parsed_device = torch.device(requested_device)
    device_index = (
        torch.cuda.current_device() if parsed_device.index is None else parsed_device.index
    )
    torch.cuda.set_device(device_index)
    args.device = f"cuda:{device_index}"
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    from transformers.utils import (
        is_causal_conv1d_available,
        is_flash_linear_attention_available,
    )

    properties = torch.cuda.get_device_properties(args.device)
    result: dict[str, Any] = {
        "ok": True,
        "no_optimizer_created": True,
        "no_optimizer_steps": True,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "requested_device": requested_device,
        "arguments": vars(args),
        "timing_method": "interleaved CUDA events with per-variant warmup",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "gpu_total_memory_bytes": properties.total_memory,
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "dependencies": {
            "transformers": _package_version("transformers"),
            "flash_linear_attention": _package_version("flash-linear-attention"),
            "causal_conv1d": _package_version("causal-conv1d"),
            "transformers_flash_linear_attention_available": (
                is_flash_linear_attention_available()
            ),
            "transformers_causal_conv1d_available": is_causal_conv1d_available(),
        },
    }
    dense = _dense_benchmark(args, torch)
    result["dense_full_ffn"] = dense
    result["ok"] = bool(dense["bf16_equivalence"]["passed"])
    _release_cuda(torch)
    if not args.skip_sparse:
        result["sparse_routing"] = _sparse_benchmark(args, torch)
        _release_cuda(torch)
    if not args.skip_combined_loss:
        result["combined_ce_kd_anchor"] = _combined_loss_benchmark(args, torch)
        _release_cuda(torch)

    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        from twen.utils import atomic_write_text

        atomic_write_text(Path(args.output), rendered)
    print(rendered, end="")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
