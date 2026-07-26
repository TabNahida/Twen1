#!/usr/bin/env python3
"""CUDA smoke for compiled streaming CE/KD/anchor under chunk checkpointing.

The smoke deliberately performs no optimizer construction or step.  It runs
both the eager oracle and the compiled reduction twice, checking outputs,
student/LM-head gradients, checkpoint recomputation, compiled-callable cache
reuse, and steady-call peak memory.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--teacher-top-k", type=int, default=64)
    parser.add_argument("--chunk-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--gradient-relative-l2-tolerance", type=float, default=5e-4)
    parser.add_argument("--output-absolute-tolerance", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", default=None, help="optional JSON report path")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "hidden_size": args.hidden_size,
        "vocab_size": args.vocab_size,
        "teacher_top_k": args.teacher_top_k,
        "chunk_tokens": args.chunk_tokens,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"positive smoke arguments required: {invalid}")
    if args.teacher_top_k >= args.vocab_size:
        raise ValueError("teacher-top-k must be smaller than vocab-size")
    if args.batch_size * args.sequence_length % args.chunk_tokens:
        raise ValueError(
            "batch-size * sequence-length must be divisible by chunk-tokens "
            "so the smoke exercises exactly one static compiled shape"
        )
    finite_positive = {
        "temperature": args.temperature,
        "gradient_relative_l2_tolerance": args.gradient_relative_l2_tolerance,
        "output_absolute_tolerance": args.output_absolute_tolerance,
    }
    invalid_floats = [
        name for name, value in finite_positive.items() if not math.isfinite(value) or value <= 0
    ]
    if invalid_floats:
        raise ValueError(f"finite positive smoke arguments required: {invalid_floats}")


def _relative_l2(actual: Any, expected: Any, *, torch: Any) -> float:
    difference = actual.float() - expected.float()
    denominator = torch.linalg.vector_norm(expected.float().reshape(-1)).clamp_min(1e-12)
    return float(torch.linalg.vector_norm(difference.reshape(-1)) / denominator)


def _comparison(actual: Any, expected: Any, *, torch: Any) -> dict[str, float | bool]:
    difference = (actual.float() - expected.float()).abs()
    return {
        "actual_finite": bool(torch.isfinite(actual).all().item()),
        "expected_finite": bool(torch.isfinite(expected).all().item()),
        "max_absolute_difference": float(difference.max().item()) if difference.numel() else 0.0,
        "relative_l2": _relative_l2(actual, expected, torch=torch),
    }


def _cache_info(function: Any) -> dict[str, int | None]:
    info = function.cache_info()
    return {
        "hits": int(info.hits),
        "misses": int(info.misses),
        "maxsize": info.maxsize,
        "currsize": int(info.currsize),
    }


def _dynamo_unique_graphs(torch: Any) -> int | None:
    try:
        return int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch import nn

    from twen.training.losses import (
        _compiled_streaming_student_sums_no_anchor,
        _compiled_streaming_student_sums_with_anchor,
        streaming_language_model_losses,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    class CountingLinear(nn.Linear):
        calls: int

        def __init__(self) -> None:
            super().__init__(
                args.hidden_size,
                args.vocab_size,
                bias=False,
                device=device,
                dtype=torch.bfloat16,
            )
            self.calls = 0

        def forward(self, value: Any) -> Any:
            self.calls += 1
            return super().forward(value)

    token_shape = (args.batch_size, args.sequence_length)
    base_hidden = torch.randn(
        (*token_shape, args.hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )
    base_anchor = torch.randn_like(base_hidden)
    labels = torch.randint(args.vocab_size, token_shape, device=device)
    mask = torch.ones(token_shape, dtype=torch.bool, device=device)
    if mask.numel() >= 4:
        mask.reshape(-1)[1] = False
        mask.reshape(-1)[-2] = False

    teacher_logits = torch.randn((*token_shape, args.vocab_size), device=device)
    teacher_indices = teacher_logits.topk(args.teacher_top_k, dim=-1).indices
    teacher_topk_logits = torch.gather(teacher_logits, -1, teacher_indices)
    scaled_teacher = teacher_logits / args.temperature
    teacher_logsumexp = torch.logsumexp(scaled_teacher, dim=-1)
    teacher_top_mass = torch.gather(
        scaled_teacher.softmax(dim=-1),
        -1,
        teacher_indices,
    ).sum(dim=-1)
    teacher_tail_logprob = torch.log1p(-teacher_top_mass.clamp_max(1.0 - 1e-7))
    del scaled_teacher, teacher_logits, teacher_top_mass

    _compiled_streaming_student_sums_no_anchor.cache_clear()
    _compiled_streaming_student_sums_with_anchor.cache_clear()
    try:
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
    except AttributeError:
        pass

    chunk_count = labels.numel() // args.chunk_tokens
    cases: dict[str, Any] = {}
    for with_anchor in (False, True):
        case_name = "anchor" if with_anchor else "no_anchor"
        cache_function = (
            _compiled_streaming_student_sums_with_anchor
            if with_anchor
            else _compiled_streaming_student_sums_no_anchor
        )

        eager_head = CountingLinear()
        compiled_head = CountingLinear()
        compiled_head.load_state_dict(eager_head.state_dict())
        eager_anchor_head = CountingLinear()
        compiled_anchor_head = CountingLinear()
        eager_anchor_head.load_state_dict(eager_head.state_dict())
        compiled_anchor_head.load_state_dict(eager_head.state_dict())
        eager_anchor_head.requires_grad_(False)
        compiled_anchor_head.requires_grad_(False)

        def invoke(
            *,
            head: Any,
            anchor_head: Any,
            with_anchor_case: bool,
            compiled: bool,
            call_index: int,
        ) -> dict[str, Any]:
            head.zero_grad(set_to_none=True)
            hidden = base_hidden.detach().clone().requires_grad_(True)
            student_calls_before = head.calls
            anchor_calls_before = anchor_head.calls
            torch.cuda.synchronize(device)
            baseline_allocated = int(torch.cuda.memory_allocated(device))
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            losses = streaming_language_model_losses(
                hidden,
                head,
                labels,
                teacher_indices,
                teacher_topk_logits,
                teacher_logsumexp,
                teacher_tail_logprob,
                temperature=args.temperature,
                mask=mask,
                anchor_hidden_states=base_anchor if with_anchor_case else None,
                anchor_lm_head=anchor_head if with_anchor_case else None,
                chunk_tokens=args.chunk_tokens,
                checkpoint_chunks=True,
                compile_loss=compiled,
            )
            total = losses.ntp + losses.teacher_kd
            if with_anchor_case:
                assert losses.anchor_kl is not None
                total = total + 0.1 * losses.anchor_kl
            else:
                assert losses.anchor_kl is None
            total.backward()
            end.record()
            end.synchronize()
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            if hidden.grad is None or head.weight.grad is None:
                raise RuntimeError("student hidden/head gradient is missing")
            result = {
                "call_index": call_index,
                "seconds": float(start.elapsed_time(end)) / 1000.0,
                "losses": {
                    "ntp": float(losses.ntp.detach()),
                    "teacher_kd": float(losses.teacher_kd.detach()),
                    "anchor_kl": (
                        None if losses.anchor_kl is None else float(losses.anchor_kl.detach())
                    ),
                    "total": float(total.detach()),
                },
                "hidden_gradient": hidden.grad.detach().cpu().clone(),
                "head_gradient": head.weight.grad.detach().cpu().clone(),
                "student_head_calls": head.calls - student_calls_before,
                "anchor_head_calls": anchor_head.calls - anchor_calls_before,
                "baseline_allocated_bytes": baseline_allocated,
                "peak_allocated_bytes": peak_allocated,
                "peak_increment_bytes": max(peak_allocated - baseline_allocated, 0),
            }
            del hidden, losses, total
            return result

        eager_runs = [
            invoke(
                head=eager_head,
                anchor_head=eager_anchor_head,
                with_anchor_case=with_anchor,
                compiled=False,
                call_index=index,
            )
            for index in (1, 2)
        ]
        compiled_runs = []
        cache_after_calls = []
        graphs_before = _dynamo_unique_graphs(torch)
        for index in (1, 2):
            compiled_runs.append(
                invoke(
                    head=compiled_head,
                    anchor_head=compiled_anchor_head,
                    with_anchor_case=with_anchor,
                    compiled=True,
                    call_index=index,
                )
            )
            cache_after_calls.append(_cache_info(cache_function))
        graphs_after = _dynamo_unique_graphs(torch)

        comparisons = []
        for eager, compiled_result in zip(eager_runs, compiled_runs, strict=True):
            output_difference = max(
                abs(compiled_result["losses"][name] - eager["losses"][name])
                for name in ("ntp", "teacher_kd", "total")
            )
            if with_anchor:
                output_difference = max(
                    output_difference,
                    abs(compiled_result["losses"]["anchor_kl"] - eager["losses"]["anchor_kl"]),
                )
            hidden_comparison = _comparison(
                compiled_result["hidden_gradient"],
                eager["hidden_gradient"],
                torch=torch,
            )
            head_comparison = _comparison(
                compiled_result["head_gradient"],
                eager["head_gradient"],
                torch=torch,
            )
            comparisons.append(
                {
                    "call_index": compiled_result["call_index"],
                    "max_loss_absolute_difference": output_difference,
                    "hidden_gradient": hidden_comparison,
                    "head_gradient": head_comparison,
                }
            )

        recompute_observed = all(
            run["student_head_calls"] >= 2 * chunk_count for run in compiled_runs
        ) and (
            not with_anchor
            or all(run["anchor_head_calls"] >= 2 * chunk_count for run in compiled_runs)
        )
        cache_reused = (
            cache_after_calls[0]["misses"] == 1
            and cache_after_calls[1]["misses"] == 1
            and cache_after_calls[1]["hits"] > cache_after_calls[0]["hits"]
        )
        numerical_match = all(
            comparison["max_loss_absolute_difference"] <= args.output_absolute_tolerance
            and comparison["hidden_gradient"]["actual_finite"]
            and comparison["head_gradient"]["actual_finite"]
            and comparison["hidden_gradient"]["relative_l2"]
            <= args.gradient_relative_l2_tolerance
            and comparison["head_gradient"]["relative_l2"]
            <= args.gradient_relative_l2_tolerance
            for comparison in comparisons
        )

        def public_run(run: dict[str, Any]) -> dict[str, Any]:
            return {
                key: value
                for key, value in run.items()
                if key not in {"hidden_gradient", "head_gradient"}
            }

        cases[case_name] = {
            "ok": numerical_match and recompute_observed and cache_reused,
            "with_anchor": with_anchor,
            "checkpoint_chunks": True,
            "compile_loss": True,
            "chunk_count": chunk_count,
            "expected_minimum_head_calls_per_forward_backward": 2 * chunk_count,
            "eager_runs": [public_run(run) for run in eager_runs],
            "compiled_runs": [public_run(run) for run in compiled_runs],
            "comparisons": comparisons,
            "recompute_observed": recompute_observed,
            "compiled_callable_cache_after_each_call": cache_after_calls,
            "compiled_callable_cache_reused": cache_reused,
            "dynamo_unique_graphs_before": graphs_before,
            "dynamo_unique_graphs_after": graphs_after,
            "numerical_match": numerical_match,
        }

    return {
        "ok": all(case["ok"] for case in cases.values()),
        "scope": "compiled_streaming_loss_checkpoint_forward_backward_smoke",
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "dtype": "bfloat16",
        "shape": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "hidden_size": args.hidden_size,
            "vocab_size": args.vocab_size,
            "teacher_top_k": args.teacher_top_k,
            "chunk_tokens": args.chunk_tokens,
        },
        "tolerances": {
            "output_absolute": args.output_absolute_tolerance,
            "gradient_relative_l2": args.gradient_relative_l2_tolerance,
        },
        "two_forward_backward_calls_per_variant": True,
        "no_optimizer_created": True,
        "no_optimizer_steps": True,
        "cases": cases,
    }


def _render(result: dict[str, Any], output: str | None) -> None:
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    print(text, end="")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_args(args)
        result = _run(args)
    except Exception as error:
        result = {
            "ok": False,
            "no_optimizer_created": True,
            "no_optimizer_steps": True,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
    _render(result, args.output)
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
