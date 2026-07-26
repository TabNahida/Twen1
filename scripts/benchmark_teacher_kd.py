#!/usr/bin/env python3
"""Benchmark the exact chunked Qwen3.5 teacher-KD path; never creates an optimizer."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--logits-chunk-tokens",
        type=int,
        action="append",
        dest="chunk_sizes",
        help="repeat to compare chunk sizes; defaults to 32, 64, 128 and 256",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", default=None)
    return parser


def _run_body(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    *,
    torch: Any,
    device: str,
) -> tuple[Any, float]:
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state
        torch.cuda.synchronize(device)
        body_seconds = time.perf_counter() - started
    return hidden, body_seconds


def _run_head(
    model: Any,
    hidden: Any,
    *,
    sequence_length: int,
    chunk_tokens: int,
    torch: Any,
    device: str,
) -> tuple[float, bool, float]:
    torch.cuda.synchronize(device)
    head_started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        finite = True
        for token_start in range(0, sequence_length, chunk_tokens):
            token_end = min(sequence_length, token_start + chunk_tokens)
            raw = model.lm_head(hidden[:, token_start:token_end])
            values, indices = torch.topk(raw, 64, dim=-1)
            log_z = torch.logsumexp(raw.float() / 2.0, dim=-1)
            log_top_mass = torch.logsumexp(values.float() / 2.0, dim=-1) - log_z
            tail = torch.log1p(-log_top_mass.exp().clamp(max=1.0 - 1e-7))
            host_outputs = (
                indices.cpu(),
                values.to(torch.bfloat16).cpu(),
                log_z.cpu(),
                tail.cpu(),
            )
            finite = finite and all(
                bool(torch.isfinite(item).all())
                for item in host_outputs
                if item.is_floating_point()
            )
        torch.cuda.synchronize(device)
        head_seconds = time.perf_counter() - head_started
    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    return head_seconds, finite, peak_gib


def main() -> int:
    args = _parser().parse_args()
    chunk_sizes = tuple(dict.fromkeys(args.chunk_sizes or (32, 64, 128, 256)))
    if (
        args.sequence_length <= 0
        or args.batch_size <= 0
        or args.warmup < 0
        or args.iterations <= 0
        or not chunk_sizes
        or min(chunk_sizes) <= 0
    ):
        raise ValueError("sequence/chunk/iteration values must be positive")
    model_path = Path(args.model)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch

    from twen.model_loading import load_qwen35_text_causal_lm

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("teacher-KD benchmarking requires CUDA")
    print("loading pinned teacher once...", file=sys.stderr, flush=True)
    load_started = time.perf_counter()
    model = load_qwen35_text_causal_lm(
        model_path,
        dtype=torch.bfloat16,
        device=args.device,
    ).eval()
    load_seconds = time.perf_counter() - load_started
    generator = torch.Generator(device="cpu").manual_seed(3407)
    input_ids = torch.randint(
        0,
        int(model.config.vocab_size),
        (args.batch_size, args.sequence_length),
        generator=generator,
        dtype=torch.long,
    ).to(args.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    body_timings: list[float] = []
    head_timings = {chunk_tokens: [] for chunk_tokens in chunk_sizes}
    peak_by_chunk = {chunk_tokens: 0.0 for chunk_tokens in chunk_sizes}
    all_finite = True
    total_iterations = args.warmup + args.iterations
    for iteration in range(total_iterations):
        print(
            f"benchmark iteration {iteration + 1}/{total_iterations}...",
            file=sys.stderr,
            flush=True,
        )
        hidden, body_seconds = _run_body(
            model,
            input_ids,
            attention_mask,
            torch=torch,
            device=args.device,
        )
        # Rotate the order so thermal/clock drift cannot systematically favor
        # one candidate. Every candidate consumes the exact same hidden tensor.
        offset = iteration % len(chunk_sizes)
        ordered_chunks = chunk_sizes[offset:] + chunk_sizes[:offset]
        for chunk_tokens in ordered_chunks:
            head_seconds, finite, peak_gib = _run_head(
                model,
                hidden,
                sequence_length=args.sequence_length,
                chunk_tokens=chunk_tokens,
                torch=torch,
                device=args.device,
            )
            all_finite = all_finite and finite
            if iteration >= args.warmup:
                head_timings[chunk_tokens].append(head_seconds)
                peak_by_chunk[chunk_tokens] = max(peak_by_chunk[chunk_tokens], peak_gib)
        if iteration >= args.warmup:
            body_timings.append(body_seconds)
        del hidden
    mean_body_seconds = statistics.mean(body_timings)
    measurements = []
    for chunk_tokens in chunk_sizes:
        head_seconds = statistics.mean(head_timings[chunk_tokens])
        total_seconds = mean_body_seconds + head_seconds
        measurements.append(
            {
                "logits_chunk_tokens": chunk_tokens,
                "mean_body_seconds": mean_body_seconds,
                "mean_chunked_head_seconds": head_seconds,
                "mean_total_seconds": total_seconds,
                "input_tokens_per_second": (
                    args.batch_size * args.sequence_length / total_seconds
                ),
                "peak_allocated_gib": peak_by_chunk[chunk_tokens],
            }
        )
    result = {
        "ok": all_finite,
        "no_optimizer_steps": True,
        "model": str(model_path),
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "temperature": 2.0,
        "top_k": 64,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "load_seconds": load_seconds,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device),
        "measurements": measurements,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        from twen.utils import atomic_write_text

        atomic_write_text(Path(args.output), rendered)
    print(rendered, end="")
    return 0 if all_finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
