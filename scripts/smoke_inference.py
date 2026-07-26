#!/usr/bin/env python3
"""Offline local-model CUDA smoke test; never creates an optimizer."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="verified local model directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=("full-logits", "teacher-kd"),
        default="full-logits",
        help="full model forward, or the chunked top-64 path used by KD generation",
    )
    parser.add_argument("--logits-chunk-tokens", type=int, default=64)
    parser.add_argument("--output", default=None, help="optional JSON result path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (
        args.sequence_length <= 0
        or args.warmup < 0
        or args.iterations <= 0
        or args.logits_chunk_tokens <= 0
    ):
        raise ValueError("sequence length/iterations must be positive and warmup non-negative")
    model_path = Path(args.model)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch

    from twen.model_loading import load_qwen35_text_causal_lm

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    load_started = time.perf_counter()
    model = load_qwen35_text_causal_lm(
        model_path,
        dtype=dtype,
        device=args.device,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started
    vocabulary_size = int(model.config.vocab_size)
    generator = torch.Generator(device="cpu").manual_seed(3407)
    input_ids = torch.randint(
        0,
        vocabulary_size,
        (1, args.sequence_length),
        generator=generator,
        dtype=torch.long,
    ).to(args.device)

    timings: list[float] = []
    body_timings: list[float] = []
    head_timings: list[float] = []
    output_shape: list[int] | None = None
    finite = True
    with torch.inference_mode(), torch.autocast(
        device_type="cuda" if args.device.startswith("cuda") else "cpu",
        dtype=torch.bfloat16,
        enabled=args.device.startswith("cuda"),
    ):
        for iteration in range(args.warmup + args.iterations):
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            started = time.perf_counter()
            if args.mode == "full-logits":
                output = model(input_ids=input_ids, use_cache=False).logits
                output_shape = list(output.shape)
                finite = finite and bool(torch.isfinite(output).all())
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                body_elapsed = time.perf_counter() - started
                head_elapsed = 0.0
            else:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
                hidden = model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).last_hidden_state
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                body_elapsed = time.perf_counter() - started
                head_started = time.perf_counter()
                for token_start in range(0, args.sequence_length, args.logits_chunk_tokens):
                    token_end = min(
                        args.sequence_length,
                        token_start + args.logits_chunk_tokens,
                    )
                    raw = model.lm_head(hidden[:, token_start:token_end])
                    values, indices = torch.topk(raw, 64, dim=-1)
                    log_z = torch.logsumexp(raw.float() / 2.0, dim=-1)
                    log_top_mass = (
                        torch.logsumexp(values.float() / 2.0, dim=-1) - log_z
                    )
                    tail = torch.log1p(-log_top_mass.exp().clamp(max=1.0 - 1e-7))
                    # Match KD's device-to-host copies without retaining a full-vocabulary tensor.
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
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                head_elapsed = time.perf_counter() - head_started
                output_shape = [1, args.sequence_length, 64]
                del hidden
            elapsed = body_elapsed + head_elapsed
            if iteration >= args.warmup:
                timings.append(elapsed)
                body_timings.append(body_elapsed)
                head_timings.append(head_elapsed)
    mean_seconds = statistics.mean(timings)
    result = {
        "ok": finite,
        "no_optimizer_steps": True,
        "mode": args.mode,
        "model": str(model_path),
        "device": args.device,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        "sequence_length": args.sequence_length,
        "output_shape": output_shape,
        "logits_shape": output_shape if args.mode == "full-logits" else None,
        "logits_chunk_tokens": (
            args.logits_chunk_tokens if args.mode == "teacher-kd" else None
        ),
        "load_seconds": load_seconds,
        "mean_forward_seconds": mean_seconds,
        "mean_body_seconds": statistics.mean(body_timings),
        "mean_chunked_head_seconds": statistics.mean(head_timings),
        "input_tokens_per_second": args.sequence_length / mean_seconds,
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(args.device) / 1024**3
            if args.device.startswith("cuda")
            else None
        ),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        from twen.utils import atomic_write_text

        atomic_write_text(Path(args.output), rendered)
    print(rendered, end="")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
