#!/usr/bin/env python3
"""Compile a small Qwen3.5 FLA TileLang backward without creating an optimizer.

This isolates compiler/JIT compatibility.  Passing T=64 does not certify the
full T=4096 TileLang kernel, which is not production-safe on the RTX 5090.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--fresh-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use an empty temporary TileLang cache so success proves JIT compilation",
    )
    return parser


def _render(result: dict[str, Any], output: str | None) -> None:
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    print(text, end="")


def _nvcc_version(cuda_home: str) -> str:
    return subprocess.run(
        [str(Path(cuda_home) / "bin" / "nvcc"), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _run(sequence_length: int) -> dict[str, Any]:
    if sequence_length <= 0 or sequence_length % 64:
        raise ValueError("sequence-length must be a positive multiple of 64")
    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home:
        raise RuntimeError(
            "CUDA_HOME must be selected before Python starts; use scripts/with_cuda_toolchain.sh"
        )
    os.environ["FLA_TILELANG"] = "1"

    import torch
    import torch.nn.functional as F
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from tilelang import env as tilelang_env

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    resolved_cuda_home = str(Path(cuda_home).resolve())
    tilelang_cuda_home = str(Path(tilelang_env.CUDA_HOME).resolve())
    if resolved_cuda_home != tilelang_cuda_home:
        raise RuntimeError(f"TileLang selected {tilelang_cuda_home}, expected {resolved_cuda_home}")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(3407)
    batch, heads, key_dim, value_dim = 1, 16, 128, 128

    def bf16(shape: tuple[int, ...]) -> Any:
        return torch.randn(shape, device=device, dtype=torch.bfloat16).requires_grad_(True)

    q = bf16((batch, sequence_length, heads, key_dim))
    k = bf16((batch, sequence_length, heads, key_dim))
    v = bf16((batch, sequence_length, heads, value_dim))
    g_raw = torch.randn(
        (batch, sequence_length, heads),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    beta_raw = bf16((batch, sequence_length, heads))
    g = -F.softplus(g_raw)
    beta = beta_raw.sigmoid()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    output, final_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    if final_state is not None:
        raise RuntimeError("unexpected final state")
    loss = output.float().square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started

    gradients = {
        name: tensor.grad
        for name, tensor in {
            "q": q,
            "k": k,
            "v": v,
            "g_raw": g_raw,
            "beta_raw": beta_raw,
        }.items()
    }
    finite = {
        name: bool(value is not None and torch.isfinite(value).all().item())
        for name, value in gradients.items()
    }
    return {
        "ok": all(finite.values()) and bool(torch.isfinite(loss).item()),
        "no_optimizer_created": True,
        "no_optimizer_steps": True,
        "optimizer_step_calls": 0,
        "forced_backend": "tilelang",
        "scope": "compiler_and_small_shape_jit_only_not_production_acceptance",
        "cuda_home": resolved_cuda_home,
        "tilelang_cuda_home": tilelang_cuda_home,
        "nvcc_version": _nvcc_version(resolved_cuda_home),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "shape": {
            "batch": batch,
            "sequence_length": sequence_length,
            "heads": heads,
            "key_dim": key_dim,
            "value_dim": value_dim,
        },
        "seconds_including_fresh_jit": seconds,
        "loss": float(loss.detach()),
        "gradient_finite": finite,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cache_context: Any = None
    try:
        if args.fresh_cache:
            cache_context = tempfile.TemporaryDirectory(prefix="twen-tilelang-smoke-")
            os.environ["TILELANG_CACHE_DIR"] = cache_context.name
        result = _run(args.sequence_length)
    except Exception as error:
        result = {
            "ok": False,
            "no_optimizer_created": True,
            "no_optimizer_steps": True,
            "optimizer_step_calls": 0,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
    finally:
        if cache_context is not None:
            cache_context.cleanup()
    _render(result, args.output)
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
