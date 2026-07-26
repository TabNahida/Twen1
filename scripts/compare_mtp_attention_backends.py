#!/usr/bin/env python3
"""Compare causal eager/SDPA and legacy unmasked MTP with one loaded module."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/models/qwen3.5-0.8b-base")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--padded", action="store_true")
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _error(left: Any, right: Any, *, mask: Any | None = None) -> dict[str, float]:
    import torch

    left = left.float()
    right = right.float()
    if mask is not None:
        while mask.ndim < left.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(left)
        left = left[mask]
        right = right[mask]
    difference = (left - right).abs()
    error_rms = float(torch.sqrt(difference.square().mean())) if difference.numel() else 0.0
    reference_rms = float(torch.sqrt(right.square().mean())) if right.numel() else 0.0
    left_flat = left.reshape(-1).double()
    right_flat = right.reshape(-1).double()
    cosine_denominator = float(torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(right_flat))
    cosine = float(torch.dot(left_flat, right_flat)) / cosine_denominator if cosine_denominator else 1.0
    return {
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
        "mean_abs": float(difference.mean()) if difference.numel() else 0.0,
        "error_rms": error_rms,
        "reference_rms": reference_rms,
        "normalized_rms": error_rms / reference_rms if reference_rms else 0.0,
        "cosine": cosine,
    }


def _mask_summary(mask: Any, *, torch: Any) -> dict[str, Any]:
    if mask is None:
        return {"kind": "none", "shape": None, "dtype": None, "fully_masked_rows": 0}
    if not isinstance(mask, torch.Tensor):
        return {"kind": type(mask).__name__}
    value = mask.detach()
    if value.dtype == torch.bool:
        allowed = value
        kind = "boolean_true_allowed"
        minimum, maximum = int(value.min()), int(value.max())
    else:
        allowed = value.eq(0)
        kind = "additive_zero_allowed"
        minimum, maximum = float(value.float().min()), float(value.float().max())
    return {
        "kind": kind,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": minimum,
        "max": maximum,
        "allowed_elements": int(allowed.sum()),
        "masked_elements": int(allowed.numel() - allowed.sum()),
        "fully_masked_rows": int((~allowed.any(dim=-1)).sum()),
        "finite": bool(torch.isfinite(value).all()) if value.is_floating_point() else True,
    }


def _load_vocabulary(checkpoint: Path, *, torch: Any, device: Any) -> tuple[Any, Any]:
    from twen.model_loading import SafetensorCheckpoint, read_qwen_text_config

    config = read_qwen_text_config(checkpoint)
    source = SafetensorCheckpoint(checkpoint)
    keys = ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight")
    key = next((candidate for candidate in keys if candidate in source), None)
    if key is None:
        raise RuntimeError("text embedding weight is absent")
    weight = source.tensor(key).to(device=device, dtype=torch.bfloat16)
    if list(weight.shape) != [int(config["vocab_size"]), int(config["hidden_size"])]:
        raise RuntimeError("embedding/config shape mismatch")
    embedding = torch.nn.Embedding(*weight.shape, device=device, dtype=torch.bfloat16)
    embedding.weight = torch.nn.Parameter(weight, requires_grad=False)
    head = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, device=device, dtype=torch.bfloat16)
    head.weight = embedding.weight
    return embedding.eval(), head.eval()


def _set_backend(mtp: Any, backend: str | None) -> dict[str, Any]:
    mtp.config._attn_implementation = backend
    ids = {id(mtp.config)}
    for layer in mtp.layers:
        layer.self_attn.config._attn_implementation = backend
        ids.add(id(layer.self_attn.config))
    return {
        "requested": backend,
        "module": mtp.config._attn_implementation,
        "layers": [layer.self_attn.config._attn_implementation for layer in mtp.layers],
        "unique_config_objects": len(ids),
    }


def _run_backend(
    backend: str | None,
    *,
    mtp: Any,
    embedding: Any,
    head: Any,
    base_hidden: Any,
    input_ids: Any,
    attention_mask: Any,
    chunk_tokens: int,
    gradient: bool,
    sdpa_kernel_name: str | None,
    torch: Any,
) -> dict[str, Any]:
    import twen.modeling.mtp as mtp_source
    from twen.training.streaming import _streaming_mtp_cross_entropy, native_mtp_target_mask

    backend_state = _set_backend(mtp, backend)
    original_mask_function = mtp_source.create_causal_mask
    masks: list[dict[str, Any]] = []

    def record_mask(*args: Any, **kwargs: Any) -> Any:
        result = original_mask_function(*args, **kwargs)
        masks.append(_mask_summary(result, torch=torch))
        return result

    mtp_source.create_causal_mask = record_mask
    hidden = base_hidden.detach().clone().requires_grad_(gradient)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(hidden.device)
    if sdpa_kernel_name == "math":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        kernel_context = sdpa_kernel(SDPBackend.MATH)
    elif sdpa_kernel_name is None:
        kernel_context = nullcontext()
    else:
        raise ValueError(f"unsupported SDPA kernel selector: {sdpa_kernel_name}")
    try:
        with kernel_context, torch.autocast("cuda", dtype=torch.bfloat16):
            output = mtp(
                hidden,
                input_ids,
                embed_tokens=embedding,
                attention_mask=attention_mask,
            )
            target_mask = native_mtp_target_mask(input_ids, attention_mask)
            loss = _streaming_mtp_cross_entropy(
                output[:, :-1],
                head,
                input_ids[:, 2:],
                target_mask,
                chunk_tokens=chunk_tokens,
                checkpoint_chunks=gradient,
                compile_loss=False,
            )
        positions = sorted({0, (output.shape[1] - 1) // 2, output.shape[1] - 1})
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = head(output[:, positions]).float().cpu()
        if gradient:
            loss.backward()
        torch.cuda.synchronize(hidden.device)
        return {
            "backend": backend_state,
            "sdpa_kernel_selector": sdpa_kernel_name,
            "mask": masks[-1],
            "loss": float(loss.detach()),
            "hidden": output.detach().float().cpu(),
            "sampled_logits": logits,
            "sample_positions": positions,
            "gradient": hidden.grad.detach().float().cpu() if hidden.grad is not None else None,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(hidden.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(hidden.device)),
        }
    finally:
        mtp_source.create_causal_mask = original_mask_function


def _without_tensors(result: Mapping[str, Any]) -> dict[str, Any]:
    omitted = {"hidden", "sampled_logits", "gradient"}
    return {key: value for key, value in result.items() if key not in omitted}


def _make_inputs(args: argparse.Namespace, *, hidden_size: int, torch: Any, device: Any) -> tuple[Any, Any, Any]:
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    input_ids = torch.randint(
        0,
        248320,
        (args.batch_size, args.sequence_length),
        generator=generator,
        dtype=torch.long,
    )
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    if args.padded:
        for batch_index in range(1, args.batch_size):
            valid = max(args.sequence_length - batch_index * max(args.sequence_length // 4, 1), 3)
            mask[batch_index, valid:] = False
            input_ids[batch_index, valid:] = 0
    hidden = torch.randn(
        (args.batch_size, args.sequence_length, hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    return hidden.to(device), input_ids.to(device), mask.to(device)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.batch_size <= 0 or args.sequence_length < 3 or args.chunk_tokens <= 0:
        raise ValueError("positive batch/chunk and sequence-length>=3 are required")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch

    from twen.model_loading import load_qwen35_mtp, read_qwen_text_config
    from twen.utils import atomic_write_text

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    checkpoint = Path(args.checkpoint).resolve()
    config = read_qwen_text_config(checkpoint)
    embedding, head = _load_vocabulary(checkpoint, torch=torch, device=device)
    mtp = load_qwen35_mtp(
        checkpoint,
        dtype=torch.bfloat16,
        device=str(device),
        trainable=False,
    ).eval()
    base_hidden, input_ids, attention_mask = _make_inputs(
        args,
        hidden_size=int(config["hidden_size"]),
        torch=torch,
        device=device,
    )
    runs = {
        "eager": _run_backend(
            "eager",
            mtp=mtp,
            embedding=embedding,
            head=head,
            base_hidden=base_hidden,
            input_ids=input_ids,
            attention_mask=attention_mask,
            chunk_tokens=args.chunk_tokens,
            gradient=True,
            sdpa_kernel_name=None,
            torch=torch,
        ),
        "sdpa_flash": _run_backend(
            "sdpa",
            mtp=mtp,
            embedding=embedding,
            head=head,
            base_hidden=base_hidden,
            input_ids=input_ids,
            attention_mask=attention_mask,
            chunk_tokens=args.chunk_tokens,
            gradient=True,
            sdpa_kernel_name=None,
            torch=torch,
        ),
        "sdpa_math": _run_backend(
            "sdpa",
            mtp=mtp,
            embedding=embedding,
            head=head,
            base_hidden=base_hidden,
            input_ids=input_ids,
            attention_mask=attention_mask,
            chunk_tokens=args.chunk_tokens,
            gradient=True,
            sdpa_kernel_name="math",
            torch=torch,
        ),
        "legacy_none": _run_backend(
            None,
            mtp=mtp,
            embedding=embedding,
            head=head,
            base_hidden=base_hidden,
            input_ids=input_ids,
            attention_mask=attention_mask,
            chunk_tokens=args.chunk_tokens,
            gradient=False,
            sdpa_kernel_name=None,
            torch=torch,
        ),
    }
    pair_mask = mtp.shifted_attention_mask(attention_mask).cpu()
    token_mask = attention_mask.cpu()
    eager = runs["eager"]
    sdpa = runs["sdpa_flash"]
    sdpa_math = runs["sdpa_math"]
    legacy = runs["legacy_none"]

    def compare(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
        loss_error = abs(float(left["loss"]) - float(right["loss"]))
        return {
            "hidden_all": _error(left["hidden"], right["hidden"]),
            "hidden_valid": _error(left["hidden"], right["hidden"], mask=pair_mask),
            "sampled_logits": _error(left["sampled_logits"], right["sampled_logits"]),
            "loss_absolute_error": loss_error,
            "loss_relative_error": loss_error / max(abs(float(right["loss"])), 1e-12),
            "gradient_all": _error(left["gradient"], right["gradient"]),
            "gradient_valid": _error(left["gradient"], right["gradient"], mask=token_mask),
        }

    eager_sdpa = compare(eager, sdpa)
    eager_math = compare(eager, sdpa_math)
    flash_math = compare(sdpa, sdpa_math)
    legacy_sdpa = {
        "hidden_valid": _error(legacy["hidden"], sdpa["hidden"], mask=pair_mask),
        "sampled_logits": _error(legacy["sampled_logits"], sdpa["sampled_logits"]),
        "loss_absolute_error": abs(legacy["loss"] - sdpa["loss"]),
        "loss_relative_error": abs(legacy["loss"] - sdpa["loss"])
        / max(abs(float(sdpa["loss"])), 1e-12),
    }
    tolerance = {
        "loss_relative": 5e-4,
        "hidden_and_logit_normalized_rms": 2e-2,
        "gradient_normalized_rms": 2.5e-2,
        "cosine_minimum": 0.9997,
    }
    accepted = bool(
        eager_sdpa["loss_relative_error"] <= tolerance["loss_relative"]
        and eager_sdpa["hidden_valid"]["normalized_rms"]
        <= tolerance["hidden_and_logit_normalized_rms"]
        and eager_sdpa["sampled_logits"]["normalized_rms"]
        <= tolerance["hidden_and_logit_normalized_rms"]
        and eager_sdpa["gradient_valid"]["normalized_rms"]
        <= tolerance["gradient_normalized_rms"]
        and eager_sdpa["hidden_valid"]["cosine"] >= tolerance["cosine_minimum"]
        and eager_sdpa["sampled_logits"]["cosine"] >= tolerance["cosine_minimum"]
        and eager_sdpa["gradient_valid"]["cosine"] >= tolerance["cosine_minimum"]
    )
    root = Path(__file__).resolve().parents[1]
    report = {
        "schema_version": 1,
        "kind": "qwen35_mtp_attention_backend_ab",
        "created_at": datetime.now(UTC).isoformat(),
        "ok": accepted,
        "no_optimizer_created": True,
        "no_optimizer_steps": True,
        "optimizer_step_calls": 0,
        "parameter_update": False,
        "same_loaded_mtp_instance_across_backends": True,
        "same_embedding_and_lm_head_across_backends": True,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "padded": args.padded,
        "valid_tokens": int(attention_mask.sum()),
        "valid_pairs": int(pair_mask.sum()),
        "checkpoint": str(checkpoint),
        "backends": {name: _without_tensors(result) for name, result in runs.items()},
        "comparison": {
            "eager_vs_sdpa_flash": eager_sdpa,
            "eager_vs_sdpa_math": eager_math,
            "sdpa_flash_vs_sdpa_math": flash_math,
            "legacy_none_vs_sdpa_flash": legacy_sdpa,
        },
        "acceptance_tolerances": tolerance,
        "diagnosis": {
            "legacy_none_mask_dispatch": "unsupported backend early-exits mask creation to None",
            "legacy_none_attention_dispatch": "AttentionInterface falls back to eager, which ignores is_causal",
            "consequence": "legacy path is bidirectional; canonical eager and SDPA are causal",
        },
        "environment": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "source": {
            "mtp_sha256": _sha256(root / "src/twen/modeling/mtp.py"),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write_text(args.output, rendered)
    print(rendered, end="")
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
