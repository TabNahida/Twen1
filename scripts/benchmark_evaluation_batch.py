#!/usr/bin/env python3
"""Read-only candidate NLL batch-size A/B on fixed authenticated validation shards."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from twen.config import load_train_config
from twen.data import validate_prepared_corpus
from twen.evaluation import (
    _configure_candidate_mode,
    _load_inference_evaluation_checkpoint,
    _nll_sum,
)
from twen.preflight import run_training_preflight
from twen.training.builder import build_transfer_model
from twen.utils import atomic_write_json, atomic_write_text, sha256_file


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_fixed_shards(manifest_path: Path, count: int) -> tuple[Any, dict[str, Any]]:
    import torch

    prepared = validate_prepared_corpus(manifest_path)
    if count <= 0 or count > len(prepared.shards):
        raise ValueError(f"shard count must be in [1, {len(prepared.shards)}]")
    tensors: dict[str, list[Any]] = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    shard_identity = []
    for entry in prepared.shards[:count]:
        path = manifest_path.parent / entry.path / "tokens.safetensors"
        loaded = load_file(str(path), device="cpu")
        for name in tensors:
            tensors[name].append(loaded[name])
        shard_identity.append(
            {
                "shard_id": entry.shard_id,
                "tensors_sha256": entry.tensors_sha256,
                "sequence_count": entry.sequence_count,
                "token_count": entry.token_count,
            }
        )
    combined = {name: torch.cat(parts, dim=0) for name, parts in tensors.items()}
    return combined, {
        "dataset_fingerprint": prepared.dataset_fingerprint,
        "shards": shard_identity,
        "sequence_count": int(combined["input_ids"].shape[0]),
        "input_token_count": sum(int(item["token_count"]) for item in shard_identity),
    }


def _run_case(
    model: Any,
    tensors: dict[str, Any],
    *,
    batch_size: int,
    repeat: int,
    device: str,
    dtype: Any,
) -> dict[str, Any]:
    import torch

    sequence_count = int(tensors["input_ids"].shape[0])
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    def forward(start: int, end: int) -> tuple[float, int]:
        input_ids = tensors["input_ids"][start:end].to(device)
        attention_mask = tensors["attention_mask"][start:end].to(device)
        labels = tensors["labels"][start:end].to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device_type,
            dtype=dtype,
            enabled=dtype == torch.bfloat16,
        ):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            nll, predicted_tokens = _nll_sum(outputs.logits, labels)
        result = float(nll), predicted_tokens
        del outputs, nll, input_ids, attention_mask, labels
        return result

    try:
        # Warm the exact physical batch shape without counting it.
        forward(0, min(batch_size, sequence_count))
        if device_type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        nll_sum = 0.0
        predicted_tokens = 0
        microbatches = 0
        for start in range(0, sequence_count, batch_size):
            end = min(start + batch_size, sequence_count)
            batch_nll, batch_tokens = forward(start, end)
            nll_sum += batch_nll
            predicted_tokens += batch_tokens
            microbatches += 1
        if device_type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        result = {
            "status": "ok",
            "batch_size": batch_size,
            "repeat": repeat,
            "sequence_count": sequence_count,
            "microbatches": microbatches,
            "predicted_tokens": predicted_tokens,
            "nll_sum": nll_sum,
            "mean_nll": nll_sum / predicted_tokens,
            "elapsed_seconds": elapsed,
            "predicted_tokens_per_second": predicted_tokens / elapsed,
            "peak_allocated_bytes": (
                torch.cuda.max_memory_allocated() if device_type == "cuda" else None
            ),
            "peak_reserved_bytes": (
                torch.cuda.max_memory_reserved() if device_type == "cuda" else None
            ),
        }
    except torch.OutOfMemoryError as exc:
        result = {
            "status": "oom",
            "batch_size": batch_size,
            "repeat": repeat,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "peak_allocated_bytes": (
                torch.cuda.max_memory_allocated() if device_type == "cuda" else None
            ),
            "peak_reserved_bytes": (
                torch.cuda.max_memory_reserved() if device_type == "cuda" else None
            ),
        }
    finally:
        gc.collect()
        if device_type == "cuda":
            torch.cuda.empty_cache()
    return result


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for batch_size in sorted({int(case["batch_size"]) for case in cases}):
        selected = [
            case
            for case in cases
            if int(case["batch_size"]) == batch_size and case["status"] == "ok"
        ]
        failures = [
            case
            for case in cases
            if int(case["batch_size"]) == batch_size and case["status"] != "ok"
        ]
        result[str(batch_size)] = {
            "successful_repeats": len(selected),
            "failed_repeats": len(failures),
            "status": "ok" if selected and not failures else "partial" if selected else "failed",
            "median_predicted_tokens_per_second": (
                statistics.median(case["predicted_tokens_per_second"] for case in selected)
                if selected
                else None
            ),
            "mean_nll_values": [case["mean_nll"] for case in selected],
            "max_peak_allocated_bytes": (
                max(int(case["peak_allocated_bytes"]) for case in selected) if selected else None
            ),
            "max_peak_reserved_bytes": (
                max(int(case["peak_reserved_bytes"]) for case in selected) if selected else None
            ),
        }
    first = result.get("1", {})
    second = result.get("2", {})
    one_speed = first.get("median_predicted_tokens_per_second")
    two_speed = second.get("median_predicted_tokens_per_second")
    result["batch2_over_batch1_speedup"] = (
        float(two_speed) / float(one_speed) if one_speed and two_speed else None
    )
    all_ok = [case for case in cases if case["status"] == "ok"]
    nll_values = [float(case["mean_nll"]) for case in all_ok]
    result["nll_max_absolute_difference"] = (
        max(nll_values) - min(nll_values) if nll_values else None
    )
    result["nll_consistent_at_1e_5"] = bool(
        nll_values and max(nll_values) - min(nll_values) <= 1e-5
    )
    return result


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is unavailable")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    config = load_train_config(args.config)
    report = run_training_preflight(config, world_size=1)
    prepared_path = Path(args.prepared_manifest).resolve()
    tensors, corpus = _load_fixed_shards(prepared_path, args.shards)
    if sha256_file(prepared_path) == config.data.manifest_sha256:
        raise ValueError("benchmark must use the independent validation corpus, not training data")
    dtype = torch.bfloat16 if config.runtime.bf16 else torch.float32
    built = build_transfer_model(config, device=args.device, dtype=dtype)
    checkpoint, checkpoint_metadata, checkpoint_lineage = (
        _load_inference_evaluation_checkpoint(
            built.model,
            config_path=args.config,
            config=config,
            report=report,
            checkpoint_path=args.checkpoint,
        )
    )
    _configure_candidate_mode(
        built.transfer_modules,
        role="candidate",
        top_k=config.architecture.top_k,
    )
    built.model.eval()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"benchmark output must be a fresh empty directory: {output}")
    script_path = Path(__file__).resolve()
    order = [int(value) for value in args.order.split(",")]
    if not order or any(value not in {1, 2} for value in order):
        raise ValueError("--order must contain only comma-separated 1 and 2 values")
    plan = {
        "schema_version": 1,
        "kind": "twen_eval_batch_ab_plan",
        "no_optimizer": True,
        "no_backward": True,
        "role": "candidate",
        "config": str(Path(args.config).resolve()),
        "current_preflight_fingerprint": report.config_fingerprint,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_complete_sha256": sha256_file(checkpoint / "COMPLETE"),
        "checkpoint_state": {
            key: checkpoint_metadata.get(key)
            for key in ("global_step", "committed_tokens", "kind", "tag")
        },
        "checkpoint_inference_lineage": checkpoint_lineage,
        "prepared_manifest": str(prepared_path),
        "prepared_manifest_sha256": sha256_file(prepared_path),
        "corpus": corpus,
        "order": order,
        "dtype": str(dtype),
        "device": args.device,
        "device_name": torch.cuda.get_device_name(torch.device(args.device)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "fla_tilelang": os.environ.get("FLA_TILELANG"),
        "script": str(script_path),
        "script_sha256": sha256_file(script_path),
    }
    plan["plan_fingerprint"] = _canonical_sha256(plan)
    atomic_write_json(output / "PLAN.json", plan)
    cases = []
    repeat_count: dict[int, int] = {1: 0, 2: 0}
    for batch_size in order:
        repeat_count[batch_size] += 1
        case = _run_case(
            built.model,
            tensors,
            batch_size=batch_size,
            repeat=repeat_count[batch_size],
            device=args.device,
            dtype=dtype,
        )
        cases.append(case)
        print(json.dumps({"event": "eval_batch_case", **case}, sort_keys=True), flush=True)
    result = {
        "schema_version": 1,
        "kind": "twen_eval_batch_ab_result",
        "no_optimizer_steps": True,
        "no_backward": True,
        "plan_fingerprint": plan["plan_fingerprint"],
        "plan_sha256": sha256_file(output / "PLAN.json"),
        "cases": cases,
        "aggregate": _aggregate(cases),
    }
    atomic_write_json(output / "result.json", result)
    manifest = {
        "schema_version": 1,
        "kind": "twen_eval_batch_ab_bundle",
        "files": {
            "PLAN.json": sha256_file(output / "PLAN.json"),
            "result.json": sha256_file(output / "result.json"),
        },
    }
    atomic_write_json(output / "MANIFEST.json", manifest)
    atomic_write_text(output / "COMPLETE", sha256_file(output / "MANIFEST.json") + "\n")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--order", default="1,2,2,1")
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = benchmark(args)
    speedup = result["aggregate"].get("batch2_over_batch1_speedup")
    if speedup is not None and not math.isfinite(float(speedup)):
        raise RuntimeError("benchmark produced a non-finite speedup")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
