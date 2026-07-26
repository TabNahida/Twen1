#!/usr/bin/env python3
"""Generate authenticated greedy continuations from a dense transfer checkpoint.

This is inference-only: it creates no optimizer, performs no backward pass, and
compares the trained candidate branch with the same model in shared-only mode.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from twen.config import load_train_config
from twen.evaluation import (
    _configure_candidate_mode,
    _load_inference_evaluation_checkpoint,
)
from twen.io.offline import enforce_offline_environment
from twen.preflight import PreflightReport, compute_batch_geometry, run_training_preflight
from twen.runtime.checkpoint import CheckpointManager
from twen.source_identity import twen_source_tree_sha256
from twen.training.builder import build_transfer_model
from twen.utils import atomic_write_json, sha256_file

DEFAULT_PROMPTS = (
    "人工智能的发展改变软件工程的主要原因是",
    'def binary_search(values, target):\n    """Return the index of target or -1."""\n',
    "The key advantage of mixture-of-experts models is",
)


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _report_from_completed_evaluation(
    *,
    evaluation_dir: str,
    config_path: str,
    checkpoint_path: str,
    config: Any,
) -> PreflightReport:
    """Reuse a just-completed evaluation's authenticated full preflight.

    A production v2 preflight authenticates hundreds of GiB of KD tensors.  The
    completed NLL evaluation already performed that exact check immediately
    before loading this checkpoint.  Reusing its immutable PLAN avoids hashing
    the training corpus a second time for a forward-only smoke sample.
    """

    root = Path(evaluation_dir).resolve()
    manifest_path = root / "manifest.json"
    complete_path = root / "COMPLETE"
    plan_path = root / "PLAN.json"
    if not all(path.is_file() for path in (manifest_path, complete_path, plan_path)):
        raise ValueError(f"evaluation evidence is incomplete: {root}")
    if complete_path.read_text(encoding="ascii").strip() != sha256_file(manifest_path):
        raise ValueError("evaluation COMPLETE does not authenticate manifest.json")
    manifest = _read_json_object(manifest_path)
    plan = _read_json_object(plan_path)
    if (
        manifest.get("kind") != "twen_nll_evaluation"
        or manifest.get("plan_sha256") != sha256_file(plan_path)
        or manifest.get("plan_fingerprint") != plan.get("plan_fingerprint")
    ):
        raise ValueError("evaluation manifest/PLAN lineage is invalid")

    resolved_config = Path(config_path).resolve()
    lineage = plan.get("checkpoint_inference_lineage")
    if not isinstance(lineage, dict):
        raise ValueError("evaluation PLAN has no checkpoint lineage")
    if (
        Path(str(plan.get("config_path"))).resolve() != resolved_config
        or lineage.get("archived_config_sha256") != sha256_file(resolved_config)
    ):
        raise ValueError("evaluation evidence belongs to a different config")

    manager = CheckpointManager(config.checkpoint.output_dir, rank=0, world_size=1)
    resolved_checkpoint = manager.resolve(checkpoint_path)
    metadata = manager.inspect(resolved_checkpoint)
    if (
        Path(str(plan.get("checkpoint"))).resolve() != resolved_checkpoint.resolve()
        or plan.get("checkpoint_complete_sha256")
        != sha256_file(resolved_checkpoint / "COMPLETE")
        or manifest.get("checkpoint_state") != plan.get("checkpoint_state")
    ):
        raise ValueError("evaluation evidence belongs to a different checkpoint")

    current_source_tree = twen_source_tree_sha256()
    if lineage.get("current_source_tree_sha256") != current_source_tree:
        raise ValueError(
            "Twen source changed after evaluation; omit --evaluation-dir to run a fresh preflight"
        )
    calibration = lineage.get("calibration_artifacts")
    if not isinstance(calibration, dict) or not calibration:
        raise ValueError("evaluation evidence has no calibration fingerprints")
    data_fingerprint = metadata.get("data_fingerprint")
    config_fingerprint = plan.get("config_fingerprint")
    if not isinstance(data_fingerprint, str) or not isinstance(config_fingerprint, str):
        raise ValueError("evaluation evidence has incomplete preflight fingerprints")
    return PreflightReport(
        config_fingerprint=config_fingerprint,
        data_fingerprint=data_fingerprint,
        source_tree_sha256=current_source_tree,
        batch=compute_batch_geometry(config, 1),
        checked_paths=(str(root),),
        calibration_fingerprints=tuple(
            (str(name), str(digest)) for name, digest in calibration.items()
        ),
    )


def sample(args: argparse.Namespace) -> dict[str, Any]:
    enforce_offline_environment()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    prompts = tuple(args.prompt or DEFAULT_PROMPTS)
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("at least one non-empty prompt is required")

    import torch
    from transformers import AutoTokenizer, GenerationConfig

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))

    config = load_train_config(args.config)
    if config.stage != "dense-oracle":
        raise ValueError("dense sampling requires a dense-oracle config")
    if args.evaluation_dir:
        report = _report_from_completed_evaluation(
            evaluation_dir=args.evaluation_dir,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            config=config,
        )
        preflight_source = "authenticated_completed_evaluation"
    else:
        report = run_training_preflight(config, world_size=1)
        preflight_source = "fresh_full_training_preflight"
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
    tokenizer = AutoTokenizer.from_pretrained(
        config.sources.tokenizer.local_path,
        local_files_only=True,
    )
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    generation_config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.max_new_tokens,
        eos_token_id=None,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        pad_token_id=pad_token_id,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        renormalize_logits=False,
        remove_invalid_values=False,
        use_cache=True,
    )
    built.model.eval()
    outputs: dict[str, list[dict[str, Any]]] = {}
    for role in ("candidate", "shared"):
        _configure_candidate_mode(
            built.transfer_modules,
            role=role,
            top_k=config.architecture.top_k,
        )
        role_outputs = []
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {name: value.to(args.device) for name, value in encoded.items()}
            input_length = int(encoded["input_ids"].shape[1])
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(args.device))
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast(
                device_type="cuda" if args.device.startswith("cuda") else "cpu",
                dtype=dtype,
                enabled=args.device.startswith("cuda"),
            ):
                generated = built.model.generate(
                    **encoded,
                    generation_config=generation_config,
                    use_cache=True,
                )
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(args.device))
            elapsed = time.perf_counter() - started
            continuation_ids = [
                int(value) for value in generated[0, input_length:].cpu().tolist()
            ]
            continuation = tokenizer.decode(
                continuation_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            role_outputs.append(
                {
                    "prompt": prompt,
                    "continuation": continuation,
                    "continuation_token_ids": continuation_ids,
                    "continuation_tokens": len(continuation_ids),
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": len(continuation_ids) / elapsed,
                }
            )
        outputs[role] = role_outputs

    result = {
        "schema_version": 1,
        "kind": "twen_dense_checkpoint_greedy_samples",
        "no_optimizer_created": True,
        "no_optimizer_steps": True,
        "no_backward": True,
        "config": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "preflight_config_fingerprint": report.config_fingerprint,
        "preflight_data_fingerprint": report.data_fingerprint,
        "preflight_source": preflight_source,
        "evaluation_evidence": (
            str(Path(args.evaluation_dir).resolve()) if args.evaluation_dir else None
        ),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_complete_sha256": sha256_file(checkpoint / "COMPLETE"),
        "checkpoint_state": {
            key: checkpoint_metadata.get(key)
            for key in ("global_step", "committed_tokens", "kind", "tag")
        },
        "checkpoint_inference_lineage": checkpoint_lineage,
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name(torch.device(args.device))
            if args.device.startswith("cuda")
            else None
        ),
        "dtype": str(dtype),
        "max_new_tokens": args.max_new_tokens,
        "greedy": True,
        "outputs": outputs,
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(torch.device(args.device)) / 1024**3
            if args.device.startswith("cuda")
            else None
        ),
    }
    atomic_write_json(Path(args.output), result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--evaluation-dir",
        help="reuse an authenticated completed NLL evaluation's full preflight evidence",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    result = sample(_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
