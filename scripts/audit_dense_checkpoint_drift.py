#!/usr/bin/env python3
"""Measure trainable-delta drift between dense DCP checkpoints on CPU.

The audit partially loads only ``model.*`` entries from an authenticated
checkpoint. It does not build the backbone/donor graph, initialize CUDA, or
create an optimizer.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any

from twen.utils import atomic_write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        type=Path,
        help="Candidate checkpoint; repeat to audit multiple checkpoints.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Atomically write the JSON audit while still rendering it to stdout.",
    )
    return parser


def _checkpoint_paths(path: Path) -> tuple[Path, Path]:
    resolved = path.expanduser().resolve()
    if resolved.name == "state":
        checkpoint = resolved.parent
        state = resolved
    else:
        checkpoint = resolved
        state = resolved / "state"
    if not checkpoint.is_dir() or not state.is_dir():
        raise ValueError(f"checkpoint/state directory does not exist: {resolved}")
    for name in ("manifest.json", "COMPLETE"):
        if not (checkpoint / name).is_file():
            raise ValueError(f"checkpoint is incomplete: missing {checkpoint / name}")
    return checkpoint, state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(state_path: Path) -> Any:
    from torch.distributed.checkpoint import FileSystemReader

    return FileSystemReader(state_path).read_metadata().state_dict_metadata


def _model_keys(metadata: Any) -> tuple[str, ...]:
    keys = tuple(sorted(key for key in metadata if key.startswith("model.")))
    if not keys:
        raise ValueError("checkpoint has no model-delta tensors")
    unexpected = [
        key for key in keys if ".adapters." not in key and not key.endswith("branch_scale")
    ]
    if unexpected:
        raise ValueError(f"unclassified trainable model entries: {unexpected[:3]}")
    return keys


def _load_model_state(state_path: Path, metadata: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    import torch
    import torch.distributed.checkpoint as dcp

    state = {
        key: torch.empty(
            tuple(metadata[key].size),
            dtype=metadata[key].properties.dtype,
            device="cpu",
        )
        for key in keys
    }
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="torch.distributed is disabled.*",
            category=UserWarning,
        )
        dcp.load(state, checkpoint_id=state_path)
    return state


def _group_drift(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    keys: tuple[str, ...],
) -> dict[str, int | float]:
    element_count = 0
    baseline_l2_squared = 0.0
    candidate_l2_squared = 0.0
    delta_l2_squared = 0.0
    dot = 0.0
    max_absolute_delta = 0.0
    for key in keys:
        baseline_tensor = baseline[key].double()
        candidate_tensor = candidate[key].double()
        delta = candidate_tensor - baseline_tensor
        element_count += baseline_tensor.numel()
        baseline_l2_squared += float((baseline_tensor * baseline_tensor).sum())
        candidate_l2_squared += float((candidate_tensor * candidate_tensor).sum())
        delta_l2_squared += float((delta * delta).sum())
        dot += float((baseline_tensor * candidate_tensor).sum())
        max_absolute_delta = max(max_absolute_delta, float(delta.abs().max()))
    return {
        "tensor_count": len(keys),
        "element_count": element_count,
        "baseline_rms": math.sqrt(baseline_l2_squared / element_count),
        "candidate_rms": math.sqrt(candidate_l2_squared / element_count),
        "delta_rms": math.sqrt(delta_l2_squared / element_count),
        "relative_l2": math.sqrt(delta_l2_squared / baseline_l2_squared),
        "cosine": dot / math.sqrt(baseline_l2_squared * candidate_l2_squared),
        "max_absolute_delta": max_absolute_delta,
    }


def _scale_summary(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    keys: tuple[str, ...],
) -> dict[str, int | float]:
    baseline_values = [float(baseline[key]) for key in keys]
    candidate_values = [float(candidate[key]) for key in keys]
    deltas = [
        candidate_value - baseline_value
        for baseline_value, candidate_value in zip(
            baseline_values,
            candidate_values,
            strict=True,
        )
    ]
    return {
        "baseline_min": min(baseline_values),
        "baseline_mean": sum(baseline_values) / len(baseline_values),
        "baseline_max": max(baseline_values),
        "candidate_min": min(candidate_values),
        "candidate_mean": sum(candidate_values) / len(candidate_values),
        "candidate_max": max(candidate_values),
        "decreased_count": sum(delta < 0 for delta in deltas),
        "unchanged_count": sum(delta == 0 for delta in deltas),
        "increased_count": sum(delta > 0 for delta in deltas),
        "minimum_delta": min(deltas),
        "maximum_delta": max(deltas),
    }


def _identity(checkpoint: Path) -> dict[str, str]:
    return {
        "path": str(checkpoint),
        "manifest_sha256": _sha256(checkpoint / "manifest.json"),
        "complete_sha256": _sha256(checkpoint / "COMPLETE"),
    }


def audit(baseline_path: Path, candidate_paths: list[Path]) -> dict[str, Any]:
    baseline_checkpoint, baseline_state_path = _checkpoint_paths(baseline_path)
    baseline_metadata = _metadata(baseline_state_path)
    keys = _model_keys(baseline_metadata)
    adapter_keys = tuple(key for key in keys if ".adapters." in key)
    scale_keys = tuple(key for key in keys if key.endswith("branch_scale"))
    baseline = _load_model_state(baseline_state_path, baseline_metadata, keys)

    candidates: list[dict[str, Any]] = []
    for candidate_path in candidate_paths:
        candidate_checkpoint, candidate_state_path = _checkpoint_paths(candidate_path)
        candidate_metadata = _metadata(candidate_state_path)
        candidate_keys = _model_keys(candidate_metadata)
        if candidate_keys != keys:
            raise ValueError(
                f"candidate model-delta inventory differs from baseline: {candidate_checkpoint}"
            )
        candidate = _load_model_state(candidate_state_path, candidate_metadata, keys)
        candidates.append(
            {
                **_identity(candidate_checkpoint),
                "adapter": _group_drift(
                    baseline,
                    candidate,
                    adapter_keys,
                ),
                "scale": _group_drift(
                    baseline,
                    candidate,
                    scale_keys,
                ),
                "scale_values": _scale_summary(
                    baseline,
                    candidate,
                    scale_keys,
                ),
            }
        )
        del candidate
        gc.collect()

    return {
        "kind": "twen_dense_checkpoint_trainable_drift_audit",
        "schema_version": 1,
        "execution": {
            "device": "cpu",
            "cuda_initialized": False,
            "model_built": False,
            "optimizer_created": False,
        },
        "baseline": _identity(baseline_checkpoint),
        "inventory": {
            "model_tensor_count": len(keys),
            "adapter_tensor_count": len(adapter_keys),
            "adapter_element_count": sum(baseline[key].numel() for key in adapter_keys),
            "scale_tensor_count": len(scale_keys),
            "scale_element_count": sum(baseline[key].numel() for key in scale_keys),
        },
        "candidates": candidates,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = audit(args.baseline, args.candidate)
    if args.output is not None:
        atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
