"""Read-only comparison of two completed training runs for recovery acceptance."""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import math
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import load_train_config
from .preflight import run_training_preflight
from .runtime.checkpoint import CheckpointManager
from .runtime.state import RNGState
from .training.builder import build_transfer_model
from .training.engine import _build_optimizer
from .training.stateful import OptimizerState, TokenLRScheduler, TrainableModelState


def _update_digest(digest: Any, value: Any) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: repr(item)):
            _update_digest(digest, key)
            _update_digest(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode() + b"\0")
        for item in value:
            _update_digest(digest, item)
        return
    if dataclasses.is_dataclass(value):
        _update_digest(digest, dataclasses.asdict(value))
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("cannot hash a non-finite checkpoint scalar")
    digest.update(type(value).__name__.encode() + b"\0")
    digest.update(repr(value).encode("utf-8"))
    digest.update(b"\0")


def _stable_digest(value: Any) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def _metrics_digest(run_dir: Path, committed_step: int) -> tuple[str, int]:
    records = []
    path = run_dir / "metrics.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                step = int(record["step"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if step <= committed_step:
                records.append(record)
    records.sort(key=lambda item: int(item["step"]))
    return _stable_digest(records), len(records)


def _all_rank_runtime_digest(checkpoint: Path) -> str:
    """Hash every saved rank's cursor/RNG, not only the comparison process rank."""

    ranks: dict[str, Any] = {}
    for path in sorted((checkpoint / "runtime").glob("rank-*.pkl")):
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        trainer = dict(payload["trainer_state"])
        extra = dict(trainer.get("extra", {}))
        extra.pop("checkpoint_request_sequence", None)
        trainer["extra"] = extra
        ranks[path.stem] = {
            "saved_rank": int(payload["saved_rank"]),
            "trainer_state": trainer,
            "data_cursor": payload["data_cursor"],
            "rng_digest": RNGState.from_dict(payload["rng_state"]).digest(),
            "rollback_applied": bool(payload["rollback_applied"]),
        }
    if not ranks:
        raise ValueError(f"checkpoint has no per-rank runtime payloads: {checkpoint}")
    return _stable_digest(ranks)


def checkpoint_run_digest(config_path: str, checkpoint: str) -> dict[str, Any]:
    """Load a checkpoint into production templates and hash every resumable state."""

    import torch

    config = load_train_config(config_path)
    report = run_training_preflight(config, world_size=1)
    dtype = torch.bfloat16 if config.runtime.bf16 else torch.float32
    built = build_transfer_model(config, device="cpu", dtype=dtype)
    optimizer = _build_optimizer(config, built)
    scheduler = TokenLRScheduler(
        optimizer,
        warmup_tokens=config.optimizer.warmup_tokens,
        max_tokens=config.optimizer.max_tokens,
        lr_schedule=config.optimizer.lr_schedule,
        min_lr_ratio=config.optimizer.min_lr_ratio,
        decay_tokens=config.optimizer.decay_tokens,
    )
    model_state = TrainableModelState(built.model)
    optimizer_state = OptimizerState(built.model, optimizer)
    manager = CheckpointManager(config.checkpoint.output_dir, rank=0, world_size=1)
    loaded = manager.load(
        {
            "model": model_state,
            "optimizer": optimizer_state,
            "scheduler": scheduler,
        },
        checkpoint,
        expected_critical_fingerprint=report.config_fingerprint,
        expected_data_fingerprint=report.data_fingerprint,
        expected_run_id=config.run_id,
        expected_stage=config.stage,
        expected_global_batch_tokens=config.data.global_batch_tokens,
        restore_rng=False,
    )
    metrics_hash, metric_records = _metrics_digest(
        Path(config.checkpoint.output_dir), loaded.trainer_state.global_step
    )
    trainer_payload = loaded.trainer_state.to_dict()
    trainer_payload.get("extra", {}).pop("checkpoint_request_sequence", None)
    result = {
        "checkpoint": str(loaded.path),
        "model": _stable_digest(model_state.state_dict()),
        "optimizer": _stable_digest(optimizer_state.state_dict()),
        "scheduler": _stable_digest(scheduler.state_dict()),
        "trainer_state": _stable_digest(trainer_payload),
        "data_cursor": _stable_digest(loaded.data_cursor.to_dict()),
        "rng": loaded.rng_state.digest(),
        "all_rank_runtime": _all_rank_runtime_digest(loaded.path),
        "metrics": metrics_hash,
        "metric_records": metric_records,
        "global_step": loaded.trainer_state.global_step,
        "committed_tokens": loaded.trainer_state.committed_tokens,
    }
    del built, optimizer, scheduler, model_state, optimizer_state
    gc.collect()
    return result


def compare_checkpoint_runs(
    config_a: str,
    checkpoint_a: str,
    config_b: str,
    checkpoint_b: str,
) -> dict[str, Any]:
    first = checkpoint_run_digest(config_a, checkpoint_a)
    second = checkpoint_run_digest(config_b, checkpoint_b)
    compared = (
        "model",
        "optimizer",
        "scheduler",
        "trainer_state",
        "data_cursor",
        "rng",
        "all_rank_runtime",
        "metrics",
        "metric_records",
        "global_step",
        "committed_tokens",
    )
    differences = {
        key: {"a": first[key], "b": second[key]} for key in compared if first[key] != second[key]
    }
    return {"equivalent": not differences, "differences": differences, "a": first, "b": second}
