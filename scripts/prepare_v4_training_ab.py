#!/usr/bin/env python3
"""Prepare isolated v4 B1/B2/B4 optimizer-step benchmark configs.

This planner never imports CUDA, constructs a model, or launches training.  It
validates one already-published prepared-text/Muon config, writes independent
short-run configurations, and records the exact user-run commands in plan.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from twen.config import ConfigError, TrainConfig, dump_resolved_config, load_train_config
from twen.utils import atomic_write_text


class PlanError(ValueError):
    """The requested benchmark plan is unsafe or inconsistent."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--fork-from", required=True)
    parser.add_argument(
        "--micro-batches",
        default="1,2,4",
        help="comma-separated physical sequence batches (default 1,2,4)",
    )
    parser.add_argument(
        "--performance-tokens",
        type=int,
        default=4_000_000,
        help="short-run token lower bound; complete runs may overshoot by one global batch",
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=None,
        help="defaults to one logical global batch",
    )
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="capture a bounded trace crossing the first optimizer step",
    )
    return parser


def _micro_batches(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise PlanError("micro-batches must be comma-separated integers") from exc
    if not values or len(set(values)) != len(values):
        raise PlanError("micro-batches must be non-empty and contain no duplicates")
    if any(value not in {1, 2, 4} for value in values):
        raise PlanError("v4 capacity sweep is restricted to micro-batches 1,2,4")
    return values


def _validate_base(config: TrainConfig) -> None:
    if config.track != "base" or config.stage != "dense-oracle":
        raise PlanError("base config must be the Base dense-oracle stage")
    if config.data.mode != "prepared-text":
        raise PlanError("base config must use data.mode=prepared-text")
    if config.optimizer.adapter_optimizer != "muon":
        raise PlanError("base config must route adapters to Muon")
    if config.losses.ntp <= 0 or config.losses.mtp <= 0:
        raise PlanError("base config must enable both causal NTP and native MTP")
    teacher_losses = {
        name: float(getattr(config.losses, name))
        for name in ("teacher_kd", "anchor_kl", "hidden_alignment")
    }
    if any(value != 0.0 for value in teacher_losses.values()):
        raise PlanError(f"prepared-text benchmark enables teacher-side losses: {teacher_losses}")
    if config.losses.hidden_alignment_batch_fraction != 0.0:
        raise PlanError("prepared-text benchmark must disable hidden-alignment batches")
    if config.runtime.teacher_cpu_offload:
        raise PlanError("prepared-text benchmark must not enable teacher CPU offload")
    if config.runtime.activation_checkpointing_on_alignment_only:
        raise PlanError("prepared-text benchmark cannot use alignment-only checkpointing")


def _case_config(
    base: TrainConfig,
    *,
    micro_batch: int,
    output_dir: Path,
    run_root: Path,
    fork_from: Path,
    performance_tokens: int,
    warmup_tokens: int,
    profile: bool,
) -> tuple[TrainConfig, dict[str, Any]]:
    config = copy.deepcopy(base)
    sequence_length = config.data.max_sequence_length
    micro_tokens = micro_batch * sequence_length
    global_tokens = config.data.global_batch_tokens
    if global_tokens % micro_tokens != 0:
        raise PlanError(
            f"global batch {global_tokens} is not divisible by B{micro_batch} "
            f"microbatch tokens {micro_tokens}"
        )
    accumulation = global_tokens // micro_tokens
    if profile and accumulation < 3:
        raise PlanError("profile plan requires at least three accumulation microbatches")

    label = f"b{micro_batch}"
    config.run_id = f"{base.run_id}-optimizer-ab-{label}"
    config.data.micro_batch_size = micro_batch
    config.optimizer.max_tokens = performance_tokens
    config.optimizer.warmup_tokens = warmup_tokens
    config.checkpoint.output_dir = str(run_root / label)
    config.checkpoint.every_steps = 1_000
    config.checkpoint.every_minutes = 1_440.0
    config.checkpoint.keep_last = 1
    config.runtime.profile = profile
    if profile:
        # Work before profiler.step() N uses schedule action N-1.  The
        # optimizer runs after profiler.step() ``accumulation``, so action
        # ``accumulation`` must remain RECORD.  This window warms the
        # penultimate microbatch, records the final microbatch, the optimizer
        # step, and the start of the following accumulation before flushing.
        config.runtime.profile_wait_steps = accumulation - 2
        config.runtime.profile_warmup_steps = 1
        config.runtime.profile_active_steps = 4
    config.validate()

    config_path = output_dir / f"{label}.yaml"
    run_dir = Path(config.checkpoint.output_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise PlanError(f"benchmark run directory is not empty: {run_dir}")
    command = [
        "bash",
        "scripts/with_cuda_toolchain.sh",
        ".venv/bin/python",
        "-m",
        "twen.cli",
        "train",
        "--stage",
        "dense-oracle",
        "--config",
        str(config_path),
        "--progress",
        "always",
        "--resume",
        "none",
        "--fork-from",
        str(fork_from),
    ]
    return config, {
        "label": label,
        "config": str(config_path),
        "run_dir": str(run_dir),
        "micro_batch_size": micro_batch,
        "micro_batch_tokens": micro_tokens,
        "global_batch_tokens": global_tokens,
        "gradient_accumulation_steps": accumulation,
        "profile": {
            "enabled": profile,
            "step_unit": "microbatch",
            "wait_steps": config.runtime.profile_wait_steps,
            "warmup_steps": config.runtime.profile_warmup_steps,
            "active_steps": config.runtime.profile_active_steps,
            "captures_optimizer_step": profile,
        },
        "command": command,
    }


def build_plan(
    base_config: Path,
    output_dir: Path,
    run_root: Path,
    fork_from: Path,
    *,
    micro_batches: Sequence[int],
    performance_tokens: int,
    warmup_tokens: int | None,
    profile: bool,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PlanError(f"output directory is not empty: {output_dir}")
    if not fork_from.is_dir():
        raise PlanError(f"fork checkpoint directory does not exist: {fork_from}")
    if not (fork_from / "COMPLETE").is_file():
        raise PlanError(f"fork checkpoint has no COMPLETE marker: {fork_from}")
    config = load_train_config(base_config)
    _validate_base(config)
    if performance_tokens <= 0:
        raise PlanError("performance-tokens must be positive")
    selected_warmup = (
        config.data.global_batch_tokens if warmup_tokens is None else warmup_tokens
    )
    if selected_warmup < 0 or selected_warmup >= performance_tokens:
        raise PlanError("warmup-tokens must be in [0, performance-tokens)")

    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for micro_batch in micro_batches:
        derived, case = _case_config(
            config,
            micro_batch=micro_batch,
            output_dir=output_dir,
            run_root=run_root,
            fork_from=fork_from,
            performance_tokens=performance_tokens,
            warmup_tokens=selected_warmup,
            profile=profile,
        )
        target = Path(case["config"])
        if target.exists():
            raise PlanError(f"refusing to replace existing benchmark config: {target}")
        dump_resolved_config(derived, target)
        cases.append(case)

    return {
        "kind": "twen_v4_optimizer_step_ab_plan",
        "schema_version": 1,
        "training_started": False,
        "cuda_initialized": False,
        "base_config": str(base_config),
        "fork_from": str(fork_from),
        "performance_tokens": performance_tokens,
        "warmup_tokens": selected_warmup,
        "cases": cases,
        "notes": [
            "Only the user may execute the listed commands because they contain optimizer steps.",
            "Run directories must be absent or empty except for console.log before --resume none.",
            "The token lower bound may overshoot by one complete global optimizer batch.",
            "Drop at least the first two committed steps from throughput statistics.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    values = _micro_batches(args.micro_batches)
    plan = build_plan(
        Path(args.base_config),
        Path(args.output_dir),
        Path(args.run_root),
        Path(args.fork_from),
        micro_batches=values,
        performance_tokens=args.performance_tokens,
        warmup_tokens=args.warmup_tokens,
        profile=args.profile,
    )
    payload = json.dumps(
        plan,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    target = Path(args.output_dir) / "plan.json"
    atomic_write_text(target, payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PlanError, ConfigError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        raise SystemExit(2) from exc
