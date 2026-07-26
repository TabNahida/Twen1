from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

_ANALYZER_PATH = Path(__file__).parents[1] / "scripts" / "analyze_dense_training.py"
_SPEC = importlib.util.spec_from_file_location("analyze_dense_training", _ANALYZER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load analyzer from {_ANALYZER_PATH}")
analyzer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analyzer)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_json_bytes(row) for row in rows))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(
    path: Path,
    *,
    prefix: str,
    lengths: tuple[int, ...],
) -> tuple[dict[str, Any], dict[str, str]]:
    sources = ("alpha_corpus", "beta_corpus", "gamma_corpus")
    shards = []
    source_by_shard = {}
    for index, length in enumerate(lengths):
        shard_id = f"{prefix}-{index:06d}"
        source = sources[index % len(sources)]
        source_by_shard[shard_id] = source
        shards.append(
            {
                "shard_id": shard_id,
                "sequence_count": length,
                "source_path": f"/fixture/{source}-{index:06d}.jsonl",
            }
        )
    value = {"schema_version": 1, "shards": shards}
    _write_json(path, value)
    return value, source_by_shard


def _field_summary(mean: float, *, spread: float = 1.0) -> dict[str, float]:
    return {
        "last": mean,
        "max": mean + spread,
        "mean": mean,
        "min": mean - spread,
        "p95": mean + spread,
    }


def _gpu_bucket(
    started: datetime,
    ended: datetime,
    *,
    samples: int,
    power: float,
    utilization: float,
    vram: float,
    temperature: float,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "kind": "twen_gpu_telemetry_aggregate",
        "source": "nvidia-smi",
        "device_index": 0,
        "raw_sample_interval_ms": 100,
        "sample_count": samples,
        "available_sample_count": samples,
        "unavailable_sample_count": 0,
        "window_started_at_utc": started.isoformat(),
        "window_ended_at_utc": ended.isoformat(),
        "fields": {
            "power_draw_w": _field_summary(power, spread=10.0),
            "power_limit_w": _field_summary(600.0, spread=0.0),
            "gpu_utilization_percent": _field_summary(utilization, spread=5.0),
            "memory_utilization_percent": _field_summary(40.0, spread=4.0),
            "vram_used_mib": _field_summary(vram, spread=100.0),
            "temperature_c": _field_summary(temperature, spread=2.0),
        },
    }


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _build_completed_run(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    run_dir = root / "runs" / "dense-fixture"
    run_dir.mkdir(parents=True)

    primary_path = root / "artifacts" / "primary" / "manifest.json"
    cooldown_path = root / "artifacts" / "cooldown" / "manifest.json"
    primary_manifest, primary_sources = _manifest(
        primary_path,
        prefix="primary",
        lengths=(11, 13, 17, 19, 23, 29),
    )
    cooldown_manifest, cooldown_sources = _manifest(
        cooldown_path,
        prefix="cooldown",
        lengths=(12, 14, 16, 18, 20, 22),
    )

    seed = 3407
    global_batch_samples = 4
    tokens_per_step = 1_000_000
    steps = 240
    cooldown_start_tokens = 210_000_000
    peak_lr = 2e-4
    primary_ids = [row["shard_id"] for row in primary_manifest["shards"]]
    primary_lengths = [row["sequence_count"] for row in primary_manifest["shards"]]
    cooldown_ids = [row["shard_id"] for row in cooldown_manifest["shards"]]
    cooldown_lengths = [row["sequence_count"] for row in cooldown_manifest["shards"]]
    replay = analyzer._SourceReplayCursor(
        analyzer._PhaseCursor(primary_ids, primary_lengths, seed=seed),
        analyzer._PhaseCursor(cooldown_ids, cooldown_lengths, seed=seed),
        cooldown_start_tokens=cooldown_start_tokens,
    )

    metrics: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    source_ntp = {"alpha_corpus": 0.08, "beta_corpus": 0.31, "gamma_corpus": 0.56}
    source_kd = {"alpha_corpus": 0.22, "beta_corpus": 0.05, "gamma_corpus": 0.38}
    source_mtp = {"alpha_corpus": 0.03, "beta_corpus": 0.18, "gamma_corpus": 0.09}
    for step in range(1, steps + 1):
        phase = replay.phase
        source_map = cooldown_sources if phase == "cooldown" else primary_sources
        shard_ids = replay.plan(global_batch_samples)
        counts = Counter(source_map[shard_id] for shard_id in shard_ids)
        fractions = {source: count / global_batch_samples for source, count in counts.items()}
        tokens = step * tokens_per_step
        progress = tokens / 100_000_000
        noise = 0.003 * math.sin(step * 0.41)
        ntp = (
            1.45
            + sum(fractions.get(source, 0.0) * value for source, value in source_ntp.items())
            - 0.040 * progress
            + noise
        )
        teacher_kd = (
            0.42
            + sum(fractions.get(source, 0.0) * value for source, value in source_kd.items())
            - 0.018 * progress
            + noise * 0.4
        )
        mtp = (
            0.95
            + sum(fractions.get(source, 0.0) * value for source, value in source_mtp.items())
            - 0.011 * progress
            + noise * 0.2
        )
        anchor_kl = 0.08 + 0.01 * fractions.get("gamma_corpus", 0.0) - 0.002 * progress
        hidden_alignment = 0.21 + 0.02 * fractions.get("beta_corpus", 0.0)
        loss = ntp + teacher_kd + 0.1 * (mtp + anchor_kl + hidden_alignment)
        if step <= 10:
            lr = peak_lr * step / 10
        elif phase == "primary":
            lr = peak_lr
        else:
            cooldown_step = step - 210
            lr = peak_lr * (1.0 - 0.9 * cooldown_step / 30)
        grad_norm = 0.58 + 0.25 * fractions.get("gamma_corpus", 0.0) + 0.04 * (step % 5)
        alignment = step % 20 == 0
        compute_seconds = 190.0 if alignment else 95.0
        wall_seconds = compute_seconds + 1.5
        metric = {
            "step": step,
            "tokens": tokens,
            "tokens_this_step": tokens_per_step,
            "data_phase": phase,
            "lr": lr,
            "loss": loss,
            "ntp": ntp,
            "teacher_kd": teacher_kd,
            "mtp": mtp,
            "anchor_kl": anchor_kl,
            "hidden_alignment": hidden_alignment,
            "grad_norm": grad_norm,
        }
        metrics.append(metric)
        telemetry.append(
            {
                "step": step,
                "tokens": tokens,
                "tokens_this_step": tokens_per_step,
                "data_phase": phase,
                "hidden_alignment_step": alignment,
                "compute_step_seconds": compute_seconds,
                "wall_clock_step_seconds": wall_seconds,
                "compute_tokens_per_second": tokens_per_step / compute_seconds,
                "wall_clock_tokens_per_second": tokens_per_step / wall_seconds,
                "data_wait_fraction": 0.01,
                "gpu_peak_allocated_gib": 24.0 + (0.5 if alignment else 0.0),
                "gpu_peak_reserved_gib": 25.0 + (0.5 if alignment else 0.0),
            }
        )
        replay.commit(samples=global_batch_samples, tokens=tokens_per_step)

    session_start = datetime(2026, 1, 2, tzinfo=UTC)
    session_end = session_start + timedelta(minutes=1)
    session_id = "fixture-session"
    checkpoint_name = "step-000000000240-milestone-complete"
    checkpoint = run_dir / checkpoint_name
    (checkpoint / "state").mkdir(parents=True)
    state_path = checkpoint / "state" / "part.bin"
    state_path.write_bytes(b"authenticated fixture state")
    metadata = {
        "global_step": steps,
        "committed_tokens": steps * tokens_per_step,
        "kind": "milestone",
        "tag": "complete",
        "run_id": "dense-fixture",
        "stage": "dense-oracle",
        "data_cursor": {
            "global_sample_index": steps * global_batch_samples,
            "global_token_index": steps * tokens_per_step,
            "extra": {"kind": "deterministic-two-phase-quality-cooldown"},
        },
    }
    metadata_path = checkpoint / "metadata.json"
    _write_json(metadata_path, metadata)
    checkpoint_manifest = {
        "algorithm": "sha256",
        "version": 1,
        "files": {
            "metadata.json": _sha256(metadata_path),
            "state/part.bin": _sha256(state_path),
        },
    }
    checkpoint_manifest_path = checkpoint / "manifest.json"
    _write_json(checkpoint_manifest_path, checkpoint_manifest)
    (checkpoint / "COMPLETE").write_text(
        f"{_sha256(checkpoint_manifest_path)}\n",
        encoding="utf-8",
    )

    config = {
        "schema_version": 1,
        "run_id": "dense-fixture",
        "track": "base",
        "stage": "dense-oracle",
        "data": {
            "manifest_path": str(primary_path.relative_to(root)),
            "manifest_sha256": _sha256(primary_path),
            "quality_cooldown_manifest_path": str(cooldown_path.relative_to(root)),
            "quality_cooldown_manifest_sha256": _sha256(cooldown_path),
            "quality_cooldown_start_tokens": cooldown_start_tokens,
            "shuffle_seed": seed,
        },
        "losses": {
            "ntp": 1.0,
            "teacher_kd": 1.0,
            "mtp": 0.1,
            "anchor_kl": 0.1,
            "hidden_alignment": 0.1,
            "dense_oracle": 0.0,
            "router_supervision": 0.0,
            "load_balance": 0.0,
            "router_z": 0.0,
        },
        "optimizer": {
            "max_tokens": steps * tokens_per_step,
            "warmup_tokens": 10_000_000,
            "lr_schedule": "warmup-stable-decay",
            "decay_tokens": 30_000_000,
            "min_lr_ratio": 0.1,
            "grad_clip_norm": 0.75,
            "adapter_lr": peak_lr,
            "lora_lr": peak_lr,
            "scale_lr": 0.001,
        },
    }
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True),
        encoding="utf-8",
    )
    events = [
        {
            "event": "session_start",
            "session_id": session_id,
            "timestamp_utc": session_start.isoformat(),
            "gradient_accumulation_steps": 1,
            "micro_batch_size": global_batch_samples,
            "world_size": 1,
            "gpu_total_memory_bytes": 32 * analyzer.GIB,
        },
        {
            "event": "train_complete",
            "session_id": session_id,
            "timestamp_utc": session_end.isoformat(),
            "step": steps,
            "tokens": steps * tokens_per_step,
            "checkpoint": str(checkpoint.relative_to(root)),
        },
    ]
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_jsonl(run_dir / "metrics.jsonl", metrics)
    _write_jsonl(run_dir / "telemetry.jsonl", telemetry)
    _write_json(
        run_dir / "rank0-session.json",
        {
            "schema_version": 1,
            "session_id": session_id,
            "run_id": "dense-fixture",
            "stage": "dense-oracle",
            "status": "completed",
            "started_at_utc": session_start.isoformat(),
            "ended_at_utc": session_end.isoformat(),
        },
    )
    (run_dir / "latest").write_text(f"{checkpoint_name}\n", encoding="utf-8")

    dashboard = root / ".twen" / "dashboard"
    outside_before = _gpu_bucket(
        session_start - timedelta(seconds=20),
        session_start - timedelta(seconds=10),
        samples=5,
        power=20.0,
        utilization=0.0,
        vram=500.0,
        temperature=30.0,
    )
    selected_rotated = _gpu_bucket(
        session_start + timedelta(seconds=10),
        session_start + timedelta(seconds=20),
        samples=4,
        power=100.0,
        utilization=80.0,
        vram=20_000.0,
        temperature=60.0,
    )
    selected_current = _gpu_bucket(
        session_start + timedelta(seconds=20),
        session_start + timedelta(seconds=30),
        samples=6,
        power=200.0,
        utilization=90.0,
        vram=22_000.0,
        temperature=70.0,
    )
    unavailable_current = {
        "schema_version": 2,
        "kind": "twen_gpu_telemetry_aggregate",
        "source": "nvidia-smi",
        "device_index": 0,
        "raw_sample_interval_ms": 100,
        "sample_count": 2,
        "available_sample_count": 0,
        "unavailable_sample_count": 2,
        "window_started_at_utc": (session_start + timedelta(seconds=30)).isoformat(),
        "window_ended_at_utc": (session_start + timedelta(seconds=40)).isoformat(),
        "fields": {},
    }
    outside_after = _gpu_bucket(
        session_end + timedelta(seconds=10),
        session_end + timedelta(seconds=10),
        samples=1,
        power=30.0,
        utilization=0.0,
        vram=600.0,
        temperature=31.0,
    )
    _write_jsonl(
        dashboard / "gpu-telemetry.jsonl.1",
        [outside_before, selected_rotated],
    )
    _write_jsonl(
        dashboard / "gpu-telemetry.jsonl",
        [selected_current, unavailable_current, outside_after],
    )
    return run_dir, state_path


def _convert_to_prepared_text_source_mix(run_dir: Path) -> None:
    config_path = run_dir / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"]["mode"] = "prepared-text"
    for key in (
        "quality_cooldown_manifest_path",
        "quality_cooldown_manifest_sha256",
        "quality_cooldown_start_tokens",
    ):
        config["data"].pop(key, None)
    config["losses"].update(
        {
            "ntp": 1.0,
            "teacher_kd": 0.0,
            "mtp": 0.1,
            "anchor_kl": 0.0,
            "hidden_alignment": 0.0,
        }
    )
    config["optimizer"].update(
        {
            "adapter_optimizer": "muon",
            "muon_adjust_lr_fn": "match_rms_adamw",
            "muon_momentum": 0.95,
            "muon_nesterov": True,
            "muon_ns_steps": 5,
        }
    )
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=True),
        encoding="utf-8",
    )

    metrics_path = run_dir / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    cumulative = {
        "alpha_corpus": 0,
        "beta_corpus": 0,
        "gamma_corpus": 0,
    }
    for row in metrics:
        step = int(row["step"])
        row["data_phase"] = "primary"
        row["loss"] = row["ntp"] + 0.1 * row["mtp"]
        for key in ("teacher_kd", "anchor_kl", "hidden_alignment"):
            row.pop(key, None)
        row["lr/adapters"] = row["lr"]
        row["lr/scale"] = row["lr"] * 3
        row["lr_adjustment_factor/adapters"] = 12.8
        row["lr_adjusted/adapters"] = row["lr"] * 12.8
        alpha = 295_000 + (step * 7_919) % 13_000
        beta = 245_000 + (step * 3_571) % 11_000
        source_tokens = {
            "alpha_corpus": alpha,
            "beta_corpus": beta,
            "gamma_corpus": int(row["tokens_this_step"]) - alpha - beta,
        }
        for source, tokens in source_tokens.items():
            cumulative[source] += tokens
            row[f"source_tokens_this_step/{source}"] = tokens
            row[f"source_tokens/{source}"] = cumulative[source]
    _write_jsonl(metrics_path, metrics)

    telemetry_path = run_dir / "telemetry.jsonl"
    telemetry = [
        json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in telemetry:
        row["data_phase"] = "primary"
    _write_jsonl(telemetry_path, telemetry)

    events_path = run_dir / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    session_start = next(row for row in events if row["event"] == "session_start")
    session_start.update(
        {
            "data_mode": "prepared-text",
            "source_mix_enabled": True,
            "source_mix_algorithm": "fixture-token-deficit-source-mix",
            "source_mix_effective_basis_points": {
                "alpha_corpus": 3000,
                "beta_corpus": 2500,
                "gamma_corpus": 4500,
            },
            "source_mix_dataset_fingerprint": "fixture-source-mix",
            "source_map_sha256": "f" * 64,
        }
    )
    _write_jsonl(events_path, events)

    checkpoint = run_dir / (run_dir / "latest").read_text(encoding="utf-8").strip()
    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["data_cursor"]["extra"] = {
        "kind": "deterministic-source-mix",
        "committed_samples_by_source": {
            "alpha_corpus": 288,
            "beta_corpus": 240,
            "gamma_corpus": 432,
        },
        "committed_tokens_by_source": cumulative,
    }
    _write_json(metadata_path, metadata)
    checkpoint_manifest_path = checkpoint / "manifest.json"
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    checkpoint_manifest["files"]["metadata.json"] = _sha256(metadata_path)
    _write_json(checkpoint_manifest_path, checkpoint_manifest)
    (checkpoint / "COMPLETE").write_text(
        f"{_sha256(checkpoint_manifest_path)}\n",
        encoding="utf-8",
    )


def _convert_to_prepared_text_phase_source_mix(run_dir: Path) -> None:
    config_path = run_dir / "resolved_config.yaml"
    original_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cooldown_fields = {
        key: original_config["data"][key]
        for key in (
            "quality_cooldown_manifest_path",
            "quality_cooldown_manifest_sha256",
            "quality_cooldown_start_tokens",
        )
    }
    _convert_to_prepared_text_source_mix(run_dir)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"].update(cooldown_fields)
    project_root = run_dir.parents[1]
    cooldown_path = project_root / cooldown_fields["quality_cooldown_manifest_path"]
    cooldown_manifest = json.loads(cooldown_path.read_text(encoding="utf-8"))
    quality_sources = ("quality_a", "quality_b")
    for index, shard in enumerate(cooldown_manifest["shards"]):
        source = quality_sources[index % len(quality_sources)]
        shard["source_path"] = f"/fixture/{source}-{index:06d}.jsonl"
        shard["token_count"] = 1_000_000
    _write_json(cooldown_path, cooldown_manifest)
    config["data"]["quality_cooldown_manifest_sha256"] = _sha256(cooldown_path)
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=True),
        encoding="utf-8",
    )

    primary_weights = {
        "alpha_corpus": 3_000,
        "beta_corpus": 2_500,
        "gamma_corpus": 4_500,
    }
    cooldown_weights = {"quality_a": 4_000, "quality_b": 6_000}
    all_sources = sorted(set(primary_weights) | set(cooldown_weights))
    cooldown_start_tokens = int(cooldown_fields["quality_cooldown_start_tokens"])
    metrics_path = run_dir / "metrics.jsonl"
    metrics = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    cumulative = dict.fromkeys(all_sources, 0)
    phase_tokens = {
        "primary": dict.fromkeys(all_sources, 0),
        "cooldown": dict.fromkeys(all_sources, 0),
    }
    for row in metrics:
        for key in tuple(row):
            if key.startswith(analyzer.SOURCE_TOKENS_STEP_PREFIX) or key.startswith(
                analyzer.SOURCE_TOKENS_TOTAL_PREFIX
            ):
                row.pop(key)
        phase = (
            "cooldown"
            if int(row["tokens"]) > cooldown_start_tokens
            else "primary"
        )
        row["data_phase"] = phase
        weights = cooldown_weights if phase == "cooldown" else primary_weights
        for source in all_sources:
            tokens = weights.get(source, 0) * 100
            cumulative[source] += tokens
            phase_tokens[phase][source] += tokens
            row[f"{analyzer.SOURCE_TOKENS_STEP_PREFIX}{source}"] = tokens
            row[f"{analyzer.SOURCE_TOKENS_TOTAL_PREFIX}{source}"] = cumulative[
                source
            ]
    _write_jsonl(metrics_path, metrics)

    telemetry_path = run_dir / "telemetry.jsonl"
    telemetry = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in telemetry:
        row["data_phase"] = (
            "cooldown"
            if int(row["tokens"]) > cooldown_start_tokens
            else "primary"
        )
    _write_jsonl(telemetry_path, telemetry)

    def phase_contract(
        phase: str,
        weights: dict[str, int],
    ) -> dict[str, Any]:
        return {
            "enabled": True,
            "algorithm": "token-deficit-corrected-source-mix-bp-v2",
            "source_map_sha256": ("a" if phase == "primary" else "b") * 64,
            "dataset_fingerprint": ("c" if phase == "primary" else "d") * 64,
            "basis_points": weights,
            "lineage_basis_points": weights,
            "effective_basis_points": weights,
            "weight_override": False,
            "seed": 3407,
        }

    primary_contract = phase_contract("primary", primary_weights)
    cooldown_contract = phase_contract("cooldown", cooldown_weights)
    events_path = run_dir / "events.jsonl"
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    session_start = next(row for row in events if row["event"] == "session_start")
    session_start.update(
        {
            "source_mix_algorithm": primary_contract["algorithm"],
            "source_mix_effective_basis_points": primary_weights,
            "source_mix_dataset_fingerprint": primary_contract[
                "dataset_fingerprint"
            ],
            "source_map_sha256": primary_contract["source_map_sha256"],
            "source_mix": {
                **primary_contract,
                "cooldown_start_tokens": cooldown_start_tokens,
                "phases": {
                    "primary": primary_contract,
                    "cooldown": cooldown_contract,
                },
            },
        }
    )
    _write_jsonl(events_path, events)

    checkpoint = run_dir / (run_dir / "latest").read_text(encoding="utf-8").strip()
    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    primary_samples = {
        source: weight * 840 // 10_000
        for source, weight in primary_weights.items()
    }
    cooldown_samples = {
        source: weight * 120 // 10_000
        for source, weight in cooldown_weights.items()
    }
    committed_samples = dict.fromkeys(all_sources, 0)
    for source, count in (*primary_samples.items(), *cooldown_samples.items()):
        committed_samples[source] += count
    metadata["data_cursor"]["extra"] = {
        "kind": "deterministic-source-mix-quality-cooldown",
        "committed_samples_by_source": committed_samples,
        "committed_tokens_by_source": cumulative,
        "phase_committed_samples_by_source": {
            "primary": {
                source: primary_samples.get(source, 0)
                for source in all_sources
            },
            "cooldown": {
                source: cooldown_samples.get(source, 0)
                for source in all_sources
            },
        },
        "phase_committed_tokens_by_source": phase_tokens,
    }
    _write_json(metadata_path, metadata)
    checkpoint_manifest_path = checkpoint / "manifest.json"
    checkpoint_manifest = json.loads(
        checkpoint_manifest_path.read_text(encoding="utf-8")
    )
    checkpoint_manifest["files"]["metadata.json"] = _sha256(metadata_path)
    _write_json(checkpoint_manifest_path, checkpoint_manifest)
    (checkpoint / "COMPLETE").write_text(
        f"{_sha256(checkpoint_manifest_path)}\n",
        encoding="utf-8",
    )


def test_completed_run_analysis_is_authenticated_read_only_and_atomic(tmp_path: Path) -> None:
    run_dir, _state_path = _build_completed_run(tmp_path)
    before = _snapshot(run_dir)

    analysis = analyzer.analyze_dense_training(run_dir)

    assert analysis["terminal_validation"]["passed"] is True
    assert analysis["terminal_validation"]["authenticated_payload_bytes"] > 0
    assert analysis["integrity"]["source_replay_matches_checkpoint_cursor"] is True
    assert analysis["source_replay"]["validation"] == {
        "all_phase_predictions_match": True,
        "final_samples": 960,
        "final_tokens": 240_000_000,
        "final_phase": "cooldown",
    }
    assert analysis["phases"]["primary_stable"]["rolling"]["window_100"]["loss"]
    assert analysis["phases"]["primary_stable"]["rolling"]["window_152"]["loss"]
    assert analysis["phases"]["primary_decay"] is None
    assert analysis["phases"]["analysis_phase"]["kind"] == "post_warmup_primary"
    assert analysis["source_adjusted"]["phase"]["kind"] == "post_warmup_primary"
    assert (
        analysis["source_adjusted"]["phase"]["points"]
        == analysis["phases"]["analysis_phase"]["points"]
    )
    slope = analysis["source_adjusted"]["common_slopes"]["loss"]["coefficients"]["tokens_per_100m"]
    assert slope["estimate"] < -0.05
    assert analysis["clipping"]["configured_threshold"] == pytest.approx(0.75)
    assert analysis["loss_formula"]["passed"] is True

    gpu = analysis["performance"]["dashboard_gpu_last_session"]
    assert gpu["available"] is True
    assert gpu["selection"]["selected_bucket_count"] == 3
    assert gpu["selection"]["rows_by_input"] == {
        "dashboard_gpu_telemetry_current": 2,
        "dashboard_gpu_telemetry_rotated": 1,
    }
    assert gpu["samples"] == {
        "sample_count": 12,
        "available_sample_count": 10,
        "unavailable_sample_count": 2,
        "raw_sample_interval_ms_values": [100],
    }
    assert gpu["fields"]["power_draw_w"]["weighted_mean"] == pytest.approx(160.0)
    assert gpu["fields"]["power_draw_w"]["bucket_mean_nearest_rank_p95"] == 200.0
    assert gpu["coverage"]["trailing_gap_seconds"] == pytest.approx(20.0)
    assert gpu["coverage"]["coverage_fraction_of_session"] == pytest.approx(0.02)
    for key in ("dashboard_gpu_telemetry_rotated", "dashboard_gpu_telemetry_current"):
        identity = analysis["inputs"][key]
        assert identity["size"] > 0
        assert len(identity["sha256"]) == 64

    output_dir = tmp_path / "analysis"
    outputs = analyzer.write_analysis(analysis, output=output_dir, run_dir=run_dir)
    json_path = Path(outputs["json"])
    markdown_path = Path(outputs["markdown_zh_cn"])
    assert json.loads(json_path.read_text(encoding="utf-8")) == analysis
    assert "NaN" not in json_path.read_text(encoding="utf-8")
    report = markdown_path.read_text(encoding="utf-8")
    assert "Dashboard GPU telemetry" in report
    assert "last rank0 session" in report
    assert "primary cosine decay" not in report
    assert "![Loss、NTP 与 MTP](charts/training_loss.svg)" in report
    assert "## 后续版本建议 (v4)" in report
    assert "## v3 建议" not in report
    chart_paths = sorted((output_dir / "charts").glob("*.svg"))
    assert {path.name for path in chart_paths} == {
        "gpu_memory.svg",
        "gpu_power.svg",
        "gpu_utilization.svg",
        "gradient_norm.svg",
        "learning_rate.svg",
        "source_token_mix.svg",
        "throughput.svg",
        "training_loss.svg",
    }
    manifest_path = output_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["kind"] == "twen_dense_training_analysis_bundle"
    assert set(manifest["files"]) == {
        "REPORT.zh-CN.md",
        "analysis.json",
        *(f"charts/{path.name}" for path in chart_paths),
    }
    for relative, identity in manifest["files"].items():
        path = output_dir / relative
        assert identity == {"size": path.stat().st_size, "sha256": _sha256(path)}
    assert (output_dir / "COMPLETE").read_text(encoding="utf-8").strip() == (_sha256(manifest_path))
    assert "next_version_priority" in analysis["interpretation"]
    assert "v3_priority" not in analysis["interpretation"]
    assert not list(output_dir.glob(".*.tmp"))
    assert _snapshot(run_dir) == before

    with pytest.raises(analyzer.AnalysisError, match="inside run-dir"):
        analyzer.write_analysis(
            analysis,
            output=run_dir / "forbidden-analysis",
            run_dir=run_dir,
        )
    assert not (run_dir / "forbidden-analysis").exists()


def test_prepared_text_ntp_mtp_run_uses_dynamic_metrics_and_logged_source_tokens(
    tmp_path: Path,
) -> None:
    run_dir, _state_path = _build_completed_run(tmp_path)
    _convert_to_prepared_text_source_mix(run_dir)

    analysis = analyzer.analyze_dense_training(run_dir)

    assert analysis["integrity"]["active_loss_components"] == ["ntp", "mtp"]
    assert analysis["integrity"]["required_metric_fields"] == [
        "loss",
        "ntp",
        "mtp",
        "grad_norm",
        "lr",
    ]
    assert analysis["loss_formula"]["formula"] == "1*ntp + 0.1*mtp"
    assert analysis["loss_formula"]["max_abs_error"] == 0
    replay = analysis["source_replay"]
    assert replay["contract"]["composition_source"] == ("logged_source_tokens_this_step")
    assert replay["contract"]["composition_unit"] == "valid_tokens"
    assert replay["sources"] == [
        "alpha_corpus",
        "beta_corpus",
        "gamma_corpus",
    ]
    assert replay["validation"]["source_tokens_sum_to_step_tokens"] is True
    assert replay["validation"]["cumulative_source_tokens_exact"] is True
    assert (
        sum(replay["token_mix"]["committed_tokens_by_source"].values())
        == (analysis["run"]["terminal_tokens"])
    )
    assert analysis["source_adjusted"]["model_kind"] == ("mixed_batch_centered_source_fractions")
    assert analysis["source_adjusted"]["source_interaction_identifiability"]["available"] is False
    assert analysis["lr_dose"]["optimizer"]["adapter_optimizer"] == "muon"
    assert analysis["lr_dose"]["optimizer"]["muon"]["observed_adjustment_factor"] == pytest.approx(
        12.8
    )

    output = tmp_path / "prepared-text-analysis"
    analyzer.write_analysis(analysis, output=output, run_dir=run_dir)
    report = (output / "REPORT.zh-CN.md").read_text(encoding="utf-8")
    assert "loss = 1*ntp + 0.1*mtp" in report
    assert "batch loss 不能拆成可信的逐来源 NLL" in report
    assert "| alpha_corpus | 30.00%" in report


@pytest.mark.parametrize(
    ("cooldown_weights", "expected_global"),
    [
        (
            {"quality": 10_000},
            {"alpha": 3_000.0, "beta": 2_000.0, "quality": 5_000.0},
        ),
        (
            {"alpha": 2_000, "beta": 8_000},
            {"alpha": 4_000.0, "beta": 6_000.0},
        ),
    ],
)
def test_logged_source_mix_cooldown_uses_phase_contracts_and_union_zero_fill(
    tmp_path: Path,
    cooldown_weights: dict[str, int],
    expected_global: dict[str, float],
) -> None:
    primary_weights = {"alpha": 6_000, "beta": 4_000}
    phase_weights = {
        "primary": primary_weights,
        "cooldown": cooldown_weights,
    }
    all_sources = sorted(set(primary_weights) | set(cooldown_weights))

    def phase_contract(
        phase: str,
        weights: dict[str, int],
    ) -> dict[str, Any]:
        return {
            "enabled": True,
            "algorithm": "token-deficit-corrected-source-mix-bp-v2",
            "source_map_sha256": ("a" if phase == "primary" else "b") * 64,
            "dataset_fingerprint": ("c" if phase == "primary" else "d") * 64,
            "basis_points": weights,
            "lineage_basis_points": weights,
            "effective_basis_points": weights,
            "weight_override": False,
            "seed": 73,
        }

    primary_contract = phase_contract("primary", primary_weights)
    cooldown_contract = phase_contract("cooldown", cooldown_weights)
    events = [
        {
            "event": "session_start",
            "gradient_accumulation_steps": 1,
            "micro_batch_size": 2,
            "world_size": 1,
            "source_mix_enabled": True,
            "source_mix_algorithm": primary_contract["algorithm"],
            "source_mix_effective_basis_points": primary_weights,
            "source_mix_dataset_fingerprint": primary_contract[
                "dataset_fingerprint"
            ],
            "source_map_sha256": primary_contract["source_map_sha256"],
            "source_mix": {
                **primary_contract,
                "cooldown_start_tokens": 20_000,
                "phases": {
                    "primary": primary_contract,
                    "cooldown": cooldown_contract,
                },
            },
        }
    ]
    cumulative = dict.fromkeys(all_sources, 0)
    metrics: list[dict[str, Any]] = []
    for step, phase in enumerate(
        ("primary", "primary", "cooldown", "cooldown"),
        start=1,
    ):
        step_tokens = {
            source: phase_weights[phase].get(source, 0)
            for source in all_sources
        }
        row: dict[str, Any] = {
            "step": step,
            "tokens": step * 10_000,
            "tokens_this_step": 10_000,
            "data_phase": phase,
        }
        for source, tokens in step_tokens.items():
            cumulative[source] += tokens
            row[f"source_tokens_this_step/{source}"] = tokens
            row[f"source_tokens/{source}"] = cumulative[source]
        metrics.append(row)

    def manifest(sources: Sequence[str], phase: str) -> dict[str, Any]:
        return {
            "shards": [
                {
                    "shard_id": f"{phase}-{index}",
                    "sequence_count": 100,
                    "token_count": 400_000,
                    "source_path": f"/fixture/{source}-000000.jsonl",
                }
                for index, source in enumerate(sources)
            ]
        }

    primary_manifest = manifest(sorted(primary_weights), "primary")
    cooldown_manifest = manifest(sorted(cooldown_weights), "cooldown")
    replay = analyzer._logged_source_token_replay(
        manifest=primary_manifest,
        manifest_path=tmp_path / "primary.json",
        cooldown_manifest=cooldown_manifest,
        cooldown_manifest_path=tmp_path / "cooldown.json",
        cooldown_start_tokens=20_000,
        metrics=metrics,
        events=events,
    )

    assert replay["sources"] == all_sources
    assert replay["contract"]["expected_basis_points"] == expected_global
    assert replay["contract"]["cooldown_start_tokens"] == 20_000
    assert replay["validation"]["phase_source_contracts_exact"] is True
    phase_mix = replay["token_mix"]["phases"]
    assert phase_mix["primary"]["observed_basis_points"] == {
        source: float(primary_weights.get(source, 0))
        for source in all_sources
    }
    assert phase_mix["cooldown"]["observed_basis_points"] == {
        source: float(cooldown_weights.get(source, 0))
        for source in all_sources
    }
    assert sum(replay["token_mix"]["committed_tokens_by_source"].values()) == (
        40_000
    )


def test_prepared_text_phase_source_mix_cooldown_report_closes_end_to_end(
    tmp_path: Path,
) -> None:
    run_dir, _state_path = _build_completed_run(tmp_path)
    _convert_to_prepared_text_phase_source_mix(run_dir)

    analysis = analyzer.analyze_dense_training(run_dir)

    replay = analysis["source_replay"]
    assert replay["validation"]["phase_source_contracts_exact"] is True
    assert replay["contract"]["cooldown_start_tokens"] == 210_000_000
    assert replay["contract"]["quality_cooldown_manifest"].endswith(
        "artifacts/cooldown/manifest.json"
    )
    assert replay["contract"]["expected_basis_points"] == {
        "alpha_corpus": 2_625.0,
        "beta_corpus": 2_187.5,
        "gamma_corpus": 3_937.5,
        "quality_a": 500.0,
        "quality_b": 750.0,
    }
    assert set(replay["token_mix"]["phases"]) == {"primary", "cooldown"}
    assert analysis["cooldown_lr_separation"]["boundary"]["window_steps"] == 20
    assert analysis["data_consumption"]["committed_tokens_by_source"] is not None

    output = tmp_path / "phase-source-mix-analysis"
    analyzer.write_analysis(analysis, output=output, run_dir=run_dir)
    assert (output / "analysis.json").is_file()
    assert (output / "REPORT.zh-CN.md").is_file()


def test_prepared_text_missing_nonzero_mtp_metric_fails_closed(tmp_path: Path) -> None:
    run_dir, _state_path = _build_completed_run(tmp_path)
    _convert_to_prepared_text_source_mix(run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    metrics[0].pop("mtp")
    _write_jsonl(metrics_path, metrics)

    with pytest.raises(analyzer.AnalysisError, match=r"metrics\[1\]\.mtp"):
        analyzer.analyze_dense_training(run_dir)


def test_phase_rows_separate_decay_but_use_all_post_warmup_primary_for_regression() -> None:
    metrics = [
        {"step": 1, "tokens": 1, "lr": 0.1, "data_phase": "primary"},
        {"step": 2, "tokens": 2, "lr": 0.2, "data_phase": "primary"},
        {"step": 3, "tokens": 3, "lr": 0.2, "data_phase": "primary"},
        {"step": 4, "tokens": 4, "lr": 0.15, "data_phase": "primary"},
        {"step": 5, "tokens": 5, "lr": 0.1, "data_phase": "primary"},
        {"step": 6, "tokens": 6, "lr": 0.05, "data_phase": "cooldown"},
    ]

    phases, analysis_phase = analyzer._phase_rows(metrics, warmup_tokens=1)

    assert [row["step"] for row in phases["warmup"]] == [1]
    assert [row["step"] for row in phases["primary_stable"]] == [2, 3]
    assert [row["step"] for row in phases["primary_decay"]] == [4, 5]
    assert [row["step"] for row in phases["cooldown"]] == [6]
    assert [row["step"] for row in analysis_phase] == [2, 3, 4, 5]


def test_incomplete_run_fails_closed(tmp_path: Path) -> None:
    run_dir, _state_path = _build_completed_run(tmp_path)
    rank_session_path = run_dir / "rank0-session.json"
    rank_session = json.loads(rank_session_path.read_text(encoding="utf-8"))
    rank_session["status"] = "running"
    _write_json(rank_session_path, rank_session)

    with pytest.raises(analyzer.AnalysisError, match="not terminal completed"):
        analyzer.analyze_dense_training(run_dir)


def test_checkpoint_payload_tampering_fails_closed(tmp_path: Path) -> None:
    run_dir, state_path = _build_completed_run(tmp_path)
    state_path.write_bytes(b"tampered fixture state")

    with pytest.raises(analyzer.AnalysisError, match="file hash mismatch"):
        analyzer.analyze_dense_training(run_dir)


def test_checkpoint_payload_symlink_fails_closed(tmp_path: Path) -> None:
    run_dir, state_path = _build_completed_run(tmp_path)
    target = state_path.with_name("target.bin")
    target.write_bytes(state_path.read_bytes())
    state_path.unlink()
    state_path.symlink_to(target.name)

    with pytest.raises(analyzer.AnalysisError, match="contains symlink"):
        analyzer.analyze_dense_training(run_dir)
