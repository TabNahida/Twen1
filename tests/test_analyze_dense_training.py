from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from collections import Counter
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
        fractions = {
            source: count / global_batch_samples for source, count in counts.items()
        }
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
        grad_norm = (
            0.58
            + 0.25 * fractions.get("gamma_corpus", 0.0)
            + 0.04 * (step % 5)
        )
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
    slope = analysis["source_adjusted"]["common_slopes"]["loss"]["coefficients"][
        "tokens_per_100m"
    ]
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
    assert not list(output_dir.glob(".*.tmp"))
    assert _snapshot(run_dir) == before

    with pytest.raises(analyzer.AnalysisError, match="inside run-dir"):
        analyzer.write_analysis(
            analysis,
            output=run_dir / "forbidden-analysis",
            run_dir=run_dir,
        )
    assert not (run_dir / "forbidden-analysis").exists()


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
