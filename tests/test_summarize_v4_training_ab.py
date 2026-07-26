from __future__ import annotations

import csv
import gzip
import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "summarize_v4_training_ab.py"
    spec = importlib.util.spec_from_file_location("summarize_v4_training_ab", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summary = _load_script()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _run_fixture(root: Path, *, micro_batch_size: int, wall_seconds: float) -> Path:
    root.mkdir()
    metrics = []
    telemetry = []
    for step in range(1, 13):
        metrics.append(
            {
                "step": step,
                "data_mode": "prepared-text",
                "tokens_this_step": 262_144,
                "tokens": step * 262_144,
                "ntp": 2.0,
                "mtp": 2.1,
                "loss": 2.21,
                "grad_norm": 0.5,
            }
        )
        telemetry.append(
            {
                "step": step,
                "compute_step_seconds": wall_seconds - 0.5,
                "wall_clock_step_seconds": wall_seconds,
                "gpu_peak_allocated_gib": 24.0 + micro_batch_size,
                "gpu_peak_reserved_gib": 25.0 + micro_batch_size,
            }
        )
    events = [
        {
            "event": "session_start",
            "data_mode": "prepared-text",
            "teacher_kd_enabled": False,
            "adapter_optimizer": "muon",
            "mtp_enabled": True,
            "world_size": 1,
            "micro_batch_size": micro_batch_size,
            "global_batch_tokens": 262_144,
            "gradient_accumulation_steps": 64 // micro_batch_size,
            "gpu_name": "fixture",
            "gpu_total_memory_bytes": 32 * 1024**3,
            "config_fingerprint": "a" * 64,
            "data_fingerprint": "b" * 64,
            "source_tree_sha256": "c" * 64,
        },
        {
            "event": "model_built",
            "hidden_teacher_enabled": False,
            "mtp_enabled": True,
            "mtp_trainable_parameters": 0,
        },
        {
            "event": "optimizer_built",
            "adapter_optimizer": "muon",
            "optimizer_bundle": True,
            "components": [
                {"optimizer": "Muon", "parameter_groups": ["adapters"]},
                {"optimizer": "AdamW", "parameter_groups": ["scale"]},
            ],
        },
    ]
    _write_jsonl(root / "metrics.jsonl", metrics)
    _write_jsonl(root / "telemetry.jsonl", telemetry)
    _write_jsonl(root / "events.jsonl", events)
    return root


def _write_power(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "power.draw [W]",
                "power.limit [W]",
                "utilization.gpu [%]",
                "utilization.memory [%]",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "t1",
                "power.draw [W]": "400",
                "power.limit [W]": "600",
                "utilization.gpu [%]": "90",
                "utilization.memory [%]": "70",
            }
        )
        writer.writerow(
            {
                "timestamp": "t2",
                "power.draw [W]": "500",
                "power.limit [W]": "600",
                "utilization.gpu [%]": "100",
                "utilization.memory [%]": "75",
            }
        )


def _write_trace(path: Path) -> None:
    payload = {
        "traceEvents": [
            {"ph": "X", "name": name, "dur": 1_000_000}
            for name in summary.REQUIRED_TRACE_SCOPES
        ]
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_complete_ab_report_selects_fastest_accepted_case(
    tmp_path: Path, capsys: object
) -> None:
    b1 = _run_fixture(tmp_path / "b1", micro_batch_size=1, wall_seconds=20.0)
    b2 = _run_fixture(tmp_path / "b2", micro_batch_size=2, wall_seconds=10.0)
    b1_power = tmp_path / "b1.csv"
    b2_power = tmp_path / "b2.csv"
    b1_trace = tmp_path / "b1.trace.json.gz"
    b2_trace = tmp_path / "b2.trace.json.gz"
    _write_power(b1_power)
    _write_power(b2_power)
    _write_trace(b1_trace)
    _write_trace(b2_trace)
    recovery = tmp_path / "recovery.json"
    recovery.write_text(json.dumps({"equivalent": True}), encoding="utf-8")
    recovery_run = tmp_path / "recovery-run"
    recovery_run.mkdir()
    _write_jsonl(
        recovery_run / "events.jsonl",
        [
            {"event": "session_start"},
            {
                "event": "checkpoint_complete",
                "reason": "stop-file",
            },
            {"event": "graceful_stop"},
            {"event": "session_start"},
            {"event": "resume"},
            {
                "event": "checkpoint_complete",
                "reason": "sigusr1",
            },
            {"event": "train_complete"},
        ],
    )

    exit_code = summary.main(
        [
            "--case",
            f"b1={b1}",
            "--case",
            f"b2={b2}",
            "--power",
            f"b1={b1_power}",
            "--power",
            f"b2={b2_power}",
            "--trace",
            f"b1={b1_trace}",
            "--trace",
            f"b2={b2_trace}",
            "--recovery-compare",
            str(recovery),
            "--recovery-run",
            str(recovery_run),
            "--require-power",
            "--require-trace",
            "--require-recovery",
        ]
    )

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert report["accepted"] is True
    assert report["training_started_by_auditor"] is False
    assert report["input_training_runs"] == 2
    assert report["recommendation"]["label"] == "b2"
    assert report["recommendation"]["micro_batch_size"] == 2
    assert report["cases"][0]["performance"]["measured_steps"] == 10
    assert report["cases"][0]["power"]["fields"]["power_draw_w"]["p95"] == 500.0
    assert report["cases"][0]["trace"]["complete"] is True
    assert report["recovery"]["equivalent"] is True
    assert report["recovery"]["stop_resume_sigusr1_complete"] is True


def test_oom_or_precommit_failure_is_reported_without_hiding_passing_case(
    tmp_path: Path, capsys: object
) -> None:
    b1 = _run_fixture(tmp_path / "b1", micro_batch_size=1, wall_seconds=20.0)
    b4 = tmp_path / "b4"
    b4.mkdir()
    _write_jsonl(
        b4 / "events.jsonl",
        [
            {
                "event": "train_failed",
                "error_type": "OutOfMemoryError",
                "error": "CUDA out of memory",
            }
        ],
    )

    exit_code = summary.main(
        [
            "--case",
            f"b1={b1}",
            "--case",
            f"b4={b4}",
        ]
    )

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert report["accepted"] is True
    assert report["all_cases_accepted"] is False
    assert report["recommendation"]["label"] == "b1"
    assert report["cases"][1]["accepted"] is False
    assert report["cases"][1]["event_counts"]["train_failed"] == 1
    assert report["cases"][1]["failure_events"][0]["error_type"] == "OutOfMemoryError"


def test_teacher_side_metric_and_missing_optional_evidence_reject_case(
    tmp_path: Path, capsys: object
) -> None:
    run = _run_fixture(tmp_path / "b1", micro_batch_size=1, wall_seconds=20.0)
    metrics = summary._read_jsonl(run / "metrics.jsonl")
    metrics[-1]["teacher_kd"] = 0.5
    _write_jsonl(run / "metrics.jsonl", metrics)

    exit_code = summary.main(
        [
            "--case",
            f"b1={run}",
            "--require-power",
            "--require-trace",
        ]
    )

    assert exit_code == 2
    report = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    reasons = report["cases"][0]["rejection_reasons"]
    assert "step 12 unexpectedly logs teacher_kd" in reasons
    assert "required power sample is absent" in reasons
    assert "required profiler scopes are absent" in reasons
    assert report["recommendation"]["available"] is False
