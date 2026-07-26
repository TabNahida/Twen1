#!/usr/bin/env python3
"""Audit short v4 optimizer-step runs without launching training.

The production training CLI owns model construction, backward, Muon/AdamW
updates, checkpoints, and recovery.  This script is deliberately read-only: it
joins the durable JSONL records from several short runs, optionally adds a
read-only nvidia-smi CSV and a bounded PyTorch profiler trace, and emits one
machine-readable A/B report.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

GIB = float(1024**3)
REQUIRED_TRACE_SCOPES = (
    "twen/data_prefetch_wait",
    "twen/h2d",
    "twen/forward",
    "twen/loss",
    "twen/mtp_forward",
    "twen/mtp_vocab_loss",
    "twen/backward",
    "twen/grad_clip",
    "twen/optimizer_step",
)


class AuditError(ValueError):
    """The supplied benchmark evidence is malformed or internally inconsistent."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        metavar="LABEL=RUN_DIR",
        help="repeat for every physical-microbatch run",
    )
    parser.add_argument(
        "--power",
        action="append",
        default=[],
        metavar="LABEL=CSV",
        help="optional read-only nvidia-smi sample for a matching case",
    )
    parser.add_argument(
        "--trace",
        action="append",
        default=[],
        metavar="LABEL=JSON_OR_JSON.GZ",
        help="optional bounded PyTorch Chrome trace for a matching case",
    )
    parser.add_argument(
        "--recovery-compare",
        default=None,
        help="optional JSON captured from `twen checkpoint compare`",
    )
    parser.add_argument(
        "--recovery-run",
        default=None,
        help="optional interrupted run whose events must prove STOP, resume, and SIGUSR1",
    )
    parser.add_argument(
        "--drop-first-steps",
        type=int,
        default=2,
        help="exclude compile/cache warmup optimizer steps from performance statistics",
    )
    parser.add_argument(
        "--min-measured-steps",
        type=int,
        default=10,
        help="minimum post-warmup optimizer steps required per accepted case",
    )
    parser.add_argument(
        "--expected-global-batch-tokens",
        type=int,
        default=262_144,
    )
    parser.add_argument(
        "--minimum-headroom-gib",
        type=float,
        default=2.0,
        help="minimum total-memory minus peak-reserved safety margin",
    )
    parser.add_argument(
        "--require-power",
        action="store_true",
        help="reject cases without a non-empty nvidia-smi sample",
    )
    parser.add_argument(
        "--require-trace",
        action="store_true",
        help="reject cases without every required twen/* profiler scope",
    )
    parser.add_argument(
        "--require-recovery",
        action="store_true",
        help="reject the overall gate unless checkpoint compare says equivalent=true",
    )
    parser.add_argument("--output", default=None)
    return parser


def _labeled_paths(values: Sequence[str], *, option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        label, separator, path = raw.partition("=")
        label = label.strip()
        path = path.strip()
        if not separator or not label or not path:
            raise AuditError(f"{option} entries must use LABEL=PATH")
        if label in result:
            raise AuditError(f"{option} contains duplicate label {label!r}")
        result[label] = Path(path)
    return result


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AuditError(f"cannot read JSONL {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuditError(f"{path}:{line_number} is invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise AuditError(f"{path}:{line_number} must contain a JSON object")
        rows.append(row)
    if not rows:
        raise AuditError(f"{path} contains no records")
    return rows


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AuditError(f"{label} must be finite")
    return result


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuditError(f"{label} must be an integer")
    return int(value)


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    if not values or not 0.0 < fraction <= 1.0:
        raise AuditError("percentile inputs are invalid")
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise AuditError("cannot summarize an empty distribution")
    finite = [_finite(value, label="distribution value") for value in values]
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "p50": _nearest_rank(finite, 0.50),
        "p95": _nearest_rank(finite, 0.95),
        "min": min(finite),
        "max": max(finite),
    }


def _strict_step_map(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        step = _integer(row.get("step"), label=f"{label}[{index}].step")
        if step in result:
            raise AuditError(f"{label} contains duplicate step {step}")
        result[step] = row
    ordered = sorted(result)
    if ordered != list(range(ordered[0], ordered[-1] + 1)):
        raise AuditError(f"{label} step sequence is not contiguous")
    return result


def _event_contract(events: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    sessions = [row for row in events if row.get("event") == "session_start"]
    optimizers = [row for row in events if row.get("event") == "optimizer_built"]
    models = [row for row in events if row.get("event") == "model_built"]
    if not sessions:
        reasons.append("events.jsonl has no session_start")
    if not optimizers:
        reasons.append("events.jsonl has no optimizer_built")
    if not models:
        reasons.append("events.jsonl has no model_built")

    for row in sessions:
        if row.get("data_mode") != "prepared-text":
            reasons.append("session data_mode is not prepared-text")
        if row.get("adapter_optimizer") != "muon":
            reasons.append("session adapter_optimizer is not muon")
        if row.get("teacher_kd_enabled") is not False:
            reasons.append("session unexpectedly enables teacher KD")
        if row.get("mtp_enabled") is not True:
            reasons.append("session does not enable native MTP")
        if _integer(row.get("world_size"), label="session.world_size") != 1:
            reasons.append("Muon benchmark world_size is not 1")

    for row in optimizers:
        if row.get("adapter_optimizer") != "muon":
            reasons.append("optimizer_built does not identify Muon")
        if row.get("optimizer_bundle") is not True:
            reasons.append("optimizer_built is not a Muon/AdamW bundle")
        components = row.get("components")
        if not isinstance(components, list):
            reasons.append("optimizer_built components are absent")
            continue
        groups = {
            group
            for component in components
            if isinstance(component, dict)
            for group in component.get("parameter_groups", [])
            if isinstance(group, str)
        }
        if groups != {"adapters", "scale"}:
            reasons.append(f"optimizer groups differ from adapters+scale: {sorted(groups)}")

    for row in models:
        if row.get("hidden_teacher_enabled") is not False:
            reasons.append("prepared-text model unexpectedly builds an online teacher")
        if row.get("mtp_enabled") is not True:
            reasons.append("model_built does not contain native MTP")
        if row.get("mtp_trainable_parameters") != 0:
            reasons.append("native MTP parameters are unexpectedly trainable")

    latest = sessions[-1] if sessions else {}
    contract = {
        "sessions": len(sessions),
        "micro_batch_size": latest.get("micro_batch_size"),
        "global_batch_tokens": latest.get("global_batch_tokens"),
        "gradient_accumulation_steps": latest.get("gradient_accumulation_steps"),
        "gpu_name": latest.get("gpu_name"),
        "gpu_total_memory_bytes": latest.get("gpu_total_memory_bytes"),
        "config_fingerprint": latest.get("config_fingerprint"),
        "data_fingerprint": latest.get("data_fingerprint"),
        "source_tree_sha256": latest.get("source_tree_sha256"),
    }
    return contract, reasons


def _normalize_header(value: str) -> str:
    result = value.strip().lower()
    for old, new in (
        ("power.draw [w]", "power_draw_w"),
        ("power.limit [w]", "power_limit_w"),
        ("utilization.gpu [%]", "utilization_gpu_percent"),
        ("utilization.memory [%]", "utilization_memory_percent"),
        ("memory.used [mib]", "memory_used_mib"),
        ("memory.free [mib]", "memory_free_mib"),
        ("temperature.gpu", "temperature_gpu_c"),
        ("clocks.current.sm [mhz]", "clocks_sm_mhz"),
    ):
        if result == old:
            return new
    return result.replace(".", "_").replace(" ", "_")


def _power_summary(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [
                {_normalize_header(str(key)): value for key, value in row.items()}
                for row in reader
            ]
    except OSError as exc:
        raise AuditError(f"cannot read power CSV {path}: {exc}") from exc
    if not rows:
        raise AuditError(f"power CSV {path} contains no samples")
    fields: dict[str, Any] = {}
    for key in (
        "power_draw_w",
        "power_limit_w",
        "utilization_gpu_percent",
        "utilization_memory_percent",
        "memory_used_mib",
        "memory_free_mib",
        "temperature_gpu_c",
        "clocks_sm_mhz",
    ):
        values: list[float] = []
        for index, row in enumerate(rows, 1):
            raw = row.get(key)
            if raw is None or not str(raw).strip():
                continue
            try:
                value = float(str(raw).strip())
            except ValueError as exc:
                raise AuditError(f"{path}:{index + 1} has invalid {key}") from exc
            values.append(_finite(value, label=f"{path}:{index + 1}.{key}"))
        if values:
            fields[key] = _distribution(values)
    if "power_draw_w" not in fields or "utilization_gpu_percent" not in fields:
        raise AuditError(f"power CSV {path} lacks power draw or GPU utilization")
    return {"path": str(path), "sample_count": len(rows), "fields": fields}


def _trace_payload(path: Path) -> Any:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return json.load(handle)
        return _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read profiler trace {path}: {exc}") from exc


def _trace_summary(path: Path) -> dict[str, Any]:
    payload = _trace_payload(path)
    events = payload.get("traceEvents") if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise AuditError(f"profiler trace {path} has no traceEvents list")
    durations: dict[str, list[float]] = {}
    for event in events:
        if (
            not isinstance(event, dict)
            or event.get("ph") != "X"
            or event.get("name") not in REQUIRED_TRACE_SCOPES
        ):
            continue
        duration_us = _finite(event.get("dur"), label=f"{path}.trace duration")
        durations.setdefault(str(event["name"]), []).append(duration_us / 1_000_000.0)
    scopes = {
        name: {
            **_distribution(values),
            "total_seconds": sum(values),
        }
        for name, values in sorted(durations.items())
    }
    missing = [name for name in REQUIRED_TRACE_SCOPES if name not in scopes]
    return {
        "path": str(path),
        "scopes": scopes,
        "missing_required_scopes": missing,
        "complete": not missing,
    }


def _recovery_summary(compare_path: Path | None, run_dir: Path | None) -> dict[str, Any]:
    payload: Mapping[str, Any] = {}
    if compare_path is not None:
        raw_payload = _read_json(compare_path)
        if not isinstance(raw_payload, dict):
            raise AuditError("recovery compare output must be a JSON object")
        payload = raw_payload
    events: list[dict[str, Any]] = []
    if run_dir is not None:
        events = _read_jsonl(run_dir / "events.jsonl")
    event_counts: dict[str, int] = {}
    for row in events:
        name = row.get("event")
        if isinstance(name, str):
            event_counts[name] = event_counts.get(name, 0) + 1
    checkpoint_reasons = {
        str(row.get("reason"))
        for row in events
        if row.get("event") == "checkpoint_complete" and row.get("reason") is not None
    }
    lifecycle_complete = bool(
        run_dir is not None
        and event_counts.get("session_start", 0) >= 2
        and event_counts.get("resume", 0) >= 1
        and event_counts.get("graceful_stop", 0) >= 1
        and event_counts.get("train_complete", 0) >= 1
        and "stop-file" in checkpoint_reasons
        and "sigusr1" in checkpoint_reasons
    )
    return {
        "provided": compare_path is not None and run_dir is not None,
        "equivalent": payload.get("equivalent") is True,
        "compare_path": str(compare_path) if compare_path is not None else None,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "payload": payload,
        "event_counts": event_counts,
        "checkpoint_reasons": sorted(checkpoint_reasons),
        "stop_resume_sigusr1_complete": lifecycle_complete,
    }


def _case_summary(
    label: str,
    run_dir: Path,
    *,
    power_path: Path | None,
    trace_path: Path | None,
    drop_first_steps: int,
    min_measured_steps: int,
    expected_global_batch_tokens: int,
    minimum_headroom_gib: float,
    require_power: bool,
    require_trace: bool,
) -> dict[str, Any]:
    metrics = _strict_step_map(
        _read_jsonl(run_dir / "metrics.jsonl"),
        label=f"{label}.metrics",
    )
    telemetry = _strict_step_map(
        _read_jsonl(run_dir / "telemetry.jsonl"),
        label=f"{label}.telemetry",
    )
    events = _read_jsonl(run_dir / "events.jsonl")
    contract, reasons = _event_contract(events)
    common_steps = sorted(set(metrics).intersection(telemetry))
    if common_steps != sorted(metrics) or common_steps != sorted(telemetry):
        reasons.append("metrics and telemetry committed steps differ")
    if drop_first_steps < 0:
        raise AuditError("drop-first-steps must be non-negative")
    selected_steps = common_steps[drop_first_steps:]
    if len(selected_steps) < min_measured_steps:
        reasons.append(
            f"only {len(selected_steps)} measured steps remain; need {min_measured_steps}"
        )
    if contract["global_batch_tokens"] != expected_global_batch_tokens:
        reasons.append(
            "global batch differs from expected "
            f"{expected_global_batch_tokens}: {contract['global_batch_tokens']}"
        )
    micro_batch_size = contract["micro_batch_size"]
    if (
        isinstance(micro_batch_size, bool)
        or not isinstance(micro_batch_size, int)
        or micro_batch_size not in {1, 2, 4}
    ):
        reasons.append(f"micro batch is not one of 1/2/4: {micro_batch_size!r}")

    token_counts: list[float] = []
    compute_seconds: list[float] = []
    wall_seconds: list[float] = []
    compute_tps: list[float] = []
    wall_tps: list[float] = []
    peak_allocated: list[float] = []
    peak_reserved: list[float] = []
    for step in selected_steps:
        metric = metrics[step]
        tele = telemetry[step]
        if metric.get("data_mode") != "prepared-text":
            reasons.append(f"step {step} data_mode is not prepared-text")
        for forbidden in (
            "teacher_kd",
            "teacher_kd_loss",
            "anchor_kl",
            "anchor_kl_loss",
            "hidden_alignment",
            "hidden_alignment_loss",
        ):
            if forbidden in metric:
                reasons.append(f"step {step} unexpectedly logs {forbidden}")
        for required in ("ntp", "mtp", "loss", "grad_norm"):
            if required not in metric:
                reasons.append(f"step {step} lacks {required}")
            else:
                _finite(metric[required], label=f"{label}.step{step}.{required}")
        tokens = _finite(metric.get("tokens_this_step"), label=f"{label}.tokens_this_step")
        compute = _finite(
            tele.get("compute_step_seconds"),
            label=f"{label}.compute_step_seconds",
        )
        wall = _finite(
            tele.get("wall_clock_step_seconds"),
            label=f"{label}.wall_clock_step_seconds",
        )
        if tokens <= 0 or compute <= 0 or wall <= 0:
            reasons.append(f"step {step} has non-positive token/time measurements")
            continue
        token_counts.append(tokens)
        compute_seconds.append(compute)
        wall_seconds.append(wall)
        compute_tps.append(tokens / compute)
        wall_tps.append(tokens / wall)
        peak_allocated.append(
            _finite(
                tele.get("gpu_peak_allocated_gib"),
                label=f"{label}.gpu_peak_allocated_gib",
            )
        )
        peak_reserved.append(
            _finite(
                tele.get("gpu_peak_reserved_gib"),
                label=f"{label}.gpu_peak_reserved_gib",
            )
        )

    total_memory_bytes = contract.get("gpu_total_memory_bytes")
    headroom_gib: float | None = None
    if (
        not isinstance(total_memory_bytes, bool)
        and isinstance(total_memory_bytes, int)
        and total_memory_bytes > 0
        and peak_reserved
    ):
        headroom_gib = total_memory_bytes / GIB - max(peak_reserved)
        if headroom_gib < minimum_headroom_gib:
            reasons.append(
                f"peak reserved leaves {headroom_gib:.3f} GiB; "
                f"need {minimum_headroom_gib:.3f} GiB"
            )
    else:
        reasons.append("GPU total memory or peak-reserved evidence is absent")

    power = _power_summary(power_path) if power_path is not None else None
    trace = _trace_summary(trace_path) if trace_path is not None else None
    if require_power and power is None:
        reasons.append("required power sample is absent")
    if require_trace and (trace is None or not trace["complete"]):
        reasons.append("required profiler scopes are absent")

    event_counts: dict[str, int] = {}
    for row in events:
        name = row.get("event")
        if isinstance(name, str):
            event_counts[name] = event_counts.get(name, 0) + 1

    performance = None
    if token_counts:
        performance = {
            "measured_steps": len(token_counts),
            "measured_tokens": int(sum(token_counts)),
            "aggregate_compute_tokens_per_second": sum(token_counts) / sum(compute_seconds),
            "aggregate_wall_tokens_per_second": sum(token_counts) / sum(wall_seconds),
            "compute_step_seconds": _distribution(compute_seconds),
            "wall_step_seconds": _distribution(wall_seconds),
            "compute_tokens_per_second": _distribution(compute_tps),
            "wall_tokens_per_second": _distribution(wall_tps),
            "gpu_peak_allocated_gib": {
                "max": max(peak_allocated),
                "p95": _nearest_rank(peak_allocated, 0.95),
            },
            "gpu_peak_reserved_gib": {
                "max": max(peak_reserved),
                "p95": _nearest_rank(peak_reserved, 0.95),
            },
            "minimum_observed_headroom_gib": headroom_gib,
        }
    return {
        "label": label,
        "run_dir": str(run_dir),
        "accepted": not reasons,
        "rejection_reasons": sorted(set(reasons)),
        "contract": contract,
        "measured_step_ids": selected_steps,
        "performance": performance,
        "power": power,
        "trace": trace,
        "event_counts": event_counts,
        "stop_resume_observed": (
            event_counts.get("graceful_stop", 0) > 0
            and event_counts.get("resume", 0) > 0
        ),
    }


def _recommendation(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    accepted = [
        case
        for case in cases
        if case.get("accepted") is True and isinstance(case.get("performance"), dict)
    ]
    if not accepted:
        return {
            "available": False,
            "label": None,
            "reason": "no case passed the configured gate",
        }
    ordered = sorted(
        accepted,
        key=lambda case: float(case["performance"]["aggregate_wall_tokens_per_second"]),
        reverse=True,
    )
    winner = ordered[0]
    return {
        "available": True,
        "label": winner["label"],
        "micro_batch_size": winner["contract"]["micro_batch_size"],
        "aggregate_wall_tokens_per_second": winner["performance"][
            "aggregate_wall_tokens_per_second"
        ],
        "selection_scope": (
            "performance-and-capacity only; global-batch quality and held-out NLL "
            "remain separate gates"
        ),
    }


def _failed_case_summary(label: str, run_dir: Path, error: AuditError) -> dict[str, Any]:
    event_counts: dict[str, int] = {}
    failure_events: list[Mapping[str, Any]] = []
    events_path = run_dir / "events.jsonl"
    if events_path.is_file():
        try:
            events = _read_jsonl(events_path)
        except AuditError:
            events = []
        for row in events:
            name = row.get("event")
            if isinstance(name, str):
                event_counts[name] = event_counts.get(name, 0) + 1
            if name == "train_failed":
                failure_events.append(row)
    return {
        "label": label,
        "run_dir": str(run_dir),
        "accepted": False,
        "rejection_reasons": [str(error)],
        "contract": {},
        "measured_step_ids": [],
        "performance": None,
        "power": None,
        "trace": None,
        "event_counts": event_counts,
        "failure_events": failure_events,
        "stop_resume_observed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.min_measured_steps <= 0:
        raise AuditError("min-measured-steps must be positive")
    if args.expected_global_batch_tokens <= 0:
        raise AuditError("expected-global-batch-tokens must be positive")
    if not math.isfinite(args.minimum_headroom_gib) or args.minimum_headroom_gib < 0:
        raise AuditError("minimum-headroom-gib must be finite and non-negative")

    cases = _labeled_paths(args.case, option="--case")
    power = _labeled_paths(args.power, option="--power")
    traces = _labeled_paths(args.trace, option="--trace")
    unknown = (set(power) | set(traces)) - set(cases)
    if unknown:
        raise AuditError(f"power/trace labels do not identify a case: {sorted(unknown)}")

    summaries = []
    for label, run_dir in cases.items():
        try:
            case = _case_summary(
                label,
                run_dir,
                power_path=power.get(label),
                trace_path=traces.get(label),
                drop_first_steps=args.drop_first_steps,
                min_measured_steps=args.min_measured_steps,
                expected_global_batch_tokens=args.expected_global_batch_tokens,
                minimum_headroom_gib=args.minimum_headroom_gib,
                require_power=args.require_power,
                require_trace=args.require_trace,
            )
        except AuditError as exc:
            case = _failed_case_summary(label, run_dir, exc)
        summaries.append(case)
    recovery = _recovery_summary(
        Path(args.recovery_compare) if args.recovery_compare else None,
        Path(args.recovery_run) if args.recovery_run else None,
    )
    recovery_passed = (
        recovery["equivalent"] and recovery["stop_resume_sigusr1_complete"]
    ) or not args.require_recovery
    recommendation = _recommendation(summaries)
    report = {
        "kind": "twen_v4_optimizer_step_ab_audit",
        "schema_version": 1,
        "read_only": True,
        "training_started_by_auditor": False,
        "input_training_runs": len(summaries),
        "expected_global_batch_tokens": args.expected_global_batch_tokens,
        "drop_first_steps": args.drop_first_steps,
        "min_measured_steps": args.min_measured_steps,
        "minimum_headroom_gib": args.minimum_headroom_gib,
        "requirements": {
            "power": args.require_power,
            "trace": args.require_trace,
            "recovery": args.require_recovery,
        },
        "cases": summaries,
        "all_cases_accepted": all(case["accepted"] for case in summaries),
        "recovery": recovery,
        "recommendation": recommendation,
        "accepted": recommendation["available"] and recovery_passed,
    }
    payload = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        raise SystemExit(2) from exc
