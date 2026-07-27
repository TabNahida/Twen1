#!/usr/bin/env python3
"""Status, dry-run, and explicitly authorized v4 governed training control."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from twen.governed import (
    GovernedControllerError,
    authenticate_checkpoint,
    authenticate_checkpoint_execution_identity,
    authorize_run,
    build_authenticated_gate_observation,
    build_governed_plan,
    build_train_command,
    controller_status,
    evaluate_hard_gates,
    generate_checkpoint_sweep,
    generate_drift_evidence,
    load_controller_state,
    render_train_command,
    verify_controller_sources,
    write_controller_state,
)
from twen.runtime.checkpoint import CheckpointError, CheckpointManager
from twen.utils import sha256_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readiness",
        type=Path,
        default=Path("locks/base-dense-v4-250m-pilot.readiness.json"),
    )
    parser.add_argument(
        "--action",
        choices=("status", "dry-run", "run"),
        default="status",
    )
    parser.add_argument("--state", type=Path)
    parser.add_argument("--ack", help="for run: exact `RUN <plan-id>` acknowledgement")
    return parser


def _state_path(plan: dict[str, Any], explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    run = plan.get("run")
    if not isinstance(run, dict) or not isinstance(run.get("output_dir"), str):
        raise GovernedControllerError("plan run output_dir is invalid")
    output = Path(run["output_dir"]).resolve()
    return output.parent / f".{output.name}.governed" / "controller-state.json"


@contextmanager
def _state_lock(state_path: Path) -> Any:
    lock_path = state_path.with_suffix(f"{state_path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise GovernedControllerError(f"cannot open no-follow controller lock: {lock_path}") from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GovernedControllerError("another governed controller owns this state") from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _checkpoint_after_segment(
    plan: dict[str, Any],
    *,
    expected_threshold: int,
) -> dict[str, Any]:
    run = plan["run"]
    manager = CheckpointManager(run["output_dir"], rank=0, world_size=1)
    try:
        checkpoint = manager.find_latest()
    except CheckpointError as exc:
        raise GovernedControllerError(
            f"no authenticated checkpoint satisfies threshold {expected_threshold}"
        ) from exc
    identity = authenticate_checkpoint(checkpoint)
    metadata = identity["metadata"]
    tokens = int(metadata.get("committed_tokens", -1))
    if tokens < expected_threshold:
        raise GovernedControllerError(
            f"training exited before governed threshold {expected_threshold}: {tokens}"
        )
    if expected_threshold < int(run["max_tokens"]):
        expected_tag = f"governed-{expected_threshold:012d}"
        extra = metadata.get("extra")
        if (
            metadata.get("kind") != "milestone"
            or metadata.get("tag") != expected_tag
            or not isinstance(extra, dict)
            or extra.get("governed_pause_at_tokens") != expected_threshold
        ):
            raise GovernedControllerError(
                "training segment did not finish with the authenticated governed pause checkpoint"
            )
    elif metadata.get("kind") != "milestone" or metadata.get("tag") != "complete":
        raise GovernedControllerError(
            "the terminal threshold did not finish with the complete milestone"
        )
    return {
        "path": identity["path"],
        "manifest_sha256": identity["manifest_sha256"],
        "complete_sha256": identity["complete_sha256"],
        "global_step": metadata.get("global_step"),
        "committed_tokens": tokens,
        "kind": metadata.get("kind"),
        "tag": metadata.get("tag"),
    }


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _required_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GovernedControllerError(f"{label} must be an object")
    return value


def _same_checkpoint_binding(
    authenticated: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> bool:
    metadata = authenticated.get("metadata")
    return bool(
        isinstance(metadata, Mapping)
        and Path(str(authenticated.get("path"))).resolve()
        == Path(str(binding.get("path"))).resolve()
        and authenticated.get("manifest_sha256") == binding.get("manifest_sha256")
        and authenticated.get("complete_sha256") == binding.get("complete_sha256")
        and metadata.get("global_step") == binding.get("global_step")
        and metadata.get("committed_tokens") == binding.get("committed_tokens")
        and metadata.get("kind") == binding.get("kind")
        and metadata.get("tag") == binding.get("tag")
    )


def _threshold_root(plan: Mapping[str, Any], threshold: int) -> Path:
    run = plan.get("run")
    if not isinstance(run, Mapping) or not isinstance(run.get("output_dir"), str):
        raise GovernedControllerError("plan run output_dir is invalid")
    return (
        Path(run["output_dir"]).resolve()
        / "governed"
        / f"threshold-{threshold:012d}"
    )


def _evaluation_command(
    plan: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    phase: str,
    output: Path,
) -> list[str]:
    formal = plan.get("formal_evidence")
    if not isinstance(formal, Mapping):
        raise GovernedControllerError("plan formal evidence is invalid")
    phases = formal.get("validation_phases")
    if not isinstance(phases, Mapping) or not isinstance(phases.get(phase), Mapping):
        raise GovernedControllerError(f"plan has no {phase} validation phase")
    phase_plan = phases[phase]
    prepared = phase_plan.get("prepared")
    contract = phase_plan.get("comparison_contract")
    config = plan.get("config")
    if not all(isinstance(value, Mapping) for value in (prepared, contract, config)):
        raise GovernedControllerError(f"{phase} validation contract is incomplete")
    batch_size = contract.get("batch_size")
    device_type = contract.get("device_type")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise GovernedControllerError(f"{phase} evaluation batch_size is invalid")
    if device_type not in {"cpu", "cuda"}:
        raise GovernedControllerError(f"{phase} evaluation device_type is invalid")
    return [
        sys.executable,
        "-m",
        "twen.cli",
        "evaluate",
        "nll",
        "--config",
        str(config["path"]),
        "--checkpoint",
        str(checkpoint["path"]),
        "--prepared-manifest",
        str(prepared["path"]),
        "--prepared-manifest-sha256",
        str(prepared["sha256"]),
        "--output",
        str(output),
        "--role",
        "candidate",
        "--batch-size",
        str(batch_size),
        "--device",
        str(device_type),
    ]


def _run_checked(
    command: list[str],
    *,
    label: str,
    plan: Mapping[str, Any],
) -> None:
    verify_controller_sources(plan)
    try:
        completed = subprocess.run(
            command,
            check=False,
            cwd=str(plan["project_root"]),
        )
    except OSError as exc:
        raise GovernedControllerError(f"cannot start {label}: {exc}") from exc
    verify_controller_sources(plan)
    if completed.returncode != 0:
        raise GovernedControllerError(
            f"{label} failed with exit code {completed.returncode}"
        )


def _history_nll(
    state: Mapping[str, Any],
    *,
    exclude_last: bool = False,
) -> list[dict[str, Any]]:
    raw_history = state.get("gate_history")
    if not isinstance(raw_history, list):
        raise GovernedControllerError("controller gate history is invalid")
    selected = raw_history[:-1] if exclude_last else raw_history
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(selected):
        if not isinstance(raw, Mapping):
            raise GovernedControllerError(f"gate_history[{index}] is invalid")
        checkpoint = raw.get("checkpoint")
        decision = raw.get("decision")
        if not isinstance(checkpoint, Mapping) or not isinstance(decision, Mapping):
            raise GovernedControllerError(f"gate_history[{index}] is incomplete")
        nll = decision.get("nll")
        if not isinstance(nll, Mapping):
            raise GovernedControllerError(f"gate_history[{index}] has no NLL evidence")
        result.append(
            {
                "checkpoint_id": checkpoint.get("path"),
                "aggregate_nll": nll.get("aggregate"),
                "chinese_nll": nll.get("chinese"),
            }
        )
    return result


def _ensure_threshold_artifacts(
    plan: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    threshold: int,
) -> tuple[Path, dict[str, Path]]:
    root = _threshold_root(plan, threshold)
    root.mkdir(parents=True, exist_ok=True)
    drift_path = root / "drift.json"
    if not drift_path.is_file():
        generate_drift_evidence(plan, checkpoint, drift_path)
    sweep_roots: dict[str, Path] = {}
    for phase in ("primary", "cooldown"):
        evaluation_root = root / "evaluations" / phase
        if not (evaluation_root / "COMPLETE").is_file():
            _run_checked(
                _evaluation_command(
                    plan,
                    checkpoint,
                    phase=phase,
                    output=evaluation_root,
                ),
                label=f"{phase} candidate NLL",
                plan=plan,
            )
        sweep_root = root / "sweeps" / phase
        if not (sweep_root / "COMPLETE").is_file():
            if sweep_root.exists():
                raise GovernedControllerError(
                    f"incomplete immutable {phase} sweep output exists: {sweep_root}"
                )
            generate_checkpoint_sweep(
                plan,
                checkpoint,
                phase=phase,
                candidate_evaluation=evaluation_root,
                output=sweep_root,
            )
        sweep_roots[phase] = sweep_root
    return drift_path, sweep_roots


def _build_threshold_decision(
    plan: dict[str, Any],
    state: Mapping[str, Any],
    checkpoint_binding: Mapping[str, Any],
    *,
    threshold: int,
    exclude_last_history: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = authenticate_checkpoint(str(checkpoint_binding["path"]))
    if not _same_checkpoint_binding(checkpoint, checkpoint_binding):
        raise GovernedControllerError("controller checkpoint binding changed")
    # Reject an otherwise self-consistent checkpoint from another config or
    # source tree before drift/NLL/sweep work can allocate resources.
    authenticate_checkpoint_execution_identity(plan, checkpoint)
    run = plan["run"]
    metrics_path = Path(str(run["output_dir"])).resolve() / "metrics.jsonl"
    drift_path, sweep_roots = _ensure_threshold_artifacts(
        plan,
        checkpoint,
        threshold=threshold,
    )
    observation = build_authenticated_gate_observation(
        plan,
        checkpoint,
        metrics_path=metrics_path,
        drift_path=drift_path,
        sweep_roots=sweep_roots,
    )
    decision = evaluate_hard_gates(
        plan,
        observation,
        history=_history_nll(state, exclude_last=exclude_last_history),
    )
    sweeps = observation["evaluation"]["sweeps"]
    evidence = {
        "checkpoint": dict(checkpoint_binding),
        "metrics": {
            "path": str(metrics_path),
            **dict(observation["metrics_prefix"]),
        },
        "drift": {
            "path": str(drift_path),
            "sha256": sha256_file(drift_path),
        },
        "sweeps": {
            phase: {
                "path": sweeps[phase]["path"],
                "manifest_sha256": sweeps[phase]["manifest_sha256"],
                "complete_sha256": sweeps[phase]["complete_sha256"],
            }
            for phase in ("primary", "cooldown")
        },
        "observation_sha256": _canonical_sha256(observation),
    }
    return decision, evidence


def _record_threshold_decision(
    state_path: Path,
    plan: dict[str, Any],
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if state.get("status") != "awaiting_evaluation":
        raise GovernedControllerError("controller is not awaiting evaluation")
    threshold = state.get("active_threshold")
    checkpoint = state.get("current_checkpoint")
    if not isinstance(threshold, int) or not isinstance(checkpoint, Mapping):
        raise GovernedControllerError("awaiting evaluation state is incomplete")
    decision, evidence = _build_threshold_decision(
        plan,
        state,
        checkpoint,
        threshold=threshold,
    )
    record = {
        "threshold": threshold,
        "checkpoint": dict(checkpoint),
        "evidence": evidence,
        "evidence_fingerprint": _canonical_sha256(evidence),
        "decision": decision,
    }
    state["gate_history"] = [*state["gate_history"], record]
    state["completed_thresholds"] = [*state["completed_thresholds"], threshold]
    state["active_threshold"] = None
    state["active_command"] = None
    state["recovery_checkpoint"] = None
    action = decision["action"]
    if action == "resume":
        state["status"] = "resume_authorized"
    elif action == "complete":
        state["status"] = "completed"
    elif action == "review":
        state["status"] = "review_required"
    else:
        state["status"] = "halted"
    state = write_controller_state(state_path, state, plan=plan)
    return state, decision


def _reauthenticate_gate_history(
    plan: dict[str, Any],
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Rebuild every gate from raw authenticated artifacts, oldest first."""

    history = state.get("gate_history")
    if not isinstance(history, list):
        raise GovernedControllerError("controller gate history is invalid")
    rebuilt: list[dict[str, Any]] = []
    replay_state: dict[str, Any] = {"gate_history": rebuilt}
    for index, raw in enumerate(history):
        if not isinstance(raw, Mapping):
            raise GovernedControllerError(f"gate_history[{index}] is invalid")
        checkpoint = raw.get("checkpoint")
        threshold = raw.get("threshold")
        if not isinstance(checkpoint, Mapping) or not isinstance(threshold, int):
            raise GovernedControllerError(f"gate_history[{index}] is incomplete")
        decision, evidence = _build_threshold_decision(
            plan,
            replay_state,
            checkpoint,
            threshold=threshold,
        )
        evidence_fingerprint = _canonical_sha256(evidence)
        if (
            decision != raw.get("decision")
            or evidence != raw.get("evidence")
            or evidence_fingerprint != raw.get("evidence_fingerprint")
        ):
            raise GovernedControllerError(
                f"gate_history[{index}] differs after full artifact reauthentication"
            )
        rebuilt.append(
            {
                "threshold": threshold,
                "checkpoint": dict(checkpoint),
                "evidence": evidence,
                "evidence_fingerprint": evidence_fingerprint,
                "decision": decision,
            }
        )
    return rebuilt


def _reauthenticate_latest_resume(
    plan: dict[str, Any],
    state: Mapping[str, Any],
) -> None:
    """Compatibility wrapper; resume now authenticates the complete history."""

    if state.get("status") != "resume_authorized":
        raise GovernedControllerError("state is not resume-authorized")
    _reauthenticate_gate_history(plan, state)


def _process_start_time_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    close = raw.rfind(")")
    if close < 0:
        raise GovernedControllerError(f"/proc/{pid}/stat has no command terminator")
    fields_from_state = raw[close + 2 :].split()
    try:
        result = int(fields_from_state[19])
    except (IndexError, ValueError) as exc:
        raise GovernedControllerError(
            f"/proc/{pid}/stat has no valid starttime"
        ) from exc
    if result < 0:
        raise GovernedControllerError(f"/proc/{pid}/stat has a negative starttime")
    return result


def _process_cmdline(pid: int) -> list[str]:
    try:
        payload = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError as exc:
        raise GovernedControllerError(
            f"cannot read training PID {pid} command line"
        ) from exc
    result = [
        item.decode("utf-8", errors="surrogateescape")
        for item in payload.split(b"\0")
        if item
    ]
    if not result:
        raise GovernedControllerError(f"training PID {pid} has an empty command line")
    return result


def _process_exists(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def _process_cwd(pid: int) -> Path:
    try:
        return Path(os.readlink(Path(f"/proc/{pid}") / "cwd")).resolve()
    except OSError as exc:
        raise GovernedControllerError(
            f"cannot authenticate training PID {pid} working directory"
        ) from exc


def _training_payload(command: list[str], *, label: str) -> list[str]:
    matches = [
        index
        for index in range(len(command) - 1)
        if command[index : index + 2] == ["-m", "twen.cli"]
    ]
    if len(matches) != 1:
        raise GovernedControllerError(f"{label} has no unique '-m twen.cli' payload")
    return command[matches[0] :]


def _active_training_pid(
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
) -> int | None:
    run = plan.get("run")
    if not isinstance(run, Mapping):
        return None
    session_path = Path(str(run.get("output_dir"))) / "rank0-session.json"
    try:
        session = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(session, Mapping) or session.get("status") != "running":
        return None
    pid = session.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise GovernedControllerError("running rank-zero session has an invalid PID")
    if not _process_exists(pid):
        return None
    active_command = state.get("active_command")
    if not isinstance(active_command, list) or not all(
        isinstance(item, str) for item in active_command
    ):
        raise GovernedControllerError("running controller state has no exact command")
    recovery = state.get("recovery_checkpoint")
    if recovery is not None and not isinstance(recovery, Mapping):
        raise GovernedControllerError("running controller recovery checkpoint is invalid")
    expected_active_command = _resume_command_from_checkpoint(
        dict(plan),
        dict(state),
        recovery,
    )
    if active_command != expected_active_command:
        raise GovernedControllerError(
            "running controller command differs from the immutable plan; "
            "refusing duplicate launch"
        )
    recorded_cmdline = session.get("process_cmdline")
    if not isinstance(recorded_cmdline, list) or not all(
        isinstance(item, str) and item for item in recorded_cmdline
    ):
        raise GovernedControllerError(
            "running rank-zero session has no exact process command line"
        )
    actual_cmdline = _process_cmdline(pid)
    expected_config = _required_mapping(plan.get("config"), label="plan.config")
    source_tree = _required_mapping(plan.get("source_tree"), label="plan.source_tree")
    dependency = _required_mapping(
        plan.get("dependency_lock"),
        label="plan.dependency_lock",
    )
    recorded_start = session.get("process_start_time_ticks")
    if (
        isinstance(recorded_start, bool)
        or not isinstance(recorded_start, int)
        or recorded_start < 0
    ):
        raise GovernedControllerError(
            "running rank-zero session has no process start identity"
        )
    actual_cwd = _process_cwd(pid)
    expected_root = Path(str(plan.get("project_root"))).resolve()
    recorded_cwd = Path(str(session.get("cwd"))).resolve()
    recorded_config = Path(str(session.get("config_path"))).resolve()
    expected_config_path = Path(str(expected_config.get("path"))).resolve()
    recorded_dependency = Path(str(session.get("dependency_lock"))).resolve()
    expected_dependency = Path(str(dependency.get("path"))).resolve()
    session_config_fingerprint = session.get("config_fingerprint")
    if (
        not isinstance(session_config_fingerprint, str)
        or len(session_config_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in session_config_fingerprint)
    ):
        raise GovernedControllerError(
            "running rank-zero session has no valid config_fingerprint"
        )
    mismatches: list[str] = []
    if session.get("hostname") != platform.node():
        mismatches.append("hostname")
    if _process_start_time_ticks(pid) != recorded_start:
        mismatches.append("process_start_time_ticks")
    if actual_cmdline != recorded_cmdline:
        mismatches.append("process_cmdline")
    if _training_payload(actual_cmdline, label="rank-zero process command") != (
        _training_payload(active_command, label="controller active_command")
    ):
        mismatches.append("active_command")
    if actual_cwd != expected_root or recorded_cwd != expected_root:
        mismatches.append("cwd")
    if (
        recorded_config != expected_config_path
        or session.get("config_sha256") != expected_config.get("sha256")
        or (
            expected_config.get("preflight_fingerprint") is not None
            and session_config_fingerprint
            != expected_config.get("preflight_fingerprint")
        )
    ):
        mismatches.append("config")
    if (
        session.get("run_id") != run.get("run_id")
        or session.get("stage") != run.get("stage")
        or session.get("world_size") != run.get("world_size")
    ):
        mismatches.append("run")
    if session.get("source_tree_sha256") != source_tree.get("sha256"):
        mismatches.append("source_tree")
    if (
        recorded_dependency != expected_dependency
        or session.get("dependency_lock_sha256") != dependency.get("sha256")
    ):
        mismatches.append("dependency_lock")
    if mismatches:
        raise GovernedControllerError(
            "running rank-zero session identity mismatch "
            f"({', '.join(sorted(set(mismatches)))}); refusing duplicate launch"
        )
    return pid


def _resume_command_from_checkpoint(
    plan: dict[str, Any],
    state: dict[str, Any],
    checkpoint: Mapping[str, Any] | None,
) -> list[str]:
    command = build_train_command(plan, state)
    if checkpoint is None:
        return command
    resume_index = command.index("--resume")
    command[resume_index + 1] = str(checkpoint["path"])
    if "--fork-from" in command:
        fork_index = command.index("--fork-from")
        del command[fork_index : fork_index + 2]
    return command


def _transition_to_awaiting(
    state_path: Path,
    plan: dict[str, Any],
    state: dict[str, Any],
    *,
    checkpoint: dict[str, Any],
    exit_code: int | None,
) -> dict[str, Any]:
    state["current_checkpoint"] = checkpoint
    state["status"] = "awaiting_evaluation"
    state["last_exit_code"] = exit_code
    state["recovery_checkpoint"] = None
    return write_controller_state(state_path, state, plan=plan)


def _reconcile_running_segment(
    state_path: Path,
    plan: dict[str, Any],
    state: dict[str, Any],
    *,
    launch_if_needed: bool,
) -> dict[str, Any]:
    threshold = state.get("active_threshold")
    if not isinstance(threshold, int):
        raise GovernedControllerError("running state has no active threshold")
    try:
        checkpoint = _checkpoint_after_segment(plan, expected_threshold=threshold)
    except GovernedControllerError:
        checkpoint = None
    if checkpoint is not None:
        return _transition_to_awaiting(
            state_path,
            plan,
            state,
            checkpoint=checkpoint,
            exit_code=state.get("last_exit_code"),
        )
    active_pid = _active_training_pid(plan, state)
    if active_pid is not None:
        raise GovernedControllerError(
            f"training segment is still active as PID {active_pid}; refusing a duplicate"
        )
    if not launch_if_needed:
        raise GovernedControllerError("training segment has no threshold checkpoint")

    run = plan["run"]
    manager = CheckpointManager(run["output_dir"], rank=0, world_size=1)
    latest = manager.find_latest_valid_with_metadata()
    recovery: dict[str, Any] | None = None
    if latest is not None:
        identity = authenticate_checkpoint(latest[0])
        metadata = identity["metadata"]
        if int(metadata.get("committed_tokens", -1)) >= threshold:
            raise GovernedControllerError(
                "latest checkpoint crossed the threshold without governed identity"
            )
        recovery = {
            "path": identity["path"],
            "manifest_sha256": identity["manifest_sha256"],
            "complete_sha256": identity["complete_sha256"],
            "global_step": metadata["global_step"],
            "committed_tokens": metadata["committed_tokens"],
            "kind": metadata["kind"],
            "tag": metadata.get("tag"),
        }
    elif Path(run["output_dir"]).exists() and any(Path(run["output_dir"]).iterdir()):
        raise GovernedControllerError(
            "partial first segment has no resumable checkpoint; manual review is required"
        )
    state["recovery_checkpoint"] = recovery
    command = _resume_command_from_checkpoint(plan, state, recovery)
    state["active_command"] = command
    verify_controller_sources(plan)
    state = write_controller_state(state_path, state, plan=plan)
    try:
        completed = subprocess.run(
            command,
            check=False,
            cwd=str(plan["project_root"]),
        )
    except OSError as exc:
        raise GovernedControllerError(f"cannot start training segment: {exc}") from exc
    verify_controller_sources(plan)
    state = load_controller_state(state_path, plan)
    state["last_exit_code"] = completed.returncode
    try:
        checkpoint = _checkpoint_after_segment(plan, expected_threshold=threshold)
    except GovernedControllerError as exc:
        if completed.returncode != 0:
            state["status"] = "failed"
            state["active_threshold"] = None
            state["active_command"] = None
            state["recovery_checkpoint"] = None
            state["failure"] = (
                f"training exit {completed.returncode}; no governed checkpoint: {exc}"
            )
            write_controller_state(state_path, state, plan=plan)
        raise GovernedControllerError(
            f"training segment did not produce threshold checkpoint: {exc}"
        ) from exc
    return _transition_to_awaiting(
        state_path,
        plan,
        state,
        checkpoint=checkpoint,
        exit_code=completed.returncode,
    )


def _run(
    args: argparse.Namespace,
    plan: dict[str, Any],
    state_path: Path,
) -> dict[str, Any]:
    authorize_run(plan, args.ack)
    with _state_lock(state_path):
        state = load_controller_state(state_path, plan)
        freshly_authorized = False
        authenticated_history_digest: str | None = None
        while True:
            status = state.get("status")
            history = state.get("gate_history")
            if not isinstance(history, list):
                raise GovernedControllerError("controller gate history is invalid")
            history_digest = _canonical_sha256(history)
            if (
                status
                in {
                    "running",
                    "awaiting_evaluation",
                    "resume_authorized",
                    "completed",
                    "halted",
                    "review_required",
                }
                and history
                and history_digest != authenticated_history_digest
            ):
                _reauthenticate_gate_history(plan, state)
                authenticated_history_digest = history_digest
            if status in {"not_started", "resume_authorized"}:
                next_threshold = controller_status(plan, state)["next_threshold"]
                if not isinstance(next_threshold, int):
                    raise GovernedControllerError("there is no remaining governed threshold")
                command = build_train_command(plan, state)
                verify_controller_sources(plan)
                state["status"] = "running"
                state["active_threshold"] = next_threshold
                state["active_command"] = command
                state["last_exit_code"] = None
                state["failure"] = None
                state["recovery_checkpoint"] = None
                state = write_controller_state(state_path, state, plan=plan)
                state = _reconcile_running_segment(
                    state_path,
                    plan,
                    state,
                    launch_if_needed=True,
                )
                freshly_authorized = False
                continue
            if status == "running":
                state = _reconcile_running_segment(
                    state_path,
                    plan,
                    state,
                    launch_if_needed=True,
                )
                continue
            if status == "awaiting_evaluation":
                state, decision = _record_threshold_decision(state_path, plan, state)
                authenticated_history_digest = _canonical_sha256(
                    state["gate_history"]
                )
                freshly_authorized = decision["action"] == "resume"
                if freshly_authorized:
                    continue
                return {
                    "ok": decision["action"] == "complete",
                    "action": decision["action"],
                    "threshold": state["completed_thresholds"][-1],
                    "checkpoint": state["current_checkpoint"],
                    "decision": decision,
                    "state": str(state_path),
                }
            if status in {"completed", "halted", "review_required"}:
                return {
                    "ok": status == "completed",
                    "action": status,
                    "checkpoint": state.get("current_checkpoint"),
                    "state": str(state_path),
                }
            raise GovernedControllerError(
                f"controller state {status!r} cannot continue automatically"
            )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = build_governed_plan(args.readiness)
        state_path = _state_path(plan, args.state)
        if args.action == "status":
            result = controller_status(plan, load_controller_state(state_path, plan))
        elif args.action == "dry-run":
            state = load_controller_state(state_path, plan)
            result = {
                **controller_status(plan, state),
                "action": "dry-run",
                "command": render_train_command(plan, state),
                "executes": False,
            }
        else:
            result = _run(args, plan, state_path)
    except GovernedControllerError as exc:
        print(
            json.dumps(
                {"ok": False, "error": type(exc).__name__, "message": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
