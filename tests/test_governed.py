from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import pickle
import platform
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from twen.cli import build_parser
from twen.governed import (
    GovernedControllerError,
    _authenticate_checkpoint_execution_identity,
    _authenticate_checkpoint_source_maps,
    _authenticate_generated_sweep,
    authenticate_drift_evidence,
    authorize_run,
    build_governed_plan,
    build_train_command,
    controller_status,
    evaluate_hard_gates,
    expected_run_ack,
    initial_controller_state,
    load_controller_state,
    read_authenticated_metrics_prefix,
    read_canonical_metrics,
    verify_controller_sources,
    write_controller_state,
)
from twen.runtime.checkpoint import CheckpointManager
from twen.runtime.state import DataCursor, RNGState, TrainerState
from twen.source_identity import twen_source_tree_sha256
from twen.training.engine import (
    _governed_pause_reached,
    _save_governed_pause_checkpoint,
    _validate_pause_at_tokens,
    run_training,
)
from twen.training.logging import JsonlMetricLogger

ROOT = Path(__file__).parents[1]
READINESS = ROOT / "locks/base-dense-v4-250m-pilot.readiness.json"


def _load_controller_script() -> ModuleType:
    path = ROOT / "scripts/govern_v4_training.py"
    spec = importlib.util.spec_from_file_location("govern_v4_training_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plan() -> dict[str, object]:
    return build_governed_plan(READINESS)


def _refingerprint(plan: dict[str, object]) -> None:
    unsigned = {key: value for key, value in plan.items() if key != "plan_id"}
    payload = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    plan["plan_id"] = hashlib.sha256(payload).hexdigest()


def _seal_state(state: dict[str, object]) -> None:
    unsigned = {key: value for key, value in state.items() if key != "state_fingerprint"}
    state["state_fingerprint"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _decision(plan: dict[str, object], action: str, *, nll: float = 1.9) -> dict[str, object]:
    value: dict[str, object] = {
        "kind": "twen_v4_governed_gate_decision",
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "action": action,
        "checks": [],
        "failures": {"stop": [], "review": [], "terminal_stop": []},
        "nll": {
            "aggregate": nll,
            "aggregate_delta_from_v3": -0.1,
            "chinese": 2.9,
            "chinese_delta_from_v3": -0.1,
        },
        "selection": {
            "checkpoint_id": "fixture",
            "aggregate_nll": nll,
            "tie_tolerance": 0.0001,
            "keeps_earlier_within_tolerance": True,
        },
    }
    value["decision_fingerprint"] = hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return value


def _progress_state(
    plan: dict[str, object],
    *,
    completed_count: int,
    status: str = "resume_authorized",
) -> dict[str, object]:
    state = initial_controller_state(plan)
    thresholds = plan["pause_thresholds"]
    assert isinstance(thresholds, list)
    history = []
    for index, threshold in enumerate(thresholds[:completed_count]):
        checkpoint = {
            "path": f"/tmp/step-{index + 1}",
            "manifest_sha256": f"{index + 1:064x}",
            "complete_sha256": f"{index + 2:064x}",
            "global_step": index + 1,
            "committed_tokens": threshold,
            "kind": "milestone",
            "tag": (
                f"governed-{threshold:012d}"
                if index + 1 < len(thresholds)
                else "complete"
            ),
        }
        evidence = {"fixture": index}
        history.append(
            {
                "threshold": threshold,
                "checkpoint": checkpoint,
                "evidence": evidence,
                "evidence_fingerprint": hashlib.sha256(
                    json.dumps(
                        evidence,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "decision": _decision(
                    plan,
                    "complete" if index + 1 == len(thresholds) else "resume",
                    nll=1.9 - index * 0.001,
                ),
            }
        )
    state.update(
        {
            "status": status,
            "completed_thresholds": thresholds[:completed_count],
            "gate_history": history,
            "current_checkpoint": history[-1]["checkpoint"] if history else None,
            "active_threshold": None,
            "active_command": None,
        }
    )
    _seal_state(state)
    return state


def _authorized_two_threshold_plan(tmp_path: Path) -> dict[str, object]:
    plan = deepcopy(_plan())
    plan["launch_enabled"] = True
    plan["readiness_issues"] = []
    plan["pause_thresholds"] = [10, 20]
    run = plan["run"]
    assert isinstance(run, dict)
    run["max_tokens"] = 20
    run["output_dir"] = str(tmp_path / "run")
    _refingerprint(plan)
    return plan


def _metrics(count: int = 50, *, grad_norm: float = 0.5) -> list[dict[str, object]]:
    return [
        {
            "step": index,
            "tokens": index * 262_144,
            "loss": 2.0,
            "ntp": 1.8,
            "mtp": 2.0,
            "grad_norm": grad_norm,
            "lr": 3e-5,
            "lr/adapters": 3e-5,
            "lr/scale": 3e-6,
            "lr_adjusted/adapters": 3e-5,
        }
        for index in range(1, count + 1)
    ]


class _PickleBackend:
    name = "governed-test-pickle"

    def save(self, stateful: dict[str, object], path: Path) -> None:
        path.mkdir(parents=True)
        with (path / "state.pkl").open("wb") as handle:
            pickle.dump(dict(stateful), handle)

    def load(
        self,
        stateful: dict[str, object] | None,
        path: Path,
    ) -> dict[str, object]:
        with (path / "state.pkl").open("rb") as handle:
            saved = pickle.load(handle)
        if stateful is None:
            return saved
        stateful.clear()
        stateful.update(saved)
        return stateful


def _checkpoint_runtime_state(step: int, tokens: int) -> tuple[TrainerState, DataCursor]:
    state = TrainerState(
        run_id="governed-retention",
        stage="dense-oracle",
        global_step=step,
        committed_tokens=tokens,
        gradient_accumulation_steps=1,
        global_batch_tokens=4096,
        micro_batch_tokens_per_rank=4096,
        world_size=1,
        top_k=8,
        loss_weights={"ntp": 1.0},
    )
    cursor = DataCursor(
        global_token_index=tokens,
        shuffle_seed=3407,
    )
    return state, cursor


def _observation(
    *,
    committed_tokens: int = 13_107_200,
    aggregate_nll: float = 1.9,
    chinese_nll: float = 2.9,
) -> dict[str, object]:
    return {
        "checkpoint": {
            "authenticated": True,
            "checkpoint_id": "step-50",
            "committed_tokens": committed_tokens,
            "lineage_passed": True,
            "phase_identity_passed": True,
            "disjointness_passed": True,
            "reference_epochs": [0, 0],
            "reused_sequences": 0,
            "reused_tokens": 0,
        },
        "metrics": _metrics(),
        "drift": {
            "authenticated": True,
            "scale_relative_l2": 0.01,
        },
        "evaluation": {
            "authenticated": True,
            "aggregate_nll": aggregate_nll,
            "chinese_nll": chinese_nll,
        },
        "baseline": {
            "aggregate_nll": 2.0,
            "chinese_nll": 3.0,
        },
    }


def _source_map_fixture(source_id: str, *, digest: str) -> dict[str, object]:
    return {
        "algorithm": "authenticated-extracted-output-map-v1",
        "prepared_dataset_fingerprint": digest,
        "extracted_manifest_sha256": digest,
        "sequence_length": 4096,
        "shards": [
            {
                "source_id": source_id,
                "shard_id": f"{source_id}-shard",
                "sequence_count": 100,
                "global_sample_start": 0,
                "output_path": f"filtered/{source_id}/train.jsonl",
                "output_sha256": digest,
            }
        ],
        "mix_basis_points": {source_id: 10_000},
    }


def _source_map_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _checkpoint_source_map_fixture() -> tuple[dict[str, object], dict[str, object]]:
    plan = deepcopy(_plan())
    primary_map = _source_map_fixture("primary", digest="1" * 64)
    cooldown_map = _source_map_fixture("cooldown", digest="2" * 64)
    maps = {"primary": primary_map, "cooldown": cooldown_map}
    dataset_fingerprints = {"primary": "3" * 64, "cooldown": "4" * 64}
    contracts = {
        phase: {
            "enabled": True,
            "algorithm": "token-deficit-corrected-source-mix-bp-v2",
            "source_map_sha256": _source_map_sha(source_map),
            "dataset_fingerprint": dataset_fingerprints[phase],
            "basis_points": {phase: 10_000},
            "lineage_basis_points": {phase: 10_000},
            "effective_basis_points": {phase: 10_000},
            "weight_override": False,
            "seed": 3407,
        }
        for phase, source_map in maps.items()
    }
    phase_samples = {"primary": {"primary": 10}, "cooldown": {"cooldown": 0}}
    phase_tokens = {"primary": {"primary": 40_000}, "cooldown": {"cooldown": 0}}
    combined_samples = {"primary": 10, "cooldown": 0}
    combined_tokens = {"primary": 40_000, "cooldown": 0}
    cursor = {
        "kind": "deterministic-source-mix-cooldown",
        "algorithm": "token-deficit-corrected-source-mix-bp-v2",
        "seed": 3407,
        "cooldown_start_tokens": 225_000_000,
        "critical_lineage_fingerprint": "5" * 64,
        "phase_committed_samples_by_source": phase_samples,
        "phase_committed_tokens_by_source": phase_tokens,
        "committed_samples_by_source": combined_samples,
        "committed_tokens_by_source": combined_tokens,
        "primary_cursor": {
            "kind": "deterministic-source-mix",
            "algorithm": "token-deficit-corrected-source-mix-bp-v2",
            "seed": 3407,
            "source_map": primary_map,
            "prepared_dataset_fingerprint": "1" * 64,
            "dataset_fingerprint": "3" * 64,
            "weights_basis_points": {"primary": 10_000},
        },
        "cooldown_cursor": {
            "kind": "deterministic-source-mix",
            "algorithm": "token-deficit-corrected-source-mix-bp-v2",
            "seed": 3407,
            "source_map": cooldown_map,
            "prepared_dataset_fingerprint": "2" * 64,
            "dataset_fingerprint": "4" * 64,
            "weights_basis_points": {"cooldown": 10_000},
        },
    }
    source_mix = {
        **contracts["primary"],
        "cooldown_start_tokens": 225_000_000,
        "phases": contracts,
        "cursor_critical_lineage_fingerprint": "5" * 64,
        "phase_committed_samples_by_source": phase_samples,
        "phase_committed_tokens_by_source": phase_tokens,
        "committed_samples_by_source": combined_samples,
        "committed_tokens_by_source": combined_tokens,
    }
    plan["training_data"] = {
        "source_mix_algorithm": "token-deficit-corrected-source-mix-bp-v2",
        "source_map_sha256": _source_map_sha(primary_map),
        "source_mix_basis_points": {"primary": 10_000},
        "source_mix_allow_weight_override": False,
        "shuffle_seed": 3407,
        "quality_cooldown_start_tokens": 225_000_000,
        "phase_source_maps": {
            phase: {
                "source_map_sha256": _source_map_sha(source_map),
                "prepared_dataset_fingerprint": (
                    source_map["prepared_dataset_fingerprint"]
                ),
            }
            for phase, source_map in maps.items()
        },
    }
    metadata = {
        "extra": {"source_mix": source_mix},
        "data_cursor": {"extra": cursor},
    }
    return plan, metadata


def test_train_cli_pause_is_opt_in_and_positive() -> None:
    parser = build_parser()
    base = ["train", "--stage", "dense-oracle", "--config", "config.yaml"]
    assert parser.parse_args(base).pause_at_tokens is None
    assert parser.parse_args([*base, "--pause-at-tokens", "13000000"]).pause_at_tokens == 13_000_000
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--pause-at-tokens", "0"])


def test_governed_pause_crosses_only_first_interim_complete_batch() -> None:
    assert _governed_pause_reached(
        previous_tokens=12_900_000,
        committed_tokens=13_100_000,
        pause_at_tokens=13_000_000,
        max_tokens=250_000_000,
    )
    assert not _governed_pause_reached(
        previous_tokens=13_100_000,
        committed_tokens=13_300_000,
        pause_at_tokens=13_000_000,
        max_tokens=250_000_000,
    )
    assert not _governed_pause_reached(
        previous_tokens=249_900_000,
        committed_tokens=250_100_000,
        pause_at_tokens=250_000_000,
        max_tokens=250_000_000,
    )


def test_pause_validation_rejects_invalid_or_past_terminal_threshold() -> None:
    assert _validate_pause_at_tokens(None, max_tokens=10) is None
    assert _validate_pause_at_tokens(10, max_tokens=10) == 10
    for value in (True, 0, -1, 11, 1.5):
        with pytest.raises(ValueError):
            _validate_pause_at_tokens(value, max_tokens=10)  # type: ignore[arg-type]


def test_optimizer_loop_logs_metric_before_pause_checkpoint_and_then_returns() -> None:
    source = inspect.getsource(run_training)
    metric = source.index("metric_logger.log(state.global_step, averaged)")
    decision = source.index("governed_pause_reached = _governed_pause_reached")
    checkpoint = source.index("pause_path = _save_governed_pause_checkpoint")
    paused = source.index('session_status = "paused"')
    assert metric < decision < checkpoint < paused


def test_pause_checkpoint_uses_exact_committed_state_and_cursor() -> None:
    manager = object()
    stateful = {"model": object()}
    state = object()
    cursor = object()
    config = object()
    report = object()
    expected = Path("/tmp/governed-checkpoint")
    with patch("twen.training.engine._checkpoint", return_value=expected) as checkpoint:
        actual = _save_governed_pause_checkpoint(
            manager,  # type: ignore[arg-type]
            stateful,
            state,  # type: ignore[arg-type]
            cursor,  # type: ignore[arg-type]
            config,  # type: ignore[arg-type]
            report,  # type: ignore[arg-type]
            pause_at_tokens=13_000_000,
            metric_logger=None,
            event_logger=None,
        )
    assert actual == expected
    checkpoint.assert_called_once_with(
        manager,
        stateful,
        state,
        cursor,
        config,
        report,
        kind="milestone",
        boundary=None,
        config_path=None,
        tag="governed-000013000000",
        reason="governed-token-threshold",
        governed_pause_at_tokens=13_000_000,
        metric_logger=None,
        event_logger=None,
    )


def test_all_nine_governed_milestones_survive_pruning_and_fresh_verification(
    tmp_path: Path,
) -> None:
    thresholds = [
        13_000_000,
        26_000_000,
        52_000_000,
        105_000_000,
        157_000_000,
        210_000_000,
        223_000_000,
        236_000_000,
        250_000_000,
    ]
    manager = CheckpointManager(
        tmp_path,
        backend=_PickleBackend(),
        keep_periodic=1,
        keep_interrupt=1,
    )
    milestones: list[Path] = []
    for step, threshold in enumerate(thresholds, start=1):
        state, cursor = _checkpoint_runtime_state(step, threshold)
        tag = "complete" if threshold == thresholds[-1] else f"governed-{threshold:012d}"
        milestones.append(
            manager.save(
                {"model": {"step": step}},
                trainer_state=state,
                data_cursor=cursor,
                rng_state=RNGState.capture(),
                critical_fingerprint="critical",
                data_fingerprint="data",
                kind="milestone",
                tag=tag,
            )
        )
        manager.save(
            {"model": {"step": step}},
            trainer_state=state,
            data_cursor=cursor,
            rng_state=RNGState.capture(),
            critical_fingerprint="critical",
            data_fingerprint="data",
            kind="periodic",
            tag=f"periodic-{step}",
        )

    assert all(path.is_dir() for path in milestones)
    assert len([path for path in tmp_path.glob("step-*-periodic-*")]) == 1
    fresh = CheckpointManager(
        tmp_path,
        backend=_PickleBackend(),
        keep_periodic=1,
        keep_interrupt=1,
    )
    assert fresh.verify(milestones[0])["tag"] == "governed-000013000000"
    assert fresh.verify(milestones[4])["committed_tokens"] == 157_000_000
    assert fresh.verify(milestones[-1])["tag"] == "complete"


def test_repository_readiness_builds_stable_blocked_plan() -> None:
    first = _plan()
    second = _plan()
    assert first == second
    assert first["pause_thresholds"] == [
        13_000_000,
        26_000_000,
        52_000_000,
        105_000_000,
        157_000_000,
        210_000_000,
        223_000_000,
        236_000_000,
        250_000_000,
    ]
    status = controller_status(first)
    assert status["blocked"] is True
    assert status["next_threshold"] == 13_000_000
    assert status["launch_enabled"] is False
    assert first["source_tree"]["sha256"] == twen_source_tree_sha256()  # type: ignore[index]
    assert Path(first["dependency_lock"]["path"]).name == "uv.lock"  # type: ignore[index]
    source_names = {
        Path(identity["path"]).name
        for identity in first["controller_sources"]  # type: ignore[union-attr]
    }
    assert {
        "govern_v4_training.py",
        "audit_dense_checkpoint_drift.py",
        "summarize_v4_checkpoint_validation.py",
        "evaluation.py",
    } <= source_names
    assert not any(
        "contract differs" in str(issue) for issue in first["readiness_issues"]
    )
    assert first["fork"]["model_only"] is True  # type: ignore[index]
    assert first["fork"]["reset_optimizer"] is True  # type: ignore[index]
    assert first["fork"]["reset_scheduler"] is True  # type: ignore[index]
    assert first["fork"]["reset_data_cursor"] is True  # type: ignore[index]


def _write_semantically_tampered_readiness(
    tmp_path: Path,
    *,
    config_mutation: tuple[str, str, object],
    readiness_contract_mutation: tuple[str, object] | None = None,
) -> Path:
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    config = yaml.safe_load(
        (ROOT / str(readiness["config_path"])).read_text(encoding="utf-8")
    )
    section, field, value = config_mutation
    config[section][field] = value
    config_path = tmp_path / str(readiness["config_path"])
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=True),
        encoding="utf-8",
    )
    if readiness_contract_mutation is not None:
        contract_field, contract_value = readiness_contract_mutation
        readiness["contract"][contract_field] = contract_value
    readiness["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    readiness_path = tmp_path / "locks" / "readiness.json"
    readiness_path.parent.mkdir(parents=True)
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
    return readiness_path


def test_paired_config_and_readiness_semantic_tamper_is_rejected(
    tmp_path: Path,
) -> None:
    readiness_path = _write_semantically_tampered_readiness(
        tmp_path,
        config_mutation=("optimizer", "adapter_lr", 9.0e-5),
        readiness_contract_mutation=("adapter_lr", 9.0e-5),
    )
    with pytest.raises(
        GovernedControllerError,
        match=r"readiness\.contract differs from the source-bound formal v4 policy",
    ):
        build_governed_plan(readiness_path)


def test_hashed_config_muon_semantic_tamper_is_rejected(tmp_path: Path) -> None:
    readiness_path = _write_semantically_tampered_readiness(
        tmp_path,
        config_mutation=("optimizer", "muon_ns_steps", 6),
    )
    with pytest.raises(
        GovernedControllerError,
        match=r"config.optimizer.muon_ns_steps differs",
    ):
        build_governed_plan(readiness_path)


@pytest.mark.parametrize(
    ("section", "field", "value", "expected"),
    (
        ("architecture", "top_k", 1, r"config\.architecture\.top_k differs"),
        (
            "runtime",
            "compile_streaming_loss",
            False,
            r"config\.runtime\.compile_streaming_loss differs",
        ),
    ),
)
def test_hashed_config_architecture_and_runtime_tamper_is_rejected(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    expected: str,
) -> None:
    readiness_path = _write_semantically_tampered_readiness(
        tmp_path,
        config_mutation=(section, field, value),
    )
    with pytest.raises(GovernedControllerError, match=expected):
        build_governed_plan(readiness_path)


def test_full_normalized_config_fingerprint_rejects_unlisted_static_tamper(
    tmp_path: Path,
) -> None:
    readiness_path = _write_semantically_tampered_readiness(
        tmp_path,
        config_mutation=("runtime", "profile_active_steps", 4),
    )
    with pytest.raises(
        GovernedControllerError,
        match=r"config normalized semantic fingerprint differs",
    ):
        build_governed_plan(readiness_path)


def test_blocked_readiness_wins_even_with_exact_ack() -> None:
    plan = _plan()
    with pytest.raises(GovernedControllerError, match="launch is blocked"):
        authorize_run(plan, expected_run_ack(plan))


def test_status_marks_historical_conclusions_as_not_reauthenticated() -> None:
    plan = _plan()
    state = _progress_state(plan, completed_count=1)
    status = controller_status(plan, state)
    assert status["history_verification"] == "not_performed"
    assert status["historical_conclusion_verified"] is False


def test_authorized_plan_still_requires_exact_plan_specific_ack() -> None:
    plan = deepcopy(_plan())
    plan["launch_enabled"] = True
    plan["readiness_issues"] = []
    _refingerprint(plan)
    with pytest.raises(GovernedControllerError, match="acknowledgement"):
        authorize_run(plan, None)
    authorize_run(plan, expected_run_ack(plan))


def test_state_is_plan_bound_and_unknown_thresholds_are_rejected(tmp_path: Path) -> None:
    plan = _plan()
    state = initial_controller_state(plan)
    state["plan_id"] = "0" * 64
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(GovernedControllerError, match="different immutable plan"):
        load_controller_state(path, plan)

    state = initial_controller_state(plan)
    state["completed_thresholds"] = [123]
    _seal_state(state)
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(GovernedControllerError, match="exact policy prefix"):
        load_controller_state(path, plan)


def test_state_rejects_forged_decision_history_and_status(tmp_path: Path) -> None:
    plan = _plan()
    state = _progress_state(plan, completed_count=1)
    state["gate_history"][0]["decision"]["action"] = "complete"  # type: ignore[index]
    _seal_state(state)
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(GovernedControllerError, match=r"decision.*fingerprint"):
        load_controller_state(path, plan)

    state = _progress_state(plan, completed_count=1)
    state["status"] = "not_started"
    _seal_state(state)
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(GovernedControllerError, match=r"not_started.*progress"):
        load_controller_state(path, plan)


def test_state_writer_seals_and_reloads_an_exact_threshold_prefix(tmp_path: Path) -> None:
    plan = _plan()
    state = _progress_state(plan, completed_count=2)
    state.pop("state_fingerprint")
    path = tmp_path / "state.json"
    sealed = write_controller_state(path, state, plan=plan)
    assert sealed["state_fingerprint"]
    assert load_controller_state(path, plan) == sealed


def test_train_command_uses_fork_once_and_terminal_milestone_has_no_pause_flag() -> None:
    plan = _plan()
    state = initial_controller_state(plan)
    first = build_train_command(plan, state)
    assert first[first.index("--resume") + 1] == "none"
    assert "--fork-from" in first
    assert first[first.index("--fork-from") + 1] == plan["fork"]["path"]  # type: ignore[index]
    assert first[first.index("--expected-config-sha256") + 1] == plan["config"]["sha256"]  # type: ignore[index]
    assert first[first.index("--pause-at-tokens") + 1] == "13000000"

    state = _progress_state(plan, completed_count=8)
    final = build_train_command(plan, state)
    assert final[final.index("--resume") + 1] == "/tmp/step-8"
    assert "--fork-from" not in final
    assert "--pause-at-tokens" not in final


def test_controller_source_drift_is_rejected(tmp_path: Path) -> None:
    plan = deepcopy(_plan())
    source = tmp_path / "controller.py"
    source.write_text("old\n", encoding="utf-8")
    plan["controller_sources"] = [{"path": str(source), "sha256": "0" * 64}]
    with pytest.raises(GovernedControllerError, match="source changed"):
        verify_controller_sources(plan)


def test_plan_bound_config_and_readiness_replacement_are_rejected(
    tmp_path: Path,
) -> None:
    plan = deepcopy(_plan())
    config_source = Path(str(plan["config"]["path"]))  # type: ignore[index]
    config_path = tmp_path / "formal.yaml"
    config_path.write_bytes(config_source.read_bytes())
    plan["config"]["path"] = str(config_path.resolve())  # type: ignore[index]
    plan["config"]["sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()  # type: ignore[index]
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_bytes(READINESS.read_bytes())
    plan["readiness"] = {
        "path": str(readiness_path.resolve()),
        "sha256": hashlib.sha256(readiness_path.read_bytes()).hexdigest(),
    }
    _refingerprint(plan)
    verify_controller_sources(plan)

    config_path.write_bytes(config_path.read_bytes() + b"\n# replaced after plan\n")
    with pytest.raises(GovernedControllerError, match="config changed after planning"):
        verify_controller_sources(plan)
    config_path.write_bytes(config_source.read_bytes())

    readiness_path.write_bytes(readiness_path.read_bytes() + b"\n")
    with pytest.raises(
        GovernedControllerError,
        match="readiness changed after planning",
    ):
        verify_controller_sources(plan)


def test_refingerprinted_plan_cannot_authorize_changed_config_semantics(
    tmp_path: Path,
) -> None:
    plan = deepcopy(_plan())
    raw = yaml.safe_load(Path(str(plan["config"]["path"])).read_text(encoding="utf-8"))  # type: ignore[index]
    raw["runtime"]["profile_active_steps"] = 4
    config_path = tmp_path / "formal.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")
    plan["config"]["path"] = str(config_path.resolve())  # type: ignore[index]
    plan["config"]["sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()  # type: ignore[index]
    _refingerprint(plan)
    with pytest.raises(
        GovernedControllerError,
        match="config normalized semantics changed after planning",
    ):
        verify_controller_sources(plan)


def test_full_source_tree_and_dependency_lock_drift_are_rejected(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "src" / "twen"
    source_root.mkdir(parents=True)
    evaluation = source_root / "evaluation.py"
    evaluation.write_text("VALUE = 1\n", encoding="utf-8")
    controller = tmp_path / "govern.py"
    controller.write_text("VALUE = 1\n", encoding="utf-8")
    dependency = tmp_path / "uv.lock"
    dependency.write_text("version = 1\n", encoding="utf-8")
    plan = deepcopy(_plan())
    plan["source_tree"] = {
        "path": str(source_root),
        "sha256": twen_source_tree_sha256(source_root),
    }
    plan["dependency_lock"] = {
        "path": str(dependency),
        "sha256": hashlib.sha256(dependency.read_bytes()).hexdigest(),
    }
    plan["controller_sources"] = [
        {
            "path": str(controller),
            "sha256": hashlib.sha256(controller.read_bytes()).hexdigest(),
        }
    ]
    verify_controller_sources(plan)

    evaluation.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(GovernedControllerError, match="source tree changed"):
        verify_controller_sources(plan)
    evaluation.write_text("VALUE = 1\n", encoding="utf-8")
    dependency.write_text("version = 2\n", encoding="utf-8")
    with pytest.raises(GovernedControllerError, match="dependency lock changed"):
        verify_controller_sources(plan)


def test_checkpoint_source_maps_cross_authenticate_all_four_identities() -> None:
    plan, metadata = _checkpoint_source_map_fixture()
    authenticated = _authenticate_checkpoint_source_maps(plan, metadata)
    assert set(authenticated) == {"primary", "cooldown"}
    assert (
        authenticated["primary"]["sha256"]
        == plan["training_data"]["source_map_sha256"]  # type: ignore[index]
    )


def test_checkpoint_source_map_rejects_self_consistent_capacity_forgery() -> None:
    plan, metadata = _checkpoint_source_map_fixture()
    forged = deepcopy(metadata)
    cursor = forged["data_cursor"]["extra"]  # type: ignore[index]
    cooldown_map = cursor["cooldown_cursor"]["source_map"]  # type: ignore[index]
    cooldown_map["shards"][0]["output_sha256"] = "9" * 64  # type: ignore[index]
    forged_sha = _source_map_sha(cooldown_map)
    source_mix = forged["extra"]["source_mix"]  # type: ignore[index]
    source_mix["phases"]["cooldown"]["source_map_sha256"] = forged_sha  # type: ignore[index]
    with pytest.raises(
        GovernedControllerError,
        match=r"cooldown source-map identity differs",
    ):
        _authenticate_checkpoint_source_maps(plan, forged)


def test_checkpoint_source_map_rejects_log_cursor_and_final_config_mismatch() -> None:
    plan, metadata = _checkpoint_source_map_fixture()
    forged_log = deepcopy(metadata)
    forged_log["extra"]["source_mix"]["phases"]["primary"]["source_map_sha256"] = (  # type: ignore[index]
        "8" * 64
    )
    with pytest.raises(
        GovernedControllerError,
        match=r"top-level source mix differs",
    ):
        _authenticate_checkpoint_source_maps(plan, forged_log)

    forged_plan = deepcopy(plan)
    forged_plan["training_data"]["source_map_sha256"] = "7" * 64  # type: ignore[index]
    with pytest.raises(
        GovernedControllerError,
        match=r"primary source map differs from the final config",
    ):
        _authenticate_checkpoint_source_maps(forged_plan, metadata)


def test_drift_authentication_recomputes_tensors_instead_of_trusting_json(
    tmp_path: Path,
) -> None:
    plan = _plan()
    fork = plan["fork"]
    assert isinstance(fork, dict)
    checkpoint = {
        "path": str(tmp_path / "candidate"),
        "manifest_sha256": "a" * 64,
        "complete_sha256": "b" * 64,
    }
    recomputed = {
        "kind": "twen_dense_checkpoint_trainable_drift_audit",
        "schema_version": 1,
        "execution": {
            "device": "cpu",
            "cuda_initialized": False,
            "model_built": False,
            "optimizer_created": False,
        },
        "baseline": {
            "path": str(Path(str(fork["path"])).resolve()),
            "manifest_sha256": "c" * 64,
            "complete_sha256": fork["complete_sha256"],
        },
        "inventory": {},
        "candidates": [
            {
                **checkpoint,
                "adapter": {"relative_l2": 0.02},
                "scale": {"relative_l2": 0.03},
                "scale_values": {},
            }
        ],
    }
    forged = deepcopy(recomputed)
    forged["candidates"][0]["scale"]["relative_l2"] = 0.001  # type: ignore[index]
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(forged), encoding="utf-8")
    module = SimpleNamespace(audit=lambda _baseline, _candidates: recomputed)
    with (
        patch("twen.governed._load_bound_script", return_value=module),
        pytest.raises(GovernedControllerError, match="tensor recomputation"),
    ):
        authenticate_drift_evidence(plan, checkpoint, path)

    path.write_text(json.dumps(recomputed), encoding="utf-8")
    with patch("twen.governed._load_bound_script", return_value=module):
        result = authenticate_drift_evidence(plan, checkpoint, path)
    assert result["authenticated"] is True
    assert result["scale_relative_l2"] == pytest.approx(0.03)


def test_sweep_authentication_rebuilds_real_evaluator_evidence(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "prepared.json"
    prepared.write_text("{}\n", encoding="utf-8")
    baseline_root = tmp_path / "baseline"
    baseline_root.mkdir()
    candidate_root = tmp_path / "candidate-evaluation"
    candidate_root.mkdir()
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    checkpoint = {
        "path": str(checkpoint_path),
        "manifest_sha256": "a" * 64,
        "complete_sha256": "b" * 64,
        "metadata": {
            "global_step": 50,
            "committed_tokens": 13_107_200,
            "kind": "milestone",
            "tag": "governed-000013000000",
        },
    }
    (candidate_root / "PLAN.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_complete_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    baseline_identity = {
        name: {"sha256": character * 64}
        for name, character in (
            ("manifest", "c"),
            ("complete", "d"),
            ("plan", "e"),
        )
    }
    plan = deepcopy(_plan())
    plan["formal_evidence"] = {
        "validation_phases": {
            "primary": {
                "prepared": {
                    "path": str(prepared),
                    "sha256": "f" * 64,
                },
                "v3_evaluation": {
                    "path": str(baseline_root),
                    **baseline_identity,
                },
            }
        }
    }
    summary = {
        "kind": "twen_v4_checkpoint_frozen_validation_sweep",
        "target_role": "candidate",
        "training_started_by_summarizer": False,
        "inputs_mutated_by_summarizer": False,
        "prepared_manifest": {
            "path": str(prepared),
            "sha256": "f" * 64,
        },
        "baseline": {
            "evaluation": {
                "root": str(baseline_root),
                **baseline_identity,
            }
        },
        "candidates": [
            {
                "checkpoint_state": dict(checkpoint["metadata"]),
                "evaluation": {"root": str(candidate_root)},
            }
        ],
    }
    sweep_root = tmp_path / "sweep"
    sweep_root.mkdir()
    (sweep_root / "MANIFEST.json").write_text("{}\n", encoding="utf-8")
    (sweep_root / "COMPLETE").write_text("x\n", encoding="utf-8")
    module = SimpleNamespace(build_summary=lambda **_kwargs: {**summary, "forged": False})
    with (
        patch("twen.governed._authenticate_output_bundle", return_value=({}, summary)),
        patch("twen.governed._load_bound_script", return_value=module),
        pytest.raises(GovernedControllerError, match="fresh evaluator authentication"),
    ):
        _authenticate_generated_sweep(
            plan,
            checkpoint,
            phase="primary",
            root=sweep_root,
        )

    module = SimpleNamespace(build_summary=lambda **_kwargs: summary)
    with (
        patch("twen.governed._authenticate_output_bundle", return_value=({}, summary)),
        patch("twen.governed._load_bound_script", return_value=module),
    ):
        authenticated = _authenticate_generated_sweep(
            plan,
            checkpoint,
            phase="primary",
            root=sweep_root,
        )
    assert authenticated["candidate"] == summary["candidates"][0]


def test_canonical_metrics_requires_complete_step_prefix(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(json.dumps({"step": step}) for step in (1, 2, 3)) + "\n",
        encoding="utf-8",
    )
    assert len(read_canonical_metrics(path, through_step=3)) == 3
    path.write_text(
        "\n".join(json.dumps({"step": step}) for step in (1, 3)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GovernedControllerError, match=r"complete 1\.\.3 prefix"):
        read_canonical_metrics(path, through_step=3)


def test_checkpoint_authenticated_metrics_prefix_rejects_modify_truncate_and_reorder(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    logger = JsonlMetricLogger(path)
    for step in range(1, 4):
        assert logger.log(step, {"tokens": step * 100, "loss": 2.0 - step / 10})
    binding = logger.snapshot_prefix(through_step=3, committed_tokens=300)
    metadata = {
        "global_step": 3,
        "committed_tokens": 300,
        "extra": {"metrics_prefix": binding},
    }
    rows, authenticated = read_authenticated_metrics_prefix(path, metadata)
    assert [row["step"] for row in rows] == [1, 2, 3]
    assert authenticated == binding

    original = path.read_bytes()
    mutations = [
        original.replace(b'"loss":1.9', b'"loss":9.9', 1),
        original[:-1],
        b"\n".join(reversed(original.rstrip(b"\n").splitlines())) + b"\n",
    ]
    for payload in mutations:
        path.write_bytes(payload)
        with pytest.raises(
            GovernedControllerError,
            match="checkpoint-authenticated bytes",
        ):
            read_authenticated_metrics_prefix(path, metadata)
    path.write_bytes(original)


def test_checkpoint_execution_identity_must_match_source_tree_and_dependency() -> None:
    plan = deepcopy(_plan())
    config = plan["config"]
    source_tree = plan["source_tree"]
    dependency = plan["dependency_lock"]
    assert isinstance(config, dict)
    assert isinstance(source_tree, dict)
    assert isinstance(dependency, dict)
    preflight_fingerprint = "f" * 64
    config["preflight_fingerprint"] = preflight_fingerprint
    metadata = {
        "critical_fingerprint": preflight_fingerprint,
        "extra": {
            "config": {
                "path": config["path"],
                "sha256": config["sha256"],
                "preflight_fingerprint": preflight_fingerprint,
            },
            "source_tree_sha256": source_tree["sha256"],
            "dependency_lock": dependency["path"],
            "dependency_lock_sha256": dependency["sha256"],
        }
    }
    assert _authenticate_checkpoint_execution_identity(plan, metadata) == {
        "config_path": config["path"],
        "config_sha256": config["sha256"],
        "config_preflight_fingerprint": preflight_fingerprint,
        "source_tree_sha256": source_tree["sha256"],
        "dependency_lock": dependency["path"],
        "dependency_lock_sha256": dependency["sha256"],
    }
    forged = deepcopy(metadata)
    forged["extra"]["source_tree_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(GovernedControllerError, match="source-tree identity"):
        _authenticate_checkpoint_execution_identity(plan, forged)
    forged = deepcopy(metadata)
    forged["extra"]["dependency_lock_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(GovernedControllerError, match="dependency-lock identity"):
        _authenticate_checkpoint_execution_identity(plan, forged)
    forged = deepcopy(metadata)
    forged["extra"]["config"]["sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(GovernedControllerError, match="config identity"):
        _authenticate_checkpoint_execution_identity(plan, forged)
    forged = deepcopy(metadata)
    forged["extra"]["config"]["path"] = "/tmp/another-formal-config.yaml"  # type: ignore[index]
    with pytest.raises(GovernedControllerError, match="config identity"):
        _authenticate_checkpoint_execution_identity(plan, forged)
    forged = deepcopy(metadata)
    forged["extra"]["config"]["preflight_fingerprint"] = "0" * 64  # type: ignore[index]
    with pytest.raises(GovernedControllerError, match="config identity"):
        _authenticate_checkpoint_execution_identity(plan, forged)
    forged = deepcopy(metadata)
    forged["critical_fingerprint"] = "0" * 64
    with pytest.raises(GovernedControllerError, match="config identity"):
        _authenticate_checkpoint_execution_identity(plan, forged)


def test_checkpoint_execution_identity_fails_before_evaluation_artifacts(
    tmp_path: Path,
) -> None:
    script = _load_controller_script()
    plan = deepcopy(_plan())
    checkpoint_binding = {
        "path": str(tmp_path / "checkpoint"),
        "manifest_sha256": "a" * 64,
        "complete_sha256": "b" * 64,
        "global_step": 50,
        "committed_tokens": 13_107_200,
        "kind": "milestone",
        "tag": "governed-000013000000",
    }
    authenticated = {
        **checkpoint_binding,
        "metadata": {
            "global_step": 50,
            "committed_tokens": 13_107_200,
            "kind": "milestone",
            "tag": "governed-000013000000",
        },
    }
    state = {"gate_history": []}
    with (
        patch.object(script, "authenticate_checkpoint", return_value=authenticated),
        patch.object(
            script,
            "authenticate_checkpoint_execution_identity",
            side_effect=GovernedControllerError("checkpoint config identity differs"),
        ) as execution_identity,
        patch.object(script, "_ensure_threshold_artifacts") as artifacts,
        pytest.raises(GovernedControllerError, match="config identity differs"),
    ):
        script._build_threshold_decision(
            plan,
            state,
            checkpoint_binding,
            threshold=13_000_000,
        )
    execution_identity.assert_called_once_with(plan, authenticated)
    assert not artifacts.called


def test_all_authenticated_gates_resume_and_final_threshold_completes() -> None:
    plan = _plan()
    assert evaluate_hard_gates(plan, _observation())["action"] == "resume"
    final = _observation(committed_tokens=250_000_000)
    assert evaluate_hard_gates(plan, final)["action"] == "complete"


def test_reuse_nonfinite_and_scale_drift_are_hard_stops() -> None:
    plan = _plan()
    for mutation in ("reuse", "nonfinite", "drift"):
        observation = _observation()
        if mutation == "reuse":
            observation["checkpoint"]["reused_sequences"] = 1  # type: ignore[index]
        elif mutation == "nonfinite":
            observation["metrics"][0]["loss"] = float("nan")  # type: ignore[index]
        else:
            observation["drift"]["scale_relative_l2"] = 0.051  # type: ignore[index]
        decision = evaluate_hard_gates(plan, observation)
        assert decision["action"] == "stop"


def test_one_clip_in_rolling_fifty_requires_review() -> None:
    observation = _observation()
    observation["metrics"][-1]["grad_norm"] = 1.01  # type: ignore[index]
    decision = evaluate_hard_gates(_plan(), observation)
    assert decision["action"] == "review"
    check = next(row for row in decision["checks"] if row["code"] == "rolling_clip_fraction")
    assert check["observed"] == pytest.approx(0.02)


def test_two_consecutive_nll_regressions_hard_stop() -> None:
    decision = evaluate_hard_gates(
        _plan(),
        _observation(aggregate_nll=2.006),
        history=[
            {
                "checkpoint_id": "previous",
                "aggregate_nll": 2.006,
                "chinese_nll": 3.0,
            }
        ],
    )
    assert decision["action"] == "stop"
    assert "aggregate_nll_consecutive_regression" in decision["failures"]["stop"]


def test_plateau_checks_two_consecutive_intervals_and_tie_keeps_earlier() -> None:
    decision = evaluate_hard_gates(
        _plan(),
        _observation(aggregate_nll=1.79991),
        history=[
            {
                "checkpoint_id": "best",
                "aggregate_nll": 1.8,
                "chinese_nll": 2.9,
            },
            {
                "checkpoint_id": "middle",
                "aggregate_nll": 1.79995,
                "chinese_nll": 2.89,
            },
        ],
    )
    check = next(row for row in decision["checks"] if row["code"] == "two_interval_improvement")
    assert check["observed"] == pytest.approx([0.00005, 0.00004])
    assert decision["action"] == "terminal_stop"

    tie = evaluate_hard_gates(
        _plan(),
        _observation(aggregate_nll=1.89995),
        history=[
            {
                "checkpoint_id": "earlier",
                "aggregate_nll": 1.9,
                "chinese_nll": 2.9,
            }
        ],
    )
    assert tie["selection"]["checkpoint_id"] == "earlier"

    non_chained_tie = evaluate_hard_gates(
        _plan(),
        _observation(aggregate_nll=1.89982),
        history=[
            {
                "checkpoint_id": "outside-global-tolerance",
                "aggregate_nll": 1.9,
                "chinese_nll": 2.9,
            },
            {
                "checkpoint_id": "earliest-near-global-min",
                "aggregate_nll": 1.89991,
                "chinese_nll": 2.89,
            },
        ],
    )
    assert (
        non_chained_tie["selection"]["checkpoint_id"]
        == "earliest-near-global-min"
    )


def test_controller_status_and_dry_run_never_create_state(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = _load_controller_script()
    state = tmp_path / "state.json"
    for action in ("status", "dry-run"):
        assert (
            script.main(
                [
                    "--readiness",
                    str(READINESS),
                    "--state",
                    str(state),
                    "--action",
                    action,
                ]
            )
            == 0
        )
        assert not state.exists()
        payload = json.loads(capsys.readouterr().out)
        assert payload["blocked"] is True
        if action == "dry-run":
            assert payload["executes"] is False


def test_controller_rejects_arbitrary_evaluate_payload_and_never_runs_blocked_segment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = _load_controller_script()
    with pytest.raises(SystemExit):
        script._parser().parse_args(["--action", "evaluate", "--observation", "fake.json"])

    plan = _plan()
    state = tmp_path / "state.json"
    with patch.object(script.subprocess, "run") as run:
        assert (
            script.main(
                [
                    "--readiness",
                    str(READINESS),
                    "--state",
                    str(state),
                    "--action",
                    "run",
                    "--ack",
                    expected_run_ack(plan),
                ]
            )
            == 2
        )
    assert not run.called
    assert not state.exists()
    assert "launch is blocked" in capsys.readouterr().err


def test_run_rechecks_controller_sources_immediately_before_subprocess(
    tmp_path: Path,
) -> None:
    script = _load_controller_script()
    plan = deepcopy(_plan())
    plan["launch_enabled"] = True
    plan["readiness_issues"] = []
    _refingerprint(plan)
    args = script._parser().parse_args(
        ["--action", "run", "--ack", expected_run_ack(plan)]
    )
    state = tmp_path / "state.json"
    with (
        patch.object(
            script,
            "verify_controller_sources",
            side_effect=GovernedControllerError("source race"),
        ) as verify,
        patch.object(script.subprocess, "run") as run,
        pytest.raises(GovernedControllerError, match="source race"),
    ):
        script._run(args, plan, state)
    verify.assert_called_once_with(plan)
    assert not run.called
    assert not state.exists()


def test_evaluation_subprocess_rechecks_execution_chain_before_and_after(
    tmp_path: Path,
) -> None:
    script = _load_controller_script()
    plan = _authorized_two_threshold_plan(tmp_path)
    completed = SimpleNamespace(returncode=0)
    with (
        patch.object(script, "verify_controller_sources") as verify,
        patch.object(script.subprocess, "run", return_value=completed) as run,
    ):
        script._run_checked(
            [sys.executable, "-m", "twen.cli", "evaluate"],
            label="fixture evaluation",
            plan=plan,
        )
    assert verify.call_count == 2
    run.assert_called_once_with(
        [sys.executable, "-m", "twen.cli", "evaluate"],
        check=False,
        cwd=str(plan["project_root"]),
    )


def test_run_closes_two_segments_through_gates_and_resumes_automatically(
    tmp_path: Path,
) -> None:
    script = _load_controller_script()
    plan = _authorized_two_threshold_plan(tmp_path)
    state_path = tmp_path / "controller-state.json"
    args = script._parser().parse_args(
        ["--action", "run", "--ack", expected_run_ack(plan)]
    )

    def reconcile(
        actual_state_path: Path,
        actual_plan: dict[str, object],
        state: dict[str, object],
        *,
        launch_if_needed: bool,
    ) -> dict[str, object]:
        assert launch_if_needed is True
        threshold = state["active_threshold"]
        assert isinstance(threshold, int)
        checkpoint = {
            "path": str(tmp_path / f"step-{threshold}"),
            "manifest_sha256": f"{threshold:064x}",
            "complete_sha256": f"{threshold + 1:064x}",
            "global_step": threshold,
            "committed_tokens": threshold,
            "kind": "milestone",
            "tag": "complete" if threshold == 20 else "governed-000000000010",
        }
        return script._transition_to_awaiting(
            actual_state_path,
            actual_plan,
            state,
            checkpoint=checkpoint,
            exit_code=0,
        )

    def decide(
        actual_plan: dict[str, object],
        _state: dict[str, object],
        _checkpoint: dict[str, object],
        *,
        threshold: int,
        exclude_last_history: bool = False,
    ) -> tuple[dict[str, object], dict[str, object]]:
        assert exclude_last_history is False
        return (
            _decision(
                actual_plan,
                "complete" if threshold == 20 else "resume",
                nll=1.9 - threshold / 100_000,
            ),
            {"threshold": threshold, "authenticated": True},
        )

    with (
        patch.object(script, "_reconcile_running_segment", side_effect=reconcile) as segments,
        patch.object(script, "_build_threshold_decision", side_effect=decide) as decisions,
    ):
        result = script._run(args, plan, state_path)

    assert result["ok"] is True
    assert result["action"] == "complete"
    assert segments.call_count == 2
    assert decisions.call_count == 2
    final = load_controller_state(state_path, plan)
    assert final["status"] == "completed"
    assert final["completed_thresholds"] == [10, 20]
    assert [row["decision"]["action"] for row in final["gate_history"]] == [
        "resume",
        "complete",
    ]


def test_running_state_reconciles_checkpoint_written_before_state_update(
    tmp_path: Path,
) -> None:
    script = _load_controller_script()
    plan = _authorized_two_threshold_plan(tmp_path)
    state_path = tmp_path / "state.json"
    state = initial_controller_state(plan)
    state.update(
        {
            "status": "running",
            "active_threshold": 10,
            "active_command": build_train_command(plan, state),
        }
    )
    _seal_state(state)
    state = write_controller_state(state_path, state, plan=plan)
    checkpoint = {
        "path": str(tmp_path / "step-10"),
        "manifest_sha256": "a" * 64,
        "complete_sha256": "b" * 64,
        "global_step": 10,
        "committed_tokens": 10,
        "kind": "milestone",
        "tag": "governed-000000000010",
    }
    with (
        patch.object(script, "_checkpoint_after_segment", return_value=checkpoint),
        patch.object(script.subprocess, "run") as run,
    ):
        reconciled = script._reconcile_running_segment(
            state_path,
            plan,
            state,
            launch_if_needed=True,
        )
    assert not run.called
    assert reconciled["status"] == "awaiting_evaluation"
    assert load_controller_state(state_path, plan)["current_checkpoint"] == checkpoint


def test_running_state_retries_from_plan_after_pre_subprocess_crash(
    tmp_path: Path,
) -> None:
    script = _load_controller_script()
    plan = _authorized_two_threshold_plan(tmp_path)
    state_path = tmp_path / "state.json"
    state = initial_controller_state(plan)
    state.update(
        {
            "status": "running",
            "active_threshold": 10,
            "active_command": build_train_command(plan, state),
        }
    )
    _seal_state(state)
    state = write_controller_state(state_path, state, plan=plan)
    checkpoint = {
        "path": str(tmp_path / "step-10"),
        "manifest_sha256": "a" * 64,
        "complete_sha256": "b" * 64,
        "global_step": 10,
        "committed_tokens": 10,
        "kind": "milestone",
        "tag": "governed-000000000010",
    }
    completed = SimpleNamespace(returncode=0)
    with (
        patch.object(
            script,
            "_checkpoint_after_segment",
            side_effect=[
                GovernedControllerError("not yet"),
                checkpoint,
            ],
        ),
        patch.object(script, "_active_training_pid", return_value=None),
        patch.object(script.subprocess, "run", return_value=completed) as run,
    ):
        reconciled = script._reconcile_running_segment(
            state_path,
            plan,
            state,
            launch_if_needed=True,
        )
    run.assert_called_once()
    assert reconciled["status"] == "awaiting_evaluation"


def test_same_length_history_tamper_during_training_subprocess_is_replayed(
    tmp_path: Path,
) -> None:
    script = _load_controller_script()
    plan = _authorized_two_threshold_plan(tmp_path)
    state_path = tmp_path / "state.json"
    state = _progress_state(plan, completed_count=1)
    original = deepcopy(state["gate_history"])
    state_path.write_text(json.dumps(state), encoding="utf-8")
    args = script._parser().parse_args(
        ["--action", "run", "--ack", expected_run_ack(plan)]
    )
    checkpoint = {
        "path": str(tmp_path / "step-20"),
        "manifest_sha256": "a" * 64,
        "complete_sha256": "b" * 64,
        "global_step": 20,
        "committed_tokens": 20,
        "kind": "milestone",
        "tag": "complete",
    }

    def tamper_during_subprocess(
        _command: list[str],
        *,
        check: bool,
        cwd: str,
    ) -> SimpleNamespace:
        assert check is False
        assert cwd == str(plan["project_root"])
        forged = json.loads(state_path.read_text(encoding="utf-8"))
        first_decision = forged["gate_history"][0]["decision"]
        first_decision["nll"]["aggregate"] = 0.01
        first_decision["decision_fingerprint"] = hashlib.sha256(
            json.dumps(
                {
                    key: value
                    for key, value in first_decision.items()
                    if key != "decision_fingerprint"
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        _seal_state(forged)
        state_path.write_text(json.dumps(forged), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    def authentic_rebuild(
        _plan_value: dict[str, object],
        replay_state: dict[str, object],
        _checkpoint: dict[str, object],
        *,
        threshold: int,
        exclude_last_history: bool = False,
    ) -> tuple[dict[str, object], dict[str, object]]:
        del exclude_last_history
        replay = replay_state["gate_history"]
        assert isinstance(replay, list)
        row = original[len(replay)]
        assert threshold == row["threshold"]
        return deepcopy(row["decision"]), deepcopy(row["evidence"])

    with (
        patch.object(
            script,
            "_checkpoint_after_segment",
            side_effect=[
                GovernedControllerError("not yet"),
                checkpoint,
            ],
        ),
        patch.object(script, "_active_training_pid", return_value=None),
        patch.object(script, "verify_controller_sources"),
        patch.object(
            script,
            "_build_threshold_decision",
            side_effect=authentic_rebuild,
        ) as rebuild,
        patch.object(
            script.subprocess,
            "run",
            side_effect=tamper_during_subprocess,
        ),
        pytest.raises(
            GovernedControllerError,
            match=r"gate_history\[0\].*full artifact reauthentication",
        ),
    ):
        script._run(args, plan, state_path)
    assert rebuild.call_count == 2


def _write_running_session_fixture(
    script: ModuleType,
    plan: dict[str, object],
    state: dict[str, object],
    *,
    pid: int = 4242,
) -> tuple[Path, list[str]]:
    active_command = state["active_command"]
    assert isinstance(active_command, list)
    payload = script._training_payload(active_command, label="fixture")
    process_cmdline = [sys.executable, "-u", *payload]
    run = plan["run"]
    config = plan["config"]
    source_tree = plan["source_tree"]
    dependency = plan["dependency_lock"]
    assert all(
        isinstance(value, dict)
        for value in (run, config, source_tree, dependency)
    )
    session_path = Path(str(run["output_dir"])) / "rank0-session.json"
    session_path.parent.mkdir(parents=True, exist_ok=True)
    session_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_id": "fixture-session",
                "pid": pid,
                "status": "running",
                "hostname": platform.node(),
                "process_start_time_ticks": 123456,
                "process_cmdline": process_cmdline,
                "argv": process_cmdline[2:],
                "cwd": plan["project_root"],
                "config_path": config["path"],
                "config_sha256": config["sha256"],
                "config_fingerprint": "f" * 64,
                "run_id": run["run_id"],
                "stage": run["stage"],
                "world_size": run["world_size"],
                "source_tree_sha256": source_tree["sha256"],
                "dependency_lock": dependency["path"],
                "dependency_lock_sha256": dependency["sha256"],
            }
        ),
        encoding="utf-8",
    )
    return session_path, process_cmdline


def test_active_training_pid_requires_complete_matching_session_identity(
    tmp_path: Path,
) -> None:
    script = _load_controller_script()
    plan = _authorized_two_threshold_plan(tmp_path)
    state = initial_controller_state(plan)
    state.update(
        {
            "status": "running",
            "active_threshold": 10,
            "active_command": build_train_command(plan, state),
        }
    )
    _seal_state(state)
    _session_path, process_cmdline = _write_running_session_fixture(
        script,
        plan,
        state,
    )
    with (
        patch.object(script, "_process_exists", return_value=True),
        patch.object(script, "_process_start_time_ticks", return_value=123456),
        patch.object(script, "_process_cmdline", return_value=process_cmdline),
        patch.object(
            script,
            "_process_cwd",
            return_value=Path(str(plan["project_root"])).resolve(),
        ),
    ):
        assert script._active_training_pid(plan, state) == 4242


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("pid_reuse", "process_start_time_ticks"),
        ("cmdline", "process_cmdline"),
        ("cwd", "cwd"),
        ("config", "config"),
    ],
)
def test_active_training_pid_rejects_reuse_and_wrong_process_identity(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    script = _load_controller_script()
    plan = _authorized_two_threshold_plan(tmp_path)
    state = initial_controller_state(plan)
    state.update(
        {
            "status": "running",
            "active_threshold": 10,
            "active_command": build_train_command(plan, state),
        }
    )
    _seal_state(state)
    session_path, process_cmdline = _write_running_session_fixture(
        script,
        plan,
        state,
    )
    start_ticks = 999 if mutation == "pid_reuse" else 123456
    actual_cmdline = (
        [*process_cmdline, "--forged"] if mutation == "cmdline" else process_cmdline
    )
    actual_cwd = (
        tmp_path / "wrong-cwd"
        if mutation == "cwd"
        else Path(str(plan["project_root"])).resolve()
    )
    if mutation == "config":
        session = json.loads(session_path.read_text(encoding="utf-8"))
        session["config_sha256"] = "0" * 64
        session_path.write_text(json.dumps(session), encoding="utf-8")
    with (
        patch.object(script, "_process_exists", return_value=True),
        patch.object(script, "_process_start_time_ticks", return_value=start_ticks),
        patch.object(script, "_process_cmdline", return_value=actual_cmdline),
        patch.object(script, "_process_cwd", return_value=actual_cwd),
        pytest.raises(GovernedControllerError, match=expected),
    ):
        script._active_training_pid(plan, state)


def test_completed_evaluation_artifacts_are_reused_after_controller_crash(
    tmp_path: Path,
) -> None:
    script = _load_controller_script()
    plan = _authorized_two_threshold_plan(tmp_path)
    root = script._threshold_root(plan, 10)
    (root / "drift.json").parent.mkdir(parents=True)
    (root / "drift.json").write_text("{}\n", encoding="utf-8")
    for phase in ("primary", "cooldown"):
        evaluation = root / "evaluations" / phase
        sweep = root / "sweeps" / phase
        evaluation.mkdir(parents=True)
        sweep.mkdir(parents=True)
        (evaluation / "COMPLETE").write_text("done\n", encoding="utf-8")
        (sweep / "COMPLETE").write_text("done\n", encoding="utf-8")
    with (
        patch.object(script, "generate_drift_evidence") as drift,
        patch.object(script, "generate_checkpoint_sweep") as sweep,
        patch.object(script, "_run_checked") as run,
    ):
        drift_path, sweep_roots = script._ensure_threshold_artifacts(
            plan,
            {"path": "/tmp/checkpoint"},
            threshold=10,
        )
    assert drift_path == root / "drift.json"
    assert set(sweep_roots) == {"primary", "cooldown"}
    assert not drift.called
    assert not sweep.called
    assert not run.called


def test_resume_reauthenticates_latest_gate_before_next_training_segment(
    tmp_path: Path,
) -> None:
    script = _load_controller_script()
    plan = _authorized_two_threshold_plan(tmp_path)
    state = _progress_state(plan, completed_count=1)
    with (
        patch.object(
            script,
            "_build_threshold_decision",
            return_value=(
                _decision(plan, "resume", nll=9.0),
                {"fixture": "changed"},
            ),
        ) as build,
        pytest.raises(GovernedControllerError, match="full artifact reauthentication"),
    ):
        script._reauthenticate_latest_resume(plan, state)
    build.assert_called_once()


def test_full_gate_history_is_rebuilt_oldest_first_from_verified_results(
    tmp_path: Path,
) -> None:
    del tmp_path
    script = _load_controller_script()
    plan = _plan()
    state = _progress_state(plan, completed_count=3)
    original = deepcopy(state["gate_history"])
    calls = 0

    def rebuild(
        _plan_value: dict[str, object],
        replay_state: dict[str, object],
        _checkpoint: dict[str, object],
        *,
        threshold: int,
        exclude_last_history: bool = False,
    ) -> tuple[dict[str, object], dict[str, object]]:
        nonlocal calls
        assert exclude_last_history is False
        replay = replay_state["gate_history"]
        assert isinstance(replay, list)
        assert len(replay) == calls
        if calls:
            assert replay[-1]["decision"] == original[calls - 1]["decision"]
        row = original[calls]
        assert threshold == row["threshold"]
        calls += 1
        return deepcopy(row["decision"]), deepcopy(row["evidence"])

    with patch.object(script, "_build_threshold_decision", side_effect=rebuild):
        rebuilt = script._reauthenticate_gate_history(plan, state)
    assert calls == 3
    assert rebuilt == original


def test_forged_early_nll_with_all_self_fingerprints_recomputed_is_rejected() -> None:
    script = _load_controller_script()
    plan = _plan()
    state = _progress_state(plan, completed_count=3)
    original = deepcopy(state["gate_history"])
    first = state["gate_history"][0]  # type: ignore[index]
    first_decision = first["decision"]
    first_decision["nll"]["aggregate"] = 0.01
    first_decision["decision_fingerprint"] = hashlib.sha256(
        json.dumps(
            {
                key: value
                for key, value in first_decision.items()
                if key != "decision_fingerprint"
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    _seal_state(state)

    def authentic_rebuild(
        _plan_value: dict[str, object],
        replay_state: dict[str, object],
        _checkpoint: dict[str, object],
        *,
        threshold: int,
        exclude_last_history: bool = False,
    ) -> tuple[dict[str, object], dict[str, object]]:
        del exclude_last_history
        replay = replay_state["gate_history"]
        assert isinstance(replay, list)
        index = len(replay)
        row = original[index]
        assert threshold == row["threshold"]
        return deepcopy(row["decision"]), deepcopy(row["evidence"])

    with (
        patch.object(
            script,
            "_build_threshold_decision",
            side_effect=authentic_rebuild,
        ),
        pytest.raises(
            GovernedControllerError,
            match=r"gate_history\[0\].*full artifact reauthentication",
        ),
    ):
        script._reauthenticate_gate_history(plan, state)


def test_completed_reentry_rejects_forged_historical_conclusion(
    tmp_path: Path,
) -> None:
    script = _load_controller_script()
    plan = _authorized_two_threshold_plan(tmp_path)
    state = _progress_state(plan, completed_count=2, status="completed")
    original = deepcopy(state["gate_history"])
    first = state["gate_history"][0]  # type: ignore[index]
    first_decision = first["decision"]
    first_decision["nll"]["aggregate"] = 0.01
    first_decision["decision_fingerprint"] = hashlib.sha256(
        json.dumps(
            {
                key: value
                for key, value in first_decision.items()
                if key != "decision_fingerprint"
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    _seal_state(state)
    state_path = tmp_path / "controller-state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    args = script._parser().parse_args(
        ["--action", "run", "--ack", expected_run_ack(plan)]
    )

    def authentic_rebuild(
        _plan_value: dict[str, object],
        replay_state: dict[str, object],
        _checkpoint: dict[str, object],
        *,
        threshold: int,
        exclude_last_history: bool = False,
    ) -> tuple[dict[str, object], dict[str, object]]:
        del exclude_last_history
        replay = replay_state["gate_history"]
        assert isinstance(replay, list)
        index = len(replay)
        row = original[index]
        assert threshold == row["threshold"]
        return deepcopy(row["decision"]), deepcopy(row["evidence"])

    with (
        patch.object(
            script,
            "_build_threshold_decision",
            side_effect=authentic_rebuild,
        ),
        pytest.raises(
            GovernedControllerError,
            match=r"gate_history\[0\].*full artifact reauthentication",
        ),
    ):
        script._run(args, plan, state_path)
