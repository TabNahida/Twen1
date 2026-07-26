from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).parents[1] / "scripts" / "sample_dense_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("sample_dense_checkpoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sampling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sampling)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_completed_evaluation_can_supply_authenticated_preflight(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "resolved.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    checkpoint = tmp_path / "run/step-000000000001-milestone-complete"
    checkpoint.mkdir(parents=True)
    (checkpoint / "COMPLETE").write_text("f" * 64 + "\n", encoding="ascii")
    checkpoint_state = {
        "global_step": 1,
        "committed_tokens": 100,
        "kind": "milestone",
        "tag": "complete",
    }
    source_sha = "a" * 64
    plan = {
        "config_path": str(config_path.resolve()),
        "config_fingerprint": "b" * 64,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_complete_sha256": _sha(checkpoint / "COMPLETE"),
        "checkpoint_state": checkpoint_state,
        "checkpoint_inference_lineage": {
            "archived_config_sha256": _sha(config_path),
            "current_source_tree_sha256": source_sha,
            "calibration_artifacts": {"layer_map": "c" * 64},
        },
        "plan_fingerprint": "d" * 64,
    }
    evaluation = tmp_path / "evaluation"
    _json(evaluation / "PLAN.json", plan)
    manifest = {
        "kind": "twen_nll_evaluation",
        "plan_sha256": _sha(evaluation / "PLAN.json"),
        "plan_fingerprint": plan["plan_fingerprint"],
        "checkpoint_state": checkpoint_state,
    }
    _json(evaluation / "manifest.json", manifest)
    (evaluation / "COMPLETE").write_text(
        _sha(evaluation / "manifest.json") + "\n",
        encoding="ascii",
    )

    class FakeCheckpointManager:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def resolve(self, _path: str) -> Path:
            return checkpoint

        def inspect(self, _path: Path) -> dict[str, str]:
            return {"data_fingerprint": "e" * 64}

    monkeypatch.setattr(sampling, "CheckpointManager", FakeCheckpointManager)
    monkeypatch.setattr(sampling, "twen_source_tree_sha256", lambda: source_sha)
    config = SimpleNamespace(
        checkpoint=SimpleNamespace(output_dir=str(checkpoint.parent)),
        data=SimpleNamespace(
            micro_batch_size=1,
            max_sequence_length=4096,
            global_batch_tokens=262144,
        ),
    )
    report = sampling._report_from_completed_evaluation(
        evaluation_dir=str(evaluation),
        config_path=str(config_path),
        checkpoint_path=str(checkpoint),
        config=config,
    )

    assert report.config_fingerprint == "b" * 64
    assert report.data_fingerprint == "e" * 64
    assert report.source_tree_sha256 == source_sha
    assert report.batch.gradient_accumulation_steps == 64
    assert dict(report.calibration_fingerprints) == {"layer_map": "c" * 64}


def test_completed_evaluation_rejects_source_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "resolved.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    checkpoint = tmp_path / "run/step"
    checkpoint.mkdir(parents=True)
    (checkpoint / "COMPLETE").write_text("f" * 64 + "\n", encoding="ascii")
    state = {"global_step": 1}
    evaluation = tmp_path / "evaluation"
    plan = {
        "config_path": str(config_path.resolve()),
        "config_fingerprint": "b" * 64,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_complete_sha256": _sha(checkpoint / "COMPLETE"),
        "checkpoint_state": state,
        "checkpoint_inference_lineage": {
            "archived_config_sha256": _sha(config_path),
            "current_source_tree_sha256": "a" * 64,
            "calibration_artifacts": {"layer_map": "c" * 64},
        },
        "plan_fingerprint": "d" * 64,
    }
    _json(evaluation / "PLAN.json", plan)
    _json(
        evaluation / "manifest.json",
        {
            "kind": "twen_nll_evaluation",
            "plan_sha256": _sha(evaluation / "PLAN.json"),
            "plan_fingerprint": plan["plan_fingerprint"],
            "checkpoint_state": state,
        },
    )
    (evaluation / "COMPLETE").write_text(
        _sha(evaluation / "manifest.json") + "\n",
        encoding="ascii",
    )

    class FakeCheckpointManager:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def resolve(self, _path: str) -> Path:
            return checkpoint

        def inspect(self, _path: Path) -> dict[str, str]:
            return {"data_fingerprint": "e" * 64}

    monkeypatch.setattr(sampling, "CheckpointManager", FakeCheckpointManager)
    monkeypatch.setattr(sampling, "twen_source_tree_sha256", lambda: "0" * 64)
    config = SimpleNamespace(
        checkpoint=SimpleNamespace(output_dir=str(checkpoint.parent)),
        data=SimpleNamespace(
            micro_batch_size=1,
            max_sequence_length=4096,
            global_batch_tokens=262144,
        ),
    )

    try:
        sampling._report_from_completed_evaluation(
            evaluation_dir=str(evaluation),
            config_path=str(config_path),
            checkpoint_path=str(checkpoint),
            config=config,
        )
    except ValueError as error:
        assert "source changed after evaluation" in str(error)
    else:
        raise AssertionError("source drift must be rejected")
