from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from twen.config import dump_resolved_config, load_train_config


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "prepare_v4_training_ab.py"
    spec = importlib.util.spec_from_file_location("prepare_v4_training_ab", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


planner = _load_script()


def _base_config(tmp_path: Path) -> Path:
    root = Path(__file__).parents[1]
    config = copy.deepcopy(load_train_config(root / "configs/base/dense-v3-500m.yaml"))
    config.run_id = "base-dense-v4-test"
    config.data.mode = "prepared-text"
    config.data.teacher_kd_manifest_path = None
    config.data.teacher_kd_manifest_sha256 = None
    config.data.quality_cooldown_manifest_path = None
    config.data.quality_cooldown_manifest_sha256 = None
    config.data.quality_cooldown_teacher_kd_manifest_path = None
    config.data.quality_cooldown_teacher_kd_manifest_sha256 = None
    config.data.quality_cooldown_start_tokens = None
    config.losses.teacher_kd = 0.0
    config.losses.anchor_kl = 0.0
    config.losses.hidden_alignment = 0.0
    config.losses.hidden_alignment_batch_fraction = 0.0
    config.optimizer.adapter_optimizer = "muon"
    config.optimizer.lr_schedule = "cosine"
    config.optimizer.decay_tokens = None
    config.runtime.teacher_cpu_offload = False
    config.runtime.activation_checkpointing_on_alignment_only = False
    config.runtime.hidden_alignment_activation_checkpoint_layer_count = None
    config.runtime.hidden_alignment_dense_transfer_checkpoint_layer_count = None
    config.checkpoint.output_dir = str(tmp_path / "unused")
    config.validate()
    path = tmp_path / "base.yaml"
    dump_resolved_config(config, path)
    return path


def test_planner_writes_isolated_configs_and_optimizer_crossing_profiles(
    tmp_path: Path, capsys: object
) -> None:
    base = _base_config(tmp_path)
    fork = tmp_path / "fork"
    fork.mkdir()
    (fork / "COMPLETE").write_text("fixture\n", encoding="utf-8")
    output = tmp_path / "configs"
    runs = tmp_path / "runs"

    exit_code = planner.main(
        [
            "--base-config",
            str(base),
            "--output-dir",
            str(output),
            "--run-root",
            str(runs),
            "--fork-from",
            str(fork),
        ]
    )

    assert exit_code == 0
    plan = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert plan["training_started"] is False
    assert plan["cuda_initialized"] is False
    assert [case["micro_batch_size"] for case in plan["cases"]] == [1, 2, 4]
    assert [
        case["gradient_accumulation_steps"] for case in plan["cases"]
    ] == [64, 32, 16]
    assert [case["profile"]["wait_steps"] for case in plan["cases"]] == [62, 30, 14]
    assert all(case["profile"]["captures_optimizer_step"] for case in plan["cases"])
    assert (output / "plan.json").is_file()

    for case in plan["cases"]:
        config = load_train_config(case["config"])
        assert config.data.mode == "prepared-text"
        assert config.optimizer.adapter_optimizer == "muon"
        assert config.optimizer.max_tokens == 4_000_000
        assert config.optimizer.warmup_tokens == 262_144
        assert config.checkpoint.output_dir == case["run_dir"]
        assert config.runtime.profile is True
        assert case["command"][-1] == str(fork)


def test_planner_refuses_kd_base_and_nonempty_output(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    kd_config = root / "configs/base/dense-v3-500m.yaml"
    fork = tmp_path / "fork"
    fork.mkdir()
    (fork / "COMPLETE").write_text("fixture\n", encoding="utf-8")

    with pytest.raises(planner.PlanError, match="prepared-text"):
        planner.build_plan(
            kd_config,
            tmp_path / "first",
            tmp_path / "runs",
            fork,
            micro_batches=(1,),
            performance_tokens=4_000_000,
            warmup_tokens=262_144,
            profile=True,
        )

    output = tmp_path / "occupied"
    output.mkdir()
    (output / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(planner.PlanError, match="not empty"):
        planner.build_plan(
            _base_config(tmp_path),
            output,
            tmp_path / "runs",
            fork,
            micro_batches=(1,),
            performance_tokens=4_000_000,
            warmup_tokens=262_144,
            profile=True,
        )


@pytest.mark.parametrize("accumulation", [64, 32, 16])
def test_profile_window_is_recording_after_final_microbatch(accumulation: int) -> None:
    torch = pytest.importorskip("torch")
    schedule = torch.profiler.schedule(
        wait=accumulation - 2,
        warmup=1,
        active=4,
        repeat=1,
    )

    # The engine calls profiler.step() after every microbatch. Optimizer work
    # follows the final call, so it executes under schedule(accumulation).
    assert "RECORD" in str(schedule(accumulation))
