from __future__ import annotations

import importlib.util
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors import safe_open

from twen.training.teacher_offload import TeacherResidencyTransition


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "benchmark_full_dense_graph.py"
    spec = importlib.util.spec_from_file_location("benchmark_full_dense_graph", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = _load_script()


def _write_text_config(root: Path, *, student: bool) -> None:
    root.mkdir()
    shape = benchmark.PRODUCTION_SHAPE
    value = {
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": shape["student_hidden_size" if student else "donor_hidden_size"],
            "intermediate_size": shape[
                "student_intermediate_size" if student else "donor_intermediate_size"
            ],
            "num_hidden_layers": shape["student_layers" if student else "donor_layers"],
            "vocab_size": shape["vocabulary_size"],
            "tie_word_embeddings": student,
            "mtp_num_hidden_layers": 1 if student else None,
            "mtp_use_dedicated_embeddings": False if student else None,
        }
    }
    (root / "config.json").write_text(json.dumps(value), encoding="utf-8")


def test_dry_run_reports_strict_production_graph_without_cuda(
    tmp_path: Path, capsys: object
) -> None:
    backbone = tmp_path / "backbone"
    teacher = tmp_path / "teacher"
    _write_text_config(backbone, student=True)
    _write_text_config(teacher, student=False)

    exit_code = benchmark.main(
        [
            "--dry-run",
            "--backbone",
            str(backbone),
            "--teacher",
            str(teacher),
            "--warmup",
            "0",
            "--repeats",
            "1",
        ]
    )

    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert result["ok"] is True
    assert result["production_shape"] is True
    assert result["production_acceptance"] is True
    assert result["batch"] == {
        "batch_size": 1,
        "sequence_length": 4096,
        "logical_tokens": 4096,
    }
    assert result["graph"]["student_layers_active"] == 24
    assert result["graph"]["parameter_update"] is False
    assert result["no_optimizer_created"] is True
    assert result["no_optimizer_steps"] is True
    assert result["optimizer_state_reserve"]["requested_gib"] == 1.5
    assert result["optimizer_state_reserve"]["touched"] is False
    assert result["runtime"]["fla_backend"] == "triton"
    assert result["runtime"]["fla_tilelang_env"] == "0"
    assert result["runtime"]["compile_streaming_loss"] is True
    assert result["runtime"]["activation_checkpointing"] is True
    assert result["runtime"]["activation_checkpoint_layer_count"] == 24
    assert result["runtime"]["activation_checkpoint_layer_indices"] == list(range(24))
    assert result["runtime"]["dense_transfer_execution"] == "expanded"
    assert result["runtime"]["dense_transfer_checkpoint_layer_count_requested"] == 0
    assert result["runtime"]["dense_transfer_checkpoint_layer_count_effective"] == 0
    assert result["runtime"]["dense_transfer_checkpoint_layer_indices"] == []
    assert result["runtime"]["dense_transfer_outer_inner_disjoint"] is True
    assert result["experimental_execution"]["mode"] == "expanded"
    assert result["experimental_execution"]["production_enabled"] is True
    assert (
        result["experimental_execution"]["selected_mode_numerical_status"]
        == "admitted_production_reference"
    )
    numerical = result["experimental_execution"]["numerical_admission"]
    assert numerical["production_reference_mode"] == "expanded"
    assert numerical["expanded_selective_checkpoint"]["status"] == "pass"
    assert numerical["differentiable_folded"]["status"] == "fail_experimental_only"
    assert result["graph"]["online_hidden_alignment"] is True
    assert result["graph"]["teacher_hidden_forward"] is True
    assert result["graph"]["native_mtp_forward"] is False
    assert result["loss_weights"]["mtp"] == 0.0
    assert result["mtp"]["enabled"] is False
    assert result["teacher_cpu_offload"]["enabled"] is False
    assert "profiling" not in result
    assert "sweep" not in result
    assert "cases" not in result


def test_optional_graph_contract_records_ordinary_batch_selective_checkpoint_and_mtp(
    tmp_path: Path, capsys: object
) -> None:
    backbone = tmp_path / "backbone"
    teacher = tmp_path / "teacher"
    _write_text_config(backbone, student=True)
    _write_text_config(teacher, student=False)

    exit_code = benchmark.main(
        [
            "--dry-run",
            "--backbone",
            str(backbone),
            "--teacher",
            str(teacher),
            "--no-hidden-alignment",
            "--activation-checkpoint-layer-count",
            "6",
            "--dense-transfer-checkpoint-layer-count",
            "4",
            "--mtp-loss-weight",
            "0.25",
            "--warmup",
            "0",
            "--repeats",
            "1",
        ]
    )

    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert result["graph"]["online_hidden_alignment"] is False
    assert result["graph"]["teacher_hidden_forward"] is False
    assert result["graph"]["teacher_donor_resident"] is True
    assert result["graph"]["native_mtp_forward"] is True
    assert result["graph"]["native_mtp_vocab_loss"] is True
    assert result["graph"]["native_mtp_parameter_update"] is False
    assert result["runtime"]["activation_checkpointing"] is True
    assert result["runtime"]["activation_checkpoint_layer_count"] == 6
    assert result["runtime"]["activation_checkpoint_layer_indices"] == [
        0,
        5,
        9,
        14,
        18,
        23,
    ]
    assert result["runtime"]["dense_transfer_checkpoint_layer_count_requested"] == 4
    assert result["runtime"]["dense_transfer_checkpoint_layer_count_effective"] == 4
    assert result["runtime"]["dense_transfer_checkpoint_layer_indices"] == [1, 8, 15, 22]
    assert result["runtime"]["dense_transfer_outer_inner_disjoint"] is True
    assert result["loss_weights"]["mtp"] == 0.25
    assert result["mtp"] == {
        "enabled": True,
        "frozen": True,
        "loss_weight": 0.25,
        "parameter_update": False,
        "source_role": "backbone",
    }


def test_dry_run_teacher_cpu_offload_contract_supports_no_hidden_ac0_and_mtp(
    tmp_path: Path, capsys: object
) -> None:
    backbone = tmp_path / "backbone"
    teacher = tmp_path / "teacher"
    _write_text_config(backbone, student=True)
    _write_text_config(teacher, student=False)

    exit_code = benchmark.main(
        [
            "--dry-run",
            "--backbone",
            str(backbone),
            "--teacher",
            str(teacher),
            "--teacher-cpu-offload",
            "--no-hidden-alignment",
            "--activation-checkpoint-layer-count",
            "0",
            "--mtp-loss-weight",
            "0.1",
        ]
    )

    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert result["runtime"]["teacher_cpu_offload"] is True
    assert result["runtime"]["activation_checkpoint_layer_count"] == 0
    assert result["graph"]["teacher_cpu_offload"] is True
    assert result["graph"]["teacher_hidden_forward"] is False
    assert result["graph"]["native_mtp_forward"] is True
    assert result["teacher_cpu_offload"] == {
        "enabled": True,
        "scope": "single_gpu_dense_donor_alias_split",
        "shared_donor_projections_resident_on_cuda": True,
        "stage_policy": "never",
        "teacher_load_device": "cpu",
        "teacher_only_residency_between_iterations": "cpu",
        "transitions_in_gpu_iteration_timing": False,
    }


def test_dry_run_folded_execution_is_permanently_experimental_only(
    tmp_path: Path, capsys: object
) -> None:
    backbone = tmp_path / "backbone"
    teacher = tmp_path / "teacher"
    _write_text_config(backbone, student=True)
    _write_text_config(teacher, student=False)

    exit_code = benchmark.main(
        [
            "--dry-run",
            "--backbone",
            str(backbone),
            "--teacher",
            str(teacher),
            "--dense-transfer-execution",
            "differentiable_folded",
        ]
    )

    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert result["ok"] is True
    assert result["production_shape"] is True
    assert result["production_acceptance"] is False
    assert result["runtime"]["dense_transfer_execution"] == "differentiable_folded"
    execution = result["experimental_execution"]
    assert execution["mode"] == "differentiable_folded"
    assert execution["production_enabled"] is False
    assert execution["selected_mode_numerical_status"] == "failed_experimental_only"
    assert execution["numerical_admission"]["production_reference_mode"] == "expanded"
    assert (
        execution["numerical_admission"]["differentiable_folded"]["evidence"]
        == "artifacts/audits/differentiable-fold-numerical-admission/"
        "FULL_GRAPH_V1_REAL_KD_ACCUMULATION_REPORT.md"
    )


def test_teacher_cpu_offload_requires_a_cuda_device_even_for_dry_run() -> None:
    args = benchmark._parser().parse_args(["--dry-run", "--teacher-cpu-offload", "--device", "cpu"])

    with pytest.raises(ValueError, match="requires exactly one CUDA device"):
        benchmark._validate_args(args)


def test_no_activation_checkpointing_resolves_to_zero_layers() -> None:
    args = benchmark._parser().parse_args(["--no-activation-checkpointing"])
    benchmark._validate_args(args)

    assert benchmark._activation_checkpoint_layer_indices(args) == ()
    contract = benchmark._base_contract(args, {"ok": True})
    assert contract["runtime"]["activation_checkpointing"] is False
    assert contract["runtime"]["activation_checkpointing_requested"] is False
    assert contract["runtime"]["activation_checkpoint_layer_count"] == 0
    assert contract["runtime"]["activation_checkpoint_layer_indices"] == []


def test_dry_run_checkpoint_sweep_contract_has_ordered_cases_and_one_model_load(
    tmp_path: Path, capsys: object
) -> None:
    backbone = tmp_path / "backbone"
    teacher = tmp_path / "teacher"
    _write_text_config(backbone, student=True)
    _write_text_config(teacher, student=False)

    exit_code = benchmark.main(
        [
            "--dry-run",
            "--backbone",
            str(backbone),
            "--teacher",
            str(teacher),
            "--no-hidden-alignment",
            "--activation-checkpoint-layer-counts",
            "16,18,20,24",
            "--dense-transfer-checkpoint-layer-count",
            "4",
            "--warmup",
            "1",
            "--repeats",
            "2",
        ]
    )

    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert result["runtime"]["activation_checkpoint_layer_count"] is None
    assert result["runtime"]["activation_checkpoint_layer_indices"] is None
    assert result["runtime"]["activation_checkpoint_layer_counts"] == [16, 18, 20, 24]
    assert result["runtime"]["dense_transfer_checkpoint_layer_count_requested"] == 4
    assert result["runtime"]["dense_transfer_checkpoint_layer_count_effective"] is None
    assert result["runtime"]["dense_transfer_checkpoint_layer_indices"] is None
    assert result["runtime"]["dense_transfer_outer_inner_disjoint"] is None
    assert [case["activation_checkpoint_layer_count"] for case in result["cases"]] == [
        16,
        18,
        20,
        24,
    ]
    assert all(
        len(case["activation_checkpoint_layer_indices"])
        == case["activation_checkpoint_layer_count"]
        for case in result["cases"]
    )
    assert result["cases"][-1]["activation_checkpoint_layer_indices"] == list(range(24))
    assert [
        case["dense_transfer_checkpoint_layer_count_effective"] for case in result["cases"]
    ] == [4, 4, 4, 0]
    assert result["cases"][-1]["dense_transfer_checkpoint_layer_indices"] == []
    for case in result["cases"]:
        assert case["dense_transfer_execution"] == "expanded"
        assert case["dense_transfer_checkpoint_layer_count_requested"] == 4
        assert case["dense_transfer_outer_inner_disjoint"] is True
        assert not set(case["activation_checkpoint_layer_indices"]).intersection(
            case["dense_transfer_checkpoint_layer_indices"]
        )
    assert result["sweep"] == {
        "axis": "activation_checkpoint_layer_count",
        "case_count": 4,
        "enabled": True,
        "independent_warmup_and_repeats": True,
        "model_loads_per_cuda_run": 1,
        "shared_optimizer_state_reserve_across_cases": True,
        "shared_teacher_donor_across_cases": True,
    }


def test_single_and_sweep_checkpoint_count_options_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        benchmark._parser().parse_args(
            [
                "--activation-checkpoint-layer-count",
                "16",
                "--activation-checkpoint-layer-counts",
                "18,20",
            ]
        )


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--activation-checkpoint-layer-count", "-1"], r"must be in \[0, 24\]"),
        (["--activation-checkpoint-layer-count", "25"], r"must be in \[0, 24\]"),
        (["--dense-transfer-checkpoint-layer-count", "-1"], r"must be in \[0, 24\]"),
        (["--dense-transfer-checkpoint-layer-count", "25"], r"must be in \[0, 24\]"),
        (
            ["--no-activation-checkpointing", "--activation-checkpoint-layer-count", "1"],
            "--no-activation-checkpointing requires",
        ),
        (
            [
                "--mtp-loss-weight",
                "0.1",
                "--sequence-length",
                "2",
                "--allow-non-production-shape",
            ],
            "requires sequence-length>=3",
        ),
        (
            ["--activation-checkpoint-layer-counts", "16,,20"],
            "must contain comma-separated integers",
        ),
        (
            ["--activation-checkpoint-layer-counts", "16,16"],
            "must not contain duplicates",
        ),
        (
            ["--activation-checkpoint-layer-counts", "16,25"],
            r"must be in \[0, 24\]",
        ),
        (
            ["--no-activation-checkpointing", "--activation-checkpoint-layer-counts", "0,1"],
            "--no-activation-checkpointing requires",
        ),
        (
            [
                "--activation-checkpoint-layer-counts",
                "20,24",
                "--gpu-telemetry-output",
                "power.csv",
            ],
            "gpu telemetry requires one activation-checkpoint case",
        ),
        (
            ["--dry-run", "--gpu-telemetry-output", "power.csv"],
            "gpu telemetry requires a CUDA benchmark",
        ),
        (
            ["--gpu-telemetry-interval-ms", "0"],
            "gpu-telemetry-interval-ms must be a positive integer",
        ),
        (
            [
                "--activation-checkpoint-layer-counts",
                "16,18",
                "--torch-profile-trace",
                "trace.json",
            ],
            "profiling options do not support activation checkpoint sweeps",
        ),
    ],
)
def test_optional_graph_arguments_are_strictly_validated(argv: list[str], message: str) -> None:
    args = benchmark._parser().parse_args(argv)

    with pytest.raises(ValueError, match=message):
        benchmark._validate_args(args)


def test_even_checkpoint_selection_is_deterministic_and_spans_depth() -> None:
    assert benchmark._evenly_spaced_layer_indices(0, total_layers=24) == ()
    assert benchmark._evenly_spaced_layer_indices(1, total_layers=24) == (12,)
    assert benchmark._evenly_spaced_layer_indices(6, total_layers=24) == (
        0,
        5,
        9,
        14,
        18,
        23,
    )
    assert benchmark._evenly_spaced_layer_indices(24, total_layers=24) == tuple(range(24))


def test_inner_transfer_checkpoint_selection_is_even_capped_and_disjoint() -> None:
    outer = (0, 5, 9, 14, 18, 23)
    assert (
        benchmark._dense_transfer_checkpoint_layer_indices(
            0,
            outer_checkpoint_layer_indices=outer,
        )
        == ()
    )
    assert benchmark._dense_transfer_checkpoint_layer_indices(
        4,
        outer_checkpoint_layer_indices=outer,
    ) == (1, 8, 15, 22)
    assert benchmark._dense_transfer_checkpoint_layer_indices(
        24,
        outer_checkpoint_layer_indices=tuple(range(22)),
    ) == (22, 23)
    assert (
        benchmark._dense_transfer_checkpoint_layer_indices(
            24,
            outer_checkpoint_layer_indices=tuple(range(24)),
        )
        == ()
    )


def test_builder_config_passes_dense_transfer_execution(tmp_path: Path) -> None:
    args = benchmark._parser().parse_args(["--dense-transfer-execution", "differentiable_folded"])
    artifacts = {
        "layer_map": tmp_path / "layer-map.json",
        "channel_map": tmp_path / "channel-map.json",
        "adapters": tmp_path / "adapters.safetensors",
    }

    config = benchmark._builder_config(args, artifacts)

    assert config.runtime.dense_transfer_execution == "differentiable_folded"


def test_dense_transfer_execution_configuration_asserts_exact_module_state() -> None:
    class _FakeTransferModule:
        def __init__(self) -> None:
            self.transfer_mlp = SimpleNamespace(
                execution_mode="expanded",
                checkpoint_token_branch=False,
            )

        def configure_transfer_execution(
            self,
            *,
            execution_mode: str,
            checkpoint_token_branch: bool,
        ) -> None:
            self.transfer_mlp.execution_mode = execution_mode
            self.transfer_mlp.checkpoint_token_branch = bool(checkpoint_token_branch)

    modules = tuple(_FakeTransferModule() for _ in range(24))
    outer = (0, 5, 9, 14, 18, 23)
    inner = (1, 8, 15, 22)

    report = benchmark._configure_dense_transfer_execution(
        modules,
        tuple(range(24)),
        execution_mode="differentiable_folded",
        outer_checkpoint_layer_indices=outer,
        inner_checkpoint_layer_indices=inner,
    )

    assert report == {
        "actual_checkpoint_layer_indices": [1, 8, 15, 22],
        "actual_execution_modes": ["differentiable_folded"],
        "all_modules_match_requested_state": True,
        "module_count": 24,
        "outer_inner_disjoint": True,
    }
    assert [
        index for index, module in enumerate(modules) if module.transfer_mlp.checkpoint_token_branch
    ] == list(inner)
    assert all(module.transfer_mlp.execution_mode == "differentiable_folded" for module in modules)


def test_dense_transfer_execution_configuration_rejects_unobservable_mismatch() -> None:
    class _LyingTransferModule:
        def __init__(self) -> None:
            self.transfer_mlp = SimpleNamespace(
                execution_mode="expanded",
                checkpoint_token_branch=False,
            )

        def configure_transfer_execution(self, **kwargs: object) -> None:
            del kwargs

    modules = tuple(_LyingTransferModule() for _ in range(24))

    with pytest.raises(RuntimeError, match="execution mismatch at layer 0"):
        benchmark._configure_dense_transfer_execution(
            modules,
            tuple(range(24)),
            execution_mode="differentiable_folded",
            outer_checkpoint_layer_indices=(),
            inner_checkpoint_layer_indices=(0,),
        )


def test_gpu_telemetry_sampler_atomically_commits_csv_and_reaps_on_error(
    tmp_path: Path,
) -> None:
    fake_nvidia_smi = tmp_path / "fake-nvidia-smi"
    fake_nvidia_smi.write_text(
        "#!/usr/bin/env python3\n"
        "import time\n"
        "while True:\n"
        "    print('2026/07/19 12:00:00.000, 500, 600, 99, 40, 2700, 25000, 7000, 72', flush=True)\n"
        "    time.sleep(0.01)\n",
        encoding="utf-8",
    )
    fake_nvidia_smi.chmod(0o755)
    output = tmp_path / "case.power.csv"
    args = benchmark._parser().parse_args(
        [
            "--gpu-telemetry-output",
            str(output),
            "--gpu-telemetry-interval-ms",
            "20",
            "--nvidia-smi",
            str(fake_nvidia_smi),
        ]
    )
    report = None
    sampler_pid = None

    with (
        pytest.raises(RuntimeError, match="graph failed after sampler start"),
        benchmark._gpu_telemetry_sampler(args) as active_report,
    ):
        assert active_report is not None
        report = active_report
        sampler_pid = int(active_report["sampler_pid"])
        time.sleep(0.04)
        raise RuntimeError("graph failed after sampler start")

    assert report is not None
    assert report["sample_count"] > 0
    assert report["cleanup"]["reaped"] is True
    assert output.is_file()
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(benchmark.GPU_TELEMETRY_COLUMNS)
    assert len(lines) > 1
    assert not list(tmp_path.glob("*.partial"))
    assert sampler_pid is not None
    with pytest.raises(ProcessLookupError):
        os.kill(sampler_pid, 0)


def test_selective_activation_checkpointing_marks_only_requested_decoder_layers() -> None:
    class _FakeRawModel:
        def __init__(self) -> None:
            self.model = SimpleNamespace(
                layers=[SimpleNamespace(gradient_checkpointing=False) for _ in range(24)]
            )
            self.enable_calls = 0
            self.disable_calls = 0

        def gradient_checkpointing_enable(self) -> None:
            self.enable_calls += 1
            for layer in self.model.layers:
                layer.gradient_checkpointing = True

        def gradient_checkpointing_disable(self) -> None:
            self.disable_calls += 1
            for layer in self.model.layers:
                layer.gradient_checkpointing = False

    raw_model = _FakeRawModel()
    selected = (0, 5, 9, 14, 18, 23)

    benchmark._configure_selective_activation_checkpointing(raw_model, selected)

    assert raw_model.enable_calls == 1
    assert [
        index for index, layer in enumerate(raw_model.model.layers) if layer.gradient_checkpointing
    ] == list(selected)

    benchmark._configure_selective_activation_checkpointing(raw_model, ())
    assert raw_model.disable_calls == 1
    assert not any(layer.gradient_checkpointing for layer in raw_model.model.layers)


def test_sweep_cases_reuse_model_and_independently_prepare_measurements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRawModel:
        def __init__(self) -> None:
            self.model = SimpleNamespace(
                layers=[SimpleNamespace(gradient_checkpointing=False) for _ in range(24)]
            )
            self.enable_calls = 0
            self.disable_calls = 0
            self.zero_grad_calls = 0

        def gradient_checkpointing_enable(self) -> None:
            self.enable_calls += 1
            for layer in self.model.layers:
                layer.gradient_checkpointing = True

        def gradient_checkpointing_disable(self) -> None:
            self.disable_calls += 1
            for layer in self.model.layers:
                layer.gradient_checkpointing = False

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none is True
            self.zero_grad_calls += 1

    class _FakeTransferModule:
        def __init__(self) -> None:
            self.transfer_mlp = SimpleNamespace(
                execution_mode="expanded",
                checkpoint_token_branch=False,
            )

        def configure_transfer_execution(
            self,
            *,
            execution_mode: str,
            checkpoint_token_branch: bool,
        ) -> None:
            self.transfer_mlp.execution_mode = execution_mode
            self.transfer_mlp.checkpoint_token_branch = bool(checkpoint_token_branch)

    cuda_calls = {"empty_cache": 0, "synchronize": 0, "reset_peak": 0}

    def count_call(name: str) -> None:
        cuda_calls[name] += 1

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            empty_cache=lambda: count_call("empty_cache"),
            synchronize=lambda device: count_call("synchronize"),
            reset_peak_memory_stats=lambda device: count_call("reset_peak"),
            memory_allocated=lambda device: 100,
            memory_reserved=lambda device: 120,
            mem_get_info=lambda device: (80, 200),
        )
    )
    iteration_calls = 0

    def fake_execute_iteration(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        nonlocal iteration_calls
        iteration_calls += 1
        return {
            "ok": True,
            "loss_finite": True,
            "gradients": {
                "finite": True,
                "present_tensors": 72,
                "missing_tensors": 0,
                "nonfinite_tensors": 0,
            },
            "timing": {"total_gpu_seconds": float(iteration_calls)},
            "throughput": {"logical_tokens_per_second_gpu": 4096.0},
            "memory": {
                "peak_allocated_bytes": 1000 + iteration_calls,
                "peak_reserved_bytes": 1200 + iteration_calls,
                "free_after_bytes": 70,
            },
        }

    monkeypatch.setattr(benchmark, "_execute_iteration", fake_execute_iteration)
    raw_model = _FakeRawModel()
    transfer_modules = tuple(_FakeTransferModule() for _ in range(24))
    built = SimpleNamespace(
        model=raw_model,
        transfer_modules=transfer_modules,
        student_layer_indices=tuple(range(24)),
    )
    args = benchmark._parser().parse_args(
        [
            "--warmup",
            "1",
            "--repeats",
            "2",
            "--dense-transfer-checkpoint-layer-count",
            "4",
        ]
    )
    cases = []
    for case_number, count in enumerate((16, 18), start=1):
        indices = benchmark._evenly_spaced_layer_indices(count, total_layers=24)
        cases.append(
            benchmark._run_activation_checkpoint_case(
                args,
                case_number=case_number,
                case_count=2,
                checkpoint_layer_count=count,
                checkpoint_layer_indices=indices,
                built=built,
                train_model=object(),
                teacher=object(),
                batch=object(),
                torch=fake_torch,
                device="cuda:0",
            )
        )

    assert iteration_calls == 6
    assert raw_model.enable_calls == 1
    assert raw_model.zero_grad_calls == 4
    assert cuda_calls == {"empty_cache": 4, "synchronize": 4, "reset_peak": 2}
    assert [case["activation_checkpoint_layer_count"] for case in cases] == [16, 18]
    assert [case["dense_transfer_checkpoint_layer_count_effective"] for case in cases] == [
        4,
        4,
    ]
    assert all(case["dense_transfer_outer_inner_disjoint"] is True for case in cases)
    assert all(
        case["dense_transfer_actual_state"]["all_modules_match_requested_state"] is True
        for case in cases
    )
    assert all(case["ok"] is True for case in cases)
    assert [sample["repeat"] for sample in cases[0]["samples"]] == [1, 2]
    assert cases[0]["health"] == {
        "consistent_present_gradient_tensor_count": True,
        "gradients_finite": True,
        "loss_finite": True,
        "maximum_missing_gradient_tensors": 0,
        "maximum_nonfinite_gradient_tensors": 0,
        "measurement_iterations": 2,
        "measurements_ok": True,
        "ok": True,
        "present_gradient_tensor_counts": [72],
        "warmup_iterations": 1,
        "warmup_ok": True,
    }
    active_indices = [
        index for index, layer in enumerate(raw_model.model.layers) if layer.gradient_checkpointing
    ]
    assert active_indices == list(cases[1]["activation_checkpoint_layer_indices"])
    active_inner_indices = [
        index
        for index, module in enumerate(transfer_modules)
        if module.transfer_mlp.checkpoint_token_branch
    ]
    assert active_inner_indices == cases[1]["dense_transfer_checkpoint_layer_indices"]
    assert not set(active_indices).intersection(active_inner_indices)


def test_ordinary_batch_skips_teacher_hidden_states_and_accounts_for_mtp_loss() -> None:
    class _FakeEvent:
        next_tick = 0

        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is True
            self.tick = -1

        def record(self) -> None:
            self.tick = _FakeEvent.next_tick
            _FakeEvent.next_tick += 1

        def elapsed_time(self, other: object) -> float:
            assert isinstance(other, _FakeEvent)
            return float(max(other.tick - self.tick, 1))

        def synchronize(self) -> None:
            return None

    fake_cuda = SimpleNamespace(
        Event=_FakeEvent,
        synchronize=lambda device: None,
        memory_allocated=lambda device: 0,
        memory_reserved=lambda device: 0,
        mem_get_info=lambda device: (90, 100),
        reset_peak_memory_stats=lambda device: None,
        max_memory_allocated=lambda device: 10,
        max_memory_reserved=lambda device: 12,
    )
    fake_torch = SimpleNamespace(
        bfloat16=torch.bfloat16,
        cuda=fake_cuda,
        profiler=SimpleNamespace(record_function=lambda name: nullcontext()),
        autocast=lambda *args, **kwargs: nullcontext(),
        no_grad=torch.no_grad,
        isfinite=torch.isfinite,
    )

    class _TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    class _Transfer:
        def __init__(self) -> None:
            self.states: list[bool] = []

        def set_transfer_enabled(self, enabled: bool) -> None:
            self.states.append(enabled)

    model = _TinyModel()
    transfer = _Transfer()
    student_hidden_requests: list[bool] = []

    def train_model(**kwargs: object) -> dict[str, torch.Tensor | None]:
        if kwargs.get("anchor_only"):
            return {"anchor_hidden_states": model.weight.detach().reshape(1, 1, 1)}
        student_hidden_requests.append(bool(kwargs["output_hidden_states"]))
        value = model.weight
        return {
            "ntp": value,
            "mtp": value * 4,
            "teacher_kd": value * 2,
            "anchor_kl": value * 3,
            "hidden_states": None,
        }

    teacher_calls = 0

    def teacher(**kwargs: object) -> object:
        del kwargs
        nonlocal teacher_calls
        teacher_calls += 1
        raise AssertionError("ordinary batch must not execute the 9B teacher")

    args = benchmark._parser().parse_args(["--no-hidden-alignment", "--mtp-loss-weight", "0.25"])
    batch = SimpleNamespace(
        input_ids=torch.ones((1, 3), dtype=torch.long),
        labels=torch.ones((1, 3), dtype=torch.long),
        attention_mask=torch.ones((1, 3), dtype=torch.bool),
        topk_indices=torch.zeros((1, 3, 1), dtype=torch.long),
        topk_logits=torch.zeros((1, 3, 1)),
        teacher_logsumexp=torch.zeros((1, 3)),
        teacher_tail_logprob=torch.zeros((1, 3)),
        temperature=2.0,
    )

    result = benchmark._execute_iteration(
        args,
        built=SimpleNamespace(
            model=model,
            transfer_modules=(transfer,),
            student_layer_indices=(0,),
        ),
        train_model=train_model,
        teacher=teacher,
        batch=batch,
        torch=fake_torch,
        device="cuda:0",
    )

    assert result["ok"] is True
    assert result["losses"] == pytest.approx(
        {
            "total": 4.3,
            "ntp": 1.0,
            "teacher_kd": 2.0,
            "anchor_kl": 3.0,
            "mtp": 4.0,
        }
    )
    assert "hidden_alignment" not in result["losses"]
    assert student_hidden_requests == [False]
    assert teacher_calls == 0
    assert transfer.states == [False, True]
    assert result["gradients"]["present_tensors"] == 1
    assert result["gradients"]["missing_tensors"] == 0


def test_teacher_cpu_offload_stages_outside_timing_and_restores_on_success_and_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    class _FakeOffload:
        parameter_bytes = 80
        buffer_bytes = 20
        staged_bytes = 100

        def __init__(self) -> None:
            self.is_staged = False
            self.stage_calls = 0
            self.restore_calls = 0

        def stage(self) -> TeacherResidencyTransition:
            assert not self.is_staged
            self.stage_calls += 1
            self.is_staged = True
            return TeacherResidencyTransition(
                operation="stage",
                seconds=0.25,
                parameter_bytes=80,
                buffer_bytes=20,
                transferred_bytes=100,
                released_cuda_bytes=0,
            )

        def restore(self) -> TeacherResidencyTransition:
            assert self.is_staged
            self.restore_calls += 1
            self.is_staged = False
            return TeacherResidencyTransition(
                operation="restore",
                seconds=0.05,
                parameter_bytes=80,
                buffer_bytes=20,
                transferred_bytes=0,
                released_cuda_bytes=100,
            )

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            synchronize=lambda device: None,
            memory_allocated=lambda device: 100,
            memory_reserved=lambda device: 120,
            mem_get_info=lambda device: (80, 200),
            reset_peak_memory_stats=lambda device: None,
            max_memory_allocated=lambda device: 150,
            max_memory_reserved=lambda device: 170,
        ),
        isfinite=torch.isfinite,
    )
    model = _TinyModel()
    built = SimpleNamespace(model=model)
    manager = _FakeOffload()
    timed_calls = 0

    def fake_timed_graph(
        *args: object, **kwargs: object
    ) -> tuple[dict[str, float], dict[str, float]]:
        del args, kwargs
        nonlocal timed_calls
        timed_calls += 1
        assert manager.is_staged is True
        model.weight.grad = torch.ones_like(model.weight)
        return (
            {"total_gpu_seconds": 2.0, "total_wall_seconds": 1.0},
            {"total": 1.0, "ntp": 1.0, "teacher_kd": 0.0, "anchor_kl": 0.0},
        )

    monkeypatch.setattr(benchmark, "_execute_timed_graph", fake_timed_graph)
    args = benchmark._parser().parse_args(["--teacher-cpu-offload"])

    result = benchmark._execute_iteration(
        args,
        built=built,
        train_model=object(),
        teacher=object(),
        teacher_offload=manager,
        batch=object(),
        torch=fake_torch,
        device="cuda:0",
    )

    assert timed_calls == 1
    assert manager.stage_calls == 1
    assert manager.restore_calls == 1
    assert manager.is_staged is False
    assert result["timing"]["total_gpu_seconds"] == 2.0
    assert result["timing"]["total_wall_seconds"] == 1.0
    assert result["timing"]["teacher_cpu_offload_stage_seconds"] == 0.25
    assert result["timing"]["teacher_cpu_offload_restore_seconds"] == 0.05
    assert result["timing"]["total_wall_seconds_including_teacher_transitions"] == 1.3
    assert result["teacher_cpu_offload"]["stage"]["transferred_bytes"] == 100
    assert result["teacher_cpu_offload"]["restore"]["released_cuda_bytes"] == 100
    assert result["teacher_cpu_offload"]["memory_before_gpu_iteration"]["teacher_staged"] is True
    assert result["teacher_cpu_offload"]["memory_after_restore"]["teacher_staged"] is False

    def failing_timed_graph(*args: object, **kwargs: object) -> object:
        del args, kwargs
        assert manager.is_staged is True
        raise RuntimeError("timed graph failed")

    monkeypatch.setattr(benchmark, "_execute_timed_graph", failing_timed_graph)
    with pytest.raises(RuntimeError, match="timed graph failed"):
        benchmark._execute_iteration(
            args,
            built=built,
            train_model=object(),
            teacher=object(),
            teacher_offload=manager,
            batch=object(),
            torch=fake_torch,
            device="cuda:0",
        )
    assert manager.stage_calls == 2
    assert manager.restore_calls == 2
    assert manager.is_staged is False


def test_teacher_cpu_offload_never_stages_for_no_hidden_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    class _NeverStageOffload:
        parameter_bytes = 80
        buffer_bytes = 20
        staged_bytes = 100
        is_staged = False

        def stage(self) -> object:
            raise AssertionError("no-hidden batch must leave teacher-only state on CPU")

        def restore(self) -> object:
            raise AssertionError("an unstaged teacher must not be restored")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            synchronize=lambda device: None,
            memory_allocated=lambda device: 100,
            memory_reserved=lambda device: 120,
            mem_get_info=lambda device: (80, 200),
            reset_peak_memory_stats=lambda device: None,
            max_memory_allocated=lambda device: 150,
            max_memory_reserved=lambda device: 170,
        ),
        isfinite=torch.isfinite,
    )
    model = _TinyModel()
    manager = _NeverStageOffload()

    def fake_timed_graph(
        *args: object, **kwargs: object
    ) -> tuple[dict[str, float], dict[str, float]]:
        del args, kwargs
        assert manager.is_staged is False
        model.weight.grad = torch.ones_like(model.weight)
        return (
            {"total_gpu_seconds": 2.0, "total_wall_seconds": 1.0},
            {"total": 1.0, "ntp": 1.0, "teacher_kd": 0.0, "anchor_kl": 0.0},
        )

    monkeypatch.setattr(benchmark, "_execute_timed_graph", fake_timed_graph)
    args = benchmark._parser().parse_args(["--teacher-cpu-offload", "--no-hidden-alignment"])
    result = benchmark._execute_iteration(
        args,
        built=SimpleNamespace(model=model),
        train_model=object(),
        teacher=object(),
        teacher_offload=manager,
        batch=object(),
        torch=fake_torch,
        device="cuda:0",
    )

    assert result["teacher_cpu_offload"]["staged_for_hidden_alignment"] is False
    assert result["teacher_cpu_offload"]["stage"] is None
    assert result["teacher_cpu_offload"]["restore"] is None
    assert result["timing"]["teacher_cpu_offload_stage_seconds"] == 0.0
    assert result["timing"]["teacher_cpu_offload_restore_seconds"] == 0.0


def test_profiling_options_default_off_and_require_a_cuda_run() -> None:
    args = benchmark._parser().parse_args([])

    assert args.torch_profile_trace is None
    assert args.cuda_profiler_api is False
    assert args.compile_streaming_loss is True
    assert "profiling" not in benchmark._base_contract(args, {"ok": True})

    eager_args = benchmark._parser().parse_args(["--no-compile-streaming-loss"])
    assert eager_args.compile_streaming_loss is False

    dry_profile = benchmark._parser().parse_args(
        ["--dry-run", "--torch-profile-trace", "trace.json"]
    )
    with pytest.raises(ValueError, match="profiling options require a CUDA benchmark"):
        benchmark._validate_args(dry_profile)


def test_larger_batch_is_still_production_shaped_without_override(
    tmp_path: Path, capsys: object
) -> None:
    backbone = tmp_path / "backbone"
    teacher = tmp_path / "teacher"
    _write_text_config(backbone, student=True)
    _write_text_config(teacher, student=False)

    for batch_size in (2, 4):
        exit_code = benchmark.main(
            [
                "--dry-run",
                "--backbone",
                str(backbone),
                "--teacher",
                str(teacher),
                "--batch-size",
                str(batch_size),
                "--warmup",
                "0",
                "--repeats",
                "1",
            ]
        )

        assert exit_code == 0
        result = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
        assert result["production_shape"] is True
        assert result["production_acceptance"] is True
        assert result["batch"] == {
            "batch_size": batch_size,
            "sequence_length": 4096,
            "logical_tokens": batch_size * 4096,
        }


def test_synthetic_batch_and_kd_cache_follow_requested_batch_shape() -> None:
    args = benchmark._parser().parse_args(
        [
            "--batch-size",
            "4",
            "--sequence-length",
            "7",
            "--teacher-top-k",
            "3",
        ]
    )

    batch = benchmark._synthetic_batch(
        args,
        vocabulary_size=97,
        torch=torch,
        device=torch.device("cpu"),
    )

    assert tuple(batch.input_ids.shape) == (4, 7)
    assert tuple(batch.labels.shape) == (4, 7)
    assert tuple(batch.attention_mask.shape) == (4, 7)
    assert tuple(batch.topk_indices.shape) == (4, 7, 3)
    assert tuple(batch.topk_logits.shape) == (4, 7, 3)
    assert tuple(batch.teacher_logsumexp.shape) == (4, 7)
    assert tuple(batch.teacher_tail_logprob.shape) == (4, 7)
    assert torch.equal(batch.labels, batch.input_ids)
    expected_base = (
        batch.input_ids + torch.arange(7, dtype=torch.long).view(1, 7) * 131
    ).remainder(97)
    expected_indices = (expected_base.unsqueeze(-1) + torch.arange(3, dtype=torch.long)).remainder(
        97
    )
    assert torch.equal(batch.topk_indices, expected_indices)


def test_non_positive_batch_size_fails_before_checkpoint_or_cuda_access(
    capsys: object,
) -> None:
    exit_code = benchmark.main(["--dry-run", "--batch-size", "0"])

    assert exit_code == 2
    result = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert result["ok"] is False
    assert result["batch"]["batch_size"] == 0
    assert "batch_size" in result["error"]["message"]


def test_temporary_calibration_is_sequential_partitioned_and_small(tmp_path: Path) -> None:
    artifacts = benchmark._write_temporary_calibration(
        tmp_path,
        student_layers=2,
        student_hidden=3,
        donor_hidden=6,
        donor_intermediate=8,
        experts=2,
        init_std=0.01,
        seed=7,
        torch=torch,
    )

    layer_map = json.loads(artifacts["layer_map"].read_text(encoding="utf-8"))
    channel_map = json.loads(artifacts["channel_map"].read_text(encoding="utf-8"))
    assert layer_map == {"student_to_donor": [0, 1]}
    assert channel_map == {"indices": [[0, 1, 2, 3], [4, 5, 6, 7]]}
    with safe_open(artifacts["adapters"], framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == {
            "layers.0.A",
            "layers.0.B",
            "layers.1.A",
            "layers.1.B",
        }
        assert tuple(handle.get_tensor("layers.0.A").shape) == (6, 3)
        assert tuple(handle.get_tensor("layers.0.B").shape) == (3, 6)
        assert handle.get_tensor("layers.0.A").dtype == torch.bfloat16
        assert 0 < float(handle.get_tensor("layers.0.A").float().std()) < 0.03


class _FakeTensor:
    def __init__(self, size: int) -> None:
        self.size = size
        self.touched = False

    def zero_(self) -> _FakeTensor:
        self.touched = True
        return self

    def numel(self) -> int:
        return self.size

    def element_size(self) -> int:
        return 1


class _FakeTorch:
    uint8 = object()

    def __init__(self) -> None:
        self.tensors: list[_FakeTensor] = []

    def empty(self, shape: tuple[int], *, dtype: object, device: object) -> _FakeTensor:
        del dtype, device
        tensor = _FakeTensor(shape[0])
        self.tensors.append(tensor)
        return tensor


def test_optimizer_state_reserve_is_chunked_touched_raw_storage() -> None:
    fake = _FakeTorch()
    chunks, report = benchmark._allocate_optimizer_state_reserve(
        11,
        chunk_bytes=4,
        torch=fake,
        device="cuda:0",
    )

    assert [tensor.numel() for tensor in chunks] == [4, 4, 3]
    assert all(tensor.touched for tensor in chunks)
    assert report == {
        "allocated_bytes": 11,
        "allocation_count": 3,
        "touched": True,
        "is_optimizer": False,
    }


def test_benchmark_source_has_no_optimizer_construction_or_step() -> None:
    source = Path(benchmark.__file__).read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert "optimizer.step(" not in source


def test_iteration_has_named_profiler_ranges() -> None:
    source = Path(benchmark.__file__).read_text(encoding="utf-8")
    for name in (
        "twen/benchmark/anchor",
        "twen/benchmark/student",
        "twen/benchmark/teacher",
        "twen/benchmark/backward",
    ):
        assert f'record_function("{name}")' in source


def test_cuda_profiler_api_range_starts_before_body_and_always_stops() -> None:
    events: list[str] = []

    class _FakeCudart:
        def cudaProfilerStart(self) -> None:
            events.append("start")

        def cudaProfilerStop(self) -> None:
            events.append("stop")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(cudart=lambda: _FakeCudart()),
    )

    with (
        pytest.raises(RuntimeError, match="profile body failed"),
        benchmark._cuda_profiler_api_range(enabled=True, torch=fake_torch),
    ):
        events.append("body")
        raise RuntimeError("profile body failed")

    assert events == ["start", "body", "stop"]


def test_torch_trace_profiles_one_separate_health_checked_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[object] = []
    trace = tmp_path / "nested" / "full-iteration.json"

    class _FakeAverages(list[SimpleNamespace]):
        def table(self, *, sort_by: str, row_limit: int) -> str:
            events.append(("table", sort_by, row_limit))
            return "Name  Self CUDA  Input Shapes\nstudent  7us  [[1, 4096]]"

    class _FakeProfile:
        def __init__(self, options: dict[str, object]) -> None:
            self.options = options

        def __enter__(self) -> _FakeProfile:
            events.append("profile_enter")
            return self

        def __exit__(self, *exc_info: object) -> None:
            del exc_info
            events.append("profile_exit")

        def export_chrome_trace(self, path: str) -> None:
            events.append(("trace_export", path))
            Path(path).write_text("{}", encoding="utf-8")

        def key_averages(self, *, group_by_input_shape: bool) -> _FakeAverages:
            events.append(("key_averages", group_by_input_shape))
            return _FakeAverages(
                [
                    SimpleNamespace(
                        key="anchor",
                        count=2,
                        cpu_time_total=11.0,
                        device_time_total=3.0,
                        self_device_time_total=1.0,
                        input_shapes=[[1, 4096]],
                    ),
                    SimpleNamespace(
                        key="student",
                        count=1,
                        cpu_time_total=13.0,
                        device_time_total=9.0,
                        self_device_time_total=7.0,
                        input_shapes=[[1, 4096], [248320, 1024]],
                    ),
                ]
            )

    class _FakeProfiler:
        ProfilerActivity = SimpleNamespace(CPU="cpu", CUDA="cuda")

        @staticmethod
        def profile(**options: object) -> _FakeProfile:
            events.append(("profile_options", options))
            return _FakeProfile(options)

    def fake_execute_iteration(*args: object, **kwargs: object) -> dict[str, object]:
        del args
        events.append(("iteration", kwargs["cuda_profiler_api"]))
        return {
            "ok": True,
            "loss_finite": True,
            "gradients": {"finite": True, "missing_tensors": 0},
        }

    monkeypatch.setattr(benchmark, "_execute_iteration", fake_execute_iteration)
    args = benchmark._parser().parse_args(
        [
            "--torch-profile-trace",
            str(trace),
            "--cuda-profiler-api",
        ]
    )

    report = benchmark._run_profiling_iteration(
        args,
        built=object(),
        train_model=object(),
        teacher=object(),
        batch=object(),
        torch=SimpleNamespace(profiler=_FakeProfiler()),
        device=object(),
    )

    assert trace.read_text(encoding="utf-8") == "{}"
    assert events[0][0] == "profile_options"  # type: ignore[index]
    profile_options = events[0][1]  # type: ignore[index]
    assert profile_options["record_shapes"] is True
    assert profile_options["profile_memory"] is True
    assert profile_options["acc_events"] is True
    assert events[1:4] == [
        "profile_enter",
        ("iteration", True),
        "profile_exit",
    ]
    assert events[4] == ("trace_export", str(trace.resolve()))
    assert events[5:] == [
        ("key_averages", True),
        ("table", "self_cuda_time_total", benchmark.PROFILE_SUMMARY_ROW_LIMIT),
    ]
    summary_text = trace.with_suffix(".summary.txt")
    assert "row_limit=100" in summary_text.read_text(encoding="utf-8")
    assert "Input Shapes" in summary_text.read_text(encoding="utf-8")
    summary_json = trace.with_suffix(".summary.json")
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["sort_by"] == "self_cuda_time_total"
    assert summary["row_limit"] >= 100
    assert summary["group_by_input_shape"] is True
    assert [row["name"] for row in summary["events"]] == ["student", "anchor"]
    assert summary["events"][0] == {
        "name": "student",
        "calls": 1,
        "cpu": 13.0,
        "cuda": 9.0,
        "self_cuda": 7.0,
        "input_shapes": [[1, 4096], [248320, 1024]],
    }
    assert report["separate_complete_iteration"] is True
    assert report["included_in_benchmark_samples"] is False
    assert report["torch_chrome_trace"] == str(trace.resolve())
    assert report["torch_profile_summary"] == str(summary_text.resolve())
    assert report["torch_profile_summary_json"] == str(summary_json.resolve())
    assert report["cuda_profiler_api"] is True
    assert report["nsys_capture_range"] == "cudaProfilerApi"
    assert report["iteration_health"]["ok"] is True


def test_sample_summary_uses_worst_memory_and_timing_statistics() -> None:
    samples = [
        {
            "timing": {"total_gpu_seconds": 2.0},
            "throughput": {"logical_tokens_per_second_gpu": 2048.0},
            "memory": {
                "peak_allocated_bytes": 10,
                "peak_reserved_bytes": 12,
                "free_after_bytes": 7,
            },
        },
        {
            "timing": {"total_gpu_seconds": 4.0},
            "throughput": {"logical_tokens_per_second_gpu": 1024.0},
            "memory": {
                "peak_allocated_bytes": 11,
                "peak_reserved_bytes": 13,
                "free_after_bytes": 6,
            },
        },
    ]

    result = benchmark._summarize_samples(samples)

    assert result["timing_seconds"]["total_gpu_seconds"]["mean"] == 3.0
    assert result["memory_worst_case"] == {
        "peak_allocated_bytes": 11,
        "peak_reserved_bytes": 13,
        "minimum_free_after_bytes": 6,
    }
