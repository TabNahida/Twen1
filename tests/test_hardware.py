from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import twen.hardware as hardware
from twen.hardware import (
    CPUInfo,
    estimate_static_training_memory,
    inspect_hardware,
)


def _manifest(root, size: int) -> None:
    root.mkdir()
    (root / "download-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": [
                    {"filename": "config.json", "expected_size": 123},
                    {"filename": "model-00001-of-00001.safetensors", "expected_size": size},
                    {"filename": "model.safetensors.index.json", "expected_size": 456},
                ],
            }
        ),
        encoding="utf-8",
    )


def _config(tmp_path, *, stage: str = "dense-oracle", active_layers=None, hidden=0.1):
    backbone = tmp_path / "backbone"
    teacher = tmp_path / "teacher"
    _manifest(backbone, 2_000_000_000)
    _manifest(teacher, 18_000_000_000)
    architecture = SimpleNamespace(
        student_hidden_size=1024,
        student_intermediate_size=3584,
        student_layers=24,
        donor_hidden_size=4096,
        donor_intermediate_size=12288,
        donor_layers=32,
        num_experts=8,
        expert_intermediate_size=1536,
        lora_rank=16,
        active_layers=lambda: tuple(range(24)) if active_layers is None else tuple(active_layers),
    )
    return SimpleNamespace(
        stage=stage,
        architecture=architecture,
        runtime=SimpleNamespace(bf16=True, sharding="fsdp2"),
        losses=SimpleNamespace(hidden_alignment=hidden),
        sources=SimpleNamespace(
            backbone=SimpleNamespace(local_path=str(backbone)),
            teacher=SimpleNamespace(local_path=str(teacher)),
        ),
    )


def _fake_torch(*, memory: int = 32 * 1024**3):
    properties = SimpleNamespace(
        name="Mock Accelerator",
        total_memory=memory,
        major=8,
        minor=9,
        multi_processor_count=128,
    )
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda index: properties,
        get_device_capability=lambda index: (8, 9),
        get_device_name=lambda index: properties.name,
    )
    return SimpleNamespace(
        __version__="2.mock",
        version=SimpleNamespace(cuda="12.8", hip=None),
        cuda=cuda,
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 9100)),
    )


def _mock_cpu() -> CPUInfo:
    return CPUInfo("Mock CPU", "x86_64", 8, 16, 16, 64 * 1024**3, 48 * 1024**3)


def _components(estimate):
    return {component.name: component for component in estimate.components}


def test_dense_static_parameter_formula_and_fsdp_sharding(tmp_path) -> None:
    config = _config(tmp_path)
    estimate = estimate_static_training_memory(config, world_size=4)
    components = _components(estimate)

    expected_donor = 24 * 3 * 4096 * 12288
    expected_trainable = 24 * (2 * 1024 * 4096 + 1)
    assert components["frozen_backbone"].bytes == 2_000_000_000
    assert components["frozen_dense_donor_ffn"].parameter_count == expected_donor
    assert components["frozen_dense_donor_ffn"].bytes == expected_donor * 2
    assert components["trainable_parameters"].parameter_count == expected_trainable
    assert components["trainable_parameters"].bytes == expected_trainable * 4
    assert components["trainable_gradients"].bytes == expected_trainable * 4
    assert components["adam_first_and_second_moments"].bytes == expected_trainable * 8
    assert components["frozen_hidden_alignment_teacher"].bytes == 18_000_000_000
    assert (
        estimate.estimated_per_device_static_bytes
        == (estimate.aggregate_known_static_bytes + 3) // 4
    )
    assert estimate.is_runtime_lower_bound
    assert any("activations" in excluded for excluded in estimate.excludes)
    assert any("NCCL" in excluded for excluded in estimate.excludes)


def test_sparse_static_parameter_formula_does_not_include_teacher(tmp_path) -> None:
    config = _config(tmp_path, stage="sparse")
    estimate = estimate_static_training_memory(config)
    components = _components(estimate)

    expected_folded = 24 * 3 * 8 * 1024 * 1536
    expected_lora = 24 * 3 * 8 * 16 * (1024 + 1536)
    expected_router = 24 * 8 * 1024
    expected_trainable = expected_lora + expected_router + 24
    assert components["frozen_folded_experts"].parameter_count == expected_folded
    assert components["trainable_parameters"].parameter_count == expected_trainable
    assert "frozen_hidden_alignment_teacher" not in components


def test_enabled_native_mtp_is_counted_as_frozen_source_state(tmp_path) -> None:
    config = _config(tmp_path)
    config.losses.mtp = 0.2
    with patch.object(
        hardware,
        "_mtp_checkpoint_parameter_count",
        return_value=(20_452_864, None),
    ):
        estimate = estimate_static_training_memory(config)

    mtp = _components(estimate)["frozen_native_mtp"]
    assert mtp.parameter_count == 20_452_864
    assert mtp.bytes == 20_452_864 * 2
    assert mtp.dtype_or_state == "BF16 parameters"


def test_single_32gb_full_dense_reuses_teacher_storage(tmp_path) -> None:
    config = _config(tmp_path)
    with (
        patch.object(hardware.importlib, "import_module", return_value=_fake_torch()),
        patch.object(hardware, "_inspect_cpu", return_value=_mock_cpu()),
    ):
        report = inspect_hardware(config)

    assert report.torch.cuda_available
    assert report.gpus[0].compute_capability == "8.9"
    assert report.gpus[0].bf16_supported is True
    warnings = {warning.code: warning for warning in report.warnings}
    assert warnings["single_32gb_dense_shared_teacher"].severity == "info"
    components = _components(report.static_memory)
    assert components["frozen_dense_donor_ffn"].bytes == 0
    assert "shared" in components["frozen_dense_donor_ffn"].dtype_or_state


def test_single_gpu_teacher_cpu_offload_separates_gpu_aliases_and_cpu_shadow(
    tmp_path,
) -> None:
    config = _config(tmp_path)
    config.runtime.teacher_cpu_offload = True
    estimate = estimate_static_training_memory(config)
    components = _components(estimate)

    mapped_bytes = 24 * 3 * 4096 * 12288 * 2
    assert components["frozen_dense_donor_ffn"].bytes == mapped_bytes
    assert "GPU-resident alias" in components["frozen_dense_donor_ffn"].dtype_or_state
    cpu_shadow = components["frozen_hidden_alignment_teacher_cpu_shadow"]
    assert cpu_shadow.bytes == 18_000_000_000 - mapped_bytes
    assert "CPU shadow" in cpu_shadow.dtype_or_state
    assert estimate.estimated_per_device_static_bytes == (
        estimate.aggregate_known_static_bytes - cpu_shadow.bytes
    )

    with (
        patch.object(hardware.importlib, "import_module", return_value=_fake_torch()),
        patch.object(hardware, "_inspect_cpu", return_value=_mock_cpu()),
    ):
        report = inspect_hardware(config)
    assert "single_32gb_dense_teacher_cpu_offload" in {
        warning.code for warning in report.warnings
    }


def test_active_layer_poc_gets_distinct_warning_and_smaller_state(tmp_path) -> None:
    config = _config(tmp_path, active_layers=[5, 11, 17, 23])
    with (
        patch.object(hardware.importlib, "import_module", return_value=_fake_torch()),
        patch.object(hardware, "_inspect_cpu", return_value=_mock_cpu()),
    ):
        report = inspect_hardware(config)

    warning_codes = {warning.code for warning in report.warnings}
    assert "dense_active_layer_poc" in warning_codes
    assert "single_32gb_full_dense_hidden_alignment" not in warning_codes
    components = _components(report.static_memory)
    assert components["frozen_dense_donor_ffn"].parameter_count == 4 * 3 * 4096 * 12288
    assert components["trainable_parameters"].parameter_count == 4 * (2 * 1024 * 4096 + 1)


def test_torch_import_is_lazy_and_cpu_failure_is_reported_without_cuda(tmp_path) -> None:
    config = _config(tmp_path, hidden=0.0)
    with (
        patch.object(
            hardware.importlib,
            "import_module",
            side_effect=ImportError("mocked missing torch"),
        ) as import_module,
        patch.object(hardware, "_inspect_cpu", return_value=_mock_cpu()),
    ):
        report = inspect_hardware(config)

    import_module.assert_called_once_with("torch")
    assert report.torch.installed is False
    assert report.gpus == ()
    assert {warning.code for warning in report.warnings} == {"torch_unavailable"}


def test_allocator_environment_is_reported(monkeypatch) -> None:
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "backend:native,expandable_segments:True")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    fake = SimpleNamespace(
        __version__="2.mock",
        version=SimpleNamespace(cuda=None, hip=None),
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: None)),
    )
    with (
        patch.object(hardware.importlib, "import_module", return_value=fake),
        patch.object(hardware, "_inspect_cpu", return_value=_mock_cpu()),
    ):
        report = inspect_hardware()

    assert report.allocator.effective_alloc_conf == "backend:native,expandable_segments:True"
    assert report.allocator.expandable_segments is True
    assert report.allocator.cuda_visible_devices == "2,3"


def test_missing_qwen_fast_path_is_reported_for_training_config(tmp_path) -> None:
    config = _config(tmp_path, active_layers=[11], hidden=0.0)
    with (
        patch.object(hardware.importlib, "import_module", return_value=_fake_torch()),
        patch.object(hardware, "_inspect_cpu", return_value=_mock_cpu()),
        patch.object(hardware, "version", side_effect=PackageNotFoundError),
    ):
        report = inspect_hardware(config)

    assert report.kernels.qwen35_fast_path_ready is False
    assert "qwen35_fast_path_missing" in {warning.code for warning in report.warnings}


@pytest.mark.parametrize("world_size", [0, -1, True])
def test_world_size_must_be_positive(tmp_path, world_size) -> None:
    with pytest.raises(ValueError, match="world_size"):
        estimate_static_training_memory(_config(tmp_path), world_size=world_size)
