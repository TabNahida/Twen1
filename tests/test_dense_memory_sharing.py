from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from twen.model_loading import freeze_module
from twen.training.builder import (
    BuildError,
    BuiltModel,
    _shared_donor_mlp_weights,
    build_transfer_model,
)
from twen.training.distributed import DistributedContext
from twen.training.engine import (
    _build_transfer_and_teacher,
    _set_dense_transfer_token_checkpointing,
)


class _DonorMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 6, bias=False)
        self.up_proj = torch.nn.Linear(4, 6, bias=False)
        self.down_proj = torch.nn.Linear(6, 4, bias=False)


class _TeacherLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _DonorMLP()


class _Teacher(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_TeacherLayer()])


class _Student(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Linear(3, 3, bias=False)
        body = torch.nn.Module()
        body.layers = torch.nn.ModuleList([layer])
        self.model = body


def _config() -> SimpleNamespace:
    architecture = SimpleNamespace(
        student_hidden_size=3,
        donor_hidden_size=4,
        student_layers=1,
        num_experts=2,
        expert_intermediate_size=3,
        expert_initialization="donor",
        active_layers=lambda: (0,),
        layer_map_path="unused-layer-map.json",
        channel_map_path="unused-channel-map.json",
        adapter_init_path="unused-adapters.safetensors",
    )
    optimizer = SimpleNamespace(
        adapter_lr=2e-4,
        router_lr=1e-3,
        lora_lr=2e-4,
        scale_lr=1e-3,
        weight_decay=0.01,
    )
    return SimpleNamespace(
        stage="dense-oracle",
        losses=SimpleNamespace(mtp=0.0),
        architecture=architecture,
        optimizer=optimizer,
        sources=SimpleNamespace(
            backbone=SimpleNamespace(local_path="unused-backbone"),
            donor=SimpleNamespace(local_path="unused-donor"),
        ),
    )


def test_dense_transfer_token_checkpointing_complements_outer_layers() -> None:
    class Controlled:
        def __init__(self) -> None:
            self.enabled: bool | None = None
            self.transfer_mlp = SimpleNamespace(checkpoint_token_branch=False)

        def configure_transfer_execution(
            self,
            *,
            execution_mode: str | None = None,
            checkpoint_token_branch: bool | None = None,
        ) -> None:
            assert execution_mode is None
            self.enabled = checkpoint_token_branch
            self.transfer_mlp.checkpoint_token_branch = bool(checkpoint_token_branch)

    modules = [Controlled(), Controlled(), Controlled()]
    inner = _set_dense_transfer_token_checkpointing(
        modules,
        (0, 12, 23),
        outer_checkpoint_layer_indices=(12,),
        enabled=True,
    )

    assert inner == (0, 23)
    assert [module.enabled for module in modules] == [True, False, True]

    selected = _set_dense_transfer_token_checkpointing(
        modules,
        (0, 12, 23),
        outer_checkpoint_layer_indices=(12,),
        enabled=True,
        checkpoint_layer_indices=(23,),
    )
    assert selected == (23,)
    assert [module.enabled for module in modules] == [False, False, True]

    ordinary = _set_dense_transfer_token_checkpointing(
        modules,
        (0, 12, 23),
        outer_checkpoint_layer_indices=(),
        enabled=True,
        checkpoint_layer_indices=(),
    )
    assert ordinary == ()
    assert [module.enabled for module in modules] == [False, False, False]

    with pytest.raises(RuntimeError, match="nested outer/transfer"):
        _set_dense_transfer_token_checkpointing(
            modules,
            (0, 12, 23),
            outer_checkpoint_layer_indices=(12,),
            enabled=True,
            checkpoint_layer_indices=(12,),
        )

    for invalid in ((23, 0), (0, 0), (7,)):
        with pytest.raises(RuntimeError):
            _set_dense_transfer_token_checkpointing(
                modules,
                (0, 12, 23),
                outer_checkpoint_layer_indices=(),
                enabled=True,
                checkpoint_layer_indices=invalid,
            )

    with pytest.raises(RuntimeError, match="different lengths"):
        _set_dense_transfer_token_checkpointing(
            modules[:2],
            (0, 12, 23),
            outer_checkpoint_layer_indices=(),
            enabled=True,
            checkpoint_layer_indices=(),
        )


def test_dense_transfer_checkpointing_rejects_module_state_mismatch() -> None:
    class Lying:
        def __init__(self) -> None:
            self.transfer_mlp = SimpleNamespace(checkpoint_token_branch=False)

        def configure_transfer_execution(self, **_kwargs: object) -> None:
            return None

    with pytest.raises(RuntimeError, match="state mismatch at layer 0"):
        _set_dense_transfer_token_checkpointing(
            [Lying()],
            (0,),
            outer_checkpoint_layer_indices=(),
            enabled=True,
            checkpoint_layer_indices=(0,),
        )


def test_builder_registers_the_exact_frozen_teacher_parameters() -> None:
    teacher = _Teacher()
    freeze_module(teacher)
    gate = teacher.layers[0].mlp.gate_proj.weight
    up = teacher.layers[0].mlp.up_proj.weight
    down = teacher.layers[0].mlp.down_proj.weight

    with (
        patch("twen.training.builder.load_qwen35_text_causal_lm", return_value=_Student()),
        patch("twen.training.builder.load_layer_mapping", return_value=(0,)),
        patch(
            "twen.training.builder.load_channel_mapping",
            return_value=(torch.tensor([[0, 1, 2], [3, 4, 5]]),),
        ),
        patch(
            "twen.training.builder._load_adapter_initialization",
            return_value=((torch.randn(4, 3), torch.randn(3, 4)),),
        ),
    ):
        built = build_transfer_model(
            _config(),  # type: ignore[arg-type]
            device="cpu",
            dtype=torch.float32,
            donor_text_model=teacher,
        )

    transfer = built.transfer_modules[0].transfer_mlp
    assert built.donor_teacher_shared is True
    assert transfer.gate_weight is gate
    assert transfer.up_weight is up
    assert transfer.down_weight is down
    assert transfer.gate_weight.untyped_storage().data_ptr() == gate.untyped_storage().data_ptr()


def test_builder_loads_enabled_native_mtp_frozen_and_outside_transfer_parameters() -> None:
    teacher = _Teacher()
    freeze_module(teacher)
    config = _config()
    config.losses.mtp = 0.25
    mtp = torch.nn.Linear(3, 3, bias=False)
    freeze_module(mtp)

    with (
        patch("twen.training.builder.load_qwen35_text_causal_lm", return_value=_Student()),
        patch("twen.training.builder.load_qwen35_mtp", return_value=mtp) as load_mtp,
        patch("twen.training.builder.load_layer_mapping", return_value=(0,)),
        patch(
            "twen.training.builder.load_channel_mapping",
            return_value=(torch.tensor([[0, 1, 2], [3, 4, 5]]),),
        ),
        patch(
            "twen.training.builder._load_adapter_initialization",
            return_value=((torch.randn(4, 3), torch.randn(3, 4)),),
        ),
    ):
        built = build_transfer_model(
            config,  # type: ignore[arg-type]
            device="cpu",
            dtype=torch.float32,
            donor_text_model=teacher,
        )

    assert built.mtp is mtp
    assert all(not parameter.requires_grad for parameter in built.mtp.parameters())
    assert all(
        candidate is not parameter
        for parameter in built.mtp.parameters()
        for module in built.transfer_modules
        for candidate in module.parameters()
    )
    load_mtp.assert_called_once_with(
        "unused-backbone",
        dtype=torch.float32,
        device="cpu",
        trainable=False,
    )


def test_shared_donor_requires_a_frozen_teacher() -> None:
    teacher = _Teacher()
    with pytest.raises(BuildError, match="requires the teacher to be frozen"):
        _shared_donor_mlp_weights(teacher, 0)


@pytest.mark.parametrize("world_size, expected_shared", [(1, True), (2, False)])
def test_engine_only_enables_cross_model_sharing_on_one_device(
    world_size: int,
    expected_shared: bool,
) -> None:
    teacher = _Teacher()
    config = SimpleNamespace(
        stage="dense-oracle",
        losses=SimpleNamespace(
            hidden_alignment=0.1,
            hidden_alignment_batch_fraction=0.05,
        ),
        architecture=SimpleNamespace(expert_initialization="donor"),
        runtime=SimpleNamespace(sharding="ddp"),
        sources=SimpleNamespace(teacher=SimpleNamespace(local_path="teacher")),
    )
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=world_size,
        device=torch.device("cpu"),
        initialized_here=False,
    )
    supplied_donor: list[object | None] = []

    def build(*_args: object, donor_text_model: object | None, **_kwargs: object) -> BuiltModel:
        supplied_donor.append(donor_text_model)
        return BuiltModel(
            model=_Student(),
            transfer_modules=(),
            student_layer_indices=(),
            donor_teacher_shared=donor_text_model is not None,
        )

    with (
        patch("twen.model_loading.load_qwen35_text_model", return_value=teacher),
        patch("twen.training.engine.build_transfer_model", side_effect=build),
        patch("twen.training.engine.wrap_frozen_text_model", side_effect=lambda value, *_: value),
    ):
        built, actual_teacher = _build_transfer_and_teacher(
            config,  # type: ignore[arg-type]
            context,
            dtype=torch.float32,
            build_device="cpu",
        )

    assert actual_teacher is teacher
    assert supplied_donor == [teacher if expected_shared else None]
    assert built.donor_teacher_shared is expected_shared
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_engine_keeps_teacher_on_cpu_when_single_gpu_offload_is_enabled() -> None:
    teacher = _Teacher()
    config = SimpleNamespace(
        stage="dense-oracle",
        losses=SimpleNamespace(
            hidden_alignment=0.1,
            hidden_alignment_batch_fraction=0.05,
        ),
        architecture=SimpleNamespace(expert_initialization="donor"),
        runtime=SimpleNamespace(sharding="fsdp2", teacher_cpu_offload=True),
        sources=SimpleNamespace(teacher=SimpleNamespace(local_path="teacher")),
    )
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cuda", 0),
        initialized_here=False,
    )

    with (
        patch("twen.model_loading.load_qwen35_text_model", return_value=teacher) as load,
        patch(
            "twen.training.engine.build_transfer_model",
            return_value=BuiltModel(
                model=_Student(),
                transfer_modules=(),
                student_layer_indices=(),
                donor_teacher_shared=True,
            ),
        ) as build,
        patch("twen.training.engine.wrap_frozen_text_model") as wrap,
    ):
        built, actual_teacher = _build_transfer_and_teacher(
            config,  # type: ignore[arg-type]
            context,
            dtype=torch.float32,
            build_device="cuda:0",
        )

    load.assert_called_once_with("teacher", dtype=torch.float32, device="cpu")
    assert build.call_args.kwargs["donor_text_model"] is teacher
    wrap.assert_not_called()
    assert built.donor_teacher_shared
    assert actual_teacher is teacher


def test_engine_rejects_teacher_cpu_offload_before_multigpu_model_load() -> None:
    config = SimpleNamespace(
        stage="dense-oracle",
        losses=SimpleNamespace(hidden_alignment=0.1),
        runtime=SimpleNamespace(sharding="fsdp2", teacher_cpu_offload=True),
    )
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=2,
        device=torch.device("cuda", 0),
        initialized_here=False,
    )
    with (
        patch("twen.model_loading.load_qwen35_text_model") as load,
        pytest.raises(RuntimeError, match="exactly one GPU"),
    ):
        _build_transfer_and_teacher(
            config,  # type: ignore[arg-type]
            context,
            dtype=torch.float32,
            build_device="cpu",
        )
    load.assert_not_called()
