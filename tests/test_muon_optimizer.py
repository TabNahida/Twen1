"""CPU-only contracts for the opt-in Muon/AdamW optimizer bundle."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from twen.config import LossConfig  # noqa: E402
from twen.data.cursor import DatasetLayout  # noqa: E402
from twen.preflight import (  # noqa: E402
    TrainingPreflightError,
    validate_optimizer_world_size,
)
from twen.runtime.checkpoint import (  # noqa: E402
    CheckpointManager,
    TorchDistributedCheckpointBackend,
)
from twen.runtime.state import DataCursor, RNGState, TrainerState  # noqa: E402
from twen.training.builder import BuiltModel  # noqa: E402
from twen.training.engine import (  # noqa: E402
    _build_optimizer,
    _clip_optimizer_gradients,
    _load_or_initialize,
)
from twen.training.stateful import (  # noqa: E402
    OptimizerBundle,
    OptimizerState,
    TokenLRScheduler,
    TrainableModelState,
    materialize_adamw_state,
    materialize_muon_state,
)


class _DenseTransfer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapters = torch.nn.Module()
        self.adapters.input_adapter = torch.nn.Linear(
            3,
            4,
            bias=False,
            dtype=torch.float32,
        )
        self.adapters.output_adapter = torch.nn.Linear(
            4,
            3,
            bias=False,
            dtype=torch.float32,
        )
        self.branch_scale = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))


def _engine_config(adapter_optimizer: str) -> SimpleNamespace:
    return SimpleNamespace(
        stage="dense-oracle",
        optimizer=SimpleNamespace(
            adapter_optimizer=adapter_optimizer,
            adapter_lr=1e-4,
            router_lr=1e-3,
            lora_lr=2e-4,
            scale_lr=3e-4,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1e-8,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_ns_coefficients=(3.4445, -4.775, 2.0315),
            muon_eps=1e-7,
            muon_ns_steps=5,
            muon_adjust_lr_fn="match_rms_adamw",
        ),
        runtime=SimpleNamespace(fused_adamw=True),
    )


def _built_dense() -> BuiltModel:
    module = _DenseTransfer()
    return BuiltModel(
        model=module,
        transfer_modules=(module,),
        student_layer_indices=(0,),
    )


def _parameter_ids(groups: list[dict[str, Any]]) -> set[int]:
    return {id(parameter) for group in groups for parameter in group["params"]}


def test_build_optimizer_keeps_legacy_single_adamw_schema() -> None:
    built = _built_dense()
    optimizer = _build_optimizer(_engine_config("adamw"), built)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert [group["name"] for group in optimizer.param_groups] == ["adapters", "scale"]
    assert all(
        set(optimizer.state[parameter]) == {"step", "exp_avg", "exp_avg_sq"}
        for group in optimizer.param_groups
        for parameter in group["params"]
    )


def test_build_optimizer_routes_only_2d_adapters_to_muon() -> None:
    built = _built_dense()
    optimizer = _build_optimizer(_engine_config("muon"), built)

    assert isinstance(optimizer, OptimizerBundle)
    assert len(optimizer.optimizers) == 2
    muon, adamw = optimizer.optimizers
    assert isinstance(muon, torch.optim.Muon)
    assert isinstance(adamw, torch.optim.AdamW)
    assert [group["name"] for group in muon.param_groups] == ["adapters"]
    assert [group["name"] for group in adamw.param_groups] == ["scale"]
    assert all(parameter.ndim == 2 for group in muon.param_groups for parameter in group["params"])
    assert all(parameter.ndim == 1 for group in adamw.param_groups for parameter in group["params"])
    assert muon.param_groups[0]["momentum"] == 0.95
    assert muon.param_groups[0]["nesterov"] is True
    assert muon.param_groups[0]["ns_steps"] == 5
    assert muon.param_groups[0]["adjust_lr_fn"] == "match_rms_adamw"
    assert all(
        set(muon.state[parameter]) == {"momentum_buffer"}
        for parameter in muon.param_groups[0]["params"]
    )
    assert all(
        set(adamw.state[parameter]) == {"step", "exp_avg", "exp_avg_sq"}
        for parameter in adamw.param_groups[0]["params"]
    )
    expected = {
        id(parameter)
        for parameter in built.transfer_modules[0].parameters()
        if parameter.requires_grad
    }
    assert _parameter_ids(optimizer.param_groups) == expected


def test_build_optimizer_rejects_non_matrix_adapter_parameter() -> None:
    built = _built_dense()
    built.transfer_modules[0].adapters.vector = torch.nn.Parameter(
        torch.ones(3, dtype=torch.float32)
    )
    with pytest.raises(RuntimeError, match="must all be 2D"):
        _build_optimizer(_engine_config("muon"), built)


class _CountingOptimizer:
    def __init__(self, parameter: Any, *, name: str, lr: float) -> None:
        self.param_groups = [{"params": [parameter], "name": name, "lr": lr}]
        self.defaults: dict[str, Any] = {}
        self.steps = 0
        self.zero_grad_calls: list[bool] = []

    def step(self) -> None:
        self.steps += 1

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.zero_grad_calls.append(set_to_none)


def test_optimizer_bundle_flattens_groups_and_delegates_without_overlap() -> None:
    first_parameter = torch.nn.Parameter(torch.ones(2, 2))
    second_parameter = torch.nn.Parameter(torch.ones(1))
    first = _CountingOptimizer(first_parameter, name="adapters", lr=1.0)
    second = _CountingOptimizer(second_parameter, name="scale", lr=2.0)
    bundle = OptimizerBundle(
        (first, second),
        expected_parameters=(first_parameter, second_parameter),
    )

    assert bundle.param_groups[0] is first.param_groups[0]
    assert bundle.param_groups[1] is second.param_groups[0]
    bundle.step()
    bundle.zero_grad(set_to_none=False)
    assert (first.steps, second.steps) == (1, 1)
    assert first.zero_grad_calls == [False]
    assert second.zero_grad_calls == [False]
    with pytest.raises(ValueError, match="closures"):
        bundle.step(lambda: None)

    overlapping = _CountingOptimizer(first_parameter, name="duplicate", lr=3.0)
    with pytest.raises(ValueError, match="overlap"):
        OptimizerBundle((first, overlapping))
    with pytest.raises(ValueError, match="missing=1"):
        OptimizerBundle((first,), expected_parameters=(first_parameter, second_parameter))


def test_optimizer_bundle_gradient_clipping_covers_muon_and_adamw_groups() -> None:
    adapter = torch.nn.Parameter(torch.ones(2, 2))
    scale = torch.nn.Parameter(torch.ones(1))
    muon = _CountingOptimizer(adapter, name="adapters", lr=1.0)
    adamw = _CountingOptimizer(scale, name="scale", lr=2.0)
    bundle = OptimizerBundle((muon, adamw))
    adapter.grad = torch.full_like(adapter, 3.0)
    scale.grad = torch.full_like(scale, 4.0)

    norm = _clip_optimizer_gradients(bundle, 1.0)

    torch.testing.assert_close(norm, torch.sqrt(torch.tensor(52.0)))
    combined = torch.cat((adapter.grad.reshape(-1), scale.grad.reshape(-1)))
    assert torch.linalg.vector_norm(combined) <= 1.000001


def test_materialize_muon_state_is_idempotent_and_step_free() -> None:
    parameter = torch.nn.Parameter(torch.randn(4, 3, dtype=torch.float32))
    optimizer = torch.optim.Muon([parameter], lr=1e-4)
    assert optimizer.state[parameter] == {}

    materialize_muon_state(optimizer)
    momentum = optimizer.state[parameter]["momentum_buffer"]
    assert torch.count_nonzero(momentum) == 0
    assert momentum.shape == parameter.shape
    assert momentum.dtype == parameter.dtype
    materialize_muon_state(optimizer)
    assert optimizer.state[parameter]["momentum_buffer"] is momentum

    optimizer.state[parameter]["unexpected"] = torch.tensor(0)
    with pytest.raises(ValueError, match="does not match"):
        materialize_muon_state(optimizer)


def test_optimizer_state_accepts_single_bundle_and_arbitrary_iterable() -> None:
    model = _DenseTransfer()
    adapter = next(model.adapters.parameters())
    scale = model.branch_scale
    muon = torch.optim.Muon([adapter], lr=1e-4)
    adamw = torch.optim.AdamW([scale], lr=3e-4)
    bundle = OptimizerBundle((muon, adamw), expected_parameters=(adapter, scale))

    assert OptimizerState(model, muon).optimizer is muon
    assert OptimizerState(model, bundle).optimizer == (muon, adamw)
    assert OptimizerState(model, (item for item in (muon, adamw))).optimizer == (
        muon,
        adamw,
    )


def test_token_scheduler_updates_real_child_optimizer_groups() -> None:
    first_parameter = torch.nn.Parameter(torch.ones(2, 2))
    second_parameter = torch.nn.Parameter(torch.ones(1))
    first = _CountingOptimizer(first_parameter, name="adapters", lr=1.0)
    second = _CountingOptimizer(second_parameter, name="scale", lr=2.0)
    bundle = OptimizerBundle((first, second))
    scheduler = TokenLRScheduler(bundle, warmup_tokens=100, max_tokens=1_000)

    assert [group["lr"] for group in bundle.param_groups] == [0.0, 0.0]
    scheduler.step_tokens(50)
    assert [group["lr"] for group in bundle.param_groups] == [0.5, 1.0]
    saved = scheduler.state_dict()

    restored_first = _CountingOptimizer(first_parameter, name="adapters", lr=1.0)
    restored_second = _CountingOptimizer(second_parameter, name="scale", lr=2.0)
    restored_bundle = OptimizerBundle((restored_first, restored_second))
    restored = TokenLRScheduler(
        restored_bundle,
        warmup_tokens=100,
        max_tokens=1_000,
    )
    restored.load_state_dict(saved)
    assert restored.consumed_tokens == 50
    assert [group["lr"] for group in restored_bundle.param_groups] == [0.5, 1.0]


def test_muon_world_size_contract_fails_closed() -> None:
    config = SimpleNamespace(optimizer=SimpleNamespace(adapter_optimizer="muon"))
    validate_optimizer_world_size(config, 1)
    with pytest.raises(TrainingPreflightError, match="requires world_size=1"):
        validate_optimizer_world_size(config, 2)
    with pytest.raises(TrainingPreflightError, match="positive integer"):
        validate_optimizer_world_size(config, True)


@pytest.mark.skipif(
    not TorchDistributedCheckpointBackend.available(),
    reason="torch.distributed.checkpoint is unavailable",
)
def test_materialized_muon_and_adamw_dcp_round_trip_without_step(tmp_path) -> None:
    model = _DenseTransfer()
    adapter_parameters = tuple(model.adapters.parameters())
    scale = model.branch_scale
    muon = torch.optim.Muon(
        [{"params": adapter_parameters, "name": "adapters", "lr": 1e-4}],
        adjust_lr_fn="match_rms_adamw",
    )
    adamw = torch.optim.AdamW(
        [{"params": [scale], "name": "scale", "lr": 3e-4}],
        betas=(0.9, 0.95),
    )
    materialize_muon_state(muon)
    materialize_adamw_state(adamw)
    bundle = OptimizerBundle(
        (muon, adamw),
        expected_parameters=(*adapter_parameters, scale),
    )
    stateful = {
        "model": TrainableModelState(model),
        "optimizer": OptimizerState(model, bundle),
    }
    trainer_state = TrainerState(
        run_id="muon-dcp-test",
        stage="dense-oracle",
        global_batch_tokens=16,
        micro_batch_tokens_per_rank=16,
    )
    manager = CheckpointManager(tmp_path, backend="dcp")
    originals = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    checkpoint = manager.save(
        stateful,
        trainer_state=trainer_state,
        data_cursor=DataCursor(),
        rng_state=RNGState.capture(),
        critical_fingerprint="config",
        data_fingerprint="data",
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for parameter in adapter_parameters:
            muon.state[parameter]["momentum_buffer"].fill_(7)
        adamw.state[scale]["exp_avg"].fill_(8)
        adamw.state[scale]["exp_avg_sq"].fill_(9)
        adamw.state[scale]["step"].fill_(10)

    manager.load(stateful, checkpoint)
    assert all(
        torch.equal(parameter, originals[name]) for name, parameter in model.named_parameters()
    )
    assert all(
        torch.count_nonzero(muon.state[parameter]["momentum_buffer"]) == 0
        for parameter in adapter_parameters
    )
    assert torch.count_nonzero(adamw.state[scale]["exp_avg"]) == 0
    assert torch.count_nonzero(adamw.state[scale]["exp_avg_sq"]) == 0
    assert adamw.state[scale]["step"].item() == 0


@pytest.mark.skipif(
    not TorchDistributedCheckpointBackend.available(),
    reason="torch.distributed.checkpoint is unavailable",
)
def test_stepped_muon_bundle_scheduler_and_optimizer_resume_exactly(tmp_path) -> None:
    model = _DenseTransfer()
    adapter_parameters = tuple(model.adapters.parameters())
    scale = model.branch_scale
    muon = torch.optim.Muon(
        [{"params": adapter_parameters, "name": "adapters", "lr": 1e-4}],
        adjust_lr_fn="match_rms_adamw",
    )
    adamw = torch.optim.AdamW(
        [{"params": [scale], "name": "scale", "lr": 3e-4}],
        betas=(0.9, 0.95),
    )
    materialize_muon_state(muon)
    materialize_adamw_state(adamw)
    bundle = OptimizerBundle(
        (muon, adamw),
        expected_parameters=(*adapter_parameters, scale),
    )
    scheduler = TokenLRScheduler(
        bundle,
        warmup_tokens=100,
        max_tokens=1_000,
    )
    scheduler.step_tokens(128)
    for parameter in (*adapter_parameters, scale):
        parameter.grad = torch.ones_like(parameter)
    bundle.step()
    bundle.zero_grad(set_to_none=True)

    expected_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    expected_muon = {
        parameter: muon.state[parameter]["momentum_buffer"].clone()
        for parameter in adapter_parameters
    }
    expected_scale_avg = adamw.state[scale]["exp_avg"].clone()
    expected_scale_sq = adamw.state[scale]["exp_avg_sq"].clone()
    expected_scale_step = adamw.state[scale]["step"].clone()
    expected_scheduler = scheduler.state_dict()

    stateful = {
        "model": TrainableModelState(model),
        "optimizer": OptimizerState(model, bundle),
        "scheduler": scheduler,
    }
    manager = CheckpointManager(tmp_path, backend="dcp")
    checkpoint = manager.save(
        stateful,
        trainer_state=TrainerState(
            run_id="muon-resume-test",
            stage="dense-oracle",
            committed_tokens=128,
            global_batch_tokens=16,
            micro_batch_tokens_per_rank=16,
        ),
        data_cursor=DataCursor(global_token_index=128),
        rng_state=RNGState.capture(),
        critical_fingerprint="config",
        data_fingerprint="data",
        extra_metadata={
            "data_mode": "prepared-text",
            "teacher_kd_manifest_sha256": None,
            "optimizer": {"adapter_optimizer": "muon", "bundle": True},
        },
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for parameter in adapter_parameters:
            muon.state[parameter]["momentum_buffer"].zero_()
        adamw.state[scale]["exp_avg"].zero_()
        adamw.state[scale]["exp_avg_sq"].zero_()
        adamw.state[scale]["step"].zero_()
    scheduler.step_tokens(256)

    loaded = manager.load(
        stateful,
        checkpoint,
        expected_critical_fingerprint="config",
        expected_data_fingerprint="data",
        expected_run_id="muon-resume-test",
    )

    assert loaded.metadata["extra"]["teacher_kd_manifest_sha256"] is None
    assert all(
        torch.equal(parameter, expected_parameters[name])
        for name, parameter in model.named_parameters()
    )
    assert all(
        torch.equal(muon.state[parameter]["momentum_buffer"], expected_muon[parameter])
        for parameter in adapter_parameters
    )
    assert torch.equal(adamw.state[scale]["exp_avg"], expected_scale_avg)
    assert torch.equal(adamw.state[scale]["exp_avg_sq"], expected_scale_sq)
    assert torch.equal(adamw.state[scale]["step"], expected_scale_step)
    assert scheduler.state_dict() == expected_scheduler

    scheduler.step_tokens(256)
    assert [group["lr"] for group in bundle.param_groups] == [
        group["lr"]
        for optimizer in bundle.optimizers
        for group in optimizer.param_groups
    ]
    assert [group["lr"] for group in bundle.param_groups] != list(
        expected_scheduler["base_lrs"]
    )


@pytest.mark.skipif(
    not TorchDistributedCheckpointBackend.available(),
    reason="torch.distributed.checkpoint is unavailable",
)
def test_fork_from_muon_checkpoint_loads_only_model_delta(tmp_path) -> None:
    source = _built_dense()
    with torch.no_grad():
        for parameter in source.model.parameters():
            parameter.fill_(2.0)
    source_optimizer = _build_optimizer(_engine_config("muon"), source)
    source_scheduler = TokenLRScheduler(
        source_optimizer,
        warmup_tokens=100,
        max_tokens=1_000,
    )
    source_manager = CheckpointManager(tmp_path / "source", backend="dcp")
    checkpoint = source_manager.save(
        {
            "model": TrainableModelState(source.model),
            "optimizer": OptimizerState(source.model, source_optimizer),
            "scheduler": source_scheduler,
        },
        trainer_state=TrainerState(
            run_id="source",
            stage="dense-oracle",
            global_batch_tokens=16,
            micro_batch_tokens_per_rank=16,
        ),
        data_cursor=DataCursor(),
        rng_state=RNGState.capture(),
        critical_fingerprint="source-config",
        data_fingerprint="source-data",
    )
    expected = {
        name: parameter.detach().clone()
        for name, parameter in source.model.named_parameters()
    }

    target = _built_dense()
    target_optimizer = _build_optimizer(_engine_config("muon"), target)
    target_scheduler = TokenLRScheduler(
        target_optimizer,
        warmup_tokens=100,
        max_tokens=1_000,
    )
    target_stateful = {
        "model": TrainableModelState(target.model),
        "optimizer": OptimizerState(target.model, target_optimizer),
        "scheduler": target_scheduler,
    }
    config = SimpleNamespace(
        run_id="target",
        stage="dense-oracle",
        data=SimpleNamespace(
            shuffle_seed=3407,
            global_batch_tokens=16,
            quality_cooldown_start_tokens=None,
        ),
        losses=LossConfig(),
    )
    report = SimpleNamespace(
        config_fingerprint="a" * 64,
        data_fingerprint="b" * 64,
        batch=SimpleNamespace(
            world_size=1,
            micro_batch_tokens_per_rank=16,
            gradient_accumulation_steps=1,
            global_batch_tokens=16,
        ),
    )
    store = SimpleNamespace(
        layout=DatasetLayout.from_shards(
            (("prepared-shard", 8),),
            fingerprint="prepared-data",
        )
    )

    state, cursor, loaded = _load_or_initialize(
        CheckpointManager(tmp_path / "target", backend="dcp"),
        target_stateful,
        config,  # type: ignore[arg-type]
        report,  # type: ignore[arg-type]
        store,  # type: ignore[arg-type]
        resume="none",
        fork_from=str(checkpoint),
    )

    assert loaded is None
    assert state.global_step == 0
    assert cursor.next_global_sample == 0
    assert all(
        torch.equal(parameter, expected[name])
        for name, parameter in target.model.named_parameters()
    )
    muon, adamw = target_optimizer.optimizers
    assert all(
        torch.count_nonzero(muon.state[parameter]["momentum_buffer"]) == 0
        for group in muon.param_groups
        for parameter in group["params"]
    )
    assert all(
        torch.count_nonzero(adamw.state[parameter]["exp_avg"]) == 0
        and torch.count_nonzero(adamw.state[parameter]["exp_avg_sq"]) == 0
        and adamw.state[parameter]["step"].item() == 0
        for group in adamw.param_groups
        for parameter in group["params"]
    )
    assert target_scheduler.consumed_tokens == 0
