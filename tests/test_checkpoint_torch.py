"""CPU-only backend checks; one test creates AdamW state but never performs a step."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from twen.runtime.checkpoint import (  # noqa: E402
    CheckpointManager,
    TorchDistributedCheckpointBackend,
)
from twen.runtime.state import DataCursor, RNGState, TrainerState  # noqa: E402
from twen.training.stateful import (  # noqa: E402
    OptimizerState,
    TrainableModelState,
    materialize_adamw_state,
)


class _TiedDeltaModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(11, 3)
        self.lm_head = torch.nn.Linear(3, 11, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.delta = torch.nn.Linear(3, 3, bias=False)
        self.register_buffer("channel_indices", torch.arange(4))
        self.embed_tokens.weight.requires_grad_(False)


class _LegacyModelState:
    """Pre-filter TrainableModelState used to verify old DCP compatibility."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def state_dict(self) -> dict[str, Any]:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        return get_model_state_dict(
            self.model,
            options=StateDictOptions(
                full_state_dict=False,
                cpu_offload=False,
                ignore_frozen_params=True,
                strict=False,
            ),
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        set_model_state_dict(
            self.model,
            state_dict,
            options=StateDictOptions(
                full_state_dict=False,
                cpu_offload=False,
                ignore_frozen_params=True,
                strict=False,
            ),
        )


def _tensor_storage_bytes(metadata: Any) -> int:
    return math.prod(metadata.size) * torch.empty((), dtype=metadata.properties.dtype).element_size()


def _runtime() -> tuple[TrainerState, DataCursor, RNGState]:
    return (
        TrainerState(
            run_id="torch-backend-test",
            stage="dense-oracle",
            global_batch_tokens=16,
            micro_batch_tokens_per_rank=16,
        ),
        DataCursor(),
        RNGState.capture(),
    )


def test_torch_file_backend_round_trip_on_cpu(tmp_path: Path) -> None:
    state, cursor, rng = _runtime()
    manager = CheckpointManager(tmp_path, backend="torch")
    manager.save(
        {"adapter": torch.arange(4)},
        trainer_state=state,
        data_cursor=cursor,
        rng_state=rng,
        critical_fingerprint="config",
        data_fingerprint="data",
    )
    target = {"adapter": torch.empty(4)}
    loaded = manager.load(target)
    assert torch.equal(loaded.stateful["adapter"], torch.arange(4))


@pytest.mark.skipif(
    not TorchDistributedCheckpointBackend.available(),
    reason="torch.distributed.checkpoint is unavailable",
)
def test_dcp_backend_round_trip_without_a_process_group(tmp_path: Path) -> None:
    state, cursor, rng = _runtime()
    manager = CheckpointManager(tmp_path, backend="dcp")
    manager.save(
        {"adapter": torch.arange(4)},
        trainer_state=state,
        data_cursor=cursor,
        rng_state=rng,
        critical_fingerprint="config",
        data_fingerprint="data",
    )
    target = {"adapter": torch.empty(4)}
    loaded = manager.load(target)
    assert torch.equal(loaded.stateful["adapter"], torch.arange(4))


@pytest.mark.skipif(
    not TorchDistributedCheckpointBackend.available(),
    reason="torch.distributed.checkpoint is unavailable",
)
def test_production_model_and_materialized_adam_state_round_trip_without_step(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(3, 2, bias=False)
    frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    model.register_parameter("frozen_source", frozen)
    optimizer = torch.optim.AdamW([model.weight], lr=1e-3)
    materialize_adamw_state(optimizer)
    original = model.weight.detach().clone()
    stateful = {
        "model": TrainableModelState(model),
        "optimizer": OptimizerState(model, optimizer),
    }
    state, cursor, rng = _runtime()
    manager = CheckpointManager(tmp_path, backend="dcp")
    checkpoint = manager.save(
        stateful,
        trainer_state=state,
        data_cursor=cursor,
        rng_state=rng,
        critical_fingerprint="config",
        data_fingerprint="data",
    )
    with torch.no_grad():
        model.weight.zero_()
    optimizer.state[model.weight]["exp_avg"].fill_(9)
    manager.load(stateful, checkpoint)
    assert torch.equal(model.weight, original)
    assert torch.count_nonzero(optimizer.state[model.weight]["exp_avg"]) == 0
    assert optimizer.state[model.weight]["step"].item() == 0


@pytest.mark.skipif(
    not TorchDistributedCheckpointBackend.available(),
    reason="torch.distributed.checkpoint is unavailable",
)
def test_trainable_model_state_omits_tied_frozen_weights_and_static_buffers(
    tmp_path: Path,
) -> None:
    from torch.distributed.checkpoint import FileSystemReader

    model = _TiedDeltaModel()
    wrapper = TrainableModelState(model)
    assert set(wrapper.state_dict()) == {"delta.weight"}

    original_delta = model.delta.weight.detach().clone()
    state, cursor, rng = _runtime()
    manager = CheckpointManager(tmp_path, backend="dcp")
    checkpoint = manager.save(
        {"model": wrapper},
        trainer_state=state,
        data_cursor=cursor,
        rng_state=rng,
        critical_fingerprint="config",
        data_fingerprint="data",
    )

    metadata = FileSystemReader(str(checkpoint / "state")).read_metadata()
    model_entries = {
        name: value
        for name, value in metadata.state_dict_metadata.items()
        if name.startswith("model.")
    }
    assert set(model_entries) == {"model.delta.weight"}
    assert sum(_tensor_storage_bytes(value) for value in model_entries.values()) == (
        original_delta.numel() * original_delta.element_size()
    )

    with torch.no_grad():
        model.delta.weight.zero_()
        model.embed_tokens.weight.fill_(7)
        model.channel_indices.fill_(9)
    manager.load({"model": wrapper}, checkpoint)
    assert torch.equal(model.delta.weight, original_delta)
    assert torch.count_nonzero(model.embed_tokens.weight != 7) == 0
    assert torch.count_nonzero(model.channel_indices != 9) == 0
    assert model.lm_head.weight is model.embed_tokens.weight


def test_trainable_model_state_production_geometry_is_exact_delta_bytes() -> None:
    class TransferLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_adapter = torch.nn.Linear(
                1024,
                4096,
                bias=False,
                device="meta",
                dtype=torch.float32,
            )
            self.output_adapter = torch.nn.Linear(
                4096,
                1024,
                bias=False,
                device="meta",
                dtype=torch.float32,
            )
            self.branch_scale = torch.nn.Parameter(
                torch.empty(1, device="meta", dtype=torch.float32)
            )
            self.register_buffer(
                "channel_indices",
                torch.empty(8, 1536, device="meta", dtype=torch.int64),
            )

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([TransferLayer() for _ in range(24)])
    model.embed_tokens = torch.nn.Embedding(
        248320,
        1024,
        device="meta",
        dtype=torch.bfloat16,
    )
    model.lm_head = torch.nn.Linear(
        1024,
        248320,
        bias=False,
        device="meta",
        dtype=torch.bfloat16,
    )
    model.lm_head.weight = model.embed_tokens.weight
    model.embed_tokens.weight.requires_grad_(False)

    state = TrainableModelState(model).state_dict()
    assert len(state) == 72
    assert sum("adapter" in name for name in state) == 48
    assert sum(name.endswith("branch_scale") for name in state) == 24
    assert not any(
        "lm_head" in name or "embed_tokens" in name or "channel_indices" in name
        for name in state
    )
    assert sum(value.numel() * value.element_size() for value in state.values()) == 805_306_464


@pytest.mark.skipif(
    not TorchDistributedCheckpointBackend.available(),
    reason="torch.distributed.checkpoint is unavailable",
)
def test_filtered_model_state_loads_a_legacy_dcp_with_frozen_aliases(
    tmp_path: Path,
) -> None:
    model = _TiedDeltaModel()
    original_delta = model.delta.weight.detach().clone()
    state, cursor, rng = _runtime()
    manager = CheckpointManager(tmp_path, backend="dcp")
    checkpoint = manager.save(
        {"model": _LegacyModelState(model)},
        trainer_state=state,
        data_cursor=cursor,
        rng_state=rng,
        critical_fingerprint="config",
        data_fingerprint="data",
    )

    with torch.no_grad():
        model.delta.weight.zero_()
        model.embed_tokens.weight.fill_(5)
        model.channel_indices.fill_(6)
    manager.load({"model": TrainableModelState(model)}, checkpoint)
    assert torch.equal(model.delta.weight, original_delta)
    assert torch.count_nonzero(model.embed_tokens.weight != 5) == 0
    assert torch.count_nonzero(model.channel_indices != 6) == 0
