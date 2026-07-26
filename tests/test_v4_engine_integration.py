from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from twen.data import (
    AuthenticatedSourceMap,
    AuthenticatedSourceShard,
    DeterministicSourceMixCursor,
    PreparedTextBatch,
)
from twen.training.distributed import DistributedContext
from twen.training.engine import (
    _build_training_record_store,
    _checkpoint,
    _learning_rate_step_metrics,
    _move_training_batch,
    _named_adjusted_learning_rates,
    _named_learning_rates,
    _optimizer_checkpoint_contract,
    _optimizer_step_and_commit,
    _prepare_source_mix_commit,
    _source_mix_log_contract,
    _source_mix_session_log_fields,
    _student_language_model_forward,
)


def _config(mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            mode=mode,
            manifest_path="prepared/manifest.json",
            teacher_kd_manifest_path=(
                "kd/manifest.json" if mode == "teacher-kd" else None
            ),
            max_sequence_length=4096,
        ),
        losses=SimpleNamespace(kd_temperature=2.0),
    )


def test_prepared_text_store_factory_never_constructs_kd_reader() -> None:
    config = _config("prepared-text")
    sentinel = object()
    with (
        patch(
            "twen.training.engine.KDRecordStore",
            side_effect=AssertionError("prepared-text factory touched KD"),
        ) as kd_store,
        patch(
            "twen.training.engine.PreparedTextRecordStore",
            return_value=sentinel,
        ) as prepared_store,
    ):
        actual = _build_training_record_store(
            config,  # type: ignore[arg-type]
            verify_shards=False,
        )

    assert actual is sentinel
    kd_store.assert_not_called()
    prepared_store.assert_called_once_with(
        "prepared/manifest.json",
        expected_sequence_length=4096,
        verify_shards=False,
    )


def test_teacher_kd_store_factory_preserves_legacy_reader_contract() -> None:
    config = _config("teacher-kd")
    sentinel = object()
    with patch(
        "twen.training.engine.KDRecordStore",
        return_value=sentinel,
    ) as kd_store:
        actual = _build_training_record_store(
            config,  # type: ignore[arg-type]
            verify_shards=True,
        )

    assert actual is sentinel
    kd_store.assert_called_once_with(
        "kd/manifest.json",
        expected_temperature=2.0,
        expected_sequence_length=4096,
        verify_shards=True,
    )


def test_prepared_text_forward_and_device_move_do_not_require_kd_attributes() -> None:
    batch = PreparedTextBatch(
        input_ids=torch.tensor([[1, 2, 3]]),
        labels=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.int64),
    )
    received: dict[str, object] = {}

    def model(**kwargs: object) -> dict[str, object]:
        received.update(kwargs)
        return {"ok": True}

    moved = _move_training_batch(
        _config("prepared-text"),  # type: ignore[arg-type]
        batch,
        torch.device("cpu"),
    )
    result = _student_language_model_forward(
        model,
        moved,
        teacher_kd_enabled=False,
        anchor_hidden_states=None,
        output_hidden_states=False,
    )

    assert result == {"ok": True}
    assert set(received) == {
        "input_ids",
        "attention_mask",
        "labels",
        "anchor_hidden_states",
        "output_hidden_states",
    }
    assert not any(name.startswith("teacher_") for name in received)


def test_legacy_optimizer_checkpoint_contract_defaults_to_single_adamw() -> None:
    config = SimpleNamespace()

    assert _optimizer_checkpoint_contract(config) == {
        "adapter_optimizer": "adamw",
        "bundle": False,
        "adapter_component": "AdamW",
        "non_adapter_component": "AdamW",
        "muon": None,
    }


def _source_mix_cursor() -> DeterministicSourceMixCursor:
    source_map = AuthenticatedSourceMap(
        prepared_dataset_fingerprint="0" * 64,
        extracted_manifest_sha256="1" * 64,
        sequence_length=4,
        shards=(
            AuthenticatedSourceShard(
                source_id="alpha",
                shard_id="alpha-shard",
                sequence_count=8,
                global_sample_start=0,
                output_path="alpha/data.safetensors",
                output_sha256="2" * 64,
            ),
            AuthenticatedSourceShard(
                source_id="beta",
                shard_id="beta-shard",
                sequence_count=8,
                global_sample_start=8,
                output_path="beta/data.safetensors",
                output_sha256="3" * 64,
            ),
        ),
        mix_basis_points=(("alpha", 7_000), ("beta", 3_000)),
    )
    return DeterministicSourceMixCursor(source_map, seed=73)


def _source_mix_report() -> SimpleNamespace:
    effective = (("alpha", 6_000), ("beta", 4_000))
    return SimpleNamespace(
        source_mix_enabled=True,
        source_mix_algorithm="token-deficit-corrected-source-mix-bp-v2",
        source_map_sha256="4" * 64,
        source_mix_dataset_fingerprint="5" * 64,
        source_mix_basis_points=effective,
        source_mix_lineage_basis_points=(("alpha", 7_000), ("beta", 3_000)),
        source_mix_effective_basis_points=effective,
        source_mix_weight_override=True,
        source_mix_seed=73,
        config_fingerprint="6" * 64,
        data_fingerprint="7" * 64,
        source_tree_sha256="8" * 64,
        activation_checkpoint_layer_indices=(),
        hidden_alignment_activation_checkpoint_layer_indices=(),
        dense_transfer_token_checkpoint_layer_indices=(),
        hidden_alignment_dense_transfer_token_checkpoint_layer_indices=(),
        calibration_fingerprints=(),
    )


def _checkpoint_config() -> SimpleNamespace:
    def source(role: str) -> SimpleNamespace:
        return SimpleNamespace(
            manifest_sha256=role * 8,
            model_id=f"test/{role}",
            revision=f"{role}-revision",
            local_path=f"models/{role}",
        )

    return SimpleNamespace(
        data=SimpleNamespace(
            mode="prepared-text",
            manifest_path="prepared/manifest.json",
            manifest_sha256="9" * 64,
            teacher_kd_manifest_path=None,
            teacher_kd_manifest_sha256=None,
            quality_cooldown_start_tokens=None,
            quality_cooldown_manifest_path=None,
            quality_cooldown_manifest_sha256=None,
            quality_cooldown_teacher_kd_manifest_path=None,
            quality_cooldown_teacher_kd_manifest_sha256=None,
        ),
        optimizer=SimpleNamespace(
            adapter_optimizer="muon",
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_ns_coefficients=(3.4445, -4.775, 2.0315),
            muon_eps=1e-7,
            muon_ns_steps=5,
            muon_adjust_lr_fn="match_rms_adamw",
        ),
        losses=SimpleNamespace(mtp=0.1),
        runtime=SimpleNamespace(
            activation_checkpointing=True,
            activation_checkpoint_layer_count=0,
            hidden_alignment_activation_checkpoint_layer_count=None,
            dense_transfer_token_checkpoint=True,
            dense_transfer_checkpoint_layer_count=0,
            hidden_alignment_dense_transfer_checkpoint_layer_count=None,
        ),
        sources=SimpleNamespace(
            backbone=source("backbone"),
            donor=source("donor"),
            teacher=source("teacher"),
            tokenizer=source("tokenizer"),
            folded_experts_path=None,
            folded_experts_sha256=None,
        ),
    )


def test_source_mix_log_contract_keeps_lineage_effective_and_override_explicit() -> None:
    report = _source_mix_report()
    expected = {
        "enabled": True,
        "algorithm": "token-deficit-corrected-source-mix-bp-v2",
        "source_map_sha256": "4" * 64,
        "dataset_fingerprint": "5" * 64,
        "basis_points": {"alpha": 6_000, "beta": 4_000},
        "lineage_basis_points": {"alpha": 7_000, "beta": 3_000},
        "effective_basis_points": {"alpha": 6_000, "beta": 4_000},
        "weight_override": True,
        "seed": 73,
    }

    assert _source_mix_log_contract(report) == expected
    assert _source_mix_session_log_fields(report) == {
        "source_mix_enabled": True,
        "source_mix_algorithm": expected["algorithm"],
        "source_map_sha256": expected["source_map_sha256"],
        "source_mix_dataset_fingerprint": expected["dataset_fingerprint"],
        "source_mix_basis_points": expected["basis_points"],
        "source_mix_lineage_basis_points": expected["lineage_basis_points"],
        "source_mix_effective_basis_points": expected["effective_basis_points"],
        "source_mix_weight_override": True,
        "source_mix_seed": 73,
    }


def test_checkpoint_persists_complete_source_mix_weight_contract(tmp_path: Path) -> None:
    class Manager:
        saved: dict[str, object] | None = None

        def save(self, _stateful: object, **kwargs: object) -> Path:
            self.saved = kwargs
            return tmp_path / "checkpoint"

    manager = Manager()
    cursor = _source_mix_cursor()
    _checkpoint(
        manager,  # type: ignore[arg-type]
        {},
        SimpleNamespace(global_step=0, committed_tokens=0),  # type: ignore[arg-type]
        cursor,
        _checkpoint_config(),  # type: ignore[arg-type]
        _source_mix_report(),  # type: ignore[arg-type]
        kind="periodic",
        boundary=None,
    )

    assert manager.saved is not None
    metadata = manager.saved["extra_metadata"]
    assert isinstance(metadata, dict)
    source_mix = metadata["source_mix"]
    assert isinstance(source_mix, dict)
    assert source_mix["basis_points"] == {"alpha": 6_000, "beta": 4_000}
    assert source_mix["lineage_basis_points"] == {"alpha": 7_000, "beta": 3_000}
    assert source_mix["effective_basis_points"] == {"alpha": 6_000, "beta": 4_000}
    assert source_mix["weight_override"] is True
    assert (
        source_mix["cursor_critical_lineage_fingerprint"]
        == cursor.critical_lineage_fingerprint
    )


class _CountingOptimizer:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


class _LegacyCursor:
    def __init__(self, call_order: list[str]) -> None:
        self.call_order = call_order
        self.committed_tokens = 0

    def commit(self, *, global_batch_samples: int, token_count: int) -> None:
        assert global_batch_samples == 8
        self.call_order.append("commit")
        self.committed_tokens += token_count


def test_source_mix_prevalidation_is_step_free_and_commit_follows_success() -> None:
    cursor = _source_mix_cursor()
    references = cursor.plan_rank_batch(8, rank=0, world_size=1)
    before = cursor.state_dict()
    payload = _prepare_source_mix_commit(
        cursor,
        references,
        (4, 3, 4, 2, 4, 3, 4, 1),
        DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
            initialized_here=False,
        ),
    )
    assert cursor.state_dict() == before
    assert cursor.pending_global_batch == tuple(references)

    optimizer = _CountingOptimizer()
    malformed = replace(payload, token_count=payload.token_count - 1)
    with pytest.raises(ValueError, match="global token total"):
        _optimizer_step_and_commit(
            optimizer,
            cursor,
            global_batch_samples=8,
            committed_tokens=malformed.token_count,
            source_mix_commit=malformed,
        )
    assert optimizer.steps == 0
    assert cursor.state_dict() == before

    _optimizer_step_and_commit(
        optimizer,
        cursor,
        global_batch_samples=8,
        committed_tokens=payload.token_count,
        source_mix_commit=payload,
    )
    assert optimizer.steps == 1
    assert cursor.next_global_sample == 8
    assert cursor.committed_tokens == payload.token_count
    assert cursor.committed_tokens_by_source == payload.valid_tokens_by_source
    assert cursor.pending_global_batch == ()


def test_optimizer_step_preserves_legacy_cursor_commit_contract() -> None:
    call_order: list[str] = []

    class OrderedOptimizer:
        def step(self) -> None:
            call_order.append("step")

    cursor = _LegacyCursor(call_order)
    _optimizer_step_and_commit(
        OrderedOptimizer(),
        cursor,  # type: ignore[arg-type]
        global_batch_samples=8,
        committed_tokens=29,
        source_mix_commit=None,
    )

    assert call_order == ["step", "commit"]
    assert cursor.committed_tokens == 29


def test_muon_step_metrics_keep_nominal_and_shape_adjusted_lr_distinct() -> None:
    optimizer = SimpleNamespace(
        param_groups=[
            {
                "name": "adapters",
                "lr": 1e-4,
                "adjust_lr_fn": "match_rms_adamw",
                "params": [torch.empty((4096, 1024), device="meta")],
            },
            {
                "name": "scale",
                "lr": 3e-4,
                "params": [torch.empty((1,), device="meta")],
            },
        ]
    )
    applied = _named_learning_rates(optimizer)
    applied_adjusted = _named_adjusted_learning_rates(optimizer)
    optimizer.param_groups[0]["lr"] = 5e-5
    optimizer.param_groups[1]["lr"] = 1.5e-4
    following = _named_learning_rates(optimizer)
    following_adjusted = _named_adjusted_learning_rates(optimizer)

    metrics = _learning_rate_step_metrics(
        applied,
        following,
        applied_adjusted,
        following_adjusted,
    )

    assert metrics["lr/adapters"] == pytest.approx(1e-4)
    assert metrics["next_lr/adapters"] == pytest.approx(5e-5)
    assert metrics["lr_adjustment_factor/adapters"] == pytest.approx(12.8)
    assert metrics["lr_adjusted/adapters"] == pytest.approx(1.28e-3)
    assert metrics["next_lr_adjusted/adapters"] == pytest.approx(6.4e-4)
    assert "lr_adjusted/scale" not in metrics
