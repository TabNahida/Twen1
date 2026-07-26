from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from twen.config import (
    ArchitectureConfig,
    CheckpointConfig,
    DataConfig,
    LossConfig,
    ModelSource,
    OptimizerConfig,
    RuntimeConfig,
    SourcesConfig,
    TrainConfig,
)
from twen.data import QualityCooldownSummary
from twen.preflight import run_training_preflight


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _preflight_config(tmp_path: Path) -> TrainConfig:
    source = tmp_path / "model"
    manifest_sha = _write(source / "download-manifest.json", b"{}\n")
    _write(
        source / "config.json",
        json.dumps({"model_type": "qwen3_5_text"}).encode("utf-8"),
    )
    model = ModelSource(
        model_id="Qwen/test",
        revision="a" * 40,
        local_path=str(source),
        manifest_sha256=manifest_sha,
    )
    prepared = tmp_path / "prepared.json"
    kd = tmp_path / "kd.json"
    prepared_sha = _write(prepared, b'{"kind":"prepared"}\n')
    kd_sha = _write(kd, b'{"kind":"kd"}\n')
    layer_map = tmp_path / "layer-map.json"
    channel_map = tmp_path / "channel-map.json"
    adapters = tmp_path / "adapters.safetensors"
    _write(layer_map, b'{"student_to_donor":[0]}\n')
    _write(channel_map, b'{"layers":[]}\n')
    _write(adapters, b"ridge-initialization-v1")
    return TrainConfig(
        run_id="preflight-fingerprint",
        track="base",
        stage="dense-oracle",
        sources=SourcesConfig(model, model, model, model),
        architecture=ArchitectureConfig(
            layer_map_path=str(layer_map),
            channel_map_path=str(channel_map),
            adapter_init_path=str(adapters),
        ),
        data=DataConfig(
            manifest_path=str(prepared),
            manifest_sha256=prepared_sha,
            teacher_kd_manifest_path=str(kd),
            teacher_kd_manifest_sha256=kd_sha,
            max_sequence_length=8,
            global_batch_tokens=8,
        ),
        losses=LossConfig(),
        optimizer=OptimizerConfig(),
        checkpoint=CheckpointConfig(str(tmp_path / "run")),
        runtime=RuntimeConfig(),
    )


@pytest.mark.parametrize(
    "artifact_field",
    ["layer_map_path", "channel_map_path", "adapter_init_path"],
)
def test_calibration_artifact_content_changes_preflight_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_field: str,
) -> None:
    config = _preflight_config(tmp_path)
    dataset_fingerprint = "prepared-dataset-v1"

    import twen.data
    import twen.preflight

    monkeypatch.setattr(twen.preflight, "enforce_offline_environment", lambda: None)
    monkeypatch.setattr(twen.preflight, "twen_source_tree_sha256", lambda: "1" * 64)
    source_root = Path(config.sources.backbone.local_path)
    source_manifest = source_root / "download-manifest.json"
    monkeypatch.setattr(
        twen.preflight,
        "_check_source",
        lambda _name, _source: (
            source_root,
            source_manifest,
            {"model_type": "qwen3_5_text", "vocab_size": 128},
        ),
    )
    monkeypatch.setattr(twen.preflight, "_audit_architecture", lambda *_args: None)
    monkeypatch.setattr(
        twen.preflight,
        "_validate_calibration_contract",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        twen.data,
        "validate_prepared_corpus",
        lambda _: SimpleNamespace(
            sequence_length=config.data.max_sequence_length,
            dataset_fingerprint=dataset_fingerprint,
            tokenizer_sha256=config.sources.tokenizer.manifest_sha256,
            shards=(),
        ),
    )
    monkeypatch.setattr(
        twen.data,
        "validate_kd_corpus_manifest",
        lambda *_args, **_kwargs: SimpleNamespace(
            dataset_fingerprint=dataset_fingerprint,
            teacher_model_id=config.sources.teacher.model_id,
            teacher_revision=config.sources.teacher.revision,
            teacher_model_sha256=config.sources.teacher.manifest_sha256,
            tokenizer_sha256=config.sources.tokenizer.manifest_sha256,
            shards=(),
        ),
    )

    before_report = run_training_preflight(config, world_size=1)
    before = before_report.config_fingerprint
    assert before_report.source_tree_sha256 == "1" * 64
    monkeypatch.setattr(twen.preflight, "twen_source_tree_sha256", lambda: "2" * 64)
    source_changed = run_training_preflight(config, world_size=1)
    assert source_changed.source_tree_sha256 == "2" * 64
    assert source_changed.config_fingerprint != before
    monkeypatch.setattr(twen.preflight, "twen_source_tree_sha256", lambda: "1" * 64)
    config.runtime.activation_checkpoint_layer_count = 0
    config.runtime.hidden_alignment_activation_checkpoint_layer_count = 8
    config.runtime.dense_transfer_token_checkpoint = True
    config.runtime.dense_transfer_checkpoint_layer_count = 0
    config.runtime.hidden_alignment_dense_transfer_checkpoint_layer_count = 16
    selective = run_training_preflight(config, world_size=1)
    assert selective.config_fingerprint != before
    assert selective.activation_checkpoint_layer_count == 0
    assert selective.hidden_alignment_activation_checkpoint_layer_count == 8
    assert selective.activation_checkpoint_layer_indices == ()
    assert selective.hidden_alignment_activation_checkpoint_layer_indices == (
        0,
        3,
        7,
        10,
        13,
        16,
        20,
        23,
    )
    assert selective.dense_transfer_checkpoint_layer_count == 0
    assert selective.hidden_alignment_dense_transfer_checkpoint_layer_count == 16
    assert selective.dense_transfer_execution == "expanded"
    assert selective.dense_transfer_token_checkpoint_layer_indices == ()
    assert selective.hidden_alignment_dense_transfer_token_checkpoint_layer_indices == (
        1,
        2,
        4,
        5,
        6,
        8,
        9,
        11,
        12,
        14,
        15,
        17,
        18,
        19,
        21,
        22,
    )
    config.runtime.activation_checkpoint_layer_count = None
    artifact = Path(getattr(config.architecture, artifact_field))
    artifact.write_bytes(artifact.read_bytes() + b"\nchanged")
    after = run_training_preflight(config, world_size=1).config_fingerprint

    assert after != before


def test_prepared_text_preflight_never_opens_poisoned_kd_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _preflight_config(tmp_path)
    config.data.mode = "prepared-text"
    config.data.teacher_kd_manifest_path = None
    config.data.teacher_kd_manifest_sha256 = None
    config.losses.teacher_kd = 0.0
    config.losses.hidden_alignment = 0.0
    config.losses.anchor_kl = 0.0
    config.validate()
    prepared_fingerprint = "prepared-text-dataset-v1"

    import twen.data
    import twen.preflight

    monkeypatch.setattr(twen.preflight, "enforce_offline_environment", lambda: None)
    monkeypatch.setattr(twen.preflight, "twen_source_tree_sha256", lambda: "1" * 64)
    source_root = Path(config.sources.backbone.local_path)
    source_manifest = source_root / "download-manifest.json"
    monkeypatch.setattr(
        twen.preflight,
        "_check_source",
        lambda _name, _source: (
            source_root,
            source_manifest,
            {"model_type": "qwen3_5_text", "vocab_size": 128},
        ),
    )
    monkeypatch.setattr(twen.preflight, "_audit_architecture", lambda *_args: None)
    monkeypatch.setattr(twen.preflight, "_validate_calibration_contract", lambda *_args: [])
    monkeypatch.setattr(
        twen.data,
        "validate_prepared_corpus",
        lambda _path: SimpleNamespace(
            sequence_length=config.data.max_sequence_length,
            dataset_fingerprint=prepared_fingerprint,
            tokenizer_sha256=config.sources.tokenizer.manifest_sha256,
            shards=(),
        ),
    )

    def poison_kd(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("prepared-text preflight touched poisoned KD state")

    monkeypatch.setattr(twen.data, "validate_kd_corpus_manifest", poison_kd)
    monkeypatch.setattr(twen.data, "validate_kd_corpus_coverage", poison_kd)
    monkeypatch.setattr(twen.data, "read_kd_manifest", poison_kd)

    report = run_training_preflight(config, world_size=1)

    assert report.data_fingerprint == config.data.manifest_sha256
    assert str(Path(config.data.manifest_path).resolve()) in report.checked_paths
    assert all("kd" not in Path(path).name.lower() for path in report.checked_paths)


def test_preflight_authenticates_second_prepared_kd_cooldown_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _preflight_config(tmp_path)
    cooldown_prepared = tmp_path / "cooldown-prepared.json"
    cooldown_kd = tmp_path / "cooldown-kd.json"
    config.data.quality_cooldown_manifest_path = str(cooldown_prepared)
    config.data.quality_cooldown_manifest_sha256 = _write(
        cooldown_prepared, b'{"kind":"cooldown-prepared"}\n'
    )
    config.data.quality_cooldown_teacher_kd_manifest_path = str(cooldown_kd)
    config.data.quality_cooldown_teacher_kd_manifest_sha256 = _write(
        cooldown_kd, b'{"kind":"cooldown-kd"}\n'
    )
    config.data.quality_cooldown_start_tokens = 50_000_000
    primary_fingerprint = "a" * 64
    cooldown_fingerprint = "b" * 64
    primary_prepared = SimpleNamespace(
        sequence_length=config.data.max_sequence_length,
        dataset_fingerprint=primary_fingerprint,
        tokenizer_sha256=config.sources.tokenizer.manifest_sha256,
        shards=(),
    )
    selected_prepared = SimpleNamespace(
        sequence_length=config.data.max_sequence_length,
        dataset_fingerprint=cooldown_fingerprint,
        tokenizer_sha256=config.sources.tokenizer.manifest_sha256,
        shards=(),
    )

    def kd(fingerprint: str) -> SimpleNamespace:
        return SimpleNamespace(
            dataset_fingerprint=fingerprint,
            teacher_model_id=config.sources.teacher.model_id,
            teacher_revision=config.sources.teacher.revision,
            teacher_model_sha256=config.sources.teacher.manifest_sha256,
            tokenizer_sha256=config.sources.tokenizer.manifest_sha256,
            shards=(),
        )

    import twen.data
    import twen.preflight

    monkeypatch.setattr(twen.preflight, "enforce_offline_environment", lambda: None)
    monkeypatch.setattr(twen.preflight, "twen_source_tree_sha256", lambda: "1" * 64)
    source_root = Path(config.sources.backbone.local_path)
    source_manifest = source_root / "download-manifest.json"
    monkeypatch.setattr(
        twen.preflight,
        "_check_source",
        lambda _name, _source: (
            source_root,
            source_manifest,
            {"model_type": "qwen3_5_text", "vocab_size": 128},
        ),
    )
    monkeypatch.setattr(twen.preflight, "_audit_architecture", lambda *_args: None)
    monkeypatch.setattr(twen.preflight, "_validate_calibration_contract", lambda *_args: [])
    monkeypatch.setattr(
        twen.data,
        "validate_prepared_corpus",
        lambda path: (
            selected_prepared
            if Path(path).resolve() == cooldown_prepared.resolve()
            else primary_prepared
        ),
    )
    monkeypatch.setattr(
        twen.data,
        "validate_kd_corpus_manifest",
        lambda path, **_kwargs: (
            kd(cooldown_fingerprint)
            if Path(path).resolve() == cooldown_kd.resolve()
            else kd(primary_fingerprint)
        ),
    )
    monkeypatch.setattr(twen.data, "validate_kd_corpus_coverage", lambda *_args: None)
    monkeypatch.setattr(
        twen.data,
        "validate_quality_cooldown_subset",
        lambda *_args, **_kwargs: QualityCooldownSummary(
            selection_policy_id="quality-v1",
            parent_dataset_fingerprint=primary_fingerprint,
            cooldown_dataset_fingerprint=cooldown_fingerprint,
            selected_shard_ids=("shard-1",),
            source_mix_token_counts=(("math", 50_000_000),),
            sequence_count=12_500,
            token_count=50_000_000,
        ),
    )

    report = run_training_preflight(config, world_size=1)

    assert report.quality_cooldown_enabled is True
    assert report.quality_cooldown_start_tokens == 50_000_000
    assert report.quality_cooldown_dataset_fingerprint == cooldown_fingerprint
    assert report.quality_cooldown_selected_shard_ids == ("shard-1",)
    assert report.quality_cooldown_source_mix_token_counts == (("math", 50_000_000),)
    assert report.data_fingerprint not in {
        config.data.manifest_sha256,
        config.data.quality_cooldown_manifest_sha256,
    }
    assert str(cooldown_prepared.resolve()) in report.checked_paths
    assert str(cooldown_kd.resolve()) in report.checked_paths
