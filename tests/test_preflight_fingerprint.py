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
from twen.data import (
    AuthenticatedSourceMap,
    AuthenticatedSourceShard,
    QualityCooldownSummary,
)
from twen.preflight import (
    TrainingPreflightError,
    _validate_inference_kd_shard_manifests,
    _validate_no_reuse_capacity,
    _validate_phase_disjointness_attestation,
    run_inference_preflight,
    run_training_preflight,
)
from twen.source_identity import twen_source_tree_sha256


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_training_and_inference_preflight_dispatch_distinct_data_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, int | None, bool]] = []
    sentinel = object()

    def fake_run(
        config: object,
        *,
        world_size: int | None,
        inference_only: bool,
    ) -> object:
        calls.append((config, world_size, inference_only))
        return sentinel

    monkeypatch.setattr("twen.preflight._run_preflight", fake_run)
    config = object()
    assert run_training_preflight(config, world_size=2) is sentinel
    assert run_inference_preflight(config, world_size=1) is sentinel
    assert calls == [(config, 2, False), (config, 1, True)]


def test_inference_kd_metadata_authentication_skips_only_tensor_checksum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_manifest = tmp_path / "manifest.json"
    shard = tmp_path / "shard-000"
    local_manifest = shard / "kd_manifest.json"
    local_sha = _write(local_manifest, b'{"kind":"teacher-kd-shard"}\n')
    entry = SimpleNamespace(
        path=shard.name,
        manifest_sha256=local_sha,
        source_shard_id="source-000",
        source_tensors_sha256="1" * 64,
        global_sample_start=0,
        global_sample_end=2,
        global_token_start=0,
        global_token_end=5,
        sequence_count=2,
        token_count=5,
        tensors_sha256="2" * 64,
    )
    corpus = SimpleNamespace(
        teacher_model_id="Qwen/test",
        teacher_revision="a" * 40,
        teacher_model_sha256="3" * 64,
        generator_source_sha256="4" * 64,
        tokenizer_sha256="5" * 64,
        dataset_fingerprint="dataset",
        shards=(entry,),
    )
    shard_manifest = SimpleNamespace(
        teacher_model_id=corpus.teacher_model_id,
        teacher_revision=corpus.teacher_revision,
        teacher_model_sha256=corpus.teacher_model_sha256,
        generator_source_sha256=corpus.generator_source_sha256,
        tokenizer_sha256=corpus.tokenizer_sha256,
        dataset_fingerprint=corpus.dataset_fingerprint,
        source_shard_id=entry.source_shard_id,
        source_tensors_sha256=entry.source_tensors_sha256,
        global_sample_start=entry.global_sample_start,
        global_sample_end=entry.global_sample_end,
        global_token_start=entry.global_token_start,
        global_token_end=entry.global_token_end,
        sequence_count=entry.sequence_count,
        token_count=entry.token_count,
        tensors_sha256=entry.tensors_sha256,
    )
    calls: list[tuple[Path, float | None, bool]] = []

    def validate_shard(
        path: Path,
        *,
        expected_temperature: float | None = None,
        verify_checksum: bool = True,
    ) -> SimpleNamespace:
        calls.append((Path(path), expected_temperature, verify_checksum))
        return shard_manifest

    monkeypatch.setattr(
        "twen.data.teacher_kd.validate_kd_shard",
        validate_shard,
    )
    assert _validate_inference_kd_shard_manifests(
        corpus_manifest,
        corpus,
        expected_temperature=2.0,
    ) == (shard_manifest,)
    assert calls == [(shard.resolve(), 2.0, False)]

    local_manifest.write_bytes(local_manifest.read_bytes() + b"tamper")
    with pytest.raises(TrainingPreflightError, match="manifest hash mismatch"):
        _validate_inference_kd_shard_manifests(
            corpus_manifest,
            corpus,
            expected_temperature=2.0,
        )
    assert calls == [(shard.resolve(), 2.0, False)]


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


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_phase_disjointness_preflight_binds_both_prepared_identities(
    tmp_path: Path,
) -> None:
    primary_manifest = tmp_path / "primary" / "manifest.json"
    cooldown_manifest = tmp_path / "cooldown" / "manifest.json"
    primary_sha = _write(primary_manifest, b'{"kind":"prepared-primary"}\n')
    cooldown_sha = _write(cooldown_manifest, b'{"kind":"prepared-cooldown"}\n')
    primary_prepared = SimpleNamespace(dataset_fingerprint="a" * 64)
    cooldown_prepared = SimpleNamespace(dataset_fingerprint="b" * 64)
    primary_map = SimpleNamespace(fingerprint="c" * 64)
    cooldown_map = SimpleNamespace(fingerprint="d" * 64)
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "twen_v4_phase_disjointness_attestation",
        "scanner_source_sha256": hashlib.sha256(
            (Path(__file__).parents[1] / "scripts" / "attest_v4_phase_disjointness.py").read_bytes()
        ).hexdigest(),
        "scanner_source_tree_sha256": twen_source_tree_sha256(),
        "primary": {
            "prepared": {
                "manifest_path": str(primary_manifest.resolve()),
                "manifest_sha256": primary_sha,
                "dataset_fingerprint": primary_prepared.dataset_fingerprint,
                "source_map_sha256": primary_map.fingerprint,
            }
        },
        "cooldown": {
            "prepared": {
                "manifest_path": str(cooldown_manifest.resolve()),
                "manifest_sha256": cooldown_sha,
                "dataset_fingerprint": cooldown_prepared.dataset_fingerprint,
                "source_map_sha256": cooldown_map.fingerprint,
            }
        },
        "scope": "authenticated_train_inventories_only",
        "metrics": {
            "stable_id_exact_matches": 0,
            "normalized_text_exact_matches": 0,
            "near_duplicate_matches": 0,
        },
        "stores_raw_text": False,
        "gates": {
            name: {
                "algorithm": algorithm,
                "matches": 0,
                "passed": True,
                **({"estimated_jaccard_threshold": 0.8} if name == "near_duplicate" else {}),
            }
            for name, algorithm in {
                "stable_id_exact": ("source-scoped-authenticated-stable-id-intersection-v1"),
                "normalized_text_exact": ("unicode-nfkc-whitespace-sha256-intersection-v1"),
                "near_duplicate": ("lexical-5gram-one-permutation-minhash-lsh-v1"),
            }.items()
        },
        "passed": True,
    }
    payload["attestation_fingerprint"] = _canonical_sha256(payload)
    attestation = tmp_path / "phase" / "attestation.json"
    attestation.parent.mkdir()
    attestation.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    complete = {
        "schema_version": 1,
        "kind": "twen_v4_phase_disjointness_complete",
        "attestation": attestation.name,
        "attestation_sha256": hashlib.sha256(attestation.read_bytes()).hexdigest(),
        "attestation_fingerprint": payload["attestation_fingerprint"],
        "passed": True,
    }
    (attestation.parent / "COMPLETE").write_text(
        json.dumps(complete, sort_keys=True),
        encoding="utf-8",
    )

    _validate_phase_disjointness_attestation(
        attestation,
        primary_manifest=primary_manifest,
        primary_prepared=primary_prepared,
        primary_source_map=primary_map,
        cooldown_manifest=cooldown_manifest,
        cooldown_prepared=cooldown_prepared,
        cooldown_source_map=cooldown_map,
    )

    with pytest.raises(
        TrainingPreflightError,
        match="cooldown source_map_sha256 mismatch",
    ):
        _validate_phase_disjointness_attestation(
            attestation,
            primary_manifest=primary_manifest,
            primary_prepared=primary_prepared,
            primary_source_map=primary_map,
            cooldown_manifest=cooldown_manifest,
            cooldown_prepared=cooldown_prepared,
            cooldown_source_map=SimpleNamespace(fingerprint="f" * 64),
        )

    gates = payload["gates"]
    assert isinstance(gates, dict)
    near_duplicate = gates["near_duplicate"]
    assert isinstance(near_duplicate, dict)
    near_duplicate["estimated_jaccard_threshold"] = 1.0
    payload.pop("attestation_fingerprint")
    payload["attestation_fingerprint"] = _canonical_sha256(payload)
    attestation.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    complete["attestation_sha256"] = hashlib.sha256(attestation.read_bytes()).hexdigest()
    complete["attestation_fingerprint"] = payload["attestation_fingerprint"]
    (attestation.parent / "COMPLETE").write_text(
        json.dumps(complete, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(
        TrainingPreflightError,
        match=r"near_duplicate.*did not pass",
    ):
        _validate_phase_disjointness_attestation(
            attestation,
            primary_manifest=primary_manifest,
            primary_prepared=primary_prepared,
            primary_source_map=primary_map,
            cooldown_manifest=cooldown_manifest,
            cooldown_prepared=cooldown_prepared,
            cooldown_source_map=cooldown_map,
        )

    near_duplicate["estimated_jaccard_threshold"] = 0.8
    payload["scanner_source_tree_sha256"] = "0" * 64
    payload.pop("attestation_fingerprint")
    payload["attestation_fingerprint"] = _canonical_sha256(payload)
    attestation.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    complete["attestation_sha256"] = hashlib.sha256(attestation.read_bytes()).hexdigest()
    complete["attestation_fingerprint"] = payload["attestation_fingerprint"]
    (attestation.parent / "COMPLETE").write_text(
        json.dumps(complete, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(
        TrainingPreflightError,
        match="scanner source tree changed",
    ):
        _validate_phase_disjointness_attestation(
            attestation,
            primary_manifest=primary_manifest,
            primary_prepared=primary_prepared,
            primary_source_map=primary_map,
            cooldown_manifest=cooldown_manifest,
            cooldown_prepared=cooldown_prepared,
            cooldown_source_map=cooldown_map,
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


def test_prepared_text_cooldown_preflight_authenticates_two_source_maps_without_kd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _preflight_config(tmp_path)
    cooldown_manifest = tmp_path / "cooldown-prepared.json"
    config.data.mode = "prepared-text"
    config.data.teacher_kd_manifest_path = None
    config.data.teacher_kd_manifest_sha256 = None
    config.data.quality_cooldown_manifest_path = str(cooldown_manifest)
    config.data.quality_cooldown_manifest_sha256 = _write(
        cooldown_manifest,
        b'{"kind":"prepared-text-cooldown"}\n',
    )
    config.data.quality_cooldown_start_tokens = 450_000_000
    config.losses.teacher_kd = 0.0
    config.losses.hidden_alignment = 0.0
    config.losses.anchor_kl = 0.0

    primary_fingerprint = "a" * 64
    cooldown_fingerprint = "b" * 64
    primary_entry = SimpleNamespace(
        shard_id="primary-shard",
        sequence_count=57_500_000,
        token_count=460_000_000,
    )
    cooldown_entry = SimpleNamespace(
        shard_id="cooldown-shard",
        sequence_count=7_500_000,
        token_count=60_000_000,
    )

    def prepared(
        fingerprint: str,
        entry: SimpleNamespace,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            sequence_length=config.data.max_sequence_length,
            sequence_count=entry.sequence_count,
            token_count=entry.token_count,
            dataset_fingerprint=fingerprint,
            tokenizer_sha256=config.sources.tokenizer.manifest_sha256,
            shards=(entry,),
            lineage={
                "kind": "authenticated_extracted_corpus",
                "ready_for_training": True,
                "research_only": False,
                "pending_audits": [],
            },
        )

    primary_prepared = prepared(primary_fingerprint, primary_entry)
    cooldown_prepared = prepared(cooldown_fingerprint, cooldown_entry)
    primary_map = AuthenticatedSourceMap(
        prepared_dataset_fingerprint=primary_fingerprint,
        extracted_manifest_sha256="c" * 64,
        sequence_length=config.data.max_sequence_length,
        shards=(
            AuthenticatedSourceShard(
                source_id="primary",
                shard_id=primary_entry.shard_id,
                sequence_count=primary_entry.sequence_count,
                global_sample_start=0,
                output_path="primary.jsonl",
                output_sha256="d" * 64,
            ),
        ),
        mix_basis_points=(("primary", 10_000),),
    )
    cooldown_map = AuthenticatedSourceMap(
        prepared_dataset_fingerprint=cooldown_fingerprint,
        extracted_manifest_sha256="e" * 64,
        sequence_length=config.data.max_sequence_length,
        shards=(
            AuthenticatedSourceShard(
                source_id="quality",
                shard_id=cooldown_entry.shard_id,
                sequence_count=cooldown_entry.sequence_count,
                global_sample_start=0,
                output_path="quality.jsonl",
                output_sha256="f" * 64,
            ),
        ),
        mix_basis_points=(("quality", 10_000),),
    )
    config.data.source_mix_algorithm = "token-deficit-corrected-source-mix-bp-v2"
    config.data.source_map_sha256 = primary_map.fingerprint
    config.data.source_mix_basis_points = {"primary": 10_000}
    config.validate()

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
        lambda path: (
            cooldown_prepared
            if Path(path).resolve() == cooldown_manifest.resolve()
            else primary_prepared
        ),
    )
    monkeypatch.setattr(
        AuthenticatedSourceMap,
        "from_prepared_manifest",
        classmethod(
            lambda _cls, value: (
                cooldown_map if value.dataset_fingerprint == cooldown_fingerprint else primary_map
            )
        ),
    )

    def poison_kd(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("prepared-text cooldown preflight touched KD")

    monkeypatch.setattr(twen.data, "validate_kd_corpus_manifest", poison_kd)
    monkeypatch.setattr(twen.data, "validate_kd_corpus_coverage", poison_kd)
    monkeypatch.setattr(twen.data, "read_kd_manifest", poison_kd)

    report = run_training_preflight(config, world_size=1)

    assert report.quality_cooldown_enabled is True
    assert report.quality_cooldown_source_mix_enabled is True
    assert report.quality_cooldown_source_map_sha256 == cooldown_map.fingerprint
    assert report.quality_cooldown_source_mix_basis_points == (("quality", 10_000),)
    assert report.quality_cooldown_source_mix_token_counts == (("quality", 60_000_000),)
    assert report.quality_cooldown_source_map_payload_json is not None
    assert str(cooldown_manifest.resolve()) in report.checked_paths
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
        "validate_prepared_corpus_for_inference",
        lambda path, **_kwargs: (
            selected_prepared
            if Path(path).resolve() == cooldown_prepared.resolve()
            else primary_prepared
        ),
    )
    kd_verification_calls: list[tuple[Path, bool]] = []

    def validate_kd(path: Path, **kwargs: object) -> SimpleNamespace:
        kd_verification_calls.append(
            (Path(path).resolve(), bool(kwargs["verify_shards"]))
        )
        return (
            kd(cooldown_fingerprint)
            if Path(path).resolve() == cooldown_kd.resolve()
            else kd(primary_fingerprint)
        )

    monkeypatch.setattr(
        twen.data,
        "validate_kd_corpus_manifest",
        validate_kd,
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
    assert kd_verification_calls == [
        (Path(config.data.teacher_kd_manifest_path).resolve(), True),
        (cooldown_kd.resolve(), True),
    ]

    kd_verification_calls.clear()
    inference_report = run_inference_preflight(config, world_size=1)
    assert inference_report.quality_cooldown_enabled is True
    assert kd_verification_calls == [
        (Path(config.data.teacher_kd_manifest_path).resolve(), False),
        (cooldown_kd.resolve(), False),
    ]


def test_no_reuse_capacity_rejects_nominally_sufficient_smoke_tail_batch() -> None:
    smoke = SimpleNamespace(
        token_count=16_013_672,
        sequence_count=3_923,
        shards=(),
    )

    with pytest.raises(
        TrainingPreflightError,
        match=r"complete tail optimizer batch.*requires_tokens=16262144",
    ):
        _validate_no_reuse_capacity(
            smoke,
            label="primary",
            phase_budget_tokens=16_000_000,
            global_batch_tokens=262_144,
            sequence_length=4_096,
            seed=73,
        )


def test_no_reuse_capacity_is_fail_closed_per_source() -> None:
    source_map = AuthenticatedSourceMap(
        prepared_dataset_fingerprint="1" * 64,
        extracted_manifest_sha256="2" * 64,
        sequence_length=4,
        shards=(
            AuthenticatedSourceShard(
                source_id="alpha",
                shard_id="alpha",
                sequence_count=4,
                global_sample_start=0,
                output_path="alpha.jsonl",
                output_sha256="3" * 64,
            ),
            AuthenticatedSourceShard(
                source_id="beta",
                shard_id="beta",
                sequence_count=4,
                global_sample_start=4,
                output_path="beta.jsonl",
                output_sha256="4" * 64,
            ),
        ),
        mix_basis_points=(("alpha", 5_000), ("beta", 5_000)),
    )
    prepared = SimpleNamespace(
        token_count=28,
        sequence_count=8,
        shards=(
            SimpleNamespace(
                shard_id="alpha",
                sequence_count=4,
                token_count=15,
            ),
            SimpleNamespace(
                shard_id="beta",
                sequence_count=4,
                token_count=13,
            ),
        ),
    )

    with pytest.raises(
        TrainingPreflightError,
        match=r"source 'beta' would wrap.*requires_tokens=14",
    ):
        _validate_no_reuse_capacity(
            prepared,
            label="primary",
            phase_budget_tokens=20,
            global_batch_tokens=8,
            sequence_length=4,
            source_map=source_map,
            source_mix_basis_points={"alpha": 5_000, "beta": 5_000},
            seed=73,
        )


def test_no_reuse_capacity_rejects_non_dense_shard_ledger() -> None:
    source_map = AuthenticatedSourceMap(
        prepared_dataset_fingerprint="1" * 64,
        extracted_manifest_sha256="2" * 64,
        sequence_length=4,
        shards=(
            AuthenticatedSourceShard(
                source_id="only",
                shard_id="only",
                sequence_count=8,
                global_sample_start=0,
                output_path="only.jsonl",
                output_sha256="3" * 64,
            ),
        ),
        mix_basis_points=(("only", 10_000),),
    )
    prepared = SimpleNamespace(
        token_count=20,
        sequence_count=8,
        shards=(
            SimpleNamespace(
                shard_id="only",
                sequence_count=8,
                token_count=20,
            ),
        ),
    )

    with pytest.raises(
        TrainingPreflightError,
        match=r"shard 'only' violates dense packing",
    ):
        _validate_no_reuse_capacity(
            prepared,
            label="primary",
            phase_budget_tokens=12,
            global_batch_tokens=8,
            sequence_length=4,
            source_map=source_map,
            source_mix_basis_points={"only": 10_000},
            seed=73,
        )
