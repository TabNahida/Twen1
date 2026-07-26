from __future__ import annotations

import copy
import math

import pytest

from twen.config import (
    ArchitectureConfig,
    CheckpointConfig,
    ConfigError,
    DataConfig,
    LossConfig,
    ModelSource,
    OptimizerConfig,
    RuntimeConfig,
    SourcesConfig,
    TrainConfig,
)


def _source(name: str) -> ModelSource:
    return ModelSource(name, "a" * 40, f"/models/{name}", "b" * 64)


def make_config() -> TrainConfig:
    return TrainConfig(
        run_id="test",
        track="base",
        stage="dense-oracle",
        sources=SourcesConfig(
            _source("small"),
            _source("large"),
            _source("large"),
            _source("small"),
        ),
        architecture=ArchitectureConfig(),
        data=DataConfig("data.json", "c" * 64, "kd.json", "d" * 64),
        losses=LossConfig(),
        optimizer=OptimizerConfig(),
        checkpoint=CheckpointConfig("runs/test"),
        runtime=RuntimeConfig(),
    )


def test_runtime_logging_change_does_not_change_fingerprint() -> None:
    first = make_config()
    second = copy.deepcopy(first)
    second.runtime.log_every_steps = 50
    second.runtime.profile = True
    second.runtime.profile_active_steps = 9
    second.runtime.expandable_segments = False
    second.data.num_workers = 16
    assert first.fingerprint() == second.fingerprint()


@pytest.mark.parametrize("field", ["bf16", "offline"])
def test_v1_requires_bf16_offline_training(field: str) -> None:
    config = make_config()
    setattr(config.runtime, field, False)
    with pytest.raises(ConfigError, match=field):
        config.validate()


@pytest.mark.parametrize(
    ("role", "reference"),
    [("tokenizer", "backbone"), ("teacher", "donor")],
)
def test_v1_source_roles_require_identical_model_lineage(role: str, reference: str) -> None:
    config = make_config()
    source = getattr(config.sources, role)
    expected = getattr(config.sources, reference)
    source.model_id = f"{expected.model_id}-wrong"
    with pytest.raises(ConfigError, match=rf"sources\.{role}\.model_id"):
        config.validate()


def test_numerical_runtime_choices_are_resume_critical() -> None:
    first = make_config()
    chunked = copy.deepcopy(first)
    chunked.runtime.loss_chunk_tokens = 64
    unfused = copy.deepcopy(first)
    unfused.runtime.fused_adamw = False
    eager_loss = copy.deepcopy(first)
    eager_loss.runtime.compile_streaming_loss = False
    assert first.fingerprint() != chunked.fingerprint()
    assert first.fingerprint() != unfused.fingerprint()
    assert first.fingerprint() != eager_loss.fingerprint()


def test_quality_cooldown_default_is_legacy_compatible_and_enabled_is_critical() -> None:
    legacy = make_config()
    canonical_data = legacy.canonical_dict()["data"]
    assert not any(name.startswith("quality_cooldown_") for name in canonical_data)

    cooldown = copy.deepcopy(legacy)
    cooldown.data.quality_cooldown_manifest_path = "cooldown-prepared.json"
    cooldown.data.quality_cooldown_manifest_sha256 = "1" * 64
    cooldown.data.quality_cooldown_teacher_kd_manifest_path = "cooldown-kd.json"
    cooldown.data.quality_cooldown_teacher_kd_manifest_sha256 = "2" * 64
    cooldown.data.quality_cooldown_start_tokens = 50_000_000
    cooldown.validate()

    assert cooldown.data.quality_cooldown_enabled()
    assert cooldown.fingerprint() != legacy.fingerprint()
    changed_start = copy.deepcopy(cooldown)
    changed_start.data.quality_cooldown_start_tokens += 1
    assert changed_start.fingerprint() != cooldown.fingerprint()


def test_teacher_kd_data_mode_default_preserves_legacy_canonical_and_fingerprint() -> None:
    legacy = make_config()

    assert legacy.data.mode == "teacher-kd"
    assert "mode" not in legacy.canonical_dict()["data"]
    assert (
        legacy.fingerprint() == "02b9a50c1ef75451417e5da04461b68692689a673f9aa03042bf5a359719c8d6"
    )


def test_prepared_text_data_mode_omits_kd_identity_and_is_resume_critical() -> None:
    legacy = make_config()
    prepared_text = copy.deepcopy(legacy)
    prepared_text.data.mode = "prepared-text"
    prepared_text.data.teacher_kd_manifest_path = None
    prepared_text.data.teacher_kd_manifest_sha256 = None
    prepared_text.losses.teacher_kd = 0.0
    prepared_text.losses.hidden_alignment = 0.0
    prepared_text.losses.anchor_kl = 0.0

    prepared_text.validate()
    canonical = prepared_text.canonical_dict()["data"]

    assert canonical["mode"] == "prepared-text"
    assert "teacher_kd_manifest_path" not in canonical
    assert "teacher_kd_manifest_sha256" not in canonical
    assert "teacher_top_k" not in canonical
    assert prepared_text.fingerprint() != legacy.fingerprint()


def test_prepared_text_quality_cooldown_uses_only_prepared_identity() -> None:
    config = _source_mixed_prepared_text_config()
    baseline = copy.deepcopy(config)
    config.data.quality_cooldown_manifest_path = "cooldown-prepared.json"
    config.data.quality_cooldown_manifest_sha256 = "4" * 64
    config.data.quality_cooldown_start_tokens = 50_000_000

    config.validate()

    canonical = config.canonical_dict()["data"]
    assert canonical["quality_cooldown_manifest_path"] == "cooldown-prepared.json"
    assert canonical["quality_cooldown_manifest_sha256"] == "4" * 64
    assert canonical["quality_cooldown_start_tokens"] == 50_000_000
    assert "quality_cooldown_teacher_kd_manifest_path" not in canonical
    assert "quality_cooldown_teacher_kd_manifest_sha256" not in canonical
    assert config.fingerprint() != baseline.fingerprint()

    unexpected_kd = copy.deepcopy(config)
    unexpected_kd.data.quality_cooldown_teacher_kd_manifest_path = "kd.json"
    unexpected_kd.data.quality_cooldown_teacher_kd_manifest_sha256 = "5" * 64
    with pytest.raises(ConfigError, match="must omit KD"):
        unexpected_kd.validate()

    incomplete = _source_mixed_prepared_text_config()
    incomplete.data.quality_cooldown_manifest_path = "cooldown-prepared.json"
    with pytest.raises(ConfigError, match="prepared path, SHA256, and start"):
        incomplete.validate()


def test_allow_corpus_reuse_default_is_legacy_compatible_and_false_is_critical() -> None:
    legacy = make_config()
    assert legacy.data.allow_corpus_reuse is True
    assert "allow_corpus_reuse" not in legacy.canonical_dict()["data"]
    assert (
        legacy.fingerprint() == "02b9a50c1ef75451417e5da04461b68692689a673f9aa03042bf5a359719c8d6"
    )

    prepared_text = _source_mixed_prepared_text_config()
    no_reuse = copy.deepcopy(prepared_text)
    no_reuse.data.allow_corpus_reuse = False
    no_reuse.validate()
    assert no_reuse.canonical_dict()["data"]["allow_corpus_reuse"] is False
    assert no_reuse.fingerprint() != prepared_text.fingerprint()

    unsupported_kd = copy.deepcopy(legacy)
    unsupported_kd.data.allow_corpus_reuse = False
    with pytest.raises(ConfigError, match=r"requires data.mode='prepared-text'"):
        unsupported_kd.validate()

    invalid = copy.deepcopy(legacy)
    invalid.data.allow_corpus_reuse = 0  # type: ignore[assignment]
    with pytest.raises(ConfigError, match="allow_corpus_reuse"):
        invalid.validate()


def test_no_reuse_prepared_text_cooldown_requires_resume_critical_disjointness() -> None:
    config = _source_mixed_prepared_text_config()
    config.data.quality_cooldown_manifest_path = "cooldown-prepared.json"
    config.data.quality_cooldown_manifest_sha256 = "4" * 64
    config.data.quality_cooldown_start_tokens = 50_000_000
    config.data.allow_corpus_reuse = False

    with pytest.raises(ConfigError, match="phase disjointness attestation"):
        config.validate()

    config.data.phase_disjointness_attestation_path = "phase-attestation.json"
    config.data.phase_disjointness_attestation_sha256 = "5" * 64
    config.validate()
    canonical = config.canonical_dict()["data"]
    assert canonical["phase_disjointness_attestation_path"] == "phase-attestation.json"
    assert canonical["phase_disjointness_attestation_sha256"] == "5" * 64

    changed = copy.deepcopy(config)
    changed.data.phase_disjointness_attestation_sha256 = "6" * 64
    changed.validate()
    assert changed.fingerprint() != config.fingerprint()


def test_phase_disjointness_does_not_extend_teacher_kd_cooldown_contract() -> None:
    config = make_config()
    config.data.quality_cooldown_manifest_path = "cooldown-prepared.json"
    config.data.quality_cooldown_manifest_sha256 = "4" * 64
    config.data.quality_cooldown_teacher_kd_manifest_path = "cooldown-kd.json"
    config.data.quality_cooldown_teacher_kd_manifest_sha256 = "5" * 64
    config.data.quality_cooldown_start_tokens = 50_000_000
    config.validate()
    assert "phase_disjointness_attestation_path" not in (config.canonical_dict()["data"])

    configured = copy.deepcopy(config)
    configured.data.phase_disjointness_attestation_path = "phase.json"
    configured.data.phase_disjointness_attestation_sha256 = "6" * 64
    with pytest.raises(ConfigError, match="only valid for prepared-text"):
        configured.validate()


def _source_mixed_prepared_text_config() -> TrainConfig:
    config = make_config()
    config.data.mode = "prepared-text"
    config.data.teacher_kd_manifest_path = None
    config.data.teacher_kd_manifest_sha256 = None
    config.data.source_mix_algorithm = "token-deficit-corrected-source-mix-bp-v2"
    config.data.source_map_sha256 = "3" * 64
    config.data.source_mix_basis_points = {"alpha": 6_000, "beta": 4_000}
    config.losses.teacher_kd = 0.0
    config.losses.hidden_alignment = 0.0
    config.losses.anchor_kl = 0.0
    return config


def test_source_mix_weight_override_is_explicit_and_resume_critical() -> None:
    inherited = _source_mixed_prepared_text_config()
    inherited.validate()
    overridden = copy.deepcopy(inherited)
    overridden.data.source_mix_allow_weight_override = True
    overridden.validate()

    assert "source_mix_allow_weight_override" in overridden.canonical_dict()["data"]
    assert overridden.fingerprint() != inherited.fingerprint()


def test_source_mix_weight_override_flag_is_strictly_scoped_and_typed() -> None:
    missing_mix = make_config()
    missing_mix.data.source_mix_allow_weight_override = True
    with pytest.raises(ConfigError, match="requires enabled source mixing"):
        missing_mix.validate()

    invalid = _source_mixed_prepared_text_config()
    invalid.data.source_mix_allow_weight_override = 1  # type: ignore[assignment]
    with pytest.raises(ConfigError, match="must be a boolean"):
        invalid.validate()


@pytest.mark.parametrize("mode", ["text", "kd", "", None])
def test_data_mode_rejects_unknown_values(mode: object) -> None:
    config = make_config()
    config.data.mode = mode  # type: ignore[assignment]

    with pytest.raises(ConfigError, match=r"data\.mode"):
        config.validate()


def test_data_mode_requires_exactly_the_matching_manifest_contract() -> None:
    missing_kd = make_config()
    missing_kd.data.teacher_kd_manifest_path = None
    missing_kd.data.teacher_kd_manifest_sha256 = None
    with pytest.raises(ConfigError, match="teacher-kd"):
        missing_kd.validate()

    unexpected_kd = make_config()
    unexpected_kd.data.mode = "prepared-text"
    with pytest.raises(ConfigError, match="must omit teacher KD"):
        unexpected_kd.validate()


def test_prepared_text_data_mode_rejects_teacher_side_losses() -> None:
    config = make_config()
    config.data.mode = "prepared-text"
    config.data.teacher_kd_manifest_path = None
    config.data.teacher_kd_manifest_sha256 = None

    with pytest.raises(ConfigError, match="zero teacher-side losses"):
        config.validate()


def test_quality_cooldown_requires_complete_independent_manifest_contract() -> None:
    config = make_config()
    config.data.quality_cooldown_manifest_path = "cooldown-prepared.json"
    with pytest.raises(ConfigError, match="requires prepared/KD paths"):
        config.validate()

    config.data.quality_cooldown_manifest_sha256 = config.data.manifest_sha256
    config.data.quality_cooldown_teacher_kd_manifest_path = "cooldown-kd.json"
    config.data.quality_cooldown_teacher_kd_manifest_sha256 = "2" * 64
    config.data.quality_cooldown_start_tokens = 50_000_000
    with pytest.raises(ConfigError, match="independent prepared and KD"):
        config.validate()


def test_quality_cooldown_start_must_precede_training_budget() -> None:
    config = make_config()
    config.data.quality_cooldown_manifest_path = "cooldown-prepared.json"
    config.data.quality_cooldown_manifest_sha256 = "1" * 64
    config.data.quality_cooldown_teacher_kd_manifest_path = "cooldown-kd.json"
    config.data.quality_cooldown_teacher_kd_manifest_sha256 = "2" * 64
    config.data.quality_cooldown_start_tokens = config.optimizer.max_tokens
    with pytest.raises(ConfigError, match=r"below optimizer\.max_tokens"):
        config.validate()


def test_selective_activation_checkpoint_count_is_resume_critical() -> None:
    legacy = make_config()
    selective = copy.deepcopy(legacy)
    selective.runtime.activation_checkpoint_layer_count = 4

    selective.validate()
    assert legacy.runtime.activation_checkpoint_layer_count is None
    assert legacy.runtime.hidden_alignment_activation_checkpoint_layer_count is None
    assert legacy.runtime.dense_transfer_checkpoint_layer_count is None
    assert legacy.runtime.hidden_alignment_dense_transfer_checkpoint_layer_count is None
    assert (
        legacy.fingerprint() == "02b9a50c1ef75451417e5da04461b68692689a673f9aa03042bf5a359719c8d6"
    )
    runtime = legacy.canonical_dict()["runtime"]
    assert "hidden_alignment_activation_checkpoint_layer_count" not in runtime
    assert "dense_transfer_checkpoint_layer_count" not in runtime
    assert "hidden_alignment_dense_transfer_checkpoint_layer_count" not in runtime
    assert legacy.fingerprint() != selective.fingerprint()


def test_dense_transfer_folded_execution_is_validated_and_resume_critical() -> None:
    legacy = make_config()
    folded = copy.deepcopy(legacy)
    folded.runtime.dense_transfer_execution = "differentiable_folded"
    folded.runtime.dense_transfer_token_checkpoint = True

    folded.validate()
    assert "dense_transfer_execution" not in legacy.canonical_dict()["runtime"]
    assert "dense_transfer_token_checkpoint" not in legacy.canonical_dict()["runtime"]
    assert folded.canonical_dict()["runtime"]["dense_transfer_execution"] == (
        "differentiable_folded"
    )
    assert folded.fingerprint() != legacy.fingerprint()


def test_expanded_dense_transfer_token_checkpoint_is_resume_critical() -> None:
    direct = make_config()
    checkpointed = copy.deepcopy(direct)
    checkpointed.runtime.dense_transfer_token_checkpoint = True

    checkpointed.validate()
    assert checkpointed.runtime.dense_transfer_execution == "expanded"
    assert checkpointed.fingerprint() != direct.fingerprint()


def test_phase_specific_checkpoint_counts_resolve_v2_policy_without_nesting() -> None:
    config = make_config()
    config.runtime.activation_checkpoint_layer_count = 0
    config.runtime.hidden_alignment_activation_checkpoint_layer_count = 8
    config.runtime.dense_transfer_token_checkpoint = True
    config.runtime.dense_transfer_checkpoint_layer_count = 0
    config.runtime.hidden_alignment_dense_transfer_checkpoint_layer_count = 16

    config.validate()
    ordinary_outer = config.activation_checkpoint_layer_indices(align_hidden=False)
    alignment_outer = config.activation_checkpoint_layer_indices(align_hidden=True)
    ordinary_inner = config.dense_transfer_checkpoint_layer_indices(
        align_hidden=False,
        outer_checkpoint_layer_indices=ordinary_outer,
    )
    alignment_inner = config.dense_transfer_checkpoint_layer_indices(
        align_hidden=True,
        outer_checkpoint_layer_indices=alignment_outer,
    )

    assert ordinary_outer == ()
    assert ordinary_inner == ()
    assert alignment_outer == (0, 3, 7, 10, 13, 16, 20, 23)
    assert alignment_inner == (1, 2, 4, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 19, 21, 22)
    assert not set(alignment_outer).intersection(alignment_inner)
    runtime = config.canonical_dict()["runtime"]
    assert runtime["activation_checkpoint_layer_count"] == 0
    assert runtime["hidden_alignment_activation_checkpoint_layer_count"] == 8
    assert runtime["dense_transfer_checkpoint_layer_count"] == 0
    assert runtime["hidden_alignment_dense_transfer_checkpoint_layer_count"] == 16
    assert config.fingerprint() != make_config().fingerprint()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hidden_alignment_activation_checkpoint_layer_count", 8),
        ("dense_transfer_checkpoint_layer_count", 4),
        ("hidden_alignment_dense_transfer_checkpoint_layer_count", 4),
    ],
)
def test_each_phase_specific_checkpoint_count_is_resume_critical(
    field: str,
    value: int,
) -> None:
    baseline = make_config()
    if "dense_transfer" in field:
        baseline.runtime.dense_transfer_token_checkpoint = True
    baseline.validate()
    changed = copy.deepcopy(baseline)
    setattr(changed.runtime, field, value)
    changed.validate()
    assert changed.fingerprint() != baseline.fingerprint()


def test_inner_subset_is_evenly_selected_from_noncontiguous_outer_complement() -> None:
    config = make_config()
    config.runtime.activation_checkpoint_layer_count = 6
    config.runtime.dense_transfer_token_checkpoint = True
    config.runtime.dense_transfer_checkpoint_layer_count = 4
    config.validate()

    outer = config.activation_checkpoint_layer_indices(align_hidden=False)
    inner = config.dense_transfer_checkpoint_layer_indices(
        align_hidden=False,
        outer_checkpoint_layer_indices=outer,
    )

    assert outer == (0, 5, 9, 14, 18, 23)
    assert inner == (1, 8, 15, 22)
    assert not set(outer).intersection(inner)


def test_inner_subset_samples_only_real_noncontiguous_active_layers() -> None:
    config = make_config()
    config.architecture.active_student_layers = [1, 3, 7, 11, 17, 23]
    config.runtime.activation_checkpoint_layer_count = 2
    config.runtime.dense_transfer_token_checkpoint = True
    config.runtime.dense_transfer_checkpoint_layer_count = 3
    config.validate()

    outer = config.activation_checkpoint_layer_indices(align_hidden=False)
    inner = config.dense_transfer_checkpoint_layer_indices(
        align_hidden=False,
        outer_checkpoint_layer_indices=outer,
    )

    assert outer == (1, 23)
    assert inner == (3, 11, 17)


def test_null_inner_counts_preserve_legacy_complete_outer_complement() -> None:
    config = make_config()
    config.runtime.activation_checkpoint_layer_count = 4
    config.runtime.dense_transfer_token_checkpoint = True
    config.validate()

    outer = config.activation_checkpoint_layer_indices(align_hidden=False)
    inner = config.dense_transfer_checkpoint_layer_indices(
        align_hidden=False,
        outer_checkpoint_layer_indices=outer,
    )

    assert outer == (0, 8, 15, 23)
    assert inner == tuple(layer for layer in range(24) if layer not in outer)

    assert config.activation_checkpoint_layer_indices(align_hidden=True) == outer


def test_inner_checkpoint_count_is_capped_by_outer_complement() -> None:
    config = make_config()
    config.runtime.activation_checkpoint_layer_count = 23
    config.runtime.dense_transfer_token_checkpoint = True
    config.runtime.dense_transfer_checkpoint_layer_count = 24
    config.validate()

    outer = config.activation_checkpoint_layer_indices(align_hidden=False)
    inner = config.dense_transfer_checkpoint_layer_indices(
        align_hidden=False,
        outer_checkpoint_layer_indices=outer,
    )

    assert len(outer) == 23
    assert len(inner) == 1
    assert not set(outer).intersection(inner)


@pytest.mark.parametrize(
    "field",
    [
        "dense_transfer_checkpoint_layer_count",
        "hidden_alignment_dense_transfer_checkpoint_layer_count",
    ],
)
def test_explicit_inner_counts_require_total_checkpoint_switch(field: str) -> None:
    config = make_config()
    setattr(config.runtime, field, 0)
    with pytest.raises(ConfigError, match="dense_transfer_token_checkpoint=true"):
        config.validate()


@pytest.mark.parametrize(
    "field",
    [
        "hidden_alignment_activation_checkpoint_layer_count",
        "dense_transfer_checkpoint_layer_count",
        "hidden_alignment_dense_transfer_checkpoint_layer_count",
    ],
)
@pytest.mark.parametrize("value", [-1, 25, True, 1.5])
def test_phase_specific_checkpoint_counts_are_strictly_validated(
    field: str,
    value: object,
) -> None:
    config = make_config()
    if "dense_transfer" in field:
        config.runtime.dense_transfer_token_checkpoint = True
    setattr(config.runtime, field, value)
    with pytest.raises(ConfigError, match=field):
        config.validate()


@pytest.mark.parametrize(
    "field",
    [
        "hidden_alignment_activation_checkpoint_layer_count",
        "dense_transfer_checkpoint_layer_count",
        "hidden_alignment_dense_transfer_checkpoint_layer_count",
    ],
)
def test_sparse_stage_rejects_explicit_phase_specific_counts(field: str) -> None:
    config = make_config()
    config.stage = "sparse"
    config.losses.hidden_alignment = 0.0
    config.sources.folded_experts_path = "artifacts/folded/model.safetensors"
    config.sources.folded_experts_sha256 = "e" * 64
    if "dense_transfer" in field:
        config.runtime.dense_transfer_token_checkpoint = True
    setattr(config.runtime, field, 0)
    with pytest.raises(ConfigError, match="only valid for dense-oracle"):
        config.validate()


@pytest.mark.parametrize("mode", ["folded", "fast", "", 1])
def test_dense_transfer_execution_rejects_unknown_modes(mode: object) -> None:
    config = make_config()
    config.runtime.dense_transfer_execution = mode  # type: ignore[assignment]
    with pytest.raises(ConfigError, match="dense_transfer_execution"):
        config.validate()


def test_selective_checkpoint_indices_sample_real_active_layer_numbers() -> None:
    config = make_config()
    config.architecture.active_student_layers = [1, 3, 7, 11, 17, 23]
    config.runtime.activation_checkpoint_layer_count = 4
    config.runtime.teacher_cpu_offload = True
    config.runtime.activation_checkpointing_on_alignment_only = True

    config.validate()
    assert config.activation_checkpoint_layer_indices(align_hidden=False) == (1, 7, 11, 23)
    assert config.activation_checkpoint_layer_indices(align_hidden=True) == (
        1,
        3,
        7,
        11,
        17,
        23,
    )


def test_null_checkpoint_count_preserves_legacy_global_and_alignment_only_policies() -> None:
    config = make_config()
    active = config.architecture.active_layers()
    assert config.activation_checkpoint_layer_indices(align_hidden=False) == active
    assert config.activation_checkpoint_layer_indices(align_hidden=True) == active

    config.runtime.teacher_cpu_offload = True
    config.runtime.activation_checkpointing_on_alignment_only = True
    config.validate()
    assert config.activation_checkpoint_layer_indices(align_hidden=False) == ()
    assert config.activation_checkpoint_layer_indices(align_hidden=True) == active


@pytest.mark.parametrize("count", [-1, 5, True, 1.5])
def test_checkpoint_layer_count_must_fit_active_layers(count: object) -> None:
    config = make_config()
    config.architecture.active_student_layers = [0, 8, 16, 23]
    config.runtime.activation_checkpoint_layer_count = count  # type: ignore[assignment]
    with pytest.raises(ConfigError, match="activation_checkpoint_layer_count"):
        config.validate()


def test_disabled_activation_checkpointing_rejects_nonzero_layer_count() -> None:
    config = make_config()
    config.runtime.activation_checkpointing = False
    config.runtime.activation_checkpoint_layer_count = 4
    with pytest.raises(ConfigError, match="activation_checkpointing=false"):
        config.validate()

    config.runtime.activation_checkpoint_layer_count = 0
    config.runtime.hidden_alignment_activation_checkpoint_layer_count = 4
    with pytest.raises(ConfigError, match="activation_checkpointing=false"):
        config.validate()


@pytest.mark.parametrize("chunk_tokens", [0, -1, True])
def test_loss_chunk_tokens_must_be_positive(chunk_tokens: int) -> None:
    config = make_config()
    config.runtime.loss_chunk_tokens = chunk_tokens
    with pytest.raises(ConfigError, match="loss_chunk_tokens"):
        config.validate()


def test_compile_streaming_loss_must_be_boolean() -> None:
    config = make_config()
    config.runtime.compile_streaming_loss = 1  # type: ignore[assignment]
    with pytest.raises(ConfigError, match="compile_streaming_loss"):
        config.validate()


def test_teacher_cpu_offload_defaults_disabled_and_is_resume_critical() -> None:
    disabled = make_config()
    enabled = copy.deepcopy(disabled)
    enabled.runtime.teacher_cpu_offload = True
    enabled.runtime.activation_checkpointing_on_alignment_only = True

    enabled.validate()
    assert disabled.runtime.teacher_cpu_offload is False
    assert disabled.runtime.activation_checkpointing_on_alignment_only is False
    assert disabled.fingerprint() != enabled.fingerprint()


@pytest.mark.parametrize(
    "field",
    ["teacher_cpu_offload", "activation_checkpointing_on_alignment_only"],
)
def test_teacher_residency_runtime_switches_must_be_boolean(field: str) -> None:
    config = make_config()
    setattr(config.runtime, field, 1)
    with pytest.raises(ConfigError, match=field):
        config.validate()


def test_alignment_only_checkpointing_requires_enabled_offload_and_checkpointing() -> None:
    config = make_config()
    config.runtime.activation_checkpointing_on_alignment_only = True
    with pytest.raises(ConfigError, match="teacher_cpu_offload=true"):
        config.validate()

    config.runtime.teacher_cpu_offload = True
    config.runtime.activation_checkpointing = False
    with pytest.raises(ConfigError, match="activation_checkpointing=true"):
        config.validate()


def test_teacher_cpu_offload_requires_dense_donor_hidden_alignment() -> None:
    config = make_config()
    config.runtime.teacher_cpu_offload = True
    config.losses.hidden_alignment = 0.0
    with pytest.raises(ConfigError, match="enabled hidden alignment"):
        config.validate()

    config.losses.hidden_alignment = 0.1
    config.architecture.expert_initialization = "random-control"
    with pytest.raises(ConfigError, match="donor initialization"):
        config.validate()


def test_learning_rate_changes_fingerprint() -> None:
    first = make_config()
    second = copy.deepcopy(first)
    second.optimizer.adapter_lr *= 2
    assert first.fingerprint() != second.fingerprint()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("adapter_lr", True, "adapter_lr"),
        ("scale_lr", "0.0001", "scale_lr"),
        ("weight_decay", False, "weight_decay"),
        ("adam_beta1", False, "adam_beta1"),
        ("warmup_tokens", 1.5, "token counts"),
        ("max_tokens", True, "token counts"),
    ],
)
def test_optimizer_numerics_reject_implicit_bool_and_string_coercion(
    field: str,
    value: object,
    message: str,
) -> None:
    config = make_config()
    setattr(config.optimizer, field, value)

    with pytest.raises(ConfigError, match=message):
        config.validate()


def test_legacy_lr_schedule_preserves_v1_canonical_shape() -> None:
    config = make_config()
    optimizer = config.canonical_dict()["optimizer"]

    assert config.optimizer.lr_schedule == "cosine"
    assert config.optimizer.min_lr_ratio == 0.1
    assert config.optimizer.decay_tokens is None
    assert "lr_schedule" not in optimizer
    assert "min_lr_ratio" not in optimizer
    assert "decay_tokens" not in optimizer


def test_legacy_optimizer_preserves_canonical_shape_and_muon_is_resume_critical() -> None:
    legacy = make_config()
    canonical = legacy.canonical_dict()["optimizer"]

    assert legacy.optimizer.adapter_optimizer == "adamw"
    assert not any(name.startswith("muon_") for name in canonical)
    assert "adapter_optimizer" not in canonical

    muon = copy.deepcopy(legacy)
    muon.optimizer.adapter_optimizer = "muon"
    muon.validate()
    muon_canonical = muon.canonical_dict()["optimizer"]
    assert muon_canonical["adapter_optimizer"] == "muon"
    assert muon_canonical["muon_adjust_lr_fn"] == "match_rms_adamw"
    assert muon_canonical["muon_ns_coefficients"] == (3.4445, -4.775, 2.0315)
    assert muon.fingerprint() != legacy.fingerprint()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("muon_momentum", 0.9),
        ("muon_nesterov", False),
        ("muon_ns_coefficients", (3.0, -4.0, 2.0)),
        ("muon_eps", 1e-6),
        ("muon_ns_steps", 4),
        ("muon_adjust_lr_fn", "original"),
    ],
)
def test_each_muon_numerical_choice_is_resume_critical(
    field: str,
    value: object,
) -> None:
    baseline = make_config()
    baseline.optimizer.adapter_optimizer = "muon"
    baseline.validate()
    changed = copy.deepcopy(baseline)
    setattr(changed.optimizer, field, value)
    changed.validate()

    assert changed.fingerprint() != baseline.fingerprint()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("adapter_optimizer", "sgd", "adapter_optimizer"),
        ("muon_momentum", 1.0, "muon_momentum"),
        ("muon_nesterov", 1, "muon_nesterov"),
        ("muon_ns_coefficients", (1.0, 2.0), "muon_ns_coefficients"),
        ("muon_eps", 0.0, "muon_eps"),
        ("muon_ns_steps", True, "muon_ns_steps"),
        ("muon_ns_steps", 100, "muon_ns_steps"),
        ("muon_adjust_lr_fn", "unknown", "muon_adjust_lr_fn"),
    ],
)
def test_muon_optimizer_fields_are_strictly_validated(
    field: str,
    value: object,
    message: str,
) -> None:
    config = make_config()
    config.optimizer.adapter_optimizer = "muon"
    setattr(config.optimizer, field, value)
    with pytest.raises(ConfigError, match=message):
        config.validate()


def test_muon_fields_cannot_silently_change_legacy_adamw() -> None:
    config = make_config()
    config.optimizer.muon_ns_steps = 3
    with pytest.raises(ConfigError, match="require adapter_optimizer='muon'"):
        config.validate()


def test_muon_ns_steps_accepts_the_documented_upper_boundary() -> None:
    config = make_config()
    config.optimizer.adapter_optimizer = "muon"
    config.optimizer.muon_ns_steps = 99

    config.validate()


def test_muon_adapter_optimizer_is_dense_only() -> None:
    config = make_config()
    config.optimizer.adapter_optimizer = "muon"
    config.stage = "sparse"
    config.losses.hidden_alignment = 0.0
    config.sources.folded_experts_path = "folded.safetensors"
    config.sources.folded_experts_sha256 = "e" * 64
    with pytest.raises(ConfigError, match="requires stage='dense-oracle'"):
        config.validate()


def test_warmup_stable_decay_is_validated_and_resume_critical() -> None:
    legacy = make_config()
    wsd = copy.deepcopy(legacy)
    wsd.optimizer.lr_schedule = "warmup-stable-decay"
    wsd.optimizer.decay_tokens = 50_000_000

    wsd.validate()
    optimizer = wsd.canonical_dict()["optimizer"]
    assert optimizer["lr_schedule"] == "warmup-stable-decay"
    assert optimizer["min_lr_ratio"] == 0.1
    assert optimizer["decay_tokens"] == 50_000_000
    assert wsd.fingerprint() != legacy.fingerprint()


@pytest.mark.parametrize("decay_tokens", [None, 0, -1, True, 500_000_001])
def test_warmup_stable_decay_requires_a_valid_final_interval(
    decay_tokens: int | None,
) -> None:
    config = make_config()
    config.optimizer.lr_schedule = "warmup-stable-decay"
    config.optimizer.decay_tokens = decay_tokens
    with pytest.raises(ConfigError, match="decay_tokens"):
        config.validate()


def test_mtp_is_disabled_by_default_and_enabled_weight_is_resume_critical() -> None:
    disabled = make_config()
    enabled = copy.deepcopy(disabled)
    enabled.losses.mtp = 0.2

    assert disabled.losses.mtp == 0.0
    enabled.validate()
    assert disabled.fingerprint() != enabled.fingerprint()


@pytest.mark.parametrize("weight", [-1.0, math.nan, math.inf, -math.inf])
def test_mtp_weight_must_be_finite_and_non_negative(weight: float) -> None:
    config = make_config()
    config.losses.mtp = weight
    with pytest.raises(ConfigError, match=r"losses\.mtp"):
        config.validate()


def test_enabled_mtp_requires_an_l_minus_two_target() -> None:
    config = make_config()
    config.losses.mtp = 0.1
    config.data.max_sequence_length = 2
    with pytest.raises(ConfigError, match="max_sequence_length>=3"):
        config.validate()


def test_model_revision_requires_a_full_provider_commit_sha() -> None:
    config = make_config()
    config.sources.backbone.revision = "abcdef0"
    with pytest.raises(ConfigError, match="immutable commit SHA"):
        config.validate()


def test_hex_identities_are_canonicalized_to_lowercase() -> None:
    config = make_config()
    config.sources.backbone.revision = "A" * 40
    config.sources.backbone.manifest_sha256 = "B" * 64
    config.data.manifest_sha256 = "C" * 64
    config.data.teacher_kd_manifest_sha256 = "D" * 64
    config.validate()
    assert config.sources.backbone.revision == "a" * 40
    assert config.sources.backbone.manifest_sha256 == "b" * 64
    assert config.data.manifest_sha256 == "c" * 64
    assert config.data.teacher_kd_manifest_sha256 == "d" * 64


def test_dense_poc_layers_are_valid_but_sparse_requires_all_layers() -> None:
    config = make_config()
    config.architecture.active_student_layers = [5, 11, 17, 23]
    config.validate()
    config.stage = "sparse"
    config.sources.folded_experts_path = "artifacts/folded/model.safetensors"
    config.sources.folded_experts_sha256 = "e" * 64
    with pytest.raises(ConfigError, match="all 24"):
        config.validate()


def test_native_router_normalization_cannot_be_disabled() -> None:
    config = make_config()
    config.architecture.norm_topk_prob = False
    with pytest.raises(ConfigError, match="norm_topk_prob"):
        config.validate()


def test_v1_requires_rank_16_expert_lora() -> None:
    config = make_config()
    config.architecture.lora_rank = 0
    with pytest.raises(ConfigError, match="lora_rank=16"):
        config.validate()


def test_sparse_config_requires_a_pinned_folded_expert_sha256() -> None:
    config = make_config()
    config.stage = "sparse"
    config.losses.hidden_alignment = 0.0
    config.sources.folded_experts_path = "artifacts/folded/model.safetensors"

    with pytest.raises(ConfigError, match="folded_experts_sha256"):
        config.validate()

    config.sources.folded_experts_sha256 = "not-a-sha256"
    with pytest.raises(ConfigError, match="folded_experts_sha256"):
        config.validate()

    config.sources.folded_experts_sha256 = "e" * 64
    config.validate()


def test_sparse_config_rejects_unused_hidden_alignment_objective() -> None:
    config = make_config()
    config.stage = "sparse"
    config.sources.folded_experts_path = "artifacts/folded/model.safetensors"
    config.sources.folded_experts_sha256 = "e" * 64
    with pytest.raises(ConfigError, match="hidden_alignment=0"):
        config.validate()


def test_random_expert_control_is_dense_only_and_resume_critical() -> None:
    donor = make_config()
    random_control = copy.deepcopy(donor)
    random_control.architecture.expert_initialization = "random-control"
    random_control.validate()
    assert random_control.fingerprint() != donor.fingerprint()

    random_control.stage = "sparse"
    random_control.sources.folded_experts_path = "artifacts/folded/model.safetensors"
    random_control.sources.folded_experts_sha256 = "e" * 64
    with pytest.raises(ConfigError, match="donor expert initialization"):
        random_control.validate()


@pytest.mark.parametrize(
    "temperature",
    [0.0, -1.0, math.nan, math.inf, -math.inf],
)
def test_kd_temperature_must_be_finite_and_positive(temperature: float) -> None:
    config = make_config()
    config.losses.kd_temperature = temperature
    with pytest.raises(ConfigError, match="kd_temperature"):
        config.validate()
