from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

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
    dump_resolved_config,
)
from twen.utils import sha256_file
from twen.v2_finalizer import (
    ALIGNMENT_CASE,
    ALIGNMENT_INNER_CHECKPOINT_LAYERS,
    ALIGNMENT_OUTER_CHECKPOINT_LAYERS,
    EXPANDED_NUMERICAL_COMPLETE,
    EXPANDED_NUMERICAL_MANIFEST,
    FOLDED_NUMERICAL_COMPLETE,
    FOLDED_NUMERICAL_MANIFEST,
    GLOBAL_BATCH_TOKENS,
    LOSS_CHUNK_TOKENS,
    ORDINARY_CASE,
    ORDINARY_INNER_CHECKPOINT_LAYERS,
    ORDINARY_OUTER_CHECKPOINT_LAYERS,
    PRODUCTION_RUNTIME,
    QUALITY_COOLDOWN_START_TOKENS,
    FinalizationError,
    FinalizationOptions,
    _authenticate_numerical_evidence,
    _authenticate_performance,
    _authenticate_preflight,
    _build_config,
    finalize_base_v2,
)


def _base_config(root: Path) -> Path:
    backbone = ModelSource(
        model_id="Qwen/Qwen3.5-0.8B-Base",
        revision="a" * 40,
        local_path="artifacts/models/backbone",
        manifest_sha256="b" * 64,
    )
    donor = ModelSource(
        model_id="Qwen/Qwen3.5-9B-Base",
        revision="c" * 40,
        local_path="artifacts/models/donor",
        manifest_sha256="d" * 64,
    )
    config = TrainConfig(
        run_id="base-dense-v1",
        track="base",
        stage="dense-oracle",
        sources=SourcesConfig(
            backbone=backbone,
            donor=donor,
            teacher=donor,
            tokenizer=backbone,
        ),
        architecture=ArchitectureConfig(),
        data=DataConfig(
            manifest_path="old-prepared.json",
            manifest_sha256="e" * 64,
            teacher_kd_manifest_path="old-kd.json",
            teacher_kd_manifest_sha256="f" * 64,
        ),
        losses=LossConfig(),
        optimizer=OptimizerConfig(),
        checkpoint=CheckpointConfig(output_dir="runs/base-dense-v1"),
        runtime=RuntimeConfig(),
    )
    path = root / "configs/base/dense-oracle.yaml"
    path.parent.mkdir(parents=True)
    dump_resolved_config(config, path)
    return path


def _dashboard(root: Path) -> Path:
    path = root / "configs/web/dashboard.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_root": "../..",
                "state_dir": ".twen/dashboard",
                "profiles": [
                    {
                        "id": "base-dense-v1",
                        "label": "v1",
                        "config": "configs/base/dense-oracle.yaml",
                        "resume": "auto",
                        "fork_from": None,
                        "launch_enabled": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _options(root: Path, *, enable_web_launch: bool = False) -> FinalizationOptions:
    return FinalizationOptions.repository_defaults(
        root=root,
        mtp_loss_weight=0.1,
        adapter_lr=2e-4,
        router_lr=1e-3,
        lora_lr=2e-4,
        scale_lr=1e-3,
        quality_cooldown_prepared_manifest=(
            "artifacts/data/base-v2-500m-quality-bundle/prepared/manifest.json"
        ),
        quality_cooldown_kd_manifest=(
            "artifacts/data/base-v2-500m-quality-bundle/kd/manifest.json"
        ),
        enable_web_launch=enable_web_launch,
    )


def _identity(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _bundle_file(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_valid_numerical_evidence(root: Path) -> dict[str, object]:
    evidence_root = (root / EXPANDED_NUMERICAL_MANIFEST).parent
    evidence_root.mkdir(parents=True, exist_ok=True)

    expanded_admission = {
        "differentiable_folded": False,
        "expanded_selective_checkpoint_numerically_admitted": True,
        "formula": "expanded",
        "production_enabled": True,
    }
    expanded_lineage = {
        "engine_source_sha256": "1" * 64,
        "modules_source_sha256": "2" * 64,
        "config_sha256": "3" * 64,
    }
    expanded_oracle = evidence_root / "expanded_selective_checkpoint_full_graph_oracle.json"
    expanded_source = evidence_root / "expanded_selective_checkpoint_full_graph_oracle.py"
    expanded_report = evidence_root / "EXPANDED_SELECTIVE_CHECKPOINT_FULL_GRAPH_REPORT.md"
    expanded_source.write_text("# expanded synthetic oracle\n", encoding="utf-8")
    expanded_report.write_text("expanded PASS\n", encoding="utf-8")
    expanded_oracle.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "twen_expanded_selective_checkpoint_real_kd_full_graph_admission",
                "ok": True,
                "scope": {
                    "formula": "expanded",
                    "batch_size": 1,
                    "sequence_length": 4096,
                    "optimizer_constructed": False,
                    "optimizer_step_calls": 0,
                    "parameters_updated": False,
                },
                "admission": expanded_admission,
                "lineage": expanded_lineage,
                "gates": {"losses_bitwise": True},
                "health": {
                    "execution_mode_expanded": True,
                    "outer_ac_disables_inner": True,
                    "parameters_unchanged": True,
                    "quick_bitwise_passed": True,
                    "selective_core_calls_exact": True,
                },
            }
        ),
        encoding="utf-8",
    )
    expanded_manifest = root / EXPANDED_NUMERICAL_MANIFEST
    expanded_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "twen_expanded_selective_checkpoint_full_graph_bundle",
                "status": "complete",
                "optimizer_constructed": False,
                "optimizer_step_calls": 0,
                "parameters_updated": False,
                "admission": expanded_admission,
                "lineage": expanded_lineage,
                "files": [
                    _bundle_file(expanded_source),
                    _bundle_file(expanded_oracle),
                    _bundle_file(expanded_report),
                ],
            }
        ),
        encoding="utf-8",
    )
    expanded_complete = root / EXPANDED_NUMERICAL_COMPLETE
    expanded_complete.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "twen_complete_marker",
                "status": "complete",
                "manifest": expanded_manifest.name,
                "manifest_sha256": sha256_file(expanded_manifest),
            }
        ),
        encoding="utf-8",
    )

    folded_admission = {
        "differentiable_folded_production_enabled": False,
        "folded_execution_status": "experimental_only",
        "strict_real_kd_accumulation": False,
        "v2_production_execution": "expanded",
    }
    folded_decision = {
        "differentiable_folded_production_enabled": False,
        "folded_execution_status": "experimental_only",
        "reason": "strict real-KD accumulated fold gate failed",
        "v2_production_execution": "expanded",
    }
    folded_oracle = evidence_root / "full_graph_v1_real_kd_accumulation_oracle.json"
    folded_source = evidence_root / "full_graph_v1_real_kd_accumulation_oracle.py"
    folded_report = evidence_root / "FULL_GRAPH_V1_REAL_KD_ACCUMULATION_REPORT.md"
    folded_source.write_text("# folded synthetic oracle\n", encoding="utf-8")
    folded_report.write_text("folded FAIL\n", encoding="utf-8")
    folded_oracle.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "twen_v1_final_real_kd_full_graph_fold_accumulation_oracle",
                "ok": False,
                "scope": {
                    "formal_training": False,
                    "microbatch_count": 4,
                    "batch_size": 1,
                    "sequence_length": 4096,
                    "optimizer_constructed": False,
                    "optimizer_step_calls": 0,
                    "parameters_updated": False,
                },
                "decision": folded_decision,
                "health": {"parameters_unchanged": True},
            }
        ),
        encoding="utf-8",
    )
    folded_manifest = root / FOLDED_NUMERICAL_MANIFEST
    folded_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "twen_v1_final_real_kd_fold_accumulation_oracle_bundle",
                "status": "complete",
                "optimizer_constructed": False,
                "optimizer_step_calls": 0,
                "parameters_updated": False,
                "admission": folded_admission,
                "decision": folded_decision,
                "files": [
                    _bundle_file(folded_source),
                    _bundle_file(folded_oracle),
                    _bundle_file(folded_report),
                ],
            }
        ),
        encoding="utf-8",
    )
    folded_complete = root / FOLDED_NUMERICAL_COMPLETE
    folded_complete.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "twen_complete_marker",
                "status": "complete",
                "manifest": folded_manifest.name,
                "manifest_sha256": sha256_file(folded_manifest),
            }
        ),
        encoding="utf-8",
    )
    return _authenticate_numerical_evidence(root)


def _performance_source(
    root: Path,
    *,
    mode: str,
    outer_layers: tuple[int, ...],
    inner_layers: tuple[int, ...],
    production_enabled: bool,
) -> Path:
    path = root / f"artifacts/benchmarks/synthetic-{mode}.json"
    path.write_text(
        json.dumps(
            {
                "ok": True,
                "production_shape": True,
                "production_acceptance": True,
                "no_optimizer_created": True,
                "no_optimizer_steps": True,
                "optimizer_step_calls": 0,
                "batch": {
                    "batch_size": 1,
                    "sequence_length": 4096,
                    "logical_tokens": 4096,
                },
                "runtime": {
                    "activation_checkpoint_layer_count": len(outer_layers),
                    "activation_checkpoint_layer_indices": list(outer_layers),
                    "dense_transfer_checkpoint_layer_count_requested": len(inner_layers),
                    "dense_transfer_checkpoint_layer_count_effective": len(inner_layers),
                    "dense_transfer_checkpoint_layer_indices": list(inner_layers),
                    "dense_transfer_checkpoint_selection_policy": (
                        "deterministic_evenly_spaced_outer_complement"
                    ),
                    "dense_transfer_outer_inner_disjoint": True,
                    "dense_transfer_execution": "expanded",
                    "loss_chunk_tokens": 512,
                    "loss_checkpoint_chunks": True,
                    "teacher_cpu_offload": True,
                },
                "dense_transfer_actual_state": {
                    "actual_checkpoint_layer_indices": list(inner_layers),
                    "actual_execution_modes": ["expanded"],
                    "all_modules_match_requested_state": True,
                    "outer_inner_disjoint": True,
                },
                "experimental_execution": {
                    "mode": "expanded",
                    "production_enabled": production_enabled,
                    "selected_mode_numerical_status": "admitted_production_reference",
                },
                "mtp": {
                    "enabled": True,
                    "frozen": True,
                    "attention_implementation": "sdpa",
                    "loss_weight": 0.1,
                },
                "optimizer_state_reserve": {
                    "requested_gib": 1.5,
                    "is_optimizer": False,
                    "resident_during_all_iterations": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _performance_row(
    root: Path,
    *,
    label: str,
    mode: str,
    outer_layers: tuple[int, ...],
    inner_layers: tuple[int, ...],
    production_enabled: bool = True,
) -> dict[str, object]:
    source = _performance_source(
        root,
        mode=mode,
        outer_layers=outer_layers,
        inner_layers=inner_layers,
        production_enabled=production_enabled,
    )
    return {
        "label": label,
        "mode": mode,
        "batch_size": 1,
        "logical_tokens": 4096,
        "activation_checkpoint_layer_count": len(outer_layers),
        "dense_transfer_checkpoint_layer_count": len(inner_layers),
        "dense_transfer_execution": "expanded",
        "status": "ok",
        "accepted": True,
        "production_acceptance": True,
        "safety_gate_passed": True,
        "teacher_cpu_offload": True,
        "optimizer_state_reserve_gib": 1.5,
        "production_accumulation_microbatches": 64,
        "mtp_loss_weight": 0.1,
        "mtp_attention_implementation": "sdpa",
        "no_optimizer_created_or_stepped": True,
        "health": {
            "ok": True,
            "loss_finite": True,
            "gradients_finite": True,
            "missing_gradient_tensors": 0,
            "nonfinite_gradient_tensors": 0,
            "present_gradient_tensor_counts": [72],
        },
        "production_tokens_per_second": 8000.0,
        "minimum_estimated_headroom_gib": 4.0,
        "minimum_nvml_physical_free_gib": 4.5,
        "source": _identity(root, source),
    }


def _patch_performance_module(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    production_enabled: bool = True,
) -> dict[str, object]:
    numerical = _write_valid_numerical_evidence(root)
    benchmark_root = root / "artifacts/benchmarks"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    report = benchmark_root / "report.json"
    approval = benchmark_root / "approval.json"
    manifest = benchmark_root / "MANIFEST.json"
    complete = benchmark_root / "COMPLETE"
    for path in (approval, manifest, complete):
        path.write_text("{}\n", encoding="utf-8")
    report.write_text(
        json.dumps(
            {
                "recommendation": {
                    "batch_size": 1,
                    "ordinary_case": ORDINARY_CASE,
                    "alignment_case": ALIGNMENT_CASE,
                    "production_config": PRODUCTION_RUNTIME,
                    "tokens_per_second": 7000.0,
                },
                "numerical_admission": {
                    "production_reference_mode": "expanded",
                    "expanded": {
                        "status": "pass",
                        "complete": numerical["expanded"]["complete"],
                        "report": numerical["expanded"]["report"],
                    },
                    "folded": {
                        "status": "fail_experimental_only",
                        "complete": numerical["folded"]["complete"],
                        "report": numerical["folded"]["report"],
                    },
                },
                "rows": [
                    _performance_row(
                        root,
                        label=ORDINARY_CASE,
                        mode="ordinary",
                        outer_layers=ORDINARY_OUTER_CHECKPOINT_LAYERS,
                        inner_layers=ORDINARY_INNER_CHECKPOINT_LAYERS,
                        production_enabled=production_enabled,
                    ),
                    _performance_row(
                        root,
                        label=ALIGNMENT_CASE,
                        mode="alignment",
                        outer_layers=ALIGNMENT_OUTER_CHECKPOINT_LAYERS,
                        inner_layers=ALIGNMENT_INNER_CHECKPOINT_LAYERS,
                        production_enabled=production_enabled,
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )
    pipeline_source = root / "scripts/prepare_base_v2_500m.py"
    pipeline_source.parent.mkdir(parents=True)
    pipeline_source.write_text("# synthetic pipeline contract\n", encoding="utf-8")
    layout = SimpleNamespace(
        performance_gate=report,
        performance_approval=approval,
        performance_manifest=manifest,
        performance_complete=complete,
    )
    module = SimpleNamespace(
        Layout=SimpleNamespace(repository_defaults=lambda _root: layout),
        _performance_gate=lambda _layout: {"ready": True},
    )
    import twen.v2_finalizer as finalizer

    monkeypatch.setattr(finalizer, "_load_pipeline_module", lambda _root: module)
    monkeypatch.setattr(
        finalizer,
        "EXPECTED_PERFORMANCE_REPORT_SHA256",
        sha256_file(report),
    )
    return numerical


def _authenticated_data() -> dict:
    return {
        "prepared": {"path": "prepared.json", "size": 1, "sha256": "1" * 64},
        "kd": {"path": "kd.json", "size": 1, "sha256": "2" * 64},
        "audit": {"path": "audit.json", "size": 1, "sha256": "3" * 64},
        "pipeline": {},
        "dataset_fingerprint": "4" * 64,
        "sequence_count": 125_000,
        "token_count": 512_000_000,
        "audit_attestation_fingerprint": "5" * 64,
        "audit_gates": {"all": {"passed": True}},
        "kd_generator_source_sha256": "6" * 64,
        "quality_cooldown": {
            "prepared": {
                "path": "cooldown/prepared/manifest.json",
                "size": 1,
                "sha256": "9" * 64,
            },
            "kd": {
                "path": "cooldown/kd/manifest.json",
                "size": 1,
                "sha256": "a" * 64,
            },
            "start_tokens": 450_000_000,
            "required_tokens": 50_000_000,
            "selection_policy_id": "reviewed-v1",
            "dataset_fingerprint": "b" * 64,
            "selected_shard_ids": ["shard-000007"],
            "source_mix_token_counts": [["math", 50_000_000]],
            "sequence_count": 12_500,
            "token_count": 50_000_000,
        },
    }


def _patch_authentication(monkeypatch: pytest.MonkeyPatch) -> None:
    import twen.v2_finalizer as finalizer

    monkeypatch.setattr(
        finalizer,
        "_authenticate_performance",
        lambda *_args: {
            "recommendation": {
                "batch_size": 1,
                "ordinary_case": ORDINARY_CASE,
                "alignment_case": ALIGNMENT_CASE,
            }
        },
    )
    monkeypatch.setattr(
        finalizer,
        "_authenticate_numerical_evidence",
        lambda _root: {
            "production_execution": "expanded",
            "historical_embedded_source_sha256": {
                "engine": "1" * 64,
                "modules": "2" * 64,
                "archived_config": "3" * 64,
            },
            "expanded": {"status": "pass", "production_enabled": True},
            "folded": {
                "status": "fail_experimental_only",
                "production_enabled": False,
            },
        },
    )
    monkeypatch.setattr(
        finalizer,
        "_authenticate_current_production_sources",
        lambda _root: {
            "evidence_role": "current_production_implementation",
            "files": {
                name: {"path": f"synthetic/{name}.py", "size": 1, "sha256": "c" * 64}
                for name in ("config", "preflight", "engine", "modules", "benchmark", "mtp")
            },
        },
    )
    monkeypatch.setattr(finalizer, "_authenticate_data", lambda *_args: _authenticated_data())
    monkeypatch.setattr(
        finalizer,
        "_authenticate_fork",
        lambda *_args: {"path": "runs/base-dense-v1/step-final"},
    )
    monkeypatch.setattr(
        finalizer,
        "_authenticate_preflight",
        lambda *_args: {
            "config_fingerprint": "7" * 64,
            "data_fingerprint": "1" * 64,
            "source_tree_sha256": "8" * 64,
            "sources": {},
        },
    )


def test_build_config_freezes_wsd_cooldown_and_expanded_b1_semantics(
    tmp_path: Path,
) -> None:
    base_path = _base_config(tmp_path)
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    from twen.config import load_train_config

    config = _build_config(
        _options(tmp_path),
        load_train_config(base_path),
        _authenticated_data(),
    )

    assert base["run_id"] == "base-dense-v1"
    assert config.run_id == "base-dense-v2-500m"
    assert config.checkpoint.output_dir == "runs/base-dense-v2-500m"
    assert config.optimizer.warmup_tokens == 5_000_000
    assert config.optimizer.max_tokens == 500_000_000
    assert config.optimizer.decay_tokens == 50_000_000
    assert config.optimizer.lr_schedule == "warmup-stable-decay"
    assert config.optimizer.min_lr_ratio == 0.1
    assert config.data.global_batch_tokens == GLOBAL_BATCH_TOKENS
    assert config.data.micro_batch_size == 1
    assert config.data.quality_cooldown_start_tokens == QUALITY_COOLDOWN_START_TOKENS
    assert config.data.quality_cooldown_manifest_path == "cooldown/prepared/manifest.json"
    assert config.data.quality_cooldown_teacher_kd_manifest_path == "cooldown/kd/manifest.json"
    assert config.losses.mtp == 0.1
    assert (
        config.activation_checkpoint_layer_indices(align_hidden=False)
        == ORDINARY_OUTER_CHECKPOINT_LAYERS
    )
    assert (
        config.activation_checkpoint_layer_indices(align_hidden=True)
        == ALIGNMENT_OUTER_CHECKPOINT_LAYERS
    )
    assert config.runtime.dense_transfer_execution == "expanded"
    assert config.runtime.dense_transfer_token_checkpoint is True
    assert config.runtime.dense_transfer_checkpoint_layer_count == 0
    assert config.runtime.hidden_alignment_dense_transfer_checkpoint_layer_count == 16
    assert (
        config.dense_transfer_checkpoint_layer_indices(align_hidden=False)
        == ORDINARY_INNER_CHECKPOINT_LAYERS
    )
    assert (
        config.dense_transfer_checkpoint_layer_indices(align_hidden=True)
        == ALIGNMENT_INNER_CHECKPOINT_LAYERS
    )
    assert config.runtime.loss_chunk_tokens == LOSS_CHUNK_TOKENS
    assert config.checkpoint.every_steps == 100
    assert config.checkpoint.every_minutes == 30.0


def test_preflight_locks_current_expanded_execution_and_exact_phase_indices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import twen.preflight as preflight_module
    from twen.config import load_train_config
    from twen.preflight import (
        BatchGeometry,
        DataGovernanceStatus,
        PreflightReport,
    )

    config = _build_config(
        _options(tmp_path),
        load_train_config(_base_config(tmp_path)),
        _authenticated_data(),
    )
    written_sources: dict[str, str] = {}
    for role in ("backbone", "donor", "teacher", "tokenizer"):
        source = getattr(config.sources, role)
        if source.local_path not in written_sources:
            manifest = tmp_path / source.local_path / "download-manifest.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(f'{{"role":"{role}"}}\n', encoding="utf-8")
            written_sources[source.local_path] = sha256_file(manifest)
        source.manifest_sha256 = written_sources[source.local_path]

    cooldown = _authenticated_data()["quality_cooldown"]
    report = PreflightReport(
        data_governance=DataGovernanceStatus(
            lineage_kind="authenticated_extracted_corpus",
            ready_for_training=True,
            research_only=False,
            pending_audits=(),
            warning=None,
        ),
        batch=BatchGeometry(
            world_size=1,
            micro_batch_tokens_per_rank=4096,
            gradient_accumulation_steps=64,
            global_batch_tokens=262_144,
        ),
        activation_checkpoint_layer_count=0,
        hidden_alignment_activation_checkpoint_layer_count=8,
        activation_checkpoint_layer_indices=ORDINARY_OUTER_CHECKPOINT_LAYERS,
        hidden_alignment_activation_checkpoint_layer_indices=(ALIGNMENT_OUTER_CHECKPOINT_LAYERS),
        dense_transfer_execution="expanded",
        dense_transfer_checkpoint_layer_count=0,
        hidden_alignment_dense_transfer_checkpoint_layer_count=16,
        dense_transfer_token_checkpoint_layer_indices=ORDINARY_INNER_CHECKPOINT_LAYERS,
        hidden_alignment_dense_transfer_token_checkpoint_layer_indices=(
            ALIGNMENT_INNER_CHECKPOINT_LAYERS
        ),
        quality_cooldown_enabled=True,
        quality_cooldown_start_tokens=450_000_000,
        quality_cooldown_dataset_fingerprint=cooldown["dataset_fingerprint"],
        quality_cooldown_sequence_count=cooldown["sequence_count"],
        quality_cooldown_token_count=cooldown["token_count"],
        quality_cooldown_selected_shard_ids=tuple(cooldown["selected_shard_ids"]),
        quality_cooldown_source_mix_token_counts=tuple(
            tuple(item) for item in cooldown["source_mix_token_counts"]
        ),
        config_fingerprint="d" * 64,
        data_fingerprint="e" * 64,
        source_tree_sha256="f" * 64,
        checked_paths=(),
        calibration_fingerprints=(),
    )
    reports = [report]
    monkeypatch.setattr(
        preflight_module,
        "run_training_preflight",
        lambda *_args, **_kw: reports[0],
    )

    authenticated = _authenticate_preflight(
        _options(tmp_path),
        config,
        _authenticated_data(),
    )
    assert authenticated["dense_transfer_execution"] == "expanded"
    assert authenticated["activation_checkpoint_layer_indices"] == []
    assert authenticated["hidden_alignment_activation_checkpoint_layer_indices"] == list(
        ALIGNMENT_OUTER_CHECKPOINT_LAYERS
    )
    assert authenticated["hidden_alignment_dense_transfer_token_checkpoint_layer_indices"] == list(
        ALIGNMENT_INNER_CHECKPOINT_LAYERS
    )

    reports[0] = replace(report, dense_transfer_execution="differentiable_folded")
    with pytest.raises(FinalizationError, match="expanded B1 outer0/8 inner0/16"):
        _authenticate_preflight(_options(tmp_path), config, _authenticated_data())


def test_finalize_publishes_only_after_expanded_and_cooldown_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _base_config(tmp_path)
    dashboard = _dashboard(tmp_path)
    _patch_authentication(monkeypatch)

    result = finalize_base_v2(_options(tmp_path))

    assert result["ok"] is True
    assert result["training_started"] is False
    assert result["optimizer_created"] is False
    config = yaml.safe_load((tmp_path / "configs/base/dense-v2-500m.yaml").read_text())
    assert config["optimizer"]["warmup_tokens"] == 5_000_000
    assert config["optimizer"]["max_tokens"] == 500_000_000
    assert config["optimizer"]["lr_schedule"] == "warmup-stable-decay"
    assert config["optimizer"]["min_lr_ratio"] == 0.1
    assert config["optimizer"]["decay_tokens"] == 50_000_000
    assert config["data"]["micro_batch_size"] == 1
    assert config["data"]["quality_cooldown_start_tokens"] == 450_000_000
    assert config["runtime"]["activation_checkpoint_layer_count"] == 0
    assert config["runtime"]["hidden_alignment_activation_checkpoint_layer_count"] == 8
    assert config["runtime"]["dense_transfer_execution"] == "expanded"
    assert config["runtime"]["dense_transfer_token_checkpoint"] is True
    assert config["runtime"]["dense_transfer_checkpoint_layer_count"] == 0
    assert config["runtime"]["hidden_alignment_dense_transfer_checkpoint_layer_count"] == 16
    assert config["runtime"]["loss_chunk_tokens"] == 512
    profile = json.loads(dashboard.read_text())["profiles"][-1]
    assert profile == {
        "id": "base-dense-v2-500m",
        "label": "Base Dense v2 500M (finalized fork)",
        "config": "configs/base/dense-v2-500m.yaml",
        "config_sha256": sha256_file(tmp_path / "configs/base/dense-v2-500m.yaml"),
        "resume": "none",
        "fork_from": "runs/base-dense-v1/step-000000000383-milestone-complete",
        "launch_enabled": False,
    }
    evidence = json.loads(
        (tmp_path / "artifacts/configuration/base-dense-v2-500m/manifest.json").read_text()
    )
    assert evidence["training_contract"]["stable_until_tokens"] == 450_000_000
    assert evidence["training_contract"]["physical_micro_batch_size"] == 1
    assert evidence["training_contract"]["gradient_accumulation_steps_single_gpu"] == 64
    assert evidence["training_contract"]["quality_cooldown_start_tokens"] == 450_000_000
    assert evidence["inputs"]["numerical_evidence"]["expanded"]["status"] == "pass"
    assert evidence["inputs"]["numerical_evidence"]["folded"]["status"] == "fail_experimental_only"
    provenance = evidence["inputs"]["source_provenance"]
    assert (
        provenance["historical_numerical_math_evidence"]["current_source_equality_required"]
        is False
    )
    assert set(provenance["current_production_implementation"]["files"]) == {
        "config",
        "preflight",
        "engine",
        "modules",
        "benchmark",
        "mtp",
    }
    assert evidence["safety"]["optimizer_step_called"] is False
    assert evidence["outputs"]["dashboard_profile"]["launch_enabled"] is False


def test_web_launch_requires_an_explicit_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _base_config(tmp_path)
    dashboard = _dashboard(tmp_path)
    _patch_authentication(monkeypatch)

    finalize_base_v2(_options(tmp_path, enable_web_launch=True))

    profile = json.loads(dashboard.read_text())["profiles"][-1]
    assert profile["launch_enabled"] is True


def test_numerical_evidence_accepts_expanded_pass_and_preserves_folded_fail(
    tmp_path: Path,
) -> None:
    authenticated = _write_valid_numerical_evidence(tmp_path)

    assert authenticated["production_execution"] == "expanded"
    assert authenticated["expanded"]["status"] == "pass"
    assert authenticated["expanded"]["production_enabled"] is True
    assert authenticated["folded"]["status"] == "fail_experimental_only"
    assert authenticated["folded"]["production_enabled"] is False
    assert authenticated["historical_embedded_source_sha256"] == {
        "engine": "1" * 64,
        "modules": "2" * 64,
        "archived_config": "3" * 64,
    }
    assert authenticated["current_production_source_equality_required"] is False


def test_numerical_evidence_rejects_hidden_folded_real_kd_fail(tmp_path: Path) -> None:
    _write_valid_numerical_evidence(tmp_path)
    manifest_path = tmp_path / FOLDED_NUMERICAL_MANIFEST
    oracle_path = manifest_path.parent / "full_graph_v1_real_kd_accumulation_oracle.json"
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    oracle["ok"] = True
    oracle_path.write_text(json.dumps(oracle), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    oracle_identity = next(item for item in manifest["files"] if item["path"] == oracle_path.name)
    oracle_identity.update(_bundle_file(oracle_path))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    complete_path = tmp_path / FOLDED_NUMERICAL_COMPLETE
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["manifest_sha256"] = sha256_file(manifest_path)
    complete_path.write_text(json.dumps(complete), encoding="utf-8")

    with pytest.raises(FinalizationError, match="folded real-KD numerical FAIL"):
        _authenticate_numerical_evidence(tmp_path)


def test_authentication_failure_leaves_config_and_dashboard_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _base_config(tmp_path)
    dashboard = _dashboard(tmp_path)
    original = dashboard.read_bytes()
    import twen.v2_finalizer as finalizer

    monkeypatch.setattr(
        finalizer,
        "_authenticate_numerical_evidence",
        lambda _root: {"production_execution": "expanded"},
    )
    monkeypatch.setattr(
        finalizer,
        "_authenticate_performance",
        lambda *_args: (_ for _ in ()).throw(FinalizationError("wrong batch")),
    )

    with pytest.raises(FinalizationError, match="wrong batch"):
        finalize_base_v2(_options(tmp_path))

    assert not (tmp_path / "configs/base/dense-v2-500m.yaml").exists()
    assert dashboard.read_bytes() == original
    assert not (tmp_path / "artifacts/configuration/base-dense-v2-500m").exists()


def test_performance_contract_rejects_old_b2_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_performance_module(tmp_path, monkeypatch)
    report = tmp_path / "artifacts/benchmarks/report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["recommendation"]["batch_size"] = 2
    report.write_text(json.dumps(payload), encoding="utf-8")
    import twen.v2_finalizer as finalizer

    monkeypatch.setattr(finalizer, "EXPECTED_PERFORMANCE_REPORT_SHA256", sha256_file(report))

    with pytest.raises(FinalizationError, match="batch_size must be 1"):
        _authenticate_performance(tmp_path)


def test_performance_contract_accepts_only_exact_expanded_b1_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_performance_module(tmp_path, monkeypatch)

    performance = _authenticate_performance(tmp_path)

    assert performance["recommendation"]["production_config"] == PRODUCTION_RUNTIME
    assert performance["ordinary"]["outer_checkpoint_layer_indices"] == []
    assert performance["ordinary"]["inner_checkpoint_layer_indices"] == []
    assert performance["alignment"]["outer_checkpoint_layer_indices"] == list(
        ALIGNMENT_OUTER_CHECKPOINT_LAYERS
    )
    assert performance["alignment"]["inner_checkpoint_layer_indices"] == list(
        ALIGNMENT_INNER_CHECKPOINT_LAYERS
    )
    assert performance["alignment"]["loss_chunk_tokens"] == 512


def test_performance_contract_rejects_production_enabled_false_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_performance_module(tmp_path, monkeypatch, production_enabled=False)

    with pytest.raises(FinalizationError, match="raw benchmark execution contract changed"):
        _authenticate_performance(tmp_path)


def test_cli_has_no_implicit_mtp_lr_or_quality_cooldown_decisions() -> None:
    from twen.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["config", "finalize-base-v2"])
    args = parser.parse_args(
        [
            "config",
            "finalize-base-v2",
            "--mtp-loss-weight",
            "0.1",
            "--adapter-lr",
            "0.0002",
            "--router-lr",
            "0.001",
            "--lora-lr",
            "0.0002",
            "--scale-lr",
            "0.001",
            "--quality-cooldown-prepared-manifest",
            "artifacts/data/quality/prepared/manifest.json",
            "--quality-cooldown-kd-manifest",
            "artifacts/data/quality/kd/manifest.json",
        ]
    )
    assert args.enable_web_launch is False
    assert args.quality_cooldown_prepared_manifest.endswith("prepared/manifest.json")
