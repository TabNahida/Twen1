"""Fail-closed finalization of the Base-v2 500M training definition.

The finalizer authenticates already-produced CPU/data/GPU evidence and writes
only a resolved YAML configuration, a fixed dashboard launch profile, and a
small certification bundle.  It never imports the training engine, constructs
a model or optimizer, initializes CUDA, or starts a training process.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from .config import ConfigError, TrainConfig, load_train_config
from .utils import atomic_write_json, atomic_write_text, sha256_file

FINALIZATION_SCHEMA_VERSION = 1
FINALIZATION_KIND = "twen_base_v2_500m_config_finalization"
FINALIZATION_COMPLETE_KIND = "twen_base_v2_500m_config_finalization_complete"
PIPELINE_COMPLETE_KIND = "twen_base_v2_500m_pipeline_complete"
PIPELINE_STATUS_KIND = "twen_base_v2_500m_pipeline_status"

RUN_ID = "base-dense-v2-500m"
PROFILE_ID = "base-dense-v2-500m"
OUTPUT_DIR = "runs/base-dense-v2-500m"
EXPECTED_TRAIN_TOKENS = 500_000_000
WARMUP_TOKENS = 5_000_000
DECAY_TOKENS = 50_000_000
STABLE_UNTIL_TOKENS = EXPECTED_TRAIN_TOKENS - DECAY_TOKENS
GLOBAL_BATCH_TOKENS = 262_144
SEQUENCE_LENGTH = 4_096
KD_TOP_K = 64
KD_TEMPERATURE = 2.0
QUALITY_COOLDOWN_TOKENS = 50_000_000
QUALITY_COOLDOWN_START_TOKENS = STABLE_UNTIL_TOKENS
LOSS_CHUNK_TOKENS = 512

EXPECTED_PERFORMANCE_REPORT_SHA256 = (
    "cf40eac976767681a704676bee960d8c220d700a847b73d62e1f2358ea15ab38"
)
ORDINARY_CASE = "b1-ordinary-ac0"
ALIGNMENT_CASE = "b1-alignment-ac8"
ORDINARY_OUTER_CHECKPOINT_LAYERS: tuple[int, ...] = ()
ALIGNMENT_OUTER_CHECKPOINT_LAYERS = (0, 3, 7, 10, 13, 16, 20, 23)
ORDINARY_INNER_CHECKPOINT_LAYERS: tuple[int, ...] = ()
ALIGNMENT_INNER_CHECKPOINT_LAYERS = (
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
# Compatibility aliases for callers that still use the old "token checkpoint"
# spelling.  These now describe the authenticated expanded inner-checkpoint set.
ORDINARY_TOKEN_CHECKPOINT_LAYERS = ORDINARY_INNER_CHECKPOINT_LAYERS
ALIGNMENT_TOKEN_CHECKPOINT_LAYERS = ALIGNMENT_INNER_CHECKPOINT_LAYERS
PRODUCTION_RUNTIME = {
    "micro_batch_size": 1,
    "activation_checkpointing": True,
    "activation_checkpointing_on_alignment_only": True,
    "activation_checkpoint_layer_count": None,
    "ordinary_outer_checkpoint_layer_count": 0,
    "alignment_outer_checkpoint_layer_count": 8,
    "ordinary_inner_checkpoint_layer_count": 0,
    "alignment_inner_checkpoint_layer_count": 16,
    "dense_transfer_execution": "expanded",
}
NUMERICAL_EVIDENCE_ROOT = "artifacts/audits/differentiable-fold-numerical-admission"
EXPANDED_NUMERICAL_MANIFEST = (
    f"{NUMERICAL_EVIDENCE_ROOT}/expanded_selective_checkpoint_full_graph_manifest.json"
)
EXPANDED_NUMERICAL_COMPLETE = (
    f"{NUMERICAL_EVIDENCE_ROOT}/EXPANDED_SELECTIVE_CHECKPOINT_FULL_GRAPH_COMPLETE"
)
FOLDED_NUMERICAL_MANIFEST = (
    f"{NUMERICAL_EVIDENCE_ROOT}/full_graph_v1_real_kd_accumulation_manifest.json"
)
FOLDED_NUMERICAL_COMPLETE = f"{NUMERICAL_EVIDENCE_ROOT}/FULL_GRAPH_V1_REAL_KD_ACCUMULATION_COMPLETE"
CURRENT_PRODUCTION_SOURCE_PATHS = {
    "config": "src/twen/config.py",
    "preflight": "src/twen/preflight.py",
    "engine": "src/twen/training/engine.py",
    "modules": "src/twen/modeling/modules.py",
    "benchmark": "scripts/benchmark_full_dense_graph.py",
    "mtp": "src/twen/modeling/mtp.py",
}


class FinalizationError(RuntimeError):
    """A required immutable input or operator decision was not satisfied."""


@dataclass(frozen=True, slots=True)
class FinalizationOptions:
    root: Path
    base_config: Path
    prepared_manifest: Path
    kd_manifest: Path
    kd_orchestration_complete: Path
    quality_cooldown_prepared_manifest: Path
    quality_cooldown_kd_manifest: Path
    pipeline_complete: Path
    fork_from: Path
    output_config: Path
    dashboard_config: Path
    evidence_dir: Path
    mtp_loss_weight: float
    adapter_lr: float
    router_lr: float
    lora_lr: float
    scale_lr: float
    enable_web_launch: bool = False

    @classmethod
    def repository_defaults(
        cls,
        *,
        root: str | Path,
        mtp_loss_weight: float,
        adapter_lr: float,
        router_lr: float,
        lora_lr: float,
        scale_lr: float,
        quality_cooldown_prepared_manifest: str | Path,
        quality_cooldown_kd_manifest: str | Path,
        enable_web_launch: bool = False,
        base_config: str | Path = "configs/base/dense-oracle.yaml",
        prepared_manifest: str | Path = "artifacts/data/base-v2-500m/manifest.json",
        kd_manifest: str | Path = "artifacts/data/base-v2-500m-kd/manifest.json",
        kd_orchestration_complete: str | Path = (
            "artifacts/data/base-v2-500m-kd-orchestration/COMPLETE"
        ),
        pipeline_complete: str | Path = "artifacts/data/base-v2-500m-pipeline/COMPLETE",
        fork_from: str | Path = ("runs/base-dense-v1/step-000000000383-milestone-complete"),
        output_config: str | Path = "configs/base/dense-v2-500m.yaml",
        dashboard_config: str | Path = "configs/web/dashboard.json",
        evidence_dir: str | Path = "artifacts/configuration/base-dense-v2-500m",
    ) -> FinalizationOptions:
        project = Path(root).expanduser().resolve()

        def resolve(value: str | Path, label: str) -> Path:
            raw = Path(value).expanduser()
            target = (raw if raw.is_absolute() else project / raw).resolve()
            try:
                target.relative_to(project)
            except ValueError as error:
                raise FinalizationError(
                    f"{label} must stay inside project root: {target}"
                ) from error
            return target

        return cls(
            root=project,
            base_config=resolve(base_config, "base config"),
            prepared_manifest=resolve(prepared_manifest, "prepared manifest"),
            kd_manifest=resolve(kd_manifest, "KD manifest"),
            kd_orchestration_complete=resolve(
                kd_orchestration_complete,
                "KD orchestration COMPLETE",
            ),
            quality_cooldown_prepared_manifest=resolve(
                quality_cooldown_prepared_manifest,
                "quality cooldown prepared manifest",
            ),
            quality_cooldown_kd_manifest=resolve(
                quality_cooldown_kd_manifest,
                "quality cooldown KD manifest",
            ),
            pipeline_complete=resolve(pipeline_complete, "pipeline COMPLETE"),
            fork_from=resolve(fork_from, "fork checkpoint"),
            output_config=resolve(output_config, "output config"),
            dashboard_config=resolve(dashboard_config, "dashboard config"),
            evidence_dir=resolve(evidence_dir, "evidence directory"),
            mtp_loss_weight=float(mtp_loss_weight),
            adapter_lr=float(adapter_lr),
            router_lr=float(router_lr),
            lora_lr=float(lora_lr),
            scale_lr=float(scale_lr),
            enable_web_launch=bool(enable_web_launch),
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _identity(root: Path, path: Path) -> dict[str, Any]:
    target = path.resolve()
    if not target.is_file():
        raise FinalizationError(f"required file is missing: {target}")
    return {
        "path": _relative(root, target),
        "size": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def _absolute_identity(path: Path) -> dict[str, Any]:
    target = path.resolve()
    if not target.is_file():
        raise FinalizationError(f"required file is missing: {target}")
    return {
        "path": str(target),
        "size": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def _relative_identity(directory: Path, path: Path) -> dict[str, Any]:
    target = path.resolve()
    root = directory.resolve()
    try:
        display = target.relative_to(root).as_posix()
    except ValueError as error:
        raise FinalizationError(f"required file escapes bundle root: {target}") from error
    if not target.is_file():
        raise FinalizationError(f"required file is missing: {target}")
    return {
        "path": display,
        "size": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FinalizationError(f"invalid or missing JSON object: {path}") from error
    if not isinstance(value, dict):
        raise FinalizationError(f"JSON root must be an object: {path}")
    return value


def _declared_identity_matches(
    root: Path,
    declared: object,
    expected_path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(declared, dict):
        raise FinalizationError(f"{label} identity is missing")
    raw_path = declared.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise FinalizationError(f"{label} identity has no path")
    candidate = Path(raw_path)
    candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
    expected = expected_path.resolve()
    if candidate != expected:
        raise FinalizationError(f"{label} path mismatch: expected {expected}, got {candidate}")
    actual = _identity(root, expected)
    if declared.get("size") != actual["size"] or declared.get("sha256") != actual["sha256"]:
        raise FinalizationError(f"{label} size/SHA256 changed")
    return actual


def _declared_project_identity(
    root: Path,
    declared: object,
    *,
    label: str,
) -> dict[str, Any]:
    """Authenticate a declared file identity without allowing paths outside root."""

    if not isinstance(declared, dict) or not isinstance(declared.get("path"), str):
        raise FinalizationError(f"{label} identity is missing")
    raw_path = Path(declared["path"])
    path = (raw_path if raw_path.is_absolute() else root / raw_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise FinalizationError(f"{label} must stay inside project root: {path}") from error
    return _declared_identity_matches(root, declared, path, label=label)


def _authenticate_evidence_bundle(
    root: Path,
    *,
    manifest_relative: str,
    complete_relative: str,
    manifest_kind: str,
    oracle_name: str,
    report_name: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Authenticate one immutable audit bundle and every declared payload file."""

    manifest_path = (root / manifest_relative).resolve()
    complete_path = (root / complete_relative).resolve()
    manifest = _load_object(manifest_path)
    complete = _load_object(complete_path)
    if (
        complete.get("schema_version") != 1
        or complete.get("kind") != "twen_complete_marker"
        or complete.get("status") != "complete"
        or complete.get("manifest") != manifest_path.name
        or complete.get("manifest_sha256") != sha256_file(manifest_path)
    ):
        raise FinalizationError(f"numerical evidence COMPLETE is invalid: {complete_path}")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != manifest_kind
        or manifest.get("status") != "complete"
        or manifest.get("optimizer_constructed") is not False
        or manifest.get("optimizer_step_calls") != 0
        or manifest.get("parameters_updated") is not False
    ):
        raise FinalizationError(f"numerical evidence manifest is invalid: {manifest_path}")

    expected_names = {
        oracle_name,
        report_name,
        oracle_name.removesuffix(".json") + ".py",
    }
    raw_files = manifest.get("files")
    if (
        not isinstance(raw_files, list)
        or len(raw_files) != len(expected_names)
        or not all(isinstance(item, dict) for item in raw_files)
        or {item.get("path") for item in raw_files if isinstance(item, dict)} != expected_names
    ):
        raise FinalizationError(
            f"numerical evidence manifest has an unexpected file inventory: {manifest_path}"
        )
    identities: dict[str, dict[str, Any]] = {}
    for raw in raw_files:
        if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
            raise FinalizationError(f"numerical evidence file identity is invalid: {manifest_path}")
        candidate = (manifest_path.parent / raw["path"]).resolve()
        if candidate.parent != manifest_path.parent or candidate.name not in expected_names:
            raise FinalizationError("numerical evidence file escaped its immutable bundle")
        actual = _identity(root, candidate)
        if raw.get("bytes") != actual["size"] or raw.get("sha256") != actual["sha256"]:
            raise FinalizationError(f"numerical evidence file size/SHA256 changed: {candidate}")
        identities[candidate.name] = actual
    oracle = _load_object(manifest_path.parent / oracle_name)
    return (
        {
            "manifest": _identity(root, manifest_path),
            "complete": _identity(root, complete_path),
            "report": identities[report_name],
            "oracle": identities[oracle_name],
            "source": identities[oracle_name.removesuffix(".json") + ".py"],
        },
        manifest,
        oracle,
    )


def _authenticate_numerical_evidence(root: Path) -> dict[str, Any]:
    """Require expanded PASS and preserve the real-KD folded FAIL.

    Expanded selective checkpointing preserves the historical formula and is
    the only admitted production path.  The folded audit is authenticated as a
    negative gate so no later throughput result can silently re-enable it.
    """

    expanded, expanded_manifest, expanded_oracle = _authenticate_evidence_bundle(
        root,
        manifest_relative=EXPANDED_NUMERICAL_MANIFEST,
        complete_relative=EXPANDED_NUMERICAL_COMPLETE,
        manifest_kind="twen_expanded_selective_checkpoint_full_graph_bundle",
        oracle_name="expanded_selective_checkpoint_full_graph_oracle.json",
        report_name="EXPANDED_SELECTIVE_CHECKPOINT_FULL_GRAPH_REPORT.md",
    )
    expanded_admission = {
        "differentiable_folded": False,
        "expanded_selective_checkpoint_numerically_admitted": True,
        "formula": "expanded",
        "production_enabled": True,
    }
    expanded_scope = expanded_oracle.get("scope")
    expanded_gates = expanded_oracle.get("gates")
    expanded_health = expanded_oracle.get("health")
    expanded_lineage = expanded_manifest.get("lineage")
    expanded_oracle_lineage = expanded_oracle.get("lineage")
    if not (
        expanded_manifest.get("admission") == expanded_admission
        and expanded_oracle.get("schema_version") == 1
        and expanded_oracle.get("kind")
        == "twen_expanded_selective_checkpoint_real_kd_full_graph_admission"
        and expanded_oracle.get("ok") is True
        and expanded_oracle.get("admission") == expanded_admission
        and isinstance(expanded_scope, dict)
        and expanded_scope.get("formula") == "expanded"
        and expanded_scope.get("batch_size") == 1
        and expanded_scope.get("sequence_length") == SEQUENCE_LENGTH
        and expanded_scope.get("optimizer_constructed") is False
        and expanded_scope.get("optimizer_step_calls") == 0
        and expanded_scope.get("parameters_updated") is False
        and isinstance(expanded_gates, dict)
        and expanded_gates.get("losses_bitwise") is True
        and isinstance(expanded_health, dict)
        and expanded_health.get("execution_mode_expanded") is True
        and expanded_health.get("outer_ac_disables_inner") is True
        and expanded_health.get("parameters_unchanged") is True
        and expanded_health.get("quick_bitwise_passed") is True
        and expanded_health.get("selective_core_calls_exact") is True
        and isinstance(expanded_lineage, dict)
        and isinstance(expanded_oracle_lineage, dict)
        and expanded_lineage.get("engine_source_sha256")
        == expanded_oracle_lineage.get("engine_source_sha256")
        and expanded_lineage.get("modules_source_sha256")
        == expanded_oracle_lineage.get("modules_source_sha256")
        and expanded_lineage.get("config_sha256") == expanded_oracle_lineage.get("config_sha256")
        and all(
            isinstance(expanded_lineage.get(name), str)
            and len(expanded_lineage[name]) == 64
            and set(expanded_lineage[name]) <= set("0123456789abcdef")
            for name in (
                "engine_source_sha256",
                "modules_source_sha256",
                "config_sha256",
            )
        )
    ):
        raise FinalizationError("expanded selective-checkpoint numerical PASS is invalid")

    folded, folded_manifest, folded_oracle = _authenticate_evidence_bundle(
        root,
        manifest_relative=FOLDED_NUMERICAL_MANIFEST,
        complete_relative=FOLDED_NUMERICAL_COMPLETE,
        manifest_kind="twen_v1_final_real_kd_fold_accumulation_oracle_bundle",
        oracle_name="full_graph_v1_real_kd_accumulation_oracle.json",
        report_name="FULL_GRAPH_V1_REAL_KD_ACCUMULATION_REPORT.md",
    )
    folded_admission = {
        "differentiable_folded_production_enabled": False,
        "folded_execution_status": "experimental_only",
        "strict_real_kd_accumulation": False,
        "v2_production_execution": "expanded",
    }
    folded_decision = folded_oracle.get("decision")
    folded_scope = folded_oracle.get("scope")
    folded_health = folded_oracle.get("health")
    if not (
        folded_manifest.get("admission") == folded_admission
        and folded_manifest.get("decision") == folded_decision
        and folded_oracle.get("schema_version") == 1
        and folded_oracle.get("kind") == "twen_v1_final_real_kd_full_graph_fold_accumulation_oracle"
        and folded_oracle.get("ok") is False
        and isinstance(folded_decision, dict)
        and folded_decision.get("differentiable_folded_production_enabled") is False
        and folded_decision.get("folded_execution_status") == "experimental_only"
        and folded_decision.get("v2_production_execution") == "expanded"
        and isinstance(folded_scope, dict)
        and folded_scope.get("formal_training") is False
        and folded_scope.get("microbatch_count") == 4
        and folded_scope.get("batch_size") == 1
        and folded_scope.get("sequence_length") == SEQUENCE_LENGTH
        and folded_scope.get("optimizer_constructed") is False
        and folded_scope.get("optimizer_step_calls") == 0
        and folded_scope.get("parameters_updated") is False
        and isinstance(folded_health, dict)
        and folded_health.get("parameters_unchanged") is True
    ):
        raise FinalizationError("folded real-KD numerical FAIL was changed or hidden")

    return {
        "production_execution": "expanded",
        "evidence_role": "historical_numerical_math_evidence",
        "historical_embedded_source_sha256": {
            "engine": expanded_lineage["engine_source_sha256"],
            "modules": expanded_lineage["modules_source_sha256"],
            "archived_config": expanded_lineage["config_sha256"],
        },
        "current_production_source_equality_required": False,
        "expanded": {**expanded, "status": "pass", "production_enabled": True},
        "folded": {
            **folded,
            "status": "fail_experimental_only",
            "production_enabled": False,
        },
    }


def _authenticate_current_production_sources(root: Path) -> dict[str, Any]:
    """Snapshot the current implementation independently of historical math evidence."""

    return {
        "evidence_role": "current_production_implementation",
        "files": {
            name: _identity(root, root / relative_path)
            for name, relative_path in CURRENT_PRODUCTION_SOURCE_PATHS.items()
        },
    }


def _validate_decisions(options: FinalizationOptions) -> None:
    values = {
        "mtp_loss_weight": options.mtp_loss_weight,
        "adapter_lr": options.adapter_lr,
        "router_lr": options.router_lr,
        "lora_lr": options.lora_lr,
        "scale_lr": options.scale_lr,
    }
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise FinalizationError(f"--{name.replace('_', '-')} must be finite and positive")
    if options.output_config.exists():
        raise FinalizationError(
            f"refusing to overwrite an existing finalized config: {options.output_config}"
        )
    if options.evidence_dir.exists():
        raise FinalizationError(
            f"refusing to overwrite an existing finalization bundle: {options.evidence_dir}"
        )
    run_dir = (options.root / OUTPUT_DIR).resolve()
    if run_dir.exists():
        raise FinalizationError(f"new v2 output directory must not already exist: {run_dir}")


def _load_pipeline_module(root: Path) -> ModuleType:
    source = root / "scripts/prepare_base_v2_500m.py"
    if not source.is_file():
        raise FinalizationError(f"500M pipeline contract source is missing: {source}")
    module_name = f"_twen_v2_pipeline_contract_{sha256_file(source)[:16]}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise FinalizationError(f"cannot load pipeline contract: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _authenticate_performance(
    root: Path,
    numerical_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate the pinned cf40 expanded-B1 report and its raw cases."""

    numerical = numerical_evidence or _authenticate_numerical_evidence(root)
    module = _load_pipeline_module(root)
    layout = module.Layout.repository_defaults(root)
    gate = module._performance_gate(layout)
    approval_gate = gate.get("approval")
    approval_reason = approval_gate.get("reason") if isinstance(approval_gate, dict) else None
    if gate.get("ready") is not True:
        raise FinalizationError(
            "performance report/approval/bundle is not ready: "
            f"{gate.get('reason') or approval_reason}"
        )
    report_path = Path(layout.performance_gate).resolve()
    report_identity = _identity(root, report_path)
    if report_identity["sha256"] != EXPECTED_PERFORMANCE_REPORT_SHA256:
        raise FinalizationError(
            "canonical performance report SHA256 changed: expected "
            f"{EXPECTED_PERFORMANCE_REPORT_SHA256}, got {report_identity['sha256']}"
        )
    report = _load_object(report_path)
    recommendation = report.get("recommendation")
    if not isinstance(recommendation, dict):
        raise FinalizationError("performance report has no recommendation")
    expected_recommendation = {
        "batch_size": 1,
        "ordinary_case": ORDINARY_CASE,
        "alignment_case": ALIGNMENT_CASE,
        "production_config": PRODUCTION_RUNTIME,
    }
    for field, expected in expected_recommendation.items():
        if recommendation.get(field) != expected:
            raise FinalizationError(
                f"performance recommendation {field} must be {expected!r}, "
                f"got {recommendation.get(field)!r}"
            )
    report_numerical = report.get("numerical_admission")
    if not isinstance(report_numerical, dict):
        raise FinalizationError("performance report has no numerical evidence contract")
    expanded_declared = report_numerical.get("expanded")
    folded_declared = report_numerical.get("folded")
    if not (
        report_numerical.get("production_reference_mode") == "expanded"
        and isinstance(expanded_declared, dict)
        and expanded_declared.get("status") == "pass"
        and isinstance(folded_declared, dict)
        and folded_declared.get("status") == "fail_experimental_only"
    ):
        raise FinalizationError("performance report changed the expanded/folded numerical gates")
    for name, declared, expected in (
        ("expanded", expanded_declared, numerical["expanded"]),
        ("folded", folded_declared, numerical["folded"]),
    ):
        _declared_identity_matches(
            root,
            declared.get("complete"),
            root / expected["complete"]["path"],
            label=f"performance {name} numerical COMPLETE",
        )
        _declared_identity_matches(
            root,
            declared.get("report"),
            root / expected["report"]["path"],
            label=f"performance {name} numerical report",
        )

    rows = report.get("rows")
    if not isinstance(rows, list):
        raise FinalizationError("performance report rows are missing")

    def selected(
        label: str,
        mode: str,
        outer_layers: tuple[int, ...],
        inner_layers: tuple[int, ...],
    ) -> dict[str, Any]:
        matches = [row for row in rows if isinstance(row, dict) and row.get("label") == label]
        if len(matches) != 1:
            raise FinalizationError(f"performance row {label!r} is not unique")
        row = matches[0]
        required = {
            "mode": mode,
            "batch_size": 1,
            "logical_tokens": SEQUENCE_LENGTH,
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
        }
        for field, expected in required.items():
            if row.get(field) != expected:
                raise FinalizationError(
                    f"performance row {label} {field} must be {expected!r}, got {row.get(field)!r}"
                )
        health = row.get("health")
        if not (
            isinstance(health, dict)
            and health.get("ok") is True
            and health.get("loss_finite") is True
            and health.get("gradients_finite") is True
            and health.get("missing_gradient_tensors") == 0
            and health.get("nonfinite_gradient_tensors") == 0
            and health.get("present_gradient_tensor_counts") == [72]
        ):
            raise FinalizationError(f"performance row {label} health contract failed")

        source_identity = _declared_project_identity(
            root,
            row.get("source"),
            label=f"performance row {label} source",
        )
        source = _load_object(root / source_identity["path"])
        runtime = source.get("runtime")
        batch = source.get("batch")
        mtp = source.get("mtp")
        reserve = source.get("optimizer_state_reserve")
        actual_state = source.get("dense_transfer_actual_state")
        execution = source.get("experimental_execution")
        if not (
            source.get("ok") is True
            and source.get("production_shape") is True
            and source.get("production_acceptance") is True
            and source.get("no_optimizer_created") is True
            and source.get("no_optimizer_steps") is True
            and source.get("optimizer_step_calls") == 0
            and isinstance(batch, dict)
            and batch.get("batch_size") == 1
            and batch.get("sequence_length") == SEQUENCE_LENGTH
            and batch.get("logical_tokens") == SEQUENCE_LENGTH
            and isinstance(runtime, dict)
            and runtime.get("activation_checkpoint_layer_count") == len(outer_layers)
            and runtime.get("activation_checkpoint_layer_indices") == list(outer_layers)
            and runtime.get("dense_transfer_checkpoint_layer_count_requested") == len(inner_layers)
            and runtime.get("dense_transfer_checkpoint_layer_count_effective") == len(inner_layers)
            and runtime.get("dense_transfer_checkpoint_layer_indices") == list(inner_layers)
            and runtime.get("dense_transfer_checkpoint_selection_policy")
            == "deterministic_evenly_spaced_outer_complement"
            and runtime.get("dense_transfer_outer_inner_disjoint") is True
            and runtime.get("dense_transfer_execution") == "expanded"
            and runtime.get("loss_chunk_tokens") == LOSS_CHUNK_TOKENS
            and runtime.get("loss_checkpoint_chunks") is True
            and runtime.get("teacher_cpu_offload") is True
            and isinstance(actual_state, dict)
            and actual_state.get("actual_checkpoint_layer_indices") == list(inner_layers)
            and actual_state.get("actual_execution_modes") == ["expanded"]
            and actual_state.get("all_modules_match_requested_state") is True
            and actual_state.get("outer_inner_disjoint") is True
            and isinstance(execution, dict)
            and execution.get("mode") == "expanded"
            and execution.get("production_enabled") is True
            and execution.get("selected_mode_numerical_status") == "admitted_production_reference"
            and isinstance(mtp, dict)
            and mtp.get("enabled") is True
            and mtp.get("frozen") is True
            and mtp.get("attention_implementation") == "sdpa"
            and mtp.get("loss_weight") == 0.1
            and isinstance(reserve, dict)
            and reserve.get("requested_gib") == 1.5
            and reserve.get("is_optimizer") is False
            and reserve.get("resident_during_all_iterations") is True
        ):
            raise FinalizationError(
                f"performance row {label} raw benchmark execution contract changed"
            )
        return {
            "label": label,
            "mode": mode,
            "batch_size": 1,
            "outer_checkpoint_layer_indices": list(outer_layers),
            "inner_checkpoint_layer_indices": list(inner_layers),
            "loss_chunk_tokens": LOSS_CHUNK_TOKENS,
            "production_tokens_per_second": row.get("production_tokens_per_second"),
            "minimum_estimated_headroom_gib": row.get("minimum_estimated_headroom_gib"),
            "minimum_nvml_physical_free_gib": row.get("minimum_nvml_physical_free_gib"),
            "source": source_identity,
        }

    ordinary = selected(
        ORDINARY_CASE,
        "ordinary",
        ORDINARY_OUTER_CHECKPOINT_LAYERS,
        ORDINARY_INNER_CHECKPOINT_LAYERS,
    )
    alignment = selected(
        ALIGNMENT_CASE,
        "alignment",
        ALIGNMENT_OUTER_CHECKPOINT_LAYERS,
        ALIGNMENT_INNER_CHECKPOINT_LAYERS,
    )
    return {
        "report": report_identity,
        "approval": _identity(root, Path(layout.performance_approval)),
        "manifest": _identity(root, Path(layout.performance_manifest)),
        "complete": _identity(root, Path(layout.performance_complete)),
        "pipeline_contract_source": _identity(root, root / "scripts/prepare_base_v2_500m.py"),
        "recommendation": recommendation,
        "ordinary": ordinary,
        "alignment": alignment,
    }


def _authenticate_pipeline_complete(
    options: FinalizationOptions,
    *,
    prepared_identity: dict[str, Any],
    audit_path: Path,
) -> dict[str, Any]:
    complete = _load_object(options.pipeline_complete)
    if (
        complete.get("schema_version") != 1
        or complete.get("kind") != PIPELINE_COMPLETE_KIND
        or complete.get("training_started") is not False
        or complete.get("gpu_kd_started") is not False
    ):
        raise FinalizationError("500M pipeline COMPLETE contract is invalid")
    _declared_identity_matches(
        options.root,
        complete.get("prepared_manifest"),
        options.prepared_manifest,
        label="pipeline prepared manifest",
    )
    audit_identity = _declared_identity_matches(
        options.root,
        complete.get("accepted_attestation"),
        audit_path,
        label="pipeline accepted audit attestation",
    )
    declared_prepared = complete["prepared_manifest"]
    if (
        declared_prepared.get("size") != prepared_identity["size"]
        or declared_prepared.get("sha256") != prepared_identity["sha256"]
    ):
        raise FinalizationError("pipeline COMPLETE does not bind this prepared manifest")
    status_declared = complete.get("status")
    if not isinstance(status_declared, dict) or not isinstance(status_declared.get("path"), str):
        raise FinalizationError("pipeline COMPLETE has no status identity")
    raw_status = Path(status_declared["path"])
    status_path = (raw_status if raw_status.is_absolute() else options.root / raw_status).resolve()
    expected_status_path = options.pipeline_complete.parent / "status.json"
    if status_path != expected_status_path.resolve():
        raise FinalizationError(
            f"pipeline COMPLETE status path mismatch: expected {expected_status_path}, "
            f"got {status_path}"
        )
    status_identity = _declared_identity_matches(
        options.root, status_declared, status_path, label="pipeline status"
    )
    status = _load_object(status_path)
    if (
        status.get("schema_version") != 1
        or status.get("kind") != PIPELINE_STATUS_KIND
        or status.get("status") != "complete"
        or status.get("training_started") is not False
        or status.get("gpu_kd_started") is not False
    ):
        raise FinalizationError("500M pipeline final status is invalid")
    _declared_identity_matches(
        options.root,
        status.get("prepared_manifest"),
        options.prepared_manifest,
        label="pipeline status prepared manifest",
    )
    _declared_identity_matches(
        options.root,
        status.get("accepted_attestation"),
        audit_path,
        label="pipeline status accepted attestation",
    )
    return {
        "complete": _identity(options.root, options.pipeline_complete),
        "status": status_identity,
        "accepted_attestation": audit_identity,
    }


def _authenticate_kd_orchestration(
    options: FinalizationOptions,
    *,
    prepared_identity: dict[str, Any],
    kd_identity: dict[str, Any],
    audit_path: Path,
) -> dict[str, Any]:
    """Require the successful generate-kd -> index-kd certification chain."""

    from .kd_orchestration import (
        COMPLETE_KIND,
        MANIFEST_KIND,
        STATUS_KIND,
        Layout,
        verify_orchestration_complete,
    )

    state_root = options.kd_orchestration_complete.parent
    if options.kd_orchestration_complete != state_root / "COMPLETE":
        raise FinalizationError("KD orchestration proof must be named COMPLETE")
    defaults = Layout.repository_defaults(options.root)
    layout = dataclasses.replace(
        defaults,
        base_config=options.base_config,
        prepared_manifest=options.prepared_manifest,
        pipeline_complete=options.pipeline_complete,
        output_root=options.kd_manifest.parent,
        state_root=state_root,
        stop_file=options.kd_manifest.parent / "STOP",
    )
    try:
        verified = verify_orchestration_complete(layout)
    except (OSError, ValueError, RuntimeError) as error:
        raise FinalizationError(
            f"KD orchestration COMPLETE failed authentication: {error}"
        ) from error
    if verified is None:
        raise FinalizationError(
            f"KD orchestration is not complete: {options.kd_orchestration_complete}"
        )
    verified_kd = verified.get("kd_manifest")
    if not isinstance(verified_kd, dict) or (
        verified_kd.get("size") != kd_identity["size"]
        or verified_kd.get("sha256") != kd_identity["sha256"]
    ):
        raise FinalizationError("KD orchestration does not bind the current KD manifest")

    complete = _load_object(options.kd_orchestration_complete)
    if (
        set(complete)
        != {
            "schema_version",
            "kind",
            "manifest",
            "kd_manifest",
            "prepared_manifest",
            "completed_at",
            "training_started",
            "optimizer_created",
        }
        or complete.get("schema_version") != 1
        or complete.get("kind") != COMPLETE_KIND
        or complete.get("training_started") is not False
        or complete.get("optimizer_created") is not False
    ):
        raise FinalizationError("KD orchestration COMPLETE fields differ from the locked schema")
    completed_prepared = _declared_identity_matches(
        options.root,
        complete.get("prepared_manifest"),
        options.prepared_manifest,
        label="KD orchestration completed prepared manifest",
    )
    if (
        completed_prepared["size"] != prepared_identity["size"]
        or completed_prepared["sha256"] != prepared_identity["sha256"]
    ):
        raise FinalizationError("KD orchestration does not bind the current prepared manifest")
    _declared_identity_matches(
        options.root,
        complete.get("kd_manifest"),
        options.kd_manifest,
        label="KD orchestration completed KD manifest",
    )

    manifest_path = state_root / "MANIFEST.json"
    manifest = _load_object(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != MANIFEST_KIND
        or manifest.get("training_started") is not False
        or manifest.get("optimizer_created") is not False
    ):
        raise FinalizationError("KD orchestration manifest contract is invalid")
    _declared_identity_matches(
        options.root,
        manifest.get("prepared_manifest"),
        options.prepared_manifest,
        label="KD orchestration prepared manifest",
    )
    _declared_identity_matches(
        options.root,
        manifest.get("kd_manifest"),
        options.kd_manifest,
        label="KD orchestration KD manifest",
    )
    pipeline_identity = _declared_identity_matches(
        options.root,
        manifest.get("pipeline_complete"),
        options.pipeline_complete,
        label="KD orchestration pipeline COMPLETE",
    )
    base_config_identity = _declared_identity_matches(
        options.root,
        manifest.get("base_config"),
        options.base_config,
        label="KD orchestration Base config",
    )
    audit_identity = _declared_identity_matches(
        options.root,
        manifest.get("audit_attestation"),
        audit_path,
        label="KD orchestration audit attestation",
    )

    status_path = state_root / "status.json"
    _declared_identity_matches(
        options.root,
        manifest.get("status"),
        status_path,
        label="KD orchestration final status",
    )
    status = _load_object(status_path)
    progress = status.get("progress")
    if (
        status.get("schema_version") != 1
        or status.get("kind") != STATUS_KIND
        or status.get("status") != "complete"
        or status.get("phase") is not None
        or status.get("error") is not None
        or status.get("training_started") is not False
        or status.get("optimizer_created") is not False
        or not isinstance(progress, dict)
        or progress.get("completed_shards") != progress.get("total_shards")
        or progress.get("completed_tokens") != progress.get("total_tokens")
        or progress.get("eta_seconds") != 0.0
    ):
        raise FinalizationError("KD orchestration final status is not a complete safe run")
    _declared_identity_matches(
        options.root,
        status.get("prepared_manifest"),
        options.prepared_manifest,
        label="KD status prepared manifest",
    )
    _declared_identity_matches(
        options.root,
        status.get("kd_manifest"),
        options.kd_manifest,
        label="KD status KD manifest",
    )
    _declared_identity_matches(
        options.root,
        status.get("audit_attestation"),
        audit_path,
        label="KD status audit attestation",
    )
    history = manifest.get("history")
    if not isinstance(history, list):
        raise FinalizationError("KD orchestration has no phase history")
    completed = [
        item.get("name")
        for item in history
        if isinstance(item, dict)
        and item.get("status") == "complete"
        and item.get("exit_code") == 0
    ]
    if len(completed) < 2 or completed[-2:] != ["generate-kd", "index-kd"]:
        raise FinalizationError(
            "KD orchestration must end with successful generate-kd and index-kd phases"
        )
    return {
        "complete": _identity(options.root, options.kd_orchestration_complete),
        "manifest": _identity(options.root, manifest_path),
        "status": _identity(options.root, status_path),
        "pipeline_complete": pipeline_identity,
        "base_config": base_config_identity,
        "audit_attestation": audit_identity,
        "completed_phases": completed[-2:],
    }


def _authenticate_quality_policy_bundle(
    options: FinalizationOptions,
    *,
    policy_path: Path,
    policy: dict[str, Any],
) -> dict[str, Any]:
    from .data.quality_policy import (
        QUALITY_POLICY_AUDIT_FILENAME,
        QUALITY_POLICY_AUDIT_KIND,
        QUALITY_POLICY_BUNDLE_KIND,
        QUALITY_POLICY_COMPLETE_FILENAME,
        QUALITY_POLICY_COMPLETE_KIND,
        QUALITY_POLICY_FILENAME,
        QUALITY_POLICY_MANIFEST_FILENAME,
        QUALITY_POLICY_REPORT_FILENAME,
    )

    policy_root = policy_path.parent
    if policy_path.name != QUALITY_POLICY_FILENAME:
        raise FinalizationError(
            f"quality cooldown policy must be the published {QUALITY_POLICY_FILENAME}"
        )
    expected_names = {
        QUALITY_POLICY_FILENAME,
        QUALITY_POLICY_AUDIT_FILENAME,
        QUALITY_POLICY_REPORT_FILENAME,
        QUALITY_POLICY_MANIFEST_FILENAME,
        QUALITY_POLICY_COMPLETE_FILENAME,
    }
    actual_names = {path.name for path in policy_root.iterdir()}
    if actual_names != expected_names or any(
        path.is_symlink() or not path.is_file() for path in policy_root.iterdir()
    ):
        raise FinalizationError("quality cooldown policy bundle has an open or unsafe inventory")

    audit_path = policy_root / QUALITY_POLICY_AUDIT_FILENAME
    report_path = policy_root / QUALITY_POLICY_REPORT_FILENAME
    manifest_path = policy_root / QUALITY_POLICY_MANIFEST_FILENAME
    complete_path = policy_root / QUALITY_POLICY_COMPLETE_FILENAME
    audit = _load_object(audit_path)
    if (
        audit.get("schema_version") != 1
        or audit.get("kind") != QUALITY_POLICY_AUDIT_KIND
        or audit.get("policy_id") != policy["policy_id"]
        or audit.get("approved_for_quality_cooldown") is not True
        or audit.get("required_cooldown_tokens") != QUALITY_COOLDOWN_TOKENS
        or audit.get("training_started") is not False
        or audit.get("teacher_kd_started") is not False
    ):
        raise FinalizationError("quality cooldown policy audit contract is invalid")
    manifest = _load_object(manifest_path)
    expected_files = [
        _relative_identity(policy_root, path) for path in (policy_path, audit_path, report_path)
    ]
    if (
        set(manifest)
        != {
            "schema_version",
            "kind",
            "policy_id",
            "selection_plan_sha256",
            "approved_for_quality_cooldown",
            "files",
            "training_started",
            "teacher_kd_started",
        }
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != QUALITY_POLICY_BUNDLE_KIND
        or manifest.get("policy_id") != policy["policy_id"]
        or manifest.get("selection_plan_sha256") != audit.get("selection_plan_sha256")
        or manifest.get("approved_for_quality_cooldown") is not True
        or manifest.get("files") != expected_files
        or manifest.get("training_started") is not False
        or manifest.get("teacher_kd_started") is not False
    ):
        raise FinalizationError("quality cooldown policy MANIFEST contract is invalid")
    complete = _load_object(complete_path)
    if (
        set(complete)
        != {
            "schema_version",
            "kind",
            "policy_id",
            "manifest",
            "manifest_sha256",
            "approved_for_quality_cooldown",
            "training_started",
            "teacher_kd_started",
        }
        or complete.get("schema_version") != 1
        or complete.get("kind") != QUALITY_POLICY_COMPLETE_KIND
        or complete.get("policy_id") != policy["policy_id"]
        or complete.get("manifest") != QUALITY_POLICY_MANIFEST_FILENAME
        or complete.get("manifest_sha256") != sha256_file(manifest_path)
        or complete.get("approved_for_quality_cooldown") is not True
        or complete.get("training_started") is not False
        or complete.get("teacher_kd_started") is not False
    ):
        raise FinalizationError("quality cooldown policy COMPLETE contract is invalid")
    return {
        "policy": _identity(options.root, policy_path),
        "audit": _identity(options.root, audit_path),
        "report": _identity(options.root, report_path),
        "manifest": _identity(options.root, manifest_path),
        "complete": _identity(options.root, complete_path),
        "selection_plan_sha256": audit["selection_plan_sha256"],
    }


def _validate_locked_quality_source_mix(summary: Any) -> dict[str, int]:
    from .data.quality_policy import DEFAULT_QUALITY_SOURCE_TOKEN_TARGETS

    source_mix = dict(summary.source_mix_token_counts)
    required_mix = dict(DEFAULT_QUALITY_SOURCE_TOKEN_TARGETS)
    if set(source_mix) != set(required_mix) or any(
        source_mix[source_id] < minimum for source_id, minimum in required_mix.items()
    ):
        raise FinalizationError(
            "quality cooldown source mix does not meet the locked six-source minimum quotas"
        )
    return source_mix


def _authenticate_quality_cooldown_publication(
    options: FinalizationOptions,
    *,
    primary_prepared_identity: dict[str, Any],
    primary_kd_identity: dict[str, Any],
    cooldown_prepared: Any,
    cooldown_kd: Any,
    cooldown_prepared_identity: dict[str, Any],
    cooldown_kd_identity: dict[str, Any],
    summary: Any,
) -> dict[str, Any]:
    """Bind finalization to the approved policy and atomically published bundle."""

    from .data.cooldown import (
        QUALITY_COOLDOWN_BUNDLE_KIND,
        QUALITY_COOLDOWN_COMPLETE_KIND,
        _assert_exact_bundle_tree,
        _validate_policy,
    )

    prepared_path = options.quality_cooldown_prepared_manifest
    kd_path = options.quality_cooldown_kd_manifest
    bundle_root = prepared_path.parent.parent
    if (
        prepared_path != bundle_root / "prepared/manifest.json"
        or kd_path != bundle_root / "kd/manifest.json"
    ):
        raise FinalizationError(
            "quality cooldown prepared/KD manifests must share one published bundle root"
        )
    bundle_path = bundle_root / "manifest.json"
    complete_path = bundle_root / "COMPLETE"
    bundle = _load_object(bundle_path)
    complete = _load_object(complete_path)
    if (
        set(complete)
        != {
            "schema_version",
            "kind",
            "status",
            "manifest",
            "training_started",
            "gpu_kd_started",
        }
        or complete.get("schema_version") != 1
        or complete.get("kind") != QUALITY_COOLDOWN_COMPLETE_KIND
        or complete.get("status") != "complete"
        or complete.get("manifest") != _relative_identity(bundle_root, bundle_path)
        or complete.get("training_started") is not False
        or complete.get("gpu_kd_started") is not False
    ):
        raise FinalizationError("quality cooldown COMPLETE contract is invalid")
    expected_bundle_fields = {
        "schema_version",
        "kind",
        "status",
        "fingerprint",
        "policy_id",
        "required_cooldown_tokens",
        "temperature",
        "inputs",
        "outputs",
        "dataset_fingerprint",
        "ordered_shard_ids",
        "source_mix_token_counts",
        "sequence_count",
        "token_count",
        "hardlinks",
        "training_started",
        "gpu_kd_started",
    }
    bundle_temperature = bundle.get("temperature")
    if (
        set(bundle) != expected_bundle_fields
        or bundle.get("schema_version") != 1
        or bundle.get("kind") != QUALITY_COOLDOWN_BUNDLE_KIND
        or bundle.get("status") != "complete"
        or bundle.get("required_cooldown_tokens") != QUALITY_COOLDOWN_TOKENS
        or isinstance(bundle_temperature, bool)
        or not isinstance(bundle_temperature, (int, float))
        or not math.isclose(
            float(bundle_temperature),
            KD_TEMPERATURE,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or bundle.get("training_started") is not False
        or bundle.get("gpu_kd_started") is not False
    ):
        raise FinalizationError("quality cooldown bundle contract is invalid")
    inputs = bundle.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "primary_prepared",
        "primary_kd",
        "selection_policy",
    }:
        raise FinalizationError("quality cooldown bundle input identities are invalid")
    expected_primary_prepared = _absolute_identity(options.prepared_manifest)
    expected_primary_kd = _absolute_identity(options.kd_manifest)
    if (
        inputs.get("primary_prepared") != expected_primary_prepared
        or inputs.get("primary_kd") != expected_primary_kd
        or expected_primary_prepared["size"] != primary_prepared_identity["size"]
        or expected_primary_prepared["sha256"] != primary_prepared_identity["sha256"]
        or expected_primary_kd["size"] != primary_kd_identity["size"]
        or expected_primary_kd["sha256"] != primary_kd_identity["sha256"]
    ):
        raise FinalizationError("quality cooldown bundle does not bind the primary manifests")
    policy_identity = _declared_project_identity(
        options.root,
        inputs.get("selection_policy"),
        label="quality cooldown selection policy",
    )
    policy_path = options.root / policy_identity["path"]
    policy = _validate_policy(
        policy_path,
        primary_prepared_sha256=primary_prepared_identity["sha256"],
        primary_kd_sha256=primary_kd_identity["sha256"],
        required_cooldown_tokens=QUALITY_COOLDOWN_TOKENS,
    )
    policy_bundle = _authenticate_quality_policy_bundle(
        options,
        policy_path=policy_path,
        policy=policy,
    )
    lineage = cooldown_prepared.lineage
    contract = lineage.get("quality_cooldown") if isinstance(lineage, dict) else None
    if (
        not isinstance(contract, dict)
        or contract.get("selection_policy_sha256") != policy_identity["sha256"]
        or contract.get("selection_policy_id") != policy["policy_id"]
    ):
        raise FinalizationError("quality cooldown lineage does not bind the approved policy")

    source_mix = _validate_locked_quality_source_mix(summary)
    ordered_ids = list(summary.selected_shard_ids)
    if (
        bundle.get("policy_id") != policy["policy_id"]
        or bundle.get("dataset_fingerprint") != summary.cooldown_dataset_fingerprint
        or bundle.get("ordered_shard_ids") != ordered_ids
        or bundle.get("source_mix_token_counts") != source_mix
        or bundle.get("sequence_count") != summary.sequence_count
        or bundle.get("token_count") != summary.token_count
        or [item["shard_id"] for item in policy["ordered_shards"]] != ordered_ids
        or policy["declared_source_mix_token_counts"] != source_mix
    ):
        raise FinalizationError("quality cooldown bundle differs from its approved selection")
    expected_outputs = {
        "prepared": _relative_identity(bundle_root, prepared_path),
        "teacher_kd": _relative_identity(bundle_root, kd_path),
    }
    if bundle.get("outputs") != expected_outputs or (
        expected_outputs["prepared"]["size"] != cooldown_prepared_identity["size"]
        or expected_outputs["prepared"]["sha256"] != cooldown_prepared_identity["sha256"]
        or expected_outputs["teacher_kd"]["size"] != cooldown_kd_identity["size"]
        or expected_outputs["teacher_kd"]["sha256"] != cooldown_kd_identity["sha256"]
    ):
        raise FinalizationError("quality cooldown bundle output identities changed")
    hardlinks = bundle.get("hardlinks")
    if not isinstance(hardlinks, list) or len(hardlinks) != 2 * len(ordered_ids):
        raise FinalizationError("quality cooldown hardlink inventory is incomplete")
    for index, item in enumerate(hardlinks):
        if not isinstance(item, dict) or item.get("same_inode") is not True:
            raise FinalizationError(f"quality cooldown hardlink {index} is invalid")
        source = Path(str(item.get("source", ""))).resolve()
        destination_value = Path(str(item.get("destination", "")))
        destination = (bundle_root / destination_value).resolve()
        if (
            destination_value.is_absolute()
            or not destination.is_relative_to(bundle_root)
            or source.is_symlink()
            or destination.is_symlink()
            or not source.is_file()
            or not destination.is_file()
            or not source.samefile(destination)
            or source.stat().st_dev != item.get("device")
            or source.stat().st_ino != item.get("inode")
            or source.stat().st_size != item.get("size")
        ):
            raise FinalizationError(f"quality cooldown hardlink {index} identity changed")
    _assert_exact_bundle_tree(bundle_root, cooldown_prepared, cooldown_kd)
    return {
        "bundle": _identity(options.root, bundle_path),
        "complete": _identity(options.root, complete_path),
        "policy_bundle": policy_bundle,
        "fingerprint": bundle["fingerprint"],
    }


def _authenticate_data(options: FinalizationOptions, base: TrainConfig) -> dict[str, Any]:
    """Inspect structural lineage now; the later preflight performs one full hash scan."""

    from .data import (
        read_prepared_manifest,
        validate_base_audit_attestation,
        validate_kd_corpus_coverage,
        validate_quality_cooldown_subset,
    )
    from .data.teacher_kd import (
        KD_GENERATOR_SOURCE_SHA256,
        TeacherKDCorpusManifest,
    )

    prepared_identity = _identity(options.root, options.prepared_manifest)
    prepared = read_prepared_manifest(options.prepared_manifest)
    if prepared.sequence_length != SEQUENCE_LENGTH:
        raise FinalizationError(
            f"prepared sequence length must be {SEQUENCE_LENGTH}, got {prepared.sequence_length}"
        )
    if prepared.token_count < EXPECTED_TRAIN_TOKENS:
        raise FinalizationError(
            f"prepared corpus has {prepared.token_count:,} tokens; at least "
            f"{EXPECTED_TRAIN_TOKENS:,} are required"
        )
    if prepared.tokenizer_sha256 != base.sources.tokenizer.manifest_sha256:
        raise FinalizationError("prepared tokenizer SHA differs from the pinned Base tokenizer")
    lineage = prepared.lineage
    if not isinstance(lineage, dict):
        raise FinalizationError("prepared corpus has no authenticated audit lineage")
    if (
        lineage.get("kind") != "authenticated_extracted_corpus"
        or lineage.get("role") != "train"
        or lineage.get("ready_for_training") is not True
        or lineage.get("research_only") is not False
        or lineage.get("pending_audits") != []
    ):
        raise FinalizationError("prepared corpus is not audit-attested ready_for_training")
    audit_lineage = lineage.get("audit_attestation")
    if not isinstance(audit_lineage, dict) or not isinstance(audit_lineage.get("path"), str):
        raise FinalizationError("prepared lineage has no audit attestation identity")
    raw_audit = Path(audit_lineage["path"])
    audit_path = (raw_audit if raw_audit.is_absolute() else options.root / raw_audit).resolve()
    try:
        audit_path.relative_to(options.root)
    except ValueError as error:
        raise FinalizationError("audit attestation must stay inside project root") from error
    audit = dict(validate_base_audit_attestation(audit_path))
    audit_identity = _identity(options.root, audit_path)
    if (
        audit.get("ready_for_training") is not True
        or audit_lineage.get("sha256") != audit_identity["sha256"]
        or audit_lineage.get("attestation_fingerprint") != audit.get("attestation_fingerprint")
        or audit_lineage.get("gates") != audit.get("gates")
    ):
        raise FinalizationError("prepared audit lineage does not bind the accepted attestation")
    pipeline = _authenticate_pipeline_complete(
        options,
        prepared_identity=prepared_identity,
        audit_path=audit_path,
    )

    kd_identity = _identity(options.root, options.kd_manifest)
    kd_raw = _load_object(options.kd_manifest)
    kd = TeacherKDCorpusManifest.from_dict(kd_raw)
    if kd.generator_source_sha256 != KD_GENERATOR_SOURCE_SHA256:
        raise FinalizationError("KD generator source changed; regenerate top-64 KD")
    validate_kd_corpus_coverage(kd, prepared)
    if (
        kd.top_k != KD_TOP_K
        or not math.isclose(kd.temperature, KD_TEMPERATURE, rel_tol=0.0, abs_tol=1e-12)
        or kd.token_count != prepared.token_count
        or kd.sequence_count != prepared.sequence_count
    ):
        raise FinalizationError("KD corpus is not complete top-64 coverage of prepared 500M data")
    teacher = base.sources.teacher
    if (
        kd.teacher_model_id != teacher.model_id
        or kd.teacher_revision != teacher.revision
        or kd.teacher_model_sha256 != teacher.manifest_sha256
        or kd.tokenizer_sha256 != base.sources.tokenizer.manifest_sha256
    ):
        raise FinalizationError("KD teacher/tokenizer source identity differs from Base pins")
    kd_orchestration = _authenticate_kd_orchestration(
        options,
        prepared_identity=prepared_identity,
        kd_identity=kd_identity,
        audit_path=audit_path,
    )

    cooldown_prepared_identity = _identity(options.root, options.quality_cooldown_prepared_manifest)
    cooldown_prepared = read_prepared_manifest(options.quality_cooldown_prepared_manifest)
    cooldown_kd_identity = _identity(options.root, options.quality_cooldown_kd_manifest)
    cooldown_kd = TeacherKDCorpusManifest.from_dict(
        _load_object(options.quality_cooldown_kd_manifest)
    )
    validate_kd_corpus_coverage(cooldown_kd, cooldown_prepared)
    if (
        cooldown_prepared.sequence_length != SEQUENCE_LENGTH
        or cooldown_prepared.tokenizer_sha256 != base.sources.tokenizer.manifest_sha256
        or cooldown_prepared.dataset_fingerprint == prepared.dataset_fingerprint
        or cooldown_kd.generator_source_sha256 != KD_GENERATOR_SOURCE_SHA256
        or cooldown_kd.top_k != KD_TOP_K
        or not math.isclose(
            cooldown_kd.temperature,
            KD_TEMPERATURE,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or cooldown_kd.teacher_model_id != teacher.model_id
        or cooldown_kd.teacher_revision != teacher.revision
        or cooldown_kd.teacher_model_sha256 != teacher.manifest_sha256
        or cooldown_kd.tokenizer_sha256 != base.sources.tokenizer.manifest_sha256
    ):
        raise FinalizationError(
            "quality cooldown prepared/KD identities differ from the Base 500M contract"
        )
    cooldown_summary = validate_quality_cooldown_subset(
        prepared,
        kd,
        cooldown_prepared,
        cooldown_kd,
        primary_prepared_manifest_sha256=prepared_identity["sha256"],
        primary_kd_manifest_sha256=kd_identity["sha256"],
        required_cooldown_tokens=QUALITY_COOLDOWN_TOKENS,
    )
    if cooldown_summary.token_count < QUALITY_COOLDOWN_TOKENS:
        raise FinalizationError("quality cooldown subset does not cover the final 50M tokens")
    cooldown_publication = _authenticate_quality_cooldown_publication(
        options,
        primary_prepared_identity=prepared_identity,
        primary_kd_identity=kd_identity,
        cooldown_prepared=cooldown_prepared,
        cooldown_kd=cooldown_kd,
        cooldown_prepared_identity=cooldown_prepared_identity,
        cooldown_kd_identity=cooldown_kd_identity,
        summary=cooldown_summary,
    )
    return {
        "prepared": prepared_identity,
        "kd": kd_identity,
        "audit": audit_identity,
        "pipeline": pipeline,
        "kd_orchestration": kd_orchestration,
        "dataset_fingerprint": prepared.dataset_fingerprint,
        "sequence_count": prepared.sequence_count,
        "token_count": prepared.token_count,
        "audit_attestation_fingerprint": audit.get("attestation_fingerprint"),
        "audit_gates": audit.get("gates"),
        "kd_generator_source_sha256": kd.generator_source_sha256,
        "quality_cooldown": {
            "prepared": cooldown_prepared_identity,
            "kd": cooldown_kd_identity,
            "start_tokens": QUALITY_COOLDOWN_START_TOKENS,
            "required_tokens": QUALITY_COOLDOWN_TOKENS,
            "selection_policy_id": cooldown_summary.selection_policy_id,
            "dataset_fingerprint": cooldown_summary.cooldown_dataset_fingerprint,
            "selected_shard_ids": list(cooldown_summary.selected_shard_ids),
            "source_mix_token_counts": [
                list(item) for item in cooldown_summary.source_mix_token_counts
            ],
            "sequence_count": cooldown_summary.sequence_count,
            "token_count": cooldown_summary.token_count,
            "publication": cooldown_publication,
        },
    }


def _authenticate_fork(options: FinalizationOptions, base: TrainConfig) -> dict[str, Any]:
    from .runtime.checkpoint import CheckpointManager

    manager = CheckpointManager(options.fork_from.parent, rank=0, world_size=1)
    metadata = manager.verify(options.fork_from)
    if (
        metadata.get("run_id") != "base-dense-v1"
        or metadata.get("stage") != "dense-oracle"
        or metadata.get("kind") != "milestone"
        or metadata.get("tag") != "complete"
        or metadata.get("rollback_applied") is not False
        or int(metadata.get("committed_tokens", -1)) < 100_000_000
    ):
        raise FinalizationError("fork checkpoint is not the completed Base dense v1 milestone")
    source_manifests = metadata.get("extra", {}).get("source_manifests")
    expected_sources = {
        "backbone": base.sources.backbone.manifest_sha256,
        "donor": base.sources.donor.manifest_sha256,
        "teacher": base.sources.teacher.manifest_sha256,
        "tokenizer": base.sources.tokenizer.manifest_sha256,
        "folded_experts": None,
    }
    if source_manifests != expected_sources:
        raise FinalizationError("fork checkpoint source manifests differ from the Base config")
    return {
        "path": _relative(options.root, options.fork_from),
        "complete": _identity(options.root, options.fork_from / "COMPLETE"),
        "manifest": _identity(options.root, options.fork_from / "manifest.json"),
        "metadata": _identity(options.root, options.fork_from / "metadata.json"),
        "run_id": metadata.get("run_id"),
        "stage": metadata.get("stage"),
        "global_step": metadata.get("global_step"),
        "committed_tokens": metadata.get("committed_tokens"),
        "tag": metadata.get("tag"),
    }


def _build_config(
    options: FinalizationOptions,
    base: TrainConfig,
    data: dict[str, Any],
) -> TrainConfig:
    config = copy.deepcopy(base)
    config.run_id = RUN_ID
    config.track = "base"
    config.stage = "dense-oracle"
    config.architecture.expert_initialization = "donor"
    config.architecture.random_expert_seed = 1701
    config.architecture.active_student_layers = None
    config.data.manifest_path = data["prepared"]["path"]
    config.data.manifest_sha256 = data["prepared"]["sha256"]
    config.data.teacher_kd_manifest_path = data["kd"]["path"]
    config.data.teacher_kd_manifest_sha256 = data["kd"]["sha256"]
    cooldown = data["quality_cooldown"]
    config.data.quality_cooldown_manifest_path = cooldown["prepared"]["path"]
    config.data.quality_cooldown_manifest_sha256 = cooldown["prepared"]["sha256"]
    config.data.quality_cooldown_teacher_kd_manifest_path = cooldown["kd"]["path"]
    config.data.quality_cooldown_teacher_kd_manifest_sha256 = cooldown["kd"]["sha256"]
    config.data.quality_cooldown_start_tokens = QUALITY_COOLDOWN_START_TOKENS
    config.data.max_sequence_length = SEQUENCE_LENGTH
    config.data.micro_batch_size = 1
    config.data.global_batch_tokens = GLOBAL_BATCH_TOKENS
    config.data.shuffle_seed = 3407
    config.data.num_workers = 4
    config.data.teacher_top_k = KD_TOP_K
    config.losses.ntp = 1.0
    config.losses.mtp = options.mtp_loss_weight
    config.losses.teacher_kd = 1.0
    config.losses.hidden_alignment = 0.1
    config.losses.anchor_kl = 0.1
    config.losses.dense_oracle = 0.0
    config.losses.router_supervision = 0.0
    config.losses.load_balance = 0.0
    config.losses.router_z = 0.0
    config.losses.kd_temperature = KD_TEMPERATURE
    config.losses.dense_oracle_batch_fraction = 0.0
    config.losses.hidden_alignment_batch_fraction = 0.05
    config.optimizer.adapter_lr = options.adapter_lr
    config.optimizer.router_lr = options.router_lr
    config.optimizer.lora_lr = options.lora_lr
    config.optimizer.scale_lr = options.scale_lr
    config.optimizer.weight_decay = 0.01
    config.optimizer.adam_beta1 = 0.9
    config.optimizer.adam_beta2 = 0.95
    config.optimizer.adam_eps = 1e-8
    config.optimizer.warmup_tokens = WARMUP_TOKENS
    config.optimizer.max_tokens = EXPECTED_TRAIN_TOKENS
    config.optimizer.lr_schedule = "warmup-stable-decay"
    config.optimizer.min_lr_ratio = 0.1
    config.optimizer.decay_tokens = DECAY_TOKENS
    config.optimizer.grad_clip_norm = 1.0
    config.checkpoint.output_dir = OUTPUT_DIR
    config.checkpoint.every_steps = 100
    config.checkpoint.every_minutes = 30.0
    config.checkpoint.keep_last = 3
    config.checkpoint.stop_file = "STOP"
    config.checkpoint.save_on_signal = True
    config.runtime.bf16 = True
    config.runtime.seed = 3407
    config.runtime.deterministic = False
    config.runtime.log_every_steps = 1
    config.runtime.profile = False
    config.runtime.profile_wait_steps = 1
    config.runtime.profile_warmup_steps = 1
    config.runtime.profile_active_steps = 3
    config.runtime.offline = True
    config.runtime.allow_tf32 = True
    config.runtime.sharding = "fsdp2"
    config.runtime.activation_checkpointing = True
    config.runtime.activation_checkpoint_layer_count = 0
    config.runtime.hidden_alignment_activation_checkpoint_layer_count = 8
    config.runtime.dense_transfer_execution = "expanded"
    config.runtime.dense_transfer_token_checkpoint = True
    config.runtime.dense_transfer_checkpoint_layer_count = 0
    config.runtime.hidden_alignment_dense_transfer_checkpoint_layer_count = 16
    config.runtime.teacher_cpu_offload = True
    config.runtime.activation_checkpointing_on_alignment_only = True
    config.runtime.fused_adamw = True
    config.runtime.loss_chunk_tokens = LOSS_CHUNK_TOKENS
    config.runtime.loss_checkpoint_chunks = True
    config.runtime.compile_streaming_loss = True
    config.runtime.expandable_segments = True
    config.validate()
    if (
        config.activation_checkpoint_layer_indices(align_hidden=False)
        != ORDINARY_OUTER_CHECKPOINT_LAYERS
    ):
        raise FinalizationError("B1 ordinary outer checkpointing did not resolve to zero layers")
    if (
        config.activation_checkpoint_layer_indices(align_hidden=True)
        != ALIGNMENT_OUTER_CHECKPOINT_LAYERS
    ):
        raise FinalizationError("B1 alignment outer checkpointing did not resolve to exact AC8")
    ordinary_inner = config.dense_transfer_checkpoint_layer_indices(
        align_hidden=False,
        outer_checkpoint_layer_indices=ORDINARY_OUTER_CHECKPOINT_LAYERS,
    )
    alignment_inner = config.dense_transfer_checkpoint_layer_indices(
        align_hidden=True,
        outer_checkpoint_layer_indices=ALIGNMENT_OUTER_CHECKPOINT_LAYERS,
    )
    if ordinary_inner != ORDINARY_INNER_CHECKPOINT_LAYERS:
        raise FinalizationError("B1 ordinary inner checkpointing did not resolve to zero layers")
    if alignment_inner != ALIGNMENT_INNER_CHECKPOINT_LAYERS:
        raise FinalizationError(
            "B1 alignment inner checkpointing did not resolve to the exact outer complement"
        )
    return config


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _authenticate_preflight(
    options: FinalizationOptions,
    config: TrainConfig,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Run the complete CPU preflight once; it performs all shard/source SHA scans."""

    from .preflight import run_training_preflight

    with _working_directory(options.root):
        report = run_training_preflight(config, world_size=1)
    if (
        report.data_governance.ready_for_training is not True
        or report.data_governance.research_only is True
        or report.data_governance.pending_audits
    ):
        raise FinalizationError("training preflight did not authenticate ready data governance")
    cooldown = data["quality_cooldown"]
    if (
        report.batch.world_size != 1
        or report.batch.micro_batch_tokens_per_rank != SEQUENCE_LENGTH
        or report.batch.gradient_accumulation_steps != 64
        or report.batch.global_batch_tokens != GLOBAL_BATCH_TOKENS
        or report.activation_checkpoint_layer_count != 0
        or report.hidden_alignment_activation_checkpoint_layer_count != 8
        or report.activation_checkpoint_layer_indices != ORDINARY_OUTER_CHECKPOINT_LAYERS
        or report.hidden_alignment_activation_checkpoint_layer_indices
        != ALIGNMENT_OUTER_CHECKPOINT_LAYERS
        or report.dense_transfer_execution != "expanded"
        or report.dense_transfer_checkpoint_layer_count != 0
        or report.hidden_alignment_dense_transfer_checkpoint_layer_count != 16
        or report.dense_transfer_token_checkpoint_layer_indices != ORDINARY_INNER_CHECKPOINT_LAYERS
        or report.hidden_alignment_dense_transfer_token_checkpoint_layer_indices
        != ALIGNMENT_INNER_CHECKPOINT_LAYERS
        or report.quality_cooldown_enabled is not True
        or report.quality_cooldown_start_tokens != QUALITY_COOLDOWN_START_TOKENS
        or report.quality_cooldown_dataset_fingerprint != cooldown["dataset_fingerprint"]
        or report.quality_cooldown_sequence_count != cooldown["sequence_count"]
        or report.quality_cooldown_token_count != cooldown["token_count"]
        or list(report.quality_cooldown_selected_shard_ids) != cooldown["selected_shard_ids"]
        or [list(item) for item in report.quality_cooldown_source_mix_token_counts]
        != cooldown["source_mix_token_counts"]
    ):
        raise FinalizationError(
            "training preflight semantics differ from expanded B1 outer0/8 inner0/16 "
            "with the locked 450M quality cooldown"
        )
    sources: dict[str, dict[str, Any]] = {}
    for role in ("backbone", "donor", "teacher", "tokenizer"):
        source = getattr(config.sources, role)
        manifest = (options.root / source.local_path / "download-manifest.json").resolve()
        identity = _identity(options.root, manifest)
        if identity["sha256"] != source.manifest_sha256:
            raise FinalizationError(f"{role} source manifest SHA differs after preflight")
        sources[role] = {
            **identity,
            "model_id": source.model_id,
            "revision": source.revision,
        }
    return {
        "config_fingerprint": report.config_fingerprint,
        "data_fingerprint": report.data_fingerprint,
        "source_tree_sha256": report.source_tree_sha256,
        "checked_paths": list(report.checked_paths),
        "calibration_fingerprints": [list(item) for item in report.calibration_fingerprints],
        "batch": dataclasses.asdict(report.batch),
        "data_governance": dataclasses.asdict(report.data_governance),
        "activation_checkpoint_layer_indices": list(report.activation_checkpoint_layer_indices),
        "hidden_alignment_activation_checkpoint_layer_indices": list(
            report.hidden_alignment_activation_checkpoint_layer_indices
        ),
        "dense_transfer_execution": report.dense_transfer_execution,
        "dense_transfer_checkpoint_layer_count": (report.dense_transfer_checkpoint_layer_count),
        "hidden_alignment_dense_transfer_checkpoint_layer_count": (
            report.hidden_alignment_dense_transfer_checkpoint_layer_count
        ),
        "dense_transfer_token_checkpoint_layer_indices": list(
            report.dense_transfer_token_checkpoint_layer_indices
        ),
        "hidden_alignment_dense_transfer_token_checkpoint_layer_indices": list(
            report.hidden_alignment_dense_transfer_token_checkpoint_layer_indices
        ),
        "quality_cooldown": {
            "enabled": report.quality_cooldown_enabled,
            "start_tokens": report.quality_cooldown_start_tokens,
            "dataset_fingerprint": report.quality_cooldown_dataset_fingerprint,
            "sequence_count": report.quality_cooldown_sequence_count,
            "token_count": report.quality_cooldown_token_count,
            "selected_shard_ids": list(report.quality_cooldown_selected_shard_ids),
            "source_mix_token_counts": [
                list(item) for item in report.quality_cooldown_source_mix_token_counts
            ],
        },
        "sources": sources,
    }


def _dashboard_update(
    options: FinalizationOptions,
    *,
    config_sha256: str,
) -> tuple[str, dict[str, Any]]:
    original = options.dashboard_config.read_text(encoding="utf-8")
    try:
        dashboard = json.loads(original)
    except json.JSONDecodeError as error:
        raise FinalizationError("dashboard config is invalid JSON") from error
    if not isinstance(dashboard, dict) or dashboard.get("schema_version") != 1:
        raise FinalizationError("dashboard config schema_version must equal 1")
    project_value = dashboard.get("project_root", ".")
    if not isinstance(project_value, str):
        raise FinalizationError("dashboard project_root is invalid")
    dashboard_root = (options.dashboard_config.parent / project_value).resolve()
    if dashboard_root != options.root:
        raise FinalizationError("dashboard project_root differs from finalizer root")
    profiles = dashboard.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise FinalizationError("dashboard has no existing profile allowlist")
    if any(isinstance(item, dict) and item.get("id") == PROFILE_ID for item in profiles):
        raise FinalizationError(f"dashboard profile already exists: {PROFILE_ID}")
    profile = {
        "id": PROFILE_ID,
        "label": "Base Dense v2 500M (finalized fork)",
        "config": _relative(options.root, options.output_config),
        "config_sha256": config_sha256,
        "resume": "none",
        "fork_from": _relative(options.root, options.fork_from),
        "launch_enabled": options.enable_web_launch,
    }
    profiles.append(profile)
    rendered = json.dumps(dashboard, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return original, {"rendered": rendered, "profile": profile}


def _render_config(config: TrainConfig) -> str:
    return yaml.safe_dump(config.canonical_dict(), sort_keys=True, allow_unicode=True)


def finalize_base_v2(options: FinalizationOptions) -> dict[str, Any]:
    """Authenticate every input and atomically publish a non-running v2 definition."""

    _validate_decisions(options)
    if not options.root.is_dir():
        raise FinalizationError(f"project root is not a directory: {options.root}")
    with _working_directory(options.root):
        base = load_train_config(options.base_config)
    if base.run_id != "base-dense-v1" or base.track != "base" or base.stage != "dense-oracle":
        raise FinalizationError("base config is not the Base dense v1 experiment definition")

    numerical_evidence = _authenticate_numerical_evidence(options.root)
    performance = _authenticate_performance(options.root, numerical_evidence)
    data = _authenticate_data(options, base)
    fork = _authenticate_fork(options, base)
    config = _build_config(options, base, data)
    production_sources_before = _authenticate_current_production_sources(options.root)
    preflight = _authenticate_preflight(options, config, data)
    production_sources = _authenticate_current_production_sources(options.root)
    if production_sources != production_sources_before:
        raise FinalizationError("current production sources changed during finalization")

    config_text = _render_config(config)
    config_sha256 = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    dashboard_original, dashboard_update = _dashboard_update(
        options,
        config_sha256=config_sha256,
    )
    config_written = False
    dashboard_written = False
    try:
        atomic_write_text(options.output_config, config_text)
        config_written = True
        with _working_directory(options.root):
            reloaded = load_train_config(options.output_config)
        if reloaded.canonical_dict() != config.canonical_dict():
            raise FinalizationError("written config differs from authenticated in-memory config")

        atomic_write_text(options.dashboard_config, dashboard_update["rendered"])
        dashboard_written = True
        from .web import load_dashboard_settings

        settings = load_dashboard_settings(options.dashboard_config)
        profile = settings.profile(PROFILE_ID)
        if (
            profile.run_id != RUN_ID
            or profile.config_sha256 != config_sha256
            or profile.resume != "none"
            or profile.fork_from != options.fork_from
            or profile.launch_enabled is not options.enable_web_launch
        ):
            raise FinalizationError("written dashboard profile changed the fixed launch contract")

        options.evidence_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = options.evidence_dir / "manifest.json"
        complete_path = options.evidence_dir / "COMPLETE"
        manifest = {
            "schema_version": FINALIZATION_SCHEMA_VERSION,
            "kind": FINALIZATION_KIND,
            "created_at": _utc_now(),
            "decisions": {
                "mtp_loss_weight": options.mtp_loss_weight,
                "peak_learning_rates": {
                    "adapter": options.adapter_lr,
                    "router": options.router_lr,
                    "lora": options.lora_lr,
                    "scale": options.scale_lr,
                },
                "web_launch_enabled": options.enable_web_launch,
            },
            "training_contract": {
                "run_id": RUN_ID,
                "output_dir": OUTPUT_DIR,
                "resume": "none",
                "fork_from": _relative(options.root, options.fork_from),
                "global_batch_tokens": GLOBAL_BATCH_TOKENS,
                "physical_micro_batch_size": 1,
                "gradient_accumulation_steps_single_gpu": 64,
                "optimizer_max_tokens": EXPECTED_TRAIN_TOKENS,
                "lr_schedule": "warmup-stable-decay",
                "warmup_tokens": WARMUP_TOKENS,
                "stable_until_tokens": STABLE_UNTIL_TOKENS,
                "decay_tokens": DECAY_TOKENS,
                "min_lr_ratio": 0.1,
                "quality_cooldown_start_tokens": QUALITY_COOLDOWN_START_TOKENS,
                "quality_cooldown_required_tokens": QUALITY_COOLDOWN_TOKENS,
                "quality_cooldown_dataset_fingerprint": data["quality_cooldown"][
                    "dataset_fingerprint"
                ],
                "ordinary_activation_checkpoint_layers": 0,
                "alignment_activation_checkpoint_layers": 8,
                "ordinary_activation_checkpoint_layer_indices": list(
                    ORDINARY_OUTER_CHECKPOINT_LAYERS
                ),
                "alignment_activation_checkpoint_layer_indices": list(
                    ALIGNMENT_OUTER_CHECKPOINT_LAYERS
                ),
                "dense_transfer_execution": "expanded",
                "dense_transfer_token_checkpoint": True,
                "ordinary_dense_transfer_token_checkpoint_layer_indices": list(
                    ORDINARY_INNER_CHECKPOINT_LAYERS
                ),
                "alignment_dense_transfer_token_checkpoint_layer_indices": list(
                    ALIGNMENT_INNER_CHECKPOINT_LAYERS
                ),
                "loss_chunk_tokens": LOSS_CHUNK_TOKENS,
                "checkpoint_every_steps": 100,
                "checkpoint_every_minutes": 30.0,
            },
            "inputs": {
                "base_config": _identity(options.root, options.base_config),
                "performance": performance,
                "numerical_evidence": numerical_evidence,
                "source_provenance": {
                    "historical_numerical_math_evidence": {
                        "manifest_embedded_source_sha256": numerical_evidence[
                            "historical_embedded_source_sha256"
                        ],
                        "current_source_equality_required": False,
                    },
                    "current_production_implementation": {
                        **production_sources,
                        "preflight_source_tree_sha256": preflight["source_tree_sha256"],
                    },
                },
                "data": data,
                "fork_checkpoint": fork,
            },
            "preflight": preflight,
            "outputs": {
                "config": _identity(options.root, options.output_config),
                "dashboard": _identity(options.root, options.dashboard_config),
                "dashboard_profile": dashboard_update["profile"],
            },
            "safety": {
                "training_started": False,
                "optimizer_created": False,
                "optimizer_step_called": False,
                "cuda_initialized": False,
                "dashboard_does_not_auto_start_training": True,
            },
        }
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(
            complete_path,
            {
                "schema_version": FINALIZATION_SCHEMA_VERSION,
                "kind": FINALIZATION_COMPLETE_KIND,
                "manifest": manifest_path.name,
                "manifest_sha256": sha256_file(manifest_path),
                "config": _identity(options.root, options.output_config),
                "dashboard_profile_id": PROFILE_ID,
                "launch_enabled": options.enable_web_launch,
                "training_started": False,
                "optimizer_created": False,
            },
        )
        return {
            "ok": True,
            "config": _identity(options.root, options.output_config),
            "evidence_manifest": _identity(options.root, manifest_path),
            "complete": _identity(options.root, complete_path),
            "dashboard_profile": dashboard_update["profile"],
            "dashboard_reload_required": True,
            "training_started": False,
            "optimizer_created": False,
        }
    except Exception:
        if dashboard_written:
            atomic_write_text(options.dashboard_config, dashboard_original)
        if config_written:
            options.output_config.unlink(missing_ok=True)
        if options.evidence_dir.exists():
            shutil.rmtree(options.evidence_dir)
        raise


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root",
    )
    parser.add_argument("--base-config", default="configs/base/dense-oracle.yaml")
    parser.add_argument(
        "--prepared-manifest",
        default="artifacts/data/base-v2-500m/manifest.json",
    )
    parser.add_argument(
        "--kd-manifest",
        default="artifacts/data/base-v2-500m-kd/manifest.json",
    )
    parser.add_argument(
        "--kd-orchestration-complete",
        default="artifacts/data/base-v2-500m-kd-orchestration/COMPLETE",
        help="successful generate-kd/index-kd orchestration certification",
    )
    parser.add_argument(
        "--quality-cooldown-prepared-manifest",
        required=True,
        help="explicit independent whole-shard prepared subset covering the final 50M tokens",
    )
    parser.add_argument(
        "--quality-cooldown-kd-manifest",
        required=True,
        help="explicit top-64 KD manifest paired with the quality-cooldown prepared subset",
    )
    parser.add_argument(
        "--pipeline-complete",
        default="artifacts/data/base-v2-500m-pipeline/COMPLETE",
    )
    parser.add_argument(
        "--fork-from",
        default="runs/base-dense-v1/step-000000000383-milestone-complete",
    )
    parser.add_argument("--output-config", default="configs/base/dense-v2-500m.yaml")
    parser.add_argument("--dashboard-config", default="configs/web/dashboard.json")
    parser.add_argument(
        "--evidence-dir",
        default="artifacts/configuration/base-dense-v2-500m",
    )
    parser.add_argument(
        "--mtp-loss-weight",
        type=float,
        required=True,
        help="explicit experimental native-MTP coefficient; 0.1 is the measured candidate",
    )
    parser.add_argument(
        "--adapter-lr",
        type=float,
        required=True,
        help="explicit peak/stable-phase adapter learning rate",
    )
    parser.add_argument(
        "--router-lr",
        type=float,
        required=True,
        help="explicit peak/stable-phase router learning rate (inactive in dense stage)",
    )
    parser.add_argument(
        "--lora-lr",
        type=float,
        required=True,
        help="explicit peak/stable-phase LoRA learning rate (inactive in dense stage)",
    )
    parser.add_argument(
        "--scale-lr",
        type=float,
        required=True,
        help="explicit peak/stable-phase branch-scale learning rate",
    )
    parser.add_argument(
        "--enable-web-launch",
        action="store_true",
        help="explicitly make the finalized fixed profile launchable; default is monitor-only",
    )


def options_from_namespace(args: argparse.Namespace) -> FinalizationOptions:
    return FinalizationOptions.repository_defaults(
        root=args.root,
        base_config=args.base_config,
        prepared_manifest=args.prepared_manifest,
        kd_manifest=args.kd_manifest,
        kd_orchestration_complete=args.kd_orchestration_complete,
        quality_cooldown_prepared_manifest=args.quality_cooldown_prepared_manifest,
        quality_cooldown_kd_manifest=args.quality_cooldown_kd_manifest,
        pipeline_complete=args.pipeline_complete,
        fork_from=args.fork_from,
        output_config=args.output_config,
        dashboard_config=args.dashboard_config,
        evidence_dir=args.evidence_dir,
        mtp_loss_weight=args.mtp_loss_weight,
        adapter_lr=args.adapter_lr,
        router_lr=args.router_lr,
        lora_lr=args.lora_lr,
        scale_lr=args.scale_lr,
        enable_web_launch=args.enable_web_launch,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_cli_arguments(parser)
    args = parser.parse_args(argv)
    try:
        result = finalize_base_v2(options_from_namespace(args))
    except (ConfigError, FinalizationError, OSError, ValueError) as error:
        print(
            json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False, indent=2),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "FinalizationError",
    "FinalizationOptions",
    "add_cli_arguments",
    "finalize_base_v2",
    "main",
    "options_from_namespace",
]


if __name__ == "__main__":
    raise SystemExit(main())
