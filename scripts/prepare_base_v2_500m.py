#!/usr/bin/env python3
"""Resumable, fail-closed Base-v2 500M data pipeline.

This script only builds, audits, filters, and tokenizes data.  It never loads a
teacher model, initializes CUDA, creates an optimizer, or starts training.
"""

from __future__ import annotations

import argparse
import codecs
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
KIND = "twen_base_v2_500m_pipeline_status"
PIPELINE_COMPLETE_KIND = "twen_base_v2_500m_pipeline_complete"
PERFORMANCE_APPROVAL_KIND = "twen_base_v2_500m_performance_approval"
EXPECTED_RECIPE_SHA256 = "aa9b774971480b634561557d731e1be9616044ead8a4354904708600e3254916"
EXPECTED_RESOLVED_LOCK_SHA256 = "c5098da9b49c2f8fe755a4cb73d107677fa47e5e0beaea492207c8bf5e009d35"
EXPECTED_BENCHMARK_REGISTRY_SHA256 = (
    "defe66fa003eb4d5d00fa92e975ec7e923e1ec7399bddf0816bbc65cedfbb5e8"
)
TOKENIZER_MANIFEST_SHA256 = "5e847be98c2d114d2fb85a67ac77f4eef416e11871c1a5580d1bc52663e2e643"
EXPECTED_TRAIN_TOKENS = 500_000_000
EXPECTED_VALIDATION_TOKENS = 20_000_000
MINIMUM_FREE_BYTES = 64 * 1024**3
PLANNING_AUDIT_DOCUMENTS_PER_SECOND = 250.0
PLANNING_MATERIALIZE_BYTES_PER_SECOND = 100 * 1024**2
PLANNING_PREPARE_TOKENS_PER_SECOND = 100_000.0
PRODUCTION_GLOBAL_BATCH_TOKENS = 262_144
MINIMUM_PERFORMANCE_HEADROOM_GIB = 3.0
_CASE_LABEL = re.compile(r"^b(?P<batch>[1-9][0-9]*)-(?P<mode>ordinary|alignment)-ac[0-9]+$")


class PipelineError(RuntimeError):
    """A fail-closed pipeline contract was not satisfied."""


class PipelineStopped(PipelineError):
    """The extractor honored its STOP file and can be resumed."""


@dataclass(frozen=True)
class Layout:
    root: Path
    recipe: Path
    resolved_lock: Path
    benchmark_registry: Path
    benchmark_root: Path
    tokenizer: Path
    frozen_validation_manifest: Path
    extracted: Path
    prepared: Path
    state_root: Path
    performance_gate: Path
    performance_approval: Path
    legacy_performance_gate: Path
    performance_manifest: Path
    performance_complete: Path

    @classmethod
    def repository_defaults(cls, root: Path) -> Layout:
        root = root.resolve()
        return cls(
            root=root,
            recipe=root / "locks/base-data-sources.json",
            resolved_lock=root / "locks/base-data-sources.resolved.json",
            benchmark_registry=root / "locks/base-benchmark-registry.json",
            benchmark_root=root / "data/benchmarks/base-v2",
            tokenizer=root / "artifacts/models/qwen3.5-0.8b-base",
            frozen_validation_manifest=root / "data/base-v3/corpus-manifest.json",
            extracted=root / "data/base-v2-500m",
            prepared=root / "artifacts/data/base-v2-500m",
            state_root=root / "artifacts/data/base-v2-500m-pipeline",
            performance_gate=(
                root / "artifacts/benchmarks/rtx5090-base-dense-utilization-report.json"
            ),
            performance_approval=(
                root / "artifacts/benchmarks/rtx5090-base-dense-utilization-report.approval.json"
            ),
            legacy_performance_gate=(
                root / "artifacts/benchmarks/rtx5090-base-dense-batch2-utilization-report.json"
            ),
            performance_manifest=(
                root / "artifacts/benchmarks/rtx5090-base-dense-utilization-report.MANIFEST.json"
            ),
            performance_complete=(
                root / "artifacts/benchmarks/rtx5090-base-dense-utilization-report.COMPLETE"
            ),
        )

    def audit(self, pass_index: int) -> Path:
        return self.root / f"artifacts/data/base-v2-500m-audit-pass-{pass_index:03d}"

    def filtered(self, pass_index: int) -> Path:
        if pass_index <= 0:
            raise ValueError("filtered pass indices start at one")
        return self.root / f"data/base-v2-500m-filtered-pass-{pass_index:03d}"

    def refill_plan(self, generation: int) -> Path:
        if generation <= 0:
            raise ValueError("refill generations start at one")
        return self.root / f"artifacts/data/base-v2-500m-refill-plan-{generation:03d}"

    def refill_raw(self, generation: int) -> Path:
        if generation <= 0:
            raise ValueError("refill generations start at one")
        return self.root / f"data/base-v2-500m-refill-raw-{generation:03d}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineError(f"invalid or missing JSON object: {path}") from error
    if not isinstance(value, dict):
        raise PipelineError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _verify_sha(path: Path, expected: str, label: str) -> dict[str, Any]:
    actual = _sha256(path)
    if actual != expected:
        raise PipelineError(f"{label} SHA256 changed: expected {expected}, got {actual}")
    return _identity(path)


def _verify_complete_bound_manifest(manifest: Path) -> dict[str, Any]:
    complete_path = manifest.parent / "COMPLETE"
    complete = _load_object(complete_path)
    if complete.get("manifest") != manifest.name:
        raise PipelineError(f"COMPLETE does not bind {manifest}")
    manifest_sha = _sha256(manifest)
    if complete.get("manifest_sha256") != manifest_sha:
        raise PipelineError(f"COMPLETE manifest SHA mismatch: {manifest}")
    return {
        "manifest": _identity(manifest),
        "complete": _identity(complete_path),
    }


def _verify_benchmark_registry(layout: Layout) -> dict[str, Any]:
    registry = _load_object(layout.benchmark_registry)
    benchmarks = registry.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise PipelineError("benchmark registry has no benchmarks")
    verified: list[dict[str, Any]] = []
    pending: list[str] = []
    for raw in benchmarks:
        if not isinstance(raw, dict) or not isinstance(raw.get("benchmark_id"), str):
            raise PipelineError("benchmark registry entry is invalid")
        benchmark_id = raw["benchmark_id"]
        if raw.get("required") is True and raw.get("status") != "ready":
            pending.append(benchmark_id)
        files = raw.get("files")
        if not isinstance(files, list) or not files:
            raise PipelineError(f"benchmark has no projected files: {benchmark_id}")
        file_identities: list[dict[str, Any]] = []
        for entry in files:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise PipelineError(f"invalid file inventory for benchmark {benchmark_id}")
            path = (layout.benchmark_root / entry["path"]).resolve()
            try:
                path.relative_to(layout.benchmark_root.resolve())
            except ValueError as error:
                raise PipelineError(f"benchmark path escapes root: {path}") from error
            identity = _identity(path)
            if identity["size"] != entry.get("size") or identity["sha256"] != entry.get("sha256"):
                raise PipelineError(f"benchmark projection identity mismatch: {path}")
            file_identities.append(identity)
        verified.append({"benchmark_id": benchmark_id, "files": file_identities})
    if pending:
        raise PipelineError(f"required benchmarks are not ready: {', '.join(sorted(pending))}")
    return {
        "registry": _verify_sha(
            layout.benchmark_registry,
            EXPECTED_BENCHMARK_REGISTRY_SHA256,
            "benchmark registry",
        ),
        "ready_benchmarks": verified,
    }


def _performance_report_contract(layout: Layout) -> dict[str, Any]:
    if not layout.performance_gate.is_file():
        result = {
            "ready": False,
            "path": str(layout.performance_gate),
            "reason": "batch-neutral utilization report has not completed",
        }
        if layout.legacy_performance_gate.is_file():
            result["legacy_report"] = {
                "identity": _identity(layout.legacy_performance_gate),
                "accepted_for_pipeline": False,
                "reason": "legacy batch2-named reports are explicitly rejected",
            }
        return result
    value = _load_object(layout.performance_gate)
    bundle = _performance_bundle_contract(layout, value)
    accepted = value.get("accepted") is True
    raw_recommendation = value.get("recommendation")
    recommendation = raw_recommendation if isinstance(raw_recommendation, dict) else {}
    ordinary_case = recommendation.get("ordinary_case") if recommendation else None
    alignment_case = recommendation.get("alignment_case") if recommendation else None
    rows = value.get("rows")
    reasons: list[str] = []
    if (
        value.get("schema_version") != 2
        or value.get("kind") != "rtx5090_base_dense_utilization_report"
    ):
        reasons.append("unsupported report schema/kind")
    if value.get("read_only_report_generation") is not True:
        reasons.append("report was not generated read-only")
    if value.get("no_optimizer_created_by_report") is not True:
        reasons.append("report generator optimizer contract failed")
    if not accepted:
        reasons.append("batch utilization report is not accepted")
    if value.get("global_batch_tokens") != PRODUCTION_GLOBAL_BATCH_TOKENS:
        reasons.append("report global batch token contract changed")
    if not isinstance(rows, list):
        reasons.append("report rows are missing")
        rows = []

    def selected(label: object, mode: str) -> dict[str, Any] | None:
        if not isinstance(label, str) or _CASE_LABEL.fullmatch(label) is None:
            reasons.append(f"recommended {mode} case label is invalid")
            return None
        matches = [row for row in rows if isinstance(row, dict) and row.get("label") == label]
        if len(matches) != 1:
            reasons.append(f"recommended {mode} row is not unique")
            return None
        row = matches[0]
        match = _CASE_LABEL.fullmatch(label)
        assert match is not None
        batch_size = int(match.group("batch"))
        if row.get("mode") != mode or row.get("batch_size") != batch_size:
            reasons.append(f"recommended {mode} row label/shape mismatch")
        if row.get("accepted") is not True or row.get("status") != "ok":
            reasons.append(f"recommended {mode} row is not accepted")
        logical_tokens = row.get("logical_tokens")
        if (
            isinstance(logical_tokens, bool)
            or not isinstance(logical_tokens, int)
            or logical_tokens <= 0
            or logical_tokens != batch_size * 4096
            or PRODUCTION_GLOBAL_BATCH_TOKENS % logical_tokens != 0
        ):
            reasons.append(f"recommended {mode} logical tokens do not divide global batch")
        for headroom_name in (
            "minimum_estimated_headroom_gib",
            "minimum_nvml_physical_free_gib",
        ):
            headroom = row.get(headroom_name)
            if (
                isinstance(headroom, bool)
                or not isinstance(headroom, (int, float))
                or not math.isfinite(float(headroom))
                or float(headroom) < MINIMUM_PERFORMANCE_HEADROOM_GIB
            ):
                reasons.append(f"recommended {mode} {headroom_name} is below 3 GiB")
        throughput = row.get("production_tokens_per_second")
        if (
            isinstance(throughput, bool)
            or not isinstance(throughput, (int, float))
            or not math.isfinite(float(throughput))
            or float(throughput) <= 0
        ):
            reasons.append(f"recommended {mode} throughput is not finite/positive")
        health = row.get("health")
        if not isinstance(health, dict) or not (
            health.get("ok") is True
            and health.get("loss_finite") is True
            and health.get("gradients_finite") is True
            and health.get("missing_gradient_tensors") == 0
            and health.get("nonfinite_gradient_tensors") == 0
            and health.get("present_gradient_tensor_counts") == [72]
        ):
            reasons.append(f"recommended {mode} finite/gradient health contract failed")
        if row.get("no_optimizer_created_or_stepped") is not True:
            reasons.append(f"recommended {mode} optimizer contract failed")
        if row.get("optimizer_state_reserve_gib") != 1.5:
            reasons.append(f"recommended {mode} optimizer reserve contract failed")
        if (
            row.get("mtp_loss_weight") != 0.1
            or row.get("mtp_attention_implementation") != "sdpa"
            or row.get("teacher_cpu_offload") is not True
        ):
            reasons.append(f"recommended {mode} MTP/offload contract failed")
        source = row.get("source")
        if not isinstance(source, dict):
            reasons.append(f"recommended {mode} source identity is missing")
        else:
            source_path = Path(str(source.get("path", "")))
            source_sha = source.get("sha256")
            if (
                not source_path.is_file()
                or not isinstance(source_sha, str)
                or _sha256(source_path) != source_sha
            ):
                reasons.append(f"recommended {mode} source artifact SHA changed")
        return row

    ordinary_row = selected(ordinary_case, "ordinary")
    alignment_row = selected(alignment_case, "alignment")
    if ordinary_row is not None and alignment_row is not None:
        if ordinary_row.get("batch_size") != alignment_row.get("batch_size"):
            reasons.append("recommended ordinary/alignment rows use different physical batches")
        if recommendation.get("batch_size") != ordinary_row.get("batch_size"):
            reasons.append("recommendation physical batch metadata mismatch")
        recommended_tps = recommendation.get("tokens_per_second")
        if (
            isinstance(recommended_tps, bool)
            or not isinstance(recommended_tps, (int, float))
            or not math.isfinite(float(recommended_tps))
            or float(recommended_tps) <= 0
        ):
            reasons.append("recommendation weighted throughput is not finite/positive")
    if not isinstance(raw_recommendation, dict):
        reasons.append("report recommendation is missing")
    provenance = value.get("source_provenance")
    provenance_files = provenance.get("files") if isinstance(provenance, dict) else None
    expected_provenance = _performance_source_identities(layout)
    provenance_mapping = {
        "benchmark": expected_provenance["benchmark_full_dense_graph"],
        "mtp": expected_provenance["native_mtp"],
    }
    for name, expected_identity in provenance_mapping.items():
        actual = provenance_files.get(name) if isinstance(provenance_files, dict) else None
        if not isinstance(actual, dict) or actual.get("sha256") != expected_identity["sha256"]:
            reasons.append(f"report {name} source provenance SHA changed")
    if not _all_finite(value):
        reasons.append("report contains non-finite numbers")
    if not bundle["ready"]:
        reasons.append(f"report bundle contract failed: {bundle['reason']}")
    ready = not reasons
    return {
        "ready": ready,
        "identity": _identity(layout.performance_gate),
        "accepted": accepted,
        "recommendation": recommendation,
        "selected_batch_size": (
            ordinary_row.get("batch_size") if ordinary_row is not None else None
        ),
        "safety_contract": {
            "global_batch_tokens": PRODUCTION_GLOBAL_BATCH_TOKENS,
            "minimum_physical_headroom_gib": MINIMUM_PERFORMANCE_HEADROOM_GIB,
            "same_ordinary_alignment_batch": True,
            "native_mtp_attention_implementation": "sdpa",
            "finite_loss_gradients_metrics": True,
            "no_optimizer_created_or_stepped": True,
        },
        "bundle": bundle,
        "reason": None if ready else "; ".join(dict.fromkeys(reasons)),
    }


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _performance_bundle_contract(
    layout: Layout,
    report: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if not layout.performance_manifest.is_file():
        reasons.append("MANIFEST is missing")
    if not layout.performance_complete.is_file():
        reasons.append("COMPLETE is missing")
    if reasons:
        return {
            "ready": False,
            "manifest_path": str(layout.performance_manifest),
            "complete_path": str(layout.performance_complete),
            "reason": "; ".join(reasons),
        }
    manifest = _load_object(layout.performance_manifest)
    complete = _load_object(layout.performance_complete)
    report_identity = _identity(layout.performance_gate)
    provenance_sha = _canonical_json_sha256(report.get("source_provenance"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "twen_rtx5090_base_dense_utilization_report_bundle"
    ):
        reasons.append("MANIFEST schema/kind mismatch")
    if manifest.get("accepted") != report.get("accepted"):
        reasons.append("MANIFEST accepted differs from report")
    if manifest.get("recommendation") != report.get("recommendation"):
        reasons.append("MANIFEST recommendation differs from report")
    if manifest.get("source_provenance_sha256") != provenance_sha:
        reasons.append("MANIFEST source provenance SHA mismatch")
    files = manifest.get("files")
    prefix = layout.performance_gate.with_suffix("")
    expected_paths = {
        "report_json": layout.performance_gate,
        "report_markdown": layout.performance_gate.with_suffix(".md"),
        "throughput_memory_svg": prefix.with_name(prefix.name + "-throughput-memory.svg"),
        "power_svg": prefix.with_name(prefix.name + "-power.svg"),
        "utilization_svg": prefix.with_name(prefix.name + "-utilization.svg"),
    }
    if not isinstance(files, dict) or set(files) != set(expected_paths):
        reasons.append("MANIFEST files must contain exactly the five report bundle keys")
    else:
        for name, expected_path in expected_paths.items():
            raw = files.get(name)
            if not isinstance(raw, dict):
                reasons.append(f"MANIFEST {name} identity is invalid")
                continue
            raw_path = Path(str(raw.get("path", "")))
            path = (
                raw_path
                if raw_path.is_absolute()
                else layout.performance_manifest.parent / raw_path
            )
            path = path.resolve()
            if path != expected_path.resolve():
                reasons.append(f"MANIFEST {name} path mismatch")
                continue
            if not path.is_file():
                reasons.append(f"MANIFEST {name} file is missing")
                continue
            if raw.get("size") != path.stat().st_size or raw.get("sha256") != _sha256(path):
                reasons.append(f"MANIFEST {name} size/SHA mismatch")

    if (
        complete.get("schema_version") != 1
        or complete.get("kind") != "twen_rtx5090_base_dense_utilization_report_complete"
    ):
        reasons.append("COMPLETE schema/kind mismatch")
    raw_manifest_path = Path(str(complete.get("manifest", "")))
    complete_manifest_path = (
        raw_manifest_path
        if raw_manifest_path.is_absolute()
        else layout.performance_complete.parent / raw_manifest_path
    ).resolve()
    if complete_manifest_path != layout.performance_manifest.resolve():
        reasons.append("COMPLETE manifest path mismatch")
    if complete.get("manifest_sha256") != _sha256(layout.performance_manifest):
        reasons.append("COMPLETE manifest SHA mismatch")
    complete_report = complete.get("report")
    if not isinstance(complete_report, dict):
        reasons.append("COMPLETE report identity is invalid")
    else:
        raw_report_path = Path(str(complete_report.get("path", "")))
        complete_report_path = (
            raw_report_path
            if raw_report_path.is_absolute()
            else layout.performance_complete.parent / raw_report_path
        ).resolve()
        if complete_report_path != layout.performance_gate.resolve():
            reasons.append("COMPLETE report path mismatch")
        if complete_report.get("sha256") != report_identity["sha256"]:
            reasons.append("COMPLETE report SHA mismatch")
    if complete.get("accepted") != report.get("accepted"):
        reasons.append("COMPLETE accepted differs from report")
    if complete.get("recommendation") != report.get("recommendation"):
        reasons.append("COMPLETE recommendation differs from report")
    if complete.get("source_provenance_sha256") != provenance_sha:
        reasons.append("COMPLETE source provenance SHA mismatch")
    return {
        "ready": not reasons,
        "manifest": _identity(layout.performance_manifest),
        "complete": _identity(layout.performance_complete),
        "source_provenance_sha256": provenance_sha,
        "reason": None if not reasons else "; ".join(dict.fromkeys(reasons)),
    }


def _performance_source_identities(layout: Layout) -> dict[str, dict[str, Any]]:
    return {
        "benchmark_full_dense_graph": _identity(
            layout.root / "scripts/benchmark_full_dense_graph.py"
        ),
        "native_mtp": _identity(layout.root / "src/twen/modeling/mtp.py"),
    }


def _performance_gate(layout: Layout) -> dict[str, Any]:
    report = _performance_report_contract(layout)
    result = dict(report)
    result["report_contract_ready"] = bool(report["ready"])
    approval_path = layout.performance_approval
    if not report["ready"]:
        result["ready"] = False
        result["approval"] = {
            "ready": False,
            "path": str(approval_path),
            "reason": "report contract is not ready",
        }
        return result
    if not approval_path.is_file():
        result["ready"] = False
        result["reason"] = "explicit post-fix performance approval is missing"
        result["approval"] = {
            "ready": False,
            "path": str(approval_path),
            "reason": "approval defaults to absent and must be generated by the main task",
        }
        return result
    approval = _load_object(approval_path)
    expected_sources = _performance_source_identities(layout)
    approval_sources = approval.get("sources")
    sources_match = approval_sources == expected_sources
    report_identity = report.get("identity")
    report_sha = report_identity.get("sha256") if isinstance(report_identity, dict) else None
    approval_ready = bool(
        approval.get("schema_version") == SCHEMA_VERSION
        and approval.get("kind") == PERFORMANCE_APPROVAL_KIND
        and approval.get("native_mtp_attention_fix_verified") is True
        and approval.get("report_sha256") == report_sha
        and approval.get("recommendation") == report.get("recommendation")
        and sources_match
    )
    result["ready"] = approval_ready
    result["reason"] = (
        None if approval_ready else "performance approval does not bind this report/source tree"
    )
    result["approval"] = {
        "ready": approval_ready,
        "identity": _identity(approval_path),
        "report_sha256_matches": approval.get("report_sha256") == report_sha,
        "recommendation_matches": approval.get("recommendation") == report.get("recommendation"),
        "source_identities_match": sources_match,
        "native_mtp_attention_fix_verified": approval.get("native_mtp_attention_fix_verified")
        is True,
    }
    return result


def write_performance_approval(layout: Layout) -> dict[str, Any]:
    """Explicitly bind one accepted post-fix report to its implementation sources."""

    report = _performance_report_contract(layout)
    if not report["ready"]:
        raise PipelineError(str(report["reason"]))
    if layout.performance_approval.exists():
        raise PipelineError(
            f"performance approval already exists; it is never overwritten: "
            f"{layout.performance_approval}"
        )
    report_identity = report.get("identity")
    if not isinstance(report_identity, dict) or not isinstance(report_identity.get("sha256"), str):
        raise PipelineError("performance report identity is invalid")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": PERFORMANCE_APPROVAL_KIND,
        "approved_at": _utc_now(),
        "approval_scope": "post-fix native MTP attention utilization evidence",
        "native_mtp_attention_fix_verified": True,
        "report_sha256": report_identity["sha256"],
        "recommendation": report["recommendation"],
        "sources": _performance_source_identities(layout),
    }
    _atomic_json(layout.performance_approval, payload)
    gate = _performance_gate(layout)
    if not gate["ready"]:
        raise PipelineError("newly written performance approval failed self-validation")
    return {
        "ok": True,
        "approval": _identity(layout.performance_approval),
        "performance_gate": gate,
        "training_started": False,
        "gpu_kd_started": False,
    }


def preflight(layout: Layout, *, require_performance_gate: bool) -> dict[str, Any]:
    protected = {
        (layout.root / "data/base-v1").resolve(),
        (layout.root / "data/base-v2").resolve(),
        (layout.root / "data/base-v3").resolve(),
    }
    managed = {layout.extracted.resolve(), layout.prepared.resolve(), layout.state_root.resolve()}
    managed.update(layout.audit(index).resolve() for index in range(64))
    managed.update(layout.filtered(index).resolve() for index in range(1, 65))
    managed.update(layout.refill_plan(index).resolve() for index in range(1, 9))
    managed.update(layout.refill_raw(index).resolve() for index in range(1, 9))
    if protected & managed:
        raise PipelineError("a managed output aliases a protected Base-v1/v2/v3 directory")
    if (layout.extracted / "INVALIDATED.json").exists():
        raise PipelineError(f"target is explicitly invalidated: {layout.extracted}")

    recipe_identity = _verify_sha(layout.recipe, EXPECTED_RECIPE_SHA256, "source recipe")
    lock_identity = _verify_sha(
        layout.resolved_lock,
        EXPECTED_RESOLVED_LOCK_SHA256,
        "resolved source lock",
    )
    tokenizer_manifest = layout.tokenizer / "download-manifest.json"
    tokenizer_identity = _verify_sha(
        tokenizer_manifest,
        TOKENIZER_MANIFEST_SHA256,
        "tokenizer download manifest",
    )
    recipe = _load_object(layout.recipe)
    resolved = _load_object(layout.resolved_lock)
    if resolved.get("recipe_sha256") != EXPECTED_RECIPE_SHA256:
        raise PipelineError("resolved lock does not bind the expected recipe")
    sources = recipe.get("sources")
    resolved_sources = resolved.get("sources")
    if not isinstance(sources, list) or not isinstance(resolved_sources, list):
        raise PipelineError("source inventories are invalid")
    if len(sources) != 6 or len(resolved_sources) != 6:
        raise PipelineError("the 500M recipe must contain exactly six sources")
    train_tokens = sum(int(source["train_token_quotas"]["sparse"]) for source in sources)
    validation_tokens = sum(int(source["validation_token_quota"]) for source in sources)
    if train_tokens != EXPECTED_TRAIN_TOKENS or validation_tokens != EXPECTED_VALIDATION_TOKENS:
        raise PipelineError("sparse profile token quotas changed")
    resolved_file_count = sum(len(source["files"]) for source in resolved_sources)
    resolved_remote_bytes = sum(
        int(entry["size"]) for source in resolved_sources for entry in source["files"]
    )
    if resolved_file_count != 1563:
        raise PipelineError(f"resolved source file count changed: {resolved_file_count}")

    frozen = _verify_complete_bound_manifest(layout.frozen_validation_manifest)
    frozen_manifest = _load_object(layout.frozen_validation_manifest)
    if frozen_manifest.get("profile") != "dense":
        raise PipelineError("frozen Base-v3 validation manifest is not the dense profile")
    if (layout.frozen_validation_manifest.parent / "INVALIDATED.json").exists():
        raise PipelineError("frozen Base-v3 validation corpus is invalidated")
    benchmark = _verify_benchmark_registry(layout)
    disk = shutil.disk_usage(layout.root)
    if disk.free < MINIMUM_FREE_BYTES:
        raise PipelineError(
            f"insufficient free disk: {disk.free} bytes; require at least {MINIMUM_FREE_BYTES}"
        )
    performance = _performance_gate(layout)
    if require_performance_gate and not performance["ready"]:
        raise PipelineError(str(performance["reason"]))
    return {
        "ok": True,
        "ready_to_run": bool(performance["ready"]),
        "checked_at": _utc_now(),
        "inputs": {
            "orchestrator": _identity(Path(__file__)),
            "recipe": recipe_identity,
            "resolved_lock": lock_identity,
            "tokenizer_manifest": tokenizer_identity,
            "frozen_validation": frozen,
            "benchmark_registry": benchmark,
        },
        "quotas": {
            "train_tokens": train_tokens,
            "validation_tokens": validation_tokens,
            "total_tokens": train_tokens + validation_tokens,
        },
        "resolved_remote_inventory": {
            "files": resolved_file_count,
            "bytes": resolved_remote_bytes,
            "download_mode": "HTTP Range streaming; full inventory is not downloaded",
        },
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "minimum_required_bytes": MINIMUM_FREE_BYTES,
        },
        "network_policy": "fallback (Hugging Face direct first, configured proxy second)",
        "performance_gate": performance,
        "training_started": False,
        "gpu_kd_started": False,
    }


def _status_path(layout: Layout) -> Path:
    return layout.state_root / "status.json"


def _read_status(layout: Layout) -> dict[str, Any]:
    path = _status_path(layout)
    if not path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "status": "not_started",
            "history": [],
        }
    value = _load_object(path)
    if value.get("schema_version") != SCHEMA_VERSION or value.get("kind") != KIND:
        raise PipelineError(f"unsupported pipeline status: {path}")
    return value


def _write_status(layout: Layout, value: dict[str, Any]) -> None:
    value["schema_version"] = SCHEMA_VERSION
    value["kind"] = KIND
    value["updated_at"] = _utc_now()
    _atomic_json(_status_path(layout), value)


def _redacted_command(command: list[str]) -> list[str]:
    result = list(command)
    for index, value in enumerate(result[:-1]):
        if value in {"--proxy", "--token"}:
            result[index + 1] = "<redacted>"
    return result


def _historical_build_rate(layout: Layout) -> float | None:
    """Estimate token/s from the immutable Base-v3 extractor completion times."""

    manifest_path = layout.frozen_validation_manifest
    if not manifest_path.is_file():
        return None
    manifest = _load_object(manifest_path)
    total_tokens = int(manifest.get("actual_train_tokens", 0)) + int(
        manifest.get("actual_validation_tokens", 0)
    )
    markers = list(manifest_path.parent.glob("extracted/*/chunk-*/COMPLETE"))
    if len(markers) < 2 or total_tokens <= 1_000_000:
        return None
    times = [marker.stat().st_mtime for marker in markers]
    elapsed = max(times) - min(times)
    if elapsed <= 0:
        return None
    # The first timestamp is observed after roughly one output chunk committed.
    return (total_tokens - 1_000_000) / elapsed


def _candidate_size_and_documents(layout: Layout) -> tuple[int, int]:
    candidates = [layout.extracted / "corpus-manifest.json"]
    candidates.extend(
        path / "corpus-manifest.json"
        for path in sorted(layout.root.glob("data/base-v2-500m-refill-raw-*"))
    )
    candidates.extend(
        path / "corpus-manifest.json"
        for path in sorted(layout.root.glob("data/base-v2-500m-filtered-pass-*"))
    )
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        return 0, 0
    manifest = _load_object(existing[-1])
    inventories = []
    for role in ("train_files", "validation_files", "attribution_files"):
        value = manifest.get(role)
        if isinstance(value, list):
            inventories.extend(entry for entry in value if isinstance(entry, dict))
    size = sum(int(entry.get("size", 0)) for entry in inventories)
    documents = int(manifest.get("actual_train_documents") or 0) + int(
        manifest.get("actual_validation_documents") or 0
    )
    if documents <= 0:
        sources = manifest.get("sources")
        if isinstance(sources, list):
            documents = sum(
                int(source.get("train_rows", 0)) + int(source.get("validation_rows", 0))
                for source in sources
                if isinstance(source, dict)
            )
    return size, documents


def _planning_eta(layout: Layout, name: str) -> tuple[float | None, str]:
    if name == "preflight":
        return 2.0, "small local identity and benchmark SHA verification"
    if name == "build-base":
        rate = _historical_build_rate(layout)
        completed = _committed_build_tokens(layout.extracted)
        if rate:
            return (
                max(0, EXPECTED_TRAIN_TOKENS + EXPECTED_VALIDATION_TOKENS - completed) / rate,
                "measured immutable Base-v3 committed-token rate",
            )
        return None, "awaiting first committed chunk"
    size, documents = _candidate_size_and_documents(layout)
    if name == "inspect-base":
        return (
            size / (500 * 1024**2) if size else None,
            "planning estimate from authenticated corpus bytes",
        )
    if name.startswith("audit-pass-"):
        return (
            documents / PLANNING_AUDIT_DOCUMENTS_PER_SECOND if documents else None,
            "conservative planning rate; audit scanner has no partial certification",
        )
    if name.startswith("materialize-pass-"):
        return (
            size / PLANNING_MATERIALIZE_BYTES_PER_SECOND if size else None,
            "planning estimate from authenticated corpus bytes",
        )
    if name.startswith("refill-plan-"):
        return (
            documents / PLANNING_AUDIT_DOCUMENTS_PER_SECOND if documents else None,
            "streaming attribution/rejection accounting; no network or GPU",
        )
    if name.startswith("refill-build-"):
        return None, "runtime targets and per-source cursors are bound by the refill plan"
    if name == "prepare-train":
        return (
            EXPECTED_TRAIN_TOKENS / PLANNING_PREPARE_TOKENS_PER_SECOND,
            "conservative tokenizer planning rate; shard completion is authoritative",
        )
    return None, "no planning estimate available"


def _begin_phase(
    layout: Layout,
    name: str,
    command: list[str] | None,
    *,
    baseline_tokens: int | None = None,
) -> None:
    state = _read_status(layout)
    previous = state.get("current_phase")
    history = state.setdefault("history", [])
    if isinstance(previous, dict) and previous.get("status") == "running":
        previous = {**previous, "status": "interrupted", "ended_at": _utc_now()}
        history.append(previous)
    initial_eta, eta_basis = _planning_eta(layout, name)
    state["status"] = "running"
    state["current_phase"] = {
        "name": name,
        "status": "running",
        "started_at": _utc_now(),
        "started_monotonic": time.monotonic(),
        "baseline_tokens": baseline_tokens,
        "command": _redacted_command(command) if command else None,
        "eta_seconds": initial_eta,
        "eta_basis": eta_basis,
    }
    _write_status(layout, state)


def _finish_phase(
    layout: Layout,
    *,
    status: str,
    exit_code: int,
    outputs: list[Path] | None = None,
    error: str | None = None,
) -> None:
    state = _read_status(layout)
    current = state.get("current_phase")
    if not isinstance(current, dict):
        raise PipelineError("cannot finish a phase without current phase metadata")
    started_monotonic = current.pop("started_monotonic", None)
    elapsed = (
        max(0.0, time.monotonic() - float(started_monotonic))
        if isinstance(started_monotonic, (int, float))
        else None
    )
    current.update(
        {
            "status": status,
            "ended_at": _utc_now(),
            "elapsed_seconds": elapsed,
            "eta_seconds": 0.0 if status in {"complete", "verified_existing"} else None,
            "exit_code": exit_code,
            "outputs": [_identity(path) for path in (outputs or []) if path.is_file()],
            "error": error,
        }
    )
    phase_index = len(state.setdefault("history", []))
    marker_suffix = "COMPLETE" if status in {"complete", "verified_existing"} else "STATE"
    marker_path = (
        layout.state_root / "phases" / f"{phase_index:03d}-{current['name']}.{marker_suffix}.json"
    )
    _atomic_json(
        marker_path,
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "twen_base_v2_500m_phase_state",
            "phase": current,
        },
    )
    current["phase_marker"] = _identity(marker_path)
    state["history"].append(current)
    state["current_phase"] = None
    state["status"] = "running" if status in {"complete", "verified_existing"} else status
    _write_status(layout, state)


def _record_in_process_phase(
    layout: Layout,
    name: str,
    callback: Any,
    *,
    outputs: list[Path] | None = None,
) -> Any:
    _begin_phase(layout, name, None)
    try:
        result = callback()
    except BaseException as error:
        _finish_phase(layout, status="failed", exit_code=1, error=str(error))
        raise
    _finish_phase(layout, status="complete", exit_code=0, outputs=outputs)
    return result


def _run_command(
    layout: Layout,
    name: str,
    command: list[str],
    *,
    outputs: list[Path],
    baseline_tokens: int | None = None,
) -> None:
    logs = layout.state_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{len(_read_status(layout).get('history', [])):03d}-{name}.log"
    _begin_phase(layout, name, command, baseline_tokens=baseline_tokens)
    state = _read_status(layout)
    assert isinstance(state.get("current_phase"), dict)
    state["current_phase"]["log"] = str(log_path.resolve())
    _write_status(layout, state)
    environment = os.environ.copy()
    source_root = str(layout.root / "src")
    environment["PYTHONPATH"] = (
        source_root
        if not environment.get("PYTHONPATH")
        else source_root + os.pathsep + environment["PYTHONPATH"]
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{_utc_now()}] command={json.dumps(_redacted_command(command))}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=layout.root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
        assert process.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while block := os.read(process.stdout.fileno(), 64 * 1024):
            rendered = decoder.decode(block)
            sys.stdout.write(rendered)
            sys.stdout.flush()
            log.write(rendered)
            log.flush()
        tail = decoder.decode(b"", final=True)
        if tail:
            sys.stdout.write(tail)
            sys.stdout.flush()
            log.write(tail)
            log.flush()
        exit_code = process.wait()
        log.write(f"[{_utc_now()}] exit_code={exit_code}\n")
    if exit_code == 75:
        _finish_phase(layout, status="stopped", exit_code=exit_code, error="STOP honored")
        raise PipelineStopped("extractor STOP honored; remove STOP and rerun the identical command")
    if exit_code != 0:
        _finish_phase(
            layout,
            status="failed",
            exit_code=exit_code,
            error=f"command failed; inspect {log_path}",
        )
        raise PipelineError(f"phase {name} failed with exit code {exit_code}: {log_path}")
    _finish_phase(layout, status="complete", exit_code=0, outputs=outputs)


def _python_command(args: argparse.Namespace, *items: str) -> list[str]:
    return [str(args.python), "-m", "twen", *items]


def _extracted_summary(manifest_path: Path) -> dict[str, Any]:
    bound = _verify_complete_bound_manifest(manifest_path)
    value = _load_object(manifest_path)
    return {
        **bound,
        "profile": value.get("profile"),
        "actual_train_tokens": value.get("actual_train_tokens"),
        "actual_validation_tokens": value.get("actual_validation_tokens"),
        "corpus_fingerprint": value.get("corpus_fingerprint"),
    }


def _verify_sparse_build(layout: Layout) -> dict[str, Any]:
    manifest = layout.extracted / "corpus-manifest.json"
    summary = _extracted_summary(manifest)
    value = _load_object(manifest)
    if value.get("profile") != "sparse":
        raise PipelineError("500M extracted corpus is not the sparse profile")
    expected = {
        "recipe_sha256": EXPECTED_RECIPE_SHA256,
        "resolved_source_lock_sha256": EXPECTED_RESOLVED_LOCK_SHA256,
        "tokenizer_manifest_sha256": TOKENIZER_MANIFEST_SHA256,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise PipelineError(f"500M extracted corpus {field} mismatch")
    if int(value.get("actual_train_tokens", 0)) < EXPECTED_TRAIN_TOKENS:
        raise PipelineError("500M extracted corpus is below its train quota")
    if int(value.get("actual_validation_tokens", 0)) < EXPECTED_VALIDATION_TOKENS:
        raise PipelineError("500M extracted corpus is below its validation quota")
    return summary


def _audit_value(attestation: Path) -> dict[str, Any]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from twen.data.audits import validate_base_audit_attestation

    value = validate_base_audit_attestation(attestation)
    return dict(value)


def _refill_plan_value(plan: Path) -> dict[str, Any]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from twen.data.refill import validate_refill_plan

    return dict(validate_refill_plan(plan))


def _refill_lineage_value(manifest: Path) -> dict[str, Any]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from twen.data.refill import validate_refill_lineage

    return dict(validate_refill_lineage(manifest))


def _quota_gate(
    layout: Layout,
    *,
    train_manifest: Path,
    validation_manifest: Path,
) -> dict[str, Any]:
    """Independently enforce every recipe quota from attribution ledgers."""

    sys.path.insert(0, str(layout.root / "src"))
    from twen.data.refill import corpus_tokens_by_source
    from twen.data.sources import load_base_data_recipe

    recipe = load_base_data_recipe(layout.recipe)
    train = corpus_tokens_by_source(train_manifest)
    validation = (
        train
        if train_manifest.resolve() == validation_manifest.resolve()
        else corpus_tokens_by_source(validation_manifest)
    )
    train_sources = train.get("sources")
    validation_sources = validation.get("sources")
    if not isinstance(train_sources, dict) or not isinstance(validation_sources, dict):
        raise PipelineError("quota accounting source summaries are invalid")
    rows: list[dict[str, Any]] = []
    passed = True
    profile = _load_object(train_manifest).get("profile")
    # Audit-filtered profiles retain the original profile as their prefix.
    recipe_profile = next(
        (name for name in recipe.profiles if str(profile).startswith(name)),
        None,
    )
    if recipe_profile is None:
        raise PipelineError(f"cannot recover recipe profile from {profile!r}")
    for source in recipe.sources:
        train_value = train_sources.get(source.source_id)
        validation_value = validation_sources.get(source.source_id)
        if not isinstance(train_value, dict) or not isinstance(validation_value, dict):
            raise PipelineError(f"quota accounting is missing source {source.source_id}")
        train_tokens = int(train_value.get("train_tokens", 0))
        validation_tokens = int(validation_value.get("validation_tokens", 0))
        train_quota = int(source.train_token_quotas[recipe_profile])
        validation_quota = int(source.validation_token_quota)
        source_passed = train_tokens >= train_quota and validation_tokens >= validation_quota
        passed = passed and source_passed
        rows.append(
            {
                "source_id": source.source_id,
                "train_tokens": train_tokens,
                "train_quota": train_quota,
                "train_deficit": max(0, train_quota - train_tokens),
                "validation_tokens": validation_tokens,
                "validation_quota": validation_quota,
                "validation_deficit": max(0, validation_quota - validation_tokens),
                "passed": source_passed,
            }
        )
    train_total = sum(int(row["train_tokens"]) for row in rows)
    validation_total = sum(int(row["validation_tokens"]) for row in rows)
    expected_train = int(recipe.profiles[recipe_profile])
    expected_validation = int(recipe.validation_tokens)
    passed = passed and train_total >= expected_train and validation_total >= expected_validation
    return {
        "passed": passed,
        "profile": recipe_profile,
        "train_manifest": _identity(train_manifest),
        "validation_manifest": _identity(validation_manifest),
        "sources": rows,
        "totals": {
            "train_tokens": train_total,
            "train_quota": expected_train,
            "train_deficit": max(0, expected_train - train_total),
            "validation_tokens": validation_total,
            "validation_quota": expected_validation,
            "validation_deficit": max(0, expected_validation - validation_total),
            "passed": train_total >= expected_train and validation_total >= expected_validation,
        },
        "gate": "every source and aggregate train/validation quota must pass",
    }


def _prepared_value(layout: Layout) -> dict[str, Any]:
    sys.path.insert(0, str(layout.root / "src"))
    from twen.data.prepared import validate_prepared_corpus

    value = validate_prepared_corpus(layout.prepared / "manifest.json")
    return value.to_dict()


def _committed_build_tokens(extracted: Path) -> int:
    manifest = extracted / "corpus-manifest.json"
    if manifest.is_file() and (extracted / "COMPLETE").is_file():
        value = _load_object(manifest)
        return int(value.get("actual_train_tokens", 0)) + int(
            value.get("actual_validation_tokens", 0)
        )
    committed = 0
    for chunk in extracted.glob("extracted/*/chunk-*/chunk.json"):
        if not (chunk.parent / "COMPLETE").is_file():
            continue
        try:
            value = _load_object(chunk)
        except PipelineError:
            continue
        committed += int(value.get("train_tokens", 0)) + int(value.get("validation_tokens", 0))
    return committed


def progress_eta(
    *, current: int, total: int, baseline: int, elapsed_seconds: float
) -> dict[str, Any]:
    delta = max(0, current - baseline)
    rate = delta / elapsed_seconds if elapsed_seconds > 0 and delta > 0 else None
    eta = max(0, total - current) / rate if rate else None
    return {
        "completed": current,
        "total": total,
        "fraction": min(1.0, current / total) if total > 0 else None,
        "rate_per_second": rate,
        "eta_seconds": eta,
        "basis": "tokens committed in the current build attempt",
    }


def snapshot(layout: Layout) -> dict[str, Any]:
    state = _read_status(layout)
    build_tokens = _committed_build_tokens(layout.extracted)
    current = state.get("current_phase")
    build_progress: dict[str, Any]
    if isinstance(current, dict) and current.get("name") == "build-base":
        started = current.get("started_at")
        try:
            elapsed = (datetime.now(UTC) - datetime.fromisoformat(str(started))).total_seconds()
        except ValueError:
            elapsed = 0.0
        build_progress = progress_eta(
            current=build_tokens,
            total=EXPECTED_TRAIN_TOKENS + EXPECTED_VALIDATION_TOKENS,
            baseline=int(current.get("baseline_tokens") or 0),
            elapsed_seconds=max(0.0, elapsed),
        )
        if build_progress["eta_seconds"] is None:
            planned_eta, basis = _planning_eta(layout, "build-base")
            build_progress["eta_seconds"] = planned_eta
            build_progress["basis"] = basis
    else:
        build_progress = progress_eta(
            current=build_tokens,
            total=EXPECTED_TRAIN_TOKENS + EXPECTED_VALIDATION_TOKENS,
            baseline=build_tokens,
            elapsed_seconds=0.0,
        )
        if build_tokens >= EXPECTED_TRAIN_TOKENS + EXPECTED_VALIDATION_TOKENS:
            build_progress["eta_seconds"] = 0.0
            build_progress["basis"] = "completed"
        else:
            planned_eta, basis = _planning_eta(layout, "build-base")
            build_progress["eta_seconds"] = planned_eta
            build_progress["basis"] = basis
    audits: list[dict[str, Any]] = []
    for path in sorted((layout.root / "artifacts/data").glob("base-v2-500m-audit-pass-*")):
        attestation = path / "attestation.json"
        if not attestation.is_file():
            audits.append({"path": str(path), "complete": False})
            continue
        try:
            value = _audit_value(attestation)
            audits.append(
                {
                    "path": str(path.resolve()),
                    "complete": True,
                    "attestation": _identity(attestation),
                    "ready_for_training": value.get("ready_for_training"),
                    "gates": value.get("gates"),
                    "metrics": value.get("metrics"),
                    "eta_seconds": 0.0,
                }
            )
        except Exception as error:
            audits.append({"path": str(path), "complete": False, "error": str(error)})
    prepared_manifest = layout.prepared / "manifest.json"
    complete = layout.state_root / "COMPLETE"
    return {
        "ok": True,
        "observed_at": _utc_now(),
        "pipeline": state,
        "performance_gate": _performance_gate(layout),
        "build": {
            "path": str(layout.extracted),
            "complete": (layout.extracted / "COMPLETE").is_file(),
            "progress": build_progress,
        },
        "audits": audits,
        "prepared": {
            "path": str(layout.prepared),
            "complete": prepared_manifest.is_file(),
            "manifest": _identity(prepared_manifest) if prepared_manifest.is_file() else None,
        },
        "complete": complete.is_file(),
        "complete_marker": _identity(complete) if complete.is_file() else None,
        "training_started": False,
        "gpu_kd_started": False,
    }


def _mark_verified(layout: Layout, name: str, outputs: list[Path]) -> None:
    _begin_phase(layout, name, None)
    _finish_phase(layout, status="verified_existing", exit_code=0, outputs=outputs)


def run_pipeline(layout: Layout, args: argparse.Namespace) -> dict[str, Any]:
    preflight_report = _record_in_process_phase(
        layout,
        "preflight",
        lambda: preflight(layout, require_performance_gate=True),
    )
    state = _read_status(layout)
    state["preflight"] = preflight_report
    _write_status(layout, state)

    extracted_manifest = layout.extracted / "corpus-manifest.json"
    if extracted_manifest.is_file() and (layout.extracted / "COMPLETE").is_file():
        _verify_sparse_build(layout)
        _mark_verified(layout, "build-base", [extracted_manifest, layout.extracted / "COMPLETE"])
    else:
        baseline = _committed_build_tokens(layout.extracted)
        command = _python_command(
            args,
            "data",
            "build-base",
            "--recipe",
            str(layout.recipe),
            "--resolved-lock",
            str(layout.resolved_lock),
            "--output",
            str(layout.extracted),
            "--tokenizer",
            str(layout.tokenizer),
            "--tokenizer-manifest-sha256",
            TOKENIZER_MANIFEST_SHA256,
            "--profile",
            "sparse",
            "--network-policy",
            "fallback",
            "--range-block-mib",
            str(args.range_block_mib),
            "--stop-file",
            str(layout.extracted / "STOP"),
            "--progress",
            args.progress,
        )
        if args.proxy:
            command.extend(["--proxy", args.proxy])
        _run_command(
            layout,
            "build-base",
            command,
            outputs=[extracted_manifest, layout.extracted / "COMPLETE"],
            baseline_tokens=baseline,
        )
        _verify_sparse_build(layout)

    inspect_log_marker = layout.state_root / "inspect-base.COMPLETE.json"
    if inspect_log_marker.is_file():
        marker = _load_object(inspect_log_marker)
        if marker.get("manifest_sha256") == _sha256(extracted_manifest):
            _mark_verified(layout, "inspect-base", [inspect_log_marker])
        else:
            inspect_log_marker.unlink()
    if not inspect_log_marker.is_file():
        _run_command(
            layout,
            "inspect-base",
            _python_command(
                args,
                "data",
                "inspect-base",
                "--manifest",
                str(extracted_manifest),
            ),
            outputs=[],
        )
        _atomic_json(
            inspect_log_marker,
            {
                "kind": "twen_base_v2_500m_inspect_complete",
                "manifest_sha256": _sha256(extracted_manifest),
                "completed_at": _utc_now(),
            },
        )

    candidate_manifest = extracted_manifest
    frozen_manifest = layout.frozen_validation_manifest
    accepted_attestation: Path | None = None
    refill_generation = 0
    refill_plans: list[dict[str, Any]] = []
    for pass_index in range(args.max_audit_passes):
        audit_root = layout.audit(pass_index)
        attestation = audit_root / "attestation.json"
        if attestation.is_file() and (audit_root / "COMPLETE").is_file():
            value = _audit_value(attestation)
            candidate_identity = value.get("candidate", {})
            frozen_identity = value.get("frozen_validation", {})
            if Path(str(candidate_identity.get("manifest_path"))).resolve() != candidate_manifest:
                raise PipelineError(f"audit pass {pass_index} binds a different candidate")
            if Path(str(frozen_identity.get("manifest_path"))).resolve() != frozen_manifest:
                raise PipelineError(f"audit pass {pass_index} binds different frozen validation")
            _mark_verified(
                layout, f"audit-pass-{pass_index:03d}", [attestation, audit_root / "COMPLETE"]
            )
        else:
            if audit_root.exists():
                raise PipelineError(
                    f"incomplete/invalid audit output is preserved; inspect and choose a new path: {audit_root}"
                )
            _run_command(
                layout,
                f"audit-pass-{pass_index:03d}",
                _python_command(
                    args,
                    "data",
                    "audit-base",
                    "--extracted-manifest",
                    str(candidate_manifest),
                    "--frozen-validation-manifest",
                    str(frozen_manifest),
                    "--benchmark-registry",
                    str(layout.benchmark_registry),
                    "--benchmark-root",
                    str(layout.benchmark_root),
                    "--near-duplicate-threshold",
                    str(args.near_duplicate_threshold),
                    "--max-findings",
                    str(args.max_findings),
                    "--output",
                    str(audit_root),
                ),
                outputs=[attestation, audit_root / "COMPLETE"],
            )
            value = _audit_value(attestation)
        if value.get("ready_for_training") is True:
            accepted_attestation = attestation
            break

        filtered_root = layout.filtered(pass_index + 1)
        filtered_manifest = filtered_root / "corpus-manifest.json"
        if filtered_manifest.is_file() and (filtered_root / "COMPLETE").is_file():
            _extracted_summary(filtered_manifest)
            filtered_value = _load_object(filtered_manifest)
            if filtered_value.get("profile") != (
                f"{_load_object(candidate_manifest).get('profile')}-audit-filtered-"
                f"{value['attestation_fingerprint'][:12]}"
            ):
                raise PipelineError(f"filtered pass {pass_index + 1} lineage mismatch")
            _mark_verified(
                layout,
                f"materialize-pass-{pass_index + 1:03d}",
                [filtered_manifest, filtered_root / "COMPLETE"],
            )
        else:
            if filtered_root.exists():
                raise PipelineError(
                    "incomplete/invalid filtered output is preserved; inspect and choose a new path: "
                    f"{filtered_root}"
                )
            _run_command(
                layout,
                f"materialize-pass-{pass_index + 1:03d}",
                _python_command(
                    args,
                    "data",
                    "materialize-audit",
                    "--audit-attestation",
                    str(attestation),
                    "--output",
                    str(filtered_root),
                ),
                outputs=[filtered_manifest, filtered_root / "COMPLETE"],
            )
            _extracted_summary(filtered_manifest)
        filtered_quota_gate = _quota_gate(
            layout,
            train_manifest=filtered_manifest,
            validation_manifest=filtered_manifest,
        )
        if not filtered_quota_gate["passed"]:
            refill_generation += 1
            if refill_generation > args.max_refill_rounds:
                raise PipelineError(
                    f"clean quotas did not converge within {args.max_refill_rounds} refill rounds"
                )
            # A refill plan can only extend an immutable raw chunk lineage.  A
            # filtered projection has no remote row cursor and must never be
            # treated as if it did.
            candidate_value = _load_object(candidate_manifest)
            if (
                not str(candidate_value.get("profile", "")).startswith("sparse")
                or not (candidate_manifest.parent / "extracted").is_dir()
            ):
                raise PipelineError(
                    "a secondary filtered audit fell below quota; no raw cursor is available "
                    "for safe refill"
                )
            plan_root = layout.refill_plan(refill_generation)
            plan_path = plan_root / "plan.json"
            if plan_path.is_file() and (plan_root / "COMPLETE").is_file():
                plan_value = _refill_plan_value(plan_path)
                if (
                    Path(str(plan_value["audit_attestation"]["path"])).resolve()
                    != attestation.resolve()
                    or Path(str(plan_value["base_raw_manifest"]["path"])).resolve()
                    != candidate_manifest.resolve()
                    or Path(str(plan_value["materialized_manifest"]["path"])).resolve()
                    != filtered_manifest.resolve()
                ):
                    raise PipelineError(f"refill plan {refill_generation} binds different inputs")
                _mark_verified(
                    layout,
                    f"refill-plan-{refill_generation:03d}",
                    [plan_path, plan_root / "COMPLETE"],
                )
            else:
                if plan_root.exists():
                    raise PipelineError(
                        f"incomplete refill plan is preserved; inspect it: {plan_root}"
                    )
                _run_command(
                    layout,
                    f"refill-plan-{refill_generation:03d}",
                    _python_command(
                        args,
                        "data",
                        "plan-base-refill",
                        "--audit-attestation",
                        str(attestation),
                        "--base-raw-manifest",
                        str(candidate_manifest),
                        "--materialized-manifest",
                        str(filtered_manifest),
                        "--recipe",
                        str(layout.recipe),
                        "--output",
                        str(plan_root),
                        "--clean-guard-ratio",
                        str(args.refill_clean_guard_ratio),
                        "--survival-guard-points",
                        str(args.refill_survival_guard_points),
                    ),
                    outputs=[plan_path, plan_root / "COMPLETE"],
                )
                plan_value = _refill_plan_value(plan_path)
            refill_root = layout.refill_raw(refill_generation)
            refill_manifest = refill_root / "corpus-manifest.json"
            if refill_manifest.is_file() and (refill_root / "COMPLETE").is_file():
                lineage = _refill_lineage_value(refill_manifest)
                if Path(str(lineage["plan"]["path"])).resolve() != plan_path.resolve():
                    raise PipelineError(f"refill raw {refill_generation} binds a different plan")
                _mark_verified(
                    layout,
                    f"refill-build-{refill_generation:03d}",
                    [refill_manifest, refill_root / "COMPLETE"],
                )
            else:
                build_refill_command = _python_command(
                    args,
                    "data",
                    "build-base-refill",
                    "--plan",
                    str(plan_path),
                    "--resolved-lock",
                    str(layout.resolved_lock),
                    "--output",
                    str(refill_root),
                    "--tokenizer",
                    str(layout.tokenizer),
                    "--tokenizer-manifest-sha256",
                    TOKENIZER_MANIFEST_SHA256,
                    "--network-policy",
                    "fallback",
                    "--range-block-mib",
                    str(args.range_block_mib),
                    "--stop-file",
                    str(refill_root / "STOP"),
                    "--progress",
                    args.progress,
                )
                if args.proxy:
                    build_refill_command.extend(["--proxy", args.proxy])
                _run_command(
                    layout,
                    f"refill-build-{refill_generation:03d}",
                    build_refill_command,
                    outputs=[refill_manifest, refill_root / "COMPLETE"],
                    baseline_tokens=_committed_build_tokens(refill_root),
                )
                lineage = _refill_lineage_value(refill_manifest)
            refill_plans.append(
                {
                    "generation": refill_generation,
                    "plan": _identity(plan_path),
                    "plan_fingerprint": plan_value["plan_fingerprint"],
                    "runtime_targets": plan_value["runtime_targets"],
                    "raw_manifest": _identity(refill_manifest),
                    "clean_quota_gate_before_refill": filtered_quota_gate,
                }
            )
            candidate_manifest = refill_manifest.resolve()
            frozen_manifest = candidate_manifest
            continue
        candidate_manifest = filtered_manifest.resolve()
        frozen_manifest = candidate_manifest
    if accepted_attestation is None:
        raise PipelineError(
            f"audit gates did not converge within {args.max_audit_passes} passes; no prepare was run"
        )

    final_quota_gate = _quota_gate(
        layout,
        train_manifest=candidate_manifest,
        validation_manifest=frozen_manifest,
    )
    if not final_quota_gate["passed"]:
        raise PipelineError(
            "accepted audit is below one or more original per-source/aggregate quotas; "
            "prepare is forbidden"
        )

    prepared_manifest = layout.prepared / "manifest.json"
    if prepared_manifest.is_file():
        prepared = _prepared_value(layout)
        lineage = prepared.get("lineage")
        if not isinstance(lineage, dict) or lineage.get("ready_for_training") is not True:
            raise PipelineError("existing prepared corpus is not attested ready_for_training")
        audit_lineage = lineage.get("audit_attestation")
        if not isinstance(audit_lineage, dict) or audit_lineage.get("sha256") != _sha256(
            accepted_attestation
        ):
            raise PipelineError("existing prepared corpus binds a different audit attestation")
        _mark_verified(layout, "prepare-train", [prepared_manifest])
    else:
        _run_command(
            layout,
            "prepare-train",
            _python_command(
                args,
                "data",
                "prepare",
                "--extracted-manifest",
                str(candidate_manifest),
                "--role",
                "train",
                "--audit-attestation",
                str(accepted_attestation),
                "--output",
                str(layout.prepared),
                "--tokenizer",
                str(layout.tokenizer),
                "--tokenizer-manifest-sha256",
                TOKENIZER_MANIFEST_SHA256,
                "--sequence-length",
                "4096",
                "--progress",
                args.progress,
            ),
            outputs=[prepared_manifest],
        )
        _prepared_value(layout)

    final_status = _read_status(layout)
    final_status.update(
        {
            "status": "complete",
            "completed_at": _utc_now(),
            "accepted_attestation": _identity(accepted_attestation),
            "final_quota_gate": final_quota_gate,
            "refill_plans": refill_plans,
            "prepared_manifest": _identity(prepared_manifest),
            "training_started": False,
            "gpu_kd_started": False,
            "next": "coordinate GPU teacher-KD generation; do not start training automatically",
        }
    )
    _write_status(layout, final_status)
    status_identity = _identity(_status_path(layout))
    _atomic_json(
        layout.state_root / "COMPLETE",
        {
            "schema_version": SCHEMA_VERSION,
            "kind": PIPELINE_COMPLETE_KIND,
            "status": status_identity,
            "accepted_attestation": _identity(accepted_attestation),
            "final_quota_gate": final_quota_gate,
            "refill_plans": refill_plans,
            "prepared_manifest": _identity(prepared_manifest),
            "training_started": False,
            "gpu_kd_started": False,
        },
    )
    return snapshot(layout)


def planned_commands(layout: Layout, args: argparse.Namespace) -> dict[str, Any]:
    build = _python_command(
        args,
        "data",
        "build-base",
        "--recipe",
        str(layout.recipe),
        "--resolved-lock",
        str(layout.resolved_lock),
        "--output",
        str(layout.extracted),
        "--tokenizer",
        str(layout.tokenizer),
        "--tokenizer-manifest-sha256",
        TOKENIZER_MANIFEST_SHA256,
        "--profile",
        "sparse",
        "--network-policy",
        "fallback",
        "--stop-file",
        str(layout.extracted / "STOP"),
        "--progress",
        args.progress,
    )
    if args.proxy:
        build.extend(["--proxy", args.proxy])
    return {
        "ok": True,
        "run": [str(args.python), str(Path(__file__).resolve()), "--action", "run"],
        "main_task_only_approval": [
            str(args.python),
            str(Path(__file__).resolve()),
            "--action",
            "approve-performance",
            "--acknowledge-native-mtp-attention-fix",
        ],
        "status": [str(args.python), str(Path(__file__).resolve()), "--action", "status"],
        "safe_stop": ["touch", str(layout.extracted / "STOP")],
        "resume": [
            "remove the STOP file after the current chunk commits",
            "rerun the exact pipeline --action run command",
        ],
        "first_build_command": _redacted_command(build),
        "audit_pattern": str(layout.audit(0)).replace("000", "NNN"),
        "filtered_pattern": str(layout.filtered(1)).replace("001", "NNN"),
        "refill_plan_pattern": str(layout.refill_plan(1)).replace("001", "NNN"),
        "refill_raw_pattern": str(layout.refill_raw(1)).replace("001", "NNN"),
        "refill_policy": {
            "clean_guard_ratio": args.refill_clean_guard_ratio,
            "survival_guard_points": args.refill_survival_guard_points,
            "per_source_and_aggregate_quotas_are_hard_gates": True,
            "original_recipe_and_chunk_fingerprints_unchanged": True,
            "network_policy": "fallback (HF direct first, proxy second)",
        },
        "prepared": str(layout.prepared),
        "training_started": False,
        "gpu_kd_started": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=("status", "preflight", "plan", "approve-performance", "run"),
        default="status",
        help="run is the only action that mutates corpus/audit/prepared outputs",
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--proxy", help="optional fallback proxy; omitted from status/log command text"
    )
    parser.add_argument("--range-block-mib", type=int, default=8)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.8)
    parser.add_argument("--max-findings", type=int, default=10_000)
    parser.add_argument("--max-audit-passes", type=int, default=32)
    parser.add_argument("--max-refill-rounds", type=int, default=8)
    parser.add_argument("--refill-clean-guard-ratio", type=float, default=0.02)
    parser.add_argument("--refill-survival-guard-points", type=float, default=0.01)
    parser.add_argument("--progress", choices=("auto", "always", "never"), default="always")
    parser.add_argument(
        "--acknowledge-native-mtp-attention-fix",
        action="store_true",
        help="required for main-task creation of the immutable post-fix performance approval",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.range_block_mib <= 0:
        raise PipelineError("range-block-mib must be positive")
    if not 0.0 < args.near_duplicate_threshold <= 1.0:
        raise PipelineError("near-duplicate-threshold must be in (0, 1]")
    if args.max_findings < 0:
        raise PipelineError("max-findings must be non-negative")
    if not 1 <= args.max_audit_passes <= 64:
        raise PipelineError("max-audit-passes must be between 1 and 64")
    if not 1 <= args.max_refill_rounds <= 8:
        raise PipelineError("max-refill-rounds must be between 1 and 8")
    if not math.isfinite(args.refill_clean_guard_ratio) or args.refill_clean_guard_ratio < 0.02:
        raise PipelineError("refill-clean-guard-ratio must be finite and at least 0.02")
    if (
        not math.isfinite(args.refill_survival_guard_points)
        or not 0.01 <= args.refill_survival_guard_points < 1.0
    ):
        raise PipelineError("refill-survival-guard-points must be finite and in [0.01, 1)")
    if args.acknowledge_native_mtp_attention_fix and args.action != "approve-performance":
        raise PipelineError(
            "--acknowledge-native-mtp-attention-fix is only valid with approve-performance"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_args(args)
        layout = Layout.repository_defaults(args.root)
        if args.action == "status":
            result = snapshot(layout)
        elif args.action == "preflight":
            result = preflight(layout, require_performance_gate=False)
        elif args.action == "plan":
            result = planned_commands(layout, args)
        elif args.action == "approve-performance":
            if not args.acknowledge_native_mtp_attention_fix:
                raise PipelineError(
                    "approve-performance requires --acknowledge-native-mtp-attention-fix"
                )
            result = write_performance_approval(layout)
        else:
            layout.state_root.mkdir(parents=True, exist_ok=True)
            lock_path = layout.state_root / "pipeline.lock"
            with lock_path.open("a+") as lock:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise PipelineError(
                        "another Base-v2 500M pipeline process is active"
                    ) from error
                result = run_pipeline(layout, args)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except PipelineStopped as error:
        print(json.dumps({"ok": False, "stopped": True, "error": str(error)}, indent=2))
        return 75
    except (PipelineError, OSError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
