"""Fail-closed orchestration for the Base-v2 500M top-64 KD artifact.

The orchestrator is intentionally separate from training.  Its only GPU child
is ``twen data generate-kd`` which uses inference mode and never constructs an
optimizer.  The final indexing/certification phase is CPU and storage only.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from .config import TrainConfig, load_train_config
from .data import (
    TeacherKDCorpusManifest,
    TeacherKDManifest,
    validate_base_audit_attestation,
    validate_kd_corpus_coverage,
    validate_kd_corpus_manifest,
    validate_kd_shard,
    validate_prepared_corpus,
)
from .data.teacher_kd import (
    KD_GENERATOR_SOURCE_SHA256,
    KD_MANIFEST_FILENAME,
    KD_TENSORS_FILENAME,
    KD_TOP_K,
)
from .preflight import _check_source
from .utils import sha256_file

SCHEMA_VERSION = 1
STATUS_KIND = "twen_base_v2_500m_kd_orchestration_status"
MANIFEST_KIND = "twen_base_v2_500m_kd_orchestration_manifest"
COMPLETE_KIND = "twen_base_v2_500m_kd_orchestration_complete"
PIPELINE_COMPLETE_KIND = "twen_base_v2_500m_pipeline_complete"
PIPELINE_STATUS_KIND = "twen_base_v2_500m_pipeline_status"

EXPECTED_MINIMUM_TOKENS = 500_000_000
EXPECTED_SEQUENCE_LENGTH = 4096
EXPECTED_TEMPERATURE = 2.0
EXPECTED_TEACHER_MODEL_ID = "Qwen/Qwen3.5-9B-Base"
DEFAULT_BATCH_SIZE = 2
DEFAULT_LOGITS_CHUNK_TOKENS = 64
DEFAULT_MINIMUM_FREE_AFTER_GIB = 64.0

# Exact dense tensor payload written by TeacherKDBatch.tensors() for each
# padded [sequence, token] position:
# input_ids I64 + labels I64 + mask BOOL + top-k indices I64[64] +
# top-k logits BF16[64] + logsumexp F32 + tail-logprob F32.
KD_BYTES_PER_PADDED_POSITION = 8 + 8 + 1 + 64 * 8 + 64 * 2 + 4 + 4

# Immutable, optimizer-free measurements already present in this repository.
# The full-corpus rate is the most honest wall estimate because it includes
# prepared reads, host copies, writes and per-shard hashing.
HISTORICAL_WALL_TOKENS_PER_SECOND = 9042.26808318264
HISTORICAL_FULL_RUN_TOKENS = 100_007_485
HISTORICAL_FULL_RUN_SEQUENCES = 24_476
HISTORICAL_FULL_RUN_BYTES = 66_669_120_553
HISTORICAL_FULL_RUN_SECONDS = 11_060.0
HISTORICAL_PADDED_POSITIONS_PER_SECOND = (
    HISTORICAL_FULL_RUN_SEQUENCES * EXPECTED_SEQUENCE_LENGTH / HISTORICAL_FULL_RUN_SECONDS
)
HISTORICAL_OVERHEAD_BYTES_PER_SHARD = math.ceil(412_713 / 120)

_BENCHMARK_FILENAMES = {
    2: "rtx5090-qwen35-9b-base-teacher-kd-batch2.json",
    4: "rtx5090-qwen35-9b-base-teacher-kd-batch4.json",
}
_EXPECTED_BENCHMARK_SHA256 = {
    2: "ef26509815d580853edf86bde30a80654fabc7a15e568d32089fee9279cf2c7b",
    4: "af278ef819ef12f2e0fcae20b514fbde5d23b932e346f54caa72972d2fa9fdad",
}
_EXPECTED_FULL_RUN_BENCHMARK_SHA256 = (
    "d7940f3450000177bee39bb708de900a15912061fd48313b9be0a2e3f3d11f0c"
)


class KDOrchestrationError(RuntimeError):
    """A prerequisite or durable artifact failed authentication."""


class KDOrchestrationStopped(KDOrchestrationError):
    """The persistent STOP request was honored at a shard boundary."""


@dataclass(frozen=True, slots=True)
class Layout:
    root: Path
    base_config: Path
    prepared_manifest: Path
    pipeline_complete: Path
    output_root: Path
    state_root: Path
    stop_file: Path
    batch2_benchmark: Path
    batch4_benchmark: Path
    full_run_benchmark: Path

    @classmethod
    def repository_defaults(cls, root: str | Path) -> Layout:
        project = Path(root).expanduser().resolve()
        output = project / "artifacts/data/base-v2-500m-kd"
        return cls(
            root=project,
            base_config=project / "configs/base/dense-oracle.yaml",
            prepared_manifest=project / "artifacts/data/base-v2-500m/manifest.json",
            pipeline_complete=project / "artifacts/data/base-v2-500m-pipeline/COMPLETE",
            output_root=output,
            state_root=project / "artifacts/data/base-v2-500m-kd-orchestration",
            stop_file=output / "STOP",
            batch2_benchmark=(
                project / "artifacts/benchmarks/rtx5090-qwen35-9b-base-teacher-kd-batch2.json"
            ),
            batch4_benchmark=(
                project / "artifacts/benchmarks/rtx5090-qwen35-9b-base-teacher-kd-batch4.json"
            ),
            full_run_benchmark=(
                project / "artifacts/benchmarks/rtx5090-base-teacher-kd-full-run.json"
            ),
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Layout:
        defaults = cls.repository_defaults(args.root)

        def resolve(value: str | Path | None, default: Path, label: str) -> Path:
            raw = Path(value) if value is not None else default
            target = (raw if raw.is_absolute() else defaults.root / raw).resolve()
            if not target.is_relative_to(defaults.root):
                raise KDOrchestrationError(f"{label} must stay inside the project root: {target}")
            return target

        output = resolve(args.output, defaults.output_root, "KD output")
        prepared_manifest = resolve(
            args.prepared_manifest,
            defaults.prepared_manifest,
            "prepared manifest",
        )
        state_root = resolve(args.state_root, defaults.state_root, "state root")

        def overlaps(left: Path, right: Path) -> bool:
            return left == right or left.is_relative_to(right) or right.is_relative_to(left)

        if output == defaults.root or overlaps(output, prepared_manifest.parent):
            raise KDOrchestrationError("KD output must not overlap the prepared corpus")
        if overlaps(output, state_root):
            raise KDOrchestrationError("KD output and orchestration state must be disjoint")
        return cls(
            root=defaults.root,
            base_config=resolve(args.base_config, defaults.base_config, "base config"),
            prepared_manifest=prepared_manifest,
            pipeline_complete=resolve(
                args.pipeline_complete,
                defaults.pipeline_complete,
                "pipeline COMPLETE",
            ),
            output_root=output,
            state_root=state_root,
            stop_file=output / "STOP",
            batch2_benchmark=defaults.batch2_benchmark,
            batch4_benchmark=defaults.batch4_benchmark,
            full_run_benchmark=defaults.full_run_benchmark,
        )


@dataclass(frozen=True, slots=True)
class AuthenticatedInputs:
    prepared: Any
    prepared_identity: dict[str, Any]
    audit: dict[str, Any]
    audit_identity: dict[str, Any]
    pipeline_complete_identity: dict[str, Any]
    pipeline_status_identity: dict[str, Any]
    config: TrainConfig
    config_identity: dict[str, Any]
    teacher_source: dict[str, Any]
    benchmark: dict[str, Any]
    full_run_benchmark: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PhaseResult:
    name: str
    command: tuple[str, ...]
    log_path: Path
    exit_code: int
    started_at: str
    ended_at: str
    elapsed_seconds: float


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_object(path: Path, *, label: str | None = None) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise KDOrchestrationError(
            f"invalid or missing {label or 'JSON object'}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise KDOrchestrationError(f"JSON root must be an object: {path}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise KDOrchestrationError(f"required file is missing: {resolved}")
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _same_identity(
    declared: object,
    actual: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if not isinstance(declared, Mapping):
        raise KDOrchestrationError(f"{label} identity is missing")
    expected = (actual["path"], actual["size"], actual["sha256"])
    observed = (declared.get("path"), declared.get("size"), declared.get("sha256"))
    if observed != expected:
        raise KDOrchestrationError(
            f"{label} identity mismatch: expected path/size/SHA {expected}, got {observed}"
        )


def _path_from_identity(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
        raise KDOrchestrationError(f"{label} has no path identity")
    raw = Path(str(value["path"]))
    path = (raw if raw.is_absolute() else root / raw).resolve()
    if not path.is_relative_to(root):
        raise KDOrchestrationError(f"{label} escapes project root: {path}")
    return path


def _validate_benchmark_evidence(
    layout: Layout,
    *,
    batch_size: int,
    logits_chunk_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    benchmark_path = {
        2: layout.batch2_benchmark,
        4: layout.batch4_benchmark,
    }.get(batch_size)
    if benchmark_path is None:
        raise KDOrchestrationError(
            f"KD batch {batch_size} has no allowlisted optimizer-free RTX 5090 evidence"
        )
    benchmark_identity = _identity(benchmark_path)
    if benchmark_identity["sha256"] != _EXPECTED_BENCHMARK_SHA256[batch_size]:
        raise KDOrchestrationError(
            "KD benchmark bytes changed; recertify before selecting this batch"
        )
    benchmark = _load_object(benchmark_path, label="KD benchmark")
    if (
        benchmark.get("ok") is not True
        or benchmark.get("no_optimizer_steps") is not True
        or benchmark.get("gpu") != "NVIDIA GeForce RTX 5090"
        or benchmark.get("model") != "artifacts/models/qwen3.5-9b-base"
        or benchmark.get("batch_size") != batch_size
        or benchmark.get("sequence_length") != EXPECTED_SEQUENCE_LENGTH
        or benchmark.get("top_k") != KD_TOP_K
        or not math.isclose(
            float(benchmark.get("temperature", math.nan)),
            EXPECTED_TEMPERATURE,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise KDOrchestrationError("KD benchmark does not authenticate the requested graph")
    measurements = benchmark.get("measurements")
    selected = None
    if isinstance(measurements, list):
        for value in measurements:
            if isinstance(value, dict) and value.get("logits_chunk_tokens") == logits_chunk_tokens:
                selected = value
                break
    if (
        not isinstance(selected, dict)
        or not math.isfinite(float(selected.get("input_tokens_per_second", math.nan)))
        or float(selected["input_tokens_per_second"]) <= 0
        or not math.isfinite(float(selected.get("peak_allocated_gib", math.nan)))
    ):
        raise KDOrchestrationError(
            f"KD chunk {logits_chunk_tokens} has no finite measurement for batch {batch_size}"
        )

    full_run_identity = _identity(layout.full_run_benchmark)
    if full_run_identity["sha256"] != _EXPECTED_FULL_RUN_BENCHMARK_SHA256:
        raise KDOrchestrationError(
            "full-corpus KD benchmark bytes changed; recertify the wall estimate"
        )
    full_run = _load_object(layout.full_run_benchmark, label="full KD benchmark")
    if (
        full_run.get("ok") is not True
        or full_run.get("no_optimizer_steps") is not True
        or full_run.get("batch_size") != DEFAULT_BATCH_SIZE
        or full_run.get("logits_chunk_tokens") != DEFAULT_LOGITS_CHUNK_TOKENS
        or full_run.get("token_count") != HISTORICAL_FULL_RUN_TOKENS
        or full_run.get("output_bytes") != HISTORICAL_FULL_RUN_BYTES
        or full_run.get("progress_elapsed_seconds") != int(HISTORICAL_FULL_RUN_SECONDS)
    ):
        raise KDOrchestrationError("full-corpus KD evidence contract changed")
    return (
        {
            "artifact": benchmark_identity,
            "measurement": dict(selected),
            "load_seconds": benchmark.get("load_seconds"),
            "no_optimizer_steps": True,
        },
        {
            "artifact": full_run_identity,
            "tokens": HISTORICAL_FULL_RUN_TOKENS,
            "bytes": HISTORICAL_FULL_RUN_BYTES,
            "elapsed_seconds": HISTORICAL_FULL_RUN_SECONDS,
            "wall_tokens_per_second": HISTORICAL_WALL_TOKENS_PER_SECOND,
            "no_optimizer_steps": True,
        },
    )


def _validate_pipeline_contract(
    layout: Layout,
    *,
    prepared_identity: dict[str, Any],
    prepared: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    complete = _load_object(layout.pipeline_complete, label="500M pipeline COMPLETE")
    if (
        complete.get("schema_version") != 1
        or complete.get("kind") != PIPELINE_COMPLETE_KIND
        or complete.get("training_started") is not False
        or complete.get("gpu_kd_started") is not False
    ):
        raise KDOrchestrationError("500M pipeline COMPLETE contract is invalid")
    _same_identity(
        complete.get("prepared_manifest"),
        prepared_identity,
        label="pipeline prepared manifest",
    )

    status_path = _path_from_identity(layout.root, complete.get("status"), label="pipeline status")
    expected_status = layout.pipeline_complete.parent / "status.json"
    if status_path != expected_status.resolve():
        raise KDOrchestrationError(
            f"pipeline status path mismatch: expected {expected_status.resolve()}, got {status_path}"
        )
    status_identity = _identity(status_path)
    _same_identity(complete.get("status"), status_identity, label="pipeline status")
    status = _load_object(status_path, label="500M pipeline status")
    if (
        status.get("schema_version") != 1
        or status.get("kind") != PIPELINE_STATUS_KIND
        or status.get("status") != "complete"
        or status.get("training_started") is not False
        or status.get("gpu_kd_started") is not False
    ):
        raise KDOrchestrationError("500M pipeline status is not a completed data-only run")
    _same_identity(
        status.get("prepared_manifest"),
        prepared_identity,
        label="pipeline status prepared manifest",
    )

    lineage = prepared.lineage
    if (
        not isinstance(lineage, Mapping)
        or lineage.get("kind") != "authenticated_extracted_corpus"
        or lineage.get("role") != "train"
        or lineage.get("ready_for_training") is not True
        or lineage.get("research_only") is not False
        or lineage.get("pending_audits") != []
    ):
        raise KDOrchestrationError(
            "prepared manifest is not an audit-attested, ready-for-training train corpus"
        )
    audit_lineage = lineage.get("audit_attestation")
    if not isinstance(audit_lineage, Mapping):
        raise KDOrchestrationError("prepared lineage has no audit attestation")
    audit_path = Path(str(audit_lineage.get("path", ""))).resolve()
    if not audit_path.is_relative_to(layout.root):
        raise KDOrchestrationError("prepared audit attestation escapes project root")
    audit_identity = _identity(audit_path)
    _same_identity(
        complete.get("accepted_attestation"),
        audit_identity,
        label="pipeline accepted attestation",
    )
    _same_identity(
        status.get("accepted_attestation"),
        audit_identity,
        label="pipeline status accepted attestation",
    )
    if (
        audit_lineage.get("sha256") != audit_identity["sha256"]
        or audit_lineage.get("bound_as") != "candidate"
        or audit_lineage.get("ready_for_training") is not True
    ):
        raise KDOrchestrationError("prepared lineage does not bind accepted audit bytes")
    audit = dict(validate_base_audit_attestation(audit_path))
    gates = audit.get("gates")
    if (
        audit.get("ready_for_training") is not True
        or audit.get("attestation_fingerprint") != audit_lineage.get("attestation_fingerprint")
        or gates != audit_lineage.get("gates")
        or not isinstance(gates, Mapping)
        or not all(
            isinstance(value, Mapping) and value.get("passed") is True for value in gates.values()
        )
    ):
        raise KDOrchestrationError("audit attestation is not a complete all-pass gate set")
    return (
        _identity(layout.pipeline_complete),
        status_identity,
        {
            "value": audit,
            "identity": audit_identity,
        },
    )


def authenticate_inputs(
    layout: Layout,
    *,
    batch_size: int,
    logits_chunk_tokens: int,
    verify_teacher_files: bool = True,
) -> AuthenticatedInputs:
    """Authenticate every durable input immediately before launching CUDA."""

    if not layout.pipeline_complete.is_file():
        raise KDOrchestrationError("500M data pipeline COMPLETE is missing")
    if not layout.prepared_manifest.is_file():
        raise KDOrchestrationError("500M prepared manifest is missing")
    prepared_identity = _identity(layout.prepared_manifest)
    prepared = validate_prepared_corpus(layout.prepared_manifest)
    if prepared.token_count < EXPECTED_MINIMUM_TOKENS:
        raise KDOrchestrationError(
            f"prepared corpus has {prepared.token_count:,} tokens; at least "
            f"{EXPECTED_MINIMUM_TOKENS:,} are required"
        )
    if prepared.sequence_length != EXPECTED_SEQUENCE_LENGTH:
        raise KDOrchestrationError(f"prepared sequence length must be {EXPECTED_SEQUENCE_LENGTH}")
    pipeline_identity, status_identity, audit_record = _validate_pipeline_contract(
        layout,
        prepared_identity=prepared_identity,
        prepared=prepared,
    )

    config_identity = _identity(layout.base_config)
    config = load_train_config(layout.base_config)
    teacher = config.sources.teacher
    tokenizer = config.sources.tokenizer
    teacher_path = Path(teacher.local_path)
    teacher_path = (
        teacher_path if teacher_path.is_absolute() else layout.root / teacher_path
    ).resolve()
    if not teacher_path.is_relative_to(layout.root):
        raise KDOrchestrationError("teacher source must stay inside the project root")
    if (
        layout.output_root == teacher_path
        or layout.output_root.is_relative_to(teacher_path)
        or teacher_path.is_relative_to(layout.output_root)
    ):
        raise KDOrchestrationError("KD output must not overlap the teacher source")
    teacher.local_path = str(teacher_path)
    if teacher.model_id != EXPECTED_TEACHER_MODEL_ID:
        raise KDOrchestrationError(
            f"teacher must remain {EXPECTED_TEACHER_MODEL_ID}, got {teacher.model_id}"
        )
    if prepared.tokenizer_sha256 != tokenizer.manifest_sha256:
        raise KDOrchestrationError("prepared/tokenizer manifest SHA mismatch")
    teacher_record: dict[str, Any] = {
        "model_id": teacher.model_id,
        "revision": teacher.revision,
        "manifest_sha256": teacher.manifest_sha256,
        "local_path": str(Path(teacher.local_path).resolve()),
        "verified_all_download_artifacts": False,
    }
    if verify_teacher_files:
        try:
            root, manifest, text_config = _check_source("teacher", teacher)
        except (OSError, RuntimeError, ValueError) as error:
            raise KDOrchestrationError(
                f"pinned teacher source failed full file authentication: {error}"
            ) from error
        if int(text_config.get("vocab_size", 0)) <= KD_TOP_K:
            raise KDOrchestrationError("teacher vocabulary is incompatible with top-64 KD")
        teacher_record.update(
            {
                "local_path": str(root.resolve()),
                "download_manifest": _identity(manifest),
                "vocab_size": int(text_config["vocab_size"]),
                "verified_all_download_artifacts": True,
            }
        )
    else:
        if not teacher_path.is_dir() or any(teacher_path.rglob("*.incomplete")):
            raise KDOrchestrationError("pinned teacher source is missing or incomplete")
        download_manifest = teacher_path / "download-manifest.json"
        download_identity = _identity(download_manifest)
        if download_identity["sha256"] != teacher.manifest_sha256:
            raise KDOrchestrationError("teacher download manifest SHA mismatch")
        teacher_record.update(
            {
                "download_manifest": download_identity,
                "full_artifact_verification_delegated_to_generate_kd": True,
            }
        )

    benchmark, full_run = _validate_benchmark_evidence(
        layout,
        batch_size=batch_size,
        logits_chunk_tokens=logits_chunk_tokens,
    )
    return AuthenticatedInputs(
        prepared=prepared,
        prepared_identity=prepared_identity,
        audit=dict(audit_record["value"]),
        audit_identity=dict(audit_record["identity"]),
        pipeline_complete_identity=pipeline_identity,
        pipeline_status_identity=status_identity,
        config=config,
        config_identity=config_identity,
        teacher_source=teacher_record,
        benchmark=benchmark,
        full_run_benchmark=full_run,
    )


def estimate_kd_storage(prepared: Any, *, shard_count: int | None = None) -> dict[str, Any]:
    """Estimate exact tensor payload from actual prepared padded geometry."""

    actual_shards = len(prepared.shards) if shard_count is None else shard_count
    padded_positions = int(prepared.sequence_count) * int(prepared.sequence_length)
    payload = padded_positions * KD_BYTES_PER_PADDED_POSITION
    overhead = actual_shards * HISTORICAL_OVERHEAD_BYTES_PER_SHARD
    estimate = payload + overhead
    lower_bound = int(prepared.token_count) * KD_BYTES_PER_PADDED_POSITION
    return {
        "valid_tokens": int(prepared.token_count),
        "sequence_count": int(prepared.sequence_count),
        "sequence_length": int(prepared.sequence_length),
        "padded_positions": padded_positions,
        "bytes_per_padded_position": KD_BYTES_PER_PADDED_POSITION,
        "unpadded_lower_bound_bytes": lower_bound,
        "tensor_payload_bytes": payload,
        "historical_overhead_bytes_per_shard": HISTORICAL_OVERHEAD_BYTES_PER_SHARD,
        "estimated_final_bytes": estimate,
        "estimated_final_decimal_gb": estimate / 1e9,
        "estimated_final_gib": estimate / 1024**3,
        "padding_overhead_bytes": payload - lower_bound,
        "basis": "actual prepared sequence_count × sequence_length × 665 plus v1 per-shard overhead",
    }


def _directory_apparent_bytes(root: Path) -> int:
    total = 0
    if not root.is_dir():
        return total
    for path in root.rglob("*"):
        with suppress(OSError):
            if path.is_file():
                total += path.stat().st_size
    return total


def _nearest_existing(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def disk_plan(
    layout: Layout,
    storage: Mapping[str, Any],
    *,
    minimum_free_after_gib: float,
) -> dict[str, Any]:
    if not math.isfinite(minimum_free_after_gib) or minimum_free_after_gib < 0:
        raise KDOrchestrationError("minimum free-after space must be finite and non-negative")
    disk = shutil.disk_usage(_nearest_existing(layout.output_root))
    existing = _directory_apparent_bytes(layout.output_root)
    final = int(storage["estimated_final_bytes"])
    remaining = max(0, final - existing)
    reserve = math.ceil(minimum_free_after_gib * 1024**3)
    required = remaining + reserve
    return {
        "filesystem_path": str(_nearest_existing(layout.output_root).resolve()),
        "total_bytes": disk.total,
        "free_bytes": disk.free,
        "existing_output_bytes": existing,
        "estimated_remaining_output_bytes": remaining,
        "minimum_free_after_bytes": reserve,
        "required_free_now_bytes": required,
        "ready": disk.free >= required,
    }


def _read_prepared_fast(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        from .data import read_prepared_manifest

        return read_prepared_manifest(path)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _shard_expected_identity(prepared: Any, entry: Any, config: TrainConfig) -> tuple[Any, ...]:
    teacher = config.sources.teacher
    return (
        teacher.model_id,
        teacher.revision,
        teacher.manifest_sha256,
        KD_GENERATOR_SOURCE_SHA256,
        config.sources.tokenizer.manifest_sha256,
        prepared.dataset_fingerprint,
        entry.shard_id,
        entry.tensors_sha256,
        entry.global_sample_start,
        entry.global_sample_end,
        entry.global_token_start,
        entry.global_token_end,
        entry.sequence_count,
        prepared.sequence_length,
        entry.token_count,
    )


def _manifest_identity(manifest: TeacherKDManifest) -> tuple[Any, ...]:
    return (
        manifest.teacher_model_id,
        manifest.teacher_revision,
        manifest.teacher_model_sha256,
        manifest.generator_source_sha256,
        manifest.tokenizer_sha256,
        manifest.dataset_fingerprint,
        manifest.source_shard_id,
        manifest.source_tensors_sha256,
        manifest.global_sample_start,
        manifest.global_sample_end,
        manifest.global_token_start,
        manifest.global_token_end,
        manifest.sequence_count,
        manifest.sequence_length,
        manifest.token_count,
    )


def _read_fast_complete_manifest(destination: Path) -> TeacherKDManifest:
    """Authenticate cheap commit metadata without rereading a multi-GB tensor."""

    marker = _load_object(destination / "COMPLETE", label="KD shard COMPLETE")
    if marker.get("schema_version") != 1 or marker.get("shard_id") != destination.name:
        raise KDOrchestrationError(f"invalid KD COMPLETE marker: {destination}")
    outputs = marker.get("outputs")
    if not isinstance(outputs, list):
        raise KDOrchestrationError(f"KD COMPLETE has no output inventory: {destination}")
    inventory = {
        item.get("path"): item
        for item in outputs
        if isinstance(item, Mapping) and isinstance(item.get("path"), str)
    }
    for filename in (KD_MANIFEST_FILENAME, KD_TENSORS_FILENAME):
        identity = inventory.get(filename)
        path = destination / filename
        if (
            not isinstance(identity, Mapping)
            or not path.is_file()
            or path.stat().st_size != identity.get("size")
        ):
            raise KDOrchestrationError(f"KD COMPLETE output size/inventory mismatch: {path}")
    value = _load_object(destination / KD_MANIFEST_FILENAME, label="KD shard manifest")
    if sha256_file(destination / KD_MANIFEST_FILENAME) != inventory[KD_MANIFEST_FILENAME].get(
        "sha256"
    ):
        raise KDOrchestrationError(f"KD manifest/COMPLETE SHA identity mismatch: {destination}")
    manifest = TeacherKDManifest.from_dict(value)
    manifest.require_temperature(EXPECTED_TEMPERATURE)
    if manifest.tensors_sha256 != inventory[KD_TENSORS_FILENAME].get("sha256"):
        raise KDOrchestrationError(
            f"KD manifest/COMPLETE tensor SHA identity mismatch: {destination}"
        )
    return manifest


def scan_progress(
    layout: Layout,
    prepared: Any,
    config: TrainConfig,
    *,
    verify_checksums: bool,
) -> dict[str, Any]:
    """Count only final, compatible COMPLETE shards; never count staging output."""

    expected_ids = {entry.shard_id for entry in prepared.shards}
    unexpected: list[str] = []
    if layout.output_root.is_dir():
        for child in layout.output_root.iterdir():
            if (
                child.is_dir()
                and not child.name.endswith(".incomplete")
                and child.name not in expected_ids
            ):
                unexpected.append(child.name)
    if unexpected:
        raise KDOrchestrationError(
            f"KD output contains unexpected final directories: {sorted(unexpected)}"
        )
    completed_shards = 0
    completed_tokens = 0
    completed_sequences = 0
    completed_bytes = 0
    for entry in prepared.shards:
        destination = layout.output_root / entry.shard_id
        if not (destination / "COMPLETE").is_file():
            continue
        manifest = (
            validate_kd_shard(
                destination,
                expected_temperature=EXPECTED_TEMPERATURE,
                verify_checksum=True,
            )
            if verify_checksums
            else _read_fast_complete_manifest(destination)
        )
        if _manifest_identity(manifest) != _shard_expected_identity(prepared, entry, config):
            raise KDOrchestrationError(
                f"existing KD shard belongs to another input contract: {entry.shard_id}"
            )
        completed_shards += 1
        completed_tokens += entry.token_count
        completed_sequences += entry.sequence_count
        completed_bytes += _directory_apparent_bytes(destination)
    total_tokens = int(prepared.token_count)
    return {
        "completed_shards": completed_shards,
        "total_shards": len(prepared.shards),
        "completed_tokens": completed_tokens,
        "total_tokens": total_tokens,
        "completed_sequences": completed_sequences,
        "total_sequences": int(prepared.sequence_count),
        "committed_output_bytes": completed_bytes,
        "fraction": completed_tokens / total_tokens,
        "percent": completed_tokens / total_tokens * 100.0,
        "remaining_tokens": total_tokens - completed_tokens,
        "checksum_mode": "full" if verify_checksums else "commit_manifest_and_size_only",
    }


def _status_path(layout: Layout) -> Path:
    return layout.state_root / "status.json"


def _read_status(layout: Layout) -> dict[str, Any]:
    path = _status_path(layout)
    if not path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": STATUS_KIND,
            "status": "not_started",
            "attempt": 0,
            "history": [],
        }
    value = _load_object(path, label="KD orchestration status")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("kind") != STATUS_KIND:
        raise KDOrchestrationError(f"unsupported KD orchestration status: {path}")
    return value


def _write_status(layout: Layout, value: Mapping[str, Any]) -> None:
    payload = dict(value)
    payload["schema_version"] = SCHEMA_VERSION
    payload["kind"] = STATUS_KIND
    payload["updated_at"] = _utc_now()
    payload["training_started"] = False
    payload["optimizer_created"] = False
    _atomic_json(_status_path(layout), payload)


def _planning_time(
    total_tokens: int,
    remaining_tokens: int,
    *,
    total_padded_positions: int,
    remaining_padded_positions: int,
) -> dict[str, Any]:
    valid_token_linear_seconds = total_tokens / HISTORICAL_WALL_TOKENS_PER_SECOND
    padded_geometry_seconds = total_padded_positions / HISTORICAL_PADDED_POSITIONS_PER_SECOND
    return {
        "historical_wall_tokens_per_second": HISTORICAL_WALL_TOKENS_PER_SECOND,
        "historical_padded_positions_per_second": (HISTORICAL_PADDED_POSITIONS_PER_SECOND),
        "valid_token_linear_seconds": valid_token_linear_seconds,
        "estimated_full_seconds": padded_geometry_seconds,
        "estimated_full_hours": padded_geometry_seconds / 3600.0,
        "estimated_remaining_seconds": (
            remaining_padded_positions / HISTORICAL_PADDED_POSITIONS_PER_SECOND
        ),
        "total_tokens": total_tokens,
        "remaining_tokens": remaining_tokens,
        "total_padded_positions": total_padded_positions,
        "remaining_padded_positions": remaining_padded_positions,
        "scope": (
            "actual prepared padded geometry scaled from the 100M full run; final "
            "independent index/SHA certification may add storage-bound time"
        ),
    }


def _public_plan(
    layout: Layout,
    inputs: AuthenticatedInputs,
    *,
    python: Path,
    batch_size: int,
    logits_chunk_tokens: int,
    minimum_free_after_gib: float,
) -> dict[str, Any]:
    storage = estimate_kd_storage(inputs.prepared)
    disk = disk_plan(layout, storage, minimum_free_after_gib=minimum_free_after_gib)
    progress = scan_progress(
        layout,
        inputs.prepared,
        inputs.config,
        verify_checksums=False,
    )
    command = generate_command(
        layout,
        inputs,
        python=python,
        batch_size=batch_size,
        logits_chunk_tokens=logits_chunk_tokens,
    )
    index = index_command(layout, python=python)
    return {
        "ok": disk["ready"],
        "ready": disk["ready"],
        "prepared": inputs.prepared_identity,
        "audit_attestation": inputs.audit_identity,
        "pipeline_complete": inputs.pipeline_complete_identity,
        "teacher": inputs.teacher_source,
        "performance_evidence": inputs.benchmark,
        "full_run_evidence": inputs.full_run_benchmark,
        "storage": storage,
        "disk": disk,
        "progress": progress,
        "time_estimate": _planning_time(
            int(inputs.prepared.token_count),
            int(progress["remaining_tokens"]),
            total_padded_positions=int(storage["padded_positions"]),
            remaining_padded_positions=(
                int(progress["total_sequences"] - progress["completed_sequences"])
                * int(inputs.prepared.sequence_length)
            ),
        ),
        "recommendation": {
            "recommended_batch_size": DEFAULT_BATCH_SIZE,
            "recommended_logits_chunk_tokens": DEFAULT_LOGITS_CHUNK_TOKENS,
            "selected_batch_size": batch_size,
            "selected_logits_chunk_tokens": logits_chunk_tokens,
            "selection_is_recommended": (
                batch_size == DEFAULT_BATCH_SIZE
                and logits_chunk_tokens == DEFAULT_LOGITS_CHUNK_TOKENS
            ),
            "reason": (
                "batch-2/chunk-64 has a complete 100M-token production run; batch-4's "
                "0.884% microbenchmark gain costs 1.101 GiB and has no full-corpus run"
            ),
            "independent_of_training_micro_batch_size": True,
            "training_micro_batch_size_ignored": inputs.config.data.micro_batch_size,
        },
        "generate_command": command,
        "index_command": index,
        "safe_stop": ["touch", str(layout.stop_file)],
        "resume": [
            "remove the persistent STOP file",
            "rerun the identical orchestration --action run command",
        ],
        "training_started": False,
        "optimizer_created": False,
    }


def generate_command(
    layout: Layout,
    inputs: AuthenticatedInputs,
    *,
    python: Path,
    batch_size: int,
    logits_chunk_tokens: int,
) -> list[str]:
    teacher = inputs.config.sources.teacher
    tokenizer = inputs.config.sources.tokenizer
    return [
        str(python),
        "-m",
        "twen",
        "data",
        "generate-kd",
        "--prepared-manifest",
        str(layout.prepared_manifest),
        "--output",
        str(layout.output_root),
        "--teacher",
        str(Path(teacher.local_path).resolve()),
        "--teacher-model-id",
        teacher.model_id,
        "--teacher-revision",
        teacher.revision,
        "--teacher-manifest-sha256",
        teacher.manifest_sha256,
        "--tokenizer-manifest-sha256",
        tokenizer.manifest_sha256,
        "--temperature",
        f"{EXPECTED_TEMPERATURE:g}",
        "--batch-size",
        str(batch_size),
        "--logits-chunk-tokens",
        str(logits_chunk_tokens),
        "--device",
        "cuda:0",
        "--stop-file",
        str(layout.stop_file),
        "--progress",
        "always",
    ]


def index_command(layout: Layout, *, python: Path) -> list[str]:
    return [
        str(python),
        "-m",
        "twen",
        "data",
        "index-kd",
        "--root",
        str(layout.output_root),
        "--output",
        str(layout.output_root / "manifest.json"),
        "--prepared-manifest",
        str(layout.prepared_manifest),
        "--temperature",
        f"{EXPECTED_TEMPERATURE:g}",
    ]


@contextmanager
def _exclusive_lock(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise KDOrchestrationError(
                "another Base-v2 500M KD orchestration process is active"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started_at={_utc_now()}\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _tee_stream(
    source: BinaryIO,
    destinations: tuple[BinaryIO, ...],
    errors: list[BaseException],
) -> None:
    try:
        while True:
            block = source.read(64 * 1024)
            if not block:
                break
            for destination in destinations:
                destination.write(block)
                destination.flush()
    except BaseException as error:  # handed back to the owning orchestration thread
        errors.append(error)


def _phase_header(name: str, command: list[str]) -> bytes:
    return (
        f"\n[{_utc_now()}] phase={name} command={json.dumps(command, ensure_ascii=False)}\n"
    ).encode()


def _update_running_progress(
    layout: Layout,
    inputs: AuthenticatedInputs,
    *,
    baseline_tokens: int,
    started_monotonic: float,
    phase: str,
    child_pid: int,
    log_path: Path,
) -> None:
    progress = scan_progress(
        layout,
        inputs.prepared,
        inputs.config,
        verify_checksums=False,
    )
    elapsed = max(0.0, time.monotonic() - started_monotonic)
    added = max(0, int(progress["completed_tokens"]) - baseline_tokens)
    rate = added / elapsed if added > 0 and elapsed > 0 else None
    eta = (
        int(progress["remaining_tokens"]) / rate
        if rate is not None and rate > 0
        else int(progress["remaining_tokens"]) / HISTORICAL_WALL_TOKENS_PER_SECOND
    )
    status = _read_status(layout)
    status.update(
        {
            "status": "stop_requested" if layout.stop_file.is_file() else "running",
            "phase": phase,
            "child_pid": child_pid,
            "log": str(log_path.resolve()),
            "progress": {
                **progress,
                "attempt_baseline_tokens": baseline_tokens,
                "attempt_added_tokens": added,
                "attempt_elapsed_seconds": elapsed,
                "attempt_wall_tokens_per_second": rate,
                "eta_seconds": eta,
            },
        }
    )
    _write_status(layout, status)


def _run_phase(
    layout: Layout,
    inputs: AuthenticatedInputs,
    *,
    name: str,
    command: list[str],
    poll_seconds: float,
    baseline_tokens: int,
) -> PhaseResult:
    logs = layout.state_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    state = _read_status(layout)
    attempt = int(state.get("attempt", 1))
    history_count = len(state.get("history", []))
    log_path = logs / f"attempt-{attempt:03d}-{history_count:03d}-{name}.log"
    console_path = layout.state_root / "console.log"
    environment = os.environ.copy()
    source_root = str(layout.root / "src")
    environment["PYTHONPATH"] = (
        source_root
        if not environment.get("PYTHONPATH")
        else source_root + os.pathsep + environment["PYTHONPATH"]
    )
    header = _phase_header(name, command)
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    with (
        log_path.open("ab", buffering=0) as phase_log,
        console_path.open("ab", buffering=0) as console,
    ):
        phase_log.write(header)
        console.write(header)
        process = subprocess.Popen(
            command,
            cwd=layout.root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        assert process.stdout is not None
        pump_errors: list[BaseException] = []
        terminal = getattr(sys.stderr, "buffer", None)
        destinations: tuple[BinaryIO, ...] = (phase_log, console)
        if terminal is not None:
            destinations = (*destinations, terminal)
        pump = threading.Thread(
            target=_tee_stream,
            args=(process.stdout, destinations, pump_errors),
            name=f"twen-kd-log-{name}",
            daemon=True,
        )
        pump.start()
        try:
            while process.poll() is None:
                if pump_errors:
                    raise KDOrchestrationError(
                        f"log writer failed for phase {name}: {pump_errors[0]}"
                    )
                _update_running_progress(
                    layout,
                    inputs,
                    baseline_tokens=baseline_tokens,
                    started_monotonic=started_monotonic,
                    phase=name,
                    child_pid=process.pid,
                    log_path=log_path,
                )
                time.sleep(poll_seconds)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30.0)
            pump.join(timeout=30.0)
            raise
        exit_code = process.wait()
        pump.join(timeout=30.0)
        if pump.is_alive():
            raise KDOrchestrationError(f"log pump did not drain for phase {name}")
        if pump_errors:
            raise KDOrchestrationError(f"log writer failed for phase {name}: {pump_errors[0]}")
        ended_at = _utc_now()
        elapsed = max(0.0, time.monotonic() - started_monotonic)
        footer = f"\n[{ended_at}] phase={name} exit_code={exit_code}\n".encode()
        phase_log.write(footer)
        console.write(footer)
        os.fsync(phase_log.fileno())
        os.fsync(console.fileno())
    return PhaseResult(
        name=name,
        command=tuple(command),
        log_path=log_path.resolve(),
        exit_code=exit_code,
        started_at=started_at,
        ended_at=ended_at,
        elapsed_seconds=elapsed,
    )


def _phase_dict(result: PhaseResult, *, status: str) -> dict[str, Any]:
    return {
        "name": result.name,
        "status": status,
        "command": list(result.command),
        "log": str(result.log_path),
        "log_sha256": sha256_file(result.log_path),
        "exit_code": result.exit_code,
        "started_at": result.started_at,
        "ended_at": result.ended_at,
        "elapsed_seconds": result.elapsed_seconds,
    }


def _record_internal_failure(layout: Layout, *, phase: str, error: BaseException) -> None:
    status = _read_status(layout)
    failures = list(status.get("failures", []))
    failures.append(
        {
            "phase": phase,
            "failed_at": _utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "log": status.get("log"),
        }
    )
    status.update(
        {
            "status": "failed",
            "phase": None,
            "child_pid": None,
            "error": f"{phase} orchestration failed: {error}",
            "failures": failures,
        }
    )
    _write_status(layout, status)


def _verify_final_output(
    layout: Layout,
    inputs: AuthenticatedInputs,
) -> tuple[TeacherKDCorpusManifest, dict[str, Any]]:
    manifest_path = layout.output_root / "manifest.json"
    # index-kd has just performed a full checksum scan.  Avoid a second 332+GB
    # scan here; authenticate the resulting lock, coverage and identities.
    corpus = validate_kd_corpus_manifest(
        manifest_path,
        expected_temperature=EXPECTED_TEMPERATURE,
        verify_shards=False,
    )
    validate_kd_corpus_coverage(corpus, inputs.prepared)
    teacher = inputs.config.sources.teacher
    tokenizer = inputs.config.sources.tokenizer
    if (
        corpus.top_k != KD_TOP_K
        or corpus.token_count != inputs.prepared.token_count
        or corpus.sequence_count != inputs.prepared.sequence_count
        or corpus.teacher_model_id != teacher.model_id
        or corpus.teacher_revision != teacher.revision
        or corpus.teacher_model_sha256 != teacher.manifest_sha256
        or corpus.tokenizer_sha256 != tokenizer.manifest_sha256
        or corpus.generator_source_sha256 != KD_GENERATOR_SOURCE_SHA256
    ):
        raise KDOrchestrationError("indexed KD corpus does not match the authenticated contract")
    return corpus, _identity(manifest_path)


def _write_completion(
    layout: Layout,
    inputs: AuthenticatedInputs,
    *,
    kd_manifest_identity: dict[str, Any],
    plan: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    final_progress = scan_progress(
        layout,
        inputs.prepared,
        inputs.config,
        verify_checksums=False,
    )
    if final_progress["completed_shards"] != final_progress["total_shards"]:
        raise KDOrchestrationError("cannot certify an incomplete KD shard set")
    status = _read_status(layout)
    status.update(
        {
            "status": "complete",
            "phase": None,
            "child_pid": None,
            "completed_at": _utc_now(),
            "prepared_manifest": inputs.prepared_identity,
            "audit_attestation": inputs.audit_identity,
            "kd_manifest": kd_manifest_identity,
            "progress": {**final_progress, "eta_seconds": 0.0},
            "history": history,
            "recommendation": plan["recommendation"],
        }
    )
    _write_status(layout, status)
    status_identity = _identity(_status_path(layout))
    orchestration_manifest = layout.state_root / "MANIFEST.json"
    _atomic_json(
        orchestration_manifest,
        {
            "schema_version": SCHEMA_VERSION,
            "kind": MANIFEST_KIND,
            "prepared_manifest": inputs.prepared_identity,
            "audit_attestation": inputs.audit_identity,
            "pipeline_complete": inputs.pipeline_complete_identity,
            "pipeline_status": inputs.pipeline_status_identity,
            "base_config": inputs.config_identity,
            "teacher": inputs.teacher_source,
            "performance_evidence": inputs.benchmark,
            "full_run_evidence": inputs.full_run_benchmark,
            "kd_generator_source_sha256": KD_GENERATOR_SOURCE_SHA256,
            "kd_manifest": kd_manifest_identity,
            "status": status_identity,
            "console_log": _identity(layout.state_root / "console.log"),
            "history": history,
            "storage_plan": plan["storage"],
            "training_started": False,
            "optimizer_created": False,
        },
    )
    manifest_identity = _identity(orchestration_manifest)
    complete = layout.state_root / "COMPLETE"
    _atomic_json(
        complete,
        {
            "schema_version": SCHEMA_VERSION,
            "kind": COMPLETE_KIND,
            "manifest": manifest_identity,
            "kd_manifest": kd_manifest_identity,
            "prepared_manifest": inputs.prepared_identity,
            "completed_at": _utc_now(),
            "training_started": False,
            "optimizer_created": False,
        },
    )
    return {
        "ok": True,
        "status": status,
        "manifest": manifest_identity,
        "complete": _identity(complete),
        "kd_manifest": kd_manifest_identity,
        "training_started": False,
        "optimizer_created": False,
    }


def verify_orchestration_complete(layout: Layout) -> dict[str, Any] | None:
    complete_path = layout.state_root / "COMPLETE"
    if not complete_path.is_file():
        return None
    complete = _load_object(complete_path, label="KD orchestration COMPLETE")
    if (
        complete.get("schema_version") != SCHEMA_VERSION
        or complete.get("kind") != COMPLETE_KIND
        or complete.get("training_started") is not False
        or complete.get("optimizer_created") is not False
    ):
        raise KDOrchestrationError("KD orchestration COMPLETE contract is invalid")
    manifest_path = _path_from_identity(
        layout.root, complete.get("manifest"), label="KD orchestration manifest"
    )
    expected = layout.state_root / "MANIFEST.json"
    if manifest_path != expected.resolve():
        raise KDOrchestrationError("KD orchestration manifest path mismatch")
    manifest_identity = _identity(manifest_path)
    _same_identity(complete.get("manifest"), manifest_identity, label="orchestration manifest")
    manifest = _load_object(manifest_path, label="KD orchestration manifest")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != MANIFEST_KIND
        or manifest.get("training_started") is not False
        or manifest.get("optimizer_created") is not False
    ):
        raise KDOrchestrationError("KD orchestration manifest contract is invalid")
    prepared_path = _path_from_identity(
        layout.root,
        complete.get("prepared_manifest"),
        label="completed prepared manifest",
    )
    if prepared_path != layout.prepared_manifest.resolve():
        raise KDOrchestrationError("completed prepared manifest path mismatch")
    prepared_identity = _identity(prepared_path)
    _same_identity(
        complete.get("prepared_manifest"),
        prepared_identity,
        label="completed prepared manifest",
    )
    _same_identity(
        manifest.get("prepared_manifest"),
        prepared_identity,
        label="manifest prepared input",
    )
    status_path = _path_from_identity(
        layout.root, manifest.get("status"), label="completed orchestration status"
    )
    if status_path != _status_path(layout).resolve():
        raise KDOrchestrationError("completed orchestration status path mismatch")
    status_identity = _identity(status_path)
    _same_identity(
        manifest.get("status"),
        status_identity,
        label="completed orchestration status",
    )
    history = manifest.get("history")
    if not isinstance(history, list):
        raise KDOrchestrationError("completed orchestration history is missing")
    for index, phase in enumerate(history):
        if not isinstance(phase, Mapping) or not isinstance(phase.get("log"), str):
            raise KDOrchestrationError(f"completed phase {index} has no log identity")
        log_path = Path(str(phase["log"])).resolve()
        if not log_path.is_relative_to(layout.state_root.resolve()) or not log_path.is_file():
            raise KDOrchestrationError(f"completed phase {index} log is missing or unsafe")
        if sha256_file(log_path) != phase.get("log_sha256"):
            raise KDOrchestrationError(f"completed phase {index} log SHA256 mismatch")
    console_path = _path_from_identity(
        layout.root, manifest.get("console_log"), label="completed console log"
    )
    if console_path != (layout.state_root / "console.log").resolve():
        raise KDOrchestrationError("completed console log path mismatch")
    console_identity = _identity(console_path)
    _same_identity(
        manifest.get("console_log"),
        console_identity,
        label="completed console log",
    )
    kd_path = _path_from_identity(layout.root, complete.get("kd_manifest"), label="KD manifest")
    kd_identity = _identity(kd_path)
    _same_identity(complete.get("kd_manifest"), kd_identity, label="KD manifest")
    _same_identity(manifest.get("kd_manifest"), kd_identity, label="manifest KD")
    return {
        "complete": _identity(complete_path),
        "manifest": manifest_identity,
        "kd_manifest": kd_identity,
    }


def run_orchestration(
    layout: Layout,
    inputs: AuthenticatedInputs,
    *,
    python: Path,
    batch_size: int,
    logits_chunk_tokens: int,
    minimum_free_after_gib: float,
    poll_seconds: float,
    phase_runner: Callable[..., PhaseResult] = _run_phase,
) -> dict[str, Any]:
    existing_complete = verify_orchestration_complete(layout)
    if existing_complete is not None:
        corpus = validate_kd_corpus_manifest(
            layout.output_root / "manifest.json",
            expected_temperature=EXPECTED_TEMPERATURE,
            verify_shards=True,
        )
        validate_kd_corpus_coverage(corpus, inputs.prepared)
        return {
            "ok": True,
            "already_complete": True,
            **existing_complete,
            "training_started": False,
            "optimizer_created": False,
        }
    if layout.stop_file.is_file():
        raise KDOrchestrationStopped(
            f"persistent STOP exists at {layout.stop_file}; remove it before resuming"
        )
    plan = _public_plan(
        layout,
        inputs,
        python=python,
        batch_size=batch_size,
        logits_chunk_tokens=logits_chunk_tokens,
        minimum_free_after_gib=minimum_free_after_gib,
    )
    if not plan["ready"]:
        disk = plan["disk"]
        raise KDOrchestrationError(
            "insufficient disk space: need "
            f"{disk['required_free_now_bytes']:,} bytes free, have {disk['free_bytes']:,}"
        )
    # This cheap pass establishes progress without adding a redundant 332GB
    # read. generate-kd itself performs the full checksum/identity scan of all
    # resumed shards before it loads the teacher, and index-kd repeats a full
    # independent scan before orchestration COMPLETE is allowed.
    initial_progress = scan_progress(
        layout,
        inputs.prepared,
        inputs.config,
        verify_checksums=False,
    )
    previous = _read_status(layout)
    history = list(previous.get("history", []))
    attempt = int(previous.get("attempt", 0)) + 1
    active_status = {
        **previous,
        "attempt": attempt,
        "status": "running",
        "phase": "generate-kd",
        "started_at": _utc_now(),
        "prepared_manifest": inputs.prepared_identity,
        "audit_attestation": inputs.audit_identity,
        "progress": initial_progress,
        "history": history,
        "recommendation": plan["recommendation"],
    }
    # Terminal diagnostics belong to the attempt that produced them.  Keeping
    # them at the top level during a successful retry makes the live dashboard
    # report a false failure and would leak that stale error into COMPLETE.
    for field in ("error", "stopped_at", "stop_file"):
        active_status.pop(field, None)
    _write_status(
        layout,
        active_status,
    )

    try:
        generate = phase_runner(
            layout,
            inputs,
            name="generate-kd",
            command=plan["generate_command"],
            poll_seconds=poll_seconds,
            baseline_tokens=int(initial_progress["completed_tokens"]),
        )
    except BaseException as error:
        _record_internal_failure(layout, phase="generate-kd", error=error)
        raise
    if generate.exit_code == 75:
        phase = _phase_dict(generate, status="stopped")
        history.append(phase)
        status = _read_status(layout)
        status.update(
            {
                "status": "stopped",
                "phase": None,
                "child_pid": None,
                "history": history,
                "stopped_at": _utc_now(),
                "stop_file": str(layout.stop_file),
            }
        )
        _write_status(layout, status)
        raise KDOrchestrationStopped(
            f"KD STOP honored; remove {layout.stop_file} and rerun the identical command"
        )
    if generate.exit_code != 0:
        phase = _phase_dict(generate, status="failed")
        history.append(phase)
        status = _read_status(layout)
        status.update(
            {
                "status": "failed",
                "phase": None,
                "child_pid": None,
                "history": history,
                "error": f"generate-kd exited {generate.exit_code}; inspect {generate.log_path}",
            }
        )
        _write_status(layout, status)
        raise KDOrchestrationError(str(status["error"]))
    history.append(_phase_dict(generate, status="complete"))

    try:
        index = phase_runner(
            layout,
            inputs,
            name="index-kd",
            command=plan["index_command"],
            poll_seconds=poll_seconds,
            baseline_tokens=int(inputs.prepared.token_count),
        )
    except BaseException as error:
        _record_internal_failure(layout, phase="index-kd", error=error)
        raise
    if index.exit_code != 0:
        phase = _phase_dict(index, status="failed")
        history.append(phase)
        status = _read_status(layout)
        status.update(
            {
                "status": "failed",
                "phase": None,
                "child_pid": None,
                "history": history,
                "error": f"index-kd exited {index.exit_code}; inspect {index.log_path}",
            }
        )
        _write_status(layout, status)
        raise KDOrchestrationError(str(status["error"]))
    history.append(_phase_dict(index, status="complete"))
    _, kd_identity = _verify_final_output(layout, inputs)
    return _write_completion(
        layout,
        inputs,
        kd_manifest_identity=kd_identity,
        plan=plan,
        history=history,
    )


def status_snapshot(layout: Layout) -> dict[str, Any]:
    state = _read_status(layout)
    prepared = _read_prepared_fast(layout.prepared_manifest)
    config = None
    with suppress(OSError, ValueError, KeyError, TypeError):
        config = load_train_config(layout.base_config)
    progress = None
    storage = None
    planning = None
    if prepared is not None:
        storage = estimate_kd_storage(prepared)
        if config is not None:
            with suppress(KDOrchestrationError, OSError, ValueError, KeyError, TypeError):
                progress = scan_progress(
                    layout,
                    prepared,
                    config,
                    verify_checksums=False,
                )
                planning = _planning_time(
                    int(prepared.token_count),
                    int(progress["remaining_tokens"]),
                    total_padded_positions=int(storage["padded_positions"]),
                    remaining_padded_positions=(
                        int(progress["total_sequences"] - progress["completed_sequences"])
                        * int(prepared.sequence_length)
                    ),
                )
    completion = verify_orchestration_complete(layout)
    return {
        "ok": True,
        "observed_at": _utc_now(),
        "state": state,
        "prepared_manifest_exists": layout.prepared_manifest.is_file(),
        "pipeline_complete_exists": layout.pipeline_complete.is_file(),
        "stop_requested": layout.stop_file.is_file(),
        "progress": progress,
        "storage": storage,
        "time_estimate": planning,
        "completion": completion,
        "safe_stop": ["touch", str(layout.stop_file)],
        "training_started": False,
        "optimizer_created": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=("status", "preflight", "plan", "run"),
        default="status",
        help="only run starts the optimizer-free teacher-KD child",
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--prepared-manifest", default=None)
    parser.add_argument("--pipeline-complete", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--state-root", default=None)
    parser.add_argument("--batch-size", type=int, choices=tuple(_BENCHMARK_FILENAMES), default=2)
    parser.add_argument("--logits-chunk-tokens", type=int, default=64)
    parser.add_argument("--minimum-free-after-gib", type=float, default=64.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument(
        "--acknowledge-gpu-kd",
        action="store_true",
        help="required with --action run; never authorizes training",
    )
    return parser


def _absolute_python_without_resolving(path: Path) -> Path:
    """Make the child interpreter absolute while preserving venv symlinks.

    Resolving ``.venv/bin/python`` follows the symlink to the base uv-managed
    interpreter and drops the virtual environment's site-packages.  The child
    must therefore receive the absolute venv entry point, not its realpath.
    """

    expanded = path.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    absolute = Path(os.path.abspath(absolute))
    if not absolute.is_file() or not os.access(absolute, os.X_OK):
        raise KDOrchestrationError(f"Python interpreter is not executable: {absolute}")
    return absolute


def _validate_args(args: argparse.Namespace) -> None:
    if args.logits_chunk_tokens <= 0:
        raise KDOrchestrationError("logits chunk size must be positive")
    if not math.isfinite(args.minimum_free_after_gib) or args.minimum_free_after_gib < 0:
        raise KDOrchestrationError("minimum free-after GiB must be finite and non-negative")
    if not math.isfinite(args.poll_seconds) or not 0.1 <= args.poll_seconds <= 60.0:
        raise KDOrchestrationError("poll seconds must be within 0.1..60")
    if args.acknowledge_gpu_kd and args.action != "run":
        raise KDOrchestrationError("--acknowledge-gpu-kd is only valid with --action run")
    if args.action == "run" and not args.acknowledge_gpu_kd:
        raise KDOrchestrationError("--action run requires --acknowledge-gpu-kd")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_args(args)
        layout = Layout.from_args(args)
        child_python = _absolute_python_without_resolving(args.python)
        if args.action == "status":
            result = status_snapshot(layout)
        elif args.action in {"preflight", "plan"}:
            inputs = authenticate_inputs(
                layout,
                batch_size=args.batch_size,
                logits_chunk_tokens=args.logits_chunk_tokens,
                verify_teacher_files=args.action == "preflight",
            )
            plan = _public_plan(
                layout,
                inputs,
                python=child_python,
                batch_size=args.batch_size,
                logits_chunk_tokens=args.logits_chunk_tokens,
                minimum_free_after_gib=args.minimum_free_after_gib,
            )
            result = {"action": args.action, **plan}
        else:
            layout.state_root.mkdir(parents=True, exist_ok=True)
            with _exclusive_lock(layout.state_root / "orchestration.lock"):
                # A pre-existing STOP is deliberately cheaper than hashing the
                # prepared corpus and 9B model source.
                if layout.stop_file.is_file():
                    raise KDOrchestrationStopped(
                        f"persistent STOP exists at {layout.stop_file}; remove it before resuming"
                    )
                inputs = authenticate_inputs(
                    layout,
                    batch_size=args.batch_size,
                    logits_chunk_tokens=args.logits_chunk_tokens,
                    # generate-kd authenticates every 9B model artifact before
                    # importing torch or selecting a CUDA device. Avoid hashing
                    # the same ~20GB source twice on the production path.
                    verify_teacher_files=False,
                )
                result = run_orchestration(
                    layout,
                    inputs,
                    python=child_python,
                    batch_size=args.batch_size,
                    logits_chunk_tokens=args.logits_chunk_tokens,
                    minimum_free_after_gib=args.minimum_free_after_gib,
                    poll_seconds=args.poll_seconds,
                )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except KDOrchestrationStopped as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "stopped": True,
                    "error": str(error),
                    "training_started": False,
                    "optimizer_created": False,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 75
    except (KDOrchestrationError, OSError, ValueError, KeyError, TypeError) as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(error),
                    "training_started": False,
                    "optimizer_created": False,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LOGITS_CHUNK_TOKENS",
    "KD_BYTES_PER_PADDED_POSITION",
    "AuthenticatedInputs",
    "KDOrchestrationError",
    "KDOrchestrationStopped",
    "Layout",
    "PhaseResult",
    "authenticate_inputs",
    "disk_plan",
    "estimate_kd_storage",
    "generate_command",
    "index_command",
    "main",
    "run_orchestration",
    "scan_progress",
    "status_snapshot",
    "verify_orchestration_complete",
]
