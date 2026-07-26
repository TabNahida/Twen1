"""Fail-closed validation performed before a training process touches CUDA."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import socket
import struct
import time
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from .config import TrainConfig
from .io.download import ArtifactSpec, verify_artifact
from .io.offline import enforce_offline_environment
from .source_identity import twen_source_tree_sha256
from .utils import sha256_file


class TrainingPreflightError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BatchGeometry:
    world_size: int
    micro_batch_tokens_per_rank: int
    gradient_accumulation_steps: int
    global_batch_tokens: int


@dataclass(frozen=True, slots=True)
class DataGovernanceStatus:
    """Operator-visible governance state for the exact prepared dataset."""

    lineage_kind: str
    research_only: bool
    ready_for_training: bool
    pending_audits: tuple[str, ...]
    warning: str | None


_MISSING_DATA_GOVERNANCE = DataGovernanceStatus(
    lineage_kind="missing",
    research_only=True,
    ready_for_training=False,
    pending_audits=("prepared_lineage_missing",),
    warning=(
        "RESEARCH-ONLY DATA: ready_for_training=false; pending audits: "
        "prepared_lineage_missing. Training is technically allowed, but this is an "
        "explicit research risk and results must not be presented as production-ready."
    ),
)

_PHASE_DISJOINTNESS_NEAR_DUPLICATE_THRESHOLD = 0.8


@dataclass(frozen=True, slots=True)
class PreflightReport:
    config_fingerprint: str
    data_fingerprint: str
    source_tree_sha256: str
    batch: BatchGeometry
    checked_paths: tuple[str, ...]
    calibration_fingerprints: tuple[tuple[str, str], ...] = ()
    data_governance: DataGovernanceStatus = _MISSING_DATA_GOVERNANCE
    teacher_cpu_shadow_bytes: int = 0
    teacher_gpu_stage_bytes: int = 0
    activation_checkpoint_layer_count: int | None = None
    hidden_alignment_activation_checkpoint_layer_count: int | None = None
    activation_checkpoint_layer_indices: tuple[int, ...] = ()
    hidden_alignment_activation_checkpoint_layer_indices: tuple[int, ...] = ()
    dense_transfer_execution: str = "expanded"
    dense_transfer_checkpoint_layer_count: int | None = None
    hidden_alignment_dense_transfer_checkpoint_layer_count: int | None = None
    dense_transfer_token_checkpoint_layer_indices: tuple[int, ...] = ()
    hidden_alignment_dense_transfer_token_checkpoint_layer_indices: tuple[int, ...] = ()
    quality_cooldown_enabled: bool = False
    quality_cooldown_start_tokens: int | None = None
    quality_cooldown_dataset_fingerprint: str | None = None
    quality_cooldown_sequence_count: int = 0
    quality_cooldown_token_count: int = 0
    quality_cooldown_selected_shard_ids: tuple[str, ...] = ()
    quality_cooldown_source_mix_token_counts: tuple[tuple[str, int], ...] = ()
    quality_cooldown_source_mix_enabled: bool = False
    quality_cooldown_source_mix_algorithm: str | None = None
    quality_cooldown_source_map_sha256: str | None = None
    quality_cooldown_source_mix_dataset_fingerprint: str | None = None
    quality_cooldown_source_mix_basis_points: tuple[tuple[str, int], ...] = ()
    quality_cooldown_source_mix_seed: int | None = None
    quality_cooldown_source_map_payload_json: str | None = None
    source_mix_enabled: bool = False
    source_mix_algorithm: str | None = None
    source_map_sha256: str | None = None
    source_mix_dataset_fingerprint: str | None = None
    source_mix_basis_points: tuple[tuple[str, int], ...] = ()
    source_mix_lineage_basis_points: tuple[tuple[str, int], ...] = ()
    source_mix_effective_basis_points: tuple[tuple[str, int], ...] = ()
    source_mix_weight_override: bool = False
    source_mix_seed: int | None = None
    source_map_payload_json: str | None = None


@dataclass(frozen=True, slots=True)
class _PreparedTextCooldownSummary:
    cooldown_dataset_fingerprint: str
    selected_shard_ids: tuple[str, ...]
    source_mix_token_counts: tuple[tuple[str, int], ...]
    sequence_count: int
    token_count: int


def _prepared_data_governance(prepared_corpus: object) -> DataGovernanceStatus:
    """Make research-only lineage prominent without turning it into an approval."""

    lineage = getattr(prepared_corpus, "lineage", None)
    if lineage is None:
        return _MISSING_DATA_GOVERNANCE
    if not isinstance(lineage, Mapping):
        raise TrainingPreflightError("prepared data lineage must be an object")
    kind = str(lineage.get("kind", "unknown"))
    research_only = lineage.get("research_only") is True
    ready_for_training = lineage.get("ready_for_training") is True
    raw_pending = lineage.get("pending_audits", ())
    if not isinstance(raw_pending, (list, tuple)) or not all(
        isinstance(item, str) and item for item in raw_pending
    ):
        raise TrainingPreflightError("prepared pending_audits must be a list of names")
    pending = tuple(str(item) for item in raw_pending)
    if ready_for_training and pending:
        raise TrainingPreflightError(
            "prepared data claims ready_for_training=true while audits remain pending: "
            + ", ".join(pending)
        )
    if ready_for_training and research_only:
        raise TrainingPreflightError(
            "prepared data cannot be both ready_for_training and research_only"
        )
    if ready_for_training:
        warning = None
    else:
        pending_text = ", ".join(pending) if pending else "unspecified_governance_review"
        warning = (
            "RESEARCH-ONLY DATA: ready_for_training=false; pending audits: "
            f"{pending_text}. Training is technically allowed, but this is an explicit "
            "research risk and results must not be presented as production-ready."
        )
    return DataGovernanceStatus(
        lineage_kind=kind,
        research_only=research_only or not ready_for_training,
        ready_for_training=ready_for_training,
        pending_audits=pending,
        warning=warning,
    )


def _validate_no_reuse_capacity(
    prepared_corpus: object,
    *,
    label: str,
    phase_budget_tokens: int,
    global_batch_tokens: int,
    sequence_length: int,
    source_map: object | None = None,
    source_mix_basis_points: Mapping[str, int] | None = None,
    seed: int,
) -> None:
    """Prove the real valid-token cursor reaches the phase boundary without reuse."""

    import math

    guarded_tokens = phase_budget_tokens + global_batch_tokens
    guarded_samples = math.ceil(guarded_tokens / sequence_length)
    token_count = getattr(prepared_corpus, "token_count", None)
    sequence_count = getattr(prepared_corpus, "sequence_count", None)
    if (
        isinstance(token_count, bool)
        or not isinstance(token_count, int)
        or isinstance(sequence_count, bool)
        or not isinstance(sequence_count, int)
    ):
        raise TrainingPreflightError(f"{label} prepared corpus capacity metadata is invalid")
    if token_count < guarded_tokens or sequence_count < guarded_samples:
        raise TrainingPreflightError(
            f"{label} prepared corpus would wrap in the complete tail optimizer "
            "batch while data.allow_corpus_reuse=false: "
            f"requires_tokens={guarded_tokens}, available_tokens={token_count}, "
            f"requires_samples={guarded_samples}, available_samples={sequence_count}"
        )
    if source_map is None:
        return

    from .data.cursor import (
        AuthenticatedSourceMap,
        DeterministicSourceMixCursor,
    )

    if not isinstance(source_map, AuthenticatedSourceMap):
        raise TrainingPreflightError(f"{label} source-map capacity identity is invalid")
    if not isinstance(source_mix_basis_points, Mapping):
        raise TrainingPreflightError(f"{label} source-mix capacity weights are missing")
    entries_by_id = {entry.shard_id: entry for entry in getattr(prepared_corpus, "shards", ())}
    available_samples: dict[str, int] = {}
    available_tokens: dict[str, int] = {}
    for source_id in source_map.source_ids:
        shards = source_map.shards_for_source(source_id)
        try:
            available_samples[source_id] = sum(
                entries_by_id[shard.shard_id].sequence_count for shard in shards
            )
            available_tokens[source_id] = sum(
                entries_by_id[shard.shard_id].token_count for shard in shards
            )
        except KeyError as error:
            raise TrainingPreflightError(
                f"{label} source-map shard is absent from prepared capacity inventory"
            ) from error
        required_source_tokens = math.ceil(
            guarded_tokens * int(source_mix_basis_points[source_id]) / 10_000
        )
        if available_tokens[source_id] < required_source_tokens:
            raise TrainingPreflightError(
                f"{label} source {source_id!r} would wrap while "
                "data.allow_corpus_reuse=false: "
                f"requires_tokens={required_source_tokens}, "
                f"available_tokens={available_tokens[source_id]}, "
                f"available_samples={available_samples[source_id]}"
            )

    if global_batch_tokens % sequence_length:
        raise TrainingPreflightError(
            f"{label} global batch tokens must be divisible by sequence length "
            "for exact no-reuse capacity replay"
        )
    global_batch_samples = global_batch_tokens // sequence_length
    if global_batch_samples <= 0:
        raise TrainingPreflightError(f"{label} global batch contains no prepared samples")

    # ``prepare_jsonl_corpus`` packs every shard densely: all sequences except
    # possibly the final one contain ``sequence_length`` valid tokens.  The
    # authenticated per-shard sequence/token counts therefore recover the
    # exact valid-token ledger without loading tensor payloads.
    final_valid_tokens: dict[str, int] = {}
    for shard in source_map.shards:
        entry = entries_by_id.get(shard.shard_id)
        if entry is None:
            raise TrainingPreflightError(
                f"{label} source-map shard is absent from prepared capacity inventory"
            )
        entry_sequence_count = getattr(entry, "sequence_count", None)
        entry_token_count = getattr(entry, "token_count", None)
        if (
            isinstance(entry_sequence_count, bool)
            or not isinstance(entry_sequence_count, int)
            or entry_sequence_count != shard.sequence_count
            or isinstance(entry_token_count, bool)
            or not isinstance(entry_token_count, int)
        ):
            raise TrainingPreflightError(
                f"{label} prepared shard {shard.shard_id!r} capacity identity is invalid"
            )
        final_tokens = entry_token_count - (entry_sequence_count - 1) * sequence_length
        if not 1 <= final_tokens <= sequence_length:
            raise TrainingPreflightError(
                f"{label} prepared shard {shard.shard_id!r} violates dense packing"
            )
        final_valid_tokens[shard.shard_id] = final_tokens

    try:
        cursor = DeterministicSourceMixCursor(
            source_map,
            source_mix_basis_points,
            seed=seed,
        )
        while cursor.committed_tokens < phase_budget_tokens:
            references = cursor.plan_global_batch(global_batch_samples)
            per_reference: list[int] = []
            valid_tokens_by_source: dict[str, int] = dict.fromkeys(
                source_map.source_ids,
                0,
            )
            for reference in references:
                if reference.epoch != 0:
                    raise TrainingPreflightError(
                        f"{label} source {reference.source_id!r} would wrap during "
                        "exact valid-token capacity replay"
                    )
                shard = entries_by_id[reference.shard_id]
                valid_tokens = (
                    final_valid_tokens[reference.shard_id]
                    if reference.shard_offset == shard.sequence_count - 1
                    else sequence_length
                )
                per_reference.append(valid_tokens)
                valid_tokens_by_source[reference.source_id] += valid_tokens
            plan_fingerprint = cursor.pending_plan_fingerprint
            if plan_fingerprint is None:
                raise TrainingPreflightError(
                    f"{label} exact capacity replay omitted its plan fingerprint"
                )
            cursor.commit(
                planned_references=references,
                plan_fingerprint=plan_fingerprint,
                valid_tokens_per_reference=per_reference,
                valid_tokens_by_source=valid_tokens_by_source,
                token_count=sum(per_reference),
            )
    except TrainingPreflightError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise TrainingPreflightError(
            f"{label} exact valid-token capacity replay failed: {error}"
        ) from error


def _require_file(path: str | Path, expected_sha256: str | None, label: str) -> Path:
    target = Path(path)
    if not target.is_file():
        raise TrainingPreflightError(f"{label} is missing: {target}")
    if expected_sha256 is not None:
        actual = sha256_file(target)
        if actual.lower() != expected_sha256.lower():
            raise TrainingPreflightError(
                f"{label} SHA256 mismatch: expected {expected_sha256}, got {actual}"
            )
    return target


def _check_source(name: str, source: object) -> tuple[Path, Path, dict[str, object]]:
    root = Path(source.local_path)
    if not root.is_dir():
        raise TrainingPreflightError(f"source {name} is missing: {root}")
    if any(root.rglob("*.incomplete")):
        raise TrainingPreflightError(f"source {name} contains an incomplete download: {root}")
    manifest = root / "download-manifest.json"
    _require_file(manifest, source.manifest_sha256, f"source {name} manifest")
    try:
        download_manifest = json.loads(manifest.read_text(encoding="utf-8"))
        raw_artifacts = download_manifest["artifacts"]
        if not isinstance(raw_artifacts, list) or not raw_artifacts:
            raise ValueError("artifacts must be a non-empty list")
        for raw in raw_artifacts:
            spec = ArtifactSpec(**raw)
            if spec.repository != source.model_id:
                raise ValueError(
                    f"artifact repository {spec.repository!r} differs from configured model_id"
                )
            if spec.revision != source.revision:
                raise ValueError(
                    f"artifact revision {spec.revision!r} differs from configured revision"
                )
            verify_artifact(root / spec.filename, spec)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise TrainingPreflightError(
            f"source {name} download manifest/artifacts failed verification: {exc}"
        ) from exc

    config_path = root / "config.json"
    _require_file(config_path, None, f"source {name} config")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingPreflightError(f"source {name} config is invalid JSON") from exc
    text_config = payload.get("text_config", payload)
    if not isinstance(text_config, dict) or text_config.get("model_type") != "qwen3_5_text":
        raise TrainingPreflightError(f"source {name} is not a dense Qwen3.5 text checkpoint")
    return root, manifest, text_config


def _require_same(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise TrainingPreflightError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def compute_batch_geometry(config: TrainConfig, world_size: int) -> BatchGeometry:
    if world_size < 1:
        raise TrainingPreflightError("WORLD_SIZE must be positive")
    micro_tokens = config.data.micro_batch_size * config.data.max_sequence_length
    denominator = micro_tokens * world_size
    if config.data.global_batch_tokens % denominator:
        raise TrainingPreflightError(
            "global_batch_tokens must be divisible by "
            "micro_batch_size * max_sequence_length * world_size; "
            f"got {config.data.global_batch_tokens} % {denominator}"
        )
    accumulation = config.data.global_batch_tokens // denominator
    if accumulation < 1:
        raise TrainingPreflightError("global batch is smaller than one distributed microbatch")
    return BatchGeometry(world_size, micro_tokens, accumulation, config.data.global_batch_tokens)


def validate_optimizer_world_size(config: TrainConfig, world_size: int) -> None:
    """Reject optimizer/distribution combinations with undefined semantics."""

    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise TrainingPreflightError("world_size must be a positive integer")
    if config.optimizer.adapter_optimizer == "muon" and world_size != 1:
        raise TrainingPreflightError(
            "optimizer.adapter_optimizer='muon' currently requires world_size=1; "
            "Newton-Schulz orthogonalization of sharded adapter matrices is not implemented"
        )


def _audit_architecture(config: TrainConfig, source_configs: dict[str, dict[str, object]]) -> None:
    from .modeling import audit_source_configs

    try:
        audit = audit_source_configs(source_configs["backbone"], source_configs["donor"])
    except (TypeError, ValueError) as exc:
        raise TrainingPreflightError(f"source architecture audit failed: {exc}") from exc
    architecture = config.architecture
    for label, actual, expected in (
        ("backbone hidden_size", audit.backbone.hidden_size, architecture.student_hidden_size),
        (
            "backbone intermediate_size",
            audit.backbone.intermediate_size,
            architecture.student_intermediate_size,
        ),
        ("backbone layers", audit.backbone.num_hidden_layers, architecture.student_layers),
        ("donor hidden_size", audit.donor.hidden_size, architecture.donor_hidden_size),
        (
            "donor intermediate_size",
            audit.donor.intermediate_size,
            architecture.donor_intermediate_size,
        ),
        ("donor layers", audit.donor.num_hidden_layers, architecture.donor_layers),
    ):
        _require_same(label, actual, expected)
    for field in ("hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size"):
        _require_same(
            f"teacher/donor {field}",
            source_configs["teacher"].get(field),
            source_configs["donor"].get(field),
        )
    _require_same(
        "backbone/teacher vocab_size",
        source_configs["backbone"].get("vocab_size"),
        source_configs["teacher"].get("vocab_size"),
    )


def _load_json_file(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingPreflightError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TrainingPreflightError(f"{label} must be a JSON object: {path}")
    return value


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _validate_phase_disjointness_attestation(
    path: Path,
    *,
    primary_manifest: Path,
    primary_prepared: object,
    primary_source_map: object,
    cooldown_manifest: Path,
    cooldown_prepared: object,
    cooldown_source_map: object,
) -> None:
    """Authenticate the no-reuse prepared-text phase separation proof."""

    value = _load_json_file(path, "phase disjointness attestation")
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "twen_v4_phase_disjointness_attestation"
    ):
        raise TrainingPreflightError(
            "phase disjointness attestation has an unsupported schema/kind"
        )
    fingerprint = value.get("attestation_fingerprint")
    fingerprint_payload = dict(value)
    fingerprint_payload.pop("attestation_fingerprint", None)
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or _canonical_json_sha256(fingerprint_payload) != fingerprint.lower()
    ):
        raise TrainingPreflightError("phase disjointness attestation fingerprint mismatch")
    scanner_sha = value.get("scanner_source_sha256")
    if (
        not isinstance(scanner_sha, str)
        or len(scanner_sha) != 64
        or any(character not in "0123456789abcdef" for character in scanner_sha.lower())
    ):
        raise TrainingPreflightError("phase disjointness attestation has no scanner source SHA256")
    scanner_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "attest_v4_phase_disjointness.py"
    )
    if not scanner_path.is_file() or sha256_file(scanner_path) != scanner_sha.lower():
        raise TrainingPreflightError(
            "phase disjointness scanner source changed; rerun the phase audit"
        )
    scanner_source_tree_sha = value.get("scanner_source_tree_sha256")
    if (
        not isinstance(scanner_source_tree_sha, str)
        or len(scanner_source_tree_sha) != 64
        or any(character not in "0123456789abcdef" for character in scanner_source_tree_sha.lower())
        or scanner_source_tree_sha.lower() != twen_source_tree_sha256()
    ):
        raise TrainingPreflightError(
            "phase disjointness scanner source tree changed; rerun the phase audit"
        )
    gates = value.get("gates")
    required_gates = {
        "stable_id_exact": ("source-scoped-authenticated-stable-id-intersection-v1"),
        "normalized_text_exact": ("unicode-nfkc-whitespace-sha256-intersection-v1"),
        "near_duplicate": ("lexical-5gram-one-permutation-minhash-lsh-v1"),
    }
    if not isinstance(gates, Mapping) or set(gates) != set(required_gates):
        raise TrainingPreflightError(
            "phase disjointness attestation does not contain the exact required gates"
        )
    metrics = value.get("metrics")
    metric_names = {
        "stable_id_exact": "stable_id_exact_matches",
        "normalized_text_exact": "normalized_text_exact_matches",
        "near_duplicate": "near_duplicate_matches",
    }
    if (
        not isinstance(metrics, Mapping)
        or value.get("scope") != "authenticated_train_inventories_only"
        or value.get("stores_raw_text") is not False
    ):
        raise TrainingPreflightError(
            "phase disjointness attestation scope/metrics contract is invalid"
        )
    for gate_name, algorithm in sorted(required_gates.items()):
        gate = gates[gate_name]
        metric_count = metrics.get(metric_names[gate_name])
        if (
            not isinstance(gate, Mapping)
            or gate.get("passed") is not True
            or gate.get("matches") != 0
            or gate.get("algorithm") != algorithm
            or isinstance(metric_count, bool)
            or not isinstance(metric_count, int)
            or metric_count != gate.get("matches")
            or (
                gate_name == "near_duplicate"
                and gate.get("estimated_jaccard_threshold")
                != _PHASE_DISJOINTNESS_NEAR_DUPLICATE_THRESHOLD
            )
        ):
            raise TrainingPreflightError(
                f"phase disjointness gate {gate_name!r} did not pass with zero matches"
            )
    if value.get("passed") is not True:
        raise TrainingPreflightError("phase disjointness attestation did not pass")

    for phase, manifest, prepared, source_map in (
        (
            "primary",
            primary_manifest,
            primary_prepared,
            primary_source_map,
        ),
        (
            "cooldown",
            cooldown_manifest,
            cooldown_prepared,
            cooldown_source_map,
        ),
    ):
        identity = value.get(phase)
        prepared_identity = identity.get("prepared") if isinstance(identity, Mapping) else None
        if not isinstance(prepared_identity, Mapping):
            raise TrainingPreflightError(
                f"phase disjointness attestation has no {phase} prepared identity"
            )
        expected = {
            "manifest_path": str(manifest.resolve()),
            "manifest_sha256": sha256_file(manifest),
            "dataset_fingerprint": getattr(
                prepared,
                "dataset_fingerprint",
                None,
            ),
            "source_map_sha256": getattr(source_map, "fingerprint", None),
        }
        for field, expected_value in expected.items():
            actual = prepared_identity.get(field)
            if field == "manifest_path":
                try:
                    actual = str(Path(str(actual)).resolve())
                except (OSError, ValueError) as error:
                    raise TrainingPreflightError(
                        f"phase disjointness {phase} manifest path is invalid"
                    ) from error
            if actual != expected_value:
                raise TrainingPreflightError(
                    f"phase disjointness {phase} {field} mismatch: "
                    f"expected {expected_value!r}, got {actual!r}"
                )

    complete_path = path.parent / "COMPLETE"
    complete = _load_json_file(
        complete_path,
        "phase disjointness COMPLETE marker",
    )
    if (
        complete.get("schema_version") != 1
        or complete.get("kind") != "twen_v4_phase_disjointness_complete"
        or complete.get("attestation") != path.name
        or complete.get("attestation_sha256") != sha256_file(path)
        or complete.get("attestation_fingerprint") != fingerprint
        or complete.get("passed") is not True
    ):
        raise TrainingPreflightError(
            "phase disjointness COMPLETE marker does not bind the attestation"
        )


def _validate_calibration_contract(
    config: TrainConfig,
    layer_map_path: Path,
    channel_map_path: Path,
    adapter_path: Path | None,
) -> list[tuple[str, Path]]:
    layer_map = _load_json_file(layer_map_path, "layer map")
    if layer_map.get("schema_version") != 2 or layer_map.get("kind") != "monotonic_same_type_cka":
        raise TrainingPreflightError("layer map has an unsupported schema/kind")
    raw_mapping = layer_map.get("student_to_donor")
    if not isinstance(raw_mapping, list) or len(raw_mapping) != config.architecture.student_layers:
        raise TrainingPreflightError("layer map does not cover all student layers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in raw_mapping):
        raise TrainingPreflightError("layer map donor indices must be integers")
    mapping = tuple(raw_mapping)
    if any(not 0 <= item < config.architecture.donor_layers for item in mapping):
        raise TrainingPreflightError("layer map contains an out-of-range donor layer")
    if any(right <= left for left, right in pairwise(mapping)):
        raise TrainingPreflightError("layer map donor indices are not strictly increasing")
    from .modeling import audit_source_configs

    source_audit = audit_source_configs(
        config.sources.backbone.local_path,
        config.sources.donor.local_path,
    )
    pairs = layer_map.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != config.architecture.student_layers:
        raise TrainingPreflightError("layer map pair records do not cover all student layers")
    for student_layer, donor_layer in enumerate(mapping):
        pair = pairs[student_layer]
        if (
            not isinstance(pair, dict)
            or pair.get("student_layer") != student_layer
            or pair.get("donor_layer") != donor_layer
        ):
            raise TrainingPreflightError(f"layer map pair {student_layer} is inconsistent")
        expected_type = source_audit.student_layer_types[student_layer]
        if source_audit.donor_layer_types[donor_layer] != expected_type:
            raise TrainingPreflightError(
                f"layer map pair {student_layer}->{donor_layer} changes layer type"
            )
        if pair.get("layer_type") != expected_type:
            raise TrainingPreflightError(
                f"layer map pair {student_layer} records the wrong layer type"
            )
    lineage = layer_map.get("lineage")
    if not isinstance(lineage, dict):
        raise TrainingPreflightError("layer map has no calibration lineage")
    for role, source_name in (("student_model", "backbone"), ("donor_model", "donor")):
        model = lineage.get(role)
        if not isinstance(model, dict):
            raise TrainingPreflightError(f"layer map lineage is missing {role}")
        _require_same(
            f"layer map {role} manifest",
            model.get("manifest_sha256"),
            getattr(config.sources, source_name).manifest_sha256,
        )

    channel_map = _load_json_file(channel_map_path, "channel map")
    if channel_map.get("schema_version") != 2 or channel_map.get("kind") != "channel_partition_map":
        raise TrainingPreflightError("channel map has an unsupported schema/kind")
    raw_layers = channel_map.get("layers")
    expected_layer_keys = {str(layer) for layer in range(config.architecture.student_layers)}
    if not isinstance(raw_layers, dict) or set(raw_layers) != expected_layer_keys:
        raise TrainingPreflightError("channel map does not cover all student layers")
    channel_lineage = channel_map.get("lineage")
    if not isinstance(channel_lineage, dict):
        raise TrainingPreflightError("channel map has no lineage")
    _require_same(
        "channel map parent layer-map SHA256",
        channel_lineage.get("layer_map_sha256"),
        sha256_file(layer_map_path),
    )
    donor_source = channel_lineage.get("donor_source")
    if not isinstance(donor_source, dict):
        raise TrainingPreflightError("channel map lineage has no donor source")
    _require_same(
        "channel map donor manifest",
        donor_source.get("manifest_sha256"),
        config.sources.donor.manifest_sha256,
    )
    for layer, donor_layer in enumerate(mapping):
        item = raw_layers.get(str(layer))
        if not isinstance(item, dict) or item.get("donor_layer") != donor_layer:
            raise TrainingPreflightError(
                f"channel map layer {layer} does not match the selected donor layer"
            )
        indices = item.get("indices")
        expected_shape = (
            config.architecture.num_experts,
            config.architecture.expert_intermediate_size,
        )
        if (
            not isinstance(indices, list)
            or len(indices) != expected_shape[0]
            or any(
                not isinstance(group, list) or len(group) != expected_shape[1] for group in indices
            )
        ):
            raise TrainingPreflightError(
                f"channel map layer {layer} must have shape {expected_shape}"
            )
        flat = [value for group in indices for value in group]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in flat):
            raise TrainingPreflightError(
                f"channel map layer {layer} contains a non-integer channel"
            )
        if sorted(flat) != list(range(config.architecture.donor_intermediate_size)):
            raise TrainingPreflightError(
                f"channel map layer {layer} is not a complete channel partition"
            )
        declared = (
            item.get("num_channels"),
            item.get("num_experts"),
            item.get("expert_size"),
        )
        expected_declared = (
            config.architecture.donor_intermediate_size,
            config.architecture.num_experts,
            config.architecture.expert_intermediate_size,
        )
        if declared != expected_declared:
            raise TrainingPreflightError(
                f"channel map layer {layer} declared dimensions are inconsistent"
            )

    extra: list[tuple[str, Path]] = []
    if adapter_path is not None:
        try:
            from safetensors import safe_open

            with safe_open(adapter_path, framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                expected_keys = {
                    f"layers.{layer}.{name}"
                    for layer in range(config.architecture.student_layers)
                    for name in ("A", "B")
                }
                if keys != expected_keys:
                    raise TrainingPreflightError(
                        "ridge adapter tensor set does not cover exactly all layer A/B pairs"
                    )
                for layer in range(config.architecture.student_layers):
                    a_shape = tuple(handle.get_slice(f"layers.{layer}.A").get_shape())
                    b_shape = tuple(handle.get_slice(f"layers.{layer}.B").get_shape())
                    if a_shape != (
                        config.architecture.donor_hidden_size,
                        config.architecture.student_hidden_size,
                    ) or b_shape != (
                        config.architecture.student_hidden_size,
                        config.architecture.donor_hidden_size,
                    ):
                        raise TrainingPreflightError(
                            f"ridge adapter layer {layer} A/B shapes are incompatible"
                        )
        except TrainingPreflightError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise TrainingPreflightError(
                f"ridge adapter safetensors validation failed: {exc}"
            ) from exc
        sidecar = adapter_path.with_suffix(".json")
        _require_file(sidecar, None, "ridge adapter sidecar")
        metadata = _load_json_file(sidecar, "ridge adapter sidecar")
        if (
            metadata.get("schema_version") != 2
            or metadata.get("kind") != "bidirectional_ridge_adapters"
        ):
            raise TrainingPreflightError("ridge adapter sidecar has an unsupported schema/kind")
        artifact = metadata.get("artifact")
        if not isinstance(artifact, dict):
            raise TrainingPreflightError("ridge sidecar has no artifact identity")
        _require_same(
            "ridge adapter artifact SHA256", artifact.get("sha256"), sha256_file(adapter_path)
        )
        parent = metadata.get("layer_map")
        if not isinstance(parent, dict):
            raise TrainingPreflightError("ridge sidecar has no layer-map identity")
        _require_same(
            "ridge parent layer-map SHA256", parent.get("sha256"), sha256_file(layer_map_path)
        )
        _require_same("ridge track", metadata.get("track"), config.track)
        model_sources = metadata.get("model_sources")
        if not isinstance(model_sources, dict):
            raise TrainingPreflightError("ridge sidecar has no model-source lineage")
        for role, source_name in (("student", "backbone"), ("donor", "donor")):
            model = model_sources.get(role)
            if not isinstance(model, dict):
                raise TrainingPreflightError(f"ridge sidecar is missing {role} source")
            _require_same(
                f"ridge {role} manifest",
                model.get("manifest_sha256"),
                getattr(config.sources, source_name).manifest_sha256,
            )
        extra.append(("adapter_init_metadata", sidecar))
    return extra


def _qwen35_rope_buffer_bytes(text_config: Mapping[str, object]) -> int:
    """Return the two non-persistent FP32 RoPE ``inv_freq`` buffers.

    Current Qwen3.5 checkpoints store ``partial_rotary_factor`` inside
    ``rope_parameters``. Accept the older top-level spelling as a fallback,
    but never let a missing top-level key override the nested source value.
    """

    head_dim = int(text_config.get("head_dim", 0))
    rope_parameters = text_config.get("rope_parameters")
    nested_factor = (
        rope_parameters.get("partial_rotary_factor")
        if isinstance(rope_parameters, Mapping)
        else None
    )
    top_level_factor = text_config.get("partial_rotary_factor")
    factor = nested_factor if nested_factor is not None else top_level_factor
    partial_rotary_factor = 1.0 if factor is None else float(factor)
    rotary_dimension = int(head_dim * partial_rotary_factor)
    if head_dim <= 0 or rotary_dimension <= 0:
        raise TrainingPreflightError("teacher Qwen3.5 RoPE dimensions must be positive")
    # Qwen3_5TextRotaryEmbedding registers current/original inv_freq.
    return 2 * ((rotary_dimension + 1) // 2) * 4


def _source_mix_weight_contract(
    config: TrainConfig,
    authenticated_source_map: object,
) -> tuple[dict[str, int], dict[str, int], bool]:
    """Resolve lineage/effective weights under an explicit fail-closed policy."""

    lineage = dict(getattr(authenticated_source_map, "source_mix_weights", {}))
    configured = getattr(config.data, "source_mix_basis_points", None)
    if not isinstance(configured, Mapping):
        raise ValueError("configured source-mix weights are missing")
    effective = dict(configured)
    override = getattr(
        config.data,
        "source_mix_allow_weight_override",
        False,
    )
    if not isinstance(override, bool):
        raise ValueError("source-mix weight override flag must be a boolean")
    if lineage == effective:
        if override:
            raise ValueError(
                "source-mix weight override is enabled but effective weights "
                "do not differ from authenticated lineage"
            )
        return lineage, effective, False
    if not override:
        raise ValueError(
            "authenticated source-mix weights differ from configured effective "
            "weights; set data.source_mix_allow_weight_override=true only for "
            "an explicitly audited capacity override"
        )
    return lineage, effective, True


def run_training_preflight(
    config: TrainConfig, *, world_size: int | None = None
) -> PreflightReport:
    """Validate every local dependency, artifact identity, and offline invariant."""

    actual_world_size = int(os.environ.get("WORLD_SIZE", "1")) if world_size is None else world_size
    validate_optimizer_world_size(config, actual_world_size)
    enforce_offline_environment()
    if not config.runtime.offline:
        raise TrainingPreflightError("training configurations must set runtime.offline=true")
    checked: list[str] = []
    source_configs: dict[str, dict[str, object]] = {}
    cached_sources: dict[
        tuple[str, str, str, str],
        tuple[Path, Path, dict[str, object]],
    ] = {}
    for name in ("backbone", "donor", "teacher", "tokenizer"):
        source = getattr(config.sources, name)
        identity = (
            source.local_path,
            source.manifest_sha256,
            source.model_id,
            source.revision,
        )
        if identity not in cached_sources:
            cached_sources[identity] = _check_source(name, source)
            root, manifest, _ = cached_sources[identity]
            checked.extend((str(root), str(manifest)))
        source_configs[name] = cached_sources[identity][2]
    _audit_architecture(config, source_configs)

    data_manifest = _require_file(
        config.data.manifest_path, config.data.manifest_sha256, "training data manifest"
    )
    from .data import validate_prepared_corpus

    prepared_corpus = validate_prepared_corpus(data_manifest)
    prepared_lineage = getattr(prepared_corpus, "lineage", None)
    if isinstance(prepared_lineage, Mapping) and isinstance(
        prepared_lineage.get("quality_cooldown"), Mapping
    ):
        raise TrainingPreflightError(
            "a quality cooldown subset view cannot be used as the primary training corpus"
        )
    data_governance = _prepared_data_governance(prepared_corpus)
    _require_same(
        "prepared sequence length",
        prepared_corpus.sequence_length,
        config.data.max_sequence_length,
    )
    _require_same(
        "prepared tokenizer manifest",
        prepared_corpus.tokenizer_sha256,
        config.sources.tokenizer.manifest_sha256,
    )
    authenticated_source_map = None
    source_mix_dataset_fingerprint: str | None = None
    source_map_payload_json: str | None = None
    source_mix_lineage_basis_points: tuple[tuple[str, int], ...] = ()
    source_mix_effective_basis_points: tuple[tuple[str, int], ...] = ()
    source_mix_weight_override = False
    if config.data.source_mix_enabled():
        from .data.cursor import (
            SOURCE_MIX_ALGORITHM,
            AuthenticatedSourceMap,
            DeterministicSourceMixCursor,
        )

        try:
            authenticated_source_map = AuthenticatedSourceMap.from_prepared_manifest(
                prepared_corpus
            )
            if authenticated_source_map.fingerprint != config.data.source_map_sha256:
                raise ValueError(
                    "authenticated source-map SHA256 differs from the configured identity"
                )
            lineage_mix, effective_mix, source_mix_weight_override = _source_mix_weight_contract(
                config,
                authenticated_source_map,
            )
            source_mix_lineage_basis_points = tuple(sorted(lineage_mix.items()))
            source_mix_effective_basis_points = tuple(sorted(effective_mix.items()))
            if config.data.source_mix_algorithm != SOURCE_MIX_ALGORITHM:
                raise ValueError("configured source-mix algorithm is unsupported")
            source_mix_cursor = DeterministicSourceMixCursor(
                authenticated_source_map,
                effective_mix,
                seed=config.data.shuffle_seed,
            )
        except (TypeError, ValueError, OSError) as error:
            raise TrainingPreflightError(
                f"authenticated source-mix preflight failed: {error}"
            ) from error
        source_mix_dataset_fingerprint = source_mix_cursor.dataset_fingerprint
        source_map_payload_json = json.dumps(
            authenticated_source_map.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    data_mode = str(getattr(config.data, "mode", "teacher-kd"))
    kd_manifest: Path | None = None
    kd_corpus: object | None = None
    checked.append(str(data_manifest))
    if data_mode == "teacher-kd":
        kd_manifest = _require_file(
            config.data.teacher_kd_manifest_path,
            config.data.teacher_kd_manifest_sha256,
            "teacher KD manifest",
        )
        from .data import (
            read_kd_manifest,
            validate_kd_corpus_coverage,
            validate_kd_corpus_manifest,
        )

        kd_corpus = validate_kd_corpus_manifest(
            kd_manifest,
            expected_temperature=config.losses.kd_temperature,
        )
        validate_kd_corpus_coverage(kd_corpus, prepared_corpus)
        _require_same(
            "prepared/KD dataset fingerprint",
            kd_corpus.dataset_fingerprint,
            prepared_corpus.dataset_fingerprint,
        )
        _require_same(
            "KD teacher model_id", kd_corpus.teacher_model_id, config.sources.teacher.model_id
        )
        _require_same(
            "KD teacher revision", kd_corpus.teacher_revision, config.sources.teacher.revision
        )
        _require_same(
            "KD teacher manifest",
            kd_corpus.teacher_model_sha256,
            config.sources.teacher.manifest_sha256,
        )
        _require_same(
            "KD tokenizer manifest",
            kd_corpus.tokenizer_sha256,
            config.sources.tokenizer.manifest_sha256,
        )
        for entry in kd_corpus.shards:
            shard_manifest = read_kd_manifest(kd_manifest.parent / entry.path)
            _require_same(
                f"KD shard {entry.path} sequence length",
                shard_manifest.sequence_length,
                config.data.max_sequence_length,
            )
            _require_same(
                f"KD shard {entry.path} vocab_size",
                shard_manifest.vocab_size,
                source_configs["teacher"].get("vocab_size"),
            )
        checked.append(str(kd_manifest))
    elif data_mode != "prepared-text":
        raise TrainingPreflightError(f"unsupported training data mode: {data_mode!r}")

    quality_cooldown_summary = None
    cooldown_data_manifest: Path | None = None
    cooldown_kd_manifest: Path | None = None
    cooldown_prepared = None
    cooldown_authenticated_source_map = None
    cooldown_source_mix_dataset_fingerprint: str | None = None
    cooldown_source_mix_basis_points: tuple[tuple[str, int], ...] = ()
    cooldown_source_map_payload_json: str | None = None
    phase_disjointness_attestation: Path | None = None
    if config.data.quality_cooldown_enabled():
        cooldown_prepared_path = config.data.quality_cooldown_manifest_path
        cooldown_prepared_sha = config.data.quality_cooldown_manifest_sha256
        cooldown_start = config.data.quality_cooldown_start_tokens
        assert (
            cooldown_prepared_path is not None
            and cooldown_prepared_sha is not None
            and cooldown_start is not None
        )
        cooldown_data_manifest = _require_file(
            cooldown_prepared_path,
            cooldown_prepared_sha,
            "quality cooldown data manifest",
        )
        cooldown_prepared = validate_prepared_corpus(cooldown_data_manifest)
        _require_same(
            "quality cooldown prepared sequence length",
            cooldown_prepared.sequence_length,
            config.data.max_sequence_length,
        )
        _require_same(
            "quality cooldown tokenizer manifest",
            cooldown_prepared.tokenizer_sha256,
            config.sources.tokenizer.manifest_sha256,
        )
        if cooldown_prepared.dataset_fingerprint == prepared_corpus.dataset_fingerprint:
            raise TrainingPreflightError(
                "quality cooldown requires an independent prepared dataset fingerprint"
            )
        required_cooldown_tokens = config.optimizer.max_tokens - cooldown_start
        if (
            data_mode == "prepared-text"
            and cooldown_prepared.token_count < required_cooldown_tokens
        ):
            raise TrainingPreflightError(
                "quality cooldown corpus is too small: "
                f"requires {required_cooldown_tokens}, has {cooldown_prepared.token_count}"
            )

        if data_mode == "teacher-kd":
            if kd_manifest is None or kd_corpus is None:
                raise TrainingPreflightError(
                    "teacher-KD quality cooldown is missing the primary KD corpus"
                )
            cooldown_kd_path = config.data.quality_cooldown_teacher_kd_manifest_path
            cooldown_kd_sha = config.data.quality_cooldown_teacher_kd_manifest_sha256
            assert cooldown_kd_path is not None and cooldown_kd_sha is not None
            cooldown_kd_manifest = _require_file(
                cooldown_kd_path,
                cooldown_kd_sha,
                "quality cooldown teacher KD manifest",
            )
            cooldown_kd = validate_kd_corpus_manifest(
                cooldown_kd_manifest,
                expected_temperature=config.losses.kd_temperature,
            )
            validate_kd_corpus_coverage(cooldown_kd, cooldown_prepared)
            _require_same(
                "quality cooldown prepared/KD dataset fingerprint",
                cooldown_kd.dataset_fingerprint,
                cooldown_prepared.dataset_fingerprint,
            )
            _require_same(
                "quality cooldown KD teacher model_id",
                cooldown_kd.teacher_model_id,
                config.sources.teacher.model_id,
            )
            _require_same(
                "quality cooldown KD teacher revision",
                cooldown_kd.teacher_revision,
                config.sources.teacher.revision,
            )
            _require_same(
                "quality cooldown KD teacher manifest",
                cooldown_kd.teacher_model_sha256,
                config.sources.teacher.manifest_sha256,
            )
            _require_same(
                "quality cooldown KD tokenizer manifest",
                cooldown_kd.tokenizer_sha256,
                config.sources.tokenizer.manifest_sha256,
            )
            for entry in cooldown_kd.shards:
                shard_manifest = read_kd_manifest(cooldown_kd_manifest.parent / entry.path)
                _require_same(
                    f"quality cooldown KD shard {entry.path} sequence length",
                    shard_manifest.sequence_length,
                    config.data.max_sequence_length,
                )
                _require_same(
                    f"quality cooldown KD shard {entry.path} vocab_size",
                    shard_manifest.vocab_size,
                    source_configs["teacher"].get("vocab_size"),
                )
            from .data import validate_quality_cooldown_subset

            quality_cooldown_summary = validate_quality_cooldown_subset(
                prepared_corpus,
                kd_corpus,
                cooldown_prepared,
                cooldown_kd,
                primary_prepared_manifest_sha256=config.data.manifest_sha256,
                primary_kd_manifest_sha256=config.data.teacher_kd_manifest_sha256,
                required_cooldown_tokens=required_cooldown_tokens,
            )
            checked.extend((str(cooldown_data_manifest), str(cooldown_kd_manifest)))
        else:
            cooldown_governance = _prepared_data_governance(cooldown_prepared)
            if (
                not cooldown_governance.ready_for_training
                or cooldown_governance.research_only
                or cooldown_governance.pending_audits
            ):
                raise TrainingPreflightError(
                    "prepared-text quality cooldown corpus must be fully training-ready"
                )
            cooldown_source_mix_token_counts: tuple[tuple[str, int], ...] = ()
            if authenticated_source_map is not None:
                from .data.cursor import (
                    AuthenticatedSourceMap,
                    DeterministicSourceMixCursor,
                )

                try:
                    cooldown_authenticated_source_map = (
                        AuthenticatedSourceMap.from_prepared_manifest(cooldown_prepared)
                    )
                    cooldown_weights = cooldown_authenticated_source_map.source_mix_weights
                    if not cooldown_weights:
                        raise ValueError("cooldown prepared lineage has no source-mix weights")
                    cooldown_cursor = DeterministicSourceMixCursor(
                        cooldown_authenticated_source_map,
                        cooldown_weights,
                        seed=config.data.shuffle_seed,
                    )
                except (TypeError, ValueError, OSError) as error:
                    raise TrainingPreflightError(
                        f"authenticated cooldown source-mix preflight failed: {error}"
                    ) from error
                if (
                    cooldown_authenticated_source_map.fingerprint
                    == authenticated_source_map.fingerprint
                ):
                    raise TrainingPreflightError(
                        "quality cooldown requires an independent source-map identity"
                    )
                cooldown_source_mix_dataset_fingerprint = cooldown_cursor.dataset_fingerprint
                cooldown_source_mix_basis_points = tuple(sorted(cooldown_weights.items()))
                cooldown_source_map_payload_json = json.dumps(
                    cooldown_authenticated_source_map.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                entries_by_id = {entry.shard_id: entry for entry in cooldown_prepared.shards}
                token_counts: dict[str, int] = {}
                for source_id in cooldown_authenticated_source_map.source_ids:
                    token_counts[source_id] = sum(
                        entries_by_id[shard.shard_id].token_count
                        for shard in (
                            cooldown_authenticated_source_map.shards_for_source(source_id)
                        )
                    )
                if sum(token_counts.values()) != cooldown_prepared.token_count:
                    raise TrainingPreflightError(
                        "cooldown source-map token inventory differs from prepared corpus"
                    )
                cooldown_source_mix_token_counts = tuple(sorted(token_counts.items()))
            quality_cooldown_summary = _PreparedTextCooldownSummary(
                cooldown_dataset_fingerprint=(cooldown_prepared.dataset_fingerprint),
                selected_shard_ids=tuple(entry.shard_id for entry in cooldown_prepared.shards),
                source_mix_token_counts=cooldown_source_mix_token_counts,
                sequence_count=cooldown_prepared.sequence_count,
                token_count=cooldown_prepared.token_count,
            )
            checked.append(str(cooldown_data_manifest))

    if config.data.phase_disjointness_attestation_path is not None:
        if (
            data_mode != "prepared-text"
            or cooldown_data_manifest is None
            or cooldown_prepared is None
            or authenticated_source_map is None
            or cooldown_authenticated_source_map is None
        ):
            raise TrainingPreflightError(
                "phase disjointness requires two authenticated prepared-text source maps"
            )
        phase_disjointness_attestation = _require_file(
            config.data.phase_disjointness_attestation_path,
            config.data.phase_disjointness_attestation_sha256,
            "phase disjointness attestation",
        )
        _validate_phase_disjointness_attestation(
            phase_disjointness_attestation,
            primary_manifest=data_manifest,
            primary_prepared=prepared_corpus,
            primary_source_map=authenticated_source_map,
            cooldown_manifest=cooldown_data_manifest,
            cooldown_prepared=cooldown_prepared,
            cooldown_source_map=cooldown_authenticated_source_map,
        )
        checked.append(str(phase_disjointness_attestation))

    if data_mode == "prepared-text" and not config.data.allow_corpus_reuse:
        primary_phase_budget = (
            int(config.data.quality_cooldown_start_tokens)
            if quality_cooldown_summary is not None
            else config.optimizer.max_tokens
        )
        _validate_no_reuse_capacity(
            prepared_corpus,
            label="primary",
            phase_budget_tokens=primary_phase_budget,
            global_batch_tokens=config.data.global_batch_tokens,
            sequence_length=config.data.max_sequence_length,
            source_map=authenticated_source_map,
            source_mix_basis_points=(
                dict(source_mix_effective_basis_points)
                if authenticated_source_map is not None
                else None
            ),
            seed=config.data.shuffle_seed,
        )
        if quality_cooldown_summary is not None:
            assert cooldown_prepared is not None
            assert config.data.quality_cooldown_start_tokens is not None
            _validate_no_reuse_capacity(
                cooldown_prepared,
                label="quality cooldown",
                phase_budget_tokens=(
                    config.optimizer.max_tokens - config.data.quality_cooldown_start_tokens
                ),
                global_batch_tokens=config.data.global_batch_tokens,
                sequence_length=config.data.max_sequence_length,
                source_map=cooldown_authenticated_source_map,
                source_mix_basis_points=(
                    dict(cooldown_source_mix_basis_points)
                    if cooldown_authenticated_source_map is not None
                    else None
                ),
                seed=config.data.shuffle_seed,
            )

    layer_map_path = _require_file(config.architecture.layer_map_path, None, "layer map")
    channel_map_path = _require_file(config.architecture.channel_map_path, None, "channel map")
    calibration_files = [
        ("layer_map", layer_map_path),
        ("channel_map", channel_map_path),
    ]
    adapter_path: Path | None = None
    if config.stage == "dense-oracle":
        adapter_path = _require_file(
            config.architecture.adapter_init_path,
            None,
            "ridge adapter initialization",
        )
        calibration_files.append(("adapter_init", adapter_path))
    calibration_files.extend(
        _validate_calibration_contract(
            config,
            layer_map_path,
            channel_map_path,
            adapter_path,
        )
    )
    checked.extend(str(path) for _, path in calibration_files)

    if config.stage == "sparse":
        folded = _require_file(
            config.sources.folded_experts_path or "",
            config.sources.folded_experts_sha256,
            "folded experts",
        )
        checked.append(str(folded))
        folded_manifest = _require_file(
            folded.parent / "manifest.json", None, "folded expert manifest"
        )
        folded_metadata = _load_json_file(folded_manifest, "folded expert manifest")
        source_checkpoint_sha = folded_metadata.get("source_checkpoint_complete_sha256")
        if (
            not isinstance(source_checkpoint_sha, str)
            or len(source_checkpoint_sha) != 64
            or any(
                character not in "0123456789abcdef" for character in source_checkpoint_sha.lower()
            )
        ):
            raise TrainingPreflightError(
                "folded expert manifest has no valid source checkpoint COMPLETE hash"
            )
        for label, actual, expected in (
            ("folded artifact name", folded_metadata.get("artifact"), folded.name),
            (
                "folded artifact SHA256",
                folded_metadata.get("artifact_sha256"),
                config.sources.folded_experts_sha256,
            ),
            ("folded track", folded_metadata.get("track"), config.track),
            (
                "folded layer count",
                folded_metadata.get("num_layers"),
                config.architecture.student_layers,
            ),
            (
                "folded expert count",
                folded_metadata.get("num_experts"),
                config.architecture.num_experts,
            ),
            (
                "folded expert width",
                folded_metadata.get("expert_intermediate_size"),
                config.architecture.expert_intermediate_size,
            ),
        ):
            _require_same(label, actual, expected)
        source_manifests = folded_metadata.get("source_manifests")
        if not isinstance(source_manifests, dict):
            raise TrainingPreflightError("folded expert manifest has no source lineage")
        _require_same(
            "folded backbone manifest",
            source_manifests.get("backbone"),
            config.sources.backbone.manifest_sha256,
        )
        _require_same(
            "folded donor manifest",
            source_manifests.get("donor"),
            config.sources.donor.manifest_sha256,
        )
        folded_calibration = folded_metadata.get("calibration_artifacts")
        if not isinstance(folded_calibration, dict):
            raise TrainingPreflightError("folded expert manifest has no calibration lineage")
        _require_same(
            "folded/current layer map",
            folded_calibration.get("layer_map"),
            sha256_file(layer_map_path),
        )
        _require_same(
            "folded/current channel map",
            folded_calibration.get("channel_map"),
            sha256_file(channel_map_path),
        )
        calibration_files.append(("folded_manifest", folded_manifest))
        checked.append(str(folded_manifest))

    batch = compute_batch_geometry(config, actual_world_size)
    teacher_cpu_shadow_bytes = 0
    teacher_gpu_stage_bytes = 0
    if config.runtime.teacher_cpu_offload:
        if actual_world_size != 1:
            raise TrainingPreflightError("runtime.teacher_cpu_offload supports exactly one GPU")
        from .hardware import estimate_static_training_memory

        memory = estimate_static_training_memory(config, world_size=actual_world_size)
        cpu_shadow = next(
            (
                component
                for component in memory.components
                if component.name == "frozen_hidden_alignment_teacher_cpu_shadow"
            ),
            None,
        )
        if cpu_shadow is None:
            raise TrainingPreflightError(
                "teacher CPU offload memory inventory has no CPU shadow component"
            )
        teacher_text_config = source_configs["teacher"]
        teacher_buffer_bytes = _qwen35_rope_buffer_bytes(teacher_text_config)
        teacher_cpu_shadow_bytes = cpu_shadow.bytes + teacher_buffer_bytes
        teacher_gpu_stage_bytes = cpu_shadow.bytes + teacher_buffer_bytes
    data_fingerprint = sha256_file(data_manifest)
    if data_mode == "prepared-text" and quality_cooldown_summary is not None:
        assert cooldown_data_manifest is not None
        data_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "kind": "two-phase-prepared-text-quality-cooldown-data-v1",
                    "primary_prepared_sha256": sha256_file(data_manifest),
                    "primary_dataset_fingerprint": (prepared_corpus.dataset_fingerprint),
                    "primary_source_map_sha256": (
                        authenticated_source_map.fingerprint
                        if authenticated_source_map is not None
                        else None
                    ),
                    "primary_source_mix_dataset_fingerprint": (source_mix_dataset_fingerprint),
                    "primary_source_mix_effective_basis_points": dict(
                        source_mix_effective_basis_points
                    ),
                    "cooldown_prepared_sha256": sha256_file(cooldown_data_manifest),
                    "cooldown_dataset_fingerprint": (
                        quality_cooldown_summary.cooldown_dataset_fingerprint
                    ),
                    "cooldown_source_map_sha256": (
                        cooldown_authenticated_source_map.fingerprint
                        if cooldown_authenticated_source_map is not None
                        else None
                    ),
                    "cooldown_source_mix_dataset_fingerprint": (
                        cooldown_source_mix_dataset_fingerprint
                    ),
                    "cooldown_source_mix_basis_points": dict(cooldown_source_mix_basis_points),
                    "cooldown_start_tokens": (config.data.quality_cooldown_start_tokens),
                    "ordered_cooldown_shard_ids": list(quality_cooldown_summary.selected_shard_ids),
                    "phase_disjointness_attestation_sha256": (
                        sha256_file(phase_disjointness_attestation)
                        if phase_disjointness_attestation is not None
                        else None
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    elif authenticated_source_map is not None:
        assert source_mix_dataset_fingerprint is not None
        data_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "kind": "authenticated-source-mixed-prepared-text-data-v2",
                    "prepared_manifest_sha256": sha256_file(data_manifest),
                    "prepared_dataset_fingerprint": (prepared_corpus.dataset_fingerprint),
                    "source_map_sha256": authenticated_source_map.fingerprint,
                    "source_mix_algorithm": config.data.source_mix_algorithm,
                    "source_mix_lineage_basis_points": dict(source_mix_lineage_basis_points),
                    "source_mix_effective_basis_points": dict(source_mix_effective_basis_points),
                    "source_mix_weight_override": source_mix_weight_override,
                    "source_mix_dataset_fingerprint": (source_mix_dataset_fingerprint),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    elif quality_cooldown_summary is not None:
        assert (
            kd_manifest is not None
            and cooldown_data_manifest is not None
            and cooldown_kd_manifest is not None
        )
        data_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "kind": "two-phase-quality-cooldown-data",
                    "primary_prepared_sha256": sha256_file(data_manifest),
                    "primary_kd_sha256": sha256_file(kd_manifest),
                    "cooldown_prepared_sha256": sha256_file(cooldown_data_manifest),
                    "cooldown_kd_sha256": sha256_file(cooldown_kd_manifest),
                    "cooldown_start_tokens": config.data.quality_cooldown_start_tokens,
                    "cooldown_dataset_fingerprint": (
                        quality_cooldown_summary.cooldown_dataset_fingerprint
                    ),
                    "ordered_shard_ids": list(quality_cooldown_summary.selected_shard_ids),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    source_tree_sha256 = twen_source_tree_sha256()
    calibration_fingerprints = tuple((name, sha256_file(path)) for name, path in calibration_files)
    critical_payload = json.dumps(
        {
            "config": config.fingerprint(),
            "calibration_artifacts": dict(calibration_fingerprints),
            "source_tree_sha256": source_tree_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    effective_config_fingerprint = hashlib.sha256(critical_payload).hexdigest()
    activation_checkpoint_layer_indices = config.activation_checkpoint_layer_indices(
        align_hidden=False
    )
    hidden_alignment_activation_checkpoint_layer_indices = (
        config.activation_checkpoint_layer_indices(align_hidden=True)
    )
    dense_transfer_token_checkpoint_layer_indices = config.dense_transfer_checkpoint_layer_indices(
        align_hidden=False,
        outer_checkpoint_layer_indices=activation_checkpoint_layer_indices,
    )
    hidden_alignment_dense_transfer_token_checkpoint_layer_indices = (
        config.dense_transfer_checkpoint_layer_indices(
            align_hidden=True,
            outer_checkpoint_layer_indices=(hidden_alignment_activation_checkpoint_layer_indices),
        )
    )
    return PreflightReport(
        config_fingerprint=effective_config_fingerprint,
        data_fingerprint=data_fingerprint,
        source_tree_sha256=source_tree_sha256,
        batch=batch,
        checked_paths=tuple(checked),
        calibration_fingerprints=calibration_fingerprints,
        data_governance=data_governance,
        teacher_cpu_shadow_bytes=teacher_cpu_shadow_bytes,
        teacher_gpu_stage_bytes=teacher_gpu_stage_bytes,
        activation_checkpoint_layer_count=(config.runtime.activation_checkpoint_layer_count),
        hidden_alignment_activation_checkpoint_layer_count=(
            config.runtime.hidden_alignment_activation_checkpoint_layer_count
        ),
        activation_checkpoint_layer_indices=activation_checkpoint_layer_indices,
        hidden_alignment_activation_checkpoint_layer_indices=(
            hidden_alignment_activation_checkpoint_layer_indices
        ),
        dense_transfer_execution=config.runtime.dense_transfer_execution,
        dense_transfer_checkpoint_layer_count=(
            config.runtime.dense_transfer_checkpoint_layer_count
        ),
        hidden_alignment_dense_transfer_checkpoint_layer_count=(
            config.runtime.hidden_alignment_dense_transfer_checkpoint_layer_count
        ),
        dense_transfer_token_checkpoint_layer_indices=(
            dense_transfer_token_checkpoint_layer_indices
        ),
        hidden_alignment_dense_transfer_token_checkpoint_layer_indices=(
            hidden_alignment_dense_transfer_token_checkpoint_layer_indices
        ),
        quality_cooldown_enabled=quality_cooldown_summary is not None,
        quality_cooldown_start_tokens=(
            config.data.quality_cooldown_start_tokens
            if quality_cooldown_summary is not None
            else None
        ),
        quality_cooldown_dataset_fingerprint=(
            quality_cooldown_summary.cooldown_dataset_fingerprint
            if quality_cooldown_summary is not None
            else None
        ),
        quality_cooldown_sequence_count=(
            quality_cooldown_summary.sequence_count if quality_cooldown_summary is not None else 0
        ),
        quality_cooldown_token_count=(
            quality_cooldown_summary.token_count if quality_cooldown_summary is not None else 0
        ),
        quality_cooldown_selected_shard_ids=(
            quality_cooldown_summary.selected_shard_ids
            if quality_cooldown_summary is not None
            else ()
        ),
        quality_cooldown_source_mix_token_counts=(
            quality_cooldown_summary.source_mix_token_counts
            if quality_cooldown_summary is not None
            else ()
        ),
        quality_cooldown_source_mix_enabled=(cooldown_authenticated_source_map is not None),
        quality_cooldown_source_mix_algorithm=(
            config.data.source_mix_algorithm
            if cooldown_authenticated_source_map is not None
            else None
        ),
        quality_cooldown_source_map_sha256=(
            cooldown_authenticated_source_map.fingerprint
            if cooldown_authenticated_source_map is not None
            else None
        ),
        quality_cooldown_source_mix_dataset_fingerprint=(cooldown_source_mix_dataset_fingerprint),
        quality_cooldown_source_mix_basis_points=(cooldown_source_mix_basis_points),
        quality_cooldown_source_mix_seed=(
            config.data.shuffle_seed if cooldown_authenticated_source_map is not None else None
        ),
        quality_cooldown_source_map_payload_json=(cooldown_source_map_payload_json),
        source_mix_enabled=authenticated_source_map is not None,
        source_mix_algorithm=(
            config.data.source_mix_algorithm if authenticated_source_map is not None else None
        ),
        source_map_sha256=(
            authenticated_source_map.fingerprint if authenticated_source_map is not None else None
        ),
        source_mix_dataset_fingerprint=source_mix_dataset_fingerprint,
        source_mix_basis_points=(
            source_mix_effective_basis_points if authenticated_source_map is not None else ()
        ),
        source_mix_lineage_basis_points=source_mix_lineage_basis_points,
        source_mix_effective_basis_points=source_mix_effective_basis_points,
        source_mix_weight_override=source_mix_weight_override,
        source_mix_seed=(
            config.data.shuffle_seed if authenticated_source_map is not None else None
        ),
        source_map_payload_json=source_map_payload_json,
    )


def _preflight_report_payload(report: PreflightReport) -> dict[str, object]:
    return {
        "config_fingerprint": report.config_fingerprint,
        "data_fingerprint": report.data_fingerprint,
        "source_tree_sha256": report.source_tree_sha256,
        "batch": {
            "world_size": report.batch.world_size,
            "micro_batch_tokens_per_rank": report.batch.micro_batch_tokens_per_rank,
            "gradient_accumulation_steps": report.batch.gradient_accumulation_steps,
            "global_batch_tokens": report.batch.global_batch_tokens,
        },
        "checked_paths": list(report.checked_paths),
        "calibration_fingerprints": [list(item) for item in report.calibration_fingerprints],
        "data_governance": {
            "lineage_kind": report.data_governance.lineage_kind,
            "research_only": report.data_governance.research_only,
            "ready_for_training": report.data_governance.ready_for_training,
            "pending_audits": list(report.data_governance.pending_audits),
            "warning": report.data_governance.warning,
        },
        "teacher_cpu_shadow_bytes": report.teacher_cpu_shadow_bytes,
        "teacher_gpu_stage_bytes": report.teacher_gpu_stage_bytes,
        "activation_checkpoint_layer_count": report.activation_checkpoint_layer_count,
        "hidden_alignment_activation_checkpoint_layer_count": (
            report.hidden_alignment_activation_checkpoint_layer_count
        ),
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
            "source_mix": {
                "enabled": report.quality_cooldown_source_mix_enabled,
                "algorithm": report.quality_cooldown_source_mix_algorithm,
                "source_map_sha256": report.quality_cooldown_source_map_sha256,
                "dataset_fingerprint": (report.quality_cooldown_source_mix_dataset_fingerprint),
                "basis_points": [
                    list(item) for item in report.quality_cooldown_source_mix_basis_points
                ],
                "seed": report.quality_cooldown_source_mix_seed,
                "source_map": (
                    json.loads(report.quality_cooldown_source_map_payload_json)
                    if report.quality_cooldown_source_map_payload_json is not None
                    else None
                ),
            },
        },
        "source_mix": {
            "enabled": report.source_mix_enabled,
            "algorithm": report.source_mix_algorithm,
            "source_map_sha256": report.source_map_sha256,
            "dataset_fingerprint": report.source_mix_dataset_fingerprint,
            "basis_points": [list(item) for item in report.source_mix_basis_points],
            "lineage_basis_points": [list(item) for item in report.source_mix_lineage_basis_points],
            "effective_basis_points": [
                list(item) for item in report.source_mix_effective_basis_points
            ],
            "weight_override": report.source_mix_weight_override,
            "seed": report.source_mix_seed,
            "source_map": (
                json.loads(report.source_map_payload_json)
                if report.source_map_payload_json is not None
                else None
            ),
        },
    }


def _preflight_report_from_payload(value: object) -> PreflightReport:
    if not isinstance(value, dict) or not isinstance(value.get("batch"), dict):
        raise TrainingPreflightError("rank zero returned an invalid preflight report")
    batch = value["batch"]
    governance = value.get("data_governance")
    if not isinstance(governance, dict):
        raise TrainingPreflightError("rank zero returned no data governance status")
    raw_pending = governance.get("pending_audits")
    if not isinstance(raw_pending, list) or not all(
        isinstance(item, str) and item for item in raw_pending
    ):
        raise TrainingPreflightError("rank zero returned invalid pending data audits")
    raw_warning = governance.get("warning")
    if raw_warning is not None and not isinstance(raw_warning, str):
        raise TrainingPreflightError("rank zero returned an invalid data governance warning")
    source_tree_sha256 = value.get("source_tree_sha256")
    if (
        not isinstance(source_tree_sha256, str)
        or len(source_tree_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_tree_sha256)
    ):
        raise TrainingPreflightError("rank zero returned an invalid Twen source tree SHA256")

    def checkpoint_count(field: str) -> int | None:
        raw = value.get(field)
        if raw is not None and (isinstance(raw, bool) or not isinstance(raw, int) or raw < 0):
            raise TrainingPreflightError(
                f"rank zero returned an invalid checkpoint layer count: {field}"
            )
        return raw

    checkpoint_layer_count = checkpoint_count("activation_checkpoint_layer_count")
    hidden_alignment_checkpoint_layer_count = checkpoint_count(
        "hidden_alignment_activation_checkpoint_layer_count"
    )
    dense_transfer_checkpoint_layer_count = checkpoint_count(
        "dense_transfer_checkpoint_layer_count"
    )
    hidden_alignment_dense_transfer_checkpoint_layer_count = checkpoint_count(
        "hidden_alignment_dense_transfer_checkpoint_layer_count"
    )
    dense_transfer_execution = value.get("dense_transfer_execution", "expanded")
    if dense_transfer_execution not in {"expanded", "differentiable_folded"}:
        raise TrainingPreflightError("rank zero returned an invalid dense transfer execution")
    raw_cooldown = value.get("quality_cooldown", {"enabled": False})
    if not isinstance(raw_cooldown, dict) or not isinstance(raw_cooldown.get("enabled"), bool):
        raise TrainingPreflightError("rank zero returned invalid quality cooldown status")
    cooldown_enabled = raw_cooldown["enabled"]
    cooldown_start = raw_cooldown.get("start_tokens")
    cooldown_fingerprint = raw_cooldown.get("dataset_fingerprint")
    cooldown_sequence_count = raw_cooldown.get("sequence_count", 0)
    cooldown_token_count = raw_cooldown.get("token_count", 0)
    raw_cooldown_shards = raw_cooldown.get("selected_shard_ids", [])
    raw_cooldown_mix = raw_cooldown.get("source_mix_token_counts", [])
    raw_cooldown_source_mix = raw_cooldown.get(
        "source_mix",
        {"enabled": False},
    )
    if not isinstance(raw_cooldown_source_mix, dict) or not isinstance(
        raw_cooldown_source_mix.get("enabled"), bool
    ):
        raise TrainingPreflightError("rank zero returned invalid cooldown source-mix status")
    cooldown_source_mix_enabled = raw_cooldown_source_mix["enabled"]
    cooldown_source_mix_algorithm = raw_cooldown_source_mix.get("algorithm")
    cooldown_source_map_sha256 = raw_cooldown_source_mix.get("source_map_sha256")
    cooldown_source_mix_dataset_fingerprint = raw_cooldown_source_mix.get("dataset_fingerprint")
    raw_cooldown_source_mix_basis_points = raw_cooldown_source_mix.get(
        "basis_points",
        [],
    )
    cooldown_source_mix_seed = raw_cooldown_source_mix.get("seed")
    raw_cooldown_source_map = raw_cooldown_source_mix.get("source_map")
    if cooldown_enabled:
        if (
            isinstance(cooldown_start, bool)
            or not isinstance(cooldown_start, int)
            or cooldown_start <= 0
            or not isinstance(cooldown_fingerprint, str)
            or not cooldown_fingerprint
            or isinstance(cooldown_sequence_count, bool)
            or not isinstance(cooldown_sequence_count, int)
            or cooldown_sequence_count <= 0
            or isinstance(cooldown_token_count, bool)
            or not isinstance(cooldown_token_count, int)
            or cooldown_token_count <= 0
        ):
            raise TrainingPreflightError("rank zero returned incomplete quality cooldown status")
    elif any(
        value not in (None, 0, [], ())
        for value in (
            cooldown_start,
            cooldown_fingerprint,
            cooldown_sequence_count,
            cooldown_token_count,
            raw_cooldown_shards,
            raw_cooldown_mix,
        )
    ):
        raise TrainingPreflightError("disabled quality cooldown contains active state")
    if not isinstance(raw_cooldown_shards, list) or not all(
        isinstance(item, str) and item for item in raw_cooldown_shards
    ):
        raise TrainingPreflightError("rank zero returned invalid cooldown shard IDs")
    if len(set(raw_cooldown_shards)) != len(raw_cooldown_shards):
        raise TrainingPreflightError("rank zero returned duplicate cooldown shard IDs")
    cooldown_mix: list[tuple[str, int]] = []
    if not isinstance(raw_cooldown_mix, list):
        raise TrainingPreflightError("rank zero returned invalid cooldown source mix")
    for item in raw_cooldown_mix:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or isinstance(item[1], bool)
            or not isinstance(item[1], int)
            or item[1] <= 0
        ):
            raise TrainingPreflightError("rank zero returned invalid cooldown source mix")
        cooldown_mix.append((item[0], item[1]))
    if cooldown_mix != sorted(cooldown_mix) or len({item[0] for item in cooldown_mix}) != len(
        cooldown_mix
    ):
        raise TrainingPreflightError("rank zero returned non-canonical cooldown source mix")
    if cooldown_enabled and (
        not raw_cooldown_shards
        or (bool(cooldown_mix) and sum(item[1] for item in cooldown_mix) != cooldown_token_count)
        or (cooldown_source_mix_enabled and not cooldown_mix)
    ):
        raise TrainingPreflightError("rank zero returned incomplete cooldown corpus inventory")

    raw_source_mix = value.get("source_mix", {"enabled": False})
    if not isinstance(raw_source_mix, dict) or not isinstance(raw_source_mix.get("enabled"), bool):
        raise TrainingPreflightError("rank zero returned invalid source-mix status")
    source_mix_enabled = raw_source_mix["enabled"]
    source_mix_algorithm = raw_source_mix.get("algorithm")
    source_map_sha256 = raw_source_mix.get("source_map_sha256")
    source_mix_dataset_fingerprint = raw_source_mix.get("dataset_fingerprint")
    raw_source_mix_basis_points = raw_source_mix.get("basis_points", [])
    raw_source_mix_lineage_basis_points = raw_source_mix.get(
        "lineage_basis_points",
        [],
    )
    raw_source_mix_effective_basis_points = raw_source_mix.get(
        "effective_basis_points",
        [],
    )
    source_mix_weight_override = raw_source_mix.get("weight_override", False)
    source_mix_seed = raw_source_mix.get("seed")
    raw_source_map = raw_source_mix.get("source_map")
    source_mix_basis_points: list[tuple[str, int]] = []
    source_mix_lineage_basis_points: list[tuple[str, int]] = []
    source_mix_effective_basis_points: list[tuple[str, int]] = []

    def source_mix_weights(
        raw: object,
        *,
        label: str,
        allow_empty: bool,
    ) -> list[tuple[str, int]]:
        if not isinstance(raw, list):
            raise TrainingPreflightError(
                f"rank zero returned invalid {label} source-mix basis points"
            )
        parsed: list[tuple[str, int]] = []
        for item in raw:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0]
                or isinstance(item[1], bool)
                or not isinstance(item[1], int)
                or item[1] <= 0
            ):
                raise TrainingPreflightError(
                    f"rank zero returned invalid {label} source-mix basis points"
                )
            parsed.append((item[0], item[1]))
        if (
            parsed != sorted(parsed)
            or len({item[0] for item in parsed}) != len(parsed)
            or (parsed and sum(item[1] for item in parsed) != 10_000)
            or (not parsed and not allow_empty)
        ):
            raise TrainingPreflightError(
                f"rank zero returned non-canonical {label} source-mix basis points"
            )
        return parsed

    cooldown_source_mix_basis_points: list[tuple[str, int]] = []
    cooldown_source_map_payload_json: str | None = None
    cooldown_source_map = None
    if cooldown_source_mix_enabled:
        from .data.cursor import (
            SOURCE_MIX_ALGORITHM,
            AuthenticatedSourceMap,
            DeterministicSourceMixCursor,
        )

        if (
            not cooldown_enabled
            or not source_mix_enabled
            or cooldown_source_mix_algorithm != SOURCE_MIX_ALGORITHM
            or not isinstance(cooldown_source_map_sha256, str)
            or len(cooldown_source_map_sha256) != 64
            or any(character not in "0123456789abcdef" for character in cooldown_source_map_sha256)
            or not isinstance(
                cooldown_source_mix_dataset_fingerprint,
                str,
            )
            or len(cooldown_source_mix_dataset_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in cooldown_source_mix_dataset_fingerprint
            )
            or isinstance(cooldown_source_mix_seed, bool)
            or not isinstance(cooldown_source_mix_seed, int)
            or not isinstance(raw_cooldown_source_map, Mapping)
        ):
            raise TrainingPreflightError(
                "rank zero returned incomplete cooldown source-mix identity"
            )
        cooldown_source_mix_basis_points = source_mix_weights(
            raw_cooldown_source_mix_basis_points,
            label="cooldown",
            allow_empty=False,
        )
        try:
            cooldown_source_map = AuthenticatedSourceMap.from_dict(raw_cooldown_source_map)
            if cooldown_source_map.fingerprint != cooldown_source_map_sha256:
                raise ValueError("cooldown source-map fingerprint mismatch")
            if cooldown_source_map.source_mix_weights != dict(cooldown_source_mix_basis_points):
                raise ValueError("cooldown source-map weights mismatch")
            cooldown_source_mix_cursor = DeterministicSourceMixCursor(
                cooldown_source_map,
                dict(cooldown_source_mix_basis_points),
                seed=cooldown_source_mix_seed,
            )
        except (TypeError, ValueError) as error:
            raise TrainingPreflightError(
                f"rank zero returned invalid cooldown source-map payload: {error}"
            ) from error
        if (
            cooldown_source_mix_cursor.dataset_fingerprint
            != cooldown_source_mix_dataset_fingerprint
        ):
            raise TrainingPreflightError(
                "rank zero cooldown source-mix dataset fingerprint mismatch"
            )
        cooldown_source_map_payload_json = json.dumps(
            cooldown_source_map.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    else:
        if any(
            item not in (None, [], ())
            for item in (
                cooldown_source_mix_algorithm,
                cooldown_source_map_sha256,
                cooldown_source_mix_dataset_fingerprint,
                raw_cooldown_source_mix_basis_points,
                cooldown_source_mix_seed,
                raw_cooldown_source_map,
            )
        ):
            raise TrainingPreflightError("disabled cooldown source mixing contains active identity")

    if source_mix_enabled:
        from .data.cursor import (
            SOURCE_MIX_ALGORITHM,
            AuthenticatedSourceMap,
            DeterministicSourceMixCursor,
        )

        if (
            source_mix_algorithm != SOURCE_MIX_ALGORITHM
            or not isinstance(source_map_sha256, str)
            or len(source_map_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_map_sha256)
            or not isinstance(source_mix_dataset_fingerprint, str)
            or len(source_mix_dataset_fingerprint) != 64
            or any(
                character not in "0123456789abcdef" for character in source_mix_dataset_fingerprint
            )
            or not isinstance(source_mix_weight_override, bool)
            or isinstance(source_mix_seed, bool)
            or not isinstance(source_mix_seed, int)
            or not isinstance(raw_source_map, Mapping)
        ):
            raise TrainingPreflightError("rank zero returned incomplete source-mix identity")
        source_mix_basis_points = source_mix_weights(
            raw_source_mix_basis_points,
            label="compatibility",
            allow_empty=False,
        )
        source_mix_lineage_basis_points = source_mix_weights(
            raw_source_mix_lineage_basis_points,
            label="lineage",
            allow_empty=True,
        )
        source_mix_effective_basis_points = source_mix_weights(
            raw_source_mix_effective_basis_points,
            label="effective",
            allow_empty=False,
        )
        if source_mix_basis_points != source_mix_effective_basis_points:
            raise TrainingPreflightError(
                "rank zero source-mix compatibility/effective weights differ"
            )
        try:
            source_map = AuthenticatedSourceMap.from_dict(raw_source_map)
            if source_map.fingerprint != source_map_sha256:
                raise ValueError("source-map fingerprint mismatch")
            if source_map.source_mix_weights != dict(source_mix_lineage_basis_points):
                raise ValueError("source-map lineage weights mismatch")
            weights_differ = source_mix_lineage_basis_points != source_mix_effective_basis_points
            if source_mix_weight_override != weights_differ:
                raise ValueError(
                    "source-mix override flag disagrees with lineage/effective weights"
                )
            source_mix_cursor = DeterministicSourceMixCursor(
                source_map,
                dict(source_mix_effective_basis_points),
                seed=source_mix_seed,
            )
        except (TypeError, ValueError) as error:
            raise TrainingPreflightError(
                f"rank zero returned invalid source-map payload: {error}"
            ) from error
        if source_mix_cursor.dataset_fingerprint != source_mix_dataset_fingerprint:
            raise TrainingPreflightError("rank zero source-mix dataset fingerprint mismatch")
        if cooldown_source_mix_enabled and (
            cooldown_source_map is None
            or cooldown_source_map.fingerprint == source_map.fingerprint
            or cooldown_source_mix_seed != source_mix_seed
        ):
            raise TrainingPreflightError(
                "rank zero cooldown source-mix identity is not independent/coordinated"
            )
        source_map_payload_json = json.dumps(
            source_map.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    else:
        if (
            any(
                item not in (None, [], ())
                for item in (
                    source_mix_algorithm,
                    source_map_sha256,
                    source_mix_dataset_fingerprint,
                    raw_source_mix_basis_points,
                    raw_source_mix_lineage_basis_points,
                    raw_source_mix_effective_basis_points,
                    source_mix_seed,
                    raw_source_map,
                )
            )
            or source_mix_weight_override is not False
        ):
            raise TrainingPreflightError("disabled source mixing contains active identity")
        source_map_payload_json = None

    def checkpoint_indices(field: str) -> tuple[int, ...]:
        raw = value.get(field, [])
        if not isinstance(raw, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in raw
        ):
            raise TrainingPreflightError(
                f"rank zero returned invalid activation checkpoint layer indices: {field}"
            )
        indices = tuple(raw)
        if tuple(sorted(set(indices))) != indices:
            raise TrainingPreflightError(
                f"rank zero returned unsorted activation checkpoint layer indices: {field}"
            )
        return indices

    activation_checkpoint_layer_indices = checkpoint_indices("activation_checkpoint_layer_indices")
    hidden_alignment_activation_checkpoint_layer_indices = checkpoint_indices(
        "hidden_alignment_activation_checkpoint_layer_indices"
    )
    dense_transfer_token_checkpoint_layer_indices = checkpoint_indices(
        "dense_transfer_token_checkpoint_layer_indices"
    )
    hidden_alignment_dense_transfer_token_checkpoint_layer_indices = checkpoint_indices(
        "hidden_alignment_dense_transfer_token_checkpoint_layer_indices"
    )
    if set(activation_checkpoint_layer_indices).intersection(
        dense_transfer_token_checkpoint_layer_indices
    ) or set(hidden_alignment_activation_checkpoint_layer_indices).intersection(
        hidden_alignment_dense_transfer_token_checkpoint_layer_indices
    ):
        raise TrainingPreflightError(
            "rank zero returned nested outer/inner activation checkpoint layers"
        )
    if (
        checkpoint_layer_count is not None
        and len(activation_checkpoint_layer_indices) != checkpoint_layer_count
    ):
        raise TrainingPreflightError(
            "rank zero activation checkpoint indices do not match the configured count"
        )
    if (
        hidden_alignment_checkpoint_layer_count is not None
        and len(hidden_alignment_activation_checkpoint_layer_indices)
        != hidden_alignment_checkpoint_layer_count
    ):
        raise TrainingPreflightError(
            "rank zero alignment checkpoint indices do not match the configured count"
        )
    if (
        dense_transfer_checkpoint_layer_count is not None
        and len(dense_transfer_token_checkpoint_layer_indices)
        > dense_transfer_checkpoint_layer_count
    ):
        raise TrainingPreflightError(
            "rank zero dense transfer checkpoint indices exceed the requested count"
        )
    if (
        hidden_alignment_dense_transfer_checkpoint_layer_count is not None
        and len(hidden_alignment_dense_transfer_token_checkpoint_layer_indices)
        > hidden_alignment_dense_transfer_checkpoint_layer_count
    ):
        raise TrainingPreflightError(
            "rank zero alignment dense transfer checkpoint indices exceed the requested count"
        )

    return PreflightReport(
        config_fingerprint=str(value["config_fingerprint"]),
        data_fingerprint=str(value["data_fingerprint"]),
        source_tree_sha256=source_tree_sha256,
        batch=BatchGeometry(
            world_size=int(batch["world_size"]),
            micro_batch_tokens_per_rank=int(batch["micro_batch_tokens_per_rank"]),
            gradient_accumulation_steps=int(batch["gradient_accumulation_steps"]),
            global_batch_tokens=int(batch["global_batch_tokens"]),
        ),
        checked_paths=tuple(str(item) for item in value["checked_paths"]),
        calibration_fingerprints=tuple(
            (str(item[0]), str(item[1])) for item in value.get("calibration_fingerprints", [])
        ),
        data_governance=DataGovernanceStatus(
            lineage_kind=str(governance.get("lineage_kind", "unknown")),
            research_only=governance.get("research_only") is True,
            ready_for_training=governance.get("ready_for_training") is True,
            pending_audits=tuple(raw_pending),
            warning=raw_warning,
        ),
        teacher_cpu_shadow_bytes=int(value.get("teacher_cpu_shadow_bytes", 0)),
        teacher_gpu_stage_bytes=int(value.get("teacher_gpu_stage_bytes", 0)),
        activation_checkpoint_layer_count=checkpoint_layer_count,
        hidden_alignment_activation_checkpoint_layer_count=(
            hidden_alignment_checkpoint_layer_count
        ),
        activation_checkpoint_layer_indices=activation_checkpoint_layer_indices,
        hidden_alignment_activation_checkpoint_layer_indices=(
            hidden_alignment_activation_checkpoint_layer_indices
        ),
        dense_transfer_execution=str(dense_transfer_execution),
        dense_transfer_checkpoint_layer_count=dense_transfer_checkpoint_layer_count,
        hidden_alignment_dense_transfer_checkpoint_layer_count=(
            hidden_alignment_dense_transfer_checkpoint_layer_count
        ),
        dense_transfer_token_checkpoint_layer_indices=(
            dense_transfer_token_checkpoint_layer_indices
        ),
        hidden_alignment_dense_transfer_token_checkpoint_layer_indices=(
            hidden_alignment_dense_transfer_token_checkpoint_layer_indices
        ),
        quality_cooldown_enabled=cooldown_enabled,
        quality_cooldown_start_tokens=(int(cooldown_start) if cooldown_enabled else None),
        quality_cooldown_dataset_fingerprint=(
            str(cooldown_fingerprint) if cooldown_enabled else None
        ),
        quality_cooldown_sequence_count=(int(cooldown_sequence_count) if cooldown_enabled else 0),
        quality_cooldown_token_count=(int(cooldown_token_count) if cooldown_enabled else 0),
        quality_cooldown_selected_shard_ids=tuple(raw_cooldown_shards),
        quality_cooldown_source_mix_token_counts=tuple(cooldown_mix),
        quality_cooldown_source_mix_enabled=cooldown_source_mix_enabled,
        quality_cooldown_source_mix_algorithm=(
            str(cooldown_source_mix_algorithm) if cooldown_source_mix_enabled else None
        ),
        quality_cooldown_source_map_sha256=(
            str(cooldown_source_map_sha256) if cooldown_source_mix_enabled else None
        ),
        quality_cooldown_source_mix_dataset_fingerprint=(
            str(cooldown_source_mix_dataset_fingerprint) if cooldown_source_mix_enabled else None
        ),
        quality_cooldown_source_mix_basis_points=tuple(cooldown_source_mix_basis_points),
        quality_cooldown_source_mix_seed=(
            cooldown_source_mix_seed if cooldown_source_mix_enabled else None
        ),
        quality_cooldown_source_map_payload_json=(cooldown_source_map_payload_json),
        source_mix_enabled=source_mix_enabled,
        source_mix_algorithm=(str(source_mix_algorithm) if source_mix_enabled else None),
        source_map_sha256=(str(source_map_sha256) if source_mix_enabled else None),
        source_mix_dataset_fingerprint=(
            str(source_mix_dataset_fingerprint) if source_mix_enabled else None
        ),
        source_mix_basis_points=tuple(source_mix_basis_points),
        source_mix_lineage_basis_points=tuple(source_mix_lineage_basis_points),
        source_mix_effective_basis_points=tuple(source_mix_effective_basis_points),
        source_mix_weight_override=(source_mix_weight_override if source_mix_enabled else False),
        source_mix_seed=(source_mix_seed if source_mix_enabled else None),
        source_map_payload_json=source_map_payload_json,
    )


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("preflight rendezvous closed early")
        chunks.extend(chunk)
    return bytes(chunks)


def _send_preflight_payload(connection: socket.socket, payload: Mapping[str, object]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    connection.sendall(struct.pack("!Q", len(encoded)) + encoded)


def _receive_preflight_payload(connection: socket.socket) -> dict[str, object]:
    size = struct.unpack("!Q", _recv_exact(connection, 8))[0]
    if size <= 0 or size > 16 * 1024 * 1024:
        raise ConnectionError("preflight rendezvous payload size is invalid")
    value = json.loads(_recv_exact(connection, size))
    if not isinstance(value, dict):
        raise ConnectionError("preflight rendezvous payload is not an object")
    return value


def _rendezvous_path(config: TrainConfig) -> Path:
    identity = ":".join(
        (
            os.environ.get("MASTER_ADDR", "localhost"),
            os.environ.get("MASTER_PORT", "single"),
            os.environ.get("TORCHELASTIC_RUN_ID", "static"),
            os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"),
        )
    )
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return Path(config.checkpoint.output_dir) / f".preflight-{suffix}.json"


def _canonical_config_digest(config: TrainConfig) -> str:
    encoded = json.dumps(
        config.canonical_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _node_identity() -> str:
    for name in ("GROUP_RANK", "NODE_RANK"):
        if name in os.environ:
            return f"{name.lower()}:{os.environ[name]}"
    return f"hostname:{socket.gethostname()}"


def _listener_family(host: str) -> socket.AddressFamily:
    """Choose a listener family compatible with the published MASTER_ADDR."""

    try:
        for family, socktype, _protocol, _canonical, _address in socket.getaddrinfo(
            host,
            None,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        ):
            if socktype == socket.SOCK_STREAM and family in {socket.AF_INET, socket.AF_INET6}:
                return family
    except socket.gaierror:
        pass
    return socket.AF_INET


def _validate_peer_preflight_requests(
    *,
    config: TrainConfig,
    report: PreflightReport,
    world_size: int,
    rank_zero_node: str,
    requests: Mapping[int, Mapping[str, object]],
) -> None:
    """Require identical configs on every rank and a full scan on every node."""

    expected_ranks = set(range(1, world_size))
    if set(requests) != expected_ranks:
        raise TrainingPreflightError(
            "preflight peer rank set is incomplete: "
            f"missing={sorted(expected_ranks - set(requests))}, "
            f"extra={sorted(set(requests) - expected_ranks)}"
        )
    config_digest = _canonical_config_digest(config)
    reference_report = _preflight_report_payload(report)
    node_reports: dict[str, object] = {rank_zero_node: reference_report}
    observed_nodes = {rank_zero_node}
    for rank, request in requests.items():
        if request.get("config_digest") != config_digest:
            raise TrainingPreflightError(
                f"rank {rank} loaded a different complete training configuration"
            )
        node = str(request.get("node_id", ""))
        if not node:
            raise TrainingPreflightError(f"rank {rank} did not identify its node")
        observed_nodes.add(node)
        local_error = request.get("local_error")
        if local_error is not None:
            raise TrainingPreflightError(
                f"node-local preflight failed on rank {rank}: {local_error}"
            )
        if int(request.get("local_rank", -1)) == 0:
            local_report = request.get("local_report")
            if local_report is None:
                raise TrainingPreflightError(
                    f"node leader rank {rank} returned no local preflight report"
                )
            if node in node_reports:
                raise TrainingPreflightError(f"multiple preflight leaders claimed node {node}")
            node_reports[node] = local_report
    missing_nodes = observed_nodes - set(node_reports)
    if missing_nodes:
        raise TrainingPreflightError(
            f"no LOCAL_RANK=0 process validated node(s): {sorted(missing_nodes)}"
        )
    for node, local_report in node_reports.items():
        if local_report != reference_report:
            raise TrainingPreflightError(
                f"node {node} has different local artifacts or preflight results"
            )


def run_coordinated_training_preflight(config: TrainConfig) -> PreflightReport:
    """Run full hashing once per node and coordinate it before CUDA startup."""

    from .utils import atomic_write_json

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size <= 1:
        return run_training_preflight(config, world_size=1)
    if not 0 <= rank < world_size:
        raise TrainingPreflightError("RANK must be within WORLD_SIZE")
    rendezvous = _rendezvous_path(config)
    rendezvous.parent.mkdir(parents=True, exist_ok=True)
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    node_id = _node_identity()
    config_digest = _canonical_config_digest(config)

    if rank == 0:
        nonce = secrets.token_hex(32)
        family = _listener_family(master_addr)
        server = socket.socket(family, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if family == socket.AF_INET6:
            with contextlib.suppress(OSError):
                server.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            server.bind(("::", 0))
        else:
            server.bind(("0.0.0.0", 0))
        server.listen(world_size)
        port = int(server.getsockname()[1])
        atomic_write_json(
            rendezvous,
            {
                "schema_version": 1,
                "nonce": nonce,
                "host": master_addr,
                "port": port,
                "world_size": world_size,
                "created_unix_ns": time.time_ns(),
            },
        )
        local_error: Exception | None = None
        coordinated_error: Exception | None = None
        report: PreflightReport | None = None
        try:
            report = run_training_preflight(config, world_size=world_size)
        except Exception as exc:
            local_error = exc
        peers: list[socket.socket] = []
        requests: dict[int, Mapping[str, object]] = {}
        try:
            peer_deadline = time.monotonic() + float(
                os.environ.get(
                    "TWEN_PREFLIGHT_PEER_TIMEOUT",
                    os.environ.get("TWEN_PREFLIGHT_WAIT_SECONDS", "86400"),
                )
            )
            while len(requests) < world_size - 1:
                remaining = peer_deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for all preflight peers")
                server.settimeout(remaining)
                connection, _ = server.accept()
                keep = False
                try:
                    connection.settimeout(min(60.0, max(1.0, remaining)))
                    request = _receive_preflight_payload(connection)
                    if request.get("nonce") != nonce:
                        continue
                    peer_rank = int(request.get("rank", -1))
                    if not 1 <= peer_rank < world_size or peer_rank in requests:
                        continue
                    connection.settimeout(None)
                    peers.append(connection)
                    requests[peer_rank] = request
                    keep = True
                finally:
                    if not keep:
                        connection.close()
            if local_error is None:
                if report is None:
                    raise TrainingPreflightError("rank-zero preflight produced no report")
                _validate_peer_preflight_requests(
                    config=config,
                    report=report,
                    world_size=world_size,
                    rank_zero_node=node_id,
                    requests=requests,
                )
        except Exception as exc:
            coordinated_error = exc

        effective_error = local_error or coordinated_error
        response: dict[str, object]
        if effective_error is None:
            assert report is not None
            response = {
                "ok": True,
                "nonce": nonce,
                "config_digest": config_digest,
                "report": _preflight_report_payload(report),
            }
        else:
            response = {
                "ok": False,
                "nonce": nonce,
                "config_digest": config_digest,
                "error_type": type(effective_error).__name__,
                "error": str(effective_error),
            }
        broadcast_errors: list[str] = []
        try:
            for connection in peers:
                try:
                    connection.settimeout(30.0)
                    _send_preflight_payload(connection, response)
                except (OSError, TimeoutError) as exc:
                    # One worker may die after submitting its request. Continue
                    # notifying every healthy peer instead of stranding them
                    # behind the first broken socket.
                    broadcast_errors.append(f"{type(exc).__name__}: {exc}")
                finally:
                    connection.close()
        finally:
            for connection in peers:
                connection.close()
            server.close()
            rendezvous.unlink(missing_ok=True)
        if local_error is not None:
            raise local_error
        if coordinated_error is not None:
            raise coordinated_error
        if broadcast_errors:
            raise TrainingPreflightError(
                "failed to deliver preflight result to one or more ranks: "
                + "; ".join(broadcast_errors)
            )
        if report is None:
            raise TrainingPreflightError("rank-zero preflight produced no report")
        return report

    deadline = time.monotonic() + float(os.environ.get("TWEN_PREFLIGHT_WAIT_SECONDS", "86400"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_report: PreflightReport | None = None
    local_error: str | None = None
    if local_rank == 0:
        try:
            local_report = run_training_preflight(config, world_size=world_size)
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
    request: dict[str, object] = {
        "nonce": None,
        "rank": rank,
        "local_rank": local_rank,
        "node_id": node_id,
        "config_digest": config_digest,
        "local_report": (
            _preflight_report_payload(local_report) if local_report is not None else None
        ),
        "local_error": local_error,
    }
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            status = json.loads(rendezvous.read_text(encoding="utf-8"))
            if status.get("schema_version") != 1 or int(status.get("world_size", -1)) != world_size:
                raise ValueError("preflight rendezvous identity mismatch")
            nonce = str(status["nonce"])
            host = str(status.get("host") or master_addr)
            port = int(status["port"])
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            with socket.create_connection((host, port), timeout=min(3.0, remaining)) as connection:
                request["nonce"] = nonce
                _send_preflight_payload(connection, request)
                connection.settimeout(remaining)
                response = _receive_preflight_payload(connection)
            if response.get("nonce") != nonce:
                raise ConnectionError("preflight response nonce mismatch")
            if response.get("config_digest") != config_digest:
                raise TrainingPreflightError(
                    "rank-zero preflight used a different complete configuration"
                )
            if not response.get("ok"):
                raise TrainingPreflightError(
                    "rank-zero preflight failed: "
                    f"{response.get('error_type')}: {response.get('error')}"
                )
            return _preflight_report_from_payload(response.get("report"))
        except TrainingPreflightError:
            raise
        except (ConnectionError, OSError, KeyError, TypeError, ValueError) as exc:
            last_error = exc
            time.sleep(0.1)
    raise TrainingPreflightError(
        f"timed out waiting for rank-zero preflight rendezvous: {last_error}"
    )
