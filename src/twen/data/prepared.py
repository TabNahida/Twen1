"""Idempotent fixed-length text preparation for teacher KD generation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ..io.download import sha256_file
from ..progress import TaskProgress
from ..utils import atomic_write_json
from .shards import ShardTransaction, is_shard_complete

PREPARED_SCHEMA_VERSION = 2
PREPARED_TENSORS = "tokens.safetensors"
PREPARED_SHARD_MANIFEST = "prepared_manifest.json"
# Schema-v2 corpora already emitted by the original explicit-input generator
# bind this exact identity.  Every new artifact derived from an authenticated
# extracted manifest uses the current source hash below, including research-only
# preparation while governance audits are still pending.  This preserves the
# frozen v1 token/KD artifacts without giving new extracted data a legacy
# generator identity.
PREPARED_GENERATOR_SOURCE_SHA256 = (
    "9dce6baac242fac4881d28566c613fce720ce0294a76553e23003f6b43db6a28"
)
AUDITED_PREPARED_GENERATOR_SOURCE_SHA256 = sha256_file(Path(__file__))
_SAFE_SHARD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_EXTRACTED_SCHEMA_VERSION = 1
_EXTRACTED_CORPUS_KIND = "twen_extracted_base_jsonl_corpus"
_EXTRACTED_COMPLETE_KIND = "twen_extracted_base_jsonl_complete"
_EXTRACTED_IDENTITY_KEYS = (
    "recipe_id",
    "recipe_sha256",
    "resolved_source_lock_sha256",
    "tokenizer_manifest_sha256",
    "extractor_source_sha256",
    "profile",
    "sources",
    "train_files",
    "validation_files",
    "attribution_files",
    "file_lists",
)
_EXTRACTED_CONTRACT_IDENTITY_KEYS = (
    "source_map",
    "source_mix",
    "format_audit",
    "license_audit",
    "materialization_audit",
)
_SOURCE_MAP_ALGORITHM = "authenticated-extracted-output-map-v1"
_SOURCE_MIX_ALGORITHM = "token-deficit-corrected-source-mix-bp-v2"
_EXTRACTED_FILE_LIST_NAMES = {
    "train": "train-files.txt",
    "validation": "validation-files.txt",
    "attribution": "attribution-files.txt",
}


def _normalized_sha256(value: str, field: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be a 64-digit SHA256")
    return normalized


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _required_file_size(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _safe_extracted_relative_path(value: object, field: str) -> PurePosixPath:
    text = _required_string(value, field)
    relative = PurePosixPath(text)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() == ".":
        raise ValueError(f"unsafe {field}: {text!r}")
    return relative


def _canonical_json_copy(value: Mapping[str, object]) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        copied = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError("prepared lineage must contain only JSON values") from error
    if not isinstance(copied, dict):  # pragma: no cover - Mapping always encodes as object
        raise ValueError("prepared lineage must be an object")
    return copied


def _normalized_prepared_lineage(
    value: Mapping[str, object] | None,
) -> dict[str, Any] | None:
    """Validate and detach schema-v2 source lineage.

    ``None`` is accepted only for compatibility with manifests emitted by the
    original schema-v2 implementation.  Every newly prepared corpus records an
    explicit lineage mode.
    """

    if value is None:
        return None
    result = _canonical_json_copy(value)
    kind = result.get("kind")
    if kind == "explicit_unreviewed":
        if result.get("research_only") is not True:
            raise ValueError("explicit_unreviewed prepared lineage must be research_only")
        pending = result.get("pending_audits")
        if (
            not isinstance(pending, list)
            or not pending
            or not all(isinstance(item, str) and item for item in pending)
        ):
            raise ValueError("explicit_unreviewed lineage must name pending audits")
        return result
    if kind != "authenticated_extracted_corpus":
        raise ValueError(f"unsupported prepared lineage kind: {kind!r}")

    _required_string(result.get("extracted_manifest_path"), "extracted_manifest_path")
    for field in (
        "extracted_manifest_sha256",
        "corpus_fingerprint",
        "recipe_sha256",
        "resolved_source_lock_sha256",
        "tokenizer_manifest_sha256",
        "extractor_source_sha256",
    ):
        _normalized_sha256(_required_string(result.get(field), field), field)
    _required_string(result.get("recipe_id"), "recipe_id")
    _required_string(result.get("profile"), "profile")
    if result.get("role") not in {"train", "validation"}:
        raise ValueError("authenticated extracted lineage role must be train or validation")
    if result.get("ready_for_data_prepare") is not True:
        raise ValueError("authenticated extracted corpus is not ready for data prepare")
    if not isinstance(result.get("ready_for_training"), bool):
        raise ValueError("ready_for_training must be boolean")
    if not isinstance(result.get("research_only"), bool):
        raise ValueError("research_only must be boolean")
    if result["research_only"] == result["ready_for_training"]:
        raise ValueError("research_only must be the inverse of ready_for_training")
    audits = result.get("audits")
    if not isinstance(audits, dict) or not all(
        isinstance(name, str) and name and isinstance(status, str) and status
        for name, status in audits.items()
    ):
        raise ValueError("authenticated extracted lineage audits must be string statuses")
    pending = result.get("pending_audits")
    if not isinstance(pending, list) or not all(isinstance(item, str) and item for item in pending):
        raise ValueError("authenticated extracted lineage pending_audits must be strings")
    expected_pending = sorted(
        name for name, status in audits.items() if status.lower().startswith("pending")
    )
    if pending != expected_pending:
        raise ValueError("prepared pending_audits differ from extracted audit statuses")
    audit_attestation = result.get("audit_attestation")
    if audit_attestation is not None:
        if not isinstance(audit_attestation, dict):
            raise ValueError("audit_attestation lineage must be an object")
        _required_string(audit_attestation.get("path"), "audit_attestation.path")
        _normalized_sha256(
            _required_string(audit_attestation.get("sha256"), "audit_attestation.sha256"),
            "audit_attestation.sha256",
        )
        _normalized_sha256(
            _required_string(
                audit_attestation.get("attestation_fingerprint"),
                "audit_attestation.attestation_fingerprint",
            ),
            "audit_attestation.attestation_fingerprint",
        )
        expected_binding = "candidate" if result.get("role") == "train" else "frozen_validation"
        if audit_attestation.get("bound_as") != expected_binding:
            raise ValueError("audit_attestation role binding mismatch")
        if audit_attestation.get("ready_for_training") is not result["ready_for_training"]:
            raise ValueError("audit_attestation readiness differs from prepared lineage")
        gates = audit_attestation.get("gates")
        if not isinstance(gates, dict) or not gates:
            raise ValueError("audit_attestation gates are missing")
    file_list = result.get("file_list")
    if not isinstance(file_list, dict):
        raise ValueError("authenticated extracted lineage file_list must be an object")
    _safe_extracted_relative_path(file_list.get("path"), "file_list.path")
    _required_file_size(file_list.get("size"), "file_list.size")
    _normalized_sha256(
        _required_string(file_list.get("sha256"), "file_list.sha256"),
        "file_list.sha256",
    )
    source_files = result.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("authenticated extracted lineage source_files cannot be empty")
    seen: set[str] = set()
    for index, item in enumerate(source_files):
        if not isinstance(item, dict):
            raise ValueError("authenticated extracted source_files entries must be objects")
        relative = _safe_extracted_relative_path(
            item.get("path"), f"source_files[{index}].path"
        ).as_posix()
        if relative in seen:
            raise ValueError(f"duplicate authenticated extracted source path: {relative}")
        seen.add(relative)
        _required_file_size(item.get("size"), f"source_files[{index}].size")
        _normalized_sha256(
            _required_string(item.get("sha256"), f"source_files[{index}].sha256"),
            f"source_files[{index}].sha256",
        )
    data_contract = result.get("data_contract")
    if data_contract is not None:
        if not isinstance(data_contract, dict):
            raise ValueError("prepared data_contract lineage must be an object")
        expected_contract_fields = {
            "schema_version",
            *_EXTRACTED_CONTRACT_IDENTITY_KEYS,
            "contract_fingerprint",
        }
        if set(data_contract) != expected_contract_fields:
            raise ValueError("prepared data_contract lineage fields differ")
        if data_contract.get("schema_version") != 1:
            raise ValueError("unsupported prepared data_contract lineage")
        for field in _EXTRACTED_CONTRACT_IDENTITY_KEYS:
            if not isinstance(data_contract.get(field), dict):
                raise ValueError(f"prepared data_contract.{field} must be an object")
        if data_contract["source_map"].get("algorithm") != _SOURCE_MAP_ALGORITHM:
            raise ValueError("prepared data_contract source_map algorithm differs")
        if data_contract["source_mix"].get("algorithm") != _SOURCE_MIX_ALGORITHM:
            raise ValueError("prepared data_contract source_mix algorithm differs")
        unsigned_contract = {
            field: data_contract[field]
            for field in (
                "schema_version",
                *_EXTRACTED_CONTRACT_IDENTITY_KEYS,
            )
        }
        if data_contract.get("contract_fingerprint") != _canonical_sha256(unsigned_contract):
            raise ValueError("prepared data_contract lineage fingerprint mismatch")
    quality = result.get("quality_cooldown")
    if quality is not None:
        if not isinstance(quality, dict):
            raise ValueError("quality_cooldown lineage must be an object")
        if (
            type(quality.get("schema_version")) is not int
            or quality.get("schema_version") != 1
            or quality.get("kind") != "authenticated_prepared_kd_subset_view"
            or quality.get("eligible") is not True
        ):
            raise ValueError("unsupported quality_cooldown lineage contract")
        if result.get("ready_for_training") is not True:
            raise ValueError("quality_cooldown lineage must be training-ready")
        expected_quality_fields = {
            "schema_version",
            "kind",
            "eligible",
            "parent_prepared_manifest_sha256",
            "parent_kd_manifest_sha256",
            "parent_dataset_fingerprint",
            "selection_policy_id",
            "selection_policy_sha256",
            "selection_basis",
            "required_cooldown_tokens",
            "ordered_parent_shard_ids",
            "shard_source_ids",
            "source_mix_token_counts",
        }
        if set(quality) != expected_quality_fields:
            raise ValueError("quality_cooldown lineage fields differ from the locked schema")
        for field in (
            "parent_prepared_manifest_sha256",
            "parent_kd_manifest_sha256",
            "parent_dataset_fingerprint",
            "selection_policy_sha256",
        ):
            _normalized_sha256(
                _required_string(quality.get(field), f"quality_cooldown.{field}"),
                f"quality_cooldown.{field}",
            )
        _required_string(
            quality.get("selection_policy_id"),
            "quality_cooldown.selection_policy_id",
        )
        _required_string(
            quality.get("selection_basis"),
            "quality_cooldown.selection_basis",
        )
        required_tokens = quality.get("required_cooldown_tokens")
        if (
            isinstance(required_tokens, bool)
            or not isinstance(required_tokens, int)
            or required_tokens <= 0
        ):
            raise ValueError("quality_cooldown.required_cooldown_tokens must be a positive integer")
        ordered = quality.get("ordered_parent_shard_ids")
        if (
            not isinstance(ordered, list)
            or not ordered
            or not all(isinstance(item, str) and item for item in ordered)
            or len(set(ordered)) != len(ordered)
        ):
            raise ValueError("quality_cooldown ordered shard IDs are invalid")
        shard_sources = quality.get("shard_source_ids")
        source_mix = quality.get("source_mix_token_counts")
        if (
            not isinstance(shard_sources, dict)
            or set(shard_sources) != set(ordered)
            or not all(
                isinstance(shard_id, str) and isinstance(source_id, str) and source_id
                for shard_id, source_id in shard_sources.items()
            )
        ):
            raise ValueError("quality_cooldown shard source IDs are invalid")
        if (
            not isinstance(source_mix, dict)
            or not source_mix
            or not all(
                isinstance(source_id, str)
                and source_id
                and isinstance(tokens, int)
                and not isinstance(tokens, bool)
                and tokens > 0
                for source_id, tokens in source_mix.items()
            )
        ):
            raise ValueError("quality_cooldown source mix is invalid")
    return result


def _explicit_unreviewed_lineage() -> dict[str, object]:
    return {
        "kind": "explicit_unreviewed",
        "research_only": True,
        "pending_audits": ["extracted_corpus_authentication"],
    }


@dataclass(frozen=True, slots=True)
class PreparedShardEntry:
    shard_id: str
    path: str
    source_path: str
    source_sha256: str
    tensors_sha256: str
    sequence_count: int
    token_count: int
    global_sample_start: int
    global_sample_end: int
    global_token_start: int
    global_token_end: int

    def __post_init__(self) -> None:
        if not _SAFE_SHARD_ID.fullmatch(self.shard_id) or self.shard_id in {".", ".."}:
            raise ValueError(f"unsafe prepared shard_id: {self.shard_id!r}")
        relative = PurePosixPath(self.path)
        if not self.path or self.path == "." or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe prepared shard path: {self.path!r}")
        if not self.source_path:
            raise ValueError("prepared source_path is required")
        object.__setattr__(
            self,
            "source_sha256",
            _normalized_sha256(self.source_sha256, "source_sha256"),
        )
        object.__setattr__(
            self,
            "tensors_sha256",
            _normalized_sha256(self.tensors_sha256, "tensors_sha256"),
        )
        if self.sequence_count <= 0 or self.token_count <= 0:
            raise ValueError("prepared shard sequence/token counts must be positive")
        if (
            self.global_sample_start < 0
            or self.global_sample_end - self.global_sample_start != self.sequence_count
        ):
            raise ValueError("prepared shard sample range/count mismatch")
        if (
            self.global_token_start < 0
            or self.global_token_end - self.global_token_start != self.token_count
        ):
            raise ValueError("prepared shard token range/count mismatch")


@dataclass(frozen=True, slots=True)
class PreparedCorpusManifest:
    dataset_fingerprint: str
    pipeline_fingerprint: str
    generator_source_sha256: str
    tokenizer_sha256: str
    sequence_length: int
    text_field: str
    shards: tuple[PreparedShardEntry, ...]
    lineage: Mapping[str, object] | None = None
    schema_version: int = PREPARED_SCHEMA_VERSION
    kind: str = "twen_prepared_text"

    def __post_init__(self) -> None:
        if self.schema_version != PREPARED_SCHEMA_VERSION or self.kind != "twen_prepared_text":
            raise ValueError("unsupported prepared corpus manifest")
        object.__setattr__(
            self,
            "dataset_fingerprint",
            _normalized_sha256(self.dataset_fingerprint, "dataset_fingerprint"),
        )
        object.__setattr__(
            self,
            "pipeline_fingerprint",
            _normalized_sha256(self.pipeline_fingerprint, "pipeline_fingerprint"),
        )
        object.__setattr__(
            self,
            "generator_source_sha256",
            _normalized_sha256(
                self.generator_source_sha256,
                "generator_source_sha256",
            ),
        )
        object.__setattr__(
            self,
            "tokenizer_sha256",
            _normalized_sha256(self.tokenizer_sha256, "tokenizer_sha256"),
        )
        object.__setattr__(self, "lineage", _normalized_prepared_lineage(self.lineage))
        if self.sequence_length <= 0:
            raise ValueError("prepared sequence_length must be positive")
        if not self.text_field:
            raise ValueError("prepared text_field is required")
        if not self.shards:
            raise ValueError("prepared corpus must contain at least one shard")
        if len({entry.shard_id for entry in self.shards}) != len(self.shards):
            raise ValueError("prepared shard_id values must be unique")
        if len({entry.path for entry in self.shards}) != len(self.shards):
            raise ValueError("prepared shard paths must be unique")
        next_sample = 0
        next_token = 0
        for entry in self.shards:
            if entry.global_sample_start != next_sample:
                raise ValueError("prepared sample ranges must be contiguous from zero")
            if entry.global_token_start != next_token:
                raise ValueError("prepared token ranges must be contiguous from zero")
            if entry.token_count > entry.sequence_count * self.sequence_length:
                raise ValueError("prepared shard token_count exceeds padded capacity")
            next_sample = entry.global_sample_end
            next_token = entry.global_token_end

    @property
    def sequence_count(self) -> int:
        return sum(item.sequence_count for item in self.shards)

    @property
    def token_count(self) -> int:
        return sum(item.token_count for item in self.shards)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.lineage is None:
            # Preserve compatibility with the original v2 serialization.  A
            # missing lineage is accepted for reading only; all new builds set
            # either authenticated_extracted_corpus or explicit_unreviewed.
            value.pop("lineage")
        value["sequence_count"] = self.sequence_count
        value["token_count"] = self.token_count
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PreparedCorpusManifest:
        if value.get("schema_version") != PREPARED_SCHEMA_VERSION:
            raise ValueError(
                "unsupported prepared corpus manifest; schema v1 lacks a safely "
                "recomputable generator/pipeline identity and must be rebuilt"
            )
        if value.get("kind") != "twen_prepared_text":
            raise ValueError("unsupported prepared corpus manifest kind")
        result = cls(
            dataset_fingerprint=str(value["dataset_fingerprint"]),
            pipeline_fingerprint=str(value["pipeline_fingerprint"]),
            generator_source_sha256=str(value["generator_source_sha256"]),
            tokenizer_sha256=str(value["tokenizer_sha256"]),
            sequence_length=int(value["sequence_length"]),
            text_field=str(value["text_field"]),
            shards=tuple(PreparedShardEntry(**item) for item in value["shards"]),
            lineage=value.get("lineage"),
        )
        if int(value.get("sequence_count", -1)) != result.sequence_count:
            raise ValueError("prepared sequence count mismatch")
        if int(value.get("token_count", -1)) != result.token_count:
            raise ValueError("prepared token count mismatch")
        return result


def read_prepared_manifest(path: str | Path) -> PreparedCorpusManifest:
    return PreparedCorpusManifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _resolved_extracted_file(root: Path, relative: PurePosixPath, field: str) -> Path:
    candidate = (root / relative.as_posix()).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field} escapes extracted corpus root: {relative}") from error
    if candidate == root:
        raise ValueError(f"{field} resolves to extracted corpus root: {relative}")
    return candidate


def _validated_extracted_data_contract(
    value: Mapping[str, object],
    inventories: Mapping[str, list[dict[str, object]]],
) -> dict[str, object] | None:
    presence = [name in value for name in _EXTRACTED_CONTRACT_IDENTITY_KEYS]
    if not any(presence):
        return None
    if not all(presence):
        raise ValueError("extracted corpus has a partial data contract audit")
    contract = {
        field: _canonical_json_copy(value[field])
        for field in _EXTRACTED_CONTRACT_IDENTITY_KEYS
        if isinstance(value[field], Mapping)
    }
    if set(contract) != set(_EXTRACTED_CONTRACT_IDENTITY_KEYS):
        raise ValueError("extracted corpus data contract entries must be objects")

    source_map = contract["source_map"]
    if (
        source_map.get("schema_version") != 1
        or source_map.get("algorithm") != _SOURCE_MAP_ALGORITHM
    ):
        raise ValueError("unsupported extracted source_map contract")
    source_map_unsigned = {
        key: source_map.get(key) for key in ("schema_version", "algorithm", "roles")
    }
    if source_map.get("fingerprint") != _canonical_sha256(source_map_unsigned):
        raise ValueError("extracted source_map contract fingerprint mismatch")
    roles = source_map.get("roles")
    if not isinstance(roles, dict) or set(roles) != {"train", "validation"}:
        raise ValueError("extracted source_map roles are invalid")
    source_ids: set[str] = set()
    for role in ("train", "validation"):
        raw_outputs = roles.get(role)
        if not isinstance(raw_outputs, list):
            raise ValueError(f"extracted source_map {role} inventory is invalid")
        normalized_outputs: list[dict[str, object]] = []
        for index, item in enumerate(raw_outputs):
            if not isinstance(item, dict):
                raise ValueError(f"extracted source_map {role}[{index}] must be an object")
            source_ids.add(
                _required_string(
                    item.get("source_id"),
                    f"source_map.{role}[{index}].source_id",
                )
            )
            normalized_outputs.append(
                {
                    "path": _safe_extracted_relative_path(
                        item.get("path"),
                        f"source_map.{role}[{index}].path",
                    ).as_posix(),
                    "size": _required_file_size(
                        item.get("size"),
                        f"source_map.{role}[{index}].size",
                    ),
                    "sha256": _normalized_sha256(
                        _required_string(
                            item.get("sha256"),
                            f"source_map.{role}[{index}].sha256",
                        ),
                        f"source_map.{role}[{index}].sha256",
                    ),
                }
            )
        if normalized_outputs != inventories[role]:
            raise ValueError(f"extracted source_map {role} differs from role inventory")

    source_mix = contract["source_mix"]
    if (
        source_mix.get("schema_version") != 1
        or source_mix.get("algorithm") != _SOURCE_MIX_ALGORITHM
        or source_mix.get("unit") != "valid_tokens"
    ):
        raise ValueError("unsupported extracted source_mix contract")
    source_mix_unsigned = {
        key: source_mix.get(key)
        for key in (
            "schema_version",
            "algorithm",
            "unit",
            "basis_points_total",
            "profile",
            "sources",
        )
    }
    if source_mix.get("fingerprint") != _canonical_sha256(source_mix_unsigned):
        raise ValueError("extracted source_mix contract fingerprint mismatch")
    raw_mix_sources = source_mix.get("sources")
    if not isinstance(raw_mix_sources, list):
        raise ValueError("extracted source_mix source inventory is invalid")
    mix_ids = {
        _required_string(item.get("source_id"), "source_mix.source_id")
        for item in raw_mix_sources
        if isinstance(item, dict)
    }
    if len(mix_ids) != len(raw_mix_sources) or mix_ids != source_ids:
        raise ValueError("extracted source_mix/source_map source sets differ")
    weights = [item.get("mix_basis_points") for item in raw_mix_sources]
    if any(
        isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0 for weight in weights
    ):
        raise ValueError("extracted source_mix weights must be positive integers")
    if (
        sum(int(weight) for weight in weights) != source_mix.get("basis_points_total")
        or source_mix.get("basis_points_total") != 10_000
    ):
        raise ValueError("extracted source_mix must total 10,000 basis points")
    for audit_name in ("format_audit", "license_audit", "materialization_audit"):
        if contract[audit_name].get("complete") is not True:
            raise ValueError(f"extracted {audit_name} is incomplete")
    return contract


def _authenticate_extracted_prepare_inputs(
    manifest_path: str | Path,
    *,
    role: str,
    tokenizer_sha256: str,
    allow_pending_research_audits: bool,
    audit_attestation: str | Path | None = None,
) -> tuple[tuple[tuple[Path, str], ...], dict[str, Any]]:
    """Authenticate an extracted corpus and resolve one exact role inventory.

    This intentionally lives here rather than importing the extractor's
    validator: changing preparation must not change the source digest pinned by
    an extraction already in flight.  The checks mirror the extractor contract
    and additionally reject symlinks escaping the corpus root.
    """

    if role not in {"train", "validation"}:
        raise ValueError("--role must be train or validation with --extracted-manifest")
    expected_tokenizer_sha = _normalized_sha256(tokenizer_sha256, "tokenizer_sha256")
    path = Path(manifest_path).resolve()
    root = path.parent.resolve()
    invalidated = root / "INVALIDATED.json"
    if invalidated.exists():
        raise ValueError(f"extracted corpus is explicitly invalidated: {invalidated}")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read extracted corpus manifest: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("extracted corpus manifest must be an object")
    if value.get("schema_version") != _EXTRACTED_SCHEMA_VERSION:
        raise ValueError("unsupported extracted corpus manifest schema")
    if value.get("kind") != _EXTRACTED_CORPUS_KIND:
        raise ValueError("unexpected extracted corpus manifest kind")

    inventories: dict[str, list[dict[str, object]]] = {}
    resolved_inventories: dict[str, list[tuple[Path, dict[str, object]]]] = {}
    seen_outputs: set[str] = set()
    for inventory_role in _EXTRACTED_FILE_LIST_NAMES:
        raw_inventory = value.get(f"{inventory_role}_files")
        if not isinstance(raw_inventory, list):
            raise ValueError(f"extracted {inventory_role} file inventory must be a list")
        inventory: list[dict[str, object]] = []
        resolved_items: list[tuple[Path, dict[str, object]]] = []
        for index, raw_item in enumerate(raw_inventory):
            if not isinstance(raw_item, dict):
                raise ValueError(f"extracted {inventory_role} file entry must be an object")
            field = f"{inventory_role}_files[{index}]"
            relative = _safe_extracted_relative_path(raw_item.get("path"), f"{field}.path")
            relative_text = relative.as_posix()
            if relative_text in seen_outputs:
                raise ValueError(f"duplicate extracted output path: {relative_text}")
            seen_outputs.add(relative_text)
            size = _required_file_size(raw_item.get("size"), f"{field}.size")
            digest = _normalized_sha256(
                _required_string(raw_item.get("sha256"), f"{field}.sha256"),
                f"{field}.sha256",
            )
            output = _resolved_extracted_file(root, relative, field)
            if not output.is_file():
                raise ValueError(f"missing extracted output: {output}")
            if output.stat().st_size != size:
                raise ValueError(f"size mismatch for extracted output: {output}")
            if sha256_file(output) != digest:
                raise ValueError(f"SHA256 mismatch for extracted output: {output}")
            item = {"path": relative_text, "size": size, "sha256": digest}
            inventory.append(item)
            resolved_items.append((output, item))
        inventories[inventory_role] = inventory
        resolved_inventories[inventory_role] = resolved_items

    if not inventories[role]:
        raise ValueError(f"extracted corpus has no {role} JSONL files")
    file_lists = value.get("file_lists")
    if not isinstance(file_lists, dict) or set(file_lists) != set(_EXTRACTED_FILE_LIST_NAMES):
        raise ValueError("extracted corpus file-list inventory is invalid")
    normalized_file_lists: dict[str, dict[str, object]] = {}
    for inventory_role, expected_name in _EXTRACTED_FILE_LIST_NAMES.items():
        raw_entry = file_lists.get(inventory_role)
        if not isinstance(raw_entry, dict):
            raise ValueError(f"extracted {inventory_role} file-list entry is invalid")
        relative = _safe_extracted_relative_path(
            raw_entry.get("path"), f"file_lists.{inventory_role}.path"
        )
        if relative.as_posix() != expected_name:
            raise ValueError(f"unexpected extracted {inventory_role} file-list path: {relative}")
        size = _required_file_size(raw_entry.get("size"), f"file_lists.{inventory_role}.size")
        digest = _normalized_sha256(
            _required_string(raw_entry.get("sha256"), f"file_lists.{inventory_role}.sha256"),
            f"file_lists.{inventory_role}.sha256",
        )
        sidecar = _resolved_extracted_file(root, relative, f"{inventory_role} file list")
        try:
            payload = sidecar.read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError(f"missing extracted {inventory_role} file list: {sidecar}") from error
        expected_payload = "".join(f"{item['path']}\n" for item in inventories[inventory_role])
        if payload != expected_payload:
            raise ValueError(
                f"extracted {inventory_role} file list differs from manifest inventory"
            )
        if sidecar.stat().st_size != size:
            raise ValueError(f"extracted {inventory_role} file-list size mismatch")
        if sha256_file(sidecar) != digest:
            raise ValueError(f"extracted {inventory_role} file-list SHA256 mismatch")
        normalized_file_lists[inventory_role] = {
            "path": relative.as_posix(),
            "size": size,
            "sha256": digest,
        }

    data_contract = _validated_extracted_data_contract(value, inventories)
    identity_keys = _EXTRACTED_IDENTITY_KEYS
    if data_contract is not None:
        identity_keys = (*identity_keys, *_EXTRACTED_CONTRACT_IDENTITY_KEYS)
    identity = {name: value.get(name) for name in identity_keys}
    actual_fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if value.get("corpus_fingerprint") != actual_fingerprint:
        raise ValueError("extracted corpus identity fingerprint mismatch")
    manifest_sha = hashlib.sha256(raw).hexdigest()
    marker_path = root / "COMPLETE"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"extracted corpus has no valid COMPLETE marker: {root}") from error
    if (
        not isinstance(marker, dict)
        or marker.get("schema_version") != _EXTRACTED_SCHEMA_VERSION
        or marker.get("kind") != _EXTRACTED_COMPLETE_KIND
        or marker.get("manifest") != path.name
        or marker.get("manifest_sha256") != manifest_sha
        or marker.get("corpus_fingerprint") != actual_fingerprint
        or marker.get("file_lists") != file_lists
        or marker.get("ready_for_training") != value.get("ready_for_training")
    ):
        raise ValueError("extracted corpus COMPLETE metadata mismatch")

    ready_for_data_prepare = value.get("ready_for_data_prepare")
    ready_for_training = value.get("ready_for_training")
    if ready_for_data_prepare is not True:
        raise ValueError("extracted corpus is not ready for data prepare")
    if not isinstance(ready_for_training, bool):
        raise ValueError("extracted ready_for_training flag must be boolean")
    audits = value.get("audits")
    if not isinstance(audits, dict) or not all(
        isinstance(name, str) and name and isinstance(status, str) and status
        for name, status in audits.items()
    ):
        raise ValueError("extracted corpus audits must be string statuses")
    audit_lineage: dict[str, object] | None = None
    if audit_attestation is not None:
        from .audits import audit_lineage_for_role

        audit_lineage = audit_lineage_for_role(
            audit_attestation,
            extracted_manifest_path=path,
            role=role,
        )
        raw_gates = audit_lineage["gates"]
        assert isinstance(raw_gates, dict)
        merged_audits = dict(audits)
        for name, gate in raw_gates.items():
            if not isinstance(name, str) or not isinstance(gate, dict):
                raise ValueError("audit attestation gates are invalid")
            status = gate.get("status")
            if not isinstance(status, str) or not status:
                raise ValueError("audit attestation gate status is invalid")
            merged_audits[name] = status
        audits = merged_audits
        ready_for_training = bool(audit_lineage["ready_for_training"])
    pending_audits = sorted(
        name for name, status in audits.items() if status.lower().startswith("pending")
    )
    if ready_for_training and pending_audits:
        raise ValueError(
            "audit attestation reports ready_for_training but the extracted "
            f"corpus still has unresolved pending audits: {pending_audits}"
        )
    if not ready_for_training and not allow_pending_research_audits:
        raise ValueError(
            "extracted corpus is not ready_for_training; after reviewing its pending audits, "
            "pass --allow-pending-research-audits to create a research-only prepared corpus"
        )

    extracted_tokenizer_sha = _normalized_sha256(
        _required_string(
            value.get("tokenizer_manifest_sha256"),
            "extracted tokenizer_manifest_sha256",
        ),
        "extracted tokenizer_manifest_sha256",
    )
    if extracted_tokenizer_sha != expected_tokenizer_sha:
        raise ValueError(
            "extracted corpus tokenizer identity differs from --tokenizer-manifest-sha256"
        )
    recipe_sha = _normalized_sha256(
        _required_string(value.get("recipe_sha256"), "recipe_sha256"),
        "recipe_sha256",
    )
    resolved_lock_sha = _normalized_sha256(
        _required_string(
            value.get("resolved_source_lock_sha256"),
            "resolved_source_lock_sha256",
        ),
        "resolved_source_lock_sha256",
    )
    extractor_sha = _normalized_sha256(
        _required_string(value.get("extractor_source_sha256"), "extractor_source_sha256"),
        "extractor_source_sha256",
    )
    selected = resolved_inventories[role]
    lineage = {
        "kind": "authenticated_extracted_corpus",
        "extracted_manifest_path": str(path),
        "extracted_manifest_sha256": manifest_sha,
        "corpus_fingerprint": actual_fingerprint,
        "recipe_id": _required_string(value.get("recipe_id"), "recipe_id"),
        "recipe_sha256": recipe_sha,
        "resolved_source_lock_sha256": resolved_lock_sha,
        "tokenizer_manifest_sha256": extracted_tokenizer_sha,
        "extractor_source_sha256": extractor_sha,
        "profile": _required_string(value.get("profile"), "profile"),
        "role": role,
        "file_list": normalized_file_lists[role],
        "source_files": [dict(item) for _, item in selected],
        "audits": dict(audits),
        "pending_audits": pending_audits,
        "ready_for_data_prepare": True,
        "ready_for_training": ready_for_training,
        "research_only": not ready_for_training,
    }
    if data_contract is not None:
        contract_unsigned = {
            "schema_version": 1,
            **data_contract,
        }
        lineage["data_contract"] = {
            **contract_unsigned,
            "contract_fingerprint": _canonical_sha256(contract_unsigned),
        }
    if audit_lineage is not None:
        lineage["audit_attestation"] = audit_lineage
    normalized_lineage = _normalized_prepared_lineage(lineage)
    assert normalized_lineage is not None
    return (
        tuple((source, str(item["sha256"])) for source, item in selected),
        normalized_lineage,
    )


def _prepared_pipeline_fingerprint(
    source_hashes: Sequence[tuple[Path, str]],
    *,
    tokenizer_sha256: str,
    sequence_length: int,
    text_field: str,
    generator_source_sha256: str,
    lineage: Mapping[str, object] | None = None,
) -> str:
    payload: dict[str, object] = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "sources": [(str(path), digest) for path, digest in source_hashes],
        "tokenizer_sha256": tokenizer_sha256,
        "sequence_length": sequence_length,
        "text_field": text_field,
        "generator_source_sha256": generator_source_sha256,
    }
    if lineage is not None:
        payload["lineage"] = _normalized_prepared_lineage(lineage)
    return _canonical_sha256(payload)


def _prepared_dataset_fingerprint(
    *,
    pipeline_fingerprint: str,
    generator_source_sha256: str,
    tokenizer_sha256: str,
    sequence_length: int,
    text_field: str,
    shards: Sequence[PreparedShardEntry],
    lineage: Mapping[str, object] | None = None,
) -> str:
    payload: dict[str, object] = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "pipeline_fingerprint": pipeline_fingerprint,
        "generator_source_sha256": generator_source_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "sequence_length": sequence_length,
        "text_field": text_field,
        "shards": [asdict(entry) for entry in shards],
    }
    if lineage is not None:
        payload["lineage"] = _normalized_prepared_lineage(lineage)
    return _canonical_sha256(payload)


def _local_prepared_manifest(
    *,
    shard_id: str,
    source_path: str,
    source_sha256: str,
    sequence_count: int,
    token_count: int,
    tensors_sha256: str,
    pipeline_fingerprint: str,
    generator_source_sha256: str,
    tokenizer_sha256: str,
    sequence_length: int,
    text_field: str,
) -> dict[str, object]:
    return {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "kind": "twen_prepared_text_shard",
        "shard_id": shard_id,
        "source_path": source_path,
        "source_sha256": source_sha256,
        "sequence_count": sequence_count,
        "token_count": token_count,
        "tensors_sha256": tensors_sha256,
        "pipeline_fingerprint": pipeline_fingerprint,
        "generator_source_sha256": generator_source_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "sequence_length": sequence_length,
        "text_field": text_field,
    }


def _read_jsonl_text(path: Path, field: str) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            text = value.get(field) if isinstance(value, dict) else None
            if not isinstance(text, str):
                raise ValueError(f"{path}:{line_number} has no string field {field!r}")
            if text:
                yield text


def _pack_source(
    source: Path,
    destination: Path,
    *,
    tokenizer: Any,
    text_field: str,
    sequence_length: int,
) -> tuple[int, int, str]:
    import torch
    from safetensors.torch import save_file

    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer must define eos_token_id")
    sequences: list[list[int]] = []
    buffer: list[int] = []
    valid_tokens = 0
    for text in _read_jsonl_text(source, text_field):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        buffer.extend(encoded)
        buffer.append(eos)
        valid_tokens += len(encoded) + 1
        while len(buffer) >= sequence_length:
            sequences.append(buffer[:sequence_length])
            del buffer[:sequence_length]
    if buffer:
        sequences.append(buffer + [eos] * (sequence_length - len(buffer)))
    if not sequences:
        raise ValueError(f"source shard contains no tokens: {source}")
    input_ids = torch.tensor(sequences, dtype=torch.int64)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    padding = input_ids.numel() - valid_tokens
    if padding:
        attention_mask[-1, -padding:] = False
    labels = input_ids.clone()
    labels[~attention_mask] = -100
    tensor_path = destination / PREPARED_TENSORS
    save_file(
        {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask},
        str(tensor_path),
    )
    return len(sequences), valid_tokens, sha256_file(tensor_path)


def prepare_jsonl_corpus(
    sources: Sequence[str | Path] | None,
    output_root: str | Path,
    *,
    tokenizer_path: str | Path,
    tokenizer_sha256: str,
    sequence_length: int,
    text_field: str = "text",
    progress: str = "auto",
    extracted_manifest: str | Path | None = None,
    role: str | None = None,
    allow_pending_research_audits: bool = False,
    audit_attestation: str | Path | None = None,
) -> Path:
    """Tokenize JSONL sources one shard at a time, retaining partial work.

    Exactly one source mode is required.  ``extracted_manifest`` authenticates
    the extractor's COMPLETE marker, sidecars, identity, and every output byte;
    explicit paths remain supported but are permanently labeled unreviewed.
    """

    from ..io.offline import (
        enforce_offline_environment,
        verify_local_download_directory,
    )

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    tokenizer_sha256 = _normalized_sha256(tokenizer_sha256, "tokenizer_sha256")
    enforce_offline_environment()
    verify_local_download_directory(
        tokenizer_path,
        expected_manifest_sha256=tokenizer_sha256,
    )
    from transformers import AutoTokenizer

    explicit_sources = tuple(sources or ())
    if extracted_manifest is not None:
        if explicit_sources:
            raise ValueError("--input and --extracted-manifest are mutually exclusive")
        if role is None:
            raise ValueError("--role is required with --extracted-manifest")
        source_hashes_tuple, lineage = _authenticate_extracted_prepare_inputs(
            extracted_manifest,
            role=role,
            tokenizer_sha256=tokenizer_sha256,
            allow_pending_research_audits=allow_pending_research_audits,
            audit_attestation=audit_attestation,
        )
        source_hashes = list(source_hashes_tuple)
    else:
        if not explicit_sources:
            raise ValueError("exactly one of --input or --extracted-manifest is required")
        if role is not None:
            raise ValueError("--role is only valid with --extracted-manifest")
        if allow_pending_research_audits:
            raise ValueError(
                "--allow-pending-research-audits is only valid with --extracted-manifest"
            )
        if audit_attestation is not None:
            raise ValueError("--audit-attestation is only valid with --extracted-manifest")
        ordered = tuple(sorted(Path(item).resolve() for item in explicit_sources))
        source_hashes = [(path, sha256_file(path)) for path in ordered]
        lineage = _explicit_unreviewed_lineage()
    generator_source_sha256 = (
        AUDITED_PREPARED_GENERATOR_SOURCE_SHA256
        if extracted_manifest is not None
        else PREPARED_GENERATOR_SOURCE_SHA256
    )
    pipeline_fingerprint = _prepared_pipeline_fingerprint(
        source_hashes,
        tokenizer_sha256=tokenizer_sha256,
        sequence_length=sequence_length,
        text_field=text_field,
        generator_source_sha256=generator_source_sha256,
        lineage=lineage,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    prepared: list[tuple[Path, Path, str, int, int, str]] = []
    with TaskProgress(
        total=len(source_hashes),
        description="prepare",
        unit="shard",
        mode=progress,
    ) as progress_bar:
        for index, (source, source_hash) in enumerate(source_hashes):
            shard_id = f"shard-{index:06d}"
            with ShardTransaction(
                root,
                shard_id,
                fingerprint=pipeline_fingerprint,
                source_fingerprint=source_hash,
            ) as transaction:
                if not transaction.complete:
                    count, tokens, tensor_hash = _pack_source(
                        source,
                        transaction.work_directory,
                        tokenizer=tokenizer,
                        text_field=text_field,
                        sequence_length=sequence_length,
                    )
                    atomic_write_json(
                        transaction.work_directory / PREPARED_SHARD_MANIFEST,
                        _local_prepared_manifest(
                            shard_id=shard_id,
                            source_path=str(source),
                            source_sha256=source_hash,
                            sequence_count=count,
                            token_count=tokens,
                            tensors_sha256=tensor_hash,
                            pipeline_fingerprint=pipeline_fingerprint,
                            generator_source_sha256=generator_source_sha256,
                            tokenizer_sha256=tokenizer_sha256,
                            sequence_length=sequence_length,
                            text_field=text_field,
                        ),
                    )
                    transaction.commit(
                        {"kind": "prepared_text", "sequences": count, "tokens": tokens}
                    )
                manifest = json.loads(
                    (transaction.final_directory / PREPARED_SHARD_MANIFEST).read_text(
                        encoding="utf-8"
                    )
                )
                prepared.append(
                    (
                        transaction.final_directory,
                        source,
                        source_hash,
                        int(manifest["sequence_count"]),
                        int(manifest["token_count"]),
                        str(manifest["tensors_sha256"]),
                    )
                )
            progress_bar.set_postfix({"shard": shard_id})
            progress_bar.update()
    sample_cursor = 0
    token_cursor = 0
    entries = []
    for directory, source, source_hash, count, tokens, tensor_hash in prepared:
        entries.append(
            PreparedShardEntry(
                shard_id=directory.name,
                path=directory.relative_to(root).as_posix(),
                source_path=str(source),
                source_sha256=source_hash,
                tensors_sha256=tensor_hash,
                sequence_count=count,
                token_count=tokens,
                global_sample_start=sample_cursor,
                global_sample_end=sample_cursor + count,
                global_token_start=token_cursor,
                global_token_end=token_cursor + tokens,
            )
        )
        sample_cursor += count
        token_cursor += tokens
    # The transaction fingerprint identifies the deterministic recipe and is
    # useful for reusing an already completed shard.  The corpus identity must
    # additionally bind the bytes that recipe actually produced.  Tokenizer or
    # preprocessing-runtime changes can otherwise yield different tensors with
    # the same source/config recipe and accidentally reuse stale teacher logits.
    dataset_fingerprint = _prepared_dataset_fingerprint(
        pipeline_fingerprint=pipeline_fingerprint,
        generator_source_sha256=generator_source_sha256,
        tokenizer_sha256=tokenizer_sha256,
        sequence_length=sequence_length,
        text_field=text_field,
        shards=entries,
        lineage=lineage,
    )
    corpus = PreparedCorpusManifest(
        dataset_fingerprint=dataset_fingerprint,
        pipeline_fingerprint=pipeline_fingerprint,
        generator_source_sha256=generator_source_sha256,
        tokenizer_sha256=tokenizer_sha256,
        sequence_length=sequence_length,
        text_field=text_field,
        shards=tuple(entries),
        lineage=lineage,
    )
    output = root / "manifest.json"
    atomic_write_json(output, corpus.to_dict())
    return output


def validate_prepared_corpus(path: str | Path) -> PreparedCorpusManifest:
    manifest_path = Path(path).resolve()
    corpus = read_prepared_manifest(manifest_path)
    root = manifest_path.parent.resolve()
    if corpus.generator_source_sha256 not in {
        PREPARED_GENERATOR_SOURCE_SHA256,
        AUDITED_PREPARED_GENERATOR_SOURCE_SHA256,
    }:
        raise ValueError(
            "prepared generator source changed; rebuild the prepared corpus "
            "instead of reusing stale shards"
        )
    is_quality_cooldown = bool(
        isinstance(corpus.lineage, Mapping)
        and isinstance(corpus.lineage.get("quality_cooldown"), Mapping)
    )
    if (
        corpus.lineage is not None
        and corpus.lineage.get("kind") == "authenticated_extracted_corpus"
        and not is_quality_cooldown
    ):
        authenticated_sources, expected_lineage = _authenticate_extracted_prepare_inputs(
            str(corpus.lineage["extracted_manifest_path"]),
            role=str(corpus.lineage["role"]),
            tokenizer_sha256=corpus.tokenizer_sha256,
            # Validation does not grant a new override.  It only verifies that
            # an already research-only artifact still matches its authenticated
            # source and records that state exactly.
            allow_pending_research_audits=True,
            audit_attestation=(
                str(corpus.lineage["audit_attestation"]["path"])
                if isinstance(corpus.lineage.get("audit_attestation"), dict)
                else None
            ),
        )
        if expected_lineage != corpus.lineage:
            raise ValueError("prepared extracted-corpus lineage no longer authenticates")
        prepared_sources = tuple(
            (Path(entry.source_path).resolve(), entry.source_sha256) for entry in corpus.shards
        )
        if prepared_sources != authenticated_sources:
            raise ValueError(
                "prepared shards do not exactly match authenticated extracted role inventory"
            )
    expected_pipeline = _prepared_pipeline_fingerprint(
        [(Path(entry.source_path), entry.source_sha256) for entry in corpus.shards],
        tokenizer_sha256=corpus.tokenizer_sha256,
        sequence_length=corpus.sequence_length,
        text_field=corpus.text_field,
        generator_source_sha256=corpus.generator_source_sha256,
        lineage=corpus.lineage,
    )
    if expected_pipeline != corpus.pipeline_fingerprint:
        raise ValueError("prepared pipeline fingerprint does not match manifest inputs")
    expected_dataset = _prepared_dataset_fingerprint(
        pipeline_fingerprint=corpus.pipeline_fingerprint,
        generator_source_sha256=corpus.generator_source_sha256,
        tokenizer_sha256=corpus.tokenizer_sha256,
        sequence_length=corpus.sequence_length,
        text_field=corpus.text_field,
        shards=corpus.shards,
        lineage=corpus.lineage,
    )
    if expected_dataset != corpus.dataset_fingerprint:
        raise ValueError("prepared dataset fingerprint does not match manifest contents")
    for entry in corpus.shards:
        directory = (root / entry.path).resolve()
        try:
            directory.relative_to(root)
        except ValueError as error:
            raise ValueError(f"prepared shard escapes root: {entry.path}") from error
        if directory == root:
            raise ValueError(f"prepared shard escapes root: {entry.path}")
        if not is_shard_complete(directory):
            raise ValueError(f"prepared shard is incomplete: {entry.path}")
        if sha256_file(directory / PREPARED_TENSORS) != entry.tensors_sha256:
            raise ValueError(f"prepared tensor hash mismatch: {entry.path}")
        local_path = directory / PREPARED_SHARD_MANIFEST
        try:
            local = json.loads(local_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid local prepared manifest: {entry.path}") from error
        expected_local = _local_prepared_manifest(
            shard_id=entry.shard_id,
            source_path=entry.source_path,
            source_sha256=entry.source_sha256,
            sequence_count=entry.sequence_count,
            token_count=entry.token_count,
            tensors_sha256=entry.tensors_sha256,
            pipeline_fingerprint=corpus.pipeline_fingerprint,
            generator_source_sha256=corpus.generator_source_sha256,
            tokenizer_sha256=corpus.tokenizer_sha256,
            sequence_length=corpus.sequence_length,
            text_field=corpus.text_field,
        )
        if local != expected_local:
            raise ValueError(f"local prepared manifest differs from corpus entry: {entry.path}")
    return corpus
