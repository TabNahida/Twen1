"""World-size-independent deterministic global data cursor."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path, PurePosixPath

CURSOR_SCHEMA_VERSION = 2
SHUFFLE_ALGORITHM = "shard-local-affine-v1"
QUALITY_COOLDOWN_CURSOR_SCHEMA_VERSION = 1
QUALITY_COOLDOWN_CURSOR_KIND = "deterministic-two-phase-quality-cooldown"
SOURCE_MAP_ALGORITHM = "authenticated-extracted-output-map-v1"
SOURCE_MIX_CURSOR_SCHEMA_VERSION = 2
SOURCE_MIX_CURSOR_KIND = "deterministic-source-mix"
SOURCE_MIX_ALGORITHM = "token-deficit-corrected-source-mix-bp-v2"
SOURCE_MIX_BASIS_POINTS = 10_000


def _stable_integer(*parts: object) -> int:
    digest = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "little")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{field_name} must be a 64-digit SHA256")
    return value.lower()


def _require_nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_nonnegative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _safe_output_path(value: object, field_name: str) -> str:
    text = _require_nonempty_string(value, field_name)
    relative = PurePosixPath(text)
    if relative.is_absolute() or relative.as_posix() == "." or ".." in relative.parts:
        raise ValueError(f"unsafe {field_name}: {text!r}")
    return relative.as_posix()


@dataclass(frozen=True, slots=True)
class DatasetLayout:
    """Flat deterministic index over immutable source shards."""

    shard_ids: tuple[str, ...]
    shard_lengths: tuple[int, ...]
    fingerprint: str
    _cumulative_ends: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _size: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.shard_ids or len(self.shard_ids) != len(self.shard_lengths):
            raise ValueError("shard_ids and shard_lengths must be non-empty and aligned")
        if len(set(self.shard_ids)) != len(self.shard_ids):
            raise ValueError("shard IDs must be unique")
        if any(length <= 0 for length in self.shard_lengths):
            raise ValueError("all shard lengths must be positive")
        if not self.fingerprint:
            raise ValueError("dataset fingerprint is required")
        total = 0
        ends: list[int] = []
        for length in self.shard_lengths:
            total += length
            ends.append(total)
        object.__setattr__(self, "_size", total)
        object.__setattr__(self, "_cumulative_ends", tuple(ends))

    @classmethod
    def from_shards(
        cls,
        shards: Mapping[str, int] | Iterable[tuple[str, int]],
        *,
        fingerprint: str | None = None,
    ) -> DatasetLayout:
        items = list(shards.items() if isinstance(shards, Mapping) else shards)
        ids = tuple(item[0] for item in items)
        lengths = tuple(int(item[1]) for item in items)
        if fingerprint is None:
            canonical = json.dumps(items, separators=(",", ":"), ensure_ascii=False)
            fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return cls(ids, lengths, fingerprint)

    @property
    def size(self) -> int:
        return self._size

    @property
    def cumulative_ends(self) -> tuple[int, ...]:
        return self._cumulative_ends

    def locate(self, flat_index: int) -> tuple[str, int]:
        if not 0 <= flat_index < self.size:
            raise IndexError(flat_index)
        ends = self.cumulative_ends
        shard_index = bisect.bisect_right(ends, flat_index)
        start = 0 if shard_index == 0 else ends[shard_index - 1]
        return self.shard_ids[shard_index], flat_index - start


@dataclass(frozen=True, slots=True)
class AuthenticatedSourceShard:
    """One prepared shard bound to an extracted-manifest output owner."""

    source_id: str
    shard_id: str
    sequence_count: int
    global_sample_start: int
    output_path: str
    output_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_id",
            _require_nonempty_string(self.source_id, "source_id"),
        )
        object.__setattr__(
            self,
            "shard_id",
            _require_nonempty_string(self.shard_id, "shard_id"),
        )
        if (
            isinstance(self.sequence_count, bool)
            or not isinstance(self.sequence_count, int)
            or self.sequence_count <= 0
        ):
            raise ValueError("source-map sequence_count must be a positive integer")
        _require_nonnegative_integer(self.global_sample_start, "global_sample_start")
        object.__setattr__(
            self,
            "output_path",
            _safe_output_path(self.output_path, "output_path"),
        )
        object.__setattr__(
            self,
            "output_sha256",
            _require_sha256(self.output_sha256, "output_sha256"),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AuthenticatedSourceMap:
    """Exact prepared-shard ownership recovered from authenticated lineage.

    Source ownership is never inferred from a shard ID, directory component,
    or filename.  ``from_prepared_manifest`` joins each prepared ``source_path``
    to the exact role inventory embedded in prepared lineage, then joins that
    relative output path to the owning ``source_id`` in the SHA256-bound
    extracted manifest.
    """

    prepared_dataset_fingerprint: str
    extracted_manifest_sha256: str
    sequence_length: int
    shards: tuple[AuthenticatedSourceShard, ...]
    mix_basis_points: tuple[tuple[str, int], ...] = ()
    algorithm: str = SOURCE_MAP_ALGORITHM

    def __post_init__(self) -> None:
        if self.algorithm != SOURCE_MAP_ALGORITHM:
            raise ValueError(f"unsupported source-map algorithm {self.algorithm!r}")
        object.__setattr__(
            self,
            "prepared_dataset_fingerprint",
            _require_sha256(
                self.prepared_dataset_fingerprint,
                "prepared_dataset_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "extracted_manifest_sha256",
            _require_sha256(
                self.extracted_manifest_sha256,
                "extracted_manifest_sha256",
            ),
        )
        if (
            isinstance(self.sequence_length, bool)
            or not isinstance(self.sequence_length, int)
            or self.sequence_length <= 0
        ):
            raise ValueError("authenticated source-map sequence_length must be positive")
        if not self.shards:
            raise ValueError("authenticated source map cannot be empty")
        if len({item.shard_id for item in self.shards}) != len(self.shards):
            raise ValueError("authenticated source-map shard IDs must be unique")
        if len({item.output_path for item in self.shards}) != len(self.shards):
            raise ValueError("authenticated source-map output paths must be unique")
        next_global_sample = 0
        for item in self.shards:
            if item.global_sample_start != next_global_sample:
                raise ValueError(
                    "authenticated source-map sample ranges must be contiguous from zero"
                )
            next_global_sample += item.sequence_count
        if self.mix_basis_points:
            mix = dict(self.mix_basis_points)
            if len(mix) != len(self.mix_basis_points):
                raise ValueError("authenticated source-map mix IDs must be unique")
            if set(mix) != set(self.source_ids):
                raise ValueError(
                    "authenticated source-map mix must cover source IDs exactly"
                )
            if any(
                isinstance(weight, bool)
                or not isinstance(weight, int)
                or weight <= 0
                for weight in mix.values()
            ):
                raise ValueError(
                    "authenticated source-map mix weights must be positive integers"
                )
            if sum(mix.values()) != SOURCE_MIX_BASIS_POINTS:
                raise ValueError(
                    "authenticated source-map mix must total 10,000 basis points"
                )

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(sorted({item.source_id for item in self.shards}))

    @property
    def sequence_count(self) -> int:
        return sum(item.sequence_count for item in self.shards)

    @property
    def source_mix_weights(self) -> dict[str, int]:
        return dict(self.mix_basis_points)

    @property
    def fingerprint(self) -> str:
        """Canonical identity of the authenticated prepared-shard ownership map."""

        return _canonical_sha256(self.to_dict())

    def shards_for_source(self, source_id: str) -> tuple[AuthenticatedSourceShard, ...]:
        result = tuple(item for item in self.shards if item.source_id == source_id)
        if not result:
            raise KeyError(f"unknown authenticated source {source_id!r}")
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "prepared_dataset_fingerprint": self.prepared_dataset_fingerprint,
            "extracted_manifest_sha256": self.extracted_manifest_sha256,
            "sequence_length": self.sequence_length,
            "shards": [item.to_dict() for item in self.shards],
            "mix_basis_points": self.source_mix_weights,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AuthenticatedSourceMap:
        """Reconstruct a source map already authenticated by coordinated preflight."""

        expected_fields = {
            "algorithm",
            "prepared_dataset_fingerprint",
            "extracted_manifest_sha256",
            "sequence_length",
            "shards",
            "mix_basis_points",
        }
        if set(payload) != expected_fields:
            raise ValueError("authenticated source-map fields differ from schema")
        raw_shards = payload.get("shards")
        if not isinstance(raw_shards, list) or not raw_shards:
            raise ValueError("authenticated source-map shards must be a non-empty list")
        raw_mix = payload.get("mix_basis_points")
        if not isinstance(raw_mix, Mapping):
            raise ValueError("authenticated source-map mix must be an object")
        if not isinstance(payload["algorithm"], str):
            raise ValueError("authenticated source-map algorithm must be a string")
        for field_name in (
            "prepared_dataset_fingerprint",
            "extracted_manifest_sha256",
        ):
            if not isinstance(payload[field_name], str):
                raise ValueError(
                    f"authenticated source-map {field_name} must be a string"
                )
        sequence_length = payload["sequence_length"]
        if (
            isinstance(sequence_length, bool)
            or not isinstance(sequence_length, int)
            or sequence_length <= 0
        ):
            raise ValueError(
                "authenticated source-map sequence_length must be a positive integer"
            )

        expected_shard_fields = {
            "source_id",
            "shard_id",
            "sequence_count",
            "global_sample_start",
            "output_path",
            "output_sha256",
        }
        shards: list[AuthenticatedSourceShard] = []
        for index, item in enumerate(raw_shards):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"authenticated source-map shards[{index}] must be an object"
                )
            if set(item) != expected_shard_fields:
                raise ValueError(
                    f"authenticated source-map shards[{index}] fields differ from schema"
                )
            for field_name in (
                "source_id",
                "shard_id",
                "output_path",
                "output_sha256",
            ):
                if not isinstance(item[field_name], str):
                    raise ValueError(
                        "authenticated source-map "
                        f"shards[{index}].{field_name} must be a string"
                    )
            for field_name in ("sequence_count", "global_sample_start"):
                value = item[field_name]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(
                        "authenticated source-map "
                        f"shards[{index}].{field_name} must be an integer"
                    )
            shards.append(
                AuthenticatedSourceShard(
                    source_id=item["source_id"],
                    shard_id=item["shard_id"],
                    sequence_count=item["sequence_count"],
                    global_sample_start=item["global_sample_start"],
                    output_path=item["output_path"],
                    output_sha256=item["output_sha256"],
                )
            )

        mix_basis_points: list[tuple[str, int]] = []
        for source_id, weight in raw_mix.items():
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError(
                    "authenticated source-map mix IDs must be non-empty strings"
                )
            if isinstance(weight, bool) or not isinstance(weight, int):
                raise ValueError(
                    "authenticated source-map mix weights must be integers"
                )
            mix_basis_points.append((source_id, weight))

        result = cls(
            algorithm=payload["algorithm"],
            prepared_dataset_fingerprint=payload["prepared_dataset_fingerprint"],
            extracted_manifest_sha256=payload["extracted_manifest_sha256"],
            sequence_length=sequence_length,
            shards=tuple(shards),
            mix_basis_points=tuple(sorted(mix_basis_points)),
        )
        if result.to_dict() != dict(payload):
            raise ValueError("authenticated source-map payload is not canonical")
        return result

    @classmethod
    def from_prepared_manifest(cls, prepared: object) -> AuthenticatedSourceMap:
        """Build an exact source map from a validated prepared manifest.

        The caller should pass the result of ``validate_prepared_corpus``.  This
        method re-authenticates the source ownership metadata but deliberately
        does not rescan prepared tensor payloads.
        """

        prepared_fingerprint = _require_sha256(
            getattr(prepared, "dataset_fingerprint", None),
            "prepared.dataset_fingerprint",
        )
        prepared_sequence_length = getattr(prepared, "sequence_length", None)
        if (
            isinstance(prepared_sequence_length, bool)
            or not isinstance(prepared_sequence_length, int)
            or prepared_sequence_length <= 0
        ):
            raise ValueError("prepared.sequence_length must be a positive integer")
        lineage = getattr(prepared, "lineage", None)
        if not isinstance(lineage, Mapping):
            raise ValueError("source mixing requires prepared authenticated lineage")
        if lineage.get("kind") != "authenticated_extracted_corpus":
            raise ValueError("source mixing requires authenticated_extracted_corpus lineage")
        if lineage.get("role") != "train":
            raise ValueError("source mixing requires authenticated train lineage")
        ready_for_training = lineage.get("ready_for_training")
        research_only = lineage.get("research_only")
        if (ready_for_training, research_only) not in {
            (True, False),
            (False, True),
        }:
            raise ValueError(
                "source mixing requires an explicit ready or research-only governance state"
            )

        manifest_path_text = _require_nonempty_string(
            lineage.get("extracted_manifest_path"),
            "lineage.extracted_manifest_path",
        )
        manifest_path = Path(manifest_path_text).expanduser().resolve()
        expected_manifest_sha = _require_sha256(
            lineage.get("extracted_manifest_sha256"),
            "lineage.extracted_manifest_sha256",
        )
        try:
            manifest_raw = manifest_path.read_bytes()
        except OSError as error:
            raise ValueError(f"cannot read authenticated extracted manifest: {manifest_path}") from error
        actual_manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
        if actual_manifest_sha != expected_manifest_sha:
            raise ValueError("authenticated extracted manifest SHA256 mismatch")
        try:
            extracted = json.loads(manifest_raw)
        except json.JSONDecodeError as error:
            raise ValueError("authenticated extracted manifest is not valid JSON") from error
        if (
            not isinstance(extracted, dict)
            or extracted.get("schema_version") != 1
            or extracted.get("kind") != "twen_extracted_base_jsonl_corpus"
        ):
            raise ValueError("unsupported authenticated extracted manifest")
        if extracted.get("corpus_fingerprint") != lineage.get("corpus_fingerprint"):
            raise ValueError("prepared/extracted corpus fingerprints differ")

        raw_source_files = lineage.get("source_files")
        if not isinstance(raw_source_files, list) or not raw_source_files:
            raise ValueError("authenticated train lineage has no source files")
        selected_files: dict[str, tuple[int, str]] = {}
        normalized_inventory: list[dict[str, object]] = []
        for index, raw_item in enumerate(raw_source_files):
            if not isinstance(raw_item, Mapping):
                raise ValueError(f"lineage.source_files[{index}] must be an object")
            path = _safe_output_path(raw_item.get("path"), f"lineage.source_files[{index}].path")
            size = _require_nonnegative_integer(
                raw_item.get("size"),
                f"lineage.source_files[{index}].size",
            )
            digest = _require_sha256(
                raw_item.get("sha256"),
                f"lineage.source_files[{index}].sha256",
            )
            if path in selected_files:
                raise ValueError(f"duplicate authenticated source output path: {path}")
            selected_files[path] = (size, digest)
            normalized_inventory.append({"path": path, "size": size, "sha256": digest})

        raw_role_inventory = extracted.get("train_files")
        if not isinstance(raw_role_inventory, list):
            raise ValueError("extracted manifest train_files inventory is invalid")
        normalized_role_inventory: list[dict[str, object]] = []
        for index, raw_item in enumerate(raw_role_inventory):
            if not isinstance(raw_item, Mapping):
                raise ValueError(f"extracted.train_files[{index}] must be an object")
            normalized_role_inventory.append(
                {
                    "path": _safe_output_path(
                        raw_item.get("path"),
                        f"extracted.train_files[{index}].path",
                    ),
                    "size": _require_nonnegative_integer(
                        raw_item.get("size"),
                        f"extracted.train_files[{index}].size",
                    ),
                    "sha256": _require_sha256(
                        raw_item.get("sha256"),
                        f"extracted.train_files[{index}].sha256",
                    ),
                }
            )
        if normalized_role_inventory != normalized_inventory:
            raise ValueError("prepared lineage differs from extracted train output inventory")

        raw_sources = extracted.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ValueError("authenticated extracted manifest has no sources")
        output_owners: dict[str, tuple[str, int, str]] = {}
        source_ids: set[str] = set()
        for source_index, raw_source in enumerate(raw_sources):
            if not isinstance(raw_source, Mapping):
                raise ValueError(f"extracted.sources[{source_index}] must be an object")
            source_id = _require_nonempty_string(
                raw_source.get("source_id"),
                f"extracted.sources[{source_index}].source_id",
            )
            if source_id in source_ids:
                raise ValueError(f"duplicate extracted source_id: {source_id}")
            source_ids.add(source_id)
            raw_chunks = raw_source.get("chunks")
            if not isinstance(raw_chunks, list):
                raise ValueError(f"extracted source {source_id!r} chunks must be a list")
            for chunk_index, raw_chunk in enumerate(raw_chunks):
                if not isinstance(raw_chunk, Mapping):
                    raise ValueError(
                        f"extracted source {source_id!r} chunk {chunk_index} must be an object"
                    )
                raw_outputs = raw_chunk.get("outputs")
                if not isinstance(raw_outputs, list):
                    raise ValueError(
                        f"extracted source {source_id!r} chunk {chunk_index} outputs "
                        "must be a list"
                    )
                for output_index, raw_output in enumerate(raw_outputs):
                    if not isinstance(raw_output, Mapping):
                        raise ValueError(
                            f"extracted source {source_id!r} output {output_index} "
                            "must be an object"
                        )
                    output_path = _safe_output_path(
                        raw_output.get("path"),
                        f"extracted.sources[{source_index}].chunks[{chunk_index}]"
                        f".outputs[{output_index}].path",
                    )
                    if output_path not in selected_files:
                        continue
                    output_identity = (
                        source_id,
                        _require_nonnegative_integer(
                            raw_output.get("size"),
                            f"extracted output {output_path!r} size",
                        ),
                        _require_sha256(
                            raw_output.get("sha256"),
                            f"extracted output {output_path!r} sha256",
                        ),
                    )
                    if output_path in output_owners:
                        raise ValueError(
                            f"authenticated output belongs to multiple source entries: "
                            f"{output_path}"
                        )
                    output_owners[output_path] = output_identity

        if set(output_owners) != set(selected_files):
            missing = sorted(set(selected_files) - set(output_owners))
            raise ValueError(
                "authenticated train inventory is underfilled by source outputs: "
                + ", ".join(missing)
            )
        for output_path, (expected_size, expected_digest) in selected_files.items():
            _, actual_size, actual_digest = output_owners[output_path]
            if (actual_size, actual_digest) != (expected_size, expected_digest):
                raise ValueError(
                    f"authenticated output metadata mismatch for {output_path}"
                )

        mix_basis_points: tuple[tuple[str, int], ...] = ()
        data_contract = lineage.get("data_contract")
        extracted_contract_fields = (
            "source_map",
            "source_mix",
            "format_audit",
            "license_audit",
            "materialization_audit",
        )
        extracted_contract_presence = [
            field in extracted for field in extracted_contract_fields
        ]
        if any(extracted_contract_presence) and not all(extracted_contract_presence):
            raise ValueError("authenticated extracted data contract is partial")
        if all(extracted_contract_presence):
            if not isinstance(data_contract, Mapping):
                raise ValueError(
                    "prepared lineage dropped the authenticated extracted data contract"
                )
            for field in extracted_contract_fields:
                if data_contract.get(field) != extracted.get(field):
                    raise ValueError(
                        f"prepared/extracted data contract differs for {field}"
                    )
            source_map_contract = data_contract.get("source_map")
            if (
                not isinstance(source_map_contract, Mapping)
                or source_map_contract.get("algorithm") != SOURCE_MAP_ALGORITHM
            ):
                raise ValueError("unsupported authenticated source_map contract")
            raw_roles = source_map_contract.get("roles")
            raw_train_map = (
                raw_roles.get("train") if isinstance(raw_roles, Mapping) else None
            )
            if not isinstance(raw_train_map, list):
                raise ValueError("authenticated source_map train role is invalid")
            explicit_owners: dict[str, tuple[str, int, str]] = {}
            for index, raw_output in enumerate(raw_train_map):
                if not isinstance(raw_output, Mapping):
                    raise ValueError(
                        f"authenticated source_map train[{index}] is invalid"
                    )
                output_path = _safe_output_path(
                    raw_output.get("path"),
                    f"source_map.train[{index}].path",
                )
                owner = (
                    _require_nonempty_string(
                        raw_output.get("source_id"),
                        f"source_map.train[{index}].source_id",
                    ),
                    _require_nonnegative_integer(
                        raw_output.get("size"),
                        f"source_map.train[{index}].size",
                    ),
                    _require_sha256(
                        raw_output.get("sha256"),
                        f"source_map.train[{index}].sha256",
                    ),
                )
                if output_path in explicit_owners:
                    raise ValueError(
                        f"duplicate authenticated source_map output: {output_path}"
                    )
                explicit_owners[output_path] = owner
            if explicit_owners != output_owners:
                raise ValueError(
                    "explicit source_map differs from authenticated chunk ownership"
                )

            source_mix_contract = data_contract.get("source_mix")
            if (
                not isinstance(source_mix_contract, Mapping)
                or source_mix_contract.get("algorithm") != SOURCE_MIX_ALGORITHM
                or source_mix_contract.get("unit") != "valid_tokens"
                or source_mix_contract.get("basis_points_total")
                != SOURCE_MIX_BASIS_POINTS
            ):
                raise ValueError("unsupported authenticated source_mix contract")
            raw_mix_sources = source_mix_contract.get("sources")
            if not isinstance(raw_mix_sources, list):
                raise ValueError("authenticated source_mix sources are invalid")
            mix: dict[str, int] = {}
            for index, raw_source in enumerate(raw_mix_sources):
                if not isinstance(raw_source, Mapping):
                    raise ValueError(
                        f"authenticated source_mix source {index} is invalid"
                    )
                source_id = _require_nonempty_string(
                    raw_source.get("source_id"),
                    f"source_mix.sources[{index}].source_id",
                )
                weight = raw_source.get("mix_basis_points")
                if (
                    isinstance(weight, bool)
                    or not isinstance(weight, int)
                    or weight <= 0
                ):
                    raise ValueError(
                        f"source_mix weight for {source_id!r} is invalid"
                    )
                if source_id in mix:
                    raise ValueError(
                        f"duplicate authenticated source_mix source: {source_id}"
                    )
                mix[source_id] = weight
            if set(mix) != source_ids or sum(mix.values()) != SOURCE_MIX_BASIS_POINTS:
                raise ValueError(
                    "authenticated source_mix does not cover source_map exactly"
                )
            mix_basis_points = tuple(sorted(mix.items()))
        elif data_contract is not None:
            raise ValueError(
                "prepared lineage has a data contract absent from extracted manifest"
            )

        manifest_root = manifest_path.parent
        resolved_outputs: dict[Path, str] = {}
        for output_path in selected_files:
            resolved = (manifest_root / output_path).resolve()
            try:
                resolved.relative_to(manifest_root)
            except ValueError as error:
                raise ValueError(
                    f"authenticated output escapes extracted manifest root: {output_path}"
                ) from error
            if resolved in resolved_outputs:
                raise ValueError(
                    "authenticated output paths alias after filesystem resolution: "
                    f"{resolved_outputs[resolved]} and {output_path}"
                )
            resolved_outputs[resolved] = output_path

        raw_prepared_shards = getattr(prepared, "shards", None)
        if not isinstance(raw_prepared_shards, tuple) or not raw_prepared_shards:
            raise ValueError("prepared manifest has no immutable shard tuple")
        mapped: list[AuthenticatedSourceShard] = []
        seen_prepared_outputs: set[str] = set()
        for index, entry in enumerate(raw_prepared_shards):
            source_path_text = _require_nonempty_string(
                getattr(entry, "source_path", None),
                f"prepared.shards[{index}].source_path",
            )
            source_path = Path(source_path_text).expanduser().resolve()
            output_path = resolved_outputs.get(source_path)
            if output_path is None:
                raise ValueError(
                    "prepared shard source_path is not an authenticated extracted output: "
                    f"{source_path}"
                )
            if output_path in seen_prepared_outputs:
                raise ValueError(
                    f"multiple prepared shards claim authenticated output: {output_path}"
                )
            seen_prepared_outputs.add(output_path)
            expected_digest = selected_files[output_path][1]
            prepared_digest = _require_sha256(
                getattr(entry, "source_sha256", None),
                f"prepared.shards[{index}].source_sha256",
            )
            if prepared_digest != expected_digest:
                raise ValueError(
                    f"prepared shard source SHA256 differs for {output_path}"
                )
            source_id = output_owners[output_path][0]
            sequence_count = getattr(entry, "sequence_count", None)
            if (
                isinstance(sequence_count, bool)
                or not isinstance(sequence_count, int)
                or sequence_count <= 0
            ):
                raise ValueError(
                    f"prepared.shards[{index}].sequence_count must be positive"
                )
            global_sample_start = _require_nonnegative_integer(
                getattr(entry, "global_sample_start", None),
                f"prepared.shards[{index}].global_sample_start",
            )
            mapped.append(
                AuthenticatedSourceShard(
                    source_id=source_id,
                    shard_id=_require_nonempty_string(
                        getattr(entry, "shard_id", None),
                        f"prepared.shards[{index}].shard_id",
                    ),
                    sequence_count=sequence_count,
                    global_sample_start=global_sample_start,
                    output_path=output_path,
                    output_sha256=expected_digest,
                )
            )
        if seen_prepared_outputs != set(selected_files):
            missing = sorted(set(selected_files) - seen_prepared_outputs)
            raise ValueError(
                "prepared corpus underfills authenticated train inventory: "
                + ", ".join(missing)
            )
        return cls(
            prepared_dataset_fingerprint=prepared_fingerprint,
            extracted_manifest_sha256=expected_manifest_sha,
            sequence_length=prepared_sequence_length,
            shards=tuple(mapped),
            mix_basis_points=mix_basis_points,
        )


def _affine_permutation(index: int, size: int, seed: int, epoch: int) -> int:
    """Stateless O(1) epoch permutation with no billion-element shuffle table."""

    if size == 1:
        return 0
    multiplier = _stable_integer("multiplier", seed, epoch) % size
    if multiplier == 0:
        multiplier = 1
    while math.gcd(multiplier, size) != 1:
        multiplier = (multiplier + 1) % size
        if multiplier == 0:
            multiplier = 1
    offset = _stable_integer("offset", seed, epoch) % size
    return (multiplier * index + offset) % size


@dataclass(frozen=True, slots=True)
class SampleReference:
    global_position: int
    epoch: int
    epoch_position: int
    flat_index: int
    shard_id: str
    shard_offset: int


@dataclass(frozen=True, slots=True)
class SourceMixSampleReference(SampleReference):
    """A global reference annotated with its deterministic source coordinate."""

    source_id: str
    source_position: int


@dataclass(frozen=True, slots=True)
class GlobalCursorState:
    schema_version: int
    dataset_fingerprint: str
    dataset_size: int
    seed: int
    next_global_sample: int
    committed_tokens: int
    shuffle: bool
    shuffle_algorithm: str = SHUFFLE_ALGORITHM

    def __post_init__(self) -> None:
        if self.schema_version != CURSOR_SCHEMA_VERSION:
            raise ValueError("unsupported cursor schema")
        if not self.dataset_fingerprint or self.dataset_size <= 0:
            raise ValueError("cursor dataset identity/size are invalid")
        if self.next_global_sample < 0 or self.committed_tokens < 0:
            raise ValueError("cursor counters cannot be negative")
        if not isinstance(self.shuffle, bool):
            raise TypeError("cursor shuffle must be a bool")
        if self.shuffle_algorithm != SHUFFLE_ALGORITHM:
            raise ValueError(f"unsupported cursor shuffle algorithm {self.shuffle_algorithm!r}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> GlobalCursorState:
        shuffle = payload["shuffle"]
        if not isinstance(shuffle, bool):
            raise TypeError("cursor shuffle must be a bool")
        return cls(
            schema_version=int(payload["schema_version"]),
            dataset_fingerprint=str(payload["dataset_fingerprint"]),
            dataset_size=int(payload["dataset_size"]),
            seed=int(payload["seed"]),
            next_global_sample=int(payload["next_global_sample"]),
            committed_tokens=int(payload["committed_tokens"]),
            shuffle=shuffle,
            shuffle_algorithm=str(payload.get("shuffle_algorithm", "legacy-global-affine")),
        )


class DeterministicGlobalCursor:
    """Plans global batches; only ``commit`` advances durable state.

    Every rank reconstructs the same global batch and selects its own slice.
    Consequently a checkpoint can resume with another world size without
    changing the global sample order.
    """

    def __init__(
        self,
        layout: DatasetLayout,
        *,
        seed: int,
        next_global_sample: int = 0,
        committed_tokens: int = 0,
        shuffle: bool = True,
    ) -> None:
        if next_global_sample < 0 or committed_tokens < 0:
            raise ValueError("cursor counters cannot be negative")
        self.layout = layout
        self.seed = seed
        self.next_global_sample = next_global_sample
        self.committed_tokens = committed_tokens
        self.shuffle = shuffle
        self._epoch_plans: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}

    def _epoch_shard_plan(self, epoch: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return a shuffled shard order and its cumulative logical lengths.

        Shuffling one flat billion-sample index makes nearly every consecutive
        sample land in another mmap file.  A full permutation does not require
        that I/O-hostile order: shuffle shard blocks first, then independently
        permute records inside each block.  This preserves exact resume/world-
        size semantics while keeping each shard handle hot for its whole block.
        """

        cached = self._epoch_plans.get(epoch)
        if cached is not None:
            return cached
        count = len(self.layout.shard_ids)
        order = tuple(
            _affine_permutation(position, count, self.seed, epoch) for position in range(count)
        )
        total = 0
        ends: list[int] = []
        for shard_index in order:
            total += self.layout.shard_lengths[shard_index]
            ends.append(total)
        result = (order, tuple(ends))
        # Training budgets normally span only a handful of corpus epochs. Keep
        # the tiny O(num_shards) plans so sample_at remains O(log num_shards).
        self._epoch_plans[epoch] = result
        return result

    def sample_at(self, global_position: int) -> SampleReference:
        if global_position < 0:
            raise IndexError(global_position)
        epoch, epoch_position = divmod(global_position, self.layout.size)
        if self.shuffle:
            order, logical_ends = self._epoch_shard_plan(epoch)
            logical_shard = bisect.bisect_right(logical_ends, epoch_position)
            logical_start = 0 if logical_shard == 0 else logical_ends[logical_shard - 1]
            shard_index = order[logical_shard]
            shard_length = self.layout.shard_lengths[shard_index]
            local_position = epoch_position - logical_start
            within_seed = _stable_integer(
                "within-shard",
                self.seed,
                epoch,
                self.layout.shard_ids[shard_index],
            )
            shard_offset = _affine_permutation(
                local_position,
                shard_length,
                within_seed,
                epoch,
            )
            original_start = 0 if shard_index == 0 else self.layout.cumulative_ends[shard_index - 1]
            flat_index = original_start + shard_offset
            shard_id = self.layout.shard_ids[shard_index]
        else:
            flat_index = epoch_position
            shard_id, shard_offset = self.layout.locate(flat_index)
        return SampleReference(
            global_position=global_position,
            epoch=epoch,
            epoch_position=epoch_position,
            flat_index=flat_index,
            shard_id=shard_id,
            shard_offset=shard_offset,
        )

    def plan_global_batch(self, global_batch_samples: int) -> tuple[SampleReference, ...]:
        if global_batch_samples <= 0:
            raise ValueError("global_batch_samples must be positive")
        return tuple(
            self.sample_at(position)
            for position in range(
                self.next_global_sample,
                self.next_global_sample + global_batch_samples,
            )
        )

    def plan_rank_batch(
        self,
        global_batch_samples: int,
        *,
        rank: int,
        world_size: int,
    ) -> tuple[SampleReference, ...]:
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank/world_size are invalid")
        if global_batch_samples % world_size:
            raise ValueError("global_batch_samples must be divisible by world_size")
        batch = self.plan_global_batch(global_batch_samples)
        return tuple(batch[index] for index in range(rank, len(batch), world_size))

    def commit(self, *, global_batch_samples: int, token_count: int) -> None:
        if global_batch_samples <= 0 or token_count <= 0:
            raise ValueError("invalid commit counters")
        self.next_global_sample += global_batch_samples
        self.committed_tokens += token_count

    def state_dict(self) -> dict[str, object]:
        return GlobalCursorState(
            schema_version=CURSOR_SCHEMA_VERSION,
            dataset_fingerprint=self.layout.fingerprint,
            dataset_size=self.layout.size,
            seed=self.seed,
            next_global_sample=self.next_global_sample,
            committed_tokens=self.committed_tokens,
            shuffle=self.shuffle,
            shuffle_algorithm=SHUFFLE_ALGORITHM,
        ).to_dict()

    @classmethod
    def from_state_dict(
        cls, layout: DatasetLayout, payload: Mapping[str, object]
    ) -> DeterministicGlobalCursor:
        state = GlobalCursorState.from_dict(payload)
        if state.schema_version != CURSOR_SCHEMA_VERSION:
            raise ValueError("unsupported cursor schema")
        if state.dataset_fingerprint != layout.fingerprint or state.dataset_size != layout.size:
            raise ValueError("cursor dataset fingerprint/size mismatch")
        return cls(
            layout,
            seed=state.seed,
            next_global_sample=state.next_global_sample,
            committed_tokens=state.committed_tokens,
            shuffle=state.shuffle,
        )


def _smooth_source_schedule(
    source_ids: tuple[str, ...],
    weights: Mapping[str, int],
    *,
    seed: int,
) -> tuple[str, ...]:
    """Return one exact, gcd-compressed smooth weighted round-robin period."""

    divisor = math.gcd(*(weights[source_id] for source_id in source_ids))
    quotas = {source_id: weights[source_id] // divisor for source_id in source_ids}
    period = sum(quotas.values())
    tie_order = sorted(
        source_ids,
        key=lambda source_id: (
            _stable_integer(SOURCE_MIX_ALGORITHM, "tie-order", seed, source_id),
            source_id,
        ),
    )
    tie_rank = {source_id: index for index, source_id in enumerate(tie_order)}
    current = dict.fromkeys(source_ids, 0)
    schedule: list[str] = []
    for _ in range(period):
        for source_id in source_ids:
            current[source_id] += quotas[source_id]
        selected = max(
            source_ids,
            key=lambda source_id: (current[source_id], -tie_rank[source_id]),
        )
        current[selected] -= period
        schedule.append(selected)
    if any(schedule.count(source_id) != quotas[source_id] for source_id in source_ids):
        raise RuntimeError("source interleave failed to realize exact basis-point quotas")
    if any(value != 0 for value in current.values()):
        raise RuntimeError("source interleave period did not return to its initial state")
    return tuple(schedule)


@dataclass(frozen=True, slots=True)
class _SourceMixProgress:
    next_global_sample: int
    committed_tokens: int
    committed_samples_by_source: tuple[tuple[str, int], ...]
    committed_tokens_by_source: tuple[tuple[str, int], ...]


class DeterministicSourceMixCursor:
    """Token-ratio-corrected source mixing over authenticated prepared shards.

    The short smooth weighted period is only the zero-history target and the
    deterministic tie-breaker.  Every new reference is selected from the
    largest cumulative token deficit, while simulating one full sequence for
    the still-unknown token count of that reference.  ``commit`` then records
    the actual non-padding token count for every planned reference; variable
    final sequence lengths are corrected by the next plan.

    Engine contract:

    1. Call ``plan_global_batch`` on every rank, or ``plan_rank_batch`` and keep
       ``pending_global_batch`` plus ``pending_plan_fingerprint``.
    2. Load the rank slice and all-gather per-reference valid-token counts in
       global reference order.
    3. Call ``commit`` with the exact pending global references, fingerprint,
       per-reference counts, their independently reduced per-source totals,
       and the independently reduced global total.

    A failed or interrupted step does not commit durable progress.  Replanning
    the same batch is idempotent; changing its size requires an explicit
    ``abort_pending_plan``.
    """

    def __init__(
        self,
        source_map: AuthenticatedSourceMap,
        weights_basis_points: Mapping[str, int] | None = None,
        *,
        seed: int,
        next_global_sample: int = 0,
        committed_tokens: int = 0,
        committed_samples_by_source: Mapping[str, int] | None = None,
        committed_tokens_by_source: Mapping[str, int] | None = None,
        shuffle: bool = True,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("source-mix seed must be an integer")
        next_sample = _require_nonnegative_integer(
            next_global_sample,
            "next_global_sample",
        )
        tokens = _require_nonnegative_integer(committed_tokens, "committed_tokens")
        if not isinstance(shuffle, bool):
            raise TypeError("source-mix shuffle must be a bool")
        if weights_basis_points is None:
            weights_basis_points = source_map.source_mix_weights
            if not weights_basis_points:
                raise ValueError(
                    "source-mix weights are required when authenticated lineage "
                    "does not carry a source_mix contract"
                )
        if not isinstance(weights_basis_points, Mapping):
            raise TypeError("source-mix basis-point weights must be a mapping")

        source_ids = source_map.source_ids
        if set(weights_basis_points) != set(source_ids):
            missing = sorted(set(source_ids) - set(weights_basis_points))
            extra = sorted(set(weights_basis_points) - set(source_ids))
            raise ValueError(
                "source-mix weights must cover the authenticated source map exactly; "
                f"missing={missing}, extra={extra}"
            )
        weights: dict[str, int] = {}
        for source_id in source_ids:
            value = weights_basis_points[source_id]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"source-mix weight for {source_id!r} must be positive integer "
                    "basis points"
                )
            weights[source_id] = value
        weight_total = sum(weights.values())
        if weight_total != SOURCE_MIX_BASIS_POINTS:
            raise ValueError(
                "source-mix basis-point weights underfill or overfill 10,000: "
                f"{weight_total}"
            )

        def normalized_counter(
            value: Mapping[str, int] | None,
            *,
            field_name: str,
            expected_total: int,
        ) -> dict[str, int]:
            raw: Mapping[str, int] = (
                dict.fromkeys(source_ids, 0) if value is None else value
            )
            if not isinstance(raw, Mapping) or set(raw) != set(source_ids):
                raise ValueError(
                    f"{field_name} must cover authenticated sources exactly"
                )
            result: dict[str, int] = {}
            for source_id in source_ids:
                result[source_id] = _require_nonnegative_integer(
                    raw[source_id],
                    f"{field_name}.{source_id}",
                )
            if sum(result.values()) != expected_total:
                raise ValueError(
                    f"{field_name} total differs from its global counter"
                )
            return result

        samples_by_source = normalized_counter(
            committed_samples_by_source,
            field_name="committed_samples_by_source",
            expected_total=next_sample,
        )
        tokens_by_source = normalized_counter(
            committed_tokens_by_source,
            field_name="committed_tokens_by_source",
            expected_total=tokens,
        )
        for source_id in source_ids:
            samples = samples_by_source[source_id]
            source_tokens = tokens_by_source[source_id]
            if source_tokens > samples * source_map.sequence_length:
                raise ValueError(
                    f"committed token count exceeds prepared capacity for {source_id}"
                )
            if (samples == 0) != (source_tokens == 0):
                raise ValueError(
                    f"committed sample/token zero-state differs for {source_id}"
                )

        self.source_map = source_map
        self.seed = seed
        self.shuffle = shuffle
        self._source_ids = source_ids
        self._weights = weights
        self._schedule = _smooth_source_schedule(source_ids, weights, seed=seed)
        self._source_cursors: dict[str, DeterministicGlobalCursor] = {}
        self._global_shard_starts: dict[str, int] = {}
        for source_id in source_ids:
            source_shards = source_map.shards_for_source(source_id)
            layout_fingerprint = _canonical_sha256(
                {
                    "source_map_algorithm": source_map.algorithm,
                    "prepared_dataset_fingerprint": source_map.prepared_dataset_fingerprint,
                    "source_id": source_id,
                    "shards": [item.to_dict() for item in source_shards],
                }
            )
            layout = DatasetLayout.from_shards(
                (
                    (item.shard_id, item.sequence_count)
                    for item in source_shards
                ),
                fingerprint=layout_fingerprint,
            )
            source_seed = _stable_integer(
                SOURCE_MIX_ALGORITHM,
                "source-cursor",
                seed,
                source_id,
            )
            self._source_cursors[source_id] = DeterministicGlobalCursor(
                layout,
                seed=source_seed,
                shuffle=shuffle,
            )
            for item in source_shards:
                self._global_shard_starts[item.shard_id] = item.global_sample_start
        self._progress = _SourceMixProgress(
            next_global_sample=next_sample,
            committed_tokens=tokens,
            committed_samples_by_source=tuple(sorted(samples_by_source.items())),
            committed_tokens_by_source=tuple(sorted(tokens_by_source.items())),
        )
        self._pending_plan: tuple[SourceMixSampleReference, ...] | None = None
        self._pending_plan_fingerprint: str | None = None

    @property
    def next_global_sample(self) -> int:
        return self._progress.next_global_sample

    @property
    def committed_tokens(self) -> int:
        return self._progress.committed_tokens

    @property
    def committed_samples_by_source(self) -> dict[str, int]:
        return dict(self._progress.committed_samples_by_source)

    @property
    def committed_tokens_by_source(self) -> dict[str, int]:
        return dict(self._progress.committed_tokens_by_source)

    @property
    def source_positions(self) -> dict[str, int]:
        """Compatibility name for per-source committed sample positions."""

        return self.committed_samples_by_source

    @property
    def token_deficit_numerators(self) -> dict[str, int]:
        return {
            source_id: (
                self._weights[source_id] * self.committed_tokens
                - self.committed_tokens_by_source[source_id]
                * SOURCE_MIX_BASIS_POINTS
            )
            for source_id in self._source_ids
        }

    @property
    def interleave_period(self) -> int:
        return len(self._schedule)

    @property
    def weights_basis_points(self) -> dict[str, int]:
        return dict(self._weights)

    @property
    def dataset_fingerprint(self) -> str:
        """Static source-map, token-ratio and algorithm identity."""

        return _canonical_sha256(
            {
                "schema_version": SOURCE_MIX_CURSOR_SCHEMA_VERSION,
                "kind": SOURCE_MIX_CURSOR_KIND,
                "algorithm": SOURCE_MIX_ALGORITHM,
                "source_map": self.source_map.to_dict(),
                "weights_basis_points": self.weights_basis_points,
                "basis_points_total": SOURCE_MIX_BASIS_POINTS,
                "nominal_tokens_per_sample": self.source_map.sequence_length,
                "seed": self.seed,
                "shuffle": self.shuffle,
                "shuffle_algorithm": SHUFFLE_ALGORITHM,
            }
        )

    def _critical_lineage_payload(self) -> dict[str, object]:
        return {
            "algorithm": SOURCE_MIX_ALGORITHM,
            "dataset_fingerprint": self.dataset_fingerprint,
            "source_map": self.source_map.to_dict(),
            "weights_basis_points": self.weights_basis_points,
            "next_global_sample": self.next_global_sample,
            "committed_tokens": self.committed_tokens,
            "committed_samples_by_source": self.committed_samples_by_source,
            "committed_tokens_by_source": self.committed_tokens_by_source,
        }

    @property
    def critical_lineage_fingerprint(self) -> str:
        return _canonical_sha256(self._critical_lineage_payload())

    def _reference(
        self,
        *,
        global_position: int,
        source_id: str,
        source_position: int,
    ) -> SourceMixSampleReference:
        local = self._source_cursors[source_id].sample_at(source_position)
        return SourceMixSampleReference(
            global_position=global_position,
            epoch=local.epoch,
            epoch_position=local.epoch_position,
            flat_index=self._global_shard_starts[local.shard_id] + local.shard_offset,
            shard_id=local.shard_id,
            shard_offset=local.shard_offset,
            source_id=source_id,
            source_position=source_position,
        )

    def _select_source(
        self,
        *,
        simulated_tokens_by_source: Mapping[str, int],
        simulated_total_tokens: int,
        global_position: int,
    ) -> str:
        target_total = simulated_total_tokens + self.source_map.sequence_length
        deficits = {
            source_id: (
                self._weights[source_id] * target_total
                - simulated_tokens_by_source[source_id]
                * SOURCE_MIX_BASIS_POINTS
            )
            for source_id in self._source_ids
        }
        largest = max(deficits.values())
        candidates = {
            source_id
            for source_id, deficit in deficits.items()
            if deficit == largest
        }
        initial_target = self._schedule[global_position % self.interleave_period]
        if initial_target in candidates:
            return initial_target
        return min(
            candidates,
            key=lambda source_id: (
                _stable_integer(
                    SOURCE_MIX_ALGORITHM,
                    "token-deficit-tie",
                    self.seed,
                    global_position,
                    source_id,
                ),
                source_id,
            ),
        )

    def _plan_fingerprint(
        self,
        references: Sequence[SourceMixSampleReference],
    ) -> str:
        return _canonical_sha256(
            {
                "algorithm": SOURCE_MIX_ALGORITHM,
                "critical_lineage_fingerprint": self.critical_lineage_fingerprint,
                "references": [asdict(reference) for reference in references],
            }
        )

    @property
    def pending_global_batch(self) -> tuple[SourceMixSampleReference, ...]:
        return self._pending_plan or ()

    @property
    def pending_plan_fingerprint(self) -> str | None:
        return self._pending_plan_fingerprint

    def abort_pending_plan(self) -> None:
        """Discard only ephemeral work; durable cursor progress is unchanged."""

        self._pending_plan = None
        self._pending_plan_fingerprint = None

    def plan_global_batch(
        self,
        global_batch_samples: int,
    ) -> tuple[SourceMixSampleReference, ...]:
        if (
            isinstance(global_batch_samples, bool)
            or not isinstance(global_batch_samples, int)
            or global_batch_samples <= 0
        ):
            raise ValueError("global_batch_samples must be a positive integer")
        if self._pending_plan is not None:
            if len(self._pending_plan) != global_batch_samples:
                raise ValueError(
                    "another source-mix batch is pending; commit or abort it first"
                )
            return self._pending_plan

        simulated_tokens = self.committed_tokens_by_source
        simulated_total = self.committed_tokens
        planned_samples = dict.fromkeys(self._source_ids, 0)
        committed_samples = self.committed_samples_by_source
        references: list[SourceMixSampleReference] = []
        for batch_offset in range(global_batch_samples):
            global_position = self.next_global_sample + batch_offset
            source_id = self._select_source(
                simulated_tokens_by_source=simulated_tokens,
                simulated_total_tokens=simulated_total,
                global_position=global_position,
            )
            source_position = (
                committed_samples[source_id] + planned_samples[source_id]
            )
            references.append(
                self._reference(
                    global_position=global_position,
                    source_id=source_id,
                    source_position=source_position,
                )
            )
            planned_samples[source_id] += 1
            simulated_tokens[source_id] += self.source_map.sequence_length
            simulated_total += self.source_map.sequence_length
        pending = tuple(references)
        self._pending_plan = pending
        self._pending_plan_fingerprint = self._plan_fingerprint(pending)
        return pending

    def plan_rank_batch(
        self,
        global_batch_samples: int,
        *,
        rank: int,
        world_size: int,
    ) -> tuple[SourceMixSampleReference, ...]:
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank/world_size are invalid")
        if global_batch_samples % world_size:
            raise ValueError("global_batch_samples must be divisible by world_size")
        batch = self.plan_global_batch(global_batch_samples)
        return tuple(batch[index] for index in range(rank, len(batch), world_size))

    def _validated_commit_progress(
        self,
        *,
        planned_references: Sequence[SourceMixSampleReference],
        plan_fingerprint: str,
        valid_tokens_per_reference: Sequence[int],
        valid_tokens_by_source: Mapping[str, int],
        token_count: int,
    ) -> _SourceMixProgress:
        """Validate one global optimizer batch and derive its next durable state."""

        if self._pending_plan is None or self._pending_plan_fingerprint is None:
            raise ValueError("source-mix commit has no pending plan")
        references = tuple(planned_references)
        if references != self._pending_plan:
            raise ValueError("source-mix commit references differ from pending plan")
        if (
            _require_sha256(plan_fingerprint, "plan_fingerprint")
            != self._pending_plan_fingerprint
        ):
            raise ValueError("source-mix commit plan fingerprint mismatch")
        per_reference = tuple(valid_tokens_per_reference)
        if len(per_reference) != len(references):
            raise ValueError(
                "valid_tokens_per_reference length differs from planned references"
            )
        for index, valid_tokens in enumerate(per_reference):
            if (
                isinstance(valid_tokens, bool)
                or not isinstance(valid_tokens, int)
                or not 1 <= valid_tokens <= self.source_map.sequence_length
            ):
                raise ValueError(
                    f"valid_tokens_per_reference[{index}] is outside prepared capacity"
                )
        if not isinstance(valid_tokens_by_source, Mapping) or set(
            valid_tokens_by_source
        ) != set(self._source_ids):
            raise ValueError(
                "valid_tokens_by_source must cover authenticated sources exactly"
            )
        normalized_by_source: dict[str, int] = {}
        for source_id in self._source_ids:
            normalized_by_source[source_id] = _require_nonnegative_integer(
                valid_tokens_by_source[source_id],
                f"valid_tokens_by_source.{source_id}",
            )
        aggregated = dict.fromkeys(self._source_ids, 0)
        committed_samples = self.committed_samples_by_source
        planned_sample_counts = dict.fromkeys(self._source_ids, 0)
        for reference, valid_tokens in zip(
            references,
            per_reference,
            strict=True,
        ):
            aggregated[reference.source_id] += valid_tokens
            planned_sample_counts[reference.source_id] += 1
        if aggregated != normalized_by_source:
            raise ValueError(
                "valid_tokens_by_source differs from planned reference aggregation"
            )
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count <= 0
            or token_count != sum(per_reference)
            or token_count != sum(normalized_by_source.values())
        ):
            raise ValueError(
                "source-mix global token total differs from reference/source totals"
            )

        committed_tokens_by_source = self.committed_tokens_by_source
        for source_id in self._source_ids:
            committed_samples[source_id] += planned_sample_counts[source_id]
            committed_tokens_by_source[source_id] += normalized_by_source[source_id]
        return _SourceMixProgress(
            next_global_sample=self.next_global_sample + len(references),
            committed_tokens=self.committed_tokens + token_count,
            committed_samples_by_source=tuple(sorted(committed_samples.items())),
            committed_tokens_by_source=tuple(
                sorted(committed_tokens_by_source.items())
            ),
        )

    def validate_commit(
        self,
        *,
        planned_references: Sequence[SourceMixSampleReference],
        plan_fingerprint: str,
        valid_tokens_per_reference: Sequence[int],
        valid_tokens_by_source: Mapping[str, int],
        token_count: int,
    ) -> None:
        """Fail closed before an optimizer update without advancing the cursor."""

        self._validated_commit_progress(
            planned_references=planned_references,
            plan_fingerprint=plan_fingerprint,
            valid_tokens_per_reference=valid_tokens_per_reference,
            valid_tokens_by_source=valid_tokens_by_source,
            token_count=token_count,
        )

    def commit(
        self,
        *,
        planned_references: Sequence[SourceMixSampleReference],
        plan_fingerprint: str,
        valid_tokens_per_reference: Sequence[int],
        valid_tokens_by_source: Mapping[str, int],
        token_count: int,
    ) -> None:
        """Atomically commit one prevalidated, successful optimizer batch."""

        self._progress = self._validated_commit_progress(
            planned_references=planned_references,
            plan_fingerprint=plan_fingerprint,
            valid_tokens_per_reference=valid_tokens_per_reference,
            valid_tokens_by_source=valid_tokens_by_source,
            token_count=token_count,
        )
        self._pending_plan = None
        self._pending_plan_fingerprint = None

    def _state_payload(self) -> dict[str, object]:
        return {
            "schema_version": SOURCE_MIX_CURSOR_SCHEMA_VERSION,
            "kind": SOURCE_MIX_CURSOR_KIND,
            "algorithm": SOURCE_MIX_ALGORITHM,
            "source_map_algorithm": SOURCE_MAP_ALGORITHM,
            "prepared_dataset_fingerprint": (
                self.source_map.prepared_dataset_fingerprint
            ),
            "source_map": self.source_map.to_dict(),
            "weights_basis_points": self.weights_basis_points,
            "basis_points_total": SOURCE_MIX_BASIS_POINTS,
            "nominal_tokens_per_sample": self.source_map.sequence_length,
            "initial_interleave_period": self.interleave_period,
            "seed": self.seed,
            "next_global_sample": self.next_global_sample,
            "committed_tokens": self.committed_tokens,
            "committed_samples_by_source": self.committed_samples_by_source,
            "committed_tokens_by_source": self.committed_tokens_by_source,
            "token_deficit_numerators": self.token_deficit_numerators,
            "shuffle": self.shuffle,
            "shuffle_algorithm": SHUFFLE_ALGORITHM,
            "dataset_fingerprint": self.dataset_fingerprint,
            "critical_lineage_fingerprint": self.critical_lineage_fingerprint,
        }

    @property
    def state_fingerprint(self) -> str:
        return _canonical_sha256(self._state_payload())

    def state_dict(self) -> dict[str, object]:
        payload = self._state_payload()
        payload["state_fingerprint"] = _canonical_sha256(payload)
        return payload

    @classmethod
    def from_state_dict(
        cls,
        source_map: AuthenticatedSourceMap,
        weights_basis_points: Mapping[str, int] | None,
        payload: Mapping[str, object],
    ) -> DeterministicSourceMixCursor:
        expected_fields = {
            "schema_version",
            "kind",
            "algorithm",
            "source_map_algorithm",
            "prepared_dataset_fingerprint",
            "source_map",
            "weights_basis_points",
            "basis_points_total",
            "nominal_tokens_per_sample",
            "initial_interleave_period",
            "seed",
            "next_global_sample",
            "committed_tokens",
            "committed_samples_by_source",
            "committed_tokens_by_source",
            "token_deficit_numerators",
            "shuffle",
            "shuffle_algorithm",
            "dataset_fingerprint",
            "critical_lineage_fingerprint",
            "state_fingerprint",
        }
        if set(payload) != expected_fields:
            raise ValueError("source-mix cursor state fields differ from schema v2")
        state_fingerprint = _require_sha256(
            payload.get("state_fingerprint"),
            "state_fingerprint",
        )
        unsigned_payload = {
            key: value for key, value in payload.items() if key != "state_fingerprint"
        }
        if _canonical_sha256(unsigned_payload) != state_fingerprint:
            raise ValueError("source-mix cursor state fingerprint mismatch")
        if (
            payload.get("schema_version") != SOURCE_MIX_CURSOR_SCHEMA_VERSION
            or payload.get("kind") != SOURCE_MIX_CURSOR_KIND
            or payload.get("algorithm") != SOURCE_MIX_ALGORITHM
            or payload.get("source_map_algorithm") != SOURCE_MAP_ALGORITHM
        ):
            raise ValueError("unsupported source-mix cursor schema or algorithm")
        if payload.get("source_map") != source_map.to_dict():
            raise ValueError("source-mix checkpoint source map changed")
        if payload.get("prepared_dataset_fingerprint") != (
            source_map.prepared_dataset_fingerprint
        ):
            raise ValueError("source-mix checkpoint prepared fingerprint changed")

        raw_weights = payload.get("weights_basis_points")
        if not isinstance(raw_weights, Mapping):
            raise ValueError("source-mix checkpoint weights are invalid")
        expected_weights = (
            source_map.source_mix_weights
            if weights_basis_points is None
            else dict(weights_basis_points)
        )
        if dict(raw_weights) != expected_weights:
            raise ValueError("source-mix checkpoint basis-point weights changed")
        seed = payload.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("source-mix checkpoint seed is invalid")
        next_sample = _require_nonnegative_integer(
            payload.get("next_global_sample"),
            "next_global_sample",
        )
        tokens = _require_nonnegative_integer(
            payload.get("committed_tokens"),
            "committed_tokens",
        )
        raw_samples_by_source = payload.get("committed_samples_by_source")
        raw_tokens_by_source = payload.get("committed_tokens_by_source")
        if not isinstance(raw_samples_by_source, Mapping) or not isinstance(
            raw_tokens_by_source,
            Mapping,
        ):
            raise ValueError("source-mix checkpoint per-source counters are invalid")
        shuffle = payload.get("shuffle")
        if not isinstance(shuffle, bool):
            raise ValueError("source-mix checkpoint shuffle is invalid")
        result = cls(
            source_map,
            expected_weights,
            seed=seed,
            next_global_sample=next_sample,
            committed_tokens=tokens,
            committed_samples_by_source={
                str(source_id): value
                for source_id, value in raw_samples_by_source.items()
            },
            committed_tokens_by_source={
                str(source_id): value
                for source_id, value in raw_tokens_by_source.items()
            },
            shuffle=shuffle,
        )
        expected_payload = result._state_payload()
        if unsigned_payload != expected_payload:
            raise ValueError("source-mix cursor state disagrees with derived state")
        return result


class DeterministicCooldownCursor:
    """Two immutable layouts with one token-coordinate phase transition.

    The primary and cooldown positions are persisted independently. The active
    phase is derived only from globally committed tokens, so changing world
    size changes rank partitioning but never sample order or transition state.
    A whole optimizer batch that starts below the threshold remains primary;
    the next batch switches after the commit crosses the threshold.
    """

    def __init__(
        self,
        primary_layout: DatasetLayout,
        cooldown_layout: DatasetLayout,
        *,
        seed: int,
        cooldown_start_tokens: int,
        shuffle: bool = True,
        _primary_cursor: DeterministicGlobalCursor | None = None,
        _cooldown_cursor: DeterministicGlobalCursor | None = None,
        _next_global_sample: int = 0,
        _committed_tokens: int = 0,
    ) -> None:
        if cooldown_start_tokens <= 0:
            raise ValueError("quality cooldown start tokens must be positive")
        if primary_layout.fingerprint == cooldown_layout.fingerprint:
            raise ValueError("quality cooldown requires distinct dataset fingerprints")
        self.primary_layout = primary_layout
        self.cooldown_layout = cooldown_layout
        self.seed = seed
        self.cooldown_start_tokens = cooldown_start_tokens
        self.shuffle = shuffle
        self._primary_cursor = _primary_cursor or DeterministicGlobalCursor(
            primary_layout, seed=seed, shuffle=shuffle
        )
        self._cooldown_cursor = _cooldown_cursor or DeterministicGlobalCursor(
            cooldown_layout, seed=seed, shuffle=shuffle
        )
        self.next_global_sample = _next_global_sample
        self.committed_tokens = _committed_tokens
        self._validate_state()

    @property
    def active_phase(self) -> str:
        return "cooldown" if self.committed_tokens >= self.cooldown_start_tokens else "primary"

    @property
    def active_layout(self) -> DatasetLayout:
        return self.cooldown_layout if self.active_phase == "cooldown" else self.primary_layout

    @property
    def phase_next_global_sample(self) -> int:
        return self._active_cursor.next_global_sample

    @property
    def _active_cursor(self) -> DeterministicGlobalCursor:
        return self._cooldown_cursor if self.active_phase == "cooldown" else self._primary_cursor

    @property
    def dataset_fingerprint(self) -> str:
        payload = json.dumps(
            {
                "kind": QUALITY_COOLDOWN_CURSOR_KIND,
                "primary": self.primary_layout.fingerprint,
                "cooldown": self.cooldown_layout.fingerprint,
                "start_tokens": self.cooldown_start_tokens,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _validate_state(self) -> None:
        cursors = (self._primary_cursor, self._cooldown_cursor)
        if any(cursor.seed != self.seed or cursor.shuffle != self.shuffle for cursor in cursors):
            raise ValueError("quality cooldown phase cursor seed/shuffle mismatch")
        if self._primary_cursor.layout != self.primary_layout:
            raise ValueError("quality cooldown primary layout mismatch")
        if self._cooldown_cursor.layout != self.cooldown_layout:
            raise ValueError("quality cooldown layout mismatch")
        if self.next_global_sample != sum(cursor.next_global_sample for cursor in cursors):
            raise ValueError("quality cooldown global sample counter mismatch")
        if self.committed_tokens != sum(cursor.committed_tokens for cursor in cursors):
            raise ValueError("quality cooldown committed token counter mismatch")
        if self.committed_tokens < self.cooldown_start_tokens and (
            self._cooldown_cursor.next_global_sample != 0
            or self._cooldown_cursor.committed_tokens != 0
        ):
            raise ValueError("quality cooldown cursor advanced before its phase")
        if self._cooldown_cursor.next_global_sample > 0 and (
            self._primary_cursor.committed_tokens < self.cooldown_start_tokens
        ):
            raise ValueError("quality cooldown cursor advanced before primary threshold")

    def plan_global_batch(self, global_batch_samples: int) -> tuple[SampleReference, ...]:
        local = self._active_cursor.plan_global_batch(global_batch_samples)
        return tuple(
            replace(reference, global_position=self.next_global_sample + index)
            for index, reference in enumerate(local)
        )

    def plan_rank_batch(
        self,
        global_batch_samples: int,
        *,
        rank: int,
        world_size: int,
    ) -> tuple[SampleReference, ...]:
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank/world_size are invalid")
        if global_batch_samples % world_size:
            raise ValueError("global_batch_samples must be divisible by world_size")
        batch = self.plan_global_batch(global_batch_samples)
        return tuple(batch[index] for index in range(rank, len(batch), world_size))

    def commit(self, *, global_batch_samples: int, token_count: int) -> None:
        active = self._active_cursor
        active.commit(
            global_batch_samples=global_batch_samples,
            token_count=token_count,
        )
        self.next_global_sample += global_batch_samples
        self.committed_tokens += token_count
        self._validate_state()

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": QUALITY_COOLDOWN_CURSOR_SCHEMA_VERSION,
            "kind": QUALITY_COOLDOWN_CURSOR_KIND,
            "dataset_fingerprint": self.dataset_fingerprint,
            "dataset_size": self.primary_layout.size + self.cooldown_layout.size,
            "seed": self.seed,
            "next_global_sample": self.next_global_sample,
            "committed_tokens": self.committed_tokens,
            "shuffle": self.shuffle,
            "shuffle_algorithm": SHUFFLE_ALGORITHM,
            "cooldown_start_tokens": self.cooldown_start_tokens,
            "active_phase": self.active_phase,
            "primary_cursor": self._primary_cursor.state_dict(),
            "cooldown_cursor": self._cooldown_cursor.state_dict(),
        }

    @classmethod
    def from_state_dict(
        cls,
        primary_layout: DatasetLayout,
        cooldown_layout: DatasetLayout,
        payload: Mapping[str, object],
        *,
        cooldown_start_tokens: int,
    ) -> DeterministicCooldownCursor:
        if (
            int(payload.get("schema_version", -1)) != QUALITY_COOLDOWN_CURSOR_SCHEMA_VERSION
            or payload.get("kind") != QUALITY_COOLDOWN_CURSOR_KIND
        ):
            raise ValueError("unsupported quality cooldown cursor schema")
        if int(payload.get("cooldown_start_tokens", -1)) != cooldown_start_tokens:
            raise ValueError("quality cooldown transition token changed on resume")
        primary_payload = payload.get("primary_cursor")
        cooldown_payload = payload.get("cooldown_cursor")
        if not isinstance(primary_payload, Mapping) or not isinstance(cooldown_payload, Mapping):
            raise ValueError("quality cooldown checkpoint is missing phase cursors")
        primary = DeterministicGlobalCursor.from_state_dict(primary_layout, primary_payload)
        cooldown = DeterministicGlobalCursor.from_state_dict(cooldown_layout, cooldown_payload)
        shuffle = payload.get("shuffle")
        if not isinstance(shuffle, bool):
            raise TypeError("quality cooldown cursor shuffle must be a bool")
        result = cls(
            primary_layout,
            cooldown_layout,
            seed=int(payload["seed"]),
            cooldown_start_tokens=cooldown_start_tokens,
            shuffle=shuffle,
            _primary_cursor=primary,
            _cooldown_cursor=cooldown,
            _next_global_sample=int(payload["next_global_sample"]),
            _committed_tokens=int(payload["committed_tokens"]),
        )
        if payload.get("dataset_fingerprint") != result.dataset_fingerprint:
            raise ValueError("quality cooldown composite dataset fingerprint changed")
        if int(payload.get("dataset_size", -1)) != (primary_layout.size + cooldown_layout.size):
            raise ValueError("quality cooldown composite dataset size changed")
        if payload.get("active_phase") != result.active_phase:
            raise ValueError("quality cooldown active phase disagrees with committed tokens")
        return result


def gradient_accumulation_steps(
    *,
    global_batch_tokens: int,
    world_size: int,
    microbatch_tokens_per_rank: int,
) -> int:
    """Compute exact accumulation needed to preserve global batch tokens."""

    if global_batch_tokens <= 0 or world_size <= 0 or microbatch_tokens_per_rank <= 0:
        raise ValueError("batch geometry values must be positive")
    denominator = world_size * microbatch_tokens_per_rank
    steps, remainder = divmod(global_batch_tokens, denominator)
    if remainder or steps == 0:
        raise ValueError(
            "global_batch_tokens must be an exact positive multiple of "
            "world_size * microbatch_tokens_per_rank"
        )
    return steps
