"""Pinned, resumable Hugging Face Parquet-to-JSONL corpus extraction.

This module is deliberately independent from the training engine.  It resolves
native Parquet objects at immutable Hub commits, reads them with HTTP Range
requests, and transactionally emits the JSONL files consumed by ``data
prepare``.  No function in this module constructs an optimizer or performs a
training step.
"""

from __future__ import annotations

import fnmatch
import gzip
import hashlib
import io
import json
import math
import os
import re
import struct
import threading
import unicodedata
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import quote
from urllib.request import Request

from aiohttp import ClientError

from ..io.download import ArtifactSpec, DownloadManager
from ..io.locking import FileLock
from ..io.proxy import ProxySettings, check_proxy_connectivity
from ..progress import TaskProgress
from ..utils import atomic_write_json, atomic_write_text, sha256_file
from .shards import (
    ShardStateError,
    ShardTransaction,
    is_shard_complete,
    read_complete_marker,
)

SOURCE_RECIPE_SCHEMA_VERSION = 1
SOURCE_RECIPE_SCHEMA_VERSION_V2 = 2
SUPPORTED_SOURCE_RECIPE_SCHEMA_VERSIONS = frozenset(
    {SOURCE_RECIPE_SCHEMA_VERSION, SOURCE_RECIPE_SCHEMA_VERSION_V2}
)
RESOLVED_SOURCE_LOCK_SCHEMA_VERSION = 1
EXTRACTED_CORPUS_SCHEMA_VERSION = 1
DERIVED_JSONL_SCHEMA_VERSION = 1
DERIVED_JSONL_ROW_INDEX_STRIDE = 4096
SOURCE_MAP_AUDIT_SCHEMA_VERSION = 1
SOURCE_MAP_AUDIT_ALGORITHM = "authenticated-extracted-output-map-v1"
SOURCE_MIX_AUDIT_SCHEMA_VERSION = 1
SOURCE_MIX_AUDIT_ALGORITHM = "token-deficit-corrected-source-mix-bp-v2"
SOURCE_MIX_BASIS_POINTS = 10_000
EXTRACTED_CONTRACT_IDENTITY_KEYS = (
    "source_map",
    "source_mix",
    "format_audit",
    "license_audit",
    "materialization_audit",
)
SCHEMA_V2_REQUIRED_IMPLEMENTATIONS = (
    "schema-v2 validation",
    "immutable jsonl_gzip download with LFS SHA256 verification",
    "streaming gzip JSON Lines extraction",
    "dotted-field access and canonical license normalization",
    "source-stratified optimizer-batch mixing",
)
EXTRACTOR_SOURCE_SHA256 = sha256_file(Path(__file__))
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SAFE_PROFILE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_COMMITTED_CHUNK = re.compile(r"^chunk-[0-9]{6}$")
_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    re.IGNORECASE,
)
_AWS_KEY = re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])")
_GITHUB_TOKEN = re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{32,255}")
_GENERIC_SECRET = re.compile(
    r"(?i)(?:api[_-]?key|secret[_-]?key|access[_-]?token|password)"
    r"\s*[:=]\s*['\"][A-Za-z0-9_./+=-]{20,}['\"]"
)
_EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])")
_IPV4 = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
_CHINESE_PHONE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
_CHINESE_ID = re.compile(
    r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])"
    r"(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx](?!\d)"
)
_US_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")


class DataSourceError(RuntimeError):
    """Raised when source identity, schema, or extraction state is unsafe."""


def verified_data_pipeline_implementations() -> tuple[str, ...]:
    """Return capabilities that can be structurally verified in this source tree.

    Activation is deliberately derived from executable symbols rather than a
    hand-maintained ``implemented=true`` flag in the recipe.  A partial merge
    therefore leaves schema-v2 recipes inspectable but not runnable.
    """

    verified: list[str] = []
    if (
        callable(globals().get("load_base_data_recipe"))
        and callable(globals().get("load_resolved_source_lock"))
    ):
        verified.append(SCHEMA_V2_REQUIRED_IMPLEMENTATIONS[0])
    if (
        callable(globals().get("download_locked_source_file"))
        and callable(globals().get("materialize_jsonl_gzip_artifact"))
        and globals().get("ArtifactSpec") is not None
    ):
        verified.append(SCHEMA_V2_REQUIRED_IMPLEMENTATIONS[1])
    if (
        callable(globals().get("iter_gzip_jsonl_rows"))
        and callable(globals().get("iter_local_jsonl_rows"))
        and globals().get("DERIVED_JSONL_ROW_INDEX_STRIDE") == 4096
    ):
        verified.append(SCHEMA_V2_REQUIRED_IMPLEMENTATIONS[2])
    if (
        callable(globals().get("get_dotted_field"))
        and callable(globals().get("normalize_license"))
    ):
        verified.append(SCHEMA_V2_REQUIRED_IMPLEMENTATIONS[3])
    try:
        from .cursor import (
            SOURCE_MIX_ALGORITHM,
            AuthenticatedSourceMap,
            DeterministicSourceMixCursor,
        )
        from .cursor import (
            SOURCE_MIX_BASIS_POINTS as CURSOR_SOURCE_MIX_BASIS_POINTS,
        )
    except (ImportError, AttributeError):
        pass
    else:
        if (
            AuthenticatedSourceMap is not None
            and DeterministicSourceMixCursor is not None
            and SOURCE_MIX_ALGORITHM == SOURCE_MIX_AUDIT_ALGORITHM
            and CURSOR_SOURCE_MIX_BASIS_POINTS == SOURCE_MIX_BASIS_POINTS
        ):
            verified.append(SCHEMA_V2_REQUIRED_IMPLEMENTATIONS[4])
    return tuple(verified)


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataSourceError(f"{field} must be a non-empty string")
    return value.strip()


def _require_positive_int(value: object, field: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataSourceError(f"{field} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise DataSourceError(f"{field} must be >= {minimum}")
    return value


def _safe_relative_pattern(
    value: object,
    field: str,
    *,
    storage_format: str = "parquet",
) -> str:
    pattern = _require_string(value, field)
    path = PurePosixPath(pattern)
    expected_suffix = {
        "parquet": ".parquet",
        "jsonl_gzip": ".json.gz",
    }.get(storage_format)
    if expected_suffix is None:
        raise DataSourceError(f"unsupported storage format in {field}: {storage_format!r}")
    if path.is_absolute() or ".." in path.parts or not pattern.endswith(expected_suffix):
        raise DataSourceError(
            f"unsafe {storage_format} path/pattern in {field}: {pattern!r}"
        )
    return path.as_posix()


def _string_tuple(value: object, field: str, *, required: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise DataSourceError(f"{field} must be a list")
    result = tuple(_require_string(item, field) for item in value)
    if required and not result:
        raise DataSourceError(f"{field} cannot be empty")
    if len(set(result)) != len(result):
        raise DataSourceError(f"{field} contains duplicates")
    return result


def _field_path(value: object, field: str) -> str:
    path = _require_string(value, field)
    parts = path.split(".")
    if any(
        not part
        or part in {".", ".."}
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", part)
        for part in parts
    ):
        raise DataSourceError(f"unsafe dotted field path in {field}: {path!r}")
    return path


_MISSING = object()


def get_dotted_field(
    row: Mapping[str, object],
    field: str,
    default: object | None = None,
) -> object:
    """Read a possibly dotted field from nested row mappings.

    An exact top-level key wins.  This preserves v1 compatibility for Parquet
    schemas that legitimately contain a dot in a column name while allowing
    schema-v2 JSON objects such as ``metadata.license``.
    """

    if field in row:
        return row[field]
    current: object = row
    for part in field.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


_LICENSE_ALIASES = {
    "public domain": "public-domain",
    "public-domain": "public-domain",
    "pd": "public-domain",
    "pdm-1.0": "public-domain",
    "cc-pddc": "public-domain",
    "cc0": "cc0-1.0",
    "cc-0": "cc0-1.0",
    "cc0-1.0": "cc0-1.0",
    "creative commons zero v1.0 universal": "cc0-1.0",
    "mit": "mit",
    "mit license": "mit",
    "apache-2.0": "apache-2.0",
    "apache 2.0": "apache-2.0",
    "apache license 2.0": "apache-2.0",
    "apache license, version 2.0": "apache-2.0",
    "bsd-2-clause": "bsd-2-clause",
    "bsd 2-clause": "bsd-2-clause",
    "bsd 2 clause": "bsd-2-clause",
    "bsd-3-clause": "bsd-3-clause",
    "bsd 3-clause": "bsd-3-clause",
    "bsd 3 clause": "bsd-3-clause",
    "isc": "isc",
    "isc license": "isc",
}
_CC_BY_PATTERN = re.compile(
    r"^(?:cc|creative commons)(?: license)?(?: attribution)?[- ]+by[- ]+"
    r"(?:v(?:ersion)?[- ]*)?(1\.0|2\.0|2\.5|3\.0|4\.0)(?: international)?$"
)
_CC_ATTRIBUTION_PATTERN = re.compile(
    r"^(?:cc|creative commons)(?: license)?[- ]+attribution[- ]+"
    r"(?:v(?:ersion)?[- ]*)?(1\.0|2\.0|2\.5|3\.0|4\.0)(?: international)?$"
)
_CC_BY_URL_PATTERN = re.compile(
    r"^https?://creativecommons\.org/licenses/by/(1\.0|2\.0|2\.5|3\.0|4\.0)"
    r"(?:/deed(?:\.[a-z-]+)?)?$"
)
_CC_ATTRIBUTION_LABEL_URL_PATTERN = re.compile(
    r"^creative commons\s*-\s*attribution\s*-\s*"
    r"https?://creativecommons\.org/licenses/by/(1\.0|2\.0|2\.5|3\.0|4\.0)"
    r"(?:/deed(?:\.[a-z-]+)?)?$"
)
_CC0_URL_PATTERN = re.compile(
    r"^https?://creativecommons\.org/publicdomain/zero/1\.0"
    r"(?:/deed(?:\.[a-z-]+)?)?$"
)
_PUBLIC_DOMAIN_URL_PATTERN = re.compile(
    r"^https?://creativecommons\.org/publicdomain/mark/1\.0"
    r"(?:/deed(?:\.[a-z-]+)?)?$"
)


def normalize_license(value: object) -> str | None:
    """Normalize one unambiguous permissive-license label.

    Composite SPDX expressions and labels with extra clauses deliberately do
    not normalize.  Callers must still compare the result with the source's
    explicit allowlist.
    """

    if not isinstance(value, str):
        return None
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    if not normalized:
        return None
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.rstrip("/")
    if normalized in _LICENSE_ALIASES:
        return _LICENSE_ALIASES[normalized]
    match = _CC_BY_PATTERN.fullmatch(normalized)
    if match is not None:
        return f"cc-by-{match.group(1)}"
    match = _CC_ATTRIBUTION_PATTERN.fullmatch(normalized)
    if match is not None:
        return f"cc-by-{match.group(1)}"
    match = _CC_BY_URL_PATTERN.fullmatch(normalized)
    if match is not None:
        return f"cc-by-{match.group(1)}"
    match = _CC_ATTRIBUTION_LABEL_URL_PATTERN.fullmatch(normalized)
    if match is not None:
        return f"cc-by-{match.group(1)}"
    if _CC0_URL_PATTERN.fullmatch(normalized):
        return "cc0-1.0"
    if _PUBLIC_DOMAIN_URL_PATTERN.fullmatch(normalized):
        return "public-domain"
    return None


@dataclass(frozen=True, slots=True)
class LockedSourceFile:
    path: str
    size: int
    sha256: str

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
        *,
        source_id: str,
        storage_format: str,
    ) -> LockedSourceFile:
        path = _safe_relative_pattern(
            value.get("path"),
            f"{source_id}.locked_files.path",
            storage_format=storage_format,
        )
        if any(character in path for character in "*?["):
            raise DataSourceError(f"{source_id}.locked_files path must be exact: {path!r}")
        size = _require_positive_int(value.get("size"), f"{source_id}.{path}.size")
        digest = _require_string(
            value.get("sha256"), f"{source_id}.{path}.sha256"
        ).lower()
        if not _SHA256.fullmatch(digest):
            raise DataSourceError(f"{source_id}.{path}.sha256 is not a SHA256")
        return cls(path=path, size=size, sha256=digest)


@dataclass(frozen=True, slots=True)
class RowFilter:
    field: str
    operator: str
    values: tuple[str, ...] = ()
    threshold: int | float | None = None

    def matches(self, row: Mapping[str, object]) -> bool:
        value = get_dotted_field(row, self.field, _MISSING)
        if self.operator in {"in", "not_in"}:
            contained = isinstance(value, str) and value in self.values
            return contained if self.operator == "in" else not contained
        if not _is_finite_number(value):
            return False
        if self.threshold is None:
            raise DataSourceError("numeric row filter lacks a threshold")
        if self.operator == "gte":
            return value >= self.threshold
        if self.operator == "lte":
            return value <= self.threshold
        raise DataSourceError(f"unsupported row filter operator: {self.operator!r}")

    def to_contract(self) -> dict[str, object]:
        """Return the stable JSON contract without changing legacy membership filters."""

        if self.operator in {"in", "not_in"}:
            return {
                "field": self.field,
                "operator": self.operator,
                "values": list(self.values),
            }
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.threshold,
        }


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not isinstance(value, float) or math.isfinite(value)


def _require_finite_number(value: object, field: str) -> int | float:
    if not _is_finite_number(value):
        raise DataSourceError(f"{field} must be a finite JSON number")
    # Canonicalize negative zero so equivalent thresholds have identical contracts.
    if isinstance(value, float) and value == 0:
        return 0.0
    return value


def _parse_row_filters(value: object, source_id: str) -> tuple[RowFilter, ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise DataSourceError(f"{source_id}.row_filters must be an object")
    filters: list[RowFilter] = []
    for raw_name, raw_values in value.items():
        name = _require_string(raw_name, f"{source_id}.row_filters key")
        if name.endswith("_not_in"):
            raw_field = name[: -len("_not_in")]
            operator = "not_in"
        elif name.endswith("_in"):
            raw_field = name[: -len("_in")]
            operator = "in"
        elif name.endswith("_gte"):
            raw_field = name[: -len("_gte")]
            operator = "gte"
        elif name.endswith("_lte"):
            raw_field = name[: -len("_lte")]
            operator = "lte"
        else:
            raise DataSourceError(
                f"{source_id}.row_filters key must end in "
                f"_in, _not_in, _gte, or _lte: {name!r}"
            )
        field = _field_path(raw_field, f"{source_id}.row_filters.{name}")
        contract_field = f"{source_id}.row_filters.{name}"
        if operator in {"in", "not_in"}:
            filters.append(
                RowFilter(
                    field=field,
                    operator=operator,
                    values=_string_tuple(raw_values, contract_field),
                )
            )
        else:
            filters.append(
                RowFilter(
                    field=field,
                    operator=operator,
                    threshold=_require_finite_number(raw_values, contract_field),
                )
            )
    return tuple(filters)


@dataclass(frozen=True, slots=True)
class SplitRecipe:
    seed: str
    modulus: int
    validation_remainder: int
    algorithm: str = "sha256_mod"
    code_group_field: str = "repo_name"

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SplitRecipe:
        algorithm = _require_string(value.get("algorithm"), "split.algorithm")
        if algorithm != "sha256_mod":
            raise DataSourceError(f"unsupported split algorithm: {algorithm!r}")
        modulus = _require_positive_int(value.get("modulus"), "split.modulus")
        remainder = _require_positive_int(
            value.get("validation_remainder"),
            "split.validation_remainder",
            allow_zero=True,
        )
        if remainder >= modulus:
            raise DataSourceError("split.validation_remainder must be below modulus")
        return cls(
            algorithm=algorithm,
            seed=_require_string(value.get("seed"), "split.seed"),
            modulus=modulus,
            validation_remainder=remainder,
            code_group_field=_require_string(
                value.get("code_group_field", "repo_name"),
                "split.code_group_field",
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceRecipe:
    source_id: str
    category: str
    repo_id: str
    revision: str
    config: str
    split: str
    file_patterns: tuple[str, ...]
    license: str
    license_scope: str
    card_url: str
    gated: bool
    text_field: str
    stable_id_fields: tuple[str, ...]
    required_fields: tuple[str, ...]
    train_token_quotas: Mapping[str, int]
    validation_token_quota: int
    min_characters: int
    max_document_tokens: int
    license_field: str | None = None
    license_allowlist: tuple[str, ...] = ()
    attribution_fields: tuple[str, ...] = ()
    reject_detected_secrets: bool = False
    storage_format: str = "parquet"
    locked_files: tuple[LockedSourceFile, ...] = ()
    split_group_fields: tuple[str, ...] = ()
    row_filters: tuple[RowFilter, ...] = ()
    license_value_mode: str = "casefold_exact"
    trust_remote_code: bool = False
    origin_group: str | None = None
    mix_basis_points: int | None = None

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
        profiles: set[str],
        *,
        schema_version: int = SOURCE_RECIPE_SCHEMA_VERSION,
        global_license_allowlist: frozenset[str] = frozenset(),
    ) -> SourceRecipe:
        source_id = _require_string(value.get("source_id"), "source_id")
        if not _SAFE_ID.fullmatch(source_id):
            raise DataSourceError(f"unsafe source_id: {source_id!r}")
        revision = _require_string(value.get("revision"), f"{source_id}.revision").lower()
        if not _SHA40.fullmatch(revision):
            raise DataSourceError(f"{source_id}.revision must be a 40-digit commit SHA")
        gated = value.get("gated")
        if not isinstance(gated, bool):
            raise DataSourceError(f"{source_id}.gated must be boolean")
        if gated:
            raise DataSourceError(
                f"gated source is forbidden in the unattended recipe: {source_id}"
            )
        trust_remote_code = value.get("trust_remote_code", False)
        if not isinstance(trust_remote_code, bool):
            raise DataSourceError(f"{source_id}.trust_remote_code must be boolean")
        if trust_remote_code:
            raise DataSourceError(
                f"trust_remote_code is forbidden for immutable data sources: {source_id}"
            )
        storage_format = _require_string(
            value.get("storage_format", "parquet"),
            f"{source_id}.storage_format",
        )
        if storage_format not in {"parquet", "jsonl_gzip"}:
            raise DataSourceError(
                f"{source_id}.storage_format is unsupported: {storage_format!r}"
            )
        if schema_version == SOURCE_RECIPE_SCHEMA_VERSION and storage_format != "parquet":
            raise DataSourceError("schema-v1 sources support native Parquet only")
        raw_quotas = value.get("train_token_quotas")
        if not isinstance(raw_quotas, Mapping):
            raise DataSourceError(f"{source_id}.train_token_quotas must be an object")
        quotas = {
            str(profile): _require_positive_int(tokens, f"{source_id}.{profile} quota")
            for profile, tokens in raw_quotas.items()
        }
        if set(quotas) != profiles:
            raise DataSourceError(
                f"{source_id} quota profiles differ: expected {sorted(profiles)}, "
                f"got {sorted(quotas)}"
            )
        license_field_value = value.get("license_field")
        license_field = (
            _require_string(license_field_value, f"{source_id}.license_field")
            if license_field_value is not None
            else None
        )
        allowlist = _string_tuple(
            value.get("license_allowlist", []),
            f"{source_id}.license_allowlist",
            required=False,
        )
        normalized_allowlist = tuple(item.casefold() for item in allowlist)
        if schema_version == SOURCE_RECIPE_SCHEMA_VERSION_V2:
            for item, normalized in zip(allowlist, normalized_allowlist, strict=True):
                if normalize_license(item) != normalized:
                    raise DataSourceError(
                        f"{source_id}.license_allowlist entry is not canonical: {item!r}"
                    )
                if global_license_allowlist and normalized not in global_license_allowlist:
                    raise DataSourceError(
                        f"{source_id}.license_allowlist entry is outside global policy: "
                        f"{item!r}"
                    )
        attribution = _string_tuple(
            value.get("attribution_fields", []),
            f"{source_id}.attribution_fields",
            required=False,
        )
        if bool(license_field) != bool(allowlist):
            raise DataSourceError(
                f"{source_id} must specify license_field and license_allowlist together"
            )
        if license_field and license_field not in value.get("required_fields", []):
            raise DataSourceError(f"{source_id}.license_field must be required")
        license_value_mode = _require_string(
            value.get(
                "license_value_mode",
                "casefold_exact"
                if schema_version == SOURCE_RECIPE_SCHEMA_VERSION
                else "canonical_after_normalization",
            ),
            f"{source_id}.license_value_mode",
        )
        if license_value_mode not in {
            "casefold_exact",
            "canonical_spdx",
            "canonical_after_normalization",
            "canonical_spdx_after_normalization",
        }:
            raise DataSourceError(
                f"{source_id}.license_value_mode is unsupported: {license_value_mode!r}"
            )
        if not license_field and "license_value_mode" in value:
            raise DataSourceError(
                f"{source_id}.license_value_mode requires license_field"
            )
        required_fields = tuple(
            _field_path(item, f"{source_id}.required_fields")
            for item in _string_tuple(
                value.get("required_fields"), f"{source_id}.required_fields"
            )
        )
        stable_id_fields = tuple(
            _field_path(item, f"{source_id}.stable_id_fields")
            for item in _string_tuple(
                value.get("stable_id_fields"), f"{source_id}.stable_id_fields"
            )
        )
        split_group_fields = tuple(
            _field_path(item, f"{source_id}.split_group_fields")
            for item in _string_tuple(
                value.get("split_group_fields", list(stable_id_fields)),
                f"{source_id}.split_group_fields",
            )
        )
        text_field = _field_path(value.get("text_field"), f"{source_id}.text_field")
        attribution_fields = tuple(
            _field_path(item, f"{source_id}.attribution_fields")
            for item in attribution
        )
        license_field = (
            _field_path(license_field, f"{source_id}.license_field")
            if license_field is not None
            else None
        )
        row_filters = _parse_row_filters(value.get("row_filters"), source_id)
        contract_fields = {
            text_field,
            *stable_id_fields,
            *(filter_.field for filter_ in row_filters),
        }
        if license_field is not None:
            contract_fields.add(license_field)
        missing_contract_fields = contract_fields - set(required_fields)
        if missing_contract_fields:
            raise DataSourceError(
                f"{source_id} contract fields must be required: "
                f"{sorted(missing_contract_fields)}"
            )
        file_patterns = tuple(
            _safe_relative_pattern(
                item,
                f"{source_id}.file_patterns",
                storage_format=storage_format,
            )
            for item in _string_tuple(value.get("file_patterns"), "file_patterns")
        )
        raw_locked_files = value.get("locked_files", [])
        if not isinstance(raw_locked_files, list):
            raise DataSourceError(f"{source_id}.locked_files must be a list")
        locked_files = tuple(
            LockedSourceFile.from_dict(
                item,
                source_id=source_id,
                storage_format=storage_format,
            )
            for item in raw_locked_files
            if isinstance(item, Mapping)
        )
        if len(locked_files) != len(raw_locked_files):
            raise DataSourceError(f"{source_id}.locked_files entries must be objects")
        if schema_version == SOURCE_RECIPE_SCHEMA_VERSION_V2:
            if not locked_files:
                raise DataSourceError(f"{source_id}.locked_files cannot be empty")
            if len({item.path for item in locked_files}) != len(locked_files):
                raise DataSourceError(f"{source_id}.locked_files paths contain duplicates")
            if set(file_patterns) != {item.path for item in locked_files}:
                raise DataSourceError(
                    f"{source_id}.file_patterns must exactly enumerate locked_files"
                )
            if not attribution_fields:
                raise DataSourceError(
                    f"{source_id}.attribution_fields cannot be empty in schema v2"
                )
        origin_group_value = value.get("origin_group")
        origin_group = (
            _require_string(origin_group_value, f"{source_id}.origin_group")
            if origin_group_value is not None
            else None
        )
        if origin_group is not None and origin_group not in {"existing", "new"}:
            raise DataSourceError(
                f"{source_id}.origin_group must be 'existing' or 'new'"
            )
        raw_mix_basis_points = value.get("mix_basis_points")
        mix_basis_points = (
            _require_positive_int(
                raw_mix_basis_points,
                f"{source_id}.mix_basis_points",
            )
            if raw_mix_basis_points is not None
            else None
        )
        if (
            schema_version == SOURCE_RECIPE_SCHEMA_VERSION_V2
            and (origin_group is None or mix_basis_points is None)
        ):
            raise DataSourceError(
                f"{source_id} requires origin_group and mix_basis_points in schema v2"
            )
        raw_reject_detected_secrets = value.get("reject_detected_secrets", False)
        if not isinstance(raw_reject_detected_secrets, bool):
            raise DataSourceError(
                f"{source_id}.reject_detected_secrets must be boolean"
            )
        raw_license = value.get("license")
        if schema_version == SOURCE_RECIPE_SCHEMA_VERSION_V2:
            raw_license = value.get("license_declaration")
        return cls(
            source_id=source_id,
            category=_require_string(value.get("category"), f"{source_id}.category"),
            repo_id=_require_string(value.get("repo_id"), f"{source_id}.repo_id"),
            revision=revision,
            config=_require_string(value.get("config"), f"{source_id}.config"),
            split=_require_string(value.get("split"), f"{source_id}.split"),
            file_patterns=file_patterns,
            license=_require_string(raw_license, f"{source_id}.license declaration"),
            license_scope=_require_string(value.get("license_scope"), f"{source_id}.license_scope"),
            card_url=_require_string(value.get("card_url"), f"{source_id}.card_url"),
            gated=gated,
            text_field=text_field,
            stable_id_fields=stable_id_fields,
            required_fields=required_fields,
            train_token_quotas=quotas,
            validation_token_quota=_require_positive_int(
                value.get("validation_token_quota"),
                f"{source_id}.validation_token_quota",
            ),
            min_characters=_require_positive_int(
                value.get("min_characters"), f"{source_id}.min_characters"
            ),
            max_document_tokens=_require_positive_int(
                value.get("max_document_tokens"),
                f"{source_id}.max_document_tokens",
            ),
            license_field=license_field,
            license_allowlist=normalized_allowlist,
            attribution_fields=attribution_fields,
            reject_detected_secrets=raw_reject_detected_secrets,
            storage_format=storage_format,
            locked_files=locked_files,
            split_group_fields=split_group_fields,
            row_filters=row_filters,
            license_value_mode=license_value_mode,
            trust_remote_code=trust_remote_code,
            origin_group=origin_group,
            mix_basis_points=mix_basis_points,
        )

    @property
    def parquet_columns(self) -> tuple[str, ...]:
        fields = (
            self.text_field,
            *self.stable_id_fields,
            *self.split_group_fields,
            *self.required_fields,
            *self.attribution_fields,
            *(filter_.field for filter_ in self.row_filters),
        )
        return tuple(
            dict.fromkeys(
                field.split(".", 1)[0] if self.storage_format == "parquet" else field
                for field in fields
            )
        )


@dataclass(frozen=True, slots=True)
class BaseDataRecipe:
    schema_version: int
    recipe_id: str
    split: SplitRecipe
    output_shard_tokens: int
    profiles: Mapping[str, int]
    validation_tokens: int
    sources: tuple[SourceRecipe, ...]
    sha256: str
    schema_status: str = "stable"
    declared_runnable: bool = True
    declared_parser_compatible: bool = True
    required_implementation: tuple[str, ...] = ()

    @property
    def verified_implementation(self) -> tuple[str, ...]:
        verified = set(verified_data_pipeline_implementations())
        return tuple(
            requirement
            for requirement in self.required_implementation
            if requirement in verified
        )

    @property
    def missing_implementation(self) -> tuple[str, ...]:
        verified = set(self.verified_implementation)
        return tuple(
            requirement
            for requirement in self.required_implementation
            if requirement not in verified
        )

    @property
    def runnable(self) -> bool:
        return (
            self.schema_status == "stable"
            and self.declared_runnable
            and self.declared_parser_compatible
            and not self.missing_implementation
        )

    def require_runnable(self, operation: str) -> None:
        reasons: list[str] = []
        if self.schema_status != "stable":
            reasons.append(f"schema_status={self.schema_status!r}")
        if not self.declared_runnable:
            reasons.append("activation.runnable=false")
        if not self.declared_parser_compatible:
            reasons.append("activation.current_parser_compatible=false")
        if self.missing_implementation:
            reasons.append(
                "unverified implementation: "
                + ", ".join(self.missing_implementation)
            )
        if reasons:
            raise DataSourceError(
                f"recipe {self.recipe_id!r} is not activated for {operation}: "
                + "; ".join(reasons)
            )


def load_base_data_recipe(path: str | Path) -> BaseDataRecipe:
    source = Path(path)
    raw = source.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise DataSourceError(f"invalid data source recipe JSON: {source}") from error
    if not isinstance(value, Mapping):
        raise DataSourceError("data source recipe must be a JSON object")
    schema_version = value.get("schema_version")
    if schema_version not in SUPPORTED_SOURCE_RECIPE_SCHEMA_VERSIONS:
        raise DataSourceError("unsupported data source recipe schema")
    expected_kinds = {
        SOURCE_RECIPE_SCHEMA_VERSION: {"twen_base_data_source_recipe"},
        SOURCE_RECIPE_SCHEMA_VERSION_V2: {
            "twen_base_data_source_recipe_v2",
            "twen_base_data_source_recipe_v2_draft",
        },
    }
    if value.get("kind") not in expected_kinds[int(schema_version)]:
        raise DataSourceError("unexpected data source recipe kind")
    global_license_allowlist: frozenset[str] = frozenset()
    if schema_version == SOURCE_RECIPE_SCHEMA_VERSION_V2:
        raw_license_policy = value.get("license_policy")
        if not isinstance(raw_license_policy, Mapping):
            raise DataSourceError("schema-v2 license_policy must be an object")
        raw_global_allowlist = _string_tuple(
            raw_license_policy.get("canonical_permissive_allowlist"),
            "license_policy.canonical_permissive_allowlist",
        )
        global_license_allowlist = frozenset(item.casefold() for item in raw_global_allowlist)
        if len(global_license_allowlist) != len(raw_global_allowlist):
            raise DataSourceError("global license allowlist contains duplicates")
        if any(normalize_license(item) != item for item in global_license_allowlist):
            raise DataSourceError("global license allowlist must use canonical values")
    raw_profiles = value.get("profiles")
    if not isinstance(raw_profiles, Mapping) or not raw_profiles:
        raise DataSourceError("profiles must be a non-empty object")
    profiles: dict[str, int] = {}
    for name, profile_value in raw_profiles.items():
        if not isinstance(name, str) or not _SAFE_PROFILE.fullmatch(name):
            raise DataSourceError(f"unsafe profile name: {name!r}")
        if not isinstance(profile_value, Mapping):
            raise DataSourceError(f"profile {name} must be an object")
        profiles[name] = _require_positive_int(
            profile_value.get("train_tokens"), f"profiles.{name}.train_tokens"
        )
    raw_sources = value.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise DataSourceError("sources must be a non-empty list")
    sources = tuple(
        SourceRecipe.from_dict(
            item,
            set(profiles),
            schema_version=int(schema_version),
            global_license_allowlist=global_license_allowlist,
        )
        for item in raw_sources
        if isinstance(item, Mapping)
    )
    if len(sources) != len(raw_sources):
        raise DataSourceError("every source must be an object")
    ids = [item.source_id for item in sources]
    if len(set(ids)) != len(ids):
        raise DataSourceError("source_id values must be unique")
    for profile, target in profiles.items():
        actual = sum(item.train_token_quotas[profile] for item in sources)
        if actual != target:
            raise DataSourceError(f"{profile} source quotas sum to {actual}, expected {target}")
    validation_tokens = _require_positive_int(value.get("validation_tokens"), "validation_tokens")
    if sum(item.validation_token_quota for item in sources) != validation_tokens:
        raise DataSourceError("source validation quotas do not sum to validation_tokens")
    if schema_version == SOURCE_RECIPE_SCHEMA_VERSION_V2:
        mix_total = sum(int(item.mix_basis_points or 0) for item in sources)
        raw_mix_contract = value.get("mix_contract")
        if not isinstance(raw_mix_contract, Mapping):
            raise DataSourceError("schema-v2 mix_contract must be an object")
        expected_mix_total = _require_positive_int(
            raw_mix_contract.get("basis_points_total"),
            "mix_contract.basis_points_total",
        )
        if mix_total != expected_mix_total:
            raise DataSourceError(
                f"source mix basis points sum to {mix_total}, expected {expected_mix_total}"
            )
        for origin in ("existing", "new"):
            expected = _require_positive_int(
                raw_mix_contract.get(f"{origin}_sources_basis_points"),
                f"mix_contract.{origin}_sources_basis_points",
                allow_zero=True,
            )
            actual = sum(
                int(item.mix_basis_points or 0)
                for item in sources
                if item.origin_group == origin
            )
            if actual != expected:
                raise DataSourceError(
                    f"{origin} source mix basis points sum to {actual}, expected {expected}"
                )
        for profile, target in profiles.items():
            for source_recipe in sources:
                numerator = target * int(source_recipe.mix_basis_points or 0)
                if numerator % expected_mix_total != 0:
                    raise DataSourceError(
                        f"{source_recipe.source_id}.{profile} mix quota is fractional"
                    )
                expected = numerator // expected_mix_total
                if source_recipe.train_token_quotas[profile] != expected:
                    raise DataSourceError(
                        f"{source_recipe.source_id}.{profile} quota differs from mix contract"
                    )
    activation = value.get("activation")
    declared_runnable = True
    declared_parser_compatible = True
    required_implementation = (
        SCHEMA_V2_REQUIRED_IMPLEMENTATIONS
        if schema_version == SOURCE_RECIPE_SCHEMA_VERSION_V2
        else ()
    )
    if activation is not None:
        if not isinstance(activation, Mapping):
            raise DataSourceError("activation must be an object")
        if not isinstance(activation.get("runnable"), bool):
            raise DataSourceError("activation.runnable must be boolean")
        declared_runnable = bool(activation["runnable"])
        parser_compatible_value = activation.get(
            "current_parser_compatible",
            declared_runnable,
        )
        if not isinstance(parser_compatible_value, bool):
            raise DataSourceError(
                "activation.current_parser_compatible must be boolean"
            )
        declared_parser_compatible = parser_compatible_value
        raw_requirements = activation.get(
            "required_implementation",
            list(required_implementation),
        )
        required_implementation = _string_tuple(
            raw_requirements,
            "activation.required_implementation",
            required=False,
        )
        if len(set(required_implementation)) != len(required_implementation):
            raise DataSourceError(
                "activation.required_implementation contains duplicates"
            )
        if schema_version == SOURCE_RECIPE_SCHEMA_VERSION_V2:
            undeclared = [
                requirement
                for requirement in SCHEMA_V2_REQUIRED_IMPLEMENTATIONS
                if requirement not in required_implementation
            ]
            if undeclared:
                raise DataSourceError(
                    "schema-v2 activation omits required implementation: "
                    + ", ".join(undeclared)
                )
    schema_status = _require_string(
        value.get("schema_status", "stable"), "schema_status"
    )
    if schema_status not in {"draft", "stable"}:
        raise DataSourceError(f"unsupported schema_status: {schema_status!r}")
    return BaseDataRecipe(
        schema_version=int(schema_version),
        recipe_id=_require_string(value.get("recipe_id"), "recipe_id"),
        split=SplitRecipe.from_dict(
            value.get("split") if isinstance(value.get("split"), Mapping) else {}
        ),
        output_shard_tokens=_require_positive_int(
            value.get("output_shard_tokens"), "output_shard_tokens"
        ),
        profiles=profiles,
        validation_tokens=validation_tokens,
        sources=sources,
        sha256=hashlib.sha256(raw).hexdigest(),
        schema_status=schema_status,
        declared_runnable=declared_runnable,
        declared_parser_compatible=declared_parser_compatible,
        required_implementation=required_implementation,
    )


def inspect_base_data_recipe(path: str | Path) -> Mapping[str, object]:
    """Return an offline, machine-readable activation and materialization plan."""

    recipe = load_base_data_recipe(path)
    source_rows = [
        {
            "source_id": source.source_id,
            "origin_group": source.origin_group,
            "mix_basis_points": source.mix_basis_points,
            "storage_format": source.storage_format,
            "locked_file_count": len(source.locked_files),
            "locked_bytes": sum(item.size for item in source.locked_files),
            "license_declaration": source.license,
            "license_scope": source.license_scope,
            "license_field": source.license_field,
            "license_value_mode": source.license_value_mode,
            "license_allowlist": list(source.license_allowlist),
            "materialization": (
                "immutable_lfs_range_stream"
                if source.storage_format == "parquet"
                else "verified_complete_lfs_download_then_streaming_gzip_jsonl"
            ),
        }
        for source in recipe.sources
    ]
    formats = sorted({source.storage_format for source in recipe.sources})
    return {
        "ok": True,
        "operation": "inspect",
        "offline": True,
        "recipe_id": recipe.recipe_id,
        "recipe_sha256": recipe.sha256,
        "schema_version": recipe.schema_version,
        "schema_status": recipe.schema_status,
        "activation": {
            "declared_runnable": recipe.declared_runnable,
            "declared_parser_compatible": recipe.declared_parser_compatible,
            "required_implementation": list(recipe.required_implementation),
            "verified_implementation": list(recipe.verified_implementation),
            "missing_implementation": list(recipe.missing_implementation),
            "runnable": recipe.runnable,
        },
        "profiles": dict(recipe.profiles),
        "validation_tokens": recipe.validation_tokens,
        "source_count": len(recipe.sources),
        "formats": formats,
        "mix_basis_points_total": sum(
            int(source.mix_basis_points or 0) for source in recipe.sources
        ),
        "locked_bytes": sum(
            item.size
            for source in recipe.sources
            for item in source.locked_files
        ),
        "sources": source_rows,
        "training_started": False,
        "network_accessed": False,
        "files_written": False,
    }


@dataclass(frozen=True, slots=True)
class ResolvedParquetFile:
    path: str
    size: int
    sha256: str
    url: str
    storage_format: str = "parquet"

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
        *,
        storage_format: str = "parquet",
    ) -> ResolvedParquetFile:
        path = _safe_relative_pattern(
            value.get("path"),
            "resolved file path",
            storage_format=storage_format,
        )
        size = _require_positive_int(value.get("size"), f"{path}.size")
        sha = _require_string(value.get("sha256"), f"{path}.sha256").lower()
        if not _SHA256.fullmatch(sha):
            raise DataSourceError(f"{path}.sha256 is not a SHA256")
        url = _require_string(value.get("url"), f"{path}.url")
        if not url.startswith("https://huggingface.co/datasets/"):
            raise DataSourceError(f"unexpected resolved dataset URL: {url}")
        return cls(
            path=path,
            size=size,
            sha256=sha,
            url=url,
            storage_format=storage_format,
        )


# Public generic name for schema v2; retain the historical class name so
# downstream v1 callers and serialized code references remain compatible.
ResolvedSourceFile = ResolvedParquetFile


@dataclass(frozen=True, slots=True)
class ResolvedSource:
    source_id: str
    repo_id: str
    revision: str
    files: tuple[ResolvedParquetFile, ...]


@dataclass(frozen=True, slots=True)
class ResolvedSourceLock:
    recipe_id: str
    recipe_sha256: str
    sources: tuple[ResolvedSource, ...]
    path: Path
    sha256: str


def _dataset_file_url(endpoint: str, repo_id: str, revision: str, filename: str) -> str:
    repo = "/".join(quote(part, safe="") for part in repo_id.split("/"))
    return f"{endpoint.rstrip('/')}/datasets/{repo}/resolve/{revision}/{quote(filename, safe='/')}"


def _lfs_identity(entry: Mapping[str, object], filename: str) -> tuple[int, str]:
    lfs = entry.get("lfs")
    if not isinstance(lfs, Mapping):
        raise DataSourceError(f"immutable source file is not an LFS object: {filename}")
    size = _require_positive_int(lfs.get("size"), f"{filename}.lfs.size")
    raw_oid = lfs.get("sha256", lfs.get("oid"))
    oid = _require_string(raw_oid, f"{filename}.lfs.oid").removeprefix("sha256:").lower()
    if not _SHA256.fullmatch(oid):
        raise DataSourceError(f"missing immutable LFS SHA256 for {filename}")
    return size, oid


MetadataFetcher = Callable[[str, str], Mapping[str, object]]


def metadata_fetcher_from_fixture(path: str | Path) -> MetadataFetcher:
    """Load deterministic Hub metadata used by offline CLI/CI verification."""

    fixture_path = Path(path)
    try:
        value = json.loads(fixture_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataSourceError(
            f"invalid offline metadata fixture: {fixture_path}"
        ) from error
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != 1
        or value.get("kind") != "twen_hf_dataset_metadata_fixture"
    ):
        raise DataSourceError("unsupported offline metadata fixture")
    raw_repositories = value.get("repositories")
    if not isinstance(raw_repositories, list) or not raw_repositories:
        raise DataSourceError("offline metadata fixture has no repositories")
    repositories: dict[tuple[str, str], Mapping[str, object]] = {}
    for index, raw_entry in enumerate(raw_repositories):
        if not isinstance(raw_entry, Mapping):
            raise DataSourceError(
                f"offline metadata fixture repository {index} is not an object"
            )
        repo_id = _require_string(
            raw_entry.get("repo_id"),
            f"repositories[{index}].repo_id",
        )
        revision = _require_string(
            raw_entry.get("revision"),
            f"repositories[{index}].revision",
        ).lower()
        if not _SHA40.fullmatch(revision):
            raise DataSourceError(
                f"repositories[{index}].revision must be a 40-digit commit SHA"
            )
        payload = raw_entry.get("payload")
        if not isinstance(payload, Mapping):
            raise DataSourceError(
                f"repositories[{index}].payload must be an object"
            )
        key = (repo_id, revision)
        if key in repositories:
            raise DataSourceError(
                f"duplicate offline metadata fixture repository: {repo_id}@{revision}"
            )
        repositories[key] = payload

    def fetch(repo_id: str, revision: str) -> Mapping[str, object]:
        key = (repo_id, revision.lower())
        try:
            return repositories[key]
        except KeyError as error:
            raise DataSourceError(
                f"offline metadata fixture lacks {repo_id}@{revision}"
            ) from error

    return fetch


def _resolution_contract_audits(
    recipe: BaseDataRecipe,
    *,
    remote_identity_verification: str,
) -> dict[str, object]:
    mix_sources = [
        {
            "source_id": source.source_id,
            "origin_group": source.origin_group,
            "mix_basis_points": source.mix_basis_points,
            "train_token_quotas": dict(source.train_token_quotas),
            "validation_token_quota": source.validation_token_quota,
        }
        for source in recipe.sources
    ]
    return {
        "activation_audit": {
            "schema_status": recipe.schema_status,
            "declared_runnable": recipe.declared_runnable,
            "declared_parser_compatible": recipe.declared_parser_compatible,
            "required_implementation": list(recipe.required_implementation),
            "verified_implementation": list(recipe.verified_implementation),
            "missing_implementation": list(recipe.missing_implementation),
            "runnable": recipe.runnable,
        },
        "source_mix": {
            "schema_version": SOURCE_MIX_AUDIT_SCHEMA_VERSION,
            "algorithm": SOURCE_MIX_AUDIT_ALGORITHM,
            "unit": "valid_tokens",
            "basis_points_total": sum(
                int(source.mix_basis_points or 0) for source in recipe.sources
            ),
            "sources": mix_sources,
        },
        "format_audit": {
            "complete": True,
            "formats": {
                storage_format: {
                    "source_count": sum(
                        source.storage_format == storage_format
                        for source in recipe.sources
                    ),
                    "locked_file_count": sum(
                        len(source.locked_files)
                        for source in recipe.sources
                        if source.storage_format == storage_format
                    ),
                }
                for storage_format in sorted(
                    {source.storage_format for source in recipe.sources}
                )
            },
        },
        "license_audit": {
            "complete": True,
            "normalization": "canonical_allowlist_before_acceptance",
            "sources": [
                {
                    "source_id": source.source_id,
                    "declaration": source.license,
                    "scope": source.license_scope,
                    "field": source.license_field,
                    "value_mode": source.license_value_mode,
                    "allowlist": list(source.license_allowlist),
                }
                for source in recipe.sources
            ],
        },
        "materialization_audit": {
            "complete": remote_identity_verification.startswith("verified_against_"),
            "remote_identity_verification": remote_identity_verification,
            "sources": [
                {
                    "source_id": source.source_id,
                    "storage_format": source.storage_format,
                    "method": (
                        "immutable_lfs_range_stream"
                        if source.storage_format == "parquet"
                        else "verified_complete_lfs_download_then_streaming_gzip_jsonl"
                    ),
                    "locked_files": [asdict(item) for item in source.locked_files],
                }
                for source in recipe.sources
            ],
        },
    }


def plan_base_data_source_resolution(
    recipe_path: str | Path,
    *,
    manager: DownloadManager | None = None,
    token: str | None = None,
    endpoint: str = "https://huggingface.co",
    metadata_fetcher: MetadataFetcher | None = None,
    verify_remote: bool = True,
) -> Mapping[str, object]:
    """Plan or verify every pinned source without writing the resolved lock.

    Schema-v2 recipes already carry audited ``locked_files``.  Hub metadata is
    checked whenever ``verify_remote`` is true and must agree byte-for-byte
    with those embedded identities.
    """

    recipe = load_base_data_recipe(recipe_path)
    recipe.require_runnable("source resolution")
    active_manager = (
        manager or DownloadManager(network_policy="fallback")
        if verify_remote and metadata_fetcher is None
        else None
    )
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def fetch(repo_id: str, revision: str) -> Mapping[str, object]:
        if metadata_fetcher is not None:
            return metadata_fetcher(repo_id, revision)
        repo = "/".join(quote(part, safe="") for part in repo_id.split("/"))
        url = f"{endpoint.rstrip('/')}/api/datasets/{repo}/revision/{revision}?blobs=true"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "User-Agent": "twen-dataset-resolver/1",
                **headers,
            },
        )
        if active_manager is None:  # pragma: no cover - guarded by caller mode
            raise DataSourceError("online metadata fetch has no download manager")
        with active_manager._open(request) as response:
            payload = json.load(response)
        if not isinstance(payload, Mapping):
            raise DataSourceError(f"Hub metadata is not an object: {repo_id}")
        return payload

    metadata_cache: dict[tuple[str, str], Mapping[str, object]] = {}
    resolved_sources: list[dict[str, object]] = []
    for source in recipe.sources:
        files: list[dict[str, object]] = []
        if verify_remote:
            key = (source.repo_id, source.revision)
            if key not in metadata_cache:
                metadata_cache[key] = fetch(*key)
            payload = metadata_cache[key]
            resolved_revision = str(payload.get("sha", "")).lower()
            if resolved_revision != source.revision:
                raise DataSourceError(
                    f"Hub revision mismatch for {source.repo_id}: "
                    f"expected {source.revision}, got {resolved_revision!r}"
                )
            siblings = payload.get("siblings")
            if not isinstance(siblings, list):
                raise DataSourceError(
                    f"Hub metadata has no sibling list: {source.repo_id}"
                )
            for entry in siblings:
                if not isinstance(entry, Mapping):
                    continue
                filename_value = entry.get("rfilename")
                if not isinstance(filename_value, str):
                    continue
                filename = PurePosixPath(filename_value).as_posix()
                if not any(
                    fnmatch.fnmatchcase(filename, pattern)
                    for pattern in source.file_patterns
                ):
                    continue
                size, oid = _lfs_identity(entry, filename)
                files.append(
                    {
                        "path": filename,
                        "size": size,
                        "sha256": oid,
                        "url": _dataset_file_url(
                            endpoint,
                            source.repo_id,
                            source.revision,
                            filename,
                        ),
                    }
                )
        else:
            if not source.locked_files:
                raise DataSourceError(
                    f"offline resolution plan requires embedded locked_files: "
                    f"{source.source_id}"
                )
            files.extend(
                {
                    "path": item.path,
                    "size": item.size,
                    "sha256": item.sha256,
                    "url": _dataset_file_url(
                        endpoint,
                        source.repo_id,
                        source.revision,
                        item.path,
                    ),
                }
                for item in source.locked_files
            )
        files.sort(key=lambda item: str(item["path"]))
        if not files:
            raise DataSourceError(
                f"no immutable source file matched {source.source_id}: "
                f"{source.file_patterns}"
            )
        if source.locked_files:
            expected_files = {
                item.path: (item.size, item.sha256) for item in source.locked_files
            }
            actual_files = {
                str(item["path"]): (int(item["size"]), str(item["sha256"]))
                for item in files
            }
            if actual_files != expected_files:
                raise DataSourceError(
                    f"Hub LFS identities differ from embedded locked_files for "
                    f"{source.source_id}"
                )
        resolved_sources.append(
            {
                "source_id": source.source_id,
                "repo_id": source.repo_id,
                "revision": source.revision,
                "config": source.config,
                "split": source.split,
                "license": source.license,
                "storage_format": source.storage_format,
                "files": files,
            }
        )
    verification = (
        "verified_against_offline_fixture"
        if metadata_fetcher is not None and verify_remote
        else (
            "verified_against_hub_metadata"
            if verify_remote
            else "embedded_lock_plan_only"
        )
    )
    audits = _resolution_contract_audits(
        recipe,
        remote_identity_verification=verification,
    )
    return {
        "schema_version": RESOLVED_SOURCE_LOCK_SCHEMA_VERSION,
        "kind": "twen_resolved_base_data_sources",
        "recipe_schema_version": recipe.schema_version,
        "recipe_id": recipe.recipe_id,
        "recipe_sha256": recipe.sha256,
        **audits,
        "sources": resolved_sources,
    }


def resolve_base_data_sources(
    recipe_path: str | Path,
    output_path: str | Path,
    *,
    manager: DownloadManager | None = None,
    token: str | None = None,
    endpoint: str = "https://huggingface.co",
    metadata_fetcher: MetadataFetcher | None = None,
) -> Path:
    """Verify and persist the immutable source lock."""

    payload = plan_base_data_source_resolution(
        recipe_path,
        manager=manager,
        token=token,
        endpoint=endpoint,
        metadata_fetcher=metadata_fetcher,
        verify_remote=True,
    )
    output = Path(output_path)
    atomic_write_json(output, payload)
    return output


def load_resolved_source_lock(path: str | Path, recipe: BaseDataRecipe) -> ResolvedSourceLock:
    lock_path = Path(path)
    raw = lock_path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise DataSourceError(f"invalid resolved source lock: {lock_path}") from error
    if not isinstance(value, Mapping):
        raise DataSourceError("resolved source lock must be an object")
    if value.get("schema_version") != RESOLVED_SOURCE_LOCK_SCHEMA_VERSION:
        raise DataSourceError("unsupported resolved source lock schema")
    if value.get("kind") != "twen_resolved_base_data_sources":
        raise DataSourceError("unexpected resolved source lock kind")
    if value.get("recipe_id") != recipe.recipe_id or value.get("recipe_sha256") != recipe.sha256:
        raise DataSourceError("resolved source lock does not bind the exact recipe bytes")
    if recipe.schema_version == SOURCE_RECIPE_SCHEMA_VERSION_V2:
        if value.get("recipe_schema_version") != recipe.schema_version:
            raise DataSourceError("resolved source lock recipe schema audit differs")
        raw_materialization = value.get("materialization_audit")
        if not isinstance(raw_materialization, Mapping):
            raise DataSourceError("resolved source lock lacks materialization audit")
        verification = _require_string(
            raw_materialization.get("remote_identity_verification"),
            "materialization_audit.remote_identity_verification",
        )
        if not verification.startswith("verified_against_"):
            raise DataSourceError(
                "resolved source lock was not verified against immutable metadata"
            )
        expected_audits = _resolution_contract_audits(
            recipe,
            remote_identity_verification=verification,
        )
        for audit_name, expected_audit in expected_audits.items():
            if value.get(audit_name) != expected_audit:
                raise DataSourceError(
                    f"resolved source lock {audit_name} differs from recipe contract"
                )
    raw_sources = value.get("sources")
    if not isinstance(raw_sources, list):
        raise DataSourceError("resolved source lock has no sources")
    recipes = {item.source_id: item for item in recipe.sources}
    sources: list[ResolvedSource] = []
    for raw_source in raw_sources:
        if not isinstance(raw_source, Mapping):
            raise DataSourceError("invalid resolved source entry")
        source_id = _require_string(raw_source.get("source_id"), "resolved.source_id")
        source_recipe = recipes.get(source_id)
        if source_recipe is None:
            raise DataSourceError(f"unknown resolved source: {source_id}")
        repo_id = _require_string(raw_source.get("repo_id"), f"{source_id}.repo_id")
        revision = _require_string(raw_source.get("revision"), f"{source_id}.revision")
        if (repo_id, revision) != (source_recipe.repo_id, source_recipe.revision):
            raise DataSourceError(f"resolved identity differs for {source_id}")
        raw_files = raw_source.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise DataSourceError(f"resolved source has no files: {source_id}")
        files = tuple(
            ResolvedParquetFile.from_dict(
                item,
                storage_format=source_recipe.storage_format,
            )
            for item in raw_files
            if isinstance(item, Mapping)
        )
        if len(files) != len(raw_files):
            raise DataSourceError(f"invalid resolved file entry: {source_id}")
        if len({item.path for item in files}) != len(files):
            raise DataSourceError(f"duplicate resolved file path: {source_id}")
        raw_storage_format = raw_source.get(
            "storage_format", source_recipe.storage_format
        )
        if raw_storage_format != source_recipe.storage_format:
            raise DataSourceError(f"resolved storage format differs for {source_id}")
        if any(
            not any(
                fnmatch.fnmatchcase(item.path, pattern) for pattern in source_recipe.file_patterns
            )
            for item in files
        ):
            raise DataSourceError(f"resolved file escapes recipe patterns: {source_id}")
        for item in files:
            expected_url = _dataset_file_url(
                "https://huggingface.co",
                source_recipe.repo_id,
                source_recipe.revision,
                item.path,
            )
            if item.url != expected_url:
                raise DataSourceError(
                    f"resolved URL does not match repo/revision/path for {source_id}: {item.path}"
                )
        if source_recipe.locked_files:
            expected_files = {
                item.path: (item.size, item.sha256)
                for item in source_recipe.locked_files
            }
            actual_files = {
                item.path: (item.size, item.sha256)
                for item in files
            }
            if actual_files != expected_files:
                raise DataSourceError(
                    f"resolved files differ from embedded locked_files for {source_id}"
                )
        sources.append(
            ResolvedSource(
                source_id=source_id,
                repo_id=repo_id,
                revision=revision,
                files=files,
            )
        )
    if {item.source_id for item in sources} != set(recipes):
        raise DataSourceError("resolved source set differs from recipe")
    ordered = tuple(sorted(sources, key=lambda item: list(recipes).index(item.source_id)))
    return ResolvedSourceLock(
        recipe_id=recipe.recipe_id,
        recipe_sha256=recipe.sha256,
        sources=ordered,
        path=lock_path,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


_RANGE_NETWORK_ERRORS = (ClientError, OSError, TimeoutError)


class _FallbackRangeHandle(io.BufferedIOBase):
    """Retry any failed direct range read through the configured proxy."""

    def __init__(
        self,
        factory: HfRangeFileFactory,
        artifact: ResolvedParquetFile,
        handle: BinaryIO,
    ) -> None:
        self._factory = factory
        self._artifact = artifact
        self._handle = handle
        self._using_proxy = False
        self._closed = False
        self._lock = threading.RLock()

    @property
    def closed(self) -> bool:
        return self._closed

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return int(self._handle.tell())

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        with self._lock:
            return int(self._handle.seek(offset, whence))

    def read(self, size: int = -1) -> bytes:
        with self._lock:
            position = self.tell()
            try:
                return self._handle.read(size)
            except _RANGE_NETWORK_ERRORS:
                if self._using_proxy:
                    raise
                with suppress(Exception):
                    self._handle.close()
                self._factory.proxy_fallback_used = True
                self._handle = self._factory._open_once(self._artifact, proxy=True)
                self._using_proxy = True
                self._handle.seek(position)
                return self._handle.read(size)

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def close(self) -> None:
        if self._closed:
            return
        self._handle.close()
        self._closed = True


class HfRangeFileFactory:
    """Open immutable Hub files with Range requests and direct-to-proxy fallback."""

    def __init__(
        self,
        *,
        network_policy: str,
        proxy_settings: ProxySettings,
        token: str | None = None,
        block_size: int = 8 * 1024 * 1024,
    ) -> None:
        if network_policy not in {"direct", "github-only", "fallback", "proxy"}:
            raise ValueError(f"invalid network policy: {network_policy}")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.network_policy = network_policy
        self.proxy_settings = proxy_settings
        self.token = token
        self.block_size = block_size
        self.proxy_fallback_used = network_policy == "proxy"
        self._proxy_checked = False

    @property
    def effective_network_policy(self) -> str:
        if self.proxy_fallback_used and self.network_policy == "fallback":
            return "proxy-fallback"
        return self.network_policy

    def _filesystem(self, *, proxy: bool):
        from fsspec.implementations.http import HTTPFileSystem

        headers = {"User-Agent": "twen-range-reader/1", "Accept-Encoding": "identity"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        options: dict[str, object] = {
            "block_size": self.block_size,
            "cache_type": "readahead",
            "client_kwargs": {"trust_env": False},
            "headers": headers,
        }
        if proxy:
            if not self._proxy_checked:
                check_proxy_connectivity(self.proxy_settings, protocol="https")
                self._proxy_checked = True
            options["proxy"] = self.proxy_settings.https_proxy
        return HTTPFileSystem(**options)

    def _open_once(self, artifact: ResolvedParquetFile, *, proxy: bool) -> BinaryIO:
        filesystem = self._filesystem(proxy=proxy)
        info = filesystem.info(artifact.url)
        response_size = info.get("size")
        if response_size is None or int(response_size) != artifact.size:
            raise DataSourceError(
                f"remote size differs from resolved LFS identity for {artifact.path}: "
                f"expected {artifact.size}, got {response_size}"
            )
        # The redirected HF CDN may expose a 64-hex cache/CAS ETag that is not
        # the Hub sibling's LFS oid.  fsspec only returns the final response
        # headers here, so treating that ETag as an LFS SHA would reject valid
        # immutable objects.  The oid remains bound by the resolved Hub metadata
        # lock; this response check proves size/range identity only.
        handle = filesystem.open(
            artifact.url,
            mode="rb",
            block_size=self.block_size,
            cache_type="readahead",
            size=artifact.size,
        )
        # Force one direct footer range now so connectivity failures occur
        # before a transactional JSONL chunk starts writing.
        handle.seek(max(artifact.size - 8, 0))
        footer = handle.read(min(8, artifact.size))
        if not footer.endswith(b"PAR1"):
            handle.close()
            raise DataSourceError(f"remote object is not Parquet: {artifact.path}")
        handle.seek(0)
        return handle

    def open(self, artifact: ResolvedParquetFile) -> BinaryIO:
        proxy_first = self.network_policy == "proxy" or self.proxy_fallback_used
        try:
            handle = self._open_once(artifact, proxy=proxy_first)
        except DataSourceError:
            raise
        except _RANGE_NETWORK_ERRORS:
            if self.network_policy != "fallback" or proxy_first:
                raise
            self.proxy_fallback_used = True
            return self._open_once(artifact, proxy=True)
        if self.network_policy == "fallback" and not proxy_first:
            return _FallbackRangeHandle(self, artifact, handle)
        return handle


@dataclass(frozen=True, slots=True)
class RowCursor:
    file_index: int = 0
    row_index: int = 0


RowIterator = Callable[
    [ResolvedParquetFile, int, Sequence[str]], Iterator[tuple[int, Mapping[str, object]]]
]


def iter_remote_parquet_rows(
    artifact: ResolvedParquetFile,
    start_row: int,
    columns: Sequence[str],
    *,
    file_factory: HfRangeFileFactory,
    batch_size: int = 512,
) -> Iterator[tuple[int, Mapping[str, object]]]:
    """Yield rows from ``start_row`` without rereading earlier row groups."""

    if start_row < 0 or batch_size <= 0:
        raise ValueError("invalid Parquet row cursor/batch size")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise DataSourceError("pyarrow is required for Base corpus extraction") from error
    with file_factory.open(artifact) as handle:
        parquet = pq.ParquetFile(handle)
        available = set(parquet.schema_arrow.names)
        missing = set(columns) - available
        if missing:
            raise DataSourceError(f"{artifact.path} lacks required columns: {sorted(missing)}")
        absolute_row = 0
        for group_index in range(parquet.num_row_groups):
            group_rows = parquet.metadata.row_group(group_index).num_rows
            group_end = absolute_row + group_rows
            if start_row >= group_end:
                absolute_row = group_end
                continue
            skipped_in_group = max(start_row - absolute_row, 0)
            emitted_in_group = 0
            for batch in parquet.iter_batches(
                batch_size=batch_size,
                row_groups=[group_index],
                columns=list(columns),
                use_threads=True,
            ):
                values = batch.to_pydict()
                for local_index in range(batch.num_rows):
                    group_local = emitted_in_group + local_index
                    if group_local < skipped_in_group:
                        continue
                    row = {name: column[local_index] for name, column in values.items()}
                    yield absolute_row + group_local, row
                emitted_in_group += batch.num_rows
            absolute_row = group_end


def _iter_json_lines(
    handle: Any,
    *,
    source_name: str,
    start_row: int,
    first_row_index: int = 0,
) -> Iterator[tuple[int, Mapping[str, object]]]:
    if start_row < 0:
        raise ValueError("JSONL row cursor cannot be negative")
    for offset, raw_line in enumerate(handle):
        row_index = first_row_index + offset
        if row_index < start_row:
            continue
        if isinstance(raw_line, bytes):
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise DataSourceError(
                    f"invalid UTF-8 JSONL at {source_name}:{row_index + 1}"
                ) from error
        elif isinstance(raw_line, str):
            line = raw_line
        else:
            raise DataSourceError(
                f"JSONL reader returned non-bytes at {source_name}:{row_index + 1}"
            )
        if not line.strip():
            raise DataSourceError(f"blank JSONL row at {source_name}:{row_index + 1}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise DataSourceError(
                f"invalid JSONL object at {source_name}:{row_index + 1}"
            ) from error
        if not isinstance(value, Mapping):
            raise DataSourceError(
                f"JSONL row is not an object at {source_name}:{row_index + 1}"
            )
        yield row_index, value


def iter_gzip_jsonl_rows(
    path: str | Path,
    start_row: int = 0,
    columns: Sequence[str] = (),
) -> Iterator[tuple[int, Mapping[str, object]]]:
    """Stream one object per gzip JSON Lines row.

    ``columns`` is accepted for ``RowIterator`` compatibility.  JSON objects
    are intentionally yielded whole so nested attribution fields are retained.
    """

    del columns
    source = Path(path)
    try:
        with gzip.open(source, "rb") as handle:
            yield from _iter_json_lines(
                handle,
                source_name=source.as_posix(),
                start_row=start_row,
            )
    except (gzip.BadGzipFile, EOFError) as error:
        raise DataSourceError(f"invalid gzip stream: {source}") from error


def iter_local_jsonl_rows(
    path: str | Path,
    start_row: int = 0,
    columns: Sequence[str] = (),
) -> Iterator[tuple[int, Mapping[str, object]]]:
    """Stream a previously materialized deterministic JSONL artifact."""

    del columns
    if start_row < 0:
        raise ValueError("JSONL row cursor cannot be negative")
    source = Path(path)
    first_row_index = 0
    with source.open("rb") as handle:
        index_path = source.with_name("row-offsets.bin")
        if start_row and index_path.is_file():
            index_size = index_path.stat().st_size
            if index_size % 8:
                raise DataSourceError(f"invalid sparse row index size: {index_path}")
            manifest_path = source.with_name("derived-manifest.json")
            try:
                manifest_value = json.loads(manifest_path.read_bytes())
                row_count = int(manifest_value["row_count"])
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                raise DataSourceError(
                    f"cannot validate sparse row index manifest: {manifest_path}"
                ) from error
            if start_row > row_count:
                raise DataSourceError(
                    f"JSONL row cursor {start_row} exceeds row count {row_count}"
                )
            if start_row == row_count:
                handle.seek(0, os.SEEK_END)
                first_row_index = start_row
                yield from _iter_json_lines(
                    handle,
                    source_name=source.as_posix(),
                    start_row=start_row,
                    first_row_index=first_row_index,
                )
                return
            entry = start_row // DERIVED_JSONL_ROW_INDEX_STRIDE
            if (entry + 1) * 8 > index_size:
                raise DataSourceError(
                    f"sparse row index does not cover row {start_row}: {index_path}"
                )
            with index_path.open("rb") as index_handle:
                index_handle.seek(entry * 8)
                raw_offset = index_handle.read(8)
            byte_offset = struct.unpack("<Q", raw_offset)[0]
            if byte_offset > source.stat().st_size:
                raise DataSourceError(f"sparse row index escapes JSONL: {index_path}")
            handle.seek(byte_offset)
            first_row_index = entry * DERIVED_JSONL_ROW_INDEX_STRIDE
        yield from _iter_json_lines(
            handle,
            source_name=source.as_posix(),
            start_row=start_row,
            first_row_index=first_row_index,
        )


@dataclass(frozen=True, slots=True)
class DerivedJsonlArtifact:
    source_id: str
    source_path: str
    compressed_path: Path
    path: Path
    index_path: Path
    manifest_path: Path
    compressed_size: int
    compressed_sha256: str
    size: int
    sha256: str
    row_count: int
    next_row_index: int
    row_index_stride: int
    shard_directory: Path


def download_locked_source_file(
    artifact: ResolvedParquetFile,
    *,
    source_id: str,
    repo_id: str,
    revision: str,
    destination_root: str | Path,
    manager: DownloadManager | None = None,
    token: str | None = None,
) -> Path:
    """Materialize one locked Parquet or gzip LFS object with resume + SHA256."""

    if not _SAFE_ID.fullmatch(source_id):
        raise DataSourceError(f"unsafe source_id: {source_id!r}")
    pinned_revision = revision.lower()
    if not _SHA40.fullmatch(pinned_revision):
        raise DataSourceError(f"{source_id}.revision must be a 40-digit commit SHA")
    expected_url = _dataset_file_url(
        "https://huggingface.co",
        repo_id,
        pinned_revision,
        artifact.path,
    )
    if artifact.url != expected_url:
        raise DataSourceError(
            f"locked source URL does not match repo/revision/path: {artifact.path}"
        )
    destination = Path(destination_root).resolve() / source_id / artifact.path
    spec = ArtifactSpec(
        provider="huggingface",
        repository=repo_id,
        revision=pinned_revision,
        filename=artifact.path,
        url=artifact.url,
        expected_size=artifact.size,
        sha256=artifact.sha256,
    )
    headers = {"Authorization": f"Bearer {token}"} if token else None
    active_manager = manager or DownloadManager(network_policy="fallback")
    return active_manager.download(spec, destination, headers=headers)


def _derived_jsonl_from_complete(
    directory: Path,
    *,
    source_id: str,
    artifact: ResolvedParquetFile,
    compressed_path: Path,
) -> DerivedJsonlArtifact:
    marker = read_complete_marker(directory)
    metadata = marker.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ShardStateError(f"derived JSONL metadata is invalid: {directory}")
    expected_source = {
        "source_id": source_id,
        "source_path": artifact.path,
        "compressed_size": artifact.size,
        "compressed_sha256": artifact.sha256,
    }
    if any(metadata.get(key) != value for key, value in expected_source.items()):
        raise ShardStateError(f"derived JSONL source identity differs: {directory}")
    row_count = _require_positive_int(
        metadata.get("row_count"),
        f"{directory}.row_count",
        allow_zero=True,
    )
    next_row_index = _require_positive_int(
        metadata.get("next_row_index"),
        f"{directory}.next_row_index",
        allow_zero=True,
    )
    if next_row_index != row_count:
        raise ShardStateError(f"derived JSONL cursor differs from row count: {directory}")
    output = directory / "data.jsonl"
    index = directory / "row-offsets.bin"
    manifest = directory / "derived-manifest.json"
    if not output.is_file() or not index.is_file() or not manifest.is_file():
        raise ShardStateError(f"derived JSONL outputs are missing: {directory}")
    size = _require_positive_int(
        metadata.get("derived_size"),
        f"{directory}.derived_size",
        allow_zero=True,
    )
    digest = _require_string(
        metadata.get("derived_sha256"),
        f"{directory}.derived_sha256",
    ).lower()
    if not _SHA256.fullmatch(digest):
        raise ShardStateError(f"derived JSONL SHA256 is invalid: {directory}")
    index_stride = _require_positive_int(
        metadata.get("row_index_stride"),
        f"{directory}.row_index_stride",
    )
    if index_stride != DERIVED_JSONL_ROW_INDEX_STRIDE:
        raise ShardStateError(f"derived JSONL row index stride differs: {directory}")
    index_entries = _require_positive_int(
        metadata.get("row_index_entries"),
        f"{directory}.row_index_entries",
        allow_zero=True,
    )
    expected_index_entries = (
        (row_count + index_stride - 1) // index_stride if row_count else 0
    )
    if index_entries != expected_index_entries or index.stat().st_size != index_entries * 8:
        raise ShardStateError(f"derived JSONL row index shape differs: {directory}")
    index_digest = _require_string(
        metadata.get("row_index_sha256"),
        f"{directory}.row_index_sha256",
    ).lower()
    if not _SHA256.fullmatch(index_digest):
        raise ShardStateError(f"derived JSONL row index SHA256 is invalid: {directory}")
    raw_outputs = marker.get("outputs")
    if not isinstance(raw_outputs, list):
        raise ShardStateError(f"derived JSONL output inventory is invalid: {directory}")
    outputs = {
        str(item.get("path")): item
        for item in raw_outputs
        if isinstance(item, Mapping)
    }
    data_output = outputs.get("data.jsonl")
    index_output = outputs.get("row-offsets.bin")
    manifest_output = outputs.get("derived-manifest.json")
    if (
        not isinstance(data_output, Mapping)
        or data_output.get("size") != size
        or data_output.get("sha256") != digest
        or not isinstance(index_output, Mapping)
        or index_output.get("size") != index_entries * 8
        or index_output.get("sha256") != index_digest
        or not isinstance(manifest_output, Mapping)
        or set(outputs)
        != {"data.jsonl", "row-offsets.bin", "derived-manifest.json"}
    ):
        raise ShardStateError(f"derived JSONL identity differs: {directory}")
    manifest_value = json.loads(manifest.read_bytes())
    if not isinstance(manifest_value, Mapping):
        raise ShardStateError(f"derived JSONL manifest is invalid: {manifest}")
    if (
        manifest_value.get("schema_version") != DERIVED_JSONL_SCHEMA_VERSION
        or manifest_value.get("kind") != "twen_derived_jsonl"
        or manifest_value.get("row_count") != row_count
        or manifest_value.get("derived_sha256") != digest
        or manifest_value.get("derived_size") != size
        or manifest_value.get("next_row_index") != next_row_index
        or manifest_value.get("row_index_stride") != index_stride
        or manifest_value.get("row_index_entries") != index_entries
        or manifest_value.get("row_index_sha256") != index_digest
    ):
        raise ShardStateError(f"derived JSONL manifest disagrees with marker: {manifest}")
    return DerivedJsonlArtifact(
        source_id=source_id,
        source_path=artifact.path,
        compressed_path=compressed_path,
        path=output,
        index_path=index,
        manifest_path=manifest,
        compressed_size=artifact.size,
        compressed_sha256=artifact.sha256,
        size=size,
        sha256=digest,
        row_count=row_count,
        next_row_index=next_row_index,
        row_index_stride=index_stride,
        shard_directory=directory,
    )


def materialize_jsonl_gzip_artifact(
    artifact: ResolvedParquetFile,
    *,
    source_id: str,
    repo_id: str,
    revision: str,
    cache_root: str | Path,
    manager: DownloadManager | None = None,
    token: str | None = None,
) -> DerivedJsonlArtifact:
    """Download, verify and transactionally decompress one locked gzip JSONL.

    Partial HTTP bytes are resumed by :class:`DownloadManager`.  A failed
    decompression leaves an identity-bound ``.incomplete`` shard; the next run
    deterministically replays that shard from the already verified compressed
    object.  Only a hash-inventoried, atomically renamed directory is reusable.
    """

    if not _SAFE_ID.fullmatch(source_id):
        raise DataSourceError(f"unsafe source_id: {source_id!r}")
    if artifact.storage_format != "jsonl_gzip" or not artifact.path.endswith(
        ".json.gz"
    ):
        raise DataSourceError(
            f"materialize_jsonl_gzip_artifact requires jsonl_gzip: {artifact.path}"
        )
    pinned_revision = revision.lower()
    if not _SHA40.fullmatch(pinned_revision):
        raise DataSourceError(f"{source_id}.revision must be a 40-digit commit SHA")
    root = Path(cache_root).resolve()
    compressed_path = download_locked_source_file(
        artifact,
        source_id=source_id,
        repo_id=repo_id,
        revision=pinned_revision,
        destination_root=root / "downloads",
        manager=manager,
        token=token,
    )

    source_identity = {
        "source_id": source_id,
        "repo_id": repo_id,
        "revision": pinned_revision,
        "path": artifact.path,
        "size": artifact.size,
        "sha256": artifact.sha256,
        "url": artifact.url,
    }
    source_fingerprint = hashlib.sha256(
        json.dumps(source_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    pipeline_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "schema_version": DERIVED_JSONL_SCHEMA_VERSION,
                "operation": "gzip-jsonl-identity-decompression",
                "encoding": "utf-8",
                "blank_line_policy": "reject",
                "object_row_policy": "mapping_only",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    shard_id = hashlib.sha256(artifact.path.encode()).hexdigest()[:24]
    transaction_root = root / "derived" / source_id
    with ShardTransaction(
        transaction_root,
        shard_id,
        fingerprint=pipeline_fingerprint,
        source_fingerprint=source_fingerprint,
    ) as transaction:
        if transaction.complete:
            return _derived_jsonl_from_complete(
                transaction.final_directory,
                source_id=source_id,
                artifact=artifact,
                compressed_path=compressed_path,
            )
        for name in (
            "data.jsonl",
            "row-offsets.bin",
            "derived-manifest.json",
            "COMPLETE",
        ):
            (transaction.work_directory / name).unlink(missing_ok=True)
        output = transaction.work_directory / "data.jsonl"
        index = transaction.work_directory / "row-offsets.bin"
        digest = hashlib.sha256()
        index_digest = hashlib.sha256()
        row_count = 0
        try:
            with gzip.open(compressed_path, "rb") as input_handle, output.open(
                "wb"
            ) as output_handle, index.open("wb") as index_handle:
                for row_index, raw_line in enumerate(input_handle):
                    if not raw_line.strip():
                        raise DataSourceError(
                            f"blank JSONL row at {artifact.path}:{row_index + 1}"
                        )
                    try:
                        value = json.loads(raw_line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise DataSourceError(
                            f"invalid JSONL object at {artifact.path}:{row_index + 1}"
                        ) from error
                    if not isinstance(value, Mapping):
                        raise DataSourceError(
                            f"JSONL row is not an object at "
                            f"{artifact.path}:{row_index + 1}"
                        )
                    if row_count % DERIVED_JSONL_ROW_INDEX_STRIDE == 0:
                        encoded_offset = struct.pack("<Q", output_handle.tell())
                        index_handle.write(encoded_offset)
                        index_digest.update(encoded_offset)
                    output_handle.write(raw_line)
                    digest.update(raw_line)
                    row_count += 1
                output_handle.flush()
                os.fsync(output_handle.fileno())
                index_handle.flush()
                os.fsync(index_handle.fileno())
        except (gzip.BadGzipFile, EOFError) as error:
            raise DataSourceError(f"invalid gzip stream: {artifact.path}") from error
        derived_sha256 = digest.hexdigest()
        derived_size = output.stat().st_size
        row_index_entries = (
            (row_count + DERIVED_JSONL_ROW_INDEX_STRIDE - 1)
            // DERIVED_JSONL_ROW_INDEX_STRIDE
            if row_count
            else 0
        )
        row_index_sha256 = index_digest.hexdigest()
        metadata = {
            "schema_version": DERIVED_JSONL_SCHEMA_VERSION,
            "source_id": source_id,
            "source_path": artifact.path,
            "compressed_size": artifact.size,
            "compressed_sha256": artifact.sha256,
            "derived_size": derived_size,
            "derived_sha256": derived_sha256,
            "row_count": row_count,
            "next_row_index": row_count,
            "row_index_stride": DERIVED_JSONL_ROW_INDEX_STRIDE,
            "row_index_entries": row_index_entries,
            "row_index_sha256": row_index_sha256,
        }
        atomic_write_json(
            transaction.work_directory / "derived-manifest.json",
            {
                "schema_version": DERIVED_JSONL_SCHEMA_VERSION,
                "kind": "twen_derived_jsonl",
                **source_identity,
                "compressed_local_path": compressed_path.relative_to(root).as_posix(),
                "derived_path": "data.jsonl",
                "derived_size": derived_size,
                "derived_sha256": derived_sha256,
                "row_count": row_count,
                "next_row_index": row_count,
                "row_index_path": "row-offsets.bin",
                "row_index_stride": DERIVED_JSONL_ROW_INDEX_STRIDE,
                "row_index_entries": row_index_entries,
                "row_index_sha256": row_index_sha256,
                "complete": True,
            },
        )
        final = transaction.commit(
            metadata,
            known_sha256={
                "data.jsonl": derived_sha256,
                "row-offsets.bin": row_index_sha256,
                "derived-manifest.json": sha256_file(
                    transaction.work_directory / "derived-manifest.json"
                ),
            },
        )
    return _derived_jsonl_from_complete(
        final,
        source_id=source_id,
        artifact=artifact,
        compressed_path=compressed_path,
    )


def _stable_split_details(
    source: SourceRecipe,
    row: Mapping[str, object],
    split: SplitRecipe,
) -> tuple[str, str, int]:
    """Return split, canonical source identity hash, and deterministic bucket."""

    identity: list[object] = []
    for field in source.stable_id_fields:
        value = get_dotted_field(row, field, _MISSING)
        if value is _MISSING or value is None or value == "":
            raise DataSourceError(f"{source.source_id} row lacks stable field {field!r}")
        identity.append(value)
    split_identity: list[object] = []
    for field in source.split_group_fields or source.stable_id_fields:
        value = get_dotted_field(row, field, _MISSING)
        if value is _MISSING or value is None or value == "":
            raise DataSourceError(
                f"{source.source_id} row lacks split-group field {field!r}"
            )
        split_identity.append(value)
    canonical_identity = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    canonical_split_identity = json.dumps(
        split_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    stable_id = hashlib.sha256(canonical_identity.encode()).hexdigest()
    digest = hashlib.sha256(
        f"{split.seed}\0{source.source_id}\0{canonical_split_identity}".encode()
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") % split.modulus
    role = "validation" if bucket == split.validation_remainder else "train"
    return role, stable_id, bucket


def stable_split_for_row(
    source: SourceRecipe,
    row: Mapping[str, object],
    split: SplitRecipe,
) -> tuple[str, str]:
    """Return a deterministic split and the canonical source identity."""

    role, stable_id, _ = _stable_split_details(source, row, split)
    return role, stable_id


def _normalized_content_hash(text: str, *, code: bool) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not code:
        normalized = unicodedata.normalize("NFKC", " ".join(normalized.split()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _contains_secret(text: str) -> bool:
    return any(
        pattern.search(text) is not None
        for pattern in (_PRIVATE_KEY, _AWS_KEY, _GITHUB_TOKEN, _GENERIC_SECRET)
    )


def _contains_basic_pii(text: str) -> bool:
    if any(
        pattern.search(text) is not None
        for pattern in (_EMAIL, _CHINESE_PHONE, _CHINESE_ID, _US_SSN)
    ):
        return True
    for match in _IPV4.finditer(text):
        try:
            octets = tuple(int(value) for value in match.group(0).split("."))
        except ValueError:
            continue
        if all(0 <= value <= 255 for value in octets):
            return True
    return False


@dataclass(slots=True)
class SourceProgress:
    next_chunk: int = 0
    cursor: RowCursor = RowCursor()
    train_tokens: int = 0
    validation_tokens: int = 0
    train_rows: int = 0
    validation_rows: int = 0


def _pipeline_fingerprint(
    recipe: BaseDataRecipe,
    resolved_lock: ResolvedSourceLock,
    source: SourceRecipe,
    *,
    profile: str,
    tokenizer_sha256: str,
) -> str:
    payload = {
        "recipe_sha256": recipe.sha256,
        "resolved_lock_sha256": resolved_lock.sha256,
        "source": asdict(source),
        "profile": profile,
        "tokenizer_sha256": tokenizer_sha256,
        "output_shard_tokens": recipe.output_shard_tokens,
        "extractor_source_sha256": EXTRACTOR_SOURCE_SHA256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _source_fingerprint(source: ResolvedSource) -> str:
    payload = {
        "source_id": source.source_id,
        "repo_id": source.repo_id,
        "revision": source.revision,
        "files": [asdict(item) for item in source.files],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_source_progress(
    root: Path,
    *,
    pipeline_fingerprint: str,
    source_fingerprint: str,
) -> tuple[SourceProgress, list[dict[str, object]]]:
    progress = SourceProgress()
    chunks: list[dict[str, object]] = []
    if not root.exists():
        return progress, chunks
    complete_directories = sorted(
        path for path in root.glob("chunk-[0-9][0-9][0-9][0-9][0-9][0-9]") if path.is_dir()
    )
    for expected_index, directory in enumerate(complete_directories):
        expected_name = f"chunk-{expected_index:06d}"
        if directory.name != expected_name:
            raise ShardStateError(
                f"non-contiguous extracted chunks in {root}: expected {expected_name}"
            )
        if not is_shard_complete(directory, verify_hashes=True):
            raise ShardStateError(f"invalid completed extraction chunk: {directory}")
        marker = read_complete_marker(directory)
        if marker.get("fingerprint") != pipeline_fingerprint:
            raise ShardStateError(f"pipeline fingerprint mismatch: {directory}")
        if marker.get("source_fingerprint") != source_fingerprint:
            raise ShardStateError(f"source fingerprint mismatch: {directory}")
        metadata = marker.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ShardStateError(f"chunk metadata is invalid: {directory}")
        progress.next_chunk = expected_index + 1
        progress.cursor = RowCursor(
            file_index=int(metadata["next_file_index"]),
            row_index=int(metadata["next_row_index"]),
        )
        progress.train_tokens += int(metadata["train_tokens"])
        progress.validation_tokens += int(metadata["validation_tokens"])
        progress.train_rows += int(metadata["train_rows"])
        progress.validation_rows += int(metadata["validation_rows"])
        chunks.append(
            {
                "path": directory.as_posix(),
                "marker": marker,
            }
        )
    return progress, chunks


def _iter_completed_texts(
    extracted_root: Path, sources: Sequence[SourceRecipe]
) -> Iterator[tuple[str, bool]]:
    source_by_id = {source.source_id: source for source in sources}
    unexpected = sorted(
        path.name
        for path in extracted_root.iterdir()
        if path.is_dir() and path.name not in source_by_id
    )
    if unexpected:
        raise ShardStateError(
            "extracted corpus contains source directories outside the current recipe: "
            f"{unexpected[:3]}"
        )
    for source in sources:
        source_directory = extracted_root / source.source_id
        if not source_directory.is_dir():
            continue
        is_code = source.category == "code"
        for chunk in sorted(source_directory.iterdir()):
            # A transaction can die after writing COMPLETE but before renaming
            # ``chunk-XXXXXX.incomplete``.  That directory is replayable work,
            # not committed data; loading it into the dedup set would make the
            # replay skip its own rows permanently.
            if (
                not chunk.is_dir()
                or not _COMMITTED_CHUNK.fullmatch(chunk.name)
                or not is_shard_complete(chunk, verify_hashes=True)
            ):
                continue
            for name in ("train.jsonl", "validation.jsonl"):
                path = chunk / name
                if not path.is_file():
                    continue
                with path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        try:
                            value = json.loads(line)
                        except json.JSONDecodeError as error:
                            raise ShardStateError(
                                f"invalid extracted JSONL at {path}:{line_number}"
                            ) from error
                        text = value.get("text") if isinstance(value, Mapping) else None
                        if not isinstance(text, str):
                            raise ShardStateError(
                                f"missing text in extracted JSONL at {path}:{line_number}"
                            )
                        yield text, is_code


def _load_seen_hashes(
    extracted_root: Path, sources: Sequence[SourceRecipe]
) -> set[str]:
    return {
        _normalized_content_hash(text, code=is_code)
        for text, is_code in _iter_completed_texts(extracted_root, sources)
    }


class _JsonlChunkWriter:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._handles: dict[str, Any] = {}

    def _handle(self, name: str):
        handle = self._handles.get(name)
        if handle is None:
            path = self.directory / name
            handle = path.open("w", encoding="utf-8")
            self._handles[name] = handle
        return handle

    def write_text(self, role: str, text: str) -> None:
        handle = self._handle(f"{role}.jsonl")
        handle.write(json.dumps({"text": text}, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")

    def write_attribution(self, value: Mapping[str, object]) -> None:
        handle = self._handle("attribution.jsonl")
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        handle.write("\n")

    def close(self) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for name, handle in self._handles.items():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            hashes[name] = sha256_file(self.directory / name)
        self._handles.clear()
        return hashes

    def abort(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


def _reset_chunk_outputs(directory: Path) -> None:
    for name in (
        "train.jsonl",
        "validation.jsonl",
        "attribution.jsonl",
        "chunk.json",
        # This marker may be durable in an unrenamed transaction after a crash.
        "COMPLETE",
    ):
        path = directory / name
        if path.exists():
            path.unlink()
    for pattern in (".COMPLETE.*.tmp", ".chunk.json.*.tmp"):
        for path in directory.glob(pattern):
            path.unlink(missing_ok=True)


def _target_reached(
    progress: SourceProgress,
    *,
    train_target: int,
    validation_target: int,
) -> bool:
    return progress.train_tokens >= train_target and progress.validation_tokens >= validation_target


def _candidate_text(
    source: SourceRecipe,
    row: Mapping[str, object],
    *,
    stats: dict[str, int],
) -> str | None:
    for field in source.required_fields:
        required_value = get_dotted_field(row, field, _MISSING)
        if required_value is _MISSING or required_value is None:
            stats["rejected_missing_field"] += 1
            return None
    for row_filter in source.row_filters:
        if not row_filter.matches(row):
            stats["rejected_row_filter"] += 1
            return None
    text = get_dotted_field(row, source.text_field, _MISSING)
    if not isinstance(text, str):
        stats["rejected_missing_field"] += 1
        return None
    text = text.strip()
    if len(text) < source.min_characters:
        stats["rejected_short"] += 1
        return None
    if source.license_field:
        license_value = get_dotted_field(row, source.license_field, _MISSING)
        if source.license_value_mode == "casefold_exact":
            normalized_license = (
                license_value.casefold() if isinstance(license_value, str) else None
            )
        else:
            normalized_license = normalize_license(license_value)
        if normalized_license not in source.license_allowlist:
            stats["rejected_license"] += 1
            return None
    if source.reject_detected_secrets and _contains_secret(text):
        stats["rejected_secret"] += 1
        return None
    if _contains_basic_pii(text):
        stats["rejected_pii"] += 1
        return None
    return text


def _write_attribution(
    writer: _JsonlChunkWriter,
    source: SourceRecipe,
    resolved: ResolvedSource,
    artifact: ResolvedParquetFile,
    row_index: int,
    row: Mapping[str, object],
    *,
    role: str,
    stable_id: str,
    split_bucket: int,
    text_sha256: str,
    token_count: int,
) -> None:
    attribution: dict[str, object] = {
        "source_id": source.source_id,
        "source_config": source.config,
        "source_split": source.split,
        "repo_id": resolved.repo_id,
        "revision": resolved.revision,
        "source_file": artifact.path,
        "source_file_url": artifact.url,
        "source_file_size": artifact.size,
        "source_file_lfs_sha256": artifact.sha256,
        "source_row": row_index,
        "stable_id": stable_id,
        "stable_id_fields": list(source.stable_id_fields),
        "split": role,
        "split_bucket": split_bucket,
        "text_sha256": text_sha256,
        "token_count_with_eos": token_count,
        "source_license": source.license,
        "content_field": source.text_field,
        "row_filter_contract": [item.to_contract() for item in source.row_filters],
        "filter_decisions": {
            "required_fields": "pass",
            "minimum_characters": "pass",
            "license_allowlist": "pass" if source.license_field else "not_applicable",
            "row_filters": "pass" if source.row_filters else "not_applicable",
            "secret_regex": "pass" if source.reject_detected_secrets else "not_applicable",
            "basic_pii_regex": "pass",
            "maximum_document_tokens": "pass",
            "global_exact_dedup": "pass_first_occurrence",
        },
    }
    stable_values: dict[str, object] = {}
    for field in source.stable_id_fields:
        value = get_dotted_field(row, field)
        if field == source.text_field:
            stable_values[field] = {
                "sha256": text_sha256,
                "stored": "content_hash_only",
            }
        elif isinstance(value, str) and len(value) > 4096:
            stable_values[field] = {
                "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                "characters": len(value),
                "stored": "hash_only_over_4096_characters",
            }
        elif isinstance(value, (str, int, float, bool)) or value is None:
            stable_values[field] = value
        else:
            stable_values[field] = str(value)
    attribution["source_stable_values"] = stable_values
    split_group_values: dict[str, object] = {}
    for field in source.split_group_fields:
        value = get_dotted_field(row, field)
        if isinstance(value, str) and len(value) > 4096:
            split_group_values[field] = {
                "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                "characters": len(value),
                "stored": "hash_only_over_4096_characters",
            }
        elif isinstance(value, (str, int, float, bool)) or value is None:
            split_group_values[field] = value
        else:
            split_group_values[field] = str(value)
    attribution["source_split_group_values"] = split_group_values
    for field in source.attribution_fields:
        value = get_dotted_field(row, field)
        if isinstance(value, (str, int, float, bool)) or value is None:
            attribution[field] = value
        else:
            attribution[field] = str(value)
    if source.license_field is not None:
        attribution["normalized_license"] = normalize_license(
            get_dotted_field(row, source.license_field)
        )
    writer.write_attribution(attribution)


def _read_chunk(
    *,
    source: SourceRecipe,
    resolved: ResolvedSource,
    recipe: BaseDataRecipe,
    progress: SourceProgress,
    train_target: int,
    validation_target: int,
    tokenizer: Any,
    row_iterator: RowIterator,
    writer: _JsonlChunkWriter,
    seen_hashes: set[str],
    max_rows_per_chunk: int = 100_000,
) -> tuple[RowCursor, dict[str, int], bool]:
    cursor = progress.cursor
    stats = {
        "rows_scanned": 0,
        "train_rows": 0,
        "validation_rows": 0,
        "train_tokens": 0,
        "validation_tokens": 0,
        "rejected_missing_field": 0,
        "rejected_short": 0,
        "rejected_license": 0,
        "rejected_row_filter": 0,
        "rejected_secret": 0,
        "rejected_pii": 0,
        "rejected_duplicate": 0,
        "rejected_long": 0,
        "rejected_quota_full": 0,
    }
    exhausted = False
    while True:
        if cursor.file_index >= len(resolved.files):
            exhausted = True
            break
        artifact = resolved.files[cursor.file_index]
        iterator = row_iterator(artifact, cursor.row_index, source.parquet_columns)
        saw_row = False
        stopped_early = False
        try:
            for row_index, row in iterator:
                saw_row = True
                cursor = RowCursor(cursor.file_index, row_index + 1)
                stats["rows_scanned"] += 1
                text = _candidate_text(source, row, stats=stats)
                if text is not None:
                    try:
                        role, stable_id, split_bucket = _stable_split_details(
                            source, row, recipe.split
                        )
                    except DataSourceError:
                        stats["rejected_missing_field"] += 1
                        role = ""
                        stable_id = ""
                        split_bucket = -1
                    role_current = (
                        progress.train_tokens + stats["train_tokens"]
                        if role == "train"
                        else progress.validation_tokens + stats["validation_tokens"]
                    )
                    role_target = train_target if role == "train" else validation_target
                    if role not in {"train", "validation"}:
                        pass
                    elif role_current >= role_target:
                        stats["rejected_quota_full"] += 1
                    else:
                        is_code = source.category == "code"
                        content_hash = _normalized_content_hash(text, code=is_code)
                        if content_hash in seen_hashes:
                            stats["rejected_duplicate"] += 1
                        else:
                            encoded = tokenizer.encode(text, add_special_tokens=False)
                            token_count = len(encoded) + 1
                            if token_count > source.max_document_tokens:
                                stats["rejected_long"] += 1
                            else:
                                writer.write_text(role, text)
                                _write_attribution(
                                    writer,
                                    source,
                                    resolved,
                                    artifact,
                                    row_index,
                                    row,
                                    role=role,
                                    stable_id=stable_id,
                                    split_bucket=split_bucket,
                                    text_sha256=content_hash,
                                    token_count=token_count,
                                )
                                seen_hashes.add(content_hash)
                                stats[f"{role}_rows"] += 1
                                stats[f"{role}_tokens"] += token_count
                chunk_tokens = stats["train_tokens"] + stats["validation_tokens"]
                complete_after_row = (
                    progress.train_tokens + stats["train_tokens"] >= train_target
                    and progress.validation_tokens + stats["validation_tokens"] >= validation_target
                )
                if (
                    chunk_tokens >= recipe.output_shard_tokens
                    or stats["rows_scanned"] >= max_rows_per_chunk
                    or complete_after_row
                ):
                    stopped_early = True
                    break
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        if stopped_early:
            break
        # Exhausting an iterator means the immutable Parquet file is complete.
        # An empty iterator is also valid when resuming exactly at its row count.
        if saw_row or not stopped_early:
            cursor = RowCursor(cursor.file_index + 1, 0)
        if stats["rows_scanned"] >= max_rows_per_chunk:
            break
    return cursor, stats, exhausted


class CorpusBuildStopped(RuntimeError):
    """Raised at a durable JSONL chunk boundary when STOP is present."""

    def __init__(self, stop_file: Path, completed_tokens: int) -> None:
        self.stop_file = stop_file
        self.completed_tokens = completed_tokens
        super().__init__(f"Base corpus build stopped by {stop_file}")


def _build_source(
    *,
    source: SourceRecipe,
    resolved: ResolvedSource,
    recipe: BaseDataRecipe,
    resolved_lock: ResolvedSourceLock,
    profile: str,
    tokenizer_sha256: str,
    tokenizer: Any,
    output_root: Path,
    row_iterator: RowIterator,
    seen_hashes: set[str],
    progress_bar: TaskProgress,
    stop_file: Path | None,
) -> tuple[SourceProgress, list[dict[str, object]]]:
    source_root = output_root / "extracted" / source.source_id
    pipeline_fingerprint = _pipeline_fingerprint(
        recipe,
        resolved_lock,
        source,
        profile=profile,
        tokenizer_sha256=tokenizer_sha256,
    )
    source_fingerprint = _source_fingerprint(resolved)
    progress, chunks = _load_source_progress(
        source_root,
        pipeline_fingerprint=pipeline_fingerprint,
        source_fingerprint=source_fingerprint,
    )
    train_target = source.train_token_quotas[profile]
    validation_target = source.validation_token_quota
    while not _target_reached(
        progress,
        train_target=train_target,
        validation_target=validation_target,
    ):
        if stop_file is not None and stop_file.exists():
            raise CorpusBuildStopped(
                stop_file,
                progress.train_tokens + progress.validation_tokens,
            )
        shard_id = f"chunk-{progress.next_chunk:06d}"
        with ShardTransaction(
            source_root,
            shard_id,
            fingerprint=pipeline_fingerprint,
            source_fingerprint=source_fingerprint,
        ) as transaction:
            if transaction.complete:
                # This can only happen if another process committed between the
                # initial scan and lock acquisition.  Rescan instead of writing.
                progress, chunks = _load_source_progress(
                    source_root,
                    pipeline_fingerprint=pipeline_fingerprint,
                    source_fingerprint=source_fingerprint,
                )
                continue
            _reset_chunk_outputs(transaction.work_directory)
            writer = _JsonlChunkWriter(transaction.work_directory)
            try:
                next_cursor, stats, exhausted = _read_chunk(
                    source=source,
                    resolved=resolved,
                    recipe=recipe,
                    progress=progress,
                    train_target=train_target,
                    validation_target=validation_target,
                    tokenizer=tokenizer,
                    row_iterator=row_iterator,
                    writer=writer,
                    seen_hashes=seen_hashes,
                )
                known_hashes = writer.close()
            except BaseException:
                writer.abort()
                raise
            if stats["rows_scanned"] == 0 and exhausted:
                raise DataSourceError(
                    f"{source.source_id} exhausted native Parquet before quotas: "
                    f"train {progress.train_tokens}/{train_target}, "
                    f"validation {progress.validation_tokens}/{validation_target}"
                )
            chunk_summary = {
                "schema_version": EXTRACTED_CORPUS_SCHEMA_VERSION,
                "source_id": source.source_id,
                "profile": profile,
                "start_file_index": progress.cursor.file_index,
                "start_row_index": progress.cursor.row_index,
                "next_file_index": next_cursor.file_index,
                "next_row_index": next_cursor.row_index,
                **stats,
            }
            atomic_write_json(transaction.work_directory / "chunk.json", chunk_summary)
            known_hashes["chunk.json"] = sha256_file(transaction.work_directory / "chunk.json")
            final = transaction.commit(chunk_summary, known_sha256=known_hashes)
        progress.next_chunk += 1
        progress.cursor = next_cursor
        progress.train_tokens += stats["train_tokens"]
        progress.validation_tokens += stats["validation_tokens"]
        progress.train_rows += stats["train_rows"]
        progress.validation_rows += stats["validation_rows"]
        marker = read_complete_marker(final)
        chunks.append({"path": final.as_posix(), "marker": marker})
        committed = stats["train_tokens"] + stats["validation_tokens"]
        progress_bar.update(committed)
        progress_bar.set_postfix(
            {
                "source": source.source_id,
                "train": progress.train_tokens,
                "val": progress.validation_tokens,
                "file": progress.cursor.file_index,
            }
        )
        if exhausted and not _target_reached(
            progress,
            train_target=train_target,
            validation_target=validation_target,
        ):
            raise DataSourceError(f"{source.source_id} exhausted before reaching token quotas")
    return progress, chunks


def _manifest_file_entry(
    output_root: Path,
    chunk_directory: Path,
    item: Mapping[str, object],
) -> dict[str, object]:
    name = _require_string(item.get("path"), "chunk output path")
    path = chunk_directory / name
    try:
        relative = path.relative_to(output_root).as_posix()
    except ValueError as error:
        raise ShardStateError(f"extracted output escapes corpus root: {path}") from error
    return {
        "path": relative,
        "size": int(item["size"]),
        "sha256": str(item["sha256"]),
    }


def _write_corpus_manifest(
    *,
    output_root: Path,
    recipe: BaseDataRecipe,
    resolved_lock: ResolvedSourceLock,
    tokenizer_sha256: str,
    profile: str,
    source_results: Sequence[
        tuple[SourceRecipe, ResolvedSource, SourceProgress, list[dict[str, object]]]
    ],
    network_policy: str,
) -> Path:
    train_files: list[dict[str, object]] = []
    validation_files: list[dict[str, object]] = []
    attribution_files: list[dict[str, object]] = []
    source_map_outputs: dict[str, list[dict[str, object]]] = {
        "train": [],
        "validation": [],
    }
    source_manifests: list[dict[str, object]] = []
    for source, resolved, progress, chunks in source_results:
        chunk_entries: list[dict[str, object]] = []
        for chunk in chunks:
            directory = Path(str(chunk["path"]))
            marker = chunk["marker"]
            if not isinstance(marker, Mapping):
                raise ShardStateError(f"invalid marker cached for {directory}")
            raw_outputs = marker.get("outputs")
            if not isinstance(raw_outputs, list):
                raise ShardStateError(f"invalid output inventory for {directory}")
            outputs: list[dict[str, object]] = []
            for raw_output in raw_outputs:
                if not isinstance(raw_output, Mapping):
                    raise ShardStateError(f"invalid output entry for {directory}")
                entry = _manifest_file_entry(output_root, directory, raw_output)
                outputs.append(entry)
                filename = Path(str(entry["path"])).name
                if filename == "train.jsonl":
                    train_files.append(entry)
                    source_map_outputs["train"].append(
                        {"source_id": source.source_id, **entry}
                    )
                elif filename == "validation.jsonl":
                    validation_files.append(entry)
                    source_map_outputs["validation"].append(
                        {"source_id": source.source_id, **entry}
                    )
                elif filename == "attribution.jsonl":
                    attribution_files.append(entry)
            metadata = marker.get("metadata")
            chunk_entries.append(
                {
                    "shard_id": marker.get("shard_id"),
                    "outputs": outputs,
                    "statistics": dict(metadata) if isinstance(metadata, Mapping) else {},
                }
            )
        source_manifests.append(
            {
                "source_id": source.source_id,
                "category": source.category,
                "repo_id": resolved.repo_id,
                "revision": resolved.revision,
                "config": source.config,
                "split": source.split,
                "storage_format": source.storage_format,
                "license": source.license,
                "license_scope": source.license_scope,
                "license_value_mode": source.license_value_mode,
                "row_filters": [item.to_contract() for item in source.row_filters],
                "locked_files": [asdict(item) for item in source.locked_files],
                "target_train_tokens": source.train_token_quotas[profile],
                "actual_train_tokens": progress.train_tokens,
                "target_validation_tokens": source.validation_token_quota,
                "actual_validation_tokens": progress.validation_tokens,
                "train_rows": progress.train_rows,
                "validation_rows": progress.validation_rows,
                "next_file_index": progress.cursor.file_index,
                "next_row_index": progress.cursor.row_index,
                "resolved_file_count": len(resolved.files),
                "chunks": chunk_entries,
            }
        )
    train_files.sort(key=lambda item: str(item["path"]))
    validation_files.sort(key=lambda item: str(item["path"]))
    attribution_files.sort(key=lambda item: str(item["path"]))
    for outputs in source_map_outputs.values():
        outputs.sort(key=lambda item: str(item["path"]))
    file_list_payloads = {
        "train": "".join(f"{item['path']}\n" for item in train_files),
        "validation": "".join(f"{item['path']}\n" for item in validation_files),
        "attribution": "".join(f"{item['path']}\n" for item in attribution_files),
    }
    file_list_names = {
        "train": "train-files.txt",
        "validation": "validation-files.txt",
        "attribution": "attribution-files.txt",
    }
    file_lists: dict[str, dict[str, object]] = {}
    for role, payload in file_list_payloads.items():
        sidecar = output_root / file_list_names[role]
        atomic_write_text(sidecar, payload)
        file_lists[role] = {
            "path": sidecar.name,
            "size": sidecar.stat().st_size,
            "sha256": sha256_file(sidecar),
        }
    source_map_unsigned = {
        "schema_version": SOURCE_MAP_AUDIT_SCHEMA_VERSION,
        "algorithm": SOURCE_MAP_AUDIT_ALGORITHM,
        "roles": source_map_outputs,
    }
    source_map = {
        **source_map_unsigned,
        "fingerprint": hashlib.sha256(
            json.dumps(
                source_map_unsigned,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    source_mix_unsigned = {
        "schema_version": SOURCE_MIX_AUDIT_SCHEMA_VERSION,
        "algorithm": SOURCE_MIX_AUDIT_ALGORITHM,
        "unit": "valid_tokens",
        "basis_points_total": sum(
            int(source.mix_basis_points or 0) for source in recipe.sources
        ),
        "profile": profile,
        "sources": [
            {
                "source_id": source.source_id,
                "origin_group": source.origin_group,
                "mix_basis_points": source.mix_basis_points,
                "target_train_tokens": source.train_token_quotas[profile],
                "actual_train_tokens": progress.train_tokens,
            }
            for source, _, progress, _ in source_results
        ],
    }
    source_mix = {
        **source_mix_unsigned,
        "fingerprint": hashlib.sha256(
            json.dumps(
                source_mix_unsigned,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    format_audit = {
        "complete": True,
        "sources": [
            {
                "source_id": source.source_id,
                "storage_format": source.storage_format,
                "resolved_file_count": len(resolved.files),
                "resolved_bytes": sum(item.size for item in resolved.files),
            }
            for source, resolved, _, _ in source_results
        ],
    }
    license_audit = {
        "complete": True,
        "normalization": "canonical_allowlist_before_acceptance",
        "attribution_inventory": file_lists["attribution"],
        "sources": [
            {
                "source_id": source.source_id,
                "declaration": source.license,
                "scope": source.license_scope,
                "field": source.license_field,
                "value_mode": source.license_value_mode,
                "allowlist": list(source.license_allowlist),
            }
            for source in recipe.sources
        ],
    }
    materialization_audit = {
        "complete": True,
        "network_policy": network_policy,
        "sources": [
            {
                "source_id": source.source_id,
                "storage_format": source.storage_format,
                "method": (
                    "immutable_lfs_range_stream"
                    if source.storage_format == "parquet"
                    else "verified_complete_lfs_download_then_streaming_gzip_jsonl"
                ),
                "input_files": [asdict(item) for item in resolved.files],
                "output_chunk_count": len(chunks),
            }
            for source, resolved, _, chunks in source_results
        ],
    }
    identity = {
        "recipe_id": recipe.recipe_id,
        "recipe_sha256": recipe.sha256,
        "resolved_source_lock_sha256": resolved_lock.sha256,
        "tokenizer_manifest_sha256": tokenizer_sha256,
        "extractor_source_sha256": EXTRACTOR_SOURCE_SHA256,
        "profile": profile,
        "sources": source_manifests,
        "train_files": train_files,
        "validation_files": validation_files,
        "attribution_files": attribution_files,
        "file_lists": file_lists,
    }
    if recipe.schema_version == SOURCE_RECIPE_SCHEMA_VERSION_V2:
        identity.update(
            {
                "source_map": source_map,
                "source_mix": source_mix,
                "format_audit": format_audit,
                "license_audit": license_audit,
                "materialization_audit": materialization_audit,
            }
        )
    corpus_fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": EXTRACTED_CORPUS_SCHEMA_VERSION,
        "kind": "twen_extracted_base_jsonl_corpus",
        **identity,
        "corpus_fingerprint": corpus_fingerprint,
        "actual_train_tokens": sum(int(item["actual_train_tokens"]) for item in source_manifests),
        "actual_validation_tokens": sum(
            int(item["actual_validation_tokens"]) for item in source_manifests
        ),
        "network_policy": network_policy,
        "audits": {
            "immutable_hub_commit_url_and_lfs_identity_lock": "complete",
            "remote_response_size_match": "complete",
            "lfs_sha256": "locked_from_pinned_hub_sibling_metadata",
            "cdn_etag_as_lfs_sha256": "not_assumed",
            "output_sha256": "complete",
            "stable_train_validation_split": "complete",
            "cross_source_exact_dedup": "complete_deterministic_first_wins",
            "code_license_allowlist": "complete",
            "code_secret_regex_scan": "complete",
            "basic_pii_regex_reject": "complete",
            "accepted_row_provenance_ledger": "complete",
            "per_document_license_normalization_allowlist": "complete",
            "declared_row_filters": "complete",
            "gzip_jsonl_complete_download_and_derived_identity": (
                "complete"
                if any(source.storage_format == "jsonl_gzip" for source in recipe.sources)
                else "not_applicable"
            ),
            "cross_source_near_dedup": "pending",
            "full_contextual_pii_scan": "pending",
            "project_benchmark_13gram_scan": "pending",
        },
        "ready_for_data_prepare": True,
        "ready_for_training": False,
    }
    manifest_path = output_root / "corpus-manifest.json"
    atomic_write_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)
    atomic_write_json(
        output_root / "COMPLETE",
        {
            "schema_version": EXTRACTED_CORPUS_SCHEMA_VERSION,
            "kind": "twen_extracted_base_jsonl_complete",
            "corpus_fingerprint": corpus_fingerprint,
            "manifest": manifest_path.name,
            "manifest_sha256": manifest_sha,
            "file_lists": file_lists,
            "ready_for_training": False,
        },
    )
    return manifest_path


def _build_base_jsonl_corpus_unlocked(
    recipe_path: str | Path,
    resolved_lock_path: str | Path,
    output_root: str | Path,
    *,
    tokenizer_path: str | Path,
    tokenizer_manifest_sha256: str,
    profile: str = "dense",
    network_policy: str = "fallback",
    proxy_url: str | None = None,
    token: str | None = None,
    range_block_size: int = 8 * 1024 * 1024,
    stop_file: str | Path | None = None,
    progress: str = "auto",
    _tokenizer: Any | None = None,
    _row_iterator: RowIterator | None = None,
) -> Path:
    """Build deterministic JSONL shards without performing a training step."""

    recipe = load_base_data_recipe(recipe_path)
    recipe.require_runnable("corpus build")
    if profile not in recipe.profiles:
        raise DataSourceError(
            f"unknown data profile {profile!r}; expected one of {sorted(recipe.profiles)}"
        )
    tokenizer_sha = tokenizer_manifest_sha256.lower()
    if not _SHA256.fullmatch(tokenizer_sha):
        raise DataSourceError("tokenizer_manifest_sha256 must be a 64-digit SHA256")
    resolved_lock = load_resolved_source_lock(resolved_lock_path, recipe)
    if _tokenizer is None:
        from ..io.offline import verify_local_download_directory

        verify_local_download_directory(
            tokenizer_path,
            expected_manifest_sha256=tokenizer_sha,
        )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    else:
        tokenizer = _tokenizer
    if getattr(tokenizer, "eos_token_id", None) is None:
        raise DataSourceError("tokenizer must define eos_token_id")

    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "INVALIDATED.json").exists():
        raise DataSourceError(
            f"extracted corpus output is explicitly invalidated: {root / 'INVALIDATED.json'}"
        )
    extracted_root = root / "extracted"
    extracted_root.mkdir(parents=True, exist_ok=True)
    seen_hashes = _load_seen_hashes(extracted_root, recipe.sources)
    resolved_by_id = {item.source_id: item for item in resolved_lock.sources}

    if _row_iterator is None:
        proxy_settings = ProxySettings.from_environment(proxy_url=proxy_url)
        file_factory = HfRangeFileFactory(
            network_policy=network_policy,
            proxy_settings=proxy_settings,
            token=token,
            block_size=range_block_size,
        )
        download_manager = DownloadManager(
            network_policy=network_policy,
            proxy_settings=proxy_settings,
        )
        resolved_for_url: dict[str, tuple[SourceRecipe, ResolvedSource]] = {}
        derived_for_url: dict[str, DerivedJsonlArtifact] = {}
        for source_recipe in recipe.sources:
            resolved_source = resolved_by_id[source_recipe.source_id]
            for resolved_file in resolved_source.files:
                if resolved_file.url in resolved_for_url:
                    raise DataSourceError(
                        f"duplicate resolved source URL: {resolved_file.url}"
                    )
                resolved_for_url[resolved_file.url] = (
                    source_recipe,
                    resolved_source,
                )

        def row_iterator(
            artifact: ResolvedParquetFile,
            start_row: int,
            columns: Sequence[str],
        ) -> Iterator[tuple[int, Mapping[str, object]]]:
            if artifact.storage_format == "parquet":
                return iter_remote_parquet_rows(
                    artifact,
                    start_row,
                    columns,
                    file_factory=file_factory,
                )
            if artifact.storage_format != "jsonl_gzip":
                raise DataSourceError(
                    f"unsupported resolved storage format: {artifact.storage_format}"
                )
            source_recipe, resolved_source = resolved_for_url[artifact.url]
            derived = derived_for_url.get(artifact.url)
            if derived is None:
                derived = materialize_jsonl_gzip_artifact(
                    artifact,
                    source_id=source_recipe.source_id,
                    repo_id=resolved_source.repo_id,
                    revision=resolved_source.revision,
                    cache_root=root / ".source-cache",
                    manager=download_manager,
                    token=token,
                )
                derived_for_url[artifact.url] = derived
            return iter_local_jsonl_rows(
                derived.path,
                start_row,
                columns,
            )

    else:
        file_factory = None
        download_manager = None
        row_iterator = _row_iterator

    initial_tokens = 0
    for source in recipe.sources:
        resolved = resolved_by_id[source.source_id]
        source_progress, _ = _load_source_progress(
            extracted_root / source.source_id,
            pipeline_fingerprint=_pipeline_fingerprint(
                recipe,
                resolved_lock,
                source,
                profile=profile,
                tokenizer_sha256=tokenizer_sha,
            ),
            source_fingerprint=_source_fingerprint(resolved),
        )
        initial_tokens += source_progress.train_tokens + source_progress.validation_tokens
    target_tokens = recipe.profiles[profile] + recipe.validation_tokens
    stop = Path(stop_file).resolve() if stop_file is not None else None
    results: list[tuple[SourceRecipe, ResolvedSource, SourceProgress, list[dict[str, object]]]] = []
    with TaskProgress(
        total=max(target_tokens, initial_tokens),
        initial=initial_tokens,
        description=f"base-{profile}",
        unit="tok",
        unit_scale=True,
        mode=progress,
    ) as progress_bar:
        for source in recipe.sources:
            resolved = resolved_by_id[source.source_id]
            source_progress, chunks = _build_source(
                source=source,
                resolved=resolved,
                recipe=recipe,
                resolved_lock=resolved_lock,
                profile=profile,
                tokenizer_sha256=tokenizer_sha,
                tokenizer=tokenizer,
                output_root=root,
                row_iterator=row_iterator,
                seen_hashes=seen_hashes,
                progress_bar=progress_bar,
                stop_file=stop,
            )
            results.append((source, resolved, source_progress, chunks))
    if file_factory is None:
        effective_policy = network_policy
    elif (
        file_factory.effective_network_policy == "proxy-fallback"
        or (
            download_manager is not None
            and download_manager.effective_network_policy == "proxy-fallback"
        )
    ):
        effective_policy = "proxy-fallback"
    else:
        effective_policy = file_factory.effective_network_policy
    return _write_corpus_manifest(
        output_root=root,
        recipe=recipe,
        resolved_lock=resolved_lock,
        tokenizer_sha256=tokenizer_sha,
        profile=profile,
        source_results=results,
        network_policy=effective_policy,
    )


def plan_base_jsonl_corpus(
    recipe_path: str | Path,
    resolved_lock_path: str | Path,
    *,
    tokenizer_manifest_sha256: str,
    profile: str = "dense",
) -> Mapping[str, object]:
    """Authenticate a build plan without touching tokenizer, network, or output."""

    recipe = load_base_data_recipe(recipe_path)
    recipe.require_runnable("corpus build dry-run")
    if profile not in recipe.profiles:
        raise DataSourceError(
            f"unknown data profile {profile!r}; expected one of "
            f"{sorted(recipe.profiles)}"
        )
    tokenizer_sha = tokenizer_manifest_sha256.lower()
    if not _SHA256.fullmatch(tokenizer_sha):
        raise DataSourceError("tokenizer_manifest_sha256 must be a 64-digit SHA256")
    resolved_lock = load_resolved_source_lock(resolved_lock_path, recipe)
    resolved_by_id = {source.source_id: source for source in resolved_lock.sources}
    return {
        "ok": True,
        "operation": "build",
        "dry_run": True,
        "offline": True,
        "recipe_id": recipe.recipe_id,
        "recipe_sha256": recipe.sha256,
        "resolved_lock": str(resolved_lock.path.resolve()),
        "resolved_lock_sha256": resolved_lock.sha256,
        "tokenizer_manifest_sha256": tokenizer_sha,
        "profile": profile,
        "target_train_tokens": recipe.profiles[profile],
        "target_validation_tokens": recipe.validation_tokens,
        "source_mix": {
            source.source_id: int(source.mix_basis_points or 0)
            for source in recipe.sources
        },
        "sources": [
            {
                "source_id": source.source_id,
                "storage_format": source.storage_format,
                "mix_basis_points": source.mix_basis_points,
                "target_train_tokens": source.train_token_quotas[profile],
                "target_validation_tokens": source.validation_token_quota,
                "resolved_file_count": len(
                    resolved_by_id[source.source_id].files
                ),
                "resolved_bytes": sum(
                    item.size
                    for item in resolved_by_id[source.source_id].files
                ),
                "license_field": source.license_field,
                "license_value_mode": source.license_value_mode,
                "materialization": (
                    "immutable_lfs_range_stream"
                    if source.storage_format == "parquet"
                    else "verified_complete_lfs_download_then_streaming_gzip_jsonl"
                ),
            }
            for source in recipe.sources
        ],
        "training_started": False,
        "network_accessed": False,
        "files_written": False,
    }


def build_base_jsonl_corpus(
    recipe_path: str | Path,
    resolved_lock_path: str | Path,
    output_root: str | Path,
    *,
    tokenizer_path: str | Path,
    tokenizer_manifest_sha256: str,
    profile: str = "dense",
    network_policy: str = "fallback",
    proxy_url: str | None = None,
    token: str | None = None,
    range_block_size: int = 8 * 1024 * 1024,
    stop_file: str | Path | None = None,
    progress: str = "auto",
    _tokenizer: Any | None = None,
    _row_iterator: RowIterator | None = None,
) -> Path:
    """Serialize builders so global exact de-duplication cannot race."""

    load_base_data_recipe(recipe_path).require_runnable("corpus build")
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / ".build-base.lock", timeout_seconds=300.0):
        return _build_base_jsonl_corpus_unlocked(
            recipe_path,
            resolved_lock_path,
            root,
            tokenizer_path=tokenizer_path,
            tokenizer_manifest_sha256=tokenizer_manifest_sha256,
            profile=profile,
            network_policy=network_policy,
            proxy_url=proxy_url,
            token=token,
            range_block_size=range_block_size,
            stop_file=stop_file,
            progress=progress,
            _tokenizer=_tokenizer,
            _row_iterator=_row_iterator,
        )


def _validate_extracted_contract_audits(
    value: Mapping[str, object],
    inventories: Mapping[str, object],
) -> bool:
    present = [name in value for name in EXTRACTED_CONTRACT_IDENTITY_KEYS]
    if not any(present):
        return False
    if not all(present):
        missing = [
            name
            for name in EXTRACTED_CONTRACT_IDENTITY_KEYS
            if name not in value
        ]
        raise DataSourceError(
            "extracted corpus has a partial contract audit: " + ", ".join(missing)
        )

    raw_sources = value.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise DataSourceError("extracted contract audit has no source inventory")
    source_ids: set[str] = set()
    derived_owners: dict[str, str] = {}
    for source_index, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, Mapping):
            raise DataSourceError("extracted source audit entry is invalid")
        source_id = _require_string(
            raw_source.get("source_id"),
            f"sources[{source_index}].source_id",
        )
        if source_id in source_ids:
            raise DataSourceError(f"duplicate extracted source_id: {source_id}")
        source_ids.add(source_id)
        raw_chunks = raw_source.get("chunks")
        if not isinstance(raw_chunks, list):
            raise DataSourceError(f"source {source_id} chunk audit is invalid")
        for raw_chunk in raw_chunks:
            if not isinstance(raw_chunk, Mapping):
                raise DataSourceError(f"source {source_id} chunk entry is invalid")
            raw_outputs = raw_chunk.get("outputs")
            if not isinstance(raw_outputs, list):
                raise DataSourceError(f"source {source_id} output audit is invalid")
            for raw_output in raw_outputs:
                if not isinstance(raw_output, Mapping):
                    raise DataSourceError(f"source {source_id} output entry is invalid")
                output_path = _require_string(
                    raw_output.get("path"),
                    f"{source_id}.output.path",
                )
                if Path(output_path).name not in {
                    "train.jsonl",
                    "validation.jsonl",
                }:
                    continue
                if output_path in derived_owners:
                    raise DataSourceError(
                        f"extracted output has multiple source owners: {output_path}"
                    )
                derived_owners[output_path] = source_id

    source_map = value.get("source_map")
    if (
        not isinstance(source_map, Mapping)
        or source_map.get("schema_version") != SOURCE_MAP_AUDIT_SCHEMA_VERSION
        or source_map.get("algorithm") != SOURCE_MAP_AUDIT_ALGORITHM
    ):
        raise DataSourceError("extracted source_map audit is unsupported")
    source_map_unsigned = {
        key: source_map.get(key)
        for key in ("schema_version", "algorithm", "roles")
    }
    expected_source_map_fingerprint = hashlib.sha256(
        json.dumps(
            source_map_unsigned,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if source_map.get("fingerprint") != expected_source_map_fingerprint:
        raise DataSourceError("extracted source_map fingerprint mismatch")
    roles = source_map.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != {"train", "validation"}:
        raise DataSourceError("extracted source_map roles are invalid")
    for role in ("train", "validation"):
        raw_outputs = roles.get(role)
        inventory = inventories.get(role)
        if not isinstance(raw_outputs, list) or not isinstance(inventory, list):
            raise DataSourceError(f"extracted source_map {role} inventory is invalid")
        normalized_outputs: list[dict[str, object]] = []
        for index, raw_output in enumerate(raw_outputs):
            if not isinstance(raw_output, Mapping):
                raise DataSourceError(
                    f"extracted source_map {role}[{index}] is invalid"
                )
            source_id = _require_string(
                raw_output.get("source_id"),
                f"source_map.{role}[{index}].source_id",
            )
            output_path = _require_string(
                raw_output.get("path"),
                f"source_map.{role}[{index}].path",
            )
            if source_id not in source_ids or derived_owners.get(output_path) != source_id:
                raise DataSourceError(
                    f"source_map owner differs for extracted output: {output_path}"
                )
            normalized_outputs.append(
                {
                    "path": output_path,
                    "size": raw_output.get("size"),
                    "sha256": raw_output.get("sha256"),
                }
            )
        if normalized_outputs != inventory:
            raise DataSourceError(
                f"source_map {role} inventory differs from extracted files"
            )

    source_mix = value.get("source_mix")
    if (
        not isinstance(source_mix, Mapping)
        or source_mix.get("schema_version") != SOURCE_MIX_AUDIT_SCHEMA_VERSION
        or source_mix.get("algorithm") != SOURCE_MIX_AUDIT_ALGORITHM
        or source_mix.get("unit") != "valid_tokens"
    ):
        raise DataSourceError("extracted source_mix audit is unsupported")
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
    expected_source_mix_fingerprint = hashlib.sha256(
        json.dumps(
            source_mix_unsigned,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if source_mix.get("fingerprint") != expected_source_mix_fingerprint:
        raise DataSourceError("extracted source_mix fingerprint mismatch")
    raw_mix_sources = source_mix.get("sources")
    if not isinstance(raw_mix_sources, list):
        raise DataSourceError("extracted source_mix sources are invalid")
    mix_ids: set[str] = set()
    mix_total = 0
    for raw_source in raw_mix_sources:
        if not isinstance(raw_source, Mapping):
            raise DataSourceError("extracted source_mix entry is invalid")
        source_id = _require_string(
            raw_source.get("source_id"),
            "source_mix.source_id",
        )
        if source_id in mix_ids:
            raise DataSourceError(f"duplicate source_mix source_id: {source_id}")
        mix_ids.add(source_id)
        weight = raw_source.get("mix_basis_points")
        if weight is not None:
            mix_total += _require_positive_int(
                weight,
                f"source_mix.{source_id}.mix_basis_points",
            )
    if mix_ids != source_ids:
        raise DataSourceError("source_mix source set differs from source_map")
    if source_mix.get("basis_points_total") != mix_total:
        raise DataSourceError("source_mix basis-point total mismatch")
    if mix_total not in {0, SOURCE_MIX_BASIS_POINTS}:
        raise DataSourceError("source_mix must total 10,000 basis points")

    for audit_name in ("format_audit", "license_audit", "materialization_audit"):
        audit = value.get(audit_name)
        if not isinstance(audit, Mapping) or audit.get("complete") is not True:
            raise DataSourceError(f"extracted {audit_name} is incomplete")
    return True


def validate_extracted_base_corpus(
    manifest_path: str | Path,
    *,
    verify_hashes: bool = True,
) -> Mapping[str, object]:
    """Validate the extraction manifest and every referenced JSONL/ledger."""

    path = Path(manifest_path).resolve()
    root = path.parent
    invalidated = root / "INVALIDATED.json"
    if invalidated.exists():
        raise DataSourceError(f"extracted corpus is explicitly invalidated: {invalidated}")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, Mapping):
        raise DataSourceError("extracted corpus manifest must be an object")
    if value.get("schema_version") != EXTRACTED_CORPUS_SCHEMA_VERSION:
        raise DataSourceError("unsupported extracted corpus manifest schema")
    if value.get("kind") != "twen_extracted_base_jsonl_corpus":
        raise DataSourceError("unexpected extracted corpus kind")
    inventories = {
        "train": value.get("train_files"),
        "validation": value.get("validation_files"),
        "attribution": value.get("attribution_files"),
    }
    files = 0
    seen_outputs: set[str] = set()
    for inventory in inventories.values():
        if not isinstance(inventory, list):
            raise DataSourceError("extracted corpus file inventory is invalid")
        for item in inventory:
            if not isinstance(item, Mapping):
                raise DataSourceError("extracted corpus file entry is invalid")
            relative = PurePosixPath(_require_string(item.get("path"), "output.path"))
            if relative.is_absolute() or ".." in relative.parts:
                raise DataSourceError(f"unsafe extracted output path: {relative}")
            relative_text = relative.as_posix()
            if relative_text in seen_outputs:
                raise DataSourceError(f"duplicate extracted output path: {relative}")
            seen_outputs.add(relative_text)
            output = root / relative
            if not output.is_file() or output.stat().st_size != int(item["size"]):
                raise DataSourceError(f"missing/size-mismatched extracted output: {output}")
            expected_sha = str(item["sha256"])
            if verify_hashes and sha256_file(output) != expected_sha:
                raise DataSourceError(f"SHA256 mismatch for extracted output: {output}")
            files += 1
    file_lists = value.get("file_lists")
    if not isinstance(file_lists, Mapping) or set(file_lists) != set(inventories):
        raise DataSourceError("extracted corpus file-list inventory is invalid")
    expected_sidecar_names = {
        "train": "train-files.txt",
        "validation": "validation-files.txt",
        "attribution": "attribution-files.txt",
    }
    for role, inventory in inventories.items():
        assert isinstance(inventory, list)
        entry = file_lists.get(role)
        if not isinstance(entry, Mapping):
            raise DataSourceError(f"extracted corpus {role} file-list entry is invalid")
        sidecar_name = _require_string(entry.get("path"), f"file_lists.{role}.path")
        if sidecar_name != expected_sidecar_names[role]:
            raise DataSourceError(f"unexpected {role} file-list path: {sidecar_name}")
        sidecar = root / sidecar_name
        expected_payload = "".join(f"{item['path']}\n" for item in inventory)
        try:
            actual_payload = sidecar.read_text(encoding="utf-8")
        except OSError as error:
            raise DataSourceError(f"missing extracted {role} file list: {sidecar}") from error
        if actual_payload != expected_payload:
            raise DataSourceError(f"extracted {role} file list differs from manifest inventory")
        if sidecar.stat().st_size != int(entry.get("size", -1)):
            raise DataSourceError(f"extracted {role} file-list size mismatch")
        if sha256_file(sidecar) != str(entry.get("sha256", "")):
            raise DataSourceError(f"extracted {role} file-list SHA256 mismatch")

    has_contract_audit = _validate_extracted_contract_audits(value, inventories)
    identity_keys = (
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
    if has_contract_audit:
        identity_keys = (*identity_keys, *EXTRACTED_CONTRACT_IDENTITY_KEYS)
    identity = {name: value.get(name) for name in identity_keys}
    actual_fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if value.get("corpus_fingerprint") != actual_fingerprint:
        raise DataSourceError("extracted corpus identity fingerprint mismatch")
    marker_path = root / "COMPLETE"
    if not marker_path.is_file():
        raise DataSourceError(f"extracted corpus has no COMPLETE marker: {root}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        not isinstance(marker, Mapping)
        or marker.get("schema_version") != EXTRACTED_CORPUS_SCHEMA_VERSION
        or marker.get("kind") != "twen_extracted_base_jsonl_complete"
        or marker.get("manifest") != path.name
        or marker.get("corpus_fingerprint") != actual_fingerprint
        or marker.get("file_lists") != file_lists
        or marker.get("ready_for_training") != value.get("ready_for_training")
    ):
        raise DataSourceError("extracted corpus COMPLETE metadata mismatch")
    if marker.get("manifest_sha256") != hashlib.sha256(raw).hexdigest():
        raise DataSourceError("extracted corpus COMPLETE/manifest SHA mismatch")
    return {
        "ok": True,
        "manifest": str(path),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "corpus_fingerprint": value.get("corpus_fingerprint"),
        "profile": value.get("profile"),
        "train_tokens": value.get("actual_train_tokens"),
        "validation_tokens": value.get("actual_validation_tokens"),
        "files": files,
        "ready_for_data_prepare": value.get("ready_for_data_prepare"),
        "ready_for_training": value.get("ready_for_training"),
        "audits": value.get("audits"),
        "source_map": value.get("source_map"),
        "source_mix": value.get("source_mix"),
        "format_audit": value.get("format_audit"),
        "license_audit": value.get("license_audit"),
        "materialization_audit": value.get("materialization_audit"),
    }


__all__ = [
    "BaseDataRecipe",
    "CorpusBuildStopped",
    "DataSourceError",
    "HfRangeFileFactory",
    "ResolvedParquetFile",
    "ResolvedSourceLock",
    "SourceRecipe",
    "build_base_jsonl_corpus",
    "inspect_base_data_recipe",
    "iter_remote_parquet_rows",
    "load_base_data_recipe",
    "load_resolved_source_lock",
    "metadata_fetcher_from_fixture",
    "plan_base_data_source_resolution",
    "plan_base_jsonl_corpus",
    "resolve_base_data_sources",
    "stable_split_for_row",
    "validate_extracted_base_corpus",
    "verified_data_pipeline_implementations",
]
