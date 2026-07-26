"""Pinned, resumable Hugging Face Parquet-to-JSONL corpus extraction.

This module is deliberately independent from the training engine.  It resolves
native Parquet objects at immutable Hub commits, reads them with HTTP Range
requests, and transactionally emits the JSONL files consumed by ``data
prepare``.  No function in this module constructs an optimizer or performs a
training step.
"""

from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import os
import re
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

from ..io.download import DownloadManager
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
RESOLVED_SOURCE_LOCK_SCHEMA_VERSION = 1
EXTRACTED_CORPUS_SCHEMA_VERSION = 1
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


def _safe_relative_pattern(value: object, field: str) -> str:
    pattern = _require_string(value, field)
    path = PurePosixPath(pattern)
    if path.is_absolute() or ".." in path.parts or not pattern.endswith(".parquet"):
        raise DataSourceError(f"unsafe native Parquet pattern in {field}: {pattern!r}")
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

    @classmethod
    def from_dict(cls, value: Mapping[str, object], profiles: set[str]) -> SourceRecipe:
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
        return cls(
            source_id=source_id,
            category=_require_string(value.get("category"), f"{source_id}.category"),
            repo_id=_require_string(value.get("repo_id"), f"{source_id}.repo_id"),
            revision=revision,
            config=_require_string(value.get("config"), f"{source_id}.config"),
            split=_require_string(value.get("split"), f"{source_id}.split"),
            file_patterns=tuple(
                _safe_relative_pattern(item, f"{source_id}.file_patterns")
                for item in _string_tuple(value.get("file_patterns"), "file_patterns")
            ),
            license=_require_string(value.get("license"), f"{source_id}.license"),
            license_scope=_require_string(value.get("license_scope"), f"{source_id}.license_scope"),
            card_url=_require_string(value.get("card_url"), f"{source_id}.card_url"),
            gated=gated,
            text_field=_require_string(value.get("text_field"), f"{source_id}.text_field"),
            stable_id_fields=_string_tuple(
                value.get("stable_id_fields"), f"{source_id}.stable_id_fields"
            ),
            required_fields=_string_tuple(
                value.get("required_fields"), f"{source_id}.required_fields"
            ),
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
            license_allowlist=tuple(item.casefold() for item in allowlist),
            attribution_fields=attribution,
            reject_detected_secrets=bool(value.get("reject_detected_secrets", False)),
        )

    @property
    def parquet_columns(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    self.text_field,
                    *self.stable_id_fields,
                    *self.required_fields,
                    *self.attribution_fields,
                )
            )
        )


@dataclass(frozen=True, slots=True)
class BaseDataRecipe:
    recipe_id: str
    split: SplitRecipe
    output_shard_tokens: int
    profiles: Mapping[str, int]
    validation_tokens: int
    sources: tuple[SourceRecipe, ...]
    sha256: str


def load_base_data_recipe(path: str | Path) -> BaseDataRecipe:
    source = Path(path)
    raw = source.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise DataSourceError(f"invalid data source recipe JSON: {source}") from error
    if not isinstance(value, Mapping):
        raise DataSourceError("data source recipe must be a JSON object")
    if value.get("schema_version") != SOURCE_RECIPE_SCHEMA_VERSION:
        raise DataSourceError("unsupported data source recipe schema")
    if value.get("kind") != "twen_base_data_source_recipe":
        raise DataSourceError("unexpected data source recipe kind")
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
        SourceRecipe.from_dict(item, set(profiles))
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
    return BaseDataRecipe(
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
    )


@dataclass(frozen=True, slots=True)
class ResolvedParquetFile:
    path: str
    size: int
    sha256: str
    url: str

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ResolvedParquetFile:
        path = _safe_relative_pattern(value.get("path"), "resolved file path")
        size = _require_positive_int(value.get("size"), f"{path}.size")
        sha = _require_string(value.get("sha256"), f"{path}.sha256").lower()
        if not _SHA256.fullmatch(sha):
            raise DataSourceError(f"{path}.sha256 is not a SHA256")
        url = _require_string(value.get("url"), f"{path}.url")
        if not url.startswith("https://huggingface.co/datasets/"):
            raise DataSourceError(f"unexpected resolved dataset URL: {url}")
        return cls(path=path, size=size, sha256=sha, url=url)


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
        raise DataSourceError(f"native Parquet is not an LFS object: {filename}")
    size = _require_positive_int(lfs.get("size"), f"{filename}.lfs.size")
    raw_oid = lfs.get("sha256", lfs.get("oid"))
    oid = _require_string(raw_oid, f"{filename}.lfs.oid").removeprefix("sha256:").lower()
    if not _SHA256.fullmatch(oid):
        raise DataSourceError(f"missing immutable LFS SHA256 for {filename}")
    return size, oid


MetadataFetcher = Callable[[str, str], Mapping[str, object]]


def resolve_base_data_sources(
    recipe_path: str | Path,
    output_path: str | Path,
    *,
    manager: DownloadManager | None = None,
    token: str | None = None,
    endpoint: str = "https://huggingface.co",
    metadata_fetcher: MetadataFetcher | None = None,
) -> Path:
    """Resolve every native Parquet path to its pinned LFS identity."""

    recipe = load_base_data_recipe(recipe_path)
    active_manager = manager or DownloadManager(network_policy="fallback")
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
        with active_manager._open(request) as response:
            payload = json.load(response)
        if not isinstance(payload, Mapping):
            raise DataSourceError(f"Hub metadata is not an object: {repo_id}")
        return payload

    metadata_cache: dict[tuple[str, str], Mapping[str, object]] = {}
    resolved_sources: list[dict[str, object]] = []
    for source in recipe.sources:
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
            raise DataSourceError(f"Hub metadata has no sibling list: {source.repo_id}")
        files: list[dict[str, object]] = []
        for entry in siblings:
            if not isinstance(entry, Mapping):
                continue
            filename_value = entry.get("rfilename")
            if not isinstance(filename_value, str):
                continue
            filename = PurePosixPath(filename_value).as_posix()
            if not any(fnmatch.fnmatchcase(filename, pattern) for pattern in source.file_patterns):
                continue
            size, oid = _lfs_identity(entry, filename)
            files.append(
                {
                    "path": filename,
                    "size": size,
                    "sha256": oid,
                    "url": _dataset_file_url(endpoint, source.repo_id, source.revision, filename),
                }
            )
        files.sort(key=lambda item: str(item["path"]))
        if not files:
            raise DataSourceError(
                f"no native Parquet matched {source.source_id}: {source.file_patterns}"
            )
        resolved_sources.append(
            {
                "source_id": source.source_id,
                "repo_id": source.repo_id,
                "revision": source.revision,
                "config": source.config,
                "split": source.split,
                "license": source.license,
                "files": files,
            }
        )
    output = Path(output_path)
    atomic_write_json(
        output,
        {
            "schema_version": RESOLVED_SOURCE_LOCK_SCHEMA_VERSION,
            "kind": "twen_resolved_base_data_sources",
            "recipe_id": recipe.recipe_id,
            "recipe_sha256": recipe.sha256,
            "sources": resolved_sources,
        },
    )
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
            ResolvedParquetFile.from_dict(item) for item in raw_files if isinstance(item, Mapping)
        )
        if len(files) != len(raw_files):
            raise DataSourceError(f"invalid resolved file entry: {source_id}")
        if len({item.path for item in files}) != len(files):
            raise DataSourceError(f"duplicate resolved file path: {source_id}")
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


def _stable_split_details(
    source: SourceRecipe,
    row: Mapping[str, object],
    split: SplitRecipe,
) -> tuple[str, str, int]:
    """Return split, canonical source identity hash, and deterministic bucket."""

    identity: list[object] = []
    for field in source.stable_id_fields:
        value = row.get(field)
        if value is None or value == "":
            raise DataSourceError(f"{source.source_id} row lacks stable field {field!r}")
        identity.append(value)
    canonical_identity = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    stable_id = hashlib.sha256(canonical_identity.encode()).hexdigest()
    digest = hashlib.sha256(
        f"{split.seed}\0{source.source_id}\0{canonical_identity}".encode()
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
        if row.get(field) is None:
            stats["rejected_missing_field"] += 1
            return None
    text = row.get(source.text_field)
    if not isinstance(text, str):
        stats["rejected_missing_field"] += 1
        return None
    text = text.strip()
    if len(text) < source.min_characters:
        stats["rejected_short"] += 1
        return None
    if source.license_field:
        license_value = row.get(source.license_field)
        if (
            not isinstance(license_value, str)
            or license_value.casefold() not in source.license_allowlist
        ):
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
        "filter_decisions": {
            "required_fields": "pass",
            "minimum_characters": "pass",
            "license_allowlist": "pass" if source.license_field else "not_applicable",
            "secret_regex": "pass" if source.reject_detected_secrets else "not_applicable",
            "basic_pii_regex": "pass",
            "maximum_document_tokens": "pass",
            "global_exact_dedup": "pass_first_occurrence",
        },
    }
    stable_values: dict[str, object] = {}
    for field in source.stable_id_fields:
        value = row.get(field)
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
    for field in source.attribution_fields:
        value = row.get(field)
        if isinstance(value, (str, int, float, bool)) or value is None:
            attribution[field] = value
        else:
            attribution[field] = str(value)
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
                elif filename == "validation.jsonl":
                    validation_files.append(entry)
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
                "license": source.license,
                "license_scope": source.license_scope,
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
        file_factory = HfRangeFileFactory(
            network_policy=network_policy,
            proxy_settings=ProxySettings.from_environment(proxy_url=proxy_url),
            token=token,
            block_size=range_block_size,
        )

        def row_iterator(
            artifact: ResolvedParquetFile,
            start_row: int,
            columns: Sequence[str],
        ) -> Iterator[tuple[int, Mapping[str, object]]]:
            return iter_remote_parquet_rows(
                artifact,
                start_row,
                columns,
                file_factory=file_factory,
            )

    else:
        file_factory = None
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
    effective_policy = (
        file_factory.effective_network_policy if file_factory is not None else network_policy
    )
    return _write_corpus_manifest(
        output_root=root,
        recipe=recipe,
        resolved_lock=resolved_lock,
        tokenizer_sha256=tokenizer_sha,
        profile=profile,
        source_results=results,
        network_policy=effective_policy,
    )


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
    "iter_remote_parquet_rows",
    "load_base_data_recipe",
    "load_resolved_source_lock",
    "resolve_base_data_sources",
    "stable_split_for_row",
    "validate_extracted_base_corpus",
]
