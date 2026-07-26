"""Verified, locked and resumable artifact downloads.

The regular HTTP stack gives every provider identical Range-resume, locking and
checksum semantics. Requests to GitHub hosts use the configured proxy. Hugging
Face is attempted directly first and, on a transport failure, retried through
the proxy; other hosts explicitly bypass ambient proxy environment variables.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import (
    HTTPSHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from .locking import FileLock
from .proxy import ProxySettings, check_proxy_connectivity

DOWNLOAD_MANIFEST_SCHEMA = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_FLOATING_REVISIONS = {
    "head",
    "latest",
    "main",
    "master",
    "refs/heads/main",
    "refs/heads/master",
}
_COMMIT_REVISION_PATTERN = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_GITHUB_PROXY_DOMAINS = ("github.com", "githubusercontent.com")
_HUGGINGFACE_DOMAINS = ("huggingface.co", "hf.co")
_NETWORK_POLICIES = {"direct", "github-only", "fallback", "proxy"}


def uses_github_proxy(url: str) -> bool:
    """Return whether an HTTP request target belongs to GitHub.

    Host matching is label-aware so names such as ``notgithub.com`` never use
    the proxy.  The check is repeated after redirects by the proxy handler.
    """

    host = (urlparse(url).hostname or "").rstrip(".").lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in _GITHUB_PROXY_DOMAINS)


def _uses_huggingface_host(url: str) -> bool:
    host = (urlparse(url).hostname or "").rstrip(".").lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in _HUGGINGFACE_DOMAINS)


class _GitHubOnlyProxyHandler(ProxyHandler):
    """Apply configured proxies only while the current request is on GitHub."""

    def __init__(self, proxies: Mapping[str, str], *, proxy_all: bool = False) -> None:
        super().__init__(dict(proxies))
        self.proxy_all = bool(proxy_all)

    def proxy_open(self, req: Request, proxy: str, type_: str):  # type: ignore[override]
        if not self.proxy_all and not uses_github_proxy(req.full_url):
            return None
        return super().proxy_open(req, proxy, type_)


class ArtifactValidationError(ValueError):
    """Raised when an artifact or its immutable identity is invalid."""


class ArtifactIntegrityError(IOError):
    """Raised when downloaded content fails size/hash validation."""


def validate_pinned_revision(revision: str) -> str:
    normalized = revision.strip()
    if not normalized:
        raise ArtifactValidationError("artifact revision is required")
    if normalized.casefold() in _FLOATING_REVISIONS:
        raise ArtifactValidationError(
            f"floating revision {revision!r} is forbidden; use a commit or immutable tag"
        )
    return normalized.lower() if _COMMIT_REVISION_PATTERN.fullmatch(normalized) else normalized


def _validate_resolved_commit(revision: str, provider: str) -> str:
    normalized = revision.strip().lower()
    if not _COMMIT_REVISION_PATTERN.fullmatch(normalized):
        raise ArtifactValidationError(
            f"{provider} metadata did not resolve to a 40/64-digit immutable commit: "
            f"{revision!r}"
        )
    return normalized


def _validate_filename(filename: str) -> str:
    path = PurePosixPath(filename)
    if not filename or path.is_absolute() or ".." in path.parts:
        raise ArtifactValidationError(f"unsafe artifact filename {filename!r}")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """Immutable expected identity of one remote artifact."""

    provider: str
    repository: str
    revision: str
    filename: str
    url: str
    expected_size: int
    sha256: str

    def __post_init__(self) -> None:
        if self.provider not in {"http", "huggingface", "modelscope"}:
            raise ArtifactValidationError(f"unsupported provider {self.provider!r}")
        if not self.repository.strip():
            raise ArtifactValidationError("repository/source identifier is required")
        object.__setattr__(self, "revision", validate_pinned_revision(self.revision))
        object.__setattr__(self, "filename", _validate_filename(self.filename))
        if not self.url.startswith(("http://", "https://")):
            raise ArtifactValidationError("artifact URL must use HTTP(S)")
        if self.expected_size < 0:
            raise ArtifactValidationError("expected_size cannot be negative")
        if not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ArtifactValidationError("sha256 must contain exactly 64 hex digits")
        object.__setattr__(self, "sha256", self.sha256.lower())

    @classmethod
    def http(
        cls,
        *,
        url: str,
        revision: str,
        expected_size: int,
        sha256: str,
        filename: str | None = None,
        source_id: str | None = None,
    ) -> ArtifactSpec:
        inferred = filename or url.rstrip("/").rsplit("/", 1)[-1]
        return cls(
            provider="http",
            repository=source_id or url,
            revision=revision,
            filename=inferred,
            url=url,
            expected_size=expected_size,
            sha256=sha256,
        )

    @classmethod
    def huggingface(
        cls,
        *,
        repo_id: str,
        revision: str,
        filename: str,
        expected_size: int,
        sha256: str,
        endpoint: str = "https://huggingface.co",
    ) -> ArtifactSpec:
        pinned = validate_pinned_revision(revision)
        safe_filename = _validate_filename(filename)
        repo_path = "/".join(quote(part, safe="") for part in repo_id.split("/"))
        url = (
            f"{endpoint.rstrip('/')}/{repo_path}/resolve/"
            f"{quote(pinned, safe='')}/{quote(safe_filename, safe='/')}"
        )
        return cls(
            provider="huggingface",
            repository=repo_id,
            revision=pinned,
            filename=safe_filename,
            url=url,
            expected_size=expected_size,
            sha256=sha256,
        )

    @classmethod
    def modelscope(
        cls,
        *,
        model_id: str,
        revision: str,
        filename: str,
        expected_size: int,
        sha256: str,
        endpoint: str = "https://modelscope.cn",
    ) -> ArtifactSpec:
        pinned = validate_pinned_revision(revision)
        safe_filename = _validate_filename(filename)
        model_path = "/".join(quote(part, safe="") for part in model_id.split("/"))
        url = (
            f"{endpoint.rstrip('/')}/models/{model_path}/resolve/"
            f"{quote(pinned, safe='')}/{quote(safe_filename, safe='/')}"
        )
        return cls(
            provider="modelscope",
            repository=model_id,
            revision=pinned,
            filename=safe_filename,
            url=url,
            expected_size=expected_size,
            sha256=sha256,
        )


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(path: str | os.PathLike[str], spec: ArtifactSpec) -> None:
    artifact = Path(path)
    if not artifact.is_file():
        raise ArtifactIntegrityError(f"artifact is missing: {artifact}")
    size = artifact.stat().st_size
    if size != spec.expected_size:
        raise ArtifactIntegrityError(
            f"size mismatch for {artifact}: expected {spec.expected_size}, got {size}"
        )
    actual_hash = sha256_file(artifact)
    if actual_hash != spec.sha256:
        raise ArtifactIntegrityError(
            f"SHA256 mismatch for {artifact}: expected {spec.sha256}, got {actual_hash}"
        )


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, sort_keys=True, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def artifact_manifest_path(path: str | os.PathLike[str]) -> Path:
    artifact = Path(path)
    return artifact.with_name(f"{artifact.name}.manifest.json")


def write_artifact_manifest(path: Path, spec: ArtifactSpec) -> Path:
    manifest_path = artifact_manifest_path(path)
    payload = {
        "schema_version": DOWNLOAD_MANIFEST_SCHEMA,
        "artifact": asdict(spec),
        "local_path": path.name,
        "verified_size": path.stat().st_size,
        "verified_sha256": sha256_file(path),
        "completed_unix_seconds": time.time(),
    }
    _atomic_json(manifest_path, payload)
    return manifest_path


def load_artifact_manifest(path: str | os.PathLike[str]) -> tuple[ArtifactSpec, Path]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("schema_version") != DOWNLOAD_MANIFEST_SCHEMA:
        raise ArtifactValidationError(f"unsupported manifest schema in {manifest_path}")
    spec = ArtifactSpec(**payload["artifact"])
    artifact_path = manifest_path.parent / payload["local_path"]
    if payload.get("verified_size") != spec.expected_size:
        raise ArtifactValidationError(f"manifest size disagrees with spec: {manifest_path}")
    if payload.get("verified_sha256") != spec.sha256:
        raise ArtifactValidationError(f"manifest SHA256 disagrees with spec: {manifest_path}")
    return spec, artifact_path


class DownloadManager:
    """Download artifacts with an explicit host routing and resume policy."""

    def __init__(
        self,
        *,
        proxy_settings: ProxySettings | None = None,
        check_proxy: bool = True,
        use_proxy: bool = True,
        network_policy: str = "fallback",
        connectivity_timeout_seconds: float = 3.0,
        request_timeout_seconds: float = 60.0,
        lock_timeout_seconds: float = 300.0,
        chunk_size: int = 1024 * 1024,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if network_policy not in _NETWORK_POLICIES:
            raise ValueError(
                f"network_policy must be one of {sorted(_NETWORK_POLICIES)}, got {network_policy!r}"
            )
        self.proxy_settings = proxy_settings or ProxySettings.from_environment()
        self.check_proxy = check_proxy
        self.use_proxy = use_proxy
        self.network_policy = network_policy if use_proxy else "direct"
        self.connectivity_timeout_seconds = connectivity_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.lock_timeout_seconds = lock_timeout_seconds
        self.chunk_size = chunk_size
        self._proxy_was_checked = False
        self._proxy_fallback_used = False

    @property
    def proxy_fallback_used(self) -> bool:
        return self._proxy_fallback_used

    @property
    def effective_network_policy(self) -> str:
        if self.network_policy == "fallback" and self._proxy_fallback_used:
            return "proxy-fallback"
        return self.network_policy

    def _opener(self):
        proxy_all = self.network_policy == "proxy" or self._proxy_fallback_used
        if self.network_policy != "direct":
            proxies = {
                "http": self.proxy_settings.http_proxy,
                "https": self.proxy_settings.https_proxy,
            }
            proxy_handler = _GitHubOnlyProxyHandler(proxies, proxy_all=proxy_all)
        else:
            proxy_handler = ProxyHandler({})
        # Supplying an explicit handler prevents urllib from consulting ambient
        # HTTP(S)_PROXY variables for direct hosts.
        return build_opener(proxy_handler, HTTPSHandler())

    def _check_proxy_once(self, url: str) -> None:
        proxy_all = self.network_policy == "proxy" or self._proxy_fallback_used
        if (
            self.network_policy != "direct"
            and self.check_proxy
            and (proxy_all or uses_github_proxy(url))
            and not self._proxy_was_checked
        ):
            check_proxy_connectivity(
                self.proxy_settings,
                timeout_seconds=self.connectivity_timeout_seconds,
                protocol="https",
            )
            self._proxy_was_checked = True

    def _open(self, request: Request):
        """Open once according to policy, optionally falling HF back to proxy."""

        url = request.full_url
        self._check_proxy_once(url)
        try:
            return self._opener().open(request, timeout=self.request_timeout_seconds)
        except HTTPError:
            raise
        except OSError:
            if (
                self.network_policy != "fallback"
                or self._proxy_fallback_used
                or not _uses_huggingface_host(url)
            ):
                raise
            self._proxy_fallback_used = True
            self._check_proxy_once(url)
            return self._opener().open(request, timeout=self.request_timeout_seconds)

    def download(
        self,
        spec: ArtifactSpec,
        destination: str | os.PathLike[str],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Path:
        """Materialize and verify ``spec`` at ``destination``.

        Partial bytes live in ``<destination>.incomplete`` and are intentionally
        retained on all failures.  An existing invalid final artifact is never
        silently overwritten.
        """

        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        incomplete = output.with_name(f"{output.name}.incomplete")
        lock_path = output.with_name(f"{output.name}.lock")
        with FileLock(lock_path, timeout_seconds=self.lock_timeout_seconds):
            if output.exists():
                verify_artifact(output, spec)
                manifest = artifact_manifest_path(output)
                if not manifest.exists():
                    write_artifact_manifest(output, spec)
                return output

            partial_size = incomplete.stat().st_size if incomplete.exists() else 0
            if partial_size > spec.expected_size:
                raise ArtifactIntegrityError(
                    f"partial file is larger than expected: {incomplete}"
                )
            if partial_size < spec.expected_size:
                request_headers = {
                    "Accept-Encoding": "identity",
                    "User-Agent": "twen-downloader/1",
                }
                if headers:
                    request_headers.update(headers)
                if partial_size:
                    request_headers["Range"] = f"bytes={partial_size}-"
                request = Request(spec.url, headers=request_headers, method="GET")
                try:
                    response = self._open(request)
                except HTTPError as error:
                    if error.code == 416 and partial_size == spec.expected_size:
                        response = None
                    else:
                        raise

                if response is not None:
                    status = getattr(response, "status", response.getcode())
                    append = partial_size > 0 and status == 206
                    # Some endpoints ignore Range and reply 200.  Restart in
                    # place instead of appending duplicate bytes.
                    mode = "ab" if append else "wb"
                    if partial_size and status not in {200, 206}:
                        response.close()
                        raise ArtifactIntegrityError(
                            f"unexpected HTTP {status} while resuming {spec.url}"
                        )
                    with response, incomplete.open(mode) as file:
                        while chunk := response.read(self.chunk_size):
                            file.write(chunk)
                        file.flush()
                        os.fsync(file.fileno())

            verify_artifact(incomplete, spec)
            os.replace(incomplete, output)
            _fsync_directory(output.parent)
            write_artifact_manifest(output, spec)
            return output

    def download_huggingface_file(
        self,
        *,
        repo_id: str,
        revision: str,
        filename: str,
        destination: str | os.PathLike[str],
        expected_size: int,
        sha256: str,
        token: str | None = None,
        endpoint: str = "https://huggingface.co",
    ) -> Path:
        spec = ArtifactSpec.huggingface(
            repo_id=repo_id,
            revision=revision,
            filename=filename,
            expected_size=expected_size,
            sha256=sha256,
            endpoint=endpoint,
        )
        headers = {"Authorization": f"Bearer {token}"} if token else None
        return self.download(spec, destination, headers=headers)

    def download_modelscope_file(
        self,
        *,
        model_id: str,
        revision: str,
        filename: str,
        destination: str | os.PathLike[str],
        expected_size: int,
        sha256: str,
        token: str | None = None,
        endpoint: str = "https://modelscope.cn",
    ) -> Path:
        spec = ArtifactSpec.modelscope(
            model_id=model_id,
            revision=revision,
            filename=filename,
            expected_size=expected_size,
            sha256=sha256,
            endpoint=endpoint,
        )
        headers = {"Authorization": f"Bearer {token}"} if token else None
        return self.download(spec, destination, headers=headers)


@dataclass(frozen=True, slots=True)
class ResolvedModelManifest:
    """Provider metadata locked to an immutable revision and exact files."""

    provider: str
    model_id: str
    requested_revision: str
    resolved_revision: str
    artifacts: tuple[ArtifactSpec, ...]

    def __post_init__(self) -> None:
        _validate_resolved_commit(self.resolved_revision, self.provider)
        if not self.artifacts:
            raise ArtifactValidationError("resolved model manifest contains no files")
        if any(spec.revision != self.resolved_revision for spec in self.artifacts):
            raise ArtifactValidationError("artifact revision differs from resolved revision")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": DOWNLOAD_MANIFEST_SCHEMA,
            "provider": self.provider,
            "model_id": self.model_id,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "artifacts": [asdict(spec) for spec in self.artifacts],
        }


MetadataFetcher = Callable[[str, Mapping[str, str]], Mapping[str, object]]
ContentHasher = Callable[[str, Mapping[str, str]], tuple[int, str]]


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _first_value(mapping: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return None


def _sha256_from_metadata(entry: Mapping[str, object]) -> str | None:
    lfs = _mapping(entry.get("lfs"))
    candidate = _first_value(
        entry,
        "sha256",
        "Sha256",
        "SHA256",
        "contentSha256",
    ) or _first_value(lfs, "sha256", "oid")
    if candidate is None:
        return None
    value = str(candidate)
    if value.startswith("sha256:"):
        value = value.removeprefix("sha256:")
    return value.lower() if _SHA256_PATTERN.fullmatch(value) else None


def _size_from_metadata(entry: Mapping[str, object]) -> int | None:
    lfs = _mapping(entry.get("lfs"))
    candidate = _first_value(entry, "size", "Size", "fileSize")
    if candidate is None:
        candidate = _first_value(lfs, "size")
    try:
        value = int(candidate) if candidate is not None else None
    except (TypeError, ValueError):
        return None
    return value if value is not None and value >= 0 else None


def _http_json_fetcher(manager: DownloadManager) -> MetadataFetcher:
    def fetch(url: str, headers: Mapping[str, str]) -> Mapping[str, object]:
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "User-Agent": "twen-manifest-resolver/1",
                **headers,
            },
        )
        with manager._open(request) as response:
            payload = json.load(response)
        if not isinstance(payload, Mapping):
            raise ArtifactValidationError(f"provider metadata is not an object: {url}")
        return payload

    return fetch


def _http_content_hasher(manager: DownloadManager) -> ContentHasher:
    def hash_content(url: str, headers: Mapping[str, str]) -> tuple[int, str]:
        request = Request(
            url,
            headers={
                "Accept-Encoding": "identity",
                "User-Agent": "twen-manifest-resolver/1",
                **headers,
            },
        )
        digest = hashlib.sha256()
        size = 0
        with manager._open(request) as response:
            while chunk := response.read(manager.chunk_size):
                digest.update(chunk)
                size += len(chunk)
        return size, digest.hexdigest()

    return hash_content


def _modelscope_payload(payload: Mapping[str, object]) -> tuple[str, list[Mapping[str, object]]]:
    data = _mapping(payload.get("Data") or payload.get("data"))
    revision = _first_value(
        data,
        "Revision",
        "revision",
        "CommitId",
        "commitId",
        "Sha",
        "sha",
    ) or _first_value(
        payload,
        "Revision",
        "revision",
        "CommitId",
        "commitId",
        "Sha",
        "sha",
    )
    raw_files = _first_value(data, "Files", "files") or _first_value(
        payload, "Files", "files", "siblings"
    )
    if not isinstance(raw_files, list):
        raise ArtifactValidationError("ModelScope metadata has no file list")
    files = [entry for entry in raw_files if isinstance(entry, Mapping)]
    if revision is None:
        # Some ModelScope responses put the immutable commit on every file.
        revisions = {
            str(value)
            for entry in files
            if (
                value := _first_value(
                    entry, "Revision", "revision", "CommitId", "commitId"
                )
            )
        }
        if len(revisions) == 1:
            revision = revisions.pop()
    if revision is None:
        raise ArtifactValidationError("ModelScope did not return an immutable revision")
    return _validate_resolved_commit(str(revision), "modelscope"), files


def resolve_model_manifest(
    provider: str,
    model_id: str,
    revision: str,
    *,
    token: str | None = None,
    endpoint: str | None = None,
    manager: DownloadManager | None = None,
    metadata: Mapping[str, object] | None = None,
    metadata_fetcher: MetadataFetcher | None = None,
    content_hasher: ContentHasher | None = None,
) -> ResolvedModelManifest:
    """Resolve a provider revision and exact file size/SHA256 lock.

    Resolution itself is network-facing and is meant to be invoked explicitly
    by the user's download CLI, never by training.  Provider metadata supplies
    SHA256 for large LFS objects.  For small Git-managed files where no SHA256
    exists, the resolver streams the content once to hash it without installing
    it; this keeps the eventual artifact lock complete and verifiable.

    ``metadata``/``metadata_fetcher``/``content_hasher`` are injectable so lock
    generation can be tested without network access.
    """

    if provider not in {"huggingface", "modelscope"}:
        raise ArtifactValidationError("provider must be huggingface or modelscope")
    if not model_id or not revision:
        raise ArtifactValidationError("model_id and requested revision are required")
    active_manager = manager or DownloadManager()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    fetch_json = metadata_fetcher or _http_json_fetcher(active_manager)
    hash_content = content_hasher or _http_content_hasher(active_manager)

    if provider == "huggingface":
        active_endpoint = (endpoint or "https://huggingface.co").rstrip("/")
        repo_path = "/".join(quote(part, safe="") for part in model_id.split("/"))
        metadata_url = (
            f"{active_endpoint}/api/models/{repo_path}/revision/"
            f"{quote(revision, safe='')}?blobs=true"
        )
        payload = metadata if metadata is not None else fetch_json(metadata_url, headers)
        resolved_value = _first_value(payload, "sha", "commitSha", "revision")
        if resolved_value is None:
            raise ArtifactValidationError("Hugging Face did not return a commit SHA")
        resolved = _validate_resolved_commit(str(resolved_value), "huggingface")
        raw_files = payload.get("siblings")
        if not isinstance(raw_files, list):
            raise ArtifactValidationError("Hugging Face metadata has no siblings list")
        file_entries = [entry for entry in raw_files if isinstance(entry, Mapping)]
        spec_factory = ArtifactSpec.huggingface
        id_keyword = "repo_id"
    else:
        active_endpoint = (endpoint or "https://modelscope.cn").rstrip("/")
        model_path = "/".join(quote(part, safe="") for part in model_id.split("/"))
        metadata_url = (
            f"{active_endpoint}/api/v1/models/{model_path}/repo/files?"
            + urlencode({"Revision": revision, "Recursive": "true"})
        )
        payload = metadata if metadata is not None else fetch_json(metadata_url, headers)
        resolved, file_entries = _modelscope_payload(payload)
        spec_factory = ArtifactSpec.modelscope
        id_keyword = "model_id"

    artifacts: list[ArtifactSpec] = []
    for entry in file_entries:
        entry_type = str(_first_value(entry, "type", "Type") or "file").lower()
        if entry_type in {"directory", "dir", "tree"}:
            continue
        filename_value = _first_value(entry, "rfilename", "path", "Path", "Name", "name")
        if filename_value is None:
            raise ArtifactValidationError("provider file entry has no filename")
        filename = _validate_filename(str(filename_value))
        size = _size_from_metadata(entry)
        sha256 = _sha256_from_metadata(entry)
        provisional = spec_factory(
            **{id_keyword: model_id},
            revision=resolved,
            filename=filename,
            expected_size=size or 0,
            sha256=sha256 or "0" * 64,
            endpoint=active_endpoint,
        )
        if sha256 is None or size is None:
            actual_size, actual_hash = hash_content(provisional.url, headers)
            if size is not None and actual_size != size:
                raise ArtifactIntegrityError(
                    f"metadata/content size mismatch for {filename}: {size} != {actual_size}"
                )
            size, sha256 = actual_size, actual_hash
        artifacts.append(
            spec_factory(
                **{id_keyword: model_id},
                revision=resolved,
                filename=filename,
                expected_size=size,
                sha256=sha256,
                endpoint=active_endpoint,
            )
        )
    artifacts.sort(key=lambda spec: spec.filename)
    return ResolvedModelManifest(
        provider=provider,
        model_id=model_id,
        requested_revision=revision,
        resolved_revision=resolved,
        artifacts=tuple(artifacts),
    )


def write_resolved_model_manifest(
    path: str | os.PathLike[str], manifest: ResolvedModelManifest
) -> Path:
    output = Path(path)
    _atomic_json(output, manifest.to_dict())
    return output


def write_download_set_manifest(
    path: str | os.PathLike[str], specs: Iterable[ArtifactSpec]
) -> Path:
    """Write the immutable expected manifest for a multi-file model/dataset."""

    manifest_path = Path(path)
    artifacts = sorted((asdict(spec) for spec in specs), key=lambda item: item["filename"])
    if not artifacts:
        raise ArtifactValidationError("download set manifest cannot be empty")
    _atomic_json(
        manifest_path,
        {"schema_version": DOWNLOAD_MANIFEST_SCHEMA, "artifacts": artifacts},
    )
    return manifest_path
