"""Transactional and idempotent sharded preprocessing."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from twen.io.locking import FileLock

SHARD_SCHEMA_VERSION = 1
_SAFE_SHARD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ShardStateError(RuntimeError):
    """Raised for corrupt output or a fingerprint-incompatible restart."""


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, sort_keys=True, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
        # COMPLETE must never become durable ahead of its payload.
        os.fsync(file.fileno())
    return digest.hexdigest()


def _output_inventory(
    root: Path, known_sha256: Mapping[str, str] | None = None
) -> list[dict[str, object]]:
    known = dict(known_sha256 or {})
    seen: set[str] = set()
    outputs: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"COMPLETE", "SHARD_STATE.json"}:
            continue
        relative = path.relative_to(root).as_posix()
        seen.add(relative)
        digest = known.get(relative)
        if digest is None:
            digest = _hash_file(path)
        else:
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ShardStateError(f"invalid known SHA256 for {relative}")
            with path.open("rb") as file:
                os.fsync(file.fileno())
        outputs.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": digest,
            }
        )
    unknown = set(known) - seen
    if unknown:
        raise ShardStateError(f"known hashes refer to missing outputs: {sorted(unknown)}")
    return outputs


def read_complete_marker(shard_directory: str | os.PathLike[str]) -> dict[str, object]:
    shard = Path(shard_directory)
    marker = shard / "COMPLETE"
    if not marker.is_file():
        raise ShardStateError(f"shard has no COMPLETE marker: {shard}")
    with marker.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("schema_version") != SHARD_SCHEMA_VERSION:
        raise ShardStateError(f"unsupported COMPLETE schema: {marker}")
    return payload


def is_shard_complete(
    shard_directory: str | os.PathLike[str], *, verify_hashes: bool = True
) -> bool:
    shard = Path(shard_directory)
    if not shard.is_dir() or not (shard / "COMPLETE").is_file():
        return False
    try:
        marker = read_complete_marker(shard)
        if verify_hashes:
            expected = marker.get("outputs")
            if not isinstance(expected, list) or expected != _output_inventory(shard):
                return False
    except (OSError, ValueError, KeyError, TypeError, ShardStateError):
        return False
    return True


class ShardTransaction:
    """One locked shard transaction whose partial work survives interruption."""

    def __init__(
        self,
        output_root: str | os.PathLike[str],
        shard_id: str,
        *,
        fingerprint: str,
        source_fingerprint: str | None = None,
        lock_timeout_seconds: float = 300.0,
    ) -> None:
        if not _SAFE_SHARD_ID.fullmatch(shard_id) or shard_id in {".", ".."}:
            raise ValueError(f"unsafe shard_id {shard_id!r}")
        if not fingerprint.strip():
            raise ValueError("pipeline fingerprint is required")
        self.output_root = Path(output_root)
        self.shard_id = shard_id
        self.fingerprint = fingerprint
        self.source_fingerprint = source_fingerprint
        self.final_directory = self.output_root / shard_id
        self.work_directory = self.output_root / f"{shard_id}.incomplete"
        self._lock = FileLock(
            self.output_root / f".{shard_id}.lock",
            timeout_seconds=lock_timeout_seconds,
        )
        self.complete = False
        self._entered = False

    def __enter__(self) -> ShardTransaction:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._lock.acquire()
        self._entered = True
        try:
            if self.final_directory.exists():
                if not is_shard_complete(self.final_directory):
                    raise ShardStateError(
                        f"final shard exists but is not complete: {self.final_directory}"
                    )
                marker = read_complete_marker(self.final_directory)
                self._check_identity(marker)
                self.complete = True
                self.work_directory = self.final_directory
                return self

            self.work_directory.mkdir(parents=True, exist_ok=True)
            state_path = self.work_directory / "SHARD_STATE.json"
            identity = {
                "schema_version": SHARD_SCHEMA_VERSION,
                "shard_id": self.shard_id,
                "fingerprint": self.fingerprint,
                "source_fingerprint": self.source_fingerprint,
            }
            if state_path.exists():
                with state_path.open("r", encoding="utf-8") as file:
                    existing = json.load(file)
                self._check_identity(existing)
            else:
                _write_json_atomic(state_path, identity)
            return self
        except BaseException:
            self._lock.release()
            self._entered = False
            raise

    def _check_identity(self, payload: Mapping[str, object]) -> None:
        expected = (self.shard_id, self.fingerprint, self.source_fingerprint)
        actual = (
            payload.get("shard_id"),
            payload.get("fingerprint"),
            payload.get("source_fingerprint"),
        )
        if actual != expected:
            raise ShardStateError(
                f"shard identity mismatch for {self.shard_id}: expected {expected}, got {actual}"
            )

    def commit(
        self,
        metadata: Mapping[str, object] | None = None,
        *,
        known_sha256: Mapping[str, str] | None = None,
    ) -> Path:
        if not self._entered:
            raise RuntimeError("ShardTransaction must be entered before commit")
        if self.complete:
            return self.final_directory
        outputs = _output_inventory(self.work_directory, known_sha256)
        marker = {
            "schema_version": SHARD_SCHEMA_VERSION,
            "shard_id": self.shard_id,
            "fingerprint": self.fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "outputs": outputs,
            "metadata": dict(metadata or {}),
            "completed_unix_seconds": time.time(),
        }
        _write_json_atomic(self.work_directory / "COMPLETE", marker)
        directory_fd = os.open(
            self.work_directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(self.work_directory, self.final_directory)
        root_fd = os.open(
            self.output_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
        self.work_directory = self.final_directory
        self.complete = True
        return self.final_directory

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._lock.release()
        self._entered = False


@dataclass(frozen=True, slots=True)
class ShardJob:
    shard_id: str
    source: str | os.PathLike[str]
    source_fingerprint: str


@dataclass(frozen=True, slots=True)
class ShardRunResult:
    shard_id: str
    path: Path
    skipped: bool


ShardProcessor = Callable[[ShardJob, Path], Mapping[str, object] | None]


def process_shards(
    jobs: Iterable[ShardJob],
    output_root: str | os.PathLike[str],
    *,
    pipeline_fingerprint: str,
    processor: ShardProcessor,
) -> list[ShardRunResult]:
    """Process only unfinished shards, preserving partial directories on error."""

    results: list[ShardRunResult] = []
    for job in jobs:
        with ShardTransaction(
            output_root,
            job.shard_id,
            fingerprint=pipeline_fingerprint,
            source_fingerprint=job.source_fingerprint,
        ) as transaction:
            skipped = transaction.complete
            if not skipped:
                metadata = processor(job, transaction.work_directory)
                transaction.commit(metadata)
            results.append(
                ShardRunResult(
                    shard_id=job.shard_id,
                    path=transaction.final_directory,
                    skipped=skipped,
                )
            )
    return results
