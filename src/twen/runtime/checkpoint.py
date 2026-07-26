"""Atomic, resumable checkpoints for Twen training.

The checkpoint directory has three independent pieces:

* ``state/`` is written by a pluggable backend.  Distributed runs use
  ``torch.distributed.checkpoint`` so model and optimizer state can be
  resharded when the world size changes.
* ``runtime/rank-*.pkl`` stores the small per-rank data cursor and RNG state.
* ``metadata.json``, ``manifest.json`` and ``COMPLETE`` form a hashed commit
  record.  A directory without a valid COMPLETE record is never resumed.

The entire directory is built under a sibling ``.incomplete`` name and renamed
only after every rank has finished writing and rank zero has fsynced it.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import fcntl
import hashlib
import inspect
import json
import os
import pickle
import re
import shutil
import socket
import stat
from collections.abc import Mapping, MutableMapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from .state import (
    CommittedBoundary,
    DataCursor,
    RNGState,
    TrainerState,
    rollback_runtime_state,
)

CHECKPOINT_SCHEMA_VERSION = 2
RUNTIME_PAYLOAD_VERSION = 1
MANIFEST_VERSION = 1
CheckpointKind = Literal["periodic", "interrupt", "milestone"]
_VALID_KINDS = frozenset(("periodic", "interrupt", "milestone"))
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TAG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class CheckpointError(RuntimeError):
    """Base class for checkpoint failures."""


class CheckpointCorruptError(CheckpointError):
    """A checkpoint is incomplete or failed hash validation."""


class IncompatibleCheckpointError(CheckpointError):
    """Resume inputs differ from those recorded in the checkpoint."""


class RunLockedError(CheckpointError):
    """Another process owns the run directory."""


@runtime_checkable
class CheckpointBackend(Protocol):
    """Backend for large, stateful model/optimizer/scheduler objects."""

    name: str

    def save(self, stateful: Mapping[str, Any], path: Path) -> None:
        """Save ``stateful`` below the newly-created ``path``."""

    def load(
        self,
        stateful: MutableMapping[str, Any] | None,
        path: Path,
    ) -> MutableMapping[str, Any]:
        """Load into templates/Stateful objects and return the populated mapping."""


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise CheckpointError("PyTorch is required to save or load training state") from exc
    return torch


def _state_dict_or_value(value: Any) -> Any:
    state_dict = getattr(value, "state_dict", None)
    return state_dict() if callable(state_dict) else value


class TorchFileBackend:
    """Single-process ``torch.save`` compatibility backend."""

    name = "torch-file"

    def save(self, stateful: Mapping[str, Any], path: Path) -> None:
        torch = _require_torch()
        if _torch_distributed_info()[1] > 1:
            raise CheckpointError(
                "TorchFileBackend is only safe for world_size=1; use the DCP backend"
            )
        path.mkdir(parents=True, exist_ok=False)
        payload = {key: _state_dict_or_value(value) for key, value in stateful.items()}
        torch.save(payload, path / "state.pt")

    def load(
        self,
        stateful: MutableMapping[str, Any] | None,
        path: Path,
    ) -> MutableMapping[str, Any]:
        torch = _require_torch()
        try:
            saved = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch before weights_only was added.
            saved = torch.load(path / "state.pt", map_location="cpu")
        if not isinstance(saved, Mapping):
            raise CheckpointCorruptError("torch-file payload is not a mapping")
        if stateful is None:
            return dict(saved)
        _load_values_into(stateful, saved)
        return stateful


def _load_values_into(targets: MutableMapping[str, Any], saved: Mapping[str, Any]) -> None:
    missing = sorted(set(targets) - set(saved))
    if missing:
        raise CheckpointCorruptError(f"checkpoint is missing stateful entries: {missing}")
    for key, target in list(targets.items()):
        value = saved[key]
        load_state_dict = getattr(target, "load_state_dict", None)
        if callable(load_state_dict):
            load_state_dict(value)
        elif isinstance(target, MutableMapping) and isinstance(value, Mapping):
            target.clear()
            target.update(value)
        else:
            targets[key] = value


class TorchDistributedCheckpointBackend:
    """``torch.distributed.checkpoint`` backend with API-version adaptation."""

    name = "torch-distributed-checkpoint"

    def __init__(self, process_group: Any = None) -> None:
        self.process_group = process_group

    @staticmethod
    def available() -> bool:
        try:
            from torch.distributed.checkpoint.state_dict_loader import load as _load  # noqa: F401
            from torch.distributed.checkpoint.state_dict_saver import save as _save  # noqa: F401
        except ImportError:
            return False
        return True

    def save(self, stateful: Mapping[str, Any], path: Path) -> None:
        try:
            from torch.distributed.checkpoint.state_dict_saver import save
        except ImportError as exc:
            raise CheckpointError("torch.distributed.checkpoint is unavailable") from exc
        self._invoke(save, dict(stateful), path, save_mode=True)

    def load(
        self,
        stateful: MutableMapping[str, Any] | None,
        path: Path,
    ) -> MutableMapping[str, Any]:
        if stateful is None:
            raise CheckpointError(
                "DCP load requires stateful templates (model/optimizer/scheduler)"
            )
        try:
            from torch.distributed.checkpoint.state_dict_loader import load
        except ImportError as exc:
            raise CheckpointError("torch.distributed.checkpoint is unavailable") from exc
        self._invoke(load, stateful, path, save_mode=False)
        return stateful

    def _invoke(
        self,
        function: Any,
        stateful: MutableMapping[str, Any] | dict[str, Any],
        path: Path,
        *,
        save_mode: bool,
    ) -> None:
        parameters = inspect.signature(function).parameters
        kwargs: dict[str, Any] = {}
        if "process_group" in parameters and self.process_group is not None:
            kwargs["process_group"] = self.process_group
        if "checkpoint_id" in parameters:
            kwargs["checkpoint_id"] = str(path)
        else:  # Compatibility with older DCP releases.
            try:
                from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
            except ImportError as exc:
                raise CheckpointError("unsupported torch.distributed.checkpoint API") from exc
            keyword = "storage_writer" if save_mode else "storage_reader"
            kwargs[keyword] = FileSystemWriter(str(path)) if save_mode else FileSystemReader(str(path))

        # All supported variants call this mapping ``state_dict``.  Use a
        # keyword so a future optional first argument cannot silently change it.
        function(state_dict=stateful, **kwargs)


@dataclass(slots=True)
class LoadedCheckpoint:
    path: Path
    metadata: dict[str, Any]
    stateful: MutableMapping[str, Any]
    trainer_state: TrainerState
    data_cursor: DataCursor
    rng_state: RNGState
    committed_boundary: CommittedBoundary | None
    source_rank: int
    rng_forked: bool


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    """Cheap evidence that an immutable checkpoint file has not changed."""

    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True, slots=True)
class _CheckpointIdentity:
    """Authenticated identity of one committed checkpoint at one path."""

    path: Path
    manifest_sha256: str
    metadata_sha256: str


@dataclass(slots=True)
class _CommitEvidence:
    """Hashes and file identities observed while constructing a commit."""

    manifest_sha256: str
    metadata_sha256: str
    files: dict[str, _FileIdentity]


class RunLock:
    """Advisory, non-blocking run-directory lock held for a trainer's lifetime."""

    def __init__(self, path: str | Path, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return not self.enabled or self._fd is not None

    def acquire(self) -> RunLock:
        if not self.enabled or self._fd is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            owner = os.read(fd, 4096).decode("utf-8", errors="replace").strip()
            os.close(fd)
            detail = f"; owner={owner}" if owner else ""
            raise RunLockedError(f"run directory is already locked{detail}") from exc

        owner = json.dumps(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "acquired_at": _utc_now(),
            },
            sort_keys=True,
        ).encode("utf-8")
        os.ftruncate(fd, 0)
        os.write(fd, owner)
        os.fsync(fd)
        self._fd = fd
        return self

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> RunLock:
        return self.acquire()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


class CheckpointManager:
    """Coordinate atomic checkpoint commits, validation, loading and retention."""

    def __init__(
        self,
        root: str | Path,
        *,
        backend: CheckpointBackend | Literal["auto", "torch", "dcp"] = "auto",
        rank: int | None = None,
        world_size: int | None = None,
        process_group: Any = None,
        keep_periodic: int = 3,
        keep_interrupt: int = 1,
    ) -> None:
        if keep_periodic < 0 or keep_interrupt < 0:
            raise ValueError("retention counts must be non-negative")
        # Pure checkpoint inspectors (for example the dashboard controller)
        # can supply both values explicitly.  In that case there is nothing
        # to autodetect, and importing torch merely to rediscover ``0/1``
        # would violate their no-PyTorch/no-CUDA process boundary.
        if rank is not None and world_size is not None:
            detected_rank, detected_world_size = rank, world_size
        else:
            detected_rank, detected_world_size = _torch_distributed_info(process_group)
        self.rank = detected_rank if rank is None else rank
        self.world_size = detected_world_size if world_size is None else world_size
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be within world_size")
        self.root = Path(root)
        self.backend_spec = backend
        self.process_group = process_group
        self.keep_periodic = keep_periodic
        self.keep_interrupt = keep_interrupt
        self._run_lock = RunLock(self.root / ".run.lock", enabled=self.rank == 0)
        self._session_lock_acquired = False
        # This is deliberately instance-local: a new process/manager must hash
        # every checkpoint before trusting it.  Entries are populated only by
        # this manager's own commit or a completed full verification.
        self._trusted_checkpoints: dict[
            _CheckpointIdentity, dict[str, _FileIdentity]
        ] = {}

    @property
    def run_lock_held(self) -> bool:
        return self._session_lock_acquired

    def acquire_run_lock(self) -> CheckpointManager:
        if self._session_lock_acquired:
            return self
        error: Exception | None = None
        try:
            self._run_lock.acquire()
        except Exception as exc:  # Propagate rank-zero lock failure to every rank.
            error = exc
        try:
            self._synchronize_error(error, "run lock")
        except Exception:
            self._run_lock.release()
            raise
        self._session_lock_acquired = True
        return self

    def release_run_lock(self) -> None:
        self._run_lock.release()
        self._session_lock_acquired = False

    def __enter__(self) -> CheckpointManager:
        return self.acquire_run_lock()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release_run_lock()

    def save(
        self,
        stateful: Mapping[str, Any],
        *,
        trainer_state: TrainerState,
        data_cursor: DataCursor,
        critical_fingerprint: Any,
        data_fingerprint: Any,
        kind: CheckpointKind = "periodic",
        rng_state: RNGState | None = None,
        committed_boundary: CommittedBoundary | None = None,
        tag: str | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Collectively create and atomically commit a checkpoint.

        If interruption occurs during gradient accumulation, the runtime cursor
        and RNG automatically roll back to ``committed_boundary``.  Model and
        optimizer parameters are still those from that boundary because no
        optimizer step has happened yet; gradients themselves are never saved.
        """

        if kind not in _VALID_KINDS:
            raise ValueError(f"invalid checkpoint kind: {kind}")
        if trainer_state.world_size != self.world_size:
            raise ValueError(
                f"TrainerState world_size={trainer_state.world_size} does not match "
                f"manager world_size={self.world_size}"
            )
        if not stateful:
            raise ValueError("stateful mapping must not be empty")
        if critical_fingerprint is None or data_fingerprint is None:
            raise ValueError("critical and data fingerprints are required")
        rng_state = rng_state or RNGState.capture()
        effective_state, effective_cursor, effective_rng, rollback_applied = rollback_runtime_state(
            trainer_state, data_cursor, rng_state, committed_boundary
        )
        if effective_state.committed_tokens != effective_cursor.global_token_index:
            raise ValueError(
                "TrainerState committed_tokens disagrees with DataCursor global_token_index: "
                f"{effective_state.committed_tokens} != {effective_cursor.global_token_index}"
            )

        checkpoint_id = _checkpoint_id(effective_state.global_step, kind, tag)
        final_path = self.root / checkpoint_id
        incomplete_path = self.root / f".{checkpoint_id}.incomplete"
        backend = self._select_save_backend()
        transient_lock = not self.run_lock_held
        if transient_lock:
            self.acquire_run_lock()
        try:
            self._prepare_incomplete(final_path, incomplete_path)
            runtime_error: Exception | None = None
            try:
                runtime_dir = incomplete_path / "runtime"
                runtime_dir.mkdir(parents=True, exist_ok=True)
                runtime_payload = {
                    "version": RUNTIME_PAYLOAD_VERSION,
                    "trainer_state": effective_state.to_dict(),
                    "data_cursor": effective_cursor.to_dict(),
                    "rng_state": effective_rng.to_dict(),
                    "committed_boundary": (
                        committed_boundary.to_dict() if committed_boundary is not None else None
                    ),
                    "rollback_applied": rollback_applied,
                    "saved_rank": self.rank,
                }
                _pickle_dump(runtime_dir / f"rank-{self.rank:05d}.pkl", runtime_payload)
            except Exception as exc:
                runtime_error = exc
            self._synchronize_error(runtime_error, "per-rank runtime checkpoint")

            backend.save(stateful, incomplete_path / "state")
            self._barrier()

            commit_error: Exception | None = None
            try:
                if self.rank == 0:
                    metadata = {
                        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                        "checkpoint_id": checkpoint_id,
                        "kind": kind,
                        "created_at": _utc_now(),
                        "backend": backend.name,
                        "run_id": effective_state.run_id,
                        "stage": effective_state.stage,
                        "global_step": effective_state.global_step,
                        "committed_tokens": effective_state.committed_tokens,
                        "trainer_state": effective_state.to_dict(),
                        "data_cursor": effective_cursor.to_dict(),
                        "saved_world_size": self.world_size,
                        "global_batch_tokens": effective_state.global_batch_tokens,
                        "gradient_accumulation_steps": (
                            effective_state.gradient_accumulation_steps
                        ),
                        "critical_fingerprint": stable_fingerprint(critical_fingerprint),
                        "data_fingerprint": stable_fingerprint(data_fingerprint),
                        "rollback_applied": rollback_applied,
                        "tag": tag,
                        "extra": dict(extra_metadata or {}),
                    }
                    _write_json(incomplete_path / "metadata.json", metadata)
                    evidence = self._commit_directory(incomplete_path, final_path)
                    # The commit pass already read and hashed every shard. Keep
                    # that evidence only after the rename and identity check;
                    # verify then authenticates the small commit record without
                    # reading the large payload a second time.
                    self._trust_committed_checkpoint(final_path, evidence)
                    self.verify(final_path)
                    self._update_latest(final_path)
                    self.prune(current=final_path)
            except Exception as exc:
                commit_error = exc
            self._synchronize_error(commit_error, "checkpoint commit")
            return final_path
        finally:
            if transient_lock:
                self.release_run_lock()

    def load(
        self,
        stateful: MutableMapping[str, Any] | None,
        checkpoint: str | Path = "auto",
        *,
        expected_critical_fingerprint: Any | None = None,
        expected_data_fingerprint: Any | None = None,
        expected_run_id: str | None = None,
        expected_stage: str | None = None,
        expected_global_batch_tokens: int | None = None,
        restore_rng: bool = False,
        strict_cuda_rng: bool = False,
    ) -> LoadedCheckpoint:
        """Validate compatibility, load stateful objects and return runtime state."""

        path, metadata = self._resolve_for_load(checkpoint)
        self._check_compatibility(
            metadata,
            expected_critical_fingerprint=expected_critical_fingerprint,
            expected_data_fingerprint=expected_data_fingerprint,
            expected_run_id=expected_run_id,
            expected_stage=expected_stage,
            expected_global_batch_tokens=expected_global_batch_tokens,
        )
        runtime_error: Exception | None = None
        trainer_state: TrainerState | None = None
        data_cursor: DataCursor | None = None
        rng_state: RNGState | None = None
        committed_boundary: CommittedBoundary | None = None
        source_rank = -1
        rng_forked = False
        backend: CheckpointBackend | None = None
        try:
            runtime_payload, source_rank = self._load_runtime_payload(path, metadata)
            trainer_state = TrainerState.from_dict(runtime_payload["trainer_state"])
            data_cursor = DataCursor.from_dict(runtime_payload["data_cursor"])
            if trainer_state.committed_tokens != data_cursor.global_token_index:
                raise CheckpointCorruptError(
                    "runtime TrainerState committed_tokens disagrees with DataCursor "
                    "global_token_index"
                )
            rng_state = RNGState.from_dict(runtime_payload["rng_state"])
            committed_raw = runtime_payload.get("committed_boundary")
            committed_boundary = (
                CommittedBoundary.from_dict(committed_raw) if committed_raw is not None else None
            )
            saved_world_size = int(metadata["saved_world_size"])
            rng_forked = self.rank >= saved_world_size
            if rng_forked:
                rng_state = rng_state.fork_for_rank(self.rank)
            backend = self._backend_for_load(str(metadata["backend"]))
        except Exception as exc:
            runtime_error = (
                exc
                if isinstance(exc, CheckpointError)
                else CheckpointCorruptError(f"per-rank runtime state is invalid: {exc}")
            )
        self._synchronize_error(runtime_error, "checkpoint runtime load")
        if (
            trainer_state is None
            or data_cursor is None
            or rng_state is None
            or backend is None
        ):
            raise CheckpointCorruptError("per-rank runtime state is invalid")
        loaded_stateful = backend.load(stateful, path / "state")
        restore_error: Exception | None = None
        if restore_rng:
            try:
                rng_state.restore(strict_cuda=strict_cuda_rng)
            except Exception as exc:
                restore_error = exc
        self._synchronize_error(restore_error, "RNG restoration")
        return LoadedCheckpoint(
            path=path,
            metadata=metadata,
            stateful=loaded_stateful,
            trainer_state=trainer_state,
            data_cursor=data_cursor,
            rng_state=rng_state,
            committed_boundary=committed_boundary,
            source_rank=source_rank,
            rng_forked=rng_forked,
        )

    def resolve(self, checkpoint: str | Path = "auto") -> Path:
        return self._resolve_verified(checkpoint)[0]

    def _resolve_verified(self, checkpoint: str | Path) -> tuple[Path, dict[str, Any]]:
        if str(checkpoint) != "auto":
            candidate = Path(checkpoint)
            if not candidate.is_absolute():
                # A bare checkpoint name is relative to this manager's run
                # root.  An explicitly pathed argument is user input relative
                # to the current working directory; prefixing the run root in
                # that case silently produces ``run/run/step-*`` for the
                # common ``--checkpoint runs/<run>/step-*`` CLI spelling.
                raw = str(checkpoint)
                explicitly_pathed = len(candidate.parts) > 1 or raw.startswith(("./", "../"))
                candidate = candidate.resolve() if explicitly_pathed else self.root / candidate
            return candidate, self.verify(candidate)

        # COMPLETE is committed before ``latest`` is atomically replaced.  A
        # process can therefore die in that narrow window and leave a valid,
        # newer directory behind an older (but still valid) pointer.  Rank
        # candidates using authenticated small metadata, then fully hash them
        # newest-first until one succeeds.  Usually only one large checkpoint
        # needs to be read.
        for path, _ in reversed(self._authenticated_checkpoints()):
            try:
                return path, self.verify(path)
            except CheckpointError:
                continue
        raise CheckpointError(f"no complete checkpoint found under {self.root}")

    def _resolve_for_load(self, checkpoint: str | Path) -> tuple[Path, dict[str, Any]]:
        """Resolve/hash on rank zero and broadcast the authenticated result."""

        if self.world_size == 1:
            return self._resolve_verified(checkpoint)
        try:
            import torch.distributed as dist
        except ImportError as exc:
            raise CheckpointError("world_size > 1 but torch.distributed is unavailable") from exc
        if not dist.is_available() or not dist.is_initialized():
            raise CheckpointError("world_size > 1 but the process group is not initialized")

        local_error: Exception | None = None
        payload: dict[str, Any] | None = None
        if self.rank == 0:
            try:
                path, metadata = self._resolve_verified(checkpoint)
                payload = {"path": str(path), "metadata": metadata, "error": None}
            except Exception as exc:
                local_error = exc
                payload = {
                    "path": None,
                    "metadata": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        objects: list[dict[str, Any] | None] = [payload]
        source_rank = 0
        if self.process_group is not None and hasattr(dist, "get_global_rank"):
            source_rank = dist.get_global_rank(self.process_group, 0)
        dist.broadcast_object_list(objects, src=source_rank, group=self.process_group)
        result = objects[0]
        if not isinstance(result, dict):
            raise CheckpointError("rank zero broadcast an invalid checkpoint resolution")
        if result.get("error") is not None:
            if local_error is not None:
                raise local_error
            raise CheckpointError(f"checkpoint verification failed on rank zero: {result['error']}")
        metadata = result.get("metadata")
        if not isinstance(metadata, dict) or not isinstance(result.get("path"), str):
            raise CheckpointError("rank zero broadcast incomplete checkpoint metadata")
        return Path(result["path"]), metadata

    def find_latest(self) -> Path:
        """Return the newest hash-valid checkpoint or raise if none exists."""

        return self.resolve("auto")

    def find_latest_valid(self) -> Path | None:
        """Inspection-friendly variant that returns ``None`` for an empty run."""

        resolved = self.find_latest_valid_with_metadata()
        return resolved[0] if resolved is not None else None

    def find_latest_valid_with_metadata(self) -> tuple[Path, dict[str, Any]] | None:
        """Return the newest valid path and metadata after one full resolution."""

        with suppress(CheckpointError):
            return self._resolve_verified("auto")
        return None

    def inspect(self, checkpoint: str | Path = "auto") -> dict[str, Any]:
        """Hash-validate a checkpoint and return its JSON metadata only."""

        return self._resolve_verified(checkpoint)[1]

    def valid_checkpoints(self) -> list[tuple[Path, dict[str, Any]]]:
        if not self.root.exists():
            return []
        valid: list[tuple[Path, dict[str, Any]]] = []
        for path in self.root.iterdir():
            if not path.is_dir() or not path.name.startswith("step-"):
                continue
            try:
                metadata = self.verify(path)
            except CheckpointError:
                continue
            valid.append((path, metadata))
        valid.sort(
            key=lambda item: (
                int(item[1].get("global_step", -1)),
                str(item[1].get("created_at", "")),
                item[0].name,
            )
        )
        return valid

    def _authenticated_checkpoints(self) -> list[tuple[Path, dict[str, Any]]]:
        """List COMPLETE checkpoints after authenticating their small metadata."""

        if not self.root.exists():
            return []
        candidates: list[tuple[Path, dict[str, Any]]] = []
        for path in self.root.iterdir():
            if not path.is_dir() or not path.name.startswith("step-"):
                continue
            with suppress(CheckpointError):
                candidates.append((path, _read_authenticated_metadata(path)))
        candidates.sort(
            key=lambda item: (
                int(item[1].get("global_step", -1)),
                str(item[1].get("created_at", "")),
                item[0].name,
            )
        )
        return candidates

    def verify(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        if not path.is_dir():
            self._forget_trusted_path(path)
            raise CheckpointCorruptError(f"checkpoint directory does not exist: {path}")
        complete_path = path / "COMPLETE"
        manifest_path = path / "manifest.json"
        try:
            complete_bytes, _ = _read_stable_file(complete_path)
            manifest_bytes, _ = _read_stable_file(manifest_path)
            expected_manifest_hash = complete_bytes.decode("ascii").strip()
        except (OSError, UnicodeDecodeError, CheckpointCorruptError) as exc:
            self._forget_trusted_path(path)
            raise CheckpointCorruptError(f"checkpoint is incomplete: {path}") from exc
        if not _SHA256_RE.fullmatch(expected_manifest_hash):
            self._forget_trusted_path(path)
            raise CheckpointCorruptError("COMPLETE does not contain a SHA256 digest")
        actual_manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        if actual_manifest_hash != expected_manifest_hash:
            self._forget_trusted_path(path)
            raise CheckpointCorruptError("manifest hash does not match COMPLETE")
        try:
            manifest = json.loads(manifest_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._forget_trusted_path(path)
            raise CheckpointCorruptError("manifest is not valid JSON") from exc
        if not isinstance(manifest, dict):
            self._forget_trusted_path(path)
            raise CheckpointCorruptError("manifest is not a JSON object")
        if manifest.get("version") != MANIFEST_VERSION:
            self._forget_trusted_path(path)
            raise CheckpointCorruptError("unsupported manifest version")
        files = manifest.get("files")
        if not isinstance(files, dict) or "metadata.json" not in files:
            self._forget_trusted_path(path)
            raise CheckpointCorruptError("manifest has no metadata entry")
        if not any(str(relative).startswith("state/") for relative in files):
            self._forget_trusted_path(path)
            raise CheckpointCorruptError("manifest has no state backend payload")
        for relative, expected_hash in files.items():
            if not isinstance(relative, str):
                self._forget_trusted_path(path)
                raise CheckpointCorruptError("manifest paths must be strings")
            _safe_child(path, relative)
            if not _SHA256_RE.fullmatch(str(expected_hash)):
                self._forget_trusted_path(path)
                raise CheckpointCorruptError(f"invalid manifest entry: {relative}")
        try:
            metadata_bytes, metadata_identity = _read_stable_file(path / "metadata.json")
            expected_metadata_hash = str(files["metadata.json"])
            if hashlib.sha256(metadata_bytes).hexdigest() != expected_metadata_hash:
                raise CheckpointCorruptError("metadata hash mismatch")
            metadata = json.loads(metadata_bytes)
            if not isinstance(metadata, dict):
                raise CheckpointCorruptError("metadata is not a JSON object")
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._forget_trusted_path(path)
            raise CheckpointCorruptError("metadata is not valid JSON") from exc
        except CheckpointCorruptError:
            self._forget_trusted_path(path)
            raise
        try:
            version = int(metadata.get("checkpoint_schema_version", 0))
            saved_world_size = int(metadata.get("saved_world_size", 0))
            global_step = int(metadata.get("global_step", -1))
            committed_tokens = int(metadata.get("committed_tokens", -1))
        except (TypeError, ValueError) as exc:
            raise CheckpointCorruptError("checkpoint metadata has invalid integer fields") from exc
        if version != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointCorruptError(
                f"unsupported checkpoint schema {version}; expected {CHECKPOINT_SCHEMA_VERSION}"
            )
        if metadata.get("checkpoint_id") != path.name:
            raise CheckpointCorruptError("checkpoint directory name does not match metadata")
        if metadata.get("kind") not in _VALID_KINDS:
            raise CheckpointCorruptError("checkpoint metadata has an invalid kind")
        if saved_world_size < 1 or global_step < 0 or committed_tokens < 0:
            raise CheckpointCorruptError("checkpoint metadata has invalid counters")
        raw_trainer_state = metadata.get("trainer_state")
        raw_data_cursor = metadata.get("data_cursor")
        if not isinstance(raw_trainer_state, Mapping) or not isinstance(raw_data_cursor, Mapping):
            raise CheckpointCorruptError("checkpoint metadata has invalid runtime state")
        try:
            state_tokens = int(raw_trainer_state.get("committed_tokens", -1))
            cursor_tokens = int(raw_data_cursor.get("global_token_index", -1))
        except (TypeError, ValueError) as exc:
            raise CheckpointCorruptError(
                "checkpoint metadata has invalid committed token counters"
            ) from exc
        if committed_tokens != state_tokens or committed_tokens != cursor_tokens:
            raise CheckpointCorruptError(
                "checkpoint committed token counters disagree between metadata, "
                "TrainerState, and DataCursor"
            )
        if not isinstance(metadata.get("backend"), str):
            raise CheckpointCorruptError("checkpoint metadata has no backend")
        for name in ("critical_fingerprint", "data_fingerprint"):
            if not _SHA256_RE.fullmatch(str(metadata.get(name, ""))):
                raise CheckpointCorruptError(f"checkpoint metadata has invalid {name}")
        for rank in range(saved_world_size):
            relative = f"runtime/rank-{rank:05d}.pkl"
            if relative not in files:
                raise CheckpointCorruptError(f"checkpoint has no runtime payload for rank {rank}")

        identity = _CheckpointIdentity(
            path=_cache_path(path),
            manifest_sha256=expected_manifest_hash,
            metadata_sha256=expected_metadata_hash,
        )
        if self._trusted_checkpoint_is_unchanged(identity, path, files):
            return metadata

        # No trust crosses manager/process boundaries.  The first observation
        # of an existing checkpoint always reads and hashes every manifest
        # entry before installing an instance-local cache entry.
        self._forget_trusted_path(path)
        verified_files: dict[str, _FileIdentity] = {}
        for relative, expected_hash in files.items():
            file_path = _safe_child(path, relative)
            if relative == "metadata.json":
                actual_hash = hashlib.sha256(metadata_bytes).hexdigest()
                file_identity = metadata_identity
            else:
                actual_hash, file_identity = _sha256_stable_file(file_path)
            if actual_hash != expected_hash:
                raise CheckpointCorruptError(f"hash mismatch: {relative}")
            verified_files[relative] = file_identity
        self._trusted_checkpoints[identity] = verified_files
        return metadata

    def _trusted_checkpoint_is_unchanged(
        self,
        identity: _CheckpointIdentity,
        path: Path,
        files: Mapping[str, Any],
    ) -> bool:
        trusted_files = self._trusted_checkpoints.get(identity)
        if trusted_files is None or trusted_files.keys() != files.keys():
            return False
        try:
            return all(
                _file_identity(_safe_child(path, relative)) == trusted_identity
                for relative, trusted_identity in trusted_files.items()
            )
        except (OSError, CheckpointCorruptError):
            return False

    def _trust_committed_checkpoint(self, path: Path, evidence: _CommitEvidence) -> None:
        """Trust hashes produced by this manager's just-completed commit pass."""

        try:
            unchanged = all(
                _file_identity(_safe_child(path, relative)) == committed_identity
                for relative, committed_identity in evidence.files.items()
            )
        except (OSError, CheckpointCorruptError):
            unchanged = False
        if not unchanged:
            self._forget_trusted_path(path)
            raise CheckpointCorruptError("checkpoint files changed while being committed")
        identity = _CheckpointIdentity(
            path=_cache_path(path),
            manifest_sha256=evidence.manifest_sha256,
            metadata_sha256=evidence.metadata_sha256,
        )
        self._forget_trusted_path(path)
        self._trusted_checkpoints[identity] = dict(evidence.files)

    def _forget_trusted_path(self, path: Path) -> None:
        cache_path = _cache_path(path)
        stale = [identity for identity in self._trusted_checkpoints if identity.path == cache_path]
        for identity in stale:
            del self._trusted_checkpoints[identity]

    def prune(self, *, current: Path | None = None) -> list[Path]:
        """Retain last N periodic/interrupt checkpoints and all milestones.

        Only fully hash-valid checkpoints count toward retention. A checkpoint
        with intact metadata but a damaged tensor shard must never displace a
        usable recovery point.
        """

        grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = {
            "periodic": [],
            "interrupt": [],
            "milestone": [],
        }
        for path, metadata in self.valid_checkpoints():
            grouped[str(metadata["kind"])].append((path, metadata))
        remove: list[Path] = []
        for kind, keep in (("periodic", self.keep_periodic), ("interrupt", self.keep_interrupt)):
            entries = sorted(
                grouped[kind],
                key=lambda item: (
                    int(item[1].get("global_step", -1)),
                    str(item[1].get("created_at", "")),
                    item[0].name,
                ),
            )
            expired = entries[:-keep] if keep else entries
            for path, _ in expired:
                if current is None or path != current:
                    remove.append(path)
        for path in remove:
            self._forget_trusted_path(path)
            _safe_rmtree(self.root, path)
        if remove:
            _fsync_dir(self.root)
        return remove

    def _prepare_incomplete(self, final_path: Path, incomplete_path: Path) -> None:
        error: Exception | None = None
        try:
            if self.rank == 0:
                self.root.mkdir(parents=True, exist_ok=True)
                if final_path.exists():
                    try:
                        self.verify(final_path)
                    except CheckpointCorruptError:
                        # Auto-resume may legitimately replay a step whose old
                        # checkpoint directory is present but corrupt. Preserve
                        # it for forensics under a name ignored by discovery,
                        # then let the replay commit the canonical name.
                        suffix = 0
                        while True:
                            extra = f"-{suffix}" if suffix else ""
                            quarantine = self.root / (
                                f".corrupt-{final_path.name}-{os.getpid()}{extra}"
                            )
                            if not quarantine.exists():
                                break
                            suffix += 1
                        self._forget_trusted_path(final_path)
                        os.replace(final_path, quarantine)
                        _fsync_dir(self.root)
                    else:
                        raise CheckpointError(
                            f"refusing to overwrite checkpoint: {final_path}"
                        )
                if incomplete_path.exists():
                    _safe_rmtree(self.root, incomplete_path)
                incomplete_path.mkdir(parents=False, exist_ok=False)
        except Exception as exc:
            error = exc
        self._synchronize_error(error, "checkpoint preparation")

    def _commit_directory(
        self, incomplete_path: Path, final_path: Path
    ) -> _CommitEvidence:
        files: dict[str, str] = {}
        file_identities: dict[str, _FileIdentity] = {}
        for file_path in sorted(incomplete_path.rglob("*")):
            if file_path.is_file() and file_path.name not in ("manifest.json", "COMPLETE"):
                relative = file_path.relative_to(incomplete_path).as_posix()
                file_hash, file_identity = _sha256_stable_file(file_path)
                files[relative] = file_hash
                file_identities[relative] = file_identity
        manifest = {"version": MANIFEST_VERSION, "algorithm": "sha256", "files": files}
        manifest_path = incomplete_path / "manifest.json"
        _write_json(manifest_path, manifest)
        manifest_hash = _sha256_file(manifest_path)
        _write_bytes(incomplete_path / "COMPLETE", f"{manifest_hash}\n".encode("ascii"))
        _fsync_tree(incomplete_path)
        os.replace(incomplete_path, final_path)
        _fsync_dir(self.root)
        return _CommitEvidence(
            manifest_sha256=manifest_hash,
            metadata_sha256=files["metadata.json"],
            files=file_identities,
        )

    def _update_latest(self, final_path: Path) -> None:
        temporary = self.root / f".latest.{os.getpid()}.tmp"
        _write_bytes(temporary, f"{final_path.name}\n".encode())
        os.replace(temporary, self.root / "latest")
        _fsync_dir(self.root)

    def _select_save_backend(self) -> CheckpointBackend:
        if not isinstance(self.backend_spec, str):
            return self.backend_spec
        if self.backend_spec == "torch":
            return TorchFileBackend()
        if self.backend_spec == "dcp":
            if not TorchDistributedCheckpointBackend.available():
                raise CheckpointError("requested DCP backend is unavailable")
            return TorchDistributedCheckpointBackend(self.process_group)
        if self.backend_spec == "auto":
            # Use DCP even for a one-rank run so that checkpoint can later be
            # resumed with more GPUs.  Plain torch.save is a compatibility
            # fallback only when DCP is absent and world_size is exactly one.
            if TorchDistributedCheckpointBackend.available():
                return TorchDistributedCheckpointBackend(self.process_group)
            if self.world_size == 1:
                return TorchFileBackend()
            raise CheckpointError(
                "distributed checkpointing requires torch.distributed.checkpoint"
            )
        raise ValueError(f"unknown checkpoint backend: {self.backend_spec}")

    def _backend_for_load(self, name: str) -> CheckpointBackend:
        if not isinstance(self.backend_spec, str):
            if self.backend_spec.name != name:
                raise CheckpointError(
                    f"checkpoint backend is {name}, supplied backend is {self.backend_spec.name}"
                )
            return self.backend_spec
        if name == TorchFileBackend.name:
            if self.world_size > 1:
                raise CheckpointError("a single-process torch-file checkpoint cannot load on many ranks")
            return TorchFileBackend()
        if name == TorchDistributedCheckpointBackend.name:
            if not TorchDistributedCheckpointBackend.available():
                raise CheckpointError("checkpoint requires unavailable DCP support")
            return TorchDistributedCheckpointBackend(self.process_group)
        raise CheckpointError(f"unknown checkpoint backend recorded in metadata: {name}")

    def _load_runtime_payload(
        self, path: Path, metadata: Mapping[str, Any]
    ) -> tuple[dict[str, Any], int]:
        saved_world_size = int(metadata.get("saved_world_size", 1))
        if saved_world_size < 1:
            raise CheckpointCorruptError("saved_world_size must be positive")
        source_rank = self.rank if self.rank < saved_world_size else self.rank % saved_world_size
        runtime_path = path / "runtime" / f"rank-{source_rank:05d}.pkl"
        try:
            payload = _pickle_load(runtime_path)
        except (OSError, pickle.PickleError, EOFError) as exc:
            raise CheckpointCorruptError(f"cannot load runtime state for rank {source_rank}") from exc
        if not isinstance(payload, dict) or payload.get("version") != RUNTIME_PAYLOAD_VERSION:
            raise CheckpointCorruptError("unsupported per-rank runtime payload")
        try:
            payload_rank = int(payload.get("saved_rank", -1))
        except (TypeError, ValueError) as exc:
            raise CheckpointCorruptError("per-rank runtime payload has an invalid rank") from exc
        if payload_rank != source_rank:
            raise CheckpointCorruptError("per-rank runtime payload has the wrong rank")
        state = payload.get("trainer_state")
        if not isinstance(state, Mapping):
            raise CheckpointCorruptError("per-rank runtime payload has no TrainerState")
        if stable_fingerprint(state) != stable_fingerprint(metadata.get("trainer_state")):
            raise CheckpointCorruptError("runtime TrainerState disagrees with metadata")
        cursor = payload.get("data_cursor")
        if stable_fingerprint(cursor) != stable_fingerprint(metadata.get("data_cursor")):
            raise CheckpointCorruptError("runtime DataCursor disagrees with metadata")
        return payload, source_rank

    def _check_compatibility(
        self,
        metadata: Mapping[str, Any],
        *,
        expected_critical_fingerprint: Any | None,
        expected_data_fingerprint: Any | None,
        expected_run_id: str | None,
        expected_stage: str | None,
        expected_global_batch_tokens: int | None,
    ) -> None:
        comparisons = (
            (
                "critical configuration",
                metadata.get("critical_fingerprint"),
                expected_critical_fingerprint,
            ),
            ("data manifest", metadata.get("data_fingerprint"), expected_data_fingerprint),
        )
        for label, saved, expected in comparisons:
            if expected is not None and saved != stable_fingerprint(expected):
                raise IncompatibleCheckpointError(
                    f"{label} fingerprint differs: saved={saved}, "
                    f"expected={stable_fingerprint(expected)}"
                )
        if expected_run_id is not None and metadata.get("run_id") != expected_run_id:
            raise IncompatibleCheckpointError("run_id differs; use fork-from for a new run")
        if expected_stage is not None and metadata.get("stage") != expected_stage:
            raise IncompatibleCheckpointError("training stage differs; use a stage milestone")
        if (
            expected_global_batch_tokens is not None
            and int(metadata.get("global_batch_tokens", -1)) != expected_global_batch_tokens
        ):
            raise IncompatibleCheckpointError(
                "global batch tokens changed; adjust gradient accumulation before resume"
            )

    def _barrier(self) -> None:
        if self.world_size == 1:
            return
        try:
            import torch.distributed as dist
        except ImportError as exc:
            raise CheckpointError("world_size > 1 but torch.distributed is unavailable") from exc
        if not dist.is_available() or not dist.is_initialized():
            raise CheckpointError("world_size > 1 but the process group is not initialized")
        dist.barrier(group=self.process_group)

    def _synchronize_error(self, local_error: Exception | None, context: str) -> None:
        """Make pre/post-collective filesystem failures visible to every rank."""

        if self.world_size == 1:
            if local_error is not None:
                raise local_error
            return
        try:
            import torch.distributed as dist
        except ImportError as exc:
            raise CheckpointError("world_size > 1 but torch.distributed is unavailable") from exc
        if not dist.is_available() or not dist.is_initialized():
            raise CheckpointError("world_size > 1 but the process group is not initialized")
        local_message = (
            f"rank {self.rank}: {type(local_error).__name__}: {local_error}"
            if local_error is not None
            else None
        )
        messages: list[str | None] = [None] * self.world_size
        dist.all_gather_object(messages, local_message, group=self.process_group)
        failure = next((message for message in messages if message is not None), None)
        if failure is not None:
            if local_error is not None:
                raise local_error
            raise CheckpointError(f"{context} failed on another rank: {failure}")


def stable_fingerprint(value: Any) -> str:
    """Canonical SHA256 for configuration/data manifests.

    A 64-character lowercase SHA256 string is accepted as an already-computed
    fingerprint.  Other values are canonicalised as JSON first.
    """

    if isinstance(value, str) and _SHA256_RE.fullmatch(value):
        return value
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _checkpoint_id(global_step: int, kind: str, tag: str | None) -> str:
    name = f"step-{global_step:012d}-{kind}"
    if tag:
        cleaned = _TAG_RE.sub("-", tag).strip(".-")
        if not cleaned:
            raise ValueError("checkpoint tag contains no safe characters")
        name += f"-{cleaned[:80]}"
    return name


def _torch_distributed_info(process_group: Any = None) -> tuple[int, int]:
    try:
        import torch.distributed as dist
    except ImportError:
        return 0, 1
    if not dist.is_available() or not dist.is_initialized():
        return 0, 1
    return dist.get_rank(group=process_group), dist.get_world_size(group=process_group)


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="microseconds")


def _write_json(path: Path, value: Any) -> None:
    encoded = (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")
    _write_bytes(path, encoded)


def _write_bytes(path: Path, value: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _pickle_dump(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=5)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def _pickle_load(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_path(path: Path) -> Path:
    """Normalize a cache key without requiring the path to keep existing."""

    return Path(os.path.abspath(path))


def _file_identity(path: Path) -> _FileIdentity:
    try:
        status = path.stat()
    except OSError as exc:
        raise CheckpointCorruptError(f"checkpoint entry is unavailable: {path}") from exc
    if not stat.S_ISREG(status.st_mode):
        raise CheckpointCorruptError(f"checkpoint entry is not a regular file: {path}")
    return _FileIdentity(
        device=status.st_dev,
        inode=status.st_ino,
        mode=status.st_mode,
        size=status.st_size,
        modified_ns=status.st_mtime_ns,
        changed_ns=status.st_ctime_ns,
    )


def _read_stable_file(path: Path) -> tuple[bytes, _FileIdentity]:
    """Read a small commit file and reject concurrent replacement/mutation."""

    before = _file_identity(path)
    try:
        value = path.read_bytes()
    except OSError as exc:
        raise CheckpointCorruptError(f"cannot read checkpoint entry: {path}") from exc
    after = _file_identity(path)
    if before != after:
        raise CheckpointCorruptError(f"checkpoint entry changed while reading: {path}")
    return value, after


def _sha256_stable_file(path: Path) -> tuple[str, _FileIdentity]:
    """Hash one payload and retain cheap evidence of the exact file read."""

    before = _file_identity(path)
    try:
        digest = _sha256_file(path)
    except OSError as exc:
        raise CheckpointCorruptError(f"cannot hash checkpoint entry: {path}") from exc
    after = _file_identity(path)
    if before != after:
        raise CheckpointCorruptError(f"checkpoint entry changed while hashing: {path}")
    return digest, after


def _read_authenticated_metadata(path: Path) -> dict[str, Any]:
    """Authenticate small commit metadata without re-reading tensor shards."""

    try:
        expected_manifest_hash = (path / "COMPLETE").read_text(encoding="ascii").strip()
        manifest_bytes = (path / "manifest.json").read_bytes()
    except OSError as exc:
        raise CheckpointCorruptError(f"checkpoint is incomplete: {path}") from exc
    if not _SHA256_RE.fullmatch(expected_manifest_hash):
        raise CheckpointCorruptError("invalid COMPLETE digest")
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_hash:
        raise CheckpointCorruptError("manifest hash does not match COMPLETE")
    try:
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict):
            raise CheckpointCorruptError("manifest is not a JSON object")
        if manifest.get("version") != MANIFEST_VERSION:
            raise CheckpointCorruptError("unsupported manifest version")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise CheckpointCorruptError("manifest has no files mapping")
        expected_metadata_hash = files["metadata.json"]
    except CheckpointCorruptError:
        raise
    except (KeyError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CheckpointCorruptError("manifest has no metadata hash") from exc
    metadata_path = path / "metadata.json"
    if not _SHA256_RE.fullmatch(str(expected_metadata_hash)):
        raise CheckpointCorruptError("manifest has an invalid metadata hash")
    if not metadata_path.is_file() or _sha256_file(metadata_path) != expected_metadata_hash:
        raise CheckpointCorruptError("metadata hash mismatch")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise CheckpointCorruptError("metadata is not a JSON object")
        version = int(metadata.get("checkpoint_schema_version", 0))
        global_step = int(metadata.get("global_step", -1))
    except (OSError, TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CheckpointCorruptError("metadata is invalid") from exc
    if version != CHECKPOINT_SCHEMA_VERSION or global_step < 0:
        raise CheckpointCorruptError("metadata schema/counters are invalid")
    if metadata.get("checkpoint_id") != path.name or metadata.get("kind") not in _VALID_KINDS:
        raise CheckpointCorruptError("metadata identity/kind is invalid")
    return metadata


def _safe_child(root: Path, relative: str) -> Path:
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise CheckpointCorruptError(f"unsafe manifest path: {relative}")
    candidate = root / candidate_relative
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise CheckpointCorruptError(f"manifest path escapes checkpoint: {relative}") from exc
    return candidate


def _safe_rmtree(root: Path, path: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise CheckpointError(f"refusing to delete outside checkpoint root: {path}") from exc
    if path == root:
        raise CheckpointError("refusing to delete checkpoint root")
    shutil.rmtree(path)


def _fsync_tree(root: Path) -> None:
    directories: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        elif path.is_dir():
            directories.append(path)
    for directory in reversed(directories):
        _fsync_dir(directory)
    _fsync_dir(root)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointBackend",
    "CheckpointCorruptError",
    "CheckpointError",
    "CheckpointManager",
    "IncompatibleCheckpointError",
    "LoadedCheckpoint",
    "RunLock",
    "RunLockedError",
    "TorchDistributedCheckpointBackend",
    "TorchFileBackend",
    "stable_fingerprint",
]
