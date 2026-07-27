"""Local, dependency-free training dashboard and guarded process controller.

The dashboard intentionally stays outside the training process.  It tails the
durable JSON/JSONL files already written by rank zero and therefore introduces
no CUDA work, profiler hooks, or synchronization into the hot path.
"""

from __future__ import annotations

import base64
import ctypes
import fcntl
import hashlib
import io
import ipaddress
import json
import math
import os
import platform
import re
import secrets
import select
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import zipfile
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import yaml

from .config import ConfigError, load_train_config
from .governed import (
    GovernedControllerError,
    build_governed_plan,
)
from .governed import (
    authorize_run as authorize_governed_run,
)
from .governed import (
    build_train_command as build_governed_train_command,
)
from .governed import (
    controller_status as governed_controller_status,
)
from .governed import (
    load_controller_state as load_governed_controller_state,
)
from .runtime.checkpoint import CheckpointManager
from .source_identity import SOURCE_TREE_HASH_SCHEMA
from .utils import sha256_file


class DashboardError(RuntimeError):
    """Raised for invalid dashboard configuration or guarded actions."""


_GOVERNED_DYNAMIC_DATA_IDENTITIES = (
    "manifest_path",
    "manifest_sha256",
    "source_map_sha256",
    "quality_cooldown_manifest_path",
    "quality_cooldown_manifest_sha256",
    "phase_disjointness_attestation_path",
    "phase_disjointness_attestation_sha256",
)


class _DashboardTermination(BaseException):
    """Internal main-thread unwind used for graceful server SIGTERM handling."""


_NVIDIA_SMI_COMMAND = (
    "/usr/lib/wsl/lib/nvidia-smi",
    "--id=0",
    "--query-gpu=power.draw,power.limit,utilization.gpu,utilization.memory,"
    "clocks.sm,memory.used,memory.free,temperature.gpu",
    "--format=csv,noheader,nounits",
    "--loop-ms=100",
)
_GPU_TELEMETRY_SAMPLE_INTERVAL_SECONDS = 0.1
_GPU_TELEMETRY_OUTPUT_TIMEOUT_SECONDS = 1.5
_GPU_TELEMETRY_HISTORY_SAMPLES = 600
_GPU_TELEMETRY_READ_POLL_SECONDS = 0.1
_GPU_TELEMETRY_RESTART_BACKOFF_SECONDS = 0.25
_GPU_TELEMETRY_MAX_RESTART_BACKOFF_SECONDS = 5.0
_GPU_TELEMETRY_MAX_PARTIAL_OUTPUT_BYTES = 64 * 1024
_GPU_TELEMETRY_JOURNAL_BUCKET_SECONDS = 10.0
_GPU_TELEMETRY_JOURNAL_MAX_BYTES = 16 * 1024 * 1024
_DASHBOARD_ACTION_LOCK_TIMEOUT_SECONDS = 5.0
_DASHBOARD_ACTION_LOCK_POLL_SECONDS = 0.05
_ACTIVE_TRAINING_STATES = frozenset(("running", "launching", "stop_requested"))
_COMPLETED_TRAINING_STATES = frozenset(("complete", "completed", "already_complete"))
_KD_ORCHESTRATION_STATUS_KIND = "twen_base_v2_500m_kd_orchestration_status"
_GPU_TELEMETRY_FIELDS = (
    "power_draw_w",
    "power_limit_w",
    "power_percent_of_limit",
    "gpu_utilization_percent",
    "memory_utilization_percent",
    "sm_clock_mhz",
    "vram_used_mib",
    "vram_free_mib",
    "vram_total_mib",
    "temperature_c",
)
_MEMFD_CLOEXEC = 0x0001
_MEMFD_ALLOW_SEALING = 0x0002
_F_ADD_SEALS = 1033
_F_GET_SEALS = 1034
_F_SEAL_SEAL = 0x0001
_F_SEAL_SHRINK = 0x0002
_F_SEAL_GROW = 0x0004
_F_SEAL_WRITE = 0x0008
_REQUIRED_MEMFD_SEALS = (
    _F_SEAL_SEAL | _F_SEAL_SHRINK | _F_SEAL_GROW | _F_SEAL_WRITE
)
_GOVERNED_SOURCE_ARCHIVE_MAX_BYTES = 64 * 1024 * 1024
_GOVERNED_SOURCE_ARCHIVE_MAX_FILES = 4096
_GOVERNED_SOURCE_ARCHIVE_MAX_UNCOMPRESSED_BYTES = 128 * 1024 * 1024


def _numeric_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "min": ordered[0],
        "mean": math.fsum(values) / len(values),
        "p95": ordered[p95_index],
        "max": ordered[-1],
        "last": values[-1],
    }


class GpuTelemetryMonitor:
    """Own one high-rate, read-only RTX telemetry stream with graceful fallback.

    The executable, GPU index, query fields, and output format are fixed argv;
    neither HTTP input nor dashboard configuration can alter the command.  The
    sampler runs outside the training process, retains raw 100 ms samples only
    in a bounded memory window, and persists bounded 10-second aggregates.
    """

    def __init__(
        self,
        *,
        process_factory: Callable[..., Any] = subprocess.Popen,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        history_samples: int = _GPU_TELEMETRY_HISTORY_SAMPLES,
        journal_path: Path | None = None,
        output_timeout_seconds: float = _GPU_TELEMETRY_OUTPUT_TIMEOUT_SECONDS,
        restart_backoff_seconds: float = _GPU_TELEMETRY_RESTART_BACKOFF_SECONDS,
        max_restart_backoff_seconds: float = (_GPU_TELEMETRY_MAX_RESTART_BACKOFF_SECONDS),
        journal_bucket_seconds: float = _GPU_TELEMETRY_JOURNAL_BUCKET_SECONDS,
        journal_max_bytes: int = _GPU_TELEMETRY_JOURNAL_MAX_BYTES,
    ) -> None:
        if isinstance(history_samples, bool) or not isinstance(history_samples, int):
            raise ValueError("GPU telemetry history size must be an integer")
        if history_samples <= 0 or history_samples > 10_000:
            raise ValueError("GPU telemetry history size must be in 1..10000")
        for value, label in (
            (output_timeout_seconds, "output timeout"),
            (restart_backoff_seconds, "restart backoff"),
            (max_restart_backoff_seconds, "maximum restart backoff"),
            (journal_bucket_seconds, "journal bucket"),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"GPU telemetry {label} must be finite and positive")
        if max_restart_backoff_seconds < restart_backoff_seconds:
            raise ValueError(
                "GPU telemetry maximum restart backoff must not be smaller than restart backoff"
            )
        if isinstance(journal_max_bytes, bool) or not isinstance(journal_max_bytes, int):
            raise ValueError("GPU telemetry journal byte limit must be an integer")
        if journal_max_bytes < 4096:
            raise ValueError("GPU telemetry journal byte limit must be at least 4096")
        self._process_factory = process_factory
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self.sample_interval_seconds = _GPU_TELEMETRY_SAMPLE_INTERVAL_SECONDS
        self.history_samples = history_samples
        self.journal_path = journal_path
        self.output_timeout_seconds = float(output_timeout_seconds)
        self.restart_backoff_seconds = float(restart_backoff_seconds)
        self.max_restart_backoff_seconds = float(max_restart_backoff_seconds)
        self.journal_bucket_seconds = float(journal_bucket_seconds)
        self.journal_max_bytes = journal_max_bytes
        self._lock = threading.Lock()
        self._latest: dict[str, Any] | None = None
        self._history: deque[dict[str, Any]] = deque(maxlen=history_samples)
        self._sequence = 0
        self._sampler_state = "not_started"
        self._sampler_pid: int | None = None
        self._restart_count = 0
        self._last_error: str | None = None
        self._last_error_at_utc: str | None = None
        self._journal_error: str | None = None
        self._journal_bucket_started_monotonic: float | None = None
        self._journal_bucket_samples: list[dict[str, Any]] = []
        self._worker_guard = threading.Lock()
        self._worker_active = False

    def _sample_metadata(self, *, sequence: int) -> dict[str, Any]:
        wall_time = float(self._wall_clock())
        if not math.isfinite(wall_time):
            raise ValueError("GPU telemetry wall clock must be finite")
        return {
            "sequence": sequence,
            "sampled_at_utc": datetime.fromtimestamp(wall_time, UTC).isoformat(),
            "sampled_at_unix_ms": round(wall_time * 1000),
            "source": "nvidia-smi",
            "device_index": 0,
        }

    @staticmethod
    def _parse(line: str) -> dict[str, float]:
        if not isinstance(line, str) or not line.strip() or "\n" in line or "\r" in line:
            raise ValueError("nvidia-smi output must be one non-empty text row")
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 8:
            raise ValueError("nvidia-smi returned an unexpected field count")
        try:
            values = [float(field) for field in fields]
        except ValueError as error:
            raise ValueError("nvidia-smi returned a non-numeric field") from error
        if not all(math.isfinite(value) for value in values):
            raise ValueError("nvidia-smi returned a non-finite field")
        (
            power_draw,
            power_limit,
            gpu_utilization,
            memory_utilization,
            sm_clock,
            vram_used,
            vram_free,
            temperature,
        ) = values
        if power_draw < 0 or power_limit <= 0:
            raise ValueError("nvidia-smi returned invalid power values")
        if not 0 <= gpu_utilization <= 100 or not 0 <= memory_utilization <= 100:
            raise ValueError("nvidia-smi returned invalid utilization")
        if sm_clock < 0 or vram_used < 0 or vram_free < 0:
            raise ValueError("nvidia-smi returned invalid clock or memory values")
        if not -100 <= temperature <= 200:
            raise ValueError("nvidia-smi returned an invalid temperature")
        return {
            "power_draw_w": power_draw,
            "power_limit_w": power_limit,
            "power_percent_of_limit": power_draw / power_limit * 100.0,
            "gpu_utilization_percent": gpu_utilization,
            "memory_utilization_percent": memory_utilization,
            "sm_clock_mhz": sm_clock,
            "vram_used_mib": vram_used,
            "vram_free_mib": vram_free,
            "vram_total_mib": vram_used + vram_free,
            "temperature_c": temperature,
        }

    @staticmethod
    def _aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("cannot aggregate an empty GPU telemetry bucket")
        available = [sample for sample in samples if sample.get("available") is True]
        errors: dict[str, int] = {}
        for sample in samples:
            error = sample.get("error")
            if isinstance(error, str):
                errors[error] = errors.get(error, 0) + 1
        fields: dict[str, Any] = {}
        for field in _GPU_TELEMETRY_FIELDS:
            values = [float(sample[field]) for sample in available if field in sample]
            fields[field] = _numeric_summary(values)
        return {
            "schema_version": 2,
            "kind": "twen_gpu_telemetry_aggregate",
            "source": "nvidia-smi",
            "device_index": 0,
            "raw_sample_interval_ms": round(_GPU_TELEMETRY_SAMPLE_INTERVAL_SECONDS * 1000),
            "sequence_first": samples[0]["sequence"],
            "sequence_last": samples[-1]["sequence"],
            "window_started_at_utc": samples[0]["sampled_at_utc"],
            "window_ended_at_utc": samples[-1]["sampled_at_utc"],
            "window_started_at_unix_ms": samples[0]["sampled_at_unix_ms"],
            "window_ended_at_unix_ms": samples[-1]["sampled_at_unix_ms"],
            "sample_count": len(samples),
            "available_sample_count": len(available),
            "unavailable_sample_count": len(samples) - len(available),
            "errors": errors,
            "fields": fields,
        }

    def _write_journal_aggregate(self, aggregate: Mapping[str, Any]) -> None:
        if self.journal_path is None:
            return
        try:
            _append_bounded_jsonl(
                self.journal_path,
                aggregate,
                max_bytes=self.journal_max_bytes,
            )
        except (OSError, TypeError, ValueError) as error:
            with self._lock:
                self._journal_error = type(error).__name__
        else:
            with self._lock:
                self._journal_error = None

    def _record(
        self,
        payload: Mapping[str, Any],
        *,
        monotonic_now: float | None = None,
    ) -> dict[str, Any]:
        now = float(self._monotonic_clock()) if monotonic_now is None else monotonic_now
        if not math.isfinite(now):
            raise ValueError("GPU telemetry monotonic clock must be finite")
        aggregate: dict[str, Any] | None = None
        with self._lock:
            self._sequence += 1
            sample = {**self._sample_metadata(sequence=self._sequence), **dict(payload)}
            self._latest = sample
            self._history.append(sample)
            if sample.get("available") is True:
                self._last_error = None
                self._last_error_at_utc = None
            else:
                error = sample.get("error")
                self._last_error = error if isinstance(error, str) else "unknown"
                self._last_error_at_utc = sample["sampled_at_utc"]
            bucket_started = self._journal_bucket_started_monotonic
            if self._journal_bucket_samples and (
                bucket_started is None
                or now < bucket_started
                or now - bucket_started >= self.journal_bucket_seconds
            ):
                aggregate = self._aggregate(self._journal_bucket_samples)
                self._journal_bucket_samples = []
                self._journal_bucket_started_monotonic = None
            if not self._journal_bucket_samples:
                self._journal_bucket_started_monotonic = now
            self._journal_bucket_samples.append(sample)
        if aggregate is not None:
            self._write_journal_aggregate(aggregate)
        return dict(sample)

    def _record_output_line(
        self,
        line: str,
        *,
        monotonic_now: float | None = None,
    ) -> bool:
        try:
            values = self._parse(line)
        except ValueError:
            self._record(
                {"available": False, "error": "invalid_output"},
                monotonic_now=monotonic_now,
            )
            return False
        self._record(
            {"available": True, "error": None, **values},
            monotonic_now=monotonic_now,
        )
        return True

    def _record_failure(
        self,
        error: str,
        *,
        monotonic_now: float | None = None,
        returncode: int | None = None,
    ) -> None:
        payload: dict[str, Any] = {"available": False, "error": error}
        if returncode is not None:
            payload["returncode"] = returncode
        self._record(payload, monotonic_now=monotonic_now)

    def flush_journal(self) -> None:
        """Persist the current partial aggregate without retaining raw 10 Hz rows."""

        with self._lock:
            if not self._journal_bucket_samples:
                return
            aggregate = self._aggregate(self._journal_bucket_samples)
            self._journal_bucket_samples = []
            self._journal_bucket_started_monotonic = None
        self._write_journal_aggregate(aggregate)

    def _window_statistics(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        available = [sample for sample in history if sample.get("available") is True]
        latest = self._latest
        fields: dict[str, Any] = {}
        for field in (
            "power_draw_w",
            "gpu_utilization_percent",
            "memory_utilization_percent",
        ):
            values = [float(sample[field]) for sample in available if field in sample]
            summary = _numeric_summary(values)
            fields[field] = {
                "current": (
                    float(latest[field])
                    if latest is not None and latest.get("available") is True and field in latest
                    else None
                ),
                "mean": summary["mean"] if summary is not None else None,
                "p95": summary["p95"] if summary is not None else None,
                "max": summary["max"] if summary is not None else None,
            }
        duration_seconds = 0.0
        if len(history) >= 2:
            duration_seconds = max(
                0.0,
                (float(history[-1]["sampled_at_unix_ms"]) - float(history[0]["sampled_at_unix_ms"]))
                / 1000.0,
            )
        return {
            "sample_count": len(history),
            "available_sample_count": len(available),
            "duration_seconds": duration_seconds,
            "nominal_duration_seconds": (self.history_samples * self.sample_interval_seconds),
            "fields": fields,
        }

    def snapshot(self) -> dict[str, Any]:
        """Return the latest sample, raw in-memory window, and window statistics."""

        with self._lock:
            history = [dict(sample) for sample in self._history]
            latest = (
                dict(self._latest)
                if self._latest is not None
                else {
                    "available": False,
                    "error": "starting",
                    "source": "nvidia-smi",
                    "device_index": 0,
                }
            )
            sampler = {
                "state": self._sampler_state,
                "pid": self._sampler_pid,
                "restart_count": self._restart_count,
                "last_error": self._last_error,
                "last_error_at_utc": self._last_error_at_utc,
                "journal_error": self._journal_error,
            }
            return {
                "latest": latest,
                "history": history,
                "window_statistics": self._window_statistics(history),
                "sample_interval_seconds": self.sample_interval_seconds,
                "history_limit": self.history_samples,
                "output_timeout_seconds": self.output_timeout_seconds,
                "journal_bucket_seconds": self.journal_bucket_seconds,
                "journal_max_bytes": self.journal_max_bytes,
                "journal_segments": 2,
                "sampler": sampler,
            }

    @staticmethod
    def _reap_process(process: Any) -> None:
        stdout = getattr(process, "stdout", None)
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=0.5)
            else:
                process.wait(timeout=0.5)
        except (OSError, subprocess.TimeoutExpired):
            with suppress(OSError):
                process.kill()
            with suppress(OSError, subprocess.TimeoutExpired):
                process.wait(timeout=0.5)
        finally:
            if stdout is not None:
                with suppress(OSError):
                    stdout.close()

    def _set_sampler_state(self, state: str, *, pid: int | None = None) -> None:
        with self._lock:
            self._sampler_state = state
            self._sampler_pid = pid

    def run_until_stopped(self, stop_event: threading.Event) -> None:
        """Read one looping child, degrade on failure, and restart with backoff."""

        with self._worker_guard:
            if self._worker_active:
                raise RuntimeError("GPU telemetry worker is already running")
            self._worker_active = True
        consecutive_failures = 0
        try:
            while not stop_event.is_set():
                self._set_sampler_state("starting")
                try:
                    process = self._process_factory(
                        _NVIDIA_SMI_COMMAND,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        text=False,
                        bufsize=0,
                        shell=False,
                    )
                except FileNotFoundError:
                    self._record_failure("not_found")
                    process = None
                except PermissionError:
                    self._record_failure("permission_denied")
                    process = None
                except OSError:
                    self._record_failure("operating_system_error")
                    process = None

                if process is not None:
                    pid = getattr(process, "pid", None)
                    self._set_sampler_state("running", pid=pid if isinstance(pid, int) else None)
                    reason: str | None = None
                    reason_returncode: int | None = None
                    invalid_already_recorded = False
                    output = getattr(process, "stdout", None)
                    last_complete_line = float(self._monotonic_clock())
                    partial = b""
                    try:
                        if output is None:
                            reason = "invalid_stream"
                        else:
                            descriptor = output.fileno()
                            os.set_blocking(descriptor, False)
                            while not stop_event.is_set() and reason is None:
                                now = float(self._monotonic_clock())
                                if not math.isfinite(now):
                                    raise ValueError("GPU telemetry monotonic clock must be finite")
                                remaining = max(
                                    0.0,
                                    self.output_timeout_seconds - (now - last_complete_line),
                                )
                                readable, _, _ = select.select(
                                    (descriptor,),
                                    (),
                                    (),
                                    min(_GPU_TELEMETRY_READ_POLL_SECONDS, remaining),
                                )
                                now = float(self._monotonic_clock())
                                if readable:
                                    try:
                                        chunk = os.read(descriptor, 64 * 1024)
                                    except BlockingIOError:
                                        continue
                                    except OSError:
                                        reason = "stream_read_failed"
                                        break
                                    if not chunk:
                                        returncode = process.poll()
                                        if returncode is None:
                                            reason = "stream_disconnected"
                                        else:
                                            reason = "command_exited"
                                            reason_returncode = int(returncode)
                                        break
                                    partial += chunk
                                    while b"\n" in partial:
                                        raw_line, partial = partial.split(b"\n", 1)
                                        raw_line = raw_line.rstrip(b"\r")
                                        if not raw_line:
                                            continue
                                        try:
                                            line = raw_line.decode("utf-8")
                                        except UnicodeDecodeError:
                                            self._record_failure(
                                                "invalid_output", monotonic_now=now
                                            )
                                            invalid_already_recorded = True
                                            reason = "invalid_output"
                                            break
                                        if not self._record_output_line(line, monotonic_now=now):
                                            invalid_already_recorded = True
                                            reason = "invalid_output"
                                            break
                                        last_complete_line = now
                                        consecutive_failures = 0
                                    if (
                                        reason is None
                                        and len(partial) > _GPU_TELEMETRY_MAX_PARTIAL_OUTPUT_BYTES
                                    ):
                                        reason = "invalid_output"
                                returncode = process.poll()
                                if returncode is not None and reason is None:
                                    reason = "command_exited"
                                    reason_returncode = int(returncode)
                                elif (
                                    reason is None
                                    and now - last_complete_line >= self.output_timeout_seconds
                                ):
                                    reason = "timeout"
                    except (OSError, ValueError):
                        reason = "operating_system_error"
                    finally:
                        self._reap_process(process)
                        self._set_sampler_state("backoff")
                    if (
                        reason is not None
                        and not invalid_already_recorded
                        and not stop_event.is_set()
                    ):
                        self._record_failure(reason, returncode=reason_returncode)

                if stop_event.is_set():
                    break
                consecutive_failures += 1
                with self._lock:
                    self._restart_count += 1
                delay = min(
                    self.restart_backoff_seconds * (2 ** min(consecutive_failures - 1, 8)),
                    self.max_restart_backoff_seconds,
                )
                self._set_sampler_state("backoff")
                stop_event.wait(delay)
        finally:
            self.flush_journal()
            self._set_sampler_state("stopped")
            with self._worker_guard:
                self._worker_active = False


@dataclass(frozen=True, slots=True)
class DashboardAuth:
    """HTTP Basic credentials required for a non-loopback dashboard bind."""

    username: str
    password: str
    source_path: Path

    @property
    def authorization_header(self) -> str:
        payload = base64.b64encode(f"{self.username}:{self.password}".encode()).decode("ascii")
        return f"Basic {payload}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp")
    payload = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, allow_nan=False)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def load_dashboard_auth(path: str | Path) -> DashboardAuth:
    """Load a private HTTP Basic credential file and reject unsafe permissions."""

    source = Path(path).resolve()
    try:
        stat_result = source.stat()
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DashboardError(f"cannot read dashboard auth file {source}: {error}") from error
    if stat_result.st_mode & 0o077:
        raise DashboardError(
            f"dashboard auth file must not be group/world accessible (chmod 600): {source}"
        )
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise DashboardError("dashboard auth file schema_version must equal 1")
    username = raw.get("username")
    password = raw.get("password")
    if not isinstance(username, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", username):
        raise DashboardError("dashboard auth username is invalid")
    if not isinstance(password, str) or len(password) < 24 or len(password) > 512:
        raise DashboardError("dashboard auth password must contain 24..512 characters")
    return DashboardAuth(username=username, password=password, source_path=source)


def ensure_dashboard_auth_file(
    path: str | Path,
    *,
    username: str = "twen",
) -> dict[str, Any]:
    """Create a mode-0600 credential once, without printing its secret."""

    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", username):
        raise DashboardError("dashboard auth username is invalid")
    target = Path(path).resolve()
    if target.exists():
        auth = load_dashboard_auth(target)
        return {
            "created": False,
            "path": str(target),
            "username": auth.username,
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "username": username,
        "password": secrets.token_urlsafe(36),
    }
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode()
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        auth = load_dashboard_auth(target)
        return {
            "created": False,
            "path": str(target),
            "username": auth.username,
        }
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    auth = load_dashboard_auth(target)
    return {"created": True, "path": str(target), "username": auth.username}


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, allow_nan=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _bounded_file_tail(path: Path, *, max_bytes: int) -> bytes:
    size = path.stat().st_size
    with path.open("rb") as handle:
        offset = max(0, size - max_bytes)
        handle.seek(offset)
        payload = handle.read(max_bytes)
    if offset:
        newline = payload.find(b"\n")
        payload = payload[newline + 1 :] if newline >= 0 else b""
    return payload


def _atomic_private_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_bounded_jsonl(
    path: Path,
    value: Mapping[str, Any],
    *,
    max_bytes: int,
) -> None:
    """Append one aggregate while bounding the active file and one backup."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(payload) > max_bytes:
        raise ValueError("GPU telemetry aggregate exceeds the journal byte limit")
    current_size = path.stat().st_size if path.exists() else 0
    if current_size and current_size + len(payload) > max_bytes:
        backup = path.with_name(path.name + ".1")
        backup.unlink(missing_ok=True)
        if current_size <= max_bytes:
            os.replace(path, backup)
        else:
            _atomic_private_bytes(
                backup,
                _bounded_file_tail(path, max_bytes=max_bytes),
            )
            path.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    with os.fdopen(descriptor, "ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _inside(root: Path, candidate: Path, *, label: str) -> Path:
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise DashboardError(f"{label} must stay inside project_root: {resolved}")
    return resolved


def _existing_file_inside(root: Path, candidate: Path) -> Path | None:
    """Resolve one existing regular file without following it outside ``root``."""

    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        return None
    return resolved


def _existing_directory_inside(root: Path, candidate: Path) -> Path | None:
    """Resolve one existing directory without following it outside ``root``."""

    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_relative_to(resolved_root) or not resolved.is_dir():
        return None
    return resolved


def _read_only_probe_flags() -> int | None:
    """Return non-blocking, no-follow flags or fail closed on unsupported hosts."""

    no_follow = getattr(os, "O_NOFOLLOW", None)
    non_blocking = getattr(os, "O_NONBLOCK", None)
    if not isinstance(no_follow, int) or not isinstance(non_blocking, int):
        return None
    return os.O_RDONLY | no_follow | non_blocking | getattr(os, "O_CLOEXEC", 0)


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_regular_file_no_follow(
    path: Path,
    *,
    max_bytes: int = 16 * 1024 * 1024,
) -> bytes | None:
    """Read one stable regular file without following or blocking on special files."""

    flags = _read_only_probe_flags()
    if (
        flags is None
        or isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes < 0
    ):
        return None
    try:
        path_before = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(path_before.st_mode)
            or path_before.st_size < 0
            or path_before.st_size > max_bytes
        ):
            return None
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        opened_before = os.fstat(descriptor)
        if _file_identity(opened_before) != _file_identity(path_before):
            return None
        payload = bytearray()
        remaining = opened_before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                return None
            payload.extend(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            return None
        opened_after = os.fstat(descriptor)
        path_after = os.stat(path, follow_symlinks=False)
        expected_identity = _file_identity(opened_before)
        if (
            _file_identity(opened_after) != expected_identity
            or _file_identity(path_after) != expected_identity
        ):
            return None
        return bytes(payload)
    except OSError:
        return None
    finally:
        with suppress(OSError):
            os.close(descriptor)


def _sha256_regular_file_no_follow(path: Path) -> str | None:
    """Hash one stable regular file without following special files."""

    payload = _read_regular_file_no_follow(path)
    return hashlib.sha256(payload).hexdigest() if payload is not None else None


def _read_json_regular_no_follow(path: Path) -> dict[str, Any] | None:
    payload = _read_regular_file_no_follow(path, max_bytes=4 * 1024 * 1024)
    if payload is None:
        return None
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _exclusive_advisory_lock_is_held(path: Path) -> bool:
    """Probe an exclusive ``flock`` through a read-only descriptor.

    Evaluation workers take an exclusive lock.  A non-blocking shared lock is
    therefore sufficient to distinguish a live worker from an abandoned lock
    file without creating, truncating, or otherwise modifying that file.
    """

    flags = _read_only_probe_flags()
    if flags is None:
        return False
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return False
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        except OSError:
            return False
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            return False
        return False
    finally:
        with suppress(OSError):
            os.close(descriptor)


def _held_run_lock_owner(path: Path) -> dict[str, Any] | None:
    """Return the stable owner of an actively held, regular trainer run lock."""

    flags = _read_only_probe_flags()
    if flags is None:
        return None
    try:
        path_before = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(path_before.st_mode)
            or path_before.st_size < 2
            or path_before.st_size > 4096
        ):
            return None
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        opened_before = os.fstat(descriptor)
        if _file_identity(opened_before) != _file_identity(path_before):
            return None
        payload = os.read(descriptor, opened_before.st_size + 1)
        if len(payload) != opened_before.st_size:
            return None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_is_held = True
        except OSError:
            return None
        else:
            lock_is_held = False
            with suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        if not lock_is_held:
            return None
        opened_after = os.fstat(descriptor)
        try:
            path_after = os.stat(path, follow_symlinks=False)
        except OSError:
            return None
        if (
            _file_identity(opened_after) != _file_identity(opened_before)
            or _file_identity(path_after) != _file_identity(opened_before)
        ):
            return None
        try:
            owner = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return dict(owner) if isinstance(owner, Mapping) else None
    finally:
        with suppress(OSError):
            os.close(descriptor)


@contextmanager
def _exclusive_existing_directory_lock(
    path: Path,
    *,
    timeout_seconds: float,
) -> Any:
    """Lock an already-existing private directory without creating lock artifacts."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        path_before = os.stat(path, follow_symlinks=False)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
    except OSError as error:
        raise DashboardError(f"cannot open existing dashboard action-lock directory: {path}") from error
    try:
        if (
            not stat.S_ISDIR(path_before.st_mode)
            or _file_identity(path_before) != _file_identity(opened)
        ):
            raise DashboardError("dashboard action-lock directory identity changed")
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as error:
                if time.monotonic() >= deadline:
                    raise DashboardError(
                        "another dashboard process is handling a training control action"
                    ) from error
                time.sleep(_DASHBOARD_ACTION_LOCK_POLL_SECONDS)
        yield
    finally:
        with suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        with suppress(OSError):
            os.close(descriptor)


@dataclass(frozen=True, slots=True)
class LaunchProfile:
    """One immutable, server-side training command allowlist entry."""

    profile_id: str
    label: str
    config_path: Path
    config_sha256: str
    run_dir: Path
    run_id: str
    stage: str
    resume: str
    fork_from: Path | None
    launch_enabled: bool
    launch_kind: str = "direct_train"
    governed_controller_path: Path | None = None
    governed_controller_sha256: str | None = None
    governed_readiness_path: Path | None = None
    governed_readiness_sha256: str | None = None
    governed_state_path: Path | None = None
    governed_plan_id: str | None = None
    governed_source_tree_sha256: str | None = None
    governed_dependency_lock_name: str | None = None
    governed_dependency_lock_sha256: str | None = None

    @property
    def start_confirmation(self) -> str:
        if self.launch_kind == "governed_v4" and self.governed_plan_id is not None:
            return f"RUN {self.governed_plan_id}"
        return f"START {self.profile_id}"

    @property
    def stop_confirmation(self) -> str:
        return f"STOP {self.run_id}"

    @property
    def save_confirmation(self) -> str:
        return f"SAVE {self.run_id}"


@dataclass(frozen=True, slots=True)
class DashboardSettings:
    dashboard_config_path: Path
    dashboard_config_sha256: str
    project_root: Path
    state_dir: Path
    profiles: tuple[LaunchProfile, ...]

    def profile(self, profile_id: str) -> LaunchProfile:
        for profile in self.profiles:
            if profile.profile_id == profile_id:
                return profile
        raise DashboardError(f"unknown launch profile: {profile_id!r}")


def _dashboard_train_identity(
    config_path: Path,
    *,
    launch_kind: str,
) -> tuple[str, str, str]:
    """Return the fixed run identity without unblocking governed PENDING data.

    A closure-stage governed profile intentionally points at the source-bound
    blocked YAML, whose dynamic data identities remain ``PENDING_*``.  The
    ordinary config loader correctly rejects those sentinels.  For that one
    profile kind, extract only the three dashboard routing fields; the
    subsequent ``build_governed_plan`` call authenticates the entire YAML
    semantics and keeps the launch blocked.
    """

    try:
        config = load_train_config(config_path)
    except ConfigError as error:
        if launch_kind != "governed_v4":
            raise
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as parse_error:
            raise DashboardError(
                f"cannot inspect blocked governed config {config_path}: {parse_error}"
            ) from parse_error
        if not isinstance(raw, Mapping):
            raise DashboardError("blocked governed config must be a YAML mapping") from error
        data = raw.get("data")
        checkpoint = raw.get("checkpoint")
        if not isinstance(data, Mapping) or not isinstance(checkpoint, Mapping):
            raise DashboardError(
                "blocked governed config lacks data/checkpoint mappings"
            ) from error
        pending = [
            field
            for field in _GOVERNED_DYNAMIC_DATA_IDENTITIES
            if isinstance(data.get(field), str) and "PENDING" in str(data[field])
        ]
        if not pending:
            raise DashboardError(
                "governed config failed validation without a PENDING data identity"
            ) from error
        run_id = raw.get("run_id")
        stage = raw.get("stage")
        output_dir = checkpoint.get("output_dir")
        if (
            not isinstance(run_id, str)
            or not run_id
            or stage not in {"dense-oracle", "sparse"}
            or not isinstance(output_dir, str)
            or not output_dir
        ):
            raise DashboardError(
                "blocked governed config has an invalid run routing identity"
            ) from error
        return run_id, stage, output_dir
    return config.run_id, config.stage, config.checkpoint.output_dir


def load_dashboard_settings(path: str | Path) -> DashboardSettings:
    """Load and fully resolve the dashboard's fixed launch allowlist."""

    source = Path(path).resolve()
    try:
        source_bytes = source.read_bytes()
        raw = json.loads(source_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DashboardError(f"cannot read dashboard config {source}: {error}") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise DashboardError("dashboard config schema_version must equal 1")
    root_value = raw.get("project_root", ".")
    if not isinstance(root_value, str) or not root_value.strip():
        raise DashboardError("project_root must be a non-empty string")
    project_root = (source.parent / root_value).resolve()
    if not project_root.is_dir():
        raise DashboardError(f"project_root is not a directory: {project_root}")
    state_value = raw.get("state_dir", ".twen/dashboard")
    if not isinstance(state_value, str) or not state_value.strip():
        raise DashboardError("state_dir must be a non-empty string")
    state_dir = _inside(project_root, project_root / state_value, label="state_dir")

    raw_profiles = raw.get("profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise DashboardError("dashboard config requires at least one profile")
    profiles: list[LaunchProfile] = []
    ids: set[str] = set()
    run_dirs: set[Path] = set()
    for index, item in enumerate(raw_profiles):
        if not isinstance(item, dict):
            raise DashboardError(f"profiles[{index}] must be an object")
        profile_id = item.get("id")
        if not isinstance(profile_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", profile_id
        ):
            raise DashboardError(f"profiles[{index}].id is not filesystem-safe")
        if profile_id in ids:
            raise DashboardError(f"duplicate profile id: {profile_id}")
        ids.add(profile_id)
        label = item.get("label", profile_id)
        if not isinstance(label, str) or not label.strip() or len(label) > 120:
            raise DashboardError(f"profiles[{index}].label must be 1..120 characters")
        config_value = item.get("config")
        if not isinstance(config_value, str) or not config_value.strip():
            raise DashboardError(f"profiles[{index}].config must be a non-empty string")
        config_path = _inside(
            project_root,
            project_root / config_value,
            label=f"profiles[{index}].config",
        )
        if not config_path.is_file():
            raise DashboardError(f"profile config does not exist: {config_path}")
        launch_enabled = item.get("launch_enabled", False)
        if not isinstance(launch_enabled, bool):
            raise DashboardError(f"profiles[{index}].launch_enabled must be boolean")
        launch_kind = item.get("launch_kind", "direct_train")
        if launch_kind not in {"direct_train", "governed_v4"}:
            raise DashboardError(
                f"profiles[{index}].launch_kind must be direct_train or governed_v4"
            )
        declared_config_sha256 = item.get("config_sha256")
        if declared_config_sha256 is None:
            if launch_enabled or launch_kind == "governed_v4":
                raise DashboardError(
                    f"profiles[{index}].config_sha256 is required for launchable or governed profiles"
                )
        elif not isinstance(declared_config_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", declared_config_sha256
        ):
            raise DashboardError(f"profiles[{index}].config_sha256 must be lowercase SHA256")
        config_sha256 = sha256_file(config_path)
        if declared_config_sha256 is not None and not secrets.compare_digest(
            declared_config_sha256,
            config_sha256,
        ):
            raise DashboardError(
                f"profiles[{index}].config SHA256 does not match the pinned allowlist identity"
            )
        run_id, stage, output_dir = _dashboard_train_identity(
            config_path,
            launch_kind=launch_kind,
        )
        run_dir = _inside(
            project_root,
            project_root / output_dir,
            label=f"profiles[{index}] training output_dir",
        )
        if run_dir in run_dirs:
            raise DashboardError(f"profiles must not share a run directory: {run_dir}")
        run_dirs.add(run_dir)
        resume = item.get("resume", "auto")
        if not isinstance(resume, str) or not resume.strip():
            raise DashboardError(f"profiles[{index}].resume must be a non-empty string")
        if resume not in {"auto", "none"}:
            resume_path = _inside(
                project_root,
                project_root / resume,
                label=f"profiles[{index}].resume",
            )
            resume = str(resume_path)
        fork_value = item.get("fork_from")
        if fork_value is not None and (not isinstance(fork_value, str) or not fork_value.strip()):
            raise DashboardError(f"profiles[{index}].fork_from must be null or a path")
        fork_from = (
            _inside(
                project_root,
                project_root / fork_value,
                label=f"profiles[{index}].fork_from",
            )
            if isinstance(fork_value, str)
            else None
        )
        if fork_from is not None and resume != "none":
            raise DashboardError("fork_from profiles must use resume='none'")
        governed_controller_path: Path | None = None
        governed_controller_sha256: str | None = None
        governed_readiness_path: Path | None = None
        governed_readiness_sha256: str | None = None
        governed_state_path: Path | None = None
        governed_plan_id: str | None = None
        governed_source_tree_sha256: str | None = None
        governed_dependency_lock_name: str | None = None
        governed_dependency_lock_sha256: str | None = None
        governed_fields = {
            "controller": item.get("governed_controller"),
            "controller_sha256": item.get("governed_controller_sha256"),
            "readiness": item.get("governed_readiness"),
            "readiness_sha256": item.get("governed_readiness_sha256"),
            "state": item.get("governed_state"),
            "plan_id": item.get("governed_plan_id"),
        }
        if launch_kind == "direct_train":
            if any(value is not None for value in governed_fields.values()):
                raise DashboardError(
                    f"profiles[{index}] direct_train must not declare governed launch fields"
                )
        else:
            if resume != "none" or fork_from is not None:
                raise DashboardError(
                    f"profiles[{index}] governed_v4 must use resume='none' and fork_from=null"
                )
            for field, value in governed_fields.items():
                if not isinstance(value, str) or not value.strip():
                    raise DashboardError(
                        f"profiles[{index}].governed_{field} must be a non-empty string"
                    )
            governed_controller_path = _inside(
                project_root,
                project_root / str(governed_fields["controller"]),
                label=f"profiles[{index}].governed_controller",
            )
            governed_readiness_path = _inside(
                project_root,
                project_root / str(governed_fields["readiness"]),
                label=f"profiles[{index}].governed_readiness",
            )
            governed_state_path = _inside(
                project_root,
                project_root / str(governed_fields["state"]),
                label=f"profiles[{index}].governed_state",
            )
            expected_governed_state_path = (
                run_dir.parent / f".{run_dir.name}.governed" / "controller-state.json"
            )
            if governed_state_path != expected_governed_state_path:
                raise DashboardError(
                    f"profiles[{index}].governed_state must equal "
                    f"{expected_governed_state_path.relative_to(project_root)}"
                )
            for governed_path, label in (
                (governed_controller_path, "controller"),
                (governed_readiness_path, "readiness"),
            ):
                if not governed_path.is_file():
                    raise DashboardError(
                        f"profiles[{index}] governed {label} does not exist: {governed_path}"
                    )
            governed_controller_sha256 = str(governed_fields["controller_sha256"])
            governed_readiness_sha256 = str(governed_fields["readiness_sha256"])
            governed_plan_id = str(governed_fields["plan_id"])
            for field, value in (
                ("governed_controller_sha256", governed_controller_sha256),
                ("governed_readiness_sha256", governed_readiness_sha256),
                ("governed_plan_id", governed_plan_id),
            ):
                if not re.fullmatch(r"[0-9a-f]{64}", value):
                    raise DashboardError(
                        f"profiles[{index}].{field} must be lowercase SHA256"
                    )
            if not secrets.compare_digest(
                sha256_file(governed_controller_path),
                governed_controller_sha256,
            ):
                raise DashboardError(
                    f"profiles[{index}] governed controller SHA256 does not match"
                )
            if not secrets.compare_digest(
                sha256_file(governed_readiness_path),
                governed_readiness_sha256,
            ):
                raise DashboardError(
                    f"profiles[{index}] governed readiness SHA256 does not match"
                )
            try:
                governed_plan = build_governed_plan(governed_readiness_path)
            except GovernedControllerError as error:
                raise DashboardError(
                    f"profiles[{index}] governed readiness is invalid: {error}"
                ) from error
            if not secrets.compare_digest(
                str(governed_plan.get("plan_id", "")),
                governed_plan_id,
            ):
                raise DashboardError(
                    f"profiles[{index}] governed plan_id does not match the pinned plan"
                )
            governed_config = governed_plan.get("config")
            governed_run = governed_plan.get("run")
            if (
                not isinstance(governed_config, Mapping)
                or Path(str(governed_config.get("path"))).resolve() != config_path
                or governed_config.get("sha256") != config_sha256
                or not isinstance(governed_run, Mapping)
                or Path(str(governed_run.get("output_dir"))).resolve() != run_dir
                or governed_run.get("run_id") != run_id
                or governed_run.get("stage") != stage
            ):
                raise DashboardError(
                    f"profiles[{index}] governed plan does not bind the profile config/run"
                )
            controller_sources = governed_plan.get("controller_sources")
            if not isinstance(controller_sources, list):
                raise DashboardError(
                    f"profiles[{index}] governed plan has no controller source inventory"
                )
            controller_matches = [
                source
                for source in controller_sources
                if isinstance(source, Mapping)
                and Path(str(source.get("path"))).resolve() == governed_controller_path
                and source.get("sha256") == governed_controller_sha256
            ]
            if len(controller_matches) != 1:
                raise DashboardError(
                    f"profiles[{index}] governed plan does not bind the pinned controller"
                )
            governed_source_tree = governed_plan.get("source_tree")
            governed_dependency_lock = governed_plan.get("dependency_lock")
            if not isinstance(governed_source_tree, Mapping) or not isinstance(
                governed_dependency_lock,
                Mapping,
            ):
                raise DashboardError(
                    f"profiles[{index}] governed plan has no source/dependency identity"
                )
            governed_source_root = Path(
                str(governed_source_tree.get("path"))
            ).resolve()
            governed_dependency_path = Path(
                str(governed_dependency_lock.get("path"))
            ).resolve()
            governed_source_tree_sha256 = str(
                governed_source_tree.get("sha256")
            )
            governed_dependency_lock_sha256 = str(
                governed_dependency_lock.get("sha256")
            )
            governed_dependency_lock_name = governed_dependency_path.name
            if (
                governed_source_root != project_root / "src/twen"
                or governed_dependency_path.parent != project_root
                or governed_dependency_lock_name not in {"uv.lock", "pyproject.toml"}
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    governed_source_tree_sha256,
                )
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    governed_dependency_lock_sha256,
                )
            ):
                raise DashboardError(
                    f"profiles[{index}] governed archive contract is invalid"
                )
        profiles.append(
            LaunchProfile(
                profile_id=profile_id,
                label=label.strip(),
                config_path=config_path,
                config_sha256=config_sha256,
                run_dir=run_dir,
                run_id=run_id,
                stage=stage,
                resume=resume,
                fork_from=fork_from,
                launch_enabled=launch_enabled,
                launch_kind=launch_kind,
                governed_controller_path=governed_controller_path,
                governed_controller_sha256=governed_controller_sha256,
                governed_readiness_path=governed_readiness_path,
                governed_readiness_sha256=governed_readiness_sha256,
                governed_state_path=governed_state_path,
                governed_plan_id=governed_plan_id,
                governed_source_tree_sha256=governed_source_tree_sha256,
                governed_dependency_lock_name=governed_dependency_lock_name,
                governed_dependency_lock_sha256=governed_dependency_lock_sha256,
            )
        )
    return DashboardSettings(
        dashboard_config_path=source,
        dashboard_config_sha256=hashlib.sha256(source_bytes).hexdigest(),
        project_root=project_root,
        state_dir=state_dir,
        profiles=tuple(profiles),
    )


class JsonlTailCache:
    """Incrementally parse append-only JSONL without rescanning on every poll."""

    def __init__(self, *, max_records: int = 20_000) -> None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        self.max_records = max_records
        self._states: dict[Path, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def read(self, path: Path, *, limit: int) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), self.max_records))
        with self._lock:
            state = self._states.setdefault(
                path,
                {
                    "identity": None,
                    "offset": 0,
                    "partial": b"",
                    "records": deque(maxlen=self.max_records),
                },
            )
            try:
                stat = path.stat()
            except FileNotFoundError:
                return []
            identity = (stat.st_dev, stat.st_ino)
            if state["identity"] != identity or stat.st_size < state["offset"]:
                state["identity"] = identity
                state["offset"] = 0
                state["partial"] = b""
                state["records"].clear()
            if stat.st_size > state["offset"]:
                with path.open("rb") as handle:
                    handle.seek(state["offset"])
                    chunk = handle.read()
                state["offset"] += len(chunk)
                payload = state["partial"] + chunk
                complete, separator, partial = payload.rpartition(b"\n")
                if separator:
                    lines = complete.splitlines()
                    state["partial"] = partial
                else:
                    lines = []
                    state["partial"] = payload
                for line in lines:
                    try:
                        record = json.loads(line)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if isinstance(record, dict):
                        state["records"].append(record)
            records = state["records"]
            return list(records)[-limit:]


_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])")


def read_console_tail(path: Path, *, lines: int = 160, max_bytes: int = 256 * 1024) -> str:
    """Read a bounded progress/log tail, treating tqdm carriage returns as lines."""

    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            payload = handle.read(max_bytes)
    except FileNotFoundError:
        return ""
    text = payload.decode("utf-8", errors="replace")
    entries = re.split(r"[\r\n]+", _ANSI_ESCAPE.sub("", text))
    cleaned = [entry[-4000:] for entry in entries if entry.strip()]
    return "\n".join(cleaned[-max(1, min(lines, 1000)) :])


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _authenticated_evaluation_device_type(
    output: Path,
    plan_path: Path | None,
) -> str | None:
    """Return a fingerprint-authenticated CPU/CUDA declaration from PLAN."""

    raw_plan_path = output / "PLAN.json"
    if plan_path is None or plan_path != raw_plan_path or raw_plan_path.is_symlink():
        return None
    flags = _read_only_probe_flags()
    if flags is None:
        return None
    try:
        path_before = os.stat(raw_plan_path, follow_symlinks=False)
        if not stat.S_ISREG(path_before.st_mode) or not 0 < path_before.st_size <= 4 * 1024 * 1024:
            return None
        descriptor = os.open(raw_plan_path, flags)
    except OSError:
        return None
    try:
        opened_before = os.fstat(descriptor)
        if _file_identity(opened_before) != _file_identity(path_before):
            return None
        payload = bytearray()
        while len(payload) < opened_before.st_size:
            chunk = os.read(
                descriptor,
                min(64 * 1024, opened_before.st_size - len(payload)),
            )
            if not chunk:
                return None
            payload.extend(chunk)
        if os.read(descriptor, 1):
            return None
        opened_after = os.fstat(descriptor)
        path_after = os.stat(raw_plan_path, follow_symlinks=False)
        expected_identity = _file_identity(opened_before)
        if (
            _file_identity(opened_after) != expected_identity
            or _file_identity(path_after) != expected_identity
        ):
            return None
    except OSError:
        return None
    finally:
        with suppress(OSError):
            os.close(descriptor)
    try:
        plan = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(plan, dict)
        or plan.get("schema_version") != 1
        or plan.get("kind") != "twen_nll_evaluation_plan"
    ):
        return None
    fingerprint = plan.get("plan_fingerprint")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        return None
    unsigned = dict(plan)
    unsigned.pop("plan_fingerprint", None)
    try:
        encoded = json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    if not secrets.compare_digest(fingerprint, hashlib.sha256(encoded).hexdigest()):
        return None
    device_type = plan.get("device_type")
    return device_type if device_type in {"cpu", "cuda"} else None


_RUN_ACTIVITY_FILES = (
    "metrics.jsonl",
    "telemetry.jsonl",
    "events.jsonl",
    "rank0-session.json",
    "console.log",
)


def _run_last_update(
    project_root: Path,
    run_dir: Path,
) -> tuple[float | None, str | None]:
    """Return a stable recency marker without trusting timestamps inside logs."""

    confined_run_dir = _existing_directory_inside(project_root, run_dir)
    if confined_run_dir is None:
        return None, None
    modified: list[float] = []
    for name in _RUN_ACTIVITY_FILES:
        path = _existing_file_inside(confined_run_dir, confined_run_dir / name)
        if path is None:
            continue
        try:
            modified.append(path.stat().st_mtime)
        except OSError:
            continue
    if not modified:
        return None, None
    latest = max(modified)
    return latest, datetime.fromtimestamp(latest, UTC).isoformat()


def _pid_alive(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    closing = stat.rfind(")")
    if closing >= 0 and stat[closing + 2 : closing + 3] == "Z":
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


@dataclass(frozen=True, slots=True)
class _LinuxProcessIdentity:
    pid: int
    start_time_ticks: int
    command: tuple[str, ...]
    cwd: Path
    executable: Path
    process_group_id: int


def _linux_process_start_time_ticks(pid: int) -> int:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError as error:
        raise DashboardError(f"cannot read process start identity for PID {pid}") from error
    close = raw.rfind(")")
    if close < 0:
        raise DashboardError(f"process stat for PID {pid} has no command terminator")
    fields_from_state = raw[close + 2 :].split()
    try:
        value = int(fields_from_state[19])
    except (IndexError, ValueError) as error:
        raise DashboardError(f"process stat for PID {pid} has no starttime") from error
    if value < 0:
        raise DashboardError(f"process stat for PID {pid} has an invalid starttime")
    return value


def _read_linux_process_identity(pid: int) -> _LinuxProcessIdentity:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1 or not _pid_alive(pid):
        raise DashboardError(f"PID {pid!r} is not a live non-init process")
    try:
        command = tuple(
            item.decode("utf-8", errors="surrogateescape")
            for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if item
        )
        cwd = Path(os.readlink(f"/proc/{pid}/cwd")).resolve()
        executable = Path(os.readlink(f"/proc/{pid}/exe")).resolve()
        process_group_id = os.getpgid(pid)
    except OSError as error:
        raise DashboardError(f"cannot read complete process identity for PID {pid}") from error
    if not command:
        raise DashboardError(f"PID {pid} has an empty command line")
    start_time_ticks = _linux_process_start_time_ticks(pid)
    if not _pid_alive(pid) or _linux_process_start_time_ticks(pid) != start_time_ticks:
        raise DashboardError(f"PID {pid} changed while reading its process identity")
    return _LinuxProcessIdentity(
        pid=pid,
        start_time_ticks=start_time_ticks,
        command=command,
        cwd=cwd,
        executable=executable,
        process_group_id=process_group_id,
    )


def _read_process_environment(pid: int) -> dict[str, str]:
    try:
        payload = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError as error:
        raise DashboardError(f"cannot read governed controller environment for PID {pid}") from error
    result: dict[str, str] = {}
    for raw in payload.split(b"\0"):
        if not raw or b"=" not in raw:
            continue
        key, value = raw.split(b"=", 1)
        decoded_key = key.decode("utf-8", errors="surrogateescape")
        if decoded_key in result:
            raise DashboardError("governed controller environment contains duplicate keys")
        result[decoded_key] = value.decode("utf-8", errors="surrogateescape")
    return result


def _sealed_process_fd_bytes(pid: int, descriptor: int) -> bytes:
    if isinstance(descriptor, bool) or not isinstance(descriptor, int) or descriptor < 0:
        raise DashboardError("governed snapshot descriptor identity is invalid")
    path = Path(f"/proc/{pid}/fd/{descriptor}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    try:
        opened = os.open(path, flags)
    except OSError as error:
        raise DashboardError("cannot open governed process snapshot descriptor") from error
    try:
        metadata = os.fstat(opened)
        seals = int(fcntl.fcntl(opened, _F_GET_SEALS))
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size < 1
            or metadata.st_size > _GOVERNED_SOURCE_ARCHIVE_MAX_BYTES
            or seals & _REQUIRED_MEMFD_SEALS != _REQUIRED_MEMFD_SEALS
        ):
            raise DashboardError("governed process snapshot is not a sealed regular memfd")
        os.lseek(opened, 0, os.SEEK_SET)
        payload = bytearray()
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(opened, min(1024 * 1024, remaining))
            if not chunk:
                raise DashboardError("governed process snapshot ended early")
            payload.extend(chunk)
            remaining -= len(chunk)
        if os.read(opened, 1):
            raise DashboardError("governed process snapshot grew while reading")
        if os.fstat(opened).st_size != metadata.st_size:
            raise DashboardError("governed process snapshot changed while reading")
        return bytes(payload)
    finally:
        with suppress(OSError):
            os.close(opened)


def _sealed_process_fd_sha256(pid: int, descriptor: int) -> str:
    return hashlib.sha256(_sealed_process_fd_bytes(pid, descriptor)).hexdigest()


def _governed_source_archive_hashes(
    payload: bytes,
    *,
    dependency_lock_name: str,
) -> tuple[str, str]:
    """Recompute the executable source and dependency identities inside a zip."""

    if dependency_lock_name not in {"uv.lock", "pyproject.toml"}:
        raise DashboardError("governed dependency-lock archive name is invalid")
    dependency_entry = f".twen-governed-dependency/{dependency_lock_name}"
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            entries = archive.infolist()
            if (
                archive.comment
                or not entries
                or len(entries) > _GOVERNED_SOURCE_ARCHIVE_MAX_FILES
            ):
                raise DashboardError("governed source archive inventory is invalid")
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)):
                raise DashboardError("governed source archive contains duplicate paths")
            total_uncompressed = sum(entry.file_size for entry in entries)
            if (
                total_uncompressed < 1
                or total_uncompressed
                > _GOVERNED_SOURCE_ARCHIVE_MAX_UNCOMPRESSED_BYTES
            ):
                raise DashboardError("governed source archive expanded size is invalid")

            source_payloads: list[tuple[str, bytes]] = []
            dependency_payload: bytes | None = None
            for entry in entries:
                name = entry.filename
                parts = name.split("/")
                if (
                    entry.is_dir()
                    or not name
                    or "\\" in name
                    or any(not part or part in {".", ".."} for part in parts)
                    or entry.flag_bits & 0x1
                    or entry.compress_type != zipfile.ZIP_DEFLATED
                    or entry.file_size
                    > _GOVERNED_SOURCE_ARCHIVE_MAX_UNCOMPRESSED_BYTES
                ):
                    raise DashboardError(
                        "governed source archive contains an unsafe entry"
                    )
                entry_payload = archive.read(entry)
                if len(entry_payload) != entry.file_size:
                    raise DashboardError("governed source archive entry ended early")
                if name == dependency_entry:
                    if dependency_payload is not None:
                        raise DashboardError(
                            "governed source archive repeats the dependency lock"
                        )
                    dependency_payload = entry_payload
                    continue
                if (
                    len(parts) < 2
                    or parts[0] != "twen"
                    or not name.endswith(".py")
                ):
                    raise DashboardError(
                        "governed source archive contains an unexpected path"
                    )
                relative = "/".join(parts[1:])
                source_payloads.append((relative, entry_payload))
    except DashboardError:
        raise
    except Exception as error:
        raise DashboardError("cannot authenticate governed source archive") from error

    if dependency_payload is None or not source_payloads:
        raise DashboardError("governed source archive is incomplete")
    source_payloads.sort(key=lambda item: item[0])
    digest = hashlib.sha256()
    digest.update(SOURCE_TREE_HASH_SCHEMA)
    for relative, source_payload in source_payloads:
        relative_bytes = relative.encode("utf-8")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(len(source_payload).to_bytes(8, "big"))
        digest.update(source_payload)
    return digest.hexdigest(), hashlib.sha256(dependency_payload).hexdigest()


def _session_source_mix(session: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Expose the authenticated runtime mix without parsing a checkpoint."""

    if session is None:
        return None
    source_mix = session.get("source_mix")
    return dict(source_mix) if isinstance(source_mix, Mapping) else None


def _process_arguments(pid: int) -> tuple[list[str], Path] | None:
    if not _pid_alive(pid):
        return None
    try:
        arguments = [
            item.decode("utf-8", errors="surrogateescape")
            for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if item
        ]
        cwd = Path(f"/proc/{pid}/cwd").resolve()
    except OSError:
        return None
    return arguments, cwd


def _argument_value(arguments: list[str], name: str) -> str | None:
    indexes = [index for index, value in enumerate(arguments) if value == name]
    if len(indexes) != 1 or indexes[0] + 1 >= len(arguments):
        return None
    return arguments[indexes[0] + 1]


def _governed_training_payload(command: list[str], *, label: str) -> list[str]:
    matches = [
        index
        for index in range(len(command) - 1)
        if command[index : index + 2] == ["-m", "twen.cli"]
    ]
    if len(matches) != 1:
        raise DashboardError(f"{label} has no unique '-m twen.cli' payload")
    index = matches[0]
    if "-c" in command[:index]:
        raise DashboardError(f"{label} uses a forbidden Python -c prefix")
    return command[index:]


def _expected_governed_active_command(
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
) -> list[str]:
    """Reconstruct the controller-owned command, including crash recovery."""

    try:
        command = build_governed_train_command(plan, state)
    except GovernedControllerError as error:
        raise DashboardError(
            f"governed controller active command is invalid: {error}"
        ) from error
    recovery = state.get("recovery_checkpoint")
    if recovery is None:
        return command
    if not isinstance(recovery, Mapping) or not isinstance(recovery.get("path"), str):
        raise DashboardError("governed controller recovery checkpoint is invalid")
    resume_indexes = [
        index for index, value in enumerate(command[:-1]) if value == "--resume"
    ]
    if len(resume_indexes) != 1:
        raise DashboardError("governed controller command has no unique resume argument")
    command[resume_indexes[0] + 1] = str(recovery["path"])
    fork_indexes = [
        index for index, value in enumerate(command[:-1]) if value == "--fork-from"
    ]
    if len(fork_indexes) > 1:
        raise DashboardError("governed controller command has duplicate fork arguments")
    if fork_indexes:
        index = fork_indexes[0]
        del command[index : index + 2]
    return command


def _process_matches_training_profile(pid: int, profile: LaunchProfile) -> bool:
    """Defend against stale/reused PIDs before signaling rank-zero training."""

    process = _process_arguments(pid)
    if process is None:
        return False
    arguments, cwd = process
    if "train" not in arguments:
        return False
    executable_markers = {
        "twen",
        "twen.cli",
        "twen.__main__",
    }
    if not any(
        Path(argument).name in executable_markers or argument in executable_markers
        for argument in arguments
    ):
        return False
    try:
        config_index = arguments.index("--config") + 1
        configured = Path(arguments[config_index])
    except (ValueError, IndexError):
        return False
    if not configured.is_absolute():
        configured = cwd / configured
    return configured.resolve() == profile.config_path


def _process_matches_governed_controller(
    pid: int,
    profile: LaunchProfile,
    controller_state: Mapping[str, Any] | None,
    *,
    project_root: Path,
) -> bool:
    """Authenticate one exact, active, sealed governed-controller root process."""

    if (
        profile.launch_kind != "governed_v4"
        or profile.governed_readiness_path is None
        or profile.governed_state_path is None
        or profile.governed_plan_id is None
        or not isinstance(profile.governed_source_tree_sha256, str)
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            profile.governed_source_tree_sha256,
        )
        or profile.governed_dependency_lock_name
        not in {"uv.lock", "pyproject.toml"}
        or not isinstance(profile.governed_dependency_lock_sha256, str)
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            profile.governed_dependency_lock_sha256,
        )
        or not isinstance(controller_state, Mapping)
        or controller_state.get("schema_version") != 1
        or controller_state.get("status") not in _ACTIVE_TRAINING_STATES
        or controller_state.get("profile_id") != profile.profile_id
        or controller_state.get("run_id") != profile.run_id
        or controller_state.get("launch_kind") != "governed_v4"
        or controller_state.get("governed_plan_id") != profile.governed_plan_id
        or controller_state.get("governed_state_path") != str(profile.governed_state_path)
        or controller_state.get("project_root") != str(project_root)
        or controller_state.get("pid") != pid
    ):
        return False
    process_group_id = controller_state.get("process_group_id")
    start_time_ticks = controller_state.get("process_start_time_ticks")
    command = controller_state.get("process_cmdline")
    executable = controller_state.get("process_executable")
    controller_fd = controller_state.get("controller_snapshot_fd")
    source_fd = controller_state.get("source_snapshot_fd")
    source_transport_fd = controller_state.get("source_snapshot_transport_fd")
    controller_snapshot_sha = controller_state.get("controller_snapshot_sha256")
    source_snapshot_sha = controller_state.get("source_snapshot_sha256")
    source_tree_sha = controller_state.get("source_tree_sha256")
    dependency_lock_sha = controller_state.get("dependency_lock_sha256")
    if (
        not isinstance(process_group_id, int)
        or isinstance(process_group_id, bool)
        or process_group_id != pid
        or not isinstance(start_time_ticks, int)
        or isinstance(start_time_ticks, bool)
        or start_time_ticks < 0
        or not isinstance(command, list)
        or not all(isinstance(item, str) and item for item in command)
        or not isinstance(executable, str)
        or not executable
        or not isinstance(controller_fd, int)
        or isinstance(controller_fd, bool)
        or controller_fd < 3
        or not isinstance(source_fd, int)
        or isinstance(source_fd, bool)
        or source_fd != 0
        or source_fd == controller_fd
        or not isinstance(source_transport_fd, int)
        or isinstance(source_transport_fd, bool)
        or source_transport_fd < 3
        or source_transport_fd == controller_fd
        or not isinstance(controller_snapshot_sha, str)
        or not secrets.compare_digest(
            controller_snapshot_sha,
            profile.governed_controller_sha256,
        )
        or not isinstance(source_snapshot_sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", source_snapshot_sha)
        or not isinstance(source_tree_sha, str)
        or not secrets.compare_digest(
            source_tree_sha,
            profile.governed_source_tree_sha256,
        )
        or not isinstance(dependency_lock_sha, str)
        or not secrets.compare_digest(
            dependency_lock_sha,
            profile.governed_dependency_lock_sha256,
        )
    ):
        return False
    expected_command = [
        sys.executable,
        "-B",
        "-P",
        f"/proc/self/fd/{controller_fd}",
        "--readiness",
        str(profile.governed_readiness_path),
        "--action",
        "run",
        "--state",
        str(profile.governed_state_path),
        "--ack",
        f"RUN {profile.governed_plan_id}",
    ]
    if command != expected_command:
        return False
    if controller_state.get("command_sha256") != hashlib.sha256(
        "\0".join(command).encode()
    ).hexdigest():
        return False
    try:
        identity = _read_linux_process_identity(pid)
        environment = _read_process_environment(pid)
        controller_sha = _sealed_process_fd_sha256(pid, controller_fd)
        source_payload = _sealed_process_fd_bytes(pid, source_fd)
        source_transport_payload = _sealed_process_fd_bytes(
            pid,
            source_transport_fd,
        )
        source_sha = hashlib.sha256(source_payload).hexdigest()
        archive_source_tree_sha, archive_dependency_lock_sha = (
            _governed_source_archive_hashes(
                source_payload,
                dependency_lock_name=profile.governed_dependency_lock_name,
            )
        )
        identity_after = _read_linux_process_identity(pid)
    except DashboardError:
        return False
    return bool(
        identity_after == identity
        and list(identity.command) == command
        and identity.start_time_ticks == start_time_ticks
        and identity.cwd == project_root
        and str(identity.executable) == executable
        and identity.executable == Path(sys.executable).resolve()
        and identity.process_group_id == process_group_id
        and environment.get("PYTHONPATH") == f"/proc/self/fd/{source_fd}"
        and environment.get("PYTHONSAFEPATH") == "1"
        and environment.get("PYTHONUNBUFFERED") == "1"
        and controller_sha == controller_snapshot_sha
        and source_sha == source_snapshot_sha
        and source_payload == source_transport_payload
        and archive_source_tree_sha == source_tree_sha
        and archive_dependency_lock_sha == dependency_lock_sha
    )


def _process_matches_profile(pid: int, profile: LaunchProfile) -> bool:
    """Verify the long-lived process owned by one dashboard launch profile."""

    if profile.launch_kind == "governed_v4":
        return False
    return _process_matches_training_profile(pid, profile)


def _process_is_twen_training(pid: int) -> bool:
    """Verify an unmanaged PID is still some Twen training command."""

    process = _process_arguments(pid)
    if process is None:
        return False
    arguments, _cwd = process
    markers = {"twen", "twen.cli", "twen.__main__"}
    return "train" in arguments and any(
        Path(argument).name in markers or argument in markers for argument in arguments
    )


ProcessFactory = Callable[..., subprocess.Popen[bytes]]


def _reap_failed_launch(process: Any) -> None:
    """Unconditionally reap a child whose launch was not durably admitted.

    Every dashboard launch requests a new session, so the returned PID is also
    the expected process-group ID.  Kill the whole group before reaping the
    child: a governed controller may already have created evaluation or
    training descendants when a later identity/state/audit write fails.
    """

    raw_pid = getattr(process, "pid", None)
    pid = (
        raw_pid
        if isinstance(raw_pid, int)
        and not isinstance(raw_pid, bool)
        and raw_pid > 1
        and raw_pid != os.getpgrp()
        else None
    )
    if pid is not None:
        with suppress(OSError):
            os.killpg(pid, signal.SIGKILL)
    else:
        kill = getattr(process, "kill", None)
        if callable(kill):
            with suppress(OSError):
                kill()
    wait = getattr(process, "wait", None)
    if callable(wait):
        with suppress(OSError, subprocess.TimeoutExpired):
            wait(timeout=5.0)


@dataclass(frozen=True, slots=True)
class _GovernedExecutionSnapshot:
    """Sealed, inherited launch bytes; live repository paths are never executed."""

    controller_fd: int
    source_fd: int
    controller_sha256: str
    source_archive_sha256: str
    source_tree_sha256: str
    dependency_lock_sha256: str

    @property
    def controller_path(self) -> str:
        return f"/proc/self/fd/{self.controller_fd}"

    @property
    def source_path(self) -> str:
        return f"/proc/self/fd/{self.source_fd}"

    @property
    def runtime_source_fd(self) -> int:
        # Popen duplicates the sealed source memfd onto stdin.  Descriptor 0
        # survives the governed controller's later close_fds=True torchrun
        # launch, keeping the complete descendant process tree on this exact
        # authenticated source archive.
        return 0

    @property
    def runtime_source_path(self) -> str:
        return f"/proc/self/fd/{self.runtime_source_fd}"

    @property
    def pass_fds(self) -> tuple[int, int]:
        return (self.controller_fd, self.source_fd)

    def close(self) -> None:
        for descriptor in self.pass_fds:
            with suppress(OSError):
                os.close(descriptor)


def _create_sealed_memfd(name: str, payload: bytes) -> int:
    """Create one read-only sealed Linux memfd or fail closed."""

    libc = ctypes.CDLL(None, use_errno=True)
    create = getattr(libc, "memfd_create", None)
    if create is None:
        raise DashboardError("Linux memfd_create is required for governed Web launch")
    create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    create.restype = ctypes.c_int
    descriptor = int(
        create(
            name.encode("ascii"),
            _MEMFD_CLOEXEC | _MEMFD_ALLOW_SEALING,
        )
    )
    if descriptor < 0:
        error_number = ctypes.get_errno()
        raise DashboardError(
            f"cannot create sealed governed launch snapshot: errno {error_number}"
        )
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise DashboardError("cannot populate governed launch snapshot")
            offset += written
        os.lseek(descriptor, 0, os.SEEK_SET)
        fcntl.fcntl(descriptor, _F_ADD_SEALS, _REQUIRED_MEMFD_SEALS)
        seals = int(fcntl.fcntl(descriptor, _F_GET_SEALS))
        if seals & _REQUIRED_MEMFD_SEALS != _REQUIRED_MEMFD_SEALS:
            raise DashboardError("governed launch snapshot is not fully sealed")
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != len(payload):
            raise DashboardError("governed launch snapshot identity is invalid")
        return descriptor
    except (OSError, ValueError):
        with suppress(OSError):
            os.close(descriptor)
        raise
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        raise


def _source_tree_archive(
    plan: Mapping[str, Any],
) -> tuple[bytes, str, str]:
    source = plan.get("source_tree")
    dependency = plan.get("dependency_lock")
    project_root = Path(str(plan.get("project_root"))).resolve()
    if not isinstance(source, Mapping) or not isinstance(dependency, Mapping):
        raise DashboardError("governed plan source/dependency identity is incomplete")
    source_root = Path(str(source.get("path"))).resolve()
    if source_root != project_root / "src/twen":
        raise DashboardError("governed source tree is not the fixed Twen package")
    expected_source_sha = source.get("sha256")
    expected_dependency_sha = dependency.get("sha256")
    if not isinstance(expected_source_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_source_sha
    ):
        raise DashboardError("governed source tree SHA256 is invalid")
    if not isinstance(expected_dependency_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_dependency_sha
    ):
        raise DashboardError("governed dependency-lock SHA256 is invalid")

    try:
        entries = sorted(source_root.rglob("*"))
    except OSError as error:
        raise DashboardError("cannot inventory governed Twen source tree") from error
    if any(path.is_symlink() for path in entries):
        raise DashboardError("governed Twen source tree contains a symlink")
    python_files = [path for path in entries if path.suffix == ".py" and path.is_file()]
    if not python_files:
        raise DashboardError("governed Twen source tree contains no Python files")

    tree_digest = hashlib.sha256()
    tree_digest.update(SOURCE_TREE_HASH_SCHEMA)
    source_payloads: list[tuple[str, bytes]] = []
    for path in python_files:
        relative = path.relative_to(source_root).as_posix()
        payload = _read_regular_file_no_follow(path)
        if payload is None:
            raise DashboardError(f"governed source changed while snapshotting: {relative}")
        relative_bytes = relative.encode("utf-8")
        tree_digest.update(len(relative_bytes).to_bytes(8, "big"))
        tree_digest.update(relative_bytes)
        tree_digest.update(len(payload).to_bytes(8, "big"))
        tree_digest.update(payload)
        source_payloads.append((f"twen/{relative}", payload))
    actual_source_sha = tree_digest.hexdigest()
    if not secrets.compare_digest(actual_source_sha, expected_source_sha):
        raise DashboardError("governed source tree changed during launch admission")

    dependency_path = Path(str(dependency.get("path"))).resolve()
    if (
        dependency_path.parent != project_root
        or dependency_path.name not in {"uv.lock", "pyproject.toml"}
    ):
        raise DashboardError("governed dependency lock is outside the fixed project root")
    dependency_payload = _read_regular_file_no_follow(
        dependency_path,
        max_bytes=64 * 1024 * 1024,
    )
    if (
        dependency_payload is None
        or not secrets.compare_digest(
            hashlib.sha256(dependency_payload).hexdigest(),
            expected_dependency_sha,
        )
    ):
        raise DashboardError("governed dependency lock changed during launch admission")

    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for relative, payload in source_payloads:
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o444 << 16
            archive.writestr(info, payload)
        lock_info = zipfile.ZipInfo(
            f".twen-governed-dependency/{dependency_path.name}",
            date_time=(1980, 1, 1, 0, 0, 0),
        )
        lock_info.compress_type = zipfile.ZIP_DEFLATED
        lock_info.external_attr = 0o444 << 16
        archive.writestr(lock_info, dependency_payload)
    archive_payload = buffer.getvalue()
    return archive_payload, actual_source_sha, expected_dependency_sha


def _build_governed_execution_snapshot(
    profile: LaunchProfile,
    plan: Mapping[str, Any],
) -> _GovernedExecutionSnapshot:
    if (
        profile.governed_controller_path is None
        or profile.governed_controller_sha256 is None
    ):
        raise DashboardError("governed controller snapshot identity is incomplete")
    controller_payload = _read_regular_file_no_follow(profile.governed_controller_path)
    if controller_payload is None:
        raise DashboardError("governed controller changed while snapshotting")
    controller_sha = hashlib.sha256(controller_payload).hexdigest()
    if not secrets.compare_digest(controller_sha, profile.governed_controller_sha256):
        raise DashboardError("governed controller changed during launch admission")
    sources = plan.get("controller_sources")
    if not isinstance(sources, list) or not any(
        isinstance(source, Mapping)
        and Path(str(source.get("path"))).resolve() == profile.governed_controller_path
        and source.get("sha256") == controller_sha
        for source in sources
    ):
        raise DashboardError("governed plan does not bind the snapshotted controller")

    archive_payload, source_tree_sha, dependency_sha = _source_tree_archive(plan)
    controller_fd = _create_sealed_memfd("twen-v4-governed-controller", controller_payload)
    try:
        source_fd = _create_sealed_memfd("twen-v4-governed-source", archive_payload)
    except BaseException:
        with suppress(OSError):
            os.close(controller_fd)
        raise
    return _GovernedExecutionSnapshot(
        controller_fd=controller_fd,
        source_fd=source_fd,
        controller_sha256=controller_sha,
        source_archive_sha256=hashlib.sha256(archive_payload).hexdigest(),
        source_tree_sha256=source_tree_sha,
        dependency_lock_sha256=dependency_sha,
    )


@dataclass(frozen=True, slots=True)
class _TrainingLaunch:
    """One server-derived command mode; no field is populated from HTTP input."""

    mode: str
    resume: str
    fork_from: Path | None
    verified_checkpoint_id: str | None = None


@dataclass(frozen=True, slots=True)
class _DashboardTask:
    """One confined dashboard task and its safe public representation."""

    summary: dict[str, Any]
    task_data: dict[str, Any]
    root: Path
    run_key: str | None = None
    console_path: Path | None = None


class DashboardController:
    """Read training state and perform serialized, allowlisted actions."""

    def __init__(
        self,
        settings: DashboardSettings,
        *,
        process_factory: ProcessFactory = subprocess.Popen,
        gpu_monitor: GpuTelemetryMonitor | None = None,
    ) -> None:
        self.settings = settings
        self.cache = JsonlTailCache()
        self._process_factory = process_factory
        self.gpu_monitor = gpu_monitor or GpuTelemetryMonitor(
            journal_path=settings.state_dir / "gpu-telemetry.jsonl"
        )
        self._action_lock = threading.Lock()
        self._state_path = settings.state_dir / "controller-state.json"
        self._audit_path = settings.state_dir / "actions.jsonl"
        self._operations_cache: tuple[float, dict[str, Any]] | None = None
        settings.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    @contextmanager
    def _serialized_action(self) -> Any:
        """Serialize mutations across threads and independently started servers."""

        with self._action_lock, _exclusive_existing_directory_lock(
            self.settings.state_dir,
            timeout_seconds=_DASHBOARD_ACTION_LOCK_TIMEOUT_SECONDS,
        ):
            yield

    def _run_file(self, run_dir: Path, name: str) -> Path | None:
        confined_run_dir = _existing_directory_inside(self.settings.project_root, run_dir)
        if confined_run_dir is None:
            return None
        return _existing_file_inside(confined_run_dir, confined_run_dir / name)

    def _run_json(self, run_dir: Path, name: str) -> dict[str, Any] | None:
        path = self._run_file(run_dir, name)
        return _read_json(path) if path is not None else None

    def _run_records(self, run_dir: Path, name: str, *, limit: int) -> list[dict[str, Any]]:
        path = self._run_file(run_dir, name)
        return self.cache.read(path, limit=limit) if path is not None else []

    def _fixed_profile_path(self, path: Path, *, label: str) -> Path:
        """Reject a profile path that was redirected after settings were loaded."""

        current = _inside(self.settings.project_root, path, label=label)
        if current != path:
            raise DashboardError(f"{label} changed after dashboard startup: {current}")
        return current

    def configuration_status(self) -> dict[str, bool]:
        """Report whether the dashboard's own fixed allowlist needs a restart."""

        current_sha256 = _sha256_regular_file_no_follow(self.settings.dashboard_config_path)
        configuration_stale = bool(
            current_sha256 is None
            or not secrets.compare_digest(
                current_sha256,
                self.settings.dashboard_config_sha256,
            )
        )
        return {
            "configuration_stale": configuration_stale,
            "restart_required": configuration_stale,
        }

    def _governed_plan(self, profile: LaunchProfile) -> dict[str, Any]:
        if (
            profile.launch_kind != "governed_v4"
            or profile.governed_controller_path is None
            or profile.governed_controller_sha256 is None
            or profile.governed_readiness_path is None
            or profile.governed_readiness_sha256 is None
            or profile.governed_state_path is None
            or profile.governed_plan_id is None
        ):
            raise DashboardError("governed profile launch contract is incomplete")
        controller_path = self._fixed_profile_path(
            profile.governed_controller_path,
            label=f"profile {profile.profile_id!r} governed controller",
        )
        readiness_path = self._fixed_profile_path(
            profile.governed_readiness_path,
            label=f"profile {profile.profile_id!r} governed readiness",
        )
        state_path = self._fixed_profile_path(
            profile.governed_state_path,
            label=f"profile {profile.profile_id!r} governed state",
        )
        for path, expected, label in (
            (
                controller_path,
                profile.governed_controller_sha256,
                "governed controller",
            ),
            (
                readiness_path,
                profile.governed_readiness_sha256,
                "governed readiness",
            ),
        ):
            current = _sha256_regular_file_no_follow(path)
            if current is None or not secrets.compare_digest(current, expected):
                raise DashboardError(
                    f"{label} changed after dashboard startup; review and restart the dashboard"
                )
        try:
            plan = build_governed_plan(readiness_path)
        except GovernedControllerError as error:
            raise DashboardError(f"governed readiness authentication failed: {error}") from error
        if not secrets.compare_digest(str(plan.get("plan_id", "")), profile.governed_plan_id):
            raise DashboardError(
                "governed plan changed after dashboard startup; review and restart the dashboard"
            )
        config = plan.get("config")
        run = plan.get("run")
        if (
            not isinstance(config, Mapping)
            or Path(str(config.get("path"))).resolve() != profile.config_path
            or config.get("sha256") != profile.config_sha256
            or not isinstance(run, Mapping)
            or Path(str(run.get("output_dir"))).resolve() != profile.run_dir
            or run.get("run_id") != profile.run_id
            or run.get("stage") != profile.stage
        ):
            raise DashboardError("governed plan no longer binds the fixed profile config/run")
        expected_state = (
            profile.run_dir.parent
            / f".{profile.run_dir.name}.governed"
            / "controller-state.json"
        )
        if state_path != expected_state:
            raise DashboardError("governed state path no longer binds the fixed run")
        return plan

    def _authorized_governed_plan(
        self,
        profile: LaunchProfile,
        confirmation: str,
    ) -> dict[str, Any]:
        plan = self._governed_plan(profile)
        try:
            authorize_governed_run(plan, confirmation)
        except GovernedControllerError as error:
            raise DashboardError(str(error)) from error
        return plan

    def _governance_status(self, profile: LaunchProfile) -> dict[str, Any] | None:
        if profile.launch_kind != "governed_v4":
            return None
        try:
            plan = self._governed_plan(profile)
            if profile.governed_state_path is None:
                raise DashboardError("governed profile state path is missing")
            state = load_governed_controller_state(profile.governed_state_path, plan)
            status = governed_controller_status(plan, state)
        except (DashboardError, GovernedControllerError, OSError, ValueError) as error:
            return {
                "kind": "twen_v4_governed_web_launch_status",
                "blocked": True,
                "launch_enabled": False,
                "web_launch_enabled": profile.launch_enabled,
                "configuration_stale": True,
                "controller_state": "invalid",
                "plan_id": profile.governed_plan_id,
                "required_ack": None,
                "blockers": [str(error)],
            }
        blockers = [
            str(value)
            for value in status.get("blockers", [])
            if isinstance(value, str) and value
        ]
        if not profile.launch_enabled:
            blockers.append("dashboard formal profile launch_enabled is false")
        return {
            "kind": "twen_v4_governed_web_launch_status",
            "blocked": bool(status.get("blocked")) or not profile.launch_enabled,
            "launch_enabled": status.get("launch_enabled") is True,
            "web_launch_enabled": profile.launch_enabled,
            "configuration_stale": False,
            "controller_state": status.get("controller_state"),
            "plan_id": status.get("plan_id"),
            "required_ack": status.get("required_ack"),
            "completed_thresholds": status.get("completed_thresholds"),
            "next_threshold": status.get("next_threshold"),
            "current_checkpoint": status.get("current_checkpoint"),
            "blockers": blockers,
        }

    def _command(
        self,
        profile: LaunchProfile,
        launch: _TrainingLaunch,
        *,
        acknowledgement: str | None = None,
        governed_snapshot: _GovernedExecutionSnapshot | None = None,
    ) -> list[str]:
        if profile.launch_kind == "governed_v4":
            if (
                profile.governed_readiness_path is None
                or profile.governed_state_path is None
                or profile.governed_plan_id is None
                or acknowledgement != f"RUN {profile.governed_plan_id}"
                or launch.mode != "governed"
                or governed_snapshot is None
            ):
                raise DashboardError("governed command admission is incomplete")
            return [
                sys.executable,
                "-B",
                "-P",
                governed_snapshot.controller_path,
                "--readiness",
                str(profile.governed_readiness_path),
                "--action",
                "run",
                "--state",
                str(profile.governed_state_path),
                "--ack",
                acknowledgement,
            ]
        # This fixed entry point unconditionally calls
        # run_coordinated_training_preflight() before importing torch or
        # constructing an optimizer.  Do not replace it with a direct engine
        # invocation or add an HTTP-controlled preflight bypass.
        command = [
            sys.executable,
            "-m",
            "twen",
            "train",
            "--stage",
            profile.stage,
            "--config",
            str(profile.config_path),
            "--resume",
            launch.resume,
            "--progress",
            "never",
        ]
        if launch.fork_from is not None:
            command.extend(("--fork-from", str(launch.fork_from)))
        return command

    def _fixed_initial_fork_launch(
        self,
        profile: LaunchProfile,
        run_dir: Path,
    ) -> _TrainingLaunch:
        """Choose first-fork or authenticated auto-resume from server state.

        Finalized v2 profiles deliberately retain ``resume=none`` plus a
        fixed ``fork_from`` checkpoint: those are the only safe semantics for
        an empty destination. Once that destination contains run state, the
        fork source must never be applied again. A second launch is admitted
        only when the normal CheckpointManager fully hash-authenticates the
        checkpoint that ``--resume auto`` will select and the immutable
        resolved configuration still matches the pinned profile.

        This is an admission check, not a replacement for training preflight.
        The fixed training entry point still validates source/data/effective
        fingerprints and invokes the resume loader before optimizer work.
        """

        if profile.resume != "none" or profile.fork_from is None:
            return _TrainingLaunch(
                mode="configured",
                resume=profile.resume,
                fork_from=profile.fork_from,
            )
        if not run_dir.exists():
            return _TrainingLaunch(
                mode="initial_fork",
                resume="none",
                fork_from=profile.fork_from,
            )
        if not run_dir.is_dir():
            raise DashboardError(f"profile run path is not a directory: {run_dir}")
        try:
            entries = tuple(run_dir.iterdir())
        except OSError as error:
            raise DashboardError(f"cannot inspect profile run directory: {run_dir}") from error
        if not entries:
            return _TrainingLaunch(
                mode="initial_fork",
                resume="none",
                fork_from=profile.fork_from,
            )

        # Never follow a checkpoint-directory symlink during the full hash
        # pass. Other run metadata may coexist with a committed checkpoint,
        # but none of it contributes trust to this decision.
        if any(item.name.startswith("step-") and item.is_symlink() for item in entries):
            raise DashboardError(
                "run directory contains a symlinked checkpoint candidate; resume refused"
            )
        manager = CheckpointManager(
            run_dir,
            backend="auto",
            rank=0,
            world_size=1,
        )
        resolved = manager.find_latest_valid_with_metadata()
        if resolved is None:
            raise DashboardError(
                "non-empty run directory has no fully authenticated committed checkpoint; "
                "automatic relaunch refused"
            )
        checkpoint_path, metadata = resolved
        expected_checkpoint_path = run_dir / checkpoint_path.name
        if (
            checkpoint_path != expected_checkpoint_path
            or checkpoint_path.is_symlink()
            or checkpoint_path.resolve(strict=True) != checkpoint_path
        ):
            raise DashboardError("authenticated checkpoint path escaped the fixed run directory")

        current_config = load_train_config(profile.config_path)
        if current_config.run_id != profile.run_id or current_config.stage != profile.stage:
            raise DashboardError(
                "pinned profile identity no longer matches its training configuration"
            )
        resolved_config_path = run_dir / "resolved_config.yaml"
        confined_resolved_config = _existing_file_inside(run_dir, resolved_config_path)
        if (
            confined_resolved_config is None
            or resolved_config_path.is_symlink()
            or confined_resolved_config != resolved_config_path
        ):
            raise DashboardError(
                "authenticated checkpoint has no fixed resolved_config.yaml; resume refused"
            )
        previous_config = load_train_config(confined_resolved_config)
        if previous_config.fingerprint() != current_config.fingerprint():
            raise DashboardError(
                "run directory resolved configuration is not exact-resume compatible "
                "with the pinned profile"
            )
        try:
            checkpoint_batch_tokens = int(metadata.get("global_batch_tokens", -1))
        except (TypeError, ValueError) as error:
            raise DashboardError(
                "authenticated checkpoint has invalid global batch metadata"
            ) from error
        if metadata.get("run_id") != current_config.run_id:
            raise DashboardError("authenticated checkpoint run_id differs from the pinned profile")
        if metadata.get("stage") != current_config.stage:
            raise DashboardError("authenticated checkpoint stage differs from the pinned profile")
        if checkpoint_batch_tokens != current_config.data.global_batch_tokens:
            raise DashboardError(
                "authenticated checkpoint global batch differs from the pinned profile"
            )
        return _TrainingLaunch(
            mode="resume_auto",
            resume="auto",
            fork_from=None,
            verified_checkpoint_id=checkpoint_path.name,
        )

    def _session(self, profile: LaunchProfile) -> tuple[dict[str, Any] | None, bool]:
        session = self._run_json(profile.run_dir, "rank0-session.json")
        if session is None:
            return None, False
        pid = session.get("pid")
        claims_running = session.get("status") == "running"
        valid_pid = isinstance(pid, int) and not isinstance(pid, bool) and pid > 1
        hostname_matches = session.get("hostname") == platform.node()
        process_matches = (
            _process_matches_training_profile(pid, profile)
            if profile.launch_kind == "governed_v4"
            else _process_matches_profile(pid, profile)
        )
        active = bool(
            claims_running
            and valid_pid
            and hostname_matches
            and process_matches
        )
        result = dict(session)
        if claims_running and not active:
            result["status"] = "stale"
            result["stale_reason"] = "rank-0 PID is absent or no longer matches this profile"
        return result, active

    def _controller_state(
        self,
        *,
        persist_exit: bool = True,
    ) -> dict[str, Any] | None:
        state = _read_json_regular_no_follow(self._state_path)
        if state is None:
            return None
        pid = state.get("pid")
        profile_id = state.get("profile_id")
        if not isinstance(pid, int) or isinstance(pid, bool) or not isinstance(profile_id, str):
            return state
        try:
            profile = self.settings.profile(profile_id)
        except DashboardError:
            return state
        if state.get("status") in {
            "launching",
            "running",
            "stop_requested",
        } and not (
            _process_matches_governed_controller(
                pid,
                profile,
                state,
                project_root=self.settings.project_root,
            )
            if profile.launch_kind == "governed_v4"
            else _process_matches_profile(pid, profile)
        ):
            state = {
                **state,
                "status": "exited",
                "updated_at_utc": _utc_now(),
            }
            if persist_exit:
                _atomic_json(self._state_path, state)
        return state

    def active_profile(
        self,
        *,
        persist_controller_exit: bool = True,
    ) -> LaunchProfile | None:
        for profile in self.settings.profiles:
            _, active = self._session(profile)
            if active:
                return profile
        state = self._controller_state(persist_exit=persist_controller_exit)
        if state and state.get("status") in {"launching", "running", "stop_requested"}:
            try:
                return self.settings.profile(str(state["profile_id"]))
            except DashboardError:
                pass
        return None

    def _verified_governed_process_group(
        self,
        profile: LaunchProfile,
        controller_state: Mapping[str, Any] | None,
    ) -> tuple[int, int]:
        if profile.launch_kind != "governed_v4" or controller_state is None:
            raise DashboardError(
                f"profile {profile.profile_id!r} has no tracked governed controller"
            )
        controller_pid = controller_state.get("pid")
        process_group_id = controller_state.get("process_group_id")
        if (
            controller_state.get("profile_id") != profile.profile_id
            or controller_state.get("launch_kind") != "governed_v4"
            or controller_state.get("status") not in _ACTIVE_TRAINING_STATES
            or controller_state.get("run_id") != profile.run_id
            or controller_state.get("governed_plan_id") != profile.governed_plan_id
            or not isinstance(controller_pid, int)
            or isinstance(controller_pid, bool)
            or controller_pid <= 1
            or not isinstance(process_group_id, int)
            or isinstance(process_group_id, bool)
            or process_group_id != controller_pid
            or process_group_id <= 1
            or process_group_id == os.getpgrp()
        ):
            raise DashboardError("governed controller process-group identity is invalid")
        if not _process_matches_governed_controller(
            controller_pid,
            profile,
            controller_state,
            project_root=self.settings.project_root,
        ):
            raise DashboardError(
                f"profile {profile.profile_id!r} has no exact active controller process group"
            )
        return controller_pid, process_group_id

    def _verified_rank0_training_pid(
        self,
        profile: LaunchProfile,
        controller_state: Mapping[str, Any] | None,
    ) -> int:
        """Authenticate the exact governed rank-zero trainer before SIGUSR1."""

        if profile.launch_kind != "governed_v4":
            raise DashboardError("strong rank-zero authentication requires a governed profile")
        if (
            not isinstance(controller_state, Mapping)
            or controller_state.get("status") not in {"launching", "running"}
        ):
            raise DashboardError("governed dashboard controller is not save-active")
        plan = self._governed_plan(profile)
        controller_pid, controller_group = self._verified_governed_process_group(
            profile,
            controller_state,
        )
        governed_state_path = profile.governed_state_path
        if governed_state_path is None:
            raise DashboardError("governed state path is missing")
        governed_state = _read_json_regular_no_follow(governed_state_path)
        if governed_state is None:
            raise DashboardError("governed controller state is not a stable regular JSON file")
        try:
            status = governed_controller_status(plan, governed_state)
        except GovernedControllerError as error:
            raise DashboardError(
                f"governed controller state authentication failed: {error}"
            ) from error
        if (
            status.get("controller_state") != "running"
            or governed_state.get("status") != "running"
        ):
            raise DashboardError("governed controller has no running training segment")
        active_command = governed_state.get("active_command")
        if not isinstance(active_command, list) or not all(
            isinstance(item, str) and item for item in active_command
        ):
            raise DashboardError("governed controller has no exact active command")
        if active_command != _expected_governed_active_command(plan, governed_state):
            raise DashboardError(
                "governed controller active command differs from the immutable plan"
            )

        run = plan.get("run")
        config = plan.get("config")
        source_tree = plan.get("source_tree")
        dependency_lock = plan.get("dependency_lock")
        if not all(
            isinstance(value, Mapping)
            for value in (run, config, source_tree, dependency_lock)
        ):
            raise DashboardError("governed plan process identity is incomplete")
        expected_root = self.settings.project_root
        if Path(str(plan.get("project_root"))).resolve() != expected_root:
            raise DashboardError("governed plan project root differs from the dashboard")
        run_dir = _existing_directory_inside(expected_root, profile.run_dir)
        if run_dir != profile.run_dir:
            raise DashboardError("governed run directory is missing or redirected")
        session_path = run_dir / "rank0-session.json"
        session = _read_json_regular_no_follow(session_path)
        if session is None:
            raise DashboardError("rank-zero session is not a stable regular JSON file")

        pid = session.get("pid")
        rank = session.get("rank")
        world_size = session.get("world_size")
        recorded_start = session.get("process_start_time_ticks")
        recorded_command = session.get("process_cmdline")
        if (
            session.get("schema_version") != 1
            or session.get("status") != "running"
            or not isinstance(pid, int)
            or isinstance(pid, bool)
            or pid <= 1
            or pid == controller_pid
            or not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank != 0
            or not isinstance(world_size, int)
            or isinstance(world_size, bool)
            or not isinstance(recorded_start, int)
            or isinstance(recorded_start, bool)
            or recorded_start < 0
            or not isinstance(recorded_command, list)
            or not all(isinstance(item, str) and item for item in recorded_command)
        ):
            raise DashboardError("rank-zero session process identity is incomplete")

        expected_world_size = run.get("world_size")
        expected_config_path = Path(str(config.get("path"))).resolve()
        expected_dependency_path = Path(str(dependency_lock.get("path"))).resolve()
        recorded_cwd = session.get("cwd")
        recorded_config_path = session.get("config_path")
        recorded_dependency_path = session.get("dependency_lock")
        if (
            session.get("hostname") != platform.node()
            or session.get("run_id") != profile.run_id
            or session.get("run_id") != run.get("run_id")
            or session.get("stage") != profile.stage
            or session.get("stage") != run.get("stage")
            or world_size != expected_world_size
            or not isinstance(recorded_cwd, str)
            or Path(recorded_cwd).resolve() != expected_root
            or not isinstance(recorded_config_path, str)
            or Path(recorded_config_path).resolve() != profile.config_path
            or Path(recorded_config_path).resolve() != expected_config_path
            or session.get("config_sha256") != profile.config_sha256
            or session.get("config_sha256") != config.get("sha256")
            or session.get("source_tree_sha256") != source_tree.get("sha256")
            or not isinstance(recorded_dependency_path, str)
            or Path(recorded_dependency_path).resolve() != expected_dependency_path
            or session.get("dependency_lock_sha256") != dependency_lock.get("sha256")
        ):
            raise DashboardError("rank-zero session differs from the governed plan")

        try:
            identity = _read_linux_process_identity(pid)
        except DashboardError as error:
            raise DashboardError("rank-zero process identity cannot be read") from error
        actual_command = list(identity.command)
        try:
            actual_payload = _governed_training_payload(
                actual_command,
                label="rank-zero process command",
            )
            active_payload = _governed_training_payload(
                active_command,
                label="governed controller active command",
            )
        except DashboardError:
            raise
        if (
            identity.pid != pid
            or actual_command != recorded_command
            or identity.start_time_ticks != recorded_start
            or identity.cwd != expected_root
            or identity.executable != Path(sys.executable).resolve()
            or identity.process_group_id != controller_group
            or Path(actual_command[0]).resolve() != identity.executable
            or actual_payload != active_payload
        ):
            raise DashboardError("rank-zero live process differs from its authenticated session")

        owner = _held_run_lock_owner(run_dir / ".run.lock")
        if (
            owner is None
            or owner.get("pid") != pid
            or owner.get("host") != socket.gethostname()
            or owner.get("host") != session.get("hostname")
        ):
            raise DashboardError("rank-zero trainer does not own the live run lock")

        try:
            identity_after = _read_linux_process_identity(pid)
        except DashboardError as error:
            raise DashboardError("rank-zero process changed during authentication") from error
        session_after = _read_json_regular_no_follow(session_path)
        governed_state_after = _read_json_regular_no_follow(governed_state_path)
        owner_after = _held_run_lock_owner(run_dir / ".run.lock")
        plan_after = self._governed_plan(profile)
        if (
            identity_after != identity
            or session_after != session
            or governed_state_after != governed_state
            or owner_after != owner
            or plan_after != plan
        ):
            raise DashboardError("rank-zero training identity changed during authentication")
        return pid

    def profile_status(self, profile: LaunchProfile) -> dict[str, Any]:
        session, active = self._session(profile)
        governance = self._governance_status(profile)
        metrics = self._run_records(profile.run_dir, "metrics.jsonl", limit=1)
        telemetry = self._run_records(profile.run_dir, "telemetry.jsonl", limit=1)
        events = self._run_records(profile.run_dir, "events.jsonl", limit=1)
        controller_state = self._controller_state()
        controller_for_profile = (
            controller_state
            if controller_state and controller_state.get("profile_id") == profile.profile_id
            else None
        )
        state = "not_started"
        if active:
            state = "running"
        elif controller_for_profile is not None and controller_for_profile.get("status") in {
            "launching",
            "running",
            "stop_requested",
        }:
            state = str(controller_for_profile["status"])
        elif session is not None:
            state = str(session.get("status", "unknown"))
        elif controller_for_profile is not None:
            state = str(controller_for_profile.get("status", "unknown"))
        if (
            profile.launch_kind == "governed_v4"
            and not active
            and (
                controller_for_profile is None
                or controller_for_profile.get("status") not in _ACTIVE_TRAINING_STATES
            )
            and isinstance(governance, Mapping)
        ):
            governed_state = governance.get("controller_state")
            if governed_state == "completed":
                state = "completed"
            elif governed_state in {"running", "awaiting_evaluation", "resume_authorized"}:
                state = "paused"
            elif governed_state in {"halted", "review_required", "failed", "invalid"}:
                state = str(governed_state)
            elif governed_state == "not_started":
                state = "not_started"
        effectively_active = active or state in _ACTIVE_TRAINING_STATES
        configuration = self.configuration_status()
        governance_ready = bool(
            governance is None
            or (
                governance.get("blocked") is False
                and governance.get("configuration_stale") is False
            )
        )
        start_available = bool(
            profile.launch_enabled
            and governance_ready
            and not effectively_active
            and state not in _COMPLETED_TRAINING_STATES
            and state not in {"halted", "review_required", "failed", "invalid"}
            and not configuration["configuration_stale"]
        )
        start_action = "start" if state == "not_started" else "resume"
        last_update_unix, last_update_utc = _run_last_update(
            self.settings.project_root,
            profile.run_dir,
        )
        return {
            "profile_id": profile.profile_id,
            "label": profile.label,
            "run_id": profile.run_id,
            "stage": profile.stage,
            "launch_kind": profile.launch_kind,
            "config_sha256": profile.config_sha256,
            "state": state,
            "active": effectively_active,
            "launch_enabled": profile.launch_enabled,
            **configuration,
            # This is deliberately only a cheap UI/status hint.  It must never
            # inspect or trust checkpoint contents during polling; start()
            # performs the full hash/compatibility admission when clicked.
            "start_available": start_available,
            "start_action": start_action,
            "preflight_enforced": True,
            "governed_controller_enforced": profile.launch_kind == "governed_v4",
            "start_confirmation": (
                governance.get("required_ack")
                if isinstance(governance, Mapping)
                and isinstance(governance.get("required_ack"), str)
                else profile.start_confirmation
            ),
            "stop_confirmation": profile.stop_confirmation,
            "save_confirmation": profile.save_confirmation,
            "save_available": active,
            "stop_available": effectively_active,
            "governance": governance,
            "session": session,
            "source_mix": _session_source_mix(session),
            "allow_corpus_reuse": (
                session.get("allow_corpus_reuse")
                if isinstance(session, Mapping)
                and isinstance(session.get("allow_corpus_reuse"), bool)
                else None
            ),
            "controller": controller_for_profile,
            "latest_metric": metrics[-1] if metrics else None,
            "latest_telemetry": telemetry[-1] if telemetry else None,
            "latest_event": events[-1] if events else None,
            "last_update_unix": last_update_unix,
            "last_update_utc": last_update_utc,
        }

    def _discovered_runs(self) -> list[tuple[str, Path, dict[str, Any] | None]]:
        """Enumerate read-only historical runs below the fixed project runs root."""

        root = _existing_directory_inside(
            self.settings.project_root,
            self.settings.project_root / "runs",
        )
        if root is None:
            return []
        configured = {profile.run_dir for profile in self.settings.profiles}
        discovered: list[tuple[str, Path, dict[str, Any] | None]] = []
        try:
            candidates = sorted(root.iterdir(), key=lambda item: item.name)
        except OSError:
            return []
        for candidate in candidates:
            # Historical discovery has no reason to trust directory symlinks.
            # Rejecting them also closes a race where a link is retargeted
            # between catalog construction and a later log-tail request.
            if candidate.is_symlink():
                continue
            run_dir = _existing_directory_inside(root, candidate)
            if run_dir is None or run_dir in configured:
                continue
            markers = (
                "metrics.jsonl",
                "telemetry.jsonl",
                "events.jsonl",
                "rank0-session.json",
                "resolved_config.yaml",
            )
            if not any(
                _existing_file_inside(run_dir, run_dir / marker) is not None for marker in markers
            ):
                continue
            relative = run_dir.relative_to(self.settings.project_root).as_posix()
            key = "history:" + hashlib.sha256(relative.encode()).hexdigest()[:16]
            session_path = _existing_file_inside(run_dir, run_dir / "rank0-session.json")
            discovered.append(
                (key, run_dir, _read_json(session_path) if session_path is not None else None)
            )
        return discovered

    def _history_status(
        self,
        key: str,
        run_dir: Path,
        session: dict[str, Any] | None,
    ) -> dict[str, Any]:
        metrics = self._run_records(run_dir, "metrics.jsonl", limit=1)
        telemetry = self._run_records(run_dir, "telemetry.jsonl", limit=1)
        events = self._run_records(run_dir, "events.jsonl", limit=1)
        state = str(session.get("status", "unknown")) if session else "not_started"
        active = False
        if session and state == "running":
            pid = session.get("pid")
            active = bool(
                session.get("hostname") == platform.node()
                and isinstance(pid, int)
                and not isinstance(pid, bool)
                and _process_is_twen_training(pid)
            )
            if not active:
                state = "stale"
        last_update_unix, last_update_utc = _run_last_update(
            self.settings.project_root,
            run_dir,
        )
        return {
            "key": key,
            "profile_id": None,
            "label": str(session.get("run_id"))
            if session and session.get("run_id")
            else run_dir.name,
            "run_id": str(session.get("run_id"))
            if session and session.get("run_id")
            else run_dir.name,
            "stage": session.get("stage") if session else None,
            "state": state,
            "active": active,
            "history": True,
            "launch_enabled": False,
            "preflight_enforced": True,
            "session": session,
            "source_mix": _session_source_mix(session),
            "allow_corpus_reuse": (
                session.get("allow_corpus_reuse")
                if isinstance(session, Mapping)
                and isinstance(session.get("allow_corpus_reuse"), bool)
                else None
            ),
            "controller": None,
            "latest_metric": metrics[-1] if metrics else None,
            "latest_telemetry": telemetry[-1] if telemetry else None,
            "latest_event": events[-1] if events else None,
            "last_update_unix": last_update_unix,
            "last_update_utc": last_update_utc,
        }

    def run_catalog(self) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for profile in self.settings.profiles:
            status = self.profile_status(profile)
            status["key"] = f"profile:{profile.profile_id}"
            status["history"] = False
            runs.append(status)
        runs.extend(
            self._history_status(key, run_dir, session)
            for key, run_dir, session in self._discovered_runs()
        )
        return runs

    def _resolve_run(self, key: str) -> tuple[Path, dict[str, Any], LaunchProfile | None]:
        if key.startswith("profile:"):
            profile = self.settings.profile(key.removeprefix("profile:"))
            status = self.profile_status(profile)
            status["key"] = key
            status["history"] = False
            return profile.run_dir, status, profile
        for discovered_key, run_dir, session in self._discovered_runs():
            if key == discovered_key:
                return run_dir, self._history_status(key, run_dir, session), None
        raise DashboardError(f"unknown run catalog key: {key!r}")

    def operations_status(self) -> dict[str, Any]:
        """Discover data/KD and evaluation artifacts under fixed roots."""

        now = time.monotonic()
        if self._operations_cache is not None and now - self._operations_cache[0] < 5.0:
            cached = self._operations_cache[1]
            raw_evaluations = cached.get("evaluations")
            for evaluation in raw_evaluations if isinstance(raw_evaluations, list) else []:
                if not isinstance(evaluation, dict) or not isinstance(evaluation.get("name"), str):
                    continue
                output = _existing_directory_inside(
                    self.settings.project_root,
                    self.settings.project_root / evaluation["name"],
                )
                if output is None:
                    evaluation["status"] = "paused"
                    evaluation["gpu_relevant"] = False
                    continue
                held = _exclusive_advisory_lock_is_held(output / ".eval.lock")
                manifest_file = _existing_file_inside(output, output / "manifest.json")
                complete_file = _existing_file_inside(output, output / "COMPLETE")
                committed = manifest_file is not None and complete_file is not None
                plan_path = _existing_file_inside(output, output / "PLAN.json")
                device_type = _authenticated_evaluation_device_type(output, plan_path)
                evaluation["status"] = (
                    "in_progress" if held else ("complete" if committed else "paused")
                )
                evaluation["device_type"] = device_type
                evaluation["gpu_relevant"] = held and device_type == "cuda"
            return cached
        evaluations: list[dict[str, Any]] = []
        evaluation_outputs: set[Path] = set()
        for evaluation_root_name in ("eval", "artifacts/evaluations"):
            evaluation_root = self.settings.project_root / evaluation_root_name
            if evaluation_root.is_dir():
                for raw_plan_path in evaluation_root.rglob("PLAN.json"):
                    plan_path = _existing_file_inside(
                        self.settings.project_root,
                        raw_plan_path,
                    )
                    if (
                        plan_path is not None
                        and plan_path == raw_plan_path
                        and not raw_plan_path.is_symlink()
                    ):
                        evaluation_outputs.add(plan_path.parent)
                for raw_lock_path in evaluation_root.rglob(".eval.lock"):
                    lock_path = _existing_file_inside(
                        self.settings.project_root,
                        raw_lock_path,
                    )
                    if (
                        lock_path is not None
                        and lock_path == raw_lock_path
                        and not raw_lock_path.is_symlink()
                        and _exclusive_advisory_lock_is_held(lock_path)
                    ):
                        output = _existing_directory_inside(
                            self.settings.project_root,
                            lock_path.parent,
                        )
                        if output is not None:
                            evaluation_outputs.add(output)
        for output in sorted(evaluation_outputs)[:200]:
            plan_path = _existing_file_inside(output, output / "PLAN.json")
            held = _exclusive_advisory_lock_is_held(output / ".eval.lock")
            if plan_path is None and not held:
                continue
            manifest_path = output / "manifest.json"
            complete_path = output / "COMPLETE"
            manifest_file = _existing_file_inside(output, manifest_path)
            complete_file = _existing_file_inside(output, complete_path)
            committed = manifest_file is not None and complete_file is not None
            status = "in_progress" if held else ("complete" if committed else "paused")
            device_type = _authenticated_evaluation_device_type(output, plan_path)
            role_completes = sum(
                1
                for item in output.rglob("COMPLETE")
                if item != complete_path and _existing_file_inside(output, item) is not None
            )
            detail_parts = [f"{role_completes} committed evaluation shard(s)"]
            manifest = _read_json(manifest_file) if manifest_file is not None else None
            if manifest:
                roles = manifest.get("roles")
                if isinstance(roles, dict):
                    role_summary = []
                    for name, value in roles.items():
                        mean_nll = value.get("mean_nll") if isinstance(value, dict) else None
                        if mean_nll is not None:
                            with suppress(TypeError, ValueError):
                                role_summary.append(f"{name} NLL {float(mean_nll):.4f}")
                    if role_summary:
                        detail_parts.append(", ".join(role_summary))
                acceptance = manifest.get("acceptance")
                if isinstance(acceptance, dict):
                    booleans = [
                        f"{name}={'pass' if value else 'fail'}"
                        for name, value in acceptance.items()
                        if isinstance(value, bool)
                    ]
                    if booleans:
                        detail_parts.append(", ".join(booleans))
            marker = manifest_file or plan_path
            if marker is None:
                marker = _existing_file_inside(output, output / ".eval.lock")
            try:
                modified_at = marker.stat().st_mtime if marker is not None else 0.0
            except OSError:
                modified_at = 0.0
            evaluations.append(
                {
                    "name": output.relative_to(self.settings.project_root).as_posix(),
                    "status": status,
                    "device_type": device_type,
                    "gpu_relevant": held and device_type == "cuda",
                    "detail": " · ".join(detail_parts),
                    "modified_at": datetime.fromtimestamp(modified_at, UTC).isoformat(),
                }
            )
        # Reports are intentionally not a dashboard task and are not scanned.
        reports: list[dict[str, Any]] = []
        data_jobs: list[dict[str, Any]] = []
        data_root = self.settings.project_root / "artifacts/data"
        status_paths: set[Path] = set()
        if data_root.is_dir():
            status_paths.update(data_root.glob("*-pipeline/status.json"))
            status_paths.update(data_root.glob("*-orchestration/status.json"))
        for raw_status_path in sorted(status_paths)[:100]:
            status_path = _existing_file_inside(self.settings.project_root, raw_status_path)
            if status_path is None:
                continue
            payload = _read_json(status_path)
            if payload is None:
                continue
            state = str(payload.get("status", "unknown"))
            current = payload.get("current_phase")
            if not isinstance(current, dict):
                current = None
            phase = payload.get("phase") or (current or {}).get("name")
            progress = payload.get("progress")
            detail_parts: list[str] = []
            if isinstance(phase, str) and phase:
                detail_parts.append(f"phase {phase}")
            if isinstance(progress, dict):
                percent = progress.get("percent")
                completed_tokens = progress.get("completed_tokens")
                total_tokens = progress.get("total_tokens")
                rate = progress.get("attempt_wall_tokens_per_second")
                eta = progress.get("eta_seconds")
                if isinstance(percent, (int, float)) and math.isfinite(float(percent)):
                    detail_parts.append(f"{float(percent):.2f}%")
                if (
                    isinstance(completed_tokens, int)
                    and not isinstance(completed_tokens, bool)
                    and isinstance(total_tokens, int)
                    and not isinstance(total_tokens, bool)
                ):
                    detail_parts.append(f"{completed_tokens:,}/{total_tokens:,} tok")
                if isinstance(rate, (int, float)) and math.isfinite(float(rate)):
                    detail_parts.append(f"{float(rate):,.0f} tok/s")
                if isinstance(eta, (int, float)) and math.isfinite(float(eta)):
                    seconds = max(0, round(float(eta)))
                    hours, remainder = divmod(seconds, 3600)
                    minutes = remainder // 60
                    detail_parts.append(
                        f"ETA {hours:d}h {minutes:02d}m" if hours else f"ETA {minutes:d}m"
                    )
            if not detail_parts:
                detail_parts.append(str(payload.get("kind", "durable data job")))
            data_jobs.append(
                {
                    "name": status_path.parent.relative_to(self.settings.project_root).as_posix(),
                    "status": state,
                    "detail": " · ".join(detail_parts),
                    "modified_at": datetime.fromtimestamp(
                        status_path.stat().st_mtime, UTC
                    ).isoformat(),
                }
            )
        value = {
            "data_jobs": data_jobs,
            "evaluations": evaluations,
            "reports": reports,
        }
        self._operations_cache = (now, value)
        return value

    @staticmethod
    def _normalized_task_state(state: object, *, active: bool = False) -> str | None:
        if active or state in {"running", "launching", "stop_requested", "in_progress"}:
            return "running"
        if state == "paused":
            return "paused"
        if state in {"complete", "completed"}:
            return "completed"
        return None

    @staticmethod
    def _safe_number(value: object) -> int | float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return value if math.isfinite(float(value)) else None

    @staticmethod
    def _safe_phase(value: object) -> str | None:
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", value):
            return None
        return value

    @classmethod
    def _safe_progress(
        cls,
        value: object,
        *,
        completed: bool = False,
    ) -> dict[str, int | float]:
        raw = value if isinstance(value, dict) else {}
        result: dict[str, int | float] = {}
        for name in (
            "percent",
            "fraction",
            "completed_tokens",
            "total_tokens",
            "remaining_tokens",
            "eta_seconds",
            "completed_shards",
            "total_shards",
            "completed_sequences",
            "total_sequences",
            "attempt_elapsed_seconds",
        ):
            safe = cls._safe_number(raw.get(name))
            if safe is not None:
                result[name] = safe
        for name in (
            "attempt_wall_tokens_per_second",
            "wall_tokens_per_second",
            "tokens_per_second",
        ):
            safe = cls._safe_number(raw.get(name))
            if safe is not None:
                result["tokens_per_second"] = safe
                break
        if completed:
            result.setdefault("percent", 100.0)
            result.setdefault("fraction", 1.0)
            result["eta_seconds"] = 0.0
        return result

    @classmethod
    def _progress_detail(cls, phase: str | None, progress: Mapping[str, Any]) -> str:
        parts = [f"phase {phase}"] if phase else []
        percent = cls._safe_number(progress.get("percent"))
        if percent is not None:
            parts.append(f"{float(percent):.2f}%")
        completed = cls._safe_number(progress.get("completed_tokens"))
        total = cls._safe_number(progress.get("total_tokens"))
        if isinstance(completed, int) and isinstance(total, int):
            parts.append(f"{completed:,}/{total:,} tok")
        rate = cls._safe_number(progress.get("tokens_per_second"))
        if rate is not None:
            parts.append(f"{float(rate):,.0f} tok/s")
        eta = cls._safe_number(progress.get("eta_seconds"))
        if eta is not None:
            seconds = max(0, round(float(eta)))
            hours, remainder = divmod(seconds, 3600)
            minutes = remainder // 60
            parts.append(f"ETA {hours:d}h {minutes:02d}m" if hours else f"ETA {minutes:d}m")
        return " · ".join(parts)

    @staticmethod
    def _task_sort_key(task: _DashboardTask) -> tuple[int, float, str]:
        modified = task.summary.get("updated_at_unix")
        timestamp = float(modified) if isinstance(modified, (int, float)) else 0.0
        return (
            0 if task.summary["state"] == "running" else 1,
            -timestamp,
            str(task.summary["key"]),
        )

    def _task_console(self, status_path: Path, payload: Mapping[str, Any]) -> Path | None:
        """Resolve a task log without allowing a status file to escape its directory."""

        task_root = status_path.parent
        raw_log = payload.get("log")
        if isinstance(raw_log, str) and 0 < len(raw_log) <= 4096:
            try:
                candidate = Path(raw_log)
                if not candidate.is_absolute():
                    candidate = task_root / candidate
                confined = _existing_file_inside(task_root, candidate)
            except (OSError, RuntimeError, ValueError):
                confined = None
            if confined is not None:
                return confined
        return _existing_file_inside(task_root, task_root / "console.log")

    def _relative_source(self, path: Path | None) -> str | None:
        confined = (
            _existing_file_inside(self.settings.project_root, path) if path is not None else None
        )
        if confined is None:
            return None
        return confined.relative_to(self.settings.project_root).as_posix()

    def _training_tasks(self) -> list[_DashboardTask]:
        tasks: list[_DashboardTask] = []
        for status in self.run_catalog():
            state = self._normalized_task_state(
                status.get("state"),
                active=bool(status.get("active")),
            )
            if (
                state is None
                and status.get("start_available") is True
                and status.get("start_action") == "resume"
            ):
                state = "paused"
            if state is None:
                continue
            run_key = str(status["key"])
            run_dir, _, _ = self._resolve_run(run_key)
            metric = status.get("latest_metric")
            metric = metric if isinstance(metric, dict) else {}
            progress: dict[str, int | float] = {}
            for name in ("step", "tokens"):
                safe = self._safe_number(metric.get(name))
                if safe is not None:
                    progress[name] = safe
            for name in ("wall_tokens_per_second", "wall_tok_s", "tokens_per_second"):
                safe = self._safe_number(metric.get(name))
                if safe is not None:
                    progress["tokens_per_second"] = safe
                    break
            profile_id = status.get("profile_id")
            phase = self._safe_phase(status.get("stage"))
            control = {
                "available": isinstance(profile_id, str),
                "profile_id": profile_id if isinstance(profile_id, str) else None,
                "launch_enabled": bool(status.get("launch_enabled", False)),
                "start_available": bool(status.get("start_available", False)),
                "start_action": status.get("start_action"),
                "preflight_enforced": bool(status.get("preflight_enforced", True)),
                "start_confirmation": status.get("start_confirmation"),
                "stop_confirmation": status.get("stop_confirmation"),
                "save_confirmation": status.get("save_confirmation"),
            }
            summary = {
                "key": run_key,
                "kind": "training",
                "label": str(status["label"]),
                "state": state,
                "active": state == "running",
                "gpu_relevant": state == "running",
                "phase": phase,
                "progress": progress,
                "detail": f"stage {phase}" if phase else "training",
                "updated_at": status.get("last_update_utc"),
                "updated_at_unix": status.get("last_update_unix"),
                "source": {"type": "training_run", "run_id": status.get("run_id")},
                "profile_id": profile_id,
                "control": control,
            }
            latest_metric = {
                name: safe
                for name in (
                    "step",
                    "tokens",
                    "loss",
                    "ntp_loss",
                    "kd_loss",
                    "mtp_loss",
                    "learning_rate",
                    "wall_tokens_per_second",
                    "compute_tokens_per_second",
                )
                if (safe := self._safe_number(metric.get(name))) is not None
            }
            tasks.append(
                _DashboardTask(
                    summary=summary,
                    task_data={
                        "kind": "training",
                        "phase": phase,
                        "progress": progress,
                        "latest_metric": latest_metric,
                        "control": control,
                    },
                    root=run_dir,
                    run_key=run_key,
                    console_path=self._run_file(run_dir, "console.log"),
                )
            )
        return tasks

    def _data_tasks(self, operations: Mapping[str, Any]) -> list[_DashboardTask]:
        tasks: list[_DashboardTask] = []
        jobs = operations.get("data_jobs")
        for job in jobs if isinstance(jobs, list) else []:
            if not isinstance(job, dict) or not isinstance(job.get("name"), str):
                continue
            task_root = _existing_directory_inside(
                self.settings.project_root,
                self.settings.project_root / job["name"],
            )
            if task_root is None:
                continue
            status_path = _existing_file_inside(task_root, task_root / "status.json")
            payload = _read_json(status_path) if status_path is not None else None
            if status_path is None or payload is None:
                continue
            state = self._normalized_task_state(payload.get("status"))
            if state is None:
                continue
            marker = str(payload.get("kind", "")).lower()
            authenticated_kd_status = marker == _KD_ORCHESTRATION_STATUS_KIND
            kind = (
                "kd"
                if authenticated_kd_status or "-kd-" in task_root.name
                else "data_pipeline"
            )
            current = payload.get("current_phase")
            current = current if isinstance(current, dict) else {}
            phase = self._safe_phase(payload.get("phase") or current.get("name"))
            if phase is None and state == "completed":
                phase = "complete"
            progress = self._safe_progress(
                payload.get("progress"),
                completed=state == "completed",
            )
            console_path = self._task_console(status_path, payload)
            source = {
                "status": self._relative_source(status_path),
                "console": self._relative_source(console_path),
            }
            relative_root = task_root.relative_to(self.settings.project_root).as_posix()
            digest = hashlib.sha256(relative_root.encode()).hexdigest()[:16]
            modified = status_path.stat().st_mtime
            label = task_root.name.removesuffix("-orchestration").removesuffix("-pipeline")
            task_data: dict[str, Any] = {
                "kind": kind,
                "phase": phase,
                "progress": progress,
                "source": source,
            }
            attempt = self._safe_number(payload.get("attempt"))
            if attempt is not None:
                task_data["attempt"] = attempt
            for name in ("optimizer_created", "training_started", "gpu_kd_started"):
                if isinstance(payload.get(name), bool):
                    task_data[name] = payload[name]
            summary = {
                "key": f"task:{kind}:{digest}",
                "kind": kind,
                "label": label,
                "state": state,
                "active": state == "running",
                "gpu_relevant": bool(
                    state == "running"
                    and authenticated_kd_status
                    and phase == "generate-kd"
                ),
                "phase": phase,
                "progress": progress,
                "detail": self._progress_detail(phase, progress),
                "updated_at": datetime.fromtimestamp(modified, UTC).isoformat(),
                "updated_at_unix": modified,
                "source": source,
                "profile_id": None,
                "control": {"available": False},
            }
            tasks.append(
                _DashboardTask(
                    summary=summary,
                    task_data=task_data,
                    root=task_root,
                    console_path=console_path,
                )
            )
        return tasks

    def _evaluation_tasks(self, operations: Mapping[str, Any]) -> list[_DashboardTask]:
        tasks: list[_DashboardTask] = []
        evaluations = operations.get("evaluations")
        for evaluation in evaluations if isinstance(evaluations, list) else []:
            if not isinstance(evaluation, dict) or not isinstance(evaluation.get("name"), str):
                continue
            task_root = _existing_directory_inside(
                self.settings.project_root,
                self.settings.project_root / evaluation["name"],
            )
            if task_root is None:
                continue
            state = self._normalized_task_state(evaluation.get("status"))
            if state is None:
                continue
            plan_path = _existing_file_inside(task_root, task_root / "PLAN.json")
            manifest_path = _existing_file_inside(task_root, task_root / "manifest.json")
            console_path = _existing_file_inside(task_root, task_root / "console.log")
            lock_path = _existing_file_inside(task_root, task_root / ".eval.lock")
            source = {
                "plan": self._relative_source(plan_path),
                "manifest": self._relative_source(manifest_path),
                "console": self._relative_source(console_path),
            }
            roles: dict[str, dict[str, int | float]] = {}
            acceptance: dict[str, bool | int | float] = {}
            manifest = _read_json(manifest_path) if manifest_path is not None else None
            raw_roles = manifest.get("roles") if manifest is not None else None
            if isinstance(raw_roles, dict):
                for name, value in raw_roles.items():
                    if not isinstance(name, str) or not isinstance(value, dict):
                        continue
                    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:64]
                    if not safe_name:
                        continue
                    safe_values: dict[str, int | float] = {}
                    for field in ("mean_nll", "perplexity", "predicted_tokens", "sequences"):
                        safe = self._safe_number(value.get(field))
                        if safe is not None:
                            safe_values[field] = safe
                    roles[safe_name] = safe_values
            raw_acceptance = manifest.get("acceptance") if manifest is not None else None
            if isinstance(raw_acceptance, dict):
                for name, value in raw_acceptance.items():
                    if not isinstance(name, str):
                        continue
                    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:64]
                    if not safe_name:
                        continue
                    if isinstance(value, bool):
                        acceptance[safe_name] = value
                    elif (safe := self._safe_number(value)) is not None:
                        acceptance[safe_name] = safe
            marker = manifest_path or plan_path or lock_path
            modified = marker.stat().st_mtime if marker is not None else 0.0
            relative_root = task_root.relative_to(self.settings.project_root).as_posix()
            digest = hashlib.sha256(relative_root.encode()).hexdigest()[:16]
            progress = (
                {"percent": 100.0, "fraction": 1.0, "eta_seconds": 0.0}
                if state == "completed"
                else {}
            )
            summary = {
                "key": f"task:evaluation:{digest}",
                "kind": "evaluation",
                "label": f"{task_root.name} evaluation",
                "state": state,
                "active": state == "running",
                "gpu_relevant": bool(state == "running" and evaluation.get("gpu_relevant") is True),
                "phase": (
                    "evaluation"
                    if state == "running"
                    else ("paused" if state == "paused" else "complete")
                ),
                "progress": progress,
                "detail": " ".join(str(evaluation.get("detail", "")).split())[:1000],
                "updated_at": datetime.fromtimestamp(modified, UTC).isoformat(),
                "updated_at_unix": modified,
                "source": source,
                "profile_id": None,
                "control": {"available": False},
            }
            tasks.append(
                _DashboardTask(
                    summary=summary,
                    task_data={
                        "kind": "evaluation",
                        "phase": summary["phase"],
                        "progress": progress,
                        "roles": roles,
                        "acceptance": acceptance,
                        "source": source,
                    },
                    root=task_root,
                    console_path=console_path,
                )
            )
        return tasks

    def _task_records(self) -> list[_DashboardTask]:
        operations = self.operations_status()
        tasks = self._training_tasks()
        tasks.extend(self._data_tasks(operations))
        tasks.extend(self._evaluation_tasks(operations))
        return sorted(tasks, key=self._task_sort_key)

    def task_catalog(self) -> list[dict[str, Any]]:
        """List running, resumable, and successfully completed dashboard tasks."""

        return [dict(task.summary) for task in self._task_records()]

    def task_selection(self) -> dict[str, Any]:
        tasks = self._task_records()
        active, default = self._task_keys(tasks)
        return {
            "tasks": [dict(task.summary) for task in tasks],
            "active_task_key": active,
            "default_task_key": default,
        }

    @staticmethod
    def _task_keys(tasks: list[_DashboardTask]) -> tuple[str | None, str | None]:
        active = next(
            (str(task.summary["key"]) for task in tasks if task.summary["active"]),
            None,
        )
        default = active or (str(tasks[0].summary["key"]) if tasks else None)
        return active, default

    def snapshot(self, task_key: str = "", *, limit: int = 1200) -> dict[str, Any]:
        tasks = self._task_records()
        active_task_key, default_task_key = self._task_keys(tasks)
        selected_key = task_key or default_task_key
        selected = next(
            (task for task in tasks if task.summary["key"] == selected_key),
            None,
        )
        limit = max(10, min(int(limit), 5000))
        status: dict[str, Any] | None = None
        public_task: dict[str, Any] | None = None
        task_data: dict[str, Any] = {}
        metrics: list[dict[str, Any]] = []
        telemetry: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        console = ""
        if selected is not None:
            public_task = dict(selected.summary)
            task_data = dict(selected.task_data)
            if selected.run_key is not None:
                run_dir, status, _ = self._resolve_run(selected.run_key)
                metrics = self._run_records(run_dir, "metrics.jsonl", limit=limit)
                telemetry = self._run_records(run_dir, "telemetry.jsonl", limit=limit)
                events = self._run_records(run_dir, "events.jsonl", limit=200)
            else:
                status = public_task
            if selected.console_path is not None:
                console = read_console_tail(selected.console_path)
        elif isinstance(selected_key, str) and selected_key.startswith(("profile:", "history:")):
            # Preserve old direct-run URLs without adding stale or not-started
            # entries to the task selector.
            run_dir, status, _ = self._resolve_run(selected_key)
            metrics = self._run_records(run_dir, "metrics.jsonl", limit=limit)
            telemetry = self._run_records(run_dir, "telemetry.jsonl", limit=limit)
            events = self._run_records(run_dir, "events.jsonl", limit=200)
            console_path = self._run_file(run_dir, "console.log")
            console = read_console_tail(console_path) if console_path is not None else ""
            task_data = {"kind": "training", "legacy_direct_run": True}
        elif selected_key is None:
            raise DashboardError("no running, resumable, or completed dashboard tasks were found")
        else:
            raise DashboardError(f"unknown task catalog key: {selected_key!r}")

        gpu_task = next(
            (
                task.summary
                for task in tasks
                if task.summary.get("active") is True and task.summary.get("gpu_relevant") is True
            ),
            None,
        )
        gpu_task_key = str(gpu_task["key"]) if gpu_task is not None else None
        gpu_telemetry = dict(self.gpu_monitor.snapshot())
        gpu_telemetry.update(
            {
                "associated_task_key": gpu_task_key,
                "associated_task_kind": gpu_task.get("kind") if gpu_task else None,
                "associated_task_label": gpu_task.get("label") if gpu_task else None,
            }
        )
        live_gpu_relevant = bool(
            public_task is not None
            and public_task.get("active") is True
            and public_task.get("gpu_relevant") is True
            and public_task.get("key") == gpu_task_key
        )
        return {
            "server_time_utc": _utc_now(),
            **self.configuration_status(),
            "task": public_task,
            "task_data": task_data,
            "active_task_key": active_task_key,
            "default_task_key": default_task_key,
            "live_gpu_relevant": live_gpu_relevant,
            "status": status,
            "gpu_telemetry": gpu_telemetry,
            "metrics": metrics,
            "telemetry": telemetry,
            "events": events,
            "console": console,
            "tasks": [dict(task.summary) for task in tasks],
            "runs": self.run_catalog(),
            "operations": self.operations_status(),
        }

    def public_profiles(self) -> list[dict[str, Any]]:
        return [self.profile_status(profile) for profile in self.settings.profiles]

    def _record_action(
        self,
        *,
        action: str,
        profile: LaunchProfile,
        outcome: str,
        fields: Mapping[str, Any] | None = None,
    ) -> None:
        _append_jsonl(
            self._audit_path,
            {
                "timestamp_utc": _utc_now(),
                "action": action,
                "profile_id": profile.profile_id,
                "run_id": profile.run_id,
                "outcome": outcome,
                **dict(fields or {}),
            },
        )

    def start(self, profile_id: str, confirmation: str) -> dict[str, Any]:
        profile = self.settings.profile(profile_id)
        if profile.launch_kind != "direct_train":
            raise DashboardError(
                f"profile {profile.profile_id!r} must use the governed start route"
            )
        return self._start_profile(profile, confirmation)

    def start_governed(self, profile_id: str, confirmation: str) -> dict[str, Any]:
        profile = self.settings.profile(profile_id)
        if profile.launch_kind != "governed_v4":
            raise DashboardError(
                f"profile {profile.profile_id!r} is not a governed v4 profile"
            )
        # This first authorization is deliberately outside every action lock.
        # A blocked request must not create a lock artifact or touch run state.
        initial_plan = self._authorized_governed_plan(profile, confirmation)
        return self._start_profile(
            profile,
            confirmation,
            initial_governed_plan=initial_plan,
        )

    def _start_profile(
        self,
        profile: LaunchProfile,
        confirmation: str,
        *,
        initial_governed_plan: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._serialized_action():
            if self.configuration_status()["configuration_stale"]:
                raise DashboardError(
                    "dashboard configuration changed after startup; "
                    "review and restart the dashboard before starting training"
                )
            if not profile.launch_enabled:
                raise DashboardError(f"launch is disabled for profile {profile.profile_id!r}")
            governed_plan: dict[str, Any] | None = None
            if profile.launch_kind == "governed_v4":
                if initial_governed_plan is None:
                    raise DashboardError("governed launch lacks its lock-free authorization")
                governed_plan = self._authorized_governed_plan(profile, confirmation)
                if governed_plan != dict(initial_governed_plan):
                    raise DashboardError(
                        "governed plan changed between lock-free and serialized admission"
                    )
            elif confirmation != profile.start_confirmation:
                raise DashboardError("start confirmation text does not match")
            config_path = self._fixed_profile_path(
                profile.config_path,
                label=f"profile {profile.profile_id!r} config",
            )
            if not config_path.is_file():
                raise DashboardError(f"allowlisted training config is missing: {config_path}")
            current_config_sha = sha256_file(config_path)
            if not secrets.compare_digest(current_config_sha, profile.config_sha256):
                raise DashboardError(
                    "allowlisted training config changed after dashboard startup; review and restart the dashboard"
                )
            run_dir = self._fixed_profile_path(
                profile.run_dir,
                label=f"profile {profile.profile_id!r} run directory",
            )
            active = self.active_profile(
                persist_controller_exit=profile.launch_kind != "governed_v4"
            )
            if active is not None:
                raise DashboardError(
                    f"training profile {active.profile_id!r} is already active; duplicate launch refused"
                )
            for key, history_run_dir, session in self._discovered_runs():
                history = self._history_status(key, history_run_dir, session)
                if history["active"]:
                    raise DashboardError(
                        f"unmanaged training run {history['run_id']!r} is already active; duplicate launch refused"
                    )
            launch = (
                _TrainingLaunch(
                    mode="governed",
                    resume="controller",
                    fork_from=None,
                )
                if profile.launch_kind == "governed_v4"
                else self._fixed_initial_fork_launch(profile, run_dir)
            )
            if launch.fork_from is not None:
                fork_from = self._fixed_profile_path(
                    launch.fork_from,
                    label=f"profile {profile.profile_id!r} fork checkpoint",
                )
                if not fork_from.is_dir():
                    raise DashboardError(f"allowlisted fork checkpoint is missing: {fork_from}")
            if profile.launch_kind == "direct_train" and launch.resume not in {"auto", "none"}:
                resume_path = self._fixed_profile_path(
                    Path(launch.resume),
                    label=f"profile {profile.profile_id!r} resume checkpoint",
                )
                if not resume_path.is_dir():
                    raise DashboardError(f"allowlisted resume checkpoint is missing: {resume_path}")

            governed_snapshot: _GovernedExecutionSnapshot | None = None
            if governed_plan is not None:
                governed_snapshot = _build_governed_execution_snapshot(
                    profile,
                    governed_plan,
                )
            process: Any | None = None
            process_pid: int | None = None
            command_digest: str | None = None
            try:
                if self.configuration_status()["configuration_stale"]:
                    raise DashboardError(
                        "dashboard configuration changed during launch admission; "
                        "review and restart the dashboard before starting training"
                    )
                if governed_plan is not None:
                    final_plan = self._authorized_governed_plan(profile, confirmation)
                    if final_plan != governed_plan:
                        raise DashboardError(
                            "governed plan changed after execution snapshot admission"
                        )
                    governed_plan = final_plan

                # All governed authorization, duplicate, config, source, and
                # dependency checks are now complete.  Filesystem mutation
                # starts below this line.
                run_dir.mkdir(parents=True, exist_ok=True)
                run_dir = self._fixed_profile_path(
                    profile.run_dir,
                    label=f"profile {profile.profile_id!r} run directory",
                )
                command = self._command(
                    profile,
                    launch,
                    acknowledgement=confirmation if governed_plan is not None else None,
                    governed_snapshot=governed_snapshot,
                )
                command_digest = hashlib.sha256("\0".join(command).encode()).hexdigest()
                expected_console_path = run_dir / "console.log"
                console_path = _inside(
                    self.settings.project_root,
                    expected_console_path,
                    label=f"profile {profile.profile_id!r} console log",
                )
                if console_path != expected_console_path:
                    raise DashboardError(
                        f"profile {profile.profile_id!r} console log was redirected by a symlink"
                    )
                environment = os.environ.copy()
                environment["PYTHONUNBUFFERED"] = "1"
                process_kwargs: dict[str, Any] = {
                    "cwd": self.settings.project_root,
                    "env": environment,
                    "stdin": subprocess.DEVNULL,
                    "stderr": subprocess.STDOUT,
                    "start_new_session": True,
                    "close_fds": True,
                }
                if governed_snapshot is not None:
                    environment["PYTHONPATH"] = governed_snapshot.runtime_source_path
                    environment["PYTHONSAFEPATH"] = "1"
                    process_kwargs["stdin"] = governed_snapshot.source_fd
                    process_kwargs["pass_fds"] = governed_snapshot.pass_fds
                with console_path.open("ab", buffering=0) as console:
                    process_kwargs["stdout"] = console
                    process = self._process_factory(command, **process_kwargs)

                process_pid = getattr(process, "pid", None)
                if (
                    not isinstance(process_pid, int)
                    or isinstance(process_pid, bool)
                    or process_pid <= 1
                ):
                    raise DashboardError("launched process returned an invalid PID")
                process_identity: _LinuxProcessIdentity | None = None
                if governed_snapshot is not None:
                    process_identity = _read_linux_process_identity(process_pid)
                    if (
                        process_identity.pid != process_pid
                        or list(process_identity.command) != command
                        or process_identity.cwd != self.settings.project_root
                        or process_identity.executable != Path(sys.executable).resolve()
                        or process_identity.process_group_id != process_pid
                        or process_identity.process_group_id == os.getpgrp()
                    ):
                        raise DashboardError(
                            "spawned governed controller identity differs from the admitted command"
                        )

                state = {
                    "schema_version": 1,
                    "profile_id": profile.profile_id,
                    "run_id": profile.run_id,
                    "pid": process_pid,
                    "process_group_id": (
                        process_identity.process_group_id
                        if process_identity is not None
                        else process_pid
                    ),
                    "status": "launching",
                    "started_at_utc": _utc_now(),
                    "updated_at_utc": _utc_now(),
                    "command_sha256": command_digest,
                    "launch_kind": profile.launch_kind,
                    "launch_mode": launch.mode,
                    "effective_resume": launch.resume,
                    "effective_fork_from": (
                        str(launch.fork_from) if launch.fork_from is not None else None
                    ),
                    "verified_checkpoint_id": launch.verified_checkpoint_id,
                    "governed_plan_id": (
                        governed_plan.get("plan_id") if governed_plan is not None else None
                    ),
                    "governed_state_path": (
                        str(profile.governed_state_path)
                        if profile.governed_state_path is not None
                        else None
                    ),
                    "resume_compatibility_gate": (
                        "training_entrypoint_preflight_and_checkpoint_loader"
                        if launch.mode == "resume_auto"
                        else None
                    ),
                    "project_root": str(self.settings.project_root),
                    "process_start_time_ticks": (
                        process_identity.start_time_ticks
                        if process_identity is not None
                        else None
                    ),
                    "process_cmdline": (
                        list(process_identity.command)
                        if process_identity is not None
                        else None
                    ),
                    "process_executable": (
                        str(process_identity.executable)
                        if process_identity is not None
                        else None
                    ),
                    "controller_snapshot_fd": (
                        governed_snapshot.controller_fd
                        if governed_snapshot is not None
                        else None
                    ),
                    "source_snapshot_fd": (
                        governed_snapshot.runtime_source_fd
                        if governed_snapshot is not None
                        else None
                    ),
                    "source_snapshot_transport_fd": (
                        governed_snapshot.source_fd
                        if governed_snapshot is not None
                        else None
                    ),
                    "controller_snapshot_sha256": (
                        governed_snapshot.controller_sha256
                        if governed_snapshot is not None
                        else None
                    ),
                    "source_snapshot_sha256": (
                        governed_snapshot.source_archive_sha256
                        if governed_snapshot is not None
                        else None
                    ),
                    "source_tree_sha256": (
                        governed_snapshot.source_tree_sha256
                        if governed_snapshot is not None
                        else None
                    ),
                    "dependency_lock_sha256": (
                        governed_snapshot.dependency_lock_sha256
                        if governed_snapshot is not None
                        else None
                    ),
                }
                _atomic_json(self._state_path, state)
                self._record_action(
                    action="start",
                    profile=profile,
                    outcome="accepted",
                    fields={
                        "pid": process_pid,
                        "process_group_id": state["process_group_id"],
                        "command_sha256": command_digest,
                        "launch_kind": profile.launch_kind,
                        "launch_mode": launch.mode,
                        "effective_resume": launch.resume,
                        "effective_fork_from": (
                            str(launch.fork_from)
                            if launch.fork_from is not None
                            else None
                        ),
                        "verified_checkpoint_id": launch.verified_checkpoint_id,
                        "governed_plan_id": (
                            governed_plan.get("plan_id")
                            if governed_plan is not None
                            else None
                        ),
                        "resume_compatibility_gate": (
                            "training_entrypoint_preflight_and_checkpoint_loader"
                            if launch.mode == "resume_auto"
                            else None
                        ),
                    },
                )
                return state
            except BaseException:
                if process is not None:
                    _reap_failed_launch(process)
                    failed_state = _read_json_regular_no_follow(self._state_path)
                    if (
                        failed_state is not None
                        and failed_state.get("profile_id") == profile.profile_id
                        and failed_state.get("pid") == process_pid
                        and failed_state.get("command_sha256") == command_digest
                    ):
                        with suppress(OSError):
                            self._state_path.unlink()
                raise
            finally:
                if governed_snapshot is not None:
                    governed_snapshot.close()

    def signal(self, profile_id: str, action: str, confirmation: str) -> dict[str, Any]:
        if action not in {"save", "stop"}:
            raise DashboardError(f"unsupported signal action: {action!r}")
        profile = self.settings.profile(profile_id)
        expected = profile.save_confirmation if action == "save" else profile.stop_confirmation
        with self._serialized_action():
            if confirmation != expected:
                raise DashboardError(f"{action} confirmation text does not match")
            session, active = (
                (None, False)
                if profile.launch_kind == "governed_v4"
                else self._session(profile)
            )
            state: dict[str, Any] | None = None
            signal_target = "process"
            process_group_id: int | None = None
            if profile.launch_kind == "governed_v4" and action == "stop":
                controller_state = self._controller_state(persist_exit=False)
                pid, process_group_id = self._verified_governed_process_group(
                    profile,
                    controller_state,
                )
                state = dict(controller_state or {})
                signal_target = "process_group"
            elif profile.launch_kind == "governed_v4":
                controller_state = self._controller_state(persist_exit=False)
                pid = self._verified_rank0_training_pid(profile, controller_state)
                state = dict(controller_state or {})
            elif active and session is not None:
                pid = int(session["pid"])
            elif action == "stop":
                controller_state = self._controller_state()
                controller_pid = (controller_state or {}).get("pid")
                controller_matches = bool(
                    controller_state is not None
                    and controller_state.get("profile_id") == profile.profile_id
                    and controller_state.get("status") in {"launching", "running"}
                    and isinstance(controller_pid, int)
                    and not isinstance(controller_pid, bool)
                    and _process_matches_profile(controller_pid, profile)
                )
                if not controller_matches:
                    raise DashboardError(
                        f"profile {profile.profile_id!r} has no verified active training PID"
                    )
                pid = int(controller_pid)
                state = dict(controller_state)
            else:
                raise DashboardError(
                    f"profile {profile.profile_id!r} has no verified active rank-0 PID"
                )
            requested_signal = signal.SIGUSR1 if action == "save" else signal.SIGTERM
            if process_group_id is None:
                os.kill(pid, requested_signal)
            else:
                os.killpg(process_group_id, requested_signal)
            state = (
                state
                or self._controller_state()
                or {
                    "schema_version": 1,
                    "profile_id": profile.profile_id,
                    "run_id": profile.run_id,
                    "pid": pid,
                    "started_at_utc": session.get("started_at_utc") if session else None,
                }
            )
            state.update(
                {
                    "status": "stop_requested" if action == "stop" else "running",
                    "updated_at_utc": _utc_now(),
                    "last_signal": requested_signal.name,
                    "last_signal_target": signal_target,
                }
            )
            _atomic_json(self._state_path, state)
            self._record_action(
                action=action,
                profile=profile,
                outcome="accepted",
                fields={
                    "pid": pid,
                    "process_group_id": process_group_id,
                    "signal": requested_signal.name,
                    "signal_target": signal_target,
                },
            )
            return {
                "ok": True,
                "action": action,
                "profile_id": profile.profile_id,
                "pid": pid,
                "process_group_id": process_group_id,
                "signal": requested_signal.name,
                "signal_target": signal_target,
            }


class DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        controller: DashboardController,
        *,
        csrf_token: str | None = None,
        auth: DashboardAuth | None = None,
        public_bind: bool = False,
    ) -> None:
        self.controller = controller
        self.csrf_token = csrf_token or secrets.token_urlsafe(32)
        self.auth = auth
        self.public_bind = public_bind
        self._access_log_lock = threading.Lock()
        self._access_log_last: dict[tuple[str, str, str], float] = {}
        super().__init__(address, DashboardRequestHandler)

    def should_log_access(self, client: str, request: str, status: str) -> bool:
        """Rate-limit identical polling/error lines while retaining periodic evidence."""

        key = (client, request, status)
        now = time.monotonic()
        with self._access_log_lock:
            previous = self._access_log_last.get(key)
            if previous is not None and now - previous < 60.0:
                return False
            self._access_log_last[key] = now
        return True


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server: DashboardHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        request = str(args[0]) if args else format
        status = str(args[1]) if len(args) > 1 else "unknown"
        if self.server.should_log_access(self.client_address[0], request, status):
            super().log_message(format, *args)

    def _authenticated(self) -> bool:
        auth = self.server.auth
        if auth is None:
            return True
        return secrets.compare_digest(
            self.headers.get("Authorization", ""),
            auth.authorization_header,
        )

    def _is_local_authority(self, authority: str) -> bool:
        try:
            hostname = urlsplit(f"//{authority}").hostname
        except ValueError:
            return False
        if hostname is None:
            return False
        if hostname.lower() == "localhost":
            return True
        with suppress(ValueError):
            return ipaddress.ip_address(hostname).is_loopback
        return False

    def _request_origin_is_local(self) -> bool:
        if not self._is_local_authority(self.headers.get("Host", "")):
            return False
        origin = self.headers.get("Origin")
        if not origin:
            return True
        try:
            parsed = urlsplit(origin)
        except ValueError:
            return False
        return parsed.scheme in {"http", "https"} and self._is_local_authority(parsed.netloc)

    def _request_origin_matches_host(self) -> bool:
        """Require same-origin controls while allowing an authenticated LAN host."""

        authority = self.headers.get("Host", "")
        try:
            host = urlsplit(f"//{authority}")
        except ValueError:
            return False
        if host.hostname is None:
            return False
        origin = self.headers.get("Origin")
        if not origin:
            return True
        try:
            parsed = urlsplit(origin)
        except ValueError:
            return False
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            return False
        try:
            ports_match = parsed.port == host.port
        except ValueError:
            return False
        return parsed.hostname.casefold() == host.hostname.casefold() and ports_match

    def _request_origin_allowed(self) -> bool:
        if self.server.public_bind:
            return self._request_origin_matches_host()
        return self._request_origin_is_local()

    def _headers(
        self,
        status: HTTPStatus,
        content_type: str,
        length: int,
        *,
        extra: Mapping[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
        )
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()

    def _json(self, status: HTTPStatus, value: Mapping[str, Any]) -> None:
        payload = json.dumps(
            dict(value), ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(payload))
        self.wfile.write(payload)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"ok": False, "error": message})

    def _unauthorized(self) -> None:
        payload = b'{"ok":false,"error":"authentication required"}'
        self._headers(
            HTTPStatus.UNAUTHORIZED,
            "application/json; charset=utf-8",
            len(payload),
            extra={"WWW-Authenticate": 'Basic realm="Twen dashboard", charset="UTF-8"'},
        )
        self.wfile.write(payload)

    def _read_body(self) -> dict[str, Any]:
        if self.headers.get_content_type() != "application/json":
            raise DashboardError("POST requests require application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise DashboardError("invalid Content-Length") from error
        if length <= 0 or length > 16 * 1024:
            raise DashboardError("request body must be between 1 and 16384 bytes")
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DashboardError("request body is not valid JSON") from error
        if not isinstance(value, dict):
            raise DashboardError("request body must be a JSON object")
        return value

    def do_GET(self) -> None:
        if not self._authenticated():
            self._unauthorized()
            return
        if not self._request_origin_allowed():
            self._error(HTTPStatus.FORBIDDEN, "invalid Host/Origin refused")
            return
        parsed = urlsplit(self.path)
        try:
            if parsed.path == "/":
                payload = files("twen").joinpath("web_static/index.html").read_bytes()
                self._headers(HTTPStatus.OK, "text/html; charset=utf-8", len(payload))
                self.wfile.write(payload)
                return
            if parsed.path == "/api/bootstrap":
                task_selection = self.server.controller.task_selection()
                self._json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "csrf_token": self.server.csrf_token,
                        **self.server.controller.configuration_status(),
                        **task_selection,
                        "profiles": self.server.controller.public_profiles(),
                        "runs": self.server.controller.run_catalog(),
                        "operations": self.server.controller.operations_status(),
                        "poll_interval_ms": 1000,
                        "control_policy": {
                            "automatic_launch": False,
                            "server_side_allowlist": True,
                            "preflight_mandatory": True,
                            "formal_v4_governed_route": "/api/governed/start",
                            "formal_v4_direct_train_refused": True,
                            "formal_v4_exact_plan_ack_required": True,
                            "csrf_required": True,
                            "authentication_required": self.server.auth is not None,
                        },
                    },
                )
                return
            if parsed.path == "/api/snapshot":
                query = parse_qs(parsed.query)
                task_key = query.get("task", query.get("run", [""]))[0]
                try:
                    limit = int(query.get("limit", ["1200"])[0])
                except ValueError as error:
                    raise DashboardError("limit must be an integer") from error
                self._json(
                    HTTPStatus.OK,
                    {"ok": True, **self.server.controller.snapshot(task_key, limit=limit)},
                )
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")
        except DashboardError as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
        except Exception as error:  # pragma: no cover - last-resort server boundary
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"internal error: {type(error).__name__}")

    def do_POST(self) -> None:
        if not self._authenticated():
            self._unauthorized()
            return
        if not self._request_origin_allowed():
            self._error(HTTPStatus.FORBIDDEN, "invalid Host/Origin refused")
            return
        parsed = urlsplit(self.path)
        if parsed.path not in {"/api/start", "/api/governed/start", "/api/signal"}:
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        if not secrets.compare_digest(self.headers.get("X-Twen-CSRF", ""), self.server.csrf_token):
            self._error(HTTPStatus.FORBIDDEN, "invalid CSRF token")
            return
        try:
            body = self._read_body()
            profile_id = body.get("profile_id")
            confirmation = body.get("confirmation")
            if not isinstance(profile_id, str) or not isinstance(confirmation, str):
                raise DashboardError("profile_id and confirmation must be strings")
            if parsed.path == "/api/start":
                result = self.server.controller.start(profile_id, confirmation)
            elif parsed.path == "/api/governed/start":
                result = self.server.controller.start_governed(profile_id, confirmation)
            else:
                action = body.get("action")
                if not isinstance(action, str):
                    raise DashboardError("action must be a string")
                result = self.server.controller.signal(profile_id, action, confirmation)
            self._json(HTTPStatus.ACCEPTED, {"ok": True, "result": result})
        except DashboardError as error:
            self._error(HTTPStatus.CONFLICT, str(error))
        except OSError as error:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"operating-system error: {error}")


def create_dashboard_server(
    settings: DashboardSettings,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    controller: DashboardController | None = None,
    csrf_token: str | None = None,
    auth: DashboardAuth | None = None,
) -> DashboardHTTPServer:
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise DashboardError("dashboard host must be a literal IP address") from error
    public_bind = not address.is_loopback
    if public_bind and auth is None:
        raise DashboardError("dashboard requires --auth-file for every non-loopback bind")
    if isinstance(port, bool) or not 0 <= int(port) <= 65535:
        raise DashboardError("dashboard port must be in 0..65535")
    return DashboardHTTPServer(
        (host, int(port)),
        controller or DashboardController(settings),
        csrf_token=csrf_token,
        auth=auth,
        public_bind=public_bind,
    )


def serve_dashboard(
    config_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    auth_file: str | Path | None = None,
) -> None:
    settings = load_dashboard_settings(config_path)
    auth = load_dashboard_auth(auth_file) if auth_file is not None else None
    server = create_dashboard_server(settings, host=host, port=port, auth=auth)
    telemetry_stop = threading.Event()
    telemetry_thread = threading.Thread(
        target=server.controller.gpu_monitor.run_until_stopped,
        args=(telemetry_stop,),
        name="twen-gpu-telemetry",
        daemon=True,
    )
    telemetry_started = False
    previous_sigterm_handler: Any = None
    sigterm_handler_installed = False
    termination_requested = False

    def request_termination(_signum: int, _frame: Any) -> None:
        nonlocal termination_requested
        if termination_requested:
            return
        termination_requested = True
        raise _DashboardTermination

    try:
        if threading.current_thread() is threading.main_thread():
            previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, request_termination)
            sigterm_handler_installed = True
        telemetry_thread.start()
        telemetry_started = True
        print(
            json.dumps(
                {
                    "event": "dashboard_start",
                    "url": f"http://{host}:{server.server_address[1]}",
                    "profiles": [profile.profile_id for profile in settings.profiles],
                    "authentication": "http-basic" if auth is not None else "none-loopback-only",
                    "auth_username": auth.username if auth is not None else None,
                    "training_started": False,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        server.serve_forever(poll_interval=0.5)
    except _DashboardTermination:
        pass
    finally:
        telemetry_stop.set()
        if telemetry_started:
            telemetry_thread.join(timeout=_GPU_TELEMETRY_OUTPUT_TIMEOUT_SECONDS + 1.0)
        server.server_close()
        if sigterm_handler_installed:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)


__all__ = [
    "DashboardAuth",
    "DashboardController",
    "DashboardError",
    "DashboardSettings",
    "GpuTelemetryMonitor",
    "JsonlTailCache",
    "LaunchProfile",
    "create_dashboard_server",
    "ensure_dashboard_auth_file",
    "load_dashboard_auth",
    "load_dashboard_settings",
    "read_console_tail",
    "serve_dashboard",
]
