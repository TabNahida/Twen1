"""Local, dependency-free training dashboard and guarded process controller.

The dashboard intentionally stays outside the training process.  It tails the
durable JSON/JSONL files already written by rank zero and therefore introduces
no CUDA work, profiler hooks, or synchronization into the hot path.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import math
import os
import platform
import re
import secrets
import select
import signal
import subprocess
import sys
import threading
import time
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

from .config import load_train_config
from .io.locking import FileLock, FileLockTimeout
from .runtime.checkpoint import CheckpointManager
from .utils import sha256_file


class DashboardError(RuntimeError):
    """Raised for invalid dashboard configuration or guarded actions."""


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
_ACTIVE_TRAINING_STATES = frozenset(("running", "launching", "stop_requested"))
_COMPLETED_TRAINING_STATES = frozenset(("complete", "completed", "already_complete"))
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

    @property
    def start_confirmation(self) -> str:
        return f"START {self.profile_id}"

    @property
    def stop_confirmation(self) -> str:
        return f"STOP {self.run_id}"

    @property
    def save_confirmation(self) -> str:
        return f"SAVE {self.run_id}"


@dataclass(frozen=True, slots=True)
class DashboardSettings:
    project_root: Path
    state_dir: Path
    profiles: tuple[LaunchProfile, ...]

    def profile(self, profile_id: str) -> LaunchProfile:
        for profile in self.profiles:
            if profile.profile_id == profile_id:
                return profile
        raise DashboardError(f"unknown launch profile: {profile_id!r}")


def load_dashboard_settings(path: str | Path) -> DashboardSettings:
    """Load and fully resolve the dashboard's fixed launch allowlist."""

    source = Path(path).resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
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
        declared_config_sha256 = item.get("config_sha256")
        if declared_config_sha256 is None:
            if launch_enabled:
                raise DashboardError(
                    f"profiles[{index}].config_sha256 is required when launch_enabled=true"
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
        train_config = load_train_config(config_path)
        run_dir = _inside(
            project_root,
            project_root / train_config.checkpoint.output_dir,
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
        profiles.append(
            LaunchProfile(
                profile_id=profile_id,
                label=label.strip(),
                config_path=config_path,
                config_sha256=config_sha256,
                run_dir=run_dir,
                run_id=train_config.run_id,
                stage=train_config.stage,
                resume=resume,
                fork_from=fork_from,
                launch_enabled=launch_enabled,
            )
        )
    return DashboardSettings(
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


def _session_source_mix(session: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Expose the authenticated runtime mix without parsing a checkpoint."""

    if session is None:
        return None
    source_mix = session.get("source_mix")
    return dict(source_mix) if isinstance(source_mix, Mapping) else None


def _process_matches_profile(pid: int, profile: LaunchProfile) -> bool:
    """Defend against stale/reused PIDs before signaling a training process."""

    if not _pid_alive(pid):
        return False
    try:
        arguments = [
            item.decode("utf-8", errors="surrogateescape")
            for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if item
        ]
        cwd = Path(f"/proc/{pid}/cwd").resolve()
    except OSError:
        return False
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


def _process_is_twen_training(pid: int) -> bool:
    """Verify an unmanaged PID is still some Twen training command."""

    if not _pid_alive(pid):
        return False
    try:
        arguments = [
            item.decode("utf-8", errors="surrogateescape")
            for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if item
        ]
    except OSError:
        return False
    markers = {"twen", "twen.cli", "twen.__main__"}
    return "train" in arguments and any(
        Path(argument).name in markers or argument in markers for argument in arguments
    )


ProcessFactory = Callable[..., subprocess.Popen[bytes]]


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
        self._action_lock_path = settings.state_dir / "controller-action.lock"
        self._state_path = settings.state_dir / "controller-state.json"
        self._audit_path = settings.state_dir / "actions.jsonl"
        self._operations_cache: tuple[float, dict[str, Any]] | None = None
        settings.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    @contextmanager
    def _serialized_action(self) -> Any:
        """Serialize mutations across threads and independently started servers."""

        with self._action_lock:
            try:
                with FileLock(
                    self._action_lock_path,
                    timeout_seconds=_DASHBOARD_ACTION_LOCK_TIMEOUT_SECONDS,
                    poll_seconds=0.05,
                ):
                    yield
            except FileLockTimeout as error:
                raise DashboardError(
                    "another dashboard process is handling a training control action"
                ) from error

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

    def _command(
        self,
        profile: LaunchProfile,
        launch: _TrainingLaunch,
    ) -> list[str]:
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
        active = bool(
            claims_running
            and valid_pid
            and hostname_matches
            and _process_matches_profile(pid, profile)
        )
        result = dict(session)
        if claims_running and not active:
            result["status"] = "stale"
            result["stale_reason"] = "rank-0 PID is absent or no longer matches this profile"
        return result, active

    def _controller_state(self) -> dict[str, Any] | None:
        state = _read_json(self._state_path)
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
        } and not _process_matches_profile(pid, profile):
            state = {
                **state,
                "status": "exited",
                "updated_at_utc": _utc_now(),
            }
            _atomic_json(self._state_path, state)
        return state

    def active_profile(self) -> LaunchProfile | None:
        for profile in self.settings.profiles:
            _, active = self._session(profile)
            if active:
                return profile
        state = self._controller_state()
        if state and state.get("status") in {"launching", "running", "stop_requested"}:
            try:
                return self.settings.profile(str(state["profile_id"]))
            except DashboardError:
                pass
        return None

    def profile_status(self, profile: LaunchProfile) -> dict[str, Any]:
        session, active = self._session(profile)
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
        effectively_active = active or state in _ACTIVE_TRAINING_STATES
        start_available = bool(
            profile.launch_enabled
            and not effectively_active
            and state not in _COMPLETED_TRAINING_STATES
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
            "config_sha256": profile.config_sha256,
            "state": state,
            "active": effectively_active,
            "launch_enabled": profile.launch_enabled,
            # This is deliberately only a cheap UI/status hint.  It must never
            # inspect or trust checkpoint contents during polling; start()
            # performs the full hash/compatibility admission when clicked.
            "start_available": start_available,
            "start_action": start_action,
            "preflight_enforced": True,
            "start_confirmation": profile.start_confirmation,
            "stop_confirmation": profile.stop_confirmation,
            "save_confirmation": profile.save_confirmation,
            "session": session,
            "source_mix": _session_source_mix(session),
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
            return self._operations_cache[1]
        evaluations: list[dict[str, Any]] = []
        plan_paths: set[Path] = set()
        for evaluation_root_name in ("eval", "artifacts/evaluations"):
            evaluation_root = self.settings.project_root / evaluation_root_name
            if evaluation_root.is_dir():
                plan_paths.update(evaluation_root.rglob("PLAN.json"))
        for raw_plan_path in sorted(plan_paths)[:200]:
            plan_path = _existing_file_inside(self.settings.project_root, raw_plan_path)
            if plan_path is None:
                continue
            output = plan_path.parent
            manifest_path = output / "manifest.json"
            complete_path = output / "COMPLETE"
            manifest_file = _existing_file_inside(output, manifest_path)
            complete_file = _existing_file_inside(output, complete_path)
            status = (
                "complete"
                if manifest_file is not None and complete_file is not None
                else "in_progress"
            )
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
            evaluations.append(
                {
                    "name": output.relative_to(self.settings.project_root).as_posix(),
                    "status": status,
                    "detail": " · ".join(detail_parts),
                    "modified_at": datetime.fromtimestamp(
                        plan_path.stat().st_mtime, UTC
                    ).isoformat(),
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
            kind = "kd" if "kd" in marker or "-kd-" in task_root.name else "data_pipeline"
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
            marker = manifest_path or plan_path
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
                "phase": "evaluation" if state == "running" else "complete",
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

        active_task = next(
            (task.summary for task in tasks if task.summary["key"] == active_task_key),
            None,
        )
        gpu_telemetry = dict(self.gpu_monitor.snapshot())
        gpu_telemetry.update(
            {
                "associated_task_key": active_task_key,
                "associated_task_kind": active_task.get("kind") if active_task else None,
                "associated_task_label": active_task.get("label") if active_task else None,
            }
        )
        live_gpu_relevant = bool(
            public_task is not None
            and public_task.get("active") is True
            and public_task.get("key") == active_task_key
        )
        return {
            "server_time_utc": _utc_now(),
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
        with self._serialized_action():
            if not profile.launch_enabled:
                raise DashboardError(f"launch is disabled for profile {profile.profile_id!r}")
            if confirmation != profile.start_confirmation:
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
            active = self.active_profile()
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
            launch = self._fixed_initial_fork_launch(profile, run_dir)
            if launch.fork_from is not None:
                fork_from = self._fixed_profile_path(
                    launch.fork_from,
                    label=f"profile {profile.profile_id!r} fork checkpoint",
                )
                if not fork_from.is_dir():
                    raise DashboardError(f"allowlisted fork checkpoint is missing: {fork_from}")
            if launch.resume not in {"auto", "none"}:
                resume_path = self._fixed_profile_path(
                    Path(launch.resume),
                    label=f"profile {profile.profile_id!r} resume checkpoint",
                )
                if not resume_path.is_dir():
                    raise DashboardError(f"allowlisted resume checkpoint is missing: {resume_path}")
            run_dir.mkdir(parents=True, exist_ok=True)
            run_dir = self._fixed_profile_path(
                profile.run_dir,
                label=f"profile {profile.profile_id!r} run directory",
            )
            command = self._command(profile, launch)
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
            with console_path.open("ab", buffering=0) as console:
                process = self._process_factory(
                    command,
                    cwd=self.settings.project_root,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=console,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            state = {
                "schema_version": 1,
                "profile_id": profile.profile_id,
                "run_id": profile.run_id,
                "pid": int(process.pid),
                "status": "launching",
                "started_at_utc": _utc_now(),
                "updated_at_utc": _utc_now(),
                "command_sha256": command_digest,
                "launch_mode": launch.mode,
                "effective_resume": launch.resume,
                "effective_fork_from": (
                    str(launch.fork_from) if launch.fork_from is not None else None
                ),
                "verified_checkpoint_id": launch.verified_checkpoint_id,
                "resume_compatibility_gate": (
                    "training_entrypoint_preflight_and_checkpoint_loader"
                    if launch.mode == "resume_auto"
                    else None
                ),
            }
            _atomic_json(self._state_path, state)
            self._record_action(
                action="start",
                profile=profile,
                outcome="accepted",
                fields={
                    "pid": process.pid,
                    "command_sha256": command_digest,
                    "launch_mode": launch.mode,
                    "effective_resume": launch.resume,
                    "effective_fork_from": (
                        str(launch.fork_from) if launch.fork_from is not None else None
                    ),
                    "verified_checkpoint_id": launch.verified_checkpoint_id,
                    "resume_compatibility_gate": (
                        "training_entrypoint_preflight_and_checkpoint_loader"
                        if launch.mode == "resume_auto"
                        else None
                    ),
                },
            )
            return state

    def signal(self, profile_id: str, action: str, confirmation: str) -> dict[str, Any]:
        if action not in {"save", "stop"}:
            raise DashboardError(f"unsupported signal action: {action!r}")
        profile = self.settings.profile(profile_id)
        expected = profile.save_confirmation if action == "save" else profile.stop_confirmation
        with self._serialized_action():
            if confirmation != expected:
                raise DashboardError(f"{action} confirmation text does not match")
            session, active = self._session(profile)
            state: dict[str, Any] | None = None
            if active and session is not None:
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
            os.kill(pid, requested_signal)
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
                }
            )
            _atomic_json(self._state_path, state)
            self._record_action(
                action=action,
                profile=profile,
                outcome="accepted",
                fields={"pid": pid, "signal": requested_signal.name},
            )
            return {
                "ok": True,
                "action": action,
                "profile_id": profile.profile_id,
                "pid": pid,
                "signal": requested_signal.name,
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
                        **task_selection,
                        "profiles": self.server.controller.public_profiles(),
                        "runs": self.server.controller.run_catalog(),
                        "operations": self.server.controller.operations_status(),
                        "poll_interval_ms": 1000,
                        "control_policy": {
                            "automatic_launch": False,
                            "server_side_allowlist": True,
                            "preflight_mandatory": True,
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
        if parsed.path not in {"/api/start", "/api/signal"}:
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
