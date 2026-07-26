"""Crash-tolerant structured training logs and terminal progress."""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _serialize(record: Mapping[str, Any]) -> str:
    """Serialize strict JSON and reject NaN, infinity, and opaque objects."""

    return json.dumps(
        dict(record),
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def _append_fsynced(path: Path, payload: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _replace_fsynced(path: Path, payload: str) -> None:
    """Atomically replace a small JSON status file and sync its directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
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


def exception_fields(error: BaseException) -> dict[str, str]:
    """Return JSON-safe error details including the exception's full traceback."""

    return {
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    }


class RankZeroSessionFile:
    """Durable rank-zero process identity and lifecycle marker.

    The file remains in the run directory after exit. Operators can inspect the
    session id and ``running`` status before signaling this exact PID instead of
    matching a broad process command line.
    """

    _RESERVED = frozenset(
        {
            "schema_version",
            "session_id",
            "pid",
            "status",
            "started_at_utc",
            "updated_at_utc",
            "ended_at_utc",
        }
    )

    def __init__(
        self,
        path: str | Path,
        *,
        session_id: str,
        fields: Mapping[str, Any] | None = None,
    ) -> None:
        if not session_id.strip():
            raise ValueError("session_id must be non-empty")
        values = dict(fields or {})
        collision = self._RESERVED.intersection(values)
        if collision:
            raise ValueError(f"session fields override reserved names: {sorted(collision)}")
        now = utc_now()
        self.path = Path(path)
        self._lock = threading.Lock()
        self._record: dict[str, Any] = {
            "schema_version": 1,
            "session_id": session_id,
            "pid": os.getpid(),
            "status": "running",
            "started_at_utc": now,
            "updated_at_utc": now,
            "ended_at_utc": None,
            **values,
        }
        _replace_fsynced(self.path, _serialize(self._record))

    def finish(
        self,
        status: str,
        fields: Mapping[str, Any] | None = None,
    ) -> None:
        if not status.strip() or status == "running":
            raise ValueError("finished session status must be non-empty and not 'running'")
        values = dict(fields or {})
        collision = self._RESERVED.intersection(values)
        if collision:
            raise ValueError(f"session fields override reserved names: {sorted(collision)}")
        with self._lock:
            now = utc_now()
            self._record.update(values)
            self._record.update(
                {
                    "status": status,
                    "updated_at_utc": now,
                    "ended_at_utc": now,
                }
            )
            _replace_fsynced(self.path, _serialize(self._record))


class JsonlMetricLogger:
    """Append one fsync'd record per committed optimizer step.

    Existing steps are detected at open time so a resumed run never writes a
    second record for an already committed step.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_step = self._read_last_step()
        self._lock = threading.Lock()

    @property
    def last_step(self) -> int:
        return self._last_step

    def _read_last_step(self) -> int:
        if not self.path.exists():
            return -1
        last = -1
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                    last = max(last, int(record["step"]))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    # A torn final line is ignored; checkpoint state remains authoritative.
                    continue
        return last

    def log(self, step: int, metrics: Mapping[str, Any]) -> bool:
        if "step" in metrics:
            raise ValueError("metrics must not override the reserved 'step' field")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")
        record = {"step": int(step), **dict(metrics)}
        payload = _serialize(record)
        with self._lock:
            if step <= self._last_step:
                return False
            _append_fsynced(self.path, payload)
            self._last_step = step
            return True

    def reconcile(self, committed_step: int) -> None:
        """Canonicalize valid committed records and discard torn/future tails."""

        if committed_step < 0:
            raise ValueError("committed_step must be non-negative")
        with self._lock:
            if not self.path.exists():
                return
            retained: list[str] = []
            last = -1
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                        step = int(record["step"])
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue
                    if last < step <= committed_step:
                        try:
                            retained.append(_serialize(record) + "\n")
                        except (TypeError, ValueError):
                            continue
                        last = step
            temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                handle.writelines(retained)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            self._last_step = last


class JsonlEventLogger:
    """Append UTC-stamped lifecycle events independently of metric recovery."""

    _RESERVED = frozenset({"event", "timestamp_utc", "session_id"})

    def __init__(self, path: str | Path, *, session_id: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or uuid.uuid4().hex
        self._lock = threading.Lock()

    def log(self, event: str, fields: Mapping[str, Any] | None = None) -> None:
        if not event.strip():
            raise ValueError("event name must be non-empty")
        values = dict(fields or {})
        collision = self._RESERVED.intersection(values)
        if collision:
            raise ValueError(f"event fields override reserved names: {sorted(collision)}")
        payload = _serialize(
            {
                "event": event,
                "timestamp_utc": utc_now(),
                "session_id": self.session_id,
                **values,
            }
        )
        with self._lock:
            _append_fsynced(self.path, payload)


class ThroughputTracker:
    """Compute stable step throughput and ETA from committed token counts."""

    def __init__(
        self,
        *,
        total_tokens: int,
        initial_tokens: int = 0,
        ema_alpha: float = 0.2,
    ) -> None:
        if total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        if not 0 <= initial_tokens <= total_tokens:
            raise ValueError("initial_tokens must be within the run token budget")
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        self.total_tokens = int(total_tokens)
        self.last_tokens = int(initial_tokens)
        self.ema_alpha = float(ema_alpha)
        self.ema_tokens_per_second: float | None = None

    def observe(self, committed_tokens: int, step_seconds: float) -> dict[str, float | int | None]:
        if committed_tokens < self.last_tokens:
            raise ValueError("committed_tokens moved backwards")
        if not math.isfinite(step_seconds) or step_seconds <= 0:
            raise ValueError("step_seconds must be finite and positive")
        delta = int(committed_tokens) - self.last_tokens
        instant = delta / step_seconds
        if self.ema_tokens_per_second is None:
            self.ema_tokens_per_second = instant
        else:
            alpha = self.ema_alpha
            self.ema_tokens_per_second = alpha * instant + (1.0 - alpha) * self.ema_tokens_per_second
        self.last_tokens = int(committed_tokens)
        remaining = max(self.total_tokens - committed_tokens, 0)
        eta = (
            remaining / self.ema_tokens_per_second
            if self.ema_tokens_per_second and self.ema_tokens_per_second > 0
            else None
        )
        return {
            "step_seconds": float(step_seconds),
            "tokens_this_step": delta,
            "tokens_per_second": float(instant),
            "tokens_per_second_ema": float(self.ema_tokens_per_second),
            "progress_percent": min(committed_tokens / self.total_tokens * 100.0, 100.0),
            "eta_seconds": float(eta) if eta is not None else None,
        }


class TrainingTelemetryTracker:
    """Track compute-only and end-to-end wall-clock training throughput.

    The wall timer starts when this object is constructed, so callers should
    construct it after checkpoint restore and other session initialization.
    Checkpoint, metric logging, progress rendering, and replay overhead between
    observations are then naturally charged to wall-clock throughput. Existing
    unprefixed throughput fields remain aliases of the compute-only values.
    """

    def __init__(
        self,
        *,
        total_tokens: int,
        initial_tokens: int = 0,
        ema_alpha: float = 0.2,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._compute = ThroughputTracker(
            total_tokens=total_tokens,
            initial_tokens=initial_tokens,
            ema_alpha=ema_alpha,
        )
        self._wall = ThroughputTracker(
            total_tokens=total_tokens,
            initial_tokens=initial_tokens,
            ema_alpha=ema_alpha,
        )
        self._clock = clock
        self._wall_started = float(clock())
        if not math.isfinite(self._wall_started):
            raise ValueError("wall clock must return a finite value")

    def observe(
        self,
        committed_tokens: int,
        compute_step_seconds: float,
    ) -> dict[str, float | int | None]:
        now = float(self._clock())
        wall_clock_step_seconds = now - self._wall_started
        if not math.isfinite(wall_clock_step_seconds) or wall_clock_step_seconds <= 0:
            raise ValueError("wall-clock step duration must be finite and positive")
        compute = self._compute.observe(committed_tokens, compute_step_seconds)
        wall = self._wall.observe(committed_tokens, wall_clock_step_seconds)
        self._wall_started = now
        return {
            # Compatibility aliases retain their historical compute-step meaning.
            **compute,
            "compute_step_seconds": compute["step_seconds"],
            "compute_tokens_per_second": compute["tokens_per_second"],
            "compute_tokens_per_second_ema": compute["tokens_per_second_ema"],
            "compute_eta_seconds": compute["eta_seconds"],
            "wall_clock_step_seconds": wall["step_seconds"],
            "wall_clock_tokens_per_second": wall["tokens_per_second"],
            "wall_clock_tokens_per_second_ema": wall["tokens_per_second_ema"],
            "wall_clock_eta_seconds": wall["eta_seconds"],
        }


def _compact_duration(seconds: Any) -> str:
    value = max(0, round(float(seconds)))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


class TrainingProgress:
    """Rank-zero token progress bar with a safe non-TTY fallback."""

    def __init__(
        self,
        *,
        total_tokens: int,
        initial_tokens: int = 0,
        mode: str = "auto",
        stream: TextIO | None = None,
        description: str = "train",
    ) -> None:
        if mode not in {"auto", "always", "never"}:
            raise ValueError("progress mode must be auto, always, or never")
        self.stream = stream or sys.stderr
        self.enabled = mode == "always" or (
            mode == "auto" and bool(getattr(self.stream, "isatty", lambda: False)())
        )
        self.current_tokens = int(initial_tokens)
        self._bar: Any | None = None
        if self.enabled:
            from tqdm import tqdm

            self._bar = tqdm(
                total=int(total_tokens),
                initial=int(initial_tokens),
                desc=description,
                unit="tok",
                unit_scale=True,
                dynamic_ncols=True,
                mininterval=0.2,
                file=self.stream,
            )

    def update(self, committed_tokens: int, metrics: Mapping[str, Any]) -> None:
        if committed_tokens < self.current_tokens:
            raise ValueError("progress cannot move backwards")
        delta = int(committed_tokens) - self.current_tokens
        self.current_tokens = int(committed_tokens)
        if self._bar is None:
            return
        self._bar.update(delta)
        postfix: dict[str, str | int] = {}
        for key, label, formatter in (
            ("loss", "loss", lambda value: f"{float(value):.4f}"),
            ("ntp_loss", "ntp", lambda value: f"{float(value):.4f}"),
            ("mtp_loss", "mtp", lambda value: f"{float(value):.4f}"),
            ("teacher_kd_loss", "kd", lambda value: f"{float(value):.4f}"),
            ("anchor_kl_loss", "anchor", lambda value: f"{float(value):.4f}"),
            ("hidden_alignment_loss", "hidden", lambda value: f"{float(value):.4f}"),
            ("dense_oracle_loss", "oracle", lambda value: f"{float(value):.4f}"),
            ("router_supervision_loss", "r-sup", lambda value: f"{float(value):.4f}"),
            ("load_balance_loss", "balance", lambda value: f"{float(value):.4f}"),
            ("router_z_loss", "r-z", lambda value: f"{float(value):.4f}"),
            ("grad_norm", "grad", lambda value: f"{float(value):.3f}"),
            ("lr", "lr", lambda value: f"{float(value):.2e}"),
            (
                "compute_tokens_per_second_ema",
                "compute tok/s",
                lambda value: f"{float(value):,.0f}",
            ),
            (
                "wall_clock_tokens_per_second_ema",
                "wall tok/s",
                lambda value: f"{float(value):,.0f}",
            ),
            ("wall_clock_eta_seconds", "ETA", _compact_duration),
            ("gpu_peak_allocated_gib", "peak GiB", lambda value: f"{float(value):.1f}"),
            ("top_k", "top-k", lambda value: str(int(value))),
        ):
            value = metrics.get(key)
            if value is not None:
                postfix[label] = formatter(value)
        if (
            "compute_tokens_per_second_ema" not in metrics
            and metrics.get("tokens_per_second_ema") is not None
        ):
            postfix["tok/s"] = f"{float(metrics['tokens_per_second_ema']):,.0f}"
        self._bar.set_postfix(postfix, refresh=True)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
