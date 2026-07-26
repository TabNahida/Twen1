"""Signal and STOP-file control for interruptible training.

Handlers only set process-local flags.  The training loop observes them after
the current microbatch, coordinates all ranks, and invokes the checkpoint
manager.  A second SIGINT is deliberately exceptional so it cannot overwrite
the last complete checkpoint while attempting another save.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import FrameType


class ImmediateExit(KeyboardInterrupt):
    """Second Ctrl-C requested an immediate, no-checkpoint exit."""

    def __init__(self, exit_code: int = 130) -> None:
        super().__init__("second SIGINT: exit immediately without checkpointing")
        self.exit_code = exit_code


@dataclass(frozen=True, slots=True)
class ControlDecision:
    """A snapshot of actions the training loop should take at a safe point."""

    should_checkpoint: bool
    should_stop: bool
    continue_after_checkpoint: bool
    hard_stop: bool
    reason: str | None
    checkpoint_generation: int


class SignalController:
    """Translate POSIX signals and a STOP file into safe-point decisions.

    SIGINT and SIGTERM request checkpoint-then-exit.  SIGUSR1 requests a
    checkpoint and continued training.  The STOP file is consumed once noticed
    so a resumed run does not immediately stop again.  A second SIGINT raises
    :class:`ImmediateExit` directly from the handler.
    """

    def __init__(
        self,
        stop_file: str | Path | None = None,
        *,
        consume_stop_file: bool = True,
    ) -> None:
        self.stop_file = Path(stop_file) if stop_file is not None else None
        self.consume_stop_file = consume_stop_file
        # Python runs signal handlers on the main thread between bytecodes.  An
        # ordinary Lock can deadlock if a signal interrupts ``poll`` while that
        # same thread holds it; RLock makes that re-entry safe.
        self._lock = threading.RLock()
        self._graceful_stop = False
        self._hard_stop = False
        self._reason: str | None = None
        self._sigint_count = 0
        self._checkpoint_requested = False
        self._checkpoint_generation = 0
        self._installed = False
        self._previous_handlers: dict[int, Callable[..., object] | int | None] = {}

    @property
    def installed(self) -> bool:
        return self._installed

    @property
    def graceful_stop_requested(self) -> bool:
        with self._lock:
            return self._graceful_stop

    @property
    def hard_stop_requested(self) -> bool:
        with self._lock:
            return self._hard_stop

    def install(self) -> SignalController:
        """Install handlers in the main thread and remember prior handlers."""

        if self._installed:
            return self
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("signal handlers can only be installed from the main thread")

        handled = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGUSR1"):
            handled.append(signal.SIGUSR1)
        for signum in handled:
            self._previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self.handle_signal)
        self._installed = True
        return self

    def restore(self) -> None:
        """Restore all handlers that were active before :meth:`install`."""

        if not self._installed:
            return
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("signal handlers can only be restored from the main thread")
        for signum, previous in self._previous_handlers.items():
            signal.signal(signum, previous)
        self._previous_handlers.clear()
        self._installed = False

    def __enter__(self) -> SignalController:
        return self.install()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.restore()

    def handle_signal(self, signum: int, frame: FrameType | None = None) -> None:
        """Signal handler; public to permit deterministic CPU-only tests."""

        del frame
        if signum == signal.SIGINT:
            with self._lock:
                self._sigint_count += 1
                if self._sigint_count >= 2:
                    self._hard_stop = True
                else:
                    self._request_graceful_stop_locked("sigint")
                hard_stop = self._hard_stop
            if hard_stop:
                raise ImmediateExit(130)
            return

        if signum == signal.SIGTERM:
            with self._lock:
                self._request_graceful_stop_locked("sigterm")
            return

        if hasattr(signal, "SIGUSR1") and signum == signal.SIGUSR1:
            with self._lock:
                self._checkpoint_requested = True
                self._checkpoint_generation += 1
                if not self._graceful_stop:
                    self._reason = "sigusr1"
            return

    def _request_graceful_stop_locked(self, reason: str) -> None:
        self._graceful_stop = True
        self._checkpoint_requested = True
        self._checkpoint_generation += 1
        self._reason = reason

    def check_stop_file(self) -> bool:
        """Notice and optionally consume a STOP file.

        Returns ``True`` only for the poll that first observes the file.
        """

        if self.stop_file is None or not self.stop_file.is_file():
            return False
        with self._lock:
            first_observation = not self._graceful_stop
            self._request_graceful_stop_locked("stop-file")
        if self.consume_stop_file:
            with suppress(OSError):
                self.stop_file.unlink()
        return first_observation

    def poll(self) -> ControlDecision:
        """Poll STOP and return a coherent control snapshot."""

        self.check_stop_file()
        with self._lock:
            should_checkpoint = self._checkpoint_requested or self._graceful_stop
            should_stop = self._graceful_stop
            return ControlDecision(
                should_checkpoint=should_checkpoint,
                should_stop=should_stop,
                continue_after_checkpoint=should_checkpoint and not should_stop,
                hard_stop=self._hard_stop,
                reason=self._reason,
                checkpoint_generation=self._checkpoint_generation,
            )

    def acknowledge_checkpoint(self, generation: int | None = None) -> None:
        """Clear a completed checkpoint request, unless a newer one arrived."""

        with self._lock:
            if generation is not None and generation != self._checkpoint_generation:
                return
            if not self._graceful_stop:
                self._checkpoint_requested = False
                if self._reason == "sigusr1":
                    self._reason = None

    def raise_if_hard_stop(self) -> None:
        with self._lock:
            hard_stop = self._hard_stop
        if hard_stop:
            raise ImmediateExit(130)


__all__ = ["ControlDecision", "ImmediateExit", "SignalController"]
