"""Small POSIX advisory file-lock helper used by resumable artifacts."""

from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path
from types import TracebackType


class FileLockTimeout(TimeoutError):
    """Raised when an artifact lock is held past the configured deadline."""


class FileLock:
    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        timeout_seconds: float = 300.0,
        poll_seconds: float = 0.1,
    ) -> None:
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds cannot be negative")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.path = Path(path)
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self._file: object | None = None

    def acquire(self) -> FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file = self.path.open("a+b")
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as error:
                if time.monotonic() >= deadline:
                    file.close()
                    raise FileLockTimeout(f"timed out waiting for {self.path}") from error
                time.sleep(self.poll_seconds)
        file.seek(0)
        file.truncate()
        file.write(f"pid={os.getpid()}\n".encode())
        file.flush()
        self._file = file
        return self

    def release(self) -> None:
        file = self._file
        if file is None:
            return
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
        file.close()
        self._file = None

    def __enter__(self) -> FileLock:
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
