"""Small stderr-only progress bars for user-run preprocessing tasks."""

from __future__ import annotations

import sys
from typing import Any, TextIO


class TaskProgress:
    """A tqdm wrapper whose ``auto`` mode is silent outside an interactive TTY.

    Machine-readable command results stay on stdout; tqdm always writes to the
    supplied stream, which defaults to stderr.
    """

    def __init__(
        self,
        *,
        total: int,
        description: str,
        unit: str,
        initial: int = 0,
        mode: str = "auto",
        unit_scale: bool = False,
        stream: TextIO | None = None,
    ) -> None:
        if mode not in {"auto", "always", "never"}:
            raise ValueError("progress mode must be auto, always, or never")
        if total < 0 or initial < 0 or initial > total:
            raise ValueError("progress total/initial values are invalid")
        self.stream = stream or sys.stderr
        self.enabled = mode == "always" or (
            mode == "auto" and bool(getattr(self.stream, "isatty", lambda: False)())
        )
        self._bar: Any | None = None
        if self.enabled:
            from tqdm import tqdm

            self._bar = tqdm(
                total=total,
                initial=initial,
                desc=description,
                unit=unit,
                unit_scale=unit_scale,
                dynamic_ncols=True,
                mininterval=0.2,
                file=self.stream,
            )

    def update(self, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("progress updates cannot be negative")
        if self._bar is not None:
            self._bar.update(amount)

    def set_postfix(self, values: dict[str, object]) -> None:
        if self._bar is not None:
            self._bar.set_postfix(values, refresh=False)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def __enter__(self) -> TaskProgress:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = ["TaskProgress"]
