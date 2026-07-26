"""Lazy random access to authenticated prepared-text token shards.

This module deliberately lives outside :mod:`twen.data.prepared`: adding a
training reader must not change the source digest that authenticates the
existing tokenizer/preparation generator.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prepared import PREPARED_TENSORS

PREPARED_TEXT_REQUIRED_TENSORS = (
    "input_ids",
    "labels",
    "attention_mask",
)


class PreparedTextSchemaError(ValueError):
    """Raised when a prepared token shard cannot satisfy the training contract."""


@dataclass(frozen=True, slots=True)
class PreparedTextRecord:
    input_ids: Any
    labels: Any
    attention_mask: Any


@dataclass(frozen=True, slots=True)
class PreparedTextBatch:
    input_ids: Any
    labels: Any
    attention_mask: Any

    def tensors(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in PREPARED_TEXT_REQUIRED_TENSORS
        }

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def record(self, index: int) -> PreparedTextRecord:
        if not 0 <= index < len(self):
            raise IndexError(index)
        return PreparedTextRecord(
            **{
                name: getattr(self, name)[index]
                for name in PREPARED_TEXT_REQUIRED_TENSORS
            }
        )


def _read_safetensors_header(path: Path) -> dict[str, dict[str, object]]:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise PreparedTextSchemaError(f"truncated safetensors file: {path}")
            header_length = struct.unpack("<Q", prefix)[0]
            if header_length <= 0 or header_length > 64 * 1024 * 1024:
                raise PreparedTextSchemaError(
                    f"invalid safetensors header length: {path}"
                )
            raw_header = handle.read(header_length)
            if len(raw_header) != header_length:
                raise PreparedTextSchemaError(f"truncated safetensors header: {path}")
    except OSError as error:
        raise PreparedTextSchemaError(f"cannot read prepared tensors: {path}") from error
    try:
        payload = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreparedTextSchemaError(f"invalid safetensors header: {path}") from error
    if not isinstance(payload, dict):
        raise PreparedTextSchemaError(f"invalid safetensors header object: {path}")
    tensors = {
        key: value
        for key, value in payload.items()
        if key != "__metadata__"
    }
    data_size = path.stat().st_size - 8 - header_length
    for name, entry in tensors.items():
        if not isinstance(entry, dict):
            raise PreparedTextSchemaError(
                f"invalid safetensors entry {name!r}: {path}"
            )
        offsets = entry.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= data_size
        ):
            raise PreparedTextSchemaError(
                f"invalid data offsets for {name!r}: {path}"
            )
    return tensors


def _validate_tensor_header(
    tensor_path: Path,
    *,
    sequence_count: int,
    sequence_length: int,
) -> None:
    header = _read_safetensors_header(tensor_path)
    expected = {
        "input_ids": ([sequence_count, sequence_length], {"I64"}),
        "labels": ([sequence_count, sequence_length], {"I64"}),
        "attention_mask": ([sequence_count, sequence_length], {"BOOL"}),
    }
    if set(header) != set(expected):
        raise PreparedTextSchemaError(
            "prepared tensor names mismatch: "
            f"expected {sorted(expected)}, got {sorted(header)}"
        )
    for name, (shape, dtypes) in expected.items():
        entry = header[name]
        if entry.get("shape") != shape:
            raise PreparedTextSchemaError(
                f"{name} shape mismatch: expected {shape}, got {entry.get('shape')}"
            )
        if entry.get("dtype") not in dtypes:
            raise PreparedTextSchemaError(
                f"{name} dtype mismatch: expected {sorted(dtypes)}, "
                f"got {entry.get('dtype')}"
            )


class PreparedTextShardDataset(Sequence[PreparedTextRecord]):
    """Lazy mmap-backed access to one prepared ``tokens.safetensors`` shard."""

    def __init__(
        self,
        shard_directory: str | Path,
        *,
        expected_sequence_count: int,
        expected_sequence_length: int,
    ) -> None:
        if (
            isinstance(expected_sequence_count, bool)
            or not isinstance(expected_sequence_count, int)
            or expected_sequence_count <= 0
            or isinstance(expected_sequence_length, bool)
            or not isinstance(expected_sequence_length, int)
            or expected_sequence_length <= 0
        ):
            raise ValueError("expected sequence count and length must be positive integers")
        self.shard_directory = Path(shard_directory)
        if self.shard_directory.name.endswith(".incomplete"):
            raise PreparedTextSchemaError(
                f"refusing prepared staging directory: {self.shard_directory}"
            )
        self.sequence_count = expected_sequence_count
        self.sequence_length = expected_sequence_length
        tensor_path = self.shard_directory / PREPARED_TENSORS
        _validate_tensor_header(
            tensor_path,
            sequence_count=expected_sequence_count,
            sequence_length=expected_sequence_length,
        )
        try:
            from safetensors import safe_open
        except ImportError as error:  # pragma: no cover - required runtime dependency
            raise RuntimeError("lazy prepared-text access requires safetensors") from error
        self._context = safe_open(
            str(tensor_path),
            framework="pt",
            device="cpu",
        )
        self._handle = self._context.__enter__()
        self._closed = False

    def __len__(self) -> int:
        return self.sequence_count

    def __getitem__(self, index: int) -> PreparedTextRecord:
        if self._closed:
            raise RuntimeError("prepared-text shard dataset is closed")
        if not 0 <= index < len(self):
            raise IndexError(index)
        return PreparedTextRecord(
            **{
                name: self._handle.get_slice(name)[index].clone()
                for name in PREPARED_TEXT_REQUIRED_TENSORS
            }
        )

    def loss_count_vectors(self, *, ignore_index: int = -100) -> tuple[Any, Any]:
        """Return per-sequence (NTP target, valid-hidden) count vectors."""

        if self._closed:
            raise RuntimeError("prepared-text shard dataset is closed")
        import torch

        labels = self._handle.get_tensor("labels")
        attention_mask = self._handle.get_tensor("attention_mask")
        hidden_counts = attention_mask.ne(0).sum(dim=1, dtype=torch.int64)
        target_counts = (
            labels[:, 1:].ne(ignore_index) & attention_mask[:, :-1].ne(0)
        ).sum(dim=1, dtype=torch.int64)
        return target_counts, hidden_counts

    def mtp_loss_count_vector(self, *, ignore_index: int = -100) -> Any:
        """Return exact per-sequence native Qwen3.5 ``L-2`` target counts."""

        if self._closed:
            raise RuntimeError("prepared-text shard dataset is closed")
        import torch

        labels = self._handle.get_tensor("labels")
        attention_mask = self._handle.get_tensor("attention_mask")
        return (
            labels[:, 2:].ne(ignore_index)
            & attention_mask[:, :-2].ne(0)
            & attention_mask[:, 1:-1].ne(0)
            & attention_mask[:, 2:].ne(0)
        ).sum(dim=1, dtype=torch.int64)

    def close(self) -> None:
        if self._closed:
            return
        self._context.__exit__(None, None, None)
        self._closed = True

    def __enter__(self) -> PreparedTextShardDataset:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - deterministic callers use close()
        with suppress(Exception):
            self.close()


def collate_prepared_text(
    records: Sequence[PreparedTextRecord],
) -> PreparedTextBatch:
    if not records:
        raise ValueError("cannot collate an empty prepared-text batch")
    try:
        import torch
    except ImportError as error:  # pragma: no cover - required runtime dependency
        raise RuntimeError("collating prepared text requires PyTorch") from error
    return PreparedTextBatch(
        **{
            name: torch.stack([getattr(record, name) for record in records])
            for name in PREPARED_TEXT_REQUIRED_TENSORS
        }
    )


__all__ = [
    "PREPARED_TEXT_REQUIRED_TENSORS",
    "PreparedTextBatch",
    "PreparedTextRecord",
    "PreparedTextSchemaError",
    "PreparedTextShardDataset",
    "collate_prepared_text",
]
