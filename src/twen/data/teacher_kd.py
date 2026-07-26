"""Versioned top-64 teacher-KD shard schema and lazy PyTorch loader.

The top-k logits stored here are *raw* teacher logits.  ``teacher_logsumexp``
is ``logsumexp(raw_logits / temperature)`` over the full vocabulary and
``teacher_tail_logprob`` is the log probability mass of all non-top-k tokens at
that same temperature.  A consumer must match the manifest temperature.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from twen.io.download import sha256_file, validate_pinned_revision

from .shards import ShardTransaction, is_shard_complete

KD_SCHEMA_VERSION = 2
KD_TOP_K = 64
KD_MANIFEST_FILENAME = "kd_manifest.json"
KD_TENSORS_FILENAME = "kd_tensors.safetensors"
KD_REQUIRED_TENSORS = (
    "input_ids",
    "labels",
    "attention_mask",
    "topk_indices",
    "topk_logits",
    "teacher_logsumexp",
    "teacher_tail_logprob",
)
KD_NORMALIZATION_DEFINITION = (
    "teacher_logsumexp=logsumexp(raw_logits/temperature);"
    "teacher_tail_logprob=log(sum_{v_not_in_topk} "
    "exp(raw_logits[v]/temperature))/exp(teacher_logsumexp)"
)


def _kd_generator_source_sha256() -> str:
    """Bind both the GPU generation algorithm and its durable schema/writer."""

    digest = hashlib.sha256()
    for source in (Path(__file__).parents[1] / "kd.py", Path(__file__)):
        payload = source.read_bytes()
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


KD_GENERATOR_SOURCE_SHA256 = _kd_generator_source_sha256()


class TeacherKDSchemaError(ValueError):
    """Raised when cached teacher data is incompatible or corrupt."""


def _require_sha256(value: str, field: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise TeacherKDSchemaError(f"{field} must be a 64-digit SHA256")
    return normalized


@dataclass(frozen=True, slots=True)
class TeacherKDManifest:
    teacher_model_id: str
    teacher_revision: str
    teacher_model_sha256: str
    generator_source_sha256: str
    tokenizer_sha256: str
    dataset_fingerprint: str
    source_tensors_sha256: str
    source_shard_id: str
    global_sample_start: int
    global_sample_end: int
    global_token_start: int
    global_token_end: int
    sequence_count: int
    sequence_length: int
    token_count: int
    vocab_size: int
    temperature: float
    top_k: int = KD_TOP_K
    logits_dtype: str = "F16"
    tensors_sha256: str | None = None
    schema_version: int = KD_SCHEMA_VERSION
    topk_logits_are_raw: bool = True
    normalization_definition: str = KD_NORMALIZATION_DEFINITION

    def __post_init__(self) -> None:
        if self.schema_version != KD_SCHEMA_VERSION:
            raise TeacherKDSchemaError("unsupported teacher-KD schema")
        if not self.teacher_model_id or not self.dataset_fingerprint or not self.source_shard_id:
            raise TeacherKDSchemaError("teacher, dataset and source shard identity are required")
        object.__setattr__(
            self, "teacher_revision", validate_pinned_revision(self.teacher_revision)
        )
        object.__setattr__(
            self,
            "teacher_model_sha256",
            _require_sha256(self.teacher_model_sha256, "teacher_model_sha256"),
        )
        object.__setattr__(
            self,
            "generator_source_sha256",
            _require_sha256(
                self.generator_source_sha256,
                "generator_source_sha256",
            ),
        )
        object.__setattr__(
            self,
            "tokenizer_sha256",
            _require_sha256(self.tokenizer_sha256, "tokenizer_sha256"),
        )
        object.__setattr__(
            self,
            "source_tensors_sha256",
            _require_sha256(self.source_tensors_sha256, "source_tensors_sha256"),
        )
        if self.tensors_sha256 is not None:
            object.__setattr__(
                self,
                "tensors_sha256",
                _require_sha256(self.tensors_sha256, "tensors_sha256"),
            )
        if self.top_k != KD_TOP_K:
            raise TeacherKDSchemaError(f"top_k must be exactly {KD_TOP_K}")
        if not self.topk_logits_are_raw:
            raise TeacherKDSchemaError("topk_logits must be unscaled/raw logits")
        if self.normalization_definition != KD_NORMALIZATION_DEFINITION:
            raise TeacherKDSchemaError("unknown teacher normalization semantics")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise TeacherKDSchemaError("temperature must be finite and positive")
        if self.logits_dtype not in {"F16", "BF16", "F32"}:
            raise TeacherKDSchemaError("logits_dtype must be F16, BF16, or F32")
        if self.sequence_count <= 0 or self.sequence_length <= 0 or self.token_count <= 0:
            raise TeacherKDSchemaError("sequence and token sizes must be positive")
        if self.token_count > self.sequence_count * self.sequence_length:
            raise TeacherKDSchemaError("token_count exceeds padded sequence capacity")
        if self.vocab_size <= self.top_k:
            raise TeacherKDSchemaError("vocab_size must exceed top_k")
        if not (
            0 <= self.global_sample_start < self.global_sample_end
            and self.global_sample_end - self.global_sample_start == self.sequence_count
        ):
            raise TeacherKDSchemaError("global sample range does not match sequence_count")
        if not (
            0 <= self.global_token_start < self.global_token_end
            and self.global_token_end - self.global_token_start == self.token_count
        ):
            raise TeacherKDSchemaError("global token range does not match token_count")

    def require_temperature(self, training_temperature: float) -> None:
        if not math.isclose(
            training_temperature, self.temperature, rel_tol=0.0, abs_tol=1e-12
        ):
            raise TeacherKDSchemaError(
                "KD temperature mismatch: cache was normalized at "
                f"{self.temperature:g}, training requested {training_temperature:g}"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TeacherKDManifest:
        if int(payload.get("schema_version", 0)) != KD_SCHEMA_VERSION:
            raise TeacherKDSchemaError(
                "unsupported teacher-KD schema; v1 has no generator identity "
                "and must be regenerated"
            )
        fields = {
            "teacher_model_id": str(payload["teacher_model_id"]),
            "teacher_revision": str(payload["teacher_revision"]),
            "teacher_model_sha256": str(payload["teacher_model_sha256"]),
            "generator_source_sha256": str(payload["generator_source_sha256"]),
            "tokenizer_sha256": str(payload["tokenizer_sha256"]),
            "dataset_fingerprint": str(payload["dataset_fingerprint"]),
            "source_tensors_sha256": str(payload["source_tensors_sha256"]),
            "source_shard_id": str(payload["source_shard_id"]),
            "global_sample_start": int(payload["global_sample_start"]),
            "global_sample_end": int(payload["global_sample_end"]),
            "global_token_start": int(payload["global_token_start"]),
            "global_token_end": int(payload["global_token_end"]),
            "sequence_count": int(payload["sequence_count"]),
            "sequence_length": int(payload["sequence_length"]),
            "token_count": int(payload["token_count"]),
            "vocab_size": int(payload["vocab_size"]),
            "temperature": float(payload["temperature"]),
            "top_k": int(payload.get("top_k", KD_TOP_K)),
            "logits_dtype": str(payload.get("logits_dtype", "F16")),
            "tensors_sha256": (
                str(payload["tensors_sha256"])
                if payload.get("tensors_sha256") is not None
                else None
            ),
            "schema_version": int(payload.get("schema_version", 0)),
            "topk_logits_are_raw": bool(payload.get("topk_logits_are_raw", False)),
            "normalization_definition": str(
                payload.get("normalization_definition", "")
            ),
        }
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class TeacherKDRecord:
    input_ids: Any
    labels: Any
    attention_mask: Any
    topk_indices: Any
    topk_logits: Any
    teacher_logsumexp: Any
    teacher_tail_logprob: Any
    temperature: float


@dataclass(frozen=True, slots=True)
class TeacherKDBatch:
    input_ids: Any
    labels: Any
    attention_mask: Any
    topk_indices: Any
    topk_logits: Any
    teacher_logsumexp: Any
    teacher_tail_logprob: Any
    temperature: float

    def tensors(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in KD_REQUIRED_TENSORS}

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def record(self, index: int) -> TeacherKDRecord:
        if not 0 <= index < len(self):
            raise IndexError(index)
        return TeacherKDRecord(
            **{name: getattr(self, name)[index] for name in KD_REQUIRED_TENSORS},
            temperature=self.temperature,
        )


def _read_safetensors_header(path: Path) -> dict[str, dict[str, object]]:
    with path.open("rb") as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise TeacherKDSchemaError(f"truncated safetensors file: {path}")
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length <= 0 or header_length > 64 * 1024 * 1024:
            raise TeacherKDSchemaError(f"invalid safetensors header length: {path}")
        raw_header = file.read(header_length)
        if len(raw_header) != header_length:
            raise TeacherKDSchemaError(f"truncated safetensors header: {path}")
    try:
        payload = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TeacherKDSchemaError(f"invalid safetensors header: {path}") from error
    tensors = {key: value for key, value in payload.items() if key != "__metadata__"}
    data_size = path.stat().st_size - 8 - header_length
    for name, entry in tensors.items():
        if not isinstance(entry, dict):
            raise TeacherKDSchemaError(f"invalid safetensors entry {name!r}: {path}")
        offsets = entry.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= data_size
        ):
            raise TeacherKDSchemaError(f"invalid data offsets for {name!r}: {path}")
    return tensors


def _expected_tensor_schema(manifest: TeacherKDManifest) -> dict[str, tuple[list[int], set[str]]]:
    n, length, top_k = (
        manifest.sequence_count,
        manifest.sequence_length,
        manifest.top_k,
    )
    return {
        "input_ids": ([n, length], {"I64"}),
        "labels": ([n, length], {"I64"}),
        "attention_mask": ([n, length], {"BOOL", "I64"}),
        "topk_indices": ([n, length, top_k], {"I64"}),
        "topk_logits": ([n, length, top_k], {manifest.logits_dtype}),
        "teacher_logsumexp": ([n, length], {"F32"}),
        "teacher_tail_logprob": ([n, length], {"F32"}),
    }


def _validate_kd_tensor_header(
    tensor_path: Path, manifest: TeacherKDManifest
) -> None:
    header = _read_safetensors_header(tensor_path)
    expected = _expected_tensor_schema(manifest)
    if set(header) != set(expected):
        raise TeacherKDSchemaError(
            f"KD tensor names mismatch: expected {sorted(expected)}, got {sorted(header)}"
        )
    for name, (shape, dtypes) in expected.items():
        entry = header[name]
        if entry.get("shape") != shape:
            raise TeacherKDSchemaError(
                f"{name} shape mismatch: expected {shape}, got {entry.get('shape')}"
            )
        if entry.get("dtype") not in dtypes:
            raise TeacherKDSchemaError(
                f"{name} dtype mismatch: expected {sorted(dtypes)}, got {entry.get('dtype')}"
            )


def read_kd_manifest(shard_directory: str | os.PathLike[str]) -> TeacherKDManifest:
    path = Path(shard_directory) / KD_MANIFEST_FILENAME
    with path.open("r", encoding="utf-8") as file:
        return TeacherKDManifest.from_dict(json.load(file))


def validate_kd_shard(
    shard_directory: str | os.PathLike[str],
    *,
    expected_temperature: float | None = None,
    require_complete: bool = True,
    verify_checksum: bool = True,
) -> TeacherKDManifest:
    shard = Path(shard_directory)
    if shard.name.endswith(".incomplete"):
        raise TeacherKDSchemaError(f"refusing KD staging directory: {shard}")
    if require_complete and not is_shard_complete(shard, verify_hashes=False):
        raise TeacherKDSchemaError(f"teacher-KD shard is not COMPLETE: {shard}")
    manifest = read_kd_manifest(shard)
    if manifest.generator_source_sha256 != KD_GENERATOR_SOURCE_SHA256:
        raise TeacherKDSchemaError(
            "KD generator source changed; regenerate the KD shard instead of "
            f"reusing stale output: {shard}"
        )
    if expected_temperature is not None:
        manifest.require_temperature(expected_temperature)
    tensor_path = shard / KD_TENSORS_FILENAME
    if not tensor_path.is_file():
        raise TeacherKDSchemaError(f"missing {KD_TENSORS_FILENAME} in {shard}")
    if manifest.tensors_sha256 is None:
        raise TeacherKDSchemaError("KD manifest has no tensor SHA256")
    if verify_checksum:
        actual_hash = sha256_file(tensor_path)
        if actual_hash != manifest.tensors_sha256:
            raise TeacherKDSchemaError(
                f"KD tensor SHA256 mismatch: expected {manifest.tensors_sha256}, got {actual_hash}"
            )
    _validate_kd_tensor_header(tensor_path, manifest)
    return manifest


def load_kd_shard(
    shard_directory: str | os.PathLike[str],
    *,
    expected_temperature: float,
    device: str = "cpu",
    verify_checksum: bool = True,
) -> tuple[TeacherKDManifest, TeacherKDBatch]:
    """Validate and load one shard; imports PyTorch/safetensors only on demand."""

    manifest = validate_kd_shard(
        shard_directory,
        expected_temperature=expected_temperature,
        verify_checksum=verify_checksum,
    )
    try:
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover - depends on runtime extra
        raise RuntimeError(
            "loading teacher-KD tensors requires the 'safetensors' package"
        ) from error
    tensors = load_file(str(Path(shard_directory) / KD_TENSORS_FILENAME), device=device)
    batch = TeacherKDBatch(
        **{name: tensors[name] for name in KD_REQUIRED_TENSORS},
        temperature=manifest.temperature,
    )
    # Header validation guarantees int64 indices.  This runtime check catches a
    # malformed/custom loader before it reaches gather/scatter in the KD loss.
    if str(batch.topk_indices.dtype) != "torch.int64":
        raise TeacherKDSchemaError("topk_indices must load as torch.int64")
    if device == "cpu":
        valid_tokens = int(batch.attention_mask.to(dtype=tensors["input_ids"].dtype).sum())
        if valid_tokens != manifest.token_count:
            raise TeacherKDSchemaError(
                f"attention_mask has {valid_tokens} tokens, manifest says {manifest.token_count}"
            )
    return manifest, batch


class TeacherKDShardDataset(Sequence[TeacherKDRecord]):
    """Lazy mmap-backed random access to one verified KD shard."""

    def __init__(
        self,
        shard_directory: str | os.PathLike[str],
        *,
        expected_temperature: float,
        verify_checksum: bool = True,
    ) -> None:
        self.shard_directory = Path(shard_directory)
        self.manifest = validate_kd_shard(
            shard_directory,
            expected_temperature=expected_temperature,
            verify_checksum=verify_checksum,
        )
        try:
            from safetensors import safe_open
        except ImportError as error:  # pragma: no cover - required runtime dependency
            raise RuntimeError("lazy KD access requires safetensors") from error
        self._context = safe_open(
            self.shard_directory / KD_TENSORS_FILENAME,
            framework="pt",
            device="cpu",
        )
        self._handle = self._context.__enter__()
        self._closed = False

    def __len__(self) -> int:
        return self.manifest.sequence_count

    def __getitem__(self, index: int) -> TeacherKDRecord:
        if self._closed:
            raise RuntimeError("KD shard dataset is closed")
        if not 0 <= index < len(self):
            raise IndexError(index)
        return TeacherKDRecord(
            **{
                name: self._handle.get_slice(name)[index].clone()
                for name in KD_REQUIRED_TENSORS
            },
            temperature=self.manifest.temperature,
        )

    def loss_token_counts(self, index: int, *, ignore_index: int = -100) -> tuple[int, int]:
        """Return (next-token targets, valid hidden tokens) without loading KD logits."""

        if self._closed:
            raise RuntimeError("KD shard dataset is closed")
        if not 0 <= index < len(self):
            raise IndexError(index)
        labels = self._handle.get_slice("labels")[index]
        attention_mask = self._handle.get_slice("attention_mask")[index]
        valid_hidden_tokens = int(attention_mask.ne(0).sum().item())
        if labels.numel() <= 1:
            target_tokens = 0
        else:
            target_tokens = int(
                (
                    labels[1:].ne(ignore_index)
                    & attention_mask[:-1].ne(0)
                ).sum().item()
            )
        return target_tokens, valid_hidden_tokens

    def close(self) -> None:
        if self._closed:
            return
        self._context.__exit__(None, None, None)
        self._closed = True

    def __enter__(self) -> TeacherKDShardDataset:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - deterministic callers use close()
        with suppress(Exception):
            self.close()


def collate_teacher_kd(records: Sequence[TeacherKDRecord]) -> TeacherKDBatch:
    if not records:
        raise ValueError("cannot collate an empty KD batch")
    temperature = records[0].temperature
    if any(record.temperature != temperature for record in records):
        raise TeacherKDSchemaError("cannot mix KD records with different temperatures")
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on runtime extra
        raise RuntimeError("collating KD records requires PyTorch") from error
    return TeacherKDBatch(
        **{
            name: torch.stack([getattr(record, name) for record in records])
            for name in KD_REQUIRED_TENSORS
        },
        temperature=temperature,
    )


TensorSerializer = Callable[[Mapping[str, Any], str], None]


def _default_tensor_serializer(tensors: Mapping[str, Any], path: str) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - depends on runtime extra
        raise RuntimeError(
            "writing teacher-KD tensors requires the 'safetensors' package"
        ) from error
    save_file(dict(tensors), path)


def write_kd_shard(
    output_root: str | os.PathLike[str],
    shard_id: str,
    *,
    manifest: TeacherKDManifest,
    batch: TeacherKDBatch,
    serializer: TensorSerializer | None = None,
) -> Path:
    """Atomically write a KD shard using the generic ``COMPLETE`` protocol."""

    if manifest.generator_source_sha256 != KD_GENERATOR_SOURCE_SHA256:
        raise TeacherKDSchemaError("KD manifest generator identity is not current")
    manifest.require_temperature(batch.temperature)
    fingerprint_payload = manifest.to_dict()
    fingerprint_payload["tensors_sha256"] = None
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    with ShardTransaction(
        output_root,
        shard_id,
        fingerprint=fingerprint,
        source_fingerprint=manifest.dataset_fingerprint,
    ) as transaction:
        if transaction.complete:
            validate_kd_shard(
                transaction.final_directory,
                expected_temperature=manifest.temperature,
            )
            return transaction.final_directory
        tensor_path = transaction.work_directory / KD_TENSORS_FILENAME
        active_serializer = serializer or _default_tensor_serializer
        active_serializer(batch.tensors(), str(tensor_path))
        tensor_hash = sha256_file(tensor_path)
        completed_manifest = replace(manifest, tensors_sha256=tensor_hash)
        manifest_path = transaction.work_directory / KD_MANIFEST_FILENAME
        temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(
                completed_manifest.to_dict(),
                file,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, manifest_path)
        _validate_kd_tensor_header(tensor_path, completed_manifest)
        manifest_hash = sha256_file(manifest_path)
        return transaction.commit(
            {
                "kind": "teacher_top64_kd",
                "temperature": manifest.temperature,
                "global_token_start": manifest.global_token_start,
                "global_token_end": manifest.global_token_end,
            },
            known_sha256={
                KD_TENSORS_FILENAME: tensor_hash,
                KD_MANIFEST_FILENAME: manifest_hash,
            },
        )


@dataclass(frozen=True, slots=True)
class TeacherKDShardEntry:
    path: str
    source_shard_id: str
    source_tensors_sha256: str
    manifest_sha256: str
    tensors_sha256: str
    global_sample_start: int
    global_sample_end: int
    global_token_start: int
    global_token_end: int
    sequence_count: int
    token_count: int

    def __post_init__(self) -> None:
        relative = PurePosixPath(self.path)
        if not self.path or self.path == "." or relative.is_absolute() or ".." in relative.parts:
            raise TeacherKDSchemaError(f"unsafe KD shard path {self.path!r}")
        if not self.source_shard_id:
            raise TeacherKDSchemaError("source_shard_id is required")
        object.__setattr__(
            self,
            "source_tensors_sha256",
            _require_sha256(self.source_tensors_sha256, "source_tensors_sha256"),
        )
        object.__setattr__(
            self, "manifest_sha256", _require_sha256(self.manifest_sha256, "manifest_sha256")
        )
        object.__setattr__(
            self, "tensors_sha256", _require_sha256(self.tensors_sha256, "tensors_sha256")
        )
        if self.global_sample_end - self.global_sample_start != self.sequence_count:
            raise TeacherKDSchemaError("shard sample range/count mismatch")
        if self.global_token_end - self.global_token_start != self.token_count:
            raise TeacherKDSchemaError("shard token range/count mismatch")
        if self.sequence_count <= 0 or self.token_count <= 0:
            raise TeacherKDSchemaError("shard sequence/token counts must be positive")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TeacherKDShardEntry:
        return cls(
            path=str(payload["path"]),
            source_shard_id=str(payload["source_shard_id"]),
            source_tensors_sha256=str(payload["source_tensors_sha256"]),
            manifest_sha256=str(payload["manifest_sha256"]),
            tensors_sha256=str(payload["tensors_sha256"]),
            global_sample_start=int(payload["global_sample_start"]),
            global_sample_end=int(payload["global_sample_end"]),
            global_token_start=int(payload["global_token_start"]),
            global_token_end=int(payload["global_token_end"]),
            sequence_count=int(payload["sequence_count"]),
            token_count=int(payload["token_count"]),
        )


@dataclass(frozen=True, slots=True)
class TeacherKDCorpusManifest:
    teacher_model_id: str
    teacher_revision: str
    teacher_model_sha256: str
    generator_source_sha256: str
    tokenizer_sha256: str
    dataset_fingerprint: str
    temperature: float
    shards: tuple[TeacherKDShardEntry, ...]
    top_k: int = KD_TOP_K
    schema_version: int = KD_SCHEMA_VERSION
    kind: str = "teacher_top64_kd_corpus"
    topk_logits_are_raw: bool = True
    normalization_definition: str = KD_NORMALIZATION_DEFINITION

    def __post_init__(self) -> None:
        if self.schema_version != KD_SCHEMA_VERSION or self.kind != "teacher_top64_kd_corpus":
            raise TeacherKDSchemaError("unsupported KD corpus manifest")
        object.__setattr__(
            self, "teacher_revision", validate_pinned_revision(self.teacher_revision)
        )
        object.__setattr__(
            self,
            "teacher_model_sha256",
            _require_sha256(self.teacher_model_sha256, "teacher_model_sha256"),
        )
        object.__setattr__(
            self,
            "generator_source_sha256",
            _require_sha256(
                self.generator_source_sha256,
                "generator_source_sha256",
            ),
        )
        object.__setattr__(
            self,
            "tokenizer_sha256",
            _require_sha256(self.tokenizer_sha256, "tokenizer_sha256"),
        )
        if not self.dataset_fingerprint or not self.shards:
            raise TeacherKDSchemaError("KD corpus requires dataset identity and shards")
        if self.top_k != KD_TOP_K or not self.topk_logits_are_raw:
            raise TeacherKDSchemaError("KD corpus must contain raw top-64 logits")
        if self.normalization_definition != KD_NORMALIZATION_DEFINITION:
            raise TeacherKDSchemaError("unknown KD corpus normalization semantics")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise TeacherKDSchemaError("KD corpus temperature must be positive")
        ordered = tuple(sorted(self.shards, key=lambda item: item.global_sample_start))
        if ordered != self.shards:
            raise TeacherKDSchemaError("KD corpus shards must be in global sample order")
        if self.shards[0].global_sample_start != 0 or self.shards[0].global_token_start != 0:
            raise TeacherKDSchemaError("KD corpus must start at prepared sample/token zero")
        for previous, current in zip(self.shards[:-1], self.shards[1:], strict=True):
            if previous.global_sample_end != current.global_sample_start:
                raise TeacherKDSchemaError("KD corpus sample ranges are not contiguous")
            if previous.global_token_end != current.global_token_start:
                raise TeacherKDSchemaError("KD corpus token ranges are not contiguous")

    @property
    def sequence_count(self) -> int:
        return sum(entry.sequence_count for entry in self.shards)

    @property
    def token_count(self) -> int:
        return sum(entry.token_count for entry in self.shards)

    def require_temperature(self, training_temperature: float) -> None:
        if not math.isclose(
            training_temperature, self.temperature, rel_tol=0.0, abs_tol=1e-12
        ):
            raise TeacherKDSchemaError(
                "KD corpus temperature mismatch: cache was normalized at "
                f"{self.temperature:g}, training requested {training_temperature:g}"
            )

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["sequence_count"] = self.sequence_count
        result["token_count"] = self.token_count
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TeacherKDCorpusManifest:
        if int(payload.get("schema_version", 0)) != KD_SCHEMA_VERSION:
            raise TeacherKDSchemaError(
                "unsupported KD corpus schema; v1 has no generator identity "
                "and must be regenerated"
            )
        raw_shards = payload.get("shards")
        if not isinstance(raw_shards, list):
            raise TeacherKDSchemaError("KD corpus shards must be a list")
        manifest = cls(
            teacher_model_id=str(payload["teacher_model_id"]),
            teacher_revision=str(payload["teacher_revision"]),
            teacher_model_sha256=str(payload["teacher_model_sha256"]),
            generator_source_sha256=str(payload["generator_source_sha256"]),
            tokenizer_sha256=str(payload["tokenizer_sha256"]),
            dataset_fingerprint=str(payload["dataset_fingerprint"]),
            temperature=float(payload["temperature"]),
            shards=tuple(TeacherKDShardEntry.from_dict(item) for item in raw_shards),
            top_k=int(payload.get("top_k", 0)),
            schema_version=int(payload.get("schema_version", 0)),
            kind=str(payload.get("kind", "")),
            topk_logits_are_raw=bool(payload.get("topk_logits_are_raw", False)),
            normalization_definition=str(payload.get("normalization_definition", "")),
        )
        if int(payload.get("sequence_count", -1)) != manifest.sequence_count:
            raise TeacherKDSchemaError("KD corpus sequence total mismatch")
        if int(payload.get("token_count", -1)) != manifest.token_count:
            raise TeacherKDSchemaError("KD corpus token total mismatch")
        return manifest


def validate_kd_corpus_coverage(
    corpus: TeacherKDCorpusManifest,
    prepared_corpus: Any,
) -> None:
    """Require a KD corpus to cover every prepared shard exactly once."""

    if corpus.dataset_fingerprint != prepared_corpus.dataset_fingerprint:
        raise TeacherKDSchemaError("KD/prepared dataset fingerprint mismatch")
    if corpus.tokenizer_sha256 != prepared_corpus.tokenizer_sha256:
        raise TeacherKDSchemaError("KD/prepared tokenizer identity mismatch")
    expected = tuple(
        (
            entry.shard_id,
            entry.tensors_sha256,
            entry.global_sample_start,
            entry.global_sample_end,
            entry.global_token_start,
            entry.global_token_end,
            entry.sequence_count,
            entry.token_count,
        )
        for entry in prepared_corpus.shards
    )
    actual = tuple(
        (
            entry.source_shard_id,
            entry.source_tensors_sha256,
            entry.global_sample_start,
            entry.global_sample_end,
            entry.global_token_start,
            entry.global_token_end,
            entry.sequence_count,
            entry.token_count,
        )
        for entry in corpus.shards
    )
    if actual != expected:
        raise TeacherKDSchemaError(
            "KD corpus does not exactly cover the prepared shard manifest; "
            f"expected {len(expected)} shards, got {len(actual)}"
        )


def write_kd_corpus_manifest(
    output_path: str | os.PathLike[str],
    shard_directories: Sequence[str | os.PathLike[str]],
    *,
    expected_temperature: float,
    prepared_corpus: Any | None = None,
) -> Path:
    """Build a deterministic corpus lock from already COMPLETE KD shards."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    collected: list[tuple[Path, TeacherKDManifest]] = []
    for raw_path in shard_directories:
        shard = Path(raw_path)
        if shard.name.endswith(".incomplete"):
            raise TeacherKDSchemaError(f"refusing KD staging directory: {shard}")
        manifest = validate_kd_shard(
            shard, expected_temperature=expected_temperature
        )
        collected.append((shard, manifest))
    if not collected:
        raise TeacherKDSchemaError("cannot build an empty KD corpus")
    collected.sort(key=lambda item: item[1].global_sample_start)
    first = collected[0][1]
    entries: list[TeacherKDShardEntry] = []
    for shard, manifest in collected:
        if (
            manifest.teacher_model_id != first.teacher_model_id
            or manifest.teacher_revision != first.teacher_revision
            or manifest.teacher_model_sha256 != first.teacher_model_sha256
            or manifest.generator_source_sha256 != first.generator_source_sha256
            or manifest.tokenizer_sha256 != first.tokenizer_sha256
            or manifest.dataset_fingerprint != first.dataset_fingerprint
            or manifest.temperature != first.temperature
        ):
            raise TeacherKDSchemaError("KD shards do not share one teacher/dataset contract")
        try:
            relative = shard.resolve().relative_to(output.parent.resolve()).as_posix()
        except ValueError as error:
            raise TeacherKDSchemaError(
                f"KD shard must be below corpus manifest directory: {shard}"
            ) from error
        if manifest.tensors_sha256 is None:  # guaranteed by validate, for narrowing
            raise TeacherKDSchemaError(f"KD shard has no tensor hash: {shard}")
        entries.append(
            TeacherKDShardEntry(
                path=relative,
                source_shard_id=manifest.source_shard_id,
                source_tensors_sha256=manifest.source_tensors_sha256,
                manifest_sha256=sha256_file(shard / KD_MANIFEST_FILENAME),
                tensors_sha256=manifest.tensors_sha256,
                global_sample_start=manifest.global_sample_start,
                global_sample_end=manifest.global_sample_end,
                global_token_start=manifest.global_token_start,
                global_token_end=manifest.global_token_end,
                sequence_count=manifest.sequence_count,
                token_count=manifest.token_count,
            )
        )
    corpus = TeacherKDCorpusManifest(
        teacher_model_id=first.teacher_model_id,
        teacher_revision=first.teacher_revision,
        teacher_model_sha256=first.teacher_model_sha256,
        generator_source_sha256=first.generator_source_sha256,
        tokenizer_sha256=first.tokenizer_sha256,
        dataset_fingerprint=first.dataset_fingerprint,
        temperature=first.temperature,
        shards=tuple(entries),
    )
    if prepared_corpus is not None:
        validate_kd_corpus_coverage(corpus, prepared_corpus)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(corpus.to_dict(), file, ensure_ascii=False, sort_keys=True, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, output)
    directory_fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return output


def validate_kd_corpus_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    expected_temperature: float,
    verify_shards: bool = True,
) -> TeacherKDCorpusManifest:
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as file:
        corpus = TeacherKDCorpusManifest.from_dict(json.load(file))
    if corpus.generator_source_sha256 != KD_GENERATOR_SOURCE_SHA256:
        raise TeacherKDSchemaError(
            "KD generator source changed; regenerate the KD corpus instead of "
            "reusing stale output"
        )
    corpus.require_temperature(expected_temperature)
    if not verify_shards:
        return corpus
    root = path.parent.resolve()
    for entry in corpus.shards:
        shard = (root / entry.path).resolve()
        try:
            shard.relative_to(root)
        except ValueError as error:
            raise TeacherKDSchemaError(f"KD shard escapes corpus root: {entry.path}") from error
        if sha256_file(shard / KD_MANIFEST_FILENAME) != entry.manifest_sha256:
            raise TeacherKDSchemaError(f"KD shard manifest hash mismatch: {entry.path}")
        shard_manifest = validate_kd_shard(
            shard, expected_temperature=expected_temperature
        )
        expected_identity = (
            corpus.teacher_model_id,
            corpus.teacher_revision,
            corpus.teacher_model_sha256,
            corpus.generator_source_sha256,
            corpus.tokenizer_sha256,
            corpus.dataset_fingerprint,
            entry.source_shard_id,
            entry.source_tensors_sha256,
            entry.global_sample_start,
            entry.global_sample_end,
            entry.global_token_start,
            entry.global_token_end,
            entry.sequence_count,
            entry.token_count,
            entry.tensors_sha256,
        )
        actual_identity = (
            shard_manifest.teacher_model_id,
            shard_manifest.teacher_revision,
            shard_manifest.teacher_model_sha256,
            shard_manifest.generator_source_sha256,
            shard_manifest.tokenizer_sha256,
            shard_manifest.dataset_fingerprint,
            shard_manifest.source_shard_id,
            shard_manifest.source_tensors_sha256,
            shard_manifest.global_sample_start,
            shard_manifest.global_sample_end,
            shard_manifest.global_token_start,
            shard_manifest.global_token_end,
            shard_manifest.sequence_count,
            shard_manifest.token_count,
            shard_manifest.tensors_sha256,
        )
        if actual_identity != expected_identity:
            raise TeacherKDSchemaError(f"KD corpus/shard identity mismatch: {entry.path}")
    return corpus


class TeacherKDCorpus:
    """Validated, one-shard-at-a-time iterator for training input pipelines."""

    def __init__(
        self,
        manifest_path: str | os.PathLike[str],
        *,
        expected_temperature: float,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.expected_temperature = expected_temperature
        self.manifest = validate_kd_corpus_manifest(
            self.manifest_path,
            expected_temperature=expected_temperature,
        )

    def iter_batches(self) -> Iterator[TeacherKDBatch]:
        for entry in self.manifest.shards:
            _, batch = load_kd_shard(
                self.manifest_path.parent / entry.path,
                expected_temperature=self.expected_temperature,
                # Construction already verified every shard exactly once.
                verify_checksum=False,
            )
            yield batch

    def __iter__(self) -> Iterator[TeacherKDRecord]:
        for batch in self.iter_batches():
            for index in range(len(batch)):
                yield batch.record(index)


def iter_kd_shard_datasets(
    root: str | os.PathLike[str],
    *,
    expected_temperature: float,
) -> Iterator[TeacherKDShardDataset]:
    """Yield complete KD shards in stable lexical order, ignoring partials."""

    for path in sorted(Path(root).iterdir()):
        if path.name.endswith(".incomplete"):
            continue
        if is_shard_complete(path, verify_hashes=False) and (
            path / KD_MANIFEST_FILENAME
        ).is_file():
            yield TeacherKDShardDataset(
                path, expected_temperature=expected_temperature
            )
