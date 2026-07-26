"""Restartable, lineage-checked calibration pipelines.

All model execution in this module is inference-only.  The commands are meant
to be launched by the user: importing the module never initializes CUDA or
loads a checkpoint.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_train_config
from .data import ShardTransaction, is_shard_complete, read_complete_marker
from .data.prepared import PREPARED_TENSORS, validate_prepared_corpus
from .io.locking import FileLock
from .model_loading import (
    freeze_module,
    load_donor_mlp_weights,
    load_qwen35_text_model,
)
from .progress import TaskProgress
from .utils import atomic_write_json, sha256_file

ACTIVATION_SCHEMA_VERSION = 2
CALIBRATION_SCHEMA_VERSION = 2
SAMPLER_ALGORITHM = "numpy-pcg64-choice-without-replacement-v1"
CKA_SAMPLE_LIMIT = 2048


class CalibrationStopped(RuntimeError):
    """A STOP file was consumed at an artifact-safe boundary."""


@dataclass(frozen=True, slots=True)
class _InputInspection:
    index: int
    path: Path
    sha256: str
    sequence_count: int
    sequence_length: int
    valid_tokens: int
    global_valid_start: int
    global_valid_end: int
    selected_flat_positions: Any

    @property
    def shard_id(self) -> str:
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", self.path.stem).strip(".-") or "input"
        return f"{self.index:06d}-{stem[:40]}-{self.sha256[:12]}"


@dataclass(frozen=True, slots=True)
class _ActivationPart:
    part_id: str
    path: Path
    sample_count: int
    batch_start: int
    batch_end: int
    selection_sha256: str
    artifact_sha256: str


@dataclass(frozen=True, slots=True)
class _ActivationCorpus:
    label: str
    source_path: Path
    source_sha256: str
    lineage: Mapping[str, Any]
    parts: tuple[_ActivationPart, ...]
    total_samples: int
    legacy: bool = False


@dataclass(frozen=True, slots=True)
class _PairedActivations:
    student: _ActivationCorpus
    donor: _ActivationCorpus
    parts: tuple[tuple[str, Path, Path], ...]
    total_samples: int
    fingerprint: str


def _resolve_collection_inputs(
    input_paths: str | Sequence[str] | None,
    prepared_manifest: str | Path | None,
) -> tuple[tuple[Path, ...], dict[str, Any]]:
    """Resolve exactly one input mode and authenticate prepared-corpus lineage."""

    if isinstance(input_paths, (str, os.PathLike)):
        input_paths = [str(input_paths)]
    explicit = tuple(input_paths or ())
    if bool(explicit) == bool(prepared_manifest):
        raise ValueError(
            "calibrate collect requires exactly one of --input or --prepared-manifest"
        )
    if explicit:
        paths = tuple(Path(path).resolve() for path in explicit)
        for path in paths:
            if not path.is_file():
                raise ValueError(f"calibration input does not exist: {path}")
        return paths, {"kind": "explicit_safetensors"}

    manifest_path = Path(str(prepared_manifest)).resolve()
    corpus = validate_prepared_corpus(manifest_path)
    root = manifest_path.parent.resolve()
    paths: list[Path] = []
    shards: list[dict[str, Any]] = []
    for entry in corpus.shards:
        path = (root / entry.path / PREPARED_TENSORS).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"prepared calibration shard escapes corpus root: {entry.path}") from error
        if not path.is_file():
            raise ValueError(f"prepared calibration tensor is missing: {path}")
        paths.append(path)
        shards.append(
            {
                "shard_id": entry.shard_id,
                "path": path.relative_to(root).as_posix(),
                "tensors_sha256": entry.tensors_sha256,
                "sequence_count": entry.sequence_count,
                "token_count": entry.token_count,
            }
        )
    if not paths:
        raise ValueError("prepared calibration corpus contains no shards")
    return tuple(paths), {
        "kind": "prepared_corpus",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_fingerprint": corpus.dataset_fingerprint,
        "tokenizer_sha256": corpus.tokenizer_sha256,
        "sequence_length": corpus.sequence_length,
        "shards": shards,
    }


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: Any) -> str:
    import numpy as np

    array = np.asarray(value, dtype="<i8").reshape(-1)
    digest = hashlib.sha256()
    digest.update(len(array).to_bytes(8, "little", signed=False))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _check_stop_file(stop_file: str | Path | None) -> None:
    if stop_file is None:
        return
    target = Path(stop_file)
    if not target.is_file():
        return
    with suppress(FileNotFoundError):
        target.unlink()
    raise CalibrationStopped(
        f"calibration stopped safely at an artifact boundary after consuming {target}"
    )


def _save_safetensors_atomic(
    tensors: Mapping[str, Any],
    output: str | Path,
    *,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    from safetensors.torch import save_file

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    # A process may have died between writing and replacing the target.  The
    # enclosing artifact lock makes every old temporary for this target stale.
    for stale in target.parent.glob(f".{target.name}.*.incomplete"):
        stale.unlink(missing_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.incomplete")
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()},
        str(temporary),
        metadata=dict(metadata or {}),
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    directory_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return target


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _model_lineage(source: Any) -> dict[str, Any]:
    root = Path(source.local_path).resolve()
    manifest = root / "download-manifest.json"
    model_config = root / "config.json"
    if not root.is_dir() or not manifest.is_file() or not model_config.is_file():
        raise ValueError(f"model source is incomplete: {root}")
    manifest_sha = sha256_file(manifest)
    if manifest_sha.lower() != source.manifest_sha256.lower():
        raise ValueError(
            f"model manifest SHA256 mismatch for {source.model_id}: "
            f"expected {source.manifest_sha256}, got {manifest_sha}"
        )
    return {
        "model_id": source.model_id,
        "revision": source.revision,
        "local_path": str(root),
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha,
        "config_path": str(model_config),
        "config_sha256": sha256_file(model_config),
    }


def _architecture_lineage(config: Any) -> dict[str, Any]:
    architecture = dataclasses.asdict(config.architecture)
    # Artifact destinations do not describe the source architecture.
    for name in ("layer_map_path", "channel_map_path", "adapter_init_path"):
        architecture.pop(name, None)
    return architecture


def _audit_local_sources(config: Any) -> None:
    """Hash every frozen source shard before producing calibration artifacts."""

    from .modeling import audit_source_configs
    from .preflight import _check_source

    _check_source("backbone", config.sources.backbone)
    _check_source("donor", config.sources.donor)
    audit_source_configs(
        config.sources.backbone.local_path,
        config.sources.donor.local_path,
    )


def _input_shape(handle: Any, name: str, path: Path) -> tuple[int, int]:
    if name not in set(handle.keys()):
        raise ValueError(f"calibration input {path} is missing {name}")
    shape = tuple(int(item) for item in handle.get_slice(name).get_shape())
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"calibration input {path} {name} must have shape [sequences, tokens]")
    return shape


def _planned_collection_parts(paths: Sequence[Path], batch_size: int) -> int:
    """Read safetensors headers only; full hashes/masks are scanned once in the plan."""

    from safetensors import safe_open

    total = 0
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            sequences, _ = _input_shape(handle, "input_ids", path)
        total += math.ceil(sequences / batch_size)
    return 2 * total


def _scan_inputs(
    paths: Sequence[Path],
    *,
    scan_batch_size: int,
    trusted_sha256: Sequence[str] | None = None,
    progress_bar: TaskProgress | None = None,
) -> list[dict[str, Any]]:
    from safetensors import safe_open

    if trusted_sha256 is not None and len(trusted_sha256) != len(paths):
        raise ValueError("trusted calibration input hashes do not cover every path")
    result: list[dict[str, Any]] = []
    global_start = 0
    for index, path in enumerate(paths):
        with safe_open(path, framework="pt", device="cpu") as handle:
            input_shape = _input_shape(handle, "input_ids", path)
            mask_shape = _input_shape(handle, "attention_mask", path)
            if input_shape != mask_shape:
                raise ValueError(f"calibration input shapes differ: {path}")
            mask_slice = handle.get_slice("attention_mask")
            valid_tokens = 0
            for start in range(0, input_shape[0], scan_batch_size):
                mask = mask_slice[start : start + scan_batch_size]
                valid_tokens += int(mask.ne(0).sum().item())
                if progress_bar is not None:
                    progress_bar.set_postfix(
                        {
                            "phase": "scan",
                            "input": index,
                            "sequence": min(start + scan_batch_size, input_shape[0]),
                        }
                    )
                    progress_bar.update()
        result.append(
            {
                "index": index,
                "path": path,
                "sha256": (
                    str(trusted_sha256[index])
                    if trusted_sha256 is not None
                    else sha256_file(path)
                ),
                "sequence_count": input_shape[0],
                "sequence_length": input_shape[1],
                "valid_tokens": valid_tokens,
                "global_valid_start": global_start,
                "global_valid_end": global_start + valid_tokens,
            }
        )
        global_start += valid_tokens
    return result


def _deterministic_indices(total: int, limit: int, seed: int) -> Any:
    import numpy as np

    if total < 0 or limit < 0:
        raise ValueError("sample sizes cannot be negative")
    count = min(total, limit)
    if count == total:
        return np.arange(total, dtype=np.int64)
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    selected = generator.choice(total, size=count, replace=False, shuffle=False)
    return np.sort(selected.astype(np.int64, copy=False))


def _assign_selected_coordinates(
    raw_inputs: Sequence[Mapping[str, Any]],
    selected: Any,
    *,
    scan_batch_size: int,
    progress_bar: TaskProgress | None = None,
) -> tuple[_InputInspection, ...]:
    import numpy as np
    from safetensors import safe_open

    inspections: list[_InputInspection] = []
    selected = np.asarray(selected, dtype=np.int64)
    for item in raw_inputs:
        begin = int(item["global_valid_start"])
        end = int(item["global_valid_end"])
        left = int(np.searchsorted(selected, begin, side="left"))
        right = int(np.searchsorted(selected, end, side="left"))
        local_valid_ordinals = selected[left:right] - begin
        local_cursor = 0
        flat_parts: list[Any] = []
        path = Path(item["path"])
        with safe_open(path, framework="pt", device="cpu") as handle:
            mask_slice = handle.get_slice("attention_mask")
            for batch_start in range(0, int(item["sequence_count"]), scan_batch_size):
                mask = mask_slice[batch_start : batch_start + scan_batch_size]
                valid_flat = np.flatnonzero(mask.ne(0).numpy().reshape(-1))
                next_cursor = local_cursor + len(valid_flat)
                lo = int(np.searchsorted(local_valid_ordinals, local_cursor, side="left"))
                hi = int(np.searchsorted(local_valid_ordinals, next_cursor, side="left"))
                if hi > lo:
                    offsets = local_valid_ordinals[lo:hi] - local_cursor
                    positions = valid_flat[offsets]
                    row = positions // int(item["sequence_length"])
                    column = positions % int(item["sequence_length"])
                    flat_parts.append(
                        (batch_start + row) * int(item["sequence_length"]) + column
                    )
                local_cursor = next_cursor
                if progress_bar is not None:
                    progress_bar.set_postfix(
                        {
                            "phase": "sample-plan",
                            "input": int(item["index"]),
                            "sequence": min(
                                batch_start + scan_batch_size,
                                int(item["sequence_count"]),
                            ),
                        }
                    )
                    progress_bar.update()
        flat = (
            np.concatenate(flat_parts).astype(np.int64, copy=False)
            if flat_parts
            else np.empty((0,), dtype=np.int64)
        )
        if len(flat) != len(local_valid_ordinals):
            raise RuntimeError(f"failed to map sampled coordinates for {path}")
        inspections.append(
            _InputInspection(
                index=int(item["index"]),
                path=path,
                sha256=str(item["sha256"]),
                sequence_count=int(item["sequence_count"]),
                sequence_length=int(item["sequence_length"]),
                valid_tokens=int(item["valid_tokens"]),
                global_valid_start=begin,
                global_valid_end=end,
                selected_flat_positions=flat,
            )
        )
    return tuple(inspections)


def _build_collection_plan(
    config_path: str | Path,
    config: Any,
    inputs: Sequence[Path],
    *,
    batch_size: int,
    max_samples: int,
    sample_seed: int,
    dtype_name: str,
    device_type: str,
    input_source: Mapping[str, Any],
    progress_bar: TaskProgress | None = None,
) -> tuple[dict[str, Any], tuple[_InputInspection, ...]]:
    raw_shards = input_source.get("shards")
    trusted_sha256 = (
        tuple(str(item["tensors_sha256"]) for item in raw_shards)
        if input_source.get("kind") == "prepared_corpus" and isinstance(raw_shards, list)
        else None
    )
    raw_inputs = _scan_inputs(
        inputs,
        scan_batch_size=batch_size,
        trusted_sha256=trusted_sha256,
        progress_bar=progress_bar,
    )
    total_valid = sum(int(item["valid_tokens"]) for item in raw_inputs)
    if total_valid <= 0:
        raise ValueError("calibration corpus has no unmasked tokens")
    selected = _deterministic_indices(total_valid, max_samples, sample_seed)
    inspections = _assign_selected_coordinates(
        raw_inputs,
        selected,
        scan_batch_size=batch_size,
        progress_bar=progress_bar,
    )
    input_plan = [
        {
            "index": item.index,
            "shard_id": item.shard_id,
            "path": str(item.path),
            "sha256": item.sha256,
            "sequence_count": item.sequence_count,
            "sequence_length": item.sequence_length,
            "valid_tokens": item.valid_tokens,
            "global_valid_start": item.global_valid_start,
            "global_valid_end": item.global_valid_end,
            "selected_samples": len(item.selected_flat_positions),
            "selection_sha256": _array_sha256(item.selected_flat_positions),
        }
        for item in inspections
    ]
    payload: dict[str, Any] = {
        "schema_version": ACTIVATION_SCHEMA_VERSION,
        "kind": "paired_pre_ffn_activation_plan",
        "train_config_path": str(Path(config_path).resolve()),
        "train_config_sha256": sha256_file(config_path),
        "track": config.track,
        "architecture": _architecture_lineage(config),
        "dtype": dtype_name,
        "device_type": device_type,
        "batch_size_sequences": batch_size,
        "activation_storage": "per-input-shard-compacted-v1",
        "sampler": {
            "algorithm": SAMPLER_ALGORITHM,
            "sample_seed": int(sample_seed),
            "max_samples_global": int(max_samples),
            "total_valid_tokens": total_valid,
            "selected_samples": len(selected),
            "coordinate_order": "input-sequence-token",
        },
        "sources": {
            "student": _model_lineage(config.sources.backbone),
            "donor": _model_lineage(config.sources.donor),
        },
        "input_source": dict(input_source),
        "ordered_inputs": input_plan,
    }
    payload["plan_fingerprint"] = _canonical_sha256(payload)
    return payload, inspections


def _install_or_validate_plan(output: Path, plan: Mapping[str, Any]) -> Path:
    path = output / "PLAN.json"
    if path.exists():
        existing = _load_json_object(path, label="calibration PLAN")
        if existing != plan:
            raise ValueError(
                f"calibration output {output} belongs to an incompatible PLAN; "
                "use a new output directory"
            )
    else:
        # Never adopt a pre-existing unplanned output tree.
        for stale in output.glob(".PLAN.json.*.tmp"):
            stale.unlink(missing_ok=True)
        unexpected = [item for item in output.iterdir() if item.name != ".collect.lock"]
        if unexpected:
            raise ValueError(
                f"calibration output contains state but no PLAN: {unexpected[0]}"
            )
        atomic_write_json(path, dict(plan))
    return path


def _part_coordinates(item: _InputInspection, batch_start: int, batch_end: int) -> Any:
    import numpy as np

    length = item.sequence_length
    flat = item.selected_flat_positions
    first = int(np.searchsorted(flat, batch_start * length, side="left"))
    last = int(np.searchsorted(flat, batch_end * length, side="left"))
    selected = flat[first:last]
    rows = selected // length - batch_start
    columns = selected % length
    if len(selected) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.stack((rows, columns), axis=1).astype(np.int64, copy=False)


def _validate_activation_file(
    path: Path,
    expected_layers: int,
    *,
    expected_rows: int | None = None,
    expected_hidden: int | None = None,
    expected_metadata: Mapping[str, str] | None = None,
) -> int:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        expected = {f"layers.{layer}" for layer in range(expected_layers)}
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            raise ValueError(
                f"activation file {path} tensor set differs; "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
        if expected_metadata is not None:
            metadata = handle.metadata() or {}
            for name, value in expected_metadata.items():
                if metadata.get(name) != value:
                    raise ValueError(f"activation file {path} metadata {name} mismatch")
        rows: int | None = None
        for name in sorted(expected):
            value = handle.get_slice(name)
            shape = tuple(int(item) for item in value.get_shape())
            if len(shape) != 2:
                raise ValueError(f"activation tensor {name} in {path} must be rank-2")
            if rows is None:
                rows = shape[0]
            elif shape[0] != rows:
                raise ValueError(f"activation layers in {path} have different sample counts")
            if expected_hidden is not None and shape[1] != expected_hidden:
                raise ValueError(
                    f"activation tensor {name} in {path} has hidden {shape[1]}, "
                    f"expected {expected_hidden}"
                )
    result = int(rows or 0)
    if expected_rows is not None and result != expected_rows:
        raise ValueError(f"activation file {path} has {result} rows, expected {expected_rows}")
    return result


def _capture_model(
    model: Any,
    batch: Mapping[str, Any],
    *,
    sample_coordinates: Any,
    device: str,
    dtype: Any,
) -> dict[str, Any]:
    import torch

    model.eval()
    body = getattr(model, "model", model)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    coordinates = torch.as_tensor(sample_coordinates, dtype=torch.long, device=device)
    captures: list[Any | None] = [None] * len(body.layers)
    hooks = []
    for layer_index, layer in enumerate(body.layers):

        def hook(_module: Any, inputs: tuple[Any, ...], index: int = layer_index) -> None:
            hidden = inputs[0].detach()
            if coordinates.numel() == 0:
                captures[index] = hidden.reshape(-1, hidden.shape[-1])[:0].cpu()
            else:
                captures[index] = hidden[coordinates[:, 0], coordinates[:, 1]].cpu()

        hooks.append(layer.mlp.register_forward_pre_hook(hook))
    try:
        device_type = "cuda" if str(device).startswith("cuda") else "cpu"
        with torch.inference_mode(), torch.autocast(
            device_type=device_type,
            dtype=dtype,
            enabled=dtype == torch.bfloat16,
        ):
            # Calling the text body avoids vocabulary-sized logits.
            body(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for item in hooks:
            item.remove()
    if any(value is None for value in captures):
        raise RuntimeError("failed to capture every FFN input")
    return {f"layers.{index}": value for index, value in enumerate(captures)}


def _part_metadata(
    *,
    plan_fingerprint: str,
    label: str,
    item: _InputInspection,
    part_index: int,
    batch_start: int,
    batch_end: int,
    coordinates: Any,
) -> dict[str, str]:
    return {
        "schema_version": str(ACTIVATION_SCHEMA_VERSION),
        "kind": "pre_ffn_activation_part",
        "plan_fingerprint": plan_fingerprint,
        "model_side": label,
        "input_sha256": item.sha256,
        "part_index": str(part_index),
        "batch_start": str(batch_start),
        "batch_end": str(batch_end),
        "sample_count": str(len(coordinates)),
        "selection_sha256": _array_sha256(coordinates),
    }


def _compacted_part_metadata(
    *, plan_fingerprint: str, label: str, item: _InputInspection
) -> dict[str, str]:
    return {
        "schema_version": str(ACTIVATION_SCHEMA_VERSION),
        "kind": "pre_ffn_activation_shard",
        "plan_fingerprint": plan_fingerprint,
        "model_side": label,
        "input_sha256": item.sha256,
        "sample_count": str(len(item.selected_flat_positions)),
        "selection_sha256": _array_sha256(item.selected_flat_positions),
    }


def _compacted_part_record(item: _InputInspection) -> dict[str, Any]:
    return {
        "part_id": f"{item.shard_id}/samples",
        "path": "activations.safetensors",
        "artifact_kind": "pre_ffn_activation_shard",
        "batch_start": 0,
        "batch_end": item.sequence_count,
        "sample_count": len(item.selected_flat_positions),
        "selection_sha256": _array_sha256(item.selected_flat_positions),
    }


def _side_shard_needs_model(
    item: _InputInspection,
    transaction: ShardTransaction,
    *,
    plan_fingerprint: str,
    label: str,
    expected_layers: int,
    expected_hidden: int,
    batch_size: int,
) -> bool:
    """Return whether an unfinished side shard must execute model forwards.

    A crash can happen after the compact activation file is durable but before
    ``ShardTransaction.commit`` installs ``COMPLETE``.  It can also happen
    after every sampled microbatch is durable but before compaction.  Both
    states are fully recoverable from authenticated files already in the
    transaction directory, so loading a multi-billion-parameter model would be
    unnecessary (and can itself prevent an otherwise cheap resume).
    """

    from safetensors import SafetensorError

    compacted_path = transaction.work_directory / "activations.safetensors"
    if compacted_path.is_file():
        try:
            _validate_activation_file(
                compacted_path,
                expected_layers,
                expected_rows=len(item.selected_flat_positions),
                expected_hidden=expected_hidden,
                expected_metadata=_compacted_part_metadata(
                    plan_fingerprint=plan_fingerprint,
                    label=label,
                    item=item,
                ),
            )
        except (OSError, ValueError, SafetensorError):
            pass
        else:
            return False

    parts_directory = transaction.work_directory / "parts"
    for part_index, batch_start in enumerate(range(0, item.sequence_count, batch_size)):
        batch_end = min(item.sequence_count, batch_start + batch_size)
        coordinates = _part_coordinates(item, batch_start, batch_end)
        if len(coordinates) == 0:
            continue
        target = parts_directory / f"part-{part_index:06d}.safetensors"
        if not target.is_file():
            return True
        try:
            _validate_activation_file(
                target,
                expected_layers,
                expected_rows=len(coordinates),
                expected_hidden=expected_hidden,
                expected_metadata=_part_metadata(
                    plan_fingerprint=plan_fingerprint,
                    label=label,
                    item=item,
                    part_index=part_index,
                    batch_start=batch_start,
                    batch_end=batch_end,
                    coordinates=coordinates,
                ),
            )
        except (OSError, ValueError, SafetensorError):
            return True
    return False


def _collect_side_shard(
    model: Any,
    item: _InputInspection,
    transaction: ShardTransaction,
    *,
    plan: Mapping[str, Any],
    label: str,
    expected_layers: int,
    expected_hidden: int,
    batch_size: int,
    device: str,
    dtype: Any,
    stop_file: str | Path | None,
    progress_bar: TaskProgress,
) -> None:
    from safetensors import SafetensorError, safe_open

    parts_directory = transaction.work_directory / "parts"
    parts_directory.mkdir(parents=True, exist_ok=True)
    compacted_path = transaction.work_directory / "activations.safetensors"
    compacted_metadata = _compacted_part_metadata(
        plan_fingerprint=str(plan["plan_fingerprint"]),
        label=label,
        item=item,
    )
    compacted_record = _compacted_part_record(item)
    if compacted_path.is_file():
        try:
            _validate_activation_file(
                compacted_path,
                expected_layers,
                expected_rows=len(item.selected_flat_positions),
                expected_hidden=expected_hidden,
                expected_metadata=compacted_metadata,
            )
        except (OSError, ValueError, SafetensorError):
            compacted_path.unlink(missing_ok=True)
        else:
            # A process can stop after compaction but before the shard marker is
            # installed.  Adopt the authenticated compact file without loading
            # either model or replaying already durable microbatches.
            for micro_part in parts_directory.glob("*.safetensors"):
                micro_part.unlink(missing_ok=True)
            with suppress(OSError):
                parts_directory.rmdir()
            progress_bar.set_postfix(
                {"phase": "compact-resume", "side": label, "input": item.index}
            )
            progress_bar.update(math.ceil(item.sequence_count / batch_size))
            transaction.commit(
                {
                    "kind": "pre_ffn_activation_side_shard",
                    "model_side": label,
                    "plan_fingerprint": plan["plan_fingerprint"],
                    "input_path": str(item.path),
                    "input_sha256": item.sha256,
                    "sample_count": len(item.selected_flat_positions),
                    "parts": [compacted_record],
                }
            )
            return
    part_records: list[dict[str, Any]] = []
    with safe_open(item.path, framework="pt", device="cpu") as handle:
        ids = handle.get_slice("input_ids")
        mask = handle.get_slice("attention_mask")
        for part_index, batch_start in enumerate(range(0, item.sequence_count, batch_size)):
            batch_end = min(item.sequence_count, batch_start + batch_size)
            coordinates = _part_coordinates(item, batch_start, batch_end)
            metadata = _part_metadata(
                plan_fingerprint=str(plan["plan_fingerprint"]),
                label=label,
                item=item,
                part_index=part_index,
                batch_start=batch_start,
                batch_end=batch_end,
                coordinates=coordinates,
            )
            target = parts_directory / f"part-{part_index:06d}.safetensors"
            valid = False
            if target.is_file():
                try:
                    _validate_activation_file(
                        target,
                        expected_layers,
                        expected_rows=len(coordinates),
                        expected_hidden=expected_hidden,
                        expected_metadata=metadata,
                    )
                    valid = True
                except (OSError, ValueError, SafetensorError):
                    # The target is inside an uncommitted transaction; a torn or
                    # corrupt microbatch is safe to replace.
                    target.unlink(missing_ok=True)
            for stale in parts_directory.glob(f".{target.name}.*.incomplete"):
                stale.unlink(missing_ok=True)
            if not valid and len(coordinates) > 0:
                if model is None:
                    raise RuntimeError("calibration model is unavailable for sampled tokens")
                batch = {
                    "input_ids": ids[batch_start:batch_end],
                    "attention_mask": mask[batch_start:batch_end],
                }
                captures = _capture_model(
                    model,
                    batch,
                    sample_coordinates=coordinates,
                    device=device,
                    dtype=dtype,
                )
                _save_safetensors_atomic(captures, target, metadata=metadata)
                _validate_activation_file(
                    target,
                    expected_layers,
                    expected_rows=len(coordinates),
                    expected_hidden=expected_hidden,
                    expected_metadata=metadata,
                )
            part_records.append(
                {
                    "part_id": f"{item.shard_id}/part-{part_index:06d}",
                    "path": target.relative_to(transaction.work_directory).as_posix(),
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                    "sample_count": len(coordinates),
                    "selection_sha256": metadata["selection_sha256"],
                }
            )
            progress_bar.set_postfix(
                {
                    "phase": "forward" if len(coordinates) else "empty-skip",
                    "side": label,
                    "input": item.index,
                    "part": part_index,
                    "samples": len(coordinates),
                }
            )
            progress_bar.update()
            _check_stop_file(stop_file)
    if sum(int(part["sample_count"]) for part in part_records) != len(
        item.selected_flat_positions
    ):
        raise RuntimeError(f"activation sample accounting failed for {item.path}")
    # Microbatch files are the fine-grained crash-recovery journal.  Downstream
    # CKA/ridge would otherwise reopen one tiny safetensors file per sequence
    # for every layer (hundreds of thousands of opens at 24×32 Base scale).
    # Compact once per prepared input shard before installing COMPLETE.
    compacted: dict[str, Any] = {}
    for layer in range(expected_layers):
        values = []
        for part in part_records:
            if int(part["sample_count"]) == 0:
                continue
            path = transaction.work_directory / str(part["path"])
            with safe_open(path, framework="pt", device="cpu") as handle:
                value = handle.get_tensor(f"layers.{layer}")
            if value.shape[0] > 0:
                values.append(value)
        if values:
            import torch

            compacted[f"layers.{layer}"] = (
                values[0] if len(values) == 1 else torch.cat(values, dim=0)
            )
        else:
            import torch

            compacted[f"layers.{layer}"] = torch.empty(
                (0, expected_hidden), dtype=dtype, device="cpu"
            )
    _save_safetensors_atomic(
        compacted,
        compacted_path,
        metadata=compacted_metadata,
    )
    _validate_activation_file(
        compacted_path,
        expected_layers,
        expected_rows=len(item.selected_flat_positions),
        expected_hidden=expected_hidden,
        expected_metadata=compacted_metadata,
    )
    for part in part_records:
        (transaction.work_directory / str(part["path"])).unlink(missing_ok=True)
    with suppress(OSError):
        parts_directory.rmdir()
    transaction.commit(
        {
            "kind": "pre_ffn_activation_side_shard",
            "model_side": label,
            "plan_fingerprint": plan["plan_fingerprint"],
            "input_path": str(item.path),
            "input_sha256": item.sha256,
            "sample_count": len(item.selected_flat_positions),
            "parts": [compacted_record],
        }
    )


def _side_fingerprint(plan: Mapping[str, Any], label: str) -> str:
    return _canonical_sha256(
        {
            "schema_version": ACTIVATION_SCHEMA_VERSION,
            "plan_fingerprint": plan["plan_fingerprint"],
            "model_side": label,
            "source": plan["sources"][label],
        }
    )


def _side_manifest_entry(
    output: Path,
    directory: Path,
    *,
    label: str,
    plan_fingerprint: str,
) -> dict[str, Any]:
    if not is_shard_complete(directory):
        raise ValueError(f"activation side shard is incomplete or corrupt: {directory}")
    marker = read_complete_marker(directory)
    metadata = marker.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"activation COMPLETE metadata is malformed: {directory}")
    if metadata.get("model_side") != label or metadata.get("plan_fingerprint") != plan_fingerprint:
        raise ValueError(f"activation COMPLETE lineage mismatch: {directory}")
    return {
        "path": directory.relative_to(output).as_posix(),
        "complete_sha256": sha256_file(directory / "COMPLETE"),
        "sample_count": int(metadata["sample_count"]),
        "parts": metadata["parts"],
    }


def collect_activations(
    config_path: str,
    input_paths: str | Sequence[str] | None,
    output_dir: str,
    *,
    prepared_manifest: str | None = None,
    device: str,
    max_samples: int,
    batch_size: int = 1,
    sample_seed: int = 3407,
    stop_file: str | None = None,
    progress: str = "auto",
) -> int:
    """Collect globally sampled, paired pre-FFN states with microbatch recovery."""

    # A pre-existing STOP is an operator request to avoid expensive corpus and
    # multi-gigabyte model validation altogether.  Later checks remain at every
    # artifact-safe boundary for requests created while the command is active.
    _check_stop_file(stop_file)
    import torch

    if max_samples <= 0 or batch_size <= 0:
        raise ValueError("collect requires max_samples > 0 and batch_size > 0")
    inputs, input_source = _resolve_collection_inputs(input_paths, prepared_manifest)
    config = load_train_config(config_path)
    if (
        input_source.get("kind") == "prepared_corpus"
        and input_source.get("tokenizer_sha256")
        != config.sources.tokenizer.manifest_sha256
    ):
        raise ValueError("prepared calibration tokenizer differs from the configured tokenizer")
    _audit_local_sources(config)
    dtype = torch.bfloat16 if config.runtime.bf16 else torch.float32
    dtype_name = "bfloat16" if config.runtime.bf16 else "float32"
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    total_parts = _planned_collection_parts(inputs, batch_size)
    with FileLock(output / ".collect.lock", timeout_seconds=300.0), TaskProgress(
        # Planning reads every attention mask twice: once to count global valid
        # tokens and once to map sampled ordinals back to coordinates.  Include
        # both passes so a large real corpus never looks stalled at 0%.
        total=total_parts * 2,
        description="calibrate-collect",
        unit="unit",
        mode=progress,
    ) as progress_bar:
        plan, inspections = _build_collection_plan(
            config_path,
            config,
            inputs,
            batch_size=batch_size,
            max_samples=max_samples,
            sample_seed=sample_seed,
            dtype_name=dtype_name,
            device_type=device_type,
            input_source=input_source,
            progress_bar=progress_bar,
        )
        plan_path = _install_or_validate_plan(output, plan)
        _check_stop_file(stop_file)

        for label, source, expected_layers, expected_hidden in (
            (
                "student",
                config.sources.backbone,
                config.architecture.student_layers,
                config.architecture.student_hidden_size,
            ),
            (
                "donor",
                config.sources.donor,
                config.architecture.donor_layers,
                config.architecture.donor_hidden_size,
            ),
        ):
            side_root = output / "sides" / label
            fingerprint = _side_fingerprint(plan, label)
            pending = False
            needs_model = False
            for item in inspections:
                with ShardTransaction(
                    side_root,
                    item.shard_id,
                    fingerprint=fingerprint,
                    source_fingerprint=item.sha256,
                ) as transaction:
                    if transaction.complete:
                        progress_bar.update(math.ceil(item.sequence_count / batch_size))
                    else:
                        pending = True
                        needs_model = needs_model or _side_shard_needs_model(
                            item,
                            transaction,
                            plan_fingerprint=str(plan["plan_fingerprint"]),
                            label=label,
                            expected_layers=expected_layers,
                            expected_hidden=expected_hidden,
                            batch_size=batch_size,
                        )
            if not pending:
                continue
            _check_stop_file(stop_file)
            model = (
                load_qwen35_text_model(source.local_path, dtype=dtype, device=device)
                if needs_model
                else None
            )
            if model is not None:
                freeze_module(model)
            try:
                for item in inspections:
                    with ShardTransaction(
                        side_root,
                        item.shard_id,
                        fingerprint=fingerprint,
                        source_fingerprint=item.sha256,
                    ) as transaction:
                        if transaction.complete:
                            continue
                        _collect_side_shard(
                            model,
                            item,
                            transaction,
                            plan=plan,
                            label=label,
                            expected_layers=expected_layers,
                            expected_hidden=expected_hidden,
                            batch_size=batch_size,
                            device=device,
                            dtype=dtype,
                            stop_file=stop_file,
                            progress_bar=progress_bar,
                        )
            finally:
                if model is not None:
                    del model
                if device_type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        shards = []
        for item in inspections:
            sides = {}
            for label in ("student", "donor"):
                directory = output / "sides" / label / item.shard_id
                sides[label] = _side_manifest_entry(
                    output,
                    directory,
                    label=label,
                    plan_fingerprint=str(plan["plan_fingerprint"]),
                )
            if sides["student"]["sample_count"] != sides["donor"]["sample_count"]:
                raise ValueError(f"paired activation counts differ for {item.shard_id}")
            student_parts = sides["student"]["parts"]
            donor_parts = sides["donor"]["parts"]
            if student_parts != donor_parts:
                raise ValueError(f"paired activation part plans differ for {item.shard_id}")
            shards.append(
                {
                    "shard_id": item.shard_id,
                    "input_path": str(item.path),
                    "input_sha256": item.sha256,
                    "sample_count": sides["student"]["sample_count"],
                    "sides": sides,
                }
            )

        manifest = {
            "schema_version": ACTIVATION_SCHEMA_VERSION,
            "kind": "paired_pre_ffn_activation_corpus",
            "plan_path": plan_path.relative_to(output).as_posix(),
            "plan_sha256": sha256_file(plan_path),
            "plan_fingerprint": plan["plan_fingerprint"],
            "lineage": {
                "train_config_path": plan["train_config_path"],
                "train_config_sha256": plan["train_config_sha256"],
                "track": plan["track"],
                "architecture": plan["architecture"],
                "dtype": plan["dtype"],
                "device_type": plan["device_type"],
                "activation_storage": plan["activation_storage"],
                "sampler": plan["sampler"],
                "sources": plan["sources"],
                "input_source": plan["input_source"],
                "ordered_inputs": plan["ordered_inputs"],
            },
            "total_samples": sum(int(item["sample_count"]) for item in shards),
            "shards": shards,
        }
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            try:
                existing = _load_json_object(manifest_path, label="activation manifest")
            except ValueError:
                existing = {}
            if existing.get("plan_fingerprint") not in (None, plan["plan_fingerprint"]):
                raise ValueError("existing activation manifest belongs to another PLAN")
        atomic_write_json(manifest_path, manifest)
    return 0


def _inventory_sha(marker: Mapping[str, Any], relative_path: str) -> str:
    outputs = marker.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("COMPLETE marker has no output inventory")
    for item in outputs:
        if isinstance(item, dict) and item.get("path") == relative_path:
            return str(item["sha256"])
    raise ValueError(f"COMPLETE inventory is missing {relative_path}")


def _activation_lineage(
    payload: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    raw = payload.get("lineage")
    if not isinstance(raw, Mapping):
        raise ValueError("activation corpus has no lineage object")
    # PLAN.json is the authenticated authority for every source and input that
    # produced the part files.  The manifest repeats that information only for
    # convenient inspection.  Reject a modified copy instead of allowing an old
    # Base/post-trained activation corpus to be relabelled by editing one JSON.
    expected_raw: dict[str, Any] = {
        "train_config_path": plan.get("train_config_path"),
        "train_config_sha256": plan.get("train_config_sha256"),
        "track": plan.get("track"),
        "architecture": plan.get("architecture"),
        "dtype": plan.get("dtype"),
        "device_type": plan.get("device_type"),
        "sampler": plan.get("sampler"),
        "sources": plan.get("sources"),
        "ordered_inputs": plan.get("ordered_inputs"),
    }
    if "activation_storage" in plan:
        expected_raw["activation_storage"] = plan.get("activation_storage")
    if "input_source" in plan:
        expected_raw["input_source"] = plan.get("input_source")
    if dict(raw) != expected_raw:
        raise ValueError("activation manifest lineage differs from its authenticated PLAN")
    lineage = {
        "schema_version": payload["schema_version"],
        "plan_fingerprint": payload["plan_fingerprint"],
        "plan_sha256": payload["plan_sha256"],
        "track": plan.get("track"),
        "architecture": plan.get("architecture"),
        "sampler": plan.get("sampler"),
        "sources": plan.get("sources"),
        "ordered_inputs": plan.get("ordered_inputs"),
    }
    if "activation_storage" in plan:
        lineage["activation_storage"] = plan.get("activation_storage")
    if "input_source" in plan:
        lineage["input_source"] = plan.get("input_source")
    if not isinstance(lineage["sources"], dict):
        raise ValueError("activation corpus lineage has no model sources")
    return lineage


def _new_activation_corpus(
    manifest_path: Path, *, label: str, expected_layers: int, expected_hidden: int
) -> _ActivationCorpus:
    payload = _load_json_object(manifest_path, label="activation manifest")
    if payload.get("schema_version") != ACTIVATION_SCHEMA_VERSION:
        raise ValueError(f"unsupported activation schema: {manifest_path}")
    plan_path = manifest_path.parent / str(payload.get("plan_path", "PLAN.json"))
    if not plan_path.is_file() or sha256_file(plan_path) != payload.get("plan_sha256"):
        raise ValueError(f"activation PLAN hash mismatch: {plan_path}")
    plan = _load_json_object(plan_path, label="activation PLAN")
    if plan.get("plan_fingerprint") != payload.get("plan_fingerprint"):
        raise ValueError("activation manifest/PLAN fingerprint mismatch")
    if _canonical_sha256({k: v for k, v in plan.items() if k != "plan_fingerprint"}) != plan.get(
        "plan_fingerprint"
    ):
        raise ValueError("activation PLAN content fingerprint mismatch")
    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError(f"activation corpus has no shards: {manifest_path}")
    parts: list[_ActivationPart] = []
    seen_ids: set[str] = set()
    for shard in raw_shards:
        if not isinstance(shard, dict) or not isinstance(shard.get("sides"), dict):
            raise ValueError(f"malformed activation shard in {manifest_path}")
        side = shard["sides"].get(label)
        if not isinstance(side, dict):
            raise ValueError(f"activation shard lacks {label} side")
        directory = manifest_path.parent / str(side["path"])
        if not is_shard_complete(directory):
            raise ValueError(f"activation side shard is incomplete or corrupt: {directory}")
        if sha256_file(directory / "COMPLETE") != side.get("complete_sha256"):
            raise ValueError(f"activation COMPLETE hash differs from manifest: {directory}")
        marker = read_complete_marker(directory)
        metadata = marker.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("model_side") != label:
            raise ValueError(f"activation side metadata mismatch: {directory}")
        if metadata.get("plan_fingerprint") != payload.get("plan_fingerprint"):
            raise ValueError(f"activation side PLAN mismatch: {directory}")
        raw_parts = metadata.get("parts")
        if raw_parts != side.get("parts") or not isinstance(raw_parts, list):
            raise ValueError(f"activation part inventory mismatch: {directory}")
        side_rows = 0
        for part in raw_parts:
            part_id = str(part["part_id"])
            if part_id in seen_ids:
                raise ValueError(f"duplicate activation part identity: {part_id}")
            seen_ids.add(part_id)
            artifact = directory / str(part["path"])
            artifact_kind = str(part.get("artifact_kind", "pre_ffn_activation_part"))
            expected_metadata = {
                "schema_version": str(ACTIVATION_SCHEMA_VERSION),
                "kind": artifact_kind,
                "plan_fingerprint": str(payload["plan_fingerprint"]),
                "model_side": label,
                "input_sha256": str(metadata["input_sha256"]),
                "sample_count": str(part["sample_count"]),
                "selection_sha256": str(part["selection_sha256"]),
            }
            if artifact_kind == "pre_ffn_activation_part":
                expected_metadata.update(
                    {
                        # part_index is stored as an unpadded integer. Derive it
                        # from the path rather than trusting the readable ID.
                        "part_index": str(
                            int(Path(str(part["path"])).stem.removeprefix("part-"))
                        ),
                        "batch_start": str(part["batch_start"]),
                        "batch_end": str(part["batch_end"]),
                    }
                )
            elif artifact_kind != "pre_ffn_activation_shard":
                raise ValueError(f"unsupported activation artifact kind: {artifact_kind}")
            rows = _validate_activation_file(
                artifact,
                expected_layers,
                expected_rows=int(part["sample_count"]),
                expected_hidden=expected_hidden,
                expected_metadata=expected_metadata,
            )
            relative = artifact.relative_to(directory).as_posix()
            parts.append(
                _ActivationPart(
                    part_id=part_id,
                    path=artifact,
                    sample_count=rows,
                    batch_start=int(part["batch_start"]),
                    batch_end=int(part["batch_end"]),
                    selection_sha256=str(part["selection_sha256"]),
                    artifact_sha256=_inventory_sha(marker, relative),
                )
            )
            side_rows += rows
        if side_rows != int(side["sample_count"]):
            raise ValueError(f"activation side sample total mismatch: {directory}")
    total = sum(part.sample_count for part in parts)
    if total != int(payload.get("total_samples", -1)):
        raise ValueError("activation corpus total_samples mismatch")
    lineage = _activation_lineage(payload, plan)
    return _ActivationCorpus(
        label=label,
        source_path=manifest_path,
        source_sha256=sha256_file(manifest_path),
        lineage=lineage,
        parts=tuple(parts),
        total_samples=total,
    )


def _legacy_single_corpus(
    path: Path, *, label: str, expected_layers: int, expected_hidden: int
) -> _ActivationCorpus:
    rows = _validate_activation_file(
        path, expected_layers, expected_hidden=expected_hidden
    )
    digest = sha256_file(path)
    part = _ActivationPart(
        part_id="legacy-single",
        path=path,
        sample_count=rows,
        batch_start=0,
        batch_end=rows,
        selection_sha256="legacy-unknown",
        artifact_sha256=digest,
    )
    return _ActivationCorpus(
        label=label,
        source_path=path,
        source_sha256=digest,
        lineage={"schema_version": 1, "kind": "legacy_single_file", "sample_count": rows},
        parts=(part,),
        total_samples=rows,
        legacy=True,
    )


def _activation_corpus(
    path: str | Path, *, label: str, expected_layers: int, expected_hidden: int
) -> _ActivationCorpus:
    source = Path(path).resolve()
    if source.is_file() and source.suffix == ".safetensors":
        raise ValueError(
            "standalone legacy activation files have no model/input lineage; "
            "rerun `twen calibrate collect` and pass its corpus directory"
        )
    manifest = source / "manifest.json" if source.is_dir() else source
    if not manifest.is_file():
        raise ValueError(f"activation corpus/manifest does not exist: {path}")
    payload = _load_json_object(manifest, label="activation manifest")
    if payload.get("kind") != "paired_pre_ffn_activation_corpus":
        raise ValueError(f"unsupported activation manifest: {manifest}")
    if payload.get("schema_version") == ACTIVATION_SCHEMA_VERSION:
        return _new_activation_corpus(
            manifest,
            label=label,
            expected_layers=expected_layers,
            expected_hidden=expected_hidden,
        )
    raise ValueError(
        "legacy activation corpora have insufficient source/coordinate lineage; "
        "rerun `twen calibrate collect`"
    )


def _paired_activation_shards(
    student_path: str | Path,
    donor_path: str | Path,
    *,
    student_layers: int,
    donor_layers: int,
    student_hidden: int,
    donor_hidden: int,
) -> _PairedActivations:
    student = _activation_corpus(
        student_path,
        label="student",
        expected_layers=student_layers,
        expected_hidden=student_hidden,
    )
    donor = _activation_corpus(
        donor_path,
        label="donor",
        expected_layers=donor_layers,
        expected_hidden=donor_hidden,
    )
    if student.total_samples != donor.total_samples:
        raise ValueError("student and donor activation sample totals differ")
    if len(student.parts) != len(donor.parts):
        raise ValueError("student and donor activation part counts differ")
    if (not student.legacy or not donor.legacy) and student.lineage != donor.lineage:
        raise ValueError("student and donor activation lineage differs")
    pairs = []
    identities = []
    for student_part, donor_part in zip(student.parts, donor.parts, strict=True):
        identity_student = (
            student_part.part_id,
            student_part.sample_count,
            student_part.batch_start,
            student_part.batch_end,
            student_part.selection_sha256,
        )
        identity_donor = (
            donor_part.part_id,
            donor_part.sample_count,
            donor_part.batch_start,
            donor_part.batch_end,
            donor_part.selection_sha256,
        )
        # Legacy standalone files have no coordinate lineage; equal rows are
        # the strongest compatibility check that format can support.
        if identity_student != identity_donor and not (
                student.legacy
                and donor.legacy
                and len(student.parts) == len(donor.parts) == 1
                and student_part.sample_count == donor_part.sample_count
        ):
            raise ValueError("student and donor activation part identity/order differs")
        pairs.append((student_part.part_id, student_part.path, donor_part.path))
        identities.append(
            {
                "part_id": student_part.part_id,
                "rows": student_part.sample_count,
                "selection_sha256": student_part.selection_sha256,
                "student_sha256": student_part.artifact_sha256,
                "donor_sha256": donor_part.artifact_sha256,
            }
        )
    fingerprint = _canonical_sha256(
        {
            "student_source_sha256": student.source_sha256,
            "donor_source_sha256": donor.source_sha256,
            "student_lineage": student.lineage,
            "donor_lineage": donor.lineage,
            "parts": identities,
        }
    )
    return _PairedActivations(
        student=student,
        donor=donor,
        parts=tuple(pairs),
        total_samples=student.total_samples,
        fingerprint=fingerprint,
    )


def _sample_layer(files: Sequence[Path], layer: int, indices: Any) -> Any:
    import numpy as np
    import torch
    from safetensors import safe_open

    selected = np.asarray(indices, dtype=np.int64)
    values = []
    offset = 0
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(f"layers.{layer}")
        end = offset + int(tensor.shape[0])
        left = int(np.searchsorted(selected, offset, side="left"))
        right = int(np.searchsorted(selected, end, side="left"))
        if right > left:
            local = torch.from_numpy((selected[left:right] - offset).copy()).long()
            values.append(tensor.index_select(0, local))
        offset = end
    if offset == 0 or not values:
        raise ValueError(f"activation corpus has no sampled rows for layer {layer}")
    return torch.cat(values, dim=0)


def calculate_layer_map(
    config_path: str,
    student_path: str,
    donor_path: str,
    output: str,
    *,
    sample_seed: int = 3407,
    stop_file: str | None = None,
    progress: str = "auto",
) -> int:
    _check_stop_file(stop_file)
    import torch
    from safetensors import safe_open

    from .modeling import audit_source_configs, linear_cka, match_layers_cka

    config = load_train_config(config_path)
    paired = _paired_activation_shards(
        student_path,
        donor_path,
        student_layers=config.architecture.student_layers,
        donor_layers=config.architecture.donor_layers,
        student_hidden=config.architecture.student_hidden_size,
        donor_hidden=config.architecture.donor_hidden_size,
    )
    if not paired.student.legacy:
        sources = paired.student.lineage.get("sources")
        if not isinstance(sources, dict):
            raise ValueError("activation corpus lineage has no model sources")
        if sources.get("student") != _model_lineage(config.sources.backbone):
            raise ValueError("activation corpus student source differs from current config")
        if sources.get("donor") != _model_lineage(config.sources.donor):
            raise ValueError("activation corpus donor source differs from current config")
    if paired.total_samples < 2:
        raise ValueError("CKA requires at least two paired activation samples")
    selected = _deterministic_indices(
        paired.total_samples, min(CKA_SAMPLE_LIMIT, paired.total_samples), sample_seed
    )
    student_files = tuple(item[1] for item in paired.parts)
    donor_files = tuple(item[2] for item in paired.parts)
    source_audit = audit_source_configs(
        config.sources.backbone.local_path,
        config.sources.donor.local_path,
    )
    student_types = source_audit.student_layer_types
    donor_types = source_audit.donor_layer_types
    lineage = {
        "activation_pair_fingerprint": paired.fingerprint,
        "student_activation_source": str(paired.student.source_path),
        "student_activation_sha256": paired.student.source_sha256,
        "donor_activation_source": str(paired.donor.source_path),
        "donor_activation_sha256": paired.donor.source_sha256,
        "sample_algorithm": SAMPLER_ALGORITHM,
        "sample_seed": int(sample_seed),
        "sample_count": len(selected),
        "sample_indices_sha256": _array_sha256(selected),
        "student_model": _model_lineage(config.sources.backbone),
        "donor_model": _model_lineage(config.sources.donor),
        "projection": {
            "dimension": 64,
            "student_seed_base": 3407,
            "donor_seed_base": 13407,
        },
    }
    pipeline_fingerprint = _canonical_sha256(lineage)
    lineage["pipeline_fingerprint"] = pipeline_fingerprint
    target = Path(output).resolve()
    work_root = target.parent / f".{target.name}.cka-work"
    target.parent.mkdir(parents=True, exist_ok=True)

    def projection_tensor(
        path: Path,
        *,
        fingerprint: str,
        label: str,
        layer: int,
    ) -> Any:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if (
                set(handle.keys()) != {"projection"}
                or metadata.get("fingerprint") != fingerprint
                or metadata.get("model_side") != label
                or int(metadata.get("layer", -1)) != layer
            ):
                raise ValueError(f"CKA projection lineage mismatch: {path}")
            value = handle.get_tensor("projection")
        if tuple(value.shape) != (len(selected), 64):
            raise ValueError(f"CKA projection shape mismatch: {path}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"CKA projection contains non-finite values: {path}")
        return value

    with FileLock(
        target.with_name(f".{target.name}.lock"), timeout_seconds=300.0
    ), TaskProgress(
        total=(
            config.architecture.student_layers
            + config.architecture.donor_layers
            + config.architecture.student_layers * config.architecture.donor_layers
        ),
        description="calibrate-layer-map",
        unit="unit",
        mode=progress,
    ) as progress_bar:
        _check_stop_file(stop_file)
        projection_paths: dict[str, list[tuple[Path, str]]] = {
            "student": [],
            "donor": [],
        }
        for label, files, layers, family_seed in (
            ("student", student_files, config.architecture.student_layers, 3407),
            ("donor", donor_files, config.architecture.donor_layers, 13407),
        ):
            for layer in range(layers):
                fingerprint = _canonical_sha256(
                    {
                        "pipeline_fingerprint": pipeline_fingerprint,
                        "kind": "cka_projection",
                        "model_side": label,
                        "layer": layer,
                    }
                )
                with ShardTransaction(
                    work_root / "projections" / label,
                    f"layer-{layer:03d}",
                    fingerprint=fingerprint,
                    source_fingerprint=paired.fingerprint,
                ) as transaction:
                    projection_path = transaction.work_directory / "projection.safetensors"
                    if transaction.complete:
                        projection_tensor(
                            projection_path,
                            fingerprint=fingerprint,
                            label=label,
                            layer=layer,
                        )
                    else:
                        sample = _sample_layer(files, layer, selected).float()
                        generator = torch.Generator(device="cpu").manual_seed(
                            family_seed + layer
                        )
                        matrix = torch.randn(
                            sample.shape[-1],
                            64,
                            generator=generator,
                            dtype=torch.float32,
                        ) / math.sqrt(64)
                        projection = sample.matmul(matrix)
                        _save_safetensors_atomic(
                            {"projection": projection},
                            projection_path,
                            metadata={
                                "schema_version": str(CALIBRATION_SCHEMA_VERSION),
                                "kind": "cka_projection",
                                "fingerprint": fingerprint,
                                "model_side": label,
                                "layer": str(layer),
                                "samples": str(len(selected)),
                                "projection_dimension": "64",
                            },
                        )
                        projection_tensor(
                            projection_path,
                            fingerprint=fingerprint,
                            label=label,
                            layer=layer,
                        )
                        transaction.commit(
                            {
                                "kind": "cka_projection",
                                "model_side": label,
                                "layer": layer,
                            }
                        )
                    projection_paths[label].append(
                        (transaction.final_directory / "projection.safetensors", fingerprint)
                    )
                progress_bar.set_postfix({"projection": f"{label}:{layer}"})
                progress_bar.update()
                _check_stop_file(stop_file)

        scores: list[list[float]] = []
        for student_layer in range(config.architecture.student_layers):
            fingerprint = _canonical_sha256(
                {
                    "pipeline_fingerprint": pipeline_fingerprint,
                    "kind": "cka_score_row",
                    "student_layer": student_layer,
                }
            )
            with ShardTransaction(
                work_root / "scores",
                f"layer-{student_layer:03d}",
                fingerprint=fingerprint,
                source_fingerprint=paired.fingerprint,
            ) as transaction:
                state_path = transaction.work_directory / "scores.json"
                row: list[float] = []
                if state_path.is_file():
                    try:
                        state = _load_json_object(state_path, label="CKA score row")
                        raw_row = state.get("scores")
                        if (
                            state.get("fingerprint") != fingerprint
                            or state.get("student_layer") != student_layer
                            or not isinstance(raw_row, list)
                            or len(raw_row) > config.architecture.donor_layers
                        ):
                            raise ValueError("score-row identity mismatch")
                        row = [float(value) for value in raw_row]
                        if any(not math.isfinite(value) for value in row):
                            raise ValueError("score row contains non-finite values")
                    except (OSError, TypeError, ValueError):
                        if transaction.complete:
                            raise
                        state_path.unlink(missing_ok=True)
                        row = []
                progress_bar.update(len(row))
                student_projection = projection_tensor(
                    projection_paths["student"][student_layer][0],
                    fingerprint=projection_paths["student"][student_layer][1],
                    label="student",
                    layer=student_layer,
                )
                for donor_layer in range(len(row), config.architecture.donor_layers):
                    donor_projection = projection_tensor(
                        projection_paths["donor"][donor_layer][0],
                        fingerprint=projection_paths["donor"][donor_layer][1],
                        label="donor",
                        layer=donor_layer,
                    )
                    row.append(linear_cka(student_projection, donor_projection))
                    atomic_write_json(
                        state_path,
                        {
                            "schema_version": CALIBRATION_SCHEMA_VERSION,
                            "kind": "cka_score_row",
                            "fingerprint": fingerprint,
                            "student_layer": student_layer,
                            "scores": row,
                        },
                    )
                    progress_bar.set_postfix(
                        {"cka": f"{student_layer}->{donor_layer}"}
                    )
                    progress_bar.update()
                    _check_stop_file(stop_file)
                if not transaction.complete:
                    transaction.commit(
                        {
                            "kind": "cka_score_row",
                            "student_layer": student_layer,
                            "donor_layers": len(row),
                        }
                    )
                scores.append(row)
            _check_stop_file(stop_file)

        match = match_layers_cka(
            cka_scores=scores,
            student_layer_types=student_types,
            donor_layer_types=donor_types,
        )
        payload = {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "kind": "monotonic_same_type_cka",
            "lineage": lineage,
            "input_fingerprint": pipeline_fingerprint,
            **match.to_dict(),
        }
        if target.exists():
            existing = _load_json_object(target, label="layer map")
            if existing.get("input_fingerprint") != payload["input_fingerprint"]:
                raise ValueError(f"layer map {target} belongs to incompatible inputs")
            if existing != payload:
                raise ValueError(f"layer map {target} differs despite matching lineage")
            return 0
        atomic_write_json(target, payload)
    return 0


def _ridge_stats_save(stats: Any, path: Path, *, cursor: int, fingerprint: str) -> None:
    state = stats.state_dict()
    tensors = {
        "small_gram": state["small_gram"],
        "large_gram": state["large_gram"],
        "cross": state["cross"],
    }
    metadata = {
        "schema_version": str(state["schema_version"]),
        "small_dim": str(state["small_dim"]),
        "large_dim": str(state["large_dim"]),
        "sample_count": str(state["sample_count"]),
        "cursor": str(cursor),
        "fingerprint": fingerprint,
    }
    _save_safetensors_atomic(tensors, path, metadata=metadata)


def _ridge_stats_load(path: Path, *, device: str, fingerprint: str) -> tuple[Any, int]:
    from safetensors import safe_open

    from .modeling import BidirectionalRidgeStats

    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("fingerprint") != fingerprint:
            raise ValueError("ridge stats fingerprint mismatch")
        state = {
            "schema_version": int(metadata["schema_version"]),
            "small_dim": int(metadata["small_dim"]),
            "large_dim": int(metadata["large_dim"]),
            "sample_count": int(metadata["sample_count"]),
            "small_gram": handle.get_tensor("small_gram"),
            "large_gram": handle.get_tensor("large_gram"),
            "cross": handle.get_tensor("cross"),
        }
        cursor = int(metadata["cursor"])
    return BidirectionalRidgeStats.from_state_dict(state, device=device), cursor


def _validate_ridge_solution(
    path: Path, *, layer: int, small_dim: int, large_dim: int, fingerprint: str
) -> dict[str, Any]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("fingerprint") != fingerprint or int(metadata.get("layer", -1)) != layer:
            raise ValueError(f"ridge solution lineage mismatch: {path}")
        a_shape = tuple(handle.get_slice("A").get_shape())
        b_shape = tuple(handle.get_slice("B").get_shape())
        if a_shape != (large_dim, small_dim) or b_shape != (small_dim, large_dim):
            raise ValueError(f"ridge solution shape mismatch: {path}")
        return {
            "donor_layer": int(metadata["donor_layer"]),
            "samples": int(metadata["sample_count"]),
            "l2": float(metadata["l2"]),
        }


def _flush_ridge_batch(stats: Any, small_parts: list[Any], large_parts: list[Any]) -> None:
    """Accumulate one reasonably sized GEMM and release its CPU part tensors."""

    if not small_parts:
        return
    import torch

    small_batch = small_parts[0] if len(small_parts) == 1 else torch.cat(small_parts, dim=0)
    large_batch = large_parts[0] if len(large_parts) == 1 else torch.cat(large_parts, dim=0)
    stats.update(small_batch, large_batch)
    small_parts.clear()
    large_parts.clear()


def calculate_ridge(
    config_path: str,
    student_path: str,
    donor_path: str,
    output: str,
    *,
    device: str,
    accumulation_dtype: str = "auto",
    batch_samples: int = 1024,
    stop_file: str | None = None,
    progress: str = "auto",
) -> int:
    _check_stop_file(stop_file)
    import torch
    from safetensors import SafetensorError, safe_open

    from .modeling import BidirectionalRidgeStats
    from .training.builder import load_layer_mapping

    if accumulation_dtype not in {"auto", "float32", "float64"}:
        raise ValueError("ridge accumulation dtype must be auto, float32, or float64")
    if (
        isinstance(batch_samples, bool)
        or not isinstance(batch_samples, int)
        or batch_samples <= 0
    ):
        raise ValueError("ridge batch_samples must be a positive integer")
    is_cuda = str(device).startswith("cuda")
    resolved_dtype_name = (
        ("float32" if is_cuda else "float64")
        if accumulation_dtype == "auto"
        else accumulation_dtype
    )
    # CUDA FP64 throughput on consumer GPUs such as RTX 5090 is intentionally
    # low.  Ridge regularization keeps the normalized covariance solve usable
    # in FP32, while CPU auto mode retains the more conservative FP64 path.
    accumulation_torch_dtype = (
        torch.float32 if resolved_dtype_name == "float32" else torch.float64
    )
    config = load_train_config(config_path)
    paired = _paired_activation_shards(
        student_path,
        donor_path,
        student_layers=config.architecture.student_layers,
        donor_layers=config.architecture.donor_layers,
        student_hidden=config.architecture.student_hidden_size,
        donor_hidden=config.architecture.donor_hidden_size,
    )
    if not paired.student.legacy:
        sources = paired.student.lineage.get("sources")
        if not isinstance(sources, dict):
            raise ValueError("activation corpus lineage has no model sources")
        if sources.get("student") != _model_lineage(config.sources.backbone):
            raise ValueError("activation corpus student source differs from current config")
        if sources.get("donor") != _model_lineage(config.sources.donor):
            raise ValueError("activation corpus donor source differs from current config")
    layer_map_path = Path(config.architecture.layer_map_path).resolve()
    mapping = load_layer_mapping(layer_map_path, config.architecture.student_layers)
    layer_map_sha = sha256_file(layer_map_path)
    student_model_lineage = _model_lineage(config.sources.backbone)
    donor_model_lineage = _model_lineage(config.sources.donor)
    target = Path(output).resolve()
    sidecar = target.with_suffix(".json")
    work_root = target.parent / f".{target.name}.ridge-work"
    l2 = 1e-4
    pipeline = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "kind": "bidirectional_ridge_pipeline",
        "activation_pair_fingerprint": paired.fingerprint,
        "layer_map_sha256": layer_map_sha,
        "mapping": list(mapping),
        "small_dim": config.architecture.student_hidden_size,
        "large_dim": config.architecture.donor_hidden_size,
        "l2": l2,
        "accumulation_device_type": "cuda" if is_cuda else "cpu",
        "accumulation_dtype": resolved_dtype_name,
        "batch_samples": int(batch_samples),
        "track": config.track,
        "architecture": _architecture_lineage(config),
        "model_sources": {
            "student": student_model_lineage,
            "donor": donor_model_lineage,
        },
    }
    pipeline_fingerprint = _canonical_sha256(pipeline)
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(
        target.with_name(f".{target.name}.lock"), timeout_seconds=300.0
    ), TaskProgress(
        total=len(mapping) * (len(paired.parts) + 1),
        description="calibrate-ridge",
        unit="unit",
        mode=progress,
    ) as progress_bar:
        if sidecar.exists():
            existing = _load_json_object(sidecar, label="adapter sidecar")
            if existing.get("pipeline_fingerprint") != pipeline_fingerprint:
                raise ValueError(f"adapter artifact {target} belongs to incompatible inputs")
            artifact = existing.get("artifact")
            if (
                not target.is_file()
                or not isinstance(artifact, dict)
                or sha256_file(target) != artifact.get("sha256")
            ):
                raise ValueError(f"adapter artifact does not match its sidecar: {target}")
            progress_bar.set_postfix({"state": "complete"})
            progress_bar.update(len(mapping) * (len(paired.parts) + 1))
            return 0

        if target.exists():
            try:
                with safe_open(target, framework="pt", device="cpu") as handle:
                    metadata = handle.metadata() or {}
                if metadata.get("pipeline_fingerprint") != pipeline_fingerprint:
                    raise ValueError("pipeline fingerprint mismatch")
            except (OSError, RuntimeError, ValueError, SafetensorError) as error:
                raise ValueError(
                    f"adapter output exists without a compatible sidecar: {target}"
                ) from error

        _check_stop_file(stop_file)
        layer_metadata: dict[str, Any] = {}
        for small_layer, large_layer in enumerate(mapping):
            layer_fingerprint = _canonical_sha256(
                {
                    "pipeline_fingerprint": pipeline_fingerprint,
                    "small_layer": small_layer,
                    "large_layer": large_layer,
                }
            )
            with ShardTransaction(
                work_root / "layers",
                f"layer-{small_layer:03d}",
                fingerprint=layer_fingerprint,
                source_fingerprint=paired.fingerprint,
            ) as transaction:
                solution_path = transaction.work_directory / "solution.safetensors"
                if transaction.complete:
                    layer_metadata[str(small_layer)] = _validate_ridge_solution(
                        solution_path,
                        layer=small_layer,
                        small_dim=config.architecture.student_hidden_size,
                        large_dim=config.architecture.donor_hidden_size,
                        fingerprint=layer_fingerprint,
                    )
                    progress_bar.set_postfix({"layer": small_layer, "state": "complete"})
                    progress_bar.update(len(paired.parts) + 1)
                    continue
                stats_path = transaction.work_directory / "stats.safetensors"
                stats = None
                cursor = 0
                if stats_path.is_file():
                    try:
                        stats, cursor = _ridge_stats_load(
                            stats_path, device=device, fingerprint=layer_fingerprint
                        )
                        if not 0 <= cursor <= len(paired.parts):
                            raise ValueError("ridge cursor is out of range")
                        if stats.dtype != accumulation_torch_dtype:
                            raise ValueError("ridge stats accumulation dtype mismatch")
                        expected_samples = sum(
                            part.sample_count for part in paired.student.parts[:cursor]
                        )
                        if stats.sample_count != expected_samples:
                            raise ValueError("ridge stats sample cursor mismatch")
                    except (
                        KeyError,
                        OSError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                        SafetensorError,
                    ):
                        stats_path.unlink(missing_ok=True)
                        solution_path.unlink(missing_ok=True)
                        stats = None
                        cursor = 0
                progress_bar.update(cursor)
                if stats is None:
                    stats = BidirectionalRidgeStats(
                        config.architecture.student_hidden_size,
                        config.architecture.donor_hidden_size,
                        device=device,
                        dtype=accumulation_torch_dtype,
                    )
                pending_small: list[Any] = []
                pending_large: list[Any] = []
                pending_rows = 0

                for part_index in range(cursor, len(paired.parts)):
                    part_id, student_file, donor_file = paired.parts[part_index]
                    with safe_open(student_file, framework="pt", device="cpu") as handle:
                        small = handle.get_tensor(f"layers.{small_layer}")
                    with safe_open(donor_file, framework="pt", device="cpu") as handle:
                        large = handle.get_tensor(f"layers.{large_layer}")
                    if small.shape[0] != large.shape[0]:
                        raise ValueError("paired ridge activation rows differ")
                    if small.shape[0] > 0:
                        pending_small.append(small)
                        pending_large.append(large)
                        pending_rows += int(small.shape[0])
                    if pending_rows >= batch_samples:
                        _flush_ridge_batch(stats, pending_small, pending_large)
                        pending_rows = 0
                    progress_bar.set_postfix(
                        {
                            "layer": small_layer,
                            "part": f"{part_index + 1}/{len(paired.parts)}",
                            "samples": stats.sample_count + pending_rows,
                        }
                    )
                    progress_bar.update()
                    # A 4096x4096 sufficient-statistics snapshot is large.
                    # Flush the sample batch and commit once per source input
                    # shard (not once per inference microbatch), which is the
                    # promised recovery boundary without multiplying I/O.
                    shard_id = part_id.split("/", 1)[0]
                    next_shard_id = (
                        paired.parts[part_index + 1][0].split("/", 1)[0]
                        if part_index + 1 < len(paired.parts)
                        else None
                    )
                    if next_shard_id != shard_id:
                        _flush_ridge_batch(stats, pending_small, pending_large)
                        pending_rows = 0
                        _ridge_stats_save(
                            stats,
                            stats_path,
                            cursor=part_index + 1,
                            fingerprint=layer_fingerprint,
                        )
                        _check_stop_file(stop_file)
                _flush_ridge_batch(stats, pending_small, pending_large)
                if stats.sample_count != paired.total_samples:
                    raise RuntimeError(
                        "ridge sample accounting failed: "
                        f"got {stats.sample_count}, expected {paired.total_samples}"
                    )
                solution = stats.solve(l2=l2, output_dtype=torch.float32)
                solution_metadata = {
                    "schema_version": str(CALIBRATION_SCHEMA_VERSION),
                    "kind": "bidirectional_ridge_layer_solution",
                    "fingerprint": layer_fingerprint,
                    "layer": str(small_layer),
                    "donor_layer": str(large_layer),
                    "sample_count": str(solution.sample_count),
                    "l2": repr(solution.l2),
                }
                _save_safetensors_atomic(
                    {"A": solution.input_adapter, "B": solution.output_adapter},
                    solution_path,
                    metadata=solution_metadata,
                )
                layer_metadata[str(small_layer)] = _validate_ridge_solution(
                    solution_path,
                    layer=small_layer,
                    small_dim=config.architecture.student_hidden_size,
                    large_dim=config.architecture.donor_hidden_size,
                    fingerprint=layer_fingerprint,
                )
                transaction.commit(
                    {
                        "kind": "bidirectional_ridge_layer",
                        "small_layer": small_layer,
                        "donor_layer": large_layer,
                        "sample_count": solution.sample_count,
                        "activation_parts": len(paired.parts),
                    }
                )
                progress_bar.set_postfix({"layer": small_layer, "state": "solved"})
                progress_bar.update()
            _check_stop_file(stop_file)

        tensors: dict[str, Any] = {}
        for layer in range(config.architecture.student_layers):
            solution_path = work_root / "layers" / f"layer-{layer:03d}" / "solution.safetensors"
            with safe_open(solution_path, framework="pt", device="cpu") as handle:
                tensors[f"layers.{layer}.A"] = handle.get_tensor("A")
                tensors[f"layers.{layer}.B"] = handle.get_tensor("B")
        _save_safetensors_atomic(
            tensors,
            target,
            metadata={
                "schema_version": str(CALIBRATION_SCHEMA_VERSION),
                "kind": "bidirectional_ridge_adapters",
                "pipeline_fingerprint": pipeline_fingerprint,
            },
        )
        artifact_sha = sha256_file(target)
        adapter_sidecar = {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "kind": "bidirectional_ridge_adapters",
            "pipeline_fingerprint": pipeline_fingerprint,
            "artifact": {
                "path": str(target),
                "size": target.stat().st_size,
                "sha256": artifact_sha,
            },
            "layer_map": {"path": str(layer_map_path), "sha256": layer_map_sha},
            "activation_sources": {
                "pair_fingerprint": paired.fingerprint,
                "student": {
                    "path": str(paired.student.source_path),
                    "sha256": paired.student.source_sha256,
                },
                "donor": {
                    "path": str(paired.donor.source_path),
                    "sha256": paired.donor.source_sha256,
                },
            },
            "model_sources": {
                "student": student_model_lineage,
                "donor": donor_model_lineage,
            },
            "track": config.track,
            "architecture": _architecture_lineage(config),
            "ridge": {
                "l2": l2,
                "accumulation_device_type": "cuda" if is_cuda else "cpu",
                "accumulation_dtype": resolved_dtype_name,
                "batch_samples": int(batch_samples),
            },
            "layers": layer_metadata,
        }
        atomic_write_json(sidecar, adapter_sidecar)
    return 0


def calculate_partitions(
    config_path: str,
    output: str,
    *,
    stop_file: str | None = None,
    progress: str = "auto",
) -> int:
    _check_stop_file(stop_file)
    from .modeling import build_channel_partition
    from .training.builder import load_layer_mapping

    config = load_train_config(config_path)
    _audit_local_sources(config)
    layer_map_path = Path(config.architecture.layer_map_path).resolve()
    mapping = load_layer_mapping(layer_map_path, config.architecture.student_layers)
    donor_lineage = _model_lineage(config.sources.donor)
    pipeline = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "kind": "channel_partition_pipeline",
        "strategy": "greedy",
        "track": config.track,
        "architecture": _architecture_lineage(config),
        "layer_map_path": str(layer_map_path),
        "layer_map_sha256": sha256_file(layer_map_path),
        "mapping": list(mapping),
        "donor_source": donor_lineage,
    }
    fingerprint = _canonical_sha256(pipeline)
    target = Path(output).resolve()
    work_root = target.parent / f".{target.name}.partition-work"
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(
        target.with_name(f".{target.name}.lock"), timeout_seconds=300.0
    ), TaskProgress(
        total=config.architecture.student_layers,
        description="calibrate-partition",
        unit="layer",
        mode=progress,
    ) as progress_bar:
        if target.exists():
            existing = _load_json_object(target, label="channel map")
            if existing.get("pipeline_fingerprint") != fingerprint:
                raise ValueError(f"channel map {target} belongs to incompatible inputs")
            progress_bar.set_postfix({"state": "complete"})
            progress_bar.update(config.architecture.student_layers)
            return 0
        _check_stop_file(stop_file)
        layers: dict[str, Any] = {}
        for student_layer, donor_layer in enumerate(mapping):
            layer_fingerprint = _canonical_sha256(
                {
                    "pipeline_fingerprint": fingerprint,
                    "student_layer": student_layer,
                    "donor_layer": donor_layer,
                }
            )
            with ShardTransaction(
                work_root / "layers",
                f"layer-{student_layer:03d}",
                fingerprint=layer_fingerprint,
                source_fingerprint=f"{donor_lineage['manifest_sha256']}:{donor_layer}",
            ) as transaction:
                part_path = transaction.work_directory / "partition.json"
                if transaction.complete:
                    value = _load_json_object(part_path, label="layer partition")
                    if value.get("fingerprint") != layer_fingerprint:
                        raise ValueError(f"partition layer lineage mismatch: {part_path}")
                else:
                    value = None
                    for stale in transaction.work_directory.glob(
                        ".partition.json.*.tmp"
                    ):
                        stale.unlink(missing_ok=True)
                    if part_path.is_file():
                        try:
                            candidate = _load_json_object(part_path, label="layer partition")
                            if candidate.get("fingerprint") == layer_fingerprint:
                                value = candidate
                        except ValueError:
                            value = None
                        if value is None:
                            part_path.unlink(missing_ok=True)
                    if value is None:
                        weights = load_donor_mlp_weights(
                            config.sources.donor.local_path, donor_layer, device="cpu"
                        )
                        partition = build_channel_partition(
                            weights["gate_proj.weight"],
                            weights["up_proj.weight"],
                            weights["down_proj.weight"],
                            num_experts=config.architecture.num_experts,
                            expert_size=config.architecture.expert_intermediate_size,
                        )
                        value = {
                            "schema_version": CALIBRATION_SCHEMA_VERSION,
                            "kind": "channel_partition_layer",
                            "fingerprint": layer_fingerprint,
                            "student_layer": student_layer,
                            "donor_layer": donor_layer,
                            **partition.to_dict(),
                        }
                        atomic_write_json(part_path, value)
                    transaction.commit(
                        {
                            "kind": "channel_partition_layer",
                            "student_layer": student_layer,
                            "donor_layer": donor_layer,
                        }
                    )
                layers[str(student_layer)] = {
                    key: item
                    for key, item in value.items()
                    if key not in {"schema_version", "kind", "fingerprint", "student_layer"}
                }
            progress_bar.set_postfix({"layer": student_layer, "donor": donor_layer})
            progress_bar.update()
            _check_stop_file(stop_file)
        atomic_write_json(
            target,
            {
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "kind": "channel_partition_map",
                "pipeline_fingerprint": fingerprint,
                "lineage": pipeline,
                "strategy": "greedy",
                "layers": layers,
            },
        )
    return 0


def run_calibration_command(args: Any) -> int:
    if args.action == "collect":
        return collect_activations(
            args.config,
            args.input,
            args.output,
            prepared_manifest=args.prepared_manifest,
            device=args.device,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            sample_seed=args.sample_seed,
            stop_file=args.stop_file,
            progress=args.progress,
        )
    if args.input or args.prepared_manifest:
        raise ValueError("--input/--prepared-manifest are only valid for calibrate collect")
    if args.action == "layer-map":
        if not args.student_activations or not args.donor_activations:
            raise ValueError("layer-map requires --student-activations and --donor-activations")
        return calculate_layer_map(
            args.config,
            args.student_activations,
            args.donor_activations,
            args.output,
            sample_seed=args.sample_seed,
            stop_file=args.stop_file,
            progress=args.progress,
        )
    if args.action == "ridge":
        if not args.student_activations or not args.donor_activations:
            raise ValueError("ridge requires --student-activations and --donor-activations")
        return calculate_ridge(
            args.config,
            args.student_activations,
            args.donor_activations,
            args.output,
            device=args.device,
            accumulation_dtype=getattr(args, "ridge_dtype", "auto"),
            batch_samples=getattr(args, "ridge_batch_samples", 1024),
            stop_file=args.stop_file,
            progress=args.progress,
        )
    if args.action == "partition":
        return calculate_partitions(
            args.config,
            args.output,
            stop_file=args.stop_file,
            progress=args.progress,
        )
    raise ValueError(f"unknown calibration action {args.action}")
