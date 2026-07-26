"""Offline-only Qwen3.5 text checkpoint loading utilities.

Official Qwen3.5 checkpoints are conditional-generation repositories whose
language model tensors are prefixed with ``model.language_model``.  The main
text loaders create the text-only causal-LM class and remap only its tensors;
the checkpoint's separate top-level ``mtp.*`` component is available through
the strict :func:`load_qwen35_mtp` loader.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CheckpointFormatError(RuntimeError):
    """The local checkpoint is missing tensors or has an unsupported layout."""


@dataclass(frozen=True, slots=True)
class TensorLocation:
    name: str
    file: Path


class SafetensorCheckpoint:
    """Read selected tensors from a local sharded safetensors checkpoint."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise CheckpointFormatError(f"checkpoint directory does not exist: {self.root}")
        index_path = self.root / "model.safetensors.index.json"
        single_path = self.root / "model.safetensors"
        if index_path.is_file():
            raw = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = raw.get("weight_map")
            if not isinstance(weight_map, dict):
                raise CheckpointFormatError(f"invalid weight_map in {index_path}")
            self._weight_map = {
                str(key): self.root / str(value) for key, value in weight_map.items()
            }
        elif single_path.is_file():
            try:
                from safetensors import safe_open
            except ImportError as exc:  # pragma: no cover - dependency error
                raise RuntimeError("safetensors is required to read model weights") from exc
            with safe_open(single_path, framework="pt", device="cpu") as handle:
                # ``safe_open`` exposes ``keys()`` but does not promise Mapping iteration.
                self._weight_map = {key: single_path for key in handle.keys()}  # noqa: SIM118
        else:
            candidates = sorted(self.root.glob("model-*.safetensors"))
            if not candidates:
                raise CheckpointFormatError(
                    f"no model.safetensors or model.safetensors.index.json under {self.root}"
                )
            raise CheckpointFormatError(
                "sharded safetensors require model.safetensors.index.json for deterministic loading"
            )
        missing_files = sorted({path for path in self._weight_map.values() if not path.is_file()})
        if missing_files:
            raise CheckpointFormatError(f"checkpoint shard is missing: {missing_files[0]}")

    def __contains__(self, name: str) -> bool:
        return name in self._weight_map

    def keys(self) -> tuple[str, ...]:
        return tuple(self._weight_map)

    def tensor(self, name: str, *, device: str = "cpu") -> Any:
        try:
            path = self._weight_map[name]
        except KeyError as exc:
            raise CheckpointFormatError(f"tensor is absent from checkpoint: {name}") from exc
        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("safetensors is required to read model weights") from exc
        with safe_open(path, framework="pt", device=device) as handle:
            return handle.get_tensor(name)

    def grouped_by_file(self) -> Iterator[tuple[Path, tuple[str, ...]]]:
        groups: dict[Path, list[str]] = {}
        for name, path in self._weight_map.items():
            groups.setdefault(path, []).append(name)
        for path in sorted(groups):
            yield path, tuple(sorted(groups[path]))


def read_qwen_text_config(root: str | Path) -> dict[str, Any]:
    config_path = Path(root) / "config.json"
    if not config_path.is_file():
        raise CheckpointFormatError(f"missing config.json: {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    text = raw.get("text_config", raw)
    if not isinstance(text, dict) or text.get("model_type") not in {
        "qwen3_5_text",
        "qwen3_5_moe_text",
    }:
        raise CheckpointFormatError("checkpoint is not a Qwen3.5 text model")
    return dict(text)


def _language_target_name(source_name: str) -> str | None:
    prefix = "model.language_model."
    if source_name.startswith(prefix):
        return "model." + source_name[len(prefix) :]
    if source_name.startswith("model.") and not source_name.startswith("model.visual."):
        # Already a text-only Qwen3.5ForCausalLM checkpoint.
        return source_name
    if source_name == "lm_head.weight":
        return source_name
    return None


def load_qwen35_text_causal_lm(
    root: str | Path,
    *,
    dtype: Any | None = None,
    device: str = "cpu",
) -> Any:
    """Load only the text causal LM from an official local Qwen3.5 checkpoint.

    This function never consults the Hub.  It intentionally fails if any
    required text tensor is missing instead of silently retaining a random
    initialization.
    """

    try:
        import torch
        from safetensors import safe_open
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch, transformers and safetensors are required") from exc

    raw_config = read_qwen_text_config(root)
    if raw_config.get("model_type") == "qwen3_5_moe_text":
        raise CheckpointFormatError("expected a dense Qwen3.5 source, got a MoE checkpoint")
    config = Qwen3_5TextConfig.from_dict(raw_config)
    if dtype is not None:
        config.dtype = dtype
    # Transformers does not honor config.dtype when constructing from a config.
    # Scope PyTorch's default dtype to the single-threaded constructor so large
    # checkpoints never exist as FP32 first. This also preserves RoPE's explicit
    # FP32 non-persistent buffers, matching normal low-precision Hub loading.
    previous_default_dtype = torch.get_default_dtype()
    try:
        if dtype is not None:
            torch.set_default_dtype(dtype)
        model = Qwen3_5ForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_default_dtype)
    checkpoint = SafetensorCheckpoint(root)
    loaded: set[str] = set()
    for shard_path, names in checkpoint.grouped_by_file():
        shard: dict[str, Any] = {}
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for source_name in names:
                target_name = _language_target_name(source_name)
                if target_name is None:
                    continue
                tensor = handle.get_tensor(source_name)
                if dtype is not None and tensor.is_floating_point():
                    tensor = tensor.to(dtype=dtype)
                shard[target_name] = tensor
        if shard:
            model.load_state_dict(shard, strict=False)
            loaded.update(shard)

    expected = set(model.state_dict())
    # Tied lm_head is allowed to be absent; tie_weights materializes the alias.
    missing = expected - loaded
    if config.tie_word_embeddings:
        missing.discard("lm_head.weight")
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise CheckpointFormatError(f"missing {len(missing)} required text tensors: {preview}")
    model.tie_weights()
    model.to(device=device)
    return model


def load_qwen35_text_model(
    root: str | Path,
    *,
    dtype: Any | None = None,
    device: str = "cpu",
) -> Any:
    """Load the decoder-only text body, excluding the vocabulary-sized LM head.

    Hidden-alignment teachers use this loader so FSDP can shard only the text
    body and the forward pass never materializes unused logits.
    """

    try:
        import torch
        from safetensors import safe_open
        from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch, transformers and safetensors are required") from exc

    raw_config = read_qwen_text_config(root)
    if raw_config.get("model_type") == "qwen3_5_moe_text":
        raise CheckpointFormatError("expected a dense Qwen3.5 source, got a MoE checkpoint")
    config = Qwen3_5TextConfig.from_dict(raw_config)
    if dtype is not None:
        config.dtype = dtype
    previous_default_dtype = torch.get_default_dtype()
    try:
        if dtype is not None:
            torch.set_default_dtype(dtype)
        model = Qwen3_5TextModel(config)
    finally:
        torch.set_default_dtype(previous_default_dtype)

    checkpoint = SafetensorCheckpoint(root)
    loaded: set[str] = set()
    for shard_path, names in checkpoint.grouped_by_file():
        shard: dict[str, Any] = {}
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for source_name in names:
                causal_name = _language_target_name(source_name)
                if causal_name is None or not causal_name.startswith("model."):
                    continue
                target_name = causal_name.removeprefix("model.")
                tensor = handle.get_tensor(source_name)
                if dtype is not None and tensor.is_floating_point():
                    tensor = tensor.to(dtype=dtype)
                shard[target_name] = tensor
        if shard:
            model.load_state_dict(shard, strict=False)
            loaded.update(shard)
    missing = set(model.state_dict()) - loaded
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise CheckpointFormatError(f"missing {len(missing)} required text-body tensors: {preview}")
    model.to(device=device)
    return model


def load_qwen35_mtp(
    root: str | Path,
    *,
    dtype: Any | None = None,
    device: str = "cpu",
    trainable: bool = False,
) -> Any:
    """Strictly load the native one-layer MTP module from a dense checkpoint.

    Every top-level ``mtp.*`` tensor must match the module's exact key set and
    shape.  Vocabulary embeddings and the LM head are intentionally absent:
    Qwen3.5 checkpoints with ``mtp_use_dedicated_embeddings=false`` reuse the
    main model's modules at call time.
    """

    try:
        import torch
        from safetensors import safe_open
        from transformers import Qwen3_5TextConfig
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch, transformers and safetensors are required") from exc

    from .modeling.mtp import Qwen35MTP

    raw_config = read_qwen_text_config(root)
    if raw_config.get("model_type") != "qwen3_5_text":
        raise CheckpointFormatError("phase-one MTP loading requires a dense Qwen3.5 source")
    if raw_config.get("mtp_num_hidden_layers") != 1:
        raise CheckpointFormatError("checkpoint must declare mtp_num_hidden_layers=1")
    if raw_config.get("mtp_use_dedicated_embeddings") is not False:
        raise CheckpointFormatError("checkpoint must declare mtp_use_dedicated_embeddings=false")

    config = Qwen3_5TextConfig.from_dict(raw_config)
    config_dtype = getattr(config, "dtype", None)
    if config_dtype is not None and not isinstance(config_dtype, torch.dtype):
        raise CheckpointFormatError(
            f"checkpoint declares an unsupported source MTP dtype: {config_dtype}"
        )
    target_dtype = dtype if dtype is not None else config_dtype
    if target_dtype is not None and not isinstance(target_dtype, torch.dtype):
        raise CheckpointFormatError(f"checkpoint declares an unsupported MTP dtype: {target_dtype}")
    if target_dtype is not None:
        config.dtype = target_dtype
    previous_default_dtype = torch.get_default_dtype()
    try:
        if target_dtype is not None:
            torch.set_default_dtype(target_dtype)
        model = Qwen35MTP(config, trainable=trainable)
    finally:
        torch.set_default_dtype(previous_default_dtype)

    checkpoint = SafetensorCheckpoint(root)
    expected_state = model.state_dict()
    expected_source_names = {f"mtp.{name}" for name in expected_state}
    checkpoint_names = checkpoint.keys()
    actual_source_names = {name for name in checkpoint_names if name.startswith("mtp.")}
    missing = sorted(expected_source_names - actual_source_names)
    unexpected = sorted(actual_source_names - expected_source_names)
    if missing or unexpected:
        raise CheckpointFormatError(
            "MTP tensor keys differ from the native one-layer schema: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )

    state: dict[str, Any] = {}
    for shard_path, names in checkpoint.grouped_by_file():
        selected = [name for name in names if name in expected_source_names]
        if not selected:
            continue
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for source_name in selected:
                target_name = source_name.removeprefix("mtp.")
                tensor = handle.get_tensor(source_name)
                expected_shape = tuple(expected_state[target_name].shape)
                if tuple(tensor.shape) != expected_shape:
                    raise CheckpointFormatError(
                        f"MTP tensor {source_name} has shape {tuple(tensor.shape)}; "
                        f"expected {expected_shape}"
                    )
                if not tensor.is_floating_point():
                    raise CheckpointFormatError(f"MTP tensor {source_name} must be floating point")
                expected_dtype = expected_state[target_name].dtype
                if config_dtype is not None and tensor.dtype != config_dtype:
                    raise CheckpointFormatError(
                        f"MTP tensor {source_name} has dtype {tensor.dtype}; "
                        f"checkpoint config declares {config_dtype}"
                    )
                if tensor.dtype != expected_dtype:
                    tensor = tensor.to(dtype=expected_dtype)
                state[target_name] = tensor

    # Key and shape checks above turn a generic load_state_dict failure into a
    # checkpoint-format error with the offending source name. ``strict=True``
    # remains the final closed-world assertion against future module changes.
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:  # pragma: no cover - defensive future-schema guard
        raise CheckpointFormatError(f"failed to load native MTP state: {exc}") from exc
    model.to(device=device)
    if trainable:
        model.requires_grad_(True)
        model.train()
    else:
        freeze_module(model)
    return model


def load_donor_mlp_weights(
    root: str | Path,
    layer_index: int,
    *,
    device: str = "cpu",
) -> Mapping[str, Any]:
    """Load one frozen donor FFN without constructing the 9B model."""

    checkpoint = SafetensorCheckpoint(root)
    prefixes = (
        f"model.language_model.layers.{layer_index}.mlp",
        f"model.layers.{layer_index}.mlp",
    )
    result: dict[str, Any] = {}
    for short_name in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
        candidate = next(
            (
                f"{prefix}.{short_name}"
                for prefix in prefixes
                if f"{prefix}.{short_name}" in checkpoint
            ),
            None,
        )
        if candidate is None:
            raise CheckpointFormatError(
                f"missing donor layer {layer_index} tensor {short_name} under {root}"
            )
        result[short_name] = checkpoint.tensor(candidate, device=device)
    return result


def freeze_module(module: Any) -> None:
    module.requires_grad_(False)
    module.eval()
