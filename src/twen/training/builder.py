"""Construct dense-transfer or sparse-transfer text models from local artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from ..config import TrainConfig
from ..model_loading import (
    freeze_module,
    load_donor_mlp_weights,
    load_qwen35_mtp,
    load_qwen35_text_causal_lm,
)


class BuildError(RuntimeError):
    pass


@dataclass(slots=True)
class BuiltModel:
    model: Any
    transfer_modules: tuple[Any, ...]
    student_layer_indices: tuple[int, ...]
    donor_teacher_shared: bool = False
    mtp: Any | None = None


def _read_json(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BuildError(f"expected JSON object: {path}")
    return value


def load_layer_mapping(path: str | Path, expected_layers: int) -> tuple[int, ...]:
    value = _read_json(path)
    raw = value.get("student_to_donor")
    if raw is None and "pairs" in value:
        raw = [item["donor_layer"] for item in value["pairs"]]
    if not isinstance(raw, list) or len(raw) != expected_layers:
        raise BuildError(f"layer map must contain {expected_layers} donor indices")
    mapping = tuple(int(item) for item in raw)
    if any(right <= left for left, right in pairwise(mapping)):
        raise BuildError("donor layer mapping must be strictly increasing")
    return mapping


def load_channel_mapping(
    path: str | Path, expected_layers: int, expected_experts: int, expert_size: int
) -> tuple[Any, ...]:
    import torch

    value = _read_json(path)
    layers = value.get("layers")
    if isinstance(layers, list):
        raw_layers = layers
    elif isinstance(layers, dict):
        raw_layers = [layers.get(str(index)) for index in range(expected_layers)]
    elif "indices" in value:
        raw_layers = [value] * expected_layers
    else:
        raise BuildError("channel map requires a layers collection or top-level indices")
    if len(raw_layers) != expected_layers or any(item is None for item in raw_layers):
        raise BuildError(f"channel map must cover exactly {expected_layers} student layers")
    result = []
    for index, item in enumerate(raw_layers):
        indices = item.get("indices") if isinstance(item, dict) else None
        tensor = torch.as_tensor(indices, dtype=torch.long)
        expected = (expected_experts, expert_size)
        if tuple(tensor.shape) != expected:
            raise BuildError(f"layer {index} channel map must have shape {expected}")
        if not torch.equal(torch.sort(tensor.reshape(-1)).values, torch.arange(tensor.numel())):
            raise BuildError(f"layer {index} channel map is not a complete partition")
        result.append(tensor)
    return tuple(result)


def _load_adapter_initialization(path: str | Path, layers: int) -> tuple[tuple[Any, Any], ...]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required") from exc

    result = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for layer in range(layers):
            a_name = f"layers.{layer}.A"
            b_name = f"layers.{layer}.B"
            if a_name not in keys or b_name not in keys:
                raise BuildError(f"adapter initialization is missing layer {layer} A/B")
            result.append((handle.get_tensor(a_name), handle.get_tensor(b_name)))
    return tuple(result)


def _load_folded_layer(
    handle: Any,
    layer: int,
    *,
    expected_experts: int,
    expected_intermediate: int,
    expected_hidden: int,
) -> tuple[Any, Any, Any, Any | None, float]:
    names = {
        "gate": f"layers.{layer}.gate_proj",
        "up": f"layers.{layer}.up_proj",
        "down": f"layers.{layer}.down_proj",
    }
    available = set(handle.keys())
    missing = [name for name in names.values() if name not in available]
    if missing:
        raise BuildError(f"folded artifact is missing {missing[0]}")
    router_name = f"layers.{layer}.router"
    scale_name = f"layers.{layer}.branch_scale"
    gate = handle.get_tensor(names["gate"])
    up = handle.get_tensor(names["up"])
    down = handle.get_tensor(names["down"])
    expected_gate = (expected_experts, expected_intermediate, expected_hidden)
    expected_down = (expected_experts, expected_hidden, expected_intermediate)
    if tuple(gate.shape) != expected_gate or tuple(up.shape) != expected_gate:
        raise BuildError(
            f"folded layer {layer} gate/up must have shape {expected_gate}; "
            f"got {tuple(gate.shape)} and {tuple(up.shape)}"
        )
    if tuple(down.shape) != expected_down:
        raise BuildError(
            f"folded layer {layer} down must have shape {expected_down}; got {tuple(down.shape)}"
        )
    router = handle.get_tensor(router_name) if router_name in available else None
    if router is not None and tuple(router.shape) != (expected_experts, expected_hidden):
        raise BuildError(
            f"folded layer {layer} router must have shape "
            f"{(expected_experts, expected_hidden)}; got {tuple(router.shape)}"
        )
    scale = float(handle.get_tensor(scale_name).float().item()) if scale_name in available else 1.0
    if not math.isfinite(scale):
        raise BuildError(f"folded layer {layer} branch_scale must be finite")
    return (
        gate,
        up,
        down,
        router,
        scale,
    )


def _random_control_mlp_weights(
    config: TrainConfig,
    donor_layer: int,
    *,
    dtype: Any,
) -> dict[str, Any]:
    """Create one deterministic frozen Kaiming-initialized donor-shaped FFN.

    This exists only to make the Stage-B random-expert control executable.  It
    deliberately uses an isolated CPU generator, so every rank builds identical
    controls without consuming or perturbing the training RNG stream.
    """

    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.architecture.random_expert_seed + donor_layer * 1009)

    def weight(rows: int, columns: int) -> Any:
        value = torch.empty((rows, columns), dtype=torch.float32, device="cpu")
        bound = 1.0 / math.sqrt(columns)
        value.uniform_(-bound, bound, generator=generator)
        return value.to(dtype=dtype)

    donor_hidden = config.architecture.donor_hidden_size
    intermediate = config.architecture.donor_intermediate_size
    return {
        "gate_proj.weight": weight(intermediate, donor_hidden),
        "up_proj.weight": weight(intermediate, donor_hidden),
        "down_proj.weight": weight(donor_hidden, intermediate),
    }


def build_parameter_groups(config: TrainConfig, transfer_modules: tuple[Any, ...]) -> list[dict[str, Any]]:
    adapters: list[Any] = []
    routers: list[Any] = []
    lora: list[Any] = []
    scales: list[Any] = []
    for module in transfer_modules:
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.endswith("branch_scale"):
                scales.append(parameter)
            elif ".router." in f".{name}." or name.startswith("router."):
                routers.append(parameter)
            elif "lora_" in name:
                lora.append(parameter)
            elif "adapter" in name:
                adapters.append(parameter)
            else:
                raise BuildError(f"unclassified trainable parameter: {name}")
    groups = []
    for name, parameters, lr, decay in (
        ("adapters", adapters, config.optimizer.adapter_lr, config.optimizer.weight_decay),
        ("router", routers, config.optimizer.router_lr, 0.0),
        ("lora", lora, config.optimizer.lora_lr, config.optimizer.weight_decay),
        ("scale", scales, config.optimizer.scale_lr, 0.0),
    ):
        if parameters:
            groups.append({"name": name, "params": parameters, "lr": lr, "weight_decay": decay})
    if not groups:
        raise BuildError("constructed model has no trainable parameters")
    return groups


def _shared_donor_mlp_weights(teacher_text_model: Any, layer_index: int) -> Mapping[str, Any]:
    """Return a frozen teacher layer's projections for zero-copy donor reuse."""

    try:
        mlp = teacher_text_model.layers[layer_index].mlp
        weights = {
            "gate_proj.weight": mlp.gate_proj.weight,
            "up_proj.weight": mlp.up_proj.weight,
            "down_proj.weight": mlp.down_proj.weight,
        }
    except (AttributeError, IndexError, TypeError) as exc:
        raise BuildError(
            f"hidden-alignment teacher has no dense MLP at donor layer {layer_index}"
        ) from exc
    if any(weight.requires_grad for weight in weights.values()):
        raise BuildError(
            "donor/teacher parameter sharing requires the teacher to be frozen before build"
        )
    return weights


def build_transfer_model(
    config: TrainConfig,
    *,
    device: str,
    dtype: Any,
    donor_text_model: Any | None = None,
) -> BuiltModel:
    """Build a text-only model without any network access."""

    import torch

    from ..modeling import SharedDenseTransferMLP, SparseTransferMLP, TransferAdapters

    model = load_qwen35_text_causal_lm(
        config.sources.backbone.local_path, dtype=dtype, device="cpu"
    )
    freeze_module(model)
    mtp = None
    # Read-only benchmark fixtures predate LossConfig and intentionally provide
    # only architecture/source fields.  Validated TrainConfig instances always
    # carry losses.mtp; treating an absent fixture field as disabled preserves
    # that narrow builder-only compatibility surface.
    mtp_weight = float(getattr(getattr(config, "losses", None), "mtp", 0.0))
    if mtp_weight > 0:
        # The published Qwen3.5 MTP block is a frozen source component.  Its
        # forward remains differentiable with respect to the student hidden
        # state, but its 15 checkpoint tensors never enter the optimizer.
        mtp = load_qwen35_mtp(
            config.sources.backbone.local_path,
            dtype=dtype,
            device="cpu",
            trainable=False,
        )
        if any(parameter.requires_grad for parameter in mtp.parameters()):
            raise BuildError("native MTP source parameters must remain frozen")
    mapping = load_layer_mapping(config.architecture.layer_map_path, config.architecture.student_layers)
    channel_maps = load_channel_mapping(
        config.architecture.channel_map_path,
        config.architecture.student_layers,
        config.architecture.num_experts,
        config.architecture.expert_intermediate_size,
    )
    modules: list[Any] = []
    active_layers = set(config.architecture.active_layers())
    if config.stage == "dense-oracle":
        runtime = getattr(config, "runtime", None)
        dense_transfer_execution = str(
            getattr(runtime, "dense_transfer_execution", "expanded")
        )
        adapter_init = _load_adapter_initialization(
            config.architecture.adapter_init_path, config.architecture.student_layers
        )
        for student_layer, donor_layer in enumerate(mapping):
            if student_layer not in active_layers:
                continue
            if config.architecture.expert_initialization == "donor":
                weights = (
                    _shared_donor_mlp_weights(donor_text_model, donor_layer)
                    if donor_text_model is not None
                    else load_donor_mlp_weights(
                        config.sources.donor.local_path,
                        donor_layer,
                    )
                )
            else:
                weights = _random_control_mlp_weights(
                    config,
                    donor_layer,
                    dtype=dtype,
                )
            a, b = adapter_init[student_layer]
            donor_gate = weights["gate_proj.weight"]
            donor_up = weights["up_proj.weight"]
            donor_down = weights["down_proj.weight"]
            if donor_text_model is None:
                donor_gate = donor_gate.to(dtype=dtype)
                donor_up = donor_up.to(dtype=dtype)
                donor_down = donor_down.to(dtype=dtype)
            elif any(
                weight.dtype != dtype for weight in (donor_gate, donor_up, donor_down)
            ):
                raise BuildError(
                    "shared donor/teacher projections must already use the training dtype"
                )
            adapters = TransferAdapters(
                config.architecture.student_hidden_size,
                config.architecture.donor_hidden_size,
                input_weight=a,
                output_weight=b,
                # AdamW state and the small adapter updates remain FP32. CUDA
                # autocast still executes the matrix multiplications in BF16.
                dtype=torch.float32,
            )
            shared = model.model.layers[student_layer].mlp
            module = SharedDenseTransferMLP(
                shared,
                donor_gate,
                donor_up,
                donor_down,
                channel_maps[student_layer],
                adapters=adapters,
                branch_scale=0.01,
                execution_mode=dense_transfer_execution,
                # The engine selects the exact per-layer complement of outer
                # decoder checkpointing before the first graph execution.
                checkpoint_token_branch=False,
            )
            model.model.layers[student_layer].mlp = module
            modules.append(module)
    else:
        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("safetensors is required") from exc
        with safe_open(
            config.sources.folded_experts_path, framework="pt", device="cpu"
        ) as handle:
            for student_layer in range(config.architecture.student_layers):
                if student_layer not in active_layers:
                    continue
                gate, up, down, router_weight, scale = _load_folded_layer(
                    handle,
                    student_layer,
                    expected_experts=config.architecture.num_experts,
                    expected_intermediate=config.architecture.expert_intermediate_size,
                    expected_hidden=config.architecture.student_hidden_size,
                )
                if router_weight is None:
                    # With all eight experts active, zero logits give exactly
                    # uniform 1/E weights.  Coupled with the E-fold native
                    # branch scale below, the first sparse forward is therefore
                    # algebraically identical to the folded dense checkpoint.
                    router_weight = torch.zeros(
                        config.architecture.num_experts,
                        config.architecture.student_hidden_size,
                        dtype=torch.float32,
                    )
                shared = model.model.layers[student_layer].mlp
                module = SparseTransferMLP(
                    shared,
                    gate.to(dtype=dtype),
                    up.to(dtype=dtype),
                    down.to(dtype=dtype),
                    router_weight.to(dtype=torch.float32),
                    top_k=config.architecture.num_experts,
                    norm_topk_prob=config.architecture.norm_topk_prob,
                    lora_rank=config.architecture.lora_rank,
                    lora_trainable_dtype=torch.float32,
                    # Native Qwen normalizes selected router probabilities to
                    # sum to one, whereas Stage B sums E slices.  Store the
                    # compensation in the mergeable static scale so top-E with
                    # a uniform router exactly preserves the dense result and
                    # the final top-2 export needs no runtime-only multiplier.
                    branch_scale=scale * config.architecture.num_experts,
                )
                model.model.layers[student_layer].mlp = module
                modules.append(module)
    if device != "cpu":
        model.to(device=device)
        if mtp is not None:
            mtp.to(device=device)
    transfer_modules = tuple(modules)
    # Validate classification now, but do not retain Parameter references:
    # composable FSDP2 replaces them with DTensor Parameters during sharding.
    build_parameter_groups(config, transfer_modules)
    return BuiltModel(
        model=model,
        transfer_modules=transfer_modules,
        student_layer_indices=tuple(sorted(active_layers)),
        donor_teacher_shared=bool(
            donor_text_model is not None
            and config.stage == "dense-oracle"
            and config.architecture.expert_initialization == "donor"
        ),
        mtp=mtp,
    )


def iter_trainable_named_parameters(model: Any) -> Iterator[tuple[str, Any]]:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            yield name, parameter


def trainable_state_dict(model: Any) -> dict[str, Any]:
    return {name: parameter.detach() for name, parameter in iter_trainable_named_parameters(model)}


def load_trainable_state_dict(model: Any, state: Mapping[str, Any]) -> None:
    import torch

    expected = dict(iter_trainable_named_parameters(model))
    if set(expected) != set(state):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise BuildError(f"trainable state mismatch; missing={missing[:3]}, extra={extra[:3]}")
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))
