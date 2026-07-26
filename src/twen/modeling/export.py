"""Helpers for a native Transformers Qwen3.5-MoE config and state dict."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from ._lazy import require_torch, require_transformers
from .errors import ExportError, ShapeError
from .folding import FoldedExperts

_DENSE_MTP_KEYS = frozenset(
    {
        "fc.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.mlp.down_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.self_attn.k_norm.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.self_attn.q_norm.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "norm.weight",
        "pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight",
    }
)


def _config_dict(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(config, Mapping):
        raw = copy.deepcopy(dict(config))
    elif hasattr(config, "to_dict"):
        raw = copy.deepcopy(dict(config.to_dict()))
    else:
        raise TypeError("backbone_config must be a mapping or expose to_dict()")
    if isinstance(raw.get("text_config"), Mapping):
        raw = copy.deepcopy(dict(raw["text_config"]))
    return raw


def build_native_moe_config(
    backbone_config: Mapping[str, Any] | Any,
    *,
    num_experts: int = 8,
    experts_per_token: int = 2,
    expert_intermediate_size: int = 1536,
    shared_expert_intermediate_size: int = 3584,
    norm_topk_prob: bool = True,
    router_aux_loss_coef: float = 0.001,
    as_transformers_config: bool = False,
) -> dict[str, Any] | Any:
    """Convert the audited 0.8B text config to native Qwen3.5-MoE fields.

    The default return value is a plain JSON-serializable dict and therefore does
    not import Transformers. ``as_transformers_config=True`` performs only local
    class construction; it never calls ``from_pretrained``.
    """

    raw = _config_dict(backbone_config)
    expected = {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "intermediate_size": 3584,
    }
    mismatched = {name: raw.get(name) for name, value in expected.items() if raw.get(name) != value}
    if mismatched:
        raise ExportError(
            f"Native export requires an audited Qwen3.5-0.8B text config; mismatches={mismatched}"
        )
    if num_experts <= 0 or expert_intermediate_size <= 0:
        raise ValueError("Expert count and intermediate size must be positive")
    if not 1 <= experts_per_token <= num_experts:
        raise ValueError("experts_per_token must be between 1 and num_experts")
    if (num_experts, experts_per_token, expert_intermediate_size) != (8, 2, 1536):
        raise ExportError("v1 native export requires exactly 8 experts of width 1536 with top-2")
    if not norm_topk_prob:
        raise ExportError("native Qwen3.5-MoE requires norm_topk_prob=true")
    if shared_expert_intermediate_size != 3584:
        raise ExportError(
            "The v1 exact-shared recipe requires shared_expert_intermediate_size=3584"
        )
    raw.update(
        {
            # Text-only v1 must use the text config. ``qwen3_5_moe`` is the
            # multimodal parent config and would instantiate a vision tower.
            "model_type": "qwen3_5_moe_text",
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "intermediate_size": 3584,
            "shared_expert_intermediate_size": int(shared_expert_intermediate_size),
            "moe_intermediate_size": int(expert_intermediate_size),
            "num_experts": int(num_experts),
            "num_experts_per_tok": int(experts_per_token),
            "norm_topk_prob": bool(norm_topk_prob),
            "router_aux_loss_coef": float(router_aux_loss_coef),
            "output_router_logits": False,
        }
    )
    # Dense-only implementation metadata must not make the native class choose a
    # dense MLP path. Unknown keys are avoided so save_pretrained remains clean.
    raw.pop("text_config", None)
    if not as_transformers_config:
        return raw

    transformers = require_transformers("native Qwen3.5-MoE config construction")
    config_class = None
    for name in ("Qwen3_5MoeTextConfig",):
        config_class = getattr(transformers, name, None)
        if config_class is not None:
            break
    if config_class is None:
        raise ExportError(
            "Installed Transformers does not expose Qwen3_5MoeTextConfig; use the "
            "project-pinned Transformers version or keep the returned dict"
        )
    kwargs = dict(raw)
    kwargs.pop("model_type", None)
    try:
        return config_class(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ExportError(f"Transformers rejected the generated MoE config: {exc}") from exc


def _layer_value(values: Mapping[int, Any] | Sequence[Any], layer: int, name: str) -> Any:
    try:
        return values[layer]
    except (KeyError, IndexError, TypeError) as exc:
        raise ExportError(f"{name} is missing layer {layer}") from exc


def _find_layer_prefix(state: Mapping[str, Any]) -> str:
    suffix = "layers.0.mlp.gate_proj.weight"
    candidates = sorted(key[: -len(suffix)] for key in state if key.endswith(suffix))
    if len(candidates) != 1:
        raise ExportError(
            "Could not uniquely discover the dense text layer prefix; pass layer_prefix explicitly. "
            f"Candidates={candidates}"
        )
    return candidates[0]


def _text_only_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize official conditional-generation keys to text CausalLM keys."""

    result: dict[str, Any] = {}
    for name, value in state.items():
        if name.startswith("model.language_model."):
            target = "model." + name.removeprefix("model.language_model.")
        elif name.startswith(("model.visual.", "visual.", "model.mtp.", "mtp.")):
            continue
        else:
            target = name
        if target.startswith(("model.visual.", "model.mtp.", "visual.", "mtp.")):
            continue
        if target in result:
            raise ExportError(
                f"Text-only key normalization produces duplicate tensor {target!r}"
            )
        result[target] = value
    return result


def _folded_tensors(value: Any) -> tuple[Any, Any, Any]:
    if isinstance(value, FoldedExperts):
        return value.gate_proj, value.up_proj, value.down_proj
    if hasattr(value, "merged_weights"):
        result = value.merged_weights()
        if not isinstance(result, tuple) or len(result) != 3:
            raise ExportError("merged_weights() must return (gate, up, down)")
        return result
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return value[0], value[1], value[2]
    if all(hasattr(value, name) for name in ("gate_proj", "up_proj", "down_proj")):
        return value.gate_proj, value.up_proj, value.down_proj
    raise TypeError(
        "Each folded layer must be FoldedExperts, MergeableExpertLoRA, or a (gate, up, down) tuple"
    )


def _router_tensor(value: Any) -> Any:
    torch = require_torch("native MoE state export")
    weight = getattr(value, "weight", value)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ShapeError("Every router must be a rank-2 Tensor or module exposing .weight")
    return weight.detach()


def export_native_moe_state(
    backbone_state: Mapping[str, Any],
    folded_experts: Mapping[int, Any] | Sequence[Any],
    router_weights: Mapping[int, Any] | Sequence[Any],
    *,
    num_layers: int = 24,
    layer_prefix: str | None = None,
    layout: str = "fused",
    branch_scales: Mapping[int, float | Any] | Sequence[float | Any] | None = None,
    target_dtype: Any | None = None,
    text_only: bool = True,
) -> dict[str, Any]:
    """Replace dense MLP keys with native shared/router/folded-expert keys.

    The default ``fused`` layout matches Qwen3.5-MoE's expert tensors:
    ``gate_up_proj=[E,2I,H]`` and ``down_proj=[E,H,I]``. ``individual`` is a
    diagnostic layout of per-expert Linear weights and is not the v1 deliverable.

    Shared down×2, branch-scale merge, LoRA merge, and orientation conversion are
    all computed in FP32. ``target_dtype`` (normally bfloat16) is applied last.
    """

    torch = require_torch("native MoE state export")
    if num_layers != 24:
        raise ExportError("The v1 Qwen3.5-0.8B backbone must export exactly 24 layers")
    if layout not in {"fused", "individual"}:
        raise ValueError("layout must be 'fused' or 'individual'")
    if not backbone_state:
        raise ExportError("backbone_state cannot be empty")
    source_state = _text_only_state(backbone_state) if text_only else dict(backbone_state)
    prefix = _find_layer_prefix(source_state) if layer_prefix is None else layer_prefix
    if prefix and not prefix.endswith("."):
        prefix += "."

    result: dict[str, Any] = {}
    for name, value in source_state.items():
        if isinstance(value, torch.Tensor):
            cloned = value.detach().clone()
            if target_dtype is not None and cloned.is_floating_point():
                cloned = cloned.to(dtype=target_dtype)
            result[name] = cloned
        else:
            result[name] = copy.deepcopy(value)

    for layer in range(num_layers):
        mlp = f"{prefix}layers.{layer}.mlp"
        dense_gate_key = f"{mlp}.gate_proj.weight"
        dense_up_key = f"{mlp}.up_proj.weight"
        dense_down_key = f"{mlp}.down_proj.weight"
        missing = [
            key
            for key in (dense_gate_key, dense_up_key, dense_down_key)
            if key not in source_state
        ]
        if missing:
            raise ExportError(f"Layer {layer} is missing dense shared weights: {missing}")
        shared_gate = source_state[dense_gate_key]
        shared_up = source_state[dense_up_key]
        shared_down = source_state[dense_down_key]
        if shared_gate.ndim != 2 or tuple(shared_up.shape) != tuple(shared_gate.shape):
            raise ShapeError(f"Layer {layer} shared gate/up weights are malformed")
        shared_intermediate, hidden = map(int, shared_gate.shape)
        if (hidden, shared_intermediate) != tuple(shared_down.shape):
            raise ShapeError(f"Layer {layer} shared down weight is malformed")
        for key in (dense_gate_key, dense_up_key, dense_down_key):
            result.pop(key, None)

        def final(tensor: Any) -> Any:
            tensor = tensor.detach().to(dtype=torch.float32)
            return tensor.to(dtype=target_dtype) if target_dtype is not None else tensor

        result[f"{mlp}.shared_expert.gate_proj.weight"] = final(shared_gate)
        result[f"{mlp}.shared_expert.up_proj.weight"] = final(shared_up)
        # sigmoid(0)=0.5, hence 2×down restores the exact original shared FFN.
        result[f"{mlp}.shared_expert.down_proj.weight"] = final(
            shared_down.to(torch.float32) * 2.0
        )
        result[f"{mlp}.shared_expert_gate.weight"] = final(
            torch.zeros((1, hidden), device=shared_gate.device, dtype=torch.float32)
        )

        gate, up, down = _folded_tensors(
            _layer_value(folded_experts, layer, "folded_experts")
        )
        if not all(isinstance(x, torch.Tensor) for x in (gate, up, down)):
            raise TypeError(f"Layer {layer} folded projections must be tensors")
        if gate.ndim != 3 or tuple(up.shape) != tuple(gate.shape):
            raise ShapeError(f"Layer {layer} folded gate/up weights are malformed")
        experts, expert_intermediate, expert_hidden = map(int, gate.shape)
        if expert_hidden != hidden or tuple(down.shape) != (
            experts,
            hidden,
            expert_intermediate,
        ):
            raise ShapeError(f"Layer {layer} folded down/hidden dimensions are malformed")
        router = _router_tensor(_layer_value(router_weights, layer, "router_weights"))
        if tuple(router.shape) != (experts, hidden):
            raise ShapeError(
                f"Layer {layer} router shape must be {(experts, hidden)}, got {tuple(router.shape)}"
            )
        result[f"{mlp}.gate.weight"] = final(router)
        scale: float | Any = 1.0
        if branch_scales is not None:
            scale = _layer_value(branch_scales, layer, "branch_scales")
        if isinstance(scale, torch.Tensor):
            if scale.numel() != 1:
                raise ShapeError(f"Layer {layer} branch scale must be scalar")
            scale = float(scale.detach().to(torch.float32).item())
        scale = float(scale)
        gate32 = gate.detach().to(torch.float32)
        up32 = up.detach().to(device=gate32.device, dtype=torch.float32)
        down32 = down.detach().to(device=gate32.device, dtype=torch.float32) * scale
        if layout == "fused":
            # Qwen3_5MoeExperts applies F.linear and chunks the output gate-first,
            # so both fused tensors remain in nn.Linear weight orientation.
            gate_up = torch.cat((gate32, up32), dim=1).contiguous()
            native_down = down32.contiguous()
            result[f"{mlp}.experts.gate_up_proj"] = final(gate_up)
            result[f"{mlp}.experts.down_proj"] = final(native_down)
        else:
            for expert in range(experts):
                base = f"{mlp}.experts.{expert}"
                result[f"{base}.gate_proj.weight"] = final(gate32[expert])
                result[f"{base}.up_proj.weight"] = final(up32[expert])
                result[f"{base}.down_proj.weight"] = final(down32[expert])
    return result


def export_native_moe_mtp_state(
    dense_mtp_state: Mapping[str, Any],
    *,
    num_experts: int = 8,
    expert_intermediate_size: int = 1536,
    target_dtype: Any | None = None,
) -> dict[str, Any]:
    """Convert dense Qwen3.5 MTP tensors to native fused MoE MTP tensors.

    The official dense MTP's FC, attention, and norm tensors are preserved.
    Its dense FFN becomes an exact shared expert: gate/up are copied, down is
    doubled, and a zero shared gate contributes ``sigmoid(0) = 0.5``.  The
    eight routed experts and router are all zero, so the native sparse block is
    algebraically identical to the source dense FFN while satisfying vLLM's
    ``Qwen3_5MoeMTP`` state schema.

    Input may use either the module-local 15 keys or the same keys prefixed by
    ``mtp.``. Mixing layouts, missing keys, extra keys, inconsistent shapes,
    devices, or dtypes fails closed. Output always uses top-level ``mtp.*``
    keys and the fused expert layout.
    """

    torch = require_torch("native Qwen3.5-MoE MTP state export")
    if not isinstance(dense_mtp_state, Mapping) or not dense_mtp_state:
        raise ExportError("dense_mtp_state must be a non-empty mapping")
    if isinstance(num_experts, bool) or num_experts != 8:
        raise ExportError("native Qwen3.5-MoE MTP export requires exactly 8 experts")
    if isinstance(expert_intermediate_size, bool) or expert_intermediate_size <= 0:
        raise ValueError("expert_intermediate_size must be positive")

    supplied_keys = set(dense_mtp_state)
    prefixed_keys = {f"mtp.{name}" for name in _DENSE_MTP_KEYS}
    if supplied_keys == _DENSE_MTP_KEYS:
        source = dict(dense_mtp_state)
    elif supplied_keys == prefixed_keys:
        source = {
            name.removeprefix("mtp."): value for name, value in dense_mtp_state.items()
        }
    else:
        local_keys = {
            name.removeprefix("mtp.") if name.startswith("mtp.") else name
            for name in supplied_keys
        }
        missing = sorted(_DENSE_MTP_KEYS - local_keys)
        unexpected = sorted(local_keys - _DENSE_MTP_KEYS)
        raise ExportError(
            "dense MTP tensor keys differ from the native 15-key schema: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )

    for name, value in source.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"dense MTP tensor {name} must be a Tensor")
        if not value.is_floating_point():
            raise ExportError(f"dense MTP tensor {name} must be floating point")
    source_dtype = source["fc.weight"].dtype
    source_device = source["fc.weight"].device
    inconsistent_dtypes = sorted(
        name for name, value in source.items() if value.dtype != source_dtype
    )
    if inconsistent_dtypes:
        raise ExportError(
            f"dense MTP tensors must share one dtype; mismatches={inconsistent_dtypes[:3]}"
        )
    inconsistent_devices = sorted(
        name for name, value in source.items() if value.device != source_device
    )
    if inconsistent_devices:
        raise ExportError(
            f"dense MTP tensors must share one device; mismatches={inconsistent_devices[:3]}"
        )
    output_dtype = source_dtype if target_dtype is None else target_dtype
    try:
        dtype_probe = torch.empty((), dtype=output_dtype, device="meta")
    except (TypeError, RuntimeError) as exc:
        raise TypeError(f"target_dtype is not a valid torch dtype: {output_dtype}") from exc
    if not dtype_probe.is_floating_point():
        raise TypeError("target_dtype must be floating point")

    def require_shape(name: str, shape: tuple[int, ...]) -> None:
        actual = tuple(source[name].shape)
        if actual != shape:
            raise ShapeError(f"dense MTP tensor {name} must have shape {shape}, got {actual}")

    hidden_shape = tuple(source["pre_fc_norm_hidden.weight"].shape)
    if len(hidden_shape) != 1 or hidden_shape[0] <= 0:
        raise ShapeError("dense MTP hidden norm must be a non-empty vector")
    hidden = int(hidden_shape[0])
    for name in (
        "norm.weight",
        "pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
    ):
        require_shape(name, (hidden,))
    require_shape("fc.weight", (hidden, hidden * 2))

    dense_gate = source["layers.0.mlp.gate_proj.weight"]
    if dense_gate.ndim != 2 or dense_gate.shape[1] != hidden or dense_gate.shape[0] <= 0:
        raise ShapeError(
            "dense MTP gate projection must have shape [shared_intermediate, hidden]"
        )
    shared_intermediate = int(dense_gate.shape[0])
    require_shape("layers.0.mlp.up_proj.weight", (shared_intermediate, hidden))
    require_shape("layers.0.mlp.down_proj.weight", (hidden, shared_intermediate))

    q_norm_shape = tuple(source["layers.0.self_attn.q_norm.weight"].shape)
    if len(q_norm_shape) != 1 or q_norm_shape[0] <= 0:
        raise ShapeError("dense MTP query norm must be a non-empty head vector")
    head_dim = int(q_norm_shape[0])
    require_shape("layers.0.self_attn.k_norm.weight", (head_dim,))
    q_proj = source["layers.0.self_attn.q_proj.weight"]
    k_proj = source["layers.0.self_attn.k_proj.weight"]
    if q_proj.ndim != 2 or q_proj.shape[1] != hidden or q_proj.shape[0] % 2:
        raise ShapeError("dense MTP q_proj must have shape [2 * attention_width, hidden]")
    attention_width = int(q_proj.shape[0] // 2)
    if attention_width <= 0 or attention_width % head_dim:
        raise ShapeError("dense MTP query width must be a positive multiple of head_dim")
    if k_proj.ndim != 2 or k_proj.shape[1] != hidden or k_proj.shape[0] % head_dim:
        raise ShapeError("dense MTP key width must be a positive multiple of head_dim")
    key_value_width = int(k_proj.shape[0])
    if key_value_width <= 0 or attention_width % key_value_width:
        raise ShapeError("dense MTP query/key head counts are incompatible")
    require_shape("layers.0.self_attn.v_proj.weight", (key_value_width, hidden))
    require_shape("layers.0.self_attn.o_proj.weight", (hidden, attention_width))

    def unchanged(value: Any) -> Any:
        return value.detach().to(dtype=output_dtype).clone().contiguous()

    result = {
        f"mtp.{name}": unchanged(value)
        for name, value in source.items()
        if not name.startswith("layers.0.mlp.")
    }
    mlp = "mtp.layers.0.mlp"
    dense_up = source["layers.0.mlp.up_proj.weight"]
    dense_down = source["layers.0.mlp.down_proj.weight"]
    result[f"{mlp}.shared_expert.gate_proj.weight"] = unchanged(dense_gate)
    result[f"{mlp}.shared_expert.up_proj.weight"] = unchanged(dense_up)
    result[f"{mlp}.shared_expert.down_proj.weight"] = (
        dense_down.detach().to(dtype=torch.float32).mul(2.0).to(dtype=output_dtype).contiguous()
    )

    def zeros(shape: tuple[int, ...]) -> Any:
        return torch.zeros(shape, dtype=output_dtype, device=source_device)

    result[f"{mlp}.shared_expert_gate.weight"] = zeros((1, hidden))
    result[f"{mlp}.gate.weight"] = zeros((num_experts, hidden))
    result[f"{mlp}.experts.gate_up_proj"] = zeros(
        (num_experts, expert_intermediate_size * 2, hidden)
    )
    result[f"{mlp}.experts.down_proj"] = zeros(
        (num_experts, hidden, expert_intermediate_size)
    )

    expected_output_keys = {
        f"mtp.{name}" for name in _DENSE_MTP_KEYS if not name.startswith("layers.0.mlp.")
    } | {
        f"{mlp}.shared_expert.gate_proj.weight",
        f"{mlp}.shared_expert.up_proj.weight",
        f"{mlp}.shared_expert.down_proj.weight",
        f"{mlp}.shared_expert_gate.weight",
        f"{mlp}.gate.weight",
        f"{mlp}.experts.gate_up_proj",
        f"{mlp}.experts.down_proj",
    }
    if set(result) != expected_output_keys:  # pragma: no cover - closed-world assertion
        raise AssertionError("native sparse MTP output keys do not match the expected schema")
    return result
