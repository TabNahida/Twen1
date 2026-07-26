from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3_5MoeTextConfig, Qwen3_5TextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeSparseMoeBlock,
)

from twen.modeling import export_native_moe_mtp_state
from twen.modeling.errors import ExportError, ShapeError
from twen.modeling.mtp import Qwen35MTP


def _dense_config() -> Qwen3_5TextConfig:
    config = Qwen3_5TextConfig.from_dict(
        {
            "model_type": "qwen3_5_text",
            "vocab_size": 41,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "hidden_act": "silu",
            "layer_types": ["linear_attention", "full_attention"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "tie_word_embeddings": True,
        }
    )
    config._attn_implementation = "eager"
    return config


def _dense_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(31)
    module = Qwen35MTP(_dense_config())
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-0.25, 0.25)
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _native_mlp_state(exported: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "mtp.layers.0.mlp."
    return {
        name.removeprefix(prefix): value
        for name, value in exported.items()
        if name.startswith(prefix)
    }


def test_dense_mtp_exports_exact_native_fused_sparse_schema() -> None:
    dense = _dense_state()
    exported = export_native_moe_mtp_state(dense, expert_intermediate_size=4)
    mlp = "mtp.layers.0.mlp"

    assert len(exported) == 19
    assert f"{mlp}.gate_proj.weight" not in exported
    assert f"{mlp}.up_proj.weight" not in exported
    assert f"{mlp}.down_proj.weight" not in exported
    assert tuple(exported[f"{mlp}.experts.gate_up_proj"].shape) == (8, 8, 16)
    assert tuple(exported[f"{mlp}.experts.down_proj"].shape) == (8, 16, 4)
    assert tuple(exported[f"{mlp}.gate.weight"].shape) == (8, 16)
    assert tuple(exported[f"{mlp}.shared_expert_gate.weight"].shape) == (1, 16)
    assert all(value.dtype == torch.float32 for value in exported.values())
    assert all(value.is_contiguous() for value in exported.values())

    for name, value in dense.items():
        if not name.startswith("layers.0.mlp."):
            torch.testing.assert_close(exported[f"mtp.{name}"], value, rtol=0, atol=0)
    torch.testing.assert_close(
        exported[f"{mlp}.shared_expert.gate_proj.weight"],
        dense["layers.0.mlp.gate_proj.weight"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        exported[f"{mlp}.shared_expert.up_proj.weight"],
        dense["layers.0.mlp.up_proj.weight"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        exported[f"{mlp}.shared_expert.down_proj.weight"],
        dense["layers.0.mlp.down_proj.weight"] * 2.0,
        rtol=0,
        atol=0,
    )
    for name in (
        f"{mlp}.shared_expert_gate.weight",
        f"{mlp}.gate.weight",
        f"{mlp}.experts.gate_up_proj",
        f"{mlp}.experts.down_proj",
    ):
        assert torch.count_nonzero(exported[name]).item() == 0
    # Exported unchanged tensors are values, not aliases into the source state.
    assert (
        exported["mtp.fc.weight"].untyped_storage().data_ptr()
        != dense["fc.weight"].untyped_storage().data_ptr()
    )


def test_native_sparse_mtp_mlp_is_algebraically_equal_to_dense_mtp_mlp() -> None:
    dense = _dense_state()
    exported = export_native_moe_mtp_state(dense, expert_intermediate_size=4)
    config = Qwen3_5MoeTextConfig(
        hidden_size=16,
        intermediate_size=32,
        shared_expert_intermediate_size=32,
        moe_intermediate_size=4,
        num_experts=8,
        num_experts_per_tok=2,
        hidden_act="silu",
    )
    native_block = Qwen3_5MoeSparseMoeBlock(config)
    native_block.load_state_dict(_native_mlp_state(exported), strict=True)

    hidden_states = torch.randn(3, 7, 16)
    dense_gate = F.linear(hidden_states, dense["layers.0.mlp.gate_proj.weight"])
    dense_up = F.linear(hidden_states, dense["layers.0.mlp.up_proj.weight"])
    expected = F.linear(
        F.silu(dense_gate) * dense_up,
        dense["layers.0.mlp.down_proj.weight"],
    )
    actual = native_block(hidden_states)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_mtp_export_accepts_prefixed_input_and_applies_one_target_dtype() -> None:
    dense = _dense_state()
    prefixed = {f"mtp.{name}": value for name, value in dense.items()}
    exported = export_native_moe_mtp_state(
        prefixed,
        expert_intermediate_size=4,
        target_dtype=torch.bfloat16,
    )

    assert all(value.dtype == torch.bfloat16 for value in exported.values())
    torch.testing.assert_close(
        exported["mtp.fc.weight"],
        dense["fc.weight"].to(torch.bfloat16),
        rtol=0,
        atol=0,
    )


def _production_meta_state() -> dict[str, torch.Tensor]:
    device = torch.device("meta")
    hidden = 1024
    shared_intermediate = 3584
    return {
        "fc.weight": torch.empty(hidden, hidden * 2, device=device),
        "layers.0.input_layernorm.weight": torch.empty(hidden, device=device),
        "layers.0.mlp.down_proj.weight": torch.empty(
            hidden, shared_intermediate, device=device
        ),
        "layers.0.mlp.gate_proj.weight": torch.empty(
            shared_intermediate, hidden, device=device
        ),
        "layers.0.mlp.up_proj.weight": torch.empty(
            shared_intermediate, hidden, device=device
        ),
        "layers.0.post_attention_layernorm.weight": torch.empty(hidden, device=device),
        "layers.0.self_attn.k_norm.weight": torch.empty(256, device=device),
        "layers.0.self_attn.k_proj.weight": torch.empty(512, hidden, device=device),
        "layers.0.self_attn.o_proj.weight": torch.empty(hidden, 2048, device=device),
        "layers.0.self_attn.q_norm.weight": torch.empty(256, device=device),
        "layers.0.self_attn.q_proj.weight": torch.empty(4096, hidden, device=device),
        "layers.0.self_attn.v_proj.weight": torch.empty(512, hidden, device=device),
        "norm.weight": torch.empty(hidden, device=device),
        "pre_fc_norm_embedding.weight": torch.empty(hidden, device=device),
        "pre_fc_norm_hidden.weight": torch.empty(hidden, device=device),
    }


def test_production_mtp_export_shapes_are_vllm_fused_layout_on_meta() -> None:
    exported = export_native_moe_mtp_state(
        _production_meta_state(),
        target_dtype=torch.bfloat16,
    )
    mlp = "mtp.layers.0.mlp"

    assert len(exported) == 19
    assert tuple(exported[f"{mlp}.experts.gate_up_proj"].shape) == (8, 3072, 1024)
    assert tuple(exported[f"{mlp}.experts.down_proj"].shape) == (8, 1024, 1536)
    assert tuple(exported[f"{mlp}.shared_expert.gate_proj.weight"].shape) == (
        3584,
        1024,
    )
    assert tuple(exported[f"{mlp}.shared_expert.down_proj.weight"].shape) == (
        1024,
        3584,
    )
    assert all(value.device.type == "meta" for value in exported.values())
    assert all(value.dtype == torch.bfloat16 for value in exported.values())


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape", "dtype"])
def test_mtp_export_rejects_corrupt_dense_state(corruption: str) -> None:
    dense = _dense_state()
    if corruption == "missing":
        dense.pop("norm.weight")
        error = ExportError
        match = "missing"
    elif corruption == "unexpected":
        dense["extra.weight"] = torch.ones(1)
        error = ExportError
        match = "unexpected"
    elif corruption == "shape":
        dense["layers.0.self_attn.o_proj.weight"] = torch.empty(16, 15)
        error = ShapeError
        match = "must have shape"
    else:
        dense["norm.weight"] = dense["norm.weight"].to(torch.bfloat16)
        error = ExportError
        match = "one dtype"

    with pytest.raises(error, match=match):
        export_native_moe_mtp_state(dense, expert_intermediate_size=4)


def test_mtp_export_rejects_non_tensor_and_non_floating_target_dtype() -> None:
    non_tensor = _dense_state()
    non_tensor["norm.weight"] = object()  # type: ignore[assignment]
    with pytest.raises(TypeError, match="must be a Tensor"):
        export_native_moe_mtp_state(non_tensor, expert_intermediate_size=4)

    with pytest.raises(TypeError, match="target_dtype must be floating point"):
        export_native_moe_mtp_state(
            _dense_state(),
            expert_intermediate_size=4,
            target_dtype=torch.int64,
        )
