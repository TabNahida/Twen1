from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3_5TextConfig
from transformers.masking_utils import create_causal_mask

from twen.model_loading import CheckpointFormatError, load_qwen35_mtp
from twen.modeling.mtp import Qwen35MTP

_NATIVE_MTP_KEYS = {
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


def _raw_text_config() -> dict[str, object]:
    return {
        "model_type": "qwen3_5_text",
        "dtype": "float32",
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
        "use_cache": True,
    }


def _config() -> Qwen3_5TextConfig:
    config = Qwen3_5TextConfig.from_dict(_raw_text_config())
    config._attn_implementation = "eager"
    return config


def _native_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(17)
    module = Qwen35MTP(_config())
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-0.2, 0.2)
    return {
        name: value.detach().clone().contiguous() for name, value in module.state_dict().items()
    }


def _write_checkpoint(root: Path, mtp_state: dict[str, torch.Tensor]) -> None:
    root.mkdir()
    parent_config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": _raw_text_config(),
        "tie_word_embeddings": True,
    }
    (root / "config.json").write_text(json.dumps(parent_config), encoding="utf-8")
    tensors = {f"mtp.{name}": value for name, value in mtp_state.items()}
    # A non-MTP tensor proves that strictness is scoped to the top-level MTP
    # component rather than rejecting the rest of an official checkpoint.
    tensors["model.language_model.embed_tokens.weight"] = torch.randn(41, 16)
    save_file(tensors, root / "model.safetensors")


def test_native_mtp_state_is_exact_and_vocabulary_modules_stay_external() -> None:
    module = Qwen35MTP(_config())
    assert set(module.state_dict()) == _NATIVE_MTP_KEYS
    assert len(tuple(module.parameters())) == 15
    assert all(not parameter.requires_grad for parameter in module.parameters())
    assert module.training is False
    assert module.layers[0].block_type == "full_attention"
    assert hasattr(module.layers[0], "self_attn")
    assert not hasattr(module.layers[0], "linear_attn")

    embed_tokens = torch.nn.Embedding(41, 16)
    lm_head = torch.nn.Linear(16, 41, bias=False)
    lm_head.weight = embed_tokens.weight
    external_weight_id = id(embed_tokens.weight)
    external_storage = embed_tokens.weight.untyped_storage().data_ptr()

    assert external_weight_id not in {id(parameter) for parameter in module.parameters()}
    assert all("embed_tokens" not in name and "lm_head" not in name for name in module.state_dict())
    assert lm_head.weight.untyped_storage().data_ptr() == external_storage

    hidden = torch.randn(2, 3, 16)
    logits = module.compute_logits(hidden, lm_head)
    torch.testing.assert_close(logits, torch.nn.functional.linear(hidden, embed_tokens.weight))
    assert lm_head.weight.untyped_storage().data_ptr() == external_storage


def test_native_mtp_defaults_to_sdpa_without_changing_checkpoint_schema() -> None:
    config = Qwen3_5TextConfig.from_dict(_raw_text_config())
    # Raw/legacy checkpoint configs carry ``None`` here.  Because Qwen35MTP
    # constructs its decoder layer directly, leaving this unresolved used to
    # select Transformers' eager fallback instead of PreTrainedModel's SDPA
    # default.
    assert config._attn_implementation is None

    module = Qwen35MTP(config)

    assert config._attn_implementation is None
    assert module.config._attn_implementation == "sdpa"
    assert set(module.state_dict()) == _NATIVE_MTP_KEYS


def test_native_mtp_default_sdpa_is_causal_under_future_perturbations() -> None:
    torch.manual_seed(31)
    config = Qwen3_5TextConfig.from_dict(_raw_text_config())
    module = Qwen35MTP(config)
    embed_tokens = torch.nn.Embedding(41, 16)
    input_ids = torch.tensor([[2, 3, 5, 7, 11, 13, 17]])
    attention_mask = torch.ones_like(input_ids)
    hidden_states = torch.randn(1, 7, 16)

    changed_ids = input_ids.clone()
    changed_ids[:, 4:] = torch.tensor([[19, 23, 29]])
    changed_hidden = hidden_states.clone()
    changed_hidden[:, 3:] += 3.0

    with torch.inference_mode():
        baseline = module(
            hidden_states,
            input_ids,
            embed_tokens=embed_tokens,
            attention_mask=attention_mask,
        )
        changed = module(
            changed_hidden,
            changed_ids,
            embed_tokens=embed_tokens,
            attention_mask=attention_mask,
        )

    assert module.config._attn_implementation == "sdpa"
    # MTP output i consumes hidden[i] and token[i+1].  Therefore hidden[3:]
    # and token[4:] are strictly future inputs for output positions 0..2.
    torch.testing.assert_close(changed[:, :3], baseline[:, :3], rtol=1e-6, atol=1e-6)
    assert not torch.allclose(changed[:, 3:], baseline[:, 3:], rtol=1e-5, atol=1e-6)


def test_native_mtp_preserves_explicit_eager_attention() -> None:
    module = Qwen35MTP(_config())

    assert module.config._attn_implementation == "eager"


@pytest.mark.parametrize("right_padding", [False, True], ids=["causal", "causal-padding"])
def test_native_mtp_cpu_sdpa_matches_eager_outputs_and_input_gradients(
    right_padding: bool,
) -> None:
    torch.manual_seed(29)
    eager = Qwen35MTP(_config())
    default_config = Qwen3_5TextConfig.from_dict(_raw_text_config())
    sdpa = Qwen35MTP(default_config)
    sdpa.load_state_dict(eager.state_dict(), strict=True)

    eager_embedding = torch.nn.Embedding(41, 16)
    sdpa_embedding = torch.nn.Embedding(41, 16)
    sdpa_embedding.load_state_dict(eager_embedding.state_dict(), strict=True)
    eager_embedding.requires_grad_(False)
    sdpa_embedding.requires_grad_(False)
    input_ids = torch.tensor([[2, 3, 5, 7, 11], [13, 17, 19, 23, 29]])
    attention_mask = torch.ones_like(input_ids)
    if right_padding:
        input_ids[1, 3:] = 0
        attention_mask[1, 3:] = 0
    eager_hidden = torch.randn(2, 5, 16, requires_grad=True)
    sdpa_hidden = eager_hidden.detach().clone().requires_grad_(True)

    eager_output = eager(
        eager_hidden,
        input_ids,
        embed_tokens=eager_embedding,
        attention_mask=attention_mask,
    )
    sdpa_output = sdpa(
        sdpa_hidden,
        input_ids,
        embed_tokens=sdpa_embedding,
        attention_mask=attention_mask,
    )
    torch.testing.assert_close(sdpa_output, eager_output, rtol=1e-5, atol=1e-6)

    output_gradient = torch.randn_like(eager_output)
    eager_gradient = torch.autograd.grad(
        (eager_output * output_gradient).sum(), eager_hidden
    )[0]
    sdpa_gradient = torch.autograd.grad(
        (sdpa_output * output_gradient).sum(), sdpa_hidden
    )[0]
    torch.testing.assert_close(sdpa_gradient, eager_gradient, rtol=1e-5, atol=1e-6)


def test_strict_loader_reads_all_native_mtp_tensors_and_freezes_by_default(
    tmp_path: Path,
) -> None:
    expected = _native_state()
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint, expected)

    loaded = load_qwen35_mtp(checkpoint, dtype=torch.float32)

    assert set(loaded.state_dict()) == _NATIVE_MTP_KEYS
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert loaded.training is False

    converted = load_qwen35_mtp(checkpoint, dtype=torch.bfloat16)
    assert {value.dtype for value in converted.state_dict().values()} == {torch.bfloat16}


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape", "dtype"])
def test_strict_loader_rejects_mtp_schema_corruption(
    tmp_path: Path,
    corruption: str,
) -> None:
    state = _native_state()
    if corruption == "missing":
        state.pop("norm.weight")
        match = "missing"
    elif corruption == "unexpected":
        state["unexpected.weight"] = torch.ones(1)
        match = "unexpected"
    elif corruption == "shape":
        state["fc.weight"] = state["fc.weight"][:-1].contiguous()
        match = "has shape"
    else:
        state["norm.weight"] = state["norm.weight"].to(torch.bfloat16)
        match = "dtype"
    checkpoint = tmp_path / corruption
    _write_checkpoint(checkpoint, state)

    with pytest.raises(CheckpointFormatError, match=match):
        load_qwen35_mtp(checkpoint, dtype=torch.float32)


class _RecordingEmbedding(torch.nn.Embedding):
    last_input_ids: torch.Tensor | None

    def __init__(self, vocabulary: int, hidden: int) -> None:
        super().__init__(vocabulary, hidden)
        self.last_input_ids = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.last_input_ids = input_ids.detach().clone()
        return super().forward(input_ids)


def _reference_mtp_forward(
    module: Qwen35MTP,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    embed_tokens: torch.nn.Module,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    shifted_hidden = hidden_states[:, :-1]
    shifted_embedding = embed_tokens(input_ids[:, 1:])
    fused = module.fc(
        torch.cat(
            (
                module.pre_fc_norm_embedding(shifted_embedding),
                module.pre_fc_norm_hidden(shifted_hidden),
            ),
            dim=-1,
        )
    )
    pair_mask = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
    text_positions = position_ids[0, ..., 1:]
    rope_positions = position_ids[1:, ..., 1:]
    causal_mask = create_causal_mask(
        config=module.config,
        inputs_embeds=fused,
        attention_mask=pair_mask,
        past_key_values=None,
        position_ids=text_positions,
    )
    rotary = module.rotary_emb(fused, rope_positions)
    for layer in module.layers:
        fused = layer(
            fused,
            position_embeddings=rotary,
            attention_mask=causal_mask,
            position_ids=text_positions,
            past_key_values=None,
            use_cache=False,
        )
    return module.norm(fused)


def test_mtp_positions_follow_the_shifted_known_token() -> None:
    batch_size = 2
    sequence_length = 5
    expected_default = torch.arange(1, sequence_length).expand(batch_size, -1)

    text_positions, rope_positions = Qwen35MTP._shift_position_ids(
        None,
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(text_positions, expected_default)
    torch.testing.assert_close(
        rope_positions,
        expected_default.unsqueeze(0).expand(3, -1, -1),
    )

    explicit_text = torch.tensor(
        [
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
        ]
    )
    text_positions, rope_positions = Qwen35MTP._shift_position_ids(
        explicit_text,
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(text_positions, explicit_text[:, 1:])
    torch.testing.assert_close(
        rope_positions,
        explicit_text[:, 1:].unsqueeze(0).expand(3, -1, -1),
    )

    explicit_mrope = torch.stack(
        tuple(explicit_text + 100 * axis for axis in range(4))
    )
    text_positions, rope_positions = Qwen35MTP._shift_position_ids(
        explicit_mrope,
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(text_positions, explicit_mrope[0, ..., 1:])
    torch.testing.assert_close(rope_positions, explicit_mrope[1:, ..., 1:])

    text_positions, rope_positions = Qwen35MTP._shift_position_ids(
        explicit_mrope[1:],
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=torch.device("cpu"),
    )
    assert text_positions is None
    torch.testing.assert_close(rope_positions, explicit_mrope[1:, ..., 1:])


def test_mtp_forward_shifts_tokens_matches_reference_and_preserves_autograd() -> None:
    torch.manual_seed(23)
    module = Qwen35MTP(_config())
    embed_tokens = _RecordingEmbedding(41, 16)
    input_ids = torch.tensor([[2, 3, 5, 7, 11], [13, 17, 19, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    text_positions = torch.arange(5).view(1, -1).expand(2, -1)
    position_ids = text_positions.unsqueeze(0).expand(4, -1, -1).clone()
    hidden_states = torch.randn(2, 5, 16, requires_grad=True)

    actual = module(
        hidden_states,
        input_ids,
        embed_tokens=embed_tokens,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    assert embed_tokens.last_input_ids is not None
    torch.testing.assert_close(embed_tokens.last_input_ids, input_ids[:, 1:])
    assert tuple(actual.shape) == (2, 4, 16)
    torch.testing.assert_close(
        module.shifted_attention_mask(attention_mask),
        torch.tensor([[True, True, True, True], [True, True, False, False]]),
    )

    reference = _reference_mtp_forward(
        module,
        hidden_states,
        input_ids,
        embed_tokens,
        attention_mask,
        position_ids,
    )
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)

    # A 2D text position tensor is equivalent to four identical Qwen3.5 MRoPE
    # position rows and exercises the HF-compatible position interface.
    actual_2d_positions = module(
        hidden_states,
        input_ids,
        embed_tokens=embed_tokens,
        attention_mask=attention_mask,
        position_ids=text_positions,
    )
    torch.testing.assert_close(actual, actual_2d_positions, rtol=1e-6, atol=1e-6)
    actual_3d_positions = module(
        hidden_states,
        input_ids,
        embed_tokens=embed_tokens,
        attention_mask=attention_mask,
        position_ids=position_ids[1:],
    )
    torch.testing.assert_close(actual, actual_3d_positions, rtol=1e-6, atol=1e-6)

    actual.square().sum().backward()
    assert hidden_states.grad is not None
    assert torch.count_nonzero(hidden_states.grad[:, :-1]).item() > 0
    assert torch.count_nonzero(hidden_states.grad[:, -1]).item() == 0
    assert embed_tokens.weight.grad is not None
    assert all(parameter.grad is None for parameter in module.parameters())
