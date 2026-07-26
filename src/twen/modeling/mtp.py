"""Native Qwen3.5 multi-token-prediction building block.

Transformers intentionally ignores the top-level ``mtp.*`` tensors shipped in
Qwen3.5 checkpoints.  This module implements that small text-only component
with Transformers' own decoder and normalization primitives while keeping the
vocabulary embedding and LM head external.  Keeping those modules external is
important: official checkpoints do not contain ``mtp.embed_tokens`` or an MTP
specific LM head when ``mtp_use_dedicated_embeddings=false``.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)


class Qwen35MTP(nn.Module):
    """One-layer native Qwen3.5 MTP predictor without vocabulary parameters.

    For every main-model state ``h_t``, Qwen3.5 combines it with the embedding
    of the already-known next token ``x_(t+1)``.  The resulting MTP state predicts
    ``x_(t+2)`` when projected through the main model's LM head.  Consequently a
    length-``L`` input produces ``L-1`` MTP hidden states, of which the first
    ``L-2`` have in-sequence training targets.

    ``embed_tokens`` and ``lm_head`` are deliberately supplied to methods rather
    than assigned as attributes.  Assigning an external :class:`~torch.nn.Module`
    would register a second vocabulary path in this module's state dict.  The
    MTP parameters are frozen by default, but the forward pass does not use
    ``no_grad``: gradients can still flow from an auxiliary loss into
    ``hidden_states`` and an externally trainable embedding.
    """

    def __init__(self, config: Any, *, trainable: bool = False) -> None:
        super().__init__()
        num_mtp_layers = getattr(config, "mtp_num_hidden_layers", None)
        if isinstance(num_mtp_layers, bool) or num_mtp_layers != 1:
            raise ValueError("phase-one Qwen3.5 MTP requires mtp_num_hidden_layers=1")
        if getattr(config, "mtp_use_dedicated_embeddings", None) is not False:
            raise ValueError(
                "Qwen35MTP requires mtp_use_dedicated_embeddings=false so vocabulary "
                "parameters can be shared"
            )

        mtp_config = copy.deepcopy(config)
        mtp_config.num_hidden_layers = 1
        mtp_config.layer_types = ["full_attention"]
        mtp_config.use_cache = False
        # This module constructs a decoder layer directly instead of going
        # through ``PreTrainedModel``, whose constructor normally resolves an
        # unspecified attention implementation to SDPA.  Leaving the raw
        # checkpoint config unset therefore silently selects Transformers'
        # eager fallback and materializes an O(sequence_length**2) FP32
        # attention matrix.  Preserve explicit reference/debug choices while
        # making the production-equivalent default explicit here.
        if getattr(mtp_config, "_attn_implementation", None) is None:
            mtp_config._attn_implementation = "sdpa"
        self.config = mtp_config

        hidden_size = int(mtp_config.hidden_size)
        self.fc = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(mtp_config, 0)])
        self.norm = Qwen3_5RMSNorm(hidden_size, eps=mtp_config.rms_norm_eps)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(
            hidden_size, eps=mtp_config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(hidden_size, eps=mtp_config.rms_norm_eps)
        # RoPE buffers are non-persistent in Transformers and therefore do not
        # add keys beyond the checkpoint's exact 15 ``mtp.*`` tensors.
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(mtp_config)

        self.requires_grad_(trainable)
        self.train(trainable)

    @staticmethod
    def shifted_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        """Return validity of every ``(h_t, x_(t+1))`` MTP input pair."""

        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        if attention_mask.shape[-1] < 2:
            raise ValueError("MTP requires a sequence length of at least two")
        return attention_mask[:, :-1].ne(0) & attention_mask[:, 1:].ne(0)

    @staticmethod
    def _shift_position_ids(
        position_ids: torch.Tensor | None,
        *,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        mtp_length = sequence_length - 1
        if position_ids is None:
            text_positions = torch.arange(mtp_length, device=device, dtype=torch.long)
            text_positions = text_positions.view(1, -1).expand(batch_size, -1)
            rope_positions = text_positions.unsqueeze(0).expand(3, -1, -1)
            return text_positions, rope_positions

        if position_ids.ndim == 2:
            if tuple(position_ids.shape) != (batch_size, sequence_length):
                raise ValueError("2D position_ids must match [batch, sequence]")
            text_positions = position_ids[:, :-1]
            rope_positions = text_positions.unsqueeze(0).expand(3, -1, -1)
            return text_positions, rope_positions

        if position_ids.ndim != 3 or position_ids.shape[0] not in {3, 4}:
            raise ValueError("position_ids must have shape [B,L], [3,B,L], or [4,B,L]")
        if tuple(position_ids.shape[1:]) != (batch_size, sequence_length):
            raise ValueError("3D position_ids must end in [batch, sequence]")
        shifted = position_ids[..., :-1]
        if shifted.shape[0] == 4:
            return shifted[0], shifted[1:]
        # Match Qwen3_5TextModel: three-axis MRoPE input has no separate text
        # position row for mask/cache handling.
        return None, shifted

    @staticmethod
    def compute_logits(hidden_states: torch.Tensor, lm_head: nn.Module) -> torch.Tensor:
        """Project MTP states through an externally owned vocabulary head."""

        if not isinstance(lm_head, nn.Module):
            raise TypeError("lm_head must be a torch.nn.Module")
        logits = lm_head(hidden_states)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("lm_head must return a Tensor")
        if logits.shape[:-1] != hidden_states.shape[:-1]:
            raise ValueError("lm_head must preserve all non-hidden dimensions")
        return logits

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        embed_tokens: nn.Module,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return native MTP states aligned as ``h_t + embed(x_(t+1))``.

        The returned sequence length is one shorter than ``input_ids``.  Padding
        is applied to attention keys through pairwise validity; callers must
        additionally mask the final MTP state because no ``x_(t+2)`` target is
        present for it.
        """

        if not isinstance(embed_tokens, nn.Module):
            raise TypeError("embed_tokens must be a torch.nn.Module")
        if hidden_states.ndim != 3 or input_ids.ndim != 2:
            raise ValueError("hidden_states/input_ids must have shape [B,L,H] and [B,L]")
        batch_size, sequence_length, hidden_size = hidden_states.shape
        if tuple(input_ids.shape) != (batch_size, sequence_length):
            raise ValueError("input_ids must match hidden_states batch and sequence dimensions")
        if hidden_size != int(self.config.hidden_size):
            raise ValueError(
                f"hidden_states width must be {self.config.hidden_size}, got {hidden_size}"
            )
        if sequence_length < 2:
            raise ValueError("MTP requires a sequence length of at least two")

        pair_mask = None
        if attention_mask is not None:
            if tuple(attention_mask.shape) != (batch_size, sequence_length):
                raise ValueError("attention_mask must match input_ids")
            pair_mask = self.shifted_attention_mask(attention_mask)

        next_token_embeddings = embed_tokens(input_ids[:, 1:])
        expected_embedding_shape = (batch_size, sequence_length - 1, hidden_size)
        if not isinstance(next_token_embeddings, torch.Tensor):
            raise TypeError("embed_tokens must return a Tensor")
        if tuple(next_token_embeddings.shape) != expected_embedding_shape:
            raise ValueError(
                "embed_tokens must return [batch, sequence-1, hidden] for shifted input_ids"
            )

        shifted_hidden = hidden_states[:, :-1]
        normalized_embedding = self.pre_fc_norm_embedding(next_token_embeddings)
        normalized_hidden = self.pre_fc_norm_hidden(shifted_hidden)
        mtp_hidden = self.fc(torch.cat((normalized_embedding, normalized_hidden), dim=-1))

        text_positions, rope_positions = self._shift_position_ids(
            position_ids,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=hidden_states.device,
        )
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=mtp_hidden,
            attention_mask=pair_mask,
            past_key_values=None,
            position_ids=text_positions,
        )
        position_embeddings = self.rotary_emb(mtp_hidden, rope_positions)
        for layer in self.layers:
            mtp_hidden = layer(
                mtp_hidden,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=text_positions,
                past_key_values=None,
                use_cache=False,
            )
        return self.norm(mtp_hidden)


__all__ = ["Qwen35MTP"]
