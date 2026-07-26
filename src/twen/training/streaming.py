"""Loss-aware causal-LM wrapper that never materializes full-sequence logits."""

from __future__ import annotations

from functools import cache
from typing import Any

import torch
from torch import nn

from .losses import streaming_language_model_losses


def native_mtp_target_mask(labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Return valid native Qwen3.5 ``h_t -> x_(t+2)`` training targets.

    The MTP input at position ``t`` consumes both the main-model state for
    ``x_t`` and the embedding of known token ``x_(t+1)``.  Its target is
    ``x_(t+2)``, so all three attention-mask positions and the final label must
    be valid.  The result has sequence length ``L-2``.
    """

    if labels.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("MTP labels and attention_mask must have shape [batch, sequence]")
    if tuple(labels.shape) != tuple(attention_mask.shape):
        raise ValueError("MTP labels and attention_mask must have identical shapes")
    return (
        labels[:, 2:].ne(-100)
        & attention_mask[:, :-2].ne(0)
        & attention_mask[:, 1:-1].ne(0)
        & attention_mask[:, 2:].ne(0)
    )


def _mtp_cross_entropy_sum_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Pure FP32 MTP CE reduction used by both eager and compiled paths."""

    logits_float = logits.float()
    active = targets.ne(-100)
    safe_targets = targets.masked_fill(~active, 0)
    target_logits = torch.gather(
        logits_float,
        -1,
        safe_targets.unsqueeze(-1),
    ).squeeze(-1)
    log_z = torch.logsumexp(logits_float, dim=-1)
    return ((log_z - target_logits) * active.to(dtype=logits_float.dtype)).sum()


@cache
def _compiled_mtp_cross_entropy_sum() -> Any:
    """Lazily compile the static CUDA MTP reduction without compiling its head."""

    return torch.compile(
        _mtp_cross_entropy_sum_from_logits,
        fullgraph=True,
        dynamic=False,
    )


def _streaming_mtp_cross_entropy(
    hidden_states: torch.Tensor,
    lm_head: nn.Module,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    chunk_tokens: int | None,
    checkpoint_chunks: bool,
    compile_loss: bool,
) -> torch.Tensor:
    """Reduce aligned MTP CE without retaining sequence-wide vocabulary logits."""

    if hidden_states.ndim != 3:
        raise ValueError("MTP hidden_states must have shape [batch, sequence, hidden]")
    if tuple(targets.shape) != tuple(hidden_states.shape[:-1]):
        raise ValueError("MTP targets must match hidden-state token dimensions")
    if tuple(target_mask.shape) != tuple(targets.shape):
        raise ValueError("MTP target_mask must match targets")
    if chunk_tokens is not None and (
        isinstance(chunk_tokens, bool)
        or not isinstance(chunk_tokens, int)
        or chunk_tokens <= 0
    ):
        raise ValueError("chunk_tokens must be None or a positive integer")
    if not isinstance(checkpoint_chunks, bool):
        raise ValueError("checkpoint_chunks must be a boolean")
    if not isinstance(compile_loss, bool):
        raise ValueError("compile_loss must be a boolean")

    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_targets = targets.masked_fill(~target_mask, -100).reshape(-1)
    valid_count = target_mask.sum()
    total = flat_hidden[..., :1].float().sum() * 0.0
    token_count = flat_targets.numel()
    if token_count == 0:
        return total
    step = token_count if chunk_tokens is None else chunk_tokens

    def chunk_sum(chunk_hidden: torch.Tensor, chunk_targets: torch.Tensor) -> torch.Tensor:
        logits = lm_head(chunk_hidden)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("lm_head must return a Tensor")
        if logits.ndim != 2 or logits.shape[0] != chunk_hidden.shape[0]:
            raise ValueError("lm_head must map [tokens, hidden] to [tokens, vocabulary]")
        reducer = (
            _compiled_mtp_cross_entropy_sum()
            if compile_loss and logits.device.type == "cuda"
            else _mtp_cross_entropy_sum_from_logits
        )
        return reducer(logits, chunk_targets)

    for start in range(0, token_count, step):
        end = min(start + step, token_count)
        chunk_hidden = flat_hidden[start:end]
        chunk_targets = flat_targets[start:end]
        should_checkpoint = (
            checkpoint_chunks
            and torch.is_grad_enabled()
            and chunk_hidden.requires_grad
        )
        if should_checkpoint:
            from torch.utils.checkpoint import checkpoint

            chunk_loss = checkpoint(
                chunk_sum,
                chunk_hidden,
                chunk_targets,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            chunk_loss = chunk_sum(chunk_hidden, chunk_targets)
        total = total + chunk_loss
    return total / valid_count.clamp_min(1).to(dtype=total.dtype)


class StreamingLossCausalLM(nn.Module):
    """Run the text body and vocabulary losses inside one distributed forward.

    Keeping the LM head call inside this wrapper is important for composable
    FSDP when the output head is tied to the input embedding.  Both aliases
    remain owned by the same outer FSDP unit while vocabulary-sized tensors are
    created only for one token chunk at a time.
    """

    def __init__(
        self,
        causal_lm: nn.Module,
        *,
        chunk_tokens: int | None,
        checkpoint_chunks: bool,
        compile_loss: bool = False,
        mtp: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(getattr(causal_lm, "model", None), nn.Module):
            raise TypeError("causal_lm.model must be a text-body Module")
        if not isinstance(getattr(causal_lm, "lm_head", None), nn.Module):
            raise TypeError("causal_lm.lm_head must be a Module")
        if not isinstance(compile_loss, bool):
            raise ValueError("compile_loss must be a boolean")
        if mtp is not None and not isinstance(mtp, nn.Module):
            raise TypeError("mtp must be a torch.nn.Module or None")
        if mtp is not None and any(parameter.requires_grad for parameter in mtp.parameters()):
            raise ValueError("native MTP source parameters must remain frozen")
        self.causal_lm = causal_lm
        self.mtp = mtp
        self.chunk_tokens = chunk_tokens
        self.checkpoint_chunks = checkpoint_chunks
        self.compile_loss = compile_loss
        if self.mtp is not None:
            self.mtp.eval()

    def train(self, mode: bool = True) -> StreamingLossCausalLM:
        """Keep the frozen source MTP deterministic when the student enters train mode."""

        super().train(mode)
        if self.mtp is not None:
            self.mtp.eval()
        return self

    def forward(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        labels: Any | None = None,
        teacher_indices: Any | None = None,
        teacher_topk_logits: Any | None = None,
        teacher_logsumexp: Any | None = None,
        teacher_tail_logprob: Any | None = None,
        temperature: float = 1.0,
        anchor_hidden_states: Any | None = None,
        output_hidden_states: bool = False,
        anchor_only: bool = False,
    ) -> dict[str, Any]:
        body_outputs = self.causal_lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        final_hidden_states = body_outputs.last_hidden_state
        if not isinstance(final_hidden_states, torch.Tensor):
            raise TypeError("causal_lm.model must return a Tensor last_hidden_state")
        if anchor_only:
            return {"anchor_hidden_states": final_hidden_states.detach()}

        required = {
            "labels": labels,
            "teacher_indices": teacher_indices,
            "teacher_topk_logits": teacher_topk_logits,
            "teacher_logsumexp": teacher_logsumexp,
            "teacher_tail_logprob": teacher_tail_logprob,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"student loss forward is missing: {', '.join(missing)}")

        losses = streaming_language_model_losses(
            final_hidden_states,
            self.causal_lm.lm_head,
            labels,
            teacher_indices,
            teacher_topk_logits,
            teacher_logsumexp,
            teacher_tail_logprob,
            temperature=temperature,
            mask=attention_mask,
            anchor_hidden_states=anchor_hidden_states,
            chunk_tokens=self.chunk_tokens,
            checkpoint_chunks=self.checkpoint_chunks,
            compile_loss=self.compile_loss,
        )
        mtp_loss = None
        if self.mtp is not None:
            if input_ids.shape[-1] < 3:
                mtp_loss = final_hidden_states[..., :1].float().sum() * 0.0
            else:
                embed_tokens = getattr(self.causal_lm.model, "embed_tokens", None)
                if not isinstance(embed_tokens, nn.Module):
                    raise TypeError("causal_lm.model.embed_tokens must be a Module for MTP")

                def mtp_forward(
                    hidden_states: torch.Tensor,
                    token_ids: torch.Tensor,
                    *,
                    shared_embedding: nn.Module,
                    mask: torch.Tensor,
                ) -> torch.Tensor:
                    # Keep the range inside the checkpointed callable so a
                    # backward recomputation is visible in profiler traces.
                    with torch.profiler.record_function("twen/mtp_forward"):
                        return self.mtp(
                            hidden_states,
                            token_ids,
                            embed_tokens=shared_embedding,
                            attention_mask=mask,
                        )

                mtp_requires_grad = final_hidden_states.requires_grad or any(
                    parameter.requires_grad for parameter in embed_tokens.parameters()
                )
                should_checkpoint_mtp = (
                    self.checkpoint_chunks
                    and self.training
                    and torch.is_grad_enabled()
                    and mtp_requires_grad
                )
                if should_checkpoint_mtp:
                    from torch.utils.checkpoint import checkpoint

                    mtp_hidden_states = checkpoint(
                        mtp_forward,
                        final_hidden_states,
                        input_ids,
                        shared_embedding=embed_tokens,
                        mask=attention_mask,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    mtp_hidden_states = mtp_forward(
                        final_hidden_states,
                        input_ids,
                        shared_embedding=embed_tokens,
                        mask=attention_mask,
                    )
                if not isinstance(mtp_hidden_states, torch.Tensor):
                    raise TypeError("mtp must return a Tensor")
                expected_shape = (*final_hidden_states.shape[:-2], input_ids.shape[-1] - 1)
                if tuple(mtp_hidden_states.shape[:-1]) != expected_shape:
                    raise ValueError("mtp must return [batch, sequence-1, hidden]")
                with torch.profiler.record_function("twen/mtp_vocab_loss"):
                    mtp_loss = _streaming_mtp_cross_entropy(
                        mtp_hidden_states[:, :-1],
                        self.causal_lm.lm_head,
                        labels[:, 2:],
                        native_mtp_target_mask(labels, attention_mask),
                        chunk_tokens=self.chunk_tokens,
                        checkpoint_chunks=self.checkpoint_chunks,
                        compile_loss=self.compile_loss,
                    )
        return {
            "ntp": losses.ntp,
            "mtp": mtp_loss,
            "teacher_kd": losses.teacher_kd,
            "anchor_kl": losses.anchor_kl,
            "hidden_states": body_outputs.hidden_states if output_hidden_states else None,
        }


__all__ = ["StreamingLossCausalLM", "native_mtp_target_mask"]
