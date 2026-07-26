"""Losses used by dense transfer and sparse routing stages."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import Any

DEFAULT_LOSS_CHUNK_TOKENS = 128


@dataclass(frozen=True, slots=True)
class StreamingLanguageModelLosses:
    """Vocabulary losses produced without a sequence-wide logits tensor."""

    ntp: Any
    teacher_kd: Any
    anchor_kl: Any | None


def _validated_chunk_tokens(chunk_tokens: int | None) -> int | None:
    if chunk_tokens is None:
        return None
    if isinstance(chunk_tokens, bool) or not isinstance(chunk_tokens, int) or chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be None or a positive integer")
    return chunk_tokens


def _validated_checkpoint_chunks(checkpoint_chunks: bool) -> bool:
    if not isinstance(checkpoint_chunks, bool):
        raise ValueError("checkpoint_chunks must be a boolean")
    return checkpoint_chunks


def _validated_compile_loss(compile_loss: bool) -> bool:
    if not isinstance(compile_loss, bool):
        raise ValueError("compile_loss must be a boolean")
    return compile_loss


def _run_loss_chunk(function: Any, *args: Any, checkpoint_chunks: bool) -> Any:
    """Run a chunk while retaining only its inputs for backward when useful."""

    import torch

    should_checkpoint = (
        checkpoint_chunks
        and torch.is_grad_enabled()
        and any(isinstance(value, torch.Tensor) and value.requires_grad for value in args)
    )
    if not should_checkpoint:
        return function(*args)

    from torch.utils.checkpoint import checkpoint

    # Loss chunks contain no randomness, so preserving RNG state would add
    # synchronization and bookkeeping without changing the result.
    return checkpoint(
        function,
        *args,
        use_reentrant=False,
        preserve_rng_state=False,
    )


def _differentiable_zero(value: Any) -> Any:
    """Return an FP32 scalar zero that remains connected to ``value``."""

    # Selecting one vocabulary entry per token avoids materializing a full
    # FP32 logits tensor.  The empty-tensor case is also differentiable.
    return value[..., :1].float().sum() * 0.0


def _chunk_ranges(token_count: int, chunk_tokens: int | None) -> Any:
    if token_count == 0:
        return
    step = token_count if chunk_tokens is None else chunk_tokens
    for start in range(0, token_count, step):
        yield start, min(start + step, token_count)


def _bucketed_topk_kl_per_token(
    student_logits: Any,
    teacher_indices: Any,
    teacher_topk_logits: Any,
    teacher_logsumexp: Any,
    teacher_tail_logprob: Any,
    *,
    temperature: float,
) -> Any:
    """Return the unreduced bucketed teacher-to-student KL for each token."""

    return _bucketed_topk_kl_per_token_from_float(
        student_logits.float(),
        teacher_indices,
        teacher_topk_logits,
        teacher_logsumexp,
        teacher_tail_logprob,
        temperature=temperature,
    )


def _bucketed_topk_kl_per_token_from_float(
    student_logits: Any,
    teacher_indices: Any,
    teacher_topk_logits: Any,
    teacher_logsumexp: Any,
    teacher_tail_logprob: Any,
    *,
    temperature: float,
) -> Any:
    """Bucketed KL from an already-FP32 student tensor.

    The streaming CE/KD/anchor path converts each projected vocabulary chunk to
    FP32 once, then uses separate T=1 and T=``temperature`` normalizers.  The
    public standalone KL path retains the same behavior through the wrapper
    above.
    """

    import torch

    scaled_student = student_logits / temperature
    student_log_z = torch.logsumexp(scaled_student, dim=-1, keepdim=True)
    selected_student = torch.gather(
        scaled_student,
        -1,
        teacher_indices.long(),
    )
    student_top_logprob = selected_student - student_log_z
    student_top_prob = student_top_logprob.exp()
    student_tail_prob = (1.0 - student_top_prob.sum(dim=-1)).clamp_min(1e-12)
    student_tail_logprob = student_tail_prob.log()

    teacher_top_logprob = (
        teacher_topk_logits.float() / temperature - teacher_logsumexp.float().unsqueeze(-1)
    )
    teacher_top_prob = teacher_top_logprob.exp()
    teacher_tail_logprob = teacher_tail_logprob.float()
    teacher_tail_prob = teacher_tail_logprob.exp()

    kl_top = (teacher_top_prob * (teacher_top_logprob - student_top_logprob)).sum(dim=-1)
    kl_tail = teacher_tail_prob * (teacher_tail_logprob - student_tail_logprob)
    return (kl_top + kl_tail) * (temperature**2)


def _streaming_student_sums_from_logits(
    student_logits: Any,
    labels: Any,
    teacher_indices: Any,
    teacher_topk_logits: Any,
    teacher_logsumexp: Any,
    teacher_tail_logprob: Any,
    weight: Any,
    *,
    temperature: float,
    reference_logits: Any | None = None,
) -> tuple[Any, Any, Any | None]:
    """Reduce one projected chunk without materializing full log-probabilities.

    NTP and anchor KL use the same student T=1 log-normalizer.  Teacher KD keeps
    its independent T=``temperature`` normalizer.  For anchor KL, the identity

    ``KL(p_ref || p_student) = logZ_student - logZ_ref
    + E_ref[reference_logits - student_logits]``

    avoids constructing either full-vocabulary log-probability tensor while
    remaining exactly equivalent to ``F.kl_div(log_softmax(student),
    softmax(reference))``.  A reference probability is still transiently
    required for the expectation and for the exact student gradient.
    """

    import torch

    student_float = student_logits.float()
    student_log_z = torch.logsumexp(student_float, dim=-1)
    active = weight.to(dtype=torch.bool)
    safe_labels = labels.masked_fill(~active, 0)
    target_logits = torch.gather(
        student_float,
        -1,
        safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    float_weight = active.to(dtype=student_float.dtype)
    ntp_sum = ((student_log_z - target_logits) * float_weight).sum()

    kd_per_token = _bucketed_topk_kl_per_token_from_float(
        student_float,
        teacher_indices,
        teacher_topk_logits,
        teacher_logsumexp,
        teacher_tail_logprob,
        temperature=temperature,
    )
    kd_sum = (kd_per_token * float_weight).sum()

    if reference_logits is None:
        return ntp_sum, kd_sum, None

    reference_float = reference_logits.float()
    reference_log_z = torch.logsumexp(reference_float, dim=-1)
    reference_probability = (
        reference_float - reference_log_z.unsqueeze(-1)
    ).exp()
    anchor_per_token = (
        student_log_z
        - reference_log_z
        + (reference_probability * (reference_float - student_float)).sum(dim=-1)
    )
    anchor_sum = (anchor_per_token * float_weight).sum()
    return ntp_sum, kd_sum, anchor_sum


def _streaming_ntp_sum_from_logits(
    student_logits: Any,
    labels: Any,
    weight: Any,
) -> Any:
    """Reduce one pure next-token chunk without any teacher-side inputs."""

    import torch

    student_float = student_logits.float()
    active = weight.to(dtype=torch.bool)
    safe_labels = labels.masked_fill(~active, 0)
    target_logits = torch.gather(
        student_float,
        -1,
        safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    log_z = torch.logsumexp(student_float, dim=-1)
    return ((log_z - target_logits) * active.to(dtype=student_float.dtype)).sum()


@cache
def _compiled_streaming_ntp_sum() -> Any:
    """Lazily compile the static CUDA pure-NTP reduction."""

    import torch

    return torch.compile(
        _streaming_ntp_sum_from_logits,
        fullgraph=True,
        dynamic=False,
    )


def _streaming_student_sums_no_anchor(
    student_logits: Any,
    labels: Any,
    teacher_indices: Any,
    teacher_topk_logits: Any,
    teacher_logsumexp: Any,
    teacher_tail_logprob: Any,
    weight: Any,
    temperature: float,
) -> tuple[Any, Any]:
    """Static no-anchor specialization around the eager reduction oracle."""

    ntp_sum, kd_sum, anchor_sum = _streaming_student_sums_from_logits(
        student_logits,
        labels,
        teacher_indices,
        teacher_topk_logits,
        teacher_logsumexp,
        teacher_tail_logprob,
        weight,
        temperature=temperature,
    )
    assert anchor_sum is None
    return ntp_sum, kd_sum


def _streaming_student_sums_with_anchor(
    student_logits: Any,
    labels: Any,
    teacher_indices: Any,
    teacher_topk_logits: Any,
    teacher_logsumexp: Any,
    teacher_tail_logprob: Any,
    weight: Any,
    reference_logits: Any,
    temperature: float,
) -> tuple[Any, Any, Any]:
    """Static anchor specialization around the eager reduction oracle."""

    ntp_sum, kd_sum, anchor_sum = _streaming_student_sums_from_logits(
        student_logits,
        labels,
        teacher_indices,
        teacher_topk_logits,
        teacher_logsumexp,
        teacher_tail_logprob,
        weight,
        temperature=temperature,
        reference_logits=reference_logits,
    )
    assert anchor_sum is not None
    return ntp_sum, kd_sum, anchor_sum


@cache
def _compiled_streaming_student_sums_no_anchor() -> Any:
    """Lazily compile and cache the static CUDA no-anchor reduction."""

    import torch

    return torch.compile(
        _streaming_student_sums_no_anchor,
        fullgraph=True,
        dynamic=False,
    )


@cache
def _compiled_streaming_student_sums_with_anchor() -> Any:
    """Lazily compile and cache the static CUDA anchor reduction."""

    import torch

    return torch.compile(
        _streaming_student_sums_with_anchor,
        fullgraph=True,
        dynamic=False,
    )


def _full_kl_per_token(student_logits: Any, reference_logits: Any) -> Any:
    """Return unreduced reference-to-student full-vocabulary KL per token."""

    import torch.nn.functional as F

    student_logprob = F.log_softmax(student_logits.float(), dim=-1)
    reference_prob = F.softmax(reference_logits.float(), dim=-1)
    return F.kl_div(
        student_logprob,
        reference_prob,
        reduction="none",
    ).sum(dim=-1)


def causal_language_model_loss(
    logits: Any,
    labels: Any,
    *,
    ignore_index: int = -100,
    chunk_tokens: int | None = DEFAULT_LOSS_CHUNK_TOKENS,
    checkpoint_chunks: bool = True,
) -> Any:
    """Next-token cross entropy without a sequence-wide FP32 logits copy.

    ``chunk_tokens`` limits the number of vocabulary rows converted to FP32
    at once.  ``None`` processes all shifted tokens in one chunk.  By default,
    each chunk is activation-checkpointed so its FP32 softmax state is
    recomputed, rather than retained until backward.  When every label is
    ignored (or the sequence has no next-token targets), the result is a
    differentiable zero rather than ``nan``.
    """

    import torch
    import torch.nn.functional as F

    chunk_tokens = _validated_chunk_tokens(chunk_tokens)
    checkpoint_chunks = _validated_checkpoint_chunks(checkpoint_chunks)
    if logits.ndim < 2:
        raise ValueError("logits must have [..., sequence, vocabulary] shape")
    if tuple(labels.shape) != tuple(logits.shape[:-1]):
        raise ValueError("labels shape must match logits without the vocabulary dimension")

    sequence_length = logits.shape[-2]
    if sequence_length <= 1:
        return _differentiable_zero(logits)

    vocabulary_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocabulary_size)
    shifted_labels = labels.new_full(labels.shape, ignore_index)
    shifted_labels[..., :-1] = labels[..., 1:]
    flat_labels = shifted_labels.reshape(-1)
    token_count = flat_labels.numel()
    total_loss = _differentiable_zero(logits)
    valid_count = torch.zeros((), dtype=torch.long, device=labels.device)

    def chunk_cross_entropy(chunk_logits: Any, chunk_labels: Any) -> Any:
        return F.cross_entropy(
            chunk_logits.float(),
            chunk_labels,
            ignore_index=ignore_index,
            reduction="sum",
        )

    # Mark each sequence's unused final logit as ignored.  This keeps logits
    # flattenable as contiguous views and avoids gathering/copying every
    # shifted chunk merely to skip one row per sequence.
    for start, end in _chunk_ranges(token_count, chunk_tokens):
        chunk_labels = flat_labels[start:end]
        chunk_logits = flat_logits[start:end]
        total_loss = total_loss + _run_loss_chunk(
            chunk_cross_entropy,
            chunk_logits,
            chunk_labels,
            checkpoint_chunks=checkpoint_chunks,
        )
        valid_count = valid_count + chunk_labels.ne(ignore_index).sum()

    return total_loss / valid_count.clamp_min(1).to(total_loss.dtype)


def bucketed_topk_kl(
    student_logits: Any,
    teacher_indices: Any,
    teacher_topk_logits: Any,
    teacher_logsumexp: Any,
    teacher_tail_logprob: Any,
    *,
    temperature: float,
    mask: Any | None = None,
    chunk_tokens: int | None = DEFAULT_LOSS_CHUNK_TOKENS,
    checkpoint_chunks: bool = True,
) -> Any:
    """KL over teacher top-k categories plus one aggregate tail bucket.

    The cached ``teacher_logsumexp`` and ``teacher_tail_logprob`` must have
    been computed at the same temperature.  Treating the unobserved vocabulary
    as one bucket makes the approximation normalized and cheap without
    inventing a uniform tail distribution.  With ``checkpoint_chunks=True``,
    FP32 intermediates are recomputed one chunk at a time during backward.
    """

    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be positive")
    chunk_tokens = _validated_chunk_tokens(chunk_tokens)
    checkpoint_chunks = _validated_checkpoint_chunks(checkpoint_chunks)
    if student_logits.ndim < 1:
        raise ValueError("student_logits must have [..., vocabulary] shape")
    token_shape = tuple(student_logits.shape[:-1])
    if tuple(teacher_indices.shape[:-1]) != token_shape:
        raise ValueError("teacher_indices token dimensions must match student_logits")
    if tuple(teacher_topk_logits.shape) != tuple(teacher_indices.shape):
        raise ValueError("teacher_topk_logits shape must match teacher_indices")
    if tuple(teacher_logsumexp.shape) != token_shape:
        raise ValueError("teacher_logsumexp shape must match student token dimensions")
    if tuple(teacher_tail_logprob.shape) != token_shape:
        raise ValueError("teacher_tail_logprob shape must match student token dimensions")
    if mask is not None and tuple(mask.shape) != token_shape:
        raise ValueError("mask shape must match student token dimensions")

    token_count = student_logits.numel() // student_logits.shape[-1]
    if token_count == 0:
        return _differentiable_zero(student_logits)

    vocabulary_size = student_logits.shape[-1]
    topk = teacher_indices.shape[-1]
    flat_student = student_logits.reshape(-1, vocabulary_size)
    flat_indices = teacher_indices.reshape(-1, topk)
    flat_topk_logits = teacher_topk_logits.reshape(-1, topk)
    flat_teacher_logsumexp = teacher_logsumexp.reshape(-1)
    flat_teacher_tail = teacher_tail_logprob.reshape(-1)
    flat_mask = None if mask is None else mask.reshape(-1)
    total_loss = _differentiable_zero(student_logits)
    total_weight = total_loss.new_zeros(())

    def chunk_bucketed_kl(
        chunk_student: Any,
        chunk_indices: Any,
        chunk_topk_logits: Any,
        chunk_teacher_logsumexp: Any,
        chunk_teacher_tail: Any,
    ) -> Any:
        return _bucketed_topk_kl_per_token(
            chunk_student,
            chunk_indices,
            chunk_topk_logits,
            chunk_teacher_logsumexp,
            chunk_teacher_tail,
            temperature=temperature,
        )

    for start, end in _chunk_ranges(token_count, chunk_tokens):
        token_loss = _run_loss_chunk(
            chunk_bucketed_kl,
            flat_student[start:end],
            flat_indices[start:end],
            flat_topk_logits[start:end],
            flat_teacher_logsumexp[start:end],
            flat_teacher_tail[start:end],
            checkpoint_chunks=checkpoint_chunks,
        )
        if flat_mask is None:
            total_loss = total_loss + token_loss.sum()
            total_weight = total_weight + token_loss.new_tensor(end - start)
        else:
            weight = flat_mask[start:end].to(token_loss.dtype)
            total_loss = total_loss + (token_loss * weight).sum()
            total_weight = total_weight + weight.sum()

    return total_loss / total_weight.clamp_min(1.0)


def masked_full_kl(
    student_logits: Any,
    reference_logits: Any,
    mask: Any | None,
    *,
    chunk_tokens: int | None = DEFAULT_LOSS_CHUNK_TOKENS,
    checkpoint_chunks: bool = True,
) -> Any:
    """Reference-to-student full-vocabulary KL, averaged over masked tokens.

    The logits are converted to FP32 one token chunk at a time.  With
    ``checkpoint_chunks=True``, FP32 intermediates are recomputed one chunk at
    a time during backward.  ``None`` for ``mask`` gives an ordinary token
    mean; an all-zero mask gives a differentiable zero.
    """

    chunk_tokens = _validated_chunk_tokens(chunk_tokens)
    checkpoint_chunks = _validated_checkpoint_chunks(checkpoint_chunks)
    if tuple(reference_logits.shape) != tuple(student_logits.shape):
        raise ValueError("reference_logits shape must match student_logits")
    if student_logits.ndim < 1:
        raise ValueError("student_logits must have [..., vocabulary] shape")
    token_shape = tuple(student_logits.shape[:-1])
    if mask is not None and tuple(mask.shape) != token_shape:
        raise ValueError("mask shape must match student token dimensions")

    token_count = student_logits.numel() // student_logits.shape[-1]
    if token_count == 0:
        return _differentiable_zero(student_logits)

    vocabulary_size = student_logits.shape[-1]
    flat_student = student_logits.reshape(-1, vocabulary_size)
    flat_reference = reference_logits.reshape(-1, vocabulary_size)
    flat_mask = None if mask is None else mask.reshape(-1)
    total_loss = _differentiable_zero(student_logits)
    total_weight = total_loss.new_zeros(())

    def chunk_full_kl(chunk_student: Any, chunk_reference: Any) -> Any:
        return _full_kl_per_token(chunk_student, chunk_reference)

    for start, end in _chunk_ranges(token_count, chunk_tokens):
        token_loss = _run_loss_chunk(
            chunk_full_kl,
            flat_student[start:end],
            flat_reference[start:end],
            checkpoint_chunks=checkpoint_chunks,
        )
        if flat_mask is None:
            total_loss = total_loss + token_loss.sum()
            total_weight = total_weight + token_loss.new_tensor(end - start)
        else:
            weight = flat_mask[start:end].to(token_loss.dtype)
            total_loss = total_loss + (token_loss * weight).sum()
            total_weight = total_weight + weight.sum()

    return total_loss / total_weight.clamp_min(1.0)


def streaming_next_token_loss(
    final_hidden_states: Any,
    lm_head: Any,
    labels: Any,
    *,
    mask: Any | None = None,
    ignore_index: int = -100,
    chunk_tokens: int | None = DEFAULT_LOSS_CHUNK_TOKENS,
    checkpoint_chunks: bool = True,
    compile_loss: bool = False,
) -> Any:
    """Project and reduce pure-text NTP without constructing KD intermediates.

    The target contract is identical to :func:`streaming_language_model_losses`:
    the hidden state at position ``t`` predicts ``labels[t + 1]`` and the
    optional mask applies to the predicting position. Each vocabulary chunk is
    released immediately after its scalar cross-entropy sum is produced.
    """

    import torch

    chunk_tokens = _validated_chunk_tokens(chunk_tokens)
    checkpoint_chunks = _validated_checkpoint_chunks(checkpoint_chunks)
    compile_loss = _validated_compile_loss(compile_loss)
    if final_hidden_states.ndim < 2:
        raise ValueError("final_hidden_states must have [..., sequence, hidden] shape")
    token_shape = tuple(final_hidden_states.shape[:-1])
    if tuple(labels.shape) != token_shape:
        raise ValueError("labels shape must match final_hidden_states token dimensions")
    if mask is not None and tuple(mask.shape) != token_shape:
        raise ValueError("mask shape must match final_hidden_states token dimensions")

    hidden_size = final_hidden_states.shape[-1]
    token_count = labels.numel()
    zero = _differentiable_zero(final_hidden_states)
    if token_count == 0:
        return zero

    shifted_labels = labels.new_full(labels.shape, ignore_index)
    if labels.shape[-1] > 1:
        shifted_labels[..., :-1] = labels[..., 1:]
    target_weight = shifted_labels.ne(ignore_index)
    if mask is not None:
        target_weight = target_weight & mask.ne(0)
        shifted_labels = shifted_labels.masked_fill(~target_weight, ignore_index)

    flat_hidden = final_hidden_states.reshape(-1, hidden_size)
    flat_labels = shifted_labels.reshape(-1)
    flat_weight = target_weight.reshape(-1)
    valid_count = flat_weight.sum()

    def chunk_sum(
        chunk_hidden: Any,
        chunk_labels: Any,
        chunk_weight: Any,
    ) -> Any:
        logits = lm_head(chunk_hidden)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("lm_head must return a Tensor")
        if logits.ndim != 2 or logits.shape[0] != chunk_hidden.shape[0]:
            raise ValueError("lm_head must map [tokens, hidden] to [tokens, vocabulary]")
        reducer = (
            _compiled_streaming_ntp_sum()
            if compile_loss and logits.device.type == "cuda"
            else _streaming_ntp_sum_from_logits
        )
        return reducer(logits, chunk_labels, chunk_weight)

    total = zero
    for start, end in _chunk_ranges(token_count, chunk_tokens):
        total = total + _run_loss_chunk(
            chunk_sum,
            flat_hidden[start:end],
            flat_labels[start:end],
            flat_weight[start:end],
            checkpoint_chunks=checkpoint_chunks,
        )
    return total / valid_count.clamp_min(1).to(dtype=total.dtype)


def streaming_language_model_losses(
    final_hidden_states: Any,
    lm_head: Any,
    labels: Any,
    teacher_indices: Any,
    teacher_topk_logits: Any,
    teacher_logsumexp: Any,
    teacher_tail_logprob: Any,
    *,
    temperature: float,
    mask: Any | None = None,
    anchor_hidden_states: Any | None = None,
    anchor_lm_head: Any | None = None,
    ignore_index: int = -100,
    chunk_tokens: int | None = DEFAULT_LOSS_CHUNK_TOKENS,
    checkpoint_chunks: bool = True,
    compile_loss: bool = False,
) -> StreamingLanguageModelLosses:
    """Project final hidden states and reduce all vocabulary losses by token chunk.

    Unlike :func:`causal_language_model_loss`, :func:`bucketed_topk_kl`, and
    :func:`masked_full_kl`, this entry point never accepts or creates a
    sequence-wide ``[..., sequence, vocabulary]`` tensor.  Each flattened
    token chunk is projected through ``lm_head`` exactly once, its NTP and
    bucketed KD losses are immediately reduced to scalars, and the logits are
    then released.  Supplying ``anchor_hidden_states`` additionally projects a
    detached reference chunk under ``no_grad`` and returns anchor KL.
    NTP, teacher KD, and anchor KL share the same shifted next-token target
    mask: padding and each sequence's unused final logit contribute to none of
    the three objectives.

    ``lm_head`` and ``anchor_lm_head`` are called as modules rather than by
    reading their weights.  This preserves autocast and module forward hooks.
    For composable FSDP, invoke this function *inside* the outer managed
    model's forward while its parameters are unsharded.  This is required when
    the head weight is tied to the input embedding; it must not be placed in a
    separate FSDP unit from its alias.  An untied head may instead be its own
    FSDP unit kept unsharded across the chunk loop.  Likewise, calling a
    trainable head outside a DDP-wrapped forward does not arrange DDP gradient
    reduction.  Twen's frozen head is safe outside DDP, while a future
    trainable head must execute inside the wrapped forward.

    With ``checkpoint_chunks=True`` (the default), the head and loss math are
    recomputed during backward so full-vocabulary chunk intermediates do not
    accumulate across the sequence.  PyTorch non-reentrant checkpointing
    restores the forward autocast context for that recomputation.  Setting
    ``compile_loss=True`` selects lazily cached, full-graph static CUDA
    specializations for the per-chunk reductions. CPU tensors always retain the
    eager oracle path.
    """

    import torch
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be positive")
    chunk_tokens = _validated_chunk_tokens(chunk_tokens)
    checkpoint_chunks = _validated_checkpoint_chunks(checkpoint_chunks)
    compile_loss = _validated_compile_loss(compile_loss)
    if final_hidden_states.ndim < 2:
        raise ValueError("final_hidden_states must have [..., sequence, hidden] shape")
    token_shape = tuple(final_hidden_states.shape[:-1])
    if tuple(labels.shape) != token_shape:
        raise ValueError("labels shape must match final_hidden_states token dimensions")
    if teacher_indices.ndim < 1 or tuple(teacher_indices.shape[:-1]) != token_shape:
        raise ValueError("teacher_indices token dimensions must match final_hidden_states")
    if tuple(teacher_topk_logits.shape) != tuple(teacher_indices.shape):
        raise ValueError("teacher_topk_logits shape must match teacher_indices")
    if tuple(teacher_logsumexp.shape) != token_shape:
        raise ValueError("teacher_logsumexp shape must match final_hidden_states token dimensions")
    if tuple(teacher_tail_logprob.shape) != token_shape:
        raise ValueError(
            "teacher_tail_logprob shape must match final_hidden_states token dimensions"
        )
    if mask is not None and tuple(mask.shape) != token_shape:
        raise ValueError("mask shape must match final_hidden_states token dimensions")
    if anchor_hidden_states is None:
        if anchor_lm_head is not None:
            raise ValueError("anchor_lm_head requires anchor_hidden_states")
    elif tuple(anchor_hidden_states.shape[:-1]) != token_shape:
        raise ValueError("anchor_hidden_states token dimensions must match final_hidden_states")

    hidden_size = final_hidden_states.shape[-1]
    token_count = labels.numel()
    zero = _differentiable_zero(final_hidden_states)
    if token_count == 0:
        return StreamingLanguageModelLosses(
            ntp=zero,
            teacher_kd=zero,
            anchor_kl=zero if anchor_hidden_states is not None else None,
        )

    topk = teacher_indices.shape[-1]
    flat_hidden = final_hidden_states.reshape(-1, hidden_size)
    flat_indices = teacher_indices.reshape(-1, topk)
    flat_topk_logits = teacher_topk_logits.reshape(-1, topk)
    flat_teacher_logsumexp = teacher_logsumexp.reshape(-1)
    flat_teacher_tail = teacher_tail_logprob.reshape(-1)
    shifted_labels = labels.new_full(labels.shape, ignore_index)
    if labels.shape[-1] > 1:
        shifted_labels[..., :-1] = labels[..., 1:]
    target_weight = shifted_labels.ne(ignore_index)
    if mask is not None:
        target_weight = target_weight & mask.ne(0)
        shifted_labels = shifted_labels.masked_fill(~target_weight, ignore_index)
    flat_labels = shifted_labels.reshape(-1)
    flat_weight = target_weight.reshape(-1)
    valid_count = flat_weight.sum()
    total_weight = flat_weight.float().sum()

    flat_anchor = None
    reference_head = anchor_lm_head
    if anchor_hidden_states is not None:
        flat_anchor = anchor_hidden_states.detach().reshape(-1, anchor_hidden_states.shape[-1])
        if reference_head is None:
            reference_head = lm_head

    def project_logits(head: Any, chunk_hidden: Any, *, name: str) -> Any:
        logits = head(chunk_hidden)
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"{name} must return a Tensor")
        if logits.ndim != 2 or logits.shape[0] != chunk_hidden.shape[0]:
            raise ValueError(f"{name} must map [tokens, hidden] to [tokens, vocabulary]")
        return logits

    def student_sums(
        chunk_hidden: Any,
        chunk_labels: Any,
        chunk_indices: Any,
        chunk_topk_logits: Any,
        chunk_teacher_logsumexp: Any,
        chunk_teacher_tail: Any,
        chunk_weight: Any,
    ) -> tuple[Any, Any]:
        chunk_logits = project_logits(lm_head, chunk_hidden, name="lm_head")
        reducer = (
            _compiled_streaming_student_sums_no_anchor()
            if compile_loss and chunk_logits.device.type == "cuda"
            else _streaming_student_sums_no_anchor
        )
        return reducer(
            chunk_logits,
            chunk_labels,
            chunk_indices,
            chunk_topk_logits,
            chunk_teacher_logsumexp,
            chunk_teacher_tail,
            chunk_weight,
            temperature,
        )

    def student_anchor_sums(
        chunk_hidden: Any,
        chunk_anchor: Any,
        chunk_labels: Any,
        chunk_indices: Any,
        chunk_topk_logits: Any,
        chunk_teacher_logsumexp: Any,
        chunk_teacher_tail: Any,
        chunk_weight: Any,
    ) -> tuple[Any, Any, Any]:
        chunk_logits = project_logits(lm_head, chunk_hidden, name="lm_head")
        assert reference_head is not None
        with torch.no_grad():
            chunk_reference_logits = project_logits(
                reference_head,
                chunk_anchor,
                name="anchor_lm_head",
            )
        if tuple(chunk_reference_logits.shape) != tuple(chunk_logits.shape):
            raise ValueError("anchor_lm_head vocabulary size must match lm_head")
        reducer = (
            _compiled_streaming_student_sums_with_anchor()
            if compile_loss and chunk_logits.device.type == "cuda"
            else _streaming_student_sums_with_anchor
        )
        return reducer(
            chunk_logits,
            chunk_labels,
            chunk_indices,
            chunk_topk_logits,
            chunk_teacher_logsumexp,
            chunk_teacher_tail,
            chunk_weight,
            chunk_reference_logits,
            temperature,
        )

    ntp_total = zero
    kd_total = zero
    anchor_total = zero if flat_anchor is not None else None
    for start, end in _chunk_ranges(token_count, chunk_tokens):
        common_args = (
            flat_labels[start:end],
            flat_indices[start:end],
            flat_topk_logits[start:end],
            flat_teacher_logsumexp[start:end],
            flat_teacher_tail[start:end],
            flat_weight[start:end],
        )
        if flat_anchor is None:
            chunk_ntp, chunk_kd = _run_loss_chunk(
                student_sums,
                flat_hidden[start:end],
                *common_args,
                checkpoint_chunks=checkpoint_chunks,
            )
        else:
            chunk_ntp, chunk_kd, chunk_anchor = _run_loss_chunk(
                student_anchor_sums,
                flat_hidden[start:end],
                flat_anchor[start:end],
                *common_args,
                checkpoint_chunks=checkpoint_chunks,
            )
            assert anchor_total is not None
            anchor_total = anchor_total + chunk_anchor
        ntp_total = ntp_total + chunk_ntp
        kd_total = kd_total + chunk_kd

    ntp = ntp_total / valid_count.clamp_min(1).to(ntp_total.dtype)
    teacher_kd = kd_total / total_weight.clamp_min(1.0).to(kd_total.dtype)
    anchor_kl = (
        None
        if anchor_total is None
        else anchor_total / total_weight.clamp_min(1.0).to(anchor_total.dtype)
    )
    return StreamingLanguageModelLosses(
        ntp=ntp,
        teacher_kd=teacher_kd,
        anchor_kl=anchor_kl,
    )


def router_z_loss(router_logits: Any, mask: Any | None = None) -> Any:
    import torch

    value = torch.logsumexp(router_logits.float(), dim=-1).square()
    if mask is None:
        return value.mean()
    weight = mask.reshape(value.shape).to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def load_balancing_loss(
    router_logits: Any,
    selected_experts: Any,
    num_experts: int,
    mask: Any | None = None,
) -> Any:
    """Switch-style differentiable expert load balancing objective."""

    import torch
    import torch.nn.functional as F

    logits = router_logits.reshape(-1, num_experts).float()
    probabilities = logits.softmax(dim=-1)
    selected = selected_experts.reshape(-1, selected_experts.shape[-1])
    assignments = F.one_hot(selected.long(), num_classes=num_experts).float().mean(dim=1)
    if mask is None:
        token_fraction = assignments.mean(dim=0)
        probability_fraction = probabilities.mean(dim=0)
    else:
        weight = mask.reshape(-1).to(probabilities.dtype)
        denominator = weight.sum().clamp_min(1.0)
        token_fraction = (assignments * weight.unsqueeze(-1)).sum(dim=0) / denominator
        probability_fraction = (probabilities * weight.unsqueeze(-1)).sum(dim=0) / denominator
    return num_experts * torch.sum(token_fraction * probability_fraction)


def best_expert_pair(expert_outputs: Any, dense_output: Any | None = None) -> Any:
    """Return the pair whose scaled sum best reconstructs the dense sum.

    ``expert_outputs`` has shape ``[..., experts, hidden]``.  All pairs are
    enumerated; with eight experts this is only 28 candidates.  The selected
    partial sums are scaled by E/2, matching the expected sum when expert
    contributions are balanced.
    """

    import torch

    if expert_outputs.ndim < 3:
        raise ValueError("expert_outputs must have [..., experts, hidden] shape")
    experts = expert_outputs.shape[-2]
    if experts < 2:
        raise ValueError("at least two experts are required")
    target = expert_outputs.sum(dim=-2) if dense_output is None else dense_output
    pairs = torch.combinations(torch.arange(experts, device=expert_outputs.device), r=2)
    candidates = []
    for pair in pairs:
        candidates.append(expert_outputs[..., pair, :].sum(dim=-2) * (experts / 2.0))
    stacked = torch.stack(candidates, dim=-2)
    errors = (stacked.float() - target.float().unsqueeze(-2)).square().mean(dim=-1)
    best = errors.argmin(dim=-1)
    return pairs[best]


def router_pair_supervision_loss(
    router_logits: Any,
    target_pairs: Any,
    mask: Any | None = None,
) -> Any:
    """Multi-label negative log likelihood for an unordered oracle pair."""

    import torch

    log_prob = router_logits.float().log_softmax(dim=-1)
    selected = torch.gather(log_prob, -1, target_pairs.long())
    token_loss = -selected.mean(dim=-1)
    if mask is None:
        return token_loss.mean()
    weight = mask.reshape(token_loss.shape).to(token_loss.dtype)
    return (token_loss * weight).sum() / weight.sum().clamp_min(1.0)
