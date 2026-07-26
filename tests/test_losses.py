from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import twen.training.losses as loss_module
from twen.training.losses import (
    best_expert_pair,
    bucketed_topk_kl,
    causal_language_model_loss,
    masked_full_kl,
    router_pair_supervision_loss,
    streaming_language_model_losses,
)


def _reference_bucketed_topk_kl(
    student_logits: torch.Tensor,
    teacher_indices: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    teacher_logsumexp: torch.Tensor,
    teacher_tail_logprob: torch.Tensor,
    *,
    temperature: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scaled_student = student_logits.float() / temperature
    student_log_z = torch.logsumexp(scaled_student, dim=-1, keepdim=True)
    selected_student = torch.gather(scaled_student, -1, teacher_indices.long())
    student_top_logprob = selected_student - student_log_z
    student_top_prob = student_top_logprob.exp()
    student_tail_logprob = (1.0 - student_top_prob.sum(dim=-1)).clamp_min(1e-12).log()
    teacher_top_logprob = (
        teacher_topk_logits.float() / temperature - teacher_logsumexp.float().unsqueeze(-1)
    )
    teacher_top_prob = teacher_top_logprob.exp()
    teacher_tail_logprob = teacher_tail_logprob.float()
    token_loss = (
        (teacher_top_prob * (teacher_top_logprob - student_top_logprob)).sum(dim=-1)
        + teacher_tail_logprob.exp() * (teacher_tail_logprob - student_tail_logprob)
    ) * temperature**2
    if mask is None:
        return token_loss.mean()
    weight = mask.to(token_loss.dtype)
    return (token_loss * weight).sum() / weight.sum().clamp_min(1.0)


def _reference_streaming_language_model_losses(
    hidden: torch.Tensor,
    head: torch.nn.Module,
    labels: torch.Tensor,
    teacher_indices: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    teacher_logsumexp: torch.Tensor,
    teacher_tail_logprob: torch.Tensor,
    *,
    temperature: float,
    mask: torch.Tensor | None,
    anchor_hidden: torch.Tensor | None,
    anchor_head: torch.nn.Module | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Original full-logit PyTorch formulas used as an independent oracle."""

    student_logits = head(hidden)
    shifted_labels = labels.new_full(labels.shape, -100)
    if labels.shape[-1] > 1:
        shifted_labels[..., :-1] = labels[..., 1:]
    target_mask = shifted_labels.ne(-100)
    if mask is not None:
        target_mask = target_mask & mask.ne(0)
        shifted_labels = shifted_labels.masked_fill(~target_mask, -100)
    denominator = target_mask.sum().clamp_min(1)

    ntp = F.cross_entropy(
        student_logits.float().reshape(-1, student_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    ) / denominator
    kd_per_token = _reference_bucketed_topk_kl(
        student_logits,
        teacher_indices,
        teacher_topk_logits,
        teacher_logsumexp,
        teacher_tail_logprob,
        temperature=temperature,
        mask=target_mask,
    )

    if anchor_hidden is None:
        return ntp, kd_per_token, None
    assert anchor_head is not None
    with torch.no_grad():
        reference_logits = anchor_head(anchor_hidden)
    anchor_per_token = F.kl_div(
        F.log_softmax(student_logits.float(), dim=-1),
        F.softmax(reference_logits.float(), dim=-1),
        reduction="none",
    ).sum(dim=-1)
    anchor_weight = target_mask.to(anchor_per_token.dtype)
    anchor = (anchor_per_token * anchor_weight).sum() / anchor_weight.sum().clamp_min(1.0)
    return ntp, kd_per_token, anchor


@pytest.mark.parametrize("checkpoint_chunks", [False, True])
def test_causal_language_model_loss_chunks_match_reference_and_gradients(
    checkpoint_chunks: bool,
) -> None:
    torch.manual_seed(11)
    labels = torch.tensor(
        [[0, 1, -100, 4, 3], [2, 0, 5, -100, 1]],
        dtype=torch.long,
    )
    expected_logits = torch.randn(2, 5, 7, requires_grad=True)
    actual_logits = expected_logits.detach().clone().requires_grad_(True)

    expected = F.cross_entropy(
        expected_logits[..., :-1, :].contiguous().float().view(-1, 7),
        labels[..., 1:].contiguous().view(-1),
        ignore_index=-100,
    )
    actual = causal_language_model_loss(
        actual_logits,
        labels,
        chunk_tokens=3,
        checkpoint_chunks=checkpoint_chunks,
    )
    expected.backward()
    actual.backward()

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        actual_logits.grad,
        expected_logits.grad,
        rtol=2e-6,
        atol=2e-6,
    )


def test_causal_language_model_loss_all_ignored_is_differentiable_zero() -> None:
    logits = torch.randn(2, 4, 5, requires_grad=True)
    labels = torch.full((2, 4), -7, dtype=torch.long)
    loss = causal_language_model_loss(
        logits,
        labels,
        ignore_index=-7,
        chunk_tokens=2,
    )
    loss.backward()
    assert loss.item() == 0.0
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize("chunk_tokens", [0, -1, True, 1.5])
def test_losses_reject_invalid_chunk_tokens(chunk_tokens: object) -> None:
    logits = torch.zeros((1, 2, 3))
    labels = torch.zeros((1, 2), dtype=torch.long)
    with pytest.raises(ValueError, match="chunk_tokens"):
        causal_language_model_loss(logits, labels, chunk_tokens=chunk_tokens)  # type: ignore[arg-type]


def test_losses_reject_non_boolean_checkpoint_chunks() -> None:
    logits = torch.zeros((1, 2, 3))
    labels = torch.zeros((1, 2), dtype=torch.long)
    with pytest.raises(ValueError, match="checkpoint_chunks"):
        causal_language_model_loss(
            logits,
            labels,
            checkpoint_chunks=1,  # type: ignore[arg-type]
        )


def test_streaming_loss_rejects_non_boolean_compile_loss() -> None:
    hidden = torch.zeros((1, 2, 3), requires_grad=True)
    head = torch.nn.Linear(3, 5, bias=False)
    labels = torch.zeros((1, 2), dtype=torch.long)
    indices = torch.zeros((1, 2, 1), dtype=torch.long)
    with pytest.raises(ValueError, match="compile_loss"):
        streaming_language_model_losses(
            hidden,
            head,
            labels,
            indices,
            torch.zeros((1, 2, 1)),
            torch.zeros((1, 2)),
            torch.zeros((1, 2)),
            temperature=1.0,
            compile_loss=1,  # type: ignore[arg-type]
        )


def test_compiled_streaming_loss_specializations_are_lazy_and_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled: list[tuple[object, bool, bool]] = []

    def fake_compile(function: object, *, fullgraph: bool, dynamic: bool) -> object:
        compiled.append((function, fullgraph, dynamic))
        return object()

    monkeypatch.setattr(torch, "compile", fake_compile)
    no_anchor_factory = loss_module._compiled_streaming_student_sums_no_anchor
    anchor_factory = loss_module._compiled_streaming_student_sums_with_anchor
    no_anchor_factory.cache_clear()
    anchor_factory.cache_clear()
    try:
        no_anchor_first = no_anchor_factory()
        no_anchor_second = no_anchor_factory()
        anchor_first = anchor_factory()
        anchor_second = anchor_factory()

        assert no_anchor_first is no_anchor_second
        assert anchor_first is anchor_second
        assert no_anchor_first is not anchor_first
        assert compiled == [
            (loss_module._streaming_student_sums_no_anchor, True, False),
            (loss_module._streaming_student_sums_with_anchor, True, False),
        ]
        assert no_anchor_factory.cache_info().misses == 1
        assert no_anchor_factory.cache_info().hits == 1
        assert anchor_factory.cache_info().misses == 1
        assert anchor_factory.cache_info().hits == 1
    finally:
        # Never retain the fake callable after monkeypatch restores torch.compile.
        no_anchor_factory.cache_clear()
        anchor_factory.cache_clear()


def test_bucketed_topk_kl_is_zero_for_identical_distribution() -> None:
    logits = torch.tensor([[[2.0, 1.0, 0.0, -1.0]]])
    indices = torch.tensor([[[0, 1]]])
    temperature = 1.0
    log_z = torch.logsumexp(logits / temperature, dim=-1)
    top_prob = torch.gather((logits / temperature).softmax(-1), -1, indices).sum(-1)
    tail_logprob = (1.0 - top_prob).log()
    loss = bucketed_topk_kl(
        logits,
        indices,
        torch.gather(logits, -1, indices),
        log_z,
        tail_logprob,
        temperature=temperature,
    )
    assert abs(float(loss)) < 1e-6


def test_bucketed_topk_kl_matches_full_kl_at_temperature_two_and_masks() -> None:
    # With a three-token vocabulary and top-2 cache, the aggregate tail has one
    # token, so bucketed KL must exactly match full-vocabulary distillation.
    teacher = torch.tensor([[[3.0, 1.0, -2.0], [-5.0, 0.0, 5.0]]], dtype=torch.float32)
    student = torch.tensor([[[1.5, -0.5, 0.25], [50.0, -50.0, 0.0]]], dtype=torch.float32)
    indices = torch.tensor([[[0, 1], [2, 1]]])
    temperature = 2.0
    teacher_scaled = teacher / temperature
    teacher_log_z = torch.logsumexp(teacher_scaled, dim=-1)
    teacher_top_logits = torch.gather(teacher, -1, indices)
    teacher_top_mass = torch.gather(teacher_scaled.softmax(dim=-1), -1, indices).sum(dim=-1)
    teacher_tail_logprob = torch.log1p(-teacher_top_mass)

    actual = bucketed_topk_kl(
        student,
        indices,
        teacher_top_logits,
        teacher_log_z,
        teacher_tail_logprob,
        temperature=temperature,
        mask=torch.tensor([[1, 0]]),
    )
    teacher_probability = F.softmax(teacher[0, 0] / temperature, dim=-1)
    expected = (
        teacher_probability
        * (
            F.log_softmax(teacher[0, 0] / temperature, dim=-1)
            - F.log_softmax(student[0, 0] / temperature, dim=-1)
        )
    ).sum() * temperature**2
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("checkpoint_chunks", [False, True])
def test_bucketed_topk_kl_chunks_match_reference_and_gradients(
    checkpoint_chunks: bool,
) -> None:
    torch.manual_seed(17)
    temperature = 1.7
    teacher = torch.randn(2, 5, 11)
    teacher_scaled = teacher / temperature
    teacher_logsumexp = torch.logsumexp(teacher_scaled, dim=-1)
    teacher_indices = teacher.topk(4, dim=-1).indices
    teacher_topk_logits = torch.gather(teacher, -1, teacher_indices)
    teacher_top_mass = torch.gather(
        teacher_scaled.softmax(dim=-1),
        -1,
        teacher_indices,
    ).sum(dim=-1)
    teacher_tail_logprob = torch.log1p(-teacher_top_mass)
    mask = torch.tensor([[1, 1, 0, 1, 0], [0, 1, 1, 1, 1]])
    expected_student = torch.randn(2, 5, 11, requires_grad=True)
    actual_student = expected_student.detach().clone().requires_grad_(True)

    expected = _reference_bucketed_topk_kl(
        expected_student,
        teacher_indices,
        teacher_topk_logits,
        teacher_logsumexp,
        teacher_tail_logprob,
        temperature=temperature,
        mask=mask,
    )
    actual = bucketed_topk_kl(
        actual_student,
        teacher_indices,
        teacher_topk_logits,
        teacher_logsumexp,
        teacher_tail_logprob,
        temperature=temperature,
        mask=mask,
        chunk_tokens=3,
        checkpoint_chunks=checkpoint_chunks,
    )
    expected.backward()
    actual.backward()

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        actual_student.grad,
        expected_student.grad,
        rtol=2e-6,
        atol=2e-6,
    )


def test_bucketed_topk_kl_all_masked_is_differentiable_zero() -> None:
    student = torch.randn(2, 3, 5, requires_grad=True)
    teacher = torch.randn(2, 3, 5)
    indices = teacher.topk(2, dim=-1).indices
    log_z = torch.logsumexp(teacher, dim=-1)
    top_logits = torch.gather(teacher, -1, indices)
    top_mass = torch.gather(teacher.softmax(dim=-1), -1, indices).sum(dim=-1)
    loss = bucketed_topk_kl(
        student,
        indices,
        top_logits,
        log_z,
        torch.log1p(-top_mass),
        temperature=1.0,
        mask=torch.zeros(2, 3),
        chunk_tokens=2,
    )
    loss.backward()
    assert loss.item() == 0.0
    torch.testing.assert_close(student.grad, torch.zeros_like(student))


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), -float("inf")])
def test_bucketed_topk_kl_rejects_non_finite_temperature(temperature: float) -> None:
    logits = torch.zeros((1, 1, 3))
    indices = torch.tensor([[[0, 1]]])
    with pytest.raises(ValueError, match="temperature"):
        bucketed_topk_kl(
            logits,
            indices,
            torch.zeros((1, 1, 2)),
            torch.zeros((1, 1)),
            torch.zeros((1, 1)),
            temperature=temperature,
        )


@pytest.mark.parametrize("checkpoint_chunks", [False, True])
def test_masked_full_kl_chunks_match_reference_and_gradients(
    checkpoint_chunks: bool,
) -> None:
    torch.manual_seed(23)
    mask = torch.tensor([[1, 0, 1, 1], [0, 1, 1, 0]])
    expected_student = torch.randn(2, 4, 13, requires_grad=True)
    actual_student = expected_student.detach().clone().requires_grad_(True)
    expected_reference = torch.randn(2, 4, 13, requires_grad=True)
    actual_reference = expected_reference.detach().clone().requires_grad_(True)

    student_logprob = F.log_softmax(expected_student.float(), dim=-1)
    reference_prob = F.softmax(expected_reference.float(), dim=-1)
    token_loss = F.kl_div(
        student_logprob,
        reference_prob,
        reduction="none",
    ).sum(dim=-1)
    weight = mask.to(token_loss.dtype)
    expected = (token_loss * weight).sum() / weight.sum().clamp_min(1.0)
    actual = masked_full_kl(
        actual_student,
        actual_reference,
        mask,
        chunk_tokens=3,
        checkpoint_chunks=checkpoint_chunks,
    )
    expected.backward()
    actual.backward()

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        actual_student.grad,
        expected_student.grad,
        rtol=2e-6,
        atol=2e-6,
    )
    torch.testing.assert_close(
        actual_reference.grad,
        expected_reference.grad,
        rtol=2e-6,
        atol=2e-6,
    )


def test_masked_full_kl_none_and_empty_masks() -> None:
    torch.manual_seed(29)
    reference = torch.randn(2, 3, 7)
    unmasked_student = torch.randn(2, 3, 7, requires_grad=True)
    expected = (
        F.kl_div(
            F.log_softmax(unmasked_student.float(), dim=-1),
            F.softmax(reference.float(), dim=-1),
            reduction="none",
        )
        .sum(dim=-1)
        .mean()
    )
    actual = masked_full_kl(
        unmasked_student,
        reference,
        None,
        chunk_tokens=None,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    expected_grad = torch.autograd.grad(expected, unmasked_student, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual, unmasked_student)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-6)

    masked_student = unmasked_student.detach().clone().requires_grad_(True)
    empty = masked_full_kl(
        masked_student,
        reference,
        torch.zeros(2, 3),
        chunk_tokens=2,
    )
    empty.backward()
    assert empty.item() == 0.0
    torch.testing.assert_close(masked_student.grad, torch.zeros_like(masked_student))


@pytest.mark.parametrize("checkpoint_chunks", [False, True])
def test_streaming_language_model_losses_match_full_logits_and_gradients(
    checkpoint_chunks: bool,
) -> None:
    torch.manual_seed(31)
    temperature = 1.8
    labels = torch.tensor(
        [[0, 1, -100, 4, 3], [2, 0, 5, -100, 1]],
        dtype=torch.long,
    )
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 0, 1, 1]])
    teacher = torch.randn(2, 5, 13)
    teacher_scaled = teacher / temperature
    teacher_indices = teacher.topk(4, dim=-1).indices
    teacher_topk_logits = torch.gather(teacher, -1, teacher_indices)
    teacher_logsumexp = torch.logsumexp(teacher_scaled, dim=-1)
    teacher_top_mass = torch.gather(
        teacher_scaled.softmax(dim=-1),
        -1,
        teacher_indices,
    ).sum(dim=-1)
    teacher_tail_logprob = torch.log1p(-teacher_top_mass)

    expected_hidden = torch.randn(2, 5, 7, requires_grad=True)
    actual_hidden = expected_hidden.detach().clone().requires_grad_(True)
    expected_anchor_hidden = torch.randn(2, 5, 7, requires_grad=True)
    actual_anchor_hidden = expected_anchor_hidden.detach().clone().requires_grad_(True)
    expected_head = torch.nn.Linear(7, 13, bias=False)
    actual_head = torch.nn.Linear(7, 13, bias=False)
    actual_head.load_state_dict(expected_head.state_dict())

    expected_logits = expected_head(expected_hidden)
    with torch.no_grad():
        expected_anchor_logits = expected_head(expected_anchor_hidden)
    effective_labels = labels.clone()
    effective_labels[..., 1:] = effective_labels[..., 1:].masked_fill(
        ~mask[..., :-1].bool(), -100
    )
    target_mask = torch.zeros_like(mask)
    target_mask[..., :-1] = effective_labels[..., 1:].ne(-100)
    expected_ntp = causal_language_model_loss(
        expected_logits,
        effective_labels,
        chunk_tokens=3,
        checkpoint_chunks=False,
    )
    expected_kd = bucketed_topk_kl(
        expected_logits,
        teacher_indices,
        teacher_topk_logits,
        teacher_logsumexp,
        teacher_tail_logprob,
        temperature=temperature,
        mask=target_mask,
        chunk_tokens=3,
        checkpoint_chunks=False,
    )
    expected_anchor = masked_full_kl(
        expected_logits,
        expected_anchor_logits,
        target_mask,
        chunk_tokens=3,
        checkpoint_chunks=False,
    )
    actual = streaming_language_model_losses(
        actual_hidden,
        actual_head,
        labels,
        teacher_indices,
        teacher_topk_logits,
        teacher_logsumexp,
        teacher_tail_logprob,
        temperature=temperature,
        mask=mask,
        anchor_hidden_states=actual_anchor_hidden,
        chunk_tokens=3,
        checkpoint_chunks=checkpoint_chunks,
    )
    assert actual.anchor_kl is not None

    torch.testing.assert_close(actual.ntp, expected_ntp, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual.teacher_kd, expected_kd, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual.anchor_kl, expected_anchor, rtol=2e-6, atol=2e-6)

    expected_total = 0.7 * expected_ntp + 1.3 * expected_kd + 0.2 * expected_anchor
    actual_total = 0.7 * actual.ntp + 1.3 * actual.teacher_kd + 0.2 * actual.anchor_kl
    expected_total.backward()
    actual_total.backward()
    torch.testing.assert_close(actual_hidden.grad, expected_hidden.grad, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(
        actual_head.weight.grad,
        expected_head.weight.grad,
        rtol=3e-6,
        atol=3e-6,
    )
    assert expected_anchor_hidden.grad is None
    assert actual_anchor_hidden.grad is None


@pytest.mark.parametrize("checkpoint_chunks", [False, True])
@pytest.mark.parametrize("with_anchor", [False, True])
@pytest.mark.parametrize("temperature", [0.65, 1.0, 2.4])
def test_streaming_combined_reduction_matches_original_formulas_and_all_gradients(
    checkpoint_chunks: bool,
    with_anchor: bool,
    temperature: float,
) -> None:
    torch.manual_seed(33)
    labels = torch.tensor(
        [[0, 2, -100, 5, 1], [3, 4, 1, -100, 2]],
        dtype=torch.long,
    )
    mask = torch.tensor([[1, 1, 0, 1, 1], [1, 0, 1, 1, 0]])
    teacher = torch.randn(2, 5, 9)
    teacher_scaled = teacher / temperature
    teacher_indices = teacher.topk(3, dim=-1).indices
    teacher_topk_logits = torch.gather(teacher, -1, teacher_indices)
    teacher_logsumexp = torch.logsumexp(teacher_scaled, dim=-1)
    teacher_top_mass = torch.gather(
        teacher_scaled.softmax(dim=-1),
        -1,
        teacher_indices,
    ).sum(dim=-1)
    teacher_tail_logprob = torch.log1p(-teacher_top_mass)

    expected_hidden = torch.randn(2, 5, 6, requires_grad=True)
    actual_hidden = expected_hidden.detach().clone().requires_grad_(True)
    expected_head = torch.nn.Linear(6, 9, bias=True)
    actual_head = torch.nn.Linear(6, 9, bias=True)
    actual_head.load_state_dict(expected_head.state_dict())

    expected_anchor_hidden = None
    actual_anchor_hidden = None
    expected_anchor_head = None
    actual_anchor_head = None
    if with_anchor:
        expected_anchor_hidden = torch.randn(2, 5, 6, requires_grad=True)
        actual_anchor_hidden = (
            expected_anchor_hidden.detach().clone().requires_grad_(True)
        )
        expected_anchor_head = torch.nn.Linear(6, 9, bias=True)
        actual_anchor_head = torch.nn.Linear(6, 9, bias=True)
        actual_anchor_head.load_state_dict(expected_anchor_head.state_dict())

    expected_ntp, expected_kd, expected_anchor = (
        _reference_streaming_language_model_losses(
            expected_hidden,
            expected_head,
            labels,
            teacher_indices,
            teacher_topk_logits,
            teacher_logsumexp,
            teacher_tail_logprob,
            temperature=temperature,
            mask=mask,
            anchor_hidden=expected_anchor_hidden,
            anchor_head=expected_anchor_head,
        )
    )
    actual = streaming_language_model_losses(
        actual_hidden,
        actual_head,
        labels,
        teacher_indices,
        teacher_topk_logits,
        teacher_logsumexp,
        teacher_tail_logprob,
        temperature=temperature,
        mask=mask,
        anchor_hidden_states=actual_anchor_hidden,
        anchor_lm_head=actual_anchor_head,
        chunk_tokens=3,
        checkpoint_chunks=checkpoint_chunks,
    )

    torch.testing.assert_close(actual.ntp, expected_ntp, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(actual.teacher_kd, expected_kd, rtol=3e-6, atol=3e-6)
    if with_anchor:
        assert actual.anchor_kl is not None
        assert expected_anchor is not None
        torch.testing.assert_close(
            actual.anchor_kl,
            expected_anchor,
            rtol=3e-6,
            atol=3e-6,
        )
        expected_total = 0.7 * expected_ntp + 1.3 * expected_kd + 0.2 * expected_anchor
        actual_total = 0.7 * actual.ntp + 1.3 * actual.teacher_kd + 0.2 * actual.anchor_kl
    else:
        assert actual.anchor_kl is None
        assert expected_anchor is None
        expected_total = 0.7 * expected_ntp + 1.3 * expected_kd
        actual_total = 0.7 * actual.ntp + 1.3 * actual.teacher_kd

    expected_total.backward()
    actual_total.backward()
    torch.testing.assert_close(
        actual_hidden.grad,
        expected_hidden.grad,
        rtol=5e-6,
        atol=5e-6,
    )
    torch.testing.assert_close(
        actual_head.weight.grad,
        expected_head.weight.grad,
        rtol=5e-6,
        atol=5e-6,
    )
    torch.testing.assert_close(
        actual_head.bias.grad,
        expected_head.bias.grad,
        rtol=5e-6,
        atol=5e-6,
    )
    if with_anchor:
        assert expected_anchor_hidden is not None
        assert actual_anchor_hidden is not None
        assert expected_anchor_head is not None
        assert actual_anchor_head is not None
        assert expected_anchor_hidden.grad is None
        assert actual_anchor_hidden.grad is None
        assert expected_anchor_head.weight.grad is None
        assert expected_anchor_head.bias.grad is None
        assert actual_anchor_head.weight.grad is None
        assert actual_anchor_head.bias.grad is None


@pytest.mark.parametrize("with_anchor", [False, True])
def test_compile_streaming_loss_explicit_true_uses_eager_fallback_on_cpu(
    with_anchor: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_compile() -> object:
        raise AssertionError("CPU streaming loss must not request torch.compile")

    monkeypatch.setattr(
        loss_module,
        "_compiled_streaming_student_sums_no_anchor",
        unexpected_compile,
    )
    monkeypatch.setattr(
        loss_module,
        "_compiled_streaming_student_sums_with_anchor",
        unexpected_compile,
    )
    torch.manual_seed(35)
    temperature = 1.7
    labels = torch.tensor([[0, 2, -100, 4], [3, 1, 5, 0]])
    mask = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1]])
    teacher = torch.randn(2, 4, 7)
    scaled_teacher = teacher / temperature
    indices = teacher.topk(3, dim=-1).indices
    topk_logits = torch.gather(teacher, -1, indices)
    teacher_logsumexp = torch.logsumexp(scaled_teacher, dim=-1)
    top_mass = torch.gather(
        scaled_teacher.softmax(dim=-1),
        -1,
        indices,
    ).sum(dim=-1)
    teacher_tail = torch.log1p(-top_mass)

    eager_hidden = torch.randn(2, 4, 5, requires_grad=True)
    fallback_hidden = eager_hidden.detach().clone().requires_grad_(True)
    eager_head = torch.nn.Linear(5, 7, bias=True)
    fallback_head = torch.nn.Linear(5, 7, bias=True)
    fallback_head.load_state_dict(eager_head.state_dict())
    anchor_hidden = torch.randn(2, 4, 5) if with_anchor else None

    eager = streaming_language_model_losses(
        eager_hidden,
        eager_head,
        labels,
        indices,
        topk_logits,
        teacher_logsumexp,
        teacher_tail,
        temperature=temperature,
        mask=mask,
        anchor_hidden_states=anchor_hidden,
        chunk_tokens=3,
        checkpoint_chunks=True,
        compile_loss=False,
    )
    fallback = streaming_language_model_losses(
        fallback_hidden,
        fallback_head,
        labels,
        indices,
        topk_logits,
        teacher_logsumexp,
        teacher_tail,
        temperature=temperature,
        mask=mask,
        anchor_hidden_states=anchor_hidden,
        chunk_tokens=3,
        checkpoint_chunks=True,
        compile_loss=True,
    )

    torch.testing.assert_close(fallback.ntp, eager.ntp, rtol=0, atol=0)
    torch.testing.assert_close(fallback.teacher_kd, eager.teacher_kd, rtol=0, atol=0)
    eager_total = eager.ntp + eager.teacher_kd
    fallback_total = fallback.ntp + fallback.teacher_kd
    if with_anchor:
        assert eager.anchor_kl is not None
        assert fallback.anchor_kl is not None
        torch.testing.assert_close(fallback.anchor_kl, eager.anchor_kl, rtol=0, atol=0)
        eager_total = eager_total + 0.1 * eager.anchor_kl
        fallback_total = fallback_total + 0.1 * fallback.anchor_kl
    else:
        assert eager.anchor_kl is None
        assert fallback.anchor_kl is None
    eager_total.backward()
    fallback_total.backward()
    torch.testing.assert_close(fallback_hidden.grad, eager_hidden.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        fallback_head.weight.grad,
        eager_head.weight.grad,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        fallback_head.bias.grad,
        eager_head.bias.grad,
        rtol=0,
        atol=0,
    )


def test_streaming_language_model_losses_never_projects_more_than_one_chunk() -> None:
    class RecordingHead(torch.nn.Linear):
        def __init__(self) -> None:
            super().__init__(5, 11, bias=False)
            self.token_counts: list[int] = []

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.token_counts.append(value.shape[0])
            return super().forward(value)

    torch.manual_seed(37)
    hidden = torch.randn(2, 5, 5, requires_grad=True)
    anchor_hidden = torch.randn_like(hidden)
    labels = torch.randint(0, 11, (2, 5))
    teacher = torch.randn(2, 5, 11)
    indices = teacher.topk(3, dim=-1).indices
    topk_logits = torch.gather(teacher, -1, indices)
    teacher_logsumexp = torch.logsumexp(teacher, dim=-1)
    top_mass = torch.gather(teacher.softmax(dim=-1), -1, indices).sum(dim=-1)
    head = RecordingHead()

    result = streaming_language_model_losses(
        hidden,
        head,
        labels,
        indices,
        topk_logits,
        teacher_logsumexp,
        torch.log1p(-top_mass),
        temperature=1.0,
        anchor_hidden_states=anchor_hidden,
        chunk_tokens=3,
        checkpoint_chunks=False,
    )

    assert result.anchor_kl is not None
    # Four chunks, each projected once for the student and once for the anchor.
    assert len(head.token_counts) == 8
    assert max(head.token_counts) == 3
    assert sum(head.token_counts) == 20


def test_streaming_language_model_losses_preserve_bf16_autocast_during_checkpoint() -> None:
    class DtypeRecordingHead(torch.nn.Linear):
        def __init__(self) -> None:
            super().__init__(4, 9, bias=False)
            self.output_dtypes: list[torch.dtype] = []

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            output = super().forward(value)
            self.output_dtypes.append(output.dtype)
            return output

    torch.manual_seed(41)
    hidden = torch.randn(1, 4, 4, requires_grad=True)
    labels = torch.randint(0, 9, (1, 4))
    teacher = torch.randn(1, 4, 9)
    indices = teacher.topk(3, dim=-1).indices
    topk_logits = torch.gather(teacher, -1, indices)
    teacher_logsumexp = torch.logsumexp(teacher, dim=-1)
    top_mass = torch.gather(teacher.softmax(dim=-1), -1, indices).sum(dim=-1)
    head = DtypeRecordingHead()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = streaming_language_model_losses(
            hidden,
            head,
            labels,
            indices,
            topk_logits,
            teacher_logsumexp,
            torch.log1p(-top_mass),
            temperature=1.0,
            chunk_tokens=2,
            checkpoint_chunks=True,
        )
        loss = result.ntp + result.teacher_kd
    loss.backward()

    assert head.output_dtypes
    assert set(head.output_dtypes) == {torch.bfloat16}
    assert torch.isfinite(hidden.grad).all()
    assert torch.isfinite(head.weight.grad).all()


def test_streaming_language_model_losses_accumulate_tied_embedding_head_gradient() -> None:
    class TiedLanguageModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(11, 6)
            self.lm_head = torch.nn.Linear(6, 11, bias=False)
            self.lm_head.weight = self.embed_tokens.weight

    torch.manual_seed(43)
    expected_model = TiedLanguageModel()
    actual_model = TiedLanguageModel()
    actual_model.load_state_dict(expected_model.state_dict())
    input_ids = torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7]])
    labels = input_ids.clone()
    teacher = torch.randn(2, 4, 11)
    indices = teacher.topk(3, dim=-1).indices
    topk_logits = torch.gather(teacher, -1, indices)
    teacher_logsumexp = torch.logsumexp(teacher, dim=-1)
    top_mass = torch.gather(teacher.softmax(dim=-1), -1, indices).sum(dim=-1)
    teacher_tail_logprob = torch.log1p(-top_mass)

    expected_hidden = expected_model.embed_tokens(input_ids)
    expected_logits = expected_model.lm_head(expected_hidden)
    expected_ntp = causal_language_model_loss(
        expected_logits,
        labels,
        chunk_tokens=3,
        checkpoint_chunks=False,
    )
    expected_kd = bucketed_topk_kl(
        expected_logits,
        indices,
        topk_logits,
        teacher_logsumexp,
        teacher_tail_logprob,
        temperature=1.0,
        mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]]),
        chunk_tokens=3,
        checkpoint_chunks=False,
    )

    actual_hidden = actual_model.embed_tokens(input_ids)
    actual = streaming_language_model_losses(
        actual_hidden,
        actual_model.lm_head,
        labels,
        indices,
        topk_logits,
        teacher_logsumexp,
        teacher_tail_logprob,
        temperature=1.0,
        chunk_tokens=3,
        checkpoint_chunks=True,
    )
    assert actual.anchor_kl is None
    (expected_ntp + expected_kd).backward()
    (actual.ntp + actual.teacher_kd).backward()

    torch.testing.assert_close(actual.ntp, expected_ntp, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual.teacher_kd, expected_kd, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        actual_model.embed_tokens.weight.grad,
        expected_model.embed_tokens.weight.grad,
        rtol=3e-6,
        atol=3e-6,
    )


def test_best_expert_pair_prefers_signal_pair() -> None:
    outputs = torch.zeros(1, 4, 2)
    outputs[0, 1] = torch.tensor([1.0, 0.0])
    outputs[0, 3] = torch.tensor([0.0, 1.0])
    pair = best_expert_pair(outputs, dense_output=torch.tensor([[2.0, 2.0]]))
    assert set(pair[0].tolist()) == {1, 3}


def test_router_pair_supervision_masks_padding_tokens() -> None:
    logits = torch.tensor([[[5.0, 4.0, 0.0], [-9.0, -8.0, 9.0]]])
    pairs = torch.tensor([[[0, 1], [0, 1]]])
    masked = router_pair_supervision_loss(
        logits,
        pairs,
        mask=torch.tensor([[1, 0]]),
    )
    expected = router_pair_supervision_loss(logits[:, :1], pairs[:, :1])
    torch.testing.assert_close(masked, expected)
