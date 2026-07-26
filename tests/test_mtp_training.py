from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from twen.training.engine import _batch_mtp_loss_token_count
from twen.training.streaming import (
    StreamingLossCausalLM,
    _streaming_mtp_cross_entropy,
    native_mtp_target_mask,
)


class _TinyBody(nn.Module):
    def __init__(self, vocabulary: int, hidden: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocabulary, hidden)
        self.embed_tokens.weight.requires_grad_(False)
        self.adapter = nn.Linear(hidden, hidden, bias=False)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        output_hidden_states: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        assert return_dict is True
        initial = self.embed_tokens(input_ids)
        hidden = self.adapter(initial)
        return SimpleNamespace(
            last_hidden_state=hidden,
            hidden_states=(initial, hidden) if output_hidden_states else None,
        )


class _TinyCausalLM(nn.Module):
    def __init__(self, vocabulary: int, hidden: int) -> None:
        super().__init__()
        self.model = _TinyBody(vocabulary, hidden)
        self.lm_head = nn.Linear(hidden, vocabulary, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight


class _CountingHead(nn.Linear):
    def __init__(self, hidden: int, vocabulary: int) -> None:
        super().__init__(hidden, vocabulary, bias=False)
        self.calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return super().forward(value)


class _FrozenMTP(nn.Module):
    """Small differentiable stand-in with the native shifted-call contract."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.calls = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        embed_tokens: nn.Module,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        del attention_mask
        self.calls += 1
        return torch.tanh(
            hidden_states[:, :-1] + self.scale * embed_tokens(input_ids[:, 1:])
        )


def _teacher_payload(
    input_ids: torch.Tensor, vocabulary: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn(*input_ids.shape, vocabulary, generator=generator)
    indices = logits.topk(2, dim=-1).indices
    topk_logits = torch.gather(logits, -1, indices)
    logsumexp = torch.logsumexp(logits, dim=-1)
    top_mass = torch.gather(logits.softmax(dim=-1), -1, indices).sum(dim=-1)
    return indices, topk_logits, logsumexp, torch.log1p(-top_mass)


def test_native_mtp_target_mask_is_exact_l_minus_two_contract() -> None:
    labels = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [0, 1, -100, 3, 4, 5],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
        ]
    )

    actual = native_mtp_target_mask(labels, attention_mask)

    assert actual.shape == (2, 4)
    assert actual.tolist() == [[True, True, False, False], [False, True, True, False]]
    batch = SimpleNamespace(labels=labels, attention_mask=attention_mask)
    assert int(_batch_mtp_loss_token_count(batch)) == 4


def test_streaming_wrapper_runs_frozen_mtp_with_l_minus_two_targets_and_gradients() -> None:
    torch.manual_seed(5)
    vocabulary, hidden = 11, 4
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])
    labels = input_ids.clone()
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
    teacher = _teacher_payload(input_ids, vocabulary)
    model = _TinyCausalLM(vocabulary, hidden)
    mtp = _FrozenMTP()
    wrapper = StreamingLossCausalLM(
        model,
        chunk_tokens=2,
        checkpoint_chunks=True,
        mtp=mtp,
    )

    wrapper.train()
    assert mtp.training is False
    outputs = wrapper(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        teacher_indices=teacher[0],
        teacher_topk_logits=teacher[1],
        teacher_logsumexp=teacher[2],
        teacher_tail_logprob=teacher[3],
    )

    assert outputs["mtp"] is not None
    with torch.no_grad():
        final_hidden = model.model.adapter(model.model.embed_tokens(input_ids))
        mtp_hidden = torch.tanh(
            final_hidden[:, :-1]
            + mtp.scale * model.model.embed_tokens(input_ids[:, 1:])
        )
        # Only t=0 and t=1 are valid: the t=2 target is padding.
        expected = F.cross_entropy(
            model.lm_head(mtp_hidden[:, :2]).float().reshape(-1, vocabulary),
            labels[:, 2:4].reshape(-1),
        )
    torch.testing.assert_close(outputs["mtp"], expected)

    outputs["mtp"].backward()
    assert mtp.calls == 2
    assert mtp.scale.grad is None
    assert model.model.embed_tokens.weight.grad is None
    assert model.model.adapter.weight.grad is not None
    assert torch.count_nonzero(model.model.adapter.weight.grad).item() > 0


def test_mtp_body_checkpoint_matches_eager_loss_and_main_gradients() -> None:
    torch.manual_seed(23)
    vocabulary, hidden = 13, 5
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    teacher = _teacher_payload(input_ids, vocabulary)

    eager_model = _TinyCausalLM(vocabulary, hidden)
    checkpoint_model = _TinyCausalLM(vocabulary, hidden)
    checkpoint_model.load_state_dict(eager_model.state_dict())
    eager_mtp = _FrozenMTP()
    checkpoint_mtp = _FrozenMTP()
    checkpoint_mtp.load_state_dict(eager_mtp.state_dict())
    eager_wrapper = StreamingLossCausalLM(
        eager_model,
        chunk_tokens=2,
        checkpoint_chunks=False,
        mtp=eager_mtp,
    )
    checkpoint_wrapper = StreamingLossCausalLM(
        checkpoint_model,
        chunk_tokens=2,
        checkpoint_chunks=True,
        mtp=checkpoint_mtp,
    )
    eager_wrapper.train()
    checkpoint_wrapper.train()

    common = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "teacher_indices": teacher[0],
        "teacher_topk_logits": teacher[1],
        "teacher_logsumexp": teacher[2],
        "teacher_tail_logprob": teacher[3],
        "output_hidden_states": True,
    }
    eager_outputs = eager_wrapper(**common)
    checkpoint_outputs = checkpoint_wrapper(**common)
    eager_hidden = eager_outputs["hidden_states"][-1]
    checkpoint_hidden = checkpoint_outputs["hidden_states"][-1]
    eager_hidden.retain_grad()
    checkpoint_hidden.retain_grad()

    torch.testing.assert_close(checkpoint_outputs["mtp"], eager_outputs["mtp"])
    assert eager_mtp.calls == 1
    assert checkpoint_mtp.calls == 1

    eager_outputs["mtp"].backward()
    checkpoint_outputs["mtp"].backward()

    # Only the checkpointed body is rerun during backward.
    assert eager_mtp.calls == 1
    assert checkpoint_mtp.calls == 2
    assert eager_hidden.grad is not None
    assert checkpoint_hidden.grad is not None
    torch.testing.assert_close(checkpoint_hidden.grad, eager_hidden.grad)
    assert eager_model.model.adapter.weight.grad is not None
    assert checkpoint_model.model.adapter.weight.grad is not None
    torch.testing.assert_close(
        checkpoint_model.model.adapter.weight.grad,
        eager_model.model.adapter.weight.grad,
    )
    assert all(parameter.grad is None for parameter in eager_mtp.parameters())
    assert all(parameter.grad is None for parameter in checkpoint_mtp.parameters())


def test_mtp_body_checkpoint_requires_training_with_gradients() -> None:
    torch.manual_seed(41)
    vocabulary, hidden = 9, 4
    input_ids = torch.tensor([[0, 1, 2, 3]])
    teacher = _teacher_payload(input_ids, vocabulary)
    mtp = _FrozenMTP()
    wrapper = StreamingLossCausalLM(
        _TinyCausalLM(vocabulary, hidden),
        chunk_tokens=2,
        checkpoint_chunks=True,
        mtp=mtp,
    )
    common = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids,
        "teacher_indices": teacher[0],
        "teacher_topk_logits": teacher[1],
        "teacher_logsumexp": teacher[2],
        "teacher_tail_logprob": teacher[3],
    }

    wrapper.train()
    with torch.no_grad():
        no_grad_outputs = wrapper(**common)
    assert no_grad_outputs["mtp"] is not None
    assert no_grad_outputs["mtp"].requires_grad is False
    assert mtp.calls == 1

    mtp.calls = 0
    wrapper.eval()
    eval_outputs = wrapper(**common)
    assert eval_outputs["mtp"] is not None
    eval_outputs["mtp"].backward()
    assert mtp.calls == 1


def test_streaming_wrapper_omits_mtp_loss_when_disabled() -> None:
    vocabulary, hidden = 7, 3
    input_ids = torch.tensor([[0, 1, 2]])
    teacher = _teacher_payload(input_ids, vocabulary)
    wrapper = StreamingLossCausalLM(
        _TinyCausalLM(vocabulary, hidden),
        chunk_tokens=2,
        checkpoint_chunks=False,
    )

    outputs = wrapper(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids,
        teacher_indices=teacher[0],
        teacher_topk_logits=teacher[1],
        teacher_logsumexp=teacher[2],
        teacher_tail_logprob=teacher[3],
    )

    assert outputs["mtp"] is None


def test_prepared_text_wrapper_skips_teacher_payload_and_preserves_native_mtp_gradient() -> None:
    torch.manual_seed(27)
    vocabulary, hidden = 11, 4
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])
    labels = input_ids.clone()
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
    model = _TinyCausalLM(vocabulary, hidden)
    mtp = _FrozenMTP()
    wrapper = StreamingLossCausalLM(
        model,
        chunk_tokens=2,
        checkpoint_chunks=True,
        mtp=mtp,
        teacher_kd_enabled=False,
    )

    outputs = wrapper(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    assert outputs["teacher_kd"] is None
    assert outputs["anchor_kl"] is None
    assert outputs["mtp"] is not None
    (outputs["ntp"] + 0.1 * outputs["mtp"]).backward()
    assert mtp.calls == 2
    assert mtp.scale.grad is None
    assert model.model.embed_tokens.weight.grad is None
    assert model.model.adapter.weight.grad is not None
    assert torch.count_nonzero(model.model.adapter.weight.grad).item() > 0


def test_prepared_text_wrapper_rejects_teacher_or_anchor_payload() -> None:
    vocabulary, hidden = 7, 3
    input_ids = torch.tensor([[0, 1, 2]])
    teacher = _teacher_payload(input_ids, vocabulary)
    wrapper = StreamingLossCausalLM(
        _TinyCausalLM(vocabulary, hidden),
        chunk_tokens=2,
        checkpoint_chunks=False,
        teacher_kd_enabled=False,
    )

    with pytest.raises(ValueError, match="teacher_kd_enabled=false"):
        wrapper(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=input_ids,
            teacher_indices=teacher[0],
        )
    with pytest.raises(ValueError, match="anchor_hidden_states"):
        wrapper(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=input_ids,
            anchor_hidden_states=torch.zeros((1, 3, hidden)),
        )


def test_streaming_wrapper_rejects_non_boolean_teacher_kd_switch() -> None:
    with pytest.raises(ValueError, match="teacher_kd_enabled"):
        StreamingLossCausalLM(
            _TinyCausalLM(7, 3),
            chunk_tokens=2,
            checkpoint_chunks=False,
            teacher_kd_enabled=0,  # type: ignore[arg-type]
        )


def test_mtp_streaming_ce_matches_eager_oracle_and_checkpoints_head() -> None:
    torch.manual_seed(29)
    hidden = torch.randn(1, 4, 3, requires_grad=True)
    targets = torch.tensor([[1, 2, 3, 4]])
    mask = torch.tensor([[True, True, False, True]])
    head = _CountingHead(3, 7)

    actual = _streaming_mtp_cross_entropy(
        hidden,
        head,
        targets,
        mask,
        chunk_tokens=2,
        checkpoint_chunks=True,
        compile_loss=True,
    )
    with torch.no_grad():
        logits = head.weight.detach().new_empty(0)
        del logits
        expected = F.cross_entropy(
            head(hidden.detach()).float().reshape(-1, 7),
            targets.masked_fill(~mask, -100).reshape(-1),
            ignore_index=-100,
        )
    # Two chunk calls happened in forward; the explicit oracle call is third.
    assert head.calls == 3
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    actual.backward()
    # Non-reentrant checkpointing recomputes both LM-head chunks in backward.
    assert head.calls == 5
    assert hidden.grad is not None
    assert torch.count_nonzero(hidden.grad).item() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_mtp_compiled_cuda_reduction_matches_eager_value_and_gradient() -> None:
    torch.manual_seed(31)
    device = torch.device("cuda")
    eager_hidden = torch.randn(1, 4, 16, device=device, dtype=torch.bfloat16)
    eager_hidden.requires_grad_(True)
    compiled_hidden = eager_hidden.detach().clone().requires_grad_(True)
    head = nn.Linear(16, 257, bias=False, device=device, dtype=torch.bfloat16)
    head.weight.requires_grad_(False)
    targets = torch.tensor([[1, 2, 3, 4]], device=device)
    mask = torch.tensor([[True, True, False, True]], device=device)

    eager = _streaming_mtp_cross_entropy(
        eager_hidden,
        head,
        targets,
        mask,
        chunk_tokens=2,
        checkpoint_chunks=False,
        compile_loss=False,
    )
    compiled = _streaming_mtp_cross_entropy(
        compiled_hidden,
        head,
        targets,
        mask,
        chunk_tokens=2,
        checkpoint_chunks=False,
        compile_loss=True,
    )
    eager.backward()
    compiled.backward()

    torch.testing.assert_close(compiled, eager, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        compiled_hidden.grad,
        eager_hidden.grad,
        rtol=2e-2,
        atol=2e-3,
    )
