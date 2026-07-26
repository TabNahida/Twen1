from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from twen.cli import build_parser
from twen.config import LossConfig
from twen.data import PreparedTextBatch, TeacherKDBatch
from twen.training.builder import BuiltModel
from twen.training.distributed import DistributedContext
from twen.training.engine import (
    _batch_loss_token_counts,
    _execute_graph_smoke_microbatch,
    _hidden_alignment_loss,
    _token_mean_contribution,
    run_training,
)
from twen.training.streaming import StreamingLossCausalLM


class _Adapters(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(hidden))

    def project_output(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.weight.T


class _DenseTransfer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.adapters = _Adapters(hidden)
        self.branch_scale = nn.Parameter(torch.tensor(0.1))
        self.transfer_mlp = SimpleNamespace(adapters=self.adapters)
        self.transfer_enabled = True
        self.last_aux = None

    def set_transfer_enabled(self, enabled: bool) -> None:
        self.transfer_enabled = enabled

    def set_record_aux(self, _enabled: bool) -> None:
        return None

    def clear_aux(self) -> None:
        self.last_aux = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.transfer_enabled:
            return hidden
        return hidden + self.branch_scale * self.adapters.project_output(hidden)


class _SparseTransfer(nn.Module):
    def __init__(self, hidden: int, experts: int) -> None:
        super().__init__()
        self.router = nn.Linear(hidden, experts, bias=False)
        self.expert_weight = nn.Parameter(torch.randn(experts, hidden, hidden) * 0.05)
        self.branch_scale = nn.Parameter(torch.tensor(0.1))
        self.transfer_enabled = True
        self.record_aux = False
        self.top_k = experts
        self.last_aux = None

    def set_transfer_enabled(self, enabled: bool) -> None:
        self.transfer_enabled = enabled

    def set_record_aux(self, enabled: bool) -> None:
        self.record_aux = enabled

    def clear_aux(self) -> None:
        self.last_aux = None

    def set_top_k(self, value: int) -> None:
        self.top_k = value

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.transfer_enabled:
            return hidden
        router_logits = self.router(hidden)
        probabilities = router_logits.softmax(dim=-1)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        expert_outputs = torch.einsum("bsh,ehd->bsed", hidden, self.expert_weight)
        selected = torch.gather(
            expert_outputs,
            -2,
            indices.unsqueeze(-1).expand(*indices.shape, hidden.shape[-1]),
        )
        routed = (selected * weights.unsqueeze(-1)).sum(dim=-2) * self.branch_scale
        scaled_all = expert_outputs * self.branch_scale
        self.last_aux = {
            "router_logits": router_logits,
            "expert_indices": indices,
            "expert_weights": weights,
            "expert_outputs": scaled_all if self.record_aux else None,
            "dense_sum": scaled_all.sum(dim=-2) if self.record_aux else None,
            "routed_output": routed,
        }
        return hidden + routed


class _RecordingHead(nn.Linear):
    def __init__(self, hidden: int, vocabulary: int) -> None:
        super().__init__(hidden, vocabulary, bias=False)
        self.token_counts: list[int] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.token_counts.append(int(value.shape[0]))
        return super().forward(value)


class _TinyBody(nn.Module):
    def __init__(self, transfer: nn.Module, *, vocabulary: int, hidden: int) -> None:
        super().__init__()
        self.transfer = transfer
        self.embed_tokens = nn.Embedding(vocabulary, hidden)
        self.embed_tokens.weight.requires_grad_(False)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        assert return_dict is True
        initial = self.embed_tokens(input_ids)
        hidden = self.transfer(initial)
        return SimpleNamespace(
            last_hidden_state=hidden,
            hidden_states=(initial, hidden) if output_hidden_states else None,
        )


class _TinyModel(nn.Module):
    def __init__(self, transfer: nn.Module, *, vocabulary: int, hidden: int) -> None:
        super().__init__()
        self.model = _TinyBody(transfer, vocabulary=vocabulary, hidden=hidden)
        self.lm_head = _RecordingHead(hidden, vocabulary)
        # Qwen3.5-0.8B ties these exact Parameter objects.
        self.lm_head.weight = self.model.embed_tokens.weight
        self.full_causal_forward_calls = 0

    def forward(self, **_kwargs: object) -> SimpleNamespace:
        self.full_causal_forward_calls += 1
        raise AssertionError("full-logit causal-LM forward must not run")


class _TinyFrozenMTP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25), requires_grad=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        embed_tokens: nn.Module,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        del attention_mask
        return hidden_states[:, :-1] + self.scale * embed_tokens(input_ids[:, 1:])


class _TinyTeacher(nn.Module):
    def __init__(self, vocabulary: int, hidden: int) -> None:
        super().__init__()
        self.register_buffer("embedding", torch.randn(vocabulary, hidden))

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        output_hidden_states: bool,
    ) -> SimpleNamespace:
        del attention_mask, use_cache, output_hidden_states
        hidden = self.embedding[input_ids]
        return SimpleNamespace(hidden_states=(hidden, hidden * 0.9))


def _batch(*, vocabulary: int = 7, temperature: float = 1.0) -> TeacherKDBatch:
    torch.manual_seed(42)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    teacher = torch.randn(1, 4, vocabulary)
    scaled = teacher / temperature
    topk_indices = teacher.topk(2, dim=-1).indices
    topk_logits = torch.gather(teacher, -1, topk_indices)
    teacher_logsumexp = torch.logsumexp(scaled, dim=-1)
    top_mass = torch.gather(scaled.softmax(dim=-1), -1, topk_indices).sum(dim=-1)
    return TeacherKDBatch(
        input_ids=input_ids,
        labels=input_ids.clone(),
        attention_mask=torch.ones_like(input_ids),
        topk_indices=topk_indices,
        topk_logits=topk_logits,
        teacher_logsumexp=teacher_logsumexp,
        teacher_tail_logprob=torch.log1p(-top_mass),
        temperature=temperature,
    )


def _config(stage: str) -> SimpleNamespace:
    losses = LossConfig(
        ntp=1.0,
        teacher_kd=1.0,
        hidden_alignment=0.2 if stage == "dense-oracle" else 0.0,
        anchor_kl=0.1,
        dense_oracle=1.0 if stage == "sparse" else 0.0,
        router_supervision=1.0 if stage == "sparse" else 0.0,
        load_balance=0.01 if stage == "sparse" else 0.0,
        router_z=0.001 if stage == "sparse" else 0.0,
        dense_oracle_batch_fraction=1.0 if stage == "sparse" else 0.0,
        hidden_alignment_batch_fraction=1.0 if stage == "dense-oracle" else 0.0,
    )
    return SimpleNamespace(
        stage=stage,
        losses=losses,
        architecture=SimpleNamespace(num_experts=2),
        runtime=SimpleNamespace(
            bf16=False,
            loss_chunk_tokens=2,
            loss_checkpoint_chunks=False,
            compile_streaming_loss=True,
        ),
    )


def _context() -> DistributedContext:
    return DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
        initialized_here=False,
    )


def _report(gradient_accumulation_steps: int) -> SimpleNamespace:
    return SimpleNamespace(
        config_fingerprint="c" * 64,
        data_fingerprint="d" * 64,
        source_tree_sha256="e" * 64,
        batch=SimpleNamespace(
            gradient_accumulation_steps=gradient_accumulation_steps,
        ),
    )


def test_dense_graph_smoke_executes_anchor_hidden_and_backward_on_cpu() -> None:
    torch.manual_seed(7)
    transfer = _DenseTransfer(hidden=4)
    model = _TinyModel(transfer, vocabulary=7, hidden=4)
    train_model = StreamingLossCausalLM(
        model,
        chunk_tokens=2,
        checkpoint_chunks=False,
        compile_loss=True,
    )
    result = _execute_graph_smoke_microbatch(
        _config("dense-oracle"),  # type: ignore[arg-type]
        _report(2),  # type: ignore[arg-type]
        _context(),
        BuiltModel(model=model, transfer_modules=(transfer,), student_layer_indices=(0,)),
        train_model,
        _batch(),
        dtype=torch.float32,
        teacher=_TinyTeacher(vocabulary=7, hidden=4),
        layer_mapping=(0,),
        data_source="cpu-fixture",
    )

    assert result["ok"] is True
    assert result["no_optimizer_created"] is True
    assert result["no_optimizer_steps"] is True
    assert result["config_fingerprint"] == "c" * 64
    assert result["data_fingerprint"] == "d" * 64
    assert result["source_tree_sha256"] == "e" * 64
    assert result["loss_finite"] is True
    assert result["grad_finite"] is True
    assert result["missing_grad_tensors"] == 0
    assert result["data_source"] == "cpu-fixture"
    assert train_model.compile_loss is True
    assert model.full_causal_forward_calls == 0
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert model.lm_head.token_counts == [2, 2, 2, 2]
    assert {"total", "ntp", "teacher_kd", "anchor_kl", "hidden_alignment"} <= set(
        result["loss_components"]
    )


def test_dense_graph_smoke_executes_native_mtp_loss_and_student_gradient() -> None:
    torch.manual_seed(13)
    transfer = _DenseTransfer(hidden=4)
    model = _TinyModel(transfer, vocabulary=7, hidden=4)
    mtp = _TinyFrozenMTP()
    config = _config("dense-oracle")
    config.losses.mtp = 0.3
    train_model = StreamingLossCausalLM(
        model,
        chunk_tokens=2,
        checkpoint_chunks=False,
        mtp=mtp,
    )
    result = _execute_graph_smoke_microbatch(
        config,  # type: ignore[arg-type]
        _report(1),  # type: ignore[arg-type]
        _context(),
        BuiltModel(
            model=model,
            transfer_modules=(transfer,),
            student_layer_indices=(0,),
            mtp=mtp,
        ),
        train_model,
        _batch(),
        dtype=torch.float32,
        teacher=_TinyTeacher(vocabulary=7, hidden=4),
        layer_mapping=(0,),
        data_source="cpu-mtp-fixture",
    )

    assert result["ok"] is True
    assert result["loss_components"]["mtp"] > 0
    assert all(parameter.grad is None for parameter in mtp.parameters())


def test_prepared_text_graph_smoke_runs_ntp_and_native_mtp_without_kd_fields() -> None:
    torch.manual_seed(17)
    transfer = _DenseTransfer(hidden=4)
    model = _TinyModel(transfer, vocabulary=7, hidden=4)
    mtp = _TinyFrozenMTP()
    config = _config("dense-oracle")
    config.data = SimpleNamespace(mode="prepared-text")
    config.losses.mtp = 0.3
    config.losses.teacher_kd = 0.0
    config.losses.anchor_kl = 0.0
    config.losses.hidden_alignment = 0.0
    kd_batch = _batch()
    batch = PreparedTextBatch(
        input_ids=kd_batch.input_ids,
        labels=kd_batch.labels,
        attention_mask=kd_batch.attention_mask,
    )
    train_model = StreamingLossCausalLM(
        model,
        chunk_tokens=2,
        checkpoint_chunks=False,
        mtp=mtp,
        teacher_kd_enabled=False,
    )

    result = _execute_graph_smoke_microbatch(
        config,  # type: ignore[arg-type]
        _report(1),  # type: ignore[arg-type]
        _context(),
        BuiltModel(
            model=model,
            transfer_modules=(transfer,),
            student_layer_indices=(0,),
            mtp=mtp,
        ),
        train_model,
        batch,
        dtype=torch.float32,
        teacher=None,
        layer_mapping=(0,),
        data_source="prepared-text-cpu-fixture",
    )

    assert result["ok"] is True
    assert result["data_mode"] == "prepared-text"
    assert set(result["loss_components"]) == {"total", "ntp", "mtp"}
    assert all(parameter.grad is None for parameter in mtp.parameters())


def test_streaming_wrapper_rejects_non_boolean_compile_loss() -> None:
    model = _TinyModel(_DenseTransfer(hidden=4), vocabulary=7, hidden=4)
    with pytest.raises(ValueError, match="compile_loss"):
        StreamingLossCausalLM(
            model,
            chunk_tokens=2,
            checkpoint_chunks=False,
            compile_loss=1,  # type: ignore[arg-type]
        )


def test_hidden_alignment_masks_padding_tokens_and_gradients() -> None:
    transfer = _DenseTransfer(hidden=2)
    target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], requires_grad=True)
    teacher = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    loss = _hidden_alignment_loss(
        (target, target),
        (teacher, teacher),
        (transfer,),
        (0,),
        (0,),
        torch.tensor([[1, 0]]),
    )

    loss.backward()
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    assert target.grad is not None
    torch.testing.assert_close(target.grad[:, 1], torch.zeros_like(target.grad[:, 1]))


def test_optimizer_batch_scaling_is_exact_token_mean_across_microbatches_and_ranks() -> None:
    batch = _batch()
    batch.labels[0, 3] = -100
    batch.attention_mask[0, 3] = 0
    target_count, hidden_count = _batch_loss_token_counts(batch)
    assert int(target_count) == 2
    assert int(hidden_count) == 3

    parameter = torch.tensor(1.0, requires_grad=True)
    global_count = torch.tensor(4097)
    contribution = _token_mean_contribution(
        parameter,
        torch.tensor(4096),
        global_count,
        world_size=1,
    ) + _token_mean_contribution(
        parameter * 9.0,
        torch.tensor(1),
        global_count,
        world_size=1,
    )
    contribution.backward()
    torch.testing.assert_close(parameter.grad, torch.tensor((4096.0 + 9.0) / 4097.0))

    # DDP/FSDP average rank gradients.  The explicit world-size factor makes
    # that average equal the global token-weighted objective.
    rank_zero = _token_mean_contribution(
        torch.tensor(1.0), torch.tensor(3), torch.tensor(4), world_size=2
    )
    rank_one = _token_mean_contribution(
        torch.tensor(5.0), torch.tensor(1), torch.tensor(4), world_size=2
    )
    torch.testing.assert_close((rank_zero + rank_one) / 2.0, torch.tensor(2.0))


def test_sparse_graph_smoke_executes_all_router_auxiliary_losses_on_cpu() -> None:
    torch.manual_seed(9)
    transfer = _SparseTransfer(hidden=4, experts=2)
    model = _TinyModel(transfer, vocabulary=7, hidden=4)
    train_model = StreamingLossCausalLM(
        model,
        chunk_tokens=2,
        checkpoint_chunks=False,
    )
    result = _execute_graph_smoke_microbatch(
        _config("sparse"),  # type: ignore[arg-type]
        _report(1),  # type: ignore[arg-type]
        _context(),
        BuiltModel(model=model, transfer_modules=(transfer,), student_layer_indices=(0,)),
        train_model,
        _batch(),
        dtype=torch.float32,
        teacher=None,
        layer_mapping=(0,),
        data_source="cpu-fixture",
    )

    assert result["ok"] is True
    assert result["grad_finite"] is True
    assert {
        "router_aux",
        "router_z",
        "load_balance",
        "dense_oracle",
        "router_supervision",
        "router_entropy",
    } <= set(result["loss_components"])


def test_train_graph_smoke_preflights_then_bypasses_optimizer_path() -> None:
    config = SimpleNamespace()
    report = SimpleNamespace(source_tree_sha256="e" * 64)
    order: list[str] = []

    def preflight(_config: object) -> object:
        order.append("preflight")
        return report

    def smoke(_config: object, actual_report: object) -> int:
        assert actual_report is report
        order.append("smoke")
        return 0

    with (
        patch("twen.training.engine.run_coordinated_training_preflight", side_effect=preflight),
        patch("twen.training.engine.twen_source_tree_sha256", return_value="e" * 64),
        patch("twen.training.engine._run_graph_smoke", side_effect=smoke),
        patch(
            "twen.training.engine._build_optimizer",
            side_effect=AssertionError("optimizer must not be constructed"),
        ),
    ):
        assert run_training(config, graph_smoke=True) == 0  # type: ignore[arg-type]
    assert order == ["preflight", "smoke"]


def test_train_rejects_source_tree_change_after_preflight() -> None:
    report = SimpleNamespace(source_tree_sha256="a" * 64)
    with (
        patch("twen.training.engine.run_coordinated_training_preflight", return_value=report),
        patch("twen.training.engine.twen_source_tree_sha256", return_value="b" * 64),
        patch("twen.training.engine._run_graph_smoke") as smoke,
        pytest.raises(RuntimeError, match="source tree changed"),
    ):
        run_training(SimpleNamespace(), graph_smoke=True)  # type: ignore[arg-type]
    smoke.assert_not_called()


def test_train_cli_graph_smoke_and_dry_run_are_mutually_exclusive() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--stage",
            "dense-oracle",
            "--config",
            "config.yaml",
            "--graph-smoke",
        ]
    )
    assert args.graph_smoke is True
    assert args.dry_run is False

    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        parser.parse_args(
            [
                "train",
                "--stage",
                "dense-oracle",
                "--config",
                "config.yaml",
                "--dry-run",
                "--graph-smoke",
            ]
        )
