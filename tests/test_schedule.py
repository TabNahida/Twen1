from types import SimpleNamespace

import pytest

from twen.training.schedule import (
    SparseTopKSchedule,
    TokenCosineSchedule,
    TokenWarmupStableDecaySchedule,
)
from twen.training.stateful import TokenLRScheduler


def test_sparse_topk_schedule() -> None:
    schedule = SparseTopKSchedule()
    assert schedule.value(0, 1_000) == 8
    assert schedule.value(100, 1_000) == 4
    assert schedule.value(190, 1_000) == 2
    assert schedule.value(1_000, 1_000) == 2


def test_token_cosine_schedule() -> None:
    schedule = TokenCosineSchedule(warmup_tokens=100, max_tokens=1_000)
    assert schedule.ratio(0) == 0.0
    assert schedule.ratio(100) == 1.0
    assert abs(schedule.ratio(1_000) - 0.1) < 1e-8


def test_token_warmup_stable_decay_boundaries_are_token_exact() -> None:
    schedule = TokenWarmupStableDecaySchedule(
        warmup_tokens=100,
        max_tokens=1_000,
        decay_tokens=200,
        min_ratio=0.2,
    )

    assert schedule.ratio(0) == 0.0
    assert schedule.ratio(50) == 0.5
    assert schedule.ratio(100) == 1.0
    assert schedule.ratio(800) == 1.0
    assert schedule.ratio(900) == pytest.approx(0.6)
    assert schedule.ratio(1_000) == pytest.approx(0.2)
    assert schedule.ratio(2_000) == pytest.approx(0.2)


def test_wsd_scheduler_state_is_resume_critical() -> None:
    optimizer = SimpleNamespace(param_groups=[{"lr": 2.0}])
    scheduler = TokenLRScheduler(
        optimizer,
        warmup_tokens=100,
        max_tokens=1_000,
        lr_schedule="warmup-stable-decay",
        min_lr_ratio=0.2,
        decay_tokens=200,
    )
    scheduler.step_tokens(900)
    state = scheduler.state_dict()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.2)
    assert state["lr_schedule"] == "warmup-stable-decay"
    assert state["min_lr_ratio"] == 0.2
    assert state["decay_tokens"] == 200

    restored_optimizer = SimpleNamespace(param_groups=[{"lr": 2.0}])
    restored = TokenLRScheduler(
        restored_optimizer,
        warmup_tokens=100,
        max_tokens=1_000,
        lr_schedule="warmup-stable-decay",
        min_lr_ratio=0.2,
        decay_tokens=200,
    )
    restored.load_state_dict(state)
    assert restored.consumed_tokens == 900
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(1.2)

    incompatible = TokenLRScheduler(
        SimpleNamespace(param_groups=[{"lr": 2.0}]),
        warmup_tokens=100,
        max_tokens=1_000,
        lr_schedule="warmup-stable-decay",
        min_lr_ratio=0.2,
        decay_tokens=100,
    )
    with pytest.raises(ValueError, match="decay_tokens changed"):
        incompatible.load_state_dict(state)


def test_legacy_scheduler_keeps_v1_state_shape_and_loads_old_state() -> None:
    optimizer = SimpleNamespace(param_groups=[{"lr": 2.0}])
    scheduler = TokenLRScheduler(
        optimizer,
        warmup_tokens=100,
        max_tokens=1_000,
    )
    expected = {
        "consumed_tokens": 0,
        "base_lrs": [2.0],
        "warmup_tokens": 100,
        "max_tokens": 1_000,
    }
    assert scheduler.state_dict() == expected

    old_state = {**expected, "consumed_tokens": 100}
    scheduler.load_state_dict(old_state)
    assert optimizer.param_groups[0]["lr"] == 2.0
