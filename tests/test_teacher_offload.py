from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import torch

from twen.training.teacher_offload import TeacherCPUOffloadManager


class _MixedTeacher(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = torch.nn.Module()
        self.up = torch.nn.Module()
        self.down = torch.nn.Module()
        self.gate.weight = torch.nn.Parameter(
            torch.empty((3, 2), device="meta"), requires_grad=False
        )
        self.up.weight = torch.nn.Parameter(torch.empty((3, 2), device="meta"), requires_grad=False)
        self.down.weight = torch.nn.Parameter(
            torch.empty((2, 3), device="meta"), requires_grad=False
        )
        self.teacher_only = torch.nn.Parameter(
            torch.arange(8, dtype=torch.float32).reshape(2, 4),
            requires_grad=False,
        )
        self.register_buffer("frequency", torch.arange(3, dtype=torch.float32))


def _manager() -> tuple[_MixedTeacher, TeacherCPUOffloadManager]:
    teacher = _MixedTeacher()
    transfer = SimpleNamespace(
        transfer_mlp=SimpleNamespace(
            gate_weight=teacher.gate.weight,
            up_weight=teacher.up.weight,
            down_weight=teacher.down.weight,
        )
    )
    manager = TeacherCPUOffloadManager.from_transfer_modules(
        teacher,
        (transfer,),
        target_device=torch.device("meta"),
    )
    return teacher, manager


def test_cpu_shadow_stage_restore_preserves_aliases_ids_and_values() -> None:
    teacher, manager = _manager()
    original_value = teacher.teacher_only.detach().clone()
    original_ids = {name: id(value) for name, value in teacher.named_parameters()}
    mapped = teacher.gate.weight

    staged = manager.stage()

    assert manager.is_staged
    assert teacher.teacher_only.device.type == "meta"
    assert teacher.frequency.device.type == "meta"
    assert teacher.gate.weight is mapped
    assert staged.transferred_bytes == original_value.numel() * 4 + 3 * 4
    assert staged.released_cuda_bytes == 0

    restored = manager.restore()

    assert not manager.is_staged
    assert teacher.teacher_only.device.type == "cpu"
    assert teacher.frequency.device.type == "cpu"
    torch.testing.assert_close(teacher.teacher_only, original_value)
    assert {name: id(value) for name, value in teacher.named_parameters()} == original_ids
    assert teacher.gate.weight is mapped
    assert restored.transferred_bytes == 0
    assert restored.released_cuda_bytes == staged.transferred_bytes


def test_staged_context_restores_cpu_shadow_when_body_raises() -> None:
    teacher, manager = _manager()

    with (
        pytest.raises(RuntimeError, match="body failed"),
        manager.staged() as session,
    ):
        assert session.restore is None
        assert teacher.teacher_only.device.type == "meta"
        raise RuntimeError("body failed")

    assert not manager.is_staged
    assert teacher.teacher_only.device.type == "cpu"
    assert teacher.frequency.device.type == "cpu"


def test_state_machine_rejects_nested_stage_and_unstaged_restore() -> None:
    _, manager = _manager()
    with pytest.raises(RuntimeError, match="not staged"):
        manager.restore()
    manager.stage()
    with pytest.raises(RuntimeError, match="already staged"):
        manager.stage()
    manager.restore()


def test_stage_validation_failure_rolls_back_to_a_reusable_split_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher, manager = _manager()
    original_validate = manager._validate_staged_residency

    def reject_staged_state() -> None:
        raise RuntimeError("post-stage validation failed")

    monkeypatch.setattr(manager, "_validate_staged_residency", reject_staged_state)
    with pytest.raises(RuntimeError, match="post-stage validation failed"):
        manager.stage()

    assert not manager.is_staged
    assert teacher.teacher_only.device.type == "cpu"
    assert teacher.frequency.device.type == "cpu"

    monkeypatch.setattr(manager, "_validate_staged_residency", original_validate)
    manager.stage()
    manager.restore()
    assert not manager.is_staged


def test_restore_transition_time_includes_the_pre_swap_synchronization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, manager = _manager()
    manager.stage()
    delay_seconds = 0.01
    monkeypatch.setattr(manager, "_synchronize", lambda: time.sleep(delay_seconds))

    restored = manager.restore()

    assert restored.seconds >= delay_seconds


def test_manager_rejects_trainable_or_incomplete_alias_teacher() -> None:
    teacher, _ = _manager()
    teacher.teacher_only.requires_grad_(True)
    with pytest.raises(ValueError, match="fully frozen"):
        TeacherCPUOffloadManager(
            teacher,
            resident_parameter_names=frozenset({"gate.weight", "up.weight", "down.weight"}),
            target_device="meta",
        )

    teacher.teacher_only.requires_grad_(False)
    transfer = SimpleNamespace(
        transfer_mlp=SimpleNamespace(
            gate_weight=teacher.gate.weight,
            up_weight=teacher.up.weight,
            down_weight=torch.nn.Parameter(torch.empty((2, 3), device="meta")),
        )
    )
    with pytest.raises(ValueError, match="aliases are incomplete"):
        TeacherCPUOffloadManager.from_transfer_modules(
            teacher,
            (transfer,),
            target_device="meta",
        )
