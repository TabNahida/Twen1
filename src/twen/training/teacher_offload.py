"""Selective single-GPU residency for a frozen, donor-aliased teacher."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TeacherResidencyTransition:
    operation: str
    seconds: float
    parameter_bytes: int
    buffer_bytes: int
    transferred_bytes: int
    released_cuda_bytes: int


@dataclass(slots=True)
class TeacherOffloadSession:
    stage: TeacherResidencyTransition
    restore: TeacherResidencyTransition | None = None


class TeacherCPUOffloadManager:
    """Keep shared donor FFNs on CUDA and teacher-exclusive state on CPU.

    The teacher must be frozen and already split: Parameters shared with the
    student donor branches reside on ``target_device`` while every other
    teacher Parameter and buffer resides on CPU.  Staging installs device
    copies by swapping TensorImpls into the existing Parameter objects and
    retains the original CPU TensorImpls as immutable shadows.  Restoring swaps
    those CPU shadows back and drops the temporary device copies, so it performs
    no device-to-host transfer.

    ``meta`` is accepted only to make ownership/state-machine tests CPU-only.
    Production construction is additionally guarded by the engine and always
    supplies one CUDA device.
    """

    def __init__(
        self,
        teacher: Any,
        *,
        resident_parameter_names: frozenset[str],
        target_device: Any,
    ) -> None:
        import torch

        self.teacher = teacher
        self.target_device = torch.device(target_device)
        if self.target_device.type not in {"cuda", "meta"}:
            raise ValueError("teacher CPU offload target must be CUDA")
        if any(parameter.requires_grad for parameter in teacher.parameters()):
            raise ValueError("teacher CPU offload requires a fully frozen teacher")

        parameters = dict(teacher.named_parameters())
        if not resident_parameter_names:
            raise ValueError("teacher CPU offload requires mapped resident Parameters")
        missing = sorted(resident_parameter_names - parameters.keys())
        if missing:
            raise ValueError(f"resident teacher Parameter is missing: {missing[0]}")
        self.resident_parameter_names = resident_parameter_names
        self.teacher_only_parameter_names = frozenset(parameters) - resident_parameter_names
        self._parameter_ids = {name: id(value) for name, value in parameters.items()}
        self._resident_parameters = {
            name: parameters[name] for name in self.resident_parameter_names
        }
        self.parameter_bytes = sum(
            int(parameters[name].numel() * parameters[name].element_size())
            for name in self.teacher_only_parameter_names
        )
        self.buffer_bytes = _unique_tensor_bytes(tuple(teacher.buffers()))
        self._parameter_shadows: dict[str, Any] | None = None
        self._buffer_shadows: list[tuple[Any, Any]] | None = None
        self._validate_initial_residency(parameters)

    @classmethod
    def from_transfer_modules(
        cls,
        teacher: Any,
        transfer_modules: tuple[Any, ...],
        *,
        target_device: Any,
    ) -> TeacherCPUOffloadManager:
        shared_parameters = []
        for module in transfer_modules:
            transfer = module.transfer_mlp
            shared_parameters.extend(
                (transfer.gate_weight, transfer.up_weight, transfer.down_weight)
            )
        expected = len(transfer_modules) * 3
        if len({id(value) for value in shared_parameters}) != expected:
            raise ValueError("mapped donor projections must be distinct Parameters")
        shared_ids = {id(value) for value in shared_parameters}
        resident_names = frozenset(
            name for name, value in teacher.named_parameters() if id(value) in shared_ids
        )
        if len(resident_names) != expected:
            raise ValueError(
                "teacher/donor aliases are incomplete: "
                f"expected {expected}, found {len(resident_names)}"
            )
        return cls(
            teacher,
            resident_parameter_names=resident_names,
            target_device=target_device,
        )

    @property
    def is_staged(self) -> bool:
        return self._parameter_shadows is not None

    @property
    def staged_bytes(self) -> int:
        return self.parameter_bytes + self.buffer_bytes

    def stage(self) -> TeacherResidencyTransition:
        """Copy teacher-only state to the target while retaining CPU shadows."""

        import torch

        if self.is_staged:
            raise RuntimeError("teacher is already staged")
        self._validate_parameter_objects()
        parameters = dict(self.teacher.named_parameters())
        parameter_shadows: dict[str, Any] = {}
        buffer_shadows: list[tuple[Any, Any]] = []
        self._synchronize()
        started = time.perf_counter()
        try:
            for name in sorted(self.teacher_only_parameter_names):
                parameter = parameters[name]
                if parameter.device.type != "cpu":
                    raise RuntimeError(
                        f"teacher-only Parameter is not CPU resident before stage: {name}"
                    )
                target_value = torch.nn.Parameter(
                    parameter.detach().to(device=self.target_device),
                    requires_grad=False,
                )
                torch.utils.swap_tensors(parameter, target_value)
                # target_value now owns the immutable original CPU TensorImpl.
                parameter_shadows[name] = target_value

            seen_buffers: set[int] = set()
            for buffer in self.teacher.buffers():
                if id(buffer) in seen_buffers:
                    continue
                seen_buffers.add(id(buffer))
                if buffer.device.type != "cpu":
                    raise RuntimeError("teacher buffer is not CPU resident before stage")
                target_value = buffer.to(device=self.target_device)
                torch.utils.swap_tensors(buffer, target_value)
                buffer_shadows.append((buffer, target_value))
            self._parameter_shadows = parameter_shadows
            self._buffer_shadows = buffer_shadows
            self._synchronize()
            self._validate_staged_residency()
        except BaseException:
            _swap_back(parameter_shadows, buffer_shadows, parameters, torch=torch)
            self._parameter_shadows = None
            self._buffer_shadows = None
            self._synchronize()
            raise
        return TeacherResidencyTransition(
            operation="stage",
            seconds=time.perf_counter() - started,
            parameter_bytes=self.parameter_bytes,
            buffer_bytes=self.buffer_bytes,
            transferred_bytes=self.staged_bytes,
            released_cuda_bytes=0,
        )

    def restore(self) -> TeacherResidencyTransition:
        """Restore CPU shadows and release device copies without a D2H copy."""

        import torch

        if self._parameter_shadows is None or self._buffer_shadows is None:
            raise RuntimeError("teacher is not staged")
        started = time.perf_counter()
        # Teacher forward/backward kernels must finish before their Parameter
        # storages are swapped out and released.  Include this barrier in the
        # transition telemetry: otherwise an asynchronous backward tail would
        # be charged to neither graph compute nor restore time.
        self._synchronize()
        parameters = dict(self.teacher.named_parameters())
        _swap_back(
            self._parameter_shadows,
            self._buffer_shadows,
            parameters,
            torch=torch,
        )
        self._parameter_shadows = None
        self._buffer_shadows = None
        self._synchronize()
        self._validate_initial_residency(parameters)
        return TeacherResidencyTransition(
            operation="restore",
            seconds=time.perf_counter() - started,
            parameter_bytes=self.parameter_bytes,
            buffer_bytes=self.buffer_bytes,
            transferred_bytes=0,
            released_cuda_bytes=self.staged_bytes,
        )

    @contextmanager
    def staged(self) -> Any:
        session = TeacherOffloadSession(stage=self.stage())
        try:
            yield session
        finally:
            session.restore = self.restore()

    def _validate_initial_residency(self, parameters: dict[str, Any]) -> None:
        self._validate_parameter_objects(parameters)
        wrong_resident = [
            name
            for name in self.resident_parameter_names
            if parameters[name].device != self.target_device
        ]
        if wrong_resident:
            raise ValueError(f"mapped donor Parameter is not target-resident: {wrong_resident[0]}")
        wrong_cpu = [
            name
            for name in self.teacher_only_parameter_names
            if parameters[name].device.type != "cpu"
        ]
        if wrong_cpu:
            raise ValueError(f"teacher-only Parameter is not CPU resident: {wrong_cpu[0]}")
        if any(buffer.device.type != "cpu" for buffer in self.teacher.buffers()):
            raise ValueError("teacher buffers must be CPU resident while offloaded")

    def _validate_staged_residency(self) -> None:
        parameters = dict(self.teacher.named_parameters())
        self._validate_parameter_objects(parameters)
        wrong = [name for name in parameters if parameters[name].device != self.target_device]
        if wrong:
            raise RuntimeError(f"staged teacher Parameter is on the wrong device: {wrong[0]}")
        if any(buffer.device != self.target_device for buffer in self.teacher.buffers()):
            raise RuntimeError("staged teacher buffer is on the wrong device")

    def _validate_parameter_objects(self, parameters: dict[str, Any] | None = None) -> None:
        current = dict(self.teacher.named_parameters()) if parameters is None else parameters
        changed = [
            name for name, value in current.items() if id(value) != self._parameter_ids[name]
        ]
        if changed:
            raise RuntimeError(f"teacher Parameter object identity changed: {changed[0]}")
        broken = [
            name
            for name, expected in self._resident_parameters.items()
            if current[name] is not expected
        ]
        if broken:
            raise RuntimeError(f"mapped donor Parameter alias changed: {broken[0]}")

    def _synchronize(self) -> None:
        if self.target_device.type == "cuda":
            import torch

            torch.cuda.synchronize(self.target_device)


def _unique_tensor_bytes(values: tuple[Any, ...]) -> int:
    seen: set[int] = set()
    total = 0
    for value in values:
        if id(value) in seen:
            continue
        seen.add(id(value))
        total += int(value.numel() * value.element_size())
    return total


def _swap_back(
    parameter_shadows: dict[str, Any],
    buffer_shadows: list[tuple[Any, Any]],
    parameters: dict[str, Any],
    *,
    torch: Any,
) -> None:
    for active_buffer, cpu_shadow in reversed(buffer_shadows):
        torch.utils.swap_tensors(active_buffer, cpu_shadow)
    for name in reversed(tuple(parameter_shadows)):
        torch.utils.swap_tensors(parameters[name], parameter_shadows[name])
    parameter_shadows.clear()
    buffer_shadows.clear()
