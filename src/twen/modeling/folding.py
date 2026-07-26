"""Exact FP32 folding of transfer adapters into aligned donor expert slices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._lazy import require_torch
from .errors import ShapeError


@dataclass(frozen=True, slots=True)
class FoldedExperts:
    """Native Linear-orientation expert projections.

    Shapes are gate/up ``[experts, intermediate, hidden]`` and down
    ``[experts, hidden, intermediate]``.
    """

    gate_proj: Any
    up_proj: Any
    down_proj: Any
    channel_indices: Any

    @property
    def num_experts(self) -> int:
        return int(self.gate_proj.shape[0])

    @property
    def intermediate_size(self) -> int:
        return int(self.gate_proj.shape[1])

    @property
    def hidden_size(self) -> int:
        return int(self.gate_proj.shape[2])

    def to(self, *, dtype: Any | None = None, device: Any | None = None) -> FoldedExperts:
        options = {key: value for key, value in {"dtype": dtype, "device": device}.items() if value is not None}
        return FoldedExperts(
            gate_proj=self.gate_proj.to(**options),
            up_proj=self.up_proj.to(**options),
            down_proj=self.down_proj.to(**options),
            channel_indices=self.channel_indices.to(device=device)
            if device is not None
            else self.channel_indices,
        )


def _weight(value: Any, name: str) -> Any:
    torch = require_torch("adapter folding")
    result = getattr(value, "weight", value)
    if not isinstance(result, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor or module exposing .weight")
    if result.ndim != 2:
        raise ShapeError(f"{name} must be rank-2, got {tuple(result.shape)}")
    return result.detach()


def _indices(partition: Any, *, channels: int, device: Any) -> Any:
    torch = require_torch("adapter folding")
    raw = partition if isinstance(partition, torch.Tensor) else getattr(partition, "indices", partition)
    result = torch.as_tensor(raw, dtype=torch.long, device=device)
    if result.ndim != 2 or result.numel() != channels:
        raise ShapeError(
            f"partition must be [experts, expert_size] with {channels} total entries"
        )
    if not bool(
        torch.equal(
            torch.sort(result.reshape(-1)).values,
            torch.arange(channels, dtype=torch.long, device=device),
        )
    ):
        raise ShapeError("partition must cover every donor channel exactly once")
    return result


def fold_expert_weights(
    gate_proj: Any,
    up_proj: Any,
    down_proj: Any,
    input_adapter: Any,
    output_adapter: Any,
    channel_partition: Any,
    *,
    output_dtype: Any | None = None,
) -> FoldedExperts:
    """Fold ``A``/``B`` into donor slices with all matmuls performed in FP32.

    ``output_dtype`` is applied only after folding. Leaving it unset returns FP32
    artifacts, which is the required safe input to LoRA/LayerScale merging.
    """

    torch = require_torch("adapter folding")
    gate = _weight(gate_proj, "gate_proj")
    up = _weight(up_proj, "up_proj")
    down = _weight(down_proj, "down_proj")
    adapter_a = _weight(input_adapter, "input_adapter (A)")
    adapter_b = _weight(output_adapter, "output_adapter (B)")
    if tuple(gate.shape) != tuple(up.shape):
        raise ShapeError("gate_proj and up_proj must have identical shapes")
    channels, donor_hidden = map(int, gate.shape)
    if tuple(down.shape) != (donor_hidden, channels):
        raise ShapeError(
            f"down_proj must be {(donor_hidden, channels)}, got {tuple(down.shape)}"
        )
    if adapter_a.shape[0] != donor_hidden:
        raise ShapeError(
            f"A output width {adapter_a.shape[0]} must match donor hidden {donor_hidden}"
        )
    small_hidden = int(adapter_a.shape[1])
    if tuple(adapter_b.shape) != (small_hidden, donor_hidden):
        raise ShapeError(
            f"B must be {(small_hidden, donor_hidden)}, got {tuple(adapter_b.shape)}"
        )
    device = adapter_a.device
    partition = _indices(channel_partition, channels=channels, device=device)
    with torch.no_grad():
        # Moving explicitly also catches accidental mixed-device source tensors.
        gate32 = gate.to(device=device, dtype=torch.float32)
        up32 = up.to(device=device, dtype=torch.float32)
        down32 = down.to(device=device, dtype=torch.float32)
        a32 = adapter_a.to(device=device, dtype=torch.float32)
        b32 = adapter_b.to(device=device, dtype=torch.float32)
        folded_gate = []
        folded_up = []
        folded_down = []
        for index in partition:
            folded_gate.append(gate32.index_select(0, index).matmul(a32))
            folded_up.append(up32.index_select(0, index).matmul(a32))
            folded_down.append(b32.matmul(down32.index_select(1, index)))
        gate_out = torch.stack(folded_gate, dim=0)
        up_out = torch.stack(folded_up, dim=0)
        down_out = torch.stack(folded_down, dim=0)
        if output_dtype is not None:
            gate_out = gate_out.to(dtype=output_dtype)
            up_out = up_out.to(dtype=output_dtype)
            down_out = down_out.to(dtype=output_dtype)
    return FoldedExperts(
        gate_proj=gate_out,
        up_proj=up_out,
        down_proj=down_out,
        channel_indices=partition,
    )


def max_fold_error(
    hidden_states: Any,
    gate_proj: Any,
    up_proj: Any,
    down_proj: Any,
    input_adapter: Any,
    output_adapter: Any,
    folded: FoldedExperts,
) -> float:
    """Return maximum absolute dense-sum error for a calibration tensor."""

    torch = require_torch("fold equivalence checking")
    functional = torch.nn.functional
    a = _weight(input_adapter, "input_adapter").to(torch.float32)
    gate = _weight(gate_proj, "gate_proj").to(device=a.device, dtype=torch.float32)
    up = _weight(up_proj, "up_proj").to(device=a.device, dtype=torch.float32)
    down = _weight(down_proj, "down_proj").to(device=a.device, dtype=torch.float32)
    b = _weight(output_adapter, "output_adapter").to(
        device=a.device, dtype=torch.float32
    )
    x = hidden_states.detach().to(device=a.device, dtype=torch.float32)
    donor_x = functional.linear(x, a)
    reference = functional.linear(
        functional.silu(functional.linear(donor_x, gate))
        * functional.linear(donor_x, up),
        down,
    )
    reference = functional.linear(reference, b)
    actual = x.new_zeros(x.shape)
    for expert in range(folded.num_experts):
        gate_e = folded.gate_proj[expert].to(device=x.device, dtype=torch.float32)
        up_e = folded.up_proj[expert].to(device=x.device, dtype=torch.float32)
        down_e = folded.down_proj[expert].to(device=x.device, dtype=torch.float32)
        actual = actual + functional.linear(
            functional.silu(functional.linear(x, gate_e)) * functional.linear(x, up_e),
            down_e,
        )
    return float((reference - actual).abs().max().item())
