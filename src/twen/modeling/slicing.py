"""Aligned SwiGLU-channel scoring and capacity-constrained expert partitioning."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ._lazy import require_torch
from .errors import ShapeError


@dataclass(frozen=True, slots=True)
class ChannelPartition:
    """A complete, disjoint assignment of FFN channels to routed experts."""

    indices: tuple[tuple[int, ...], ...]
    contribution_totals: tuple[float, ...]
    num_channels: int
    expert_size: int
    strategy: str

    @property
    def num_experts(self) -> int:
        return len(self.indices)

    @property
    def max_mean_ratio(self) -> float:
        mean = sum(self.contribution_totals) / len(self.contribution_totals)
        return max(self.contribution_totals) / mean if mean > 0 else 1.0

    def as_tensor(self, *, device: str | Any | None = None) -> Any:
        torch = require_torch("channel partition tensor conversion")
        return torch.tensor(self.indices, dtype=torch.long, device=device)

    def to_dict(self) -> dict[str, Any]:
        return {
            "indices": [list(group) for group in self.indices],
            "contribution_totals": list(self.contribution_totals),
            "num_channels": self.num_channels,
            "num_experts": self.num_experts,
            "expert_size": self.expert_size,
            "strategy": self.strategy,
            "max_mean_ratio": self.max_mean_ratio,
        }


def _shape(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in value.shape)
    except (AttributeError, TypeError) as exc:
        raise TypeError("Projection weights must be tensor-like objects with shape") from exc


def channel_contribution_scores(
    gate_proj: Any,
    up_proj: Any,
    down_proj: Any,
    *,
    activation_rms: Any | None = None,
) -> Any:
    """Score aligned gate-row/up-row/down-column channel triples.

    The weight-only score is ``sqrt(||gate_i|| * ||up_i||) * ||down_:,i||``.
    If calibrated post-SwiGLU RMS is supplied, it additionally scales each score.
    Only relative scores are used by partitioning.
    """

    torch = require_torch("FFN channel contribution scoring")
    if not all(isinstance(x, torch.Tensor) for x in (gate_proj, up_proj, down_proj)):
        raise TypeError("gate_proj, up_proj and down_proj must be torch.Tensor instances")
    gate_shape, up_shape, down_shape = map(_shape, (gate_proj, up_proj, down_proj))
    if len(gate_shape) != 2 or len(up_shape) != 2 or len(down_shape) != 2:
        raise ShapeError("SwiGLU projection weights must all be rank-2")
    if gate_shape != up_shape:
        raise ShapeError(f"gate/up shapes differ: {gate_shape} vs {up_shape}")
    intermediate, hidden = gate_shape
    if down_shape != (hidden, intermediate):
        raise ShapeError(
            f"down shape {down_shape} must be [hidden, intermediate]={(hidden, intermediate)}"
        )
    with torch.no_grad():
        # Relative ranking does not benefit from FP64, while three 9B FFN
        # projections expanded to FP64 can exceed calibration memory budgets.
        gate = gate_proj.detach().to(dtype=torch.float32)
        up = up_proj.detach().to(device=gate.device, dtype=torch.float32)
        down = down_proj.detach().to(device=gate.device, dtype=torch.float32)
        gate_norm = torch.linalg.vector_norm(gate, dim=1)
        up_norm = torch.linalg.vector_norm(up, dim=1)
        down_norm = torch.linalg.vector_norm(down, dim=0)
        scores = torch.sqrt(gate_norm * up_norm) * down_norm
        if activation_rms is not None:
            if not isinstance(activation_rms, torch.Tensor):
                activation_rms = torch.as_tensor(activation_rms)
            if tuple(activation_rms.shape) != (intermediate,):
                raise ShapeError(
                    f"activation_rms must have shape {(intermediate,)}, "
                    f"got {tuple(activation_rms.shape)}"
                )
            activation = activation_rms.detach().to(device=gate.device, dtype=torch.float32)
            if bool((activation < 0).any()):
                raise ShapeError("activation_rms cannot contain negative values")
            scores = scores * activation
        if not bool(torch.isfinite(scores).all()):
            raise ShapeError("Channel contribution scores contain NaN or infinity")
        return scores


def _score_list(scores: Any) -> list[float]:
    if hasattr(scores, "detach") and hasattr(scores, "cpu"):
        scores = scores.detach().cpu().reshape(-1).tolist()
    try:
        result = [float(x) for x in scores]
    except (TypeError, ValueError) as exc:
        raise TypeError("scores must be a one-dimensional numeric sequence") from exc
    if not result:
        raise ValueError("scores cannot be empty")
    if any(not math.isfinite(x) or x < 0 for x in result):
        raise ValueError("scores must be finite and non-negative")
    return result


def _validate_partition(groups: Iterable[Iterable[int]], channels: int, size: int) -> None:
    normalized = [list(group) for group in groups]
    if any(len(group) != size for group in normalized):
        raise ShapeError("Every expert partition must have exactly expert_size channels")
    flat = [item for group in normalized for item in group]
    if sorted(flat) != list(range(channels)):
        raise ShapeError("Channel partition must cover every channel exactly once")


def build_channel_partition(
    gate_proj: Any | None = None,
    up_proj: Any | None = None,
    down_proj: Any | None = None,
    *,
    scores: Sequence[float] | Any | None = None,
    activation_rms: Any | None = None,
    num_experts: int = 8,
    expert_size: int = 1536,
    strategy: str = "greedy",
) -> ChannelPartition:
    """Build the deterministic 12288→8×1536 aligned-channel split.

    ``greedy`` assigns each next-highest contribution to the least-loaded expert
    that still has capacity. ``round_robin`` is retained as an ablation and uses
    a serpentine sorted assignment. Gate/up rows and corresponding down columns
    are represented by the same indices, so callers cannot accidentally unalign
    the SwiGLU triple.
    """

    if isinstance(num_experts, bool) or num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if isinstance(expert_size, bool) or expert_size <= 0:
        raise ValueError("expert_size must be positive")
    if scores is None:
        if gate_proj is None or up_proj is None or down_proj is None:
            raise ValueError("Provide scores or all three donor projection weights")
        score_values = _score_list(
            channel_contribution_scores(
                gate_proj, up_proj, down_proj, activation_rms=activation_rms
            )
        )
    else:
        if any(value is not None for value in (gate_proj, up_proj, down_proj, activation_rms)):
            raise ValueError("Explicit scores are mutually exclusive with projection inputs")
        score_values = _score_list(scores)
    channels = len(score_values)
    if channels != num_experts * expert_size:
        raise ShapeError(
            f"{channels} channels cannot form {num_experts} experts of size {expert_size}"
        )
    ordered = sorted(range(channels), key=lambda i: (-score_values[i], i))
    groups: list[list[int]] = [[] for _ in range(num_experts)]
    totals = [0.0] * num_experts
    if strategy == "greedy":
        for channel in ordered:
            candidates = [i for i, group in enumerate(groups) if len(group) < expert_size]
            expert = min(candidates, key=lambda i: (totals[i], len(groups[i]), i))
            groups[expert].append(channel)
            totals[expert] += score_values[channel]
    elif strategy == "round_robin":
        # Reverse direction every row to avoid consistently giving one expert the
        # strongest item within each group of num_experts channels.
        for offset in range(0, channels, num_experts):
            row = ordered[offset : offset + num_experts]
            if (offset // num_experts) % 2:
                row = list(reversed(row))
            for expert, channel in enumerate(row):
                groups[expert].append(channel)
                totals[expert] += score_values[channel]
    else:
        raise ValueError("strategy must be 'greedy' or 'round_robin'")

    # Original-index order improves contiguous gathers and makes artifacts stable.
    groups = [sorted(group) for group in groups]
    _validate_partition(groups, channels, expert_size)
    totals = [sum(score_values[channel] for channel in group) for group in groups]
    return ChannelPartition(
        indices=tuple(tuple(group) for group in groups),
        contribution_totals=tuple(totals),
        num_channels=channels,
        expert_size=expert_size,
        strategy=strategy,
    )
