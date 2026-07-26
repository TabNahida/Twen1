"""Linear CKA scoring and monotonic, same-layer-type matching."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ._lazy import require_torch
from .audit import normalize_layer_type
from .errors import LayerMappingError, ShapeError


@dataclass(frozen=True, slots=True)
class LayerPair:
    student_layer: int
    donor_layer: int
    layer_type: str
    cka: float


@dataclass(frozen=True, slots=True)
class LayerMatch:
    """Result of order-preserving student-to-donor layer assignment."""

    pairs: tuple[LayerPair, ...]
    total_cka: float
    mean_cka: float

    @property
    def student_to_donor(self) -> tuple[int, ...]:
        return tuple(pair.donor_layer for pair in self.pairs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "student_to_donor": list(self.student_to_donor),
            "total_cka": self.total_cka,
            "mean_cka": self.mean_cka,
            "pairs": [
                {
                    "student_layer": pair.student_layer,
                    "donor_layer": pair.donor_layer,
                    "layer_type": pair.layer_type,
                    "cka": pair.cka,
                }
                for pair in self.pairs
            ],
        }


def _flatten_samples(tensor: Any, name: str) -> Any:
    torch = require_torch("linear CKA")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim < 2:
        raise ShapeError(f"{name} must have at least [samples, features], got {tuple(tensor.shape)}")
    # Hidden-state tensors are conventionally [..., hidden]. Treat every leading
    # batch/sequence position as a corresponding calibration sample.
    return tensor.detach().reshape(-1, tensor.shape[-1])


def linear_cka(x: Any, y: Any, *, eps: float = 1e-12) -> float:
    """Compute centered linear CKA for two activation matrices.

    The last dimension contains features and all leading batch/sequence positions
    are corresponding calibration samples. Computation is detached and performed
    in float64 on the tensors' current device.
    """

    torch = require_torch("linear CKA")
    x2 = _flatten_samples(x, "x")
    y2 = _flatten_samples(y, "y")
    if x2.shape[0] != y2.shape[0]:
        raise ShapeError(f"CKA sample counts differ: {x2.shape[0]} != {y2.shape[0]}")
    if x2.shape[0] < 2:
        raise ShapeError("CKA requires at least two calibration samples")
    with torch.no_grad():
        x2 = x2.to(dtype=torch.float64)
        y2 = y2.to(device=x2.device, dtype=torch.float64)
        x2 = x2 - x2.mean(dim=0, keepdim=True)
        y2 = y2 - y2.mean(dim=0, keepdim=True)
        cross = x2.transpose(0, 1).matmul(y2)
        xx = x2.transpose(0, 1).matmul(x2)
        yy = y2.transpose(0, 1).matmul(y2)
        numerator = cross.square().sum()
        denominator = torch.sqrt(xx.square().sum() * yy.square().sum())
        if not bool(torch.isfinite(denominator)) or float(denominator.item()) <= eps:
            raise ShapeError("CKA is undefined for constant or non-finite activations")
        value = numerator / denominator.clamp_min(eps)
        return float(value.clamp(0.0, 1.0).item())


def linear_cka_matrix(
    student_activations: Sequence[Any], donor_activations: Sequence[Any]
) -> list[list[float]]:
    """Calculate every student/donor CKA pair without retaining autograd graphs."""

    if not student_activations or not donor_activations:
        raise LayerMappingError("Activation collections must be non-empty")
    return [
        [linear_cka(student, donor) for donor in donor_activations]
        for student in student_activations
    ]


def _scores_to_lists(scores: Any) -> list[list[float]]:
    if hasattr(scores, "detach") and hasattr(scores, "cpu"):
        scores = scores.detach().cpu().tolist()
    try:
        result = [[float(value) for value in row] for row in scores]
    except (TypeError, ValueError) as exc:
        raise LayerMappingError("cka_scores must be a rectangular numeric matrix") from exc
    if not result or not result[0]:
        raise LayerMappingError("cka_scores must be non-empty")
    width = len(result[0])
    if any(len(row) != width for row in result):
        raise LayerMappingError("cka_scores must be rectangular")
    if any(not math.isfinite(value) for row in result for value in row):
        raise LayerMappingError("cka_scores contains NaN or infinity")
    return result


def _better(
    candidate: tuple[float, tuple[int, ...]] | None,
    incumbent: tuple[float, tuple[int, ...]] | None,
) -> tuple[float, tuple[int, ...]] | None:
    if candidate is None:
        return incumbent
    if incumbent is None:
        return candidate
    if candidate[0] > incumbent[0] + 1e-15:
        return candidate
    if incumbent[0] > candidate[0] + 1e-15:
        return incumbent
    # Stable manifests: equal-score matches choose the lexicographically earliest
    # donor layer sequence rather than depending on loop/order implementation.
    return candidate if candidate[1] < incumbent[1] else incumbent


def match_layers_cka(
    student_activations: Sequence[Any] | None = None,
    donor_activations: Sequence[Any] | None = None,
    *,
    cka_scores: Any | None = None,
    student_layer_types: Sequence[str],
    donor_layer_types: Sequence[str],
) -> LayerMatch:
    """Find the maximum-CKA monotonic injection into donor layers.

    Supply either the two activation collections or a precomputed ``cka_scores``
    matrix. A student layer may only match a donor layer of the same normalized
    type, and donor indices are strictly increasing.
    """

    if cka_scores is None:
        if student_activations is None or donor_activations is None:
            raise LayerMappingError(
                "Provide cka_scores or both student_activations and donor_activations"
            )
        scores = linear_cka_matrix(student_activations, donor_activations)
    else:
        if student_activations is not None or donor_activations is not None:
            raise LayerMappingError(
                "cka_scores is mutually exclusive with activation collections"
            )
        scores = _scores_to_lists(cka_scores)

    n_student = len(scores)
    n_donor = len(scores[0])
    if n_student > n_donor:
        raise LayerMappingError(
            f"Cannot inject {n_student} student layers into {n_donor} donor layers"
        )
    if len(student_layer_types) != n_student or len(donor_layer_types) != n_donor:
        raise LayerMappingError(
            "Layer-type lengths must match the CKA matrix dimensions: "
            f"got ({len(student_layer_types)}, {len(donor_layer_types)}) vs "
            f"({n_student}, {n_donor})"
        )
    student_types = tuple(normalize_layer_type(x) for x in student_layer_types)
    donor_types = tuple(normalize_layer_type(x) for x in donor_layer_types)

    # dp[i][j] is the best assignment of the first i student layers using only
    # the first j donor layers. A state holds score and the full deterministic path.
    dp: list[list[tuple[float, tuple[int, ...]] | None]] = [
        [None] * (n_donor + 1) for _ in range(n_student + 1)
    ]
    for j in range(n_donor + 1):
        dp[0][j] = (0.0, ())
    for i in range(1, n_student + 1):
        for j in range(1, n_donor + 1):
            best = dp[i][j - 1]  # Skip donor j-1.
            previous = dp[i - 1][j - 1]
            if previous is not None and student_types[i - 1] == donor_types[j - 1]:
                matched = (
                    previous[0] + scores[i - 1][j - 1],
                    previous[1] + (j - 1,),
                )
                best = _better(matched, best)
            dp[i][j] = best

    answer = dp[n_student][n_donor]
    if answer is None:
        counts_student = {kind: student_types.count(kind) for kind in set(student_types)}
        counts_donor = {kind: donor_types.count(kind) for kind in set(donor_types)}
        raise LayerMappingError(
            "No strictly monotonic same-type layer mapping exists; "
            f"student counts={counts_student}, donor counts={counts_donor}"
        )
    total, path = answer
    pairs = tuple(
        LayerPair(
            student_layer=i,
            donor_layer=j,
            layer_type=student_types[i],
            cka=scores[i][j],
        )
        for i, j in enumerate(path)
    )
    return LayerMatch(pairs=pairs, total_cka=total, mean_cka=total / n_student)
