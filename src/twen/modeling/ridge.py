"""Streaming, bias-free bidirectional ridge initialization for A/B adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._lazy import require_torch
from .errors import ShapeError


@dataclass(frozen=True, slots=True)
class RidgeSolution:
    """Adapter weights in ``torch.nn.Linear`` orientation.

    ``input_adapter`` has shape ``[large_dim, small_dim]`` (A), and
    ``output_adapter`` has shape ``[small_dim, large_dim]`` (B).
    """

    input_adapter: Any
    output_adapter: Any
    sample_count: int
    l2: float


class BidirectionalRidgeStats:
    """Accumulate paired hidden-state sufficient statistics without saving samples."""

    schema_version = 1

    def __init__(
        self,
        small_dim: int,
        large_dim: int,
        *,
        device: str | Any = "cpu",
        dtype: Any | None = None,
    ) -> None:
        if small_dim <= 0 or large_dim <= 0:
            raise ValueError("small_dim and large_dim must be positive")
        torch = require_torch("ridge adapter statistics")
        if dtype is None:
            dtype = torch.float64
        if not getattr(dtype, "is_floating_point", False):
            raise TypeError("Ridge accumulation dtype must be floating point")
        self.small_dim = int(small_dim)
        self.large_dim = int(large_dim)
        self.device = torch.device(device)
        self.dtype = dtype
        self.sample_count = 0
        self.small_gram = torch.zeros(
            (self.small_dim, self.small_dim), device=self.device, dtype=dtype
        )
        self.large_gram = torch.zeros(
            (self.large_dim, self.large_dim), device=self.device, dtype=dtype
        )
        self.cross = torch.zeros(
            (self.small_dim, self.large_dim), device=self.device, dtype=dtype
        )

    def update(self, small: Any, large: Any) -> None:
        """Add one shard of corresponding pre-FFN hidden states."""

        torch = require_torch("ridge adapter statistics")
        if not isinstance(small, torch.Tensor) or not isinstance(large, torch.Tensor):
            raise TypeError("small and large activations must be torch.Tensor instances")
        if small.ndim < 2 or large.ndim < 2:
            raise ShapeError("Ridge activations must have at least two dimensions")
        if small.shape[:-1] != large.shape[:-1]:
            raise ShapeError(
                f"Paired activation sample shapes differ: {tuple(small.shape[:-1])} vs "
                f"{tuple(large.shape[:-1])}"
            )
        if small.shape[-1] != self.small_dim or large.shape[-1] != self.large_dim:
            raise ShapeError(
                "Ridge feature dimensions differ from initialized dimensions: "
                f"got ({small.shape[-1]}, {large.shape[-1]}), expected "
                f"({self.small_dim}, {self.large_dim})"
            )
        small_2d = small.detach().reshape(-1, self.small_dim)
        large_2d = large.detach().reshape(-1, self.large_dim)
        if small_2d.shape[0] == 0:
            return
        with torch.no_grad():
            small_2d = small_2d.to(device=self.device, dtype=self.dtype)
            large_2d = large_2d.to(device=self.device, dtype=self.dtype)
            if not bool(torch.isfinite(small_2d).all()) or not bool(
                torch.isfinite(large_2d).all()
            ):
                raise ShapeError("Ridge activations contain NaN or infinity")
            self.small_gram.add_(small_2d.transpose(0, 1).matmul(small_2d))
            self.large_gram.add_(large_2d.transpose(0, 1).matmul(large_2d))
            self.cross.add_(small_2d.transpose(0, 1).matmul(large_2d))
            self.sample_count += int(small_2d.shape[0])

    def merge_(self, other: BidirectionalRidgeStats) -> BidirectionalRidgeStats:
        """Merge independently accumulated calibration shards."""

        if (self.small_dim, self.large_dim) != (other.small_dim, other.large_dim):
            raise ShapeError("Cannot merge ridge stats with different feature dimensions")
        torch = require_torch("ridge adapter statistics")
        with torch.no_grad():
            self.small_gram.add_(other.small_gram.to(self.device, self.dtype))
            self.large_gram.add_(other.large_gram.to(self.device, self.dtype))
            self.cross.add_(other.cross.to(self.device, self.dtype))
            self.sample_count += other.sample_count
        return self

    def solve(self, *, l2: float = 1e-4, output_dtype: Any | None = None) -> RidgeSolution:
        """Solve small→large A and large→small B with normalized ridge loss.

        ``l2`` is applied to covariance matrices (``X.T @ X / n``), making it
        independent of calibration shard size. No intercept is fit because A/B
        must remain bias-free for exact inference-time folding.
        """

        if not isinstance(l2, (float, int)) or l2 < 0:
            raise ValueError(f"l2 must be a non-negative scalar, got {l2!r}")
        if self.sample_count <= 0:
            raise ShapeError("Cannot solve ridge adapters before accumulating samples")
        torch = require_torch("ridge adapter solve")
        n = float(self.sample_count)
        small_cov = self.small_gram / n
        large_cov = self.large_gram / n
        cross_cov = self.cross / n
        eye_small = torch.eye(self.small_dim, device=self.device, dtype=self.dtype)
        eye_large = torch.eye(self.large_dim, device=self.device, dtype=self.dtype)
        try:
            # Xs @ W ~= Xl, while Linear stores W.T.
            small_to_large = torch.linalg.solve(
                small_cov + float(l2) * eye_small, cross_cov
            )
            # Xl @ V ~= Xs. cross_cov.T is Xl.T @ Xs / n.
            large_to_small = torch.linalg.solve(
                large_cov + float(l2) * eye_large, cross_cov.transpose(0, 1)
            )
        except RuntimeError as exc:
            raise ShapeError(
                "Ridge solve failed; use positive l2 or inspect degenerate calibration activations"
            ) from exc
        input_adapter = small_to_large.transpose(0, 1).contiguous()
        output_adapter = large_to_small.transpose(0, 1).contiguous()
        if output_dtype is not None:
            input_adapter = input_adapter.to(dtype=output_dtype)
            output_adapter = output_adapter.to(dtype=output_dtype)
        return RidgeSolution(
            input_adapter=input_adapter,
            output_adapter=output_adapter,
            sample_count=self.sample_count,
            l2=float(l2),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "small_dim": self.small_dim,
            "large_dim": self.large_dim,
            "sample_count": self.sample_count,
            "small_gram": self.small_gram,
            "large_gram": self.large_gram,
            "cross": self.cross,
        }

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Any], *, device: str | Any = "cpu"
    ) -> BidirectionalRidgeStats:
        if state.get("schema_version") != cls.schema_version:
            raise ValueError(
                f"Unsupported ridge stats schema {state.get('schema_version')!r}; "
                f"expected {cls.schema_version}"
            )
        torch = require_torch("ridge adapter statistics")
        required = ("small_gram", "large_gram", "cross")
        if any(name not in state for name in required):
            raise ValueError("Ridge state is missing one or more sufficient-statistic tensors")
        if not all(isinstance(state[name], torch.Tensor) for name in required):
            raise TypeError("Ridge sufficient statistics must be torch.Tensor instances")
        small_dim = int(state["small_dim"])
        large_dim = int(state["large_dim"])
        expected_shapes = {
            "small_gram": (small_dim, small_dim),
            "large_gram": (large_dim, large_dim),
            "cross": (small_dim, large_dim),
        }
        malformed = {
            name: tuple(state[name].shape)
            for name, expected in expected_shapes.items()
            if tuple(state[name].shape) != expected
        }
        if malformed:
            raise ShapeError(f"Malformed ridge sufficient-statistic shapes: {malformed}")
        dtype = state["small_gram"].dtype
        result = cls(
            small_dim, large_dim, device=device, dtype=dtype
        )
        result.small_gram.copy_(state["small_gram"].to(device=result.device, dtype=dtype))
        result.large_gram.copy_(state["large_gram"].to(device=result.device, dtype=dtype))
        result.cross.copy_(state["cross"].to(device=result.device, dtype=dtype))
        count = int(state["sample_count"])
        if count < 0:
            raise ValueError("Ridge sample_count cannot be negative")
        result.sample_count = count
        return result
