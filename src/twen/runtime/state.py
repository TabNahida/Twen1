"""Versioned runtime state used by Twen's resumable training loop.

This module deliberately has no hard dependency on NumPy or PyTorch at import
time.  A training process will normally have both installed, while small CPU
tools (checkpoint inspection, retention and recovery) should remain usable
without importing CUDA.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import pickle
import random
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

TRAINER_STATE_VERSION = 1
DATA_CURSOR_VERSION = 1
RNG_STATE_VERSION = 1
COMMITTED_BOUNDARY_VERSION = 1


class StateVersionError(ValueError):
    """Raised when runtime state is newer than this version of Twen."""


def _check_version(name: str, found: int, supported: int) -> None:
    if found < 1:
        raise StateVersionError(f"{name} version must be positive, got {found}")
    if found > supported:
        raise StateVersionError(
            f"{name} version {found} is newer than supported version {supported}"
        )


@dataclass(slots=True)
class DataCursor:
    """A globally meaningful, versioned input-data position.

    ``global_sample_index`` and ``global_token_index`` describe committed work.
    The shard-local fields make it possible for an iterable dataset to seek
    without replaying all earlier shards.  Dataset implementations may keep
    format-specific information in ``extra``.
    """

    shard_index: int = 0
    sample_index: int = 0
    token_offset: int = 0
    global_sample_index: int = 0
    global_token_index: int = 0
    epoch: int = 0
    shuffle_seed: int = 0
    shard_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    version: int = DATA_CURSOR_VERSION

    def __post_init__(self) -> None:
        _check_version("DataCursor", self.version, DATA_CURSOR_VERSION)
        for name in (
            "shard_index",
            "sample_index",
            "token_offset",
            "global_sample_index",
            "global_token_index",
            "epoch",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataCursor:
        data = dict(value)
        version = int(data.get("version", 1))
        _check_version("DataCursor", version, DATA_CURSOR_VERSION)
        data["version"] = version
        return cls(**data)

    def clone(self) -> DataCursor:
        return self.from_dict(self.to_dict())

    @classmethod
    def from_global_cursor_state(cls, value: Mapping[str, Any]) -> DataCursor:
        """Wrap ``DeterministicGlobalCursor.state_dict()`` without importing data code."""

        payload = copy.deepcopy(dict(value))
        return cls(
            global_sample_index=int(payload["next_global_sample"]),
            global_token_index=int(payload["committed_tokens"]),
            shuffle_seed=int(payload["seed"]),
            extra=payload,
        )

    def to_global_cursor_state(self) -> dict[str, Any]:
        """Return the exact data-layer cursor payload stored at this boundary."""

        payload: Mapping[str, Any] = self.extra
        # Accept the briefly-used nested representation for forward migration
        # of early development checkpoints.
        if isinstance(self.extra.get("global_cursor_state"), Mapping):
            payload = self.extra["global_cursor_state"]
        if "next_global_sample" not in payload:
            raise ValueError("DataCursor does not contain a global cursor state")
        result = copy.deepcopy(dict(payload))
        if int(result.get("next_global_sample", -1)) != self.global_sample_index:
            raise ValueError("global cursor sample position disagrees with DataCursor")
        if int(result.get("committed_tokens", -1)) != self.global_token_index:
            raise ValueError("global cursor token position disagrees with DataCursor")
        return result


@dataclass(slots=True)
class RNGState:
    """Python, NumPy, CPU Torch and per-CUDA-rank RNG state.

    The fields intentionally remain opaque Python objects.  They live in the
    hashed per-rank runtime payload rather than JSON metadata and are therefore
    losslessly serialised with pickle/``torch.save``.
    """

    python_state: object
    numpy_state: object | None = None
    torch_cpu_state: object | None = None
    torch_cuda_states: list[object] = field(default_factory=list)
    torch_cuda_devices: list[int] = field(default_factory=list)
    version: int = RNG_STATE_VERSION

    def __post_init__(self) -> None:
        _check_version("RNGState", self.version, RNG_STATE_VERSION)
        if len(self.torch_cuda_states) != len(self.torch_cuda_devices):
            raise ValueError("CUDA RNG states and device indices must have equal length")
        if any(device < 0 for device in self.torch_cuda_devices):
            raise ValueError("CUDA device indices must be non-negative")

    @classmethod
    def capture(cls) -> RNGState:
        numpy_state: object | None = None
        with suppress(ImportError):
            import numpy as np

            numpy_state = np.random.get_state()

        torch_cpu_state: object | None = None
        torch_cuda_states: list[object] = []
        torch_cuda_devices: list[int] = []
        with suppress(ImportError):
            import torch

            torch_cpu_state = torch.get_rng_state().clone()
            # Do not initialize CUDA merely to take a checkpoint.  Training has
            # already initialized the rank-local device before this function is
            # called.  Querying every visible device here would create N-by-N
            # CUDA contexts under torchrun, so each rank records only its
            # current device.
            if torch.cuda.is_initialized():
                device = torch.cuda.current_device()
                torch_cuda_devices = [device]
                torch_cuda_states = [torch.cuda.get_rng_state(device).clone()]

        return cls(
            python_state=copy.deepcopy(random.getstate()),
            numpy_state=copy.deepcopy(numpy_state),
            torch_cpu_state=torch_cpu_state,
            torch_cuda_states=torch_cuda_states,
            torch_cuda_devices=torch_cuda_devices,
        )

    def restore(self, *, strict_cuda: bool = False) -> None:
        random.setstate(copy.deepcopy(self.python_state))

        if self.numpy_state is not None:
            try:
                import numpy as np

                np.random.set_state(copy.deepcopy(self.numpy_state))
            except ImportError as exc:
                raise RuntimeError("checkpoint contains NumPy RNG state but NumPy is unavailable") from exc

        if self.torch_cpu_state is not None:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("checkpoint contains Torch RNG state but Torch is unavailable") from exc
            torch.set_rng_state(self.torch_cpu_state)
            if self.torch_cuda_states:
                if not torch.cuda.is_available():
                    if strict_cuda:
                        raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
                else:
                    current_devices = torch.cuda.device_count()
                    current_device = torch.cuda.current_device()
                    if strict_cuda:
                        if any(device >= current_devices for device in self.torch_cuda_devices):
                            raise RuntimeError("a checkpoint CUDA device is unavailable")
                        if (
                            len(self.torch_cuda_devices) == 1
                            and self.torch_cuda_devices[0] != current_device
                        ):
                            raise RuntimeError(
                                "rank-local CUDA device changed: checkpoint has "
                                f"{self.torch_cuda_devices[0]}, runtime has {current_device}"
                            )
                        pairs = list(
                            zip(self.torch_cuda_devices, self.torch_cuda_states, strict=True)
                        )
                    elif len(self.torch_cuda_states) == 1:
                        # World-size changes may move this logical rank to a
                        # different local device.  Preserve the stream while
                        # allowing the device ordinal to change.
                        pairs = [(current_device, self.torch_cuda_states[0])]
                    else:
                        pairs = [
                            (device, state)
                            for device, state in zip(
                                self.torch_cuda_devices,
                                self.torch_cuda_states,
                                strict=True,
                            )
                            if device < current_devices
                        ]
                    for device, state in pairs:
                        torch.cuda.set_rng_state(state, device=device)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "python_state": copy.deepcopy(self.python_state),
            "numpy_state": copy.deepcopy(self.numpy_state),
            "torch_cpu_state": (
                self.torch_cpu_state.clone()
                if hasattr(self.torch_cpu_state, "clone")
                else copy.deepcopy(self.torch_cpu_state)
            ),
            "torch_cuda_states": [
                state.clone() if hasattr(state, "clone") else copy.deepcopy(state)
                for state in self.torch_cuda_states
            ],
            "torch_cuda_devices": list(self.torch_cuda_devices),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RNGState:
        data = dict(value)
        version = int(data.get("version", 1))
        _check_version("RNGState", version, RNG_STATE_VERSION)
        data["version"] = version
        data.setdefault("torch_cuda_states", [])
        data.setdefault("torch_cuda_devices", list(range(len(data["torch_cuda_states"]))))
        return cls(**data)

    def clone(self) -> RNGState:
        return self.from_dict(self.to_dict())

    def digest(self) -> str:
        """Return a diagnostic digest without exposing the RNG state."""

        # Pickling a tensor includes storage bookkeeping that is not stable
        # across a save/load round trip.  Convert array/tensor payloads to an
        # explicit shape/dtype/bytes representation before hashing.
        canonical = _canonical_rng_value(self.to_dict())
        return hashlib.sha256(pickle.dumps(canonical, protocol=5)).hexdigest()

    def fork_for_rank(self, rank: int) -> RNGState:
        """Deterministically derive an independent stream for a newly-added rank.

        This does not mutate process-global RNG state.  It is used only when a
        checkpoint resumes with more ranks than it originally contained.
        """

        if rank < 0:
            raise ValueError("rank must be non-negative")
        material = f"twen-rng-rank-fork:{self.digest()}:{rank}".encode()
        seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
        python_state = random.Random(seed).getstate()

        numpy_state: object | None = None
        if self.numpy_state is not None:
            try:
                import numpy as np
            except ImportError as exc:
                raise RuntimeError("cannot fork NumPy RNG state without NumPy") from exc
            numpy_state = np.random.RandomState(seed % (2**32)).get_state()

        torch_cpu_state: object | None = None
        cuda_states: list[object] = []
        cuda_devices: list[int] = []
        if self.torch_cpu_state is not None:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("cannot fork Torch RNG state without Torch") from exc
            torch_seed = seed % ((2**63) - 1)
            cpu_generator = torch.Generator(device="cpu")
            cpu_generator.manual_seed(torch_seed)
            torch_cpu_state = cpu_generator.get_state()
            if self.torch_cuda_states and torch.cuda.is_initialized():
                device = torch.cuda.current_device()
                cuda_generator = torch.Generator(device=f"cuda:{device}")
                cuda_generator.manual_seed(torch_seed)
                cuda_states = [cuda_generator.get_state()]
                cuda_devices = [device]

        return RNGState(
            python_state=python_state,
            numpy_state=numpy_state,
            torch_cpu_state=torch_cpu_state,
            torch_cuda_states=cuda_states,
            torch_cuda_devices=cuda_devices,
        )


@dataclass(slots=True)
class TrainerState:
    """Small, JSON-compatible state committed at optimizer-step boundaries."""

    run_id: str
    stage: str
    global_step: int = 0
    committed_tokens: int = 0
    micro_step_in_accumulation: int = 0
    gradient_accumulation_steps: int = 1
    global_batch_tokens: int = 0
    micro_batch_tokens_per_rank: int = 0
    world_size: int = 1
    top_k: int | None = None
    curriculum_position: float = 0.0
    loss_weights: dict[str, float] = field(default_factory=dict)
    router_stats: dict[str, float] = field(default_factory=dict)
    dynamic_loss_scale: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    version: int = TRAINER_STATE_VERSION

    def __post_init__(self) -> None:
        _check_version("TrainerState", self.version, TRAINER_STATE_VERSION)
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if not self.stage:
            raise ValueError("stage must not be empty")
        for name in (
            "global_step",
            "committed_tokens",
            "micro_step_in_accumulation",
            "global_batch_tokens",
            "micro_batch_tokens_per_rank",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.world_size < 1:
            raise ValueError("world_size must be positive")
        if self.micro_step_in_accumulation > self.gradient_accumulation_steps:
            raise ValueError(
                "micro_step_in_accumulation cannot exceed "
                "gradient_accumulation_steps"
            )
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0.0 <= self.curriculum_position <= 1.0:
            raise ValueError("curriculum_position must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrainerState:
        data = dict(value)
        version = int(data.get("version", 1))
        _check_version("TrainerState", version, TRAINER_STATE_VERSION)
        data["version"] = version
        return cls(**data)

    def clone(self) -> TrainerState:
        return self.from_dict(self.to_dict())

    def accumulation_for(self, *, world_size: int, micro_batch_tokens_per_rank: int) -> int:
        """Compute resume accumulation while preserving global batch tokens."""

        if world_size < 1 or micro_batch_tokens_per_rank < 1:
            raise ValueError("world_size and micro_batch_tokens_per_rank must be positive")
        if self.global_batch_tokens < 1:
            raise ValueError("checkpoint does not define global_batch_tokens")
        denominator = world_size * micro_batch_tokens_per_rank
        accumulation, remainder = divmod(self.global_batch_tokens, denominator)
        if remainder or accumulation < 1:
            raise ValueError(
                f"global batch ({self.global_batch_tokens} tokens) cannot be represented by "
                f"world_size={world_size}, micro_batch_tokens_per_rank={micro_batch_tokens_per_rank}"
            )
        return accumulation


@dataclass(slots=True)
class CommittedBoundary:
    """Rollback point captured immediately before an optimizer step begins.

    Model parameters and optimizer state do not change during partial gradient
    accumulation.  Saving this boundary's cursor and RNG together with their
    unchanged stateful objects lets resume replay the entire interrupted step
    without skipping samples or applying a duplicate update.
    """

    trainer_state: TrainerState
    data_cursor: DataCursor
    rng_state: RNGState
    version: int = COMMITTED_BOUNDARY_VERSION

    def __post_init__(self) -> None:
        _check_version("CommittedBoundary", self.version, COMMITTED_BOUNDARY_VERSION)
        if self.trainer_state.micro_step_in_accumulation != 0:
            raise ValueError("a committed boundary must be captured at micro-step zero")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "trainer_state": self.trainer_state.to_dict(),
            "data_cursor": self.data_cursor.to_dict(),
            "rng_state": self.rng_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CommittedBoundary:
        version = int(value.get("version", 1))
        _check_version("CommittedBoundary", version, COMMITTED_BOUNDARY_VERSION)
        return cls(
            version=version,
            trainer_state=TrainerState.from_dict(value["trainer_state"]),
            data_cursor=DataCursor.from_dict(value["data_cursor"]),
            rng_state=RNGState.from_dict(value["rng_state"]),
        )

    def clone(self) -> CommittedBoundary:
        return self.from_dict(self.to_dict())


def capture_committed_boundary(
    trainer_state: TrainerState,
    data_cursor: DataCursor,
    rng_state: RNGState | None = None,
) -> CommittedBoundary:
    """Snapshot the exact start of the next optimizer step.

    Call this after the previous optimizer step is committed and before reading
    the first microbatch of the next step.
    """

    if trainer_state.micro_step_in_accumulation != 0:
        raise ValueError("capture_committed_boundary must run before gradient accumulation starts")
    return CommittedBoundary(
        trainer_state=trainer_state.clone(),
        data_cursor=data_cursor.clone(),
        rng_state=(rng_state or RNGState.capture()).clone(),
    )


def rollback_runtime_state(
    trainer_state: TrainerState,
    data_cursor: DataCursor,
    rng_state: RNGState,
    committed_boundary: CommittedBoundary | None,
) -> tuple[TrainerState, DataCursor, RNGState, bool]:
    """Select checkpoint runtime state, rolling back a partial accumulation."""

    if trainer_state.micro_step_in_accumulation == 0:
        return trainer_state.clone(), data_cursor.clone(), rng_state.clone(), False
    if committed_boundary is None:
        raise ValueError(
            "cannot checkpoint partial gradient accumulation without a committed boundary"
        )
    if committed_boundary.trainer_state.global_step != trainer_state.global_step:
        raise ValueError(
            "committed boundary belongs to a different optimizer step: "
            f"boundary={committed_boundary.trainer_state.global_step}, "
            f"current={trainer_state.global_step}"
        )
    return (
        committed_boundary.trainer_state.clone(),
        committed_boundary.data_cursor.clone(),
        committed_boundary.rng_state.clone(),
        True,
    )


def _canonical_rng_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            (key, _canonical_rng_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, tuple):
        return ("tuple", tuple(_canonical_rng_value(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_canonical_rng_value(item) for item in value))
    if hasattr(value, "detach") and hasattr(value, "dtype") and hasattr(value, "shape"):
        tensor = value.detach().cpu().contiguous().reshape(-1)
        return (
            "tensor",
            str(value.dtype),
            tuple(value.shape),
            bytes(tensor.tolist()),
        )
    if hasattr(value, "tobytes") and hasattr(value, "dtype") and hasattr(value, "shape"):
        return (
            "array",
            str(value.dtype),
            tuple(value.shape),
            value.tobytes(order="C"),
        )
    return value


__all__ = [
    "COMMITTED_BOUNDARY_VERSION",
    "DATA_CURSOR_VERSION",
    "RNG_STATE_VERSION",
    "TRAINER_STATE_VERSION",
    "CommittedBoundary",
    "DataCursor",
    "RNGState",
    "StateVersionError",
    "TrainerState",
    "capture_committed_boundary",
    "rollback_runtime_state",
]
