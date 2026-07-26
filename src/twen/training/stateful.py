"""Small Stateful wrappers used by torch.distributed.checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .schedule import TokenCosineSchedule, TokenWarmupStableDecaySchedule


class TrainableModelState:
    """Checkpoint only trainable deltas; frozen source weights stay referenced by hash."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def _checkpoint_parameter_names(self, available_names: set[str]) -> set[str]:
        """Return one checkpoint FQN for every distinct trainable Parameter.

        ``StateDictOptions(ignore_frozen_params=True)`` is not sufficient for a
        tied frozen weight.  PyTorch currently drops one alias and can retain
        the other; for Qwen3.5 that caused the frozen 485 MiB tied LM head to be
        written into every delta checkpoint.  Buffers are also returned by
        ``get_model_state_dict`` even though Twen reconstructs them from the
        authenticated source/configuration artifacts.

        Enumerating aliases explicitly makes ``requires_grad`` the source of
        truth.  Selecting one available alias per Parameter also avoids writing
        a future trainable tied weight twice.
        """

        aliases_by_parameter: dict[int, list[str]] = {}
        for name, parameter in self.model.named_parameters(remove_duplicate=False):
            if parameter.requires_grad:
                aliases_by_parameter.setdefault(id(parameter), []).append(name)

        selected: set[str] = set()
        missing: list[tuple[str, ...]] = []
        for aliases in aliases_by_parameter.values():
            checkpoint_name = next(
                (name for name in aliases if name in available_names),
                None,
            )
            if checkpoint_name is None:
                missing.append(tuple(aliases))
            else:
                selected.add(checkpoint_name)
        if missing:
            raise RuntimeError(
                f"model state dict omitted trainable parameters; missing_aliases={missing[:3]}"
            )
        return selected

    def state_dict(self) -> dict[str, Any]:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        state = get_model_state_dict(
            self.model,
            options=StateDictOptions(
                full_state_dict=False,
                cpu_offload=False,
                ignore_frozen_params=True,
                strict=False,
            ),
        )
        selected = self._checkpoint_parameter_names(set(state))
        return {name: value for name, value in state.items() if name in selected}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        expected_entries = set(self.state_dict())
        supplied_entries = set(state_dict)
        missing = sorted(expected_entries - supplied_entries)
        extra = sorted(supplied_entries - expected_entries)
        if missing or extra:
            raise ValueError(
                "checkpoint model-delta entries differ from the production template; "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
        set_model_state_dict(
            self.model,
            dict(state_dict),
            options=StateDictOptions(
                full_state_dict=False,
                cpu_offload=False,
                ignore_frozen_params=True,
                strict=False,
            ),
        )


class OptimizerState:
    """FQN/DTensor optimizer state for exact DCP resharding."""

    def __init__(self, model: Any, optimizer: Any) -> None:
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> dict[str, Any]:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_optimizer_state_dict,
        )

        return get_optimizer_state_dict(
            self.model,
            self.optimizer,
            options=StateDictOptions(full_state_dict=False, cpu_offload=False),
        )

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_optimizer_state_dict,
        )

        set_optimizer_state_dict(
            self.model,
            self.optimizer,
            dict(state_dict),
            options=StateDictOptions(full_state_dict=False, cpu_offload=False),
        )


def materialize_adamw_state(optimizer: Any) -> None:
    """Create AdamW moment templates without performing an optimizer step."""

    import torch

    for group in optimizer.param_groups:
        if group.get("amsgrad", False):
            raise ValueError("Twen's resumable AdamW schema does not use AMSGrad")
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            if state:
                required = {"step", "exp_avg", "exp_avg_sq"}
                if set(state) != required:
                    raise ValueError("existing AdamW state does not match Twen's schema")
                continue
            step_device = (
                parameter.device
                if group.get("fused", False) or group.get("capturable", False)
                else "cpu"
            )
            state["step"] = torch.tensor(0.0, dtype=torch.float32, device=step_device)
            state["exp_avg"] = torch.zeros_like(parameter)
            state["exp_avg_sq"] = torch.zeros_like(parameter)


class TokenLRScheduler:
    """Serializable token-based LR scheduler independent of optimizer step count."""

    def __init__(
        self,
        optimizer: Any,
        *,
        warmup_tokens: int,
        max_tokens: int,
        lr_schedule: str = "cosine",
        min_lr_ratio: float = 0.1,
        decay_tokens: int | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.min_lr_ratio = float(min_lr_ratio)
        self.decay_tokens = decay_tokens
        if lr_schedule == "cosine":
            if decay_tokens is not None:
                raise ValueError("decay_tokens is only valid for warmup-stable-decay")
            self.schedule = TokenCosineSchedule(
                warmup_tokens,
                max_tokens,
                min_ratio=self.min_lr_ratio,
            )
        elif lr_schedule == "warmup-stable-decay":
            if decay_tokens is None:
                raise ValueError("warmup-stable-decay requires decay_tokens")
            self.schedule = TokenWarmupStableDecaySchedule(
                warmup_tokens,
                max_tokens,
                decay_tokens,
                min_ratio=self.min_lr_ratio,
            )
        else:
            raise ValueError(f"unsupported LR schedule: {lr_schedule!r}")
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.consumed_tokens = 0
        self._apply()

    def _apply(self) -> None:
        ratio = self.schedule.ratio(self.consumed_tokens)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base * ratio

    def step_tokens(self, consumed_tokens: int) -> None:
        if consumed_tokens < self.consumed_tokens:
            raise ValueError("scheduler token count cannot move backwards")
        self.consumed_tokens = int(consumed_tokens)
        self._apply()

    def state_dict(self) -> dict[str, Any]:
        value = {
            "consumed_tokens": self.consumed_tokens,
            "base_lrs": self.base_lrs,
            "warmup_tokens": self.schedule.warmup_tokens,
            "max_tokens": self.schedule.max_tokens,
        }
        # Do not alter the serialized shape of a legacy v1 scheduler.  New or
        # non-default schedules carry every resume-critical choice explicitly.
        if not (
            self.lr_schedule == "cosine" and self.min_lr_ratio == 0.1 and self.decay_tokens is None
        ):
            value["lr_schedule"] = self.lr_schedule
            value["min_lr_ratio"] = self.min_lr_ratio
            if self.decay_tokens is not None:
                value["decay_tokens"] = self.decay_tokens
        return value

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if int(state_dict["warmup_tokens"]) != self.schedule.warmup_tokens:
            raise ValueError("scheduler warmup_tokens changed across resume")
        if int(state_dict["max_tokens"]) != self.schedule.max_tokens:
            raise ValueError("scheduler max_tokens changed across resume")
        saved_schedule = str(state_dict.get("lr_schedule", "cosine"))
        saved_min_ratio = float(state_dict.get("min_lr_ratio", 0.1))
        raw_decay_tokens = state_dict.get("decay_tokens")
        saved_decay_tokens = None if raw_decay_tokens is None else int(raw_decay_tokens)
        if saved_schedule != self.lr_schedule:
            raise ValueError("scheduler lr_schedule changed across resume")
        if saved_min_ratio != self.min_lr_ratio:
            raise ValueError("scheduler min_lr_ratio changed across resume")
        if saved_decay_tokens != self.decay_tokens:
            raise ValueError("scheduler decay_tokens changed across resume")
        base = [float(value) for value in state_dict["base_lrs"]]
        if len(base) != len(self.optimizer.param_groups):
            raise ValueError("optimizer parameter-group count changed across resume")
        self.base_lrs = base
        self.consumed_tokens = int(state_dict["consumed_tokens"])
        self._apply()
