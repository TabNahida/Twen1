"""Minimal distributed setup with DDP and optional composable FSDP2."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: Any
    initialized_here: bool

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0


def initialize_distributed() -> DistributedContext:
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("real Twen training requires CUDA; use --dry-run for CPU validation")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    initialized_here = False
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        initialized_here = True
    return DistributedContext(rank, local_rank, world_size, torch.device("cuda", local_rank), initialized_here)


def finalize_distributed(context: DistributedContext, *, barrier: bool = True) -> None:
    import torch.distributed as dist

    if context.initialized_here and dist.is_initialized():
        if barrier:
            dist.barrier()
        dist.destroy_process_group()


def wrap_distributed(
    model: Any,
    context: DistributedContext,
    sharding: str,
    *,
    transformer_model: Any | None = None,
) -> Any:
    if context.world_size == 1:
        return model
    if sharding == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        return DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            find_unused_parameters=True,
        )
    if sharding == "fsdp2":
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        policy = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=None, output_dtype=None)
        layer_owner = model if transformer_model is None else transformer_model
        for layer in layer_owner.model.layers:
            transfer = getattr(getattr(layer, "mlp", None), "transfer_mlp", None)
            adapters = getattr(transfer, "adapters", None)
            if adapters is not None:
                # B is also called by the hidden-alignment loss after the parent
                # layer forward; manage it as its own FSDP unit and keep B
                # unsharded through backward. A can reshard after its one call.
                fully_shard(
                    adapters.input_adapter,
                    mp_policy=policy,
                    reshard_after_forward=True,
                )
                fully_shard(
                    adapters.output_adapter,
                    mp_policy=policy,
                    reshard_after_forward=False,
                )
            fully_shard(layer, mp_policy=policy, reshard_after_forward=True)
        fully_shard(model, mp_policy=policy, reshard_after_forward=True)
        return model
    raise ValueError(f"unknown sharding mode {sharding!r}")


def wrap_frozen_text_model(
    model: Any,
    context: DistributedContext,
    sharding: str,
) -> Any:
    """Place a no-grad hidden-state teacher without DDP gradient machinery."""

    if context.world_size == 1 or sharding == "ddp":
        return model.to(device=context.device)
    if sharding != "fsdp2":
        raise ValueError(f"unknown sharding mode {sharding!r}")
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    policy = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=None, output_dtype=None)
    for layer in model.layers:
        fully_shard(layer, mp_policy=policy, reshard_after_forward=True)
    fully_shard(model, mp_policy=policy, reshard_after_forward=True)
    return model


@contextlib.contextmanager
def accumulation_sync(model: Any, *, should_sync: bool, sharding: str) -> Iterator[None]:
    if sharding == "ddp" and hasattr(model, "no_sync") and not should_sync:
        with model.no_sync():
            yield
        return
    if sharding == "fsdp2" and hasattr(model, "set_requires_gradient_sync"):
        model.set_requires_gradient_sync(should_sync)
        try:
            yield
        finally:
            if not should_sync:
                model.set_requires_gradient_sync(True)
        return
    yield


def all_reduce_sum(value: Any, context: DistributedContext) -> Any:
    if context.world_size > 1:
        import torch.distributed as dist

        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def all_reduce_max(value: Any, context: DistributedContext) -> Any:
    if context.world_size > 1:
        import torch.distributed as dist

        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value
