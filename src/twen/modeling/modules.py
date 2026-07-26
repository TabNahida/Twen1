"""Lazy PyTorch modules for dense transfer and folded sparse routing.

The public classes are created on first attribute access. Consequently importing
``twen.modeling`` for config/manifest work does not import torch or initialize a
CUDA runtime.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from functools import partial
from typing import Any

from ._lazy import require_torch
from .errors import ShapeError

__all__ = [
    "DenseTransferMLP",
    "MergeableExpertLoRA",
    "SharedDenseTransferMLP",
    "SparseTransferMLP",
    "TransferAdapters",
]

_CLASS_CACHE: dict[str, type[Any]] = {}


def _build_classes() -> dict[str, type[Any]]:
    if _CLASS_CACHE:
        return _CLASS_CACHE
    torch = require_torch("transfer model modules")
    nn = torch.nn
    functional = torch.nn.functional
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        create_selective_checkpoint_contexts,
    )
    from torch.utils.checkpoint import (
        checkpoint as activation_checkpoint,
    )

    def projection_weight(value: Any, name: str) -> Any:
        weight = getattr(value, "weight", value)
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"{name} must be a Tensor or module exposing .weight")
        if weight.ndim != 2:
            raise ShapeError(f"{name} must be rank-2, got shape {tuple(weight.shape)}")
        return weight

    def frozen_projection_parameter(weight: Any) -> Any:
        """Register a frozen projection without duplicating an existing Parameter.

        Dense Stage-B donor projections and the online hidden-alignment teacher
        come from the same pinned 9B checkpoint.  When the caller supplies an
        already-frozen teacher Parameter, keeping that exact object lets both
        module trees share its CUDA storage.  Plain tensors (and trainable
        Parameters from other callers) retain the previous defensive behavior:
        they are detached and wrapped as a new frozen Parameter.
        """

        if isinstance(weight, nn.Parameter) and not weight.requires_grad:
            return weight
        return nn.Parameter(weight.detach(), requires_grad=False)

    def partition_tensor(value: Any, *, channels: int, device: Any) -> Any:
        indices = value if isinstance(value, torch.Tensor) else getattr(value, "indices", value)
        result = torch.as_tensor(indices, dtype=torch.long, device=device)
        if result.ndim != 2:
            raise ShapeError(
                f"channel_partition must be [experts, expert_size], got {tuple(result.shape)}"
            )
        if result.numel() != channels:
            raise ShapeError(
                f"channel_partition contains {result.numel()} entries for {channels} channels"
            )
        ordered = torch.sort(result.reshape(-1)).values
        expected = torch.arange(channels, dtype=torch.long, device=device)
        if not bool(torch.equal(ordered, expected)):
            raise ShapeError("channel_partition must contain every donor channel exactly once")
        return result

    def validate_folded(gate: Any, up: Any, down: Any) -> tuple[int, int, int]:
        if not all(isinstance(x, torch.Tensor) for x in (gate, up, down)):
            raise TypeError("Folded expert projections must be torch.Tensor instances")
        if gate.ndim != 3 or up.ndim != 3 or down.ndim != 3:
            raise ShapeError("Folded expert projections must be rank-3")
        if tuple(gate.shape) != tuple(up.shape):
            raise ShapeError(f"Folded gate/up shapes differ: {gate.shape} vs {up.shape}")
        experts, intermediate, hidden = map(int, gate.shape)
        if tuple(down.shape) != (experts, hidden, intermediate):
            raise ShapeError(
                f"Folded down shape must be {(experts, hidden, intermediate)}, "
                f"got {tuple(down.shape)}"
            )
        return experts, intermediate, hidden

    def validate_topk(
        indices: Any,
        weights: Any,
        token_shape: tuple[int, ...],
        experts: int,
        *,
        check_range: bool = True,
    ) -> None:
        if not isinstance(indices, torch.Tensor) or indices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError("expert_indices must be an int32/int64 torch.Tensor")
        if tuple(indices.shape[:-1]) != token_shape or indices.shape[-1] <= 0:
            raise ShapeError(
                f"expert_indices must have shape {token_shape} + [top_k], "
                f"got {tuple(indices.shape)}"
            )
        # Range checks synchronize when ``indices`` is a CUDA tensor. They are
        # required for public caller-provided routing, but redundant for indices
        # returned by this module's own ``torch.topk`` call.
        if check_range and (bool((indices < 0).any()) or bool((indices >= experts).any())):
            raise ShapeError(f"expert_indices must be in [0, {experts})")
        if weights is not None and tuple(weights.shape) != tuple(indices.shape):
            raise ShapeError("expert_weights shape must equal expert_indices shape")

    class TransferAdapters(nn.Module):
        """Bias-free A/B maps that can be folded exactly into donor projections."""

        def __init__(
            self,
            small_hidden_size: int,
            donor_hidden_size: int,
            *,
            input_weight: Any | None = None,
            output_weight: Any | None = None,
            device: Any | None = None,
            dtype: Any | None = None,
        ) -> None:
            super().__init__()
            if small_hidden_size <= 0 or donor_hidden_size <= 0:
                raise ValueError("Adapter hidden sizes must be positive")
            self.small_hidden_size = int(small_hidden_size)
            self.donor_hidden_size = int(donor_hidden_size)
            self.input_adapter = nn.Linear(
                self.small_hidden_size,
                self.donor_hidden_size,
                bias=False,
                device=device,
                dtype=dtype,
            )
            self.output_adapter = nn.Linear(
                self.donor_hidden_size,
                self.small_hidden_size,
                bias=False,
                device=device,
                dtype=dtype,
            )
            if input_weight is None:
                nn.init.xavier_uniform_(self.input_adapter.weight)
            else:
                expected = (self.donor_hidden_size, self.small_hidden_size)
                if tuple(input_weight.shape) != expected:
                    raise ShapeError(
                        f"input_weight (A) must be {expected}, got {tuple(input_weight.shape)}"
                    )
                with torch.no_grad():
                    self.input_adapter.weight.copy_(input_weight)
            if output_weight is None:
                nn.init.xavier_uniform_(self.output_adapter.weight)
            else:
                expected = (self.small_hidden_size, self.donor_hidden_size)
                if tuple(output_weight.shape) != expected:
                    raise ShapeError(
                        f"output_weight (B) must be {expected}, got {tuple(output_weight.shape)}"
                    )
                with torch.no_grad():
                    self.output_adapter.weight.copy_(output_weight)

        @property
        def A(self) -> Any:
            return self.input_adapter.weight

        @property
        def B(self) -> Any:
            return self.output_adapter.weight

        def project_input(self, hidden_states: Any) -> Any:
            return self.input_adapter(hidden_states)

        def project_output(self, donor_states: Any) -> Any:
            return self.output_adapter(donor_states)

        def forward(self, hidden_states: Any) -> Any:
            return self.project_input(hidden_states)

    class DenseTransferMLP(nn.Module):
        """Frozen sliced donor SwiGLU plus trainable per-layer A/B and scale.

        With no top-k inputs, all expert slices are summed exactly. With
        ``expert_indices``/``expert_weights``, only the routed mixture is returned.
        ``return_expert_outputs`` returns ``(output, stacked_expert_outputs)``.
        """

        def __init__(
            self,
            gate_proj: Any,
            up_proj: Any,
            down_proj: Any,
            channel_partition: Any,
            *,
            adapters: Any | None = None,
            input_adapter: Any | None = None,
            output_adapter: Any | None = None,
            branch_scale: float = 1.0,
            trainable_scale: bool = True,
            execution_mode: str = "expanded",
            checkpoint_token_branch: bool = False,
        ) -> None:
            super().__init__()
            gate = projection_weight(gate_proj, "gate_proj")
            up = projection_weight(up_proj, "up_proj")
            down = projection_weight(down_proj, "down_proj")
            if tuple(gate.shape) != tuple(up.shape):
                raise ShapeError("Donor gate_proj and up_proj shapes must match")
            channels, donor_hidden = map(int, gate.shape)
            if tuple(down.shape) != (donor_hidden, channels):
                raise ShapeError(
                    f"Donor down_proj shape must be {(donor_hidden, channels)}, "
                    f"got {tuple(down.shape)}"
                )
            if gate.device != up.device or gate.device != down.device:
                raise ShapeError("Donor projections must start on the same device")
            partition = partition_tensor(
                channel_partition, channels=channels, device=gate.device
            )
            # Frozen Parameters (instead of buffers) let FSDP2 shard the 3.6B
            # donor-FFN values while trainable-only checkpointing still omits them.
            self.gate_weight = frozen_projection_parameter(gate)
            self.up_weight = frozen_projection_parameter(up)
            self.down_weight = frozen_projection_parameter(down)
            self.register_buffer("channel_indices", partition, persistent=True)
            self.num_experts = int(partition.shape[0])
            self.expert_size = int(partition.shape[1])
            if adapters is None:
                if input_adapter is not None:
                    small_hidden = int(input_adapter.shape[1])
                elif output_adapter is not None:
                    small_hidden = int(output_adapter.shape[0])
                else:
                    raise ValueError(
                        "Provide TransferAdapters or initialized input_adapter/output_adapter weights"
                    )
                adapters = TransferAdapters(
                    small_hidden,
                    donor_hidden,
                    input_weight=input_adapter,
                    output_weight=output_adapter,
                    device=gate.device,
                    dtype=gate.dtype,
                )
            if not isinstance(adapters, TransferAdapters):
                raise TypeError("adapters must be a TransferAdapters instance")
            if adapters.donor_hidden_size != donor_hidden:
                raise ShapeError("Adapter donor width does not match donor FFN hidden width")
            self.adapters = adapters
            self.branch_scale = nn.Parameter(
                torch.tensor([float(branch_scale)], device=gate.device, dtype=torch.float32),
                requires_grad=trainable_scale,
            )
            self.execution_mode = "expanded"
            self.checkpoint_token_branch = False
            self.configure_execution(
                execution_mode=execution_mode,
                checkpoint_token_branch=checkpoint_token_branch,
            )
            self.last_aux: dict[str, Any] | None = None
            self.record_aux_enabled = False

        @property
        def small_hidden_size(self) -> int:
            return self.adapters.small_hidden_size

        def _expert(self, donor_hidden: Any, expert: int) -> Any:
            index = self.channel_indices[expert]
            gate = self.gate_weight.index_select(0, index)
            up = self.up_weight.index_select(0, index)
            down = self.down_weight.index_select(1, index)
            activated = functional.silu(functional.linear(donor_hidden, gate))
            activated = activated * functional.linear(donor_hidden, up)
            return self.adapters.project_output(functional.linear(activated, down))

        def _all_experts(self, donor_hidden: Any) -> Any:
            return torch.stack(
                [self._expert(donor_hidden, expert) for expert in range(self.num_experts)],
                dim=-2,
            )

        def _dense_full(self, donor_hidden: Any) -> Any:
            """Evaluate the complete donor FFN without materializing expert slices.

            The channel partition is a permutation of the full intermediate
            dimension. Summing every sliced donor output and then applying the
            linear output adapter is therefore algebraically identical to one
            full gate/up/down evaluation followed by one adapter evaluation.
            """

            activated = functional.silu(functional.linear(donor_hidden, self.gate_weight))
            activated = activated * functional.linear(donor_hidden, self.up_weight)
            donor_output = functional.linear(activated, self.down_weight)
            return self.adapters.project_output(donor_output)

        @staticmethod
        def _expanded_token_core(
            hidden_states: Any,
            adapter_a: Any,
            adapter_b: Any,
            gate_weight: Any,
            up_weight: Any,
            down_weight: Any,
        ) -> Any:
            donor_hidden = functional.linear(hidden_states, adapter_a)
            activated = functional.silu(functional.linear(donor_hidden, gate_weight))
            activated = activated * functional.linear(donor_hidden, up_weight)
            donor_output = functional.linear(activated, down_weight)
            return functional.linear(donor_output, adapter_b)

        def _expanded_checkpoint_contexts(self) -> tuple[Any, Any]:
            """Cache only the donor down-projection output during branch replay.

            Gate/up outputs dominate expanded-path activation memory, so they
            are recomputed.  The down projection is one of the three large
            donor GEMMs and its output is much smaller; caching that exact
            result avoids repeating the down GEMM while retaining substantially
            less memory than the two intermediate-width gate/up tensors.
            """

            intermediate = int(self.gate_weight.shape[0])
            donor_hidden = int(self.gate_weight.shape[1])

            def policy(_context: Any, operation: Any, *args: Any, **_kwargs: Any) -> Any:
                if operation == torch.ops.aten.mm.default and len(args) >= 2:
                    left, right = args[:2]
                    if (
                        isinstance(left, torch.Tensor)
                        and isinstance(right, torch.Tensor)
                        and left.ndim == 2
                        and right.ndim == 2
                        and int(left.shape[-1]) == intermediate
                        and tuple(right.shape) == (intermediate, donor_hidden)
                    ):
                        return CheckpointPolicy.MUST_SAVE
                return CheckpointPolicy.PREFER_RECOMPUTE

            return create_selective_checkpoint_contexts(policy)

        def _checkpointed_expanded_dense(self, hidden_states: Any) -> Any:
            return activation_checkpoint(
                self._expanded_token_core,
                hidden_states,
                self.adapters.A,
                self.adapters.B,
                self.gate_weight,
                self.up_weight,
                self.down_weight,
                use_reentrant=False,
                context_fn=partial(self._expanded_checkpoint_contexts),
            )

        @staticmethod
        def _folded_token_core(
            hidden_states: Any,
            folded_gate: Any,
            folded_up: Any,
            folded_down: Any,
        ) -> Any:
            """Evaluate tokens using differentiably folded dense projections."""

            activated = functional.silu(functional.linear(hidden_states, folded_gate))
            activated = activated * functional.linear(hidden_states, folded_up)
            return functional.linear(activated, folded_down)

        def _differentiable_folded_dense(self, hidden_states: Any) -> Any:
            """Fold A/B once, outside the optional token-only checkpoint.

            Keeping the three fold GEMMs outside the non-reentrant checkpoint is
            important: backward then recomputes only the token-dependent SwiGLU
            branch, rather than repeating the large, token-independent G@A,
            U@A and B@D products.  The folded matrices are explicit checkpoint
            inputs so gradients still reach both adapters.
            """

            folded_gate = self.gate_weight.matmul(self.adapters.A)
            folded_up = self.up_weight.matmul(self.adapters.A)
            folded_down = self.adapters.B.matmul(self.down_weight)
            if self.checkpoint_token_branch and torch.is_grad_enabled():
                return activation_checkpoint(
                    self._folded_token_core,
                    hidden_states,
                    folded_gate,
                    folded_up,
                    folded_down,
                    use_reentrant=False,
                )
            return self._folded_token_core(
                hidden_states,
                folded_gate,
                folded_up,
                folded_down,
            )

        def configure_execution(
            self,
            *,
            execution_mode: str | None = None,
            checkpoint_token_branch: bool | None = None,
        ) -> None:
            """Select the dense branch implementation and token checkpoint policy."""

            mode = self.execution_mode if execution_mode is None else execution_mode
            if mode not in {"expanded", "differentiable_folded"}:
                raise ValueError(
                    "execution_mode must be 'expanded' or 'differentiable_folded'"
                )
            checkpoint_enabled = (
                self.checkpoint_token_branch
                if checkpoint_token_branch is None
                else bool(checkpoint_token_branch)
            )
            self.execution_mode = mode
            self.checkpoint_token_branch = checkpoint_enabled

        def _sparse_route(self, donor_hidden: Any, indices: Any, weights: Any) -> Any:
            token_shape = tuple(donor_hidden.shape[:-1])
            flat_hidden = donor_hidden.reshape(-1, donor_hidden.shape[-1])
            flat_indices = indices.reshape(-1, indices.shape[-1])
            flat_weights = weights.reshape_as(flat_indices).to(dtype=donor_hidden.dtype)
            result = donor_hidden.new_zeros((flat_hidden.shape[0], self.small_hidden_size))
            for expert in range(self.num_experts):
                routing_weight = (flat_weights * (flat_indices == expert)).sum(dim=-1)
                selected = torch.nonzero(routing_weight != 0, as_tuple=False).flatten()
                if selected.numel() == 0:
                    continue
                values = self._expert(flat_hidden.index_select(0, selected), expert)
                result.index_add_(
                    0,
                    selected,
                    values * routing_weight.index_select(0, selected).unsqueeze(-1),
                )
            return result.reshape((*token_shape, self.small_hidden_size))

        def set_record_aux(self, enabled: bool) -> None:
            """Enable/disable aux capture for calls made by an unmodified HF decoder."""

            self.record_aux_enabled = bool(enabled)

        @contextmanager
        def aux_recording(self, enabled: bool = True) -> Any:
            """Temporarily capture router/expert diagnostics without changing HF calls."""

            previous = self.record_aux_enabled
            self.record_aux_enabled = bool(enabled)
            try:
                yield self
            finally:
                self.record_aux_enabled = previous

        def forward(
            self,
            hidden_states: Any,
            *,
            expert_indices: Any | None = None,
            expert_weights: Any | None = None,
            return_expert_outputs: bool = False,
            record_aux: bool | None = None,
        ) -> Any:
            should_record = self.record_aux_enabled if record_aux is None else bool(record_aux)
            if hidden_states.shape[-1] != self.small_hidden_size:
                raise ShapeError(
                    f"Expected hidden width {self.small_hidden_size}, got {hidden_states.shape[-1]}"
                )
            token_shape = tuple(hidden_states.shape[:-1])
            if expert_indices is None:
                if expert_weights is not None:
                    raise ValueError("expert_weights requires expert_indices")
                if should_record or return_expert_outputs:
                    donor_hidden = self.adapters.project_input(hidden_states)
                    all_outputs = self._all_experts(donor_hidden)
                    output = all_outputs.sum(dim=-2)
                elif self.execution_mode == "differentiable_folded":
                    all_outputs = None
                    output = self._differentiable_folded_dense(hidden_states)
                elif self.checkpoint_token_branch and torch.is_grad_enabled():
                    all_outputs = None
                    output = self._checkpointed_expanded_dense(hidden_states)
                else:
                    donor_hidden = self.adapters.project_input(hidden_states)
                    all_outputs = None
                    output = self._dense_full(donor_hidden)
            else:
                donor_hidden = self.adapters.project_input(hidden_states)
                validate_topk(expert_indices, expert_weights, token_shape, self.num_experts)
                if expert_weights is None:
                    expert_weights = torch.full(
                        expert_indices.shape,
                        1.0 / expert_indices.shape[-1],
                        device=expert_indices.device,
                        dtype=hidden_states.dtype,
                    )
                if should_record or return_expert_outputs:
                    all_outputs = self._all_experts(donor_hidden)
                    gather_index = expert_indices.unsqueeze(-1).expand(
                        (*expert_indices.shape, self.small_hidden_size)
                    )
                    chosen = torch.gather(all_outputs, -2, gather_index)
                    output = (chosen * expert_weights.unsqueeze(-1)).sum(dim=-2)
                else:
                    all_outputs = None
                    output = self._sparse_route(donor_hidden, expert_indices, expert_weights)
            scale = self.branch_scale.to(dtype=output.dtype)
            output = output * scale
            if all_outputs is not None:
                all_outputs = all_outputs * scale
            if should_record:
                assert all_outputs is not None
                self.last_aux = {
                    "expert_indices": expert_indices,
                    "expert_weights": expert_weights,
                    "expert_outputs": all_outputs,
                    "dense_sum": all_outputs.sum(dim=-2),
                    "routed_output": output,
                }
            else:
                self.last_aux = None
            if return_expert_outputs:
                assert all_outputs is not None
                return output, all_outputs
            return output

    class SharedDenseTransferMLP(nn.Module):
        """Drop-in HF MLP replacement: frozen 0.8B shared FFN + dense donor branch."""

        def __init__(
            self,
            shared_mlp: Any,
            gate_proj: Any,
            up_proj: Any,
            down_proj: Any,
            channel_partition: Any,
            *,
            adapters: Any | None = None,
            input_adapter: Any | None = None,
            output_adapter: Any | None = None,
            branch_scale: float = 0.0,
            trainable_scale: bool = True,
            freeze_shared: bool = True,
            execution_mode: str = "expanded",
            checkpoint_token_branch: bool = False,
        ) -> None:
            super().__init__()
            if not isinstance(shared_mlp, nn.Module):
                raise TypeError("shared_mlp must be a torch.nn.Module")
            self.shared_mlp = shared_mlp
            if freeze_shared:
                self.shared_mlp.requires_grad_(False)
            self.transfer_mlp = DenseTransferMLP(
                gate_proj,
                up_proj,
                down_proj,
                channel_partition,
                adapters=adapters,
                input_adapter=input_adapter,
                output_adapter=output_adapter,
                branch_scale=branch_scale,
                trainable_scale=trainable_scale,
                execution_mode=execution_mode,
                checkpoint_token_branch=checkpoint_token_branch,
            )
            self.transfer_enabled = True

        @property
        def last_aux(self) -> dict[str, Any] | None:
            return self.transfer_mlp.last_aux

        @property
        def record_aux_enabled(self) -> bool:
            return self.transfer_mlp.record_aux_enabled

        def set_record_aux(self, enabled: bool) -> None:
            self.transfer_mlp.set_record_aux(enabled)

        def clear_aux(self) -> None:
            self.transfer_mlp.last_aux = None

        def configure_transfer_execution(
            self,
            *,
            execution_mode: str | None = None,
            checkpoint_token_branch: bool | None = None,
        ) -> None:
            self.transfer_mlp.configure_execution(
                execution_mode=execution_mode,
                checkpoint_token_branch=checkpoint_token_branch,
            )

        def set_transfer_enabled(self, enabled: bool) -> None:
            """Toggle the donor branch; disabled mode is an exact shared-only path."""

            self.transfer_enabled = bool(enabled)

        @contextmanager
        def transfer_mode(self, enabled: bool) -> Any:
            previous = self.transfer_enabled
            self.transfer_enabled = bool(enabled)
            try:
                yield self
            finally:
                self.transfer_enabled = previous

        @contextmanager
        def shared_only(self) -> Any:
            with self.transfer_mode(False):
                yield self

        @contextmanager
        def aux_recording(self, enabled: bool = True) -> Any:
            with self.transfer_mlp.aux_recording(enabled):
                yield self

        def forward(
            self,
            hidden_states: Any,
            *,
            expert_indices: Any | None = None,
            expert_weights: Any | None = None,
            return_expert_outputs: bool = False,
            record_aux: bool | None = None,
        ) -> Any:
            shared = self.shared_mlp(hidden_states)
            if not self.transfer_enabled:
                # Preserve diagnostics from the preceding routed pass so an
                # anchor-KL shared-only pass cannot erase router/expert losses.
                return shared
            transfer = self.transfer_mlp(
                hidden_states,
                expert_indices=expert_indices,
                expert_weights=expert_weights,
                return_expert_outputs=return_expert_outputs,
                record_aux=record_aux,
            )
            if return_expert_outputs:
                routed, experts = transfer
                return shared + routed, experts
            return shared + transfer

    class MergeableExpertLoRA(nn.Module):
        """Frozen folded expert bank with rank-16-by-default trainable LoRA deltas."""

        def __init__(
            self,
            gate_proj: Any,
            up_proj: Any,
            down_proj: Any,
            *,
            rank: int = 16,
            alpha: float = 16.0,
            dropout: float = 0.0,
            trainable_dtype: Any | None = None,
        ) -> None:
            super().__init__()
            experts, intermediate, hidden = validate_folded(gate_proj, up_proj, down_proj)
            if isinstance(rank, bool) or rank <= 0:
                raise ValueError("LoRA rank must be positive")
            if alpha <= 0:
                raise ValueError("LoRA alpha must be positive")
            if not 0.0 <= dropout < 1.0:
                raise ValueError("LoRA dropout must be in [0, 1)")
            self.num_experts = experts
            self.intermediate_size = intermediate
            self.hidden_size = hidden
            self.rank = int(rank)
            self.alpha = float(alpha)
            self.scaling = self.alpha / self.rank
            self.dropout = nn.Dropout(float(dropout))
            self.gate_proj = nn.Parameter(gate_proj.detach(), requires_grad=False)
            self.up_proj = nn.Parameter(up_proj.detach(), requires_grad=False)
            self.down_proj = nn.Parameter(down_proj.detach(), requires_grad=False)
            options = {
                "device": gate_proj.device,
                "dtype": trainable_dtype or gate_proj.dtype,
            }
            self.gate_lora_a = nn.Parameter(
                torch.empty((experts, rank, hidden), **options)
            )
            self.gate_lora_b = nn.Parameter(
                torch.zeros((experts, intermediate, rank), **options)
            )
            self.up_lora_a = nn.Parameter(torch.empty((experts, rank, hidden), **options))
            self.up_lora_b = nn.Parameter(
                torch.zeros((experts, intermediate, rank), **options)
            )
            self.down_lora_a = nn.Parameter(
                torch.empty((experts, rank, intermediate), **options)
            )
            self.down_lora_b = nn.Parameter(torch.zeros((experts, hidden, rank), **options))
            for parameter in (self.gate_lora_a, self.up_lora_a, self.down_lora_a):
                nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
            self.register_buffer(
                "_merged_flag", torch.tensor(False, device=gate_proj.device), persistent=True
            )
            # Keep the persistent tensor for checkpoint compatibility, but never
            # read it from the forward hot path: Tensor.item() synchronizes CUDA.
            self._merged = False

        @property
        def merged(self) -> bool:
            return self._merged

        def _load_from_state_dict(
            self,
            state_dict: Any,
            prefix: str,
            local_metadata: Any,
            strict: bool,
            missing_keys: list[str],
            unexpected_keys: list[str],
            error_msgs: list[str],
        ) -> None:
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            # A state load may replace or copy the persistent flag. Synchronize
            # the Python hot-path cache once at that explicit state boundary.
            if not self._merged_flag.is_meta:
                self._merged = bool(self._merged_flag.item())

        def _linear_lora(self, x: Any, base: Any, a: Any, b: Any) -> Any:
            result = functional.linear(x, base)
            if not self.merged:
                dropped = self.dropout(x)
                result = result + functional.linear(functional.linear(dropped, a), b) * self.scaling
            return result

        def _expert(self, hidden: Any, expert: int) -> Any:
            gate = self._linear_lora(
                hidden,
                self.gate_proj[expert],
                self.gate_lora_a[expert],
                self.gate_lora_b[expert],
            )
            up = self._linear_lora(
                hidden,
                self.up_proj[expert],
                self.up_lora_a[expert],
                self.up_lora_b[expert],
            )
            activated = functional.silu(gate) * up
            return self._linear_lora(
                activated,
                self.down_proj[expert],
                self.down_lora_a[expert],
                self.down_lora_b[expert],
            )

        def all_expert_outputs(self, hidden_states: Any) -> Any:
            return torch.stack(
                [self._expert(hidden_states, e) for e in range(self.num_experts)], dim=-2
            )

        def _all_expert_input_linear(self, hidden: Any, base: Any, a: Any, b: Any) -> Any:
            """Apply every expert projection to one shared token matrix.

            ``hidden`` is ``[tokens, in]`` while the weights are
            ``[experts, out, in]``.  Keeping the expert dimension inside the
            contractions avoids one Python dispatch per expert.  When LoRA
            dropout is enabled, expanding before dropout preserves independent
            masks for each expert, matching the scalar expert implementation.
            """

            expert_hidden = hidden.unsqueeze(0).expand(self.num_experts, -1, -1)
            result = torch.bmm(expert_hidden, base.transpose(1, 2))
            if self.merged:
                return result.transpose(0, 1)
            dropped = self.dropout(expert_hidden)
            low_rank = torch.bmm(dropped, a.transpose(1, 2))
            result = result + torch.bmm(low_rank, b.transpose(1, 2)) * self.scaling
            return result.transpose(0, 1)

        def _all_expert_batched_linear(self, hidden: Any, base: Any, a: Any, b: Any) -> Any:
            """Apply expert-local projections to ``[tokens, experts, in]``."""

            expert_hidden = hidden.transpose(0, 1)
            result = torch.bmm(expert_hidden, base.transpose(1, 2))
            if self.merged:
                return result.transpose(0, 1)
            dropped = self.dropout(expert_hidden)
            low_rank = torch.bmm(dropped, a.transpose(1, 2))
            result = result + torch.bmm(low_rank, b.transpose(1, 2)) * self.scaling
            return result.transpose(0, 1)

        def _all_expert_outputs_vectorized(self, hidden_states: Any) -> Any:
            """Evaluate the full expert bank without dynamic CUDA indexing."""

            token_shape = tuple(hidden_states.shape[:-1])
            flat_hidden = hidden_states.reshape(-1, self.hidden_size)
            gate = self._all_expert_input_linear(
                flat_hidden,
                self.gate_proj,
                self.gate_lora_a,
                self.gate_lora_b,
            )
            up = self._all_expert_input_linear(
                flat_hidden,
                self.up_proj,
                self.up_lora_a,
                self.up_lora_b,
            )
            activated = functional.silu(gate) * up
            output = self._all_expert_batched_linear(
                activated,
                self.down_proj,
                self.down_lora_a,
                self.down_lora_b,
            )
            # Keep any FP32 promotion from the LoRA residual through routing;
            # ``forward`` restores the public hidden-state/autocast dtype only
            # after the weighted expert accumulation.
            return output.reshape((*token_shape, self.num_experts, self.hidden_size))

        def forward(
            self,
            hidden_states: Any,
            *,
            expert_indices: Any | None = None,
            expert_weights: Any | None = None,
            return_expert_outputs: bool = False,
            _indices_prevalidated: bool = False,
        ) -> Any:
            if hidden_states.shape[-1] != self.hidden_size:
                raise ShapeError(
                    f"Expected hidden width {self.hidden_size}, got {hidden_states.shape[-1]}"
                )
            token_shape = tuple(hidden_states.shape[:-1])
            full_expert_route = (
                isinstance(expert_indices, torch.Tensor)
                and expert_indices.ndim > 0
                and expert_indices.shape[-1] == self.num_experts
            )
            if expert_indices is None or return_expert_outputs or full_expert_route:
                all_outputs = (
                    self._all_expert_outputs_vectorized(hidden_states)
                    if full_expert_route
                    else self.all_expert_outputs(hidden_states)
                )
                if expert_indices is None:
                    output = all_outputs.sum(dim=-2)
                else:
                    validate_topk(
                        expert_indices,
                        expert_weights,
                        token_shape,
                        self.num_experts,
                        check_range=not _indices_prevalidated,
                    )
                    if expert_weights is None:
                        expert_weights = torch.full(
                            expert_indices.shape,
                            1.0 / expert_indices.shape[-1],
                            device=expert_indices.device,
                            dtype=hidden_states.dtype,
                        )
                    gather_index = expert_indices.unsqueeze(-1).expand(
                        (*expert_indices.shape, self.hidden_size)
                    )
                    selected = torch.gather(all_outputs, -2, gather_index)
                    output = (selected * expert_weights.unsqueeze(-1)).sum(dim=-2)
                # Whether this is top-E, an aux-recording pass, or a dense sum,
                # retain promoted precision through the reduction and restore
                # the public autocast/hidden dtype only at the boundary.
                output = output.to(dtype=hidden_states.dtype)
                all_outputs = all_outputs.to(dtype=hidden_states.dtype)
                return (output, all_outputs) if return_expert_outputs else output

            validate_topk(
                expert_indices,
                expert_weights,
                token_shape,
                self.num_experts,
                check_range=not _indices_prevalidated,
            )
            if expert_weights is None:
                expert_weights = torch.full(
                    expert_indices.shape,
                    1.0 / expert_indices.shape[-1],
                    device=expert_indices.device,
                    dtype=hidden_states.dtype,
                )
            flat_hidden = hidden_states.reshape(-1, self.hidden_size)
            flat_indices = expert_indices.reshape(-1, expert_indices.shape[-1])
            # FP32 LoRA residuals can promote an expert output under CUDA
            # autocast. Accumulate routed contributions in the widest dtype
            # used by the active base/LoRA computation, then cast once at the
            # module boundary. This avoids lossy per-expert casts while keeping
            # the outward dtype identical to the hidden states.
            active_weights = (self.gate_proj, self.up_proj, self.down_proj)
            if not self.merged:
                active_weights += (
                    self.gate_lora_a,
                    self.gate_lora_b,
                    self.up_lora_a,
                    self.up_lora_b,
                    self.down_lora_a,
                    self.down_lora_b,
                )
            accumulator_dtype = hidden_states.dtype
            for weight in active_weights:
                accumulator_dtype = torch.promote_types(accumulator_dtype, weight.dtype)
            flat_weights = expert_weights.reshape_as(flat_indices).to(accumulator_dtype)
            result = torch.zeros(
                (flat_hidden.shape[0], self.hidden_size),
                device=hidden_states.device,
                dtype=accumulator_dtype,
            )
            for expert in range(self.num_experts):
                routing_weight = (flat_weights * (flat_indices == expert)).sum(dim=-1)
                selected = torch.nonzero(routing_weight != 0, as_tuple=False).flatten()
                if selected.numel() == 0:
                    continue
                values = self._expert(flat_hidden.index_select(0, selected), expert).to(
                    dtype=accumulator_dtype
                )
                result.index_add_(
                    0,
                    selected,
                    values * routing_weight.index_select(0, selected).unsqueeze(-1),
                )
            return result.reshape((*token_shape, self.hidden_size)).to(
                dtype=hidden_states.dtype
            )

        def _deltas(self, *, dtype: Any = None) -> tuple[Any, Any, Any]:
            target_dtype = dtype or self.gate_proj.dtype
            scale = self.scaling
            gate = torch.bmm(
                self.gate_lora_b.to(dtype=target_dtype),
                self.gate_lora_a.to(dtype=target_dtype),
            ) * scale
            up = torch.bmm(
                self.up_lora_b.to(dtype=target_dtype),
                self.up_lora_a.to(dtype=target_dtype),
            ) * scale
            down = torch.bmm(
                self.down_lora_b.to(dtype=target_dtype),
                self.down_lora_a.to(dtype=target_dtype),
            ) * scale
            return gate, up, down

        def merged_weights(self, *, dtype: Any | None = None) -> tuple[Any, Any, Any]:
            """Materialize base+LoRA, defaulting to FP32 for safe final export."""

            target_dtype = dtype or torch.float32
            if self.merged:
                return tuple(
                    value.detach().to(dtype=target_dtype).clone()
                    for value in (self.gate_proj, self.up_proj, self.down_proj)
                )
            gate_delta, up_delta, down_delta = self._deltas(dtype=target_dtype)
            return (
                self.gate_proj.to(dtype=target_dtype) + gate_delta,
                self.up_proj.to(dtype=target_dtype) + up_delta,
                self.down_proj.to(dtype=target_dtype) + down_delta,
            )

        def merge_(self) -> MergeableExpertLoRA:
            """Merge in-place only when FP32 bases make that operation loss-safe."""

            if self.merged:
                return self
            if self.gate_proj.dtype != torch.float32:
                raise RuntimeError(
                    "In-place LoRA merge requires FP32 base weights; use merged_weights() "
                    "to materialize an FP32 export from BF16 bases"
                )
            gate, up, down = self._deltas(dtype=torch.float32)
            with torch.no_grad():
                self.gate_proj.add_(gate)
                self.up_proj.add_(up)
                self.down_proj.add_(down)
                self._merged_flag.fill_(True)
                self._merged = True
            return self

        def unmerge_(self) -> MergeableExpertLoRA:
            if not self.merged:
                return self
            gate, up, down = self._deltas(dtype=self.gate_proj.dtype)
            with torch.no_grad():
                self.gate_proj.sub_(gate)
                self.up_proj.sub_(up)
                self.down_proj.sub_(down)
                self._merged_flag.fill_(False)
                self._merged = False
            return self

        def train(self, mode: bool = True) -> MergeableExpertLoRA:
            if mode and self.merged:
                raise RuntimeError("Unmerge LoRA weights before returning the expert bank to train mode")
            return super().train(mode)

    class SparseTransferMLP(nn.Module):
        """Drop-in shared+top-k folded-expert MLP for sparse distillation."""

        def __init__(
            self,
            shared_mlp: Any,
            gate_proj: Any,
            up_proj: Any,
            down_proj: Any,
            router: Any,
            *,
            top_k: int = 2,
            norm_topk_prob: bool = True,
            lora_rank: int = 16,
            lora_alpha: float = 16.0,
            lora_dropout: float = 0.0,
            lora_trainable_dtype: Any | None = None,
            branch_scale: float = 1.0,
            trainable_scale: bool = True,
            freeze_shared: bool = True,
        ) -> None:
            super().__init__()
            if not isinstance(shared_mlp, nn.Module):
                raise TypeError("shared_mlp must be a torch.nn.Module")
            self.shared_mlp = shared_mlp
            if freeze_shared:
                self.shared_mlp.requires_grad_(False)
            self.experts = MergeableExpertLoRA(
                gate_proj,
                up_proj,
                down_proj,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                trainable_dtype=lora_trainable_dtype,
            )
            if isinstance(router, torch.Tensor):
                expected = (self.experts.num_experts, self.experts.hidden_size)
                if tuple(router.shape) != expected:
                    raise ShapeError(f"router weight must be {expected}, got {tuple(router.shape)}")
                router_module = nn.Linear(
                    self.experts.hidden_size,
                    self.experts.num_experts,
                    bias=False,
                    device=router.device,
                    dtype=router.dtype,
                )
                with torch.no_grad():
                    router_module.weight.copy_(router)
                router = router_module
            if not isinstance(router, nn.Module):
                raise TypeError("router must be a Tensor weight or torch.nn.Module")
            self.router = router
            self.norm_topk_prob = bool(norm_topk_prob)
            self.set_top_k(top_k)
            self.branch_scale = nn.Parameter(
                torch.tensor(
                    [float(branch_scale)], device=gate_proj.device, dtype=torch.float32
                ),
                requires_grad=trainable_scale,
            )
            self.last_aux: dict[str, Any] | None = None
            self.record_aux_enabled = False
            self.transfer_enabled = True
            self.dense_oracle_enabled = False

        def set_top_k(self, value: int) -> None:
            if isinstance(value, bool) or not 1 <= value <= self.experts.num_experts:
                raise ValueError(
                    f"top_k must be in [1, {self.experts.num_experts}], got {value!r}"
                )
            self.top_k = int(value)

        def set_record_aux(self, enabled: bool) -> None:
            """Enable aux capture when the parent HF decoder only calls ``mlp(x)``."""

            self.record_aux_enabled = bool(enabled)

        def clear_aux(self) -> None:
            self.last_aux = None

        def set_transfer_enabled(self, enabled: bool) -> None:
            """Toggle router/experts while preserving the frozen shared FFN."""

            self.transfer_enabled = bool(enabled)

        def set_dense_oracle_enabled(self, enabled: bool) -> None:
            """Use the exact sum of all folded experts, bypassing the router."""

            self.dense_oracle_enabled = bool(enabled)

        @contextmanager
        def transfer_mode(self, enabled: bool) -> Any:
            previous = self.transfer_enabled
            self.transfer_enabled = bool(enabled)
            try:
                yield self
            finally:
                self.transfer_enabled = previous

        @contextmanager
        def shared_only(self) -> Any:
            with self.transfer_mode(False):
                yield self

        @contextmanager
        def aux_recording(self, enabled: bool = True) -> Any:
            previous = self.record_aux_enabled
            self.record_aux_enabled = bool(enabled)
            try:
                yield self
            finally:
                self.record_aux_enabled = previous

        def _route(self, hidden_states: Any, top_k: int) -> tuple[Any, Any, Any]:
            logits = self.router(hidden_states)
            expected = (*hidden_states.shape[:-1], self.experts.num_experts)
            if tuple(logits.shape) != expected:
                raise ShapeError(
                    f"router output must have shape {expected}, got {tuple(logits.shape)}"
                )
            probabilities = torch.softmax(logits.to(torch.float32), dim=-1)
            weights, indices = torch.topk(probabilities, top_k, dim=-1)
            if self.norm_topk_prob:
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
            return logits, indices, weights.to(dtype=hidden_states.dtype)

        def forward(
            self,
            hidden_states: Any,
            *,
            top_k: int | None = None,
            expert_indices: Any | None = None,
            expert_weights: Any | None = None,
            return_expert_outputs: bool = False,
            record_aux: bool | None = None,
        ) -> Any:
            should_record = self.record_aux_enabled if record_aux is None else bool(record_aux)
            if not self.transfer_enabled:
                # Anchor-KL commonly runs immediately after the routed forward;
                # retain that routed pass's aux tensors for subsequent losses.
                return self.shared_mlp(hidden_states)
            if self.dense_oracle_enabled:
                need_all = should_record or return_expert_outputs
                dense_result = self.experts(
                    hidden_states,
                    return_expert_outputs=need_all,
                )
                if need_all:
                    dense, all_outputs = dense_result
                else:
                    dense = dense_result
                    all_outputs = None
                # ``branch_scale`` is stored in native routed-average units.
                # Sparse construction multiplies the dense-sum scale by E so a
                # uniform top-E route is an exact warm start.  Divide by E when
                # explicitly reconstructing the unnormalized all-expert oracle.
                scale = self.branch_scale.to(dtype=dense.dtype) / self.experts.num_experts
                routed = dense * scale
                output = self.shared_mlp(hidden_states) + routed
                self.last_aux = {
                    "router_logits": None,
                    "expert_indices": None,
                    "expert_weights": None,
                    "expert_outputs": (
                        all_outputs * scale if all_outputs is not None else None
                    ),
                    "dense_sum": routed,
                    "routed_output": routed,
                }
                if return_expert_outputs:
                    assert all_outputs is not None
                    return output, all_outputs * scale
                return output
            selected_top_k = self.top_k if top_k is None else int(top_k)
            if not 1 <= selected_top_k <= self.experts.num_experts:
                raise ValueError("Requested top_k is outside the expert count")
            router_logits = None
            indices_prevalidated = False
            if expert_indices is None:
                if expert_weights is not None:
                    raise ValueError("expert_weights requires expert_indices")
                router_logits, expert_indices, expert_weights = self._route(
                    hidden_states, selected_top_k
                )
                indices_prevalidated = True
            else:
                validate_topk(
                    expert_indices,
                    expert_weights,
                    tuple(hidden_states.shape[:-1]),
                    self.experts.num_experts,
                )
                indices_prevalidated = True
                if expert_weights is None:
                    expert_weights = torch.full(
                        expert_indices.shape,
                        1.0 / expert_indices.shape[-1],
                        device=expert_indices.device,
                        dtype=hidden_states.dtype,
                    )
            need_all = should_record or return_expert_outputs
            expert_result = self.experts(
                hidden_states,
                expert_indices=expert_indices,
                expert_weights=expert_weights,
                return_expert_outputs=need_all,
                _indices_prevalidated=indices_prevalidated,
            )
            if need_all:
                routed, all_outputs = expert_result
            else:
                routed = expert_result
                all_outputs = None
            routed = routed * self.branch_scale.to(dtype=routed.dtype)
            output = self.shared_mlp(hidden_states) + routed
            # Router diagnostics are intentionally retained every batch for load
            # balancing and z-loss. Full expert outputs are opt-in because they
            # activate all experts and are suitable only for dense-oracle samples.
            self.last_aux = {
                "router_logits": router_logits,
                "expert_indices": expert_indices,
                "expert_weights": expert_weights,
                "expert_outputs": None,
                "dense_sum": None,
                "routed_output": routed,
            }
            if should_record:
                assert all_outputs is not None
                scaled_all = all_outputs * self.branch_scale.to(dtype=all_outputs.dtype)
                self.last_aux["expert_outputs"] = scaled_all
                self.last_aux["dense_sum"] = scaled_all.sum(dim=-2)
            if return_expert_outputs:
                assert all_outputs is not None
                return output, all_outputs * self.branch_scale.to(dtype=all_outputs.dtype)
            return output

    classes = {
        cls.__name__: cls
        for cls in (
            TransferAdapters,
            DenseTransferMLP,
            SharedDenseTransferMLP,
            MergeableExpertLoRA,
            SparseTransferMLP,
        )
    }
    for name, cls in classes.items():
        cls.__module__ = __name__
        cls.__qualname__ = name
    _CLASS_CACHE.update(classes)
    globals().update(classes)
    return _CLASS_CACHE


def __getattr__(name: str) -> Any:
    if name in __all__:
        return _build_classes()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
