from __future__ import annotations

import unittest
from unittest import mock

try:
    import torch
except (ImportError, OSError):
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional test dependency")
class TorchModelingTests(unittest.TestCase):
    def setUp(self) -> None:
        from twen.modeling import build_channel_partition

        torch.manual_seed(7)
        self.small = 3
        self.large = 4
        self.channels = 6
        self.gate = torch.randn(self.channels, self.large)
        self.up = torch.randn(self.channels, self.large)
        self.down = torch.randn(self.large, self.channels)
        self.a = torch.randn(self.large, self.small)
        self.b = torch.randn(self.small, self.large)
        self.partition = build_channel_partition(
            scores=list(range(1, self.channels + 1)), num_experts=2, expert_size=3
        )

    def test_bidirectional_ridge_recovers_linear_maps(self) -> None:
        from twen.modeling import BidirectionalRidgeStats

        small = torch.randn(128, 3, dtype=torch.float64)
        mapping = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.5]],
            dtype=torch.float64,
        )
        large = small @ mapping.T
        stats = BidirectionalRidgeStats(3, 3)
        stats.update(small[:64], large[:64])
        stats.update(small[64:], large[64:])
        solution = stats.solve(l2=1e-10)
        torch.testing.assert_close(solution.input_adapter, mapping, rtol=1e-7, atol=1e-7)
        torch.testing.assert_close(
            solution.output_adapter, torch.linalg.inv(mapping), rtol=1e-7, atol=1e-7
        )

    def test_linear_cka_flattens_batch_and_sequence_as_samples(self) -> None:
        from twen.modeling import linear_cka

        activations = torch.randn(2, 7, 4)
        self.assertAlmostEqual(linear_cka(activations, activations * 3.0), 1.0, places=10)

    def test_fp32_fold_is_equivalent_to_adapted_dense_donor(self) -> None:
        from twen.modeling import fold_expert_weights, max_fold_error

        folded = fold_expert_weights(
            self.gate, self.up, self.down, self.a, self.b, self.partition
        )
        self.assertEqual(folded.gate_proj.shape, (2, 3, 3))
        self.assertEqual(folded.down_proj.shape, (2, 3, 3))
        x = torch.randn(11, self.small)
        self.assertLess(
            max_fold_error(x, self.gate, self.up, self.down, self.a, self.b, folded),
            2e-5,
        )

    def test_dense_hf_replacement_and_aux_switch(self) -> None:
        from twen.modeling import SharedDenseTransferMLP, TransferAdapters

        shared = torch.nn.Linear(self.small, self.small, bias=False)
        adapters = TransferAdapters(
            self.small,
            self.large,
            input_weight=self.a,
            output_weight=self.b,
        )
        module = SharedDenseTransferMLP(
            shared,
            self.gate,
            self.up,
            self.down,
            self.partition,
            adapters=adapters,
            branch_scale=0.01,
        )
        x = torch.randn(2, 5, self.small)
        self.assertEqual(module(x).shape, x.shape)
        self.assertIsNone(module.last_aux)
        with module.aux_recording():
            self.assertEqual(module(x).shape, x.shape)
            self.assertEqual(module.last_aux["expert_outputs"].shape, (2, 5, 2, 3))
        self.assertFalse(module.record_aux_enabled)
        prior_aux = module.last_aux
        with module.shared_only():
            torch.testing.assert_close(module(x), shared(x))
            self.assertIs(module.last_aux, prior_aux)
        self.assertTrue(all(not p.requires_grad for p in shared.parameters()))

    def test_dense_full_fast_path_matches_expert_sum_and_gradients(self) -> None:
        from twen.modeling import DenseTransferMLP, TransferAdapters

        adapters = TransferAdapters(
            self.small,
            self.large,
            input_weight=self.a,
            output_weight=self.b,
        )
        module = DenseTransferMLP(
            self.gate,
            self.up,
            self.down,
            self.partition,
            adapters=adapters,
            branch_scale=0.37,
        )
        x_fast = torch.randn(2, 5, self.small, requires_grad=True)
        with mock.patch.object(
            module,
            "_all_experts",
            side_effect=AssertionError("dense fast path materialized expert outputs"),
        ):
            fast = module(x_fast)
        fast.square().mean().backward()
        fast_gradients = {
            "input": x_fast.grad.detach().clone(),
            "A": module.adapters.A.grad.detach().clone(),
            "B": module.adapters.B.grad.detach().clone(),
            "scale": module.branch_scale.grad.detach().clone(),
        }

        module.zero_grad(set_to_none=True)
        x_reference = x_fast.detach().clone().requires_grad_(True)
        with module.aux_recording():
            reference = module(x_reference)
            self.assertEqual(
                module.last_aux["expert_outputs"].shape,
                (2, 5, 2, self.small),
            )
        reference.square().mean().backward()
        reference_gradients = {
            "input": x_reference.grad.detach().clone(),
            "A": module.adapters.A.grad.detach().clone(),
            "B": module.adapters.B.grad.detach().clone(),
            "scale": module.branch_scale.grad.detach().clone(),
        }

        torch.testing.assert_close(fast, reference, rtol=2e-5, atol=2e-6)
        for name in fast_gradients:
            torch.testing.assert_close(
                fast_gradients[name],
                reference_gradients[name],
                rtol=3e-5,
                atol=3e-6,
            )
        explicit, expert_outputs = module(
            x_fast.detach(),
            return_expert_outputs=True,
        )
        torch.testing.assert_close(explicit, reference.detach(), rtol=2e-5, atol=2e-6)
        self.assertEqual(expert_outputs.shape, (2, 5, 2, self.small))

    def test_differentiable_folded_dense_matches_expanded_fp32_gradients(self) -> None:
        from twen.modeling import DenseTransferMLP, TransferAdapters

        def build(mode: str, checkpoint_tokens: bool = False):
            return DenseTransferMLP(
                self.gate.clone(),
                self.up.clone(),
                self.down.clone(),
                self.partition,
                adapters=TransferAdapters(
                    self.small,
                    self.large,
                    input_weight=self.a.clone(),
                    output_weight=self.b.clone(),
                ),
                branch_scale=0.37,
                execution_mode=mode,
                checkpoint_token_branch=checkpoint_tokens,
            )

        expanded = build("expanded")
        folded = build("differentiable_folded")
        upstream = torch.randn(2, 5, self.small)
        base_input = torch.randn(2, 5, self.small)
        inputs = [base_input.clone().requires_grad_(True) for _ in range(2)]
        outputs = [expanded(inputs[0]), folded(inputs[1])]
        for output in outputs:
            (output * upstream).sum().backward()

        torch.testing.assert_close(outputs[0], outputs[1], rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(inputs[0].grad, inputs[1].grad, rtol=4e-5, atol=4e-6)
        for expanded_value, folded_value in (
            (expanded.adapters.A.grad, folded.adapters.A.grad),
            (expanded.adapters.B.grad, folded.adapters.B.grad),
            (expanded.branch_scale.grad, folded.branch_scale.grad),
        ):
            torch.testing.assert_close(
                expanded_value,
                folded_value,
                rtol=4e-5,
                atol=4e-6,
            )

    def test_folded_token_checkpoint_recomputes_only_token_core(self) -> None:
        from twen.modeling import DenseTransferMLP, TransferAdapters

        module = DenseTransferMLP(
            self.gate,
            self.up,
            self.down,
            self.partition,
            adapters=TransferAdapters(
                self.small,
                self.large,
                input_weight=self.a,
                output_weight=self.b,
            ),
            execution_mode="differentiable_folded",
            checkpoint_token_branch=True,
        )
        x = torch.randn(2, 5, self.small, requires_grad=True)
        original = module._folded_token_core
        with mock.patch.object(module, "_folded_token_core", wraps=original) as token_core:
            module(x).square().mean().backward()

        self.assertEqual(token_core.call_count, 2)
        self.assertIsNotNone(module.adapters.A.grad)
        self.assertIsNotNone(module.adapters.B.grad)
        self.assertIsNotNone(module.branch_scale.grad)

        module.configure_execution(checkpoint_token_branch=False)
        self.assertFalse(module.checkpoint_token_branch)
        module.configure_execution(execution_mode="expanded", checkpoint_token_branch=True)
        self.assertEqual(module.execution_mode, "expanded")
        self.assertTrue(module.checkpoint_token_branch)

    def test_expanded_selective_checkpoint_matches_direct_gradients(self) -> None:
        from twen.modeling import DenseTransferMLP, TransferAdapters

        def build(checkpoint_tokens: bool):
            return DenseTransferMLP(
                self.gate.clone(),
                self.up.clone(),
                self.down.clone(),
                self.partition,
                adapters=TransferAdapters(
                    self.small,
                    self.large,
                    input_weight=self.a.clone(),
                    output_weight=self.b.clone(),
                ),
                branch_scale=0.37,
                execution_mode="expanded",
                checkpoint_token_branch=checkpoint_tokens,
            )

        direct = build(False)
        checkpointed = build(True)
        base_input = torch.randn(2, 5, self.small)
        inputs = [base_input.clone().requires_grad_(True) for _ in range(2)]
        upstream = torch.randn_like(base_input)
        direct_output = direct(inputs[0])
        with mock.patch.object(
            checkpointed,
            "_expanded_token_core",
            wraps=checkpointed._expanded_token_core,
        ) as token_core:
            checkpointed_output = checkpointed(inputs[1])
            (checkpointed_output * upstream).sum().backward()
        (direct_output * upstream).sum().backward()

        self.assertEqual(token_core.call_count, 2)
        torch.testing.assert_close(direct_output, checkpointed_output, rtol=0, atol=0)
        torch.testing.assert_close(inputs[0].grad, inputs[1].grad, rtol=0, atol=0)
        for direct_value, checkpointed_value in (
            (direct.adapters.A.grad, checkpointed.adapters.A.grad),
            (direct.adapters.B.grad, checkpointed.adapters.B.grad),
            (direct.branch_scale.grad, checkpointed.branch_scale.grad),
        ):
            torch.testing.assert_close(direct_value, checkpointed_value, rtol=0, atol=0)

    def test_dense_donor_reuses_existing_frozen_parameters(self) -> None:
        from twen.modeling import DenseTransferMLP, TransferAdapters

        gate = torch.nn.Parameter(self.gate.clone(), requires_grad=False)
        up = torch.nn.Parameter(self.up.clone(), requires_grad=False)
        down = torch.nn.Parameter(self.down.clone(), requires_grad=False)
        module = DenseTransferMLP(
            gate,
            up,
            down,
            self.partition,
            adapters=TransferAdapters(
                self.small,
                self.large,
                input_weight=self.a,
                output_weight=self.b,
            ),
        )

        self.assertIs(module.gate_weight, gate)
        self.assertIs(module.up_weight, up)
        self.assertIs(module.down_weight, down)
        self.assertEqual(
            module.gate_weight.untyped_storage().data_ptr(),
            gate.untyped_storage().data_ptr(),
        )

        # A caller-owned trainable source must not be frozen as a side effect.
        trainable_gate = torch.nn.Parameter(self.gate.clone(), requires_grad=True)
        defensive = DenseTransferMLP(
            trainable_gate,
            self.up,
            self.down,
            self.partition,
            adapters=TransferAdapters(
                self.small,
                self.large,
                input_weight=self.a,
                output_weight=self.b,
            ),
        )
        self.assertIsNot(defensive.gate_weight, trainable_gate)
        self.assertTrue(trainable_gate.requires_grad)
        self.assertFalse(defensive.gate_weight.requires_grad)

    def test_sparse_router_and_lora_merge(self) -> None:
        from twen.modeling import SparseTransferMLP, fold_expert_weights

        folded = fold_expert_weights(
            self.gate, self.up, self.down, self.a, self.b, self.partition
        )
        shared = torch.nn.Linear(self.small, self.small, bias=False)
        router = torch.randn(2, self.small)
        module = SparseTransferMLP(
            shared,
            folded.gate_proj,
            folded.up_proj,
            folded.down_proj,
            router,
            top_k=1,
            lora_rank=2,
            lora_alpha=2.0,
        )
        x = torch.randn(2, 4, self.small)
        output = module(x)
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(module.last_aux["expert_indices"].shape, (2, 4, 1))
        self.assertIsNone(module.last_aux["dense_sum"])
        with module.aux_recording():
            output = module(x)
            self.assertEqual(module.last_aux["dense_sum"].shape, x.shape)
        module.set_dense_oracle_enabled(True)
        expected_dense = (
            shared(x)
            + module.experts(x) * module.branch_scale / module.experts.num_experts
        )
        torch.testing.assert_close(module(x), expected_dense)
        self.assertIsNone(module.last_aux["router_logits"])
        module.set_dense_oracle_enabled(False)
        prior_aux = module.last_aux
        with module.shared_only():
            torch.testing.assert_close(module(x), shared(x))
            self.assertIs(module.last_aux, prior_aux)
        with torch.no_grad():
            module.experts.gate_lora_b.normal_(std=0.01)
            module.experts.up_lora_b.normal_(std=0.01)
            module.experts.down_lora_b.normal_(std=0.01)
        before_merge = module.experts(x)
        gate, up, down = module.experts.merged_weights()
        manual = torch.zeros_like(x)
        for expert in range(2):
            intermediate = torch.nn.functional.silu(
                torch.nn.functional.linear(x, gate[expert])
            ) * torch.nn.functional.linear(x, up[expert])
            manual += torch.nn.functional.linear(intermediate, down[expert])
        torch.testing.assert_close(before_merge, manual, rtol=1e-5, atol=1e-5)
        module.experts.merge_()
        torch.testing.assert_close(module.experts(x), before_merge, rtol=1e-5, atol=1e-5)
        module.experts.eval()
        module.experts.unmerge_()

    def test_full_expert_route_avoids_dispatch_and_matches_gradients(self) -> None:
        from twen.modeling import SparseTransferMLP, fold_expert_weights

        folded = fold_expert_weights(
            self.gate, self.up, self.down, self.a, self.b, self.partition
        )
        module = SparseTransferMLP(
            torch.nn.Linear(self.small, self.small, bias=False),
            folded.gate_proj,
            folded.up_proj,
            folded.down_proj,
            torch.randn(2, self.small),
            top_k=2,
            lora_rank=2,
            lora_alpha=2.0,
        )
        with torch.no_grad():
            module.experts.gate_lora_b.normal_(std=0.03)
            module.experts.up_lora_b.normal_(std=0.03)
            module.experts.down_lora_b.normal_(std=0.03)

        upstream = torch.randn(2, 4, self.small)
        x_fast = torch.randn(2, 4, self.small, requires_grad=True)
        with (
            mock.patch.object(
                module.experts,
                "_expert",
                side_effect=AssertionError("full route dispatched a Python expert"),
            ),
            mock.patch.object(
                torch,
                "nonzero",
                side_effect=AssertionError("full route called torch.nonzero"),
            ),
        ):
            fast = module(x_fast)
        self.assertIsNotNone(module.last_aux)
        fast_router_logits = module.last_aux["router_logits"].detach().clone()
        fast_indices = module.last_aux["expert_indices"].detach().clone()
        fast_weights = module.last_aux["expert_weights"].detach().clone()
        fast_routed = module.last_aux["routed_output"].detach().clone()
        self.assertIsNone(module.last_aux["expert_outputs"])
        self.assertIsNone(module.last_aux["dense_sum"])
        (fast * upstream).sum().backward()
        fast_output = fast.detach().clone()
        fast_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        fast_input_gradient = x_fast.grad.detach().clone()

        module.clear_aux()
        module.zero_grad(set_to_none=True)
        x_reference = x_fast.detach().clone().requires_grad_(True)
        router_logits = module.router(x_reference)
        probabilities = router_logits.float().softmax(dim=-1)
        weights, indices = torch.topk(probabilities, module.experts.num_experts, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        weights = weights.to(dtype=x_reference.dtype)

        flat_hidden = x_reference.reshape(-1, module.experts.hidden_size)
        flat_indices = indices.reshape(-1, indices.shape[-1])
        flat_weights = weights.reshape_as(flat_indices)
        routed = x_reference.new_zeros((flat_hidden.shape[0], module.experts.hidden_size))
        for expert in range(module.experts.num_experts):
            routing_weight = (flat_weights * (flat_indices == expert)).sum(dim=-1)
            selected = torch.nonzero(routing_weight != 0, as_tuple=False).flatten()
            values = module.experts._expert(flat_hidden.index_select(0, selected), expert)
            routed.index_add_(
                0,
                selected,
                values * routing_weight.index_select(0, selected).unsqueeze(-1),
            )
        scaled_routed = routed.reshape_as(x_reference) * module.branch_scale.to(
            dtype=x_reference.dtype
        )
        reference = module.shared_mlp(x_reference) + scaled_routed
        (reference * upstream).sum().backward()

        torch.testing.assert_close(fast_output, reference.detach(), rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(fast_router_logits, router_logits.detach())
        torch.testing.assert_close(fast_indices, indices)
        torch.testing.assert_close(fast_weights, weights)
        torch.testing.assert_close(fast_routed, scaled_routed.detach(), rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(
            fast_input_gradient,
            x_reference.grad,
            rtol=3e-5,
            atol=3e-6,
        )
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(
                    fast_gradients[name],
                    parameter.grad,
                    rtol=3e-5,
                    atol=3e-6,
                )

        module.zero_grad(set_to_none=True)
        probe = x_reference.detach()
        with torch.no_grad(), module.aux_recording():
            module(probe)
            recorded_experts = module.last_aux["expert_outputs"]
            recorded_dense = module.last_aux["dense_sum"]
            reference_experts = torch.stack(
                [module.experts._expert(probe, expert) for expert in range(2)],
                dim=-2,
            ) * module.branch_scale.to(dtype=probe.dtype)
        torch.testing.assert_close(recorded_experts, reference_experts, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(
            recorded_dense,
            reference_experts.sum(dim=-2),
            rtol=2e-5,
            atol=2e-6,
        )

    def test_mixed_dtype_sparse_routes_preserve_boundary_dtype_and_gradients(self) -> None:
        from twen.modeling import SparseTransferMLP

        experts, hidden, intermediate, rank = 8, 8, 6, 2

        def build(*, merged: bool) -> SparseTransferMLP:
            base_dtype = torch.float32 if merged else torch.bfloat16
            module = SparseTransferMLP(
                torch.nn.Identity(),
                torch.randn(experts, intermediate, hidden, dtype=base_dtype),
                torch.randn(experts, intermediate, hidden, dtype=base_dtype),
                torch.randn(experts, hidden, intermediate, dtype=base_dtype),
                torch.randn(experts, hidden, dtype=torch.float32),
                top_k=experts,
                lora_rank=rank,
                lora_alpha=float(rank),
                lora_trainable_dtype=torch.float32,
            )
            with torch.no_grad():
                module.experts.gate_lora_b.normal_(std=0.03)
                module.experts.up_lora_b.normal_(std=0.03)
                module.experts.down_lora_b.normal_(std=0.03)
            if merged:
                module.eval()
                module.experts.merge_()
            return module

        torch.manual_seed(19)
        for merged in (False, True):
            for top_k in (8, 4, 2):
                with self.subTest(merged=merged, top_k=top_k):
                    module = build(merged=merged)
                    inputs = torch.randn(
                        2,
                        5,
                        hidden,
                        dtype=torch.bfloat16,
                        requires_grad=True,
                    )

                    # CUDA autocast can expose FP32 expert outputs when FP32
                    # LoRA residuals are added to BF16 bases. Force that
                    # promotion on CPU so this regression test exercises the
                    # same accumulator and public-dtype boundary.
                    if top_k == experts:
                        original_full = module.experts._all_expert_outputs_vectorized
                        promoted = mock.patch.object(
                            module.experts,
                            "_all_expert_outputs_vectorized",
                            side_effect=lambda value, function=original_full: function(
                                value
                            ).float(),
                        )
                    else:
                        original_expert = module.experts._expert
                        promoted = mock.patch.object(
                            module.experts,
                            "_expert",
                            side_effect=lambda value, expert, function=original_expert: function(
                                value, expert
                            ).float(),
                        )
                    with promoted, torch.autocast(
                        device_type="cpu",
                        dtype=torch.bfloat16,
                    ):
                        output = module(inputs, top_k=top_k)
                    self.assertEqual(output.dtype, torch.bfloat16)
                    output.float().square().mean().backward()

                    self.assertIsNotNone(inputs.grad)
                    self.assertEqual(inputs.grad.dtype, torch.bfloat16)
                    self.assertTrue(torch.isfinite(inputs.grad).all())
                    self.assertGreater(float(inputs.grad.float().abs().sum()), 0.0)
                    for parameter in (module.router.weight, module.branch_scale):
                        self.assertIsNotNone(parameter.grad)
                        self.assertEqual(parameter.grad.dtype, torch.float32)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

                    lora_parameters = {
                        name: parameter
                        for name, parameter in module.experts.named_parameters()
                        if "lora_" in name
                    }
                    self.assertTrue(lora_parameters)
                    if merged:
                        self.assertTrue(
                            all(parameter.grad is None for parameter in lora_parameters.values())
                        )
                    else:
                        for parameter in lora_parameters.values():
                            self.assertIsNotNone(parameter.grad)
                            self.assertEqual(parameter.grad.dtype, torch.float32)
                            self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertGreater(
                            sum(
                                float(parameter.grad.abs().sum())
                                for parameter in lora_parameters.values()
                            ),
                            0.0,
                        )

    def test_lora_python_merge_state_tracks_state_dict_loads(self) -> None:
        from twen.modeling import MergeableExpertLoRA, fold_expert_weights

        folded = fold_expert_weights(
            self.gate, self.up, self.down, self.a, self.b, self.partition
        )

        def build() -> MergeableExpertLoRA:
            return MergeableExpertLoRA(
                folded.gate_proj.clone(),
                folded.up_proj.clone(),
                folded.down_proj.clone(),
                rank=2,
                alpha=2.0,
            )

        source = build()
        with torch.no_grad():
            source.gate_lora_b.normal_(std=0.01)
            source.up_lora_b.normal_(std=0.01)
            source.down_lora_b.normal_(std=0.01)
        x = torch.randn(2, 4, self.small)
        expected = source(x)
        unmerged_state = {
            name: value.detach().clone() for name, value in source.state_dict().items()
        }

        source.merge_()
        self.assertTrue(source.merged)
        merged_state = {
            name: value.detach().clone() for name, value in source.state_dict().items()
        }
        restored = build()
        restored.load_state_dict(merged_state)
        self.assertTrue(restored.merged)
        with self.assertRaisesRegex(RuntimeError, "Unmerge LoRA weights"):
            restored.train()

        with mock.patch.object(
            torch.Tensor,
            "item",
            side_effect=AssertionError("forward read the persistent merge tensor"),
        ):
            restored_output = restored(x)
        torch.testing.assert_close(restored_output, expected, rtol=1e-5, atol=1e-5)

        restored.eval()
        restored.unmerge_()
        self.assertFalse(restored.merged)
        torch.testing.assert_close(restored(x), expected, rtol=1e-5, atol=1e-5)

        restored.merge_()
        self.assertTrue(restored.merged)
        restored.load_state_dict(unmerged_state)
        self.assertFalse(restored.merged)
        torch.testing.assert_close(restored(x), expected, rtol=1e-5, atol=1e-5)

    def test_internal_router_indices_skip_range_sync_but_external_indices_validate(self) -> None:
        from twen.modeling import ShapeError, SparseTransferMLP, fold_expert_weights

        folded = fold_expert_weights(
            self.gate, self.up, self.down, self.a, self.b, self.partition
        )
        module = SparseTransferMLP(
            torch.nn.Linear(self.small, self.small, bias=False),
            folded.gate_proj,
            folded.up_proj,
            folded.down_proj,
            torch.randn(2, self.small),
            top_k=1,
            lora_rank=2,
            lora_alpha=2.0,
        )
        x = torch.randn(2, 4, self.small)
        with mock.patch.object(
            torch.Tensor,
            "any",
            side_effect=AssertionError("internal router performed a range reduction"),
        ):
            self.assertEqual(module(x).shape, x.shape)

        invalid = torch.full((2, 4, 1), 2, dtype=torch.int64)
        with self.assertRaisesRegex(ShapeError, "expert_indices must be in"):
            module(x, expert_indices=invalid)
        with self.assertRaisesRegex(ShapeError, "expert_indices must be in"):
            module.experts(x, expert_indices=invalid)

    def test_uniform_top_e_is_exact_dense_fold_warm_start(self) -> None:
        from twen.modeling import SparseTransferMLP, fold_expert_weights

        folded = fold_expert_weights(
            self.gate, self.up, self.down, self.a, self.b, self.partition
        )
        shared = torch.nn.Linear(self.small, self.small, bias=False)
        dense_scale = 0.37
        module = SparseTransferMLP(
            shared,
            folded.gate_proj,
            folded.up_proj,
            folded.down_proj,
            torch.zeros(2, self.small),
            top_k=2,
            lora_rank=2,
            lora_alpha=2.0,
            branch_scale=dense_scale * 2,
        )
        x = torch.randn(3, 4, self.small)
        expected = shared(x) + module.experts(x) * dense_scale
        torch.testing.assert_close(module(x), expected, rtol=1e-5, atol=1e-5)
        module.set_dense_oracle_enabled(True)
        torch.testing.assert_close(module(x), expected, rtol=1e-5, atol=1e-5)

    def test_native_state_uses_fixed_shared_gate_and_compensated_down(self) -> None:
        from twen.modeling import export_native_moe_state, fold_expert_weights

        folded = fold_expert_weights(
            self.gate, self.up, self.down, self.a, self.b, self.partition
        )
        state = {}
        folded_layers = []
        routers = []
        for layer in range(24):
            prefix = f"model.language_model.layers.{layer}.mlp"
            state[f"{prefix}.gate_proj.weight"] = torch.randn(5, self.small)
            state[f"{prefix}.up_proj.weight"] = torch.randn(5, self.small)
            state[f"{prefix}.down_proj.weight"] = torch.randn(self.small, 5)
            folded_layers.append(folded)
            routers.append(torch.randn(2, self.small))
        state["model.visual.fake.weight"] = torch.ones(1)
        original_down = state["model.language_model.layers.0.mlp.down_proj.weight"].clone()
        exported = export_native_moe_state(state, folded_layers, routers)
        prefix = "model.layers.0.mlp"
        torch.testing.assert_close(
            exported[f"{prefix}.shared_expert.down_proj.weight"], original_down * 2
        )
        self.assertEqual(
            torch.count_nonzero(exported[f"{prefix}.shared_expert_gate.weight"]).item(), 0
        )
        self.assertEqual(exported[f"{prefix}.experts.gate_up_proj"].shape, (2, 6, 3))
        self.assertEqual(exported[f"{prefix}.experts.down_proj"].shape, (2, 3, 3))
        self.assertNotIn(f"{prefix}.gate_proj.weight", exported)
        self.assertNotIn("model.visual.fake.weight", exported)
        try:
            from transformers import Qwen3_5MoeTextConfig
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                Qwen3_5MoeSparseMoeBlock,
            )
        except ImportError:
            self.skipTest("Installed Transformers lacks Qwen3.5-MoE")
        config = Qwen3_5MoeTextConfig(
            hidden_size=self.small,
            moe_intermediate_size=3,
            shared_expert_intermediate_size=5,
            num_experts=2,
            num_experts_per_tok=1,
        )
        block = Qwen3_5MoeSparseMoeBlock(config)
        native_layer = {
            name.removeprefix(f"{prefix}."): value
            for name, value in exported.items()
            if name.startswith(f"{prefix}.")
        }
        block.load_state_dict(native_layer, strict=True)

    def test_native_state_has_production_fused_expert_shapes_on_meta(self) -> None:
        """Lock the 8x1536 production axes without allocating the ~0.9B bank."""

        from twen.modeling import export_native_moe_state

        hidden = 1024
        shared_intermediate = 3584
        experts = 8
        expert_intermediate = 1536
        device = torch.device("meta")
        folded = (
            torch.empty((experts, expert_intermediate, hidden), device=device),
            torch.empty((experts, expert_intermediate, hidden), device=device),
            torch.empty((experts, hidden, expert_intermediate), device=device),
        )
        router = torch.empty((experts, hidden), device=device)
        state = {}
        for layer in range(24):
            prefix = f"model.layers.{layer}.mlp"
            state[f"{prefix}.gate_proj.weight"] = torch.empty(
                (shared_intermediate, hidden), device=device
            )
            state[f"{prefix}.up_proj.weight"] = torch.empty(
                (shared_intermediate, hidden), device=device
            )
            state[f"{prefix}.down_proj.weight"] = torch.empty(
                (hidden, shared_intermediate), device=device
            )

        exported = export_native_moe_state(
            state,
            [folded] * 24,
            [router] * 24,
            target_dtype=torch.bfloat16,
        )
        prefix = "model.layers.0.mlp"
        self.assertEqual(
            exported[f"{prefix}.experts.gate_up_proj"].shape,
            (8, 3072, 1024),
        )
        self.assertEqual(
            exported[f"{prefix}.experts.down_proj"].shape,
            (8, 1024, 1536),
        )
        self.assertEqual(exported[f"{prefix}.gate.weight"].shape, (8, 1024))
        self.assertEqual(
            exported[f"{prefix}.shared_expert.gate_proj.weight"].shape,
            (3584, 1024),
        )
        self.assertEqual(
            exported[f"{prefix}.shared_expert_gate.weight"].shape,
            (1, 1024),
        )
        self.assertEqual(
            exported[f"{prefix}.experts.gate_up_proj"].dtype,
            torch.bfloat16,
        )

    def test_folded_artifact_shapes_are_checked_against_sparse_config(self) -> None:
        from twen.training.builder import BuildError, _load_folded_layer

        class Handle:
            def __init__(self, tensors):
                self.tensors = tensors

            def keys(self):
                return self.tensors.keys()

            def get_tensor(self, name):
                return self.tensors[name]

        tensors = {
            # Internally self-consistent 4x3 experts, but the requested sparse
            # contract below is 2x6. Equal total width must not be enough.
            "layers.0.gate_proj": torch.zeros(4, 3, 5),
            "layers.0.up_proj": torch.zeros(4, 3, 5),
            "layers.0.down_proj": torch.zeros(4, 5, 3),
            "layers.0.router": torch.zeros(4, 5),
        }
        with self.assertRaisesRegex(BuildError, "gate/up must have shape"):
            _load_folded_layer(
                Handle(tensors),
                0,
                expected_experts=2,
                expected_intermediate=6,
                expected_hidden=5,
            )


if __name__ == "__main__":
    unittest.main()
