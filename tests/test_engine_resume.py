"""Resume-control tests that never construct an optimizer or model."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from twen.config import LossConfig
from twen.data.cursor import (
    DatasetLayout,
    DeterministicCooldownCursor,
    DeterministicGlobalCursor,
)
from twen.runtime.checkpoint import CheckpointError
from twen.runtime.state import DataCursor, RNGState, TrainerState
from twen.training.engine import (
    PROFILE_STEP_UNIT,
    _advance_profiler_after_microbatch,
    _build_trainer_state,
    _checkpoint_phase_log_fields,
    _checkpoint_policy_requires_update,
    _effective_activation_checkpointing,
    _is_read_only_completed_resume,
    _load_or_initialize,
    _loss_metric_aliases,
    _named_learning_rates,
    _set_activation_checkpointing,
    _set_selective_activation_checkpointing,
)


def _inputs() -> tuple[Any, Any, Any]:
    config = SimpleNamespace(
        run_id="resume-test",
        stage="dense-oracle",
        data=SimpleNamespace(shuffle_seed=123, global_batch_tokens=4096),
        losses=LossConfig(),
    )
    report = SimpleNamespace(
        config_fingerprint="a" * 64,
        data_fingerprint="b" * 64,
        batch=SimpleNamespace(
            world_size=2,
            micro_batch_tokens_per_rank=1024,
            gradient_accumulation_steps=2,
            global_batch_tokens=4096,
        ),
    )
    store = SimpleNamespace(
        layout=DatasetLayout.from_shards(
            (("shard-a", 8),),
            fingerprint="dataset-v1",
        )
    )
    return config, report, store


class _FailingManager:
    def __init__(self, root: Path, message: str = "no complete checkpoint") -> None:
        self.root = root
        self.message = message
        self.load_calls = 0

    def load(self, *args: Any, **kwargs: Any) -> Any:
        self.load_calls += 1
        raise CheckpointError(self.message)


class EngineResumeControlTest(unittest.TestCase):
    def test_activation_checkpoint_toggle_clears_embedding_hooks_idempotently(self) -> None:
        class Model:
            checkpointing = False
            input_hooks = 0

            def disable_input_require_grads(self) -> None:
                self.input_hooks = 0

            def enable_input_require_grads(self) -> None:
                self.input_hooks += 1

            def gradient_checkpointing_enable(self) -> None:
                self.checkpointing = True
                self.enable_input_require_grads()

            def gradient_checkpointing_disable(self) -> None:
                self.checkpointing = False

        model = Model()
        _set_activation_checkpointing(model, True)
        _set_activation_checkpointing(model, True)
        self.assertTrue(model.checkpointing)
        self.assertEqual(model.input_hooks, 1)
        _set_activation_checkpointing(model, False)
        self.assertFalse(model.checkpointing)
        self.assertEqual(model.input_hooks, 0)

    def test_selective_checkpointing_toggles_exact_layers_without_duplicate_hooks(self) -> None:
        class Layer:
            gradient_checkpointing = False

        class Model:
            def __init__(self) -> None:
                self.model = SimpleNamespace(layers=[Layer() for _ in range(24)])
                self.input_hooks = 0
                self.enable_calls = 0
                self.disable_calls = 0

            def disable_input_require_grads(self) -> None:
                self.input_hooks = 0

            def gradient_checkpointing_enable(self) -> None:
                self.enable_calls += 1
                self.input_hooks += 1
                for layer in self.model.layers:
                    layer.gradient_checkpointing = True

            def gradient_checkpointing_disable(self) -> None:
                self.disable_calls += 1
                for layer in self.model.layers:
                    layer.gradient_checkpointing = False

        model = Model()
        ordinary = (0, 8, 15, 23)
        _set_selective_activation_checkpointing(model, ordinary)
        self.assertEqual(model.enable_calls, 1)
        self.assertEqual(model.input_hooks, 1)
        self.assertEqual(
            tuple(
                index
                for index, layer in enumerate(model.model.layers)
                if layer.gradient_checkpointing
            ),
            ordinary,
        )

        _set_selective_activation_checkpointing(model, tuple(range(24)))
        self.assertEqual(model.enable_calls, 1)
        self.assertEqual(model.input_hooks, 1)
        self.assertTrue(all(layer.gradient_checkpointing for layer in model.model.layers))

        _set_selective_activation_checkpointing(model, ())
        self.assertEqual(model.disable_calls, 1)
        self.assertEqual(model.input_hooks, 0)
        self.assertFalse(any(layer.gradient_checkpointing for layer in model.model.layers))

        _set_selective_activation_checkpointing(model, ordinary)
        self.assertEqual(model.enable_calls, 2)
        self.assertEqual(model.input_hooks, 1)

    def test_alignment_only_checkpoint_policy_changes_only_selected_steps(self) -> None:
        config = SimpleNamespace(
            runtime=SimpleNamespace(
                activation_checkpointing=True,
                activation_checkpointing_on_alignment_only=True,
            )
        )
        self.assertFalse(_effective_activation_checkpointing(config, align_hidden=False))
        self.assertTrue(_effective_activation_checkpointing(config, align_hidden=True))
        config.runtime.activation_checkpoint_layer_count = 4
        self.assertTrue(_effective_activation_checkpointing(config, align_hidden=False))
        config.runtime.hidden_alignment_activation_checkpoint_layer_count = 0
        self.assertFalse(_effective_activation_checkpointing(config, align_hidden=True))
        config.runtime.hidden_alignment_activation_checkpoint_layer_count = 8
        self.assertTrue(_effective_activation_checkpointing(config, align_hidden=True))
        config.runtime.activation_checkpointing_on_alignment_only = False
        self.assertTrue(_effective_activation_checkpointing(config, align_hidden=False))
        config.runtime.activation_checkpoint_layer_count = 0
        self.assertFalse(_effective_activation_checkpointing(config, align_hidden=False))

    def test_phase_switch_reconfigures_when_only_inner_policy_changes(self) -> None:
        outer = (0, 3, 7, 10, 13, 16, 20, 23)
        ordinary_inner: tuple[int, ...] = ()
        alignment_inner = (1, 2, 4, 5)

        self.assertTrue(
            _checkpoint_policy_requires_update(
                outer,
                ordinary_inner,
                outer,
                alignment_inner,
            )
        )
        self.assertFalse(
            _checkpoint_policy_requires_update(
                outer,
                alignment_inner,
                outer,
                alignment_inner,
            )
        )

    def test_checkpoint_phase_log_contract_records_configured_and_effective_policy(self) -> None:
        config = SimpleNamespace(
            runtime=SimpleNamespace(
                activation_checkpoint_layer_count=0,
                hidden_alignment_activation_checkpoint_layer_count=8,
                dense_transfer_execution="expanded",
                dense_transfer_checkpoint_layer_count=0,
                hidden_alignment_dense_transfer_checkpoint_layer_count=16,
            )
        )
        ordinary = _checkpoint_phase_log_fields(
            config,
            align_hidden=False,
            outer_checkpoint_layer_indices=(),
            inner_checkpoint_layer_indices=(),
        )
        alignment_outer = (0, 3, 7, 10, 13, 16, 20, 23)
        alignment_inner = (1, 2, 4, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 19, 21, 22)
        alignment = _checkpoint_phase_log_fields(
            config,
            align_hidden=True,
            outer_checkpoint_layer_indices=alignment_outer,
            inner_checkpoint_layer_indices=alignment_inner,
        )

        self.assertEqual(
            ordinary,
            {
                "hidden_alignment_step": False,
                "activation_checkpointing_effective": False,
                "activation_checkpoint_layer_count_configured": 0,
                "activation_checkpoint_layer_count_effective": 0,
                "activation_checkpoint_layer_indices_effective": [],
                "dense_transfer_execution": "expanded",
                "dense_transfer_checkpoint_layer_count_configured": 0,
                "dense_transfer_token_checkpoint_layer_count_effective": 0,
                "dense_transfer_token_checkpoint_layer_indices_effective": [],
            },
        )
        self.assertEqual(alignment["activation_checkpoint_layer_count_configured"], 8)
        self.assertEqual(alignment["activation_checkpoint_layer_count_effective"], 8)
        self.assertEqual(
            alignment["activation_checkpoint_layer_indices_effective"],
            list(alignment_outer),
        )
        self.assertEqual(alignment["dense_transfer_checkpoint_layer_count_configured"], 16)
        self.assertEqual(
            alignment["dense_transfer_token_checkpoint_layer_count_effective"],
            16,
        )
        self.assertEqual(
            alignment["dense_transfer_token_checkpoint_layer_indices_effective"],
            list(alignment_inner),
        )

    def test_trainer_checkpoint_state_records_explicit_mtp_weight(self) -> None:
        config, report, _store = _inputs()
        config.losses.mtp = 0.3

        state = _build_trainer_state(config, report)

        self.assertEqual(state.loss_weights["mtp"], 0.3)

    def test_learning_rate_snapshot_is_read_only_and_preserves_group_names(self) -> None:
        optimizer = SimpleNamespace(
            param_groups=[
                {"name": "adapters", "lr": 2.0e-4},
                {"name": "router", "lr": 1.0e-3},
            ]
        )
        self.assertEqual(
            _named_learning_rates(optimizer),
            (("adapters", 2.0e-4), ("router", 1.0e-3)),
        )
        self.assertEqual(optimizer.param_groups[0]["lr"], 2.0e-4)

    def test_profiler_schedule_advances_in_microbatch_units(self) -> None:
        class Profiler:
            steps = 0

            def step(self) -> None:
                self.steps += 1

        profiler = Profiler()
        for _ in range(3):
            _advance_profiler_after_microbatch(profiler)
        self.assertEqual(PROFILE_STEP_UNIT, "microbatch")
        self.assertEqual(profiler.steps, 3)

    def test_loss_aliases_include_only_evaluated_components(self) -> None:
        metrics = {
            "ntp": 1.0,
            "mtp": 1.5,
            "teacher_kd": 2.0,
            "anchor_kl": 3.0,
            "hidden_alignment": 4.0,
            "load_balance": 5.0,
            "router_z": 6.0,
            "dense_oracle": 7.0,
            "router_supervision": 8.0,
        }
        base = _loss_metric_aliases(
            metrics,
            include_anchor=False,
            include_hidden=False,
            include_sparse=False,
            include_dense=False,
        )
        self.assertEqual(base, {"ntp_loss": 1.0, "teacher_kd_loss": 2.0})

        with_mtp = _loss_metric_aliases(
            metrics,
            include_anchor=False,
            include_hidden=False,
            include_sparse=False,
            include_dense=False,
            include_mtp=True,
        )
        self.assertEqual(
            with_mtp,
            {"ntp_loss": 1.0, "mtp_loss": 1.5, "teacher_kd_loss": 2.0},
        )

        all_components = _loss_metric_aliases(
            metrics,
            include_anchor=True,
            include_hidden=True,
            include_sparse=True,
            include_dense=True,
        )
        self.assertEqual(
            set(all_components),
            {
                "ntp_loss",
                "teacher_kd_loss",
                "anchor_kl_loss",
                "hidden_alignment_loss",
                "load_balance_loss",
                "router_z_loss",
                "dense_oracle_loss",
                "router_supervision_loss",
            },
        )

    def test_auto_resume_attempts_only_one_load_and_explains_first_launch(self) -> None:
        config, report, store = _inputs()
        with tempfile.TemporaryDirectory() as directory:
            manager = _FailingManager(Path(directory))
            with self.assertRaisesRegex(CheckpointError, "first launch must use --resume none"):
                _load_or_initialize(
                    manager,  # type: ignore[arg-type]
                    {},
                    config,  # type: ignore[arg-type]
                    report,  # type: ignore[arg-type]
                    store,  # type: ignore[arg-type]
                    resume="auto",
                    fork_from=None,
                )
            self.assertEqual(manager.load_calls, 1)

    def test_auto_resume_does_not_mislabel_a_corrupt_committed_directory(self) -> None:
        config, report, store = _inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "step-000000000001-periodic").mkdir()
            manager = _FailingManager(root, "hash mismatch")
            with self.assertRaisesRegex(CheckpointError, "hash mismatch"):
                _load_or_initialize(
                    manager,  # type: ignore[arg-type]
                    {},
                    config,  # type: ignore[arg-type]
                    report,  # type: ignore[arg-type]
                    store,  # type: ignore[arg-type]
                    resume="auto",
                    fork_from=None,
                )
            self.assertEqual(manager.load_calls, 1)

    def test_loaded_state_reconfigures_batch_geometry_and_returns_checkpoint(self) -> None:
        config, report, store = _inputs()
        global_cursor = DeterministicGlobalCursor(store.layout, seed=123)
        global_cursor.commit(global_batch_samples=2, token_count=2048)
        loaded = SimpleNamespace(
            trainer_state=TrainerState(
                run_id="resume-test",
                stage="dense-oracle",
                global_step=1,
                committed_tokens=2048,
                world_size=1,
                global_batch_tokens=4096,
                micro_batch_tokens_per_rank=4096,
            ),
            data_cursor=DataCursor.from_global_cursor_state(global_cursor.state_dict()),
            rng_state=RNGState.capture(),
            metadata={"kind": "periodic", "tag": None},
            path=Path("checkpoint"),
        )

        class Manager:
            root = Path(".")

            def load(self, *args: Any, **kwargs: Any) -> Any:
                return loaded

        state, cursor, returned = _load_or_initialize(
            Manager(),  # type: ignore[arg-type]
            {},
            config,  # type: ignore[arg-type]
            report,  # type: ignore[arg-type]
            store,  # type: ignore[arg-type]
            resume="auto",
            fork_from=None,
        )
        self.assertIs(returned, loaded)
        self.assertEqual(state.world_size, 2)
        self.assertEqual(state.gradient_accumulation_steps, 2)
        self.assertEqual(cursor.next_global_sample, 2)
        self.assertEqual(cursor.committed_tokens, 2048)

    def test_loaded_cursor_rejects_inconsistent_nested_global_state(self) -> None:
        config, report, store = _inputs()
        global_cursor = DeterministicGlobalCursor(store.layout, seed=123)
        global_cursor.commit(global_batch_samples=2, token_count=2048)
        data_cursor = DataCursor.from_global_cursor_state(global_cursor.state_dict())
        data_cursor.extra["committed_tokens"] = 1024
        loaded = SimpleNamespace(
            trainer_state=TrainerState(
                run_id="resume-test",
                stage="dense-oracle",
                global_step=1,
                committed_tokens=2048,
                world_size=1,
                global_batch_tokens=4096,
                micro_batch_tokens_per_rank=4096,
            ),
            data_cursor=data_cursor,
            rng_state=RNGState.capture(),
            metadata={"kind": "periodic", "tag": None},
            path=Path("checkpoint"),
        )

        class Manager:
            root = Path(".")

            def load(self, *args: Any, **kwargs: Any) -> Any:
                return loaded

        with self.assertRaisesRegex(ValueError, "token position disagrees"):
            _load_or_initialize(
                Manager(),  # type: ignore[arg-type]
                {},
                config,  # type: ignore[arg-type]
                report,  # type: ignore[arg-type]
                store,  # type: ignore[arg-type]
                resume="auto",
                fork_from=None,
            )

    def test_quality_cooldown_resume_restores_both_cursors_and_phase(self) -> None:
        config, report, store = _inputs()
        config.data.quality_cooldown_start_tokens = 1024
        cooldown_store = SimpleNamespace(
            layout=DatasetLayout.from_shards(
                (("quality-shard", 8),), fingerprint="quality-dataset-v1"
            )
        )
        global_cursor = DeterministicCooldownCursor(
            store.layout,
            cooldown_store.layout,
            seed=123,
            cooldown_start_tokens=1024,
        )
        global_cursor.commit(global_batch_samples=2, token_count=2048)
        global_cursor.commit(global_batch_samples=2, token_count=2048)
        loaded = SimpleNamespace(
            trainer_state=TrainerState(
                run_id="resume-test",
                stage="dense-oracle",
                global_step=2,
                committed_tokens=4096,
                world_size=1,
                global_batch_tokens=4096,
                micro_batch_tokens_per_rank=4096,
            ),
            data_cursor=DataCursor.from_global_cursor_state(global_cursor.state_dict()),
            rng_state=RNGState.capture(),
            metadata={"kind": "periodic", "tag": None},
            path=Path("checkpoint"),
        )

        class Manager:
            root = Path(".")

            def load(self, *args: Any, **kwargs: Any) -> Any:
                return loaded

        state, cursor, returned = _load_or_initialize(
            Manager(),  # type: ignore[arg-type]
            {},
            config,  # type: ignore[arg-type]
            report,  # type: ignore[arg-type]
            store,  # type: ignore[arg-type]
            cooldown_store=cooldown_store,  # type: ignore[arg-type]
            resume="auto",
            fork_from=None,
        )

        self.assertIs(returned, loaded)
        self.assertIsInstance(cursor, DeterministicCooldownCursor)
        assert isinstance(cursor, DeterministicCooldownCursor)
        self.assertEqual(cursor.active_phase, "cooldown")
        self.assertEqual(cursor.committed_tokens, state.committed_tokens)
        self.assertEqual(cursor.next_global_sample, 4)
        self.assertEqual(cursor.phase_next_global_sample, 2)

    def test_quality_cooldown_requires_second_store_before_model_resume(self) -> None:
        config, report, store = _inputs()
        config.data.quality_cooldown_start_tokens = 1024
        with self.assertRaisesRegex(RuntimeError, "requires a cooldown store"):
            _load_or_initialize(
                _FailingManager(Path(".")),  # type: ignore[arg-type]
                {},
                config,  # type: ignore[arg-type]
                report,  # type: ignore[arg-type]
                store,  # type: ignore[arg-type]
                resume="none",
                fork_from=None,
            )

    def test_only_terminal_complete_milestone_is_read_only(self) -> None:
        state = TrainerState(
            run_id="resume-test",
            stage="dense-oracle",
            committed_tokens=100,
        )
        complete = SimpleNamespace(metadata={"kind": "milestone", "tag": "complete"})
        periodic = SimpleNamespace(metadata={"kind": "periodic", "tag": None})
        self.assertTrue(
            _is_read_only_completed_resume(state, complete, max_tokens=100)  # type: ignore[arg-type]
        )
        self.assertFalse(
            _is_read_only_completed_resume(state, periodic, max_tokens=100)  # type: ignore[arg-type]
        )
        self.assertFalse(_is_read_only_completed_resume(state, None, max_tokens=100))
        self.assertFalse(
            _is_read_only_completed_resume(state, complete, max_tokens=101)  # type: ignore[arg-type]
        )

    def test_fork_from_resolves_a_cwd_relative_external_checkpoint(self) -> None:
        config, report, store = _inputs()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            source = Path(directory) / "step-000000000001-periodic"
            source.mkdir()
            relative = source.relative_to(Path.cwd())

            class Manager:
                root = Path("new-run")
                checkpoint: Path | None = None
                stateful_keys: tuple[str, ...] | None = None

                def load(self, stateful: Any, checkpoint: Path) -> Any:
                    self.checkpoint = checkpoint
                    self.stateful_keys = tuple(stateful)
                    return None

            manager = Manager()
            _load_or_initialize(
                manager,  # type: ignore[arg-type]
                {"model": object()},
                config,  # type: ignore[arg-type]
                report,  # type: ignore[arg-type]
                store,  # type: ignore[arg-type]
                resume="none",
                fork_from=str(relative),
            )
            self.assertEqual(manager.checkpoint, source.resolve())
            self.assertEqual(manager.stateful_keys, ("model",))


if __name__ == "__main__":
    unittest.main()
