from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from twen.calibration import (
    CalibrationStopped,
    _activation_lineage,
    _ActivationCorpus,
    _ActivationPart,
    _capture_model,
    _collect_side_shard,
    _compacted_part_metadata,
    _deterministic_indices,
    _InputInspection,
    _PairedActivations,
    _resolve_collection_inputs,
    _save_safetensors_atomic,
    _side_shard_needs_model,
    _validate_activation_file,
    calculate_layer_map,
    calculate_partitions,
    calculate_ridge,
    collect_activations,
)
from twen.config import ArchitectureConfig
from twen.data import PreparedCorpusManifest, PreparedShardEntry, ShardTransaction
from twen.data.prepared import (
    PREPARED_GENERATOR_SOURCE_SHA256,
    _local_prepared_manifest,
    _prepared_dataset_fingerprint,
    _prepared_pipeline_fingerprint,
)
from twen.progress import TaskProgress
from twen.utils import sha256_file


class _TinyLayer(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.mlp = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden)


class _TinyBody(torch.nn.Module):
    def __init__(self, layers: int, hidden_size: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_TinyLayer(hidden_size) for _ in range(layers)]
        )
        self.hidden_size = hidden_size

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
    ) -> torch.Tensor:
        del attention_mask, use_cache
        hidden = torch.nn.functional.one_hot(
            input_ids, num_classes=self.hidden_size
        ).float()
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class _TinyWrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyBody(layers=2, hidden_size=3)


class CalibrationCpuTest(unittest.TestCase):
    def test_preexisting_stop_skips_expensive_validation_for_every_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invocations = (
                lambda stop: collect_activations(
                    "missing.yaml",
                    ["missing.safetensors"],
                    str(root / "collect"),
                    device="cpu",
                    max_samples=2,
                    stop_file=str(stop),
                    progress="never",
                ),
                lambda stop: calculate_layer_map(
                    "missing.yaml",
                    "student",
                    "donor",
                    str(root / "layer-map.json"),
                    stop_file=str(stop),
                    progress="never",
                ),
                lambda stop: calculate_partitions(
                    "missing.yaml",
                    str(root / "channel-map.json"),
                    stop_file=str(stop),
                    progress="never",
                ),
                lambda stop: calculate_ridge(
                    "missing.yaml",
                    "student",
                    "donor",
                    str(root / "adapters.safetensors"),
                    device="cpu",
                    stop_file=str(stop),
                    progress="never",
                ),
            )
            for invoke in invocations:
                stop = root / "STOP"
                stop.touch()
                with patch("twen.calibration.load_train_config") as load_config, self.assertRaises(
                    CalibrationStopped
                ):
                    invoke(stop)
                load_config.assert_not_called()
                self.assertFalse(stop.exists())

    def test_prepared_manifest_resolves_authenticated_fixed_shard_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_hashes = [
                (Path(f"input-{index}.jsonl"), str(index + 1) * 64)
                for index in range(2)
            ]
            pipeline_fingerprint = _prepared_pipeline_fingerprint(
                source_hashes,
                tokenizer_sha256="c" * 64,
                sequence_length=3,
                text_field="text",
                generator_source_sha256=PREPARED_GENERATOR_SOURCE_SHA256,
            )
            entries = []
            for index, shard_id in enumerate(("shard-z", "shard-a")):
                with ShardTransaction(
                    root,
                    shard_id,
                    fingerprint=pipeline_fingerprint,
                    source_fingerprint=f"source-{index}",
                ) as transaction:
                    tensor_path = transaction.work_directory / "tokens.safetensors"
                    tensor_path.write_bytes(f"tensor-{index}".encode())
                    tensor_hash = sha256_file(tensor_path)
                    local = _local_prepared_manifest(
                        shard_id=shard_id,
                        source_path=f"input-{index}.jsonl",
                        source_sha256=str(index + 1) * 64,
                        sequence_count=1,
                        token_count=3,
                        tensors_sha256=tensor_hash,
                        pipeline_fingerprint=pipeline_fingerprint,
                        generator_source_sha256=PREPARED_GENERATOR_SOURCE_SHA256,
                        tokenizer_sha256="c" * 64,
                        sequence_length=3,
                        text_field="text",
                    )
                    (transaction.work_directory / "prepared_manifest.json").write_text(
                        json.dumps(local), encoding="utf-8"
                    )
                    transaction.commit({"kind": "prepared_text"})
                entries.append(
                    PreparedShardEntry(
                        shard_id=shard_id,
                        path=shard_id,
                        source_path=f"input-{index}.jsonl",
                        source_sha256=str(index + 1) * 64,
                        tensors_sha256=tensor_hash,
                        sequence_count=1,
                        token_count=3,
                        global_sample_start=index,
                        global_sample_end=index + 1,
                        global_token_start=index * 3,
                        global_token_end=(index + 1) * 3,
                    )
                )
            dataset_fingerprint = _prepared_dataset_fingerprint(
                pipeline_fingerprint=pipeline_fingerprint,
                generator_source_sha256=PREPARED_GENERATOR_SOURCE_SHA256,
                tokenizer_sha256="c" * 64,
                sequence_length=3,
                text_field="text",
                shards=entries,
            )
            corpus = PreparedCorpusManifest(
                dataset_fingerprint=dataset_fingerprint,
                pipeline_fingerprint=pipeline_fingerprint,
                generator_source_sha256=PREPARED_GENERATOR_SOURCE_SHA256,
                tokenizer_sha256="c" * 64,
                sequence_length=3,
                text_field="text",
                shards=tuple(entries),
            )
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(corpus.to_dict()), encoding="utf-8")

            paths, lineage = _resolve_collection_inputs(None, manifest)
            self.assertEqual(
                [path.parent.name for path in paths],
                ["shard-z", "shard-a"],
            )
            self.assertEqual(lineage["kind"], "prepared_corpus")
            self.assertEqual(lineage["manifest_sha256"], sha256_file(manifest))
            self.assertEqual(
                [item["shard_id"] for item in lineage["shards"]],
                ["shard-z", "shard-a"],
            )
            with self.assertRaisesRegex(ValueError, "exactly one"):
                _resolve_collection_inputs([str(paths[0])], manifest)

    def test_schema_v2_lineage_preserves_both_model_sources(self) -> None:
        sources = {"student": {"manifest_sha256": "a"}, "donor": {"manifest_sha256": "b"}}
        plan = {
            "train_config_path": "config.yaml",
            "train_config_sha256": "config-hash",
            "track": "base",
            "architecture": {},
            "dtype": "float32",
            "device_type": "cpu",
            "sampler": {},
            "sources": sources,
            "ordered_inputs": [],
        }
        payload = {
            "schema_version": 2,
            "plan_fingerprint": "plan",
            "plan_sha256": "hash",
            "lineage": dict(plan),
        }
        lineage = _activation_lineage(payload, plan)
        self.assertEqual(lineage["sources"], sources)
        payload["lineage"] = {**plan, "track": "posttrained"}
        with self.assertRaisesRegex(ValueError, "authenticated PLAN"):
            _activation_lineage(payload, plan)

    def test_global_sampler_is_deterministic_and_budgeted(self) -> None:
        first = _deterministic_indices(100, 11, 3407)
        second = _deterministic_indices(100, 11, 3407)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(first), 11)
        self.assertTrue(np.all(first[:-1] < first[1:]))
        self.assertEqual(len(_deterministic_indices(5, 100, 3407)), 5)

    def test_capture_is_cpu_inference_only_and_supports_empty_parts(self) -> None:
        model = _TinyWrapper()
        batch = {
            "input_ids": torch.tensor([[0, 1, 2], [2, 1, 0]]),
            "attention_mask": torch.ones((2, 3), dtype=torch.long),
        }
        selected = _capture_model(
            model,
            batch,
            sample_coordinates=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
            device="cpu",
            dtype=torch.float32,
        )
        self.assertEqual(
            {name: tuple(value.shape) for name, value in selected.items()},
            {"layers.0": (2, 3), "layers.1": (2, 3)},
        )
        self.assertTrue(all(not value.requires_grad for value in selected.values()))

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "part.safetensors"
            empty = {name: value[:0] for name, value in selected.items()}
            _save_safetensors_atomic(empty, target, metadata={"part": "empty"})
            rows = _validate_activation_file(
                target,
                2,
                expected_rows=0,
                expected_hidden=3,
                expected_metadata={"part": "empty"},
            )
            self.assertEqual(rows, 0)

    def test_collection_skips_model_forward_for_unsampled_microbatches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "tokens.safetensors"
            _save_safetensors_atomic(
                {
                    "input_ids": torch.tensor([[0, 1, 2], [2, 1, 0]]),
                    "attention_mask": torch.ones((2, 3), dtype=torch.long),
                },
                input_path,
            )
            item = _InputInspection(
                index=0,
                path=input_path,
                sha256=sha256_file(input_path),
                sequence_count=2,
                sequence_length=3,
                valid_tokens=6,
                global_valid_start=0,
                global_valid_end=6,
                selected_flat_positions=np.empty((0,), dtype=np.int64),
            )
            with ShardTransaction(
                root / "activations",
                item.shard_id,
                fingerprint="side-fingerprint",
                source_fingerprint=item.sha256,
            ) as transaction, TaskProgress(
                total=2,
                description="fixture",
                unit="part",
                mode="never",
            ) as progress:
                _collect_side_shard(
                    None,
                    item,
                    transaction,
                    plan={"plan_fingerprint": "plan-fingerprint"},
                    label="student",
                    expected_layers=2,
                    expected_hidden=3,
                    batch_size=1,
                    device="cpu",
                    dtype=torch.float32,
                    stop_file=None,
                    progress_bar=progress,
                )
            shard = root / "activations" / item.shard_id
            self.assertFalse((shard / "parts").exists())
            self.assertEqual(
                _validate_activation_file(
                    shard / "activations.safetensors",
                    expected_layers=2,
                    expected_rows=0,
                    expected_hidden=3,
                ),
                0,
            )
            marker = json.loads((shard / "COMPLETE").read_text())
            self.assertEqual(len(marker["metadata"]["parts"]), 1)
            self.assertEqual(
                marker["metadata"]["parts"][0]["artifact_kind"],
                "pre_ffn_activation_shard",
            )

    def test_collection_compacts_microbatch_journal_in_sample_order(self) -> None:
        from safetensors import safe_open

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "tokens.safetensors"
            ids = torch.tensor([[0, 1, 2], [2, 1, 0]])
            mask = torch.ones((2, 3), dtype=torch.long)
            _save_safetensors_atomic(
                {"input_ids": ids, "attention_mask": mask}, input_path
            )
            item = _InputInspection(
                index=0,
                path=input_path,
                sha256=sha256_file(input_path),
                sequence_count=2,
                sequence_length=3,
                valid_tokens=6,
                global_valid_start=0,
                global_valid_end=6,
                selected_flat_positions=np.asarray([1, 5], dtype=np.int64),
            )
            model = _TinyWrapper()
            expected = _capture_model(
                model,
                {"input_ids": ids, "attention_mask": mask},
                sample_coordinates=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
                device="cpu",
                dtype=torch.float32,
            )
            with ShardTransaction(
                root / "activations",
                item.shard_id,
                fingerprint="side-fingerprint",
                source_fingerprint=item.sha256,
            ) as transaction, TaskProgress(
                total=2,
                description="fixture",
                unit="part",
                mode="never",
            ) as progress:
                _collect_side_shard(
                    model,
                    item,
                    transaction,
                    plan={"plan_fingerprint": "plan-fingerprint"},
                    label="student",
                    expected_layers=2,
                    expected_hidden=3,
                    batch_size=1,
                    device="cpu",
                    dtype=torch.float32,
                    stop_file=None,
                    progress_bar=progress,
                )
            compacted = root / "activations" / item.shard_id / "activations.safetensors"
            with safe_open(compacted, framework="pt", device="cpu") as handle:
                tensor_names = tuple(handle.keys())
                actual = {name: handle.get_tensor(name) for name in tensor_names}
            self.assertEqual(set(actual), set(expected))
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name])

    def test_compact_resume_does_not_require_reloading_the_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "tokens.safetensors"
            _save_safetensors_atomic(
                {
                    "input_ids": torch.tensor([[0, 1, 2], [2, 1, 0]]),
                    "attention_mask": torch.ones((2, 3), dtype=torch.long),
                },
                input_path,
            )
            item = _InputInspection(
                index=0,
                path=input_path,
                sha256=sha256_file(input_path),
                sequence_count=2,
                sequence_length=3,
                valid_tokens=6,
                global_valid_start=0,
                global_valid_end=6,
                selected_flat_positions=np.asarray([1, 5], dtype=np.int64),
            )
            with ShardTransaction(
                root / "activations",
                item.shard_id,
                fingerprint="side-fingerprint",
                source_fingerprint=item.sha256,
            ) as transaction:
                compacted = transaction.work_directory / "activations.safetensors"
                _save_safetensors_atomic(
                    {
                        "layers.0": torch.randn(2, 3),
                        "layers.1": torch.randn(2, 3),
                    },
                    compacted,
                    metadata=_compacted_part_metadata(
                        plan_fingerprint="plan-fingerprint",
                        label="student",
                        item=item,
                    ),
                )
                self.assertFalse(
                    _side_shard_needs_model(
                        item,
                        transaction,
                        plan_fingerprint="plan-fingerprint",
                        label="student",
                        expected_layers=2,
                        expected_hidden=3,
                        batch_size=1,
                    )
                )

                compacted.unlink()
                self.assertTrue(
                    _side_shard_needs_model(
                        item,
                        transaction,
                        plan_fingerprint="plan-fingerprint",
                        label="student",
                        expected_layers=2,
                        expected_hidden=3,
                        batch_size=1,
                    )
                )

    def test_layer_map_resumes_inside_a_cka_score_row(self) -> None:
        from twen.modeling import linear_cka as real_linear_cka

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            student_file = root / "student.safetensors"
            donor_file = root / "donor.safetensors"
            _save_safetensors_atomic(
                {
                    "layers.0": torch.randn(5, 3),
                    "layers.1": torch.randn(5, 3),
                },
                student_file,
            )
            _save_safetensors_atomic(
                {
                    "layers.0": torch.randn(5, 4),
                    "layers.1": torch.randn(5, 4),
                    "layers.2": torch.randn(5, 4),
                },
                donor_file,
            )
            student_source = SimpleNamespace(model_id="student", local_path="student")
            donor_source = SimpleNamespace(model_id="donor", local_path="donor")
            sources = {
                "student": {"model_id": "student"},
                "donor": {"model_id": "donor"},
            }
            student_part = _ActivationPart(
                "part-0", student_file, 5, 0, 1, "selection", "student-sha"
            )
            donor_part = _ActivationPart(
                "part-0", donor_file, 5, 0, 1, "selection", "donor-sha"
            )
            paired = _PairedActivations(
                student=_ActivationCorpus(
                    "student",
                    student_file,
                    "student-source",
                    {"sources": sources},
                    (student_part,),
                    5,
                ),
                donor=_ActivationCorpus(
                    "donor",
                    donor_file,
                    "donor-source",
                    {"sources": sources},
                    (donor_part,),
                    5,
                ),
                parts=(("part-0", student_file, donor_file),),
                total_samples=5,
                fingerprint="paired-fingerprint",
            )
            config = SimpleNamespace(
                architecture=SimpleNamespace(
                    student_layers=2,
                    donor_layers=3,
                    student_hidden_size=3,
                    donor_hidden_size=4,
                ),
                sources=SimpleNamespace(
                    backbone=student_source,
                    donor=donor_source,
                ),
            )
            audit = SimpleNamespace(
                student_layer_types=("linear", "full"),
                donor_layer_types=("linear", "linear", "full"),
            )
            stop = root / "STOP"
            output = root / "layer_map.json"
            first_calls = 0

            def stop_after_first(x, y):
                nonlocal first_calls
                first_calls += 1
                value = real_linear_cka(x, y)
                stop.write_text("stop\n")
                return value

            common = (
                patch("twen.calibration.load_train_config", return_value=config),
                patch("twen.calibration._paired_activation_shards", return_value=paired),
                patch(
                    "twen.calibration._model_lineage",
                    side_effect=lambda source: sources[source.model_id],
                ),
                patch("twen.modeling.audit_source_configs", return_value=audit),
            )
            with common[0], common[1], common[2], common[3], patch(
                "twen.modeling.linear_cka", side_effect=stop_after_first
            ), self.assertRaises(CalibrationStopped):
                calculate_layer_map(
                    "config.yaml",
                    str(student_file),
                    str(donor_file),
                    str(output),
                    stop_file=str(stop),
                )
            self.assertEqual(first_calls, 1)
            self.assertFalse(stop.exists())

            resumed_calls = 0

            def count_resumed(x, y):
                nonlocal resumed_calls
                resumed_calls += 1
                return real_linear_cka(x, y)

            common = (
                patch("twen.calibration.load_train_config", return_value=config),
                patch("twen.calibration._paired_activation_shards", return_value=paired),
                patch(
                    "twen.calibration._model_lineage",
                    side_effect=lambda source: sources[source.model_id],
                ),
                patch("twen.modeling.audit_source_configs", return_value=audit),
            )
            with common[0], common[1], common[2], common[3], patch(
                "twen.modeling.linear_cka", side_effect=count_resumed
            ):
                calculate_layer_map(
                    "config.yaml",
                    str(student_file),
                    str(donor_file),
                    str(output),
                    stop_file=str(stop),
                )
            self.assertEqual(resumed_calls, 5)
            self.assertEqual(len(json.loads(output.read_text())["pairs"]), 2)

    def test_ridge_batches_small_parts_and_records_numeric_mode(self) -> None:
        from twen.modeling import BidirectionalRidgeStats

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            student_parts = []
            donor_parts = []
            paired_parts = []
            for index, (part_id, rows) in enumerate(
                (
                    ("shard-a/part-000000", 2),
                    ("shard-a/part-000001", 2),
                    ("shard-b/part-000000", 2),
                )
            ):
                student_file = root / f"student-{index}.safetensors"
                donor_file = root / f"donor-{index}.safetensors"
                _save_safetensors_atomic(
                    {"layers.0": torch.randn(rows, 2)}, student_file
                )
                _save_safetensors_atomic(
                    {"layers.0": torch.randn(rows, 3)}, donor_file
                )
                student_parts.append(
                    _ActivationPart(part_id, student_file, rows, 0, 1, str(index), "s")
                )
                donor_parts.append(
                    _ActivationPart(part_id, donor_file, rows, 0, 1, str(index), "d")
                )
                paired_parts.append((part_id, student_file, donor_file))

            student_source = SimpleNamespace(model_id="student")
            donor_source = SimpleNamespace(model_id="donor")
            sources = {
                "student": {"model_id": "student"},
                "donor": {"model_id": "donor"},
            }
            paired = _PairedActivations(
                student=_ActivationCorpus(
                    "student",
                    root / "student-manifest.json",
                    "student-source",
                    {"sources": sources},
                    tuple(student_parts),
                    6,
                    legacy=True,
                ),
                donor=_ActivationCorpus(
                    "donor",
                    root / "donor-manifest.json",
                    "donor-source",
                    {"sources": sources},
                    tuple(donor_parts),
                    6,
                    legacy=True,
                ),
                parts=tuple(paired_parts),
                total_samples=6,
                fingerprint="paired-fingerprint",
            )
            layer_map = root / "layer-map.json"
            layer_map.write_text('{"student_to_donor": [0]}\n', encoding="utf-8")
            architecture = ArchitectureConfig(
                student_hidden_size=2,
                student_layers=1,
                donor_hidden_size=3,
                donor_layers=1,
                layer_map_path=str(layer_map),
            )
            config = SimpleNamespace(
                architecture=architecture,
                sources=SimpleNamespace(backbone=student_source, donor=donor_source),
                track="base",
            )
            output = root / "adapters.safetensors"
            update_rows = []
            real_update = BidirectionalRidgeStats.update

            def record_update(stats, small, large):
                update_rows.append(int(small.shape[0]))
                return real_update(stats, small, large)

            with (
                patch("twen.calibration.load_train_config", return_value=config),
                patch("twen.calibration._paired_activation_shards", return_value=paired),
                patch(
                    "twen.calibration._model_lineage",
                    side_effect=lambda source: sources[source.model_id],
                ),
                patch("twen.training.builder.load_layer_mapping", return_value=(0,)),
                patch.object(BidirectionalRidgeStats, "update", new=record_update),
            ):
                calculate_ridge(
                    "config.yaml",
                    "student",
                    "donor",
                    str(output),
                    device="cpu",
                    accumulation_dtype="float32",
                    batch_samples=3,
                    progress="never",
                )
            self.assertEqual(update_rows, [4, 2])
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(metadata["ridge"]["accumulation_dtype"], "float32")
            self.assertEqual(metadata["ridge"]["batch_samples"], 3)
            self.assertEqual(metadata["layers"]["0"]["samples"], 6)


if __name__ == "__main__":
    unittest.main()
