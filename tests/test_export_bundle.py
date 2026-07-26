from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from twen.evaluation import _verify_export_bundle
from twen.export import (
    _bundle_inventory,
    _copy_tokenizer_files,
    _load_trainable_only,
    _prepare_bundle_directory,
)
from twen.runtime.checkpoint import CheckpointManager
from twen.runtime.state import DataCursor, RNGState, TrainerState
from twen.utils import atomic_write_json, sha256_file


class _ExportDeltaModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(11, 3)
        self.lm_head = torch.nn.Linear(3, 11, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.delta = torch.nn.Linear(3, 3, bias=False)
        self.register_buffer("channel_indices", torch.arange(4))
        self.embed_tokens.weight.requires_grad_(False)


class _LegacyExportState:
    """Model payload shape emitted before trainable-only alias filtering."""

    def __init__(self, model: _ExportDeltaModel) -> None:
        self.model = model

    def state_dict(self) -> dict[str, Any]:
        return {
            "lm_head.weight": self.model.lm_head.weight.detach(),
            "channel_indices": self.model.channel_indices.detach(),
            "delta.weight": self.model.delta.weight.detach(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        raise AssertionError("legacy state is save-only in this test")


class ExportBundleTest(unittest.TestCase):
    def test_bundle_inventory_hashes_all_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            (source / "tokenizer.json").write_text("{}\n")
            (source / "tokenizer_config.json").write_text("{}\n")
            expected = {
                "model.safetensors",
                "config.json",
                "twen_manifest.json",
                "tokenizer.json",
                "tokenizer_config.json",
            }
            _prepare_bundle_directory(output, expected)
            (output / "model.safetensors").write_bytes(b"model")
            (output / "config.json").write_text("{}\n")
            copied = _copy_tokenizer_files(source, output)
            inventory = _bundle_inventory(
                output,
                {"model.safetensors", "config.json", *copied},
            )
            self.assertEqual(set(inventory), expected - {"twen_manifest.json"})
            self.assertTrue(all(len(item["sha256"]) == 64 for item in inventory.values()))

    def test_bundle_directory_rejects_stale_unexpected_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "stale-tokenizer.json").write_text("{}\n")
            with self.assertRaisesRegex(RuntimeError, "unexpected stale entry"):
                _prepare_bundle_directory(output, {"model.safetensors"})

    def test_complete_marker_authenticates_the_full_bundle_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.safetensors"
            artifact.write_bytes(b"model")
            manifest = {
                "artifact_sha256": sha256_file(artifact),
                "bundle_files": {
                    artifact.name: {
                        "size": artifact.stat().st_size,
                        "sha256": sha256_file(artifact),
                    }
                },
            }
            manifest_path = root / "twen_manifest.json"
            atomic_write_json(manifest_path, manifest)
            (root / "COMPLETE").write_text(f"{sha256_file(manifest_path)}\n")
            verified_root, verified_manifest = _verify_export_bundle(root)
            self.assertEqual(verified_root, root.resolve())
            self.assertEqual(verified_manifest, manifest)

    def test_export_loader_reads_trainable_delta_from_a_legacy_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            model = _ExportDeltaModel()
            original_delta = model.delta.weight.detach().clone()
            manager = CheckpointManager(run_dir, backend="dcp")
            checkpoint = manager.save(
                {"model": _LegacyExportState(model)},
                trainer_state=TrainerState(
                    run_id="legacy-export",
                    stage="dense-oracle",
                    global_batch_tokens=16,
                    micro_batch_tokens_per_rank=16,
                ),
                data_cursor=DataCursor(),
                rng_state=RNGState.capture(),
                critical_fingerprint="config",
                data_fingerprint="data",
            )

            with torch.no_grad():
                model.delta.weight.zero_()
                model.embed_tokens.weight.fill_(5)
                model.channel_indices.fill_(6)
            loaded = _load_trainable_only(
                model,
                SimpleNamespace(
                    checkpoint=SimpleNamespace(output_dir=str(run_dir)),
                    run_id="legacy-export",
                    stage="dense-oracle",
                    data=SimpleNamespace(global_batch_tokens=16),
                ),
                SimpleNamespace(
                    config_fingerprint="config",
                    data_fingerprint="data",
                ),
                str(checkpoint),
            )
            self.assertEqual(loaded, checkpoint)
            self.assertTrue(torch.equal(model.delta.weight, original_delta))
            self.assertEqual(torch.count_nonzero(model.embed_tokens.weight != 5), 0)
            self.assertEqual(torch.count_nonzero(model.channel_indices != 6), 0)


if __name__ == "__main__":
    unittest.main()
