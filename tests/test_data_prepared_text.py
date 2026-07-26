from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from twen.data.prepared import (
    PREPARED_GENERATOR_SOURCE_SHA256,
    PreparedCorpusManifest,
    PreparedShardEntry,
)
from twen.data.prepared_text import (
    PREPARED_TEXT_REQUIRED_TENSORS,
    PreparedTextSchemaError,
    PreparedTextShardDataset,
)
from twen.training.data import PreparedTextRecordStore, move_prepared_text_batch


def _write_prepared_fixture(
    root: Path,
    *,
    tensors: dict[str, torch.Tensor] | None = None,
) -> tuple[Path, dict[str, torch.Tensor]]:
    shard = root / "source-000"
    shard.mkdir()
    if tensors is None:
        input_ids = torch.tensor(
            [[10, 11, 12, 13, 13], [20, 21, 21, 23, 24]],
            dtype=torch.int64,
        )
        labels = torch.tensor(
            [[10, 11, 12, 13, -100], [20, 21, -100, 23, 24]],
            dtype=torch.int64,
        )
        attention_mask = torch.tensor(
            [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]],
            dtype=torch.bool,
        )
        tensors = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
    save_file(tensors, shard / "tokens.safetensors")
    entry = PreparedShardEntry(
        shard_id="source-000",
        path="source-000",
        source_path="source.jsonl",
        source_sha256="1" * 64,
        tensors_sha256="2" * 64,
        sequence_count=2,
        token_count=9,
        global_sample_start=0,
        global_sample_end=2,
        global_token_start=0,
        global_token_end=9,
    )
    manifest = PreparedCorpusManifest(
        dataset_fingerprint="3" * 64,
        pipeline_fingerprint="4" * 64,
        generator_source_sha256=PREPARED_GENERATOR_SOURCE_SHA256,
        tokenizer_sha256="5" * 64,
        sequence_length=5,
        text_field="text",
        shards=(entry,),
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path, tensors


def test_prepared_text_store_reads_only_token_tensors_and_exact_loss_counts(
    tmp_path: Path,
) -> None:
    manifest_path, tensors = _write_prepared_fixture(tmp_path)
    store = PreparedTextRecordStore(
        manifest_path,
        expected_sequence_length=5,
        cache_shards=1,
        verify_shards=False,
    )
    first = SimpleNamespace(shard_id="source-000", shard_offset=0)
    second = SimpleNamespace(shard_id="source-000", shard_offset=1)
    try:
        batch = store.batch((second, first))

        assert set(batch.tensors()) == set(PREPARED_TEXT_REQUIRED_TENSORS)
        assert tuple(batch.input_ids.shape) == (2, 5)
        torch.testing.assert_close(batch.input_ids[0], tensors["input_ids"][1])
        torch.testing.assert_close(batch.labels[1], tensors["labels"][0])
        assert store.layout.shard_ids == ("source-000",)
        assert store.layout.size == 2
        assert store.optimizer_batch_token_counts((first, second)) == (6, 9)
        assert store.optimizer_batch_mtp_token_count((first, second)) == 4
    finally:
        store.close()


def test_prepared_text_single_record_batch_is_contiguous_and_moves_all_tensors(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_prepared_fixture(tmp_path)
    store = PreparedTextRecordStore(
        manifest_path,
        expected_sequence_length=5,
        verify_shards=False,
    )
    reference = SimpleNamespace(shard_id="source-000", shard_offset=0)
    try:
        batch = store.batch((reference,))

        assert tuple(batch.input_ids.shape) == (1, 5)
        assert batch.input_ids.is_contiguous()
        assert batch.record(0).input_ids.data_ptr() == batch.input_ids.data_ptr()
        moved = move_prepared_text_batch(batch, torch.device("cpu"))
        torch.testing.assert_close(moved.input_ids, batch.input_ids)
        assert moved.attention_mask.dtype is torch.bool
    finally:
        store.close()


def test_prepared_text_shard_rejects_extra_or_wrong_shaped_tensors(tmp_path: Path) -> None:
    tensors = {
        "input_ids": torch.zeros((2, 5), dtype=torch.int64),
        "labels": torch.zeros((2, 5), dtype=torch.int64),
        "attention_mask": torch.ones((2, 5), dtype=torch.bool),
        "topk_logits": torch.zeros((2, 5, 64)),
    }
    manifest_path, _ = _write_prepared_fixture(tmp_path, tensors=tensors)

    with pytest.raises(PreparedTextSchemaError, match="tensor names"):
        PreparedTextRecordStore(
            manifest_path,
            expected_sequence_length=5,
            verify_shards=False,
        ).record(SimpleNamespace(shard_id="source-000", shard_offset=0))

    wrong_shape_root = tmp_path / "wrong-shape"
    wrong_shape_root.mkdir()
    wrong = {
        "input_ids": torch.zeros((2, 4), dtype=torch.int64),
        "labels": torch.zeros((2, 4), dtype=torch.int64),
        "attention_mask": torch.ones((2, 4), dtype=torch.bool),
    }
    wrong_manifest, _ = _write_prepared_fixture(wrong_shape_root, tensors=wrong)
    with (
        pytest.raises(PreparedTextSchemaError, match="shape mismatch"),
        PreparedTextShardDataset(
            wrong_manifest.parent / "source-000",
            expected_sequence_count=2,
            expected_sequence_length=5,
        ),
    ):
        pass


def test_prepared_text_store_rejects_sequence_length_mismatch(tmp_path: Path) -> None:
    manifest_path, _ = _write_prepared_fixture(tmp_path)

    with pytest.raises(ValueError, match="sequence length"):
        PreparedTextRecordStore(
            manifest_path,
            expected_sequence_length=4,
            verify_shards=False,
        )
