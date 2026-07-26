from __future__ import annotations

import json
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from twen.data.prepared import (
    PREPARED_GENERATOR_SOURCE_SHA256,
    PreparedCorpusManifest,
    PreparedShardEntry,
)
from twen.data.teacher_kd import (
    KD_GENERATOR_SOURCE_SHA256,
    KD_REQUIRED_TENSORS,
    TeacherKDBatch,
    TeacherKDCorpus,
    TeacherKDManifest,
    TeacherKDSchemaError,
    TeacherKDShardDataset,
    load_kd_shard,
    validate_kd_corpus_manifest,
    validate_kd_shard,
    write_kd_corpus_manifest,
    write_kd_shard,
)
from twen.kd import _normalize_cuda_device


def _manifest(
    *,
    source_shard_id: str = "source-000",
    source_tensors_sha256: str = "2" * 64,
    global_sample_start: int = 0,
    global_token_start: int = 0,
) -> TeacherKDManifest:
    return TeacherKDManifest(
        teacher_model_id="Qwen/Qwen3.5-9B-Base",
        teacher_revision="a" * 40,
        teacher_model_sha256="b" * 64,
        generator_source_sha256=KD_GENERATOR_SOURCE_SHA256,
        tokenizer_sha256="c" * 64,
        dataset_fingerprint="d" * 64,
        source_tensors_sha256=source_tensors_sha256,
        source_shard_id=source_shard_id,
        global_sample_start=global_sample_start,
        global_sample_end=global_sample_start + 2,
        global_token_start=global_token_start,
        global_token_end=global_token_start + 5,
        sequence_count=2,
        sequence_length=3,
        token_count=5,
        vocab_size=248320,
        temperature=2.0,
    )


def _fake_safetensors_serializer(manifest: TeacherKDManifest):
    shapes = {
        "input_ids": ([2, 3], "I64"),
        "labels": ([2, 3], "I64"),
        "attention_mask": ([2, 3], "I64"),
        "topk_indices": ([2, 3, 64], "I64"),
        "topk_logits": ([2, 3, 64], manifest.logits_dtype),
        "teacher_logsumexp": ([2, 3], "F32"),
        "teacher_tail_logprob": ([2, 3], "F32"),
    }
    bytes_per_value = {"I64": 8, "F16": 2, "BF16": 2, "F32": 4}

    def serialize(tensors, destination: str) -> None:
        self_contained = set(tensors)
        if self_contained != set(KD_REQUIRED_TENSORS):
            raise AssertionError(self_contained)
        offset = 0
        header = {}
        for name, (shape, dtype) in shapes.items():
            count = 1
            for dimension in shape:
                count *= dimension
            byte_count = count * bytes_per_value[dtype]
            header[name] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + byte_count],
            }
            offset += byte_count
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        padding = (8 - len(encoded) % 8) % 8
        encoded += b" " * padding
        Path(destination).write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * offset)

    return serialize


class _FakeScalar:
    def __init__(self, value: int) -> None:
        self.value = value

    def __int__(self) -> int:
        return self.value


class _FakeTensor:
    def __init__(self, shape, dtype="torch.int64", valid_tokens=5) -> None:
        self.shape = shape
        self.dtype = dtype
        self.valid_tokens = valid_tokens

    def __getitem__(self, index):
        return _FakeTensor(self.shape[1:], self.dtype, self.valid_tokens)

    def to(self, dtype=None):
        return self

    def sum(self):
        return _FakeScalar(self.valid_tokens)


class TeacherKDSchemaTest(unittest.TestCase):
    def test_cuda_device_shorthand_gets_an_explicit_index(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_normalize_cuda_device("cuda"), "cuda:0")
        with patch.dict("os.environ", {"LOCAL_RANK": "3"}, clear=True):
            self.assertEqual(_normalize_cuda_device("cuda"), "cuda:3")
        self.assertEqual(_normalize_cuda_device("cuda:2"), "cuda:2")
        self.assertEqual(_normalize_cuda_device("cpu"), "cpu")

    def _write(
        self,
        directory: str,
        *,
        shard_id: str = "kd-000",
        source_shard_id: str = "source-000",
        source_tensors_sha256: str = "2" * 64,
        global_sample_start: int = 0,
        global_token_start: int = 0,
    ) -> Path:
        manifest = _manifest(
            source_shard_id=source_shard_id,
            source_tensors_sha256=source_tensors_sha256,
            global_sample_start=global_sample_start,
            global_token_start=global_token_start,
        )
        placeholder = object()
        batch = TeacherKDBatch(
            **{name: placeholder for name in KD_REQUIRED_TENSORS},
            temperature=2.0,
        )
        return write_kd_shard(
            directory,
            shard_id,
            manifest=manifest,
            batch=batch,
            serializer=_fake_safetensors_serializer(manifest),
        )

    def test_schema_is_complete_verified_and_temperature_locked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shard = self._write(directory)
            manifest = validate_kd_shard(shard, expected_temperature=2.0)
            self.assertTrue(manifest.topk_logits_are_raw)
            self.assertEqual(manifest.top_k, 64)
            with self.assertRaisesRegex(TeacherKDSchemaError, "temperature mismatch"):
                validate_kd_shard(shard, expected_temperature=1.0)

    def test_schema_v1_and_staging_directories_are_rejected(self) -> None:
        legacy = _manifest().to_dict()
        legacy["schema_version"] = 1
        legacy.pop("generator_source_sha256")
        with self.assertRaisesRegex(TeacherKDSchemaError, "must be regenerated"):
            TeacherKDManifest.from_dict(legacy)

        with tempfile.TemporaryDirectory() as directory:
            shard = self._write(directory)
            staging = shard.with_name(f"{shard.name}.incomplete")
            shard.rename(staging)
            with self.assertRaisesRegex(TeacherKDSchemaError, "staging"):
                validate_kd_shard(staging, expected_temperature=2.0)

    def test_load_and_dataset_expose_loss_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shard = self._write(directory)
            tensors = {
                "input_ids": _FakeTensor((2, 3)),
                "labels": _FakeTensor((2, 3)),
                "attention_mask": _FakeTensor((2, 3), valid_tokens=5),
                "topk_indices": _FakeTensor((2, 3, 64)),
                "topk_logits": _FakeTensor((2, 3, 64), "torch.float16"),
                "teacher_logsumexp": _FakeTensor((2, 3), "torch.float32"),
                "teacher_tail_logprob": _FakeTensor((2, 3), "torch.float32"),
            }
            package = types.ModuleType("safetensors")
            package.__path__ = []
            torch_module = types.ModuleType("safetensors.torch")
            torch_module.load_file = lambda path, device="cpu": tensors
            with patch.dict(
                sys.modules,
                {"safetensors": package, "safetensors.torch": torch_module},
            ):
                manifest, batch = load_kd_shard(shard, expected_temperature=2.0)
                self.assertEqual(manifest.temperature, 2.0)
                self.assertEqual(batch.topk_indices.dtype, "torch.int64")
            with TeacherKDShardDataset(
                shard,
                expected_temperature=2.0,
            ) as dataset:
                record = dataset[0]
                for field in KD_REQUIRED_TENSORS:
                    self.assertTrue(hasattr(record, field))
                self.assertEqual(record.temperature, 2.0)
                self.assertEqual(dataset.loss_token_counts(0), (0, 0))

    def test_writer_refuses_batch_temperature_mismatch(self) -> None:
        manifest = _manifest()
        batch = TeacherKDBatch(
            **{name: object() for name in KD_REQUIRED_TENSORS},
            temperature=1.0,
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(TeacherKDSchemaError),
        ):
            write_kd_shard(
                directory,
                "kd-000",
                manifest=manifest,
                batch=batch,
                serializer=lambda *_: None,
            )

    def test_corpus_manifest_locks_contiguous_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = self._write(directory)
            second = self._write(
                directory,
                shard_id="kd-001",
                source_shard_id="source-001",
                global_sample_start=2,
                global_token_start=5,
            )
            corpus_path = write_kd_corpus_manifest(
                Path(directory) / "manifest.json",
                [second, first],
                expected_temperature=2.0,
            )
            corpus = validate_kd_corpus_manifest(
                corpus_path, expected_temperature=2.0
            )
            self.assertEqual(corpus.sequence_count, 4)
            self.assertEqual(corpus.token_count, 10)
            self.assertEqual([entry.path for entry in corpus.shards], ["kd-000", "kd-001"])
            with self.assertRaisesRegex(TeacherKDSchemaError, "temperature mismatch"):
                TeacherKDCorpus(corpus_path, expected_temperature=1.0)

    def test_corpus_index_rejects_a_complete_but_partial_prepared_prefix(self) -> None:
        prepared = PreparedCorpusManifest(
            dataset_fingerprint="d" * 64,
            pipeline_fingerprint="e" * 64,
            generator_source_sha256=PREPARED_GENERATOR_SOURCE_SHA256,
            tokenizer_sha256="c" * 64,
            sequence_length=3,
            text_field="text",
            shards=(
                PreparedShardEntry(
                    shard_id="source-000",
                    path="source-000",
                    source_path="input-0.jsonl",
                    source_sha256="1" * 64,
                    tensors_sha256="2" * 64,
                    sequence_count=2,
                    token_count=5,
                    global_sample_start=0,
                    global_sample_end=2,
                    global_token_start=0,
                    global_token_end=5,
                ),
                PreparedShardEntry(
                    shard_id="source-001",
                    path="source-001",
                    source_path="input-1.jsonl",
                    source_sha256="3" * 64,
                    tensors_sha256="4" * 64,
                    sequence_count=2,
                    token_count=5,
                    global_sample_start=2,
                    global_sample_end=4,
                    global_token_start=5,
                    global_token_end=10,
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            first = self._write(directory)
            with self.assertRaisesRegex(TeacherKDSchemaError, "exactly cover"):
                write_kd_corpus_manifest(
                    Path(directory) / "manifest.json",
                    [first],
                    expected_temperature=2.0,
                    prepared_corpus=prepared,
                )

    def test_corpus_coverage_binds_exact_prepared_tensor_sha(self) -> None:
        def prepared(tensor_sha: str) -> PreparedCorpusManifest:
            return PreparedCorpusManifest(
                dataset_fingerprint="d" * 64,
                pipeline_fingerprint="e" * 64,
                generator_source_sha256=PREPARED_GENERATOR_SOURCE_SHA256,
                tokenizer_sha256="c" * 64,
                sequence_length=3,
                text_field="text",
                shards=(
                    PreparedShardEntry(
                        shard_id="source-000",
                        path="source-000",
                        source_path="input.jsonl",
                        source_sha256="1" * 64,
                        tensors_sha256=tensor_sha,
                        sequence_count=2,
                        token_count=5,
                        global_sample_start=0,
                        global_sample_end=2,
                        global_token_start=0,
                        global_token_end=5,
                    ),
                ),
            )

        with tempfile.TemporaryDirectory() as directory:
            shard = self._write(directory, source_tensors_sha256="2" * 64)
            write_kd_corpus_manifest(
                Path(directory) / "manifest.json",
                [shard],
                expected_temperature=2.0,
                prepared_corpus=prepared("2" * 64),
            )
            with self.assertRaisesRegex(TeacherKDSchemaError, "exactly cover"):
                write_kd_corpus_manifest(
                    Path(directory) / "wrong.json",
                    [shard],
                    expected_temperature=2.0,
                    prepared_corpus=prepared("9" * 64),
                )

    def test_prepared_manifest_rejects_duplicate_paths_and_range_gaps(self) -> None:
        first = PreparedShardEntry(
            shard_id="source-000",
            path="shared",
            source_path="input-0.jsonl",
            source_sha256="1" * 64,
            tensors_sha256="2" * 64,
            sequence_count=2,
            token_count=5,
            global_sample_start=0,
            global_sample_end=2,
            global_token_start=0,
            global_token_end=5,
        )
        duplicate_path = PreparedShardEntry(
            shard_id="source-001",
            path="shared",
            source_path="input-1.jsonl",
            source_sha256="3" * 64,
            tensors_sha256="4" * 64,
            sequence_count=2,
            token_count=5,
            global_sample_start=2,
            global_sample_end=4,
            global_token_start=5,
            global_token_end=10,
        )
        with self.assertRaisesRegex(ValueError, "paths must be unique"):
            PreparedCorpusManifest(
                dataset_fingerprint="d" * 64,
                pipeline_fingerprint="e" * 64,
                generator_source_sha256=PREPARED_GENERATOR_SOURCE_SHA256,
                tokenizer_sha256="c" * 64,
                sequence_length=3,
                text_field="text",
                shards=(first, duplicate_path),
            )

        gap = PreparedShardEntry(
            shard_id="source-001",
            path="source-001",
            source_path="input-1.jsonl",
            source_sha256="3" * 64,
            tensors_sha256="4" * 64,
            sequence_count=2,
            token_count=5,
            global_sample_start=3,
            global_sample_end=5,
            global_token_start=5,
            global_token_end=10,
        )
        with self.assertRaisesRegex(ValueError, "sample ranges"):
            PreparedCorpusManifest(
                dataset_fingerprint="d" * 64,
                pipeline_fingerprint="e" * 64,
                generator_source_sha256=PREPARED_GENERATOR_SOURCE_SHA256,
                tokenizer_sha256="c" * 64,
                sequence_length=3,
                text_field="text",
                shards=(first, gap),
            )


if __name__ == "__main__":
    unittest.main()
