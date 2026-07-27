from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from twen.cli import build_parser
from twen.data.audits import build_base_audit_attestation
from twen.data.inference_prepared import validate_prepared_corpus_for_inference
from twen.data.prepared import (
    AUDITED_PREPARED_GENERATOR_SOURCE_SHA256,
    PREPARED_GENERATOR_SOURCE_SHA256,
    PreparedCorpusManifest,
    _authenticate_extracted_prepare_inputs,
    _local_prepared_manifest,
    _prepared_dataset_fingerprint,
    _prepared_pipeline_fingerprint,
    prepare_jsonl_corpus,
    read_prepared_manifest,
    validate_prepared_corpus,
)
from twen.io.download import sha256_file

TOKENIZER_SHA = "c" * 64


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [1 + ord(character) % 31 for character in text]


def _file_entry(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_extracted_fixture(root: Path, *, ready_for_training: bool = False) -> Path:
    root.mkdir(parents=True)
    train_relative = "source/chunk-000000/train.jsonl"
    validation_relative = "source/chunk-000000/validation.jsonl"
    (root / train_relative).parent.mkdir(parents=True)
    (root / train_relative).write_text('{"text":"train text"}\n', encoding="utf-8")
    (root / validation_relative).write_text('{"text":"validation text"}\n', encoding="utf-8")
    inventories = {
        "train": [_file_entry(root, train_relative)],
        "validation": [_file_entry(root, validation_relative)],
        "attribution": [],
    }
    file_lists: dict[str, dict[str, object]] = {}
    for role, entries in inventories.items():
        sidecar = root / f"{role}-files.txt"
        sidecar.write_text("".join(f"{entry['path']}\n" for entry in entries), encoding="utf-8")
        file_lists[role] = _file_entry(root, sidecar.name)
    identity = {
        "recipe_id": "fixture-base",
        "recipe_sha256": "a" * 64,
        "resolved_source_lock_sha256": "b" * 64,
        "tokenizer_manifest_sha256": TOKENIZER_SHA,
        "extractor_source_sha256": "d" * 64,
        "profile": "dense",
        "sources": [],
        "train_files": inventories["train"],
        "validation_files": inventories["validation"],
        "attribution_files": inventories["attribution"],
        "file_lists": file_lists,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    audits = {
        "output_sha256": "complete",
        **({} if ready_for_training else {"near_dedup": "pending"}),
    }
    manifest_value = {
        "schema_version": 1,
        "kind": "twen_extracted_base_jsonl_corpus",
        **identity,
        "corpus_fingerprint": fingerprint,
        "actual_train_tokens": 10,
        "actual_validation_tokens": 10,
        "network_policy": "direct",
        "audits": audits,
        "ready_for_data_prepare": True,
        "ready_for_training": ready_for_training,
    }
    manifest = root / "corpus-manifest.json"
    manifest.write_text(json.dumps(manifest_value, sort_keys=True), encoding="utf-8")
    (root / "COMPLETE").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "twen_extracted_base_jsonl_complete",
                "corpus_fingerprint": fingerprint,
                "manifest": manifest.name,
                "manifest_sha256": sha256_file(manifest),
                "file_lists": file_lists,
                "ready_for_training": ready_for_training,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _write_benchmark_registry(root: Path) -> tuple[Path, Path]:
    benchmark_root = root / "benchmarks"
    benchmark_root.mkdir()
    benchmark = benchmark_root / "fixture.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "question": (
                    "one two three four five six seven eight nine ten eleven twelve thirteen"
                )
            }
        )
        + "\n",
        encoding="utf-8",
    )
    registry = root / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "twen_benchmark_13gram_registry",
                "registry_id": "fixture",
                "benchmarks": [
                    {
                        "benchmark_id": "fixture",
                        "required": True,
                        "status": "ready",
                        "revision": "e" * 40,
                        "files": [
                            {
                                **_file_entry(benchmark_root, benchmark.name),
                                "format": "jsonl",
                                "text_fields": ["question"],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return registry, benchmark_root


def _rewrite_as_historical_prepared(
    manifest: Path,
    *,
    generator_source_sha256: str,
) -> str:
    original = read_prepared_manifest(manifest)
    source_hashes = [
        (Path(entry.source_path), entry.source_sha256)
        for entry in original.shards
    ]
    pipeline_fingerprint = _prepared_pipeline_fingerprint(
        source_hashes,
        tokenizer_sha256=original.tokenizer_sha256,
        sequence_length=original.sequence_length,
        text_field=original.text_field,
        generator_source_sha256=generator_source_sha256,
        lineage=original.lineage,
    )
    dataset_fingerprint = _prepared_dataset_fingerprint(
        pipeline_fingerprint=pipeline_fingerprint,
        generator_source_sha256=generator_source_sha256,
        tokenizer_sha256=original.tokenizer_sha256,
        sequence_length=original.sequence_length,
        text_field=original.text_field,
        shards=original.shards,
        lineage=original.lineage,
    )
    rewritten = PreparedCorpusManifest(
        dataset_fingerprint=dataset_fingerprint,
        pipeline_fingerprint=pipeline_fingerprint,
        generator_source_sha256=generator_source_sha256,
        tokenizer_sha256=original.tokenizer_sha256,
        sequence_length=original.sequence_length,
        text_field=original.text_field,
        shards=original.shards,
        lineage=original.lineage,
    )
    for entry in original.shards:
        shard = manifest.parent / entry.path
        local_path = shard / "prepared_manifest.json"
        local = _local_prepared_manifest(
            shard_id=entry.shard_id,
            source_path=entry.source_path,
            source_sha256=entry.source_sha256,
            sequence_count=entry.sequence_count,
            token_count=entry.token_count,
            tensors_sha256=entry.tensors_sha256,
            pipeline_fingerprint=pipeline_fingerprint,
            generator_source_sha256=generator_source_sha256,
            tokenizer_sha256=original.tokenizer_sha256,
            sequence_length=original.sequence_length,
            text_field=original.text_field,
        )
        local_path.write_text(json.dumps(local, sort_keys=True), encoding="utf-8")
        complete_path = shard / "COMPLETE"
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        for output in complete["outputs"]:
            if output["path"] == "prepared_manifest.json":
                output["size"] = local_path.stat().st_size
                output["sha256"] = sha256_file(local_path)
        complete_path.write_text(json.dumps(complete, sort_keys=True), encoding="utf-8")
    manifest.write_text(
        json.dumps(rewritten.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    return sha256_file(manifest)


def _write_historical_prepared_fixture(root: Path) -> tuple[Path, str]:
    extracted = _write_extracted_fixture(root / "extracted")
    registry, benchmark_root = _write_benchmark_registry(root)
    audit = build_base_audit_attestation(
        extracted,
        extracted,
        registry,
        benchmark_root,
        root / "audit",
    )
    with (
        patch("twen.io.offline.enforce_offline_environment"),
        patch("twen.io.offline.verify_local_download_directory"),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_Tokenizer()),
    ):
        output = prepare_jsonl_corpus(
            None,
            root / "prepared",
            tokenizer_path=root / "tokenizer",
            tokenizer_sha256=TOKENIZER_SHA,
            sequence_length=4,
            progress="never",
            extracted_manifest=extracted,
            role="train",
            audit_attestation=audit,
        )
    pinned_manifest_sha = _rewrite_as_historical_prepared(
        output,
        generator_source_sha256="1" * 64,
    )
    return output, pinned_manifest_sha


class PreparedExtractedLineageTest(unittest.TestCase):
    def test_cli_extracted_mode_is_mutually_exclusive_and_names_override(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "data",
                "prepare",
                "--extracted-manifest",
                "corpus-manifest.json",
                "--role",
                "validation",
                "--allow-pending-research-audits",
                "--output",
                "prepared",
                "--tokenizer",
                "tokenizer",
                "--tokenizer-manifest-sha256",
                TOKENIZER_SHA,
            ]
        )
        self.assertIsNone(args.input)
        self.assertEqual(args.role, "validation")
        self.assertTrue(args.allow_pending_research_audits)
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "data",
                    "prepare",
                    "--input",
                    "input.jsonl",
                    "--extracted-manifest",
                    "corpus-manifest.json",
                    "--output",
                    "prepared",
                    "--tokenizer",
                    "tokenizer",
                    "--tokenizer-manifest-sha256",
                    TOKENIZER_SHA,
                ]
            )

    def test_pending_extracted_audits_fail_closed_then_record_research_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _write_extracted_fixture(Path(directory) / "extracted")
            with self.assertRaisesRegex(ValueError, "allow-pending-research-audits"):
                _authenticate_extracted_prepare_inputs(
                    manifest,
                    role="train",
                    tokenizer_sha256=TOKENIZER_SHA,
                    allow_pending_research_audits=False,
                )
            sources, lineage = _authenticate_extracted_prepare_inputs(
                manifest,
                role="train",
                tokenizer_sha256=TOKENIZER_SHA,
                allow_pending_research_audits=True,
            )
            self.assertEqual(len(sources), 1)
            self.assertEqual(lineage["extracted_manifest_sha256"], sha256_file(manifest))
            self.assertEqual(lineage["role"], "train")
            self.assertEqual(lineage["recipe_sha256"], "a" * 64)
            self.assertEqual(lineage["resolved_source_lock_sha256"], "b" * 64)
            self.assertEqual(lineage["pending_audits"], ["near_dedup"])
            self.assertTrue(lineage["research_only"])
            self.assertFalse(lineage["ready_for_training"])
            self.assertEqual(lineage["source_files"][0]["size"], sources[0][0].stat().st_size)

    def test_prepare_and_validator_bind_exact_extracted_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extracted = _write_extracted_fixture(root / "extracted")
            with (
                patch("twen.io.offline.enforce_offline_environment"),
                patch("twen.io.offline.verify_local_download_directory"),
                patch("transformers.AutoTokenizer.from_pretrained", return_value=_Tokenizer()),
            ):
                output = prepare_jsonl_corpus(
                    None,
                    root / "prepared",
                    tokenizer_path=root / "tokenizer",
                    tokenizer_sha256=TOKENIZER_SHA,
                    sequence_length=4,
                    progress="never",
                    extracted_manifest=extracted,
                    role="validation",
                    allow_pending_research_audits=True,
                )
            prepared = read_prepared_manifest(output)
            assert prepared.lineage is not None
            self.assertEqual(prepared.lineage["kind"], "authenticated_extracted_corpus")
            self.assertEqual(prepared.lineage["role"], "validation")
            self.assertEqual(
                prepared.generator_source_sha256,
                AUDITED_PREPARED_GENERATOR_SOURCE_SHA256,
            )
            self.assertEqual(len(prepared.shards), 1)
            self.assertEqual(
                Path(prepared.shards[0].source_path).name,
                "validation.jsonl",
            )
            self.assertEqual(validate_prepared_corpus(output), prepared)

            selected = Path(prepared.shards[0].source_path)
            selected.write_text('{"text":"tampered"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "(size|SHA256) mismatch"):
                validate_prepared_corpus(output)

    def test_historical_generator_is_inference_only_and_manifest_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, pinned_manifest_sha = _write_historical_prepared_fixture(root)

            with self.assertRaisesRegex(ValueError, "generator source changed"):
                validate_prepared_corpus(output)
            inferred = validate_prepared_corpus_for_inference(
                output,
                expected_manifest_sha256=pinned_manifest_sha,
            )
            self.assertEqual(
                inferred.generator_source_sha256,
                "1" * 64,
            )
            with self.assertRaisesRegex(ValueError, "pinned SHA256"):
                validate_prepared_corpus_for_inference(
                    output,
                    expected_manifest_sha256="2" * 64,
                )

            tensors = output.parent / inferred.shards[0].path / "tokens.safetensors"
            with tensors.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "(incomplete|tensor hash)"):
                validate_prepared_corpus_for_inference(
                    output,
                    expected_manifest_sha256=pinned_manifest_sha,
                )

    def test_historical_inference_recomputes_dataset_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output, _ = _write_historical_prepared_fixture(Path(directory))
            value = json.loads(output.read_text(encoding="utf-8"))
            value["dataset_fingerprint"] = "f" * 64
            output.write_text(
                json.dumps(value, sort_keys=True),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "dataset fingerprint"):
                validate_prepared_corpus_for_inference(
                    output,
                    expected_manifest_sha256=sha256_file(output),
                )

    def test_historical_inference_reauthenticates_extracted_source_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output, pinned_manifest_sha = _write_historical_prepared_fixture(
                Path(directory)
            )
            prepared = read_prepared_manifest(output)
            source = Path(prepared.shards[0].source_path)
            source.write_text('{"text":"tampered source"}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "(size|SHA256) mismatch"):
                validate_prepared_corpus_for_inference(
                    output,
                    expected_manifest_sha256=pinned_manifest_sha,
                )

    def test_explicit_inputs_are_never_presented_as_authenticated_extracted_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            source.write_text('{"text":"manual"}\n', encoding="utf-8")
            with (
                patch("twen.io.offline.enforce_offline_environment"),
                patch("twen.io.offline.verify_local_download_directory"),
                patch("transformers.AutoTokenizer.from_pretrained", return_value=_Tokenizer()),
            ):
                output = prepare_jsonl_corpus(
                    [source],
                    root / "prepared",
                    tokenizer_path=root / "tokenizer",
                    tokenizer_sha256=TOKENIZER_SHA,
                    sequence_length=4,
                    progress="never",
                )
            prepared = validate_prepared_corpus(output)
            assert prepared.lineage is not None
            self.assertEqual(prepared.lineage["kind"], "explicit_unreviewed")
            self.assertEqual(
                prepared.generator_source_sha256,
                PREPARED_GENERATOR_SOURCE_SHA256,
            )
            self.assertTrue(prepared.lineage["research_only"])


if __name__ == "__main__":
    unittest.main()
