from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from twen.cli import build_parser
from twen.data.audits import (
    DataAuditError,
    audit_lineage_for_role,
    build_base_audit_attestation,
    inspect_benchmark_registry,
    materialize_filtered_base_corpus,
    validate_base_audit_attestation,
)
from twen.data.prepared import (
    AUDITED_PREPARED_GENERATOR_SOURCE_SHA256,
    _authenticate_extracted_prepare_inputs,
    prepare_jsonl_corpus,
    validate_prepared_corpus,
)
from twen.data.sources import validate_extracted_base_corpus
from twen.io.download import sha256_file

TOKENIZER_SHA = "c" * 64


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [1 + ord(character) % 31 for character in text]


def _entry(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_extracted(
    root: Path,
    *,
    train_texts: list[str],
    validation_texts: list[str],
    source_id: str = "source_a",
) -> Path:
    root.mkdir(parents=True)
    train_relative = f"extracted/{source_id}/chunk-000000/train.jsonl"
    validation_relative = f"extracted/{source_id}/chunk-000000/validation.jsonl"
    (root / train_relative).parent.mkdir(parents=True)
    (root / train_relative).write_text(
        "".join(json.dumps({"text": text}) + "\n" for text in train_texts),
        encoding="utf-8",
    )
    (root / validation_relative).write_text(
        "".join(json.dumps({"text": text}) + "\n" for text in validation_texts),
        encoding="utf-8",
    )
    inventories = {
        "train": [_entry(root, train_relative)],
        "validation": [_entry(root, validation_relative)],
        "attribution": [],
    }
    file_lists: dict[str, dict[str, object]] = {}
    for role, entries in inventories.items():
        sidecar = root / f"{role}-files.txt"
        sidecar.write_text(
            "".join(f"{entry['path']}\n" for entry in entries),
            encoding="utf-8",
        )
        file_lists[role] = _entry(root, sidecar.name)
    outputs = [*inventories["train"], *inventories["validation"]]
    identity = {
        "recipe_id": "audit-fixture",
        "recipe_sha256": "a" * 64,
        "resolved_source_lock_sha256": "b" * 64,
        "tokenizer_manifest_sha256": TOKENIZER_SHA,
        "extractor_source_sha256": "d" * 64,
        "profile": "dense",
        "sources": [
            {
                "source_id": source_id,
                "category": "general",
                "chunks": [{"shard_id": "chunk-000000", "outputs": outputs}],
            }
        ],
        "train_files": inventories["train"],
        "validation_files": inventories["validation"],
        "attribution_files": inventories["attribution"],
        "file_lists": file_lists,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_value = {
        "schema_version": 1,
        "kind": "twen_extracted_base_jsonl_corpus",
        **identity,
        "corpus_fingerprint": fingerprint,
        "actual_train_tokens": len(train_texts) * 100,
        "actual_validation_tokens": len(validation_texts) * 100,
        "network_policy": "direct",
        "audits": {
            "output_sha256": "complete",
            "cross_source_near_dedup": "pending",
            "full_contextual_pii_scan": "pending",
            "project_benchmark_13gram_scan": "pending",
        },
        "ready_for_data_prepare": True,
        "ready_for_training": False,
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
                "ready_for_training": False,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _write_registry(
    root: Path,
    *,
    benchmark_text: str,
    ready: bool = True,
) -> tuple[Path, Path]:
    benchmark_root = root / "benchmarks"
    benchmark_root.mkdir(parents=True)
    benchmark = benchmark_root / "fixture.jsonl"
    benchmark.write_text(json.dumps({"question": benchmark_text}) + "\n", encoding="utf-8")
    item: dict[str, object] = {
        "benchmark_id": "fixture",
        "required": True,
        "status": "ready" if ready else "pending_immutable_revision_and_local_jsonl",
        "revision": "e" * 40 if ready else None,
        "files": [],
    }
    if ready:
        item["files"] = [
            {
                **_entry(benchmark_root, benchmark.name),
                "format": "jsonl",
                "text_fields": ["question"],
            }
        ]
    registry = root / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "twen_benchmark_13gram_registry",
                "registry_id": "fixture",
                "benchmarks": [item],
            }
        ),
        encoding="utf-8",
    )
    return registry, benchmark_root


def _clean_inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    candidate = _write_extracted(
        root / "candidate",
        train_texts=[
            "A standalone training document about astronomy, geology, chemistry, and careful observation."
        ],
        validation_texts=["candidate validation is not selected by the audit"],
    )
    frozen = _write_extracted(
        root / "frozen",
        train_texts=["unused frozen training row"],
        validation_texts=["A frozen validation document about music theory and orchestration."],
    )
    registry, benchmark_root = _write_registry(
        root,
        benchmark_text="one two three four five six seven eight nine ten eleven twelve thirteen",
    )
    return candidate, frozen, registry, benchmark_root


def test_clean_audit_attests_both_roles_and_unlocks_prepare() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        candidate, frozen, registry, benchmark_root = _clean_inputs(root)
        attestation = build_base_audit_attestation(
            candidate,
            frozen,
            registry,
            benchmark_root,
            root / "audit",
        )
        value = validate_base_audit_attestation(attestation)
        assert value["ready_for_training"] is True
        assert all(gate["passed"] for gate in value["gates"].values())
        assert (
            audit_lineage_for_role(
                attestation,
                extracted_manifest_path=candidate,
                role="train",
            )["bound_as"]
            == "candidate"
        )
        assert (
            audit_lineage_for_role(
                attestation,
                extracted_manifest_path=frozen,
                role="validation",
            )["bound_as"]
            == "frozen_validation"
        )

        sources, lineage = _authenticate_extracted_prepare_inputs(
            candidate,
            role="train",
            tokenizer_sha256=TOKENIZER_SHA,
            allow_pending_research_audits=False,
            audit_attestation=attestation,
        )
        assert len(sources) == 1
        assert lineage["ready_for_training"] is True
        assert lineage["research_only"] is False
        assert lineage["pending_audits"] == []

        with (
            patch("twen.io.offline.enforce_offline_environment"),
            patch("twen.io.offline.verify_local_download_directory"),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=_Tokenizer()),
        ):
            prepared_path = prepare_jsonl_corpus(
                None,
                root / "prepared",
                tokenizer_path=root / "tokenizer",
                tokenizer_sha256=TOKENIZER_SHA,
                sequence_length=32,
                progress="never",
                extracted_manifest=candidate,
                role="train",
                audit_attestation=attestation,
            )
        prepared = validate_prepared_corpus(prepared_path)
        assert prepared.generator_source_sha256 == AUDITED_PREPARED_GENERATOR_SOURCE_SHA256
        assert prepared.lineage is not None
        assert prepared.lineage["ready_for_training"] is True


def test_exact_and_near_validation_leakage_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        shared = " ".join(f"token{index}" for index in range(80))
        candidate = _write_extracted(
            root / "candidate",
            train_texts=[
                shared,
                shared.replace("token40", "changed40"),
                "A clean retained document about pottery, kilns, glaze, and artistic craft.",
            ],
            validation_texts=["unused candidate validation"],
        )
        frozen = _write_extracted(
            root / "frozen",
            train_texts=["unused"],
            validation_texts=[shared],
            source_id="source_b",
        )
        registry, benchmark_root = _write_registry(
            root,
            benchmark_text="one two three four five six seven eight nine ten eleven twelve thirteen",
        )
        attestation = build_base_audit_attestation(
            candidate,
            frozen,
            registry,
            benchmark_root,
            root / "audit",
            max_findings=1,
        )
        value = validate_base_audit_attestation(attestation)
        assert value["ready_for_training"] is False
        assert value["metrics"]["train_validation_exact_matches"] == 1
        assert value["metrics"]["cross_source_exact_matches"] >= 1
        assert value["gates"]["cross_source_exact_dedup"]["passed"] is False
        assert value["metrics"]["train_validation_near_matches"] >= 1
        assert value["metrics"]["findings_truncated"] > 0
        assert value["metrics"]["rejection_events"] > value["metrics"]["findings_recorded"]
        assert value["rejection_ledger"]["complete"] is True
        with pytest.raises(ValueError, match="allow-pending-research-audits"):
            _authenticate_extracted_prepare_inputs(
                candidate,
                role="train",
                tokenizer_sha256=TOKENIZER_SHA,
                allow_pending_research_audits=False,
                audit_attestation=attestation,
            )

        filtered = materialize_filtered_base_corpus(attestation, root / "filtered")
        report = validate_extracted_base_corpus(filtered)
        assert report["ready_for_training"] is False
        filtered_value = json.loads(filtered.read_text(encoding="utf-8"))
        retained = "".join(
            (filtered.parent / item["path"]).read_text(encoding="utf-8")
            for item in filtered_value["train_files"]
        )
        assert "pottery" in retained
        assert "token40" not in retained

        rescanned = build_base_audit_attestation(
            filtered,
            filtered,
            registry,
            benchmark_root,
            root / "filtered-audit",
        )
        assert validate_base_audit_attestation(rescanned)["ready_for_training"] is True


def test_contextual_pii_and_benchmark_overlap_are_hashed_not_copied() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        benchmark = "one two three four five six seven eight nine ten eleven twelve thirteen"
        candidate = _write_extracted(
            root / "candidate",
            train_texts=[f"phone: +1 415 555 2671. {benchmark}"],
            validation_texts=["unused"],
        )
        frozen = _write_extracted(
            root / "frozen",
            train_texts=["unused"],
            validation_texts=["clean frozen validation"],
        )
        registry, benchmark_root = _write_registry(root, benchmark_text=benchmark)
        attestation = build_base_audit_attestation(
            candidate,
            frozen,
            registry,
            benchmark_root,
            root / "audit",
        )
        value = validate_base_audit_attestation(attestation)
        assert value["metrics"]["contextual_pii_documents"] == 1
        assert value["metrics"]["benchmark_overlap_documents"] == 1
        findings = (attestation.parent / "findings.jsonl").read_text(encoding="utf-8")
        assert "+1 415" not in findings
        assert benchmark not in findings


def test_pending_registry_stays_pending_and_tampering_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        candidate, frozen, _, _ = _clean_inputs(root)
        registry, benchmark_root = _write_registry(
            root / "pending",
            benchmark_text="unused pending benchmark",
            ready=False,
        )
        report = inspect_benchmark_registry(registry, benchmark_root=benchmark_root)
        assert report["ready"] is False
        assert report["pending_benchmarks"] == ["fixture"]
        attestation = build_base_audit_attestation(
            candidate,
            frozen,
            registry,
            benchmark_root,
            root / "audit-pending",
        )
        value = validate_base_audit_attestation(attestation)
        assert value["ready_for_training"] is False
        assert value["gates"]["project_benchmark_13gram_scan"]["status"].startswith("pending")

        (attestation.parent / "findings.jsonl").write_text("tampered\n", encoding="utf-8")
        with pytest.raises(DataAuditError, match=r"(size|SHA256)"):
            validate_base_audit_attestation(attestation)


def test_cli_exposes_audit_and_attested_prepare_contract() -> None:
    parser = build_parser()
    audit = parser.parse_args(
        [
            "data",
            "audit-base",
            "--extracted-manifest",
            "candidate.json",
            "--frozen-validation-manifest",
            "validation.json",
            "--benchmark-registry",
            "registry.json",
            "--benchmark-root",
            "benchmarks",
            "--output",
            "audit",
        ]
    )
    assert audit.near_duplicate_threshold == 0.8
    materialize = parser.parse_args(
        [
            "data",
            "materialize-audit",
            "--audit-attestation",
            "audit/attestation.json",
            "--output",
            "filtered",
        ]
    )
    assert materialize.output == "filtered"
    prepare = parser.parse_args(
        [
            "data",
            "prepare",
            "--extracted-manifest",
            "candidate.json",
            "--role",
            "train",
            "--audit-attestation",
            "audit/attestation.json",
            "--output",
            "prepared",
            "--tokenizer",
            "tokenizer",
            "--tokenizer-manifest-sha256",
            TOKENIZER_SHA,
        ]
    )
    assert prepare.audit_attestation == "audit/attestation.json"
