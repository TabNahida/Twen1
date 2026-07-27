from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from twen.cli import build_parser
from twen.data import phase_exclusion
from twen.data.audits import (
    AUDIT_KIND,
    AUDIT_SCHEMA_VERSION,
    AUDIT_SOURCE_SHA256,
    DataAuditError,
)
from twen.data.prepared import _authenticate_extracted_prepare_inputs
from twen.data.sources import validate_extracted_base_corpus
from twen.utils import atomic_write_json, sha256_file


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _identity(path: Path, *, relative: str) -> dict[str, object]:
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _stable(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_extracted(
    root: Path,
    *,
    source_id: str,
    train: list[tuple[str, str, int]],
    validation: list[tuple[str, str, int]],
) -> Path:
    chunk = root / f"extracted/{source_id}/chunk-000000"
    chunk.mkdir(parents=True)
    train_path = chunk / "train.jsonl"
    validation_path = chunk / "validation.jsonl"
    attribution_path = chunk / "attribution.jsonl"
    train_path.write_bytes(
        b"".join(
            (json.dumps({"text": text}, ensure_ascii=False) + "\n").encode() for _, text, _ in train
        )
    )
    validation_path.write_bytes(
        b"".join(
            (json.dumps({"text": text}, ensure_ascii=False) + "\n").encode()
            for _, text, _ in validation
        )
    )
    attribution_rows: list[dict[str, object]] = []
    for split, rows in (("train", train), ("validation", validation)):
        for stable_name, text, tokens in rows:
            attribution_rows.append(
                {
                    "source_id": source_id,
                    "stable_id": _stable(stable_name),
                    "split": split,
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "token_count_with_eos": tokens,
                    "source_license": "fixture-license",
                }
            )
    attribution_path.write_bytes(
        b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in attribution_rows)
    )
    relative_train = train_path.relative_to(root).as_posix()
    relative_validation = validation_path.relative_to(root).as_posix()
    relative_attribution = attribution_path.relative_to(root).as_posix()
    train_files = [_identity(train_path, relative=relative_train)]
    validation_files = [_identity(validation_path, relative=relative_validation)]
    attribution_files = [_identity(attribution_path, relative=relative_attribution)]
    inventories = {
        "train": train_files,
        "validation": validation_files,
        "attribution": attribution_files,
    }
    file_lists: dict[str, dict[str, object]] = {}
    for role, inventory in inventories.items():
        sidecar = root / f"{role}-files.txt"
        sidecar.write_text(
            "".join(f"{entry['path']}\n" for entry in inventory),
            encoding="utf-8",
        )
        file_lists[role] = _identity(sidecar, relative=sidecar.name)
    source_map_unsigned = {
        "schema_version": 1,
        "algorithm": "authenticated-extracted-output-map-v1",
        "roles": {
            "train": [{"source_id": source_id, **train_files[0]}],
            "validation": [{"source_id": source_id, **validation_files[0]}],
        },
    }
    source_map = {
        **source_map_unsigned,
        "fingerprint": _canonical_sha256(source_map_unsigned),
    }
    source_mix_unsigned = {
        "schema_version": 1,
        "algorithm": "token-deficit-corrected-source-mix-bp-v2",
        "unit": "valid_tokens",
        "basis_points_total": 10_000,
        "profile": "fixture",
        "sources": [
            {
                "source_id": source_id,
                "origin_group": "fixture",
                "mix_basis_points": 10_000,
                "target_train_tokens": sum(row[2] for row in train),
                "actual_train_tokens": sum(row[2] for row in train),
            }
        ],
    }
    source_mix = {
        **source_mix_unsigned,
        "fingerprint": _canonical_sha256(source_mix_unsigned),
    }
    outputs = [train_files[0], validation_files[0], attribution_files[0]]
    sources = [
        {
            "source_id": source_id,
            "category": "fixture",
            "actual_train_tokens": sum(row[2] for row in train),
            "actual_validation_tokens": sum(row[2] for row in validation),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "chunks": [
                {
                    "shard_id": "fixture",
                    "outputs": outputs,
                    "statistics": {},
                }
            ],
        }
    ]
    format_audit = {"complete": True, "sources": [{"source_id": source_id}]}
    license_audit = {
        "complete": True,
        "attribution_inventory": file_lists["attribution"],
        "sources": [{"source_id": source_id, "license": "fixture-license"}],
    }
    materialization_audit = {
        "complete": True,
        "network_policy": "offline-fixture",
        "sources": [{"source_id": source_id}],
    }
    identity = {
        "recipe_id": "fixture-recipe",
        "recipe_sha256": "1" * 64,
        "resolved_source_lock_sha256": "2" * 64,
        "tokenizer_manifest_sha256": "3" * 64,
        "extractor_source_sha256": "4" * 64,
        "profile": "fixture",
        "sources": sources,
        "train_files": train_files,
        "validation_files": validation_files,
        "attribution_files": attribution_files,
        "file_lists": file_lists,
        "source_map": source_map,
        "source_mix": source_mix,
        "format_audit": format_audit,
        "license_audit": license_audit,
        "materialization_audit": materialization_audit,
    }
    fingerprint = _canonical_sha256(identity)
    manifest_value = {
        "schema_version": 1,
        "kind": "twen_extracted_base_jsonl_corpus",
        **identity,
        "corpus_fingerprint": fingerprint,
        "actual_train_tokens": sum(row[2] for row in train),
        "actual_validation_tokens": sum(row[2] for row in validation),
        "actual_train_documents": len(train),
        "actual_validation_documents": len(validation),
        "audits": {"fixture": "pending"},
        "ready_for_data_prepare": True,
        "ready_for_training": False,
    }
    manifest = root / "corpus-manifest.json"
    atomic_write_json(manifest, manifest_value)
    atomic_write_json(
        root / "COMPLETE",
        {
            "schema_version": 1,
            "kind": "twen_extracted_base_jsonl_complete",
            "corpus_fingerprint": fingerprint,
            "manifest": manifest.name,
            "manifest_sha256": sha256_file(manifest),
            "file_lists": file_lists,
            "ready_for_training": False,
        },
    )
    validate_extracted_base_corpus(manifest)
    return manifest


def _write_audit(root: Path, manifest: Path) -> Path:
    root.mkdir(parents=True)
    findings = root / "findings.jsonl"
    rejections = root / "rejections.jsonl"
    registry = root / "benchmark-registry.json"
    findings.write_bytes(b"")
    rejections.write_bytes(b"")
    registry.write_text("{}\n", encoding="utf-8")
    manifest_value = json.loads(manifest.read_text())
    common = {
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "corpus_fingerprint": manifest_value["corpus_fingerprint"],
    }
    gates = {
        "cross_source_exact_dedup": {
            "passed": True,
            "status": "complete_no_matches",
        },
        "cross_source_near_dedup": {
            "passed": True,
            "status": "complete_no_matches",
        },
        "full_contextual_pii_scan": {
            "passed": True,
            "status": "complete_no_findings",
        },
        "project_benchmark_13gram_scan": {
            "passed": True,
            "status": "complete_no_matches",
        },
        "train_vs_frozen_validation_exact_dedup": {
            "passed": True,
            "status": "complete_no_matches",
        },
    }
    value = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "kind": AUDIT_KIND,
        "audit_source_sha256": AUDIT_SOURCE_SHA256,
        "candidate": {**common, "role": "train"},
        "frozen_validation": {**common, "role": "validation"},
        "benchmark_registry": {
            "path": str(registry.resolve()),
            "sha256": sha256_file(registry),
        },
        "findings": _identity(findings, relative=findings.name),
        "rejection_ledger": {
            **_identity(rejections, relative=rejections.name),
            "complete": True,
        },
        "gates": gates,
        "ready_for_training": True,
    }
    value["attestation_fingerprint"] = _canonical_sha256(value)
    attestation = root / "attestation.json"
    atomic_write_json(attestation, value)
    atomic_write_json(
        root / "COMPLETE",
        {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "kind": "twen_base_corpus_audit_complete",
            "attestation": attestation.name,
            "attestation_sha256": sha256_file(attestation),
            "attestation_fingerprint": value["attestation_fingerprint"],
            "ready_for_training": True,
        },
    )
    return attestation


def _fixture_pair(
    tmp_path: Path,
    *,
    primary_train: list[tuple[str, str, int]],
    cooldown_train: list[tuple[str, str, int]],
) -> tuple[Path, Path, Path, Path]:
    validation = [("validation", "byte-preserved validation 文本", 7)]
    primary = _write_extracted(
        tmp_path / "primary",
        source_id="shared_source",
        train=primary_train,
        validation=validation,
    )
    cooldown = _write_extracted(
        tmp_path / "cooldown",
        source_id="shared_source",
        train=cooldown_train,
        validation=validation,
    )
    return (
        primary,
        _write_audit(tmp_path / "primary-audit", primary),
        cooldown,
        _write_audit(tmp_path / "cooldown-audit", cooldown),
    )


def _materialize(
    pair: tuple[Path, Path, Path, Path],
    output: Path,
) -> Path:
    primary, primary_audit, cooldown, cooldown_audit = pair
    return phase_exclusion.materialize_phase_excluded_cooldown(
        primary_manifest=primary,
        primary_audit=primary_audit,
        cooldown_manifest=cooldown,
        cooldown_audit=cooldown_audit,
        output_root=output,
    )


def _texts(manifest: Path, role: str) -> list[str]:
    value = json.loads(manifest.read_text())
    result: list[str] = []
    for entry in value[f"{role}_files"]:
        path = manifest.parent / entry["path"]
        result.extend(json.loads(line)["text"] for line in path.read_text().splitlines())
    return result


def test_phase_exclusion_contract_and_cli_are_explicit() -> None:
    assert (
        phase_exclusion.PHASE_EXCLUSION_ALGORITHM
        == "source-scoped-authenticated-stable-id-set-difference-v1"
    )
    assert phase_exclusion.PHASE_EXCLUSION_SCHEMA_VERSION == 1
    args = build_parser().parse_args(
        [
            "data",
            "exclude-cooldown-phase",
            "--primary-manifest",
            "primary.json",
            "--primary-audit",
            "primary-audit.json",
            "--cooldown-manifest",
            "cooldown.json",
            "--cooldown-audit",
            "cooldown-audit.json",
            "--output",
            "output",
        ]
    )
    assert args.primary_manifest == "primary.json"
    assert args.cooldown_audit == "cooldown-audit.json"


def test_legal_source_scoped_set_difference_publishes_authenticated_corpus(
    tmp_path: Path,
) -> None:
    pair = _fixture_pair(
        tmp_path,
        primary_train=[("overlap", "primary text", 3)],
        cooldown_train=[
            ("overlap", "excluded cooldown text", 5),
            ("retained", "retained cooldown text", 11),
        ],
    )
    output = _materialize(pair, tmp_path / "output")
    assert _texts(output, "train") == ["retained cooldown text"]
    report = validate_extracted_base_corpus(output)
    assert report["ready_for_training"] is False
    with pytest.raises(ValueError, match="not ready_for_training"):
        _authenticate_extracted_prepare_inputs(
            output,
            role="train",
            tokenizer_sha256="3" * 64,
            allow_pending_research_audits=False,
        )
    output_value = json.loads(output.read_text())
    assert output_value["audits"]["cross_source_exact_dedup"] == (
        "pending_reaudit_phase_excluded_output"
    )
    attestation = phase_exclusion.validate_phase_exclusion_output(output)
    assert attestation["requires_independent_audit"] is True
    assert attestation["ready_for_training"] is False
    assert attestation["metrics"]["intersecting_stable_keys"] == 1
    assert attestation["metrics"]["excluded_cooldown_train_documents"] == 1
    ledger = (output.parent / "phase-exclusion-ledger.jsonl").read_text()
    assert _stable("overlap") in ledger
    assert "excluded cooldown text" not in ledger


def test_internal_stable_key_duplicates_are_members_not_rejections(
    tmp_path: Path,
) -> None:
    pair = _fixture_pair(
        tmp_path,
        primary_train=[("overlap", "primary text", 3)],
        cooldown_train=[
            ("retained-group", "retained one", 5),
            ("retained-group", "retained two", 7),
            ("overlap", "excluded one", 11),
            ("overlap", "excluded two", 13),
        ],
    )
    output = _materialize(pair, tmp_path / "output")
    assert _texts(output, "train") == ["retained one", "retained two"]
    attestation = phase_exclusion.validate_phase_exclusion_output(output)
    assert attestation["metrics"]["cooldown_duplicate_stable_key_rows"] == 2
    assert attestation["metrics"]["intersecting_stable_keys"] == 1
    assert attestation["metrics"]["excluded_cooldown_train_documents"] == 2


def test_validation_inventory_is_byte_preserved(tmp_path: Path) -> None:
    pair = _fixture_pair(
        tmp_path,
        primary_train=[("primary", "primary", 3)],
        cooldown_train=[("cooldown", "cooldown", 5)],
    )
    cooldown = pair[2]
    cooldown_value = json.loads(cooldown.read_text())
    input_validation = cooldown.parent / cooldown_value["validation_files"][0]["path"]
    before = input_validation.read_bytes()
    output = _materialize(pair, tmp_path / "output")
    output_value = json.loads(output.read_text())
    output_validation = output.parent / output_value["validation_files"][0]["path"]
    assert output_validation.read_bytes() == before
    assert output_value["validation_files"] == cooldown_value["validation_files"]


def test_input_hash_and_toctou_drift_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = _fixture_pair(
        tmp_path,
        primary_train=[("primary", "primary", 3)],
        cooldown_train=[("cooldown", "cooldown", 5)],
    )
    cooldown_value = json.loads(pair[2].read_text())
    cooldown_train = pair[2].parent / cooldown_value["train_files"][0]["path"]
    cooldown_train.write_bytes(cooldown_train.read_bytes() + b" ")
    with pytest.raises(DataAuditError, match="mismatched extracted output"):
        _materialize(pair, tmp_path / "hash-output")
    assert not (tmp_path / "hash-output").exists()

    clean_pair = _fixture_pair(
        tmp_path / "toctou",
        primary_train=[("primary", "primary", 3)],
        cooldown_train=[("cooldown", "cooldown", 5)],
    )
    original = phase_exclusion._phase_input_snapshot
    calls = 0

    def drifting_snapshot(*args: object):
        nonlocal calls
        calls += 1
        primary, cooldown = original(*args)
        if calls == 2:
            cooldown = replace(cooldown, audit_sha256="0" * 64)
        return primary, cooldown

    monkeypatch.setattr(
        phase_exclusion,
        "_phase_input_snapshot",
        drifting_snapshot,
    )
    output = tmp_path / "toctou-output"
    with pytest.raises(DataAuditError, match="input identity changed"):
        _materialize(clean_pair, output)
    assert not output.exists()
    assert not tuple(tmp_path.glob(".toctou-output.phase-exclusion-*"))


def test_symlink_and_path_escape_are_rejected(tmp_path: Path) -> None:
    pair = _fixture_pair(
        tmp_path,
        primary_train=[("primary", "primary", 3)],
        cooldown_train=[("cooldown", "cooldown", 5)],
    )
    cooldown_value = json.loads(pair[2].read_text())
    train = pair[2].parent / cooldown_value["train_files"][0]["path"]
    outside = tmp_path / "outside-train.jsonl"
    outside.write_bytes(train.read_bytes())
    train.unlink()
    train.symlink_to(outside)
    with pytest.raises(DataAuditError, match="must not be a symlink"):
        _materialize(pair, tmp_path / "symlink-output")

    escape = tmp_path / "escape"
    escape.mkdir()
    outside_manifest = tmp_path / "outside-manifest.json"
    outside_manifest.write_text("{}\n")
    manifest_link = escape / "manifest.json"
    os.symlink(outside_manifest, manifest_link)
    with pytest.raises(DataAuditError, match="manifest must not be a symlink"):
        phase_exclusion.materialize_phase_excluded_cooldown(
            primary_manifest=manifest_link,
            primary_audit=pair[1],
            cooldown_manifest=pair[2],
            cooldown_audit=pair[3],
            output_root=tmp_path / "escape-output",
        )


def test_atomic_failure_leaves_no_output_or_staging_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = _fixture_pair(
        tmp_path,
        primary_train=[("primary", "primary", 3)],
        cooldown_train=[("cooldown", "cooldown", 5)],
    )

    def fail(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        raise RuntimeError("injected attribution publication failure")

    monkeypatch.setattr(phase_exclusion, "_write_filtered_attribution", fail)
    output = tmp_path / "atomic-output"
    with pytest.raises(RuntimeError, match="injected"):
        _materialize(pair, output)
    assert not output.exists()
    assert not tuple(tmp_path.glob(".atomic-output.phase-exclusion-*"))
