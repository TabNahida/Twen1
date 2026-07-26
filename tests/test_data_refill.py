from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from twen.cli import build_parser
from twen.data.audits import (
    build_base_audit_attestation,
    materialize_filtered_base_corpus,
    validate_base_audit_attestation,
)
from twen.data.refill import (
    REFILL_SOURCE_SHA256,
    RefillError,
    build_refill_lineage,
    corpus_tokens_by_source,
    create_refill_plan,
    validate_refill_lineage,
    validate_refill_plan,
)
from twen.data.shards import read_complete_marker
from twen.data.sources import (
    build_base_jsonl_corpus,
    load_base_data_recipe,
    stable_split_for_row,
)
from twen.io.download import sha256_file

TOKENIZER_SHA = "c" * 64


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _resign_plan_bundle(plan_path: Path) -> None:
    value = json.loads(plan_path.read_text(encoding="utf-8"))
    value.pop("plan_fingerprint")
    value["plan_fingerprint"] = hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    plan_path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    complete_path = plan_path.parent / "COMPLETE"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["plan_sha256"] = sha256_file(plan_path)
    complete["plan_fingerprint"] = value["plan_fingerprint"]
    complete["frozen_validation_manifest_sha256"] = value["frozen_validation_manifest"]["sha256"]
    complete_path.write_text(
        json.dumps(complete, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [1, 2, 3]


def _write_recipe(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "twen_base_data_source_recipe",
                "recipe_id": "refill-fixture-v1",
                "split": {
                    "algorithm": "sha256_mod",
                    "seed": "refill-fixture-seed",
                    "modulus": 2,
                    "validation_remainder": 0,
                    "code_group_field": "repo_name",
                },
                "output_shard_tokens": 8,
                "profiles": {"dense": {"train_tokens": 16}},
                "validation_tokens": 8,
                "sources": [
                    {
                        "source_id": "code_fixture",
                        "category": "code",
                        "repo_id": "example/code",
                        "revision": "a" * 40,
                        "config": "native_parquet",
                        "split": "train",
                        "file_patterns": ["data/train-*.parquet"],
                        "license": "MIT",
                        "license_scope": "fixture",
                        "card_url": "https://huggingface.co/datasets/example/code",
                        "gated": False,
                        "text_field": "code",
                        "stable_id_fields": ["repo_name", "path"],
                        "required_fields": ["code", "repo_name", "path", "license"],
                        "license_field": "license",
                        "license_allowlist": ["mit"],
                        "attribution_fields": ["repo_name", "path", "license"],
                        "reject_detected_secrets": True,
                        "train_token_quotas": {"dense": 16},
                        "validation_token_quota": 8,
                        "min_characters": 4,
                        "max_document_tokens": 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_resolved(path: Path, recipe_path: Path) -> None:
    recipe = load_base_data_recipe(recipe_path)
    source = recipe.sources[0]
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "twen_resolved_base_data_sources",
                "recipe_id": recipe.recipe_id,
                "recipe_sha256": recipe.sha256,
                "sources": [
                    {
                        "source_id": source.source_id,
                        "repo_id": source.repo_id,
                        "revision": source.revision,
                        "config": source.config,
                        "split": source.split,
                        "license": source.license,
                        "files": [
                            {
                                "path": "data/train-00000.parquet",
                                "size": 8,
                                "sha256": "b" * 64,
                                "url": (
                                    "https://huggingface.co/datasets/example/code/resolve/"
                                    + "a" * 40
                                    + "/data/train-00000.parquet"
                                ),
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_registry(root: Path, benchmark: str) -> tuple[Path, Path]:
    benchmark_root = root / "benchmarks"
    benchmark_root.mkdir()
    projected = benchmark_root / "fixture.jsonl"
    projected.write_text(json.dumps({"question": benchmark}) + "\n", encoding="utf-8")
    registry = root / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "twen_benchmark_13gram_registry",
                "benchmarks": [
                    {
                        "benchmark_id": "fixture",
                        "required": True,
                        "status": "ready",
                        "revision": "d" * 40,
                        "files": [
                            {
                                "path": projected.name,
                                "size": projected.stat().st_size,
                                "sha256": sha256_file(projected),
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


def _role_rows(recipe_path: Path) -> dict[str, list[tuple[str, str]]]:
    recipe = load_base_data_recipe(recipe_path)
    source = recipe.sources[0]
    result: dict[str, list[tuple[str, str]]] = {"train": [], "validation": []}
    index = 0
    while len(result["train"]) < 8 or len(result["validation"]) < 6:
        repo = f"owner/repo-{index}"
        path = f"file-{index}.py"
        role = stable_split_for_row(
            source,
            {"repo_name": repo, "path": path},
            recipe.split,
        )[0]
        result[role].append((repo, path))
        index += 1
    return result


def test_refill_plan_hardlinks_and_preserves_original_fingerprint(tmp_path: Path) -> None:
    recipe_path = tmp_path / "recipe.json"
    resolved_path = tmp_path / "resolved.json"
    _write_recipe(recipe_path)
    _write_resolved(resolved_path, recipe_path)
    roles = _role_rows(recipe_path)
    benchmark = "one two three four five six seven eight nine ten eleven twelve thirteen"
    initial_train = [benchmark, "clean train alpha", "clean train beta", "clean train gamma"]
    initial_validation = ["clean validation alpha", "clean validation beta"]
    extra_train = ["refill train delta", "refill train epsilon", "refill train zeta"]
    extra_validation = ["refill validation gamma", "refill validation delta"]
    rows: list[dict[str, str]] = []
    role_offsets = {"train": 0, "validation": 0}
    for role, texts in (
        ("train", initial_train),
        ("validation", initial_validation),
        ("train", extra_train),
        ("validation", extra_validation),
    ):
        for text in texts:
            repo, path = roles[role][role_offsets[role]]
            role_offsets[role] += 1
            rows.append({"code": text, "repo_name": repo, "path": path, "license": "mit"})

    def row_iterator(_artifact, start_row, _columns):
        return iter(enumerate(rows[start_row:], start=start_row))

    raw_manifest = build_base_jsonl_corpus(
        recipe_path,
        resolved_path,
        tmp_path / "raw",
        tokenizer_path=tmp_path / "unused",
        tokenizer_manifest_sha256=TOKENIZER_SHA,
        profile="dense",
        progress="never",
        _tokenizer=_Tokenizer(),
        _row_iterator=row_iterator,
    )
    registry, benchmark_root = _write_registry(tmp_path, benchmark)
    attestation = build_base_audit_attestation(
        raw_manifest,
        raw_manifest,
        registry,
        benchmark_root,
        tmp_path / "audit",
    )
    assert validate_base_audit_attestation(attestation)["ready_for_training"] is False
    filtered = materialize_filtered_base_corpus(attestation, tmp_path / "filtered")
    assert corpus_tokens_by_source(filtered)["train_tokens"] == 12

    plan_path = create_refill_plan(
        audit_attestation_path=attestation,
        base_raw_manifest_path=raw_manifest,
        materialized_manifest_path=filtered,
        recipe_path=recipe_path,
        output_root=tmp_path / "plan",
    )
    plan = validate_refill_plan(plan_path)
    source_plan = plan["sources"][0]
    assert plan["a0_attestation_sha256"] == sha256_file(attestation)
    assert plan["original_raw_manifest_sha256"] == sha256_file(raw_manifest)
    assert source_plan["unique_rejections"]["train_tokens"] == 4
    assert source_plan["train"]["clean_guard_ratio"] == 0.02
    assert source_plan["train"]["survival_guard_points"] == 0.01
    assert source_plan["train"]["runtime_raw_target_tokens"] == 23
    assert source_plan["validation"]["runtime_raw_target_tokens"] == 10

    tampered_root = tmp_path / "tampered-plan"
    shutil.copytree(plan_path.parent, tampered_root)
    tampered_plan = tampered_root / "plan.json"
    tampered_value = json.loads(tampered_plan.read_text(encoding="utf-8"))
    tampered_value["frozen_validation_manifest"] = _identity(filtered)
    tampered_plan.write_text(
        json.dumps(tampered_value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _resign_plan_bundle(tampered_plan)
    with pytest.raises(RefillError, match="audit frozen validation identity differs"):
        validate_refill_plan(tampered_plan)

    missing_root = tmp_path / "missing-frozen-plan"
    shutil.copytree(plan_path.parent, missing_root)
    missing_plan = missing_root / "plan.json"
    missing_value = json.loads(missing_plan.read_text(encoding="utf-8"))
    missing_value["frozen_validation_manifest"]["path"] = str(
        tmp_path / "missing-frozen/corpus-manifest.json"
    )
    missing_plan.write_text(
        json.dumps(missing_value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _resign_plan_bundle(missing_plan)
    with pytest.raises(RefillError, match="frozen_validation_manifest is missing"):
        validate_refill_plan(missing_plan)

    raw_value = json.loads(raw_manifest.read_text(encoding="utf-8"))
    raw_manifest_sha_before_refill = sha256_file(raw_manifest)
    first_output = raw_value["sources"][0]["chunks"][0]["outputs"][0]["path"]
    original_file = raw_manifest.parent / first_output
    original_fingerprint = read_complete_marker(original_file.parent)["fingerprint"]
    merged = build_refill_lineage(
        plan_path=plan_path,
        resolved_lock_path=resolved_path,
        output_root=tmp_path / "merged",
        tokenizer_path=tmp_path / "unused",
        tokenizer_manifest_sha256=TOKENIZER_SHA,
        progress="never",
        _tokenizer=_Tokenizer(),
        _row_iterator=row_iterator,
    )
    lineage = validate_refill_lineage(merged)
    assert sha256_file(raw_manifest) == raw_manifest_sha_before_refill
    assert lineage["train_tokens"] >= 23
    assert lineage["validation_tokens"] >= 10
    linked_file = merged.parent / original_file.relative_to(raw_manifest.parent)
    assert linked_file.stat().st_dev == original_file.stat().st_dev
    assert linked_file.stat().st_ino == original_file.stat().st_ino
    merged_source = merged.parent / "extracted/code_fixture"
    new_chunks = sorted(merged_source.glob("chunk-*"))[int(source_plan["original_chunk_count"]) :]
    assert new_chunks
    assert all(
        read_complete_marker(directory)["fingerprint"] == original_fingerprint
        for directory in new_chunks
    )
    assert original_fingerprint == source_plan["original_pipeline_fingerprint"]
    merged_complete = json.loads((merged.parent / "COMPLETE").read_text(encoding="utf-8"))
    assert merged_complete["refill_lineage"]["plan_fingerprint"] == plan["plan_fingerprint"]
    assert merged_complete["refill_lineage"]["hardlink_inventory"]["sha256"]
    assert merged_complete["refill_lineage"]["new_chunk_inventory"]["sha256"]

    class InjectedFinalizationCrash(RuntimeError):
        pass

    for fault_point in ("ordinary_pair_authenticated", "lineage_manifest_written"):
        fault_root = tmp_path / f"resume-{fault_point}"

        def inject(point: str, *, expected: str = fault_point) -> None:
            if point == expected:
                raise InjectedFinalizationCrash(point)

        with pytest.raises(InjectedFinalizationCrash, match=fault_point):
            build_refill_lineage(
                plan_path=plan_path,
                resolved_lock_path=resolved_path,
                output_root=fault_root,
                tokenizer_path=tmp_path / "unused",
                tokenizer_manifest_sha256=TOKENIZER_SHA,
                progress="never",
                _tokenizer=_Tokenizer(),
                _row_iterator=row_iterator,
                _finalization_fault=inject,
            )
        interrupted_manifest = json.loads(
            (fault_root / "corpus-manifest.json").read_text(encoding="utf-8")
        )
        interrupted_complete = json.loads((fault_root / "COMPLETE").read_text(encoding="utf-8"))
        if fault_point == "ordinary_pair_authenticated":
            assert "refill_lineage" not in interrupted_manifest
        else:
            assert "refill_lineage" in interrupted_manifest
        assert "refill_lineage" not in interrupted_complete
        assert (fault_root / "REFILL_FINALIZATION.json").is_file()

        resumed = build_refill_lineage(
            plan_path=plan_path,
            resolved_lock_path=resolved_path,
            output_root=fault_root,
            tokenizer_path=tmp_path / "unused",
            tokenizer_manifest_sha256=TOKENIZER_SHA,
            progress="never",
            _tokenizer=_Tokenizer(),
            _row_iterator=row_iterator,
        )
        assert validate_refill_lineage(resumed)["ok"] is True
        assert not (fault_root / "REFILL_FINALIZATION.json").exists()

    expanded_audit = build_base_audit_attestation(
        merged,
        merged,
        registry,
        benchmark_root,
        tmp_path / "expanded-audit",
    )
    expanded_filtered = materialize_filtered_base_corpus(
        expanded_audit,
        tmp_path / "expanded-filtered",
    )
    expanded_tokens = corpus_tokens_by_source(expanded_filtered)
    assert expanded_tokens["train_tokens"] >= 16
    assert expanded_tokens["validation_tokens"] >= 8


def test_refill_plan_tampering_fails_closed(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "twen_base_corpus_refill_plan",
                "refill_source_sha256": REFILL_SOURCE_SHA256,
                "plan_fingerprint": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "COMPLETE").write_text("{}", encoding="utf-8")
    with pytest.raises(RefillError, match="fingerprint mismatch"):
        validate_refill_plan(plan)


def test_cli_refill_defaults_to_hf_direct_then_proxy_fallback() -> None:
    parser = build_parser()
    plan = parser.parse_args(
        [
            "data",
            "plan-base-refill",
            "--audit-attestation",
            "audit/attestation.json",
            "--base-raw-manifest",
            "raw/corpus-manifest.json",
            "--materialized-manifest",
            "filtered/corpus-manifest.json",
            "--output",
            "plan",
        ]
    )
    assert plan.clean_guard_ratio == 0.02
    assert plan.survival_guard_points == 0.01
    build = parser.parse_args(
        [
            "data",
            "build-base-refill",
            "--plan",
            "plan/plan.json",
            "--resolved-lock",
            "resolved.json",
            "--output",
            "raw-refill",
            "--tokenizer",
            "tokenizer",
            "--tokenizer-manifest-sha256",
            TOKENIZER_SHA,
        ]
    )
    assert build.network_policy == "fallback"
