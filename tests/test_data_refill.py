from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import twen.data.refill as refill_module
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
    EXTRACTED_CONTRACT_IDENTITY_KEYS,
    build_base_jsonl_corpus,
    load_base_data_recipe,
    resolve_base_data_sources,
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
    for identity_name, complete_name in (
        ("audit_attestation", "audit_attestation_sha256"),
        ("base_raw_manifest", "base_raw_manifest_sha256"),
        ("materialized_manifest", "materialized_manifest_sha256"),
        ("frozen_validation_manifest", "frozen_validation_manifest_sha256"),
    ):
        complete[complete_name] = value[identity_name]["sha256"]
    complete_path.write_text(
        json.dumps(complete, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _resign_extracted_manifest(manifest: Path) -> None:
    value = json.loads(manifest.read_text(encoding="utf-8"))
    identity_keys = (
        "recipe_id",
        "recipe_sha256",
        "resolved_source_lock_sha256",
        "tokenizer_manifest_sha256",
        "extractor_source_sha256",
        "profile",
        "sources",
        "train_files",
        "validation_files",
        "attribution_files",
        "file_lists",
        *EXTRACTED_CONTRACT_IDENTITY_KEYS,
    )
    identity = {name: value.get(name) for name in identity_keys}
    value["corpus_fingerprint"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    complete_path = manifest.parent / "COMPLETE"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["corpus_fingerprint"] = value["corpus_fingerprint"]
    complete["manifest_sha256"] = sha256_file(manifest)
    complete_path.write_text(
        json.dumps(complete, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [1, 2, 3]


def _write_recipe(path: Path, *, storage_format: str) -> None:
    source_path = (
        "data/train-00000.json.gz" if storage_format == "jsonl_gzip" else "data/train-00000.parquet"
    )
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "schema_status": "stable",
                "kind": "twen_base_data_source_recipe_v2",
                "recipe_id": "refill-fixture-v2",
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
                "mix_contract": {
                    "basis_points_total": 10_000,
                    "existing_sources_basis_points": 10_000,
                    "new_sources_basis_points": 0,
                },
                "license_policy": {
                    "canonical_permissive_allowlist": ["mit"],
                },
                "sources": [
                    {
                        "source_id": "code_fixture",
                        "origin_group": "existing",
                        "mix_basis_points": 10_000,
                        "category": "code",
                        "repo_id": "example/code",
                        "revision": "a" * 40,
                        "config": "native_parquet",
                        "split": "train",
                        "storage_format": storage_format,
                        "file_patterns": [source_path],
                        "locked_files": [
                            {
                                "path": source_path,
                                "size": 8,
                                "sha256": "b" * 64,
                            }
                        ],
                        "license_declaration": "MIT",
                        "license_scope": "fixture",
                        "card_url": "https://huggingface.co/datasets/example/code",
                        "gated": False,
                        "trust_remote_code": False,
                        "text_field": "code",
                        "stable_id_fields": ["repo_name", "path"],
                        "split_group_fields": ["repo_name"],
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
    source = load_base_data_recipe(recipe_path).sources[0]
    resolve_base_data_sources(
        recipe_path,
        path,
        metadata_fetcher=lambda _repo_id, _revision: {
            "sha": "a" * 40,
            "siblings": [
                {
                    "rfilename": source.locked_files[0].path,
                    "lfs": {
                        "size": 8,
                        "oid": "sha256:" + "b" * 64,
                    },
                }
            ],
        },
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


@pytest.mark.parametrize("storage_format", ["parquet", "jsonl_gzip"])
def test_refill_plan_hardlinks_and_preserves_original_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    storage_format: str,
) -> None:
    recipe_path = tmp_path / "recipe.json"
    resolved_path = tmp_path / "resolved.json"
    _write_recipe(recipe_path, storage_format=storage_format)
    _write_resolved(resolved_path, recipe_path)
    roles = _role_rows(recipe_path)
    benchmark = "one two three four five six seven eight nine ten eleven twelve thirteen"
    initial_train = [benchmark, "clean train alpha", "clean train beta", "clean train gamma"]
    initial_validation = ["clean validation alpha", "clean validation beta"]
    extra_train = ["refill train delta", "refill train epsilon", "refill train zeta"]
    extra_validation = [
        "refill validation gamma",
        "refill validation delta",
        "refill validation epsilon",
        "refill validation zeta",
    ]
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
    assert corpus_tokens_by_source(filtered)["validation_tokens"] == 8
    substituted_root = tmp_path / "substituted-filtered"
    shutil.copytree(filtered.parent, substituted_root)
    substituted_manifest = substituted_root / "corpus-manifest.json"
    substituted_value = json.loads(substituted_manifest.read_text(encoding="utf-8"))
    substituted_value["materialization_audit"]["audit_attestation_sha256"] = "0" * 64
    substituted_manifest.write_text(
        json.dumps(substituted_value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _resign_extracted_manifest(substituted_manifest)
    with pytest.raises(RefillError, match="does not bind the refill projection field"):
        create_refill_plan(
            audit_attestation_path=attestation,
            base_raw_manifest_path=raw_manifest,
            materialized_manifest_path=substituted_manifest,
            recipe_path=recipe_path,
            output_root=tmp_path / "substituted-create-plan",
        )

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

    substituted_plan_root = tmp_path / "substituted-validation-plan"
    shutil.copytree(plan_path.parent, substituted_plan_root)
    substituted_plan = substituted_plan_root / "plan.json"
    substituted_plan_value = json.loads(substituted_plan.read_text(encoding="utf-8"))
    substituted_plan_value["materialized_manifest"] = _identity(substituted_manifest)
    substituted_plan.write_text(
        json.dumps(
            substituted_plan_value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _resign_plan_bundle(substituted_plan)
    with pytest.raises(RefillError, match="does not bind the refill projection field"):
        validate_refill_plan(substituted_plan)

    lowered_root = tmp_path / "lowered-target-plan"
    shutil.copytree(plan_path.parent, lowered_root)
    lowered_plan = lowered_root / "plan.json"
    lowered_value = json.loads(lowered_plan.read_text(encoding="utf-8"))
    lowered_value["sources"][0]["validation"]["runtime_raw_target_tokens"] = 1
    lowered_value["sources"][0]["validation"]["additional_raw_tokens"] = -7
    lowered_value["runtime_targets"]["validation_tokens"] = 1
    lowered_plan.write_text(
        json.dumps(lowered_value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _resign_plan_bundle(lowered_plan)
    with pytest.raises(RefillError, match="derived sources semantics changed"):
        validate_refill_plan(lowered_plan)

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
    gzip_materializations: list[dict[str, object]] = []
    local_iterations: list[tuple[Path, int, tuple[str, ...]]] = []
    parquet_iterations: list[tuple[object, int, tuple[str, ...]]] = []

    def fake_materialize(artifact, **kwargs):
        gzip_materializations.append({"artifact": artifact, **kwargs})
        return SimpleNamespace(path=tmp_path / "derived.jsonl")

    def fake_iter_local(path, start_row, columns):
        local_iterations.append((Path(path), start_row, tuple(columns)))
        return iter(enumerate(rows[start_row:], start=start_row))

    def fake_iter_remote(artifact, start_row, columns, *, file_factory):
        assert file_factory is not None
        parquet_iterations.append((artifact, start_row, tuple(columns)))
        return iter(enumerate(rows[start_row:], start=start_row))

    def forbidden_gzip_materialization(*_args, **_kwargs):
        raise AssertionError("Parquet refill must not materialize a gzip artifact")

    def forbidden_local_jsonl(*_args, **_kwargs):
        raise AssertionError("Parquet refill must not use the local JSONL reader")

    def forbidden_remote_parquet(*_args, **_kwargs):
        raise AssertionError("jsonl_gzip refill must not use the Parquet range reader")

    if storage_format == "jsonl_gzip":
        monkeypatch.setattr(
            refill_module,
            "materialize_jsonl_gzip_artifact",
            fake_materialize,
        )
        monkeypatch.setattr(refill_module, "iter_local_jsonl_rows", fake_iter_local)
        monkeypatch.setattr(
            refill_module,
            "iter_remote_parquet_rows",
            forbidden_remote_parquet,
        )
    else:
        monkeypatch.setattr(
            refill_module,
            "materialize_jsonl_gzip_artifact",
            forbidden_gzip_materialization,
        )
        monkeypatch.setattr(
            refill_module,
            "iter_local_jsonl_rows",
            forbidden_local_jsonl,
        )
        monkeypatch.setattr(
            refill_module,
            "iter_remote_parquet_rows",
            fake_iter_remote,
        )
    merged = build_refill_lineage(
        plan_path=plan_path,
        resolved_lock_path=resolved_path,
        output_root=tmp_path / "merged",
        tokenizer_path=tmp_path / "unused",
        tokenizer_manifest_sha256=TOKENIZER_SHA,
        progress="never",
        _tokenizer=_Tokenizer(),
    )
    if storage_format == "jsonl_gzip":
        assert gzip_materializations
        assert gzip_materializations[0]["cache_root"] == raw_manifest.parent / ".source-cache"
        assert local_iterations
        assert not parquet_iterations
    else:
        assert parquet_iterations
        assert not gzip_materializations
        assert not local_iterations
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


def test_zero_clean_validation_uses_same_source_train_survival() -> None:
    target = refill_module._guarded_target(
        role="validation",
        quota=8,
        observed_raw=8,
        observed_clean=0,
        clean_guard_ratio=0.02,
        survival_guard_points=0.01,
        zero_clean_fallback={
            "role": "train",
            "observed_raw_tokens": 16,
            "observed_clean_tokens": 12,
        },
    )
    assert target["observed_survival"] == 0.0
    assert target["planning_survival"] == 0.75
    assert target["guarded_survival"] == 0.74
    assert target["additional_raw_tokens"] == 13
    assert target["runtime_raw_target_tokens"] == 21
    assert target["survival_evidence"] == {
        "mode": "same-source-role-fallback-for-zero-clean",
        "target_role": "validation",
        "evidence_role": "train",
        "observed_raw_tokens": 16,
        "observed_clean_tokens": 12,
        "observed_survival": 0.75,
    }


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
