# ruff: noqa: RUF001

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
    content_quality_rejection_reasons,
    inspect_benchmark_registry,
    materialize_filtered_base_corpus,
    validate_base_audit_attestation,
)
from twen.data.cursor import AuthenticatedSourceMap
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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "博彩通开户注册，体育投注送彩金；博狗博彩客服提供投注平台。",
            "gambling_or_seo_stitching_spam",
        ),
        (
            "后後发發台臺万萬云雲书書乐樂国國学學后後发發台臺万萬云雲书書乐樂国國学學",
            "mixed_chinese_script_conversion_artifact",
        ),
        (
            ("这是一个足够长、应被精确识别的重复段落。" * 4)
            + "\n"
            + ("这是一个足够长、应被精确识别的重复段落。" * 4),
            "repeated_paragraph",
        ),
        ("正常开头后出现损坏字节\ufffd和锟斤拷。", "mojibake_or_garbled_text"),
        (
            "正文。\n上一篇：甲\n下一篇：乙\n相关阅读\n猜你喜欢\n"
            "扫码手机观看\n版权与免责声明\n责任编辑：某人",
            "crawler_boilerplate_or_abnormal_boundaries",
        ),
    ],
)
def test_content_quality_rejection_reasons_are_explicit(
    text: str,
    expected: str,
) -> None:
    assert expected in content_quality_rejection_reasons(text)


def test_content_quality_policy_is_conservative_for_code_and_clean_text() -> None:
    repeated_code = "def f():\\n    return 1\\n" * 10
    assert content_quality_rejection_reasons(repeated_code, category="code") == ()
    assert content_quality_rejection_reasons(
        "这是一段结构正常的中文教材正文，介绍牛顿定律及其适用条件。"
    ) == ()
    assert content_quality_rejection_reasons(
        "财政部门公布彩票公益金审计报告，并要求销售机构落实未成年人保护。"
        "报告讨论彩票监管、预算收入和公益项目绩效，不包含投注入口。"
    ) == ()
    assert content_quality_rejection_reasons(
        "本文对照繁體字與简体字的规范写法，例如「後」对应“后”、"
        "「發」对应“发”，供语言学课程查阅。"
    ) == ()


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


def _write_extracted_v2(
    root: Path,
    *,
    train_by_source: dict[str, list[str]],
    validation_by_source: dict[str, list[str]],
    mix_basis_points: dict[str, int],
) -> Path:
    source_ids = sorted(train_by_source)
    assert source_ids
    assert set(validation_by_source) == set(source_ids)
    assert set(mix_basis_points) == set(source_ids)
    assert sum(mix_basis_points.values()) == 10_000
    root.mkdir(parents=True)
    inventories: dict[str, list[dict[str, object]]] = {
        "train": [],
        "validation": [],
        "attribution": [],
    }
    source_map_roles: dict[str, list[dict[str, object]]] = {
        "train": [],
        "validation": [],
    }
    sources: list[dict[str, object]] = []
    actual_tokens: dict[str, dict[str, int]] = {
        "train": {},
        "validation": {},
    }
    for source_id in source_ids:
        directory = root / f"extracted/{source_id}/chunk-000000"
        directory.mkdir(parents=True)
        outputs: list[dict[str, object]] = []
        attribution_rows: list[dict[str, object]] = []
        for role, texts in (
            ("train", train_by_source[source_id]),
            ("validation", validation_by_source[source_id]),
        ):
            output = directory / f"{role}.jsonl"
            output.write_text(
                "".join(json.dumps({"text": text}) + "\n" for text in texts),
                encoding="utf-8",
            )
            relative = output.relative_to(root).as_posix()
            entry = _entry(root, relative)
            inventories[role].append(entry)
            source_map_roles[role].append({"source_id": source_id, **entry})
            outputs.append(entry)
            token_total = 0
            for index, text in enumerate(texts):
                token_count = len(text) + 1
                token_total += token_count
                normalized = " ".join(text.strip().split())
                attribution_rows.append(
                    {
                        "source_id": source_id,
                        "split": role,
                        "stable_id": hashlib.sha256(
                            f"{source_id}\0{role}\0{index}".encode()
                        ).hexdigest(),
                        "text_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
                        "token_count_with_eos": token_count,
                        "normalized_license": "cc-by-4.0",
                    }
                )
            actual_tokens[role][source_id] = token_total
        attribution = directory / "attribution.jsonl"
        attribution.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n"
                for row in attribution_rows
            ),
            encoding="utf-8",
        )
        attribution_entry = _entry(
            root,
            attribution.relative_to(root).as_posix(),
        )
        inventories["attribution"].append(attribution_entry)
        outputs.append(attribution_entry)
        sources.append(
            {
                "source_id": source_id,
                "category": "general",
                "repo_id": f"fixture/{source_id}",
                "revision": "e" * 40,
                "config": "default",
                "split": "train",
                "storage_format": "jsonl_gzip",
                "license": "CC-BY-4.0",
                "license_scope": "per-document",
                "target_train_tokens": actual_tokens["train"][source_id],
                "actual_train_tokens": actual_tokens["train"][source_id],
                "target_validation_tokens": actual_tokens["validation"][source_id],
                "actual_validation_tokens": actual_tokens["validation"][source_id],
                "train_rows": len(train_by_source[source_id]),
                "validation_rows": len(validation_by_source[source_id]),
                "chunks": [
                    {
                        "shard_id": "chunk-000000",
                        "outputs": outputs,
                        "statistics": {},
                    }
                ],
            }
        )
    for inventory in (*inventories.values(), *source_map_roles.values()):
        inventory.sort(key=lambda item: str(item["path"]))
    file_lists: dict[str, dict[str, object]] = {}
    for role, entries in inventories.items():
        sidecar = root / f"{role}-files.txt"
        sidecar.write_text(
            "".join(f"{entry['path']}\n" for entry in entries),
            encoding="utf-8",
        )
        file_lists[role] = _entry(root, sidecar.name)
    source_map_unsigned = {
        "schema_version": 1,
        "algorithm": "authenticated-extracted-output-map-v1",
        "roles": source_map_roles,
    }
    source_map = {
        **source_map_unsigned,
        "fingerprint": hashlib.sha256(
            json.dumps(
                source_map_unsigned,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }
    source_mix_unsigned = {
        "schema_version": 1,
        "algorithm": "token-deficit-corrected-source-mix-bp-v2",
        "unit": "valid_tokens",
        "basis_points_total": 10_000,
        "profile": "dense",
        "sources": [
            {
                "source_id": source_id,
                "origin_group": "existing",
                "mix_basis_points": mix_basis_points[source_id],
                "target_train_tokens": actual_tokens["train"][source_id],
                "actual_train_tokens": actual_tokens["train"][source_id],
            }
            for source_id in source_ids
        ],
    }
    source_mix = {
        **source_mix_unsigned,
        "fingerprint": hashlib.sha256(
            json.dumps(
                source_mix_unsigned,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }
    format_audit = {
        "complete": True,
        "sources": [
            {
                "source_id": source_id,
                "storage_format": "jsonl_gzip",
                "resolved_file_count": 1,
                "resolved_bytes": 1,
            }
            for source_id in source_ids
        ],
    }
    license_audit = {
        "complete": True,
        "normalization": "canonical_allowlist_before_acceptance",
        "attribution_inventory": file_lists["attribution"],
        "sources": [
            {
                "source_id": source_id,
                "declaration": "CC-BY-4.0",
                "scope": "per-document",
                "field": "license",
                "value_mode": "canonical_after_normalization",
                "allowlist": ["cc-by-4.0"],
            }
            for source_id in source_ids
        ],
    }
    materialization_audit = {
        "complete": True,
        "network_policy": "offline-fixture",
        "sources": [
            {
                "source_id": source_id,
                "storage_format": "jsonl_gzip",
                "method": "fixture",
                "input_files": [],
                "output_chunk_count": 1,
            }
            for source_id in source_ids
        ],
    }
    identity = {
        "recipe_id": "audit-v2-fixture",
        "recipe_sha256": "a" * 64,
        "resolved_source_lock_sha256": "b" * 64,
        "tokenizer_manifest_sha256": TOKENIZER_SHA,
        "extractor_source_sha256": "d" * 64,
        "profile": "dense",
        "sources": sources,
        "train_files": inventories["train"],
        "validation_files": inventories["validation"],
        "attribution_files": inventories["attribution"],
        "file_lists": file_lists,
        "source_map": source_map,
        "source_mix": source_mix,
        "format_audit": format_audit,
        "license_audit": license_audit,
        "materialization_audit": materialization_audit,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_value = {
        "schema_version": 1,
        "kind": "twen_extracted_base_jsonl_corpus",
        **identity,
        "corpus_fingerprint": fingerprint,
        "actual_train_tokens": sum(actual_tokens["train"].values()),
        "actual_validation_tokens": sum(actual_tokens["validation"].values()),
        "network_policy": "offline-fixture",
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


def test_v2_materialization_preserves_source_contract_through_reaudit_and_prepare() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        shared = " ".join(f"shared{index}" for index in range(80))
        retained_a = (
            "A retained source-a document about ceramics, mineral glazes, firing "
            "temperatures, and careful studio practice."
        )
        retained_b = (
            "A retained source-b document about orchestral voicing, counterpoint, "
            "cadences, and acoustic balance."
        )
        candidate = _write_extracted_v2(
            root / "candidate-v2",
            train_by_source={
                "source_a": [shared, retained_a],
                "source_b": [retained_b],
            },
            validation_by_source={
                "source_a": ["unused candidate validation about botany"],
                "source_b": ["unused candidate validation about navigation"],
            },
            mix_basis_points={"source_a": 6_000, "source_b": 4_000},
        )
        frozen = _write_extracted_v2(
            root / "frozen-v2",
            train_by_source={
                "source_a": ["unused frozen train about sculpture"],
                "source_b": ["unused frozen train about architecture"],
            },
            validation_by_source={
                "source_a": [
                    shared,
                    "Frozen validation about harmonic analysis and musical form.",
                ],
                "source_b": [
                    "Frozen validation about marine ecology and coral reef surveys."
                ],
            },
            mix_basis_points={"source_a": 6_000, "source_b": 4_000},
        )
        registry, benchmark_root = _write_registry(
            root,
            benchmark_text=(
                "one two three four five six seven eight nine ten eleven twelve thirteen"
            ),
        )
        attestation = build_base_audit_attestation(
            candidate,
            frozen,
            registry,
            benchmark_root,
            root / "audit-v2",
        )
        audited = validate_base_audit_attestation(attestation)
        assert audited["ready_for_training"] is False
        assert audited["metrics"]["train_validation_exact_matches"] == 1

        filtered = materialize_filtered_base_corpus(
            attestation,
            root / "filtered-v2",
        )
        report = validate_extracted_base_corpus(filtered)
        assert report["ready_for_training"] is False
        assert report["source_map"] is not None
        assert report["source_mix"] is not None
        assert report["format_audit"]["complete"] is True
        assert report["license_audit"]["complete"] is True
        assert report["materialization_audit"]["complete"] is True
        filtered_value = json.loads(filtered.read_text(encoding="utf-8"))
        mix = {
            item["source_id"]: item
            for item in filtered_value["source_mix"]["sources"]
        }
        assert {
            source_id: item["mix_basis_points"]
            for source_id, item in mix.items()
        } == {"source_a": 6_000, "source_b": 4_000}
        assert mix["source_a"]["target_train_tokens"] == (
            len(shared) + len(retained_a) + 2
        )
        assert mix["source_a"]["actual_train_tokens"] == len(retained_a) + 1
        assert mix["source_b"]["actual_train_tokens"] == len(retained_b) + 1
        assert (
            filtered_value["license_audit"]["attribution_inventory"]
            == filtered_value["file_lists"]["attribution"]
        )
        assert filtered_value["materialization_audit"][
            "parent_candidate_manifest_sha256"
        ] == sha256_file(candidate)
        assert filtered_value["materialization_audit"][
            "parent_frozen_validation_manifest_sha256"
        ] == sha256_file(frozen)
        assert filtered_value["materialization_audit"][
            "audit_attestation_sha256"
        ] == sha256_file(attestation)

        rescanned = build_base_audit_attestation(
            filtered,
            filtered,
            registry,
            benchmark_root,
            root / "filtered-v2-audit",
        )
        assert validate_base_audit_attestation(rescanned)["ready_for_training"] is True
        with (
            patch("twen.io.offline.enforce_offline_environment"),
            patch("twen.io.offline.verify_local_download_directory"),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=_Tokenizer()),
        ):
            prepared_path = prepare_jsonl_corpus(
                None,
                root / "prepared-v2",
                tokenizer_path=root / "tokenizer",
                tokenizer_sha256=TOKENIZER_SHA,
                sequence_length=32,
                progress="never",
                extracted_manifest=filtered,
                role="train",
                audit_attestation=rescanned,
            )
        prepared = validate_prepared_corpus(prepared_path)
        authenticated = AuthenticatedSourceMap.from_prepared_manifest(prepared)
        assert authenticated.source_ids == ("source_a", "source_b")
        assert authenticated.source_mix_weights == {
            "source_a": 6_000,
            "source_b": 4_000,
        }


def test_v2_materialization_refuses_to_drop_a_contracted_train_source() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rejected = " ".join(f"leak{index}" for index in range(80))
        candidate = _write_extracted_v2(
            root / "candidate-v2",
            train_by_source={
                "source_a": ["Retained source-a training document about geometry."],
                "source_b": [rejected],
            },
            validation_by_source={
                "source_a": ["Unused candidate validation about poetry."],
                "source_b": ["Unused candidate validation about geology."],
            },
            mix_basis_points={"source_a": 5_000, "source_b": 5_000},
        )
        frozen = _write_extracted_v2(
            root / "frozen-v2",
            train_by_source={
                "source_a": ["Unused frozen source-a train."],
                "source_b": ["Unused frozen source-b train."],
            },
            validation_by_source={
                "source_a": ["Frozen validation about agricultural history."],
                "source_b": [rejected],
            },
            mix_basis_points={"source_a": 5_000, "source_b": 5_000},
        )
        registry, benchmark_root = _write_registry(
            root,
            benchmark_text=(
                "one two three four five six seven eight nine ten eleven twelve thirteen"
            ),
        )
        attestation = build_base_audit_attestation(
            candidate,
            frozen,
            registry,
            benchmark_root,
            root / "audit-v2",
        )
        with pytest.raises(
            DataAuditError,
            match="removed every train document from contracted source 'source_b'",
        ):
            materialize_filtered_base_corpus(attestation, root / "filtered-v2")


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
