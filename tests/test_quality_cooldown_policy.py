from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from twen.cli import build_parser
from twen.data.cooldown import _validate_policy
from twen.data.prepared import PreparedCorpusManifest, PreparedShardEntry
from twen.data.quality_policy import (
    DEFAULT_QUALITY_COOLDOWN_TOKENS,
    DEFAULT_QUALITY_POLICY_ID,
    DEFAULT_QUALITY_POLICY_SEED,
    DEFAULT_QUALITY_SOURCE_TOKEN_TARGETS,
    QUALITY_POLICY_AUDIT_FILENAME,
    QUALITY_POLICY_COMPLETE_FILENAME,
    QUALITY_POLICY_FILENAME,
    QUALITY_POLICY_MANIFEST_FILENAME,
    QUALITY_POLICY_REPORT_FILENAME,
    generate_quality_cooldown_policy,
)
from twen.data.teacher_kd import (
    TeacherKDCorpusManifest,
    TeacherKDShardEntry,
)
from twen.utils import atomic_write_json, sha256_file

TARGETS = tuple((source_id, 100) for source_id, _ in DEFAULT_QUALITY_SOURCE_TOKEN_TARGETS)


@dataclass(frozen=True, slots=True)
class _PolicyFixture:
    root: Path
    prepared_path: Path
    kd_path: Path
    extracted_path: Path
    prepared: PreparedCorpusManifest
    kd: TeacherKDCorpusManifest


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _make_fixture(tmp_path: Path) -> _PolicyFixture:
    extracted_root = tmp_path / "extracted"
    extracted_root.mkdir()
    source_ids = (*(source_id for source_id, _ in TARGETS), "unused_general")
    source_files: list[dict[str, object]] = []
    extracted_sources: list[dict[str, object]] = []
    prepared_rows: list[tuple[str, str, int]] = []
    for source_index, source_id in enumerate(source_ids):
        shard_tokens = (60, 55, 50) if source_id != "unused_general" else (80,)
        outputs: list[dict[str, object]] = []
        for shard_index, token_count in enumerate(shard_tokens):
            relative = f"filtered/train/part-{source_index:03d}-{shard_index:03d}.jsonl"
            source_path = extracted_root / relative
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(
                json.dumps({"text": f"{source_id} fixture {shard_index}"}) + "\n",
                encoding="utf-8",
            )
            output = {
                "path": relative,
                "size": source_path.stat().st_size,
                "sha256": sha256_file(source_path),
            }
            outputs.append(output)
            source_files.append(output)
            prepared_rows.append((source_id, str(source_path.resolve()), token_count))
        extracted_sources.append(
            {
                "source_id": source_id,
                "chunks": [
                    {
                        "shard_id": f"chunk-{source_index:03d}",
                        "outputs": outputs,
                    }
                ],
            }
        )
    extracted_path = extracted_root / "corpus-manifest.json"
    atomic_write_json(extracted_path, {"sources": extracted_sources})
    lineage = {
        "kind": "authenticated_extracted_corpus",
        "extracted_manifest_path": str(extracted_path.resolve()),
        "extracted_manifest_sha256": sha256_file(extracted_path),
        "corpus_fingerprint": "a" * 64,
        "recipe_id": "quality-policy-fixture",
        "recipe_sha256": "b" * 64,
        "resolved_source_lock_sha256": "c" * 64,
        "tokenizer_manifest_sha256": "d" * 64,
        "extractor_source_sha256": "e" * 64,
        "profile": "dense",
        "role": "train",
        "file_list": {"path": "train-files.txt", "size": 0, "sha256": "f" * 64},
        "source_files": source_files,
        "audits": {"clean": "pass"},
        "pending_audits": [],
        "ready_for_data_prepare": True,
        "ready_for_training": True,
        "research_only": False,
    }

    prepared_entries: list[PreparedShardEntry] = []
    token_cursor = 0
    for index, (source_id, source_path, token_count) in enumerate(prepared_rows):
        shard_id = f"shard-{index:06d}"
        prepared_entries.append(
            PreparedShardEntry(
                shard_id=shard_id,
                path=shard_id,
                source_path=source_path,
                source_sha256=_sha(f"source:{source_id}:{index}"),
                tensors_sha256=_sha(f"prepared:{source_id}:{index}"),
                sequence_count=1,
                token_count=token_count,
                global_sample_start=index,
                global_sample_end=index + 1,
                global_token_start=token_cursor,
                global_token_end=token_cursor + token_count,
            )
        )
        token_cursor += token_count
    prepared = PreparedCorpusManifest(
        dataset_fingerprint="1" * 64,
        pipeline_fingerprint="2" * 64,
        generator_source_sha256="3" * 64,
        tokenizer_sha256="d" * 64,
        sequence_length=100,
        text_field="text",
        shards=tuple(prepared_entries),
        lineage=lineage,
    )
    prepared_path = tmp_path / "prepared/manifest.json"
    atomic_write_json(prepared_path, prepared.to_dict())

    kd_entries = tuple(
        TeacherKDShardEntry(
            path=entry.path,
            source_shard_id=entry.shard_id,
            source_tensors_sha256=entry.tensors_sha256,
            manifest_sha256=_sha(f"kd-manifest:{entry.shard_id}"),
            tensors_sha256=_sha(f"kd-tensors:{entry.shard_id}"),
            global_sample_start=entry.global_sample_start,
            global_sample_end=entry.global_sample_end,
            global_token_start=entry.global_token_start,
            global_token_end=entry.global_token_end,
            sequence_count=entry.sequence_count,
            token_count=entry.token_count,
        )
        for entry in prepared.shards
    )
    kd = TeacherKDCorpusManifest(
        teacher_model_id="Qwen/fixture-teacher",
        teacher_revision="4" * 40,
        teacher_model_sha256="5" * 64,
        generator_source_sha256="6" * 64,
        tokenizer_sha256=prepared.tokenizer_sha256,
        dataset_fingerprint=prepared.dataset_fingerprint,
        temperature=2.0,
        shards=kd_entries,
    )
    kd_path = tmp_path / "kd/manifest.json"
    atomic_write_json(kd_path, kd.to_dict())
    return _PolicyFixture(
        root=tmp_path,
        prepared_path=prepared_path,
        kd_path=kd_path,
        extracted_path=extracted_path,
        prepared=prepared,
        kd=kd,
    )


def _generate(
    fixture: _PolicyFixture,
    output: Path,
    *,
    approve: bool = False,
    seed: str = "fixture-seed",
    targets: tuple[tuple[str, int], ...] = TARGETS,
) -> dict[str, object]:
    with (
        patch(
            "twen.data.quality_policy.validate_prepared_corpus",
            return_value=fixture.prepared,
        ),
        patch(
            "twen.data.quality_policy.validate_kd_corpus_manifest",
            return_value=fixture.kd,
        ),
    ):
        return generate_quality_cooldown_policy(
            prepared_manifest_path=fixture.prepared_path,
            kd_manifest_path=fixture.kd_path,
            output_root=output,
            approve=approve,
            policy_id="fixture-policy-v1",
            seed=seed,
            source_token_targets=targets,
        )


def test_default_base_v2_source_targets_are_the_locked_50m_mix() -> None:
    assert DEFAULT_QUALITY_SOURCE_TOKEN_TARGETS == (
        ("english_fineweb_edu_dedup", 15_000_000),
        ("math_finemath_4plus", 15_000_000),
        ("code_github_clean_allowlisted", 7_500_000),
        ("chinese_fineweb2_cmn_hani", 5_000_000),
        ("science_cosmopedia_openstax", 5_000_000),
        ("science_cosmopedia_stanford", 2_500_000),
    )
    assert DEFAULT_QUALITY_COOLDOWN_TOKENS == 50_000_000


def test_dry_plan_is_deterministic_read_only_and_records_quota_evidence(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    output = tmp_path / "policy-output"

    first = _generate(fixture, output)
    second = _generate(fixture, output)

    assert first == second
    assert first["dry_run"] is True
    assert first["approved_for_quality_cooldown"] is False
    assert first["training_started"] is False
    assert first["teacher_kd_started"] is False
    assert first["published_files"] is None
    assert not output.exists()
    assert not (tmp_path / ".policy-output.quality-policy.lock").exists()
    assert not (tmp_path / ".policy-output.quality-policy.incomplete").exists()

    policy = first["policy"]
    assert isinstance(policy, dict)
    assert set(policy) == {
        "schema_version",
        "kind",
        "policy_id",
        "approved_for_quality_cooldown",
        "selection_basis",
        "parent_prepared_manifest_sha256",
        "parent_kd_manifest_sha256",
        "required_cooldown_tokens",
        "ordered_shards",
        "declared_source_mix_token_counts",
    }
    assert policy["approved_for_quality_cooldown"] is False
    assert policy["required_cooldown_tokens"] == 600
    assert len(policy["ordered_shards"]) == 12

    audit = first["audit"]
    assert isinstance(audit, dict)
    assert set(audit) == {
        "schema_version",
        "kind",
        "policy_id",
        "approved_for_quality_cooldown",
        "selection_plan_sha256",
        "selection_basis",
        "selection_rule",
        "inputs",
        "required_cooldown_tokens",
        "actual_cooldown_tokens",
        "overshoot_tokens",
        "selected_sequence_count",
        "selected_shard_count",
        "source_results",
        "selected_shards",
        "checks",
        "training_started",
        "teacher_kd_started",
    }
    assert audit["training_started"] is False
    assert audit["teacher_kd_started"] is False
    assert audit["actual_cooldown_tokens"] >= 600
    assert audit["overshoot_tokens"] == audit["actual_cooldown_tokens"] - 600
    for row in audit["source_results"]:
        assert row["selected_tokens"] >= row["target_tokens"] == 100
        assert row["overshoot_tokens"] == row["selected_tokens"] - 100
        assert row["selected_shard_count"] == 2
    selected = audit["selected_shards"]
    assert [row["global_order"] for row in selected] == list(range(12))
    assert [row["global_order_sha256"] for row in selected] == sorted(
        row["global_order_sha256"] for row in selected
    )


def test_selection_plan_is_bound_to_the_fixed_seed(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    first = _generate(fixture, tmp_path / "first", seed="seed-a")
    second = _generate(fixture, tmp_path / "second", seed="seed-b")

    assert first["audit"]["selection_plan_sha256"] != second["audit"]["selection_plan_sha256"]
    assert first["audit"]["selection_rule"]["seed"] == "seed-a"
    assert second["audit"]["selection_rule"]["seed"] == "seed-b"


def test_explicit_approval_atomically_publishes_closed_authenticated_bundle(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    output = tmp_path / "published-policy"

    dry = _generate(fixture, tmp_path / "dry-target")
    result = _generate(fixture, output, approve=True)

    assert result["dry_run"] is False
    assert result["approved_for_quality_cooldown"] is True
    assert result["audit"]["selection_plan_sha256"] == dry["audit"]["selection_plan_sha256"]
    assert result["policy"]["ordered_shards"] == dry["policy"]["ordered_shards"]
    expected_names = {
        QUALITY_POLICY_FILENAME,
        QUALITY_POLICY_AUDIT_FILENAME,
        QUALITY_POLICY_REPORT_FILENAME,
        QUALITY_POLICY_MANIFEST_FILENAME,
        QUALITY_POLICY_COMPLETE_FILENAME,
    }
    assert {path.name for path in output.iterdir()} == expected_names
    policy = json.loads((output / QUALITY_POLICY_FILENAME).read_text())
    audit = json.loads((output / QUALITY_POLICY_AUDIT_FILENAME).read_text())
    manifest = json.loads((output / QUALITY_POLICY_MANIFEST_FILENAME).read_text())
    complete = json.loads((output / QUALITY_POLICY_COMPLETE_FILENAME).read_text())
    assert policy["approved_for_quality_cooldown"] is True
    assert audit["approved_for_quality_cooldown"] is True
    assert set(manifest) == {
        "schema_version",
        "kind",
        "policy_id",
        "selection_plan_sha256",
        "approved_for_quality_cooldown",
        "files",
        "training_started",
        "teacher_kd_started",
    }
    assert {item["path"] for item in manifest["files"]} == {
        QUALITY_POLICY_FILENAME,
        QUALITY_POLICY_AUDIT_FILENAME,
        QUALITY_POLICY_REPORT_FILENAME,
    }
    for item in manifest["files"]:
        path = output / item["path"]
        assert item == {
            "path": item["path"],
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    assert set(complete) == {
        "schema_version",
        "kind",
        "policy_id",
        "manifest",
        "manifest_sha256",
        "approved_for_quality_cooldown",
        "training_started",
        "teacher_kd_started",
    }
    assert complete["manifest_sha256"] == sha256_file(output / QUALITY_POLICY_MANIFEST_FILENAME)
    assert complete["training_started"] is False
    assert complete["teacher_kd_started"] is False
    _validate_policy(
        output / QUALITY_POLICY_FILENAME,
        primary_prepared_sha256=sha256_file(fixture.prepared_path),
        primary_kd_sha256=sha256_file(fixture.kd_path),
        required_cooldown_tokens=600,
    )


def test_refuses_insufficient_source_duplicate_targets_and_overwrite(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    with pytest.raises(ValueError, match="requires at least 1000"):
        _generate(
            fixture,
            tmp_path / "insufficient",
            targets=((TARGETS[0][0], 1000),),
        )

    with pytest.raises(ValueError, match="duplicate quality cooldown source target"):
        generate_quality_cooldown_policy(
            prepared_manifest_path=fixture.prepared_path,
            kd_manifest_path=fixture.kd_path,
            output_root=tmp_path / "duplicate-target",
            source_token_targets=((TARGETS[0][0], 1), (TARGETS[0][0], 2)),
        )

    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "user-data"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing overwrite"):
        _generate(fixture, existing, approve=True)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_refuses_duplicate_parent_shard_and_manifest_sha_drift(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    duplicate_prepared = SimpleNamespace(
        shards=(fixture.prepared.shards[0], fixture.prepared.shards[0]),
        lineage=fixture.prepared.lineage,
        dataset_fingerprint=fixture.prepared.dataset_fingerprint,
        token_count=fixture.prepared.shards[0].token_count * 2,
    )
    with (
        patch(
            "twen.data.quality_policy.validate_prepared_corpus",
            return_value=duplicate_prepared,
        ),
        patch(
            "twen.data.quality_policy.validate_kd_corpus_manifest",
            return_value=fixture.kd,
        ),
        patch("twen.data.quality_policy.validate_kd_corpus_coverage"),
        pytest.raises(ValueError, match="duplicate parent prepared shard"),
    ):
        generate_quality_cooldown_policy(
            prepared_manifest_path=fixture.prepared_path,
            kd_manifest_path=fixture.kd_path,
            output_root=tmp_path / "duplicate-parent",
            source_token_targets=TARGETS,
        )

    def mutate_prepared(_path: Path) -> PreparedCorpusManifest:
        with fixture.prepared_path.open("a", encoding="utf-8") as handle:
            handle.write(" \n")
        return fixture.prepared

    with (
        patch(
            "twen.data.quality_policy.validate_prepared_corpus",
            side_effect=mutate_prepared,
        ),
        patch(
            "twen.data.quality_policy.validate_kd_corpus_manifest",
            return_value=fixture.kd,
        ),
        pytest.raises(ValueError, match="SHA changed during validation"),
    ):
        generate_quality_cooldown_policy(
            prepared_manifest_path=fixture.prepared_path,
            kd_manifest_path=fixture.kd_path,
            output_root=tmp_path / "sha-drift",
            source_token_targets=TARGETS,
        )


def test_cli_is_dry_by_default_and_requires_explicit_approve_flag() -> None:
    parser = build_parser()
    base = [
        "data",
        "generate-cooldown-policy",
        "--prepared-manifest",
        "prepared.json",
        "--kd-manifest",
        "kd.json",
        "--output",
        "policy-output",
    ]
    dry = parser.parse_args(base)
    approved = parser.parse_args([*base, "--approve"])

    assert dry.approve is False
    assert approved.approve is True
    assert dry.policy_id == DEFAULT_QUALITY_POLICY_ID
    assert dry.selection_seed == DEFAULT_QUALITY_POLICY_SEED
