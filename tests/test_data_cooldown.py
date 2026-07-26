from __future__ import annotations

from dataclasses import replace

import pytest

from twen.data import (
    PreparedCorpusManifest,
    PreparedShardEntry,
    TeacherKDCorpusManifest,
    TeacherKDShardEntry,
    validate_quality_cooldown_subset,
)


def _prepared_entry(
    shard_id: str,
    source_id: str,
    tensor_digit: str,
    *,
    sample_start: int,
    token_start: int,
    token_count: int,
) -> PreparedShardEntry:
    return PreparedShardEntry(
        shard_id=shard_id,
        path=shard_id,
        source_path=f"/data/{source_id}/chunk/train.jsonl",
        source_sha256=(tensor_digit.upper() * 64),
        tensors_sha256=tensor_digit * 64,
        sequence_count=2,
        token_count=token_count,
        global_sample_start=sample_start,
        global_sample_end=sample_start + 2,
        global_token_start=token_start,
        global_token_end=token_start + token_count,
    )


def _kd_entry(
    prepared: PreparedShardEntry,
    kd_digit: str,
) -> TeacherKDShardEntry:
    return TeacherKDShardEntry(
        path=prepared.shard_id,
        source_shard_id=prepared.shard_id,
        source_tensors_sha256=prepared.tensors_sha256,
        manifest_sha256=(kd_digit.upper() * 64),
        tensors_sha256=kd_digit * 64,
        global_sample_start=prepared.global_sample_start,
        global_sample_end=prepared.global_sample_end,
        global_token_start=prepared.global_token_start,
        global_token_end=prepared.global_token_end,
        sequence_count=prepared.sequence_count,
        token_count=prepared.token_count,
    )


def _ready_lineage(quality: dict[str, object]) -> dict[str, object]:
    return {
        "kind": "authenticated_extracted_corpus",
        "extracted_manifest_path": "/data/raw/corpus-manifest.json",
        "extracted_manifest_sha256": "1" * 64,
        "corpus_fingerprint": "2" * 64,
        "recipe_sha256": "3" * 64,
        "resolved_source_lock_sha256": "4" * 64,
        "tokenizer_manifest_sha256": "5" * 64,
        "extractor_source_sha256": "6" * 64,
        "recipe_id": "quality-v1",
        "profile": "quality-cooldown",
        "role": "train",
        "ready_for_data_prepare": True,
        "ready_for_training": True,
        "research_only": False,
        "audits": {"full_scan": "complete"},
        "pending_audits": [],
        "file_list": {"path": "train-files.txt", "size": 1, "sha256": "7" * 64},
        "source_files": [{"path": "source/train.jsonl", "size": 1, "sha256": "8" * 64}],
        "quality_cooldown": quality,
    }


def _contracts() -> tuple[
    PreparedCorpusManifest,
    TeacherKDCorpusManifest,
    PreparedCorpusManifest,
    TeacherKDCorpusManifest,
]:
    parent_entries = (
        _prepared_entry(
            "shard-000000",
            "math_finemath_4plus",
            "a",
            sample_start=0,
            token_start=0,
            token_count=12,
        ),
        _prepared_entry(
            "shard-000001",
            "science_openstax",
            "b",
            sample_start=2,
            token_start=12,
            token_count=14,
        ),
        _prepared_entry(
            "shard-000002",
            "web_general",
            "c",
            sample_start=4,
            token_start=26,
            token_count=16,
        ),
    )
    primary = PreparedCorpusManifest(
        dataset_fingerprint="9" * 64,
        pipeline_fingerprint="a" * 64,
        generator_source_sha256="b" * 64,
        tokenizer_sha256="5" * 64,
        sequence_length=8,
        text_field="text",
        shards=parent_entries,
        lineage=None,
    )
    primary_kd_entries = tuple(
        _kd_entry(entry, digit)
        for entry, digit in zip(parent_entries, ("d", "e", "f"), strict=True)
    )
    primary_kd = TeacherKDCorpusManifest(
        teacher_model_id="Qwen/test",
        teacher_revision="1" * 40,
        teacher_model_sha256="2" * 64,
        generator_source_sha256="3" * 64,
        tokenizer_sha256="5" * 64,
        dataset_fingerprint=primary.dataset_fingerprint,
        temperature=2.0,
        shards=primary_kd_entries,
    )

    cooldown_entries = (
        replace(
            parent_entries[0],
            global_sample_start=0,
            global_sample_end=2,
            global_token_start=0,
            global_token_end=12,
        ),
        replace(
            parent_entries[1],
            global_sample_start=2,
            global_sample_end=4,
            global_token_start=12,
            global_token_end=26,
        ),
    )
    quality = {
        "schema_version": 1,
        "kind": "authenticated_prepared_kd_subset_view",
        "eligible": True,
        "parent_prepared_manifest_sha256": "0" * 64,
        "parent_kd_manifest_sha256": "1" * 64,
        "parent_dataset_fingerprint": primary.dataset_fingerprint,
        "selection_policy_sha256": "2" * 64,
        "selection_policy_id": "math-science-reviewed-v1",
        "selection_basis": "explicit reviewed fixture policy",
        "required_cooldown_tokens": 26,
        "ordered_parent_shard_ids": [entry.shard_id for entry in cooldown_entries],
        "shard_source_ids": {
            "shard-000000": "math_finemath_4plus",
            "shard-000001": "science_openstax",
        },
        "source_mix_token_counts": {
            "math_finemath_4plus": 12,
            "science_openstax": 14,
        },
    }
    cooldown = PreparedCorpusManifest(
        dataset_fingerprint="8" * 64,
        pipeline_fingerprint="c" * 64,
        generator_source_sha256="b" * 64,
        tokenizer_sha256="5" * 64,
        sequence_length=8,
        text_field="text",
        shards=cooldown_entries,
        lineage=_ready_lineage(quality),
    )
    cooldown_kd_entries = (
        replace(
            primary_kd_entries[0],
            global_sample_start=0,
            global_sample_end=2,
            global_token_start=0,
            global_token_end=12,
        ),
        replace(
            primary_kd_entries[1],
            global_sample_start=2,
            global_sample_end=4,
            global_token_start=12,
            global_token_end=26,
        ),
    )
    cooldown_kd = replace(
        primary_kd,
        dataset_fingerprint=cooldown.dataset_fingerprint,
        shards=cooldown_kd_entries,
    )
    return primary, primary_kd, cooldown, cooldown_kd


def test_quality_cooldown_authenticates_ordered_parent_tensor_subset() -> None:
    primary, primary_kd, cooldown, cooldown_kd = _contracts()

    summary = validate_quality_cooldown_subset(
        primary,
        primary_kd,
        cooldown,
        cooldown_kd,
        primary_prepared_manifest_sha256="0" * 64,
        primary_kd_manifest_sha256="1" * 64,
        required_cooldown_tokens=26,
    )

    assert summary.selection_policy_id == "math-science-reviewed-v1"
    assert summary.selected_shard_ids == ("shard-000000", "shard-000001")
    assert summary.source_mix_token_counts == (
        ("math_finemath_4plus", 12),
        ("science_openstax", 14),
    )


def test_quality_cooldown_rejects_forged_kd_tensor() -> None:
    primary, primary_kd, cooldown, cooldown_kd = _contracts()
    forged = replace(cooldown_kd.shards[1], tensors_sha256="0" * 64)
    cooldown_kd = replace(cooldown_kd, shards=(cooldown_kd.shards[0], forged))

    with pytest.raises(ValueError, match="authenticated parent tensor"):
        validate_quality_cooldown_subset(
            primary,
            primary_kd,
            cooldown,
            cooldown_kd,
            primary_prepared_manifest_sha256="0" * 64,
            primary_kd_manifest_sha256="1" * 64,
            required_cooldown_tokens=26,
        )


def test_quality_cooldown_rejects_unverifiable_source_mix() -> None:
    primary, primary_kd, cooldown, cooldown_kd = _contracts()
    lineage = dict(cooldown.lineage or {})
    quality = dict(lineage["quality_cooldown"])
    quality["source_mix_token_counts"] = {"math_finemath_4plus": 26}
    lineage["quality_cooldown"] = quality
    cooldown = replace(cooldown, lineage=lineage)

    with pytest.raises(ValueError, match="source mix"):
        validate_quality_cooldown_subset(
            primary,
            primary_kd,
            cooldown,
            cooldown_kd,
            primary_prepared_manifest_sha256="0" * 64,
            primary_kd_manifest_sha256="1" * 64,
            required_cooldown_tokens=26,
        )
