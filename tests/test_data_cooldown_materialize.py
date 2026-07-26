from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from twen.data.cooldown import materialize_quality_cooldown_view
from twen.data.prepared import (
    PREPARED_TENSORS,
    PreparedCorpusManifest,
    PreparedShardEntry,
    prepare_jsonl_corpus,
    validate_prepared_corpus,
)
from twen.data.teacher_kd import (
    KD_GENERATOR_SOURCE_SHA256,
    KD_MANIFEST_FILENAME,
    KD_TENSORS_FILENAME,
    TeacherKDBatch,
    TeacherKDCorpusManifest,
    TeacherKDManifest,
    validate_kd_corpus_manifest,
    write_kd_corpus_manifest,
    write_kd_shard,
)
from twen.utils import atomic_write_json, sha256_file

TOKENIZER_SHA256 = "c" * 64
SOURCE_IDS = ("math_finemath_4plus", "science_openstax", "web_general")


class _TinyTokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [1 + ord(character) % 31 for character in text]


@dataclass(frozen=True, slots=True)
class _MaterializeFixture:
    root: Path
    prepared_manifest: Path
    kd_manifest: Path
    prepared: PreparedCorpusManifest
    kd: TeacherKDCorpusManifest
    source_id_by_shard: dict[str, str]


def _file_entry(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_authenticated_extracted_fixture(root: Path) -> Path:
    root.mkdir(parents=True)
    train_files: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    for index, source_id in enumerate(SOURCE_IDS):
        relative = f"extracted/{source_id}/chunk-000000/train.jsonl"
        output = root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "text": (
                        f"{source_id} reviewed training document {index}; "
                        "enough text to produce several fixed-length token rows."
                    )
                }
            )
            + "\n",
            encoding="utf-8",
        )
        entry = _file_entry(root, relative)
        train_files.append(entry)
        sources.append(
            {
                "source_id": source_id,
                "category": "quality-fixture",
                "chunks": [
                    {
                        "shard_id": "chunk-000000",
                        "outputs": [entry],
                    }
                ],
            }
        )

    validation_relative = "extracted/math_finemath_4plus/chunk-000000/validation.jsonl"
    validation_path = root / validation_relative
    validation_path.write_text(
        json.dumps({"text": "held-out validation fixture"}) + "\n",
        encoding="utf-8",
    )
    validation_files = [_file_entry(root, validation_relative)]
    sources[0]["chunks"][0]["outputs"].append(validation_files[0])  # type: ignore[index]

    inventories = {
        "train": train_files,
        "validation": validation_files,
        "attribution": [],
    }
    file_lists: dict[str, dict[str, object]] = {}
    for role, entries in inventories.items():
        sidecar = root / f"{role}-files.txt"
        sidecar.write_text(
            "".join(f"{entry['path']}\n" for entry in entries),
            encoding="utf-8",
        )
        file_lists[role] = _file_entry(root, sidecar.name)

    identity = {
        "recipe_id": "quality-cooldown-materializer-fixture",
        "recipe_sha256": "a" * 64,
        "resolved_source_lock_sha256": "b" * 64,
        "tokenizer_manifest_sha256": TOKENIZER_SHA256,
        "extractor_source_sha256": "d" * 64,
        "profile": "dense",
        "sources": sources,
        "train_files": train_files,
        "validation_files": validation_files,
        "attribution_files": [],
        "file_lists": file_lists,
    }
    corpus_fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_value = {
        "schema_version": 1,
        "kind": "twen_extracted_base_jsonl_corpus",
        **identity,
        "corpus_fingerprint": corpus_fingerprint,
        "actual_train_tokens": 300,
        "actual_validation_tokens": 30,
        "network_policy": "direct",
        "audits": {"output_sha256": "complete"},
        "ready_for_data_prepare": True,
        "ready_for_training": True,
    }
    manifest = root / "corpus-manifest.json"
    atomic_write_json(manifest, manifest_value)
    atomic_write_json(
        root / "COMPLETE",
        {
            "schema_version": 1,
            "kind": "twen_extracted_base_jsonl_complete",
            "corpus_fingerprint": corpus_fingerprint,
            "manifest": manifest.name,
            "manifest_sha256": sha256_file(manifest),
            "file_lists": file_lists,
            "ready_for_training": True,
        },
    )
    return manifest


def _write_kd_fixture(
    root: Path,
    prepared: PreparedCorpusManifest,
) -> Path:
    shard_directories: list[Path] = []
    for entry in prepared.shards:
        sequence_count = entry.sequence_count
        sequence_length = prepared.sequence_length
        manifest = TeacherKDManifest(
            teacher_model_id="Qwen/fixture-teacher",
            teacher_revision="1" * 40,
            teacher_model_sha256="2" * 64,
            generator_source_sha256=KD_GENERATOR_SOURCE_SHA256,
            tokenizer_sha256=TOKENIZER_SHA256,
            dataset_fingerprint=prepared.dataset_fingerprint,
            source_tensors_sha256=entry.tensors_sha256,
            source_shard_id=entry.shard_id,
            global_sample_start=entry.global_sample_start,
            global_sample_end=entry.global_sample_end,
            global_token_start=entry.global_token_start,
            global_token_end=entry.global_token_end,
            sequence_count=sequence_count,
            sequence_length=sequence_length,
            token_count=entry.token_count,
            vocab_size=128,
            temperature=2.0,
        )
        batch = TeacherKDBatch(
            input_ids=torch.zeros((sequence_count, sequence_length), dtype=torch.int64),
            labels=torch.zeros((sequence_count, sequence_length), dtype=torch.int64),
            attention_mask=torch.ones((sequence_count, sequence_length), dtype=torch.bool),
            topk_indices=torch.zeros((sequence_count, sequence_length, 64), dtype=torch.int64),
            topk_logits=torch.zeros((sequence_count, sequence_length, 64), dtype=torch.float16),
            teacher_logsumexp=torch.zeros((sequence_count, sequence_length), dtype=torch.float32),
            teacher_tail_logprob=torch.zeros(
                (sequence_count, sequence_length), dtype=torch.float32
            ),
            temperature=2.0,
        )
        shard_directories.append(
            write_kd_shard(root, entry.shard_id, manifest=manifest, batch=batch)
        )
    return write_kd_corpus_manifest(
        root / "manifest.json",
        shard_directories,
        expected_temperature=2.0,
        prepared_corpus=prepared,
    )


@pytest.fixture(scope="module")
def materialize_fixture(
    tmp_path_factory: pytest.TempPathFactory,
) -> _MaterializeFixture:
    root = tmp_path_factory.mktemp("quality-cooldown-materialize")
    extracted_manifest = _write_authenticated_extracted_fixture(root / "extracted")
    with (
        patch("twen.io.offline.enforce_offline_environment"),
        patch("twen.io.offline.verify_local_download_directory"),
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=_TinyTokenizer(),
        ),
    ):
        prepared_manifest = prepare_jsonl_corpus(
            None,
            root / "prepared",
            tokenizer_path=root / "tokenizer",
            tokenizer_sha256=TOKENIZER_SHA256,
            sequence_length=8,
            progress="never",
            extracted_manifest=extracted_manifest,
            role="train",
        )
    prepared = validate_prepared_corpus(prepared_manifest)
    kd_manifest = _write_kd_fixture(root / "kd", prepared)
    kd = validate_kd_corpus_manifest(kd_manifest, expected_temperature=2.0)

    source_by_path = {
        str(
            (
                extracted_manifest.parent / f"extracted/{source_id}/chunk-000000/train.jsonl"
            ).resolve()
        ): source_id
        for source_id in SOURCE_IDS
    }
    source_id_by_shard = {
        entry.shard_id: source_by_path[str(Path(entry.source_path).resolve())]
        for entry in prepared.shards
    }
    assert len(prepared.shards) == 3
    return _MaterializeFixture(
        root=root,
        prepared_manifest=prepared_manifest,
        kd_manifest=kd_manifest,
        prepared=prepared,
        kd=kd,
        source_id_by_shard=source_id_by_shard,
    )


def _write_policy(
    fixture: _MaterializeFixture,
    path: Path,
    *,
    selected: tuple[PreparedShardEntry, ...] | None = None,
    source_overrides: dict[str, str] | None = None,
    required_tokens: int | None = None,
) -> tuple[Path, int]:
    selected = selected or fixture.prepared.shards[:2]
    overrides = source_overrides or {}
    ordered: list[dict[str, str]] = []
    source_mix: dict[str, int] = {}
    for entry in selected:
        source_id = overrides.get(
            entry.shard_id,
            fixture.source_id_by_shard[entry.shard_id],
        )
        ordered.append({"shard_id": entry.shard_id, "source_id": source_id})
        source_mix[source_id] = source_mix.get(source_id, 0) + entry.token_count
    required = (
        sum(entry.token_count for entry in selected) if required_tokens is None else required_tokens
    )
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "kind": "twen_quality_cooldown_selection_policy",
            "policy_id": f"reviewed-{path.stem}",
            "approved_for_quality_cooldown": True,
            "selection_basis": "explicit reviewed CPU fixture rule",
            "parent_prepared_manifest_sha256": sha256_file(fixture.prepared_manifest),
            "parent_kd_manifest_sha256": sha256_file(fixture.kd_manifest),
            "required_cooldown_tokens": required,
            "ordered_shards": ordered,
            "declared_source_mix_token_counts": source_mix,
        },
    )
    return path, required


def _materialize(
    fixture: _MaterializeFixture,
    policy: Path,
    output: Path,
    required_tokens: int,
    *,
    dry_run: bool = False,
) -> dict[str, object]:
    return materialize_quality_cooldown_view(
        prepared_manifest_path=fixture.prepared_manifest,
        kd_manifest_path=fixture.kd_manifest,
        selection_policy_path=policy,
        output_root=output,
        required_cooldown_tokens=required_tokens,
        dry_run=dry_run,
    )


def test_materializer_dry_run_then_publishes_hardlinked_rebased_idempotent_view(
    materialize_fixture: _MaterializeFixture,
) -> None:
    fixture = materialize_fixture
    policy, required = _write_policy(fixture, fixture.root / "policy-success.json")
    output = fixture.root / "cooldown-success"
    staging = output.with_name(f".{output.name}.incomplete")

    dry = _materialize(fixture, policy, output, required, dry_run=True)
    assert dry["dry_run"] is True
    assert dry["training_started"] is False
    assert dry["gpu_kd_started"] is False
    assert not output.exists()
    assert not staging.exists()
    assert not (output.parent / f".{output.name}.materialize-cooldown.lock").exists()

    first = _materialize(fixture, policy, output, required)
    assert first["skipped_existing"] is False
    assert first["dataset_fingerprint"] == dry["dataset_fingerprint"]
    cooldown_prepared = validate_prepared_corpus(output / "prepared/manifest.json")
    cooldown_kd = validate_kd_corpus_manifest(
        output / "kd/manifest.json",
        expected_temperature=2.0,
    )
    assert cooldown_prepared.dataset_fingerprint != fixture.prepared.dataset_fingerprint
    assert cooldown_kd.dataset_fingerprint == cooldown_prepared.dataset_fingerprint

    sample_cursor = 0
    token_cursor = 0
    parent_prepared = {entry.shard_id: entry for entry in fixture.prepared.shards}
    parent_kd = {entry.source_shard_id: entry for entry in fixture.kd.shards}
    for prepared_entry, kd_entry in zip(
        cooldown_prepared.shards,
        cooldown_kd.shards,
        strict=True,
    ):
        assert prepared_entry.global_sample_start == sample_cursor
        assert prepared_entry.global_token_start == token_cursor
        assert kd_entry.global_sample_start == sample_cursor
        assert kd_entry.global_token_start == token_cursor
        sample_cursor = prepared_entry.global_sample_end
        token_cursor = prepared_entry.global_token_end

        prepared_source = (
            fixture.prepared_manifest.parent
            / parent_prepared[prepared_entry.shard_id].path
            / PREPARED_TENSORS
        )
        prepared_view = output / "prepared" / prepared_entry.path / PREPARED_TENSORS
        assert (prepared_source.stat().st_dev, prepared_source.stat().st_ino) == (
            prepared_view.stat().st_dev,
            prepared_view.stat().st_ino,
        )

        kd_source = (
            fixture.kd_manifest.parent
            / parent_kd[prepared_entry.shard_id].path
            / KD_TENSORS_FILENAME
        )
        kd_view = output / "kd" / kd_entry.path / KD_TENSORS_FILENAME
        assert (kd_source.stat().st_dev, kd_source.stat().st_ino) == (
            kd_view.stat().st_dev,
            kd_view.stat().st_ino,
        )
        assert kd_entry.manifest_sha256 == sha256_file(
            output / "kd" / kd_entry.path / KD_MANIFEST_FILENAME
        )

    second = _materialize(fixture, policy, output, required)
    assert second["skipped_existing"] is True
    assert second["dataset_fingerprint"] == first["dataset_fingerprint"]


def test_materializer_rejects_forged_source_insufficient_tokens_and_duplicate_shards(
    materialize_fixture: _MaterializeFixture,
) -> None:
    fixture = materialize_fixture
    first = fixture.prepared.shards[0]

    forged_policy, forged_required = _write_policy(
        fixture,
        fixture.root / "policy-forged-source.json",
        source_overrides={first.shard_id: "forged_source"},
    )
    with pytest.raises(ValueError, match="source_id is not authenticated"):
        _materialize(
            fixture,
            forged_policy,
            fixture.root / "cooldown-forged-source",
            forged_required,
        )

    insufficient_policy, insufficient_required = _write_policy(
        fixture,
        fixture.root / "policy-insufficient.json",
        selected=(first,),
        required_tokens=first.token_count + 1,
    )
    with pytest.raises(ValueError, match="requires at least"):
        _materialize(
            fixture,
            insufficient_policy,
            fixture.root / "cooldown-insufficient",
            insufficient_required,
        )

    duplicate_policy, duplicate_required = _write_policy(
        fixture,
        fixture.root / "policy-duplicate.json",
        selected=(first, first),
    )
    with pytest.raises(ValueError, match="repeats parent shard"):
        _materialize(
            fixture,
            duplicate_policy,
            fixture.root / "cooldown-duplicate",
            duplicate_required,
        )


@pytest.mark.parametrize("source_kind", ["prepared", "kd"])
def test_materializer_rejects_source_output_overlap(
    materialize_fixture: _MaterializeFixture,
    source_kind: str,
) -> None:
    fixture = materialize_fixture
    policy, required = _write_policy(
        fixture,
        fixture.root / f"policy-overlap-{source_kind}.json",
    )
    source_root = (
        fixture.prepared_manifest.parent
        if source_kind == "prepared"
        else fixture.kd_manifest.parent
    )
    expected_label = "prepared" if source_kind == "prepared" else "KD"
    with pytest.raises(ValueError, match=f"overlap primary {expected_label}"):
        _materialize(
            fixture,
            policy,
            source_root / "nested-cooldown-output",
            required,
        )


def test_materializer_rejects_lock_policy_collision_without_mutating_policy(
    materialize_fixture: _MaterializeFixture,
) -> None:
    fixture = materialize_fixture
    output = fixture.root / "cooldown-lock-policy-collision"
    policy_path = output.parent / f".{output.name}.materialize-cooldown.lock"
    policy, required = _write_policy(fixture, policy_path)
    original_bytes = policy.read_bytes()
    original_sha256 = sha256_file(policy)

    with pytest.raises(ValueError, match="lock path collides with selection policy"):
        _materialize(fixture, policy, output, required)

    assert policy.read_bytes() == original_bytes
    assert sha256_file(policy) == original_sha256
    assert not output.exists()
    assert not output.with_name(f".{output.name}.incomplete").exists()


def test_materializer_rejects_corrupt_existing_and_authenticated_staging_tree(
    materialize_fixture: _MaterializeFixture,
) -> None:
    fixture = materialize_fixture
    policy, required = _write_policy(fixture, fixture.root / "policy-corrupt-tree.json")
    output = fixture.root / "cooldown-corrupt-tree"
    _materialize(fixture, policy, output, required)

    (output / "unexpected.txt").write_text("not in the locked tree\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tree differs from the closed inventory"):
        _materialize(fixture, policy, output, required)

    resume_output = fixture.root / "cooldown-corrupt-staging"
    staging = resume_output.with_name(f".{resume_output.name}.incomplete")
    staging.mkdir()
    shutil.copyfile(output / "STATE.json", staging / "STATE.json")
    (staging / "unexpected.txt").write_text("stale staging payload\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unplanned artifacts"):
        _materialize(fixture, policy, resume_output, required)


def test_published_validator_does_not_recreate_a_copied_tensor(
    materialize_fixture: _MaterializeFixture,
) -> None:
    fixture = materialize_fixture
    policy, required = _write_policy(fixture, fixture.root / "policy-copied-tensor.json")
    output = fixture.root / "cooldown-copied-tensor"
    _materialize(fixture, policy, output, required)
    cooldown = validate_prepared_corpus(output / "prepared/manifest.json")
    selected = cooldown.shards[0]
    source = (
        fixture.prepared_manifest.parent
        / next(
            entry.path for entry in fixture.prepared.shards if entry.shard_id == selected.shard_id
        )
        / PREPARED_TENSORS
    )
    destination = output / "prepared" / selected.path / PREPARED_TENSORS
    destination.unlink()
    shutil.copyfile(source, destination)

    with pytest.raises(ValueError, match="not a hardlink"):
        _materialize(fixture, policy, output, required)


def test_published_validator_rejects_a_tensor_symlink(
    materialize_fixture: _MaterializeFixture,
) -> None:
    fixture = materialize_fixture
    policy, required = _write_policy(fixture, fixture.root / "policy-tensor-symlink.json")
    output = fixture.root / "cooldown-tensor-symlink"
    _materialize(fixture, policy, output, required)
    cooldown = validate_prepared_corpus(output / "prepared/manifest.json")
    selected = cooldown.shards[0]
    source = (
        fixture.prepared_manifest.parent
        / next(
            entry.path for entry in fixture.prepared.shards if entry.shard_id == selected.shard_id
        )
        / PREPARED_TENSORS
    )
    destination = output / "prepared" / selected.path / PREPARED_TENSORS
    destination.unlink()
    destination.symlink_to(source)

    with pytest.raises(ValueError, match=r"(symlink|escapes output root)"):
        _materialize(fixture, policy, output, required)


@pytest.mark.parametrize("corpus_kind", ["prepared", "kd"])
def test_materializer_rejects_corrupt_shard_complete_contract(
    materialize_fixture: _MaterializeFixture,
    corpus_kind: str,
) -> None:
    fixture = materialize_fixture
    policy, required = _write_policy(
        fixture,
        fixture.root / f"policy-corrupt-complete-{corpus_kind}.json",
    )
    output = fixture.root / f"cooldown-corrupt-complete-{corpus_kind}"
    _materialize(fixture, policy, output, required)
    cooldown = validate_prepared_corpus(output / "prepared/manifest.json")
    marker_path = output / corpus_kind / cooldown.shards[0].path / "COMPLETE"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["fingerprint"] = "f" * 64
    atomic_write_json(marker_path, marker)

    with pytest.raises(ValueError, match="shard COMPLETE"):
        _materialize(fixture, policy, output, required)
