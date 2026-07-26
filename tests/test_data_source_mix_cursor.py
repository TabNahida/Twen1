from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from twen.cli import _prepared_identity_payload
from twen.data.cursor import (
    SOURCE_MAP_ALGORITHM,
    SOURCE_MIX_ALGORITHM,
    AuthenticatedSourceMap,
    DeterministicSourceMixCooldownCursor,
    DeterministicSourceMixCursor,
)
from twen.data.prepared import PreparedCorpusManifest, PreparedShardEntry


def _sha(value: bytes | str) -> str:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _file_identity(root: Path, relative: str, content: str) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    raw = path.read_bytes()
    return {
        "path": relative,
        "size": len(raw),
        "sha256": _sha(raw),
    }


def _prepared_fixture(
    tmp_path: Path,
    *,
    omit_last_prepared_shard: bool = False,
    with_data_contract: bool = False,
) -> tuple[PreparedCorpusManifest, Path]:
    root = tmp_path / "extracted"
    root.mkdir(parents=True)
    # Opaque output names make it impossible for an implementation to recover
    # source ownership by splitting a filename or directory component.
    alpha_outputs = [
        _file_identity(root, "opaque/object-17.payload", "alpha one"),
        _file_identity(root, "opaque/object-z.payload", "alpha two"),
    ]
    beta_outputs = [
        _file_identity(root, "elsewhere/no-owner-in-name.payload", "beta"),
    ]
    train_files = [*alpha_outputs, *beta_outputs]
    extracted = {
        "schema_version": 1,
        "kind": "twen_extracted_base_jsonl_corpus",
        "corpus_fingerprint": _sha("extracted-corpus"),
        "train_files": train_files,
        "sources": [
            {
                "source_id": "alpha",
                "chunks": [
                    {
                        "outputs": [
                            alpha_outputs[0],
                            {
                                "path": "metadata/not-selected.json",
                                "size": 1,
                                "sha256": _sha("metadata"),
                            },
                        ]
                    },
                    {"outputs": [alpha_outputs[1]]},
                ],
            },
            {
                "source_id": "beta",
                "chunks": [{"outputs": beta_outputs}],
            },
        ],
    }
    data_contract: dict[str, object] | None = None
    if with_data_contract:
        source_map_unsigned = {
            "schema_version": 1,
            "algorithm": SOURCE_MAP_ALGORITHM,
            "roles": {
                "train": [
                    {"source_id": "alpha", **alpha_outputs[0]},
                    {"source_id": "alpha", **alpha_outputs[1]},
                    {"source_id": "beta", **beta_outputs[0]},
                ],
                "validation": [],
            },
        }
        source_mix_unsigned = {
            "schema_version": 1,
            "algorithm": SOURCE_MIX_ALGORITHM,
            "unit": "valid_tokens",
            "basis_points_total": 10_000,
            "profile": "fixture",
            "sources": [
                {
                    "source_id": "alpha",
                    "origin_group": "existing",
                    "mix_basis_points": 7_000,
                    "target_train_tokens": 70,
                    "actual_train_tokens": 70,
                },
                {
                    "source_id": "beta",
                    "origin_group": "new",
                    "mix_basis_points": 3_000,
                    "target_train_tokens": 30,
                    "actual_train_tokens": 30,
                },
            ],
        }
        contract_fields = {
            "source_map": {
                **source_map_unsigned,
                "fingerprint": _canonical_sha(source_map_unsigned),
            },
            "source_mix": {
                **source_mix_unsigned,
                "fingerprint": _canonical_sha(source_mix_unsigned),
            },
            "format_audit": {"complete": True},
            "license_audit": {"complete": True},
            "materialization_audit": {"complete": True},
        }
        extracted.update(contract_fields)
        data_contract_unsigned = {"schema_version": 1, **contract_fields}
        data_contract = {
            **data_contract_unsigned,
            "contract_fingerprint": _canonical_sha(data_contract_unsigned),
        }
    extracted_path = root / "corpus-manifest.json"
    extracted_path.write_text(
        json.dumps(extracted, sort_keys=True),
        encoding="utf-8",
    )
    file_list = root / "train-files.txt"
    file_list.write_text(
        "".join(f"{item['path']}\n" for item in train_files),
        encoding="utf-8",
    )
    lineage = {
        "kind": "authenticated_extracted_corpus",
        "extracted_manifest_path": str(extracted_path.resolve()),
        "extracted_manifest_sha256": _sha(extracted_path.read_bytes()),
        "corpus_fingerprint": extracted["corpus_fingerprint"],
        "recipe_id": "source-mix-fixture",
        "recipe_sha256": _sha("recipe"),
        "resolved_source_lock_sha256": _sha("source-lock"),
        "tokenizer_manifest_sha256": _sha("tokenizer"),
        "extractor_source_sha256": _sha("extractor"),
        "profile": "fixture",
        "role": "train",
        "file_list": {
            "path": file_list.name,
            "size": file_list.stat().st_size,
            "sha256": _sha(file_list.read_bytes()),
        },
        "source_files": train_files,
        "audits": {"near_dedup": "complete"},
        "pending_audits": [],
        "ready_for_data_prepare": True,
        "ready_for_training": True,
        "research_only": False,
    }
    if data_contract is not None:
        lineage["data_contract"] = data_contract
    counts = (5, 7, 11)
    prepared_entries: list[PreparedShardEntry] = []
    sample_start = 0
    token_start = 0
    selected_train_files = train_files[:-1] if omit_last_prepared_shard else train_files
    for index, (source_file, count) in enumerate(
        zip(
            selected_train_files,
            counts[: len(selected_train_files)],
            strict=True,
        )
    ):
        prepared_entries.append(
            PreparedShardEntry(
                shard_id=f"shard-{index:06d}",
                path=f"prepared-{index:06d}",
                source_path=str((root / str(source_file["path"])).resolve()),
                source_sha256=str(source_file["sha256"]),
                tensors_sha256=_sha(f"tensor-{index}"),
                sequence_count=count,
                token_count=count * 4,
                global_sample_start=sample_start,
                global_sample_end=sample_start + count,
                global_token_start=token_start,
                global_token_end=token_start + count * 4,
            )
        )
        sample_start += count
        token_start += count * 4
    prepared = PreparedCorpusManifest(
        dataset_fingerprint=_sha(f"prepared-{int(omit_last_prepared_shard)}"),
        pipeline_fingerprint=_sha("pipeline"),
        generator_source_sha256=_sha("generator"),
        tokenizer_sha256=_sha("tokenizer"),
        sequence_length=4,
        text_field="text",
        shards=tuple(prepared_entries),
        lineage=lineage,
    )
    return prepared, extracted_path


def _cursor(tmp_path: Path) -> tuple[AuthenticatedSourceMap, DeterministicSourceMixCursor]:
    prepared, _ = _prepared_fixture(tmp_path)
    source_map = AuthenticatedSourceMap.from_prepared_manifest(prepared)
    return source_map, DeterministicSourceMixCursor(
        source_map,
        {"alpha": 7_000, "beta": 3_000},
        seed=73,
    )


def _commit(
    cursor: DeterministicSourceMixCursor | DeterministicSourceMixCooldownCursor,
    references,
    valid_tokens_per_reference: list[int] | tuple[int, ...],
) -> dict[str, int]:
    by_source = dict.fromkeys(cursor.source_map.source_ids, 0)
    for reference, valid_tokens in zip(
        references,
        valid_tokens_per_reference,
        strict=True,
    ):
        by_source[reference.source_id] += valid_tokens
    assert cursor.pending_plan_fingerprint is not None
    cursor.commit(
        planned_references=references,
        plan_fingerprint=cursor.pending_plan_fingerprint,
        valid_tokens_per_reference=valid_tokens_per_reference,
        valid_tokens_by_source=by_source,
        token_count=sum(valid_tokens_per_reference),
    )
    return by_source


def _cooldown_cursor(
    tmp_path: Path,
    *,
    cooldown_start_tokens: int = 20,
) -> tuple[
    AuthenticatedSourceMap,
    AuthenticatedSourceMap,
    DeterministicSourceMixCooldownCursor,
]:
    primary_map, _ = _cursor(tmp_path)
    cooldown_map = replace(
        primary_map,
        prepared_dataset_fingerprint=_sha("cooldown-prepared"),
    )
    cursor = DeterministicSourceMixCooldownCursor(
        primary_map,
        {"alpha": 7_000, "beta": 3_000},
        cooldown_map,
        {"alpha": 4_000, "beta": 6_000},
        seed=73,
        cooldown_start_tokens=cooldown_start_tokens,
    )
    return primary_map, cooldown_map, cursor


def test_source_map_uses_authenticated_output_ownership_not_filename(
    tmp_path: Path,
) -> None:
    prepared, _ = _prepared_fixture(tmp_path)
    source_map = AuthenticatedSourceMap.from_prepared_manifest(prepared)

    assert source_map.algorithm == SOURCE_MAP_ALGORITHM
    assert source_map.prepared_dataset_fingerprint == prepared.dataset_fingerprint
    assert [(item.shard_id, item.source_id) for item in source_map.shards] == [
        ("shard-000000", "alpha"),
        ("shard-000001", "alpha"),
        ("shard-000002", "beta"),
    ]
    assert source_map.shards[-1].output_path == "elsewhere/no-owner-in-name.payload"


def test_authenticated_source_map_transport_round_trips_canonically(
    tmp_path: Path,
) -> None:
    prepared, _ = _prepared_fixture(tmp_path, with_data_contract=True)
    source_map = AuthenticatedSourceMap.from_prepared_manifest(prepared)

    restored = AuthenticatedSourceMap.from_dict(source_map.to_dict())

    assert restored == source_map
    assert restored.to_dict() == source_map.to_dict()
    assert restored.fingerprint == source_map.fingerprint


def test_prepared_cli_identity_distinguishes_extracted_and_prepared_source_maps(
    tmp_path: Path,
) -> None:
    prepared, _ = _prepared_fixture(tmp_path, with_data_contract=True)
    manifest = tmp_path / "prepared-manifest.json"
    manifest.write_text(json.dumps(prepared.to_dict()), encoding="utf-8")
    source_map = AuthenticatedSourceMap.from_prepared_manifest(prepared)
    assert prepared.lineage is not None
    data_contract = prepared.lineage["data_contract"]
    assert isinstance(data_contract, dict)
    extracted_source_map = data_contract["source_map"]
    assert isinstance(extracted_source_map, dict)

    payload = _prepared_identity_payload(manifest, prepared)

    assert payload["prepared_dataset_fingerprint"] == prepared.dataset_fingerprint
    assert payload["prepared_source_map_sha256"] == source_map.fingerprint
    assert payload["extracted_source_map_sha256"] == extracted_source_map["fingerprint"]
    assert payload["prepared_source_map_sha256"] != payload["extracted_source_map_sha256"]
    assert payload["source_mix_algorithm"] == SOURCE_MIX_ALGORITHM
    assert payload["source_mix_basis_points"] == {"alpha": 7_000, "beta": 3_000}


def test_authenticated_source_map_transport_rejects_malformed_and_bool_fields(
    tmp_path: Path,
) -> None:
    prepared, _ = _prepared_fixture(tmp_path, with_data_contract=True)
    source_map = AuthenticatedSourceMap.from_prepared_manifest(prepared)
    canonical = source_map.to_dict()

    malformed_payloads: list[tuple[dict[str, object], str]] = []

    sequence_bool = copy.deepcopy(canonical)
    sequence_bool["sequence_length"] = True
    malformed_payloads.append((sequence_bool, "positive integer"))

    ignored_shard = copy.deepcopy(canonical)
    assert isinstance(ignored_shard["shards"], list)
    ignored_shard["shards"].append(True)
    malformed_payloads.append((ignored_shard, "must be an object"))

    shard_count_bool = copy.deepcopy(canonical)
    assert isinstance(shard_count_bool["shards"], list)
    shard_count_bool["shards"][0]["sequence_count"] = True
    malformed_payloads.append((shard_count_bool, "sequence_count must be an integer"))

    shard_extra_field = copy.deepcopy(canonical)
    assert isinstance(shard_extra_field["shards"], list)
    shard_extra_field["shards"][0]["unexpected"] = "silently ignored before"
    malformed_payloads.append((shard_extra_field, "fields differ from schema"))

    coerced_source_id = copy.deepcopy(canonical)
    assert isinstance(coerced_source_id["shards"], list)
    coerced_source_id["shards"][0]["source_id"] = 7
    malformed_payloads.append((coerced_source_id, "source_id must be a string"))

    mix_weight_bool = copy.deepcopy(canonical)
    assert isinstance(mix_weight_bool["mix_basis_points"], dict)
    mix_weight_bool["mix_basis_points"]["alpha"] = True
    malformed_payloads.append((mix_weight_bool, "mix weights must be integers"))

    normalized_sha = copy.deepcopy(canonical)
    assert isinstance(normalized_sha["prepared_dataset_fingerprint"], str)
    normalized_sha["prepared_dataset_fingerprint"] = str(
        normalized_sha["prepared_dataset_fingerprint"]
    ).upper()
    malformed_payloads.append((normalized_sha, "payload is not canonical"))

    for payload, error in malformed_payloads:
        with pytest.raises(ValueError, match=error):
            AuthenticatedSourceMap.from_dict(payload)


def test_zero_history_uses_short_exact_token_target_period(tmp_path: Path) -> None:
    _, cursor = _cursor(tmp_path)
    assert cursor.interleave_period == 10

    first_period = cursor.plan_global_batch(cursor.interleave_period)
    assert Counter(item.source_id for item in first_period) == {
        "alpha": 7,
        "beta": 3,
    }
    cursor.abort_pending_plan()
    planned = cursor.plan_global_batch(1_000)
    assert Counter(item.source_id for item in planned) == {
        "alpha": 700,
        "beta": 300,
    }
    for prefix_length in range(1, 101):
        counts = Counter(item.source_id for item in planned[:prefix_length])
        assert abs(counts["alpha"] - prefix_length * 0.7) <= 1
        assert abs(counts["beta"] - prefix_length * 0.3) <= 1


def test_variable_final_sequence_lengths_drive_next_batch_token_correction(
    tmp_path: Path,
) -> None:
    _, cursor = _cursor(tmp_path)
    first = cursor.plan_global_batch(10)
    short_alpha = [1 if item.source_id == "alpha" else 4 for item in first]
    first_by_source = _commit(cursor, first, short_alpha)
    assert first_by_source == {"alpha": 7, "beta": 12}
    assert cursor.committed_tokens_by_source == first_by_source
    initial_deficit = abs(cursor.token_deficit_numerators["alpha"])

    corrected = cursor.plan_global_batch(10)
    corrected_counts = Counter(item.source_id for item in corrected)
    assert corrected_counts["alpha"] > 7
    _commit(cursor, corrected, [4] * len(corrected))

    assert abs(cursor.token_deficit_numerators["alpha"]) < initial_deficit
    assert sum(cursor.committed_tokens_by_source.values()) == cursor.committed_tokens


def test_authenticated_lineage_supplies_source_mix_without_external_weights(
    tmp_path: Path,
) -> None:
    prepared, _ = _prepared_fixture(tmp_path, with_data_contract=True)
    source_map = AuthenticatedSourceMap.from_prepared_manifest(prepared)
    assert source_map.source_mix_weights == {"alpha": 7_000, "beta": 3_000}
    cursor = DeterministicSourceMixCursor(source_map, seed=73)
    assert cursor.weights_basis_points == source_map.source_mix_weights


def test_each_source_retains_shard_block_and_local_affine_shuffle(
    tmp_path: Path,
) -> None:
    source_map, cursor = _cursor(tmp_path)
    planned = cursor.plan_global_batch(80)
    for source_id in source_map.source_ids:
        source_size = sum(
            item.sequence_count for item in source_map.shards if item.source_id == source_id
        )
        first_epoch = [
            item
            for item in planned
            if item.source_id == source_id and item.source_position < source_size
        ]
        assert [item.source_position for item in first_epoch] == list(range(source_size))
        expected_flat = {
            item.global_sample_start + offset
            for item in source_map.shards
            if item.source_id == source_id
            for offset in range(item.sequence_count)
        }
        assert {item.flat_index for item in first_epoch} == expected_flat
        shard_runs = [
            item.shard_id
            for index, item in enumerate(first_epoch)
            if index == 0 or first_epoch[index - 1].shard_id != item.shard_id
        ]
        assert len(shard_runs) == sum(item.source_id == source_id for item in source_map.shards)


def test_plan_binding_and_commit_replace_all_token_counters_atomically(
    tmp_path: Path,
) -> None:
    _, cursor = _cursor(tmp_path)
    before = cursor.state_dict()
    planned = cursor.plan_global_batch(16)
    assert planned == cursor.plan_global_batch(16)
    assert cursor.state_dict() == before
    assert cursor.pending_global_batch == planned
    assert cursor.pending_plan_fingerprint is not None

    by_source = {
        source_id: sum(4 for reference in planned if reference.source_id == source_id)
        for source_id in cursor.source_map.source_ids
    }
    with pytest.raises(ValueError, match="global token total"):
        cursor.commit(
            planned_references=planned,
            plan_fingerprint=cursor.pending_plan_fingerprint,
            valid_tokens_per_reference=[4] * len(planned),
            valid_tokens_by_source=by_source,
            token_count=0,
        )
    assert cursor.state_dict() == before

    cursor.validate_commit(
        planned_references=planned,
        plan_fingerprint=cursor.pending_plan_fingerprint,
        valid_tokens_per_reference=[4] * len(planned),
        valid_tokens_by_source=by_source,
        token_count=64,
    )
    assert cursor.state_dict() == before
    assert cursor.pending_global_batch == planned
    assert cursor.pending_plan_fingerprint is not None

    _commit(cursor, planned, [4] * len(planned))
    assert cursor.next_global_sample == 16
    assert cursor.committed_tokens == 64
    assert cursor.committed_tokens_by_source == by_source
    assert sum(cursor.source_positions.values()) == 16
    assert cursor.pending_global_batch == ()
    assert cursor.pending_plan_fingerprint is None


def test_commit_rejects_reference_fingerprint_source_and_total_tamper(
    tmp_path: Path,
) -> None:
    _, cursor = _cursor(tmp_path)
    planned = cursor.plan_global_batch(8)
    assert cursor.pending_plan_fingerprint is not None
    valid_tokens = [4] * len(planned)
    by_source = {
        source_id: sum(
            valid_tokens[index]
            for index, reference in enumerate(planned)
            if reference.source_id == source_id
        )
        for source_id in cursor.source_map.source_ids
    }
    before = cursor.state_dict()

    with pytest.raises(ValueError, match="references differ"):
        cursor.commit(
            planned_references=tuple(reversed(planned)),
            plan_fingerprint=cursor.pending_plan_fingerprint,
            valid_tokens_per_reference=valid_tokens,
            valid_tokens_by_source=by_source,
            token_count=sum(valid_tokens),
        )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        cursor.commit(
            planned_references=planned,
            plan_fingerprint="0" * 64,
            valid_tokens_per_reference=valid_tokens,
            valid_tokens_by_source=by_source,
            token_count=sum(valid_tokens),
        )
    wrong_source = dict(by_source)
    wrong_source["alpha"] += 1
    wrong_source["beta"] -= 1
    with pytest.raises(ValueError, match="reference aggregation"):
        cursor.commit(
            planned_references=planned,
            plan_fingerprint=cursor.pending_plan_fingerprint,
            valid_tokens_per_reference=valid_tokens,
            valid_tokens_by_source=wrong_source,
            token_count=sum(valid_tokens),
        )
    with pytest.raises(ValueError, match="global token total"):
        cursor.commit(
            planned_references=planned,
            plan_fingerprint=cursor.pending_plan_fingerprint,
            valid_tokens_per_reference=valid_tokens,
            valid_tokens_by_source=by_source,
            token_count=sum(valid_tokens) - 1,
        )
    assert cursor.state_dict() == before
    _commit(cursor, planned, valid_tokens)


def test_resume_restores_exact_future_and_authenticates_all_state(
    tmp_path: Path,
) -> None:
    source_map, cursor = _cursor(tmp_path)
    committed = cursor.plan_global_batch(37)
    variable_tokens = [1 + (index % source_map.sequence_length) for index in range(len(committed))]
    committed_by_source = _commit(cursor, committed, variable_tokens)
    state = cursor.state_dict()
    assert state["algorithm"] == SOURCE_MIX_ALGORITHM
    assert state["prepared_dataset_fingerprint"] == (source_map.prepared_dataset_fingerprint)
    assert state["source_map"] == source_map.to_dict()
    assert state["weights_basis_points"] == {"alpha": 7_000, "beta": 3_000}
    assert state["seed"] == 73
    assert state["next_global_sample"] == 37
    assert state["committed_tokens"] == sum(variable_tokens)
    assert state["committed_tokens_by_source"] == committed_by_source
    assert state["committed_samples_by_source"] == cursor.source_positions
    assert state["critical_lineage_fingerprint"] == (cursor.critical_lineage_fingerprint)

    restored = DeterministicSourceMixCursor.from_state_dict(
        source_map,
        {"alpha": 7_000, "beta": 3_000},
        state,
    )
    assert restored.state_dict() == state
    assert restored.plan_global_batch(41) == cursor.plan_global_batch(41)

    counter_tamper = copy.deepcopy(state)
    counter_tamper["committed_tokens"] = 1
    with pytest.raises(ValueError, match="state fingerprint"):
        DeterministicSourceMixCursor.from_state_dict(
            source_map,
            {"alpha": 7_000, "beta": 3_000},
            counter_tamper,
        )

    derived_tamper = copy.deepcopy(state)
    assert isinstance(derived_tamper["committed_samples_by_source"], dict)
    derived_tamper["committed_samples_by_source"]["alpha"] += 1
    unsigned = {key: value for key, value in derived_tamper.items() if key != "state_fingerprint"}
    derived_tamper["state_fingerprint"] = _canonical_sha(unsigned)
    with pytest.raises(ValueError, match="total differs"):
        DeterministicSourceMixCursor.from_state_dict(
            source_map,
            {"alpha": 7_000, "beta": 3_000},
            derived_tamper,
        )

    token_tamper = copy.deepcopy(state)
    assert isinstance(token_tamper["committed_tokens_by_source"], dict)
    token_tamper["committed_tokens_by_source"]["alpha"] += 1
    unsigned = {key: value for key, value in token_tamper.items() if key != "state_fingerprint"}
    token_tamper["state_fingerprint"] = _canonical_sha(unsigned)
    with pytest.raises(ValueError, match="total differs"):
        DeterministicSourceMixCursor.from_state_dict(
            source_map,
            {"alpha": 7_000, "beta": 3_000},
            token_tamper,
        )

    map_tamper = copy.deepcopy(state)
    assert isinstance(map_tamper["source_map"], dict)
    assert isinstance(map_tamper["source_map"]["shards"], list)
    map_tamper["source_map"]["shards"][0]["output_path"] = "other/output"
    unsigned = {key: value for key, value in map_tamper.items() if key != "state_fingerprint"}
    map_tamper["state_fingerprint"] = _canonical_sha(unsigned)
    with pytest.raises(ValueError, match="source map changed"):
        DeterministicSourceMixCursor.from_state_dict(
            source_map,
            {"alpha": 7_000, "beta": 3_000},
            map_tamper,
        )

    with pytest.raises(ValueError, match="weights changed"):
        DeterministicSourceMixCursor.from_state_dict(
            source_map,
            {"alpha": 8_000, "beta": 2_000},
            state,
        )


def test_world_size_changes_only_global_batch_rank_partition(tmp_path: Path) -> None:
    committed_states: list[dict[str, object]] = []
    for world_size in (1, 2, 3, 4, 5, 6, 10, 12):
        _, cursor = _cursor(tmp_path / f"world-{world_size}")
        prefix = cursor.plan_global_batch(13)
        _commit(cursor, prefix, [1 + index % 4 for index in range(13)])
        expected = cursor.plan_global_batch(60)
        expected_fingerprint = cursor.pending_plan_fingerprint
        rank_items = [
            item
            for rank in range(world_size)
            for item in cursor.plan_rank_batch(
                60,
                rank=rank,
                world_size=world_size,
            )
        ]
        assert sorted(
            rank_items,
            key=lambda item: item.global_position,
        ) == list(expected)
        assert cursor.pending_plan_fingerprint == expected_fingerprint
        globally_ordered = sorted(
            rank_items,
            key=lambda item: item.global_position,
        )
        _commit(
            cursor,
            globally_ordered,
            [1 + item.global_position % 4 for item in globally_ordered],
        )
        committed_states.append(cursor.state_dict())
    assert all(state == committed_states[0] for state in committed_states[1:])


def test_source_mix_cooldown_switches_only_after_crossing_batch_commit(
    tmp_path: Path,
) -> None:
    _, _, cursor = _cooldown_cursor(
        tmp_path,
        cooldown_start_tokens=20,
    )

    first = cursor.plan_global_batch(4)
    assert cursor.active_phase == "primary"
    assert [item.global_position for item in first] == list(range(4))
    _commit(cursor, first, [4] * 4)
    assert cursor.active_phase == "primary"

    crossing = cursor.plan_global_batch(4)
    assert cursor.active_phase == "primary"
    assert all(item.source_id in {"alpha", "beta"} for item in crossing)
    _commit(cursor, crossing, [4] * 4)

    assert cursor.committed_tokens == 32
    assert cursor.active_phase == "cooldown"
    state = cursor.state_dict()
    assert state["committed_samples_by_source"] == (
        cursor.committed_samples_by_source
    )
    assert state["committed_tokens_by_source"] == cursor.committed_tokens_by_source
    assert state["phase_committed_samples_by_source"] == (
        cursor.phase_committed_samples_by_source
    )
    assert state["phase_committed_tokens_by_source"] == (
        cursor.phase_committed_tokens_by_source
    )
    cooldown = cursor.plan_global_batch(4)
    assert [item.global_position for item in cooldown] == list(range(8, 12))
    assert min(item.source_position for item in cooldown) == 0
    assert Counter(item.source_id for item in cooldown) != Counter(item.source_id for item in first)


@pytest.mark.parametrize("checkpoint_after_batches", [1, 2, 3])
def test_source_mix_cooldown_resume_is_byte_equivalent_before_and_after_boundary(
    tmp_path: Path,
    checkpoint_after_batches: int,
) -> None:
    primary_map, cooldown_map, continuous = _cooldown_cursor(
        tmp_path,
        cooldown_start_tokens=20,
    )
    batches = (
        (4, (4, 3, 2, 1)),
        (4, (4, 4, 4, 4)),
        (4, (1, 2, 3, 4)),
        (4, (4, 4, 3, 3)),
    )
    states: list[dict[str, object]] = []
    for batch_size, valid_tokens in batches:
        planned = continuous.plan_global_batch(batch_size)
        _commit(continuous, planned, valid_tokens)
        states.append(continuous.state_dict())

    resumed = DeterministicSourceMixCooldownCursor.from_state_dict(
        primary_map,
        {"alpha": 7_000, "beta": 3_000},
        cooldown_map,
        {"alpha": 4_000, "beta": 6_000},
        states[checkpoint_after_batches - 1],
        cooldown_start_tokens=20,
    )
    for batch_size, valid_tokens in batches[checkpoint_after_batches:]:
        planned = resumed.plan_global_batch(batch_size)
        _commit(resumed, planned, valid_tokens)

    assert resumed.state_dict() == continuous.state_dict()
    assert resumed.plan_global_batch(12) == continuous.plan_global_batch(12)


def test_source_mix_cooldown_resume_rejects_threshold_map_and_state_tamper(
    tmp_path: Path,
) -> None:
    primary_map, cooldown_map, cursor = _cooldown_cursor(tmp_path)
    planned = cursor.plan_global_batch(8)
    _commit(cursor, planned, [4] * 8)
    state = cursor.state_dict()

    with pytest.raises(ValueError, match="transition token changed"):
        DeterministicSourceMixCooldownCursor.from_state_dict(
            primary_map,
            {"alpha": 7_000, "beta": 3_000},
            cooldown_map,
            {"alpha": 4_000, "beta": 6_000},
            state,
            cooldown_start_tokens=21,
        )

    changed_cooldown_map = replace(
        cooldown_map,
        prepared_dataset_fingerprint=_sha("changed-cooldown"),
    )
    with pytest.raises(ValueError, match="source map changed"):
        DeterministicSourceMixCooldownCursor.from_state_dict(
            primary_map,
            {"alpha": 7_000, "beta": 3_000},
            changed_cooldown_map,
            {"alpha": 4_000, "beta": 6_000},
            state,
            cooldown_start_tokens=20,
        )

    tampered = copy.deepcopy(state)
    tampered["committed_tokens"] = 1
    with pytest.raises(ValueError, match="state fingerprint"):
        DeterministicSourceMixCooldownCursor.from_state_dict(
            primary_map,
            {"alpha": 7_000, "beta": 3_000},
            cooldown_map,
            {"alpha": 4_000, "beta": 6_000},
            tampered,
            cooldown_start_tokens=20,
        )


def test_source_mix_cooldown_world_size_only_changes_rank_partition(
    tmp_path: Path,
) -> None:
    states: list[dict[str, object]] = []
    for world_size in (1, 2, 4, 8):
        _, _, cursor = _cooldown_cursor(
            tmp_path / f"cooldown-world-{world_size}",
            cooldown_start_tokens=20,
        )
        prefix = cursor.plan_global_batch(8)
        _commit(cursor, prefix, [4] * 8)
        expected = cursor.plan_global_batch(16)
        rank_items = [
            item
            for rank in range(world_size)
            for item in cursor.plan_rank_batch(
                16,
                rank=rank,
                world_size=world_size,
            )
        ]
        ordered = sorted(rank_items, key=lambda item: item.global_position)
        assert ordered == list(expected)
        _commit(cursor, ordered, [1 + item.global_position % 4 for item in ordered])
        states.append(cursor.state_dict())
    assert all(state == states[0] for state in states[1:])


def test_source_map_and_weight_underfill_fail_closed(tmp_path: Path) -> None:
    prepared, _ = _prepared_fixture(
        tmp_path,
        omit_last_prepared_shard=True,
    )
    with pytest.raises(ValueError, match="underfills authenticated train inventory"):
        AuthenticatedSourceMap.from_prepared_manifest(prepared)

    complete_prepared, _ = _prepared_fixture(tmp_path / "complete")
    source_map = AuthenticatedSourceMap.from_prepared_manifest(complete_prepared)
    with pytest.raises(ValueError, match="underfill or overfill"):
        DeterministicSourceMixCursor(
            source_map,
            {"alpha": 7_000, "beta": 2_999},
            seed=1,
        )
    with pytest.raises(ValueError, match=r"cover .* exactly"):
        DeterministicSourceMixCursor(
            source_map,
            {"alpha": 10_000},
            seed=1,
        )


def test_extracted_manifest_and_output_metadata_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    prepared, extracted_path = _prepared_fixture(tmp_path)
    extracted = json.loads(extracted_path.read_text(encoding="utf-8"))
    extracted["unbound_tamper"] = True
    extracted_path.write_text(json.dumps(extracted, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        AuthenticatedSourceMap.from_prepared_manifest(prepared)

    prepared2, extracted_path2 = _prepared_fixture(tmp_path / "metadata")
    extracted2 = json.loads(extracted_path2.read_text(encoding="utf-8"))
    extracted2["sources"][0]["chunks"][0]["outputs"][0]["sha256"] = "0" * 64
    extracted_path2.write_text(json.dumps(extracted2, sort_keys=True), encoding="utf-8")
    assert prepared2.lineage is not None
    changed_lineage = dict(prepared2.lineage)
    changed_lineage["extracted_manifest_sha256"] = _sha(extracted_path2.read_bytes())
    rebound = replace(prepared2, lineage=changed_lineage)
    with pytest.raises(ValueError, match="output metadata mismatch"):
        AuthenticatedSourceMap.from_prepared_manifest(rebound)
