from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CAPACITY_TEMPLATE = ROOT / "locks/base-data-sources-v4-250m.capacity-attestation.json"
READINESS_TEMPLATE = ROOT / "locks/base-dense-v4-250m-pilot.readiness.json"


def _load_script() -> ModuleType:
    path = ROOT / "scripts/close_v4_250m_formal_evidence.py"
    spec = importlib.util.spec_from_file_location(
        "close_v4_250m_formal_evidence",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


closure = _load_script()
semantic_scanner = closure._load_script(
    "audit_v4_chinese_semantic_noise.py",
    "_test_twen_v4_chinese_semantic_noise",
)
SEMANTIC_MINIMUMS = {
    "minimum_risk_samples_per_phase": 32,
    "minimum_control_samples_per_phase": 32,
}


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _fake_identity(path: str, digit: str = "a") -> dict[str, object]:
    return {
        "path": path,
        "size": 123,
        "sha256": digit * 64,
    }


def _fake_phase(
    phase: str,
    capacity: dict[str, object],
) -> dict[str, object]:
    stage = capacity["stages"][phase]  # type: ignore[index]
    rows = stage["per_source_capacity"]
    source_tokens = {
        str(row["source_id"]): int(row["required_clean_tokens"]) + 4096 for row in rows
    }
    validation_tokens = {source_id: 4096 for source_id in source_tokens}
    weights = {str(row["source_id"]): int(row["mix_basis_points"]) for row in rows}
    train_tokens = sum(source_tokens.values())
    train_samples = int(stage["required_prepared_samples"]) + 1
    prefix = f"/fixture/{phase}"
    return {
        "phase": phase,
        "extracted": {
            "manifest_path": f"{prefix}/corpus-manifest.json",
            "manifest_sha256": "1" * 64,
            "corpus_fingerprint": "2" * 64,
            "complete": _fake_identity(f"{prefix}/COMPLETE", "3"),
        },
        "audit": {
            "path": f"{prefix}/audit/attestation.json",
            "sha256": "4" * 64,
            "attestation_fingerprint": "5" * 64,
            "ready_for_training": True,
        },
        "train": SimpleNamespace(token_count=train_tokens, sequence_count=train_samples),
        "train_source_map": SimpleNamespace(source_mix_weights=weights),
        "train_prepared": {
            "manifest_path": f"{prefix}/train/manifest.json",
            "manifest_sha256": "6" * 64,
            "dataset_fingerprint": "7" * 64,
            "source_map_sha256": "8" * 64,
            "token_count": train_tokens,
            "sequence_count": train_samples,
            "available_unique_tokens": train_tokens,
            "available_unique_samples": train_samples,
            "sequence_length": 4096,
        },
        "source_tokens": source_tokens,
        "validation": SimpleNamespace(
            token_count=sum(validation_tokens.values()),
            sequence_count=len(validation_tokens),
        ),
        "validation_source_map": SimpleNamespace(source_mix_weights=weights),
        "validation_prepared": {
            "manifest_path": f"{prefix}/validation/manifest.json",
            "manifest_sha256": "9" * 64,
            "dataset_fingerprint": "a" * 64,
            "source_map_sha256": "b" * 64,
            "token_count": sum(validation_tokens.values()),
            "sequence_count": len(validation_tokens),
            "available_unique_tokens": sum(validation_tokens.values()),
            "available_unique_samples": len(validation_tokens),
            "sequence_length": 4096,
        },
        "validation_source_tokens": validation_tokens,
        "attribution": {
            **_fake_identity(f"{prefix}/attribution-files.txt", "c"),
            "wikipedia_contract": {
                "source_id": closure.CHINESE_SEMANTIC_SOURCE_ID,
                "required_fields": list(closure.WIKIPEDIA_ATTRIBUTION_FIELDS),
                "authenticated_rows": 10,
                "expected_rows": 10,
                "inventory_fingerprint": "d" * 64,
                "passed": True,
                "authorizes_training": False,
            },
        },
    }


def _passing_gate(algorithm: str, *, near: bool = False) -> dict[str, object]:
    value: dict[str, object] = {
        "algorithm": algorithm,
        "matches": 0,
        "passed": True,
    }
    if near:
        value["estimated_jaccard_threshold"] = 0.8
    return value


def _pending_semantic_gate() -> dict[str, object]:
    return {
        "required": True,
        "status": "pending_authenticated_chinese_semantic_quality_audit",
        "source_id": closure.CHINESE_SEMANTIC_SOURCE_ID,
        "required_bundle": {
            "manifest_kind": closure.CHINESE_SEMANTIC_MANIFEST_KIND,
            "complete_kind": closure.CHINESE_SEMANTIC_COMPLETE_KIND,
            "attestation_kind": closure.CHINESE_SEMANTIC_ATTESTATION_KIND,
        },
        "required_gates": {
            "all_selected_shards_authenticated": True,
            "complete_streaming_scan": True,
            "control_samples_per_phase_gte": 32,
            "high_precision_conversion_documents_eq": 0,
            "malformed_punctuation_documents_eq": 0,
            "manual_review_passed": True,
            "reviewed_at_timezone_aware_iso8601_required": True,
            "reviewer_placeholder_forbidden": True,
            "risk_samples_per_phase_gte": 32,
        },
        "observed": None,
        "passed": False,
        "authorizes_training": False,
    }


def _fake_semantic_audit() -> dict[str, object]:
    return {
        "root": "/fixture/chinese-semantic-audit",
        "source_id": closure.CHINESE_SEMANTIC_SOURCE_ID,
        "manifest": _fake_identity(
            "/fixture/chinese-semantic-audit/MANIFEST.json",
            "a",
        ),
        "complete": _fake_identity(
            "/fixture/chinese-semantic-audit/COMPLETE",
            "b",
        ),
        "attestation": _fake_identity(
            "/fixture/chinese-semantic-audit/attestation.json",
            "c",
        ),
        "attestation_fingerprint": "d" * 64,
        "inputs": {
            phase: _fake_identity(f"/fixture/{phase}/corpus-manifest.json", "e")
            for phase in ("primary", "cooldown")
        },
        "phases": {
            phase: {
                "documents": 10,
                "authenticated_shards": 1,
                "sample_count": 2,
            }
            for phase in ("primary", "cooldown")
        },
        "gates": {
            "all_selected_shards_authenticated": True,
            "complete_streaming_scan": True,
            "high_precision_conversion_documents": 0,
            "high_precision_conversion_passed": True,
            "malformed_punctuation_documents": 0,
            "malformed_punctuation_passed": True,
            "manual_review_passed": True,
        },
        "manual_review": {
            "reviewed_samples": 4,
            "unacceptable_samples": 0,
            "passed": True,
        },
        "passed": True,
        "authorizes_training": False,
    }


def _pending_readiness_fixture() -> dict[str, object]:
    readiness = _json(READINESS_TEMPLATE)
    readiness["chinese_semantic_quality_gate"] = _pending_semantic_gate()
    readiness["pause_evaluation_policy"]["controller_implemented"] = True  # type: ignore[index]
    capabilities = readiness["launch_command_capabilities"]
    capabilities["automatically_pauses_at_policy_thresholds"] = True  # type: ignore[index]
    capabilities["automatically_runs_checkpoint_validation"] = True  # type: ignore[index]
    capabilities["automatically_enforces_post_launch_hard_stops"] = True  # type: ignore[index]
    controller_path = ROOT / "scripts/govern_v4_training.py"
    readiness["governed_controller"] = {
        "path": "scripts/govern_v4_training.py",
        "sha256": hashlib.sha256(controller_path.read_bytes()).hexdigest(),
        "twen_source_tree_sha256": closure.twen_source_tree_sha256(),
        "implemented": True,
    }
    return readiness


def _fake_snapshot() -> dict[str, object]:
    capacity = _json(CAPACITY_TEMPLATE)
    readiness = _pending_readiness_fixture()
    checkpoint_sha = readiness["fork_policy"]["required_checkpoint_complete_sha256"]  # type: ignore[index]
    return {
        "capacity_template": capacity,
        "capacity_template_identity": _fake_identity(str(CAPACITY_TEMPLATE), "d"),
        "readiness_template": readiness,
        "readiness_template_identity": _fake_identity(str(READINESS_TEMPLATE), "e"),
        "blocked_config": _fake_identity(
            str(ROOT / "configs/base/dense-v4-250m-pilot.blocked.yaml"),
            "f",
        ),
        "phases": {phase: _fake_phase(phase, capacity) for phase in ("primary", "cooldown")},
        "phase_disjointness": {
            "identity": _fake_identity("/fixture/phase/attestation.json", "1"),
            "attestation_fingerprint": "2" * 64,
            "gates": {
                "stable_id_exact": _passing_gate(
                    "source-scoped-authenticated-stable-id-intersection-v1"
                ),
                "normalized_text_exact": _passing_gate(
                    "unicode-nfkc-whitespace-sha256-intersection-v1"
                ),
                "near_duplicate": _passing_gate(
                    "lexical-5gram-one-permutation-minhash-lsh-v1",
                    near=True,
                ),
            },
        },
        "formal_validation_disjointness": {
            "identity": _fake_identity("/fixture/formal/attestation.json", "3"),
            "attestation_fingerprint": "4" * 64,
            "near_duplicate_threshold": 0.8,
            "gates": {
                "train_validation_stable_id": _passing_gate(
                    "source-scoped-authenticated-stable-id-intersection-v1"
                ),
                "train_validation_normalized_exact": _passing_gate(
                    "unicode-nfkc-whitespace-sha256-intersection-v1"
                ),
                "train_validation_near_duplicate": _passing_gate(
                    "lexical-5gram-one-permutation-minhash-lsh-v1",
                    near=True,
                ),
            },
            "passed": True,
        },
        "chinese_semantic_audit": _fake_semantic_audit(),
        "formal_baseline": {
            "root": "/fixture/formal-baseline",
            "summary": _fake_identity("/fixture/formal-baseline/summary.json", "5"),
            "manifest": _fake_identity("/fixture/formal-baseline/MANIFEST.json", "6"),
            "complete": _fake_identity("/fixture/formal-baseline/COMPLETE", "7"),
            "checkpoint_complete_sha256": checkpoint_sha,
            "gate": {
                "passed": True,
                "authorizes_training": False,
                "training_started_by_summarizer": False,
            },
        },
        "closure_source_sha256": "8" * 64,
        "twen_source_tree_sha256": "9" * 64,
    }


def _closure_identity() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": closure.CLOSURE_KIND,
        "input_fingerprint": "a" * 64,
        "closure_source_sha256": "b" * 64,
        "twen_source_tree_sha256": "c" * 64,
        "launch_enabled": False,
        "authorizes_training": False,
        "training_started": False,
    }


def test_closed_outputs_pass_data_and_formal_gates_but_never_authorize_launch(
    tmp_path: Path,
) -> None:
    snapshot = _fake_snapshot()
    capacity = closure._closed_capacity(
        snapshot,
        closure_identity=_closure_identity(),
    )
    assert capacity["overall"]["passed"] is True
    assert capacity["launch_enabled"] is False
    assert capacity["authorizes_training"] is False
    assert capacity["training_started"] is False
    assert (
        capacity["phase_disjointness"]["stable_id_exact"]["algorithm"]
        == "source-scoped-authenticated-stable-id-intersection-v1"
    )
    assert set(capacity["phase_disjointness"]) == {
        "stable_id_exact",
        "normalized_text_exact",
        "near_duplicate",
    }
    assert capacity["config"]["contains_pending_identity_sentinels"] is True
    assert all(stage["passed"] is True for stage in capacity["stages"].values())
    assert all(
        stage["license_audit"]["wikipedia_attribution_contract"]["passed"] is True
        and stage["license_audit"]["wikipedia_attribution_contract"][
            "authorizes_training"
        ]
        is False
        for stage in capacity["stages"].values()
    )

    capacity_path = tmp_path / "capacity-attestation.json"
    capacity_path.write_text(json.dumps(capacity), encoding="utf-8")
    readiness = closure._closed_readiness(
        snapshot,
        closure_identity=_closure_identity(),
        capacity_path=capacity_path,
        capacity_identity={
            **_fake_identity(str(capacity_path), "d"),
            "passed": True,
            "authorizes_training": False,
        },
    )
    assert readiness["formal_validation_gate"]["passed"] is True
    assert readiness["formal_validation_gate"]["authorizes_training"] is False
    assert readiness["chinese_semantic_quality_gate"]["passed"] is True
    assert readiness["chinese_semantic_quality_gate"]["authorizes_training"] is False
    assert (
        readiness["chinese_semantic_quality_gate"]["observed"]
        == (snapshot["chinese_semantic_audit"])
    )
    assert readiness["calibration_gate"]["passed"] is False
    assert readiness["pause_evaluation_policy"]["controller_implemented"] is True
    assert readiness["launch_enabled"] is False
    assert readiness["authorizes_training"] is False
    assert readiness["launch_command_after_all_gates_pass"] is None
    assert Path(str(readiness["project_root"])).resolve() == ROOT.resolve()
    assert readiness["governed_controller"]["implemented"] is True
    assert (
        readiness["launch_command_capabilities"]["automatically_pauses_at_policy_thresholds"]
        is True
    )
    assert (
        readiness["launch_command_capabilities"]["automatically_runs_checkpoint_validation"] is True
    )
    assert (
        readiness["launch_command_capabilities"]["automatically_enforces_post_launch_hard_stops"]
        is True
    )
    assert len(readiness["blockers"]) == 4
    assert any("PENDING data identities" in item for item in readiness["blockers"])
    assert any("13M low-LR calibration" in item for item in readiness["blockers"])
    assert any("Wikipedia CC-BY-SA" in item for item in readiness["blockers"])
    assert any("final launch config" in item for item in readiness["blockers"])
    assert not any("external governed pause" in item for item in readiness["blockers"])
    assert not any("formal train/validation" in item for item in readiness["blockers"])
    assert not any("semantic quality" in item for item in readiness["blockers"])
    controller_path = Path(str(readiness["governed_controller"]["path"]))
    assert controller_path == ROOT / "scripts/govern_v4_training.py"
    assert (
        readiness["governed_controller"]["sha256"]
        == hashlib.sha256(controller_path.read_bytes()).hexdigest()
    )


def test_capacity_underfill_fails_before_any_output() -> None:
    snapshot = _fake_snapshot()
    stage = snapshot["capacity_template"]["stages"]["primary"]  # type: ignore[index]
    source = stage["per_source_capacity"][0]
    source_id = source["source_id"]
    snapshot["phases"]["primary"]["source_tokens"][source_id] = (  # type: ignore[index]
        int(source["required_clean_tokens"]) - 1
    )
    with pytest.raises(closure.ClosureError, match="governed capacity underfill"):
        closure._closed_capacity(
            snapshot,
            closure_identity=_closure_identity(),
        )


def test_validation_source_inventory_uses_validation_contract_without_train_mixing() -> None:
    source_map_unsigned = {
        "schema_version": 1,
        "algorithm": "authenticated-extracted-output-map-v1",
        "roles": {
            "train": [],
            "validation": [
                {
                    "source_id": "source-a",
                    "path": "filtered/source-a/validation.jsonl",
                    "sha256": "a" * 64,
                    "size": 11,
                },
                {
                    "source_id": "source-b",
                    "path": "filtered/source-b/validation.jsonl",
                    "sha256": "b" * 64,
                    "size": 22,
                },
            ],
        },
    }
    source_map = {
        **source_map_unsigned,
        "fingerprint": closure._canonical_sha256(source_map_unsigned),
    }
    prepared = SimpleNamespace(
        lineage={
            "role": "validation",
            "source_files": [
                {
                    "path": "filtered/source-a/validation.jsonl",
                    "sha256": "a" * 64,
                    "size": 11,
                },
                {
                    "path": "filtered/source-b/validation.jsonl",
                    "sha256": "b" * 64,
                    "size": 22,
                },
            ],
            "data_contract": {"source_map": source_map},
        },
        shards=(
            SimpleNamespace(
                shard_id="shard-a",
                source_path="/root/filtered/source-a/validation.jsonl",
                source_sha256="a" * 64,
                token_count=101,
            ),
            SimpleNamespace(
                shard_id="shard-b",
                source_path="/root/filtered/source-b/validation.jsonl",
                source_sha256="b" * 64,
                token_count=202,
            ),
        ),
        token_count=303,
    )
    validation_map, source_tokens = closure._validation_source_inventory(prepared)
    assert validation_map.fingerprint == source_map["fingerprint"]
    assert validation_map.source_ids == ("source-a", "source-b")
    assert source_tokens == {"source-a": 101, "source-b": 202}


def test_validation_source_inventory_rejects_unowned_shard() -> None:
    source_map_unsigned = {
        "schema_version": 1,
        "algorithm": "authenticated-extracted-output-map-v1",
        "roles": {
            "train": [],
            "validation": [
                {
                    "source_id": "source-a",
                    "path": "filtered/source-a/validation.jsonl",
                    "sha256": "a" * 64,
                    "size": 11,
                }
            ],
        },
    }
    source_map = {
        **source_map_unsigned,
        "fingerprint": closure._canonical_sha256(source_map_unsigned),
    }
    prepared = SimpleNamespace(
        lineage={
            "role": "validation",
            "source_files": [
                {
                    "path": "filtered/source-a/validation.jsonl",
                    "sha256": "a" * 64,
                    "size": 11,
                }
            ],
            "data_contract": {"source_map": source_map},
        },
        shards=(
            SimpleNamespace(
                shard_id="shard-x",
                source_path="/root/filtered/source-x/validation.jsonl",
                source_sha256="f" * 64,
                token_count=1,
            ),
        ),
        token_count=1,
    )
    with pytest.raises(closure.ClosureError, match="ambiguous source ownership"):
        closure._validation_source_inventory(prepared)


def test_atomic_bundle_reauthenticates_and_rejects_changed_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = _fake_snapshot()
    args = Namespace(output=tmp_path / "closed")
    monkeypatch.setattr(closure, "_authenticate_inputs", lambda _args: snapshot)
    result = closure.close_formal_evidence(args)
    output = Path(result["output"])
    assert output.is_dir()
    manifest = _json(output / "MANIFEST.json")
    complete = _json(output / "COMPLETE")
    assert manifest["launch_enabled"] is False
    assert manifest["authorizes_training"] is False
    assert manifest["inputs"]["chinese_semantic_audit"]["passed"] is True
    assert manifest["inputs"]["chinese_semantic_audit"]["authorizes_training"] is False
    assert (
        complete["manifest_sha256"]
        == hashlib.sha256((output / "MANIFEST.json").read_bytes()).hexdigest()
    )
    readiness = _json(output / "readiness.json")
    assert readiness["launch_enabled"] is False
    assert readiness["launch_command_after_all_gates_pass"] is None

    changed = copy.deepcopy(snapshot)
    changed["formal_baseline"]["summary"]["sha256"] = "0" * 64  # type: ignore[index]
    calls = iter((snapshot, changed))
    monkeypatch.setattr(closure, "_authenticate_inputs", lambda _args: next(calls))
    changed_output = tmp_path / "changed"
    with pytest.raises(closure.ClosureError, match="changed during publication"):
        closure.close_formal_evidence(Namespace(output=changed_output))
    assert not changed_output.exists()
    assert not list(tmp_path.glob(".changed.incomplete-*"))


def test_atomic_directory_publish_never_replaces_existing_empty_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "payload").write_text("authenticated", encoding="utf-8")

    with pytest.raises(closure.ClosureError, match="appeared during publication"):
        closure._rename_directory_noreplace(source, destination)

    assert source.is_dir()
    assert (source / "payload").read_text(encoding="utf-8") == "authenticated"
    assert destination.is_dir()
    assert not list(destination.iterdir())


def _bundle_identity(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _semantic_phase_inputs(tmp_path: Path) -> dict[str, dict[str, object]]:
    phases: dict[str, dict[str, object]] = {}
    for index, phase in enumerate(("primary", "cooldown"), 1):
        root = tmp_path / phase
        relative = (
            Path("filtered")
            / closure.CHINESE_SEMANTIC_SOURCE_ID
            / "chunk-000000"
            / "train.jsonl"
        )
        shard = root / relative
        shard.parent.mkdir(parents=True)
        texts = []
        if phase == "primary":
            texts.extend(
                f"这是第{risk_index}条风险复核正文"
                "\\n第一段\\n第二段\\n第三段。"
                for risk_index in range(7)
            )
        texts.extend(
            f"这是 {phase} 阶段第{ordinary_index}条结构完整、"
            "语义连贯且标点正常的中文百科正文。"
            for ordinary_index in range(64)
        )
        shard.write_text(
            "".join(
                json.dumps(
                    {"text": text},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
                for text in texts
            ),
            encoding="utf-8",
        )
        fingerprint = str(index) * 64
        path = root / "corpus-manifest.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "twen_extracted_base_jsonl_corpus",
                    "corpus_fingerprint": fingerprint,
                    "format_audit": {
                        "complete": True,
                        "filtered_outputs": {
                            "train": [
                                {
                                    "path": relative.as_posix(),
                                    "size": shard.stat().st_size,
                                    "sha256": hashlib.sha256(
                                        shard.read_bytes()
                                    ).hexdigest(),
                                    "source_id": (
                                        closure.CHINESE_SEMANTIC_SOURCE_ID
                                    ),
                                }
                            ],
                            "validation": [],
                        },
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        phases[phase] = {
            "extracted": {
                "manifest_path": str(path.resolve()),
                "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "corpus_fingerprint": fingerprint,
            }
        }
    return phases


def _resign_semantic_bundle(root: Path) -> None:
    attestation_path = root / "attestation.json"
    attestation = _json(attestation_path)
    attestation.pop("attestation_fingerprint", None)
    attestation["attestation_fingerprint"] = closure._canonical_sha256(attestation)
    attestation_path.write_text(
        json.dumps(attestation, sort_keys=True),
        encoding="utf-8",
    )

    manifest_path = root / "MANIFEST.json"
    manifest = _json(manifest_path)
    manifest["attestation_fingerprint"] = attestation["attestation_fingerprint"]
    manifest["created_at"] = attestation["created_at"]
    manifest["files"]["attestation.json"] = {  # type: ignore[index]
        key: value
        for key, value in _bundle_identity(root, "attestation.json").items()
        if key != "path"
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )

    complete = _json(root / "COMPLETE")
    complete["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (root / "COMPLETE").write_text(
        json.dumps(complete, sort_keys=True),
        encoding="utf-8",
    )


def _write_semantic_bundle(
    tmp_path: Path,
) -> tuple[Path, dict[str, dict[str, object]]]:
    phases = _semantic_phase_inputs(tmp_path)
    primary = Path(str(phases["primary"]["extracted"]["manifest_path"]))
    cooldown = Path(str(phases["cooldown"]["extracted"]["manifest_path"]))
    pending = tmp_path / "semantic-audit-pending"
    base_args = {
        "primary_manifest": primary,
        "cooldown_manifest": cooldown,
        "source_id": closure.CHINESE_SEMANTIC_SOURCE_ID,
        "risk_samples_per_phase": semantic_scanner.MIN_RISK_SAMPLES_PER_PHASE,
        "control_samples_per_phase": semantic_scanner.MIN_CONTROL_SAMPLES_PER_PHASE,
    }
    semantic_scanner.run(
        Namespace(
            **base_args,
            output=pending,
            manual_decisions=None,
        )
    )
    template = _json(pending / "manual-review-template.json")
    manual_path = tmp_path / "manual-decisions.json"
    manual_path.write_text(
        json.dumps(
            {
                "schema_version": semantic_scanner.SCHEMA_VERSION,
                "kind": semantic_scanner.DECISIONS_KIND,
                "samples_sha256": template["samples_sha256"],
                "reviewer": "fixture-reviewer",
                "reviewed_at": "2026-07-27T00:00:00+00:00",
                "decisions": [
                    {
                        "sample_id": row["sample_id"],
                        "verdict": "acceptable",
                        "notes": "",
                    }
                    for row in template["decisions"]  # type: ignore[index]
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    root = tmp_path / "semantic-audit"
    result = semantic_scanner.run(
        Namespace(
            **base_args,
            output=root,
            manual_decisions=manual_path,
        )
    )
    assert result["passed"] is True
    return root, phases


def test_chinese_semantic_audit_bundle_is_fully_authenticated(
    tmp_path: Path,
) -> None:
    root, phases = _write_semantic_bundle(tmp_path)
    value = closure._validate_chinese_semantic_audit_bundle(
        root,
        phases=phases,
        **SEMANTIC_MINIMUMS,
    )
    assert value["source_id"] == closure.CHINESE_SEMANTIC_SOURCE_ID
    assert value["passed"] is True
    assert value["authorizes_training"] is False
    assert value["manual_review"]["unacceptable_samples"] == 0
    assert value["manual_review"]["reviewed_samples"] == 128
    assert value["phases"]["primary"]["risk_population_documents"] == 7
    assert value["phases"]["primary"]["sample_strata"] == {
        "risk": 7,
        "review_challenge": 25,
        "control": 32,
    }
    assert value["phases"]["cooldown"]["risk_population_documents"] == 0
    assert value["phases"]["cooldown"]["sample_strata"] == {
        "risk": 0,
        "review_challenge": 32,
        "control": 32,
    }


@pytest.mark.parametrize(
    "quota_field",
    ("risk_samples_per_phase", "control_samples_per_phase"),
)
def test_chinese_semantic_audit_rejects_resigned_quota_below_readiness(
    tmp_path: Path,
    quota_field: str,
) -> None:
    root, phases = _write_semantic_bundle(tmp_path)
    attestation = _json(root / "attestation.json")
    attestation["samples"][quota_field] = 31  # type: ignore[index]
    (root / "attestation.json").write_text(
        json.dumps(attestation, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    _resign_semantic_bundle(root)

    with pytest.raises(
        closure.ClosureError,
        match="sample quotas are below authenticated readiness minimum",
    ):
        closure._validate_chinese_semantic_audit_bundle(
            root,
            phases=phases,
            **SEMANTIC_MINIMUMS,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("reviewer", "REPLACE_WITH_REVIEWER"),
        ("reviewed_at", "REPLACE_WITH_ISO_8601_TIMESTAMP"),
        ("reviewed_at", "2026-07-27T00:00:00"),
    ),
)
def test_chinese_semantic_audit_rejects_resigned_placeholder_review_metadata(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    root, phases = _write_semantic_bundle(tmp_path)
    attestation = _json(root / "attestation.json")
    manual_summary = attestation["manual_review"]  # type: ignore[index]
    manual_path = Path(str(manual_summary["path"]))  # type: ignore[index]
    manual = _json(manual_path)
    manual[field] = value
    manual_path.write_text(
        json.dumps(manual, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    manual_summary[field] = value  # type: ignore[index]
    manual_summary["size"] = manual_path.stat().st_size  # type: ignore[index]
    manual_summary["sha256"] = hashlib.sha256(manual_path.read_bytes()).hexdigest()  # type: ignore[index]
    (root / "attestation.json").write_text(
        json.dumps(attestation, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    _resign_semantic_bundle(root)

    with pytest.raises(
        closure.ClosureError,
        match="manual review failed recomputation",
    ):
        closure._validate_chinese_semantic_audit_bundle(
            root,
            phases=phases,
            **SEMANTIC_MINIMUMS,
        )


def test_chinese_semantic_audit_is_a_required_closure_cli_input() -> None:
    action = next(
        item for item in closure._parser()._actions if item.dest == "chinese_semantic_audit"
    )
    assert action.required is True


@pytest.mark.parametrize(
    ("mutation", "error"),
        (
            ("attestation_kind", "attestation is not a passing bound record"),
            ("authorizes_training", "attestation is not a passing bound record"),
            ("primary_identity", "deterministic recomputation"),
            ("phase_statistics", "deterministic recomputation"),
            ("statistical_gate", "statistical/manual gates did not all pass"),
            ("manual_gate", "manual review differs from recomputed samples"),
            ("scanner_path", "scanner source identity or policy differs"),
            ("scanner_sha256", "scanner source identity or policy differs"),
            ("scanner_policy", "scanner source identity or policy differs"),
    ),
)
def test_chinese_semantic_audit_rejects_resigned_nonpassing_evidence(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    root, phases = _write_semantic_bundle(tmp_path)
    attestation = _json(root / "attestation.json")
    if mutation == "attestation_kind":
        attestation["kind"] = "wrong-kind"
    elif mutation == "authorizes_training":
        attestation["authorizes_training"] = True
    elif mutation == "primary_identity":
        attestation["inputs"]["primary"]["sha256"] = "0" * 64  # type: ignore[index]
    elif mutation == "phase_statistics":
        attestation["phases"]["primary"]["documents"] = 1_000_000  # type: ignore[index]
    elif mutation == "statistical_gate":
        attestation["gates"]["high_precision_conversion_documents"] = 1  # type: ignore[index]
    elif mutation == "manual_gate":
        attestation["manual_review"]["passed"] = False  # type: ignore[index]
    elif mutation == "scanner_path":
        attestation["scanner"]["path"] = "/fixture/wrong-scanner.py"  # type: ignore[index]
    elif mutation == "scanner_sha256":
        attestation["scanner"]["sha256"] = "0" * 64  # type: ignore[index]
    else:
        attestation["scanner"]["policy"]["manual_unacceptable_samples_eq"] = 1  # type: ignore[index]
    (root / "attestation.json").write_text(
        json.dumps(attestation, sort_keys=True),
        encoding="utf-8",
    )
    _resign_semantic_bundle(root)
    with pytest.raises(closure.ClosureError, match=error):
        closure._validate_chinese_semantic_audit_bundle(
            root,
            phases=phases,
            **SEMANTIC_MINIMUMS,
        )


def test_chinese_semantic_audit_rejects_resigned_sample_payload(
    tmp_path: Path,
) -> None:
    root, phases = _write_semantic_bundle(tmp_path)
    samples_path = root / "samples.jsonl"
    rows = [
        json.loads(line)
        for line in samples_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["excerpt"] = "伪造但哈希自洽的样本文本"
    samples_payload = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    ).encode("utf-8")
    samples_path.write_bytes(samples_payload)
    samples_sha = hashlib.sha256(samples_payload).hexdigest()

    attestation = _json(root / "attestation.json")
    attestation["samples"]["size"] = len(samples_payload)  # type: ignore[index]
    attestation["samples"]["sha256"] = samples_sha  # type: ignore[index]
    manual_path = Path(str(attestation["manual_review"]["path"]))  # type: ignore[index]
    manual = _json(manual_path)
    manual["samples_sha256"] = samples_sha
    manual_path.write_text(
        json.dumps(manual, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    attestation["manual_review"]["size"] = manual_path.stat().st_size  # type: ignore[index]
    attestation["manual_review"]["sha256"] = hashlib.sha256(  # type: ignore[index]
        manual_path.read_bytes()
    ).hexdigest()
    (root / "attestation.json").write_text(
        json.dumps(attestation, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    manifest = _json(root / "MANIFEST.json")
    manifest["files"]["samples.jsonl"] = {  # type: ignore[index]
        "size": len(samples_payload),
        "sha256": samples_sha,
    }
    (root / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    _resign_semantic_bundle(root)

    with pytest.raises(closure.ClosureError, match="deterministic recomputation"):
        closure._validate_chinese_semantic_audit_bundle(
            root,
            phases=phases,
            **SEMANTIC_MINIMUMS,
        )


def test_chinese_semantic_audit_parses_real_manual_verdicts(
    tmp_path: Path,
) -> None:
    root, phases = _write_semantic_bundle(tmp_path)
    attestation = _json(root / "attestation.json")
    manual_path = Path(str(attestation["manual_review"]["path"]))  # type: ignore[index]
    manual = _json(manual_path)
    manual["decisions"][0]["verdict"] = "other_quality_noise"  # type: ignore[index]
    manual_path.write_text(
        json.dumps(manual, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    attestation["manual_review"]["size"] = manual_path.stat().st_size  # type: ignore[index]
    attestation["manual_review"]["sha256"] = hashlib.sha256(  # type: ignore[index]
        manual_path.read_bytes()
    ).hexdigest()
    (root / "attestation.json").write_text(
        json.dumps(attestation, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    _resign_semantic_bundle(root)

    with pytest.raises(
        closure.ClosureError,
        match="manual review differs from recomputed samples",
    ):
        closure._validate_chinese_semantic_audit_bundle(
            root,
            phases=phases,
            **SEMANTIC_MINIMUMS,
        )


def test_chinese_semantic_audit_rejects_payload_changed_after_manifest(
    tmp_path: Path,
) -> None:
    root, phases = _write_semantic_bundle(tmp_path)
    with (root / "samples.jsonl").open("ab") as handle:
        handle.write(b'{"sample_id":"tampered"}\n')
    with pytest.raises(closure.ClosureError, match="payload identity differs"):
        closure._validate_chinese_semantic_audit_bundle(
            root,
            phases=phases,
            **SEMANTIC_MINIMUMS,
        )


def test_formal_baseline_bundle_is_fully_hashed_and_rebuilt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = _fake_snapshot()
    phases = snapshot["phases"]
    root = tmp_path / "baseline"
    (root / "charts").mkdir(parents=True)
    summary = {
        "schema_version": 1,
        "kind": closure.FORMAL_BASELINE_KIND,
        "gate": {
            "passed": True,
            "authorizes_training": False,
            "training_started_by_summarizer": False,
        },
        "phases": [
            {
                "phase": phase,
                "prepared": {
                    "path": phases[phase]["validation_prepared"]["manifest_path"],
                    "sha256": phases[phase]["validation_prepared"]["manifest_sha256"],
                    "dataset_fingerprint": phases[phase]["validation_prepared"][
                        "dataset_fingerprint"
                    ],
                    "token_count": phases[phase]["validation_prepared"]["available_unique_tokens"],
                    "sequence_count": phases[phase]["validation_prepared"][
                        "available_unique_samples"
                    ],
                },
                "evaluation": {
                    "root": f"/fixture/evaluation/{phase}",
                },
            }
            for phase in ("primary", "cooldown")
        ],
        "legacy_six_source_baseline": {
            "identity": {
                "summary": _fake_identity("/fixture/legacy/summary.json", "1"),
            },
            "checkpoint_artifact": {
                "complete_sha256": "2" * 64,
            },
        },
    }
    (root / "summary.json").write_text(
        json.dumps(summary, sort_keys=True),
        encoding="utf-8",
    )
    (root / "REPORT.zh-CN.md").write_text("report\n", encoding="utf-8")
    (root / "charts/formal-source-nll.svg").write_text("<svg/>\n", encoding="utf-8")
    (root / "charts/formal-source-tokens.svg").write_text("<svg/>\n", encoding="utf-8")
    files = {
        relative: _bundle_identity(root, relative)
        for relative in (
            "summary.json",
            "REPORT.zh-CN.md",
            "charts/formal-source-nll.svg",
            "charts/formal-source-tokens.svg",
        )
    }
    manifest = {
        "schema_version": 1,
        "kind": closure.FORMAL_BASELINE_BUNDLE_KIND,
        "gate": summary["gate"],
        "files": files,
    }
    (root / "MANIFEST.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    (root / "COMPLETE").write_text(
        hashlib.sha256((root / "MANIFEST.json").read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )
    monkeypatch.setattr(
        closure,
        "_load_script",
        lambda *_args: SimpleNamespace(build_summary=lambda **_kwargs: summary),
    )
    value = closure._validate_formal_baseline_bundle(
        root,
        phases=phases,
        validation_disjointness_path=tmp_path / "disjointness.json",
    )
    assert value["checkpoint_complete_sha256"] == "2" * 64

    (root / "charts/formal-source-nll.svg").write_text("<svg>tampered</svg>\n", encoding="utf-8")
    with pytest.raises(closure.ClosureError, match="payload identity differs"):
        closure._validate_formal_baseline_bundle(
            root,
            phases=phases,
            validation_disjointness_path=tmp_path / "disjointness.json",
        )


def test_pending_template_fixture_is_controller_ready_and_fail_closed(
    tmp_path: Path,
) -> None:
    capacity_fixture = tmp_path / "capacity.json"
    readiness_fixture = tmp_path / "readiness.json"
    capacity_fixture.write_text(
        json.dumps(_json(CAPACITY_TEMPLATE)),
        encoding="utf-8",
    )
    readiness_value = _pending_readiness_fixture()
    readiness_value["required_capacity_attestation"] = str(capacity_fixture)
    readiness_fixture.write_text(
        json.dumps(readiness_value),
        encoding="utf-8",
    )
    capacity, readiness, _config = closure._require_template_policy(
        capacity_fixture,
        readiness_fixture,
    )
    assert capacity["launch_enabled"] is False
    assert (
        capacity["phase_disjointness"]["stable_id_exact"]["algorithm"]
        == "source-scoped-authenticated-stable-id-intersection-v1"
    )
    assert readiness["launch_enabled"] is False
    assert readiness["chinese_semantic_quality_gate"]["passed"] is False
    assert readiness["pause_evaluation_policy"]["controller_implemented"] is True
    assert readiness["launch_command_after_all_gates_pass"] is None


@pytest.mark.parametrize(
    "field",
    tuple(closure.FORMAL_V4_PENDING_CONFIG_IDENTITIES),
)
def test_template_policy_requires_every_exact_pending_config_identity(
    tmp_path: Path,
    field: str,
) -> None:
    config = yaml.safe_load(
        (ROOT / "configs/base/dense-v4-250m-pilot.blocked.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["data"][field] = f"FORGED_{field}"
    config_path = tmp_path / "blocked.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    capacity = _json(CAPACITY_TEMPLATE)
    readiness = _pending_readiness_fixture()
    capacity_path = tmp_path / "capacity.json"
    readiness_path = tmp_path / "readiness.json"
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    capacity["config"] = {
        "contains_pending_identity_sentinels": True,
        "path": str(config_path),
        "sha256": config_sha,
    }
    readiness["config_path"] = str(config_path)
    readiness["config_sha256"] = config_sha
    readiness["required_capacity_attestation"] = str(capacity_path)
    capacity_path.write_text(json.dumps(capacity), encoding="utf-8")
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")

    with pytest.raises(
        closure.ClosureError,
        match="dynamic identity differs from the exact PENDING contract",
    ):
        closure._require_template_policy(capacity_path, readiness_path)


@pytest.mark.parametrize(
    "mutation",
    (
        "self_consistent_contract",
        "contract_fingerprint",
        "required_acknowledgement",
        "observed_acknowledgement",
        "passed",
        "authorizes_training",
    ),
)
def test_template_policy_requires_exact_pending_wikipedia_license_gate(
    tmp_path: Path,
    mutation: str,
) -> None:
    capacity = _json(CAPACITY_TEMPLATE)
    readiness = _pending_readiness_fixture()
    capacity_path = tmp_path / "capacity.json"
    readiness_path = tmp_path / "readiness.json"
    readiness["required_capacity_attestation"] = str(capacity_path)
    gate = readiness["wikipedia_license_gate"]
    if mutation == "self_consistent_contract":
        gate["contract"]["revision"] = "0" * 40  # type: ignore[index]
        fingerprint = closure._canonical_sha256(gate["contract"])  # type: ignore[arg-type]
        gate["contract_fingerprint"] = fingerprint  # type: ignore[index]
        gate["required_acknowledgement"] = (  # type: ignore[index]
            f"ACCEPT V4 WIKIPEDIA LICENSE {fingerprint}"
        )
    elif mutation == "contract_fingerprint":
        gate["contract_fingerprint"] = "0" * 64  # type: ignore[index]
    elif mutation == "required_acknowledgement":
        gate["required_acknowledgement"] = "ACCEPT SOMETHING ELSE"  # type: ignore[index]
    elif mutation == "observed_acknowledgement":
        gate["observed_acknowledgement"] = gate["required_acknowledgement"]  # type: ignore[index]
    elif mutation == "passed":
        gate["passed"] = True  # type: ignore[index]
    else:
        gate["authorizes_training"] = True  # type: ignore[index]
    capacity_path.write_text(json.dumps(capacity), encoding="utf-8")
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")

    with pytest.raises(
        closure.ClosureError,
        match="Wikipedia license gate is not the exact pending contract",
    ):
        closure._require_template_policy(capacity_path, readiness_path)


@pytest.mark.parametrize(
    "mutation",
    (
        "recipe_id",
        "recipe_sha256",
        "resolved_source_lock_sha256",
        "self_consistent_recipe_sha256",
        "self_consistent_resolved_sha256",
    ),
)
def test_phase_extracted_identity_must_match_capacity_recipe_and_resolved_lock(
    mutation: str,
) -> None:
    capacity = _json(CAPACITY_TEMPLATE)
    stage = copy.deepcopy(capacity["stages"]["primary"])  # type: ignore[index]
    extracted = {
        "recipe_id": stage["recipe"]["recipe_id"],
        "recipe_sha256": stage["recipe"]["sha256"],
        "resolved_source_lock_sha256": stage["resolved_lock"]["sha256"],
    }
    if mutation in extracted:
        extracted[mutation] = "0" * (40 if mutation == "recipe_id" else 64)
    elif mutation == "self_consistent_recipe_sha256":
        extracted["recipe_sha256"] = "0" * 64
        stage["recipe"]["sha256"] = "0" * 64
    else:
        extracted["resolved_source_lock_sha256"] = "0" * 64
        stage["resolved_lock"]["sha256"] = "0" * 64
    with pytest.raises(closure.ClosureError):
        closure._require_phase_recipe_contract(
            extracted,
            phase="primary",
            capacity_stage=stage,
        )


def test_phase_extracted_identity_accepts_current_authenticated_contracts() -> None:
    capacity = _json(CAPACITY_TEMPLATE)
    exception_fingerprints = {
        "primary": "7792dc6480e5bd4eff4ceba211a1c4ea7ead5a2f0f08fce06c2a98dd937a8d03",
        "cooldown": "2fd0eee045ba70c4e89fa09a1f8cf7b8972180c6e78993ed831154e92dddd8b9",
    }
    for phase in ("primary", "cooldown"):
        stage = capacity["stages"][phase]  # type: ignore[index]
        extracted = {
            "recipe_id": stage["recipe"]["recipe_id"],
            "recipe_sha256": stage["recipe"]["sha256"],
            "resolved_source_lock_sha256": stage["resolved_lock"]["sha256"],
        }
        recipe = closure._require_phase_recipe_contract(
            extracted,
            phase=phase,
            capacity_stage=stage,
        )
        assert recipe["recipe_id"] == stage["recipe"]["recipe_id"]
        assert (
            closure._require_native_schema_v2_wikipedia_exception(
                recipe,
                phase=phase,
            )
            == exception_fingerprints[phase]
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "wrong_source",
        "wrong_schema",
        "wrong_exception",
        "wrong_attribution",
        "policy_authorizes_training",
        "source_authorizes_training",
        "duplicate_exception_source",
    ),
)
def test_phase_recipe_contract_rejects_self_consistent_wikipedia_contract_change(
    tmp_path: Path,
    mutation: str,
) -> None:
    capacity = _json(CAPACITY_TEMPLATE)
    stage = copy.deepcopy(capacity["stages"]["primary"])  # type: ignore[index]
    recipe = _json(ROOT / str(stage["recipe"]["path"]))
    wikipedia = next(
        source
        for source in recipe["sources"]  # type: ignore[union-attr]
        if source["source_id"] == closure.CHINESE_SEMANTIC_SOURCE_ID
    )
    policy_exception = recipe["license_policy"][  # type: ignore[index]
        "source_specific_share_alike_exceptions"
    ][0]
    if mutation == "wrong_source":
        wikipedia["source_id"] = "self_consistent_wrong_source"
        policy_exception["source_id"] = "self_consistent_wrong_source"
        recipe["attribution_contract"]["source_id"] = (  # type: ignore[index]
            "self_consistent_wrong_source"
        )
    elif mutation == "wrong_schema":
        recipe["schema_version"] = 1
        recipe["kind"] = "twen_base_data_source_recipe"
    elif mutation == "wrong_exception":
        wikipedia["share_alike_exception"]["exception_id"] = (  # type: ignore[index]
            "self-consistent-wrong-exception"
        )
        policy_exception["exception_id"] = "self-consistent-wrong-exception"
    elif mutation == "wrong_attribution":
        recipe["attribution_contract"]["fields"] = ["id", "url"]  # type: ignore[index]
    elif mutation == "policy_authorizes_training":
        policy_exception["authorizes_training"] = True
    elif mutation == "source_authorizes_training":
        wikipedia["share_alike_exception"]["authorizes_training"] = True  # type: ignore[index]
    else:
        duplicate = copy.deepcopy(wikipedia)
        duplicate["source_id"] = "duplicate_exception_source"
        recipe["sources"].append(duplicate)  # type: ignore[union-attr]
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
    recipe_sha = hashlib.sha256(recipe_path.read_bytes()).hexdigest()

    resolved = _json(ROOT / str(stage["resolved_lock"]["path"]))
    resolved["recipe_sha256"] = recipe_sha
    resolved_path = tmp_path / "resolved.json"
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
    resolved_sha = hashlib.sha256(resolved_path.read_bytes()).hexdigest()

    stage["recipe"]["path"] = str(recipe_path)
    stage["recipe"]["sha256"] = recipe_sha
    stage["resolved_lock"]["path"] = str(resolved_path)
    stage["resolved_lock"]["sha256"] = resolved_sha
    extracted = {
        "recipe_id": stage["recipe"]["recipe_id"],
        "recipe_sha256": recipe_sha,
        "resolved_source_lock_sha256": resolved_sha,
    }
    with pytest.raises(
        closure.ClosureError,
        match="Wikipedia recipe/license/attribution contract differs",
    ):
        closure._require_phase_recipe_contract(
            extracted,
            phase="primary",
            capacity_stage=stage,
        )


def _write_wikipedia_attribution_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "corpus"
    path = root / "filtered/attribution/attribution.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {
            "source_id": closure.CHINESE_SEMANTIC_SOURCE_ID,
            "repo_id": closure.WIKIPEDIA_REPO_ID,
            "revision": closure.WIKIPEDIA_REVISION,
            "source_license": closure.WIKIPEDIA_LICENSE,
            "id": str(index),
            "url": f"https://zh.wikipedia.org/wiki/{index}",
            "title": f"title-{index}",
        }
        for index in range(2)
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    extracted: dict[str, object] = {
        "license_audit": {
            "complete": True,
            "sources": [
                {
                    "source_id": closure.CHINESE_SEMANTIC_SOURCE_ID,
                    "declaration": closure.WIKIPEDIA_LICENSE,
                    "scope": closure.WIKIPEDIA_LICENSE_SCOPE,
                    "field": None,
                    "allowlist": [],
                }
            ],
        },
        "sources": [
            {
                "source_id": closure.CHINESE_SEMANTIC_SOURCE_ID,
                "train_rows": 1,
                "validation_rows": 1,
            }
        ],
        "attribution_files": [
            {
                "path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ],
    }
    return root / "corpus-manifest.json", extracted


def test_wikipedia_attribution_inventory_covers_every_retained_row(
    tmp_path: Path,
) -> None:
    manifest_path, extracted = _write_wikipedia_attribution_fixture(tmp_path)
    evidence = closure._wikipedia_attribution_evidence(manifest_path, extracted)
    assert evidence["authenticated_rows"] == 2
    assert evidence["expected_rows"] == 2
    assert evidence["required_fields"] == ["id", "url", "title"]
    assert evidence["passed"] is True
    assert evidence["authorizes_training"] is False


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("missing_title", "violates the immutable contract"),
        ("wrong_revision", "violates the immutable contract"),
        ("missing_row", "coverage differs"),
        ("wrong_license_audit", "license audit differs"),
    ),
)
def test_wikipedia_attribution_inventory_rejects_contract_or_coverage_drift(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    manifest_path, extracted = _write_wikipedia_attribution_fixture(tmp_path)
    attribution_path = (
        manifest_path.parent / extracted["attribution_files"][0]["path"]  # type: ignore[index]
    )
    if mutation in {"missing_title", "wrong_revision"}:
        rows = [
            json.loads(line)
            for line in attribution_path.read_text(encoding="utf-8").splitlines()
        ]
        if mutation == "missing_title":
            rows[0].pop("title")
        else:
            rows[0]["revision"] = "0" * 40
        attribution_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    elif mutation == "missing_row":
        extracted["sources"][0]["train_rows"] = 2  # type: ignore[index]
    else:
        extracted["license_audit"]["sources"][0]["declaration"] = "CC0"  # type: ignore[index]
    with pytest.raises(closure.ClosureError, match=error):
        closure._wikipedia_attribution_evidence(manifest_path, extracted)


def test_planning_generator_uses_source_scoped_stable_id_contract() -> None:
    path = ROOT / "scripts/prepare_v4_250m_data_plan.py"
    spec = importlib.util.spec_from_file_location("prepare_v4_250m_data_plan", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generated = module._make_capacity_attestation()
    assert (
        generated["phase_disjointness"]["stable_id_exact"]["algorithm"]
        == "source-scoped-authenticated-stable-id-intersection-v1"
    )


@pytest.mark.parametrize(
    ("target", "field", "value"),
    (
        ("capacity", "launch_enabled", True),
        ("capacity", "authorizes_training", True),
        ("readiness", "launch_enabled", True),
        ("readiness", "launch_command_after_all_gates_pass", "train now"),
    ),
)
def test_template_policy_rejects_any_launch_enablement(
    tmp_path: Path,
    target: str,
    field: str,
    value: object,
) -> None:
    capacity = _json(CAPACITY_TEMPLATE)
    readiness = _pending_readiness_fixture()
    capacity_path = tmp_path / "capacity.json"
    readiness_path = tmp_path / "readiness.json"
    readiness["required_capacity_attestation"] = str(capacity_path)
    if target == "capacity":
        capacity[field] = value
    else:
        readiness[field] = value
    capacity_path.write_text(json.dumps(capacity), encoding="utf-8")
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
    with pytest.raises(closure.ClosureError):
        closure._require_template_policy(capacity_path, readiness_path)
