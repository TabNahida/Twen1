from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "summarize_v4_formal_validation.py"
    spec = importlib.util.spec_from_file_location(
        "summarize_v4_formal_validation",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


formal = _load_script()


def _lineage(phase: str) -> dict[str, object]:
    validation = [
        {
            "source_id": source,
            "path": f"filtered/{source}/chunk-000000/validation.jsonl",
            "sha256": f"{index + 1:064x}",
        }
        for index, source in enumerate(sorted(formal.EXPECTED_PHASE_SOURCES[phase]))
    ]
    return {
        "kind": "authenticated_extracted_corpus",
        "role": "validation",
        "ready_for_training": True,
        "research_only": False,
        "audit_attestation": {
            "bound_as": "frozen_validation",
            "ready_for_training": True,
            "gates": {"all": {"passed": True}},
        },
        "data_contract": {
            "source_map": {
                "roles": {
                    "validation": validation,
                }
            }
        },
    }


def _prepared_payload(phase: str) -> dict[str, object]:
    lineage = _lineage(phase)
    validation = lineage["data_contract"]["source_map"]["roles"]["validation"]  # type: ignore[index]
    return {
        "lineage": lineage,
        "shards": [
            {
                "shard_id": f"shard-{index:06d}",
                "source_path": f"/fixture/{row['path']}",
                "source_sha256": row["sha256"],
            }
            for index, row in enumerate(validation)
        ],
    }


def test_governed_lineage_requires_all_phase_sources() -> None:
    payload = _prepared_payload("cooldown")
    assert (
        formal._governed_lineage_sources(payload, phase="cooldown")
        == (formal.EXPECTED_PHASE_SOURCES["cooldown"])
    )
    payload["lineage"]["data_contract"]["source_map"]["roles"]["validation"].pop()  # type: ignore[index]
    with pytest.raises(formal.FormalValidationError, match="source coverage differs"):
        formal._governed_lineage_sources(payload, phase="cooldown")


def test_governed_lineage_rejects_research_only() -> None:
    payload = _prepared_payload("primary")
    payload["lineage"]["ready_for_training"] = False  # type: ignore[index]
    payload["lineage"]["research_only"] = True  # type: ignore[index]
    with pytest.raises(formal.FormalValidationError, match="not a governed"):
        formal._governed_lineage_sources(payload, phase="primary")


def test_combine_sources_uses_predicted_token_weighting() -> None:
    phases = [
        {
            "phase": "primary",
            "sources": [
                {
                    "source": "science_arxiv_open_permissive",
                    "shards": 1,
                    "sequences": 1,
                    "input_tokens": 11,
                    "predicted_tokens": 10,
                    "nll_sum": 20.0,
                }
            ],
        },
        {
            "phase": "cooldown",
            "sources": [
                {
                    "source": "science_arxiv_open_permissive",
                    "shards": 1,
                    "sequences": 1,
                    "input_tokens": 31,
                    "predicted_tokens": 30,
                    "nll_sum": 90.0,
                },
                {
                    "source": "public_domain_project_gutenberg",
                    "shards": 1,
                    "sequences": 1,
                    "input_tokens": 21,
                    "predicted_tokens": 20,
                    "nll_sum": 50.0,
                },
            ],
        },
    ]
    rows = {row["source"]: row for row in formal._combine_sources(phases)}
    arxiv = rows["science_arxiv_open_permissive"]
    assert arxiv["predicted_tokens"] == 40
    assert arxiv["nll_sum"] == 110.0
    assert arxiv["mean_nll"] == pytest.approx(2.75)
    assert arxiv["phases"] == ["primary", "cooldown"]
    assert arxiv["source_group"] == "additional_v4"


def test_disjointness_attestation_must_bind_both_prepared_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attestation_path = tmp_path / "attestation.json"
    attestation_path.write_text("{}\n", encoding="utf-8")
    prepared: dict[str, dict[str, object]] = {}
    phase_identities: dict[str, dict[str, object]] = {}
    for index, phase in enumerate(("primary", "cooldown"), start=1):
        manifest_path = tmp_path / f"{phase}-prepared.json"
        manifest_path.write_text("{}\n", encoding="utf-8")
        extracted_path = tmp_path / f"{phase}-corpus-manifest.json"
        audit_path = tmp_path / f"{phase}-audit.json"
        prepared[phase] = {
            "path": manifest_path,
            "sha256": f"{index:064x}",
            "dataset_fingerprint": f"{index + 10:064x}",
            "sequence_count": index * 10,
            "token_count": index * 100,
            "lineage_identity": {
                "extracted_manifest_path": str(extracted_path),
                "extracted_manifest_sha256": f"{index + 20:064x}",
                "corpus_fingerprint": f"{index + 30:064x}",
                "audit_attestation": {
                    "path": str(audit_path),
                    "sha256": f"{index + 40:064x}",
                },
            },
        }
        phase_identities[phase] = {
            "manifest_path": str(extracted_path),
            "manifest_sha256": f"{index + 20:064x}",
            "audit_attestation_path": str(audit_path),
            "audit_attestation_sha256": f"{index + 40:064x}",
            "validation_prepared": {
                "manifest_path": str(manifest_path),
                "manifest_sha256": f"{index:064x}",
                "dataset_fingerprint": f"{index + 10:064x}",
                "sequence_count": index * 10,
                "token_count": index * 100,
            },
        }
    value = {
        "passed": True,
        "near_duplicate_threshold": 0.8,
        "phases": phase_identities,
        "gates": {"all_disjoint": {"passed": True}},
        "identity_reverification": {"passed": True},
        "attestation_fingerprint": "a" * 64,
        "scanner_source_sha256": "b" * 64,
        "phase_attestation_validator_source_sha256": "c" * 64,
        "scanner_source_tree_sha256": "d" * 64,
    }
    module = SimpleNamespace(
        REQUIRED_NEAR_DUPLICATE_THRESHOLD=0.8,
        validate_formal_validation_disjointness_attestation=lambda _path: value,
    )
    monkeypatch.setattr(
        formal,
        "_load_validation_disjointness_module",
        lambda: module,
    )
    sweep = SimpleNamespace(_sha256=lambda _path: "e" * 64)
    result = formal._load_validation_disjointness(
        attestation_path,
        prepared=prepared,
        sweep=sweep,
    )
    assert result["passed"] is True
    assert result["near_duplicate_threshold"] == 0.8

    phase_identities["cooldown"]["validation_prepared"]["manifest_sha256"] = (  # type: ignore[index]
        "f" * 64
    )
    with pytest.raises(formal.FormalValidationError, match="cooldown prepared"):
        formal._load_validation_disjointness(
            attestation_path,
            prepared=prepared,
            sweep=sweep,
        )

    phase_identities["cooldown"]["validation_prepared"]["manifest_sha256"] = (  # type: ignore[index]
        "2".zfill(64)
    )

    def replace_during_validation(_path: Path) -> dict[str, object]:
        attestation_path.write_text('{"changed":true}\n', encoding="utf-8")
        return value

    module.validate_formal_validation_disjointness_attestation = replace_during_validation
    with pytest.raises(formal.FormalValidationError, match="changed during authentication"):
        formal._load_validation_disjointness(
            attestation_path,
            prepared=prepared,
            sweep=sweep,
        )


def test_v3_checkpoint_comparison_binds_checkpoint_complete_hash(
    tmp_path: Path,
) -> None:
    expected_checkpoint = tmp_path / "expected-checkpoint"
    other_checkpoint = tmp_path / "other-checkpoint"
    expected_checkpoint.mkdir()
    other_checkpoint.mkdir()
    (expected_checkpoint / "COMPLETE").write_text("expected-manifest\n", encoding="ascii")
    (other_checkpoint / "COMPLETE").write_text("other-manifest\n", encoding="ascii")
    lineage = {
        "archived_config_sha256": "1" * 64,
        "saved_critical_fingerprint": "2" * 64,
        "saved_source_tree_sha256": "3" * 64,
    }
    state = {
        "global_step": 1912,
        "committed_tokens": 500_009_962,
        "kind": "milestone",
        "tag": "complete",
    }
    legacy = {
        "checkpoint_state": state,
        "checkpoint_artifact": {
            "path": str(expected_checkpoint.resolve()),
            "complete_sha256": formal._sha256_file(expected_checkpoint / "COMPLETE"),
        },
        "evaluation_harness": {
            "checkpoint_inference_lineage": lineage,
        },
    }
    evaluation = {
        "run_id": "base-dense-v3-500m",
        "checkpoint_state": state,
        "plan": {
            "checkpoint": str(other_checkpoint),
            "checkpoint_complete_sha256": formal._sha256_file(other_checkpoint / "COMPLETE"),
        },
        "harness": {
            "checkpoint_inference_lineage": lineage,
        },
    }

    with pytest.raises(
        formal.FormalValidationError,
        match="checkpoint COMPLETE differs",
    ):
        formal._assert_v3_checkpoint(
            evaluation,
            legacy=legacy,
            phase="primary",
        )

    relocated_checkpoint = tmp_path / "relocated-checkpoint"
    relocated_checkpoint.mkdir()
    (relocated_checkpoint / "COMPLETE").write_bytes((expected_checkpoint / "COMPLETE").read_bytes())
    evaluation["plan"] = {
        "checkpoint": str(relocated_checkpoint),
        "checkpoint_complete_sha256": formal._sha256_file(relocated_checkpoint / "COMPLETE"),
    }
    formal._assert_v3_checkpoint(
        evaluation,
        legacy=legacy,
        phase="primary",
    )


def test_formal_numeric_contract_requires_identical_phase_evaluation() -> None:
    comparison = {
        "track": "base",
        "stage": "dense-oracle",
        "expert_initialization": "donor",
        "prepared_manifest_sha256": "a" * 64,
        "prepared_dataset_fingerprint": "b" * 64,
        "batch_size": 1,
        "device_type": "cuda",
        "dtype": "bfloat16",
    }
    lineage = {
        "archived_config_sha256": "1" * 64,
        "saved_critical_fingerprint": "2" * 64,
        "saved_source_tree_sha256": "3" * 64,
    }
    phases = [
        {
            "phase": phase,
            "comparison_contract": dict(comparison),
            "evaluation_harness": {
                "config_fingerprint": "4" * 64,
                "runtime": {"torch_version": "2.11.0+cu130"},
                "checkpoint_inference_lineage": lineage,
            },
        }
        for phase in ("primary", "cooldown")
    ]
    legacy = {
        "comparison_contract": {
            **comparison,
            "prepared_manifest_sha256": "5" * 64,
            "prepared_dataset_fingerprint": "6" * 64,
        },
        "evaluation_harness": {
            "config_fingerprint": "7" * 64,
            "runtime": {"torch_version": "2.10.0+cu130"},
        },
    }
    result = formal._formal_numeric_contract(phases, legacy=legacy)
    assert result["formal_phases_identical"] is True
    assert result["legacy_core_contract_identical"] is True
    assert result["legacy_runtime_identical"] is False
    assert result["legacy_current_preflight_fingerprint_identical"] is False

    phases[1]["comparison_contract"]["batch_size"] = 2  # type: ignore[index]
    with pytest.raises(formal.FormalValidationError, match="numeric contracts differ"):
        formal._formal_numeric_contract(phases, legacy=legacy)
