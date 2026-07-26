from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "attest_v4_formal_validation_disjointness.py"
    spec = importlib.util.spec_from_file_location(
        "attest_v4_formal_validation_disjointness",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


attest = _load_script()


def test_role_stable_ids_are_source_scoped_and_role_filtered(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    ledger = root / "extracted" / "source" / "chunk-000000" / "attribution.jsonl"
    ledger.parent.mkdir(parents=True)
    rows = [
        {
            "source_id": "source",
            "stable_id": "1" * 64,
            "split": "train",
            "token_count_with_eos": 11,
        },
        {
            "source_id": "source",
            "stable_id": "2" * 64,
            "split": "validation",
            "token_count_with_eos": 7,
        },
    ]
    ledger.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    corpus = SimpleNamespace(
        manifest_path=root / "corpus-manifest.json",
        value={
            "attribution_files": [
                {
                    "path": ledger.relative_to(root).as_posix(),
                }
            ]
        },
    )
    assert list(attest._iter_role_stable_ids(corpus, role="validation")) == [
        (
            "source",
            "2" * 64,
            "extracted/source/chunk-000000/attribution.jsonl",
            2,
            7,
        )
    ]


def test_reference_index_matches_stable_exact_and_near(tmp_path: Path) -> None:
    index = attest._ReferenceIndex(tmp_path / "index.sqlite3")
    try:
        text = "alpha beta gamma delta epsilon zeta eta theta"
        digest = attest._normalized_text_sha256(text)
        signature = attest._one_permutation_signature(
            attest._lexical_tokens(text),
            digest,
        )
        index.add_stable_id(
            source_id="source",
            stable_id="a" * 64,
            phase="primary",
            role="train",
            path="train.jsonl",
            line_number=1,
        )
        index.add_document(
            normalized_sha=digest,
            source_id="source",
            phase="primary",
            role="train",
            path="train.jsonl",
            line_number=1,
            signature=signature,
        )
        index.commit()
        assert index.match_stable_id(
            source_id="source",
            stable_id="a" * 64,
        ) == ("primary", "train", "train.jsonl", 1)
        assert index.match_exact(digest) == (
            "source",
            "primary",
            "train",
            "train.jsonl",
            1,
        )
        near_text = "alpha beta gamma delta epsilon zeta eta theta iota"
        near_digest = attest._normalized_text_sha256(near_text)
        near_signature = attest._one_permutation_signature(
            attest._lexical_tokens(near_text),
            near_digest,
        )
        match = index.match_near(
            normalized_sha=near_digest,
            signature=near_signature,
            threshold=0.5,
        )
        assert match is not None
        assert match[:5] == (
            "source",
            "primary",
            "train",
            "train.jsonl",
            1,
        )
        assert 0.5 <= match[-1] <= 1.0
    finally:
        index.close()


def test_normalized_exact_identity_folds_whitespace_and_unicode() -> None:
    assert attest._normalized_text_sha256(
        "\N{FULLWIDTH LATIN CAPITAL LETTER A}  B\nC"
    ) == attest._normalized_text_sha256("A B C")


def test_identity_reverification_detects_source_and_input_changes() -> None:
    sources = {
        "formal_validation_scanner_sha256": "1" * 64,
        "phase_attestation_validator_sha256": "2" * 64,
        "twen_source_tree_sha256": "3" * 64,
    }
    phases = {
        "primary": {"manifest_sha256": "4" * 64},
        "cooldown": {"manifest_sha256": "5" * 64},
    }
    phase_attestation = {"sha256": "6" * 64}
    result = attest._assert_end_identity_matches_start(
        scanner_sources_start=sources,
        scanner_sources_end=sources,
        phases_start=phases,
        phases_end=phases,
        phase_attestation_start=phase_attestation,
        phase_attestation_end=phase_attestation,
    )
    assert result["passed"] is True
    assert result["input_identity_start_sha256"] == result["input_identity_end_sha256"]

    changed_sources = {**sources, "twen_source_tree_sha256": "7" * 64}
    with pytest.raises(attest.DataAuditError, match="scanner source identity changed"):
        attest._assert_end_identity_matches_start(
            scanner_sources_start=sources,
            scanner_sources_end=changed_sources,
            phases_start=phases,
            phases_end=phases,
            phase_attestation_start=phase_attestation,
            phase_attestation_end=phase_attestation,
        )

    changed_phases = {**phases, "cooldown": {"manifest_sha256": "8" * 64}}
    with pytest.raises(attest.DataAuditError, match="input identity changed"):
        attest._assert_end_identity_matches_start(
            scanner_sources_start=sources,
            scanner_sources_end=sources,
            phases_start=phases,
            phases_end=changed_phases,
            phase_attestation_start=phase_attestation,
            phase_attestation_end=phase_attestation,
        )


def test_union_scanner_detects_cross_phase_train_validation_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpora = {phase: SimpleNamespace(phase=phase) for phase in attest.PHASES}
    identities = {
        phase: {
            "phase": phase,
            "manifest_path": f"/{phase}/corpus-manifest.json",
            "manifest_sha256": ("1" if phase == "primary" else "2") * 64,
            "audit_attestation_path": f"/{phase}/audit/attestation.json",
            "audit_attestation_sha256": ("3" if phase == "primary" else "4") * 64,
            "prepared": {"manifest_path": f"/{phase}/train/manifest.json"},
            "validation_prepared": {"manifest_path": f"/{phase}/validation/manifest.json"},
        }
        for phase in attest.PHASES
    }
    stable_rows = {
        ("primary", "train"): [("shared", "a" * 64, "primary/train.jsonl", 1, 8)],
        ("cooldown", "train"): [("shared", "b" * 64, "cooldown/train.jsonl", 1, 8)],
        ("primary", "validation"): [
            ("shared", "b" * 64, "primary/validation.jsonl", 1, 8),
            ("shared", "c" * 64, "primary/validation.jsonl", 2, 8),
        ],
        ("cooldown", "validation"): [
            ("shared", "a" * 64, "cooldown/validation.jsonl", 1, 8),
            ("shared", "c" * 64, "cooldown/validation.jsonl", 2, 8),
        ],
    }
    primary_near = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
        "nu xi omicron pi rho sigma tau upsilon"
    )
    cooldown_near = (
        "one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
    )
    documents = {
        ("primary", "train"): [
            ("primary/train.jsonl", "shared", "text", 1, "primary exact payload"),
            ("primary/train.jsonl", "shared", "text", 2, primary_near),
        ],
        ("cooldown", "train"): [
            ("cooldown/train.jsonl", "shared", "text", 1, "cooldown exact payload"),
            ("cooldown/train.jsonl", "shared", "text", 2, cooldown_near),
        ],
        ("primary", "validation"): [
            ("primary/validation.jsonl", "shared", "text", 1, "cooldown exact payload"),
            (
                "primary/validation.jsonl",
                "shared",
                "text",
                2,
                cooldown_near + " twentyone",
            ),
            ("primary/validation.jsonl", "shared", "text", 3, "validation union duplicate"),
        ],
        ("cooldown", "validation"): [
            ("cooldown/validation.jsonl", "shared", "text", 1, "primary exact payload"),
            (
                "cooldown/validation.jsonl",
                "shared",
                "text",
                2,
                primary_near + " phi",
            ),
            ("cooldown/validation.jsonl", "shared", "text", 3, "validation union duplicate"),
        ],
    }

    def fake_attested_phase(**kwargs: object) -> tuple[object, dict[str, object]]:
        phase = str(kwargs["phase"])
        return corpora[phase], identities[phase]

    monkeypatch.setattr(attest, "_load_phase_module", lambda: SimpleNamespace())
    monkeypatch.setattr(attest, "_attested_phase", fake_attested_phase)
    monkeypatch.setattr(
        attest,
        "_validate_phase_attestation",
        lambda *_args, **_kwargs: {
            "path": "/phase/attestation.json",
            "sha256": "5" * 64,
            "attestation_fingerprint": "6" * 64,
            "gates": {},
        },
    )
    scanner_sources = {
        "formal_validation_scanner_sha256": "7" * 64,
        "phase_attestation_validator_sha256": "8" * 64,
        "twen_source_tree_sha256": "9" * 64,
    }
    monkeypatch.setattr(attest, "_scanner_sources_identity", lambda: scanner_sources)
    monkeypatch.setattr(
        attest,
        "_iter_role_stable_ids",
        lambda corpus, *, role: iter(stable_rows[(corpus.phase, role)]),
    )
    monkeypatch.setattr(
        attest,
        "_iter_jsonl_documents",
        lambda corpus, role: iter(documents[(corpus.phase, role)]),
    )

    output = tmp_path / "formal-union"
    path = attest.build_formal_validation_disjointness_attestation(
        primary_manifest=tmp_path / "primary-manifest.json",
        primary_audit=tmp_path / "primary-audit.json",
        primary_train_prepared=tmp_path / "primary-train.json",
        primary_validation_prepared=tmp_path / "primary-validation.json",
        cooldown_manifest=tmp_path / "cooldown-manifest.json",
        cooldown_audit=tmp_path / "cooldown-audit.json",
        cooldown_train_prepared=tmp_path / "cooldown-train.json",
        cooldown_validation_prepared=tmp_path / "cooldown-validation.json",
        phase_disjointness_attestation=tmp_path / "phase-attestation.json",
        output_root=output,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["metrics"]["train_validation_stable_id_matches"] == 2
    assert payload["metrics"]["train_validation_normalized_exact_matches"] == 2
    assert payload["metrics"]["train_validation_near_duplicate_matches"] >= 2
    assert payload["metrics"]["validation_internal_stable_id_matches"] == 1
    assert payload["metrics"]["validation_internal_normalized_exact_matches"] == 1


def test_phase_attestation_validator_rejects_stale_source_and_weakened_gates(
    tmp_path: Path,
) -> None:
    phase_module = attest._load_phase_module()

    def identity(phase: str, marker: str) -> dict[str, object]:
        return {
            "manifest_path": f"/{phase}/corpus-manifest.json",
            "manifest_sha256": marker * 64,
            "corpus_fingerprint": (marker.upper() if marker.isalpha() else "f") * 64,
            "audit_attestation_path": f"/{phase}/audit/attestation.json",
            "audit_attestation_sha256": ("a" if phase == "primary" else "b") * 64,
            "audit_attestation_fingerprint": ("c" if phase == "primary" else "d") * 64,
            "prepared": {
                "manifest_path": f"/{phase}/train/manifest.json",
                "manifest_sha256": ("e" if phase == "primary" else "f") * 64,
                "dataset_fingerprint": ("1" if phase == "primary" else "2") * 64,
                "source_map_sha256": ("3" if phase == "primary" else "4") * 64,
            },
            "phase": phase,
            "validation_prepared": {
                "manifest_path": f"/{phase}/validation/manifest.json",
                "manifest_sha256": ("5" if phase == "primary" else "6") * 64,
                "dataset_fingerprint": ("7" if phase == "primary" else "8") * 64,
                "sequence_count": 1,
                "token_count": 4,
            },
        }

    identities = {
        "primary": identity("primary", "9"),
        "cooldown": identity("cooldown", "0"),
    }
    payload: dict[str, object] = {
        "schema_version": phase_module.SCHEMA_VERSION,
        "kind": phase_module.KIND,
        "scanner_source_sha256": attest.sha256_file(Path(phase_module.__file__)),
        "scanner_source_tree_sha256": attest.twen_source_tree_sha256(),
        "scope": "authenticated_train_inventories_only",
        "metrics": {
            "stable_id_exact_matches": 0,
            "normalized_text_exact_matches": 0,
            "near_duplicate_matches": 0,
        },
        "gates": {
            "stable_id_exact": {
                "algorithm": phase_module.STABLE_ID_ALGORITHM,
                "matches": 0,
                "passed": True,
            },
            "normalized_text_exact": {
                "algorithm": phase_module.NORMALIZED_EXACT_ALGORITHM,
                "matches": 0,
                "passed": True,
            },
            "near_duplicate": {
                "algorithm": phase_module.NEAR_DUPLICATE_ALGORITHM,
                "estimated_jaccard_threshold": phase_module.REQUIRED_NEAR_DUPLICATE_THRESHOLD,
                "matches": 0,
                "passed": True,
            },
        },
        "stores_raw_text": False,
        "passed": True,
        **{
            phase: {
                key: value
                for key, value in phase_identity.items()
                if key not in {"phase", "validation_prepared"}
            }
            for phase, phase_identity in identities.items()
        },
    }
    path = tmp_path / "phase" / "attestation.json"
    path.parent.mkdir()

    def publish() -> None:
        payload.pop("attestation_fingerprint", None)
        payload["attestation_fingerprint"] = attest._canonical_sha256(payload)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        complete = {
            "schema_version": phase_module.SCHEMA_VERSION,
            "kind": "twen_v4_phase_disjointness_complete",
            "attestation": path.name,
            "attestation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "attestation_fingerprint": payload["attestation_fingerprint"],
            "passed": True,
        }
        (path.parent / "COMPLETE").write_text(
            json.dumps(complete, sort_keys=True),
            encoding="utf-8",
        )

    publish()
    attest._validate_phase_attestation(
        path,
        identities=identities,
        phase_module=phase_module,
    )

    payload["scanner_source_tree_sha256"] = "f" * 64
    publish()
    with pytest.raises(attest.DataAuditError, match="scanner identity/scope is stale"):
        attest._validate_phase_attestation(
            path,
            identities=identities,
            phase_module=phase_module,
        )

    payload["scanner_source_tree_sha256"] = attest.twen_source_tree_sha256()
    payload["gates"] = {"arbitrary": {"passed": True, "matches": 0}}
    publish()
    with pytest.raises(attest.DataAuditError, match="gate/metrics contract differs"):
        attest._validate_phase_attestation(
            path,
            identities=identities,
            phase_module=phase_module,
        )


def test_formal_attestation_validator_requires_exact_zero_match_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scanner_sources = attest._scanner_sources_identity()
    phases = {
        phase: {
            "manifest_path": f"/{phase}/corpus-manifest.json",
            "audit_attestation_path": f"/{phase}/audit/attestation.json",
            "prepared": {"manifest_path": f"/{phase}/train/manifest.json"},
            "validation_prepared": {"manifest_path": f"/{phase}/validation/manifest.json"},
        }
        for phase in attest.PHASES
    }
    phase_identity = {"path": str(tmp_path / "phase-attestation.json"), "passed": True}
    reverification = attest._assert_end_identity_matches_start(
        scanner_sources_start=scanner_sources,
        scanner_sources_end=scanner_sources,
        phases_start=phases,
        phases_end=phases,
        phase_attestation_start=phase_identity,
        phase_attestation_end=phase_identity,
    )
    gate_contract = {
        "train_validation_stable_id": (
            attest.STABLE_ID_ALGORITHM,
            "train_validation_stable_id_matches",
        ),
        "train_validation_normalized_exact": (
            attest.NORMALIZED_EXACT_ALGORITHM,
            "train_validation_normalized_exact_matches",
        ),
        "train_validation_near_duplicate": (
            attest.NEAR_DUPLICATE_ALGORITHM,
            "train_validation_near_duplicate_matches",
        ),
        "validation_internal_stable_id": (
            attest.STABLE_ID_ALGORITHM,
            "validation_internal_stable_id_matches",
        ),
        "validation_internal_normalized_exact": (
            attest.NORMALIZED_EXACT_ALGORITHM,
            "validation_internal_normalized_exact_matches",
        ),
        "validation_internal_near_duplicate": (
            attest.NEAR_DUPLICATE_ALGORITHM,
            "validation_internal_near_duplicate_matches",
        ),
    }
    payload: dict[str, object] = {
        "schema_version": attest.SCHEMA_VERSION,
        "kind": attest.KIND,
        "scanner_source_sha256": scanner_sources["formal_validation_scanner_sha256"],
        "phase_attestation_validator_source_sha256": scanner_sources[
            "phase_attestation_validator_sha256"
        ],
        "scanner_source_tree_sha256": scanner_sources["twen_source_tree_sha256"],
        "scope": "primary+cooldown validation union vs primary+cooldown train union",
        "near_duplicate_threshold": attest.REQUIRED_NEAR_DUPLICATE_THRESHOLD,
        "phase_train_disjointness": phase_identity,
        "phases": phases,
        "identity_reverification": reverification,
        "metrics": {metric: 0 for _, metric in gate_contract.values()},
        "gates": {
            name: {
                "algorithm": algorithm,
                "matches": 0,
                "passed": True,
                **(
                    {"estimated_jaccard_threshold": (attest.REQUIRED_NEAR_DUPLICATE_THRESHOLD)}
                    if algorithm == attest.NEAR_DUPLICATE_ALGORITHM
                    else {}
                ),
            }
            for name, (algorithm, _metric) in gate_contract.items()
        },
        "stores_raw_text": False,
        "passed": True,
    }
    path = tmp_path / "formal" / "attestation.json"
    path.parent.mkdir()

    def publish() -> None:
        payload.pop("attestation_fingerprint", None)
        payload["attestation_fingerprint"] = attest._canonical_sha256(payload)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        (path.parent / "COMPLETE").write_text(
            json.dumps(
                {
                    "schema_version": attest.SCHEMA_VERSION,
                    "kind": attest.COMPLETE_KIND,
                    "attestation": path.name,
                    "attestation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "attestation_fingerprint": payload["attestation_fingerprint"],
                    "passed": payload["passed"],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        attest,
        "_attested_phase",
        lambda **kwargs: (SimpleNamespace(), phases[str(kwargs["phase"])]),
    )
    monkeypatch.setattr(
        attest,
        "_validate_phase_attestation",
        lambda *_args, **_kwargs: phase_identity,
    )
    publish()
    assert attest.validate_formal_validation_disjointness_attestation(path)["passed"] is True

    gates = payload["gates"]
    assert isinstance(gates, dict)
    removed = gates.pop("validation_internal_stable_id")
    publish()
    with pytest.raises(attest.DataAuditError, match="gate/metrics/scope contract differs"):
        attest.validate_formal_validation_disjointness_attestation(path)

    gates["validation_internal_stable_id"] = removed
    stable_gate = gates["train_validation_stable_id"]
    assert isinstance(stable_gate, dict)
    stable_gate["matches"] = 1
    stable_gate["passed"] = True
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    metrics["train_validation_stable_id_matches"] = 1
    publish()
    with pytest.raises(attest.DataAuditError, match="gate semantics differ"):
        attest.validate_formal_validation_disjointness_attestation(path)

    stable_gate["matches"] = 0
    stable_gate["passed"] = True
    metrics["train_validation_stable_id_matches"] = 0
    publish()

    def replace_attestation_during_validation(
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        path.write_text("{}\n", encoding="utf-8")
        return phase_identity

    monkeypatch.setattr(
        attest,
        "_validate_phase_attestation",
        replace_attestation_during_validation,
    )
    with pytest.raises(attest.DataAuditError, match="changed during validation"):
        attest.validate_formal_validation_disjointness_attestation(path)


def test_cli_returns_nonzero_when_any_gate_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "attestation.json"
    monkeypatch.setattr(
        attest,
        "build_formal_validation_disjointness_attestation",
        lambda **_kwargs: output,
    )
    monkeypatch.setattr(
        attest,
        "validate_formal_validation_disjointness_attestation",
        lambda _path: {
            "attestation_fingerprint": "1" * 64,
            "passed": False,
            "gates": {"no_overlap": {"passed": False}},
        },
    )
    monkeypatch.setattr(attest, "sha256_file", lambda _path: "2" * 64)
    arguments: list[str] = []
    for name in (
        "primary-manifest",
        "primary-audit",
        "primary-train-prepared",
        "primary-validation-prepared",
        "cooldown-manifest",
        "cooldown-audit",
        "cooldown-train-prepared",
        "cooldown-validation-prepared",
        "phase-disjointness-attestation",
        "output",
    ):
        arguments.extend((f"--{name}", str(tmp_path / name)))
    assert attest.main(arguments) == 2
    captured = capsys.readouterr()
    assert '"ok": false' in captured.out
    assert "gates failed" in captured.err
