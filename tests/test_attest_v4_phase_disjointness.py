from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from twen.data.audits import DataAuditError


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "attest_v4_phase_disjointness.py"
    spec = importlib.util.spec_from_file_location(
        "attest_v4_phase_disjointness",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


phase = _load_script()


def _signature(text: str) -> tuple[str, tuple[int, ...]]:
    digest = phase._normalized_text_sha256(text)
    return digest, phase._one_permutation_signature(
        phase._lexical_tokens(text),
        digest,
    )


def test_phase_index_matches_source_scoped_stable_and_normalized_exact(
    tmp_path: Path,
) -> None:
    index = phase._PhaseIndex(tmp_path / "index.sqlite3")
    try:
        index.add_stable_id("same_source", "a" * 64, "primary/a.jsonl", 1)
        assert index.match_stable_id("same_source", "a" * 64) == (
            "primary/a.jsonl",
            1,
        )
        assert index.match_stable_id("different_source", "a" * 64) is None

        digest, signature = _signature("\uff21  concise\nreference   document")
        index.add_document(
            source_id="primary_source",
            path="primary/train.jsonl",
            line_number=7,
            normalized_sha=digest,
            signature=signature,
        )
        assert phase._normalized_text_sha256("A concise reference document") == digest
        assert index.match_exact(digest) == (
            "primary_source",
            "primary/train.jsonl",
            7,
        )
    finally:
        index.close()


def test_phase_index_reports_near_but_excludes_exact_from_near(
    tmp_path: Path,
) -> None:
    index = phase._PhaseIndex(tmp_path / "index.sqlite3")
    try:
        primary = (
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda "
            "mu nu xi omicron pi rho sigma tau"
        )
        primary_digest, primary_signature = _signature(primary)
        index.add_document(
            source_id="primary_source",
            path="primary/train.jsonl",
            line_number=3,
            normalized_sha=primary_digest,
            signature=primary_signature,
        )
        index.commit()

        assert (
            index.match_near(
                normalized_sha=primary_digest,
                signature=primary_signature,
                threshold=0.5,
            )
            is None
        )

        cooldown = primary + " upsilon"
        cooldown_digest, cooldown_signature = _signature(cooldown)
        near = index.match_near(
            normalized_sha=cooldown_digest,
            signature=cooldown_signature,
            threshold=0.5,
        )
        assert near is not None
        assert near[:4] == (
            "primary_source",
            "primary/train.jsonl",
            3,
            primary_digest,
        )
        assert near[4] >= 0.5
    finally:
        index.close()


def test_phase_scanner_constants_match_capacity_contract() -> None:
    assert phase.STABLE_ID_ALGORITHM.startswith("source-scoped-authenticated")
    assert phase.NORMALIZED_EXACT_ALGORITHM == "unicode-nfkc-whitespace-sha256-intersection-v1"
    assert phase.NEAR_DUPLICATE_ALGORITHM == "lexical-5gram-one-permutation-minhash-lsh-v1"
    assert phase.REQUIRED_NEAR_DUPLICATE_THRESHOLD == 0.8


def test_phase_scanner_rejects_weaker_near_duplicate_threshold(tmp_path: Path) -> None:
    with pytest.raises(
        DataAuditError,
        match=r"requires near-duplicate threshold 0.8",
    ):
        phase.build_phase_disjointness_attestation(
            primary_manifest=tmp_path / "missing-primary.json",
            primary_audit=tmp_path / "missing-primary-audit.json",
            primary_prepared=tmp_path / "missing-primary-prepared.json",
            cooldown_manifest=tmp_path / "missing-cooldown.json",
            cooldown_audit=tmp_path / "missing-cooldown-audit.json",
            cooldown_prepared=tmp_path / "missing-cooldown-prepared.json",
            output_root=tmp_path / "output",
            threshold=1.0,
        )


def test_phase_scanner_rejects_source_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = iter(
        (
            ("primary", {"prepared": {"dataset_fingerprint": "1" * 64}}),
            ("cooldown", {"prepared": {"dataset_fingerprint": "2" * 64}}),
            ("primary", {"prepared": {"dataset_fingerprint": "1" * 64}}),
            ("cooldown", {"prepared": {"dataset_fingerprint": "2" * 64}}),
        )
    )

    def fake_attested_corpus(*_args: object) -> tuple[object, dict[str, object], dict[str, object]]:
        name, identity = next(identities)
        return SimpleNamespace(name=name), {}, identity

    source_trees = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(phase, "_attested_corpus", fake_attested_corpus)
    monkeypatch.setattr(phase, "_iter_train_stable_ids", lambda _corpus: iter(()))
    monkeypatch.setattr(
        phase,
        "_iter_jsonl_documents",
        lambda _corpus, _role: iter(()),
    )
    monkeypatch.setattr(phase, "twen_source_tree_sha256", lambda: next(source_trees))

    output = tmp_path / "phase-attestation"
    with pytest.raises(
        DataAuditError,
        match="scanner source changed during the scan",
    ):
        phase.build_phase_disjointness_attestation(
            primary_manifest=tmp_path / "primary.json",
            primary_audit=tmp_path / "primary-audit.json",
            primary_prepared=tmp_path / "primary-prepared.json",
            cooldown_manifest=tmp_path / "cooldown.json",
            cooldown_audit=tmp_path / "cooldown-audit.json",
            cooldown_prepared=tmp_path / "cooldown-prepared.json",
            output_root=output,
        )
    assert not output.exists()
    assert not tuple(tmp_path.glob(".phase-attestation.tmp-*"))


def test_phase_scanner_rejects_input_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = iter(
        (
            ("primary", {"prepared": {"dataset_fingerprint": "1" * 64}}),
            ("cooldown", {"prepared": {"dataset_fingerprint": "2" * 64}}),
            ("primary", {"prepared": {"dataset_fingerprint": "3" * 64}}),
            ("cooldown", {"prepared": {"dataset_fingerprint": "2" * 64}}),
        )
    )

    def fake_attested_corpus(*_args: object) -> tuple[object, dict[str, object], dict[str, object]]:
        name, identity = next(identities)
        return SimpleNamespace(name=name), {}, identity

    monkeypatch.setattr(phase, "_attested_corpus", fake_attested_corpus)
    monkeypatch.setattr(phase, "_iter_train_stable_ids", lambda _corpus: iter(()))
    monkeypatch.setattr(
        phase,
        "_iter_jsonl_documents",
        lambda _corpus, _role: iter(()),
    )
    monkeypatch.setattr(phase, "twen_source_tree_sha256", lambda: "a" * 64)

    output = tmp_path / "phase-attestation"
    with pytest.raises(
        DataAuditError,
        match="input identity changed during the scan",
    ):
        phase.build_phase_disjointness_attestation(
            primary_manifest=tmp_path / "primary.json",
            primary_audit=tmp_path / "primary-audit.json",
            primary_prepared=tmp_path / "primary-prepared.json",
            cooldown_manifest=tmp_path / "cooldown.json",
            cooldown_audit=tmp_path / "cooldown-audit.json",
            cooldown_prepared=tmp_path / "cooldown-prepared.json",
            output_root=output,
        )
    assert not output.exists()
    assert not tuple(tmp_path.glob(".phase-attestation.tmp-*"))


def test_phase_scanner_cli_returns_nonzero_when_gates_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    attestation = tmp_path / "attestation.json"
    attestation.write_text(
        json.dumps(
            {
                "passed": False,
                "attestation_fingerprint": "a" * 64,
                "gates": {"near_duplicate": {"passed": False}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        phase,
        "build_phase_disjointness_attestation",
        lambda **_kwargs: attestation,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "attest_v4_phase_disjointness.py",
            "--primary-manifest",
            "primary.json",
            "--primary-audit",
            "primary-audit.json",
            "--primary-prepared",
            "primary-prepared.json",
            "--cooldown-manifest",
            "cooldown.json",
            "--cooldown-audit",
            "cooldown-audit.json",
            "--cooldown-prepared",
            "cooldown-prepared.json",
            "--output",
            "output",
        ],
    )

    assert phase.main() == 2
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert result["passed"] is False
