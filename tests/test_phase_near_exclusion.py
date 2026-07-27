from __future__ import annotations

import hashlib
import io
import json
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from twen.cli import build_parser
from twen.data import phase_near_exclusion as near
from twen.data.audits import DataAuditError, _normalize_text
from twen.data.phase_exclusion import _Attribution
from twen.source_identity import twen_source_tree_sha256


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _failed_attestation(
    root: Path,
    *,
    primary_identity: dict[str, object],
    cooldown_identity: dict[str, object],
    count: int = 35,
    example_count: int | None = None,
    source_tree_sha256: str | None = None,
) -> tuple[Path, str]:
    examples = []
    for index in range(count if example_count is None else example_count):
        examples.append(
            {
                "primary": {
                    "source_id": "primary",
                    "path": "primary/train.jsonl",
                    "line": index + 1,
                    "normalized_text_sha256": f"{index + 1:064x}",
                },
                "cooldown": {
                    "source_id": "cooldown",
                    "path": "cooldown/train.jsonl",
                    "line": index + 1,
                    "normalized_text_sha256": f"{index + 1000:064x}",
                },
                "estimated_jaccard": 0.8125,
            }
        )
    scanner = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "attest_v4_phase_disjointness.py"
    )
    value = {
        "schema_version": 1,
        "kind": "twen_v4_phase_disjointness_attestation",
        "scanner_source_sha256": hashlib.sha256(scanner.read_bytes()).hexdigest(),
        "scanner_source_tree_sha256": (
            source_tree_sha256 or twen_source_tree_sha256()
        ),
        "primary": primary_identity,
        "cooldown": cooldown_identity,
        "scope": "authenticated_train_inventories_only",
        "metrics": {
            "stable_id_exact_matches": 0,
            "normalized_text_exact_matches": 0,
            "near_duplicate_matches": count,
        },
        "gates": {
            "stable_id_exact": {
                "algorithm": near._STABLE_ID_ALGORITHM,
                "matches": 0,
                "passed": True,
            },
            "normalized_text_exact": {
                "algorithm": near._NORMALIZED_EXACT_ALGORITHM,
                "matches": 0,
                "passed": True,
            },
            "near_duplicate": {
                "algorithm": near._NEAR_DUPLICATE_ALGORITHM,
                "estimated_jaccard_threshold": 0.8,
                "matches": count,
                "passed": False,
            },
        },
        "examples": {
            "stable_id_exact": [],
            "normalized_text_exact": [],
            "near_duplicate": examples,
        },
        "stores_raw_text": False,
        "passed": False,
    }
    value["attestation_fingerprint"] = near._canonical_sha256(value)
    root.mkdir(parents=True)
    path = root / "attestation.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    attestation_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "COMPLETE").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "twen_v4_phase_disjointness_complete",
                "attestation": path.name,
                "attestation_sha256": attestation_sha,
                "attestation_fingerprint": value["attestation_fingerprint"],
                "passed": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path, attestation_sha


def test_failed_phase_attestation_requires_exhaustive_35_and_external_pin(
    tmp_path: Path,
) -> None:
    primary = {"manifest_path": "/primary", "prepared": {"manifest_path": "/p"}}
    cooldown = {"manifest_path": "/cooldown", "prepared": {"manifest_path": "/c"}}
    path, digest = _failed_attestation(
        tmp_path / "complete",
        primary_identity=primary,
        cooldown_identity=cooldown,
    )
    evidence = near._authenticate_failed_phase_attestation(
        attestation_path=path,
        expected_attestation_sha256=digest,
        expected_near_matches=35,
        primary_identity=primary,
        cooldown_identity=cooldown,
    )
    assert len(evidence.matches) == 35
    assert len({match.cooldown.location for match in evidence.matches}) == 35

    with pytest.raises(DataAuditError, match="external pin"):
        near._authenticate_failed_phase_attestation(
            attestation_path=path,
            expected_attestation_sha256="0" * 64,
            expected_near_matches=35,
            primary_identity=primary,
            cooldown_identity=cooldown,
        )

    truncated, truncated_sha = _failed_attestation(
        tmp_path / "truncated",
        primary_identity=primary,
        cooldown_identity=cooldown,
        count=35,
        example_count=20,
    )
    with pytest.raises(DataAuditError, match="truncated"):
        near._authenticate_failed_phase_attestation(
            attestation_path=truncated,
            expected_attestation_sha256=truncated_sha,
            expected_near_matches=35,
            primary_identity=primary,
            cooldown_identity=cooldown,
        )


def test_failed_phase_attestation_rejects_source_tree_drift(tmp_path: Path) -> None:
    primary = {"manifest_path": "/primary", "prepared": {"manifest_path": "/p"}}
    cooldown = {"manifest_path": "/cooldown", "prepared": {"manifest_path": "/c"}}
    path, digest = _failed_attestation(
        tmp_path / "drift",
        primary_identity=primary,
        cooldown_identity=cooldown,
        source_tree_sha256="f" * 64,
    )
    with pytest.raises(DataAuditError, match="source tree changed"):
        near._authenticate_failed_phase_attestation(
            attestation_path=path,
            expected_attestation_sha256=digest,
            expected_near_matches=35,
            primary_identity=primary,
            cooldown_identity=cooldown,
        )


def test_every_one_of_35_near_pairs_is_recomputed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = " ".join(f"token_{index}" for index in range(400))
    primary_locator = near._DocumentLocator(
        source_id="primary",
        path="primary/train.jsonl",
        line=1,
        normalized_text_sha256=near._normalized_text_sha256(common),
    )
    primary_texts = {primary_locator: common}
    cooldown_texts: dict[near._DocumentLocator, str] = {}
    matches = []
    for index in range(35):
        text = f"{common} distinct_tail_{index}"
        locator = near._DocumentLocator(
            source_id="cooldown",
            path="cooldown/train.jsonl",
            line=index + 1,
            normalized_text_sha256=near._normalized_text_sha256(text),
        )
        cooldown_texts[locator] = text
        primary_signature = near._one_permutation_signature(
            near._lexical_tokens(common),
            primary_locator.normalized_text_sha256,
        )
        cooldown_signature = near._one_permutation_signature(
            near._lexical_tokens(text),
            locator.normalized_text_sha256,
        )
        similarity = near._signature_similarity(
            primary_signature,
            cooldown_signature,
        )
        assert similarity >= 0.8
        matches.append(
            near._NearMatch(
                primary=primary_locator,
                cooldown=locator,
                estimated_jaccard=similarity,
            )
        )

    def locator_texts(
        _phase: object,
        locators: set[near._DocumentLocator],
        *,
        label: str,
    ) -> dict[near._DocumentLocator, str]:
        source = primary_texts if label == "primary" else cooldown_texts
        return {locator: source[locator] for locator in locators}

    monkeypatch.setattr(near, "_load_locator_texts", locator_texts)
    calls = 0
    original_similarity = near._signature_similarity

    def counted_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
        nonlocal calls
        calls += 1
        return original_similarity(left, right)

    monkeypatch.setattr(near, "_signature_similarity", counted_similarity)
    verified = near._recompute_near_matches(
        object(),
        object(),
        tuple(matches),
    )
    assert len(verified) == 35
    assert calls == 35


def _attribution(
    *,
    line: int,
    text: str,
    split: str,
    source_id: str = "code_source",
) -> _Attribution:
    normalized = _normalize_text(text, code=True)
    return _Attribution(
        relative="chunk/attribution.jsonl",
        line_number=line,
        source_id=source_id,
        stable_id=f"{line:064x}",
        text_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
        token_count=10 + line,
        split=split,
    )


def test_locator_projection_uses_audited_code_hash_and_preserves_validation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "input"
    train = root / "chunk/train.jsonl"
    validation = root / "chunk/validation.jsonl"
    train.parent.mkdir(parents=True)
    selected_text = "def  answer():\r\n    return  42"
    retained_text = "def retained():\n    return 7"
    validation_text = "validation bytes\r\nremain exact"
    train.write_bytes(
        b"".join(
            (
                (json.dumps({"text": selected_text}) + "\n").encode(),
                (json.dumps({"text": retained_text}) + "\n").encode(),
            )
        )
    )
    validation.write_bytes(
        (json.dumps({"text": validation_text}) + "\n").encode()
    )
    validation_before = validation.read_bytes()
    phase = SimpleNamespace(
        manifest_path=root / "corpus-manifest.json",
        value={
            "train_files": [{"path": "chunk/train.jsonl"}],
            "validation_files": [{"path": "chunk/validation.jsonl"}],
        },
    )
    records = (
        _attribution(line=1, text=selected_text, split="train"),
        _attribution(line=2, text=retained_text, split="train"),
        _attribution(line=3, text=validation_text, split="validation"),
    )
    queues: dict[tuple[str, str], deque[_Attribution]] = defaultdict(deque)
    for record in records:
        queues[record.document_key].append(record)
    locator = near._DocumentLocator(
        source_id="code_source",
        path="chunk/train.jsonl",
        line=1,
        normalized_text_sha256=near._normalized_text_sha256(selected_text),
    )
    match = near._NearMatch(
        primary=near._DocumentLocator(
            source_id="primary",
            path="primary/train.jsonl",
            line=9,
            normalized_text_sha256="a" * 64,
        ),
        cooldown=locator,
        estimated_jaccard=0.8125,
    )
    work = tmp_path / "output"
    excluded: set[tuple[str, int]] = set()
    states: dict[tuple[str, str], set[bool]] = defaultdict(set)
    ledger = io.StringIO()
    train_result = near._copy_documents_excluding_near_locations(
        phase=phase,
        work=work,
        role="train",
        owners={"chunk/train.jsonl": "code_source"},
        source_categories={"code_source": "code"},
        attribution=queues,
        targets={locator.location: match},
        excluded_attribution_locations=excluded,
        selection_states=states,
        ledger_handle=ledger,
    )
    validation_result = near._copy_documents_excluding_near_locations(
        phase=phase,
        work=work,
        role="validation",
        owners={"chunk/validation.jsonl": "code_source"},
        source_categories={"code_source": "code"},
        attribution=queues,
        targets={locator.location: match},
        excluded_attribution_locations=excluded,
        selection_states=states,
        ledger_handle=ledger,
    )
    near._validate_unambiguous_selection_states(states)
    assert train_result[3] == 1
    assert train_result[4] == {locator.location}
    assert validation_result[3] == 0
    assert (work / "chunk/validation.jsonl").read_bytes() == validation_before
    output_rows = [
        json.loads(line)["text"]
        for line in (work / "chunk/train.jsonl").read_text().splitlines()
    ]
    assert output_rows == [retained_text]
    assert excluded == {("chunk/attribution.jsonl", 1)}
    assert selected_text not in ledger.getvalue()


def test_partial_duplicate_text_locator_projection_fails_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "input"
    train = root / "chunk/train.jsonl"
    train.parent.mkdir(parents=True)
    text = "def duplicate():\n    return 1"
    row = (json.dumps({"text": text}) + "\n").encode()
    train.write_bytes(row + row)
    phase = SimpleNamespace(
        manifest_path=root / "corpus-manifest.json",
        value={"train_files": [{"path": "chunk/train.jsonl"}]},
    )
    records = (
        _attribution(line=1, text=text, split="train"),
        _attribution(line=2, text=text, split="train"),
    )
    queues: dict[tuple[str, str], deque[_Attribution]] = defaultdict(deque)
    for record in records:
        queues[record.document_key].append(record)
    locator = near._DocumentLocator(
        source_id="code_source",
        path="chunk/train.jsonl",
        line=1,
        normalized_text_sha256=near._normalized_text_sha256(text),
    )
    match = near._NearMatch(
        primary=near._DocumentLocator(
            source_id="primary",
            path="primary/train.jsonl",
            line=1,
            normalized_text_sha256="b" * 64,
        ),
        cooldown=locator,
        estimated_jaccard=0.8125,
    )
    states: dict[tuple[str, str], set[bool]] = defaultdict(set)
    near._copy_documents_excluding_near_locations(
        phase=phase,
        work=tmp_path / "output",
        role="train",
        owners={"chunk/train.jsonl": "code_source"},
        source_categories={"code_source": "code"},
        attribution=queues,
        targets={locator.location: match},
        excluded_attribution_locations=set(),
        selection_states=states,
        ledger_handle=io.StringIO(),
    )
    with pytest.raises(DataAuditError, match="partial duplicate-text"):
        near._validate_unambiguous_selection_states(states)


def test_near_exclusion_cli_requires_external_pin_and_match_count() -> None:
    args = build_parser().parse_args(
        [
            "data",
            "exclude-cooldown-phase-near",
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
            "--phase-attestation",
            "failed.json",
            "--phase-attestation-sha256",
            "1" * 64,
            "--expected-near-matches",
            "35",
            "--output",
            "output",
        ]
    )
    assert args.phase_attestation_sha256 == "1" * 64
    assert args.expected_near_matches == 35
