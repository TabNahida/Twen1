#!/usr/bin/env python3
"""Prove formal primary+cooldown validation is disjoint from all formal train data.

The existing phase-disjointness gate compares the two training phases.  This
companion gate authenticates that evidence, both phases' governed train and
validation prepared manifests, then compares the union of both validation
roles against the union of both train roles by source-scoped stable ID,
normalized exact text, and the project's MinHash/LSH near-duplicate policy.

The command is read-only with respect to every input.  It stores no raw text,
does not load a model, and never starts training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from twen.data.audits import (
    DataAuditError,
    _band_keys,
    _iter_jsonl_documents,
    _lexical_tokens,
    _one_permutation_signature,
    _packed_signature,
    _signature_similarity,
    _unpacked_signature,
)
from twen.data.prepared import validate_prepared_corpus
from twen.io.download import sha256_file
from twen.source_identity import twen_source_tree_sha256
from twen.utils import atomic_write_json

SCHEMA_VERSION = 1
KIND = "twen_v4_formal_validation_disjointness_attestation"
COMPLETE_KIND = "twen_v4_formal_validation_disjointness_complete"
NEAR_DUPLICATE_ALGORITHM = "lexical-5gram-one-permutation-minhash-lsh-v1"
NORMALIZED_EXACT_ALGORITHM = "unicode-nfkc-whitespace-sha256-intersection-v1"
STABLE_ID_ALGORITHM = "source-scoped-authenticated-stable-id-intersection-v1"
REQUIRED_NEAR_DUPLICATE_THRESHOLD = 0.8
PHASES = ("primary", "cooldown")


def _scanner_sources_identity() -> dict[str, str]:
    """Hash every source file that determines this scan's semantics."""

    return {
        "formal_validation_scanner_sha256": sha256_file(Path(__file__)),
        "phase_attestation_validator_sha256": sha256_file(
            Path(__file__).with_name("attest_v4_phase_disjointness.py")
        ),
        "twen_source_tree_sha256": twen_source_tree_sha256(),
    }


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _input_identity_fingerprint(
    phases: Mapping[str, Mapping[str, object]],
    phase_attestation: Mapping[str, object],
) -> str:
    return _canonical_sha256(
        {
            "phases": phases,
            "phase_train_disjointness": phase_attestation,
        }
    )


def _assert_end_identity_matches_start(
    *,
    scanner_sources_start: Mapping[str, str],
    scanner_sources_end: Mapping[str, str],
    phases_start: Mapping[str, Mapping[str, object]],
    phases_end: Mapping[str, Mapping[str, object]],
    phase_attestation_start: Mapping[str, object],
    phase_attestation_end: Mapping[str, object],
) -> dict[str, object]:
    if scanner_sources_end != scanner_sources_start:
        raise DataAuditError("scanner source identity changed during formal validation scan")
    input_start = _input_identity_fingerprint(
        phases_start,
        phase_attestation_start,
    )
    input_end = _input_identity_fingerprint(
        phases_end,
        phase_attestation_end,
    )
    if input_end != input_start:
        raise DataAuditError("formal validation input identity changed during scan")
    return {
        "scanner_sources_start": dict(scanner_sources_start),
        "scanner_sources_end": dict(scanner_sources_end),
        "input_identity_start_sha256": input_start,
        "input_identity_end_sha256": input_end,
        "passed": True,
    }


def _load_phase_module() -> ModuleType:
    path = Path(__file__).with_name("attest_v4_phase_disjointness.py")
    source_before = path.read_bytes()
    spec = importlib.util.spec_from_file_location("_twen_v4_phase_disjointness", path)
    if spec is None or spec.loader is None:
        raise DataAuditError(f"cannot load phase-disjointness validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if path.read_bytes() != source_before:
        raise DataAuditError(f"phase-disjointness validator changed while loading: {path}")
    return module


def _normalized_text_sha256(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", " ".join(text.split()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _safe_inventory_paths(
    manifest_path: Path,
    value: Mapping[str, object],
    field: str,
) -> tuple[Path, ...]:
    raw = value.get(field)
    if not isinstance(raw, list):
        raise DataAuditError(f"extracted manifest has no {field}")
    result: list[Path] = []
    root = manifest_path.parent.resolve()
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise DataAuditError(f"{field}[{index}] must be an object")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative:
            raise DataAuditError(f"{field}[{index}].path is invalid")
        candidate = (root / relative).resolve()
        if candidate == root or root not in candidate.parents:
            raise DataAuditError(f"{field}[{index}].path escapes corpus root")
        result.append(candidate)
    return tuple(result)


def _iter_role_stable_ids(
    corpus: Any,
    *,
    role: str,
) -> Iterator[tuple[str, str, str, int, int]]:
    if role not in {"train", "validation"}:
        raise DataAuditError(f"invalid attribution role: {role}")
    paths = _safe_inventory_paths(
        corpus.manifest_path,
        corpus.value,
        "attribution_files",
    )
    for path in paths:
        relative = path.relative_to(corpus.manifest_path.parent).as_posix()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DataAuditError(
                        f"invalid attribution JSONL at {path}:{line_number}"
                    ) from error
                if not isinstance(value, Mapping) or value.get("split") != role:
                    continue
                source_id = value.get("source_id")
                stable_id = value.get("stable_id")
                token_count = value.get("token_count_with_eos")
                if (
                    not isinstance(source_id, str)
                    or not source_id
                    or not isinstance(stable_id, str)
                    or len(stable_id) != 64
                    or any(character not in "0123456789abcdef" for character in stable_id)
                    or isinstance(token_count, bool)
                    or not isinstance(token_count, int)
                    or token_count <= 0
                ):
                    raise DataAuditError(f"invalid {role} attribution at {path}:{line_number}")
                yield source_id, stable_id, relative, line_number, token_count


class _ReferenceIndex:
    """Disk-backed exact/stable/LSH index with authenticated document locators."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE stable_ids (
                id INTEGER PRIMARY KEY,
                source_id TEXT NOT NULL,
                stable_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                role TEXT NOT NULL,
                file_path TEXT NOT NULL,
                line_number INTEGER NOT NULL
            );
            CREATE INDEX stable_ids_key ON stable_ids(source_id,stable_id);
            CREATE TABLE docs (
                id INTEGER PRIMARY KEY,
                normalized_sha TEXT NOT NULL,
                source_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                role TEXT NOT NULL,
                file_path TEXT NOT NULL,
                line_number INTEGER NOT NULL,
                signature BLOB NOT NULL
            );
            CREATE INDEX docs_normalized_sha ON docs(normalized_sha);
            CREATE TABLE bands (band_key BLOB NOT NULL, doc_id INTEGER NOT NULL);
            CREATE INDEX bands_key ON bands(band_key);
            """
        )

    def close(self) -> None:
        self.connection.close()

    def commit(self) -> None:
        self.connection.commit()

    def add_stable_id(
        self,
        *,
        source_id: str,
        stable_id: str,
        phase: str,
        role: str,
        path: str,
        line_number: int,
    ) -> None:
        self.connection.execute(
            "INSERT INTO stable_ids"
            "(source_id,stable_id,phase,role,file_path,line_number) "
            "VALUES(?,?,?,?,?,?)",
            (source_id, stable_id, phase, role, path, line_number),
        )

    def match_stable_id(
        self,
        *,
        source_id: str,
        stable_id: str,
    ) -> tuple[str, str, str, int] | None:
        row = self.connection.execute(
            "SELECT phase,role,file_path,line_number FROM stable_ids "
            "WHERE source_id=? AND stable_id=? LIMIT 1",
            (source_id, stable_id),
        ).fetchone()
        return None if row is None else (str(row[0]), str(row[1]), str(row[2]), int(row[3]))

    def add_document(
        self,
        *,
        normalized_sha: str,
        source_id: str,
        phase: str,
        role: str,
        path: str,
        line_number: int,
        signature: Sequence[int],
    ) -> None:
        packed = _packed_signature(signature)
        cursor = self.connection.execute(
            "INSERT INTO docs"
            "(normalized_sha,source_id,phase,role,file_path,line_number,signature) "
            "VALUES(?,?,?,?,?,?,?)",
            (
                normalized_sha,
                source_id,
                phase,
                role,
                path,
                line_number,
                packed,
            ),
        )
        doc_id = int(cursor.lastrowid)
        self.connection.executemany(
            "INSERT INTO bands(band_key,doc_id) VALUES(?,?)",
            ((key, doc_id) for key in _band_keys(signature)),
        )

    def match_exact(
        self,
        normalized_sha: str,
    ) -> tuple[str, str, str, str, int] | None:
        row = self.connection.execute(
            "SELECT source_id,phase,role,file_path,line_number FROM docs "
            "WHERE normalized_sha=? LIMIT 1",
            (normalized_sha,),
        ).fetchone()
        return (
            None
            if row is None
            else (str(row[0]), str(row[1]), str(row[2]), str(row[3]), int(row[4]))
        )

    def match_near(
        self,
        *,
        normalized_sha: str,
        signature: Sequence[int],
        threshold: float,
    ) -> tuple[str, str, str, str, int, str, float] | None:
        candidate_ids: set[int] = set()
        for key in _band_keys(signature):
            candidate_ids.update(
                int(row[0])
                for row in self.connection.execute(
                    "SELECT doc_id FROM bands WHERE band_key=?",
                    (key,),
                )
            )
        best: tuple[str, str, str, str, int, str, float] | None = None
        for doc_id in candidate_ids:
            row = self.connection.execute(
                "SELECT normalized_sha,source_id,phase,role,file_path,line_number,signature "
                "FROM docs WHERE id=?",
                (doc_id,),
            ).fetchone()
            if row is None or row[0] == normalized_sha:
                continue
            similarity = _signature_similarity(
                signature,
                _unpacked_signature(row[6]),
            )
            if similarity < threshold:
                continue
            candidate = (
                str(row[1]),
                str(row[2]),
                str(row[3]),
                str(row[4]),
                int(row[5]),
                str(row[0]),
                similarity,
            )
            if best is None or candidate[-1] > best[-1]:
                best = candidate
        return best


def _prepared_validation_identity(
    *,
    manifest_path: Path,
    audit_path: Path,
    prepared_path: Path,
) -> dict[str, object]:
    prepared_file = prepared_path.resolve()
    prepared = validate_prepared_corpus(prepared_file)
    lineage = getattr(prepared, "lineage", None)
    if (
        not isinstance(lineage, Mapping)
        or lineage.get("role") != "validation"
        or lineage.get("ready_for_training") is not True
        or lineage.get("research_only") is not False
        or lineage.get("pending_audits")
        or Path(str(lineage.get("extracted_manifest_path"))).resolve() != manifest_path.resolve()
        or lineage.get("extracted_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise DataAuditError(
            f"prepared validation is not governed by {manifest_path}: {prepared_file}"
        )
    audit_lineage = lineage.get("audit_attestation")
    if (
        not isinstance(audit_lineage, Mapping)
        or audit_lineage.get("bound_as") != "frozen_validation"
        or audit_lineage.get("ready_for_training") is not True
        or Path(str(audit_lineage.get("path"))).resolve() != audit_path.resolve()
        or audit_lineage.get("sha256") != sha256_file(audit_path)
    ):
        raise DataAuditError(
            f"prepared validation does not bind the supplied audit: {prepared_file}"
        )
    return {
        "manifest_path": str(prepared_file),
        "manifest_sha256": sha256_file(prepared_file),
        "dataset_fingerprint": prepared.dataset_fingerprint,
        "sequence_count": prepared.sequence_count,
        "token_count": prepared.token_count,
    }


def _attested_phase(
    *,
    phase: str,
    manifest_path: str | Path,
    audit_path: str | Path,
    train_prepared_path: str | Path,
    validation_prepared_path: str | Path,
    phase_module: ModuleType,
) -> tuple[Any, dict[str, object]]:
    manifest_file = Path(manifest_path).resolve()
    audit_file = Path(audit_path).resolve()
    train_prepared_file = Path(train_prepared_path).resolve()
    corpus, audit, train_identity = phase_module._attested_corpus(
        manifest_file,
        audit_file,
        train_prepared_file,
    )
    frozen = audit.get("frozen_validation")
    if (
        not isinstance(frozen, Mapping)
        or frozen.get("role") != "validation"
        or frozen.get("manifest_sha256") != corpus.manifest_sha256
        or Path(str(frozen.get("manifest_path"))).resolve() != corpus.manifest_path
    ):
        raise DataAuditError(
            f"{phase} audit does not bind the same finalized corpus as frozen validation"
        )
    validation_identity = _prepared_validation_identity(
        manifest_path=manifest_file,
        audit_path=audit_file,
        prepared_path=Path(validation_prepared_path),
    )
    return (
        corpus,
        {
            **train_identity,
            "phase": phase,
            "validation_prepared": validation_identity,
        },
    )


def _validate_phase_attestation(
    path: Path,
    *,
    identities: Mapping[str, Mapping[str, object]],
    phase_module: ModuleType,
) -> dict[str, object]:
    attestation = path.resolve()
    raw = attestation.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise DataAuditError(f"invalid phase-disjointness attestation: {attestation}") from error
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != phase_module.SCHEMA_VERSION
        or value.get("kind") != phase_module.KIND
        or value.get("passed") is not True
    ):
        raise DataAuditError("phase-disjointness attestation is not passing")
    fingerprint = value.get("attestation_fingerprint")
    unsigned = dict(value)
    unsigned.pop("attestation_fingerprint", None)
    if fingerprint != _canonical_sha256(unsigned):
        raise DataAuditError("phase-disjointness attestation fingerprint mismatch")
    phase_scanner_path = Path(str(phase_module.__file__)).resolve()
    if (
        value.get("scanner_source_sha256") != sha256_file(phase_scanner_path)
        or value.get("scanner_source_tree_sha256") != twen_source_tree_sha256()
        or value.get("scope") != "authenticated_train_inventories_only"
        or value.get("stores_raw_text") is not False
    ):
        raise DataAuditError("phase-disjointness scanner identity/scope is stale")
    complete_path = attestation.parent / "COMPLETE"
    try:
        complete_raw = complete_path.read_bytes()
        complete = json.loads(complete_raw)
    except (OSError, json.JSONDecodeError) as error:
        raise DataAuditError("phase-disjointness has no valid COMPLETE") from error
    if (
        not isinstance(complete, Mapping)
        or complete.get("schema_version") != phase_module.SCHEMA_VERSION
        or complete.get("kind") != "twen_v4_phase_disjointness_complete"
        or complete.get("attestation") != attestation.name
        or complete.get("attestation_sha256") != hashlib.sha256(raw).hexdigest()
        or complete.get("attestation_fingerprint") != fingerprint
        or complete.get("passed") is not True
    ):
        raise DataAuditError("phase-disjointness COMPLETE mismatch")
    gates = value.get("gates")
    metrics = value.get("metrics")
    required_gates = {
        "stable_id_exact": (
            phase_module.STABLE_ID_ALGORITHM,
            "stable_id_exact_matches",
        ),
        "normalized_text_exact": (
            phase_module.NORMALIZED_EXACT_ALGORITHM,
            "normalized_text_exact_matches",
        ),
        "near_duplicate": (
            phase_module.NEAR_DUPLICATE_ALGORITHM,
            "near_duplicate_matches",
        ),
    }
    if (
        not isinstance(gates, Mapping)
        or set(gates) != set(required_gates)
        or not isinstance(metrics, Mapping)
    ):
        raise DataAuditError("phase-disjointness gate/metrics contract differs")
    for gate_name, (algorithm, metric_name) in required_gates.items():
        gate = gates[gate_name]
        metric_count = metrics.get(metric_name)
        if (
            not isinstance(gate, Mapping)
            or gate.get("algorithm") != algorithm
            or gate.get("passed") is not True
            or gate.get("matches") != 0
            or isinstance(metric_count, bool)
            or not isinstance(metric_count, int)
            or metric_count != gate.get("matches")
            or (
                gate_name == "near_duplicate"
                and gate.get("estimated_jaccard_threshold")
                != phase_module.REQUIRED_NEAR_DUPLICATE_THRESHOLD
            )
        ):
            raise DataAuditError(f"phase-disjointness gate {gate_name!r} differs")
    for phase in PHASES:
        actual = value.get(phase)
        expected = identities[phase]
        expected_train_identity = {
            key: item
            for key, item in expected.items()
            if key not in {"phase", "validation_prepared"}
        }
        if not isinstance(actual, Mapping) or dict(actual) != expected_train_identity:
            raise DataAuditError(f"phase-disjointness attestation does not bind {phase} inputs")
    if attestation.read_bytes() != raw or complete_path.read_bytes() != complete_raw:
        raise DataAuditError("phase-disjointness attestation changed during validation")
    return {
        "path": str(attestation),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "attestation_fingerprint": fingerprint,
        "gates": gates,
    }


def _locator(
    *,
    source_id: str,
    phase: str,
    role: str,
    path: str,
    line: int,
    digest_field: str,
    digest: str,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "phase": phase,
        "role": role,
        "path": path,
        "line": line,
        digest_field: digest,
    }


def _index_locator(
    match: Sequence[Any],
    *,
    digest_field: str,
    digest: str,
) -> dict[str, object]:
    return _locator(
        source_id=str(match[0]),
        phase=str(match[1]),
        role=str(match[2]),
        path=str(match[3]),
        line=int(match[4]),
        digest_field=digest_field,
        digest=digest,
    )


def _stable_locator(
    *,
    source_id: str,
    stable_id: str,
    match: Sequence[Any],
) -> dict[str, object]:
    return _locator(
        source_id=source_id,
        phase=str(match[0]),
        role=str(match[1]),
        path=str(match[2]),
        line=int(match[3]),
        digest_field="stable_id",
        digest=stable_id,
    )


def _record_example(
    examples: dict[str, list[dict[str, object]]],
    name: str,
    payload: dict[str, object],
    *,
    max_examples: int,
) -> None:
    if len(examples[name]) < max_examples:
        examples[name].append(payload)


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def build_formal_validation_disjointness_attestation(
    *,
    primary_manifest: str | Path,
    primary_audit: str | Path,
    primary_train_prepared: str | Path,
    primary_validation_prepared: str | Path,
    cooldown_manifest: str | Path,
    cooldown_audit: str | Path,
    cooldown_train_prepared: str | Path,
    cooldown_validation_prepared: str | Path,
    phase_disjointness_attestation: str | Path,
    output_root: str | Path,
    threshold: float = REQUIRED_NEAR_DUPLICATE_THRESHOLD,
    max_examples: int = 20,
    progress_every: int = 10_000,
    progress: bool = False,
) -> Path:
    if threshold != REQUIRED_NEAR_DUPLICATE_THRESHOLD:
        raise DataAuditError(
            "formal validation disjointness requires near-duplicate threshold "
            f"{REQUIRED_NEAR_DUPLICATE_THRESHOLD}"
        )
    if max_examples < 0:
        raise DataAuditError("max_examples must be non-negative")
    if progress_every <= 0:
        raise DataAuditError("progress_every must be positive")
    scanner_sources_start = _scanner_sources_identity()
    phase_module = _load_phase_module()
    corpora: dict[str, Any] = {}
    identities: dict[str, dict[str, object]] = {}
    phase_inputs = (
        (
            "primary",
            primary_manifest,
            primary_audit,
            primary_train_prepared,
            primary_validation_prepared,
        ),
        (
            "cooldown",
            cooldown_manifest,
            cooldown_audit,
            cooldown_train_prepared,
            cooldown_validation_prepared,
        ),
    )
    for phase, manifest, audit, train_prepared, validation_prepared in phase_inputs:
        corpora[phase], identities[phase] = _attested_phase(
            phase=phase,
            manifest_path=manifest,
            audit_path=audit,
            train_prepared_path=train_prepared,
            validation_prepared_path=validation_prepared,
            phase_module=phase_module,
        )
    phase_identity = _validate_phase_attestation(
        Path(phase_disjointness_attestation),
        identities=identities,
        phase_module=phase_module,
    )

    output = Path(output_root).resolve()
    if output.exists():
        raise DataAuditError(f"output already exists; choose a new directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.tmp-",
            dir=output.parent,
        )
    )
    train_db = work / "train-index.sqlite3"
    validation_db = work / "validation-index.sqlite3"
    train_index = _ReferenceIndex(train_db)
    validation_index = _ReferenceIndex(validation_db)
    metrics: dict[str, Any] = {
        "train_documents": {phase: 0 for phase in PHASES},
        "train_attribution_rows": {phase: 0 for phase in PHASES},
        "train_attributed_tokens": {phase: 0 for phase in PHASES},
        "validation_documents": {phase: 0 for phase in PHASES},
        "validation_attribution_rows": {phase: 0 for phase in PHASES},
        "validation_attributed_tokens": {phase: 0 for phase in PHASES},
        "train_validation_stable_id_matches": 0,
        "train_validation_normalized_exact_matches": 0,
        "train_validation_near_duplicate_matches": 0,
        "validation_internal_stable_id_matches": 0,
        "validation_internal_normalized_exact_matches": 0,
        "validation_internal_near_duplicate_matches": 0,
    }
    examples: dict[str, list[dict[str, object]]] = {
        "train_validation_stable_id": [],
        "train_validation_normalized_exact": [],
        "train_validation_near_duplicate": [],
        "validation_internal_stable_id": [],
        "validation_internal_normalized_exact": [],
        "validation_internal_near_duplicate": [],
    }
    completed = False
    try:
        for phase in PHASES:
            corpus = corpora[phase]
            for source_id, stable_id, path, line, token_count in _iter_role_stable_ids(
                corpus,
                role="train",
            ):
                train_index.add_stable_id(
                    source_id=source_id,
                    stable_id=stable_id,
                    phase=phase,
                    role="train",
                    path=path,
                    line_number=line,
                )
                metrics["train_attribution_rows"][phase] += 1
                metrics["train_attributed_tokens"][phase] += token_count
                if metrics["train_attribution_rows"][phase] % progress_every == 0:
                    train_index.commit()
                    _progress(
                        progress,
                        f"{phase} train attribution {metrics['train_attribution_rows'][phase]:,}",
                    )
            train_index.commit()
            for path, source_id, _category, line, text in _iter_jsonl_documents(
                corpus,
                "train",
            ):
                normalized_sha = _normalized_text_sha256(text)
                signature = _one_permutation_signature(
                    _lexical_tokens(text),
                    normalized_sha,
                )
                train_index.add_document(
                    normalized_sha=normalized_sha,
                    source_id=source_id,
                    phase=phase,
                    role="train",
                    path=path,
                    line_number=line,
                    signature=signature,
                )
                metrics["train_documents"][phase] += 1
                if metrics["train_documents"][phase] % progress_every == 0:
                    train_index.commit()
                    _progress(
                        progress,
                        f"{phase} train text {metrics['train_documents'][phase]:,}",
                    )
            train_index.commit()

        for phase in PHASES:
            corpus = corpora[phase]
            for source_id, stable_id, path, line, token_count in _iter_role_stable_ids(
                corpus,
                role="validation",
            ):
                train_match = train_index.match_stable_id(
                    source_id=source_id,
                    stable_id=stable_id,
                )
                validation_match = validation_index.match_stable_id(
                    source_id=source_id,
                    stable_id=stable_id,
                )
                if train_match is not None:
                    metrics["train_validation_stable_id_matches"] += 1
                    _record_example(
                        examples,
                        "train_validation_stable_id",
                        {
                            "train": _stable_locator(
                                source_id=source_id,
                                stable_id=stable_id,
                                match=train_match,
                            ),
                            "validation": _locator(
                                source_id=source_id,
                                phase=phase,
                                role="validation",
                                path=path,
                                line=line,
                                digest_field="stable_id",
                                digest=stable_id,
                            ),
                        },
                        max_examples=max_examples,
                    )
                if validation_match is not None:
                    metrics["validation_internal_stable_id_matches"] += 1
                    _record_example(
                        examples,
                        "validation_internal_stable_id",
                        {
                            "first": _stable_locator(
                                source_id=source_id,
                                stable_id=stable_id,
                                match=validation_match,
                            ),
                            "second": _locator(
                                source_id=source_id,
                                phase=phase,
                                role="validation",
                                path=path,
                                line=line,
                                digest_field="stable_id",
                                digest=stable_id,
                            ),
                        },
                        max_examples=max_examples,
                    )
                validation_index.add_stable_id(
                    source_id=source_id,
                    stable_id=stable_id,
                    phase=phase,
                    role="validation",
                    path=path,
                    line_number=line,
                )
                metrics["validation_attribution_rows"][phase] += 1
                metrics["validation_attributed_tokens"][phase] += token_count
            validation_index.commit()

            for path, source_id, _category, line, text in _iter_jsonl_documents(
                corpus,
                "validation",
            ):
                normalized_sha = _normalized_text_sha256(text)
                signature = _one_permutation_signature(
                    _lexical_tokens(text),
                    normalized_sha,
                )
                document = _locator(
                    source_id=source_id,
                    phase=phase,
                    role="validation",
                    path=path,
                    line=line,
                    digest_field="normalized_text_sha256",
                    digest=normalized_sha,
                )
                train_exact = train_index.match_exact(normalized_sha)
                train_near = train_index.match_near(
                    normalized_sha=normalized_sha,
                    signature=signature,
                    threshold=threshold,
                )
                validation_exact = validation_index.match_exact(normalized_sha)
                validation_near = validation_index.match_near(
                    normalized_sha=normalized_sha,
                    signature=signature,
                    threshold=threshold,
                )
                if train_exact is not None:
                    metrics["train_validation_normalized_exact_matches"] += 1
                    _record_example(
                        examples,
                        "train_validation_normalized_exact",
                        {
                            "train": _index_locator(
                                train_exact,
                                digest_field="normalized_text_sha256",
                                digest=normalized_sha,
                            ),
                            "validation": document,
                        },
                        max_examples=max_examples,
                    )
                if train_near is not None:
                    metrics["train_validation_near_duplicate_matches"] += 1
                    _record_example(
                        examples,
                        "train_validation_near_duplicate",
                        {
                            "train": _index_locator(
                                train_near,
                                digest_field="normalized_text_sha256",
                                digest=str(train_near[5]),
                            ),
                            "validation": document,
                            "estimated_jaccard": float(train_near[6]),
                        },
                        max_examples=max_examples,
                    )
                if validation_exact is not None:
                    metrics["validation_internal_normalized_exact_matches"] += 1
                    _record_example(
                        examples,
                        "validation_internal_normalized_exact",
                        {
                            "first": _index_locator(
                                validation_exact,
                                digest_field="normalized_text_sha256",
                                digest=normalized_sha,
                            ),
                            "second": document,
                        },
                        max_examples=max_examples,
                    )
                if validation_near is not None:
                    metrics["validation_internal_near_duplicate_matches"] += 1
                    _record_example(
                        examples,
                        "validation_internal_near_duplicate",
                        {
                            "first": _index_locator(
                                validation_near,
                                digest_field="normalized_text_sha256",
                                digest=str(validation_near[5]),
                            ),
                            "second": document,
                            "estimated_jaccard": float(validation_near[6]),
                        },
                        max_examples=max_examples,
                    )
                validation_index.add_document(
                    normalized_sha=normalized_sha,
                    source_id=source_id,
                    phase=phase,
                    role="validation",
                    path=path,
                    line_number=line,
                    signature=signature,
                )
                metrics["validation_documents"][phase] += 1
            validation_index.commit()
        end_identities: dict[str, dict[str, object]] = {}
        for phase, manifest, audit, train_prepared, validation_prepared in phase_inputs:
            _, end_identities[phase] = _attested_phase(
                phase=phase,
                manifest_path=manifest,
                audit_path=audit,
                train_prepared_path=train_prepared,
                validation_prepared_path=validation_prepared,
                phase_module=phase_module,
            )
        end_phase_identity = _validate_phase_attestation(
            Path(phase_disjointness_attestation),
            identities=end_identities,
            phase_module=phase_module,
        )
        identity_reverification = _assert_end_identity_matches_start(
            scanner_sources_start=scanner_sources_start,
            scanner_sources_end=_scanner_sources_identity(),
            phases_start=identities,
            phases_end=end_identities,
            phase_attestation_start=phase_identity,
            phase_attestation_end=end_phase_identity,
        )
        completed = True
    finally:
        train_index.close()
        validation_index.close()
        if not completed:
            shutil.rmtree(work, ignore_errors=True)

    for database in (train_db, validation_db):
        database.unlink(missing_ok=True)
        Path(str(database) + "-wal").unlink(missing_ok=True)
        Path(str(database) + "-shm").unlink(missing_ok=True)
    gates = {
        "train_validation_stable_id": {
            "algorithm": STABLE_ID_ALGORITHM,
            "matches": metrics["train_validation_stable_id_matches"],
            "passed": metrics["train_validation_stable_id_matches"] == 0,
        },
        "train_validation_normalized_exact": {
            "algorithm": NORMALIZED_EXACT_ALGORITHM,
            "matches": metrics["train_validation_normalized_exact_matches"],
            "passed": metrics["train_validation_normalized_exact_matches"] == 0,
        },
        "train_validation_near_duplicate": {
            "algorithm": NEAR_DUPLICATE_ALGORITHM,
            "estimated_jaccard_threshold": threshold,
            "matches": metrics["train_validation_near_duplicate_matches"],
            "passed": metrics["train_validation_near_duplicate_matches"] == 0,
        },
        "validation_internal_stable_id": {
            "algorithm": STABLE_ID_ALGORITHM,
            "matches": metrics["validation_internal_stable_id_matches"],
            "passed": metrics["validation_internal_stable_id_matches"] == 0,
        },
        "validation_internal_normalized_exact": {
            "algorithm": NORMALIZED_EXACT_ALGORITHM,
            "matches": metrics["validation_internal_normalized_exact_matches"],
            "passed": metrics["validation_internal_normalized_exact_matches"] == 0,
        },
        "validation_internal_near_duplicate": {
            "algorithm": NEAR_DUPLICATE_ALGORITHM,
            "estimated_jaccard_threshold": threshold,
            "matches": metrics["validation_internal_near_duplicate_matches"],
            "passed": metrics["validation_internal_near_duplicate_matches"] == 0,
        },
    }
    passed = all(bool(gate["passed"]) for gate in gates.values())
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "scanner_source_sha256": scanner_sources_start["formal_validation_scanner_sha256"],
        "phase_attestation_validator_source_sha256": scanner_sources_start[
            "phase_attestation_validator_sha256"
        ],
        "scanner_source_tree_sha256": scanner_sources_start["twen_source_tree_sha256"],
        "scope": "primary+cooldown validation union vs primary+cooldown train union",
        "near_duplicate_threshold": threshold,
        "phase_train_disjointness": phase_identity,
        "phases": identities,
        "identity_reverification": identity_reverification,
        "metrics": metrics,
        "gates": gates,
        "examples": examples,
        "stores_raw_text": False,
        "passed": passed,
    }
    payload["attestation_fingerprint"] = _canonical_sha256(payload)
    attestation = work / "attestation.json"
    try:
        atomic_write_json(attestation, payload)
        atomic_write_json(
            work / "COMPLETE",
            {
                "schema_version": SCHEMA_VERSION,
                "kind": COMPLETE_KIND,
                "attestation": attestation.name,
                "attestation_sha256": sha256_file(attestation),
                "attestation_fingerprint": payload["attestation_fingerprint"],
                "passed": passed,
            },
        )
        os.replace(work, output)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return output / "attestation.json"


def validate_formal_validation_disjointness_attestation(
    path: str | Path,
) -> Mapping[str, object]:
    attestation = Path(path).resolve()
    raw = attestation.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise DataAuditError(f"invalid formal validation attestation: {attestation}") from error
    if not isinstance(value, dict):
        raise DataAuditError("formal validation attestation must be an object")
    scanner_sources = _scanner_sources_identity()
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != KIND
        or value.get("scanner_source_sha256") != scanner_sources["formal_validation_scanner_sha256"]
        or value.get("phase_attestation_validator_source_sha256")
        != scanner_sources["phase_attestation_validator_sha256"]
        or value.get("scanner_source_tree_sha256") != scanner_sources["twen_source_tree_sha256"]
    ):
        raise DataAuditError("unsupported or stale formal validation attestation")
    if value.get("near_duplicate_threshold") != REQUIRED_NEAR_DUPLICATE_THRESHOLD:
        raise DataAuditError("formal validation near-duplicate threshold differs")
    fingerprint = value.get("attestation_fingerprint")
    unsigned = dict(value)
    unsigned.pop("attestation_fingerprint", None)
    if fingerprint != _canonical_sha256(unsigned):
        raise DataAuditError("formal validation attestation fingerprint mismatch")
    phases = value.get("phases")
    if not isinstance(phases, Mapping) or set(phases) != set(PHASES):
        raise DataAuditError("formal validation phase identities are invalid")
    phase_module = _load_phase_module()
    current_identities: dict[str, dict[str, object]] = {}
    for phase in PHASES:
        identity = phases[phase]
        if not isinstance(identity, Mapping):
            raise DataAuditError(f"formal validation {phase} identity is invalid")
        train_prepared = identity.get("prepared")
        validation_prepared = identity.get("validation_prepared")
        if not isinstance(train_prepared, Mapping) or not isinstance(
            validation_prepared,
            Mapping,
        ):
            raise DataAuditError(f"formal validation {phase} prepared identity is invalid")
        _, current_identity = _attested_phase(
            phase=phase,
            manifest_path=str(identity.get("manifest_path")),
            audit_path=str(identity.get("audit_attestation_path")),
            train_prepared_path=str(train_prepared.get("manifest_path")),
            validation_prepared_path=str(validation_prepared.get("manifest_path")),
            phase_module=phase_module,
        )
        if current_identity != identity:
            raise DataAuditError(f"formal validation {phase} input identity changed")
        current_identities[phase] = current_identity
    phase_identity = value.get("phase_train_disjointness")
    if not isinstance(phase_identity, Mapping):
        raise DataAuditError("formal validation phase-train evidence is missing")
    phase_path = Path(str(phase_identity.get("path"))).resolve()
    current_phase_identity = _validate_phase_attestation(
        phase_path,
        identities=current_identities,
        phase_module=phase_module,
    )
    if current_phase_identity != phase_identity:
        raise DataAuditError("formal validation phase-train evidence changed")
    expected_reverification = _assert_end_identity_matches_start(
        scanner_sources_start=scanner_sources,
        scanner_sources_end=_scanner_sources_identity(),
        phases_start=current_identities,
        phases_end=current_identities,
        phase_attestation_start=current_phase_identity,
        phase_attestation_end=current_phase_identity,
    )
    if value.get("identity_reverification") != expected_reverification:
        raise DataAuditError("formal validation identity reverification is invalid")
    gates = value.get("gates")
    metrics = value.get("metrics")
    required_gates = {
        "train_validation_stable_id": (
            STABLE_ID_ALGORITHM,
            "train_validation_stable_id_matches",
        ),
        "train_validation_normalized_exact": (
            NORMALIZED_EXACT_ALGORITHM,
            "train_validation_normalized_exact_matches",
        ),
        "train_validation_near_duplicate": (
            NEAR_DUPLICATE_ALGORITHM,
            "train_validation_near_duplicate_matches",
        ),
        "validation_internal_stable_id": (
            STABLE_ID_ALGORITHM,
            "validation_internal_stable_id_matches",
        ),
        "validation_internal_normalized_exact": (
            NORMALIZED_EXACT_ALGORITHM,
            "validation_internal_normalized_exact_matches",
        ),
        "validation_internal_near_duplicate": (
            NEAR_DUPLICATE_ALGORITHM,
            "validation_internal_near_duplicate_matches",
        ),
    }
    if (
        value.get("scope") != "primary+cooldown validation union vs primary+cooldown train union"
        or value.get("stores_raw_text") is not False
        or not isinstance(gates, Mapping)
        or set(gates) != set(required_gates)
        or not isinstance(metrics, Mapping)
    ):
        raise DataAuditError("formal validation gate/metrics/scope contract differs")
    computed = True
    for name, (algorithm, metric_name) in required_gates.items():
        gate = gates[name]
        metric_count = metrics.get(metric_name)
        if (
            not isinstance(gate, Mapping)
            or gate.get("algorithm") != algorithm
            or isinstance(gate.get("matches"), bool)
            or not isinstance(gate.get("matches"), int)
            or int(gate["matches"]) < 0
            or isinstance(metric_count, bool)
            or not isinstance(metric_count, int)
            or metric_count < 0
            or metric_count != gate.get("matches")
            or gate.get("passed") is not (gate.get("matches") == 0)
            or (
                algorithm == NEAR_DUPLICATE_ALGORITHM
                and gate.get("estimated_jaccard_threshold") != REQUIRED_NEAR_DUPLICATE_THRESHOLD
            )
        ):
            raise DataAuditError(f"formal validation {name} gate semantics differ")
        computed = computed and gate.get("passed") is True
    if value.get("passed") is not computed:
        raise DataAuditError("formal validation passed status differs from gates")
    complete_path = attestation.parent / "COMPLETE"
    try:
        complete_raw = complete_path.read_bytes()
        complete = json.loads(complete_raw)
    except (OSError, json.JSONDecodeError) as error:
        raise DataAuditError("formal validation attestation has no valid COMPLETE") from error
    if (
        not isinstance(complete, Mapping)
        or complete.get("schema_version") != SCHEMA_VERSION
        or complete.get("kind") != COMPLETE_KIND
        or complete.get("attestation") != attestation.name
        or complete.get("attestation_sha256") != hashlib.sha256(raw).hexdigest()
        or complete.get("attestation_fingerprint") != fingerprint
        or complete.get("passed") is not computed
    ):
        raise DataAuditError("formal validation COMPLETE mismatch")
    if attestation.read_bytes() != raw or complete_path.read_bytes() != complete_raw:
        raise DataAuditError("formal validation attestation changed during validation")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-manifest", required=True)
    parser.add_argument("--primary-audit", required=True)
    parser.add_argument("--primary-train-prepared", required=True)
    parser.add_argument("--primary-validation-prepared", required=True)
    parser.add_argument("--cooldown-manifest", required=True)
    parser.add_argument("--cooldown-audit", required=True)
    parser.add_argument("--cooldown-train-prepared", required=True)
    parser.add_argument("--cooldown-validation-prepared", required=True)
    parser.add_argument("--phase-disjointness-attestation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=REQUIRED_NEAR_DUPLICATE_THRESHOLD,
    )
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    try:
        output = build_formal_validation_disjointness_attestation(
            primary_manifest=args.primary_manifest,
            primary_audit=args.primary_audit,
            primary_train_prepared=args.primary_train_prepared,
            primary_validation_prepared=args.primary_validation_prepared,
            cooldown_manifest=args.cooldown_manifest,
            cooldown_audit=args.cooldown_audit,
            cooldown_train_prepared=args.cooldown_train_prepared,
            cooldown_validation_prepared=args.cooldown_validation_prepared,
            phase_disjointness_attestation=args.phase_disjointness_attestation,
            output_root=args.output,
            threshold=args.near_duplicate_threshold,
            max_examples=args.max_examples,
            progress_every=args.progress_every,
            progress=args.progress,
        )
        value = validate_formal_validation_disjointness_attestation(output)
    except (DataAuditError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ok": value["passed"] is True,
                "attestation": str(output),
                "sha256": sha256_file(output),
                "attestation_fingerprint": value["attestation_fingerprint"],
                "passed": value["passed"],
                "gates": value["gates"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if value["passed"] is not True:
        print("error: formal validation disjointness gates failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
