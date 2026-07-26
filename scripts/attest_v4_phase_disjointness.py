#!/usr/bin/env python3
"""Authenticate exact and near-disjointness between two Base train phases.

The scanner is deliberately read-only with respect to its input corpora.  It
first authenticates both extracted-corpus manifests and their successful
``audit-base`` attestations, then compares only the authenticated ``train``
inventories.  The result is committed as a new atomic evidence directory.

No text is copied into the attestation.  A bounded set of matches records only
source/path/line identities and content hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from twen.data.audits import (
    DataAuditError,
    _band_keys,
    _iter_jsonl_documents,
    _lexical_tokens,
    _load_corpus,
    _one_permutation_signature,
    _packed_signature,
    _signature_similarity,
    _unpacked_signature,
    validate_base_audit_attestation,
)
from twen.data.cursor import AuthenticatedSourceMap
from twen.data.prepared import validate_prepared_corpus
from twen.io.download import sha256_file
from twen.source_identity import twen_source_tree_sha256
from twen.utils import atomic_write_json

SCHEMA_VERSION = 1
KIND = "twen_v4_phase_disjointness_attestation"
NEAR_DUPLICATE_ALGORITHM = "lexical-5gram-one-permutation-minhash-lsh-v1"
NORMALIZED_EXACT_ALGORITHM = "unicode-nfkc-whitespace-sha256-intersection-v1"
STABLE_ID_ALGORITHM = "source-scoped-authenticated-stable-id-intersection-v1"
REQUIRED_NEAR_DUPLICATE_THRESHOLD = 0.8


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _normalized_text_sha256(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", " ".join(text.split()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _locator(
    *,
    source_id: str,
    path: str,
    line: int,
    digest_field: str,
    digest: str,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "path": path,
        "line": line,
        digest_field: digest,
    }


def _safe_manifest_inventory(
    manifest_path: Path,
    value: Mapping[str, object],
    field: str,
) -> tuple[Path, ...]:
    raw_inventory = value.get(field)
    if not isinstance(raw_inventory, list):
        raise DataAuditError(f"extracted manifest has no {field}")
    result: list[Path] = []
    for index, raw_entry in enumerate(raw_inventory):
        if not isinstance(raw_entry, Mapping):
            raise DataAuditError(f"{field}[{index}] must be an object")
        relative = raw_entry.get("path")
        if not isinstance(relative, str) or not relative:
            raise DataAuditError(f"{field}[{index}].path is invalid")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise DataAuditError(f"{field}[{index}].path is unsafe")
        result.append(manifest_path.parent / relative_path)
    return tuple(result)


def _iter_train_stable_ids(
    corpus: Any,
) -> Iterator[tuple[str, str, str, int, int]]:
    paths = _safe_manifest_inventory(
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
                if not isinstance(value, Mapping) or value.get("split") != "train":
                    continue
                source_id = value.get("source_id")
                stable_id = value.get("stable_id")
                token_count = value.get("token_count_with_eos")
                if (
                    not isinstance(source_id, str)
                    or not source_id
                    or not isinstance(stable_id, str)
                    or len(stable_id) != 64
                    or isinstance(token_count, bool)
                    or not isinstance(token_count, int)
                    or token_count <= 0
                ):
                    raise DataAuditError(f"invalid train attribution at {path}:{line_number}")
                yield source_id, stable_id, relative, line_number, token_count


def _attested_corpus(
    manifest_path: str | Path,
    audit_path: str | Path,
    prepared_path: str | Path,
) -> tuple[Any, Mapping[str, object], dict[str, object]]:
    corpus = _load_corpus(manifest_path)
    audit_file = Path(audit_path).resolve()
    audit = validate_base_audit_attestation(audit_file)
    if audit.get("ready_for_training") is not True:
        raise DataAuditError(f"audit is not ready_for_training: {audit_file}")
    candidate = audit.get("candidate")
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("role") != "train"
        or candidate.get("manifest_sha256") != corpus.manifest_sha256
        or Path(str(candidate.get("manifest_path"))).resolve() != corpus.manifest_path
    ):
        raise DataAuditError(f"audit candidate does not bind corpus manifest: {audit_file}")
    prepared_file = Path(prepared_path).resolve()
    prepared = validate_prepared_corpus(prepared_file)
    lineage = getattr(prepared, "lineage", None)
    if (
        not isinstance(lineage, Mapping)
        or lineage.get("ready_for_training") is not True
        or lineage.get("research_only") is True
        or lineage.get("pending_audits")
    ):
        raise DataAuditError(f"prepared corpus is not fully training-ready: {prepared_file}")
    try:
        source_map = AuthenticatedSourceMap.from_prepared_manifest(prepared)
    except (TypeError, ValueError, OSError) as error:
        raise DataAuditError(
            f"prepared corpus has no authenticated train source-map: {prepared_file}"
        ) from error
    if source_map.extracted_manifest_sha256 != corpus.manifest_sha256:
        raise DataAuditError(
            "prepared source-map extracted-manifest identity differs from the "
            f"audited corpus: {prepared_file}"
        )
    return (
        corpus,
        audit,
        {
            "manifest_path": str(corpus.manifest_path),
            "manifest_sha256": corpus.manifest_sha256,
            "corpus_fingerprint": corpus.corpus_fingerprint,
            "audit_attestation_path": str(audit_file),
            "audit_attestation_sha256": sha256_file(audit_file),
            "audit_attestation_fingerprint": audit["attestation_fingerprint"],
            "prepared": {
                "manifest_path": str(prepared_file),
                "manifest_sha256": sha256_file(prepared_file),
                "dataset_fingerprint": prepared.dataset_fingerprint,
                "source_map_sha256": source_map.fingerprint,
            },
        },
    )


class _PhaseIndex:
    """Disk-backed primary index used for bounded-memory cross-phase scans."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE stable_ids (
                source_id TEXT NOT NULL,
                stable_id TEXT NOT NULL,
                path TEXT NOT NULL,
                line_number INTEGER NOT NULL,
                PRIMARY KEY(source_id, stable_id)
            );
            CREATE TABLE docs (
                id INTEGER PRIMARY KEY,
                normalized_sha TEXT NOT NULL,
                source_id TEXT NOT NULL,
                path TEXT NOT NULL,
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
        source_id: str,
        stable_id: str,
        path: str,
        line_number: int,
    ) -> None:
        try:
            self.connection.execute(
                "INSERT INTO stable_ids(source_id,stable_id,path,line_number) VALUES(?,?,?,?)",
                (source_id, stable_id, path, line_number),
            )
        except sqlite3.IntegrityError as error:
            raise DataAuditError(
                "primary train attribution contains a duplicate source-scoped "
                f"stable ID: {source_id}/{stable_id}"
            ) from error

    def match_stable_id(
        self,
        source_id: str,
        stable_id: str,
    ) -> tuple[str, int] | None:
        row = self.connection.execute(
            "SELECT path,line_number FROM stable_ids WHERE source_id=? AND stable_id=?",
            (source_id, stable_id),
        ).fetchone()
        return None if row is None else (str(row[0]), int(row[1]))

    def add_document(
        self,
        *,
        source_id: str,
        path: str,
        line_number: int,
        normalized_sha: str,
        signature: Sequence[int],
    ) -> None:
        packed = _packed_signature(signature)
        cursor = self.connection.execute(
            "INSERT INTO docs(normalized_sha,source_id,path,line_number,signature) "
            "VALUES(?,?,?,?,?)",
            (normalized_sha, source_id, path, line_number, packed),
        )
        doc_id = int(cursor.lastrowid)
        self.connection.executemany(
            "INSERT INTO bands(band_key,doc_id) VALUES(?,?)",
            ((key, doc_id) for key in _band_keys(signature)),
        )

    def match_exact(
        self,
        normalized_sha: str,
    ) -> tuple[str, str, int] | None:
        row = self.connection.execute(
            "SELECT source_id,path,line_number FROM docs WHERE normalized_sha=? LIMIT 1",
            (normalized_sha,),
        ).fetchone()
        return None if row is None else (str(row[0]), str(row[1]), int(row[2]))

    def match_near(
        self,
        *,
        normalized_sha: str,
        signature: Sequence[int],
        threshold: float,
    ) -> tuple[str, str, int, str, float] | None:
        candidate_ids: set[int] = set()
        for key in _band_keys(signature):
            candidate_ids.update(
                int(row[0])
                for row in self.connection.execute(
                    "SELECT doc_id FROM bands WHERE band_key=?",
                    (key,),
                )
            )
        best: tuple[str, str, int, str, float] | None = None
        for doc_id in candidate_ids:
            row = self.connection.execute(
                "SELECT normalized_sha,source_id,path,line_number,signature FROM docs WHERE id=?",
                (doc_id,),
            ).fetchone()
            if row is None or row[0] == normalized_sha:
                continue
            similarity = _signature_similarity(
                signature,
                _unpacked_signature(row[4]),
            )
            if similarity < threshold:
                continue
            candidate = (
                str(row[1]),
                str(row[2]),
                int(row[3]),
                str(row[0]),
                similarity,
            )
            if best is None or candidate[-1] > best[-1]:
                best = candidate
        return best


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def build_phase_disjointness_attestation(
    *,
    primary_manifest: str | Path,
    primary_audit: str | Path,
    primary_prepared: str | Path,
    cooldown_manifest: str | Path,
    cooldown_audit: str | Path,
    cooldown_prepared: str | Path,
    output_root: str | Path,
    threshold: float = REQUIRED_NEAR_DUPLICATE_THRESHOLD,
    max_examples: int = 20,
    progress_every: int = 10_000,
    progress: bool = False,
) -> Path:
    if threshold != REQUIRED_NEAR_DUPLICATE_THRESHOLD:
        raise DataAuditError(
            "formal phase-disjointness requires near-duplicate threshold "
            f"{REQUIRED_NEAR_DUPLICATE_THRESHOLD}"
        )
    if max_examples < 0:
        raise DataAuditError("max_examples must be non-negative")
    if progress_every <= 0:
        raise DataAuditError("progress_every must be positive")

    scanner_path = Path(__file__).resolve()
    scanner_source_sha256 = sha256_file(scanner_path)
    scanner_source_tree_sha256 = twen_source_tree_sha256()
    primary, _, primary_identity = _attested_corpus(
        primary_manifest,
        primary_audit,
        primary_prepared,
    )
    cooldown, _, cooldown_identity = _attested_corpus(
        cooldown_manifest,
        cooldown_audit,
        cooldown_prepared,
    )
    output = Path(output_root).resolve()
    if output.exists():
        raise DataAuditError(
            f"phase-disjointness output already exists; choose a new directory: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.tmp-",
            dir=output.parent,
        )
    )
    database = work / "phase-index.sqlite3"
    index = _PhaseIndex(database)
    metrics: dict[str, int] = {
        "primary_train_documents": 0,
        "primary_train_attribution_rows": 0,
        "primary_train_attributed_tokens": 0,
        "cooldown_train_documents": 0,
        "cooldown_train_attribution_rows": 0,
        "cooldown_train_attributed_tokens": 0,
        "stable_id_exact_matches": 0,
        "normalized_text_exact_matches": 0,
        "near_duplicate_matches": 0,
    }
    examples: dict[str, list[dict[str, object]]] = {
        "stable_id_exact": [],
        "normalized_text_exact": [],
        "near_duplicate": [],
    }
    scan_completed = False
    try:
        for source_id, stable_id, path, line_number, token_count in _iter_train_stable_ids(primary):
            index.add_stable_id(source_id, stable_id, path, line_number)
            metrics["primary_train_attribution_rows"] += 1
            metrics["primary_train_attributed_tokens"] += token_count
            if metrics["primary_train_attribution_rows"] % progress_every == 0:
                index.commit()
                _progress(
                    progress,
                    f"primary attribution {metrics['primary_train_attribution_rows']:,} rows",
                )
        index.commit()

        for path, source_id, _category, line_number, text in _iter_jsonl_documents(
            primary, "train"
        ):
            normalized_sha = _normalized_text_sha256(text)
            signature = _one_permutation_signature(
                _lexical_tokens(text),
                normalized_sha,
            )
            index.add_document(
                source_id=source_id,
                path=path,
                line_number=line_number,
                normalized_sha=normalized_sha,
                signature=signature,
            )
            metrics["primary_train_documents"] += 1
            if metrics["primary_train_documents"] % progress_every == 0:
                index.commit()
                _progress(
                    progress,
                    f"primary text {metrics['primary_train_documents']:,} documents",
                )
        index.commit()

        for source_id, stable_id, path, line_number, token_count in _iter_train_stable_ids(
            cooldown
        ):
            match = index.match_stable_id(source_id, stable_id)
            metrics["cooldown_train_attribution_rows"] += 1
            metrics["cooldown_train_attributed_tokens"] += token_count
            if match is not None:
                metrics["stable_id_exact_matches"] += 1
                if len(examples["stable_id_exact"]) < max_examples:
                    examples["stable_id_exact"].append(
                        {
                            "primary": _locator(
                                source_id=source_id,
                                path=match[0],
                                line=match[1],
                                digest_field="stable_id",
                                digest=stable_id,
                            ),
                            "cooldown": _locator(
                                source_id=source_id,
                                path=path,
                                line=line_number,
                                digest_field="stable_id",
                                digest=stable_id,
                            ),
                        }
                    )
            if metrics["cooldown_train_attribution_rows"] % progress_every == 0:
                _progress(
                    progress,
                    f"cooldown attribution {metrics['cooldown_train_attribution_rows']:,} rows",
                )

        for path, source_id, _category, line_number, text in _iter_jsonl_documents(
            cooldown, "train"
        ):
            normalized_sha = _normalized_text_sha256(text)
            signature = _one_permutation_signature(
                _lexical_tokens(text),
                normalized_sha,
            )
            exact = index.match_exact(normalized_sha)
            if exact is not None:
                metrics["normalized_text_exact_matches"] += 1
                if len(examples["normalized_text_exact"]) < max_examples:
                    examples["normalized_text_exact"].append(
                        {
                            "primary": _locator(
                                source_id=exact[0],
                                path=exact[1],
                                line=exact[2],
                                digest_field="normalized_text_sha256",
                                digest=normalized_sha,
                            ),
                            "cooldown": _locator(
                                source_id=source_id,
                                path=path,
                                line=line_number,
                                digest_field="normalized_text_sha256",
                                digest=normalized_sha,
                            ),
                        }
                    )
            near = index.match_near(
                normalized_sha=normalized_sha,
                signature=signature,
                threshold=threshold,
            )
            if near is not None:
                metrics["near_duplicate_matches"] += 1
                if len(examples["near_duplicate"]) < max_examples:
                    examples["near_duplicate"].append(
                        {
                            "primary": _locator(
                                source_id=near[0],
                                path=near[1],
                                line=near[2],
                                digest_field="normalized_text_sha256",
                                digest=near[3],
                            ),
                            "cooldown": _locator(
                                source_id=source_id,
                                path=path,
                                line=line_number,
                                digest_field="normalized_text_sha256",
                                digest=normalized_sha,
                            ),
                            "estimated_jaccard": near[4],
                        }
                    )
            metrics["cooldown_train_documents"] += 1
            if metrics["cooldown_train_documents"] % progress_every == 0:
                _progress(
                    progress,
                    f"cooldown text {metrics['cooldown_train_documents']:,} documents",
                )
        scan_completed = True
    finally:
        index.close()
        if not scan_completed:
            shutil.rmtree(work, ignore_errors=True)

    try:
        _, _, final_primary_identity = _attested_corpus(
            primary_manifest,
            primary_audit,
            primary_prepared,
        )
        _, _, final_cooldown_identity = _attested_corpus(
            cooldown_manifest,
            cooldown_audit,
            cooldown_prepared,
        )
        if (
            final_primary_identity != primary_identity
            or final_cooldown_identity != cooldown_identity
        ):
            raise DataAuditError("phase-disjointness input identity changed during the scan")
        if (
            sha256_file(scanner_path) != scanner_source_sha256
            or twen_source_tree_sha256() != scanner_source_tree_sha256
        ):
            raise DataAuditError("phase-disjointness scanner source changed during the scan")
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise

    database.unlink(missing_ok=True)
    Path(str(database) + "-wal").unlink(missing_ok=True)
    Path(str(database) + "-shm").unlink(missing_ok=True)
    gates = {
        "stable_id_exact": {
            "algorithm": STABLE_ID_ALGORITHM,
            "matches": metrics["stable_id_exact_matches"],
            "passed": metrics["stable_id_exact_matches"] == 0,
        },
        "normalized_text_exact": {
            "algorithm": NORMALIZED_EXACT_ALGORITHM,
            "matches": metrics["normalized_text_exact_matches"],
            "passed": metrics["normalized_text_exact_matches"] == 0,
        },
        "near_duplicate": {
            "algorithm": NEAR_DUPLICATE_ALGORITHM,
            "estimated_jaccard_threshold": threshold,
            "matches": metrics["near_duplicate_matches"],
            "passed": metrics["near_duplicate_matches"] == 0,
        },
    }
    passed = all(bool(gate["passed"]) for gate in gates.values())
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "scanner_source_sha256": scanner_source_sha256,
        "scanner_source_tree_sha256": scanner_source_tree_sha256,
        "primary": primary_identity,
        "cooldown": cooldown_identity,
        "scope": "authenticated_train_inventories_only",
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
                "kind": "twen_v4_phase_disjointness_complete",
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-manifest", required=True)
    parser.add_argument("--primary-audit", required=True)
    parser.add_argument("--primary-prepared", required=True)
    parser.add_argument("--cooldown-manifest", required=True)
    parser.add_argument("--cooldown-audit", required=True)
    parser.add_argument("--cooldown-prepared", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=REQUIRED_NEAR_DUPLICATE_THRESHOLD,
    )
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    output = build_phase_disjointness_attestation(
        primary_manifest=args.primary_manifest,
        primary_audit=args.primary_audit,
        primary_prepared=args.primary_prepared,
        cooldown_manifest=args.cooldown_manifest,
        cooldown_audit=args.cooldown_audit,
        cooldown_prepared=args.cooldown_prepared,
        output_root=args.output,
        threshold=args.near_duplicate_threshold,
        max_examples=args.max_examples,
        progress_every=args.progress_every,
        progress=args.progress,
    )
    value = json.loads(output.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "ok": bool(value["passed"]),
                "attestation": str(output),
                "sha256": sha256_file(output),
                "attestation_fingerprint": value["attestation_fingerprint"],
                "passed": value["passed"],
                "gates": value["gates"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if value["passed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
