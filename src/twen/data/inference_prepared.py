"""Forward-only authentication for manifest-pinned historical prepared corpora.

This module is deliberately separate from :mod:`twen.data.prepared`.  The
audited preparation generator identity is the SHA256 of that module, so adding
inference compatibility there would invalidate otherwise authentic training
artifacts.  Nothing in this module authorizes optimization, KD generation,
calibration, cooldown materialization, or export.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from ..io.download import sha256_file
from .prepared import (
    AUDITED_PREPARED_GENERATOR_SOURCE_SHA256,
    PREPARED_GENERATOR_SOURCE_SHA256,
    PREPARED_SHARD_MANIFEST,
    PREPARED_TENSORS,
    PreparedCorpusManifest,
    _authenticate_extracted_prepare_inputs,
    _canonical_sha256,
    _local_prepared_manifest,
    _normalized_prepared_lineage,
    _normalized_sha256,
    _prepared_dataset_fingerprint,
    _prepared_pipeline_fingerprint,
    _required_file_size,
    _required_string,
    _resolved_extracted_file,
    _safe_extracted_relative_path,
    validate_prepared_corpus,
)
from .shards import is_shard_complete


def _historical_audit_lineage_for_inference(
    pinned: Mapping[str, object],
    *,
    extracted_manifest_path: Path,
    extracted_manifest_sha256: str,
    corpus_fingerprint: str,
    role: str,
) -> dict[str, object]:
    """Authenticate historical audit evidence without blessing its old code."""

    attestation_path = Path(
        _required_string(pinned.get("path"), "historical audit path")
    ).resolve()
    expected_attestation_sha = _normalized_sha256(
        _required_string(pinned.get("sha256"), "historical audit sha256"),
        "historical audit sha256",
    )
    try:
        raw = attestation_path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read historical audit attestation: {attestation_path}"
        ) from error
    if hashlib.sha256(raw).hexdigest() != expected_attestation_sha:
        raise ValueError("historical audit attestation differs from prepared lineage")
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("kind") != "twen_base_corpus_audit_attestation"
    ):
        raise ValueError("unsupported historical audit attestation")
    _normalized_sha256(
        _required_string(value.get("audit_source_sha256"), "historical audit source"),
        "historical audit source",
    )
    fingerprint = _normalized_sha256(
        _required_string(
            value.get("attestation_fingerprint"),
            "historical audit attestation_fingerprint",
        ),
        "historical audit attestation_fingerprint",
    )
    unsigned = dict(value)
    unsigned.pop("attestation_fingerprint", None)
    if _canonical_sha256(unsigned) != fingerprint:
        raise ValueError("historical audit attestation fingerprint mismatch")

    identity_name = "candidate" if role == "train" else "frozen_validation"
    identity = value.get(identity_name)
    if not isinstance(identity, Mapping):
        raise ValueError("historical audit corpus identity is missing")
    if (
        Path(str(identity.get("manifest_path"))).resolve()
        != extracted_manifest_path
        or identity.get("manifest_sha256") != extracted_manifest_sha256
        or identity.get("corpus_fingerprint") != corpus_fingerprint
        or identity.get("role") != role
    ):
        raise ValueError("historical audit does not bind the prepared extracted role")

    gates = value.get("gates")
    if not isinstance(gates, Mapping) or not gates:
        raise ValueError("historical audit gates are missing")
    computed_ready = all(
        isinstance(gate, Mapping) and gate.get("passed") is True
        for gate in gates.values()
    )
    if value.get("ready_for_training") is not computed_ready:
        raise ValueError("historical audit readiness differs from its gates")

    registry = value.get("benchmark_registry")
    if not isinstance(registry, Mapping):
        raise ValueError("historical audit benchmark registry identity is missing")
    registry_path = Path(
        _required_string(registry.get("path"), "historical benchmark registry path")
    ).resolve()
    registry_sha = _normalized_sha256(
        _required_string(
            registry.get("sha256"),
            "historical benchmark registry sha256",
        ),
        "historical benchmark registry sha256",
    )
    if sha256_file(registry_path) != registry_sha:
        raise ValueError("historical audit benchmark registry changed")

    for field in ("findings", "rejection_ledger"):
        sidecar_identity = value.get(field)
        if not isinstance(sidecar_identity, Mapping):
            raise ValueError(f"historical audit {field} identity is missing")
        relative = _safe_extracted_relative_path(
            sidecar_identity.get("path"),
            f"historical audit {field}.path",
        )
        sidecar = _resolved_extracted_file(
            attestation_path.parent,
            relative,
            f"historical audit {field}",
        )
        expected_size = _required_file_size(
            sidecar_identity.get("size"),
            f"historical audit {field}.size",
        )
        expected_sha = _normalized_sha256(
            _required_string(
                sidecar_identity.get("sha256"),
                f"historical audit {field}.sha256",
            ),
            f"historical audit {field}.sha256",
        )
        if (
            not sidecar.is_file()
            or sidecar.stat().st_size != expected_size
            or sha256_file(sidecar) != expected_sha
        ):
            raise ValueError(f"historical audit {field} changed")
    rejection = value["rejection_ledger"]
    assert isinstance(rejection, Mapping)
    if (
        rejection.get("complete") is not True
        or rejection.get("stores_raw_text") is not False
    ):
        raise ValueError("historical audit rejection ledger contract differs")

    complete_path = attestation_path.parent / "COMPLETE"
    try:
        complete_raw = complete_path.read_bytes()
        complete = json.loads(complete_raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("historical audit has no valid COMPLETE marker") from error
    if (
        not isinstance(complete, Mapping)
        or complete.get("schema_version") != 1
        or complete.get("kind") != "twen_base_corpus_audit_complete"
        or complete.get("attestation") != attestation_path.name
        or complete.get("attestation_sha256") != expected_attestation_sha
        or complete.get("attestation_fingerprint") != fingerprint
        or complete.get("ready_for_training") is not computed_ready
    ):
        raise ValueError("historical audit COMPLETE metadata mismatch")
    if attestation_path.read_bytes() != raw or complete_path.read_bytes() != complete_raw:
        raise ValueError("historical audit evidence changed during validation")

    result = {
        "path": str(attestation_path),
        "sha256": expected_attestation_sha,
        "attestation_fingerprint": fingerprint,
        "bound_as": identity_name,
        "gates": json.loads(json.dumps(gates, sort_keys=True)),
        "ready_for_training": computed_ready,
    }
    if result != pinned:
        raise ValueError("historical audit lineage snapshot differs from its evidence")
    return result


def _authenticate_historical_prepared_lineage(
    corpus: PreparedCorpusManifest,
) -> tuple[tuple[Path, str], ...]:
    lineage = corpus.lineage
    if (
        not isinstance(lineage, Mapping)
        or lineage.get("kind") != "authenticated_extracted_corpus"
    ):
        raise ValueError(
            "historical prepared inference requires authenticated extracted-corpus lineage"
        )
    extracted_manifest_path = Path(
        _required_string(
            lineage.get("extracted_manifest_path"),
            "prepared extracted_manifest_path",
        )
    ).resolve()
    authenticated_sources, base_lineage = _authenticate_extracted_prepare_inputs(
        extracted_manifest_path,
        role=_required_string(lineage.get("role"), "prepared role"),
        tokenizer_sha256=corpus.tokenizer_sha256,
        allow_pending_research_audits=True,
        audit_attestation=None,
    )
    pinned_audit = lineage.get("audit_attestation")
    if not isinstance(pinned_audit, Mapping):
        raise ValueError(
            "historical prepared inference requires a pinned audit attestation"
        )
    audit_lineage = _historical_audit_lineage_for_inference(
        pinned_audit,
        extracted_manifest_path=extracted_manifest_path,
        extracted_manifest_sha256=_required_string(
            base_lineage.get("extracted_manifest_sha256"),
            "extracted manifest sha256",
        ),
        corpus_fingerprint=_required_string(
            base_lineage.get("corpus_fingerprint"),
            "extracted corpus fingerprint",
        ),
        role=_required_string(lineage.get("role"), "prepared role"),
    )
    gates = audit_lineage["gates"]
    assert isinstance(gates, dict)
    merged_audits = dict(base_lineage["audits"])
    for name, gate in gates.items():
        if (
            not isinstance(name, str)
            or not isinstance(gate, dict)
            or not isinstance(gate.get("status"), str)
        ):
            raise ValueError("historical audit gate status is invalid")
        merged_audits[name] = gate["status"]
    expected_lineage = dict(base_lineage)
    expected_lineage.update(
        {
            "audits": merged_audits,
            "pending_audits": sorted(
                name
                for name, status in merged_audits.items()
                if status.lower().startswith("pending")
            ),
            "ready_for_training": audit_lineage["ready_for_training"],
            "research_only": not bool(audit_lineage["ready_for_training"]),
            "audit_attestation": audit_lineage,
        }
    )
    if _normalized_prepared_lineage(expected_lineage) != corpus.lineage:
        raise ValueError(
            "historical prepared extracted-corpus lineage no longer authenticates"
        )
    return authenticated_sources


def _validate_historical_prepared_corpus(
    corpus: PreparedCorpusManifest,
    *,
    manifest_path: Path,
) -> PreparedCorpusManifest:
    root = manifest_path.parent.resolve()
    if (
        not isinstance(corpus.lineage, Mapping)
        or corpus.lineage.get("kind") != "authenticated_extracted_corpus"
    ):
        raise ValueError(
            "historical prepared inference requires authenticated extracted-corpus lineage"
        )
    is_quality_cooldown = isinstance(corpus.lineage.get("quality_cooldown"), Mapping)
    if not is_quality_cooldown:
        authenticated_sources = _authenticate_historical_prepared_lineage(corpus)
        prepared_sources = tuple(
            (Path(entry.source_path).resolve(), entry.source_sha256)
            for entry in corpus.shards
        )
        if prepared_sources != authenticated_sources:
            raise ValueError(
                "prepared shards do not exactly match authenticated extracted role inventory"
            )

    expected_pipeline = _prepared_pipeline_fingerprint(
        [(Path(entry.source_path), entry.source_sha256) for entry in corpus.shards],
        tokenizer_sha256=corpus.tokenizer_sha256,
        sequence_length=corpus.sequence_length,
        text_field=corpus.text_field,
        generator_source_sha256=corpus.generator_source_sha256,
        lineage=corpus.lineage,
    )
    if expected_pipeline != corpus.pipeline_fingerprint:
        raise ValueError("prepared pipeline fingerprint does not match manifest inputs")
    expected_dataset = _prepared_dataset_fingerprint(
        pipeline_fingerprint=corpus.pipeline_fingerprint,
        generator_source_sha256=corpus.generator_source_sha256,
        tokenizer_sha256=corpus.tokenizer_sha256,
        sequence_length=corpus.sequence_length,
        text_field=corpus.text_field,
        shards=corpus.shards,
        lineage=corpus.lineage,
    )
    if expected_dataset != corpus.dataset_fingerprint:
        raise ValueError("prepared dataset fingerprint does not match manifest contents")
    for entry in corpus.shards:
        directory = (root / entry.path).resolve()
        try:
            directory.relative_to(root)
        except ValueError as error:
            raise ValueError(f"prepared shard escapes root: {entry.path}") from error
        if directory == root:
            raise ValueError(f"prepared shard escapes root: {entry.path}")
        if not is_shard_complete(directory):
            raise ValueError(f"prepared shard is incomplete: {entry.path}")
        if sha256_file(directory / PREPARED_TENSORS) != entry.tensors_sha256:
            raise ValueError(f"prepared tensor hash mismatch: {entry.path}")
        local_path = directory / PREPARED_SHARD_MANIFEST
        try:
            local = json.loads(local_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid local prepared manifest: {entry.path}") from error
        expected_local = _local_prepared_manifest(
            shard_id=entry.shard_id,
            source_path=entry.source_path,
            source_sha256=entry.source_sha256,
            sequence_count=entry.sequence_count,
            token_count=entry.token_count,
            tensors_sha256=entry.tensors_sha256,
            pipeline_fingerprint=corpus.pipeline_fingerprint,
            generator_source_sha256=corpus.generator_source_sha256,
            tokenizer_sha256=corpus.tokenizer_sha256,
            sequence_length=corpus.sequence_length,
            text_field=corpus.text_field,
        )
        if local != expected_local:
            raise ValueError(
                f"local prepared manifest differs from corpus entry: {entry.path}"
            )
    return corpus


def validate_prepared_corpus_for_inference(
    path: str | Path,
    *,
    expected_manifest_sha256: str,
) -> PreparedCorpusManifest:
    """Authenticate a pinned corpus for forward-only evaluation.

    Current-generator corpora still pass through the unmodified strict
    validator.  A historical generator is accepted only with authenticated
    extracted-corpus lineage after all source, audit, fingerprint, COMPLETE,
    tensor, and local-manifest identities are rechecked.
    """

    manifest_path = Path(path).resolve()
    expected_sha = _normalized_sha256(
        expected_manifest_sha256,
        "expected prepared manifest SHA256",
    )
    try:
        manifest_raw = manifest_path.read_bytes()
        raw_value = json.loads(manifest_raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read prepared corpus manifest: {manifest_path}"
        ) from error
    if not isinstance(raw_value, dict):
        raise ValueError("prepared corpus manifest must be an object")
    if hashlib.sha256(manifest_raw).hexdigest() != expected_sha:
        raise ValueError("inference prepared manifest differs from its pinned SHA256")
    corpus = PreparedCorpusManifest.from_dict(raw_value)
    if corpus.generator_source_sha256 in {
        PREPARED_GENERATOR_SOURCE_SHA256,
        AUDITED_PREPARED_GENERATOR_SOURCE_SHA256,
    }:
        validated = validate_prepared_corpus(manifest_path)
    else:
        validated = _validate_historical_prepared_corpus(
            corpus,
            manifest_path=manifest_path,
        )
    if manifest_path.read_bytes() != manifest_raw:
        raise ValueError("prepared corpus manifest changed during validation")
    return validated
