#!/usr/bin/env python3
# ruff: noqa: RUF001
"""Authenticate and summarize the v4 formal frozen-validation baseline.

This command is reporting-only.  It never loads a model, starts training, or
changes an evaluation input.  It requires two completed, governed prepared
validation corpora (primary and cooldown), completed candidate-role NLL
evaluations from the same v3-final checkpoint, and the authenticated legacy
six-source v3 baseline summary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

SCHEMA_VERSION = 1
KIND = "twen_v4_formal_frozen_validation_baseline"
BUNDLE_KIND = "twen_v4_formal_frozen_validation_baseline_bundle"
TARGET_ROLE = "candidate"
LEGACY_SUMMARY_KIND = "twen_v4_checkpoint_frozen_validation_sweep"
LEGACY_BUNDLE_KIND = "twen_v4_checkpoint_validation_sweep_bundle"
NUMERIC_COMPARISON_FIELDS = (
    "track",
    "stage",
    "expert_initialization",
    "batch_size",
    "device_type",
    "dtype",
)

EXPECTED_PHASE_SOURCES = {
    "primary": frozenset(
        {
            "chinese_wikipedia_zh_20231101",
            "english_fineweb_edu_dedup",
            "math_finemath_4plus",
            "code_github_clean_allowlisted",
            "science_cosmopedia_openstax",
            "science_cosmopedia_stanford",
            "science_arxiv_open_permissive",
            "code_stackv2_edu_permissive",
            "multilingual_common_corpus_permissive",
        }
    ),
    "cooldown": frozenset(
        {
            "chinese_wikipedia_zh_20231101",
            "math_finemath_4plus",
            "science_arxiv_open_permissive",
            "science_cosmopedia_openstax",
            "science_cosmopedia_stanford",
            "education_libretexts_permissive",
            "public_domain_project_gutenberg",
            "code_github_clean_allowlisted",
        }
    ),
}
ADDITIONAL_SOURCES = frozenset(
    {
        "chinese_wikipedia_zh_20231101",
        "science_arxiv_open_permissive",
        "code_stackv2_edu_permissive",
        "multilingual_common_corpus_permissive",
        "education_libretexts_permissive",
        "public_domain_project_gutenberg",
    }
)
LEGACY_SOURCES = frozenset(
    {
        "chinese_fineweb2_cmn_hani",
        "english_fineweb_edu_dedup",
        "math_finemath_4plus",
        "code_github_clean_allowlisted",
        "science_cosmopedia_openstax",
        "science_cosmopedia_stanford",
    }
)
FORMAL_SOURCE_REPLACEMENTS = {
    "chinese_fineweb2_cmn_hani": "chinese_wikipedia_zh_20231101",
}


class FormalValidationError(ValueError):
    """A formal validation input is incomplete or unauthenticated."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-prepared", required=True, type=Path)
    parser.add_argument("--primary-evaluation", required=True, type=Path)
    parser.add_argument("--cooldown-prepared", required=True, type=Path)
    parser.add_argument("--cooldown-evaluation", required=True, type=Path)
    parser.add_argument(
        "--validation-disjointness-attestation",
        required=True,
        type=Path,
        help="passing primary+cooldown train/validation union disjointness evidence",
    )
    parser.add_argument(
        "--legacy-summary",
        required=True,
        type=Path,
        help="authenticated v4 checkpoint-sweep summary containing its v3 baseline",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="replace only an already authenticated bundle",
    )
    return parser


def _load_sweep_module() -> ModuleType:
    path = Path(__file__).with_name("summarize_v4_checkpoint_validation.py")
    source_before = path.read_bytes()
    spec = importlib.util.spec_from_file_location("_twen_v4_checkpoint_sweep", path)
    if spec is None or spec.loader is None:
        raise FormalValidationError(f"cannot load checkpoint summarizer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if path.read_bytes() != source_before:
        raise FormalValidationError(f"checkpoint summarizer changed while loading: {path}")
    return module


def _load_validation_disjointness_module() -> ModuleType:
    path = Path(__file__).with_name("attest_v4_formal_validation_disjointness.py")
    source_before = path.read_bytes()
    spec = importlib.util.spec_from_file_location(
        "_twen_v4_formal_validation_disjointness",
        path,
    )
    if spec is None or spec.loader is None:
        raise FormalValidationError(f"cannot load formal validation disjointness validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if path.read_bytes() != source_before:
        raise FormalValidationError(
            f"formal validation disjointness validator changed while loading: {path}"
        )
    return module


def _reporter_sources_identity(sweep: ModuleType) -> dict[str, str]:
    sweep_path = Path(str(sweep.__file__)).resolve()
    return {
        "formal_validation_reporter_sha256": _sha256_file(Path(__file__).resolve()),
        "checkpoint_summarizer_sha256": _sha256_file(sweep_path),
    }


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalValidationError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FormalValidationError(f"{label} must be a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_artifact_from_plan(
    plan: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    raw_checkpoint = plan.get("checkpoint")
    expected_complete_sha256 = plan.get("checkpoint_complete_sha256")
    if (
        not isinstance(raw_checkpoint, str)
        or not raw_checkpoint
        or not isinstance(expected_complete_sha256, str)
        or len(expected_complete_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in expected_complete_sha256.lower()
        )
    ):
        raise FormalValidationError(f"{label} has no authenticated checkpoint identity")
    checkpoint = Path(raw_checkpoint).resolve()
    complete = checkpoint / "COMPLETE"
    if not complete.is_file() or _sha256_file(complete) != expected_complete_sha256.lower():
        raise FormalValidationError(f"{label} checkpoint COMPLETE identity mismatch")
    return {
        "path": str(checkpoint),
        "complete_path": str(complete),
        "complete_sha256": expected_complete_sha256.lower(),
    }


def _governed_lineage_sources(
    payload: Mapping[str, Any],
    *,
    phase: str,
) -> frozenset[str]:
    lineage = payload.get("lineage")
    if not isinstance(lineage, Mapping):
        raise FormalValidationError(f"{phase} prepared manifest has no lineage")
    if (
        lineage.get("kind") != "authenticated_extracted_corpus"
        or lineage.get("role") != "validation"
        or lineage.get("ready_for_training") is not True
        or lineage.get("research_only") is not False
    ):
        raise FormalValidationError(
            f"{phase} prepared corpus is not a governed, training-ready validation role"
        )
    attestation = lineage.get("audit_attestation")
    if (
        not isinstance(attestation, Mapping)
        or attestation.get("bound_as") != "frozen_validation"
        or attestation.get("ready_for_training") is not True
    ):
        raise FormalValidationError(
            f"{phase} prepared validation has no passing frozen-validation attestation"
        )
    gates = attestation.get("gates")
    if not isinstance(gates, Mapping) or not gates:
        raise FormalValidationError(f"{phase} audit attestation has no gates")
    failed = sorted(
        str(name)
        for name, gate in gates.items()
        if not isinstance(gate, Mapping) or gate.get("passed") is not True
    )
    if failed:
        raise FormalValidationError(
            f"{phase} frozen-validation audit has non-passing gates: {failed}"
        )
    contract = lineage.get("data_contract")
    source_map = contract.get("source_map") if isinstance(contract, Mapping) else None
    roles = source_map.get("roles") if isinstance(source_map, Mapping) else None
    validation = roles.get("validation") if isinstance(roles, Mapping) else None
    if not isinstance(validation, list) or not validation:
        raise FormalValidationError(
            f"{phase} prepared lineage has no authenticated validation source map"
        )
    sources: set[str] = set()
    mapped_identities: set[tuple[str, str]] = set()
    for index, row in enumerate(validation):
        if not isinstance(row, Mapping):
            raise FormalValidationError(f"{phase} source_map.validation[{index}] is not an object")
        source_id = row.get("source_id")
        path = row.get("path")
        digest = row.get("sha256")
        if (
            not isinstance(source_id, str)
            or not source_id
            or not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise FormalValidationError(
                f"{phase} source_map.validation[{index}] identity is invalid"
            )
        sources.add(source_id)
        mapped_identities.add((path, digest))
    expected = EXPECTED_PHASE_SOURCES[phase]
    if sources != expected:
        raise FormalValidationError(
            f"{phase} validation source coverage differs "
            f"(missing={sorted(expected - sources)}, extra={sorted(sources - expected)})"
        )
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise FormalValidationError(f"{phase} prepared manifest has no shards")
    unmatched: list[str] = []
    for row in shards:
        if not isinstance(row, Mapping):
            raise FormalValidationError(f"{phase} prepared shard is not an object")
        source_path = row.get("source_path")
        source_sha = row.get("source_sha256")
        if not isinstance(source_path, str) or not isinstance(source_sha, str):
            raise FormalValidationError(f"{phase} prepared shard source identity is invalid")
        normalized = source_path.replace("\\", "/")
        if not any(
            normalized.endswith(f"/{path}") and source_sha == digest
            for path, digest in mapped_identities
        ):
            unmatched.append(str(row.get("shard_id")))
    if unmatched:
        raise FormalValidationError(
            f"{phase} prepared shards are outside its authenticated validation map: {unmatched}"
        )
    return frozenset(sources)


def _load_governed_prepared(
    path: Path,
    *,
    phase: str,
    sweep: ModuleType,
) -> dict[str, Any]:
    try:
        from twen.data import validate_prepared_corpus

        validate_prepared_corpus(path)
    except (OSError, ValueError) as exc:
        raise FormalValidationError(
            f"{phase} prepared corpus failed full tensor/lineage authentication: {exc}"
        ) from exc
    payload = _read_json(path.resolve(), label=f"{phase} prepared manifest")
    sources = _governed_lineage_sources(payload, phase=phase)
    try:
        prepared = sweep._load_prepared_manifest(path)
    except sweep.SweepError as exc:
        raise FormalValidationError(str(exc)) from exc
    prepared["phase"] = phase
    prepared["sources"] = sorted(sources)
    prepared["lineage_identity"] = {
        "extracted_manifest_path": payload["lineage"]["extracted_manifest_path"],
        "extracted_manifest_sha256": payload["lineage"]["extracted_manifest_sha256"],
        "corpus_fingerprint": payload["lineage"]["corpus_fingerprint"],
        "audit_attestation": payload["lineage"]["audit_attestation"],
    }
    return prepared


def _load_validation_disjointness(
    path: Path,
    *,
    prepared: Mapping[str, Mapping[str, Any]],
    sweep: ModuleType,
) -> dict[str, Any]:
    module = _load_validation_disjointness_module()
    attestation_path = path.resolve()
    identity_before = {
        "sha256": _sha256_file(attestation_path),
        "size": attestation_path.stat().st_size,
    }
    try:
        value = module.validate_formal_validation_disjointness_attestation(attestation_path)
    except (OSError, ValueError) as exc:
        raise FormalValidationError(
            f"formal validation disjointness attestation failed authentication: {exc}"
        ) from exc
    identity_after = {
        "sha256": _sha256_file(attestation_path),
        "size": attestation_path.stat().st_size,
    }
    if identity_after != identity_before:
        raise FormalValidationError(
            "formal validation disjointness attestation changed during authentication"
        )
    if value.get("passed") is not True:
        raise FormalValidationError(
            "formal validation train/validation disjointness gates did not pass"
        )
    if value.get("near_duplicate_threshold") != module.REQUIRED_NEAR_DUPLICATE_THRESHOLD:
        raise FormalValidationError("formal validation disjointness near-duplicate policy differs")
    phases = value.get("phases")
    if not isinstance(phases, Mapping):
        raise FormalValidationError("formal validation disjointness has no phase identities")
    for phase in ("primary", "cooldown"):
        actual = prepared[phase]
        identity = phases.get(phase)
        if not isinstance(identity, Mapping):
            raise FormalValidationError(f"formal validation disjointness has no {phase} identity")
        validation = identity.get("validation_prepared")
        if not isinstance(validation, Mapping):
            raise FormalValidationError(
                f"formal validation disjointness has no {phase} prepared validation"
            )
        if (
            Path(str(validation.get("manifest_path"))).resolve() != Path(actual["path"]).resolve()
            or validation.get("manifest_sha256") != actual["sha256"]
            or validation.get("dataset_fingerprint") != actual["dataset_fingerprint"]
            or validation.get("sequence_count") != actual["sequence_count"]
            or validation.get("token_count") != actual["token_count"]
        ):
            raise FormalValidationError(
                f"formal validation disjointness does not bind {phase} prepared input"
            )
        lineage = actual["lineage_identity"]
        audit = lineage.get("audit_attestation")
        if (
            Path(str(identity.get("manifest_path"))).resolve()
            != Path(str(lineage.get("extracted_manifest_path"))).resolve()
            or identity.get("manifest_sha256") != lineage.get("extracted_manifest_sha256")
            or not isinstance(audit, Mapping)
            or Path(str(identity.get("audit_attestation_path"))).resolve()
            != Path(str(audit.get("path"))).resolve()
            or identity.get("audit_attestation_sha256") != audit.get("sha256")
        ):
            raise FormalValidationError(
                f"formal validation disjointness lineage differs for {phase}"
            )
    gates = value.get("gates")
    reverification = value.get("identity_reverification")
    if not isinstance(gates, Mapping) or not isinstance(
        reverification,
        Mapping,
    ):
        raise FormalValidationError("formal validation disjointness evidence is incomplete")
    return {
        "path": str(attestation_path),
        "sha256": identity_before["sha256"],
        "attestation_fingerprint": value["attestation_fingerprint"],
        "scanner_source_sha256": value["scanner_source_sha256"],
        "phase_attestation_validator_source_sha256": value[
            "phase_attestation_validator_source_sha256"
        ],
        "scanner_source_tree_sha256": value["scanner_source_tree_sha256"],
        "near_duplicate_threshold": value["near_duplicate_threshold"],
        "identity_reverification": json.loads(json.dumps(reverification, sort_keys=True)),
        "gates": json.loads(json.dumps(gates, sort_keys=True)),
        "passed": True,
    }


def _load_legacy_baseline(path: Path, *, sweep: ModuleType) -> dict[str, Any]:
    path = path.resolve()
    root = path.parent
    manifest_path = root / "MANIFEST.json"
    complete_path = root / "COMPLETE"
    if not manifest_path.is_file() or not complete_path.is_file():
        raise FormalValidationError("legacy summary is outside an authenticated report bundle")
    manifest = _read_json(manifest_path, label="legacy bundle manifest")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != LEGACY_BUNDLE_KIND
    ):
        raise FormalValidationError("legacy bundle kind/schema is unsupported")
    try:
        expected = complete_path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise FormalValidationError(f"cannot read legacy COMPLETE: {exc}") from exc
    if expected != sweep._sha256(manifest_path):
        raise FormalValidationError("legacy COMPLETE does not authenticate MANIFEST.json")
    files = manifest.get("files")
    summary_entry = files.get(path.name) if isinstance(files, Mapping) else None
    if (
        not isinstance(summary_entry, Mapping)
        or summary_entry.get("sha256") != sweep._sha256(path)
        or summary_entry.get("size") != path.stat().st_size
    ):
        raise FormalValidationError("legacy bundle does not authenticate its summary")
    value = _read_json(path, label="legacy v3 baseline summary")
    baseline = value.get("baseline")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != LEGACY_SUMMARY_KIND
        or not isinstance(baseline, Mapping)
        or baseline.get("run_id") != "base-dense-v3-500m"
    ):
        raise FormalValidationError("legacy summary does not contain the v3-final baseline")
    rows = baseline.get("sources")
    if not isinstance(rows, list):
        raise FormalValidationError("legacy v3 baseline has no per-source rows")
    sources = {str(row.get("source")) for row in rows if isinstance(row, Mapping)}
    if sources != LEGACY_SOURCES:
        raise FormalValidationError(
            "legacy v3 baseline source coverage differs from the frozen six-source contract"
        )
    checkpoint_state = baseline.get("checkpoint_state")
    harness = baseline.get("evaluation_harness")
    comparison_contract = value.get("comparison_contract")
    if (
        not isinstance(checkpoint_state, Mapping)
        or not isinstance(harness, Mapping)
        or not isinstance(comparison_contract, Mapping)
    ):
        raise FormalValidationError("legacy v3 baseline lineage is incomplete")
    evaluation_identity = baseline.get("evaluation")
    plan_identity = (
        evaluation_identity.get("plan") if isinstance(evaluation_identity, Mapping) else None
    )
    if not isinstance(plan_identity, Mapping):
        raise FormalValidationError("legacy v3 baseline has no authenticated evaluation PLAN")
    legacy_plan_path = Path(str(plan_identity.get("path"))).resolve()
    if (
        not legacy_plan_path.is_file()
        or plan_identity.get("sha256") != sweep._sha256(legacy_plan_path)
        or plan_identity.get("size") != legacy_plan_path.stat().st_size
    ):
        raise FormalValidationError("legacy v3 evaluation PLAN identity mismatch")
    legacy_plan = _read_json(legacy_plan_path, label="legacy v3 evaluation PLAN")
    plan_fingerprint = legacy_plan.get("plan_fingerprint")
    unsigned_plan = {key: item for key, item in legacy_plan.items() if key != "plan_fingerprint"}
    if (
        legacy_plan.get("kind") != "twen_nll_evaluation_plan"
        or not isinstance(plan_fingerprint, str)
        or plan_fingerprint != sweep._canonical_sha256(unsigned_plan)
    ):
        raise FormalValidationError("legacy v3 evaluation PLAN fingerprint is invalid")
    if legacy_plan.get("checkpoint_state") != checkpoint_state:
        raise FormalValidationError(
            "legacy v3 evaluation PLAN checkpoint_state differs from its summary"
        )
    checkpoint_artifact = _checkpoint_artifact_from_plan(
        legacy_plan,
        label="legacy v3 evaluation PLAN",
    )
    return {
        "identity": {
            "summary": sweep._identity(path),
            "manifest": sweep._identity(manifest_path),
            "complete": sweep._identity(complete_path),
        },
        "checkpoint_state": dict(checkpoint_state),
        "checkpoint_artifact": checkpoint_artifact,
        "comparison_contract": dict(comparison_contract),
        "evaluation_harness": json.loads(json.dumps(harness, sort_keys=True)),
        "overall": dict(baseline["overall"]),
        "sources": [dict(row) for row in rows],
    }


def _assert_v3_checkpoint(
    evaluation: Mapping[str, Any],
    *,
    legacy: Mapping[str, Any],
    phase: str,
) -> None:
    if evaluation.get("run_id") != "base-dense-v3-500m":
        raise FormalValidationError(f"{phase} evaluation is not the v3-final run")
    if evaluation.get("checkpoint_state") != legacy["checkpoint_state"]:
        raise FormalValidationError(
            f"{phase} evaluation checkpoint differs from the authenticated v3 final"
        )
    current_checkpoint = _checkpoint_artifact_from_plan(
        evaluation.get("plan") if isinstance(evaluation.get("plan"), Mapping) else {},
        label=f"{phase} evaluation PLAN",
    )
    expected_checkpoint = legacy.get("checkpoint_artifact")
    if not isinstance(expected_checkpoint, Mapping) or current_checkpoint[
        "complete_sha256"
    ] != expected_checkpoint.get("complete_sha256"):
        raise FormalValidationError(
            f"{phase} evaluation checkpoint COMPLETE differs from the authenticated v3 final"
        )
    current = evaluation["harness"]["checkpoint_inference_lineage"]
    expected = legacy["evaluation_harness"]["checkpoint_inference_lineage"]
    for field in (
        "archived_config_sha256",
        "saved_critical_fingerprint",
        "saved_source_tree_sha256",
    ):
        if current.get(field) != expected.get(field):
            raise FormalValidationError(f"{phase} evaluation checkpoint lineage differs at {field}")


def _phase_summary(
    *,
    phase: str,
    prepared: Mapping[str, Any],
    evaluation_path: Path,
    legacy: Mapping[str, Any],
    sweep: ModuleType,
) -> dict[str, Any]:
    try:
        evaluation = sweep._load_completed_evaluation(
            evaluation_path,
            prepared=prepared,
            label=f"v3-final-{phase}",
        )
    except sweep.SweepError as exc:
        raise FormalValidationError(str(exc)) from exc
    _assert_v3_checkpoint(evaluation, legacy=legacy, phase=phase)
    public = sweep._public_evaluation(evaluation)
    public["comparison_contract"] = dict(evaluation["comparison_contract"])
    observed = {str(row["source"]) for row in public["sources"]}
    expected = EXPECTED_PHASE_SOURCES[phase]
    if observed != expected:
        raise FormalValidationError(
            f"{phase} evaluated source coverage differs "
            f"(missing={sorted(expected - observed)}, extra={sorted(observed - expected)})"
        )
    return {
        "phase": phase,
        "prepared": {
            "path": str(prepared["path"]),
            "sha256": prepared["sha256"],
            "dataset_fingerprint": prepared["dataset_fingerprint"],
            "token_count": prepared["token_count"],
            "sequence_count": prepared["sequence_count"],
            "shard_count": len(prepared["shards"]),
            "lineage": prepared["lineage_identity"],
        },
        **public,
    }


def _formal_numeric_contract(
    phases: Sequence[Mapping[str, Any]],
    *,
    legacy: Mapping[str, Any],
) -> dict[str, Any]:
    if len(phases) != len(EXPECTED_PHASE_SOURCES):
        raise FormalValidationError("formal evaluation phase inventory is incomplete")
    phase_contracts: dict[str, dict[str, Any]] = {}
    for phase in phases:
        phase_name = str(phase["phase"])
        comparison = phase.get("comparison_contract")
        harness = phase.get("evaluation_harness")
        if not isinstance(comparison, Mapping) or not isinstance(harness, Mapping):
            raise FormalValidationError(f"{phase_name} evaluation numeric contract is incomplete")
        phase_contracts[phase_name] = {
            **{field: comparison.get(field) for field in NUMERIC_COMPARISON_FIELDS},
            "config_fingerprint": harness.get("config_fingerprint"),
            "runtime": harness.get("runtime"),
            "checkpoint_inference_lineage": harness.get("checkpoint_inference_lineage"),
        }
    primary = phase_contracts.get("primary")
    cooldown = phase_contracts.get("cooldown")
    if primary is None or cooldown is None or cooldown != primary:
        raise FormalValidationError(
            "primary/cooldown evaluation numeric contracts differ "
            "(track/stage/expert/batch/device/dtype/config/runtime/lineage)"
        )
    legacy_comparison = legacy.get("comparison_contract")
    legacy_harness = legacy.get("evaluation_harness")
    if not isinstance(legacy_comparison, Mapping) or not isinstance(legacy_harness, Mapping):
        raise FormalValidationError("legacy evaluation numeric contract is incomplete")
    legacy_core = {field: legacy_comparison.get(field) for field in NUMERIC_COMPARISON_FIELDS}
    formal_core = {field: primary.get(field) for field in NUMERIC_COMPARISON_FIELDS}
    if formal_core != legacy_core:
        raise FormalValidationError(
            "formal evaluation track/stage/expert/batch/device/dtype differs "
            "from the authenticated v3 baseline"
        )
    legacy_runtime = legacy_harness.get("runtime")
    legacy_config_fingerprint = legacy_harness.get("config_fingerprint")
    return {
        "formal": primary,
        "legacy": {
            **legacy_core,
            "config_fingerprint": legacy_config_fingerprint,
            "runtime": legacy_runtime,
        },
        "formal_phases_identical": True,
        "legacy_core_contract_identical": True,
        "legacy_runtime_identical": primary["runtime"] == legacy_runtime,
        "legacy_current_preflight_fingerprint_identical": (
            primary["config_fingerprint"] == legacy_config_fingerprint
        ),
        "interpretation": (
            "runtime/config drift versus the historical evaluation is reported, while "
            "both formal phases are required to be numerically identical"
        ),
    }


def _combine_sources(phases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "phases": [],
            "shards": 0,
            "sequences": 0,
            "input_tokens": 0,
            "predicted_tokens": 0,
            "nll_sum": 0.0,
        }
    )
    for phase in phases:
        phase_name = str(phase["phase"])
        for row in phase["sources"]:
            source = str(row["source"])
            value = totals[source]
            value["phases"].append(phase_name)
            for field in (
                "shards",
                "sequences",
                "input_tokens",
                "predicted_tokens",
            ):
                value[field] += int(row[field])
            value["nll_sum"] += float(row["nll_sum"])
    result: list[dict[str, Any]] = []
    for source, value in sorted(totals.items()):
        predicted = int(value["predicted_tokens"])
        if predicted <= 0:
            raise FormalValidationError(f"{source} has no predicted tokens")
        mean_nll = float(value["nll_sum"]) / predicted
        result.append(
            {
                "source": source,
                **value,
                "mean_nll": mean_nll,
                "perplexity": math.exp(mean_nll) if mean_nll < 700 else None,
                "source_group": ("additional_v4" if source in ADDITIONAL_SOURCES else "legacy_six"),
            }
        )
    return result


def _legacy_comparison(
    combined: Sequence[Mapping[str, Any]],
    legacy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    current = {str(row["source"]): row for row in combined}
    result: list[dict[str, Any]] = []
    for legacy_row in legacy["sources"]:
        source = str(legacy_row["source"])
        if source not in current:
            if source not in FORMAL_SOURCE_REPLACEMENTS:
                raise FormalValidationError(
                    f"formal validation unexpectedly removed legacy source {source}"
                )
            continue
        formal = current[source]
        delta = float(formal["mean_nll"]) - float(legacy_row["mean_nll"])
        result.append(
            {
                "source": source,
                "legacy_mean_nll": float(legacy_row["mean_nll"]),
                "formal_mean_nll": float(formal["mean_nll"]),
                "formal_minus_legacy_nll": delta,
                "legacy_predicted_tokens": int(legacy_row["predicted_tokens"]),
                "formal_predicted_tokens": int(formal["predicted_tokens"]),
                "comparable_as_model_delta": False,
                "reason": "same checkpoint, different frozen-validation corpus",
            }
        )
    return result


def _source_replacements(
    combined: Sequence[Mapping[str, Any]],
    legacy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    current = {str(row["source"]): row for row in combined}
    previous = {str(row["source"]): row for row in legacy["sources"]}
    result: list[dict[str, Any]] = []
    for legacy_source, formal_source in sorted(FORMAL_SOURCE_REPLACEMENTS.items()):
        if legacy_source not in previous or legacy_source in current:
            raise FormalValidationError(
                f"legacy replacement source state is invalid: {legacy_source}"
            )
        if formal_source not in current or formal_source in previous:
            raise FormalValidationError(
                f"formal replacement source state is invalid: {formal_source}"
            )
        legacy_row = previous[legacy_source]
        formal_row = current[formal_source]
        result.append(
            {
                "legacy_source": legacy_source,
                "formal_source": formal_source,
                "legacy_mean_nll": float(legacy_row["mean_nll"]),
                "formal_mean_nll": float(formal_row["mean_nll"]),
                "legacy_predicted_tokens": int(legacy_row["predicted_tokens"]),
                "formal_predicted_tokens": int(formal_row["predicted_tokens"]),
                "comparable_as_model_delta": False,
                "formal_minus_legacy_nll": None,
                "reason": (
                    "source replacement and different frozen-validation corpus; "
                    "no NLL delta is claimed"
                ),
            }
        )
    return result


def _load_formal_inputs(
    *,
    primary_prepared: Path,
    primary_evaluation: Path,
    cooldown_prepared: Path,
    cooldown_evaluation: Path,
    validation_disjointness_attestation: Path,
    legacy_summary: Path,
    sweep: ModuleType,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    legacy = _load_legacy_baseline(legacy_summary, sweep=sweep)
    prepared = {
        phase: _load_governed_prepared(path, phase=phase, sweep=sweep)
        for phase, path in (
            ("primary", primary_prepared),
            ("cooldown", cooldown_prepared),
        )
    }
    validation_disjointness = _load_validation_disjointness(
        validation_disjointness_attestation,
        prepared=prepared,
        sweep=sweep,
    )
    phases = [
        _phase_summary(
            phase=phase,
            prepared=prepared[phase],
            evaluation_path=evaluation,
            legacy=legacy,
            sweep=sweep,
        )
        for phase, evaluation in (
            ("primary", primary_evaluation),
            ("cooldown", cooldown_evaluation),
        )
    ]
    return legacy, prepared, validation_disjointness, phases


def _formal_input_snapshot(
    *,
    legacy: Mapping[str, Any],
    validation_disjointness: Mapping[str, Any],
    phases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "legacy": legacy,
        "validation_disjointness": validation_disjointness,
        "phases": phases,
    }


def build_summary(
    *,
    primary_prepared: Path,
    primary_evaluation: Path,
    cooldown_prepared: Path,
    cooldown_evaluation: Path,
    validation_disjointness_attestation: Path,
    legacy_summary: Path,
) -> dict[str, Any]:
    sweep = _load_sweep_module()
    reporter_sources_start = _reporter_sources_identity(sweep)
    legacy, _prepared, validation_disjointness, phases = _load_formal_inputs(
        primary_prepared=primary_prepared,
        primary_evaluation=primary_evaluation,
        cooldown_prepared=cooldown_prepared,
        cooldown_evaluation=cooldown_evaluation,
        validation_disjointness_attestation=validation_disjointness_attestation,
        legacy_summary=legacy_summary,
        sweep=sweep,
    )
    input_snapshot = _formal_input_snapshot(
        legacy=legacy,
        validation_disjointness=validation_disjointness,
        phases=phases,
    )
    input_identity_start = sweep._canonical_sha256(input_snapshot)
    numeric_contract = _formal_numeric_contract(phases, legacy=legacy)
    combined = _combine_sources(phases)
    observed = {str(row["source"]) for row in combined}
    required = EXPECTED_PHASE_SOURCES["primary"] | EXPECTED_PHASE_SOURCES["cooldown"]
    if observed != required or not observed >= ADDITIONAL_SOURCES:
        raise FormalValidationError("combined formal validation lacks required source coverage")
    predicted_tokens = sum(int(row["predicted_tokens"]) for row in combined)
    nll_sum = sum(float(row["nll_sum"]) for row in combined)
    (
        legacy_end,
        _prepared_end,
        validation_disjointness_end,
        phases_end,
    ) = _load_formal_inputs(
        primary_prepared=primary_prepared,
        primary_evaluation=primary_evaluation,
        cooldown_prepared=cooldown_prepared,
        cooldown_evaluation=cooldown_evaluation,
        validation_disjointness_attestation=validation_disjointness_attestation,
        legacy_summary=legacy_summary,
        sweep=sweep,
    )
    input_identity_end = sweep._canonical_sha256(
        _formal_input_snapshot(
            legacy=legacy_end,
            validation_disjointness=validation_disjointness_end,
            phases=phases_end,
        )
    )
    if input_identity_end != input_identity_start:
        raise FormalValidationError("formal validation inputs changed while building the report")
    reporter_sources_end = _reporter_sources_identity(sweep)
    if reporter_sources_end != reporter_sources_start:
        raise FormalValidationError("formal validation reporter source changed during reporting")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "scope": "v3-final candidate-role NTP NLL; no training performed",
        "identity_reverification": {
            "input_identity_start_sha256": input_identity_start,
            "input_identity_end_sha256": input_identity_end,
            "passed": True,
        },
        "reporter_provenance": {
            "source_identity_start": reporter_sources_start,
            "source_identity_end": reporter_sources_end,
            "passed": True,
        },
        "checkpoint_state": legacy["checkpoint_state"],
        "evaluation_numeric_contract": numeric_contract,
        "formal_validation_disjointness": validation_disjointness,
        "legacy_six_source_baseline": legacy,
        "phases": phases,
        "combined": {
            "source_count": len(combined),
            "required_additional_sources": sorted(ADDITIONAL_SOURCES),
            "required_additional_sources_present": True,
            "sequences": sum(int(phase["overall"]["sequences"]) for phase in phases),
            "input_tokens": sum(int(phase["prepared"]["token_count"]) for phase in phases),
            "predicted_tokens": predicted_tokens,
            "nll_sum": nll_sum,
            "mean_nll": nll_sum / predicted_tokens,
            "perplexity": math.exp(nll_sum / predicted_tokens),
            "sources": combined,
        },
        "legacy_six_corpus_shift_comparison": _legacy_comparison(combined, legacy),
        "legacy_source_replacements": _source_replacements(combined, legacy),
        "gate": {
            "passed": validation_disjointness["passed"] is True,
            "conditions": {
                "prepared_lineage_governed": True,
                "frozen_validation_attestations_passed": True,
                "train_validation_union_disjointness_passed": validation_disjointness["passed"]
                is True,
                "primary_source_coverage_complete": True,
                "cooldown_source_coverage_complete": True,
                "additional_source_coverage_complete": True,
                "same_authenticated_v3_final_checkpoint": True,
                "candidate_role_full_shard_coverage": True,
            },
            "authorizes_training": False,
            "training_started_by_summarizer": False,
        },
    }


def _table_row(values: Sequence[Any]) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


def _markdown(summary: Mapping[str, Any]) -> str:
    combined = summary["combined"]
    source_rows = [
        _table_row(
            (
                row["source"],
                ", ".join(row["phases"]),
                f"{int(row['predicted_tokens']):,}",
                f"{float(row['mean_nll']):.6f}",
                f"{float(row['perplexity']):.4f}",
            )
        )
        for row in combined["sources"]
    ]
    comparison_rows = [
        _table_row(
            (
                row["source"],
                f"{float(row['legacy_mean_nll']):.6f}",
                f"{float(row['formal_mean_nll']):.6f}",
                f"{float(row['formal_minus_legacy_nll']):+.6f}",
            )
        )
        for row in summary["legacy_six_corpus_shift_comparison"]
    ]
    replacement_rows = [
        _table_row(
            (
                row["legacy_source"],
                row["formal_source"],
                f"{float(row['legacy_mean_nll']):.6f}",
                f"{float(row['formal_mean_nll']):.6f}",
                "不可比较",
            )
        )
        for row in summary["legacy_source_replacements"]
    ]
    phase_rows = [
        _table_row(
            (
                phase["phase"],
                len(phase["sources"]),
                phase["prepared"]["shard_count"],
                f"{int(phase['overall']['predicted_tokens']):,}",
                f"{float(phase['overall']['mean_nll']):.6f}",
            )
        )
        for phase in summary["phases"]
    ]
    return f"""# v4 250M 正式 frozen-validation：v3 final 基线

结论：来源级 frozen-validation 门已通过。primary/cooldown prepared validation 都来自
`role=validation` 的治理后不可变语料；validation union 与两阶段 train union 在 stable ID、
normalized exact 和 MinHash near-duplicate（阈值 0.8）三层均无重叠，validation 内部也无
重复。两次评测绑定同一个 v3 final checkpoint。该报告只建立正式训练前的 NLL 基线，
不启动、也不授权训练。

## 总览

| phase | 来源数 | shard | predicted tokens | mean NLL |
|---|---:|---:|---:|---:|
{chr(10).join(phase_rows)}

合并口径：{int(combined["predicted_tokens"]):,} predicted tokens，
mean NLL `{float(combined["mean_nll"]):.6f}`，PPL `{float(combined["perplexity"]):.4f}`。

![正式来源 NLL](charts/formal-source-nll.svg)

![正式来源 token 覆盖](charts/formal-source-tokens.svg)

## 正式来源级 v3 baseline

| source | phase | predicted tokens | mean NLL | PPL |
|---|---|---:|---:|---:|
{chr(10).join(source_rows)}

新增来源 Wikipedia、ArXiv、StackV2、CommonCorpus、LibreTexts、Gutenberg
均有独立统计；旧六来源中的英文、数学、GitHub code、OpenStax、Stanford 五来源继续保留。

## 与旧基线共享的五个来源

| source | 旧 frozen NLL | 正式 frozen NLL | 正式−旧 |
|---|---:|---:|---:|
{chr(10).join(comparison_rows)}

这里是同一 v3 checkpoint 在不同 held-out 语料上的 corpus-shift 诊断，不能解释成模型
提升或退化。后续 v4 checkpoint 必须复用本报告两份完全相同的 prepared manifest，才可把
NLL 差异解释为 checkpoint 差异。

## 中文来源替换

| 旧来源 | 正式来源 | 旧 frozen NLL | 正式 frozen NLL | delta |
|---|---|---:|---:|---|
{chr(10).join(replacement_rows)}

FineWeb2 中文与 Wikipedia 是不同来源、不同 held-out 语料；上表仅披露替换及各自绝对
NLL，不计算、也不暗示可比较的模型质量 delta。

## 门禁边界

- prepared manifest、dataset fingerprint、source map、audit attestation 和每个 evaluation
  shard 的 COMPLETE/输出 SHA 均已重新认证；
- primary+cooldown validation union 与两阶段 train union 的 stable-ID、normalized-exact、
  MinHash near-duplicate（estimated Jaccard ≥ 0.8）隔离证明已通过，validation 内部同样通过；
- primary 与 cooldown 都完整覆盖其 recipe 来源，六个新增来源全部存在；
- 两次 evaluation 的 v3 checkpoint state 及保存时 lineage 与旧六来源基线一致；
- 本工具不运行 forward、不修改输入、不启动训练；`gate.authorizes_training=false`。
"""


def _verify_existing(root: Path, *, sweep: ModuleType) -> None:
    manifest_path = root / "MANIFEST.json"
    complete_path = root / "COMPLETE"
    if not manifest_path.is_file() or not complete_path.is_file():
        raise FormalValidationError(f"cannot replace incomplete bundle: {root}")
    if complete_path.read_text(encoding="ascii").strip() != sweep._sha256(manifest_path):
        raise FormalValidationError(f"cannot replace unauthenticated bundle: {root}")
    manifest = _read_json(manifest_path, label="existing formal bundle")
    if manifest.get("kind") != BUNDLE_KIND or manifest.get("schema_version") != SCHEMA_VERSION:
        raise FormalValidationError(f"cannot replace unsupported bundle: {root}")


def write_bundle(
    summary: Mapping[str, Any],
    output: Path,
    *,
    replace_existing: bool = False,
) -> dict[str, Any]:
    sweep = _load_sweep_module()
    provenance = summary.get("reporter_provenance")
    expected_reporter_sources = (
        provenance.get("source_identity_end") if isinstance(provenance, Mapping) else None
    )
    if (
        not isinstance(expected_reporter_sources, Mapping)
        or provenance.get("source_identity_start") != expected_reporter_sources
        or provenance.get("passed") is not True
        or _reporter_sources_identity(sweep) != expected_reporter_sources
    ):
        raise FormalValidationError(
            "formal validation summary has stale reporter source provenance"
        )
    output = output.resolve()
    staging = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    if staging.exists():
        raise FormalValidationError(f"stale staging directory exists: {staging}")
    (staging / "charts").mkdir(parents=True)
    try:
        summary_path = staging / "summary.json"
        report_path = staging / "REPORT.zh-CN.md"
        nll_chart = staging / "charts" / "formal-source-nll.svg"
        token_chart = staging / "charts" / "formal-source-tokens.svg"
        sweep._atomic_write(summary_path, sweep._json_text(summary))
        sweep._atomic_write(report_path, _markdown(summary))
        sources = summary["combined"]["sources"]
        sweep._atomic_write(
            nll_chart,
            sweep._delta_svg(
                title="v3-final NLL on governed v4 formal validation",
                groups=[str(row["source"]) for row in sources],
                series={"mean NLL": [float(row["mean_nll"]) for row in sources]},
                y_label="Token-weighted mean NLL",
            ),
        )
        sweep._atomic_write(
            token_chart,
            sweep._delta_svg(
                title="Governed v4 formal validation coverage",
                groups=[str(row["source"]) for row in sources],
                series={"predicted tokens": [float(row["predicted_tokens"]) for row in sources]},
                y_label="Predicted tokens",
            ),
        )
        payloads = (summary_path, report_path, nll_chart, token_chart)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": BUNDLE_KIND,
            "gate": summary["gate"],
            "files": {
                path.relative_to(staging).as_posix(): sweep._identity(path, relative_to=staging)
                for path in payloads
            },
        }
        manifest_path = staging / "MANIFEST.json"
        sweep._atomic_write(manifest_path, sweep._json_text(manifest))
        sweep._atomic_write(staging / "COMPLETE", sweep._sha256(manifest_path) + "\n")
        if _reporter_sources_identity(sweep) != expected_reporter_sources:
            raise FormalValidationError(
                "formal validation reporter source changed while writing the bundle"
            )
        if output.exists():
            if output.is_dir() and sweep._tree_identity(output) == sweep._tree_identity(staging):
                shutil.rmtree(staging)
            elif not replace_existing:
                raise FormalValidationError(f"output already exists: {output}")
            else:
                _verify_existing(output, sweep=sweep)
                backup = output.with_name(f".{output.name}.replaced-{os.getpid()}")
                os.replace(output, backup)
                try:
                    os.replace(staging, output)
                except BaseException:
                    os.replace(backup, output)
                    raise
                shutil.rmtree(backup)
        else:
            os.replace(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "output": str(output),
        "summary": str(output / "summary.json"),
        "report_zh_cn": str(output / "REPORT.zh-CN.md"),
        "manifest": str(output / "MANIFEST.json"),
        "manifest_sha256": sweep._sha256(output / "MANIFEST.json"),
        "complete": str(output / "COMPLETE"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = build_summary(
            primary_prepared=args.primary_prepared,
            primary_evaluation=args.primary_evaluation,
            cooldown_prepared=args.cooldown_prepared,
            cooldown_evaluation=args.cooldown_evaluation,
            validation_disjointness_attestation=args.validation_disjointness_attestation,
            legacy_summary=args.legacy_summary,
        )
        result = write_bundle(
            summary,
            args.output,
            replace_existing=args.replace_existing,
        )
    except FormalValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
