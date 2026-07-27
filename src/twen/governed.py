"""Fail-closed contracts for externally governed v4 training.

This module is deliberately CPU-only.  It builds an immutable plan from the
readiness lock, reports whether a launch can be authorized, and evaluates
already-authenticated checkpoint observations.  It never imports the training
engine, initializes CUDA, or launches work on import.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from .source_identity import twen_source_tree_sha256
from .utils import atomic_write_json, sha256_file

PLAN_KIND = "twen_v4_governed_training_plan"
STATE_KIND = "twen_v4_governed_training_state"
GATE_KIND = "twen_v4_governed_gate_decision"
SCHEMA_VERSION = 1

REQUIRED_FINITE_METRICS = (
    "loss",
    "ntp",
    "mtp",
    "grad_norm",
    "lr",
    "lr/adapters",
    "lr/scale",
    "lr_adjusted/adapters",
)

FORMAL_V4_PRIMARY_SOURCE_MIX = {
    "chinese_wikipedia_zh_20231101": 2_400,
    "code_github_clean_allowlisted": 900,
    "code_stackv2_edu_permissive": 400,
    "english_fineweb_edu_dedup": 2_500,
    "math_finemath_4plus": 1_400,
    "multilingual_common_corpus_permissive": 500,
    "science_arxiv_open_permissive": 1_200,
    "science_cosmopedia_openstax": 400,
    "science_cosmopedia_stanford": 300,
}

FORMAL_V4_COOLDOWN_SOURCE_MIX = {
    "chinese_wikipedia_zh_20231101": 2_500,
    "code_github_clean_allowlisted": 500,
    "education_libretexts_permissive": 400,
    "math_finemath_4plus": 2_200,
    "public_domain_project_gutenberg": 400,
    "science_arxiv_open_permissive": 2_200,
    "science_cosmopedia_openstax": 1_000,
    "science_cosmopedia_stanford": 800,
}

FORMAL_V4_PAUSE_THRESHOLDS = (
    13_000_000,
    26_000_000,
    52_000_000,
    105_000_000,
    157_000_000,
    210_000_000,
    223_000_000,
    236_000_000,
    250_000_000,
)

# This is intentionally compiled into the source-bound authorizer rather than
# learned from the readiness/config pair.  A self-consistent edit of both
# launch files therefore cannot silently redefine the reviewed formal run.
FORMAL_V4_CONFIG_POLICY: dict[str, Any] = {
    "run_id": "base-dense-v4-250m-pilot",
    "checkpoint_output_dir": "runs/base-dense-v4-250m-pilot",
    "student_layers": 24,
    "student_hidden_size": 1_024,
    "student_intermediate_size": 3_584,
    "donor_layers": 32,
    "donor_hidden_size": 4_096,
    "donor_intermediate_size": 12_288,
    "expert_intermediate_size": 1_536,
    "num_experts": 8,
    "top_k": 2,
    "norm_topk_prob": True,
    "lora_rank": 16,
    "expert_initialization": "donor",
    "active_student_layers": None,
    "adapter_init_path": "artifacts/calibration/base/adapters.safetensors",
    "channel_map_path": "artifacts/calibration/base/channel_map.json",
    "layer_map_path": "artifacts/calibration/base/layer_map.json",
    "track": "base",
    "stage": "dense-oracle",
    "data_mode": "prepared-text",
    "objective": "base_text_ntp_plus_native_mtp_no_9b_logits",
    "source_mix_algorithm": "token-deficit-corrected-source-mix-bp-v2",
    "primary_source_mix_basis_points": FORMAL_V4_PRIMARY_SOURCE_MIX,
    "cooldown_source_mix_basis_points": FORMAL_V4_COOLDOWN_SOURCE_MIX,
    "shuffle_seed": 3_407,
    "primary_tokens": 225_000_000,
    "cooldown_tokens": 25_000_000,
    "max_tokens": 250_000_000,
    "cooldown_start_tokens": 225_000_000,
    "sequence_length": 4_096,
    "world_size": 1,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 64,
    "global_batch_tokens": 262_144,
    "complete_tail_batch_required": True,
    "allow_corpus_reuse": False,
    "source_mix_allow_weight_override": False,
    "checkpoint_every_steps": 50,
    "checkpoint_every_minutes": 20.0,
    "checkpoint_keep_last": 3,
    "checkpoint_save_on_signal": True,
    "checkpoint_stop_file": "STOP",
    "runtime_bf16": True,
    "runtime_allow_tf32": True,
    "runtime_fused_adamw": True,
    "runtime_activation_checkpointing": True,
    "runtime_activation_checkpoint_layer_count": 0,
    "runtime_loss_checkpoint_chunks": True,
    "runtime_loss_chunk_tokens": 512,
    "runtime_compile_streaming_loss": True,
    "runtime_expandable_segments": True,
    "runtime_sharding": "fsdp2",
    "runtime_teacher_cpu_offload": False,
    "adapter_optimizer": "muon",
    "scale_optimizer": "adamw",
    "adapter_lr": 3.0e-5,
    "lora_lr": 3.0e-5,
    "scale_lr": 3.0e-6,
    "warmup_tokens": 10_000_000,
    "lr_schedule": "cosine",
    "min_lr_ratio": 0.1,
    "grad_clip_norm": 1.0,
    "muon_momentum": 0.95,
    "muon_nesterov": True,
    "muon_ns_coefficients": [3.4445, -4.775, 2.0315],
    "muon_eps": 1.0e-7,
    "muon_ns_steps": 5,
    "muon_adjust_lr_fn": "match_rms_adamw",
    "ntp_weight": 1.0,
    "mtp_weight": 0.1,
    "native_mtp_head_frozen": True,
    "teacher_logits_kd": False,
    "teacher_kd_weight": 0.0,
    "anchor_kl_weight": 0.0,
    "hidden_alignment_weight": 0.0,
    "dense_oracle_weight": 0.0,
    "router_z_weight": 0.0,
    "load_balance_weight": 0.0,
    "router_supervision_weight": 0.0,
    "dense_oracle_batch_fraction": 0.0,
    "hidden_alignment_batch_fraction": 0.0,
}

# The already-produced readiness/capacity evidence uses this deliberately
# smaller public contract.  Config-only implementation details are still
# source-bound above and are authenticated directly against the hashed YAML;
# they are not retroactively claimed as fields in older attestations.
FORMAL_V4_ATTESTED_CONTRACT = {
    key: FORMAL_V4_CONFIG_POLICY[key]
    for key in (
        "track",
        "stage",
        "data_mode",
        "objective",
        "source_mix_algorithm",
        "primary_tokens",
        "cooldown_tokens",
        "max_tokens",
        "cooldown_start_tokens",
        "sequence_length",
        "world_size",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "global_batch_tokens",
        "complete_tail_batch_required",
        "allow_corpus_reuse",
        "adapter_optimizer",
        "scale_optimizer",
        "adapter_lr",
        "lora_lr",
        "scale_lr",
        "warmup_tokens",
        "lr_schedule",
        "min_lr_ratio",
        "ntp_weight",
        "mtp_weight",
        "native_mtp_head_frozen",
        "teacher_logits_kd",
        "teacher_kd_weight",
        "anchor_kl_weight",
        "hidden_alignment_weight",
        "dense_oracle_weight",
    )
}

FORMAL_V4_FORK_PATH = (
    "runs/base-dense-v3-500m/step-000000001912-milestone-complete"
)
FORMAL_V4_FORK_COMPLETE_SHA256 = (
    "3a21a50e35de74ecd0ff5b8f00aa29ed6c83f746fc2cf97d4da6b0536262b6c7"
)
FORMAL_V4_FORBIDDEN_WARM_STARTS = (
    "runs/base-dense-v4-16m-smoke",
    "runs/base-dense-v4-13m-low-lr-calibration",
)
FORMAL_V4_DYNAMIC_CONFIG_IDENTITY_FIELDS = (
    "manifest_path",
    "manifest_sha256",
    "source_map_sha256",
    "quality_cooldown_manifest_path",
    "quality_cooldown_manifest_sha256",
    "phase_disjointness_attestation_path",
    "phase_disjointness_attestation_sha256",
)
FORMAL_V4_NORMALIZED_CONFIG_SHA256 = (
    "3902ecfbe05a73c0079fa88ddd3336c064c785b059f9bf30316f40b927ea7c81"
)
FORMAL_V4_DISJOINTNESS_ALGORITHMS = {
    "stable_id_exact": "source-scoped-authenticated-stable-id-intersection-v1",
    "normalized_text_exact": "unicode-nfkc-whitespace-sha256-intersection-v1",
    "near_duplicate": "lexical-5gram-one-permutation-minhash-lsh-v1",
}
FORMAL_V4_CHINESE_SEMANTIC_SOURCE_ID = "chinese_wikipedia_zh_20231101"
FORMAL_V4_CHINESE_SEMANTIC_REQUIRED_BUNDLE = {
    "manifest_kind": "twen_v4_chinese_semantic_noise_bundle",
    "complete_kind": "twen_v4_chinese_semantic_noise_complete",
    "attestation_kind": "twen_v4_chinese_semantic_noise_attestation",
}
FORMAL_V4_CHINESE_SEMANTIC_REQUIRED_GATES = {
    "all_selected_shards_authenticated": True,
    "complete_streaming_scan": True,
    "control_samples_per_phase_gte": 32,
    "high_precision_conversion_documents_eq": 0,
    "malformed_punctuation_documents_eq": 0,
    "manual_review_passed": True,
    "reviewed_at_timezone_aware_iso8601_required": True,
    "reviewer_placeholder_forbidden": True,
    "risk_samples_per_phase_gte": 32,
}
FORMAL_V4_CHINESE_SEMANTIC_SCANNER_POLICY = {
    "conversion_markers": [
        "da_nian_ye_substitution",
        "engine_substitution",
        "brand_command_substitution",
        "view_character_expansion",
        "cannot_substitution",
        "appeared_substitution",
    ],
    "high_precision_conversion_documents_eq": 0,
    "malformed_punctuation_documents_eq": 0,
    "manual_unacceptable_samples_eq": 0,
    "risk_samples_per_phase_gte": 32,
    "control_samples_per_phase_gte": 32,
    "reviewer_placeholder_forbidden": True,
    "reviewed_at_timezone_aware_iso8601_required": True,
}
FORMAL_V4_CHINESE_SEMANTIC_AUDIT_GATES = {
    "all_selected_shards_authenticated": True,
    "complete_streaming_scan": True,
    "high_precision_conversion_documents": 0,
    "high_precision_conversion_passed": True,
    "malformed_punctuation_documents": 0,
    "malformed_punctuation_passed": True,
    "manual_review_passed": True,
}
FORMAL_V4_WIKIPEDIA_LICENSE_CONTRACT = {
    "source_id": FORMAL_V4_CHINESE_SEMANTIC_SOURCE_ID,
    "repo_id": "wikimedia/wikipedia",
    "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    "declared_license": "CC-BY-SA-3.0 AND GFDL",
    "scope": "formal-v4-250m-primary-and-cooldown-only",
    "attribution_fields": ["id", "url", "title"],
    "obligations": [
        "retain the generated attribution manifests with the training evidence",
        "document the CC-BY-SA-3.0/GFDL source in model and data reports",
        "perform a separate final review of model/distribution compliance",
    ],
}
FORMAL_V4_WIKIPEDIA_LICENSE_FINGERPRINT = (
    "fbc16551a1d7c0b9020852be97f751af2759442bd4d22bade707cd50a4fa3762"
)
FORMAL_V4_RELEASE_KIND = "twen_v4_250m_formal_release"
FORMAL_V4_RELEASE_BUNDLE_KIND = "twen_v4_250m_formal_release_bundle"
FORMAL_V4_RELEASE_COMPLETE_KIND = "twen_v4_250m_formal_release_complete"
FORMAL_V4_RELEASE_CONFIG_NAME = "dense-v4-250m-pilot.yaml"
FORMAL_V4_RELEASE_READINESS_NAME = "readiness.json"
FORMAL_V4_RELEASE_MANIFEST_NAME = "MANIFEST.json"
FORMAL_V4_RELEASE_COMPLETE_NAME = "COMPLETE"
FORMAL_V4_CLOSURE_BUNDLE_KIND = (
    "twen_v4_250m_formal_evidence_closure_bundle"
)
FORMAL_V4_CLOSURE_COMPLETE_KIND = (
    "twen_v4_250m_formal_evidence_closure_complete"
)
FORMAL_V4_CALIBRATION_ATTESTATION_KIND = (
    "twen_v4_13m_calibration_release_attestation"
)
FORMAL_V4_CALIBRATION_COMPLETE_KIND = (
    "twen_v4_13m_calibration_release_attestation_complete"
)


class GovernedControllerError(RuntimeError):
    """The governed plan, state, authorization, or evidence is unsafe."""


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GovernedControllerError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GovernedControllerError(f"{label} must be a JSON object: {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_authenticated_script(
    path: Path,
    *,
    module_name: str,
    label: str,
) -> ModuleType:
    resolved = path.resolve()
    if not resolved.is_file():
        raise GovernedControllerError(f"{label} does not exist: {resolved}")
    source_before = resolved.read_bytes()
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise GovernedControllerError(f"cannot load {label}: {resolved}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    if resolved.read_bytes() != source_before:
        raise GovernedControllerError(f"{label} changed while loading: {resolved}")
    return module


def _sha256_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise GovernedControllerError(f"{label} must be a SHA256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise GovernedControllerError(f"{label} must be a 64-digit SHA256")
    return normalized


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GovernedControllerError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _finite(value: Any, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GovernedControllerError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise GovernedControllerError(f"{label} must be finite")
    return result


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GovernedControllerError(f"{label} must be an object")
    return value


def _require_exact_contract(
    value: Any,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Require an exact typed JSON contract, including its key inventory."""

    actual = dict(_mapping(value, label=label))
    if _canonical_sha256(actual) != _canonical_sha256(dict(expected)):
        raise GovernedControllerError(
            f"{label} differs from the source-bound formal v4 policy"
        )
    return actual


def _require_config_fields(
    value: Any,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Recursively authenticate the policy-bearing subset of a YAML mapping."""

    actual = _mapping(value, label=label)
    for field, expected_value in expected.items():
        field_label = f"{label}.{field}"
        if isinstance(expected_value, Mapping):
            _require_config_fields(
                actual.get(field),
                expected_value,
                label=field_label,
            )
            continue
        actual_value = actual.get(field)
        if _canonical_sha256(actual_value) != _canonical_sha256(expected_value):
            raise GovernedControllerError(
                f"{field_label} differs from the source-bound formal v4 policy"
            )


def _normalized_formal_config_sha256(value: Mapping[str, Any]) -> str:
    """Hash the entire YAML semantics while abstracting only final data IDs."""

    try:
        normalized = json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        data = normalized["data"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GovernedControllerError(
            "formal config cannot be normalized as a JSON mapping"
        ) from exc
    if not isinstance(data, dict):
        raise GovernedControllerError("formal config data section is invalid")
    for field in FORMAL_V4_DYNAMIC_CONFIG_IDENTITY_FIELDS:
        if field not in data:
            raise GovernedControllerError(
                f"formal config dynamic identity is missing: data.{field}"
            )
        data[field] = f"<DYNAMIC:{field}>"
    return _canonical_sha256(normalized)


def _project_path(project_root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise GovernedControllerError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _readiness_project_root(
    readiness_file: Path,
    readiness: Mapping[str, Any],
) -> Path:
    explicit = readiness.get("project_root")
    if explicit is not None:
        root = _project_path(
            readiness_file.parent,
            explicit,
            label="readiness.project_root",
        )
        if not (root / "src/twen").is_dir() or not (root / "pyproject.toml").is_file():
            raise GovernedControllerError(
                f"readiness project_root is not a Twen checkout: {root}"
            )
        return root
    for candidate in (readiness_file.parent, *readiness_file.parents):
        if (candidate / "src/twen").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate.resolve()
    # Compatibility for isolated negative-test fixtures. Release readiness
    # emitted outside locks/ must carry the explicit root above.
    return readiness_file.parent.parent.resolve()


def _optional_binding(
    project_root: Path,
    value: Any,
    *,
    label: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if value is None:
        return None, f"{label} is not bound"
    try:
        path = _project_path(project_root, value, label=label)
    except GovernedControllerError as exc:
        return None, str(exc)
    if not path.is_file():
        return {"path": str(path), "sha256": None}, f"{label} does not exist: {path}"
    return {"path": str(path), "sha256": sha256_file(path)}, None


def _authenticated_file_identity(
    value: Any,
    *,
    expected_path: Path,
    label: str,
) -> dict[str, Any]:
    identity = _mapping(value, label=label)
    path = Path(str(identity.get("path"))).expanduser().resolve()
    expected = expected_path.expanduser().resolve()
    size = _integer(identity.get("size"), label=f"{label}.size")
    digest = _sha256_string(identity.get("sha256"), label=f"{label}.sha256")
    if (
        path != expected
        or not path.is_file()
        or path.stat().st_size != size
        or sha256_file(path) != digest
    ):
        raise GovernedControllerError(f"{label} identity differs")
    return {
        "path": str(path),
        "size": size,
        "sha256": digest,
    }


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        raise GovernedControllerError(
            f"{label} key inventory differs: expected {sorted(expected)}, "
            f"got {sorted(str(key) for key in value)}"
        )


def _current_file_identity(path: Path, *, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise GovernedControllerError(f"{label} is missing or is a symlink: {resolved}")
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _relative_file_identity(
    root: Path,
    relative: str,
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    identity = _mapping(value, label=label)
    _require_exact_keys(identity, {"path", "size", "sha256"}, label=label)
    if identity.get("path") != relative:
        raise GovernedControllerError(f"{label}.path differs")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise GovernedControllerError(f"{label}.path escapes its bundle")
    actual = _current_file_identity(path, label=label)
    if (
        identity.get("size") != actual["size"]
        or _sha256_string(identity.get("sha256"), label=f"{label}.sha256")
        != actual["sha256"]
    ):
        raise GovernedControllerError(f"{label} identity differs")
    return actual


def _strict_directory_files(
    root: Path,
    *,
    expected_files: set[str],
    label: str,
) -> None:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise GovernedControllerError(f"{label} is missing or is a symlink: {resolved}")
    entries = list(resolved.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise GovernedControllerError(f"{label} contains a symlink")
    observed_files = {
        path.relative_to(resolved).as_posix()
        for path in entries
        if path.is_file()
    }
    if observed_files != expected_files:
        raise GovernedControllerError(
            f"{label} file inventory differs: expected {sorted(expected_files)}, "
            f"got {sorted(observed_files)}"
        )
    expected_directories = {
        parent.as_posix()
        for relative in expected_files
        for parent in Path(relative).parents
        if parent.as_posix() != "."
    }
    observed_directories = {
        path.relative_to(resolved).as_posix()
        for path in entries
        if path.is_dir()
    }
    if observed_directories != expected_directories:
        raise GovernedControllerError(f"{label} directory inventory differs")


def _validate_readiness_fingerprint(
    readiness: Mapping[str, Any],
    *,
    required: bool,
) -> str | None:
    raw = readiness.get("readiness_fingerprint")
    if raw is None and not required:
        return None
    fingerprint = _sha256_string(raw, label="readiness.readiness_fingerprint")
    unsigned = {
        key: value
        for key, value in readiness.items()
        if key != "readiness_fingerprint"
    }
    if _canonical_sha256(unsigned) != fingerprint:
        raise GovernedControllerError("readiness fingerprint is invalid")
    return fingerprint


def _authenticate_closure_release_binding(
    value: Any,
    *,
    project_root: Path,
) -> dict[str, Any]:
    binding = _mapping(value, label="release contract formal_closure")
    _require_exact_keys(
        binding,
        {
            "path",
            "manifest_sha256",
            "complete_sha256",
            "bundle_fingerprint",
        },
        label="release contract formal_closure",
    )
    root = _project_path(
        project_root,
        binding.get("path"),
        label="release contract formal_closure.path",
    )
    expected_files = {
        "capacity-attestation.json",
        "readiness.json",
        FORMAL_V4_RELEASE_MANIFEST_NAME,
        FORMAL_V4_RELEASE_COMPLETE_NAME,
    }
    _strict_directory_files(
        root,
        expected_files=expected_files,
        label="formal closure bundle",
    )
    manifest_path = root / FORMAL_V4_RELEASE_MANIFEST_NAME
    complete_path = root / FORMAL_V4_RELEASE_COMPLETE_NAME
    manifest = _read_json(manifest_path, label="formal closure MANIFEST")
    complete = _read_json(complete_path, label="formal closure COMPLETE")
    manifest_unsigned = {
        key: item
        for key, item in manifest.items()
        if key != "bundle_fingerprint"
    }
    bundle_fingerprint = _sha256_string(
        manifest.get("bundle_fingerprint"),
        label="formal closure bundle_fingerprint",
    )
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != FORMAL_V4_CLOSURE_BUNDLE_KIND
        or manifest.get("launch_enabled") is not False
        or manifest.get("authorizes_training") is not False
        or manifest.get("training_started") is not False
        or _canonical_sha256(manifest_unsigned) != bundle_fingerprint
    ):
        raise GovernedControllerError("formal closure MANIFEST contract differs")
    if (
        complete.get("schema_version") != SCHEMA_VERSION
        or complete.get("kind") != FORMAL_V4_CLOSURE_COMPLETE_KIND
        or complete.get("manifest") != FORMAL_V4_RELEASE_MANIFEST_NAME
        or complete.get("manifest_sha256") != sha256_file(manifest_path)
        or complete.get("bundle_fingerprint") != bundle_fingerprint
        or complete.get("launch_enabled") is not False
        or complete.get("authorizes_training") is not False
        or complete.get("training_started") is not False
    ):
        raise GovernedControllerError(
            "formal closure COMPLETE does not authenticate MANIFEST"
        )
    manifest_sha = sha256_file(manifest_path)
    complete_sha = sha256_file(complete_path)
    if (
        _sha256_string(
            binding.get("manifest_sha256"),
            label="release contract formal_closure.manifest_sha256",
        )
        != manifest_sha
        or _sha256_string(
            binding.get("complete_sha256"),
            label="release contract formal_closure.complete_sha256",
        )
        != complete_sha
        or binding.get("bundle_fingerprint") != bundle_fingerprint
    ):
        raise GovernedControllerError("release formal_closure identity differs")
    files = _mapping(manifest.get("files"), label="formal closure MANIFEST.files")
    if set(files) != {"capacity-attestation.json", "readiness.json"}:
        raise GovernedControllerError("formal closure payload inventory differs")
    payloads = {
        relative: _relative_file_identity(
            root,
            relative,
            files[relative],
            label=f"formal closure MANIFEST.files.{relative}",
        )
        for relative in ("capacity-attestation.json", "readiness.json")
    }
    return {
        "path": str(root),
        "manifest": _current_file_identity(
            manifest_path,
            label="formal closure MANIFEST",
        ),
        "complete": _current_file_identity(
            complete_path,
            label="formal closure COMPLETE",
        ),
        "bundle_fingerprint": bundle_fingerprint,
        "payloads": payloads,
        "closed_readiness": _read_json(
            root / "readiness.json",
            label="formal closure readiness",
        ),
    }


def _authenticate_release_report_bundle(
    value: Any,
    *,
    project_root: Path,
    label: str,
) -> dict[str, Any]:
    binding = _mapping(value, label=label)
    _require_exact_keys(
        binding,
        {
            "path",
            "manifest_sha256",
            "complete_sha256",
            "manifest_kind",
            "complete_kind",
        },
        label=label,
    )
    root = _project_path(project_root, binding.get("path"), label=f"{label}.path")
    manifest_path = root / FORMAL_V4_RELEASE_MANIFEST_NAME
    complete_path = root / FORMAL_V4_RELEASE_COMPLETE_NAME
    manifest = _read_json(manifest_path, label=f"{label} MANIFEST")
    files = _mapping(manifest.get("files"), label=f"{label} MANIFEST.files")
    if not files or not all(isinstance(relative, str) for relative in files):
        raise GovernedControllerError(f"{label} payload inventory is invalid")
    expected_files = set(files) | {
        FORMAL_V4_RELEASE_MANIFEST_NAME,
        FORMAL_V4_RELEASE_COMPLETE_NAME,
    }
    _strict_directory_files(root, expected_files=expected_files, label=label)
    manifest_sha = sha256_file(manifest_path)
    complete_sha = sha256_file(complete_path)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != binding.get("manifest_kind")
        or _sha256_string(
            binding.get("manifest_sha256"),
            label=f"{label}.manifest_sha256",
        )
        != manifest_sha
        or _sha256_string(
            binding.get("complete_sha256"),
            label=f"{label}.complete_sha256",
        )
        != complete_sha
    ):
        raise GovernedControllerError(f"{label} MANIFEST/COMPLETE identity differs")
    complete_payload = complete_path.read_bytes()
    try:
        complete_json = json.loads(complete_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        try:
            marker = complete_payload.decode("ascii", errors="strict").strip()
        except UnicodeDecodeError as exc:
            raise GovernedControllerError(f"{label} COMPLETE is invalid") from exc
        if binding.get("complete_kind") is not None or marker != manifest_sha:
            raise GovernedControllerError(
                f"{label} COMPLETE does not authenticate MANIFEST"
            ) from None
    else:
        complete = _mapping(complete_json, label=f"{label} COMPLETE")
        if (
            complete.get("manifest_sha256") != manifest_sha
            or (
                complete.get("manifest") is not None
                and complete.get("manifest") != FORMAL_V4_RELEASE_MANIFEST_NAME
            )
            or (
                binding.get("complete_kind") is not None
                and complete.get("kind") != binding.get("complete_kind")
            )
        ):
            raise GovernedControllerError(
                f"{label} COMPLETE does not authenticate MANIFEST"
            )
    for relative, raw_identity in files.items():
        _relative_file_identity(
            root,
            str(relative),
            raw_identity,
            label=f"{label} MANIFEST.files.{relative}",
        )
    return {
        "path": str(root),
        "manifest_sha256": manifest_sha,
        "complete_sha256": complete_sha,
        "manifest_kind": binding.get("manifest_kind"),
        "complete_kind": binding.get("complete_kind"),
    }


def _authenticate_calibration_release_binding(
    value: Any,
    *,
    project_root: Path,
    closure: Mapping[str, Any],
) -> dict[str, Any]:
    binding = _mapping(value, label="release contract calibration_attestation")
    required_keys = {
        "path",
        "size",
        "sha256",
        "complete",
        "attestation_fingerprint",
        "calibration_config",
        "evidence",
        "candidate_checkpoints",
        "final_checkpoint",
        "evaluation",
    }
    _require_exact_keys(
        binding,
        required_keys,
        label="release contract calibration_attestation",
    )
    attestation_path = _project_path(
        project_root,
        binding.get("path"),
        label="release contract calibration_attestation.path",
    )
    attestation_identity = _authenticated_file_identity(
        {
            "path": str(attestation_path),
            "size": binding.get("size"),
            "sha256": binding.get("sha256"),
        },
        expected_path=attestation_path,
        label="release calibration attestation",
    )
    complete_path = attestation_path.with_name(FORMAL_V4_RELEASE_COMPLETE_NAME)
    complete_identity = _authenticated_file_identity(
        binding.get("complete"),
        expected_path=complete_path,
        label="release calibration COMPLETE",
    )
    attestation = _read_json(
        attestation_path,
        label="release calibration attestation",
    )
    complete = _read_json(complete_path, label="release calibration COMPLETE")
    attestation_unsigned = {
        key: item
        for key, item in attestation.items()
        if key != "attestation_fingerprint"
    }
    attestation_fingerprint = _sha256_string(
        attestation.get("attestation_fingerprint"),
        label="release calibration attestation_fingerprint",
    )
    if (
        attestation.get("schema_version") != SCHEMA_VERSION
        or attestation.get("kind") != FORMAL_V4_CALIBRATION_ATTESTATION_KIND
        or attestation.get("status")
        != "passed_authenticated_quality_gate_but_does_not_authorize_formal_training"
        or attestation.get("passed") is not True
        or attestation.get("authorizes_training") is not False
        or attestation.get("training_started") is not False
        or _canonical_sha256(attestation_unsigned) != attestation_fingerprint
        or binding.get("attestation_fingerprint") != attestation_fingerprint
    ):
        raise GovernedControllerError("release calibration attestation differs")
    if (
        complete.get("schema_version") != SCHEMA_VERSION
        or complete.get("kind") != FORMAL_V4_CALIBRATION_COMPLETE_KIND
        or complete.get("attestation") != attestation_path.name
        or complete.get("attestation_sha256") != attestation_identity["sha256"]
        or complete.get("attestation_fingerprint") != attestation_fingerprint
        or complete.get("passed") is not True
        or complete.get("authorizes_training") is not False
        or complete.get("training_started") is not False
    ):
        raise GovernedControllerError(
            "release calibration COMPLETE does not authenticate attestation"
        )
    attestor = _mapping(
        attestation.get("attestor"),
        label="release calibration attestor",
    )
    attestor_path = (project_root / "scripts/attest_v4_13m_calibration_release.py").resolve()
    if (
        _project_path(
            project_root,
            attestor.get("path"),
            label="release calibration attestor.path",
        )
        != attestor_path
        or not attestor_path.is_file()
        or attestor.get("sha256") != sha256_file(attestor_path)
    ):
        raise GovernedControllerError("release calibration attestor identity differs")
    formal_closure_binding = {
        "path": closure["path"],
        "manifest_sha256": closure["manifest"]["sha256"],
        "complete_sha256": closure["complete"]["sha256"],
        "bundle_fingerprint": closure["bundle_fingerprint"],
    }
    if _canonical_sha256(attestation.get("formal_closure")) != _canonical_sha256(
        formal_closure_binding
    ):
        raise GovernedControllerError(
            "release calibration attestation binds another formal closure"
        )
    closed_calibration_gate = _mapping(
        closure["closed_readiness"].get("calibration_gate"),
        label="closed readiness calibration_gate",
    )
    if (
        attestation.get("calibration_gate_contract_fingerprint")
        != _canonical_sha256(closed_calibration_gate)
    ):
        raise GovernedControllerError(
            "release calibration attestation binds another calibration gate"
        )
    config_binding = _mapping(
        binding.get("calibration_config"),
        label="release calibration config",
    )
    config_path = _project_path(
        project_root,
        config_binding.get("path"),
        label="release calibration config.path",
    )
    config_identity = _authenticated_file_identity(
        config_binding,
        expected_path=config_path,
        label="release calibration config",
    )
    if _canonical_sha256(attestation.get("calibration_config")) != _canonical_sha256(
        config_identity
    ):
        raise GovernedControllerError(
            "release calibration config differs from attestation"
        )
    evidence = _mapping(
        binding.get("evidence"),
        label="release calibration evidence",
    )
    attested_evidence = _mapping(
        attestation.get("evidence"),
        label="release calibration attested evidence",
    )
    if set(evidence) != set(attested_evidence) or set(evidence) != {
        "training_report_bundle",
        "checkpoint_validation_bundle",
        "checkpoint_drift_audit_bundle",
    }:
        raise GovernedControllerError(
            "release calibration evidence inventory differs"
        )
    authenticated_evidence: dict[str, Any] = {}
    for name in sorted(evidence):
        if _canonical_sha256(evidence[name]) != _canonical_sha256(
            attested_evidence[name]
        ):
            raise GovernedControllerError(
                f"release calibration evidence differs from attestation: {name}"
            )
        authenticated_evidence[name] = _authenticate_release_report_bundle(
            evidence[name],
            project_root=project_root,
            label=f"release calibration evidence.{name}",
        )
    raw_candidates = binding.get("candidate_checkpoints")
    attested_candidates = attestation.get("candidate_checkpoints")
    if (
        not isinstance(raw_candidates, list)
        or not raw_candidates
        or not isinstance(attested_candidates, list)
        or len(raw_candidates) != len(attested_candidates)
    ):
        raise GovernedControllerError(
            "release calibration candidate checkpoint inventory differs"
        )
    candidates: list[dict[str, Any]] = []
    for index, (row, attested) in enumerate(
        zip(raw_candidates, attested_candidates, strict=True)
    ):
        public = _mapping(
            row,
            label=f"release calibration candidate_checkpoints[{index}]",
        )
        source = _mapping(
            attested,
            label=f"release calibration attested candidate_checkpoints[{index}]",
        )
        for field in ("path", "manifest_sha256", "complete_sha256"):
            if public.get(field) != source.get(field):
                raise GovernedControllerError(
                    "release calibration candidate checkpoint differs from "
                    f"attestation: {index}"
                )
        checkpoint_path = _project_path(
            project_root,
            public.get("path"),
            label=f"release calibration candidate_checkpoints[{index}].path",
        )
        authenticated = authenticate_checkpoint(checkpoint_path)
        metadata = _mapping(
            authenticated.get("metadata"),
            label=f"release calibration candidate_checkpoints[{index}].metadata",
        )
        if (
            authenticated.get("manifest_sha256")
            != _sha256_string(
                public.get("manifest_sha256"),
                label=(
                    "release calibration "
                    f"candidate_checkpoints[{index}].manifest_sha256"
                ),
            )
            or authenticated.get("complete_sha256")
            != _sha256_string(
                public.get("complete_sha256"),
                label=(
                    "release calibration "
                    f"candidate_checkpoints[{index}].complete_sha256"
                ),
            )
            or any(
                public.get(field) != metadata.get(field)
                for field in ("global_step", "committed_tokens", "kind", "tag")
            )
        ):
            raise GovernedControllerError(
                f"release calibration candidate checkpoint identity differs: {index}"
            )
        candidates.append(
            {
                key: public.get(key)
                for key in (
                    "path",
                    "manifest_sha256",
                    "complete_sha256",
                    "global_step",
                    "committed_tokens",
                    "kind",
                    "tag",
                )
            }
        )
    final_checkpoint = _mapping(
        binding.get("final_checkpoint"),
        label="release calibration final_checkpoint",
    )
    if (
        _canonical_sha256(final_checkpoint) != _canonical_sha256(candidates[-1])
        or _canonical_sha256(attestation.get("final_checkpoint"))
        != _canonical_sha256(attested_candidates[-1])
    ):
        raise GovernedControllerError(
            "release calibration final checkpoint is not the last candidate"
        )
    evaluation = _mapping(
        binding.get("evaluation"),
        label="release calibration evaluation",
    )
    if evaluation.get("passed") is not True:
        raise GovernedControllerError("release calibration evaluation did not pass")
    return {
        "attestation": attestation_identity,
        "complete": complete_identity,
        "attestation_fingerprint": attestation_fingerprint,
        "attestor": {
            "path": str(attestor_path),
            "sha256": sha256_file(attestor_path),
        },
        "calibration_config": config_identity,
        "evidence": authenticated_evidence,
        "candidate_checkpoints": candidates,
        "final_checkpoint": dict(final_checkpoint),
        "evaluation": dict(evaluation),
    }


def _authenticate_formal_release_bundle(
    readiness_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    readiness_file = readiness_path.expanduser().resolve()
    root = readiness_file.parent
    expected_files = {
        FORMAL_V4_RELEASE_CONFIG_NAME,
        FORMAL_V4_RELEASE_READINESS_NAME,
        FORMAL_V4_RELEASE_MANIFEST_NAME,
        FORMAL_V4_RELEASE_COMPLETE_NAME,
    }
    _strict_directory_files(
        root,
        expected_files=expected_files,
        label="formal release bundle",
    )
    if readiness_file != root / FORMAL_V4_RELEASE_READINESS_NAME:
        raise GovernedControllerError(
            "launch-enabled readiness is not the formal release readiness"
        )
    config_path = root / FORMAL_V4_RELEASE_CONFIG_NAME
    manifest_path = root / FORMAL_V4_RELEASE_MANIFEST_NAME
    complete_path = root / FORMAL_V4_RELEASE_COMPLETE_NAME
    readiness = _read_json(readiness_file, label="formal release readiness")
    readiness_fingerprint = _validate_readiness_fingerprint(
        readiness,
        required=True,
    )
    manifest = _read_json(manifest_path, label="formal release MANIFEST")
    complete = _read_json(complete_path, label="formal release COMPLETE")
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "kind",
            "release_fingerprint",
            "release_contract",
            "acknowledgements",
            "files",
            "launch_enabled",
            "authorizes_training",
            "training_started",
            "web_profile_changed",
            "bundle_fingerprint",
        },
        label="formal release MANIFEST",
    )
    manifest_unsigned = {
        key: item
        for key, item in manifest.items()
        if key != "bundle_fingerprint"
    }
    bundle_fingerprint = _sha256_string(
        manifest.get("bundle_fingerprint"),
        label="formal release bundle_fingerprint",
    )
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != FORMAL_V4_RELEASE_BUNDLE_KIND
        or manifest.get("launch_enabled") is not True
        or manifest.get("authorizes_training") is not True
        or manifest.get("training_started") is not False
        or manifest.get("web_profile_changed") is not False
        or _canonical_sha256(manifest_unsigned) != bundle_fingerprint
    ):
        raise GovernedControllerError("formal release MANIFEST contract differs")
    _require_exact_keys(
        complete,
        {
            "schema_version",
            "kind",
            "manifest",
            "manifest_sha256",
            "bundle_fingerprint",
            "release_fingerprint",
            "launch_enabled",
            "authorizes_training",
            "training_started",
            "web_profile_changed",
        },
        label="formal release COMPLETE",
    )
    if (
        complete.get("schema_version") != SCHEMA_VERSION
        or complete.get("kind") != FORMAL_V4_RELEASE_COMPLETE_KIND
        or complete.get("manifest") != FORMAL_V4_RELEASE_MANIFEST_NAME
        or complete.get("manifest_sha256") != sha256_file(manifest_path)
        or complete.get("bundle_fingerprint") != bundle_fingerprint
        or complete.get("release_fingerprint") != manifest.get("release_fingerprint")
        or complete.get("launch_enabled") is not True
        or complete.get("authorizes_training") is not True
        or complete.get("training_started") is not False
        or complete.get("web_profile_changed") is not False
    ):
        raise GovernedControllerError(
            "formal release COMPLETE does not authenticate MANIFEST"
        )
    files = _mapping(manifest.get("files"), label="formal release MANIFEST.files")
    if set(files) != {
        FORMAL_V4_RELEASE_CONFIG_NAME,
        FORMAL_V4_RELEASE_READINESS_NAME,
    }:
        raise GovernedControllerError("formal release payload inventory differs")
    config_identity = _relative_file_identity(
        root,
        FORMAL_V4_RELEASE_CONFIG_NAME,
        files[FORMAL_V4_RELEASE_CONFIG_NAME],
        label="formal release config",
    )
    readiness_identity = _relative_file_identity(
        root,
        FORMAL_V4_RELEASE_READINESS_NAME,
        files[FORMAL_V4_RELEASE_READINESS_NAME],
        label="formal release readiness",
    )
    contract = _mapping(
        manifest.get("release_contract"),
        label="formal release contract",
    )
    _require_exact_keys(
        contract,
        {
            "schema_version",
            "kind",
            "project_root",
            "output",
            "publisher",
            "source_tree",
            "dependency_lock",
            "formal_closure",
            "capacity_attestation",
            "closed_readiness",
            "calibration_attestation",
            "gates",
            "wikipedia_license",
            "final_config",
            "launch_enabled_after_publish",
            "authorizes_training_after_publish",
            "training_started",
            "web_profile_changed",
        },
        label="formal release contract",
    )
    release_fingerprint = _sha256_string(
        manifest.get("release_fingerprint"),
        label="formal release release_fingerprint",
    )
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("kind") != FORMAL_V4_RELEASE_KIND
        or _canonical_sha256(contract) != release_fingerprint
        or contract.get("project_root") != str(project_root.resolve())
        or contract.get("launch_enabled_after_publish") is not True
        or contract.get("authorizes_training_after_publish") is not True
        or contract.get("training_started") is not False
        or contract.get("web_profile_changed") is not False
    ):
        raise GovernedControllerError("formal release contract/fingerprint differs")
    output = _mapping(contract.get("output"), label="formal release contract.output")
    _require_exact_keys(
        output,
        {
            "path",
            "config_path",
            "readiness_path",
            "manifest_path",
            "complete_path",
        },
        label="formal release contract.output",
    )
    expected_output = {
        "path": str(root),
        "config_path": str(config_path),
        "readiness_path": str(readiness_file),
        "manifest_path": str(manifest_path),
        "complete_path": str(complete_path),
    }
    if dict(output) != expected_output:
        raise GovernedControllerError("formal release output paths differ")
    publisher = _mapping(
        contract.get("publisher"),
        label="formal release publisher",
    )
    _require_exact_keys(publisher, {"path", "sha256"}, label="formal release publisher")
    publisher_path = (project_root / "scripts/publish_v4_250m_release.py").resolve()
    if (
        _project_path(
            project_root,
            publisher.get("path"),
            label="formal release publisher.path",
        )
        != publisher_path
        or not publisher_path.is_file()
        or publisher.get("sha256") != sha256_file(publisher_path)
    ):
        raise GovernedControllerError("formal release publisher identity differs")
    source_tree = _mapping(
        contract.get("source_tree"),
        label="formal release source_tree",
    )
    _require_exact_keys(
        source_tree,
        {"path", "sha256"},
        label="formal release source_tree",
    )
    source_tree_path = (project_root / "src/twen").resolve()
    if (
        _project_path(
            project_root,
            source_tree.get("path"),
            label="formal release source_tree.path",
        )
        != source_tree_path
        or source_tree.get("sha256") != twen_source_tree_sha256(source_tree_path)
    ):
        raise GovernedControllerError("formal release source-tree identity differs")
    dependency_path = project_root / "uv.lock"
    if not dependency_path.is_file():
        dependency_path = project_root / "pyproject.toml"
    dependency = _authenticated_file_identity(
        contract.get("dependency_lock"),
        expected_path=dependency_path,
        label="formal release dependency_lock",
    )
    closure = _authenticate_closure_release_binding(
        contract.get("formal_closure"),
        project_root=project_root,
    )
    capacity_binding = _authenticated_file_identity(
        contract.get("capacity_attestation"),
        expected_path=Path(closure["payloads"]["capacity-attestation.json"]["path"]),
        label="formal release capacity_attestation",
    )
    closed_readiness_binding = _authenticated_file_identity(
        contract.get("closed_readiness"),
        expected_path=Path(closure["payloads"]["readiness.json"]["path"]),
        label="formal release closed_readiness",
    )
    calibration = _authenticate_calibration_release_binding(
        contract.get("calibration_attestation"),
        project_root=project_root,
        closure=closure,
    )
    final_config = _mapping(
        contract.get("final_config"),
        label="formal release final_config",
    )
    _require_exact_keys(
        final_config,
        {"sha256", "normalized_semantic_sha256"},
        label="formal release final_config",
    )
    if (
        final_config.get("sha256") != config_identity["sha256"]
        or final_config.get("normalized_semantic_sha256")
        != FORMAL_V4_NORMALIZED_CONFIG_SHA256
    ):
        raise GovernedControllerError("formal release final config identity differs")
    acknowledgements = _mapping(
        manifest.get("acknowledgements"),
        label="formal release acknowledgements",
    )
    _require_exact_keys(
        acknowledgements,
        {"formal_release", "wikipedia_license"},
        label="formal release acknowledgements",
    )
    wikipedia = _mapping(
        contract.get("wikipedia_license"),
        label="formal release wikipedia_license",
    )
    _require_exact_keys(
        wikipedia,
        {"contract", "contract_fingerprint", "required_acknowledgement"},
        label="formal release wikipedia_license",
    )
    expected_wikipedia_ack = (
        f"ACCEPT V4 WIKIPEDIA LICENSE {FORMAL_V4_WIKIPEDIA_LICENSE_FINGERPRINT}"
    )
    expected_formal_ack = f"AUTHORIZE V4 {release_fingerprint}"
    if (
        _canonical_sha256(wikipedia.get("contract"))
        != _canonical_sha256(FORMAL_V4_WIKIPEDIA_LICENSE_CONTRACT)
        or wikipedia.get("contract_fingerprint")
        != FORMAL_V4_WIKIPEDIA_LICENSE_FINGERPRINT
        or wikipedia.get("required_acknowledgement") != expected_wikipedia_ack
        or acknowledgements.get("formal_release") != expected_formal_ack
        or acknowledgements.get("wikipedia_license") != expected_wikipedia_ack
    ):
        raise GovernedControllerError(
            "formal release acknowledgement contract differs"
        )
    release = _mapping(readiness.get("release"), label="readiness.release")
    _require_exact_keys(
        release,
        {
            "kind",
            "release_fingerprint",
            "publisher",
            "formal_release_acknowledgement",
            "wikipedia_license_acknowledgement",
            "web_profile_changed",
            "training_started",
        },
        label="readiness.release",
    )
    readiness_config = _mapping(
        readiness.get("config_identity"),
        label="readiness.config_identity",
    )
    if (
        release.get("kind") != FORMAL_V4_RELEASE_KIND
        or release.get("release_fingerprint") != release_fingerprint
        or _canonical_sha256(release.get("publisher")) != _canonical_sha256(publisher)
        or release.get("formal_release_acknowledgement") != expected_formal_ack
        or release.get("wikipedia_license_acknowledgement")
        != expected_wikipedia_ack
        or release.get("web_profile_changed") is not False
        or release.get("training_started") is not False
        or readiness.get("config_path") != str(config_path)
        or readiness.get("config_sha256") != config_identity["sha256"]
        or readiness_config.get("path") != str(config_path)
        or readiness_config.get("size") != config_identity["size"]
        or readiness_config.get("sha256") != config_identity["sha256"]
        or readiness_config.get("normalized_semantic_sha256")
        != FORMAL_V4_NORMALIZED_CONFIG_SHA256
        or readiness.get("status") != "authorized_for_governed_v4_250m_launch"
        or readiness.get("blockers") != []
        or readiness.get("launch_enabled") is not True
        or readiness.get("authorizes_training") is not True
        or readiness.get("training_started") is not False
    ):
        raise GovernedControllerError(
            "formal release readiness authorization differs"
        )
    return {
        "root": str(root),
        "manifest": _current_file_identity(
            manifest_path,
            label="formal release MANIFEST",
        ),
        "complete": _current_file_identity(
            complete_path,
            label="formal release COMPLETE",
        ),
        "config": config_identity,
        "readiness": readiness_identity,
        "readiness_fingerprint": readiness_fingerprint,
        "release_fingerprint": release_fingerprint,
        "bundle_fingerprint": bundle_fingerprint,
        "publisher": {
            "path": str(publisher_path),
            "sha256": sha256_file(publisher_path),
        },
        "source_tree": {
            "path": str(source_tree_path),
            "sha256": source_tree.get("sha256"),
        },
        "dependency_lock": dependency,
        "formal_closure": {
            key: item
            for key, item in closure.items()
            if key != "closed_readiness"
        },
        "capacity_attestation": capacity_binding,
        "closed_readiness": closed_readiness_binding,
        "calibration": calibration,
        "acknowledgements": dict(acknowledgements),
    }


def _release_gate_bindings(
    readiness: Mapping[str, Any],
    project_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Authenticate source-bound semantic and license release gates."""

    values: dict[str, Any] = {}
    issues: list[str] = []
    try:
        license_gate = _mapping(
            readiness.get("wikipedia_license_gate"),
            label="wikipedia_license_gate",
        )
        contract = _require_exact_contract(
            license_gate.get("contract"),
            FORMAL_V4_WIKIPEDIA_LICENSE_CONTRACT,
            label="wikipedia_license_gate.contract",
        )
        fingerprint = _sha256_string(
            license_gate.get("contract_fingerprint"),
            label="wikipedia_license_gate.contract_fingerprint",
        )
        expected_acknowledgement = (
            f"ACCEPT V4 WIKIPEDIA LICENSE {FORMAL_V4_WIKIPEDIA_LICENSE_FINGERPRINT}"
        )
        if (
            _canonical_sha256(contract)
            != FORMAL_V4_WIKIPEDIA_LICENSE_FINGERPRINT
            or fingerprint != FORMAL_V4_WIKIPEDIA_LICENSE_FINGERPRINT
            or license_gate.get("required") is not True
            or license_gate.get("required_acknowledgement")
            != expected_acknowledgement
        ):
            raise GovernedControllerError(
                "Wikipedia license gate differs from the source-bound policy"
            )
        accepted = (
            license_gate.get("status")
            == "accepted_explicit_user_acknowledgement"
            and license_gate.get("observed_acknowledgement")
            == expected_acknowledgement
            and license_gate.get("passed") is True
            and license_gate.get("authorizes_training") is True
        )
        if not accepted:
            issues.append(
                "wikipedia_license_gate does not contain the exact accepted acknowledgement"
            )
        values["wikipedia_license"] = {
            "contract": contract,
            "contract_fingerprint": fingerprint,
            "required_acknowledgement": expected_acknowledgement,
            "observed_acknowledgement": license_gate.get(
                "observed_acknowledgement"
            ),
            "passed": license_gate.get("passed") is True,
            "authorizes_training": (
                license_gate.get("authorizes_training") is True
            ),
        }
    except GovernedControllerError as exc:
        issues.append(str(exc))

    try:
        semantic_gate = _mapping(
            readiness.get("chinese_semantic_quality_gate"),
            label="chinese_semantic_quality_gate",
        )
        if (
            semantic_gate.get("required") is not True
            or semantic_gate.get("source_id")
            != FORMAL_V4_CHINESE_SEMANTIC_SOURCE_ID
            or _canonical_sha256(semantic_gate.get("required_bundle"))
            != _canonical_sha256(FORMAL_V4_CHINESE_SEMANTIC_REQUIRED_BUNDLE)
            or _canonical_sha256(semantic_gate.get("required_gates"))
            != _canonical_sha256(FORMAL_V4_CHINESE_SEMANTIC_REQUIRED_GATES)
        ):
            raise GovernedControllerError(
                "Chinese semantic quality gate differs from the source-bound policy"
            )
        accepted = (
            semantic_gate.get("status")
            == "passed_authenticated_chinese_semantic_quality_audit"
            and semantic_gate.get("passed") is True
            and semantic_gate.get("authorizes_training") is True
        )
        observed = semantic_gate.get("observed")
        if not accepted or not isinstance(observed, Mapping):
            issues.append(
                "chinese_semantic_quality_gate does not authorize training"
            )
        else:
            root = _project_path(
                project_root,
                observed.get("root"),
                label="chinese_semantic_quality_gate.observed.root",
            )
            manifest_identity = _authenticated_file_identity(
                observed.get("manifest"),
                expected_path=root / "MANIFEST.json",
                label="chinese_semantic_quality_gate.observed.manifest",
            )
            complete_identity = _authenticated_file_identity(
                observed.get("complete"),
                expected_path=root / "COMPLETE",
                label="chinese_semantic_quality_gate.observed.complete",
            )
            attestation_identity = _authenticated_file_identity(
                observed.get("attestation"),
                expected_path=root / "attestation.json",
                label="chinese_semantic_quality_gate.observed.attestation",
            )
            manifest = _read_json(
                Path(manifest_identity["path"]),
                label="Chinese semantic MANIFEST",
            )
            complete = _read_json(
                Path(complete_identity["path"]),
                label="Chinese semantic COMPLETE",
            )
            attestation = _read_json(
                Path(attestation_identity["path"]),
                label="Chinese semantic attestation",
            )
            files = _mapping(
                manifest.get("files"),
                label="Chinese semantic MANIFEST files",
            )
            required_files = {
                "attestation.json",
                "manual-review-template.json",
                "samples.jsonl",
            }
            if set(files) != required_files:
                raise GovernedControllerError(
                    "Chinese semantic MANIFEST file inventory differs"
                )
            for relative, raw_identity in files.items():
                identity = _mapping(
                    raw_identity,
                    label=f"Chinese semantic MANIFEST files.{relative}",
                )
                payload_path = (root / relative).resolve()
                payload_size = _integer(
                    identity.get("size"),
                    label=f"Chinese semantic MANIFEST files.{relative}.size",
                )
                payload_sha = _sha256_string(
                    identity.get("sha256"),
                    label=f"Chinese semantic MANIFEST files.{relative}.sha256",
                )
                if (
                    root.resolve() not in payload_path.parents
                    or not payload_path.is_file()
                    or payload_path.stat().st_size != payload_size
                    or sha256_file(payload_path) != payload_sha
                ):
                    raise GovernedControllerError(
                        f"Chinese semantic payload identity differs: {relative}"
                    )
            attestation_unsigned = {
                key: value
                for key, value in attestation.items()
                if key != "attestation_fingerprint"
            }
            attestation_fingerprint = _sha256_string(
                attestation.get("attestation_fingerprint"),
                label="Chinese semantic attestation fingerprint",
            )
            scanner = _mapping(
                attestation.get("scanner"),
                label="Chinese semantic scanner",
            )
            scanner_path = _project_path(
                project_root,
                scanner.get("path"),
                label="Chinese semantic scanner path",
            )
            expected_scanner = (
                project_root / "scripts/audit_v4_chinese_semantic_noise.py"
            ).resolve()
            if (
                manifest.get("schema_version") != SCHEMA_VERSION
                or manifest.get("kind")
                != FORMAL_V4_CHINESE_SEMANTIC_REQUIRED_BUNDLE["manifest_kind"]
                or manifest.get("passed") is not True
                or manifest.get("authorizes_training") is not False
                or complete.get("schema_version") != SCHEMA_VERSION
                or complete.get("kind")
                != FORMAL_V4_CHINESE_SEMANTIC_REQUIRED_BUNDLE["complete_kind"]
                or complete.get("manifest") != "MANIFEST.json"
                or complete.get("manifest_sha256")
                != manifest_identity["sha256"]
                or complete.get("passed") is not True
                or complete.get("authorizes_training") is not False
                or attestation.get("schema_version") != SCHEMA_VERSION
                or attestation.get("kind")
                != FORMAL_V4_CHINESE_SEMANTIC_REQUIRED_BUNDLE[
                    "attestation_kind"
                ]
                or attestation.get("source_id")
                != FORMAL_V4_CHINESE_SEMANTIC_SOURCE_ID
                or attestation.get("passed") is not True
                or attestation.get("authorizes_training") is not False
                or attestation.get("status")
                != "passed_quality_gate_but_does_not_authorize_training"
                or _canonical_sha256(attestation_unsigned)
                != attestation_fingerprint
                or manifest.get("attestation_fingerprint")
                != attestation_fingerprint
                or observed.get("attestation_fingerprint")
                != attestation_fingerprint
                or _canonical_sha256(attestation.get("gates"))
                != _canonical_sha256(
                    FORMAL_V4_CHINESE_SEMANTIC_AUDIT_GATES
                )
                or scanner_path != expected_scanner
                or not scanner_path.is_file()
                or scanner.get("sha256") != sha256_file(scanner_path)
                or _canonical_sha256(scanner.get("policy"))
                != _canonical_sha256(
                    FORMAL_V4_CHINESE_SEMANTIC_SCANNER_POLICY
                )
            ):
                raise GovernedControllerError(
                    "Chinese semantic audit bundle differs from the "
                    "source-bound release contract"
                )
            samples = _mapping(
                attestation.get("samples"),
                label="Chinese semantic samples",
            )
            risk_sample_size = _integer(
                samples.get("risk_samples_per_phase"),
                label="Chinese semantic risk sample quota",
                minimum=FORMAL_V4_CHINESE_SEMANTIC_REQUIRED_GATES[
                    "risk_samples_per_phase_gte"
                ],
            )
            control_sample_size = _integer(
                samples.get("control_samples_per_phase"),
                label="Chinese semantic control sample quota",
                minimum=FORMAL_V4_CHINESE_SEMANTIC_REQUIRED_GATES[
                    "control_samples_per_phase_gte"
                ],
            )
            inputs = _mapping(
                attestation.get("inputs"),
                label="Chinese semantic inputs",
            )
            phase_statistics = _mapping(
                attestation.get("phases"),
                label="Chinese semantic phase statistics",
            )
            if set(inputs) != {"primary", "cooldown"} or set(
                phase_statistics
            ) != {"primary", "cooldown"}:
                raise GovernedControllerError(
                    "Chinese semantic phase inventory differs"
                )
            scanner_module = _load_authenticated_script(
                scanner_path,
                module_name="_twen_governed_chinese_semantic_scanner",
                label="Chinese semantic scanner",
            )
            scanner_sha_before = sha256_file(scanner_path)
            try:
                recomputed = scanner_module.recompute_scan(
                    primary_manifest=_project_path(
                        project_root,
                        _mapping(
                            inputs["primary"],
                            label="Chinese semantic primary input",
                        ).get("path"),
                        label="Chinese semantic primary manifest",
                    ),
                    cooldown_manifest=_project_path(
                        project_root,
                        _mapping(
                            inputs["cooldown"],
                            label="Chinese semantic cooldown input",
                        ).get("path"),
                        label="Chinese semantic cooldown manifest",
                    ),
                    source_id=FORMAL_V4_CHINESE_SEMANTIC_SOURCE_ID,
                    risk_sample_size=risk_sample_size,
                    control_sample_size=control_sample_size,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise GovernedControllerError(
                    "Chinese semantic deterministic recomputation failed: "
                    f"{exc}"
                ) from exc
            if (
                sha256_file(scanner_path) != scanner_sha_before
                or scanner_sha_before != scanner.get("sha256")
            ):
                raise GovernedControllerError(
                    "Chinese semantic scanner changed during recomputation"
                )
            recomputed_samples = recomputed.get("samples")
            recomputed_payload = recomputed.get("samples_payload")
            samples_path = root / "samples.jsonl"
            if (
                _canonical_sha256(recomputed.get("inputs"))
                != _canonical_sha256(inputs)
                or _canonical_sha256(recomputed.get("phases"))
                != _canonical_sha256(phase_statistics)
                or not isinstance(recomputed_samples, list)
                or not isinstance(recomputed_payload, bytes)
                or samples.get("path") != "samples.jsonl"
                or samples.get("size") != len(recomputed_payload)
                or samples.get("sha256") != recomputed.get("samples_sha256")
                or samples.get("count") != len(recomputed_samples)
                or samples_path.read_bytes() != recomputed_payload
            ):
                raise GovernedControllerError(
                    "Chinese semantic scan or samples differ from deterministic "
                    "recomputation"
                )
            expected_template = scanner_module._manual_template(
                samples_sha256=str(recomputed["samples_sha256"]),
                samples=recomputed_samples,
            )
            actual_template = _read_json(
                root / "manual-review-template.json",
                label="Chinese semantic manual-review template",
            )
            if _canonical_sha256(actual_template) != _canonical_sha256(
                expected_template
            ):
                raise GovernedControllerError(
                    "Chinese semantic manual-review template differs from "
                    "deterministic recomputation"
                )
            manual = _mapping(
                attestation.get("manual_review"),
                label="Chinese semantic manual review",
            )
            manual_path = _project_path(
                project_root,
                manual.get("path"),
                label="Chinese semantic manual review path",
            )
            try:
                recomputed_manual = scanner_module._load_manual_decisions(
                    manual_path,
                    sample_sha256=str(recomputed["samples_sha256"]),
                    samples=recomputed_samples,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise GovernedControllerError(
                    f"Chinese semantic manual review failed recomputation: {exc}"
                ) from exc
            recomputed_manual["status"] = (
                "passed" if recomputed_manual.get("passed") is True else "failed"
            )
            if (
                _canonical_sha256(recomputed_manual)
                != _canonical_sha256(manual)
                or recomputed_manual.get("passed") is not True
                or not manual_path.is_file()
                or manual.get("size") != manual_path.stat().st_size
                or manual.get("sha256") != sha256_file(manual_path)
                or manual.get("reviewed_samples") != samples.get("count")
                or manual.get("unacceptable_samples") != 0
                or manual.get("unacceptable_rate") != 0
            ):
                raise GovernedControllerError(
                    "Chinese semantic manual review does not pass"
                )
            values["chinese_semantic_quality"] = {
                "root": str(root),
                "manifest": manifest_identity,
                "complete": complete_identity,
                "attestation": attestation_identity,
                "attestation_fingerprint": attestation_fingerprint,
                "scanner": {
                    "path": str(scanner_path),
                    "sha256": scanner.get("sha256"),
                },
                "inputs": {
                    phase: dict(
                        _mapping(
                            recomputed["inputs"][phase],
                            label=f"recomputed Chinese semantic {phase} input",
                        )
                    )
                    for phase in ("primary", "cooldown")
                },
                "passed": True,
                "authorizes_training": True,
            }
    except GovernedControllerError as exc:
        issues.append(str(exc))
    return values, issues


def _authenticate_output_bundle(
    root: Path,
    *,
    expected_manifest_sha256: Any,
    expected_complete_sha256: Any,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = root / "MANIFEST.json"
    complete_path = root / "COMPLETE"
    summary_path = root / "summary.json"
    if not all(path.is_file() for path in (manifest_path, complete_path, summary_path)):
        raise GovernedControllerError(f"{label} is incomplete: {root}")
    manifest_sha = sha256_file(manifest_path)
    complete_sha = sha256_file(complete_path)
    if manifest_sha != _sha256_string(
        expected_manifest_sha256,
        label=f"{label}.manifest_sha256",
    ):
        raise GovernedControllerError(f"{label} MANIFEST identity differs from readiness")
    if complete_sha != _sha256_string(
        expected_complete_sha256,
        label=f"{label}.complete_sha256",
    ):
        raise GovernedControllerError(f"{label} COMPLETE identity differs from readiness")
    try:
        authenticated_manifest_sha = complete_path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise GovernedControllerError(f"cannot read {label} COMPLETE: {exc}") from exc
    if authenticated_manifest_sha != manifest_sha:
        raise GovernedControllerError(f"{label} COMPLETE does not authenticate MANIFEST")
    manifest = _read_json(manifest_path, label=f"{label} MANIFEST")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or "summary.json" not in files:
        raise GovernedControllerError(f"{label} MANIFEST has no summary inventory")
    for relative, raw_identity in files.items():
        if (
            not isinstance(relative, str)
            or not isinstance(raw_identity, Mapping)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise GovernedControllerError(f"{label} contains an unsafe output identity")
        path = (root / relative).resolve()
        if root.resolve() not in path.parents:
            raise GovernedControllerError(f"{label} output escapes its root")
        if (
            not path.is_file()
            or path.stat().st_size
            != _integer(
                raw_identity.get("size"),
                label=f"{label}.{relative}.size",
            )
            or sha256_file(path)
            != _sha256_string(
                raw_identity.get("sha256"),
                label=f"{label}.{relative}.sha256",
            )
        ):
            raise GovernedControllerError(f"{label} output identity differs: {relative}")
    return manifest, _read_json(summary_path, label=f"{label} summary")


def _formal_bindings(
    readiness: Mapping[str, Any],
    project_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    formal = _mapping(readiness.get("formal_validation_gate"), label="formal_validation_gate")
    baseline = _mapping(
        formal.get("v3_final_frozen_validation_baseline"),
        label="formal_validation_gate.v3_final_frozen_validation_baseline",
    )
    disjointness = _mapping(
        formal.get("train_validation_union_disjointness"),
        label="formal_validation_gate.train_validation_union_disjointness",
    )
    values: dict[str, Any] = {"validation_phases": {}}
    issues: list[str] = []
    disjoint_binding, disjoint_issue = _optional_binding(
        project_root,
        disjointness.get("attestation_path"),
        label="formal_disjointness_attestation",
    )
    values["formal_disjointness_attestation"] = disjoint_binding
    if disjoint_issue is not None:
        issues.append(disjoint_issue)
    elif disjoint_binding is not None:
        expected = disjointness.get("attestation_sha256")
        if expected is None or disjoint_binding["sha256"] != expected:
            issues.append("formal disjointness attestation identity differs from readiness")
        else:
            try:
                attestation = _read_json(
                    Path(str(disjoint_binding["path"])),
                    label="formal disjointness attestation",
                )
                unsigned = {
                    key: value
                    for key, value in attestation.items()
                    if key != "attestation_fingerprint"
                }
                fingerprint = _sha256_string(
                    attestation.get("attestation_fingerprint"),
                    label="formal_disjointness.attestation_fingerprint",
                )
                if (
                    attestation.get("kind")
                    != "twen_v4_formal_validation_disjointness_attestation"
                    or attestation.get("schema_version") != 1
                    or attestation.get("passed") is not True
                    or attestation.get("near_duplicate_threshold") != 0.8
                    or fingerprint != _canonical_sha256(unsigned)
                    or disjointness.get("attestation_fingerprint") != fingerprint
                ):
                    raise GovernedControllerError(
                        "formal disjointness attestation contract is invalid"
                    )
                raw_gates = _mapping(
                    attestation.get("gates"),
                    label="formal_disjointness.gates",
                )
                expected_gates = {
                    "train_validation_stable_id": "stable_id_exact",
                    "train_validation_normalized_exact": "normalized_text_exact",
                    "train_validation_near_duplicate": "near_duplicate",
                    "validation_internal_stable_id": "stable_id_exact",
                    "validation_internal_normalized_exact": "normalized_text_exact",
                    "validation_internal_near_duplicate": "near_duplicate",
                }
                if set(raw_gates) != set(expected_gates):
                    raise GovernedControllerError(
                        "formal disjointness gate inventory differs"
                    )
                for gate_name, algorithm_name in expected_gates.items():
                    gate = _mapping(
                        raw_gates.get(gate_name),
                        label=f"formal_disjointness.gates.{gate_name}",
                    )
                    if (
                        gate.get("algorithm")
                        != FORMAL_V4_DISJOINTNESS_ALGORITHMS[algorithm_name]
                        or gate.get("matches") != 0
                        or gate.get("passed") is not True
                        or (
                            algorithm_name == "near_duplicate"
                            and gate.get("estimated_jaccard_threshold") != 0.8
                        )
                    ):
                        raise GovernedControllerError(
                            f"formal disjointness gate failed: {gate_name}"
                        )
                phase_identity = _mapping(
                    attestation.get("phase_train_disjointness"),
                    label="formal_disjointness.phase_train_disjointness",
                )
                values["formal_phase_train_disjointness"] = {
                    "path": str(
                        _project_path(
                            project_root,
                            phase_identity.get("path"),
                            label="formal phase_train_disjointness.path",
                        )
                    ),
                    "sha256": _sha256_string(
                        phase_identity.get("sha256"),
                        label="formal phase_train_disjointness.sha256",
                    ),
                    "attestation_fingerprint": _sha256_string(
                        phase_identity.get("attestation_fingerprint"),
                        label=(
                            "formal phase_train_disjointness."
                            "attestation_fingerprint"
                        ),
                    ),
                }
                values["formal_disjointness_attestation"] = {
                    **disjoint_binding,
                    "attestation_fingerprint": fingerprint,
                }
            except GovernedControllerError as exc:
                issues.append(str(exc))
    if disjointness.get("near_duplicate_threshold") != 0.8:
        issues.append("formal disjointness near-duplicate threshold differs")
    for field in (
        "stable_id_exact_passed",
        "normalized_text_exact_passed",
        "near_duplicate_passed",
    ):
        if disjointness.get(field) is not True:
            issues.append(f"formal disjointness gate is not passing: {field}")

    bundle_raw = baseline.get("bundle_path")
    if not isinstance(bundle_raw, str) or not bundle_raw:
        values["v3_formal_baseline_bundle"] = None
        issues.append("v3_formal_baseline_bundle is not bound")
        return values, issues
    bundle_root = _project_path(
        project_root,
        bundle_raw,
        label="v3_formal_baseline_bundle",
    )
    values["v3_formal_baseline_bundle"] = {
        "path": str(bundle_root),
        "manifest_sha256": baseline.get("manifest_sha256"),
        "complete_sha256": baseline.get("complete_sha256"),
    }
    try:
        if (
            baseline.get("passed") is not True
            or baseline.get("checkpoint_complete_sha256")
            != FORMAL_V4_FORK_COMPLETE_SHA256
        ):
            raise GovernedControllerError(
                "v3 formal baseline is not the source-bound final checkpoint"
            )
        _manifest, summary = _authenticate_output_bundle(
            bundle_root,
            expected_manifest_sha256=baseline.get("manifest_sha256"),
            expected_complete_sha256=baseline.get("complete_sha256"),
            label="v3 formal baseline bundle",
        )
        if summary.get("kind") != "twen_v4_formal_frozen_validation_baseline":
            raise GovernedControllerError("v3 formal baseline summary kind differs")
        phase_rows = summary.get("phases")
        if not isinstance(phase_rows, list):
            raise GovernedControllerError("v3 formal baseline phase inventory is invalid")
        rows = {
            row.get("phase"): row for row in phase_rows if isinstance(row, Mapping)
        }
        if set(rows) != {"primary", "cooldown"}:
            raise GovernedControllerError(
                "v3 formal baseline must contain primary and cooldown phases"
            )
        for phase, row in rows.items():
            prepared = _mapping(row.get("prepared"), label=f"{phase}.prepared")
            evaluation = _mapping(row.get("evaluation"), label=f"{phase}.evaluation")
            prepared_path = _project_path(
                project_root,
                prepared.get("path"),
                label=f"{phase}.prepared.path",
            )
            prepared_sha = _sha256_string(
                prepared.get("sha256"),
                label=f"{phase}.prepared.sha256",
            )
            if not prepared_path.is_file() or sha256_file(prepared_path) != prepared_sha:
                raise GovernedControllerError(
                    f"{phase} formal prepared manifest identity differs"
                )
            evaluation_root = _project_path(
                project_root,
                evaluation.get("root"),
                label=f"{phase}.evaluation.root",
            )
            values["validation_phases"][phase] = {
                "prepared": {
                    "path": str(prepared_path),
                    "sha256": prepared_sha,
                    "dataset_fingerprint": prepared.get("dataset_fingerprint"),
                },
                "v3_evaluation": {
                    "path": str(evaluation_root),
                    "manifest": dict(
                        _mapping(
                            evaluation.get("manifest"),
                            label=f"{phase}.evaluation.manifest",
                        )
                    ),
                    "complete": dict(
                        _mapping(
                            evaluation.get("complete"),
                            label=f"{phase}.evaluation.complete",
                        )
                    ),
                    "plan": dict(
                        _mapping(
                            evaluation.get("plan"),
                            label=f"{phase}.evaluation.plan",
                        )
                    ),
                },
                "comparison_contract": dict(
                    _mapping(
                        row.get("comparison_contract"),
                        label=f"{phase}.comparison_contract",
                    )
                ),
            }
    except GovernedControllerError as exc:
        issues.append(str(exc))
    return values, issues


def _capacity_source_map_bindings(
    readiness: Mapping[str, Any],
    project_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Authenticate the closed capacity record without rehashing tensor payloads."""

    raw_binding = readiness.get("capacity_attestation")
    if not isinstance(raw_binding, Mapping):
        return {}, ["closed capacity attestation is not bound"]
    issues: list[str] = []
    try:
        path = _project_path(
            project_root,
            raw_binding.get("path"),
            label="capacity_attestation.path",
        )
        expected_size = _integer(
            raw_binding.get("size"),
            label="capacity_attestation.size",
            minimum=1,
        )
        expected_sha = _sha256_string(
            raw_binding.get("sha256"),
            label="capacity_attestation.sha256",
        )
        if (
            not path.is_file()
            or path.stat().st_size != expected_size
            or sha256_file(path) != expected_sha
        ):
            raise GovernedControllerError(
                "closed capacity attestation identity differs from readiness"
            )
        capacity = _read_json(path, label="closed capacity attestation")
        unsigned = {
            key: value
            for key, value in capacity.items()
            if key != "attestation_fingerprint"
        }
        fingerprint = _sha256_string(
            capacity.get("attestation_fingerprint"),
            label="capacity_attestation.attestation_fingerprint",
        )
        overall = _mapping(
            capacity.get("overall"),
            label="capacity_attestation.overall",
        )
        if (
            capacity.get("kind") != "twen_v4_250m_capacity_attestation"
            or capacity.get("schema_version") != 1
            or overall.get("passed") is not True
            or capacity.get("launch_enabled") is not False
            or capacity.get("authorizes_training") is not False
            or fingerprint != _canonical_sha256(unsigned)
            or raw_binding.get("attestation_fingerprint") != fingerprint
            or raw_binding.get("passed") is not True
        ):
            raise GovernedControllerError(
                "closed capacity attestation contract is invalid"
            )
        _require_exact_contract(
            capacity.get("training_contract"),
            FORMAL_V4_ATTESTED_CONTRACT,
            label="capacity_attestation.training_contract",
        )
        stages = _mapping(capacity.get("stages"), label="capacity_attestation.stages")
        if set(stages) != {"primary", "cooldown"}:
            raise GovernedControllerError(
                "closed capacity attestation must bind both data phases"
            )
        phase_bindings: dict[str, Any] = {}
        source_mixes: dict[str, dict[str, int]] = {}
        for phase in ("primary", "cooldown"):
            stage = _mapping(stages.get(phase), label=f"capacity.{phase}")
            expected_tokens = (
                FORMAL_V4_ATTESTED_CONTRACT["primary_tokens"]
                if phase == "primary"
                else FORMAL_V4_ATTESTED_CONTRACT["cooldown_tokens"]
            )
            if (
                stage.get("stage") != phase
                or stage.get("required_training_tokens") != expected_tokens
                or stage.get("required_tail_batch_tokens")
                != FORMAL_V4_ATTESTED_CONTRACT["global_batch_tokens"]
            ):
                raise GovernedControllerError(
                    f"closed capacity {phase} token contract differs"
                )
            expected_mix = (
                FORMAL_V4_PRIMARY_SOURCE_MIX
                if phase == "primary"
                else FORMAL_V4_COOLDOWN_SOURCE_MIX
            )
            raw_rows = stage.get("per_source_capacity")
            if not isinstance(raw_rows, list):
                raise GovernedControllerError(
                    f"closed capacity {phase} source inventory is invalid"
                )
            stage_mix: dict[str, int] = {}
            for index, raw_row in enumerate(raw_rows):
                row = _mapping(raw_row, label=f"capacity.{phase}.sources[{index}]")
                source_id = row.get("source_id")
                if not isinstance(source_id, str) or not source_id:
                    raise GovernedControllerError(
                        f"closed capacity {phase} source ID is invalid"
                    )
                if source_id in stage_mix:
                    raise GovernedControllerError(
                        f"closed capacity {phase} repeats source {source_id}"
                    )
                stage_mix[source_id] = _integer(
                    row.get("mix_basis_points"),
                    label=f"capacity.{phase}.{source_id}.mix_basis_points",
                    minimum=1,
                )
            if _canonical_sha256(stage_mix) != _canonical_sha256(expected_mix):
                raise GovernedControllerError(
                    f"closed capacity {phase} source mix differs from formal policy"
                )
            source_mixes[phase] = dict(sorted(stage_mix.items()))
            extracted = _mapping(
                stage.get("extracted_identity"),
                label=f"capacity.{phase}.extracted_identity",
            )
            extracted_manifest_path = _project_path(
                project_root,
                extracted.get("manifest_path"),
                label=f"capacity.{phase}.extracted_manifest",
            )
            extracted_manifest_sha = _sha256_string(
                extracted.get("manifest_sha256"),
                label=f"capacity.{phase}.extracted_manifest_sha256",
            )
            extracted_corpus_fingerprint = _sha256_string(
                extracted.get("corpus_fingerprint"),
                label=f"capacity.{phase}.extracted_corpus_fingerprint",
            )
            if (
                not extracted_manifest_path.is_file()
                or sha256_file(extracted_manifest_path)
                != extracted_manifest_sha
            ):
                raise GovernedControllerError(
                    f"closed capacity {phase} extracted identity differs"
                )
            prepared = _mapping(
                stage.get("prepared_identity"),
                label=f"capacity.{phase}.prepared_identity",
            )
            manifest_path = _project_path(
                project_root,
                prepared.get("manifest_path"),
                label=f"capacity.{phase}.prepared_manifest",
            )
            manifest_sha = _sha256_string(
                prepared.get("manifest_sha256"),
                label=f"capacity.{phase}.prepared_manifest_sha256",
            )
            if (
                stage.get("passed") is not True
                or prepared.get("passed") is not True
                or not manifest_path.is_file()
                or sha256_file(manifest_path) != manifest_sha
            ):
                raise GovernedControllerError(
                    f"closed capacity {phase} prepared identity differs"
                )
            phase_bindings[phase] = {
                "extracted_manifest_path": str(extracted_manifest_path),
                "extracted_manifest_sha256": extracted_manifest_sha,
                "extracted_corpus_fingerprint": extracted_corpus_fingerprint,
                "prepared_manifest_path": str(manifest_path),
                "prepared_manifest_sha256": manifest_sha,
                "prepared_dataset_fingerprint": _sha256_string(
                    prepared.get("dataset_fingerprint"),
                    label=f"capacity.{phase}.dataset_fingerprint",
                ),
                "source_map_sha256": _sha256_string(
                    prepared.get("source_map_sha256"),
                    label=f"capacity.{phase}.source_map_sha256",
                ),
            }

        phase_gates = _mapping(
            capacity.get("phase_disjointness"),
            label="capacity_attestation.phase_disjointness",
        )
        if set(phase_gates) != {
            "stable_id_exact",
            "normalized_text_exact",
            "near_duplicate",
        }:
            raise GovernedControllerError(
                "closed capacity phase-disjointness gate inventory differs"
            )
        for name, raw_gate in phase_gates.items():
            gate = _mapping(
                raw_gate,
                label=f"capacity_attestation.phase_disjointness.{name}",
            )
            if (
                gate.get("algorithm")
                != FORMAL_V4_DISJOINTNESS_ALGORITHMS[name]
                or gate.get("passed") is not True
                or gate.get("result") != 0
            ):
                raise GovernedControllerError(
                    f"closed capacity phase-disjointness gate failed: {name}"
                )
            if name == "near_duplicate" and gate.get("threshold") != 0.8:
                raise GovernedControllerError(
                    "closed capacity near-duplicate threshold differs"
                )
        phase_identity = _mapping(
            capacity.get("phase_disjointness_attestation"),
            label="capacity_attestation.phase_disjointness_attestation",
        )
        phase_path = _project_path(
            project_root,
            phase_identity.get("path"),
            label="capacity_attestation.phase_disjointness_attestation.path",
        )
        phase_sha = _sha256_string(
            phase_identity.get("sha256"),
            label="capacity_attestation.phase_disjointness_attestation.sha256",
        )
        if (
            phase_identity.get("passed") is not True
            or not phase_path.is_file()
            or sha256_file(phase_path) != phase_sha
        ):
            raise GovernedControllerError(
                "closed capacity phase-disjointness identity differs"
            )
        phase_attestation = _read_json(
            phase_path,
            label="phase-disjointness attestation",
        )
        phase_unsigned = {
            key: value
            for key, value in phase_attestation.items()
            if key != "attestation_fingerprint"
        }
        phase_fingerprint = _sha256_string(
            phase_attestation.get("attestation_fingerprint"),
            label="phase_disjointness.attestation_fingerprint",
        )
        if (
            phase_attestation.get("kind")
            != "twen_v4_phase_disjointness_attestation"
            or phase_attestation.get("schema_version") != 1
            or phase_attestation.get("passed") is not True
            or phase_fingerprint != _canonical_sha256(phase_unsigned)
            or phase_identity.get("attestation_fingerprint")
            != phase_fingerprint
        ):
            raise GovernedControllerError(
                "phase-disjointness attestation contract is invalid"
            )
        attested_gates = _mapping(
            phase_attestation.get("gates"),
            label="phase_disjointness.gates",
        )
        if set(attested_gates) != set(FORMAL_V4_DISJOINTNESS_ALGORITHMS):
            raise GovernedControllerError(
                "phase-disjointness attestation gate inventory differs"
            )
        for name, algorithm in FORMAL_V4_DISJOINTNESS_ALGORITHMS.items():
            gate = _mapping(
                attested_gates.get(name),
                label=f"phase_disjointness.gates.{name}",
            )
            if (
                gate.get("algorithm") != algorithm
                or gate.get("matches") != 0
                or gate.get("passed") is not True
                or (
                    name == "near_duplicate"
                    and gate.get("estimated_jaccard_threshold") != 0.8
                )
            ):
                raise GovernedControllerError(
                    f"phase-disjointness attestation gate failed: {name}"
                )
        for phase in ("primary", "cooldown"):
            phase_value = _mapping(
                phase_attestation.get(phase),
                label=f"phase_disjointness.{phase}",
            )
            prepared = _mapping(
                phase_value.get("prepared"),
                label=f"phase_disjointness.{phase}.prepared",
            )
            binding = phase_bindings[phase]
            if (
                Path(str(prepared.get("manifest_path"))).resolve()
                != Path(str(binding["prepared_manifest_path"])).resolve()
                or prepared.get("manifest_sha256")
                != binding["prepared_manifest_sha256"]
                or prepared.get("dataset_fingerprint")
                != binding["prepared_dataset_fingerprint"]
                or prepared.get("source_map_sha256")
                != binding["source_map_sha256"]
            ):
                raise GovernedControllerError(
                    f"phase-disjointness {phase} prepared identity differs"
                )
        return {
            "path": str(path),
            "sha256": expected_sha,
            "training_contract": dict(FORMAL_V4_ATTESTED_CONTRACT),
            "phases": phase_bindings,
            "source_mix_basis_points": source_mixes,
            "phase_disjointness_attestation": {
                "path": str(phase_path),
                "sha256": phase_sha,
                "attestation_fingerprint": phase_fingerprint,
            },
        }, issues
    except GovernedControllerError as exc:
        issues.append(str(exc))
        return {}, issues


def build_governed_plan(readiness_path: str | Path) -> dict[str, Any]:
    """Authenticate a readiness lock and return its immutable controller plan."""

    raw_readiness_file = Path(readiness_path).expanduser()
    if raw_readiness_file.is_symlink():
        raise GovernedControllerError("readiness path must not be a symlink")
    readiness_file = raw_readiness_file.resolve()
    readiness = _read_json(readiness_file, label="readiness")
    if readiness.get("kind") != "twen_v4_250m_pilot_readiness":
        raise GovernedControllerError("unsupported readiness kind")
    _integer(readiness.get("schema_version"), label="readiness.schema_version", minimum=1)
    project_root = _readiness_project_root(readiness_file, readiness)
    launch_requested = (
        readiness.get("launch_enabled") is True
        or readiness.get("authorizes_training") is True
    )
    readiness_fingerprint = _validate_readiness_fingerprint(
        readiness,
        required=launch_requested,
    )
    release_bundle: dict[str, Any] | None = None
    if launch_requested:
        release_bundle = _authenticate_formal_release_bundle(
            readiness_file,
            project_root=project_root,
        )
        if (
            release_bundle.get("readiness_fingerprint")
            != readiness_fingerprint
            or release_bundle.get("readiness", {}).get("sha256")
            != sha256_file(readiness_file)
        ):
            raise GovernedControllerError(
                "formal release readiness changed during plan construction"
            )

    config_path = _project_path(
        project_root,
        readiness.get("config_path"),
        label="readiness.config_path",
    )
    if not config_path.is_file():
        raise GovernedControllerError(f"readiness config does not exist: {config_path}")
    expected_config_sha = _sha256_string(
        readiness.get("config_sha256"),
        label="readiness.config_sha256",
    )
    actual_config_sha = sha256_file(config_path)
    if actual_config_sha != expected_config_sha:
        raise GovernedControllerError(
            "readiness config SHA256 mismatch: "
            f"expected {expected_config_sha}, got {actual_config_sha}"
        )

    contract = _require_exact_contract(
        readiness.get("contract"),
        FORMAL_V4_ATTESTED_CONTRACT,
        label="readiness.contract",
    )
    policy = FORMAL_V4_CONFIG_POLICY
    max_tokens = _integer(contract.get("max_tokens"), label="contract.max_tokens", minimum=1)
    world_size = _integer(contract.get("world_size"), label="contract.world_size", minimum=1)
    stage = contract.get("stage")
    if stage not in {"dense-oracle", "sparse"}:
        raise GovernedControllerError("contract.stage is invalid")
    pause_policy = _mapping(
        readiness.get("pause_evaluation_policy"),
        label="readiness.pause_evaluation_policy",
    )
    raw_thresholds = pause_policy.get("pause_at_committed_tokens")
    if not isinstance(raw_thresholds, list) or not raw_thresholds:
        raise GovernedControllerError("pause thresholds must be a non-empty list")
    thresholds = tuple(
        _integer(value, label=f"pause threshold {index}", minimum=1)
        for index, value in enumerate(raw_thresholds)
    )
    if tuple(sorted(set(thresholds))) != thresholds:
        raise GovernedControllerError("pause thresholds must be strictly increasing and unique")
    if thresholds[-1] != max_tokens:
        raise GovernedControllerError("the final pause threshold must equal max_tokens")
    if thresholds != FORMAL_V4_PAUSE_THRESHOLDS:
        raise GovernedControllerError(
            "pause thresholds differ from the source-bound formal v4 policy"
        )
    checkpoint_every_steps = _integer(
        pause_policy.get("checkpoint_every_steps"),
        label="pause_policy.checkpoint_every_steps",
        minimum=1,
    )
    if checkpoint_every_steps != policy["checkpoint_every_steps"]:
        raise GovernedControllerError(
            "pause-policy checkpoint interval differs from the formal contract"
        )

    issues = [
        blocker
        for blocker in readiness.get("blockers", [])
        if isinstance(blocker, str) and blocker
    ]
    if readiness.get("launch_enabled") is not True:
        issues.append("readiness launch_enabled is not true")
    if readiness.get("authorizes_training") is not True:
        issues.append("readiness authorizes_training is not true")
    for gate_name in ("calibration_gate", "formal_validation_gate"):
        gate = _mapping(readiness.get(gate_name), label=gate_name)
        if gate.get("passed") is not True or gate.get("authorizes_training") is not True:
            issues.append(f"{gate_name} does not authorize training")
    if pause_policy.get("controller_implemented") is not True:
        issues.append("readiness does not attest the governed controller implementation")
    else:
        controller_identity = readiness.get("governed_controller")
        if not isinstance(controller_identity, Mapping):
            issues.append("readiness has no governed controller source identity")
        else:
            controller_path = _project_path(
                project_root,
                controller_identity.get("path"),
                label="readiness.governed_controller.path",
            )
            expected_controller_path = (
                project_root / "scripts/govern_v4_training.py"
            ).resolve()
            if (
                controller_identity.get("implemented") is not True
                or controller_path != expected_controller_path
                or not controller_path.is_file()
                or sha256_file(controller_path)
                != controller_identity.get("sha256")
                or controller_identity.get("twen_source_tree_sha256")
                != twen_source_tree_sha256(project_root / "src/twen")
            ):
                issues.append(
                    "readiness governed controller source identity differs"
                )

    release_bindings, release_issues = _release_gate_bindings(
        readiness,
        project_root,
    )
    issues.extend(release_issues)
    formal_bindings, formal_issues = _formal_bindings(readiness, project_root)
    issues.extend(formal_issues)
    capacity_bindings, capacity_issues = _capacity_source_map_bindings(
        readiness,
        project_root,
    )
    issues.extend(capacity_issues)
    phase_source_maps = capacity_bindings.get("phases", {})
    if not isinstance(phase_source_maps, Mapping):
        phase_source_maps = {}
    semantic_binding = release_bindings.get("chinese_semantic_quality")
    semantic_inputs = (
        semantic_binding.get("inputs")
        if isinstance(semantic_binding, Mapping)
        else None
    )
    if isinstance(semantic_inputs, Mapping):
        for phase in ("primary", "cooldown"):
            semantic_input = semantic_inputs.get(phase)
            capacity_phase = phase_source_maps.get(phase)
            if (
                not isinstance(semantic_input, Mapping)
                or not isinstance(capacity_phase, Mapping)
                or Path(str(semantic_input.get("path"))).resolve()
                != Path(
                    str(capacity_phase.get("extracted_manifest_path"))
                ).resolve()
                or semantic_input.get("sha256")
                != capacity_phase.get("extracted_manifest_sha256")
                or semantic_input.get("corpus_fingerprint")
                != capacity_phase.get("extracted_corpus_fingerprint")
            ):
                issues.append(
                    f"Chinese semantic {phase} input differs from closed "
                    "capacity extracted identity"
                )
    formal_phase_identity = formal_bindings.get(
        "formal_phase_train_disjointness"
    )
    capacity_phase_identity = capacity_bindings.get(
        "phase_disjointness_attestation"
    )
    if (
        isinstance(formal_phase_identity, Mapping)
        and isinstance(capacity_phase_identity, Mapping)
        and (
            Path(str(formal_phase_identity.get("path"))).resolve()
            != Path(str(capacity_phase_identity.get("path"))).resolve()
            or formal_phase_identity.get("sha256")
            != capacity_phase_identity.get("sha256")
            or formal_phase_identity.get("attestation_fingerprint")
            != capacity_phase_identity.get("attestation_fingerprint")
        )
    ):
        issues.append(
            "formal-validation evidence binds another phase-disjointness attestation"
        )

    fork_policy = _mapping(readiness.get("fork_policy"), label="readiness.fork_policy")
    if (
        fork_policy.get("required_model_only_checkpoint") != FORMAL_V4_FORK_PATH
        or fork_policy.get("required_checkpoint_complete_sha256")
        != FORMAL_V4_FORK_COMPLETE_SHA256
        or fork_policy.get("reset_optimizer_and_scheduler") is not True
        or fork_policy.get("forbidden_warm_starts")
        != list(FORMAL_V4_FORBIDDEN_WARM_STARTS)
        or readiness.get("fork_from") != FORMAL_V4_FORK_PATH
    ):
        raise GovernedControllerError(
            "fork policy differs from the source-bound model-only reset contract"
        )
    fork_path = _project_path(
        project_root,
        fork_policy.get("required_model_only_checkpoint"),
        label="fork_policy.required_model_only_checkpoint",
    )
    expected_fork_complete_sha = _sha256_string(
        fork_policy.get("required_checkpoint_complete_sha256"),
        label="fork_policy.required_checkpoint_complete_sha256",
    )
    fork_complete = fork_path / "COMPLETE"
    actual_fork_complete_sha = sha256_file(fork_complete) if fork_complete.is_file() else None
    if actual_fork_complete_sha != expected_fork_complete_sha:
        issues.append("v3 fork checkpoint COMPLETE identity is missing or mismatched")

    output_dir = project_root / "runs" / str(config_path.stem)
    run_id: str | None = None
    config_data_contract: dict[str, Any] = {}
    preflight_config_fingerprint: str | None = None
    try:
        import yaml

        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise GovernedControllerError(
            f"authenticated config YAML could not be read: {exc}"
        ) from exc
    raw_config = _mapping(raw_config, label="authenticated config")
    _require_config_fields(
        raw_config,
        {
            "run_id": policy["run_id"],
            "track": policy["track"],
            "stage": policy["stage"],
            "architecture": {
                "student_layers": policy["student_layers"],
                "student_hidden_size": policy["student_hidden_size"],
                "student_intermediate_size": policy[
                    "student_intermediate_size"
                ],
                "donor_layers": policy["donor_layers"],
                "donor_hidden_size": policy["donor_hidden_size"],
                "donor_intermediate_size": policy["donor_intermediate_size"],
                "expert_intermediate_size": policy[
                    "expert_intermediate_size"
                ],
                "num_experts": policy["num_experts"],
                "top_k": policy["top_k"],
                "norm_topk_prob": policy["norm_topk_prob"],
                "lora_rank": policy["lora_rank"],
                "expert_initialization": policy["expert_initialization"],
                "active_student_layers": policy["active_student_layers"],
                "adapter_init_path": policy["adapter_init_path"],
                "channel_map_path": policy["channel_map_path"],
                "layer_map_path": policy["layer_map_path"],
            },
            "data": {
                "mode": policy["data_mode"],
                "max_sequence_length": policy["sequence_length"],
                "micro_batch_size": policy["micro_batch_size"],
                "global_batch_tokens": policy["global_batch_tokens"],
                "source_mix_algorithm": policy["source_mix_algorithm"],
                "source_mix_basis_points": policy[
                    "primary_source_mix_basis_points"
                ],
                "source_mix_allow_weight_override": policy[
                    "source_mix_allow_weight_override"
                ],
                "shuffle_seed": policy["shuffle_seed"],
                "quality_cooldown_start_tokens": policy[
                    "cooldown_start_tokens"
                ],
                "allow_corpus_reuse": policy["allow_corpus_reuse"],
            },
            "optimizer": {
                "adapter_optimizer": policy["adapter_optimizer"],
                "adapter_lr": policy["adapter_lr"],
                "lora_lr": policy["lora_lr"],
                "scale_lr": policy["scale_lr"],
                "warmup_tokens": policy["warmup_tokens"],
                "max_tokens": policy["max_tokens"],
                "lr_schedule": policy["lr_schedule"],
                "min_lr_ratio": policy["min_lr_ratio"],
                "grad_clip_norm": policy["grad_clip_norm"],
                "muon_momentum": policy["muon_momentum"],
                "muon_nesterov": policy["muon_nesterov"],
                "muon_ns_coefficients": policy["muon_ns_coefficients"],
                "muon_eps": policy["muon_eps"],
                "muon_ns_steps": policy["muon_ns_steps"],
                "muon_adjust_lr_fn": policy["muon_adjust_lr_fn"],
            },
            "losses": {
                "ntp": policy["ntp_weight"],
                "mtp": policy["mtp_weight"],
                "teacher_kd": policy["teacher_kd_weight"],
                "anchor_kl": policy["anchor_kl_weight"],
                "hidden_alignment": policy["hidden_alignment_weight"],
                "dense_oracle": policy["dense_oracle_weight"],
                "router_z": policy["router_z_weight"],
                "load_balance": policy["load_balance_weight"],
                "router_supervision": policy["router_supervision_weight"],
                "dense_oracle_batch_fraction": policy[
                    "dense_oracle_batch_fraction"
                ],
                "hidden_alignment_batch_fraction": policy[
                    "hidden_alignment_batch_fraction"
                ],
            },
            "checkpoint": {
                "every_steps": policy["checkpoint_every_steps"],
                "every_minutes": policy["checkpoint_every_minutes"],
                "keep_last": policy["checkpoint_keep_last"],
                "save_on_signal": policy["checkpoint_save_on_signal"],
                "stop_file": policy["checkpoint_stop_file"],
                "output_dir": policy["checkpoint_output_dir"],
            },
            "runtime": {
                "bf16": policy["runtime_bf16"],
                "allow_tf32": policy["runtime_allow_tf32"],
                "fused_adamw": policy["runtime_fused_adamw"],
                "activation_checkpointing": policy[
                    "runtime_activation_checkpointing"
                ],
                "activation_checkpoint_layer_count": policy[
                    "runtime_activation_checkpoint_layer_count"
                ],
                "loss_checkpoint_chunks": policy[
                    "runtime_loss_checkpoint_chunks"
                ],
                "loss_chunk_tokens": policy["runtime_loss_chunk_tokens"],
                "compile_streaming_loss": policy[
                    "runtime_compile_streaming_loss"
                ],
                "expandable_segments": policy["runtime_expandable_segments"],
                "sharding": policy["runtime_sharding"],
                "teacher_cpu_offload": policy[
                    "runtime_teacher_cpu_offload"
                ],
            },
        },
        label="config",
    )
    normalized_config_sha = _normalized_formal_config_sha256(raw_config)
    if normalized_config_sha != FORMAL_V4_NORMALIZED_CONFIG_SHA256:
        raise GovernedControllerError(
            "config normalized semantic fingerprint differs from the "
            "source-bound formal v4 policy"
        )
    data = _mapping(raw_config.get("data"), label="config.data")
    optimizer = _mapping(raw_config.get("optimizer"), label="config.optimizer")
    checkpoint = _mapping(raw_config.get("checkpoint"), label="config.checkpoint")
    denominator = (
        _integer(
            data.get("max_sequence_length"),
            label="config.data.max_sequence_length",
            minimum=1,
        )
        * world_size
        * _integer(
            data.get("micro_batch_size"),
            label="config.data.micro_batch_size",
            minimum=1,
        )
    )
    global_batch_tokens = _integer(
        data.get("global_batch_tokens"),
        label="config.data.global_batch_tokens",
        minimum=1,
    )
    gradient_accumulation_steps, remainder = divmod(
        global_batch_tokens,
        denominator,
    )
    if (
        remainder
        or gradient_accumulation_steps
        != policy["gradient_accumulation_steps"]
        or denominator * gradient_accumulation_steps
        != global_batch_tokens
    ):
        raise GovernedControllerError(
            "formal batch geometry does not reproduce global_batch_tokens"
        )
    config_max_tokens = _integer(
        optimizer.get("max_tokens"),
        label="config.optimizer.max_tokens",
        minimum=1,
    )
    cooldown_start = _integer(
        data.get("quality_cooldown_start_tokens"),
        label="config.data.quality_cooldown_start_tokens",
        minimum=1,
    )
    if (
        cooldown_start != policy["primary_tokens"]
        or config_max_tokens - cooldown_start != policy["cooldown_tokens"]
        or config_max_tokens != policy["max_tokens"]
    ):
        raise GovernedControllerError(
            "formal primary/cooldown token geometry differs"
        )
    if checkpoint.get("every_steps") != pause_policy.get("checkpoint_every_steps"):
        raise GovernedControllerError(
            "config checkpoint interval differs from pause policy"
        )

    if isinstance(raw_config.get("run_id"), str) and raw_config["run_id"]:
        run_id = raw_config["run_id"]
    else:
        raise GovernedControllerError("config.run_id must be a non-empty string")
    if not isinstance(checkpoint.get("output_dir"), str):
        raise GovernedControllerError("config.checkpoint.output_dir is invalid")
    output_dir = _project_path(
        project_root,
        checkpoint["output_dir"],
        label="checkpoint.output_dir",
    )
    config_data_contract = {
        field: data.get(field)
        for field in (
            "manifest_path",
            "manifest_sha256",
            "source_mix_algorithm",
            "source_map_sha256",
            "source_mix_basis_points",
            "source_mix_allow_weight_override",
            "shuffle_seed",
            "quality_cooldown_manifest_path",
            "quality_cooldown_manifest_sha256",
            "phase_disjointness_attestation_path",
            "phase_disjointness_attestation_sha256",
            "quality_cooldown_start_tokens",
            "allow_corpus_reuse",
        )
    }
    config_data_contract["phase_source_maps"] = dict(phase_source_maps)
    config_data_contract["phase_source_mix_basis_points"] = capacity_bindings.get(
        "source_mix_basis_points",
        {},
    )
    identity_fields = (
        "manifest_path",
        "manifest_sha256",
        "source_map_sha256",
        "quality_cooldown_manifest_path",
        "quality_cooldown_manifest_sha256",
        "phase_disjointness_attestation_path",
        "phase_disjointness_attestation_sha256",
    )
    has_pending_identities = any(
        not isinstance(data.get(field), str)
        or not data[field]
        or "PENDING" in data[field]
        for field in identity_fields
    )
    if has_pending_identities:
        issues.append("authenticated config retains missing or PENDING data identities")
    else:
        try:
            from .config import load_train_config

            preflight_config_fingerprint = _sha256_string(
                load_train_config(config_path).fingerprint(),
                label="config preflight fingerprint",
            )
        except (OSError, ValueError) as exc:
            issues.append(
                "authenticated config cannot produce its preflight fingerprint: "
                f"{exc}"
            )

    primary = phase_source_maps.get("primary")
    cooldown = phase_source_maps.get("cooldown")
    if isinstance(primary, Mapping) and (
        Path(str(primary["prepared_manifest_path"])).resolve()
        != _project_path(
            project_root,
            data.get("manifest_path"),
            label="config.data.manifest_path",
        )
        or primary["prepared_manifest_sha256"] != data.get("manifest_sha256")
        or primary["source_map_sha256"] != data.get("source_map_sha256")
    ):
        issues.append("primary config identities differ from closed capacity evidence")
    if isinstance(cooldown, Mapping) and (
        Path(str(cooldown["prepared_manifest_path"])).resolve()
        != _project_path(
            project_root,
            data.get("quality_cooldown_manifest_path"),
            label="config.data.quality_cooldown_manifest_path",
        )
        or cooldown["prepared_manifest_sha256"]
        != data.get("quality_cooldown_manifest_sha256")
    ):
        issues.append("cooldown config identities differ from closed capacity evidence")
    phase_identity = capacity_bindings.get("phase_disjointness_attestation")
    if isinstance(phase_identity, Mapping) and (
        Path(str(phase_identity["path"])).resolve()
        != _project_path(
            project_root,
            data.get("phase_disjointness_attestation_path"),
            label="config.data.phase_disjointness_attestation_path",
        )
        or phase_identity["sha256"]
        != data.get("phase_disjointness_attestation_sha256")
    ):
        issues.append(
            "config phase-disjointness identity differs from closed capacity evidence"
        )

    hard_stop = _mapping(pause_policy.get("hard_stop"), label="pause_policy.hard_stop")
    validation_stop = _mapping(
        pause_policy.get("validation_stop"),
        label="pause_policy.validation_stop",
    )
    controller_source_paths = (
        project_root / "src/twen/governed.py",
        project_root / "src/twen/training/engine.py",
        project_root / "src/twen/cli.py",
        project_root / "src/twen/evaluation.py",
        project_root / "scripts/govern_v4_training.py",
        project_root / "scripts/audit_dense_checkpoint_drift.py",
        project_root / "scripts/summarize_v4_checkpoint_validation.py",
    )
    if not all(path.is_file() for path in controller_source_paths):
        issues.append("one or more governed controller source files are missing")
    source_tree_root = project_root / "src/twen"
    dependency_lock = project_root / "uv.lock"
    if not dependency_lock.is_file():
        dependency_lock = project_root / "pyproject.toml"
    if not dependency_lock.is_file():
        issues.append("governed dependency lock is missing")
    plan: dict[str, Any] = {
        "kind": PLAN_KIND,
        "schema_version": SCHEMA_VERSION,
        "project_root": str(project_root),
        "readiness": {
            "path": str(readiness_file),
            "sha256": sha256_file(readiness_file),
        },
        "config": {
            "path": str(config_path),
            "sha256": actual_config_sha,
            "normalized_semantic_sha256": normalized_config_sha,
            "preflight_fingerprint": preflight_config_fingerprint,
        },
        "run": {
            "run_id": run_id,
            "stage": stage,
            "world_size": world_size,
            "max_tokens": max_tokens,
            "output_dir": str(output_dir),
        },
        "formal_contract": {
            "attested": dict(contract),
            "source_bound_config_policy": dict(policy),
        },
        "training_data": config_data_contract,
        "fork": {
            "path": str(fork_path),
            "complete_sha256": actual_fork_complete_sha,
            "expected_complete_sha256": expected_fork_complete_sha,
            "model_only": True,
            "reset_optimizer": True,
            "reset_scheduler": True,
            "reset_data_cursor": True,
            "reset_optimizer_scheduler_cursor": True,
            "forbidden_warm_starts": list(FORMAL_V4_FORBIDDEN_WARM_STARTS),
        },
        "formal_evidence": {
            **formal_bindings,
            "release_gates": release_bindings,
        },
        "controller_sources": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in controller_source_paths
            if path.is_file()
        ],
        "source_tree": {
            "path": str(source_tree_root.resolve()),
            "sha256": twen_source_tree_sha256(source_tree_root),
        },
        "dependency_lock": (
            {
                "path": str(dependency_lock.resolve()),
                "sha256": sha256_file(dependency_lock),
            }
            if dependency_lock.is_file()
            else None
        ),
        "release_bundle": release_bundle,
        "pause_thresholds": list(thresholds),
        "gates": {
            "hard_stop": dict(hard_stop),
            "validation_stop": dict(validation_stop),
            "grad_clip_norm": float(policy["grad_clip_norm"]),
            "required_finite_metrics": list(REQUIRED_FINITE_METRICS),
        },
        "launch_enabled": readiness.get("launch_enabled") is True,
        "readiness_issues": sorted(set(issues)),
    }
    plan["plan_id"] = _canonical_sha256(plan)
    return plan


def expected_run_ack(plan: Mapping[str, Any]) -> str:
    plan_id = _sha256_string(plan.get("plan_id"), label="plan.plan_id")
    unsigned = {key: value for key, value in plan.items() if key != "plan_id"}
    if _canonical_sha256(unsigned) != plan_id:
        raise GovernedControllerError("governed plan fingerprint is invalid")
    return f"RUN {plan_id}"


def authorize_run(plan: Mapping[str, Any], acknowledgement: str | None) -> None:
    """Require both readiness authorization and the exact per-plan ACK."""

    issues = plan.get("readiness_issues")
    if not isinstance(issues, list):
        raise GovernedControllerError("plan readiness_issues is invalid")
    if plan.get("launch_enabled") is not True or issues:
        rendered = "; ".join(str(issue) for issue in issues) or "launch_enabled is false"
        raise GovernedControllerError(f"governed launch is blocked: {rendered}")
    expected = expected_run_ack(plan)
    if acknowledgement != expected:
        raise GovernedControllerError(f"explicit acknowledgement must equal {expected!r}")


def _seal_controller_state(state: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in state.items() if key != "state_fingerprint"}
    result["state_fingerprint"] = _canonical_sha256(result)
    return result


def _checkpoint_binding(value: Any, *, label: str) -> Mapping[str, Any]:
    checkpoint = _mapping(value, label=label)
    if not isinstance(checkpoint.get("path"), str) or not checkpoint["path"]:
        raise GovernedControllerError(f"{label}.path must be non-empty")
    for field in ("manifest_sha256", "complete_sha256"):
        _sha256_string(checkpoint.get(field), label=f"{label}.{field}")
    _integer(checkpoint.get("global_step"), label=f"{label}.global_step", minimum=1)
    _integer(checkpoint.get("committed_tokens"), label=f"{label}.committed_tokens", minimum=1)
    if checkpoint.get("kind") not in {"periodic", "interrupt", "milestone"}:
        raise GovernedControllerError(f"{label}.kind is invalid")
    if checkpoint.get("tag") is not None and not isinstance(checkpoint.get("tag"), str):
        raise GovernedControllerError(f"{label}.tag is invalid")
    return checkpoint


def _validate_decision(value: Any, plan: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    decision = _mapping(value, label=label)
    if (
        decision.get("kind") != GATE_KIND
        or decision.get("schema_version") != SCHEMA_VERSION
        or decision.get("plan_id") != plan.get("plan_id")
    ):
        raise GovernedControllerError(f"{label} is not bound to this plan")
    fingerprint = _sha256_string(
        decision.get("decision_fingerprint"),
        label=f"{label}.decision_fingerprint",
    )
    unsigned = {
        key: item for key, item in decision.items() if key != "decision_fingerprint"
    }
    if _canonical_sha256(unsigned) != fingerprint:
        raise GovernedControllerError(f"{label} fingerprint is invalid")
    if decision.get("action") not in {
        "resume",
        "complete",
        "stop",
        "review",
        "terminal_stop",
    }:
        raise GovernedControllerError(f"{label}.action is invalid")
    return decision


def _validate_controller_state(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    expected_run_ack(plan)
    if state.get("kind") != STATE_KIND or state.get("schema_version") != SCHEMA_VERSION:
        raise GovernedControllerError("unsupported controller state")
    if state.get("plan_id") != plan.get("plan_id"):
        raise GovernedControllerError("controller state belongs to a different immutable plan")
    fingerprint = _sha256_string(
        state.get("state_fingerprint"),
        label="controller state_fingerprint",
    )
    unsigned = {key: value for key, value in state.items() if key != "state_fingerprint"}
    if _canonical_sha256(unsigned) != fingerprint:
        raise GovernedControllerError("controller state fingerprint is invalid")
    status = state.get("status")
    allowed_statuses = {
        "not_started",
        "running",
        "awaiting_evaluation",
        "resume_authorized",
        "completed",
        "halted",
        "review_required",
        "failed",
    }
    if status not in allowed_statuses:
        raise GovernedControllerError("controller state status is invalid")
    thresholds = list(plan.get("pause_thresholds", []))
    completed = state.get("completed_thresholds")
    if not isinstance(completed, list):
        raise GovernedControllerError("controller completed_thresholds must be a list")
    if completed != thresholds[: len(completed)]:
        raise GovernedControllerError(
            "controller completed_thresholds must be an exact policy prefix"
        )
    history = state.get("gate_history")
    if not isinstance(history, list) or len(history) != len(completed):
        raise GovernedControllerError(
            "controller gate_history must exactly cover completed_thresholds"
        )
    for index, raw in enumerate(history):
        record = _mapping(raw, label=f"gate_history[{index}]")
        if record.get("threshold") != thresholds[index]:
            raise GovernedControllerError("gate_history threshold order differs from policy")
        checkpoint = _checkpoint_binding(
            record.get("checkpoint"),
            label=f"gate_history[{index}].checkpoint",
        )
        terminal_threshold = index + 1 == len(thresholds)
        expected_tag = (
            "complete"
            if terminal_threshold
            else f"governed-{thresholds[index]:012d}"
        )
        if (
            checkpoint.get("kind") != "milestone"
            or checkpoint.get("tag") != expected_tag
        ):
            raise GovernedControllerError(
                f"gate_history[{index}] is not the required retained milestone"
            )
        tokens = int(checkpoint["committed_tokens"])
        if tokens < thresholds[index] or (
            index + 1 < len(thresholds) and tokens >= thresholds[index + 1]
        ):
            raise GovernedControllerError(
                f"gate_history[{index}] checkpoint is outside its threshold interval"
            )
        decision = _validate_decision(
            record.get("decision"),
            plan,
            label=f"gate_history[{index}].decision",
        )
        evidence = _mapping(
            record.get("evidence"),
            label=f"gate_history[{index}].evidence",
        )
        evidence_fingerprint = _sha256_string(
            record.get("evidence_fingerprint"),
            label=f"gate_history[{index}].evidence_fingerprint",
        )
        if _canonical_sha256(evidence) != evidence_fingerprint:
            raise GovernedControllerError(
                f"gate_history[{index}] evidence fingerprint is invalid"
            )
        expected_action = (
            "resume"
            if index + 1 < len(thresholds)
            else "complete"
        )
        if decision.get("action") not in {
            expected_action,
            "stop",
            "review",
            "terminal_stop",
        }:
            raise GovernedControllerError(
                f"gate_history[{index}] decision cannot authorize this transition"
            )
        if index + 1 < len(history) and decision.get("action") != "resume":
            raise GovernedControllerError(
                f"gate_history[{index}] did not authorize the following threshold"
            )

    current_raw = state.get("current_checkpoint")
    current = (
        _checkpoint_binding(current_raw, label="current_checkpoint")
        if current_raw is not None
        else None
    )
    active = state.get("active_threshold")
    next_threshold = thresholds[len(completed)] if len(completed) < len(thresholds) else None
    if status == "not_started":
        if completed or history or current is not None or active is not None:
            raise GovernedControllerError("not_started state contains progress")
    elif status == "running":
        if next_threshold is None or active != next_threshold:
            raise GovernedControllerError("running state has an invalid active threshold")
        if completed and (
            current is None
            or dict(current) != dict(history[-1]["checkpoint"])
        ):
            raise GovernedControllerError("running state lost its resume checkpoint")
        if not completed and current is not None:
            raise GovernedControllerError("first running segment cannot have a checkpoint")
        command = state.get("active_command")
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise GovernedControllerError("running state has no exact active command")
        recovery_raw = state.get("recovery_checkpoint")
        if recovery_raw is not None:
            recovery = _checkpoint_binding(
                recovery_raw,
                label="recovery_checkpoint",
            )
            recovery_tokens = int(recovery["committed_tokens"])
            lower_bound = (
                int(history[-1]["checkpoint"]["committed_tokens"])
                if completed
                else 0
            )
            if not lower_bound <= recovery_tokens < int(active):
                raise GovernedControllerError(
                    "running recovery checkpoint is outside the active segment"
                )
    elif status == "awaiting_evaluation":
        if next_threshold is None or active != next_threshold or current is None:
            raise GovernedControllerError("awaiting_evaluation state is incomplete")
        tokens = int(current["committed_tokens"])
        if tokens < next_threshold or (
            len(completed) + 1 < len(thresholds)
            and tokens >= thresholds[len(completed) + 1]
        ):
            raise GovernedControllerError(
                "awaiting_evaluation checkpoint is outside the active threshold"
            )
    elif status == "resume_authorized":
        if (
            not completed
            or next_threshold is None
            or active is not None
            or current is None
            or dict(current) != dict(history[-1]["checkpoint"])
            or history[-1]["decision"].get("action") != "resume"
        ):
            raise GovernedControllerError("resume_authorized state is inconsistent")
    elif status == "completed":
        if (
            len(completed) != len(thresholds)
            or active is not None
            or current is None
            or history[-1]["decision"].get("action") != "complete"
        ):
            raise GovernedControllerError("completed state is inconsistent")
    elif status in {"halted", "review_required"}:
        if not completed or active is not None or current is None:
            raise GovernedControllerError(f"{status} state is inconsistent")
        expected_actions = (
            {"review"} if status == "review_required" else {"stop", "terminal_stop"}
        )
        if history[-1]["decision"].get("action") not in expected_actions:
            raise GovernedControllerError(f"{status} decision differs")
    if status != "running" and state.get("recovery_checkpoint") is not None:
        raise GovernedControllerError(
            "only a running segment may bind a recovery checkpoint"
        )


def initial_controller_state(plan: Mapping[str, Any]) -> dict[str, Any]:
    return _seal_controller_state({
        "kind": STATE_KIND,
        "schema_version": SCHEMA_VERSION,
        "plan_id": _sha256_string(plan.get("plan_id"), label="plan.plan_id"),
        "status": "not_started",
        "current_checkpoint": None,
        "completed_thresholds": [],
        "gate_history": [],
        "active_threshold": None,
        "active_command": None,
        "last_exit_code": None,
        "failure": None,
        "recovery_checkpoint": None,
    })


def load_controller_state(
    path: str | Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        return initial_controller_state(plan)
    state = _read_json(state_path, label="controller state")
    _validate_controller_state(state, plan)
    return state


def write_controller_state(
    path: str | Path,
    state: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically fsync a fully materialized controller state."""

    sealed = _seal_controller_state(state)
    _validate_controller_state(sealed, plan)
    atomic_write_json(path, sealed)
    return sealed


def controller_status(
    plan: Mapping[str, Any],
    state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state = dict(state or initial_controller_state(plan))
    _validate_controller_state(state, plan)
    completed = list(state.get("completed_thresholds", []))
    thresholds = list(plan.get("pause_thresholds", []))
    next_threshold = thresholds[len(completed)] if len(completed) < len(thresholds) else None
    blocked = plan.get("launch_enabled") is not True or bool(plan.get("readiness_issues"))
    return {
        "ok": not blocked,
        "action": "status",
        "plan_id": plan.get("plan_id"),
        "launch_enabled": plan.get("launch_enabled") is True,
        "blocked": blocked,
        "blockers": list(plan.get("readiness_issues", [])),
        "controller_state": state.get("status"),
        "current_checkpoint": state.get("current_checkpoint"),
        "completed_thresholds": completed,
        "next_threshold": next_threshold,
        "history_verification": "not_performed",
        "historical_conclusion_verified": False,
        "required_ack": expected_run_ack(plan),
    }


def build_train_command(
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
) -> list[str]:
    """Build exactly one launch/resume segment without executing it."""

    status = controller_status(plan, state)
    threshold = status["next_threshold"]
    if threshold is None:
        raise GovernedControllerError("all governed thresholds are already complete")
    run = _mapping(plan.get("run"), label="plan.run")
    world_size = _integer(run.get("world_size"), label="plan.run.world_size", minimum=1)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        "-m",
        "twen.cli",
        "train",
        "--stage",
        str(run["stage"]),
        "--config",
        str(_mapping(plan.get("config"), label="plan.config")["path"]),
        "--expected-config-sha256",
        str(_mapping(plan.get("config"), label="plan.config")["sha256"]),
        "--progress",
        "always",
    ]
    checkpoint = state.get("current_checkpoint")
    if checkpoint is None:
        command.extend(
            [
                "--resume",
                "none",
                "--fork-from",
                str(_mapping(plan.get("fork"), label="plan.fork")["path"]),
            ]
        )
    elif isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("path"), str):
        command.extend(["--resume", checkpoint["path"]])
    else:
        raise GovernedControllerError("controller current_checkpoint is invalid")
    max_tokens = _integer(run.get("max_tokens"), label="plan.run.max_tokens", minimum=1)
    if threshold < max_tokens:
        command.extend(["--pause-at-tokens", str(threshold)])
    return command


def render_train_command(plan: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    return shlex.join(build_train_command(plan, state))


def read_canonical_metrics(
    path: str | Path,
    *,
    through_step: int,
) -> list[dict[str, Any]]:
    """Read a complete canonical metric prefix and reject gaps/duplicates."""

    metric_path = Path(path)
    rows: list[dict[str, Any]] = []
    try:
        with metric_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise GovernedControllerError(
                        f"metrics line {line_number} is invalid JSON"
                    ) from exc
                if not isinstance(row, dict):
                    raise GovernedControllerError(
                        f"metrics line {line_number} must be an object"
                    )
                step = _integer(row.get("step"), label=f"metrics line {line_number} step", minimum=1)
                if step > through_step:
                    break
                rows.append(row)
    except OSError as exc:
        raise GovernedControllerError(f"cannot read canonical metrics {metric_path}: {exc}") from exc
    expected = list(range(1, through_step + 1))
    actual = [int(row["step"]) for row in rows]
    if actual != expected:
        raise GovernedControllerError(
            f"canonical metrics are not a complete 1..{through_step} prefix"
        )
    return rows


def read_authenticated_metrics_prefix(
    path: str | Path,
    checkpoint_metadata: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read exactly the metrics bytes authenticated by checkpoint metadata."""

    metadata = _mapping(checkpoint_metadata, label="checkpoint.metadata")
    extra = _mapping(metadata.get("extra"), label="checkpoint.metadata.extra")
    binding = _mapping(
        extra.get("metrics_prefix"),
        label="checkpoint.metrics_prefix",
    )
    if (
        binding.get("kind") != "twen_metrics_jsonl_prefix"
        or binding.get("schema_version") != 1
        or binding.get("relative_path") != "metrics.jsonl"
    ):
        raise GovernedControllerError("checkpoint metrics-prefix contract is invalid")
    through_step = _integer(
        binding.get("through_step"),
        label="checkpoint.metrics_prefix.through_step",
        minimum=1,
    )
    committed_tokens = _integer(
        binding.get("committed_tokens"),
        label="checkpoint.metrics_prefix.committed_tokens",
        minimum=1,
    )
    record_count = _integer(
        binding.get("record_count"),
        label="checkpoint.metrics_prefix.record_count",
        minimum=1,
    )
    prefix_size = _integer(
        binding.get("prefix_size_bytes"),
        label="checkpoint.metrics_prefix.prefix_size_bytes",
        minimum=1,
    )
    prefix_sha = _sha256_string(
        binding.get("prefix_sha256"),
        label="checkpoint.metrics_prefix.prefix_sha256",
    )
    if (
        through_step != metadata.get("global_step")
        or committed_tokens != metadata.get("committed_tokens")
        or record_count != through_step
    ):
        raise GovernedControllerError(
            "checkpoint metrics-prefix counters differ from checkpoint state"
        )
    metric_path = Path(path).expanduser().resolve()
    try:
        with metric_path.open("rb") as handle:
            payload = handle.read(prefix_size)
    except OSError as exc:
        raise GovernedControllerError(
            f"cannot read checkpoint-authenticated metrics {metric_path}: {exc}"
        ) from exc
    if (
        len(payload) != prefix_size
        or hashlib.sha256(payload).hexdigest() != prefix_sha
        or not payload.endswith(b"\n")
    ):
        raise GovernedControllerError(
            "metrics JSONL prefix differs from the checkpoint-authenticated bytes"
        )
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line:
            raise GovernedControllerError(
                f"authenticated metrics line {line_number} is blank"
            )
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GovernedControllerError(
                f"authenticated metrics line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise GovernedControllerError(
                f"authenticated metrics line {line_number} must be an object"
            )
        rows.append(row)
    steps = [row.get("step") for row in rows]
    if steps != list(range(1, through_step + 1)):
        raise GovernedControllerError(
            f"authenticated metrics are not the exact 1..{through_step} prefix"
        )
    if (
        len(rows) != record_count
        or rows[-1].get("tokens") != committed_tokens
    ):
        raise GovernedControllerError(
            "authenticated metrics do not end at checkpoint committed tokens"
        )
    return rows, dict(binding)


def authenticate_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    """Fully hash-verify one checkpoint on CPU and return bound metadata."""

    from .runtime.checkpoint import CheckpointManager

    path = Path(checkpoint).expanduser().resolve()
    manager = CheckpointManager(path.parent, rank=0, world_size=1)
    metadata = manager.verify(path)
    return {
        "path": str(path),
        "manifest_sha256": sha256_file(path / "manifest.json"),
        "complete_sha256": sha256_file(path / "COMPLETE"),
        "metadata": metadata,
    }


def verify_controller_sources(plan: Mapping[str, Any]) -> None:
    for label in ("readiness", "config"):
        identity = _mapping(plan.get(label), label=f"plan.{label}")
        path = Path(str(identity.get("path"))).resolve()
        expected = _sha256_string(
            identity.get("sha256"),
            label=f"plan.{label}.sha256",
        )
        if not path.is_file() or sha256_file(path) != expected:
            raise GovernedControllerError(
                f"{label} changed after planning: {path}"
            )
        if label == "config":
            try:
                import yaml

                raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as exc:
                raise GovernedControllerError(
                    f"config cannot be reauthenticated after planning: {path}"
                ) from exc
            normalized = _normalized_formal_config_sha256(
                _mapping(raw_config, label="plan-bound config")
            )
            if (
                normalized != FORMAL_V4_NORMALIZED_CONFIG_SHA256
                or normalized != identity.get("normalized_semantic_sha256")
            ):
                raise GovernedControllerError(
                    f"config normalized semantics changed after planning: {path}"
                )
            expected_preflight = identity.get("preflight_fingerprint")
            if expected_preflight is not None:
                expected_preflight = _sha256_string(
                    expected_preflight,
                    label="plan.config.preflight_fingerprint",
                )
                try:
                    from .config import load_train_config

                    actual_preflight = load_train_config(path).fingerprint()
                except (OSError, ValueError) as exc:
                    raise GovernedControllerError(
                        f"config preflight identity changed after planning: {path}"
                    ) from exc
                if actual_preflight != expected_preflight:
                    raise GovernedControllerError(
                        f"config preflight identity changed after planning: {path}"
                    )

    release_binding = plan.get("release_bundle")
    if release_binding is not None:
        expected_release = _mapping(
            release_binding,
            label="plan.release_bundle",
        )
        readiness_identity = _mapping(
            plan.get("readiness"),
            label="plan.readiness",
        )
        project_root = Path(str(plan.get("project_root"))).resolve()
        try:
            current_release = _authenticate_formal_release_bundle(
                Path(str(readiness_identity.get("path"))),
                project_root=project_root,
            )
        except (GovernedControllerError, OSError, ValueError) as exc:
            raise GovernedControllerError(
                f"formal release evidence changed after planning: {exc}"
            ) from exc
        if _canonical_sha256(current_release) != _canonical_sha256(
            expected_release
        ):
            raise GovernedControllerError(
                "formal release evidence changed after planning"
            )

    source_tree = _mapping(plan.get("source_tree"), label="plan.source_tree")
    source_root = Path(str(source_tree.get("path"))).resolve()
    expected_tree = _sha256_string(
        source_tree.get("sha256"),
        label="plan.source_tree.sha256",
    )
    if not source_root.is_dir() or twen_source_tree_sha256(source_root) != expected_tree:
        raise GovernedControllerError(
            f"Twen source tree changed after planning: {source_root}"
        )
    dependency = _mapping(
        plan.get("dependency_lock"),
        label="plan.dependency_lock",
    )
    dependency_path = Path(str(dependency.get("path"))).resolve()
    dependency_sha = _sha256_string(
        dependency.get("sha256"),
        label="plan.dependency_lock.sha256",
    )
    if (
        not dependency_path.is_file()
        or sha256_file(dependency_path) != dependency_sha
    ):
        raise GovernedControllerError(
            f"dependency lock changed after planning: {dependency_path}"
        )
    sources = plan.get("controller_sources")
    if not isinstance(sources, list) or not sources:
        raise GovernedControllerError("plan has no controller source identities")
    for index, raw in enumerate(sources):
        identity = _mapping(raw, label=f"controller_sources[{index}]")
        path = Path(str(identity.get("path"))).resolve()
        expected = _sha256_string(
            identity.get("sha256"),
            label=f"controller_sources[{index}].sha256",
        )
        if not path.is_file() or sha256_file(path) != expected:
            raise GovernedControllerError(f"controller source changed after planning: {path}")


def _bound_controller_source(plan: Mapping[str, Any], filename: str) -> Path:
    sources = plan.get("controller_sources")
    if not isinstance(sources, list):
        raise GovernedControllerError("plan has no controller source identities")
    matches = [
        _mapping(raw, label=f"controller source {filename}")
        for raw in sources
        if isinstance(raw, Mapping) and Path(str(raw.get("path"))).name == filename
    ]
    if len(matches) != 1:
        raise GovernedControllerError(f"plan must bind exactly one {filename}")
    path = Path(str(matches[0]["path"])).resolve()
    expected = _sha256_string(matches[0].get("sha256"), label=f"{filename}.sha256")
    if not path.is_file() or sha256_file(path) != expected:
        raise GovernedControllerError(f"controller source changed after planning: {path}")
    return path


def _load_bound_script(
    plan: Mapping[str, Any],
    filename: str,
    *,
    module_name: str,
) -> ModuleType:
    path = _bound_controller_source(plan, filename)
    before = sha256_file(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise GovernedControllerError(f"cannot load plan-bound controller source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if sha256_file(path) != before:
        raise GovernedControllerError(f"controller source changed while loading: {path}")
    return module


def authenticate_drift_evidence(
    plan: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    path: str | Path,
) -> dict[str, Any]:
    """Authenticate controller-produced CPU drift output against both checkpoints."""

    verify_controller_sources(plan)
    evidence_path = Path(path).resolve()
    payload = _read_json(evidence_path, label="drift evidence")
    if (
        payload.get("kind") != "twen_dense_checkpoint_trainable_drift_audit"
        or payload.get("schema_version") != 1
    ):
        raise GovernedControllerError("drift evidence kind/schema differs")
    audit_module = _load_bound_script(
        plan,
        "audit_dense_checkpoint_drift.py",
        module_name="_twen_governed_drift_verifier",
    )
    try:
        recomputed = audit_module.audit(
            Path(str(_mapping(plan.get("fork"), label="plan.fork")["path"])),
            [Path(str(checkpoint.get("path")))],
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise GovernedControllerError(f"cannot recompute drift evidence: {exc}") from exc
    verify_controller_sources(plan)
    if payload != recomputed:
        raise GovernedControllerError(
            "drift evidence differs from a fresh plan-bound tensor recomputation"
        )
    baseline = _mapping(payload.get("baseline"), label="drift.baseline")
    fork = _mapping(plan.get("fork"), label="plan.fork")
    if (
        Path(str(baseline.get("path"))).resolve() != Path(str(fork.get("path"))).resolve()
        or baseline.get("complete_sha256") != fork.get("complete_sha256")
    ):
        raise GovernedControllerError("drift baseline is not the plan-bound v3 checkpoint")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise GovernedControllerError("drift evidence must contain exactly one candidate")
    candidate = _mapping(candidates[0], label="drift.candidate")
    if (
        Path(str(candidate.get("path"))).resolve()
        != Path(str(checkpoint.get("path"))).resolve()
        or candidate.get("manifest_sha256") != checkpoint.get("manifest_sha256")
        or candidate.get("complete_sha256") != checkpoint.get("complete_sha256")
    ):
        raise GovernedControllerError("drift candidate is not the authenticated pause checkpoint")
    scale = _mapping(candidate.get("scale"), label="drift.candidate.scale")
    relative_l2 = _finite(
        scale.get("relative_l2"),
        label="drift.candidate.scale.relative_l2",
        minimum=0.0,
    )
    return {
        "authenticated": True,
        "path": str(evidence_path),
        "sha256": sha256_file(evidence_path),
        "scale_relative_l2": relative_l2,
    }


def generate_drift_evidence(
    plan: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    path: str | Path,
) -> dict[str, Any]:
    """Recompute and atomically publish one plan-bound CPU drift audit."""

    verify_controller_sources(plan)
    output = Path(path).resolve()
    audit_module = _load_bound_script(
        plan,
        "audit_dense_checkpoint_drift.py",
        module_name="_twen_governed_drift_generator",
    )
    try:
        payload = audit_module.audit(
            Path(str(_mapping(plan.get("fork"), label="plan.fork")["path"])),
            [Path(str(checkpoint.get("path")))],
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise GovernedControllerError(f"cannot generate drift evidence: {exc}") from exc
    verify_controller_sources(plan)
    atomic_write_json(output, payload)
    return {
        "path": str(output),
        "sha256": sha256_file(output),
    }


def generate_checkpoint_sweep(
    plan: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    phase: str,
    candidate_evaluation: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Generate the authenticated v3-vs-candidate sweep for one formal phase."""

    verify_controller_sources(plan)
    formal = _mapping(plan.get("formal_evidence"), label="plan.formal_evidence")
    phases = _mapping(formal.get("validation_phases"), label="formal.validation_phases")
    phase_plan = _mapping(phases.get(phase), label=f"formal.validation_phases.{phase}")
    prepared = _mapping(phase_plan.get("prepared"), label=f"{phase}.prepared")
    baseline = _mapping(phase_plan.get("v3_evaluation"), label=f"{phase}.v3_evaluation")
    module = _load_bound_script(
        plan,
        "summarize_v4_checkpoint_validation.py",
        module_name=f"_twen_governed_{phase}_sweep_generator",
    )
    output_path = Path(output).resolve()
    try:
        module.generate(
            prepared_manifest=Path(str(prepared["path"])),
            baseline_path=Path(str(baseline["path"])),
            baseline_label="v3",
            candidate_paths=[("candidate", Path(candidate_evaluation).resolve())],
            output=output_path,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise GovernedControllerError(f"cannot generate {phase} checkpoint sweep: {exc}") from exc
    verify_controller_sources(plan)
    return _authenticate_generated_sweep(
        plan,
        checkpoint,
        phase=phase,
        root=output_path,
    )


def _authenticate_generated_sweep(
    plan: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    phase: str,
    root: str | Path,
) -> dict[str, Any]:
    verify_controller_sources(plan)
    sweep_root = Path(root).resolve()
    manifest_path = sweep_root / "MANIFEST.json"
    complete_path = sweep_root / "COMPLETE"
    _manifest, summary = _authenticate_output_bundle(
        sweep_root,
        expected_manifest_sha256=sha256_file(manifest_path),
        expected_complete_sha256=sha256_file(complete_path),
        label=f"{phase} checkpoint sweep",
    )
    if (
        summary.get("kind") != "twen_v4_checkpoint_frozen_validation_sweep"
        or summary.get("target_role") != "candidate"
        or summary.get("training_started_by_summarizer") is not False
        or summary.get("inputs_mutated_by_summarizer") is not False
    ):
        raise GovernedControllerError(f"{phase} checkpoint sweep contract differs")
    formal = _mapping(plan.get("formal_evidence"), label="plan.formal_evidence")
    phases = _mapping(formal.get("validation_phases"), label="formal.validation_phases")
    phase_plan = _mapping(phases.get(phase), label=f"formal.validation_phases.{phase}")
    prepared = _mapping(phase_plan.get("prepared"), label=f"{phase}.prepared")
    summary_prepared = _mapping(summary.get("prepared_manifest"), label="sweep.prepared")
    if (
        Path(str(summary_prepared.get("path"))).resolve()
        != Path(str(prepared.get("path"))).resolve()
        or summary_prepared.get("sha256") != prepared.get("sha256")
    ):
        raise GovernedControllerError(f"{phase} sweep used another frozen manifest")
    baseline_plan = _mapping(
        phase_plan.get("v3_evaluation"),
        label=f"{phase}.v3_evaluation",
    )
    baseline = _mapping(summary.get("baseline"), label="sweep.baseline")
    baseline_identity = _mapping(baseline.get("evaluation"), label="sweep.baseline.evaluation")
    if Path(str(baseline_identity.get("root"))).resolve() != Path(
        str(baseline_plan.get("path"))
    ).resolve():
        raise GovernedControllerError(f"{phase} sweep used another v3 evaluation")
    for name in ("manifest", "complete", "plan"):
        if _mapping(
            baseline_identity.get(name),
            label=f"sweep.baseline.evaluation.{name}",
        ).get("sha256") != _mapping(
            baseline_plan.get(name),
            label=f"plan.baseline.{name}",
        ).get("sha256"):
            raise GovernedControllerError(f"{phase} v3 evaluation identity changed")

    candidates = summary.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise GovernedControllerError(f"{phase} sweep must contain one candidate")
    candidate = _mapping(candidates[0], label="sweep.candidate")
    checkpoint_state = _mapping(
        candidate.get("checkpoint_state"),
        label="sweep.candidate.checkpoint_state",
    )
    metadata = _mapping(checkpoint.get("metadata"), label="checkpoint.metadata")
    if (
        checkpoint_state.get("global_step") != metadata.get("global_step")
        or checkpoint_state.get("committed_tokens") != metadata.get("committed_tokens")
        or checkpoint_state.get("kind") != metadata.get("kind")
        or checkpoint_state.get("tag") != metadata.get("tag")
    ):
        raise GovernedControllerError(f"{phase} evaluation checkpoint state differs")
    candidate_identity = _mapping(
        candidate.get("evaluation"),
        label="sweep.candidate.evaluation",
    )
    candidate_root = Path(str(candidate_identity.get("root"))).resolve()
    candidate_plan = _read_json(candidate_root / "PLAN.json", label=f"{phase} candidate PLAN")
    if (
        Path(str(candidate_plan.get("checkpoint"))).resolve()
        != Path(str(checkpoint.get("path"))).resolve()
        or candidate_plan.get("checkpoint_complete_sha256")
        != checkpoint.get("complete_sha256")
    ):
        raise GovernedControllerError(f"{phase} evaluation used another checkpoint")
    sweep_module = _load_bound_script(
        plan,
        "summarize_v4_checkpoint_validation.py",
        module_name=f"_twen_governed_{phase}_sweep_verifier",
    )
    try:
        rebuilt = sweep_module.build_summary(
            prepared_manifest=Path(str(prepared["path"])),
            baseline_path=Path(str(baseline_plan["path"])),
            baseline_label="v3",
            candidate_paths=[("candidate", candidate_root)],
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise GovernedControllerError(
            f"{phase} evaluator evidence failed fresh authentication: {exc}"
        ) from exc
    verify_controller_sources(plan)
    if rebuilt != summary:
        raise GovernedControllerError(
            f"{phase} sweep summary differs from fresh evaluator authentication"
        )
    return {
        "phase": phase,
        "path": str(sweep_root),
        "manifest_sha256": sha256_file(manifest_path),
        "complete_sha256": sha256_file(complete_path),
        "baseline": baseline,
        "candidate": candidate,
    }


def _source_map_identity(value: Any, *, label: str) -> dict[str, Any]:
    source_map = _mapping(value, label=label)
    if source_map.get("algorithm") != "authenticated-extracted-output-map-v1":
        raise GovernedControllerError(f"{label} algorithm is invalid")
    prepared_fingerprint = _sha256_string(
        source_map.get("prepared_dataset_fingerprint"),
        label=f"{label}.prepared_dataset_fingerprint",
    )
    _sha256_string(
        source_map.get("extracted_manifest_sha256"),
        label=f"{label}.extracted_manifest_sha256",
    )
    _integer(
        source_map.get("sequence_length"),
        label=f"{label}.sequence_length",
        minimum=1,
    )
    raw_shards = source_map.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise GovernedControllerError(f"{label}.shards must be non-empty")
    source_ids: set[str] = set()
    shard_ids: set[str] = set()
    output_paths: set[str] = set()
    next_sample = 0
    for index, raw_shard in enumerate(raw_shards):
        shard = _mapping(raw_shard, label=f"{label}.shards[{index}]")
        source_id = shard.get("source_id")
        shard_id = shard.get("shard_id")
        output_path = shard.get("output_path")
        if not all(
            isinstance(item, str) and item
            for item in (source_id, shard_id, output_path)
        ):
            raise GovernedControllerError(
                f"{label}.shards[{index}] has an invalid source/shard/output identity"
            )
        assert isinstance(source_id, str)
        assert isinstance(shard_id, str)
        assert isinstance(output_path, str)
        if shard_id in shard_ids or output_path in output_paths:
            raise GovernedControllerError(f"{label} has duplicate shard identities")
        count = _integer(
            shard.get("sequence_count"),
            label=f"{label}.shards[{index}].sequence_count",
            minimum=1,
        )
        start = _integer(
            shard.get("global_sample_start"),
            label=f"{label}.shards[{index}].global_sample_start",
        )
        if start != next_sample:
            raise GovernedControllerError(
                f"{label} sample ranges are not contiguous from zero"
            )
        _sha256_string(
            shard.get("output_sha256"),
            label=f"{label}.shards[{index}].output_sha256",
        )
        next_sample += count
        source_ids.add(source_id)
        shard_ids.add(shard_id)
        output_paths.add(output_path)
    weights = _mapping(
        source_map.get("mix_basis_points"),
        label=f"{label}.mix_basis_points",
    )
    normalized_weights: dict[str, int] = {}
    for source_id, raw_weight in weights.items():
        if not isinstance(source_id, str) or not source_id:
            raise GovernedControllerError(f"{label} has an invalid source ID")
        normalized_weights[source_id] = _integer(
            raw_weight,
            label=f"{label}.mix_basis_points.{source_id}",
            minimum=1,
        )
    if set(normalized_weights) != source_ids or sum(normalized_weights.values()) != 10_000:
        raise GovernedControllerError(
            f"{label} source weights do not cover the source map at 10,000 bp"
        )
    return {
        "sha256": _canonical_sha256(source_map),
        "prepared_dataset_fingerprint": prepared_fingerprint,
        "mix_basis_points": dict(sorted(normalized_weights.items())),
    }


def _source_mix_phase_contract(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    contract = _mapping(value, label=label)
    if contract.get("enabled") is not True:
        raise GovernedControllerError(f"{label} is not enabled")
    algorithm = contract.get("algorithm")
    if algorithm != "token-deficit-corrected-source-mix-bp-v2":
        raise GovernedControllerError(f"{label} algorithm is invalid")
    weights: dict[str, dict[str, int]] = {}
    for field in (
        "basis_points",
        "lineage_basis_points",
        "effective_basis_points",
    ):
        raw_weights = _mapping(contract.get(field), label=f"{label}.{field}")
        normalized: dict[str, int] = {}
        for source_id, raw_weight in raw_weights.items():
            if not isinstance(source_id, str) or not source_id:
                raise GovernedControllerError(
                    f"{label}.{field} has an invalid source ID"
                )
            normalized[source_id] = _integer(
                raw_weight,
                label=f"{label}.{field}.{source_id}",
                minimum=1,
            )
        if sum(normalized.values()) != 10_000:
            raise GovernedControllerError(f"{label}.{field} does not total 10,000 bp")
        weights[field] = dict(sorted(normalized.items()))
    if weights["basis_points"] != weights["effective_basis_points"]:
        raise GovernedControllerError(
            f"{label} compatibility/effective source weights differ"
        )
    weight_override = contract.get("weight_override")
    if not isinstance(weight_override, bool):
        raise GovernedControllerError(f"{label}.weight_override is invalid")
    return {
        "algorithm": algorithm,
        "source_map_sha256": _sha256_string(
            contract.get("source_map_sha256"),
            label=f"{label}.source_map_sha256",
        ),
        "dataset_fingerprint": _sha256_string(
            contract.get("dataset_fingerprint"),
            label=f"{label}.dataset_fingerprint",
        ),
        "basis_points": weights["basis_points"],
        "lineage_basis_points": weights["lineage_basis_points"],
        "effective_basis_points": weights["effective_basis_points"],
        "weight_override": weight_override,
        "seed": _integer(contract.get("seed"), label=f"{label}.seed"),
    }


def _authenticate_checkpoint_source_maps(
    plan: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-check config, capacity, checkpoint log, and both cursor source maps."""

    training_data = _mapping(plan.get("training_data"), label="plan.training_data")
    expected_phases = _mapping(
        training_data.get("phase_source_maps"),
        label="plan.training_data.phase_source_maps",
    )
    if set(expected_phases) != {"primary", "cooldown"}:
        raise GovernedControllerError(
            "plan has no authenticated primary/cooldown source-map bindings"
        )
    expected_algorithm = training_data.get("source_mix_algorithm")
    if expected_algorithm != "token-deficit-corrected-source-mix-bp-v2":
        raise GovernedControllerError("plan source-mix algorithm is invalid")
    raw_config_weights = _mapping(
        training_data.get("source_mix_basis_points"),
        label="plan.training_data.source_mix_basis_points",
    )
    config_weights = {
        str(source_id): _integer(
            weight,
            label=f"plan.training_data.source_mix_basis_points.{source_id}",
            minimum=1,
        )
        for source_id, weight in raw_config_weights.items()
    }
    if sum(config_weights.values()) != 10_000:
        raise GovernedControllerError(
            "plan primary source-mix weights do not total 10,000 bp"
        )
    config_weights = dict(sorted(config_weights.items()))
    config_override = training_data.get("source_mix_allow_weight_override")
    if not isinstance(config_override, bool):
        raise GovernedControllerError("plan source-mix override flag is invalid")
    config_seed = _integer(
        training_data.get("shuffle_seed"),
        label="plan.training_data.shuffle_seed",
    )

    extra = _mapping(metadata.get("extra"), label="checkpoint.metadata.extra")
    source_mix = _mapping(extra.get("source_mix"), label="checkpoint.source_mix")
    raw_phase_contracts = _mapping(
        source_mix.get("phases"),
        label="checkpoint.source_mix.phases",
    )
    if set(raw_phase_contracts) != {"primary", "cooldown"}:
        raise GovernedControllerError(
            "checkpoint source-mix log must bind both data phases"
        )
    phase_contracts = {
        phase: _source_mix_phase_contract(
            raw_phase_contracts[phase],
            label=f"checkpoint.source_mix.phases.{phase}",
        )
        for phase in ("primary", "cooldown")
    }
    primary_contract = _source_mix_phase_contract(
        source_mix,
        label="checkpoint.source_mix",
    )
    if primary_contract != phase_contracts["primary"]:
        raise GovernedControllerError(
            "checkpoint top-level source mix differs from its primary phase"
        )
    cooldown_start = _integer(
        source_mix.get("cooldown_start_tokens"),
        label="checkpoint.source_mix.cooldown_start_tokens",
        minimum=1,
    )
    if cooldown_start != training_data.get("quality_cooldown_start_tokens"):
        raise GovernedControllerError(
            "checkpoint cooldown transition differs from the final config"
        )

    data_cursor = _mapping(metadata.get("data_cursor"), label="checkpoint.data_cursor")
    cursor = _mapping(data_cursor.get("extra"), label="checkpoint.data_cursor.extra")
    if (
        cursor.get("kind") != "deterministic-source-mix-cooldown"
        or cursor.get("algorithm") != expected_algorithm
        or cursor.get("cooldown_start_tokens") != cooldown_start
        or cursor.get("seed") != config_seed
    ):
        raise GovernedControllerError(
            "checkpoint phase cursor contract differs from the final config"
        )
    raw_leaves = {
        "primary": cursor.get("primary_cursor"),
        "cooldown": cursor.get("cooldown_cursor"),
    }
    result: dict[str, Any] = {}
    for phase in ("primary", "cooldown"):
        leaf = _mapping(raw_leaves[phase], label=f"checkpoint.cursor.{phase}")
        if (
            leaf.get("kind") != "deterministic-source-mix"
            or leaf.get("algorithm") != expected_algorithm
            or leaf.get("seed") != config_seed
        ):
            raise GovernedControllerError(
                f"checkpoint {phase} cursor contract differs from the final config"
            )
        source_map = _source_map_identity(
            leaf.get("source_map"),
            label=f"checkpoint.cursor.{phase}.source_map",
        )
        contract = phase_contracts[phase]
        expected = _mapping(
            expected_phases.get(phase),
            label=f"plan.training_data.phase_source_maps.{phase}",
        )
        leaf_weights_raw = _mapping(
            leaf.get("weights_basis_points"),
            label=f"checkpoint.cursor.{phase}.weights_basis_points",
        )
        leaf_weights = {
            str(source_id): _integer(
                weight,
                label=f"checkpoint.cursor.{phase}.weights_basis_points.{source_id}",
                minimum=1,
            )
            for source_id, weight in leaf_weights_raw.items()
        }
        leaf_weights = dict(sorted(leaf_weights.items()))
        if (
            source_map["sha256"] != contract["source_map_sha256"]
            or source_map["sha256"] != expected.get("source_map_sha256")
            or source_map["prepared_dataset_fingerprint"]
            != expected.get("prepared_dataset_fingerprint")
            or leaf.get("prepared_dataset_fingerprint")
            != source_map["prepared_dataset_fingerprint"]
            or leaf.get("dataset_fingerprint") != contract["dataset_fingerprint"]
            or leaf_weights != contract["effective_basis_points"]
            or source_map["mix_basis_points"] != contract["lineage_basis_points"]
            or contract["seed"] != config_seed
        ):
            raise GovernedControllerError(
                f"checkpoint {phase} source-map identity differs from "
                "capacity/config/cursor evidence"
            )
        if phase == "primary":
            if (
                source_map["sha256"] != training_data.get("source_map_sha256")
                or leaf_weights != config_weights
                or contract["weight_override"] is not config_override
            ):
                raise GovernedControllerError(
                    "checkpoint primary source map differs from the final config"
                )
            if config_override is False and (
                contract["lineage_basis_points"] != config_weights
            ):
                raise GovernedControllerError(
                    "checkpoint primary lineage weights require an undeclared override"
                )
        elif (
            contract["weight_override"] is not False
            or contract["lineage_basis_points"] != contract["effective_basis_points"]
        ):
            raise GovernedControllerError(
                "checkpoint cooldown source weights contain an invalid override"
            )
        result[phase] = {
            **source_map,
            "source_mix_dataset_fingerprint": contract["dataset_fingerprint"],
            "effective_basis_points": contract["effective_basis_points"],
        }

    if (
        source_mix.get("cursor_critical_lineage_fingerprint")
        != cursor.get("critical_lineage_fingerprint")
        or source_mix.get("phase_committed_samples_by_source")
        != cursor.get("phase_committed_samples_by_source")
        or source_mix.get("phase_committed_tokens_by_source")
        != cursor.get("phase_committed_tokens_by_source")
        or source_mix.get("committed_samples_by_source")
        != cursor.get("committed_samples_by_source")
        or source_mix.get("committed_tokens_by_source")
        != cursor.get("committed_tokens_by_source")
    ):
        raise GovernedControllerError(
            "checkpoint source-mix ledger differs from its authenticated cursor"
        )
    return result


def _cursor_reuse_observation(metadata: Mapping[str, Any]) -> tuple[list[int], int, int]:
    cursor = _mapping(metadata.get("data_cursor"), label="checkpoint.data_cursor")
    epochs: list[int] = [
        _integer(cursor.get("epoch"), label="checkpoint.data_cursor.epoch")
    ]
    extra = _mapping(cursor.get("extra"), label="checkpoint.data_cursor.extra")
    leaves: list[Mapping[str, Any]]
    if extra.get("kind") == "deterministic-source-mix-cooldown":
        leaves = [
            _mapping(extra.get("primary_cursor"), label="cursor.primary_cursor"),
            _mapping(extra.get("cooldown_cursor"), label="cursor.cooldown_cursor"),
        ]
    else:
        leaves = [extra]
    reused_sequences = 0
    reused_tokens = 0
    for index, leaf in enumerate(leaves):
        source_map = _mapping(leaf.get("source_map"), label=f"cursor[{index}].source_map")
        shards = source_map.get("shards")
        if not isinstance(shards, list):
            raise GovernedControllerError(f"cursor[{index}] source_map has no shards")
        capacities: dict[str, int] = {}
        for shard_index, raw in enumerate(shards):
            shard = _mapping(raw, label=f"cursor[{index}].shards[{shard_index}]")
            source_id = str(shard.get("source_id"))
            capacity = _integer(
                shard.get("sequence_count"),
                label=f"cursor[{index}].shards[{shard_index}].sequence_count",
            )
            capacities[source_id] = capacities.get(source_id, 0) + capacity
        committed = _mapping(
            leaf.get("committed_samples_by_source"),
            label=f"cursor[{index}].committed_samples_by_source",
        )
        sequence_length = _integer(
            source_map.get("sequence_length"),
            label=f"cursor[{index}].source_map.sequence_length",
            minimum=1,
        )
        for source_id, raw_count in committed.items():
            count = _integer(
                raw_count,
                label=f"cursor[{index}].committed_samples_by_source.{source_id}",
            )
            overflow = max(count - capacities.get(str(source_id), 0), 0)
            reused_sequences += overflow
            reused_tokens += overflow * sequence_length
    return epochs, reused_sequences, reused_tokens


def _authenticate_checkpoint_execution_identity(
    plan: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-check checkpoint config, source, and dependency identities."""

    extra = _mapping(metadata.get("extra"), label="checkpoint.metadata.extra")
    plan_config = _mapping(plan.get("config"), label="plan.config")
    checkpoint_config = _mapping(
        extra.get("config"),
        label="checkpoint.metadata.extra.config",
    )
    source_tree = _mapping(plan.get("source_tree"), label="plan.source_tree")
    dependency = _mapping(
        plan.get("dependency_lock"),
        label="plan.dependency_lock",
    )
    expected_source = _sha256_string(
        source_tree.get("sha256"),
        label="plan.source_tree.sha256",
    )
    expected_dependency = _sha256_string(
        dependency.get("sha256"),
        label="plan.dependency_lock.sha256",
    )
    checkpoint_dependency_path = Path(str(extra.get("dependency_lock"))).resolve()
    expected_dependency_path = Path(str(dependency.get("path"))).resolve()
    expected_config_path = Path(str(plan_config.get("path"))).resolve()
    checkpoint_config_path = Path(str(checkpoint_config.get("path"))).resolve()
    expected_config_sha = _sha256_string(
        plan_config.get("sha256"),
        label="plan.config.sha256",
    )
    expected_preflight_fingerprint = _sha256_string(
        plan_config.get("preflight_fingerprint"),
        label="plan.config.preflight_fingerprint",
    )
    checkpoint_config_sha = _sha256_string(
        checkpoint_config.get("sha256"),
        label="checkpoint.config.sha256",
    )
    preflight_fingerprint = _sha256_string(
        checkpoint_config.get("preflight_fingerprint"),
        label="checkpoint.config.preflight_fingerprint",
    )
    if (
        checkpoint_config_path != expected_config_path
        or checkpoint_config_sha != expected_config_sha
        or preflight_fingerprint != expected_preflight_fingerprint
        or metadata.get("critical_fingerprint") != preflight_fingerprint
    ):
        raise GovernedControllerError(
            "checkpoint config identity differs from the governed plan"
        )
    if extra.get("source_tree_sha256") != expected_source:
        raise GovernedControllerError(
            "checkpoint Twen source-tree identity differs from the governed plan"
        )
    if (
        checkpoint_dependency_path != expected_dependency_path
        or extra.get("dependency_lock_sha256") != expected_dependency
    ):
        raise GovernedControllerError(
            "checkpoint dependency-lock identity differs from the governed plan"
        )
    return {
        "config_path": str(expected_config_path),
        "config_sha256": expected_config_sha,
        "config_preflight_fingerprint": preflight_fingerprint,
        "source_tree_sha256": expected_source,
        "dependency_lock": str(expected_dependency_path),
        "dependency_lock_sha256": expected_dependency,
    }


def authenticate_checkpoint_execution_identity(
    plan: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate a checkpoint's complete plan-bound execution identity.

    This public guard is intentionally cheap relative to drift/NLL evaluation
    and must run before any such artifact generation.  It reauthenticates the
    current readiness/config/source chain as well as the identities sealed in
    the checkpoint manifest.
    """

    verify_controller_sources(plan)
    metadata = _mapping(checkpoint.get("metadata"), label="checkpoint.metadata")
    return _authenticate_checkpoint_execution_identity(plan, metadata)


def build_authenticated_gate_observation(
    plan: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    metrics_path: str | Path,
    drift_path: str | Path,
    sweep_roots: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Construct gate input only from controller-authenticated evidence."""

    verify_controller_sources(plan)
    metadata = _mapping(checkpoint.get("metadata"), label="checkpoint.metadata")
    run = _mapping(plan.get("run"), label="plan.run")
    training_data = _mapping(plan.get("training_data"), label="plan.training_data")
    extra = _mapping(metadata.get("extra"), label="checkpoint.metadata.extra")
    execution_identity = _authenticate_checkpoint_execution_identity(plan, metadata)
    if (
        metadata.get("run_id") != run.get("run_id")
        or metadata.get("stage") != run.get("stage")
        or extra.get("allow_corpus_reuse") is not False
        or training_data.get("allow_corpus_reuse") is not False
        or extra.get("data_manifest_sha256") != training_data.get("manifest_sha256")
    ):
        raise GovernedControllerError("checkpoint training lineage differs from the plan")
    quality = _mapping(extra.get("quality_cooldown"), label="checkpoint.quality_cooldown")
    phase_attestation = _mapping(
        extra.get("phase_disjointness_attestation"),
        label="checkpoint.phase_disjointness_attestation",
    )
    if (
        quality.get("prepared_manifest_sha256")
        != training_data.get("quality_cooldown_manifest_sha256")
        or phase_attestation.get("sha256")
        != training_data.get("phase_disjointness_attestation_sha256")
    ):
        raise GovernedControllerError("checkpoint phase identities differ from the plan")
    source_maps = _authenticate_checkpoint_source_maps(plan, metadata)
    epochs, reused_sequences, reused_tokens = _cursor_reuse_observation(metadata)
    rows, metrics_prefix = read_authenticated_metrics_prefix(
        metrics_path,
        metadata,
    )
    drift = authenticate_drift_evidence(plan, checkpoint, drift_path)
    if set(sweep_roots) != {"primary", "cooldown"}:
        raise GovernedControllerError("both primary and cooldown evaluations are required")
    sweeps = {
        phase: _authenticate_generated_sweep(
            plan,
            checkpoint,
            phase=phase,
            root=sweep_roots[phase],
        )
        for phase in ("primary", "cooldown")
    }

    candidate_nll_sum = 0.0
    candidate_tokens = 0
    baseline_nll_sum = 0.0
    baseline_tokens = 0
    candidate_chinese_nll_sum = 0.0
    candidate_chinese_tokens = 0
    baseline_chinese_nll_sum = 0.0
    baseline_chinese_tokens = 0
    for sweep in sweeps.values():
        candidate = _mapping(sweep["candidate"], label="sweep.candidate")
        baseline = _mapping(sweep["baseline"], label="sweep.baseline")
        for row, prefix in ((candidate, "candidate"), (baseline, "baseline")):
            overall = _mapping(row.get("overall"), label=f"{prefix}.overall")
            tokens = _integer(
                overall.get("predicted_tokens"),
                label=f"{prefix}.overall.predicted_tokens",
                minimum=1,
            )
            nll_sum = _finite(overall.get("nll_sum"), label=f"{prefix}.overall.nll_sum")
            if prefix == "candidate":
                candidate_tokens += tokens
                candidate_nll_sum += nll_sum
            else:
                baseline_tokens += tokens
                baseline_nll_sum += nll_sum
            sources = row.get("sources")
            if not isinstance(sources, list):
                raise GovernedControllerError(f"{prefix} source inventory is invalid")
            chinese_rows = [
                _mapping(value, label=f"{prefix}.source")
                for value in sources
                if isinstance(value, Mapping)
                and str(value.get("source")).startswith("chinese_")
            ]
            if not chinese_rows:
                raise GovernedControllerError(f"{prefix} has no Chinese validation source")
            for chinese_row in chinese_rows:
                source_tokens = _integer(
                    chinese_row.get("predicted_tokens"),
                    label=f"{prefix}.chinese.predicted_tokens",
                    minimum=1,
                )
                source_nll_sum = _finite(
                    chinese_row.get("nll_sum"),
                    label=f"{prefix}.chinese.nll_sum",
                )
                if prefix == "candidate":
                    candidate_chinese_tokens += source_tokens
                    candidate_chinese_nll_sum += source_nll_sum
                else:
                    baseline_chinese_tokens += source_tokens
                    baseline_chinese_nll_sum += source_nll_sum

    formal = _mapping(plan.get("formal_evidence"), label="plan.formal_evidence")
    disjointness = _mapping(
        formal.get("formal_disjointness_attestation"),
        label="formal.formal_disjointness_attestation",
    )
    disjoint_path = Path(str(disjointness.get("path"))).resolve()
    if (
        not disjoint_path.is_file()
        or sha256_file(disjoint_path) != disjointness.get("sha256")
    ):
        raise GovernedControllerError("plan-bound formal disjointness evidence changed")
    return {
        "checkpoint": {
            "authenticated": True,
            "checkpoint_id": metadata.get("checkpoint_id"),
            "committed_tokens": metadata.get("committed_tokens"),
            "lineage_passed": True,
            "phase_identity_passed": True,
            "disjointness_passed": True,
            "reference_epochs": epochs,
            "reused_sequences": reused_sequences,
            "reused_tokens": reused_tokens,
            "source_maps": source_maps,
            "execution_identity": execution_identity,
        },
        "metrics": rows,
        "metrics_prefix": metrics_prefix,
        "drift": drift,
        "evaluation": {
            "authenticated": True,
            "aggregate_nll": candidate_nll_sum / candidate_tokens,
            "chinese_nll": candidate_chinese_nll_sum / candidate_chinese_tokens,
            "sweeps": sweeps,
        },
        "baseline": {
            "aggregate_nll": baseline_nll_sum / baseline_tokens,
            "chinese_nll": baseline_chinese_nll_sum / baseline_chinese_tokens,
        },
    }


def _gate(
    code: str,
    *,
    passed: bool,
    severity: str,
    observed: Any,
    limit: Any,
) -> dict[str, Any]:
    return {
        "code": code,
        "passed": bool(passed),
        "severity": severity,
        "observed": observed,
        "limit": limit,
    }


def evaluate_hard_gates(
    plan: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    history: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Evaluate checkpoint, reuse, finite/clip, drift, and NLL stop gates."""

    checkpoint = _mapping(observation.get("checkpoint"), label="observation.checkpoint")
    drift = _mapping(observation.get("drift"), label="observation.drift")
    evaluation = _mapping(observation.get("evaluation"), label="observation.evaluation")
    baseline = _mapping(observation.get("baseline"), label="observation.baseline")
    metrics_raw = observation.get("metrics")
    if not isinstance(metrics_raw, Sequence) or isinstance(metrics_raw, (str, bytes)):
        raise GovernedControllerError("observation.metrics must be a sequence")
    metrics = [_mapping(row, label=f"metrics[{index}]") for index, row in enumerate(metrics_raw)]
    if not metrics:
        raise GovernedControllerError("observation.metrics cannot be empty")

    gates = _mapping(plan.get("gates"), label="plan.gates")
    hard_stop = _mapping(gates.get("hard_stop"), label="plan.gates.hard_stop")
    validation_stop = _mapping(
        gates.get("validation_stop"),
        label="plan.gates.validation_stop",
    )
    required_finite = tuple(gates.get("required_finite_metrics", REQUIRED_FINITE_METRICS))
    finite_failures: list[str] = []
    for row_index, row in enumerate(metrics):
        for name in required_finite:
            try:
                _finite(row.get(name), label=f"metrics[{row_index}].{name}")
            except GovernedControllerError:
                finite_failures.append(f"{row_index}:{name}")

    grad_clip_norm = _finite(
        gates.get("grad_clip_norm", 1.0),
        label="plan.gates.grad_clip_norm",
        minimum=0.0,
    )
    rolling = metrics[-50:]
    clipped = sum(float(row["grad_norm"]) > grad_clip_norm for row in rolling)
    clip_fraction = clipped / len(rolling)
    clip_limit = _finite(
        hard_stop.get("rolling_50_step_clip_fraction_gt"),
        label="hard_stop.rolling_50_step_clip_fraction_gt",
        minimum=0.0,
    )

    epochs_raw = checkpoint.get("reference_epochs", [checkpoint.get("epoch")])
    if not isinstance(epochs_raw, Sequence) or isinstance(epochs_raw, (str, bytes)):
        raise GovernedControllerError("checkpoint.reference_epochs must be a sequence")
    epochs = [
        _integer(value, label=f"checkpoint.reference_epochs[{index}]")
        for index, value in enumerate(epochs_raw)
    ]
    reused_sequences = _integer(
        checkpoint.get("reused_sequences"),
        label="checkpoint.reused_sequences",
    )
    reused_tokens = _integer(
        checkpoint.get("reused_tokens"),
        label="checkpoint.reused_tokens",
    )
    scale_relative_l2 = _finite(
        drift.get("scale_relative_l2"),
        label="drift.scale_relative_l2",
        minimum=0.0,
    )
    scale_limit = _finite(
        hard_stop.get("scale_relative_l2_gt"),
        label="hard_stop.scale_relative_l2_gt",
        minimum=0.0,
    )

    checks = [
        _gate(
            "checkpoint_authenticated",
            passed=checkpoint.get("authenticated") is True,
            severity="stop",
            observed=checkpoint.get("authenticated"),
            limit=True,
        ),
        _gate(
            "checkpoint_lineage",
            passed=checkpoint.get("lineage_passed") is True,
            severity="stop",
            observed=checkpoint.get("lineage_passed"),
            limit=True,
        ),
        _gate(
            "phase_identity",
            passed=checkpoint.get("phase_identity_passed") is True,
            severity="stop",
            observed=checkpoint.get("phase_identity_passed"),
            limit=True,
        ),
        _gate(
            "formal_disjointness",
            passed=checkpoint.get("disjointness_passed") is True,
            severity="stop",
            observed=checkpoint.get("disjointness_passed"),
            limit=True,
        ),
        _gate(
            "reference_epoch_zero",
            passed=bool(epochs) and all(epoch == 0 for epoch in epochs),
            severity="stop",
            observed=epochs,
            limit=0,
        ),
        _gate(
            "no_corpus_reuse",
            passed=reused_sequences == 0 and reused_tokens == 0,
            severity="stop",
            observed={"sequences": reused_sequences, "tokens": reused_tokens},
            limit=0,
        ),
        _gate(
            "required_metrics_finite",
            passed=not finite_failures,
            severity="stop",
            observed=finite_failures,
            limit=[],
        ),
        _gate(
            "scale_relative_l2",
            passed=scale_relative_l2 <= scale_limit,
            severity="stop",
            observed=scale_relative_l2,
            limit=scale_limit,
        ),
        _gate(
            "rolling_clip_fraction",
            passed=clip_fraction <= clip_limit,
            severity="review",
            observed=clip_fraction,
            limit=clip_limit,
        ),
        _gate(
            "drift_authenticated",
            passed=drift.get("authenticated") is True,
            severity="stop",
            observed=drift.get("authenticated"),
            limit=True,
        ),
        _gate(
            "evaluation_authenticated",
            passed=evaluation.get("authenticated") is True,
            severity="stop",
            observed=evaluation.get("authenticated"),
            limit=True,
        ),
    ]

    aggregate = _finite(evaluation.get("aggregate_nll"), label="evaluation.aggregate_nll")
    chinese = _finite(evaluation.get("chinese_nll"), label="evaluation.chinese_nll")
    baseline_aggregate = _finite(
        baseline.get("aggregate_nll"),
        label="baseline.aggregate_nll",
    )
    baseline_chinese = _finite(baseline.get("chinese_nll"), label="baseline.chinese_nll")
    aggregate_delta = aggregate - baseline_aggregate
    chinese_delta = chinese - baseline_chinese
    prior = [
        {
            "checkpoint_id": row.get("checkpoint_id"),
            "aggregate_nll": _finite(
                row.get("aggregate_nll"),
                label=f"history[{index}].aggregate_nll",
            ),
            "chinese_nll": _finite(
                row.get("chinese_nll"),
                label=f"history[{index}].chinese_nll",
            ),
        }
        for index, row in enumerate(history)
    ]
    series = [
        *prior,
        {
            "checkpoint_id": checkpoint.get("checkpoint_id"),
            "aggregate_nll": aggregate,
            "chinese_nll": chinese,
        },
    ]

    aggregate_once = _finite(
        validation_stop.get("aggregate_nll_above_v3_once"),
        label="validation_stop.aggregate_nll_above_v3_once",
        minimum=0.0,
    )
    aggregate_twice = _finite(
        validation_stop.get("aggregate_nll_above_v3_twice"),
        label="validation_stop.aggregate_nll_above_v3_twice",
        minimum=0.0,
    )
    chinese_once = _finite(
        validation_stop.get("chinese_nll_above_v3_once"),
        label="validation_stop.chinese_nll_above_v3_once",
        minimum=0.0,
    )
    chinese_twice = _finite(
        validation_stop.get("chinese_nll_above_v3_twice"),
        label="validation_stop.chinese_nll_above_v3_twice",
        minimum=0.0,
    )
    prior_aggregate_delta = (
        prior[-1]["aggregate_nll"] - baseline_aggregate if prior else None
    )
    prior_chinese_delta = prior[-1]["chinese_nll"] - baseline_chinese if prior else None
    checks.extend(
        [
            _gate(
                "aggregate_nll_single_regression",
                passed=aggregate_delta <= aggregate_once,
                severity="stop",
                observed=aggregate_delta,
                limit=aggregate_once,
            ),
            _gate(
                "aggregate_nll_consecutive_regression",
                passed=not (
                    prior_aggregate_delta is not None
                    and prior_aggregate_delta > aggregate_twice
                    and aggregate_delta > aggregate_twice
                ),
                severity="stop",
                observed=[prior_aggregate_delta, aggregate_delta],
                limit=aggregate_twice,
            ),
            _gate(
                "chinese_nll_single_regression",
                passed=chinese_delta <= chinese_once,
                severity="stop",
                observed=chinese_delta,
                limit=chinese_once,
            ),
            _gate(
                "chinese_nll_consecutive_regression",
                passed=not (
                    prior_chinese_delta is not None
                    and prior_chinese_delta > chinese_twice
                    and chinese_delta > chinese_twice
                ),
                severity="stop",
                observed=[prior_chinese_delta, chinese_delta],
                limit=chinese_twice,
            ),
        ]
    )

    regression_limit = _finite(
        validation_stop.get("nll_regression_from_best_twice"),
        label="validation_stop.nll_regression_from_best_twice",
        minimum=0.0,
    )
    consecutive_best_regressions = False
    if len(series) >= 3:
        last_two_flags: list[bool] = []
        for index in range(len(series) - 2, len(series)):
            best_before = min(row["aggregate_nll"] for row in series[:index])
            last_two_flags.append(series[index]["aggregate_nll"] - best_before > regression_limit)
        consecutive_best_regressions = all(last_two_flags)
    checks.append(
        _gate(
            "aggregate_nll_regression_from_best_twice",
            passed=not consecutive_best_regressions,
            severity="stop",
            observed=consecutive_best_regressions,
            limit=regression_limit,
        )
    )

    improvement_limit = _finite(
        validation_stop.get("two_major_evaluations_improvement_below"),
        label="validation_stop.two_major_evaluations_improvement_below",
        minimum=0.0,
    )
    last_improvements: list[float] = []
    plateau = False
    if len(series) >= 3:
        last_improvements = [
            series[index - 1]["aggregate_nll"] - series[index]["aggregate_nll"]
            for index in range(len(series) - 2, len(series))
        ]
        plateau = all(improvement < improvement_limit for improvement in last_improvements)
    checks.append(
        _gate(
            "two_interval_improvement",
            passed=not plateau,
            severity="terminal_stop",
            observed=last_improvements,
            limit=improvement_limit,
        )
    )

    hard_failures = [
        check["code"]
        for check in checks
        if not check["passed"] and check["severity"] == "stop"
    ]
    review_failures = [
        check["code"]
        for check in checks
        if not check["passed"] and check["severity"] == "review"
    ]
    terminal_failures = [
        check["code"]
        for check in checks
        if not check["passed"] and check["severity"] == "terminal_stop"
    ]
    if hard_failures:
        action = "stop"
    elif review_failures:
        action = "review"
    elif terminal_failures:
        action = "terminal_stop"
    else:
        run = _mapping(plan.get("run"), label="plan.run")
        action = (
            "complete"
            if _integer(
                checkpoint.get("committed_tokens"),
                label="checkpoint.committed_tokens",
            )
            >= _integer(run.get("max_tokens"), label="plan.run.max_tokens", minimum=1)
            else "resume"
        )

    tie_tolerance = _finite(
        validation_stop.get("selection_tie_tolerance"),
        label="validation_stop.selection_tie_tolerance",
        minimum=0.0,
    )
    global_minimum = min(row["aggregate_nll"] for row in series)
    selected = next(
        row
        for row in series
        if row["aggregate_nll"] <= global_minimum + tie_tolerance
    )
    result = {
        "kind": GATE_KIND,
        "schema_version": SCHEMA_VERSION,
        "plan_id": plan.get("plan_id"),
        "action": action,
        "checks": checks,
        "failures": {
            "stop": hard_failures,
            "review": review_failures,
            "terminal_stop": terminal_failures,
        },
        "nll": {
            "aggregate": aggregate,
            "aggregate_delta_from_v3": aggregate_delta,
            "chinese": chinese,
            "chinese_delta_from_v3": chinese_delta,
        },
        "selection": {
            "checkpoint_id": selected.get("checkpoint_id"),
            "aggregate_nll": selected["aggregate_nll"],
            "tie_tolerance": tie_tolerance,
            "keeps_earlier_within_tolerance": True,
        },
    }
    result["decision_fingerprint"] = _canonical_sha256(result)
    return result


__all__ = [
    "GATE_KIND",
    "PLAN_KIND",
    "SCHEMA_VERSION",
    "STATE_KIND",
    "GovernedControllerError",
    "authenticate_checkpoint",
    "authenticate_checkpoint_execution_identity",
    "authenticate_drift_evidence",
    "authorize_run",
    "build_authenticated_gate_observation",
    "build_governed_plan",
    "build_train_command",
    "controller_status",
    "evaluate_hard_gates",
    "expected_run_ack",
    "generate_checkpoint_sweep",
    "generate_drift_evidence",
    "initial_controller_state",
    "load_controller_state",
    "read_authenticated_metrics_prefix",
    "read_canonical_metrics",
    "render_train_command",
    "verify_controller_sources",
    "write_controller_state",
]
