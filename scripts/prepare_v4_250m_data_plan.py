#!/usr/bin/env python3
"""Generate the fail-closed v4 225M+25M data and training drafts.

This script performs no network access and never starts training.  It derives
the stage recipes from the already-reviewed v4 schema-v2 recipe, adds the
remote identities audited for the new stage-specific shards, and emits an
intentionally non-loadable training config until authenticated prepared
manifest identities replace the ``PENDING_*`` sentinels.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_RECIPE = ROOT / "locks/base-data-sources-v4.json"
BASE_CONFIG = ROOT / "configs/base/dense-v4-16m-smoke.yaml"

PRIMARY_RECIPE = ROOT / "locks/base-data-sources-v4-primary.json"
COOLDOWN_RECIPE = ROOT / "locks/base-data-sources-v4-cooldown.json"
PRIMARY_RESOLVED = ROOT / "locks/base-data-sources-v4-primary.resolved.json"
COOLDOWN_RESOLVED = ROOT / "locks/base-data-sources-v4-cooldown.resolved.json"
CAPACITY_ATTESTATION = ROOT / "locks/base-data-sources-v4-250m.capacity-attestation.json"
BLOCKED_CONFIG = ROOT / "configs/base/dense-v4-250m-pilot.blocked.yaml"
READINESS = ROOT / "locks/base-dense-v4-250m-pilot.readiness.json"

MATERIALIZATION_PROFILE = "materialization"
PRIMARY_TRAINING_TOKENS = 225_000_000
COOLDOWN_TRAINING_TOKENS = 25_000_000
GLOBAL_BATCH_TOKENS = 262_144
# Recipes request a small integral-token guard above max_tokens+one global
# batch.  Per-source quota arithmetic in schema v2 is exact at 10,000 bp.
PRIMARY_MATERIALIZATION_TOKENS = 225_270_000
COOLDOWN_MATERIALIZATION_TOKENS = 25_270_000
PRIMARY_VALIDATION_TOKENS = 1_000_000
COOLDOWN_VALIDATION_TOKENS = 250_000
SEQUENCE_LENGTH = 4096
RAW_SAFETY_MULTIPLIER = 1.10
FORMAL_ADAPTER_LR = 3.0e-5
FORMAL_SCALE_LR = 3.0e-6
FORMAL_WARMUP_TOKENS = 10_000_000
FORMAL_CHECKPOINT_EVERY_STEPS = 50
PAUSE_EVALUATION_TOKENS = (
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
REQUIRED_ADDITIONAL_VALIDATION_SOURCES = (
    "science_arxiv_open_permissive",
    "code_stackv2_edu_permissive",
    "multilingual_common_corpus_permissive",
    "education_libretexts_permissive",
    "public_domain_project_gutenberg",
)

RETENTION = {
    "english_fineweb_edu_dedup": (0.9499, "base-v4-smoke-r3-r4-governance"),
    "chinese_fineweb2_cmn_hani": (0.8702, "base-v4-smoke-r3-r4-governance"),
    "math_finemath_4plus": (0.7374, "base-v4-smoke-r3-r4-governance"),
    "code_github_clean_allowlisted": (0.7585, "base-v4-smoke-r3-r4-governance"),
    "science_cosmopedia_openstax": (0.8577, "base-v4-smoke-r3-r4-governance"),
    "science_cosmopedia_stanford": (0.7454, "base-v4-smoke-r3-r4-governance"),
    "science_arxiv_open_permissive": (
        0.4534,
        "base-v4-smoke-r3-r4-governance",
    ),
    "code_stackv2_edu_permissive": (0.9263, "base-v4-smoke-r3-r4-governance"),
    "multilingual_common_corpus_permissive": (
        0.8042,
        "base-v4-smoke-r3-r4-governance",
    ),
    "education_libretexts_permissive": (0.9167, "base-v4-smoke-r3-r4-governance"),
    "public_domain_project_gutenberg": (0.3444, "base-v4-smoke-r3-r4-governance"),
}

PRIMARY_MIX = {
    "chinese_fineweb2_cmn_hani": 2400,
    "english_fineweb_edu_dedup": 2500,
    "math_finemath_4plus": 1400,
    "code_github_clean_allowlisted": 900,
    "science_cosmopedia_openstax": 400,
    "science_cosmopedia_stanford": 300,
    "science_arxiv_open_permissive": 1200,
    "code_stackv2_edu_permissive": 400,
    "multilingual_common_corpus_permissive": 500,
}

COOLDOWN_MIX = {
    "chinese_fineweb2_cmn_hani": 2500,
    "math_finemath_4plus": 2200,
    "science_arxiv_open_permissive": 2200,
    "science_cosmopedia_openstax": 1000,
    "science_cosmopedia_stanford": 800,
    "education_libretexts_permissive": 400,
    "public_domain_project_gutenberg": 400,
    "code_github_clean_allowlisted": 500,
}

ALTERNATE_FILES = {
    "chinese_fineweb2_cmn_hani": {
        "path": "data/cmn_Hani/train/004_00072.parquet",
        "size": 1_183_319_384,
        "sha256": "1a12f849443704a260195472bfd4d047d22a0e4db58fce66d870a443e6ce85be",
    },
    "math_finemath_4plus": {
        "path": "finemath-4plus/train-00061-of-00064.parquet",
        "size": 288_814_138,
        "sha256": "c78473e2f04cefe953ebe26c8d5fdee1b1a4b70c4b95a0210314304a9ba175e7",
    },
    "code_github_clean_allowlisted": {
        "path": "data/train-00571-of-00880.parquet",
        "size": 352_793_086,
        "sha256": "035dc344e445b1a5ee2471f5225a9b39ddf12d69fdedf14a895f559cbf05e4c9",
    },
    "science_cosmopedia_openstax": {
        "path": "data/openstax/train-00000-of-00002.parquet",
        "size": 173_501_101,
        "sha256": "bb0d51bf863f11feafb99ea9af56acec1b8b6837413f51a7498282c7e8f37381",
    },
    "science_cosmopedia_stanford": {
        "path": "data/stanford/train-00003-of-00013.parquet",
        "size": 254_262_994,
        "sha256": "2190925f386734e57e66da066f543937a65d6d65b4afe4dbf2f4fdfe8983286a",
    },
    "science_arxiv_open_permissive": {
        "path": "arxiv-papers-0006.json.gz",
        "size": 840_230_674,
        "sha256": "60dbfe42596e6255eb2e54566c715c2ebddb77df0d6030154a293f570d933f32",
    },
}

PRIMARY_FILES = {
    "science_arxiv_open_permissive": {
        "path": "arxiv-papers-0005.json.gz",
        "size": 842_526_499,
        "sha256": "f54481c7318b01aa8e459d1da519d839ed2af694fd9a2c33b86d3b8be63e7470",
    }
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _raw_quota(clean_quota: int, retention: float) -> int:
    return int(math.ceil(clean_quota / retention * RAW_SAFETY_MULTIPLIER / 1000.0) * 1000)


def _make_recipe(
    base: dict[str, Any],
    *,
    stage: str,
    mix: dict[str, int],
    materialization_tokens: int,
    training_tokens: int,
    validation_tokens: int,
) -> dict[str, Any]:
    if sum(mix.values()) != 10_000:
        raise ValueError(f"{stage} mix does not total 10,000 basis points")
    source_templates = {str(source["source_id"]): source for source in base["sources"]}
    sources: list[dict[str, Any]] = []
    capacity_rows: list[dict[str, Any]] = []
    for source_id, basis_points in mix.items():
        source = copy.deepcopy(source_templates[source_id])
        if stage == "primary" and source_id in PRIMARY_FILES:
            locked = PRIMARY_FILES[source_id]
            source["file_patterns"] = [locked["path"]]
            source["locked_files"] = [locked]
        if stage == "cooldown" and source_id in ALTERNATE_FILES:
            locked = ALTERNATE_FILES[source_id]
            source["file_patterns"] = [locked["path"]]
            source["locked_files"] = [locked]
        source["mix_basis_points"] = basis_points
        clean_quota = materialization_tokens * basis_points // 10_000
        validation_quota = validation_tokens * basis_points // 10_000
        retention, evidence = RETENTION[source_id]
        raw_quota = _raw_quota(clean_quota, retention)
        source["train_token_quotas"] = {MATERIALIZATION_PROFILE: clean_quota}
        source["validation_token_quota"] = validation_quota
        source["capacity_plan"] = {
            "clean_materialization_token_quota": clean_quota,
            "raw_materialization_token_quota": raw_quota,
            "governance_retention_rate": retention,
            "retention_evidence": evidence,
            "raw_safety_multiplier": RAW_SAFETY_MULTIPLIER,
            "passed": False,
        }
        sources.append(source)
        capacity_rows.append(
            {
                "source_id": source_id,
                "mix_basis_points": basis_points,
                "required_clean_tokens": clean_quota,
                "planned_raw_tokens": raw_quota,
                "governance_retention_rate": retention,
                "retention_evidence": evidence,
                "passed": False,
            }
        )

    recipe = copy.deepcopy(base)
    recipe["recipe_id"] = f"base-v4-250m-{stage}-20260727-r1"
    recipe["activation"] = {
        **recipe["activation"],
        "reason": (
            "Source resolution and corpus materialization are implemented. "
            "Training remains disabled until the independent capacity, "
            "quality, license, and phase-disjointness attestation passes."
        ),
    }
    recipe["objective"] = {
        "route": "base_dense_adapter",
        "training_text_objective": "causal_ntp_plus_qwen35_native_mtp",
        "teacher_logits_kd": False,
        "teacher_hidden_alignment": False,
    }
    recipe["split"] = {
        **recipe["split"],
        "seed": f"twen-base-v4-250m-{stage}-20260727",
    }
    recipe["profiles"] = {
        MATERIALIZATION_PROFILE: {
            "description": (
                f"{stage} protected materialization capacity for the 250M "
                "pilot; this is not a training-launch authorization"
            ),
            "train_tokens": materialization_tokens,
        }
    }
    recipe["validation_tokens"] = validation_tokens
    existing_basis_points = sum(
        int(source["mix_basis_points"])
        for source in sources
        if source["origin_group"] == "existing"
    )
    recipe["mix_contract"] = {
        **recipe["mix_contract"],
        "existing_sources_basis_points": existing_basis_points,
        "new_sources_basis_points": 10_000 - existing_basis_points,
    }
    recipe["sources"] = sources
    recipe["stage_contract"] = {
        "stage": stage,
        "planned_training_tokens": training_tokens,
        "materialization_tokens": materialization_tokens,
        "global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "sequence_length": SEQUENCE_LENGTH,
        "minimum_prepared_tokens": training_tokens + GLOBAL_BATCH_TOKENS,
        "minimum_prepared_samples": math.ceil(
            (training_tokens + GLOBAL_BATCH_TOKENS) / SEQUENCE_LENGTH
        ),
        "raw_safety_multiplier": RAW_SAFETY_MULTIPLIER,
        "raw_materialization_tokens": sum(int(row["planned_raw_tokens"]) for row in capacity_rows),
        "sources": capacity_rows,
    }
    recipe["launch_policy"] = {
        "launch_enabled": False,
        "required_attestation": ("locks/base-data-sources-v4-250m.capacity-attestation.json"),
        "data_allow_corpus_reuse": False,
    }
    return recipe


def _make_blocked_config() -> dict[str, Any]:
    value = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{BASE_CONFIG} must contain a YAML object")
    value["run_id"] = "base-dense-v4-250m-pilot"
    value["checkpoint"]["output_dir"] = "runs/base-dense-v4-250m-pilot"
    value["checkpoint"]["every_steps"] = FORMAL_CHECKPOINT_EVERY_STEPS
    data = value["data"]
    data["allow_corpus_reuse"] = False
    data["global_batch_tokens"] = GLOBAL_BATCH_TOKENS
    data["manifest_path"] = "PENDING_PRIMARY_PREPARED_MANIFEST"
    data["manifest_sha256"] = "PENDING_PRIMARY_PREPARED_MANIFEST_SHA256"
    data["source_map_sha256"] = "PENDING_PRIMARY_SOURCE_MAP_SHA256"
    data["source_mix_allow_weight_override"] = False
    data["source_mix_basis_points"] = dict(sorted(PRIMARY_MIX.items()))
    data["quality_cooldown_manifest_path"] = "PENDING_COOLDOWN_PREPARED_MANIFEST"
    data["quality_cooldown_manifest_sha256"] = "PENDING_COOLDOWN_PREPARED_MANIFEST_SHA256"
    data["quality_cooldown_start_tokens"] = PRIMARY_TRAINING_TOKENS
    data["phase_disjointness_attestation_path"] = (
        "PENDING_PRIMARY_COOLDOWN_PHASE_DISJOINTNESS_ATTESTATION"
    )
    data["phase_disjointness_attestation_sha256"] = (
        "PENDING_PRIMARY_COOLDOWN_PHASE_DISJOINTNESS_ATTESTATION_SHA256"
    )
    losses = value["losses"]
    losses.update(
        {
            "anchor_kl": 0.0,
            "dense_oracle": 0.0,
            "hidden_alignment": 0.0,
            "teacher_kd": 0.0,
            "mtp": 0.1,
            "ntp": 1.0,
        }
    )
    optimizer = value["optimizer"]
    optimizer.update(
        {
            "adapter_lr": FORMAL_ADAPTER_LR,
            "adapter_optimizer": "muon",
            "lora_lr": FORMAL_ADAPTER_LR,
            "lr_schedule": "cosine",
            "max_tokens": PRIMARY_TRAINING_TOKENS + COOLDOWN_TRAINING_TOKENS,
            "min_lr_ratio": 0.1,
            "scale_lr": FORMAL_SCALE_LR,
            "warmup_tokens": FORMAL_WARMUP_TOKENS,
        }
    )
    value["runtime"]["teacher_cpu_offload"] = False
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved_identity(recipe_path: Path, lock_path: Path) -> dict[str, Any]:
    recipe = _load_json(recipe_path)
    identity: dict[str, Any] = {
        "path": str(lock_path.relative_to(ROOT)),
        "sha256": None,
        "remote_identity_verification": None,
        "passed": False,
    }
    if not lock_path.is_file():
        return identity
    lock = _load_json(lock_path)
    verification = (
        lock.get("materialization_audit", {}).get("remote_identity_verification")
        if isinstance(lock.get("materialization_audit"), dict)
        else None
    )
    passed = (
        lock.get("recipe_id") == recipe.get("recipe_id")
        and lock.get("recipe_sha256") == _sha256(recipe_path)
        and isinstance(verification, str)
        and verification.startswith("verified_against_")
    )
    return {
        **identity,
        "sha256": _sha256(lock_path),
        "remote_identity_verification": verification,
        "passed": passed,
    }


def _capacity_stage(
    *,
    stage: str,
    recipe_path: Path,
    lock_path: Path,
    training_tokens: int,
) -> dict[str, Any]:
    recipe = _load_json(recipe_path)
    required_tokens = training_tokens + GLOBAL_BATCH_TOKENS
    required_samples = math.ceil(required_tokens / SEQUENCE_LENGTH)
    protected_sample_tokens = required_samples * SEQUENCE_LENGTH
    source_rows = recipe["stage_contract"]["sources"]
    return {
        "stage": stage,
        "recipe": {
            "path": str(recipe_path.relative_to(ROOT)),
            "sha256": _sha256(recipe_path),
            "recipe_id": recipe["recipe_id"],
        },
        "resolved_lock": _resolved_identity(recipe_path, lock_path),
        "required_training_tokens": training_tokens,
        "required_tail_batch_tokens": GLOBAL_BATCH_TOKENS,
        "required_prepared_tokens": required_tokens,
        "required_prepared_samples": required_samples,
        "protected_sample_tokens": protected_sample_tokens,
        "protected_sample_margin_tokens": protected_sample_tokens - required_tokens,
        "planned_clean_materialization_tokens": recipe["profiles"][MATERIALIZATION_PROFILE][
            "train_tokens"
        ],
        "planned_raw_materialization_tokens": recipe["stage_contract"][
            "raw_materialization_tokens"
        ],
        "locked_remote_bytes": sum(
            int(item["size"]) for source in recipe["sources"] for item in source["locked_files"]
        ),
        "prepared_identity": {
            "manifest_path": None,
            "manifest_sha256": None,
            "dataset_fingerprint": None,
            "source_map_sha256": None,
            "available_unique_tokens": None,
            "available_unique_samples": None,
            "margin_tokens": None,
            "passed": False,
        },
        "per_source_capacity": [
            {
                **row,
                "available_governed_unique_tokens": None,
                "margin_tokens": None,
                "passed": False,
            }
            for row in source_rows
        ],
        "quality_audit": {
            "audit_attestation_path": None,
            "audit_attestation_sha256": None,
            "passed": False,
        },
        "license_audit": {
            "materialized_attribution_manifest_sha256": None,
            "passed": False,
        },
        "passed": False,
    }


def _make_capacity_attestation() -> dict[str, Any]:
    primary = _capacity_stage(
        stage="primary",
        recipe_path=PRIMARY_RECIPE,
        lock_path=PRIMARY_RESOLVED,
        training_tokens=PRIMARY_TRAINING_TOKENS,
    )
    cooldown = _capacity_stage(
        stage="cooldown",
        recipe_path=COOLDOWN_RECIPE,
        lock_path=COOLDOWN_RESOLVED,
        training_tokens=COOLDOWN_TRAINING_TOKENS,
    )
    return {
        "schema_version": 1,
        "kind": "twen_v4_250m_capacity_attestation",
        "status": "blocked_pending_materialization",
        "launch_enabled": False,
        "training_started": False,
        "training_contract": {
            "max_tokens": 250_000_000,
            "cooldown_start_tokens": 225_000_000,
            "global_batch_tokens": GLOBAL_BATCH_TOKENS,
            "sequence_length": SEQUENCE_LENGTH,
            "allow_corpus_reuse": False,
            "complete_tail_batch_required": True,
        },
        "config": {
            "path": str(BLOCKED_CONFIG.relative_to(ROOT)),
            "sha256": _sha256(BLOCKED_CONFIG),
            "contains_pending_identity_sentinels": True,
        },
        "stages": {"primary": primary, "cooldown": cooldown},
        "phase_disjointness": {
            "stable_id_exact": {
                "algorithm": "authenticated-stable-id-set-intersection-v1",
                "result": None,
                "passed": False,
            },
            "normalized_text_exact": {
                "algorithm": "unicode-nfkc-whitespace-sha256-intersection-v1",
                "result": None,
                "passed": False,
            },
            "near_duplicate": {
                "algorithm": "lexical-5gram-one-permutation-minhash-lsh-v1",
                "threshold": 0.8,
                "result": None,
                "passed": False,
            },
        },
        "overall": {
            "required_clean_tokens": (
                primary["required_prepared_tokens"] + cooldown["required_prepared_tokens"]
            ),
            "available_clean_tokens": None,
            "margin_tokens": None,
            "passed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare generated bytes with checked-in artifacts without writing",
    )
    args = parser.parse_args()
    base = _load_json(BASE_RECIPE)
    primary = _make_recipe(
        base,
        stage="primary",
        mix=PRIMARY_MIX,
        materialization_tokens=PRIMARY_MATERIALIZATION_TOKENS,
        training_tokens=PRIMARY_TRAINING_TOKENS,
        validation_tokens=PRIMARY_VALIDATION_TOKENS,
    )
    cooldown = _make_recipe(
        base,
        stage="cooldown",
        mix=COOLDOWN_MIX,
        materialization_tokens=COOLDOWN_MATERIALIZATION_TOKENS,
        training_tokens=COOLDOWN_TRAINING_TOKENS,
        validation_tokens=COOLDOWN_VALIDATION_TOKENS,
    )
    config = _make_blocked_config()
    generated_json = {
        PRIMARY_RECIPE: json.dumps(primary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        COOLDOWN_RECIPE: json.dumps(cooldown, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    }
    config_bytes = yaml.safe_dump(
        config,
        allow_unicode=True,
        sort_keys=False,
    )
    if args.check:
        for path, expected in generated_json.items():
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                raise SystemExit(f"generated artifact differs: {path}")
        if (
            not BLOCKED_CONFIG.is_file()
            or BLOCKED_CONFIG.read_text(encoding="utf-8") != config_bytes
        ):
            raise SystemExit(f"generated artifact differs: {BLOCKED_CONFIG}")
        config_sha256 = _sha256(BLOCKED_CONFIG)
        capacity = _load_json(CAPACITY_ATTESTATION)
        capacity_config = capacity.get("config")
        if (
            not isinstance(capacity_config, dict)
            or capacity_config.get("path") != str(BLOCKED_CONFIG.relative_to(ROOT))
            or capacity_config.get("sha256") != config_sha256
        ):
            raise SystemExit(
                f"generated artifact has stale config identity: {CAPACITY_ATTESTATION}"
            )
        readiness = _load_json(READINESS)
        if (
            readiness.get("config_path") != str(BLOCKED_CONFIG.relative_to(ROOT))
            or readiness.get("config_sha256") != config_sha256
        ):
            raise SystemExit(f"generated artifact has stale config identity: {READINESS}")
        return 0

    for path, value in ((PRIMARY_RECIPE, primary), (COOLDOWN_RECIPE, cooldown)):
        _write_json(path, value)
    BLOCKED_CONFIG.write_text(config_bytes, encoding="utf-8")
    capacity_attestation = _make_capacity_attestation()
    _write_json(CAPACITY_ATTESTATION, capacity_attestation)
    primary_resolved = _resolved_identity(PRIMARY_RECIPE, PRIMARY_RESOLVED)
    cooldown_resolved = _resolved_identity(COOLDOWN_RECIPE, COOLDOWN_RESOLVED)
    unresolved_locks = [
        stage
        for stage, identity in (
            ("primary", primary_resolved),
            ("cooldown", cooldown_resolved),
        )
        if not identity["passed"]
    ]
    blockers = [
        "prepared manifest and source-map PENDING sentinels are unresolved",
        "per-source governed unique capacity has not been attested",
        "primary/cooldown stable-ID exact and near disjointness has not passed",
        "ArXiv per-document versioned-license yield is not yet materialized",
        "Chinese semantic conversion noise remains a manual/statistical review gate",
        ("frozen validation does not yet cover ArXiv and every newly introduced formal source"),
    ]
    if unresolved_locks:
        blockers.insert(
            0,
            "remote-resolved locks are unbound for: " + ", ".join(unresolved_locks),
        )
    readiness = {
        "schema_version": 1,
        "kind": "twen_v4_250m_pilot_readiness",
        "launch_enabled": False,
        "training_started": False,
        "config_path": str(BLOCKED_CONFIG.relative_to(ROOT)),
        "config_sha256": _sha256(BLOCKED_CONFIG),
        "fork_from": "runs/base-dense-v3-500m/step-000000001912-milestone-complete",
        "contract": {
            "max_tokens": 250_000_000,
            "cooldown_start_tokens": 225_000_000,
            "global_batch_tokens": GLOBAL_BATCH_TOKENS,
            "allow_corpus_reuse": False,
            "adapter_optimizer": "muon",
            "adapter_lr": FORMAL_ADAPTER_LR,
            "lora_lr": FORMAL_ADAPTER_LR,
            "scale_lr": FORMAL_SCALE_LR,
            "warmup_tokens": FORMAL_WARMUP_TOKENS,
            "lr_schedule": "cosine",
            "min_lr_ratio": 0.1,
            "objective": "base_text_ntp_plus_native_mtp_no_9b_logits",
        },
        "fork_policy": {
            "required_model_only_checkpoint": (
                "runs/base-dense-v3-500m/step-000000001912-milestone-complete"
            ),
            "reset_optimizer_and_scheduler": True,
            "forbidden_warm_starts": [
                "runs/base-dense-v4-16m-smoke",
                "runs/base-dense-v4-13m-low-lr-calibration",
            ],
        },
        "pause_evaluation_policy": {
            "checkpoint_every_steps": FORMAL_CHECKPOINT_EVERY_STEPS,
            "pause_at_committed_tokens": list(PAUSE_EVALUATION_TOKENS),
            "required_additional_validation_sources": list(REQUIRED_ADDITIONAL_VALIDATION_SOURCES),
            "hard_stop": {
                "nonfinite_metric": True,
                "data_reuse_or_epoch_gt_zero": True,
                "phase_identity_or_disjointness_failure": True,
                "checkpoint_lineage_failure": True,
                "scale_relative_l2_gt": 0.05,
                "rolling_50_step_clip_fraction_gt": 0.01,
            },
            "validation_stop": {
                "aggregate_nll_above_v3_once": 0.010,
                "aggregate_nll_above_v3_twice": 0.005,
                "chinese_nll_above_v3_once": 0.05,
                "chinese_nll_above_v3_twice": 0.03,
                "nll_regression_from_best_twice": 0.001,
                "two_major_evaluations_improvement_below": 0.0001,
                "selection_tie_tolerance": 0.0001,
            },
        },
        "recipe_identities": {
            "primary": {
                "path": str(PRIMARY_RECIPE.relative_to(ROOT)),
                "sha256": _sha256(PRIMARY_RECIPE),
            },
            "cooldown": {
                "path": str(COOLDOWN_RECIPE.relative_to(ROOT)),
                "sha256": _sha256(COOLDOWN_RECIPE),
            },
        },
        "resolved_lock_identities": {
            "primary": primary_resolved,
            "cooldown": cooldown_resolved,
        },
        "required_capacity_attestation": (
            "locks/base-data-sources-v4-250m.capacity-attestation.json"
        ),
        "blockers": blockers,
        "launch_command_after_all_gates_pass": [
            ".venv/bin/python",
            "-m",
            "twen.cli",
            "train",
            "--stage",
            "dense-oracle",
            "--config",
            str(BLOCKED_CONFIG.relative_to(ROOT)),
            "--resume",
            "none",
            "--fork-from",
            "runs/base-dense-v3-500m/step-000000001912-milestone-complete",
            "--progress",
            "always",
        ],
    }
    _write_json(READINESS, readiness)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
