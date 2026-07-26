from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

from twen.cli import _prepared_identity_payload, main
from twen.data.sources import SCHEMA_V2_REQUIRED_IMPLEMENTATIONS


def _write_recipe(root: Path) -> Path:
    recipe = {
        "schema_version": 2,
        "schema_status": "stable",
        "kind": "twen_base_data_source_recipe_v2",
        "recipe_id": "cli-v2-offline-fixture",
        "activation": {
            "runnable": True,
            "current_parser_compatible": True,
            "required_implementation": list(SCHEMA_V2_REQUIRED_IMPLEMENTATIONS),
        },
        "split": {
            "algorithm": "sha256_mod",
            "seed": "cli-v2-offline-fixture",
            "modulus": 2,
            "validation_remainder": 0,
        },
        "output_shard_tokens": 2,
        "profiles": {"smoke": {"train_tokens": 4}},
        "validation_tokens": 2,
        "mix_contract": {
            "basis_points_total": 10_000,
            "existing_sources_basis_points": 10_000,
            "new_sources_basis_points": 0,
        },
        "license_policy": {
            "canonical_permissive_allowlist": ["apache-2.0"],
        },
        "sources": [
            {
                "source_id": "offline_fixture",
                "origin_group": "existing",
                "mix_basis_points": 10_000,
                "category": "education",
                "repo_id": "example/offline",
                "revision": "a" * 40,
                "config": "default",
                "split": "train",
                "storage_format": "parquet",
                "file_patterns": ["data/fixture.parquet"],
                "locked_files": [
                    {
                        "path": "data/fixture.parquet",
                        "size": 17,
                        "sha256": "b" * 64,
                    }
                ],
                "license_declaration": "Apache-2.0",
                "license_scope": "fixture",
                "card_url": (
                    "https://huggingface.co/datasets/example/offline/blob/"
                    + "a" * 40
                    + "/README.md"
                ),
                "gated": False,
                "trust_remote_code": False,
                "text_field": "text",
                "stable_id_fields": ["id"],
                "split_group_fields": ["id"],
                "required_fields": ["id", "text"],
                "attribution_fields": ["id"],
                "train_token_quotas": {"smoke": 4},
                "validation_token_quota": 2,
                "min_characters": 1,
                "max_document_tokens": 64,
            }
        ],
    }
    path = root / "recipe.json"
    path.write_text(json.dumps(recipe), encoding="utf-8")
    return path


def _write_metadata_fixture(root: Path) -> Path:
    value = {
        "schema_version": 1,
        "kind": "twen_hf_dataset_metadata_fixture",
        "repositories": [
            {
                "repo_id": "example/offline",
                "revision": "a" * 40,
                "payload": {
                    "sha": "a" * 40,
                    "siblings": [
                        {
                            "rfilename": "data/fixture.parquet",
                            "lfs": {
                                "size": 17,
                                "sha256": "b" * 64,
                            },
                        }
                    ],
                },
            }
        ],
    }
    path = root / "metadata-fixture.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _run_json(argv: list[str]) -> tuple[int, dict[str, object]]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        code = main(argv)
    payload = json.loads(stdout.getvalue())
    assert isinstance(payload, dict)
    return code, payload


def test_validation_prepared_identity_does_not_project_a_train_source_map(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    prepared = SimpleNamespace(
        dataset_fingerprint="d" * 64,
        lineage={
            "role": "validation",
            "data_contract": {
                "source_map": {"fingerprint": "s" * 64},
                "source_mix": {"algorithm": "token-deficit-corrected-source-mix-bp-v2"},
            },
        },
    )

    payload = _prepared_identity_payload(manifest, prepared)

    assert payload["prepared_dataset_fingerprint"] == "d" * 64
    assert payload["prepared_source_map_sha256"] is None
    assert payload["extracted_source_map_sha256"] is None
    assert payload["source_mix_algorithm"] is None
    assert payload["source_mix_basis_points"] == {}


def test_data_inspect_alias_is_offline_and_reports_activation(tmp_path: Path) -> None:
    recipe = _write_recipe(tmp_path)
    code, payload = _run_json(
        ["data", "inspect", "--recipe", str(recipe)]
    )
    assert code == 0
    assert payload["operation"] == "inspect"
    assert payload["offline"] is True
    assert payload["formats"] == ["parquet"]
    assert payload["mix_basis_points_total"] == 10_000
    assert payload["files_written"] is False
    activation = payload["activation"]
    assert isinstance(activation, dict)
    assert activation["runnable"] is True
    assert activation["missing_implementation"] == []


def test_data_resolve_dry_run_neither_networks_nor_writes(tmp_path: Path) -> None:
    recipe = _write_recipe(tmp_path)
    output = tmp_path / "must-not-exist.json"
    code, payload = _run_json(
        [
            "data",
            "resolve",
            "--recipe",
            str(recipe),
            "--output",
            str(output),
            "--dry-run",
        ]
    )
    assert code == 0
    assert payload["operation"] == "resolve"
    assert payload["dry_run"] is True
    assert payload["network_accessed"] is False
    assert payload["files_written"] is False
    assert not output.exists()
    planned = payload["planned_lock"]
    assert isinstance(planned, dict)
    assert planned["materialization_audit"]["remote_identity_verification"] == (
        "embedded_lock_plan_only"
    )


def test_data_resolve_fixture_then_build_dry_run_is_fully_offline(
    tmp_path: Path,
) -> None:
    recipe = _write_recipe(tmp_path)
    fixture = _write_metadata_fixture(tmp_path)
    resolved = tmp_path / "resolved.json"
    code, payload = _run_json(
        [
            "data",
            "resolve",
            "--recipe",
            str(recipe),
            "--metadata-fixture",
            str(fixture),
            "--output",
            str(resolved),
        ]
    )
    assert code == 0
    assert payload["offline_fixture"] is True
    assert payload["network_policy"] == "offline-fixture"
    assert resolved.is_file()
    resolved_value = json.loads(resolved.read_text(encoding="utf-8"))
    assert resolved_value["source_mix"]["basis_points_total"] == 10_000
    assert resolved_value["format_audit"]["complete"] is True
    assert resolved_value["license_audit"]["complete"] is True
    assert resolved_value["materialization_audit"]["complete"] is True

    output = tmp_path / "must-not-build"
    code, build = _run_json(
        [
            "data",
            "build",
            "--recipe",
            str(recipe),
            "--resolved-lock",
            str(resolved),
            "--output",
            str(output),
            "--tokenizer",
            str(tmp_path / "unused-tokenizer"),
            "--tokenizer-manifest-sha256",
            "c" * 64,
            "--profile",
            "smoke",
            "--dry-run",
        ]
    )
    assert code == 0
    assert build["operation"] == "build"
    assert build["dry_run"] is True
    assert build["network_accessed"] is False
    assert build["files_written"] is False
    assert build["target_train_tokens"] == 4
    assert build["source_mix"] == {"offline_fixture": 10_000}
    assert not output.exists()


def test_unknown_activation_requirement_is_inspectable_but_not_runnable(
    tmp_path: Path,
) -> None:
    recipe_path = _write_recipe(tmp_path)
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe["activation"]["required_implementation"].append(
        "future implementation absent from this tree"
    )
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")

    code, payload = _run_json(
        ["data", "inspect", "--recipe", str(recipe_path)]
    )
    assert code == 0
    activation = payload["activation"]
    assert isinstance(activation, dict)
    assert activation["runnable"] is False
    assert activation["missing_implementation"] == [
        "future implementation absent from this tree"
    ]

    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        code = main(
            [
                "data",
                "resolve",
                "--recipe",
                str(recipe_path),
                "--output",
                str(tmp_path / "blocked.json"),
                "--dry-run",
            ]
        )
    assert code == 2
    assert "unverified implementation" in stderr.getvalue()
    assert not (tmp_path / "blocked.json").exists()
