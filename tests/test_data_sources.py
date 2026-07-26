from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import ClientConnectionError

from twen.data.shards import ShardTransaction
from twen.data.sources import (
    DataSourceError,
    HfRangeFileFactory,
    ResolvedParquetFile,
    SourceRecipe,
    _load_seen_hashes,
    build_base_jsonl_corpus,
    load_base_data_recipe,
    load_resolved_source_lock,
    resolve_base_data_sources,
    stable_split_for_row,
    validate_extracted_base_corpus,
)
from twen.io.proxy import ProxySettings


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [1, 2, 3]


def _write_recipe(path: Path) -> None:
    value = {
        "schema_version": 1,
        "kind": "twen_base_data_source_recipe",
        "recipe_id": "fixture-v1",
        "split": {
            "algorithm": "sha256_mod",
            "seed": "fixture-seed",
            "modulus": 2,
            "validation_remainder": 0,
            "code_group_field": "repo_name",
        },
        "output_shard_tokens": 4,
        "profiles": {"dense": {"train_tokens": 4}},
        "validation_tokens": 4,
        "sources": [
            {
                "source_id": "code_fixture",
                "category": "code",
                "repo_id": "example/code",
                "revision": "a" * 40,
                "config": "native_parquet",
                "split": "train",
                "file_patterns": ["data/train-*.parquet"],
                "license": "per-row SPDX field",
                "license_scope": "fixture",
                "card_url": "https://huggingface.co/datasets/example/code",
                "gated": False,
                "text_field": "code",
                "stable_id_fields": ["repo_name"],
                "required_fields": ["code", "repo_name", "path", "license"],
                "license_field": "license",
                "license_allowlist": ["mit"],
                "attribution_fields": ["repo_name", "path", "license"],
                "reject_detected_secrets": True,
                "train_token_quotas": {"dense": 4},
                "validation_token_quota": 4,
                "min_characters": 4,
                "max_document_tokens": 64,
            }
        ],
    }
    path.write_text(json.dumps(value), encoding="utf-8")


class DataSourceRecipeTest(unittest.TestCase):
    def test_repository_recipe_has_exact_profile_totals(self) -> None:
        recipe = load_base_data_recipe("locks/base-data-sources.json")
        self.assertEqual(recipe.profiles["dense"], 100_000_000)
        self.assertEqual(recipe.profiles["sparse"], 500_000_000)
        self.assertEqual(recipe.validation_tokens, 20_000_000)
        chinese = next(
            item for item in recipe.sources if item.source_id == "chinese_fineweb2_cmn_hani"
        )
        self.assertEqual(chinese.config, "cmn_Hani")
        code = next(item for item in recipe.sources if item.category == "code")
        self.assertEqual(
            set(code.license_allowlist),
            {"apache-2.0", "bsd-2-clause", "bsd-3-clause", "cc0-1.0", "isc", "mit"},
        )

    def test_resolver_locks_only_matching_lfs_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe_path = root / "recipe.json"
            _write_recipe(recipe_path)
            output = root / "resolved.json"

            def fetch(repo_id: str, revision: str):
                self.assertEqual(repo_id, "example/code")
                return {
                    "sha": revision,
                    "siblings": [
                        {
                            "rfilename": "data/train-00000.parquet",
                            "lfs": {"size": 123, "oid": "sha256:" + "b" * 64},
                        },
                        {"rfilename": "README.md", "size": 10},
                    ],
                }

            resolve_base_data_sources(
                recipe_path,
                output,
                metadata_fetcher=fetch,
            )
            recipe = load_base_data_recipe(recipe_path)
            resolved = load_resolved_source_lock(output, recipe)
            self.assertEqual(len(resolved.sources), 1)
            file = resolved.sources[0].files[0]
            self.assertEqual(file.path, "data/train-00000.parquet")
            self.assertEqual(file.size, 123)
            self.assertEqual(file.sha256, "b" * 64)
            self.assertIn("/datasets/example/code/resolve/", file.url)

    def test_code_split_groups_an_entire_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recipe_path = Path(directory) / "recipe.json"
            _write_recipe(recipe_path)
            recipe = load_base_data_recipe(recipe_path)
            source = recipe.sources[0]
            first = stable_split_for_row(
                source,
                {"repo_name": "owner/repo", "path": "a.py"},
                recipe.split,
            )[0]
            second = stable_split_for_row(
                source,
                {"repo_name": "owner/repo", "path": "b.py"},
                recipe.split,
            )[0]
            self.assertEqual(first, second)


class CorpusExtractionTest(unittest.TestCase):
    def test_dedup_scan_ignores_unrenamed_complete_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe_path = root / "recipe.json"
            _write_recipe(recipe_path)
            recipe = load_base_data_recipe(recipe_path)
            extracted = root / "extracted"
            source_root = extracted / recipe.sources[0].source_id
            with ShardTransaction(
                source_root,
                "chunk-000000",
                fingerprint="pipeline",
                source_fingerprint="source",
            ) as transaction:
                (transaction.work_directory / "train.jsonl").write_text(
                    '{"text":"must be replayed"}\n', encoding="utf-8"
                )
                final = transaction.commit()
            staging = final.with_name(final.name + ".incomplete")
            final.rename(staging)

            self.assertTrue((staging / "COMPLETE").is_file())
            self.assertEqual(_load_seen_hashes(extracted, recipe.sources), set())

    def test_build_filters_license_and_writes_attribution_for_every_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe_path = root / "recipe.json"
            _write_recipe(recipe_path)
            recipe = load_base_data_recipe(recipe_path)
            source: SourceRecipe = recipe.sources[0]
            resolved_path = root / "resolved.json"
            resolved_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "twen_resolved_base_data_sources",
                        "recipe_id": recipe.recipe_id,
                        "recipe_sha256": recipe.sha256,
                        "sources": [
                            {
                                "source_id": source.source_id,
                                "repo_id": source.repo_id,
                                "revision": source.revision,
                                "config": source.config,
                                "split": source.split,
                                "license": source.license,
                                "files": [
                                    {
                                        "path": "data/train-00000.parquet",
                                        "size": 8,
                                        "sha256": "b" * 64,
                                        "url": (
                                            "https://huggingface.co/datasets/example/code/resolve/"
                                            + "a" * 40
                                            + "/data/train-00000.parquet"
                                        ),
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            role_repos: dict[str, str] = {}
            index = 0
            while set(role_repos) != {"train", "validation"}:
                repo = f"owner/repo-{index}"
                role = stable_split_for_row(
                    source,
                    {"repo_name": repo},
                    recipe.split,
                )[0]
                role_repos.setdefault(role, repo)
                index += 1
            rows = [
                {
                    "code": "print('wrong license')",
                    "repo_name": "owner/gpl",
                    "path": "bad.py",
                    "license": "gpl-3.0",
                },
                {
                    "code": "print('train accepted')",
                    "repo_name": role_repos["train"],
                    "path": "train.py",
                    "license": "mit",
                },
                {
                    "code": "print('validation accepted')",
                    "repo_name": role_repos["validation"],
                    "path": "validation.py",
                    "license": "mit",
                },
            ]

            def row_iterator(_artifact, start_row, columns):
                self.assertTrue({"code", "repo_name", "path", "license"} <= set(columns))
                return iter(enumerate(rows[start_row:], start=start_row))

            manifest = build_base_jsonl_corpus(
                recipe_path,
                resolved_path,
                root / "corpus",
                tokenizer_path=root / "unused",
                tokenizer_manifest_sha256="c" * 64,
                profile="dense",
                progress="never",
                _tokenizer=_Tokenizer(),
                _row_iterator=row_iterator,
            )
            report = validate_extracted_base_corpus(manifest)
            self.assertEqual(report["train_tokens"], 4)
            self.assertEqual(report["validation_tokens"], 4)
            self.assertFalse(report["ready_for_training"])
            value = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(len(value["attribution_files"]), 2)
            ledger_rows = []
            for item in value["attribution_files"]:
                path = manifest.parent / item["path"]
                ledger_rows.extend(json.loads(line) for line in path.read_text().splitlines())
            self.assertEqual({item["license"] for item in ledger_rows}, {"mit"})
            self.assertEqual({item["split"] for item in ledger_rows}, {"train", "validation"})
            self.assertTrue(all(item["text_sha256"] for item in ledger_rows))
            self.assertTrue(all(item["source_file_url"] for item in ledger_rows))
            self.assertTrue(all(item["source_file_lfs_sha256"] for item in ledger_rows))
            self.assertTrue(all(item["split_bucket"] >= 0 for item in ledger_rows))
            self.assertTrue(all(item["filter_decisions"] for item in ledger_rows))
            self.assertTrue(all(item["source_stable_values"] for item in ledger_rows))

            train_list = manifest.parent / "train-files.txt"
            original_train_list = train_list.read_text(encoding="utf-8")
            train_list.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(DataSourceError, "file list"):
                validate_extracted_base_corpus(manifest)
            train_list.write_text(original_train_list, encoding="utf-8")
            self.assertTrue(validate_extracted_base_corpus(manifest)["ok"])

            (manifest.parent / "INVALIDATED.json").write_text(
                '{"reason":"fixture invalidation"}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(DataSourceError, "explicitly invalidated"):
                validate_extracted_base_corpus(manifest)


class RangeProxyTest(unittest.TestCase):
    def test_fsspec_proxy_is_explicit_request_option(self) -> None:
        artifact = ResolvedParquetFile(
            path="data/train.parquet",
            size=8,
            sha256="d" * 64,
            url=(
                "https://huggingface.co/datasets/example/data/resolve/"
                + "a" * 40
                + "/data/train.parquet"
            ),
        )
        filesystem = unittest.mock.Mock()
        # A redirected CDN may return a SHA-shaped ETag unrelated to the Hub
        # sibling's LFS oid; only the response size is authoritative here.
        filesystem.info.return_value = {"size": 8, "ETag": '"' + "e" * 64 + '"'}
        filesystem.open.return_value = io.BytesIO(b"xxxxPAR1")
        factory = HfRangeFileFactory(
            network_policy="proxy",
            proxy_settings=ProxySettings(
                http_proxy="http://proxy.test:8080",
                https_proxy="http://proxy.test:8080",
            ),
        )
        with (
            patch("twen.data.sources.check_proxy_connectivity"),
            patch(
                "fsspec.implementations.http.HTTPFileSystem",
                return_value=filesystem,
            ) as constructor,
            factory.open(artifact) as handle,
        ):
            self.assertEqual(handle.read(4), b"xxxx")
        options = constructor.call_args.kwargs
        self.assertEqual(options["proxy"], "http://proxy.test:8080")
        self.assertFalse(options["client_kwargs"]["trust_env"])
        self.assertEqual(options["headers"]["Accept-Encoding"], "identity")

    def test_fallback_switches_to_proxy_after_a_later_range_failure(self) -> None:
        artifact = ResolvedParquetFile(
            path="data/train.parquet",
            size=8,
            sha256="d" * 64,
            url=(
                "https://huggingface.co/datasets/example/data/resolve/"
                + "a" * 40
                + "/data/train.parquet"
            ),
        )

        class LateFailure(io.BytesIO):
            def read(self, size: int = -1) -> bytes:
                raise ClientConnectionError("direct CDN range failed")

        factory = HfRangeFileFactory(
            network_policy="fallback",
            proxy_settings=ProxySettings(
                http_proxy="http://proxy.test:8080",
                https_proxy="http://proxy.test:8080",
            ),
        )
        with patch.object(
            factory,
            "_open_once",
            side_effect=[LateFailure(b"xxxxPAR1"), io.BytesIO(b"xxxxPAR1")],
        ) as open_once, factory.open(artifact) as handle:
            self.assertEqual(handle.read(4), b"xxxx")

        self.assertTrue(factory.proxy_fallback_used)
        self.assertEqual(open_once.call_args_list[0].kwargs, {"proxy": False})
        self.assertEqual(open_once.call_args_list[1].kwargs, {"proxy": True})


if __name__ == "__main__":
    unittest.main()
