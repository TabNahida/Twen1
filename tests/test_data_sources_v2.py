from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from twen.data.prepared import _authenticate_extracted_prepare_inputs
from twen.data.sources import (
    DataSourceError,
    ResolvedParquetFile,
    _candidate_text,
    build_base_jsonl_corpus,
    download_locked_source_file,
    get_dotted_field,
    iter_gzip_jsonl_rows,
    iter_local_jsonl_rows,
    load_base_data_recipe,
    load_resolved_source_lock,
    materialize_jsonl_gzip_artifact,
    normalize_license,
    resolve_base_data_sources,
    stable_split_for_row,
    validate_extracted_base_corpus,
)
from twen.io.download import verify_artifact

ROOT = Path(__file__).resolve().parents[1]
V4_RECIPE = ROOT / "locks/base-data-sources-v4.json"


def _v1_recipe() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "twen_base_data_source_recipe",
        "recipe_id": "v1-compatibility-fixture",
        "split": {
            "algorithm": "sha256_mod",
            "seed": "fixture",
            "modulus": 2,
            "validation_remainder": 0,
            "code_group_field": "id",
        },
        "output_shard_tokens": 10,
        "profiles": {"dense": {"train_tokens": 10}},
        "validation_tokens": 2,
        "sources": [
            {
                "source_id": "v1_fixture",
                "category": "text",
                "repo_id": "example/v1",
                "revision": "a" * 40,
                "config": "default",
                "split": "train",
                "file_patterns": ["data/*.parquet"],
                "license": "fixture",
                "license_scope": "fixture",
                "card_url": "https://huggingface.co/datasets/example/v1",
                "gated": False,
                "text_field": "text",
                "stable_id_fields": ["id"],
                "required_fields": ["id", "text"],
                "train_token_quotas": {"dense": 10},
                "validation_token_quota": 2,
                "min_characters": 1,
                "max_document_tokens": 64,
            }
        ],
    }


def _filter_stats() -> dict[str, int]:
    return {
        "rejected_missing_field": 0,
        "rejected_short": 0,
        "rejected_license": 0,
        "rejected_row_filter": 0,
        "rejected_secret": 0,
        "rejected_pii": 0,
    }


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [1]


def test_v4_active_recipe_parses_locked_parquet_and_gzip_without_breaking_v1() -> None:
    recipe = load_base_data_recipe(V4_RECIPE)
    assert recipe.schema_version == 2
    assert recipe.schema_status == "stable"
    assert recipe.declared_runnable is True
    assert recipe.runnable is True
    assert len(recipe.sources) == 12
    assert {source.storage_format for source in recipe.sources} == {
        "parquet",
        "jsonl_gzip",
    }
    assert all(source.locked_files for source in recipe.sources)
    assert sum(int(source.mix_basis_points or 0) for source in recipe.sources) == 10_000

    libretexts = next(
        source
        for source in recipe.sources
        if source.source_id == "education_libretexts_permissive"
    )
    assert libretexts.text_field == "text"
    assert libretexts.license_field == "metadata.license"
    assert libretexts.split_group_fields == ("metadata.book_url",)
    assert libretexts.locked_files[0].path.endswith(".json.gz")

    common = next(
        source
        for source in recipe.sources
        if source.source_id == "multilingual_common_corpus_permissive"
    )
    assert common.storage_format == "parquet"
    assert {item.field for item in common.row_filters} == {
        "language",
        "language_type",
        "open_type",
        "collection",
    }

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "v1.json"
        path.write_text(json.dumps(_v1_recipe()), encoding="utf-8")
        v1 = load_base_data_recipe(path)
    assert v1.schema_version == 1
    assert v1.sources[0].storage_format == "parquet"
    assert v1.sources[0].split_group_fields == ("id",)
    assert v1.sources[0].locked_files == ()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["sources"][0]["locked_files"][0].update(
                {"sha256": "not-a-sha"}
            ),
            "sha256 is not a SHA256",
        ),
        (
            lambda value: value["sources"][0]["locked_files"][0].update(
                {"path": "../escape.parquet"}
            ),
            "unsafe parquet path/pattern",
        ),
        (
            lambda value: value["sources"][0].update(
                {"file_patterns": ["fineweb-edu-dedup/*.parquet"]}
            ),
            "must exactly enumerate locked_files",
        ),
    ],
)
def test_v2_embedded_file_identity_is_fail_closed(mutation, message: str) -> None:
    value = json.loads(V4_RECIPE.read_text(encoding="utf-8"))
    mutation(value)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "recipe.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(DataSourceError, match=message):
            load_base_data_recipe(path)


def test_v2_resolver_reaudits_embedded_lfs_identities_without_network() -> None:
    raw_recipe = json.loads(V4_RECIPE.read_text(encoding="utf-8"))
    metadata: dict[tuple[str, str], dict[str, object]] = {}
    for source in raw_recipe["sources"]:
        key = (source["repo_id"], source["revision"])
        payload = metadata.setdefault(
            key,
            {
                "sha": source["revision"],
                "siblings": [],
            },
        )
        siblings = payload["siblings"]
        assert isinstance(siblings, list)
        for locked in source["locked_files"]:
            entry = {
                "rfilename": locked["path"],
                "lfs": {
                    "size": locked["size"],
                    "oid": f"sha256:{locked['sha256']}",
                },
            }
            if entry not in siblings:
                siblings.append(entry)

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "resolved.json"
        resolve_base_data_sources(
            V4_RECIPE,
            output,
            metadata_fetcher=lambda repo_id, revision: metadata[(repo_id, revision)],
        )
        recipe = load_base_data_recipe(V4_RECIPE)
        resolved = load_resolved_source_lock(output, recipe)
        assert len(resolved.sources) == len(recipe.sources)
        formats = {
            artifact.storage_format
            for source in resolved.sources
            for artifact in source.files
        }
        assert formats == {"parquet", "jsonl_gzip"}

        first_source = raw_recipe["sources"][0]
        key = (first_source["repo_id"], first_source["revision"])
        first_lfs = metadata[key]["siblings"][0]["lfs"]
        first_lfs["size"] += 1
        with pytest.raises(DataSourceError, match="embedded locked_files"):
            resolve_base_data_sources(
                V4_RECIPE,
                Path(directory) / "mismatch.json",
                metadata_fetcher=lambda repo_id, revision: metadata[(repo_id, revision)],
            )


def test_dotted_fields_license_normalization_filters_and_group_split() -> None:
    recipe = load_base_data_recipe(V4_RECIPE)
    assert get_dotted_field(
        {"metadata": {"license": "CC BY 4.0"}},
        "metadata.license",
    ) == "CC BY 4.0"
    assert normalize_license("CC BY 4.0") == "cc-by-4.0"
    assert normalize_license("Creative Commons Attribution 4.0") == "cc-by-4.0"
    assert (
        normalize_license("https://creativecommons.org/licenses/by/4.0/")
        == "cc-by-4.0"
    )
    assert (
        normalize_license(
            "Creative Commons - Attribution - "
            "https://creativecommons.org/licenses/by/4.0/"
        )
        == "cc-by-4.0"
    )
    assert (
        normalize_license(
            "Creative Commons - Attribution-ShareAlike - "
            "https://creativecommons.org/licenses/by-sa/4.0/"
        )
        is None
    )
    assert (
        normalize_license(
            "Creative Commons - Attribution - "
            "https://creativecommons.org/licenses/by-sa/4.0/"
        )
        is None
    )
    assert normalize_license("MIT License") == "mit"
    assert normalize_license("MIT OR GPL-3.0") is None

    common = next(
        source
        for source in recipe.sources
        if source.source_id == "multilingual_common_corpus_permissive"
    )
    accepted = {
        "identifier": "doc-1",
        "collection": "Government release",
        "open_type": "Open Government",
        "curator": "fixture",
        "license": "MIT License",
        "language": "English",
        "language_type": "Written",
        "text": "A" * common.min_characters,
    }
    assert _candidate_text(common, accepted, stats=_filter_stats()) == accepted["text"]

    rejected_collection = dict(accepted, collection="Wikipedia")
    stats = _filter_stats()
    assert _candidate_text(common, rejected_collection, stats=stats) is None
    assert stats["rejected_row_filter"] == 1

    rejected_license = dict(accepted, license="GPL-3.0")
    stats = _filter_stats()
    assert _candidate_text(common, rejected_license, stats=stats) is None
    assert stats["rejected_license"] == 1

    libretexts = next(
        source
        for source in recipe.sources
        if source.source_id == "education_libretexts_permissive"
    )
    row_a = {"id": "a", "metadata": {"book_url": "https://book/one"}}
    row_b = {"id": "b", "metadata": {"book_url": "https://book/one"}}
    role_a, stable_a = stable_split_for_row(libretexts, row_a, recipe.split)
    role_b, stable_b = stable_split_for_row(libretexts, row_b, recipe.split)
    assert role_a == role_b
    assert stable_a != stable_b


def test_gzip_jsonl_reader_streams_nested_rows_and_resumes_by_row() -> None:
    rows = [
        {"id": "一", "metadata": {"license": "CC BY 4.0"}},
        {"id": "two", "metadata": {"license": "MIT"}},
    ]
    raw = b"".join(
        json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n" for row in rows
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fixture.json.gz"
        path.write_bytes(gzip.compress(raw, mtime=0))
        assert list(iter_gzip_jsonl_rows(path, start_row=1)) == [(1, rows[1])]

        path.write_bytes(gzip.compress(raw + b"\n", mtime=0))
        with pytest.raises(DataSourceError, match="blank JSONL row"):
            list(iter_gzip_jsonl_rows(path))


class _FixtureDownloadManager:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls = []

    def download(self, spec, destination, *, headers=None):
        self.calls.append((spec, Path(destination), headers))
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            output.write_bytes(self.payload)
        verify_artifact(output, spec)
        return output


class _BrokenGzipReader:
    def __init__(self, first_line: bytes) -> None:
        self.first_line = first_line

    def __enter__(self):
        def rows():
            yield self.first_line
            raise OSError("simulated interrupted decompression")

        return rows()

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_gzip_materialization_is_verified_transactional_and_recoverable() -> None:
    rows = [{"id": "one"}, {"id": "two"}]
    raw_lines = [
        json.dumps(row, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    ]
    compressed = gzip.compress(b"".join(raw_lines), mtime=0)
    artifact = ResolvedParquetFile(
        path="data/fixture.json.gz",
        size=len(compressed),
        sha256=hashlib.sha256(compressed).hexdigest(),
        url=(
            "https://huggingface.co/datasets/example/v2/resolve/"
            + "a" * 40
            + "/data/fixture.json.gz"
        ),
        storage_format="jsonl_gzip",
    )
    manager = _FixtureDownloadManager(compressed)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with patch(
            "twen.data.sources.gzip.open",
            return_value=_BrokenGzipReader(raw_lines[0]),
        ), pytest.raises(OSError, match="interrupted decompression"):
            materialize_jsonl_gzip_artifact(
                artifact,
                source_id="fixture_v2",
                repo_id="example/v2",
                revision="a" * 40,
                cache_root=root,
                manager=manager,
            )
        incomplete = list((root / "derived" / "fixture_v2").glob("*.incomplete"))
        assert len(incomplete) == 1
        assert (incomplete[0] / "data.jsonl").read_bytes() == raw_lines[0]

        derived = materialize_jsonl_gzip_artifact(
            artifact,
            source_id="fixture_v2",
            repo_id="example/v2",
            revision="a" * 40,
            cache_root=root,
            manager=manager,
        )
        assert derived.path.read_bytes() == b"".join(raw_lines)
        assert derived.row_count == 2
        assert derived.next_row_index == 2
        assert derived.row_index_stride == 4096
        assert derived.index_path.read_bytes() == b"\x00" * 8
        assert derived.size == len(b"".join(raw_lines))
        assert derived.sha256 == hashlib.sha256(b"".join(raw_lines)).hexdigest()
        assert list(iter_local_jsonl_rows(derived.path, start_row=1)) == [(1, rows[1])]
        assert list(iter_local_jsonl_rows(derived.path, start_row=2)) == []
        assert not list((root / "derived" / "fixture_v2").glob("*.incomplete"))

        original_mtime = derived.path.stat().st_mtime_ns
        reused = materialize_jsonl_gzip_artifact(
            artifact,
            source_id="fixture_v2",
            repo_id="example/v2",
            revision="a" * 40,
            cache_root=root,
            manager=manager,
        )
        assert reused == derived
        assert reused.path.stat().st_mtime_ns == original_mtime
        assert len(manager.calls) == 3
        assert manager.calls[-1][0].provider == "huggingface"
        assert manager.calls[-1][0].expected_size == len(compressed)
        assert manager.calls[-1][0].sha256 == hashlib.sha256(compressed).hexdigest()


def test_full_object_downloader_accepts_locked_parquet_identity_too() -> None:
    payload = b"PAR1fixturePAR1"
    artifact = ResolvedParquetFile(
        path="data/fixture.parquet",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        url=(
            "https://huggingface.co/datasets/example/v2/resolve/"
            + "a" * 40
            + "/data/fixture.parquet"
        ),
    )
    manager = _FixtureDownloadManager(payload)
    with tempfile.TemporaryDirectory() as directory:
        output = download_locked_source_file(
            artifact,
            source_id="fixture_v2",
            repo_id="example/v2",
            revision="a" * 40,
            destination_root=directory,
            manager=manager,
        )
        assert output.read_bytes() == payload
        assert manager.calls[0][0].filename == "data/fixture.parquet"


def test_sparse_derived_cursor_seeks_near_requested_row() -> None:
    rows = [{"id": index} for index in range(4_100)]
    raw = b"".join(
        json.dumps(row, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )
    compressed = gzip.compress(raw, mtime=0)
    artifact = ResolvedParquetFile(
        path="data/indexed.json.gz",
        size=len(compressed),
        sha256=hashlib.sha256(compressed).hexdigest(),
        url=(
            "https://huggingface.co/datasets/example/v2/resolve/"
            + "a" * 40
            + "/data/indexed.json.gz"
        ),
        storage_format="jsonl_gzip",
    )
    with tempfile.TemporaryDirectory() as directory:
        derived = materialize_jsonl_gzip_artifact(
            artifact,
            source_id="indexed_fixture",
            repo_id="example/v2",
            revision="a" * 40,
            cache_root=directory,
            manager=_FixtureDownloadManager(compressed),
        )
        assert derived.index_path.stat().st_size == 16
        assert list(iter_local_jsonl_rows(derived.path, start_row=4_097)) == [
            (index, rows[index]) for index in range(4_097, 4_100)
        ]


def test_v2_nested_filter_and_attribution_flow_through_transactional_builder() -> None:
    locked_size = 123
    locked_sha = "b" * 64
    revision = "a" * 40
    source_value = {
        "source_id": "nested_fixture",
        "origin_group": "existing",
        "mix_basis_points": 10_000,
        "category": "education",
        "repo_id": "example/v2",
        "revision": revision,
        "config": "default",
        "split": "train",
        "storage_format": "jsonl_gzip",
        "file_patterns": ["data/fixture.json.gz"],
        "locked_files": [
            {
                "path": "data/fixture.json.gz",
                "size": locked_size,
                "sha256": locked_sha,
            }
        ],
        "license_declaration": "per-row",
        "license_scope": "fixture",
        "card_url": f"https://huggingface.co/datasets/example/v2/blob/{revision}/README.md",
        "gated": False,
        "trust_remote_code": False,
        "text_field": "text",
        "stable_id_fields": ["id"],
        "split_group_fields": ["metadata.book_url"],
        "required_fields": [
            "id",
            "text",
            "metadata.book_url",
            "metadata.kind",
            "metadata.license",
        ],
        "license_field": "metadata.license",
        "license_allowlist": ["cc-by-4.0"],
        "license_value_mode": "canonical_after_normalization",
        "row_filters": {"metadata.kind_in": ["book"]},
        "attribution_fields": [
            "id",
            "metadata.book_url",
            "metadata.kind",
            "metadata.license",
        ],
        "train_token_quotas": {"dense": 8},
        "validation_token_quota": 2,
        "min_characters": 1,
        "max_document_tokens": 64,
    }
    recipe_value = {
        "schema_version": 2,
        "schema_status": "stable",
        "kind": "twen_base_data_source_recipe_v2",
        "recipe_id": "nested-v2-fixture",
        "split": {
            "algorithm": "sha256_mod",
            "seed": "nested-fixture",
            "modulus": 2,
            "validation_remainder": 0,
            "grouping": "source split_group_fields",
        },
        "output_shard_tokens": 4,
        "profiles": {"dense": {"train_tokens": 8}},
        "validation_tokens": 2,
        "mix_contract": {
            "basis_points_total": 10_000,
            "existing_sources_basis_points": 10_000,
            "new_sources_basis_points": 0,
        },
        "license_policy": {
            "canonical_permissive_allowlist": ["cc-by-4.0"],
        },
        "sources": [source_value],
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        recipe_path = root / "recipe.json"
        recipe_path.write_text(json.dumps(recipe_value), encoding="utf-8")
        recipe = load_base_data_recipe(recipe_path)
        source = recipe.sources[0]
        resolved_path = root / "resolved.json"
        resolve_base_data_sources(
            recipe_path,
            resolved_path,
            metadata_fetcher=lambda _repo_id, _revision: {
                "sha": revision,
                "siblings": [
                    {
                        "rfilename": "data/fixture.json.gz",
                        "lfs": {
                            "size": locked_size,
                            "sha256": locked_sha,
                        },
                    }
                ],
            },
        )

        role_rows: dict[str, list[dict[str, object]]] = {
            "train": [],
            "validation": [],
        }
        candidate = 0
        while len(role_rows["train"]) < 4 or not role_rows["validation"]:
            row = {
                "id": f"doc-{candidate}",
                "text": f"educational fixture {candidate}",
                "metadata": {
                    "book_url": f"https://books/{candidate}",
                    "kind": "book",
                    "license": "CC BY 4.0",
                },
            }
            role = stable_split_for_row(source, row, recipe.split)[0]
            role_rows[role].append(row)
            candidate += 1
        rows = [
            {
                "id": "rejected",
                "text": "wrong type",
                "metadata": {
                    "book_url": "https://books/rejected",
                    "kind": "article",
                    "license": "CC BY 4.0",
                },
            },
            *role_rows["train"][:4],
            role_rows["validation"][0],
        ]

        def row_iterator(_artifact, start_row, columns):
            assert "metadata.license" in columns
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
        assert report["train_tokens"] == 8
        assert report["validation_tokens"] == 2

        manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
        assert manifest_value["source_map"]["algorithm"] == (
            "authenticated-extracted-output-map-v1"
        )
        assert manifest_value["source_mix"]["algorithm"] == (
            "token-deficit-corrected-source-mix-bp-v2"
        )
        assert manifest_value["source_mix"]["basis_points_total"] == 10_000
        assert manifest_value["format_audit"]["complete"] is True
        assert manifest_value["license_audit"]["complete"] is True
        assert manifest_value["materialization_audit"]["complete"] is True
        authenticated_sources, lineage = _authenticate_extracted_prepare_inputs(
            manifest,
            role="train",
            tokenizer_sha256="c" * 64,
            allow_pending_research_audits=True,
        )
        assert len(authenticated_sources) == len(manifest_value["train_files"])
        assert lineage["data_contract"]["source_map"] == manifest_value["source_map"]
        assert lineage["data_contract"]["source_mix"] == manifest_value["source_mix"]
        ledger = []
        for entry in manifest_value["attribution_files"]:
            path = manifest.parent / entry["path"]
            ledger.extend(json.loads(line) for line in path.read_text().splitlines())
        assert len(ledger) == 5
        assert {entry["normalized_license"] for entry in ledger} == {"cc-by-4.0"}
        assert all(entry["metadata.license"] == "CC BY 4.0" for entry in ledger)
        assert all(entry["source_split_group_values"] for entry in ledger)
        chunk_stats = [
            chunk["statistics"]
            for source_manifest in manifest_value["sources"]
            for chunk in source_manifest["chunks"]
        ]
        assert sum(int(stats["rejected_row_filter"]) for stats in chunk_stats) == 1
