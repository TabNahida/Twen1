from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from twen.data.audits import inspect_benchmark_registry
from twen.data.benchmarks import (
    BASE_BENCHMARK_RECIPES,
    BenchmarkMaterializationError,
    convert_benchmark_sources,
    materialize_base_benchmarks,
    resolve_benchmark_source_lock,
)

REVISION = "1" * 40


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_gsm_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "question": [
                    "one two three four five six seven eight nine ten eleven twelve thirteen"
                ],
                "answer": ["A deterministic worked answer."],
            }
        ),
        path,
    )


def _registry(path: Path, *, revision: str = REVISION, license_name: str = "MIT") -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "twen_benchmark_13gram_registry",
                "registry_id": "fixture",
                "benchmarks": [
                    {
                        "benchmark_id": "gsm8k",
                        "required": True,
                        "status": "pending_native_source_lock_and_local_jsonl",
                        "provider": "huggingface",
                        "dataset_id": "fixture/gsm8k",
                        "source_url": "https://huggingface.co/datasets/fixture/gsm8k",
                        "declared_license": license_name,
                        "license_review": "required_before_lock",
                        "revision": revision,
                        "source_files": [],
                        "files": [],
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _metadata(source: Path, *, card_license: str = "mit") -> dict[str, object]:
    return {
        "sha": REVISION,
        "gated": False,
        "private": False,
        "cardData": {"license": [card_license]},
        "siblings": [
            {
                "rfilename": "README.md",
                "size": 123,
                "blobId": "2" * 40,
            },
            {
                "rfilename": "main/test-00000-of-00001.parquet",
                "size": source.stat().st_size,
                "lfs": {"size": source.stat().st_size, "sha256": _sha(source)},
            },
        ],
    }


def test_materialization_locks_sources_is_recoverable_and_audit_ready(tmp_path: Path) -> None:
    source = tmp_path / "upstream.parquet"
    _write_gsm_parquet(source)
    registry = _registry(tmp_path / "registry.json")
    root = tmp_path / "benchmarks"
    calls = {"metadata": 0, "download": 0}

    def fetch(dataset_id: str, revision: str) -> dict[str, object]:
        calls["metadata"] += 1
        assert dataset_id == "fixture/gsm8k"
        assert revision == REVISION
        return _metadata(source)

    def download(identity: dict[str, object], destination: Path) -> Path:
        calls["download"] += 1
        assert identity["revision"] == REVISION
        assert identity["sha256"] == _sha(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return destination

    result = materialize_base_benchmarks(
        registry,
        root,
        metadata_fetcher=fetch,
        artifact_downloader=download,
        recipes={"gsm8k": BASE_BENCHMARK_RECIPES["gsm8k"]},
    )
    assert result["ready"] is True
    assert calls == {"metadata": 1, "download": 1}
    value = json.loads(registry.read_text(encoding="utf-8"))
    entry = value["benchmarks"][0]
    assert entry["status"] == "ready"
    assert entry["source_files"][0]["sha256"] == _sha(source)
    assert entry["license_evidence"]["git_blob_sha1"] == "2" * 40
    output = root / "gsm8k.jsonl"
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["prompt"].endswith("twelve thirteen")
    assert row["reference"] == "A deterministic worked answer."
    report = inspect_benchmark_registry(registry, benchmark_root=root)
    assert report["ready"] is True
    assert report["pending_benchmarks"] == []

    def no_metadata(dataset_id: str, revision: str) -> dict[str, object]:
        raise AssertionError((dataset_id, revision))

    def no_download(identity: dict[str, object], destination: Path) -> Path:
        raise AssertionError((identity, destination))

    second = materialize_base_benchmarks(
        registry,
        root,
        metadata_fetcher=no_metadata,
        artifact_downloader=no_download,
        recipes={"gsm8k": BASE_BENCHMARK_RECIPES["gsm8k"]},
    )
    assert second["ready"] is True


def test_pinned_card_license_mismatch_never_becomes_ready(tmp_path: Path) -> None:
    source = tmp_path / "upstream.parquet"
    _write_gsm_parquet(source)
    registry = _registry(tmp_path / "registry.json", license_name="MIT")
    with pytest.raises(BenchmarkMaterializationError, match="license mismatch"):
        materialize_base_benchmarks(
            registry,
            tmp_path / "benchmarks",
            metadata_fetcher=lambda _dataset, _revision: _metadata(
                source, card_license="cc-by-4.0"
            ),
            artifact_downloader=lambda _source, destination: destination,
            recipes={"gsm8k": BASE_BENCHMARK_RECIPES["gsm8k"]},
        )
    value = json.loads(registry.read_text(encoding="utf-8"))
    assert value["benchmarks"][0]["status"] != "ready"


def test_floating_revision_is_rejected_before_network(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.json", revision="main")

    def unexpected_fetch(dataset_id: str, revision: str) -> dict[str, object]:
        raise AssertionError((dataset_id, revision))

    with pytest.raises(BenchmarkMaterializationError, match="immutable commit"):
        materialize_base_benchmarks(
            registry,
            tmp_path / "benchmarks",
            metadata_fetcher=unexpected_fetch,
            artifact_downloader=lambda _source, destination: destination,
            recipes={"gsm8k": BASE_BENCHMARK_RECIPES["gsm8k"]},
        )


def test_ready_output_hash_drift_is_rejected_without_network(tmp_path: Path) -> None:
    source = tmp_path / "upstream.parquet"
    _write_gsm_parquet(source)
    registry = _registry(tmp_path / "registry.json")
    root = tmp_path / "benchmarks"

    def download(_identity: dict[str, object], destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return destination

    materialize_base_benchmarks(
        registry,
        root,
        metadata_fetcher=lambda _dataset, _revision: _metadata(source),
        artifact_downloader=download,
        recipes={"gsm8k": BASE_BENCHMARK_RECIPES["gsm8k"]},
    )
    (root / "gsm8k.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(BenchmarkMaterializationError, match="size-mismatched output"):
        materialize_base_benchmarks(
            registry,
            root,
            metadata_fetcher=lambda _dataset, _revision: (_ for _ in ()).throw(
                AssertionError("network must not be used")
            ),
            artifact_downloader=lambda _source, destination: destination,
            recipes={"gsm8k": BASE_BENCHMARK_RECIPES["gsm8k"]},
        )


def test_rebuild_reuses_locked_metadata_and_requires_byte_identical_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "upstream.parquet"
    _write_gsm_parquet(source)
    registry = _registry(tmp_path / "registry.json")
    root = tmp_path / "benchmarks"

    def download(_identity: dict[str, object], destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source, destination)
        return destination

    materialize_base_benchmarks(
        registry,
        root,
        metadata_fetcher=lambda _dataset, _revision: _metadata(source),
        artifact_downloader=download,
        recipes={"gsm8k": BASE_BENCHMARK_RECIPES["gsm8k"]},
    )
    before = (root / "gsm8k.jsonl").read_bytes()
    materialize_base_benchmarks(
        registry,
        root,
        metadata_fetcher=lambda _dataset, _revision: (_ for _ in ()).throw(
            AssertionError("locked rebuild must not query metadata")
        ),
        artifact_downloader=download,
        rebuild_outputs=True,
        recipes={"gsm8k": BASE_BENCHMARK_RECIPES["gsm8k"]},
    )
    assert (root / "gsm8k.jsonl").read_bytes() == before


def test_resolver_requires_exact_native_file_count(tmp_path: Path) -> None:
    source = tmp_path / "upstream.parquet"
    _write_gsm_parquet(source)
    entry = {
        "benchmark_id": "gsm8k",
        "dataset_id": "fixture/gsm8k",
        "declared_license": "MIT",
        "revision": REVISION,
    }
    metadata = _metadata(source)
    metadata["siblings"] = metadata["siblings"][:1]
    with pytest.raises(BenchmarkMaterializationError, match="expected 1 source files"):
        resolve_benchmark_source_lock(entry, BASE_BENCHMARK_RECIPES["gsm8k"], metadata)


def test_mbpp_full_and_sanitized_schema_variants_project_to_same_fields(tmp_path: Path) -> None:
    full = tmp_path / "full.parquet"
    sanitized = tmp_path / "sanitized.parquet"
    pq.write_table(
        pa.table(
            {
                "task_id": [1],
                "text": ["Full task prompt"],
                "code": ["def f(): return 1"],
                "test_list": [["assert f() == 1"]],
                "test_setup_code": ["import math"],
                "challenge_test_list": [["assert int(f()) == 1"]],
            }
        ),
        full,
    )
    pq.write_table(
        pa.table(
            {
                "source_file": ["source.ipynb"],
                "task_id": [2],
                "prompt": ["Sanitized task prompt"],
                "code": ["def g(): return 2"],
                "test_imports": [["from math import isclose"]],
                "test_list": [["assert g() == 2"]],
            }
        ),
        sanitized,
    )
    result = convert_benchmark_sources(
        "mbpp",
        BASE_BENCHMARK_RECIPES["mbpp"],
        [
            (
                {"path": "full/test-00000-of-00001.parquet"},
                full,
            ),
            (
                {"path": "sanitized/test-00000-of-00001.parquet"},
                sanitized,
            ),
        ],
        tmp_path / "mbpp.jsonl",
    )
    assert result["records"] == 2
    rows = [json.loads(line) for line in (tmp_path / "mbpp.jsonl").read_text().splitlines()]
    assert rows[0]["prompt"] == "Full task prompt"
    assert isinstance(rows[0]["reference"], str)
    assert "import math" in rows[0]["reference"]
    assert rows[1]["prompt"] == "Sanitized task prompt"
    assert "from math import isclose" in rows[1]["reference"]


def test_converter_migration_requires_authenticated_old_output(tmp_path: Path) -> None:
    source = tmp_path / "gsm.parquet"
    output = tmp_path / "gsm.jsonl"
    _write_gsm_parquet(source)
    identity = convert_benchmark_sources(
        "gsm8k",
        BASE_BENCHMARK_RECIPES["gsm8k"],
        [({"path": "main/test-00000-of-00001.parquet"}, source)],
        output,
    )
    pq.write_table(
        pa.table(
            {
                "question": ["a changed benchmark question with enough words for a stable test"],
                "answer": ["a changed answer"],
            }
        ),
        source,
    )
    with pytest.raises(BenchmarkMaterializationError, match="refusing to overwrite"):
        convert_benchmark_sources(
            "gsm8k",
            BASE_BENCHMARK_RECIPES["gsm8k"],
            [({"path": "main/test-00000-of-00001.parquet"}, source)],
            output,
        )
    migrated = convert_benchmark_sources(
        "gsm8k",
        BASE_BENCHMARK_RECIPES["gsm8k"],
        [({"path": "main/test-00000-of-00001.parquet"}, source)],
        output,
        expected_existing=identity,
    )
    assert migrated["sha256"] != identity["sha256"]
