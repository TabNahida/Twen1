"""Immutable, resumable benchmark materialization for 13-gram audits.

The registry is the source of truth for dataset commits and native Parquet
identities.  Hugging Face metadata is needed only while an entry has no
``source_files`` lock.  Afterwards the exact source objects can be recovered
offline from the local cache or downloaded again with Range-resume and SHA256
verification.

This module intentionally handles public benchmark text only.  It never loads
a model, constructs an optimizer, or touches CUDA.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import Request

from ..io.download import ArtifactSpec, DownloadManager, sha256_file
from ..io.locking import FileLock
from ..utils import atomic_write_json

BENCHMARK_REGISTRY_SCHEMA_VERSION = 1
BENCHMARK_REGISTRY_KIND = "twen_benchmark_13gram_registry"
BENCHMARK_CONVERSION_SCHEMA_VERSION = 1
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SAFE_BENCHMARK_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class BenchmarkMaterializationError(ValueError):
    """Raised when benchmark provenance, source data, or output is invalid."""


Projector = Callable[[Mapping[str, object]], dict[str, object]]


@dataclass(frozen=True, slots=True)
class BenchmarkRecipe:
    benchmark_id: str
    source_patterns: tuple[str, ...]
    expected_source_files: int
    parquet_columns: tuple[str, ...]
    text_fields: tuple[str, ...]
    projector: Projector
    native_id_field: str | None = None


def _text(value: object, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise BenchmarkMaterializationError(f"{field} must be a string")
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized and not allow_empty:
        raise BenchmarkMaterializationError(f"{field} cannot be empty")
    return normalized


def _string_list(value: object, field: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise BenchmarkMaterializationError(f"{field} must be a list")
    result = [_text(item, f"{field}[]", allow_empty=allow_empty) for item in value]
    if not result and not allow_empty:
        raise BenchmarkMaterializationError(f"{field} cannot be empty")
    return result


def _numbered_prompt(question: str, choices: Sequence[str], labels: Sequence[str]) -> str:
    if len(choices) != len(labels) or not choices:
        raise BenchmarkMaterializationError("multiple-choice labels/text have invalid lengths")
    return "\n".join(
        (
            question,
            *(f"{label}. {choice}" for label, choice in zip(labels, choices, strict=True)),
        )
    )


def _project_gsm8k(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "prompt": _text(row.get("question"), "gsm8k.question"),
        "reference": _text(row.get("answer"), "gsm8k.answer"),
    }


def _project_math(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "prompt": _text(row.get("problem"), "math.problem"),
        "reference": _text(row.get("solution"), "math.solution"),
    }


def _project_mmlu(row: Mapping[str, object]) -> dict[str, object]:
    question = _text(row.get("question"), "mmlu.question")
    choices = _string_list(row.get("choices"), "mmlu.choices")
    if len(choices) > 26:
        raise BenchmarkMaterializationError("mmlu has more than 26 choices")
    labels = [chr(ord("A") + index) for index in range(len(choices))]
    return {"prompt": _numbered_prompt(question, choices, labels)}


def _project_arc(row: Mapping[str, object]) -> dict[str, object]:
    question = _text(row.get("question"), "arc.question")
    choices = row.get("choices")
    if not isinstance(choices, Mapping):
        raise BenchmarkMaterializationError("arc.choices must be an object")
    texts = _string_list(choices.get("text"), "arc.choices.text")
    labels = _string_list(choices.get("label"), "arc.choices.label")
    return {"prompt": _numbered_prompt(question, texts, labels)}


def _project_humaneval(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "prompt": _text(row.get("prompt"), "humaneval.prompt"),
        "reference": [
            _text(row.get("canonical_solution"), "humaneval.canonical_solution"),
            _text(row.get("test"), "humaneval.test"),
        ],
    }


def _project_mbpp(row: Mapping[str, object]) -> dict[str, object]:
    references = [_text(row.get("code"), "mbpp.code")]
    references.extend(_string_list(row.get("test_list"), "mbpp.test_list"))
    prompt_value = row.get("text") if isinstance(row.get("text"), str) else row.get("prompt")
    setup_value = row.get("test_setup_code", "")
    setup = _text(setup_value, "mbpp.test_setup_code", allow_empty=True)
    if setup:
        references.append(setup)
    challenge_value = row.get("challenge_test_list", [])
    challenge_tests = _string_list(challenge_value, "mbpp.challenge_test_list", allow_empty=True)
    references.extend(item for item in challenge_tests if item)
    imports_value = row.get("test_imports", [])
    test_imports = _string_list(imports_value, "mbpp.test_imports", allow_empty=True)
    references.extend(item for item in test_imports if item)
    return {
        "prompt": _text(prompt_value, "mbpp.text/prompt"),
        # Keep one lexical stream so short code/import/assert fragments still
        # participate in a 13-gram gate when their complete task does.
        "reference": "\n".join(references),
    }


def _project_ceval(row: Mapping[str, object]) -> dict[str, object]:
    question = _text(row.get("question"), "ceval.question")
    labels = ("A", "B", "C", "D")
    choices = [_text(row.get(label), f"ceval.{label}") for label in labels]
    return {"prompt": _numbered_prompt(question, choices, labels)}


BASE_BENCHMARK_RECIPES: dict[str, BenchmarkRecipe] = {
    "gsm8k": BenchmarkRecipe(
        benchmark_id="gsm8k",
        source_patterns=("main/test-*.parquet",),
        expected_source_files=1,
        parquet_columns=("question", "answer"),
        text_fields=("prompt", "reference"),
        projector=_project_gsm8k,
    ),
    "math": BenchmarkRecipe(
        benchmark_id="math",
        source_patterns=("*/test-*.parquet",),
        expected_source_files=7,
        parquet_columns=("problem", "solution"),
        text_fields=("prompt", "reference"),
        projector=_project_math,
    ),
    "mmlu": BenchmarkRecipe(
        benchmark_id="mmlu",
        source_patterns=("all/test-*.parquet",),
        expected_source_files=1,
        parquet_columns=("question", "choices"),
        text_fields=("prompt",),
        projector=_project_mmlu,
    ),
    "arc": BenchmarkRecipe(
        benchmark_id="arc",
        source_patterns=("ARC-Challenge/test-*.parquet", "ARC-Easy/test-*.parquet"),
        expected_source_files=2,
        parquet_columns=("id", "question", "choices"),
        text_fields=("prompt",),
        projector=_project_arc,
        native_id_field="id",
    ),
    "humaneval": BenchmarkRecipe(
        benchmark_id="humaneval",
        source_patterns=("openai_humaneval/test-*.parquet",),
        expected_source_files=1,
        parquet_columns=("task_id", "prompt", "canonical_solution", "test"),
        text_fields=("prompt", "reference"),
        projector=_project_humaneval,
        native_id_field="task_id",
    ),
    "mbpp": BenchmarkRecipe(
        benchmark_id="mbpp",
        source_patterns=("full/test-*.parquet", "sanitized/test-*.parquet"),
        expected_source_files=2,
        # The published ``full`` and ``sanitized`` configs intentionally use
        # different schemas (text vs prompt; setup/challenge vs imports).  Read
        # each small test Parquet in full and project their shared semantics.
        parquet_columns=(),
        text_fields=("prompt", "reference"),
        projector=_project_mbpp,
        native_id_field="task_id",
    ),
    "ceval": BenchmarkRecipe(
        benchmark_id="ceval",
        source_patterns=("*/test-*.parquet",),
        expected_source_files=52,
        parquet_columns=("id", "question", "A", "B", "C", "D"),
        text_fields=("prompt",),
        projector=_project_ceval,
        native_id_field="id",
    ),
}


MetadataFetcher = Callable[[str, str], Mapping[str, object]]
ArtifactDownloader = Callable[[Mapping[str, object], Path], Path]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_relative(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise BenchmarkMaterializationError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() == "." or ".." in path.parts:
        raise BenchmarkMaterializationError(f"unsafe {field}: {value!r}")
    return path.as_posix()


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BenchmarkMaterializationError(f"{field} must be a positive integer")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value.lower()):
        raise BenchmarkMaterializationError(f"{field} must be a 64-digit SHA256")
    return value.lower()


def _revision(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA40.fullmatch(value.lower()):
        raise BenchmarkMaterializationError(
            f"{field} must be a 40-digit immutable commit, never main/master"
        )
    return value.lower()


def _normalize_license(value: object) -> str:
    values = value if isinstance(value, list) else [value]
    normalized = {
        str(item).strip().casefold().replace("_", "-")
        for item in values
        if isinstance(item, str) and item.strip()
    }
    aliases = {
        "mit": "MIT",
        "cc-by-4.0": "CC-BY-4.0",
        "cc-by-sa-4.0": "CC-BY-SA-4.0",
        "cc-by-nc-sa-4.0": "CC-BY-NC-SA-4.0",
    }
    if len(normalized) != 1:
        raise BenchmarkMaterializationError(
            f"dataset card must declare exactly one recognized license, got {value!r}"
        )
    item = next(iter(normalized))
    try:
        return aliases[item]
    except KeyError as error:
        raise BenchmarkMaterializationError(
            f"unrecognized dataset-card license {item!r}; manual review required"
        ) from error


def load_benchmark_registry(path: str | Path) -> dict[str, object]:
    registry_path = Path(path)
    try:
        value = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkMaterializationError(f"invalid benchmark registry: {registry_path}") from error
    if not isinstance(value, dict):
        raise BenchmarkMaterializationError("benchmark registry must be an object")
    if value.get("schema_version") != BENCHMARK_REGISTRY_SCHEMA_VERSION:
        raise BenchmarkMaterializationError("unsupported benchmark registry schema")
    if value.get("kind") != BENCHMARK_REGISTRY_KIND:
        raise BenchmarkMaterializationError("unexpected benchmark registry kind")
    benchmarks = value.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise BenchmarkMaterializationError("benchmark registry must contain entries")
    seen: set[str] = set()
    for index, entry in enumerate(benchmarks):
        if not isinstance(entry, dict):
            raise BenchmarkMaterializationError(f"benchmarks[{index}] must be an object")
        benchmark_id = entry.get("benchmark_id")
        if (
            not isinstance(benchmark_id, str)
            or not _SAFE_BENCHMARK_ID.fullmatch(benchmark_id)
            or benchmark_id in seen
        ):
            raise BenchmarkMaterializationError(f"invalid/duplicate benchmark_id {benchmark_id!r}")
        seen.add(benchmark_id)
    return value


def _entry_map(registry: Mapping[str, object]) -> dict[str, dict[str, object]]:
    entries = registry.get("benchmarks")
    if not isinstance(entries, list):  # guarded by load_benchmark_registry
        raise BenchmarkMaterializationError("benchmark registry has no entries")
    return {str(entry["benchmark_id"]): entry for entry in entries if isinstance(entry, dict)}


def _validate_registry_coverage(
    registry: Mapping[str, object], recipes: Mapping[str, BenchmarkRecipe]
) -> dict[str, dict[str, object]]:
    entries = _entry_map(registry)
    missing = sorted(set(recipes) - set(entries))
    unknown = sorted(set(entries) - set(recipes))
    if missing or unknown:
        raise BenchmarkMaterializationError(
            f"registry/recipe mismatch: missing={missing}, unknown={unknown}"
        )
    for benchmark_id, entry in entries.items():
        if entry.get("provider") != "huggingface":
            raise BenchmarkMaterializationError(f"{benchmark_id}.provider must be huggingface")
        dataset_id = entry.get("dataset_id")
        if not isinstance(dataset_id, str) or dataset_id.count("/") != 1:
            raise BenchmarkMaterializationError(f"{benchmark_id}.dataset_id is invalid")
        _revision(entry.get("revision"), f"{benchmark_id}.revision")
        if not isinstance(entry.get("declared_license"), str):
            raise BenchmarkMaterializationError(f"{benchmark_id}.declared_license is missing")
    return entries


def _metadata_fetcher(
    manager: DownloadManager,
    *,
    endpoint: str,
    token: str | None,
) -> MetadataFetcher:
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def fetch(dataset_id: str, revision: str) -> Mapping[str, object]:
        repo = "/".join(quote(part, safe="") for part in dataset_id.split("/"))
        url = f"{endpoint.rstrip('/')}/api/datasets/{repo}/revision/{revision}?blobs=true"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "User-Agent": "twen-benchmark-resolver/1",
                **headers,
            },
        )
        with manager._open(request) as response:
            payload = json.load(response)
        if not isinstance(payload, Mapping):
            raise BenchmarkMaterializationError(f"Hub metadata is not an object: {dataset_id}")
        return payload

    return fetch


def _source_identity(entry: Mapping[str, object], benchmark_id: str) -> dict[str, object]:
    path = _safe_relative(entry.get("rfilename"), f"{benchmark_id}.source.path")
    lfs = entry.get("lfs")
    if not isinstance(lfs, Mapping):
        raise BenchmarkMaterializationError(f"{benchmark_id} source is not LFS: {path}")
    return {
        "path": path,
        "size": _positive_int(lfs.get("size"), f"{path}.lfs.size"),
        "sha256": _sha256(
            str(lfs.get("sha256", lfs.get("oid", ""))).removeprefix("sha256:"),
            f"{path}.lfs.sha256",
        ),
        "format": "parquet",
    }


def resolve_benchmark_source_lock(
    entry: dict[str, object],
    recipe: BenchmarkRecipe,
    metadata: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Resolve and validate one pinned Hub metadata response."""

    benchmark_id = recipe.benchmark_id
    revision = _revision(entry.get("revision"), f"{benchmark_id}.revision")
    resolved = str(metadata.get("sha", "")).lower()
    if resolved != revision:
        raise BenchmarkMaterializationError(
            f"{benchmark_id} Hub revision mismatch: expected {revision}, got {resolved!r}"
        )
    if metadata.get("gated") is not False or metadata.get("private") is not False:
        raise BenchmarkMaterializationError(
            f"{benchmark_id} is gated/private; unattended materialization is forbidden"
        )
    card = metadata.get("cardData")
    if not isinstance(card, Mapping):
        raise BenchmarkMaterializationError(f"{benchmark_id} has no dataset-card metadata")
    declared_license = _normalize_license(entry.get("declared_license"))
    card_license = _normalize_license(card.get("license"))
    if declared_license != card_license:
        raise BenchmarkMaterializationError(
            f"{benchmark_id} license mismatch: registry={declared_license}, card={card_license}"
        )
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list):
        raise BenchmarkMaterializationError(f"{benchmark_id} metadata has no sibling list")
    source_files: list[dict[str, object]] = []
    readme: Mapping[str, object] | None = None
    for raw in siblings:
        if not isinstance(raw, Mapping):
            continue
        filename = raw.get("rfilename")
        if filename == "README.md":
            readme = raw
        if not isinstance(filename, str) or not any(
            fnmatch.fnmatchcase(filename, pattern) for pattern in recipe.source_patterns
        ):
            continue
        source_files.append(_source_identity(raw, benchmark_id))
    source_files.sort(key=lambda item: str(item["path"]))
    if len(source_files) != recipe.expected_source_files:
        raise BenchmarkMaterializationError(
            f"{benchmark_id} expected {recipe.expected_source_files} source files, "
            f"resolved {len(source_files)}"
        )
    for pattern in recipe.source_patterns:
        if not any(fnmatch.fnmatchcase(str(item["path"]), pattern) for item in source_files):
            raise BenchmarkMaterializationError(
                f"{benchmark_id} source pattern matched nothing: {pattern}"
            )
    if readme is None:
        raise BenchmarkMaterializationError(f"{benchmark_id} has no pinned README.md")
    readme_size = _positive_int(readme.get("size"), f"{benchmark_id}.README.md.size")
    readme_blob = str(readme.get("blobId", "")).lower()
    if not _GIT_SHA1.fullmatch(readme_blob):
        raise BenchmarkMaterializationError(f"{benchmark_id} README has no Git blob SHA1")
    dataset_id = str(entry["dataset_id"])
    repo = "/".join(quote(part, safe="") for part in dataset_id.split("/"))
    evidence = {
        "source": "pinned_huggingface_dataset_card",
        "path": "README.md",
        "size": readme_size,
        "git_blob_sha1": readme_blob,
        "card_license": card_license,
        "url": f"https://huggingface.co/datasets/{repo}/blob/{revision}/README.md",
    }
    existing_evidence = entry.get("license_evidence")
    if existing_evidence not in (None, {}) and existing_evidence != evidence:
        raise BenchmarkMaterializationError(
            f"{benchmark_id} pinned license evidence differs from registry"
        )
    return source_files, evidence


def _validated_locked_sources(
    entry: Mapping[str, object], recipe: BenchmarkRecipe
) -> list[dict[str, object]]:
    raw_files = entry.get("source_files")
    if not isinstance(raw_files, list) or len(raw_files) != recipe.expected_source_files:
        raise BenchmarkMaterializationError(
            f"{recipe.benchmark_id}.source_files is absent/incomplete"
        )
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_files):
        if not isinstance(raw, Mapping):
            raise BenchmarkMaterializationError(
                f"{recipe.benchmark_id}.source_files[{index}] must be an object"
            )
        path = _safe_relative(raw.get("path"), f"{recipe.benchmark_id}.source_files[{index}].path")
        if path in seen or not any(
            fnmatch.fnmatchcase(path, pattern) for pattern in recipe.source_patterns
        ):
            raise BenchmarkMaterializationError(
                f"{recipe.benchmark_id} has duplicate/unselected source path: {path}"
            )
        seen.add(path)
        if raw.get("format") != "parquet":
            raise BenchmarkMaterializationError(f"{path}.format must be parquet")
        result.append(
            {
                "path": path,
                "size": _positive_int(raw.get("size"), f"{path}.size"),
                "sha256": _sha256(raw.get("sha256"), f"{path}.sha256"),
                "format": "parquet",
            }
        )
    result.sort(key=lambda item: str(item["path"]))
    if result != raw_files:
        raise BenchmarkMaterializationError(
            f"{recipe.benchmark_id}.source_files must be canonical path-sorted identities"
        )
    return result


def _dataset_url(endpoint: str, dataset_id: str, revision: str, filename: str) -> str:
    repo = "/".join(quote(part, safe="") for part in dataset_id.split("/"))
    return (
        f"{endpoint.rstrip('/')}/datasets/{repo}/resolve/{quote(revision, safe='')}/"
        f"{quote(filename, safe='/')}"
    )


def _artifact_downloader(
    manager: DownloadManager,
    *,
    endpoint: str,
    token: str | None,
) -> ArtifactDownloader:
    headers = {"Authorization": f"Bearer {token}"} if token else None

    def download(source: Mapping[str, object], destination: Path) -> Path:
        spec = ArtifactSpec.http(
            url=str(source["url"]),
            source_id=str(source["source_id"]),
            revision=str(source["revision"]),
            filename=str(source["path"]),
            expected_size=int(source["size"]),
            sha256=str(source["sha256"]),
        )
        return manager.download(spec, destination, headers=headers)

    return download


def _write_jsonl_transactionally(
    output: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    expected_existing: Mapping[str, object] | None = None,
) -> tuple[int, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.incomplete")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary_size = temporary.stat().st_size
    temporary_sha = sha256_file(temporary)
    if output.exists():
        if output.stat().st_size != temporary_size or sha256_file(output) != temporary_sha:
            if expected_existing is None:
                temporary.unlink(missing_ok=True)
                raise BenchmarkMaterializationError(
                    f"refusing to overwrite non-identical benchmark output: {output}"
                )
            old_size = _positive_int(expected_existing.get("size"), f"{output}.old.size")
            old_sha = _sha256(expected_existing.get("sha256"), f"{output}.old.sha256")
            if output.stat().st_size != old_size or sha256_file(output) != old_sha:
                temporary.unlink(missing_ok=True)
                raise BenchmarkMaterializationError(
                    f"existing output does not match its authenticated registry identity: {output}"
                )
            os.replace(temporary, output)
            directory_fd = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        else:
            temporary.unlink()
    else:
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return temporary_size, temporary_sha


def convert_benchmark_sources(
    benchmark_id: str,
    recipe: BenchmarkRecipe,
    sources: Sequence[tuple[Mapping[str, object], Path]],
    output: str | Path,
    *,
    expected_existing: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Project exact native Parquet rows to deterministic audit JSONL."""

    import pyarrow.parquet as parquet

    rows: list[dict[str, object]] = []
    for source, source_path in sources:
        relative = str(source["path"])
        pure = PurePosixPath(relative)
        config = pure.parts[0] if len(pure.parts) > 1 else "default"
        split = pure.name.split("-", 1)[0]
        if split != "test":
            raise BenchmarkMaterializationError(
                f"{benchmark_id} source is not final test split: {relative}"
            )
        parquet_file = parquet.ParquetFile(source_path)
        missing = sorted(set(recipe.parquet_columns) - set(parquet_file.schema_arrow.names))
        if missing:
            raise BenchmarkMaterializationError(
                f"{benchmark_id} source {relative} lacks columns: {missing}"
            )
        source_row = 0
        for batch in parquet_file.iter_batches(
            batch_size=1024,
            columns=list(recipe.parquet_columns) if recipe.parquet_columns else None,
            use_threads=False,
        ):
            for native in batch.to_pylist():
                record: dict[str, object] = {
                    "benchmark_id": benchmark_id,
                    "config": config,
                    "source_path": relative,
                    "source_row": source_row,
                    "split": split,
                    **recipe.projector(native),
                }
                if recipe.native_id_field is not None:
                    native_id = native.get(recipe.native_id_field)
                    if not isinstance(native_id, (str, int)) or isinstance(native_id, bool):
                        raise BenchmarkMaterializationError(
                            f"{benchmark_id}.{recipe.native_id_field} has invalid type"
                        )
                    record["native_id"] = native_id
                rows.append(record)
                source_row += 1
    output_path = Path(output)
    size, digest = _write_jsonl_transactionally(
        output_path,
        rows,
        expected_existing=expected_existing,
    )
    return {
        "path": output_path.name,
        "size": size,
        "sha256": digest,
        "format": "jsonl",
        "text_fields": list(recipe.text_fields),
        "records": len(rows),
    }


def _validate_ready_output(entry: Mapping[str, object], root: Path) -> None:
    benchmark_id = str(entry["benchmark_id"])
    files = entry.get("files")
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], Mapping):
        raise BenchmarkMaterializationError(f"ready {benchmark_id} must have one output file")
    output = files[0]
    relative = _safe_relative(output.get("path"), f"{benchmark_id}.files[0].path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise BenchmarkMaterializationError(f"{benchmark_id} output escapes root") from error
    expected_size = _positive_int(output.get("size"), f"{relative}.size")
    expected_sha = _sha256(output.get("sha256"), f"{relative}.sha256")
    if output.get("format") != "jsonl":
        raise BenchmarkMaterializationError(f"{relative}.format must be jsonl")
    if not path.is_file() or path.stat().st_size != expected_size:
        raise BenchmarkMaterializationError(f"missing/size-mismatched output: {path}")
    if sha256_file(path) != expected_sha:
        raise BenchmarkMaterializationError(f"output SHA256 mismatch: {path}")


def materialize_base_benchmarks(
    registry_path: str | Path,
    benchmark_root: str | Path,
    *,
    manager: DownloadManager | None = None,
    token: str | None = None,
    endpoint: str = "https://huggingface.co",
    refresh_source_lock: bool = False,
    rebuild_outputs: bool = False,
    metadata_fetcher: MetadataFetcher | None = None,
    artifact_downloader: ArtifactDownloader | None = None,
    recipes: Mapping[str, BenchmarkRecipe] = BASE_BENCHMARK_RECIPES,
    lock_timeout_seconds: float = 300.0,
) -> dict[str, object]:
    """Lock, download, convert, and authenticate the Base benchmark registry."""

    registry_file = Path(registry_path)
    root = Path(benchmark_root)
    root.mkdir(parents=True, exist_ok=True)
    active_manager = manager or DownloadManager(network_policy="fallback")
    fetch = metadata_fetcher or _metadata_fetcher(
        active_manager,
        endpoint=endpoint,
        token=token,
    )
    download = artifact_downloader or _artifact_downloader(
        active_manager,
        endpoint=endpoint,
        token=token,
    )
    lock_path = registry_file.with_name(f".{registry_file.name}.materialize.lock")
    with FileLock(lock_path, timeout_seconds=lock_timeout_seconds):
        registry = load_benchmark_registry(registry_file)
        entries = _validate_registry_coverage(registry, recipes)
        for benchmark_id in sorted(recipes):
            recipe = recipes[benchmark_id]
            entry = entries[benchmark_id]
            if (
                entry.get("status") == "ready"
                and not refresh_source_lock
                and not rebuild_outputs
            ):
                _validated_locked_sources(entry, recipe)
                _validate_ready_output(entry, root)
                continue
            if refresh_source_lock or not entry.get("source_files"):
                metadata = fetch(str(entry["dataset_id"]), str(entry["revision"]))
                source_files, evidence = resolve_benchmark_source_lock(entry, recipe, metadata)
                if entry.get("source_files") and entry["source_files"] != source_files:
                    raise BenchmarkMaterializationError(
                        f"{benchmark_id} native source lock changed at an immutable revision"
                    )
                entry["source_files"] = source_files
                entry["license_evidence"] = evidence
                if entry.get("status") != "ready":
                    entry["status"] = "source_locked_pending_local_jsonl"
                    entry["license_review"] = "metadata_verified_at_pinned_revision"
                    atomic_write_json(registry_file, registry)
            source_files = _validated_locked_sources(entry, recipe)
            if entry.get("status") == "ready" and not rebuild_outputs:
                _validate_ready_output(entry, root)
                continue
            local_sources: list[tuple[Mapping[str, object], Path]] = []
            for source in source_files:
                cache_path = root / ".source-cache" / benchmark_id / str(source["path"])
                download_source = {
                    **source,
                    "source_id": f"huggingface-dataset:{entry['dataset_id']}",
                    "revision": entry["revision"],
                    "url": _dataset_url(
                        endpoint,
                        str(entry["dataset_id"]),
                        str(entry["revision"]),
                        str(source["path"]),
                    ),
                }
                materialized = download(download_source, cache_path)
                if materialized != cache_path:
                    materialized = Path(materialized)
                if (
                    not materialized.is_file()
                    or materialized.stat().st_size != int(source["size"])
                    or sha256_file(materialized) != str(source["sha256"])
                ):
                    raise BenchmarkMaterializationError(
                        f"downloaded source identity mismatch: {materialized}"
                    )
                local_sources.append((source, materialized))
            output = root / f"{benchmark_id}.jsonl"
            output_identity = convert_benchmark_sources(
                benchmark_id,
                recipe,
                local_sources,
                output,
                expected_existing=(
                    entry["files"][0]
                    if rebuild_outputs
                    and isinstance(entry.get("files"), list)
                    and entry["files"]
                    and isinstance(entry["files"][0], Mapping)
                    else None
                ),
            )
            entry["files"] = [output_identity]
            entry["status"] = "ready"
            entry["license_review"] = "metadata_verified_at_pinned_revision"
            entry["conversion"] = {
                "schema_version": BENCHMARK_CONVERSION_SCHEMA_VERSION,
                "kind": "twen_final_test_benchmark_projection",
                "selection": "final_test_split_only",
                "source_inventory_sha256": _canonical_sha256(source_files),
                "converter_source_sha256": sha256_file(Path(__file__)),
                "record_count": output_identity["records"],
            }
            atomic_write_json(registry_file, registry)
    return {
        "registry": str(registry_file.resolve()),
        "benchmark_root": str(root.resolve()),
        "ready": True,
        "benchmarks": {
            benchmark_id: entries[benchmark_id]["files"][0]
            for benchmark_id in sorted(entries)
        },
        "network_policy": active_manager.effective_network_policy,
    }


__all__ = [
    "BASE_BENCHMARK_RECIPES",
    "BENCHMARK_CONVERSION_SCHEMA_VERSION",
    "BenchmarkMaterializationError",
    "BenchmarkRecipe",
    "convert_benchmark_sources",
    "load_benchmark_registry",
    "materialize_base_benchmarks",
    "resolve_benchmark_source_lock",
]
