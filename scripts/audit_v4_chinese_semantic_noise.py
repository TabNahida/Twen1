#!/usr/bin/env python3
"""Authenticate and review Chinese semantic-conversion/stitching noise for v4.

This audit is deliberately separate from the generic corpus scanner.  The
generic scanner catches structural corruption, PII, exact/near duplicates,
and mixed simplified/traditional scripts, but it cannot reliably identify
same-script substitutions such as ``大年夜众``.  This command:

* authenticates every selected Chinese JSONL shard against each extracted
  corpus manifest;
* performs a complete deterministic scan for conservative, high-precision
  conversion and sentence-stitching indicators;
* emits deterministic risk and control samples for semantic review; and
* optionally authenticates a complete review-decision file.

It never rewrites training data and never authorizes training by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIMARY = (
    ROOT
    / "data/base-v4-250m-primary-r1-refill-filtered-pass-002/corpus-manifest.json"
)
DEFAULT_COOLDOWN = (
    ROOT
    / "artifacts/data/base-v4-250m-cooldown-phase-near-excluded-v2/"
    "corpus-manifest.json"
)
DEFAULT_SOURCE_ID = "chinese_fineweb2_cmn_hani"
SCHEMA_VERSION = 1
ATTESTATION_KIND = "twen_v4_chinese_semantic_noise_attestation"
MANIFEST_KIND = "twen_v4_chinese_semantic_noise_bundle"
COMPLETE_KIND = "twen_v4_chinese_semantic_noise_complete"
DECISIONS_KIND = "twen_v4_chinese_semantic_noise_manual_decisions"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_MALFORMED_PUNCTUATION = re.compile(
    r"(?:\uff0c\u3002|\uff0c\u3002\uff0c\u3002|\u3002\u3002|"
    r"\uff1b\u3002|\uff1a\u3002|\u3001\u3002|\uff0c\uff1b|"
    r"\uff0c\uff1a|\u3002\?|\u3002\uff1f|\uff1f\u3002|\uff01\u3002)"
)
_LITERAL_ESCAPED_NEWLINE = re.compile(r"(?<!\\)\\n")

# Keep this list conservative.  Every item is either an unmistakable
# same-script conversion artifact in ordinary prose or a highly specific
# phrase observed in the pinned FineWeb2 Chinese material.  Ambiguous words
# such as “软件体系” and legitimate classical phrases such as “乾坤” are
# intentionally not hard failures.
_CONVERSION_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("da_nian_ye_substitution", re.compile(r"大年夜(?:众|家|多|型|小|量|幅|陆|学|事|会|车|门|半|概|部|都|城|约|胆|雨|风|雪|战|火|气|赛|牌|片|作|师|夫|妈|哥|姐|叔|爷|王|臣|街|道|桥|厦|厅|楼|棚|田|海|河|江|山|湾|洋|洲|国|区|县|镇|村|院|校|班|队|组|类|号|字|写|声|笑|哭|吃|喝|喊|叫|意|抵|致|于|概|要|抵|幅|片|块|批|笔|宗|起|早|晚|年)")),
    ("engine_substitution", re.compile(r"发念头")),
    ("brand_command_substitution", re.compile(r"品商标令力")),
    ("view_character_expansion", re.compile(r"(?:可不雅|不雅众|不雅看|不雅点|外不雅|宏不雅|客不雅|主不雅)")),
    ("cannot_substitution", re.compile(r"(?:弗成忽视|弗成能|弗成以|弗成靠|弗成避免)")),
    ("appeared_substitution", re.compile(r"(?:涌如今|呈如今|出如今)")),
)
_CONVERSION_MARKER_NAMES = frozenset(name for name, _pattern in _CONVERSION_MARKERS)


class AuditError(ValueError):
    """Raised when an input or review bundle is unsafe or unauthenticated."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    descriptor = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            temporary.unlink()
        raise


def _write_json(path: Path, value: Any) -> None:
    _write_bytes_atomic(
        path,
        (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise AuditError(f"{label} must be a lowercase SHA256")
    return value


def _safe_relative(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AuditError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise AuditError(f"{label} must remain relative")
    return path


@dataclass(frozen=True)
class Shard:
    phase: str
    manifest_root: Path
    relative_path: Path
    size: int
    sha256: str

    @property
    def path(self) -> Path:
        return self.manifest_root / self.relative_path


@dataclass(frozen=True)
class Sample:
    score: int
    value: dict[str, Any]


class _LowestSamples:
    """Keep the lexicographically lowest deterministic sample hashes."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._heap: list[tuple[int, str, dict[str, Any]]] = []

    def add(self, sample: Sample) -> None:
        if self.capacity <= 0:
            return
        sample_id = str(sample.value["sample_id"])
        item = (-sample.score, sample_id, sample.value)
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, item)
            return
        if item > self._heap[0]:
            heapq.heapreplace(self._heap, item)

    def values(self) -> list[dict[str, Any]]:
        return [
            value
            for _negative, _sample_id, value in sorted(
                self._heap,
                key=lambda item: (-item[0], item[1]),
            )
        ]


def _manifest_shards(
    manifest_path: Path,
    *,
    phase: str,
    source_id: str,
) -> tuple[dict[str, Any], list[Shard]]:
    try:
        payload = manifest_path.read_bytes()
        manifest = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read {phase} corpus manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise AuditError(f"{phase} corpus manifest must be an object")
    format_audit = manifest.get("format_audit")
    if not isinstance(format_audit, Mapping) or format_audit.get("complete") is not True:
        raise AuditError(f"{phase} corpus manifest has no complete format audit")
    outputs = format_audit.get("filtered_outputs")
    if not isinstance(outputs, Mapping) or not isinstance(outputs.get("train"), list):
        raise AuditError(f"{phase} corpus manifest has no filtered train inventory")
    root = manifest_path.parent.resolve()
    shards: list[Shard] = []
    seen: set[Path] = set()
    for index, raw in enumerate(outputs["train"]):
        if not isinstance(raw, Mapping) or raw.get("source_id") != source_id:
            continue
        relative = _safe_relative(
            raw.get("path"),
            label=f"{phase}.filtered_outputs.train[{index}].path",
        )
        if relative in seen:
            raise AuditError(f"{phase} repeats Chinese shard {relative}")
        seen.add(relative)
        size = raw.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise AuditError(f"{phase} shard {relative} has invalid size")
        shards.append(
            Shard(
                phase=phase,
                manifest_root=root,
                relative_path=relative,
                size=size,
                sha256=_require_sha256(
                    raw.get("sha256"),
                    label=f"{phase} shard {relative} sha256",
                ),
            )
        )
    if not shards:
        raise AuditError(f"{phase} has no {source_id} train shards")
    return (
        {
            "path": str(manifest_path.resolve()),
            "size": len(payload),
            "sha256": _sha256_bytes(payload),
            "corpus_fingerprint": manifest.get("corpus_fingerprint"),
            "shard_count": len(shards),
        },
        shards,
    )


def _risk_reasons(text: str) -> tuple[list[str], Counter[str]]:
    reasons: list[str] = []
    occurrences: Counter[str] = Counter()
    for name, pattern in _CONVERSION_MARKERS:
        count = len(pattern.findall(text))
        if count:
            occurrences[name] = count
            reasons.append(name)
    malformed_count = len(_MALFORMED_PUNCTUATION.findall(text))
    if malformed_count:
        occurrences["malformed_punctuation"] = malformed_count
        if malformed_count >= 2:
            reasons.append("malformed_punctuation")
    literal_newline_count = len(_LITERAL_ESCAPED_NEWLINE.findall(text))
    if literal_newline_count:
        occurrences["literal_escaped_newline"] = literal_newline_count
        if literal_newline_count >= 3:
            reasons.append("literal_escaped_newline")
    return sorted(set(reasons)), occurrences


def _sample_value(
    *,
    phase: str,
    shard: Shard,
    line_number: int,
    text: str,
    reasons: Sequence[str],
    stratum: str,
) -> Sample:
    text_sha = _sha256_bytes(text.encode("utf-8"))
    sample_id = _sha256_bytes(
        (
            phase
            + "\0"
            + shard.sha256
            + "\0"
            + str(line_number)
            + "\0"
            + text_sha
        ).encode("utf-8")
    )
    excerpt = re.sub(r"\s+", " ", text).strip()[:1600]
    return Sample(
        score=int(sample_id, 16),
        value={
            "sample_id": sample_id,
            "phase": phase,
            "stratum": stratum,
            "shard_path": str(shard.relative_path),
            "shard_sha256": shard.sha256,
            "line_number": line_number,
            "text_sha256": text_sha,
            "text_characters": len(text),
            "cjk_characters": len(_CJK.findall(text)),
            "risk_reasons": list(reasons),
            "excerpt": excerpt,
        },
    )


def _scan_phase(
    *,
    phase: str,
    shards: Sequence[Shard],
    risk_sample_size: int,
    control_sample_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    risk_samples = _LowestSamples(risk_sample_size)
    control_samples = _LowestSamples(control_sample_size)
    documents = 0
    characters = 0
    utf8_bytes = 0
    cjk_characters = 0
    risk_documents = 0
    conversion_documents = 0
    marker_documents: Counter[str] = Counter()
    marker_occurrences: Counter[str] = Counter()

    for shard in shards:
        path = shard.path
        try:
            metadata = path.stat()
        except OSError as exc:
            raise AuditError(f"cannot stat {phase} shard {path}: {exc}") from exc
        if not path.is_file() or metadata.st_size != shard.size:
            raise AuditError(f"{phase} shard size differs: {path}")
        if _sha256_file(path) != shard.sha256:
            raise AuditError(f"{phase} shard SHA256 differs: {path}")
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError as exc:
            raise AuditError(f"cannot open {phase} shard {path}: {exc}") from exc
        with handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AuditError(
                        f"{phase} shard {path}:{line_number} is invalid JSON: {exc}"
                    ) from exc
                if not isinstance(row, Mapping) or not isinstance(row.get("text"), str):
                    raise AuditError(
                        f"{phase} shard {path}:{line_number} has no text string"
                    )
                text = row["text"]
                reasons, occurrences = _risk_reasons(text)
                documents += 1
                characters += len(text)
                utf8_bytes += len(text.encode("utf-8"))
                cjk_characters += len(_CJK.findall(text))
                for name, count in occurrences.items():
                    marker_occurrences[name] += count
                    marker_documents[name] += 1
                if any(name in occurrences for name in _CONVERSION_MARKER_NAMES):
                    conversion_documents += 1
                if reasons:
                    risk_documents += 1
                    risk_samples.add(
                        _sample_value(
                            phase=phase,
                            shard=shard,
                            line_number=line_number,
                            text=text,
                            reasons=reasons,
                            stratum="risk",
                        )
                    )
                else:
                    control_samples.add(
                        _sample_value(
                            phase=phase,
                            shard=shard,
                            line_number=line_number,
                            text=text,
                            reasons=(),
                            stratum="control",
                        )
                    )
    if documents <= 0:
        raise AuditError(f"{phase} Chinese shard inventory is empty")
    samples = risk_samples.values() + control_samples.values()
    samples.sort(key=lambda row: (row["stratum"], row["sample_id"]))
    return (
        {
            "documents": documents,
            "characters": characters,
            "utf8_bytes": utf8_bytes,
            "cjk_characters": cjk_characters,
            "risk_documents": risk_documents,
            "risk_document_rate": risk_documents / documents,
            "high_precision_conversion_documents": conversion_documents,
            "high_precision_conversion_document_rate": conversion_documents / documents,
            "marker_documents": dict(sorted(marker_documents.items())),
            "marker_occurrences": dict(sorted(marker_occurrences.items())),
            "authenticated_shards": len(shards),
            "sample_count": len(samples),
            "risk_sample_count": sum(row["stratum"] == "risk" for row in samples),
            "control_sample_count": sum(row["stratum"] == "control" for row in samples),
        },
        samples,
    )


def _load_manual_decisions(
    path: Path,
    *,
    sample_sha256: str,
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read manual decisions: {exc}") from exc
    if not isinstance(value, Mapping):
        raise AuditError("manual decisions must be an object")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != DECISIONS_KIND
        or value.get("samples_sha256") != sample_sha256
    ):
        raise AuditError("manual decisions do not bind this sample inventory")
    reviewer = value.get("reviewer")
    reviewed_at = value.get("reviewed_at")
    decisions = value.get("decisions")
    if (
        not isinstance(reviewer, str)
        or not reviewer.strip()
        or not isinstance(reviewed_at, str)
        or not reviewed_at.strip()
        or not isinstance(decisions, list)
    ):
        raise AuditError("manual decisions metadata is incomplete")
    expected = {str(row["sample_id"]) for row in samples}
    observed: dict[str, str] = {}
    allowed = {
        "acceptable",
        "semantic_conversion_noise",
        "sentence_stitching_noise",
        "other_quality_noise",
    }
    for index, raw in enumerate(decisions):
        if not isinstance(raw, Mapping):
            raise AuditError(f"manual decisions[{index}] must be an object")
        sample_id = raw.get("sample_id")
        verdict = raw.get("verdict")
        if (
            not isinstance(sample_id, str)
            or sample_id not in expected
            or sample_id in observed
            or verdict not in allowed
        ):
            raise AuditError(f"manual decisions[{index}] is invalid or duplicated")
        observed[sample_id] = str(verdict)
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        raise AuditError(
            f"manual decisions must cover every emitted sample; missing {len(missing)}"
        )
    bad = sum(verdict != "acceptable" for verdict in observed.values())
    return {
        "path": str(path.resolve()),
        "size": len(payload),
        "sha256": _sha256_bytes(payload),
        "reviewer": reviewer.strip(),
        "reviewed_at": reviewed_at,
        "reviewed_samples": len(observed),
        "unacceptable_samples": bad,
        "unacceptable_rate": bad / len(observed) if observed else 1.0,
        "verdict_counts": dict(sorted(Counter(observed.values()).items())),
        "passed": bad == 0,
    }


def _manual_template(
    *,
    samples_sha256: str,
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": DECISIONS_KIND,
        "samples_sha256": samples_sha256,
        "reviewer": "REPLACE_WITH_REVIEWER",
        "reviewed_at": "REPLACE_WITH_ISO_8601_TIMESTAMP",
        "decisions": [
            {
                "sample_id": row["sample_id"],
                "verdict": "REPLACE_WITH_VERDICT",
                "notes": "",
            }
            for row in samples
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-manifest", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--cooldown-manifest", type=Path, default=DEFAULT_COOLDOWN)
    parser.add_argument("--source-id", default=DEFAULT_SOURCE_ID)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--risk-samples-per-phase", type=int, default=32)
    parser.add_argument("--control-samples-per-phase", type=int, default=32)
    parser.add_argument("--manual-decisions", type=Path)
    return parser


def _validate_count(value: int, *, label: str) -> int:
    if isinstance(value, bool) or value < 1 or value > 1_000:
        raise AuditError(f"{label} must be in [1, 1000]")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise AuditError(f"immutable output already exists: {output}")
    risk_size = _validate_count(
        args.risk_samples_per_phase,
        label="risk-samples-per-phase",
    )
    control_size = _validate_count(
        args.control_samples_per_phase,
        label="control-samples-per-phase",
    )
    inputs: dict[str, Any] = {}
    phase_shards: dict[str, list[Shard]] = {}
    for phase, raw_path in (
        ("primary", args.primary_manifest),
        ("cooldown", args.cooldown_manifest),
    ):
        path = raw_path.expanduser().resolve()
        identity, shards = _manifest_shards(
            path,
            phase=phase,
            source_id=args.source_id,
        )
        inputs[phase] = identity
        phase_shards[phase] = shards

    phase_stats: dict[str, Any] = {}
    samples: list[dict[str, Any]] = []
    for phase in ("primary", "cooldown"):
        stats, phase_samples = _scan_phase(
            phase=phase,
            shards=phase_shards[phase],
            risk_sample_size=risk_size,
            control_sample_size=control_size,
        )
        phase_stats[phase] = stats
        samples.extend(phase_samples)
    samples.sort(key=lambda row: (row["phase"], row["stratum"], row["sample_id"]))
    samples_payload = b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for row in samples
    )
    samples_sha = _sha256_bytes(samples_payload)

    manual: dict[str, Any]
    if args.manual_decisions is None:
        manual = {
            "path": None,
            "sha256": None,
            "reviewed_samples": 0,
            "unacceptable_samples": None,
            "unacceptable_rate": None,
            "passed": False,
            "status": "pending_complete_manual_review",
        }
    else:
        manual = _load_manual_decisions(
            args.manual_decisions.expanduser().resolve(),
            sample_sha256=samples_sha,
            samples=samples,
        )
        manual["status"] = "passed" if manual["passed"] else "failed"

    high_precision_documents = sum(
        stats["high_precision_conversion_documents"]
        for stats in phase_stats.values()
    )
    stitching_documents = sum(
        stats["marker_documents"].get("malformed_punctuation", 0)
        for stats in phase_stats.values()
    )
    statistical_passed = high_precision_documents == 0 and stitching_documents == 0
    passed = statistical_passed and manual.get("passed") is True
    attestation: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": ATTESTATION_KIND,
        "created_at": datetime.now(UTC).isoformat(),
        "source_id": args.source_id,
        "scanner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
            "policy": {
                "conversion_markers": [name for name, _pattern in _CONVERSION_MARKERS],
                "high_precision_conversion_documents_eq": 0,
                "malformed_punctuation_documents_eq": 0,
                "manual_unacceptable_samples_eq": 0,
            },
        },
        "inputs": inputs,
        "phases": phase_stats,
        "samples": {
            "path": "samples.jsonl",
            "size": len(samples_payload),
            "sha256": samples_sha,
            "count": len(samples),
            "risk_samples_per_phase": risk_size,
            "control_samples_per_phase": control_size,
        },
        "manual_review": manual,
        "gates": {
            "all_selected_shards_authenticated": True,
            "complete_streaming_scan": True,
            "high_precision_conversion_documents": high_precision_documents,
            "high_precision_conversion_passed": high_precision_documents == 0,
            "malformed_punctuation_documents": stitching_documents,
            "malformed_punctuation_passed": stitching_documents == 0,
            "manual_review_passed": manual.get("passed") is True,
        },
        "passed": passed,
        "authorizes_training": False,
        "status": (
            "passed_quality_gate_but_does_not_authorize_training"
            if passed
            else (
                "failed_statistical_quality_gate"
                if not statistical_passed
                else "pending_or_failed_manual_review"
            )
        ),
    }
    attestation["attestation_fingerprint"] = _canonical_sha256(attestation)

    output.mkdir(parents=True, exist_ok=False)
    _write_bytes_atomic(output / "samples.jsonl", samples_payload)
    _write_json(
        output / "manual-review-template.json",
        _manual_template(samples_sha256=samples_sha, samples=samples),
    )
    _write_json(output / "attestation.json", attestation)
    files = {}
    for name in (
        "attestation.json",
        "manual-review-template.json",
        "samples.jsonl",
    ):
        path = output / name
        files[name] = {
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "created_at": attestation["created_at"],
        "attestation_fingerprint": attestation["attestation_fingerprint"],
        "passed": passed,
        "authorizes_training": False,
        "files": files,
    }
    _write_json(output / "MANIFEST.json", manifest)
    complete = {
        "schema_version": SCHEMA_VERSION,
        "kind": COMPLETE_KIND,
        "manifest": "MANIFEST.json",
        "manifest_sha256": _sha256_file(output / "MANIFEST.json"),
        "passed": passed,
        "authorizes_training": False,
    }
    _write_json(output / "COMPLETE", complete)
    return attestation


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(args)
    except AuditError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
