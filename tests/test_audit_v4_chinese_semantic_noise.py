# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_module() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "audit_v4_chinese_semantic_noise.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_v4_chinese_semantic_noise",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_module()
SAMPLES_PER_STRATUM = AUDIT.MIN_RISK_SAMPLES_PER_PHASE
SAMPLES_PER_PHASE = (
    AUDIT.MIN_RISK_SAMPLES_PER_PHASE + AUDIT.MIN_CONTROL_SAMPLES_PER_PHASE
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ordinary_text(index: int) -> str:
    return f"这是第{index}条结构完整且语义连贯的中文测试正文。"


def _risk_text(index: int) -> str:
    return f"这是第{index}条风险测试正文\\n第一段\\n第二段\\n第三段。"


def _soft_indicator_text(index: int) -> str:
    return f"这是第{index}条待复核正文\\n后续内容保持连贯。"


def _write_corpus(
    root: Path,
    texts: list[str],
    *,
    pad_ordinary_to: int | None = SAMPLES_PER_PHASE,
) -> Path:
    corpus_texts = list(texts)
    if pad_ordinary_to is not None:
        corpus_texts.extend(
            _ordinary_text(index)
            for index in range(len(corpus_texts), pad_ordinary_to)
        )
    relative = Path(
        "filtered/chinese_fineweb2_cmn_hani/chunk-000000/train.jsonl"
    )
    shard = root / relative
    shard.parent.mkdir(parents=True)
    shard.write_text(
        "".join(
            json.dumps({"text": text}, ensure_ascii=False) + "\n"
            for text in corpus_texts
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "kind": "twen_extracted_base_jsonl_corpus",
        "corpus_fingerprint": "c" * 64,
        "format_audit": {
            "complete": True,
            "filtered_outputs": {
                "train": [
                    {
                        "path": str(relative),
                        "size": shard.stat().st_size,
                        "sha256": _sha256(shard),
                        "source_id": "chinese_fineweb2_cmn_hani",
                    }
                ],
                "validation": [],
            },
        },
    }
    path = root / "corpus-manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _scan_one_phase(
    tmp_path: Path,
    texts: list[str],
    *,
    risk_sample_size: int = SAMPLES_PER_STRATUM,
    control_sample_size: int = SAMPLES_PER_STRATUM,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _write_corpus(
        tmp_path,
        texts,
        pad_ordinary_to=None,
    )
    _identity, shards = AUDIT._manifest_shards(
        manifest,
        phase="primary",
        source_id="chinese_fineweb2_cmn_hani",
    )
    return AUDIT._scan_phase(
        phase="primary",
        shards=shards,
        risk_sample_size=risk_sample_size,
        control_sample_size=control_sample_size,
    )


def _args(
    *,
    primary: Path,
    cooldown: Path,
    output: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        primary_manifest=primary,
        cooldown_manifest=cooldown,
        source_id="chinese_fineweb2_cmn_hani",
        output=output,
        risk_samples_per_phase=AUDIT.MIN_RISK_SAMPLES_PER_PHASE,
        control_samples_per_phase=AUDIT.MIN_CONTROL_SAMPLES_PER_PHASE,
        manual_decisions=None,
    )


def test_cli_requires_explicit_phase_manifests_and_source_id(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        AUDIT._parser().parse_args(["--output", str(tmp_path / "evidence")])


def test_full_scan_fails_closed_on_conversion_and_stitching_noise(
    tmp_path: Path,
) -> None:
    primary = _write_corpus(
        tmp_path / "primary",
        [
            "这是一段结构清晰的中文教材正文，用于介绍牛顿运动定律。",
            "下面为大年夜家介绍这台汽车的发念头，内容存在机械转换污染。",
        ],
    )
    cooldown = _write_corpus(
        tmp_path / "cooldown",
        [
            "第一条新闻结束，。第二条新闻又开始，。这不是自然的文章边界。",
            "另一段正常、连贯且主题一致的中文说明文字。",
        ],
    )
    output = tmp_path / "evidence"

    result = AUDIT.run(
        _args(primary=primary, cooldown=cooldown, output=output)
    )

    assert result["passed"] is False
    assert result["authorizes_training"] is False
    assert result["status"] == "failed_statistical_quality_gate"
    assert (
        result["phases"]["primary"]["marker_documents"][
            "da_nian_ye_substitution"
        ]
        == 1
    )
    assert (
        result["phases"]["primary"]["marker_documents"][
            "engine_substitution"
        ]
        == 1
    )
    assert (
        result["phases"]["cooldown"]["marker_documents"][
            "malformed_punctuation"
        ]
        == 1
    )
    assert (output / "samples.jsonl").is_file()
    assert (output / "manual-review-template.json").is_file()
    assert (output / "MANIFEST.json").is_file()
    assert (output / "COMPLETE").is_file()
    samples = [
        json.loads(line)
        for line in (output / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    template = json.loads(
        (output / "manual-review-template.json").read_text(encoding="utf-8")
    )
    assert len(samples) == 2 * SAMPLES_PER_PHASE
    assert len({row["sample_id"] for row in samples}) == 2 * SAMPLES_PER_PHASE
    assert len(template["decisions"]) == 2 * SAMPLES_PER_PHASE
    assert {
        row["sample_id"] for row in template["decisions"]
    } == {
        row["sample_id"] for row in samples
    }


def test_clean_statistical_scan_still_requires_complete_manual_review(
    tmp_path: Path,
) -> None:
    primary = _write_corpus(
        tmp_path / "primary",
        ["这是一段结构完整、语义连贯的中文教材正文。"],
    )
    cooldown = _write_corpus(
        tmp_path / "cooldown",
        ["这是一段主题明确、标点正常的中文复习材料。"],
    )

    result = AUDIT.run(
        _args(
            primary=primary,
            cooldown=cooldown,
            output=tmp_path / "evidence",
        )
    )

    assert result["gates"]["high_precision_conversion_passed"] is True
    assert result["gates"]["malformed_punctuation_passed"] is True
    assert result["gates"]["manual_review_passed"] is False
    assert result["status"] == "pending_or_failed_manual_review"
    assert result["passed"] is False


@pytest.mark.parametrize(
    (
        "risk_population",
        "expected_risk_samples",
        "expected_challenge_samples",
        "population_exhausted",
        "population_fully_sampled",
    ),
    (
        (0, 0, SAMPLES_PER_STRATUM, True, True),
        (7, 7, SAMPLES_PER_STRATUM - 7, True, True),
        (
            SAMPLES_PER_STRATUM,
            SAMPLES_PER_STRATUM,
            0,
            False,
            True,
        ),
        (
            SAMPLES_PER_STRATUM + 8,
            SAMPLES_PER_STRATUM,
            0,
            False,
            False,
        ),
    ),
)
def test_three_stratum_contract_at_risk_population_boundaries(
    tmp_path: Path,
    risk_population: int,
    expected_risk_samples: int,
    expected_challenge_samples: int,
    population_exhausted: bool,
    population_fully_sampled: bool,
) -> None:
    stats, samples = _scan_one_phase(
        tmp_path,
        [
            *(_risk_text(index) for index in range(risk_population)),
            *(_ordinary_text(index) for index in range(SAMPLES_PER_PHASE)),
        ],
    )

    assert stats["risk_documents"] == risk_population
    assert stats["risk_population_documents"] == risk_population
    assert stats["risk_population_exhausted"] is population_exhausted
    assert stats["risk_population_fully_sampled"] is population_fully_sampled
    assert stats["risk_sample_count"] == expected_risk_samples
    assert stats["review_challenge_sample_count"] == expected_challenge_samples
    assert stats["review_challenge_soft_indicator_sample_count"] == 0
    assert (
        stats["review_challenge_fallback_sample_count"]
        == expected_challenge_samples
    )
    assert stats["risk_or_review_challenge_sample_count"] == SAMPLES_PER_STRATUM
    assert stats["control_sample_count"] == SAMPLES_PER_STRATUM
    assert stats["sample_count"] == SAMPLES_PER_PHASE
    assert stats["sample_strata"] == {
        "risk": expected_risk_samples,
        "review_challenge": expected_challenge_samples,
        "control": SAMPLES_PER_STRATUM,
    }
    assert stats["sample_contract_satisfied"] is True

    strata = {
        stratum: {
            row["sample_id"]
            for row in samples
            if row["stratum"] == stratum
        }
        for stratum in ("risk", "review_challenge", "control")
    }
    assert strata["risk"].isdisjoint(strata["review_challenge"])
    assert strata["risk"].isdisjoint(strata["control"])
    assert strata["review_challenge"].isdisjoint(strata["control"])
    assert len(set().union(*strata.values())) == SAMPLES_PER_PHASE
    assert all(
        row["review_reasons"]
        == ["deterministic_risk_population_shortfall_fallback"]
        for row in samples
        if row["stratum"] == "review_challenge"
    )
    assert all(
        row["review_reasons"] == []
        for row in samples
        if row["stratum"] in {"risk", "control"}
    )


def test_soft_indicators_fill_challenge_before_ordinary_fallback(
    tmp_path: Path,
) -> None:
    risk_count = 7
    soft_count = 10
    fallback_count = SAMPLES_PER_STRATUM - risk_count - soft_count
    stats, samples = _scan_one_phase(
        tmp_path,
        [
            *(_risk_text(index) for index in range(risk_count)),
            *(_soft_indicator_text(index) for index in range(soft_count)),
            *(
                _ordinary_text(index)
                for index in range(fallback_count + SAMPLES_PER_STRATUM)
            ),
        ],
    )

    challenges = [
        row for row in samples if row["stratum"] == "review_challenge"
    ]
    controls = [row for row in samples if row["stratum"] == "control"]
    soft_challenges = [
        row
        for row in challenges
        if row["review_reasons"] == ["soft_indicator:literal_escaped_newline"]
    ]
    fallback_challenges = [
        row
        for row in challenges
        if row["review_reasons"]
        == ["deterministic_risk_population_shortfall_fallback"]
    ]

    assert stats["soft_indicator_documents"] == soft_count
    assert stats["review_challenge_soft_indicator_sample_count"] == soft_count
    assert stats["review_challenge_fallback_sample_count"] == fallback_count
    assert len(soft_challenges) == soft_count
    assert len(fallback_challenges) == fallback_count
    assert len(controls) == SAMPLES_PER_STRATUM
    assert {
        row["sample_id"] for row in fallback_challenges
    }.isdisjoint({row["sample_id"] for row in controls})
    assert all(row["review_reasons"] == [] for row in controls)


def test_ordinary_fallback_and_control_partition_is_disjoint(
    tmp_path: Path,
) -> None:
    stats, samples = _scan_one_phase(
        tmp_path,
        [_ordinary_text(index) for index in range(SAMPLES_PER_PHASE)],
    )

    challenges = {
        row["sample_id"]
        for row in samples
        if row["stratum"] == "review_challenge"
    }
    controls = {
        row["sample_id"]
        for row in samples
        if row["stratum"] == "control"
    }
    assert len(challenges) == SAMPLES_PER_STRATUM
    assert len(controls) == SAMPLES_PER_STRATUM
    assert challenges.isdisjoint(controls)
    assert len(challenges | controls) == SAMPLES_PER_PHASE
    assert stats["ordinary_documents"] == SAMPLES_PER_PHASE
    assert stats["review_challenge_fallback_sample_count"] == SAMPLES_PER_STRATUM


def test_scan_fails_closed_when_disjoint_sample_contract_is_impossible(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        AUDIT.AuditError,
        match="cannot satisfy disjoint manual-review sample contract",
    ):
        _scan_one_phase(
            tmp_path,
            [_ordinary_text(index) for index in range(SAMPLES_PER_PHASE - 1)],
        )


def test_detector_thresholds_and_hard_gate_inputs_are_unchanged() -> None:
    conversion_reasons, conversion_occurrences = AUDIT._risk_reasons(
        "这段文字包含发念头这一确定的转换标记。"
    )
    malformed_once_reasons, malformed_once_occurrences = AUDIT._risk_reasons(
        "第一句结束，。第二句继续。"
    )
    malformed_twice_reasons, malformed_twice_occurrences = AUDIT._risk_reasons(
        "第一句结束，。第二句结束，。第三句继续。"
    )
    newline_twice_reasons, newline_twice_occurrences = AUDIT._risk_reasons(
        "第一段\\n第二段\\n第三段。"
    )
    newline_thrice_reasons, newline_thrice_occurrences = AUDIT._risk_reasons(
        "第一段\\n第二段\\n第三段\\n第四段。"
    )

    assert conversion_reasons == ["engine_substitution"]
    assert conversion_occurrences["engine_substitution"] == 1
    assert malformed_once_reasons == []
    assert malformed_once_occurrences["malformed_punctuation"] == 1
    assert malformed_twice_reasons == ["malformed_punctuation"]
    assert malformed_twice_occurrences["malformed_punctuation"] == 2
    assert newline_twice_reasons == []
    assert newline_twice_occurrences["literal_escaped_newline"] == 2
    assert newline_thrice_reasons == ["literal_escaped_newline"]
    assert newline_thrice_occurrences["literal_escaped_newline"] == 3


@pytest.mark.parametrize(
    ("risk_samples", "control_samples"),
    (
        (1, AUDIT.MIN_CONTROL_SAMPLES_PER_PHASE),
        (AUDIT.MIN_RISK_SAMPLES_PER_PHASE - 1, AUDIT.MIN_CONTROL_SAMPLES_PER_PHASE),
        (True, AUDIT.MIN_CONTROL_SAMPLES_PER_PHASE),
        (AUDIT.MIN_RISK_SAMPLES_PER_PHASE, 1),
        (AUDIT.MIN_RISK_SAMPLES_PER_PHASE, AUDIT.MIN_CONTROL_SAMPLES_PER_PHASE - 1),
        (AUDIT.MIN_RISK_SAMPLES_PER_PHASE, True),
    ),
)
def test_scan_rejects_manual_sample_quota_below_policy(
    risk_samples: int,
    control_samples: int,
) -> None:
    with pytest.raises(AUDIT.AuditError, match=r"must be in \[32, 1000\]"):
        AUDIT.recompute_scan(
            primary_manifest=Path("/not-read"),
            cooldown_manifest=Path("/not-read"),
            source_id="chinese_fineweb2_cmn_hani",
            risk_sample_size=risk_samples,
            control_sample_size=control_samples,
        )


@pytest.mark.parametrize(
    ("reviewer", "reviewed_at", "error"),
    (
        (
            "REPLACE_WITH_REVIEWER",
            "2026-07-27T00:00:00+00:00",
            "metadata is incomplete",
        ),
        (
            "fixture-reviewer",
            "REPLACE_WITH_ISO_8601_TIMESTAMP",
            "metadata is incomplete",
        ),
        (
            "fixture-reviewer",
            "2026-07-27T00:00:00",
            "timezone-aware ISO-8601",
        ),
        (
            "fixture-reviewer",
            "not-a-time",
            "timezone-aware ISO-8601",
        ),
        (
            "fixture-reviewer",
            "2026-07-27T00:00:00+00:60",
            "timezone-aware ISO-8601",
        ),
        (
            "fixture-reviewer",
            "2026-07-27T00:00:00+24:00",
            "timezone-aware ISO-8601",
        ),
    ),
)
def test_manual_review_rejects_placeholder_or_invalid_timestamp_metadata(
    tmp_path: Path,
    reviewer: str,
    reviewed_at: str,
    error: str,
) -> None:
    samples = [{"sample_id": "sample-1"}]
    decision_path = tmp_path / "manual-decisions.json"
    decision_path.write_text(
        json.dumps(
            {
                "schema_version": AUDIT.SCHEMA_VERSION,
                "kind": AUDIT.DECISIONS_KIND,
                "samples_sha256": "a" * 64,
                "reviewer": reviewer,
                "reviewed_at": reviewed_at,
                "decisions": [
                    {
                        "sample_id": "sample-1",
                        "verdict": "acceptable",
                        "notes": "",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AUDIT.AuditError, match=error):
        AUDIT._load_manual_decisions(
            decision_path,
            sample_sha256="a" * 64,
            samples=samples,
        )


def test_scan_rejects_a_shard_changed_after_manifest_creation(
    tmp_path: Path,
) -> None:
    primary = _write_corpus(
        tmp_path / "primary",
        ["这是原始且已经绑定哈希的正文。"],
    )
    cooldown = _write_corpus(
        tmp_path / "cooldown",
        ["这是另一个阶段的正常正文。"],
    )
    shard = (
        primary.parent
        / "filtered/chinese_fineweb2_cmn_hani/chunk-000000/train.jsonl"
    )
    shard.write_text(
        json.dumps({"text": "身份被替换"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(AUDIT.AuditError, match=r"shard (size|SHA256) differs"):
        AUDIT.run(
            _args(
                primary=primary,
                cooldown=cooldown,
                output=tmp_path / "evidence",
            )
        )
