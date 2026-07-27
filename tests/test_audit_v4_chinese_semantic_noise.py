# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_corpus(root: Path, texts: list[str]) -> Path:
    relative = Path(
        "filtered/chinese_fineweb2_cmn_hani/chunk-000000/train.jsonl"
    )
    shard = root / relative
    shard.parent.mkdir(parents=True)
    shard.write_text(
        "".join(
            json.dumps({"text": text}, ensure_ascii=False) + "\n"
            for text in texts
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
        risk_samples_per_phase=2,
        control_samples_per_phase=2,
        manual_decisions=None,
    )


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
