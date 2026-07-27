#!/usr/bin/env python3
"""Build an authenticated dense training/validation report with stdlib SVG charts.

The script is intentionally read-only with respect to the run and evaluation
directories.  It requires a completed evaluation and writes a separate report
bundle containing the derived JSON summary, Markdown report, and SVG figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#4f46e5",
    "#65a30d",
)

AUTHENTICATED_REPORT_KINDS = frozenset(
    {
        "twen_v1_final_validation_report",
        "twen_dense_final_validation_report",
    }
)
GPU_MEASUREMENT_PATTERN = re.compile(
    r"(?P<number>[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf(?:inity)?))"
    r"(?:\s*(?P<unit>[%A-Za-z]+))?",
    flags=re.IGNORECASE,
)
V3_RUN_ID = "base-dense-v3-500m"
V3_MTP_AFFECTED_SOURCE_PATH = "src/twen/modeling/mtp.py"
V3_MTP_AFFECTED_GIT_BLOB_SHA1 = "6e5ba4fd397946ce224252948bed7b692db97ecf"
V3_MTP_FIX_COMMIT = "c9a08cfec134ba7b3fa808f5a76836b62b5646b5"


def _methodology_errata_for_run(run_id: str) -> list[dict[str, Any]]:
    if run_id != V3_RUN_ID:
        return []
    return [
        {
            "id": "mtp_rope_position_alignment",
            "disclosed_on": "2026-07-26",
            "affected_source": {
                "path": V3_MTP_AFFECTED_SOURCE_PATH,
                "git_blob_sha1": V3_MTP_AFFECTED_GIT_BLOB_SHA1,
            },
            "observed_rope_position_offset_tokens": 0,
            "required_rope_position_offset_tokens": 1,
            "validation_objective": "ntp_only",
            "fully_native_aligned_mtp_claim_supported": False,
            "causal_mtp_benefit_claim_supported": False,
            "fixed_by_commit": V3_MTP_FIX_COMMIT,
        }
    ]


def _methodology_erratum_section(run_id: str, *, zh_cn: bool) -> str:
    if not _methodology_errata_for_run(run_id):
        return ""
    if zh_cn:
        return """\
> **方法学勘误 (2026-07-26)**: 训练后独立代码审查确认, 本轮虽然严格加载了
> Qwen3.5 checkpoint 的 15 张原生 `mtp.*` 参数, 并按
> `h_t + embed(x_(t+1)) -> x_(t+2)` 构造辅助目标, 但 MTP decoder 的 RoPE
> `position_ids` 错用了 `t`, 正确位置应为 `t+1`。因此, 本报告记录的 NTP-only
> candidate/shared/teacher validation NLL、PPL、吞吐和训练日志仍是原始实测事实;
> 但 v3 的 MTP 路径不能再称为“完全原生对齐”, 也不能据此作 MTP 增益的因果结论。
> 该问题已在 v4 启动前由提交 `c9a08cf` 修复并加入独立位置对齐回归测试。

"""
    return """\
> **Methodology erratum (2026-07-26):** A post-training independent code audit
> confirmed that this run strictly loaded all 15 native `mtp.*` tensors and
> formed the auxiliary `h_t + embed(x_(t+1)) -> x_(t+2)` objective, but the MTP
> decoder applied RoPE position `t` instead of the correct shifted position
> `t+1`. The NTP-only candidate/shared/teacher validation NLL, perplexity,
> throughput, and recorded training telemetry in this report remain the
> measurements that were actually produced. However, v3's MTP path must not be
> described as fully native-aligned or used for a causal MTP-benefit claim.
> Commit `c9a08cf` fixed the issue and added an independent position-alignment
> regression test before v4 launch.

"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [_finite(row.get(key)) for row in rows]
    selected = [value for value in values if value is not None]
    return statistics.fmean(selected) if selected else None


def _weighted_mean(
    rows: Sequence[Mapping[str, Any]], key: str, weight_key: str = "tokens_this_step"
) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = _finite(row.get(key))
        weight = _finite(row.get(weight_key))
        if value is None or weight is None or weight <= 0:
            continue
        numerator += value * weight
        denominator += weight
    return numerator / denominator if denominator else None


def _fmt(value: Any, digits: int = 4) -> str:
    number = _finite(value)
    if number is None:
        return "n/a"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.3f}M"
    if abs(number) >= 10_000:
        return f"{number:,.0f}"
    if abs(number) >= 100:
        return f"{number:,.1f}"
    return f"{number:.{digits}f}"


def _source_id(source_path: str) -> str:
    normalized = source_path.replace("\\", "/")
    marker = "/extracted/"
    if marker not in normalized:
        return "unknown"
    tail = normalized.split(marker, 1)[1]
    return tail.split("/", 1)[0] or "unknown"


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _training_source_id(source_path: str) -> str:
    source = _source_id(source_path)
    if source != "unknown":
        return source
    name = Path(source_path).name
    name = re.sub(r"\.(jsonl|parquet)$", "", name)
    name = re.sub(r"[-_]\d{6}$", "", name)
    return name or "unknown"


def _invert_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("matrix must be non-empty and square")
    augmented = [
        [float(value) for value in row]
        + [1.0 if column == index else 0.0 for column in range(size)]
        for index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("regression matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    left - factor * right
                    for left, right in zip(augmented[row], augmented[column], strict=True)
                ]
    return [row[size:] for row in augmented]


def _matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    if not left or not right or len(left[0]) != len(right):
        raise ValueError("matrix dimensions do not align")
    columns = list(zip(*right, strict=True))
    return [
        [sum(a * b for a, b in zip(row, column, strict=True)) for column in columns] for row in left
    ]


def _source_fixed_effect_regression(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    sources: Sequence[str],
    hac_lag: int = 50,
) -> dict[str, Any]:
    selected: list[tuple[list[float], float, float]] = []
    encoded_sources = tuple(sources[:-1])
    for row in rows:
        value = _finite(row.get(metric))
        tokens = _finite(row.get("tokens"))
        fractions = row.get("_source_fractions")
        if value is None or tokens is None or not isinstance(fractions, Mapping):
            continue
        design = [
            1.0,
            *(float(fractions.get(source, 0.0)) for source in encoded_sources),
            tokens / 100_000_000.0,
        ]
        selected.append((design, value, tokens))
    if len(selected) <= len(encoded_sources) + 3:
        raise ValueError(f"too few rows for source-conditioned {metric} regression")
    design = [row[0] for row in selected]
    values = [row[1] for row in selected]
    width = len(design[0])
    xtx = [
        [sum(row[left] * row[right] for row in design) for right in range(width)]
        for left in range(width)
    ]
    inverse = _invert_matrix(xtx)
    xty = [
        sum(row[index] * value for row, value in zip(design, values, strict=True))
        for index in range(width)
    ]
    beta = [
        sum(inverse[row][column] * xty[column] for column in range(width)) for row in range(width)
    ]
    fitted = [
        sum(coefficient * value for coefficient, value in zip(beta, row, strict=True))
        for row in design
    ]
    residuals = [value - estimate for value, estimate in zip(values, fitted, strict=True)]
    value_mean = statistics.fmean(values)
    total_sum_squares = sum((value - value_mean) ** 2 for value in values)
    residual_sum_squares = sum(value * value for value in residuals)
    meat = [[0.0 for _ in range(width)] for _ in range(width)]
    for row, residual in zip(design, residuals, strict=True):
        for left in range(width):
            for right in range(width):
                meat[left][right] += residual * residual * row[left] * row[right]
    effective_lag = min(hac_lag, len(design) - 1)
    for lag in range(1, effective_lag + 1):
        weight = 1.0 - lag / (effective_lag + 1.0)
        for index in range(lag, len(design)):
            current = design[index]
            previous = design[index - lag]
            product = residuals[index] * residuals[index - lag] * weight
            for left in range(width):
                for right in range(width):
                    meat[left][right] += product * (
                        current[left] * previous[right] + previous[left] * current[right]
                    )
    covariance = _matmul(_matmul(inverse, meat), inverse)
    slope = beta[-1]
    slope_se = math.sqrt(max(covariance[-1][-1], 0.0))
    source_effects = {sources[-1]: 0.0}
    source_effects.update({source: beta[index + 1] for index, source in enumerate(encoded_sources)})
    centered_effect = statistics.fmean(source_effects.values())
    adjusted = [
        {
            "tokens": tokens,
            "value": value
            - sum(source_effects[source] * float(fractions.get(source, 0.0)) for source in sources)
            + centered_effect,
        }
        for (_, value, tokens), row in zip(selected, rows, strict=True)
        if isinstance((fractions := row.get("_source_fractions")), Mapping)
    ]
    return {
        "metric": metric,
        "rows": len(selected),
        "sources": list(sources),
        "reference_source": sources[-1],
        "r_squared": (
            1.0 - residual_sum_squares / total_sum_squares if total_sum_squares > 0 else None
        ),
        "raw_standard_deviation": statistics.pstdev(values),
        "residual_standard_deviation": statistics.pstdev(residuals),
        "slope_per_100m_tokens": slope,
        "hac_lag": effective_lag,
        "hac_standard_error": slope_se,
        "confidence_95": [slope - 1.96 * slope_se, slope + 1.96 * slope_se],
        "source_effects_relative_to_reference": source_effects,
        "source_adjusted_series": adjusted,
    }


def _source_conditioned_training_analysis(
    config: Mapping[str, Any], metrics: Sequence[dict[str, Any]]
) -> dict[str, Any] | None:
    data = config.get("data")
    if not isinstance(data, Mapping):
        return None
    primary_value = data.get("manifest_path")
    if not isinstance(primary_value, str):
        return None
    primary_path = Path(primary_value)
    if not primary_path.is_file():
        return None
    primary = _read_json(primary_path)
    primary_shards = primary.get("shards")
    if not isinstance(primary_shards, list) or not primary_shards:
        return None
    from twen.data.cursor import (
        DatasetLayout,
        DeterministicCooldownCursor,
        DeterministicGlobalCursor,
    )

    def layout_and_sources(
        manifest: Mapping[str, Any],
    ) -> tuple[DatasetLayout, dict[str, str]]:
        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError("training manifest has no shards")
        return (
            DatasetLayout.from_shards(
                ((str(entry["shard_id"]), int(entry["sequence_count"])) for entry in shards),
                fingerprint=str(manifest["dataset_fingerprint"]),
            ),
            {
                str(entry["shard_id"]): _training_source_id(str(entry.get("source_path", "")))
                for entry in shards
            },
        )

    primary_layout, primary_sources = layout_and_sources(primary)
    cooldown_value = data.get("quality_cooldown_manifest_path")
    cooldown_start = data.get("quality_cooldown_start_tokens")
    cooldown_sources: dict[str, str] = {}
    if isinstance(cooldown_value, str) and cooldown_start is not None:
        cooldown_path = Path(cooldown_value)
        if not cooldown_path.is_file():
            return None
        cooldown = _read_json(cooldown_path)
        cooldown_layout, cooldown_sources = layout_and_sources(cooldown)
        cursor: Any = DeterministicCooldownCursor(
            primary_layout,
            cooldown_layout,
            seed=int(data.get("shuffle_seed", 3407)),
            cooldown_start_tokens=int(cooldown_start),
            shuffle=True,
        )
    else:
        cursor = DeterministicGlobalCursor(
            primary_layout,
            seed=int(data.get("shuffle_seed", 3407)),
            shuffle=True,
        )
    max_sequence_length = int(data["max_sequence_length"])
    global_batch_tokens = int(data["global_batch_tokens"])
    if global_batch_tokens % max_sequence_length:
        raise ValueError("global batch tokens are not divisible by max sequence length")
    batch_samples = global_batch_tokens // max_sequence_length
    all_sources = sorted(set(primary_sources.values()) | set(cooldown_sources.values()))
    pure_batches = 0
    for row in metrics:
        phase = str(row.get("data_phase", "primary"))
        references = cursor.plan_global_batch(batch_samples)
        source_map = cooldown_sources if phase == "cooldown" else primary_sources
        counts: dict[str, int] = defaultdict(int)
        for reference in references:
            counts[source_map[reference.shard_id]] += 1
        row["_source_fractions"] = {
            source: count / batch_samples for source, count in counts.items()
        }
        pure_batches += len(counts) == 1
        expected_phase = getattr(cursor, "active_phase", "primary")
        if phase != expected_phase:
            raise ValueError(
                f"replayed data phase differs at step {row.get('step')}: "
                f"{phase} != {expected_phase}"
            )
        cursor.commit(
            global_batch_samples=batch_samples,
            token_count=int(row["tokens_this_step"]),
        )
    warmup_tokens = int(config["optimizer"]["warmup_tokens"])
    post_warmup_primary = [
        row
        for row in metrics
        if row.get("data_phase", "primary") == "primary" and int(row["tokens"]) > warmup_tokens
    ]
    regressions = {
        metric: _source_fixed_effect_regression(
            post_warmup_primary,
            metric=metric,
            sources=all_sources,
        )
        for metric in ("loss", "ntp", "teacher_kd", "mtp", "anchor_kl")
        if any(_finite(row.get(metric)) is not None for row in post_warmup_primary)
    }
    post_warmup_pure_batches = sum(
        len(row["_source_fractions"]) == 1 for row in post_warmup_primary
    )
    source_means: dict[str, dict[str, Any]] = {}
    for source in all_sources:
        weights = [float(row["_source_fractions"].get(source, 0.0)) for row in post_warmup_primary]
        total_weight = sum(weights)
        if total_weight <= 0:
            continue
        values: dict[str, Any] = {"effective_steps": total_weight}
        values.update(
            {
                metric: sum(
                    float(row[metric]) * weight
                    for row, weight in zip(post_warmup_primary, weights, strict=True)
                    if _finite(row.get(metric)) is not None
                )
                / total_weight
                for metric in ("loss", "ntp", "teacher_kd", "mtp", "anchor_kl")
                if any(_finite(row.get(metric)) is not None for row in post_warmup_primary)
            }
        )
        source_means[source] = values
    return {
        "method": "deterministic cursor replay plus source fixed-effects OLS/Newey-West HAC",
        "batch_samples": batch_samples,
        "phase_replay_matches_all_steps": True,
        "steps": len(metrics),
        "pure_source_batches": pure_batches,
        "pure_source_batch_fraction": pure_batches / len(metrics),
        "analysis_phase": {
            "kind": "post_warmup_primary",
            "includes": ["primary_stable", "primary_decay"],
            "excludes": ["warmup", "cooldown"],
            "rows": len(post_warmup_primary),
        },
        "post_warmup_primary_rows": len(post_warmup_primary),
        "post_warmup_primary_pure_source_batches": post_warmup_pure_batches,
        "post_warmup_primary_pure_source_batch_fraction": (
            post_warmup_pure_batches / len(post_warmup_primary)
        ),
        "sources": all_sources,
        "source_weighted_means": source_means,
        "regressions": regressions,
    }


def _parse_gpu_measurement(
    value: str,
    *,
    field: str,
    expected_unit: str | None,
) -> float:
    match = GPU_MEASUREMENT_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid GPU runtime sample value for {field}: {value!r}")
    unit = match.group("unit")
    if unit is not None and unit != expected_unit:
        expected = "a bare number" if expected_unit is None else f"{expected_unit!r} or no unit"
        raise ValueError(
            f"unexpected unit for GPU runtime sample field {field}: {unit!r}; expected {expected}"
        )
    result = float(match.group("number"))
    if not math.isfinite(result):
        raise ValueError(f"non-finite GPU runtime sample value for {field}: {value!r}")
    return result


def _summarize_gpu_sample(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError(f"GPU runtime sample is empty: {path}")
    field_aliases = {
        "power.draw [W]": ("power_draw_w", "W"),
        "power.limit [W]": ("power_limit_w", "W"),
        "clocks.current.sm [MHz]": ("sm_clock_mhz", "MHz"),
        "clocks.current.memory [MHz]": ("memory_clock_mhz", "MHz"),
        "utilization.gpu [%]": ("gpu_utilization_percent", "%"),
        "utilization.memory [%]": ("memory_utilization_percent", "%"),
        "memory.used [MiB]": ("memory_used_mib", "MiB"),
        "memory.free [MiB]": ("memory_free_mib", "MiB"),
        "temperature.gpu": ("temperature_c", None),
    }
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows):
        normalized = {key.strip(): value.strip() for key, value in source.items()}
        row: dict[str, Any] = {
            "sample": index,
            "timestamp": normalized.get("timestamp"),
            "pstate": normalized.get("pstate"),
        }
        for raw, (target, expected_unit) in field_aliases.items():
            value = normalized.get(raw)
            if value not in {None, "", "N/A", "[N/A]"}:
                row[target] = _parse_gpu_measurement(
                    value,
                    field=raw,
                    expected_unit=expected_unit,
                )
        rows.append(row)
    statistics_by_field: dict[str, dict[str, float]] = {}
    for field, _expected_unit in field_aliases.values():
        values = [float(row[field]) for row in rows if field in row]
        if values:
            statistics_by_field[field] = {
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
            }
    pstates: dict[str, int] = defaultdict(int)
    for row in rows:
        pstates[str(row.get("pstate"))] += 1
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "sample_count": len(rows),
        "pstates": dict(sorted(pstates.items())),
        "statistics": statistics_by_field,
        "samples": rows,
    }


def _nice_range(values: Sequence[float]) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        span = max(abs(low) * 0.1, 1.0)
        return low - span, high + span
    pad = (high - low) * 0.07
    return low - pad, high + pad


def _svg_document(body: str, *, width: int = 1200, height: int = 650) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        '<rect width="100%" height="100%" fill="#ffffff"/>\n'
        "<style>text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#111827}"
        ".grid{stroke:#e5e7eb;stroke-width:1}.axis{stroke:#374151;stroke-width:1.5}"
        ".tick{font-size:13px;fill:#4b5563}.title{font-size:24px;font-weight:700}"
        ".label{font-size:15px;font-weight:600}.legend{font-size:14px}</style>\n"
        f"{body}\n</svg>\n"
    )


def _write_line_chart(
    path: Path,
    *,
    title: str,
    series: Mapping[str, Sequence[tuple[float, float]]],
    x_label: str,
    y_label: str,
    horizontal_lines: Sequence[tuple[float, str]] = (),
) -> None:
    cleaned: dict[str, list[tuple[float, float]]] = {}
    for name, points in series.items():
        valid = [
            (float(x), float(y))
            for x, y in points
            if math.isfinite(float(x)) and math.isfinite(float(y))
        ]
        if valid:
            cleaned[name] = valid
    if not cleaned:
        raise ValueError(f"line chart {title!r} has no finite points")
    all_x = [point[0] for points in cleaned.values() for point in points]
    all_y = [point[1] for points in cleaned.values() for point in points]
    all_y.extend(value for value, _ in horizontal_lines if math.isfinite(value))
    x_min, x_max = _nice_range(all_x)
    y_min, y_max = _nice_range(all_y)
    left, top, right, bottom = 105.0, 70.0, 1160.0, 555.0

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    def sy(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    parts = [f'<text class="title" x="{left}" y="38">{html.escape(title)}</text>']
    for index in range(6):
        fraction = index / 5
        x = left + fraction * (right - left)
        y = bottom - fraction * (bottom - top)
        x_value = x_min + fraction * (x_max - x_min)
        y_value = y_min + fraction * (y_max - y_min)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}"/>')
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>')
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="578" text-anchor="middle">{html.escape(_fmt(x_value, 2))}</text>'
        )
        parts.append(
            f'<text class="tick" x="96" y="{y + 5:.1f}" text-anchor="end">{html.escape(_fmt(y_value, 3))}</text>'
        )
    parts.extend(
        (
            f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
            f'<text class="label" x="{(left + right) / 2:.1f}" y="620" text-anchor="middle">{html.escape(x_label)}</text>',
            f'<text class="label" x="24" y="{(top + bottom) / 2:.1f}" text-anchor="middle" transform="rotate(-90 24 {(top + bottom) / 2:.1f})">{html.escape(y_label)}</text>',
        )
    )
    for value, label in horizontal_lines:
        y = sy(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#6b7280" stroke-width="1.5" stroke-dasharray="7 5"/>'
        )
        parts.append(
            f'<text class="tick" x="{right - 5}" y="{y - 6:.1f}" text-anchor="end">{html.escape(label)}</text>'
        )
    legend_x = left
    for index, (name, points) in enumerate(cleaned.items()):
        color = COLORS[index % len(COLORS)]
        coordinates = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        parts.append(
            f'<line x1="{legend_x}" y1="602" x2="{legend_x + 24}" y2="602" stroke="{color}" stroke-width="4"/>'
        )
        parts.append(f'<text class="legend" x="{legend_x + 30}" y="607">{html.escape(name)}</text>')
        legend_x += max(140, len(name) * 9 + 55)
    _atomic_write(path, _svg_document("\n".join(parts)))


def _write_grouped_bar_chart(
    path: Path,
    *,
    title: str,
    groups: Sequence[str],
    series: Mapping[str, Sequence[float]],
    y_label: str,
) -> None:
    if not groups or not series:
        raise ValueError(f"bar chart {title!r} has no data")
    if any(len(values) != len(groups) for values in series.values()):
        raise ValueError("bar series length differs from group count")
    values = [float(value) for row in series.values() for value in row if math.isfinite(value)]
    if not values:
        raise ValueError(f"bar chart {title!r} has no finite values")
    y_min = min(0.0, min(values))
    y_max = max(values)
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0
    pad = (y_max - y_min) * 0.08
    y_max += pad
    if y_min < 0:
        y_min -= pad
    left, top, right, bottom = 105.0, 70.0, 1160.0, 535.0

    def sy(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    parts = [f'<text class="title" x="{left}" y="38">{html.escape(title)}</text>']
    for index in range(6):
        fraction = index / 5
        y = bottom - fraction * (bottom - top)
        value = y_min + fraction * (y_max - y_min)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>')
        parts.append(
            f'<text class="tick" x="96" y="{y + 5:.1f}" text-anchor="end">{html.escape(_fmt(value, 3))}</text>'
        )
    zero_y = sy(0.0)
    parts.append(
        f'<line class="axis" x1="{left}" y1="{zero_y:.1f}" x2="{right}" y2="{zero_y:.1f}"/>'
    )
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>')
    count = len(series)
    group_width = (right - left) / len(groups)
    bar_width = min(48.0, group_width * 0.75 / count)
    for group_index, label in enumerate(groups):
        center = left + (group_index + 0.5) * group_width
        for series_index, (_name, row) in enumerate(series.items()):
            value = float(row[group_index])
            x = center + (series_index - (count - 1) / 2) * bar_width - bar_width * 0.43
            y = min(zero_y, sy(value))
            height = max(1.0, abs(sy(value) - zero_y))
            color = COLORS[series_index % len(COLORS)]
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width * 0.86:.1f}" height="{height:.1f}" fill="{color}" opacity="0.9"/>'
            )
        display = label if len(label) <= 22 else label[:20] + "…"
        parts.append(
            f'<text class="tick" x="{center:.1f}" y="558" text-anchor="middle">{html.escape(display)}</text>'
        )
    parts.append(
        f'<text class="label" x="24" y="{(top + bottom) / 2:.1f}" text-anchor="middle" transform="rotate(-90 24 {(top + bottom) / 2:.1f})">{html.escape(y_label)}</text>'
    )
    legend_x = left
    for index, name in enumerate(series):
        color = COLORS[index % len(COLORS)]
        parts.append(f'<rect x="{legend_x}" y="594" width="18" height="12" fill="{color}"/>')
        parts.append(f'<text class="legend" x="{legend_x + 25}" y="605">{html.escape(name)}</text>')
        legend_x += max(140, len(name) * 9 + 50)
    _atomic_write(path, _svg_document("\n".join(parts)))


def _validate_checkpoint(checkpoint: Path) -> dict[str, Any]:
    manifest_path = checkpoint / "manifest.json"
    complete_path = checkpoint / "COMPLETE"
    manifest = _read_json(manifest_path)
    expected = complete_path.read_text(encoding="ascii").strip()
    actual = _sha256_file(manifest_path)
    if expected != actual:
        raise ValueError(f"checkpoint COMPLETE mismatch: {checkpoint}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("checkpoint manifest has no file inventory")
    for relative, digest in files.items():
        path = checkpoint / relative
        if not path.is_file() or _sha256_file(path) != digest:
            raise ValueError(f"checkpoint inventory mismatch: {path}")
    metadata = _read_json(checkpoint / "metadata.json")
    return {
        "path": str(checkpoint.resolve()),
        "complete_sha256": _sha256_file(complete_path),
        "manifest_sha256": actual,
        "inventory_file_count": len(files),
        "metadata": metadata,
    }


def _summarize_training_lifecycle(
    events: Sequence[Mapping[str, Any]],
    *,
    elapsed_wall_seconds: float,
    canonical_step_wall_seconds: float,
) -> dict[str, Any]:
    lifecycle_names = {
        "session_start",
        "initialized",
        "resume",
        "train_start",
        "graceful_stop",
        "train_complete",
    }
    retained_fields = (
        "event",
        "session_id",
        "timestamp_utc",
        "step",
        "tokens",
        "checkpoint",
        "checkpoint_kind",
        "checkpoint_tag",
        "fork_from",
        "reason",
    )
    lifecycle_events = [
        {key: event[key] for key in retained_fields if key in event}
        for event in events
        if event.get("event") in lifecycle_names
    ]
    train_starts = [event for event in events if event.get("event") == "train_start"]
    session_starts = [event for event in events if event.get("event") == "session_start"]
    resumes = [event for event in events if event.get("event") == "resume"]
    graceful_stops = [event for event in events if event.get("event") == "graceful_stop"]
    terminal_session_ids = {
        str(event["session_id"])
        for event in events
        if event.get("event") in {"graceful_stop", "train_complete"}
        and isinstance(event.get("session_id"), str)
    }
    started_session_ids = {
        str(event["session_id"])
        for event in session_starts
        if isinstance(event.get("session_id"), str)
    }
    outside_canonical = max(0.0, elapsed_wall_seconds - canonical_step_wall_seconds)
    return {
        "elapsed_wall_seconds": elapsed_wall_seconds,
        "canonical_committed_step_wall_seconds": canonical_step_wall_seconds,
        "outside_canonical_committed_step_wall_seconds": outside_canonical,
        "elapsed_includes_resume_intervals": bool(resumes or len(train_starts) > 1),
        "session_count": len(session_starts) if session_starts else len(train_starts),
        "train_start_count": len(train_starts),
        "resume_count": len(resumes),
        "graceful_stop_count": len(graceful_stops),
        "sessions_without_terminal_event_count": len(started_session_ids - terminal_session_ids),
        "events": lifecycle_events,
        "accounting_caveat": (
            "Elapsed wall spans the first train_start through train_complete. The canonical "
            "committed-step wall sum excludes checkpoint writes, model construction/preflight, "
            "inter-session pauses, and compute replayed after rollback; their difference must "
            "not be interpreted as idle time alone."
        ),
    }


def _summarize_training(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = _read_jsonl(run_dir / "metrics.jsonl")
    telemetry = _read_jsonl(run_dir / "telemetry.jsonl")
    events = _read_jsonl(run_dir / "events.jsonl")
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("resolved config is not a mapping")
    if not metrics or len(metrics) != len(telemetry):
        raise ValueError("training metrics/telemetry are empty or have different lengths")
    metric_steps = [int(row["step"]) for row in metrics]
    telemetry_steps = [int(row["step"]) for row in telemetry]
    if metric_steps != telemetry_steps or metric_steps != list(range(1, len(metrics) + 1)):
        raise ValueError("training metrics are not a contiguous step sequence")
    final_events = [event for event in events if event.get("event") == "train_complete"]
    if len(final_events) != 1:
        raise ValueError("run does not contain exactly one train_complete event")
    final_checkpoint = Path(str(final_events[0]["checkpoint"]))
    if not final_checkpoint.is_absolute():
        final_checkpoint = (Path.cwd() / final_checkpoint).resolve()
    checkpoint = _validate_checkpoint(final_checkpoint)
    checkpoint_metadata = checkpoint["metadata"]
    if int(checkpoint_metadata["global_step"]) != metric_steps[-1]:
        raise ValueError("final checkpoint step differs from metrics")
    if int(checkpoint_metadata["committed_tokens"]) != int(metrics[-1]["tokens"]):
        raise ValueError("final checkpoint token count differs from metrics")
    start = next(event for event in events if event.get("event") == "train_start")
    end = final_events[0]
    wall_duration = (
        _parse_timestamp(str(end["timestamp_utc"])) - _parse_timestamp(str(start["timestamp_utc"]))
    ).total_seconds()
    clip_threshold = float(config["optimizer"]["grad_clip_norm"])
    clip_count = sum(float(row["grad_norm"]) > clip_threshold for row in metrics)
    alignment_rows = [row for row in metrics if "hidden_alignment_loss" in row]
    checkpoint_events = [
        event
        for event in events
        if event.get("event") == "checkpoint_complete"
        and _finite(event.get("duration_seconds")) is not None
    ]
    first = metrics[: min(50, len(metrics))]
    last = metrics[-min(50, len(metrics)) :]
    source_conditioned = _source_conditioned_training_analysis(config, metrics)
    total_tokens = sum(int(row["tokens_this_step"]) for row in telemetry)
    compute_seconds = sum(float(row["compute_step_seconds"]) for row in telemetry)
    wall_step_seconds = sum(float(row["wall_clock_step_seconds"]) for row in telemetry)
    data_wait_seconds = sum(float(row["data_wait_seconds"]) for row in telemetry)
    lifecycle = _summarize_training_lifecycle(
        events,
        elapsed_wall_seconds=wall_duration,
        canonical_step_wall_seconds=wall_step_seconds,
    )
    summary = {
        "run_id": config.get("run_id"),
        "track": config.get("track"),
        "stage": config.get("stage"),
        "steps": len(metrics),
        "committed_tokens": int(metrics[-1]["tokens"]),
        "tokens_from_step_sum": total_tokens,
        "wall_duration_seconds": wall_duration,
        "compute_step_seconds": compute_seconds,
        "wall_step_seconds": wall_step_seconds,
        "aggregate_compute_tokens_per_second": total_tokens / compute_seconds,
        "aggregate_wall_step_tokens_per_second": total_tokens / wall_step_seconds,
        "end_to_end_tokens_per_second": int(metrics[-1]["tokens"]) / wall_duration,
        "lifecycle": lifecycle,
        "data_wait_seconds": data_wait_seconds,
        "data_wait_fraction_of_compute": data_wait_seconds / compute_seconds,
        "ordinary_steps": len(metrics) - len(alignment_rows),
        "alignment_steps": len(alignment_rows),
        "grad_clip_threshold": clip_threshold,
        "grad_norm_over_clip_threshold_steps": clip_count,
        "grad_norm_over_clip_threshold_fraction": clip_count / len(metrics),
        "grad_norm_max": max(float(row["grad_norm"]) for row in metrics),
        "gpu_peak_allocated_gib": max(float(row["gpu_peak_allocated_gib"]) for row in telemetry),
        "gpu_peak_reserved_gib": max(float(row["gpu_peak_reserved_gib"]) for row in telemetry),
        "checkpoint_count": len(checkpoint_events),
        "checkpoint_duration_seconds_total": sum(
            float(row["duration_seconds"]) for row in checkpoint_events
        ),
        "checkpoint_duration_seconds_mean": _mean(checkpoint_events, "duration_seconds"),
        "checkpoint_duration_seconds_max": max(
            float(row["duration_seconds"]) for row in checkpoint_events
        ),
        "first_50_weighted": {
            key: _weighted_mean(first, key)
            for key in ("loss", "ntp", "teacher_kd", "anchor_kl", "grad_norm")
        },
        "last_50_weighted": {
            key: _weighted_mean(last, key)
            for key in ("loss", "ntp", "teacher_kd", "anchor_kl", "grad_norm")
        },
        "source_conditioned_analysis": source_conditioned,
        "checkpoint": checkpoint,
        "config": {
            "max_tokens": int(config["optimizer"]["max_tokens"]),
            "warmup_tokens": int(config["optimizer"]["warmup_tokens"]),
            "lr_schedule": str(config["optimizer"].get("lr_schedule", "cosine")),
            "min_lr_ratio": float(config["optimizer"].get("min_lr_ratio", 0.1)),
            "decay_tokens": config["optimizer"].get("decay_tokens"),
            "micro_batch_size": int(config["data"]["micro_batch_size"]),
            "global_batch_tokens": int(config["data"]["global_batch_tokens"]),
            "quality_cooldown_start_tokens": config["data"].get("quality_cooldown_start_tokens"),
            "losses": config["losses"],
            "mtp_present": "mtp" in config["losses"],
            "mtp_weight": float(config["losses"].get("mtp", 0.0)),
        },
    }
    raw = {
        "metrics": metrics,
        "telemetry": telemetry,
        "events": events,
        "checkpoint_events": checkpoint_events,
    }
    return summary, raw


def _load_completed_evaluation(evaluation_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = evaluation_dir / "manifest.json"
    complete_path = evaluation_dir / "COMPLETE"
    if not manifest_path.is_file() or not complete_path.is_file():
        raise ValueError(f"evaluation is incomplete: {evaluation_dir}")
    expected = complete_path.read_text(encoding="ascii").strip()
    actual = _sha256_file(manifest_path)
    if expected != actual:
        raise ValueError("evaluation COMPLETE does not authenticate manifest.json")
    manifest = _read_json(manifest_path)
    plan_path = evaluation_dir / "PLAN.json"
    if manifest.get("plan_sha256") != _sha256_file(plan_path):
        raise ValueError("evaluation PLAN hash mismatch")
    roles = manifest.get("roles")
    if not isinstance(roles, dict) or not {"candidate", "shared", "teacher"} <= set(roles):
        raise ValueError("dense final evaluation requires candidate/shared/teacher roles")
    raw_results: dict[str, dict[str, dict[str, Any]]] = {}
    for role in ("candidate", "shared", "teacher"):
        role_results: dict[str, dict[str, Any]] = {}
        for entry in roles[role].get("shards", []):
            complete = evaluation_dir / str(entry["path"]) / "COMPLETE"
            if not complete.is_file() or _sha256_file(complete) != entry["complete_sha256"]:
                raise ValueError(f"evaluation shard COMPLETE mismatch: {complete}")
            marker = _read_json(complete)
            outputs = marker.get("outputs")
            if not isinstance(outputs, list) or not outputs:
                raise ValueError(f"evaluation shard marker has no output inventory: {complete}")
            for output in outputs:
                if not isinstance(output, dict) or not isinstance(output.get("path"), str):
                    raise ValueError(f"evaluation shard output inventory is invalid: {complete}")
                path = complete.parent / output["path"]
                if (
                    not path.is_file()
                    or path.stat().st_size != int(output.get("size", -1))
                    or _sha256_file(path) != output.get("sha256")
                ):
                    raise ValueError(f"evaluation shard output hash/size mismatch: {path}")
            result = _read_json(complete.parent / "result.json")
            shard_id = str(result.get("source_shard_id"))
            if result.get("role") != role or shard_id in role_results:
                raise ValueError(f"evaluation role/shard lineage mismatch: {complete.parent}")
            role_results[shard_id] = result
        if len(role_results) != len(roles[role].get("shards", [])):
            raise ValueError(f"role {role} shard inventory mismatch")
        raw_results[role] = role_results
    identity = {
        "path": str(evaluation_dir.resolve()),
        "manifest_sha256": actual,
        "complete_sha256": _sha256_file(complete_path),
        "plan_sha256": _sha256_file(plan_path),
    }
    return manifest, {"identity": identity, "shard_results": raw_results}


def _summarize_validation(
    evaluation_dir: Path, prepared_manifest_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluation, raw = _load_completed_evaluation(evaluation_dir)
    prepared = _read_json(prepared_manifest_path)
    plan = _read_json(evaluation_dir / "PLAN.json")
    if plan.get("prepared_manifest_sha256") != _sha256_file(prepared_manifest_path):
        raise ValueError("evaluation used a different prepared validation manifest")
    if plan.get("prepared_dataset_fingerprint") != prepared.get("dataset_fingerprint"):
        raise ValueError("evaluation/prepared dataset fingerprint mismatch")
    entries = prepared.get("shards")
    if not isinstance(entries, list) or not entries:
        raise ValueError("prepared validation manifest has no shards")
    shard_results = raw["shard_results"]
    expected_ids = {str(entry["shard_id"]) for entry in entries}
    for role, results in shard_results.items():
        if set(results) != expected_ids:
            raise ValueError(f"role {role} did not cover every validation shard")
    source_totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "shards": 0,
            "sequences": 0,
            "input_tokens": 0,
            "roles": defaultdict(lambda: {"nll_sum": 0.0, "predicted_tokens": 0}),
        }
    )
    shard_rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        shard_id = str(entry["shard_id"])
        source = _source_id(str(entry.get("source_path", "")))
        aggregate = source_totals[source]
        aggregate["shards"] += 1
        aggregate["sequences"] += int(entry["sequence_count"])
        aggregate["input_tokens"] += int(entry["token_count"])
        row: dict[str, Any] = {
            "index": index,
            "shard_id": shard_id,
            "source": source,
            "sequences": int(entry["sequence_count"]),
            "input_tokens": int(entry["token_count"]),
            "roles": {},
        }
        for role in ("candidate", "shared", "teacher"):
            result = shard_results[role][shard_id]
            role_total = aggregate["roles"][role]
            role_total["nll_sum"] += float(result["nll_sum"])
            role_total["predicted_tokens"] += int(result["predicted_tokens"])
            row["roles"][role] = {
                "mean_nll": float(result["mean_nll"]),
                "predicted_tokens": int(result["predicted_tokens"]),
            }
        row["candidate_over_shared_nll_improvement"] = (
            row["roles"]["shared"]["mean_nll"] - row["roles"]["candidate"]["mean_nll"]
        )
        shard_rows.append(row)
    sources: list[dict[str, Any]] = []
    for source, totals in sorted(source_totals.items()):
        role_summary: dict[str, Any] = {}
        for role, role_total in totals["roles"].items():
            mean_nll = role_total["nll_sum"] / role_total["predicted_tokens"]
            role_summary[role] = {
                **role_total,
                "mean_nll": mean_nll,
                "perplexity": math.exp(mean_nll) if mean_nll < 700 else None,
            }
        denominator = role_summary["shared"]["mean_nll"] - role_summary["teacher"]["mean_nll"]
        closed = (
            (role_summary["shared"]["mean_nll"] - role_summary["candidate"]["mean_nll"])
            / denominator
            if denominator > 0
            else None
        )
        sources.append(
            {
                "source": source,
                "shards": totals["shards"],
                "sequences": totals["sequences"],
                "input_tokens": totals["input_tokens"],
                "roles": role_summary,
                "teacher_gap_closed_fraction": closed,
            }
        )
    roles = {
        role: {
            key: evaluation["roles"][role][key]
            for key in ("sequences", "predicted_tokens", "nll_sum", "mean_nll", "perplexity")
        }
        for role in ("candidate", "shared", "teacher")
    }
    for role in ("candidate", "shared", "teacher"):
        directories = [
            evaluation_dir / str(entry["path"]) for entry in evaluation["roles"][role]["shards"]
        ]
        started = min((directory / "SHARD_STATE.json").stat().st_mtime for directory in directories)
        completed = max(
            float(_read_json(directory / "COMPLETE")["completed_unix_seconds"])
            for directory in directories
        )
        duration = completed - started
        if duration <= 0:
            raise ValueError(f"role {role} has invalid wall-clock timing")
        roles[role]["timing"] = {
            "started_unix_seconds": started,
            "completed_unix_seconds": completed,
            "duration_seconds": duration,
            "predicted_tokens_per_second": int(roles[role]["predicted_tokens"]) / duration,
            "start_evidence": "SHARD_STATE filesystem mtime",
            "completion_evidence": "authenticated COMPLETE.completed_unix_seconds",
        }
    governance = prepared.get("lineage", {})
    runtime = dict(evaluation.get("runtime", plan.get("runtime", {})))
    runtime.setdefault("dtype", plan.get("dtype"))
    gpu_sample = _summarize_gpu_sample(evaluation_dir / "runtime-gpu-sample.csv")
    summary = {
        "identity": raw["identity"],
        "checkpoint_state": evaluation.get("checkpoint_state"),
        "checkpoint_inference_lineage": evaluation.get(
            "checkpoint_inference_lineage", plan.get("checkpoint_inference_lineage", {})
        ),
        "runtime": runtime,
        "runtime_gpu_sample": gpu_sample,
        "roles": roles,
        "acceptance": evaluation.get("acceptance", {}),
        "prepared_manifest": {
            "path": str(prepared_manifest_path.resolve()),
            "sha256": _sha256_file(prepared_manifest_path),
            "dataset_fingerprint": prepared.get("dataset_fingerprint"),
            "sequence_count": int(prepared["sequence_count"]),
            "input_token_count": int(prepared["token_count"]),
            "shard_count": len(entries),
        },
        "sources": sources,
        "shards": shard_rows,
        "data_governance": {
            "lineage_kind": governance.get("kind"),
            "role": governance.get("role"),
            "research_only": bool(governance.get("research_only", False)),
            "ready_for_training": bool(governance.get("ready_for_training", False)),
            "pending_audits": list(governance.get("pending_audits", [])),
            "audits": governance.get("audits", {}),
        },
    }
    return summary, raw


def _require_finite_number(value: Any, *, field: str) -> float:
    number = _finite(value)
    if number is None:
        raise ValueError(f"baseline summary has no finite {field}")
    return number


def _load_authenticated_report_summary(
    summary_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = summary_path.resolve()
    if summary_path.name != "summary.json":
        raise ValueError("baseline summary must be the summary.json at a report bundle root")
    bundle_dir = summary_path.parent
    manifest_path = bundle_dir / "MANIFEST.json"
    complete_path = bundle_dir / "COMPLETE"
    if not summary_path.is_file() or not manifest_path.is_file() or not complete_path.is_file():
        raise ValueError(f"baseline report bundle is incomplete: {bundle_dir}")
    expected_manifest_sha256 = complete_path.read_text(encoding="ascii").strip()
    manifest_sha256 = _sha256_file(manifest_path)
    if expected_manifest_sha256 != manifest_sha256:
        raise ValueError("baseline report COMPLETE does not authenticate MANIFEST.json")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "twen_report_bundle":
        raise ValueError("baseline report MANIFEST.json kind/schema is unsupported")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("baseline report MANIFEST.json has no file inventory")
    entry = files.get("summary.json")
    if not isinstance(entry, dict):
        raise ValueError("baseline report MANIFEST.json does not inventory summary.json")
    if summary_path.stat().st_size != int(entry.get("size", -1)) or _sha256_file(
        summary_path
    ) != entry.get("sha256"):
        raise ValueError("baseline report summary.json hash/size mismatch")
    summary = _read_json(summary_path)
    if summary.get("schema_version") != 1 or summary.get("kind") not in AUTHENTICATED_REPORT_KINDS:
        raise ValueError("baseline summary kind/schema is unsupported")
    training = summary.get("training")
    validation = summary.get("validation")
    if not isinstance(training, dict) or not isinstance(validation, dict):
        raise ValueError("baseline summary has no training/validation lineage")
    run_id = training.get("run_id")
    checkpoint = training.get("checkpoint")
    validation_identity = validation.get("identity")
    if (
        not isinstance(run_id, str)
        or not run_id
        or not isinstance(checkpoint, dict)
        or not isinstance(validation_identity, dict)
    ):
        raise ValueError("baseline summary training/evaluation identity is incomplete")
    checkpoint_manifest_sha256 = checkpoint.get("manifest_sha256")
    evaluation_manifest_sha256 = validation_identity.get("manifest_sha256")
    if (
        not isinstance(checkpoint_manifest_sha256, str)
        or manifest.get("source_checkpoint_manifest_sha256") != checkpoint_manifest_sha256
    ):
        raise ValueError("baseline checkpoint lineage differs from report MANIFEST.json")
    if (
        not isinstance(evaluation_manifest_sha256, str)
        or manifest.get("source_evaluation_manifest_sha256") != evaluation_manifest_sha256
    ):
        raise ValueError("baseline evaluation lineage differs from report MANIFEST.json")
    roles = validation.get("roles")
    acceptance = validation.get("acceptance")
    prepared = validation.get("prepared_manifest")
    if (
        not isinstance(roles, dict)
        or not isinstance(acceptance, dict)
        or not isinstance(prepared, dict)
    ):
        raise ValueError("baseline summary validation metrics/lineage are incomplete")
    for role in ("candidate", "shared", "teacher"):
        role_metrics = roles.get(role)
        if not isinstance(role_metrics, dict):
            raise ValueError(f"baseline summary is missing role {role}")
        mean_nll = _require_finite_number(
            role_metrics.get("mean_nll"), field=f"validation.roles.{role}.mean_nll"
        )
        perplexity = _require_finite_number(
            role_metrics.get("perplexity"),
            field=f"validation.roles.{role}.perplexity",
        )
        if mean_nll < 0 or perplexity <= 0:
            raise ValueError(f"baseline summary has invalid {role} NLL/perplexity")
        expected_perplexity = math.exp(mean_nll) if mean_nll < 700 else None
        if expected_perplexity is None or not math.isclose(
            perplexity, expected_perplexity, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(f"baseline summary {role} perplexity is inconsistent with mean NLL")
        predicted_tokens = role_metrics.get("predicted_tokens")
        if isinstance(predicted_tokens, bool) or not isinstance(predicted_tokens, int):
            raise ValueError(f"baseline summary has invalid {role} predicted token count")
        if predicted_tokens <= 0:
            raise ValueError(f"baseline summary has non-positive {role} predicted token count")
    _require_finite_number(
        acceptance.get("teacher_gap_closed_fraction"),
        field="validation.acceptance.teacher_gap_closed_fraction",
    )
    for field in ("sha256", "dataset_fingerprint"):
        if not isinstance(prepared.get(field), str) or not prepared[field]:
            raise ValueError(f"baseline summary prepared-manifest {field} is missing")
    identity = {
        "path": str(summary_path),
        "summary_sha256": _sha256_file(summary_path),
        "bundle_manifest_path": str(manifest_path),
        "bundle_manifest_sha256": manifest_sha256,
        "bundle_complete_path": str(complete_path),
        "bundle_complete_sha256": _sha256_file(complete_path),
        "report_kind": summary["kind"],
    }
    return summary, identity


def _metric_change(*, baseline: float, current: float, lower_is_better: bool) -> dict[str, Any]:
    absolute_change = current - baseline
    relative_change_fraction = (
        absolute_change / baseline if not math.isclose(baseline, 0.0) else None
    )
    improvement = -absolute_change if lower_is_better else absolute_change
    return {
        "baseline": baseline,
        "current": current,
        "absolute_change": absolute_change,
        "relative_change_fraction": relative_change_fraction,
        "relative_change_percent": (
            relative_change_fraction * 100.0 if relative_change_fraction is not None else None
        ),
        "improvement": improvement,
        "better_direction": "lower" if lower_is_better else "higher",
    }


def _cross_version_history_from_authenticated_baseline(
    *,
    baseline_summary: Mapping[str, Any],
    baseline_identity: Mapping[str, Any],
    training: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    baseline_training = baseline_summary["training"]
    baseline_validation = baseline_summary["validation"]
    baseline_run_id = str(baseline_training["run_id"])
    authenticated_by = str(baseline_identity["summary_sha256"])
    history: list[dict[str, Any]] = []
    nested = baseline_summary.get("baseline_comparison")
    if isinstance(nested, Mapping):
        nested_baseline = nested.get("baseline")
        nested_current = nested.get("current")
        nested_roles = nested.get("roles")
        nested_gap = nested.get("teacher_gap_closed_fraction")
        nested_comparability = nested.get("comparability")
        if not all(
            isinstance(value, Mapping)
            for value in (
                nested_baseline,
                nested_current,
                nested_roles,
                nested_gap,
                nested_comparability,
            )
        ):
            raise ValueError("authenticated baseline has malformed nested baseline comparison")
        assert isinstance(nested_baseline, Mapping)
        assert isinstance(nested_current, Mapping)
        assert isinstance(nested_roles, Mapping)
        assert isinstance(nested_gap, Mapping)
        assert isinstance(nested_comparability, Mapping)
        if not nested_comparability or not all(
            value is True for value in nested_comparability.values()
        ):
            raise ValueError("nested baseline comparison is not fully comparable")
        if nested_current.get("run_id") != baseline_run_id:
            raise ValueError("nested baseline current run differs from authenticated baseline")
        if (
            nested_current.get("checkpoint_manifest_sha256")
            != baseline_training["checkpoint"]["manifest_sha256"]
            or nested_current.get("evaluation_manifest_sha256")
            != baseline_validation["identity"]["manifest_sha256"]
        ):
            raise ValueError("nested baseline current lineage differs from authenticated baseline")
        nested_current_prepared = nested_current.get("prepared_manifest")
        if not isinstance(nested_current_prepared, Mapping):
            raise ValueError("nested baseline current prepared-manifest is missing")
        for field in (
            "sha256",
            "dataset_fingerprint",
            "sequence_count",
            "input_token_count",
            "shard_count",
        ):
            if nested_current_prepared.get(field) != baseline_validation["prepared_manifest"].get(
                field
            ):
                raise ValueError(
                    "nested baseline current prepared-manifest differs from "
                    f"authenticated baseline: {field}"
                )
        for role in ("candidate", "shared", "teacher"):
            role_comparison = nested_roles.get(role)
            if not isinstance(role_comparison, Mapping):
                raise ValueError(f"nested baseline comparison is missing role {role}")
            direct_role = baseline_validation["roles"][role]
            for metric in ("mean_nll", "perplexity"):
                metric_comparison = role_comparison.get(metric)
                if not isinstance(metric_comparison, Mapping):
                    raise ValueError(f"nested baseline comparison is missing {role} {metric}")
                nested_current_value = _require_finite_number(
                    metric_comparison.get("current"),
                    field=f"nested.roles.{role}.{metric}.current",
                )
                if not math.isclose(
                    nested_current_value,
                    float(direct_role[metric]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"nested baseline current {role} {metric} differs from "
                        "authenticated baseline"
                    )
            token_comparison = role_comparison.get("predicted_tokens")
            if (
                not isinstance(token_comparison, Mapping)
                or int(token_comparison.get("current", -1)) != int(direct_role["predicted_tokens"])
                or token_comparison.get("match") is not True
            ):
                raise ValueError(f"nested baseline current {role} token count is inconsistent")
        nested_gap_current = _require_finite_number(
            nested_gap.get("current"),
            field="nested.teacher_gap_closed_fraction.current",
        )
        direct_gap = float(baseline_validation["acceptance"]["teacher_gap_closed_fraction"])
        if not math.isclose(
            nested_gap_current,
            direct_gap,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "nested baseline current teacher-gap-closed differs from authenticated baseline"
            )
        prior_candidate = _require_finite_number(
            nested_roles["candidate"]["mean_nll"].get("baseline"),
            field="nested.roles.candidate.mean_nll.baseline",
        )
        prior_gap = _require_finite_number(
            nested_gap.get("baseline"),
            field="nested.teacher_gap_closed_fraction.baseline",
        )
        prior_run_id = nested_baseline.get("run_id")
        if not isinstance(prior_run_id, str) or not prior_run_id:
            raise ValueError("nested baseline run_id is missing")
        history.append(
            {
                "run_id": prior_run_id,
                "candidate_mean_nll": prior_candidate,
                "teacher_gap_closed_fraction": prior_gap,
                "checkpoint_manifest_sha256": nested_baseline.get("checkpoint_manifest_sha256"),
                "evaluation_manifest_sha256": nested_baseline.get("evaluation_manifest_sha256"),
                "provenance": "authenticated_nested_baseline_comparison",
                "authenticated_by_summary_sha256": authenticated_by,
            }
        )
    history.extend(
        (
            {
                "run_id": baseline_run_id,
                "candidate_mean_nll": float(baseline_validation["roles"]["candidate"]["mean_nll"]),
                "teacher_gap_closed_fraction": float(
                    baseline_validation["acceptance"]["teacher_gap_closed_fraction"]
                ),
                "checkpoint_manifest_sha256": baseline_training["checkpoint"]["manifest_sha256"],
                "evaluation_manifest_sha256": baseline_validation["identity"]["manifest_sha256"],
                "provenance": "authenticated_baseline_report_summary",
                "authenticated_by_summary_sha256": authenticated_by,
            },
            {
                "run_id": str(training["run_id"]),
                "candidate_mean_nll": float(validation["roles"]["candidate"]["mean_nll"]),
                "teacher_gap_closed_fraction": float(
                    validation["acceptance"]["teacher_gap_closed_fraction"]
                ),
                "checkpoint_manifest_sha256": training["checkpoint"]["manifest_sha256"],
                "evaluation_manifest_sha256": validation["identity"]["manifest_sha256"],
                "provenance": "current_authenticated_evaluation",
            },
        )
    )
    run_ids = [row["run_id"] for row in history]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("cross-version history contains duplicate run IDs")
    return history


def _build_baseline_comparison(
    *,
    baseline_summary: Mapping[str, Any],
    baseline_identity: Mapping[str, Any],
    training: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_training = baseline_summary["training"]
    baseline_validation = baseline_summary["validation"]
    baseline_prepared = baseline_validation["prepared_manifest"]
    current_prepared = validation["prepared_manifest"]
    for field in ("sha256", "dataset_fingerprint"):
        if baseline_prepared.get(field) != current_prepared.get(field):
            raise ValueError(f"baseline/current validation prepared-manifest {field} mismatch")
    for field in (
        "sequence_count",
        "input_token_count",
        "shard_count",
    ):
        if int(baseline_prepared.get(field, -1)) != int(current_prepared.get(field, -2)):
            raise ValueError(f"baseline/current validation prepared-manifest {field} mismatch")
    role_comparisons: dict[str, Any] = {}
    token_counts_match = True
    for role in ("candidate", "shared", "teacher"):
        baseline_role = baseline_validation["roles"][role]
        current_role = validation["roles"][role]
        baseline_tokens = int(baseline_role["predicted_tokens"])
        current_tokens = int(current_role["predicted_tokens"])
        token_counts_match &= baseline_tokens == current_tokens
        role_comparisons[role] = {
            "predicted_tokens": {
                "baseline": baseline_tokens,
                "current": current_tokens,
                "match": baseline_tokens == current_tokens,
            },
            "mean_nll": _metric_change(
                baseline=float(baseline_role["mean_nll"]),
                current=float(current_role["mean_nll"]),
                lower_is_better=True,
            ),
            "perplexity": _metric_change(
                baseline=float(baseline_role["perplexity"]),
                current=float(current_role["perplexity"]),
                lower_is_better=True,
            ),
        }
    if not token_counts_match:
        raise ValueError("baseline/current validation role predicted-token counts mismatch")
    baseline_gap = float(baseline_validation["acceptance"]["teacher_gap_closed_fraction"])
    current_gap = _require_finite_number(
        validation["acceptance"].get("teacher_gap_closed_fraction"),
        field="current validation acceptance.teacher_gap_closed_fraction",
    )
    cross_version_history = _cross_version_history_from_authenticated_baseline(
        baseline_summary=baseline_summary,
        baseline_identity=baseline_identity,
        training=training,
        validation=validation,
    )
    return {
        "baseline": {
            "run_id": baseline_training["run_id"],
            "report": dict(baseline_identity),
            "checkpoint_manifest_sha256": baseline_training["checkpoint"]["manifest_sha256"],
            "evaluation_manifest_sha256": baseline_validation["identity"]["manifest_sha256"],
            "prepared_manifest": {
                field: baseline_prepared[field]
                for field in (
                    "sha256",
                    "dataset_fingerprint",
                    "sequence_count",
                    "input_token_count",
                    "shard_count",
                )
            },
        },
        "current": {
            "run_id": training["run_id"],
            "checkpoint_manifest_sha256": training["checkpoint"]["manifest_sha256"],
            "evaluation_manifest_sha256": validation["identity"]["manifest_sha256"],
            "prepared_manifest": {
                field: current_prepared[field]
                for field in (
                    "sha256",
                    "dataset_fingerprint",
                    "sequence_count",
                    "input_token_count",
                    "shard_count",
                )
            },
        },
        "comparability": {
            "same_prepared_manifest_sha256": True,
            "same_dataset_fingerprint": True,
            "same_sequence_input_token_and_shard_counts": True,
            "same_role_predicted_token_counts": True,
        },
        "roles": role_comparisons,
        "teacher_gap_closed_fraction": _metric_change(
            baseline=baseline_gap,
            current=current_gap,
            lower_is_better=False,
        ),
        "cross_version_history": cross_version_history,
    }


def _validated_cross_version_history(
    comparison: Mapping[str, Any],
) -> list[tuple[str, float, float]]:
    raw_history = comparison.get("cross_version_history")
    if not isinstance(raw_history, list) or not raw_history:
        raise ValueError("baseline comparison has no cross-version history")
    history: list[tuple[str, float, float]] = []
    for index, raw_row in enumerate(raw_history):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"cross-version history row {index} is not an object")
        run_id = raw_row.get("run_id")
        candidate_nll = _finite(raw_row.get("candidate_mean_nll"))
        gap_closed = _finite(raw_row.get("teacher_gap_closed_fraction"))
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"cross-version history row {index} has no run_id")
        if candidate_nll is None or candidate_nll < 0:
            raise ValueError(f"cross-version history row {index} has invalid candidate mean NLL")
        if gap_closed is None:
            raise ValueError(f"cross-version history row {index} has invalid teacher-gap-closed")
        history.append((run_id, candidate_nll, gap_closed))
    run_ids = [run_id for run_id, _, _ in history]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("cross-version history contains duplicate run IDs")
    return history


def _make_charts(
    output_dir: Path,
    training: Mapping[str, Any],
    training_raw: Mapping[str, Any],
    validation: Mapping[str, Any],
    baseline_comparison: Mapping[str, Any] | None = None,
) -> list[Path]:
    charts = output_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    metrics = training_raw["metrics"]
    telemetry = training_raw["telemetry"]
    paths: list[Path] = []

    def points(rows: Iterable[Mapping[str, Any]], x: str, y: str) -> list[tuple[float, float]]:
        result = []
        for row in rows:
            left = _finite(row.get(x))
            right = _finite(row.get(y))
            if left is not None and right is not None:
                result.append((left, right))
        return result

    path = charts / "training_loss.svg"
    loss_series = {
        "total": points(metrics, "tokens", "loss"),
        "NTP": points(metrics, "tokens", "ntp"),
        "teacher KD": points(metrics, "tokens", "teacher_kd"),
        "anchor KL": points(metrics, "tokens", "anchor_kl"),
        "hidden alignment": points(metrics, "tokens", "hidden_alignment"),
    }
    if any(_finite(row.get("mtp")) is not None for row in metrics):
        loss_series["MTP"] = points(metrics, "tokens", "mtp")
    _write_line_chart(
        path,
        title=f"{training['run_id']} training loss components",
        series=loss_series,
        x_label="Committed training tokens",
        y_label="Loss",
    )
    paths.append(path)
    if any(_finite(row.get("mtp")) is not None for row in metrics):
        window = min(50, len(metrics))

        def rolling(key: str) -> list[tuple[float, float]]:
            values: list[tuple[float, float]] = []
            running = 0.0
            queue: list[float] = []
            for row in metrics:
                value = _finite(row.get(key))
                token = _finite(row.get("tokens"))
                if value is None or token is None:
                    continue
                queue.append(value)
                running += value
                if len(queue) > window:
                    running -= queue.pop(0)
                if len(queue) == window:
                    values.append((token, running / window))
            return values

        path = charts / "training_loss_smoothed.svg"
        _write_line_chart(
            path,
            title=f"{training['run_id']} loss components ({window}-step moving mean)",
            series={
                "total": rolling("loss"),
                "NTP": rolling("ntp"),
                "teacher KD": rolling("teacher_kd"),
                "MTP": rolling("mtp"),
                "anchor KL": rolling("anchor_kl"),
            },
            x_label="Committed training tokens",
            y_label="Moving-mean loss",
        )
        paths.append(path)
        source_analysis = training.get("source_conditioned_analysis")
        if isinstance(source_analysis, Mapping):
            regressions = source_analysis.get("regressions")
            loss_regression = regressions.get("loss") if isinstance(regressions, Mapping) else None
            adjusted_rows = (
                loss_regression.get("source_adjusted_series")
                if isinstance(loss_regression, Mapping)
                else None
            )
            if isinstance(adjusted_rows, list) and len(adjusted_rows) >= window:
                adjusted_points: list[tuple[float, float]] = []
                queue: list[float] = []
                running = 0.0
                for row in adjusted_rows:
                    token = _finite(row.get("tokens"))
                    value = _finite(row.get("value"))
                    if token is None or value is None:
                        continue
                    queue.append(value)
                    running += value
                    if len(queue) > window:
                        running -= queue.pop(0)
                    if len(queue) == window:
                        adjusted_points.append((token, running / window))
                path = charts / "training_source_adjusted_loss.svg"
                _write_line_chart(
                    path,
                    title=(
                        f"{training['run_id']} raw vs source-adjusted total loss "
                        f"({window}-step moving mean)"
                    ),
                    series={
                        "raw total": rolling("loss"),
                        "source-adjusted total": adjusted_points,
                    },
                    x_label="Committed training tokens",
                    y_label="Moving-mean total loss",
                )
                paths.append(path)
            source_means = source_analysis.get("source_weighted_means")
            if isinstance(source_means, Mapping) and source_means:
                names = list(source_means)
                path = charts / "training_loss_by_source.svg"
                _write_grouped_bar_chart(
                    path,
                    title=f"{training['run_id']} post-warmup primary loss by source",
                    groups=names,
                    series={"total loss": [float(source_means[name]["loss"]) for name in names]},
                    y_label="Mean training loss",
                )
                paths.append(path)
    path = charts / "gradient_norm.svg"
    _write_line_chart(
        path,
        title="Gradient norm before clipping",
        series={"grad norm": points(metrics, "tokens", "grad_norm")},
        x_label="Committed training tokens",
        y_label="Global gradient norm",
        horizontal_lines=[(float(training["grad_clip_threshold"]), "clip threshold")],
    )
    paths.append(path)
    path = charts / "learning_rate.svg"
    _write_line_chart(
        path,
        title="Learning-rate schedule",
        series={
            "adapters": points(metrics, "tokens", "lr/adapters"),
            "scale": points(metrics, "tokens", "lr/scale"),
        },
        x_label="Committed training tokens",
        y_label="Learning rate",
    )
    paths.append(path)
    path = charts / "throughput.svg"
    _write_line_chart(
        path,
        title="Training throughput",
        series={
            "compute tok/s": points(telemetry, "tokens", "compute_tokens_per_second"),
            "wall tok/s": points(telemetry, "tokens", "wall_clock_tokens_per_second"),
            "compute EMA": points(telemetry, "tokens", "compute_tokens_per_second_ema"),
            "wall EMA": points(telemetry, "tokens", "wall_clock_tokens_per_second_ema"),
        },
        x_label="Committed training tokens",
        y_label="Tokens / second",
    )
    paths.append(path)
    path = charts / "gpu_memory.svg"
    _write_line_chart(
        path,
        title="Training GPU memory",
        series={
            "allocated": points(telemetry, "tokens", "gpu_allocated_gib"),
            "reserved": points(telemetry, "tokens", "gpu_reserved_gib"),
            "peak allocated": points(telemetry, "tokens", "gpu_peak_allocated_gib"),
            "peak reserved": points(telemetry, "tokens", "gpu_peak_reserved_gib"),
        },
        x_label="Committed training tokens",
        y_label="GiB",
    )
    paths.append(path)
    checkpoint_events = training_raw["checkpoint_events"]
    path = charts / "checkpoint_duration.svg"
    _write_line_chart(
        path,
        title="Checkpoint write duration",
        series={"checkpoint": points(checkpoint_events, "step", "duration_seconds")},
        x_label="Optimizer step",
        y_label="Seconds",
    )
    paths.append(path)
    role_order = ["candidate", "shared", "teacher"]
    path = charts / "validation_nll.svg"
    _write_grouped_bar_chart(
        path,
        title="Final validation mean NLL",
        groups=role_order,
        series={"mean NLL": [float(validation["roles"][role]["mean_nll"]) for role in role_order]},
        y_label="Mean NLL (lower is better)",
    )
    paths.append(path)
    path = charts / "validation_perplexity.svg"
    _write_grouped_bar_chart(
        path,
        title="Final validation perplexity",
        groups=role_order,
        series={
            "perplexity": [float(validation["roles"][role]["perplexity"]) for role in role_order]
        },
        y_label="Perplexity (lower is better)",
    )
    paths.append(path)
    path = charts / "validation_role_throughput.svg"
    _write_grouped_bar_chart(
        path,
        title="Final validation throughput by role",
        groups=role_order,
        series={
            "predicted tok/s": [
                float(validation["roles"][role]["timing"]["predicted_tokens_per_second"])
                for role in role_order
            ]
        },
        y_label="Predicted tokens / second",
    )
    paths.append(path)
    source_names = [row["source"] for row in validation["sources"]]
    path = charts / "validation_source_nll.svg"
    _write_grouped_bar_chart(
        path,
        title="Validation NLL by source",
        groups=source_names,
        series={
            role: [float(row["roles"][role]["mean_nll"]) for row in validation["sources"]]
            for role in role_order
        },
        y_label="Mean NLL (token-weighted)",
    )
    paths.append(path)
    path = charts / "validation_source_tokens.svg"
    _write_grouped_bar_chart(
        path,
        title="Validation input-token composition",
        groups=source_names,
        series={"input tokens": [float(row["input_tokens"]) for row in validation["sources"]]},
        y_label="Input tokens",
    )
    paths.append(path)
    path = charts / "validation_shard_nll.svg"
    _write_line_chart(
        path,
        title="Validation NLL across authenticated shards",
        series={
            role: [
                (float(row["index"]), float(row["roles"][role]["mean_nll"]))
                for row in validation["shards"]
            ]
            for role in role_order
        },
        x_label="Prepared-manifest shard index",
        y_label="Mean NLL",
    )
    paths.append(path)
    gpu_sample = validation.get("runtime_gpu_sample")
    if gpu_sample:
        sample_rows = gpu_sample["samples"]
        path = charts / "validation_gpu_power.svg"
        power_limit = gpu_sample["statistics"].get("power_limit_w", {}).get("mean")
        _write_line_chart(
            path,
            title="Validation GPU power sample",
            series={"power draw": points(sample_rows, "sample", "power_draw_w")},
            x_label="One-second sample index",
            y_label="Watts",
            horizontal_lines=(
                [(float(power_limit), "power limit")] if power_limit is not None else []
            ),
        )
        paths.append(path)
        path = charts / "validation_gpu_utilization.svg"
        _write_line_chart(
            path,
            title="Validation GPU utilization sample",
            series={
                "GPU": points(sample_rows, "sample", "gpu_utilization_percent"),
                "memory": points(sample_rows, "sample", "memory_utilization_percent"),
            },
            x_label="One-second sample index",
            y_label="Utilization (%)",
        )
        paths.append(path)
    if baseline_comparison is not None:
        history = _validated_cross_version_history(baseline_comparison)
        run_ids = [run_id for run_id, _, _ in history]
        path = charts / "validation_candidate_nll_history.svg"
        _write_grouped_bar_chart(
            path,
            title="Candidate validation NLL across authenticated versions",
            groups=run_ids,
            series={"candidate mean NLL": [candidate_nll for _, candidate_nll, _ in history]},
            y_label="Mean NLL (lower is better)",
        )
        paths.append(path)
        path = charts / "validation_teacher_gap_closed_history.svg"
        _write_grouped_bar_chart(
            path,
            title="Teacher gap closed across authenticated versions",
            groups=run_ids,
            series={"teacher gap closed": [gap_closed * 100.0 for _, _, gap_closed in history]},
            y_label="Teacher gap closed (%)",
        )
        paths.append(path)
    return paths


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join((head, rule, *body))


def _fmt_signed(value: Any, digits: int) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{number:+.{digits}f}"


def _fmt_signed_percent(value: Any, digits: int = 3) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{number:+.{digits}f}%"


def _build_baseline_comparison_section(comparison: Mapping[str, Any] | None, *, zh_cn: bool) -> str:
    if comparison is None:
        return ""
    baseline = comparison["baseline"]
    current = comparison["current"]
    role_rows = []
    for role in ("candidate", "shared", "teacher"):
        values = comparison["roles"][role]
        nll = values["mean_nll"]
        perplexity = values["perplexity"]
        role_rows.append(
            (
                role,
                _fmt(nll["baseline"], 6),
                _fmt(nll["current"], 6),
                _fmt_signed(nll["absolute_change"], 6),
                _fmt_signed_percent(nll["relative_change_percent"]),
                _fmt(perplexity["baseline"], 4),
                _fmt(perplexity["current"], 4),
                _fmt_signed(perplexity["absolute_change"], 4),
                _fmt_signed_percent(perplexity["relative_change_percent"]),
            )
        )
    gap = comparison["teacher_gap_closed_fraction"]
    gap_rows = [
        (
            "teacher gap closed",
            _fmt(gap["baseline"], 6),
            _fmt(gap["current"], 6),
            _fmt_signed(gap["absolute_change"], 6),
            _fmt_signed_percent(gap["relative_change_percent"]),
        )
    ]
    history = _validated_cross_version_history(comparison)
    history_rows = [
        (
            run_id,
            _fmt(candidate_nll, 6),
            f"{gap_closed * 100.0:.3f}%",
        )
        for run_id, candidate_nll, gap_closed in history
    ]
    report_identity = baseline["report"]
    if zh_cn:
        return f"""
## 已认证的 baseline 对照

Baseline 为 `{baseline["run_id"]}`, 当前结果为 `{current["run_id"]}`。对照只在以下条件全部
严格相等后才生成: validation prepared-manifest SHA256、dataset fingerprint、序列/输入
token/shard 数, 以及 candidate/shared/teacher 的预测 token 数。因此这里的绝对和相对变化
来自同一 held-out 任务与同一组 label, 不是跨数据集比较。

{
            _markdown_table(
                (
                    "角色",
                    "baseline NLL",
                    "当前 NLL",
                    "NLL 绝对变化",
                    "NLL 相对变化",
                    "baseline PPL",
                    "当前 PPL",
                    "PPL 绝对变化",
                    "PPL 相对变化",
                ),
                role_rows,
            )
        }

{
            _markdown_table(
                ("指标", "baseline", "当前", "绝对变化", "相对变化"),
                gap_rows,
            )
        }

### 同一 held-out 集的跨版本历史

下表与两张图仅使用已认证 lineage 中记录的版本; 每一行都对应完全相同的 prepared
validation manifest 和三角色预测 token 口径。

{
            _markdown_table(
                ("run_id", "candidate mean NLL", "teacher gap closed"),
                history_rows,
            )
        }

![跨版本 candidate NLL](charts/validation_candidate_nll_history.svg)

![跨版本 teacher gap closed](charts/validation_teacher_gap_closed_history.svg)

NLL/困惑度的负变化表示改善; teacher-gap-closed 的正变化表示改善。Baseline 的
`summary.json` 由相邻 `MANIFEST.json` 清单认证, 该 manifest 再由 `COMPLETE` 认证:

- Baseline summary SHA256: `{report_identity["summary_sha256"]}`
- Baseline report manifest SHA256: `{report_identity["bundle_manifest_sha256"]}`
- Baseline checkpoint manifest SHA256: `{baseline["checkpoint_manifest_sha256"]}`
- Baseline evaluation manifest SHA256: `{baseline["evaluation_manifest_sha256"]}`
"""
    return f"""
## Authenticated baseline comparison

The baseline is `{baseline["run_id"]}` and the current result is `{current["run_id"]}`.  This
comparison is emitted only after exact matches on validation prepared-manifest SHA256, dataset
fingerprint, sequence/input-token/shard counts, and predicted-token counts for all three roles.
The absolute and relative changes therefore use the same held-out task and labels.

{
        _markdown_table(
            (
                "role",
                "baseline NLL",
                "current NLL",
                "NLL absolute Δ",
                "NLL relative Δ",
                "baseline PPL",
                "current PPL",
                "PPL absolute Δ",
                "PPL relative Δ",
            ),
            role_rows,
        )
    }

{
        _markdown_table(
            ("metric", "baseline", "current", "absolute Δ", "relative Δ"),
            gap_rows,
        )
    }

### Cross-version history on the same held-out set

The table and figures include only versions carried by the authenticated lineage.  Every row uses
the identical prepared validation manifest and predicted-token accounting for all three roles.

{
        _markdown_table(
            ("run_id", "candidate mean NLL", "teacher gap closed"),
            history_rows,
        )
    }

![Candidate NLL across versions](charts/validation_candidate_nll_history.svg)

![Teacher gap closed across versions](charts/validation_teacher_gap_closed_history.svg)

Negative NLL/perplexity changes are improvements; a positive teacher-gap-closed change is an
improvement.  The baseline `summary.json` is inventoried by its adjacent `MANIFEST.json`, which is
itself authenticated by `COMPLETE`:

- Baseline summary SHA256: `{report_identity["summary_sha256"]}`
- Baseline report manifest SHA256: `{report_identity["bundle_manifest_sha256"]}`
- Baseline checkpoint manifest SHA256: `{baseline["checkpoint_manifest_sha256"]}`
- Baseline evaluation manifest SHA256: `{baseline["evaluation_manifest_sha256"]}`
"""


def _build_training_lifecycle_section(training: Mapping[str, Any], *, zh_cn: bool) -> str:
    lifecycle = training.get("lifecycle")
    if not isinstance(lifecycle, Mapping):
        return ""
    rows = []
    raw_events = lifecycle.get("events")
    if isinstance(raw_events, Sequence) and not isinstance(raw_events, (str, bytes)):
        for event in raw_events:
            if not isinstance(event, Mapping):
                continue
            detail = ""
            if isinstance(event.get("checkpoint"), str):
                detail = f"checkpoint={Path(str(event['checkpoint'])).name}"
            elif isinstance(event.get("fork_from"), str):
                detail = f"fork={Path(str(event['fork_from'])).name}"
            elif event.get("reason") is not None:
                detail = f"reason={event['reason']}"
            rows.append(
                (
                    event.get("event", "n/a"),
                    event.get("session_id", "n/a"),
                    event.get("step", "n/a"),
                    (
                        f"{int(event['tokens']):,}"
                        if isinstance(event.get("tokens"), int)
                        else "n/a"
                    ),
                    event.get("timestamp_utc", "n/a"),
                    detail or "n/a",
                )
            )
    elapsed_hours = float(lifecycle["elapsed_wall_seconds"]) / 3600.0
    canonical_hours = float(lifecycle["canonical_committed_step_wall_seconds"]) / 3600.0
    outside_hours = float(lifecycle["outside_canonical_committed_step_wall_seconds"]) / 3600.0
    session_count = int(lifecycle["session_count"])
    resume_count = int(lifecycle["resume_count"])
    graceful_count = int(lifecycle["graceful_stop_count"])
    unterminated_count = int(lifecycle["sessions_without_terminal_event_count"])
    timeline = _markdown_table(
        (
            "事件" if zh_cn else "event",
            "session",
            "step",
            "token" if zh_cn else "tokens",
            "UTC",
            "详情" if zh_cn else "detail",
        ),
        rows,
    )
    if zh_cn:
        return f"""
### 恢复与墙钟口径

- 从首次 `train_start` 到 `train_complete` 的 elapsed wall 为
  **{elapsed_hours:.3f} h**, 对应 **{_fmt(training["end_to_end_tokens_per_second"], 1)}
  tok/s**。这个分母有意包含跨 session 停顿、重新初始化及回滚后的重放工作。
- 最终 canonical 日志中已提交 optimizer step 的 active wall 合计
  **{canonical_hours:.3f} h**, 对应
  **{_fmt(training["aggregate_wall_step_tokens_per_second"], 1)} tok/s**。
- 两者相差 **{outside_hours:.3f} h**; 该差值还包含 checkpoint 写盘、模型构建/preflight
  等开销, 不能全部解释为 GPU 空闲。
- 生命周期记录 {session_count} 个 session、{resume_count} 次 resume、
  {graceful_count} 次 graceful stop; {unterminated_count} 个 session 没有终止事件。

{timeline}
"""
    return f"""
### Resume and wall-clock accounting

- Elapsed wall from the first `train_start` through `train_complete` was
  **{elapsed_hours:.3f} h**, or **{_fmt(training["end_to_end_tokens_per_second"], 1)}
  tok/s**.  This denominator intentionally includes inter-session pauses, reinitialization,
  and work replayed after rollback.
- Canonical active wall for committed optimizer steps was **{canonical_hours:.3f} h**, or
  **{_fmt(training["aggregate_wall_step_tokens_per_second"], 1)} tok/s**.
- The **{outside_hours:.3f} h** difference also contains checkpoint writes, model
  construction/preflight, and other overhead; it must not be read as GPU idle time alone.
- The lifecycle records {session_count} sessions, {resume_count} resumes,
  {graceful_count} graceful stops, and {unterminated_count} sessions without a terminal event.

{timeline}
"""


def _build_report(
    training: Mapping[str, Any],
    validation: Mapping[str, Any],
    baseline_comparison: Mapping[str, Any] | None = None,
) -> str:
    roles = validation["roles"]
    acceptance = validation["acceptance"]
    first = training["first_50_weighted"]
    last = training["last_50_weighted"]
    governance = validation["data_governance"]
    runtime = validation.get("runtime", {})
    inference_lineage = validation.get("checkpoint_inference_lineage", {})
    gpu_sample = validation.get("runtime_gpu_sample")
    role_rows = [
        (
            role,
            f"{int(roles[role]['predicted_tokens']):,}",
            _fmt(roles[role]["mean_nll"], 6),
            _fmt(roles[role]["perplexity"], 4),
            _fmt(roles[role]["timing"]["predicted_tokens_per_second"], 1),
        )
        for role in ("candidate", "shared", "teacher")
    ]
    source_rows = []
    for source in validation["sources"]:
        source_rows.append(
            (
                source["source"],
                source["shards"],
                f"{source['input_tokens']:,}",
                _fmt(source["roles"]["candidate"]["mean_nll"], 5),
                _fmt(source["roles"]["shared"]["mean_nll"], 5),
                _fmt(source["roles"]["teacher"]["mean_nll"], 5),
                _fmt(source["teacher_gap_closed_fraction"], 4),
            )
        )
    pending = ", ".join(governance["pending_audits"]) or "none"
    gate = acceptance.get("dense_gap_gate_pass")
    gate_text = "PASS" if gate is True else "FAIL" if gate is False else "not computed"
    gap = acceptance.get("teacher_gap_closed_fraction")
    ntp_change = float(last["ntp"]) - float(first["ntp"])
    kd_change = float(last["teacher_kd"]) - float(first["teacher_kd"])
    anchor_change = float(last["anchor_kl"]) - float(first["anchor_kl"])
    run_id = str(training["run_id"])
    config = training["config"]
    methodology_errata = _methodology_errata_for_run(run_id)
    methodology_erratum_section = _methodology_erratum_section(run_id, zh_cn=False)
    baseline_section = _build_baseline_comparison_section(baseline_comparison, zh_cn=False)
    lifecycle_section = _build_training_lifecycle_section(training, zh_cn=False)
    if config["lr_schedule"] == "warmup-stable-decay":
        decay_tokens = int(config["decay_tokens"])
        schedule_description = (
            f"linear warmup for {config['warmup_tokens']:,} tokens, a stable plateau through "
            f"{config['max_tokens'] - decay_tokens:,} tokens, then a {decay_tokens:,}-token "
            f"cosine decay to {config['min_lr_ratio']:.3f}× peak LR"
        )
    else:
        schedule_description = (
            f"linear warmup for {config['warmup_tokens']:,} tokens followed by full-run cosine "
            f"decay to {config['min_lr_ratio']:.3f}× peak LR at "
            f"{config['max_tokens']:,} tokens"
        )
    if config["mtp_present"] and config["mtp_weight"] > 0:
        if methodology_errata:
            mtp_description = (
                f"The immutable run logged an MTP objective at weight **{config['mtp_weight']}**. "
                "Its 15 checkpoint-native parameters remained frozen and outside the optimizer, "
                "while its loss propagated through the student hidden states. Because of the "
                "one-token RoPE offset disclosed above, this describes the path that actually "
                "ran and is not a claim of fully native-aligned Qwen3.5 MTP execution."
            )
        else:
            mtp_description = (
                f"The immutable run enabled the native Qwen3.5 MTP objective at weight "
                f"**{config['mtp_weight']}**.  Its source head remained frozen and outside the "
                "optimizer, while its loss propagated through the student hidden states."
            )
        smoothed_loss_figure = (
            "\n![Training losses, 50-step moving mean](charts/training_loss_smoothed.svg)\n"
        )
    else:
        mtp_description = (
            f"The immutable run had **MTP=0** (`mtp` was "
            f"{'present' if config['mtp_present'] else 'absent'} in the resolved loss mapping), "
            "so it must not be described as MTP-trained."
        )
        smoothed_loss_figure = ""
    cooldown_start = config.get("quality_cooldown_start_tokens")
    cooldown_note = (
        f" The data stream also switches to the authenticated quality-cooldown corpus at "
        f"{int(cooldown_start):,} tokens; any loss discontinuity at that exact boundary is "
        "confounded with the data change and cannot be attributed to LR alone."
        if cooldown_start is not None
        else ""
    )
    source_analysis = training.get("source_conditioned_analysis")
    if isinstance(source_analysis, Mapping):
        regressions = source_analysis["regressions"]
        regression_rows = [
            (
                metric,
                _fmt(result["slope_per_100m_tokens"], 5),
                (f"[{_fmt(result['confidence_95'][0], 5)}, {_fmt(result['confidence_95'][1], 5)}]"),
                _fmt(result["r_squared"], 4),
                _fmt(result["raw_standard_deviation"], 4),
                _fmt(result["residual_standard_deviation"], 4),
            )
            for metric, result in regressions.items()
        ]
        source_loss_rows = [
            (
                source,
                _fmt(values["effective_steps"], 1),
                _fmt(values.get("loss"), 4),
                _fmt(values.get("ntp"), 4),
                _fmt(values.get("teacher_kd"), 4),
                _fmt(values.get("mtp"), 4),
            )
            for source, values in source_analysis["source_weighted_means"].items()
        ]
        source_analysis_section = f"""
## Source-conditioned training analysis

Deterministic cursor replay matched every logged data phase and reconstructed the source mixture
of all {source_analysis["steps"]:,} optimizer batches.  In the post-warmup primary phase,
pure-source batches accounted for
{source_analysis["post_warmup_primary_pure_source_batches"]:,} /
{source_analysis["post_warmup_primary_rows"]:,}
({source_analysis["post_warmup_primary_pure_source_batch_fraction"] * 100:.2f}%).
This fixed analysis window includes both the stable and cosine-decay portions of primary data,
but excludes warmup and quality cooldown.  The table below fits
each metric on that phase using source fixed effects plus committed tokens; uncertainty is a
Newey-West HAC interval with lag 50.

{_markdown_table(("metric", "slope / 100M", "HAC 95% CI", "R²", "raw SD", "residual SD"), regression_rows)}

{_markdown_table(("source", "effective steps", "total", "NTP", "KD", "MTP"), source_loss_rows)}

For total loss, source composition explains
{float(regressions["loss"]["r_squared"]) * 100:.2f}% of raw variance, while the controlled trend is
{float(regressions["loss"]["slope_per_100m_tokens"]):+.5f} per 100M tokens.  Therefore the raw
near-plateau is not evidence of zero learning: source-dependent difficulty and the mixed objectives
hide a statistically negative within-source trend.  This remains training evidence, not a substitute
for held-out NLL.

![Raw vs source-adjusted training loss](charts/training_source_adjusted_loss.svg)

![Post-warmup primary training loss by source](charts/training_loss_by_source.svg)
"""
    else:
        source_analysis_section = ""
    checkpoint_path = Path(str(training["checkpoint"]["path"]))
    run_dir = checkpoint_path.parent
    evaluation_dir = Path(str(validation["identity"]["path"]))
    prepared_path = Path(str(validation["prepared_manifest"]["path"]))
    if gpu_sample:
        gpu_stats = gpu_sample["statistics"]
        power = gpu_stats["power_draw_w"]
        power_limit = gpu_stats["power_limit_w"]
        gpu_util = gpu_stats["gpu_utilization_percent"]
        memory_util = gpu_stats["memory_utilization_percent"]
        sm_clock = gpu_stats["sm_clock_mhz"]
        temperature = gpu_stats["temperature_c"]
        gpu_runtime_section = f"""
## Validation GPU runtime sample

- {gpu_sample["sample_count"]} one-second samples, all P-states: `{gpu_sample["pstates"]}`.
- Power: {power["mean"]:.2f} W mean, {power["min"]:.2f} to {power["max"]:.2f} W range;
  configured limit {power_limit["mean"]:.0f} W.
- GPU utilization: {gpu_util["mean"]:.2f}% mean, {gpu_util["min"]:.0f} to {gpu_util["max"]:.0f}% range;
  memory utilization: {memory_util["mean"]:.2f}% mean,
  {memory_util["min"]:.0f} to {memory_util["max"]:.0f}% range.
- SM clock: {sm_clock["mean"]:.1f} MHz mean, {sm_clock["min"]:.0f} to {sm_clock["max"]:.0f} MHz;
  temperature: {temperature["mean"]:.2f} °C mean,
  {temperature["min"]:.0f} to {temperature["max"]:.0f} °C.
- Raw sample SHA256: `{gpu_sample["sha256"]}`.  This was a read-only `nvidia-smi` sample;
  no CUPTI subscriber or profiler was attached.

![Validation GPU power](charts/validation_gpu_power.svg)

![Validation GPU utilization](charts/validation_gpu_utilization.svg)
"""
    else:
        gpu_runtime_section = ""
    return f"""# {run_id} final validation report

This bundle evaluates the immutable final checkpoint at step {training["steps"]:,}
({training["committed_tokens"]:,} committed training tokens) on the separately prepared,
authenticated validation corpus.  Evaluation is forward-only (`inference_mode`): no optimizer
state, backward pass, or parameter update is involved.

{methodology_erratum_section}\
## Executive result

{_markdown_table(("role", "predicted tokens", "mean NLL", "perplexity", "wall tok/s"), role_rows)}

- Teacher gap closed: **{_fmt(gap, 5)}**; configured dense acceptance gate: **{gate_text}**
  (threshold 0.10).
- Every role covered {validation["prepared_manifest"]["shard_count"]} / {validation["prepared_manifest"]["shard_count"]}
  shards and {validation["prepared_manifest"]["sequence_count"]:,} sequences.  Role token counts
  are equal, so role comparisons use identical labels.
- `candidate` is the trained transfer branch, `shared` disables that branch on the same backbone,
  and `teacher` is the frozen Qwen3.5-9B reference.

![Validation NLL](charts/validation_nll.svg)

![Validation perplexity](charts/validation_perplexity.svg)

![Validation role throughput](charts/validation_role_throughput.svg)

{baseline_section}

## Training trajectory

- Aggregate compute throughput: **{_fmt(training["aggregate_compute_tokens_per_second"], 1)} tok/s**;
  optimizer-step wall throughput: **{_fmt(training["aggregate_wall_step_tokens_per_second"], 1)} tok/s**;
  full train-start to final-checkpoint throughput: **{_fmt(training["end_to_end_tokens_per_second"], 1)} tok/s**.
- The first-start-to-final elapsed span was
  **{training["wall_duration_seconds"] / 3600:.3f} h**; its restart-aware accounting is
  separated below.  The run wrote {training["checkpoint_count"]} checkpoints.  Checkpoint writes consumed
  {training["checkpoint_duration_seconds_total"] / 60:.2f} min in total
  (mean {_fmt(training["checkpoint_duration_seconds_mean"], 2)} s, max
  {_fmt(training["checkpoint_duration_seconds_max"], 2)} s).
- Data wait was {training["data_wait_seconds"]:.3f} s, or
  {training["data_wait_fraction_of_compute"] * 100:.4f}% of measured compute-step time.
- Peak GPU memory was {training["gpu_peak_allocated_gib"]:.3f} GiB allocated /
  {training["gpu_peak_reserved_gib"]:.3f} GiB reserved.
- Pre-clip gradient norm exceeded {training["grad_clip_threshold"]:.3f} on
  {training["grad_norm_over_clip_threshold_steps"]} / {training["steps"]} steps
  ({training["grad_norm_over_clip_threshold_fraction"] * 100:.2f}%).
- Hidden-alignment batches: {training["alignment_steps"]} / {training["steps"]}; ordinary batches:
  {training["ordinary_steps"]} / {training["steps"]}.

{lifecycle_section}

Token-weighted first-50 to last-50 changes: NTP {first["ntp"]:.5f} → {last["ntp"]:.5f}
({ntp_change:+.5f}), teacher KD {first["teacher_kd"]:.5f} → {last["teacher_kd"]:.5f}
({kd_change:+.5f}), and anchor KL {first["anchor_kl"]:.5f} → {last["anchor_kl"]:.5f}
({anchor_change:+.5f}).  These are noisy training-batch metrics, not substitutes for held-out NLL.

![Training losses](charts/training_loss.svg)
{smoothed_loss_figure}

![Gradient norm](charts/gradient_norm.svg)

![Learning rate](charts/learning_rate.svg)

![Throughput](charts/throughput.svg)

![GPU memory](charts/gpu_memory.svg)

![Checkpoint duration](charts/checkpoint_duration.svg)

{gpu_runtime_section}

{source_analysis_section}

## Validation by data source

{_markdown_table(("source", "shards", "input tokens", "candidate NLL", "shared NLL", "teacher NLL", "gap closed"), source_rows)}

Source means are weighted by predicted tokens, not averaged across shards.

![Source NLL](charts/validation_source_nll.svg)

![Source token composition](charts/validation_source_tokens.svg)

![Shard NLL](charts/validation_shard_nll.svg)

The shard chart preserves prepared-manifest order; `summary.json` contains exact per-shard values,
source IDs, token counts, and candidate-over-shared deltas for downstream analysis.

## Interpretation and limits

- The final held-out role comparison is the quality evidence for this `{run_id}` checkpoint.  The training
  loss curve alone cannot establish underfitting because batches and objective mixtures vary by step.
  A low teacher-gap-closed fraction, together with continued held-out gains in a future checkpoint
  sweep, would support increasing data/token budget; this single final evaluation cannot locate the
  optimal stopping point because the run did not perform held-out validation at multiple checkpoints.
- {mtp_description}
- The immutable schedule used {schedule_description}.{cooldown_note}
- Data status is **research_only={str(governance["research_only"]).lower()}** and
  **ready_for_training={str(governance["ready_for_training"]).lower()}**.  Pending audits:
  `{pending}`.  Until those audits are complete, results are research evidence and must not be
  presented as production-ready or contamination-free.

## Reproducibility and integrity

- Final checkpoint: `{training["checkpoint"]["path"]}`
- Checkpoint manifest SHA256: `{training["checkpoint"]["manifest_sha256"]}`; all
  {training["checkpoint"]["inventory_file_count"]} inventoried files were re-hashed successfully.
- Evaluation: `{validation["identity"]["path"]}`
- Evaluation manifest SHA256: `{validation["identity"]["manifest_sha256"]}`
- Evaluation PLAN SHA256: `{validation["identity"]["plan_sha256"]}`
- Validation prepared-manifest SHA256: `{validation["prepared_manifest"]["sha256"]}`
- Validation dataset fingerprint: `{validation["prepared_manifest"]["dataset_fingerprint"]}`
- Evaluation runtime: torch `{runtime.get("torch_version")}`, CUDA `{runtime.get("cuda_version")}`,
  `{runtime.get("device_name")}` (`{runtime.get("device")}`, compute capability
  `{runtime.get("compute_capability")}`), dtype `{runtime.get("dtype")}`,
  `FLA_TILELANG={runtime.get("fla_tilelang")}`,
  `CUDA_HOME={runtime.get("cuda_home")}`.  `FLA_TILELANG=0` selects the production Triton path
  rather than the experimental TileLang path.
- Checkpoint load mode: `{inference_lineage.get("mode")}`.  Saved/current exact-training
  fingerprints match: `{inference_lineage.get("exact_training_fingerprint_match")}`;
  saved/current source trees match: `{inference_lineage.get("source_tree_match")}`.  The saved
  source tree is `{inference_lineage.get("saved_source_tree_sha256")}` and the evaluation source
  tree is `{inference_lineage.get("current_source_tree_sha256")}`.  This is authenticated
  forward-only compatibility and **must not** be described as exact-resume compatibility.

Canonical reproduction uses an absolute checkpoint path.  The checkpoint resolver now treats a
bare `step-*` name as relative to `checkpoint.output_dir`, while a relative argument containing a
directory is relative to the current working directory.  This avoids the historical
`runs/<run>/runs/<run>/step-*` double-prefix trap.

```bash
FLA_TILELANG=0 bash scripts/with_cuda_toolchain.sh .venv/bin/python -m twen evaluate nll \\
  --config {run_dir / "resolved_config.yaml"} \\
  --checkpoint {checkpoint_path} \\
  --prepared-manifest {prepared_path} \\
  --prepared-manifest-sha256 {validation["prepared_manifest"]["sha256"]} \\
  --output {evaluation_dir} \\
  --role candidate --role shared --role teacher --batch-size 1 --device cuda:0
```

`MANIFEST.json` authenticates every report payload and figure; `COMPLETE` authenticates that
manifest.  All figures are standalone SVG generated with Python's standard library.
"""


def _build_report_zh_cn(
    training: Mapping[str, Any],
    validation: Mapping[str, Any],
    baseline_comparison: Mapping[str, Any] | None = None,
) -> str:
    roles = validation["roles"]
    acceptance = validation["acceptance"]
    governance = validation["data_governance"]
    runtime = validation.get("runtime", {})
    lineage = validation.get("checkpoint_inference_lineage", {})
    first = training["first_50_weighted"]
    last = training["last_50_weighted"]
    run_id = str(training["run_id"])
    config = training["config"]
    methodology_errata = _methodology_errata_for_run(run_id)
    methodology_erratum_section = _methodology_erratum_section(run_id, zh_cn=True)
    baseline_section = _build_baseline_comparison_section(baseline_comparison, zh_cn=True)
    lifecycle_section = _build_training_lifecycle_section(training, zh_cn=True)
    gap = acceptance.get("teacher_gap_closed_fraction")
    gate = acceptance.get("dense_gap_gate_pass")
    gate_text = "通过" if gate is True else "未通过" if gate is False else "未计算"
    role_rows = [
        (
            role,
            f"{int(roles[role]['predicted_tokens']):,}",
            _fmt(roles[role]["mean_nll"], 6),
            _fmt(roles[role]["perplexity"], 4),
            _fmt(roles[role]["timing"]["predicted_tokens_per_second"], 1),
        )
        for role in ("candidate", "shared", "teacher")
    ]
    source_rows = [
        (
            source["source"],
            source["shards"],
            f"{source['input_tokens']:,}",
            _fmt(source["roles"]["candidate"]["mean_nll"], 5),
            _fmt(source["roles"]["shared"]["mean_nll"], 5),
            _fmt(source["roles"]["teacher"]["mean_nll"], 5),
            _fmt(source["teacher_gap_closed_fraction"], 4),
        )
        for source in validation["sources"]
    ]
    if config["lr_schedule"] == "warmup-stable-decay":
        decay_tokens = int(config["decay_tokens"])
        schedule_text = (
            f"{config['warmup_tokens']:,} token 线性 warmup, 随后稳定到 "
            f"{config['max_tokens'] - decay_tokens:,} token, 再用最后 "
            f"{decay_tokens:,} token 余弦退火到峰值的 {config['min_lr_ratio']:.3f} 倍"
        )
    else:
        schedule_text = (
            f"{config['warmup_tokens']:,} token 线性 warmup, 随后在剩余预算内全程余弦退火, "
            f"到 {config['max_tokens']:,} token 时降至峰值的 "
            f"{config['min_lr_ratio']:.3f} 倍"
        )
    if config["mtp_present"] and config["mtp_weight"] > 0:
        if methodology_errata:
            mtp_text = (
                f"本轮 loss 日志中明确启用了 MTP `{config['mtp_weight']}`; 15 张 checkpoint "
                "原生参数保持 frozen、不进入 optimizer, MTP loss 会经 student hidden state "
                "回传到可训练适配矩阵。但其 RoPE 位置存在上述一 token 错位, 因此这里只陈述"
                "实际执行路径, 不再把它表述为完全对齐的 Qwen3.5 原生 MTP forward。"
            )
        else:
            mtp_text = (
                f"本轮明确启用了 **Qwen3.5 原生 MTP**, loss 权重为 "
                f"**{config['mtp_weight']}**。MTP 源头的 15 张参数保持 frozen、不进入 optimizer, "
                "但 MTP loss 会经 student hidden state 回传到可训练适配矩阵。"
            )
        smoothed_loss_figure = (
            "\n![训练 loss 分量 (50-step 滑动平均)](charts/training_loss_smoothed.svg)\n"
        )
    else:
        mtp_text = (
            f"本轮为 **MTP=0**; `mtp` 字段在 resolved loss 配置中"
            f"{'存在但为零' if config['mtp_present'] else '不存在'}, 因此不能描述为 MTP 训练。"
        )
        smoothed_loss_figure = ""
    cooldown_start = config.get("quality_cooldown_start_tokens")
    cooldown_text = (
        f"数据流在 {int(cooldown_start):,} token 同时切换到已认证的 quality-cooldown 语料; "
        "该边界的 loss 跳变与数据分布变化完全混杂, 不能只归因于 LR。"
        if cooldown_start is not None
        else "本轮没有单独的 quality-cooldown 数据切换。"
    )
    checkpoint_path = Path(str(training["checkpoint"]["path"]))
    run_dir = checkpoint_path.parent
    evaluation_dir = Path(str(validation["identity"]["path"]))
    prepared_path = Path(str(validation["prepared_manifest"]["path"]))
    if lineage.get("exact_training_fingerprint_match") is True:
        compatibility_text = (
            "saved/current exact-training fingerprint 与 source tree 均匹配。此次操作仍然只是"
            " forward-only inference validation; 是否允许训练恢复继续由 checkpoint loader "
            "的完整合同判定。"
        )
    else:
        compatibility_text = (
            "本次仅通过 forward-only inference validation 的 lineage compatibility; "
            "source-tree 或训练 fingerprint 并非 exact-resume-compatible, "
            "绝不能据此继续旧 run 训练。"
        )
    source_analysis = training.get("source_conditioned_analysis")
    if isinstance(source_analysis, Mapping):
        regressions = source_analysis["regressions"]
        regression_rows = [
            (
                metric,
                _fmt(result["slope_per_100m_tokens"], 5),
                (f"[{_fmt(result['confidence_95'][0], 5)}, {_fmt(result['confidence_95'][1], 5)}]"),
                _fmt(result["r_squared"], 4),
                _fmt(result["raw_standard_deviation"], 4),
                _fmt(result["residual_standard_deviation"], 4),
            )
            for metric, result in regressions.items()
        ]
        source_loss_rows = [
            (
                source,
                _fmt(values["effective_steps"], 1),
                _fmt(values.get("loss"), 4),
                _fmt(values.get("ntp"), 4),
                _fmt(values.get("teacher_kd"), 4),
                _fmt(values.get("mtp"), 4),
            )
            for source, values in source_analysis["source_weighted_means"].items()
        ]
        source_analysis_section = f"""
## 按数据源校正的训练分析

使用不可变 manifest 和确定性 cursor 逐步重放后, 所有 data phase 均与日志一致。
warmup 后的 primary 阶段中,
{source_analysis["post_warmup_primary_pure_source_batches"]:,} /
{source_analysis["post_warmup_primary_rows"]:,} 个 optimizer batch
({source_analysis["post_warmup_primary_pure_source_batch_fraction"] * 100:.2f}%)
是单一数据源。这个固定窗口同时包含 primary 的 stable 与 cosine-decay 段, 排除 warmup
和 quality cooldown。
下表对该阶段每项指标拟合 `source mix 固定效应 + committed tokens`;
置信区间为 lag 50 的 Newey-West HAC 95% CI。

{_markdown_table(("指标", "每 100M slope", "HAC 95% CI", "R²", "raw SD", "残差 SD"), regression_rows)}

{_markdown_table(("数据源", "等效 step", "total", "NTP", "KD", "MTP"), source_loss_rows)}

total loss 的来源组成解释了 {float(regressions["loss"]["r_squared"]) * 100:.2f}% raw 方差;
控制来源后, 趋势为每 100M token
{float(regressions["loss"]["slope_per_100m_tokens"]):+.5f}。因此原始曲线接近平台并不等于
"完全没有学习": 不同语料难度和多目标混合遮住了显著为负的 within-source 趋势。
这仍是训练集证据, 不能替代 held-out NLL。

![Raw 与 source-adjusted loss](charts/training_source_adjusted_loss.svg)

![各来源 post-warmup primary loss](charts/training_loss_by_source.svg)
"""
    else:
        source_analysis_section = ""
    gpu_sample = validation.get("runtime_gpu_sample")
    if gpu_sample:
        gpu_stats = gpu_sample["statistics"]
        power = gpu_stats["power_draw_w"]
        limit = gpu_stats["power_limit_w"]
        gpu_util = gpu_stats["gpu_utilization_percent"]
        memory_util = gpu_stats["memory_utilization_percent"]
        temperature = gpu_stats["temperature_c"]
        power_ratio = power["mean"] / max(limit["mean"], 1.0)
        if power_ratio >= 0.95:
            power_interpretation = (
                'Validation 的平均功耗已接近或触及功耗墙, 不能再用"功耗没吃满"解释该阶段吞吐。'
            )
        else:
            power_interpretation = (
                f"Validation 平均只达到功耗上限的 {power_ratio * 100:.1f}%, "
                "因此该只读评测没有触及 600 W 功耗墙; 这不代表训练阶段也没有触及。"
            )
        gpu_section = f"""
## Validation 期间的 GPU 遥测

- 只读采集 {gpu_sample["sample_count"]} 个 1 秒样本, 全程 P-state 为
  `{gpu_sample["pstates"]}`; 没有挂载 profiler 或 CUPTI subscriber。
- 功耗平均 **{power["mean"]:.2f} W**, 范围 {power["min"]:.2f} 至
  {power["max"]:.2f} W, 功耗上限恒为 {limit["mean"]:.0f} W。
- GPU 利用率平均 {gpu_util["mean"]:.2f}% ({gpu_util["min"]:.0f} 至
  {gpu_util["max"]:.0f}%), 显存利用率平均 {memory_util["mean"]:.2f}%; 温度平均
  {temperature["mean"]:.2f} °C。
- {power_interpretation}
- 利用率会在 microbatch、词表 head/CE 与 shard 原子提交边界之间波动; 不能仅凭功耗值
  断言算子是否饱和。
- 原始 CSV SHA256: `{gpu_sample["sha256"]}`。

![Validation GPU 功耗](charts/validation_gpu_power.svg)

![Validation GPU 利用率](charts/validation_gpu_utilization.svg)
"""
    else:
        gpu_section = ""
    pending = ", ".join(governance["pending_audits"]) or "无"
    return f"""# {run_id} 最终 validation 报告

本报告针对不可变的 `{run_id}` final checkpoint (step {training["steps"]:,}, 累计
{training["committed_tokens"]:,} 个训练 token), 在独立、已认证的 validation 语料上完成
candidate/shared/teacher 三角色的全量 NLL 评测。评测全程为 `torch.inference_mode()` 前向:
没有 optimizer state、没有 backward, 也没有任何参数更新。

{methodology_erratum_section}\
## 核心结论

{_markdown_table(("角色", "预测 token", "平均 NLL", "困惑度", "wall tok/s"), role_rows)}

- teacher gap closed fraction 为 **{_fmt(gap, 5)}**; 项目中 dense gate 的 0.10
  阈值结论为 **{gate_text}**。
- 三个角色都覆盖 {validation["prepared_manifest"]["shard_count"]} / {validation["prepared_manifest"]["shard_count"]}
  个 shard、{validation["prepared_manifest"]["sequence_count"]:,} 条序列; 预测 token 数完全一致,
  因而比较使用的是同一组 label。
- `candidate` 为训练后的 transfer 分支; `shared` 在同一 0.8B backbone 上关闭 transfer
  分支; `teacher` 为冻结的 Qwen3.5-9B 参照模型。

![最终 validation NLL](charts/validation_nll.svg)

![最终 validation 困惑度](charts/validation_perplexity.svg)

![各角色 validation 吞吐](charts/validation_role_throughput.svg)

{baseline_section}

## {run_id} 训练过程

- 汇总 compute 吞吐为 **{_fmt(training["aggregate_compute_tokens_per_second"], 1)} tok/s**;
  optimizer-step wall 吞吐为 **{_fmt(training["aggregate_wall_step_tokens_per_second"], 1)} tok/s**;
  从 train_start 到 final checkpoint 的端到端吞吐为
  **{_fmt(training["end_to_end_tokens_per_second"], 1)} tok/s**。
- 首次 `train_start` 到 final 的 elapsed 跨度为
  {training["wall_duration_seconds"] / 3600:.3f} 小时, 恢复感知口径在下方单列。
  共写入 {training["checkpoint_count"]} 个 checkpoint; 写盘总耗时
  {training["checkpoint_duration_seconds_total"] / 60:.2f} 分钟, 平均
  {_fmt(training["checkpoint_duration_seconds_mean"], 2)} 秒, 最大
  {_fmt(training["checkpoint_duration_seconds_max"], 2)} 秒。
- data wait 合计 {training["data_wait_seconds"]:.3f} 秒, 只占 compute-step 时间的
  {training["data_wait_fraction_of_compute"] * 100:.4f}%。训练峰值显存为
  {training["gpu_peak_allocated_gib"]:.3f} GiB allocated /
  {training["gpu_peak_reserved_gib"]:.3f} GiB reserved。
- 裁剪前 grad norm 在 {training["grad_norm_over_clip_threshold_steps"]} /
  {training["steps"]} 步超过阈值 {training["grad_clip_threshold"]:.3f}, 比例
  {training["grad_norm_over_clip_threshold_fraction"] * 100:.2f}%。hidden-alignment batch 为
  {training["alignment_steps"]} 步, ordinary batch 为 {training["ordinary_steps"]} 步。

{lifecycle_section}

- 前 50 步到后 50 步的 token 加权指标: NTP {first["ntp"]:.5f} 到
  {last["ntp"]:.5f}, teacher KD {first["teacher_kd"]:.5f} 到
  {last["teacher_kd"]:.5f}, anchor KL {first["anchor_kl"]:.5f} 到
  {last["anchor_kl"]:.5f}。这些训练 batch 的混合目标有明显采样噪声, 不能替代 held-out NLL。

![训练 loss 分量](charts/training_loss.svg)
{smoothed_loss_figure}

![裁剪前 grad norm](charts/gradient_norm.svg)

![学习率曲线](charts/learning_rate.svg)

![训练吞吐](charts/throughput.svg)

![训练显存](charts/gpu_memory.svg)

![Checkpoint 写盘耗时](charts/checkpoint_duration.svg)

{gpu_section}

{source_analysis_section}

## 按数据源拆分的 validation

{_markdown_table(("数据源", "shard", "输入 token", "candidate NLL", "shared NLL", "teacher NLL", "gap closed"), source_rows)}

表中 source NLL 按预测 token 加权, 并非对 shard 均值做简单平均。`summary.json` 同时保留每个
shard 的精确 NLL、token 数、source ID 和 candidate 相对 shared 的改善量。

![按 source 的 NLL](charts/validation_source_nll.svg)

![Validation 数据组成](charts/validation_source_tokens.svg)

![逐 shard NLL](charts/validation_shard_nll.svg)

## 欠拟合、MTP 与退火口径

- 是否欠拟合应首先看本次 held-out candidate/shared/teacher 差距, 不能只看训练 loss。
  如果 teacher gap closed 仍低, 并且后续同口径 validation checkpoint sweep 在更多 token 后持续改善,
  才能更有把握地支持“增加数据和 token budget”。本轮没有 validation 时间序列, 因此本次
  单个 final 点不能确定最佳停止位置, 也不能被称为 best checkpoint。
- {mtp_text}
- 不可变 LR 日程为: {schedule_text}。{cooldown_text}

## 数据治理限定

- 当前 lineage 为 `{governance["lineage_kind"]}`, 角色为 `{governance["role"]}`;
  **research_only={str(governance["research_only"]).lower()}**,
  **ready_for_training={str(governance["ready_for_training"]).lower()}**。
- 尚未完成的审计: `{pending}`。在 cross-source near-dedup、完整上下文 PII 扫描和项目
  benchmark 13-gram contamination 扫描完成前, 本结果只能作为研究证据, 不能宣称数据无污染或
  production-ready。

## 完整性、source-tree 与复现限定

- Final checkpoint: `{training["checkpoint"]["path"]}`
- Checkpoint manifest SHA256: `{training["checkpoint"]["manifest_sha256"]}`; 清单中的
  {training["checkpoint"]["inventory_file_count"]} 个文件已逐一重新计算 SHA256。
- Evaluation manifest SHA256: `{validation["identity"]["manifest_sha256"]}`
- Evaluation PLAN SHA256: `{validation["identity"]["plan_sha256"]}`
- Validation prepared manifest SHA256: `{validation["prepared_manifest"]["sha256"]}`
- 运行环境: torch `{runtime.get("torch_version")}`、CUDA `{runtime.get("cuda_version")}`、
  `{runtime.get("device_name")}`、compute capability `{runtime.get("compute_capability")}`、
  dtype `{runtime.get("dtype")}`、`FLA_TILELANG={runtime.get("fla_tilelang")}`、
  `CUDA_HOME={runtime.get("cuda_home")}`。`FLA_TILELANG=0` 明确使用 production Triton 路径。
- Checkpoint 加载模式为 `{lineage.get("mode")}`。saved/current exact-training fingerprint 是否
  一致: `{lineage.get("exact_training_fingerprint_match")}`; saved/current source tree 是否一致:
  `{lineage.get("source_tree_match")}`。训练时 source tree 为
  `{lineage.get("saved_source_tree_sha256")}`, 评测时为
  `{lineage.get("current_source_tree_sha256")}`。

这里的结论是: source model、calibration、训练/KD data manifest、loss weights、top_k、run
geometry 和 DCP trainable key/shape 均通过认证。{compatibility_text}

复现时使用绝对 checkpoint 路径最清晰:

```bash
FLA_TILELANG=0 bash scripts/with_cuda_toolchain.sh .venv/bin/python -m twen evaluate nll \\
  --config {run_dir / "resolved_config.yaml"} \\
  --checkpoint {checkpoint_path} \\
  --prepared-manifest {prepared_path} \\
  --prepared-manifest-sha256 {validation["prepared_manifest"]["sha256"]} \\
  --output {evaluation_dir} \\
  --role candidate --role shared --role teacher --batch-size 1 --device cuda:0
```

`MANIFEST.json` 认证中英文报告、`summary.json`、GPU 遥测 CSV 和全部 SVG; `COMPLETE`
再认证该 manifest。
"""


def generate_report(
    *,
    run_dir: Path,
    evaluation_dir: Path,
    prepared_manifest: Path,
    output_dir: Path,
    baseline_summary: Path | None = None,
) -> dict[str, Any]:
    training, training_raw = _summarize_training(run_dir.resolve())
    validation, _ = _summarize_validation(evaluation_dir.resolve(), prepared_manifest.resolve())
    checkpoint_state = validation.get("checkpoint_state") or {}
    if int(checkpoint_state.get("global_step", -1)) != training["steps"]:
        raise ValueError("validation checkpoint step differs from final training checkpoint")
    if int(checkpoint_state.get("committed_tokens", -1)) != training["committed_tokens"]:
        raise ValueError("validation checkpoint tokens differ from final training checkpoint")
    baseline_comparison = None
    if baseline_summary is not None:
        baseline, baseline_identity = _load_authenticated_report_summary(baseline_summary)
        baseline_comparison = _build_baseline_comparison(
            baseline_summary=baseline,
            baseline_identity=baseline_identity,
            training=training,
            validation=validation,
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_paths = _make_charts(
        output_dir,
        training,
        training_raw,
        validation,
        baseline_comparison,
    )
    methodology_errata = _methodology_errata_for_run(str(training["run_id"]))
    summary = {
        "schema_version": 1,
        "kind": "twen_dense_final_validation_report",
        "training": training,
        "validation": validation,
        **({"methodology_errata": methodology_errata} if methodology_errata else {}),
        **({"baseline_comparison": baseline_comparison} if baseline_comparison is not None else {}),
    }
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "REPORT.md"
    report_zh_path = output_dir / "REPORT.zh-CN.md"
    _write_json(summary_path, summary)
    _atomic_write(
        report_path,
        _build_report(training, validation, baseline_comparison),
    )
    _atomic_write(
        report_zh_path,
        _build_report_zh_cn(training, validation, baseline_comparison),
    )
    payloads = [summary_path, report_path, report_zh_path, *chart_paths]
    greedy_samples_path = output_dir / "greedy-samples.json"
    if greedy_samples_path.is_file():
        payloads.append(greedy_samples_path)
    gpu_sample_path = evaluation_dir.resolve() / "runtime-gpu-sample.csv"
    if gpu_sample_path.is_file():
        copied_sample = output_dir / "runtime-gpu-sample.csv"
        _atomic_write(copied_sample, gpu_sample_path.read_text(encoding="utf-8"))
        payloads.append(copied_sample)
    inventory = {
        path.relative_to(output_dir).as_posix(): {
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in sorted(payloads)
    }
    manifest = {
        "schema_version": 1,
        "kind": "twen_report_bundle",
        "source_evaluation_manifest_sha256": validation["identity"]["manifest_sha256"],
        "source_checkpoint_manifest_sha256": training["checkpoint"]["manifest_sha256"],
        **(
            {
                "source_baseline_report_manifest_sha256": baseline_comparison["baseline"]["report"][
                    "bundle_manifest_sha256"
                ],
                "source_baseline_summary_sha256": baseline_comparison["baseline"]["report"][
                    "summary_sha256"
                ],
            }
            if baseline_comparison is not None
            else {}
        ),
        "files": inventory,
    }
    manifest_path = output_dir / "MANIFEST.json"
    _write_json(manifest_path, manifest)
    complete_path = output_dir / "COMPLETE"
    _atomic_write(complete_path, _sha256_file(manifest_path) + "\n")
    return {
        "report": str(report_path),
        "report_zh_cn": str(report_zh_path),
        "summary": str(summary_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "figures": [str(path) for path in chart_paths],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--prepared-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        help=(
            "optional authenticated completed report summary used for exact "
            "same-validation comparison"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = generate_report(
        run_dir=args.run_dir,
        evaluation_dir=args.evaluation,
        prepared_manifest=args.prepared_manifest,
        output_dir=args.output,
        baseline_summary=args.baseline_summary,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
