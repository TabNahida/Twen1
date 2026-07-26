from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_v1_validation_report.py"
SPEC = importlib.util.spec_from_file_location("generate_v1_validation_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
reporting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporting)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _checkpoint(root: Path, *, steps: int, tokens: int) -> Path:
    checkpoint = root / f"step-{steps:012d}-milestone-complete"
    metadata = {
        "global_step": steps,
        "committed_tokens": tokens,
        "kind": "milestone",
        "tag": "complete",
    }
    _json(checkpoint / "metadata.json", metadata)
    manifest = {
        "algorithm": "sha256",
        "files": {"metadata.json": _sha(checkpoint / "metadata.json")},
        "version": 1,
    }
    _json(checkpoint / "manifest.json", manifest)
    (checkpoint / "COMPLETE").write_text(_sha(checkpoint / "manifest.json") + "\n")
    return checkpoint


def _run_fixture(
    root: Path,
    checkpoint: Path,
    *,
    run_id: str = "fixture-v3-current",
    mtp_weight: float | None = None,
) -> Path:
    run = root / "run"
    metrics = []
    telemetry = []
    for step in range(1, 4):
        tokens = step * 100
        metrics.append(
            {
                "step": step,
                "tokens": tokens,
                "tokens_this_step": 100,
                "loss": 3.1 - step * 0.1,
                "ntp": 2.5 - step * 0.1,
                "teacher_kd": 0.5 - step * 0.02,
                "anchor_kl": 0.05 + step * 0.01,
                "hidden_alignment": 0.2 if step == 2 else 0.0,
                "grad_norm": 0.8 + step * 0.2,
                "lr/adapters": 2e-4 / step,
                "lr/scale": 1e-3 / step,
                **({"hidden_alignment_loss": 0.2} if step == 2 else {}),
            }
        )
        telemetry.append(
            {
                "step": step,
                "tokens": tokens,
                "tokens_this_step": 100,
                "compute_step_seconds": 0.1,
                "wall_clock_step_seconds": 0.11,
                "compute_tokens_per_second": 1000 + step,
                "wall_clock_tokens_per_second": 900 + step,
                "compute_tokens_per_second_ema": 1000 + step,
                "wall_clock_tokens_per_second_ema": 900 + step,
                "data_wait_seconds": 0.001,
                "gpu_allocated_gib": 10 + step,
                "gpu_reserved_gib": 12 + step,
                "gpu_peak_allocated_gib": 11 + step,
                "gpu_peak_reserved_gib": 13 + step,
            }
        )
    _jsonl(run / "metrics.jsonl", metrics)
    _jsonl(run / "telemetry.jsonl", telemetry)
    _jsonl(
        run / "events.jsonl",
        [
            {"event": "train_start", "timestamp_utc": "2026-01-01T00:00:00+00:00"},
            {
                "event": "checkpoint_complete",
                "step": 3,
                "duration_seconds": 2.0,
                "timestamp_utc": "2026-01-01T00:00:03+00:00",
            },
            {
                "event": "train_complete",
                "checkpoint": str(checkpoint),
                "timestamp_utc": "2026-01-01T00:00:04+00:00",
            },
        ],
    )
    losses = {"ntp": 1.0, "teacher_kd": 1.0}
    if mtp_weight is not None:
        losses["mtp"] = mtp_weight
    config = {
        "run_id": run_id,
        "track": "base",
        "stage": "dense-oracle",
        "optimizer": {
            "grad_clip_norm": 1.0,
            "max_tokens": 300,
            "warmup_tokens": 30,
        },
        "data": {"micro_batch_size": 1, "global_batch_tokens": 100},
        "losses": losses,
    }
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return run


def _evaluation_fixture(root: Path, prepared: Path) -> Path:
    evaluation = root / "evaluation"
    prepared_payload = {
        "schema_version": 2,
        "kind": "twen_prepared_text",
        "dataset_fingerprint": "dataset-fixture",
        "sequence_count": 2,
        "token_count": 20,
        "lineage": {
            "kind": "authenticated_extracted_corpus",
            "role": "validation",
            "research_only": True,
            "ready_for_training": False,
            "pending_audits": ["near_dedup"],
            "audits": {"near_dedup": "pending"},
        },
        "shards": [
            {
                "shard_id": "shard-000000",
                "source_path": "/data/extracted/source_a/chunk-0/validation.jsonl",
                "sequence_count": 1,
                "token_count": 10,
            },
            {
                "shard_id": "shard-000001",
                "source_path": "/data/extracted/source_b/chunk-0/validation.jsonl",
                "sequence_count": 1,
                "token_count": 10,
            },
        ],
    }
    _json(prepared, prepared_payload)
    plan = {
        "prepared_manifest_sha256": _sha(prepared),
        "prepared_dataset_fingerprint": "dataset-fixture",
    }
    _json(evaluation / "PLAN.json", plan)
    role_nll = {
        "candidate": (2.0, 2.2),
        "shared": (3.0, 3.2),
        "teacher": (1.0, 1.2),
    }
    roles: dict[str, object] = {}
    for role, values in role_nll.items():
        entries = []
        nll_sum = 0.0
        predicted_tokens = 0
        for index, mean_nll in enumerate(values):
            shard_id = f"shard-{index:06d}"
            role_path = Path("roles") / role / shard_id
            directory = evaluation / role_path
            result = {
                "role": role,
                "source_shard_id": shard_id,
                "sequences": 1,
                "predicted_tokens": 9,
                "nll_sum": mean_nll * 9,
                "mean_nll": mean_nll,
            }
            _json(directory / "SHARD_STATE.json", {"shard_id": shard_id})
            _json(directory / "result.json", result)
            _json(
                directory / "COMPLETE",
                {
                    "completed_unix_seconds": (directory / "SHARD_STATE.json").stat().st_mtime
                    + 1.0,
                    "outputs": [
                        {
                            "path": "result.json",
                            "sha256": _sha(directory / "result.json"),
                            "size": (directory / "result.json").stat().st_size,
                        }
                    ],
                },
            )
            entries.append(
                {
                    "path": role_path.as_posix(),
                    "complete_sha256": _sha(directory / "COMPLETE"),
                }
            )
            nll_sum += mean_nll * 9
            predicted_tokens += 9
        mean = nll_sum / predicted_tokens
        roles[role] = {
            "sequences": 2,
            "predicted_tokens": predicted_tokens,
            "nll_sum": nll_sum,
            "mean_nll": mean,
            "perplexity": reporting.math.exp(mean),
            "shards": entries,
        }
    manifest = {
        "schema_version": 1,
        "kind": "twen_nll_evaluation",
        "plan_sha256": _sha(evaluation / "PLAN.json"),
        "checkpoint_state": {"global_step": 3, "committed_tokens": 300},
        "roles": roles,
        "acceptance": {
            "teacher_gap_closed_fraction": 0.5,
            "dense_gap_gate_pass": True,
        },
    }
    _json(evaluation / "manifest.json", manifest)
    (evaluation / "COMPLETE").write_text(_sha(evaluation / "manifest.json") + "\n")
    return evaluation


def _baseline_report_fixture(
    root: Path,
    prepared: Path,
    *,
    prepared_sha256: str | None = None,
    report_kind: str = "twen_dense_final_validation_report",
    nested_current_candidate_mean_nll: float | None = None,
) -> Path:
    report = root / "baseline-report"
    checkpoint_manifest_sha256 = "a" * 64
    evaluation_manifest_sha256 = "b" * 64
    prior_checkpoint_manifest_sha256 = "c" * 64
    prior_evaluation_manifest_sha256 = "d" * 64
    role_nll = {
        "candidate": 2.5,
        "shared": 3.1,
        "teacher": 1.1,
    }
    prior_role_nll = {
        "candidate": 2.7,
        "shared": 3.1,
        "teacher": 1.1,
    }
    prepared_identity = {
        "sha256": prepared_sha256 or _sha(prepared),
        "dataset_fingerprint": "dataset-fixture",
        "sequence_count": 2,
        "input_token_count": 20,
        "shard_count": 2,
    }
    summary = {
        "schema_version": 1,
        "kind": report_kind,
        "training": {
            "run_id": "fixture-v2-baseline",
            "checkpoint": {
                "manifest_sha256": checkpoint_manifest_sha256,
            },
        },
        "validation": {
            "identity": {
                "manifest_sha256": evaluation_manifest_sha256,
            },
            "prepared_manifest": prepared_identity,
            "roles": {
                role: {
                    "mean_nll": mean_nll,
                    "perplexity": reporting.math.exp(mean_nll),
                    "predicted_tokens": 18,
                }
                for role, mean_nll in role_nll.items()
            },
            "acceptance": {
                "teacher_gap_closed_fraction": 0.3,
            },
        },
        "baseline_comparison": {
            "baseline": {
                "run_id": "fixture-v1-baseline",
                "checkpoint_manifest_sha256": prior_checkpoint_manifest_sha256,
                "evaluation_manifest_sha256": prior_evaluation_manifest_sha256,
                "prepared_manifest": dict(prepared_identity),
            },
            "current": {
                "run_id": "fixture-v2-baseline",
                "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
                "evaluation_manifest_sha256": evaluation_manifest_sha256,
                "prepared_manifest": dict(prepared_identity),
            },
            "comparability": {
                "same_prepared_manifest_sha256": True,
                "same_dataset_fingerprint": True,
                "same_sequence_input_token_and_shard_counts": True,
                "same_role_predicted_token_counts": True,
            },
            "roles": {
                role: {
                    "predicted_tokens": {
                        "baseline": 18,
                        "current": 18,
                        "match": True,
                    },
                    "mean_nll": reporting._metric_change(
                        baseline=prior_role_nll[role],
                        current=(
                            nested_current_candidate_mean_nll
                            if role == "candidate" and nested_current_candidate_mean_nll is not None
                            else role_nll[role]
                        ),
                        lower_is_better=True,
                    ),
                    "perplexity": reporting._metric_change(
                        baseline=reporting.math.exp(prior_role_nll[role]),
                        current=reporting.math.exp(role_nll[role]),
                        lower_is_better=True,
                    ),
                }
                for role in ("candidate", "shared", "teacher")
            },
            "teacher_gap_closed_fraction": reporting._metric_change(
                baseline=0.2,
                current=0.3,
                lower_is_better=False,
            ),
        },
    }
    summary_path = report / "summary.json"
    _json(summary_path, summary)
    manifest = {
        "schema_version": 1,
        "kind": "twen_report_bundle",
        "source_checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "source_evaluation_manifest_sha256": evaluation_manifest_sha256,
        "files": {
            "summary.json": {
                "sha256": _sha(summary_path),
                "size": summary_path.stat().st_size,
            }
        },
    }
    _json(report / "MANIFEST.json", manifest)
    (report / "COMPLETE").write_text(_sha(report / "MANIFEST.json") + "\n")
    return summary_path


def test_report_generator_builds_authenticated_svg_bundle(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoints", steps=3, tokens=300)
    run = _run_fixture(tmp_path, checkpoint)
    prepared = tmp_path / "prepared" / "manifest.json"
    evaluation = _evaluation_fixture(tmp_path, prepared)
    (evaluation / "runtime-gpu-sample.csv").write_text(
        "timestamp, power.draw [W], power.limit [W], clocks.current.sm [MHz], "
        "clocks.current.memory [MHz], pstate, utilization.gpu [%], "
        "utilization.memory [%], memory.used [MiB], memory.free [MiB], temperature.gpu\n"
        "2026/01/01 00:00:00.000, 400 W, 600 W, 2600 MHz, 13801 MHz, P1, "
        "80 %, 25 %, 24000 MiB, 8000 MiB, 70\n"
        "2026/01/01 00:00:01.000, 450 W, 600 W, 2700 MHz, 13801 MHz, P1, "
        "90 %, 30 %, 24000 MiB, 8000 MiB, 72\n",
        encoding="utf-8",
    )
    output = tmp_path / "report"
    greedy_samples = output / "greedy-samples.json"
    _json(greedy_samples, {"kind": "twen_dense_greedy_samples", "samples": []})

    result = reporting.generate_report(
        run_dir=run,
        evaluation_dir=evaluation,
        prepared_manifest=prepared,
        output_dir=output,
    )

    assert Path(result["report"]).is_file()
    assert Path(result["report_zh_cn"]).is_file()
    assert len(result["figures"]) == 14
    assert all(
        Path(path).read_text(encoding="utf-8").startswith("<?xml") for path in result["figures"]
    )
    manifest = json.loads((output / "MANIFEST.json").read_text())
    assert (output / "COMPLETE").read_text().strip() == _sha(output / "MANIFEST.json")
    assert manifest["files"]["REPORT.md"]["sha256"] == _sha(output / "REPORT.md")
    assert manifest["files"]["REPORT.zh-CN.md"]["sha256"] == _sha(output / "REPORT.zh-CN.md")
    assert manifest["files"]["runtime-gpu-sample.csv"]["sha256"] == _sha(
        output / "runtime-gpu-sample.csv"
    )
    assert manifest["files"]["greedy-samples.json"] == {
        "sha256": _sha(greedy_samples),
        "size": greedy_samples.stat().st_size,
    }
    report = (output / "REPORT.md").read_text()
    assert "Teacher gap closed" in report
    assert "research_only=true" in report
    assert "Validation GPU runtime sample" in report
    report_zh = (output / "REPORT.zh-CN.md").read_text()
    assert "forward-only inference validation" in report_zh
    assert "MTP=0" in report_zh
    assert "research_only=true" in report_zh
    assert "没有触及 600 W 功耗墙" in report_zh
    summary = json.loads((output / "summary.json").read_text())
    assert "methodology_errata" not in summary
    gpu_sample = summary["validation"]["runtime_gpu_sample"]
    assert gpu_sample["statistics"]["power_draw_w"]["mean"] == pytest.approx(425.0)
    assert gpu_sample["statistics"]["gpu_utilization_percent"]["mean"] == pytest.approx(85.0)


def test_v3_report_emits_authenticated_mtp_position_erratum(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoints", steps=3, tokens=300)
    run = _run_fixture(
        tmp_path,
        checkpoint,
        run_id=reporting.V3_RUN_ID,
        mtp_weight=0.1,
    )
    prepared = tmp_path / "prepared" / "manifest.json"
    evaluation = _evaluation_fixture(tmp_path, prepared)
    output = tmp_path / "report"

    reporting.generate_report(
        run_dir=run,
        evaluation_dir=evaluation,
        prepared_manifest=prepared,
        output_dir=output,
    )

    summary = json.loads((output / "summary.json").read_text())
    assert summary["methodology_errata"] == [
        {
            "id": "mtp_rope_position_alignment",
            "disclosed_on": "2026-07-26",
            "affected_source": {
                "path": reporting.V3_MTP_AFFECTED_SOURCE_PATH,
                "git_blob_sha1": reporting.V3_MTP_AFFECTED_GIT_BLOB_SHA1,
            },
            "observed_rope_position_offset_tokens": 0,
            "required_rope_position_offset_tokens": 1,
            "validation_objective": "ntp_only",
            "fully_native_aligned_mtp_claim_supported": False,
            "causal_mtp_benefit_claim_supported": False,
            "fixed_by_commit": reporting.V3_MTP_FIX_COMMIT,
        }
    ]
    report = (output / "REPORT.md").read_text(encoding="utf-8")
    report_zh = (output / "REPORT.zh-CN.md").read_text(encoding="utf-8")
    assert "Methodology erratum (2026-07-26)" in report
    assert "not a claim of fully native-aligned Qwen3.5 MTP execution" in report
    assert "方法学勘误 (2026-07-26)" in report_zh
    assert "不再把它表述为完全对齐的 Qwen3.5 原生 MTP forward" in report_zh
    manifest = json.loads((output / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["files"]["summary.json"] == {
        "sha256": _sha(output / "summary.json"),
        "size": (output / "summary.json").stat().st_size,
    }
    assert (output / "COMPLETE").read_text().strip() == _sha(output / "MANIFEST.json")


def test_gpu_sample_accepts_bare_numeric_values(tmp_path: Path) -> None:
    sample = tmp_path / "runtime-gpu-sample.csv"
    sample.write_text(
        "timestamp,pstate,power.draw [W],utilization.gpu [%],temperature.gpu\n"
        "2026/01/01 00:00:00.000,P1,400,80,70\n",
        encoding="utf-8",
    )

    result = reporting._summarize_gpu_sample(sample)

    assert result is not None
    assert result["statistics"]["power_draw_w"]["mean"] == 400.0
    assert result["statistics"]["gpu_utilization_percent"]["mean"] == 80.0
    assert result["statistics"]["temperature_c"]["mean"] == 70.0


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("400 kW", "unexpected unit"),
        ("NaN W", "non-finite"),
        ("+Infinity W", "non-finite"),
    ],
)
def test_gpu_sample_rejects_wrong_units_and_non_finite_values(
    tmp_path: Path,
    value: str,
    message: str,
) -> None:
    sample = tmp_path / "runtime-gpu-sample.csv"
    sample.write_text(
        f"timestamp,pstate,power.draw [W]\n2026/01/01 00:00:00.000,P1,{value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        reporting._summarize_gpu_sample(sample)


def test_report_generator_compares_authenticated_version_history(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoints", steps=3, tokens=300)
    run = _run_fixture(tmp_path, checkpoint)
    prepared = tmp_path / "prepared" / "manifest.json"
    evaluation = _evaluation_fixture(tmp_path, prepared)
    baseline = _baseline_report_fixture(tmp_path, prepared)
    output = tmp_path / "report"

    result = reporting.generate_report(
        run_dir=run,
        evaluation_dir=evaluation,
        prepared_manifest=prepared,
        output_dir=output,
        baseline_summary=baseline,
    )

    summary = json.loads((output / "summary.json").read_text())
    comparison = summary["baseline_comparison"]
    candidate = comparison["roles"]["candidate"]
    assert candidate["mean_nll"]["absolute_change"] == pytest.approx(-0.4)
    assert candidate["mean_nll"]["relative_change_fraction"] == pytest.approx(-0.16)
    gap = comparison["teacher_gap_closed_fraction"]
    assert gap["absolute_change"] == pytest.approx(0.2)
    assert gap["relative_change_fraction"] == pytest.approx(2 / 3)
    assert all(comparison["comparability"].values())
    history = comparison["cross_version_history"]
    assert [row["run_id"] for row in history] == [
        "fixture-v1-baseline",
        "fixture-v2-baseline",
        "fixture-v3-current",
    ]
    assert [row["candidate_mean_nll"] for row in history] == pytest.approx([2.7, 2.5, 2.1])
    assert [row["teacher_gap_closed_fraction"] for row in history] == pytest.approx([0.2, 0.3, 0.5])
    assert history[0]["provenance"] == "authenticated_nested_baseline_comparison"
    assert history[1]["provenance"] == "authenticated_baseline_report_summary"
    figure_names = {Path(path).name for path in result["figures"]}
    assert "validation_candidate_nll_history.svg" in figure_names
    assert "validation_teacher_gap_closed_history.svg" in figure_names
    assert len(result["figures"]) == 14
    report = (output / "REPORT.md").read_text()
    report_zh = (output / "REPORT.zh-CN.md").read_text()
    assert "Authenticated baseline comparison" in report
    assert "NLL absolute Δ" in report
    assert "Cross-version history on the same held-out set" in report
    assert "charts/validation_candidate_nll_history.svg" in report
    assert "charts/validation_teacher_gap_closed_history.svg" in report
    assert "已认证的 baseline 对照" in report_zh
    assert "NLL 绝对变化" in report_zh
    assert "同一 held-out 集的跨版本历史" in report_zh
    assert "charts/validation_candidate_nll_history.svg" in report_zh
    assert "charts/validation_teacher_gap_closed_history.svg" in report_zh
    manifest = json.loads((output / "MANIFEST.json").read_text())
    assert manifest["source_baseline_summary_sha256"] == _sha(baseline)
    assert manifest["source_baseline_report_manifest_sha256"] == _sha(
        baseline.parent / "MANIFEST.json"
    )


def test_baseline_summary_tamper_fails_closed(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared.json"
    _json(prepared, {"dataset_fingerprint": "dataset-fixture"})
    baseline = _baseline_report_fixture(tmp_path, prepared)
    baseline.write_text(baseline.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"summary\.json hash/size mismatch"):
        reporting._load_authenticated_report_summary(baseline)


def test_baseline_with_different_validation_manifest_fails_closed(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoints", steps=3, tokens=300)
    run = _run_fixture(tmp_path, checkpoint)
    prepared = tmp_path / "prepared" / "manifest.json"
    evaluation = _evaluation_fixture(tmp_path, prepared)
    baseline = _baseline_report_fixture(
        tmp_path,
        prepared,
        prepared_sha256="c" * 64,
    )

    with pytest.raises(ValueError, match="prepared-manifest sha256 mismatch"):
        reporting.generate_report(
            run_dir=run,
            evaluation_dir=evaluation,
            prepared_manifest=prepared,
            output_dir=tmp_path / "report",
            baseline_summary=baseline,
        )


def test_nested_baseline_current_metric_tamper_fails_closed(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoints", steps=3, tokens=300)
    run = _run_fixture(tmp_path, checkpoint)
    prepared = tmp_path / "prepared" / "manifest.json"
    evaluation = _evaluation_fixture(tmp_path, prepared)
    baseline = _baseline_report_fixture(
        tmp_path,
        prepared,
        nested_current_candidate_mean_nll=2.4,
    )

    with pytest.raises(
        ValueError,
        match="nested baseline current candidate mean_nll differs",
    ):
        reporting.generate_report(
            run_dir=run,
            evaluation_dir=evaluation,
            prepared_manifest=prepared,
            output_dir=tmp_path / "report",
            baseline_summary=baseline,
        )


def test_completed_evaluation_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="incomplete"):
        reporting._load_completed_evaluation(tmp_path)
