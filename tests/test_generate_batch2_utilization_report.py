from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "generate_batch2_utilization_report.py"
    spec = importlib.util.spec_from_file_location("generate_batch2_utilization_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reporter = _load_script()


def _case(*, batch_size: int, ac: int, tps: float, free_gib: float, alignment: bool) -> dict:
    logical_tokens = batch_size * 4096
    timing = {
        "teacher_cpu_offload_stage_seconds": {"mean": 0.64 if alignment else 0.0},
        "teacher_cpu_offload_restore_seconds": {"mean": 0.02 if alignment else 0.0},
    }
    sample = {
        "ok": True,
        "loss_finite": True,
        "gradients": {
            "finite": True,
            "present_tensors": 72,
            "missing_tensors": 0,
            "nonfinite_tensors": 0,
        },
        "memory": {
            "estimated_free_at_peak_reserved_bytes": round(free_gib * reporter.GIB),
        },
    }
    return {
        "ok": True,
        "activation_checkpoint_layer_count": ac,
        "samples": [sample],
        "health": {
            "ok": True,
            "loss_finite": True,
            "gradients_finite": True,
            "maximum_missing_gradient_tensors": 0,
            "maximum_nonfinite_gradient_tensors": 0,
            "present_gradient_tensor_counts": [72],
        },
        "summary": {
            "throughput": {
                "logical_tokens_per_second_gpu": {"mean": tps},
                "logical_tokens_per_second_wall": {"mean": tps * 0.99},
            },
            "timing_seconds": timing,
            "memory_worst_case": {
                "peak_allocated_bytes": round((29.0 - free_gib) * reporter.GIB),
                "peak_reserved_bytes": round((29.5 - free_gib) * reporter.GIB),
                "minimum_free_after_bytes": round(free_gib * reporter.GIB),
            },
        },
        "logical_tokens": logical_tokens,
    }


def _payload(
    *,
    batch_size: int,
    case: dict,
    alignment: bool,
    cases: list[dict] | None = None,
    mtp_attention: str | None = "sdpa",
) -> dict:
    value = {
        "batch": {
            "batch_size": batch_size,
            "sequence_length": 4096,
            "logical_tokens": batch_size * 4096,
        },
        "runtime": {"activation_checkpoint_layer_count": case["activation_checkpoint_layer_count"]},
        "loss_weights": {"mtp": 0.1},
        "mtp": {"attention_implementation": mtp_attention},
        "teacher_cpu_offload": {"enabled": True},
        "optimizer_state_reserve": {"requested_gib": 1.5},
        "no_optimizer_created": True,
        "no_optimizer_steps": True,
        "optimizer_step_calls": 0,
        "ok": True,
        "production_acceptance": True,
        "graph": {"online_hidden_alignment": alignment, "parameter_update": False},
        **case,
    }
    if cases is not None:
        value["cases"] = cases
        value["runtime"]["activation_checkpoint_layer_count"] = None
    return value


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_power(
    path: Path,
    draws: tuple[float, ...],
    *,
    utilizations: tuple[float, ...] | None = None,
    clocks: tuple[float, ...] | None = None,
) -> None:
    utilizations = utilizations or (99.0,) * len(draws)
    clocks = clocks or (2700.0,) * len(draws)
    assert len(draws) == len(utilizations) == len(clocks)
    header = (
        "timestamp,power_draw_w,power_limit_w,utilization_gpu_percent,"
        "utilization_memory_percent,clocks_sm_mhz,memory_used_mib,memory_free_mib,"
        "temperature_gpu_c\n"
    )
    rows = [
        f"2026/07/19 12:00:0{index},{draw},600,{utilization},40,{clock},25000,7000,72"
        for index, (draw, utilization, clock) in enumerate(
            zip(draws, utilizations, clocks, strict=True)
        )
    ]
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _nsys_summary() -> dict:
    identity = {"path": "raw-file", "size": 123, "sha256": "a" * 64}
    return {
        "schema_version": 1,
        "kind": "twen_sanitized_nsys_kernel_acceptance",
        "publishable_raw_profile": False,
        "capture": {
            "batch_size": 1,
            "activation_checkpoint_layer_count": 0,
            "cuda_profiler_api": True,
            "torch_profiler_enabled": False,
            "benchmark": {**identity, "path": "benchmark.json"},
            "nsys_report": {**identity, "path": "capture.nsys-rep"},
            "sqlite": {**identity, "path": "capture.sqlite"},
        },
        "security": {
            "clean_environment_allowlist": True,
            "captured_environment_values_exported": False,
            "sensitive_environment_variable_name_matches": 0,
            "raw_profile_in_report_bundle": False,
        },
        "acceptance": {
            "ok": True,
            "production_shape": True,
            "mtp_attention_implementation": "sdpa",
            "gradient_tensors_present": 72,
            "gradient_tensors_missing": 0,
            "gradient_tensors_nonfinite": 0,
            "optimizer_created": False,
            "optimizer_steps": 0,
            "flash_forward_instances": 14,
            "flash_backward_instances": 21,
            "eager_softmax_instances": 0,
        },
        "kernel_time": {
            "total_cuda_kernel_time_ms": 100.0,
            "dense_gemm": {"instances": 10, "time_ms": 70.0, "time_percent": 70.0},
            "compiled_triton_reductions": {
                "instances": 5,
                "time_ms": 10.0,
                "time_percent": 10.0,
            },
            "fla_recurrent_attention": {
                "instances": 4,
                "time_ms": 8.0,
                "time_percent": 8.0,
            },
            "mtp_sdpa_flash": {"instances": 3, "time_ms": 5.0, "time_percent": 5.0},
        },
    }


def _assert_bundle(
    *,
    report_path: Path,
    manifest_path: Path,
    complete_path: Path,
    expected_files: dict[str, str],
) -> tuple[dict, dict, dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    provenance_sha = _canonical_sha256(report["source_provenance"])
    assert manifest["schema_version"] == 1
    assert manifest["kind"] == "twen_rtx5090_base_dense_utilization_report_bundle"
    assert manifest["accepted"] == report["accepted"]
    assert manifest["recommendation"] == report["recommendation"]
    assert manifest["source_provenance_sha256"] == provenance_sha
    assert set(manifest["files"]) == set(expected_files)
    for key, name in expected_files.items():
        path = manifest_path.parent / name
        assert manifest["files"][key] == {
            "path": name,
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
    assert complete == {
        "schema_version": 1,
        "kind": "twen_rtx5090_base_dense_utilization_report_complete",
        "manifest": manifest_path.name,
        "manifest_sha256": _sha256(manifest_path),
        "report": {"path": report_path.name, "sha256": _sha256(report_path)},
        "accepted": report["accepted"],
        "recommendation": report["recommendation"],
        "source_provenance_sha256": provenance_sha,
    }
    return report, manifest, complete


def test_report_generator_selects_fastest_safe_case_and_writes_batch_neutral_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    b1_ordinary_path = tmp_path / "b1-ordinary.json"
    b1_alignment_path = tmp_path / "b1-alignment-ac24.json"
    b2_ac24_path = tmp_path / "b2-ordinary-ac24.json"
    b2_ac20_path = tmp_path / "b2-ordinary-ac20.json"
    b2_ac16_path = tmp_path / "b2-ordinary-ac16.json"
    b2_alignment_path = tmp_path / "b2-alignment-ac24.json"
    b4_ordinary_path = tmp_path / "b4-ordinary-ac24.json"
    b4_alignment_path = tmp_path / "b4-alignment-ac24.json"
    nsys_summary_path = tmp_path / "nsys-summary.json"
    output_prefix = tmp_path / "utilization-report"
    _write_json(nsys_summary_path, _nsys_summary())

    b1_ac0 = _case(batch_size=1, ac=0, tps=7000.0, free_gib=3.5, alignment=False)
    _write_json(
        b1_ordinary_path,
        _payload(batch_size=1, case=b1_ac0, alignment=False),
    )
    _write_power(b1_ordinary_path.with_suffix(".power.csv"), (420.0, 500.0, 550.0))
    b1_alignment = _case(batch_size=1, ac=24, tps=4000.0, free_gib=5.5, alignment=True)
    _write_json(
        b1_alignment_path,
        _payload(batch_size=1, case=b1_alignment, alignment=True),
    )
    _write_power(b1_alignment_path.with_suffix(".power.csv"), (430.0, 510.0, 560.0))

    batch2_cases = (
        (b2_ac24_path, _case(batch_size=2, ac=24, tps=6100.0, free_gib=5.0, alignment=False)),
        (b2_ac20_path, _case(batch_size=2, ac=20, tps=6600.0, free_gib=3.2, alignment=False)),
        (b2_ac16_path, _case(batch_size=2, ac=16, tps=6900.0, free_gib=2.5, alignment=False)),
    )
    for path, case in batch2_cases:
        _write_json(path, _payload(batch_size=2, case=case, alignment=False))
        _write_power(path.with_suffix(".power.csv"), (410.0, 510.0, 560.0))

    b2_alignment = _case(batch_size=2, ac=24, tps=3600.0, free_gib=4.0, alignment=True)
    _write_json(
        b2_alignment_path,
        _payload(batch_size=2, case=b2_alignment, alignment=True),
    )
    _write_power(b2_alignment_path.with_suffix(".power.csv"), (430.0, 520.0, 570.0))

    b4_ordinary = _case(batch_size=4, ac=24, tps=6300.0, free_gib=4.2, alignment=False)
    _write_json(
        b4_ordinary_path,
        _payload(batch_size=4, case=b4_ordinary, alignment=False),
    )
    _write_power(b4_ordinary_path.with_suffix(".power.csv"), (440.0, 530.0, 580.0))
    b4_alignment = _case(batch_size=4, ac=24, tps=3300.0, free_gib=3.8, alignment=True)
    _write_json(
        b4_alignment_path,
        _payload(batch_size=4, case=b4_alignment, alignment=True),
    )
    _write_power(b4_alignment_path.with_suffix(".power.csv"), (450.0, 540.0, 590.0))

    input_args = [
        "--ordinary-b1",
        str(b1_ordinary_path),
        "--alignment-b1",
        str(b1_alignment_path),
        "--ordinary-batch2",
        str(b2_ac24_path),
        "--ordinary-batch2",
        str(b2_ac20_path),
        "--ordinary-batch2",
        str(b2_ac16_path),
        "--alignment-batch2",
        str(b2_alignment_path),
        "--ordinary-candidate",
        str(b4_ordinary_path),
        "--alignment-candidate",
        str(b4_alignment_path),
        "--nsys-summary",
        str(nsys_summary_path),
    ]
    exit_code = reporter.main(
        [
            *input_args,
            "--output-prefix",
            str(output_prefix),
        ]
    )

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == {"ok": True, "output_prefix": str(output_prefix.resolve())}
    report_path = output_prefix.with_name(output_prefix.name + ".json")
    markdown_path = output_prefix.with_name(output_prefix.name + ".md")
    flat_expected_files = {
        "report_json": report_path.name,
        "report_markdown": markdown_path.name,
        "throughput_memory_svg": output_prefix.name + "-throughput-memory.svg",
        "power_svg": output_prefix.name + "-power.svg",
        "utilization_svg": output_prefix.name + "-utilization.svg",
    }
    report, _, _ = _assert_bundle(
        report_path=report_path,
        manifest_path=output_prefix.with_name(output_prefix.name + ".MANIFEST.json"),
        complete_path=output_prefix.with_name(output_prefix.name + ".COMPLETE"),
        expected_files=flat_expected_files,
    )
    assert report["accepted"] is True
    assert report["recommendation"]["batch_size"] == 1
    assert report["recommendation"]["ordinary_case"] == "b1-ordinary-ac0"
    assert report["recommendation"]["alignment_case"] == "b1-alignment-ac24"
    assert report["recommendation"]["production_config"] == {
        "micro_batch_size": 1,
        "activation_checkpointing": True,
        "activation_checkpointing_on_alignment_only": True,
        "activation_checkpoint_layer_count": None,
    }
    assert report["kernel_profile"]["source"]["sha256"] == _sha256(nsys_summary_path)
    assert report["kernel_profile"]["source"]["path"] == nsys_summary_path.name
    assert report["kernel_profile"]["summary"] == _nsys_summary()
    assert [item["batch_size"] for item in report["batch_comparisons"]] == [1, 2, 4]
    b2_comparison = next(item for item in report["batch_comparisons"] if item["batch_size"] == 2)
    assert b2_comparison["ordinary_case"] == "b2-ordinary-ac20"
    assert b2_comparison["alignment_case"] == "b2-alignment-ac24"
    assert (
        report["weighted_comparison"]["batch2_tokens_per_second"]
        < report["weighted_comparison"]["batch1_tokens_per_second"]
    )
    ac20 = next(row for row in report["rows"] if row["label"] == "b2-ordinary-ac20")
    assert ac20["health"]["present_gradient_tensor_counts"] == [72]
    assert ac20["power"]["sample_count"] == 3
    assert ac20["power"]["active_window"]["sample_count"] == 3
    assert ac20["power"]["stats"]["power_draw_w"]["p95"] == 560.0
    assert ac20["estimated_active_tokens_per_joule"] == pytest.approx(
        ac20["production_tokens_per_second"]
        / ac20["power"]["active_window"]["stats"]["power_draw_w"]["mean"]
    )
    assert len(ac20["source"]["sha256"]) == 64
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "95% ordinary / 5% alignment" in markdown
    assert "batch-2 利用率验收" not in markdown
    assert "| batch-1 | b1-ordinary-ac0 | b1-alignment-ac24 |" in markdown
    assert "| batch-2 | b2-ordinary-ac20 | b2-alignment-ac24 |" in markdown
    assert "| batch-4 | b4-ordinary-ac24 | b4-alignment-ac24 |" in markdown
    assert "whole/active" in markdown
    assert "不是逐 token 积分能耗" in markdown
    assert "不参与选档" in markdown
    assert "dense GEMM | 70.00" in markdown
    assert "compiled Triton reductions | 10.00" in markdown
    assert "FLA recurrent attention | 8.00" in markdown
    assert "MTP SDPA flash | 5.00" in markdown
    assert "raw profile in bundle=`false`" in markdown
    assert "activation_checkpoint_layer_count=null" in markdown
    for suffix in ("throughput-memory", "power", "utilization"):
        path = output_prefix.with_name(output_prefix.name + f"-{suffix}.svg")
        rendered = path.read_text(encoding="utf-8")
        assert rendered.startswith("<svg")
        assert rendered.endswith("</svg>\n")
        assert path.name in markdown

    output_dir = tmp_path / "directory-bundle"
    assert reporter.main([*input_args, "--output-dir", str(output_dir)]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "output_dir": str(output_dir.resolve()),
    }
    directory_expected_files = {
        "report_json": "report.json",
        "report_markdown": "report.md",
        "throughput_memory_svg": "throughput-memory.svg",
        "power_svg": "power.svg",
        "utilization_svg": "utilization.svg",
    }
    _assert_bundle(
        report_path=output_dir / "report.json",
        manifest_path=output_dir / "MANIFEST.json",
        complete_path=output_dir / "COMPLETE",
        expected_files=directory_expected_files,
    )


def test_active_power_window_is_inclusive_and_keeps_internal_bubbles(
    tmp_path: Path,
) -> None:
    path = tmp_path / "b2-ordinary-ac24.json"
    case = _case(batch_size=2, ac=24, tps=6200.0, free_gib=4.0, alignment=False)
    _write_json(path, _payload(batch_size=2, case=case, alignment=False))
    _write_power(
        path.with_suffix(".power.csv"),
        (50.0, 200.0, 600.0, 400.0, 50.0),
        utilizations=(10.0, 60.0, 20.0, 70.0, 10.0),
        clocks=(900.0, 2000.0, 1500.0, 2500.0, 800.0),
    )

    row = reporter._single_row(
        path,
        mode="ordinary",
        global_batch_tokens=262144,
        require_power=True,
        candidate=True,
    )

    power = row["power"]
    assert power["sample_count"] == 5
    assert power["stats"]["power_draw_w"]["mean"] == 260.0
    active = power["active_window"]
    assert active["sample_range"] == {"first_index": 1, "last_index": 3}
    assert active["sample_count"] == 3
    assert active["stats"]["power_draw_w"]["mean"] == 400.0
    assert active["stats"]["utilization_gpu_percent"]["mean"] == 50.0
    assert active["stats"]["clocks_sm_mhz"]["mean"] == 2000.0
    assert row["estimated_active_tokens_per_joule"] == pytest.approx(
        row["production_tokens_per_second"] / 400.0
    )

    idle_path = tmp_path / "idle.power.csv"
    _write_power(
        idle_path,
        (100.0, 110.0),
        utilizations=(10.0, 49.0),
    )
    idle = reporter._power_summary(idle_path)
    assert idle["active_window"] == {
        "criterion": "first_to_last_gpu_utilization_at_least_threshold_inclusive",
        "utilization_gpu_threshold_percent": 50.0,
        "sample_range": None,
        "sample_count": 0,
        "stats": {},
    }


def test_batch1_baseline_requires_fixed_sdpa_contract(tmp_path: Path) -> None:
    path = tmp_path / "b1-ordinary-ac0.json"
    case = _case(batch_size=1, ac=0, tps=6800.0, free_gib=4.0, alignment=False)
    payload = _payload(
        batch_size=1,
        case=case,
        alignment=False,
        mtp_attention=None,
    )
    payload["mtp"].pop("attention_implementation")
    _write_json(path, payload)
    _write_power(path.with_suffix(".power.csv"), (450.0, 540.0, 590.0))

    row = reporter._b1_ordinary_row(
        path,
        ac_layers=0,
        global_batch_tokens=262144,
    )

    assert row["candidate"] is False
    assert row["accepted"] is False
    assert row["mtp_attention_implementation"] is None


@pytest.mark.parametrize(
    ("section", "field", "invalid"),
    (
        ("capture", "batch_size", 2),
        ("security", "sensitive_environment_variable_name_matches", 1),
        ("acceptance", "eager_softmax_instances", 1),
        ("acceptance", "flash_backward_instances", 0),
    ),
)
def test_nsys_summary_fails_closed(
    tmp_path: Path, section: str, field: str, invalid: object
) -> None:
    path = tmp_path / "invalid-summary.json"
    summary = _nsys_summary()
    summary[section][field] = invalid
    _write_json(path, summary)

    with pytest.raises(ValueError, match="Nsight summary failed strict acceptance"):
        reporter._validated_nsys_summary(path)


@pytest.mark.parametrize(
    ("batch_size", "mtp_attention"),
    ((2, "eager"), (4, None)),
)
def test_candidate_requires_sdpa_mtp_attention(
    tmp_path: Path, batch_size: int, mtp_attention: str | None
) -> None:
    path = tmp_path / f"b{batch_size}-ordinary-ac24.json"
    case = _case(
        batch_size=batch_size,
        ac=24,
        tps=6200.0,
        free_gib=4.0,
        alignment=False,
    )
    payload = _payload(
        batch_size=batch_size,
        case=case,
        alignment=False,
        mtp_attention=mtp_attention,
    )
    if mtp_attention is None:
        payload["mtp"].pop("attention_implementation")
    _write_json(path, payload)
    _write_power(path.with_suffix(".power.csv"), (450.0, 540.0, 590.0))

    row = reporter._single_row(
        path,
        mode="ordinary",
        global_batch_tokens=262144,
        require_power=True,
        candidate=True,
    )

    assert row["accepted"] is False
    assert row["status"] == "failed"
    assert row["mtp_attention_implementation"] == mtp_attention


def test_experimental_execution_must_be_production_enabled(tmp_path: Path) -> None:
    path = tmp_path / "b2-ordinary-ac0-experimental.json"
    case = _case(batch_size=2, ac=0, tps=9200.0, free_gib=4.0, alignment=False)
    payload = _payload(batch_size=2, case=case, alignment=False)
    payload["experimental_execution"] = {
        "checkpoint_token_branch_only": True,
        "production_enabled": False,
    }
    _write_json(path, payload)
    _write_power(path.with_suffix(".power.csv"), (450.0, 540.0, 590.0))

    row = reporter._single_row(
        path,
        mode="ordinary",
        global_batch_tokens=262144,
        require_power=True,
        candidate=True,
    )

    assert row["accepted"] is False
    assert row["status"] == "experimental-not-production"
    assert row["production_execution_enabled"] is False
    assert row["experimental_execution"] == payload["experimental_execution"]
