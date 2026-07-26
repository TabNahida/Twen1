from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reporter = _load(
    ROOT / "scripts/generate_expanded_production_sweep_report.py",
    "generate_expanded_production_sweep_report_test",
)


def test_locked_report_reproduces_recommendation_formulas_and_gates() -> None:
    report = reporter.build_report(ROOT)
    recommendation = report["recommendation"]
    assert recommendation["batch_size"] == 1
    assert recommendation["ordinary_case"] == "b1-ordinary-ac0"
    assert recommendation["alignment_case"] == "b1-alignment-ac8"
    assert recommendation["tokens_per_second"] == pytest.approx(
        6263.998075874394,
        abs=1e-9,
    )
    assert report["comparisons"]["b1_ordinary_vs_safe_b2_gpu_percent"] == pytest.approx(
        9.964459634727763,
        abs=1e-9,
    )
    assert report["numerical_admission"]["expanded"]["status"] == "pass"
    assert report["numerical_admission"]["folded"]["status"] == "fail_experimental_only"

    rows = {row["key"]: row for row in report["rows"]}
    assert rows["b1_ordinary_final_n10"]["accepted"] is True
    assert rows["b1_alignment_final_n10"]["accepted"] is True
    assert rows["b2_ordinary_outer0_inner24"]["accepted"] is True
    assert rows["b2_ordinary_outer8_inner8"]["accepted"] is False
    assert rows["b1_alignment_outer0_inner24_unsafe"]["accepted"] is False
    assert rows["b2_ordinary_outer4_inner4_failure"]["status"] == "capacity-failure"
    assert all(
        row["health"]["present_gradient_tensor_counts"] == [72]
        for row in report["rows"]
        if row["benchmark_ok"]
    )
    assert all(row["power"]["high_utilization"]["sample_count"] > 0 for row in report["rows"])


def test_generation_is_idempotent_preserves_approval_and_passes_pipeline_contract(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "expanded-sweep"
    prefix = tmp_path / "canonical/rtx5090-base-dense-utilization-report"
    approval = prefix.with_name(prefix.name + ".approval.json")
    approval.parent.mkdir(parents=True)
    approval.write_bytes(b"locked approval sentinel\n")

    arguments = [
        "--root",
        str(ROOT),
        "--report-dir",
        str(report_dir),
        "--canonical-prefix",
        str(prefix),
    ]
    assert reporter.main(arguments) == 0
    first = reporter._tree_identities(report_dir)
    assert reporter.main(arguments) == 0
    assert reporter._tree_identities(report_dir) == first
    assert approval.read_bytes() == b"locked approval sentinel\n"

    expected_sweep = {
        "COMPLETE",
        "MANIFEST.json",
        "REPORT.zh-CN.md",
        "summary.json",
        "charts/high-utilization-power.svg",
        "charts/throughput-headroom.svg",
        "charts/timing-mixture.svg",
    }
    assert set(reporter._tree_identities(report_dir)) == expected_sweep
    assert all((report_dir / item).stat().st_size > 0 for item in expected_sweep)

    pipeline = _load(
        ROOT / "scripts/prepare_base_v2_500m.py",
        "prepare_base_v2_500m_expanded_report_test",
    )
    layout = pipeline.Layout.repository_defaults(ROOT)
    layout = dataclasses.replace(
        layout,
        performance_gate=prefix.with_suffix(".json"),
        performance_manifest=prefix.with_name(prefix.name + ".MANIFEST.json"),
        performance_complete=prefix.with_name(prefix.name + ".COMPLETE"),
    )
    contract = pipeline._performance_report_contract(layout)
    assert contract["ready"] is True, contract["reason"]
    assert contract["bundle"]["ready"] is True


def test_locked_input_sha_and_numerical_admission_fail_closed(tmp_path: Path) -> None:
    corrupted = tmp_path / "input.json"
    corrupted.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 changed"):
        reporter._require_sha(corrupted, "0" * 64, "fixture")

    selected = ROOT / reporter.CASE_SPECS[0].relative_json
    payload = json.loads(selected.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(payload)
    tampered["experimental_execution"]["numerical_admission"]["expanded_selective_checkpoint"][
        "status"
    ] = "fail"
    with pytest.raises(ValueError, match="expanded PASS / folded FAIL"):
        reporter._validate_numerical(tampered, "tampered")
