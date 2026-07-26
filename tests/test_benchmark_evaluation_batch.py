from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_evaluation_batch.py"
SPEC = importlib.util.spec_from_file_location("benchmark_evaluation_batch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_aggregate_reports_speedup_memory_and_nll_consistency() -> None:
    cases = [
        {
            "status": "ok",
            "batch_size": 1,
            "predicted_tokens_per_second": 100.0,
            "mean_nll": 2.0,
            "peak_allocated_bytes": 10,
            "peak_reserved_bytes": 12,
        },
        {
            "status": "ok",
            "batch_size": 2,
            "predicted_tokens_per_second": 150.0,
            "mean_nll": 2.0 + 1e-7,
            "peak_allocated_bytes": 20,
            "peak_reserved_bytes": 24,
        },
        {
            "status": "ok",
            "batch_size": 2,
            "predicted_tokens_per_second": 170.0,
            "mean_nll": 2.0 + 2e-7,
            "peak_allocated_bytes": 22,
            "peak_reserved_bytes": 26,
        },
        {
            "status": "ok",
            "batch_size": 1,
            "predicted_tokens_per_second": 120.0,
            "mean_nll": 2.0,
            "peak_allocated_bytes": 11,
            "peak_reserved_bytes": 13,
        },
    ]

    result = benchmark._aggregate(cases)

    assert result["1"]["median_predicted_tokens_per_second"] == 110.0
    assert result["2"]["median_predicted_tokens_per_second"] == 160.0
    assert result["2"]["max_peak_allocated_bytes"] == 22
    assert result["batch2_over_batch1_speedup"] == 160.0 / 110.0
    assert result["nll_consistent_at_1e_5"]


def test_benchmark_source_contains_no_optimizer_step_or_backward() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert ".backward(" not in source
    assert "optimizer.step(" not in source
    assert '"no_optimizer_steps": True' in source
    assert '"no_backward": True' in source
