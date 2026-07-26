from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "summarize_v4_checkpoint_validation.py"
    spec = importlib.util.spec_from_file_location(
        "summarize_v4_checkpoint_validation",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _load_script()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prepared_fixture(root: Path) -> Path:
    shards = [
        {
            "shard_id": "shard-000000",
            "path": "shard-000000",
            "source_path": "/fixture/extracted/source_a/chunk-000000/validation.jsonl",
            "tensors_sha256": "a" * 64,
            "token_count": 11,
            "sequence_count": 1,
        },
        {
            "shard_id": "shard-000001",
            "path": "shard-000001",
            "source_path": "/fixture/extracted/source_a/chunk-000001/validation.jsonl",
            "tensors_sha256": "b" * 64,
            "token_count": 21,
            "sequence_count": 1,
        },
        {
            "shard_id": "shard-000002",
            "path": "shard-000002",
            "source_path": "/fixture/extracted/source_b/chunk-000000/validation.jsonl",
            "tensors_sha256": "c" * 64,
            "token_count": 31,
            "sequence_count": 1,
        },
    ]
    path = root / "prepared" / "manifest.json"
    _write_json(
        path,
        {
            "schema_version": 2,
            "kind": "twen_prepared_text",
            "dataset_fingerprint": "dataset-fixture-v1",
            "token_count": 63,
            "sequence_count": 3,
            "sequence_length": 32,
            "shards": shards,
        },
    )
    return path


def _evaluation_fixture(
    root: Path,
    *,
    prepared: Path,
    nll_sums: list[float],
    step: int,
    predicted_tokens: list[int] | None = None,
    dtype: str = "bfloat16",
    device_type: str = "cuda",
) -> Path:
    root.mkdir()
    prepared_payload = json.loads(prepared.read_text(encoding="utf-8"))
    shard_tokens = predicted_tokens or [10, 20, 30]
    checkpoint_state = {
        "global_step": step,
        "committed_tokens": step * 100,
        "kind": "milestone",
        "tag": f"step-{step}",
    }
    runtime = {
        "device": "cuda:0" if device_type == "cuda" else "cpu",
        "device_name": "fixture",
    }
    config_fingerprint = "1" * 64
    source_tree_sha256 = "3" * 64
    plan = {
        "schema_version": 1,
        "kind": "twen_nll_evaluation_plan",
        "checkpoint": str(root / "checkpoint"),
        "checkpoint_state": checkpoint_state,
        "prepared_manifest": str(prepared),
        "prepared_manifest_sha256": _sha256(prepared),
        "prepared_dataset_fingerprint": prepared_payload["dataset_fingerprint"],
        "expert_initialization": "donor",
        "roles": ["candidate"],
        "batch_size": 1,
        "device_type": device_type,
        "dtype": dtype,
        "runtime": runtime,
        "config_fingerprint": config_fingerprint,
        "checkpoint_inference_lineage": {
            "mode": "forward_only_lineage_compatible_not_exact_resume",
            "archived_config_sha256": "2" * 64,
            "saved_critical_fingerprint": config_fingerprint,
            "current_preflight_fingerprint": config_fingerprint,
            "saved_source_tree_sha256": source_tree_sha256,
            "current_source_tree_sha256": source_tree_sha256,
            "exact_training_fingerprint_match": True,
            "source_tree_match": True,
        },
    }
    plan["plan_fingerprint"] = _canonical_sha256(plan)
    plan_path = root / "PLAN.json"
    _write_json(plan_path, plan)

    role_fingerprint = _canonical_sha256(
        {"plan_fingerprint": plan["plan_fingerprint"], "role": "candidate"}
    )
    manifest_shards = []
    role_nll_sum = 0.0
    role_tokens = 0
    for index, prepared_shard in enumerate(prepared_payload["shards"]):
        shard_id = prepared_shard["shard_id"]
        shard_fingerprint = _canonical_sha256(
            {
                "role_fingerprint": role_fingerprint,
                "source_shard_id": shard_id,
                "source_tensors_sha256": prepared_shard["tensors_sha256"],
                "sequence_count": prepared_shard["sequence_count"],
                "token_count": prepared_shard["token_count"],
            }
        )
        shard_root = root / "roles" / "candidate" / shard_id
        result = {
            "schema_version": 1,
            "kind": "nll_shard_result",
            "fingerprint": shard_fingerprint,
            "role": "candidate",
            "source_shard_id": shard_id,
            "sequences": 1,
            "nll_sum": nll_sums[index],
            "predicted_tokens": shard_tokens[index],
            "mean_nll": nll_sums[index] / shard_tokens[index],
        }
        result_path = shard_root / "result.json"
        _write_json(result_path, result)
        outputs = [
            {
                "path": "result.json",
                "sha256": _sha256(result_path),
                "size": result_path.stat().st_size,
            }
        ]
        marker = {
            "schema_version": 1,
            "shard_id": shard_id,
            "fingerprint": shard_fingerprint,
            "source_fingerprint": prepared_shard["tensors_sha256"],
            "metadata": {
                "kind": "nll_shard_result",
                "role": "candidate",
                "predicted_tokens": shard_tokens[index],
            },
            "outputs": outputs,
        }
        complete_path = shard_root / "COMPLETE"
        _write_json(complete_path, marker)
        manifest_shards.append(
            {
                "source_shard_id": shard_id,
                "path": f"roles/candidate/{shard_id}",
                "complete_sha256": _sha256(complete_path),
                "outputs": outputs,
            }
        )
        role_nll_sum += nll_sums[index]
        role_tokens += shard_tokens[index]

    mean_nll = role_nll_sum / role_tokens
    manifest = {
        "schema_version": 1,
        "kind": "twen_nll_evaluation",
        "plan_path": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "plan_fingerprint": plan["plan_fingerprint"],
        "run_id": root.name,
        "track": "base",
        "stage": "dense-oracle",
        "expert_initialization": "donor",
        "checkpoint_state": checkpoint_state,
        "runtime": runtime,
        "roles": {
            "candidate": {
                "role": "candidate",
                "sequences": 3,
                "predicted_tokens": role_tokens,
                "nll_sum": role_nll_sum,
                "mean_nll": mean_nll,
                "perplexity": math.exp(mean_nll),
                "shards": manifest_shards,
            }
        },
    }
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    (root / "COMPLETE").write_text(_sha256(manifest_path) + "\n", encoding="ascii")
    return root


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_authenticated_multi_candidate_bundle_uses_token_weighted_source_deltas(
    tmp_path: Path,
) -> None:
    prepared = _prepared_fixture(tmp_path)
    baseline = _evaluation_fixture(
        tmp_path / "baseline",
        prepared=prepared,
        nll_sums=[20.0, 80.0, 90.0],
        step=100,
    )
    step40 = _evaluation_fixture(
        tmp_path / "step40",
        prepared=prepared,
        nll_sums=[19.0, 70.0, 96.0],
        step=40,
    )
    step50 = _evaluation_fixture(
        tmp_path / "step50",
        prepared=prepared,
        nll_sums=[21.0, 76.0, 84.0],
        step=50,
    )
    input_hashes = {
        "prepared": _sha256(prepared),
        "baseline": _tree_hashes(baseline),
        "step40": _tree_hashes(step40),
        "step50": _tree_hashes(step50),
    }
    output = tmp_path / "report"

    result = sweep.generate(
        prepared_manifest=prepared,
        baseline_path=baseline,
        baseline_label="v3",
        candidate_paths=[("step40", step40), ("step50", step50)],
        output=output,
    )

    assert set(
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    ) == {
        "COMPLETE",
        "MANIFEST.json",
        "REPORT.zh-CN.md",
        "charts/overall-nll-delta.svg",
        "charts/per-source-nll-delta.svg",
        "summary.json",
    }
    assert (output / "COMPLETE").read_text().strip() == _sha256(output / "MANIFEST.json")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["selection"]["label"] == "step50"
    assert summary["baseline"]["overall"]["mean_nll"] == pytest.approx(190.0 / 60.0)
    assert summary["candidates"][0]["overall"][
        "nll_delta_candidate_minus_baseline"
    ] == pytest.approx((185.0 - 190.0) / 60.0)
    step40_sources = {row["source"]: row for row in summary["candidates"][0]["sources"]}
    assert step40_sources["source_a"]["mean_nll"] == pytest.approx(89.0 / 30.0)
    assert step40_sources["source_a"]["nll_delta_candidate_minus_baseline"] == pytest.approx(
        (89.0 - 100.0) / 30.0
    )
    assert (
        summary["candidates"][0]["evaluation_harness"]["checkpoint_inference_lineage"][
            "current_source_tree_sha256"
        ]
        == "3" * 64
    )
    report = (output / "REPORT.zh-CN.md").read_text(encoding="utf-8")
    assert "负值表示改善" in report
    assert "评测 harness 身份" in report
    assert "||" not in report
    assert "<svg" in (output / "charts/overall-nll-delta.svg").read_text(encoding="utf-8")
    assert result["selection"]["label"] == "step50"
    assert _sha256(prepared) == input_hashes["prepared"]
    assert _tree_hashes(baseline) == input_hashes["baseline"]
    assert _tree_hashes(step40) == input_hashes["step40"]
    assert _tree_hashes(step50) == input_hashes["step50"]


def test_single_candidate_cli_writes_complete_bundle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _prepared_fixture(tmp_path)
    baseline = _evaluation_fixture(
        tmp_path / "baseline",
        prepared=prepared,
        nll_sums=[20.0, 80.0, 90.0],
        step=100,
    )
    candidate = _evaluation_fixture(
        tmp_path / "candidate",
        prepared=prepared,
        nll_sums=[19.0, 70.0, 96.0],
        step=62,
    )
    output = tmp_path / "single"

    exit_code = sweep.main(
        [
            "--prepared-manifest",
            str(prepared),
            "--baseline",
            str(baseline / "manifest.json"),
            "--candidate",
            f"step62={candidate}",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["selection"]["label"] == "step62"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert [row["label"] for row in summary["candidates"]] == ["step62"]


@pytest.mark.parametrize("corruption", ["top-complete", "shard-result"])
def test_authentication_rejects_corrupt_completed_evidence(
    tmp_path: Path,
    corruption: str,
) -> None:
    prepared = _prepared_fixture(tmp_path)
    baseline = _evaluation_fixture(
        tmp_path / "baseline",
        prepared=prepared,
        nll_sums=[20.0, 80.0, 90.0],
        step=100,
    )
    candidate = _evaluation_fixture(
        tmp_path / "candidate",
        prepared=prepared,
        nll_sums=[19.0, 70.0, 96.0],
        step=62,
    )
    if corruption == "top-complete":
        (candidate / "COMPLETE").write_text("0" * 64 + "\n", encoding="ascii")
    else:
        result = candidate / "roles" / "candidate" / "shard-000000" / "result.json"
        result.write_text(result.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(sweep.SweepError, match=r"COMPLETE|hash/size"):
        sweep.build_summary(
            prepared_manifest=prepared,
            baseline_path=baseline,
            baseline_label="v3",
            candidate_paths=[("step62", candidate)],
        )


def test_unsigned_plan_change_and_authenticated_numerics_mismatch_are_rejected(
    tmp_path: Path,
) -> None:
    prepared = _prepared_fixture(tmp_path)
    baseline = _evaluation_fixture(
        tmp_path / "baseline",
        prepared=prepared,
        nll_sums=[20.0, 80.0, 90.0],
        step=100,
    )
    unsigned = _evaluation_fixture(
        tmp_path / "unsigned",
        prepared=prepared,
        nll_sums=[19.0, 70.0, 96.0],
        step=40,
    )
    plan_path = unsigned / "PLAN.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["batch_size"] = 2
    _write_json(plan_path, plan)
    with pytest.raises(sweep.SweepError, match="PLAN fingerprint"):
        sweep.build_summary(
            prepared_manifest=prepared,
            baseline_path=baseline,
            baseline_label="v3",
            candidate_paths=[("unsigned", unsigned)],
        )

    float32 = _evaluation_fixture(
        tmp_path / "float32",
        prepared=prepared,
        nll_sums=[19.0, 70.0, 96.0],
        step=50,
        dtype="float32",
    )
    with pytest.raises(sweep.SweepError, match="batch/device/dtype"):
        sweep.build_summary(
            prepared_manifest=prepared,
            baseline_path=baseline,
            baseline_label="v3",
            candidate_paths=[("float32", float32)],
        )


def test_equal_total_but_different_per_shard_token_counts_are_rejected(
    tmp_path: Path,
) -> None:
    prepared = _prepared_fixture(tmp_path)
    baseline = _evaluation_fixture(
        tmp_path / "baseline",
        prepared=prepared,
        nll_sums=[20.0, 80.0, 90.0],
        step=100,
    )
    candidate = _evaluation_fixture(
        tmp_path / "candidate",
        prepared=prepared,
        nll_sums=[18.0, 84.0, 90.0],
        predicted_tokens=[9, 21, 30],
        step=62,
    )

    with pytest.raises(sweep.SweepError, match="shard-000000 processed different"):
        sweep.build_summary(
            prepared_manifest=prepared,
            baseline_path=baseline,
            baseline_label="v3",
            candidate_paths=[("step62", candidate)],
        )
