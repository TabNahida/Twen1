from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "prepare_base_v2_500m.py"
    spec = importlib.util.spec_from_file_location("prepare_base_v2_500m", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


pipeline = _load_script()


def test_layout_never_aliases_invalidated_or_frozen_directories(tmp_path: Path) -> None:
    layout = pipeline.Layout.repository_defaults(tmp_path)

    assert layout.extracted == (tmp_path / "data/base-v2-500m").resolve()
    assert layout.extracted not in {
        (tmp_path / "data/base-v1").resolve(),
        (tmp_path / "data/base-v2").resolve(),
        (tmp_path / "data/base-v3").resolve(),
    }
    assert (
        layout.frozen_validation_manifest
        == (tmp_path / "data/base-v3/corpus-manifest.json").resolve()
    )
    assert layout.audit(0).name == "base-v2-500m-audit-pass-000"
    assert layout.filtered(1).name == "base-v2-500m-filtered-pass-001"
    assert layout.refill_plan(1).name == "base-v2-500m-refill-plan-001"
    assert layout.refill_raw(1).name == "base-v2-500m-refill-raw-001"
    assert layout.performance_gate.name == "rtx5090-base-dense-utilization-report.json"
    assert layout.performance_approval.name == (
        "rtx5090-base-dense-utilization-report.approval.json"
    )
    assert layout.performance_manifest.name == (
        "rtx5090-base-dense-utilization-report.MANIFEST.json"
    )
    assert layout.performance_complete.name == "rtx5090-base-dense-utilization-report.COMPLETE"


def test_progress_eta_uses_only_tokens_committed_in_current_attempt() -> None:
    report = pipeline.progress_eta(
        current=220,
        total=520,
        baseline=120,
        elapsed_seconds=20.0,
    )

    assert report["rate_per_second"] == 5.0
    assert report["eta_seconds"] == 60.0
    assert report["fraction"] == 220 / 520


def test_committed_build_tokens_ignores_incomplete_chunks(tmp_path: Path) -> None:
    extracted = tmp_path / "data/base-v2-500m"
    complete_chunk = extracted / "extracted/source/chunk-000000"
    incomplete_chunk = extracted / "extracted/source/chunk-000001"
    complete_chunk.mkdir(parents=True)
    incomplete_chunk.mkdir(parents=True)
    (complete_chunk / "COMPLETE").write_text("{}", encoding="utf-8")
    (complete_chunk / "chunk.json").write_text(
        json.dumps({"train_tokens": 90, "validation_tokens": 10}), encoding="utf-8"
    )
    (incomplete_chunk / "chunk.json").write_text(
        json.dumps({"train_tokens": 900, "validation_tokens": 100}), encoding="utf-8"
    )

    assert pipeline._committed_build_tokens(extracted) == 100


def test_plan_is_data_only_and_preserves_fallback_network_policy(tmp_path: Path) -> None:
    layout = pipeline.Layout.repository_defaults(tmp_path)
    args = pipeline._parser().parse_args(
        ["--root", str(tmp_path), "--action", "plan", "--python", "/venv/python"]
    )

    plan = pipeline.planned_commands(layout, args)

    command = plan["first_build_command"]
    assert command[:3] == ["/venv/python", "-m", "twen"]
    assert command[command.index("--profile") + 1] == "sparse"
    assert command[command.index("--network-policy") + 1] == "fallback"
    assert plan["training_started"] is False
    assert plan["gpu_kd_started"] is False
    assert plan["refill_policy"]["clean_guard_ratio"] == 0.02
    assert plan["refill_policy"]["survival_guard_points"] == 0.01
    assert plan["refill_policy"]["per_source_and_aggregate_quotas_are_hard_gates"] is True


def test_proxy_is_redacted_from_persistent_plan(tmp_path: Path) -> None:
    layout = pipeline.Layout.repository_defaults(tmp_path)
    args = pipeline._parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "--action",
            "plan",
            "--proxy",
            "http://user:secret@example.invalid:7890",
        ]
    )

    plan = pipeline.planned_commands(layout, args)

    command = plan["first_build_command"]
    assert command[command.index("--proxy") + 1] == "<redacted>"
    assert "secret" not in json.dumps(plan)


def test_completed_phase_has_sha_bound_complete_marker(tmp_path: Path) -> None:
    layout = pipeline.Layout.repository_defaults(tmp_path)

    pipeline._begin_phase(layout, "preflight", None)
    pipeline._finish_phase(layout, status="complete", exit_code=0)

    status = pipeline._read_status(layout)
    phase = status["history"][0]
    marker = Path(phase["phase_marker"]["path"])
    assert marker.name == "000-preflight.COMPLETE.json"
    assert marker.is_file()
    assert phase["phase_marker"]["sha256"] == pipeline._sha256(marker)
    assert phase["eta_seconds"] == 0.0


def test_command_log_drains_carriage_return_progress(tmp_path: Path) -> None:
    layout = pipeline.Layout.repository_defaults(tmp_path)
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.write('half\\r'); sys.stdout.flush(); print('done')",
    ]

    pipeline._run_command(layout, "progress-probe", command, outputs=[])

    status = pipeline._read_status(layout)
    phase = status["history"][0]
    log = Path(phase["log"]).read_text(encoding="utf-8")
    assert "half" in log
    assert "done" in log
    assert phase["status"] == "complete"


def test_performance_gate_requires_coordinated_safe_recommendation(tmp_path: Path) -> None:
    layout = pipeline.Layout.repository_defaults(tmp_path)
    layout.performance_gate.parent.mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/benchmark_full_dense_graph.py").write_text(
        "# fixed benchmark\n", encoding="utf-8"
    )
    (tmp_path / "src/twen/modeling").mkdir(parents=True)
    (tmp_path / "src/twen/modeling/mtp.py").write_text("# fixed MTP\n", encoding="utf-8")
    ordinary_source = tmp_path / "ordinary.json"
    alignment_source = tmp_path / "alignment.json"
    ordinary_source.write_text("{}\n", encoding="utf-8")
    alignment_source.write_text("{}\n", encoding="utf-8")

    def row(label: str, mode: str, batch_size: int, source: Path) -> dict:
        return {
            "label": label,
            "mode": mode,
            "batch_size": batch_size,
            "logical_tokens": batch_size * 4096,
            "activation_checkpoint_layer_count": 16 if mode == "ordinary" else 24,
            "status": "ok",
            "accepted": True,
            "production_tokens_per_second": 7000.0,
            "minimum_estimated_headroom_gib": 3.5,
            "minimum_nvml_physical_free_gib": 3.75,
            "health": {
                "ok": True,
                "loss_finite": True,
                "gradients_finite": True,
                "missing_gradient_tensors": 0,
                "nonfinite_gradient_tensors": 0,
                "present_gradient_tensor_counts": [72],
            },
            "no_optimizer_created_or_stepped": True,
            "optimizer_state_reserve_gib": 1.5,
            "mtp_loss_weight": 0.1,
            "mtp_attention_implementation": "sdpa",
            "teacher_cpu_offload": True,
            "source": {"path": str(source), "sha256": pipeline._sha256(source)},
        }

    def report(ordinary_batch: int, alignment_batch: int) -> dict:
        ordinary_label = f"b{ordinary_batch}-ordinary-ac16"
        alignment_label = f"b{alignment_batch}-alignment-ac24"
        return {
            "schema_version": 2,
            "kind": "rtx5090_base_dense_utilization_report",
            "read_only_report_generation": True,
            "no_optimizer_created_by_report": True,
            "accepted": True,
            "global_batch_tokens": 262_144,
            "recommendation": {
                "batch_size": ordinary_batch,
                "ordinary_case": ordinary_label,
                "alignment_case": alignment_label,
                "tokens_per_second": 6900.0,
            },
            "source_provenance": {
                "files": {
                    "benchmark": {
                        "path": str(tmp_path / "scripts/benchmark_full_dense_graph.py"),
                        "sha256": pipeline._sha256(
                            tmp_path / "scripts/benchmark_full_dense_graph.py"
                        ),
                    },
                    "mtp": {
                        "path": str(tmp_path / "src/twen/modeling/mtp.py"),
                        "sha256": pipeline._sha256(tmp_path / "src/twen/modeling/mtp.py"),
                    },
                }
            },
            "rows": [
                row(ordinary_label, "ordinary", ordinary_batch, ordinary_source),
                row(alignment_label, "alignment", alignment_batch, alignment_source),
            ],
        }

    layout.performance_gate.write_text(
        json.dumps(report(2, 4)),
        encoding="utf-8",
    )

    assert pipeline._performance_gate(layout)["ready"] is False

    layout.performance_gate.write_text(
        json.dumps(report(4, 4)),
        encoding="utf-8",
    )
    report_value = json.loads(layout.performance_gate.read_text(encoding="utf-8"))
    prefix = layout.performance_gate.with_suffix("")
    bundle_paths = {
        "report_json": layout.performance_gate,
        "report_markdown": layout.performance_gate.with_suffix(".md"),
        "throughput_memory_svg": prefix.with_name(prefix.name + "-throughput-memory.svg"),
        "power_svg": prefix.with_name(prefix.name + "-power.svg"),
        "utilization_svg": prefix.with_name(prefix.name + "-utilization.svg"),
    }
    for name, path in bundle_paths.items():
        if name != "report_json":
            path.write_text(f"{name}\n", encoding="utf-8")
    provenance_sha = pipeline._canonical_json_sha256(report_value["source_provenance"])
    pipeline._atomic_json(
        layout.performance_manifest,
        {
            "schema_version": 1,
            "kind": "twen_rtx5090_base_dense_utilization_report_bundle",
            "accepted": True,
            "recommendation": report_value["recommendation"],
            "source_provenance_sha256": provenance_sha,
            "files": {
                name: {
                    "path": str(path),
                    "size": path.stat().st_size,
                    "sha256": pipeline._sha256(path),
                }
                for name, path in bundle_paths.items()
            },
        },
    )
    pipeline._atomic_json(
        layout.performance_complete,
        {
            "schema_version": 1,
            "kind": "twen_rtx5090_base_dense_utilization_report_complete",
            "manifest": layout.performance_manifest.name,
            "manifest_sha256": pipeline._sha256(layout.performance_manifest),
            "report": {
                "path": str(layout.performance_gate),
                "sha256": pipeline._sha256(layout.performance_gate),
            },
            "accepted": True,
            "recommendation": report_value["recommendation"],
            "source_provenance_sha256": provenance_sha,
        },
    )
    gate = pipeline._performance_gate(layout)
    assert gate["ready"] is False
    assert "approval is missing" in gate["reason"]

    approved = pipeline.write_performance_approval(layout)
    assert approved["performance_gate"]["ready"] is True
    assert layout.performance_approval.is_file()

    (tmp_path / "src/twen/modeling/mtp.py").write_text("# changed again\n", encoding="utf-8")
    assert pipeline._performance_gate(layout)["ready"] is False


def test_legacy_batch2_named_report_is_explicitly_rejected(tmp_path: Path) -> None:
    layout = pipeline.Layout.repository_defaults(tmp_path)
    layout.legacy_performance_gate.parent.mkdir(parents=True)
    layout.legacy_performance_gate.write_text('{"accepted": true}\n', encoding="utf-8")

    gate = pipeline._performance_report_contract(layout)

    assert gate["ready"] is False
    assert gate["legacy_report"]["accepted_for_pipeline"] is False
    assert "explicitly rejected" in gate["legacy_report"]["reason"]
