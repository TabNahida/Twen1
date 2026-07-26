from __future__ import annotations

import contextlib
import io
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import twen.kd_orchestration as orchestration
from twen.v2_finalizer import FinalizationOptions, _authenticate_kd_orchestration


def _identity(path: Path) -> dict[str, object]:
    return orchestration._identity(path)


def _config(tmp_path: Path, *, training_micro_batch_size: int = 1) -> SimpleNamespace:
    teacher = SimpleNamespace(
        model_id="Qwen/Qwen3.5-9B-Base",
        revision="a" * 40,
        manifest_sha256="b" * 64,
        local_path=str(tmp_path / "teacher"),
    )
    tokenizer = SimpleNamespace(manifest_sha256="c" * 64)
    return SimpleNamespace(
        sources=SimpleNamespace(teacher=teacher, tokenizer=tokenizer),
        data=SimpleNamespace(micro_batch_size=training_micro_batch_size),
    )


def _entry() -> SimpleNamespace:
    return SimpleNamespace(
        shard_id="shard-000000",
        tensors_sha256="d" * 64,
        global_sample_start=0,
        global_sample_end=2,
        global_token_start=0,
        global_token_end=8_000,
        sequence_count=2,
        token_count=8_000,
    )


def _prepared(*, token_count: int = 8_000) -> SimpleNamespace:
    entry = _entry()
    sequence_count = math.ceil(token_count / 4096)
    entry.token_count = token_count
    entry.global_token_end = token_count
    entry.sequence_count = sequence_count
    entry.global_sample_end = sequence_count
    return SimpleNamespace(
        shards=(entry,),
        sequence_count=sequence_count,
        sequence_length=4096,
        token_count=token_count,
        tokenizer_sha256="c" * 64,
        dataset_fingerprint="e" * 64,
        lineage=None,
    )


def test_storage_estimate_uses_actual_padded_geometry() -> None:
    prepared = _prepared()

    result = orchestration.estimate_kd_storage(prepared)

    assert result["bytes_per_padded_position"] == 665
    assert result["padded_positions"] == 2 * 4096
    assert result["tensor_payload_bytes"] == 2 * 4096 * 665
    assert result["unpadded_lower_bound_bytes"] == 8_000 * 665
    assert result["padding_overhead_bytes"] == (2 * 4096 - 8_000) * 665


def test_generate_command_uses_kd_batch_two_independent_of_training_b1(
    tmp_path: Path,
) -> None:
    layout = orchestration.Layout.repository_defaults(tmp_path)
    inputs = SimpleNamespace(config=_config(tmp_path, training_micro_batch_size=1))

    command = orchestration.generate_command(
        layout,
        inputs,
        python=Path("/venv/python"),
        batch_size=2,
        logits_chunk_tokens=64,
    )

    assert command[:5] == ["/venv/python", "-m", "twen", "data", "generate-kd"]
    assert command[command.index("--batch-size") + 1] == "2"
    assert command[command.index("--logits-chunk-tokens") + 1] == "64"
    assert "train" not in command
    assert not any("optimizer" in item for item in command)


def test_child_python_path_preserves_virtualenv_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venv_python = tmp_path / ".venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(sys.executable).resolve())
    monkeypatch.chdir(tmp_path)

    selected = orchestration._absolute_python_without_resolving(Path(".venv/bin/python"))

    assert selected == venv_python
    assert selected.is_symlink()
    assert selected.resolve() == Path(sys.executable).resolve()


def test_repository_kd_recommendation_is_sha_bound_optimizer_free_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    layout = orchestration.Layout.repository_defaults(root)

    benchmark, full_run = orchestration._validate_benchmark_evidence(
        layout,
        batch_size=2,
        logits_chunk_tokens=64,
    )

    assert benchmark["no_optimizer_steps"] is True
    assert benchmark["measurement"]["input_tokens_per_second"] == pytest.approx(10_425.424661178547)
    assert full_run["wall_tokens_per_second"] == pytest.approx(9_042.26808318264)
    assert full_run["no_optimizer_steps"] is True


def _write_pipeline_fixture(
    layout: orchestration.Layout,
    *,
    ready: bool = True,
) -> tuple[SimpleNamespace, dict[str, object]]:
    layout.prepared_manifest.parent.mkdir(parents=True)
    layout.prepared_manifest.write_text('{"prepared": true}\n', encoding="utf-8")
    audit = layout.root / "artifacts/data/audit/attestation.json"
    audit.parent.mkdir(parents=True)
    audit.write_text('{"audit": true}\n', encoding="utf-8")
    audit_identity = _identity(audit)
    gates = {"exact_duplicates": {"passed": ready}}
    prepared = _prepared(token_count=500_000_000)
    prepared.lineage = {
        "kind": "authenticated_extracted_corpus",
        "role": "train",
        "ready_for_training": ready,
        "research_only": not ready,
        "pending_audits": [] if ready else ["exact_duplicates"],
        "audit_attestation": {
            "path": str(audit.resolve()),
            "sha256": audit_identity["sha256"],
            "attestation_fingerprint": "f" * 64,
            "bound_as": "candidate",
            "ready_for_training": ready,
            "gates": gates,
        },
    }
    prepared_identity = _identity(layout.prepared_manifest)
    status_path = layout.pipeline_complete.parent / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": orchestration.PIPELINE_STATUS_KIND,
                "status": "complete",
                "training_started": False,
                "gpu_kd_started": False,
                "prepared_manifest": prepared_identity,
                "accepted_attestation": audit_identity,
            }
        ),
        encoding="utf-8",
    )
    status_identity = _identity(status_path)
    layout.pipeline_complete.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": orchestration.PIPELINE_COMPLETE_KIND,
                "training_started": False,
                "gpu_kd_started": False,
                "prepared_manifest": prepared_identity,
                "accepted_attestation": audit_identity,
                "status": status_identity,
            }
        ),
        encoding="utf-8",
    )
    return prepared, {
        "ready_for_training": ready,
        "attestation_fingerprint": "f" * 64,
        "gates": gates,
    }


def test_pipeline_gate_binds_real_prepared_sha_and_all_pass_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = orchestration.Layout.repository_defaults(tmp_path)
    prepared, audit = _write_pipeline_fixture(layout)
    monkeypatch.setattr(orchestration, "validate_base_audit_attestation", lambda _path: audit)

    result = orchestration._validate_pipeline_contract(
        layout,
        prepared_identity=_identity(layout.prepared_manifest),
        prepared=prepared,
    )

    assert result[2]["value"]["ready_for_training"] is True
    layout.prepared_manifest.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(orchestration.KDOrchestrationError, match="identity mismatch"):
        orchestration._validate_pipeline_contract(
            layout,
            prepared_identity=_identity(layout.prepared_manifest),
            prepared=prepared,
        )


def test_pipeline_gate_rejects_non_ready_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = orchestration.Layout.repository_defaults(tmp_path)
    prepared, audit = _write_pipeline_fixture(layout, ready=False)
    monkeypatch.setattr(orchestration, "validate_base_audit_attestation", lambda _path: audit)

    with pytest.raises(orchestration.KDOrchestrationError, match="not an audit-attested"):
        orchestration._validate_pipeline_contract(
            layout,
            prepared_identity=_identity(layout.prepared_manifest),
            prepared=prepared,
        )


def test_progress_counts_only_compatible_final_complete_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = orchestration.Layout.repository_defaults(tmp_path)
    prepared = _prepared()
    config = _config(tmp_path)
    final = layout.output_root / "shard-000000"
    final.mkdir(parents=True)
    (final / "COMPLETE").write_text("{}\n", encoding="utf-8")
    (layout.output_root / "shard-000001.incomplete").mkdir()
    manifest = SimpleNamespace(
        teacher_model_id=config.sources.teacher.model_id,
        teacher_revision=config.sources.teacher.revision,
        teacher_model_sha256=config.sources.teacher.manifest_sha256,
        generator_source_sha256=orchestration.KD_GENERATOR_SOURCE_SHA256,
        tokenizer_sha256=config.sources.tokenizer.manifest_sha256,
        dataset_fingerprint=prepared.dataset_fingerprint,
        source_shard_id=prepared.shards[0].shard_id,
        source_tensors_sha256=prepared.shards[0].tensors_sha256,
        global_sample_start=0,
        global_sample_end=2,
        global_token_start=0,
        global_token_end=8_000,
        sequence_count=2,
        sequence_length=4096,
        token_count=8_000,
    )
    calls: list[bool] = []

    def validate(*_args, **kwargs):
        calls.append(kwargs["verify_checksum"])
        return manifest

    monkeypatch.setattr(orchestration, "validate_kd_shard", validate)

    progress = orchestration.scan_progress(layout, prepared, config, verify_checksums=True)

    assert progress["completed_shards"] == 1
    assert progress["completed_tokens"] == 8_000
    assert progress["percent"] == 100.0
    assert calls == [True]


def test_single_instance_lock_is_nonblocking(tmp_path: Path) -> None:
    lock = tmp_path / "orchestration.lock"
    with (
        orchestration._exclusive_lock(lock),
        pytest.raises(orchestration.KDOrchestrationError, match="another"),
        orchestration._exclusive_lock(lock),
    ):
        pass


def test_preexisting_stop_returns_75_before_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = orchestration.Layout.repository_defaults(tmp_path)
    layout.stop_file.parent.mkdir(parents=True)
    layout.stop_file.touch()
    called = False

    def authenticate(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("must not authenticate after persistent STOP")

    monkeypatch.setattr(orchestration, "authenticate_inputs", authenticate)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        code = orchestration.main(
            [
                "--root",
                str(tmp_path),
                "--action",
                "run",
                "--acknowledge-gpu-kd",
            ]
        )

    assert code == 75
    assert called is False
    assert json.loads(stdout.getvalue())["stopped"] is True


def test_phase_runner_tees_complete_child_output_to_phase_and_console_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = orchestration.Layout.repository_defaults(tmp_path)
    monkeypatch.setattr(orchestration, "_update_running_progress", lambda *_a, **_k: None)

    result = orchestration._run_phase(
        layout,
        SimpleNamespace(),
        name="log-probe",
        command=[
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('half\\r'); print('done')",
        ],
        poll_seconds=0.1,
        baseline_tokens=0,
    )

    assert result.exit_code == 0
    assert "half" in result.log_path.read_text(encoding="utf-8")
    assert "done" in result.log_path.read_text(encoding="utf-8")
    console = (layout.state_root / "console.log").read_text(encoding="utf-8")
    assert "half" in console
    assert "done" in console


def _authenticated_inputs(
    layout: orchestration.Layout, prepared: SimpleNamespace
) -> orchestration.AuthenticatedInputs:
    layout.prepared_manifest.parent.mkdir(parents=True, exist_ok=True)
    layout.prepared_manifest.write_text("{}\n", encoding="utf-8")
    layout.pipeline_complete.parent.mkdir(parents=True, exist_ok=True)
    layout.pipeline_complete.write_text("{}\n", encoding="utf-8")
    audit = layout.root / "audit.json"
    audit.write_text("{}\n", encoding="utf-8")
    config_path = layout.base_config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("config\n", encoding="utf-8")
    return orchestration.AuthenticatedInputs(
        prepared=prepared,
        prepared_identity=_identity(layout.prepared_manifest),
        audit={},
        audit_identity=_identity(audit),
        pipeline_complete_identity=_identity(layout.pipeline_complete),
        pipeline_status_identity=_identity(layout.pipeline_complete),
        config=_config(layout.root),
        config_identity=_identity(config_path),
        teacher_source={"verified_all_download_artifacts": True},
        benchmark={"no_optimizer_steps": True},
        full_run_benchmark={"no_optimizer_steps": True},
    )


def test_successful_orchestration_writes_sha_bound_complete_without_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = orchestration.Layout.repository_defaults(tmp_path)
    orchestration._write_status(
        layout,
        {
            "schema_version": orchestration.SCHEMA_VERSION,
            "kind": orchestration.STATUS_KIND,
            "status": "failed",
            "attempt": 2,
            "history": [],
            "error": "stale failure from attempt 2",
        },
    )
    prepared = _prepared(token_count=500_000_000)
    inputs = _authenticated_inputs(layout, prepared)
    progress = {
        "completed_shards": 1,
        "total_shards": 1,
        "completed_tokens": 500_000_000,
        "total_tokens": 500_000_000,
        "completed_sequences": 2,
        "total_sequences": 2,
        "committed_output_bytes": 1,
        "fraction": 1.0,
        "percent": 100.0,
        "remaining_tokens": 0,
        "checksum_mode": "full",
    }
    monkeypatch.setattr(orchestration, "scan_progress", lambda *_a, **_k: progress)
    kd_manifest = layout.output_root / "manifest.json"
    kd_manifest.parent.mkdir(parents=True)
    kd_manifest.write_text('{"kd": true}\n', encoding="utf-8")
    monkeypatch.setattr(
        orchestration,
        "_verify_final_output",
        lambda *_a, **_k: (SimpleNamespace(), _identity(kd_manifest)),
    )
    phases: list[str] = []

    def runner(_layout, _inputs, *, name, command, **_kwargs):
        phases.append(name)
        log = layout.state_root / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"{name} complete\n", encoding="utf-8")
        with (layout.state_root / "console.log").open("a", encoding="utf-8") as console:
            console.write(f"{name} complete\n")
        return orchestration.PhaseResult(
            name=name,
            command=tuple(command),
            log_path=log,
            exit_code=0,
            started_at="2026-01-01T00:00:00+00:00",
            ended_at="2026-01-01T00:01:00+00:00",
            elapsed_seconds=60.0,
        )

    result = orchestration.run_orchestration(
        layout,
        inputs,
        python=Path("/venv/python"),
        batch_size=2,
        logits_chunk_tokens=64,
        minimum_free_after_gib=0.0,
        poll_seconds=0.1,
        phase_runner=runner,
    )

    assert phases == ["generate-kd", "index-kd"]
    assert result["optimizer_created"] is False
    complete = json.loads((layout.state_root / "COMPLETE").read_text())
    assert complete["kind"] == orchestration.COMPLETE_KIND
    assert complete["optimizer_created"] is False
    assert complete["training_started"] is False
    assert "error" not in json.loads(
        (layout.state_root / "status.json").read_text(encoding="utf-8")
    )
    assert orchestration.verify_orchestration_complete(layout) is not None
    options = FinalizationOptions.repository_defaults(
        root=tmp_path,
        mtp_loss_weight=0.1,
        adapter_lr=2e-4,
        router_lr=1e-3,
        lora_lr=2e-4,
        scale_lr=1e-3,
        quality_cooldown_prepared_manifest="cooldown/prepared/manifest.json",
        quality_cooldown_kd_manifest="cooldown/kd/manifest.json",
    )
    authenticated = _authenticate_kd_orchestration(
        options,
        prepared_identity={
            "path": layout.prepared_manifest.relative_to(tmp_path).as_posix(),
            "size": layout.prepared_manifest.stat().st_size,
            "sha256": orchestration.sha256_file(layout.prepared_manifest),
        },
        kd_identity={
            "path": kd_manifest.relative_to(tmp_path).as_posix(),
            "size": kd_manifest.stat().st_size,
            "sha256": orchestration.sha256_file(kd_manifest),
        },
        audit_path=tmp_path / "audit.json",
    )
    assert authenticated["completed_phases"] == ["generate-kd", "index-kd"]
