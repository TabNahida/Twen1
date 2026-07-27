from __future__ import annotations

import base64
import fcntl
import hashlib
import io
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import zipfile
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from twen.config import load_train_config
from twen.governed import GovernedControllerError, build_governed_plan
from twen.io.locking import FileLock
from twen.runtime.checkpoint import CHECKPOINT_SCHEMA_VERSION, MANIFEST_VERSION
from twen.source_identity import twen_source_tree_sha256
from twen.web import (
    DashboardAuth,
    DashboardController,
    DashboardError,
    DashboardSettings,
    GpuTelemetryMonitor,
    JsonlTailCache,
    LaunchProfile,
    _exclusive_advisory_lock_is_held,
    _governed_source_archive_hashes,
    _held_run_lock_owner,
    _LinuxProcessIdentity,
    _process_matches_governed_controller,
    _sha256_regular_file_no_follow,
    _source_tree_archive,
    create_dashboard_server,
    ensure_dashboard_auth_file,
    load_dashboard_auth,
    load_dashboard_settings,
    read_console_tail,
    serve_dashboard,
)


def _profile(tmp_path: Path, *, launch_enabled: bool = False) -> LaunchProfile:
    config_path = tmp_path / "fixed.yaml"
    config_path.write_text("fixed training config placeholder\n", encoding="utf-8")
    fork_from = (tmp_path / "runs/base-v1/step-final").resolve()
    fork_from.mkdir(parents=True, exist_ok=True)
    return LaunchProfile(
        profile_id="base-v2",
        label="Base v2",
        config_path=config_path.resolve(),
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        run_dir=(tmp_path / "runs/base-v2").resolve(),
        run_id="base-v2",
        stage="dense-oracle",
        resume="none",
        fork_from=fork_from,
        launch_enabled=launch_enabled,
    )


def _settings(tmp_path: Path, *, launch_enabled: bool = False) -> DashboardSettings:
    dashboard_config_path = (tmp_path / "dashboard.json").resolve()
    dashboard_config_path.write_text('{"schema_version":1}\n', encoding="utf-8")
    return DashboardSettings(
        dashboard_config_path=dashboard_config_path,
        dashboard_config_sha256=hashlib.sha256(dashboard_config_path.read_bytes()).hexdigest(),
        project_root=tmp_path.resolve(),
        state_dir=(tmp_path / ".twen/dashboard").resolve(),
        profiles=(_profile(tmp_path, launch_enabled=launch_enabled),),
    )


def _governed_settings(
    tmp_path: Path,
    *,
    launch_enabled: bool = True,
) -> DashboardSettings:
    dashboard_config_path = (tmp_path / "dashboard.json").resolve()
    dashboard_config_path.write_text('{"schema_version":1}\n', encoding="utf-8")
    config_path = (tmp_path / "configs/base/formal.yaml").resolve()
    config_path.parent.mkdir(parents=True)
    config_path.write_text("fixed governed training config\n", encoding="utf-8")
    controller_path = (tmp_path / "scripts/govern_v4_training.py").resolve()
    controller_path.parent.mkdir(parents=True)
    controller_path.write_text("# governed controller\n", encoding="utf-8")
    readiness_path = (tmp_path / "locks/formal.readiness.json").resolve()
    readiness_path.parent.mkdir(parents=True)
    readiness_path.write_text('{"kind":"readiness"}\n', encoding="utf-8")
    source_root = (tmp_path / "src/twen").resolve()
    source_root.mkdir(parents=True)
    (source_root / "__init__.py").write_text(
        "SNAPSHOT_MARKER = 'authenticated'\n",
        encoding="utf-8",
    )
    dependency_path = (tmp_path / "uv.lock").resolve()
    dependency_path.write_text("version = 1\n", encoding="utf-8")
    run_dir = (tmp_path / "runs/base-v4-formal").resolve()
    plan_id = "a" * 64
    profile = LaunchProfile(
        profile_id="base-v4-formal",
        label="Base v4 formal",
        config_path=config_path,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        run_dir=run_dir,
        run_id="base-v4-formal",
        stage="dense-oracle",
        resume="none",
        fork_from=None,
        launch_enabled=launch_enabled,
        launch_kind="governed_v4",
        governed_controller_path=controller_path,
        governed_controller_sha256=hashlib.sha256(controller_path.read_bytes()).hexdigest(),
        governed_readiness_path=readiness_path,
        governed_readiness_sha256=hashlib.sha256(readiness_path.read_bytes()).hexdigest(),
        governed_state_path=(
            run_dir.parent / f".{run_dir.name}.governed" / "controller-state.json"
        ),
        governed_plan_id=plan_id,
        governed_source_tree_sha256=twen_source_tree_sha256(source_root),
        governed_dependency_lock_name=dependency_path.name,
        governed_dependency_lock_sha256=hashlib.sha256(dependency_path.read_bytes()).hexdigest(),
    )
    return DashboardSettings(
        dashboard_config_path=dashboard_config_path,
        dashboard_config_sha256=hashlib.sha256(dashboard_config_path.read_bytes()).hexdigest(),
        project_root=tmp_path.resolve(),
        state_dir=(tmp_path / ".twen/dashboard").resolve(),
        profiles=(profile,),
    )


class _FakeGovernedSnapshot:
    controller_fd = 71
    source_fd = 72
    controller_sha256 = "1" * 64
    source_archive_sha256 = "2" * 64
    source_tree_sha256 = "3" * 64
    dependency_lock_sha256 = "4" * 64

    def __init__(self) -> None:
        self.closed = False

    @property
    def controller_path(self) -> str:
        return f"/proc/self/fd/{self.controller_fd}"

    @property
    def source_path(self) -> str:
        return f"/proc/self/fd/{self.source_fd}"

    @property
    def runtime_source_fd(self) -> int:
        return 0

    @property
    def runtime_source_path(self) -> str:
        return "/proc/self/fd/0"

    @property
    def pass_fds(self) -> tuple[int, int]:
        return self.controller_fd, self.source_fd

    def close(self) -> None:
        self.closed = True


def _governed_rank0_identity(
    profile: LaunchProfile,
    project_root: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    list[str],
    _LinuxProcessIdentity,
    dict[str, object],
]:
    dependency_path = project_root / "uv.lock"
    assert profile.governed_source_tree_sha256 is not None
    assert profile.governed_dependency_lock_sha256 is not None
    active_payload = [
        "-m",
        "twen.cli",
        "train",
        "--stage",
        profile.stage,
        "--config",
        str(profile.config_path),
        "--progress",
        "always",
        "--resume",
        "none",
    ]
    active_command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=1",
        *active_payload,
    ]
    process_command = [sys.executable, "-u", *active_payload]
    plan: dict[str, object] = {
        "plan_id": profile.governed_plan_id,
        "project_root": str(project_root),
        "config": {
            "path": str(profile.config_path),
            "sha256": profile.config_sha256,
        },
        "run": {
            "run_id": profile.run_id,
            "stage": profile.stage,
            "world_size": 1,
            "output_dir": str(profile.run_dir),
        },
        "source_tree": {
            "path": str(project_root / "src/twen"),
            "sha256": profile.governed_source_tree_sha256,
        },
        "dependency_lock": {
            "path": str(dependency_path),
            "sha256": profile.governed_dependency_lock_sha256,
        },
    }
    governed_state: dict[str, object] = {
        "status": "running",
        "active_command": active_command,
    }
    session: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "pid": 6161,
        "rank": 0,
        "world_size": 1,
        "hostname": platform.node(),
        "run_id": profile.run_id,
        "stage": profile.stage,
        "process_start_time_ticks": 987654,
        "process_cmdline": process_command,
        "cwd": str(project_root),
        "config_path": str(profile.config_path),
        "config_sha256": profile.config_sha256,
        "source_tree_sha256": profile.governed_source_tree_sha256,
        "dependency_lock": str(dependency_path),
        "dependency_lock_sha256": profile.governed_dependency_lock_sha256,
    }
    identity = _LinuxProcessIdentity(
        pid=6161,
        start_time_ticks=987654,
        command=tuple(process_command),
        cwd=project_root,
        executable=Path(sys.executable).resolve(),
        process_group_id=5151,
    )
    owner: dict[str, object] = {
        "pid": 6161,
        "host": platform.node(),
        "acquired_at": "2026-07-27T00:00:00+00:00",
    }
    return plan, governed_state, session, active_command, identity, owner


def _write_evaluation_plan(
    output: Path,
    *,
    device_type: str | None,
    valid_fingerprint: bool = True,
) -> Path:
    plan: dict[str, object] = {
        "schema_version": 1,
        "kind": "twen_nll_evaluation_plan",
    }
    if device_type is not None:
        plan["device_type"] = device_type
    encoded = json.dumps(
        plan,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    plan["plan_fingerprint"] = (
        hashlib.sha256(encoded).hexdigest() if valid_fingerprint else "0" * 64
    )
    path = output / "PLAN.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def _write_committed_checkpoint(
    run_dir: Path,
    *,
    run_id: str = "base-v2",
    stage: str = "dense-oracle",
    global_batch_tokens: int = 262_144,
) -> Path:
    """Write the smallest artifact accepted by the production full-hash verifier."""

    checkpoint = run_dir / "step-000000000007-periodic"
    (checkpoint / "runtime").mkdir(parents=True)
    (checkpoint / "state").mkdir()
    (checkpoint / "runtime/rank-00000.pkl").write_bytes(b"authenticated runtime payload")
    (checkpoint / "state/model.distcp").write_bytes(b"authenticated state payload")
    metadata = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_id": checkpoint.name,
        "kind": "periodic",
        "created_at": "2026-07-24T00:00:00+00:00",
        "backend": "torch-distributed-checkpoint",
        "run_id": run_id,
        "stage": stage,
        "global_step": 7,
        "committed_tokens": global_batch_tokens * 7,
        "trainer_state": {"committed_tokens": global_batch_tokens * 7},
        "data_cursor": {"global_token_index": global_batch_tokens * 7},
        "saved_world_size": 1,
        "global_batch_tokens": global_batch_tokens,
        "gradient_accumulation_steps": 64,
        "critical_fingerprint": "a" * 64,
        "data_fingerprint": "b" * 64,
        "rollback_applied": False,
        "tag": None,
        "extra": {},
    }
    (checkpoint / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    files = {}
    for path in sorted(checkpoint.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "COMPLETE"}:
            files[path.relative_to(checkpoint).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    manifest_bytes = json.dumps(
        {"version": MANIFEST_VERSION, "algorithm": "sha256", "files": files},
        sort_keys=True,
    ).encode()
    (checkpoint / "manifest.json").write_bytes(manifest_bytes)
    (checkpoint / "COMPLETE").write_text(
        hashlib.sha256(manifest_bytes).hexdigest() + "\n",
        encoding="ascii",
    )
    return checkpoint


def _fake_train_config(
    *,
    fingerprint: str = "current-config",
    run_id: str = "base-v2",
    stage: str = "dense-oracle",
    global_batch_tokens: int = 262_144,
) -> SimpleNamespace:
    return SimpleNamespace(
        run_id=run_id,
        stage=stage,
        data=SimpleNamespace(global_batch_tokens=global_batch_tokens),
        fingerprint=lambda: fingerprint,
    )


def test_packaged_dashboard_html_is_available() -> None:
    html = files("twen").joinpath("web_static/index.html").read_text(encoding="utf-8")
    assert "Twen 任务监控" in html
    assert "/api/snapshot?task=" in html
    assert "实时 GPU" in html
    assert "power draw 与 600 W limit" in html
    assert 'id="gpuPowerChart"' in html
    assert 'id="gpuThermalChart"' in html
    assert 'id="gpuVramChart"' in html
    assert 'id="runningTasks"' in html
    assert 'id="completedTasks"' in html
    assert 'id="taskProgressChart"' in html
    assert "运行日志实时 tok/s" in html
    assert "虚线为 attempt 累计 wall 均值" in html
    assert "parseConsoleTokenRate" in html
    assert "live_tokens_per_second" in html
    assert "attempt_tokens_per_second" in html
    assert "chart-tooltip" in html
    assert "pointermove" in html
    assert "grid-template-rows: auto minmax(230px, 1fr)" in html
    assert 'id="evaluationSection"' in html
    assert 'id="configurationWarning"' in html
    assert 'id="governanceWarning"' in html
    assert "Dashboard 配置已变更" in html
    assert "正式 v4 governed 启动门尚未通过" in html
    assert "configuration_stale" in html
    assert 'id="launchSelect"' in html
    assert 'id="pausedTasks"' in html
    assert "profile.start_available === true" in html
    assert 'profile.start_action === "resume" ? "恢复" : "首次启动"' in html
    assert 'profile.launch_kind === "governed_v4" ? "/api/governed/start"' in html
    assert 'status.launch_kind === "governed_v4" ? "/api/governed/start"' in html
    assert "完整 RUN ACK" in html
    assert "任务通过 preflight 后会进入运行中列表" in html
    assert 'addSummary("MTP Loss"' in html
    assert 'id="sourceMixChart"' in html
    assert '"数据阶段",' in html
    assert 'addSummary("来源配比"' in html
    assert 'addSummary(\n            "语料回绕"' in html
    assert '"source_tokens/"' in html
    assert 'key.startsWith("phase_source_tokens/")' in html
    assert '{ key: "mtp_loss", label: "MTP"' in html
    assert '{ key: "anchor_kl_loss", label: "anchor"' in html
    assert '{ key: "hidden_alignment", label: "hidden"' in html
    assert 'metric?.["lr_adjusted/adapters"]' in html
    assert 'metric?.["lr_adjustment_factor/adapters"]' in html
    assert 'addSummary(\n            "Muon 更新系数"' in html
    assert '{ key: "adapter_lr_nominal", label: "adapter nominal"' in html
    assert '{ key: "adapter_lr_adjusted", label: "Muon adjusted"' in html
    assert 'const fresh = await request("/api/bootstrap")' in html
    assert "live_gpu_relevant" in html
    assert 'task.kind === "kd"' in html
    assert "Reports" not in html
    assert "reportList" not in html
    assert "linear-gradient" not in html
    assert "radial-gradient" not in html
    assert "box-shadow" not in html
    assert "--radius: 6px" in html


def _gpu_line(
    power: float = 401.25,
    *,
    gpu_utilization: float = 99.0,
    memory_utilization: float = 37.0,
) -> str:
    return f"{power}, 600.00, {gpu_utilization}, {memory_utilization}, 2940, 26978, 5210, 62"


def _wait_for(predicate: object, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if callable(predicate) and predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached before timeout")


class _FakeLoopingProcess:
    def __init__(self, pid: int) -> None:
        read_descriptor, self._write_descriptor = os.pipe()
        self.stdout = os.fdopen(read_descriptor, "rb", buffering=0)
        self.pid = pid
        self.returncode: int | None = None
        self.terminate_called = False
        self.kill_called = False

    def emit(self, line: str) -> None:
        assert self._write_descriptor is not None
        os.write(self._write_descriptor, (line + "\n").encode())

    def disconnect(self) -> None:
        if self._write_descriptor is not None:
            os.close(self._write_descriptor)
            self._write_descriptor = None

    def exit(self, returncode: int) -> None:
        self.returncode = returncode
        self.disconnect()

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_called = True
        if self.returncode is None:
            self.returncode = -signal.SIGTERM
        self.disconnect()

    def kill(self) -> None:
        self.kill_called = True
        if self.returncode is None:
            self.returncode = -signal.SIGKILL
        self.disconnect()

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired(cmd="fake-nvidia-smi", timeout=timeout)
        return self.returncode


class _FakeLoopingProcessFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.processes: list[_FakeLoopingProcess] = []
        self.created = threading.Event()

    def __call__(self, *args: object, **kwargs: object) -> _FakeLoopingProcess:
        self.calls.append((args, kwargs))
        process = _FakeLoopingProcess(9000 + len(self.processes))
        self.processes.append(process)
        self.created.set()
        return process


def test_gpu_telemetry_parses_values_bounds_window_and_computes_statistics() -> None:
    monitor = GpuTelemetryMonitor(
        wall_clock=lambda: 1_700_000_000.125,
        history_samples=2,
    )

    assert monitor._record_output_line(_gpu_line(100.0, gpu_utilization=10.0))
    assert monitor._record_output_line(_gpu_line(300.0, gpu_utilization=50.0))
    assert monitor._record_output_line(_gpu_line(500.0, gpu_utilization=90.0))
    first = monitor.snapshot()
    second = monitor.snapshot()

    latest = first["latest"]
    assert latest["available"] is True
    assert latest["power_draw_w"] == 500.0
    assert latest["power_limit_w"] == 600.0
    assert latest["power_percent_of_limit"] == pytest.approx(500 / 6)
    assert latest["gpu_utilization_percent"] == 90.0
    assert latest["memory_utilization_percent"] == 37.0
    assert latest["sm_clock_mhz"] == 2940.0
    assert latest["vram_used_mib"] == 26978.0
    assert latest["vram_free_mib"] == 5210.0
    assert latest["vram_total_mib"] == 32188.0
    assert latest["temperature_c"] == 62.0
    assert latest["sampled_at_unix_ms"] == 1_700_000_000_125
    assert second == first
    assert [row["power_draw_w"] for row in first["history"]] == [300.0, 500.0]
    assert first["sample_interval_seconds"] == 0.1
    power = first["window_statistics"]["fields"]["power_draw_w"]
    assert power == {"current": 500.0, "mean": 400.0, "p95": 500.0, "max": 500.0}
    utilization = first["window_statistics"]["fields"]["gpu_utilization_percent"]
    assert utilization == {
        "current": 90.0,
        "mean": 70.0,
        "p95": 90.0,
        "max": 90.0,
    }


@pytest.mark.parametrize(
    "line",
    ("N/A, 600, 99", "1, 2, 3", "1,2,3,4,5,6,7,8\n1,2,3,4,5,6,7,8"),
)
def test_gpu_telemetry_invalid_rows_degrade_without_raising(line: str) -> None:
    monitor = GpuTelemetryMonitor()

    assert not monitor._record_output_line(line)
    snapshot = monitor.snapshot()

    assert snapshot["latest"]["available"] is False
    assert snapshot["latest"]["error"] == "invalid_output"
    assert snapshot["history"] == [snapshot["latest"]]


def test_gpu_telemetry_uses_one_fixed_long_lived_shell_free_process() -> None:
    factory = _FakeLoopingProcessFactory()
    monitor = GpuTelemetryMonitor(
        process_factory=factory,
        output_timeout_seconds=0.5,
    )
    stop = threading.Event()
    thread = threading.Thread(target=monitor.run_until_stopped, args=(stop,))
    thread.start()
    assert factory.created.wait(timeout=1)
    factory.processes[0].emit(_gpu_line())
    _wait_for(lambda: monitor.snapshot()["latest"].get("available") is True)

    assert len(factory.calls) == 1
    arguments, options = factory.calls[0]
    assert arguments == (
        (
            "/usr/lib/wsl/lib/nvidia-smi",
            "--id=0",
            "--query-gpu=power.draw,power.limit,utilization.gpu,utilization.memory,"
            "clocks.sm,memory.used,memory.free,temperature.gpu",
            "--format=csv,noheader,nounits",
            "--loop-ms=100",
        ),
    )
    assert options == {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.DEVNULL,
        "text": False,
        "bufsize": 0,
        "shell": False,
    }
    assert monitor.snapshot()["sampler"]["pid"] == 9000

    stop.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert factory.processes[0].terminate_called
    assert monitor.snapshot()["sampler"]["state"] == "stopped"


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    (("disconnect", "stream_disconnected"), ("exit", "command_exited")),
)
def test_gpu_telemetry_stream_failure_degrades_and_restarts(
    failure: str,
    expected_error: str,
) -> None:
    factory = _FakeLoopingProcessFactory()
    monitor = GpuTelemetryMonitor(
        process_factory=factory,
        output_timeout_seconds=0.5,
        restart_backoff_seconds=0.01,
        max_restart_backoff_seconds=0.02,
    )
    stop = threading.Event()
    thread = threading.Thread(target=monitor.run_until_stopped, args=(stop,))
    thread.start()
    assert factory.created.wait(timeout=1)
    first = factory.processes[0]
    if failure == "exit":
        first.exit(7)
    else:
        first.disconnect()
    _wait_for(lambda: len(factory.processes) >= 2)
    factory.processes[1].emit(_gpu_line(455.0))
    _wait_for(lambda: monitor.snapshot()["latest"].get("power_draw_w") == 455.0)

    snapshot = monitor.snapshot()
    failures = [row for row in snapshot["history"] if row.get("error") == expected_error]
    assert len(failures) == 1
    if failure == "exit":
        assert failures[0]["returncode"] == 7
    assert snapshot["sampler"]["restart_count"] >= 1

    stop.set()
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_gpu_telemetry_timeout_degrades_and_restarts() -> None:
    factory = _FakeLoopingProcessFactory()
    monitor = GpuTelemetryMonitor(
        process_factory=factory,
        output_timeout_seconds=0.03,
        restart_backoff_seconds=0.01,
        max_restart_backoff_seconds=0.02,
    )
    stop = threading.Event()
    thread = threading.Thread(target=monitor.run_until_stopped, args=(stop,))
    thread.start()
    _wait_for(lambda: len(factory.processes) >= 2)

    assert any(row.get("error") == "timeout" for row in monitor.snapshot()["history"])
    stop.set()
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_gpu_telemetry_journal_uses_aggregates_and_flushes_partial_bucket(
    tmp_path: Path,
) -> None:
    wall_values = iter((1_700_000_000.0, 1_700_000_001.0, 1_700_000_011.0))
    journal = tmp_path / "gpu-telemetry.jsonl"
    monitor = GpuTelemetryMonitor(
        wall_clock=lambda: next(wall_values),
        journal_path=journal,
    )

    monitor._record_output_line(_gpu_line(300.0), monotonic_now=0.0)
    monitor._record_output_line(_gpu_line(500.0), monotonic_now=1.0)
    monitor._record_output_line(_gpu_line(400.0), monotonic_now=11.0)
    monitor.flush_journal()

    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert len(records) == 2
    first = records[0]
    assert first["kind"] == "twen_gpu_telemetry_aggregate"
    assert first["raw_sample_interval_ms"] == 100
    assert first["sample_count"] == first["available_sample_count"] == 2
    assert first["unavailable_sample_count"] == 0
    assert first["fields"]["power_draw_w"] == {
        "last": 500.0,
        "max": 500.0,
        "mean": 400.0,
        "min": 300.0,
        "p95": 500.0,
    }
    assert "power_draw_w" not in first
    assert records[1]["sample_count"] == 1


def test_gpu_telemetry_journal_rotates_to_a_bounded_two_segment_budget(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "gpu-telemetry.jsonl"
    monitor = GpuTelemetryMonitor(
        wall_clock=lambda: 1_700_000_000.0,
        journal_path=journal,
        journal_bucket_seconds=0.01,
        journal_max_bytes=4096,
    )
    for index in range(12):
        monitor._record_output_line(
            _gpu_line(300.0 + index),
            monotonic_now=float(index),
        )
    monitor.flush_journal()

    backup = journal.with_name(journal.name + ".1")
    assert journal.stat().st_size <= 4096
    assert backup.stat().st_size <= 4096
    for path in (backup, journal):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert records
        assert all(row["kind"] == "twen_gpu_telemetry_aggregate" for row in records)


def test_controller_defaults_to_persistent_gpu_journal(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    controller = DashboardController(settings)

    assert controller.gpu_monitor.journal_path == settings.state_dir / "gpu-telemetry.jsonl"


def test_real_dashboard_config_remains_a_blocked_template() -> None:
    project_root = Path(__file__).resolve().parents[1]
    settings = load_dashboard_settings(project_root / "configs/web/dashboard.json")
    assert settings.project_root == project_root.resolve()
    assert [profile.profile_id for profile in settings.profiles] == [
        "base-dense-v1",
        "base-dense-v2-500m",
        "base-dense-v3-500m",
        "base-dense-v4-16m-smoke",
        "base-dense-v4-13m-low-lr-calibration",
        "base-dense-v4-13m-formal-lr-calibration",
    ]
    _, v2, v3, v4, calibration, formal_lr_calibration = settings.profiles
    launchable = [profile for profile in settings.profiles if profile.launch_enabled]
    assert launchable == []
    assert calibration.label == (
        "Base Dense v4 13M (blocked: use authenticated admission snapshot)"
    )
    assert formal_lr_calibration.label == (
        "Base Dense v4 13M formal-LR calibration (disabled template)"
    )
    assert (
        v2.resume
        == v3.resume
        == v4.resume
        == calibration.resume
        == formal_lr_calibration.resume
        == "none"
    )
    assert v2.fork_from == v3.fork_from
    assert (
        v3.fork_from
        == (project_root / "runs/base-dense-v1/step-000000000383-milestone-complete").resolve()
    )
    assert (
        v4.fork_from
        == (project_root / "runs/base-dense-v3-500m/step-000000001912-milestone-complete").resolve()
    )
    assert calibration.fork_from == v4.fork_from
    assert formal_lr_calibration.fork_from == v4.fork_from
    calibration_config = load_train_config(calibration.config_path)
    assert calibration_config.run_id == "base-dense-v4-13m-low-lr-calibration"
    assert calibration_config.optimizer.adapter_optimizer == "muon"
    assert calibration_config.optimizer.adapter_lr == 5e-5
    assert calibration_config.optimizer.scale_lr == 1e-5
    assert calibration_config.optimizer.max_tokens == 13_000_000
    assert calibration_config.optimizer.warmup_tokens == 5_000_000
    assert calibration_config.optimizer.lr_schedule == "cosine"
    assert calibration_config.data.allow_corpus_reuse is False
    formal_lr_config = load_train_config(formal_lr_calibration.config_path)
    assert formal_lr_config.run_id == "base-dense-v4-13m-formal-lr-calibration"
    assert formal_lr_config.optimizer.adapter_optimizer == "muon"
    assert formal_lr_config.optimizer.adapter_lr == 3e-5
    assert formal_lr_config.optimizer.scale_lr == 3e-6
    assert formal_lr_config.optimizer.max_tokens == 13_000_000
    assert formal_lr_config.optimizer.warmup_tokens == 10_000_000
    assert formal_lr_config.optimizer.lr_schedule == "cosine"
    assert formal_lr_config.data.allow_corpus_reuse is False
    assert all(
        profile.config_sha256 == hashlib.sha256(profile.config_path.read_bytes()).hexdigest()
        for profile in (v2, v3, v4, calibration, formal_lr_calibration)
    )
    assert all(
        profile.config_path.is_relative_to(settings.project_root) for profile in settings.profiles
    )
    assert all(
        profile.run_dir.is_relative_to(settings.project_root) for profile in settings.profiles
    )
    assert settings.state_dir.is_relative_to(settings.project_root)


def test_authenticated_calibration_dashboard_snapshot_is_launchable() -> None:
    project_root = Path(__file__).resolve().parents[1]
    admission_root = project_root / "locks/base-dense-v4-13m-calibration-admission-pass-002"
    dashboard_path = admission_root / "dashboard.json"
    settings = load_dashboard_settings(dashboard_path)
    launchable = [profile for profile in settings.profiles if profile.launch_enabled]
    assert [profile.profile_id for profile in launchable] == [
        "base-dense-v4-13m-low-lr-calibration"
    ]
    assert launchable[0].start_confirmation == ("START base-dense-v4-13m-low-lr-calibration")

    manifest_path = admission_root / "MANIFEST.json"
    complete_path = admission_root / "COMPLETE"
    admission_path = admission_root / "admission.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    recorded = json.loads(admission_path.read_text(encoding="utf-8"))
    unsigned_manifest = {
        key: value for key, value in manifest.items() if key != "bundle_fingerprint"
    }
    assert (
        manifest["bundle_fingerprint"]
        == hashlib.sha256(
            json.dumps(
                unsigned_manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    assert complete["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert (
        manifest["files"]["admission.json"]["sha256"]
        == hashlib.sha256(admission_path.read_bytes()).hexdigest()
    )
    assert (
        manifest["files"]["dashboard.json"]["sha256"]
        == hashlib.sha256(dashboard_path.read_bytes()).hexdigest()
    )
    assert recorded["acknowledgement"] == (
        "ACCEPT V4 WIKIPEDIA LICENSE "
        "fbc16551a1d7c0b9020852be97f751af2759442bd4d22bade707cd50a4fa3762"
    )
    assert recorded["authorizes_calibration_launch"] is True
    assert recorded["authorizes_formal_training"] is False
    assert recorded["training_started"] is False
    raw_dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    raw_profile = next(
        profile
        for profile in raw_dashboard["profiles"]
        if profile["id"] == "base-dense-v4-13m-low-lr-calibration"
    )
    inline = raw_profile["calibration_admission"]
    assert inline["admission_sha256"] == hashlib.sha256(admission_path.read_bytes()).hexdigest()
    assert inline["admission_fingerprint"] == recorded["admission_fingerprint"]
    assert inline["authorizes_formal_training"] is False
    assert inline["training_started"] is False


def test_governed_dashboard_binds_blocked_pending_formal_config(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs/base/dense-v4-250m-pilot.blocked.yaml"
    controller_path = project_root / "scripts/govern_v4_training.py"
    readiness_path = project_root / "locks/base-dense-v4-250m-pilot.readiness.json"
    plan = build_governed_plan(readiness_path)
    dashboard = tmp_path / "dashboard.json"
    dashboard.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_root": str(project_root),
                "state_dir": ".twen/dashboard",
                "profiles": [
                    {
                        "id": "base-dense-v4-250m-pilot",
                        "label": "Base dense v4 formal (blocked)",
                        "config": str(config_path.relative_to(project_root)),
                        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                        "resume": "none",
                        "fork_from": None,
                        "launch_enabled": False,
                        "launch_kind": "governed_v4",
                        "governed_controller": str(controller_path.relative_to(project_root)),
                        "governed_controller_sha256": hashlib.sha256(
                            controller_path.read_bytes()
                        ).hexdigest(),
                        "governed_readiness": str(readiness_path.relative_to(project_root)),
                        "governed_readiness_sha256": hashlib.sha256(
                            readiness_path.read_bytes()
                        ).hexdigest(),
                        "governed_state": (
                            "runs/.base-dense-v4-250m-pilot.governed/controller-state.json"
                        ),
                        "governed_plan_id": plan["plan_id"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    settings = load_dashboard_settings(dashboard)
    profile = settings.profiles[0]
    assert profile.launch_kind == "governed_v4"
    assert profile.launch_enabled is False
    assert profile.run_id == "base-dense-v4-250m-pilot"
    assert profile.stage == "dense-oracle"
    assert profile.run_dir == (project_root / "runs/base-dense-v4-250m-pilot").resolve()
    assert profile.governed_source_tree_sha256 == plan["source_tree"]["sha256"]  # type: ignore[index]
    assert (
        profile.governed_dependency_lock_name
        == Path(  # type: ignore[index]
            plan["dependency_lock"]["path"]
        ).name
    )
    assert profile.governed_dependency_lock_sha256 == plan["dependency_lock"]["sha256"]  # type: ignore[index]
    assert plan["config"]["preflight_fingerprint"] is None  # type: ignore[index]
    assert plan["readiness_issues"]


def _dashboard_with_real_train_config(
    root: Path,
    *,
    launch_enabled: bool,
    config_sha256: str | None,
) -> tuple[Path, Path]:
    source_root = Path(__file__).resolve().parents[1]
    config = root / "configs/base/dense-oracle.yaml"
    config.parent.mkdir(parents=True)
    config.write_bytes((source_root / "configs/base/dense-oracle.yaml").read_bytes())
    profile = {
        "id": "base-v2",
        "config": "configs/base/dense-oracle.yaml",
        "resume": "none",
        "fork_from": None,
        "launch_enabled": launch_enabled,
    }
    if config_sha256 is not None:
        profile["config_sha256"] = config_sha256
    dashboard = root / "configs/web/dashboard.json"
    dashboard.parent.mkdir(parents=True)
    dashboard.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_root": "../..",
                "state_dir": ".twen/dashboard",
                "profiles": [profile],
            }
        ),
        encoding="utf-8",
    )
    return dashboard, config


def test_launch_enabled_profile_requires_matching_pinned_config_sha256(tmp_path: Path) -> None:
    dashboard, config = _dashboard_with_real_train_config(
        tmp_path,
        launch_enabled=True,
        config_sha256=None,
    )
    with pytest.raises(DashboardError, match="config_sha256 is required"):
        load_dashboard_settings(dashboard)

    raw = json.loads(dashboard.read_text())
    raw["profiles"][0]["config_sha256"] = "0" * 64
    dashboard.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DashboardError, match="does not match the pinned"):
        load_dashboard_settings(dashboard)

    raw["profiles"][0]["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    dashboard.write_text(json.dumps(raw), encoding="utf-8")
    settings = load_dashboard_settings(dashboard)
    assert settings.dashboard_config_path == dashboard.resolve()
    assert settings.dashboard_config_sha256 == hashlib.sha256(dashboard.read_bytes()).hexdigest()
    assert settings.profiles[0].launch_enabled is True
    assert settings.profiles[0].config_sha256 == raw["profiles"][0]["config_sha256"]


def test_dashboard_config_rejects_path_escape(tmp_path: Path) -> None:
    source = tmp_path / "dashboard.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_root": ".",
                "state_dir": "../escaped",
                "profiles": [{"id": "x", "config": "config.yaml"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DashboardError, match="inside project_root"):
        load_dashboard_settings(source)


def test_jsonl_tail_cache_is_incremental_and_waits_for_complete_line(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(b'{"step":1,"loss":2.0}\n{"step":2')
    cache = JsonlTailCache(max_records=3)
    assert cache.read(path, limit=3) == [{"step": 1, "loss": 2.0}]
    with path.open("ab") as handle:
        handle.write(b',"loss":1.5}\n')
    assert cache.read(path, limit=3) == [
        {"step": 1, "loss": 2.0},
        {"step": 2, "loss": 1.5},
    ]
    path.write_text('{"step":7}\n', encoding="utf-8")
    assert cache.read(path, limit=3) == [{"step": 7}]


def test_snapshot_preserves_muon_learning_rate_observability_metrics(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    profile = settings.profiles[0]
    profile.run_dir.mkdir(parents=True)
    expected_source_mix = {
        "enabled": True,
        "basis_points": {"alpha": 6000, "beta": 4000},
        "lineage_basis_points": {"alpha": 7000, "beta": 3000},
        "effective_basis_points": {"alpha": 6000, "beta": 4000},
        "weight_override": True,
    }
    (profile.run_dir / "rank0-session.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "hostname": platform.node(),
                "pid": 1234,
                "run_id": profile.run_id,
                "stage": profile.stage,
                "source_mix": expected_source_mix,
            }
        ),
        encoding="utf-8",
    )
    expected = {
        "step": 9,
        "tokens": 2_359_296,
        "lr/adapters": 0.0001,
        "lr_adjusted/adapters": 0.00128,
        "lr_adjustment_factor/adapters": 12.8,
    }
    (profile.run_dir / "metrics.jsonl").write_text(
        json.dumps(expected) + "\n",
        encoding="utf-8",
    )

    snapshot = DashboardController(settings).snapshot(f"profile:{profile.profile_id}")

    assert snapshot["metrics"][-1] == expected
    assert snapshot["status"]["latest_metric"] == expected
    assert snapshot["status"]["source_mix"] == expected_source_mix


def test_console_tail_removes_ansi_and_splits_progress_updates(tmp_path: Path) -> None:
    path = tmp_path / "console.log"
    path.write_bytes(b"old\r\x1b[31mprogress 50%\x1b[0m\rprogress 100%\n")
    assert read_console_tail(path, lines=2) == "progress 50%\nprogress 100%"


def test_start_is_fixed_allowlisted_and_requires_confirmation(tmp_path: Path) -> None:
    calls = []

    def factory(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=4242)

    settings = _settings(tmp_path, launch_enabled=True)
    controller = DashboardController(settings, process_factory=factory)
    with pytest.raises(DashboardError, match="confirmation"):
        controller.start("base-v2", "yes")
    result = controller.start("base-v2", "START base-v2")
    assert result["pid"] == 4242
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == [
        command[0],
        "-m",
        "twen",
        "train",
        "--stage",
        "dense-oracle",
        "--config",
        str(settings.profiles[0].config_path),
        "--resume",
        "none",
        "--progress",
        "never",
        "--fork-from",
        str(settings.profiles[0].fork_from),
    ]
    assert kwargs["cwd"] == tmp_path.resolve()
    assert kwargs["start_new_session"] is True
    assert isinstance(command, list)
    assert "--dry-run" not in command
    assert "--graph-smoke" not in command
    assert (
        json.loads((settings.state_dir / "controller-state.json").read_text())["status"]
        == "launching"
    )
    assert result["launch_mode"] == "initial_fork"
    assert result["effective_resume"] == "none"
    assert result["effective_fork_from"] == str(settings.profiles[0].fork_from)
    assert result["verified_checkpoint_id"] is None
    assert controller.profile_status(settings.profiles[0])["preflight_enforced"] is True
    actions = [
        json.loads(line) for line in (settings.state_dir / "actions.jsonl").read_text().splitlines()
    ]
    assert actions[-1]["outcome"] == "accepted"
    assert actions[-1]["launch_mode"] == "initial_fork"


def test_governed_profile_refuses_direct_route_and_requires_exact_user_ack(
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    settings = _governed_settings(tmp_path)
    profile = settings.profiles[0]
    plan = {
        "plan_id": profile.governed_plan_id,
        "config": {
            "path": str(profile.config_path),
            "sha256": profile.config_sha256,
        },
        "run": {
            "run_id": profile.run_id,
            "stage": profile.stage,
            "output_dir": str(profile.run_dir),
        },
    }

    def factory(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(pid=5151)

    def process_identity(pid: int) -> _LinuxProcessIdentity:
        assert calls
        return _LinuxProcessIdentity(
            pid=pid,
            start_time_ticks=123456,
            command=tuple(calls[-1][0]),
            cwd=tmp_path.resolve(),
            executable=Path(sys.executable).resolve(),
            process_group_id=pid,
        )

    def authorize(_plan: object, acknowledgement: str | None) -> None:
        if acknowledgement != f"RUN {profile.governed_plan_id}":
            raise GovernedControllerError("explicit acknowledgement does not match")

    controller = DashboardController(settings, process_factory=factory)
    with pytest.raises(DashboardError, match="governed start route"):
        controller.start(profile.profile_id, f"RUN {profile.governed_plan_id}")
    assert not profile.run_dir.exists()
    assert calls == []

    with (
        patch.object(controller, "_governed_plan", return_value=plan),
        patch("twen.web.authorize_governed_run", side_effect=authorize),
        pytest.raises(DashboardError, match="acknowledgement"),
    ):
        controller.start_governed(profile.profile_id, "START base-v4-formal")
    assert not profile.run_dir.exists()
    assert not profile.governed_state_path.exists()
    assert calls == []

    acknowledgement = f"RUN {profile.governed_plan_id}"
    snapshot = _FakeGovernedSnapshot()
    with (
        patch.object(controller, "_governed_plan", return_value=plan),
        patch("twen.web.authorize_governed_run", side_effect=authorize),
        patch(
            "twen.web._build_governed_execution_snapshot",
            return_value=snapshot,
        ),
        patch(
            "twen.web._read_linux_process_identity",
            side_effect=process_identity,
        ),
        patch("twen.web.os.getpgrp", return_value=9999),
    ):
        result = controller.start_governed(profile.profile_id, acknowledgement)

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == [
        sys.executable,
        "-B",
        "-P",
        snapshot.controller_path,
        "--readiness",
        str(profile.governed_readiness_path),
        "--action",
        "run",
        "--state",
        str(profile.governed_state_path),
        "--ack",
        acknowledgement,
    ]
    assert "train" not in command
    assert kwargs["start_new_session"] is True
    assert kwargs["pass_fds"] == snapshot.pass_fds
    assert kwargs["stdin"] == snapshot.source_fd
    assert kwargs["env"]["PYTHONPATH"] == snapshot.runtime_source_path
    assert kwargs["env"]["PYTHONSAFEPATH"] == "1"
    assert result["launch_kind"] == "governed_v4"
    assert result["launch_mode"] == "governed"
    assert result["process_group_id"] == 5151
    assert result["process_start_time_ticks"] == 123456
    assert result["process_cmdline"] == command
    assert result["process_executable"] == str(Path(sys.executable).resolve())
    assert result["source_snapshot_fd"] == snapshot.runtime_source_fd
    assert result["source_snapshot_transport_fd"] == snapshot.source_fd
    assert result["governed_plan_id"] == profile.governed_plan_id
    assert result["governed_state_path"] == str(profile.governed_state_path)
    assert snapshot.closed is True


@pytest.mark.parametrize("failure_stage", ["process_identity", "state_write", "action_log"])
def test_post_spawn_launch_failure_kills_process_group_and_removes_owned_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    settings = _governed_settings(tmp_path)
    profile = settings.profiles[0]
    plan = {
        "plan_id": profile.governed_plan_id,
        "config": {
            "path": str(profile.config_path),
            "sha256": profile.config_sha256,
        },
        "run": {
            "run_id": profile.run_id,
            "stage": profile.stage,
            "output_dir": str(profile.run_dir),
        },
    }
    calls: list[list[str]] = []
    waits: list[float] = []

    class FailedProcess:
        pid = 5151

        @staticmethod
        def wait(*, timeout: float) -> int:
            waits.append(timeout)
            return -signal.SIGKILL

        @staticmethod
        def kill() -> None:
            raise AssertionError("verified fresh-session launch must be killed by process group")

    def factory(command: list[str], **_kwargs: object) -> FailedProcess:
        calls.append(command)
        return FailedProcess()

    def process_identity(pid: int) -> _LinuxProcessIdentity:
        if failure_stage == "process_identity":
            raise DashboardError("cannot authenticate spawned process")
        return _LinuxProcessIdentity(
            pid=pid,
            start_time_ticks=123456,
            command=tuple(calls[-1]),
            cwd=tmp_path.resolve(),
            executable=Path(sys.executable).resolve(),
            process_group_id=pid,
        )

    snapshot = _FakeGovernedSnapshot()
    controller = DashboardController(settings, process_factory=factory)
    monkeypatch.setattr(controller, "_governed_plan", lambda _profile: plan)
    monkeypatch.setattr("twen.web.authorize_governed_run", lambda *_args: None)
    monkeypatch.setattr(
        "twen.web._build_governed_execution_snapshot",
        lambda *_args: snapshot,
    )
    monkeypatch.setattr("twen.web._read_linux_process_identity", process_identity)
    monkeypatch.setattr("twen.web.os.getpgrp", lambda: 9999)
    killed_groups: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        "twen.web.os.killpg",
        lambda pid, requested_signal: killed_groups.append((pid, requested_signal)),
    )
    if failure_stage == "state_write":
        monkeypatch.setattr(
            "twen.web._atomic_json",
            lambda *_args: (_ for _ in ()).throw(OSError("state write failed")),
        )
    elif failure_stage == "action_log":
        monkeypatch.setattr(
            controller,
            "_record_action",
            lambda **_kwargs: (_ for _ in ()).throw(OSError("action log failed")),
        )

    with pytest.raises((DashboardError, OSError)):
        controller.start_governed(
            profile.profile_id,
            f"RUN {profile.governed_plan_id}",
        )

    assert len(calls) == 1
    assert killed_groups == [(5151, signal.SIGKILL)]
    assert waits == [5.0]
    assert snapshot.closed is True
    assert not controller._state_path.exists()
    assert not controller._audit_path.exists()


def test_governed_launch_executes_only_sealed_pre_popen_bytes(
    tmp_path: Path,
) -> None:
    settings = _governed_settings(tmp_path)
    profile = settings.profiles[0]
    source_root = tmp_path / "src/twen"
    source_root.mkdir(parents=True, exist_ok=True)
    source_file = source_root / "__init__.py"
    source_file.write_text("SNAPSHOT_MARKER = 'authenticated'\n", encoding="utf-8")
    dependency_path = tmp_path / "uv.lock"
    dependency_path.write_text("version = 1\n", encoding="utf-8")
    original_controller = profile.governed_controller_path.read_bytes()
    original_source = source_file.read_bytes()
    original_dependency = dependency_path.read_bytes()
    plan = {
        "plan_id": profile.governed_plan_id,
        "project_root": str(tmp_path.resolve()),
        "config": {
            "path": str(profile.config_path),
            "sha256": profile.config_sha256,
        },
        "run": {
            "run_id": profile.run_id,
            "stage": profile.stage,
            "output_dir": str(profile.run_dir),
        },
        "controller_sources": [
            {
                "path": str(profile.governed_controller_path),
                "sha256": profile.governed_controller_sha256,
            }
        ],
        "source_tree": {
            "path": str(source_root.resolve()),
            "sha256": twen_source_tree_sha256(source_root),
        },
        "dependency_lock": {
            "path": str(dependency_path.resolve()),
            "sha256": hashlib.sha256(original_dependency).hexdigest(),
        },
    }
    captured: dict[str, object] = {}

    def factory(command: list[str], **kwargs: object) -> SimpleNamespace:
        controller_fd, source_fd = kwargs["pass_fds"]
        captured["command"] = list(command)
        captured["controller_fd"] = controller_fd
        captured["source_fd"] = source_fd
        captured["stdin"] = kwargs["stdin"]
        captured["pythonpath"] = kwargs["env"]["PYTHONPATH"]
        profile.governed_controller_path.write_text(
            "# replaced after final admission\n",
            encoding="utf-8",
        )
        source_file.write_text("SNAPSHOT_MARKER = 'replaced'\n", encoding="utf-8")
        nested = subprocess.run(
            [
                sys.executable,
                "-P",
                "-c",
                (
                    "import subprocess,sys;"
                    "completed=subprocess.run("
                    "[sys.executable,'-P','-c',"
                    "'import twen; print(twen.SNAPSHOT_MARKER)'],"
                    "check=False,capture_output=True,text=True,close_fds=True);"
                    "sys.stdout.write(completed.stdout);"
                    "sys.stderr.write(completed.stderr);"
                    "raise SystemExit(completed.returncode)"
                ),
            ],
            check=False,
            stdin=kwargs["stdin"],
            pass_fds=kwargs["pass_fds"],
            env=kwargs["env"],
            capture_output=True,
            text=True,
            close_fds=True,
        )
        captured["nested_returncode"] = nested.returncode
        captured["nested_stdout"] = nested.stdout
        captured["nested_stderr"] = nested.stderr
        captured["controller"] = Path(f"/proc/self/fd/{controller_fd}").read_bytes()
        captured["source_archive"] = Path(f"/proc/self/fd/{source_fd}").read_bytes()
        with zipfile.ZipFile(f"/proc/self/fd/{source_fd}") as archive:
            captured["source"] = archive.read("twen/__init__.py")
            captured["dependency"] = archive.read(".twen-governed-dependency/uv.lock")
        return SimpleNamespace(pid=5151)

    def process_identity(pid: int) -> _LinuxProcessIdentity:
        return _LinuxProcessIdentity(
            pid=pid,
            start_time_ticks=123456,
            command=tuple(captured["command"]),
            cwd=tmp_path.resolve(),
            executable=Path(sys.executable).resolve(),
            process_group_id=pid,
        )

    controller = DashboardController(settings, process_factory=factory)
    with (
        patch.object(controller, "_governed_plan", return_value=plan),
        patch("twen.web.authorize_governed_run"),
        patch(
            "twen.web._read_linux_process_identity",
            side_effect=process_identity,
        ),
        patch("twen.web.os.getpgrp", return_value=9999),
    ):
        state = controller.start_governed(
            profile.profile_id,
            f"RUN {profile.governed_plan_id}",
        )

    assert captured["controller"] == original_controller
    assert captured["source"] == original_source
    assert captured["dependency"] == original_dependency
    assert captured["nested_returncode"] == 0, captured["nested_stderr"]
    assert captured["nested_stdout"] == "authenticated\n"
    assert state["controller_snapshot_sha256"] == hashlib.sha256(original_controller).hexdigest()
    assert state["source_snapshot_sha256"] == hashlib.sha256(captured["source_archive"]).hexdigest()
    assert state["source_tree_sha256"] == plan["source_tree"]["sha256"]
    assert state["dependency_lock_sha256"] == plan["dependency_lock"]["sha256"]
    assert state["process_cmdline"][3] == (f"/proc/self/fd/{captured['controller_fd']}")
    assert captured["stdin"] == captured["source_fd"]
    assert captured["pythonpath"] == "/proc/self/fd/0"
    assert state["source_snapshot_fd"] == 0
    assert state["source_snapshot_transport_fd"] == captured["source_fd"]
    with pytest.raises(OSError):
        os.fstat(int(captured["controller_fd"]))
    with pytest.raises(OSError):
        os.fstat(int(captured["source_fd"]))


def test_blocked_governed_readiness_fails_before_process_or_run_state_write(
    tmp_path: Path,
) -> None:
    calls: list[object] = []
    settings = _governed_settings(tmp_path)
    profile = settings.profiles[0]
    plan = {
        "plan_id": profile.governed_plan_id,
        "config": {
            "path": str(profile.config_path),
            "sha256": profile.config_sha256,
        },
        "run": {
            "run_id": profile.run_id,
            "stage": profile.stage,
            "output_dir": str(profile.run_dir),
        },
    }
    controller = DashboardController(
        settings,
        process_factory=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with (
        patch.object(controller, "_governed_plan", return_value=plan),
        patch(
            "twen.web.authorize_governed_run",
            side_effect=GovernedControllerError("governed launch is blocked: calibration"),
        ),
        pytest.raises(DashboardError, match="governed launch is blocked"),
    ):
        controller.start_governed(
            profile.profile_id,
            f"RUN {profile.governed_plan_id}",
        )

    assert calls == []
    assert not profile.run_dir.exists()
    assert not profile.governed_state_path.exists()
    assert not controller._state_path.exists()
    assert not controller._audit_path.exists()
    assert list(settings.state_dir.iterdir()) == []


@pytest.mark.parametrize("blocked_authorization", [2, 3])
def test_governed_reauthorization_failure_remains_zero_write(
    tmp_path: Path,
    blocked_authorization: int,
) -> None:
    calls: list[object] = []
    settings = _governed_settings(tmp_path)
    profile = settings.profiles[0]
    plan = {
        "plan_id": profile.governed_plan_id,
        "config": {
            "path": str(profile.config_path),
            "sha256": profile.config_sha256,
        },
        "run": {
            "run_id": profile.run_id,
            "stage": profile.stage,
            "output_dir": str(profile.run_dir),
        },
    }
    authorizations = 0

    def authorize(_plan: object, _acknowledgement: str | None) -> None:
        nonlocal authorizations
        authorizations += 1
        if authorizations == blocked_authorization:
            raise GovernedControllerError("governed launch became blocked")

    snapshot = _FakeGovernedSnapshot()
    controller = DashboardController(
        settings,
        process_factory=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    with (
        patch.object(controller, "_governed_plan", return_value=plan),
        patch("twen.web.authorize_governed_run", side_effect=authorize),
        patch(
            "twen.web._build_governed_execution_snapshot",
            return_value=snapshot,
        ) as build_snapshot,
        pytest.raises(DashboardError, match="became blocked"),
    ):
        controller.start_governed(
            profile.profile_id,
            f"RUN {profile.governed_plan_id}",
        )

    assert authorizations == blocked_authorization
    assert calls == []
    assert not profile.run_dir.exists()
    assert not profile.governed_state_path.exists()
    assert not controller._state_path.exists()
    assert not controller._audit_path.exists()
    assert list(settings.state_dir.iterdir()) == []
    if blocked_authorization == 2:
        build_snapshot.assert_not_called()
        assert snapshot.closed is False
    else:
        build_snapshot.assert_called_once_with(profile, plan)
        assert snapshot.closed is True


def test_governed_process_matcher_requires_exact_controller_contract(tmp_path: Path) -> None:
    profile = _governed_settings(tmp_path).profiles[0]
    assert profile.governed_source_tree_sha256 is not None
    assert profile.governed_dependency_lock_sha256 is not None
    controller_fd = 71
    source_fd = 0
    source_transport_fd = 72
    executable = Path(sys.executable).resolve()
    arguments = [
        sys.executable,
        "-B",
        "-P",
        f"/proc/self/fd/{controller_fd}",
        "--readiness",
        str(profile.governed_readiness_path),
        "--action",
        "run",
        "--state",
        str(profile.governed_state_path),
        "--ack",
        f"RUN {profile.governed_plan_id}",
    ]
    state = {
        "schema_version": 1,
        "profile_id": profile.profile_id,
        "run_id": profile.run_id,
        "launch_kind": "governed_v4",
        "governed_plan_id": profile.governed_plan_id,
        "governed_state_path": str(profile.governed_state_path),
        "project_root": str(tmp_path.resolve()),
        "status": "running",
        "pid": 5151,
        "process_group_id": 5151,
        "process_start_time_ticks": 123456,
        "process_cmdline": arguments,
        "process_executable": str(executable),
        "controller_snapshot_fd": controller_fd,
        "source_snapshot_fd": source_fd,
        "source_snapshot_transport_fd": source_transport_fd,
        "controller_snapshot_sha256": profile.governed_controller_sha256,
        "source_snapshot_sha256": None,
        "source_tree_sha256": profile.governed_source_tree_sha256,
        "dependency_lock_sha256": profile.governed_dependency_lock_sha256,
        "command_sha256": hashlib.sha256("\0".join(arguments).encode()).hexdigest(),
    }
    archive_payload, source_tree_sha, dependency_lock_sha = _source_tree_archive(
        {
            "project_root": str(tmp_path.resolve()),
            "source_tree": {
                "path": str((tmp_path / "src/twen").resolve()),
                "sha256": profile.governed_source_tree_sha256,
            },
            "dependency_lock": {
                "path": str((tmp_path / "uv.lock").resolve()),
                "sha256": profile.governed_dependency_lock_sha256,
            },
        }
    )
    assert source_tree_sha == profile.governed_source_tree_sha256
    assert dependency_lock_sha == profile.governed_dependency_lock_sha256
    state["source_snapshot_sha256"] = hashlib.sha256(archive_payload).hexdigest()
    identity = _LinuxProcessIdentity(
        pid=5151,
        start_time_ticks=123456,
        command=tuple(arguments),
        cwd=tmp_path.resolve(),
        executable=executable,
        process_group_id=5151,
    )
    with (
        patch("twen.web._read_linux_process_identity", return_value=identity),
        patch(
            "twen.web._read_process_environment",
            return_value={
                "PYTHONPATH": f"/proc/self/fd/{source_fd}",
                "PYTHONSAFEPATH": "1",
                "PYTHONUNBUFFERED": "1",
            },
        ),
        patch(
            "twen.web._sealed_process_fd_sha256",
            return_value=profile.governed_controller_sha256,
        ),
        patch(
            "twen.web._sealed_process_fd_bytes",
            return_value=archive_payload,
        ),
    ):
        assert _process_matches_governed_controller(
            5151,
            profile,
            state,
            project_root=tmp_path.resolve(),
        )
        malicious_buffer = io.BytesIO()
        with zipfile.ZipFile(
            malicious_buffer,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr(
                "twen/__init__.py",
                b"SNAPSHOT_MARKER = 'forged'\n",
            )
            archive.writestr(
                ".twen-governed-dependency/uv.lock",
                (tmp_path / "uv.lock").read_bytes(),
            )
        malicious_payload = malicious_buffer.getvalue()
        malicious_state = {
            **state,
            "source_snapshot_sha256": hashlib.sha256(malicious_payload).hexdigest(),
        }
        with patch(
            "twen.web._sealed_process_fd_bytes",
            return_value=malicious_payload,
        ):
            assert not _process_matches_governed_controller(
                5151,
                profile,
                malicious_state,
                project_root=tmp_path.resolve(),
            )
            assert not _process_matches_governed_controller(
                5151,
                profile,
                {
                    **malicious_state,
                    "source_tree_sha256": "0" * 64,
                },
                project_root=tmp_path.resolve(),
            )
        changed = list(arguments)
        changed[changed.index("--action") + 1] = "status"
        changed_identity = _LinuxProcessIdentity(
            pid=5151,
            start_time_ticks=123456,
            command=tuple(changed),
            cwd=tmp_path.resolve(),
            executable=executable,
            process_group_id=5151,
        )
        with patch(
            "twen.web._read_linux_process_identity",
            return_value=changed_identity,
        ):
            assert not _process_matches_governed_controller(
                5151,
                profile,
                state,
                project_root=tmp_path.resolve(),
            )
        for field, value in (
            ("status", "exited"),
            ("process_start_time_ticks", 123455),
            ("process_group_id", 6161),
            ("project_root", str(tmp_path / "other")),
        ):
            changed_state = {**state, field: value}
            assert not _process_matches_governed_controller(
                5151,
                profile,
                changed_state,
                project_root=tmp_path.resolve(),
            )
        for changed_identity in (
            _LinuxProcessIdentity(
                pid=5151,
                start_time_ticks=123455,
                command=tuple(arguments),
                cwd=tmp_path.resolve(),
                executable=executable,
                process_group_id=5151,
            ),
            _LinuxProcessIdentity(
                pid=5151,
                start_time_ticks=123456,
                command=tuple(arguments),
                cwd=(tmp_path / "other").resolve(),
                executable=executable,
                process_group_id=5151,
            ),
            _LinuxProcessIdentity(
                pid=5151,
                start_time_ticks=123456,
                command=tuple(arguments),
                cwd=tmp_path.resolve(),
                executable=executable,
                process_group_id=6161,
            ),
        ):
            with patch(
                "twen.web._read_linux_process_identity",
                return_value=changed_identity,
            ):
                assert not _process_matches_governed_controller(
                    5151,
                    profile,
                    state,
                    project_root=tmp_path.resolve(),
                )


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "extra", "traversal", "stored", "missing_dependency", "malformed"],
)
def test_governed_source_archive_requires_exact_safe_inventory(
    mutation: str,
) -> None:
    if mutation == "malformed":
        payload = b"not a zip archive"
    else:
        buffer = io.BytesIO()
        compression = zipfile.ZIP_STORED if mutation == "stored" else zipfile.ZIP_DEFLATED
        with zipfile.ZipFile(buffer, mode="w", compression=compression) as archive:
            archive.writestr("twen/__init__.py", b"SAFE = True\n")
            if mutation == "duplicate":
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr("twen/__init__.py", b"SAFE = False\n")
            elif mutation == "extra":
                archive.writestr("payload.py", b"raise RuntimeError\n")
            elif mutation == "traversal":
                archive.writestr("twen/../payload.py", b"raise RuntimeError\n")
            if mutation != "missing_dependency":
                archive.writestr(
                    ".twen-governed-dependency/uv.lock",
                    b"version = 1\n",
                )
        payload = buffer.getvalue()

    with pytest.raises(DashboardError):
        _governed_source_archive_hashes(
            payload,
            dependency_lock_name="uv.lock",
        )


def test_second_start_uses_authenticated_auto_resume_without_fork_or_torch_import(
    tmp_path: Path,
) -> None:
    calls = []

    def factory(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=4343)

    settings = _settings(tmp_path, launch_enabled=True)
    profile = settings.profiles[0]
    checkpoint = _write_committed_checkpoint(profile.run_dir)
    (profile.run_dir / "resolved_config.yaml").write_text("fixed\n", encoding="utf-8")
    (profile.run_dir / "console.log").write_text("previous attempt\n", encoding="utf-8")
    (profile.run_dir / "rank0-session.json").write_text(
        json.dumps({"status": "stopped", "pid": 1234}),
        encoding="utf-8",
    )
    # A resumed v2 run must not depend on the old v1 fork source still existing.
    profile.fork_from.rmdir()
    controller = DashboardController(settings, process_factory=factory)
    with (
        patch("twen.web.load_train_config", return_value=_fake_train_config()),
        patch(
            "twen.runtime.checkpoint._torch_distributed_info",
            side_effect=AssertionError("dashboard checkpoint inspection imported torch"),
        ),
    ):
        result = controller.start("base-v2", "START base-v2")

    assert len(calls) == 1
    command, _ = calls[0]
    assert command[command.index("--resume") + 1] == "auto"
    assert "--fork-from" not in command
    assert result["launch_mode"] == "resume_auto"
    assert result["effective_resume"] == "auto"
    assert result["effective_fork_from"] is None
    assert result["verified_checkpoint_id"] == checkpoint.name
    assert (
        result["resume_compatibility_gate"] == "training_entrypoint_preflight_and_checkpoint_loader"
    )
    actions = [
        json.loads(line) for line in (settings.state_dir / "actions.jsonl").read_text().splitlines()
    ]
    assert actions[-1]["launch_mode"] == "resume_auto"
    assert actions[-1]["verified_checkpoint_id"] == checkpoint.name


@pytest.mark.parametrize(
    ("session_status", "expected_status"),
    [("stopped", "stopped"), ("running", "stale")],
)
def test_interrupted_profile_stays_visible_and_resumable_without_poll_time_hashing(
    tmp_path: Path,
    session_status: str,
    expected_status: str,
) -> None:
    settings = _settings(tmp_path, launch_enabled=True)
    profile = settings.profiles[0]
    profile.run_dir.mkdir(parents=True)
    (profile.run_dir / "rank0-session.json").write_text(
        json.dumps(
            {
                "status": session_status,
                "hostname": platform.node(),
                "pid": 999_999_999,
                "run_id": profile.run_id,
                "stage": profile.stage,
            }
        ),
        encoding="utf-8",
    )
    controller = DashboardController(settings)

    with patch(
        "twen.web.CheckpointManager.find_latest_valid_with_metadata",
        side_effect=AssertionError("dashboard polling must not hash checkpoints"),
    ):
        public = controller.public_profiles()[0]
        selection = controller.task_selection()

    assert public["state"] == expected_status
    assert public["active"] is False
    assert public["start_available"] is True
    assert public["start_action"] == "resume"
    paused = next(task for task in selection["tasks"] if task["key"] == "profile:base-v2")
    assert paused["state"] == "paused"
    assert paused["control"]["start_available"] is True
    assert paused["control"]["start_action"] == "resume"


def test_completed_launch_profile_remains_completed_and_is_not_restartable(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, launch_enabled=True)
    profile = settings.profiles[0]
    profile.run_dir.mkdir(parents=True)
    (profile.run_dir / "rank0-session.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "hostname": platform.node(),
                "pid": 1234,
                "run_id": profile.run_id,
                "stage": profile.stage,
            }
        ),
        encoding="utf-8",
    )
    controller = DashboardController(settings)

    public = controller.public_profiles()[0]
    selection = controller.task_selection()

    assert public["state"] == "completed"
    assert public["start_available"] is False
    completed = next(task for task in selection["tasks"] if task["key"] == "profile:base-v2")
    assert completed["state"] == "completed"


@pytest.mark.parametrize("dirty_name", ["metrics.jsonl", ".step-000000000001.incomplete"])
def test_second_start_fails_closed_for_nonempty_run_without_authenticated_checkpoint(
    tmp_path: Path,
    dirty_name: str,
) -> None:
    calls = []
    settings = _settings(tmp_path, launch_enabled=True)
    profile = settings.profiles[0]
    profile.run_dir.mkdir(parents=True)
    dirty = profile.run_dir / dirty_name
    if dirty_name.endswith(".incomplete"):
        dirty.mkdir()
    else:
        dirty.write_text('{"loss": 1.0}\n', encoding="utf-8")
    controller = DashboardController(
        settings,
        process_factory=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(DashboardError, match="no fully authenticated committed checkpoint"):
        controller.start("base-v2", "START base-v2")

    assert calls == []
    assert not (settings.state_dir / "actions.jsonl").exists()


def test_second_start_rejects_authenticated_checkpoint_with_changed_resolved_config(
    tmp_path: Path,
) -> None:
    calls = []
    settings = _settings(tmp_path, launch_enabled=True)
    profile = settings.profiles[0]
    _write_committed_checkpoint(profile.run_dir)
    (profile.run_dir / "resolved_config.yaml").write_text("old\n", encoding="utf-8")
    controller = DashboardController(
        settings,
        process_factory=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    configs = [
        _fake_train_config(fingerprint="current"),
        _fake_train_config(fingerprint="previous"),
    ]

    with (
        patch("twen.web.load_train_config", side_effect=configs),
        pytest.raises(DashboardError, match="not exact-resume compatible"),
    ):
        controller.start("base-v2", "START base-v2")

    assert calls == []


def test_second_start_rejects_checkpoint_whose_committed_payload_was_tampered(
    tmp_path: Path,
) -> None:
    calls = []
    settings = _settings(tmp_path, launch_enabled=True)
    profile = settings.profiles[0]
    checkpoint = _write_committed_checkpoint(profile.run_dir)
    (profile.run_dir / "resolved_config.yaml").write_text("fixed\n", encoding="utf-8")
    (checkpoint / "state/model.distcp").write_bytes(b"tampered after COMPLETE")
    controller = DashboardController(
        settings,
        process_factory=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(DashboardError, match="no fully authenticated committed checkpoint"):
        controller.start("base-v2", "START base-v2")

    assert calls == []


def test_second_start_rejects_symlinked_checkpoint_candidate(tmp_path: Path) -> None:
    calls = []
    project = tmp_path / "project"
    project.mkdir()
    settings = _settings(project, launch_enabled=True)
    profile = settings.profiles[0]
    profile.run_dir.mkdir(parents=True)
    outside = tmp_path / "step-000000000007-periodic"
    outside.mkdir()
    (profile.run_dir / outside.name).symlink_to(outside, target_is_directory=True)
    controller = DashboardController(
        settings,
        process_factory=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(DashboardError, match="symlinked checkpoint candidate"):
        controller.start("base-v2", "START base-v2")

    assert calls == []


def test_cross_process_action_lock_refuses_a_second_launcher_without_deadlock(
    tmp_path: Path,
) -> None:
    calls: list[object] = []
    settings = _settings(tmp_path, launch_enabled=True)
    controller = DashboardController(
        settings,
        process_factory=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    descriptor = os.open(settings.state_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (
            patch("twen.web._DASHBOARD_ACTION_LOCK_TIMEOUT_SECONDS", 0.0),
            pytest.raises(DashboardError, match="another dashboard process"),
        ):
            controller.start("base-v2", "START base-v2")
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    assert calls == []
    assert not (settings.state_dir / "actions.jsonl").exists()


def test_held_run_lock_owner_requires_regular_nofollow_actively_locked_file(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / ".run.lock"
    owner = {
        "pid": os.getpid(),
        "host": platform.node(),
        "acquired_at": "2026-07-27T00:00:00+00:00",
    }
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.write(descriptor, json.dumps(owner).encode())
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert _held_run_lock_owner(lock_path) == owner
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        assert _held_run_lock_owner(lock_path) is None
    finally:
        os.close(descriptor)
    redirected = tmp_path / "redirected.lock"
    redirected.symlink_to(lock_path)
    assert _held_run_lock_owner(redirected) is None


def test_start_rejects_run_directory_symlink_escape(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    settings = _settings(project, launch_enabled=True)
    outside = tmp_path / "outside-run"
    outside.mkdir()
    settings.profiles[0].run_dir.symlink_to(outside, target_is_directory=True)
    controller = DashboardController(settings)

    with pytest.raises(DashboardError, match="inside project_root"):
        controller.start("base-v2", "START base-v2")


def test_launch_disabled_profile_cannot_start(tmp_path: Path) -> None:
    controller = DashboardController(_settings(tmp_path))
    with pytest.raises(DashboardError, match="disabled"):
        controller.start("base-v2", "START base-v2")


def test_allowlisted_config_must_not_change_after_server_start(tmp_path: Path) -> None:
    settings = _settings(tmp_path, launch_enabled=True)
    controller = DashboardController(settings)
    assert controller.profile_status(settings.profiles[0])["config_sha256"] == (
        settings.profiles[0].config_sha256
    )
    settings.profiles[0].config_path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(DashboardError, match="changed after dashboard startup"):
        controller.start("base-v2", "START base-v2")


def test_dashboard_config_change_blocks_start_but_not_save_or_stop(tmp_path: Path) -> None:
    settings = _settings(tmp_path, launch_enabled=True)
    profile = settings.profiles[0]
    controller = DashboardController(settings)
    settings.dashboard_config_path.write_text('{"schema_version":1,"changed":true}\n')

    public = controller.public_profiles()[0]
    assert public["configuration_stale"] is True
    assert public["restart_required"] is True
    assert public["start_available"] is False
    with pytest.raises(DashboardError, match="restart the dashboard"):
        controller.start("base-v2", "START base-v2")

    profile.run_dir.mkdir(parents=True)
    (profile.run_dir / "rank0-session.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_id": "session-fixed",
                "status": "running",
                "hostname": platform.node(),
                "pid": 31337,
            }
        ),
        encoding="utf-8",
    )
    with (
        patch("twen.web._process_matches_profile", return_value=True),
        patch("twen.web.os.kill") as kill,
    ):
        saved = controller.signal("base-v2", "save", "SAVE base-v2")
        stopped = controller.signal("base-v2", "stop", "STOP base-v2")
    assert saved["signal"] == "SIGUSR1"
    assert stopped["signal"] == "SIGTERM"
    assert kill.call_args_list == [
        call(31337, signal.SIGUSR1),
        call(31337, signal.SIGTERM),
    ]


@pytest.mark.parametrize("race", ["replace", "append"])
def test_dashboard_config_hash_fails_closed_if_file_changes_during_read(
    tmp_path: Path,
    race: str,
) -> None:
    settings = _settings(tmp_path)
    path = settings.dashboard_config_path
    original_read = os.read
    raced = False

    def racing_read(descriptor: int, size: int) -> bytes:
        nonlocal raced
        chunk = original_read(descriptor, size)
        if not raced:
            raced = True
            if race == "replace":
                replacement = path.with_suffix(".replacement")
                replacement.write_bytes(path.read_bytes())
                os.replace(replacement, path)
            else:
                with path.open("ab") as handle:
                    handle.write(b" ")
        return chunk

    with patch("twen.web.os.read", side_effect=racing_read):
        status = DashboardController(settings).configuration_status()
    assert raced is True
    assert status["configuration_stale"] is True
    assert status["restart_required"] is True


def test_dashboard_config_is_rechecked_immediately_before_process_creation(
    tmp_path: Path,
) -> None:
    calls: list[tuple[object, dict[str, object]]] = []

    def factory(command: object, **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(pid=4242)

    settings = _settings(tmp_path, launch_enabled=True)
    controller = DashboardController(settings, process_factory=factory)
    original_admission = controller._fixed_initial_fork_launch

    def mutate_during_admission(
        profile: LaunchProfile,
        run_dir: Path,
    ) -> object:
        launch = original_admission(profile, run_dir)
        settings.dashboard_config_path.write_text(
            '{"schema_version":1,"changed":true}\n',
            encoding="utf-8",
        )
        return launch

    with (
        patch.object(
            controller,
            "_fixed_initial_fork_launch",
            side_effect=mutate_during_admission,
        ),
        pytest.raises(DashboardError, match="changed during launch admission"),
    ):
        controller.start("base-v2", "START base-v2")
    assert calls == []


def test_duplicate_active_profile_is_refused(tmp_path: Path) -> None:
    settings = _settings(tmp_path, launch_enabled=True)
    profile = settings.profiles[0]
    profile.run_dir.mkdir(parents=True)
    (profile.run_dir / "rank0-session.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "running",
                "hostname": platform.node(),
                "pid": 31337,
            }
        ),
        encoding="utf-8",
    )
    controller = DashboardController(settings)
    with (
        patch("twen.web._process_matches_profile", return_value=True),
        pytest.raises(DashboardError, match="duplicate launch refused"),
    ):
        controller.start("base-v2", "START base-v2")


def test_stop_uses_verified_rank_zero_pid_and_sigterm(tmp_path: Path) -> None:
    settings = _settings(tmp_path, launch_enabled=True)
    profile = settings.profiles[0]
    profile.run_dir.mkdir(parents=True)
    (profile.run_dir / "rank0-session.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_id": "session-fixed",
                "status": "running",
                "hostname": platform.node(),
                "pid": 31337,
            }
        ),
        encoding="utf-8",
    )
    controller = DashboardController(settings)
    with (
        patch("twen.web._process_matches_profile", return_value=True),
        patch("twen.web.os.kill") as kill,
    ):
        result = controller.signal("base-v2", "stop", "STOP base-v2")
    kill.assert_called_once_with(31337, signal.SIGTERM)
    assert result["signal"] == "SIGTERM"
    state = json.loads((settings.state_dir / "controller-state.json").read_text())
    assert state["status"] == "stop_requested"


def test_stop_can_terminate_a_verified_launching_process_before_rank0_session(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, launch_enabled=True)
    controller = DashboardController(settings)
    controller._state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_id": "base-v2",
                "run_id": "base-v2",
                "pid": 31337,
                "status": "launching",
                "started_at_utc": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    with (
        patch("twen.web._process_matches_profile", return_value=True),
        patch("twen.web.os.kill") as kill,
    ):
        result = controller.signal("base-v2", "stop", "STOP base-v2")

    kill.assert_called_once_with(31337, signal.SIGTERM)
    assert result["signal"] == "SIGTERM"
    assert json.loads(controller._state_path.read_text())["status"] == "stop_requested"


def test_governed_stop_signals_verified_controller_process_group_during_evaluation(
    tmp_path: Path,
) -> None:
    settings = _governed_settings(tmp_path)
    profile = settings.profiles[0]
    controller = DashboardController(settings)
    controller._state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_id": profile.profile_id,
                "run_id": profile.run_id,
                "pid": 5151,
                "process_group_id": 5151,
                "launch_kind": "governed_v4",
                "governed_plan_id": profile.governed_plan_id,
                "status": "running",
                "started_at_utc": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    with (
        patch("twen.web._process_matches_governed_controller", return_value=True),
        patch("twen.web.os.getpgrp", return_value=999),
        patch("twen.web.os.getpgid", return_value=5151),
        patch("twen.web.os.killpg") as kill_group,
        patch("twen.web.os.kill") as kill_process,
    ):
        result = controller.signal(
            profile.profile_id,
            "stop",
            profile.stop_confirmation,
        )

    kill_group.assert_called_once_with(5151, signal.SIGTERM)
    kill_process.assert_not_called()
    assert result["signal_target"] == "process_group"
    assert result["process_group_id"] == 5151
    state = json.loads(controller._state_path.read_text())
    assert state["status"] == "stop_requested"
    assert state["last_signal_target"] == "process_group"


def test_governed_save_only_signals_verified_rank_zero_training_process(
    tmp_path: Path,
) -> None:
    settings = _governed_settings(tmp_path)
    profile = settings.profiles[0]
    controller = DashboardController(settings)
    state = {
        "schema_version": 1,
        "profile_id": profile.profile_id,
        "run_id": profile.run_id,
        "pid": 5151,
        "process_group_id": 5151,
        "launch_kind": "governed_v4",
        "governed_plan_id": profile.governed_plan_id,
        "status": "running",
    }

    with (
        patch.object(controller, "_controller_state", return_value=state),
        patch.object(
            controller,
            "_verified_rank0_training_pid",
            return_value=6161,
        ) as verified_rank0,
        patch("twen.web.os.kill") as kill_process,
        patch("twen.web.os.killpg") as kill_group,
    ):
        result = controller.signal(
            profile.profile_id,
            "save",
            profile.save_confirmation,
        )

    kill_process.assert_called_once_with(6161, signal.SIGUSR1)
    kill_group.assert_not_called()
    verified_rank0.assert_called_once_with(profile, state)
    assert result["signal_target"] == "process"
    assert result["process_group_id"] is None


def test_governed_rank0_authentication_binds_session_process_plan_and_lock(
    tmp_path: Path,
) -> None:
    settings = _governed_settings(tmp_path)
    profile = settings.profiles[0]
    profile.run_dir.mkdir(parents=True)
    plan, governed_state, session, active_command, identity, owner = _governed_rank0_identity(
        profile, tmp_path.resolve()
    )

    def read_identity(path: Path) -> dict[str, object] | None:
        if path == profile.governed_state_path:
            return dict(governed_state)
        if path == profile.run_dir / "rank0-session.json":
            return dict(session)
        return None

    controller = DashboardController(settings)
    controller_state = {"status": "running"}
    with (
        patch.object(controller, "_governed_plan", return_value=plan),
        patch.object(
            controller,
            "_verified_governed_process_group",
            return_value=(5151, 5151),
        ),
        patch(
            "twen.web.governed_controller_status",
            return_value={"controller_state": "running"},
        ),
        patch(
            "twen.web._expected_governed_active_command",
            return_value=active_command,
        ),
        patch("twen.web._read_json_regular_no_follow", side_effect=read_identity),
        patch("twen.web._read_linux_process_identity", return_value=identity),
        patch("twen.web._held_run_lock_owner", return_value=owner),
    ):
        assert controller._verified_rank0_training_pid(profile, controller_state) == 6161


@pytest.mark.parametrize(
    "mismatch",
    [
        "missing_rank",
        "python_c",
        "pid_reuse",
        "wrong_pgid",
        "wrong_cwd",
        "wrong_config",
        "missing_lock",
        "wrong_lock_owner",
        "nonrunning_governed_state",
        "wrong_active_command",
    ],
)
def test_governed_save_rejects_any_weak_or_mismatched_rank0_identity(
    tmp_path: Path,
    mismatch: str,
) -> None:
    settings = _governed_settings(tmp_path)
    profile = settings.profiles[0]
    profile.run_dir.mkdir(parents=True)
    plan, governed_state, session, active_command, identity, owner = _governed_rank0_identity(
        profile, tmp_path.resolve()
    )
    expected_active_command = list(active_command)
    if mismatch == "missing_rank":
        session.pop("rank")
    elif mismatch == "python_c":
        payload_index = next(
            index
            for index in range(len(active_command) - 1)
            if active_command[index : index + 2] == ["-m", "twen.cli"]
        )
        malicious = [
            sys.executable,
            "-c",
            "print('not a trainer')",
            *active_command[payload_index:],
        ]
        session["process_cmdline"] = malicious
        identity = _LinuxProcessIdentity(
            pid=identity.pid,
            start_time_ticks=identity.start_time_ticks,
            command=tuple(malicious),
            cwd=identity.cwd,
            executable=identity.executable,
            process_group_id=identity.process_group_id,
        )
    elif mismatch == "pid_reuse":
        identity = _LinuxProcessIdentity(
            pid=identity.pid,
            start_time_ticks=identity.start_time_ticks + 1,
            command=identity.command,
            cwd=identity.cwd,
            executable=identity.executable,
            process_group_id=identity.process_group_id,
        )
    elif mismatch == "wrong_pgid":
        identity = _LinuxProcessIdentity(
            pid=identity.pid,
            start_time_ticks=identity.start_time_ticks,
            command=identity.command,
            cwd=identity.cwd,
            executable=identity.executable,
            process_group_id=6161,
        )
    elif mismatch == "wrong_cwd":
        identity = _LinuxProcessIdentity(
            pid=identity.pid,
            start_time_ticks=identity.start_time_ticks,
            command=identity.command,
            cwd=(tmp_path / "other").resolve(),
            executable=identity.executable,
            process_group_id=identity.process_group_id,
        )
    elif mismatch == "wrong_config":
        session["config_sha256"] = "0" * 64
    elif mismatch == "missing_lock":
        owner = None
    elif mismatch == "wrong_lock_owner":
        owner["pid"] = 9999
    elif mismatch == "nonrunning_governed_state":
        governed_state["status"] = "awaiting_evaluation"
    elif mismatch == "wrong_active_command":
        governed_state["active_command"] = [*active_command, "--unexpected"]
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(mismatch)

    def read_identity(path: Path) -> dict[str, object] | None:
        if path == profile.governed_state_path:
            return dict(governed_state)
        if path == profile.run_dir / "rank0-session.json":
            return dict(session)
        return None

    controller_state = {
        "schema_version": 1,
        "profile_id": profile.profile_id,
        "run_id": profile.run_id,
        "status": "running",
    }
    controller = DashboardController(settings)
    with (
        patch.object(controller, "_controller_state", return_value=controller_state),
        patch.object(controller, "_governed_plan", return_value=plan),
        patch.object(
            controller,
            "_verified_governed_process_group",
            return_value=(5151, 5151),
        ),
        patch(
            "twen.web.governed_controller_status",
            return_value={"controller_state": "running"},
        ),
        patch(
            "twen.web._expected_governed_active_command",
            return_value=expected_active_command,
        ),
        patch("twen.web._read_json_regular_no_follow", side_effect=read_identity),
        patch("twen.web._read_linux_process_identity", return_value=identity),
        patch("twen.web._held_run_lock_owner", return_value=owner),
        patch("twen.web.os.kill") as kill_process,
        pytest.raises(DashboardError),
    ):
        controller.signal(
            profile.profile_id,
            "save",
            profile.save_confirmation,
        )

    kill_process.assert_not_called()
    assert not controller._state_path.exists()
    assert not controller._audit_path.exists()


def test_governed_root_remains_active_without_rank_zero_during_evaluation(
    tmp_path: Path,
) -> None:
    settings = _governed_settings(tmp_path)
    profile = settings.profiles[0]
    controller = DashboardController(settings)
    controller._state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_id": profile.profile_id,
                "run_id": profile.run_id,
                "pid": 5151,
                "process_group_id": 5151,
                "launch_kind": "governed_v4",
                "status": "running",
            }
        ),
        encoding="utf-8",
    )
    governance = {
        "blocked": False,
        "configuration_stale": False,
        "controller_state": "awaiting_evaluation",
        "required_ack": f"RUN {profile.governed_plan_id}",
        "blockers": [],
    }

    with (
        patch("twen.web._process_matches_governed_controller", return_value=True),
        patch.object(controller, "_governance_status", return_value=governance),
    ):
        status = controller.profile_status(profile)

    assert status["state"] == "running"
    assert status["active"] is True
    assert status["stop_available"] is True
    assert status["save_available"] is False
    assert status["governed_controller_enforced"] is True


def test_discovery_includes_history_evaluation_and_omits_reports(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    history = tmp_path / "runs/older"
    history.mkdir(parents=True)
    (history / "metrics.jsonl").write_text('{"step":4,"loss":1.25}\n', encoding="utf-8")
    evaluation = tmp_path / "artifacts/evaluations/base-v1"
    evaluation.mkdir(parents=True)
    (evaluation / "PLAN.json").write_text("{}\n", encoding="utf-8")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "base-v1.html").write_text("<h1>report</h1>", encoding="utf-8")
    kd_status = tmp_path / "artifacts/data/base-v2-500m-kd-orchestration/status.json"
    kd_status.parent.mkdir(parents=True)
    kd_status.write_text(
        json.dumps(
            {
                "kind": "twen_base_v2_500m_kd_orchestration_status",
                "status": "running",
                "phase": "generate-kd",
                "progress": {
                    "percent": 25.0,
                    "completed_tokens": 125_000_000,
                    "total_tokens": 500_000_000,
                    "attempt_wall_tokens_per_second": 9_100.0,
                    "eta_seconds": 41_208.0,
                },
            }
        ),
        encoding="utf-8",
    )
    controller = DashboardController(settings)
    catalog = controller.run_catalog()
    older = next(item for item in catalog if item["history"] and item["run_id"] == "older")
    assert older["preflight_enforced"] is True
    assert older["last_update_unix"] is not None
    assert older["last_update_utc"] is not None
    operations = controller.operations_status()
    assert operations["data_jobs"][0]["status"] == "running"
    assert "25.00%" in operations["data_jobs"][0]["detail"]
    assert "9,100 tok/s" in operations["data_jobs"][0]["detail"]
    assert operations["evaluations"][0]["status"] == "paused"
    assert operations["reports"] == []


def test_incomplete_evaluation_is_active_only_while_eval_lock_is_held(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    evaluation = tmp_path / "artifacts/evaluations/base-v1"
    evaluation.mkdir(parents=True)
    _write_evaluation_plan(evaluation, device_type="cuda")
    controller = DashboardController(settings)

    paused_operations = controller.operations_status()
    assert paused_operations["evaluations"][0]["status"] == "paused"
    paused_task = controller.task_catalog()[0]
    assert paused_task["kind"] == "evaluation"
    assert paused_task["state"] == "paused"
    assert paused_task["active"] is False
    paused_snapshot = controller.snapshot(paused_task["key"])
    assert paused_snapshot["active_task_key"] is None
    assert paused_snapshot["live_gpu_relevant"] is False
    assert paused_snapshot["gpu_telemetry"]["associated_task_key"] is None

    with FileLock(evaluation / ".eval.lock", timeout_seconds=0):
        running_operations = controller.operations_status()
        assert running_operations["evaluations"][0]["status"] == "in_progress"
        running_task = controller.task_catalog()[0]
        assert running_task["state"] == "running"
        assert running_task["active"] is True
        assert running_task["gpu_relevant"] is True
        running_snapshot = controller.snapshot(running_task["key"])
        assert running_snapshot["active_task_key"] == running_task["key"]
        assert running_snapshot["live_gpu_relevant"] is True
        assert running_snapshot["gpu_telemetry"]["associated_task_key"] == running_task["key"]

    resumed_paused = controller.task_catalog()[0]
    assert resumed_paused["state"] == "paused"
    assert resumed_paused["active"] is False


@pytest.mark.parametrize(
    ("device_type", "valid_fingerprint"),
    [
        ("cpu", True),
        (None, True),
        ("cuda", False),
    ],
)
def test_held_cpu_or_unauthenticated_evaluation_never_owns_gpu(
    tmp_path: Path,
    device_type: str | None,
    valid_fingerprint: bool,
) -> None:
    settings = _settings(tmp_path)
    evaluation = tmp_path / "artifacts/evaluations/base-v1"
    evaluation.mkdir(parents=True)
    _write_evaluation_plan(
        evaluation,
        device_type=device_type,
        valid_fingerprint=valid_fingerprint,
    )
    controller = DashboardController(settings)

    with FileLock(evaluation / ".eval.lock", timeout_seconds=0):
        task = controller.task_catalog()[0]
        assert task["state"] == "running"
        assert task["active"] is True
        assert task["gpu_relevant"] is False
        snapshot = controller.snapshot(task["key"])
        assert snapshot["active_task_key"] == task["key"]
        assert snapshot["live_gpu_relevant"] is False
        assert snapshot["gpu_telemetry"]["associated_task_key"] is None


def test_held_lock_only_evaluation_is_discovered_without_claiming_gpu(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    evaluation = tmp_path / "artifacts/evaluations/initializing"
    evaluation.mkdir(parents=True)
    controller = DashboardController(settings)

    with FileLock(evaluation / ".eval.lock", timeout_seconds=0):
        tasks = controller.task_catalog()
        assert len(tasks) == 1
        task = tasks[0]
        assert task["kind"] == "evaluation"
        assert task["state"] == "running"
        assert task["active"] is True
        assert task["gpu_relevant"] is False
        assert task["source"]["plan"] is None
        snapshot = controller.snapshot(task["key"])
        assert snapshot["active_task_key"] == task["key"]
        assert snapshot["gpu_telemetry"]["associated_task_key"] is None


def test_held_lock_overrides_completed_evaluation_until_release(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    evaluation = tmp_path / "artifacts/evaluations/recheck"
    evaluation.mkdir(parents=True)
    _write_evaluation_plan(evaluation, device_type="cuda")
    (evaluation / "manifest.json").write_text("{}\n", encoding="utf-8")
    (evaluation / "COMPLETE").write_text("committed\n", encoding="utf-8")
    controller = DashboardController(settings)

    assert controller.operations_status()["evaluations"][0]["status"] == "complete"
    with FileLock(evaluation / ".eval.lock", timeout_seconds=0):
        operation = controller.operations_status()["evaluations"][0]
        assert operation["status"] == "in_progress"
        assert operation["gpu_relevant"] is True
        task = controller.task_catalog()[0]
        assert task["state"] == "running"
        assert task["gpu_relevant"] is True
    assert controller.operations_status()["evaluations"][0]["status"] == "complete"
    assert controller.task_catalog()[0]["state"] == "completed"


def test_cpu_evaluation_cannot_steal_gpu_owner_from_training(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    profile = settings.profiles[0]
    profile.run_dir.mkdir(parents=True)
    (profile.run_dir / "rank0-session.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "running",
                "hostname": platform.node(),
                "pid": 31337,
            }
        ),
        encoding="utf-8",
    )
    evaluation = tmp_path / "artifacts/evaluations/cpu"
    evaluation.mkdir(parents=True)
    plan_path = _write_evaluation_plan(evaluation, device_type="cpu")
    future = time.time() + 60
    os.utime(plan_path, (future, future))
    controller = DashboardController(settings)

    with (
        FileLock(evaluation / ".eval.lock", timeout_seconds=0),
        patch("twen.web._process_matches_profile", return_value=True),
    ):
        tasks = controller.task_catalog()
        evaluation_task = next(task for task in tasks if task["kind"] == "evaluation")
        training_task = next(task for task in tasks if task["kind"] == "training")
        snapshot = controller.snapshot(evaluation_task["key"])
    assert snapshot["active_task_key"] == evaluation_task["key"]
    assert snapshot["live_gpu_relevant"] is False
    assert snapshot["gpu_telemetry"]["associated_task_key"] == training_task["key"]
    assert snapshot["gpu_telemetry"]["associated_task_kind"] == "training"


def test_read_only_probes_fail_closed_without_required_flags(tmp_path: Path) -> None:
    path = tmp_path / "regular"
    path.write_text("fixed\n", encoding="utf-8")
    for missing_flag in ("O_NOFOLLOW", "O_NONBLOCK"):
        with patch.object(os, missing_flag, None):
            assert _sha256_regular_file_no_follow(path) is None
            assert _exclusive_advisory_lock_is_held(path) is False


def test_read_only_probes_fail_closed_for_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("fixed\n", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)

    assert _sha256_regular_file_no_follow(link) is None
    assert _exclusive_advisory_lock_is_held(link) is False


def test_symlinked_evaluation_lock_is_not_discovered_as_active(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    evaluation = tmp_path / "artifacts/evaluations/symlinked"
    evaluation.mkdir(parents=True)
    real_lock = tmp_path / "real-eval.lock"
    (evaluation / ".eval.lock").symlink_to(real_lock)
    controller = DashboardController(settings)

    with FileLock(real_lock, timeout_seconds=0):
        assert controller.operations_status()["evaluations"] == []
        assert controller.task_catalog() == []


def test_read_only_probes_do_not_block_on_fifo(tmp_path: Path) -> None:
    fifo = tmp_path / "probe.fifo"
    os.mkfifo(fifo)

    def run_probe(probe: object, result: list[object]) -> None:
        assert callable(probe)
        result.append(probe(fifo))

    for probe, expected in (
        (_sha256_regular_file_no_follow, None),
        (_exclusive_advisory_lock_is_held, False),
    ):
        result: list[object] = []
        thread = threading.Thread(target=run_probe, args=(probe, result), daemon=True)
        thread.start()
        thread.join(timeout=0.5)
        assert not thread.is_alive()
        assert result == [expected]


def test_unlock_failure_degrades_evaluation_to_paused_without_status_error(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    evaluation = tmp_path / "artifacts/evaluations/base-v1"
    evaluation.mkdir(parents=True)
    _write_evaluation_plan(evaluation, device_type="cuda")
    (evaluation / ".eval.lock").touch()
    controller = DashboardController(settings)

    def fail_unlock(_descriptor: int, operation: int) -> None:
        if operation == fcntl.LOCK_UN:
            raise OSError("simulated unlock failure")

    with patch("twen.web.fcntl.flock", side_effect=fail_unlock):
        operation = controller.operations_status()["evaluations"][0]
    assert operation["status"] == "paused"
    assert operation["gpu_relevant"] is False


def test_unified_tasks_select_running_kd_and_bind_live_gpu_to_it(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    profile = settings.profiles[0]
    profile.run_dir.mkdir(parents=True)
    (profile.run_dir / "rank0-session.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "hostname": platform.node(),
                "pid": 1234,
                "run_id": "base-v1",
                "stage": "dense-oracle",
            }
        ),
        encoding="utf-8",
    )
    (profile.run_dir / "metrics.jsonl").write_text(
        '{"step":383,"tokens":100151046,"loss":3.2075}\n',
        encoding="utf-8",
    )
    kd_root = tmp_path / "artifacts/data/base-v2-500m-kd-orchestration"
    (kd_root / "logs").mkdir(parents=True)
    attempt_log = kd_root / "logs/attempt-004-generate-kd.log"
    attempt_log.write_text("live KD output\n", encoding="utf-8")
    (kd_root / "status.json").write_text(
        json.dumps(
            {
                "kind": "twen_base_v2_500m_kd_orchestration_status",
                "status": "running",
                "phase": "generate-kd",
                "attempt": 4,
                "log": str(attempt_log),
                "optimizer_created": False,
                "training_started": False,
                "command": ["python", "--secret-path", "/outside/private"],
                "progress": {
                    "percent": 16.35,
                    "fraction": 0.1635,
                    "completed_tokens": 84_489_972,
                    "total_tokens": 516_719_389,
                    "attempt_wall_tokens_per_second": 8_727.8,
                    "eta_seconds": 49_523.0,
                    "completed_shards": 104,
                    "total_shards": 641,
                    "completed_sequences": 20_671,
                    "total_sequences": 126_457,
                },
            }
        ),
        encoding="utf-8",
    )
    failed_root = tmp_path / "artifacts/data/failed-pipeline"
    failed_root.mkdir()
    (failed_root / "status.json").write_text(
        json.dumps({"status": "failed"}),
        encoding="utf-8",
    )
    gpu_monitor = GpuTelemetryMonitor()
    gpu_monitor._record_output_line(_gpu_line(590.0))
    controller = DashboardController(settings, gpu_monitor=gpu_monitor)

    selection = controller.task_selection()
    assert selection["active_task_key"] == selection["default_task_key"]
    assert selection["tasks"][0]["kind"] == "kd"
    assert selection["tasks"][0]["state"] == "running"
    assert {task["state"] for task in selection["tasks"]} <= {"running", "completed"}
    assert not any("failed-pipeline" in task["label"] for task in selection["tasks"])

    snapshot = controller.snapshot()
    kd_key = selection["active_task_key"]
    assert snapshot["task"]["key"] == kd_key
    assert snapshot["task_data"]["kind"] == "kd"
    assert snapshot["task_data"]["attempt"] == 4
    assert snapshot["task_data"]["progress"]["completed_tokens"] == 84_489_972
    assert snapshot["task_data"]["progress"]["tokens_per_second"] == 8_727.8
    assert snapshot["task_data"]["progress"]["eta_seconds"] == 49_523.0
    assert snapshot["task_data"]["progress"]["completed_shards"] == 104
    assert snapshot["console"] == "live KD output"
    assert snapshot["live_gpu_relevant"] is True
    assert snapshot["gpu_telemetry"]["associated_task_key"] == kd_key
    assert snapshot["gpu_telemetry"]["associated_task_kind"] == "kd"
    serialized = json.dumps(snapshot["task_data"])
    assert "/outside/private" not in serialized
    assert str(tmp_path) not in serialized

    completed_key = next(task["key"] for task in selection["tasks"] if task["kind"] == "training")
    historical = controller.snapshot(completed_key)
    assert historical["task"]["state"] == "completed"
    assert historical["live_gpu_relevant"] is False
    assert historical["gpu_telemetry"]["associated_task_key"] == kd_key


@pytest.mark.parametrize(
    ("status_kind", "phase"),
    [
        ("twen_base_v2_500m_kd_orchestration_status", "index-kd"),
        ("kd_orchestration_status", "generate-kd"),
    ],
)
def test_non_gpu_or_unauthenticated_kd_status_cannot_claim_gpu(
    tmp_path: Path,
    status_kind: str,
    phase: str,
) -> None:
    settings = _settings(tmp_path)
    kd_root = tmp_path / "artifacts/data/candidate-kd-orchestration"
    kd_root.mkdir(parents=True)
    (kd_root / "status.json").write_text(
        json.dumps(
            {
                "kind": status_kind,
                "status": "running",
                "phase": phase,
                "training_started": False,
                "optimizer_created": False,
            }
        ),
        encoding="utf-8",
    )
    controller = DashboardController(settings)

    task = next(task for task in controller.task_catalog() if task["kind"] == "kd")
    snapshot = controller.snapshot(task["key"])

    assert task["active"] is True
    assert task["gpu_relevant"] is False
    assert snapshot["active_task_key"] == task["key"]
    assert snapshot["live_gpu_relevant"] is False
    assert snapshot["gpu_telemetry"]["associated_task_key"] is None


def test_running_cpu_data_pipeline_cannot_steal_gpu_owner_from_training(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    profile = settings.profiles[0]
    profile.run_dir.mkdir(parents=True)
    (profile.run_dir / "rank0-session.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "running",
                "hostname": platform.node(),
                "pid": 31337,
            }
        ),
        encoding="utf-8",
    )
    pipeline = tmp_path / "artifacts/data/formal-pipeline"
    pipeline.mkdir(parents=True)
    status_path = pipeline / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "kind": "formal_data_pipeline_status",
                "status": "running",
                "phase": "audit",
            }
        ),
        encoding="utf-8",
    )
    future = time.time() + 60
    os.utime(status_path, (future, future))
    controller = DashboardController(settings)

    with patch("twen.web._process_matches_profile", return_value=True):
        tasks = controller.task_catalog()
        pipeline_task = next(task for task in tasks if task["kind"] == "data_pipeline")
        training_task = next(task for task in tasks if task["kind"] == "training")
        snapshot = controller.snapshot(pipeline_task["key"])
    assert pipeline_task["active"] is True
    assert pipeline_task["gpu_relevant"] is False
    assert snapshot["active_task_key"] == pipeline_task["key"]
    assert snapshot["live_gpu_relevant"] is False
    assert snapshot["gpu_telemetry"]["associated_task_key"] == training_task["key"]


def test_task_console_rejects_status_declared_path_outside_task_root(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    outside = tmp_path / "private.log"
    outside.write_text("must not leak\n", encoding="utf-8")
    kd_root = tmp_path / "artifacts/data/safe-kd-orchestration"
    kd_root.mkdir(parents=True)
    (kd_root / "console.log").write_text("confined fallback\n", encoding="utf-8")
    (kd_root / "status.json").write_text(
        json.dumps(
            {
                "kind": "twen_base_v2_500m_kd_orchestration_status",
                "status": "running",
                "log": str(outside),
                "phase": str(outside),
                "progress": {"completed_tokens": 1, "total_tokens": 2},
            }
        ),
        encoding="utf-8",
    )
    controller = DashboardController(settings)

    snapshot = controller.snapshot()

    assert snapshot["console"] == "confined fallback"
    assert snapshot["task"]["source"]["console"].endswith("/console.log")
    assert str(outside) not in json.dumps(snapshot["task_data"])

    (kd_root / "status.json").write_text(
        json.dumps({"kind": "kd_orchestration_status", "status": "complete"}),
        encoding="utf-8",
    )
    completed = controller.snapshot()
    assert completed["active_task_key"] is None
    assert completed["live_gpu_relevant"] is False
    assert completed["gpu_telemetry"]["associated_task_key"] is None
    assert completed["gpu_telemetry"]["associated_task_kind"] is None


def test_historical_run_discovery_and_log_tail_skip_symlink_escapes(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    settings = _settings(project)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "metrics.jsonl").write_text('{"step":99,"loss":0.01}\n', encoding="utf-8")

    escaped_run = project / "runs/escaped"
    escaped_run.symlink_to(outside, target_is_directory=True)
    history = project / "runs/history"
    history.mkdir()
    (history / "resolved_config.yaml").write_text("fixed\n", encoding="utf-8")
    (history / "metrics.jsonl").symlink_to(outside / "metrics.jsonl")

    controller = DashboardController(settings)
    catalog = controller.run_catalog()

    assert not any(item.get("run_id") == "escaped" for item in catalog)
    safe_history = next(item for item in catalog if item.get("run_id") == "history")
    assert safe_history["latest_metric"] is None
    snapshot = controller.snapshot(safe_history["key"])
    assert snapshot["metrics"] == []


def test_http_home_bootstrap_snapshot_and_csrf_guard(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    profile = settings.profiles[0]
    profile.run_dir.mkdir(parents=True)
    (profile.run_dir / "metrics.jsonl").write_text(
        '{"step":1,"tokens":4096,"loss":2.5}\n', encoding="utf-8"
    )
    kd_root = tmp_path / "artifacts/data/base-v2-kd-orchestration"
    kd_root.mkdir(parents=True)
    (kd_root / "console.log").write_text("KD live\n", encoding="utf-8")
    (kd_root / "status.json").write_text(
        json.dumps(
            {
                "kind": "kd_orchestration_status",
                "status": "running",
                "phase": "generate-kd",
                "progress": {"completed_tokens": 10, "total_tokens": 100},
            }
        ),
        encoding="utf-8",
    )
    gpu_monitor = GpuTelemetryMonitor()
    gpu_monitor._record_output_line(_gpu_line())
    controller = DashboardController(
        settings,
        gpu_monitor=gpu_monitor,
    )
    try:
        server = create_dashboard_server(
            settings,
            port=0,
            controller=controller,
            csrf_token="csrf-test",
        )
    except PermissionError:
        pytest.skip("sandbox forbids loopback sockets")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(root + "/", timeout=3) as response:
            assert response.status == 200
            assert b"<!doctype html>" in response.read()
            assert response.headers["X-Frame-Options"] == "DENY"
        with urlopen(root + "/api/bootstrap", timeout=3) as response:
            bootstrap = json.load(response)
        assert bootstrap["csrf_token"] == "csrf-test"
        assert bootstrap["poll_interval_ms"] == 1000
        assert bootstrap["runs"][0]["key"] == "profile:base-v2"
        assert bootstrap["tasks"][0]["kind"] == "kd"
        assert bootstrap["active_task_key"] == bootstrap["tasks"][0]["key"]
        assert bootstrap["default_task_key"] == bootstrap["active_task_key"]
        assert bootstrap["configuration_stale"] is False
        assert bootstrap["restart_required"] is False
        assert bootstrap["profiles"][0]["configuration_stale"] is False
        assert bootstrap["profiles"][0]["restart_required"] is False
        assert bootstrap["control_policy"] == {
            "automatic_launch": False,
            "server_side_allowlist": True,
            "preflight_mandatory": True,
            "formal_v4_governed_route": "/api/governed/start",
            "formal_v4_direct_train_refused": True,
            "formal_v4_exact_plan_ack_required": True,
            "csrf_required": True,
            "authentication_required": False,
        }
        with urlopen(root + "/api/snapshot?run=profile%3Abase-v2", timeout=3) as response:
            snapshot = json.load(response)
        assert snapshot["metrics"][-1]["loss"] == 2.5
        assert snapshot["gpu_telemetry"]["latest"]["power_draw_w"] == 401.25
        assert snapshot["gpu_telemetry"]["latest"]["power_limit_w"] == 600.0
        assert snapshot["gpu_telemetry"]["sample_interval_seconds"] == 0.1
        assert snapshot["runs"][0]["key"] == "profile:base-v2"
        assert snapshot["live_gpu_relevant"] is False
        assert snapshot["gpu_telemetry"]["associated_task_kind"] == "kd"

        kd_key = bootstrap["active_task_key"]
        with urlopen(
            root + "/api/snapshot?task=" + kd_key + "&run=profile%3Abase-v2",
            timeout=3,
        ) as response:
            kd_snapshot = json.load(response)
        assert kd_snapshot["task"]["kind"] == "kd"
        assert kd_snapshot["console"] == "KD live"
        assert kd_snapshot["live_gpu_relevant"] is True

        settings.dashboard_config_path.write_text(
            '{"schema_version":1,"changed":true}\n',
            encoding="utf-8",
        )
        with urlopen(root + "/api/bootstrap", timeout=3) as response:
            stale_bootstrap = json.load(response)
        assert stale_bootstrap["configuration_stale"] is True
        assert stale_bootstrap["restart_required"] is True
        assert stale_bootstrap["profiles"][0]["configuration_stale"] is True
        assert stale_bootstrap["profiles"][0]["restart_required"] is True

        request = Request(
            root + "/api/start",
            method="POST",
            data=json.dumps({"profile_id": "base-v2", "confirmation": "START base-v2"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(HTTPError) as denied:
            urlopen(request, timeout=3)
        assert denied.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_http_basic_auth_guards_every_page_and_api(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    auth = DashboardAuth("twen", "p" * 32, tmp_path / "auth.json")
    try:
        server = create_dashboard_server(settings, port=0, auth=auth)
    except PermissionError:
        pytest.skip("sandbox forbids loopback sockets")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with pytest.raises(HTTPError) as denied:
            urlopen(root + "/", timeout=3)
        assert denied.value.code == 401
        assert denied.value.headers["WWW-Authenticate"].startswith("Basic ")

        encoded = base64.b64encode(f"{auth.username}:{auth.password}".encode()).decode()
        request = Request(root + "/api/bootstrap", headers={"Authorization": f"Basic {encoded}"})
        with urlopen(request, timeout=3) as response:
            assert json.load(response)["ok"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_dashboard_refuses_non_loopback_bind(tmp_path: Path) -> None:
    with pytest.raises(DashboardError, match="requires --auth-file"):
        create_dashboard_server(_settings(tmp_path), host="0.0.0.0", port=8765)


def test_dashboard_auth_file_is_private_and_stable(tmp_path: Path) -> None:
    path = tmp_path / "http-auth.json"
    first = ensure_dashboard_auth_file(path, username="twen")
    second = ensure_dashboard_auth_file(path, username="ignored")
    auth = load_dashboard_auth(path)

    assert first["created"] is True
    assert second["created"] is False
    assert auth.username == "twen"
    assert len(auth.password) >= 24
    assert path.stat().st_mode & 0o077 == 0

    path.chmod(0o644)
    with pytest.raises(DashboardError, match="chmod 600"):
        load_dashboard_auth(path)


def test_non_loopback_server_requires_auth_before_binding(tmp_path: Path) -> None:
    auth = DashboardAuth("twen", "x" * 32, tmp_path / "auth.json")
    with patch("twen.web.DashboardHTTPServer") as server:
        create_dashboard_server(
            _settings(tmp_path),
            host="0.0.0.0",
            port=8765,
            auth=auth,
        )
    assert server.call_args.kwargs["auth"] is auth
    assert server.call_args.kwargs["public_bind"] is True


def test_serve_dashboard_sigterm_stops_and_flushes_gpu_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)

    class Monitor:
        def __init__(self) -> None:
            self.flushed = False

        def run_until_stopped(self, stop: threading.Event) -> None:
            assert stop.wait(timeout=2)
            self.flushed = True

    monitor = Monitor()

    class Server:
        controller = SimpleNamespace(gpu_monitor=monitor)
        server_address = ("0.0.0.0", 8765)
        closed = False

        def serve_forever(self, *, poll_interval: float) -> None:
            assert poll_interval == 0.5
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler)
            handler(signal.SIGTERM, None)

        def server_close(self) -> None:
            self.closed = True

    server = Server()
    previous = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr("twen.web.load_dashboard_settings", lambda _path: settings)
    monkeypatch.setattr("twen.web.create_dashboard_server", lambda *args, **kwargs: server)

    serve_dashboard("unused.json", host="0.0.0.0", port=8765)

    assert monitor.flushed is True
    assert server.closed is True
    assert signal.getsignal(signal.SIGTERM) is previous
