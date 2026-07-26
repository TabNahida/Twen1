"""Command line interface for Twen1.

Importing this module is side-effect free: it does not initialize CUDA, access
the network, or construct a model.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import ConfigError, load_train_config


def _print_json(value: Any) -> None:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _enable_expandable_segments_allocator() -> None:
    """Honor the runtime allocator contract without discarding user options.

    ``setdefault`` is insufficient here: a useful pre-existing option such as
    ``max_split_size_mb`` would silently prevent ``expandable_segments`` from
    being enabled even though the resolved training config says it is active.
    Replace any explicit true/false value and preserve every unrelated option.
    """

    key = "PYTORCH_ALLOC_CONF"
    options = [option.strip() for option in os.environ.get(key, "").split(",") if option.strip()]
    options = [
        option for option in options if option.split(":", 1)[0].strip() != "expandable_segments"
    ]
    options.append("expandable_segments:True")
    os.environ[key] = ",".join(options)


def _cmd_config_validate(args: argparse.Namespace) -> int:
    config = load_train_config(args.config)
    _print_json(
        {
            "ok": True,
            "run_id": config.run_id,
            "stage": config.stage,
            "track": config.track,
            "critical_fingerprint": config.fingerprint(),
        }
    )
    return 0


def _cmd_config_finalize_base_v2(args: argparse.Namespace) -> int:
    from .v2_finalizer import finalize_base_v2, options_from_namespace

    _print_json(finalize_base_v2(options_from_namespace(args)))
    return 0


def _cmd_proxy_check(args: argparse.Namespace) -> int:
    from .io.proxy import apply_proxy_environment, check_proxy_connectivity

    settings = apply_proxy_environment(proxy_url=args.proxy)
    result = check_proxy_connectivity(settings, timeout_seconds=args.timeout)
    _print_json(
        {"ok": True, **dataclasses.asdict(result), "environment": settings.as_environment()}
    )
    return 0


def _cmd_hardware_inspect(args: argparse.Namespace) -> int:
    config = load_train_config(args.config) if args.config else None
    if config is not None and config.runtime.expandable_segments:
        _enable_expandable_segments_allocator()
    from .hardware import inspect_hardware

    _print_json(inspect_hardware(config))
    return 0


def _cmd_download_set(args: argparse.Namespace) -> int:
    from .io.download import ArtifactSpec, DownloadManager, write_download_set_manifest
    from .io.proxy import ProxySettings

    settings = ProxySettings.from_environment(proxy_url=args.proxy)
    payload = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError("download spec must contain a non-empty artifacts list")
    specs = [ArtifactSpec(**item) for item in raw_artifacts]
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    manager = DownloadManager(
        proxy_settings=ProxySettings.from_environment(proxy_url=args.proxy),
        check_proxy=True,
        network_policy=args.network_policy,
    )
    for spec in specs:
        manager.download(spec, root / spec.filename)
    manifest = write_download_set_manifest(root / "download-manifest.json", specs)
    from .utils import sha256_file

    _print_json(
        {
            "ok": True,
            "files": len(specs),
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "resolved_revision": specs[0].revision,
            "network_policy": manager.effective_network_policy,
            "github_proxy": settings.http_proxy,
            "proxy_fallback_used": manager.proxy_fallback_used,
        }
    )
    return 0


def _cmd_download_lock_model(args: argparse.Namespace) -> int:
    from .io import (
        DownloadManager,
        ProxySettings,
        resolve_model_manifest,
        write_resolved_model_manifest,
    )

    manager = DownloadManager(
        proxy_settings=ProxySettings.from_environment(proxy_url=args.proxy),
        check_proxy=True,
        network_policy=args.network_policy,
    )
    resolved = resolve_model_manifest(
        args.provider,
        args.model_id,
        args.revision,
        token=args.token,
        manager=manager,
    )
    output = write_resolved_model_manifest(args.output, resolved)
    _print_json(
        {
            "ok": True,
            "model_id": resolved.model_id,
            "resolved_revision": resolved.resolved_revision,
            "files": len(resolved.artifacts),
            "output": str(output),
            "network_policy": manager.effective_network_policy,
            "proxy_fallback_used": manager.proxy_fallback_used,
        }
    )
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    from .preflight import run_training_preflight

    config = load_train_config(args.config)
    report = run_training_preflight(config, world_size=args.world_size)
    _print_json(report)
    return 0


def _cmd_data_index_kd(args: argparse.Namespace) -> int:
    from .training.data import build_kd_index
    from .utils import sha256_file

    output = build_kd_index(
        args.root,
        args.output,
        temperature=args.temperature,
        prepared_manifest=args.prepared_manifest,
    )
    _print_json({"ok": True, "manifest": str(output), "sha256": sha256_file(output)})
    return 0


def _cmd_data_prepare(args: argparse.Namespace) -> int:
    from .data import prepare_jsonl_corpus, read_prepared_manifest
    from .utils import sha256_file

    output = prepare_jsonl_corpus(
        args.input,
        args.output,
        tokenizer_path=args.tokenizer,
        tokenizer_sha256=args.tokenizer_manifest_sha256,
        sequence_length=args.sequence_length,
        text_field=args.text_field,
        progress=args.progress,
        extracted_manifest=args.extracted_manifest,
        role=args.role,
        allow_pending_research_audits=args.allow_pending_research_audits,
        audit_attestation=args.audit_attestation,
    )
    prepared = read_prepared_manifest(output)
    lineage = prepared.lineage or {}
    _print_json(
        {
            "ok": True,
            "manifest": str(output),
            "sha256": sha256_file(output),
            "lineage_kind": lineage.get("kind"),
            "research_only": lineage.get("research_only"),
            "pending_audits": lineage.get("pending_audits", []),
        }
    )
    return 0


def _cmd_data_inspect_benchmark_registry(args: argparse.Namespace) -> int:
    from .data import inspect_benchmark_registry

    _print_json(
        inspect_benchmark_registry(
            args.registry,
            benchmark_root=args.benchmark_root,
            verify_hashes=not args.no_verify_hashes,
        )
    )
    return 0


def _cmd_data_audit_base(args: argparse.Namespace) -> int:
    from .data import build_base_audit_attestation, validate_base_audit_attestation
    from .utils import sha256_file

    output = build_base_audit_attestation(
        args.extracted_manifest,
        args.frozen_validation_manifest,
        args.benchmark_registry,
        args.benchmark_root,
        args.output,
        near_duplicate_threshold=args.near_duplicate_threshold,
        max_findings=args.max_findings,
    )
    value = validate_base_audit_attestation(output)
    _print_json(
        {
            "ok": True,
            "attestation": str(output),
            "sha256": sha256_file(output),
            "attestation_fingerprint": value["attestation_fingerprint"],
            "ready_for_training": value["ready_for_training"],
            "gates": value["gates"],
        }
    )
    return 0


def _cmd_data_materialize_audit(args: argparse.Namespace) -> int:
    from .data import materialize_filtered_base_corpus, validate_extracted_base_corpus
    from .utils import sha256_file

    output = materialize_filtered_base_corpus(args.audit_attestation, args.output)
    report = validate_extracted_base_corpus(output)
    _print_json(
        {
            "ok": True,
            "manifest": str(output),
            "sha256": sha256_file(output),
            "ready_for_training": report["ready_for_training"],
            "next": "rerun data audit-base against this filtered manifest",
        }
    )
    return 0


def _cmd_data_materialize_cooldown(args: argparse.Namespace) -> int:
    from .data import materialize_quality_cooldown_view

    result = materialize_quality_cooldown_view(
        prepared_manifest_path=args.prepared_manifest,
        kd_manifest_path=args.kd_manifest,
        selection_policy_path=args.selection_policy,
        output_root=args.output,
        required_cooldown_tokens=args.required_cooldown_tokens,
        dry_run=args.dry_run,
    )
    _print_json(result)
    return 0


def _cmd_data_generate_cooldown_policy(args: argparse.Namespace) -> int:
    from .data.quality_policy import generate_quality_cooldown_policy

    result = generate_quality_cooldown_policy(
        prepared_manifest_path=args.prepared_manifest,
        kd_manifest_path=args.kd_manifest,
        output_root=args.output,
        approve=args.approve,
        policy_id=args.policy_id,
        seed=args.selection_seed,
    )
    _print_json(result)
    return 0


def _cmd_data_resolve_sources(args: argparse.Namespace) -> int:
    from .data.sources import resolve_base_data_sources
    from .io.download import DownloadManager
    from .io.proxy import ProxySettings
    from .utils import sha256_file

    manager = DownloadManager(
        proxy_settings=ProxySettings.from_environment(proxy_url=args.proxy),
        check_proxy=True,
        network_policy=args.network_policy,
    )
    output = resolve_base_data_sources(
        args.recipe,
        args.output,
        manager=manager,
        token=args.token,
    )
    _print_json(
        {
            "ok": True,
            "resolved_lock": str(output),
            "sha256": sha256_file(output),
            "network_policy": manager.effective_network_policy,
            "proxy_fallback_used": manager.proxy_fallback_used,
        }
    )
    return 0


def _cmd_data_build_base(args: argparse.Namespace) -> int:
    from .data.sources import CorpusBuildStopped, build_base_jsonl_corpus
    from .utils import sha256_file

    try:
        output = build_base_jsonl_corpus(
            args.recipe,
            args.resolved_lock,
            args.output,
            tokenizer_path=args.tokenizer,
            tokenizer_manifest_sha256=args.tokenizer_manifest_sha256,
            profile=args.profile,
            network_policy=args.network_policy,
            proxy_url=args.proxy,
            token=args.token,
            range_block_size=args.range_block_mib * 1024 * 1024,
            stop_file=args.stop_file,
            progress=args.progress,
        )
    except CorpusBuildStopped as error:
        _print_json(
            {
                "ok": False,
                "stopped": True,
                "stop_file": str(error.stop_file),
                "completed_tokens_in_active_source": error.completed_tokens,
                "resume": "rerun the identical command after removing STOP",
            }
        )
        return 75
    _print_json(
        {
            "ok": True,
            "manifest": str(output),
            "sha256": sha256_file(output),
            "training_started": False,
        }
    )
    return 0


def _cmd_data_plan_base_refill(args: argparse.Namespace) -> int:
    from .data.refill import create_refill_plan, validate_refill_plan
    from .utils import sha256_file

    output = create_refill_plan(
        audit_attestation_path=args.audit_attestation,
        base_raw_manifest_path=args.base_raw_manifest,
        materialized_manifest_path=args.materialized_manifest,
        recipe_path=args.recipe,
        output_root=args.output,
        clean_guard_ratio=args.clean_guard_ratio,
        survival_guard_points=args.survival_guard_points,
    )
    value = validate_refill_plan(output)
    _print_json(
        {
            "ok": True,
            "plan": str(output),
            "sha256": sha256_file(output),
            "plan_fingerprint": value["plan_fingerprint"],
            "runtime_targets": value["runtime_targets"],
            "training_started": False,
            "gpu_kd_started": False,
        }
    )
    return 0


def _cmd_data_build_base_refill(args: argparse.Namespace) -> int:
    from .data.refill import build_refill_lineage, validate_refill_lineage
    from .data.sources import CorpusBuildStopped

    try:
        output = build_refill_lineage(
            plan_path=args.plan,
            resolved_lock_path=args.resolved_lock,
            output_root=args.output,
            tokenizer_path=args.tokenizer,
            tokenizer_manifest_sha256=args.tokenizer_manifest_sha256,
            network_policy=args.network_policy,
            proxy_url=args.proxy,
            token=args.token,
            range_block_size=args.range_block_mib * 1024 * 1024,
            stop_file=args.stop_file,
            progress=args.progress,
        )
    except CorpusBuildStopped as error:
        _print_json(
            {
                "ok": False,
                "stopped": True,
                "stop_file": str(error.stop_file),
                "completed_tokens_in_active_source": error.completed_tokens,
                "resume": "rerun the identical refill command after removing STOP",
                "training_started": False,
                "gpu_kd_started": False,
            }
        )
        return 75
    _print_json(validate_refill_lineage(output))
    return 0


def _cmd_data_inspect_base(args: argparse.Namespace) -> int:
    from .data.sources import validate_extracted_base_corpus

    _print_json(
        validate_extracted_base_corpus(
            args.manifest,
            verify_hashes=not args.no_verify_hashes,
        )
    )
    return 0


def _cmd_data_generate_kd(args: argparse.Namespace) -> int:
    from .kd import KDGenerationStopped, generate_teacher_kd
    from .utils import sha256_file

    try:
        output = generate_teacher_kd(
            prepared_manifest=args.prepared_manifest,
            output_root=args.output,
            teacher_path=args.teacher,
            teacher_model_id=args.teacher_model_id,
            teacher_revision=args.teacher_revision,
            teacher_manifest_sha256=args.teacher_manifest_sha256,
            tokenizer_manifest_sha256=args.tokenizer_manifest_sha256,
            temperature=args.temperature,
            batch_size=args.batch_size,
            logits_chunk_tokens=args.logits_chunk_tokens,
            device=args.device,
            stop_file=args.stop_file,
            worker_index=args.worker_index,
            num_workers=args.num_workers,
            progress=args.progress,
        )
    except KDGenerationStopped as error:
        _print_json({"ok": False, "stopped": True, **error.to_dict()})
        return 75
    _print_json({"ok": True, "manifest": str(output), "sha256": sha256_file(output)})
    return 0


def _cmd_checkpoint_inspect(args: argparse.Namespace) -> int:
    from .runtime.checkpoint import CheckpointManager

    manager = CheckpointManager(args.run_dir)
    resolved = manager.find_latest_valid_with_metadata()
    if resolved is None:
        _print_json({"ok": True, "checkpoint": None})
        return 0
    checkpoint, metadata = resolved
    _print_json({"ok": True, "checkpoint": str(checkpoint), "metadata": metadata})
    return 0


def _cmd_checkpoint_compare(args: argparse.Namespace) -> int:
    from .io.offline import enforce_offline_environment

    enforce_offline_environment()
    from .recovery import compare_checkpoint_runs

    result = compare_checkpoint_runs(
        args.config_a,
        args.checkpoint_a,
        args.config_b,
        args.checkpoint_b,
    )
    _print_json(result)
    return 0 if result["equivalent"] else 1


def _cmd_checkpoint_request(args: argparse.Namespace) -> int:
    import platform
    import signal

    session_path = Path(args.run_dir) / "rank0-session.json"
    try:
        session = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read active rank-0 session: {session_path}") from exc
    if (
        not isinstance(session, dict)
        or session.get("schema_version") != 1
        or session.get("status") != "running"
    ):
        raise RuntimeError(f"rank-0 session is not running: {session_path}")
    hostname = session.get("hostname")
    if hostname != platform.node():
        raise RuntimeError(f"rank-0 runs on host {hostname!r}; issue this request from that host")
    pid = session.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise RuntimeError(f"rank-0 session contains an invalid PID: {pid!r}")
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    if proc_cmdline.exists():
        try:
            command = proc_cmdline.read_bytes().replace(b"\0", b" ")
        except OSError as exc:
            raise RuntimeError(f"cannot verify rank-0 PID {pid}") from exc
        if b"twen.cli" not in command or b"train" not in command:
            raise RuntimeError(f"rank-0 PID {pid} no longer belongs to a Twen training process")
    requested_signal = signal.SIGUSR1 if args.action == "save" else signal.SIGTERM
    try:
        os.kill(pid, requested_signal)
    except ProcessLookupError as exc:
        raise RuntimeError(f"rank-0 PID {pid} no longer exists") from exc
    _print_json(
        {
            "ok": True,
            "action": args.action,
            "pid": pid,
            "signal": requested_signal.name,
            "session_id": session.get("session_id"),
        }
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .io.offline import enforce_offline_environment

    # Set offline flags before importing Transformers or the training engine.
    enforce_offline_environment()
    config = load_train_config(args.config)
    if args.profile is not None:
        config.runtime.profile = bool(args.profile)
    if config.runtime.expandable_segments:
        _enable_expandable_segments_allocator()
    if config.stage != args.stage:
        raise ConfigError(f"CLI stage {args.stage!r} does not match config stage {config.stage!r}")
    if (
        not args.graph_smoke
        and args.resume == "none"
        and Path(config.checkpoint.output_dir).exists()
    ):
        run_dir = Path(config.checkpoint.output_dir)
        # Shell pipelines open console.log before the torchrun worker reaches
        # this check; it is operational output, not resumable run state.
        harmless = {".run.lock", "console.log"}
        meaningful = [
            item
            for item in run_dir.iterdir()
            if item.name not in harmless
            and not item.name.startswith(".preflight-")
            and not item.name.endswith(".incomplete")
            and not (item.name.startswith(".latest.") and item.name.endswith(".tmp"))
        ]
        # A process may die after writing the immutable resolved config but
        # before its first checkpoint. Reusing that exact config is safe.
        if len(meaningful) == 1 and meaningful[0].name == "resolved_config.yaml":
            previous = load_train_config(meaningful[0])
            if previous.fingerprint() == config.fingerprint():
                meaningful = []
        if meaningful:
            raise RuntimeError(
                "--resume none refuses existing run state; choose --resume auto or a new run_id/output_dir"
            )
    from .training.engine import run_training

    return int(
        run_training(
            config,
            resume=args.resume,
            fork_from=args.fork_from,
            dry_run=args.dry_run,
            graph_smoke=args.graph_smoke,
            progress=args.progress,
        )
    )


def _cmd_web_serve(args: argparse.Namespace) -> int:
    from .web import serve_dashboard

    serve_dashboard(
        args.dashboard_config,
        host=args.host,
        port=args.port,
        auth_file=args.auth_file,
    )
    return 0


def _cmd_web_init_auth(args: argparse.Namespace) -> int:
    from .web import ensure_dashboard_auth_file

    _print_json(ensure_dashboard_auth_file(args.output, username=args.username))
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from .calibration import CalibrationStopped, run_calibration_command

    try:
        code = int(run_calibration_command(args))
    except CalibrationStopped as error:
        _print_json(
            {
                "ok": False,
                "stopped": True,
                "action": args.action,
                "output": str(Path(args.output).resolve()),
                "message": str(error),
            }
        )
        return 75
    if code == 0:
        from .utils import sha256_file

        output = Path(args.output).resolve()
        artifact = output / "manifest.json" if args.action == "collect" else output
        result: dict[str, Any] = {
            "ok": True,
            "action": args.action,
            "output": str(output),
        }
        if artifact.is_file():
            result["artifact"] = {
                "path": str(artifact),
                "size": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        if args.action == "ridge":
            sidecar = output.with_suffix(".json")
            if sidecar.is_file():
                result["sidecar"] = {
                    "path": str(sidecar),
                    "sha256": sha256_file(sidecar),
                }
        _print_json(result)
    return code


def _cmd_fold(args: argparse.Namespace) -> int:
    from .export import fold_dense_checkpoint

    result = fold_dense_checkpoint(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        device=args.device,
    )
    _print_json(result)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from .export import export_native_moe

    result = export_native_moe(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        device=args.device,
    )
    _print_json(result)
    return 0


def _cmd_evaluate_nll(args: argparse.Namespace) -> int:
    from .io.offline import enforce_offline_environment

    enforce_offline_environment()
    from .evaluation import EvaluationStopped, evaluate_nll

    try:
        result = evaluate_nll(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            prepared_manifest_path=args.prepared_manifest,
            output_dir=args.output,
            roles=args.role,
            batch_size=args.batch_size,
            device=args.device,
            stop_file=args.stop_file,
            dense_baseline_manifest_path=args.dense_baseline_manifest,
            random_baseline_manifest_path=args.random_baseline_manifest,
        )
    except EvaluationStopped as error:
        _print_json({"ok": False, "stopped": True, "message": str(error)})
        return 75
    _print_json({"ok": True, **result})
    return 0


def _cmd_evaluate_inference_consistency(args: argparse.Namespace) -> int:
    from .io.offline import enforce_offline_environment

    enforce_offline_environment()
    from .evaluation import verify_inference_consistency

    result = verify_inference_consistency(
        model_path=args.model,
        prompts=args.prompt,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        chat=args.chat,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    _print_json({"ok": bool(result["consistent"]), **result})
    return 0 if result["consistent"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="twen", description="Twen1 MoE transfer toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    config = sub.add_parser("config", help="configuration operations")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    validate = config_sub.add_parser("validate")
    validate.add_argument("--config", required=True)
    validate.set_defaults(func=_cmd_config_validate)
    config_finalize_v2 = config_sub.add_parser(
        "finalize-base-v2",
        help="authenticate completed 500M/KD/performance inputs and publish a v2 definition",
    )
    from .v2_finalizer import add_cli_arguments

    add_cli_arguments(config_finalize_v2)
    config_finalize_v2.set_defaults(func=_cmd_config_finalize_base_v2)

    proxy = sub.add_parser("proxy", help="proxy operations")
    proxy_sub = proxy.add_subparsers(dest="proxy_command", required=True)
    proxy_check = proxy_sub.add_parser("check")
    proxy_check.add_argument("--proxy", default=None)
    proxy_check.add_argument("--timeout", type=float, default=3.0)
    proxy_check.set_defaults(func=_cmd_proxy_check)

    hardware = sub.add_parser("hardware", help="read-only hardware and memory inspection")
    hardware_sub = hardware.add_subparsers(dest="hardware_command", required=True)
    hardware_inspect = hardware_sub.add_parser("inspect")
    hardware_inspect.add_argument(
        "--config",
        default=None,
        help="optional training config for a static per-device memory lower bound",
    )
    hardware_inspect.set_defaults(func=_cmd_hardware_inspect)

    download = sub.add_parser("download", help="verified resumable downloads")
    download_sub = download.add_subparsers(dest="download_command", required=True)
    download_set = download_sub.add_parser("set")
    download_set.add_argument("--spec", required=True, help="immutable artifact-set JSON")
    download_set.add_argument("--output", required=True)
    download_set.add_argument("--proxy", default=None)
    download_set.add_argument(
        "--network-policy",
        choices=("github-only", "fallback", "proxy", "direct"),
        default="fallback",
        help="GitHub uses the proxy; unreachable Hugging Face retries through it",
    )
    download_set.set_defaults(func=_cmd_download_set)
    lock_model = download_sub.add_parser("lock-model")
    lock_model.add_argument("--provider", required=True, choices=("huggingface", "modelscope"))
    lock_model.add_argument("--model-id", required=True)
    lock_model.add_argument("--revision", default="main")
    lock_model.add_argument("--output", required=True)
    lock_model.add_argument("--proxy", default=None)
    lock_model.add_argument(
        "--network-policy",
        choices=("github-only", "fallback", "proxy", "direct"),
        default="fallback",
        help="GitHub uses the proxy; unreachable Hugging Face retries through it",
    )
    lock_model.add_argument("--token", default=None)
    lock_model.set_defaults(func=_cmd_download_lock_model)

    preflight = sub.add_parser("preflight", help="offline training preflight")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--world-size", type=int, default=None)
    preflight.set_defaults(func=_cmd_preflight)

    data = sub.add_parser("data", help="data/KD shard operations")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    resolve_sources = data_sub.add_parser(
        "resolve-sources",
        help="lock native Base Parquet paths to immutable Hub LFS identities",
    )
    resolve_sources.add_argument("--recipe", default="locks/base-data-sources.json")
    resolve_sources.add_argument("--output", required=True)
    resolve_sources.add_argument("--proxy", default=None)
    resolve_sources.add_argument("--token", default=None)
    resolve_sources.add_argument(
        "--network-policy",
        choices=("github-only", "fallback", "proxy", "direct"),
        default="fallback",
        help="try Hugging Face directly, then use the configured proxy if unreachable",
    )
    resolve_sources.set_defaults(func=_cmd_data_resolve_sources)
    build_base = data_sub.add_parser(
        "build-base",
        help="range-stream pinned Parquet into resumable train/validation JSONL",
    )
    build_base.add_argument("--recipe", default="locks/base-data-sources.json")
    build_base.add_argument("--resolved-lock", required=True)
    build_base.add_argument("--output", required=True)
    build_base.add_argument("--tokenizer", required=True)
    build_base.add_argument("--tokenizer-manifest-sha256", required=True)
    build_base.add_argument("--profile", choices=("poc", "dense", "sparse"), default="dense")
    build_base.add_argument("--proxy", default=None)
    build_base.add_argument("--token", default=None)
    build_base.add_argument(
        "--network-policy",
        choices=("github-only", "fallback", "proxy", "direct"),
        default="fallback",
    )
    build_base.add_argument(
        "--range-block-mib",
        type=int,
        default=8,
        help="HTTP Range read/cache block size in MiB",
    )
    build_base.add_argument(
        "--stop-file",
        default=None,
        help="stop safely at the next committed JSONL chunk when this file exists",
    )
    build_base.add_argument(
        "--progress",
        choices=("auto", "always", "never"),
        default="auto",
    )
    build_base.set_defaults(func=_cmd_data_build_base)
    plan_base_refill = data_sub.add_parser(
        "plan-base-refill",
        help="derive a SHA-bound per-source refill plan from a complete audit ledger",
    )
    plan_base_refill.add_argument("--audit-attestation", required=True)
    plan_base_refill.add_argument("--base-raw-manifest", required=True)
    plan_base_refill.add_argument("--materialized-manifest", required=True)
    plan_base_refill.add_argument("--recipe", default="locks/base-data-sources.json")
    plan_base_refill.add_argument("--output", required=True)
    plan_base_refill.add_argument("--clean-guard-ratio", type=float, default=0.02)
    plan_base_refill.add_argument("--survival-guard-points", type=float, default=0.01)
    plan_base_refill.set_defaults(func=_cmd_data_plan_base_refill)
    build_base_refill = data_sub.add_parser(
        "build-base-refill",
        help="hard-link an immutable raw lineage and Range-stream past its source cursors",
    )
    build_base_refill.add_argument("--plan", required=True)
    build_base_refill.add_argument("--resolved-lock", required=True)
    build_base_refill.add_argument("--output", required=True)
    build_base_refill.add_argument("--tokenizer", required=True)
    build_base_refill.add_argument("--tokenizer-manifest-sha256", required=True)
    build_base_refill.add_argument("--proxy", default=None)
    build_base_refill.add_argument("--token", default=None)
    build_base_refill.add_argument(
        "--network-policy",
        choices=("github-only", "fallback", "proxy", "direct"),
        default="fallback",
    )
    build_base_refill.add_argument("--range-block-mib", type=int, default=8)
    build_base_refill.add_argument("--stop-file", default=None)
    build_base_refill.add_argument(
        "--progress",
        choices=("auto", "always", "never"),
        default="auto",
    )
    build_base_refill.set_defaults(func=_cmd_data_build_base_refill)
    inspect_base = data_sub.add_parser(
        "inspect-base", help="validate extracted Base JSONL/ledger inventory"
    )
    inspect_base.add_argument("--manifest", required=True)
    inspect_base.add_argument("--no-verify-hashes", action="store_true")
    inspect_base.set_defaults(func=_cmd_data_inspect_base)
    inspect_registry = data_sub.add_parser(
        "inspect-benchmark-registry",
        help="validate the immutable local inputs for the project 13-gram gate",
    )
    inspect_registry.add_argument("--registry", required=True)
    inspect_registry.add_argument("--benchmark-root", required=True)
    inspect_registry.add_argument("--no-verify-hashes", action="store_true")
    inspect_registry.set_defaults(func=_cmd_data_inspect_benchmark_registry)
    audit_base = data_sub.add_parser(
        "audit-base",
        help="stream exact/near-dup, contextual PII, and benchmark-overlap gates",
    )
    audit_base.add_argument("--extracted-manifest", required=True)
    audit_base.add_argument("--frozen-validation-manifest", required=True)
    audit_base.add_argument("--benchmark-registry", required=True)
    audit_base.add_argument("--benchmark-root", required=True)
    audit_base.add_argument("--output", required=True)
    audit_base.add_argument("--near-duplicate-threshold", type=float, default=0.8)
    audit_base.add_argument("--max-findings", type=int, default=10_000)
    audit_base.set_defaults(func=_cmd_data_audit_base)
    materialize_audit = data_sub.add_parser(
        "materialize-audit",
        help="materialize the complement of a complete audit rejection ledger",
    )
    materialize_audit.add_argument("--audit-attestation", required=True)
    materialize_audit.add_argument("--output", required=True)
    materialize_audit.set_defaults(func=_cmd_data_materialize_audit)
    generate_cooldown_policy = data_sub.add_parser(
        "generate-cooldown-policy",
        help="dry-plan or explicitly publish the deterministic Base-v2 50M quality policy",
    )
    generate_cooldown_policy.add_argument("--prepared-manifest", required=True)
    generate_cooldown_policy.add_argument("--kd-manifest", required=True)
    generate_cooldown_policy.add_argument("--output", required=True)
    generate_cooldown_policy.add_argument(
        "--policy-id",
        default="base-v2-50m-quality-cooldown-v1",
    )
    generate_cooldown_policy.add_argument(
        "--selection-seed",
        default="twen-base-v2-quality-cooldown-seed-v1",
        help="fixed deterministic selection seed recorded in the independent audit",
    )
    generate_cooldown_policy.add_argument(
        "--approve",
        action="store_true",
        help="publish the closed policy/audit bundle; omitted means read-only dry plan",
    )
    generate_cooldown_policy.set_defaults(func=_cmd_data_generate_cooldown_policy)
    materialize_cooldown = data_sub.add_parser(
        "materialize-cooldown",
        help="materialize an authenticated whole-shard quality-cooldown view",
    )
    materialize_cooldown.add_argument("--prepared-manifest", required=True)
    materialize_cooldown.add_argument("--kd-manifest", required=True)
    materialize_cooldown.add_argument("--selection-policy", required=True)
    materialize_cooldown.add_argument("--output", required=True)
    materialize_cooldown.add_argument(
        "--required-cooldown-tokens",
        required=True,
        type=_positive_int,
    )
    materialize_cooldown.add_argument("--dry-run", action="store_true")
    materialize_cooldown.set_defaults(func=_cmd_data_materialize_cooldown)
    index_kd = data_sub.add_parser("index-kd")
    index_kd.add_argument("--root", required=True)
    index_kd.add_argument("--output", required=True)
    index_kd.add_argument("--prepared-manifest", required=True)
    index_kd.add_argument("--temperature", type=float, default=2.0)
    index_kd.set_defaults(func=_cmd_data_index_kd)
    prepare = data_sub.add_parser(
        "prepare",
        help="tokenize explicit JSONL or an authenticated extracted-corpus role",
    )
    prepare_inputs = prepare.add_mutually_exclusive_group(required=True)
    prepare_inputs.add_argument(
        "--input",
        action="append",
        help="repeat for each explicit JSONL shard; lineage is marked explicit_unreviewed",
    )
    prepare_inputs.add_argument(
        "--extracted-manifest",
        help="authenticated corpus-manifest.json produced by data build-base",
    )
    prepare.add_argument(
        "--role",
        choices=("train", "validation"),
        help="required with --extracted-manifest; selects its exact authenticated inventory",
    )
    prepare.add_argument(
        "--allow-pending-research-audits",
        action="store_true",
        help=(
            "explicitly allow an extracted corpus with ready_for_training=false; "
            "the prepared artifact remains research_only and records every pending audit"
        ),
    )
    prepare.add_argument(
        "--audit-attestation",
        help=(
            "authenticated attestation from data audit-base; its candidate/frozen role "
            "must bind the selected extracted manifest"
        ),
    )
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--tokenizer", required=True)
    prepare.add_argument("--tokenizer-manifest-sha256", required=True)
    prepare.add_argument("--sequence-length", type=int, default=4096)
    prepare.add_argument("--text-field", default="text")
    prepare.add_argument(
        "--progress",
        choices=("auto", "always", "never"),
        default="auto",
        help="stderr shard progress; auto only renders on a TTY",
    )
    prepare.set_defaults(func=_cmd_data_prepare)
    generate_kd = data_sub.add_parser("generate-kd")
    generate_kd.add_argument("--prepared-manifest", required=True)
    generate_kd.add_argument("--output", required=True)
    generate_kd.add_argument("--teacher", required=True)
    generate_kd.add_argument("--teacher-model-id", required=True)
    generate_kd.add_argument("--teacher-revision", required=True)
    generate_kd.add_argument("--teacher-manifest-sha256", required=True)
    generate_kd.add_argument("--tokenizer-manifest-sha256", required=True)
    generate_kd.add_argument("--temperature", type=float, default=2.0)
    generate_kd.add_argument("--batch-size", type=int, default=1)
    generate_kd.add_argument(
        "--logits-chunk-tokens",
        type=int,
        default=64,
        help="full-vocabulary head chunk size (default 64, tuned on this RTX 5090)",
    )
    generate_kd.add_argument("--device", default="cuda")
    generate_kd.add_argument("--stop-file", default=None)
    generate_kd.add_argument("--worker-index", type=int, default=None)
    generate_kd.add_argument("--num-workers", type=int, default=None)
    generate_kd.add_argument(
        "--progress",
        choices=("auto", "always", "never"),
        default="auto",
        help="stderr token progress; auto only renders on a TTY",
    )
    generate_kd.set_defaults(func=_cmd_data_generate_kd)

    checkpoint = sub.add_parser("checkpoint", help="checkpoint operations")
    checkpoint_sub = checkpoint.add_subparsers(dest="checkpoint_command", required=True)
    inspect_parser = checkpoint_sub.add_parser("inspect")
    inspect_parser.add_argument("--run-dir", required=True)
    inspect_parser.set_defaults(func=_cmd_checkpoint_inspect)
    compare_parser = checkpoint_sub.add_parser(
        "compare", help="read-only exact recovery comparison using production state templates"
    )
    compare_parser.add_argument("--config-a", required=True)
    compare_parser.add_argument("--checkpoint-a", default="auto")
    compare_parser.add_argument("--config-b", required=True)
    compare_parser.add_argument("--checkpoint-b", default="auto")
    compare_parser.set_defaults(func=_cmd_checkpoint_compare)
    request_parser = checkpoint_sub.add_parser(
        "request",
        help="signal the exact active rank-0 process using rank0-session.json",
    )
    request_parser.add_argument("--run-dir", required=True)
    request_parser.add_argument(
        "--action",
        choices=("save", "stop"),
        default="save",
        help="save checkpoints and continue; stop checkpoints and exit",
    )
    request_parser.set_defaults(func=_cmd_checkpoint_request)

    train = sub.add_parser("train", help="run a user-controlled training stage")
    train.add_argument("--stage", required=True, choices=("dense-oracle", "sparse"))
    train.add_argument("--config", required=True)
    train.add_argument(
        "--resume",
        default="auto",
        help="auto (default), none, or an explicit checkpoint directory",
    )
    train.add_argument("--fork-from", default=None)
    train.add_argument(
        "--progress",
        choices=("auto", "always", "never"),
        default="auto",
        help="rank-zero token progress bar; auto only renders on a TTY",
    )
    train.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override runtime.profile for a bounded PyTorch trace",
    )
    validation_mode = train.add_mutually_exclusive_group()
    validation_mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate assets/configuration only; never constructs an optimizer",
    )
    validation_mode.add_argument(
        "--graph-smoke",
        action="store_true",
        help="run one distributed forward/backward microbatch without optimizer or checkpoint state",
    )
    train.set_defaults(func=_cmd_train)

    web = sub.add_parser("web", help="local training dashboard operations")
    web_sub = web.add_subparsers(dest="web_command", required=True)
    web_serve = web_sub.add_parser(
        "serve",
        help="serve the read-only metrics UI and guarded training controls",
    )
    web_serve.add_argument("--dashboard-config", required=True)
    web_serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="literal bind IP; non-loopback binds require --auth-file",
    )
    web_serve.add_argument("--port", type=int, default=8765)
    web_serve.add_argument(
        "--auth-file",
        help="mode-0600 JSON credential created by `twen web init-auth`",
    )
    web_serve.set_defaults(func=_cmd_web_serve)
    web_auth = web_sub.add_parser(
        "init-auth",
        help="create a private HTTP Basic credential for an authenticated LAN bind",
    )
    web_auth.add_argument("--output", required=True)
    web_auth.add_argument("--username", default="twen")
    web_auth.set_defaults(func=_cmd_web_init_auth)

    calibrate = sub.add_parser("calibrate", help="user-run activation calibration")
    calibrate.add_argument("action", choices=("collect", "layer-map", "ridge", "partition"))
    calibrate.add_argument("--config", required=True)
    calibrate.add_argument("--student-activations")
    calibrate.add_argument("--donor-activations")
    calibration_inputs = calibrate.add_mutually_exclusive_group()
    calibration_inputs.add_argument(
        "--input",
        action="append",
        help="repeatable collect input safetensors with input_ids/attention_mask",
    )
    calibration_inputs.add_argument(
        "--prepared-manifest",
        default=None,
        help="collect every tokens.safetensors in this authenticated prepared manifest",
    )
    calibrate.add_argument("--device", default="cuda")
    calibrate.add_argument("--max-samples", type=int, default=8192)
    calibrate.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="number of input sequences per inference microbatch during collect",
    )
    calibrate.add_argument(
        "--sample-seed",
        type=int,
        default=3407,
        help="deterministic global token sampler seed (also used by CKA subsampling)",
    )
    calibrate.add_argument(
        "--ridge-dtype",
        choices=("auto", "float32", "float64"),
        default="auto",
        help="ridge statistics dtype; auto uses float32 on CUDA and float64 on CPU",
    )
    calibrate.add_argument(
        "--ridge-batch-samples",
        type=int,
        default=1024,
        help="paired activation rows per ridge statistics update",
    )
    calibrate.add_argument(
        "--stop-file",
        default=None,
        help="consume this file and stop at the next transaction-safe boundary",
    )
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument(
        "--progress",
        choices=("auto", "always", "never"),
        default="auto",
        help="stderr calibration progress; auto only renders on a TTY",
    )
    calibrate.set_defaults(func=_cmd_calibrate)

    fold = sub.add_parser("fold", help="fold A/B into routed expert weights")
    fold.add_argument("--config", required=True)
    fold.add_argument("--checkpoint", required=True)
    fold.add_argument("--output", required=True)
    fold.add_argument("--device", default="cuda")
    fold.set_defaults(func=_cmd_fold)

    export = sub.add_parser("export", help="export native Qwen3.5-MoE checkpoint")
    export.add_argument("--config", required=True)
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--device", default="cpu")
    export.set_defaults(func=_cmd_export)

    evaluate = sub.add_parser("evaluate", help="user-run, resumable evaluation")
    evaluate_sub = evaluate.add_subparsers(dest="evaluate_command", required=True)
    evaluate_nll = evaluate_sub.add_parser("nll")
    evaluate_nll.add_argument("--config", required=True)
    evaluate_nll.add_argument("--checkpoint", default="auto")
    evaluate_nll.add_argument("--prepared-manifest", required=True)
    evaluate_nll.add_argument("--output", required=True)
    evaluate_nll.add_argument(
        "--role",
        action="append",
        choices=("candidate", "shared", "dense-oracle", "teacher"),
        help="repeat to override the stage-default role set",
    )
    evaluate_nll.add_argument("--batch-size", type=int, default=1)
    evaluate_nll.add_argument("--device", default="cuda")
    evaluate_nll.add_argument("--stop-file", default=None)
    evaluate_nll.add_argument(
        "--dense-baseline-manifest",
        default=None,
        help="completed Stage-B donor evaluation manifest; required for sparse acceptance",
    )
    evaluate_nll.add_argument(
        "--random-baseline-manifest",
        default=None,
        help="completed random-control dense evaluation used by the donor-vs-random gate",
    )
    evaluate_nll.set_defaults(func=_cmd_evaluate_nll)
    inference_consistency = evaluate_sub.add_parser(
        "inference-consistency",
        help="compare fixed-length greedy token IDs from Transformers and vLLM",
    )
    inference_consistency.add_argument("--model", required=True)
    inference_consistency.add_argument("--prompt", action="append", required=True)
    inference_consistency.add_argument("--output", required=True)
    inference_consistency.add_argument("--max-new-tokens", type=int, default=32)
    inference_consistency.add_argument("--chat", action="store_true")
    inference_consistency.add_argument("--tensor-parallel-size", type=int, default=1)
    inference_consistency.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
    )
    inference_consistency.set_defaults(func=_cmd_evaluate_inference_consistency)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt as error:
        exit_code = int(getattr(error, "exit_code", 130))
        print(f"interrupted: {error}", file=sys.stderr)
        return exit_code
    except (ConfigError, ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
