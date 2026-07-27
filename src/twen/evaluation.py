"""User-run, shard-resumable NLL evaluation for stage acceptance gates."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import yaml

from .config import TrainConfig, load_train_config
from .data import (
    ShardTransaction,
    read_complete_marker,
    validate_prepared_corpus_for_inference,
)
from .io.locking import FileLock
from .io.offline import enforce_offline_environment
from .preflight import PreflightReport, run_inference_preflight
from .utils import atomic_write_json, atomic_write_text, sha256_file

EVALUATION_SCHEMA_VERSION = 1
EVALUATION_ROLES = frozenset(("candidate", "shared", "dense-oracle", "teacher"))


class EvaluationStopped(RuntimeError):
    """Evaluation stopped at a replay-safe microbatch boundary."""


def _normalize_evaluation_device(device: str) -> str:
    """Give CUDA's implicit device an explicit index for runtime metadata APIs."""

    if device == "cuda":
        return "cuda:0"
    return device


def _validate_inference_checkpoint_lineage(
    *,
    config_path: str | Path,
    config: TrainConfig,
    report: PreflightReport,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate an old training checkpoint for forward-only evaluation.

    Exact-resume fingerprints intentionally include the source tree.  That is
    correct for continuing optimization but would make every completed run
    unevaluable after even a reporting-only code change.  This narrower gate
    is used only by NLL inference: it checks immutable source/calibration/data
    lineage, archived loss state, run geometry, and lets DCP enforce exact
    trainable key/shape compatibility.  It never authorizes resume/export.
    """

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("evaluation config is not a mapping")
    extra = metadata.get("extra")
    trainer = metadata.get("trainer_state")
    if not isinstance(extra, Mapping) or not isinstance(trainer, Mapping):
        raise ValueError("checkpoint lacks authenticated training lineage")
    expected_sources = {
        name: getattr(config.sources, name).manifest_sha256
        for name in ("backbone", "donor", "teacher", "tokenizer")
    }
    expected_sources["folded_experts"] = config.sources.folded_experts_sha256
    if extra.get("source_manifests") != expected_sources:
        raise ValueError("checkpoint source manifests differ from the evaluation config")
    expected_calibration = dict(report.calibration_fingerprints)
    if extra.get("calibration_artifacts") != expected_calibration:
        raise ValueError("checkpoint calibration artifacts differ from current verified files")
    if extra.get("data_manifest_sha256") != config.data.manifest_sha256:
        raise ValueError("checkpoint training-data manifest differs from the evaluation config")
    if extra.get("teacher_kd_manifest_sha256") != config.data.teacher_kd_manifest_sha256:
        raise ValueError("checkpoint teacher-KD manifest differs from the evaluation config")
    archived_losses = raw.get("losses")
    if not isinstance(archived_losses, Mapping) or trainer.get("loss_weights") != dict(
        archived_losses
    ):
        raise ValueError("checkpoint loss weights differ from the archived resolved config")
    expected_top_k = None if config.stage == "dense-oracle" else config.architecture.top_k
    if trainer.get("top_k") != expected_top_k:
        raise ValueError("checkpoint top_k differs from the archived resolved config")
    trainer_identity = (
        trainer.get("run_id"),
        trainer.get("stage"),
        int(trainer.get("global_batch_tokens", -1)),
    )
    expected_identity = (
        config.run_id,
        config.stage,
        config.data.global_batch_tokens,
    )
    if trainer_identity != expected_identity:
        raise ValueError("checkpoint trainer identity differs from the evaluation config")
    saved_source_tree = extra.get("source_tree_sha256")
    current_source_tree = report.source_tree_sha256
    return {
        "mode": "forward_only_lineage_compatible_not_exact_resume",
        "saved_critical_fingerprint": metadata.get("critical_fingerprint"),
        "current_preflight_fingerprint": report.config_fingerprint,
        "exact_training_fingerprint_match": (
            metadata.get("critical_fingerprint") == report.config_fingerprint
        ),
        "saved_source_tree_sha256": saved_source_tree,
        "current_source_tree_sha256": current_source_tree,
        "source_tree_match": saved_source_tree == current_source_tree,
        "archived_config_path": str(Path(config_path).resolve()),
        "archived_config_sha256": sha256_file(config_path),
        "source_manifests": expected_sources,
        "calibration_artifacts": expected_calibration,
        "loss_weights": dict(archived_losses),
        "top_k": expected_top_k,
    }


def _load_inference_evaluation_checkpoint(
    model: Any,
    *,
    config_path: str,
    config: TrainConfig,
    report: PreflightReport,
    checkpoint_path: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Load trainable deltas under the inference-only compatibility gate."""

    from .training.stateful import TrainableModelState

    manager, resolved, _metadata, lineage = _inspect_inference_evaluation_checkpoint(
        config_path=config_path,
        config=config,
        report=report,
        checkpoint_path=checkpoint_path,
    )
    loaded = manager.load(
        {"model": TrainableModelState(model)},
        resolved,
        # Deliberately inference-only: source-tree drift is recorded above,
        # while every immutable model/data/calibration identity remains strict.
        expected_critical_fingerprint=None,
        expected_data_fingerprint=report.data_fingerprint,
        expected_run_id=config.run_id,
        expected_stage=config.stage,
        expected_global_batch_tokens=config.data.global_batch_tokens,
    )
    return loaded.path, dict(loaded.metadata), lineage


def _inspect_inference_evaluation_checkpoint(
    *,
    config_path: str,
    config: TrainConfig,
    report: PreflightReport,
    checkpoint_path: str,
) -> tuple[Any, Path, dict[str, Any], dict[str, Any]]:
    """Authenticate checkpoint metadata without loading tensors into a model."""

    from .runtime.checkpoint import CheckpointManager

    manager = CheckpointManager(config.checkpoint.output_dir, rank=0, world_size=1)
    resolved = manager.resolve(checkpoint_path)
    metadata = dict(manager.inspect(resolved))
    lineage = _validate_inference_checkpoint_lineage(
        config_path=config_path,
        config=config,
        report=report,
        metadata=metadata,
    )
    return manager, resolved, metadata, lineage


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dense_control_fingerprint(config: TrainConfig) -> str:
    """Identity shared by donor and random-control Stage-B configurations."""

    payload = config.critical_dict()
    payload.pop("run_id", None)
    architecture = payload["architecture"]
    architecture.pop("expert_initialization", None)
    architecture.pop("random_expert_seed", None)
    return _canonical_sha256(payload)


def _check_stop_file(stop_file: str | Path | None) -> None:
    if stop_file is None:
        return
    path = Path(stop_file)
    if not path.is_file():
        return
    path.unlink(missing_ok=True)
    raise EvaluationStopped(
        f"evaluation stopped safely after consuming {path}; rerun the same command"
    )


def _install_or_validate_plan(root: Path, plan: Mapping[str, Any]) -> Path:
    path = root / "PLAN.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != plan:
            raise ValueError(f"evaluation output {root} belongs to a different immutable plan")
        return path
    unexpected = [item for item in root.iterdir() if item.name not in {".eval.lock", "STOP"}]
    if unexpected:
        raise ValueError(f"evaluation output has state but no PLAN: {unexpected[0]}")
    atomic_write_json(path, dict(plan))
    return path


def _nll_sum(logits: Any, labels: Any) -> tuple[Any, int]:
    import torch.nn.functional as F

    shifted_logits = logits[..., :-1, :].contiguous().float()
    shifted_labels = labels[..., 1:].contiguous()
    token_count = int(shifted_labels.ne(-100).sum().item())
    if token_count <= 0:
        return shifted_logits.new_zeros(()), 0
    loss = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss, token_count


def _configure_candidate_mode(
    modules: Sequence[Any],
    *,
    role: str,
    top_k: int,
) -> None:
    if role not in {"candidate", "shared", "dense-oracle"}:
        raise ValueError(f"invalid candidate evaluation role: {role}")
    for module in modules:
        module.set_record_aux(False)
        module.clear_aux()
        module.set_transfer_enabled(role != "shared")
        if hasattr(module, "set_dense_oracle_enabled"):
            module.set_dense_oracle_enabled(role == "dense-oracle")
        elif role == "dense-oracle":
            # Dense-stage modules already sum every routed slice exactly.
            pass
        if hasattr(module, "set_top_k"):
            module.set_top_k(top_k)


def _read_progress(path: Path, *, fingerprint: str, sequence_count: int) -> dict[str, Any]:
    if not path.is_file():
        return {"next_sequence": 0, "nll_sum": 0.0, "predicted_tokens": 0}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("fingerprint") != fingerprint
            or value.get("schema_version") != EVALUATION_SCHEMA_VERSION
        ):
            raise ValueError("progress identity mismatch")
        next_sequence = int(value["next_sequence"])
        nll_sum = float(value["nll_sum"])
        predicted_tokens = int(value["predicted_tokens"])
        if (
            not 0 <= next_sequence <= sequence_count
            or not math.isfinite(nll_sum)
            or nll_sum < 0
            or predicted_tokens < 0
        ):
            raise ValueError("progress counters are invalid")
    except (KeyError, OSError, TypeError, ValueError):
        path.unlink(missing_ok=True)
        return {"next_sequence": 0, "nll_sum": 0.0, "predicted_tokens": 0}
    return {
        "next_sequence": next_sequence,
        "nll_sum": nll_sum,
        "predicted_tokens": predicted_tokens,
    }


def _evaluate_role(
    model: Any,
    prepared: Any,
    prepared_manifest: Path,
    output_root: Path,
    *,
    role: str,
    role_fingerprint: str,
    batch_size: int,
    device: str,
    dtype: Any,
    use_bf16: bool,
    stop_file: str | Path | None,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    role_root = output_root / "roles" / role
    totals = {"nll_sum": 0.0, "predicted_tokens": 0, "sequences": 0}
    shard_results = []
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    model.eval()
    for entry in prepared.shards:
        _check_stop_file(stop_file)
        shard_fingerprint = _canonical_sha256(
            {
                "role_fingerprint": role_fingerprint,
                "source_shard_id": entry.shard_id,
                "source_tensors_sha256": entry.tensors_sha256,
                "sequence_count": entry.sequence_count,
                "token_count": entry.token_count,
            }
        )
        with ShardTransaction(
            role_root,
            entry.shard_id,
            fingerprint=shard_fingerprint,
            source_fingerprint=entry.tensors_sha256,
        ) as transaction:
            result_path = transaction.work_directory / "result.json"
            if transaction.complete:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if (
                    result.get("fingerprint") != shard_fingerprint
                    or result.get("role") != role
                    or int(result.get("sequences", -1)) != entry.sequence_count
                    or int(result.get("predicted_tokens", -1)) <= 0
                    or not math.isfinite(float(result.get("nll_sum", math.nan)))
                    or float(result.get("nll_sum", -1.0)) < 0
                ):
                    raise ValueError(f"evaluation result lineage mismatch: {result_path}")
            else:
                tensors = load_file(
                    str(prepared_manifest.parent / entry.path / "tokens.safetensors"),
                    device="cpu",
                )
                sequence_count = int(tensors["input_ids"].shape[0])
                if sequence_count != entry.sequence_count:
                    raise ValueError(f"prepared sequence count changed for {entry.shard_id}")
                progress_path = transaction.work_directory / "progress.json"
                progress = _read_progress(
                    progress_path,
                    fingerprint=shard_fingerprint,
                    sequence_count=sequence_count,
                )
                for start in range(progress["next_sequence"], sequence_count, batch_size):
                    end = min(start + batch_size, sequence_count)
                    input_ids = tensors["input_ids"][start:end].to(device)
                    attention_mask = tensors["attention_mask"][start:end].to(device)
                    labels = tensors["labels"][start:end].to(device)
                    with (
                        torch.inference_mode(),
                        torch.autocast(
                            device_type=device_type,
                            dtype=dtype,
                            enabled=use_bf16,
                        ),
                    ):
                        logits = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                        ).logits
                        batch_nll, batch_tokens = _nll_sum(logits, labels)
                    progress["next_sequence"] = end
                    progress["nll_sum"] += float(batch_nll)
                    progress["predicted_tokens"] += batch_tokens
                    atomic_write_json(
                        progress_path,
                        {
                            "schema_version": EVALUATION_SCHEMA_VERSION,
                            "kind": "nll_progress",
                            "fingerprint": shard_fingerprint,
                            **progress,
                        },
                    )
                    _check_stop_file(stop_file)
                result = {
                    "schema_version": EVALUATION_SCHEMA_VERSION,
                    "kind": "nll_shard_result",
                    "fingerprint": shard_fingerprint,
                    "role": role,
                    "source_shard_id": entry.shard_id,
                    "sequences": sequence_count,
                    "nll_sum": progress["nll_sum"],
                    "predicted_tokens": progress["predicted_tokens"],
                    "mean_nll": progress["nll_sum"] / max(progress["predicted_tokens"], 1),
                }
                atomic_write_json(result_path, result)
                transaction.commit(
                    {
                        "kind": "nll_shard_result",
                        "role": role,
                        "predicted_tokens": progress["predicted_tokens"],
                    }
                )
            totals["nll_sum"] += float(result["nll_sum"])
            totals["predicted_tokens"] += int(result["predicted_tokens"])
            totals["sequences"] += int(result["sequences"])
            marker = read_complete_marker(transaction.final_directory)
            shard_results.append(
                {
                    "source_shard_id": entry.shard_id,
                    "path": transaction.final_directory.relative_to(output_root).as_posix(),
                    "complete_sha256": sha256_file(transaction.final_directory / "COMPLETE"),
                    "outputs": marker["outputs"],
                }
            )
        print(
            json.dumps(
                {
                    "event": "eval_shard",
                    "role": role,
                    "shard": entry.shard_id,
                    "predicted_tokens": totals["predicted_tokens"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _check_stop_file(stop_file)
    if totals["predicted_tokens"] <= 0:
        raise ValueError(f"evaluation role {role} produced no predicted tokens")
    mean_nll = totals["nll_sum"] / totals["predicted_tokens"]
    return {
        "role": role,
        **totals,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll) if mean_nll < 700 else None,
        "shards": shard_results,
    }


def _load_evaluation_baseline(
    path: str | Path,
    *,
    expected_track: str,
    expected_stage: str,
    expected_expert_initialization: str,
    prepared_manifest_sha256: str,
    prepared_dataset_fingerprint: str,
    expected_batch_size: int,
    expected_device_type: str,
    expected_dtype: str,
    expected_control_fingerprint: str | None = None,
    expected_checkpoint: str | None = None,
    expected_checkpoint_complete_sha256: str | None = None,
    expected_config_fingerprint: str | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Authenticate a completed evaluation used by a cross-stage gate."""

    manifest_path = Path(path).resolve()
    complete_path = manifest_path.parent / "COMPLETE"
    plan_path = manifest_path.parent / "PLAN.json"
    if not manifest_path.is_file() or not complete_path.is_file() or not plan_path.is_file():
        raise ValueError(f"evaluation baseline is incomplete: {manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    if complete_path.read_text(encoding="ascii").strip() != manifest_sha:
        raise ValueError(f"evaluation baseline COMPLETE is invalid: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != EVALUATION_SCHEMA_VERSION
        or payload.get("kind") != "twen_nll_evaluation"
        or payload.get("plan_sha256") != sha256_file(plan_path)
        or payload.get("plan_fingerprint") != plan.get("plan_fingerprint")
    ):
        raise ValueError(f"evaluation baseline manifest/PLAN lineage is invalid: {manifest_path}")
    if _canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    ) != plan.get("plan_fingerprint"):
        raise ValueError(f"evaluation baseline PLAN fingerprint is invalid: {manifest_path}")
    expected = (
        expected_track,
        expected_stage,
        expected_expert_initialization,
        prepared_manifest_sha256,
        prepared_dataset_fingerprint,
        expected_batch_size,
        expected_device_type,
        expected_dtype,
    )
    actual = (
        payload.get("track"),
        payload.get("stage"),
        payload.get("expert_initialization"),
        plan.get("prepared_manifest_sha256"),
        plan.get("prepared_dataset_fingerprint"),
        plan.get("batch_size"),
        plan.get("device_type"),
        plan.get("dtype"),
    )
    if actual != expected:
        raise ValueError(
            "evaluation baseline differs in track/stage/expert initialization/corpus/numerics"
        )
    checkpoint_state = payload.get("checkpoint_state")
    if not isinstance(checkpoint_state, dict) or checkpoint_state != plan.get("checkpoint_state"):
        raise ValueError("evaluation baseline has no authenticated checkpoint progress")
    for field in ("global_step", "committed_tokens"):
        if (
            isinstance(checkpoint_state.get(field), bool)
            or int(checkpoint_state.get(field, -1)) < 0
        ):
            raise ValueError("evaluation baseline checkpoint progress is invalid")
    if checkpoint_state.get("kind") not in {"periodic", "interrupt", "milestone"}:
        raise ValueError("evaluation baseline checkpoint kind is invalid")
    if (
        expected_control_fingerprint is not None
        and payload.get("dense_control_fingerprint") != expected_control_fingerprint
    ):
        raise ValueError(
            "random-control baseline changed a dense training setting other than its control initialization"
        )
    if (
        expected_checkpoint is not None
        and Path(str(plan.get("checkpoint"))).resolve() != Path(expected_checkpoint).resolve()
    ):
        raise ValueError("Stage-B baseline did not evaluate the checkpoint used for folding")
    if (
        expected_checkpoint_complete_sha256 is not None
        and plan.get("checkpoint_complete_sha256") != expected_checkpoint_complete_sha256
    ):
        raise ValueError("Stage-B baseline checkpoint COMPLETE hash differs from the fold lineage")
    if (
        expected_config_fingerprint is not None
        and plan.get("config_fingerprint") != expected_config_fingerprint
    ):
        raise ValueError("Stage-B baseline config fingerprint differs from the fold lineage")
    baseline_roles = payload.get("roles")
    if not isinstance(baseline_roles, dict) or not {"candidate", "shared"} <= set(baseline_roles):
        raise ValueError("evaluation baseline must contain candidate and shared roles")
    token_counts = {
        int(baseline_roles[role].get("predicted_tokens", -1)) for role in ("candidate", "shared")
    }
    if len(token_counts) != 1 or next(iter(token_counts)) <= 0:
        raise ValueError("evaluation baseline roles processed inconsistent token counts")
    identity = {
        "path": str(manifest_path),
        "sha256": manifest_sha,
        "complete_sha256": sha256_file(complete_path),
        "plan_sha256": sha256_file(plan_path),
    }
    return payload, identity


def _acceptance_metrics(
    stage: str,
    roles: Mapping[str, Mapping[str, Any]],
    *,
    dense_baseline: Mapping[str, Any] | None = None,
    random_baseline: Mapping[str, Any] | None = None,
    expert_initialization: str = "donor",
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if {"shared", "candidate", "teacher"} <= set(roles):
        shared = float(roles["shared"]["mean_nll"])
        candidate = float(roles["candidate"]["mean_nll"])
        teacher = float(roles["teacher"]["mean_nll"])
        denominator = shared - teacher
        fraction = (shared - candidate) / denominator if denominator > 0 else None
        result["teacher_gap_closed_fraction"] = fraction
        if stage == "dense-oracle" and expert_initialization == "donor":
            result["dense_gap_gate_pass"] = bool(
                fraction is not None and math.isfinite(fraction) and fraction >= 0.10
            )
    if stage == "sparse" and dense_baseline is not None and {"shared", "candidate"} <= set(roles):
        baseline_roles = dense_baseline["roles"]
        shared = float(baseline_roles["shared"]["mean_nll"])
        candidate = float(roles["candidate"]["mean_nll"])
        dense = float(baseline_roles["candidate"]["mean_nll"])
        denominator = shared - dense
        fraction = (shared - candidate) / denominator if denominator > 0 else None
        result["dense_oracle_gain_retained_fraction"] = fraction
        result["sparse_retention_gate_pass"] = bool(
            fraction is not None and math.isfinite(fraction) and fraction >= 0.90
        )
        result["stage_b_dense_candidate_nll"] = dense
        result["stage_b_shared_nll"] = shared
    if stage == "dense-oracle" and random_baseline is not None and "candidate" in roles:
        random_nll = float(random_baseline["roles"]["candidate"]["mean_nll"])
        donor_nll = float(roles["candidate"]["mean_nll"])
        improvement = random_nll - donor_nll
        result["random_control_candidate_nll"] = random_nll
        result["donor_over_random_nll_improvement"] = improvement
        result["donor_beats_random_control"] = bool(
            math.isfinite(improvement) and improvement > 0.0
        )
    return result


def evaluate_nll(
    *,
    config_path: str,
    checkpoint_path: str,
    prepared_manifest_path: str,
    output_dir: str,
    roles: Sequence[str] | None = None,
    batch_size: int = 1,
    device: str = "cuda",
    stop_file: str | None = None,
    dense_baseline_manifest_path: str | None = None,
    random_baseline_manifest_path: str | None = None,
) -> dict[str, Any]:
    """Own the evaluation lock across the complete CPU/GPU lifecycle."""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / ".eval.lock", timeout_seconds=300.0):
        return _evaluate_nll_while_locked(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            prepared_manifest_path=prepared_manifest_path,
            output_dir=output,
            roles=roles,
            batch_size=batch_size,
            device=device,
            stop_file=stop_file,
            dense_baseline_manifest_path=dense_baseline_manifest_path,
            random_baseline_manifest_path=random_baseline_manifest_path,
        )


def _evaluate_nll_while_locked(
    *,
    config_path: str,
    checkpoint_path: str,
    prepared_manifest_path: str,
    output_dir: Path,
    roles: Sequence[str] | None = None,
    batch_size: int = 1,
    device: str = "cuda",
    stop_file: str | None = None,
    dense_baseline_manifest_path: str | None = None,
    random_baseline_manifest_path: str | None = None,
) -> dict[str, Any]:
    """Evaluate stage/shared/teacher NLL without backward or optimizer state."""

    enforce_offline_environment()
    device = _normalize_evaluation_device(device)
    if batch_size <= 0:
        raise ValueError("evaluation batch_size must be positive")
    config: TrainConfig = load_train_config(config_path)
    report: PreflightReport = run_inference_preflight(config, world_size=1)
    prepared_path = Path(prepared_manifest_path).resolve()
    prepared_manifest_sha = sha256_file(prepared_path)
    prepared = validate_prepared_corpus_for_inference(
        prepared_path,
        expected_manifest_sha256=prepared_manifest_sha,
    )
    if prepared.tokenizer_sha256 != config.sources.tokenizer.manifest_sha256:
        raise ValueError("evaluation corpus tokenizer differs from the run tokenizer")
    evaluation_device_type = "cuda" if device.startswith("cuda") else "cpu"
    evaluation_dtype = "bfloat16" if config.runtime.bf16 else "float32"
    dense_baseline: dict[str, Any] | None = None
    dense_baseline_identity: dict[str, str] | None = None
    random_baseline: dict[str, Any] | None = None
    random_baseline_identity: dict[str, str] | None = None
    dense_control_fingerprint = (
        _dense_control_fingerprint(config) if config.stage == "dense-oracle" else None
    )
    if config.stage == "sparse":
        if not dense_baseline_manifest_path:
            raise ValueError(
                "sparse acceptance requires --dense-baseline-manifest from the completed Stage-B evaluation"
            )
        folded_manifest_path = (
            Path(config.sources.folded_experts_path or "").resolve().parent / "manifest.json"
        )
        folded_lineage = json.loads(folded_manifest_path.read_text(encoding="utf-8"))
        dense_baseline, dense_baseline_identity = _load_evaluation_baseline(
            dense_baseline_manifest_path,
            expected_track=config.track,
            expected_stage="dense-oracle",
            expected_expert_initialization="donor",
            prepared_manifest_sha256=prepared_manifest_sha,
            prepared_dataset_fingerprint=prepared.dataset_fingerprint,
            expected_batch_size=batch_size,
            expected_device_type=evaluation_device_type,
            expected_dtype=evaluation_dtype,
            expected_checkpoint=str(folded_lineage["source_checkpoint"]),
            expected_checkpoint_complete_sha256=str(
                folded_lineage["source_checkpoint_complete_sha256"]
            ),
            expected_config_fingerprint=str(folded_lineage["source_config_fingerprint"]),
        )
    elif dense_baseline_manifest_path:
        raise ValueError("--dense-baseline-manifest is only valid for sparse evaluation")
    if random_baseline_manifest_path:
        if config.stage != "dense-oracle" or config.architecture.expert_initialization != "donor":
            raise ValueError(
                "--random-baseline-manifest is only valid for a donor-initialized dense evaluation"
            )
        random_baseline, random_baseline_identity = _load_evaluation_baseline(
            random_baseline_manifest_path,
            expected_track=config.track,
            expected_stage="dense-oracle",
            expected_expert_initialization="random-control",
            prepared_manifest_sha256=prepared_manifest_sha,
            prepared_dataset_fingerprint=prepared.dataset_fingerprint,
            expected_batch_size=batch_size,
            expected_device_type=evaluation_device_type,
            expected_dtype=evaluation_dtype,
            expected_control_fingerprint=dense_control_fingerprint,
        )
    selected_roles = tuple(
        roles
        or (
            ("candidate", "shared", "teacher")
            if config.stage == "dense-oracle"
            else ("candidate", "shared", "dense-oracle", "teacher")
        )
    )
    if not selected_roles or len(set(selected_roles)) != len(selected_roles):
        raise ValueError("evaluation roles must be non-empty and unique")
    unknown = set(selected_roles) - EVALUATION_ROLES
    if unknown:
        raise ValueError(f"unknown evaluation roles: {sorted(unknown)}")
    if (dense_baseline is not None or random_baseline is not None) and not {
        "candidate",
        "shared",
    } <= set(selected_roles):
        raise ValueError("cross-run acceptance gates require candidate and shared roles")
    output = output_dir
    output.mkdir(parents=True, exist_ok=True)
    active_stop_file = Path(stop_file) if stop_file else output / "STOP"
    _check_stop_file(active_stop_file)

    # Authenticate checkpoint metadata and publish the immutable plan before
    # allocating the model or loading any checkpoint tensors onto the GPU.
    import torch

    dtype = torch.bfloat16 if config.runtime.bf16 else torch.float32
    checkpoint_manager, resolved_checkpoint, checkpoint_metadata, checkpoint_lineage = (
        _inspect_inference_evaluation_checkpoint(
            config_path=config_path,
            config=config,
            report=report,
            checkpoint_path=checkpoint_path,
        )
    )
    checkpoint_complete_sha = sha256_file(resolved_checkpoint / "COMPLETE")
    checkpoint_state = {
        "global_step": int(checkpoint_metadata["global_step"]),
        "committed_tokens": int(checkpoint_metadata["committed_tokens"]),
        "kind": str(checkpoint_metadata["kind"]),
        "tag": checkpoint_metadata.get("tag"),
    }
    if random_baseline is not None and random_baseline.get("checkpoint_state") != checkpoint_state:
        raise ValueError(
            "donor and random-control evaluations must use checkpoints at identical training progress"
        )
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("evaluation requested CUDA but CUDA is unavailable")
        torch.cuda.set_device(torch.device(device))
    plan = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "kind": "twen_nll_evaluation_plan",
        "config_path": str(Path(config_path).resolve()),
        "config_fingerprint": report.config_fingerprint,
        "preflight_mode": "forward_only_historical_prepared_compatible",
        "checkpoint": str(resolved_checkpoint.resolve()),
        "checkpoint_complete_sha256": checkpoint_complete_sha,
        "checkpoint_state": checkpoint_state,
        "checkpoint_inference_lineage": checkpoint_lineage,
        "prepared_manifest": str(prepared_path),
        "prepared_manifest_sha256": prepared_manifest_sha,
        "prepared_dataset_fingerprint": prepared.dataset_fingerprint,
        "prepared_generator_source_sha256": prepared.generator_source_sha256,
        "expert_initialization": config.architecture.expert_initialization,
        "dense_control_fingerprint": dense_control_fingerprint,
        "dense_baseline": dense_baseline_identity,
        "random_baseline": random_baseline_identity,
        "roles": list(selected_roles),
        "batch_size": batch_size,
        "device_type": evaluation_device_type,
        "dtype": evaluation_dtype,
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(torch.device(device))
                if device.startswith("cuda")
                else "cpu"
            ),
            "compute_capability": (
                ".".join(
                    str(part) for part in torch.cuda.get_device_capability(torch.device(device))
                )
                if device.startswith("cuda")
                else None
            ),
            "fla_tilelang": os.environ.get("FLA_TILELANG"),
            "cuda_home": os.environ.get("CUDA_HOME"),
        },
    }
    plan["plan_fingerprint"] = _canonical_sha256(plan)
    plan_path = _install_or_validate_plan(output, plan)

    from .training.builder import build_transfer_model
    from .training.stateful import TrainableModelState

    built = build_transfer_model(config, device=device, dtype=dtype)
    loaded = checkpoint_manager.load(
        {"model": TrainableModelState(built.model)},
        resolved_checkpoint,
        expected_critical_fingerprint=None,
        expected_data_fingerprint=report.data_fingerprint,
        expected_run_id=config.run_id,
        expected_stage=config.stage,
        expected_global_batch_tokens=config.data.global_batch_tokens,
    )
    if (
        loaded.path != resolved_checkpoint
        or sha256_file(loaded.path / "COMPLETE") != checkpoint_complete_sha
    ):
        raise RuntimeError("evaluation checkpoint identity changed after PLAN publication")

    results: dict[str, Any] = {}
    # ``evaluate_nll`` already owns .eval.lock. Keep this scope explicit so
    # every model and teacher cleanup remains inside the locked lifecycle.
    with nullcontext():
        try:
            candidate_roles = [
                role for role in selected_roles if role in {"candidate", "shared", "dense-oracle"}
            ]
            for role in candidate_roles:
                _configure_candidate_mode(
                    built.transfer_modules,
                    role=role,
                    top_k=config.architecture.top_k,
                )
                role_fingerprint = _canonical_sha256(
                    {"plan_fingerprint": plan["plan_fingerprint"], "role": role}
                )
                results[role] = _evaluate_role(
                    built.model,
                    prepared,
                    prepared_path,
                    output,
                    role=role,
                    role_fingerprint=role_fingerprint,
                    batch_size=batch_size,
                    device=device,
                    dtype=dtype,
                    use_bf16=config.runtime.bf16,
                    stop_file=active_stop_file,
                )
        finally:
            del built
            gc.collect()
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if "teacher" in selected_roles:
            from .model_loading import freeze_module, load_qwen35_text_causal_lm

            teacher = load_qwen35_text_causal_lm(
                config.sources.teacher.local_path,
                dtype=dtype,
                device=device,
            )
            freeze_module(teacher)
            try:
                role_fingerprint = _canonical_sha256(
                    {"plan_fingerprint": plan["plan_fingerprint"], "role": "teacher"}
                )
                results["teacher"] = _evaluate_role(
                    teacher,
                    prepared,
                    prepared_path,
                    output,
                    role="teacher",
                    role_fingerprint=role_fingerprint,
                    batch_size=batch_size,
                    device=device,
                    dtype=dtype,
                    use_bf16=config.runtime.bf16,
                    stop_file=active_stop_file,
                )
            finally:
                del teacher
                gc.collect()
                if device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        token_counts = {int(value["predicted_tokens"]) for value in results.values()}
        if len(token_counts) != 1:
            raise RuntimeError("evaluation roles processed different token counts")
        evaluated_tokens = next(iter(token_counts))
        for label, baseline in (
            ("Stage-B dense", dense_baseline),
            ("random-control", random_baseline),
        ):
            if (
                baseline is not None
                and int(baseline["roles"]["candidate"]["predicted_tokens"]) != evaluated_tokens
            ):
                raise RuntimeError(f"{label} baseline processed a different token count")
            if baseline is not None:
                current_shared = float(results["shared"]["mean_nll"])
                baseline_shared = float(baseline["roles"]["shared"]["mean_nll"])
                if not math.isclose(
                    current_shared,
                    baseline_shared,
                    rel_tol=1e-5,
                    abs_tol=1e-6,
                ):
                    raise RuntimeError(
                        f"{label} shared-only NLL differs under supposedly identical evaluation numerics"
                    )
        manifest = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "kind": "twen_nll_evaluation",
            "plan_path": str(plan_path),
            "plan_sha256": sha256_file(plan_path),
            "plan_fingerprint": plan["plan_fingerprint"],
            "run_id": config.run_id,
            "track": config.track,
            "stage": config.stage,
            "expert_initialization": config.architecture.expert_initialization,
            "dense_control_fingerprint": dense_control_fingerprint,
            "checkpoint_state": checkpoint_state,
            "checkpoint_inference_lineage": checkpoint_lineage,
            "runtime": plan["runtime"],
            "roles": results,
            "baselines": {
                "dense_stage_b": dense_baseline_identity,
                "random_control": random_baseline_identity,
            },
            "acceptance": _acceptance_metrics(
                config.stage,
                results,
                dense_baseline=dense_baseline,
                random_baseline=random_baseline,
                expert_initialization=config.architecture.expert_initialization,
            ),
        }
        manifest_path = output / "manifest.json"
        atomic_write_json(manifest_path, manifest)
        atomic_write_text(output / "COMPLETE", f"{sha256_file(manifest_path)}\n")
    return {
        "manifest": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "acceptance": manifest["acceptance"],
        "roles": {
            name: {
                "mean_nll": value["mean_nll"],
                "perplexity": value["perplexity"],
                "predicted_tokens": value["predicted_tokens"],
            }
            for name, value in results.items()
        },
    }


def _verify_export_bundle(model_path: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(model_path).resolve()
    manifest_path = root / "twen_manifest.json"
    complete_path = root / "COMPLETE"
    if not root.is_dir() or not manifest_path.is_file() or not complete_path.is_file():
        raise ValueError(f"export bundle is incomplete: {root}")
    expected_manifest_sha = complete_path.read_text(encoding="ascii").strip()
    if sha256_file(manifest_path) != expected_manifest_sha:
        raise ValueError("export COMPLETE does not authenticate twen_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("bundle_files")
    if not isinstance(files, dict) or not files:
        raise ValueError("export manifest has no bundle_files inventory")
    unexpected_directories = [item for item in root.iterdir() if not item.is_file()]
    if unexpected_directories:
        raise ValueError(
            f"export bundle contains an unexpected directory: {unexpected_directories[0]}"
        )
    expected_names = set(files) | {"twen_manifest.json", "COMPLETE"}
    actual_names = {item.name for item in root.iterdir() if item.is_file()}
    if actual_names != expected_names:
        raise ValueError(
            "export bundle file set differs from its manifest: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    for name, identity in files.items():
        path = root / name
        if (
            not isinstance(identity, dict)
            or path.stat().st_size != int(identity.get("size", -1))
            or sha256_file(path) != identity.get("sha256")
        ):
            raise ValueError(f"export bundle file failed integrity validation: {name}")
    return root, manifest


def verify_inference_consistency(
    *,
    model_path: str,
    prompts: Sequence[str],
    output_path: str,
    max_new_tokens: int = 32,
    chat: bool = False,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
) -> dict[str, Any]:
    """Compare fixed-length greedy token IDs from Transformers and vLLM."""

    enforce_offline_environment()
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("at least one non-empty prompt is required")
    if max_new_tokens <= 0 or tensor_parallel_size <= 0:
        raise ValueError("token/GPU counts must be positive")
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    root, bundle_manifest = _verify_export_bundle(model_path)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    if not torch.cuda.is_available():
        raise RuntimeError("inference consistency requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if chat
        else prompt
        for prompt in prompts
    ]
    model = AutoModelForCausalLM.from_pretrained(
        root,
        local_files_only=True,
        dtype=torch.bfloat16,
    ).to("cuda")
    model.eval()
    # Never inherit arbitrary chat/generation settings from the tokenizer
    # bundle for an engine-equivalence test.  A fresh, neutral greedy config
    # avoids repetition penalties, beam search, forced tokens, stop strings,
    # and other source defaults that vLLM would not apply.
    pad_token_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    generation_config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,
        eos_token_id=None,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        pad_token_id=pad_token_id,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        renormalize_logits=False,
        remove_invalid_values=False,
        use_cache=True,
    )
    transformers_tokens: list[list[int]] = []
    try:
        for prompt in rendered:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {name: value.to("cuda") for name, value in inputs.items()}
            input_length = int(inputs["input_ids"].shape[1])
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    use_cache=True,
                )
            transformers_tokens.append(
                [int(value) for value in generated[0, input_length:].cpu().tolist()]
            )
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    from vllm import LLM, SamplingParams

    engine = LLM(
        model=str(root),
        tokenizer=str(root),
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=False,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_log_stats=True,
    )
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        max_tokens=max_new_tokens,
        # With ignore_eos=True, min_tokens is unnecessary.  Keeping it at the
        # neutral zero is important: vLLM masks EOS logits before min_tokens,
        # while the HF side treats EOS as an ordinary token for this test.
        min_tokens=0,
        ignore_eos=True,
        stop=None,
        stop_token_ids=None,
    )
    try:
        generated = engine.generate(rendered, sampling, use_tqdm=False)
        vllm_tokens = [[int(value) for value in item.outputs[0].token_ids] for item in generated]
    finally:
        del engine
        gc.collect()
        torch.cuda.empty_cache()

    comparisons = []
    for prompt, hf_tokens, fast_tokens in zip(
        prompts,
        transformers_tokens,
        vllm_tokens,
        strict=True,
    ):
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(zip(hf_tokens, fast_tokens, strict=False))
                if left != right
            ),
            None,
        )
        if mismatch is None and len(hf_tokens) != len(fast_tokens):
            mismatch = min(len(hf_tokens), len(fast_tokens))
        comparisons.append(
            {
                "prompt": prompt,
                "transformers_token_ids": hf_tokens,
                "vllm_token_ids": fast_tokens,
                "equal": hf_tokens == fast_tokens,
                "first_mismatch": mismatch,
            }
        )
    result = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "kind": "transformers_vllm_token_consistency",
        "model_path": str(root),
        "model_artifact_sha256": bundle_manifest["artifact_sha256"],
        "max_new_tokens": max_new_tokens,
        "chat": chat,
        "tensor_parallel_size": tensor_parallel_size,
        "consistent": all(item["equal"] for item in comparisons),
        "comparisons": comparisons,
    }
    target = Path(output_path).resolve()
    atomic_write_json(target, result)
    return {"output": str(target), "sha256": sha256_file(target), **result}


__all__ = [
    "EVALUATION_ROLES",
    "EvaluationStopped",
    "evaluate_nll",
    "verify_inference_consistency",
]
