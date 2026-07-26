"""User-invoked dense/sparse training engine with exact interruption rollback."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import platform
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..config import TrainConfig, dump_resolved_config
from ..data import (
    AuthenticatedSourceMap,
    DeterministicCooldownCursor,
    DeterministicGlobalCursor,
    DeterministicSourceMixCursor,
    SourceMixSampleReference,
)
from ..preflight import PreflightReport, run_coordinated_training_preflight
from ..runtime.checkpoint import CheckpointError, CheckpointManager, LoadedCheckpoint
from ..runtime.signals import ImmediateExit, SignalController
from ..runtime.state import (
    DataCursor,
    RNGState,
    TrainerState,
    capture_committed_boundary,
)
from ..source_identity import twen_source_tree_sha256
from ..utils import sha256_file
from .builder import BuiltModel, build_parameter_groups, build_transfer_model, load_layer_mapping
from .data import (
    KDRecordStore,
    PreparedTextRecordStore,
    move_kd_batch,
    move_prepared_text_batch,
)
from .distributed import (
    DistributedContext,
    accumulation_sync,
    all_reduce_max,
    all_reduce_sum,
    finalize_distributed,
    initialize_distributed,
    wrap_distributed,
    wrap_frozen_text_model,
)
from .logging import (
    JsonlEventLogger,
    JsonlMetricLogger,
    RankZeroSessionFile,
    TrainingProgress,
    TrainingTelemetryTracker,
    exception_fields,
    utc_now,
)
from .losses import (
    best_expert_pair,
    load_balancing_loss,
    router_pair_supervision_loss,
    router_z_loss,
)
from .schedule import SparseTopKSchedule
from .stateful import (
    OptimizerBundle,
    OptimizerState,
    TokenLRScheduler,
    TrainableModelState,
    materialize_adamw_state,
    materialize_muon_state,
)
from .streaming import StreamingLossCausalLM, native_mtp_target_mask
from .teacher_offload import TeacherCPUOffloadManager


def _rank_zero_print(context: DistributedContext, value: Mapping[str, Any]) -> None:
    if context.is_rank_zero:
        print(json.dumps(dict(value), ensure_ascii=False, sort_keys=True), flush=True)


def _data_governance_log_fields(report: PreflightReport) -> dict[str, object]:
    """Keep the research-only warning in every durable training session log."""

    return {"data_governance": dataclasses.asdict(report.data_governance)}


PROFILE_STEP_UNIT = "microbatch"


def _advance_profiler_after_microbatch(profiler: Any) -> None:
    """Advance the bounded profiler once per microbatch, not optimizer step."""

    profiler.step()


class _NullProfiler:
    def __enter__(self) -> _NullProfiler:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def step(self) -> None:
        return None


def _build_profiler(
    config: TrainConfig,
    run_dir: Path,
    context: DistributedContext,
) -> Any:
    if not config.runtime.profile:
        return _NullProfiler()
    import torch

    trace_dir = run_dir / "profiles" / f"rank-{context.rank:05d}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=config.runtime.profile_wait_steps,
            warmup=config.runtime.profile_warmup_steps,
            active=config.runtime.profile_active_steps,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir), use_gzip=True),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )


def _cuda_memory_metrics(device: Any) -> dict[str, float]:
    import torch

    gib = float(1024**3)
    return {
        "gpu_allocated_gib": torch.cuda.memory_allocated(device) / gib,
        "gpu_reserved_gib": torch.cuda.memory_reserved(device) / gib,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
    }


def _set_seed(
    seed: int, context: DistributedContext, deterministic: bool, allow_tf32: bool
) -> None:
    import random

    import numpy as np
    import torch

    rank_seed = seed + context.rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if deterministic:
        torch.use_deterministic_algorithms(True)


TrainingDataCursor = (
    DeterministicGlobalCursor
    | DeterministicCooldownCursor
    | DeterministicSourceMixCursor
)
TrainingRecordStore = KDRecordStore | PreparedTextRecordStore


def _training_data_mode(config: TrainConfig) -> str:
    return str(getattr(getattr(config, "data", None), "mode", "teacher-kd"))


def _teacher_kd_data_enabled(config: TrainConfig) -> bool:
    """Return whether batches carry the authenticated cached-teacher payload."""

    return _training_data_mode(config) == "teacher-kd"


def _source_mix_enabled(config: TrainConfig) -> bool:
    data = getattr(config, "data", None)
    enabled = getattr(data, "source_mix_enabled", None)
    if callable(enabled):
        return bool(enabled())
    # Legacy tests and downstream callers may provide the historical
    # SimpleNamespace-shaped config.  Absence of every opt-in field must retain
    # the pre-v4 cursor contract instead of failing during resume discovery.
    return getattr(data, "source_mix_algorithm", None) is not None


def _source_mix_log_contract(report: PreflightReport) -> dict[str, object]:
    """Return the explicit lineage/effective weight contract for durable logs."""

    effective = tuple(
        getattr(
            report,
            "source_mix_effective_basis_points",
            getattr(report, "source_mix_basis_points", ()),
        )
        or ()
    )
    lineage = tuple(
        getattr(report, "source_mix_lineage_basis_points", ()) or ()
    )
    override = bool(getattr(report, "source_mix_weight_override", False))
    return {
        "enabled": bool(getattr(report, "source_mix_enabled", False)),
        "algorithm": getattr(report, "source_mix_algorithm", None),
        "source_map_sha256": getattr(report, "source_map_sha256", None),
        "dataset_fingerprint": getattr(
            report,
            "source_mix_dataset_fingerprint",
            None,
        ),
        # Compatibility alias remains the effective runtime mix.
        "basis_points": dict(effective),
        "lineage_basis_points": dict(lineage),
        "effective_basis_points": dict(effective),
        "weight_override": override,
        "seed": getattr(report, "source_mix_seed", None),
    }


def _source_mix_session_log_fields(report: PreflightReport) -> dict[str, object]:
    """Return the flat session-event compatibility fields plus v4 identities."""

    contract = _source_mix_log_contract(report)
    return {
        "source_mix_enabled": contract["enabled"],
        "source_mix_algorithm": contract["algorithm"],
        "source_map_sha256": contract["source_map_sha256"],
        "source_mix_dataset_fingerprint": contract["dataset_fingerprint"],
        # The historical field remains an alias for the effective runtime mix.
        "source_mix_basis_points": contract["basis_points"],
        "source_mix_lineage_basis_points": contract["lineage_basis_points"],
        "source_mix_effective_basis_points": contract["effective_basis_points"],
        "source_mix_weight_override": contract["weight_override"],
        "source_mix_seed": contract["seed"],
    }


def _source_map_from_preflight(
    config: TrainConfig,
    report: PreflightReport,
) -> AuthenticatedSourceMap:
    """Consume only the source map emitted by full coordinated preflight."""

    if not _source_mix_enabled(config) or not bool(
        getattr(report, "source_mix_enabled", False)
    ):
        raise RuntimeError("source-map reconstruction requires enabled source mixing")
    if report.source_map_payload_json is None:
        raise RuntimeError("preflight omitted the authenticated source-map payload")
    try:
        payload = json.loads(report.source_map_payload_json)
        if not isinstance(payload, Mapping):
            raise ValueError("source-map payload must be an object")
        source_map = AuthenticatedSourceMap.from_dict(payload)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"preflight source-map payload is invalid: {error}"
        ) from error
    configured_weights = config.data.source_mix_basis_points
    lineage_weights = source_map.source_mix_weights
    configured_override = bool(
        getattr(config.data, "source_mix_allow_weight_override", False)
    )
    report_lineage_weights = dict(
        getattr(report, "source_mix_lineage_basis_points", ()) or ()
    )
    report_effective_weights = dict(
        getattr(
            report,
            "source_mix_effective_basis_points",
            report.source_mix_basis_points,
        )
        or ()
    )
    if (
        source_map.fingerprint != config.data.source_map_sha256
        or source_map.fingerprint != report.source_map_sha256
        or report.source_mix_algorithm != config.data.source_mix_algorithm
        or dict(report.source_mix_basis_points) != configured_weights
        or report_effective_weights != configured_weights
        or report_lineage_weights != lineage_weights
        or bool(getattr(report, "source_mix_weight_override", False))
        != configured_override
        or configured_override != (lineage_weights != configured_weights)
        or report.source_mix_seed != config.data.shuffle_seed
    ):
        raise RuntimeError("preflight/runtime source-mix identity mismatch")
    cursor = DeterministicSourceMixCursor(
        source_map,
        configured_weights,
        seed=config.data.shuffle_seed,
    )
    if cursor.dataset_fingerprint != report.source_mix_dataset_fingerprint:
        raise RuntimeError("preflight/runtime source-mix dataset fingerprint mismatch")
    return source_map


@dataclasses.dataclass(frozen=True, slots=True)
class _SourceMixCommitPayload:
    planned_references: tuple[SourceMixSampleReference, ...]
    plan_fingerprint: str
    valid_tokens_per_reference: tuple[int, ...]
    valid_tokens_by_source: dict[str, int]
    token_count: int

    def validate(self, cursor: DeterministicSourceMixCursor) -> None:
        cursor.validate_commit(
            planned_references=self.planned_references,
            plan_fingerprint=self.plan_fingerprint,
            valid_tokens_per_reference=self.valid_tokens_per_reference,
            valid_tokens_by_source=self.valid_tokens_by_source,
            token_count=self.token_count,
        )

    def commit(self, cursor: DeterministicSourceMixCursor) -> None:
        cursor.commit(
            planned_references=self.planned_references,
            plan_fingerprint=self.plan_fingerprint,
            valid_tokens_per_reference=self.valid_tokens_per_reference,
            valid_tokens_by_source=self.valid_tokens_by_source,
            token_count=self.token_count,
        )


def _prepare_source_mix_commit(
    cursor: DeterministicSourceMixCursor,
    rank_references: Sequence[SourceMixSampleReference],
    local_valid_tokens: Sequence[int],
    context: DistributedContext,
) -> _SourceMixCommitPayload:
    """Cross-rank authenticate one pending source-mix update without committing it."""

    import torch

    pending = cursor.pending_global_batch
    fingerprint = cursor.pending_plan_fingerprint
    if not pending or fingerprint is None:
        raise RuntimeError("source-mix optimizer batch has no pending global plan")
    expected_rank_references = tuple(
        pending[index] for index in range(context.rank, len(pending), context.world_size)
    )
    references = tuple(rank_references)
    if references != expected_rank_references:
        raise RuntimeError("rank source-mix references differ from the pending global plan")
    counts = tuple(local_valid_tokens)
    if len(counts) != len(references):
        raise RuntimeError(
            "rank source-mix valid-token count differs from its planned references"
        )
    for index, count in enumerate(counts):
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 1 <= count <= cursor.source_map.sequence_length
        ):
            raise RuntimeError(
                f"rank source-mix valid-token count {index} is outside prepared capacity"
            )

    source_ids = cursor.source_map.source_ids
    source_indices = {source_id: index for index, source_id in enumerate(source_ids)}
    reference_count = len(pending)
    source_offset = reference_count
    total_offset = source_offset + len(source_ids)
    fingerprint_offset = total_offset + 1
    reduced = torch.zeros(
        fingerprint_offset + 32,
        dtype=torch.int64,
        device=context.device,
    )
    for local_index, (reference, count) in enumerate(
        zip(references, counts, strict=True)
    ):
        global_index = context.rank + local_index * context.world_size
        reduced[global_index] = count
        reduced[source_offset + source_indices[reference.source_id]] += count
        reduced[total_offset] += count
    reduced[fingerprint_offset:] = torch.tensor(
        tuple(bytes.fromhex(fingerprint)),
        dtype=torch.int64,
        device=context.device,
    )
    all_reduce_sum(reduced, context)
    values = tuple(int(value) for value in reduced.cpu().tolist())
    expected_fingerprint = tuple(
        byte * context.world_size for byte in bytes.fromhex(fingerprint)
    )
    if values[fingerprint_offset:] != expected_fingerprint:
        raise RuntimeError("source-mix pending plan fingerprint differs across ranks")

    payload = _SourceMixCommitPayload(
        planned_references=pending,
        plan_fingerprint=fingerprint,
        valid_tokens_per_reference=values[:reference_count],
        valid_tokens_by_source={
            source_id: values[source_offset + index]
            for index, source_id in enumerate(source_ids)
        },
        token_count=values[total_offset],
    )
    # This is deliberately separate from ``commit``: malformed cross-rank
    # accounting must fail before any optimizer component mutates a parameter.
    payload.validate(cursor)
    return payload


def _optimizer_step_and_commit(
    optimizer: Any,
    cursor: TrainingDataCursor,
    *,
    global_batch_samples: int,
    committed_tokens: int,
    source_mix_commit: _SourceMixCommitPayload | None,
) -> None:
    """Mutate optimizer first only after all cursor accounting can commit."""

    if source_mix_commit is not None:
        if not isinstance(cursor, DeterministicSourceMixCursor):
            raise RuntimeError("source-mix commit payload was built for a legacy cursor")
        if committed_tokens != source_mix_commit.token_count:
            raise RuntimeError(
                "loaded optimizer-batch valid tokens differ from the "
                "authenticated source-mix reference counts"
            )
        source_mix_commit.validate(cursor)
        optimizer.step()
        source_mix_commit.commit(cursor)
        return
    optimizer.step()
    cursor.commit(
        global_batch_samples=global_batch_samples,
        token_count=committed_tokens,
    )


def _build_training_record_store(
    config: TrainConfig,
    *,
    manifest_path: str | Path | None = None,
    verify_shards: bool,
) -> TrainingRecordStore:
    """Construct exactly one mode-specific store without probing the other mode."""

    if _teacher_kd_data_enabled(config):
        selected_manifest = (
            config.data.teacher_kd_manifest_path
            if manifest_path is None
            else manifest_path
        )
        if selected_manifest is None:
            raise RuntimeError("teacher-kd mode requires a KD manifest")
        return KDRecordStore(
            selected_manifest,
            expected_temperature=config.losses.kd_temperature,
            expected_sequence_length=config.data.max_sequence_length,
            verify_shards=verify_shards,
        )
    if _training_data_mode(config) != "prepared-text":
        raise RuntimeError(f"unsupported training data mode: {_training_data_mode(config)!r}")
    if manifest_path is not None and Path(manifest_path) != Path(config.data.manifest_path):
        raise RuntimeError("prepared-text mode does not support a separate cooldown store")
    return PreparedTextRecordStore(
        config.data.manifest_path,
        expected_sequence_length=config.data.max_sequence_length,
        verify_shards=verify_shards,
    )


def _move_training_batch(config: TrainConfig, batch: Any, device: Any) -> Any:
    """Move one mode-specific host batch without reading absent KD attributes."""

    if _teacher_kd_data_enabled(config):
        return move_kd_batch(batch, device)
    return move_prepared_text_batch(batch, device)


def _runtime_cursor(cursor: TrainingDataCursor) -> DataCursor:
    return DataCursor(
        global_sample_index=cursor.next_global_sample,
        global_token_index=cursor.committed_tokens,
        shuffle_seed=cursor.seed,
        extra=cursor.state_dict(),
    )


def _build_trainer_state(
    config: TrainConfig,
    report: PreflightReport,
    *,
    global_step: int = 0,
    committed_tokens: int = 0,
    top_k: int | None = None,
) -> TrainerState:
    return TrainerState(
        run_id=config.run_id,
        stage=config.stage,
        global_step=global_step,
        committed_tokens=committed_tokens,
        gradient_accumulation_steps=report.batch.gradient_accumulation_steps,
        global_batch_tokens=report.batch.global_batch_tokens,
        micro_batch_tokens_per_rank=report.batch.micro_batch_tokens_per_rank,
        world_size=report.batch.world_size,
        top_k=top_k,
        loss_weights={
            key: float(value) for key, value in dataclasses.asdict(config.losses).items()
        },
    )


def _build_optimizer(config: TrainConfig, built: BuiltModel) -> Any:
    import torch

    parameters = [
        parameter
        for module in built.transfer_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    invalid = [
        (tuple(parameter.shape), str(parameter.dtype))
        for parameter in parameters
        if not parameter.is_floating_point() or parameter.dtype != torch.float32
    ]
    if invalid:
        raise RuntimeError(
            f"all trainable adapter/router/LoRA/scale parameters must be FP32; found {invalid[:3]}"
        )
    parameter_groups = build_parameter_groups(config, built.transfer_modules)
    if config.optimizer.adapter_optimizer == "adamw":
        fused = bool(
            config.runtime.fused_adamw and all(parameter.is_cuda for parameter in parameters)
        )
        optimizer = torch.optim.AdamW(
            parameter_groups,
            betas=(config.optimizer.adam_beta1, config.optimizer.adam_beta2),
            eps=config.optimizer.adam_eps,
            fused=fused,
        )
        materialize_adamw_state(optimizer)
        return optimizer

    if config.optimizer.adapter_optimizer != "muon":
        raise RuntimeError(f"unsupported adapter optimizer: {config.optimizer.adapter_optimizer!r}")
    if config.stage != "dense-oracle":
        raise RuntimeError("Muon adapter optimization currently requires dense-oracle stage")
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # pragma: no cover - locked Torch provides DTensor
        DTensor = ()  # type: ignore[assignment,misc]
    if any(isinstance(parameter, DTensor) for parameter in parameters):
        raise RuntimeError(
            "Muon adapter optimization refuses DTensor parameters; "
            "use the enforced single-GPU/world-size-1 path"
        )
    muon_class = getattr(torch.optim, "Muon", None)
    if muon_class is None:
        raise RuntimeError(
            "adapter_optimizer='muon' requires a PyTorch build with torch.optim.Muon"
        )

    adapter_groups = [group for group in parameter_groups if group.get("name") == "adapters"]
    adamw_groups = [group for group in parameter_groups if group.get("name") != "adapters"]
    if not adapter_groups:
        raise RuntimeError("Muon was selected but no adapter parameter group was constructed")
    unexpected_adamw_groups = [
        str(group.get("name", "unnamed")) for group in adamw_groups if group.get("name") != "scale"
    ]
    if unexpected_adamw_groups:
        raise RuntimeError(
            "dense Muon optimization only permits AdamW branch scales; "
            f"found groups={unexpected_adamw_groups}"
        )
    if not adamw_groups:
        raise RuntimeError("dense Muon optimization requires an AdamW branch-scale group")

    invalid_adapters = [
        tuple(parameter.shape)
        for group in adapter_groups
        for parameter in group["params"]
        if parameter.ndim != 2
    ]
    if invalid_adapters:
        raise RuntimeError(f"Muon adapters must all be 2D matrices; found {invalid_adapters[:3]}")
    invalid_scales = [
        tuple(parameter.shape)
        for group in adamw_groups
        for parameter in group["params"]
        if parameter.ndim != 1
    ]
    if invalid_scales:
        raise RuntimeError(
            f"AdamW branch scales must all be 1D tensors; found {invalid_scales[:3]}"
        )

    muon = muon_class(
        adapter_groups,
        momentum=config.optimizer.muon_momentum,
        nesterov=config.optimizer.muon_nesterov,
        ns_coefficients=config.optimizer.muon_ns_coefficients,
        eps=config.optimizer.muon_eps,
        ns_steps=config.optimizer.muon_ns_steps,
        adjust_lr_fn=config.optimizer.muon_adjust_lr_fn,
    )
    materialize_muon_state(muon)
    adamw_parameters = [parameter for group in adamw_groups for parameter in group["params"]]
    fused_adamw = bool(
        config.runtime.fused_adamw and all(parameter.is_cuda for parameter in adamw_parameters)
    )
    adamw = torch.optim.AdamW(
        adamw_groups,
        betas=(config.optimizer.adam_beta1, config.optimizer.adam_beta2),
        eps=config.optimizer.adam_eps,
        fused=fused_adamw,
    )
    materialize_adamw_state(adamw)
    return OptimizerBundle(
        (muon, adamw),
        expected_parameters=parameters,
    )


def _named_learning_rates(optimizer: Any) -> tuple[tuple[str, float], ...]:
    """Snapshot optimizer group LRs without advancing optimizer/scheduler state."""

    return tuple(
        (str(group.get("name", "unnamed")), float(group["lr"])) for group in optimizer.param_groups
    )


def _muon_lr_adjustment_factor(
    adjust_lr_fn: str | None,
    shape: tuple[int, ...],
) -> float:
    """Mirror Torch Muon's documented matrix-shape LR adjustment."""

    if len(shape) != 2:
        raise RuntimeError(f"Muon LR adjustment requires a 2D shape, got {shape}")
    rows, columns = shape
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        return math.sqrt(max(1.0, rows / columns))
    if adjust_lr_fn == "match_rms_adamw":
        return 0.2 * math.sqrt(max(rows, columns))
    raise RuntimeError(f"unsupported Muon LR adjustment: {adjust_lr_fn!r}")


def _named_adjusted_learning_rates(
    optimizer: Any,
) -> tuple[tuple[str, float, float, float], ...]:
    """Return name, nominal LR, adjusted coefficient, and shape factor."""

    result: list[tuple[str, float, float, float]] = []
    for group in optimizer.param_groups:
        if "adjust_lr_fn" not in group:
            continue
        factors = {
            _muon_lr_adjustment_factor(
                group.get("adjust_lr_fn"),
                tuple(parameter.shape),
            )
            for parameter in group["params"]
        }
        if len(factors) != 1:
            raise RuntimeError(
                "one Muon parameter group has multiple LR adjustment factors; "
                "split it into shape-homogeneous groups"
            )
        factor = factors.pop()
        nominal = float(group["lr"])
        result.append(
            (
                str(group.get("name", "unnamed")),
                nominal,
                nominal * factor,
                factor,
            )
        )
    return tuple(result)


def _learning_rate_step_metrics(
    applied_learning_rates: Sequence[tuple[str, float]],
    next_learning_rates: Sequence[tuple[str, float]],
    applied_adjusted_learning_rates: Sequence[tuple[str, float, float, float]],
    next_adjusted_learning_rates: Sequence[tuple[str, float, float, float]],
) -> dict[str, float]:
    """Render nominal and Muon shape-adjusted rates with an explicit time axis."""

    applied = tuple(applied_learning_rates)
    following = tuple(next_learning_rates)
    if tuple(name for name, _ in applied) != tuple(name for name, _ in following):
        raise RuntimeError("optimizer parameter-group names changed across scheduler update")
    metrics: dict[str, float] = {}
    for (name, applied_lr), (_, next_lr) in zip(applied, following, strict=True):
        metrics[f"lr/{name}"] = float(applied_lr)
        metrics[f"next_lr/{name}"] = float(next_lr)

    applied_adjusted = {
        name: (nominal, adjusted, factor)
        for name, nominal, adjusted, factor in applied_adjusted_learning_rates
    }
    next_adjusted = {
        name: (nominal, adjusted, factor)
        for name, nominal, adjusted, factor in next_adjusted_learning_rates
    }
    if set(applied_adjusted) != set(next_adjusted):
        raise RuntimeError("Muon parameter groups changed across scheduler update")
    nominal_by_name = dict(applied)
    next_nominal_by_name = dict(following)
    for name in sorted(applied_adjusted):
        nominal, adjusted, factor = applied_adjusted[name]
        next_nominal, next_adjusted_lr, next_factor = next_adjusted[name]
        if (
            nominal != nominal_by_name.get(name)
            or next_nominal != next_nominal_by_name.get(name)
            or factor != next_factor
        ):
            raise RuntimeError("Muon nominal/adjusted LR snapshots are inconsistent")
        metrics[f"lr_adjusted/{name}"] = float(adjusted)
        metrics[f"lr_adjustment_factor/{name}"] = float(factor)
        metrics[f"next_lr_adjusted/{name}"] = float(next_adjusted_lr)
    return metrics


def _clip_optimizer_gradients(
    optimizer: Any,
    max_norm: float,
) -> Any:
    """Clip the flattened, disjoint parameter set of a single optimizer or bundle."""

    import torch

    parameters = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise RuntimeError("optimizer parameter groups overlap during gradient clipping")
    return torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm,
        error_if_nonfinite=True,
    )


def _hidden_teacher_enabled(config: TrainConfig) -> bool:
    """Whether any optimizer batch can evaluate hidden alignment."""

    return bool(
        _teacher_kd_data_enabled(config)
        and config.stage == "dense-oracle"
        and config.losses.hidden_alignment > 0
        and config.losses.hidden_alignment_batch_fraction > 0
    )


def _set_activation_checkpointing(model: Any, enabled: bool) -> None:
    """Toggle HF checkpointing without accumulating input-gradient hooks."""

    # Transformers' enable method installs an embedding hook, while its disable
    # method removes that hook only for PEFT models.  Explicitly clearing hooks
    # on both transitions makes repeated optimizer-step toggles idempotent.
    model.disable_input_require_grads()
    if enabled:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
        model.disable_input_require_grads()


def _set_selective_activation_checkpointing(
    model: Any,
    layer_indices: tuple[int, ...],
) -> None:
    """Checkpoint exactly the requested Qwen decoder layers.

    Transformers installs a version-specific checkpoint callable and an input
    gradient hook through ``gradient_checkpointing_enable``.  Install those
    only on the empty-to-nonempty transition, then narrow the per-layer flags;
    this avoids accumulating embedding hooks as ordinary and hidden-alignment
    steps alternate between selective and full checkpointing.
    """

    layers = tuple(getattr(getattr(model, "model", None), "layers", ()))
    if not layers:
        raise RuntimeError("selective activation checkpointing requires Qwen decoder layers")
    selected = tuple(layer_indices)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in selected):
        raise ValueError("activation checkpoint layer indices must be integers")
    if tuple(sorted(set(selected))) != selected:
        raise ValueError("activation checkpoint layer indices must be sorted and unique")
    if any(index < 0 or index >= len(layers) for index in selected):
        raise ValueError(f"activation checkpoint layer indices must be in [0, {len(layers)})")
    previously_enabled = any(
        bool(getattr(layer, "gradient_checkpointing", False)) for layer in layers
    )
    if selected and not previously_enabled:
        model.disable_input_require_grads()
        model.gradient_checkpointing_enable()
    elif not selected and previously_enabled:
        model.gradient_checkpointing_disable()
        model.disable_input_require_grads()
    selected_set = set(selected)
    for index, layer in enumerate(layers):
        if not hasattr(layer, "gradient_checkpointing"):
            raise RuntimeError(f"Qwen decoder layer {index} lacks gradient checkpoint support")
        layer.gradient_checkpointing = index in selected_set
    actual = tuple(
        index
        for index, layer in enumerate(layers)
        if bool(getattr(layer, "gradient_checkpointing", False))
    )
    if actual != selected:
        raise RuntimeError(
            f"selective activation checkpointing mismatch: expected={selected}, actual={actual}"
        )


def _set_dense_transfer_token_checkpointing(
    transfer_modules: tuple[Any, ...] | list[Any],
    student_layer_indices: tuple[int, ...] | list[int],
    *,
    outer_checkpoint_layer_indices: tuple[int, ...],
    enabled: bool,
    checkpoint_layer_indices: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """Checkpoint the exact selected transfer branches outside outer checkpoints.

    A decoder-layer checkpoint already recomputes its MLP during backward.  An
    inner transfer-branch checkpoint on the same layer would nest the two and
    repeat work without buying useful activation memory.  ``None`` retains the
    legacy enabled behavior of checkpointing the complete outer complement;
    an explicit tuple applies the phase-specific resolver's deterministic subset.
    """

    modules = tuple(transfer_modules)
    layers = tuple(student_layer_indices)
    if len(modules) != len(layers):
        raise RuntimeError("transfer module and student-layer mappings have different lengths")
    if tuple(sorted(set(layers))) != layers:
        raise RuntimeError("student transfer layer indices must be sorted and unique")
    outer_indices = tuple(outer_checkpoint_layer_indices)
    if tuple(sorted(set(outer_indices))) != outer_indices:
        raise RuntimeError("outer checkpoint layer indices must be sorted and unique")
    outer = set(outer_indices)
    if not outer.issubset(layers):
        raise RuntimeError("outer checkpoint layers are not active transfer layers")
    if checkpoint_layer_indices is None:
        selected = tuple(layer for layer in layers if enabled and layer not in outer)
    else:
        selected = tuple(checkpoint_layer_indices)
        if tuple(sorted(set(selected))) != selected:
            raise RuntimeError("inner checkpoint layer indices must be sorted and unique")
        if not set(selected).issubset(layers):
            raise RuntimeError("inner checkpoint layers are not active transfer layers")
        if selected and not enabled:
            raise RuntimeError("disabled dense transfer checkpointing cannot select inner layers")
    if outer.intersection(selected):
        raise RuntimeError("nested outer/transfer activation checkpointing is forbidden")
    selected_set = set(selected)
    inner: list[int] = []
    for layer, module in zip(layers, modules, strict=True):
        use_inner = bool(enabled and layer in selected_set)
        configure = getattr(module, "configure_transfer_execution", None)
        if not callable(configure):
            if enabled:
                raise RuntimeError(
                    f"dense transfer module at layer {layer} lacks execution controls"
                )
            continue
        configure(checkpoint_token_branch=use_inner)
        transfer = getattr(module, "transfer_mlp", None)
        actual = getattr(transfer, "checkpoint_token_branch", None)
        if not isinstance(actual, bool) or actual != use_inner:
            raise RuntimeError(
                "dense transfer checkpoint state mismatch at layer "
                f"{layer}: expected={use_inner}, actual={actual!r}"
            )
        if actual:
            inner.append(int(layer))
    actual_inner = tuple(inner)
    if actual_inner != selected:
        raise RuntimeError(
            f"dense transfer checkpoint policy mismatch: expected={selected}, actual={actual_inner}"
        )
    return actual_inner


def _effective_activation_checkpoint_layer_indices(
    config: TrainConfig,
    *,
    align_hidden: bool,
) -> tuple[int, ...]:
    return config.activation_checkpoint_layer_indices(align_hidden=align_hidden)


def _effective_dense_transfer_checkpoint_layer_indices(
    config: TrainConfig,
    *,
    align_hidden: bool,
    outer_checkpoint_layer_indices: tuple[int, ...],
) -> tuple[int, ...]:
    return config.dense_transfer_checkpoint_layer_indices(
        align_hidden=align_hidden,
        outer_checkpoint_layer_indices=outer_checkpoint_layer_indices,
    )


def _checkpoint_policy_requires_update(
    current_outer: tuple[int, ...],
    current_inner: tuple[int, ...],
    desired_outer: tuple[int, ...],
    desired_inner: tuple[int, ...],
) -> bool:
    """Return true when either phase-specific checkpoint policy changed."""

    return current_outer != desired_outer or current_inner != desired_inner


def _checkpoint_phase_log_fields(
    config: TrainConfig,
    *,
    align_hidden: bool,
    outer_checkpoint_layer_indices: tuple[int, ...],
    inner_checkpoint_layer_indices: tuple[int, ...],
) -> dict[str, Any]:
    """Canonical checkpoint policy fields shared by smoke, metrics and telemetry."""

    return {
        "hidden_alignment_step": bool(align_hidden),
        "activation_checkpointing_effective": bool(outer_checkpoint_layer_indices),
        "activation_checkpoint_layer_count_configured": (
            config.runtime.hidden_alignment_activation_checkpoint_layer_count
            if align_hidden
            else config.runtime.activation_checkpoint_layer_count
        ),
        "activation_checkpoint_layer_count_effective": len(outer_checkpoint_layer_indices),
        "activation_checkpoint_layer_indices_effective": list(outer_checkpoint_layer_indices),
        "dense_transfer_execution": config.runtime.dense_transfer_execution,
        "dense_transfer_checkpoint_layer_count_configured": (
            config.runtime.hidden_alignment_dense_transfer_checkpoint_layer_count
            if align_hidden
            else config.runtime.dense_transfer_checkpoint_layer_count
        ),
        "dense_transfer_token_checkpoint_layer_count_effective": len(
            inner_checkpoint_layer_indices
        ),
        "dense_transfer_token_checkpoint_layer_indices_effective": list(
            inner_checkpoint_layer_indices
        ),
    }


def _effective_activation_checkpointing(config: TrainConfig, *, align_hidden: bool) -> bool:
    resolver = getattr(config, "activation_checkpoint_layer_indices", None)
    if callable(resolver):
        return bool(resolver(align_hidden=align_hidden))
    if not config.runtime.activation_checkpointing:
        return False
    alignment_count = getattr(
        config.runtime,
        "hidden_alignment_activation_checkpoint_layer_count",
        None,
    )
    if align_hidden and alignment_count is not None:
        return alignment_count != 0
    if align_hidden and config.runtime.activation_checkpointing_on_alignment_only:
        return True
    ordinary_count = getattr(config.runtime, "activation_checkpoint_layer_count", None)
    if ordinary_count is not None:
        return ordinary_count != 0
    return not config.runtime.activation_checkpointing_on_alignment_only


def _build_transfer_and_teacher(
    config: TrainConfig,
    context: DistributedContext,
    *,
    dtype: Any,
    build_device: str,
) -> tuple[BuiltModel, Any | None]:
    """Build the student and optional teacher with zero-copy donor reuse.

    On one device the dense donor FFNs are exactly the mapped MLP Parameters
    already resident in the frozen 9B hidden-alignment teacher.  Registering
    those same Parameters in the transfer branches avoids a redundant 6.75 GiB
    BF16 copy while preserving both forwards exactly.  Separate FSDP roots
    cannot safely own the same sharded Parameter, so multi-rank execution keeps
    the established independent-loading path.
    """

    teacher_cpu_offload = bool(getattr(config.runtime, "teacher_cpu_offload", False))
    if teacher_cpu_offload and context.world_size != 1:
        raise RuntimeError("runtime.teacher_cpu_offload supports exactly one GPU")

    teacher = None
    if _hidden_teacher_enabled(config):
        from ..model_loading import freeze_module, load_qwen35_text_model

        teacher_device = (
            "cpu"
            if teacher_cpu_offload
            or (context.world_size > 1 and config.runtime.sharding == "fsdp2")
            else str(context.device)
        )
        teacher = load_qwen35_text_model(
            config.sources.teacher.local_path,
            dtype=dtype,
            device=teacher_device,
        )
        freeze_module(teacher)

    shared_donor_teacher = bool(
        teacher is not None
        and context.world_size == 1
        and config.architecture.expert_initialization == "donor"
    )
    built = build_transfer_model(
        config,
        device=build_device,
        dtype=dtype,
        donor_text_model=teacher if shared_donor_teacher else None,
    )
    if teacher is not None and not teacher_cpu_offload:
        teacher = wrap_frozen_text_model(
            teacher,
            context,
            config.runtime.sharding,
        )
    if teacher_cpu_offload and not built.donor_teacher_shared:
        raise RuntimeError("teacher CPU offload requires exact donor/teacher sharing")
    return built, teacher


def _set_transfer_enabled(modules: tuple[Any, ...], enabled: bool) -> None:
    for module in modules:
        module.set_transfer_enabled(enabled)


def _set_record_aux(modules: tuple[Any, ...], enabled: bool) -> None:
    for module in modules:
        module.set_record_aux(enabled)
        module.clear_aux()


def _set_sparse_top_k(modules: tuple[Any, ...], top_k: int) -> None:
    for module in modules:
        if hasattr(module, "set_top_k"):
            module.set_top_k(top_k)


def _fraction_selected(position: int, fraction: float) -> bool:
    if fraction <= 0:
        return False
    if fraction >= 1:
        return True
    interval = max(1, round(1.0 / fraction))
    return position % interval == 0


def _loss_metric_aliases(
    metrics: Mapping[str, Any],
    *,
    include_anchor: bool,
    include_hidden: bool,
    include_sparse: bool,
    include_dense: bool,
    include_mtp: bool = False,
) -> dict[str, Any]:
    """Expose explicit loss names only for components evaluated on this step."""

    sources = {
        "ntp_loss": "ntp",
        "teacher_kd_loss": "teacher_kd",
    }
    if include_mtp:
        sources["mtp_loss"] = "mtp"
    if include_anchor:
        sources["anchor_kl_loss"] = "anchor_kl"
    if include_hidden:
        sources["hidden_alignment_loss"] = "hidden_alignment"
    if include_sparse:
        sources["load_balance_loss"] = "load_balance"
        sources["router_z_loss"] = "router_z"
    if include_dense:
        sources["dense_oracle_loss"] = "dense_oracle"
        sources["router_supervision_loss"] = "router_supervision"
    return {alias: metrics[source] for alias, source in sources.items() if source in metrics}


def _anchor_hidden_states(
    model: Any,
    modules: tuple[Any, ...],
    batch: Any,
    *,
    dtype: Any,
    enabled: bool,
) -> Any:
    import torch

    _set_transfer_enabled(modules, False)
    try:
        device_type = getattr(getattr(batch.input_ids, "device", None), "type", "cuda")
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device_type,
                dtype=dtype,
                enabled=enabled,
            ),
        ):
            outputs = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                anchor_only=True,
            )
            return outputs["anchor_hidden_states"]
    finally:
        _set_transfer_enabled(modules, True)


def _student_language_model_forward(
    model: Any,
    batch: Any,
    *,
    teacher_kd_enabled: bool = True,
    anchor_hidden_states: Any | None,
    output_hidden_states: bool,
) -> Mapping[str, Any]:
    arguments = {
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask,
        "labels": batch.labels,
        "anchor_hidden_states": anchor_hidden_states,
        "output_hidden_states": output_hidden_states,
    }
    if teacher_kd_enabled:
        arguments.update(
            {
                "teacher_indices": batch.topk_indices,
                "teacher_topk_logits": batch.topk_logits,
                "teacher_logsumexp": batch.teacher_logsumexp,
                "teacher_tail_logprob": batch.teacher_tail_logprob,
                "temperature": batch.temperature,
            }
        )
    return model(
        **arguments,
    )


def _router_auxiliary_loss(
    config: TrainConfig,
    modules: tuple[Any, ...],
    *,
    dense: bool,
    mask: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    device = next(modules[0].parameters()).device
    total = torch.zeros((), device=device, dtype=torch.float32)
    counts = 0
    diagnostics = {
        "router_z": torch.zeros((), device=device, dtype=torch.float32),
        "load_balance": torch.zeros((), device=device, dtype=torch.float32),
        "dense_oracle": torch.zeros((), device=device, dtype=torch.float32),
        "router_supervision": torch.zeros((), device=device, dtype=torch.float32),
        "router_entropy": torch.zeros((), device=device, dtype=torch.float32),
        **{
            f"expert_usage_{expert}": torch.zeros((), device=device, dtype=torch.float32)
            for expert in range(config.architecture.num_experts)
        },
    }
    for module in modules:
        aux = module.last_aux
        if not aux or aux.get("router_logits") is None:
            continue
        logits = aux["router_logits"]
        indices = aux["expert_indices"]
        z = router_z_loss(logits, mask)
        balance = load_balancing_loss(
            logits,
            indices,
            config.architecture.num_experts,
            mask,
        )
        layer_loss = config.losses.router_z * z + config.losses.load_balance * balance
        diagnostics["router_z"] += z.detach()
        diagnostics["load_balance"] += balance.detach()
        probabilities = logits.float().softmax(dim=-1)
        token_entropy = -(probabilities * probabilities.clamp_min(1e-20).log()).sum(dim=-1)
        token_weight = mask.reshape(token_entropy.shape).to(token_entropy.dtype)
        denominator = token_weight.sum().clamp_min(1.0)
        entropy = (token_entropy * token_weight).sum() / denominator
        diagnostics["router_entropy"] += entropy.detach()
        assignments = (
            F.one_hot(
                indices.long(),
                num_classes=config.architecture.num_experts,
            )
            .float()
            .mean(dim=-2)
        )
        usage = (assignments * token_weight.unsqueeze(-1)).sum(
            dim=tuple(range(assignments.ndim - 1))
        ) / denominator
        for expert in range(config.architecture.num_experts):
            diagnostics[f"expert_usage_{expert}"] += usage[expert].detach()
        if dense:
            expert_outputs = aux.get("expert_outputs")
            dense_sum = aux.get("dense_sum")
            routed = aux.get("routed_output")
            if expert_outputs is None or dense_sum is None or routed is None:
                raise RuntimeError("dense-oracle batch did not record per-expert outputs")
            reconstructed = routed * config.architecture.num_experts
            dense_token_loss = 1.0 - F.cosine_similarity(
                reconstructed.float(),
                dense_sum.float(),
                dim=-1,
            )
            dense_loss = (dense_token_loss * token_weight).sum() / denominator
            target_pair = best_expert_pair(expert_outputs.detach(), dense_sum.detach())
            pair_loss = router_pair_supervision_loss(logits, target_pair, mask)
            layer_loss = layer_loss + config.losses.dense_oracle * dense_loss
            layer_loss = layer_loss + config.losses.router_supervision * pair_loss
            diagnostics["dense_oracle"] += dense_loss.detach()
            diagnostics["router_supervision"] += pair_loss.detach()
        total = total + layer_loss
        counts += 1
    if counts:
        total = total / counts
        diagnostics = {key: value / counts for key, value in diagnostics.items()}
    return total, diagnostics


def _hidden_alignment_loss(
    small_hidden: tuple[Any, ...],
    teacher_hidden: tuple[Any, ...],
    modules: tuple[Any, ...],
    layer_mapping: tuple[int, ...],
    student_layer_indices: tuple[int, ...],
    mask: Any,
) -> Any:
    import torch
    import torch.nn.functional as F

    losses = []
    for module, student_layer in zip(modules, student_layer_indices, strict=True):
        donor_layer = layer_mapping[student_layer]
        adapter = module.transfer_mlp.adapters
        mapped_teacher = adapter.project_output(teacher_hidden[donor_layer + 1].detach())
        target = small_hidden[student_layer + 1]
        token_loss = 1.0 - F.cosine_similarity(target.float(), mapped_teacher.float(), dim=-1)
        weight = mask.reshape(token_loss.shape).to(token_loss.dtype)
        losses.append((token_loss * weight).sum() / weight.sum().clamp_min(1.0))
    return torch.stack(losses).mean()


def _batch_loss_token_counts(batch: Any) -> tuple[Any, Any]:
    """Return next-token target count and valid hidden-token count on device."""

    target_mask = batch.labels[..., 1:].ne(-100) & batch.attention_mask[..., :-1].ne(0)
    return target_mask.sum(), batch.attention_mask.ne(0).sum()


def _batch_mtp_loss_token_count(batch: Any) -> Any:
    """Return the exact native Qwen3.5 ``L-2`` target count on device."""

    return native_mtp_target_mask(batch.labels, batch.attention_mask).sum()


def _token_mean_contribution(
    mean_loss: Any,
    local_token_count: Any,
    global_token_count: Any,
    *,
    world_size: int,
) -> Any:
    """Scale one microbatch mean for DDP/FSDP global token-mean reduction.

    Distributed wrappers average gradients across ranks.  Multiplying each
    local numerator by ``world_size / global_count`` before backward makes that
    rank average equal the sum over every token in the optimizer batch divided
    by the exact all-rank token count.
    """

    return (
        mean_loss
        * local_token_count.to(dtype=mean_loss.dtype)
        * (float(world_size) / global_token_count.to(dtype=mean_loss.dtype))
    )


def _coordinate_control(
    controller: SignalController, context: DistributedContext
) -> tuple[bool, bool, int, str | None]:
    import torch

    decision = controller.poll()
    if context.world_size == 1:
        return (
            decision.should_checkpoint,
            decision.should_stop,
            decision.checkpoint_generation,
            decision.reason,
        )
    flags = torch.tensor(
        [int(decision.should_checkpoint), int(decision.should_stop)],
        device=context.device,
        dtype=torch.int32,
    )
    all_reduce_max(flags, context)
    checkpoint_flag, stop_flag = flags.cpu().tolist()
    return bool(checkpoint_flag), bool(stop_flag), decision.checkpoint_generation, decision.reason


def _optimizer_checkpoint_contract(config: TrainConfig) -> dict[str, Any]:
    """Return a JSON-safe optimizer identity for audit and resume forensics."""

    optimizer = getattr(config, "optimizer", None)
    adapter_optimizer = str(getattr(optimizer, "adapter_optimizer", "adamw"))
    contract: dict[str, Any] = {
        "adapter_optimizer": adapter_optimizer,
        "bundle": adapter_optimizer == "muon",
        "adapter_component": "Muon" if adapter_optimizer == "muon" else "AdamW",
        "non_adapter_component": "AdamW",
    }
    if adapter_optimizer == "muon":
        contract["muon"] = {
            "momentum": float(optimizer.muon_momentum),
            "nesterov": bool(optimizer.muon_nesterov),
            "ns_coefficients": [float(value) for value in optimizer.muon_ns_coefficients],
            "eps": float(optimizer.muon_eps),
            "ns_steps": int(optimizer.muon_ns_steps),
            "adjust_lr_fn": optimizer.muon_adjust_lr_fn,
        }
    else:
        contract["muon"] = None
    return contract


def _checkpoint(
    manager: CheckpointManager,
    stateful: Mapping[str, Any],
    state: TrainerState,
    cursor: TrainingDataCursor,
    config: TrainConfig,
    report: PreflightReport,
    *,
    kind: str,
    boundary: Any | None,
    tag: str | None = None,
    reason: str | None = None,
    event_logger: JsonlEventLogger | None = None,
) -> Path:
    import torch

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "unknown", None
    lock_path = Path("uv.lock") if Path("uv.lock").is_file() else Path("pyproject.toml")
    dependency_hash = sha256_file(lock_path) if lock_path.is_file() else None
    fields = {
        "kind": kind,
        "tag": tag,
        "reason": reason,
        "step": state.global_step,
        "tokens": state.committed_tokens,
    }
    if event_logger is not None:
        event_logger.log("checkpoint_start", fields)
    started = time.perf_counter()
    try:
        with torch.profiler.record_function(f"twen/checkpoint/{kind}"):
            path = manager.save(
                stateful,
                trainer_state=state,
                data_cursor=_runtime_cursor(cursor),
                critical_fingerprint=report.config_fingerprint,
                data_fingerprint=report.data_fingerprint,
                kind=kind,
                rng_state=RNGState.capture(),
                committed_boundary=boundary,
                tag=tag,
                extra_metadata={
                    "reason": reason,
                    "git_commit": commit,
                    "git_dirty": dirty,
                    "source_tree_sha256": report.source_tree_sha256,
                    "dependency_lock": str(lock_path),
                    "dependency_lock_sha256": dependency_hash,
                    "data_mode": _training_data_mode(config),
                    "source_mix": {
                        **_source_mix_log_contract(report),
                        "cursor_critical_lineage_fingerprint": (
                            cursor.critical_lineage_fingerprint
                            if isinstance(cursor, DeterministicSourceMixCursor)
                            else None
                        ),
                        "committed_samples_by_source": (
                            cursor.committed_samples_by_source
                            if isinstance(cursor, DeterministicSourceMixCursor)
                            else {}
                        ),
                        "committed_tokens_by_source": (
                            cursor.committed_tokens_by_source
                            if isinstance(cursor, DeterministicSourceMixCursor)
                            else {}
                        ),
                    },
                    "optimizer": _optimizer_checkpoint_contract(config),
                    "data_manifest_sha256": config.data.manifest_sha256,
                    "teacher_kd_manifest_sha256": getattr(
                        config.data,
                        "teacher_kd_manifest_sha256",
                        None,
                    ),
                    "quality_cooldown": {
                        "enabled": (
                            getattr(config.data, "quality_cooldown_start_tokens", None) is not None
                        ),
                        "start_tokens": getattr(config.data, "quality_cooldown_start_tokens", None),
                        "prepared_manifest_sha256": (
                            getattr(
                                config.data,
                                "quality_cooldown_manifest_sha256",
                                None,
                            )
                        ),
                        "teacher_kd_manifest_sha256": (
                            getattr(
                                config.data,
                                "quality_cooldown_teacher_kd_manifest_sha256",
                                None,
                            )
                        ),
                    },
                    "mtp": {
                        "enabled": config.losses.mtp > 0,
                        "loss_weight": config.losses.mtp,
                        "source_role": "backbone" if config.losses.mtp > 0 else None,
                        "source_manifest_sha256": (
                            config.sources.backbone.manifest_sha256
                            if config.losses.mtp > 0
                            else None
                        ),
                        "parameters_frozen": True,
                        "checkpointed_as_trainable_delta": False,
                    },
                    "activation_checkpointing": {
                        "enabled": config.runtime.activation_checkpointing,
                        "configured_layer_count": (
                            config.runtime.activation_checkpoint_layer_count
                        ),
                        "hidden_alignment_configured_layer_count": getattr(
                            config.runtime,
                            "hidden_alignment_activation_checkpoint_layer_count",
                            None,
                        ),
                        "ordinary_layer_indices": list(report.activation_checkpoint_layer_indices),
                        "hidden_alignment_layer_indices": list(
                            report.hidden_alignment_activation_checkpoint_layer_indices
                        ),
                    },
                    "dense_transfer_checkpointing": {
                        "enabled": getattr(
                            config.runtime,
                            "dense_transfer_token_checkpoint",
                            False,
                        ),
                        "ordinary_configured_layer_count": getattr(
                            config.runtime,
                            "dense_transfer_checkpoint_layer_count",
                            None,
                        ),
                        "hidden_alignment_configured_layer_count": getattr(
                            config.runtime,
                            "hidden_alignment_dense_transfer_checkpoint_layer_count",
                            None,
                        ),
                        "ordinary_layer_indices": list(
                            getattr(
                                report,
                                "dense_transfer_token_checkpoint_layer_indices",
                                (),
                            )
                        ),
                        "hidden_alignment_layer_indices": list(
                            getattr(
                                report,
                                "hidden_alignment_dense_transfer_token_checkpoint_layer_indices",
                                (),
                            )
                        ),
                    },
                    "data_manifests": {
                        "prepared": config.data.manifest_path,
                        "teacher_kd": getattr(
                            config.data,
                            "teacher_kd_manifest_path",
                            None,
                        ),
                        "quality_cooldown_prepared": (
                            getattr(config.data, "quality_cooldown_manifest_path", None)
                        ),
                        "quality_cooldown_teacher_kd": (
                            getattr(
                                config.data,
                                "quality_cooldown_teacher_kd_manifest_path",
                                None,
                            )
                        ),
                    },
                    "calibration_artifacts": dict(report.calibration_fingerprints),
                    "source_manifests": {
                        "backbone": config.sources.backbone.manifest_sha256,
                        "donor": config.sources.donor.manifest_sha256,
                        "teacher": config.sources.teacher.manifest_sha256,
                        "tokenizer": config.sources.tokenizer.manifest_sha256,
                        "folded_experts": config.sources.folded_experts_sha256,
                    },
                    "source_locations": {
                        role: {
                            "model_id": getattr(config.sources, role).model_id,
                            "revision": getattr(config.sources, role).revision,
                            "local_path": getattr(config.sources, role).local_path,
                        }
                        for role in ("backbone", "donor", "teacher", "tokenizer")
                    }
                    | {"folded_experts": config.sources.folded_experts_path},
                },
            )
    except BaseException as exc:
        if event_logger is not None:
            event_logger.log(
                "checkpoint_failed",
                {
                    **fields,
                    "duration_seconds": time.perf_counter() - started,
                    **exception_fields(exc),
                },
            )
        raise
    if event_logger is not None:
        event_logger.log(
            "checkpoint_complete",
            {
                **fields,
                "duration_seconds": time.perf_counter() - started,
                "path": str(path),
            },
        )
    return path


def _load_or_initialize(
    manager: CheckpointManager,
    stateful: dict[str, Any],
    config: TrainConfig,
    report: PreflightReport,
    store: TrainingRecordStore,
    *,
    cooldown_store: TrainingRecordStore | None = None,
    resume: str,
    fork_from: str | None,
) -> tuple[TrainerState, TrainingDataCursor, LoadedCheckpoint | None]:
    if fork_from and resume != "none":
        raise RuntimeError("--fork-from requires --resume none")
    cooldown_start = getattr(config.data, "quality_cooldown_start_tokens", None)
    source_map = (
        _source_map_from_preflight(config, report)
        if _source_mix_enabled(config)
        else None
    )
    if source_map is not None:
        cursor: TrainingDataCursor = DeterministicSourceMixCursor(
            source_map,
            config.data.source_mix_basis_points,
            seed=config.data.shuffle_seed,
        )
    elif cooldown_start is None:
        cursor: TrainingDataCursor = DeterministicGlobalCursor(
            store.layout, seed=config.data.shuffle_seed
        )
    else:
        if cooldown_store is None:
            raise RuntimeError("quality cooldown config requires a cooldown KD store")
        cursor = DeterministicCooldownCursor(
            store.layout,
            cooldown_store.layout,
            seed=config.data.shuffle_seed,
            cooldown_start_tokens=int(cooldown_start),
        )
    if fork_from:
        source_checkpoint = Path(fork_from).expanduser()
        if not source_checkpoint.is_absolute():
            source_checkpoint = source_checkpoint.resolve()
        manager.load({"model": stateful["model"]}, source_checkpoint)
        return _build_trainer_state(config, report), cursor, None
    if resume == "none":
        return _build_trainer_state(config, report), cursor, None
    try:
        loaded = manager.load(
            stateful,
            resume,
            expected_critical_fingerprint=report.config_fingerprint,
            expected_data_fingerprint=report.data_fingerprint,
            expected_run_id=config.run_id,
            expected_stage=config.stage,
            expected_global_batch_tokens=config.data.global_batch_tokens,
            restore_rng=True,
            strict_cuda_rng=False,
        )
    except CheckpointError as exc:
        has_committed_directory = manager.root.exists() and any(
            item.is_dir() and item.name.startswith("step-") for item in manager.root.iterdir()
        )
        if resume == "auto" and not has_committed_directory:
            raise CheckpointError(
                "--resume auto found no complete checkpoint; first launch must use --resume none"
            ) from exc
        raise
    state = loaded.trainer_state
    state.world_size = report.batch.world_size
    state.micro_batch_tokens_per_rank = report.batch.micro_batch_tokens_per_rank
    state.gradient_accumulation_steps = report.batch.gradient_accumulation_steps
    state.micro_step_in_accumulation = 0
    cursor_payload = loaded.data_cursor.to_global_cursor_state()
    if source_map is not None:
        cursor = DeterministicSourceMixCursor.from_state_dict(
            source_map,
            config.data.source_mix_basis_points,
            cursor_payload,
        )
    elif cooldown_start is None:
        cursor = DeterministicGlobalCursor.from_state_dict(store.layout, cursor_payload)
    else:
        assert cooldown_store is not None
        cursor = DeterministicCooldownCursor.from_state_dict(
            store.layout,
            cooldown_store.layout,
            cursor_payload,
            cooldown_start_tokens=int(cooldown_start),
        )
    return state, cursor, loaded


def _is_read_only_completed_resume(
    state: TrainerState,
    loaded: LoadedCheckpoint | None,
    *,
    max_tokens: int,
) -> bool:
    """True only for a previously committed terminal milestone."""

    return bool(
        loaded is not None
        and state.committed_tokens >= max_tokens
        and loaded.metadata.get("kind") == "milestone"
        and loaded.metadata.get("tag") == "complete"
    )


def _device_synchronize(device: Any) -> None:
    """Synchronize CUDA timing boundaries while remaining CPU-testable."""

    if getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.synchronize(device)


def _local_gradient_health(model: Any) -> tuple[bool, int, int]:
    """Return finite-gradient status, present gradients, and missing gradients.

    Composable FSDP exposes DTensor gradients.  Inspecting the local shard avoids
    an accidental gather solely for this smoke-test diagnostic.
    """

    import torch

    finite = True
    present = 0
    missing = 0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None:
            missing += 1
            continue
        present += 1
        if hasattr(gradient, "to_local"):
            gradient = gradient.to_local()
        if not bool(torch.isfinite(gradient).all().item()):
            finite = False
    return finite and present > 0, present, missing


def _execute_graph_smoke_microbatch(
    config: TrainConfig,
    report: PreflightReport,
    context: DistributedContext,
    built: BuiltModel,
    train_model: Any,
    batch: Any,
    *,
    dtype: Any,
    teacher: Any | None,
    layer_mapping: tuple[int, ...],
    data_source: str,
) -> dict[str, Any]:
    """Execute one production-shaped forward/loss/backward without an optimizer."""

    import torch

    raw_model = built.model
    teacher_kd_enabled = _teacher_kd_data_enabled(config)
    record_dense = config.stage == "sparse" and _fraction_selected(
        0,
        config.losses.dense_oracle_batch_fraction,
    )
    align_hidden = teacher is not None and _fraction_selected(
        0,
        config.losses.hidden_alignment_batch_fraction,
    )
    _set_record_aux(built.transfer_modules, record_dense)

    if context.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)
    _device_synchronize(context.device)
    forward_started = time.perf_counter()

    anchor_hidden_states = None
    if config.losses.anchor_kl > 0:
        anchor_hidden_states = _anchor_hidden_states(
            train_model,
            built.transfer_modules,
            batch,
            dtype=dtype,
            enabled=config.runtime.bf16,
        )

    with torch.autocast(
        device_type=context.device.type,
        dtype=dtype,
        enabled=config.runtime.bf16,
    ):
        outputs = _student_language_model_forward(
            train_model,
            batch,
            teacher_kd_enabled=teacher_kd_enabled,
            anchor_hidden_states=anchor_hidden_states,
            output_hidden_states=align_hidden,
        )
        ntp = outputs["ntp"]
        kd = outputs["teacher_kd"]
        loss = config.losses.ntp * ntp
        components: dict[str, Any] = {
            "total": loss,
            "ntp": ntp,
        }
        if teacher_kd_enabled:
            if kd is None:
                raise RuntimeError("teacher-kd mode omitted the streaming KD loss")
            loss = loss + config.losses.teacher_kd * kd
            components["teacher_kd"] = kd
        elif kd is not None:
            raise RuntimeError("prepared-text mode unexpectedly produced a teacher KD loss")
        if config.losses.mtp > 0:
            mtp = outputs["mtp"]
            if mtp is None:
                raise RuntimeError("MTP loss is enabled but the student graph omitted it")
            loss = loss + config.losses.mtp * mtp
            components["mtp"] = mtp
        if anchor_hidden_states is not None:
            anchor = outputs["anchor_kl"]
            if anchor is None:
                raise RuntimeError("streaming student forward omitted requested anchor KL")
            loss = loss + config.losses.anchor_kl * anchor
            components["anchor_kl"] = anchor

        if config.stage == "sparse":
            router_aux, router_metrics = _router_auxiliary_loss(
                config,
                built.transfer_modules,
                dense=record_dense,
                mask=batch.attention_mask,
            )
            loss = loss + router_aux
            components["router_aux"] = router_aux
            components.update(router_metrics)

        if align_hidden:
            assert teacher is not None
            with torch.no_grad():
                teacher_outputs = teacher(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                )
            hidden_alignment = _hidden_alignment_loss(
                outputs["hidden_states"],
                teacher_outputs.hidden_states,
                built.transfer_modules,
                layer_mapping,
                built.student_layer_indices,
                batch.attention_mask,
            )
            loss = loss + config.losses.hidden_alignment * hidden_alignment
            components["hidden_alignment"] = hidden_alignment

        components["total"] = loss
        scaled_loss = loss / report.batch.gradient_accumulation_steps

    _device_synchronize(context.device)
    forward_seconds = time.perf_counter() - forward_started
    backward_started = time.perf_counter()
    scaled_loss.backward()
    _device_synchronize(context.device)
    backward_seconds = time.perf_counter() - backward_started

    local_grad_finite, local_grad_tensors, local_missing_grad_tensors = _local_gradient_health(
        raw_model
    )
    if context.device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(context.device)
        peak_reserved = torch.cuda.max_memory_reserved(context.device)
    else:
        peak_allocated = 0
        peak_reserved = 0

    # Use worst-rank timings/memory, mean losses, and a global finite/coverage
    # decision.  This keeps rank zero's one-line JSON representative under
    # DDP/FSDP rather than reporting only its local shard.
    timing = torch.tensor(
        [forward_seconds, backward_seconds],
        device=context.device,
        dtype=torch.float64,
    )
    all_reduce_max(timing, context)
    memory = torch.tensor(
        [peak_allocated, peak_reserved],
        device=context.device,
        dtype=torch.int64,
    )
    all_reduce_max(memory, context)
    component_names = sorted(components)
    component_values = torch.stack(
        [components[name].detach().to(dtype=torch.float64) for name in component_names]
    )
    local_loss_finite = all(
        bool(torch.isfinite(value.detach()).all().item()) for value in components.values()
    )
    all_reduce_sum(component_values, context)
    component_values.div_(context.world_size)
    health = torch.tensor(
        [
            int(local_loss_finite),
            int(local_grad_finite),
            local_grad_tensors,
            local_missing_grad_tensors,
        ],
        device=context.device,
        dtype=torch.int64,
    )
    all_reduce_sum(health, context)
    loss_finite = int(health[0].item()) == context.world_size
    grad_finite = int(health[1].item()) == context.world_size
    missing_grad_tensors = int(health[3].item())
    ok = loss_finite and grad_finite and missing_grad_tensors == 0
    return {
        "ok": ok,
        "no_optimizer_created": True,
        "no_optimizer_steps": True,
        "config_fingerprint": report.config_fingerprint,
        "data_fingerprint": report.data_fingerprint,
        "source_tree_sha256": report.source_tree_sha256,
        "stage": config.stage,
        "data_mode": _training_data_mode(config),
        "world_size": context.world_size,
        "data_source": data_source,
        "donor_teacher_shared": built.donor_teacher_shared,
        "micro_batch_size_per_rank": len(batch),
        "forward_seconds": float(timing[0].item()),
        "backward_seconds": float(timing[1].item()),
        "peak_allocated_bytes": int(memory[0].item()),
        "peak_reserved_bytes": int(memory[1].item()),
        "loss_components": {
            name: float(value.item())
            for name, value in zip(component_names, component_values, strict=True)
        },
        "loss_finite": loss_finite,
        "grad_finite": grad_finite,
        "grad_tensors": int(health[2].item()),
        "missing_grad_tensors": missing_grad_tensors,
    }


def _run_graph_smoke(config: TrainConfig, report: PreflightReport) -> int:
    """Build the target distributed graph and run exactly one optimizer-free batch."""

    import torch

    context = initialize_distributed()
    store: TrainingRecordStore | None = None
    try:
        if report.batch.world_size != context.world_size:
            raise RuntimeError("preflight WORLD_SIZE changed before distributed initialization")
        _set_seed(
            config.runtime.seed,
            context,
            config.runtime.deterministic,
            config.runtime.allow_tf32,
        )
        dtype = torch.bfloat16 if config.runtime.bf16 else torch.float32
        build_device = (
            "cpu"
            if context.world_size > 1 and config.runtime.sharding == "fsdp2"
            else str(context.device)
        )
        built, teacher = _build_transfer_and_teacher(
            config,
            context,
            dtype=dtype,
            build_device=build_device,
        )
        raw_model = built.model
        raw_model.config.use_cache = False
        align_hidden = teacher is not None and _fraction_selected(
            0,
            config.losses.hidden_alignment_batch_fraction,
        )
        activation_checkpoint_layer_indices = _effective_activation_checkpoint_layer_indices(
            config,
            align_hidden=align_hidden,
        )
        expected_activation_checkpoint_layers = (
            report.hidden_alignment_activation_checkpoint_layer_indices
            if align_hidden
            else report.activation_checkpoint_layer_indices
        )
        if activation_checkpoint_layer_indices != expected_activation_checkpoint_layers:
            raise RuntimeError(
                "preflight/runtime outer checkpoint policy mismatch: "
                f"expected={expected_activation_checkpoint_layers}, "
                f"actual={activation_checkpoint_layer_indices}"
            )
        _set_selective_activation_checkpointing(
            raw_model,
            activation_checkpoint_layer_indices,
        )
        desired_transfer_checkpoint_layer_indices = (
            _effective_dense_transfer_checkpoint_layer_indices(
                config,
                align_hidden=align_hidden,
                outer_checkpoint_layer_indices=activation_checkpoint_layer_indices,
            )
        )
        transfer_token_checkpoint_layer_indices = _set_dense_transfer_token_checkpointing(
            built.transfer_modules,
            built.student_layer_indices,
            outer_checkpoint_layer_indices=activation_checkpoint_layer_indices,
            enabled=config.runtime.dense_transfer_token_checkpoint,
            checkpoint_layer_indices=desired_transfer_checkpoint_layer_indices,
        )
        expected_transfer_token_checkpoint_layers = (
            report.hidden_alignment_dense_transfer_token_checkpoint_layer_indices
            if align_hidden
            else report.dense_transfer_token_checkpoint_layer_indices
        )
        if transfer_token_checkpoint_layer_indices != expected_transfer_token_checkpoint_layers:
            raise RuntimeError(
                "preflight/runtime dense transfer checkpoint policy mismatch: "
                f"expected={expected_transfer_token_checkpoint_layers}, "
                f"actual={transfer_token_checkpoint_layer_indices}"
            )
        teacher_offload = (
            TeacherCPUOffloadManager.from_transfer_modules(
                teacher,
                built.transfer_modules,
                target_device=context.device,
            )
            if config.runtime.teacher_cpu_offload and teacher is not None
            else None
        )
        loss_model = StreamingLossCausalLM(
            raw_model,
            chunk_tokens=config.runtime.loss_chunk_tokens,
            checkpoint_chunks=config.runtime.loss_checkpoint_chunks,
            compile_loss=config.runtime.compile_streaming_loss,
            mtp=built.mtp,
            teacher_kd_enabled=_teacher_kd_data_enabled(config),
        )
        train_model = wrap_distributed(
            loss_model,
            context,
            config.runtime.sharding,
            transformer_model=raw_model,
        )
        train_model.train()

        if config.stage == "sparse":
            active_top_k = SparseTopKSchedule(
                config.architecture.num_experts,
                config.architecture.top_k,
            ).value(0, config.optimizer.max_tokens)
            _set_sparse_top_k(built.transfer_modules, active_top_k)

        store = _build_training_record_store(
            config,
            verify_shards=False,
        )
        # A non-shuffled cursor intentionally takes the authenticated corpus's
        # first records.  Rank zero includes record zero; other ranks receive
        # the adjacent records needed for a legal distributed microbatch.
        cursor = DeterministicGlobalCursor(store.layout, seed=0, shuffle=False)
        references = cursor.plan_rank_batch(
            config.data.micro_batch_size * context.world_size,
            rank=context.rank,
            world_size=context.world_size,
        )
        batch = _move_training_batch(config, store.batch(references), context.device)
        data_source = (
            "preflight_kd_manifest_first_microbatch"
            if _teacher_kd_data_enabled(config)
            else "preflight_prepared_manifest_first_microbatch"
        )

        layer_mapping = load_layer_mapping(
            config.architecture.layer_map_path,
            config.architecture.student_layers,
        )
        if teacher_offload is None:
            result = _execute_graph_smoke_microbatch(
                config,
                report,
                context,
                built,
                train_model,
                batch,
                dtype=dtype,
                teacher=teacher,
                layer_mapping=layer_mapping,
                data_source=data_source,
            )
        else:
            with teacher_offload.staged() as offload_session:
                result = _execute_graph_smoke_microbatch(
                    config,
                    report,
                    context,
                    built,
                    train_model,
                    batch,
                    dtype=dtype,
                    teacher=teacher,
                    layer_mapping=layer_mapping,
                    data_source=data_source,
                )
            assert offload_session.restore is not None
            result["teacher_cpu_offload"] = {
                "stage": dataclasses.asdict(offload_session.stage),
                "restore": dataclasses.asdict(offload_session.restore),
            }
        result.update(
            _checkpoint_phase_log_fields(
                config,
                align_hidden=align_hidden,
                outer_checkpoint_layer_indices=activation_checkpoint_layer_indices,
                inner_checkpoint_layer_indices=(transfer_token_checkpoint_layer_indices),
            )
        )
        _rank_zero_print(context, result)
        return 0 if result["ok"] else 2
    finally:
        if store is not None:
            store.close()
        finalize_distributed(context, barrier=False)


def run_training(
    config: TrainConfig,
    *,
    resume: str = "auto",
    fork_from: str | None = None,
    dry_run: bool = False,
    graph_smoke: bool = False,
    progress: str = "auto",
) -> int:
    """Run one explicitly requested stage. This function is never called implicitly."""

    if dry_run and graph_smoke:
        raise ValueError("dry_run and graph_smoke are mutually exclusive")

    report = run_coordinated_training_preflight(config)
    current_source_tree = twen_source_tree_sha256()
    if current_source_tree != report.source_tree_sha256:
        raise RuntimeError(
            "Twen source tree changed after coordinated preflight: "
            f"expected {report.source_tree_sha256}, got {current_source_tree}"
        )
    if dry_run:
        print(json.dumps(dataclasses.asdict(report), ensure_ascii=False, indent=2, default=str))
        return 0
    if graph_smoke:
        return _run_graph_smoke(config, report)

    import torch

    context = initialize_distributed()
    manager: CheckpointManager | None = None
    store: TrainingRecordStore | None = None
    cooldown_store: TrainingRecordStore | None = None
    prefetched_batches: Any | None = None
    event_logger: JsonlEventLogger | None = None
    progress_ui: TrainingProgress | None = None
    session_file: RankZeroSessionFile | None = None
    session_status = "failed"
    state: TrainerState | None = None
    teacher_offload: TeacherCPUOffloadManager | None = None
    try:
        if report.batch.world_size != context.world_size:
            raise RuntimeError("preflight WORLD_SIZE changed before distributed initialization")
        _set_seed(
            config.runtime.seed, context, config.runtime.deterministic, config.runtime.allow_tf32
        )
        run_dir = Path(config.checkpoint.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        metric_logger = (
            JsonlMetricLogger(run_dir / "metrics.jsonl") if context.is_rank_zero else None
        )
        telemetry_logger = (
            JsonlMetricLogger(run_dir / "telemetry.jsonl") if context.is_rank_zero else None
        )
        event_logger = JsonlEventLogger(run_dir / "events.jsonl") if context.is_rank_zero else None
        manager = CheckpointManager(
            run_dir,
            rank=context.rank,
            world_size=context.world_size,
            keep_periodic=config.checkpoint.keep_last,
            keep_interrupt=1,
        )
        teacher_kd_enabled = _teacher_kd_data_enabled(config)
        # Acquire before allocating models/optimizer so a duplicate launcher
        # cannot consume GPUs while waiting to discover the run conflict.
        manager.acquire_run_lock()
        if event_logger is not None:
            source_mix_contract = _source_mix_log_contract(report)
            session_file = RankZeroSessionFile(
                run_dir / "rank0-session.json",
                session_id=event_logger.session_id,
                fields={
                    "run_id": config.run_id,
                    "stage": config.stage,
                    "rank": context.rank,
                    "world_size": context.world_size,
                    "hostname": platform.node(),
                    "source_mix": source_mix_contract,
                },
            )
            properties = torch.cuda.get_device_properties(context.device)
            event_logger.log(
                "session_start",
                {
                    "run_id": config.run_id,
                    "stage": config.stage,
                    "track": config.track,
                    "data_mode": _training_data_mode(config),
                    "teacher_kd_enabled": _teacher_kd_data_enabled(config),
                    "adapter_optimizer": config.optimizer.adapter_optimizer,
                    "rank": context.rank,
                    "world_size": context.world_size,
                    "device": str(context.device),
                    "gpu_name": properties.name,
                    "gpu_total_memory_bytes": properties.total_memory,
                    "gpu_compute_capability": f"{properties.major}.{properties.minor}",
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "pid": os.getpid(),
                    "rank0_session_file": str(session_file.path.resolve()),
                    "config_fingerprint": report.config_fingerprint,
                    "data_fingerprint": report.data_fingerprint,
                    "source_tree_sha256": report.source_tree_sha256,
                    **_source_mix_session_log_fields(report),
                    "source_mix": source_mix_contract,
                    **_data_governance_log_fields(report),
                    "progress_mode": progress,
                    "profile_enabled": config.runtime.profile,
                    "profile_step_unit": PROFILE_STEP_UNIT,
                    "profile_wait_steps": config.runtime.profile_wait_steps,
                    "profile_warmup_steps": config.runtime.profile_warmup_steps,
                    "profile_active_steps": config.runtime.profile_active_steps,
                    "loss_chunk_tokens": config.runtime.loss_chunk_tokens,
                    "loss_checkpoint_chunks": config.runtime.loss_checkpoint_chunks,
                    "compile_streaming_loss": config.runtime.compile_streaming_loss,
                    "mtp_enabled": config.losses.mtp > 0,
                    "mtp_loss_weight": config.losses.mtp,
                    "mtp_source_role": "backbone" if config.losses.mtp > 0 else None,
                    "mtp_trainable": False,
                    "activation_checkpointing": config.runtime.activation_checkpointing,
                    "activation_checkpoint_layer_count": (
                        config.runtime.activation_checkpoint_layer_count
                    ),
                    "hidden_alignment_activation_checkpoint_layer_count": (
                        config.runtime.hidden_alignment_activation_checkpoint_layer_count
                    ),
                    "activation_checkpoint_layer_indices": list(
                        report.activation_checkpoint_layer_indices
                    ),
                    "hidden_alignment_activation_checkpoint_layer_indices": list(
                        report.hidden_alignment_activation_checkpoint_layer_indices
                    ),
                    "teacher_cpu_offload": config.runtime.teacher_cpu_offload,
                    "teacher_cpu_shadow_bytes": report.teacher_cpu_shadow_bytes,
                    "teacher_gpu_stage_bytes": report.teacher_gpu_stage_bytes,
                    "activation_checkpointing_on_alignment_only": (
                        config.runtime.activation_checkpointing_on_alignment_only
                    ),
                    "dense_transfer_execution": config.runtime.dense_transfer_execution,
                    "dense_transfer_token_checkpoint": (
                        config.runtime.dense_transfer_token_checkpoint
                    ),
                    "dense_transfer_checkpoint_layer_count": (
                        config.runtime.dense_transfer_checkpoint_layer_count
                    ),
                    "hidden_alignment_dense_transfer_checkpoint_layer_count": (
                        config.runtime.hidden_alignment_dense_transfer_checkpoint_layer_count
                    ),
                    "dense_transfer_token_checkpoint_layer_indices": list(
                        report.dense_transfer_token_checkpoint_layer_indices
                    ),
                    "hidden_alignment_dense_transfer_token_checkpoint_layer_indices": list(
                        report.hidden_alignment_dense_transfer_token_checkpoint_layer_indices
                    ),
                    "allow_tf32": config.runtime.allow_tf32,
                    "fused_adamw_requested": config.runtime.fused_adamw,
                    "sharding": config.runtime.sharding,
                    "micro_batch_size": config.data.micro_batch_size,
                    "global_batch_tokens": report.batch.global_batch_tokens,
                    "micro_batch_tokens_per_rank": report.batch.micro_batch_tokens_per_rank,
                    "gradient_accumulation_steps": report.batch.gradient_accumulation_steps,
                    "data_prefetch_depth": config.data.num_workers,
                    "allocator_config": os.environ.get("PYTORCH_ALLOC_CONF"),
                },
            )
        resolved_path = run_dir / "resolved_config.yaml"
        resolved_error: str | None = None
        if context.is_rank_zero:
            try:
                if resolved_path.exists():
                    from ..config import load_train_config

                    previous = load_train_config(resolved_path)
                    if previous.fingerprint() != config.fingerprint():
                        raise RuntimeError(
                            "run directory resolved_config.yaml has a different critical fingerprint"
                        )
                else:
                    dump_resolved_config(config, resolved_path)
            except Exception as exc:
                resolved_error = f"{type(exc).__name__}: {exc}"
        if context.world_size > 1:
            import torch.distributed as dist

            payload = [resolved_error]
            dist.broadcast_object_list(payload, src=0)
            resolved_error = payload[0]
        if resolved_error is not None:
            raise RuntimeError(f"resolved configuration setup failed: {resolved_error}")

        dtype = torch.bfloat16 if config.runtime.bf16 else torch.float32
        build_device = (
            "cpu"
            if context.world_size > 1 and config.runtime.sharding == "fsdp2"
            else str(context.device)
        )
        built, teacher = _build_transfer_and_teacher(
            config,
            context,
            dtype=dtype,
            build_device=build_device,
        )
        teacher_offload = (
            TeacherCPUOffloadManager.from_transfer_modules(
                teacher,
                built.transfer_modules,
                target_device=context.device,
            )
            if config.runtime.teacher_cpu_offload and teacher is not None
            else None
        )
        if event_logger is not None:
            event_logger.log(
                "model_built",
                {
                    "donor_teacher_shared": built.donor_teacher_shared,
                    "hidden_teacher_enabled": teacher is not None,
                    "teacher_cpu_offload": teacher_offload is not None,
                    "teacher_cpu_shadow_bytes": (
                        teacher_offload.staged_bytes if teacher_offload is not None else 0
                    ),
                    "teacher_gpu_stage_bytes": (
                        teacher_offload.staged_bytes if teacher_offload is not None else 0
                    ),
                    "active_student_layers": len(built.student_layer_indices),
                    "mtp_enabled": built.mtp is not None,
                    "mtp_parameters": (
                        sum(int(parameter.numel()) for parameter in built.mtp.parameters())
                        if built.mtp is not None
                        else 0
                    ),
                    "mtp_trainable_parameters": (
                        sum(
                            int(parameter.numel())
                            for parameter in built.mtp.parameters()
                            if parameter.requires_grad
                        )
                        if built.mtp is not None
                        else 0
                    ),
                    "dense_transfer_execution": config.runtime.dense_transfer_execution,
                },
            )
        raw_model = built.model
        raw_model.config.use_cache = False
        activation_checkpoint_layer_indices_enabled = (
            _effective_activation_checkpoint_layer_indices(
                config,
                align_hidden=False,
            )
        )
        if (
            activation_checkpoint_layer_indices_enabled
            != report.activation_checkpoint_layer_indices
        ):
            raise RuntimeError("ordinary outer checkpoint policy changed after preflight")
        _set_selective_activation_checkpointing(
            raw_model,
            activation_checkpoint_layer_indices_enabled,
        )
        desired_transfer_checkpoint_layer_indices = (
            _effective_dense_transfer_checkpoint_layer_indices(
                config,
                align_hidden=False,
                outer_checkpoint_layer_indices=(activation_checkpoint_layer_indices_enabled),
            )
        )
        transfer_token_checkpoint_layer_indices_enabled = _set_dense_transfer_token_checkpointing(
            built.transfer_modules,
            built.student_layer_indices,
            outer_checkpoint_layer_indices=activation_checkpoint_layer_indices_enabled,
            enabled=config.runtime.dense_transfer_token_checkpoint,
            checkpoint_layer_indices=desired_transfer_checkpoint_layer_indices,
        )
        if (
            transfer_token_checkpoint_layer_indices_enabled
            != report.dense_transfer_token_checkpoint_layer_indices
        ):
            raise RuntimeError("ordinary dense transfer checkpoint policy changed after preflight")
        loss_model = StreamingLossCausalLM(
            raw_model,
            chunk_tokens=config.runtime.loss_chunk_tokens,
            checkpoint_chunks=config.runtime.loss_checkpoint_chunks,
            compile_loss=config.runtime.compile_streaming_loss,
            mtp=built.mtp,
            teacher_kd_enabled=_teacher_kd_data_enabled(config),
        )
        train_model = wrap_distributed(
            loss_model,
            context,
            config.runtime.sharding,
            transformer_model=raw_model,
        )
        # The source loader freezes parameters and leaves the model in eval mode.
        # Training mode is still required for HF activation checkpointing and the
        # (currently zero) train-time dropout behavior of trainable modules.
        train_model.train()
        optimizer = _build_optimizer(config, built)
        if event_logger is not None:
            optimizer_components = getattr(optimizer, "optimizers", (optimizer,))
            event_logger.log(
                "optimizer_built",
                {
                    "optimizer": type(optimizer).__name__,
                    "adapter_optimizer": config.optimizer.adapter_optimizer,
                    "optimizer_bundle": isinstance(optimizer, OptimizerBundle),
                    "fused": bool(optimizer.defaults.get("fused", False)),
                    "components": [
                        {
                            "optimizer": type(component).__name__,
                            "fused": bool(component.defaults.get("fused", False)),
                            "parameter_groups": [
                                str(group.get("name", "unnamed"))
                                for group in component.param_groups
                            ],
                        }
                        for component in optimizer_components
                    ],
                    "parameter_groups": len(optimizer.param_groups),
                    "learning_rates": [
                        {
                            "name": name,
                            "nominal": nominal,
                            "adjusted_update_coefficient": adjusted,
                            "adjustment_factor": factor,
                            "adjust_lr_fn": next(
                                (
                                    group.get("adjust_lr_fn")
                                    for group in optimizer.param_groups
                                    if str(group.get("name", "unnamed")) == name
                                ),
                                None,
                            ),
                        }
                        for name, nominal, adjusted, factor in (
                            _named_adjusted_learning_rates(optimizer)
                        )
                    ],
                    "trainable_parameters": sum(
                        int(parameter.numel())
                        for group in optimizer.param_groups
                        for parameter in group["params"]
                    ),
                },
            )
        scheduler = TokenLRScheduler(
            optimizer,
            warmup_tokens=config.optimizer.warmup_tokens,
            max_tokens=config.optimizer.max_tokens,
            lr_schedule=config.optimizer.lr_schedule,
            min_lr_ratio=config.optimizer.min_lr_ratio,
            decay_tokens=config.optimizer.decay_tokens,
        )
        stateful = {
            "model": TrainableModelState(raw_model),
            "optimizer": OptimizerState(raw_model, optimizer),
            "scheduler": scheduler,
        }
        store = _build_training_record_store(
            config,
            verify_shards=False,  # Full hash validation already ran in preflight.
        )
        if config.data.quality_cooldown_enabled():
            cooldown_kd_path = config.data.quality_cooldown_teacher_kd_manifest_path
            assert cooldown_kd_path is not None
            cooldown_store = _build_training_record_store(
                config,
                manifest_path=cooldown_kd_path,
                verify_shards=False,  # Full hash validation already ran in preflight.
            )
        layer_mapping = load_layer_mapping(
            config.architecture.layer_map_path, config.architecture.student_layers
        )

        stop_file = run_dir / config.checkpoint.stop_file if context.is_rank_zero else None
        controller = SignalController(stop_file)
        sparse_schedule = SparseTopKSchedule(
            config.architecture.num_experts, config.architecture.top_k
        )
        global_batch_samples = config.data.global_batch_tokens // config.data.max_sequence_length
        last_checkpoint_time = time.monotonic()
        profiler = _build_profiler(config, run_dir, context)

        with manager, controller, profiler:
            state, cursor, loaded_checkpoint = _load_or_initialize(
                manager,
                stateful,
                config,
                report,
                store,
                cooldown_store=cooldown_store,
                resume=resume,
                fork_from=fork_from,
            )
            if metric_logger is not None:
                metric_logger.reconcile(state.global_step)
            if telemetry_logger is not None:
                telemetry_logger.reconcile(state.global_step)
            if context.is_rank_zero:
                progress_ui = TrainingProgress(
                    total_tokens=config.optimizer.max_tokens,
                    initial_tokens=min(state.committed_tokens, config.optimizer.max_tokens),
                    mode=progress,
                    description=f"{config.run_id}:{config.stage}",
                )
            if event_logger is not None:
                event_logger.log(
                    "resume" if loaded_checkpoint is not None else "initialized",
                    {
                        "step": state.global_step,
                        "tokens": state.committed_tokens,
                        "checkpoint": (
                            str(loaded_checkpoint.path) if loaded_checkpoint is not None else None
                        ),
                        "checkpoint_kind": (
                            loaded_checkpoint.metadata.get("kind")
                            if loaded_checkpoint is not None
                            else None
                        ),
                        "checkpoint_tag": (
                            loaded_checkpoint.metadata.get("tag")
                            if loaded_checkpoint is not None
                            else None
                        ),
                        "saved_world_size": (
                            loaded_checkpoint.metadata.get("saved_world_size")
                            if loaded_checkpoint is not None
                            else None
                        ),
                        "current_world_size": context.world_size,
                        "data_phase": (
                            cursor.active_phase
                            if isinstance(cursor, DeterministicCooldownCursor)
                            else "primary"
                        ),
                        "quality_cooldown_start_tokens": (
                            config.data.quality_cooldown_start_tokens
                        ),
                        "rng_forked": (
                            loaded_checkpoint.rng_forked if loaded_checkpoint is not None else False
                        ),
                        "fork_from": fork_from,
                        "source_mix": (
                            {
                                **_source_mix_log_contract(report),
                                "committed_samples_by_source": (
                                    cursor.committed_samples_by_source
                                ),
                                "committed_tokens_by_source": (
                                    cursor.committed_tokens_by_source
                                ),
                                "critical_lineage_fingerprint": (
                                    cursor.critical_lineage_fingerprint
                                ),
                            }
                            if isinstance(cursor, DeterministicSourceMixCursor)
                            else None
                        ),
                    },
                )
            if _is_read_only_completed_resume(
                state,
                loaded_checkpoint,
                max_tokens=config.optimizer.max_tokens,
            ):
                _rank_zero_print(
                    context,
                    {
                        "event": "train_already_complete",
                        "checkpoint": str(loaded_checkpoint.path),
                        "step": state.global_step,
                        "tokens": state.committed_tokens,
                    },
                )
                if event_logger is not None:
                    event_logger.log(
                        "train_already_complete",
                        {
                            "checkpoint": str(loaded_checkpoint.path),
                            "step": state.global_step,
                            "tokens": state.committed_tokens,
                        },
                    )
                session_status = "already_complete"
                return 0
            scheduler.step_tokens(state.committed_tokens)
            train_start_fields = {
                "run_id": config.run_id,
                "stage": config.stage,
                "step": state.global_step,
                "tokens": state.committed_tokens,
                "world_size": context.world_size,
                "data_phase": (
                    cursor.active_phase
                    if isinstance(cursor, DeterministicCooldownCursor)
                    else "primary"
                ),
                "quality_cooldown_start_tokens": (config.data.quality_cooldown_start_tokens),
            }
            if event_logger is not None:
                event_logger.log("train_start", train_start_fields)
            _rank_zero_print(context, {"event": "train_start", **train_start_fields})
            telemetry_tracker = TrainingTelemetryTracker(
                total_tokens=config.optimizer.max_tokens,
                initial_tokens=min(state.committed_tokens, config.optimizer.max_tokens),
            )
            while state.committed_tokens < config.optimizer.max_tokens:
                (
                    checkpoint_requested,
                    stop_requested,
                    generation,
                    reason,
                ) = _coordinate_control(controller, context)
                if checkpoint_requested:
                    request_sequence = int(state.extra.get("checkpoint_request_sequence", 0)) + 1
                    state.extra["checkpoint_request_sequence"] = request_sequence
                    _checkpoint(
                        manager,
                        stateful,
                        state,
                        cursor,
                        config,
                        report,
                        kind="interrupt",
                        boundary=None,
                        tag=f"request-{request_sequence:06d}",
                        reason=reason or ("remote-stop" if stop_requested else "remote-checkpoint"),
                        event_logger=event_logger,
                    )
                    last_checkpoint_time = time.monotonic()
                    controller.acknowledge_checkpoint(generation)
                    if stop_requested:
                        if event_logger is not None:
                            event_logger.log(
                                "graceful_stop",
                                {"step": state.global_step, "tokens": state.committed_tokens},
                            )
                        _rank_zero_print(
                            context,
                            {
                                "event": "graceful_stop",
                                "step": state.global_step,
                                "tokens": state.committed_tokens,
                            },
                        )
                        session_status = "stopped"
                        return 0
                if config.stage == "sparse":
                    active_top_k = sparse_schedule.value(
                        state.committed_tokens, config.optimizer.max_tokens
                    )
                    _set_sparse_top_k(built.transfer_modules, active_top_k)
                    state.top_k = active_top_k
                step_started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(context.device)
                state.micro_step_in_accumulation = 0
                step_data_phase = (
                    cursor.active_phase
                    if isinstance(cursor, DeterministicCooldownCursor)
                    else "primary"
                )
                active_store = cooldown_store if step_data_phase == "cooldown" else store
                if active_store is None:
                    raise RuntimeError(
                        f"training data store is missing for phase {step_data_phase}"
                    )
                boundary = capture_committed_boundary(state, _runtime_cursor(cursor))
                rank_references = cursor.plan_rank_batch(
                    global_batch_samples,
                    rank=context.rank,
                    world_size=context.world_size,
                )
                optimizer.zero_grad(set_to_none=True)
                local_valid_tokens = torch.zeros((), device=context.device, dtype=torch.int64)
                step_metrics: dict[str, Any] = {}
                replay_step = False
                stop_after_checkpoint = False
                # Select whole global optimizer batches from the durable step
                # index. Using local micro-step geometry would change this
                # curriculum after a world-size resume because accumulation is
                # automatically adjusted.
                record_dense = config.stage == "sparse" and _fraction_selected(
                    state.global_step,
                    config.losses.dense_oracle_batch_fraction,
                )
                align_hidden = teacher is not None and _fraction_selected(
                    state.global_step,
                    config.losses.hidden_alignment_batch_fraction,
                )
                desired_activation_checkpoint_layer_indices = (
                    _effective_activation_checkpoint_layer_indices(
                        config,
                        align_hidden=align_hidden,
                    )
                )
                desired_transfer_checkpoint_layer_indices = (
                    _effective_dense_transfer_checkpoint_layer_indices(
                        config,
                        align_hidden=align_hidden,
                        outer_checkpoint_layer_indices=(
                            desired_activation_checkpoint_layer_indices
                        ),
                    )
                )
                expected_outer = (
                    report.hidden_alignment_activation_checkpoint_layer_indices
                    if align_hidden
                    else report.activation_checkpoint_layer_indices
                )
                expected_inner = (
                    report.hidden_alignment_dense_transfer_token_checkpoint_layer_indices
                    if align_hidden
                    else report.dense_transfer_token_checkpoint_layer_indices
                )
                if desired_activation_checkpoint_layer_indices != expected_outer:
                    raise RuntimeError(
                        "outer checkpoint policy changed after preflight: "
                        f"expected={expected_outer}, "
                        f"actual={desired_activation_checkpoint_layer_indices}"
                    )
                if desired_transfer_checkpoint_layer_indices != expected_inner:
                    raise RuntimeError(
                        "dense transfer checkpoint policy changed after preflight: "
                        f"expected={expected_inner}, "
                        f"actual={desired_transfer_checkpoint_layer_indices}"
                    )
                if _checkpoint_policy_requires_update(
                    activation_checkpoint_layer_indices_enabled,
                    transfer_token_checkpoint_layer_indices_enabled,
                    desired_activation_checkpoint_layer_indices,
                    desired_transfer_checkpoint_layer_indices,
                ):
                    if (
                        desired_activation_checkpoint_layer_indices
                        != activation_checkpoint_layer_indices_enabled
                    ):
                        _set_selective_activation_checkpointing(
                            raw_model,
                            desired_activation_checkpoint_layer_indices,
                        )
                    activation_checkpoint_layer_indices_enabled = (
                        desired_activation_checkpoint_layer_indices
                    )
                    transfer_token_checkpoint_layer_indices_enabled = (
                        _set_dense_transfer_token_checkpointing(
                            built.transfer_modules,
                            built.student_layer_indices,
                            outer_checkpoint_layer_indices=(
                                desired_activation_checkpoint_layer_indices
                            ),
                            enabled=config.runtime.dense_transfer_token_checkpoint,
                            checkpoint_layer_indices=(desired_transfer_checkpoint_layer_indices),
                        )
                    )
                    if transfer_token_checkpoint_layer_indices_enabled != expected_inner:
                        raise RuntimeError(
                            "dense transfer checkpoint policy changed after preflight"
                        )
                teacher_stage_transition = None
                teacher_restore_transition = None
                reference_batches = tuple(
                    rank_references[
                        micro_step * config.data.micro_batch_size : (micro_step + 1)
                        * config.data.micro_batch_size
                    ]
                    for micro_step in range(report.batch.gradient_accumulation_steps)
                )
                data_token_count_started = time.perf_counter()
                source_mix_commit: _SourceMixCommitPayload | None = None
                with torch.profiler.record_function("twen/data_token_counts"):
                    local_target_count, local_hidden_count = (
                        active_store.optimizer_batch_token_counts(rank_references)
                    )
                    local_mtp_count = (
                        active_store.optimizer_batch_mtp_token_count(rank_references)
                        if config.losses.mtp > 0
                        else 0
                    )
                    if isinstance(cursor, DeterministicSourceMixCursor):
                        source_mix_commit = _prepare_source_mix_commit(
                            cursor,
                            rank_references,
                            active_store.optimizer_batch_valid_token_counts(
                                rank_references
                            ),
                            context,
                        )
                data_token_count_seconds = time.perf_counter() - data_token_count_started
                global_loss_counts = torch.tensor(
                    [local_target_count, local_hidden_count, local_mtp_count],
                    device=context.device,
                    dtype=torch.int64,
                )
                all_reduce_sum(global_loss_counts, context)
                if bool((global_loss_counts[:2] <= 0).any().item()):
                    raise RuntimeError(
                        "optimizer batch has no next-token targets or valid hidden tokens"
                    )
                if config.losses.mtp > 0 and int(global_loss_counts[2].item()) <= 0:
                    raise RuntimeError("optimizer batch has no valid native MTP L-2 targets")
                global_target_count = global_loss_counts[0]
                global_hidden_count = global_loss_counts[1]
                global_mtp_count = global_loss_counts[2]
                prefetched_batches = active_store.iter_prefetched_batches(
                    reference_batches,
                    prefetch_depth=config.data.num_workers,
                    pin_memory=True,
                )
                data_prefetch_wait_seconds = 0.0
                if teacher_offload is not None:
                    if align_hidden:
                        teacher_stage_transition = teacher_offload.stage()
                    elif teacher_offload.is_staged:
                        raise RuntimeError(
                            "teacher offload remained staged on a non-alignment step"
                        )
                for micro_step in range(report.batch.gradient_accumulation_steps):
                    data_wait_started = time.perf_counter()
                    with torch.profiler.record_function("twen/data_prefetch_wait"):
                        try:
                            host_batch = next(prefetched_batches)
                        except StopIteration as error:
                            raise RuntimeError(
                                f"{_training_data_mode(config)} prefetch ended before the optimizer "
                                "batch was complete"
                            ) from error
                    data_prefetch_wait_seconds += time.perf_counter() - data_wait_started
                    with torch.profiler.record_function("twen/h2d"):
                        batch = _move_training_batch(config, host_batch, context.device)
                    batch_target_count, batch_hidden_count = _batch_loss_token_counts(batch)
                    batch_mtp_count = (
                        _batch_mtp_loss_token_count(batch)
                        if config.losses.mtp > 0
                        else batch_target_count.new_zeros(())
                    )
                    _set_record_aux(built.transfer_modules, record_dense)
                    anchor_hidden_states = None
                    if config.losses.anchor_kl > 0:
                        with torch.profiler.record_function("twen/anchor_forward"):
                            anchor_hidden_states = _anchor_hidden_states(
                                train_model,
                                built.transfer_modules,
                                batch,
                                dtype=dtype,
                                enabled=config.runtime.bf16,
                            )
                    should_sync = micro_step + 1 == report.batch.gradient_accumulation_steps
                    with accumulation_sync(
                        train_model, should_sync=should_sync, sharding=config.runtime.sharding
                    ):
                        with torch.autocast(
                            device_type="cuda", dtype=dtype, enabled=config.runtime.bf16
                        ):
                            with torch.profiler.record_function("twen/forward"):
                                outputs = _student_language_model_forward(
                                    train_model,
                                    batch,
                                    teacher_kd_enabled=teacher_kd_enabled,
                                    anchor_hidden_states=anchor_hidden_states,
                                    output_hidden_states=align_hidden,
                                )
                            with torch.profiler.record_function("twen/loss"):
                                ntp = outputs["ntp"]
                                kd = outputs["teacher_kd"]
                                target_mean = config.losses.ntp * ntp
                                if teacher_kd_enabled:
                                    if kd is None:
                                        raise RuntimeError(
                                            "teacher-kd mode omitted the streaming KD loss"
                                        )
                                    target_mean = (
                                        target_mean + config.losses.teacher_kd * kd
                                    )
                                elif kd is not None:
                                    raise RuntimeError(
                                        "prepared-text mode unexpectedly produced a teacher KD loss"
                                    )
                                if anchor_hidden_states is not None:
                                    anchor = outputs["anchor_kl"]
                                    if anchor is None:
                                        raise RuntimeError(
                                            "streaming student forward omitted requested anchor KL"
                                        )
                                    target_mean = target_mean + config.losses.anchor_kl * anchor
                                else:
                                    anchor = ntp.new_zeros(())
                                loss = _token_mean_contribution(
                                    target_mean,
                                    batch_target_count,
                                    global_target_count,
                                    world_size=context.world_size,
                                )
                                if config.losses.mtp > 0:
                                    mtp = outputs["mtp"]
                                    if mtp is None:
                                        raise RuntimeError(
                                            "MTP loss is enabled but the student graph omitted it"
                                        )
                                    loss = loss + _token_mean_contribution(
                                        config.losses.mtp * mtp,
                                        batch_mtp_count,
                                        global_mtp_count,
                                        world_size=context.world_size,
                                    )
                                else:
                                    mtp = None
                                router_aux = ntp.new_zeros(())
                                router_metrics: dict[str, Any] = {}
                                if config.stage == "sparse":
                                    router_aux, router_metrics = _router_auxiliary_loss(
                                        config,
                                        built.transfer_modules,
                                        dense=record_dense,
                                        mask=batch.attention_mask,
                                    )
                                    loss = loss + (
                                        router_aux / report.batch.gradient_accumulation_steps
                                    )
                                if align_hidden:
                                    assert teacher is not None
                                    with (
                                        torch.no_grad(),
                                        torch.profiler.record_function("twen/teacher_forward"),
                                    ):
                                        teacher_outputs = teacher(
                                            input_ids=batch.input_ids,
                                            attention_mask=batch.attention_mask,
                                            use_cache=False,
                                            output_hidden_states=True,
                                        )
                                    hidden_alignment = _hidden_alignment_loss(
                                        outputs["hidden_states"],
                                        teacher_outputs.hidden_states,
                                        built.transfer_modules,
                                        layer_mapping,
                                        built.student_layer_indices,
                                        batch.attention_mask,
                                    )
                                    loss = loss + _token_mean_contribution(
                                        config.losses.hidden_alignment * hidden_alignment,
                                        batch_hidden_count,
                                        global_hidden_count,
                                        world_size=context.world_size,
                                    )
                                else:
                                    hidden_alignment = ntp.new_zeros(())
                                scaled_loss = loss
                        with torch.profiler.record_function("twen/backward"):
                            scaled_loss.backward()

                    state.micro_step_in_accumulation = micro_step + 1
                    local_valid_tokens.add_(batch.attention_mask.sum(dtype=torch.int64))
                    target_weight = batch_target_count.to(dtype=ntp.dtype)
                    hidden_weight = batch_hidden_count.to(dtype=ntp.dtype)
                    values = {
                        "__target_count": batch_target_count.detach(),
                        "__hidden_count": (
                            batch_hidden_count.detach()
                            if align_hidden
                            else batch_hidden_count.new_zeros(())
                        ),
                        "ntp": ntp.detach() * target_weight,
                        **router_metrics,
                    }
                    if teacher_kd_enabled:
                        assert kd is not None
                        values["teacher_kd"] = kd.detach() * target_weight
                        values["anchor_kl"] = anchor.detach() * target_weight
                        values["hidden_alignment"] = (
                            hidden_alignment.detach() * hidden_weight
                        )
                    if mtp is not None:
                        mtp_weight = batch_mtp_count.to(dtype=mtp.dtype)
                        values["__mtp_count"] = batch_mtp_count.detach()
                        values["mtp"] = mtp.detach() * mtp_weight
                    for key, value in values.items():
                        if key in step_metrics:
                            step_metrics[key] = step_metrics[key] + value
                        else:
                            step_metrics[key] = value

                    # Loop locals remain strongly referenced across iterations, and Python
                    # evaluates the next forward's RHS before replacing them.  Release the
                    # large teacher/student hidden-state tuples only after backward and
                    # detached metric accumulation, but before interrupt checkpointing or
                    # the next microbatch.  The CUDA allocator can reuse these blocks without
                    # an empty-cache call or synchronization.
                    del outputs
                    del anchor_hidden_states
                    if align_hidden:
                        del teacher_outputs
                    del scaled_loss, loss, target_mean
                    del ntp, kd, anchor, hidden_alignment, router_aux, mtp
                    del values, router_metrics
                    del batch, host_batch

                    checkpoint_requested, stop_requested, generation, reason = _coordinate_control(
                        controller, context
                    )
                    if checkpoint_requested:
                        if teacher_offload is not None and teacher_offload.is_staged:
                            teacher_restore_transition = teacher_offload.restore()
                        optimizer.zero_grad(set_to_none=True)
                        request_sequence = (
                            int(state.extra.get("checkpoint_request_sequence", 0)) + 1
                        )
                        state.extra["checkpoint_request_sequence"] = request_sequence
                        boundary.trainer_state.extra["checkpoint_request_sequence"] = (
                            request_sequence
                        )
                        _checkpoint(
                            manager,
                            stateful,
                            state,
                            cursor,
                            config,
                            report,
                            kind="interrupt",
                            boundary=boundary,
                            tag=f"request-{request_sequence:06d}",
                            reason=reason
                            or ("remote-stop" if stop_requested else "remote-checkpoint"),
                            event_logger=event_logger,
                        )
                        last_checkpoint_time = time.monotonic()
                        controller.acknowledge_checkpoint(generation)
                        if stop_requested:
                            stop_after_checkpoint = True
                        else:
                            boundary.rng_state.restore(strict_cuda=False)
                            state = boundary.trainer_state.clone()
                            replay_step = True
                        # Profiler schedule units are microbatches, including one
                        # that ends at an interruption-safe checkpoint boundary.
                        _advance_profiler_after_microbatch(profiler)
                        break
                    for module in built.transfer_modules:
                        module.clear_aux()
                    _advance_profiler_after_microbatch(profiler)

                if teacher_offload is not None and teacher_offload.is_staged:
                    teacher_restore_transition = teacher_offload.restore()
                if teacher_stage_transition is not None:
                    if teacher_restore_transition is None:
                        raise RuntimeError("staged teacher was not restored after optimizer batch")
                    if event_logger is not None:
                        event_logger.log(
                            "teacher_cpu_offload_step",
                            {
                                "step": state.global_step,
                                "completed_microbatches": state.micro_step_in_accumulation,
                                "stage": dataclasses.asdict(teacher_stage_transition),
                                "restore": dataclasses.asdict(teacher_restore_transition),
                                "replay_requested": replay_step,
                                "stop_requested": stop_after_checkpoint,
                            },
                        )

                # A signal may break the loop with queued host batches. Cancel
                # those deterministic, uncommitted reads before replay/exit.
                prefetched_batches.close()
                prefetched_batches = None

                if stop_after_checkpoint:
                    if event_logger is not None:
                        event_logger.log(
                            "graceful_stop",
                            {"step": state.global_step, "tokens": state.committed_tokens},
                        )
                    _rank_zero_print(
                        context,
                        {
                            "event": "graceful_stop",
                            "step": state.global_step,
                            "tokens": state.committed_tokens,
                        },
                    )
                    session_status = "stopped"
                    return 0
                if replay_step:
                    continue

                token_tensor = local_valid_tokens
                all_reduce_sum(token_tensor, context)
                committed_tokens = int(token_tensor.item())
                if source_mix_commit is not None:
                    if not isinstance(cursor, DeterministicSourceMixCursor):
                        raise RuntimeError(
                            "source-mix commit payload was built for a legacy cursor"
                        )
                    if committed_tokens != source_mix_commit.token_count:
                        raise RuntimeError(
                            "loaded optimizer-batch valid tokens differ from the "
                            "authenticated source-mix reference counts"
                        )
                    # Revalidate immediately before mutation.  The earlier
                    # cross-rank validation happened before any forward work;
                    # this also proves the pending cursor plan was not changed
                    # while the optimizer batch was executing.
                    source_mix_commit.validate(cursor)
                with torch.profiler.record_function("twen/grad_clip"):
                    grad_norm = _clip_optimizer_gradients(
                        optimizer,
                        config.optimizer.grad_clip_norm,
                    )
                # Capture the rates that are actually applied by this update.
                # The token scheduler is advanced after the commit so the live
                # param-group values then belong to the *next* optimizer step.
                applied_learning_rates = _named_learning_rates(optimizer)
                applied_adjusted_learning_rates = _named_adjusted_learning_rates(
                    optimizer
                )
                with torch.profiler.record_function("twen/optimizer_step"):
                    _optimizer_step_and_commit(
                        optimizer,
                        cursor,
                        global_batch_samples=global_batch_samples,
                        committed_tokens=committed_tokens,
                        source_mix_commit=source_mix_commit,
                    )
                optimizer.zero_grad(set_to_none=True)
                state.global_step += 1
                state.committed_tokens = cursor.committed_tokens
                state.micro_step_in_accumulation = 0
                next_data_phase = (
                    cursor.active_phase
                    if isinstance(cursor, DeterministicCooldownCursor)
                    else "primary"
                )
                state.extra["data_phase"] = next_data_phase
                if next_data_phase != step_data_phase and event_logger is not None:
                    event_logger.log(
                        "quality_cooldown_started",
                        {
                            "step": state.global_step,
                            "tokens": state.committed_tokens,
                            "configured_start_tokens": (config.data.quality_cooldown_start_tokens),
                            "last_primary_batch_started_tokens": (
                                boundary.trainer_state.committed_tokens
                            ),
                        },
                    )
                state.curriculum_position = min(
                    state.committed_tokens / config.optimizer.max_tokens, 1.0
                )
                scheduler.step_tokens(state.committed_tokens)
                next_learning_rates = _named_learning_rates(optimizer)
                next_adjusted_learning_rates = _named_adjusted_learning_rates(
                    optimizer
                )
                metric_keys = sorted(step_metrics)
                if metric_keys:
                    metric_tensor = torch.stack(
                        [step_metrics[key].to(dtype=torch.float64) for key in metric_keys]
                    )
                    all_reduce_sum(metric_tensor, context)
                    step_metrics = {
                        key: float(value)
                        for key, value in zip(
                            metric_keys, metric_tensor.cpu().tolist(), strict=True
                        )
                    }
                reduced_target_count = step_metrics.pop("__target_count", 0.0)
                reduced_hidden_count = step_metrics.pop("__hidden_count", 0.0)
                reduced_mtp_count = step_metrics.pop("__mtp_count", 0.0)
                if reduced_target_count <= 0:
                    raise RuntimeError("committed optimizer batch has no next-token targets")
                averaged = {"ntp": step_metrics["ntp"] / reduced_target_count}
                if teacher_kd_enabled:
                    averaged.update(
                        {
                            key: step_metrics[key] / reduced_target_count
                            for key in ("teacher_kd", "anchor_kl")
                        }
                    )
                    averaged["hidden_alignment"] = (
                        step_metrics["hidden_alignment"] / reduced_hidden_count
                        if reduced_hidden_count > 0
                        else 0.0
                    )
                if config.losses.mtp > 0:
                    if reduced_mtp_count <= 0:
                        raise RuntimeError("committed optimizer batch has no native MTP targets")
                    averaged["mtp"] = step_metrics["mtp"] / reduced_mtp_count
                auxiliary_denominator = (
                    context.world_size * report.batch.gradient_accumulation_steps
                )
                for key, value in step_metrics.items():
                    if key not in {
                        "ntp",
                        "teacher_kd",
                        "anchor_kl",
                        "hidden_alignment",
                        "mtp",
                    }:
                        averaged[key] = value / auxiliary_denominator
                averaged["loss"] = (
                    config.losses.ntp * averaged["ntp"]
                    + config.losses.teacher_kd * averaged.get("teacher_kd", 0.0)
                    + config.losses.anchor_kl * averaged.get("anchor_kl", 0.0)
                    + config.losses.hidden_alignment
                    * averaged.get("hidden_alignment", 0.0)
                    + config.losses.mtp * averaged.get("mtp", 0.0)
                )
                if config.stage == "sparse":
                    averaged["loss"] += (
                        config.losses.router_z * averaged.get("router_z", 0.0)
                        + config.losses.load_balance * averaged.get("load_balance", 0.0)
                        + config.losses.dense_oracle * averaged.get("dense_oracle", 0.0)
                        + config.losses.router_supervision * averaged.get("router_supervision", 0.0)
                    )
                averaged.update(
                    {
                        "tokens": state.committed_tokens,
                        "tokens_this_step": committed_tokens,
                        "lr": applied_learning_rates[0][1],
                        "next_lr": next_learning_rates[0][1],
                        "top_k": state.top_k,
                        "grad_norm": float(grad_norm.detach().cpu()),
                        "data_mode": _training_data_mode(config),
                        "data_phase": step_data_phase,
                        "next_data_phase": next_data_phase,
                    }
                )
                averaged.update(
                    _checkpoint_phase_log_fields(
                        config,
                        align_hidden=align_hidden,
                        outer_checkpoint_layer_indices=(
                            activation_checkpoint_layer_indices_enabled
                        ),
                        inner_checkpoint_layer_indices=(
                            transfer_token_checkpoint_layer_indices_enabled
                        ),
                    )
                )
                if config.losses.mtp > 0:
                    averaged["mtp_target_tokens_this_step"] = int(reduced_mtp_count)
                # Keep established metric names while adding explicit loss aliases
                # for terminal rendering and downstream report tooling.
                averaged.update(
                    _loss_metric_aliases(
                        averaged,
                        include_anchor=config.losses.anchor_kl > 0,
                        include_hidden=align_hidden,
                        include_sparse=config.stage == "sparse",
                        include_dense=record_dense,
                        include_mtp=config.losses.mtp > 0,
                    )
                )
                averaged.update(
                    _learning_rate_step_metrics(
                        applied_learning_rates,
                        next_learning_rates,
                        applied_adjusted_learning_rates,
                        next_adjusted_learning_rates,
                    )
                )
                if source_mix_commit is not None:
                    assert isinstance(cursor, DeterministicSourceMixCursor)
                    for source_id, tokens in (
                        source_mix_commit.valid_tokens_by_source.items()
                    ):
                        averaged[f"source_tokens_this_step/{source_id}"] = tokens
                    for source_id, tokens in (
                        cursor.committed_tokens_by_source.items()
                    ):
                        averaged[f"source_tokens/{source_id}"] = tokens
                state.router_stats = {
                    key: float(value)
                    for key, value in averaged.items()
                    if key
                    in {
                        "router_z",
                        "load_balance",
                        "dense_oracle",
                        "router_supervision",
                        "router_entropy",
                    }
                    or key.startswith("expert_usage_")
                }
                state.extra["last_committed_metrics"] = {
                    key: float(value)
                    for key, value in averaged.items()
                    if isinstance(value, (int, float))
                }
                compute_step_seconds = time.perf_counter() - step_started
                memory_metrics = _cuda_memory_metrics(context.device)
                if metric_logger is not None:
                    metric_logger.log(state.global_step, averaged)
                due_step = state.global_step % config.checkpoint.every_steps == 0
                due_time = (
                    time.monotonic() - last_checkpoint_time >= config.checkpoint.every_minutes * 60
                )
                if due_step or due_time:
                    _checkpoint(
                        manager,
                        stateful,
                        state,
                        cursor,
                        config,
                        report,
                        kind="periodic",
                        boundary=None,
                        event_logger=event_logger,
                    )
                    last_checkpoint_time = time.monotonic()
                telemetry = telemetry_tracker.observe(
                    state.committed_tokens,
                    compute_step_seconds,
                )
                telemetry.update(memory_metrics)
                telemetry.update(
                    {
                        "timestamp_utc": utc_now(),
                        "tokens": state.committed_tokens,
                        "step": state.global_step,
                        "data_token_count_seconds": data_token_count_seconds,
                        "data_prefetch_wait_seconds": data_prefetch_wait_seconds,
                        "data_wait_seconds": (
                            data_token_count_seconds + data_prefetch_wait_seconds
                        ),
                        "data_wait_fraction": (
                            data_token_count_seconds + data_prefetch_wait_seconds
                        )
                        / compute_step_seconds,
                        "teacher_cpu_offload_stage_seconds": (
                            teacher_stage_transition.seconds
                            if teacher_stage_transition is not None
                            else 0.0
                        ),
                        "teacher_cpu_offload_restore_seconds": (
                            teacher_restore_transition.seconds
                            if teacher_restore_transition is not None
                            else 0.0
                        ),
                        "teacher_cpu_offload_transferred_bytes": (
                            teacher_stage_transition.transferred_bytes
                            if teacher_stage_transition is not None
                            else 0
                        ),
                        "teacher_cpu_offload_released_cuda_bytes": (
                            teacher_restore_transition.released_cuda_bytes
                            if teacher_restore_transition is not None
                            else 0
                        ),
                        "data_phase": step_data_phase,
                        "next_data_phase": next_data_phase,
                    }
                )
                telemetry.update(
                    _checkpoint_phase_log_fields(
                        config,
                        align_hidden=align_hidden,
                        outer_checkpoint_layer_indices=(
                            activation_checkpoint_layer_indices_enabled
                        ),
                        inner_checkpoint_layer_indices=(
                            transfer_token_checkpoint_layer_indices_enabled
                        ),
                    )
                )
                # JsonlMetricLogger owns the canonical step field.
                telemetry_record = {key: value for key, value in telemetry.items() if key != "step"}
                if telemetry_logger is not None:
                    telemetry_logger.log(state.global_step, telemetry_record)
                if progress_ui is not None:
                    progress_ui.update(
                        min(state.committed_tokens, config.optimizer.max_tokens),
                        {**averaged, **telemetry_record},
                    )
                if (
                    context.is_rank_zero
                    and (progress_ui is None or not progress_ui.enabled)
                    and state.global_step % config.runtime.log_every_steps == 0
                ):
                    _rank_zero_print(
                        context,
                        {
                            "event": "step",
                            "step": state.global_step,
                            **averaged,
                            **telemetry_record,
                        },
                    )

            final_path = _checkpoint(
                manager,
                stateful,
                state,
                cursor,
                config,
                report,
                kind="milestone",
                boundary=None,
                tag="complete",
                event_logger=event_logger,
            )
            if event_logger is not None:
                event_logger.log(
                    "train_complete",
                    {
                        "checkpoint": str(final_path),
                        "step": state.global_step,
                        "tokens": state.committed_tokens,
                    },
                )
            _rank_zero_print(
                context,
                {
                    "event": "train_complete",
                    "checkpoint": str(final_path),
                    "step": state.global_step,
                    "tokens": state.committed_tokens,
                },
            )
            session_status = "completed"
            return 0
    except ImmediateExit as exc:
        session_status = "immediate_exit"
        if event_logger is not None:
            event_logger.log("immediate_exit", exception_fields(exc))
        raise
    except BaseException as exc:
        session_status = "failed"
        if event_logger is not None:
            event_logger.log(
                "train_failed",
                exception_fields(exc),
            )
        raise
    finally:
        if teacher_offload is not None and teacher_offload.is_staged:
            try:
                emergency_restore = teacher_offload.restore()
                if event_logger is not None:
                    event_logger.log(
                        "teacher_cpu_offload_emergency_restore",
                        dataclasses.asdict(emergency_restore),
                    )
            except Exception as offload_error:
                if event_logger is not None:
                    event_logger.log(
                        "teacher_cpu_offload_restore_failed",
                        exception_fields(offload_error),
                    )
        if session_file is not None:
            final_fields = (
                {"step": state.global_step, "tokens": state.committed_tokens}
                if state is not None
                else None
            )
            try:
                session_file.finish(session_status, final_fields)
            except Exception as session_error:
                if event_logger is not None:
                    event_logger.log(
                        "session_file_update_failed",
                        exception_fields(session_error),
                    )
        if progress_ui is not None:
            progress_ui.close()
        if manager is not None:
            manager.release_run_lock()
        if prefetched_batches is not None:
            prefetched_batches.close()
        if store is not None:
            store.close()
        if cooldown_store is not None:
            cooldown_store.close()
        finalize_distributed(context, barrier=False)
