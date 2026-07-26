"""Typed configuration and resume-critical fingerprinting.

The project intentionally keeps configuration loading small and explicit.  A
checkpoint fingerprints every value that can change the optimization result;
runtime-only values may change across a resume without pretending that a new
experiment is the old one.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

SCHEMA_VERSION = 1
_UNPINNED_REVISIONS = {"", "main", "master", "latest", "none", "null"}
_SHA_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$", re.IGNORECASE)


class ConfigError(ValueError):
    """Raised when a run configuration is incomplete or unsafe to resume."""


@dataclass(slots=True)
class ModelSource:
    model_id: str
    revision: str
    local_path: str
    manifest_sha256: str

    def validate(self, name: str) -> None:
        if not self.model_id:
            raise ConfigError(f"sources.{name}.model_id is required")
        if not self.local_path:
            raise ConfigError(f"sources.{name}.local_path is required")
        if self.revision.strip().lower() in _UNPINNED_REVISIONS or not _SHA_PATTERN.match(
            self.revision
        ):
            raise ConfigError(
                f"sources.{name}.revision must be an immutable commit SHA, got {self.revision!r}"
            )
        if not re.fullmatch(r"[0-9a-fA-F]{64}", self.manifest_sha256):
            raise ConfigError(f"sources.{name}.manifest_sha256 must be a SHA256 hex digest")
        self.revision = self.revision.lower()
        self.manifest_sha256 = self.manifest_sha256.lower()


@dataclass(slots=True)
class SourcesConfig:
    backbone: ModelSource
    donor: ModelSource
    teacher: ModelSource
    tokenizer: ModelSource
    folded_experts_path: str | None = None
    folded_experts_sha256: str | None = None

    def validate(self, *, stage: str) -> None:
        if stage != "sparse":
            return
        if not self.folded_experts_path:
            raise ConfigError("sparse stage requires sources.folded_experts_path")
        if not self.folded_experts_sha256 or not re.fullmatch(
            r"[0-9a-fA-F]{64}", self.folded_experts_sha256
        ):
            raise ConfigError(
                "sparse stage requires sources.folded_experts_sha256 as a 64-character SHA256"
            )
        self.folded_experts_sha256 = self.folded_experts_sha256.lower()


@dataclass(slots=True)
class ArchitectureConfig:
    student_hidden_size: int = 1024
    student_intermediate_size: int = 3584
    student_layers: int = 24
    donor_hidden_size: int = 4096
    donor_intermediate_size: int = 12288
    donor_layers: int = 32
    num_experts: int = 8
    expert_intermediate_size: int = 1536
    top_k: int = 2
    lora_rank: int = 16
    norm_topk_prob: bool = True
    expert_initialization: str = "donor"
    random_expert_seed: int = 1701
    layer_map_path: str = "artifacts/calibration/layer_map.json"
    channel_map_path: str = "artifacts/calibration/channel_map.json"
    adapter_init_path: str = "artifacts/calibration/adapters.safetensors"
    active_student_layers: list[int] | None = None

    def validate(self) -> None:
        expected = self.num_experts * self.expert_intermediate_size
        if expected != self.donor_intermediate_size:
            raise ConfigError(
                "num_experts * expert_intermediate_size must exactly partition donor FFN "
                f"({expected} != {self.donor_intermediate_size})"
            )
        if not 1 <= self.top_k <= self.num_experts:
            raise ConfigError("top_k must be in [1, num_experts]")
        if self.student_layers > self.donor_layers:
            raise ConfigError("student_layers cannot exceed donor_layers for monotonic injection")
        if self.student_hidden_size <= 0 or self.donor_hidden_size <= 0:
            raise ConfigError("hidden sizes must be positive")
        if self.lora_rank != 16:
            raise ConfigError("v1 requires architecture.lora_rank=16")
        if not self.norm_topk_prob:
            raise ConfigError(
                "v1 requires architecture.norm_topk_prob=true because native "
                "Qwen3.5-MoE inference always normalizes selected router weights"
            )
        if self.expert_initialization not in {"donor", "random-control"}:
            raise ConfigError(
                "architecture.expert_initialization must be 'donor' or 'random-control'"
            )
        if isinstance(self.random_expert_seed, bool) or self.random_expert_seed < 0:
            raise ConfigError("architecture.random_expert_seed must be a non-negative integer")
        active = self.active_student_layers
        if active is not None:
            if not active or any(isinstance(item, bool) for item in active):
                raise ConfigError("architecture.active_student_layers must be a non-empty list")
            if sorted(set(active)) != active:
                raise ConfigError(
                    "architecture.active_student_layers must be sorted and contain no duplicates"
                )
            if active[0] < 0 or active[-1] >= self.student_layers:
                raise ConfigError("architecture.active_student_layers contains an invalid layer")

    def active_layers(self) -> tuple[int, ...]:
        if self.active_student_layers is None:
            return tuple(range(self.student_layers))
        return tuple(self.active_student_layers)


@dataclass(slots=True)
class DataConfig:
    manifest_path: str
    manifest_sha256: str
    teacher_kd_manifest_path: str
    teacher_kd_manifest_sha256: str
    max_sequence_length: int = 4096
    micro_batch_size: int = 1
    global_batch_tokens: int = 262_144
    shuffle_seed: int = 3407
    num_workers: int = 4
    teacher_top_k: int = 64
    # Optional, independently authenticated whole-shard subset used only after
    # ``quality_cooldown_start_tokens``.  All five fields are an atomic
    # contract; the all-None default preserves every legacy v1 data path and
    # critical fingerprint.
    quality_cooldown_manifest_path: str | None = None
    quality_cooldown_manifest_sha256: str | None = None
    quality_cooldown_teacher_kd_manifest_path: str | None = None
    quality_cooldown_teacher_kd_manifest_sha256: str | None = None
    quality_cooldown_start_tokens: int | None = None

    def quality_cooldown_enabled(self) -> bool:
        return self.quality_cooldown_start_tokens is not None

    def validate(self) -> None:
        for name, digest in (
            ("manifest_sha256", self.manifest_sha256),
            ("teacher_kd_manifest_sha256", self.teacher_kd_manifest_sha256),
        ):
            if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                raise ConfigError(f"data.{name} must be a SHA256 hex digest")
        self.manifest_sha256 = self.manifest_sha256.lower()
        self.teacher_kd_manifest_sha256 = self.teacher_kd_manifest_sha256.lower()
        if self.max_sequence_length <= 0 or self.micro_batch_size <= 0:
            raise ConfigError("sequence length and micro batch size must be positive")
        if self.global_batch_tokens <= 0:
            raise ConfigError("global_batch_tokens must be positive")
        if isinstance(self.num_workers, bool) or self.num_workers < 1:
            raise ConfigError(
                "data.num_workers must be positive (it controls the bounded KD prefetch depth)"
            )
        if self.teacher_top_k != 64:
            raise ConfigError("the current top-64 KD contract requires teacher_top_k=64")
        cooldown_values = (
            self.quality_cooldown_manifest_path,
            self.quality_cooldown_manifest_sha256,
            self.quality_cooldown_teacher_kd_manifest_path,
            self.quality_cooldown_teacher_kd_manifest_sha256,
            self.quality_cooldown_start_tokens,
        )
        if any(value is not None for value in cooldown_values) and not all(
            value is not None for value in cooldown_values
        ):
            raise ConfigError(
                "quality cooldown requires prepared/KD paths, SHA256 values, and start tokens"
            )
        if self.quality_cooldown_enabled():
            for name in (
                "quality_cooldown_manifest_path",
                "quality_cooldown_teacher_kd_manifest_path",
            ):
                value = getattr(self, name)
                if not isinstance(value, str) or not value:
                    raise ConfigError(f"data.{name} must be a non-empty path")
            for name in (
                "quality_cooldown_manifest_sha256",
                "quality_cooldown_teacher_kd_manifest_sha256",
            ):
                digest = getattr(self, name)
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                    raise ConfigError(f"data.{name} must be a SHA256 hex digest")
                setattr(self, name, digest.lower())
            start = self.quality_cooldown_start_tokens
            if isinstance(start, bool) or not isinstance(start, int) or start <= 0:
                raise ConfigError("data.quality_cooldown_start_tokens must be a positive integer")
            if (
                self.quality_cooldown_manifest_path == self.manifest_path
                or self.quality_cooldown_manifest_sha256 == self.manifest_sha256
                or self.quality_cooldown_teacher_kd_manifest_path == self.teacher_kd_manifest_path
                or self.quality_cooldown_teacher_kd_manifest_sha256
                == self.teacher_kd_manifest_sha256
            ):
                raise ConfigError(
                    "quality cooldown requires independent prepared and KD manifest identities"
                )


@dataclass(slots=True)
class LossConfig:
    ntp: float = 1.0
    # Project-specific auxiliary coefficient.  Qwen3.5 publishes the native
    # MTP weights/architecture but no training-loss coefficient, so zero is the
    # only honest compatibility default and every enabled run must choose an
    # explicit non-zero value in its resolved configuration.
    mtp: float = 0.0
    teacher_kd: float = 1.0
    hidden_alignment: float = 0.1
    anchor_kl: float = 0.1
    dense_oracle: float = 1.0
    router_supervision: float = 1.0
    load_balance: float = 0.01
    router_z: float = 0.001
    kd_temperature: float = 2.0
    dense_oracle_batch_fraction: float = 0.1
    hidden_alignment_batch_fraction: float = 0.05

    def validate(self) -> None:
        values = dataclasses.asdict(self)
        for name, value in values.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ConfigError(f"losses.{name} must be finite")
        for name in (
            "ntp",
            "mtp",
            "teacher_kd",
            "hidden_alignment",
            "anchor_kl",
            "dense_oracle",
            "router_supervision",
            "load_balance",
            "router_z",
        ):
            if float(values[name]) < 0:
                raise ConfigError(f"losses.{name} must be non-negative")
        if self.kd_temperature <= 0:
            raise ConfigError("losses.kd_temperature must be positive")
        for name in ("dense_oracle_batch_fraction", "hidden_alignment_batch_fraction"):
            if not 0.0 <= float(values[name]) <= 1.0:
                raise ConfigError(f"losses.{name} must be in [0, 1]")


@dataclass(slots=True)
class OptimizerConfig:
    adapter_lr: float = 2e-4
    router_lr: float = 1e-3
    lora_lr: float = 2e-4
    scale_lr: float = 1e-3
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    warmup_tokens: int = 10_000_000
    max_tokens: int = 500_000_000
    # These three opt-in fields preserve the legacy v1 configuration and
    # checkpoint contract when left at their defaults.  WSD keeps the peak LR
    # stable until the final ``decay_tokens`` of the run.
    lr_schedule: str = "cosine"
    min_lr_ratio: float = 0.1
    decay_tokens: int | None = None
    grad_clip_norm: float = 1.0

    def validate(self) -> None:
        for name in ("adapter_lr", "router_lr", "lora_lr", "scale_lr"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ConfigError(f"optimizer.{name} must be finite and positive")
        for name in ("weight_decay", "adam_eps", "grad_clip_norm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ConfigError(f"optimizer.{name} must be finite and non-negative")
        for name in ("adam_beta1", "adam_beta2"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value < 1.0:
                raise ConfigError(f"optimizer.{name} must be in [0, 1)")
        if self.adam_eps <= 0 or self.grad_clip_norm <= 0:
            raise ConfigError("optimizer.adam_eps and grad_clip_norm must be positive")
        if self.warmup_tokens < 0 or self.max_tokens <= 0:
            raise ConfigError("optimizer token counts are invalid")
        if self.warmup_tokens >= self.max_tokens:
            raise ConfigError("optimizer.warmup_tokens must be smaller than max_tokens")
        if self.lr_schedule not in {"cosine", "warmup-stable-decay"}:
            raise ConfigError("optimizer.lr_schedule must be 'cosine' or 'warmup-stable-decay'")
        if (
            isinstance(self.min_lr_ratio, bool)
            or not isinstance(self.min_lr_ratio, (int, float))
            or not math.isfinite(float(self.min_lr_ratio))
            or not 0.0 <= float(self.min_lr_ratio) <= 1.0
        ):
            raise ConfigError("optimizer.min_lr_ratio must be finite and in [0, 1]")
        if self.lr_schedule == "cosine":
            if self.decay_tokens is not None:
                raise ConfigError("optimizer.decay_tokens is only valid for warmup-stable-decay")
        elif (
            isinstance(self.decay_tokens, bool)
            or not isinstance(self.decay_tokens, int)
            or self.decay_tokens <= 0
            or self.decay_tokens > self.max_tokens - self.warmup_tokens
        ):
            raise ConfigError(
                "warmup-stable-decay requires optimizer.decay_tokens in "
                "[1, max_tokens - warmup_tokens]"
            )


@dataclass(slots=True)
class CheckpointConfig:
    output_dir: str
    every_steps: int = 100
    every_minutes: float = 10.0
    keep_last: int = 3
    stop_file: str = "STOP"
    save_on_signal: bool = True

    def validate(self) -> None:
        if self.every_steps <= 0 or self.every_minutes <= 0:
            raise ConfigError("checkpoint intervals must be positive")
        if self.keep_last < 1:
            raise ConfigError("checkpoint.keep_last must be >= 1")
        if not self.save_on_signal:
            raise ConfigError("v1 requires checkpoint.save_on_signal=true")


@dataclass(slots=True)
class RuntimeConfig:
    bf16: bool = True
    seed: int = 3407
    deterministic: bool = False
    log_every_steps: int = 1
    profile: bool = False
    profile_wait_steps: int = 1
    profile_warmup_steps: int = 1
    profile_active_steps: int = 3
    offline: bool = True
    allow_tf32: bool = True
    sharding: str = "fsdp2"
    activation_checkpointing: bool = True
    # ``None`` preserves the legacy global policy: checkpoint every active
    # layer, or only alignment steps when the alignment-only switch is set.
    # An explicit integer selects that many active layer indices on ordinary
    # steps; alignment-only mode still expands alignment steps to every active
    # layer.  The value is resume-critical because recomputation can change the
    # numerical optimization trajectory.
    activation_checkpoint_layer_count: int | None = None
    # Optional alignment-phase override for outer decoder checkpointing.  None
    # preserves the historical policy above (including alignment-only mode).
    # An explicit value is resume-critical and selects exactly that many active
    # student layers during hidden-alignment optimizer batches.
    hidden_alignment_activation_checkpoint_layer_count: int | None = None
    # Dense Stage-B can either execute the historical expanded
    # B(D(silu(G(Ax))*U(Ax))) branch or differentiably fold A/B into the
    # frozen donor projections for each microbatch.  The latter changes BF16
    # association and is therefore resume-critical rather than a cosmetic
    # performance switch.
    dense_transfer_execution: str = "expanded"
    # Checkpoint only the token-dependent transfer branch. Folded execution
    # keeps fold GEMMs outside replay; expanded execution uses selective
    # checkpointing that caches the exact down-projection result. The engine
    # disables either form on decoder layers already covered by outer activation
    # checkpointing so nested recomputation cannot occur.
    dense_transfer_token_checkpoint: bool = False
    # Phase-specific counts cap inner selective transfer checkpoints.  None
    # preserves the historical enabled behavior of checkpointing the complete
    # outer-layer complement; explicit counts select deterministic evenly spaced
    # subsets and remain resume-critical.
    dense_transfer_checkpoint_layer_count: int | None = None
    hidden_alignment_dense_transfer_checkpoint_layer_count: int | None = None
    teacher_cpu_offload: bool = False
    activation_checkpointing_on_alignment_only: bool = False
    fused_adamw: bool = True
    loss_chunk_tokens: int = 128
    loss_checkpoint_chunks: bool = True
    compile_streaming_loss: bool = True
    expandable_segments: bool = True

    def validate(self) -> None:
        if not self.bf16:
            raise ConfigError("v1 requires runtime.bf16=true")
        if not self.offline:
            raise ConfigError("v1 training requires runtime.offline=true")
        if self.log_every_steps <= 0:
            raise ConfigError("runtime.log_every_steps must be positive")
        if self.profile_wait_steps < 0 or self.profile_warmup_steps < 0:
            raise ConfigError("runtime profiler wait/warmup steps must be non-negative")
        if self.profile_active_steps <= 0:
            raise ConfigError("runtime.profile_active_steps must be positive")
        if isinstance(self.loss_chunk_tokens, bool) or self.loss_chunk_tokens <= 0:
            raise ConfigError("runtime.loss_chunk_tokens must be a positive integer")
        if not isinstance(self.compile_streaming_loss, bool):
            raise ConfigError("runtime.compile_streaming_loss must be a boolean")
        for name in (
            "activation_checkpoint_layer_count",
            "hidden_alignment_activation_checkpoint_layer_count",
            "dense_transfer_checkpoint_layer_count",
            "hidden_alignment_dense_transfer_checkpoint_layer_count",
        ):
            count = getattr(self, name)
            if count is not None and (isinstance(count, bool) or not isinstance(count, int)):
                raise ConfigError(f"runtime.{name} must be an integer or null")
        if not isinstance(self.teacher_cpu_offload, bool):
            raise ConfigError("runtime.teacher_cpu_offload must be a boolean")
        if self.dense_transfer_execution not in {"expanded", "differentiable_folded"}:
            raise ConfigError(
                "runtime.dense_transfer_execution must be 'expanded' or 'differentiable_folded'"
            )
        if not isinstance(self.dense_transfer_token_checkpoint, bool):
            raise ConfigError("runtime.dense_transfer_token_checkpoint must be a boolean")
        if not isinstance(self.activation_checkpointing_on_alignment_only, bool):
            raise ConfigError(
                "runtime.activation_checkpointing_on_alignment_only must be a boolean"
            )


@dataclass(slots=True)
class TrainConfig:
    run_id: str
    track: str
    stage: str
    sources: SourcesConfig
    architecture: ArchitectureConfig
    data: DataConfig
    losses: LossConfig
    optimizer: OptimizerConfig
    checkpoint: CheckpointConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ConfigError(
                f"unsupported config schema {self.schema_version}; expected {SCHEMA_VERSION}"
            )
        if self.track not in {"base", "posttrained"}:
            raise ConfigError("track must be 'base' or 'posttrained'")
        if self.stage not in {"dense-oracle", "sparse"}:
            raise ConfigError("stage must be 'dense-oracle' or 'sparse'")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.run_id):
            raise ConfigError("run_id must be a filesystem-safe identifier")
        for name in ("backbone", "donor", "teacher", "tokenizer"):
            getattr(self.sources, name).validate(name)
        for role, reference in (("tokenizer", "backbone"), ("teacher", "donor")):
            source = getattr(self.sources, role)
            expected = getattr(self.sources, reference)
            for field_name in ("model_id", "revision", "manifest_sha256"):
                if getattr(source, field_name) != getattr(expected, field_name):
                    raise ConfigError(
                        f"v1 requires sources.{role}.{field_name} to match "
                        f"sources.{reference}.{field_name}"
                    )
        self.sources.validate(stage=self.stage)
        self.architecture.validate()
        self.data.validate()
        self.losses.validate()
        self.optimizer.validate()
        self.checkpoint.validate()
        self.runtime.validate()
        if self.data.quality_cooldown_enabled():
            start = self.data.quality_cooldown_start_tokens
            assert start is not None
            if start >= self.optimizer.max_tokens:
                raise ConfigError(
                    "data.quality_cooldown_start_tokens must be below optimizer.max_tokens"
                )
        checkpoint_layer_count = self.runtime.activation_checkpoint_layer_count
        active_layer_count = len(self.architecture.active_layers())
        checkpoint_counts = {
            "activation_checkpoint_layer_count": checkpoint_layer_count,
            "hidden_alignment_activation_checkpoint_layer_count": (
                self.runtime.hidden_alignment_activation_checkpoint_layer_count
            ),
            "dense_transfer_checkpoint_layer_count": (
                self.runtime.dense_transfer_checkpoint_layer_count
            ),
            "hidden_alignment_dense_transfer_checkpoint_layer_count": (
                self.runtime.hidden_alignment_dense_transfer_checkpoint_layer_count
            ),
        }
        for name, count in checkpoint_counts.items():
            if count is not None and not 0 <= count <= active_layer_count:
                raise ConfigError(
                    f"runtime.{name} must be in [0, {active_layer_count}] "
                    "for the configured active student layers"
                )
        if not self.runtime.activation_checkpointing and any(
            count not in (None, 0)
            for count in (
                checkpoint_layer_count,
                self.runtime.hidden_alignment_activation_checkpoint_layer_count,
            )
        ):
            raise ConfigError(
                "runtime.activation_checkpointing=false requires "
                "ordinary/alignment activation checkpoint layer counts to be 0 or null"
            )
        if not self.runtime.dense_transfer_token_checkpoint and any(
            count is not None
            for count in (
                self.runtime.dense_transfer_checkpoint_layer_count,
                self.runtime.hidden_alignment_dense_transfer_checkpoint_layer_count,
            )
        ):
            raise ConfigError(
                "explicit dense transfer checkpoint layer counts require "
                "runtime.dense_transfer_token_checkpoint=true"
            )
        if self.losses.mtp > 0 and self.data.max_sequence_length < 3:
            raise ConfigError("losses.mtp>0 requires data.max_sequence_length>=3 for an L-2 target")
        if self.runtime.sharding not in {"ddp", "fsdp2"}:
            raise ConfigError("runtime.sharding must be 'ddp' or 'fsdp2'")
        if self.runtime.teacher_cpu_offload and (
            self.stage != "dense-oracle"
            or self.architecture.expert_initialization != "donor"
            or self.losses.hidden_alignment <= 0
            or self.losses.hidden_alignment_batch_fraction <= 0
        ):
            raise ConfigError(
                "runtime.teacher_cpu_offload requires dense-oracle donor initialization "
                "with enabled hidden alignment"
            )
        if self.runtime.activation_checkpointing_on_alignment_only and (
            not self.runtime.teacher_cpu_offload or not self.runtime.activation_checkpointing
        ):
            raise ConfigError(
                "runtime.activation_checkpointing_on_alignment_only requires "
                "teacher_cpu_offload=true and activation_checkpointing=true"
            )
        if self.stage != "dense-oracle" and (
            self.runtime.dense_transfer_execution != "expanded"
            or self.runtime.dense_transfer_token_checkpoint
            or self.runtime.hidden_alignment_activation_checkpoint_layer_count is not None
            or self.runtime.dense_transfer_checkpoint_layer_count is not None
            or self.runtime.hidden_alignment_dense_transfer_checkpoint_layer_count is not None
        ):
            raise ConfigError(
                "runtime phase-specific dense transfer/hidden-alignment checkpoint controls "
                "are only valid for dense-oracle"
            )
        if (
            self.stage == "sparse"
            and len(self.architecture.active_layers()) != self.architecture.student_layers
        ):
            raise ConfigError("sparse/native v1 requires all 24 student layers to be active")
        if self.stage == "sparse" and self.architecture.expert_initialization != "donor":
            raise ConfigError("sparse/native v1 requires donor expert initialization")
        if self.stage == "sparse" and self.losses.hidden_alignment != 0:
            raise ConfigError(
                "sparse/native v1 does not construct an online hidden-alignment teacher; "
                "set losses.hidden_alignment=0"
            )

    def canonical_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        optimizer = value["optimizer"]
        # Keep legacy v1 resolved configs and critical fingerprints byte-for-
        # byte compatible.  The old scheduler was exactly cosine with a fixed
        # 0.1 minimum ratio and no separate cooldown interval.
        if (
            optimizer.get("lr_schedule") == "cosine"
            and float(optimizer.get("min_lr_ratio", 0.1)) == 0.1
            and optimizer.get("decay_tokens") is None
        ):
            optimizer.pop("lr_schedule", None)
            optimizer.pop("min_lr_ratio", None)
            optimizer.pop("decay_tokens", None)
        runtime = value["runtime"]
        data = value["data"]
        if data.get("quality_cooldown_start_tokens") is None:
            for name in (
                "quality_cooldown_manifest_path",
                "quality_cooldown_manifest_sha256",
                "quality_cooldown_teacher_kd_manifest_path",
                "quality_cooldown_teacher_kd_manifest_sha256",
                "quality_cooldown_start_tokens",
            ):
                data.pop(name, None)
        # New phase-specific counts are opt-in. Omitting every None preserves
        # historical v1 resolved configs and their critical fingerprints.
        for name in (
            "hidden_alignment_activation_checkpoint_layer_count",
            "dense_transfer_checkpoint_layer_count",
            "hidden_alignment_dense_transfer_checkpoint_layer_count",
        ):
            if runtime.get(name) is None:
                runtime.pop(name, None)
        # Preserve legacy v1 resolved configs and their critical fingerprint.
        # The historical execution is exactly expanded with no inner branch
        # checkpoint, so omit the new fields only for that compatibility pair.
        if (
            runtime.get("dense_transfer_execution") == "expanded"
            and runtime.get("dense_transfer_token_checkpoint") is False
        ):
            runtime.pop("dense_transfer_execution", None)
            runtime.pop("dense_transfer_token_checkpoint", None)
        return value

    def critical_dict(self) -> dict[str, Any]:
        """Values that must match for an exact resume.

        Worker count, logging cadence, profiling, checkpoint cadence, and the
        output path do not affect the optimization trajectory and may change.
        """

        value = self.canonical_dict()
        value["data"].pop("num_workers", None)
        value["checkpoint"].pop("output_dir", None)
        value["checkpoint"].pop("every_steps", None)
        value["checkpoint"].pop("every_minutes", None)
        value["checkpoint"].pop("keep_last", None)
        value["checkpoint"].pop("stop_file", None)
        value["runtime"].pop("log_every_steps", None)
        value["runtime"].pop("profile", None)
        value["runtime"].pop("profile_wait_steps", None)
        value["runtime"].pop("profile_warmup_steps", None)
        value["runtime"].pop("profile_active_steps", None)
        value["runtime"].pop("expandable_segments", None)
        return value

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.critical_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def activation_checkpoint_layer_indices(self, *, align_hidden: bool) -> tuple[int, ...]:
        """Resolve the exact active decoder layers checkpointed for one step.

        Explicit counts are sampled over ``active_student_layers`` rather than
        assuming that active layer numbers are contiguous.  Sampling includes
        both endpoints and uses integer nearest-neighbour rounding, matching
        the full-graph benchmark contract without a numpy/version dependency.
        """

        if not self.runtime.activation_checkpointing:
            return ()
        active_layers = self.architecture.active_layers()
        alignment_count = self.runtime.hidden_alignment_activation_checkpoint_layer_count
        if align_hidden and alignment_count is not None:
            count = alignment_count
        elif align_hidden and self.runtime.activation_checkpointing_on_alignment_only:
            return active_layers
        else:
            count = self.runtime.activation_checkpoint_layer_count
        if (
            not align_hidden
            and self.runtime.activation_checkpointing_on_alignment_only
            and count is None
        ):
            return ()
        if count is None:
            return active_layers
        return self._evenly_spaced_active_layers(
            active_layers,
            count,
            field_name=(
                "hidden_alignment_activation_checkpoint_layer_count"
                if align_hidden and alignment_count is not None
                else "activation_checkpoint_layer_count"
            ),
        )

    @staticmethod
    def _evenly_spaced_active_layers(
        active_layers: tuple[int, ...],
        count: int,
        *,
        field_name: str,
    ) -> tuple[int, ...]:
        if count == 0:
            return ()
        if not 0 < count <= len(active_layers):
            # Validated TrainConfig instances cannot reach this branch, but
            # keeping the resolver fail-closed protects programmatic callers.
            raise ConfigError(f"runtime.{field_name} exceeds active student layers")
        if count == 1:
            return (active_layers[len(active_layers) // 2],)
        denominator = count - 1
        last_position = len(active_layers) - 1
        positions = tuple(
            (2 * position * last_position + denominator) // (2 * denominator)
            for position in range(count)
        )
        if len(set(positions)) != count:  # pragma: no cover - count <= active layers
            raise RuntimeError(f"activation checkpoint positions are not unique: {positions}")
        return tuple(active_layers[position] for position in positions)

    def dense_transfer_checkpoint_layer_indices(
        self,
        *,
        align_hidden: bool,
        outer_checkpoint_layer_indices: tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
        """Resolve exact inner checkpoints from the selected outer complement."""

        if not self.runtime.dense_transfer_token_checkpoint:
            if (
                self.runtime.dense_transfer_checkpoint_layer_count is not None
                or self.runtime.hidden_alignment_dense_transfer_checkpoint_layer_count is not None
            ):
                raise ConfigError(
                    "dense transfer checkpoint counts require the token checkpoint switch"
                )
            return ()
        active_layers = self.architecture.active_layers()
        outer = (
            self.activation_checkpoint_layer_indices(align_hidden=align_hidden)
            if outer_checkpoint_layer_indices is None
            else tuple(outer_checkpoint_layer_indices)
        )
        if tuple(sorted(set(outer))) != outer or not set(outer).issubset(active_layers):
            raise ConfigError(
                "outer checkpoint layer indices must be a sorted subset of active layers"
            )
        outer_set = set(outer)
        available = tuple(layer for layer in active_layers if layer not in outer_set)
        requested = (
            self.runtime.hidden_alignment_dense_transfer_checkpoint_layer_count
            if align_hidden
            else self.runtime.dense_transfer_checkpoint_layer_count
        )
        if requested is None:
            return available
        effective = min(requested, len(available))
        selected = self._evenly_spaced_active_layers(
            available,
            effective,
            field_name=(
                "hidden_alignment_dense_transfer_checkpoint_layer_count"
                if align_hidden
                else "dense_transfer_checkpoint_layer_count"
            ),
        )
        if set(selected).intersection(outer):  # pragma: no cover - complement construction
            raise RuntimeError("nested outer/inner activation checkpointing is forbidden")
        return selected


T = TypeVar("T")


def _coerce_dataclass(cls: type[T], value: Mapping[str, Any], path: str) -> T:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping")
    hints = get_type_hints(cls)
    fields = {item.name: item for item in dataclasses.fields(cls)}
    unknown = sorted(set(value) - set(fields))
    if unknown:
        raise ConfigError(f"unknown keys at {path}: {', '.join(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, item in fields.items():
        if name not in value:
            if item.default is dataclasses.MISSING and item.default_factory is dataclasses.MISSING:
                raise ConfigError(f"missing required key {path}.{name}")
            continue
        raw = value[name]
        target = hints.get(name, item.type)
        origin = get_origin(target)
        candidates = get_args(target) if origin is not None else ()
        nested_type = None
        if dataclasses.is_dataclass(target):
            nested_type = target
        else:
            nested_type = next((arg for arg in candidates if dataclasses.is_dataclass(arg)), None)
        kwargs[name] = _coerce_dataclass(nested_type, raw, f"{path}.{name}") if nested_type else raw
    return cls(**kwargs)


def load_train_config(path: str | Path) -> TrainConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ConfigError("top-level YAML document must be a mapping")
    config = _coerce_dataclass(TrainConfig, raw, "config")
    config.validate()
    return config


def dump_resolved_config(config: TrainConfig, path: str | Path) -> None:
    from .utils import atomic_write_text

    target = Path(path)
    rendered = yaml.safe_dump(config.canonical_dict(), sort_keys=True, allow_unicode=True)
    atomic_write_text(target, rendered)
