"""Read-only hardware inspection and static training-memory estimates.

Importing this module does not import PyTorch or initialize CUDA.  The public
inspection function imports PyTorch lazily and only queries version/device
properties; it never constructs a model, optimizer, or training process.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import math
import os
import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import TrainConfig

_GIB = 1024**3
_WEIGHT_SUFFIXES = (".safetensors", ".bin")
_LOWER_BOUND_EXCLUDES = (
    "activations and saved tensors",
    "LM/KD logits and loss workspaces",
    "NCCL/FSDP collectives and all-gather buffers",
    "CUDA kernels, graphs, allocator fragmentation, and temporary workspaces",
    "dataloader pinned-memory and host-side model construction peaks",
)


@dataclass(frozen=True, slots=True)
class CPUInfo:
    model: str | None
    architecture: str
    physical_cores: int | None
    logical_cores: int | None
    affinity_cores: int | None
    memory_total_bytes: int | None
    memory_available_bytes: int | None


@dataclass(frozen=True, slots=True)
class TorchInfo:
    installed: bool
    version: str | None
    cuda_available: bool
    cuda_runtime_version: str | None
    hip_runtime_version: str | None
    cudnn_version: int | None
    visible_device_count: int
    import_error: str | None = None


@dataclass(frozen=True, slots=True)
class GPUInfo:
    index: int
    name: str
    total_memory_bytes: int
    compute_capability: str | None
    bf16_supported: bool | None
    multiprocessor_count: int | None


@dataclass(frozen=True, slots=True)
class AllocatorEnvironment:
    pytorch_alloc_conf: str | None
    pytorch_cuda_alloc_conf: str | None
    effective_alloc_conf: str | None
    expandable_segments: bool | None
    cuda_visible_devices: str | None


@dataclass(frozen=True, slots=True)
class AcceleratorKernels:
    flash_linear_attention_version: str | None
    causal_conv1d_version: str | None
    qwen35_fast_path_ready: bool


@dataclass(frozen=True, slots=True)
class MemoryComponent:
    name: str
    parameter_count: int | None
    bytes: int
    dtype_or_state: str
    source: str


@dataclass(frozen=True, slots=True)
class HardwareWarning:
    severity: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class StaticMemoryEstimate:
    stage: str
    active_layers: int
    total_layers: int
    world_size: int
    sharding: str
    components: tuple[MemoryComponent, ...]
    aggregate_known_static_bytes: int
    estimated_per_device_static_bytes: int
    is_runtime_lower_bound: bool
    excludes: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HardwareReport:
    cpu: CPUInfo
    torch: TorchInfo
    gpus: tuple[GPUInfo, ...]
    allocator: AllocatorEnvironment
    kernels: AcceleratorKernels
    static_memory: StaticMemoryEstimate | None
    warnings: tuple[HardwareWarning, ...]


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _inspect_cpu() -> CPUInfo:
    cpuinfo = _read_text(Path("/proc/cpuinfo"))
    model: str | None = None
    physical_pairs: set[tuple[str, str]] = set()
    if cpuinfo:
        for block in cpuinfo.split("\n\n"):
            values: dict[str, str] = {}
            for line in block.splitlines():
                key, separator, value = line.partition(":")
                if separator:
                    values[key.strip()] = value.strip()
            model = model or values.get("model name") or values.get("Hardware")
            physical_id = values.get("physical id")
            core_id = values.get("core id")
            if physical_id is not None and core_id is not None:
                physical_pairs.add((physical_id, core_id))

    affinity_cores: int | None = None
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        with contextlib.suppress(OSError):
            affinity_cores = len(get_affinity(0))

    memory: dict[str, int] = {}
    meminfo = _read_text(Path("/proc/meminfo"))
    if meminfo:
        for line in meminfo.splitlines():
            key, separator, raw = line.partition(":")
            if not separator:
                continue
            fields = raw.split()
            if fields and fields[0].isdigit():
                # Linux exposes these values in KiB.
                memory[key] = int(fields[0]) * 1024

    return CPUInfo(
        model=model or platform.processor() or None,
        architecture=platform.machine(),
        physical_cores=len(physical_pairs) or None,
        logical_cores=os.cpu_count(),
        affinity_cores=affinity_cores,
        memory_total_bytes=memory.get("MemTotal"),
        memory_available_bytes=memory.get("MemAvailable"),
    )


def _allocator_environment() -> AllocatorEnvironment:
    current = os.environ.get("PYTORCH_ALLOC_CONF")
    legacy = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    effective = current or legacy
    expandable: bool | None = None
    if effective:
        for item in effective.split(","):
            key, separator, value = item.partition(":")
            if separator and key.strip().lower() == "expandable_segments":
                normalized = value.strip().lower()
                if normalized in {"true", "1", "yes", "on"}:
                    expandable = True
                elif normalized in {"false", "0", "no", "off"}:
                    expandable = False
    return AllocatorEnvironment(
        pytorch_alloc_conf=current,
        pytorch_cuda_alloc_conf=legacy,
        effective_alloc_conf=effective,
        expandable_segments=expandable,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )


def _accelerator_kernels() -> AcceleratorKernels:
    def installed(distribution: str) -> str | None:
        try:
            return version(distribution)
        except PackageNotFoundError:
            return None

    flash = installed("flash-linear-attention")
    conv = installed("causal-conv1d")
    return AcceleratorKernels(
        flash_linear_attention_version=flash,
        causal_conv1d_version=conv,
        qwen35_fast_path_ready=flash is not None and conv is not None,
    )


def _manifest_weight_bytes(root: str | Path) -> tuple[int | None, str | None]:
    """Return declared model-weight bytes without hashing or opening any shard."""

    manifest = Path(root) / "download-manifest.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"could not read {manifest}: {exc}"
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        return None, f"{manifest} has no artifacts list"
    total = 0
    matched = 0
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        filename = str(artifact.get("filename", "")).lower()
        if not filename.endswith(_WEIGHT_SUFFIXES):
            continue
        size = artifact.get("expected_size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            return None, f"{manifest} has an invalid expected_size for {filename}"
        total += size
        matched += 1
    if not matched:
        return None, f"{manifest} declares no model weight files"
    return total, None


def _manifest_component(
    name: str,
    root: str | Path,
    *,
    bf16: bool,
) -> tuple[MemoryComponent | None, str | None]:
    size, error = _manifest_weight_bytes(root)
    if size is None:
        return None, error
    # The pinned Qwen sources are BF16.  Training explicitly casts floating
    # checkpoint tensors to FP32 when runtime.bf16 is disabled.
    scaled_size = size if bf16 else size * 2
    return (
        MemoryComponent(
            name=name,
            parameter_count=None,
            bytes=scaled_size,
            dtype_or_state="BF16 weight-file proxy" if bf16 else "FP32 (2x BF16 file proxy)",
            source=str(Path(root) / "download-manifest.json"),
        ),
        None,
    )


def _checkpoint_weight_map(root: str | Path) -> tuple[dict[str, str] | None, str | None]:
    """Read a safetensors weight map without importing torch or loading tensors."""

    directory = Path(root)
    indices = sorted(directory.glob("*.safetensors.index.json"))
    if indices:
        index = indices[0]
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"could not read {index}: {exc}"
        raw = payload.get("weight_map")
        if not isinstance(raw, dict) or not raw:
            return None, f"{index} has no weight_map"
        weight_map = {
            str(name): str(filename)
            for name, filename in raw.items()
            if isinstance(name, str) and isinstance(filename, str)
        }
        if len(weight_map) != len(raw):
            return None, f"{index} contains an invalid weight_map entry"
        return weight_map, None

    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        return None, f"{directory} has no safetensors checkpoint/index"
    try:
        from safetensors import safe_open
    except ImportError as exc:
        return None, f"safetensors is unavailable: {exc}"
    weight_map: dict[str, str] = {}
    try:
        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for name in handle:
                    if name in weight_map:
                        return None, f"duplicate tensor {name!r} across {directory}"
                    weight_map[name] = shard.name
    except (OSError, RuntimeError) as exc:
        return None, f"could not inspect {directory}: {exc}"
    return weight_map, None


def _text_checkpoint_parameter_count(
    root: str | Path,
    *,
    causal_lm: bool,
) -> tuple[int | None, str | None]:
    """Count the exact text tensors selected by Twen's local Qwen loaders.

    The upstream checkpoints are multimodal and also contain MTP tensors.  A
    whole-checkpoint file-size proxy therefore overstates live training memory
    by several GiB.  Shapes are read from safetensors headers only; tensor data
    is never materialized.
    """

    directory = Path(root)
    weight_map, error = _checkpoint_weight_map(directory)
    if weight_map is None:
        return None, error

    names = tuple(weight_map)
    multimodal_prefix = "model.language_model."
    if any(name.startswith(multimodal_prefix) for name in names):
        selected = {name for name in names if name.startswith(multimodal_prefix)}
    else:
        # Also support an already extracted text-only HF checkpoint.
        selected = {
            name
            for name in names
            if name.startswith("model.")
            and not name.startswith(("model.visual.", "model.vision.", "model.mtp."))
        }
    if causal_lm and "lm_head.weight" in weight_map:
        selected.add("lm_head.weight")
    if not selected:
        return None, f"{directory} has no recognizable Qwen text-model tensors"

    by_shard: dict[str, list[str]] = {}
    for name in selected:
        by_shard.setdefault(weight_map[name], []).append(name)
    try:
        from safetensors import safe_open
    except ImportError as exc:
        return None, f"safetensors is unavailable: {exc}"

    parameters = 0
    try:
        for filename, shard_names in by_shard.items():
            path = directory / filename
            with safe_open(path, framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                for name in shard_names:
                    if name not in available:
                        return None, f"{path} is missing indexed tensor {name!r}"
                    shape = handle.get_slice(name).get_shape()
                    parameters += math.prod(int(dimension) for dimension in shape)
    except (OSError, RuntimeError, ValueError) as exc:
        return None, f"could not inspect safetensors headers in {directory}: {exc}"
    return parameters, None


def _mtp_checkpoint_parameter_count(root: str | Path) -> tuple[int | None, str | None]:
    """Count exactly the top-level native ``mtp.*`` tensors from shard headers."""

    directory = Path(root)
    weight_map, error = _checkpoint_weight_map(directory)
    if weight_map is None:
        return None, error
    selected = {name for name in weight_map if name.startswith("mtp.")}
    if not selected:
        return None, f"{directory} has no native top-level MTP tensors"
    by_shard: dict[str, list[str]] = {}
    for name in selected:
        by_shard.setdefault(weight_map[name], []).append(name)
    try:
        from safetensors import safe_open
    except ImportError as exc:
        return None, f"safetensors is unavailable: {exc}"

    parameters = 0
    try:
        for filename, shard_names in by_shard.items():
            path = directory / filename
            with safe_open(path, framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                for name in shard_names:
                    if name not in available:
                        return None, f"{path} is missing indexed tensor {name!r}"
                    shape = handle.get_slice(name).get_shape()
                    parameters += math.prod(int(dimension) for dimension in shape)
    except (OSError, RuntimeError, ValueError) as exc:
        return None, f"could not inspect MTP safetensors headers in {directory}: {exc}"
    return parameters, None


def _text_model_component(
    name: str,
    root: str | Path,
    *,
    causal_lm: bool,
    bf16: bool,
) -> tuple[MemoryComponent | None, str | None]:
    parameters, error = _text_checkpoint_parameter_count(root, causal_lm=causal_lm)
    if parameters is None:
        # Keep inspection useful for synthetic manifests and partially staged
        # projects, but label the fallback honestly as a storage proxy.
        return _manifest_component(name, root, bf16=bf16)
    bytes_per_parameter = 2 if bf16 else 4
    return (
        MemoryComponent(
            name=name,
            parameter_count=parameters,
            bytes=parameters * bytes_per_parameter,
            dtype_or_state="BF16 parameters" if bf16 else "FP32 parameters",
            source=f"exact safetensors text tensor shapes in {Path(root)}",
        ),
        error,
    )


def estimate_static_training_memory(
    config: TrainConfig,
    *,
    world_size: int = 1,
) -> StaticMemoryEstimate:
    """Estimate known resident parameter/state memory without constructing them.

    The result is a runtime lower bound.  When model shards are present, text
    parameter counts come from safetensors headers and exclude vision tensors.
    Native MTP tensors are counted only when ``losses.mtp`` enables that graph.
    Synthetic or partially staged text sources fall back to manifest file-size
    proxies.
    """

    if isinstance(world_size, bool) or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    adapter_optimizer = str(
        getattr(getattr(config, "optimizer", None), "adapter_optimizer", "adamw")
    )
    if adapter_optimizer not in {"adamw", "muon"}:
        raise ValueError(f"unsupported adapter optimizer: {adapter_optimizer!r}")
    if adapter_optimizer == "muon" and config.stage != "dense-oracle":
        raise ValueError("Muon adapter memory estimation requires dense-oracle stage")
    if adapter_optimizer == "muon" and world_size != 1:
        raise ValueError(
            "Muon adapter memory estimation requires world_size=1 because sharded "
            "matrix orthogonalization is unsupported"
        )
    architecture = config.architecture
    active_layers = len(architecture.active_layers())
    total_layers = int(architecture.student_layers)
    frozen_bytes_per_parameter = 2 if config.runtime.bf16 else 4
    components: list[MemoryComponent] = []
    notes: list[str] = []

    backbone, error = _text_model_component(
        "frozen_backbone",
        config.sources.backbone.local_path,
        causal_lm=True,
        bf16=config.runtime.bf16,
    )
    if backbone is not None:
        components.append(backbone)
    elif error is not None:
        notes.append(f"Frozen backbone omitted from known total: {error}")

    if float(getattr(config.losses, "mtp", 0.0)) > 0:
        mtp_parameters, mtp_error = _mtp_checkpoint_parameter_count(
            config.sources.backbone.local_path
        )
        if mtp_parameters is None:
            notes.append(f"Frozen native MTP omitted from known total: {mtp_error}")
        else:
            components.append(
                MemoryComponent(
                    name="frozen_native_mtp",
                    parameter_count=mtp_parameters,
                    bytes=mtp_parameters * frozen_bytes_per_parameter,
                    dtype_or_state=(
                        "BF16 parameters" if config.runtime.bf16 else "FP32 parameters"
                    ),
                    source=(
                        "exact top-level mtp.* safetensors shapes in "
                        f"{Path(config.sources.backbone.local_path)}"
                    ),
                )
            )

    hidden_teacher_enabled = bool(
        config.stage == "dense-oracle"
        and float(config.losses.hidden_alignment) > 0
        and float(getattr(config.losses, "hidden_alignment_batch_fraction", 1.0)) > 0
    )
    donor_teacher_shared = bool(
        hidden_teacher_enabled
        and world_size == 1
        and getattr(architecture, "expert_initialization", "donor") == "donor"
    )
    teacher_cpu_offload = bool(
        donor_teacher_shared and getattr(config.runtime, "teacher_cpu_offload", False)
    )
    offloaded_cpu_bytes = 0

    if config.stage == "dense-oracle":
        donor_parameters = (
            active_layers
            * 3
            * int(architecture.donor_hidden_size)
            * int(architecture.donor_intermediate_size)
        )
        donor_bytes = donor_parameters * frozen_bytes_per_parameter
        components.append(
            MemoryComponent(
                name="frozen_dense_donor_ffn",
                parameter_count=donor_parameters,
                bytes=(
                    donor_bytes
                    if teacher_cpu_offload
                    else (0 if donor_teacher_shared else donor_bytes)
                ),
                dtype_or_state=(
                    "GPU-resident alias subset of CPU-offloaded teacher"
                    if teacher_cpu_offload
                    else "shared alias of frozen teacher parameters"
                    if donor_teacher_shared
                    else ("BF16 parameters" if config.runtime.bf16 else "FP32 parameters")
                ),
                source=(
                    "mapped MLP Parameters kept on GPU while teacher-only state is offloaded"
                    if teacher_cpu_offload
                    else "zero-copy mapped MLP Parameters in frozen text teacher"
                    if donor_teacher_shared
                    else "3 * active_layers * donor_hidden_size * donor_intermediate_size"
                ),
            )
        )
        if teacher_cpu_offload:
            notes.append(
                "Single-device mapped donor FFNs remain GPU-resident while teacher-exclusive "
                "state is a CPU shadow staged only for hidden-alignment optimizer steps."
            )
        elif donor_teacher_shared:
            notes.append(
                "Single-device dense donor FFNs alias the mapped frozen teacher MLP "
                "Parameters; their 6.75 GiB storage is counted once in the teacher body."
            )
        channel_map_values = active_layers * int(architecture.donor_intermediate_size)
        components.append(
            MemoryComponent(
                name="dense_channel_map_buffers",
                parameter_count=None,
                bytes=channel_map_values * 8,
                dtype_or_state="int64 buffers",
                source="active_layers * donor_intermediate_size",
            )
        )
        adapter_parameters = (
            active_layers
            * 2
            * int(architecture.student_hidden_size)
            * int(architecture.donor_hidden_size)
        )
        scale_parameters = active_layers
        trainable_parameters = adapter_parameters + scale_parameters
    elif config.stage == "sparse":
        folded_parameters = (
            active_layers
            * 3
            * int(architecture.num_experts)
            * int(architecture.student_hidden_size)
            * int(architecture.expert_intermediate_size)
        )
        components.append(
            MemoryComponent(
                name="frozen_folded_experts",
                parameter_count=folded_parameters,
                bytes=folded_parameters * frozen_bytes_per_parameter,
                dtype_or_state="BF16 parameters" if config.runtime.bf16 else "FP32 parameters",
                source=(
                    "3 * active_layers * num_experts * student_hidden_size "
                    "* expert_intermediate_size"
                ),
            )
        )
        rank = int(architecture.lora_rank)
        lora_parameters = (
            active_layers
            * 3
            * int(architecture.num_experts)
            * rank
            * (int(architecture.student_hidden_size) + int(architecture.expert_intermediate_size))
        )
        router_parameters = (
            active_layers * int(architecture.num_experts) * int(architecture.student_hidden_size)
        )
        adapter_parameters = 0
        scale_parameters = active_layers
        trainable_parameters = lora_parameters + router_parameters + active_layers
    else:
        raise ValueError(f"unsupported training stage: {config.stage!r}")

    if hidden_teacher_enabled:
        teacher, error = _text_model_component(
            "frozen_hidden_alignment_teacher",
            config.sources.teacher.local_path,
            causal_lm=False,
            bf16=config.runtime.bf16,
        )
        if teacher is not None:
            if teacher_cpu_offload:
                teacher_only_bytes = max(teacher.bytes - donor_bytes, 0)
                teacher_only_parameters = (
                    max(teacher.parameter_count - donor_parameters, 0)
                    if teacher.parameter_count is not None
                    else None
                )
                offloaded_cpu_bytes += teacher_only_bytes
                components.append(
                    MemoryComponent(
                        name="frozen_hidden_alignment_teacher_cpu_shadow",
                        parameter_count=teacher_only_parameters,
                        bytes=teacher_only_bytes,
                        dtype_or_state=(
                            "BF16 CPU shadow; temporarily staged on alignment steps"
                            if config.runtime.bf16
                            else "FP32 CPU shadow; temporarily staged on alignment steps"
                        ),
                        source=(
                            f"teacher text body minus {active_layers} mapped donor FFNs"
                        ),
                    )
                )
                notes.append(
                    f"Alignment steps temporarily add {teacher_only_bytes / _GIB:.3f} GiB "
                    "of staged teacher Parameters to GPU static residency."
                )
            else:
                components.append(teacher)
        elif error is not None:
            notes.append(f"Hidden-alignment teacher omitted from known total: {error}")

    trainable_bytes = trainable_parameters * 4
    for name, multiplier, state in (
        ("trainable_parameters", 1, "FP32 parameters"),
        ("trainable_gradients", 1, "FP32 gradients"),
    ):
        components.append(
            MemoryComponent(
                name=name,
                parameter_count=trainable_parameters * multiplier,
                bytes=trainable_bytes * multiplier,
                dtype_or_state=state,
                source="exact Twen transfer-module parameter formula",
            )
        )
    if adapter_optimizer == "adamw":
        components.append(
            MemoryComponent(
                name="adam_first_and_second_moments",
                parameter_count=trainable_parameters * 2,
                bytes=trainable_bytes * 2,
                dtype_or_state="2 x FP32 Adam moments",
                source="exact Twen transfer-module parameter formula",
            )
        )
    else:
        # Validated Muon configurations are dense-only: every adapter matrix
        # has one FP32 momentum buffer while each scalar scale retains AdamW's
        # two FP32 moments.
        components.extend(
            (
                MemoryComponent(
                    name="muon_adapter_momentum",
                    parameter_count=adapter_parameters,
                    bytes=adapter_parameters * 4,
                    dtype_or_state="1 x FP32 Muon momentum",
                    source="2 * active_layers * student_hidden_size * donor_hidden_size",
                ),
                MemoryComponent(
                    name="adam_scale_first_and_second_moments",
                    parameter_count=scale_parameters * 2,
                    bytes=scale_parameters * 2 * 4,
                    dtype_or_state="2 x FP32 Adam moments",
                    source="2 * active_layers scalar branch-scale states",
                ),
            )
        )
        notes.append(
            "Muon stores one adapter momentum buffer; scalar branch scales retain "
            "two AdamW moments."
        )

    aggregate = sum(component.bytes for component in components)
    gpu_resident_aggregate = aggregate - offloaded_cpu_bytes
    sharding = str(config.runtime.sharding)
    divisor = world_size if sharding == "fsdp2" and world_size > 1 else 1
    per_device = (gpu_resident_aggregate + divisor - 1) // divisor
    if divisor > 1:
        notes.append(
            "Per-device static bytes assume ideal FSDP2 sharding; transient all-gathers and "
            "uneven layer sizes are excluded."
        )
    elif world_size > 1 and sharding == "ddp":
        notes.append("DDP replicates the known static model/optimizer state on every device.")
    if config.stage == "dense-oracle":
        notes.append(
            "Production loss projection is token-chunked; full-sequence student/anchor "
            "vocabulary logits are not resident at once."
        )

    return StaticMemoryEstimate(
        stage=str(config.stage),
        active_layers=active_layers,
        total_layers=total_layers,
        world_size=world_size,
        sharding=sharding,
        components=tuple(components),
        aggregate_known_static_bytes=aggregate,
        estimated_per_device_static_bytes=per_device,
        is_runtime_lower_bound=True,
        excludes=_LOWER_BOUND_EXCLUDES,
        notes=tuple(notes),
    )


def _torch_and_gpus() -> tuple[TorchInfo, tuple[GPUInfo, ...]]:
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        return (
            TorchInfo(False, None, False, None, None, None, 0, f"{type(exc).__name__}: {exc}"),
            (),
        )

    cuda = torch.cuda
    try:
        available = bool(cuda.is_available())
    except Exception:
        available = False
    try:
        count = int(cuda.device_count()) if available else 0
    except Exception:
        count = 0
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    hip_version = getattr(getattr(torch, "version", None), "hip", None)
    try:
        cudnn_version = torch.backends.cudnn.version()
    except Exception:
        cudnn_version = None

    gpus: list[GPUInfo] = []
    for index in range(count):
        try:
            properties = cuda.get_device_properties(index)
            major = getattr(properties, "major", None)
            minor = getattr(properties, "minor", None)
            if major is None or minor is None:
                major, minor = cuda.get_device_capability(index)
            capability = f"{int(major)}.{int(minor)}"
            # Native CUDA BF16 tensor cores start with Ampere (SM 8.x).  ROCm
            # does not expose a CUDA-style capability with the same semantics.
            bf16_supported: bool | None
            if cuda_version is not None:
                bf16_supported = int(major) >= 8
            elif hip_version is not None:
                try:
                    bf16_supported = bool(cuda.is_bf16_supported())
                except Exception:
                    bf16_supported = None
            else:
                bf16_supported = None
            gpus.append(
                GPUInfo(
                    index=index,
                    name=str(getattr(properties, "name", None) or cuda.get_device_name(index)),
                    total_memory_bytes=int(properties.total_memory),
                    compute_capability=capability,
                    bf16_supported=bf16_supported,
                    multiprocessor_count=(
                        int(properties.multi_processor_count)
                        if getattr(properties, "multi_processor_count", None) is not None
                        else None
                    ),
                )
            )
        except Exception:
            # A partially visible/broken device should not make the read-only
            # host report unusable.  TorchInfo still records the visible count.
            continue

    return (
        TorchInfo(
            installed=True,
            version=str(getattr(torch, "__version__", "unknown")),
            cuda_available=available,
            cuda_runtime_version=str(cuda_version) if cuda_version is not None else None,
            hip_runtime_version=str(hip_version) if hip_version is not None else None,
            cudnn_version=int(cudnn_version) if cudnn_version is not None else None,
            visible_device_count=count,
        ),
        tuple(gpus),
    )


def _warnings(
    config: TrainConfig | None,
    torch_info: TorchInfo,
    gpus: tuple[GPUInfo, ...],
    kernels: AcceleratorKernels,
    estimate: StaticMemoryEstimate | None,
) -> tuple[HardwareWarning, ...]:
    result: list[HardwareWarning] = []
    if not torch_info.installed:
        result.append(
            HardwareWarning("error", "torch_unavailable", "PyTorch could not be imported.")
        )
    elif not torch_info.cuda_available or not gpus:
        result.append(
            HardwareWarning(
                "error",
                "cuda_unavailable",
                "No queryable CUDA device is available; real Twen training requires CUDA.",
            )
        )
    if config is None:
        return tuple(result)

    if not kernels.qwen35_fast_path_ready:
        result.append(
            HardwareWarning(
                "high",
                "qwen35_fast_path_missing",
                "Qwen3.5 CUDA fast path requires both flash-linear-attention and "
                "causal-conv1d; the PyTorch fallback materially underutilizes this GPU.",
            )
        )

    if config.runtime.bf16:
        unsupported = [gpu.index for gpu in gpus if gpu.bf16_supported is not True]
        if unsupported:
            result.append(
                HardwareWarning(
                    "error",
                    "bf16_unsupported",
                    f"runtime.bf16=true but BF16 support was not confirmed for GPU(s) {unsupported}.",
                )
            )

    active = len(config.architecture.active_layers())
    full_dense = config.stage == "dense-oracle" and active == config.architecture.student_layers
    hidden_alignment = config.stage == "dense-oracle" and float(config.losses.hidden_alignment) > 0
    single_32gb_class = len(gpus) == 1 and 28 * _GIB <= gpus[0].total_memory_bytes <= 36 * _GIB
    if full_dense and hidden_alignment and single_32gb_class:
        donor_shared = bool(
            estimate is not None
            and any(
                component.name == "frozen_dense_donor_ffn"
                and "alias" in component.dtype_or_state
                for component in estimate.components
            )
        )
        teacher_offload = bool(getattr(config.runtime, "teacher_cpu_offload", False))
        result.append(
            HardwareWarning(
                "info" if donor_shared else "high",
                (
                    "single_32gb_dense_teacher_cpu_offload"
                    if donor_shared and teacher_offload
                    else "single_32gb_dense_shared_teacher"
                    if donor_shared
                    else "single_32gb_full_dense_hidden_alignment"
                ),
                (
                    "The full 24-layer single-device design keeps mapped teacher MLP aliases "
                    "on GPU and the teacher-exclusive CPU shadow off device between alignment "
                    "steps.  The optimizer-free 4096-token graph smoke remains the authoritative "
                    "staged-step activation/workspace fit check."
                    if teacher_offload
                    else "The full 24-layer single-device design reuses mapped teacher MLP weights "
                    "and token-chunks LM-head losses.  Static state now leaves material headroom, "
                    "but the optimizer-free 4096-token graph smoke remains the authoritative "
                    "activation/workspace fit check."
                    if donor_shared
                    else "A full-layer dense run with an independently resident donor and "
                    "hidden-alignment teacher is high risk on one 32GB-class GPU."
                ),
            )
        )
    elif config.stage == "dense-oracle" and active < config.architecture.student_layers:
        result.append(
            HardwareWarning(
                "info",
                "dense_active_layer_poc",
                f"This is an active-layer PoC ({active}/{config.architecture.student_layers} "
                "layers): donor FFN and adapter state shrink roughly with active layers, but the "
                "backbone and hidden-alignment teacher do not.  A successful PoC does not prove "
                "the full dense run will fit.",
            )
        )

    if estimate is not None and len(gpus) == 1:
        ratio = estimate.estimated_per_device_static_bytes / max(gpus[0].total_memory_bytes, 1)
        if ratio >= 0.8:
            result.append(
                HardwareWarning(
                    "high",
                    "static_memory_headroom_low",
                    f"Known static state is about {ratio:.0%} of device memory before all "
                    "excluded runtime allocations.",
                )
            )
    return tuple(result)


def inspect_hardware(config: TrainConfig | None = None) -> HardwareReport:
    """Collect a side-effect-free host/device report and optional static estimate."""

    cpu = _inspect_cpu()
    torch_info, gpus = _torch_and_gpus()
    estimate = None
    kernels = _accelerator_kernels()
    if config is not None:
        estimate = estimate_static_training_memory(config, world_size=max(1, len(gpus)))
    return HardwareReport(
        cpu=cpu,
        torch=torch_info,
        gpus=gpus,
        allocator=_allocator_environment(),
        kernels=kernels,
        static_memory=estimate,
        warnings=_warnings(config, torch_info, gpus, kernels, estimate),
    )
