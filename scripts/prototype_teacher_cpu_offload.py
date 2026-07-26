#!/usr/bin/env python3
"""Isolated single-GPU prototype for selective Qwen3.5 teacher CPU residency.

This script never creates an optimizer and does not alter the production
builder or engine.  It keeps the mapped donor FFN Parameters on CUDA by exact
object alias while moving only teacher-exclusive Parameters and buffers for a
short hidden-state forward, then returns them to CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base/dense-oracle.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument(
        "--retain-cpu-shadow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="retain frozen CPU storage so post-forward restore requires no D2H copy",
    )
    parser.add_argument("--output", default=None)
    return parser


def _render(result: dict[str, Any], output: str | None) -> None:
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    print(text, end="")


def _mapped_parameter_names(layer_mapping: Iterable[int]) -> frozenset[str]:
    return frozenset(
        f"layers.{layer}.mlp.{projection}.weight"
        for layer in layer_mapping
        for projection in PROJECTIONS
    )


def _checkpoint_inventory(
    root: str | Path,
    mapped_parameter_names: frozenset[str],
    *,
    runtime_element_size: int,
) -> dict[str, Any]:
    from safetensors import safe_open

    from twen.model_loading import SafetensorCheckpoint

    checkpoint = SafetensorCheckpoint(root)
    dtype_width = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8}
    serialized_bytes: Counter[str] = Counter()
    runtime_bytes: Counter[str] = Counter()
    elements: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    parameter_count = 0
    for shard_path, source_names in checkpoint.grouped_by_file():
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for source_name in source_names:
                prefix = "model.language_model."
                if not source_name.startswith(prefix):
                    continue
                model_name = source_name.removeprefix(prefix)
                value = handle.get_slice(source_name)
                shape = tuple(value.get_shape())
                dtype = value.get_dtype()
                if dtype not in dtype_width:
                    raise RuntimeError(f"unsupported checkpoint dtype {dtype}: {source_name}")
                count = math.prod(shape)
                category = "mapped_donor_ffn" if model_name in mapped_parameter_names else "teacher_only"
                serialized_bytes[category] += count * dtype_width[dtype]
                runtime_bytes[category] += count * runtime_element_size
                elements[category] += count
                dtypes[dtype] += 1
                parameter_count += 1
    total_serialized_bytes = sum(serialized_bytes.values())
    total_runtime_bytes = sum(runtime_bytes.values())
    return {
        "text_parameter_tensors": parameter_count,
        "dtype_tensor_counts": dict(sorted(dtypes.items())),
        "mapped_donor_ffn": {
            "elements": elements["mapped_donor_ffn"],
            "runtime_bytes": runtime_bytes["mapped_donor_ffn"],
            "runtime_gib": runtime_bytes["mapped_donor_ffn"] / 2**30,
            "checkpoint_serialized_bytes": serialized_bytes["mapped_donor_ffn"],
        },
        "teacher_only": {
            "elements": elements["teacher_only"],
            "runtime_bytes": runtime_bytes["teacher_only"],
            "runtime_gib": runtime_bytes["teacher_only"] / 2**30,
            "checkpoint_serialized_bytes": serialized_bytes["teacher_only"],
        },
        "text_total": {
            "elements": sum(elements.values()),
            "runtime_bytes": total_runtime_bytes,
            "runtime_gib": total_runtime_bytes / 2**30,
            "checkpoint_serialized_bytes": total_serialized_bytes,
        },
        "teacher_only_round_trip": {
            "runtime_bytes": 2 * runtime_bytes["teacher_only"],
            "runtime_gib": 2 * runtime_bytes["teacher_only"] / 2**30,
        },
    }


def _storage_alias(left: Any, right: Any) -> bool:
    return bool(
        left is right
        and left.device == right.device
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        and left.storage_offset() == right.storage_offset()
        and tuple(left.shape) == tuple(right.shape)
    )


def _toy_module_to_alias_contract(device: Any, *, torch: Any) -> dict[str, Any]:
    """Prove both useful and dangerous sides of current ``Module.to`` semantics."""

    parameter = torch.nn.Parameter(torch.randn(4, 4), requires_grad=False)
    teacher_owner = torch.nn.Module()
    donor_owner = torch.nn.Module()
    teacher_owner.register_parameter("weight", parameter)
    donor_owner.register_parameter("weight", parameter)
    initial_id = id(parameter)
    donor_owner.to(device=device)
    cuda_alias_preserved = (
        teacher_owner.weight is donor_owner.weight
        and id(teacher_owner.weight) == initial_id
        and teacher_owner.weight.device == device
    )
    teacher_owner.to(device="cpu")
    whole_teacher_offload_moves_donor = (
        teacher_owner.weight is donor_owner.weight
        and teacher_owner.weight.device.type == "cpu"
        and donor_owner.weight.device.type == "cpu"
    )
    return {
        "student_to_cuda_preserves_external_parameter_object_alias": cuda_alias_preserved,
        "whole_teacher_to_cpu_also_moves_aliased_donor": whole_teacher_offload_moves_donor,
        "overwrite_module_params_on_conversion": bool(
            torch.__future__.get_overwrite_module_params_on_conversion()
        ),
        "swap_module_params_on_conversion": bool(
            torch.__future__.get_swap_module_params_on_conversion()
        ),
    }


def _unique_bytes(values: Iterable[Any]) -> int:
    seen: set[int] = set()
    total = 0
    for value in values:
        key = id(value)
        if key in seen:
            continue
        seen.add(key)
        total += int(value.numel() * value.element_size())
    return total


def _device_inventory(
    model: Any,
    mapped_names: frozenset[str],
) -> dict[str, Any]:
    parameters = dict(model.named_parameters())
    teacher_only = [value for name, value in parameters.items() if name not in mapped_names]
    mapped = [parameters[name] for name in mapped_names]
    buffers = tuple(model.buffers())
    return {
        "mapped_parameter_bytes": _unique_bytes(mapped),
        "teacher_only_parameter_bytes": _unique_bytes(teacher_only),
        "buffer_bytes": _unique_bytes(buffers),
        "mapped_devices": dict(sorted(Counter(str(value.device) for value in mapped).items())),
        "teacher_only_devices": dict(
            sorted(Counter(str(value.device) for value in teacher_only).items())
        ),
        "buffer_devices": dict(sorted(Counter(str(value.device) for value in buffers).items())),
    }


def _selective_move(
    model: Any,
    parameter_names: frozenset[str],
    target: Any,
    *,
    torch: Any,
) -> dict[str, Any]:
    """Move frozen teacher-only state while preserving every Parameter object id."""

    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("selective offload requires a fully frozen teacher")
    parameters = dict(model.named_parameters())
    missing = sorted(parameter_names - parameters.keys())
    if missing:
        raise RuntimeError(f"teacher-only parameter names disappeared: {missing[:3]}")
    original_ids = {name: id(parameters[name]) for name in parameter_names}
    moved_bytes = 0
    for name in sorted(parameter_names):
        parameter = parameters[name]
        if parameter.device == target:
            continue
        moved_bytes += int(parameter.numel() * parameter.element_size())
        replacement = torch.nn.Parameter(
            parameter.detach().to(device=target),
            requires_grad=False,
        )
        torch.utils.swap_tensors(parameter, replacement)
    buffer_bytes = 0
    seen_buffers: set[int] = set()
    for buffer in model.buffers():
        if id(buffer) in seen_buffers or buffer.device == target:
            continue
        seen_buffers.add(id(buffer))
        buffer_bytes += int(buffer.numel() * buffer.element_size())
        torch.utils.swap_tensors(buffer, buffer.to(device=target))
    current = dict(model.named_parameters())
    ids_preserved = all(id(current[name]) == original_ids[name] for name in parameter_names)
    return {
        "parameter_bytes": moved_bytes,
        "buffer_bytes": buffer_bytes,
        "transferred_bytes": moved_bytes + buffer_bytes,
        "parameter_object_ids_preserved": ids_preserved,
    }


def _stage_with_cpu_shadow(
    model: Any,
    parameter_names: frozenset[str],
    target: Any,
    *,
    torch: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Install CUDA copies while retaining the frozen original CPU storages."""

    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("CPU-shadow staging requires a fully frozen teacher")
    parameters = dict(model.named_parameters())
    parameter_shadows: dict[str, Any] = {}
    parameter_bytes = 0
    original_ids = {name: id(parameters[name]) for name in parameter_names}
    for name in sorted(parameter_names):
        parameter = parameters[name]
        if parameter.device.type != "cpu":
            raise RuntimeError(f"teacher-only parameter is not CPU resident before stage: {name}")
        parameter_bytes += int(parameter.numel() * parameter.element_size())
        cuda_value = torch.nn.Parameter(
            parameter.detach().to(device=target),
            requires_grad=False,
        )
        torch.utils.swap_tensors(parameter, cuda_value)
        # After the swap, ``cuda_value`` owns the original CPU TensorImpl.
        parameter_shadows[name] = cuda_value

    buffer_shadows: list[tuple[Any, Any]] = []
    buffer_bytes = 0
    seen_buffers: set[int] = set()
    for buffer in model.buffers():
        if id(buffer) in seen_buffers:
            continue
        seen_buffers.add(id(buffer))
        if buffer.device.type != "cpu":
            raise RuntimeError("teacher buffer is not CPU resident before stage")
        buffer_bytes += int(buffer.numel() * buffer.element_size())
        cuda_value = buffer.to(device=target)
        torch.utils.swap_tensors(buffer, cuda_value)
        buffer_shadows.append((buffer, cuda_value))
    current = dict(model.named_parameters())
    ids_preserved = all(id(current[name]) == original_ids[name] for name in parameter_names)
    report = {
        "parameter_bytes": parameter_bytes,
        "buffer_bytes": buffer_bytes,
        "transferred_bytes": parameter_bytes + buffer_bytes,
        "cpu_shadow_bytes": parameter_bytes + buffer_bytes,
        "parameter_object_ids_preserved": ids_preserved,
    }
    state = {
        "parameter_shadows": parameter_shadows,
        "buffer_shadows": buffer_shadows,
        "parameter_bytes": parameter_bytes,
        "buffer_bytes": buffer_bytes,
    }
    return report, state


def _restore_cpu_shadow(
    model: Any,
    state: dict[str, Any],
    *,
    torch: Any,
) -> dict[str, Any]:
    """Swap frozen CPU storages back and drop CUDA copies without a D2H copy."""

    parameters = dict(model.named_parameters())
    original_ids = {name: id(parameters[name]) for name in state["parameter_shadows"]}
    for name, cpu_shadow in state["parameter_shadows"].items():
        torch.utils.swap_tensors(parameters[name], cpu_shadow)
    for active_buffer, cpu_shadow in state["buffer_shadows"]:
        torch.utils.swap_tensors(active_buffer, cpu_shadow)
    released_bytes = int(state["parameter_bytes"] + state["buffer_bytes"])
    state["parameter_shadows"].clear()
    state["buffer_shadows"].clear()
    current = dict(model.named_parameters())
    ids_preserved = all(id(current[name]) == original_ids[name] for name in original_ids)
    return {
        "parameter_bytes": 0,
        "buffer_bytes": 0,
        "transferred_bytes": 0,
        "released_cuda_bytes": released_bytes,
        "parameter_object_ids_preserved": ids_preserved,
    }


def _mapped_alias_report(
    teacher: Any,
    built: Any,
    layer_mapping: tuple[int, ...],
) -> dict[str, Any]:
    alias_count = 0
    expected = 0
    for student_layer, module in zip(
        built.student_layer_indices,
        built.transfer_modules,
        strict=True,
    ):
        donor_layer = layer_mapping[student_layer]
        teacher_mlp = teacher.layers[donor_layer].mlp
        transfer = module.transfer_mlp
        for projection, transfer_name in zip(
            PROJECTIONS,
            ("gate_weight", "up_weight", "down_weight"),
            strict=True,
        ):
            expected += 1
            alias_count += int(
                _storage_alias(
                    getattr(teacher_mlp, projection).weight,
                    getattr(transfer, transfer_name),
                )
            )
    return {
        "expected": expected,
        "exact_object_and_storage_aliases": alias_count,
        "all_exact": alias_count == expected,
    }


def _cuda_memory(device: Any, *, torch: Any) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "free_bytes": int(free),
        "device_total_bytes": int(total),
    }


def _timed_move(function: Any, device: Any, *, torch: Any) -> tuple[dict[str, Any], float]:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    report = function()
    torch.cuda.synchronize(device)
    return report, time.perf_counter() - started


def _run_real_model(
    args: argparse.Namespace,
    config: Any,
    layer_mapping: tuple[int, ...],
    mapped_names: frozenset[str],
    checkpoint_inventory: dict[str, Any],
) -> dict[str, Any]:
    import torch

    from twen.model_loading import freeze_module, load_qwen35_text_model
    from twen.training.builder import build_transfer_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --metadata-only in a restricted environment")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(config.runtime.seed)
    torch.cuda.manual_seed_all(config.runtime.seed)
    dtype = torch.bfloat16 if config.runtime.bf16 else torch.float32
    toy_contract = _toy_module_to_alias_contract(device, torch=torch)
    if not toy_contract["student_to_cuda_preserves_external_parameter_object_alias"]:
        raise RuntimeError("this PyTorch build does not preserve the required external alias")
    if not toy_contract["whole_teacher_to_cpu_also_moves_aliased_donor"]:
        raise RuntimeError("unexpected Module.to alias semantics")

    memory_start = _cuda_memory(device, torch=torch)
    print("loading frozen teacher on CPU...", file=sys.stderr, flush=True)
    load_started = time.perf_counter()
    teacher = load_qwen35_text_model(
        config.sources.teacher.local_path,
        dtype=dtype,
        device="cpu",
    )
    freeze_module(teacher)
    teacher_load_seconds = time.perf_counter() - load_started

    print("building student and moving mapped donor aliases to CUDA...", file=sys.stderr, flush=True)
    build_started = time.perf_counter()
    built = build_transfer_model(
        config,
        device=str(device),
        dtype=dtype,
        donor_text_model=teacher,
    )
    torch.cuda.synchronize(device)
    build_seconds = time.perf_counter() - build_started
    aliases_after_build = _mapped_alias_report(teacher, built, layer_mapping)
    if not aliases_after_build["all_exact"]:
        raise RuntimeError("student CUDA placement broke mapped donor/teacher aliases")
    if not built.donor_teacher_shared:
        raise RuntimeError("builder did not mark donor/teacher storage as shared")

    all_parameter_names = frozenset(name for name, _ in teacher.named_parameters())
    teacher_only_names = all_parameter_names - mapped_names
    split_inventory = _device_inventory(teacher, mapped_names)
    expected_teacher_only_bytes = checkpoint_inventory["teacher_only"]["runtime_bytes"]
    if split_inventory["teacher_only_parameter_bytes"] != expected_teacher_only_bytes:
        raise RuntimeError("live teacher-only bytes disagree with checkpoint inventory")
    if (
        split_inventory["mapped_parameter_bytes"]
        != checkpoint_inventory["mapped_donor_ffn"]["runtime_bytes"]
    ):
        raise RuntimeError("live mapped donor bytes disagree with checkpoint inventory")
    if split_inventory["mapped_devices"] != {str(device): len(mapped_names)}:
        raise RuntimeError(f"mapped donors are not all resident on {device}")
    if split_inventory["teacher_only_devices"] != {"cpu": len(teacher_only_names)}:
        raise RuntimeError("teacher-exclusive parameters are not initially CPU resident")
    memory_split = _cuda_memory(device, torch=torch)

    ids_before = {name: id(value) for name, value in teacher.named_parameters()}
    print("staging teacher-only parameters on CUDA...", file=sys.stderr, flush=True)
    shadow_state = None
    if args.retain_cpu_shadow:
        stage_payload, stage_seconds = _timed_move(
            lambda: _stage_with_cpu_shadow(
                teacher,
                teacher_only_names,
                device,
                torch=torch,
            ),
            device,
            torch=torch,
        )
        stage_report, shadow_state = stage_payload
    else:
        stage_report, stage_seconds = _timed_move(
            lambda: _selective_move(teacher, teacher_only_names, device, torch=torch),
            device,
            torch=torch,
        )
    memory_staged = _cuda_memory(device, torch=torch)
    aliases_staged = _mapped_alias_report(teacher, built, layer_mapping)
    staged_inventory = _device_inventory(teacher, mapped_names)
    if not aliases_staged["all_exact"]:
        raise RuntimeError("selective H2D staging broke mapped donor aliases")

    print("running short teacher hidden-state forward...", file=sys.stderr, flush=True)
    input_ids = torch.arange(args.sequence_length, device=device).remainder(
        int(teacher.config.vocab_size)
    ).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    torch.cuda.synchronize(device)
    forward_started = time.perf_counter()
    with torch.no_grad():
        outputs = teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_started
    hidden_shape = tuple(outputs.last_hidden_state.shape)
    hidden_count = len(outputs.hidden_states)
    hidden_finite = bool(torch.isfinite(outputs.last_hidden_state).all().item())
    del outputs, input_ids, attention_mask

    print("offloading teacher-only parameters to CPU...", file=sys.stderr, flush=True)
    if shadow_state is not None:
        offload_report, offload_seconds = _timed_move(
            lambda: _restore_cpu_shadow(teacher, shadow_state, torch=torch),
            device,
            torch=torch,
        )
    else:
        offload_report, offload_seconds = _timed_move(
            lambda: _selective_move(
                teacher,
                teacher_only_names,
                torch.device("cpu"),
                torch=torch,
            ),
            device,
            torch=torch,
        )
    memory_offloaded = _cuda_memory(device, torch=torch)
    aliases_offloaded = _mapped_alias_report(teacher, built, layer_mapping)
    offloaded_inventory = _device_inventory(teacher, mapped_names)
    ids_after = {name: id(value) for name, value in teacher.named_parameters()}

    identity_preserved = all(ids_after[name] == ids_before[name] for name in ids_before)
    bytes_staged = stage_report["transferred_bytes"]
    bytes_offloaded = offload_report["transferred_bytes"]
    ok = bool(
        aliases_staged["all_exact"]
        and aliases_offloaded["all_exact"]
        and stage_report["parameter_object_ids_preserved"]
        and offload_report["parameter_object_ids_preserved"]
        and identity_preserved
        and hidden_finite
        and staged_inventory["teacher_only_devices"] == {str(device): len(teacher_only_names)}
        and offloaded_inventory["teacher_only_devices"] == {"cpu": len(teacher_only_names)}
    )
    return {
        "ok": ok,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "fla_tilelang": os.environ.get("FLA_TILELANG"),
        "retain_cpu_shadow": args.retain_cpu_shadow,
        "toy_module_to_contract": toy_contract,
        "teacher_load_seconds": teacher_load_seconds,
        "student_build_and_cuda_place_seconds": build_seconds,
        "mapped_alias_after_student_build": aliases_after_build,
        "split_inventory": split_inventory,
        "staged_inventory": staged_inventory,
        "offloaded_inventory": offloaded_inventory,
        "all_teacher_parameter_object_ids_preserved": identity_preserved,
        "stage": {
            **stage_report,
            "seconds": stage_seconds,
            "effective_gb_per_second": bytes_staged / stage_seconds / 1e9,
        },
        "short_teacher_forward": {
            "sequence_length": args.sequence_length,
            "seconds": forward_seconds,
            "last_hidden_state_shape": hidden_shape,
            "hidden_state_tensors": hidden_count,
            "finite": hidden_finite,
        },
        "offload": {
            **offload_report,
            "seconds": offload_seconds,
            "effective_gb_per_second": (
                bytes_offloaded / offload_seconds / 1e9 if bytes_offloaded else None
            ),
        },
        "mapped_alias_staged": aliases_staged,
        "mapped_alias_offloaded": aliases_offloaded,
        "memory": {
            "start": memory_start,
            "split": memory_split,
            "staged": memory_staged,
            "offloaded": memory_offloaded,
            "staging_allocated_delta_bytes": (
                memory_staged["allocated_bytes"] - memory_split["allocated_bytes"]
            ),
            "offload_allocated_drop_bytes": (
                memory_staged["allocated_bytes"] - memory_offloaded["allocated_bytes"]
            ),
        },
        "no_optimizer_created": True,
        "no_optimizer_steps": True,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    from twen.config import load_train_config
    from twen.training.builder import load_layer_mapping

    if args.sequence_length <= 0:
        raise ValueError("sequence-length must be positive")
    config = load_train_config(args.config)
    if config.stage != "dense-oracle":
        raise ValueError("prototype requires a dense-oracle config")
    layer_mapping = load_layer_mapping(
        config.architecture.layer_map_path,
        config.architecture.student_layers,
    )
    mapped_names = _mapped_parameter_names(layer_mapping)
    checkpoint_inventory = _checkpoint_inventory(
        config.sources.teacher.local_path,
        mapped_names,
        runtime_element_size=2 if config.runtime.bf16 else 4,
    )
    result: dict[str, Any] = {
        "ok": True,
        "scope": "isolated_single_gpu_teacher_cpu_offload_prototype",
        "config": str(Path(args.config).resolve()),
        "teacher": str(Path(config.sources.teacher.local_path).resolve()),
        "mapped_donor_layers": list(layer_mapping),
        "mapped_projection_parameter_names": len(mapped_names),
        "checkpoint_inventory": checkpoint_inventory,
        "metadata_only": args.metadata_only,
        "no_optimizer_created": True,
        "no_optimizer_steps": True,
    }
    if not args.metadata_only:
        real = _run_real_model(
            args,
            config,
            layer_mapping,
            mapped_names,
            checkpoint_inventory,
        )
        result["real_model"] = real
        result["ok"] = bool(real["ok"])
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _run(args)
    except Exception as error:
        result = {
            "ok": False,
            "no_optimizer_created": True,
            "no_optimizer_steps": True,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
    _render(result, args.output)
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
