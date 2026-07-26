"""User-invoked folding and native Qwen3.5-MoE export orchestration."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .config import load_train_config
from .model_loading import (
    load_qwen35_mtp,
    load_qwen35_text_causal_lm,
    read_qwen_text_config,
)
from .preflight import PreflightReport, run_training_preflight
from .runtime.checkpoint import CheckpointManager
from .training.builder import build_transfer_model
from .training.stateful import TrainableModelState
from .utils import atomic_write_json, atomic_write_text, sha256_file


def _save_safetensors_atomic(tensors: dict[str, Any], path: Path) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()},
        str(temporary),
    )
    os.replace(temporary, path)


def _load_trainable_only(
    model: Any,
    config: Any,
    report: PreflightReport,
    checkpoint_path: str,
) -> Path:
    run_dir = config.checkpoint.output_dir
    manager = CheckpointManager(run_dir, rank=0, world_size=1)
    loaded = manager.load(
        {"model": TrainableModelState(model)},
        checkpoint_path,
        expected_critical_fingerprint=report.config_fingerprint,
        expected_data_fingerprint=report.data_fingerprint,
        expected_run_id=config.run_id,
        expected_stage=config.stage,
        expected_global_batch_tokens=config.data.global_batch_tokens,
    )
    return loaded.path


def fold_dense_checkpoint(
    *,
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    device: str = "cuda",
) -> dict[str, Any]:
    """Materialize BF16 routed experts after FP32 A/B multiplication."""

    import torch

    from .modeling import fold_expert_weights

    config = load_train_config(config_path)
    if config.stage != "dense-oracle":
        raise ValueError("fold requires a dense-oracle configuration")
    if config.architecture.expert_initialization != "donor":
        raise ValueError("random-control experts are evaluation-only and cannot be folded")
    if len(config.architecture.active_layers()) != config.architecture.student_layers:
        raise ValueError("fold requires a completed 24-layer dense-oracle run")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("fold requested CUDA but CUDA is unavailable")
    report = run_training_preflight(config, world_size=1)
    built = build_transfer_model(config, device=device, dtype=torch.bfloat16)
    loaded_path = _load_trainable_only(
        built.model, config, report, checkpoint_path
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, Any] = {}
    for layer, module in enumerate(built.transfer_modules):
        transfer = module.transfer_mlp
        folded = fold_expert_weights(
            transfer.gate_weight,
            transfer.up_weight,
            transfer.down_weight,
            transfer.adapters.input_adapter,
            transfer.adapters.output_adapter,
            transfer.channel_indices,
            output_dtype=torch.bfloat16,
        )
        tensors[f"layers.{layer}.gate_proj"] = folded.gate_proj.cpu()
        tensors[f"layers.{layer}.up_proj"] = folded.up_proj.cpu()
        tensors[f"layers.{layer}.down_proj"] = folded.down_proj.cpu()
        tensors[f"layers.{layer}.branch_scale"] = transfer.branch_scale.detach().float().cpu()
    artifact = output / "model.safetensors"
    _save_safetensors_atomic(tensors, artifact)
    manifest = {
        "schema_version": 1,
        "kind": "twen_folded_experts",
        "source_checkpoint": str(loaded_path),
        "source_checkpoint_complete_sha256": sha256_file(loaded_path / "COMPLETE"),
        "source_config_fingerprint": report.config_fingerprint,
        "raw_config_fingerprint": config.fingerprint(),
        "track": config.track,
        "backbone_revision": config.sources.backbone.revision,
        "donor_revision": config.sources.donor.revision,
        "source_manifests": {
            "backbone": config.sources.backbone.manifest_sha256,
            "donor": config.sources.donor.manifest_sha256,
        },
        "calibration_artifacts": dict(report.calibration_fingerprints),
        "num_layers": config.architecture.student_layers,
        "num_experts": config.architecture.num_experts,
        "expert_intermediate_size": config.architecture.expert_intermediate_size,
        "dtype": "bfloat16",
        "artifact": artifact.name,
        "artifact_sha256": sha256_file(artifact),
    }
    atomic_write_json(output / "manifest.json", manifest)
    return {
        "artifact": str(artifact),
        "sha256": manifest["artifact_sha256"],
        "manifest": str(output / "manifest.json"),
    }


_TOKENIZER_BUNDLE_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}


def _copy_tokenizer_files(source: Path, target: Path) -> tuple[str, ...]:
    copied = []
    for name in sorted(_TOKENIZER_BUNDLE_FILES):
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, target / name)
            copied.append(name)
    if "tokenizer.json" not in copied and "tokenizer.model" not in copied:
        raise RuntimeError(f"tokenizer source has no tokenizer.json/model: {source}")
    return tuple(copied)


def _prepare_bundle_directory(target: Path, expected_names: set[str]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in target.iterdir():
        stale_atomic = (
            item.is_file()
            and item.name.startswith(".")
            and item.name.endswith((".tmp", ".incomplete"))
        )
        if stale_atomic:
            item.unlink()
            continue
        if not item.is_file() or item.name not in expected_names:
            raise RuntimeError(
                f"export output contains an unexpected stale entry: {item}; "
                "use a fresh directory"
            )


def _bundle_inventory(target: Path, names: set[str]) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for name in sorted(names):
        path = target / name
        if not path.is_file():
            raise RuntimeError(f"export bundle is missing {path}")
        inventory[name] = {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return inventory


def export_native_moe(
    *,
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    device: str = "cpu",
) -> dict[str, Any]:
    """Merge sparse LoRA/scale and emit a standard text-only native MoE checkpoint."""

    import torch

    from .modeling import (
        build_native_moe_config,
        export_native_moe_mtp_state,
        export_native_moe_state,
    )

    config = load_train_config(config_path)
    if config.stage != "sparse":
        raise ValueError("native export requires a sparse configuration")
    report = run_training_preflight(config, world_size=1)
    built = build_transfer_model(config, device=device, dtype=torch.bfloat16)
    loaded_path = _load_trainable_only(
        built.model, config, report, checkpoint_path
    )
    backbone = load_qwen35_text_causal_lm(
        config.sources.backbone.local_path, dtype=torch.bfloat16, device="cpu"
    )
    state = export_native_moe_state(
        backbone.state_dict(),
        [module.experts for module in built.transfer_modules],
        [module.router for module in built.transfer_modules],
        num_layers=config.architecture.student_layers,
        layout="fused",
        branch_scales=[module.branch_scale for module in built.transfer_modules],
        target_dtype=torch.bfloat16,
        text_only=True,
    )
    # Qwen3.5 ships MTP as a dense auxiliary decoder even when the target
    # backbone is converted to native MoE.  vLLM constructs a sparse MTP block
    # for qwen3_5_moe_text, so preserve the official FC/attention/norm tensors
    # and convert only its dense FFN to an algebraically exact shared expert.
    # Routed MTP experts remain zero; no untrained donor mapping is invented.
    source_mtp = load_qwen35_mtp(
        config.sources.backbone.local_path,
        dtype=torch.bfloat16,
        device="cpu",
    )
    mtp_state = export_native_moe_mtp_state(
        source_mtp.state_dict(),
        num_experts=config.architecture.num_experts,
        expert_intermediate_size=config.architecture.expert_intermediate_size,
        target_dtype=torch.bfloat16,
    )
    duplicate_mtp_keys = sorted(set(state) & set(mtp_state))
    if duplicate_mtp_keys:
        raise RuntimeError(
            f"native model state already contains MTP tensors: {duplicate_mtp_keys[:3]}"
        )
    state.update(mtp_state)
    for layer, module in enumerate(built.transfer_modules):
        experts = module.experts
        expected = (
            config.architecture.num_experts,
            config.architecture.expert_intermediate_size,
            config.architecture.student_hidden_size,
        )
        if tuple(experts.gate_proj.shape) != expected or tuple(experts.up_proj.shape) != expected:
            raise RuntimeError(
                f"layer {layer} folded gate/up shape violates configured native export contract"
            )
        expected_down = (
            config.architecture.num_experts,
            config.architecture.student_hidden_size,
            config.architecture.expert_intermediate_size,
        )
        if tuple(experts.down_proj.shape) != expected_down:
            raise RuntimeError(
                f"layer {layer} folded down shape violates configured native export contract"
            )
    native_config = build_native_moe_config(
        read_qwen_text_config(config.sources.backbone.local_path),
        num_experts=config.architecture.num_experts,
        experts_per_token=config.architecture.top_k,
        expert_intermediate_size=config.architecture.expert_intermediate_size,
        shared_expert_intermediate_size=config.architecture.student_intermediate_size,
        norm_topk_prob=config.architecture.norm_topk_prob,
    )
    output = Path(output_dir)
    tokenizer_source = Path(config.sources.tokenizer.local_path)
    tokenizer_names = {
        name for name in _TOKENIZER_BUNDLE_FILES if (tokenizer_source / name).is_file()
    }
    if not {"tokenizer.json", "tokenizer.model"} & tokenizer_names:
        raise RuntimeError(
            f"tokenizer source has no tokenizer.json/model: {tokenizer_source}"
        )
    expected_bundle = {
        "COMPLETE",
        "model.safetensors",
        "config.json",
        "twen_manifest.json",
        *tokenizer_names,
    }
    _prepare_bundle_directory(output, expected_bundle)
    artifact = output / "model.safetensors"
    _save_safetensors_atomic(state, artifact)
    atomic_write_json(output / "config.json", native_config)
    copied_tokenizer = _copy_tokenizer_files(tokenizer_source, output)
    bundle_files = _bundle_inventory(
        output,
        {"model.safetensors", "config.json", *copied_tokenizer},
    )
    manifest = {
        "schema_version": 1,
        "kind": "qwen3_5_moe_text",
        "source_checkpoint": str(loaded_path),
        "source_config_fingerprint": report.config_fingerprint,
        "raw_config_fingerprint": config.fingerprint(),
        "track": config.track,
        "backbone_revision": config.sources.backbone.revision,
        "donor_revision": config.sources.donor.revision,
        "source_manifests": {
            "backbone": config.sources.backbone.manifest_sha256,
            "donor": config.sources.donor.manifest_sha256,
            "folded_experts": config.sources.folded_experts_sha256,
        },
        "tokenizer_source": {
            "model_id": config.sources.tokenizer.model_id,
            "revision": config.sources.tokenizer.revision,
            "manifest_sha256": config.sources.tokenizer.manifest_sha256,
        },
        "calibration_artifacts": dict(report.calibration_fingerprints),
        "artifact": artifact.name,
        "artifact_sha256": sha256_file(artifact),
        "bundle_files": bundle_files,
        "adapters_present": False,
        "vision_present": False,
        "mtp_present": True,
        "mtp": {
            "source": "backbone_native_mtp",
            "num_hidden_layers": 1,
            "dedicated_embeddings": False,
            "shares_embedding_and_lm_head": True,
            "dense_ffn_conversion": "exact_shared_expert_zero_routed",
            "tensor_count": len(mtp_state),
        },
    }
    manifest_path = output / "twen_manifest.json"
    atomic_write_json(manifest_path, manifest)
    atomic_write_text(output / "COMPLETE", f"{sha256_file(manifest_path)}\n")
    return {
        "artifact": str(artifact),
        "sha256": manifest["artifact_sha256"],
        "config": str(output / "config.json"),
        "manifest": str(manifest_path),
    }
