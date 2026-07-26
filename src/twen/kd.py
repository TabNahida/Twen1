"""User-run, shard-resumable top-64 teacher-logit generation."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import (
    TeacherKDBatch,
    TeacherKDManifest,
    is_shard_complete,
    validate_kd_shard,
    validate_prepared_corpus,
    write_kd_corpus_manifest,
    write_kd_shard,
)
from .data.teacher_kd import KD_GENERATOR_SOURCE_SHA256
from .model_loading import freeze_module, load_qwen35_text_causal_lm
from .progress import TaskProgress
from .utils import atomic_write_json


@dataclass(frozen=True, slots=True)
class KDGenerationStopped(RuntimeError):
    """A persistent STOP file was observed at a KD shard boundary."""

    stop_file: str
    worker_index: int
    num_workers: int
    completed_shards: tuple[str, ...] = ()
    assigned_shards: int | None = None

    def __str__(self) -> str:
        progress = (
            f"{len(self.completed_shards)}/{self.assigned_shards} assigned shards"
            if self.assigned_shards is not None
            else f"{len(self.completed_shards)} completed shards"
        )
        return (
            f"teacher KD worker {self.worker_index}/{self.num_workers} stopped safely after "
            f"{progress}; remove {self.stop_file} before resuming"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stop_file": self.stop_file,
            "worker_index": self.worker_index,
            "num_workers": self.num_workers,
            "completed_shards": list(self.completed_shards),
            "completed_count": len(self.completed_shards),
            "assigned_count": self.assigned_shards,
            "message": str(self),
        }


def _normalize_cuda_device(device: str) -> str:
    """Give the CUDA shorthand an explicit visible-device index."""

    if device == "cuda":
        return f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
    return device


def _raise_if_stopped(
    stop_file: str | None,
    *,
    worker_index: int,
    num_workers: int,
    completed: list[Path] | tuple[Path, ...] = (),
    assigned_shards: int | None = None,
) -> None:
    if stop_file is None or not Path(stop_file).is_file():
        return
    raise KDGenerationStopped(
        stop_file=str(Path(stop_file)),
        worker_index=worker_index,
        num_workers=num_workers,
        completed_shards=tuple(path.name for path in completed),
        assigned_shards=assigned_shards,
    )


def generate_teacher_kd(
    *,
    prepared_manifest: str,
    output_root: str,
    teacher_path: str,
    teacher_model_id: str,
    teacher_revision: str,
    teacher_manifest_sha256: str,
    tokenizer_manifest_sha256: str,
    temperature: float = 2.0,
    batch_size: int = 1,
    logits_chunk_tokens: int = 64,
    device: str = "cuda",
    stop_file: str | None = None,
    worker_index: int | None = None,
    num_workers: int | None = None,
    progress: str = "auto",
) -> Path:
    """Generate raw top-64 logits without materializing full [B,L,V] logits."""

    if (
        not math.isfinite(temperature)
        or temperature <= 0
        or batch_size <= 0
        or logits_chunk_tokens <= 0
    ):
        raise ValueError("temperature, batch_size and logits_chunk_tokens must be positive")
    actual_workers = int(
        num_workers if num_workers is not None else os.environ.get("WORLD_SIZE", "1")
    )
    actual_worker = int(
        worker_index if worker_index is not None else os.environ.get("RANK", "0")
    )
    if actual_workers < 1 or not 0 <= actual_worker < actual_workers:
        raise ValueError("KD worker index must be within num_workers")
    # STOP should be cheap even when validating the teacher/prepared corpus
    # would otherwise hash hundreds of gigabytes. It remains in place so every
    # torchrun worker observes the same request.
    _raise_if_stopped(
        stop_file,
        worker_index=actual_worker,
        num_workers=actual_workers,
    )

    from .config import ModelSource
    from .io.offline import enforce_offline_environment
    from .preflight import _check_source

    enforce_offline_environment()
    teacher_source = ModelSource(
        model_id=teacher_model_id,
        revision=teacher_revision,
        local_path=teacher_path,
        manifest_sha256=teacher_manifest_sha256,
    )
    teacher_source.validate("teacher")
    teacher_revision = teacher_source.revision
    teacher_manifest_sha256 = teacher_source.manifest_sha256
    tokenizer_manifest_sha256 = tokenizer_manifest_sha256.lower()
    _, _, teacher_text_config = _check_source("teacher", teacher_source)

    import torch
    from safetensors.torch import load_file

    # torch.cuda.set_device requires an explicit index even though tensor
    # transfers accept the shorthand "cuda".  In a single-process run
    # LOCAL_RANK is absent, so normalize to the first visible device.
    device = _normalize_cuda_device(device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("KD generation requested CUDA but CUDA is unavailable")
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    prepared_path = Path(prepared_manifest)
    prepared = validate_prepared_corpus(prepared_path)
    if prepared.tokenizer_sha256.lower() != tokenizer_manifest_sha256.lower():
        raise ValueError(
            "prepared tokenizer identity differs from --tokenizer-manifest-sha256"
        )
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    assigned = tuple(
        entry
        for index, entry in enumerate(prepared.shards)
        if index % actual_workers == actual_worker
    )
    if not assigned:
        raise ValueError(
            f"KD worker {actual_worker}/{actual_workers} has no assigned prepared shards"
        )
    completed: list[Path] = []
    completed_tokens = 0
    pending = []
    for entry in assigned:
        _raise_if_stopped(
            stop_file,
            worker_index=actual_worker,
            num_workers=actual_workers,
            completed=completed,
            assigned_shards=len(assigned),
        )
        destination = output / entry.shard_id
        if is_shard_complete(destination, verify_hashes=False):
            existing = validate_kd_shard(
                destination,
                expected_temperature=temperature,
            )
            expected_identity = (
                teacher_model_id,
                teacher_revision,
                teacher_manifest_sha256.lower(),
                KD_GENERATOR_SOURCE_SHA256,
                tokenizer_manifest_sha256.lower(),
                prepared.dataset_fingerprint,
                entry.shard_id,
                entry.tensors_sha256,
                entry.global_sample_start,
                entry.global_sample_end,
                entry.global_token_start,
                entry.global_token_end,
                entry.sequence_count,
                prepared.sequence_length,
                entry.token_count,
                int(teacher_text_config["vocab_size"]),
            )
            actual_identity = (
                existing.teacher_model_id,
                existing.teacher_revision,
                existing.teacher_model_sha256,
                existing.generator_source_sha256,
                existing.tokenizer_sha256,
                existing.dataset_fingerprint,
                existing.source_shard_id,
                existing.source_tensors_sha256,
                existing.global_sample_start,
                existing.global_sample_end,
                existing.global_token_start,
                existing.global_token_end,
                existing.sequence_count,
                existing.sequence_length,
                existing.token_count,
                existing.vocab_size,
            )
            if actual_identity != expected_identity:
                raise ValueError(
                    f"existing KD shard {destination} belongs to a different "
                    "teacher/prepared-data contract"
                )
            completed.append(destination)
            completed_tokens += entry.token_count
        else:
            pending.append(entry)
    teacher = None
    with TaskProgress(
        total=sum(entry.token_count for entry in assigned),
        initial=completed_tokens,
        description=f"teacher-kd[{actual_worker}/{actual_workers}]",
        unit="tok",
        unit_scale=True,
        mode=progress,
    ) as progress_bar:
        if pending:
            _raise_if_stopped(
                stop_file,
                worker_index=actual_worker,
                num_workers=actual_workers,
                completed=completed,
                assigned_shards=len(assigned),
            )
            teacher = load_qwen35_text_causal_lm(
                teacher_path, dtype=torch.bfloat16, device=device
            )
            freeze_module(teacher)
        for entry in pending:
            _raise_if_stopped(
                stop_file,
                worker_index=actual_worker,
                num_workers=actual_workers,
                completed=completed,
                assigned_shards=len(assigned),
            )
            destination = output / entry.shard_id
            assert teacher is not None
            source = prepared_path.parent / entry.path / "tokens.safetensors"
            batch = load_file(str(source), device="cpu")
            n, length = batch["input_ids"].shape
            top_indices = torch.empty((n, length, 64), dtype=torch.int64)
            top_logits = torch.empty((n, length, 64), dtype=torch.bfloat16)
            teacher_logsumexp = torch.empty((n, length), dtype=torch.float32)
            teacher_tail_logprob = torch.empty((n, length), dtype=torch.float32)
            for batch_start in range(0, n, batch_size):
                batch_end = min(n, batch_start + batch_size)
                input_ids = batch["input_ids"][batch_start:batch_end].to(device)
                attention_mask = batch["attention_mask"][batch_start:batch_end].to(device)
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda" if device.startswith("cuda") else "cpu",
                    dtype=torch.bfloat16,
                    enabled=True,
                ):
                    hidden = teacher.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    ).last_hidden_state
                    for token_start in range(0, length, logits_chunk_tokens):
                        token_end = min(length, token_start + logits_chunk_tokens)
                        raw = teacher.lm_head(hidden[:, token_start:token_end])
                        values, indices = torch.topk(raw, 64, dim=-1)
                        scaled = raw.float() / temperature
                        log_z = torch.logsumexp(scaled, dim=-1)
                        log_top_mass = (
                            torch.logsumexp(values.float() / temperature, dim=-1) - log_z
                        )
                        top_mass = log_top_mass.exp().clamp(max=1.0 - 1e-7)
                        tail = torch.log1p(-top_mass)
                        selection = (
                            slice(batch_start, batch_end),
                            slice(token_start, token_end),
                        )
                        top_indices[selection] = indices.cpu()
                        top_logits[selection] = values.to(torch.bfloat16).cpu()
                        teacher_logsumexp[selection] = log_z.cpu()
                        teacher_tail_logprob[selection] = tail.cpu()
                del hidden
                valid_tokens = int(
                    batch["attention_mask"][batch_start:batch_end].sum().item()
                )
                progress_bar.update(valid_tokens)
            kd_batch = TeacherKDBatch(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                attention_mask=batch["attention_mask"],
                topk_indices=top_indices,
                topk_logits=top_logits,
                teacher_logsumexp=teacher_logsumexp,
                teacher_tail_logprob=teacher_tail_logprob,
                temperature=temperature,
            )
            manifest = TeacherKDManifest(
                teacher_model_id=teacher_model_id,
                teacher_revision=teacher_revision,
                teacher_model_sha256=teacher_manifest_sha256,
                generator_source_sha256=KD_GENERATOR_SOURCE_SHA256,
                tokenizer_sha256=tokenizer_manifest_sha256,
                dataset_fingerprint=prepared.dataset_fingerprint,
                source_tensors_sha256=entry.tensors_sha256,
                source_shard_id=entry.shard_id,
                global_sample_start=entry.global_sample_start,
                global_sample_end=entry.global_sample_end,
                global_token_start=entry.global_token_start,
                global_token_end=entry.global_token_end,
                sequence_count=entry.sequence_count,
                sequence_length=prepared.sequence_length,
                token_count=entry.token_count,
                vocab_size=int(teacher.config.vocab_size),
                temperature=temperature,
                logits_dtype="BF16",
            )
            completed.append(
                write_kd_shard(output, entry.shard_id, manifest=manifest, batch=kd_batch)
            )
            progress_bar.set_postfix(
                {"shards": f"{len(completed)}/{len(assigned)}", "last": entry.shard_id}
            )
    if actual_workers > 1:
        status = output / f"worker-{actual_worker:05d}-of-{actual_workers:05d}.json"
        atomic_write_json(
            status,
            {
                "schema_version": 1,
                "kind": "teacher_kd_worker_complete",
                "worker_index": actual_worker,
                "num_workers": actual_workers,
                "assigned_shards": [entry.shard_id for entry in assigned],
                "completed_shards": [path.name for path in completed],
                "prepared_manifest": str(prepared_path),
                "dataset_fingerprint": prepared.dataset_fingerprint,
                "teacher_revision": teacher_revision,
                "generator_source_sha256": KD_GENERATOR_SOURCE_SHA256,
                "temperature": temperature,
            },
        )
        return status
    return write_kd_corpus_manifest(
        output / "manifest.json",
        completed,
        expected_temperature=temperature,
        prepared_corpus=prepared,
    )
