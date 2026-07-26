"""Random-access adapters over immutable KD and prepared-text corpus locks."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..data import (
    DatasetLayout,
    PreparedCorpusManifest,
    PreparedTextBatch,
    PreparedTextRecord,
    PreparedTextShardDataset,
    SampleReference,
    TeacherKDBatch,
    TeacherKDCorpusManifest,
    TeacherKDRecord,
    TeacherKDShardDataset,
    collate_prepared_text,
    collate_teacher_kd,
    read_kd_manifest,
    read_prepared_manifest,
    validate_kd_corpus_manifest,
    validate_prepared_corpus,
    write_kd_corpus_manifest,
)
from ..data.teacher_kd import KD_TENSORS_FILENAME


def build_kd_index(
    root: str | Path,
    output: str | Path,
    *,
    temperature: float,
    prepared_manifest: str | Path,
) -> Path:
    """Build the canonical corpus lock from complete shard directories."""

    root_path = Path(root)
    prepared = validate_prepared_corpus(prepared_manifest)
    # The prepared manifest is the canonical shard inventory.  Never glob the
    # output root: a process can die after syncing COMPLETE inside
    # ``<shard>.incomplete`` but before the atomic rename, and staging output
    # must not become part of a durable KD corpus lock.
    shards = []
    for entry in prepared.shards:
        path = root_path / entry.shard_id
        if path.name.endswith(".incomplete"):
            raise ValueError(f"refusing KD staging directory: {path}")
        if (
            path.is_dir()
            and (path / "COMPLETE").is_file()
            and (path / "kd_manifest.json").is_file()
        ):
            shards.append(path)
    return write_kd_corpus_manifest(
        output,
        shards,
        expected_temperature=temperature,
        prepared_corpus=prepared,
    )


class KDRecordStore:
    """Small LRU over complete shards with a world-size-independent layout."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        expected_temperature: float,
        expected_sequence_length: int,
        cache_shards: int = 32,
        verify_shards: bool = True,
    ) -> None:
        if cache_shards < 1:
            raise ValueError("cache_shards must be positive")
        self.manifest_path = Path(manifest_path)
        self.manifest: TeacherKDCorpusManifest = validate_kd_corpus_manifest(
            self.manifest_path,
            expected_temperature=expected_temperature,
            verify_shards=verify_shards,
        )
        self.cache_shards = cache_shards
        self.expected_sequence_length = expected_sequence_length
        self._entries = {entry.source_shard_id: entry for entry in self.manifest.shards}
        self._cache: OrderedDict[str, TeacherKDShardDataset] = OrderedDict()
        # Per-sequence loss denominators are tiny (~4 KiB per 1M-token shard)
        # and immutable. Cache all of them after one sequential labels/mask
        # scan instead of issuing two small random mmap reads for every sample.
        self._loss_count_cache: dict[str, tuple[Any, Any]] = {}
        self._mtp_loss_count_cache: dict[str, Any] = {}
        # One I/O thread is enough to overlap mmap slicing + pinning with the
        # current GPU microbatch.  A bounded queue (configured by data.num_workers)
        # prevents unbounded host RAM use and keeps all safe_open access on one
        # thread, avoiding undocumented concurrent-handle semantics.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="twen-kd-prefetch")
        self._closed = False
        for entry in self.manifest.shards:
            shard_manifest = read_kd_manifest(self.manifest_path.parent / entry.path)
            if shard_manifest.sequence_length != expected_sequence_length:
                raise ValueError(
                    f"KD shard {entry.path} sequence length {shard_manifest.sequence_length} "
                    f"!= training {expected_sequence_length}"
                )

    @property
    def layout(self) -> DatasetLayout:
        return DatasetLayout.from_shards(
            (
                (entry.source_shard_id, entry.sequence_count)
                for entry in self.manifest.shards
            ),
            fingerprint=self.manifest.dataset_fingerprint,
        )

    def _dataset(self, shard_id: str) -> TeacherKDShardDataset:
        existing = self._cache.pop(shard_id, None)
        if existing is not None:
            self._cache[shard_id] = existing
            return existing
        try:
            entry = self._entries[shard_id]
        except KeyError as exc:
            raise KeyError(f"unknown KD shard {shard_id}") from exc
        dataset = TeacherKDShardDataset(
            self.manifest_path.parent / entry.path,
            expected_temperature=self.manifest.temperature,
            # Rank zero already performed a full hash scan before CUDA/model
            # construction; random-access workers validate headers/lineage but
            # must not rehash hundreds of GB on every LRU reopen.
            verify_checksum=False,
        )
        manifest = dataset.manifest
        expected_identity = (
            self.manifest.teacher_model_id,
            self.manifest.teacher_revision,
            self.manifest.teacher_model_sha256,
            self.manifest.tokenizer_sha256,
            self.manifest.dataset_fingerprint,
            entry.source_shard_id,
            entry.source_tensors_sha256,
            entry.global_sample_start,
            entry.global_sample_end,
            entry.global_token_start,
            entry.global_token_end,
            entry.sequence_count,
            entry.token_count,
            entry.tensors_sha256,
        )
        actual_identity = (
            manifest.teacher_model_id,
            manifest.teacher_revision,
            manifest.teacher_model_sha256,
            manifest.tokenizer_sha256,
            manifest.dataset_fingerprint,
            manifest.source_shard_id,
            manifest.source_tensors_sha256,
            manifest.global_sample_start,
            manifest.global_sample_end,
            manifest.global_token_start,
            manifest.global_token_end,
            manifest.sequence_count,
            manifest.token_count,
            manifest.tensors_sha256,
        )
        if actual_identity != expected_identity:
            dataset.close()
            raise ValueError(f"KD shard {entry.path} changed after corpus preflight")
        self._cache[shard_id] = dataset
        while len(self._cache) > self.cache_shards:
            _, expired = self._cache.popitem(last=False)
            expired.close()
        return dataset

    def record(self, reference: SampleReference) -> TeacherKDRecord:
        return self._dataset(reference.shard_id)[reference.shard_offset]

    def batch(self, references: Sequence[SampleReference]) -> TeacherKDBatch:
        records = [self.record(reference) for reference in references]
        if len(records) == 1:
            # The production RTX 5090 path uses micro_batch_size=1.  torch.stack
            # would make a second ~2.7MB pageable copy after the mmap slice was
            # already cloned, immediately before pin_memory makes its own copy.
            # An unsqueeze is a contiguous view with identical batch semantics.
            record = records[0]
            return TeacherKDBatch(
                input_ids=record.input_ids.unsqueeze(0),
                labels=record.labels.unsqueeze(0),
                attention_mask=record.attention_mask.unsqueeze(0),
                topk_indices=record.topk_indices.unsqueeze(0),
                topk_logits=record.topk_logits.unsqueeze(0),
                teacher_logsumexp=record.teacher_logsumexp.unsqueeze(0),
                teacher_tail_logprob=record.teacher_tail_logprob.unsqueeze(0),
                temperature=record.temperature,
            )
        return collate_teacher_kd(records)

    def _shard_loss_count_vectors(self, shard_id: str) -> tuple[Any, Any]:
        cached = self._loss_count_cache.get(shard_id)
        if cached is not None:
            return cached
        # Authenticate/open through the normal LRU path first. Full file hashes
        # were checked by preflight; this training-only scan reads exactly the
        # two small tensors needed for loss denominators and never touches KD
        # logits. Running on the sole I/O executor keeps safe_open serialized.
        self._dataset(shard_id)
        entry = self._entries[shard_id]
        tensor_path = self.manifest_path.parent / entry.path / KD_TENSORS_FILENAME
        import torch
        from safetensors import safe_open

        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            labels = handle.get_tensor("labels")
            attention_mask = handle.get_tensor("attention_mask")
            hidden_counts = attention_mask.ne(0).sum(dim=1, dtype=torch.int64)
            target_counts = (
                labels[:, 1:].ne(-100) & attention_mask[:, :-1].ne(0)
            ).sum(dim=1, dtype=torch.int64)
        cached = (target_counts, hidden_counts)
        self._loss_count_cache[shard_id] = cached
        return cached

    def _loss_token_counts(
        self, references: Sequence[SampleReference]
    ) -> tuple[int, int]:
        target_tokens = 0
        valid_hidden_tokens = 0
        for reference in references:
            target_counts, hidden_counts = self._shard_loss_count_vectors(
                reference.shard_id
            )
            target_tokens += int(target_counts[reference.shard_offset])
            valid_hidden_tokens += int(hidden_counts[reference.shard_offset])
        return target_tokens, valid_hidden_tokens

    def optimizer_batch_token_counts(
        self, references: Sequence[SampleReference]
    ) -> tuple[int, int]:
        """Pre-read exact loss denominators on the store's sole I/O thread."""

        if self._closed:
            raise RuntimeError("KDRecordStore is closed")
        return self._executor.submit(
            self._loss_token_counts,
            tuple(references),
        ).result()

    def _valid_token_counts(
        self,
        references: Sequence[SampleReference],
    ) -> tuple[int, ...]:
        return tuple(
            int(self._shard_loss_count_vectors(reference.shard_id)[1][reference.shard_offset])
            for reference in references
        )

    def optimizer_batch_valid_token_counts(
        self,
        references: Sequence[SampleReference],
    ) -> tuple[int, ...]:
        """Return exact non-padding tokens aligned with the planned references."""

        if self._closed:
            raise RuntimeError("KDRecordStore is closed")
        return self._executor.submit(
            self._valid_token_counts,
            tuple(references),
        ).result()

    def _shard_mtp_loss_count_vector(self, shard_id: str) -> Any:
        cached = self._mtp_loss_count_cache.get(shard_id)
        if cached is not None:
            return cached
        # MTP is disabled by default, so keep its extra immutable count vector
        # out of the normal NTP/KD path.  Enabled runs scan the same two small
        # tensors once per shard and then use this in-memory vector thereafter.
        self._dataset(shard_id)
        entry = self._entries[shard_id]
        tensor_path = self.manifest_path.parent / entry.path / KD_TENSORS_FILENAME
        import torch
        from safetensors import safe_open

        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            labels = handle.get_tensor("labels")
            attention_mask = handle.get_tensor("attention_mask")
            counts = (
                labels[:, 2:].ne(-100)
                & attention_mask[:, :-2].ne(0)
                & attention_mask[:, 1:-1].ne(0)
                & attention_mask[:, 2:].ne(0)
            ).sum(dim=1, dtype=torch.int64)
        self._mtp_loss_count_cache[shard_id] = counts
        return counts

    def _mtp_loss_token_count(self, references: Sequence[SampleReference]) -> int:
        target_tokens = 0
        for reference in references:
            counts = self._shard_mtp_loss_count_vector(reference.shard_id)
            target_tokens += int(counts[reference.shard_offset])
        return target_tokens

    def optimizer_batch_mtp_token_count(
        self, references: Sequence[SampleReference]
    ) -> int:
        """Pre-read the exact native MTP ``L-2`` loss denominator."""

        if self._closed:
            raise RuntimeError("KDRecordStore is closed")
        return self._executor.submit(
            self._mtp_loss_token_count,
            tuple(references),
        ).result()

    def _prefetch_batch(
        self,
        references: Sequence[SampleReference],
        *,
        pin_memory: bool,
    ) -> TeacherKDBatch:
        batch = self.batch(references)
        if not pin_memory:
            return batch
        values = {
            name: tensor.pin_memory()
            for name, tensor in batch.tensors().items()
        }
        return TeacherKDBatch(**values, temperature=batch.temperature)

    def iter_prefetched_batches(
        self,
        reference_batches: Iterable[Sequence[SampleReference]],
        *,
        prefetch_depth: int,
        pin_memory: bool = True,
    ) -> Iterator[TeacherKDBatch]:
        """Yield ordered batches while one CPU worker prepares future batches."""

        if self._closed:
            raise RuntimeError("KDRecordStore is closed")
        if prefetch_depth < 1:
            raise ValueError("prefetch_depth must be positive")
        source = iter(reference_batches)
        pending: list[Future[TeacherKDBatch]] = []

        def fill() -> None:
            while len(pending) < prefetch_depth:
                try:
                    references = next(source)
                except StopIteration:
                    break
                pending.append(
                    self._executor.submit(
                        self._prefetch_batch,
                        tuple(references),
                        pin_memory=pin_memory,
                    )
                )

        fill()
        try:
            while pending:
                future = pending.pop(0)
                batch = future.result()
                fill()
                yield batch
        finally:
            for future in pending:
                future.cancel()

    def close(self) -> None:
        if self._closed:
            return
        self._executor.shutdown(wait=True, cancel_futures=True)
        while self._cache:
            _, dataset = self._cache.popitem(last=False)
            dataset.close()
        self._closed = True

    def __del__(self) -> None:  # pragma: no cover - engine closes deterministically
        try:
            if hasattr(self, "_executor"):
                self.close()
        except Exception:
            pass


class PreparedTextRecordStore:
    """Small LRU over prepared token shards without loading any KD tensors."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        expected_sequence_length: int,
        cache_shards: int = 32,
        verify_shards: bool = True,
    ) -> None:
        if cache_shards < 1:
            raise ValueError("cache_shards must be positive")
        if not isinstance(verify_shards, bool):
            raise ValueError("verify_shards must be a boolean")
        self.manifest_path = Path(manifest_path)
        self.manifest: PreparedCorpusManifest = (
            validate_prepared_corpus(self.manifest_path)
            if verify_shards
            else read_prepared_manifest(self.manifest_path)
        )
        if self.manifest.sequence_length != expected_sequence_length:
            raise ValueError(
                f"prepared corpus sequence length {self.manifest.sequence_length} "
                f"!= training {expected_sequence_length}"
            )
        self.cache_shards = cache_shards
        self.expected_sequence_length = expected_sequence_length
        self._entries = {
            entry.shard_id: entry
            for entry in self.manifest.shards
        }
        self._cache: OrderedDict[str, PreparedTextShardDataset] = OrderedDict()
        self._loss_count_cache: dict[str, tuple[Any, Any]] = {}
        self._mtp_loss_count_cache: dict[str, Any] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="twen-prepared-text-prefetch",
        )
        self._closed = False

    @property
    def layout(self) -> DatasetLayout:
        return DatasetLayout.from_shards(
            (
                (entry.shard_id, entry.sequence_count)
                for entry in self.manifest.shards
            ),
            fingerprint=self.manifest.dataset_fingerprint,
        )

    def _dataset(self, shard_id: str) -> PreparedTextShardDataset:
        if self._closed:
            raise RuntimeError("PreparedTextRecordStore is closed")
        existing = self._cache.pop(shard_id, None)
        if existing is not None:
            self._cache[shard_id] = existing
            return existing
        try:
            entry = self._entries[shard_id]
        except KeyError as error:
            raise KeyError(f"unknown prepared-text shard {shard_id}") from error
        root = self.manifest_path.parent.resolve()
        shard = (root / entry.path).resolve()
        try:
            shard.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"prepared-text shard escapes corpus root: {entry.path}"
            ) from error
        if shard == root:
            raise ValueError(f"prepared-text shard resolves to corpus root: {entry.path}")
        dataset = PreparedTextShardDataset(
            shard,
            expected_sequence_count=entry.sequence_count,
            expected_sequence_length=self.expected_sequence_length,
        )
        self._cache[shard_id] = dataset
        while len(self._cache) > self.cache_shards:
            _, expired = self._cache.popitem(last=False)
            expired.close()
        return dataset

    def record(self, reference: SampleReference) -> PreparedTextRecord:
        return self._dataset(reference.shard_id)[reference.shard_offset]

    def batch(self, references: Sequence[SampleReference]) -> PreparedTextBatch:
        records = [self.record(reference) for reference in references]
        if len(records) == 1:
            record = records[0]
            return PreparedTextBatch(
                **{
                    name: getattr(record, name).unsqueeze(0)
                    for name in ("input_ids", "labels", "attention_mask")
                }
            )
        return collate_prepared_text(records)

    def _shard_loss_count_vectors(self, shard_id: str) -> tuple[Any, Any]:
        cached = self._loss_count_cache.get(shard_id)
        if cached is None:
            cached = self._dataset(shard_id).loss_count_vectors()
            self._loss_count_cache[shard_id] = cached
        return cached

    def _loss_token_counts(
        self,
        references: Sequence[SampleReference],
    ) -> tuple[int, int]:
        target_tokens = 0
        valid_hidden_tokens = 0
        for reference in references:
            target_counts, hidden_counts = self._shard_loss_count_vectors(
                reference.shard_id
            )
            target_tokens += int(target_counts[reference.shard_offset])
            valid_hidden_tokens += int(hidden_counts[reference.shard_offset])
        return target_tokens, valid_hidden_tokens

    def optimizer_batch_token_counts(
        self,
        references: Sequence[SampleReference],
    ) -> tuple[int, int]:
        if self._closed:
            raise RuntimeError("PreparedTextRecordStore is closed")
        return self._executor.submit(
            self._loss_token_counts,
            tuple(references),
        ).result()

    def _valid_token_counts(
        self,
        references: Sequence[SampleReference],
    ) -> tuple[int, ...]:
        return tuple(
            int(self._shard_loss_count_vectors(reference.shard_id)[1][reference.shard_offset])
            for reference in references
        )

    def optimizer_batch_valid_token_counts(
        self,
        references: Sequence[SampleReference],
    ) -> tuple[int, ...]:
        if self._closed:
            raise RuntimeError("PreparedTextRecordStore is closed")
        return self._executor.submit(
            self._valid_token_counts,
            tuple(references),
        ).result()

    def _shard_mtp_loss_count_vector(self, shard_id: str) -> Any:
        cached = self._mtp_loss_count_cache.get(shard_id)
        if cached is None:
            cached = self._dataset(shard_id).mtp_loss_count_vector()
            self._mtp_loss_count_cache[shard_id] = cached
        return cached

    def _mtp_loss_token_count(
        self,
        references: Sequence[SampleReference],
    ) -> int:
        target_tokens = 0
        for reference in references:
            counts = self._shard_mtp_loss_count_vector(reference.shard_id)
            target_tokens += int(counts[reference.shard_offset])
        return target_tokens

    def optimizer_batch_mtp_token_count(
        self,
        references: Sequence[SampleReference],
    ) -> int:
        if self._closed:
            raise RuntimeError("PreparedTextRecordStore is closed")
        return self._executor.submit(
            self._mtp_loss_token_count,
            tuple(references),
        ).result()

    def _prefetch_batch(
        self,
        references: Sequence[SampleReference],
        *,
        pin_memory: bool,
    ) -> PreparedTextBatch:
        batch = self.batch(references)
        if not pin_memory:
            return batch
        return PreparedTextBatch(
            **{
                name: tensor.pin_memory()
                for name, tensor in batch.tensors().items()
            }
        )

    def iter_prefetched_batches(
        self,
        reference_batches: Iterable[Sequence[SampleReference]],
        *,
        prefetch_depth: int,
        pin_memory: bool = True,
    ) -> Iterator[PreparedTextBatch]:
        if self._closed:
            raise RuntimeError("PreparedTextRecordStore is closed")
        if prefetch_depth < 1:
            raise ValueError("prefetch_depth must be positive")
        source = iter(reference_batches)
        pending: list[Future[PreparedTextBatch]] = []

        def fill() -> None:
            while len(pending) < prefetch_depth:
                try:
                    references = next(source)
                except StopIteration:
                    break
                pending.append(
                    self._executor.submit(
                        self._prefetch_batch,
                        tuple(references),
                        pin_memory=pin_memory,
                    )
                )

        fill()
        try:
            while pending:
                future = pending.pop(0)
                batch = future.result()
                fill()
                yield batch
        finally:
            for future in pending:
                future.cancel()

    def close(self) -> None:
        if self._closed:
            return
        self._executor.shutdown(wait=True, cancel_futures=True)
        while self._cache:
            _, dataset = self._cache.popitem(last=False)
            dataset.close()
        self._closed = True

    def __del__(self) -> None:  # pragma: no cover - engine closes deterministically
        try:
            if hasattr(self, "_executor"):
                self.close()
        except Exception:
            pass


def move_kd_batch(batch: TeacherKDBatch, device: Any) -> TeacherKDBatch:
    values = {
        name: tensor.to(device=device, non_blocking=True)
        for name, tensor in batch.tensors().items()
    }
    return TeacherKDBatch(**values, temperature=batch.temperature)


def move_prepared_text_batch(
    batch: PreparedTextBatch,
    device: Any,
) -> PreparedTextBatch:
    return PreparedTextBatch(
        **{
            name: tensor.to(device=device, non_blocking=True)
            for name, tensor in batch.tensors().items()
        }
    )
