from __future__ import annotations

import unittest
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from twen.data.teacher_kd import KD_REQUIRED_TENSORS, TeacherKDBatch, TeacherKDRecord
from twen.training.data import KDRecordStore


class KDRecordStorePrefetchTest(unittest.TestCase):
    def test_single_record_batch_uses_a_zero_copy_batch_dimension(self) -> None:
        store = KDRecordStore.__new__(KDRecordStore)
        tensors = {
            name: torch.arange(6, dtype=torch.int64).reshape(2, 3)
            for name in KD_REQUIRED_TENSORS
        }
        record = TeacherKDRecord(**tensors, temperature=2.0)
        store.record = lambda _reference: record

        with patch("twen.training.data.collate_teacher_kd") as collate:
            batch = store.batch((SimpleNamespace(),))

        collate.assert_not_called()
        self.assertEqual(tuple(batch.input_ids.shape), (1, 2, 3))
        self.assertEqual(batch.input_ids.data_ptr(), record.input_ids.data_ptr())
        self.assertTrue(batch.input_ids.is_contiguous())

    def test_optimizer_batch_counts_use_the_store_io_executor(self) -> None:
        store = KDRecordStore.__new__(KDRecordStore)
        store._closed = False
        store._cache = OrderedDict()
        store._executor = ThreadPoolExecutor(max_workers=1)

        store._shard_loss_count_vectors = lambda _shard_id: (
            torch.tensor([1, 2, 3]),
            torch.tensor([2, 3, 4]),
        )
        store._shard_mtp_loss_count_vector = lambda _shard_id: torch.tensor([0, 1, 2])
        references = (
            SimpleNamespace(shard_id="a", shard_offset=0),
            SimpleNamespace(shard_id="a", shard_offset=2),
        )
        self.assertEqual(store.optimizer_batch_token_counts(references), (4, 6))
        self.assertEqual(store.optimizer_batch_mtp_token_count(references), 2)
        store.close()

    def test_shard_loss_counts_are_vectorized_and_cached(self) -> None:
        store = KDRecordStore.__new__(KDRecordStore)
        store._loss_count_cache = {}
        store._entries = {"a": SimpleNamespace(path="shard-a")}
        store.manifest_path = Path("root/manifest.json")
        store._dataset = lambda _shard_id: object()

        labels = torch.tensor([[10, 11, 12, -100], [20, 21, -100, -100]])
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)

        class Handle:
            def get_tensor(self, name: str):
                return {"labels": labels, "attention_mask": mask}[name]

        class Context:
            def __enter__(self):
                return Handle()

            def __exit__(self, *_args):
                return None

        with patch("safetensors.safe_open", return_value=Context()) as opened:
            first = store._shard_loss_count_vectors("a")
            second = store._shard_loss_count_vectors("a")

        opened.assert_called_once()
        self.assertEqual(first[0].tolist(), [2, 1])
        self.assertEqual(first[1].tolist(), [3, 2])
        self.assertIs(first, second)

    def test_shard_mtp_counts_are_l_minus_two_vectorized_and_cached(self) -> None:
        store = KDRecordStore.__new__(KDRecordStore)
        store._mtp_loss_count_cache = {}
        store._entries = {"a": SimpleNamespace(path="shard-a")}
        store.manifest_path = Path("root/manifest.json")
        store._dataset = lambda _shard_id: object()

        labels = torch.tensor([[10, 11, 12, 13, -100], [20, 21, -100, 23, 24]])
        mask = torch.tensor(
            [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]],
            dtype=torch.bool,
        )

        class Handle:
            def get_tensor(self, name: str):
                return {"labels": labels, "attention_mask": mask}[name]

        class Context:
            def __enter__(self):
                return Handle()

            def __exit__(self, *_args):
                return None

        with patch("safetensors.safe_open", return_value=Context()) as opened:
            first = store._shard_mtp_loss_count_vector("a")
            second = store._shard_mtp_loss_count_vector("a")

        opened.assert_called_once()
        self.assertEqual(first.tolist(), [2, 2])
        self.assertIs(first, second)

    def test_bounded_prefetch_preserves_order_and_can_cancel_cleanly(self) -> None:
        store = KDRecordStore.__new__(KDRecordStore)
        store._closed = False
        store._cache = OrderedDict()
        store._executor = ThreadPoolExecutor(max_workers=1)

        def make_batch(references):
            value = int(references[0])
            tensors = {
                name: torch.tensor([value], dtype=torch.int64)
                for name in KD_REQUIRED_TENSORS
            }
            return TeacherKDBatch(**tensors, temperature=2.0)

        store.batch = make_batch
        iterator = store.iter_prefetched_batches(
            ((index,) for index in range(6)),
            prefetch_depth=3,
            pin_memory=False,
        )
        first = next(iterator)
        second = next(iterator)
        self.assertEqual(int(first.input_ids[0]), 0)
        self.assertEqual(int(second.input_ids[0]), 1)
        iterator.close()
        store.close()
        self.assertTrue(store._closed)


if __name__ == "__main__":
    unittest.main()
