from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from twen.data.shards import (
    ShardJob,
    ShardStateError,
    ShardTransaction,
    is_shard_complete,
    process_shards,
)


class ShardTransactionTest(unittest.TestCase):
    def test_partial_work_survives_and_complete_shard_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(KeyboardInterrupt), ShardTransaction(
                root, "shard-000", fingerprint="pipeline-v1", source_fingerprint="src-v1"
            ) as transaction:
                (transaction.work_directory / "resume.bin").write_bytes(b"partial")
                raise KeyboardInterrupt
            partial = root / "shard-000.incomplete"
            self.assertEqual((partial / "resume.bin").read_bytes(), b"partial")
            self.assertFalse((partial / "COMPLETE").exists())

            with ShardTransaction(
                root, "shard-000", fingerprint="pipeline-v1", source_fingerprint="src-v1"
            ) as transaction:
                self.assertEqual(
                    (transaction.work_directory / "resume.bin").read_bytes(), b"partial"
                )
                (transaction.work_directory / "output.bin").write_bytes(b"done")
                final = transaction.commit({"rows": 1})
            self.assertTrue(is_shard_complete(final))
            self.assertFalse(partial.exists())

            with ShardTransaction(
                root, "shard-000", fingerprint="pipeline-v1", source_fingerprint="src-v1"
            ) as transaction:
                self.assertTrue(transaction.complete)

    def test_identity_change_refuses_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ShardTransaction(root, "s0", fingerprint="v1") as transaction:
                (transaction.work_directory / "data").write_bytes(b"x")
                transaction.commit()
            with (
                self.assertRaises(ShardStateError),
                ShardTransaction(root, "s0", fingerprint="v2"),
            ):
                pass

    def test_process_shards_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            calls: list[str] = []

            def processor(job, work_directory):
                calls.append(job.shard_id)
                (work_directory / "rows.jsonl").write_text(job.shard_id, encoding="utf-8")
                return {"source": str(job.source)}

            jobs = [ShardJob("s0", "input-0", "hash-0"), ShardJob("s1", "input-1", "hash-1")]
            first = process_shards(
                jobs, directory, pipeline_fingerprint="tokenizer-v1", processor=processor
            )
            second = process_shards(
                jobs, directory, pipeline_fingerprint="tokenizer-v1", processor=processor
            )
            self.assertEqual(calls, ["s0", "s1"])
            self.assertTrue(all(not item.skipped for item in first))
            self.assertTrue(all(item.skipped for item in second))


if __name__ == "__main__":
    unittest.main()
