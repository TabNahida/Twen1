from __future__ import annotations

import unittest

from twen.data.cursor import (
    DatasetLayout,
    DeterministicCooldownCursor,
    DeterministicGlobalCursor,
    gradient_accumulation_steps,
)


class GlobalCursorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = DatasetLayout.from_shards(
            [("shard-a", 5), ("shard-b", 7)], fingerprint="dataset-v1"
        )

    def test_epoch_mapping_is_a_permutation(self) -> None:
        cursor = DeterministicGlobalCursor(self.layout, seed=123)
        first_epoch = cursor.plan_global_batch(self.layout.size)
        self.assertEqual(sorted(item.flat_index for item in first_epoch), list(range(12)))
        self.assertEqual(len({(item.shard_id, item.shard_offset) for item in first_epoch}), 12)

    def test_shuffle_keeps_each_shard_in_one_io_local_block(self) -> None:
        layout = DatasetLayout.from_shards(
            [(f"shard-{index}", 5 + index) for index in range(8)],
            fingerprint="many-shards-v1",
        )
        items = DeterministicGlobalCursor(layout, seed=99).plan_global_batch(layout.size)
        runs = []
        for item in items:
            if not runs or runs[-1] != item.shard_id:
                runs.append(item.shard_id)
        self.assertEqual(len(runs), len(layout.shard_ids))
        self.assertEqual(set(runs), set(layout.shard_ids))

    def test_world_size_changes_only_rank_partition(self) -> None:
        cursor = DeterministicGlobalCursor(self.layout, seed=7, next_global_sample=3)
        expected = cursor.plan_global_batch(12)
        for world_size in (1, 2, 3, 4, 6):
            rank_items = [
                item
                for rank in range(world_size)
                for item in cursor.plan_rank_batch(12, rank=rank, world_size=world_size)
            ]
            reconstructed = sorted(rank_items, key=lambda item: item.global_position)
            self.assertEqual(reconstructed, list(expected))

    def test_only_commit_advances_and_state_restores(self) -> None:
        cursor = DeterministicGlobalCursor(self.layout, seed=9)
        first = cursor.plan_global_batch(4)
        replay = cursor.plan_global_batch(4)
        self.assertEqual(first, replay)
        cursor.commit(global_batch_samples=4, token_count=77)
        restored = DeterministicGlobalCursor.from_state_dict(self.layout, cursor.state_dict())
        self.assertEqual(restored.next_global_sample, 4)
        self.assertEqual(restored.committed_tokens, 77)
        self.assertEqual(restored.plan_global_batch(4), cursor.plan_global_batch(4))

    def test_changed_dataset_rejects_saved_cursor(self) -> None:
        state = DeterministicGlobalCursor(self.layout, seed=1).state_dict()
        other = DatasetLayout.from_shards([("shard-a", 12)], fingerprint="different")
        with self.assertRaises(ValueError):
            DeterministicGlobalCursor.from_state_dict(other, state)

    def test_accumulation_preserves_global_token_batch(self) -> None:
        self.assertEqual(
            gradient_accumulation_steps(
                global_batch_tokens=8192,
                world_size=4,
                microbatch_tokens_per_rank=512,
            ),
            4,
        )
        with self.assertRaises(ValueError):
            gradient_accumulation_steps(
                global_batch_tokens=8000,
                world_size=3,
                microbatch_tokens_per_rank=512,
            )

    def test_quality_cooldown_switches_once_and_restores_both_phase_cursors(self) -> None:
        cooldown_layout = DatasetLayout.from_shards(
            [("quality-a", 3), ("quality-b", 4)],
            fingerprint="quality-dataset-v1",
        )
        cursor = DeterministicCooldownCursor(
            self.layout,
            cooldown_layout,
            seed=17,
            cooldown_start_tokens=100,
        )
        primary_batch = cursor.plan_global_batch(4)
        self.assertTrue(all(item.shard_id.startswith("shard-") for item in primary_batch))
        cursor.commit(global_batch_samples=4, token_count=101)
        self.assertEqual(cursor.active_phase, "cooldown")
        cooldown_batch = cursor.plan_global_batch(4)
        self.assertTrue(all(item.shard_id.startswith("quality-") for item in cooldown_batch))
        self.assertEqual([item.global_position for item in cooldown_batch], [4, 5, 6, 7])
        cursor.commit(global_batch_samples=4, token_count=77)

        restored = DeterministicCooldownCursor.from_state_dict(
            self.layout,
            cooldown_layout,
            cursor.state_dict(),
            cooldown_start_tokens=100,
        )
        self.assertEqual(restored.active_phase, "cooldown")
        self.assertEqual(restored.next_global_sample, 8)
        self.assertEqual(restored.committed_tokens, 178)
        self.assertEqual(restored.plan_global_batch(4), cursor.plan_global_batch(4))

    def test_quality_cooldown_world_size_changes_only_rank_partition(self) -> None:
        cooldown_layout = DatasetLayout.from_shards(
            [("quality-a", 5), ("quality-b", 7)],
            fingerprint="quality-dataset-v1",
        )
        cursor = DeterministicCooldownCursor(
            self.layout,
            cooldown_layout,
            seed=23,
            cooldown_start_tokens=10,
        )
        cursor.commit(global_batch_samples=3, token_count=11)
        expected = cursor.plan_global_batch(12)
        for world_size in (1, 2, 3, 4, 6):
            rank_items = [
                item
                for rank in range(world_size)
                for item in cursor.plan_rank_batch(12, rank=rank, world_size=world_size)
            ]
            self.assertEqual(
                sorted(rank_items, key=lambda item: item.global_position),
                list(expected),
            )

    def test_quality_cooldown_resume_rejects_tampered_phase_state(self) -> None:
        cooldown_layout = DatasetLayout.from_shards(
            [("quality", 4)], fingerprint="quality-dataset-v1"
        )
        cursor = DeterministicCooldownCursor(
            self.layout,
            cooldown_layout,
            seed=31,
            cooldown_start_tokens=100,
        )
        state = cursor.state_dict()
        state["active_phase"] = "cooldown"
        with self.assertRaisesRegex(ValueError, "active phase"):
            DeterministicCooldownCursor.from_state_dict(
                self.layout,
                cooldown_layout,
                state,
                cooldown_start_tokens=100,
            )


if __name__ == "__main__":
    unittest.main()
