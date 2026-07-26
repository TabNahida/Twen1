from __future__ import annotations

import unittest

from twen.modeling import (
    ConfigAuditError,
    LayerMappingError,
    audit_source_configs,
    build_channel_partition,
    build_native_moe_config,
    match_layers_cka,
)


def dense_config(hidden: int, intermediate: int, layers: int) -> dict[str, object]:
    return {
        "model_type": "qwen3_5_text",
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": layers,
        "layer_types": [
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            for i in range(layers)
        ],
    }


class ConfigAuditTests(unittest.TestCase):
    def test_exact_pair_and_nested_text_config(self) -> None:
        small = {"model_type": "qwen3_5", "text_config": dense_config(1024, 3584, 24)}
        donor = dense_config(4096, 12288, 32)
        audit = audit_source_configs(small, donor)
        self.assertTrue(audit.compatible)
        self.assertEqual(audit.backbone.full_attention_layers, (3, 7, 11, 15, 19, 23))
        self.assertEqual(len(audit.donor.layer_types), 32)

    def test_wrong_source_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigAuditError, "unexpected"):
            audit_source_configs(
                dense_config(2048, 3584, 24), dense_config(4096, 12288, 32)
            )

    def test_native_config_has_v1_contract(self) -> None:
        config = build_native_moe_config(dense_config(1024, 3584, 24))
        self.assertEqual(config["model_type"], "qwen3_5_moe_text")
        self.assertEqual(config["num_experts"], 8)
        self.assertEqual(config["num_experts_per_tok"], 2)
        self.assertEqual(config["moe_intermediate_size"], 1536)
        self.assertEqual(config["shared_expert_intermediate_size"], 3584)
        self.assertTrue(config["norm_topk_prob"])


class MappingTests(unittest.TestCase):
    def test_monotonic_same_type_dp_uses_global_optimum(self) -> None:
        # Student 0 locally prefers donor 2, but taking donor 0 permits the much
        # stronger student-1/donor-2 match.
        scores = [[0.8, 0.0, 0.9], [0.1, 0.0, 1.0]]
        match = match_layers_cka(
            cka_scores=scores,
            student_layer_types=["linear", "linear"],
            donor_layer_types=["linear", "full", "linear"],
        )
        self.assertEqual(match.student_to_donor, (0, 2))
        self.assertAlmostEqual(match.total_cka, 1.8)

    def test_same_type_can_make_mapping_impossible(self) -> None:
        with self.assertRaises(LayerMappingError):
            match_layers_cka(
                cka_scores=[[1.0, 0.0], [0.0, 1.0]],
                student_layer_types=["full", "full"],
                donor_layer_types=["linear", "full"],
            )


class PartitionTests(unittest.TestCase):
    def test_partition_is_complete_capacity_bounded_and_deterministic(self) -> None:
        scores = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        first = build_channel_partition(scores=scores, num_experts=2, expert_size=4)
        second = build_channel_partition(scores=scores, num_experts=2, expert_size=4)
        self.assertEqual(first, second)
        self.assertEqual(tuple(map(len, first.indices)), (4, 4))
        self.assertEqual(
            sorted(channel for group in first.indices for channel in group), list(range(8))
        )
        self.assertLess(first.max_mean_ratio, 1.1)

    def test_partition_requires_exact_capacity(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot form"):
            build_channel_partition(scores=[1.0] * 7, num_experts=2, expert_size=4)


if __name__ == "__main__":
    unittest.main()
