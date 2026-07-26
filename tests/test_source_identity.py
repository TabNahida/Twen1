from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from twen.source_identity import twen_source_tree_sha256
from twen.training.engine import _checkpoint


def _write_tree(root: Path, files: tuple[tuple[str, bytes], ...]) -> None:
    for relative, payload in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def test_source_tree_hash_is_order_stable_and_ignores_non_python(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    files = (("a.py", b"a = 1\n"), ("nested/b.py", b"b = 2\n"))
    _write_tree(first, files)
    _write_tree(second, tuple(reversed(files)))
    (second / "nested/cache.pyc").write_bytes(b"ignored")

    assert twen_source_tree_sha256(first) == twen_source_tree_sha256(second)


def test_source_tree_hash_binds_paths_and_contents(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    changed_content = tmp_path / "changed-content"
    changed_path = tmp_path / "changed-path"
    _write_tree(baseline, (("a.py", b"same\n"), ("b.py", b"other\n")))
    _write_tree(changed_content, (("a.py", b"changed\n"), ("b.py", b"other\n")))
    _write_tree(changed_path, (("moved.py", b"same\n"), ("b.py", b"other\n")))

    identity = twen_source_tree_sha256(baseline)
    assert identity != twen_source_tree_sha256(changed_content)
    assert identity != twen_source_tree_sha256(changed_path)


def test_source_tree_hash_rejects_an_empty_tree(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="contains no Python files"):
        twen_source_tree_sha256(tmp_path)


def test_repository_source_tree_identity_is_sha256() -> None:
    identity = twen_source_tree_sha256()
    assert len(identity) == 64
    assert set(identity) <= set("0123456789abcdef")


def test_checkpoint_metadata_records_source_tree_identity() -> None:
    captured: dict[str, object] = {}

    class Manager:
        def save(self, _stateful, **kwargs):
            captured.update(kwargs)
            return Path("checkpoint")

    source = SimpleNamespace(
        manifest_sha256="m" * 64,
        model_id="Qwen/test",
        revision="r" * 40,
        local_path="model",
    )
    config = SimpleNamespace(
        losses=SimpleNamespace(mtp=0.2),
        runtime=SimpleNamespace(
            activation_checkpointing=True,
            activation_checkpoint_layer_count=4,
            hidden_alignment_activation_checkpoint_layer_count=8,
            dense_transfer_token_checkpoint=True,
            dense_transfer_checkpoint_layer_count=0,
            hidden_alignment_dense_transfer_checkpoint_layer_count=16,
        ),
        data=SimpleNamespace(
            mode="prepared-text",
            manifest_sha256="d" * 64,
            teacher_kd_manifest_sha256=None,
            manifest_path="prepared.json",
            teacher_kd_manifest_path=None,
        ),
        optimizer=SimpleNamespace(
            adapter_optimizer="muon",
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_ns_coefficients=(3.4445, -4.775, 2.0315),
            muon_eps=1e-7,
            muon_ns_steps=5,
            muon_adjust_lr_fn="match_rms_adamw",
        ),
        sources=SimpleNamespace(
            backbone=source,
            donor=source,
            teacher=source,
            tokenizer=source,
            folded_experts_sha256=None,
            folded_experts_path=None,
        ),
    )
    report = SimpleNamespace(
        config_fingerprint="c" * 64,
        data_fingerprint="f" * 64,
        source_tree_sha256="s" * 64,
        calibration_fingerprints=(),
        activation_checkpoint_layer_indices=(0, 8, 15, 23),
        hidden_alignment_activation_checkpoint_layer_indices=(0, 3, 7, 10, 13, 16, 20, 23),
        dense_transfer_token_checkpoint_layer_indices=(),
        hidden_alignment_dense_transfer_token_checkpoint_layer_indices=(
            1,
            2,
            4,
            5,
            6,
            8,
            9,
            11,
            12,
            14,
            15,
            17,
            18,
            19,
            21,
            22,
        ),
    )
    state = SimpleNamespace(global_step=3, committed_tokens=4096)
    completed = SimpleNamespace(stdout="commit\n")
    with (
        patch("twen.training.engine.subprocess.run", return_value=completed),
        patch("twen.training.engine.RNGState.capture", return_value=object()),
        patch("twen.training.engine._runtime_cursor", return_value=object()),
    ):
        _checkpoint(
            Manager(),  # type: ignore[arg-type]
            {},
            state,  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            config,  # type: ignore[arg-type]
            report,  # type: ignore[arg-type]
            kind="periodic",
            boundary=None,
        )

    metadata = captured["extra_metadata"]
    assert isinstance(metadata, dict)
    assert metadata["source_tree_sha256"] == "s" * 64
    assert metadata["data_mode"] == "prepared-text"
    assert metadata["teacher_kd_manifest_sha256"] is None
    assert metadata["data_manifests"]["teacher_kd"] is None
    assert metadata["optimizer"] == {
        "adapter_optimizer": "muon",
        "bundle": True,
        "adapter_component": "Muon",
        "non_adapter_component": "AdamW",
        "muon": {
            "momentum": 0.95,
            "nesterov": True,
            "ns_coefficients": [3.4445, -4.775, 2.0315],
            "eps": 1e-7,
            "ns_steps": 5,
            "adjust_lr_fn": "match_rms_adamw",
        },
    }
    assert metadata["mtp"] == {
        "enabled": True,
        "loss_weight": 0.2,
        "source_role": "backbone",
        "source_manifest_sha256": "m" * 64,
        "parameters_frozen": True,
        "checkpointed_as_trainable_delta": False,
    }
    assert metadata["activation_checkpointing"] == {
        "enabled": True,
        "configured_layer_count": 4,
        "hidden_alignment_configured_layer_count": 8,
        "ordinary_layer_indices": [0, 8, 15, 23],
        "hidden_alignment_layer_indices": [0, 3, 7, 10, 13, 16, 20, 23],
    }
    assert metadata["dense_transfer_checkpointing"] == {
        "enabled": True,
        "ordinary_configured_layer_count": 0,
        "hidden_alignment_configured_layer_count": 16,
        "ordinary_layer_indices": [],
        "hidden_alignment_layer_indices": [
            1,
            2,
            4,
            5,
            6,
            8,
            9,
            11,
            12,
            14,
            15,
            17,
            18,
            19,
            21,
            22,
        ],
    }
    assert metadata["quality_cooldown"] == {
        "enabled": False,
        "start_tokens": None,
        "prepared_manifest_sha256": None,
        "teacher_kd_manifest_sha256": None,
    }
