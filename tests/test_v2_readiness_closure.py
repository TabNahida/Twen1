from __future__ import annotations

from types import SimpleNamespace

import pytest

from twen.data.quality_policy import DEFAULT_QUALITY_SOURCE_TOKEN_TARGETS
from twen.v2_finalizer import (
    FinalizationError,
    FinalizationOptions,
    _authenticate_kd_orchestration,
    _authenticate_quality_cooldown_publication,
    _validate_locked_quality_source_mix,
)


def _options(tmp_path):
    return FinalizationOptions.repository_defaults(
        root=tmp_path,
        mtp_loss_weight=0.1,
        adapter_lr=2e-4,
        router_lr=1e-3,
        lora_lr=2e-4,
        scale_lr=1e-3,
        quality_cooldown_prepared_manifest=(
            "artifacts/data/base-v2-500m-quality-bundle/prepared/manifest.json"
        ),
        quality_cooldown_kd_manifest=(
            "artifacts/data/base-v2-500m-quality-bundle/kd/manifest.json"
        ),
    )


def test_finalizer_requires_kd_orchestration_complete(tmp_path) -> None:
    options = _options(tmp_path)

    with pytest.raises(FinalizationError, match="KD orchestration is not complete"):
        _authenticate_kd_orchestration(
            options,
            prepared_identity={"path": "prepared", "size": 1, "sha256": "1" * 64},
            kd_identity={"path": "kd", "size": 1, "sha256": "2" * 64},
            audit_path=tmp_path / "audit.json",
        )


def test_finalizer_requires_published_cooldown_bundle_complete(tmp_path) -> None:
    options = _options(tmp_path)
    bundle_root = options.quality_cooldown_prepared_manifest.parent.parent
    bundle_root.mkdir(parents=True)

    with pytest.raises(FinalizationError, match="invalid or missing JSON object"):
        _authenticate_quality_cooldown_publication(
            options,
            primary_prepared_identity={"path": "prepared", "size": 1, "sha256": "1" * 64},
            primary_kd_identity={"path": "kd", "size": 1, "sha256": "2" * 64},
            cooldown_prepared=SimpleNamespace(),
            cooldown_kd=SimpleNamespace(),
            cooldown_prepared_identity={
                "path": "cooldown-prepared",
                "size": 1,
                "sha256": "3" * 64,
            },
            cooldown_kd_identity={
                "path": "cooldown-kd",
                "size": 1,
                "sha256": "4" * 64,
            },
            summary=SimpleNamespace(),
        )


def test_finalizer_locks_all_six_quality_source_minimums() -> None:
    exact = SimpleNamespace(source_mix_token_counts=DEFAULT_QUALITY_SOURCE_TOKEN_TARGETS)
    assert _validate_locked_quality_source_mix(exact) == dict(DEFAULT_QUALITY_SOURCE_TOKEN_TARGETS)

    missing = SimpleNamespace(source_mix_token_counts=DEFAULT_QUALITY_SOURCE_TOKEN_TARGETS[:-1])
    with pytest.raises(FinalizationError, match="six-source minimum quotas"):
        _validate_locked_quality_source_mix(missing)

    under = list(DEFAULT_QUALITY_SOURCE_TOKEN_TARGETS)
    source_id, minimum = under[0]
    under[0] = (source_id, minimum - 1)
    with pytest.raises(FinalizationError, match="six-source minimum quotas"):
        _validate_locked_quality_source_mix(SimpleNamespace(source_mix_token_counts=tuple(under)))
