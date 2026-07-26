from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts/profile_dense_gemm_ncu.py"
    spec = importlib.util.spec_from_file_location("profile_dense_gemm_ncu", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("operation", ["forward", "input-grad", "weight-grad"])
def test_dry_run_contract_has_production_b2_shapes(operation: str) -> None:
    module = _module()
    args = module._parser().parse_args(["--operation", operation, "--dry-run"])
    module._validate(args)
    contract = module._contract(args)

    assert contract["logical_tokens"] == 8192
    assert contract["dtype"] == "bfloat16"
    assert contract["nominal_flop"] > 0
    assert contract["no_optimizer_created"] is True
    assert contract["optimizer_steps"] == 0
    assert contract["parameter_updates"] is False


def test_probe_rejects_invalid_geometry() -> None:
    module = _module()
    args = module._parser().parse_args(["--batch-size", "0", "--dry-run"])
    with pytest.raises(ValueError, match="batch_size"):
        module._validate(args)
