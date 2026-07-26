from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")
import torch.distributed.checkpoint as dcp  # noqa: E402


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "audit_dense_checkpoint_drift.py"
    spec = importlib.util.spec_from_file_location(
        "audit_dense_checkpoint_drift",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit_script = _load_script()


def _checkpoint(
    root: Path,
    *,
    adapter: torch.Tensor,
    scale: torch.Tensor,
) -> Path:
    root.mkdir(parents=True)
    dcp.save(
        {
            "model.layer.adapters.weight": adapter,
            "model.layer.branch_scale": scale,
        },
        checkpoint_id=root / "state",
    )
    (root / "manifest.json").write_text('{"fixture":true}\n', encoding="utf-8")
    (root / "COMPLETE").write_text("fixture\n", encoding="utf-8")
    return root


def test_audit_measures_adapter_and_scale_drift_and_atomically_writes_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline = _checkpoint(
        tmp_path / "baseline",
        adapter=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        scale=torch.tensor([0.02]),
    )
    candidate = _checkpoint(
        tmp_path / "candidate",
        adapter=torch.tensor([[1.0, 2.0], [3.0, 5.0]]),
        scale=torch.tensor([0.01]),
    )
    output = tmp_path / "report" / "analysis.json"

    assert (
        audit_script.main(
            [
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    rendered = json.loads(capsys.readouterr().out)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == rendered
    assert not list(output.parent.glob(".*.tmp"))
    assert persisted["execution"] == {
        "cuda_initialized": False,
        "device": "cpu",
        "model_built": False,
        "optimizer_created": False,
    }
    assert persisted["inventory"] == {
        "adapter_element_count": 4,
        "adapter_tensor_count": 1,
        "model_tensor_count": 2,
        "scale_element_count": 1,
        "scale_tensor_count": 1,
    }
    measured = persisted["candidates"][0]
    assert measured["adapter"]["delta_rms"] == pytest.approx(0.5)
    assert measured["scale"]["relative_l2"] == pytest.approx(0.5)
    assert measured["scale_values"]["decreased_count"] == 1
    assert measured["scale_values"]["increased_count"] == 0
