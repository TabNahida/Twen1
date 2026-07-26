from __future__ import annotations

from pathlib import Path


def test_dashboard_user_service_is_persistent_and_cuda_wrapped() -> None:
    root = Path(__file__).parents[1]
    unit = (root / "deploy/systemd/twen-dashboard.service").read_text(encoding="utf-8")

    assert "Restart=always" in unit
    assert "UMask=0077" in unit
    assert " --host 0.0.0.0 --port 8765 " in unit
    assert (
        "ExecStart=/usr/bin/bash "
        "/media/data1/Project/AI/Twen1/scripts/with_cuda_toolchain.sh "
        "/media/data1/Project/AI/Twen1/.venv/bin/python -m twen web serve "
    ) in unit
