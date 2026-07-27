from __future__ import annotations

from pathlib import Path


def test_dashboard_user_service_is_persistent_and_cuda_wrapped() -> None:
    root = Path(__file__).parents[1]
    unit = (root / "deploy/systemd/twen-dashboard.service").read_text(encoding="utf-8")

    assert "Restart=always" in unit
    assert "LimitNOFILE=1048576" in unit
    assert "UMask=0077" in unit
    assert " --host 0.0.0.0 --port 8765 " in unit
    assert (
        "ExecStartPre=/media/data1/Project/AI/Twen1/.venv/bin/python "
        "/media/data1/Project/AI/Twen1/scripts/"
        "authorize_v4_13m_calibration.py verify "
    ) in unit
    assert (
        "--dashboard-config "
        "locks/base-dense-v4-13m-calibration-admission-pass-002/"
        "dashboard.json "
    ) in unit
    assert (
        "ReadOnlyPaths=/media/data1/Project/AI/Twen1/locks/"
        "base-dense-v4-13m-calibration-admission-pass-002"
    ) in unit
    assert (
        "ExecStart=/usr/bin/bash "
        "/media/data1/Project/AI/Twen1/scripts/with_cuda_toolchain.sh "
        "/media/data1/Project/AI/Twen1/.venv/bin/python -m twen web serve "
    ) in unit
