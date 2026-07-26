from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPORTS_ROOT = Path(__file__).parents[1] / "docs" / "reports"
PUBLISHED_BUNDLES = (
    "base-dense-v1-final-validation",
    "base-dense-v2-500m-final-validation",
    "base-dense-v3-500m-final-validation",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("bundle_name", PUBLISHED_BUNDLES)
def test_published_report_bundle_identity_is_complete(bundle_name: str) -> None:
    root = REPORTS_ROOT / bundle_name
    manifest_path = root / "MANIFEST.json"
    complete_path = root / "COMPLETE"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest["files"]

    actual_payloads = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"MANIFEST.json", "COMPLETE"}
    }
    assert actual_payloads == set(declared)
    for relative_path, identity in declared.items():
        payload = root / relative_path
        assert payload.stat().st_size == identity["size"]
        assert _sha256(payload) == identity["sha256"]

    assert complete_path.read_text(encoding="utf-8").strip() == _sha256(manifest_path)
