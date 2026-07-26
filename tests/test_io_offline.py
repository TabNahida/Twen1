from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from twen.io.download import ArtifactSpec, write_artifact_manifest, write_download_set_manifest
from twen.io.offline import LocalAsset, assert_training_offline_ready, run_offline_preflight


class OfflinePreflightTest(unittest.TestCase):
    def test_exact_file_is_verified_and_environment_forced_offline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            payload = b"tokenizer"
            path.write_bytes(payload)
            spec = ArtifactSpec.http(
                url="https://example.invalid/tokenizer.json",
                source_id="fixture",
                revision="v1",
                expected_size=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
            write_artifact_manifest(path, spec)
            environment: dict[str, str] = {}
            report = assert_training_offline_ready(
                [LocalAsset("tokenizer", path, kind="file")], environment=environment
            )
            self.assertTrue(report.ok)
            self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
            self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")

    def test_directory_set_and_incomplete_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model"
            root.mkdir()
            file = root / "config.json"
            file.write_bytes(b"{}")
            spec = ArtifactSpec.http(
                url="https://example.invalid/config.json",
                source_id="fixture",
                revision="v1",
                expected_size=2,
                sha256=hashlib.sha256(b"{}").hexdigest(),
            )
            manifest = write_download_set_manifest(root / "download-manifest.json", [spec])
            asset = LocalAsset("model", root, kind="directory", manifest=manifest)
            self.assertTrue(run_offline_preflight([asset]).ok)
            (root / "weights.incomplete").write_bytes(b"partial")
            report = run_offline_preflight([asset])
            self.assertFalse(report.ok)
            self.assertIn("incomplete", report.errors[0])


if __name__ == "__main__":
    unittest.main()
