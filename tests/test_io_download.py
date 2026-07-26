from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError
from urllib.request import ProxyHandler, Request

from twen.io.download import (
    ArtifactIntegrityError,
    ArtifactSpec,
    ArtifactValidationError,
    DownloadManager,
    _GitHubOnlyProxyHandler,
    load_artifact_manifest,
    resolve_model_manifest,
    uses_github_proxy,
)


class _Response(io.BytesIO):
    def __init__(self, data: bytes, status: int) -> None:
        super().__init__(data)
        self.status = status

    def getcode(self) -> int:
        return self.status


class _Opener:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        return self.response


class _FailingOpener:
    def open(self, request, timeout):
        raise URLError("direct route unavailable")


class _TimeoutOpener:
    def open(self, request, timeout):
        raise TimeoutError("direct route timed out")


class DownloadTest(unittest.TestCase):
    def test_only_github_hosts_are_proxy_routed(self) -> None:
        self.assertTrue(uses_github_proxy("https://github.com/org/repo/releases/file"))
        self.assertTrue(uses_github_proxy("https://release-assets.githubusercontent.com/file"))
        self.assertFalse(uses_github_proxy("https://huggingface.co/Qwen/model"))
        self.assertFalse(uses_github_proxy("https://cdn-lfs.huggingface.co/file"))
        self.assertFalse(uses_github_proxy("https://notgithub.com/file"))

    def test_proxy_handler_rechecks_every_request_host(self) -> None:
        handler = _GitHubOnlyProxyHandler(
            {"http": "http://proxy.example:8080", "https": "http://proxy.example:8080"}
        )
        with patch.object(ProxyHandler, "proxy_open", return_value="proxied") as proxy_open:
            direct = handler.proxy_open(
                Request("https://huggingface.co/Qwen/model"),
                "http://proxy.example:8080",
                "https",
            )
            self.assertIsNone(direct)
            proxy_open.assert_not_called()
            proxied = handler.proxy_open(
                Request("https://release-assets.githubusercontent.com/file"),
                "http://proxy.example:8080",
                "https",
            )
            self.assertEqual(proxied, "proxied")
            proxy_open.assert_called_once()

    @patch("twen.io.download.check_proxy_connectivity")
    def test_huggingface_never_checks_github_proxy(self, check_proxy) -> None:
        manager = DownloadManager(check_proxy=True, use_proxy=True)
        manager._check_proxy_once("https://huggingface.co/api/models/Qwen/example")
        check_proxy.assert_not_called()
        manager._check_proxy_once("https://api.github.com/repos/org/repo")
        check_proxy.assert_called_once()

    def test_default_policy_falls_huggingface_back_to_proxy(self) -> None:
        manager = DownloadManager(check_proxy=False)
        success = _Opener(_Response(b"{}", 200))
        openers = iter((_FailingOpener(), success))
        manager._opener = lambda: next(openers)
        response = manager._open(Request("https://huggingface.co/api/models/Qwen/example"))
        self.assertEqual(response.read(), b"{}")
        self.assertTrue(manager.proxy_fallback_used)

    def test_huggingface_fallback_retries_once_through_proxy(self) -> None:
        manager = DownloadManager(
            check_proxy=False,
            network_policy="fallback",
        )
        success = _Opener(_Response(b"{}", 200))
        openers = iter((_FailingOpener(), success))
        manager._opener = lambda: next(openers)
        response = manager._open(Request("https://huggingface.co/api/models/Qwen/example"))
        self.assertEqual(response.read(), b"{}")
        self.assertTrue(manager.proxy_fallback_used)
        self.assertEqual(manager.effective_network_policy, "proxy-fallback")

    def test_huggingface_timeout_also_falls_back_to_proxy(self) -> None:
        manager = DownloadManager(check_proxy=False, network_policy="fallback")
        success = _Opener(_Response(b"{}", 200))
        openers = iter((_TimeoutOpener(), success))
        manager._opener = lambda: next(openers)
        response = manager._open(Request("https://huggingface.co/api/models/Qwen/example"))
        self.assertEqual(response.read(), b"{}")
        self.assertTrue(manager.proxy_fallback_used)

    def test_fallback_never_retries_an_unrelated_host(self) -> None:
        manager = DownloadManager(check_proxy=False, network_policy="fallback")
        manager._opener = lambda: _FailingOpener()
        with self.assertRaises(URLError):
            manager._open(Request("https://pypi.org/simple/tqdm/"))
        self.assertFalse(manager.proxy_fallback_used)

    def test_commit_revisions_are_lowercase_but_immutable_tags_preserve_case(self) -> None:
        commit = ArtifactSpec.http(
            url="https://example.invalid/model.bin",
            revision="A" * 40,
            expected_size=0,
            sha256="B" * 64,
        )
        tag = ArtifactSpec.http(
            url="https://example.invalid/model.bin",
            revision="Release-V1",
            expected_size=0,
            sha256="B" * 64,
        )
        self.assertEqual(commit.revision, "a" * 40)
        self.assertEqual(tag.revision, "Release-V1")

    def test_resume_range_verify_and_manifest(self) -> None:
        payload = b"a verified resumable payload"
        digest = hashlib.sha256(payload).hexdigest()
        spec = ArtifactSpec.http(
            url="https://example.invalid/file.bin",
            source_id="fixture",
            revision="v1.2.3",
            filename="file.bin",
            expected_size=len(payload),
            sha256=digest,
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "file.bin"
            prefix = payload[:9]
            destination.with_name("file.bin.incomplete").write_bytes(prefix)
            opener = _Opener(_Response(payload[len(prefix) :], 206))
            manager = DownloadManager(check_proxy=False, use_proxy=False, chunk_size=4)
            manager._opener = lambda: opener
            result = manager.download(spec, destination)
            self.assertEqual(result.read_bytes(), payload)
            request = opener.requests[0][0]
            self.assertEqual(request.get_header("Range"), f"bytes={len(prefix)}-")
            manifest_path = destination.with_name("file.bin.manifest.json")
            manifested_spec, manifested_file = load_artifact_manifest(manifest_path)
            self.assertEqual(manifested_spec, spec)
            self.assertEqual(manifested_file, destination)

    def test_server_ignoring_range_restarts_partial(self) -> None:
        payload = b"whole content"
        spec = ArtifactSpec.http(
            url="https://example.invalid/file.bin",
            revision="immutable-v2",
            expected_size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "file.bin"
            destination.with_name("file.bin.incomplete").write_bytes(b"stale")
            manager = DownloadManager(check_proxy=False, use_proxy=False)
            manager._opener = lambda: _Opener(_Response(payload, 200))
            manager.download(spec, destination)
            self.assertEqual(destination.read_bytes(), payload)

    def test_invalid_final_file_is_never_overwritten(self) -> None:
        payload = b"expected"
        spec = ArtifactSpec.http(
            url="https://example.invalid/file.bin",
            revision="immutable-v2",
            expected_size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "file.bin"
            destination.write_bytes(b"bad")
            with self.assertRaises(ArtifactIntegrityError):
                DownloadManager(check_proxy=False, use_proxy=False).download(spec, destination)
            self.assertEqual(destination.read_bytes(), b"bad")

    def test_floating_revision_is_rejected_for_artifacts(self) -> None:
        with self.assertRaises(ArtifactValidationError):
            ArtifactSpec.http(
                url="https://example.invalid/file",
                revision="main",
                expected_size=0,
                sha256=hashlib.sha256(b"").hexdigest(),
            )

    def test_huggingface_resolver_locks_commit_and_hashes_missing_metadata(self) -> None:
        config = b"{}"
        weights_hash = hashlib.sha256(b"weights").hexdigest()
        metadata = {
            "sha": "1" * 40,
            "siblings": [
                {"rfilename": "config.json", "size": len(config)},
                {
                    "rfilename": "model.safetensors",
                    "size": 7,
                    "lfs": {"sha256": weights_hash, "size": 7},
                },
            ],
        }
        hashed_urls: list[str] = []

        def content_hasher(url, headers):
            hashed_urls.append(url)
            return len(config), hashlib.sha256(config).hexdigest()

        manifest = resolve_model_manifest(
            "huggingface",
            "Qwen/example",
            "main",
            metadata=metadata,
            content_hasher=content_hasher,
            manager=DownloadManager(check_proxy=False, use_proxy=False),
        )
        self.assertEqual(manifest.resolved_revision, "1" * 40)
        self.assertEqual([item.filename for item in manifest.artifacts], ["config.json", "model.safetensors"])
        self.assertEqual(len(hashed_urls), 1)
        self.assertIn("/resolve/" + "1" * 40 + "/config.json", hashed_urls[0])
        json.dumps(manifest.to_dict())

    def test_modelscope_resolver_parses_provider_fields(self) -> None:
        digest = hashlib.sha256(b"abc").hexdigest()
        metadata = {
            "Data": {
                "Revision": "2" * 40,
                "Files": [
                    {"Path": "dir", "Type": "tree"},
                    {"Path": "config.json", "Type": "blob", "Size": 3, "Sha256": digest},
                ],
            }
        }
        manifest = resolve_model_manifest(
            "modelscope",
            "Qwen/example",
            "v1",
            metadata=metadata,
            content_hasher=lambda *_: self.fail("metadata hash should be sufficient"),
            manager=DownloadManager(check_proxy=False, use_proxy=False),
        )
        self.assertEqual(len(manifest.artifacts), 1)
        self.assertEqual(manifest.artifacts[0].sha256, digest)


if __name__ == "__main__":
    unittest.main()
