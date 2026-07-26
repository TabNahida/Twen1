"""Fail-closed offline preflight for training processes."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path

from .download import (
    DOWNLOAD_MANIFEST_SCHEMA,
    ArtifactSpec,
    load_artifact_manifest,
    sha256_file,
    verify_artifact,
)

OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "WANDB_MODE": "offline",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


class OfflinePreflightError(RuntimeError):
    """Raised before training when a local dependency is incomplete."""


def enforce_offline_environment(
    environment: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Force libraries into offline mode and return loader-safe arguments."""

    target = os.environ if environment is None else environment
    target.update(OFFLINE_ENVIRONMENT)
    return {"local_files_only": True}


@dataclass(frozen=True, slots=True)
class LocalAsset:
    """One model, tokenizer, data tree, or exact local file needed by training."""

    name: str
    path: str | os.PathLike[str]
    kind: str = "auto"
    manifest: str | os.PathLike[str] | None = None
    expected_size: int | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("asset name is required")
        if self.kind not in {"auto", "file", "directory"}:
            raise ValueError("kind must be auto, file, or directory")
        if (self.expected_size is None) != (self.sha256 is None):
            raise ValueError("expected_size and sha256 must be supplied together")


@dataclass(slots=True)
class OfflinePreflightReport:
    checked_assets: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            details = "\n  - ".join(self.errors)
            raise OfflinePreflightError(f"offline training preflight failed:\n  - {details}")


def _verify_set_manifest(root: Path, manifest_path: Path) -> None:
    with manifest_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("schema_version") != DOWNLOAD_MANIFEST_SCHEMA:
        raise OfflinePreflightError(f"unsupported manifest schema: {manifest_path}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise OfflinePreflightError(f"empty artifact set manifest: {manifest_path}")
    for raw_spec in artifacts:
        spec = ArtifactSpec(**raw_spec)
        verify_artifact(root / spec.filename, spec)


def verify_local_download_directory(
    root: str | os.PathLike[str],
    *,
    expected_manifest_sha256: str,
) -> Path:
    """Verify a locally downloaded artifact set before an offline loader uses it."""

    directory = Path(root)
    if not directory.is_dir():
        raise OfflinePreflightError(f"download directory is missing: {directory}")
    incomplete = next(directory.rglob("*.incomplete"), None)
    if incomplete is not None:
        raise OfflinePreflightError(f"download directory contains {incomplete}")
    manifest = directory / "download-manifest.json"
    if not manifest.is_file():
        raise OfflinePreflightError(f"download manifest is missing: {manifest}")
    actual = sha256_file(manifest)
    if actual.lower() != expected_manifest_sha256.lower():
        raise OfflinePreflightError(
            "download manifest SHA256 mismatch: "
            f"expected {expected_manifest_sha256}, got {actual}"
        )
    _verify_set_manifest(directory, manifest)
    return manifest


def run_offline_preflight(
    assets: Iterable[LocalAsset],
    *,
    require_manifests: bool = True,
    environment: MutableMapping[str, str] | None = None,
) -> OfflinePreflightReport:
    """Verify local assets and force all common loaders offline.

    Call this before constructing a Transformers tokenizer/model.  No library
    SDK is imported and no network fallback is attempted.
    """

    enforce_offline_environment(environment)
    report = OfflinePreflightReport()
    for asset in assets:
        path = Path(asset.path)
        report.checked_assets.append(asset.name)
        if not path.exists():
            report.errors.append(f"{asset.name}: missing {path}")
            continue
        expected_kind = asset.kind
        if expected_kind == "file" and not path.is_file():
            report.errors.append(f"{asset.name}: expected file, got {path}")
            continue
        if expected_kind == "directory" and not path.is_dir():
            report.errors.append(f"{asset.name}: expected directory, got {path}")
            continue

        if path.is_dir():
            incomplete = next(path.rglob("*.incomplete"), None)
            if incomplete is not None:
                report.errors.append(f"{asset.name}: incomplete artifact {incomplete}")
                continue

        manifest = Path(asset.manifest) if asset.manifest is not None else None
        if manifest is None and path.is_file():
            candidate = path.with_name(f"{path.name}.manifest.json")
            if candidate.exists():
                manifest = candidate
        if require_manifests and manifest is None:
            report.errors.append(f"{asset.name}: no integrity manifest configured")
            continue
        if manifest is not None and not manifest.is_file():
            report.errors.append(f"{asset.name}: missing manifest {manifest}")
            continue

        try:
            if asset.expected_size is not None and asset.sha256 is not None:
                direct_spec = ArtifactSpec.http(
                    url="https://offline.invalid/artifact",
                    source_id=asset.name,
                    revision="local-pinned",
                    filename=path.name,
                    expected_size=asset.expected_size,
                    sha256=asset.sha256,
                )
                verify_artifact(path, direct_spec)
            if manifest is not None:
                with manifest.open("r", encoding="utf-8") as file:
                    manifest_payload = json.load(file)
                if "artifacts" in manifest_payload:
                    root = path if path.is_dir() else path.parent
                    _verify_set_manifest(root, manifest)
                elif "artifact" in manifest_payload:
                    spec, manifested_path = load_artifact_manifest(manifest)
                    verify_artifact(manifested_path, spec)
                    if path.is_file() and manifested_path.resolve() != path.resolve():
                        raise OfflinePreflightError(
                            f"manifest {manifest} points to {manifested_path}, not {path}"
                        )
                else:
                    raise OfflinePreflightError(f"unknown manifest layout: {manifest}")
        except (OSError, ValueError, KeyError, TypeError, OfflinePreflightError) as error:
            report.errors.append(f"{asset.name}: {error}")
    return report


def assert_training_offline_ready(
    assets: Iterable[LocalAsset],
    *,
    require_manifests: bool = True,
    environment: MutableMapping[str, str] | None = None,
) -> OfflinePreflightReport:
    report = run_offline_preflight(
        assets,
        require_manifests=require_manifests,
        environment=environment,
    )
    report.raise_for_errors()
    return report
