"""Small dependency helpers.

Neither torch nor transformers is imported while importing :mod:`twen.modeling`.
This matters for manifest/audit tools, which should work in a lightweight CPU
environment without accidentally initializing CUDA.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

from .errors import OptionalDependencyError


def require_torch(feature: str) -> ModuleType:
    """Import and return torch only when a tensor feature is invoked."""

    try:
        return import_module("torch")
    except (ImportError, OSError) as exc:  # OSError also covers broken CUDA wheels.
        raise OptionalDependencyError(
            f"{feature} requires PyTorch, but torch could not be imported. "
            "Install the project's training dependencies before using this feature."
        ) from exc


def require_transformers(feature: str) -> ModuleType:
    """Import and return transformers only for an explicitly requested feature."""

    try:
        return import_module("transformers")
    except (ImportError, OSError) as exc:
        raise OptionalDependencyError(
            f"{feature} requires Transformers, but transformers could not be imported. "
            "Install the project's model dependencies first."
        ) from exc
