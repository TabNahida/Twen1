"""Exceptions raised by the Twen model-conversion subsystem."""

from __future__ import annotations


class ModelingError(RuntimeError):
    """Base class for model auditing and conversion errors."""


class OptionalDependencyError(ModelingError):
    """A lazily loaded optional dependency is unavailable."""


class ConfigAuditError(ModelingError, ValueError):
    """A source checkpoint config is incompatible with the transfer recipe."""


class ShapeError(ModelingError, ValueError):
    """Tensor or matrix dimensions violate a model invariant."""


class LayerMappingError(ModelingError, ValueError):
    """A monotonic, layer-type-compatible mapping cannot be constructed."""


class ExportError(ModelingError, ValueError):
    """Native MoE export inputs are incomplete or inconsistent."""
