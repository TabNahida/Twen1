"""Qwen3.5 dense-to-MoE transfer building blocks.

All audit/mapping metadata helpers import without PyTorch. Module classes listed
below are resolved lazily on first access.
"""

from __future__ import annotations

from typing import Any

from .audit import Qwen35Shape, SourceConfigAudit, audit_source_configs
from .errors import (
    ConfigAuditError,
    ExportError,
    LayerMappingError,
    ModelingError,
    OptionalDependencyError,
    ShapeError,
)
from .export import (
    build_native_moe_config,
    export_native_moe_mtp_state,
    export_native_moe_state,
)
from .folding import FoldedExperts, fold_expert_weights, max_fold_error
from .mapping import (
    LayerMatch,
    LayerPair,
    linear_cka,
    linear_cka_matrix,
    match_layers_cka,
)
from .ridge import BidirectionalRidgeStats, RidgeSolution
from .slicing import (
    ChannelPartition,
    build_channel_partition,
    channel_contribution_scores,
)

_LAZY_MODULE_CLASSES = frozenset(
    {
        "TransferAdapters",
        "DenseTransferMLP",
        "SharedDenseTransferMLP",
        "MergeableExpertLoRA",
        "SparseTransferMLP",
    }
)

__all__ = [
    "BidirectionalRidgeStats",
    "ChannelPartition",
    "ConfigAuditError",
    "DenseTransferMLP",
    "ExportError",
    "FoldedExperts",
    "LayerMappingError",
    "LayerMatch",
    "LayerPair",
    "MergeableExpertLoRA",
    "ModelingError",
    "OptionalDependencyError",
    "Qwen35Shape",
    "RidgeSolution",
    "ShapeError",
    "SharedDenseTransferMLP",
    "SourceConfigAudit",
    "SparseTransferMLP",
    "TransferAdapters",
    "audit_source_configs",
    "build_channel_partition",
    "build_native_moe_config",
    "channel_contribution_scores",
    "export_native_moe_mtp_state",
    "export_native_moe_state",
    "fold_expert_weights",
    "linear_cka",
    "linear_cka_matrix",
    "match_layers_cka",
    "max_fold_error",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULE_CLASSES:
        from . import modules

        value = getattr(modules, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
