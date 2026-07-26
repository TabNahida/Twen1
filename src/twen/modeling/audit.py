"""Offline audit of the two dense Qwen3.5 source configs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigAuditError

_DENSE_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_text"})


@dataclass(frozen=True, slots=True)
class Qwen35Shape:
    """The architecture fields used by the transfer implementation."""

    role: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    layer_types: tuple[str, ...]
    model_type: str | None
    source: str

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        return tuple(i for i, kind in enumerate(self.layer_types) if kind == "full")


@dataclass(frozen=True, slots=True)
class SourceConfigAudit:
    """Validated model shapes for the small backbone and 9B donor."""

    backbone: Qwen35Shape
    donor: Qwen35Shape
    compatible: bool = True

    @property
    def student_layer_types(self) -> tuple[str, ...]:
        return self.backbone.layer_types

    @property
    def donor_layer_types(self) -> tuple[str, ...]:
        return self.donor.layer_types


def _as_config_dict(config: Mapping[str, Any] | str | Path | Any) -> tuple[dict[str, Any], str]:
    if isinstance(config, Mapping):
        raw = dict(config)
        source = "<mapping>"
    elif isinstance(config, (str, Path)):
        path = Path(config).expanduser()
        if path.is_dir():
            path = path / "config.json"
        if not path.is_file():
            raise ConfigAuditError(
                f"Config does not exist locally: {path}. Auditing never downloads a config."
            )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigAuditError(f"Cannot read valid JSON config from {path}: {exc}") from exc
        source = str(path.resolve())
    elif hasattr(config, "to_dict"):
        raw = dict(config.to_dict())
        source = f"<{type(config).__module__}.{type(config).__qualname__}>"
    else:
        raise TypeError(
            "config must be a mapping, a local config.json path/directory, or expose to_dict()"
        )

    # Official multimodal configs wrap the language config. v1 deliberately audits
    # and exports only that text component.
    text = raw.get("text_config")
    if isinstance(text, Mapping):
        raw = dict(text)
        source = f"{source}#text_config"
    return raw, source


def _positive_int(raw: Mapping[str, Any], field: str, source: str) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigAuditError(f"{source}: {field} must be a positive integer, got {value!r}")
    return value


def normalize_layer_type(value: Any) -> str:
    """Normalize Qwen/Transformers spelling to ``linear`` or ``full``."""

    key = str(value).strip().lower().replace("-", "_")
    aliases = {
        "linear": "linear",
        "linear_attention": "linear",
        "gated_delta_net": "linear",
        "full": "full",
        "full_attention": "full",
        "attention": "full",
        "sliding_attention": "full",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ConfigAuditError(
            f"Unknown layer type {value!r}; expected linear_attention or full_attention"
        ) from exc


def _layer_types(raw: Mapping[str, Any], layers: int, source: str) -> tuple[str, ...]:
    declared = raw.get("layer_types")
    if declared is not None:
        if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes)):
            raise ConfigAuditError(f"{source}: layer_types must be a sequence")
        if len(declared) != layers:
            raise ConfigAuditError(
                f"{source}: layer_types has {len(declared)} entries, expected {layers}"
            )
        return tuple(normalize_layer_type(item) for item in declared)

    interval = raw.get("full_attention_interval", 4)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise ConfigAuditError(
            f"{source}: full_attention_interval must be a positive integer, got {interval!r}"
        )
    # Qwen3.5's published dense checkpoints use three linear layers followed by
    # one full-attention layer. The final layer in every interval is full.
    return tuple("full" if (i + 1) % interval == 0 else "linear" for i in range(layers))


def _audit_one(
    config: Mapping[str, Any] | str | Path | Any,
    *,
    role: str,
    expected_hidden: int,
    expected_intermediate: int,
    expected_layers: int,
) -> Qwen35Shape:
    raw, source = _as_config_dict(config)
    model_type_raw = raw.get("model_type")
    model_type = str(model_type_raw) if model_type_raw is not None else None
    if model_type is None:
        raise ConfigAuditError(f"{source}: {role} config is missing model_type")
    if model_type not in _DENSE_MODEL_TYPES:
        raise ConfigAuditError(
            f"{source}: {role} must be a dense Qwen3.5 text config; "
            f"model_type={model_type!r} is unsupported"
        )

    hidden = _positive_int(raw, "hidden_size", source)
    intermediate = _positive_int(raw, "intermediate_size", source)
    layers = _positive_int(raw, "num_hidden_layers", source)
    expected = (expected_hidden, expected_intermediate, expected_layers)
    actual = (hidden, intermediate, layers)
    if actual != expected:
        raise ConfigAuditError(
            f"{source}: unexpected {role} shape (hidden, ffn, layers)={actual}; "
            f"this recipe requires {expected}"
        )

    layer_types = _layer_types(raw, layers, source)
    expected_pattern = tuple(
        "full" if (i + 1) % 4 == 0 else "linear" for i in range(expected_layers)
    )
    if layer_types != expected_pattern:
        raise ConfigAuditError(
            f"{source}: {role} does not follow Qwen3.5's 3-linear/1-full layer pattern"
        )

    # SwiGLU folding relies on all three projections being bias-free. Official
    # configs normally omit this flag, meaning false.
    if raw.get("mlp_bias", False) is not False or raw.get("use_mlp_bias", False) is not False:
        raise ConfigAuditError(f"{source}: biased MLP projections cannot be folded exactly")
    if str(raw.get("hidden_act", "silu")).lower() not in {"silu", "swish"}:
        raise ConfigAuditError(
            f"{source}: {role} must use SwiGLU/SILU; hidden_act={raw.get('hidden_act')!r}"
        )

    return Qwen35Shape(
        role=role,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=layers,
        layer_types=layer_types,
        model_type=model_type,
        source=source,
    )


def audit_source_configs(
    backbone_config: Mapping[str, Any] | str | Path | Any,
    donor_config: Mapping[str, Any] | str | Path | Any,
) -> SourceConfigAudit:
    """Validate the exact 0.8B-backbone/9B-donor architecture pair.

    Paths are strictly local. This function never imports Transformers and never
    invokes ``from_pretrained``.
    """

    backbone = _audit_one(
        backbone_config,
        role="Qwen3.5-0.8B backbone",
        expected_hidden=1024,
        expected_intermediate=3584,
        expected_layers=24,
    )
    donor = _audit_one(
        donor_config,
        role="Qwen3.5-9B donor",
        expected_hidden=4096,
        expected_intermediate=12288,
        expected_layers=32,
    )
    if backbone.full_attention_layers != (3, 7, 11, 15, 19, 23):
        raise ConfigAuditError("Backbone full-attention positions do not match the transfer recipe")
    if donor.full_attention_layers != (3, 7, 11, 15, 19, 23, 27, 31):
        raise ConfigAuditError("Donor full-attention positions do not match the transfer recipe")
    return SourceConfigAudit(backbone=backbone, donor=donor)
