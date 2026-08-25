from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, TypeAlias


ReasoningSupport: TypeAlias = Literal[
    "unsupported", "optional", "mandatory", "unknown"
]
ReasoningControl: TypeAlias = Literal[
    "none", "boolean", "effort", "thinking_type", "adaptive"
]
ThinkingIntent: TypeAlias = Literal["enable", "disable", "default"]


@dataclass(frozen=True)
class ReasoningCapability:
    """Provider/model reasoning behavior expressed independently of wire format."""

    support: ReasoningSupport
    control: ReasoningControl
    supported_efforts: tuple[str, ...] = ()
    source: str = "builtin"


@dataclass(frozen=True)
class ReasoningDecision:
    """Declarative request update produced by a provider adapter."""

    capability: ReasoningCapability
    updates: Mapping[str, Any] = field(default_factory=dict)
    removable_fields: frozenset[str] = frozenset()
    warning: str | None = None

