from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from docutranslate.agents.provider.provider import ProviderType

from .capability import (
    ReasoningCapability,
    ReasoningControl,
    ReasoningDecision,
    ReasoningSupport,
    ThinkingIntent,
)


def reasoning_decision(
    support: ReasoningSupport,
    control: ReasoningControl,
    updates: dict[str, Any] | None = None,
    *,
    efforts: tuple[str, ...] = (),
    warning: str | None = None,
) -> ReasoningDecision:
    """Build a provider-independent reasoning decision."""
    request_updates = updates or {}
    return ReasoningDecision(
        capability=ReasoningCapability(
            support=support,
            control=control,
            supported_efforts=efforts,
        ),
        updates=request_updates,
        removable_fields=frozenset(request_updates),
        warning=warning,
    )


@dataclass(frozen=True)
class ReasoningAdapter:
    """Base contract implemented by each provider integration."""

    provider: ProviderType

    def decide(self, model_id: str, intent: ThinkingIntent) -> ReasoningDecision:
        if intent == "default":
            return reasoning_decision("unknown", "none")
        return self._generic_effort(intent)

    def _generic_effort(self, intent: ThinkingIntent) -> ReasoningDecision:
        value = "medium" if intent == "enable" else "none"
        return reasoning_decision(
            "unknown",
            "effort",
            {"reasoning_effort": value},
            efforts=("none", "low", "medium", "high"),
        )
