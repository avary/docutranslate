"""Backward-compatible facade for the former tuple-based thinking mapping.

New request construction uses :class:`ReasoningController`. This module remains
available for integrations that imported ``get_thinking_mode`` directly.
"""

from typing import Any, TypeAlias

from docutranslate.agents.provider import ProviderType

from docutranslate.agents.provider.registry import get_reasoning_adapter

ThinkingField: TypeAlias = str
ThinkingValue: TypeAlias = str | dict[str, Any] | bool
ThinkingConfig: TypeAlias = tuple[ThinkingField, ThinkingValue, ThinkingValue] | None


def get_thinking_mode_by_model_id(model_id: str) -> ThinkingConfig:
    # A model name does not identify the gateway's wire protocol. Do not guess a
    # vendor-specific field from names such as qwen/glm/gemini.
    return get_thinking_mode("default", model_id)


def get_thinking_mode(provider: ProviderType, model_id: str) -> ThinkingConfig:
    adapter = get_reasoning_adapter(provider)
    enabled = adapter.decide(model_id, "enable").updates
    disabled = adapter.decide(model_id, "disable").updates
    for field in enabled.keys() & disabled.keys():
        enable_value = enabled[field]
        disable_value = disabled[field]
        if isinstance(enable_value, (str, dict, bool)) and isinstance(
            disable_value, (str, dict, bool)
        ):
            return field, enable_value, disable_value
    return None
