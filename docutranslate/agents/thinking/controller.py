from __future__ import annotations

import json
import time
from threading import Lock
from typing import Any

import httpx

from docutranslate.agents.provider import ProviderType

from docutranslate.agents.provider.registry import get_reasoning_adapter
from .capability import ReasoningDecision, ThinkingIntent


_rejected_capabilities: dict[tuple[str, str, str, str], float] = {}
_cache_lock = Lock()
_REJECTION_CACHE_TTL_SECONDS = 3600.0


def clear_reasoning_compatibility_cache() -> None:
    """Clear runtime negotiation state. Primarily useful for isolated tests."""
    with _cache_lock:
        _rejected_capabilities.clear()


class ReasoningController:
    """Translate public thinking intent into provider-specific request fields."""

    _ERROR_STATUSES = {400, 422}

    def __init__(
        self,
        *,
        provider: ProviderType,
        model_id: str,
        base_url: str,
        intent: ThinkingIntent,
        user_extra_body: dict[str, Any] | None = None,
    ):
        self.provider = provider
        self.model_id = model_id
        self.base_url = base_url.rstrip("/").lower()
        self.intent = intent
        self.user_extra_body = user_extra_body or {}
        self.adapter = get_reasoning_adapter(provider)
        self._warned: set[str] = set()
        self._last_decision: ReasoningDecision | None = None

    @property
    def cache_key(self) -> tuple[str, str, str, str]:
        return (self.base_url, self.provider, self.model_id.lower(), self.intent)

    def apply(self, data: dict[str, Any]) -> str | None:
        decision = self.adapter.decide(self.model_id, self.intent)
        self._last_decision = decision
        now = time.monotonic()
        with _cache_lock:
            rejected_at = _rejected_capabilities.get(self.cache_key)
            rejected = (
                rejected_at is not None
                and now - rejected_at < _REJECTION_CACHE_TTL_SECONDS
            )
            if rejected_at is not None and not rejected:
                _rejected_capabilities.pop(self.cache_key, None)
        if not rejected:
            data.update(decision.updates)
            return self._once(decision.warning)
        return self._once(
            f"模型 {self.model_id} 使用已缓存的思考参数兼容降级；"
            "将在缓存过期后重新探测。"
        )

    def _once(self, warning: str | None) -> str | None:
        if not warning or warning in self._warned:
            return None
        self._warned.add(warning)
        return warning

    def is_parameter_compatibility_error(
        self, error: httpx.HTTPStatusError, data: dict[str, Any]
    ) -> bool:
        if error.response.status_code not in self._ERROR_STATUSES:
            return False
        decision = self._last_decision or self.adapter.decide(
            self.model_id, self.intent
        )
        removable = {
            field
            for field in decision.removable_fields - self.user_extra_body.keys()
            if field in data
        }
        if not removable:
            return False
        body = error.response.text.lower()
        # A 400/422 response that identifies an adapter-added field is a safe
        # signal: the request was rejected before generation, so retrying once
        # without that field cannot duplicate a successful completion.
        return any(field.lower() in body for field in removable)

    def fallback_request(
        self, data: dict[str, Any], error: httpx.HTTPStatusError
    ) -> tuple[dict[str, Any], str] | None:
        if not self.is_parameter_compatibility_error(error, data):
            return None
        decision = self._last_decision or self.adapter.decide(
            self.model_id, self.intent
        )
        removable = decision.removable_fields - self.user_extra_body.keys()
        fallback = dict(data)
        for field in removable:
            fallback.pop(field, None)
        if fallback == data:
            return None
        with _cache_lock:
            _rejected_capabilities[self.cache_key] = time.monotonic()
        fields = ", ".join(sorted(removable))
        warning = (
            f"接口拒绝模型 {self.model_id} 的思考参数 ({fields})，"
            "本次移除后重试，并已缓存兼容结果。"
        )
        return fallback, warning


def parse_extra_body(value: str | None) -> dict[str, Any] | None:
    if not value or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else {}
