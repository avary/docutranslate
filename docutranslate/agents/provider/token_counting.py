from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .provider import ProviderType


TokenUsage = tuple[int, int, int, int, int]
ZERO_TOKEN_USAGE: TokenUsage = (0, 0, 0, 0, 0)


def _as_non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def _first_int(mapping: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        if key in mapping:
            return _as_non_negative_int(mapping[key])
    return 0


def _nested_int(mapping: Mapping[str, Any], containers: tuple[str, ...], key: str) -> int:
    for container in containers:
        details = mapping.get(container)
        if isinstance(details, Mapping) and key in details:
            return _as_non_negative_int(details[key])
    return 0


class TokenUsageParser:
    """Parse input, cached, output, reasoning, and total token usage."""

    name = "base"

    def parse(self, response_data: Mapping[str, Any]) -> TokenUsage:
        raise NotImplementedError


class OpenAIUsageParser(TokenUsageParser):
    name = "openai"

    def parse(self, response_data: Mapping[str, Any]) -> TokenUsage:
        usage = response_data.get("usage")
        if not isinstance(usage, Mapping):
            return ZERO_TOKEN_USAGE

        input_tokens = _first_int(usage, "prompt_tokens", "input_tokens")
        output_tokens = _first_int(usage, "completion_tokens", "output_tokens")
        total_tokens = _first_int(usage, "total_tokens")
        cached_tokens = _nested_int(
            usage,
            ("prompt_tokens_details", "input_tokens_details"),
            "cached_tokens",
        )
        if not cached_tokens:
            cached_tokens = _first_int(
                usage,
                "cache_read_input_tokens",
                "cached_tokens",
            )
        reasoning_tokens = _nested_int(
            usage,
            ("completion_tokens_details", "output_tokens_details"),
            "reasoning_tokens",
        )
        if not total_tokens and input_tokens + output_tokens > 0:
            total_tokens = input_tokens + output_tokens
        return input_tokens, cached_tokens, output_tokens, reasoning_tokens, total_tokens


class DeepSeekUsageParser(OpenAIUsageParser):
    name = "deepseek"

    def parse(self, response_data: Mapping[str, Any]) -> TokenUsage:
        parsed = super().parse(response_data)
        usage = response_data.get("usage")
        if not isinstance(usage, Mapping):
            return parsed
        cached_tokens = _first_int(usage, "prompt_cache_hit_tokens") or parsed[1]
        return parsed[0], cached_tokens, parsed[2], parsed[3], parsed[4]


class DashScopeUsageParser(TokenUsageParser):
    name = "dashscope"

    def parse(self, response_data: Mapping[str, Any]) -> TokenUsage:
        usage = response_data.get("usage")
        if not isinstance(usage, Mapping):
            return ZERO_TOKEN_USAGE
        input_tokens = _first_int(usage, "input_tokens")
        output_tokens = _first_int(usage, "output_tokens")
        total_tokens = _first_int(usage, "total_tokens")
        cached_tokens = _first_int(
            usage,
            "cache_read_input_tokens",
            "cached_tokens",
        ) or _nested_int(
            usage,
            ("input_tokens_details", "prompt_tokens_details"),
            "cached_tokens",
        )
        reasoning_tokens = _first_int(usage, "reasoning_tokens") or _nested_int(
            usage,
            ("output_tokens_details", "completion_tokens_details"),
            "reasoning_tokens",
        )
        if not total_tokens and input_tokens + output_tokens > 0:
            total_tokens = input_tokens + output_tokens
        return input_tokens, cached_tokens, output_tokens, reasoning_tokens, total_tokens


class GeminiUsageParser(TokenUsageParser):
    name = "gemini"

    def parse(self, response_data: Mapping[str, Any]) -> TokenUsage:
        usage = response_data.get("usageMetadata")
        if not isinstance(usage, Mapping):
            usage = response_data.get("usage_metadata")
        if not isinstance(usage, Mapping):
            usage = response_data.get("usage")
        if not isinstance(usage, Mapping):
            return ZERO_TOKEN_USAGE

        input_tokens = _first_int(usage, "promptTokenCount", "total_input_tokens")
        output_tokens = _first_int(
            usage,
            "candidatesTokenCount",
            "total_output_tokens",
        )
        cached_tokens = _first_int(
            usage,
            "cachedContentTokenCount",
            "total_cached_tokens",
        )
        reasoning_tokens = _first_int(
            usage,
            "thoughtsTokenCount",
            "total_thought_tokens",
        )
        total_tokens = _first_int(usage, "totalTokenCount", "total_tokens")
        if (
            not reasoning_tokens
            and total_tokens > input_tokens + output_tokens
            and input_tokens + output_tokens > 0
        ):
            reasoning_tokens = total_tokens - input_tokens - output_tokens
        if not total_tokens and input_tokens + output_tokens > 0:
            total_tokens = input_tokens + output_tokens
        return input_tokens, cached_tokens, output_tokens, reasoning_tokens, total_tokens


DEFAULT_USAGE_PARSER = OpenAIUsageParser()
DEEPSEEK_USAGE_PARSER = DeepSeekUsageParser()
DASHSCOPE_USAGE_PARSER = DashScopeUsageParser()
GEMINI_USAGE_PARSER = GeminiUsageParser()


_PROVIDER_USAGE_PARSERS: dict[ProviderType, TokenUsageParser] = {
    "deepseek": DEEPSEEK_USAGE_PARSER,
    "aliyuncs": DASHSCOPE_USAGE_PARSER,
    "google": GEMINI_USAGE_PARSER,
    # These providers currently expose the OpenAI usage schema. Keeping them
    # explicit documents the protocol choice and lets each diverge later.
    "minimax": DEFAULT_USAGE_PARSER,
    "mimo": DEFAULT_USAGE_PARSER,
    "bigmodel": DEFAULT_USAGE_PARSER,
    "volces": DEFAULT_USAGE_PARSER,
    "siliconflow": DEFAULT_USAGE_PARSER,
    "openrouter": DEFAULT_USAGE_PARSER,
    "litellm": DEFAULT_USAGE_PARSER,
    "ollama": DEFAULT_USAGE_PARSER,
    "default": DEFAULT_USAGE_PARSER,
}


@dataclass(frozen=True)
class ProviderTokenUsageParser:
    """Use the provider parser first and default parser as compatibility fallback."""

    primary: TokenUsageParser
    fallback: TokenUsageParser = DEFAULT_USAGE_PARSER

    @property
    def name(self) -> str:
        return self.primary.name

    def parse(self, response_data: Mapping[str, Any]) -> TokenUsage:
        try:
            parsed = self.primary.parse(response_data)
        except Exception:
            parsed = ZERO_TOKEN_USAGE
        if self.primary is self.fallback:
            return parsed
        try:
            fallback_parsed = self.fallback.parse(response_data)
        except Exception:
            fallback_parsed = ZERO_TOKEN_USAGE

        # A provider parser can partially recognize a compatibility payload
        # (for example, only ``total_tokens``). Prefer whichever interpretation
        # recovered more fields instead of treating any non-zero value as a
        # successful provider-specific parse.
        if sum(value > 0 for value in fallback_parsed) > sum(
            value > 0 for value in parsed
        ):
            return fallback_parsed
        return parsed


def get_provider_token_usage_parser(
    provider: ProviderType | str,
) -> ProviderTokenUsageParser:
    primary = _PROVIDER_USAGE_PARSERS.get(provider, DEFAULT_USAGE_PARSER)
    return ProviderTokenUsageParser(primary=primary)
