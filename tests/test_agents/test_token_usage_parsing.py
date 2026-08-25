import logging

import pytest

from docutranslate.agents.agent import (
    Agent,
    AgentConfig,
    _StreamingResponseAccumulator,
    extract_token_info,
)
from docutranslate.agents.provider.token_counting import (
    ProviderTokenUsageParser,
    TokenUsageParser,
    get_provider_token_usage_parser,
)


@pytest.mark.parametrize(
    ("provider", "response", "expected"),
    [
        (
            "default",
            {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 40,
                    "total_tokens": 140,
                    "prompt_tokens_details": {"cached_tokens": 60},
                    "completion_tokens_details": {"reasoning_tokens": 10},
                }
            },
            (100, 60, 40, 10, 140),
        ),
        (
            "deepseek",
            {
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 30,
                    "total_tokens": 150,
                    "prompt_cache_hit_tokens": 80,
                    "completion_tokens_details": {"reasoning_tokens": 12},
                }
            },
            (120, 80, 30, 12, 150),
        ),
        (
            "aliyuncs",
            {
                "usage": {
                    "input_tokens": 90,
                    "output_tokens": 20,
                    "cache_read_input_tokens": 50,
                    "reasoning_tokens": 8,
                    "total_tokens": 110,
                }
            },
            (90, 50, 20, 8, 110),
        ),
        (
            "google",
            {
                "usageMetadata": {
                    "promptTokenCount": 70,
                    "candidatesTokenCount": 25,
                    "cachedContentTokenCount": 30,
                    "thoughtsTokenCount": 15,
                    "totalTokenCount": 110,
                }
            },
            (70, 30, 25, 15, 110),
        ),
        (
            "aliyuncs",
            {
                "usage": {
                    "prompt_tokens": 3019,
                    "completion_tokens": 101,
                    "total_tokens": 3120,
                    "prompt_tokens_details": {"cached_tokens": 2048},
                    "completion_tokens_details": {"reasoning_tokens": 40},
                }
            },
            (3019, 2048, 101, 40, 3120),
        ),
        (
            "minimax",
            {
                "usage": {
                    "prompt_tokens": 26,
                    "completion_tokens": 223,
                    "total_tokens": 249,
                    "prompt_tokens_details": {"cached_tokens": 12},
                    "completion_tokens_details": {"reasoning_tokens": 214},
                }
            },
            (26, 12, 223, 214, 249),
        ),
        (
            "mimo",
            {
                "usage": {
                    "prompt_tokens": 57,
                    "completion_tokens": 72,
                    "total_tokens": 129,
                    "prompt_tokens_details": {"cached_tokens": 32},
                    "completion_tokens_details": {"reasoning_tokens": 20},
                }
            },
            (57, 32, 72, 20, 129),
        ),
        (
            "litellm",
            {
                "usage": {
                    "input_tokens": 44,
                    "output_tokens": 11,
                    "input_tokens_details": {"cached_tokens": 22},
                    "output_tokens_details": {"reasoning_tokens": 5},
                }
            },
            (44, 22, 11, 5, 55),
        ),
        (
            "bigmodel",
            {
                "usage": {
                    "prompt_tokens": 1200,
                    "completion_tokens": 300,
                    "total_tokens": 1500,
                    "prompt_tokens_details": {"cached_tokens": 800},
                    "completion_tokens_details": {"reasoning_tokens": 180},
                }
            },
            (1200, 800, 300, 180, 1500),
        ),
        (
            "litellm",
            {
                "usage": {
                    "prompt_tokens": 13,
                    "completion_tokens": 43,
                    "total_tokens": 56,
                    "prompt_tokens_details": {"cached_tokens": 8},
                    "cache_read_input_tokens": 8,
                }
            },
            (13, 8, 43, 0, 56),
        ),
        (
            "volces",
            {
                "usage": {
                    "prompt_tokens": 989,
                    "completion_tokens": 601,
                    "total_tokens": 1590,
                    "prompt_tokens_details": {"cached_tokens": 400},
                    "completion_tokens_details": {"reasoning_tokens": 250},
                }
            },
            (989, 400, 601, 250, 1590),
        ),
        (
            "siliconflow",
            {
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 1540,
                    "total_tokens": 1555,
                    "completion_tokens_details": {"reasoning_tokens": 1190},
                    "prompt_tokens_details": {"cached_tokens": 0},
                    "prompt_cache_hit_tokens": 9,
                    "prompt_cache_miss_tokens": 6,
                }
            },
            (15, 9, 1540, 1190, 1555),
        ),
        (
            "openrouter",
            {
                "usage": {
                    "prompt_tokens": 10339,
                    "completion_tokens": 60,
                    "total_tokens": 10399,
                    "prompt_tokens_details": {
                        "cached_tokens": 10318,
                        "cache_write_tokens": 0,
                    },
                    "completion_tokens_details": {"reasoning_tokens": 30},
                }
            },
            (10339, 10318, 60, 30, 10399),
        ),
        (
            "ollama",
            {
                "prompt_eval_count": 11,
                "eval_count": 18,
            },
            (11, 0, 18, 0, 29),
        ),
        (
            "ollama",
            {
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 18,
                    "total_tokens": 29,
                }
            },
            (11, 0, 18, 0, 29),
        ),
    ],
)
def test_provider_usage_parsers(provider, response, expected):
    parser = get_provider_token_usage_parser(provider)

    assert parser.parse(response) == expected


@pytest.mark.parametrize(
    ("provider", "parser_name"),
    [
        ("default", "openai"),
        ("deepseek", "deepseek"),
        ("aliyuncs", "dashscope"),
        ("google", "gemini"),
        ("minimax", "minimax"),
        ("mimo", "mimo"),
        ("bigmodel", "bigmodel"),
        ("volces", "volcengine"),
        ("siliconflow", "siliconflow"),
        ("openrouter", "openrouter"),
        ("litellm", "litellm"),
        ("ollama", "ollama"),
    ],
)
def test_every_provider_has_an_explicit_usage_parser(provider, parser_name):
    assert get_provider_token_usage_parser(provider).name == parser_name


def test_provider_parser_falls_back_to_default_openai_shape():
    response = {
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "total_tokens": 15,
        }
    }

    assert get_provider_token_usage_parser("google").parse(response) == (
        12,
        0,
        3,
        0,
        15,
    )


class _BrokenUsageParser(TokenUsageParser):
    def parse(self, response_data):
        raise ValueError("broken provider parser")


def test_parser_exception_falls_back_to_default():
    parser = ProviderTokenUsageParser(primary=_BrokenUsageParser())
    response = {
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 2,
            "total_tokens": 10,
        }
    }

    assert parser.parse(response) == (8, 0, 2, 0, 10)


def test_unknown_or_malformed_usage_returns_safe_zero_values():
    assert extract_token_info({}, "unknown-provider") == (0, 0, 0, 0, 0)
    assert extract_token_info({"usage": "invalid"}) == (0, 0, 0, 0, 0)


def test_legacy_extract_token_info_call_remains_compatible():
    assert extract_token_info(
        {"usage": {"prompt_tokens": 3, "completion_tokens": 2}}
    ) == (3, 0, 2, 0, 5)


def test_streaming_accumulator_preserves_gemini_usage_metadata():
    accumulator = _StreamingResponseAccumulator()
    accumulator.feed(
        b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}'
        b'\n\ndata: {"usageMetadata":{"promptTokenCount":7,'
        b'"candidatesTokenCount":2,"totalTokenCount":9}}\n\ndata: [DONE]\n\n'
    )

    result = accumulator.finish(b"")

    assert result["usageMetadata"]["totalTokenCount"] == 9
    assert get_provider_token_usage_parser("google").parse(result) == (
        7,
        0,
        2,
        0,
        9,
    )


def test_provider_does_not_change_existing_pre_request_estimate():
    configs = [
        AgentConfig(
            base_url="https://example.invalid/v1",
            api_key="test",
            model_id="model",
            provider=provider,
            logger=logging.getLogger("token-usage-test"),
        )
        for provider in ("google", "deepseek", "litellm", "default")
    ]

    assert {Agent(config)._estimate_tokens("中文 English") for config in configs} == {5}
