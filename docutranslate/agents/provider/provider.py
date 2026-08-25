from typing import Literal, TypeAlias

ProviderType: TypeAlias = Literal[
    "minimax",
    "ollama",
    "bigmodel",
    "aliyuncs",
    "volces",
    "google",
    "siliconflow",
    "deepseek",
    "openrouter",
    "mimo",
    "litellm",
    "default",
]


def get_provider_by_domain(domain: str) -> ProviderType:
    """Resolve well-known OpenAI-compatible endpoints without inspecting model IDs."""
    domain = domain.strip().lower()
    hostname = domain.split(":", 1)[0]

    if hostname == "open.bigmodel.cn":
        return "bigmodel"
    elif hostname == "dashscope.aliyuncs.com" or hostname.startswith("dashscope-"):
        return "aliyuncs"
    elif hostname.endswith(".maas.aliyuncs.com"):
        return "aliyuncs"
    elif hostname.startswith("ark.") and hostname.endswith(".volces.com"):
        return "volces"
    elif hostname == "generativelanguage.googleapis.com":
        return "google"
    elif hostname in {"api.siliconflow.cn", "api.siliconflow.com"}:
        return "siliconflow"
    elif hostname == "api.deepseek.com":
        return "deepseek"
    elif hostname == "api.minimaxi.com":
        return "minimax"
    elif hostname == "openrouter.ai":
        return "openrouter"
    elif hostname == "api.xiaomimimo.com":
        return "mimo"
    elif hostname.startswith("token-plan-") and hostname.endswith(".xiaomimimo.com"):
        return "mimo"
    elif hostname == "litellm" or hostname.startswith("litellm."):
        return "litellm"
    elif domain.endswith(":11434"):
        return "ollama"
    return "default"
