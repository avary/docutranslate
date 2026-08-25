from docutranslate.agents.thinking.base import ReasoningAdapter

from .bigmodel import BigModelAdapter
from .dashscope import DashScopeAdapter
from .deepseek import DeepSeekAdapter
from .gemini import GoogleAdapter
from .litellm import LiteLLMAdapter
from .mimo import MiMoAdapter
from .minimax import MiniMaxAdapter
from .ollama import OllamaAdapter
from .openai import OpenAIAdapter
from .openrouter import OpenRouterAdapter
from .provider import ProviderType
from .siliconflow import SiliconFlowAdapter
from .volcengine import VolcEngineAdapter


_ADAPTERS: dict[ProviderType, ReasoningAdapter] = {
    "minimax": MiniMaxAdapter("minimax"),
    "ollama": OllamaAdapter("ollama"),
    "bigmodel": BigModelAdapter("bigmodel"),
    "aliyuncs": DashScopeAdapter("aliyuncs"),
    "volces": VolcEngineAdapter("volces"),
    "google": GoogleAdapter("google"),
    "siliconflow": SiliconFlowAdapter("siliconflow"),
    "deepseek": DeepSeekAdapter("deepseek"),
    "openrouter": OpenRouterAdapter("openrouter"),
    "mimo": MiMoAdapter("mimo"),
    "litellm": LiteLLMAdapter("litellm"),
    "default": OpenAIAdapter("default"),
}


def get_reasoning_adapter(provider: ProviderType) -> ReasoningAdapter:
    """Return the platform implementation for the normalized provider ID."""
    return _ADAPTERS.get(provider, _ADAPTERS["default"])
