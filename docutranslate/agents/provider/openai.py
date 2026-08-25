from docutranslate.agents.thinking.base import ReasoningAdapter, reasoning_decision
from docutranslate.agents.thinking.capability import ReasoningDecision, ThinkingIntent


class OpenAIAdapter(ReasoningAdapter):
    _NON_REASONING_PREFIXES = ("gpt-3", "gpt-4", "chatgpt-4", "text-")
    _MANDATORY_PREFIXES = ("o1", "o3", "o4")

    def decide(self, model_id: str, intent: ThinkingIntent) -> ReasoningDecision:
        model = model_id.lower()
        if intent == "default":
            return reasoning_decision("unknown", "none")
        if model.startswith(self._NON_REASONING_PREFIXES):
            return reasoning_decision(
                "unsupported",
                "none",
                warning=f"模型 {model_id} 不支持可配置思考模式，已忽略 {intent}。",
            )
        if model.startswith(self._MANDATORY_PREFIXES) and intent == "disable":
            return reasoning_decision(
                "mandatory",
                "effort",
                warning=f"模型 {model_id} 的思考不能可靠关闭，已使用模型默认行为。",
            )
        support = "optional" if model.startswith("gpt-5") else "unknown"
        value = "medium" if intent == "enable" else "none"
        return reasoning_decision(
            support,
            "effort",
            {"reasoning_effort": value},
            efforts=("none", "low", "medium", "high", "xhigh", "max"),
        )
