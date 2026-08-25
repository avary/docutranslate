from docutranslate.agents.thinking.base import ReasoningAdapter, reasoning_decision
from docutranslate.agents.thinking.capability import ReasoningDecision, ThinkingIntent


class OpenRouterAdapter(ReasoningAdapter):
    _MANDATORY_MARKERS = ("/deepseek-r1", "-thinking", "/gemini-3")

    def decide(self, model_id: str, intent: ThinkingIntent) -> ReasoningDecision:
        model = model_id.lower()
        mandatory = any(marker in model for marker in self._MANDATORY_MARKERS)
        if intent == "default":
            return reasoning_decision("mandatory" if mandatory else "unknown", "none")
        if mandatory and intent == "disable":
            return reasoning_decision(
                "mandatory",
                "effort",
                warning=f"OpenRouter 模型 {model_id} 标记为强制思考，已使用模型默认行为。",
            )
        reasoning = (
            {"enabled": True, "exclude": True}
            if intent == "enable"
            else {"effort": "none", "exclude": True}
        )
        return reasoning_decision(
            "mandatory" if mandatory else "unknown",
            "effort",
            {"reasoning": reasoning},
        )
