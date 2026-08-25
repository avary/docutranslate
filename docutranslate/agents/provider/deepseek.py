from docutranslate.agents.thinking.base import ReasoningAdapter, reasoning_decision
from docutranslate.agents.thinking.capability import ReasoningDecision, ThinkingIntent


class DeepSeekAdapter(ReasoningAdapter):
    def decide(self, model_id: str, intent: ThinkingIntent) -> ReasoningDecision:
        if intent == "default":
            return reasoning_decision("optional", "none")
        value = "enabled" if intent == "enable" else "disabled"
        return reasoning_decision(
            "optional", "thinking_type", {"thinking": {"type": value}}
        )
