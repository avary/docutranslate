import re

from docutranslate.agents.thinking.base import ReasoningAdapter, reasoning_decision
from docutranslate.agents.thinking.capability import ReasoningDecision, ThinkingIntent


class MiniMaxAdapter(ReasoningAdapter):
    def decide(self, model_id: str, intent: ThinkingIntent) -> ReasoningDecision:
        model = model_id.lower()
        is_m3 = bool(re.search(r"(?:^|[/_-])minimax[-_]?m3(?:$|[/_.-])", model))
        is_m2 = "minimax-m2" in model or "minimax_m2" in model
        auxiliary = {"reasoning_split": True} if intent != "disable" else {}

        if intent == "default":
            return reasoning_decision(
                "optional" if is_m3 else ("mandatory" if is_m2 else "unknown"),
                "none",
                auxiliary,
            )
        if is_m2 and intent == "disable":
            return reasoning_decision(
                "mandatory",
                "adaptive",
                warning=f"模型 {model_id} 的思考不能关闭，已使用模型默认行为。",
            )

        thinking_type = "adaptive" if intent == "enable" else "disabled"
        updates = {"thinking": {"type": thinking_type}, **auxiliary}
        return reasoning_decision(
            "optional" if is_m3 else "unknown",
            "adaptive",
            updates,
        )
