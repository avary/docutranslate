from docutranslate.agents.thinking.base import ReasoningAdapter, reasoning_decision
from docutranslate.agents.thinking.capability import ReasoningDecision, ThinkingIntent


class GoogleAdapter(ReasoningAdapter):
    def decide(self, model_id: str, intent: ThinkingIntent) -> ReasoningDecision:
        model = model_id.lower()
        mandatory = "gemini-3" in model or (
            "gemini-2.5-pro" in model and "flash" not in model
        )
        unsupported = any(
            marker in model
            for marker in ("gemini-1", "gemini-2.0", "embedding", "imagen")
        )
        if intent == "default":
            support = "unsupported" if unsupported else (
                "mandatory" if mandatory else "optional"
            )
            return reasoning_decision(support, "none")
        if unsupported:
            return reasoning_decision(
                "unsupported",
                "none",
                warning=f"模型 {model_id} 不支持可配置思考模式，已忽略 {intent}。",
            )
        if mandatory and intent == "disable":
            return reasoning_decision(
                "mandatory",
                "effort",
                warning=f"模型 {model_id} 不允许关闭思考，已使用模型默认行为。",
            )
        value = "medium" if intent == "enable" else "none"
        return reasoning_decision(
            "mandatory" if mandatory else "optional",
            "effort",
            {"reasoning_effort": value},
            efforts=("none", "minimal", "low", "medium", "high"),
        )
