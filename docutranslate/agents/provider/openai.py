import re

from docutranslate.agents.thinking.base import ReasoningAdapter, reasoning_decision
from docutranslate.agents.thinking.capability import ReasoningDecision, ThinkingIntent


class OpenAIAdapter(ReasoningAdapter):
    _NON_REASONING_PREFIXES = ("gpt-3", "gpt-4", "chatgpt-4", "text-")
    _MANDATORY_PREFIXES = ("o1", "o3", "o4")
    _EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")

    @staticmethod
    def _gpt5_minor(model: str) -> int | None:
        """Return the gpt-5.x minor version, or None when the ID is not gpt-5.x."""
        match = re.search(r"gpt-5\.(\d+)", model)
        return int(match.group(1)) if match else None

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
        # gpt-5-pro 仅支持 high 推理强度，思考无法关闭。
        if model.startswith("gpt-5-pro"):
            if intent == "disable":
                return reasoning_decision(
                    "mandatory",
                    "effort",
                    warning=f"模型 {model_id} 仅支持 high 推理强度且无法关闭思考，"
                    "已使用模型默认行为。",
                )
            return reasoning_decision(
                "optional",
                "effort",
                {"reasoning_effort": "high"},
                efforts=self._EFFORTS,
            )
        support = "optional" if model.startswith("gpt-5") else "unknown"
        # OpenAI 官方：gpt-5.1 及之后支持 "none"（关闭思考）；gpt-5.1 之前
        # （含 gpt-5 本体）不支持 "none"，最低只能到 "minimal"。
        minor = self._gpt5_minor(model)
        if intent == "enable":
            value = "medium"
        elif minor is not None and minor >= 1:
            value = "none"
        elif model.startswith("gpt-5"):
            value = "minimal"
        else:
            value = "none"
        return reasoning_decision(
            support,
            "effort",
            {"reasoning_effort": value},
            efforts=self._EFFORTS,
        )