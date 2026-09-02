from docutranslate.agents.thinking.base import reasoning_decision
from docutranslate.agents.thinking.capability import ReasoningDecision, ThinkingIntent

from .deepseek import DeepSeekAdapter


class BigModelAdapter(DeepSeekAdapter):
    """Zhipu BigModel GLM: thinking.type wire format with mandatory models.

    GLM-5.3/5.3-Flash 与 GLM-4.7 为强制思考模型，API 拒绝 thinking.type=disabled，
    因此对禁用请求返回 mandatory 并保持模型默认行为（与官方文档一致）。
    """

    _MANDATORY_MARKERS = ("glm-5.3", "glm-4.7")

    def decide(self, model_id: str, intent: ThinkingIntent) -> ReasoningDecision:
        model = model_id.lower()
        mandatory = any(marker in model for marker in self._MANDATORY_MARKERS)
        if intent == "default":
            return reasoning_decision(
                "mandatory" if mandatory else "optional", "none"
            )
        if mandatory and intent == "disable":
            return reasoning_decision(
                "mandatory",
                "thinking_type",
                warning=f"模型 {model_id} 为强制思考模型，已忽略关闭请求。",
            )
        return super().decide(model_id, intent)