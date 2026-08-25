from docutranslate.agents.thinking.base import ReasoningAdapter, reasoning_decision
from docutranslate.agents.thinking.capability import ReasoningDecision, ThinkingIntent


class LiteLLMAdapter(ReasoningAdapter):
    """Use LiteLLM's provider-neutral OpenAI-compatible reasoning parameter.

    A LiteLLM model name may be an arbitrary proxy alias, so the underlying
    provider protocol must not be inferred from the model ID. Unsupported
    deployments are handled by the controller's one-shot compatibility retry.
    """

    def decide(self, model_id: str, intent: ThinkingIntent) -> ReasoningDecision:
        if intent == "default":
            return reasoning_decision("unknown", "none")
        return reasoning_decision(
            "unknown",
            "effort",
            {"reasoning_effort": "medium" if intent == "enable" else "none"},
            efforts=("none", "low", "medium", "high"),
        )
