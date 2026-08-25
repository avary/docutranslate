"""Unified reasoning/thinking capability layer.

``thinking`` remains the package and public configuration name for backward
compatibility; provider-specific wire formats live in ``agents.provider``.
"""

from .capability import ReasoningCapability, ReasoningDecision, ThinkingIntent
from .controller import ReasoningController

__all__ = [
    "ReasoningCapability",
    "ReasoningController",
    "ReasoningDecision",
    "ThinkingIntent",
]
