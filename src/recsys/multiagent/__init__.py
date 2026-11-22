"""Stage 3.0 bounded multi-agent orchestration public API."""

from recsys.multiagent.agents import (
    CriticVerificationAgent,
    PreferenceRetrievalAgent,
    RecommendationAgent,
)
from recsys.multiagent.contracts import AgentAttempt, CriticIssue, CriticVerdict, MultiAgentResult
from recsys.multiagent.orchestrator import MultiAgentOrchestrator
from recsys.multiagent.source_guard import TypedSourceGuard

__all__ = [
    "PreferenceRetrievalAgent", "RecommendationAgent", "CriticVerificationAgent",
    "MultiAgentOrchestrator", "TypedSourceGuard", "CriticIssue", "CriticVerdict", "AgentAttempt", "MultiAgentResult",
]
