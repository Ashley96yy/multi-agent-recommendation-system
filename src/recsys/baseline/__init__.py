"""Single-agent retrieval-augmented recommendation baseline."""

from recsys.baseline.pipeline import SingleAgentRAG, build_context
from recsys.baseline.contracts import Recommendation, SingleAgentAnswer

__all__ = ["SingleAgentRAG", "build_context", "Recommendation", "SingleAgentAnswer"]
