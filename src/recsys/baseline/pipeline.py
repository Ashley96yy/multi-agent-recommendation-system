"""One-pass RAG baseline: retrieval -> bounded context -> single generator -> structural checks.

Structural checks are deliberately NOT a separate Critic agent and do not prove
the natural-language explanation is grounded. This baseline is designed for
controlled later comparison against verified multi-agent workflows.
"""

import json
from collections.abc import Callable
from typing import Protocol

from pydantic import ValidationError

from recsys.baseline.contracts import SingleAgentAnswer
from recsys.retrieval.retriever import CatalogRetriever
from recsys.schemas import RetrievalResult, UserPreferences


class Generator(Protocol):
    def generate(self, instructions: str, context: str) -> str: ...


SYSTEM_INSTRUCTIONS = """You are a product recommendation assistant.
Use ONLY the retrieved products and evidence snippets in the supplied JSON context.
Never invent product IDs, prices, attributes, citations, or scientific effects.
Honor all exclusion, budget, and category constraints.
Return EXACTLY one JSON object with schema:
{"recommendations":[{"product_id":"...", "reason":"...", "evidence_ids":["..."]}],
 "no_suitable_match":false, "note":null}
If no supported recommendation exists, use recommendations=[], no_suitable_match=true.
Refer to actual provided evidence IDs only. Do not assume missing properties are true.
Do not output Markdown or any text surrounding JSON."""


def build_context(result: RetrievalResult, *, max_excerpt_chars: int = 250) -> str:
    """Build tightly scoped source payload with only visible retrieved evidence."""
    if max_excerpt_chars < 1:
        raise ValueError("max_excerpt_chars must be positive")
    data = {
        "query": result.query,
        "catalog_sha256": result.catalog_sha256,
        "no_match_reason": result.no_match_reason,
        "products": [
            {
                "product": c.product.model_dump(mode="json"),
                "evidence": [
                    {
                        "evidence_id": e.evidence_id,
                        "source_id": e.source_id,
                        "source_type": e.source_type,
                        "supported_field": e.supported_field,
                        "excerpt": e.excerpt[:max_excerpt_chars],
                    }
                    for e in c.evidence
                ],
            }
            for c in result.candidates
        ],
    }
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


class SingleAgentRAG:
    def __init__(self, retriever: CatalogRetriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    def recommend(self, preferences: UserPreferences, *, top_k: int = 5) -> tuple[SingleAgentAnswer, RetrievalResult]:
        """One generator invocation maximum; fail closed on invalid model output."""
        retrieval = self.retriever.retrieve(preferences, top_k=top_k)
        if not retrieval.candidates:
            return SingleAgentAnswer(recommendations=[], no_suitable_match=True,
                                     note=retrieval.no_match_reason), retrieval
        raw = self.generator.generate(SYSTEM_INSTRUCTIONS, build_context(retrieval))
        try:
            answer = SingleAgentAnswer.model_validate_json(raw)
        except ValidationError as exc:
            raise ValueError("generator returned invalid response schema") from exc
        eligible = {c.product.product_id: c for c in retrieval.candidates}
        for recommendation in answer.recommendations:
            candidate = eligible.get(recommendation.product_id)
            if candidate is None:
                raise ValueError("generator recommended a product outside retrieved candidates")
            if not set(recommendation.evidence_ids) <= {e.evidence_id for e in candidate.evidence}:
                raise ValueError("generator cited absent or cross-product evidence IDs")
        return answer, retrieval
