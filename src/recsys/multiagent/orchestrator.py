"""Bounded three-role orchestration: zero or one revision, then final gate."""

from pydantic import ValidationError

from recsys.baseline.contracts import SingleAgentAnswer
from recsys.multiagent.agents import (
    CriticVerificationAgent,
    PreferenceRetrievalAgent,
    RecommendationAgent,
)
from recsys.multiagent.contracts import AgentAttempt, CriticVerdict, MultiAgentResult
from recsys.schemas import UserPreferences


class MultiAgentOrchestrator:
    """Candidate retrieval -> draft -> critic -> [one revision -> critic].

    Rejected proposals never appear as final answers. This implementation
    does not make correctness or factual-entailment claims about model prose.
    """

    def __init__(
        self,
        retrieval_agent: PreferenceRetrievalAgent,
        recommendation_agent: RecommendationAgent,
        critic_agent: CriticVerificationAgent,
        *,
        max_revisions: int = 1,
    ):
        if type(max_revisions) is not int or max_revisions not in (0, 1):
            raise ValueError("max_revisions must be 0 or 1")
        self.retrieval_agent = retrieval_agent
        self.recommendation_agent = recommendation_agent
        self.critic_agent = critic_agent
        self.max_revisions = max_revisions

    def run(self, preferences: UserPreferences, *, top_k: int = 3) -> MultiAgentResult:
        retrieval = self.retrieval_agent.run(preferences, top_k=top_k)
        if not retrieval.candidates:
            return MultiAgentResult(
                status="no_candidates", retrieval=retrieval, answer=None,
                generator_calls=0, note=retrieval.no_match_reason,
            )

        attempts: list[AgentAttempt] = []
        feedback: CriticVerdict | None = None
        for revision in range(self.max_revisions + 1):
            try:
                raw = self.recommendation_agent.propose(retrieval, feedback=feedback)
                answer = SingleAgentAnswer.model_validate_json(raw)
                parse_valid = True
            except Exception:
                # A generator/transport failure also counts as one attempted
                # proposal; return no unverified result. Never retry beyond
                # the explicitly configured single revision.
                answer = None
                parse_valid = False

            verdict = self.critic_agent.review(
                answer, preferences, retrieval, invalid_generation=not parse_valid,
            )
            attempts.append(AgentAttempt(
                attempt_number=revision + 1,
                is_revision=bool(revision),
                parse_valid=parse_valid,
                verdict=verdict,
                proposed_product_ids=(
                    [rec.product_id for rec in answer.recommendations] if answer is not None else []
                ),
                generator_calls=1,
            ))
            if verdict.approved:
                return MultiAgentResult(
                    status="abstained" if answer.no_suitable_match else "approved",
                    retrieval=retrieval, answer=answer,
                    attempts=attempts, revision_used=bool(revision),
                    generator_calls=len(attempts),
                )
            feedback = verdict

        return MultiAgentResult(
            status="rejected", retrieval=retrieval, answer=None,
            attempts=attempts, revision_used=(len(attempts) == 2),
            generator_calls=len(attempts),
            note="No proposal passed the deterministic critic within the revision budget",
        )
