"""Bounded three-role orchestration: zero or one revision, then final gate."""

from recsys.baseline.contracts import SingleAgentAnswer
from recsys.multiagent.agents import (
    CriticVerificationAgent,
    PreferenceRetrievalAgent,
    RecommendationAgent,
)
from recsys.multiagent.contracts import AgentAttempt, CriticVerdict, MultiAgentResult
from recsys.multiagent.output_policy import FormatPolicy, normalize_json_response
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
        format_policy: FormatPolicy = "strict",
        evidence_preflight: bool = False,
    ):
        if type(max_revisions) is not int or max_revisions not in (0, 1):
            raise ValueError("max_revisions must be 0 or 1")
        self.retrieval_agent = retrieval_agent
        self.recommendation_agent = recommendation_agent
        self.critic_agent = critic_agent
        if format_policy not in ("strict", "safe_json_normalization"):
            raise ValueError("unknown format policy")
        if type(evidence_preflight) is not bool:
            raise ValueError("evidence_preflight must be bool")
        self.max_revisions = max_revisions
        self.format_policy = format_policy
        self.evidence_preflight = evidence_preflight

    def run(self, preferences: UserPreferences, *, top_k: int = 3) -> MultiAgentResult:
        retrieval = self.retrieval_agent.run(preferences, top_k=top_k)
        if not retrieval.candidates:
            return MultiAgentResult(
                status="no_candidates", retrieval=retrieval, answer=None,
                generator_calls=0, note=retrieval.no_match_reason,
            )

        guard = self.critic_agent.source_guard
        if guard is not None:
            guard.bind(retrieval.catalog_sha256)
        if self.evidence_preflight and preferences.required_attributes:
            needed = {f"attributes.{name}" for name in preferences.required_attributes}
            supported = [
                c for c in retrieval.candidates
                if needed <= {e.supported_field for e in c.evidence}
                and (guard is None or guard.eligible_for_required_fields(
                    c.product.product_id, preferences.required_attributes
                ))
            ]
            if not supported:
                return MultiAgentResult(
                    status="insufficient_evidence", retrieval=retrieval, answer=None,
                    generator_calls=0,
                    note="No retrieved product has citations for all required attribute fields",
                )
            if len(supported) < len(retrieval.candidates):
                retrieval = retrieval.model_copy(update={"candidates": supported})

        attempts: list[AgentAttempt] = []
        feedback: CriticVerdict | None = None
        for revision in range(self.max_revisions + 1):
            format_actions: list[str] = []
            try:
                raw = self.recommendation_agent.propose(retrieval, feedback=feedback)
                normalized, format_actions = normalize_json_response(raw, self.format_policy)
                answer = SingleAgentAnswer.model_validate_json(normalized)
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
                format_actions=format_actions,
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
