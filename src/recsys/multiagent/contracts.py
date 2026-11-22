"""Strict Stage 3.0 contracts for bounded multi-agent recommendation workflows.

Structural checking and simple policy heuristics are NOT semantic entailment.
"""

from typing import Literal

from pydantic import Field, model_validator

from recsys.baseline.contracts import SingleAgentAnswer
from recsys.schemas.contracts import Contract
from recsys.schemas.retrieval import RetrievalResult


IssueCode = Literal[
    "invalid_generation",
    "product_not_retrieved",
    "constraint_violation",
    "invalid_citation",
    "required_attribute_uncited",
    "prohibited_claim_phrase",
]


class CriticIssue(Contract):
    code: IssueCode
    detail: str = Field(min_length=1)
    product_id: str | None = None


class CriticVerdict(Contract):
    approved: bool
    issues: list[CriticIssue] = Field(default_factory=list)
    review_level: Literal["structural_and_policy_heuristics"] = "structural_and_policy_heuristics"
    semantic_entailment_verified: Literal[False] = False

    @model_validator(mode="after")
    def verify_consistency(self) -> "CriticVerdict":
        if self.approved != (len(self.issues) == 0):
            raise ValueError("approved must agree with the presence of issues")
        return self


class AgentAttempt(Contract):
    attempt_number: int = Field(ge=1, le=2)
    is_revision: bool
    parse_valid: bool
    verdict: CriticVerdict
    proposed_product_ids: list[str] = Field(default_factory=list)
    generator_calls: int = Field(ge=1, le=1)

    @model_validator(mode="after")
    def validate_attempt_index(self) -> "AgentAttempt":
        if self.is_revision != (self.attempt_number == 2):
            raise ValueError("is_revision must match attempt_number")
        return self


class MultiAgentResult(Contract):
    status: Literal["approved", "abstained", "rejected", "no_candidates"]
    retrieval: RetrievalResult
    answer: SingleAgentAnswer | None
    attempts: list[AgentAttempt] = Field(default_factory=list)
    revision_used: bool = False
    generator_calls: int = Field(ge=0, le=2)
    note: str | None = None

    @model_validator(mode="after")
    def check_execution(self) -> "MultiAgentResult":
        if self.generator_calls != len(self.attempts):
            raise ValueError("generator_calls must match attempt count")
        if self.revision_used != (len(self.attempts) == 2):
            raise ValueError("revision_used must match two attempts")
        if [attempt.attempt_number for attempt in self.attempts] != list(range(1, len(self.attempts) + 1)):
            raise ValueError("attempts must be sequential")
        if self.status in {"approved", "abstained"}:
            if self.answer is None or not self.attempts or not self.attempts[-1].verdict.approved:
                raise ValueError("accepted result needs an approved final verdict and answer")
            if (self.status == "abstained") != self.answer.no_suitable_match:
                raise ValueError("status and model abstention must match")
        elif self.answer is not None:
            raise ValueError("rejected/no-candidates results cannot expose unapproved proposals")
        if self.status == "no_candidates" and (self.retrieval.candidates or self.attempts):
            raise ValueError("no-candidates path must bypass generation")
        if self.status == "rejected" and not self.attempts:
            raise ValueError("rejected path requires at least one attempted generation")
        return self
