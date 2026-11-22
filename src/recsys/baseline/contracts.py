"""Contracts for one-pass recommendation; keep generation separate from verification."""

from pydantic import Field, model_validator

from recsys.schemas.contracts import Contract


class Recommendation(Contract):
    product_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_evidence(self) -> "Recommendation":
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("duplicate evidence IDs")
        if not self.product_id.strip() or not self.reason.strip():
            raise ValueError("blank product ID or reason")
        return self


class SingleAgentAnswer(Contract):
    recommendations: list[Recommendation] = Field(default_factory=list)
    no_suitable_match: bool
    note: str | None = None

    @model_validator(mode="after")
    def exclusive_outcome(self) -> "SingleAgentAnswer":
        if bool(self.recommendations) == self.no_suitable_match:
            raise ValueError("recommendations and no_suitable_match are mutually exclusive")
        if len(set(r.product_id for r in self.recommendations)) != len(self.recommendations):
            raise ValueError("duplicate recommended product IDs")
        return self
