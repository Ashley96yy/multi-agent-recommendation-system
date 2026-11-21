"""Strict output contracts for retrieval (before any recommendation or verification)."""

from typing import Literal

from pydantic import Field, model_validator

from recsys.schemas.contracts import Contract, EvidenceReference, Product


class RetrievedProduct(Contract):
    product: Product
    similarity: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    evidence: list[EvidenceReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_evidence_links(self) -> "RetrievedProduct":
        if any(reference.product_id != self.product.product_id for reference in self.evidence):
            raise ValueError("retrieved evidence must belong to the candidate product")
        return self


class RetrievalResult(Contract):
    query: str = Field(min_length=1)
    embedder_name: str = Field(min_length=1)
    catalog_sha256: str = Field(min_length=64, max_length=64)
    indexed_products: int = Field(ge=1)
    eligible_products: int = Field(ge=0)
    filtered_by_reason: dict[str, int] = Field(default_factory=dict)
    candidates: list[RetrievedProduct] = Field(default_factory=list)
    no_match_reason: Literal["no_eligible_products", "no_vector_match"] | None = None

    @model_validator(mode="after")
    def check_status(self) -> "RetrievalResult":
        if bool(self.candidates) == bool(self.no_match_reason):
            raise ValueError("exactly one of candidates or no_match_reason is required")
        if len(self.candidates) > self.eligible_products:
            raise ValueError("more results than eligible products")
        if self.eligible_products + sum(self.filtered_by_reason.values()) != self.indexed_products:
            raise ValueError("filtered counts do not reconcile to indexed products")
        return self
