"""Stage 1.2: strict, evidence-aware data contracts. No LLM logic lives here."""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Product(Contract):
    product_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    brand: str = Field(min_length=1)
    category: str = Field(min_length=1)
    price_usd: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    attributes: dict[str, bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_identifiers(self) -> "Product":
        if not self.product_id.strip() or not self.title.strip():
            raise ValueError("product_id and title must not be whitespace")
        if not self.brand.strip() or not self.category.strip():
            raise ValueError("brand and category must not be whitespace")
        if any(not key.strip() for key in self.attributes):
            raise ValueError("attribute keys must not be empty")
        return self


class UserPreferences(Contract):
    request: str = Field(min_length=1)
    max_budget_usd: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    required_attributes: set[str] = Field(default_factory=set)
    required_category: str | None = None
    excluded_product_ids: set[str] = Field(default_factory=set)
    excluded_brands: set[str] = Field(default_factory=set)

    @model_validator(mode="after")
    def check_preferences(self) -> "UserPreferences":
        if not self.request.strip():
            raise ValueError("request must not be whitespace")
        if self.required_category is not None and not self.required_category.strip():
            raise ValueError("required_category must not be whitespace")
        for group in (
            self.required_attributes,
            self.excluded_product_ids,
            self.excluded_brands,
        ):
            if any(not item.strip() for item in group):
                raise ValueError("preference values must not be empty")
        return self


class EvidenceReference(Contract):
    evidence_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    supported_field: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_evidence(self) -> "EvidenceReference":
        if any(
            not getattr(self, field).strip()
            for field in ("evidence_id", "product_id", "source_id", "source_type", "supported_field", "excerpt")
        ):
            raise ValueError("evidence fields must not be whitespace")
        return self


class Catalog(Contract):
    products: list[Product] = Field(min_length=1)
    evidence: list[EvidenceReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_links(self) -> "Catalog":
        ids = [p.product_id for p in self.products]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate product_id")
        evidence_ids = [e.evidence_id for e in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("duplicate evidence_id")

        by_id = {p.product_id: p for p in self.products}
        for record in self.evidence:
            if record.product_id not in by_id:
                raise ValueError(f"unknown product_id in evidence: {record.product_id}")
            product = by_id[record.product_id]
            allowed = {"title", "brand", "category", "price_usd"} | {
                f"attributes.{key}" for key in product.attributes
            }
            if record.supported_field not in allowed:
                raise ValueError(f"unknown supported_field: {record.supported_field}")
        return self
