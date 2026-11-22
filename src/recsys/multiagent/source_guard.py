"""Fail-closed provenance checks using a vetted, *typed* source-document fixture.

Only verifies the explicitly annotated fields and verbatim linked source excerpts.
It cannot decide whether arbitrary natural-language recommendation reasons are
semantically entailed by documents. All example sources are FICTIONAL.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recsys.evaluation.source_audit import SourceFixture, audit_source_fixture
from recsys.retrieval.catalog import load_catalog


@dataclass(frozen=True)
class SourceFinding:
    code: str
    product_id: str
    field: str
    detail: str


class TypedSourceGuard:
    """Provenance-bound source revision/conflict validator.

    Construction re-audits the complete source/citation index, then snapshots
    the SHA-256 of catalog + source documents. The catalog is never mutated.
    """

    def __init__(self, *, catalog_path: Path, source_path: Path):
        self.audit = audit_source_fixture(catalog_path, source_path)
        catalog, _ = load_catalog(catalog_path)
        fixture = SourceFixture.model_validate_json(source_path.read_bytes())
        self.catalog_sha256: str = self.audit["catalog_sha256"]
        self.source_docs_sha256: str = self.audit["source_docs_sha256"]
        self.field_status = {
            (row["product_id"], row["field"]): row
            for row in self.audit["field_results"]
        }
        self._products = {product.product_id: product for product in catalog.products}
        self._documents = {doc.source_id: doc for doc in fixture.documents}
        self._evidence = {e.evidence_id: e for e in catalog.evidence}
        self._active_claims: dict[tuple[str, str], list[tuple[str, Any]]] = {}
        inactive = {
            document.supersedes
            for document in fixture.documents
            if document.supersedes is not None
        }
        # The full closure of a supersession chain is also inactive.
        for doc in fixture.documents:
            prev = doc.supersedes
            while prev:
                inactive.add(prev)
                prev = self._documents[prev].supersedes
        self._inactive_source_ids = inactive
        for doc in fixture.documents:
            if doc.source_id not in inactive:
                for claim in doc.claims:
                    self._active_claims.setdefault(
                        (doc.product_id, claim.supported_field), []
                    ).append((doc.source_id, claim.asserted_value))

    def bind(self, retrieval_catalog_sha: str) -> None:
        if retrieval_catalog_sha != self.catalog_sha256:
            raise ValueError("source guard catalog SHA-256 differs from retrieval snapshot")

    def is_field_supported(self, product_id: str, field: str) -> bool:
        """A boolean required attribute is safe only with active, consistent support."""
        status = self.field_status.get((product_id, field))
        return status is not None and status["status"] == "consistent"

    def eligible_for_required_fields(self, product_id: str, required_attrs: set[str]) -> bool:
        return all(
            self.is_field_supported(product_id, f"attributes.{attribute}")
            for attribute in required_attrs
        )

    def evaluate(
        self,
        *,
        product_id: str,
        required_attributes: set[str],
        evidence_ids: set[str],
    ) -> list[SourceFinding]:
        """Check field provenance, declared conflicts and superseded citations.

        This does NOT infer claims from a recommendation.reason free-text string.
        """
        if product_id not in self._products:
            return [SourceFinding(
                "source_claim_missing", product_id, "product_id",
                "Product is not in the bound source catalog",
            )]
        findings: list[SourceFinding] = []
        fields = {f"attributes.{name}" for name in required_attributes}
        for eid in sorted(evidence_ids):
            ref = self._evidence.get(eid)
            if ref is None or ref.product_id != product_id:
                findings.append(SourceFinding(
                    "source_claim_missing", product_id, "citation",
                    "Citation missing or attached to another product",
                ))
                continue
            if ref.source_id in self._inactive_source_ids:
                findings.append(SourceFinding(
                    "superseded_evidence", product_id, ref.supported_field,
                    "Cited source was superseded by a later document",
                ))
            if ref.supported_field.startswith("attributes."):
                fields.add(ref.supported_field)
        for field in sorted(fields):
            status = self.field_status.get((product_id, field))
            state = status["status"] if status else "missing"
            if state == "conflict":
                findings.append(SourceFinding(
                    "source_conflict", product_id, field,
                    "Active authored documents assert contradictory values",
                ))
            elif state in {"stale_metadata_after_revision", "contradicted_metadata"}:
                findings.append(SourceFinding(
                    "source_stale_metadata", product_id, field,
                    "Active documented value contradicts current catalog metadata",
                ))
            elif state == "missing":
                findings.append(SourceFinding(
                    "source_claim_missing", product_id, field,
                    "No active annotated source claim supports this catalog field",
                ))
        return findings

    def classify_annotated_claim(
        self, *, product_id: str, field: str,
        asserted_value: bool | str, evidence_ids: set[str],
    ) -> tuple[str, str]:
        """Classify only an EXPLICIT typed atomic claim against the active sources.

        These rules apply to a manually/externally extracted field assertion.
        They do NOT extract facts from prose or prove open-text entailment.
        Labels are deliberately distinct: a clear active opposite is
        contradicted, conflicting active sources are ambiguous, missing or
        invalid citations to an otherwise consistent value are unsupported.
        """
        product = self._products.get(product_id)
        if product is None:
            return "unsupported", "unknown_product"
        if field.startswith("attributes."):
            key = field[len("attributes."):]
            if key not in product.attributes or type(asserted_value) is not bool:
                return "unsupported", "invalid_typed_attribute"
        elif field == "price_usd":
            from decimal import Decimal, InvalidOperation
            if type(asserted_value) is not str:
                return "unsupported", "invalid_typed_price"
            try:
                value = Decimal(asserted_value)
            except InvalidOperation:
                return "unsupported", "invalid_typed_price"
            if not value.is_finite() or value < 0:
                return "unsupported", "invalid_typed_price"
        else:
            return "unsupported", "field_outside_typed_scope"

        active = self._active_claims.get((product_id, field), [])
        if not active:
            return "unsupported", "no_active_source_assertion"

        def comparable(v):
            return Decimal(v) if field == "price_usd" else v

        values = {comparable(v) for _, v in active}
        if len(values) != 1:
            return "ambiguous", "active_sources_disagree"
        if comparable(asserted_value) not in values:
            return "contradicted", "active_sources_assert_opposite"
        if not evidence_ids:
            return "unsupported", "claim_has_no_citation"
        active_source_ids = {source_id for source_id, _ in active}
        for eid in evidence_ids:
            evidence = self._evidence.get(eid)
            if evidence is None:
                return "unsupported", "unknown_citation_id"
            if evidence.product_id != product_id:
                return "unsupported", "citation_other_product"
            if evidence.supported_field != field:
                return "unsupported", "citation_wrong_field"
            if evidence.source_id not in active_source_ids:
                return "unsupported", "citation_not_active"
        return "supported", "active_value_and_all_citations_match"

    def verify_typed_claim(
        self, *,
        product_id: str, field: str, asserted_value: bool | str,
        evidence_ids: set[str],
    ) -> bool:
        """Validate a *declared typed atomic claim*, not prose entailment.

        The value must match both catalog metadata and every active authored
        assertion of the field, and at least one cited active source must
        explicitly assert the same field/value. Fail closed on contradictions.
        """
        product = self._products.get(product_id)
        if product is None or not evidence_ids:
            return False
        if field.startswith("attributes."):
            key = field[len("attributes."):]
            if key not in product.attributes or type(asserted_value) is not bool:
                return False
            if product.attributes[key] is not asserted_value:
                return False
            if not self.is_field_supported(product_id, field):
                return False
        elif field == "price_usd":
            if type(asserted_value) is not str or product.price_usd is None:
                return False
            try:
                from decimal import Decimal
                if Decimal(asserted_value) != product.price_usd:
                    return False
            except Exception:
                return False
        else:
            return False
        active = self._active_claims.get((product_id, field), [])
        if not active:
            return False
        if any(value != asserted_value for _, value in active):
            return False
        sources = {source_id for source_id, _ in active}
        # Fail closed on *every* cited identifier: one valid source cannot
        # launder a fabricated, off-field, cross-product or obsolete citation.
        for eid in evidence_ids:
            evidence = self._evidence.get(eid)
            if (
                evidence is None
                or evidence.product_id != product_id
                or evidence.supported_field != field
                or evidence.source_id not in sources
                or evidence.source_id in self._inactive_source_ids
            ):
                return False
        return True
