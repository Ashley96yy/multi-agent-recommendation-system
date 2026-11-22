"""Audit manually authored FICTIONAL source passages and indexed citation integrity.

Document-level contradiction / supersession checks are deterministic and
narrow. They do NOT infer semantic entailment from arbitrary language.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from recsys.retrieval.catalog import load_catalog


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    supported_field: str = Field(min_length=1)
    asserted_value: bool | str
    excerpt: str = Field(min_length=1)


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    effective_date: str = Field(min_length=1)
    supersedes: str | None = None
    text: str = Field(min_length=1)
    claims: list[Claim] = Field(min_length=1)

    @model_validator(mode="after")
    def check_excerpts(self):
        date.fromisoformat(self.effective_date)
        for claim in self.claims:
            if claim.excerpt not in self.text:
                raise ValueError("claim excerpt must be verbatim in document text")
        return self


class SourceFixture(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    fixture_id: str = Field(min_length=1)
    provenance: str = Field(min_length=1)
    documents: list[Document] = Field(min_length=1)


def audit_source_fixture(catalog_path: Path, source_path: Path) -> dict:
    catalog, catalog_hash = load_catalog(catalog_path)
    source_bytes = source_path.read_bytes()
    fixture = SourceFixture.model_validate_json(source_bytes)
    product_by_id = {p.product_id: p for p in catalog.products}
    docs = {d.source_id: d for d in fixture.documents}
    if len(docs) != len(fixture.documents):
        raise ValueError("duplicate source document ID")
    if len(set((d.product_id, d.effective_date, d.source_id) for d in docs.values())) != len(docs):
        raise ValueError("duplicated document entries")

    for d in docs.values():
        if d.product_id not in product_by_id:
            raise ValueError("document refers to unknown product")
        if d.supersedes is not None:
            older = docs.get(d.supersedes)
            if older is None or older.product_id != d.product_id:
                raise ValueError("superseded document missing or belongs to another product")
            if older.effective_date >= d.effective_date:
                raise ValueError("revision must postdate superseded document")

    def revision_chain(start: Document) -> set[str]:
        visited = set()
        ptr = start
        while ptr.supersedes:
            if ptr.supersedes in visited:
                raise ValueError("revision supersession cycle")
            visited.add(ptr.supersedes)
            ptr = docs[ptr.supersedes]
        return visited

    superseded = set()
    for d in docs.values():
        superseded.update(revision_chain(d))

    claim_keys = Counter()
    statements: dict[tuple[str, str], list[tuple[Document, bool | str]]] = defaultdict(list)
    for d in docs.values():
        product = product_by_id[d.product_id]
        for claim in d.claims:
            field = claim.supported_field
            if field.startswith("attributes."):
                key = field[len("attributes."):]
                if key not in product.attributes or type(claim.asserted_value) is not bool:
                    raise ValueError("attribute claim must be a known boolean field")
            elif field == "price_usd":
                if type(claim.asserted_value) is not str:
                    raise ValueError("price claim must be a decimal string")
                try:
                    value = Decimal(claim.asserted_value)
                except InvalidOperation as exc:
                    raise ValueError("invalid price claim decimal") from exc
                if not value.is_finite() or value < 0:
                    raise ValueError("invalid price claim value")
            else:
                raise ValueError("source audit only accepts known attribute or price claims")
            claim_keys[(d.product_id, d.source_id, d.source_type, field, claim.excerpt)] += 1
            statements[(d.product_id, field)].append((d, claim.asserted_value))

    linked_keys = Counter(
        (e.product_id, e.source_id, e.source_type, e.supported_field, e.excerpt)
        for e in catalog.evidence
    )
    if claim_keys != linked_keys:
        raise ValueError("catalog evidence must exactly match source excerpts, fields and document IDs")

    results = []
    for product in catalog.products:
        for attribute in sorted(product.attributes):
            field = f"attributes.{attribute}"
            observations = statements.get((product.product_id, field), [])
            active = [(d, v) for d, v in observations if d.source_id not in superseded]
            distinct = {v for _, v in active}
            known_value = product.attributes[attribute]
            has_revision = any(d.supersedes is not None for d, _ in observations)
            if not active:
                status = "missing"
            elif len(distinct) > 1:
                status = "conflict"
            elif next(iter(distinct)) != known_value:
                status = "stale_metadata_after_revision" if has_revision else "contradicted_metadata"
            else:
                status = "consistent"
            results.append({
                "product_id": product.product_id,
                "field": field,
                "metadata_value": known_value,
                "status": status,
                "active_source_ids": [d.source_id for d, _ in active],
                "superseded_source_ids": [d.source_id for d, _ in observations
                                          if d.source_id in superseded],
                "active_asserted_values": [value for _, value in active],
            })
    return {
        "fixture_id": fixture.fixture_id,
        "catalog_sha256": catalog_hash,
        "source_docs_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_document_count": len(docs),
        "source_claim_count": sum(claim_keys.values()),
        "field_status_counts": dict(Counter(row["status"] for row in results)),
        "field_results": results,
        "semantic_entailment_verified": False,
        "real_product_verification": False,
        "limitations": (
            "Fictional independently authored passages and typed annotation claims "
            "are audited for index/source consistency and explicit contradictory "
            "facts; this is not independent real-world product verification or "
            "an open-text semantic-entailment judge."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/source_evidence_demo/catalog.json"))
    parser.add_argument("--sources", type=Path, default=Path("data/source_evidence_demo/source_documents.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/source-evidence-audit.json"))
    args = parser.parse_args()
    report = audit_source_fixture(args.catalog, args.sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        key: report[key] for key in ("fixture_id", "source_document_count",
                                     "source_claim_count", "field_status_counts")
    }, indent=2))


if __name__ == "__main__":
    main()
