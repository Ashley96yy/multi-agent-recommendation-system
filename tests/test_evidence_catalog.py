"""Evidence enrichment asserts only internal, synthetic metadata traceability."""

import hashlib
import json
from pathlib import Path

from recsys.evaluation.evidence_catalog import (
    attribute_coverage,
    derive_attribute_evidence,
    run,
)
from recsys.retrieval.catalog import load_catalog
from recsys.schemas.contracts import Catalog


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "data/retrieval_eval/catalog.json"


def test_existing_catalog_has_no_independent_attribute_evidence():
    source, _ = load_catalog(ORIGINAL)
    values = attribute_coverage(source)
    assert values["known_attribute_fields"] > 0
    assert values["with_linked_declaration"] == 0


def test_derived_attribute_declarations_are_complete_and_deterministic():
    source, _ = load_catalog(ORIGINAL)
    a = derive_attribute_evidence(source)
    b = derive_attribute_evidence(source)
    assert a == b
    assert a.products == source.products
    assert a.evidence[:len(source.evidence)] == source.evidence
    assert attribute_coverage(a)["coverage"] == 1.0
    assert len({x.evidence_id for x in a.evidence}) == len(a.evidence)
    assert all(
        x.source_type == "synthetic_metadata_derivation"
        and "Fictional metadata declaration" in x.excerpt
        for x in a.evidence[len(source.evidence):]
    )


def test_unknown_price_is_not_fabricated_and_false_attributes_are_explicit():
    source, _ = load_catalog(ORIGINAL)
    augmented = derive_attribute_evidence(source)
    assert next(p for p in augmented.products if p.product_id == "SYN-R13").price_usd is None
    assert not any(e.product_id == "SYN-R13" and e.supported_field == "price_usd"
                   for e in augmented.evidence)
    known_false = next(e for e in augmented.evidence
                       if e.product_id == "SYN-R02"
                       and e.supported_field == "attributes.fragrance_free")
    assert known_false.excerpt.endswith("fragrance_free=false.")


def test_augmented_catalog_is_valid_and_manifest_disclaims_source_independence(tmp_path):
    out = tmp_path / "enriched.json"
    manifest_path = tmp_path / "enriched-manifest.json"
    report = run(ORIGINAL, out, manifest_path)
    assert report["independent_evidence"] is False
    assert report["before"]["coverage"] == 0.0
    assert report["after"]["coverage"] == 1.0
    assert report["derived_catalog_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert report == json.loads(manifest_path.read_text())
    Catalog.model_validate_json(out.read_bytes())


def test_running_enrichment_twice_does_not_duplicate_evidence():
    source, _ = load_catalog(ORIGINAL)
    augmented = derive_attribute_evidence(source)
    repeated = derive_attribute_evidence(augmented)
    assert repeated == augmented
