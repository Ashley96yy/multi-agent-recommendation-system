"""Synthetic attribute-evidence enrichment, explicitly derived from metadata.

This creates *internally traceable* synthetic attribute declarations. It does
NOT add independent product documentation, validate real-world efficacy, or
establish semantic entailment of generated recommendations.
"""

import argparse
import hashlib
import json
from pathlib import Path

from recsys.retrieval.catalog import load_catalog
from recsys.schemas.contracts import Catalog, EvidenceReference


def attribute_coverage(catalog: Catalog) -> dict:
    known_fields = {
        (p.product_id, f"attributes.{attr}")
        for p in catalog.products for attr in p.attributes
    }
    documented_fields = {
        (e.product_id, e.supported_field)
        for e in catalog.evidence if e.supported_field.startswith("attributes.")
    }
    documented = known_fields & documented_fields
    return {
        "known_attribute_fields": len(known_fields),
        "with_linked_declaration": len(documented),
        "missing_declaration": len(known_fields - documented),
        "coverage": len(documented) / len(known_fields) if known_fields else None,
    }


def derive_attribute_evidence(catalog: Catalog) -> Catalog:
    """Create deterministic synthetic field declarations only for known values.

    Source type is intentionally synthetic; the new evidence is derived from
    exactly the same metadata field, not an independent information source.
    """
    records = list(catalog.evidence)
    existing = {(e.product_id, e.supported_field) for e in records}
    existing_ids = {e.evidence_id for e in records}
    for product in catalog.products:
        for attribute, value in sorted(product.attributes.items()):
            field = f"attributes.{attribute}"
            if (product.product_id, field) in existing:
                continue
            evidence_id = f"E-{product.product_id}-A-{attribute}"
            if evidence_id in existing_ids:
                raise ValueError("generated synthetic evidence ID collides with existing source")
            records.append(EvidenceReference(
                evidence_id=evidence_id,
                product_id=product.product_id,
                source_id="synthetic-metadata-derived-attribute-v2",
                source_type="synthetic_metadata_derivation",
                supported_field=field,
                excerpt=f"Fictional metadata declaration for {product.product_id}: {attribute}={str(value).lower()}.",
            ))
            existing.add((product.product_id, field))
            existing_ids.add(evidence_id)
    return Catalog.model_validate({
        "products": catalog.products,
        "evidence": records,
    })


def run(source_catalog: Path, output_catalog: Path, output_manifest: Path) -> dict:
    original, source_sha = load_catalog(source_catalog)
    augmented = derive_attribute_evidence(original)
    before, after = attribute_coverage(original), attribute_coverage(augmented)
    if before["known_attribute_fields"] != after["known_attribute_fields"]:
        raise AssertionError("attribute evidence enrichment modified source attribute facts")
    if any(p1 != p2 for p1, p2 in zip(original.products, augmented.products)):
        raise AssertionError("product facts changed")
    data = augmented.model_dump_json(indent=2) + "\n"
    output_catalog.parent.mkdir(parents=True, exist_ok=True)
    output_catalog.write_text(data, encoding="utf-8")
    artifact_sha = hashlib.sha256(data.encode("utf-8")).hexdigest()
    manifest = {
        "source_catalog_sha256": source_sha,
        "derived_catalog_sha256": artifact_sha,
        "source_type": "synthetic_metadata_derivation",
        "new_evidence_records": len(augmented.evidence) - len(original.evidence),
        "before": before,
        "after": after,
        "independent_evidence": False,
        "disclaimer": (
            "All added field declarations are generated directly from existing "
            "fictional catalog attributes; they are not external corroboration "
            "or independent verification of real product properties."
        ),
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/retrieval_eval/catalog.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/evaluation/derived_attribute_catalog.json"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("artifacts/evaluation/derived_attribute_manifest.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output, args.manifest), indent=2))


if __name__ == "__main__":
    main()
