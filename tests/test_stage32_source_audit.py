"""Fictional source audit covers contradictory, superseded and absent fields."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from recsys.evaluation.source_audit import audit_source_fixture


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/source_evidence_demo/catalog.json"
SOURCES = ROOT / "data/source_evidence_demo/source_documents.json"


def test_source_excerpt_linkage_and_expected_synthetic_statuses():
    result = audit_source_fixture(CATALOG, SOURCES)
    assert result["source_document_count"] == 7
    assert result["source_claim_count"] == 12
    assert len(result["catalog_sha256"]) == 64
    assert len(result["source_docs_sha256"]) == 64
    assert result["semantic_entailment_verified"] is False
    assert result["real_product_verification"] is False
    statuses = {(x["product_id"], x["field"]): x for x in result["field_results"]}
    assert statuses["SYN-E01", "attributes.fragrance_free"]["status"] == "consistent"
    assert statuses["SYN-E02", "attributes.fragrance_free"]["status"] == "conflict"
    assert statuses["SYN-E03", "attributes.mineral"]["status"] == "stale_metadata_after_revision"
    assert statuses["SYN-E04", "attributes.fragrance_free"]["status"] == "missing"
    assert result["field_status_counts"] == {
        "consistent": 4, "conflict": 1, "stale_metadata_after_revision": 1, "missing": 1,
    }


def test_revision_supersedes_prior_source_but_does_not_silently_fix_product_metadata():
    result = audit_source_fixture(CATALOG, SOURCES)
    label = next(x for x in result["field_results"]
                 if x["product_id"] == "SYN-E03" and x["field"] == "attributes.mineral")
    assert label["active_asserted_values"] == [False]
    assert label["metadata_value"] is True
    assert label["active_source_ids"] == ["DOC-E03-SPEC-V2"]
    assert label["superseded_source_ids"] == ["DOC-E03-SPEC-V1"]


def test_fabricated_verbatim_excerpt_is_rejected(tmp_path):
    source = json.loads(SOURCES.read_text())
    source["documents"][0]["claims"][0]["excerpt"] = "something not present in the document"
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValidationError, match="verbatim"):
        audit_source_fixture(CATALOG, path)


def test_nonexistent_source_linkage_and_false_source_values_rejected(tmp_path):
    source = json.loads(SOURCES.read_text())
    source["documents"][0]["source_id"] = "UNLISTED"
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="exactly match"):
        audit_source_fixture(CATALOG, path)
    source = json.loads(SOURCES.read_text())
    source["documents"][0]["claims"][0]["asserted_value"] = "yes"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="known boolean"):
        audit_source_fixture(CATALOG, path)


def test_bad_supersession_detected(tmp_path):
    source = json.loads(SOURCES.read_text())
    source["documents"][4]["supersedes"] = "DOC-E02-SPEC"
    path = tmp_path / "bad-revision.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="another product"):
        audit_source_fixture(CATALOG, path)
