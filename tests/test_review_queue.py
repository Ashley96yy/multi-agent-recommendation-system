"""Human-review queue must never silently assign support labels."""

import json
from pathlib import Path

import pytest

from recsys.evaluation.review_queue import build_review_queue, export_review_csv
from recsys.retrieval.catalog import load_catalog


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/retrieval_eval/catalog.json"


def report(*, evidence_id="E-SYN-R11-T"):
    _, sha = load_catalog(CATALOG)
    raw = json.dumps({
        "recommendations": [{
            "product_id": "SYN-R11",
            "reason": "A soothing balm proven to cure all skin conditions.",
            "evidence_ids": [evidence_id],
        }],
        "no_suitable_match": False,
    })
    return {
        "data_hashes": {"catalog_sha256": sha},
        "per_case": [{
            "case_id": "RT08", "group": "red_team", "status": "accepted",
            "raw_output": raw,
        }, {
            "case_id": "N01", "group": "red_team", "status": "abstained",
        }],
    }


def test_export_contains_claim_and_source_with_blank_review_label(tmp_path):
    source = tmp_path / "report.json"
    output = tmp_path / "review.csv"
    source.write_text(json.dumps(report()))
    assert export_review_csv(source, CATALOG, output) == 1
    exported = output.read_text()
    assert "E-SYN-R11-T" in exported
    assert "proven to cure" in exported
    assert "Fictional listing title" in exported
    assert "support_label" in exported
    assert build_review_queue(report(), CATALOG)[0]["support_label"] == ""


def test_incorrect_provenance_cannot_generate_review():
    bad = report()
    bad["data_hashes"]["catalog_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance"):
        build_review_queue(bad, CATALOG)


def test_cross_product_citation_rejected():
    with pytest.raises(ValueError, match="wrong product"):
        build_review_queue(report(evidence_id="E-SYN-R05-T"), CATALOG)


def test_accepted_recommendation_without_raw_output_fails_closed():
    bad = report()
    del bad["per_case"][0]["raw_output"]
    with pytest.raises(ValueError, match="missing actual"):
        build_review_queue(bad, CATALOG)
