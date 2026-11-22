"""Real-model review queue exporter tests use a fake report, no model inference."""

import csv
import json
from pathlib import Path

import pytest

from recsys.evaluation.model_reason_review import export_model_review
from recsys.retrieval.catalog import load_catalog


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/retrieval_eval/catalog.json"


def fake_report():
    catalog, sha = load_catalog(CATALOG)
    pid = catalog.products[0].product_id
    eid = next(e.evidence_id for e in catalog.evidence if e.product_id == pid)
    content = json.dumps({
        "recommendations": [{
            "product_id": pid,
            "reason": "Claim about a fictional product. Citation membership alone is not proof.",
            "evidence_ids": [eid],
        }],
        "no_suitable_match": False,
    })
    rows = []
    for i in range(25):
        row = {
            "case_id": f"CASE-{i:02d}",
            "status": "approved" if i < 9 else "rejected",
            "raw_first_output": content if i < 9 else "invalid",
            "recommended_ids": [pid] if i < 9 else [],
        }
        rows.append(row)
    return {
        "suite_id": "single-agent-baseline-synthetic-25-v1",
        "data_hashes": {"catalog_sha256": sha},
        "generator": "fake-offline-generator",
        "matrix": {
            "physical_model_invocations": 19,
            "arms": {
                "no_preflight__safe_json_normalization": {
                    "case_count": 25, "critic_approved_count": 9,
                    "per_case": rows,
                },
            },
        },
    }


def extract(tmp_path, report):
    source = tmp_path / "model-report.json"
    output = tmp_path / "blind-model-reasons.csv"
    manifest = tmp_path / "manifest.json"
    source.write_text(json.dumps(report))
    result = export_model_review(source, CATALOG, output, manifest)
    return result, output, manifest


def test_export_true_text_and_source_excerpts_with_zero_claim_annotations(tmp_path):
    manifest, csv_path, manifest_path = extract(tmp_path, fake_report())
    assert manifest["approved_case_count"] == 9
    assert manifest["recommendation_reason_rows"] == 9
    assert manifest["atomic_claims_manually_extracted"] == 0
    assert manifest["semantic_groundedness"] is None
    assert manifest["unsupported_claim_rate"] is None
    assert len(manifest["source_report_sha256"]) == 64
    assert json.loads(manifest_path.read_text()) == manifest
    with csv_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 9
    assert all(row["reviewer_label"] == "" and row["atomic_claim_span"] == ""
               for row in rows)
    assert all("author_label" not in row and "prediction" not in row for row in rows)
    assert "Citation membership alone" in rows[0]["model_reason"]
    assert json.loads(rows[0]["cited_evidence_json"])[0]["excerpt"]


def test_invalid_catalog_provenance_fails_closed(tmp_path):
    data = fake_report()
    data["data_hashes"]["catalog_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance"):
        extract(tmp_path, data)


def test_cross_product_or_fake_citations_cannot_be_exported_as_approved(tmp_path):
    data = fake_report()
    raw = json.loads(data["matrix"]["arms"]["no_preflight__safe_json_normalization"]["per_case"][0]["raw_first_output"])
    raw["recommendations"][0]["evidence_ids"] = ["E-FAKE"]
    data["matrix"]["arms"]["no_preflight__safe_json_normalization"]["per_case"][0]["raw_first_output"] = json.dumps(raw)
    with pytest.raises(ValueError, match="invalid or cross-product"):
        extract(tmp_path, data)


def test_mismatched_recommended_ids_and_duplicate_case_ids_rejected(tmp_path):
    data = fake_report()
    data["matrix"]["arms"]["no_preflight__safe_json_normalization"]["per_case"][0]["recommended_ids"] = ["NOT-REAL"]
    with pytest.raises(ValueError, match="reported IDs"):
        extract(tmp_path, data)
    data = fake_report()
    rows = data["matrix"]["arms"]["no_preflight__safe_json_normalization"]["per_case"]
    rows[1]["case_id"] = rows[0]["case_id"]
    with pytest.raises(ValueError, match="case records incomplete"):
        extract(tmp_path, data)


def test_missing_raw_output_refused(tmp_path):
    data = fake_report()
    data["matrix"]["arms"]["no_preflight__safe_json_normalization"]["per_case"][0]["raw_first_output"] = None
    with pytest.raises(ValueError, match="no raw text"):
        extract(tmp_path, data)
