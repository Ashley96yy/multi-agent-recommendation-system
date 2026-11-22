"""Stage 3.4 atomic claim benchmark (offline, authored fictional annotations).

These tests do NOT evaluate free-text model entailment or real product truth.
"""

import csv
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from recsys.evaluation.atomic_claims import (
    CASES, LABELS, AtomicClaimCase, AtomicClaimSuite,
    classify, evaluate_claims, export_blind_review_csv, load_frozen_claims,
    naive_citation_baseline,
)


ROOT = Path(__file__).resolve().parents[1]


def run():
    suite, labels, guard, hashes = load_frozen_claims()
    report = evaluate_claims(suite, labels, guard, hashes)
    return suite, labels, guard, hashes, report


def test_frozen_synthetic_labels_and_provenance_are_complete():
    suite, labels, guard, hashes, result = run()
    assert len(suite.cases) == 26
    assert len(labels) == 26
    assert result["typed_case_count"] == 22
    assert result["free_text_review_required_count"] == 4
    assert result["automated_claim_coverage_rate"] == 22 / 26
    assert set(case.case_id for case in suite.cases) == set(labels)
    assert all(len(hash) == 64 for hash in hashes.values())
    assert hashes["catalog_sha256"] == guard.catalog_sha256
    assert hashes["source_documents_sha256"] == guard.source_docs_sha256
    assert not result["author_labels_independently_adjudicated"]
    assert result["unsupported_claim_rate_real_model"] is None
    assert result["real_world_claim_groundedness"] is None


def test_four_class_typed_predictions_match_fictional_author_oracle():
    _, _, _, _, report = run()
    typed = [r for r in report["per_case"] if r["kind"] == "typed"]
    assert len(typed) == 22
    assert all(r["matches_author_label"] is True for r in typed)
    assert report["typed_judge"]["author_label_agreement_count"] == 22
    assert report["typed_judge"]["macro_f1_against_author_labels"] == 1.0
    assert {r["prediction"] for r in typed} == {
        "supported", "contradicted", "unsupported", "ambiguous"
    }
    assert report["typed_judge"]["per_label"]["supported"]["support"] == 6
    assert report["typed_judge"]["per_label"]["contradicted"]["support"] == 5
    assert report["typed_judge"]["per_label"]["unsupported"]["support"] == 6
    assert report["typed_judge"]["per_label"]["ambiguous"]["support"] == 5


def test_naive_citation_existence_is_not_grounding():
    _, _, _, _, report = run()
    naive = report["naive_citation_only_control"]
    assert naive["author_label_agreement_count"] == 8
    assert naive["typed_evaluated"] == 22
    assert naive["macro_f1_against_author_labels"] < 1
    by_id = {row["case_id"]: row for row in report["per_case"]}
    assert by_id["T10"]["naive_citation_baseline_prediction"] == "supported"
    assert by_id["T10"]["prediction"] == "contradicted"
    assert by_id["T18"]["naive_citation_baseline_prediction"] == "supported"
    assert by_id["T18"]["prediction"] == "ambiguous"
    assert by_id["T15"]["prediction"] == "unsupported"


def test_free_text_never_gets_semantic_label_without_independent_review():
    _, _, _, _, report = run()
    unknowns = [row for row in report["per_case"] if row["kind"] == "open_text"]
    assert len(unknowns) == 4
    assert all(row["prediction"] == "needs_human_review" for row in unknowns)
    assert all(row["matches_author_label"] is None for row in unknowns)
    assert all(row["method"] == "abstain_unstructured_claim" for row in unknowns)
    assert {x["author_label"] for x in unknowns} == {
        "supported", "ambiguous", "unsupported"
    }


def test_blind_review_csv_has_original_excerpts_without_gold_or_model_predictions(tmp_path):
    _, _, guard, _, report = run()
    output = tmp_path / "review.csv"
    assert export_blind_review_csv(report, guard, output) == 26
    with output.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 26
    assert all(row["reviewer_label"] == "" and row["reviewer_id"] == "" for row in rows)
    assert all("author_label" not in row and "prediction" not in row for row in rows)
    rev = next(row for row in rows if row["case_id"] == "T10")
    citations = json.loads(rev["source_evidence_json"])
    assert citations[0]["superseded"] is True
    assert citations[0]["source_id"] == "DOC-E03-SPEC-V1"
    invalid = next(row for row in rows if row["case_id"] == "T15")
    assert any(item["source_id"] == "DOC-E02-QC" for item in json.loads(invalid["source_evidence_json"]))


def test_runtime_source_guard_four_labels_and_no_citation_laundering():
    _, _, guard, _, _ = run()
    assert guard.classify_annotated_claim(
        product_id="SYN-E01", field="attributes.fragrance_free",
        asserted_value=True, evidence_ids={"E-SOURCE-001"}
    ) == ("supported", "active_value_and_all_citations_match")
    assert guard.classify_annotated_claim(
        product_id="SYN-E03", field="attributes.mineral",
        asserted_value=True, evidence_ids={"E-SOURCE-008"}
    )[0] == "contradicted"
    assert guard.classify_annotated_claim(
        product_id="SYN-E02", field="attributes.fragrance_free",
        asserted_value=True, evidence_ids={"E-SOURCE-004"}
    )[0] == "ambiguous"
    assert guard.classify_annotated_claim(
        product_id="SYN-E01", field="attributes.fragrance_free",
        asserted_value=True, evidence_ids={"E-SOURCE-001", "E-SOURCE-007"}
    ) == ("unsupported", "citation_other_product")
    assert guard.classify_annotated_claim(
        product_id="SYN-E04", field="attributes.fragrance_free",
        asserted_value=True, evidence_ids={"E-SOURCE-012"}
    )[0] == "unsupported"


def test_claim_schema_catches_unextractable_or_misleading_fields():
    with pytest.raises(ValidationError, match="verbatim"):
        AtomicClaimCase(
            case_id="broken", kind="typed", product_id="SYN-E01",
            reason="The product is good", claim_span="not in the answer",
            field="attributes.fragrance_free", asserted_value=True,
        )
    with pytest.raises(ValidationError, match="must not pretend"):
        AtomicClaimCase(
            case_id="broken", kind="open_text", product_id="SYN-E01",
            reason="This is fiction", claim_span="fiction",
            field="attributes.fragrance_free", asserted_value=True,
        )
    with pytest.raises(ValidationError, match="duplicate"):
        AtomicClaimCase(
            case_id="broken", kind="typed", product_id="SYN-E01",
            reason="No added fragrance", claim_span="fragrance",
            field="attributes.fragrance_free", asserted_value=True,
            evidence_ids=["E-SOURCE-001", "E-SOURCE-001"],
        )


def test_gold_label_leakage_and_duplicate_ids_rejected(tmp_path):
    case_obj = json.loads(CASES.read_text())
    case_obj["cases"][1]["case_id"] = case_obj["cases"][0]["case_id"]
    case_file = tmp_path / "dup-cases.json"
    case_file.write_text(json.dumps(case_obj))
    with pytest.raises(ValidationError, match="duplicate atomic"):
        load_frozen_claims(case_file, LABELS)
    case_obj = json.loads(CASES.read_text())
    case_obj["cases"][0]["author_label"] = "supported"
    case_file.write_text(json.dumps(case_obj))
    with pytest.raises(ValidationError, match="Extra inputs"):
        load_frozen_claims(case_file, LABELS)


def test_wrong_label_ids_and_outside_repo_paths_rejected(tmp_path):
    gold = json.loads(LABELS.read_text())
    gold["labels"][0]["case_id"] = "WRONG_CASE_ID"
    path = tmp_path / "wrong-labels.json"
    path.write_text(json.dumps(gold))
    with pytest.raises(ValueError, match="cover each case"):
        load_frozen_claims(CASES, path)

    cases = json.loads(CASES.read_text())
    cases["sources_path"] = "../../escaping.json"
    outside = tmp_path / "escape.json"
    outside.write_text(json.dumps(cases))
    with pytest.raises(ValueError, match="escapes repository"):
        load_frozen_claims(outside, LABELS)


def test_provenance_mismatch_blocks_replacement_catalog(tmp_path):
    cases = json.loads(CASES.read_text())
    cases["catalog_path"] = "data/retrieval_eval/catalog.json"
    changed = tmp_path / "wrong-catalog.json"
    changed.write_text(json.dumps(cases))
    with pytest.raises(ValueError):
        load_frozen_claims(changed, LABELS)
