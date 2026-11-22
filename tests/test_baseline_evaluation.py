"""Offline Stage 2.2 evaluation tests. These do NOT execute Qwen."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from recsys.baseline import SingleAgentRAG
from recsys.baseline.generators import FixedJSONGenerator
from recsys.evaluation.baseline_score import _quantile, evaluate_baseline
from recsys.evaluation.baseline_suite import BaselineCase, load_baseline_suite
from recsys.retrieval import CatalogRetriever


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/retrieval_eval/catalog.json"
SUITE = ROOT / "data/baseline_eval/suite.json"


def setup():
    retriever = CatalogRetriever.from_json(CATALOG)
    cases, hashes = load_baseline_suite(SUITE, repo_root=ROOT, retriever=retriever)
    return retriever, cases, hashes


def test_manifest_contains_exactly_15_normal_10_adversarial_cases():
    _, cases, hashes = setup()
    assert len(cases) == 25
    assert len({case.case_id for case in cases}) == 25
    assert sum(case.group == "normal" for case in cases) == 15
    assert sum(case.group == "red_team" for case in cases) == 10
    assert sum(case.expected_type == "no_match" for case in cases) == 6
    assert all(len(value) == 64 for value in hashes.values())


def test_all_negative_cases_fail_closed_without_model_call():
    retriever, cases, _ = setup()
    generator = FixedJSONGenerator("not a valid answer")
    negatives = [c for c in cases if c.expected_type == "no_match"]
    report = evaluate_baseline(SingleAgentRAG(retriever, generator), negatives)
    assert report["no_match_zero_call_rate"] == 1.0
    assert report["model_invocations"] == 0
    assert report["generation_errors"] == 0
    assert all(c["status"] == "abstained" for c in report["per_case"])
    assert report["red_team_forbidden_phrase_leak_case_rate"] is None


def test_fixture_reference_product_cannot_be_ineligible(tmp_path):
    retriever, _, _ = setup()
    suite = json.loads(SUITE.read_text())
    suite["red_team_cases"][6]["expected_product_ids"] = ["SYN-R09"]
    bad = tmp_path / "modified-suite.json"
    bad.write_text(json.dumps(suite))
    with pytest.raises(ValueError, match="not eligible"):
        load_baseline_suite(bad, repo_root=ROOT, retriever=retriever)


def test_manifest_catalog_hash_mismatch_rejected():
    wrong_retriever = CatalogRetriever.from_json(ROOT / "data/sample_catalog/catalog.json")
    with pytest.raises(ValueError, match="does not match"):
        load_baseline_suite(SUITE, repo_root=ROOT, retriever=wrong_retriever)


def test_untrusted_benchmark_paths_cannot_escape_repo(tmp_path):
    retriever, _, _ = setup()
    suite = json.loads(SUITE.read_text())
    suite["retrieval_suite_path"] = "../../outside.json"
    bad = tmp_path / "bad-path.json"
    bad.write_text(json.dumps(suite))
    with pytest.raises(ValueError, match="inside the repository"):
        load_baseline_suite(bad, repo_root=ROOT, retriever=retriever)


def test_duplicate_case_id_rejected(tmp_path):
    retriever, _, _ = setup()
    suite = json.loads(SUITE.read_text())
    suite["red_team_cases"][1]["case_id"] = suite["red_team_cases"][0]["case_id"]
    bad = tmp_path / "duplicate.json"
    bad.write_text(json.dumps(suite))
    with pytest.raises(ValidationError, match="duplicate benchmark case"):
        load_baseline_suite(bad, repo_root=ROOT, retriever=retriever)


def test_structural_citation_is_not_semantic_support():
    retriever, cases, _ = setup()
    case = next(x for x in cases if x.case_id == "Q11")
    generator = FixedJSONGenerator(json.dumps({
        "recommendations": [{
            "product_id": "SYN-R12",
            "reason": "This fictional product is fragrance-free and travel-size.",
            "evidence_ids": ["E-SYN-R12-T"],
        }],
        "no_suitable_match": False,
    }))
    report = evaluate_baseline(SingleAgentRAG(retriever, generator), [case], top_k=5)
    row = report["per_case"][0]
    assert row["status"] == "accepted"
    assert row["structural_citation_valid"] is True
    assert row["required_attribute_field_coverage"] == 0.0
    assert report["unsupported_claim_rate"] is None
    assert report["semantic_citation_groundedness"] is None


def test_red_team_forbidden_phrase_is_detected_but_not_called_entailment():
    retriever, cases, _ = setup()
    case = next(x for x in cases if x.case_id == "RT08")
    generator = FixedJSONGenerator(json.dumps({
        "recommendations": [{
            "product_id": "SYN-R11",
            "reason": "This balm is clinically proven to cure eczema.",
            "evidence_ids": ["E-SYN-R11-T"],
        }],
        "no_suitable_match": False,
    }))
    report = evaluate_baseline(SingleAgentRAG(retriever, generator), [case], top_k=5)
    assert report["red_team_success_rate"] == 0
    assert "clinically proven" in report["per_case"][0]["forbidden_phrase_leaks"]
    assert report["red_team_forbidden_phrase_leak_case_rate"] == 1.0
    assert report["unsupported_claim_rate"] is None


def test_invalid_json_records_failure_without_masking_it():
    retriever, cases, _ = setup()
    case = next(x for x in cases if x.case_id == "Q11")
    generator = FixedJSONGenerator("NOT_JSON")
    report = evaluate_baseline(SingleAgentRAG(retriever, generator), [case])
    assert report["generation_errors"] == 1
    assert report["normal_success_rate"] == 0.0
    assert report["per_case"][0]["error_type"] == "ValueError"


def test_latency_quantiles_and_empty_run():
    assert _quantile([], 0.5) is None
    assert _quantile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert _quantile([1.0, 2.0, 3.0], 0.95) == 2.9
    with pytest.raises(ValueError, match="unique and nonempty"):
        evaluate_baseline(None, [])
