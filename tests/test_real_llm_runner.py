"""Offline checks for the real-model runner's contracts (no model download)."""

import json
from pathlib import Path

import pytest

from recsys.baseline import SingleAgentRAG
from recsys.baseline.generators import FixedJSONGenerator
from recsys.baseline.local_qwen import LocalQwenGenerator, MODEL_REVISION
from recsys.evaluation.run_single_agent import evaluate_single_agent, load_smoke_cases
from recsys.retrieval import CatalogRetriever


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "data/llm_smoke/cases.json"
CATALOG = ROOT / "data/sample_catalog/catalog.json"


def test_smoke_suite_is_frozen_and_has_supported_outcomes():
    suite, digest = load_smoke_cases(SUITE)
    assert len(digest) == 64
    assert len(suite["cases"]) == 4
    assert [c["expected_type"] for c in suite["cases"]].count("recommendation") == 2
    assert [c["expected_type"] for c in suite["cases"]].count("no_match") == 2


def test_evaluator_counts_accepted_output_and_abstention_without_second_call():
    suite, _ = load_smoke_cases(SUITE)
    accepted = {
        "recommendations": [{
            "product_id": "SYN-001",
            "reason": "Synthetic listing describes a fragrance-free formula.",
            "evidence_ids": ["E-SYN-001"],
        }],
        "no_suitable_match": False,
    }
    generator = FixedJSONGenerator(json.dumps(accepted))
    rag = SingleAgentRAG(CatalogRetriever.from_json(CATALOG), generator)
    report = evaluate_single_agent(rag, [suite["cases"][0], suite["cases"][2]])
    assert report["positive_accept_rate"] == 1.0
    assert report["negative_no_match_rate"] == 1.0
    assert report["model_invocations"] == 1
    assert report["generation_failures"] == 0
    assert report["per_case"][0]["recommended_product_ids"] == ["SYN-001"]
    assert report["per_case"][1]["model_invocations"] == 0


def test_evaluator_records_invalid_model_response_instead_of_fabricating_success():
    suite, _ = load_smoke_cases(SUITE)
    generator = FixedJSONGenerator("not even json")
    rag = SingleAgentRAG(CatalogRetriever.from_json(CATALOG), generator)
    report = evaluate_single_agent(rag, [suite["cases"][0]])
    assert report["positive_accept_rate"] == 0.0
    assert report["generation_failures"] == 1
    assert report["model_invocations"] == 1
    assert report["per_case"][0]["status"] == "error"
    assert report["per_case"][0]["error_type"] == "ValueError"


def test_no_match_cases_require_zero_model_invocations():
    suite, _ = load_smoke_cases(SUITE)
    generator = FixedJSONGenerator("no JSON needed")
    rag = SingleAgentRAG(CatalogRetriever.from_json(CATALOG), generator)
    report = evaluate_single_agent(rag, suite["cases"][2:])
    assert report["negative_no_match_rate"] == 1.0
    assert generator.call_count == 0
    assert all(case["status"] == "abstained" for case in report["per_case"])


def test_malformed_smoke_suite_rejected(tmp_path):
    suite, _ = load_smoke_cases(SUITE)
    suite["cases"][1]["case_id"] = suite["cases"][0]["case_id"]
    file = tmp_path / "bad.json"
    file.write_text(json.dumps(suite))
    with pytest.raises(ValueError, match="duplicate smoke case"):
        load_smoke_cases(file)


def test_model_sha_is_full_length_and_invalid_revision_fails_before_download():
    assert len(MODEL_REVISION) == 40
    with pytest.raises(ValueError, match="revision"):
        LocalQwenGenerator(revision="main")
    with pytest.raises(ValueError, match="max_new_tokens"):
        LocalQwenGenerator(max_new_tokens=0)
