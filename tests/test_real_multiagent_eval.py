"""Offline paired-run tests. No model weights are downloaded here."""

import json
from pathlib import Path

import pytest

from recsys.evaluation.baseline_suite import load_baseline_suite
from recsys.evaluation.run_multiagent_eval import (
    TracedGenerator,
    compare_to_original,
    evaluate_paired,
    no_revision_projection,
    score_case,
    summarize_arm,
)
from recsys.multiagent import (
    CriticVerificationAgent,
    MultiAgentOrchestrator,
    PreferenceRetrievalAgent,
    RecommendationAgent,
)
from recsys.retrieval import CatalogRetriever


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/retrieval_eval/catalog.json"
SUITE = ROOT / "data/baseline_eval/suite.json"


class FakeQwen:
    """Offline deterministic simulator, deliberately NOT a model-quality test."""

    def __init__(self, *, invalid_first=False, forbidden=False):
        self.call_count = 0
        self.last_prompt_tokens = 20
        self.last_generated_tokens = 12
        self.invalid_first = invalid_first
        self.forbidden = forbidden

    def generate(self, instructions, context):
        self.call_count += 1
        payload = json.loads(context)
        if self.invalid_first and "REVISION:" not in instructions:
            return "invalid json from simulated model"
        candidate = payload["products"][0]
        item = candidate["product"]
        evidence = candidate["evidence"][0]["evidence_id"]
        return json.dumps({
            "recommendations": [{
                "product_id": item["product_id"],
                "reason": ("clinically proven to cure eczema" if self.forbidden
                           else "Only a fictional title and price are described."),
                "evidence_ids": [evidence],
            }],
            "no_suitable_match": False,
        })


def setup():
    retriever = CatalogRetriever.from_json(CATALOG)
    cases, hashes = load_baseline_suite(SUITE, repo_root=ROOT, retriever=retriever)
    return retriever, cases, hashes


def test_every_case_reuses_identical_first_draft_when_revisions_allowed():
    retriever, cases, _ = setup()
    fake = FakeQwen(invalid_first=True)
    result = evaluate_paired(
        cases, retriever, fake, top_k=3, require_attribute_citations=False,
    )
    assert result["design"] == "paired_shared_first_draft"
    assert result["zero_revision"]["case_count"] == 25
    assert result["one_revision"]["case_count"] == 25
    assert result["zero_revision"]["approved_count"] == 0
    assert result["zero_revision"]["no_match_zero_call_count"] == 6
    assert result["one_revision"]["no_match_zero_call_count"] == 6
    assert result["one_revision"]["approved_count"] == 19
    assert result["physical_model_invocations"] == 38
    assert result["first_draft_model_invocations"] == 19
    assert result["revision_model_invocations"] == 19
    assert fake.call_count == 38
    for a, b in zip(result["zero_revision"]["per_case"],
                    result["one_revision"]["per_case"]):
        assert a["case_id"] == b["case_id"]
        if a["logical_generator_attempts"]:
            assert a["calls"][0] == b["calls"][0]
            assert b["logical_generator_attempts"] == 2
        else:
            assert b["logical_generator_attempts"] == 0
    assert result["unsupported_claim_rate"] is None
    assert result["semantic_citation_groundedness"] is None


def test_approved_first_draft_does_not_need_second_inference():
    retriever, cases, _ = setup()
    fake = FakeQwen()
    report = evaluate_paired(
        cases, retriever, fake, require_attribute_citations=False,
    )
    assert report["physical_model_invocations"] == 19
    assert report["revision_model_invocations"] == 0
    assert report["zero_revision"]["per_case"] == report["one_revision"]["per_case"]


def test_strict_attribute_citations_expose_known_evidence_gap():
    retriever, cases, _ = setup()
    fake = FakeQwen()
    report = evaluate_paired(
        cases, retriever, fake, require_attribute_citations=True,
    )
    assert report["one_revision"]["critic_issue_counts"].get("required_attribute_uncited", 0) > 0
    assert report["one_revision"]["rejected_count"] > 0
    assert any(
        row["revision_used"] and row["status"] == "rejected"
        for row in report["one_revision"]["per_case"]
    )


def test_prohibited_phrase_not_laundered_by_revision_or_format_validator():
    retriever, cases, _ = setup()
    fake = FakeQwen(forbidden=True)
    report = evaluate_paired(cases, retriever, fake, require_attribute_citations=False)
    bad = next(x for x in report["one_revision"]["per_case"] if x["case_id"] == "RT08")
    assert bad["status"] == "rejected"
    assert not bad["passed"]
    assert bad["critic_issues_by_attempt"] == [
        ["prohibited_claim_phrase", "prohibited_claim_phrase"],
        ["prohibited_claim_phrase", "prohibited_claim_phrase"],
    ]
    assert bad["semantic_entailment_verified"] is False
    assert fake.call_count <= 38


def test_projection_preserves_initial_first_attempt_without_extra_generator_call():
    retriever, cases, _ = setup()
    case = next(c for c in cases if c.case_id == "Q03")
    model = FakeQwen(invalid_first=True)
    trace = TracedGenerator(model)
    orchestrator = MultiAgentOrchestrator(
        PreferenceRetrievalAgent(retriever),
        RecommendationAgent(trace),
        CriticVerificationAgent(require_attribute_citations=False),
    )
    result = orchestrator.run(case.preferences, top_k=3)
    assert model.call_count == 2
    projection = no_revision_projection(result)
    assert projection.status == "rejected"
    assert projection.generator_calls == 1
    assert projection.revision_used is False
    assert len(projection.attempts) == 1
    assert len(trace.calls) == 2
    assert projection.attempts[0] == result.attempts[0]
    assert model.call_count == 2


def test_summary_keeps_schema_and_grounding_metrics_separate():
    retriever, cases, _ = setup()
    model = FakeQwen()
    result = evaluate_paired(cases, retriever, model, require_attribute_citations=False)
    for arm in (result["zero_revision"], result["one_revision"]):
        assert arm["semantic_citation_groundedness"] is None
        assert arm["unsupported_claim_rate"] is None
        assert arm["tokens"]["prompt_total"] > 0
        assert arm["generation_seconds"]["p95_generated_case"] is not None


def test_invalid_case_count_rejected():
    retriever, cases, _ = setup()
    with pytest.raises(ValueError, match="25"):
        evaluate_paired(cases[:24], retriever, FakeQwen())


def test_historical_comparison_requires_provenance_match():
    report = {
        "data_hashes": {"catalog_sha256": "a", "manifest_sha256": "b"},
        "zero_revision": {"per_case": [{"case_id": "Q01", "logical_generator_attempts": 1,
                                        "calls": [{"raw_output": "x"}]}]},
    }
    with pytest.raises(ValueError, match="hashes"):
        compare_to_original(report, {"data_hashes": {"catalog_sha256": "other"}})
