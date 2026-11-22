"""Optional Stage 3.2 policies preserve the historical strict default.

All tests are deterministic with fixed JSON outputs, NOT real Qwen evaluation.
"""

import json
from pathlib import Path

import pytest

from recsys.baseline.generators import FixedJSONGenerator
from recsys.evaluation.format_ablation import normalize_output
from recsys.multiagent import (
    CriticVerificationAgent, MultiAgentOrchestrator,
    PreferenceRetrievalAgent, RecommendationAgent,
)
from recsys.multiagent.output_policy import normalize_json_response
from recsys.retrieval import CatalogRetriever
from recsys.schemas import UserPreferences


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data/source_evidence_demo/catalog.json"
OLD_CATALOG = ROOT / "data/retrieval_eval/catalog.json"


def valid_answer(product="SYN-E01", evidence="E-SOURCE-001"):
    return json.dumps({
        "recommendations": [{
            "product_id": product,
            "reason": "A fictional product description is linked.",
            "evidence_ids": [evidence],
        }],
        "no_suitable_match": False,
    })


def workflow(path, output, *, revisions=0, mode="strict", preflight=False):
    retriever = CatalogRetriever.from_json(path)
    agent = FixedJSONGenerator(output)
    controller = MultiAgentOrchestrator(
        PreferenceRetrievalAgent(retriever), RecommendationAgent(agent),
        CriticVerificationAgent(require_attribute_citations=True),
        max_revisions=revisions, format_policy=mode, evidence_preflight=preflight,
    )
    return controller, agent


def test_strict_historical_default_unchanged_and_safe_opt_in_unwraps_whole_fence():
    raw = "```json\n" + valid_answer() + "\n```"
    pref = UserPreferences(request="fragrance-free cream",
                           required_category="moisturizer",
                           required_attributes={"fragrance_free"})
    old, calls = workflow(SOURCES, raw)
    strict = old.run(pref)
    assert strict.status == "rejected" and calls.call_count == 1
    assert not strict.attempts[0].parse_valid
    assert strict.attempts[0].format_actions == []

    upgraded, calls2 = workflow(SOURCES, raw, mode="safe_json_normalization")
    repaired = upgraded.run(pref)
    assert repaired.status == "approved"
    assert repaired.generator_calls == 1 and calls2.call_count == 1
    assert repaired.attempts[0].format_actions == ["strip_whole_json_fence"]
    assert repaired.attempts[0].verdict.semantic_entailment_verified is False


def test_missing_flag_is_added_only_when_recommendations_nonempty():
    answer = json.loads(valid_answer())
    answer.pop("no_suitable_match")
    policy, generator = workflow(SOURCES, json.dumps(answer),
                                 mode="safe_json_normalization")
    result = policy.run(UserPreferences(request="cream", required_category="moisturizer",
                                        required_attributes={"fragrance_free"}))
    assert result.status == "approved" and generator.call_count == 1
    assert result.attempts[0].format_actions == ["set_missing_no_suitable_match_false"]
    empty, actions = normalize_json_response('{"recommendations":[]}', "safe_json_normalization")
    assert empty == '{"recommendations":[]}' and not actions


def test_safe_normalization_never_changes_reason_or_unsafe_claim():
    unsafe = json.loads(valid_answer())
    unsafe["recommendations"][0]["reason"] = "Clinically proven to cure eczema."
    raw = json.dumps(unsafe)
    normalized, actions = normalize_json_response(raw, "safe_json_normalization")
    assert json.loads(normalized) == unsafe and actions == []
    controller, _ = workflow(SOURCES, raw, mode="safe_json_normalization")
    result = controller.run(UserPreferences(request="moisturizer", required_category="moisturizer"))
    assert result.status == "rejected"
    assert "prohibited_claim_phrase" in {
        issue.code for issue in result.attempts[0].verdict.issues
    }


def test_safe_normalization_cannot_launder_hallucinated_product_or_citation():
    raw = valid_answer(product="SYN-FAKE", evidence="FAKE")
    controller, _ = workflow(SOURCES, raw, mode="safe_json_normalization")
    result = controller.run(UserPreferences(request="moisturizer", required_category="moisturizer"))
    assert result.status == "rejected"
    assert result.attempts[0].verdict.issues[0].code == "product_not_retrieved"


def test_output_policy_matches_frozen_format_ablation_on_exact_cases():
    good = valid_answer()
    variants = [
        good, "```json\n" + good + "\n```",
        json.dumps({"recommendations": json.loads(good)["recommendations"]}),
        "non-json ```json\n"+good+"\n``` trailing",
        '{"recommendations":[]}', '{"recommendations":[],"no_suitable_match":true}',
    ]
    for raw in variants:
        expected = normalize_output(raw, "fences_and_missing_flag")
        actual = normalize_json_response(raw, "safe_json_normalization")
        assert expected == actual


def test_preflight_skips_unsatisfiable_source_fields_before_model_call():
    retriever = CatalogRetriever.from_json(OLD_CATALOG)
    controller = MultiAgentOrchestrator(
        PreferenceRetrievalAgent(retriever),
        RecommendationAgent(FixedJSONGenerator("not used")),
        CriticVerificationAgent(), evidence_preflight=True,
    )
    prefs = UserPreferences(request="face moisturizer", required_category="moisturizer",
                            required_attributes={"fragrance_free"})
    result = controller.run(prefs)
    assert result.status == "insufficient_evidence"
    assert result.retrieval.candidates
    assert result.generator_calls == 0
    assert result.answer is None
    assert result.attempts == []
    assert "required attribute fields" in result.note


def test_preflight_works_on_authored_source_and_filters_uncited_candidates():
    controller, generator = workflow(SOURCES, valid_answer(), preflight=True)
    pref = UserPreferences(request="cream", required_category="moisturizer",
                           required_attributes={"fragrance_free"})
    result = controller.run(pref)
    assert result.status == "approved" and generator.call_count == 1

    all_cat = CatalogRetriever.from_json(SOURCES)
    controller2 = MultiAgentOrchestrator(
        PreferenceRetrievalAgent(all_cat),
        RecommendationAgent(FixedJSONGenerator(valid_answer())),
        CriticVerificationAgent(),
        evidence_preflight=True,
    )
    outcome = controller2.run(UserPreferences(
        request="fragrance free", required_attributes={"fragrance_free"},
    ), top_k=4)
    assert "SYN-E04" not in [p.product.product_id for p in outcome.retrieval.candidates]


def test_preflight_reports_missing_specific_evidence_without_generating():
    controller, fake = workflow(SOURCES, valid_answer(), preflight=True)
    result = controller.run(UserPreferences(
        request="gentle face cleanser", required_category="cleanser",
        required_attributes={"fragrance_free"},
    ))
    assert result.status == "insufficient_evidence"
    assert result.generator_calls == 0
    assert fake.call_count == 0


def test_preflight_and_output_policy_validate_config():
    retrieval = PreferenceRetrievalAgent(CatalogRetriever.from_json(SOURCES))
    recommendation = RecommendationAgent(FixedJSONGenerator(valid_answer()))
    critic = CriticVerificationAgent()
    with pytest.raises(ValueError, match="format policy"):
        MultiAgentOrchestrator(retrieval, recommendation, critic, format_policy="auto-rewrite")
    with pytest.raises(ValueError, match="must be bool"):
        MultiAgentOrchestrator(retrieval, recommendation, critic, evidence_preflight=1)
