"""Stage 3.3: conflict-aware typed-source checks (fictional documents only)."""

import json
from pathlib import Path

import pytest

from recsys.baseline.generators import FixedJSONGenerator
from recsys.multiagent import (
    CriticVerificationAgent, MultiAgentOrchestrator, PreferenceRetrievalAgent,
    RecommendationAgent, TypedSourceGuard,
)
from recsys.retrieval import CatalogRetriever
from recsys.schemas import UserPreferences


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/source_evidence_demo/catalog.json"
SOURCES = ROOT / "data/source_evidence_demo/source_documents.json"
HISTORICAL = ROOT / "data/retrieval_eval/catalog.json"


def guard():
    return TypedSourceGuard(catalog_path=CATALOG, source_path=SOURCES)


def answer(product_id, eid, reason="Only information supported by a fictional source is described."):
    return json.dumps({
        "recommendations": [{"product_id": product_id,
                             "reason": reason, "evidence_ids": [eid]}],
        "no_suitable_match": False,
    })


def make_workflow(source, *, preflight=False, max_revisions=0):
    model = FixedJSONGenerator(source)
    flow = MultiAgentOrchestrator(
        PreferenceRetrievalAgent(CatalogRetriever.from_json(CATALOG)),
        RecommendationAgent(model),
        CriticVerificationAgent(source_guard=guard()),
        max_revisions=max_revisions, evidence_preflight=preflight,
        format_policy="safe_json_normalization",
    )
    return flow, model


def prefs(text, category, attribute):
    return UserPreferences(request=text, required_category=category,
                           required_attributes={attribute})


def test_consistent_required_fields_with_active_citations_are_approvable():
    flow, model = make_workflow(answer("SYN-E01", "E-SOURCE-001"))
    result = flow.run(prefs("Cloud Calm Facial Cream", "moisturizer", "fragrance_free"), top_k=4)
    assert result.status == "approved"
    assert result.generator_calls == 1
    assert model.call_count == 1
    assert result.attempts[0].verdict.semantic_entailment_verified is False


def test_conflicting_sources_are_rejected_and_preflight_saves_a_call():
    query = prefs("Dusk Barrier Balm", "balm", "fragrance_free")
    strict, generator = make_workflow(answer("SYN-E02", "E-SOURCE-004"))
    rejected = strict.run(query)
    assert rejected.status == "rejected"
    assert "source_conflict" in {i.code for i in rejected.attempts[0].verdict.issues}
    assert generator.call_count == 1

    guarded, model = make_workflow(answer("SYN-E02", "E-SOURCE-004"), preflight=True)
    result = guarded.run(query)
    assert result.status == "insufficient_evidence"
    assert result.generator_calls == 0 and model.call_count == 0


def test_stale_attribute_after_revision_is_rejected():
    query = prefs("Daylight Mineral Sun Cream", "sunscreen", "mineral")
    flow, _ = make_workflow(answer("SYN-E03", "E-SOURCE-008"))
    result = flow.run(query)
    assert result.status == "rejected"
    codes = {i.code for i in result.attempts[0].verdict.issues}
    assert "source_stale_metadata" in codes
    assert "superseded_evidence" in codes


def test_citation_to_old_source_is_rejected_even_when_other_attribute_is_consistent():
    flow, _ = make_workflow(answer("SYN-E03", "E-SOURCE-008"))
    result = flow.run(UserPreferences(
        request="Daylight Sun Cream", required_category="sunscreen",
        required_attributes={"fragrance_free"},
    ), top_k=4)
    assert result.status == "rejected"
    assert "superseded_evidence" in {i.code for i in result.attempts[0].verdict.issues}


def test_missing_fragrance_declaration_is_rejected_despite_metadata_true():
    query = prefs("Meadow Gentle Wash", "cleanser", "fragrance_free")
    flow, _ = make_workflow(answer("SYN-E04", "E-SOURCE-012"))
    result = flow.run(query)
    assert result.status == "rejected"
    codes = {i.code for i in result.attempts[0].verdict.issues}
    assert "source_claim_missing" in codes
    assert "required_attribute_uncited" in codes


def test_claim_level_verifier_checks_explicit_typed_claim_and_active_source():
    index = guard()
    assert index.verify_typed_claim(
        product_id="SYN-E01", field="attributes.fragrance_free",
        asserted_value=True, evidence_ids={"E-SOURCE-001"},
    )
    assert not index.verify_typed_claim(
        product_id="SYN-E01", field="attributes.fragrance_free",
        asserted_value=False, evidence_ids={"E-SOURCE-001"},
    )
    assert not index.verify_typed_claim(
        product_id="SYN-E02", field="attributes.fragrance_free",
        asserted_value=True, evidence_ids={"E-SOURCE-004"},
    )
    assert not index.verify_typed_claim(
        product_id="SYN-E03", field="attributes.mineral",
        asserted_value=True, evidence_ids={"E-SOURCE-008"},
    )
    assert not index.verify_typed_claim(
        product_id="SYN-E04", field="attributes.fragrance_free",
        asserted_value=True, evidence_ids={"E-SOURCE-012"},
    )
    assert index.verify_typed_claim(
        product_id="SYN-E01", field="price_usd",
        asserted_value="21.00", evidence_ids={"E-SOURCE-003"},
    )
    assert not index.verify_typed_claim(
        product_id="SYN-E01", field="price_usd",
        asserted_value="29.00", evidence_ids={"E-SOURCE-003"},
    )


def test_source_snapshot_binding_prevents_wrong_catalog_reuse_before_generation():
    bad = MultiAgentOrchestrator(
        PreferenceRetrievalAgent(CatalogRetriever.from_json(HISTORICAL)),
        RecommendationAgent(FixedJSONGenerator("not called")),
        CriticVerificationAgent(source_guard=guard()),
        max_revisions=0, evidence_preflight=True,
    )
    with pytest.raises(ValueError, match="catalog SHA-256"):
        bad.run(UserPreferences(request="face moisturizer"), top_k=3)


def test_old_baseline_without_guard_still_uses_original_critic_behavior():
    query = prefs("Dusk Barrier Balm", "balm", "fragrance_free")
    model = FixedJSONGenerator(answer("SYN-E02", "E-SOURCE-004"))
    flow = MultiAgentOrchestrator(
        PreferenceRetrievalAgent(CatalogRetriever.from_json(CATALOG)),
        RecommendationAgent(model), CriticVerificationAgent(), max_revisions=0,
    )
    result = flow.run(query, top_k=4)
    assert result.status == "approved"  # Illustrates the OLD guard's blind spot.
