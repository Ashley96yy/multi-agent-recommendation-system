"""Stage 2.0: offline tests for one-pass retrieval-grounded generator wiring."""

import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from recsys.baseline import SingleAgentRAG, build_context
from recsys.baseline.contracts import SingleAgentAnswer
from recsys.baseline.generators import FixedJSONGenerator, OpenAICompatibleGenerator
from recsys.retrieval import CatalogRetriever
from recsys.schemas import UserPreferences


CATALOG = Path(__file__).resolve().parents[1] / "data/sample_catalog/catalog.json"


def make_rag(payload: dict | str):
    generator = FixedJSONGenerator(json.dumps(payload) if isinstance(payload, dict) else payload)
    return SingleAgentRAG(CatalogRetriever.from_json(CATALOG), generator), generator


def prefs(**kwargs):
    return UserPreferences(request="fragrance free moisturizer", **kwargs)


def recommendation(pid="SYN-001", evidence_ids=None):
    return {
        "recommendations": [{
            "product_id": pid,
            "reason": "Catalog lists this as fragrance-free.",
            "evidence_ids": evidence_ids if evidence_ids is not None else ["E-SYN-001"],
        }],
        "no_suitable_match": False,
    }


def test_one_pass_produces_bounded_recommendation_with_real_citations():
    rag, stub = make_rag(recommendation())
    answer, retrieval = rag.recommend(
        prefs(max_budget_usd=Decimal("30"), required_attributes={"fragrance_free"})
    )
    assert stub.call_count == 1
    assert answer.recommendations[0].product_id == "SYN-001"
    context = json.loads(build_context(retrieval))
    assert [x["product"]["product_id"] for x in context["products"]] == ["SYN-001"]
    assert all("source_id" in ref for x in context["products"] for ref in x["evidence"])


def test_no_candidates_skip_generator_and_abstain():
    rag, stub = make_rag("not-json")
    answer, result = rag.recommend(prefs(max_budget_usd=Decimal("1")))
    assert answer.no_suitable_match is True
    assert not answer.recommendations
    assert result.no_match_reason == "no_eligible_products"
    assert stub.call_count == 0


def test_llm_unknown_product_is_rejected():
    rag, _ = make_rag(recommendation(pid="SYN-999"))
    with pytest.raises(ValueError, match="outside retrieved candidates"):
        rag.recommend(prefs())


def test_llm_wrong_product_citation_is_rejected():
    rag, _ = make_rag(recommendation(evidence_ids=["E-SYN-003"]))
    with pytest.raises(ValueError, match="cross-product"):
        rag.recommend(prefs())


def test_llm_bogus_citation_is_rejected():
    rag, _ = make_rag(recommendation(evidence_ids=["E-NOT-A-REAL-SOURCE"]))
    with pytest.raises(ValueError, match="absent"):
        rag.recommend(prefs())


def test_invalid_json_and_unexpected_fields_fail_closed():
    rag, _ = make_rag("invalid JSON!")
    with pytest.raises(ValueError, match="invalid response schema"):
        rag.recommend(prefs())
    rag, _ = make_rag({**recommendation(), "extra": "not allowed"})
    with pytest.raises(ValueError, match="invalid response schema"):
        rag.recommend(prefs())


def test_no_suitable_match_cannot_conflict_with_recommendations():
    rag, _ = make_rag({**recommendation(), "no_suitable_match": True})
    with pytest.raises(ValueError, match="invalid response schema"):
        rag.recommend(prefs())


def test_blank_reason_and_duplicate_evidence_are_rejected():
    bad = recommendation()
    bad["recommendations"][0]["reason"] = "  "
    rag, _ = make_rag(bad)
    with pytest.raises(ValueError, match="invalid response schema"):
        rag.recommend(prefs())
    bad = recommendation()
    bad["recommendations"][0]["evidence_ids"] = ["E-SYN-001", "E-SYN-001"]
    rag, _ = make_rag(bad)
    with pytest.raises(ValueError, match="invalid response schema"):
        rag.recommend(prefs())


def test_generator_can_abstain_even_when_candidates_exist():
    rag, stub = make_rag({"recommendations": [], "no_suitable_match": True,
                          "note": "Insufficient evidence"})
    answer, _ = rag.recommend(prefs())
    assert answer.no_suitable_match is True
    assert stub.call_count == 1


def test_context_has_no_outside_catalog_data_and_supports_truncation():
    rag, _ = make_rag(recommendation())
    _, retrieval = rag.recommend(prefs())
    data = json.loads(build_context(retrieval, max_excerpt_chars=5))
    assert data["catalog_sha256"] == retrieval.catalog_sha256
    assert all(len(e["excerpt"]) <= 5 for item in data["products"] for e in item["evidence"])
    with pytest.raises(ValueError, match="positive"):
        build_context(retrieval, max_excerpt_chars=0)


def test_generator_rejects_untrusted_plaintext_http():
    with pytest.raises(ValueError, match="HTTPS"):
        OpenAICompatibleGenerator(base_url="http://some-remote-server.example/v1",
                                  model="test-model")
    client = OpenAICompatibleGenerator(base_url="http://127.0.0.1:8000/v1",
                                       model="test-model")
    assert client.base_url == "http://127.0.0.1:8000/v1"
