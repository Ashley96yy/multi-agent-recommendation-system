"""Stage 1.3: deterministic, offline catalog retrieval behavior."""

from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from recsys.retrieval import CatalogRetriever, HashingEmbedder
from recsys.retrieval.embeddings import normalized_vectors
from recsys.schemas import Catalog, UserPreferences, RetrievedProduct


DATA = Path(__file__).resolve().parents[1] / "data/sample_catalog/catalog.json"


def preferences(query="fragrance free moisturizer", **overrides):
    return UserPreferences(request=query, **overrides)


def test_retrieval_is_deterministic_and_source_linked():
    retriever = CatalogRetriever.from_json(DATA)
    p = preferences(max_budget_usd=Decimal("30"), required_attributes={"fragrance_free"})
    one = retriever.retrieve(p)
    two = retriever.retrieve(p)
    assert one == two
    assert one.embedder_name.startswith("lexical-hashing-v1")
    assert len(one.catalog_sha256) == 64
    assert [x.product.product_id for x in one.candidates] == ["SYN-001"]
    assert {x.evidence_id for x in one.candidates[0].evidence} == {"E-SYN-001", "E-SYN-002"}
    assert one.eligible_products == 1
    assert one.filtered_by_reason["unknown_price"] == 1
    assert one.filtered_by_reason["missing_required_attribute"] == 1


def test_missing_price_is_not_assumed_below_budget():
    result = CatalogRetriever.from_json(DATA).retrieve(
        preferences("daily facial cream", max_budget_usd=Decimal("30"))
    )
    assert result.filtered_by_reason["unknown_price"] == 1
    assert all(c.product.product_id != "SYN-003" for c in result.candidates)


def test_known_false_and_missing_attributes_are_both_ineligible():
    result = CatalogRetriever.from_json(DATA).retrieve(
        preferences(required_attributes={"fragrance_free"})
    )
    assert [x.product.product_id for x in result.candidates] == ["SYN-001"]


def test_category_brand_and_id_exclusions():
    retriever = CatalogRetriever.from_json(DATA)
    found = retriever.retrieve(preferences(required_category="MOISTURIZER",
                                             excluded_brands={"example botanics"},
                                             excluded_product_ids={"SYN-003"}))
    assert [c.product.product_id for c in found.candidates] == ["SYN-001"]
    assert found.filtered_by_reason["excluded_brand"] == 1
    assert found.filtered_by_reason["excluded_product_id"] == 1


def test_no_eligible_products_explains_failure():
    result = CatalogRetriever.from_json(DATA).retrieve(
        preferences(max_budget_usd=Decimal("1"))
    )
    assert not result.candidates
    assert result.no_match_reason == "no_eligible_products"
    assert result.eligible_products == 0


def test_no_vector_match_is_distinct_from_no_eligible_products():
    result = CatalogRetriever.from_json(DATA).retrieve(preferences("zzzyyyxxx"))
    assert result.eligible_products == 3
    assert result.no_match_reason == "no_vector_match"
    assert result.candidates == []


def test_topk_score_order_and_repeated_query_are_stable():
    retriever = CatalogRetriever.from_json(DATA, HashingEmbedder(dimensions=512))
    results = retriever.retrieve(preferences("moisturizer"), top_k=2)
    assert len(results.candidates) == 2
    assert [c.similarity for c in results.candidates] == sorted(
        (c.similarity for c in results.candidates), reverse=True
    )
    assert results == retriever.retrieve(preferences("moisturizer"), top_k=2)


def test_retrieval_requires_valid_inputs():
    retriever = CatalogRetriever.from_json(DATA)
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve(preferences(), top_k=0)
    with pytest.raises(ValueError, match="min_similarity"):
        retriever.retrieve(preferences(), min_similarity=float("nan"))
    with pytest.raises(TypeError):
        retriever.retrieve({"request": "moisturizer"})


def test_reject_invalid_embedding_matrices():
    with pytest.raises(ValueError, match="shape"):
        normalized_vectors(np.ones((2, 3)), 3)
    with pytest.raises(ValueError, match="non-finite"):
        normalized_vectors(np.array([[float("nan")]]), 1)


def test_retrieved_product_rejects_cross_product_evidence():
    cat = Catalog.model_validate_json(DATA.read_text())
    with pytest.raises(ValidationError, match="must belong"):
        RetrievedProduct(product=cat.products[1], similarity=0.1, evidence=[cat.evidence[0]])


def test_hashing_embedding_reproducible_without_network():
    left = HashingEmbedder().embed(["fragrance free", "moisturizer"])
    right = HashingEmbedder().embed(["fragrance free", "moisturizer"])
    assert np.array_equal(left, right)
    assert np.allclose(np.linalg.norm(left, axis=1), [1, 1])
