"""Stage 1.2 contract tests; use fictional products only."""

import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from recsys.schemas import Catalog, EvidenceReference, Product, UserPreferences


CATALOG = Path(__file__).resolve().parents[1] / "data" / "sample_catalog" / "catalog.json"


def product(**overrides):
    payload = {
        "product_id": "SYN-100",
        "title": "Test Lotion",
        "brand": "Test Brand",
        "category": "moisturizer",
        "price_usd": Decimal("9.99"),
        "attributes": {"fragrance_free": True},
    }
    payload.update(overrides)
    return payload


def evidence(**overrides):
    payload = {
        "evidence_id": "E-100",
        "product_id": "SYN-100",
        "source_id": "synthetic-source",
        "source_type": "synthetic_catalog",
        "supported_field": "attributes.fragrance_free",
        "excerpt": "Synthetic example: fragrance-free.",
    }
    payload.update(overrides)
    return payload


def test_sample_catalog_loads_and_preserves_unknowns():
    catalog = Catalog.model_validate_json(CATALOG.read_text(encoding="utf-8"))
    assert len(catalog.products) == 3
    assert len(catalog.evidence) == 3
    assert catalog.products[0].price_usd == Decimal("24.50")
    assert catalog.products[2].price_usd is None
    assert "fragrance_free" not in catalog.products[2].attributes


def test_products_reject_negative_prices_and_unknown_keys():
    with pytest.raises(ValidationError):
        Product.model_validate(product(price_usd=Decimal("-1")))
    with pytest.raises(ValidationError):
        Product.model_validate(product(invented_field="not allowed"))


def test_product_rejects_blank_identifiers():
    with pytest.raises(ValidationError):
        Product.model_validate(product(product_id="  "))


def test_preferences_are_explicit_and_validated():
    request = UserPreferences.model_validate({
        "request": "Fragrance-free lotion below $30",
        "max_budget_usd": Decimal("30.00"),
        "required_attributes": ["fragrance_free"],
        "excluded_product_ids": ["SYN-002"],
    })
    assert request.max_budget_usd == Decimal("30.00")
    assert "fragrance_free" in request.required_attributes
    with pytest.raises(ValidationError):
        UserPreferences.model_validate({"request": "  "})
    with pytest.raises(ValidationError):
        UserPreferences.model_validate({"request": "anything", "max_budget_usd": Decimal("-0.01")})


def test_catalog_rejects_duplicate_products_and_evidence_ids():
    with pytest.raises(ValidationError, match="duplicate product_id"):
        Catalog.model_validate({"products": [product(), product()]})
    with pytest.raises(ValidationError, match="duplicate evidence_id"):
        Catalog.model_validate({"products": [product()], "evidence": [evidence(), evidence()]})


def test_catalog_rejects_dangling_or_invalid_field_evidence():
    with pytest.raises(ValidationError, match="unknown product_id"):
        Catalog.model_validate({
            "products": [product()],
            "evidence": [evidence(product_id="SYN-404")],
        })
    with pytest.raises(ValidationError, match="unknown supported_field"):
        Catalog.model_validate({
            "products": [product()],
            "evidence": [evidence(supported_field="attributes.nonexistent")],
        })


def test_contract_roundtrip_json():
    original = Catalog.model_validate({"products": [product()], "evidence": [evidence()]})
    reconstructed = Catalog.model_validate(json.loads(original.model_dump_json()))
    assert reconstructed == original
