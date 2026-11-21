"""Catalog loading and field-safe searchable text."""

import hashlib
from pathlib import Path

from recsys.schemas.contracts import Catalog, Product


def load_catalog(path: str | Path) -> tuple[Catalog, str]:
    """Parse a catalog and return SHA-256 of its exact bytes for provenance."""
    raw = Path(path).read_bytes()
    catalog = Catalog.model_validate_json(raw)
    return catalog, hashlib.sha256(raw).hexdigest()


def searchable_text(product: Product) -> str:
    """Only index real field values; true attributes contribute positive keywords."""
    known_positive_attributes = [
        key.replace("_", " ") for key, value in sorted(product.attributes.items()) if value is True
    ]
    return " ".join(
        [product.title, product.brand, product.category, *known_positive_attributes]
    )
