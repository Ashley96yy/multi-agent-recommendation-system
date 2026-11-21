"""Validated contracts shared by the future retrieval and agent modules."""

from recsys.schemas.retrieval import RetrievalResult, RetrievedProduct
from recsys.schemas.contracts import (
    Catalog,
    EvidenceReference,
    Product,
    UserPreferences,
)

__all__ = ["Catalog", "EvidenceReference", "Product", "UserPreferences", "RetrievedProduct", "RetrievalResult"]
