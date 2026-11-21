"""Metadata-first candidate filtering followed by embedding cosine search."""

from collections import Counter, defaultdict
from collections.abc import Sequence
from decimal import Decimal
from typing import Protocol

import numpy as np

from recsys.retrieval.catalog import load_catalog, searchable_text
from recsys.retrieval.embeddings import HashingEmbedder, normalized_vectors
from recsys.schemas.contracts import Catalog, EvidenceReference, Product, UserPreferences
from recsys.schemas.retrieval import RetrievalResult, RetrievedProduct


class EmbeddingBackend(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class CatalogRetriever:
    """In-memory exact vector search, suitable for a small demo corpus.

    For larger datasets, replace the vector-search backend without changing
    preference filtering or the output/evidence contracts.
    """

    def __init__(self, catalog: Catalog, catalog_sha256: str, embedder: EmbeddingBackend):
        if len(catalog_sha256) != 64:
            raise ValueError("catalog_sha256 must contain 64 hex characters")
        self.catalog = catalog
        self.catalog_sha256 = catalog_sha256
        self.embedder = embedder
        self._vectors = normalized_vectors(
            embedder.embed([searchable_text(p) for p in catalog.products]),
            len(catalog.products),
        )
        self._evidence: dict[str, list[EvidenceReference]] = defaultdict(list)
        for item in catalog.evidence:
            self._evidence[item.product_id].append(item)

    @classmethod
    def from_json(cls, path, embedder: EmbeddingBackend | None = None) -> "CatalogRetriever":
        catalog, sha = load_catalog(path)
        return cls(catalog, sha, embedder if embedder is not None else HashingEmbedder())

    @staticmethod
    def _filter_reason(product: Product, prefs: UserPreferences) -> str | None:
        if product.product_id in prefs.excluded_product_ids:
            return "excluded_product_id"
        if product.brand.casefold() in {s.strip().casefold() for s in prefs.excluded_brands}:
            return "excluded_brand"
        if prefs.required_category is not None and (
            product.category.casefold() != prefs.required_category.strip().casefold()
        ):
            return "category_mismatch"
        if prefs.max_budget_usd is not None:
            if product.price_usd is None:
                return "unknown_price"
            if product.price_usd > prefs.max_budget_usd:
                return "over_budget"
        if any(product.attributes.get(key) is not True for key in prefs.required_attributes):
            return "missing_required_attribute"
        return None

    def retrieve(
        self,
        preferences: UserPreferences,
        *,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> RetrievalResult:
        if not isinstance(preferences, UserPreferences):
            raise TypeError("retrieve requires a validated UserPreferences object")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if not np.isfinite(min_similarity) or not -1.0 <= min_similarity < 1.0:
            raise ValueError("min_similarity must be finite and in [-1, 1)")

        excluded = Counter()
        eligible_indices: list[int] = []
        for i, product in enumerate(self.catalog.products):
            reason = self._filter_reason(product, preferences)
            if reason:
                excluded[reason] += 1
            else:
                eligible_indices.append(i)

        results: list[RetrievedProduct] = []
        if eligible_indices:
            query_vectors = normalized_vectors(self.embedder.embed([preferences.request]), 1)
            if query_vectors.shape[1] != self._vectors.shape[1]:
                raise ValueError("query embedding dimension differs from indexed embeddings")
            query = query_vectors[0]
            scores = self._vectors[eligible_indices] @ query
            ranked = sorted(
                ((float(score), i) for score, i in zip(scores, eligible_indices)
                 if float(score) > min_similarity),
                key=lambda pair: (-pair[0], self.catalog.products[pair[1]].product_id),
            )[:top_k]
            for score, idx in ranked:
                product = self.catalog.products[idx]
                results.append(RetrievedProduct(
                    product=product, similarity=max(-1.0, min(1.0, score)),
                    evidence=list(self._evidence[product.product_id]),
                ))

        return RetrievalResult(
            query=preferences.request,
            embedder_name=self.embedder.name,
            catalog_sha256=self.catalog_sha256,
            indexed_products=len(self.catalog.products),
            eligible_products=len(eligible_indices),
            filtered_by_reason=dict(excluded),
            candidates=results,
            no_match_reason=None if results else (
                "no_eligible_products" if not eligible_indices else "no_vector_match"
            ),
        )
