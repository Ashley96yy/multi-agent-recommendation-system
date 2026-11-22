"""Frozen, synthetic retrieval benchmark with clear denominators and provenance."""

import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from recsys.retrieval import CatalogRetriever
from recsys.schemas import UserPreferences


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1)
    preferences: UserPreferences
    relevant_product_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def check_ground_truth(self) -> "EvaluationCase":
        if len(set(self.relevant_product_ids)) != len(self.relevant_product_ids):
            raise ValueError("duplicate relevant product ID")
        return self


class NegativeCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1)
    preferences: UserPreferences
    expected_reason: str = "no_eligible_products"


class EvaluationSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    suite_id: str = Field(min_length=1)
    description: str
    cases: list[EvaluationCase] = Field(min_length=1)
    no_match_cases: list[NegativeCase] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_case_ids(self) -> "EvaluationSuite":
        ids = [c.case_id for c in [*self.cases, *self.no_match_cases]]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate benchmark case IDs")
        return self


def retrieval_metrics(returned_ids: list[str], relevant_ids: list[str], k: int) -> dict[str, float]:
    """Binary relevance; Recall@K denominator is total relevant documents.

    MRR@K only counts the first relevant retrieved result; NDCG@K is normalized
    by ideal ordering with min(k, number of relevant documents) hits.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not relevant_ids or len(set(relevant_ids)) != len(relevant_ids):
        raise ValueError("relevant IDs must be unique and nonempty")
    if len(set(returned_ids)) != len(returned_ids):
        raise ValueError("retrieved IDs must be unique")
    relevant = set(relevant_ids)
    hits = [i for i, product_id in enumerate(returned_ids[:k], start=1) if product_id in relevant]
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    dcg = sum(1 / math.log2(i + 1) for i in hits)
    return {
        "recall": len(hits) / len(relevant),
        "hit_rate": float(bool(hits)),
        "mrr": (1 / hits[0]) if hits else 0.0,
        "ndcg": dcg / ideal,
    }


def evaluate_retriever(
    retriever: CatalogRetriever,
    suite: EvaluationSuite,
    *,
    ks: tuple[int, ...] = (1, 3, 5),
    verify_ground_truth: bool = True,
) -> dict[str, Any]:
    """Evaluate fixed cases, track per-case results and negatives separately."""
    if not ks or len(set(ks)) != len(ks) or any(not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError("ks must be unique positive integers")
    known_ids = {p.product_id for p in retriever.catalog.products}
    if verify_ground_truth:
        for case in suite.cases:
            if not set(case.relevant_product_ids) <= known_ids:
                raise ValueError(f"{case.case_id}: ground truth references unknown product")
            for relevant in case.relevant_product_ids:
                product = next(p for p in retriever.catalog.products if p.product_id == relevant)
                reason = retriever._filter_reason(product, case.preferences)
                if reason is not None:
                    raise ValueError(f"{case.case_id}: labeled relevant item {relevant} is ineligible: {reason}")
    per_case = []
    max_k = max(ks)
    aggregate = {f"{m}@{k}": [] for k in ks for m in ("recall", "hit_rate", "mrr", "ndcg")}
    for case in suite.cases:
        response = retriever.retrieve(case.preferences, top_k=max_k)
        retrieved = [candidate.product.product_id for candidate in response.candidates]
        measurements = {}
        for k in ks:
            metrics = retrieval_metrics(retrieved, case.relevant_product_ids, k)
            for metric, value in metrics.items():
                key = f"{metric}@{k}"
                aggregate[key].append(value)
                measurements[key] = value
        per_case.append({
            "case_id": case.case_id,
            "relevant_product_ids": case.relevant_product_ids,
            "retrieved_product_ids": retrieved,
            "metrics": measurements,
            "no_match_reason": response.no_match_reason,
        })
    negative_results = []
    for case in suite.no_match_cases:
        response = retriever.retrieve(case.preferences, top_k=max_k)
        negative_results.append({
            "case_id": case.case_id,
            "expected_reason": case.expected_reason,
            "actual_reason": response.no_match_reason,
            "returned_product_ids": [x.product.product_id for x in response.candidates],
            "passed": not response.candidates and response.no_match_reason == case.expected_reason,
        })
    return {
        "suite_id": suite.suite_id,
        "embedder": retriever.embedder.name,
        "catalog_sha256": retriever.catalog_sha256,
        "case_count": len(suite.cases),
        "negative_case_count": len(suite.no_match_cases),
        "ks": list(ks),
        "metrics": {key: sum(values) / len(values) for key, values in aggregate.items()},
        "negative_pass_rate": (
            sum(row["passed"] for row in negative_results) / len(negative_results)
            if negative_results else None
        ),
        "per_case": per_case,
        "negative_cases": negative_results,
    }


def load_suite(path: str | Path) -> tuple[EvaluationSuite, str]:
    raw = Path(path).read_bytes()
    return EvaluationSuite.model_validate_json(raw), hashlib.sha256(raw).hexdigest()
