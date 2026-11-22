"""Frozen 15+10 single-agent benchmark contracts and ground-truth checks."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from recsys.evaluation.retrieval import load_suite
from recsys.retrieval.retriever import CatalogRetriever
from recsys.schemas import UserPreferences


class BaselineCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1)
    group: Literal["normal", "red_team"]
    attack_type: str | None = None
    preferences: UserPreferences
    expected_type: Literal["recommendation", "no_match"]
    expected_product_ids: list[str]
    forbidden_phrases: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_expectations(self) -> "BaselineCase":
        if self.expected_type == "recommendation" and not self.expected_product_ids:
            raise ValueError("recommendation case must have expected IDs")
        if self.expected_type == "no_match" and self.expected_product_ids:
            raise ValueError("no-match case cannot have expected IDs")
        if len(self.expected_product_ids) != len(set(self.expected_product_ids)):
            raise ValueError("duplicate expected product ID")
        if self.group == "red_team" and not self.attack_type:
            raise ValueError("red-team case requires attack type")
        if any(not term.strip() for term in self.forbidden_phrases):
            raise ValueError("forbidden phrases must not be blank")
        return self


class BaselineSuiteManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]
    suite_id: str = Field(min_length=1)
    description: str
    catalog_path: str
    retrieval_suite_path: str
    normal_case_ids: list[str] = Field(min_length=1)
    red_team_cases: list[BaselineCase] = Field(min_length=1)

    @model_validator(mode="after")
    def check_case_ids(self) -> "BaselineSuiteManifest":
        if len(self.normal_case_ids) != 15 or len(self.red_team_cases) != 10:
            raise ValueError("expected exactly 15 normal and 10 red-team cases")
        ids = [*self.normal_case_ids, *(c.case_id for c in self.red_team_cases)]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate benchmark case ID")
        if any(c.group != "red_team" for c in self.red_team_cases):
            raise ValueError("red-team cases must have group red_team")
        return self


def load_baseline_suite(
    manifest_path: str | Path,
    *,
    repo_root: str | Path,
    retriever: CatalogRetriever,
) -> tuple[list[BaselineCase], dict[str, str]]:
    """Load a fixed suite and prevent accidental changes to labeled relevance.

    Paths are relative to the trusted repository root, not to model output.
    SHA-256 fingerprints are recorded for the manifest, source evaluation
    labels, and catalog, so future ablations can use the exact same inputs.
    """
    root = Path(repo_root).resolve()
    raw = Path(manifest_path).read_bytes()
    manifest = BaselineSuiteManifest.model_validate_json(raw)
    retrieval_path = (root / manifest.retrieval_suite_path).resolve()
    catalog_path = (root / manifest.catalog_path).resolve()
    if not retrieval_path.is_relative_to(root) or not catalog_path.is_relative_to(root):
        raise ValueError("benchmark paths must stay inside the repository")
    retrieval_suite, retrieval_sha = load_suite(retrieval_path)
    actual_catalog_sha = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    if actual_catalog_sha != retriever.catalog_sha256:
        raise ValueError("retriever catalog does not match baseline suite manifest")

    existing = {c.case_id: c for c in retrieval_suite.cases}
    if set(manifest.normal_case_ids) != set(existing):
        raise ValueError("normal case IDs must exactly match the frozen retrieval suite")
    cases = [
        BaselineCase(
            case_id=case_id,
            group="normal",
            preferences=existing[case_id].preferences,
            expected_type="recommendation",
            expected_product_ids=existing[case_id].relevant_product_ids,
        )
        for case_id in manifest.normal_case_ids
    ] + manifest.red_team_cases
    known = {p.product_id: p for p in retriever.catalog.products}
    for case in cases:
        if case.expected_type == "recommendation":
            for product_id in case.expected_product_ids:
                if product_id not in known:
                    raise ValueError(f"{case.case_id}: unknown ground truth ID {product_id}")
                reason = retriever._filter_reason(known[product_id], case.preferences)
                if reason is not None:
                    raise ValueError(f"{case.case_id}: expected product is not eligible: {reason}")
        else:
            if any(retriever._filter_reason(p, case.preferences) is None
                   for p in retriever.catalog.products):
                raise ValueError(f"{case.case_id}: no-match case has eligible products")
    return cases, {
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "retrieval_labels_sha256": retrieval_sha,
        "catalog_sha256": actual_catalog_sha,
    }
