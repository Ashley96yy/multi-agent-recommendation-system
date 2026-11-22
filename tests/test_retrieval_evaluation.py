"""Offline evaluation tests; no pretrained model download in standard CI."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from recsys.evaluation.retrieval import evaluate_retriever, load_suite, retrieval_metrics
from recsys.evaluation.run_retrieval import run
from recsys.retrieval import CatalogRetriever, HashingEmbedder, MiniLMEmbedder


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/retrieval_eval/catalog.json"
SUITE = ROOT / "data/retrieval_eval/suite.json"


def test_hand_computed_ranking_metrics():
    scores = retrieval_metrics(["other", "A", "B"], ["A", "B"], 3)
    assert scores["recall"] == 1
    assert scores["hit_rate"] == 1
    assert scores["mrr"] == 0.5
    assert 0 < scores["ndcg"] < 1
    assert retrieval_metrics(["other"], ["A"], 1)["recall"] == 0
    assert retrieval_metrics(["A", "B"], ["A", "B"], 1)["recall"] == 0.5


def test_duplicate_predictions_or_ground_truth_fail():
    with pytest.raises(ValueError, match="unique"):
        retrieval_metrics(["A", "A"], ["A"], 2)
    with pytest.raises(ValueError, match="unique"):
        retrieval_metrics(["A"], ["A", "A"], 2)


def test_evaluation_suite_is_frozen_and_valid():
    suite, sha = load_suite(SUITE)
    retriever = CatalogRetriever.from_json(CATALOG, HashingEmbedder())
    report = evaluate_retriever(retriever, suite)
    assert len(sha) == 64
    assert report["case_count"] == 15
    assert report["negative_case_count"] == 3
    assert report["negative_pass_rate"] == 1.0
    assert all(0 <= metric <= 1 for metric in report["metrics"].values())
    assert len(report["per_case"]) == 15
    assert report["catalog_sha256"] == retriever.catalog_sha256


def test_retrieval_ranks_are_repeatable_and_no_labels_leak_into_retriever():
    suite, _ = load_suite(SUITE)
    retriever = CatalogRetriever.from_json(CATALOG)
    assert evaluate_retriever(retriever, suite) == evaluate_retriever(retriever, suite)
    assert not hasattr(retriever, "relevant_product_ids")


def test_suite_rejects_invalid_ground_truth_and_duplicate_case_ids():
    suite, _ = load_suite(SUITE)
    broken = suite.model_dump(mode="json")
    broken["cases"][0]["relevant_product_ids"] = ["NONEXISTENT"]
    with pytest.raises(ValueError, match="unknown product"):
        evaluate_retriever(CatalogRetriever.from_json(CATALOG), type(suite).model_validate(broken))
    broken = suite.model_dump(mode="json")
    broken["cases"][1]["case_id"] = broken["cases"][0]["case_id"]
    with pytest.raises(ValidationError, match="duplicate benchmark case"):
        type(suite).model_validate(broken)


def test_ineligible_ground_truth_is_rejected():
    suite, _ = load_suite(SUITE)
    broken = suite.model_dump(mode="json")
    broken["cases"][0]["relevant_product_ids"] = ["SYN-R02"]
    with pytest.raises(ValueError, match="ineligible"):
        evaluate_retriever(CatalogRetriever.from_json(CATALOG), type(suite).model_validate(broken))


def test_runner_records_data_hashes_and_versions():
    result = run(CATALOG, SUITE, backend="hashing", revision="unused", device="cpu")
    assert len(result["catalog_sha256"]) == 64
    assert len(result["suite_sha256"]) == 64
    assert result["environment"]["numpy"]
    assert result["environment"]["sentence_transformers"] is None
    assert "synthetic" in result["disclaimer"].lower()


def test_unpinned_minilm_is_rejected_before_loading_weights():
    with pytest.raises(ValueError, match="revision"):
        MiniLMEmbedder(revision="main")
