"""Offline simulation of the real-Qwen matrix runner; no model download."""

import json
from pathlib import Path

import numpy as np

from recsys.evaluation.baseline_suite import load_baseline_suite
from recsys.evaluation.run_live_reliability import (
    evaluate_matrix, evaluate_source_guard_demo, PerCaseSharedGenerator,
)
from recsys.multiagent import TypedSourceGuard
from recsys.retrieval import CatalogRetriever


ROOT = Path(__file__).resolve().parents[1]


class AllPositiveEmbedder:
    name = "offline-test-positive-vectors"

    def embed(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32)


class FakeModel:
    """Deterministic model replacement tests call accounting, NOT Qwen quality."""

    def __init__(self):
        self.call_count = 0
        self.last_prompt_tokens = 30
        self.last_generated_tokens = 25

    def generate(self, instructions, context):
        self.call_count += 1
        data = json.loads(context)
        first = data["products"][0]
        payload = {
            "recommendations": [{
                "product_id": first["product"]["product_id"],
                "evidence_ids": [first["evidence"][0]["evidence_id"]],
                "reason": "Synthetic item has a listing, not a medical claim.",
            }],
        }
        # Fixed intentionally malformed format to test normalization effect.
        return "```json\n" + json.dumps(payload) + "\n```"


def test_matrix_shares_exact_first_model_calls_without_doubling_physical_cost():
    model = FakeModel()
    retriever = CatalogRetriever.from_json(
        ROOT / "data/retrieval_eval/catalog.json",
        embedder=AllPositiveEmbedder(),
    )
    cases, _ = load_baseline_suite(
        ROOT / "data/baseline_eval/suite.json", repo_root=ROOT, retriever=retriever,
    )
    report = evaluate_matrix(cases, retriever, model)
    assert report["physical_model_invocations"] == 19
    assert model.call_count == 19
    original = report["arms"]["no_preflight__strict"]
    normalized = report["arms"]["no_preflight__safe_json_normalization"]
    guarded = report["arms"]["preflight__safe_json_normalization"]
    assert original["logical_generator_calls"] == 19
    assert normalized["logical_generator_calls"] == 19
    assert guarded["logical_generator_calls"] == 10
    assert guarded["insufficient_evidence_count"] == 9
    assert original["critic_approved_count"] == 0
    assert normalized["critic_approved_count"] > 0
    assert all(len(arm["per_case"]) == 25 for arm in report["arms"].values())
    assert all(arm["no_match_zero_call_count"] == 6 for arm in report["arms"].values())
    for a, b in zip(original["per_case"], normalized["per_case"]):
        assert a["case_id"] == b["case_id"]
        assert a["raw_first_output"] == b["raw_first_output"]
        assert a["prompt_context_sha256"] == b["prompt_context_sha256"]
    assert report["unsupported_claim_rate"] is None


def test_cached_error_cannot_cause_silent_multiple_retries():
    class AlwaysBroken(FakeModel):
        def generate(self, instructions, context):
            self.call_count += 1
            raise RuntimeError("injected failure")

    broken = AlwaysBroken()
    cached = PerCaseSharedGenerator(broken)
    for _ in range(2):
        try:
            cached.generate("system", "context")
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected cached generator exception")
    assert broken.call_count == 1
    assert len(cached.physical_calls) == 1


def test_fictional_source_demo_skips_three_unreliable_fields_even_with_real_model_adapter():
    retriever = CatalogRetriever.from_json(
        ROOT / "data/source_evidence_demo/catalog.json",
        embedder=AllPositiveEmbedder(),
    )
    model = FakeModel()
    guard = TypedSourceGuard(
        catalog_path=ROOT / "data/source_evidence_demo/catalog.json",
        source_path=ROOT / "data/source_evidence_demo/source_documents.json",
    )
    report = evaluate_source_guard_demo(retriever, model, source_guard=guard)
    assert report["physical_model_invocations"] == 1
    assert model.call_count == 1
    assert [r["status"] for r in report["results"][1:]] == ["insufficient_evidence"] * 3
    assert [r["physical_model_invocations"] for r in report["results"]] == [1, 0, 0, 0]
    assert report["semantic_citation_groundedness"] is None
