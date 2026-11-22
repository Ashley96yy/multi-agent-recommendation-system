"""Real Qwen Stage 3.3 2x2 reliability ablation with shared first drafts.

The four logical arms are generated from two independent factors:
- strict vs deterministic safe JSON normalization (post-processing only)
- source field preflight off vs on (may suppress calls/change prompts)

Within each case identical first prompts share one REAL generated Qwen output.
All unique prompts invoke a real model and physical tokens are never double
counted. No revisions are used in this matrix (isolate first-pass policies).
"""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Protocol

from recsys.baseline.local_qwen import MODEL_ID, MODEL_REVISION, LocalQwenGenerator
from recsys.evaluation.baseline_suite import load_baseline_suite
from recsys.evaluation.run_single_agent import MINILM_REVISION
from recsys.multiagent import (
    CriticVerificationAgent, MultiAgentOrchestrator, PreferenceRetrievalAgent,
    RecommendationAgent, TypedSourceGuard,
)
from recsys.retrieval import CatalogRetriever, MiniLMEmbedder
from recsys.schemas import UserPreferences


ROOT = Path(__file__).resolve().parents[3]
FORMATS = ("strict", "safe_json_normalization")
PREFLIGHT = (False, True)


class ModelProtocol(Protocol):
    call_count: int
    last_prompt_tokens: int
    last_generated_tokens: int

    def generate(self, instructions: str, context: str) -> str: ...


class PerCaseSharedGenerator:
    """Caches *identical* first prompts only within the same request case."""

    def __init__(self, model: ModelProtocol):
        self.model = model
        self._cache: dict[str, dict] = {}
        self.physical_calls: list[dict] = []

    def reset_case(self):
        self._cache.clear()

    @staticmethod
    def key(instructions: str, context: str) -> str:
        return hashlib.sha256((instructions + "\0" + context).encode()).hexdigest()

    def generate(self, instructions: str, context: str) -> str:
        key = self.key(instructions, context)
        if key not in self._cache:
            before = self.model.call_count
            start = perf_counter()
            entry = {
                "prompt_context_sha256": key,
                "raw_output": None,
                "prompt_tokens": None,
                "generated_tokens": None,
                "error_type": None,
                "error_detail": None,
            }
            try:
                entry["raw_output"] = self.model.generate(instructions, context)
            except Exception as exc:
                entry["error_type"] = type(exc).__name__
                entry["error_detail"] = str(exc)[:250]
            finally:
                entry["generation_seconds"] = round(perf_counter() - start, 4)
                entry["physical_model_invocations"] = self.model.call_count - before
                if entry["physical_model_invocations"] > 0:
                    entry["prompt_tokens"] = self.model.last_prompt_tokens
                    entry["generated_tokens"] = self.model.last_generated_tokens
                self._cache[key] = entry
                self.physical_calls.append(entry)
        entry = self._cache[key]
        if entry["error_type"]:
            raise RuntimeError(
                "Cached real generator failure: " + entry["error_type"]
            )
        return entry["raw_output"]


def summarize(records: list[dict]) -> dict:
    normal = [r for r in records if r["group"] == "normal"]
    red = [r for r in records if r["group"] == "red_team"]
    pos = [r for r in records if r["expected_type"] == "recommendation"]
    neg = [r for r in records if r["expected_type"] == "no_match"]
    invoked = [r for r in records if r["generator_calls"] > 0]
    return {
        "case_count": len(records),
        "normal_success_count": sum(r["passed"] for r in normal),
        "red_team_success_count": sum(r["passed"] for r in red),
        "positive_success_count": sum(r["passed"] for r in pos),
        "no_match_zero_call_count": sum(r["passed"] for r in neg),
        "critic_approved_count": sum(r["status"] == "approved" for r in records),
        "insufficient_evidence_count": sum(r["status"] == "insufficient_evidence" for r in records),
        "rejected_count": sum(r["status"] == "rejected" for r in records),
        "logical_generator_calls": sum(r["generator_calls"] for r in records),
        "logical_prompt_tokens": sum(r["prompt_tokens"] for r in invoked),
        "logical_generated_tokens": sum(r["generated_tokens"] for r in invoked),
        "format_action_counts": dict(Counter(
            action for r in records for action in r["format_actions"]
        )),
        "critic_issue_counts": dict(Counter(
            code for r in records for code in r["critic_issue_codes"]
        )),
        "semantic_citation_groundedness": None,
        "unsupported_claim_rate": None,
        "per_case": records,
    }


def evaluate_matrix(cases, retriever, model: ModelProtocol, *, top_k=3) -> dict:
    if len(cases) != 25 or len(set(c.case_id for c in cases)) != 25:
        raise ValueError("full frozen 25-case suite required")
    cache = PerCaseSharedGenerator(model)
    start_calls = model.call_count
    rows: dict[str, list] = {
        f"{'preflight' if enabled else 'no_preflight'}__{policy}": []
        for enabled in PREFLIGHT for policy in FORMATS
    }
    physical = []
    for case in cases:
        cache.reset_case()
        physical_start = len(cache.physical_calls)
        for enabled in PREFLIGHT:
            for policy in FORMATS:
                key = f"{'preflight' if enabled else 'no_preflight'}__{policy}"
                orchestrator = MultiAgentOrchestrator(
                    PreferenceRetrievalAgent(retriever),
                    RecommendationAgent(cache),
                    CriticVerificationAgent(),
                    max_revisions=0,
                    format_policy=policy,
                    evidence_preflight=enabled,
                )
                result = orchestrator.run(case.preferences, top_k=top_k)
                if result.generator_calls not in (0, 1):
                    raise ValueError("unexpected revision during first-pass ablation")
                matched_call = None
                if result.generator_calls:
                    # At this point cached prompt key remains identical across
                    # JSON formatting arms and any unchanged candidate prompt.
                    current = list(cache._cache.values())
                    if not current:
                        raise ValueError("logical model call was not recorded")
                    matched_call = current[-1]
                recommended = [
                    item.product_id for item in result.answer.recommendations
                ] if result.answer is not None else []
                reason = " ".join(
                    item.reason for item in result.answer.recommendations
                ).casefold() if result.answer is not None else ""
                leaked = [
                    phrase for phrase in case.forbidden_phrases
                    if phrase.casefold() in reason
                ]
                if case.expected_type == "no_match":
                    passed = result.status == "no_candidates" and result.generator_calls == 0
                else:
                    passed = (
                        result.status == "approved"
                        and bool(recommended)
                        and set(recommended).issubset(case.expected_product_ids)
                        and not leaked
                    )
                row = {
                    "case_id": case.case_id, "group": case.group,
                    "expected_type": case.expected_type,
                    "status": result.status, "passed": bool(passed),
                    "retrieved_candidate_ids": [
                        r.product.product_id for r in result.retrieval.candidates
                    ],
                    "recommended_ids": recommended,
                    "generator_calls": result.generator_calls,
                    "format_actions": [
                        a for step in result.attempts for a in step.format_actions
                    ],
                    "critic_issue_codes": [
                        issue.code for step in result.attempts
                        for issue in step.verdict.issues
                    ],
                    "prompt_context_sha256": (
                        matched_call["prompt_context_sha256"] if matched_call else None
                    ),
                    "prompt_tokens": (matched_call["prompt_tokens"] or 0) if matched_call else 0,
                    "generated_tokens": (matched_call["generated_tokens"] or 0) if matched_call else 0,
                    "raw_first_output": matched_call["raw_output"] if matched_call else None,
                    "semantic_entailment_verified": False,
                }
                rows[key].append(row)
        physical.extend(cache.physical_calls[physical_start:])
    actual = model.call_count - start_calls
    total = sum(x["physical_model_invocations"] for x in physical)
    if actual != total:
        raise ValueError("physical Qwen call count mismatch")
    result = {
        "design": "real_qwen_first_pass_2x2_shared_identical_prompts",
        "physical_model_invocations": actual,
        "physical_call_records": physical,
        "format_policy_only_postprocessing": True,
        "preflight_enabled_may_change_prompt_or_skip_generation": True,
        "model_revisions_per_request": 0,
        "arms": {key: summarize(value) for key, value in rows.items()},
        "unsupported_claim_rate": None,
        "semantic_citation_groundedness": None,
        "limitations": (
            "The four arms share a physical Qwen generation only when the "
            "case and exact prompt/context agree. Arm logical token counts "
            "overlap and must not be summed as physical cost. Attribute source "
            "availability is not factual semantic claim verification."
        ),
    }
    for policy in FORMATS:
        original = result["arms"][f"no_preflight__{policy}"]
        guarded = result["arms"][f"preflight__{policy}"]
        if original["case_count"] != 25 or guarded["case_count"] != 25:
            raise ValueError("missing benchmark cases")
        if original["logical_generator_calls"] < guarded["logical_generator_calls"]:
            raise ValueError("source preflight should not add logical model calls")
    return result


def evaluate_source_guard_demo(retriever, model: ModelProtocol, *, source_guard) -> dict:
    """Real Qwen smoke: 1 consistent case generates; 3 untrusted cases skip."""
    cases = [
        ("consistent", UserPreferences(
            request="Cloud Calm Facial Cream", required_category="moisturizer",
            required_attributes={"fragrance_free"},
        )),
        ("conflict", UserPreferences(
            request="Dusk Barrier Balm", required_category="balm",
            required_attributes={"fragrance_free"},
        )),
        ("stale_revision", UserPreferences(
            request="Daylight Mineral Sun Cream", required_category="sunscreen",
            required_attributes={"mineral"},
        )),
        ("missing_source", UserPreferences(
            request="Meadow Gentle Wash", required_category="cleanser",
            required_attributes={"fragrance_free"},
        )),
    ]
    results = []
    before = model.call_count
    for label, preferences in cases:
        control = MultiAgentOrchestrator(
            PreferenceRetrievalAgent(retriever), RecommendationAgent(model),
            CriticVerificationAgent(source_guard=source_guard),
            max_revisions=0, evidence_preflight=True,
            format_policy="safe_json_normalization",
        )
        record = {"case_id": label}
        start = model.call_count
        result = control.run(preferences, top_k=4)
        record.update({
            "status": result.status,
            "physical_model_invocations": model.call_count - start,
            "critic_issue_codes": [
                issue.code for attempt in result.attempts for issue in attempt.verdict.issues
            ],
            "generation_approved": result.status == "approved",
            "semantic_entailment_verified": False,
        })
        results.append(record)
    return {
        "source_catalog_sha256": retriever.catalog_sha256,
        "source_docs_sha256": source_guard.source_docs_sha256,
        "physical_model_invocations": model.call_count - before,
        "results": results,
        "semantic_citation_groundedness": None,
        "disclaimer": "FICTIONAL typed source fixtures only. The one real generation is a smoke check, not semantic claim accuracy.",
    }


def run() -> dict:
    original_retriever = CatalogRetriever.from_json(
        ROOT / "data/retrieval_eval/catalog.json",
        embedder=MiniLMEmbedder(revision=MINILM_REVISION, device="cpu"),
    )
    cases, hashes = load_baseline_suite(
        ROOT / "data/baseline_eval/suite.json",
        repo_root=ROOT, retriever=original_retriever,
    )
    model = LocalQwenGenerator(model_id=MODEL_ID, revision=MODEL_REVISION, device="cpu", max_new_tokens=240)
    matrix = evaluate_matrix(cases, original_retriever, model)
    fictional_retriever = CatalogRetriever.from_json(
        ROOT / "data/source_evidence_demo/catalog.json",
        embedder=MiniLMEmbedder(revision=MINILM_REVISION, device="cpu"),
    )
    guard = TypedSourceGuard(
        catalog_path=ROOT / "data/source_evidence_demo/catalog.json",
        source_path=ROOT / "data/source_evidence_demo/source_documents.json",
    )
    source_demo = evaluate_source_guard_demo(fictional_retriever, model, source_guard=guard)
    return {
        "suite_id": "single-agent-baseline-synthetic-25-v1",
        "data_hashes": hashes,
        "generator": model.name,
        "embedder": original_retriever.embedder.name,
        "top_k": 3,
        "max_new_tokens": 240,
        "environment": {
            "python": platform.python_version(),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
        },
        "matrix": matrix,
        "fictional_source_guard_smoke": source_demo,
        "total_physical_model_invocations": (
            matrix["physical_model_invocations"] + source_demo["physical_model_invocations"]
        ),
        "unsupported_claim_rate": None,
        "semantic_citation_groundedness": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/stage33-real-qwen-matrix.json"))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "total_physical_model_invocations": result["total_physical_model_invocations"],
        "matrix_physical_model_invocations": result["matrix"]["physical_model_invocations"],
        "arms": {key: {name: arm[name] for name in (
            "positive_success_count", "critic_approved_count",
            "insufficient_evidence_count", "logical_generator_calls",
        )} for key, arm in result["matrix"]["arms"].items()},
        "source_smoke": result["fictional_source_guard_smoke"]["results"],
    }, indent=2))


if __name__ == "__main__":
    main()
