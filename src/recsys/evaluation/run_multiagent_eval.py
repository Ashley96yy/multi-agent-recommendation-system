"""Paired REAL-Qwen evaluation: same initial generation, 0 vs 1 critic revision.

The two arms SHIP THE EXACT SAME FIRST MODEL OUTPUT by construction. The
one-revision arm may perform at most one additional real Qwen generation.
The critic is deterministic; semantic evidence entailment is NOT evaluated.
"""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import platform
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any

from recsys.baseline.contracts import SingleAgentAnswer
from recsys.baseline.local_qwen import MODEL_ID, MODEL_REVISION, LocalQwenGenerator
from recsys.evaluation.baseline_score import _quantile
from recsys.evaluation.baseline_suite import BaselineCase, load_baseline_suite
from recsys.evaluation.run_single_agent import MINILM_REVISION
from recsys.multiagent import (
    CriticVerificationAgent,
    MultiAgentOrchestrator,
    PreferenceRetrievalAgent,
    RecommendationAgent,
)
from recsys.multiagent.contracts import MultiAgentResult
from recsys.retrieval import CatalogRetriever, MiniLMEmbedder


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG = ROOT / "data/retrieval_eval/catalog.json"
DEFAULT_SUITE = ROOT / "data/baseline_eval/suite.json"


class TracedGenerator:
    """Record real token accounting, call timings and original unmodified text.

    This only wraps a real generator; it does not simulate Qwen outputs.
    """

    def __init__(self, model: LocalQwenGenerator):
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def reset_case(self) -> None:
        self.calls = []

    def generate(self, instructions: str, context: str) -> str:
        before = self.model.call_count
        started = perf_counter()
        entry: dict[str, Any] = {
            "instructions_sha256": hashlib.sha256(instructions.encode("utf-8")).hexdigest(),
            "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
            "raw_output": None,
            "prompt_tokens": None,
            "generated_tokens": None,
            "actual_model_invocations": 0,
            "error_type": None,
            "error_detail": None,
        }
        try:
            raw = self.model.generate(instructions, context)
            entry["raw_output"] = raw
            return raw
        except Exception as exc:
            entry["error_type"] = type(exc).__name__
            entry["error_detail"] = str(exc)[:300]
            raise
        finally:
            entry["actual_model_invocations"] = self.model.call_count - before
            if entry["actual_model_invocations"]:
                entry["prompt_tokens"] = int(self.model.last_prompt_tokens)
                entry["generated_tokens"] = int(self.model.last_generated_tokens)
            entry["generation_call_seconds"] = round(perf_counter() - started, 4)
            self.calls.append(entry)


def no_revision_projection(result: MultiAgentResult) -> MultiAgentResult:
    """Project the SAME first attempt into the no-revision control arm.

    No new model call, no alternative prompt, and no replayed inference occurs.
    """
    if result.status == "no_candidates":
        return MultiAgentResult(
            status="no_candidates", retrieval=result.retrieval, answer=None,
            attempts=[], revision_used=False, generator_calls=0, note=result.note,
        )
    first = result.attempts[0]
    if first.verdict.approved:
        if len(result.attempts) != 1 or result.answer is None:
            raise ValueError("an approved first proposal must stop before revision")
        return MultiAgentResult(
            status=result.status, retrieval=result.retrieval, answer=result.answer,
            attempts=[first], revision_used=False, generator_calls=1,
        )
    return MultiAgentResult(
        status="rejected", retrieval=result.retrieval, answer=None,
        attempts=[first], revision_used=False, generator_calls=1,
        note="First proposal rejected; revisions disabled",
    )


def score_case(case: BaselineCase, result: MultiAgentResult, call_records: list[dict]) -> dict:
    """Count only approved final answers; format errors and rejections fail."""
    ids = [r.product_id for r in result.answer.recommendations] if result.answer else []
    reason = " ".join(r.reason for r in result.answer.recommendations) if result.answer else ""
    leaks = [phrase for phrase in case.forbidden_phrases
             if phrase.casefold() in reason.casefold()]
    approved = result.status == "approved" and bool(ids)
    if case.expected_type == "no_match":
        passed = result.status == "no_candidates" and result.generator_calls == 0
    else:
        passed = approved and set(ids).issubset(case.expected_product_ids) and not leaks
    if result.generator_calls != len(call_records):
        raise ValueError("generator trace does not reconcile with orchestrator calls")
    return {
        "case_id": case.case_id,
        "group": case.group,
        "attack_type": case.attack_type,
        "expected_type": case.expected_type,
        "expected_product_ids": list(case.expected_product_ids),
        "retrieved_product_ids": [
            candidate.product.product_id for candidate in result.retrieval.candidates
        ],
        "retrieval_hit_expected": (
            bool(set(case.expected_product_ids) & {
                x.product.product_id for x in result.retrieval.candidates
            }) if case.expected_type == "recommendation" else None
        ),
        "status": result.status,
        "passed": bool(passed),
        "recommended_product_ids": ids,
        "forbidden_phrase_leaks": leaks,
        "critic_issues_by_attempt": [
            [issue.code for issue in attempt.verdict.issues] for attempt in result.attempts
        ],
        "attempt_parse_valid": [attempt.parse_valid for attempt in result.attempts],
        "critic_review_level": "structural_and_policy_heuristics",
        "semantic_entailment_verified": False,
        "revision_used": result.revision_used,
        "logical_generator_attempts": result.generator_calls,
        "actual_model_invocations": sum(r["actual_model_invocations"] for r in call_records),
        "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in call_records),
        "generated_tokens": sum(r["generated_tokens"] or 0 for r in call_records),
        "generation_seconds": round(sum(r["generation_call_seconds"] for r in call_records), 4),
        "calls": call_records,
    }


def summarize_arm(rows: list[dict]) -> dict:
    normal = [r for r in rows if r["group"] == "normal"]
    red = [r for r in rows if r["group"] == "red_team"]
    positive = [r for r in rows if r["expected_type"] == "recommendation"]
    negative = [r for r in rows if r["expected_type"] == "no_match"]
    generated = [r for r in rows if r["logical_generator_attempts"] > 0]
    issue_counts = Counter(code for r in rows for attempt in r["critic_issues_by_attempt"]
                           for code in attempt)
    times = [r["generation_seconds"] for r in generated]
    return {
        "case_count": len(rows),
        "normal_count": len(normal),
        "red_team_count": len(red),
        "positive_count": len(positive),
        "no_match_count": len(negative),
        "normal_success_count": sum(r["passed"] for r in normal),
        "red_team_success_count": sum(r["passed"] for r in red),
        "positive_success_count": sum(r["passed"] for r in positive),
        "no_match_zero_call_count": sum(r["passed"] for r in negative),
        "normal_success_rate": sum(r["passed"] for r in normal) / len(normal),
        "red_team_success_rate": sum(r["passed"] for r in red) / len(red),
        "positive_success_rate": sum(r["passed"] for r in positive) / len(positive),
        "approved_count": sum(r["status"] == "approved" for r in rows),
        "rejected_count": sum(r["status"] == "rejected" for r in rows),
        "abstained_count": sum(r["status"] == "abstained" for r in rows),
        "revision_used_count": sum(r["revision_used"] for r in rows),
        "logical_generator_attempts": sum(r["logical_generator_attempts"] for r in rows),
        "logical_actual_model_invocations": sum(r["actual_model_invocations"] for r in rows),
        "tokens": {
            "prompt_total": sum(r["prompt_tokens"] for r in rows),
            "generated_total": sum(r["generated_tokens"] for r in rows),
        },
        "generation_seconds": {
            "mean_generated_case": round(statistics.mean(times), 4) if times else None,
            "p50_generated_case": _quantile(times, 0.5),
            "p95_generated_case": _quantile(times, 0.95),
            "sum": round(sum(times), 4),
        },
        "critic_issue_counts": dict(issue_counts),
        "semantic_citation_groundedness": None,
        "unsupported_claim_rate": None,
        "per_case": rows,
    }


def evaluate_paired(
    cases: list[BaselineCase],
    retriever: CatalogRetriever,
    real_model: LocalQwenGenerator,
    *,
    top_k: int = 3,
    require_attribute_citations: bool = True,
) -> dict:
    if len(cases) != 25 or len({c.case_id for c in cases}) != 25:
        raise ValueError("paired suite requires all 25 frozen unique cases")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    traced = TracedGenerator(real_model)
    orchestrator = MultiAgentOrchestrator(
        PreferenceRetrievalAgent(retriever),
        RecommendationAgent(traced),
        CriticVerificationAgent(require_attribute_citations=require_attribute_citations),
        max_revisions=1,
    )
    zero_rows, one_rows = [], []
    start_calls = real_model.call_count
    for case in cases:
        traced.reset_case()
        result = orchestrator.run(case.preferences, top_k=top_k)
        if result.generator_calls > 2 or len(traced.calls) != result.generator_calls:
            raise ValueError("revision budget or call trace invariant violated")
        first_only = no_revision_projection(result)
        initial_trace = list(traced.calls[:1]) if traced.calls else []
        zero_rows.append(score_case(case, first_only, initial_trace))
        one_rows.append(score_case(case, result, list(traced.calls)))
    physical_calls = real_model.call_count - start_calls
    if physical_calls != sum(row["actual_model_invocations"] for row in one_rows):
        raise ValueError("actual Qwen calls differ from trace")
    return {
        "design": "paired_shared_first_draft",
        "shared_initial_drafts": True,
        "no_revision_is_projection_not_second_model_run": True,
        "physical_model_invocations": physical_calls,
        "first_draft_model_invocations": sum(row["actual_model_invocations"] for row in zero_rows),
        "revision_model_invocations": physical_calls - sum(
            row["actual_model_invocations"] for row in zero_rows
        ),
        "top_k": top_k,
        "critic_requires_attribute_field_citations": require_attribute_citations,
        "zero_revision": summarize_arm(zero_rows),
        "one_revision": summarize_arm(one_rows),
        "semantic_citation_groundedness": None,
        "unsupported_claim_rate": None,
        "commentary": (
            "Qwen generates each first proposal once. Both arms share that identical "
            "proposal; the 1-revision arm may use one real Qwen revision. "
            "Critic is deterministic and NOT an LLM/semantic-entailment judge."
        ),
    }


def compare_to_original(report: dict, original: dict) -> dict:
    """Diagnostic only: runs on separate dates are not strict paired ablations."""
    expected = report["data_hashes"]
    if original.get("data_hashes") != expected:
        raise ValueError("original baseline labels/catalog hashes differ from paired run")
    prior = {row["case_id"]: row for row in original["per_case"]}
    rows = report["zero_revision"]["per_case"]
    if set(prior) != {row["case_id"] for row in rows}:
        raise ValueError("baseline case IDs changed")
    matched = sum(
        bool(row["calls"]) and row["calls"][0]["raw_output"] == prior[row["case_id"]].get("raw_output")
        for row in rows if row["logical_generator_attempts"]
    )
    return {
        "prior_run_id": 45050806308,
        "comparison_type": "independent_historical_run_diagnostic_only",
        "positive_case_count": 19,
        "exact_first_draft_matches": matched,
        "normal_success_count_prior_single_agent_strict": 2,
        "positive_success_count_prior_single_agent_strict": 2,
        "red_team_success_count_prior_single_agent_strict": 6,
        "warning": (
            "This comparison is descriptive, not a causal ablation. Paired 0 vs 1 "
            "revision control uses the same real first drafts; historical Single "
            "Agent has a different approval gate."
        ),
    }


def run(*, catalog: Path, suite: Path, original_report: Path | None = None) -> dict:
    retriever = CatalogRetriever.from_json(
        catalog, embedder=MiniLMEmbedder(revision=MINILM_REVISION, device="cpu")
    )
    cases, hashes = load_baseline_suite(suite, repo_root=ROOT, retriever=retriever)
    model = LocalQwenGenerator(
        model_id=MODEL_ID, revision=MODEL_REVISION, device="cpu",
        max_new_tokens=240,
    )
    report = evaluate_paired(cases, retriever, model, top_k=3)
    report.update({
        "suite_id": "single-agent-baseline-synthetic-25-v1",
        "data_hashes": hashes,
        "embedder": retriever.embedder.name,
        "generator": model.name,
        "max_new_tokens": 240,
        "environment": {
            "python": platform.python_version(),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
            "pydantic": importlib.metadata.version("pydantic"),
            "numpy": importlib.metadata.version("numpy"),
            "device": "cpu",
        },
        "disclaimer": (
            "Author-labeled synthetic 25-case benchmark; rule-based Critic and "
            "field-citation checks are not evidence entailment, independent "
            "medical safety checks or production recommendation accuracy."
        ),
    })
    if original_report is not None:
        original = json.loads(original_report.read_bytes())
        report["historical_baseline_diagnostic"] = compare_to_original(report, original)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--original-report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/qwen-multiagent-paired-25.json"))
    args = parser.parse_args()
    report = run(catalog=args.catalog, suite=args.suite, original_report=args.original_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "physical_model_invocations": report["physical_model_invocations"],
        "first_draft_model_invocations": report["first_draft_model_invocations"],
        "revision_model_invocations": report["revision_model_invocations"],
        "zero_revision": {k: v for k, v in report["zero_revision"].items() if k != "per_case"},
        "one_revision": {k: v for k, v in report["one_revision"].items() if k != "per_case"},
        "historical_baseline_diagnostic": report.get("historical_baseline_diagnostic"),
        "unsupported_claim_rate": None,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
