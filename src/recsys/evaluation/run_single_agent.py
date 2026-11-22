"""Real one-pass Single-Agent RAG smoke evaluator.

All scenarios in the default fixture are synthetic. Records actual generation
outcomes, failures and timings; never claims independent grounding verification.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
from pathlib import Path
from time import perf_counter

from recsys.baseline import SingleAgentRAG
from recsys.baseline.local_qwen import MODEL_ID, MODEL_REVISION, LocalQwenGenerator
from recsys.retrieval import CatalogRetriever, HashingEmbedder, MiniLMEmbedder
from recsys.schemas import UserPreferences


DEFAULT_SUITE = Path("data/llm_smoke/cases.json")
MINILM_REVISION = "826711e54e001c83835913827a843d8dd0a1def9"


def load_smoke_cases(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    suite = json.loads(raw)
    if not isinstance(suite, dict) or not isinstance(suite.get("cases"), list) or not suite["cases"]:
        raise ValueError("smoke suite must include nonempty cases")
    if len({c["case_id"] for c in suite["cases"]}) != len(suite["cases"]):
        raise ValueError("duplicate smoke case IDs")
    for case in suite["cases"]:
        if case["expected_type"] not in {"recommendation", "no_match"}:
            raise ValueError("unknown smoke case expected type")
        if case["expected_type"] == "recommendation" and not case["expected_product_ids"]:
            raise ValueError("positive smoke case requires expected IDs")
        if case["expected_type"] == "no_match" and case["expected_product_ids"]:
            raise ValueError("negative smoke case must not have expected IDs")
    return suite, hashlib.sha256(raw).hexdigest()


def evaluate_single_agent(rag: SingleAgentRAG, cases: list[dict]) -> dict:
    """Run each fixed case once, recording observed failures instead of hiding them."""
    per_case = []
    latencies = []
    positive_pass = []
    negative_pass = []
    for case in cases:
        preferences = UserPreferences.model_validate_json(json.dumps(case["preferences"]))
        prior_calls = getattr(rag.generator, "call_count", 0)
        start = perf_counter()
        record = {
            "case_id": case["case_id"],
            "expected_type": case["expected_type"],
            "expected_product_ids": case["expected_product_ids"],
            "status": "error",
            "passed": False,
            "model_invocations": 0,
            "recommended_product_ids": [],
            "cited_evidence_ids": [],
            "error_type": None,
            "error_detail": None,
        }
        try:
            answer, retrieved = rag.recommend(preferences, top_k=1)
            recommended_ids = [r.product_id for r in answer.recommendations]
            evidence_ids = [ref for r in answer.recommendations for ref in r.evidence_ids]
            record["recommended_product_ids"] = recommended_ids
            record["cited_evidence_ids"] = evidence_ids
            record["retrieval_candidates"] = [
                c.product.product_id for c in retrieved.candidates
            ]
            record["status"] = "abstained" if answer.no_suitable_match else "accepted"
            if case["expected_type"] == "recommendation":
                record["passed"] = bool(recommended_ids) and all(
                    pid in case["expected_product_ids"] for pid in recommended_ids
                )
            else:
                record["passed"] = bool(answer.no_suitable_match) and not recommended_ids
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            record["error_detail"] = str(exc)[:250]
        elapsed = perf_counter() - start
        record["latency_seconds"] = round(elapsed, 4)
        record["model_invocations"] = getattr(rag.generator, "call_count", 0) - prior_calls
        record["prompt_tokens"] = (
            getattr(rag.generator, "last_prompt_tokens", None)
            if record["model_invocations"] else None
        )
        record["generated_tokens"] = (
            getattr(rag.generator, "last_generated_tokens", None)
            if record["model_invocations"] else None
        )
        # For tests with synthetic data only, store a bounded untrusted output
        # to diagnose failures without accidentally claiming a verified answer.
        if record["model_invocations"]:
            record["raw_output"] = getattr(rag.generator, "last_response", None)
        if record["model_invocations"]:
            latencies.append(elapsed)
        if case["expected_type"] == "recommendation":
            positive_pass.append(bool(record["passed"]))
        else:
            negative_pass.append(bool(record["passed"]) and record["model_invocations"] == 0)
        per_case.append(record)
    return {
        "case_count": len(cases),
        "positive_case_count": len(positive_pass),
        "negative_case_count": len(negative_pass),
        "positive_accept_rate": (sum(positive_pass) / len(positive_pass)) if positive_pass else None,
        "negative_no_match_rate": (sum(negative_pass) / len(negative_pass)) if negative_pass else None,
        "model_invocations": sum(row["model_invocations"] for row in per_case),
        "generation_failures": sum(bool(row["error_type"]) for row in per_case),
        "mean_generation_case_latency_seconds": (
            round(statistics.mean(latencies), 4) if latencies else None
        ),
        "per_case": per_case,
    }


def run(*, suite_path: Path, catalog_path: Path, retrieval_backend: str = "minilm",
        model_id: str = MODEL_ID, revision: str = MODEL_REVISION,
        max_new_tokens: int = 240) -> dict:
    suite, suite_sha = load_smoke_cases(suite_path)
    if retrieval_backend not in {"minilm", "hashing"}:
        raise ValueError("retrieval_backend must be 'minilm' or 'hashing'")
    embedder = (
        MiniLMEmbedder(revision=MINILM_REVISION, device="cpu")
        if retrieval_backend == "minilm" else HashingEmbedder()
    )
    retriever = CatalogRetriever.from_json(catalog_path, embedder=embedder)
    generator = LocalQwenGenerator(model_id=model_id, revision=revision,
                                   device="cpu", max_new_tokens=max_new_tokens)
    rag = SingleAgentRAG(retriever, generator)
    report = evaluate_single_agent(rag, suite["cases"])
    report.update({
        "suite_id": suite["suite_id"],
        "suite_sha256": suite_sha,
        "catalog_sha256": retriever.catalog_sha256,
        "retrieval_backend": embedder.name,
        "generator": generator.name,
        "max_new_tokens": max_new_tokens,
        "environment": {
            "python": platform.python_version(),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "sentence_transformers": (
                importlib.metadata.version("sentence-transformers")
                if retrieval_backend == "minilm" else None
            ),
            "cpu": platform.processor(),
        },
        "disclaimer": (
            "Small synthetic integration smoke. Structurally valid citations are "
            "NOT evidence-entailment or recommendation-quality validation."
        ),
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--catalog", type=Path, default=Path("data/sample_catalog/catalog.json"))
    parser.add_argument("--retrieval-backend", choices=["minilm", "hashing"], default="minilm")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=240)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/qwen-single-agent.json"))
    args = parser.parse_args()
    report = run(
        suite_path=args.suite, catalog_path=args.catalog,
        retrieval_backend=args.retrieval_backend,
        model_id=args.model_id, revision=args.revision,
        max_new_tokens=args.max_new_tokens,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "suite_id", "generator", "retrieval_backend", "case_count",
        "positive_accept_rate", "negative_no_match_rate", "model_invocations",
        "generation_failures", "mean_generation_case_latency_seconds",
        "catalog_sha256", "suite_sha256",
    )}, indent=2))


if __name__ == "__main__":
    main()
