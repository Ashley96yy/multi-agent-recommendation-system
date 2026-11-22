"""Run fixed 15+10 synthetic Single-Agent RAG baseline evaluation on real Qwen.

Observations are saved regardless of success. Do not claim semantic grounding
from structural citation checks or attribute-field coverage proxies.
"""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from recsys.baseline import SingleAgentRAG
from recsys.baseline.local_qwen import MODEL_ID, MODEL_REVISION, LocalQwenGenerator
from recsys.evaluation.baseline_score import evaluate_baseline
from recsys.evaluation.baseline_suite import load_baseline_suite
from recsys.evaluation.run_single_agent import MINILM_REVISION
from recsys.retrieval import CatalogRetriever, HashingEmbedder, MiniLMEmbedder


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUITE = ROOT / "data/baseline_eval/suite.json"
DEFAULT_CATALOG = ROOT / "data/retrieval_eval/catalog.json"


def run(
    *,
    suite_path: Path,
    catalog_path: Path,
    retrieval_backend: str = "minilm",
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
    top_k: int = 3,
    max_new_tokens: int = 240,
) -> dict:
    if retrieval_backend not in {"minilm", "hashing"}:
        raise ValueError("unknown retrieval backend")
    embedder = (
        MiniLMEmbedder(revision=MINILM_REVISION, device="cpu")
        if retrieval_backend == "minilm" else HashingEmbedder()
    )
    retriever = CatalogRetriever.from_json(catalog_path, embedder=embedder)
    cases, hashes = load_baseline_suite(suite_path, repo_root=ROOT, retriever=retriever)
    generator = LocalQwenGenerator(
        model_id=model_id, revision=revision, device="cpu", max_new_tokens=max_new_tokens
    )
    report = evaluate_baseline(SingleAgentRAG(retriever, generator), cases, top_k=top_k)
    report.update({
        "suite_id": "single-agent-baseline-synthetic-25-v1",
        "data_hashes": hashes,
        "retrieval_backend": retriever.embedder.name,
        "generator": generator.name,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "environment": {
            "python": platform.python_version(),
            "pydantic": importlib.metadata.version("pydantic"),
            "numpy": importlib.metadata.version("numpy"),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "sentence_transformers": (
                importlib.metadata.version("sentence-transformers")
                if retrieval_backend == "minilm" else None
            ),
        },
        "disclaimer": (
            "Small synthetic author-labeled benchmark. Structural citation validity "
            "and field-link coverage are NOT factual entailment. "
            "No independent unsupported-claim rate has been established."
        ),
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--retrieval-backend", choices=["minilm", "hashing"], default="minilm")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=240)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/evaluation/single-agent-baseline-25.json"))
    args = parser.parse_args()
    report = run(suite_path=args.suite, catalog_path=args.catalog,
                 retrieval_backend=args.retrieval_backend,
                 model_id=args.model_id, revision=args.revision,
                 top_k=args.top_k, max_new_tokens=args.max_new_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    keys = [
        "suite_id", "case_count", "normal_count", "red_team_count",
        "normal_success_rate", "red_team_success_rate",
        "positive_success_rate", "no_match_zero_call_rate",
        "expected_retrieval_hit_rate_at_k", "accepted_count",
        "required_attribute_field_citation_coverage_among_applicable",
        "red_team_forbidden_phrase_leak_case_rate",
        "generation_errors", "model_invocations", "tokens",
        "generated_case_latency_seconds", "unsupported_claim_rate",
        "semantic_citation_groundedness", "data_hashes",
    ]
    print(json.dumps({key: report[key] for key in keys}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
