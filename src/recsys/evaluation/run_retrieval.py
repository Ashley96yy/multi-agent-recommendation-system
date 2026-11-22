"""Run frozen synthetic benchmark; print JSON metrics and write an optional artifact."""

import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from recsys.evaluation.retrieval import evaluate_retriever, load_suite
from recsys.retrieval import CatalogRetriever, HashingEmbedder, MiniLMEmbedder


DEFAULT_MODEL_REVISION = "826711e54e001c83835913827a843d8dd0a1def9"


def run(catalog: Path, suite_path: Path, *, backend: str, revision: str, device: str):
    suite, suite_sha = load_suite(suite_path)
    embedder = (
        HashingEmbedder() if backend == "hashing" else MiniLMEmbedder(
            revision=revision, device=device
        )
    )
    retriever = CatalogRetriever.from_json(catalog, embedder=embedder)
    report = evaluate_retriever(retriever, suite)
    report["suite_sha256"] = suite_sha
    report["environment"] = {
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "pydantic": importlib.metadata.version("pydantic"),
        "sentence_transformers": (
            importlib.metadata.version("sentence-transformers") if backend == "minilm" else None
        ),
        "device": device if backend == "minilm" else "cpu",
    }
    report["disclaimer"] = (
        "Small synthetic evaluation with author-labeled relevance; not a production "
        "or real-world skincare recommendation accuracy claim."
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/retrieval_eval/catalog.json"))
    parser.add_argument("--suite", type=Path, default=Path("data/retrieval_eval/suite.json"))
    parser.add_argument("--backend", choices=["hashing", "minilm"], default="hashing")
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.catalog, args.suite, backend=args.backend,
                 revision=args.revision, device=args.device)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        key: result[key] for key in ("suite_id", "embedder", "case_count", "negative_case_count",
                                    "metrics", "negative_pass_rate", "catalog_sha256", "suite_sha256")
    }, indent=2))


if __name__ == "__main__":
    main()
