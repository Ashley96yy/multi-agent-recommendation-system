"""Execute REAL pinned Qwen on 26 authored fictional atomic claim cases.

Never call this an independent human-calibrated evaluation; author gold labels
are joined ONLY AFTER all model outputs are generated and recorded.
"""

import argparse
from hashlib import sha256
import importlib.metadata
import json
import platform
from pathlib import Path

from recsys.baseline.local_qwen import LocalQwenGenerator, MODEL_ID, MODEL_REVISION
from recsys.evaluation.atomic_claims import load_frozen_claims
from recsys.evaluation.model_critic import run_model_critic, evaluate_predictions
from recsys.evaluation.source_audit import SourceFixture


def evaluate(*, model, cases_path=None, labels_path=None) -> dict:
    kwargs = {}
    if cases_path is not None:
        kwargs["cases_path"] = cases_path
    if labels_path is not None:
        kwargs["labels_path"] = labels_path
    suite, gold, guard, provenance = load_frozen_claims(**kwargs)
    from recsys.evaluation.atomic_claims import ROOT
    source_path = (ROOT / suite.sources_path).resolve()
    fixture = SourceFixture.model_validate_json(source_path.read_bytes())
    generation = run_model_critic(suite, fixture, guard._evidence, model)
    scored = evaluate_predictions(suite, gold, generation)
    return {
        "suite_id": suite.suite_id,
        "model": getattr(model, "name", "offline-test-double"),
        "decoding": "greedy",
        "max_new_tokens": getattr(model, "max_new_tokens", None),
        "source_sha256": provenance,
        "cases_generated": len(generation["predictions"]),
        "generator": generation,
        "comparison_against_author_labels": scored,
        "independent_human_reviews_received": 0,
        "human_calibrated_accuracy": None,
        "real_world_semantic_groundedness": None,
        "notes": (
            "This is a real model-based Critic over individually supplied "
            "claims and fictional documents. No claim extraction from "
            "recommendation paragraphs or independent human labeling has "
            "occurred. Author-written labels are used only for post-hoc "
            "evaluation, never for prompt construction."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/evaluation/stage35-qwen-atomic-claims.json"))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    if not 32 <= args.max_new_tokens <= 240:
        raise ValueError("max-new-tokens must be 32..240")
    model = LocalQwenGenerator(
        model_id=MODEL_ID, revision=MODEL_REVISION, device="cpu",
        max_new_tokens=args.max_new_tokens, max_prompt_tokens=3000,
    )
    result = evaluate(model=model)
    result["environment"] = {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "pydantic": importlib.metadata.version("pydantic"),
        "device": "cpu",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({
        "model": result["model"], "cases": result["cases_generated"],
        "real_model_invocations": result["generator"]["physical_model_invocations"],
        "prompt_tokens": result["generator"]["total_prompt_tokens"],
        "generated_tokens": result["generator"]["total_generated_tokens"],
        "author_label_accuracy_all": result["comparison_against_author_labels"]["all"]["accuracy_including_abstentions"],
        "author_label_accuracy_typed": result["comparison_against_author_labels"]["typed"]["accuracy_including_abstentions"],
        "author_label_accuracy_open_text": result["comparison_against_author_labels"]["open_text"]["accuracy_including_abstentions"],
        "human_calibrated_accuracy": None,
    }, indent=2))


if __name__ == "__main__":
    main()
