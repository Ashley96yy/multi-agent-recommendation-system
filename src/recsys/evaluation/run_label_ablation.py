"""Stage 3.6: real Qwen 26-case prompt and TOKEN-CONSTRAINT ablation.

Three NEW independent real-model arms:
  plain_label: short prompt, strict bare-label contract
  decision_label: improved decision-tree prompt, same strict contract
  decision_label_constrained: SAME improved prompt + true token mask

Historical Stage 3.5 JSON+explanation results stay untouched and are
descriptive only (different output contract); no fake JSON rationale is added.
Author labels are accessed only after generating all raw outputs.
"""

import argparse
from hashlib import sha256
import importlib.metadata
import json
import platform
from pathlib import Path
from time import perf_counter

from recsys.baseline.local_qwen import LocalQwenGenerator, MODEL_ID, MODEL_REVISION
from recsys.evaluation.atomic_claims import FOUR_LABELS, load_frozen_claims
from recsys.evaluation.label_constraints import ConstrainedLabelQwen
from recsys.evaluation.model_critic import build_evidence_context, evaluate_predictions
from recsys.evaluation.source_audit import SourceFixture


PLAIN_PROMPT = """You are auditing a fictional product claim against its cited
documents. Label the claim based only on the current sources, respecting
document revisions. Choose one exact lowercase word:
supported
contradicted
unsupported
ambiguous
Return only that label, without JSON, punctuation or explanation."""

DECISION_PROMPT = """Task: classify ONE fictional product claim against the
provided primary source documents. Ignore general knowledge and stale
catalog metadata. Prefer active source text to superseded versions.
Decision order:
1. Active documents disagree on the same relevant fact -> ambiguous.
2. Active documents clearly assert the opposite fact -> contradicted.
3. No active passage asserts the fact, or a citation is absent/invalid/
wrong field/wrong product/superseded -> unsupported.
4. Active passages agree with the claim and ALL citations match the
same product, relevant field and active evidence -> supported.
Medical efficacy without source evidence is unsupported.
Output ONE and ONLY ONE of these exact lowercase words:
supported
contradicted
unsupported
ambiguous
No JSON, markdown, extra words, rationale or punctuation."""

ARM_CONFIG = (
    ("plain_label", PLAIN_PROMPT, False),
    ("decision_label", DECISION_PROMPT, False),
    ("decision_label_constrained", DECISION_PROMPT, True),
)


def parse_bare_label(raw: str) -> tuple[str, str | None]:
    """No normalization, no synonyms, no JSON rescue and no autocorrection."""
    if isinstance(raw, str) and raw in FOUR_LABELS:
        return raw, None
    return "needs_human_review", "not_an_exact_canonical_label"


def generate_arms_without_labels(suite, fixture, evidence_index, model) -> dict:
    """No reference label is ever passed to a generator or used in prompts."""
    constrained = ConstrainedLabelQwen(model)
    initial_count = model.call_count
    rows = {arm: [] for arm, _, _ in ARM_CONFIG}
    for case in suite.cases:
        context = build_evidence_context(case, fixture, evidence_index)
        for arm, prompt, use_constraint in ARM_CONFIG:
            generator = constrained if use_constraint else model
            before = model.call_count
            start = perf_counter()
            raw = None
            issue = None
            try:
                raw = generator.generate(prompt, context)
                prediction, issue = parse_bare_label(raw)
            except Exception as exc:
                prediction = "needs_human_review"
                issue = type(exc).__name__
            used = model.call_count - before
            if used not in (0, 1):
                raise ValueError("hidden extra model call detected")
            rows[arm].append({
                "case_id": case.case_id,
                "kind": case.kind,
                "prediction": prediction,
                "error": issue,
                "raw_model_response": raw,
                "model_calls": used,
                "prompt_tokens": model.last_prompt_tokens if used else 0,
                "generated_tokens": model.last_generated_tokens if used else 0,
                "generation_seconds": round(model.last_generation_seconds, 5) if used else 0,
                "wall_seconds": round(perf_counter() - start, 5),
                "prompt_sha256": sha256(prompt.encode()).hexdigest(),
                "context_sha256": sha256(context.encode()).hexdigest(),
                "constraint": (
                    "transformers_prefix_allowed_tokens_fn_trie"
                    if use_constraint else "none"
                ),
                "raw_output_contract": "one_exact_lowercase_label_no_json",
            })
    count = model.call_count - initial_count
    if count != sum(row["model_calls"] for cases in rows.values() for row in cases):
        raise ValueError("actual model invocation accounting mismatch")
    return {
        "experiment_type": "new_independent_26_case_prompt_and_token_constraint_ablation",
        "fixed_case_count": len(suite.cases),
        "physical_model_invocations": count,
        "model": model.name,
        "four_allowed_labels": list(FOUR_LABELS),
        "arms": {
            arm: {
                "prompt_sha256": sha256(prompt.encode()).hexdigest(),
                "uses_true_token_constraint": constrained_flag,
                "contract": "canonical_label_only_not_json",
                "physical_model_invocations": sum(r["model_calls"] for r in rows[arm]),
                "valid_label_outputs": sum(r["prediction"] in FOUR_LABELS for r in rows[arm]),
                "invalid_outputs": sum(r["prediction"] not in FOUR_LABELS for r in rows[arm]),
                "prompt_tokens": sum(r["prompt_tokens"] for r in rows[arm]),
                "generated_tokens": sum(r["generated_tokens"] for r in rows[arm]),
                "generation_seconds": round(sum(r["generation_seconds"] for r in rows[arm]), 4),
                "per_case": rows[arm],
            }
            for arm, prompt, constrained_flag in ARM_CONFIG
        },
    }


def score_after_inference(suite, labels, generated) -> dict:
    """Gold labels joined only after the 78 raw model outputs are recorded."""
    result = {}
    for arm, output in generated["arms"].items():
        prediction_data = {"predictions": output["per_case"]}
        result[arm] = evaluate_predictions(suite, labels, prediction_data)
    return result


def run_real() -> dict:
    suite, author_labels, guard, provenance = load_frozen_claims()
    from recsys.evaluation.atomic_claims import ROOT
    fixture = SourceFixture.model_validate_json((ROOT / suite.sources_path).read_bytes())
    model = LocalQwenGenerator(
        model_id=MODEL_ID, revision=MODEL_REVISION, device="cpu",
        max_new_tokens=32, max_prompt_tokens=3000,
    )
    generated = generate_arms_without_labels(suite, fixture, guard._evidence, model)
    scores = score_after_inference(suite, author_labels, generated)
    return {
        "suite_id": suite.suite_id,
        "source_sha256": provenance,
        "generation": generated,
        "author_label_comparison": scores,
        "environment": {
            "python": platform.python_version(),
            "transformers": importlib.metadata.version("transformers"),
            "torch": importlib.metadata.version("torch"),
            "pydantic": importlib.metadata.version("pydantic"),
            "device": "cpu",
            "decode": "greedy",
        },
        "human_calibrated_accuracy": None,
        "real_world_groundedness": None,
        "additional_human_reviews": 0,
        "limitations": (
            "Three new 26-case real Qwen generation arms use unchanged "
            "fictional cases and author-written reference labels. Only "
            "canonical label tokens, not rationale or full JSON, are "
            "constrained. Prompt difference is isolated in two free-label "
            "arms; constraint difference is isolated on the SAME decision "
            "prompt. Historical Stage 3.5 JSON baseline differs in output "
            "contract and cannot support a causal comparison. These "
            "observations are not independently human-calibrated."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/evaluation/stage36-real-qwen-label-ablation.json"),
    )
    args = parser.parse_args()
    report = run_real()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)+"\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "physical_model_invocations": report["generation"]["physical_model_invocations"],
        "arms": {
            key: {
                "valid_labels": value["valid_label_outputs"],
                "invalid_labels": value["invalid_outputs"],
                "author_correct_overall": report["author_label_comparison"][key]["all"]["correct_overall"],
                "author_correct_typed": report["author_label_comparison"][key]["typed"]["correct_overall"],
                "author_correct_open_text": report["author_label_comparison"][key]["open_text"]["correct_overall"],
                "tokens": value["prompt_tokens"] + value["generated_tokens"],
                "generation_seconds": value["generation_seconds"],
            }
            for key, value in report["generation"]["arms"].items()
        },
        "human_calibrated_accuracy": None,
    }, indent=2))


if __name__ == "__main__":
    main()
