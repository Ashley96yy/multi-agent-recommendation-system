"""Post-hoc deterministic format-repair ablation on FROZEN Qwen outputs.

This analysis DOES NOT call a model or change product IDs, recommendation
reason text, citation IDs, retrieval candidates, or source evidence.
Its purpose is to isolate mechanical format compliance from factual quality.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from pydantic import ValidationError

from recsys.baseline.contracts import SingleAgentAnswer
from recsys.evaluation.baseline_suite import load_baseline_suite
from recsys.retrieval import CatalogRetriever

ROOT = Path(__file__).resolve().parents[3]
SOURCE_RUN_ID = 45050806308
SOURCE_COMMIT_SHA = "897fc21841487a8d375aec27d8df314f28f13813"
SOURCE_ARTIFACT_SHA256 = "e6b45b6abf77bff7fdec3167b3b0d4fe82605ca21f2c5868c892040cdfae2635"

MODES = ("strict", "fences_only", "fences_and_missing_flag")
EXACT_CODE_FENCE = re.compile(
    r"\A\s*```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```\s*\Z",
    flags=re.IGNORECASE,
)


def normalize_output(raw: str, mode: str) -> tuple[str, list[str]]:
    """Never rewrite the recommendation contents: only unwrap an exact
    whole-response JSON fence and/or add a provable absent metadata field.
    """
    if mode not in MODES:
        raise ValueError("unsupported ablation mode")
    if not isinstance(raw, str):
        raise TypeError("model output must be text")
    actions = []
    normalized = raw
    if mode != "strict":
        match = EXACT_CODE_FENCE.fullmatch(normalized)
        if match:
            normalized = match.group("body")
            actions.append("strip_whole_json_fence")
    if mode == "fences_and_missing_flag":
        try:
            obj = json.loads(normalized)
        except json.JSONDecodeError:
            # Never attempt fuzzy JSON repair or infer a missing closing brace.
            return normalized, actions
        if (
            isinstance(obj, dict)
            and "no_suitable_match" not in obj
            and isinstance(obj.get("recommendations"), list)
            and bool(obj["recommendations"])
        ):
            obj["no_suitable_match"] = False
            normalized = json.dumps(obj, ensure_ascii=False)
            actions.append("set_missing_no_suitable_match_false")
    return normalized, actions


def _check_output(
    raw: str, mode: str, allowed_ids: list[str], evidence_by_product: dict[str, set[str]],
) -> tuple[SingleAgentAnswer | None, list[str], str | None]:
    normalized, actions = normalize_output(raw, mode)
    try:
        answer = SingleAgentAnswer.model_validate_json(normalized)
    except ValidationError:
        return None, actions, "invalid_schema_or_json"
    for recommendation in answer.recommendations:
        if recommendation.product_id not in allowed_ids:
            return None, actions, "product_outside_retrieval"
        if not set(recommendation.evidence_ids) <= evidence_by_product.get(recommendation.product_id, set()):
            return None, actions, "absent_or_cross_product_citation"
    return answer, actions, None


def replay_ablation(source_report: dict, cases: list, catalog, *, source_json_sha256: str) -> dict:
    """All arms replay exact same model outputs, candidate IDs and labels."""
    if source_report.get("case_count") != 25 or source_report.get("model_invocations") != 19:
        raise ValueError("source is not the frozen 25-case Qwen baseline")
    case_by_id = {case.case_id: case for case in cases}
    source_rows = source_report["per_case"]
    if len(source_rows) != 25 or {r["case_id"] for r in source_rows} != set(case_by_id):
        raise ValueError("source and baseline suite case IDs do not match")
    evidence_by_product: dict[str, set[str]] = {
        product.product_id: {record.evidence_id for record in catalog.evidence
                             if record.product_id == product.product_id}
        for product in catalog.products
    }
    all_modes = {}
    for mode in MODES:
        rows = []
        for row in source_rows:
            case = case_by_id[row["case_id"]]
            invocations = row.get("model_invocations")
            if invocations not in (0, 1):
                raise ValueError("unexpected model invocation count in frozen source")
            scored = {
                "case_id": case.case_id,
                "group": case.group,
                "expected_type": case.expected_type,
                "model_invocations": invocations,
                "format_actions": [],
                "accepted": False,
                "success": False,
                "error": None,
                "recommended_product_ids": [],
                "cited_evidence_ids": [],
                "forbidden_phrase_leaks": [],
            }
            if invocations == 0:
                if row.get("raw_output") is not None:
                    raise ValueError("zero-call case unexpectedly contains LLM output")
                scored["accepted"] = case.expected_type == "no_match" and row["status"] == "abstained"
                scored["success"] = scored["accepted"]
                if not scored["accepted"]:
                    scored["error"] = "unexpected_no_call_outcome"
            else:
                raw = row.get("raw_output")
                if not isinstance(raw, str):
                    raise ValueError("generated case has no raw model output")
                answer, actions, error = _check_output(
                    raw, mode, row["retrieved_product_ids"], evidence_by_product,
                )
                scored["format_actions"] = actions
                scored["error"] = error
                if answer is not None:
                    scored["accepted"] = bool(answer.recommendations)
                    scored["recommended_product_ids"] = [r.product_id for r in answer.recommendations]
                    scored["cited_evidence_ids"] = [eid for r in answer.recommendations for eid in r.evidence_ids]
                    reasons = " ".join(r.reason for r in answer.recommendations).casefold()
                    scored["forbidden_phrase_leaks"] = [
                        phrase for phrase in case.forbidden_phrases if phrase.casefold() in reasons
                    ]
                    if case.expected_type == "recommendation":
                        scored["success"] = (
                            scored["accepted"]
                            and set(scored["recommended_product_ids"]).issubset(case.expected_product_ids)
                            and not scored["forbidden_phrase_leaks"]
                        )
                    else:
                        # Do not credit unexpected generation as a valid no-call abstention.
                        scored["success"] = False
            rows.append(scored)

        normals = [r for r in rows if r["group"] == "normal"]
        red = [r for r in rows if r["group"] == "red_team"]
        positives = [r for r in rows if r["expected_type"] == "recommendation"]
        errors = Counter(r["error"] for r in rows if r["error"])
        actions = Counter(action for r in rows for action in r["format_actions"])
        all_modes[mode] = {
            "accepted_generated_count": sum(r["accepted"] for r in rows if r["model_invocations"]),
            "validation_errors": sum(bool(r["error"]) for r in rows),
            "error_types": dict(errors),
            "normal_success_count": sum(r["success"] for r in normals),
            "normal_success_rate": sum(r["success"] for r in normals) / len(normals),
            "red_team_success_count": sum(r["success"] for r in red),
            "red_team_success_rate": sum(r["success"] for r in red) / len(red),
            "positive_success_count": sum(r["success"] for r in positives),
            "positive_success_rate": sum(r["success"] for r in positives) / len(positives),
            "format_actions": dict(actions),
            "model_calls_added": 0,
            "per_case": rows,
        }

    strict = all_modes["strict"]
    # Match the original frozen baseline's strict evaluation before claiming
    # any post-hoc recovery. The corrected forbidden-phrase denominator is N/A.
    if (
        strict["accepted_generated_count"] != source_report["accepted_count"]
        or strict["validation_errors"] != source_report["generation_errors"]
        or abs(strict["normal_success_rate"] - source_report["normal_success_rate"]) > 1e-12
        or abs(strict["red_team_success_rate"] - source_report["red_team_success_rate"]) > 1e-12
    ):
        raise ValueError("strict replay does not reproduce original measured baseline")

    return {
        "source_run_id": SOURCE_RUN_ID,
        "source_experiment_commit": SOURCE_COMMIT_SHA,
        "source_report_json_sha256": source_json_sha256,
        "source_artifact_sha256": SOURCE_ARTIFACT_SHA256,
        "source_data_hashes": source_report["data_hashes"],
        "source_model_invocations": source_report["model_invocations"],
        "extra_model_invocations": 0,
        "source_prompt_tokens": source_report["tokens"]["prompt_total"],
        "source_generated_tokens": source_report["tokens"]["generated_total"],
        "modes": all_modes,
        "semantic_citation_groundedness": None,
        "unsupported_claim_rate": None,
        "disclaimer": (
            "Post-hoc mechanical repair of frozen synthetic model strings ONLY: "
            "not new model inference, verified claim entailment, or additional "
            "semantic recommendation accuracy."
        ),
    }


def run(*, original_report: Path, catalog_path: Path, manifest_path: Path) -> dict:
    from recsys.evaluation.baseline_suite import load_baseline_suite
    source_bytes = original_report.read_bytes()
    source = json.loads(source_bytes)
    retriever = CatalogRetriever.from_json(catalog_path)
    cases, hashes = load_baseline_suite(manifest_path, repo_root=ROOT, retriever=retriever)
    if source.get("data_hashes") != hashes:
        raise ValueError("source experiment does not match frozen catalog/label hashes")
    return replay_ablation(
        source, cases, retriever.catalog, source_json_sha256=hashlib.sha256(source_bytes).hexdigest(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-report", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/retrieval_eval/catalog.json")
    parser.add_argument("--suite", type=Path, default=ROOT / "data/baseline_eval/suite.json")
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/posthoc-format-ablation.json"))
    args = parser.parse_args()
    result = run(original_report=args.original_report, catalog_path=args.catalog,
                 manifest_path=args.suite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        mode: {
            k: arm[k] for k in ("accepted_generated_count", "validation_errors",
                                "normal_success_count", "red_team_success_count",
                                "positive_success_count", "format_actions")
        } for mode, arm in result["modes"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
