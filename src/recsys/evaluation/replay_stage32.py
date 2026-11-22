"""Post-hoc Stage 3.2 analysis of *recorded* real Qwen first drafts.

No new LLM calls, no changed prompts. Observed retrieval candidate IDs are
reused verbatim; placeholder similarity=1 is a schema-only value and is
NEVER interpreted/reported as a new retrieval score.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from recsys.baseline.contracts import SingleAgentAnswer
from recsys.evaluation.baseline_suite import load_baseline_suite
from recsys.multiagent.agents import CriticVerificationAgent
from recsys.multiagent.output_policy import normalize_json_response
from recsys.retrieval import CatalogRetriever
from recsys.schemas.retrieval import RetrievalResult, RetrievedProduct


ROOT = Path(__file__).resolve().parents[3]
SOURCE_RUN_ID = 45053525292


def replay_first_drafts(report: dict, *, catalog_path: Path, suite_path: Path,
                        source_sha256: str) -> dict:
    catalog_retriever = CatalogRetriever.from_json(catalog_path)
    cases, hashes = load_baseline_suite(
        suite_path, repo_root=ROOT, retriever=catalog_retriever
    )
    if report.get("data_hashes") != hashes:
        raise ValueError("recorded model experiment and replay inputs differ")
    if report.get("physical_model_invocations") != 35:
        raise ValueError("recorded paired model experiment expected 35 physical calls")
    rows = report.get("zero_revision", {}).get("per_case", [])
    if len(rows) != 25 or {x["case_id"] for x in rows} != {c.case_id for c in cases}:
        raise ValueError("original 25 case IDs do not match")
    row_by_case = {x["case_id"]: x for x in rows}
    product_map = {p.product_id: p for p in catalog_retriever.catalog.products}
    evidence_by_product: dict[str, list] = {
        pid: [e for e in catalog_retriever.catalog.evidence if e.product_id == pid]
        for pid in product_map
    }
    critic = CriticVerificationAgent()
    scores = {policy: [] for policy in ("strict", "safe_json_normalization")}
    preflight = []
    for case in cases:
        row = row_by_case[case.case_id]
        calls = row["calls"]
        if row["logical_generator_attempts"] == 0:
            if calls or case.expected_type != "no_match":
                raise ValueError("unexpected zero-call record in frozen source")
            for policy in scores:
                scores[policy].append({
                    "case_id": case.case_id, "status": "no_candidates",
                    "approved": False, "passed": True, "format_actions": [],
                    "critic_codes": [],
                })
            continue
        if len(calls) != 1 or not isinstance(calls[0].get("raw_output"), str):
            raise ValueError("missing original first-draft model output")
        ids = row["retrieved_product_ids"]
        if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(product_map):
            raise ValueError("invalid recorded candidate IDs")
        by_reason = Counter()
        eligible = 0
        for product in product_map.values():
            reason = catalog_retriever._filter_reason(product, case.preferences)
            if reason:
                by_reason[reason] += 1
            else:
                eligible += 1
        if not set(ids) <= {
            pid for pid, product in product_map.items()
            if catalog_retriever._filter_reason(product, case.preferences) is None
        }:
            raise ValueError("recorded candidates violate metadata constraints")
        retrieval = RetrievalResult(
            query=case.preferences.request,
            embedder_name="recorded_MinILM_candidate_ids",
            catalog_sha256=hashes["catalog_sha256"],
            indexed_products=len(product_map),
            eligible_products=eligible,
            filtered_by_reason=dict(by_reason),
            # Placeholder score needed by the existing RetrievedProduct
            # contract. This experiment does NOT measure similarity.
            candidates=[
                RetrievedProduct(product=product_map[pid], similarity=1.0,
                                 evidence=evidence_by_product[pid])
                for pid in ids
            ],
        )
        required_fields = {f"attributes.{key}" for key in case.preferences.required_attributes}
        has_any_coverage = (
            not required_fields
            or any(required_fields <= {e.supported_field for e in c.evidence}
                   for c in retrieval.candidates)
        )
        preflight.append({
            "case_id": case.case_id,
            "missing_required_field_citations": not has_any_coverage,
            "could_skip_before_generation": not has_any_coverage,
            "claim_entailment_verified": False,
        })
        for policy in scores:
            normalized, actions = normalize_json_response(calls[0]["raw_output"], policy)
            try:
                answer = SingleAgentAnswer.model_validate_json(normalized)
                verdict = critic.review(answer, case.preferences, retrieval)
            except (ValidationError, ValueError):
                answer = None
                verdict = critic.review(None, case.preferences, retrieval, invalid_generation=True)
            approved = bool(verdict.approved and answer and answer.recommendations)
            ids_out = [x.product_id for x in answer.recommendations] if approved else []
            phrases = " ".join(x.reason for x in answer.recommendations).casefold() if approved else ""
            leaked = any(p.casefold() in phrases for p in case.forbidden_phrases)
            passed = approved and set(ids_out).issubset(case.expected_product_ids) and not leaked
            scores[policy].append({
                "case_id": case.case_id,
                "status": "approved" if approved else "rejected",
                "approved": approved,
                "passed": bool(passed),
                "format_actions": actions,
                "critic_codes": [i.code for i in verdict.issues],
            })
    result = {}
    for policy, records in scores.items():
        generated = [x for x in records if x["status"] != "no_candidates"]
        result[policy] = {
            "case_count": len(records),
            "accepted_generation_count": sum(x["approved"] for x in generated),
            "positive_success_count": sum(x["passed"] for x in generated),
            "no_match_zero_call_count": sum(x["status"] == "no_candidates" for x in records),
            "critic_issue_counts": dict(Counter(code for x in generated for code in x["critic_codes"])),
            "normalization_actions": dict(Counter(action for x in records
                                                  for action in x["format_actions"])),
            "per_case": records,
        }
    strict = result["strict"]
    if strict["accepted_generation_count"] != report["zero_revision"]["approved_count"]:
        raise ValueError("strict replay must exactly match original critic approval")
    if strict["positive_success_count"] != report["zero_revision"]["positive_success_count"]:
        raise ValueError("strict replay must exactly match original strict success")
    return {
        "source_run_id": SOURCE_RUN_ID,
        "source_report_sha256": source_sha256,
        "source_data_hashes": hashes,
        "recorded_first_draft_count": len(preflight),
        "posthoc_preflight_potential_zero_call_count": sum(
            x["could_skip_before_generation"] for x in preflight
        ),
        "posthoc_preflight_details": preflight,
        "strategies": result,
        "additional_model_invocations": 0,
        "semantic_citation_groundedness": None,
        "unsupported_claim_rate": None,
        "limitation": (
            "Frozen first-draft re-evaluation only; no new model inference. "
            "Preflight exclusions are hypothetical: changing candidate sets "
            "would change live prompts. Normalized outputs were never regenerated. "
            "Valid citation-field IDs do not prove source claim entailment."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-report", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/retrieval_eval/catalog.json")
    parser.add_argument("--suite", type=Path, default=ROOT / "data/baseline_eval/suite.json")
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/evaluation/stage32-recorded-first-draft-replay.json"))
    args = parser.parse_args()
    raw = args.original_report.read_bytes()
    report = replay_first_drafts(
        json.loads(raw), catalog_path=args.catalog, suite_path=args.suite,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "recorded_first_draft_count": report["recorded_first_draft_count"],
        "posthoc_preflight_potential_zero_call_count": report["posthoc_preflight_potential_zero_call_count"],
        "strategies": {
            key: {name: value[name] for name in (
                "accepted_generation_count", "positive_success_count",
                "no_match_zero_call_count", "critic_issue_counts", "normalization_actions"
            )}
            for key, value in report["strategies"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
