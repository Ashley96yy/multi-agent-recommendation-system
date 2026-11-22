"""Stage 4.0 reproducible E2E smoke: six scripted cases, optional REAL Qwen.

Scripted results only verify orchestrator behavior and are NOT LLM metrics.
The optional real case is a smoke test, not a 25-case quality benchmark.
"""

import argparse
import hashlib
import json
from pathlib import Path

from recsys.demo.service import (
    DemoService, WALKTHROUGHS_PATH, preferences_from_input,
)


def run_checks(*, real_qwen_smoke: bool = False) -> dict:
    service = DemoService()
    rows = []
    for case in service.suite.cases:
        record = service.run_walkthrough(case.scenario_id)
        rows.append({
            "scenario_id": case.scenario_id,
            "status": record["status"],
            "expected_status": case.expected_status,
            "generator_calls": record["actual_backend_calls"],
            "expected_calls": case.expected_generator_calls,
            "revision_used": record["revision_used"],
            "critic_issue_codes": [
                issue["code"] for attempt in record["attempts"]
                for issue in attempt["critic_issues"]
            ],
            "source_statuses": sorted({
                f["status"] for f in record["source_field_diagnostics"]
            }),
            "final_recommendations": [
                rec["product_id"] for rec in record["final_approved_recommendations_only"]
            ],
            "human_semantic_groundedness_measured": False,
        })
    if len(rows) != 6:
        raise ValueError("exactly six fixed scripted smoke scenarios expected")
    required_statuses = {
        "consistent": "approved", "revision": "approved",
        "conflicting": "insufficient_evidence",
        "superseded": "insufficient_evidence",
        "missing": "insufficient_evidence",
        "hard_filter": "no_candidates",
    }
    if {r["scenario_id"]: r["status"] for r in rows} != required_statuses:
        raise ValueError("scripted scenario regression detected")
    if {r["scenario_id"]: r["generator_calls"] for r in rows} != {
        "consistent": 1, "revision": 2,
        "conflicting": 0, "superseded": 0, "missing": 0,
        "hard_filter": 0,
    }:
        raise ValueError("scripted generation call regression")
    result = {
        "suite_id": service.suite.fixture_id,
        "walkthroughs_sha256": hashlib.sha256(WALKTHROUGHS_PATH.read_bytes()).hexdigest(),
        "source_catalog_sha256": service.retriever.catalog_sha256,
        "source_documents_sha256": service.guard.source_docs_sha256,
        "scripted_mode_not_real_llm": True,
        "scripted_scenarios": rows,
        "scripted_scenarios_passed": len(rows),
        "real_qwen_smoke": None,
        "human_semantic_groundedness": None,
        "real_world_product_verification": False,
    }
    if real_qwen_smoke:
        prefs = preferences_from_input(
            request="Cloud Calm Facial Cream fragrance-free moisturizer",
            required_category="moisturizer", required_attributes=["fragrance_free"],
        )
        one = service.run_live(prefs, max_revisions=0)
        if one["mode"] != "live_qwen" or one["actual_backend_calls"] != 1:
            raise ValueError("real Qwen smoke expected one physical inference")
        if one["logical_generator_attempts"] != 1:
            raise ValueError("unexpected extra real Qwen proposal")
        # Retain full actual raw text, even when unapproved. Do not rewrite.
        result["real_qwen_smoke"] = one
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-qwen-smoke", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/evaluation/stage40-demo-e2e.json"))
    args = parser.parse_args()
    report = run_checks(real_qwen_smoke=args.real_qwen_smoke)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "scripted_scenarios_passed": report["scripted_scenarios_passed"],
        "scripted_statuses": {
            row["scenario_id"]: row["status"]
            for row in report["scripted_scenarios"]
        },
        "real_qwen_calls": report["real_qwen_smoke"]["actual_backend_calls"]
        if report["real_qwen_smoke"] is not None else 0,
        "real_qwen_status": report["real_qwen_smoke"]["status"]
        if report["real_qwen_smoke"] is not None else "not_requested",
        "human_semantic_groundedness": None,
    }, indent=2))


if __name__ == "__main__":
    main()
