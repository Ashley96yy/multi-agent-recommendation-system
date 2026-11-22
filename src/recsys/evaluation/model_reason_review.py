"""Blind claim-review queue extracted from RECORDED REAL Qwen reasons.

Does not label, split, infer or verify any free-form factual statement. The
goal is to allow independent reviewers to extract all atomic claims from the
actual answers rather than pretending a structural citation implies support.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from recsys.baseline.contracts import SingleAgentAnswer
from recsys.multiagent.output_policy import normalize_json_response
from recsys.retrieval.catalog import load_catalog


EXPECTED_ORIGINAL_RUN = 45056291366
ARM = "no_preflight__safe_json_normalization"


def export_model_review(
    report_path: Path, catalog_path: Path, output_path: Path,
    manifest_path: Path,
) -> dict:
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    catalog, catalog_sha = load_catalog(catalog_path)
    if report.get("suite_id") != "single-agent-baseline-synthetic-25-v1":
        raise ValueError("real model source report is not the frozen 25-case suite")
    if report.get("data_hashes", {}).get("catalog_sha256") != catalog_sha:
        raise ValueError("model output provenance differs from catalog")
    if report.get("matrix", {}).get("physical_model_invocations") != 19:
        raise ValueError("source must record 19 physical Qwen first drafts")
    arm = report["matrix"]["arms"][ARM]
    if arm["case_count"] != 25 or arm["critic_approved_count"] != 9:
        raise ValueError("unexpected original real Qwen evaluation configuration")
    cases = arm["per_case"]
    if len(cases) != 25 or len({c["case_id"] for c in cases}) != 25:
        raise ValueError("original real Qwen case records incomplete")
    evidence = {e.evidence_id: e for e in catalog.evidence}
    fields = [
        "case_id", "product_id", "model_reason", "cited_evidence_json",
        "source_run_id", "model_version", "atomic_claim_span", "atomic_claim_field",
        "reviewer_label", "reviewer_id", "review_notes",
    ]
    review_rows = []
    approved = [row for row in cases if row["status"] == "approved"]
    for row in approved:
        raw = row.get("raw_first_output")
        if not isinstance(raw, str):
            raise ValueError("approved model response has no raw text")
        normalized, _ = normalize_json_response(raw, "safe_json_normalization")
        answer = SingleAgentAnswer.model_validate_json(normalized)
        if sorted(x.product_id for x in answer.recommendations) != sorted(row["recommended_ids"]):
            raise ValueError("reported IDs differ from the actual model output")
        for rec in answer.recommendations:
            excerpts = []
            for eid in rec.evidence_ids:
                e = evidence.get(eid)
                if e is None or e.product_id != rec.product_id:
                    raise ValueError("approved model response has invalid or cross-product citation")
                excerpts.append({
                    "evidence_id": eid, "source_id": e.source_id,
                    "supported_field": e.supported_field, "excerpt": e.excerpt,
                })
            review_rows.append({
                "case_id": row["case_id"],
                "product_id": rec.product_id,
                "model_reason": rec.reason,
                "cited_evidence_json": json.dumps(excerpts, ensure_ascii=False),
                "source_run_id": EXPECTED_ORIGINAL_RUN,
                "model_version": report["generator"],
                "atomic_claim_span": "", "atomic_claim_field": "",
                "reviewer_label": "", "reviewer_id": "", "review_notes": "",
            })
    if len(approved) != 9 or not review_rows:
        raise ValueError("expected nine approved Qwen cases for this recorded run")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review_rows)
    manifest = {
        "source_run_id": EXPECTED_ORIGINAL_RUN,
        "source_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "catalog_sha256": catalog_sha,
        "arm": ARM,
        "approved_case_count": len(approved),
        "recommendation_reason_rows": len(review_rows),
        "atomic_claims_manually_extracted": 0,
        "semantic_groundedness": None,
        "unsupported_claim_rate": None,
        "limitations": (
            "One row per approved real-Qwen recommendation reason, NOT per "
            "atomic factual claim. Blind reviewer must split every reason "
            "into all factual claims, judge cited evidence, and document omissions."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-report", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("data/retrieval_eval/catalog.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/evaluation/stage34-real-qwen-blind-review.csv"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("artifacts/evaluation/stage34-real-qwen-review-manifest.json"))
    args = parser.parse_args()
    print(json.dumps(export_model_review(
        args.real_report, args.catalog, args.output, args.manifest,
    ), indent=2))


if __name__ == "__main__":
    main()
