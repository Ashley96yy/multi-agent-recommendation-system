"""Prepare a transparent human-review queue from real synthetic benchmark outputs.

This exports candidate recommendation reasons and source excerpts; it does not
evaluate semantic truth or automatically assign factual-grounding scores.
"""

import argparse
import csv
import json
from pathlib import Path

from recsys.retrieval.catalog import load_catalog


FIELDS = [
    "case_id", "group", "product_id", "reason",
    "citation_ids", "cited_evidence_json", "atomic_claim_span",
    "support_label", "reviewer_id", "review_notes",
]


def build_review_queue(report: dict, catalog_path: Path) -> list[dict[str, str]]:
    catalog, sha = load_catalog(catalog_path)
    if report.get("data_hashes", {}).get("catalog_sha256") != sha:
        raise ValueError("review catalog does not match benchmark provenance SHA")
    sources = {e.evidence_id: e for e in catalog.evidence}
    queue = []
    for row in report["per_case"]:
        if row["status"] != "accepted":
            continue
        raw = row.get("raw_output")
        if not isinstance(raw, str):
            raise ValueError(f"{row['case_id']}: accepted result missing actual model output")
        answer = json.loads(raw)
        for item in answer.get("recommendations", []):
            ids = item["evidence_ids"]
            if not ids or any(eid not in sources or sources[eid].product_id != item["product_id"]
                              for eid in ids):
                raise ValueError("source reference is missing or belongs to wrong product")
            cited = [
                {
                    "evidence_id": eid,
                    "source_id": sources[eid].source_id,
                    "supported_field": sources[eid].supported_field,
                    "excerpt": sources[eid].excerpt,
                }
                for eid in ids
            ]
            queue.append({
                "case_id": row["case_id"],
                "group": row["group"],
                "product_id": item["product_id"],
                "reason": item["reason"],
                "citation_ids": json.dumps(ids, ensure_ascii=False),
                "cited_evidence_json": json.dumps(cited, ensure_ascii=False),
                "atomic_claim_span": "",
                "support_label": "",
                "reviewer_id": "",
                "review_notes": "",
            })
    return queue


def export_review_csv(report_path: Path, catalog_path: Path, output: Path) -> int:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = build_review_queue(report, catalog_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("data/retrieval_eval/catalog.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = export_review_csv(args.report, args.catalog, args.output)
    print(f"Human-review queue exported: {count} unadjudicated recommendations")


if __name__ == "__main__":
    main()
