"""Blind independent human-review packets and agreement analysis.

No reviewer labels are fabricated. Report human-calibrated metrics ONLY when
two complete independent human submissions exist and have been validated.
Disagreements remain unresolved until explicit independent adjudication.
"""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path

from recsys.evaluation.atomic_claims import FOUR_LABELS, load_frozen_claims
from recsys.evaluation.model_critic import build_evidence_context
from recsys.evaluation.source_audit import SourceFixture


HEADERS = (
    "case_id", "kind", "product_id", "reason", "claim_span",
    "source_context_sha256", "source_context_json",
    "reviewer_id", "reviewer_label", "review_notes",
)


def build_review_packet(output_dir: Path) -> dict:
    suite, _, guard, provenance = load_frozen_claims()
    from recsys.evaluation.atomic_claims import ROOT
    fixture = SourceFixture.model_validate_json((ROOT / suite.sources_path).read_bytes())
    rows = []
    for case in suite.cases:
        context = build_evidence_context(case, fixture, guard._evidence)
        rows.append({
            "case_id": case.case_id,
            "kind": case.kind,
            "product_id": case.product_id,
            "reason": case.reason,
            "claim_span": case.claim_span,
            "source_context_sha256": sha256(context.encode("utf-8")).hexdigest(),
            "source_context_json": context,
            "reviewer_id": "",
            "reviewer_label": "",
            "review_notes": "",
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    for slot in ("A", "B"):
        path = output_dir / f"stage35-blind-reviewer-{slot}.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=HEADERS)
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "suite_id": suite.suite_id,
        "cases": len(rows),
        "source_provenance": provenance,
        "reviewer_submissions_received": 0,
        "independent_human_agreement": None,
        "human_calibrated_model_accuracy": None,
        "instructions": (
            "Assign reviewers A and B separately; each labels all 26 cases "
            "using supported/contradicted/unsupported/ambiguous. Do not "
            "share authored labels, model predictions or each other's labels. "
            "Record reviewer_id and reason for each judgment, resolve "
            "disagreements only through separately documented adjudication."
        ),
    }
    (output_dir / "stage35-human-review-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8"
    )
    return manifest


def _load_submitted_review(path: Path, expected: dict, *, seen_reviewer_ids: set[str]) -> dict:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(HEADERS):
            raise ValueError("review CSV header does not match blind packet")
        rows = list(reader)
    if len(rows) != len(expected) or len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("review must contain all distinct fixed cases")
    by_case = {row["case_id"]: row for row in rows}
    if set(by_case) != set(expected):
        raise ValueError("review case ID mismatch")
    ids = {row["reviewer_id"].strip() for row in rows}
    if len(ids) != 1 or not next(iter(ids)):
        raise ValueError("review requires one nonempty reviewer_id")
    reviewer_id = next(iter(ids))
    if reviewer_id in seen_reviewer_ids:
        raise ValueError("two submissions must use distinct reviewer IDs")
    seen_reviewer_ids.add(reviewer_id)
    for case_id, row in by_case.items():
        ref = expected[case_id]
        if any(row[key] != ref[key] for key in (
            "case_id", "kind", "product_id", "reason", "claim_span",
            "source_context_sha256", "source_context_json",
        )):
            raise ValueError("review source context was modified")
        if row["reviewer_label"] not in FOUR_LABELS:
            raise ValueError("review must contain exactly four-class completed labels")
        if not row["review_notes"].strip():
            raise ValueError("review notes must justify each judgment")
    return {case_id: row["reviewer_label"] for case_id, row in by_case.items()}


def assess_independent_reviews(
    a_path: Path, b_path: Path, *, model_report: Path | None = None,
) -> dict:
    """Only process complete, separately signed reviews; no gold-label access."""
    suite, _, guard, provenance = load_frozen_claims()
    from recsys.evaluation.atomic_claims import ROOT
    fixture = SourceFixture.model_validate_json((ROOT / suite.sources_path).read_bytes())
    expected = {}
    for case in suite.cases:
        context = build_evidence_context(case, fixture, guard._evidence)
        expected[case.case_id] = {
            "case_id": case.case_id, "kind": case.kind,
            "product_id": case.product_id,
            "reason": case.reason, "claim_span": case.claim_span,
            "source_context_sha256": sha256(context.encode()).hexdigest(),
            "source_context_json": context,
        }
    reviewer_ids = set()
    a = _load_submitted_review(a_path, expected, seen_reviewer_ids=reviewer_ids)
    b = _load_submitted_review(b_path, expected, seen_reviewer_ids=reviewer_ids)
    confusion = {label: {other: 0 for other in FOUR_LABELS} for label in FOUR_LABELS}
    for case_id in expected:
        confusion[a[case_id]][b[case_id]] += 1
    agree = sum(a[case_id] == b[case_id] for case_id in expected)
    total = len(expected)
    pa = agree / total
    a_freq = Counter(a.values())
    b_freq = Counter(b.values())
    chance = sum(a_freq[label] * b_freq[label] / total**2 for label in FOUR_LABELS)
    kappa = (pa - chance)/(1 - chance) if chance != 1 else None
    consensus = {
        case_id: a[case_id] for case_id in expected if a[case_id] == b[case_id]
    }
    report = {
        "suite_id": suite.suite_id,
        "total_reviewed": total,
        "independent_reviewer_count": 2,
        "independent_agreement": pa,
        "cohens_kappa": kappa,
        "consensus_cases": len(consensus),
        "disagreement_cases": [cid for cid in expected if cid not in consensus],
        "confusion_reviewer_a_by_b": confusion,
        "source_provenance": provenance,
        "fully_adjudicated": len(consensus) == total,
        "human_calibrated_model_accuracy": None,
        "notes": "Two signed complete review sheets validated; disagreements are not automatically resolved.",
    }
    if model_report is not None:
        obj = json.loads(model_report.read_bytes())
        if obj.get("suite_id") != suite.suite_id or obj.get("source_sha256") != provenance:
            raise ValueError("model prediction provenance differs from human review")
        predictions = obj["generator"]["predictions"]
        if len(predictions) != total or len({r["case_id"] for r in predictions}) != total:
            raise ValueError("model prediction set incomplete")
        by_id = {r["case_id"]: r for r in predictions}
        if set(by_id) != set(expected):
            raise ValueError("model case IDs differ")
        valid = [cid for cid in consensus if by_id[cid]["prediction"] in FOUR_LABELS]
        matches = sum(by_id[cid]["prediction"] == consensus[cid] for cid in valid)
        report["model_consensus_subset"] = {
            "consensus_cases": len(consensus),
            "valid_predictions_on_consensus_cases": len(valid),
            "matches": matches,
            "accuracy_including_invalid_on_consensus_cases": matches/len(consensus) if consensus else None,
            "selective_accuracy": matches/len(valid) if valid else None,
            "not_final_human_calibrated_accuracy": bool(len(consensus) < total),
        }
        if len(consensus) == total:
            report["human_calibrated_model_accuracy"] = matches / total
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path,
                        default=Path("artifacts/evaluation/stage35-blind-review"))
    parser.add_argument("--review-a", type=Path)
    parser.add_argument("--review-b", type=Path)
    parser.add_argument("--model-report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/stage35-human-agreement.json"))
    args = parser.parse_args()
    if (args.review_a is None) != (args.review_b is None):
        parser.error("Both independent review sheets are required for analysis")
    if args.review_a is None:
        print(json.dumps(build_review_packet(args.packet_dir), indent=2))
    else:
        result = assess_independent_reviews(
            args.review_a, args.review_b, model_report=args.model_report,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
