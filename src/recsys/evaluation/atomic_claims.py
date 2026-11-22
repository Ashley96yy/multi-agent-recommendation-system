"""Auditable Stage 3.4 claim-level evaluation on INVENTED source documents.

This is typed-assertion adjudication, NOT automatic extraction or semantic
entailment for free-form text. Open-text examples explicitly abstain for
independent review. Gold labels are author-written in a separate file.
"""

import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from recsys.multiagent.source_guard import TypedSourceGuard
from recsys.retrieval.catalog import load_catalog
from recsys.schemas.contracts import Contract


ROOT = Path(__file__).resolve().parents[3]
CASES = ROOT / "data/atomic_claim_eval/cases.json"
LABELS = ROOT / "data/atomic_claim_eval/author_labels.json"
FOUR_LABELS = ("supported", "contradicted", "unsupported", "ambiguous")
Outcome = Literal["supported", "contradicted", "unsupported", "ambiguous"]
Prediction = Literal["supported", "contradicted", "unsupported", "ambiguous", "needs_human_review"]


class AtomicClaimCase(Contract):
    case_id: str = Field(min_length=1)
    kind: Literal["typed", "open_text"]
    product_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    claim_span: str = Field(min_length=1)
    field: str | None = None
    asserted_value: bool | str | None = None
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_atomic_scope(self) -> "AtomicClaimCase":
        if self.claim_span not in self.reason:
            raise ValueError("atomic claim span must occur verbatim in reason")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("duplicate evidence ID in atomic claim")
        if any(not eid.strip() for eid in self.evidence_ids):
            raise ValueError("blank evidence ID")
        if self.kind == "typed":
            if self.field is None or not self.field.strip():
                raise ValueError("typed claim requires field")
            if type(self.asserted_value) not in (bool, str):
                raise ValueError("typed claim must declare explicit boolean or string value")
        elif self.field is not None or self.asserted_value is not None:
            raise ValueError("open-text claim must not pretend a typed field was extracted")
        return self


class AtomicClaimSuite(Contract):
    schema_version: Literal[1]
    suite_id: str = Field(min_length=1)
    catalog_path: str = Field(min_length=1)
    sources_path: str = Field(min_length=1)
    description: str = Field(min_length=1)
    cases: list[AtomicClaimCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_cases(self) -> "AtomicClaimSuite":
        if len(self.cases) != 26 or sum(c.kind == "typed" for c in self.cases) != 22:
            raise ValueError("frozen suite must contain 22 typed and four open-text cases")
        if len({c.case_id for c in self.cases}) != len(self.cases):
            raise ValueError("duplicate atomic case_id")
        return self


class ReferenceLabel(Contract):
    case_id: str = Field(min_length=1)
    label: Outcome
    rationale: str = Field(min_length=1)


class AuthorLabels(Contract):
    schema_version: Literal[1]
    suite_id: str
    annotation_source: str = Field(min_length=1)
    labels: list[ReferenceLabel] = Field(min_length=1)


def _resolve(repo_root: Path, relative_path: str) -> Path:
    path = (repo_root / relative_path).resolve()
    if not path.is_relative_to(repo_root):
        raise ValueError("evaluation reference path escapes repository")
    return path


def load_frozen_claims(
    cases_path: Path = CASES, labels_path: Path = LABELS,
    *, repo_root: Path = ROOT,
) -> tuple[AtomicClaimSuite, dict[str, ReferenceLabel], TypedSourceGuard, dict[str, str]]:
    repo = repo_root.resolve()
    cases_bytes = cases_path.read_bytes()
    label_bytes = labels_path.read_bytes()
    suite = AtomicClaimSuite.model_validate_json(cases_bytes)
    gold = AuthorLabels.model_validate_json(label_bytes)
    if gold.suite_id != suite.suite_id:
        raise ValueError("author-label suite ID mismatch")
    if len(gold.labels) != len(suite.cases):
        raise ValueError("author-label count differs from frozen cases")
    label_map = {row.case_id: row for row in gold.labels}
    if len(label_map) != len(gold.labels) or set(label_map) != {case.case_id for case in suite.cases}:
        raise ValueError("author labels must cover each case once, no duplicates")
    catalog_path = _resolve(repo, suite.catalog_path)
    source_path = _resolve(repo, suite.sources_path)
    guard = TypedSourceGuard(catalog_path=catalog_path, source_path=source_path)
    catalog, catalog_hash = load_catalog(catalog_path)
    if guard.catalog_sha256 != catalog_hash:
        raise ValueError("source guard and catalog provenance differ")
    by_id = {product.product_id: product for product in catalog.products}
    for case in suite.cases:
        product = by_id.get(case.product_id)
        if product is None:
            raise ValueError(f"{case.case_id}: unknown product")
        if case.kind == "typed":
            if case.field == "price_usd":
                if type(case.asserted_value) is not str:
                    raise ValueError("price claims need a decimal string")
                try:
                    value = Decimal(case.asserted_value)
                except InvalidOperation as exc:
                    raise ValueError("invalid price assertion") from exc
                if not value.is_finite() or value < 0:
                    raise ValueError("invalid price assertion")
            elif not case.field.startswith("attributes.") or (
                case.field.removeprefix("attributes.") not in product.attributes
                or type(case.asserted_value) is not bool
            ):
                raise ValueError(f"{case.case_id}: unknown or untyped attribute")
    return suite, label_map, guard, {
        "cases_sha256": hashlib.sha256(cases_bytes).hexdigest(),
        "author_labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
        "catalog_sha256": catalog_hash,
        "source_documents_sha256": guard.source_docs_sha256,
    }


def classify(case: AtomicClaimCase, guard: TypedSourceGuard) -> dict:
    """Return no semantic prediction for unstructured free-text reasons."""
    if case.kind == "open_text":
        return {
            "prediction": "needs_human_review",
            "decision_reason": "free_text_claim_not_automatically_extracted_or_entailment_checked",
            "method": "abstain_unstructured_claim",
        }
    prediction, decision_reason = guard.classify_annotated_claim(
        product_id=case.product_id,
        field=case.field,
        asserted_value=case.asserted_value,
        evidence_ids=set(case.evidence_ids),
    )
    if prediction not in FOUR_LABELS:
        raise ValueError("typed guard returned unrecognized label")
    return {
        "prediction": prediction,
        "decision_reason": decision_reason,
        "method": "typed_active_source_adjudication",
    }


def naive_citation_baseline(case: AtomicClaimCase, guard: TypedSourceGuard) -> str | None:
    """Purposely weak structural citation control, on typed cases only."""
    if case.kind != "typed":
        return None
    if case.evidence_ids and all(
        (e := guard._evidence.get(eid)) is not None
        and e.product_id == case.product_id
        for eid in case.evidence_ids
    ):
        return "supported"
    return "unsupported"


def _metrics(rows: list[dict], *, prediction_field: str) -> dict:
    evaluated = [r for r in rows if r["kind"] == "typed"]
    confusion = {
        truth: {pred: 0 for pred in FOUR_LABELS}
        for truth in FOUR_LABELS
    }
    for row in evaluated:
        pred = row[prediction_field]
        if pred not in FOUR_LABELS:
            raise ValueError("typed cases cannot be silently excluded from agreement")
        confusion[row["author_label"]][pred] += 1
    per_label = {}
    for label in FOUR_LABELS:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[gold][label] for gold in FOUR_LABELS)
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else 0.0
        )
        per_label[label] = {
            "support": support,
            "precision": precision, "recall": recall, "f1": f1,
        }
    correct = sum(confusion[label][label] for label in FOUR_LABELS)
    return {
        "typed_evaluated": len(evaluated),
        "author_label_agreement_count": correct,
        "author_label_agreement_rate": correct / len(evaluated),
        "macro_f1_against_author_labels": sum(p["f1"] for p in per_label.values()) / len(per_label),
        "confusion_by_author_label": confusion,
        "per_label": per_label,
    }


def evaluate_claims(suite: AtomicClaimSuite, labels: dict[str, ReferenceLabel],
                    guard: TypedSourceGuard, provenance: dict[str, str]) -> dict:
    rows = []
    for case in suite.cases:
        # No gold labels enter the classifier: score only AFTER predictions.
        decision = classify(case, guard)
        predicted = decision["prediction"]
        author = labels[case.case_id]
        rows.append({
            "case_id": case.case_id, "kind": case.kind,
            "product_id": case.product_id,
            "reason": case.reason, "claim_span": case.claim_span,
            "field": case.field, "asserted_value": case.asserted_value,
            "evidence_ids": list(case.evidence_ids),
            **decision,
            "naive_citation_baseline_prediction": naive_citation_baseline(case, guard),
            "author_label": author.label, "author_rationale": author.rationale,
            "matches_author_label": predicted == author.label if case.kind == "typed" else None,
        })
    typed = [r for r in rows if r["kind"] == "typed"]
    open_text = [r for r in rows if r["kind"] == "open_text"]
    if any(r["prediction"] != "needs_human_review" for r in open_text):
        raise ValueError("free-text cases must abstain until independent adjudication")
    return {
        "suite_id": suite.suite_id,
        "case_count": len(rows),
        "typed_case_count": len(typed),
        "free_text_review_required_count": len(open_text),
        "automated_claim_coverage_rate": len(typed) / len(rows),
        "typed_judge": _metrics(rows, prediction_field="prediction"),
        "naive_citation_only_control": _metrics(
            rows, prediction_field="naive_citation_baseline_prediction"
        ),
        "provenance": provenance,
        "prediction_method": "deterministic_typed_source_claim_classification_only",
        "author_labels_independently_adjudicated": False,
        "free_text_semantic_entailment_measured": False,
        "real_world_claim_groundedness": None,
        "unsupported_claim_rate_real_model": None,
        "limitations": (
            "All 26 claims and gold labels are authored for a fictional fixture. "
            "22 are already explicitly structured; the classifier does not "
            "extract facts from model prose. Four open-text claims abstain for "
            "independent blind review. Agreement with author labels does not "
            "establish real-world semantic entailment or generalization."
        ),
        "per_case": rows,
    }


def export_blind_review_csv(report: dict, guard: TypedSourceGuard, path: Path) -> int:
    """Export all examples without exposing author gold labels to reviewers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id", "kind", "product_id", "reason", "claim_span",
        "claim_field", "asserted_value", "source_evidence_json",
        "reviewer_label", "reviewer_id", "review_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in report["per_case"]:
            evidence = [
                {
                    "evidence_id": eid,
                    "source_id": guard._evidence[eid].source_id,
                    "supported_field": guard._evidence[eid].supported_field,
                    "excerpt": guard._evidence[eid].excerpt,
                    "superseded": guard._evidence[eid].source_id in guard._inactive_source_ids,
                } if eid in guard._evidence else {
                    "evidence_id": eid, "missing_reference": True,
                }
                for eid in row["evidence_ids"]
            ]
            writer.writerow({
                "case_id": row["case_id"], "kind": row["kind"],
                "product_id": row["product_id"], "reason": row["reason"],
                "claim_span": row["claim_span"], "claim_field": row["field"],
                "asserted_value": row["asserted_value"],
                "source_evidence_json": json.dumps(evidence, ensure_ascii=False),
                "reviewer_label": "", "reviewer_id": "", "review_notes": "",
            })
    return len(report["per_case"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/evaluation/stage34-atomic-claims.json"))
    parser.add_argument("--blind-review-csv", type=Path,
                        default=Path("artifacts/evaluation/stage34-blind-review.csv"))
    args = parser.parse_args()
    suite, labels, guard, hashes = load_frozen_claims(args.cases, args.labels)
    result = evaluate_claims(suite, labels, guard, hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                           encoding="utf-8")
    exported = export_blind_review_csv(result, guard, args.blind_review_csv)
    print(json.dumps({
        "suite_id": result["suite_id"], "case_count": result["case_count"],
        "typed_case_count": result["typed_case_count"],
        "free_text_review_required_count": result["free_text_review_required_count"],
        "typed_author_label_agreement": result["typed_judge"]["author_label_agreement_count"],
        "naive_author_label_agreement": result["naive_citation_only_control"]["author_label_agreement_count"],
        "blind_review_rows": exported,
        "real_world_claim_groundedness": result["real_world_claim_groundedness"],
    }, indent=2))


if __name__ == "__main__":
    main()
