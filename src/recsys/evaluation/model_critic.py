"""Real model-based *evidence support classifier* with an explicit review contract.

This module judges the FIXED 26 FICTIONAL authored claim cases, using raw
primary-source text. It never trains on, prompts with, or changes author labels.
It does NOT extract all atomic facts from real recommendation paragraphs.
"""

from collections import Counter
from hashlib import sha256
import json
import re
from time import perf_counter
from typing import Protocol, Literal

from pydantic import Field, ValidationError

from recsys.evaluation.atomic_claims import (
    AtomicClaimCase, AtomicClaimSuite, AuthorLabels, ReferenceLabel,
    FOUR_LABELS, _metrics,
)
from recsys.evaluation.source_audit import SourceFixture
from recsys.schemas.contracts import Contract


class CriticGenerator(Protocol):
    call_count: int
    last_prompt_tokens: int
    last_generated_tokens: int
    last_generation_seconds: float

    def generate(self, instructions: str, context: str) -> str: ...


class ModelClaimVerdict(Contract):
    label: Literal["supported", "contradicted", "unsupported", "ambiguous"]
    rationale: str = Field(min_length=1, max_length=1000)


SYSTEM = """You are a source-grounded evidence auditor on FICTIONAL products.
Assess ONE atomic claim using the original source passages and cited IDs.
The documents, not general knowledge or potentially outdated catalog labels,
are the only source of truth. Source revisions supersede older documents.
Four mutually exclusive outcomes:
- supported: active unambiguous source asserts the claim AND every cited
  evidence ID matches this product, field/claim, and active source.
- contradicted: active unambiguous source asserts an opposite claim
  (including when an old citation was superseded).
- ambiguous: active source passages disagree, so both values are plausible.
- unsupported: no active source backs it, no citation, an invented/off-field/
  other-product citation, or an unsubstantiated efficacy/medical claim.
Choose ambiguity first if active documents disagree. Choose contradiction
before unsupported if clear active documents assert the opposite.
For paraphrases, interpret what the words mean, not just matching tokens.
Do not assume a cited document supports claims that it does not actually make.
Return exactly one JSON object and nothing else:
{"label":"supported|contradicted|unsupported|ambiguous","rationale":"brief evidence-focused explanation"}
Do not add fields. Do not use markdown fences."""


FENCE = re.compile(r"\A\s*```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```\s*\Z", re.I)


def _documents_by_id(fixture: SourceFixture):
    docs = {d.source_id: d for d in fixture.documents}
    if len(docs) != len(fixture.documents):
        raise ValueError("duplicate source document IDs")
    inactive = set()
    for doc in fixture.documents:
        previous = doc.supersedes
        seen = set()
        while previous is not None:
            if previous in seen or previous not in docs:
                raise ValueError("cyclic or missing superseded document")
            seen.add(previous)
            inactive.add(previous)
            previous = docs[previous].supersedes
    return docs, inactive


def build_evidence_context(
    case: AtomicClaimCase, fixture: SourceFixture, evidence_index: dict,
) -> str:
    """Use only source passages and citation metadata; never author gold labels.

    Document status is computed solely from supersession links, not from a
    label oracle. Other-product citations are visible as other-product IDs.
    """
    documents, inactive = _documents_by_id(fixture)
    if not any(doc.product_id == case.product_id for doc in fixture.documents):
        raise ValueError("unknown product in source fixture")
    docs = [
        {
            "source_id": doc.source_id,
            "date": doc.effective_date,
            "supersedes": doc.supersedes,
            "is_superseded": doc.source_id in inactive,
            "verbatim_text": doc.text,
        }
        for doc in fixture.documents if doc.product_id == case.product_id
    ]
    citations = []
    for eid in case.evidence_ids:
        record = evidence_index.get(eid)
        if record is None:
            citations.append({"evidence_id": eid, "lookup_status": "not_found"})
        else:
            citations.append({
                "evidence_id": eid, "product_id": record.product_id,
                "source_id": record.source_id, "field": record.supported_field,
                "verbatim_excerpt": record.excerpt,
                "is_superseded": record.source_id in inactive,
            })
    payload = {
        "product_id": case.product_id,
        "atomic_claim_text": case.claim_span,
        "full_reason_text": case.reason,
        "explicit_typed_claim_field": case.field,
        "explicit_typed_claim_value": case.asserted_value,
        "provided_citation_ids": case.evidence_ids,
        "citation_lookups": citations,
        "product_primary_source_documents": docs,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_verdict(raw: str) -> tuple[ModelClaimVerdict | None, str | None]:
    """Only remove a COMPLETE Markdown fence; never infer a missing label."""
    if not isinstance(raw, str):
        return None, "non_text_model_response"
    match = FENCE.fullmatch(raw)
    if match:
        raw = match.group(1)
    try:
        return ModelClaimVerdict.model_validate_json(raw), None
    except (ValidationError, ValueError):
        return None, "invalid_json_or_schema"


def run_model_critic(
    suite: AtomicClaimSuite, fixture: SourceFixture, evidence_index: dict,
    generator: CriticGenerator,
) -> dict:
    """Run one real generation per case; model cannot access reference labels."""
    by_case = {}
    start_calls = generator.call_count
    for case in suite.cases:
        if case.case_id in by_case:
            raise ValueError("duplicate case ID")
        context = build_evidence_context(case, fixture, evidence_index)
        start = perf_counter()
        before = generator.call_count
        raw = None
        error = None
        try:
            raw = generator.generate(SYSTEM, context)
        except Exception as exc:
            error = type(exc).__name__
        counted_calls = generator.call_count - before
        if counted_calls not in (0, 1):
            raise ValueError("model invoked more than once per claim")
        verdict, parse_error = parse_verdict(raw) if error is None else (None, error)
        by_case[case.case_id] = {
            "case_id": case.case_id, "kind": case.kind,
            "prediction": verdict.label if verdict else "needs_human_review",
            "model_rationale": verdict.rationale if verdict else None,
            "error": parse_error,
            "prompt_sha256": sha256(SYSTEM.encode()).hexdigest(),
            "context_sha256": sha256(context.encode()).hexdigest(),
            "raw_model_response": raw,
            "model_calls": counted_calls,
            "prompt_tokens": int(generator.last_prompt_tokens) if counted_calls else 0,
            "generated_tokens": int(generator.last_generated_tokens) if counted_calls else 0,
            "generation_seconds": float(generator.last_generation_seconds) if counted_calls else 0.0,
            "elapsed_seconds": round(perf_counter() - start, 4),
        }
    actual = generator.call_count - start_calls
    if actual != sum(row["model_calls"] for row in by_case.values()):
        raise ValueError("physical model calls disagree with trace")
    return {
        "experiment_type": "real_model_based_atomic_evidence_classifier",
        "source_scope": "fictional_primary_sources_no_independent_labels",
        "predictions": list(by_case.values()),
        "physical_model_invocations": actual,
        "total_prompt_tokens": sum(row["prompt_tokens"] for row in by_case.values()),
        "total_generated_tokens": sum(row["generated_tokens"] for row in by_case.values()),
        "total_generation_seconds": round(sum(row["generation_seconds"] for row in by_case.values()), 4),
        "total_wall_seconds": round(sum(row["elapsed_seconds"] for row in by_case.values()), 4),
        "failed_or_invalid_count": sum(row["prediction"] == "needs_human_review" for row in by_case.values()),
    }


def evaluate_predictions(
    suite: AtomicClaimSuite,
    author_labels: dict[str, ReferenceLabel],
    predictions: dict,
) -> dict:
    """Join gold labels strictly *after* model inference; score abstentions.

    The 22 typed and 4 open-text subsets MUST remain separate. This is not
    human-calibrated performance, since all labels were written by authors.
    """
    model_rows = predictions["predictions"]
    by_id = {row["case_id"]: row for row in model_rows}
    if len(model_rows) != len(suite.cases) or len(by_id) != len(model_rows):
        raise ValueError("incomplete or duplicate model predictions")
    if set(by_id) != {case.case_id for case in suite.cases} or set(author_labels) != set(by_id):
        raise ValueError("model predictions, cases and author labels are not aligned")
    rows = []
    for case in suite.cases:
        model_row = by_id[case.case_id]
        predicted = model_row["prediction"]
        if predicted not in (*FOUR_LABELS, "needs_human_review"):
            raise ValueError("unrecognized model verdict")
        reference = author_labels[case.case_id]
        rows.append({
            "case_id": case.case_id, "kind": case.kind,
            "author_label": reference.label,
            "model_prediction": predicted,
            "model_error": model_row["error"],
            "match": predicted == reference.label,
        })
    def stats(subset):
        valid = [r for r in subset if r["model_prediction"] in FOUR_LABELS]
        matched = sum(r["match"] for r in valid)
        confusion = {
            label: {pred: 0 for pred in FOUR_LABELS}
            for label in FOUR_LABELS
        }
        for row in valid:
            confusion[row["author_label"]][row["model_prediction"]] += 1
        per_label = {}
        for label in FOUR_LABELS:
            tp = confusion[label][label]
            actual = sum(r["author_label"] == label for r in subset)
            predicted_count = sum(r["model_prediction"] == label for r in valid)
            prec = tp / predicted_count if predicted_count else None
            recall = tp / actual if actual else None
            f1 = (2*prec*recall/(prec+recall)) if prec is not None and recall is not None and prec+recall else 0.0
            per_label[label] = {
                "author_support": actual, "precision": prec,
                "recall_including_abstentions": recall, "f1_including_abstentions": f1,
            }
        return {
            "cases": len(subset),
            "valid_predictions": len(valid),
            "abstentions_or_invalid": len(subset) - len(valid),
            "coverage": len(valid)/len(subset) if subset else None,
            "correct_overall": matched,
            "accuracy_including_abstentions": matched/len(subset) if subset else None,
            "selective_accuracy_valid_only": matched/len(valid) if valid else None,
            "macro_f1_including_abstentions": sum(row["f1_including_abstentions"] for row in per_label.values())/len(FOUR_LABELS),
            "confusion_by_author_label": confusion,
            "per_label": per_label,
        }
    return {
        "all": stats(rows),
        "typed": stats([r for r in rows if r["kind"] == "typed"]),
        "open_text": stats([r for r in rows if r["kind"] == "open_text"]),
        "per_case": rows,
        "reference_type": "same_project_author_written_synthetic_labels",
        "independent_human_review_complete": False,
        "human_calibrated_accuracy": None,
        "real_world_semantic_groundedness": None,
    }
