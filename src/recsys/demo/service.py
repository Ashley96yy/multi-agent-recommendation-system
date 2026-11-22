"""Auditable end-to-end demo service, independent of the Gradio front end.

Walkthrough mode replays PREWRITTEN FICTIONAL output, never a real LLM.
Live mode lazily loads a pinned Qwen only if the evidence preflight allows a
generation. No rejected proposal can appear as an approved recommendation.
"""

from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import Field, model_validator

from recsys.baseline.local_qwen import MODEL_ID, MODEL_REVISION
from recsys.multiagent import (
    CriticVerificationAgent, MultiAgentOrchestrator,
    PreferenceRetrievalAgent, RecommendationAgent, TypedSourceGuard,
)
from recsys.retrieval import CatalogRetriever
from recsys.schemas import UserPreferences
from recsys.schemas.contracts import Contract


ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = ROOT / "data/source_evidence_demo/catalog.json"
SOURCES_PATH = ROOT / "data/source_evidence_demo/source_documents.json"
WALKTHROUGHS_PATH = ROOT / "data/demo/walkthroughs.json"


class Walkthrough(Contract):
    scenario_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    preferences: UserPreferences
    max_revisions: Literal[0, 1]
    generator_outputs: list[str]
    expected_status: Literal[
        "approved", "rejected", "no_candidates", "insufficient_evidence", "abstained"
    ]
    expected_generator_calls: int = Field(ge=0, le=2)

    @model_validator(mode="after")
    def check_fixture(self):
        if self.expected_generator_calls != len(self.generator_outputs):
            raise ValueError("expected model attempts must equal scripted responses")
        for raw in self.generator_outputs:
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("scripted responses must be nonempty text")
        return self


class WalkthroughSuite(Contract):
    schema_version: Literal[1]
    fixture_id: str
    provenance: str = Field(min_length=1)
    cases: list[Walkthrough] = Field(min_length=1)

    @model_validator(mode="after")
    def check_cases(self):
        names = [x.scenario_id for x in self.cases]
        if len(set(names)) != len(names):
            raise ValueError("duplicate scripted demo scenario ID")
        return self


class ScriptedGenerator:
    """A transparent demonstration fixture, NOT an LLM or a generative agent."""

    name = "PREWRITTEN fixture replay (not Qwen)"

    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.call_count = 0

    def generate(self, instructions: str, context: str) -> str:
        index = self.call_count
        self.call_count += 1
        if index >= len(self.outputs):
            raise RuntimeError("script exhausted: unexpected unbudgeted proposal")
        return self.outputs[index]


@lru_cache(maxsize=1)
def _pinned_qwen_cpu():
    """Only executed after an explicit real-model request passes the preflight."""
    from recsys.baseline.local_qwen import LocalQwenGenerator
    return LocalQwenGenerator(
        model_id=MODEL_ID, revision=MODEL_REVISION,
        device="cpu", max_new_tokens=240,
    )


class LazyPinnedQwen:
    """Per-request accounting across a process-cached, optional real model."""

    name = f"REAL local Qwen {MODEL_ID}@{MODEL_REVISION}"

    def __init__(self):
        self.call_count = 0
        self.last_prompt_tokens = None
        self.last_generated_tokens = None
        self.last_generation_seconds = None

    def generate(self, instructions: str, context: str) -> str:
        model = _pinned_qwen_cpu()
        before = model.call_count
        try:
            return model.generate(instructions, context)
        finally:
            used = model.call_count - before
            self.call_count += used
            if used:
                self.last_prompt_tokens = model.last_prompt_tokens
                self.last_generated_tokens = model.last_generated_tokens
                self.last_generation_seconds = model.last_generation_seconds


class TracingGenerator:
    """Instrument one generator request without changing its text/decisions."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.calls: list[dict] = []

    def generate(self, instructions: str, context: str) -> str:
        before = getattr(self.delegate, "call_count", None)
        start = perf_counter()
        record = {
            "proposal_index": len(self.calls) + 1,
            "system_prompt_sha256": hashlib.sha256(instructions.encode()).hexdigest(),
            "retrieved_context_sha256": hashlib.sha256(context.encode()).hexdigest(),
            "backend_name": getattr(self.delegate, "name", type(self.delegate).__name__),
            "raw_unverified_output": None,
            "transport_error": None,
        }
        try:
            raw = self.delegate.generate(instructions, context)
            record["raw_unverified_output"] = raw
            return raw
        except Exception as exc:
            record["transport_error"] = type(exc).__name__
            raise
        finally:
            after = getattr(self.delegate, "call_count", None)
            record["actual_backend_calls"] = (
                after - before if before is not None and after is not None else None
            )
            record["total_proposal_elapsed_seconds"] = round(perf_counter() - start, 4)
            for name in ("last_prompt_tokens", "last_generated_tokens", "last_generation_seconds"):
                record[name] = getattr(self.delegate, name, None) if record["actual_backend_calls"] else None
            self.calls.append(record)


def preferences_from_input(
    *, request: str, required_category: str | None = None,
    required_attributes: list[str] | None = None,
    max_budget_usd: str | float | None = None,
    excluded_product_ids: str = "", excluded_brands: str = "",
) -> UserPreferences:
    """Pydantic JSON validation enforces strict Decimal and set contracts."""
    if not isinstance(request, str) or not request.strip() or len(request) > 1000:
        raise ValueError("Enter a request of 1–1000 non-whitespace characters.")
    if max_budget_usd in ("", None):
        budget = None
    else:
        try:
            decimal = Decimal(str(max_budget_usd))
        except (ValueError, ArithmeticError) as exc:
            raise ValueError("Budget must be a finite nonnegative US-dollar amount.") from exc
        if not decimal.is_finite() or decimal < 0:
            raise ValueError("Budget must be a finite nonnegative US-dollar amount.")
        budget = str(decimal)
    def split_text(value: str) -> list[str]:
        if not isinstance(value, str):
            raise ValueError("Exclusions must be comma-separated text.")
        return [item.strip() for item in value.split(",") if item.strip()]

    data = {
        "request": request.strip(),
        "required_category": required_category or None,
        "required_attributes": sorted(set(required_attributes or [])),
        "max_budget_usd": budget,
        "excluded_product_ids": split_text(excluded_product_ids),
        "excluded_brands": split_text(excluded_brands),
    }
    return UserPreferences.model_validate_json(json.dumps(data))


class DemoService:
    """Uses the original Stage 3.x orchestrator and source-document guard."""

    def __init__(
        self, *, catalog_path: Path = CATALOG_PATH,
        source_path: Path = SOURCES_PATH,
        walkthroughs_path: Path = WALKTHROUGHS_PATH,
    ):
        self.catalog_path = catalog_path
        self.source_path = source_path
        self.guard = TypedSourceGuard(catalog_path=catalog_path, source_path=source_path)
        self.retriever = CatalogRetriever.from_json(catalog_path)
        self.guard.bind(self.retriever.catalog_sha256)
        self.suite = WalkthroughSuite.model_validate_json(walkthroughs_path.read_bytes())
        self.scenarios = {c.scenario_id: c for c in self.suite.cases}

    def run_walkthrough(self, scenario_id: str) -> dict:
        if scenario_id not in self.scenarios:
            raise ValueError("Unknown fixed walkthrough ID; only labeled fixtures can be replayed.")
        case = self.scenarios[scenario_id]
        report = self._run(
            case.preferences, ScriptedGenerator(case.generator_outputs),
            max_revisions=case.max_revisions, mode="scripted_fixture",
        )
        if (report["status"], report["logical_generator_attempts"]) != (
            case.expected_status, case.expected_generator_calls
        ):
            raise RuntimeError("Fixed walkthrough outcome drifted; do not present it as valid.")
        report["walkthrough"] = {
            "scenario_id": case.scenario_id,
            "title": case.title,
            "description": case.description,
            "fixture_id": self.suite.fixture_id,
            "provenance": self.suite.provenance,
        }
        return report

    def run_live(
        self, preferences: UserPreferences, *, max_revisions: int = 1,
        generator=None, top_k: int = 4,
    ) -> dict:
        if not isinstance(preferences, UserPreferences):
            raise TypeError("run_live needs validated preferences")
        if generator is None:
            generator = LazyPinnedQwen()
        return self._run(
            preferences, generator,
            max_revisions=max_revisions, top_k=top_k,
            mode="live_qwen" if isinstance(generator, LazyPinnedQwen) else "injected_generator_test",
        )

    def _run(self, preferences, generator, *, max_revisions, top_k=4, mode):
        if type(max_revisions) is not int or max_revisions not in (0, 1):
            raise ValueError("Only zero or one revision is supported")
        if type(top_k) is not int or not 1 <= top_k <= 4:
            raise ValueError("top_k must be 1–4")
        tracer = TracingGenerator(generator)
        controller = MultiAgentOrchestrator(
            PreferenceRetrievalAgent(self.retriever),
            RecommendationAgent(tracer),
            CriticVerificationAgent(source_guard=self.guard),
            max_revisions=max_revisions,
            format_policy="safe_json_normalization",
            evidence_preflight=True,
        )
        began = perf_counter()
        result = controller.run(preferences, top_k=top_k)
        by_id = {c.product.product_id: c for c in result.retrieval.candidates}
        recommended = []
        if result.status == "approved" and result.answer is not None:
            for rec in result.answer.recommendations:
                candidate = by_id[rec.product_id]
                referenced = {e.evidence_id: e for e in candidate.evidence}
                recommended.append({
                    "product_id": rec.product_id,
                    "title": candidate.product.title,
                    "price_usd": str(candidate.product.price_usd) if candidate.product.price_usd is not None else None,
                    "reason_unverified_natural_language": rec.reason,
                    "cited_evidence": [self._evidence(e) for ref in rec.evidence_ids
                                        if (e := referenced.get(ref)) is not None],
                })
        attempts = []
        for index, attempt in enumerate(result.attempts):
            trace = tracer.calls[index] if index < len(tracer.calls) else {}
            attempts.append({
                **trace,
                "attempt_number": attempt.attempt_number,
                "is_revision": attempt.is_revision,
                "json_schema_valid": attempt.parse_valid,
                "critic_approved": attempt.verdict.approved,
                "critic_issues": [issue.model_dump(mode="json") for issue in attempt.verdict.issues],
                "format_actions": attempt.format_actions,
                "proposed_product_ids_UNVERIFIED": attempt.proposed_product_ids,
                "semantic_entailment_verified": False,
            })
        diagnosis = []
        evidence = []
        for candidate in result.retrieval.candidates:
            p = candidate.product
            diagnosis.extend([
                self.guard.field_status[(p.product_id, f"attributes.{key}")]
                for key in sorted(p.attributes)
                if (p.product_id, f"attributes.{key}") in self.guard.field_status
            ])
            evidence.extend(self._evidence(e) for e in candidate.evidence)
        return {
            "mode": mode,
            "status": result.status,
            "status_note": result.note,
            "preferences": preferences.model_dump(mode="json"),
            "final_approved_recommendations_only": recommended,
            "retrieval": {
                "catalog_sha256": result.retrieval.catalog_sha256,
                "source_docs_sha256": self.guard.source_docs_sha256,
                "embedding": result.retrieval.embedder_name,
                "indexed_products": result.retrieval.indexed_products,
                "eligible_products": result.retrieval.eligible_products,
                "filtered_by_reason": result.retrieval.filtered_by_reason,
                "no_match_reason": result.retrieval.no_match_reason,
                "candidates": [{
                    "product_id": c.product.product_id,
                    "title": c.product.title,
                    "similarity": c.similarity,
                    "metadata": c.product.model_dump(mode="json"),
                } for c in result.retrieval.candidates],
            },
            "source_evidence": evidence,
            "source_field_diagnostics": diagnosis,
            "attempts": attempts,
            "logical_generator_attempts": result.generator_calls,
            "actual_backend_calls": sum(
                r["actual_backend_calls"] or 0 for r in tracer.calls
            ),
            "revision_used": result.revision_used,
            "generated_prompt_tokens": (
                sum(r["last_prompt_tokens"] or 0 for r in tracer.calls)
                if mode == "live_qwen" else None
            ),
            "generated_completion_tokens": (
                sum(r["last_generated_tokens"] or 0 for r in tracer.calls)
                if mode == "live_qwen" else None
            ),
            "request_elapsed_seconds": round(perf_counter() - began, 4),
            "independent_semantic_groundedness": None,
            "limitations": (
                "All products and source passages are fictional. Final recommendations "
                "passed hard constraints, citation linkage and typed source checks, "
                "NOT independent natural-language entailment. Rejected drafts are "
                "visible only in the explicitly unverified attempt trace."
            ),
        }

    def _evidence(self, reference) -> dict:
        source = self.guard._documents.get(reference.source_id)
        return {
            "evidence_id": reference.evidence_id,
            "product_id": reference.product_id,
            "supported_field": reference.supported_field,
            "source_id": reference.source_id,
            "source_type": reference.source_type,
            "verbatim_excerpt": reference.excerpt,
            "effective_date": source.effective_date if source else None,
            "source_superseded": reference.source_id in self.guard._inactive_source_ids,
        }


def render_markdown(report: dict) -> str:
    """Summary contains ONLY approved results; rejected drafts never leak here."""
    from html import escape
    status = report["status"]
    heading = {
        "approved": "Approved by deterministic checks",
        "abstained": "Model abstained",
        "rejected": "Rejected: no approved final draft",
        "insufficient_evidence": "Insufficient active evidence — no generation",
        "no_candidates": "No eligible catalog candidates — no generation",
    }[status]
    lines = [
        f"### {heading}",
        f"**Mode:** {escape(report['mode'])} · "
        f"**Proposal attempts:** {report['logical_generator_attempts']} · "
        f"**Revision used:** {'yes' if report['revision_used'] else 'no'}",
    ]
    for rec in report["final_approved_recommendations_only"]:
        lines += [
            f"**{escape(rec['title'])}** ({escape(rec['product_id'])})",
            f"Price (fictional): ${escape(rec['price_usd'] or 'unknown')}",
            f"Reason (model/script text, *not semantically verified*): "
            f"{escape(rec['reason_unverified_natural_language'])}",
            f"Linked evidence IDs: {', '.join(escape(e['evidence_id']) for e in rec['cited_evidence'])}",
        ]
    if not report["final_approved_recommendations_only"]:
        lines.append("**No unapproved recommendation is displayed.**")
    if report["status_note"]:
        lines.append("Detail: " + escape(report["status_note"]))
    lines.append(
        "\n> Fictional products and sources only. Verified structural/typed "
        "citations do not prove that generated natural-language claims are true."
    )
    return "\n\n".join(lines)
