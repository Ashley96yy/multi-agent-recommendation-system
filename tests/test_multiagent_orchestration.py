"""Deterministic Stage 3.0 tests; model stubs are NOT actual LLM experiments."""

import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from recsys.baseline.contracts import SingleAgentAnswer
from recsys.multiagent import (
    CriticVerificationAgent,
    MultiAgentOrchestrator,
    PreferenceRetrievalAgent,
    RecommendationAgent,
)
from recsys.retrieval import CatalogRetriever
from recsys.schemas import UserPreferences


CATALOG = Path(__file__).resolve().parents[1] / "data/sample_catalog/catalog.json"


class SequenceGenerator:
    """Records calls, returning precisely one prewritten response per call."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.call_count = 0
        self.instructions = []
        self.contexts = []

    def generate(self, instructions, context):
        self.instructions.append(instructions)
        self.contexts.append(json.loads(context))
        pos = self.call_count
        self.call_count += 1
        if pos >= len(self.outputs):
            raise AssertionError("unbudgeted model call")
        return self.outputs[pos]


def payload(*, pid="SYN-001", citations=None, reason="Fictional catalog says fragrance-free."):
    return json.dumps({
        "recommendations": [{
            "product_id": pid,
            "reason": reason,
            "evidence_ids": citations if citations is not None else ["E-SYN-001"],
        }],
        "no_suitable_match": False,
    })


def prefs(**kwargs):
    return UserPreferences(request="fragrance free moisturizer", **kwargs)


def workflow(generator, max_revisions=1, **critic_options):
    retriever = CatalogRetriever.from_json(CATALOG)
    return MultiAgentOrchestrator(
        PreferenceRetrievalAgent(retriever),
        RecommendationAgent(generator),
        CriticVerificationAgent(**critic_options),
        max_revisions=max_revisions,
    )


def test_hard_no_candidate_bypasses_all_generation_and_critique():
    generator = SequenceGenerator()
    result = workflow(generator).run(prefs(max_budget_usd=Decimal("1")))
    assert result.status == "no_candidates"
    assert result.answer is None
    assert result.generator_calls == 0
    assert result.attempts == []
    assert result.note == "no_eligible_products"
    assert generator.call_count == 0


def test_valid_first_pass_approved_and_only_one_generation():
    generator = SequenceGenerator(payload())
    result = workflow(generator).run(
        prefs(required_attributes={"fragrance_free"}, max_budget_usd=Decimal("30"))
    )
    assert result.status == "approved"
    assert result.answer.recommendations[0].product_id == "SYN-001"
    assert result.revision_used is False
    assert result.generator_calls == 1
    assert result.attempts[0].verdict.approved is True
    assert result.attempts[0].verdict.semantic_entailment_verified is False
    assert len(generator.contexts[0]["products"]) == 1


def test_invalid_json_gets_one_revision_and_never_reuses_prior_raw_text():
    generator = SequenceGenerator("not-json", payload())
    result = workflow(generator).run(prefs(required_attributes={"fragrance_free"}))
    assert result.status == "approved"
    assert result.generator_calls == 2
    assert result.revision_used
    assert [a.is_revision for a in result.attempts] == [False, True]
    assert not result.attempts[0].parse_valid
    assert result.attempts[0].verdict.issues[0].code == "invalid_generation"
    assert result.attempts[1].verdict.approved
    assert "invalid_generation" in generator.instructions[1]
    assert "not-json" not in generator.instructions[1]
    assert generator.contexts[0] == generator.contexts[1]


def test_cross_product_citation_is_rejected_then_revised():
    generator = SequenceGenerator(payload(citations=["E-SYN-003"]), payload())
    result = workflow(generator).run(prefs(required_attributes={"fragrance_free"}))
    assert result.status == "approved"
    assert {x.code for x in result.attempts[0].verdict.issues} >= {
        "invalid_citation", "required_attribute_uncited",
    }
    assert result.attempts[-1].verdict.approved


def test_unproven_medical_claim_heuristic_rejected_then_revised():
    generator = SequenceGenerator(
        payload(reason="This item is clinically proven to cure eczema."),
        payload(reason="Its fictional catalog attribute is fragrance-free."),
    )
    result = workflow(generator).run(prefs(required_attributes={"fragrance_free"}))
    assert result.status == "approved"
    assert "prohibited_claim_phrase" in {e.code for e in result.attempts[0].verdict.issues}
    assert result.answer.recommendations[0].reason == "Its fictional catalog attribute is fragrance-free."
    assert result.attempts[-1].verdict.semantic_entailment_verified is False


def test_missing_attribute_field_citation_triggers_revision():
    generator = SequenceGenerator(
        payload(citations=["E-SYN-002"]), payload(citations=["E-SYN-001"]),
    )
    result = workflow(generator).run(prefs(required_attributes={"fragrance_free"}))
    assert result.status == "approved"
    assert "required_attribute_uncited" in {e.code for e in result.attempts[0].verdict.issues}


def test_two_invalid_proposals_fail_closed_and_do_not_expose_unapproved_answer():
    generator = SequenceGenerator(payload(pid="SYN-HALLUCINATED"), payload(pid="SYN-999"))
    result = workflow(generator).run(prefs())
    assert result.status == "rejected"
    assert result.answer is None
    assert result.revision_used is True
    assert result.generator_calls == 2
    assert all(not attempt.verdict.approved for attempt in result.attempts)
    assert generator.call_count == 2


def test_zero_revision_budget_never_calls_generator_twice():
    generator = SequenceGenerator("invalid", payload())
    result = workflow(generator, max_revisions=0).run(prefs())
    assert result.status == "rejected"
    assert result.generator_calls == 1
    assert not result.revision_used
    assert generator.call_count == 1


def test_critic_checks_hard_constraints_against_untrusted_retrieval_result():
    retriever = CatalogRetriever.from_json(CATALOG)
    retrieval = retriever.retrieve(prefs())
    answer = SingleAgentAnswer.model_validate_json(payload())
    verdict = CriticVerificationAgent().review(
        answer, prefs(max_budget_usd=Decimal("1")), retrieval,
    )
    assert not verdict.approved
    assert "constraint_violation" in {i.code for i in verdict.issues}


def test_llm_abstention_is_an_explicit_non_recommendation():
    generator = SequenceGenerator(json.dumps({
        "recommendations": [], "no_suitable_match": True,
        "note": "Insufficient evidence",
    }))
    result = workflow(generator).run(prefs())
    assert result.status == "abstained"
    assert result.answer.no_suitable_match
    assert result.generator_calls == 1


def test_revisions_must_be_only_zero_or_one():
    agent = PreferenceRetrievalAgent(CatalogRetriever.from_json(CATALOG))
    generator = RecommendationAgent(SequenceGenerator(payload()))
    critic = CriticVerificationAgent()
    for budget in (-1, 2, True, 1.1):
        with pytest.raises(ValueError, match="max_revisions"):
            MultiAgentOrchestrator(agent, generator, critic, max_revisions=budget)


def test_stage_3_does_not_mutate_single_agent_baseline_or_catalog():
    baseline = Path(__file__).resolve().parents[1] / "src/recsys/baseline/pipeline.py"
    assert "class SingleAgentRAG" in baseline.read_text()
    catalog = json.loads(CATALOG.read_text())
    assert len(catalog["products"]) == 3
    assert len(catalog["evidence"]) == 3
