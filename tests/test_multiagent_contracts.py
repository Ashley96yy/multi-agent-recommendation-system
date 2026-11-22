"""Strict schema and execution-trace integrity tests."""

import pytest
from pydantic import ValidationError

from recsys.multiagent.contracts import AgentAttempt, CriticIssue, CriticVerdict, MultiAgentResult


def issue():
    return CriticIssue(code="invalid_generation", detail="Bad JSON")


def verdict(bad=False):
    return CriticVerdict(approved=not bad, issues=[issue()] if bad else [])


def test_verdict_cannot_claim_approval_when_issues_exist():
    with pytest.raises(ValidationError, match="approved must agree"):
        CriticVerdict(approved=True, issues=[issue()])
    with pytest.raises(ValidationError, match="approved must agree"):
        CriticVerdict(approved=False, issues=[])


def test_entailment_flag_cannot_be_turned_true():
    with pytest.raises(ValidationError):
        CriticVerdict(approved=True, semantic_entailment_verified=True)


def test_agent_attempt_rejects_non_bounded_call_count():
    with pytest.raises(ValidationError):
        AgentAttempt(attempt_number=1, is_revision=False, parse_valid=True,
                     verdict=verdict(), generator_calls=2)
    with pytest.raises(ValidationError):
        AgentAttempt(attempt_number=3, is_revision=True, parse_valid=True,
                     verdict=verdict(), generator_calls=1)


def test_execution_trace_extra_keys_are_rejected():
    with pytest.raises(ValidationError):
        CriticIssue(code="invalid_generation", detail="Invalid", invented="hidden value")


def test_unapproved_proposal_cannot_be_represented_as_final_answer():
    # A result cannot expose any answer for a rejected status, even if
    # a caller tries to manufacture one.
    from recsys.baseline.contracts import SingleAgentAnswer
    from recsys.retrieval import CatalogRetriever
    from recsys.schemas import UserPreferences
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data/sample_catalog/catalog.json"
    retrieval = CatalogRetriever.from_json(path).retrieve(
        UserPreferences(request="moisturizer")
    )
    with pytest.raises(ValidationError, match="cannot expose"):
        MultiAgentResult(
            status="rejected", retrieval=retrieval,
            answer=SingleAgentAnswer(recommendations=[], no_suitable_match=True),
            attempts=[AgentAttempt(attempt_number=1, is_revision=False, parse_valid=False,
                                   verdict=verdict(bad=True), generator_calls=1)],
            generator_calls=1,
        )
