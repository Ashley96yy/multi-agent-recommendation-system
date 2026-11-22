"""Stage 4.0 complete offline demo tests; zero actual Qwen downloads/calls."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from recsys.demo.service import (
    DemoService, ScriptedGenerator, TracingGenerator, WalkthroughSuite,
    preferences_from_input, render_markdown,
)
from recsys.schemas import UserPreferences


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data/demo/walkthroughs.json"


@pytest.fixture()
def service():
    return DemoService()


@pytest.mark.parametrize(
    "scenario,expected_status,calls",
    [
        ("consistent", "approved", 1),
        ("revision", "approved", 2),
        ("conflicting", "insufficient_evidence", 0),
        ("superseded", "insufficient_evidence", 0),
        ("missing", "insufficient_evidence", 0),
        ("hard_filter", "no_candidates", 0),
    ],
)
def test_all_fixed_reproducible_scenarios(service, scenario, expected_status, calls):
    record = service.run_walkthrough(scenario)
    assert record["mode"] == "scripted_fixture"
    assert record["status"] == expected_status
    assert record["logical_generator_attempts"] == calls
    assert record["actual_backend_calls"] == calls
    assert len(record["attempts"]) == calls
    assert record["generated_prompt_tokens"] is None
    assert record["generated_completion_tokens"] is None
    assert len(record["retrieval"]["catalog_sha256"]) == 64
    assert len(record["retrieval"]["source_docs_sha256"]) == 64
    assert record["independent_semantic_groundedness"] is None
    assert "PREWRITTEN" in record["attempts"][0]["backend_name"] if calls else True


def test_approved_response_is_grounded_to_fictional_source_and_citation(service):
    result = service.run_walkthrough("consistent")
    assert result["status"] == "approved"
    assert not result["revision_used"]
    assert result["final_approved_recommendations_only"][0]["product_id"] == "SYN-E01"
    refs = result["final_approved_recommendations_only"][0]["cited_evidence"]
    assert [e["evidence_id"] for e in refs] == ["E-SOURCE-001"]
    assert "made without added fragrance" in refs[0]["verbatim_excerpt"]
    assert refs[0]["source_superseded"] is False
    assert "unverified" in render_markdown(result).lower()


def test_revision_does_not_leak_rejected_medical_claim_to_final_recommendations(service):
    report = service.run_walkthrough("revision")
    assert report["revision_used"]
    assert [x["is_revision"] for x in report["attempts"]] == [False, True]
    assert [x["critic_approved"] for x in report["attempts"]] == [False, True]
    assert "prohibited_claim_phrase" in {
        issue["code"] for issue in report["attempts"][0]["critic_issues"]
    }
    assert "clinically proven" in report["attempts"][0]["raw_unverified_output"]
    assert "clinically proven" not in render_markdown(report)
    assert all(
        "clinically proven" not in item["reason_unverified_natural_language"]
        for item in report["final_approved_recommendations_only"]
    )


@pytest.mark.parametrize("scenario,status", [
    ("conflicting", "conflict"),
    ("superseded", "stale_metadata_after_revision"),
    ("missing", "missing"),
])
def test_unsafe_required_fields_fail_before_any_generation(service, scenario, status):
    report = service.run_walkthrough(scenario)
    assert report["status"] == "insufficient_evidence"
    assert report["attempts"] == []
    assert report["final_approved_recommendations_only"] == []
    assert any(x["status"] == status for x in report["source_field_diagnostics"])
    assert len(report["source_evidence"]) > 0
    if scenario == "superseded":
        assert any(x["source_superseded"] for x in report["source_evidence"])


def test_hard_filtered_request_has_no_candidates_and_no_source_recommendation(service):
    report = service.run_walkthrough("hard_filter")
    assert report["status"] == "no_candidates"
    assert report["retrieval"]["filtered_by_reason"]["over_budget"] >= 1
    assert report["retrieval"]["candidates"] == []
    assert report["source_evidence"] == []
    assert "No unapproved recommendation" in render_markdown(report)


def test_live_injected_model_obeys_same_original_orchestrator_and_guard(service):
    original = service.run_walkthrough("consistent")
    raw = original["attempts"][0]["raw_unverified_output"]
    generator = ScriptedGenerator([raw])
    prefs = preferences_from_input(
        request="Cloud Calm Facial Cream fragrance-free moisturizer",
        required_category="moisturizer", required_attributes=["fragrance_free"],
        max_budget_usd="25.00",
    )
    report = service.run_live(prefs, generator=generator)
    assert report["mode"] == "injected_generator_test"
    assert report["status"] == "approved"
    assert generator.call_count == 1
    assert report["actual_backend_calls"] == 1
    assert report["generated_prompt_tokens"] is None


def test_live_invalid_generator_outputs_are_visible_only_as_unverified_trace(service):
    generator = ScriptedGenerator(['<script>alert(1)</script>', 'not-json'])
    prefs = UserPreferences(request="Cloud Calm Facial Cream", required_category="moisturizer",
                            required_attributes={"fragrance_free"})
    report = service.run_live(prefs, generator=generator)
    assert report["status"] == "rejected"
    assert report["logical_generator_attempts"] == 2
    assert report["actual_backend_calls"] == 2
    assert report["final_approved_recommendations_only"] == []
    assert all(x["critic_issues"][0]["code"] == "invalid_generation" for x in report["attempts"])
    assert "<script>" not in render_markdown(report)


def test_no_eligible_candidate_never_initializes_qwen(service, monkeypatch):
    import recsys.demo.service as module
    monkeypatch.setattr(module, "_pinned_qwen_cpu", lambda: (_ for _ in ()).throw(
        AssertionError("Qwen model must not initialize on fail-closed path")
    ))
    prefs = preferences_from_input(request="Cloud Calm Facial Cream",
                                   required_category="moisturizer",
                                   max_budget_usd="0")
    report = service.run_live(prefs)
    assert report["status"] == "no_candidates"
    assert report["actual_backend_calls"] == 0
    conflict = preferences_from_input(request="Dusk Barrier Balm",
                                      required_category="balm",
                                      required_attributes=["fragrance_free"])
    report2 = service.run_live(conflict)
    assert report2["status"] == "insufficient_evidence"
    assert report2["actual_backend_calls"] == 0


def test_preferences_validate_decimal_filters_and_exclusions():
    p = preferences_from_input(
        request="Cloud Calm", required_category="moisturizer",
        max_budget_usd="25.90",
        required_attributes=["fragrance_free", "fragrance_free"],
        excluded_product_ids="SYN-E02, SYN-E04,",
        excluded_brands=" Fictional Orbit, ",
    )
    assert str(p.max_budget_usd) == "25.90"
    assert p.required_attributes == {"fragrance_free"}
    assert p.excluded_product_ids == {"SYN-E02", "SYN-E04"}
    assert p.excluded_brands == {"Fictional Orbit"}
    for budget in ("-1", "NaN", "Infinity", "not a number"):
        with pytest.raises(ValueError, match="Budget"):
            preferences_from_input(request="cream", max_budget_usd=budget)
    with pytest.raises(ValueError, match="request"):
        preferences_from_input(request=" ")
    with pytest.raises(ValueError, match="request"):
        preferences_from_input(request="x" * 1001)


def test_fixed_replay_guard_against_unknown_scenario_or_tampered_expected_outcome(service, tmp_path):
    with pytest.raises(ValueError, match="Unknown fixed"):
        service.run_walkthrough("custom script")
    raw = json.loads(CASES.read_text())
    raw["cases"][0]["expected_status"] = "rejected"
    modified = tmp_path / "drift.json"
    modified.write_text(json.dumps(raw))
    other = DemoService(walkthroughs_path=modified)
    with pytest.raises(RuntimeError, match="drifted"):
        other.run_walkthrough("consistent")


def test_fixture_duplicate_ids_and_unknown_fields_are_rejected():
    raw = json.loads(CASES.read_text())
    raw["cases"][1]["scenario_id"] = raw["cases"][0]["scenario_id"]
    with pytest.raises(ValidationError, match="duplicate scripted"):
        WalkthroughSuite.model_validate_json(json.dumps(raw))
    raw = json.loads(CASES.read_text())
    raw["cases"][0]["invented_result"] = 1
    with pytest.raises(ValidationError, match="Extra inputs"):
        WalkthroughSuite.model_validate_json(json.dumps(raw))


def test_trace_detects_generator_errors_without_exposing_unapproved_proposals(service):
    class Broken:
        call_count = 0
        name = "broken-test"

        def generate(self, instructions, context):
            self.call_count += 1
            raise RuntimeError("broken upstream")

    prefs = UserPreferences(
        request="Cloud Calm Facial Cream", required_category="moisturizer",
        required_attributes={"fragrance_free"}
    )
    report = service.run_live(prefs, generator=Broken(), max_revisions=0)
    assert report["status"] == "rejected"
    assert report["attempts"][0]["transport_error"] == "RuntimeError"
    assert report["actual_backend_calls"] == 1
    assert report["final_approved_recommendations_only"] == []
