"""Offline audit tests using fake model responses, NOT fabricated model scores."""

from hashlib import sha256
import json

import pytest

from recsys.evaluation.audit_model_critic_outputs import audit_model_output


def fake_record():
    raw = [
        '{"label":"ambiguous","rationale":"Conflicting sources."}',
        '{"label":"ambiguity","rationale":"Uncertainty."}',
        '{"label":"ambiguous","rationale":"he says "maybe" without details."}',
    ]
    result = raw + [raw[0]] * 23
    records = []
    scored = []
    for i, s in enumerate(result):
        pred = "ambiguous" if i != 1 and i != 2 else "needs_human_review"
        records.append({"case_id": f"C{i:02d}", "raw_model_response": s, "prediction": pred})
        scored.append({"case_id": f"C{i:02d}", "match": i == 0})
    return {
        "suite_id": "atomic-claims-fictional-v1",
        "model": "offline-test-double",
        "source_sha256": {"cases_sha256": "a" * 64},
        "generator": {"predictions": records, "physical_model_invocations": 26},
        "comparison_against_author_labels": {
            "all": {"valid_predictions": 24, "correct_overall": 1},
            "per_case": scored,
        },
    }


def test_lexical_raw_labels_do_not_become_predictions():
    data = fake_record()
    out = audit_model_output(data, report_sha256="a"*64)
    assert out["raw_json_parseable_count"] == 25
    assert out["raw_json_invalid_count"] == 1
    assert out["strict_schema_compliant_count"] == 24
    assert out["parseable_but_invalid_schema_count"] == 1
    assert out["raw_label_prefix_counts_not_predictions"] == {
        "ambiguous": 25, "ambiguity": 1
    }
    assert out["repaired_output_count"] == 0
    assert out["additional_llm_calls"] == 0
    assert out["author_label_correct_including_abstentions"] == 1
    assert out["per_case"][1]["actual_scored_prediction"] == "needs_human_review"
    assert out["per_case"][2]["actual_scored_prediction"] == "needs_human_review"


def test_modified_predictions_or_missing_case_id_refused():
    data = fake_record()
    data["generator"]["predictions"][0]["prediction"] = "supported"
    with pytest.raises(ValueError, match="strict source parser"):
        audit_model_output(data, report_sha256="a"*64)
    data = fake_record()
    data["generator"]["predictions"][1]["case_id"] = "C00"
    with pytest.raises(ValueError, match="duplicate"):
        audit_model_output(data, report_sha256="a"*64)


def test_traces_and_previously_recorded_metrics_must_reconcile():
    data = fake_record()
    data["comparison_against_author_labels"]["all"]["valid_predictions"] = 2
    with pytest.raises(ValueError, match="schema audit"):
        audit_model_output(data, report_sha256="a"*64)
    data = fake_record()
    data["comparison_against_author_labels"]["all"]["correct_overall"] = 5
    with pytest.raises(ValueError, match="author-label metrics inconsistent"):
        audit_model_output(data, report_sha256="a"*64)
