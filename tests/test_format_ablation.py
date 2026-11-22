"""No-model format-repair tests: no factual claims or ID edits are allowed."""

import json

import pytest

from recsys.evaluation.format_ablation import _check_output, normalize_output


RESPONSE = {
    "recommendations": [{
        "product_id": "SYN-R11",
        "reason": "An unsupported claim must never be rewritten by a parser.",
        "evidence_ids": ["E-SYN-R11-T"],
    }],
    "no_suitable_match": False,
}


def test_strict_mode_preserves_exact_raw_response():
    raw = json.dumps(RESPONSE)
    processed, actions = normalize_output(raw, "strict")
    assert processed == raw
    assert actions == []


def test_whole_markdown_json_fences_removed_only_when_exactly_wrapped():
    raw = "```json\n" + json.dumps(RESPONSE) + "\n```"
    result, actions = normalize_output(raw, "fences_only")
    assert json.loads(result) == RESPONSE
    assert actions == ["strip_whole_json_fence"]
    embedded = 'Here are the results:\n' + raw
    assert normalize_output(embedded, "fences_only") == (embedded, [])


def test_missing_flag_only_filled_for_nonempty_recommendations():
    without = {k: v for k, v in RESPONSE.items() if k != "no_suitable_match"}
    fixed, actions = normalize_output(json.dumps(without), "fences_and_missing_flag")
    assert json.loads(fixed) == RESPONSE
    assert actions == ["set_missing_no_suitable_match_false"]
    empty, actions = normalize_output('{"recommendations":[]}', "fences_and_missing_flag")
    assert json.loads(empty) == {"recommendations": []}
    assert actions == []


def test_never_change_conflicting_bool_or_repair_invalid_json():
    conflicting = {**RESPONSE, "no_suitable_match": True}
    raw = json.dumps(conflicting)
    fixed, actions = normalize_output(raw, "fences_and_missing_flag")
    assert json.loads(fixed) == conflicting
    assert actions == []
    assert normalize_output("some text {broken}", "fences_and_missing_flag") == (
        "some text {broken}", [],
    )


def test_normalization_preserves_reason_product_id_and_evidence_ids():
    raw = "```json\n" + json.dumps({k: v for k, v in RESPONSE.items()
                                         if k != "no_suitable_match"}) + "\n```"
    fixed, actions = normalize_output(raw, "fences_and_missing_flag")
    normalized = json.loads(fixed)
    assert normalized["recommendations"] == RESPONSE["recommendations"]
    assert actions == ["strip_whole_json_fence", "set_missing_no_suitable_match_false"]


def test_known_product_valid_citation_is_accepted_and_unsafe_reason_left_unchanged():
    raw = json.dumps(RESPONSE)
    parsed, actions, error = _check_output(
        raw, "strict", ["SYN-R11"], {"SYN-R11": {"E-SYN-R11-T"}}
    )
    assert error is None and actions == []
    assert parsed.recommendations[0].reason == RESPONSE["recommendations"][0]["reason"]


def test_unknown_product_and_cross_product_citations_not_repaired():
    raw = json.dumps(RESPONSE)
    _, _, err = _check_output(raw, "fences_and_missing_flag", ["SYN-R12"],
                               {"SYN-R11": {"E-SYN-R11-T"}})
    assert err == "product_outside_retrieval"
    _, _, err = _check_output(raw, "fences_and_missing_flag", ["SYN-R11"],
                               {"SYN-R11": {"E-SYN-R12-T"}})
    assert err == "absent_or_cross_product_citation"


def test_missing_required_membership_still_causes_schema_error_when_not_safe_to_infer():
    raw = '{"recommendations":[]}'
    answer, actions, err = _check_output(raw, "fences_and_missing_flag",
                                        ["SYN-R11"], {})
    assert answer is None and err == "invalid_schema_or_json"


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="unsupported"):
        normalize_output(json.dumps(RESPONSE), "rewrite_reasons")
