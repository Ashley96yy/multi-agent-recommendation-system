"""Offline model-based Critic tests; all model outputs here are mock strings."""

import json

import pytest

from recsys.evaluation.atomic_claims import load_frozen_claims, FOUR_LABELS
from recsys.evaluation.model_critic import (
    ModelClaimVerdict, SYSTEM, build_evidence_context, evaluate_predictions,
    parse_verdict, run_model_critic,
)
from recsys.evaluation.source_audit import SourceFixture
from recsys.evaluation.atomic_claims import ROOT


class FakeGenerator:
    def __init__(self, *, malformed_indices=frozenset()):
        self.call_count = 0
        self.last_prompt_tokens = 123
        self.last_generated_tokens = 17
        self.last_generation_seconds = 0.01
        self.max_new_tokens = 96
        self.name = "offline-mock"
        self.malformed_indices = malformed_indices
        self.contexts = []

    def generate(self, instructions, context):
        self.call_count += 1
        self.contexts.append(json.loads(context))
        assert instructions == SYSTEM
        if self.call_count in self.malformed_indices:
            return "not a json object"
        return json.dumps({"label": "supported", "rationale": "An uncalibrated mock decision."})


def setup():
    suite, gold, guard, hashes = load_frozen_claims()
    fixture = SourceFixture.model_validate_json((ROOT / suite.sources_path).read_bytes())
    return suite, gold, guard, fixture, hashes


def test_real_prompt_uses_primary_passages_and_never_sees_author_oracle():
    suite, gold, guard, fixture, hashes = setup()
    for case in suite.cases:
        context = build_evidence_context(case, fixture, guard._evidence)
        data = json.loads(context)
        assert data["atomic_claim_text"] == case.claim_span
        assert data["product_id"] == case.product_id
        assert "author_label" not in context
        assert "author_rationale" not in context
        assert "oracle" not in context.lower()
        assert all("verbatim_text" in doc and "is_superseded" in doc
                   for doc in data["product_primary_source_documents"])
    revised = next(c for c in suite.cases if c.case_id == "T10")
    data = json.loads(build_evidence_context(revised, fixture, guard._evidence))
    old = next(x for x in data["product_primary_source_documents"]
               if x["source_id"] == "DOC-E03-SPEC-V1")
    assert old["is_superseded"] is True
    assert any("non-mineral" in d["verbatim_text"]
               for d in data["product_primary_source_documents"])
    cross = next(c for c in suite.cases if c.case_id == "T15")
    cited = json.loads(build_evidence_context(cross, fixture, guard._evidence))["citation_lookups"]
    assert any(x["product_id"] == "SYN-E02" for x in cited)


def test_strict_verdict_format_and_whole_json_fence_only():
    valid = json.dumps({"label": "contradicted", "rationale": "Newer active source differs."})
    verdict, err = parse_verdict(valid)
    assert not err and verdict.label == "contradicted"
    verdict, err = parse_verdict("```json\n"+valid+"\n```")
    assert not err and verdict.label == "contradicted"
    for raw in (
        '{"label":"made-up","rationale":"bad"}',
        '{"label":"supported"}',
        '{"label":"supported","rationale":"ok","invented":true}',
        'here is a verdict '+valid,
        '{"label":"supported","rationale":""}',
    ):
        verdict, err = parse_verdict(raw)
        assert verdict is None and err == "invalid_json_or_schema"


def test_mock_model_is_called_once_per_claim_with_26_recorded_outputs():
    suite, gold, guard, fixture, hashes = setup()
    generator = FakeGenerator(malformed_indices={2, 26})
    report = run_model_critic(suite, fixture, guard._evidence, generator)
    assert report["physical_model_invocations"] == 26
    assert report["failed_or_invalid_count"] == 2
    assert generator.call_count == 26
    assert len(report["predictions"]) == 26
    assert report["total_prompt_tokens"] == 26*123
    assert report["total_generated_tokens"] == 26*17
    assert all(len(row["context_sha256"]) == 64 for row in report["predictions"])
    assert all("author_label" not in str(obj) for obj in generator.contexts)
    scores = evaluate_predictions(suite, gold, report)
    assert scores["all"]["cases"] == 26
    assert scores["typed"]["cases"] == 22
    assert scores["open_text"]["cases"] == 4
    assert scores["all"]["valid_predictions"] == 24
    assert scores["all"]["abstentions_or_invalid"] == 2
    assert scores["independent_human_review_complete"] is False
    assert scores["human_calibrated_accuracy"] is None


def test_unknown_labels_missing_cases_and_duplicate_ids_rejected():
    suite, gold, guard, fixture, hashes = setup()
    model = FakeGenerator()
    report = run_model_critic(suite, fixture, guard._evidence, model)
    copy = json.loads(json.dumps(report))
    copy["predictions"][0]["prediction"] = "anything"
    with pytest.raises(ValueError, match="unrecognized"):
        evaluate_predictions(suite, gold, copy)
    copy = json.loads(json.dumps(report))
    copy["predictions"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        evaluate_predictions(suite, gold, copy)
    copy = json.loads(json.dumps(report))
    copy["predictions"][1]["case_id"] = copy["predictions"][0]["case_id"]
    with pytest.raises(ValueError, match="incomplete or duplicate"):
        evaluate_predictions(suite, gold, copy)


def test_macro_scoring_counts_invalid_decisions_as_misses():
    suite, gold, guard, fixture, hashes = setup()
    model = FakeGenerator(malformed_indices=set(range(1, 27)))
    predictions = run_model_critic(suite, fixture, guard._evidence, model)
    scores = evaluate_predictions(suite, gold, predictions)
    assert scores["all"]["coverage"] == 0
    assert scores["all"]["correct_overall"] == 0
    assert scores["all"]["accuracy_including_abstentions"] == 0
    assert scores["all"]["selective_accuracy_valid_only"] is None
    assert scores["all"]["macro_f1_including_abstentions"] == 0.0
