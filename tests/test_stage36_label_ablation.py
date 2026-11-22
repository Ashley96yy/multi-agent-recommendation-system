"""Offline 3-arm ablation accounting. These test doubles are NOT Qwen results."""

import json

import pytest

from recsys.evaluation.atomic_claims import load_frozen_claims
from recsys.evaluation.model_critic import build_evidence_context
from recsys.evaluation.run_label_ablation import (
    ARM_CONFIG, DECISION_PROMPT, generate_arms_without_labels,
    parse_bare_label, score_after_inference,
)
from recsys.evaluation.source_audit import SourceFixture
from recsys.evaluation.atomic_claims import ROOT
import recsys.evaluation.run_label_ablation as ablation


class FakeBase:
    name = "offline-test-model"
    last_prompt_tokens = 120
    last_generated_tokens = 3
    last_generation_seconds = 0.02

    def __init__(self):
        self.call_count = 0
        self.contexts = []

    def generate(self, instructions, context):
        self.call_count += 1
        self.contexts.append(json.loads(context))
        assert "author_label" not in context
        assert "rationale" not in context
        return "ambiguity" if instructions == DECISION_PROMPT else "supported"


class FakeConstrained:
    def __init__(self, base):
        self.base = base

    def generate(self, instructions, context):
        assert instructions == DECISION_PROMPT
        self.base.call_count += 1
        self.base.last_generated_tokens = 2
        self.base.contexts.append(json.loads(context))
        return "contradicted"


def fixture():
    suite, gold, guard, hashes = load_frozen_claims()
    sources = SourceFixture.model_validate_json((ROOT / suite.sources_path).read_bytes())
    return suite, gold, guard, sources


def test_unmodified_label_parser_never_converts_synonyms_or_json():
    for raw in ("supported", "contradicted", "unsupported", "ambiguous"):
        assert parse_bare_label(raw) == (raw, None)
    for bad in ("ambiguity", "ambiguously", "Supported", "supported\n",
                '"supported"', '{"label":"supported"}', " support"):
        label, reason = parse_bare_label(bad)
        assert label == "needs_human_review"
        assert reason == "not_an_exact_canonical_label"


def test_three_new_model_arms_call_once_per_case_and_isolate_two_factors(monkeypatch):
    monkeypatch.setattr(ablation, "ConstrainedLabelQwen", FakeConstrained)
    suite, gold, guard, sources = fixture()
    model = FakeBase()
    run = generate_arms_without_labels(suite, sources, guard._evidence, model)
    assert run["physical_model_invocations"] == 78
    assert model.call_count == 78
    assert set(run["arms"]) == {
        "plain_label", "decision_label", "decision_label_constrained"
    }
    plain = run["arms"]["plain_label"]
    decision = run["arms"]["decision_label"]
    constrained = run["arms"]["decision_label_constrained"]
    assert plain["valid_label_outputs"] == 26
    assert decision["invalid_outputs"] == 26
    assert constrained["valid_label_outputs"] == 26
    assert all(len(arm["per_case"]) == 26 and arm["physical_model_invocations"] == 26
               for arm in run["arms"].values())
    assert decision["prompt_sha256"] == constrained["prompt_sha256"]
    assert plain["prompt_sha256"] != decision["prompt_sha256"]
    for a, b in zip(decision["per_case"], constrained["per_case"]):
        assert a["case_id"] == b["case_id"]
        assert a["context_sha256"] == b["context_sha256"]
        assert a["prompt_sha256"] == b["prompt_sha256"]
        assert a["raw_output_contract"] == "one_exact_lowercase_label_no_json"
    scored = score_after_inference(suite, gold, run)
    assert scored["decision_label"]["all"]["correct_overall"] == 0
    assert scored["decision_label"]["all"]["coverage"] == 0
    assert scored["decision_label_constrained"]["all"]["coverage"] == 1
    assert scored["decision_label_constrained"]["typed"]["cases"] == 22
    assert scored["decision_label_constrained"]["open_text"]["cases"] == 4
    assert scored["decision_label_constrained"]["human_calibrated_accuracy"] is None
    assert len(model.contexts) == 78
    assert all("author_label" not in str(context) for context in model.contexts)


def test_previously_fixed_atomic_claim_evidence_inputs_are_not_mutated(monkeypatch):
    monkeypatch.setattr(ablation, "ConstrainedLabelQwen", FakeConstrained)
    suite, gold, guard, sources = fixture()
    context = build_evidence_context(suite.cases[0], sources, guard._evidence)
    model = FakeBase()
    generate_arms_without_labels(suite, sources, guard._evidence, model)
    assert build_evidence_context(suite.cases[0], sources, guard._evidence) == context
    assert len(gold) == 26
    assert len(ARM_CONFIG) == 3
