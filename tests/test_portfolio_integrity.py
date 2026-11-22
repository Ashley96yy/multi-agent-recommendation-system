"""Stage 4.1 checks: curated source links and honest experiment claims.

These validate portfolio documentation only. They are NOT new model tests,
independent human measurements, or reinterpretation of the original runs.
"""

import json
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/portfolio/evidence_register.json"
NEW_DOCS = (
    ROOT / "README.md",
    ROOT / "docs/FINAL_EVALUATION.md",
    ROOT / "docs/SYSTEM_ARCHITECTURE.md",
    ROOT / "docs/RECRUITER_DEMO.md",
)


def registry():
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_curated_historical_evidence_has_exactly_nine_unique_verified_sources():
    data = registry()
    assert data["schema_version"] == 1
    assert data["repository"] == "Ashley96yy/multi-agent-recommendation-system"
    assert data["new_model_experiments_in_closeout"] == 0
    assert data["independent_human_groundedness"] is None
    assert data["real_world_unsupported_claim_rate"] is None
    studies = data["records"]
    assert len(studies) == 9
    assert len({x["study_id"] for x in studies}) == len(studies)
    assert len({x["source_doc"] for x in studies}) == len(studies)
    for entry in studies:
        assert (ROOT / entry["source_doc"]).is_file()
        assert entry["metrics"] and entry["cohort"] and entry["valid_interpretation"]
        assert entry["evidence_type"]
        if entry["run_url"] is not None:
            assert entry["run_url"].startswith(
                "https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/"
            )


def test_closeout_register_keeps_physical_calls_and_limits_distinct():
    studies = {r["study_id"]: r for r in registry()["records"]}
    assert studies["stage22_single_agent"]["metrics"]["positive_strict_success"] == "2/19"
    assert studies["stage22_single_agent"]["metrics"]["real_model_calls"] == 19
    assert studies["stage23_format_replay"]["metrics"]["new_llm_calls"] == 0
    assert studies["stage23_format_replay"]["metrics"]["normalized_positive_success"] == "12/19"
    assert studies["stage31_paired_multiagent"]["metrics"]["strict_positive_success_no_revision"] == "0/19"
    assert studies["stage31_paired_multiagent"]["metrics"]["strict_positive_success_one_revision"] == "0/19"
    assert studies["stage31_paired_multiagent"]["metrics"]["total_physical_calls"] == 35
    assert studies["stage33_live_preflight"]["metrics"]["logical_calls_without_preflight"] == 19
    assert studies["stage33_live_preflight"]["metrics"]["logical_calls_with_preflight"] == 10
    assert studies["stage33_live_preflight"]["metrics"]["matrix_physical_calls"] == 19
    assert studies["stage34_typed_claims"]["metrics"]["typed_rule_author_agreement"] == "22/22"
    assert studies["stage34_typed_claims"]["metrics"]["independent_human_reviews"] == 0
    assert studies["stage35_model_critic"]["metrics"]["correct_against_authored_labels"] == "0/26"
    assert studies["stage35_model_critic"]["metrics"]["strict_valid_schema"] == "3/26"
    assert studies["stage36_constraint_ablation"]["metrics"]["physical_calls"] == 78
    assert studies["stage36_constraint_ablation"]["metrics"]["constrained_prediction_changes"] == 0
    assert studies["stage36_constraint_ablation"]["metrics"]["constrained_label_author_agreement"] == "8/26"
    assert studies["stage40_interactive_demo"]["metrics"]["scripted_scenarios_passed"] == "6/6"
    assert studies["stage40_interactive_demo"]["metrics"]["real_smoke_physical_calls"] == 1
    assert studies["stage40_interactive_demo"]["metrics"]["tests_in_optional_e2e_job"] == 179


@pytest.mark.parametrize("path", NEW_DOCS)
def test_all_new_document_local_markdown_links_exist(path):
    text = path.read_text(encoding="utf-8")
    # Links are constrained to ordinary relative docs, not GitHub URLs,
    # source-code spans or Mermaid node syntax.
    relative_links = re.findall(r"\]\(([^)]+)\)", text)
    assert relative_links, f"{path.name} has no references"
    for literal in relative_links:
        target = literal.strip().split("#", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        normalized = (path.parent / target).resolve()
        assert normalized.is_relative_to(ROOT), f"external repo path: {literal}"
        assert normalized.is_file(), f"broken local link in {path}: {literal}"


def test_final_overview_links_every_primary_experiment_and_preserves_limitations():
    final = (ROOT / "docs/FINAL_EVALUATION.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for entry in registry()["records"]:
        assert Path(entry["source_doc"]).name in final
        if entry["run_url"] is not None:
            assert entry["run_url"] in final
    for expected in (
        "2/19", "12/19", "0/19", "19", "10", "22/22", "8/22",
        "3/26", "0/26", "8/26", "179", "6/6",
    ):
        assert expected in final
    for key in (
        "fictional", "not", "unmeasured", "deterministic", "revision", "human",
    ):
        assert key in final.lower()
    assert "PREWRITTEN" in readme
    assert "not three LLMs" in readme
    assert "no hosted public app" in readme
    assert "real-world unsupported-claim rate" in final.lower()


def test_recruiter_walkthrough_has_no_unrun_model_quality_claim():
    guide = (ROOT / "docs/RECRUITER_DEMO.md").read_text(encoding="utf-8")
    assert "PREWRITTEN" in guide
    assert "two-minute" in guide.lower()
    assert "not a correctness benchmark" in guide
    assert "0/19" in guide
    assert "26/26" in guide
    assert "8/26" in guide
    assert "no" in guide.lower() and "hosted/public URL" in guide
