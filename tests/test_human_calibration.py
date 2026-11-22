"""Simulated review sheets exercise validation; NOT real human annotations."""

import csv
import json
from pathlib import Path

import pytest

from recsys.evaluation.human_calibration import (
    build_review_packet, assess_independent_reviews,
)
from recsys.evaluation.atomic_claims import load_frozen_claims


def read(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def create_completed_fake_reviews(tmp_path):
    path = tmp_path / "blind"
    manifest = build_review_packet(path)
    labels = load_frozen_claims()[1]
    a_path = path / "stage35-blind-reviewer-A.csv"
    b_path = path / "stage35-blind-reviewer-B.csv"
    for reviewer_id, file in [("mock-reviewer-a", a_path), ("mock-reviewer-b", b_path)]:
        rows = read(file)
        for row in rows:
            row["reviewer_id"] = reviewer_id
            row["reviewer_label"] = labels[row["case_id"]].label
            row["review_notes"] = "Synthetic fixture test annotation, not a real review."
        write(file, rows)
    return manifest, a_path, b_path


def test_blind_packet_contains_all_evidence_but_no_label_leakage(tmp_path):
    root = tmp_path / "packet"
    manifest = build_review_packet(root)
    assert manifest["cases"] == 26
    assert manifest["independent_human_agreement"] is None
    assert manifest["reviewer_submissions_received"] == 0
    for who in "AB":
        rows = read(root / f"stage35-blind-reviewer-{who}.csv")
        assert len(rows) == 26
        assert all(row["reviewer_label"] == "" and row["reviewer_id"] == "" for row in rows)
        assert all("author_label" not in row and "model_prediction" not in row for row in rows)
        assert all(len(row["source_context_sha256"]) == 64 for row in rows)
        assert all(json.loads(row["source_context_json"])["product_primary_source_documents"] for row in rows)
    assert json.loads((root / "stage35-human-review-manifest.json").read_text()) == manifest


def test_two_signed_complete_mock_reviews_allow_kappa_one(tmp_path):
    _, a_path, b_path = create_completed_fake_reviews(tmp_path)
    result = assess_independent_reviews(a_path, b_path)
    assert result["independent_reviewer_count"] == 2
    assert result["independent_agreement"] == 1.0
    assert result["cohens_kappa"] == 1.0
    assert result["consensus_cases"] == 26
    assert result["fully_adjudicated"] is True
    assert result["human_calibrated_model_accuracy"] is None


def test_disagreements_not_auto_resolved(tmp_path):
    _, a_path, b_path = create_completed_fake_reviews(tmp_path)
    rows = read(b_path)
    rows[0]["reviewer_label"] = (
        "unsupported" if rows[0]["reviewer_label"] != "unsupported" else "supported"
    )
    write(b_path, rows)
    result = assess_independent_reviews(a_path, b_path)
    assert result["consensus_cases"] == 25
    assert result["fully_adjudicated"] is False
    assert len(result["disagreement_cases"]) == 1


def test_identical_ids_bad_source_hash_and_missing_labels_are_rejected(tmp_path):
    _, a_path, b_path = create_completed_fake_reviews(tmp_path)
    rows = read(b_path)
    for row in rows:
        row["reviewer_id"] = "mock-reviewer-a"
    write(b_path, rows)
    with pytest.raises(ValueError, match="distinct reviewer"):
        assess_independent_reviews(a_path, b_path)
    _, a_path, b_path = create_completed_fake_reviews(tmp_path)
    rows = read(b_path)
    rows[0]["source_context_sha256"] = "0"*64
    write(b_path, rows)
    with pytest.raises(ValueError, match="source context was modified"):
        assess_independent_reviews(a_path, b_path)
    _, a_path, b_path = create_completed_fake_reviews(tmp_path)
    rows = read(b_path)
    rows[0]["reviewer_label"] = ""
    write(b_path, rows)
    with pytest.raises(ValueError, match="completed labels"):
        assess_independent_reviews(a_path, b_path)


def test_one_reviewer_cannot_claim_human_calibrated_result(tmp_path):
    root = tmp_path / "pack"
    build_review_packet(root)
    a_path = root / "stage35-blind-reviewer-A.csv"
    b_path = root / "stage35-blind-reviewer-B.csv"
    with pytest.raises(ValueError, match="reviewer_id"):
        assess_independent_reviews(a_path, b_path)
