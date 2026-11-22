"""Post-hoc output-contract failure audit of a FROZEN real-Qwen run.

No new model inference and no repair: labels outside the four frozen
classes remain invalid. Prefix regex inspection is for error diagnostics,
never for a hidden prediction or accuracy score.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from recsys.evaluation.model_critic import parse_verdict


LABEL_PREFIX = re.compile(r'"label"\s*:\s*"([^"]{1,80})"')


def audit_model_output(report: dict, *, report_sha256: str) -> dict:
    run = report.get("generator", {})
    items = run.get("predictions", [])
    if report.get("suite_id") != "atomic-claims-fictional-v1":
        raise ValueError("unexpected authored-claim source suite")
    if len(items) != 26 or run.get("physical_model_invocations") != 26:
        raise ValueError("must audit all 26 real Qwen calls")
    if len({row["case_id"] for row in items}) != 26:
        raise ValueError("duplicate model output case ID")
    details = []
    raw_value_counts = Counter()
    syntactic_json_count = 0
    strict_count = 0
    invalid_label_count = 0
    for row in items:
        raw = row["raw_model_response"]
        if not isinstance(raw, str):
            raise ValueError("missing original real model output")
        try:
            parsed = json.loads(raw)
            valid_json = True
            syntactic_json_count += 1
        except json.JSONDecodeError:
            parsed = None
            valid_json = False
        match = LABEL_PREFIX.search(raw)
        diagnostic_label = match.group(1) if match else None
        if diagnostic_label is not None:
            raw_value_counts[diagnostic_label] += 1
        valid, issue = parse_verdict(raw)
        expected_prediction = valid.label if valid is not None else "needs_human_review"
        if row["prediction"] != expected_prediction:
            raise ValueError("recorded prediction does not match strict source parser")
        if valid is not None:
            strict_count += 1
        elif valid_json and isinstance(parsed, dict) and "label" in parsed:
            invalid_label_count += 1
        details.append({
            "case_id": row["case_id"], "json_parseable": valid_json,
            "raw_label_prefix_for_diagnostics_only": diagnostic_label,
            "strict_schema_compliant": valid is not None,
            "actual_scored_prediction": expected_prediction,
            "parse_error": issue,
        })
    scored = report["comparison_against_author_labels"]
    if scored["all"]["valid_predictions"] != strict_count:
        raise ValueError("schema audit and recorded valid verdict count disagree")
    if scored["all"]["correct_overall"] != sum(x["match"] for x in scored["per_case"]):
        raise ValueError("author-label metrics inconsistent with scored case rows")
    return {
        "source_report_sha256": report_sha256,
        "source_suite_id": report["suite_id"],
        "source_model": report["model"],
        "source_frozen_input_hashes": report["source_sha256"],
        "real_model_invocations": run["physical_model_invocations"],
        "raw_json_parseable_count": syntactic_json_count,
        "raw_json_invalid_count": len(items) - syntactic_json_count,
        "strict_schema_compliant_count": strict_count,
        "parseable_but_invalid_schema_count": syntactic_json_count - strict_count,
        "parseable_with_invalid_label_count": invalid_label_count,
        "raw_label_prefix_counts_not_predictions": dict(raw_value_counts),
        "author_label_correct_including_abstentions": scored["all"]["correct_overall"],
        "real_human_calibration": None,
        "repaired_output_count": 0,
        "additional_llm_calls": 0,
        "per_case": details,
        "disclaimer": (
            "Diagnostic lexical prefix counts are NOT repaired semantic "
            "predictions, and no result has been upgraded. JSON and four-class "
            "contract compliance are distinct from actual claim understanding."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/evaluation/stage35-model-format-failure-audit.json"
    ))
    args = parser.parse_args()
    raw = args.report.read_bytes()
    audit = audit_model_output(json.loads(raw), report_sha256=sha256(raw).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({
        key: audit[key] for key in (
            "real_model_invocations", "raw_json_parseable_count",
            "raw_json_invalid_count", "strict_schema_compliant_count",
            "parseable_but_invalid_schema_count",
            "raw_label_prefix_counts_not_predictions",
            "author_label_correct_including_abstentions",
            "additional_llm_calls"
        )
    }, indent=2))


if __name__ == "__main__":
    main()
