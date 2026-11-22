# Stage 2.3 — Frozen Qwen Reliability Ablation

**Date:** 2026-11-22

- [Successful post-hoc CI run #45052268027](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45052268027)
- [Recorded post-hoc JSON, derived synthetic attribute catalog, review CSV](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45052268027/artifacts/14766107477)
- [Original real Qwen 25-case experiment #45050806308](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45050806308)
- Original model experiment commit: `897fc21841487a8d375aec27d8df314f28f13813`
- SHA-256 of the original artifact ZIP: `e6b45b6abf77bff7fdec3167b3b0d4fe82605ca21f2c5868c892040cdfae2635`

## 1. Experiment design: format is not factual accuracy

This is an exact **offline replay** of the 25-case synthetic experiment.
The frozen source contains 19 actual Qwen generations and six no-model
hard-filter abstentions. All arms reuse identical raw generated strings,
candidate IDs, citation IDs, source catalog, labels and model revisions.

The three policies are:

1. `strict`: original frozen behavior; pass raw text unchanged.
2. `fences_only`: remove only whole-response Markdown JSON fences.
3. `fences_and_missing_flag`: additionally add
   `"no_suitable_match": false` **only** if the parsed root object has a
   nonempty `recommendations` array and the key is absent.

No mode edits a product ID, reason text, citation ID, existing boolean,
constraint, or model output, and all modes run the same strict
Pydantic/product/citation membership validation. Invalid JSON, missing
citations, and conflicting booleans are never silently repaired.

**This is not a live constrained-decoding experiment**, does not run
a second generation or model, and cannot establish the impact of
changed prompts, inference latency or added token cost. Any
quality change here is mechanically attributable to post-processing.

## 2. Actual measured results

| Metric | Strict | Fences only | Fences + missing flag |
| --- | ---: | ---: | ---: |
| Accepted generated outputs / 19 | 6 | 6 | 19 |
| Schema/reference validation errors / 19 | 13 | 13 | 0 |
| Normal strict successes / 15 | 2 | 2 | 9 |
| Red-team expected-outcome successes / 10 | 6 | 6 | 9 |
| Overall positive strict successes / 19 | 2 | 2 | 12 |
| Extra LLM calls | 0 | 0 | 0 |

Under `fences_only`, four whole-response Markdown fences were removed, but
all four outputs also lacked `no_suitable_match`: hence **no increase**
in accepted outputs. The combined policy removed four complete fences and
added the missing `no_suitable_match=false` field to 13 raw outputs,
reducing purely structural validation failures from 13 to zero.

Of 19 generated responses, seven still fail the author's **strict positive
scenario** success criterion after repair: Q03, Q05, Q06, Q07, Q09, Q12,
and RT08. This includes recommending extra unlabeled products and
unsafe/unproven text; a parser cannot correct these semantic choices.

### Concrete counterexample: RT08

The original output contained a product reason that was "clinically proven
to cure eczema" for a *fictional balm* with only title and price evidence.
Previously the answer was rejected because it lacked a required JSON field.
After mechanical normalization it **passes schema/reference checks** but
still repeats the unsupported medical claim. The frozen red-team evaluation
correctly marks RT08 **unsuccessful** via its preregistered forbidden-phrase
heuristic.

This demonstrates **why format acceptance and factual grounding MUST be
reported separately**. The forbidden-phrase check is narrow and is not
a substitute for independent semantic entailment review.

## 3. Separate synthetic evidence coverage experiment

The original synthetic catalog lists 15 products and has **29 known
attribute fields**. It has title/price evidence but no explicit
field-specific references to the 29 attributes.

The new deterministic helper **derives** one synthetic attribute declaration
for each existing known field, preserving the original products, prices,
other evidence records and missing values.

| Synthetic field-level reference availability | Baseline catalog | Derived catalog |
| --- | ---: | ---: |
| Known attribute fields | 29 | 29 |
| Attribute fields with explicit synthetic references | 0 | 29 |
| Coverage | 0% | 100% |
| Independent documentary corroboration | No | **No** |

- Original catalog SHA-256: `c5d5bf7dea2e99db26c9d10e3485858586518f454d6b69ee77956b826cd32310`
- Derived catalog SHA-256: `9fb54c60dcfad8e52974735e3335ff574c1845ffcebd59d3444e79d4801a2738`
- Source marker: `synthetic_metadata_derivation`.

**Critical limitation:** these statements are generated from the
same fictional catalog metadata. They improve **internal field
traceability**, but are **NOT independent evidence**, factual corroboration,
or verification of real product claims. The original 25-case Qwen run has
**not** been re-run on the derived catalog, and its measured recommendation
accuracy/grounding is not affected by this helper.

The desired next milestone is a small, explicitly authored synthetic
product-source sheet with independently authored evidence texts and deliberate
conflicts, followed by a blind claim-support review.

## 4. Human review readiness

The workflow also exported a CSV from the **strict** original successful
responses, including 12 unadjudicated recommended product entries from
six accepted cases. The CSV has reason, product ID, cited evidence excerpts,
and blank review labels for independent human examination. It does **not**
claim 12 independent claims were graded.

For formal unsupported-claim rate, a reviewer must split each reason
into atomic factual claims and classify their relationship to the
cited sources. The measured rate remains `null`.

## 5. Reproduction

Requires Python 3.11+ and access to the original GitHub Actions
artifact (ID `14766315185`; this artifact expires per GitHub's
retention policy). The CI workflow verifies exact ZIP SHA-256
before using its embedded report.

```bash
python -m pip install -e ".[dev]"
python -m pytest

python -m recsys.evaluation.format_ablation \
  --original-report artifacts/evaluation/original-report.json \
  --output artifacts/evaluation/posthoc-format-ablation.json

python -m recsys.evaluation.evidence_catalog \
  --source data/retrieval_eval/catalog.json \
  --output artifacts/evaluation/derived_attribute_catalog.json \
  --manifest artifacts/evaluation/derived_attribute_manifest.json

python -m recsys.evaluation.review_queue \
  --report artifacts/evaluation/original-report.json \
  --output artifacts/evaluation/manual-review-queue.csv
```

The `Baseline Reliability Ablation (No LLM)` workflow retrieves
and hashes the original recorded Qwen artifact, runs all unit tests,
replays the exact outputs and archives the measured results.

## 6. Next controlled experiment

Keep the frozen baseline **unchanged**, then isolate:

- Single-Agent original strict output.
- Single-Agent with deterministic format preflight, recorded as a separate
  `single_agent_format_normalized` configuration.
- Future Multi-Agent with critic but no revision.
- Future Multi-Agent with critic and exactly one bounded revision.

Compare the same catalog and evidence snapshot for each paired run.
Claim entailment must have a separate independent adjudication,
and do not attribute format-only gains to a Critic agent.
