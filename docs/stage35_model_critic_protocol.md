# Stage 3.5 — Model-Based Claim Critic and Independent Human Calibration Protocol

**Scope:** real Qwen2.5-0.5B-Instruct is asked to judge **26 individually
provided claims** against **fictional** primary-source passages. This is a
**model-based evidence classifier**, not a claim-extraction pipeline
or a production recommendation Critic.

## Scientific question

Does a pinned small LLM recognize source support, contradiction, ambiguity
and invalid/missing citations over the same fixed authored examples on which
the deterministic typed-source rules were audited?

Crucially, agreement with **same-project author labels** is not independent
human evaluation. We distinguish it from future independent reviews and
do not claim general semantic-groundedness performance.

## Frozen experimental conditions

- Cases: `data/atomic_claim_eval/cases.json`, containing **22 typed**
  and **4 open-text** author-composed atomic claims. No case modifications
  or new test labels are made here.
- Gold: `data/atomic_claim_eval/author_labels.json`. This file is loaded
  only for **post-inference scoring**; no gold label or author rationale
  is included in any model prompt.
- Sources: the Stage 3.2 manually invented four-product, seven-document
  `data/source_evidence_demo` fixture. Citation metadata, whole primary
  product-specific source passages, effective dates, supersession links
  and whether each source has been superseded are provided.
- Generator: **Qwen/Qwen2.5-0.5B-Instruct** pinned to
  `7ae557604adf67be50417f59c2c2f167def9a775`, CPU,
  greedy decoding, one call per claim, up to 96 new tokens.
- The model receives the claim span and full reason, typed attribute/value
  **when previously annotated**, citation IDs and their linked source
  excerpts, plus all product source texts. It does not receive the
  author label or a prior Critic verdict.
- No retrieval or recommendation-generation model is called; this task
  tests the **evidence reviewer only**.
- Output contract: JSON object with exactly one `label` in
  `supported`, `contradicted`, `unsupported`, `ambiguous`, plus a
  nonempty `rationale`. An exact whole Markdown JSON fence may be stripped,
  but missing labels, arbitrary text, and invented fields are **not**
  repaired. Invalid outputs count as `needs_human_review`, never
  as successful verdicts.

### Label precedence

1. If active annotated sources materially disagree, mark **Ambiguous**.
2. If active unambiguous sources assert an opposite fact, mark
   **Contradicted**, including where obsolete citations conflict with
   newer revision documents.
3. If the claim is not supported by sources or has missing/incorrect/
   cross-product/off-field/obsolete citations, mark **Unsupported**.
4. Only when the original cited current sources substantiate the
   specific claim and all citations are relevant, mark **Supported**.

The LLM is asked to interpret natural-language passages, rather than
reading the deterministic guard's computed source finding or its gold
labels. The prompt is necessarily brief and the model is very small;
a failure is a finding, not grounds to silently increase model size.

## Benchmarks and denominators

The Stage 3.4 controls remain as already recorded:

| Control | Scope | Reference |
| --- | --- | --- |
| Naive citation-ID membership | 22 explicitly typed claims | 8/22 matched author labels |
| Deterministic typed source guard | same 22 typed claims | 22/22 matched author labels |
| Free-text deterministic guard | 4 open-text claims | 4 abstentions, **no** accuracy score |
| Real model-based Qwen critic | all 26 claims | To be reported from raw real-model run |

Scores are measured separately on all 26 examples, the 22 typed
examples, and the 4 open-text examples. For each subset we report
**valid JSON verdict coverage**, agreement with author labels **counting
invalid outputs as misses**, and optional selective agreement on valid
responses only. Confusion matrices and per-class scores are archived.
These are descriptive author-labeled synthetic metrics, **not independent
human-calibrated accuracy**.

Every model attempt logs the unmodified raw response, prompt hash,
source-context hash, model call count, prompt/generation tokens and
latency. One call per claim is enforced and audited.

## Independent human calibration (prepared, not fabricated)

The separate `recsys.evaluation.human_calibration` module generates
**two identical but independent blank review packets** (Reviewer A and
Reviewer B). Each case includes its claim, full product source passages,
source-revision flags and cited evidence snippets. It contains **no
author reference label and no model prediction**.

For genuine human calibration, two people should label all 26
cases **independently**, fill their own reviewer IDs and explanatory notes,
then upload the completed worksheets. The loader rejects duplicate
reviewer IDs, missing labels, altered original source evidence,
duplicate/missing case IDs, or incomplete rationales. It reports percent
agreement, Cohen's kappa and a disagreement list. If reviewers differ,
the system **does not invent an adjudicated label**.

A source-grounded model comparison can then be computed against
the **reviewer-consensus subset**, with its denominator and remaining
disagreements stated. A full human-calibrated accuracy value is reported
only if all cases have a consensus, following the implemented validation
contract. If adjudication is needed, it must be explicit; two reviewer
IDs in a file alone are not proof that two independent people reviewed it.

**Current status:** no completed human submissions. Human agreement,
human-calibrated model accuracy, real-world source truth and
real-model recommendation unsupported-claim rate remain **unmeasured**.

### Reproduction

```bash
python -m pip install -e ".[dev,local-llm]"
python -m pytest

python -m recsys.evaluation.human_calibration \
  --packet-dir artifacts/evaluation/stage35-blind-review

python -m recsys.evaluation.run_model_critic \
  --max-new-tokens 96 \
  --output artifacts/evaluation/stage35-real-qwen-critic.json
```

Once two **real** complete reviews exist:

```bash
python -m recsys.evaluation.human_calibration \
  --review-a path/to/completed-A.csv \
  --review-b path/to/completed-B.csv \
  --model-report artifacts/evaluation/stage35-real-qwen-critic.json \
  --output artifacts/evaluation/stage35-human-agreement.json
```

The GitHub Actions `Stage 3.5 Real Qwen Evidence Critic` workflow
runs the pinned real Qwen model and saves the complete JSON record plus
**blank** reviewer packets as artifacts.

## Limits and next development

The 26 cases were authored using the same fictional sources as the rule
logic, so deterministic guard accuracy reflects engineered contract
consistency, not generalization. No corpus of independently sourced
real product claims is used. There is **no automatic exhaustive extraction
of atomic facts from actual recommendation reasons**. Until that is
implemented and independently reviewed, this LLM classifier should
remain a separate explicit claim-review module, rather than a silent
boolean gate in the live recommendation system.
