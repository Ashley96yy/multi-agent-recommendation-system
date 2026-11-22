# Stage 3.5 — Actual Real Qwen Model-Critic Results (Negative Result)

**Date:** 2026-11-22. The pinned real Qwen CPU run **completed successfully**
and archived all 26 unmodified generations. The model **did not produce a
usable four-way claim classifier under this exact prompt and output contract**.

- [Successful real Qwen experiment #45060484582](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45060484582)
- [Original raw model output and blank independent-review packet](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45060484582/artifacts/14768979481)
- Real experiment commit: `bb9adc981202dabaf3442fd56c411d5c595b88d4`.
- Exact model: `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`.
- Model conditions: CPU, greedy, 96 maximum new tokens, one actual model call per claim.
- Python 3.11.18, PyTorch 2.6.0+cpu, Transformers 4.57.6, Pydantic 2.14.4.
- Fixed cases/catalog/source SHA-256 values, raw responses, token usage,
  latencies and prompt/context hashes are embedded in the experiment JSON.

## 1. What was evaluated

The model was shown 26 **manually authored atomic claim instances** over
**fictional** product-source passages: 22 with typed product fields and four
with unstructured claim spans. It received cited IDs, source text,
revision relationships and supersession flags, but **never** author reference
labels or their rationales in prompts.

The real model was asked to output exactly one JSON object with a label
in Supported, Contradicted, Unsupported, Ambiguous and a brief explanation.
A syntactically valid JSON response with any other label is **not**
considered a valid four-class verdict. Invalid responses are counted
as `needs_human_review` and are failures against the author labels.
There are no hidden retries, synonym repairs or label substitutions.

## 2. Measured real model performance

| Metric | Observed |
| --- | ---: |
| Frozen fictional atomic claims | **26** |
| Physical Qwen model calls | **26** |
| Raw JSON-decodable outputs | **23/26** |
| Strict four-class Schema-compliant outputs | **3/26** |
| Invalid or noncompliant outputs | **23/26** |
| Correct verdicts against author reference labels | **0/26** |
| Typed agreement with authored labels | **0/22** |
| Open-text agreement with authored labels | **0/4** |
| Valid-output coverage, all cases | **3/26 = 11.5%** |
| Valid-output coverage, typed | **2/22 = 9.1%** |
| Valid-output coverage, open text | **1/4 = 25%** |
| Prompt tokens (actual generator usage) | **13,686** |
| Generated tokens (actual generator usage) | **1,346** |
| Generation-only CPU time | **302.29 seconds** |
| Independent human labels received | **0** |
| Human-calibrated semantic accuracy | **Not measured** |

All three strict-schema valid verdicts were `ambiguous`, and none
matched their author-provided reference label. The reported
0/26 is an **abstention-inclusive, author-labeled synthetic score**.
It does not measure general-purpose real-world correctness.

### Diagnosing the failure: label-vocabulary collapse

Reviewing all 26 original model responses shows the following exact
raw label prefix frequencies:

| Literal raw label text | Occurrences |
| --- | ---: |
| `ambiguity` (not an allowed label) | **20** |
| `ambiguously` (not an allowed label) | **3** |
| `ambiguous` (allowed) | **3** |

The mismatch has **two separate components**:

- **20** fully JSON-decodable objects used an out-of-contract label
  (`ambiguity` or `ambiguously`). Thus valid JSON did **not** imply a
  valid four-class verdict.
- **3** responses were invalid JSON at the syntax level (some cited
  literal quotation marks in explanations without escaping them).
- Only **3** responses followed the complete expected Pydantic contract,
  and all three made the wrong author-label classification.

The lexical prefix frequencies above are **diagnostic only**. They
are **not** silently transformed into `ambiguous` predictions, since
that would change the declared output contract **after seeing test data**.
The raw Qwen model showed an extreme preference for ambiguity-related
wording under this single prompt. This may reflect the prompt wording,
small model capacity, output-format unreliability and/or the fictional
evidence context. The experiment does **not** isolate these factors.

A separate, no-LLM diagnostic workflow compares raw JSON syntax,
four-class contract compliance and raw label prefix counts against the
immutable original run, preserving the negative result.

## 3. Compare against existing fixed controls honestly

| Approach | Typed fictional claims | Open-text fictional claims |
| --- | --- | --- |
| Naive citation-ID-only control (Stage 3.4) | **8/22** author-label agreement | Not evaluated |
| Deterministic typed source guard (Stage 3.4) | **22/22** on authored fixture | Abstained from all 4 |
| **Real Qwen2.5-0.5B source-text Critic (this run)** | **0/22**, 2 schema-valid verdicts | **0/4**, 1 schema-valid verdict |

The deterministic rules had access to pre-structured product fields and
typed source assertions. The model saw source passages and needed to
follow four-way language labels. **These are not equally capable
semantic systems**, and the 22/22 deterministic result reflects
the engineered authored fixture, not superior real-world entailment.

**Do not claim multi-agent or LLM-Critic performance gains from this run.**

## 4. Independent human-review calibration status

The workflow generated two **blank** 26-case review packets, each
including full primary source passages, supersession status, citation
metadata and the claim span, without any author reference labels or
model predictions. The two-reviewer intake implementation can check:

- Independent reviewer IDs, complete case coverage and reasons for
  each verdict.
- Exact unchanged source evidence SHA-256/context values.
- Inter-rater percentage agreement, four-class confusion and Cohen's
  kappa.
- Reviewer-consensus model agreement only on the agreed subset;
  disagreements are not silently adjudicated.

**No actual independent reviewer submissions exist yet.** The model's
human-calibrated accuracy is therefore `null`. The files are *review
templates*, not evidence that human reviews occurred. Even two
different reviewer IDs are not proof of two genuinely independent
human participants; review provenance must be verified outside code.

## 5. Next controlled improvements

The right next experiment is **not** to hide or rewrite these outputs.
Preserve this entire source run as a fixed failure baseline, then isolate
one factor at a time using the same model, cases and source passages:

1. Constrain the output to the exact four label tokens using a controlled
   vocabulary/schema-decoding experiment, separately from prose
   explanation quality; retain invalid/abstention accounting.
2. Pre-register a short prompt-only condition with a valid JSON
   example and label definitions, then rerun all 26 cases as a
   **new independent arm**, without changing this report.
3. If these changes establish structural compliance, compare a
   larger available model or an NLI-style verifier on identical cases,
   using balanced per-class metrics rather than only raw accuracy.
4. Obtain two independent blind human reviews and adjudicate
   disagreements before calling any reported score
   *human-calibrated*.
5. Separately extract and annotate **all** atomic factual claims from
   the actual recommendation reasons. This model Critic is currently
   only an **explicit atomic-claim** evaluator, not a deployed
   free-text recommendation safety system.

**Conclusion:** The real-model engineering path and traceable evaluation
work, but **this 0.5B model/prompt is not a reliable semantic Critic**
under the specified strict four-way contract. Negative findings and
complete failed generations have been preserved.
