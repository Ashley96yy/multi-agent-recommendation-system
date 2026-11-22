# Evidence-Grounded Multi-Agent Recommendation System

Independent **Applied AI / AI Engineering** portfolio project focused on reliable,
source-grounded product recommendation and a controlled comparison of
**Single-Agent RAG** against a **three-agent verification workflow**.

> **Current state:** Retrieval, real pinned-Qwen recommendation generation,
> bounded three-role orchestration and deterministic source checks have
> been measured on fictional cases. A **separate real Qwen model-based
> atomic-claim Critic** was evaluated in Stage 3.5 and **failed** its strict
> four-class contract: only 3/26 outputs were schema-valid, and **0/26**
> matched the author-written case labels. No improvement is claimed.
> Real recommendation-prose groundedness and independent human-calibrated
> accuracy **have not been measured**. This is not three LLM agents
> collaborating end to end.

## Research question

**Does multi-agent verification reduce unsupported product claims and improve
constraint satisfaction enough to justify its additional latency and tokens?**

The project will eventually compare the same pinned model, retrieval evidence,
and fixed scenarios using three controlled configurations:

1. Single-Agent RAG (measured on frozen 15+10 synthetic cases).
2. Multi-role workflow with Critic, no revision (real-Qwen paired experiment measured).
3. Same workflow with one bounded revision (real-Qwen paired experiment measured).

## Implemented components

| Component | Status | Notes |
| --- | --- | --- |
| Strict data contracts | Complete | Pydantic; synthetic products, preferences, linked evidence |
| Metadata-safe catalog filtering | Complete | Exclusions, category, budget and required attributes |
| Vector retrieval | Complete | Offline lexical baseline + optional pinned MiniLM |
| Retrieval evaluation | Complete | 15 fictional cases, 3 negative cases, Recall/MRR/NDCG |
| One-pass Single-Agent RAG | Complete | Retrieved evidence, JSON output, strict candidate/citation ID checks |
| Local Qwen model adapter | Smoke verified | Pinned public 0.5B model; 2 actual calls and 2/2 structural positives in a tiny synthetic run |
| Frozen single-agent evaluation | Measured | 15 normal + 10 adversarial synthetic cases; strict 2/19 positive successes, 13 format/schema failures |
| Format-only replay ablation | Measured | Same frozen Qwen outputs: strict 2/19 vs normalized 12/19 positive successes; no new model calls |
| Synthetic attribute provenance | Measured | 29/29 metadata-derived field references versus 0/29 before; NOT independent evidence |
| Evidence review export | Implemented | Source-backed CSV review queue, with human claim labels deliberately blank |
| Typed atomic claim evaluation | Measured on fictional authored labels | 26 cases: 22 typed supported/contradicted/unsupported/ambiguous, 4 open-text human-review abstentions; 22/22 typed fixture agreement, not real semantic accuracy |
| Model-based atomic claim Critic (Stage 3.5) | Real pinned-Qwen negative result measured | 26 real calls; 3/26 valid four-class outputs, 0/26 author-label matches; extremely ambiguity-biased, no real groundedness claim |
| Independent human calibration | Review workflow prepared, no actual reviews yet | Separate blank Reviewer A/B packets, source hashes, complete reviewer labeling validation, Cohen's kappa and blind independent review protocol; no human-calibrated score claimed |
| Stage 3.2 opt-in output and evidence preflight | Tested | Strict remains default; safe JSON normalization, field-level citation preflight, zero-call insufficient-evidence path |
| Source document contradiction audit | Tested with fictional documents | 7 invented passages and 12 source claims; 1 active conflict, 1 stale revision, 1 missing field; NOT a general semantic verifier |
| Stage 3.3 runtime source guard | Implemented and unit-tested | Opt-in active-source conflict, supersession, missing claim, stale metadata and explicit typed claim checks; not a language entailment model |
| Recorded Stage 3.2 output replay | Measured offline | Original 19 real Qwen first drafts: strict 3 approved / 0 label successes; normalized 9 approved / 4 label successes; zero new Qwen calls |
| Three-role orchestration and critic / revision | Real Qwen integration measured | Shared first draft, 19 initial Qwen calls + 16 real revisions; neither arm achieved a strict positive success on 19 synthetic positives |
| Full model comparison and deployment | Not implemented | Await broader verified evaluation |

## Reproduce lightweight tests

Requires Python 3.11+.

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m recsys.evaluation.run_retrieval --backend hashing \
  --output artifacts/evaluation/hashing.json
```

## Reproduce actual semantic retrieval

```bash
python -m pip install -e ".[semantic]"
python -m recsys.evaluation.run_retrieval --backend minilm \
  --output artifacts/evaluation/minilm.json
```

The pinned MiniLM evaluation completed successfully on a tiny synthetic suite:
**Recall@5 = 1.0000**, versus **0.9333** for the lexical baseline. This is
**not a real-world effectiveness claim**. [Measured results](docs/retrieval_results_20261122.md).

## Run opt-in local Qwen smoke

```bash
python -m pip install -e ".[semantic,local-llm]"
python -m recsys.evaluation.run_single_agent \
  --retrieval-backend minilm \
  --output artifacts/evaluation/qwen-single-agent.json
```

This downloads public model weights. A separate GitHub Actions
`Qwen Single-Agent Smoke` workflow runs the same code on CPU, and captures
per-case outcomes, tokens, latency, pinned model versions, and failure reasons.
Real model execution was independently checked in [the recorded Qwen CPU smoke](docs/real_single_agent_qwen_results_20261122.md); the unit tests alone would not establish that.

## Evaluate the 15 + 10 single-agent baseline

The fixed-case experiment separates expected-product retrieval, structurally
accepted recommendations, hard no-match filtering, citation **field coverage**,
red-team literal phrase leakage, model failures, latency and token usage.
**Factual claim support is not automatically graded**; unresolved semantic
groundedness remains `null` until independently reviewed.

```bash
python -m pip install -e ".[dev,semantic,local-llm]"
python -m pytest
python -m recsys.evaluation.run_baseline_eval \
  --retrieval-backend minilm --top-k 3 \
  --output artifacts/evaluation/single-agent-baseline-25.json
python -m recsys.evaluation.review_queue \
  --report artifacts/evaluation/single-agent-baseline-25.json \
  --output artifacts/evaluation/manual-review-queue.csv
```

See [the frozen evaluation protocol](docs/baseline_evaluation_protocol.md)
for scoring definitions, [the measured strict 25-case baseline](docs/single_agent_baseline_results_20261122.md),
and [the format-only reliability ablation](docs/baseline_reliability_ablation_20261122.md).
The measured normalization improves structural acceptance but **does not independently
verify factual recommendation claims or establish production accuracy**.

## Bounded Multi-Agent Foundation (Stage 3.0)

The new `recsys.multiagent` module provides:
- `PreferenceRetrievalAgent` — validated preferences and metadata-safe retrieval.
- `RecommendationAgent` — one candidate generation per attempt using the
  original Single-Agent baseline context; structured critic feedback only on revision.
- `CriticVerificationAgent` — deterministic hard constraints, citation
  membership, required-attribute field references and selected forbidden-phrase
  checks. **Not** an independent language-model fact checker.
- `MultiAgentOrchestrator(max_revisions=0|1)` — no candidate → no calls,
  otherwise one draft and at most one further attempt; rejected proposals
  never become user-visible final answers.

Unit tests use fixed JSON generators; the separate Stage 3.1 GitHub Actions
workflow ran **real pinned Qwen** on the original frozen corpus.
The paired study shared each first draft across the two arms, and a
single revision was **not** beneficial under this model, schema and
evidence configuration: 0/19 positive strict successes for both arms,
3/19 structurally approved proposals each, and 16 extra Qwen calls with
one revision. Missing attribute-field evidence and invalid model JSON
were the main blocks. No real semantic claim-entailment metric was computed.
Read the [architecture](docs/multiagent_architecture.md), the
[paired experiment protocol](docs/real_multiagent_qwen_protocol.md), and
the [actual measured results](docs/real_multiagent_qwen_results_20261122.md).

To reproduce the real CPU evaluation:

```bash
python -m pip install -e ".[dev,semantic,local-llm]"
python -m recsys.evaluation.run_multiagent_eval \
  --catalog data/retrieval_eval/catalog.json \
  --suite data/baseline_eval/suite.json \
  --output artifacts/evaluation/multiagent-qwen-paired-25.json
```

## Stage 3.2 — Separate output reliability and source evidence

Two **opt-in** orchestrator policies preserve the historical strict model
runs unchanged:

- `format_policy="safe_json_normalization"` only removes whole-response
  Markdown JSON fences or adds an unambiguous missing boolean. It never
  edits product IDs, citations, written product reasons, or constraints.
- `evidence_preflight=True` filters candidates without the required
  attribute-field citations; when no retrieved candidate has the needed
  source records it exits with `insufficient_evidence` and zero model calls.

The independently authored **fictional** source-document fixture and
deterministic audit separately recognize contradictory active passages,
superseded source declarations, stale product metadata and missing
field support. It does not verify actual brands or natural-language
entailment.

A fresh **offline replay of the frozen 19 actual Qwen first drafts** measured
3 vs 9 Critic approvals and 0 vs 4 strict positive label successes (original
strict JSON vs safe normalization). Nine earlier requests potentially lacked
the source fields required by preflight. These are **post-hoc and
hypothetical**, respectively; no new Qwen calls or real savings were measured.
See [protocol](docs/stage32_structured_output_and_evidence.md) and
[recorded results](docs/stage32_recorded_results_20261122.md).

Reproduce the source audit (no model weights):

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m recsys.evaluation.source_audit \
  --catalog data/source_evidence_demo/catalog.json \
  --sources data/source_evidence_demo/source_documents.json \
  --output artifacts/evaluation/stage32-source-audit.json
```

## Stage 3.3 — Live reliability and conflict-aware checks

Stage 3.3 adds an **optional** `TypedSourceGuard` to the existing
deterministic Critic and field-coverage preflight. It validates typed
declarations in the separately authored **fictional** source documents,
detects contradictory active evidence, superseded citations, outdated
product metadata and missing field declarations, and can prevent a Qwen
call when required evidence cannot be trusted.

A **completed real pinned-Qwen CPU experiment** compared the unchanged
frozen 25 synthetic cases with a **2×2 first-pass policy matrix**:
strict versus safe JSON normalization, and field preflight off versus on.
The recorded positive successes were **0/19 (strict)** and **4/19
(normalized)** under both preflight settings. Source preflight reduced
the arm's logical generation calls **from 19 to 10** by rejecting nine
cases with unavailable field evidence; it did **not** add successes.
Identical first prompts shared actual model outputs: the full matrix
made 19 **physical** Qwen calls, plus one new fictional-source smoke
call. No revision loop was included.

The guard only establishes consistency of typed source assertions and
citation provenance, **not** whether arbitrary natural-language reasons
are entailed by documents. See [the Stage 3.3 experiment protocol](docs/stage33_live_reliability_protocol.md)
and [measured real Qwen results](docs/stage33_real_qwen_results_20261122.md).

## Stage 3.4 — Atomic claim evidence evaluation

To avoid conflating valid citation IDs with factual entailment, this
stage provides a **separately frozen author-labeled** atomic-claim
benchmark over 7 **fictional** source documents. For 22 already
structured product-field assertions, the deterministic source guard
classifies evidence as Supported, Contradicted, Unsupported or
Ambiguous; another four free-text claims **abstain for blind human
review**. Author-label agreement on the engineered typed fixture was
22/22, versus 8/22 for a naive citation-ID-only control. These figures
are **not model accuracy**: no real Qwen output was newly labeled,
no semantic extraction was evaluated, and no independent annotator has
reviewed the cases. A 26-row blind-review CSV was generated with cited
excerpts but without the author's gold labels.

Separately, the original **real Qwen** Stage 3.3 output artifact was read
without any new generation. A second blinded review CSV now contains **15
actual model recommendation reasons from 9 previously approved cases**,
plus their source excerpts. These are complete reasons, **not 15 labeled
atomic claims**: all per-claim extraction/reviewer fields remain blank,
and the model's factual groundedness is **still unmeasured**.

See the [Stage 3.4 evaluation protocol and real offline results](
docs/stage34_atomic_claim_grounding_20261122.md).

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m recsys.evaluation.atomic_claims \
  --output artifacts/evaluation/stage34-atomic-claims.json \
  --blind-review-csv artifacts/evaluation/stage34-blind-review.csv
```

## Stage 3.5 — Separate model-based claim Critic

A real pinned local **Qwen2.5-0.5B-Instruct** is asked to classify the
**26 fixed synthetic atomic claims**, using complete fictional primary
source passages and citation metadata. The labels are Supported,
Contradicted, Unsupported and Ambiguous; invalid model answers are
recorded as requiring review rather than silently being fixed.
Author-written gold labels are used **only after inference**.
The **actual CPU experiment ran all 26 calls** and yielded 3 valid
four-class outputs and **0 author-label agreements**; the model tended
to produce `ambiguity` or `ambiguously` rather than the required
`ambiguous`. A separate no-LLM audit verified **23 JSON-decodable
responses, 20 of which used invalid class vocabulary**, plus three
JSON-syntax errors. Nothing was silently repaired.
This model is **not** yet integrated as an automatic semantic gate
for all recommendation prose, and claim segmentation is not automatic.

Two blank, independent human-review packets are also produced. The
reviewing workflow checks completeness, source hashes, separate
reviewer IDs and inter-reviewer agreement, but **no human ratings
have been received**. Human-calibrated accuracy and actual real-Qwen
recommendation groundedness therefore remain **unmeasured**.

See the [Stage 3.5 protocol](docs/stage35_model_critic_protocol.md)
and [measured negative-result report](docs/stage35_real_qwen_critic_results_20261122.md).

```bash
python -m pip install -e ".[dev,local-llm]"
python -m pytest
python -m recsys.evaluation.run_model_critic \
  --output artifacts/evaluation/stage35-real-qwen-critic.json
python -m recsys.evaluation.human_calibration \
  --packet-dir artifacts/evaluation/stage35-blind-review
```

## Design priorities

- **Evidence and provenance:** catalog SHA-256, explicit source references,
  and deterministic input/output contracts.
- **Fail closed:** missing metadata and hallucinated IDs are rejected rather
  than silently accepted.
- **Controlled ablations:** retrieval, generator, test set, and grading must
  stay matched across architectures.
- **Reproducibility:** recorded CI logs and raw experiment artifacts;
  no invented model metrics.

## Documentation

- [Catalog retrieval](docs/retrieval.md)
- [Semantic retrieval protocol](docs/semantic_retrieval_evaluation.md)
- [Recorded MiniLM results](docs/retrieval_results_20261122.md)
- [Single-agent baseline](docs/single_agent_baseline.md)
- [Real Qwen smoke protocol](docs/real_single_agent_qwen_smoke.md)
- [Recorded real Qwen results](docs/real_single_agent_qwen_results_20261122.md)
- [Single-agent 15+10 evaluation protocol](docs/baseline_evaluation_protocol.md)
- [Measured 25-case Single-Agent results](docs/single_agent_baseline_results_20261122.md)
- [Stage 2.3 format-only replay and synthetic evidence coverage results](docs/baseline_reliability_ablation_20261122.md)
- [Stage 3.0 bounded three-role architecture](docs/multiagent_architecture.md)
- [Real paired Qwen Multi-Agent experiment protocol](docs/real_multiagent_qwen_protocol.md)
- [Measured Stage 3.1 real Qwen paired results](docs/real_multiagent_qwen_results_20261122.md)
- [Stage 3.2 output/evidence protocol](docs/stage32_structured_output_and_evidence.md)
- [Recorded Stage 3.2 replay and fictional source-audit results](docs/stage32_recorded_results_20261122.md)
- [Stage 3.3 live reliability and conflict-aware source guard protocol](docs/stage33_live_reliability_protocol.md)
- [Measured Stage 3.3 real Qwen reliability and source-guard results](docs/stage33_real_qwen_results_20261122.md)
- [Stage 3.4 authored atomic-claim audit and blind human-review protocol](docs/stage34_atomic_claim_grounding_20261122.md)
- [Stage 3.5 real Qwen atomic Critic and independent human-review protocol](docs/stage35_model_critic_protocol.md)
- [Measured Stage 3.5 real Qwen 0/26 failure analysis](docs/stage35_real_qwen_critic_results_20261122.md)
- [Provenance and reuse](docs/PROVENANCE.md)

## Provenance

Inspired by SJSU DATA 298A Team 5 Topic 22's evidence-grounded recommender
research direction. The implementation in this repository is independent:
no team code, restricted dataset, or pretrained team checkpoints were copied.
All included catalog entries are fictional and for development/evaluation.
