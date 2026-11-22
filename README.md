# Evidence-Grounded Multi-Agent Recommendation System

Independent **Applied AI / AI Engineering** portfolio project focused on reliable,
source-grounded product recommendation and a controlled comparison of
**Single-Agent RAG** against a **three-agent verification workflow**.

> **Current state:** Retrieval and a real Qwen Single-Agent baseline are
> implemented. Stage 3.1 has now **measured** a genuine Qwen-backed
> three-role workflow with a **deterministic** Critic and at most one
> revision on 15 normal + 10 red-team **synthetic** requests.
> **No positive quality gain was observed from the revision**. This
> is NOT three LLM agents or independent semantic fact verification.
> All actual metrics and limitations are recorded in the experiment report.

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
| Evidence entailment verification | Not implemented | Citation IDs alone do not prove claim support |
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
- [Provenance and reuse](docs/PROVENANCE.md)

## Provenance

Inspired by SJSU DATA 298A Team 5 Topic 22's evidence-grounded recommender
research direction. The implementation in this repository is independent:
no team code, restricted dataset, or pretrained team checkpoints were copied.
All included catalog entries are fictional and for development/evaluation.
