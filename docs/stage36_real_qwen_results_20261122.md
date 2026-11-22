# Stage 3.6 — Measured Real Qwen Prompt and Token Constraint Ablation

**Date:** 2026-11-22. **Outcome:** Structured label validity improved
substantially under a shorter output contract and explicit decision prompt,
but **semantic claim classification remained weak**. Enforcing the same
decision prompt with genuine token masking **changed zero of its 26 model
predictions** on the fixed synthetic set.

- [Actual successful 78-call Qwen CPU Actions run #45062087041](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45062087041)
- [Raw 26×3 model outputs, per-case scored labels, token and latency artifact](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45062087041/artifacts/14770326115)
- **Exact real experiment commit:** `1dd5c96ab05fbc16c43b0c4cfd14abe1ba3867ac`
- **Model:** pinned `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`.
- **Data:** unchanged 26 author-written claims (22 typed; 4 open text)
  over seven **fictional** product source documents; identical per-case
  user context. Author reference labels kept separate from model prompts.
- **Decoding:** CPU greedy, three new arms × 26 **independent physical
  generations**; no hidden retries, model switching or post-hoc corrections.

## Results

| Metric | Brief plain-label prompt | Explicit decision-order prompt | Same decision-order prompt + true prefix token mask |
| --- | ---: | ---: | ---: |
| Real Qwen calls | 26 | 26 | 26 |
| Exact canonical label outputs | **18/26** | **26/26** | **26/26** |
| Invalid outputs | **8/26** | 0 | 0 |
| Correct vs author-provided synthetic labels | **7/26** | **8/26** | **8/26** |
| Correct on 22 typed claims | 6/22 | 8/22 | 8/22 |
| Correct on 4 open-text claims | 1/4 | 0/4 | 0/4 |
| Prompt tokens | 8,642 | 11,268 | 11,268 |
| Generated tokens | 64 | 84 | 84 |
| Total actual tokens | 8,706 | 11,352 | 11,352 |
| Generation-only CPU time | 82.36s | 110.21s | 110.19s |

Across all **78 physical Qwen calls**, 31,410 total prompt+generated
tokens were consumed (8,706 + 11,352 + 11,352). Calls are genuinely
independent; no cached first draft was credited as a new generation.

This is a **different output contract** from Stage 3.5: the new arms
produce a **single lowercase label**, not the former JSON object with
an LLM-authored rationale. Therefore old Stage 3.5's 0/26 author agreement
and 3/26 JSON validity is historical motivation only, not a causal
matched fourth experimental arm.

## What improved, and what did NOT

### Prompt engineering versus output contract

The brief plain-label instruction returned these exact raw words:

- `ambiguous`: 8
- `unsupported`: 5
- `supported`: 5
- **`unspecified`: 7** (invalid)
- **`un-supported`: 1** (invalid)

The decision-order prompt explicitly defines how to handle revisions,
conflicts, missing citations and opposite-source facts; under the same
bare-label contract, it produced **26/26 canonical labels**. This
improved compliance by eight cases, but author-label agreement by only
**one** case (7/26 to 8/26). The prompt was designed after examining
Stage 3.5's failures on this same set, so these numbers are
**exploratory, not blind held-out generalization**.

### The true token constraint

The constrained implementation uses Transformers'
`prefix_allowed_tokens_fn` to restrict **next-token logits** to a
canonical four-label tokenizer trie and EOS. Its system prompt and
input evidence contexts are **identical** to the unconstrained
decision-order arm. The recorded 26 label strings are **exactly the
same** in those two arms. It therefore had **zero observed incremental
gain** in structural compliance or author-label agreement under this
prompt, on these 26 cases.

**Engineering benefit:** unlike a prompt-only instruction, the mask
prevents the model from emitting out-of-vocabulary class tokens during
this specific label-only generation. This is a genuine decoding
constraint, not text replacement after generation.

**Scope limit:** it constrains **only the label string**, not a full
JSON object, free-text reasoning or citations. A downstream Pydantic
wrapper would be an **application-created typed object**, not a
model-authored JSON rationale.

### Remaining classification failures

The improved-prompt model predicted:

| Predicted label | Count |
| --- | ---: |
| Contradicted | **16** |
| Supported | 5 |
| Unsupported | 5 |
| Ambiguous | **0** |

On the **six author-labeled ambiguous** examples, it predicted
contradicted (3), supported (1), or unsupported (2); **none** were
correct. It also mislabeled straightforward supported price/attribute
claims as contradicted (e.g. `T03`, `T04`, `T05`,
`T06`). The masked arm repeated **all 26** of those decisions.

Thus **valid formatting does not imply factual source understanding**.
A source-aware and fail-closed production system should not treat this
8/26 exploratory classifier as a reliable automatic semantic Critic.

## Cautions about causal interpretation

- The pair of **decision-order arms** has identical prompt and context;
  the only changed mechanism is the token whitelist. In this run
  there was **no label-level difference**, and CPU token usage was
  the same. Small timing differences are not evidence of speed benefit.
- The **brief versus decision-order arms** share the label-only
  output contract and sources, isolating the instruction change in
  this implementation; the decisions remain strongly biased.
- Arms were executed in a fixed within-case order; CPU timing may be
  affected by runtime/caching and is descriptive, not a randomized
  performance benchmark.
- Dataset and author labels are both constructed for this project.
  **No independent human review**, real-world product evidence,
  exhaustive extraction of real model-generated reasons, or
  calibrated semantic groundedness metric is available.
- All real-world and human-calibrated metrics remain **null**, not zero.

## Reproduction and next priority

```bash
python -m pip install -e ".[dev,local-llm]"
python -m pytest
python -m recsys.evaluation.run_label_ablation \
  --output artifacts/evaluation/stage36-real-qwen-label-ablation.json
```

Raw outputs, per-arm confusion matrices, SHA-256 prompts/context,
per-case outcomes and actual token timings are in the original
GitHub Actions Artifact.

**Recommendation:** Preserve this negative semantic finding and do
not keep adding prompt variants to the same tiny synthetic benchmark.
Stage 4 should focus on a usable **end-to-end demo** with the reliable
deterministic typed-source guard as the primary verification gate.
A label-only Qwen model Critic may be displayed as **experimental
advisory information** with clear uncertainty, but should not override
document conflicts or invent unsupported product claims. For valid
human-calibrated accuracy, obtain independent blind review before
making general semantic claims.
