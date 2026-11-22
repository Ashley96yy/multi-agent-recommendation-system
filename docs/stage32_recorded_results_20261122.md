# Stage 3.2 — Recorded Output-Policy Replay and Fictional Source Audit

**Verified:** 2026-11-22

- [Successful offline GitHub Actions run #45055644771](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45055644771)
- [Raw experiment results artifact](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45055644771/artifacts/14766978813)
- Original **real Qwen** generation source: [Stage 3.1 run #45053525292](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45053525292), 19 original first drafts + 6 deterministic no-match cases.
- Exact frozen labels and original catalog were reused; **zero new Qwen model invocations**.

## Recorded format-only first-draft replay

| Metric | Strict format (original) | Safe JSON normalization (opt-in) |
| --- | ---: | ---: |
| Raw real Qwen initial generations analyzed | 19 | 19 |
| Critic-approved generated proposals | **3/19** | **9/19** |
| Frozen-label strict positive successes | **0/19** | **4/19** |
| No-eligible-product requests skipped | 6/6 | 6/6 |
| Missing `no_suitable_match=false` fields inserted | 0 | 13 |
| Complete Markdown JSON fences removed | 0 | 4 |
| Additional Qwen calls | **0** | **0** |

- Strict replay exactly reproduced the earlier real-Qwen Critic approvals
  (3) and strict positive successes (0) before calculating the new scores.
- After format normalization, the Critic still rejected other proposals.
  Its aggregated issue counts included **15 missing attribute-field citation
  issues** and **2 prohibited-claim phrase issues**. Some claims that
  previously failed schema parsing became visible to the Critic; JSON
  validity and provenance membership are not claim truth.
- The policies were run on **identical frozen model strings**, not separate
  live model calls. These outcomes measure **post-hoc recoverability**, not
  the latency, token cost or output distribution of a reconfigured live Qwen
  workflow.
- A successful Critic approval is **not** human adjudication or independent
  semantic support for every recommendation sentence.

## Potential deterministic evidence-preflight exclusions

Under the **original** retrieved candidate IDs, nine of the nineteen
eligible requests had **no candidate** with explicit citation records
covering *all required positive attribute fields*.

A new opt-in `evidence_preflight=True` can reject that class of request
without issuing a generator call and restrict candidate context to
evidence-field-complete products when any are available.

**9/19 is a hypothetical preflight count, not nine measured Qwen calls saved.**
Actual model outputs would change if candidate sets or prompt context
changed. This stage did not rerun Qwen with preflight enabled or claim a
measured runtime/monetary saving.

## New fictional source fixture audit

Separate to the frozen original catalog, Stage 3.2 authored seven fictional
source passages with twelve independently authored typed source claims
across four fictional products. The auditor compared indexed evidence to
the literal source passages and checked typed contradictions,
supersession, and missing support.

| Field-audit outcome | Count |
| --- | ---: |
| Consistent | 4 |
| Conflicting active sources | 1 |
| Metadata stale after newer revision | 1 |
| Required attribute lacking any source claim | 1 |

These seven field outcomes cover all fictional boolean attribute fields
in this small fixture. Cases include contradictory fragrance declarations,
a superseded mineral-sunscreen claim, and a missing ingredient label.

**Scope:** The passages were written specifically for testing; they are
**not external product documentation or independent evidence of real-world
claims**. The source auditor relies on annotated typed claims and verbatim
substring checks; it is **not a natural-language entailment model**.
The runtime field preflight checks *citation availability*, not these
conflict/staleness findings. Consequently, conflict-aware runtime
verification remains a separate future task.

## Next controlled study

To establish live impacts, run the unchanged pinned Qwen under a matched
matrix of strict vs safe normalization and preflight on/off with the same
snapshot, seeds and labels. Retain all errors and count physical generator
calls. Separately adjudicate atomic factual claims against the original
source excerpts; until then, `semantic_citation_groundedness` and
`unsupported_claim_rate` are **not measured**.

These post-hoc outcomes must never be presented as production accuracy,
a successful LLM Critic, independent medical advice validation, or
an observed cost reduction.
