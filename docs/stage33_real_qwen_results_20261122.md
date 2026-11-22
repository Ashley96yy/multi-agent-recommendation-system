# Stage 3.3 — Verified Live Qwen Reliability and Conflict-Guard Results

**Date:** 2026-11-22. This is a **new real CPU Qwen inference run**,
not merely a post-hoc replay of the Stage 3.1 experiment.

- [Successful actual Qwen GitHub Actions run #45056291366](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45056291366)
- [Full original real-Qwen JSON output artifact](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45056291366/artifacts/14766939207)
- **Exact real-model experimental commit:** `a947c353bf19aa04a55440762f17ae7e5c6dd00f`
- Model: `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`
- MiniLM: `sentence-transformers/all-MiniLM-L6-v2@826711e54e001c83835913827a843d8dd0a1def9`
- `top_k=3`; greedy, maximum 240 generated tokens; GitHub Actions CPU.
- Original 15 fictional products and the frozen 15 ordinary + 10
  adversarial scenarios; **no new scenarios or labels slipped into the
  historical benchmark**.

## Recorded first-pass 2×2 reliability matrix

| Metric | No preflight + strict | No preflight + normalized | Preflight + strict | Preflight + normalized |
| --- | ---: | ---: | ---: | ---: |
| Synthetic cases | 25 | 25 | 25 | 25 |
| Author-label positive successes / 19 | **0** | **4** | **0** | **4** |
| Generated proposals Critic-approved / 19 original positives | **3** | **9** | **3** | **9** |
| Correct hard-filter no-match cases | 6/6 | 6/6 | 6/6 | 6/6 |
| Zero-call insufficient-evidence cases | 0 | 0 | **9** | **9** |
| Logical generator calls for this arm | 19 | 19 | **10** | **10** |

The **19 real physical Qwen first-draft generations** are reused across
logical arms only when the case's prompt and source context are **byte-for-byte
identical**. The format factor is *deterministic post-processing* and never
changes the prompt; preflight skips 9 original field-unsupported requests.
For the remaining 10, the source candidate context is unchanged.

A **distinct real Qwen call** was also made for the separate fictional source
smoke, resulting in **20 total physical Qwen invocations** across this workflow.
Do **not** add the four logical-arm call counts (19 + 19 + 10 + 10);
that would dramatically overstate actual generation cost. Full raw outputs,
per-call generated/prompt tokens, generation durations, input hashes,
critic findings and per-case statuses are in the archived JSON artifact.

This new live experiment reproduced the earlier qualitative conclusion:
**format-only normalization restores structurally eligible outputs**, whereas
requiring citations for absent catalog fields causes a deliberate
fail-closed no-recommendation outcome. The normalized 4/19 is author-labeled
strict product selection, **not factual groundedness**.

### Real call reduction interpretation

Evidence preflight changes the *logical call requirement* for a stand-alone
no-revision system from **19 to 10** on these exact synthetic inputs.
Nine positive cases return `insufficient_evidence` and are counted as
**failed to recommend**, not as successful positive predictions.

These 9 calls were **not measured as nine fewer physical model calls in
this full 2×2 combined job**, because the non-preflight experimental arms
still generated their proposals once and then shared them with other arms.
The counts describe a controlled policy difference. No monetary API
savings are claimed.

## Real Qwen fictional source-guard smoke

A separate four-case run used the **authored, fictional** source fixture,
not the original 15-product benchmark:

| Source situation | Outcome | Physical Qwen calls |
| --- | --- | ---: |
| Consistent active source + explicitly supported field | Approved by structural/typed-source Critic | **1** |
| Contradictory active fragrance claims | `insufficient_evidence` | **0** |
| Latest revision contradicts mineral-sunscreen metadata | `insufficient_evidence` | **0** |
| No fragrance declaration despite catalog metadata | `insufficient_evidence` | **0** |

This establishes that the runtime **conflict-aware preflight** can block
untrustworthy *annotated* source fields before real model generation. The
approved consistent example still needs independent human assessment of
any free-form generated reason. The smoke is one benign successful
case, not a statistically generalizable result.

## Verifier scope and risks

The new optional `TypedSourceGuard` re-audits seven **fictional** source
documents with twelve manually typed field claims, verifies their
verbatim excerpt links and supersession history, and binds its findings
to the exact catalog SHA-256. In unit tests it identifies:

- Active source contradictions.
- Citation to an obsolete, superseded document.
- Current metadata contradicted by a newer document.
- Missing typed source claim even when catalog metadata asserts the field.
- Mixed valid and fabricated/cross-product citations in one typed claim.

The runtime Critic now knows those **specific typed-source provenance
failures**. It does **not** extract every factual statement from a
free-text `reason`, infer truth or automatically judge whether arbitrary
prose is entailed by citations. The selected prohibited-phrase heuristic
is likewise not a semantic verifier. The fixture passages are invented,
not independent real-world evidence of ingredients or medical effectiveness.

**Unmeasured:** semantic citation groundedness, rate of unsupported atomic
claims, real-user recommendation relevance, production safety and
real-dollar API cost. They remain `null`, not 0% or 100%.

## Next step

For a genuine semantic-verification claim, the next phase must introduce
an explicit atomic-claim extraction contract and independent adjudication
against primary-source excerpts, covering ambiguity, conflict, superseded
sources and unsupported medical assertions. Evaluate claim labels separately
from strict schema/ID checks, with a common pinned model and corpus for
single-agent, deterministic critic and model-based critic conditions.
Do not attribute a format-only gain to multi-agent reasoning.

**Conclusion:** Stage 3.3 verified real Qwen integration and repeatable,
auditable policy differences, while maintaining a fail-closed pathway for
untrustworthy typed evidence. It is a useful engineering milestone—not
proof of broad semantic groundedness.
