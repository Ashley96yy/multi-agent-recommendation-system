# Stage 3.3 — Real Qwen Reliability Matrix and Conflict-Aware Source Guard

## Goals / what is and is not a model experiment

Stage 3.1 found that a real Qwen2.5-0.5B one-revision workflow did not
improve strict synthetic positive success and spent extra tokens on malformed
responses and unsupported source requirements. Stage 3.2 introduced an
**opt-in** JSON normalization policy and preflight, first measured using
*offline replays*. Stage 3.3 runs a **new real Qwen** first-pass experiment
using those policies and integrates typed-source conflict awareness into an
opt-in deterministic Critic.

Historical Stages 2.1–3.2 code, source files, model snapshots, baseline
labels and measured results are not altered. No multi-agent superiority
claim or semantic-groundedness metric is inferred from structural success.

## A. Exact 2×2 live reliability matrix

On the unchanged **15 normal + 10 red-team** author-labeled synthetic suite,
with original 15-product catalog, pinned MiniLM and Qwen weights:

| Factor | Level 1 | Level 2 |
| --- | --- | --- |
| Format policy | `strict` | `safe_json_normalization` |
| Attribute field evidence preflight | disabled | enabled |
| Revision budget | 0 | 0 |
| Critic | unchanged deterministic metadata + reference checks | same |

All arms use MiniLM top_k=3; Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, 240 maximum output
tokens, CPU greedy decoding; MiniLM revision
`826711e54e001c83835913827a843d8dd0a1def9`.

The **identical first prompt and retrieved context** within a case has
one physically executed real Qwen generation, reused across all logical
arms with exactly matching prompt text/context bytes. No generated output is
rewritten except the two deterministic, safe JSON metadata operations.
The recorded `physical_model_invocations` count is **not** the sum
of per-arm `logical_generator_calls`: that would double-count shared calls.

For the original corpus, preflight should bypass requests that have
metadata-eligible retrieved products but lack explicitly linked required
attribute field citations. No-match requests are separately handled by
hard filters. Positive scenarios skipped for missing source fields count
as **unsuccessful recommendations**, not as corrected predictions.

**Isolation:** The format-only factor never changes prompts; evidence
preflight can modify candidate context, and the runner only reuses a
physical generation if that context and instructions are identical.
This stage isolates first-pass reliability factors without confounding
them with the one-revision loop.

## B. Runtime source-aware Critic

A new opt-in `TypedSourceGuard` uses the **separate FICTIONAL 4-product,
7-document, 12-typed-claim fixture** from Stage 3.2. It re-runs the full
source index audit, binds to exact product catalog SHA-256, and validates
whether a cited document is still active; whether active source
declarations disagree; whether an active revision contradicts catalog
metadata; and whether a required field has any active claim.

The deterministic Critic can now flag these codes on supported typed
source fields:

- `source_conflict`: contradictory active typed source claims.
- `source_stale_metadata`: newer active declaration contradicts metadata.
- `source_claim_missing`: no active annotated source claim.
- `superseded_evidence`: citation links to a superseded document.

For **explicit typed atomic claims**, `verify_typed_claim(...)` verifies
the claimed boolean or listed price matches both metadata and the *active*
annotated authored source, and that the cited ID belongs to an active
matching record. It rejects unsupported, contradicted or stale typed
claims. **No free-text factual claim extraction** is implemented.

When the optional field preflight is enabled, a conflicting/stale/missing
required attribute causes the generator to be bypassed rather than asked
to fabricate a repair. The source guard is disabled by default on the
historical catalog. There is **no truth guarantee for arbitrary
recommendation reasons**: the source guard verifies annotated fields
and provenance, not open-text semantic entailment.

## C. Separate real-Qwen fictional evidence smoke

Four hand-designed inputs use the *new* fictional source catalog:

1. Consistent face cream: one actual Qwen generation attempted; acceptance
   depends on the real output and is not guaranteed.
2. Conflicting barrier balm fragrance: preflight abstains without Qwen.
3. Superseded mineral sunscreen assertion: preflight abstains without Qwen.
4. Fragrance-free metadata without any source declaration: preflight
   abstains without Qwen.

This is a small engineering safety-control check on **invented** sources,
not an evaluation on real consumer products.

## Metrics, reproducibility and limitations

The real run reports per case and arm: strict author-label success,
no-match successes, approved/rejected/insufficient evidence counts, logical
LLM calls, shared raw output and prompt/context hash, Critic issue counts
and format actions. Physical generation records separately contain
per-call tokens and generation durations.

If a product query has no eligible data, it receives zero model calls.
An eligible product skipped due to evidence preflight does not become a
successful positive-case recommendation.

Run with Python 3.11+ and public model downloads:

```bash
python -m pip install -e ".[dev,semantic,local-llm]"
python -m pytest
python -m recsys.evaluation.run_live_reliability \
  --output artifacts/evaluation/stage33-real-qwen-matrix.json
```

The dedicated `Stage 3.3 Real Qwen Reliability and Source Guard`
GitHub Actions workflow runs the same experiment and archives the
**unmodified real Qwen outputs** and evaluation decisions.

All evidence sources are written fictionally for testing. The source audit
only checks typed annotations and verbatim excerpts; it does not adjudicate
the truth of arbitrary natural-language sentences. Independent
claim-level human review or a calibrated semantic verifier is necessary
before reporting factual unsupported-claim rate or true groundedness.
These remain **null**.
