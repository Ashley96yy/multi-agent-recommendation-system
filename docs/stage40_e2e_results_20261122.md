# Stage 4.0 — Verified Interactive Demo E2E Results

**Date:** 2026-11-22

- [Successful real Gradio + Qwen E2E workflow #45063616834](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45063616834)
- [Immutable full JSON trace Artifact, including unmodified real Qwen output](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45063616834/artifacts/14770138788)
- **Exact tested code commit:** `d311cd01ae3cc7880b2fb36b7352b1ffc7ec3510`
- **Full suite:** `179 passed` with optional Gradio/local Qwen dependencies installed; Gradio `Blocks` successfully constructed **52 components and 3 bound event callbacks**, without launching a public server.

## What was measured

The project now has two separate, explicitly identified modes:

1. **Scripted walkthrough:** prewritten fictional JSON responses used as
   a test double, exercising the real retrieval/orchestration/typed source
   audit code. All six are deterministic and **not model inferences**.
2. **Real Qwen:** a pinned public `Qwen/Qwen2.5-0.5B-Instruct`
   with CPU greedy decoding invoked only when an eligible request
   survives metadata filtering and source evidence preflight. The
   dedicated CI smoke issued exactly **one real Qwen inference** on one
   fictional active-source recommendation; no generated response was
   silently repaired beyond the documented safe JSON-format policy.

| Walkthrough scenario | Observed result | Scripted proposal calls | What is verified |
| --- | --- | ---: | --- |
| Supported `SYN-E01` | `approved` | 1 | Correct candidate, linked active fragrance-free excerpt |
| Bounded revision | `approved` after second attempt | 2 | First medical phrase rejected, only revised draft displayed as final |
| Active source conflict | `insufficient_evidence` | **0** | Conflicting product specification and fictional QC record |
| Superseded source | `insufficient_evidence` | **0** | Current revision disagrees with old mineral metadata |
| Missing attribute source | `insufficient_evidence` | **0** | Required fragrance statement unavailable |
| Budget hard-filter | `no_candidates` | **0** | $5 price cap excludes all relevant fictional products |

**All six** fixed scripted cases passed the dedicated E2E assertions.
The first two exercise full recommendation/critic flow; the other
four illustrate **fail-closed zero-call control**, not pretend Qwen
successes.

### Real-model smoke

| Metric | Observed |
| --- | --- |
| Real model | `Qwen/Qwen2.5-0.5B-Instruct`, pinned public revision `7ae557604adf67be50417f59c2c2f167def9a775` |
| Physical real Qwen invocations | **1** |
| Logical proposal attempts | **1** |
| Final deterministic Critic verdict | `approved` |
| Prompt / completion tokens | Exact real counts preserved in JSON trace |
| New independent semantic annotations | **0** |
| Claim-groundedness quality on real-world products | **Not measured** |

The actual generation and exact retrieved source IDs, excerpts,
Pydantic schema actions, proposal timestamps and Critic issue codes
remain available in the **unmodified archived JSON**. The approved
sample demonstrates a **structurally valid positive control**, not
that Qwen is reliably correct or that its free-form reason is
semantically entailed by sources.

## Deployment / portfolio status

This milestone creates a *runnable* local Gradio UI, with:

- A guided scripted tab that works without any model weights.
- A separate opt-in pinned Qwen tab accepting a custom request,
  category, budget, attributes, exclusions and zero/one revision.
- Retrieval and source provenance panels, source conflict metadata,
  original evidence excerpts and an explicit raw-unverified
  attempt-history panel.
- Fail-closed answer handling: **rejected drafts never appear as
  final approved recommendations**.
- Browser binding only to `127.0.0.1` and Gradio `share=False`.

**Not yet verified:** production server deployment, internet-facing
auth/security, a real shopping backend, browser-driven end-user
acceptance testing, large-scale recommendation accuracy, or semantic
citation groundedness. UI construction and event binding are
independently tested; no public-hosted URL is claimed.

To run locally, follow [Stage 4.0 Demo Guide](stage40_interactive_demo.md).
The next and final planned milestone, **Stage 4.1**, will consolidate
cross-stage metrics, honest benchmark limitations, a recruiter-facing
walkthrough and final repository closeout rather than adding new
unverified model techniques.
