# Stage 3.1 — Measured Real Qwen Multi-Agent Paired Evaluation

**Date:** 2026-11-22. **Outcome:** The real 25-case CPU experiment executed
successfully. **One revision did not improve strict recommendation success**
and consumed additional model calls, tokens and time.

- [Full real Qwen GitHub Actions run #45053525292](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45053525292)
- [Original raw Qwen paired JSON artifact](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45053525292/artifacts/14766901381)
- **Exact experiment commit:** `d5325115147c835be5400a2945f32cc28f0fb06d`
- Model: `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`
- Retrieval: `sentence-transformers/all-MiniLM-L6-v2@826711e54e001c83835913827a843d8dd0a1def9`
- Same unmodified **15 fictional products, 15 normal cases, 10 red-team cases**
  and frozen labels as the Stage 2.2 Single-Agent baseline.
- CPU, greedy decoding, 240 max new tokens, retrieval `top_k=3`, strict JSON
  parser and deterministic structural/policy Critic.
- No post-hoc JSON normalization, additional model, independent evidence
  adjudication, or model-data changes.

## Experimental design

Each of 19 requests with eligible products had one real Qwen first draft.
The **no-revision arm is a projection of that very same first proposal and
Critic verdict**, not a separately generated model response. The
**one-revision arm** may call Qwen once more if the shared first proposal
fails the Critic. The six no-eligible-product cases skip Qwen in both arms.

This is a **paired revision-policy experiment**, not two independent model
runs, and it is **not three LLMs collaborating**. Only the Recommendation
role uses Qwen; retrieval and Critic roles are deterministic Python.

## Observed results

| Metric | 0 revisions | Up to 1 revision |
| --- | ---: | ---: |
| Synthetic cases | 25 | 25 |
| Strict positive successes | **0/19** | **0/19** |
| Normal-case strict successes | **0/15** | **0/15** |
| Red-team expected-outcome successes | **6/10** | **6/10** |
| Correct no-candidate skips | 6/6 | 6/6 |
| First/final approved generated cases | 3/19 | 3/19 |
| Rejected generated cases | 16/19 | 16/19 |
| Revision calls issued | 0 | **16** |
| Logical Qwen calls for this arm | 19 | **35** |
| Prompt tokens attributed to arm | 10,537 | **20,109** |
| Generated tokens attributed to arm | 2,023 | **3,645** |
| Mean generation-only seconds per generated case | 21.0287 | **38.5323** |
| Generation-only P50 / P95 seconds | 14.8657 / 30.8662 | **29.3455 / 62.5430** |

**Physical calls made in total = 35**: 19 first drafts shared by both
comparison arms and 16 real revision calls. Do not sum the 19 and 35
logical-arm counts into 54 physical calls. Model loading time is excluded
from the listed generation latency; values are descriptive CPU measurements.

**Reproducibility check:** all **19/19 first-draft raw model outputs exactly
matched** the independent historical Stage 2.2 Single-Agent model outputs.
The first instructions, model revision, retriever, catalog and labels were
unchanged. Historic Single-Agent strict success was **2/19**, whereas the
strict-Critic no-revision control here scored **0/19**. This difference
includes a **different acceptance gate**; it is not causal evidence that
multi-agent reasoning decreased factual accuracy.

## Root cause analysis — inspected original model traces

Among the 19 real first drafts:

- **13** were rejected for invalid strict generation/schema (some missing
  required JSON fields or using Markdown fenced JSON).
- **3** were parseable, but the Critic rejected them because required
  attributes did **not** have matching field-level evidence citations.
  The cases were **Q01, Q03, and Q15**; there were six missing-field issue
  records total, because multiple attributes/products were involved.
- **3** were approved structurally (Q05, Q06, Q12), but **none** satisfied
  the author-labeled strict recommendation criterion: each included
  additional unlabeled products.
- The 16 rejected initial cases triggered one real Qwen revision each.
  **Every one of the 16 revised outputs failed the strict JSON/schema
  requirement.** No revision became a Critic-approved recommendation.
  Some model outputs switched into fenced Markdown JSON; others omitted
  the required `no_suitable_match` field.
- In adversarial case **RT08**, an unsupported claim that a fictional
  product could `cure eczema` appeared in the unapproved model output
  and remained in the revision. Neither attempt passed the strict schema
  gate, so that statement was **not an accepted final system answer**.

### Why the required-attribute check cannot be repaired by a text-only retry

The frozen original catalog includes title/price references but has **no
explicit evidence references for 29 known attribute fields**. The
deterministic Critic's strict policy requires cited
`attributes.<field_name>` entries whenever those attributes are
explicitly required by a user. Q01, Q03, and Q15 therefore could not
meet this field-citation policy **even with perfect JSON and valid product
IDs**. Repeating generation cannot create real missing source records.

Stage 2.3 built a **separate** metadata-derived synthetic catalog with
field references, but it was intentionally not substituted here. Those
synthetic declarations are not independent corroboration, and swapping
catalogs between study arms would confound the comparison.

## Interpretation, not a victory claim

- On this **small, author-constructed** dataset, the deterministic Critic
  did **fail closed** on missing reference coverage and invalid outputs.
- But giving Qwen2.5-0.5B one feedback-coded revision **did not repair
  any rejected proposal**, instead increasing logical prompt tokens by
  **9,572**, generated tokens by **1,622**, and total generation-only
  time by approximately **332.57 seconds**.
- This does **not** demonstrate multi-agent superiority or inferiority
  on production recommendation data. It demonstrates that **missing
  evidence cannot be addressed by LLM retries** and that a small generator
  may fail strict JSON formatting on both passes.
- **Groundedness, factual unsupported-claim rate, semantic entailment,
  real-user relevance and real-dollar cost remain unmeasured**.
  `semantic_entailment_verified=false` is intentionally present in
  all trace records. Heuristic phrase checks are not a claim-level judge.

## Next engineering decisions

1. Keep this run, the original Single-Agent baseline and the Stage 2.3
   format-only replay **unchanged** as separate auditable reference points.
2. Add structured output/constrained decoding as a **separate controlled
   factor**, then test the same decoding policy for the Single-Agent and
   both Multi-Agent configurations.
3. Detect **unresolvable missing field evidence before calling a revision
   model**, so the system can fail closed without wasting tokens.
4. Build a **separate, independently authored synthetic product-source
   set**, including explicit attribute claims, contradictory documents,
   missing facts and revisions, and label atomic factual claims for human
   review. Do not present metadata-derived fields as independent evidence.
5. Only then test a real model-based Critic or verifier with a calibrated
   claim-support evaluation, keeping first-pass generator, candidate set,
   labels, decoding policy and evidence fixed across comparison arms.

**Result:** Stage 3.1 verifies that the real Qwen-integrated bounded
multi-role workflow executes and records fair paired traces. It does
**not** establish a recommendation-quality gain from a revision loop.
