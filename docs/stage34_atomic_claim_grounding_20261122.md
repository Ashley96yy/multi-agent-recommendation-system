# Stage 3.4 — Auditable Atomic Claim Grounding on Fictional Sources

**Run date:** 2026-11-22

- [Successful Stage 3.4 GitHub Actions run #45059212641](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45059212641)
- [Full predictions, source provenance and *blind* human-review CSV](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45059212641/artifacts/14768881666)
- Fixed case inputs: `data/atomic_claim_eval/cases.json`
- Separate author reference labels: `data/atomic_claim_eval/author_labels.json`
- Fictional primary sources: `data/source_evidence_demo/source_documents.json`
- Separate fictional indexed catalog: `data/source_evidence_demo/catalog.json`
- Logic: `recsys.evaluation.atomic_claims` and the typed-source classifier
  in `recsys.multiagent.source_guard.TypedSourceGuard`
- **No LLM was called** and the historical 15+10 Qwen benchmark was **not**
  reinterpreted as a semantic groundedness experiment.

## 1. What was actually evaluated

A frozen 26-case **synthetic author-labeled atomic claim** set contains:

- 22 **explicitly typed** assertions (product ID, field, boolean or decimal
  value, exact text span, cited evidence IDs), plus a separately authored
  four-class reference label.
- Four **open-text** claims for which the automatic adjudicator must
  return `needs_human_review`, including a fictional medical-efficacy
  claim, a statement of uncertainty and two paraphrases. Their
  human-authored reference labels are **excluded** from the automated
  agreement denominator.

The typed examples contain six Supported, five Contradicted, six
Unsupported and five Ambiguous author labels. All examples and gold
labels were written for the same small fictional document fixture.
**This is not independent reviewer agreement, a real-world ground-truth
dataset, or a test of natural-language claim extraction.**

### Four-label decision rules (explicit typed fields ONLY)

| Label | Deterministic meaning on active annotated sources |
| --- | --- |
| **Supported** | All active source declarations agree with the asserted value and **every citation ID** links to an active source for exactly the right product and field |
| **Contradicted** | Active unambiguous source declarations explicitly assert the opposite value, including when an older cited document was superseded |
| **Unsupported** | The field has no active source assertion, or the requested fact has missing, off-field, unknown, cross-product or superseded citation evidence despite active consistent information |
| **Ambiguous** | Active annotated source documents disagree on the value; the system refuses to resolve that conflict by choosing one convenient citation |

These categories label the **evidence support of a specific typed atomic
claim**, not the truth of the underlying product on the real market. In the
newer mineral-sunscreen revision example, catalog metadata remains stale,
so the active source document governs claim-level support while the
runtime product-eligibility guard can separately fail closed on the
metadata inconsistency.

The judge does **not** convert arbitrary prose to fields, does **not**
infer that a reason's *every* statement is represented by its submitted
claims, and does **not** judge nuanced paraphrases, clinical efficacy or
causal claims. Such examples are explicitly held for independent review.

## 2. Observed offline evaluation results

| Measurement | Recorded result |
| --- | ---: |
| Frozen cases | 26 |
| Typed cases automatically adjudicated | 22 |
| Open-text cases requiring independent review | **4** |
| Automated claim coverage | 22/26 = 84.6% |
| Four-class typed judge agreement with **author** labels | **22/22** |
| Naive control agreement with author labels | **8/22** |
| Four-class typed macro-F1 vs author labels | **1.00** |
| Real-world semantic groundedness / unsupported-claim rate | **Not measured** |

The simplistic control counts a typed claim as supported merely when
every cited ID exists and belongs to the product. It fails on
contradicted, conflicted, wrong-field or superseded evidence; its
8/22 author-label agreement makes that limitation visible.

**Do not present 22/22 as model accuracy or generalization.** This is
deterministic agreement with hand-written test cases from the same
fictional source fixture; the agreement would need independent labels,
an unseen evaluation corpus and claim-extraction coverage analysis to
support a robust quality claim.

### Representative cases

- `T01`: **Supported** — the cited fictional specification explicitly
  declares no added fragrance.
- `T10`: **Contradicted** — the citation references an older mineral-UV
  assertion, while the effective newer document says non-mineral.
- `T12`: **Unsupported** — fragrance-free metadata has no supporting
  passage in the active cleanser source listing.
- `T15`: **Unsupported** — a valid fragrance source is mixed with a
  citation from another product; one good citation cannot hide a bad one.
- `T18`: **Ambiguous** — an active fictional product spec and a batch
  quality-control document disagree about added fragrance.
- `F23`: **Needs human review** — the fictional model-like text claims
  clinical efficacy; citation membership alone cannot establish this.

## 3. Reproducibility and auditability

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m recsys.evaluation.atomic_claims \
  --cases data/atomic_claim_eval/cases.json \
  --labels data/atomic_claim_eval/author_labels.json \
  --output artifacts/evaluation/stage34-atomic-claims.json \
  --blind-review-csv artifacts/evaluation/stage34-blind-review.csv
```

The machine-readable result records, for each case:

- The explicitly typed or open-text claim and exact reason span.
- The product and cited evidence IDs.
- Predicted label or explicit human-review abstention.
- A transparent decision reason and the method used.
- The author reference label and explanation **only in the scored
  report**, never in the blind-review CSV.
- Full SHA-256 provenance hashes for cases, author labels, catalog
  and fictional source documents.

The CSV contains claim text, source excerpts, their source IDs, and
whether a citation was superseded, with empty
`reviewer_label`/`reviewer_id`/`review_notes` columns. It
deliberately omits author labels and automated predictions so reviewers
can evaluate evidence independently before unblinding. It has 26 rows.
No independent person has yet submitted such reviews.

## 4. Critical limitations and next phase

This stage **provides the evaluation foundation for semantic groundedness**,
but it is **not yet a general semantic groundedness metric**. Four
unstructured cases abstain correctly; real model-generated reasons have
not yet been exhaustively split into atomic claims. Neither a semantic
LLM Critic nor a human-calibrated NLI judge has been evaluated here.

A later phase should:

1. Pre-register how each accepted real Qwen reason will be segmented
   into *all* atomic factual claims; track omitted or unextractable spans.
2. Blindly annotate each span against its primary cited passages,
   including contradictory and revised sources, ideally with a second
   reviewer and disagreement adjudication.
3. Compare deterministic typed-source checks, a pinned model-based
   Critic, and a no-Critic control on the **same** corpus and first
   generator outputs, separately reporting cost, recall and human-labeled
   precision/recall for each support class.
4. Only report a real unsupported-claim rate with a defined denominator,
   complete reviewed claim sample and adjudication protocol.

All merchandise and source-document details in this stage are **fictional**.
No medical product claim, ingredient listing or performance benefit is
independently established by these tests.
