# Stage 2.1 — Recorded Real Single-Agent Qwen Smoke

**Verified:** 2026-11-22. **Run:** [Qwen Single-Agent Smoke #45050082816](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45050082816) (completed successfully).

**Raw artifact:** [qwen-single-agent-real-cpu](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45050082816/artifacts/14765545092), containing real per-case outcomes, model outputs, generated token counts, and recorded library versions.

**Source GitHub commit:** `aedd6de980a04eebfd9bbfa0c97ddede7ec99363`. This run verified the explicit-attention-mask implementation.

## Observed real inference

| Metric | Measured result |
| --- | ---: |
| Fixed synthetic cases | 4 |
| Positive request cases | 2 |
| Valid positive recommendation outputs | 2 / 2 |
| Hard-filter no-match cases | 2 |
| Correct no-match with zero LLM call | 2 / 2 |
| Actual model generation calls | 2 |
| Generation or validation failures | 0 |
| Mean positive-case pipeline latency, excluding model loading | 14.4089 seconds |

Both positive recommendations passed the **structural** JSON/Pydantic,
retrieved-product-ID, and linked-evidence-ID checks. These checks do not
independently verify that the *natural-language recommendation reasons* are
entailed by the cited snippets. The negative cases used deterministic
filtering and correctly skipped Qwen.

## Reproducibility

- Qwen generator: `Qwen/Qwen2.5-0.5B-Instruct`
- Pinned revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- MiniLM retrieval: `sentence-transformers/all-MiniLM-L6-v2`
- Pinned retrieval revision: `826711e54e001c83835913827a843d8dd0a1def9`
- Retrieval catalog SHA-256: `ac96800757c649b82cd94f8872ecea06bf1da0fd16cda2c6b5a538fbc6351a18`
- Fixed-case suite SHA-256: `64bb45c1263e4ab4d2493943d61cf4ffede1a10fbd8c196f86b7b2073831a5c5`
- Runtime: GitHub Actions, Python 3.11, Ubuntu CPU, greedy `do_sample=False`
- The final run supplied an explicit attention mask; the earlier attention-mask and deprecated-dtype warnings were absent.
- Dependency versions and per-request measured tokens are recorded in the uploaded JSON artifact.

## Interpretation and limitations

This provides direct evidence that the integration works end-to-end with a
**real downloaded Qwen model and real MiniLM embeddings** on tiny fictional
data, with no private API key or user-owned GPU.

**Do not interpret 2/2 as 100% recommendation accuracy or citation
groundedness in production.** The catalog has only three fictional products
and the labels are author-constructed. Structured constraints are supplied
rather than extracted from free text. No hallucination/claim entailment
judge, semantic citation verifier, uncertainty-calibrated reviewer,
multi-agent critic, or revision loop has been evaluated yet.

For a research-grade baseline, expand the fixed scenarios and add separate
claim-support and preference-compliance scoring before comparing it with
the three-agent workflow.
