# Stage 3.1 — Real Qwen Paired Multi-Agent Experiment

This is the first measured **model-backed** recommendation-agent experiment
using the Stage 3.0 orchestration framework. It compares **the same first
real Qwen generation** under two Critic/controller conditions:

- **0 revision:** deterministic Critic decides whether the first proposal
  can be approved. No generation repair is allowed.
- **At most 1 revision:** the exact same first Qwen proposal undergoes the
  exact same Critic; if rejected, the real Qwen generator may make one
  additional attempt with bounded Critic issue-code feedback. Final answer
  must still pass the deterministic Critic.

The Critic and Preference & Retrieval roles are **Python rule-based
components**, not separate LLMs. Only the Recommendation role runs real Qwen.
This is a *multi-role controlled workflow*, **not** evidence of three LLMs
collaborating or semantic fact verification.

## Fixed evaluation settings

- **Data:** `data/retrieval_eval/catalog.json` (15 fictional product records).
- **Cases:** `data/baseline_eval/suite.json` (15 normal + 10 red-team),
  with existing original retrieval evaluation labels.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` pinned revision
  `826711e54e001c83835913827a843d8dd0a1def9`.
- **Generator:** `Qwen/Qwen2.5-0.5B-Instruct` pinned revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- **Inference:** CPU, `do_sample=False`, up to 240 new tokens,
  `top_k=3` retrieval, original unchanged recommendation instructions
  for first drafts.
- **Verification:** hard metadata constraints, membership of product/evidence
  IDs, explicit required-attribute field citation, five selected prohibited
  phrase strings. No independent semantic entailment judge.
- **No output normalization:** generated JSON is parsed strictly. Stage 2.3
  format normalization is **not** silently applied to either arm.

### Pairing and accounting

This experiment **does not generate two independent first drafts**. For each
positive case it physically generates the first Qwen proposal **once** with
the same unmodified initial prompt as the Single-Agent baseline. Both arms
are scored from that **identical first proposal and Critic verdict**.
The revision arm may then call the same real Qwen model once more. The
`zero_revision` arm is a **projection** of the first attempt, not a second
physical generation.

Reports distinguish:

- `physical_model_invocations` — Qwen calls actually executed.
- `first_draft_model_invocations` — calls shared by both arms.
- `revision_model_invocations` — additional physical calls used only by the
  revision arm.
- Per-arm **logical** generator attempts / tokens / generation time. Since
  first calls are shared, adding per-arm totals would **double-count**
  physical cost. Latency measures model-call time excluding model loading,
  not end-to-end network or application overhead.
- Per-case first and revised raw model outputs, Critic issue codes,
  output status, generation tokens, result IDs, and prompt/context SHA-256.

This pairing isolates the incremental **one-revision controller effect**.
The earlier historical Single-Agent strict 25-case baseline uses the same
model, catalog, labels and initial prompt but ran separately, and the Critic
approval gate differs. Its scores may be shown as context, but historical
cross-run comparisons must **not** be described as a randomized or
individually paired architecture effect. The experiment optionally checks
the exact text agreement of first drafts against the archived baseline.

## Important known limitation: missing independent evidence

The original catalog has title/price references but **no explicit evidence
records for 29 known attribute fields**. The Stage 3.0 Critic **requires
field citations** for each explicitly required positive attribute.
Therefore many genuine metadata-eligible proposals will be rejected because
that requirement is **impossible to satisfy** with the original catalog.
This is an intentionally stringent fail-closed policy, but it
confounds any simple "Multi-Agent improves accuracy" claim.

Stage 2.3 generated a separate synthetic metadata-derived catalog with
29/29 field citations, **not independent product corroboration**.
**We do not use it here**, because changing the evidence corpus would make
this comparison incomparable to the frozen Single-Agent benchmark.

A future study could re-run *all* configurations on the same augmented
catalog, plus genuinely independently authored or sourced evidence, and
measure semantic claim entailment with calibrated human annotation.

## Success metrics and their limits

The judge uses the **frozen author-labeled** relevant IDs and expected
no-match cases. A positive success needs the final Critic to approve at least
one recommended product, with all IDs within the expected relevant set and
no pre-registered forbidden phrase appearing in the final reason.
Six structurally impossible requests must abstain without invoking Qwen.
We separately report approval, rejection, model calls, token totals,
generation-only latency and detailed Critic issue counts.

**The following remain null:** real-world relevance, unsupported atomic
claim rate and semantic citation groundedness. Narrow prohibited-phrase
checks are **not** language understanding. A structurally approved answer
may still make false claims.

## Reproduction

The dedicated `Real Qwen Multi-Agent Paired 25` Actions workflow downloads
open model weights, runs the unchanged full case set and publishes a complete
`qwen-multiagent-paired-25-real-cpu` JSON artifact. The workflow also loads
the original frozen baseline ZIP (artifact `14766315185`) and verifies its
SHA-256 before producing a historical diagnostic. GitHub's artifact
retention policy may eventually make that historical download unavailable.

Local reproduction (public weights download; Python 3.11+):

```bash
python -m pip install -e ".[dev,semantic,local-llm]"
python -m pytest
python -m recsys.evaluation.run_multiagent_eval \
  --catalog data/retrieval_eval/catalog.json \
  --suite data/baseline_eval/suite.json \
  --output artifacts/evaluation/multiagent-qwen-paired-25.json
```

Do not claim success until model-inference steps and artifact completeness
checks have finished. A green unit-test job only validates deterministic
wiring and contracts, not model quality. No model improvement results are
invented.
