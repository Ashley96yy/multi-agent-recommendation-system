# Stage 3.6 — True Token-Constrained Labels vs Prompt-Only Critic

## Purpose and controls

Stage 3.5's *real* Qwen2.5-0.5B-Instruct evidence critic produced a
**negative baseline**: only 3/26 schema-conforming JSON answers, and 0/26
matched the author labels on manually fabricated sources. On its 26
unmodified first drafts, 20 JSON objects used invalid label words
(`ambiguity` or `ambiguously`) and three outputs were invalid JSON.

Stage 3.6 preserves Stage 3.5, all 26 fictional atomic-claim inputs,
source passages, catalog, typed rules, human-review packets and author
reference labels unchanged. It tests **two separately identifiable
engineering factors** over the same 26 cases and pinned model. We do not
change to a larger model or tune on the 26 reference labels.

### Three NEW model arms

| Arm | System prompt | Decoding restriction | Model output contract |
| --- | --- | --- | --- |
| `plain_label` | Brief instruction to select one of four labels | Greedy only | Exact lowercase label only |
| `decision_label` | Clear, explicit decision-order rules for conflict, contradiction, missing citations and support | Greedy only | Exact lowercase label only |
| `decision_label_constrained` | **Identical prompt to `decision_label`** | Actual next-token four-label prefix mask | Exact lowercase label only |

- **Prompt effect:** `plain_label` vs `decision_label` changes only the
  system instructions (the user evidence context and label-only output
  contract are identical).
- **Decoding constraint effect:** `decision_label` vs
  `decision_label_constrained` uses the exact same system prompt and
  context, but a real Transformer decoding constraint in the latter.
- **Historical Stage 3.5:** original long system prompt with JSON label
  and free-text rationale, 0/26 author-label agreement. It is descriptive
  context only, **not a causal fourth arm**, because the prompt and output
  contract have both changed.

Every arm invokes the pinned **Qwen/Qwen2.5-0.5B-Instruct**
`7ae557604adf67be50417f59c2c2f167def9a775` once per case,
with the **same** four-product, seven-fictional-document corpus, the
**same 22 typed + 4 open-text** claims, no samples/temperature,
no retries or relabeling, on CPU. The first two arms allow up to 32
new tokens; the constrained arm caps tokens at the longest label
tokenization plus EOS. There are **78 genuinely independent real Qwen
model calls**, not 26 shared completions.

## What makes the constraint real?

`CanonicalLabelTrie` computes the exact tokenizer token sequences for
`supported`, `contradicted`, `unsupported`, `ambiguous`.
The `ConstrainedLabelQwen` adapter passes a
`prefix_allowed_tokens_fn` callback to Hugging Face
`AutoModelForCausalLM.generate()`. At each generation step the logits
for **all other token IDs are disallowed**, and only EOS is allowed after
a complete canonical label. The generated token string is validated
against the exact four labels before it becomes a prediction.

This is a **true token-level vocabulary constraint**. It is *not* a
post-hoc synonym mapping (we never convert `ambiguity` to
`ambiguous`), JSON correction, prompt suggestion, or externally
edited model response. The unconstrained arms also reject any whitespace,
markdown, prose, JSON or noncanonical spelling.

**Scope limit:** only the *label tokens* are constrained. The model does
**not** generate a JSON object or an explanation in this stage; downstream
code can wrap a validated label as a typed value, but that would be an
**application-produced object**, not model-generated JSON/rationale.
This experiment proves structural label compliance when the decoder
functions as intended, **not semantic source understanding**.

Reference API: [Hugging Face Transformers
`prefix_allowed_tokens_fn` (4.57 documentation)](
https://huggingface.co/docs/transformers/v4.57.1/internal/generation_utils).

## Measurement contract

For every model invocation we record:

- Original prompt + evidence-context SHA-256 and case ID.
- Unmodified raw model output, the actual strict label or explicit
  `needs_human_review` for invalid output.
- Real prompt and generated token counts, generation and wall-clock time,
  model call counts, and applied constraint name.
- Post-inference four-class evaluation against **separate, previously
  written fictional author labels**, including invalid outputs as errors.
- Separate all-26, typed-22 and open-text-4 case denominators, confusion
  matrices and per-class F1.

The Stage 3.6 prompts were designed **after inspecting Stage 3.5's
failures on this same 26-case suite**, so any apparent gain is exploratory
and must not be sold as a blind hold-out benchmark. The real experiment
runs arms in fixed order for each claim; CPU latency comparisons may
reflect ordering or caching effects and are descriptive rather than
controlled throughput measurements.

All scores are **author-label agreement on the tiny engineered fictional
fixture**. The model never sees the author reference labels during
inference, but this is **not independent human calibration** and not
external real-world claim truth. Model-based Critic performance on the
15 actual recommendation reasons from Stage 3.4 is still unmeasured.
Two genuinely independent blind human submissions are not available.

## Reproduce

```bash
python -m pip install -e ".[dev,local-llm]"
python -m pytest
python -m recsys.evaluation.run_label_ablation \
  --output artifacts/evaluation/stage36-real-qwen-label-ablation.json
```

The dedicated `Stage 3.6 Real Qwen Label Decoding Ablation` GitHub
Actions workflow checks that all 78 calls occurred, the constrained
arm has 26 structurally valid canonical labels, the two decision arms
have identical prompt/context hashes per claim, and every author-label
comparison explicitly reports human calibration as unavailable.
It uploads raw output, token/call traces and per-case failure analysis.

**Do not interpret improved label compliance as improved factual
groundedness.** If a constrained decoder still chooses the wrong
four-class label, that is a genuine semantic error that format
constraints alone cannot fix.
