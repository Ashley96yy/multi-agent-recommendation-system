# Stage 3.2 — Controlled Output Reliability and Evidence Verification

## Why this stage exists

The frozen Stage 3.1 real Qwen experiment evaluated 25 **fictional**
recommendation scenarios with a deterministic Critic and at most one revision.
It found **zero successful strict positive cases** in either revision arm,
while the revision arm used **16 additional Qwen calls**. This stage addresses
two separately observed engineering weaknesses:

1. JSON-format failures. Original 0.5B Qwen first attempts often omitted
   the required `no_suitable_match` boolean or wrapped JSON in Markdown.
2. Missing attribute evidence. The original 15-product catalog contains
   title/price excerpts but lacks linked source records for many required
   attribute fields.

**Original experiments, original data, and the historical Single-Agent
pipeline are NOT modified.** New runtime policies are opt-in.

## A. Opt-in format policy

`MultiAgentOrchestrator(..., format_policy="strict")` is still the default,
matching Stages 3.0/3.1.

`format_policy="safe_json_normalization"` performs only two deterministic
operations on a response:

- Remove one complete whole-response Markdown JSON code fence.
- If the JSON root contains a **nonempty** recommendations array but omits
  `no_suitable_match`, insert `false` (not if any boolean is already present).

It never alters recommended IDs, cited evidence IDs, written reasons,
excluded constraints, existing boolean values, or invalid JSON contents.
The strict Pydantic response validator and Critic still run afterward.
Policy actions are recorded in each `AgentAttempt.format_actions`.

This is **post-processing**, not constrained decoding, prompt improvement,
semantic repair, model feedback, or an LLM fact-checker. Real constrained
decoding with JSON schemas may be investigated in a separate experiment.

## B. Optional early evidence preflight

`evidence_preflight=False` is the default historical behavior.

When enabled, the orchestrator checks **before calling any generator** that
each eligible retrieved product has a citation record for every explicitly
required positive attribute. It removes evidence-incomplete candidates
from the prompt. If *no* retrieved candidate has all required field records,
it returns `status="insufficient_evidence"` and **zero model calls**.

This prevents wasting Qwen calls or revisions on conditions that can never
pass the current deterministic field-citation Critic. It also **changes the
candidate context**, so its impact on live recommendations must not be
interpreted using the old model answers as if they were newly generated.

**Crucial limitation:** this policy validates **field-reference availability
only**, not the truth of the excerpt. A document could contradict another
document; a valid citation ID does not establish natural-language support.

## C. New separate synthetic multi-source fixture

Two new files are supplied:

- `data/source_evidence_demo/source_documents.json` — seven manually
  written **fictional** source passages. Each passage has explicitly typed
  claims and literal-verbatim excerpts.
- `data/source_evidence_demo/catalog.json` — a separate four-product
  fictional catalog whose evidence excerpts are indexed from those passages.

The fixture deliberately includes these known situations:

| Product | Fixture finding |
| --- | --- |
| `SYN-E01` | Two consistent, specifically cited positive attributes |
| `SYN-E02` | The fictional product spec and batch QC document **conflict** on fragrance |
| `SYN-E03` | Later fictional revision **supersedes** a former mineral-sunscreen claim, making metadata stale |
| `SYN-E04` | Product metadata declares fragrance-free, but the fictional listing **does not support** it |

The `recsys.evaluation.source_audit` module checks:

- Every evidence excerpt appears **verbatim** in its claimed source text.
- Every evidence record links to the correct product, document, source type
  and field; no extra/orphan references are silently accepted.
- Revision documents postdate and supersede the correct earlier source.
- Active contradictory typed claims, superseded facts, stale product
  metadata and missing source-claim coverage are separately reported.
- The input catalog and source-documents each retain exact SHA-256
  provenance hashes.

This fixture is **not actual ecommerce data**. Its documents were composed
for testing, and even independent *fictional* passages do not constitute
real-world corroboration. Typed claims provide a deterministic oracle for
the cases we wrote. The auditor is **not** a general natural-language
entailment model and cannot determine whether unseen arbitrary text is
true or false.

## D. Honest comparison using recorded real Qwen outputs

The `recsys.evaluation.replay_stage32` runner consumes the original
Stage 3.1 GitHub Actions artifact (run
[45053525292](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45053525292);
artifact ID `14766901381`). It reuses the same recorded 25-case IDs,
same retrieved candidate ID ordering, same frozen catalog and labels,
and exact raw real Qwen **first drafts**.

It applies strict and safe-format policies **offline**, then reuses the
deterministic Critic against the original cited fields. Strict replay must
reproduce the historical **three Critic-approved proposals and zero
strict positive successes** before normalization results are trusted.

The runner also calculates how many original requests *would have failed*
the field-availability preflight. Those are hypothetical model calls saved,
**not measured runtime savings or live rerun outcomes**. Changing the candidate
set would have changed the generated output, so the preflight policy is
**not** evaluated as a paired live model run in this stage.

The same original 19 initial Qwen drafts are shared; **zero new Qwen calls**
are performed. The report never claims semantic citation groundedness or
unsupported factual claim rate, both of which remain `null`.

## Reproduce

Python 3.11+; local deterministic tests need no model download:

```bash
python -m pip install -e ".[dev]"
python -m pytest

python -m recsys.evaluation.source_audit \
  --catalog data/source_evidence_demo/catalog.json \
  --sources data/source_evidence_demo/source_documents.json \
  --output artifacts/evaluation/stage32-source-audit.json
```

For a post-hoc replay, recover the original real Qwen JSON from the
Stage 3.1 artifact as `artifacts/evaluation/stage31-real-qwen-original.json`,
then run:

```bash
python -m recsys.evaluation.replay_stage32 \
  --original-report artifacts/evaluation/stage31-real-qwen-original.json \
  --output artifacts/evaluation/stage32-first-draft-replay.json
```

The standalone GitHub Actions workflow
`Stage 3.2 Structured Output and Source Audit` fetches the exact artifact,
runs all offline tests and the evidence audit, then uploads replay JSON,
source audit JSON and the original artifact ZIP hash. The experiment is
computationally inexpensive because it does not reload or call Qwen.

## Next evaluation standard

The next **live** study must hold model/prompt/constraints/source-version
fixed per arm and separately test whether these opt-in policies improve
strict outcome success and reduce actual cost. A calibrated semantic
claim-support judge or independently adjudicated human sample is still
required before claiming evidence-grounded generation quality.
