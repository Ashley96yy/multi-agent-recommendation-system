# Stage 3.0 — Multi-Agent Architecture & Contracts

This stage implements the **three role boundaries and deterministic bounded
controller**, not yet a measured LLM-based three-agent system.

## Agent roles

1. **Preference & Retrieval Agent**: accepts validated structured
   `UserPreferences` and reuses the existing metadata-filtered
   `CatalogRetriever`; no natural-language preference extraction yet.
2. **Recommendation Agent**: calls the supplied generator once per proposal,
   reusing the *same* `SYSTEM_INSTRUCTIONS` and `build_context` as
   the measured Single-Agent baseline on the initial attempt. If the critic
   rejects the proposal, the second attempt receives bounded structured
   **issue codes**, not the raw prior model output. A future Qwen integration
   can use the existing `LocalQwenGenerator` adapter.
3. **Critic & Verification Agent**: a **deterministic rules-based critic**.
   Checks whether products belong to the retrieved set; whether hard
   budget/category/brand/attribute restrictions still hold; whether
   citations belong to the product; whether required positive attribute
   fields have associated cited evidence; and a short list of explicitly
   prohibited phrases (including unsupported medical-cure claims).

**Important:** the critic is not currently a trained language-model
agent, calibrated fact checker or independent semantic-entailment judge.
A citation may have a valid ID without supporting the entire natural-language
reason. Synthetic metadata-derived attribute evidence is *not independent
real-world product corroboration*. Every `CriticVerdict` explicitly contains
`semantic_entailment_verified=false`.

## Bounded controller

```text
Validated preferences
    |
    v
Preference & Retrieval  ---- no eligible candidates --> no_candidates
    |
    v
Recommendation (one proposal)
    |
    v
Deterministic Critic -- approved --> approved or abstained
    |
    +-- rejected --> revision budget?
                          | no --> rejected (no answer disclosed)
                          | yes
                          v
                     Recommendation (one revised proposal)
                          |
                          v
                     Deterministic Critic
                          | approved --> approved or abstained
                          | rejected --> rejected (no answer disclosed)
```

Configure `max_revisions=0` to model a no-revision workflow, or `1`
for one revision. Other values are rejected. At most **two generator calls**
can occur per user request, with no hidden retry loop. When no products are
retrieved, the generator is skipped entirely. All failed proposal
attempts and their issue codes remain in the trace, but rejected answers
are not returned as accepted output.

### Example (offline, deterministic only)

```python
import json
from recsys.baseline.generators import FixedJSONGenerator
from recsys.retrieval import CatalogRetriever
from recsys.schemas import UserPreferences
from recsys.multiagent import (
    PreferenceRetrievalAgent, RecommendationAgent,
    CriticVerificationAgent, MultiAgentOrchestrator,
)

retriever = CatalogRetriever.from_json("data/sample_catalog/catalog.json")
fake = FixedJSONGenerator(json.dumps({
    "recommendations": [{
        "product_id": "SYN-001",
        "reason": "Its fictional catalog lists a fragrance-free attribute.",
        "evidence_ids": ["E-SYN-001"],
    }],
    "no_suitable_match": False,
}))
system = MultiAgentOrchestrator(
    PreferenceRetrievalAgent(retriever),
    RecommendationAgent(fake),
    CriticVerificationAgent(),
    max_revisions=1,
)
result = system.run(UserPreferences(
    request="fragrance-free moisturizer", required_attributes={"fragrance_free"}
))
print(result.status, result.generator_calls, result.revision_used)
```

This example uses a **fixed JSON test double**, not actual multi-agent
LLM inference. Unit tests establish controller logic, constraints, evidence
ID membership and hard call limits only.

## Comparison design

The existing results remain unchanged:
- Stage 2.2: one-pass Qwen baseline on 15 normal + 10 red-team synthetic
  scenarios, 2/19 strict positive successes, 13/19 schema failures.
- Stage 2.3: **post-hoc** format-only replay on the same Qwen output,
  normalized strict positive successes 12/19, **zero new Qwen calls**.

A future *live* comparison must use the same frozen dataset, model revision,
retrieval backend, candidate cutoff and evidence snapshot across
Single-Agent, Multi-Agent with 0 revision, and Multi-Agent with 1 revision.
Track generated token counts, attempted calls, latency, fail-closed rate,
constraints, and independent **atomic-claim** evidence adjudication.

Do not interpret an improved Pydantic acceptance rate caused by structured
feedback as evidence of better factual groundedness. When experiments
use an enriched synthetic catalog, clearly mark that as a **separate data
configuration**, and repeat the relevant control arm on that same catalog
to avoid confounding the multi-agent effect with evidence coverage changes.
