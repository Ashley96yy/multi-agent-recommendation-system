# Stage 2.0 — Single-Agent RAG Baseline

Single-agent pipeline (one generation pass only):

1. Validate structured `UserPreferences`.
2. Filter catalog metadata, rank products, and retrieve linked evidence with
   the Stage 1.3 / 1.4 retriever.
3. Build JSON context containing **only** retrieved candidate products and
   their evidence references.
4. Call one `Generator.generate(instructions, context)`.
5. Parse `SingleAgentAnswer` with strict Pydantic schema and reject
   hallucinated product IDs, absent evidence IDs, or cross-product citations.
6. If no products survive retrieval, return `no_suitable_match` without
   calling a generator.

This is the **single-agent control condition**, not the multi-agent system.
There is **no critic, no automated revision, and no model-verified reason
groundedness**. Structural citation checks ensure identifiers appear in the
retrieved evidence but do not establish that the natural-language claims are
entailed by their excerpts.

## Running the offline tests

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

CI uses a `FixedJSONGenerator`, which is a **test double** and proves only
integration behavior, not actual LLM quality.

## Optional local OpenAI-compatible model

The `OpenAICompatibleGenerator` adapter sends one chat-completion request to
a trusted model endpoint with `response_format={"type":"json_object"}`.
For example, with a locally running compatible server:

```python
from decimal import Decimal
from recsys.baseline import SingleAgentRAG
from recsys.baseline.generators import OpenAICompatibleGenerator
from recsys.retrieval import CatalogRetriever
from recsys.schemas import UserPreferences

retriever = CatalogRetriever.from_json("data/sample_catalog/catalog.json")
generator = OpenAICompatibleGenerator(
    base_url="http://127.0.0.1:8000/v1", model="YOUR_MODEL_ID"
)
answer, retrieval = SingleAgentRAG(retriever, generator).recommend(
    UserPreferences(request="fragrance free moisturizer under $30",
                    required_attributes={"fragrance_free"},
                    max_budget_usd=Decimal("30"))
)
print(answer.model_dump_json(indent=2))
```

This example uses the deterministic lexical retrieval baseline. For a real
semantic retrieval + generation experiment, initialize the retriever with
`MiniLMEmbedder` and its pinned revision as documented in
`docs/semantic_retrieval_evaluation.md`.

LLM endpoints vary in support for JSON response mode. Failed output or network
errors are **not silently repaired**, because a single generation attempt is
part of the controlled baseline definition. No real inference result, model
quality metric, or end-to-end latency benchmark is claimed from these tests.
