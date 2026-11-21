# Stage 1.3 — Catalog Retrieval

This is a **metadata-filtered vector-search foundation**, not a complete RAG or recommender agent.

## Running offline

Install and test:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Example:

```python
from decimal import Decimal
from recsys.retrieval import CatalogRetriever
from recsys.schemas import UserPreferences

retriever = CatalogRetriever.from_json("data/sample_catalog/catalog.json")
preferences = UserPreferences(
    request="fragrance free moisturizer",
    max_budget_usd=Decimal("30"),
    required_attributes={"fragrance_free"},
)
result = retriever.retrieve(preferences, top_k=5)
for candidate in result.candidates:
    print(candidate.product.product_id, candidate.similarity)
    print([e.evidence_id for e in candidate.evidence])
```

### What is guaranteed

- Product metadata filters run *before* similarity ranking.
- Product-ID and brand exclusions, category, maximum budget, and required true-valued attributes are supported.
- Unknown prices do not pass a stated budget; unknown attributes do not satisfy positive requirements.
- Each result includes matching evidence records and the catalog file's SHA-256.
- Ineligible items are counted by the first failed rule, not double-counted.
- Exact in-memory cosine search is used; ties are broken by product ID.
- A clear no-match reason is returned instead of fabricated products.

### What is not guaranteed

- The default `HashingEmbedder` is an **offline lexical-vector baseline**, not semantic retrieval or a trained embedding model.
- The presence of an evidence excerpt does **not** establish that it factually supports its attributed field. A later verification module must check that.
- There are no LLM calls, agent instructions, citation-groundedness claims, or real product recommendations.
- Users must supply structured `UserPreferences`; free-text parsing comes later.
- This proof of concept uses three deliberately synthetic products, not the team's restricted data.

### Optional semantic backend

Install `python -m pip install -e ".[semantic]"` and initialize
`MiniLMEmbedder(revision="<FULL_40_CHARACTER_HUGGING_FACE_COMMIT_SHA>")`.
The model download and inference require a separate environment with appropriate
network and local compute. CI does not download models or claim semantic search quality.

Later milestones should separately evaluate relevance and retrieval recall before
promoting an embedding backend to the benchmark configuration.
