# Stage 1.4 — Semantic Retrieval Evaluation

The evaluation measures candidate retrieval over a **fully synthetic** catalog
(15 fictional skincare listings). All 15 positive queries and three negative
(no-eligible-product) cases are authored for this project, not sourced from
human relevance judges or a real ecommerce dataset.

## Reproducibility

- The model backend is `sentence-transformers/all-MiniLM-L6-v2`, pinned to
  Hugging Face revision `826711e54e001c83835913827a843d8dd0a1def9`.
- The lexical hashing backend is an offline baseline, not a trained model.
- `catalog_sha256` and `suite_sha256` identify the exact evaluated files.
- Experiment output records versions for Python, NumPy, Pydantic, and
  Sentence Transformers; results are uploaded as CI artifacts.
- CPU is the default to avoid requiring the developer's GPU for evaluation.

To reproduce the lexical test:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m recsys.evaluation.run_retrieval --backend hashing --output artifacts/evaluation/hashing.json
```

To run a real semantic experiment (downloads public weights):

```bash
python -m pip install -e ".[semantic]"
python -m recsys.evaluation.run_retrieval --backend minilm --device cpu --output artifacts/evaluation/minilm.json
```

The dedicated `Semantic Retrieval Evaluation` GitHub Actions workflow runs
both backends on identical frozen inputs, and uploads **measured**, not
hand-entered results. It is deliberately separate from lightweight unit-test CI.

## Metrics

Each query has an explicit nonempty set of relevant product IDs.
For every cutoff K=1,3,5:
- Recall@K = relevant retrieved in top K / all labeled relevant.
- Hit Rate@K = fraction of queries having >=1 relevant result in top K.
- MRR@K = mean reciprocal rank of the first relevant result in top K.
- NDCG@K = binary relevance discounted gain / ideal discounted gain.

Separately, the three no-match cases are scored by exact expected no-match
reason and absence of candidates. Benchmark creation verifies that labeled
relevant products are eligible under each query's metadata filters, to avoid
mistaking a filter violation for an embedding failure.

## Limitations

This toy benchmark has a tiny catalog, author-written relevance labels and
potential wording overlap; its results cannot support real-world quality or
safety claims. No classifier/LLM parses the preferences: metadata is structured.
The existence of a linked evidence excerpt does not validate its truth.
The semantic workflow may fail due to Hub access, dependency incompatibilities,
or network restrictions; only a successful model-loading and inference run
counts as actual semantic execution. Compare per-case outcomes before claiming
MiniLM outperforms the lexical baseline.
