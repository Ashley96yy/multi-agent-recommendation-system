# Stage 1.4 — Measured synthetic retrieval results

**Run:** [GitHub Actions Semantic Retrieval Evaluation #45048522798](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45048522798)

**Artifact:** [synthetic-retrieval-evaluation](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45048522798/artifacts/14764442484) (contains `hashing.json` and `minilm.json`).

This is a **successful recorded run on 2026-11-22**, at GitHub commit
`e3d073b16b24d8a61ce04d0e102f597e6452862d`.
The model was downloaded, weights loaded, and CPU inference and scoring completed.

- Source catalog: `data/retrieval_eval/catalog.json` — SHA-256 `c5d5bf7dea2e99db26c9d10e3485858586518f454d6b69ee77956b826cd32310`
- Label suite: `data/retrieval_eval/suite.json` — SHA-256 `0e34ce175217267bd16aa6694cf0d2ce6152c2370fe36343561347a3749e7cd0`
- 15 author-constructed positive scenarios; 3 negative, no-eligible-product scenarios
- Model: `sentence-transformers/all-MiniLM-L6-v2` revision `826711e54e001c83835913827a843d8dd0a1def9`; runner: Python 3.11 on Ubuntu CPU

| Metric | Lexical hashing | MiniLM |
|:--|--:|--:|
| Recall@1 | 0.8000 | 0.8667 |
| Recall@3 | 0.9333 | 1.0000 |
| Recall@5 | 0.9333 | 1.0000 |
| Hit Rate@5 | 0.9333 | 1.0000 |
| MRR@5 | 0.9000 | 0.9667 |
| NDCG@5 | 0.9087 | 0.9754 |
| Negative no-match pass rate | 1.0000 | 1.0000 |

**Interpretation:** MiniLM ranked labeled products more effectively on this
tiny **synthetic, manually labeled** dataset. This is not evidence of
production recommendation accuracy, independent human relevance, safety,
or a statistically established generalization gain. Negative checks verify
the metadata filtering pathway, not semantic correctness or LLM groundedness.

Raw per-case results, environment library versions and provenance hashes are
in the artifact, rather than re-created by hand.
