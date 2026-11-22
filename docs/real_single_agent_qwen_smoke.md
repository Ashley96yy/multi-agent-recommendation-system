# Stage 2.1 — Real Qwen Single-Agent RAG Integration

## Purpose

Demonstrate **real, auditable end-to-end inference** across
MiniLM embeddings → metadata-filtered retrieval → one Qwen generation →
strict output/citation-ID checks. This is a small CPU smoke experiment, **not**
a production model benchmark, multi-agent implementation, or independent
assessment of whether recommendation prose is supported by the evidence.

## Model choice

- Generation: `Qwen/Qwen2.5-0.5B-Instruct` (small CPU-capable pilot, not the
  final planned 7B/9B model)
- Model revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- Retrieval: `sentence-transformers/all-MiniLM-L6-v2`
- MiniLM revision: `826711e54e001c83835913827a843d8dd0a1def9`
- Token decoding: greedy `do_sample=False`, `max_new_tokens=240`
- No quantization, prompt repair, additional generations, retries or critic.

The pinned Qwen revision was checked to contain the model config, tokenizer,
and safetensors weights. Only open public model weights and fictional
products are used. No external API keys, team data, or actual user histories.

## Fixed smoke cases

See `data/llm_smoke/cases.json`.

- 2 positive requests with eligible `SYN-001` and associated fictional evidence.
- 2 negative requests where no product satisfies the hard constraints.
- A positive case passes only if a recommendation with the expected ID
  survives structural checks.
- A negative case passes only if the system abstains **without** invoking
  Qwen (deterministic prefilter).
- Schema violations, unsupported product IDs, nonexistent citation IDs,
  and model failures are recorded rather than concealed.

## Running

The expensive CPU experiment runs in the dedicated
`Qwen Single-Agent Smoke` GitHub Actions workflow. Its artifact contains
`qwen-single-agent.json` with actual metrics, per-case statuses, model
versions, source dataset SHA-256 values, and raw output for these *synthetic
cases only*.

With a suitable local Python 3.11 environment:

```bash
python -m pip install -e ".[dev,semantic,local-llm]"
python -m pytest
python -m recsys.evaluation.run_single_agent --retrieval-backend minilm \
    --output artifacts/evaluation/qwen-single-agent.json
```

The first local call downloads public weights and can be slow on a CPU.

The standard lightweight CI does **not** download or execute Qwen.
Only the dedicated smoke workflow supplies evidence of real LLM execution.
Do **not** mark inference as successful until the workflow passes and the
artifact reports a nonzero number of model invocations and actual outputs.

## Interpreting results

`positive_accept_rate` measures **schema-accepted recommendations with
allowed candidate IDs** on two small synthetic inputs. It is not a factual
groundedness, medical safety, recommendation relevance or citation-entailment
metric. `negative_no_match_rate` tests the deterministic hard-filter bypass
of model generation on two synthetic cases. Per-case latency includes the
retrieval/generation pipeline but excludes initial model loading.

Small-model failures and abstentions are valid findings. They should not be
silently changed or reported as successful outputs. Later controlled
single-agent vs multi-agent comparisons must use a larger frozen evaluation
suite, the same model and retrieval evidence, and explicit semantic
verification of claim support.
