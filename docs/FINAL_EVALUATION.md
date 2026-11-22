# Final Evaluation — Evidence-Grounded Multi-Agent Recommendation System

**Portfolio closeout:** 2026-11-22 · **New model calls in this closeout: 0**

This document consolidates **existing, successful, recorded GitHub Actions experiments**. It is **not a new matched benchmark**, and numbers across distinct policies must not be summed, treated as comparable accuracy estimates, or interpreted as a production score. The detailed reports, source run links, and metrics are indexed in [the machine-readable evidence register](../data/portfolio/evidence_register.json). All catalog items, source documents, and labels used here are **fictional or author-constructed**.

## Executive takeaway

We implemented a three-role **Preference & Retrieval → Recommendation → Critic & Verification** workflow, with **zero or one bounded revision**, and a local Gradio demonstrator. The retrieval and deterministic checks prevent excluded or otherwise ineligible items from proceeding, verify cited IDs and typed source provenance, and **fail closed before generation** when required source fields are contradictory, obsolete, or unavailable.

The system is **not** three independent LLM agents: retrieval and the main approval Critic are deterministic Python; the recommendation role calls a real pinned Qwen model when enabled. A separate experimental Qwen atomic-claim classifier was evaluated but **is not the source of truth for live approvals**.

**The important observed result is a negative one:** on the original tiny synthetic author-labeled set, one extra model revision did **not** improve strict recommendation success (0/19 in both paired arms) despite an additional 16 real Qwen calls. Output-format recovery improved mechanical acceptance, while the experiment with a small language-model Critic did not establish semantic groundedness.

## Benchmarks: measured values and interpretation

| Stage / experiment | Exact population and source | Observed result | What it does *not* prove |
| --- | --- | --- | --- |
| 1.4: retrieval backend | 15 positive author-labeled retrieval cases, 3 hard no-match cases | MiniLM **Recall@5 1.0000**, lexical hashing **0.9333** | General relevance or independent accuracy |
| 2.2: real-Qwen Single-Agent | 25 fictional requests: 15 normal + 10 red-team; 19 positive and 6 impossible | **2/19** strict positive successes; **13/19** generated outputs invalid by strict schema; 19 Qwen calls | Factual entailment or production success rate |
| 2.3: format-only *offline replay* | Same saved 19 Qwen generations + 6 no-match, without new inference | Strict **2/19** → normalized **12/19** positive author-label successes; **0** additional model calls | A real live-model gain or semantic claim verification |
| 3.1: real-Qwen paired deterministic Critic | Same 25 fictional requests; **19 physical first drafts shared** between two arms | **0/19** strict positive success with 0 or 1 revision; up to 1 revision caused **16 additional physical Qwen calls** (35 total) | Superiority of one architecture; model revision efficacy |
| 3.3: real first-pass policy matrix | Same 25 fictional cases, 4 logical arms, 19 shared first generations | With normalized output **4/19** strict positive successes, with or without preflight; preflight logical calls **19 → 10**, but nine positive requests exit as insufficient evidence | Nine saved physical model calls in that combined matrix, or improved positive quality |
| 3.4: typed-source verifier | 22 explicitly typed plus 4 free-text author-composed claims on seven fictional source passages | Typed rule agreement **22/22** vs naive citation-only **8/22**; four free-text examples abstain | Independent human-groundedness or free-text entailment |
| 3.5: separate model-based Qwen Critic | 26 authored fictional atomic claims; strict JSON plus rationale | **3/26** schema-valid outputs, **0/26** matched author labels | Deployable semantic fact-checking |
| 3.6: real-Qwen prompt/decoding ablation | Three *new* independent arms × 26 authored claims = **78 calls** | Brief prompt **18/26** valid labels and **7/26** author matches; improved prompt **26/26** and **8/26**; true token mask **26/26** and **8/26**, changing **0** predictions vs improved prompt | That a valid four-class label means the source verdict is correct |
| 4.0: runnable local Gradio | Six **prewritten** fictional E2E scenarios plus a separate one-request real Qwen smoke | **6/6 scripted** expected flows; **179** tests passed in optional UI job; **1 physical real Qwen** request structurally approved; Gradio built **52 components** and **3 callbacks** | Hosted deployment, user acceptance testing, or real-world recommendation accuracy |

### Original experiment sources

- [Stage 1.4 MiniLM versus hashing](retrieval_results_20261122.md) · [Recorded run](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45048522798)
- [Stage 2.2 strict Single-Agent](single_agent_baseline_results_20261122.md) · [Recorded run](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45050806308)
- [Stage 2.3 frozen format-only replay](baseline_reliability_ablation_20261122.md) · [Recorded run](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45052268027)
- [Stage 3.1 real paired revision experiment](real_multiagent_qwen_results_20261122.md) · [Recorded run](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45053525292)
- [Stage 3.3 real first-pass preflight](stage33_real_qwen_results_20261122.md) · [Recorded run](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45056291366)
- [Stage 3.4 author-labeled atomic claim audit](stage34_atomic_claim_grounding_20261122.md) · [Recorded run](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45059212641)
- [Stage 3.5 real model-critic negative result](stage35_real_qwen_critic_results_20261122.md) · [Recorded run](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45060484582)
- [Stage 3.6 78-call decoding and prompt comparison](stage36_real_qwen_results_20261122.md) · [Recorded run](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45062087041)
- [Stage 4.0 local UI and one-call Qwen smoke](stage40_e2e_results_20261122.md) · [Recorded run](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45063616834)

## Four findings worth explaining in an AI Engineer interview

### 1. Retrieval success is not recommendation success

Stage 2.2's top-3 retrieval contained an author-labeled expected product in **19/19** positive cases, but strict end-to-end recommendation success was just **2/19**. Most generated responses failed a required structured response contract (**13/19**). This isolates a practical system reliability bottleneck after retrieval. The highly favorable retrieval score reflects a **tiny authored catalog**, not production retrieval generalization.

### 2. Retries cannot manufacture absent evidence

Stage 3.1 re-used the same 19 real first drafts and issued **16 actual Qwen revision calls** for rejected proposals. None of those revisions passed the strict acceptance gate. Three of the initial parseable proposals lacked references for required attribute fields, which a model retry cannot create. For the same controlled paired policy, strict success remained **0/19** and generation overhead increased. Comparing this to the earlier Single-Agent's **2/19** must account for the **stricter deterministic Critic gate**: it is not a causal measure of "multi-agent reasoning lowers accuracy."

### 3. Validate source claims before calling the model

Stage 3.3's evidence preflight reduced the *logical standalone policy* requirement from **19 to 10** generator calls by refusing nine positive requests with insufficient field-level references. Those nine are **failures to recommend**, not successful recommendations. The 2×2 benchmark shared physical first drafts across arms: it did **not** physically save nine calls across the combined experiment. On a separate manually authored source-document fixture, conflicting active sources, a superseded mineral declaration and a missing fragrance declaration all led to **zero-call abstentions**.

### 4. Schema compliance is not groundedness

Stage 2.3's offline normalization increased strict case success from 2/19 to 12/19 **without new model calls**, but did not rewrite unsupported medical assertions; the documented `RT08` failure illustrates this risk. Stage 3.5's small Qwen Critic was mostly unable to obey a four-class JSON contract, and Stage 3.6's stronger label-only prompt plus genuine next-token whitelist produced **26/26 legal labels but only 8/26 author-label matches**. None of the six authored Ambiguous cases was recognized in the improved-prompt arm. Therefore the interactive demo uses the auditable **typed deterministic source guard** for hard acceptance decisions and does not silently use the experimental semantic Critic.

## Reproducibility, provenance and denominators

The exact original model and embedding versions are pinned in the respective reports. The main model is `Qwen/Qwen2.5-0.5B-Instruct`, commit `7ae557604adf67be50417f59c2c2f167def9a775`; when MiniLM is used, it is `all-MiniLM-L6-v2`, commit `826711e54e001c83835913827a843d8dd0a1def9`. CPU, greedy generation, top-k, source snapshots, original prompt hashes, per-case outcomes and tokens are archived in the original GitHub Actions artifacts.

**Do not add token counts across shared experimental arms**, and do not interpret offline-replay scores as newly generated samples. The Stage 4.0 demo uses a **lexical hashing retriever by default** to allow offline use; its real Qwen optional mode is therefore not a reproduction of the MiniLM benchmark. The demo's six scripted outputs are test fixtures, not model inferences.

Some historical artifact downloads are subject to GitHub Actions retention. Source commits and report pages are durable repository files; a link to a particular Actions artifact is not a promise that the ZIP will always remain available.

## Explicitly not established

**Real-world unsupported-claim rate, independent human-calibrated semantic accuracy, and production recommendation accuracy are unmeasured.** No zero or perfect values are imputed for these missing quantities.

- Independent human-labeled semantic citation entailment, real-model unsupported-claim rate, arbitrary free-text atomic-claim extraction or completed independent reviewer consensus. The human-review CSVs and two-reviewer protocol exist; **zero independent judgments** are reported.
- Factual truth or safety of *real* skincare products; all products and passages in these experiments are fictional.
- General recommendation performance or statistically representative improvements across actual consumers.
- Dollar API cost, deployed reliability, authenticated web-service security, live product inventory, hosted application URL, usability study or production-grade latency guarantees.
- Genuine multi-LLM-agent collaboration. The live demo has three **functional roles**, only one of which is model-generated, with deterministic retrieval and deterministic Critic verification.

**Closeout criterion:** the project is portfolio-demo complete with a verified local executable UI and auditable negative results, **not a production recommendation system**. See [the local recruiter walkthrough](RECRUITER_DEMO.md) and [implementation architecture](SYSTEM_ARCHITECTURE.md).
