# Recruiter / Hiring Manager Demo — Two-Minute Walkthrough

**Audience:** Applied AI Engineer, ML Engineer or AI-backed backend engineering hiring team. **Deployment:** locally runnable Gradio demo; **no hosted/public URL** is claimed.

## Elevator pitch (20 seconds)

> I built an evidence-grounded recommendation workflow with three functional roles: preference-aware retrieval, local Qwen generation and a deterministic Critic with at most one revision. The system tracks actual citations and source versions, refuses recommendations with conflicting or missing required evidence, and makes invalid model outputs auditable rather than silently accepting them. I also ran controlled Qwen experiments that exposed when retries and output formatting don't actually improve factual reliability.

**Precision:** this means three engineering roles, **not three LLMs**; the real model-generated role is recommendation. The demonstration corpus and source documents are **fictional**.

## Open the local demo (no model needed)

From a clean checkout on Python 3.11+:

```bash
git clone https://github.com/Ashley96yy/multi-agent-recommendation-system.git
cd multi-agent-recommendation-system
python -m pip install -e ".[demo]"
python -m recsys.demo.app
```

Visit **http://127.0.0.1:7860**. Keep the default **Guided walkthrough** tab. It runs the real retrieval/source guard/orchestrator against **PREWRITTEN JSON drafts**; no Qwen weights are downloaded, no API key is required and the app exposes no public share link.

## Timed live walkthrough (about 2 minutes)

| Time | UI action | What to point out |
| --- | --- | --- |
| 0:00–0:20 | Select **01 · Supported source** → Run deterministic walkthrough | One fictional product is eligible. Show the final approved item, `E-SOURCE-001`, and the quoted active source passage. Clarify that the natural-language reason is not semantically certified. |
| 0:20–0:50 | Select **02 · One bounded revision** → Run | Open **4 · Agent attempts and bounded revision trace**. First draft contains the *fictional unsupported* clinical phrase; deterministic Critic emits `prohibited_claim_phrase`; exactly one revised draft is accepted, while the rejected text is never displayed as an approved final recommendation. |
| 0:50–1:15 | Select **03 · Conflicting source records** | Two active fictional sources disagree on fragrance. The result is `insufficient_evidence` with **zero generator calls**. Show the field diagnostics and source excerpts. |
| 1:15–1:35 | Select **04 · Superseded formulation** | The old mineral-UV statement was superseded by a later non-mineral revision; the required product attribute is blocked before any model generation. |
| 1:35–1:50 | Select **05 · Required source is missing**, then **06 · Budget hard-filter** | Differentiate **insufficient documented evidence** from **no eligible candidate**. Both skip generation but for different reasons. |
| 1:50–2:00 | Open **calls/status metrics**, then link to the results report | Point out source/catalog SHA-256, actual model calls vs scripted calls, revision count and the honest failures in the benchmark. |

This two-minute path is fully reproducible **offline** once dependencies are installed. The six scenarios use **scripted** example drafts; they prove application branching and traceability, not model quality. The project separately recorded **one actual Qwen** positive-control run in GitHub Actions for Stage 4.0, linked in [the E2E evidence report](stage40_e2e_results_20261122.md).

## If a technical interviewer asks for real inference

Install the opt-in model dependency:

```bash
python -m pip install -e ".[demo,local-llm]"
python -m recsys.demo.app
```

Choose **Real local Qwen**, enter `Cloud Calm Facial Cream fragrance-free moisturizer`, choose category `moisturizer` and required attribute `fragrance_free`, and run. The public pinned Qwen model may download on first eligible request and run on CPU; responses can be **approved, abstained or rejected**. This is one demo request, not a correctness benchmark. For a no-call demonstration, request `Dusk Barrier Balm` with required category `balm` and `fragrance_free`: the conflict-aware source preflight should refuse generation even in real mode.

**Live response not guaranteed:** the small model may produce invalid JSON or unsupported prose, and one revision may fail. That is why the result includes a Critic trace and **never rebrands a rejected draft as approved**.

## Engineering discussion (about 90 seconds after the demo)

**Q: Why multiple agents / roles rather than one prompt?**

The retrieval role makes auditable metadata decisions before spending model tokens; the Recommendation role generates a candidate only from the eligible evidence context; the Critic role enforces hard source and citation rules after generation. The bounded controller can react to a specific failure code once. These are independently testable service boundaries, not a claim of superior LLM intelligence. The paired real Qwen test found **no positive accuracy gain from the extra revision** at this model size, and the extra calls had measurable cost.

**Q: Why not trust the LLM Critic?**

A separate real pinned-Qwen experiment on 26 fictional author-labeled claims produced only **3/26** schema-valid four-way JSON outputs, none matching author labels. An improved label-only prompt gave **26/26 legal labels** but only **8/26** matches, and real token masking changed **zero** class decisions. Structural validity did not solve source reasoning. The runtime approval therefore relies on **typed deterministic source checks**, while arbitrary free-text entailment is acknowledged as unmeasured.

**Q: Which production concerns did you consider?**

Fail-closed source checks, input schemas, provenance-bound source snapshots, version/supersession handling, hard constraints, physical-versus-logical model call accounting, bounded revisions, zero-call exits, and explicit separation of untrusted drafts from accepted output. The app runs on localhost only. This project **does not** claim secure hosted deployment, real inventory data, or production user validation.

**Q: What does the evaluation actually establish?**

The strongest technical evidence is *system behavior and failure analysis*: on the original 19 positive synthetic Single-Agent cases, the expected product was retrieved in top-3 for **19/19**, but only **2/19** requests met strict author-labeled end-to-end success. A separate paired Critic/revision policy achieved **0/19** in both arms and consumed **16 additional physical Qwen calls** for revisions. That 0-vs-2 difference involves a **different acceptance gate** and should not be presented as causal agent comparison. We retained the original runs and exact raw failures.

## Five artifacts to show instead of scrolling through the whole repository

1. [README](../README.md) — concise what/why, quickstart and outcome snapshot.
2. [System architecture](SYSTEM_ARCHITECTURE.md) — real deterministic/model boundary and the fail-closed path.
3. [Final evaluation](FINAL_EVALUATION.md) — every recorded experiment, denominators, caveats and primary run links.
4. [Interactive demo implementation](../src/recsys/demo/app.py) and [walkthrough fixture](../data/demo/walkthroughs.json) — actual runnable interface and expected edge cases.
5. [Stage 4.0 real Qwen and Gradio CI run](https://github.com/Ashley96yy/multi-agent-recommendation-system/actions/runs/45063616834) — 179 passed in the optional full-runtime job, six scripted E2E cases and one real-Qwen model call.

## Integrity checklist for a resume or interview

Accurate: *Built an auditable, three-role evidence-grounded recommendation workflow; implemented Pydantic contracts, deterministic hard filters/source revision checks, one bounded revision, Gradio demonstration, and real Qwen CPU evaluation with preserved raw failure traces.*

Avoid: *"Three collaborating LLM agents", "the Critic catches all hallucinations", "proved Multi-Agent beats Single-Agent", "100% semantic groundedness", "all 6 demo outcomes generated by Qwen", or "deployed production-ready service".*

**Public demonstration data is fictional.** Any real human evaluation or production claims require a separate dataset, source validation and independent assessment.
