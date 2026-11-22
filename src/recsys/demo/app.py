"""Local-only Gradio UI for auditable fictional multi-agent recommendations.

The default walkthrough is PREWRITTEN, NOT a generative model. Actual pinned
Qwen is opt-in, lazy-loaded after the deterministic eligibility/source gate.
"""

import argparse

from recsys.demo.service import DemoService, preferences_from_input, render_markdown


def view_data(report: dict) -> tuple:
    """Do not turn unapproved attempt drafts into final recommendations."""
    metrics = {
        "backend": report["mode"],
        "result_status": report["status"],
        "physical_backend_calls": report["actual_backend_calls"],
        "logical_proposal_attempts": report["logical_generator_attempts"],
        "revision_used": report["revision_used"],
        "prompt_tokens": report["generated_prompt_tokens"],
        "generated_tokens": report["generated_completion_tokens"],
        "request_elapsed_seconds": report["request_elapsed_seconds"],
        "semantic_entailment_verified": False,
        "independent_human_groundedness_score": None,
        "model_output_rejected_if_unverified": True,
    }
    return (
        render_markdown(report),
        metrics,
        report["retrieval"],
        report["source_field_diagnostics"],
        report["source_evidence"],
        report["attempts"],
    )


def build_app(service: DemoService | None = None):
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            'Gradio is optional. Install the UI: python -m pip install -e ".[demo]"'
        ) from exc

    service = service if service is not None else DemoService()
    scenario_items = [(case.title, case.scenario_id) for case in service.suite.cases]
    initial = view_data(service.run_walkthrough("consistent"))

    def sections(*, defaults=None):
        defaults = defaults or (None,) * 6
        with gr.Row():
            with gr.Column(scale=3):
                summary = gr.Markdown(value=defaults[0], label="Final answer / abstention")
            with gr.Column(scale=2):
                metrics = gr.JSON(value=defaults[1], label="Calls and decision status")
        with gr.Accordion("1 · Retrieval and hard constraints", open=True):
            retrieval = gr.JSON(value=defaults[2], label="Metadata-filtered candidates and retrieval hash")
        with gr.Accordion("2 · Typed source conflict / revision / missing-field findings", open=True):
            fields = gr.JSON(value=defaults[3], label="Source-field diagnoses (fictional documents)")
        with gr.Accordion("3 · Verbatim source evidence, citations and revision status", open=False):
            evidence = gr.JSON(value=defaults[4], label="Citation ID, source ID, excerpt and supersession")
        with gr.Accordion("4 · Agent attempts and bounded revision trace — UNVERIFIED RAW TEXT", open=False):
            attempts = gr.JSON(value=defaults[5], label="Unverified drafts, Critic issues and prompt hashes")
        return [summary, metrics, retrieval, fields, evidence, attempts]

    with gr.Blocks(title="Evidence-Grounded Agent Lab · Fictional Source Demo") as app:
        gr.Markdown(
            "# Evidence-Grounded Multi-Agent Recommendation Lab\n"
            "**Retrieval → Recommendation → Deterministic Critic → at most one Revision**\n\n"
            "All products, sources and prices in this demo are **fictional**. "
            "The source guard checks explicit typed assertions, source conflicts, "
            "superseded records and citation linkage. **It cannot prove free-text "
            "model explanations are factually entailed by sources.**"
        )
        with gr.Tabs():
            with gr.Tab("Guided walkthrough · zero model download"):
                gr.Markdown(
                    "### Reproducible scripted scenarios\n"
                    "The six scenarios use **prewritten model-like outputs**, not real "
                    "Qwen inference. They exercise the exact production-style "
                    "orchestrator and source-verification control flow, including "
                    "a rejected draft and one bounded revision."
                )
                scenario = gr.Dropdown(
                    choices=scenario_items, value="consistent",
                    label="Choose a fixed fictional scenario",
                )
                description = gr.Markdown(service.scenarios["consistent"].description)
                run_scripted = gr.Button("Run deterministic walkthrough", variant="primary")
                scripted_outputs = sections(defaults=initial)
                scenario.change(
                    fn=lambda key: service.scenarios[key].description,
                    inputs=[scenario], outputs=[description],
                    queue=False,
                )
                run_scripted.click(
                    fn=lambda key: view_data(service.run_walkthrough(key)),
                    inputs=[scenario], outputs=scripted_outputs,
                )

            with gr.Tab("Real local Qwen · optional download"):
                gr.Markdown(
                    "### Real pinned Qwen 0.5B (experimental)\n"
                    "Runs the actual public pinned **Qwen2.5-0.5B-Instruct** "
                    "on your CPU only **if metadata filtering and evidence "
                    "preflight permit generation**. Model weights download at "
                    "first eligible request. The Qwen prose is **not** "
                    "independently semantically verified; a negative outcome "
                    "or rejected JSON stays rejected."
                )
                with gr.Row():
                    request = gr.Textbox(
                        label="Product request", lines=2,
                        value="Cloud Calm Facial Cream fragrance-free moisturizer",
                    )
                    category = gr.Dropdown(
                        label="Required category",
                        choices=["(any)", "moisturizer", "balm", "sunscreen", "cleanser"],
                        value="moisturizer",
                    )
                with gr.Row():
                    attributes = gr.CheckboxGroup(
                        label="Required attributes (strict source preflight)",
                        choices=["fragrance_free", "sensitive_skin", "barrier_support", "mineral"],
                        value=["fragrance_free"],
                    )
                    budget = gr.Number(
                        label="Maximum price (fictional USD; clear for no limit)",
                        value=30, precision=2,
                    )
                with gr.Row():
                    excluded_ids = gr.Textbox(
                        label="Exclude product IDs (comma separated)",
                        placeholder="SYN-E01, SYN-E02",
                    )
                    excluded_brands = gr.Textbox(
                        label="Exclude brands (comma separated)",
                        placeholder="Fictional Orbit",
                    )
                revisions = gr.Slider(
                    label="Maximum revision attempts (0 or 1)",
                    minimum=0, maximum=1, step=1, value=1,
                )
                run_real = gr.Button("Run real Qwen with source guard", variant="primary")
                live_outputs = sections()
                def handle_real(req, cat, attrs, max_price, ids, brands, rev):
                    try:
                        prefs = preferences_from_input(
                            request=req,
                            required_category=None if cat == "(any)" else cat,
                            required_attributes=attrs,
                            max_budget_usd=max_price,
                            excluded_product_ids=ids,
                            excluded_brands=brands,
                        )
                        report = service.run_live(prefs, max_revisions=int(rev))
                        return view_data(report)
                    except Exception as exc:
                        # Never echo private user input or untrusted exception
                        # payloads to the UI.
                        raise gr.Error(
                            f"Unable to execute request ({type(exc).__name__}). "
                            "Check the category, budget and optional Qwen runtime."
                        ) from None

                run_real.click(
                    fn=handle_real,
                    inputs=[
                        request, category, attributes, budget,
                        excluded_ids, excluded_brands, revisions,
                    ],
                    outputs=live_outputs,
                )
        gr.Markdown(
            "---\n"
            "**What this demo proves:** bounded workflow, reproducible "
            "hard filters, typed-source document checks, evidence citations, "
            "and transparent Critic failures. **What it does not prove:** "
            "natural-language entailment, clinical claims, real-world product "
            "truth, or independent human-calibrated accuracy."
        )
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.host != "127.0.0.1":
        parser.error("This educational demo intentionally binds only to 127.0.0.1.")
    if not 1024 <= args.port <= 65535:
        parser.error("port must be 1024–65535")
    app = build_app()
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        show_error=False,
    )


if __name__ == "__main__":
    main()
