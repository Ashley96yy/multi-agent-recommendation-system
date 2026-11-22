"""UI contract checks: no live model downloads in Gradio construction."""

import pytest

from recsys.demo.app import view_data
from recsys.demo.service import DemoService


def test_view_data_contains_only_explicit_final_and_unverified_trace():
    service = DemoService()
    report = service.run_walkthrough("revision")
    view = view_data(report)
    assert len(view) == 6
    summary, metrics, retrieval, fields, evidence, attempts = view
    assert metrics["result_status"] == "approved"
    assert metrics["revision_used"] is True
    assert metrics["physical_backend_calls"] == 2
    assert metrics["semantic_entailment_verified"] is False
    assert "clinically proven" not in summary
    assert any("clinically proven" in a["raw_unverified_output"] for a in attempts)
    assert retrieval["catalog_sha256"] == report["retrieval"]["catalog_sha256"]
    assert evidence and fields


def test_optional_gradio_builds_real_interactive_components_without_loading_qwen(monkeypatch):
    gr = pytest.importorskip("gradio")
    import recsys.demo.service as module
    monkeypatch.setattr(module, "_pinned_qwen_cpu", lambda: (_ for _ in ()).throw(
        AssertionError("UI construction must not preload a real Qwen model")
    ))
    from recsys.demo.app import build_app
    app = build_app(DemoService())
    assert isinstance(app, gr.Blocks)
    config = app.get_config_file()
    assert len(config["components"]) >= 12
    assert len(config["dependencies"]) >= 2
