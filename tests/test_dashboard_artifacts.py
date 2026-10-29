"""Tests for Streamlit dashboard artifact readiness and QA bundle compliance (`AGENTS.md` §10, §14)."""

from __future__ import annotations

import json
from pathlib import Path

from app.streamlit_app import REGISTERED_PAGES, load_cached_bundle

from inventory_di.config import load_config


def test_dashboard_cached_bundle_and_pages() -> None:
    assert len(REGISTERED_PAGES) == 5
    bundle = load_cached_bundle("configs/demo.yaml")
    assert bundle["status"] == "OK"
    assert bundle["cfg"].data_mode == "synthetic_demo"

    cfg = load_config("configs/demo.yaml")
    qa_path = Path(cfg.data.artifacts_dir) / "dashboard_qa_bundle.json"
    assert qa_path.exists()
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    assert qa["provenance_badge"] == "DEMO / SYNTHETIC"
    assert qa["page_d_risk_controls"]["retrain_in_ui"] is False
    assert qa["offline_ready"] is True
    assert qa["missing_artifact_guidance_present"] is True

    # Missing M5 data should return status == "MISSING" gracefully
    m5_bundle = load_cached_bundle("configs/m5.yaml")
    if not Path("data/raw/m5/calendar.csv").exists():
        assert m5_bundle["status"] == "MISSING"
        assert len(m5_bundle["missing_files"]) > 0
