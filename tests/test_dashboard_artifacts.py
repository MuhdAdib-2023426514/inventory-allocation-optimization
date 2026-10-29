"""Tests for Streamlit dashboard artifact readiness and QA bundle compliance (`AGENTS.md` §10, §14)."""

from __future__ import annotations

import json
from pathlib import Path

from app.streamlit_app import REGISTERED_PAGES, load_cached_bundle

from inventory_di.config import load_config
from inventory_di.evaluation.backtest import build_allocation_instance_from_config_df


def test_dashboard_cached_bundle_and_pages(demo_release_config) -> None:
    assert len(REGISTERED_PAGES) == 5
    bundle = load_cached_bundle(demo_release_config)
    assert bundle["status"] == "OK"
    assert bundle["cfg"].data_mode == "synthetic_demo"

    cfg = load_config(demo_release_config)
    qa_path = Path(cfg.data.artifacts_dir) / "dashboard_qa_bundle.json"
    assert qa_path.exists()
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    assert qa["provenance_badge"] == "DEMO / SYNTHETIC"
    assert qa["page_d_risk_controls"]["retrain_in_ui"] is False
    assert qa["offline_ready"] is True
    assert qa["missing_artifact_guidance_present"] is True

    # Verify What-If Simulation overrides (opening stock & shortage penalty multiplier)
    inv_df = bundle["inventory_config"]
    inst_override = build_allocation_instance_from_config_df(
        inv_df,
        supply_multiplier=0.8,
        opening_stock_override=3,
        shortage_penalty_multiplier=1.5,
    )
    assert (inst_override.I == 3).all()
    assert inst_override.b[0, 0] == 9.0

    # Verify M5 bundle when M5 artifacts exist
    m5_bundle = load_cached_bundle("configs/m5.yaml")
    if Path("artifacts/m5_v2/run_manifest.json").exists():
        assert m5_bundle["status"] == "OK"
        assert m5_bundle["cfg"].data_mode == "m5_sales_proxy"


def test_dashboard_rejects_changed_artifact_and_invalidates_cache(demo_release_config):
    cfg = load_config(demo_release_config)
    path = Path(cfg.data.artifacts_dir) / "summary_metrics.parquet"
    original = path.read_bytes()
    assert load_cached_bundle(demo_release_config)["status"] == "OK"
    try:
        path.write_bytes(original + b"corrupted")
        assert load_cached_bundle(demo_release_config)["status"] == "INVALID"
    finally:
        path.write_bytes(original)
    assert load_cached_bundle(demo_release_config)["status"] == "OK"
