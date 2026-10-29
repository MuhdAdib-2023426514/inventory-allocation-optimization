"""Tests for the 7-policy rolling-origin backtest, origin-wise block-bootstrap CIs, and non-negative oracle regret (`AGENTS.md` §8, §9, §14)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from inventory_di.config import load_config
from inventory_di.evaluation.metrics import EVALUATED_POLICIES


def test_backtest_artifacts_and_7_policy_parity(demo_release_config) -> None:
    cfg = load_config(demo_release_config)
    art_dir = Path(cfg.data.artifacts_dir)

    alloc_df = pd.read_parquet(art_dir / "allocation_results.parquet")
    metrics_df = pd.read_parquet(art_dir / "backtest_metrics.parquet")
    summary_df = pd.read_parquet(art_dir / "summary_metrics.parquet")
    sens_df = pd.read_parquet(art_dir / "risk_sensitivity.parquet")
    manifest = json.loads((art_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert manifest["data_mode"] == "synthetic_demo"
    assert manifest["economic_currency"] == "MU"
    assert manifest["inventory_data_provenance"] == "synthetic"
    assert manifest["bootstrap_unit"] == "forecast_origin"
    assert set(summary_df["policy"]) == set(EVALUATED_POLICIES)

    # Every origin must evaluate all 7 policies with optimal solver status and non-negative oracle regret
    for origin, grp in metrics_df.groupby("origin"):
        assert set(grp["policy"]) == set(EVALUATED_POLICIES), f"Missing policy at origin {origin}"
        assert grp["solver_status"].isin(["optimal", "heuristic_feasible"]).all()
        assert (grp["oracle_regret_mu"] >= -1e-6).all()

    assert (alloc_df["is_feasible"]).all()
    assert (alloc_df["allocated_units"] >= 0).all()
    assert len(sens_df) > 0
    assert (sens_df["model_cvar_cost_mu"] + 1e-6 >= sens_df["model_expected_cost_mu"]).all()
