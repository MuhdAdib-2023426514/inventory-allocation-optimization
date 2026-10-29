"""Tests for chronological splits, strictly pre-origin features, and leakage prevention."""

from __future__ import annotations

import numpy as np
import pytest

from inventory_di.config import load_config
from inventory_di.data.m5 import M5_PLACEMENT_MESSAGE, check_m5_files_exist
from inventory_di.data.synthetic import generate_synthetic_demo_dataset


def test_synthetic_demo_splits_and_leakage_invariants() -> None:
    cfg = load_config("configs/demo.yaml")
    weekly_df, inv_df, lineage = generate_synthetic_demo_dataset(cfg)

    assert lineage["data_mode"] == "synthetic_demo"
    assert lineage["provenance_badge"] == "DEMO / SYNTHETIC"
    assert lineage["test_demand_clipped"] is False
    assert lineage["cohort_selection_split"] == "train"
    assert lineage["cap_and_support_split"] == "train"
    assert lineage["calibration_split"] == "val"
    assert lineage["copula_fit_split"] == "train"

    # Whole-origin chronological block split checks
    train_origins = sorted(weekly_df[weekly_df["split"] == "train"]["forecast_origin"].unique())
    val_origins = sorted(weekly_df[weekly_df["split"] == "val"]["forecast_origin"].unique())
    test_origins = sorted(weekly_df[weekly_df["split"] == "test"]["forecast_origin"].unique())

    assert len(train_origins) == cfg.data.train_weeks
    assert len(val_origins) == cfg.data.val_weeks
    assert len(test_origins) == cfg.data.test_weeks
    assert max(train_origins) < min(val_origins)
    assert max(val_origins) < min(test_origins)

    # Every origin must map to exactly one split across all store-SKU pairs
    for origin, grp in weekly_df.groupby("forecast_origin"):
        assert grp["split"].nunique() == 1, f"Origin {origin} has mixed splits"

    # Feature cutoff strictly <= forecast_origin < target_start <= target_end
    assert (weekly_df["max_feature_date"] <= weekly_df["forecast_origin"]).all()
    assert (weekly_df["forecast_origin"] < weekly_df["target_start"]).all()
    assert (weekly_df["target_start"] <= weekly_df["target_end"]).all()

    # Verify lag_1w_units equals previous week's observed_units_proxy (no contemporaneous leakage)
    for (_, _), grp in weekly_df.groupby(["store_id", "sku_id"]):
        g = grp.sort_values("forecast_origin").reset_index(drop=True)
        expected_lag1 = g["observed_units_proxy"].shift(1).dropna().to_numpy()
        actual_lag1 = g["lag_1w_units"].iloc[1:].to_numpy()
        np.testing.assert_allclose(actual_lag1, expected_lag1)

    assert (inv_df["data_provenance"] == "synthetic").all()


def test_m5_missing_files_helpful_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError) as exc_info:
        check_m5_files_exist(tmp_path / "nonexistent_m5")
    assert "M5 RAW DATASET FILES NOT FOUND" in str(exc_info.value)
    assert M5_PLACEMENT_MESSAGE[:40] in str(exc_info.value)
