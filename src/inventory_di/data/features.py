"""Strictly pre-origin feature computation and cohort parameter estimation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "lag_1w_units",
    "lag_2w_units",
    "rolling_4w_mean",
    "rolling_8w_mean",
    "rolling_4w_std",
    "week_of_year_sin",
    "week_of_year_cos",
    "event_flag_target_week",
    "lag_price_mu",
]


def compute_weekly_features_strictly_pre_origin(raw_weekly: pd.DataFrame) -> pd.DataFrame:
    """Compute strictly pre-origin rolling features per (store_id, sku_id).

    Input `raw_weekly` must have columns:
    - forecast_origin (YYYY-MM-DD, end date of historical information window)
    - target_start (YYYY-MM-DD, strictly > forecast_origin)
    - target_end (YYYY-MM-DD, >= target_start)
    - store_id, sku_id
    - observed_units_proxy (int >= 0, realized units during [target_start, target_end])
    - event_flag_target_week (float 0/1 known-in-advance calendar event indicator)
    - price_at_origin_mu (float > 0, historical/scheduled price available at forecast_origin)

    Note on temporal causality:
    At `forecast_origin` t_w (which precedes target week w), the most recently completed
    7-day target window is week w-1 (`target_end` <= `forecast_origin`). Thus `lag_1w_units`
    is `observed_units_proxy` shifted by 1 week, `lag_2w_units` is shifted by 2 weeks, and
    rolling statistics are computed on shifted series (`shift(1)`).
    """
    df = raw_weekly.sort_values(["store_id", "sku_id", "forecast_origin"]).copy()

    # Verify target_start > forecast_origin for every row
    if not (df["target_start"] > df["forecast_origin"]).all():
        raise ValueError("Found rows where target_start <= forecast_origin (temporal leakage!)")

    records: list[pd.DataFrame] = []
    for (_, _), grp in df.groupby(["store_id", "sku_id"], sort=False):
        g = grp.sort_values("forecast_origin").copy()
        s_shifted = g["observed_units_proxy"].shift(1)
        g["lag_1w_units"] = s_shifted.fillna(0.0).astype(float)
        g["lag_2w_units"] = g["observed_units_proxy"].shift(2).fillna(g["lag_1w_units"]).astype(float)
        g["rolling_4w_mean"] = s_shifted.rolling(4, min_periods=1).mean().fillna(0.0).astype(float)
        g["rolling_8w_mean"] = s_shifted.rolling(8, min_periods=1).mean().fillna(0.0).astype(float)
        g["rolling_4w_std"] = s_shifted.rolling(4, min_periods=2).std().fillna(1.0).astype(float)

        # Calendar seasonality known at forecast_origin
        dt = pd.to_datetime(g["target_start"])
        iso_week = dt.dt.isocalendar().week.astype(float)
        g["week_of_year_sin"] = np.sin(2.0 * math.pi * iso_week / 52.0)
        g["week_of_year_cos"] = np.cos(2.0 * math.pi * iso_week / 52.0)

        # Price known at or before forecast_origin (lagged by 1 week if not scheduled)
        g["lag_price_mu"] = g["price_at_origin_mu"].shift(1).fillna(1.0).astype(float)
        # Max feature date is strictly <= forecast_origin
        g["max_feature_date"] = g["forecast_origin"]
        records.append(g)

    out = pd.concat(records, ignore_index=True)
    # Drop the initial warm-up row per store-SKU where lag_1w had no prior history if desired,
    # or retain rows after warm-up >= 2 weeks
    valid_origins = sorted(out["forecast_origin"].unique())[2:]
    out = out[out["forecast_origin"].isin(valid_origins)].reset_index(drop=True)
    return out


def derive_training_only_inventory_config(
    weekly_df: pd.DataFrame,
    opening_stock_units: int,
    default_store_sku_cap: int,
    warehouse_supply_ratio: float,
    store_total_capacity_units: int,
    unit_shortage_penalty_mu: float,
    unit_holding_cost_mu: float,
    unit_ship_cost_mu: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Estimate store-SKU caps, warehouse supply, and PMF support H strictly on split == 'train'."""
    train_df = weekly_df[weekly_df["split"] == "train"]
    if train_df.empty:
        raise ValueError("Cannot derive inventory config: train split is empty")

    stores = sorted(train_df["store_id"].unique())
    skus = sorted(train_df["sku_id"].unique())

    # Compute per-SKU mean weekly total demand on training split only
    sku_supplies: dict[str, int] = {}
    store_sku_caps: dict[tuple[str, str], int] = {}
    for sku in skus:
        sku_train = train_df[train_df["sku_id"] == sku]
        weekly_totals = sku_train.groupby("forecast_origin")["observed_units_proxy"].sum()
        mean_total = float(weekly_totals.mean())
        # Scarcity regime: warehouse supply is ~warehouse_supply_ratio of mean total weekly demand
        w_p = max(len(stores), int(round(mean_total * warehouse_supply_ratio)))
        sku_supplies[sku] = min(w_p, len(stores) * default_store_sku_cap)

        for store in stores:
            cell = sku_train[sku_train["store_id"] == store]["observed_units_proxy"]
            q90 = float(np.percentile(cell, 90)) if len(cell) > 0 else float(default_store_sku_cap)
            cap = min(default_store_sku_cap, max(4, int(math.ceil(q90 * 1.15))))
            store_sku_caps[(store, sku)] = cap

    # Ensure warehouse supply does not exceed sum of store caps
    for sku in skus:
        sum_caps = sum(store_sku_caps[(s, sku)] for s in stores)
        sku_supplies[sku] = min(sku_supplies[sku], max(1, int(round(0.8 * sum_caps))))

    # Ensure store_total_capacity_units is binding or meaningful relative to sum of caps
    rows = []
    for store in stores:
        for sku in skus:
            # Slight store-specific cost variation (deterministic from store index) for realistic tie-breaking
            s_idx = stores.index(store)
            p_idx = skus.index(sku)
            b_sp = round(unit_shortage_penalty_mu + 0.25 * (p_idx % 3) + 0.1 * (s_idx % 3), 4)
            h_sp = round(unit_holding_cost_mu + 0.05 * (p_idx % 2), 4)
            t_sp = round(unit_ship_cost_mu + 0.05 * s_idx, 4)
            rows.append(
                {
                    "store_id": store,
                    "sku_id": sku,
                    "opening_stock_units": int(opening_stock_units),
                    "store_sku_unit_cap": int(store_sku_caps[(store, sku)]),
                    "warehouse_supply_units": int(sku_supplies[sku]),
                    "unit_shortage_penalty_mu": float(b_sp),
                    "unit_holding_cost_mu": float(h_sp),
                    "unit_ship_cost_mu": float(t_sp),
                    "store_total_capacity_units": int(store_total_capacity_units),
                    "data_provenance": "synthetic",
                }
            )

    inv_df = pd.DataFrame(rows)
    # Choose H from training split 95th percentile bounded by config max_demand_support_h
    train_q95 = int(math.ceil(float(np.percentile(train_df["observed_units_proxy"], 95))))
    max_cap = int(inv_df["store_sku_unit_cap"].max()) + int(inv_df["opening_stock_units"].max())
    recommended_h = max(train_q95, max_cap + 2)
    tail_values = train_df[train_df["observed_units_proxy"] > recommended_h]["observed_units_proxy"]
    tail_mean_excess = (
        float((tail_values - recommended_h).mean()) if len(tail_values) > 0 else 2.0
    )

    stats = {
        "fitted_on_split": "train",
        "recommended_support_h": int(recommended_h),
        "tail_mean_excess_above_h": float(max(1.0, tail_mean_excess)),
        "train_overflow_rate": float((train_df["observed_units_proxy"] > recommended_h).mean()),
    }
    return inv_df, stats
