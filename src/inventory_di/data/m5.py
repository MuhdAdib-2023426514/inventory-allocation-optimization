"""Walmart M5 dataset ingestion (`data_mode="m5_sales_proxy"`).

Reads user-provided M5 CSV files from `data/raw/m5/`, filters `cat_id == "HOUSEHOLD"`,
deterministically selects stores and SKUs using ONLY the training window, aggregates
daily sales to aligned 7-day weekly demand targets, joins calendar and lagged prices
without temporal leakage, and outputs the normalized `weekly_demand` and
`inventory_config` tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from inventory_di.config import ExperimentConfig
from inventory_di.data.features import (
    compute_weekly_features_strictly_pre_origin,
    derive_training_only_inventory_config,
)
from inventory_di.data.splits import assign_chronological_splits
from inventory_di.io import sha256_file, validate_inventory_config_df, validate_weekly_demand_df

M5_PLACEMENT_MESSAGE = """
================================================================================
M5 RAW DATASET FILES NOT FOUND IN `data/raw/m5/`
================================================================================
Per AGENTS.md §1.10 & §4, third-party competition data cannot be committed or
downloaded with hard-coded tokens. To run `mode=m5` (`data_mode="m5_sales_proxy"`):

1. Accept the competition rules at:
   https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
2. Download and place the following files into `data/raw/m5/`:
   - `calendar.csv`
   - `sell_prices.csv`
   - `sales_train_evaluation.csv` (or `sales_train_validation.csv`)
3. Re-run:
   python -m inventory_di.cli prepare --config configs/m5.yaml

To run the fully verified synthetic smoke-test pipeline (`DEMO / SYNTHETIC`):
   python -m inventory_di.cli make-demo --config configs/demo.yaml
================================================================================
""".strip()


def check_m5_files_exist(raw_dir: str | Path) -> dict[str, Path]:
    """Verify M5 raw files exist and return resolved paths, or raise FileNotFoundError."""
    base = Path(raw_dir)
    cal_path = base / "calendar.csv"
    price_path = base / "sell_prices.csv"
    eval_path = base / "sales_train_evaluation.csv"
    val_path = base / "sales_train_validation.csv"

    sales_path = eval_path if eval_path.exists() else (val_path if val_path.exists() else None)
    if not cal_path.exists() or not price_path.exists() or sales_path is None:
        raise FileNotFoundError(M5_PLACEMENT_MESSAGE)

    return {
        "calendar": cal_path,
        "sell_prices": price_path,
        "sales": sales_path,
    }


def ingest_m5_dataset(
    cfg: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Ingest M5 files, filter HOUSEHOLD cohort on training window only, and build weekly contracts."""
    if cfg.data_mode != "m5_sales_proxy":
        raise ValueError(f"ingest_m5_dataset requires data_mode='m5_sales_proxy', got {cfg.data_mode}")

    paths = check_m5_files_exist(cfg.data.raw_m5_dir)
    raw_hashes = {k: sha256_file(p) for k, p in paths.items()}

    cal_df = pd.read_csv(paths["calendar"])
    sales_header = pd.read_csv(paths["sales"], nrows=5)
    d_cols = [c for c in sales_header.columns if c.startswith("d_")]

    # Limit to trailing (num_weeks + 2) * 7 days to avoid melting all ~59M rows in quick/standard runs
    needed_days = (cfg.data.num_weeks + 2) * 7
    if len(d_cols) < needed_days:
        needed_days = (len(d_cols) // 7) * 7
    selected_d_cols = d_cols[-needed_days:]

    # Determine training day columns strictly before val + test weeks
    holdout_days = (cfg.data.val_weeks + cfg.data.test_weeks) * 7
    train_d_cols = selected_d_cols[:-holdout_days]
    if len(train_d_cols) < 28:
        raise ValueError("Insufficient training days in M5 sales file for cohort selection")

    usecols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", *selected_d_cols]
    sales_df = pd.read_csv(paths["sales"], usecols=usecols)
    hh_df = sales_df[sales_df["cat_id"] == "HOUSEHOLD"].copy()
    if hh_df.empty:
        raise ValueError("No rows with cat_id == 'HOUSEHOLD' found in M5 sales file")

    # Select stores and SKUs deterministically using ONLY `train_d_cols`
    train_vals = hh_df[train_d_cols].to_numpy(dtype=float)
    hh_df["_train_mean_daily"] = train_vals.mean(axis=1)
    hh_df["_train_nonzero_rate"] = (train_vals > 0).mean(axis=1)

    # Pick stores with highest regular HOUSEHOLD coverage in training window
    store_scores = (
        hh_df.groupby("store_id")["_train_nonzero_rate"]
        .mean()
        .sort_values(ascending=False)
    )
    chosen_stores = sorted(store_scores.head(cfg.data.num_stores).index.tolist())
    hh_stores = hh_df[hh_df["store_id"].isin(chosen_stores)].copy()

    # Select SKUs deterministically on `train_d_cols` according to `cfg.data.demand_regime` (P2.1)
    sku_stats = hh_stores.groupby("item_id").agg(
        mean_daily=("_train_mean_daily", "mean"),
        nonzero_rate=("_train_nonzero_rate", "mean"),
        store_count=("store_id", "nunique"),
    )
    sku_stats["adi"] = 1.0 / sku_stats["nonzero_rate"].clip(lower=1e-4)

    regime_mode = getattr(cfg.data, "demand_regime", "regular")
    full_coverage = sku_stats[sku_stats["store_count"] == len(chosen_stores)].copy()

    reg_pool = full_coverage[
        (full_coverage["mean_daily"] >= 0.25)
        & (full_coverage["mean_daily"] <= 2.2)
        & (full_coverage["nonzero_rate"] >= 0.35)
    ].sort_values(by=["nonzero_rate", "mean_daily"], ascending=[False, True])

    int_pool = full_coverage[
        (full_coverage["mean_daily"] >= 0.08)
        & (full_coverage["mean_daily"] <= 0.65)
        & (full_coverage["nonzero_rate"] >= 0.08)
        & (full_coverage["nonzero_rate"] <= 0.28)
    ].sort_values(by=["adi", "mean_daily"], ascending=[False, False])

    if regime_mode == "intermittent":
        eligible_skus = int_pool if len(int_pool) >= cfg.data.num_skus else full_coverage.sort_values(
            by="nonzero_rate", ascending=True
        )
        chosen_skus = sorted(eligible_skus.head(cfg.data.num_skus).index.tolist())
    elif regime_mode == "mixed":
        n_reg = cfg.data.num_skus // 2
        n_int = cfg.data.num_skus - n_reg
        chosen_skus = sorted(
            reg_pool.head(n_reg).index.tolist() + int_pool.head(n_int).index.tolist()
        )
    else:
        eligible_skus = reg_pool if len(reg_pool) >= cfg.data.num_skus else full_coverage.sort_values(
            by="nonzero_rate", ascending=False
        )
        chosen_skus = sorted(eligible_skus.head(cfg.data.num_skus).index.tolist())

    sku_regime_metadata = {
        str(sku): {
            "regime": "intermittent" if float(sku_stats.loc[sku, "adi"]) >= 3.0 else "regular",
            "train_mean_daily": round(float(sku_stats.loc[sku, "mean_daily"]), 4),
            "train_nonzero_rate": round(float(sku_stats.loc[sku, "nonzero_rate"]), 4),
            "train_adi": round(float(sku_stats.loc[sku, "adi"]), 3),
        }
        for sku in chosen_skus
    }

    cohort_df = hh_stores[hh_stores["item_id"].isin(chosen_skus)].copy()
    melted = cohort_df.melt(
        id_vars=["item_id", "store_id"],
        value_vars=selected_d_cols,
        var_name="d",
        value_name="daily_units",
    )

    cal_sub = cal_df[["d", "date", "wm_yr_wk", "event_name_1"]].copy()
    cal_sub["has_event"] = cal_sub["event_name_1"].notna().astype(float)
    merged = melted.merge(cal_sub, on="d", how="left", validate="many_to_one")

    if merged["date"].isna().any() or len(merged) != len(melted):
        raise ValueError("Calendar join lost coverage or changed row count")

    prices_df = pd.read_csv(paths["sell_prices"])
    prices_sub = prices_df[
        prices_df["store_id"].isin(chosen_stores) & prices_df["item_id"].isin(chosen_skus)
    ][["store_id", "item_id", "wm_yr_wk", "sell_price"]].copy()
    merged = merged.merge(prices_sub, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="many_to_one")
    if len(merged) != len(melted):
        raise ValueError("Price join changed row count")
    merged = merged.sort_values(["store_id", "item_id", "date"])
    merged["sell_price"] = merged.groupby(["store_id", "item_id"])["sell_price"].ffill().fillna(1.0)

    # Map selected_d_cols into aligned 7-day blocks
    d_order = {d_name: idx for idx, d_name in enumerate(selected_d_cols)}
    merged["day_idx"] = merged["d"].map(d_order)
    merged["week_block"] = merged["day_idx"] // 7

    # Aggregate to 7-day weekly target per (week_block, store_id, item_id)
    weekly_agg = (
        merged.groupby(["week_block", "store_id", "item_id"], as_index=False)
        .agg(
            target_start=("date", "min"),
            target_end=("date", "max"),
            observed_units_proxy=("daily_units", "sum"),
            event_flag_target_week=("has_event", "max"),
            price_at_origin_mu=("sell_price", "first"),
        )
        .rename(columns={"item_id": "sku_id"})
    )
    # Forecast origin is strictly 1 calendar day before target_start
    weekly_agg["forecast_origin"] = (
        pd.to_datetime(weekly_agg["target_start"]) - pd.Timedelta(days=1)
    ).dt.strftime("%Y-%m-%d")
    weekly_agg["observed_units_proxy"] = weekly_agg["observed_units_proxy"].round().astype(int)
    weekly_agg["data_mode"] = "m5_sales_proxy"

    feat_df = compute_weekly_features_strictly_pre_origin(weekly_agg)
    split_map = assign_chronological_splits(
        feat_df["forecast_origin"],
        train_weeks=cfg.data.train_weeks,
        val_weeks=cfg.data.val_weeks,
        test_weeks=cfg.data.test_weeks,
    )
    feat_df["split"] = feat_df["forecast_origin"].map(split_map)

    inv_df, support_stats = derive_training_only_inventory_config(
        weekly_df=feat_df,
        opening_stock_units=cfg.economics.opening_stock_units,
        default_store_sku_cap=cfg.economics.store_sku_unit_cap,
        warehouse_supply_ratio=cfg.economics.warehouse_supply_ratio,
        store_total_capacity_units=cfg.economics.store_total_capacity_units,
        unit_shortage_penalty_mu=cfg.economics.unit_shortage_penalty_mu,
        unit_holding_cost_mu=cfg.economics.unit_holding_cost_mu,
        unit_ship_cost_mu=cfg.economics.unit_ship_cost_mu,
    )

    validate_weekly_demand_df(feat_df)
    validate_inventory_config_df(inv_df)

    train_max_origin = str(feat_df[feat_df["split"] == "train"]["forecast_origin"].max())
    records_audit = [
        {
            "forecast_origin": str(r.forecast_origin),
            "target_start": str(r.target_start),
            "target_end": str(r.target_end),
            "feature_max_date": str(r.max_feature_date),
            "price_max_date": str(r.max_feature_date),
            "split": str(r.split),
            "store_id": str(r.store_id),
            "sku_id": str(r.sku_id),
        }
        for r in feat_df.itertuples(index=False)
    ]

    lineage = {
        "data_mode": "m5_sales_proxy",
        "provenance_badge": "M5 sales proxy + simulated inventory",
        "m5_raw_file_hashes": raw_hashes,
        "num_stores": len(chosen_stores),
        "num_skus": len(chosen_skus),
        "chosen_stores": chosen_stores,
        "chosen_skus": chosen_skus,
        "demand_regime": getattr(cfg.data, "demand_regime", "regular"),
        "sku_regime_metadata": sku_regime_metadata,
        "cohort_selection_split": "train",
        "price_policy": "Per-series chronological forward-fill; fixed 1.0 fallback; previous-week feature only",
        "adi_frequency": "daily, inverse of mean store nonzero-sales rate",
        "cohort_filters": {"regular_mean_daily": [0.25, 2.2], "regular_nonzero_rate_min": 0.35,
                           "intermittent_mean_daily": [0.08, 0.65], "intermittent_nonzero_rate": [0.08, 0.28]},
        "cap_and_support_split": "train",
        "calibration_split": "val",
        "copula_fit_split": "train",
        "test_demand_clipped": False,
        "proportional_Lookback_uses_future": False,
        "oracle_used_in_policy_features": False,
        "support_stats": support_stats,
        "records": records_audit,
        "cohort_selection_end_date": train_max_origin,
        "H_estimation_end_date": train_max_origin,
        "copula_fit_end_date": train_max_origin,
        "calibration_fit_splits": ["val"],
        "proportional_baseline_max_lag_date_offset": 0,
        "realized_test_demand_clipped": False,
    }
    return feat_df, inv_df, lineage
