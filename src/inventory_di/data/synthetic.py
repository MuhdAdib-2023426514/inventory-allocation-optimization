"""Synthetic demo data generator (`data_mode="synthetic_demo"`).

Generates controlled store-SKU weekly demand with store effects, SKU base rates,
calendar seasonality, shared cross-store weekly shocks, and controlled high-demand
shortage weeks. Explicitly isolated from empirical M5 data.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from inventory_di.config import ExperimentConfig
from inventory_di.data.features import (
    compute_weekly_features_strictly_pre_origin,
    derive_training_only_inventory_config,
)
from inventory_di.data.splits import assign_chronological_splits
from inventory_di.io import validate_inventory_config_df, validate_weekly_demand_df


def generate_synthetic_demo_dataset(
    cfg: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build synthetic weekly_demand and inventory_config tables for demo mode."""
    if cfg.data_mode != "synthetic_demo":
        raise ValueError(f"generate_synthetic_demo_dataset requires data_mode='synthetic_demo', got {cfg.data_mode}")

    rng = np.random.default_rng(cfg.global_seed)
    num_stores = cfg.data.num_stores
    num_skus = cfg.data.num_skus
    total_weeks = cfg.data.num_weeks + 2  # +2 warm-up weeks for strictly lagged features

    stores = [f"DEMO_STORE_{i + 1:02d}" for i in range(num_stores)]
    skus = [f"HOUSEHOLD_DEMO_{j + 1:03d}" for j in range(num_skus)]

    # Base store multipliers and SKU base rates (chosen so typical weekly demand is 3..11 units)
    store_factors = np.linspace(1.35, 0.75, num_stores)
    sku_bases = np.linspace(5.5, 4.2, num_skus)

    start_origin = date(2025, 1, 5)
    raw_rows: list[dict[str, Any]] = []

    for w in range(total_weeks):
        origin_dt = start_origin + timedelta(days=7 * w)
        target_start_dt = origin_dt + timedelta(days=1)
        target_end_dt = origin_dt + timedelta(days=7)

        # Shared weekly shock across stores (creates cross-store rank correlation)
        shared_shock = float(rng.normal(0.0, 0.22))
        event_flag = 1.0 if (w % 6 == 4) else 0.0
        # Controlled high-demand surge every 9th week
        surge_mult = 1.35 if (w % 9 == 7) else 1.0

        season_angle = 2.0 * math.pi * (w % 52) / 52.0
        seasonal_mult = 1.0 + 0.22 * math.sin(season_angle) + 0.10 * math.cos(season_angle)

        for s_idx, store_id in enumerate(stores):
            for p_idx, sku_id in enumerate(skus):
                base_price = 4.50 + 1.25 * p_idx - 0.15 * (w // 12)
                price_noise = float(rng.uniform(-0.10, 0.10))
                price_mu = round(max(1.0, base_price + price_noise), 2)

                # Autoregressive / covariate-driven expected rate so features are genuinely informative
                promo_boost = 1.25 if event_flag > 0.5 else 1.0
                expected_rate = (
                    sku_bases[p_idx]
                    * store_factors[s_idx]
                    * seasonal_mult
                    * promo_boost
                    * surge_mult
                    * math.exp(shared_shock)
                )
                # Negative-binomial / Poisson-gamma mixture draw for realistic retail dispersion
                gamma_shape = 6.0
                gamma_scale = max(0.2, expected_rate / gamma_shape)
                lam = float(rng.gamma(gamma_shape, gamma_scale))
                units = int(rng.poisson(lam))

                raw_rows.append(
                    {
                        "forecast_origin": origin_dt.isoformat(),
                        "target_start": target_start_dt.isoformat(),
                        "target_end": target_end_dt.isoformat(),
                        "store_id": store_id,
                        "sku_id": sku_id,
                        "observed_units_proxy": max(0, units),
                        "event_flag_target_week": float(event_flag),
                        "price_at_origin_mu": float(price_mu),
                        "data_mode": "synthetic_demo",
                    }
                )

    raw_df = pd.DataFrame(raw_rows)
    feat_df = compute_weekly_features_strictly_pre_origin(raw_df)

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
        "data_mode": "synthetic_demo",
        "provenance_badge": "DEMO / SYNTHETIC",
        "num_stores": num_stores,
        "num_skus": num_skus,
        "num_origins": int(feat_df["forecast_origin"].nunique()),
        "cohort_selection_split": "train",
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
