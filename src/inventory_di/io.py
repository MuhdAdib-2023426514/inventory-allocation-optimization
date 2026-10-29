"""Data contract validators, hashing utilities, and Parquet/JSON I/O helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

WEEKLY_DEMAND_REQUIRED_COLS = {
    "forecast_origin",
    "target_start",
    "target_end",
    "store_id",
    "sku_id",
    "observed_units_proxy",
    "lag_1w_units",
    "lag_2w_units",
    "rolling_4w_mean",
    "rolling_8w_mean",
    "rolling_4w_std",
    "week_of_year_sin",
    "week_of_year_cos",
    "event_flag_target_week",
    "lag_price_mu",
    "max_feature_date",
    "split",
    "data_mode",
}

INVENTORY_CONFIG_REQUIRED_COLS = {
    "store_id",
    "sku_id",
    "opening_stock_units",
    "store_sku_unit_cap",
    "warehouse_supply_units",
    "unit_shortage_penalty_mu",
    "unit_holding_cost_mu",
    "unit_ship_cost_mu",
    "store_total_capacity_units",
    "data_provenance",
}

ALLOCATION_RESULTS_REQUIRED_COLS = {
    "run_id",
    "forecast_origin",
    "policy",
    "store_id",
    "sku_id",
    "allocated_units",
    "opening_stock_units",
    "warehouse_supply_units",
    "store_sku_unit_cap",
    "store_total_capacity_units",
    "scenario_profile",
    "data_mode",
    "is_feasible",
    "solver_status",
}

BACKTEST_METRICS_REQUIRED_COLS = {
    "run_id",
    "origin",
    "policy",
    "total_cost_mu",
    "shortage_cost_mu",
    "holding_cost_mu",
    "shipping_cost_mu",
    "filled_units",
    "unmet_units",
    "fill_rate",
    "oracle_regret_mu",
    "model_expected_cost_mu",
    "model_cvar_cost_mu",
    "solver_status",
    "runtime_seconds",
    "data_mode",
}


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    """Return current git commit hash or 'uncommitted'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "uncommitted"


def get_environment_snapshot() -> dict[str, Any]:
    """Capture Python and core package versions."""
    pkgs = {}
    for mod_name in [
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "torch",
        "pyepo",
        "pyomo",
        "highspy",
        "sklearn",
        "plotly",
        "streamlit",
        "pydantic",
        "duckdb",
    ]:
        try:
            mod = __import__(mod_name)
            pkgs[mod_name] = getattr(mod, "__version__", "installed")
        except ImportError:
            pkgs[mod_name] = None
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": pkgs,
    }


def validate_weekly_demand_df(df: pd.DataFrame) -> None:
    """Validate weekly_demand.parquet contract."""
    missing = WEEKLY_DEMAND_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"weekly_demand DataFrame missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("weekly_demand DataFrame is empty")
    if df.duplicated(subset=["forecast_origin", "store_id", "sku_id"]).any():
        raise ValueError("Duplicate (forecast_origin, store_id, sku_id) in weekly_demand")
    if (df["observed_units_proxy"] < 0).any():
        raise ValueError("Negative observed_units_proxy found in weekly_demand")
    if not set(df["split"].unique()).issubset({"train", "val", "test"}):
        raise ValueError(f"Invalid split values: {df['split'].unique()}")


def validate_inventory_config_df(df: pd.DataFrame) -> None:
    """Validate inventory_config.parquet contract."""
    missing = INVENTORY_CONFIG_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"inventory_config DataFrame missing columns: {sorted(missing)}")
    if df.duplicated(subset=["store_id", "sku_id"]).any():
        raise ValueError("Duplicate (store_id, sku_id) in inventory_config")
    if (df["data_provenance"] != "synthetic").any():
        raise ValueError("inventory_config data_provenance must be 'synthetic'")
    # Check warehouse_supply_units is consistent per SKU
    for sku_id, grp in df.groupby("sku_id"):
        if grp["warehouse_supply_units"].nunique() != 1:
            raise ValueError(f"SKU {sku_id} has inconsistent warehouse_supply_units across stores")


def validate_allocation_results_df(df: pd.DataFrame) -> None:
    """Validate allocation_results.parquet contract."""
    missing = ALLOCATION_RESULTS_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"allocation_results DataFrame missing columns: {sorted(missing)}")
    if (df["allocated_units"] < 0).any():
        raise ValueError("Negative allocated_units found in allocation_results")


def validate_backtest_metrics_df(df: pd.DataFrame) -> None:
    """Validate backtest_metrics.parquet contract."""
    missing = BACKTEST_METRICS_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"backtest_metrics DataFrame missing columns: {sorted(missing)}")


def save_parquet(df: pd.DataFrame, path: str | Path) -> str:
    """Save DataFrame to Parquet and return SHA-256 digest."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return sha256_file(out_path)


def save_json(data: dict[str, Any], path: str | Path) -> str:
    """Save dictionary as formatted JSON and return SHA-256 digest."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")
    return sha256_file(out_path)
