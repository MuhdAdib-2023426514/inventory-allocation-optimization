"""Data contract validators, hashing utilities, and Parquet/JSON I/O helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
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
    _units(df, ["observed_units_proxy"])
    if df[list(WEEKLY_DEMAND_REQUIRED_COLS)].isna().any().any():
        raise ValueError("Null values in weekly demand contract")
    for col in ["forecast_origin", "target_start", "target_end", "max_feature_date"]:
        pd.to_datetime(df[col], errors="raise")
    if not ((df.max_feature_date <= df.forecast_origin) & (df.forecast_origin < df.target_start)
            & (df.target_start <= df.target_end)).all():
        raise ValueError("Invalid feature/target temporal ordering")
    if not ((pd.to_datetime(df.target_end) - pd.to_datetime(df.target_start)).dt.days == 6).all():
        raise ValueError("Targets must span seven calendar days")
    if (df.groupby("forecast_origin").split.nunique() != 1).any():
        raise ValueError("Mixed splits at same origin")
    split_ranges = [df.loc[df.split == split, "forecast_origin"] for split in ("train", "val", "test")]
    if any(v.empty for v in split_ranges) or any(a.max() >= b.min() for a, b in zip(split_ranges, split_ranges[1:])):
        raise ValueError("Splits must be complete chronological blocks")
    expected = df.store_id.nunique() * df.sku_id.nunique()
    if (df.groupby("forecast_origin").size() != expected).any():
        raise ValueError("Incomplete store-SKU cohort")
    numeric = df.select_dtypes(include="number")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Nonfinite weekly features")


def validate_inventory_config_df(df: pd.DataFrame) -> None:
    """Validate inventory_config.parquet contract."""
    missing = INVENTORY_CONFIG_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"inventory_config DataFrame missing columns: {sorted(missing)}")
    if df.duplicated(subset=["store_id", "sku_id"]).any():
        raise ValueError("Duplicate (store_id, sku_id) in inventory_config")
    if (df["data_provenance"] != "synthetic").any():
        raise ValueError("inventory_config data_provenance must be 'synthetic'")
    _units(df, ["opening_stock_units", "store_sku_unit_cap", "warehouse_supply_units", "store_total_capacity_units"])
    costs = df[["unit_shortage_penalty_mu", "unit_holding_cost_mu", "unit_ship_cost_mu"]].to_numpy()
    if df.empty or not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("Invalid inventory economics")
    if len(df) != df.store_id.nunique() * df.sku_id.nunique():
        raise ValueError("Incomplete inventory cohort")
    for _, group in df.groupby("store_id"):
        if group.store_total_capacity_units.nunique() != 1 or group.opening_stock_units.sum() > group.store_total_capacity_units.iloc[0]:
            raise ValueError("Inconsistent capacity or opening stock exceeds store capacity")
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
    _units(df, ["allocated_units"])
    if df.duplicated(["run_id", "forecast_origin", "policy", "store_id", "sku_id"]).any():
        raise ValueError("Duplicate allocation row")
    if not df.is_feasible.all() or not df.solver_status.isin(["optimal", "heuristic_feasible"]).all():
        raise ValueError("Rejected nonoptimal or infeasible scientific allocation")


def validate_backtest_metrics_df(df: pd.DataFrame) -> None:
    """Validate backtest_metrics.parquet contract."""
    missing = BACKTEST_METRICS_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"backtest_metrics DataFrame missing columns: {sorted(missing)}")
    if df.empty or df.duplicated(["run_id", "origin", "policy"]).any():
        raise ValueError("Empty or duplicate policy/origin metrics")
    if not np.isfinite(df.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Nonfinite backtest metrics")
    if not df.solver_status.isin(["optimal", "heuristic_feasible"]).all() or (df.oracle_regret_mu < -1e-6).any():
        raise ValueError("Nonoptimal result or negative hindsight regret")


def _units(df, columns):
    values = df[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.rint(values)).any():
        raise ValueError(f"Expected finite nonnegative integer quantities: {columns}")


def source_provenance() -> dict[str, Any]:
    """Fingerprint scientific source and record the actual working tree state."""
    package = Path(__file__).resolve().parent
    checkout = package.parents[1]
    root = checkout if (checkout / "src").is_dir() else package
    if (root / "src").is_dir():
        hashes = {str(p.relative_to(root)): sha256_file(p) for p in sorted((root / "src").rglob("*.py"))}
    else:
        hashes = {"src/inventory_di/" + str(p.relative_to(package)): sha256_file(p) for p in sorted(package.rglob("*.py"))}

    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False)
    return {"git_commit": get_git_commit(), "worktree_status": dirty.stdout,
            "source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "source_files": hashes}


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
