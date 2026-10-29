"""Configuration loading and validation for inventory decision intelligence."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    num_stores: int = Field(ge=1)
    num_skus: int = Field(ge=1)
    num_weeks: int = Field(ge=8)
    train_weeks: int = Field(ge=4)
    val_weeks: int = Field(ge=2)
    test_weeks: int = Field(ge=2)
    demand_regime: Literal["regular", "intermittent", "mixed"] = "regular"
    raw_m5_dir: str = "data/raw/m5"
    processed_dir: str = "data/demo"
    artifacts_dir: str = "artifacts/demo"
    reports_dir: str = "reports/demo"

    @model_validator(mode="after")
    def check_split_sum(self) -> DataConfig:
        total = self.train_weeks + self.val_weeks + self.test_weeks
        if total > self.num_weeks:
            raise ValueError(
                f"train_weeks ({self.train_weeks}) + val_weeks ({self.val_weeks}) + "
                f"test_weeks ({self.test_weeks}) = {total} > num_weeks ({self.num_weeks})"
            )
        return self


class EconomicsConfig(BaseModel):
    currency: Literal["MU"] = "MU"
    opening_stock_units: int = Field(default=0, ge=0)
    store_sku_unit_cap: int = Field(default=12, ge=1)
    warehouse_supply_ratio: float = Field(default=0.78, gt=0.0, le=2.0)
    store_total_capacity_units: int = Field(default=22, ge=1)
    unit_shortage_penalty_mu: float = Field(default=6.0, ge=0.0)
    unit_holding_cost_mu: float = Field(default=1.0, ge=0.0)
    unit_ship_cost_mu: float = Field(default=0.25, ge=0.0)


class ModelConfig(BaseModel):
    max_demand_support_h: int = Field(default=16, ge=2)
    hidden_dim: int = Field(default=32, ge=4)
    learning_rate: float = Field(default=0.015, gt=0.0)
    epochs: int = Field(default=35, ge=1)
    batch_size: int = Field(default=8, ge=1)
    lambda_dfl: float = Field(default=1.0, ge=0.0)
    lambda_dfl_grid: list[float] = Field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0])
    spoplus_processes: int = Field(default=1, ge=1)


class DROConfig(BaseModel):
    num_scenarios: int = Field(default=40, ge=2)
    alpha: float = Field(default=0.90, ge=0.0, lt=1.0)
    risk_weight: float = Field(default=0.50, ge=0.0, le=1.0)
    alpha_grid: list[float] = Field(default_factory=lambda: [0.0, 0.50, 0.75, 0.90, 0.95])
    risk_weight_grid: list[float] = Field(default_factory=lambda: [0.0, 0.25, 0.50, 0.75, 1.0])
    supply_stress_multipliers: list[float] = Field(default_factory=lambda: [0.6, 0.8, 1.0, 1.2])


class EvaluationConfig(BaseModel):
    trailing_weeks_proportional: int = Field(default=8, ge=1)
    bootstrap_resamples: int = Field(default=200, ge=10)
    bootstrap_block_size: int = Field(default=2, ge=1)
    seeds: list[int] = Field(default_factory=lambda: [42, 101, 202])
    independent_training_seeds: bool = True
    lead_time_weeks: int = Field(default=0, ge=0)
    carryover_decay: float = Field(default=1.0, ge=0.0, le=1.0)


class ExperimentConfig(BaseModel):
    mode: Literal["demo", "m5"]
    data_mode: Literal["synthetic_demo", "m5_sales_proxy"]
    global_seed: int = 42
    run_id: str = "demo_v1"
    data: DataConfig
    economics: EconomicsConfig
    model: ModelConfig
    dro: DROConfig
    evaluation: EvaluationConfig
    profile: Literal["standard", "quick"] = "standard"
    config_path: str = ""
    config_sha256: str = ""

    @model_validator(mode="after")
    def check_mode_consistency(self) -> ExperimentConfig:
        if self.mode == "demo" and self.data_mode != "synthetic_demo":
            raise ValueError("mode='demo' requires data_mode='synthetic_demo'")
        if self.mode == "m5" and self.data_mode != "m5_sales_proxy":
            raise ValueError("mode='m5' requires data_mode='m5_sales_proxy'")
        return self


def load_config(path: str | Path, quick: bool = False) -> ExperimentConfig:
    """Load and validate an ExperimentConfig from YAML."""
    cfg_path = Path(path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    raw_bytes = cfg_path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    payload = yaml.safe_load(raw_bytes.decode("utf-8"))
    cfg = ExperimentConfig.model_validate(payload)
    cfg.config_path = str(cfg_path)
    cfg.config_sha256 = sha256
    if quick:
        cfg.profile = "quick"
        cfg.run_id += "_quick"
        for field in ("processed_dir", "artifacts_dir", "reports_dir"):
            setattr(cfg.data, field, getattr(cfg.data, field) + "_quick")
        cfg.model.epochs = min(cfg.model.epochs, 12)
        cfg.dro.num_scenarios = min(cfg.dro.num_scenarios, 25)
        cfg.evaluation.bootstrap_resamples = min(cfg.evaluation.bootstrap_resamples, 50)
    return cfg
