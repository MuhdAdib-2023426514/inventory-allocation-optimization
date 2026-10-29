# Reproducibility & Contract Schema Reference (`experiment-reproducibility`)

Aligned with [`AGENTS.md`](../../../AGENTS.md) §4, §8, §9, §14, and §15.

## 1.Normalized Parquet Table Contracts (`AGENTS.md` §4)

1. **`weekly_demand.parquet`**:
   - Required columns: `forecast_origin`, `target_start`, `target_end`, `store_id`, `sku_id`, `observed_units_proxy`, pre-origin feature columns, `split`.
   - Invariants: `observed_units_proxy` integer $\ge 0$; unique key `(forecast_origin, store_id, sku_id)`.
2. **`inventory_config.parquet`**:
   - Required columns: `store_id`, `sku_id`, `opening_stock_units`, `store_sku_unit_cap`, `warehouse_supply_units`, `unit_shortage_penalty_mu`, `unit_holding_cost_mu`, `unit_ship_cost_mu`, `store_total_capacity_units`, `data_provenance`.
   - Invariants: `data_provenance == "synthetic"`; `warehouse_supply_units` unique per `sku_id` (no double counting across stores).
3. **`allocation_results.parquet`**:
   - Required columns: `run_id`, `forecast_origin`, `policy`, `store_id`, `sku_id`, `allocated_units`, `opening_stock_units`, `warehouse_supply_units`, `scenario_profile`, `data_mode`, constraint verification fields.
4. **`backtest_metrics.parquet`**:
   - Required columns: `run_id`, `origin`, `policy`, `total_cost_mu`, `shortage_cost_mu`, `holding_cost_mu`, `shipping_cost_mu`, `filled_units`, `unmet_units`, `fill_rate`, `oracle_regret_mu`, `solver_status`, `runtime_seconds`, `data_mode`.

## 2. Run Manifest JSON Schema (`--manifest`)

```json
{
  "run_id": "demo_20261029_seed42",
  "git_commit": "uncommitted",
  "git_dirty": true,
  "python_version": "3.14.0",
  "platform": "macOS",
  "global_seed": 42,
  "config_path": "configs/demo.yaml",
  "config_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "data_mode": "synthetic_demo",
  "economic_currency": "MU",
  "inventory_data_provenance": "synthetic",
  "dataset_hashes": {
    "weekly_demand.parquet": "a1b2c3d4"
  },
  "model_checkpoints": {
    "supervised_pmf": {"path": "artifacts/demo/supervised.pt", "sha256": "1111aaaa"},
    "dfl_pmf": {"path": "artifacts/demo/dfl.pt", "sha256": "2222bbbb"}
  },
  "test_origins": ["2016-01-31", "2016-02-07", "2016-02-14", "2016-02-21"],
  "common_scenario_seed": 1001,
  "num_scenarios": 50,
  "policies_evaluated": [
    "equal_allocation",
    "proportional_allocation",
    "supervised_pto",
    "supervised_dro",
    "dfl_pto",
    "dfl_dro",
    "clairvoyant_oracle"
  ],
  "policy_origin_Fingerprints": {
    "equal_allocation": "fp_env_01",
    "proportional_allocation": "fp_env_01",
    "supervised_pto": "fp_env_01",
    "supervised_dro": "fp_env_01",
    "dfl_pto": "fp_env_01",
    "dfl_dro": "fp_env_01",
    "clairvoyant_oracle": "fp_env_01"
  },
  "solver_statuses": {
    "supervised_pto": ["optimal", "optimal", "optimal", "optimal"],
    "supervised_dro": ["optimal", "optimal", "optimal", "optimal"],
    "dfl_pto": ["optimal", "optimal", "optimal", "optimal"],
    "dfl_dro": ["optimal", "optimal", "optimal", "optimal"],
    "clairvoyant_oracle": ["optimal", "optimal", "optimal", "optimal"]
  },
  "metrics_recorded": [
    "total_cost_mu",
    "shortage_cost_mu",
    "holding_cost_mu",
    "shipping_cost_mu",
    "filled_units",
    "unmet_units",
    "fill_rate",
    "oracle_regret_mu",
    "solver_status",
    "runtime_seconds",
    "cvar_90_cost_mu"
  ],
  "bootstrap_unit": "forecast_origin",
  "small_origin_caveat_noted": true
}
```
