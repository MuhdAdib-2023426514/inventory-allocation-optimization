---
name: experiment-reproducibility
description: >-
  Enforce end-to-end experiment reproducibility, artifact provenance, seed
  determinism, synthetic vs M5 namespace isolation, model checkpoint hashing,
  solver outcome logging, and apples-to-apples 7-policy evaluation tables.
  Activate when configuring experiment runs, writing parquet/report outputs,
  executing backtests, or updating implementation status and results summaries.
---

# Experiment Reproducibility Skill

This skill ensures every experiment run, artifact bundle, and reported table satisfies the provenance, determinism, and completeness rules in [`AGENTS.md`](../../../AGENTS.md) (§1, §4, §8, §9, §12, §14, §15).

## Triggering Conditions

Invoke this skill automatically when:
- Implementing or modifying `src/inventory_di/config.py`, `src/inventory_di/io.py`, `src/inventory_di/cli.py`, `src/inventory_di/evaluation/backtest.py`, or `src/inventory_di/evaluation/metrics.py`.
- Writing or validating `weekly_demand.parquet`, `inventory_config.parquet`, `allocation_results.parquet`, `backtest_metrics.parquet`, or `reports/results_summary.md`.
- Auditing whether a run is `synthetic_demo` vs `m5_sales_proxy` or updating `docs/IMPLEMENTATION_STATUS.md`.

## Required Inputs

1. **Authoritative Contract**: [`AGENTS.md`](../../../AGENTS.md) §1, §4, §8, §9, §12, §14, and §15.
2. **Manifest & Contract Schema**: [`references/manifest_schema.md`](./references/manifest_schema.md).
3. **Target Experiment Manifest**: Run manifest JSON and/or environment snapshot.

## Execution Workflow

1. **Run the Reproducibility Verifier Self-Test & Environment Probe**:
   ```bash
   .venv/bin/python .agents/skills/experiment-reproducibility/scripts/verify_reproducibility.py --self-test
   .venv/bin/python .agents/skills/experiment-reproducibility/scripts/verify_reproducibility.py --check-env
   ```
2. **Enforce Complete Run Metadata (`AGENTS.md` §4, §9)**:
   - Record `run_id`, `git_commit` (or `"uncommitted"` with worktree status), `python_version`, `platform`, `global_seed`, `config_path`, `config_sha256`, and `data_mode`.
   - Record SHA-256 checksums for input datasets and trained model checkpoints (`supervised` and `dfl`).
3. **Enforce Strict `synthetic_demo` vs `m5_sales_proxy` Isolation (`AGENTS.md` §0, §1.3, §4, §14)**:
   - Require `data_mode` to be explicitly `"synthetic_demo"` or `"m5_sales_proxy"`.
   - If `data_mode == "m5_sales_proxy"`, verify that real M5 raw file hashes (`calendar.csv`, `sell_prices.csv`, and `sales_train_evaluation.csv` or `sales_train_validation.csv`) are present and non-empty. Never allow synthetic outputs to be labeled as empirical M5 results.
   - Require `economic_currency == "MU"` and `inventory_data_provenance == "synthetic"`.
4. **Enforce Apples-to-Apples 7-Policy Comparison (`AGENTS.md` §1.7, §8)**:
   - Verify all 7 required policies are evaluated at the exact same held-out test origins with identical `opening_stock_units`, `warehouse_supply_units`, `store_total_capacity_units`, and `scenario_seed`:
     1. `equal_allocation`
     2. `proportional_allocation`
     3. `supervised_pto`
     4. `supervised_dro`
     5. `dfl_pto`
     6. `dfl_dro`
     7. `clairvoyant_oracle` (hindsight regret reference only)
5. **Enforce Solver Status Logging & Origin-Wise Bootstrap Metrics (`AGENTS.md` §8, §14)**:
   - Every policy-origin evaluation must log `solver_status` and `runtime_seconds`.
   - Aggregate metrics must include `total_cost_mu`, `shortage_cost_mu`, `holding_cost_mu`, `shipping_cost_mu`, `filled_units`, `unmet_units`, `fill_rate`, `oracle_regret_mu`, and tail-loss (CVaR/P90) statistics.
   - Confidence intervals must be computed via block bootstrap across **forecast origins** (never across correlated store–SKU rows), with explicit small-sample caveats when the number of origins is small ($< 20$).

## Verification Procedures

Validate an experiment manifest JSON file:
```bash
.venv/bin/python .agents/skills/experiment-reproducibility/scripts/verify_reproducibility.py --manifest path/to/run_manifest.json
```

## Expected Outputs

- JSON verification report detailing metadata completeness, provenance isolation, policy parity, solver status hygiene, and environment package diagnostics.
- Exit code `0` on full compliance; exit code `1` on missing metadata, mixed data modes, missing policies, or row-level bootstrap misuse.

## Completion Criteria

- `verify_reproducibility.py --self-test` exits `0`.
- Every reported table or artifact bundle has a corresponding verified run manifest with zero provenance or policy-parity errors.
