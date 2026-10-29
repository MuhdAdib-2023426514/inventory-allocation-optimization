---
name: data-leakage-audit
description: >-
  Audit datasets, chronological splits, feature pipelines, SKU/store cohort
  selection, probability calibration, scenario copula estimation, and evaluation
  harnesses for temporal and target leakage. Activate when creating or modifying
  M5 ingestion, synthetic data generation, rolling features, train/val/test
  splits, PMF calibration, DRO scenario sampling, or backtest evaluation.
---

# Data Leakage Audit Skill

This skill enforces strict temporal causality across all stages of the inventory allocation pipeline as mandated by [`AGENTS.md`](../../../AGENTS.md) (§1.2, §4, §6, §7, §8, §9, §14). Every decision at `forecast_origin` $t_0$ must depend exclusively on information available on or before $t_0$.

## Triggering Conditions

Invoke this skill automatically when:
- Implementing or modifying `src/inventory_di/data/` (`m5.py`, `synthetic.py`, `features.py`, `splits.py`).
- Selecting the store/SKU cohort, setting allocation caps $K_{sp}$, or choosing discrete PMF support $H$ and overflow tail parameters.
- Implementing probability calibration, empirical copula / rank-bootstrap scenario generation, or `proportional_allocation` trailing demand weights.
- Building or reviewing `src/inventory_di/evaluation/backtest.py` or `tests/test_data_leakage.py`.

## Required Inputs

1. **Authoritative Contract**: [`AGENTS.md`](../../../AGENTS.md) §1.2, §4, §6, §7, §8, §9, and §14.
2. **Leakage Vector Catalog**: [`references/leakage_vectors.md`](./references/leakage_vectors.md).
3. **Target Artifacts**: Pipeline lineage JSON manifest and/or Python source directory (`src/`, `tests/`).

## Execution Workflow

1. **Run the Deterministic Leakage Auditor Self-Test**:
   ```bash
   .venv/bin/python .agents/skills/data-leakage-audit/scripts/audit_leakage.py --self-test
   ```
2. **Audit Whole-Block Chronological Splits (`AGENTS.md` §4, §9)**:
   - Verify `split` (`train`, `val`, `test`) partitions time by **whole `forecast_origin` blocks**:
     $$\max(\text{origin}_{\text{train}}) < \min(\text{origin}_{\text{val}}) \le \max(\text{origin}_{\text{val}}) < \min(\text{origin}_{\text{test}})$$
   - Reject random row splits (`train_test_split(..., shuffle=True)`) or interleaved store–SKU split assignments at the same origin.
3. **Audit Feature Engineering & Price Covariates (`AGENTS.md` §4)**:
   - For every origin $t_0$ and 7-day target window $[t_{\text{start}}, t_{\text{end}}]$, require:
     $$\text{max\_feature\_date} \le t_0 < t_{\text{start}} \le t_{\text{end}}$$
   - Ensure M5 `sell_prices.csv` covariates use only historical lagged prices or prices explicitly scheduled at $t_0$; reject unlagged future realized prices.
4. **Audit Cohort Selection, Support $H$, and Caps $K_{sp}$ (`AGENTS.md` §2, §4, §6)**:
   - Confirm store/SKU filtering (`cat_id == "HOUSEHOLD"`), allocation caps $K_{sp}$, PMF support $H$, and tail distribution parameters are estimated strictly on `split == "train"`.
   - Confirm realized evaluation demand $D_{sp}$ is **never clipped at $H$**.
5. **Audit Probability Calibration & Scenario Generation (`AGENTS.md` §6, §7, §9)**:
   - Confirm PMF temperature/isotonic/probability calibration fits on `val` (or `train`) only—never `test`.
   - Confirm empirical copula / rank-bootstrap weekly demand shocks are estimated on `train` residuals/ranks only—never `test`.
6. **Audit Baseline & Oracle Isolation (`AGENTS.md` §8)**:
   - Confirm `proportional_allocation` computes trailing demand weights strictly from weeks ending $\le t_0$.
   - Confirm `clairvoyant_oracle` is restricted to hindsight regret calculation and never feeds into policy selection or feature state.

## Verification Procedures

Audit a lineage manifest JSON and scan source code for forbidden patterns:
```bash
.venv/bin/python .agents/skills/data-leakage-audit/scripts/audit_leakage.py --manifest path/to/lineage_manifest.json --scan-dir src
```

## Expected Outputs

- JSON audit report listing checked temporal gates, AST/source scan findings, and any detected leakage vectors.
- Exit code `0` if zero leakage violations are found; exit code `1` if any temporal or target leakage is detected.

## Completion Criteria

- `audit_leakage.py --self-test` exits `0`.
- Both static source inspection (`--scan-dir`) and runtime lineage manifest auditing (`--manifest`) report `"status": "PASS"`.
