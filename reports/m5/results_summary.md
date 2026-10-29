# Inventory Allocation Experiment Results Summary

- **Data Provenance / Mode:** M5 Sales Proxy + Simulated Inventory (`data_mode='m5_sales_proxy'`)
- **Run ID:** `m5_v1` (Global Seed: `42`, Git Commit: `89a083b`)
- **Economic Units:** Generic Monetary Units (`MU`), synthetic inventory/operating constraints (`data_provenance='synthetic'`).
- **Held-out Test Origins:** `12` chronological weekly origins.
- **Lowest Mean Out-of-Sample Cost Policy (Non-Oracle):** `proportional_allocation` (**1298.91 MU**, Fill Rate **73.9%**, Mean Oracle Regret **278.76 MU**).

## 1. Primary 7-Policy Out-of-Sample Comparison

| Policy | Mean Cost (MU) | 95% Origin Block-Bootstrap CI (MU) | Realized CVaR_0.90 (MU) | Mean Oracle Regret (MU) | Fill Rate (%) | Mean Unmet Units | Diff vs Equal (MU / %) | Diff vs Supervised PTO (MU / %) | Solver Status |
| :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `equal_allocation` | 1432.35 | [1338.37, 1499.66] | 1843.00 | 412.20 | 71.2% | 186.58 | +0.00 (+0.0%) | +109.58 (+8.3%) | `optimal` |
| `proportional_allocation` | 1298.91 | [1202.40, 1372.74] | 1711.69 | 278.76 | 73.9% | 169.00 | -133.43 (-9.3%) | -23.85 (-1.8%) | `optimal` |
| `supervised_pto` | 1322.76 | [1213.17, 1405.20] | 1708.12 | 302.61 | 73.5% | 172.00 | -109.58 (-7.7%) | +0.00 (+0.0%) | `optimal` |
| `supervised_dro` | 1315.14 | [1224.25, 1386.15] | 1677.40 | 294.99 | 73.6% | 171.17 | -117.20 (-8.2%) | -7.62 (-0.6%) | `optimal` |
| `dfl_pto` | 1312.70 | [1209.55, 1389.80] | 1686.33 | 292.55 | 73.6% | 170.75 | -119.64 (-8.4%) | -10.06 (-0.8%) | `optimal` |
| `dfl_dro` | 1328.63 | [1237.12, 1399.58] | 1689.26 | 308.48 | 73.3% | 173.00 | -103.71 (-7.2%) | +5.87 (+0.4%) | `optimal` |
| `clairvoyant_oracle` | 1020.15 | [900.48, 1099.12] | 1536.59 | 0.00 | 79.2% | 135.42 | -412.20 (-28.8%) | -302.61 (-22.9%) | `optimal` |

## 2. Scientific Verdict & Key Findings

- **Baseline `supervised_pto` achieved lower mean cost than `dfl_dro` on this cohort:** `dfl_dro` averaged **1328.63 MU** vs. **1322.76 MU** for `supervised_pto` (**+5.87 MU**), while `dfl_pto` differed from `supervised_pto` by **-10.06 MU**.
- **Heuristic Baselines:** `equal_allocation` averaged **1432.35 MU** (fill rate **71.2%**) and `proportional_allocation` averaged **1298.91 MU** (fill rate **73.9%**).
- **DFL vs. Supervised PTO:** `dfl_pto` averaged **1312.70 MU** (regret **292.55 MU**) compared to `supervised_pto` at **1322.76 MU** (regret **302.61 MU**).
- **DRO Tail-Risk Protection:** `supervised_dro` achieved realized tail cost $\text{CVaR}_{0.90} = 1677.40\text{ MU}$ and `dfl_dro` achieved $\text{CVaR}_{0.90} = 1689.26\text{ MU}$ (under probability-reweighting ambiguity set $\mathcal{P}_\alpha$ with $\alpha=0.90, \text{risk\_weight}=0.50$).
- **Statistical Caveat:** Computed via moving-block bootstrap (block_size=2, B=200) across 12 held-out forecast origins. Because 12 < 20 origins is a small time-series sample, intervals are descriptive uncertainty bands and should not be over-interpreted as asymptotic p-values.

## 3. Model Expected Cost vs. Model CVaR vs. Realized Historical Test Cost

Per `AGENTS.md` §7 & §10, model-implied scenario expectations/CVaR and realized held-out test costs are strictly separated:

| Policy | Model-Implied Expected Cost (MU) | Model-Implied CVaR_0.90 (MU) | Realized Held-Out Mean Cost (MU) | Realized Held-Out CVaR_0.90 (MU) |
| :--- | ---: | ---: | ---: | ---: |
| `equal_allocation` | 1659.12 | 2158.40 | 1432.35 | 1843.00 |
| `proportional_allocation` | 1540.00 | 2019.98 | 1298.91 | 1711.69 |
| `supervised_pto` | 1503.21 | 1995.83 | 1322.76 | 1708.12 |
| `supervised_dro` | 1465.81 | 1928.96 | 1315.14 | 1677.40 |
| `dfl_pto` | 1537.26 | 2045.36 | 1312.70 | 1686.33 |
| `dfl_dro` | 1500.12 | 1981.05 | 1328.63 | 1689.26 |
| `clairvoyant_oracle` | 1727.63 | 2219.29 | 1020.15 | 1536.59 |

## 4. Decision-Focused Learning (`lambda_DFL`) Validation Ablation

| SKU | `lambda_DFL` | Validation NLL | Validation MAE | Validation WAPE | Validation Decision Regret (MU) | Calibrated Temperature |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `HOUSEHOLD_1_055` | 0.00 | 3.0340 | 4.505 | 0.320 | 17.594 | 2.50 |
| `HOUSEHOLD_1_055` | 0.50 | 3.0097 | 4.442 | 0.316 | 16.613 | 2.25 |
| `HOUSEHOLD_1_055` | 1.00 | 2.9453 | 4.127 | 0.293 | 13.956 | 1.45 |
| `HOUSEHOLD_1_055` | 2.00 | 2.9633 | 4.121 | 0.293 | 13.044 | 1.45 |

## 5. Scenario Calibration Diagnostics & Validation DRO Tuning (Priorities 2 & 3)

- **Validation-Split Scenario Calibration (`split == 'val'`):**
  - `supervised` copula scenarios: raw mean bias `-0.23` units $\rightarrow$ calibrated bias `+0.01` units; Energy Score `25.35` $\rightarrow$ `21.90` (mean scale factor `1.00`).
  - `dfl` (`JointMultiSKUMarginalAllocationModel`) copula scenarios: raw mean bias `-0.81` units $\rightarrow$ calibrated bias `-0.14` units; Energy Score `25.37` $\rightarrow$ `22.22` (mean scale factor `1.04`).
- **Validation-Selected DRO Hyperparameters (`split == 'val'`):**
  - `supervised_dro`: selected $\alpha^* = 0.90, \text{risk\_weight}^* = 0.25$.
  - `dfl_dro`: selected $\alpha^* = 0.90, \text{risk\_weight}^* = 0.25$.

## 6. Multi-Seed Stability Across Seeds `[42, 101, 202]` (Priority 4)

| Policy | Mean Cost Across Seeds (MU) | Seed SD (MU) | Mean Fill Rate (%) |
| :--- | ---: | ---: | ---: |
| `clairvoyant_oracle` | 1020.15 | 0.00 | 79.2% |
| `dfl_dro` | 1328.50 | 2.88 | 73.3% |
| `dfl_pto` | 1312.70 | 0.00 | 73.6% |
| `equal_allocation` | 1432.35 | 0.00 | 71.2% |
| `proportional_allocation` | 1298.91 | 0.00 | 73.9% |
| `supervised_dro` | 1318.30 | 4.25 | 73.5% |
| `supervised_pto` | 1322.76 | 0.00 | 73.5% |

## 7. Rolling Inventory Carryover & Replenishment Lead-Time Evaluation (Priority 7)

| Lead Time (Weeks) | Policy | Rolling Mean Cost (MU) | Shortage Cost (MU) | Holding Cost (MU) | Fill Rate (%) | Ending Inventory (Units) |
| :---: | :--- | ---: | ---: | ---: | ---: | ---: |
| `0` | `equal_allocation` | 2080.03 | 1718.05 | 227.48 | 57.3% | 304 |
| `0` | `proportional_allocation` | 1620.85 | 1308.64 | 158.60 | 67.5% | 185 |
| `0` | `supervised_pto` | 1224.10 | 952.02 | 100.71 | 76.4% | 105 |
| `0` | `supervised_dro` | 1233.16 | 962.95 | 99.37 | 76.2% | 115 |
| `0` | `dfl_pto` | 1212.68 | 942.87 | 98.07 | 76.7% | 105 |
| `0` | `dfl_dro` | 1230.94 | 960.23 | 99.78 | 76.2% | 113 |
| `0` | `clairvoyant_oracle` | 1020.15 | 847.38 | 0.00 | 78.9% | 0 |
| `1` | `equal_allocation` | 2122.68 | 1819.43 | 174.75 | 54.9% | 266 |
| `1` | `proportional_allocation` | 1876.10 | 1603.92 | 135.40 | 60.2% | 163 |
| `1` | `supervised_pto` | 1608.62 | 1366.21 | 93.52 | 66.2% | 112 |
| `1` | `supervised_dro` | 1602.70 | 1363.23 | 90.56 | 66.3% | 108 |
| `1` | `dfl_pto` | 1608.40 | 1366.68 | 92.66 | 66.2% | 118 |
| `1` | `dfl_dro` | 1608.15 | 1367.95 | 91.48 | 66.2% | 112 |
| `1` | `clairvoyant_oracle` | 1653.98 | 1454.55 | 57.38 | 64.0% | 64 |

## 8. Methodological Caveats & Limitations

1. **Observed Sales != Latent Demand:** Even when run on M5 data, observed unit sales are right-censored by historical stockouts; in `synthetic_demo` mode (`DEMO / SYNTHETIC`), demand is generated from a documented Poisson-Gamma weekly process with shared cross-store shocks.
2. **Synthetic Operating Economics:** Opening stock, warehouse supply caps, store capacities, and unit costs (`MU`) are explicitly synthetic (`data_provenance='synthetic'`).
3. **Explicit Ambiguity Set:** DRO policies optimize over the scenario-probability reweighting ambiguity set $\mathcal{P}_\alpha = \{ w \ge 0 : \sum_j w_j = 1, w_j \le p_j / (1-\alpha) \}$ (equivalent to Rockafellar–Uryasev $\text{CVaR}_\alpha$), **not** Wasserstein DRO.
