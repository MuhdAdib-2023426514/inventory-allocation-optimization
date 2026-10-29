# Inventory Allocation Experiment Results Summary

- **Data Provenance / Mode:** M5 Sales Proxy + Simulated Inventory (`data_mode='m5_sales_proxy'`) (Demand Regime: `regular`)
- **Run ID:** `m5_v1` (Global Seed: `42`, Git Commit: `351c485`)
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
- **DRO Tail-Risk Protection:** `supervised_dro` achieved realized tail cost $\text{CVaR}_{0.90} = 1677.40\text{ MU}$ (validation-selected $\alpha^*=0.90, \text{risk\_weight}^*=0.25$) and `dfl_dro` achieved $\text{CVaR}_{0.90} = 1689.26\text{ MU}$ (validation-selected $\alpha^*=0.90, \text{risk\_weight}^*=0.25$, config default $\alpha=0.90, \text{risk\_weight}=0.50$).
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
  - `supervised_dro`: selected $\alpha^* = 0.90, \text{risk\_weight}^* = 0.25$ (config default $\alpha = 0.90, \text{risk\_weight} = 0.50$).
  - `dfl_dro`: selected $\alpha^* = 0.90, \text{risk\_weight}^* = 0.25$ (config default $\alpha = 0.90, \text{risk\_weight} = 0.50$).

## 6. Independent Training-Seed & Scenario-Seed Stability Across Seeds `[42, 101, 202]` (Priority 4 / P1.2)

Each seed independently initializes and trains both `supervised` and `dfl` neural networks (`training_seed_independent = True`) and draws independent copula scenario sets:

| Policy | Mean Cost Across Seeds (MU) | Seed SD (MU) | Mean Regret Across Seeds (MU) | Mean Fill Rate (%) |
| :--- | ---: | ---: | ---: | ---: |
| `clairvoyant_oracle` | 1020.15 | 0.00 | 0.00 | 79.2% |
| `dfl_dro` | 1320.57 | 6.98 | 300.42 | 73.4% |
| `dfl_pto` | 1305.74 | 6.05 | 285.59 | 73.8% |
| `equal_allocation` | 1432.35 | 0.00 | 412.20 | 71.2% |
| `proportional_allocation` | 1298.91 | 0.00 | 278.76 | 73.9% |
| `supervised_dro` | 1324.02 | 12.67 | 303.87 | 73.3% |
| `supervised_pto` | 1322.71 | 4.41 | 302.56 | 73.4% |

## 7. Rolling Inventory Carryover & Replenishment Lead-Time Evaluation (Priority 7 / P1.1)

For `clairvoyant_oracle`, the multi-period horizon MILP (`solve_multiperiod_carryover_oracle`) optimizes jointly across $t=0 \dots T-1$ with exact carryover state transitions and shared initial pipeline inventory, guaranteeing nonnegative rolling oracle regret ($\text{Rolling Regret} \ge 0$) at both $L=0$ and $L=1$:

| Lead Time (Weeks) | Policy | Rolling Mean Cost (MU) | Rolling Oracle Regret (MU) | Shortage Cost (MU) | Holding Cost (MU) | Fill Rate (%) | Ending Inventory (Units) |
| :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `equal_allocation` | 2080.03 | 1091.26 | 1718.05 | 227.48 | 57.3% | 304 |
| `0` | `proportional_allocation` | 1620.85 | 632.08 | 1308.64 | 158.60 | 67.5% | 185 |
| `0` | `supervised_pto` | 1224.10 | 235.33 | 952.02 | 100.71 | 76.4% | 105 |
| `0` | `supervised_dro` | 1233.16 | 244.39 | 962.95 | 99.37 | 76.2% | 115 |
| `0` | `dfl_pto` | 1212.68 | 223.91 | 942.87 | 98.07 | 76.7% | 105 |
| `0` | `dfl_dro` | 1230.94 | 242.18 | 960.23 | 99.78 | 76.2% | 113 |
| `0` | `clairvoyant_oracle` | 988.77 | 0.00 | 798.38 | 14.45 | 80.2% | 0 |
| `1` | `equal_allocation` | 1860.34 | 863.85 | 1538.97 | 177.93 | 61.8% | 267 |
| `1` | `proportional_allocation` | 1568.04 | 571.55 | 1282.03 | 132.06 | 68.1% | 152 |
| `1` | `supervised_pto` | 1436.95 | 440.46 | 1166.99 | 110.14 | 71.1% | 125 |
| `1` | `supervised_dro` | 1398.87 | 402.38 | 1132.89 | 104.37 | 71.9% | 121 |
| `1` | `dfl_pto` | 1394.87 | 398.38 | 1130.98 | 102.40 | 72.0% | 114 |
| `1` | `dfl_dro` | 1381.32 | 384.83 | 1117.43 | 101.42 | 72.3% | 123 |
| `1` | `clairvoyant_oracle` | 996.49 | 0.00 | 804.74 | 16.14 | 80.0% | 0 |

## 8. Full-Horizon Cost–Risk–Service-Level Trade-Off Frontier (`P1.3`)

Evaluated across all held-out test origins over $(\alpha, \text{risk\_weight})$:

| Policy Family | `alpha` | `risk_weight` | Realized Mean Cost (MU) | Realized CVaR_0.90 (MU) | Fill Rate (%) | Period No-Stockout Rate (%) | Mean Shortage Cost (MU) | Mean Holding Cost (MU) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dfl_dro` | 0.50 | 0.00 | 1325.65 | 1679.65 | 73.3% | 40.5% | 1084.02 | 61.08 |
| `dfl_dro` | 0.50 | 0.25 | 1324.35 | 1683.21 | 73.4% | 40.3% | 1082.92 | 60.92 |
| `dfl_dro` | 0.50 | 0.50 | 1325.63 | 1684.41 | 73.3% | 40.2% | 1083.93 | 61.09 |
| `dfl_dro` | 0.50 | 0.75 | 1328.07 | 1691.67 | 73.3% | 40.5% | 1085.95 | 61.42 |
| `dfl_dro` | 0.50 | 1.00 | 1331.81 | 1693.88 | 73.2% | 40.2% | 1089.13 | 61.94 |
| `dfl_dro` | 0.75 | 0.00 | 1325.65 | 1679.65 | 73.3% | 40.5% | 1084.02 | 61.08 |
| `dfl_dro` | 0.75 | 0.25 | 1326.18 | 1683.24 | 73.3% | 40.8% | 1084.45 | 61.17 |
| `dfl_dro` | 0.75 | 0.50 | 1326.83 | 1682.01 | 73.3% | 40.7% | 1084.98 | 61.25 |
| `dfl_dro` | 0.75 | 0.75 | 1333.69 | 1689.26 | 73.2% | 40.8% | 1090.83 | 62.18 |
| `dfl_dro` | 0.75 | 1.00 | 1342.86 | 1726.73 | 73.0% | 40.0% | 1098.67 | 63.47 |
| `dfl_dro` | 0.90 | 0.00 | 1325.65 | 1679.65 | 73.3% | 40.5% | 1084.02 | 61.08 |
| `dfl_dro` | 0.90 | 0.25 | 1328.63 | 1689.26 | 73.3% | 40.0% | 1086.60 | 61.48 |
| `dfl_dro` | 0.90 | 0.50 | 1331.19 | 1690.28 | 73.2% | 40.8% | 1088.79 | 61.80 |
| `dfl_dro` | 0.90 | 0.75 | 1347.13 | 1722.51 | 72.9% | 40.5% | 1102.38 | 64.02 |
| `dfl_dro` | 0.90 | 1.00 | 1363.25 | 1729.38 | 72.5% | 40.3% | 1117.49 | 65.58 |
| `dfl_dro` | 0.95 | 0.00 | 1325.65 | 1679.65 | 73.3% | 40.5% | 1084.02 | 61.08 |
| `dfl_dro` | 0.95 | 0.25 | 1329.46 | 1678.48 | 73.3% | 39.8% | 1087.16 | 61.55 |
| `dfl_dro` | 0.95 | 0.50 | 1336.95 | 1698.78 | 73.1% | 40.5% | 1093.60 | 62.59 |
| `dfl_dro` | 0.95 | 0.75 | 1357.46 | 1717.82 | 72.6% | 40.3% | 1112.82 | 64.37 |
| `dfl_dro` | 0.95 | 1.00 | 1392.32 | 1743.47 | 71.7% | 40.3% | 1147.12 | 66.40 |
| `supervised_dro` | 0.50 | 0.00 | 1319.38 | 1686.22 | 73.5% | 41.2% | 1078.59 | 60.21 |
| `supervised_dro` | 0.50 | 0.25 | 1315.06 | 1684.83 | 73.6% | 40.8% | 1074.89 | 59.62 |
| `supervised_dro` | 0.50 | 0.50 | 1309.62 | 1677.26 | 73.7% | 41.3% | 1070.20 | 58.85 |
| `supervised_dro` | 0.50 | 0.75 | 1307.84 | 1671.05 | 73.7% | 41.5% | 1068.61 | 58.59 |
| `supervised_dro` | 0.50 | 1.00 | 1313.84 | 1665.95 | 73.6% | 41.0% | 1073.75 | 59.45 |
| `supervised_dro` | 0.75 | 0.00 | 1319.38 | 1686.22 | 73.5% | 41.2% | 1078.59 | 60.21 |
| `supervised_dro` | 0.75 | 0.25 | 1314.65 | 1685.01 | 73.6% | 40.7% | 1074.55 | 59.54 |
| `supervised_dro` | 0.75 | 0.50 | 1314.58 | 1684.99 | 73.6% | 40.8% | 1074.58 | 59.52 |
| `supervised_dro` | 0.75 | 0.75 | 1318.75 | 1685.92 | 73.5% | 40.7% | 1078.20 | 60.10 |
| `supervised_dro` | 0.75 | 1.00 | 1326.69 | 1712.42 | 73.3% | 40.8% | 1085.00 | 61.22 |
| `supervised_dro` | 0.90 | 0.00 | 1319.38 | 1686.22 | 73.5% | 41.2% | 1078.59 | 60.21 |
| `supervised_dro` | 0.90 | 0.25 | 1315.14 | 1677.40 | 73.6% | 40.7% | 1074.93 | 59.61 |
| `supervised_dro` | 0.90 | 0.50 | 1313.98 | 1678.41 | 73.6% | 41.5% | 1073.91 | 59.46 |
| `supervised_dro` | 0.90 | 0.75 | 1332.16 | 1706.85 | 73.2% | 41.5% | 1089.82 | 61.84 |
| `supervised_dro` | 0.90 | 1.00 | 1348.18 | 1725.36 | 72.8% | 41.2% | 1104.55 | 63.57 |
| `supervised_dro` | 0.95 | 0.00 | 1319.38 | 1686.22 | 73.5% | 41.2% | 1078.59 | 60.21 |
| `supervised_dro` | 0.95 | 0.25 | 1318.95 | 1703.20 | 73.5% | 40.5% | 1078.03 | 60.14 |
| `supervised_dro` | 0.95 | 0.50 | 1328.28 | 1720.60 | 73.3% | 41.0% | 1085.99 | 61.43 |
| `supervised_dro` | 0.95 | 0.75 | 1337.85 | 1724.58 | 73.0% | 41.5% | 1095.74 | 61.78 |
| `supervised_dro` | 0.95 | 1.00 | 1370.23 | 1758.88 | 72.3% | 40.7% | 1126.32 | 64.61 |

## 9. Demand Regime Generalization: Regular vs. Intermittent M5 SKUs (`P2.1`)

Comparison across the 10 regular-selling `HOUSEHOLD` SKUs (`ADI <= 1.25`) vs. 10 intermittent/lumpy `HOUSEHOLD` SKUs (`ADI >= 1.35`, selected strictly on `split == 'train'`):

| Policy | Regular Mean Cost (MU) | Regular CVaR_0.90 (MU) | Regular Fill Rate (%) | Intermittent Mean Cost (MU) | Intermittent CVaR_0.90 (MU) | Intermittent Fill Rate (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `equal_allocation` | 1432.35 | 1843.00 | 71.2% | 174.45 | 269.21 | 56.2% |
| `proportional_allocation` | 1298.91 | 1711.69 | 73.9% | 176.94 | 274.42 | 55.3% |
| `supervised_pto` | 1322.76 | 1708.12 | 73.5% | 172.89 | 253.60 | 55.4% |
| `supervised_dro` | 1315.14 | 1677.40 | 73.6% | 171.30 | 256.92 | 56.1% |
| `dfl_pto` | 1312.70 | 1686.33 | 73.6% | 175.62 | 253.09 | 54.5% |
| `dfl_dro` | 1328.63 | 1689.26 | 73.3% | 172.59 | 254.54 | 55.6% |
| `clairvoyant_oracle` | 1020.15 | 1536.59 | 79.2% | 89.85 | 160.78 | 73.7% |

## 10. Methodological Caveats & Limitations

1. **Observed Sales != Latent Demand:** Even when run on M5 data, observed unit sales are right-censored by historical stockouts; in `synthetic_demo` mode (`DEMO / SYNTHETIC`), demand is generated from a documented Poisson-Gamma weekly process with shared cross-store shocks.
2. **Synthetic Operating Economics:** Opening stock, warehouse supply caps, store capacities, and unit costs (`MU`) are explicitly synthetic (`data_provenance='synthetic'`).
3. **Explicit Ambiguity Set:** DRO policies optimize over the scenario-probability reweighting ambiguity set $\mathcal{P}_\alpha = \{ w \ge 0 : \sum_j w_j = 1, w_j \le p_j / (1-\alpha) \}$ (equivalent to Rockafellar–Uryasev $\text{CVaR}_\alpha$), **not** Wasserstein DRO.
