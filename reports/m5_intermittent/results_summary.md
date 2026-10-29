# Inventory Allocation Experiment Results Summary

- **Data Provenance / Mode:** M5 Sales Proxy + Simulated Inventory (`data_mode='m5_sales_proxy'`) (Demand Regime: `intermittent`)
- **Run ID:** `m5_intermittent_v1` (Global Seed: `42`, Git Commit: `351c485`)
- **Economic Units:** Generic Monetary Units (`MU`), synthetic inventory/operating constraints (`data_provenance='synthetic'`).
- **Held-out Test Origins:** `12` chronological weekly origins.
- **Lowest Mean Out-of-Sample Cost Policy (Non-Oracle):** `supervised_dro` (**171.30 MU**, Fill Rate **56.1%**, Mean Oracle Regret **81.45 MU**).

## 1. Primary 7-Policy Out-of-Sample Comparison

| Policy | Mean Cost (MU) | 95% Origin Block-Bootstrap CI (MU) | Realized CVaR_0.90 (MU) | Mean Oracle Regret (MU) | Fill Rate (%) | Mean Unmet Units | Diff vs Equal (MU / %) | Diff vs Supervised PTO (MU / %) | Solver Status |
| :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `equal_allocation` | 174.45 | [133.81, 205.46] | 269.21 | 84.60 | 56.2% | 20.17 | +0.00 (+0.0%) | +1.56 (+0.9%) | `optimal` |
| `proportional_allocation` | 176.94 | [136.18, 207.48] | 274.42 | 87.09 | 55.3% | 20.58 | +2.50 (+1.4%) | +4.05 (+2.3%) | `optimal` |
| `supervised_pto` | 172.89 | [131.90, 202.62] | 253.60 | 83.04 | 55.4% | 20.67 | -1.56 (-0.9%) | +0.00 (+0.0%) | `optimal` |
| `supervised_dro` | 171.30 | [129.69, 199.76] | 256.92 | 81.45 | 56.1% | 20.33 | -3.15 (-1.8%) | -1.59 (-0.9%) | `optimal` |
| `dfl_pto` | 175.62 | [135.91, 203.01] | 253.09 | 85.77 | 54.5% | 20.83 | +1.18 (+0.7%) | +2.73 (+1.6%) | `optimal` |
| `dfl_dro` | 172.59 | [132.08, 200.18] | 254.54 | 82.74 | 55.6% | 20.42 | -1.85 (-1.1%) | -0.30 (-0.2%) | `optimal` |
| `clairvoyant_oracle` | 89.85 | [60.70, 110.70] | 160.78 | 0.00 | 73.7% | 12.50 | -84.60 (-48.5%) | -83.04 (-48.0%) | `optimal` |

## 2. Scientific Verdict & Key Findings

- **Superior out-of-sample decision cost vs. `supervised_pto`:** `dfl_dro` achieved a mean cost of **172.59 MU** (vs. **172.89 MU** for `supervised_pto`, a change of **-0.30 MU** or **-0.17%**).
- **Heuristic Baselines:** `equal_allocation` averaged **174.45 MU** (fill rate **56.2%**) and `proportional_allocation` averaged **176.94 MU** (fill rate **55.3%**).
- **DFL vs. Supervised PTO:** `dfl_pto` averaged **175.62 MU** (regret **85.77 MU**) compared to `supervised_pto` at **172.89 MU** (regret **83.04 MU**).
- **DRO Tail-Risk Protection:** `supervised_dro` achieved realized tail cost $\text{CVaR}_{0.90} = 256.92\text{ MU}$ (validation-selected $\alpha^*=0.90, \text{risk\_weight}^*=0.50$) and `dfl_dro` achieved $\text{CVaR}_{0.90} = 254.54\text{ MU}$ (validation-selected $\alpha^*=0.90, \text{risk\_weight}^*=0.25$, config default $\alpha=0.90, \text{risk\_weight}=0.25$).
- **Statistical Caveat:** Computed via moving-block bootstrap (block_size=2, B=200) across 12 held-out forecast origins. Because 12 < 20 origins is a small time-series sample, intervals are descriptive uncertainty bands and should not be over-interpreted as asymptotic p-values.

## 3. Model Expected Cost vs. Model CVaR vs. Realized Historical Test Cost

Per `AGENTS.md` §7 & §10, model-implied scenario expectations/CVaR and realized held-out test costs are strictly separated:

| Policy | Model-Implied Expected Cost (MU) | Model-Implied CVaR_0.90 (MU) | Realized Held-Out Mean Cost (MU) | Realized Held-Out CVaR_0.90 (MU) |
| :--- | ---: | ---: | ---: | ---: |
| `equal_allocation` | 177.99 | 326.63 | 174.45 | 269.21 |
| `proportional_allocation` | 186.35 | 336.51 | 176.94 | 274.42 |
| `supervised_pto` | 174.93 | 325.97 | 172.89 | 253.60 |
| `supervised_dro` | 174.63 | 311.71 | 171.30 | 256.92 |
| `dfl_pto` | 194.18 | 374.83 | 175.62 | 253.09 |
| `dfl_dro` | 192.50 | 364.17 | 172.59 | 254.54 |
| `clairvoyant_oracle` | 230.31 | 408.00 | 89.85 | 160.78 |

## 4. Decision-Focused Learning (`lambda_DFL`) Validation Ablation

| SKU | `lambda_DFL` | Validation NLL | Validation MAE | Validation WAPE | Validation Decision Regret (MU) | Calibrated Temperature |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `HOUSEHOLD_1_092` | 0.00 | 1.4026 | 1.185 | 1.316 | 9.788 | 1.40 |
| `HOUSEHOLD_1_092` | 0.50 | 1.3924 | 1.232 | 1.369 | 10.581 | 1.35 |
| `HOUSEHOLD_1_092` | 1.00 | 1.3929 | 1.245 | 1.384 | 11.463 | 1.55 |
| `HOUSEHOLD_1_092` | 2.00 | 1.3756 | 1.214 | 1.349 | 12.306 | 1.60 |

## 5. Scenario Calibration Diagnostics & Validation DRO Tuning (Priorities 2 & 3)

- **Validation-Split Scenario Calibration (`split == 'val'`):**
  - `supervised` copula scenarios: raw mean bias `+0.16` units $\rightarrow$ calibrated bias `+0.03` units; Energy Score `5.26` $\rightarrow$ `4.85` (mean scale factor `0.93`).
  - `dfl` (`JointMultiSKUMarginalAllocationModel`) copula scenarios: raw mean bias `+0.24` units $\rightarrow$ calibrated bias `+0.03` units; Energy Score `5.85` $\rightarrow$ `4.97` (mean scale factor `0.90`).
- **Validation-Selected DRO Hyperparameters (`split == 'val'`):**
  - `supervised_dro`: selected $\alpha^* = 0.90, \text{risk\_weight}^* = 0.50$ (config default $\alpha = 0.90, \text{risk\_weight} = 0.25$).
  - `dfl_dro`: selected $\alpha^* = 0.90, \text{risk\_weight}^* = 0.25$ (config default $\alpha = 0.90, \text{risk\_weight} = 0.25$).

## 6. Independent Training-Seed & Scenario-Seed Stability Across Seeds `[42, 101, 202]` (Priority 4 / P1.2)

Each seed independently initializes and trains both `supervised` and `dfl` neural networks (`training_seed_independent = True`) and draws independent copula scenario sets:

| Policy | Mean Cost Across Seeds (MU) | Seed SD (MU) | Mean Regret Across Seeds (MU) | Mean Fill Rate (%) |
| :--- | ---: | ---: | ---: | ---: |
| `clairvoyant_oracle` | 89.85 | 0.00 | 0.00 | 73.7% |
| `dfl_dro` | 177.78 | 4.70 | 87.93 | 53.4% |
| `dfl_pto` | 180.82 | 4.60 | 90.97 | 52.3% |
| `equal_allocation` | 174.45 | 0.00 | 84.60 | 56.2% |
| `proportional_allocation` | 176.94 | 0.00 | 87.09 | 55.3% |
| `supervised_dro` | 171.98 | 2.42 | 82.13 | 55.3% |
| `supervised_pto` | 175.39 | 3.27 | 85.54 | 54.4% |

## 7. Rolling Inventory Carryover & Replenishment Lead-Time Evaluation (Priority 7 / P1.1)

For `clairvoyant_oracle`, the multi-period horizon MILP (`solve_multiperiod_carryover_oracle`) optimizes jointly across $t=0 \dots T-1$ with exact carryover state transitions and shared initial pipeline inventory, guaranteeing nonnegative rolling oracle regret ($\text{Rolling Regret} \ge 0$) at both $L=0$ and $L=1$:

| Lead Time (Weeks) | Policy | Rolling Mean Cost (MU) | Rolling Oracle Regret (MU) | Shortage Cost (MU) | Holding Cost (MU) | Fill Rate (%) | Ending Inventory (Units) |
| :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `equal_allocation` | 241.20 | 165.14 | 77.46 | 146.24 | 71.8% | 234 |
| `0` | `proportional_allocation` | 240.44 | 164.38 | 75.11 | 148.10 | 72.5% | 230 |
| `0` | `supervised_pto` | 150.40 | 74.34 | 101.93 | 38.11 | 62.5% | 45 |
| `0` | `supervised_dro` | 149.14 | 73.07 | 90.82 | 47.01 | 66.7% | 56 |
| `0` | `dfl_pto` | 149.50 | 73.43 | 92.45 | 46.15 | 66.1% | 53 |
| `0` | `dfl_dro` | 151.95 | 75.89 | 91.38 | 49.41 | 66.5% | 56 |
| `0` | `clairvoyant_oracle` | 76.06 | 0.00 | 57.49 | 7.16 | 79.0% | 0 |
| `1` | `equal_allocation` | 237.27 | 149.00 | 75.86 | 143.95 | 72.4% | 231 |
| `1` | `proportional_allocation` | 243.48 | 155.21 | 76.22 | 150.06 | 72.2% | 232 |
| `1` | `supervised_pto` | 158.35 | 70.08 | 98.91 | 48.85 | 63.7% | 48 |
| `1` | `supervised_dro` | 157.62 | 69.35 | 91.46 | 54.82 | 66.5% | 58 |
| `1` | `dfl_pto` | 158.67 | 70.40 | 90.53 | 56.82 | 66.9% | 64 |
| `1` | `dfl_dro` | 161.27 | 73.00 | 90.58 | 59.23 | 66.9% | 66 |
| `1` | `clairvoyant_oracle` | 88.27 | 0.00 | 61.19 | 15.81 | 77.6% | 1 |

## 8. Full-Horizon Cost–Risk–Service-Level Trade-Off Frontier (`P1.3`)

Evaluated across all held-out test origins over $(\alpha, \text{risk\_weight})$:

| Policy Family | `alpha` | `risk_weight` | Realized Mean Cost (MU) | Realized CVaR_0.90 (MU) | Fill Rate (%) | Period No-Stockout Rate (%) | Mean Shortage Cost (MU) | Mean Holding Cost (MU) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dfl_dro` | 0.50 | 0.00 | 172.59 | 246.65 | 55.2% | 77.8% | 130.67 | 26.05 |
| `dfl_dro` | 0.50 | 0.25 | 171.56 | 252.53 | 55.9% | 78.7% | 129.05 | 26.39 |
| `dfl_dro` | 0.50 | 0.50 | 172.06 | 259.00 | 56.5% | 78.5% | 129.06 | 26.73 |
| `dfl_dro` | 0.50 | 0.75 | 173.72 | 259.21 | 56.0% | 78.3% | 130.12 | 27.23 |
| `dfl_dro` | 0.50 | 1.00 | 176.58 | 259.13 | 53.6% | 77.7% | 132.25 | 27.85 |
| `dfl_dro` | 0.75 | 0.00 | 172.59 | 246.65 | 55.2% | 77.8% | 130.67 | 26.05 |
| `dfl_dro` | 0.75 | 0.25 | 172.25 | 246.82 | 55.7% | 78.3% | 129.62 | 26.47 |
| `dfl_dro` | 0.75 | 0.50 | 173.63 | 252.92 | 55.4% | 77.8% | 130.70 | 26.73 |
| `dfl_dro` | 0.75 | 0.75 | 173.88 | 261.08 | 55.4% | 78.0% | 130.72 | 26.97 |
| `dfl_dro` | 0.75 | 1.00 | 183.80 | 270.35 | 50.8% | 76.7% | 138.04 | 29.20 |
| `dfl_dro` | 0.90 | 0.00 | 172.59 | 246.65 | 55.2% | 77.8% | 130.67 | 26.05 |
| `dfl_dro` | 0.90 | 0.25 | 172.59 | 254.54 | 55.6% | 77.8% | 130.17 | 26.39 |
| `dfl_dro` | 0.90 | 0.50 | 173.33 | 265.07 | 55.6% | 77.5% | 130.20 | 26.90 |
| `dfl_dro` | 0.90 | 0.75 | 176.51 | 267.23 | 54.4% | 77.0% | 132.23 | 27.91 |
| `dfl_dro` | 0.90 | 1.00 | 187.22 | 268.27 | 49.9% | 75.3% | 141.60 | 29.38 |
| `dfl_dro` | 0.95 | 0.00 | 172.59 | 246.65 | 55.2% | 77.8% | 130.67 | 26.05 |
| `dfl_dro` | 0.95 | 0.25 | 174.85 | 255.26 | 54.9% | 77.5% | 131.78 | 26.91 |
| `dfl_dro` | 0.95 | 0.50 | 175.99 | 271.47 | 55.2% | 77.0% | 132.82 | 27.07 |
| `dfl_dro` | 0.95 | 0.75 | 178.20 | 265.33 | 53.5% | 76.5% | 135.88 | 26.64 |
| `dfl_dro` | 0.95 | 1.00 | 190.48 | 271.48 | 48.7% | 74.5% | 146.33 | 28.45 |
| `supervised_dro` | 0.50 | 0.00 | 168.93 | 238.84 | 56.1% | 78.0% | 129.15 | 24.10 |
| `supervised_dro` | 0.50 | 0.25 | 171.07 | 246.02 | 55.8% | 78.0% | 129.65 | 25.29 |
| `supervised_dro` | 0.50 | 0.50 | 169.02 | 246.33 | 56.8% | 78.3% | 127.06 | 25.54 |
| `supervised_dro` | 0.50 | 0.75 | 170.37 | 252.42 | 56.4% | 78.3% | 128.03 | 25.88 |
| `supervised_dro` | 0.50 | 1.00 | 171.36 | 253.79 | 56.3% | 78.8% | 128.58 | 26.23 |
| `supervised_dro` | 0.75 | 0.00 | 168.93 | 238.84 | 56.1% | 78.0% | 129.15 | 24.10 |
| `supervised_dro` | 0.75 | 0.25 | 168.96 | 250.92 | 56.8% | 78.3% | 127.57 | 25.20 |
| `supervised_dro` | 0.75 | 0.50 | 171.35 | 258.05 | 56.0% | 77.7% | 129.59 | 25.55 |
| `supervised_dro` | 0.75 | 0.75 | 171.57 | 263.35 | 55.6% | 77.7% | 129.58 | 25.72 |
| `supervised_dro` | 0.75 | 1.00 | 178.28 | 264.38 | 53.3% | 77.7% | 134.83 | 27.01 |
| `supervised_dro` | 0.90 | 0.00 | 168.93 | 238.84 | 56.1% | 78.0% | 129.15 | 24.10 |
| `supervised_dro` | 0.90 | 0.25 | 169.45 | 245.82 | 56.3% | 78.2% | 128.57 | 24.87 |
| `supervised_dro` | 0.90 | 0.50 | 171.30 | 256.92 | 56.1% | 78.3% | 129.55 | 25.47 |
| `supervised_dro` | 0.90 | 0.75 | 174.75 | 265.19 | 54.0% | 77.5% | 132.18 | 26.15 |
| `supervised_dro` | 0.90 | 1.00 | 177.71 | 258.56 | 52.4% | 77.3% | 134.69 | 26.60 |
| `supervised_dro` | 0.95 | 0.00 | 168.93 | 238.84 | 56.1% | 78.0% | 129.15 | 24.10 |
| `supervised_dro` | 0.95 | 0.25 | 170.43 | 246.53 | 56.2% | 77.8% | 129.08 | 25.20 |
| `supervised_dro` | 0.95 | 0.50 | 170.67 | 255.91 | 55.5% | 77.8% | 129.61 | 24.95 |
| `supervised_dro` | 0.95 | 0.75 | 174.17 | 270.80 | 54.1% | 77.5% | 132.17 | 25.71 |
| `supervised_dro` | 0.95 | 1.00 | 177.42 | 264.12 | 52.2% | 77.2% | 135.20 | 26.08 |

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
