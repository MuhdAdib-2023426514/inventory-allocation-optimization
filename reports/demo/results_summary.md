# Inventory Allocation Experiment Results Summary

- **Data Provenance / Mode:** DEMO / SYNTHETIC (`data_mode='synthetic_demo'`) (Demand Regime: `regular`)
- **Run ID:** `demo_v1` (Global Seed: `42`, Git Commit: `351c485`)
- **Economic Units:** Generic Monetary Units (`MU`), synthetic inventory/operating constraints (`data_provenance='synthetic'`).
- **Held-out Test Origins:** `8` chronological weekly origins.
- **Lowest Mean Out-of-Sample Cost Policy (Non-Oracle):** `dfl_pto` (**51.32 MU**, Fill Rate **85.4%**, Mean Oracle Regret **27.59 MU**).

## 1. Primary 7-Policy Out-of-Sample Comparison

| Policy | Mean Cost (MU) | 95% Origin Block-Bootstrap CI (MU) | Realized CVaR_0.90 (MU) | Mean Oracle Regret (MU) | Fill Rate (%) | Mean Unmet Units | Diff vs Equal (MU / %) | Diff vs Supervised PTO (MU / %) | Solver Status |
| :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `equal_allocation` | 53.13 | [35.06, 75.01] | 117.15 | 29.41 | 84.5% | 5.12 | +0.00 (+0.0%) | +1.76 (+3.4%) | `optimal` |
| `proportional_allocation` | 52.16 | [34.48, 76.87] | 131.90 | 28.44 | 85.2% | 5.00 | -0.97 (-1.8%) | +0.79 (+1.5%) | `optimal` |
| `supervised_pto` | 51.37 | [36.11, 70.46] | 109.80 | 27.64 | 85.6% | 4.88 | -1.76 (-3.3%) | +0.00 (+0.0%) | `optimal` |
| `supervised_dro` | 56.91 | [36.46, 84.41] | 132.05 | 33.18 | 81.8% | 5.62 | +3.78 (+7.1%) | +5.54 (+10.8%) | `optimal` |
| `dfl_pto` | 51.32 | [34.71, 71.34] | 109.80 | 27.59 | 85.4% | 4.88 | -1.81 (-3.4%) | -0.05 (-0.1%) | `optimal` |
| `dfl_dro` | 55.04 | [38.89, 78.49] | 124.65 | 31.31 | 82.6% | 5.38 | +1.91 (+3.6%) | +3.67 (+7.1%) | `optimal` |
| `clairvoyant_oracle` | 23.73 | [8.04, 50.41] | 109.05 | 0.00 | 93.2% | 2.75 | -29.41 (-55.3%) | -27.64 (-53.8%) | `optimal` |

## 2. Scientific Verdict & Key Findings

- **Baseline `supervised_pto` achieved lower mean cost than `dfl_dro` on this cohort:** `dfl_dro` averaged **55.04 MU** vs. **51.37 MU** for `supervised_pto` (**+3.67 MU**), while `dfl_pto` differed from `supervised_pto` by **-0.05 MU**.
- **Heuristic Baselines:** `equal_allocation` averaged **53.13 MU** (fill rate **84.5%**) and `proportional_allocation` averaged **52.16 MU** (fill rate **85.2%**).
- **DFL vs. Supervised PTO:** `dfl_pto` averaged **51.32 MU** (regret **27.59 MU**) compared to `supervised_pto` at **51.37 MU** (regret **27.64 MU**).
- **DRO Tail-Risk Protection:** `supervised_dro` achieved realized tail cost $\text{CVaR}_{0.90} = 132.05\text{ MU}$ (validation-selected $\alpha^*=0.90, \text{risk\_weight}^*=0.25$) and `dfl_dro` achieved $\text{CVaR}_{0.90} = 124.65\text{ MU}$ (validation-selected $\alpha^*=0.90, \text{risk\_weight}^*=0.25$, config default $\alpha=0.90, \text{risk\_weight}=0.50$).
- **Statistical Caveat:** Computed via moving-block bootstrap (block_size=2, B=200) across 8 held-out forecast origins. Because 8 < 20 origins is a small time-series sample, intervals are descriptive uncertainty bands and should not be over-interpreted as asymptotic p-values.

## 3. Model Expected Cost vs. Model CVaR vs. Realized Historical Test Cost

Per `AGENTS.md` §7 & §10, model-implied scenario expectations/CVaR and realized held-out test costs are strictly separated:

| Policy | Model-Implied Expected Cost (MU) | Model-Implied CVaR_0.90 (MU) | Realized Held-Out Mean Cost (MU) | Realized Held-Out CVaR_0.90 (MU) |
| :--- | ---: | ---: | ---: | ---: |
| `equal_allocation` | 71.80 | 152.38 | 53.13 | 117.15 |
| `proportional_allocation` | 71.41 | 154.03 | 52.16 | 131.90 |
| `supervised_pto` | 72.37 | 151.05 | 51.37 | 109.80 |
| `supervised_dro` | 69.78 | 141.16 | 56.91 | 132.05 |
| `dfl_pto` | 67.61 | 152.99 | 51.32 | 109.80 |
| `dfl_dro` | 64.46 | 146.83 | 55.04 | 124.65 |
| `clairvoyant_oracle` | 99.71 | 202.59 | 23.73 | 109.05 |

## 4. Decision-Focused Learning (`lambda_DFL`) Validation Ablation

| SKU | `lambda_DFL` | Validation NLL | Validation MAE | Validation WAPE | Validation Decision Regret (MU) | Calibrated Temperature |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `HOUSEHOLD_DEMO_001` | 0.00 | 2.5208 | 2.365 | 0.424 | 10.838 | 2.10 |
| `HOUSEHOLD_DEMO_001` | 0.50 | 2.5918 | 2.282 | 0.409 | 10.825 | 1.20 |
| `HOUSEHOLD_DEMO_001` | 1.00 | 2.5341 | 2.021 | 0.362 | 10.838 | 1.15 |
| `HOUSEHOLD_DEMO_001` | 2.00 | 2.4910 | 1.975 | 0.354 | 9.969 | 1.15 |

## 5. Scenario Calibration Diagnostics & Validation DRO Tuning (Priorities 2 & 3)

- **Validation-Split Scenario Calibration (`split == 'val'`):**
  - `supervised` copula scenarios: raw mean bias `+2.61` units $\rightarrow$ calibrated bias `+0.28` units; Energy Score `7.01` $\rightarrow$ `4.73` (mean scale factor `0.71`).
  - `dfl` (`JointMultiSKUMarginalAllocationModel`) copula scenarios: raw mean bias `+1.86` units $\rightarrow$ calibrated bias `+0.16` units; Energy Score `6.29` $\rightarrow$ `4.78` (mean scale factor `0.76`).
- **Validation-Selected DRO Hyperparameters (`split == 'val'`):**
  - `supervised_dro`: selected $\alpha^* = 0.90, \text{risk\_weight}^* = 0.25$ (config default $\alpha = 0.90, \text{risk\_weight} = 0.50$).
  - `dfl_dro`: selected $\alpha^* = 0.90, \text{risk\_weight}^* = 0.25$ (config default $\alpha = 0.90, \text{risk\_weight} = 0.50$).

## 6. Independent Training-Seed & Scenario-Seed Stability Across Seeds `[42, 101, 202]` (Priority 4 / P1.2)

Each seed independently initializes and trains both `supervised` and `dfl` neural networks (`training_seed_independent = True`) and draws independent copula scenario sets:

| Policy | Mean Cost Across Seeds (MU) | Seed SD (MU) | Mean Regret Across Seeds (MU) | Mean Fill Rate (%) |
| :--- | ---: | ---: | ---: | ---: |
| `clairvoyant_oracle` | 23.73 | 0.00 | 0.00 | 93.2% |
| `dfl_dro` | 54.30 | 1.02 | 30.57 | 83.2% |
| `dfl_pto` | 53.09 | 3.13 | 29.37 | 84.4% |
| `equal_allocation` | 53.13 | 0.00 | 29.41 | 84.5% |
| `proportional_allocation` | 52.16 | 0.00 | 28.44 | 85.2% |
| `supervised_dro` | 55.30 | 1.46 | 31.57 | 82.9% |
| `supervised_pto` | 52.50 | 2.05 | 28.78 | 84.8% |

## 7. Rolling Inventory Carryover & Replenishment Lead-Time Evaluation (Priority 7 / P1.1)

For `clairvoyant_oracle`, the multi-period horizon MILP (`solve_multiperiod_carryover_oracle`) optimizes jointly across $t=0 \dots T-1$ with exact carryover state transitions and shared initial pipeline inventory, guaranteeing nonnegative rolling oracle regret ($\text{Rolling Regret} \ge 0$) at both $L=0$ and $L=1$:

| Lead Time (Weeks) | Policy | Rolling Mean Cost (MU) | Rolling Oracle Regret (MU) | Shortage Cost (MU) | Holding Cost (MU) | Fill Rate (%) | Ending Inventory (Units) |
| :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `equal_allocation` | 50.80 | 36.42 | 10.07 | 31.72 | 93.6% | 52 |
| `0` | `proportional_allocation` | 54.43 | 40.05 | 13.95 | 31.61 | 91.2% | 52 |
| `0` | `supervised_pto` | 45.08 | 30.69 | 9.26 | 27.03 | 94.1% | 45 |
| `0` | `supervised_dro` | 44.32 | 29.94 | 12.50 | 23.55 | 92.2% | 37 |
| `0` | `dfl_pto` | 43.83 | 29.44 | 9.26 | 25.91 | 94.1% | 42 |
| `0` | `dfl_dro` | 44.79 | 30.41 | 13.40 | 23.22 | 91.7% | 36 |
| `0` | `clairvoyant_oracle` | 14.38 | 0.00 | 2.25 | 4.67 | 98.5% | 0 |
| `1` | `equal_allocation` | 47.29 | 28.55 | 10.06 | 28.36 | 93.6% | 48 |
| `1` | `proportional_allocation` | 48.92 | 30.18 | 14.81 | 25.55 | 90.7% | 47 |
| `1` | `supervised_pto` | 48.01 | 29.27 | 12.29 | 26.91 | 92.2% | 49 |
| `1` | `supervised_dro` | 46.59 | 27.86 | 12.43 | 25.69 | 92.2% | 42 |
| `1` | `dfl_pto` | 47.84 | 29.11 | 13.88 | 25.43 | 91.2% | 44 |
| `1` | `dfl_dro` | 48.47 | 29.74 | 13.21 | 26.69 | 91.7% | 45 |
| `1` | `clairvoyant_oracle` | 18.74 | 0.00 | 6.14 | 5.31 | 96.1% | 0 |

## 8. Full-Horizon Cost–Risk–Service-Level Trade-Off Frontier (`P1.3`)

Evaluated across all held-out test origins over $(\alpha, \text{risk\_weight})$:

| Policy Family | `alpha` | `risk_weight` | Realized Mean Cost (MU) | Realized CVaR_0.90 (MU) | Fill Rate (%) | Period No-Stockout Rate (%) | Mean Shortage Cost (MU) | Mean Holding Cost (MU) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dfl_dro` | 0.50 | 0.00 | 55.01 | 124.60 | 83.0% | 68.8% | 33.56 | 12.14 |
| `dfl_dro` | 0.50 | 0.25 | 55.96 | 132.00 | 82.7% | 68.8% | 34.36 | 12.27 |
| `dfl_dro` | 0.50 | 0.50 | 56.86 | 132.00 | 82.3% | 68.8% | 35.12 | 12.39 |
| `dfl_dro` | 0.50 | 0.75 | 55.93 | 132.00 | 82.7% | 68.8% | 34.34 | 12.26 |
| `dfl_dro` | 0.50 | 1.00 | 55.91 | 132.00 | 82.7% | 68.8% | 34.34 | 12.26 |
| `dfl_dro` | 0.75 | 0.00 | 55.01 | 124.60 | 83.0% | 68.8% | 33.56 | 12.14 |
| `dfl_dro` | 0.75 | 0.25 | 56.83 | 124.60 | 82.1% | 68.8% | 35.07 | 12.39 |
| `dfl_dro` | 0.75 | 0.50 | 55.95 | 124.70 | 82.3% | 68.8% | 34.34 | 12.26 |
| `dfl_dro` | 0.75 | 0.75 | 55.99 | 124.70 | 82.4% | 70.8% | 34.35 | 12.27 |
| `dfl_dro` | 0.75 | 1.00 | 58.65 | 124.70 | 80.8% | 68.8% | 36.60 | 12.64 |
| `dfl_dro` | 0.90 | 0.00 | 55.01 | 124.60 | 83.0% | 68.8% | 33.56 | 12.14 |
| `dfl_dro` | 0.90 | 0.25 | 55.04 | 124.65 | 82.6% | 68.8% | 33.59 | 12.14 |
| `dfl_dro` | 0.90 | 0.50 | 56.01 | 124.65 | 81.9% | 68.8% | 34.41 | 12.27 |
| `dfl_dro` | 0.90 | 0.75 | 55.72 | 124.60 | 81.5% | 68.8% | 34.43 | 12.01 |
| `dfl_dro` | 0.90 | 1.00 | 55.38 | 117.10 | 81.2% | 68.8% | 34.41 | 11.75 |
| `dfl_dro` | 0.95 | 0.00 | 55.01 | 124.60 | 83.0% | 68.8% | 33.56 | 12.14 |
| `dfl_dro` | 0.95 | 0.25 | 54.13 | 124.65 | 83.2% | 70.8% | 32.81 | 12.01 |
| `dfl_dro` | 0.95 | 0.50 | 54.56 | 124.60 | 82.8% | 70.8% | 33.54 | 11.74 |
| `dfl_dro` | 0.95 | 0.75 | 59.78 | 124.65 | 77.6% | 72.9% | 40.79 | 10.44 |
| `dfl_dro` | 0.95 | 1.00 | 70.07 | 124.55 | 70.9% | 60.4% | 50.18 | 11.46 |
| `supervised_dro` | 0.50 | 0.00 | 52.28 | 124.65 | 84.5% | 70.8% | 31.22 | 11.75 |
| `supervised_dro` | 0.50 | 0.25 | 55.03 | 132.05 | 83.2% | 72.9% | 33.58 | 12.14 |
| `supervised_dro` | 0.50 | 0.50 | 57.71 | 132.05 | 81.5% | 70.8% | 35.86 | 12.51 |
| `supervised_dro` | 0.50 | 0.75 | 58.64 | 131.95 | 80.7% | 66.7% | 36.64 | 12.64 |
| `supervised_dro` | 0.50 | 1.00 | 58.64 | 131.95 | 80.7% | 66.7% | 36.64 | 12.64 |
| `supervised_dro` | 0.75 | 0.00 | 52.28 | 124.65 | 84.5% | 70.8% | 31.22 | 11.75 |
| `supervised_dro` | 0.75 | 0.25 | 55.89 | 124.60 | 82.6% | 70.8% | 34.28 | 12.26 |
| `supervised_dro` | 0.75 | 0.50 | 55.88 | 124.60 | 82.6% | 70.8% | 34.23 | 12.25 |
| `supervised_dro` | 0.75 | 0.75 | 56.82 | 124.60 | 82.1% | 68.8% | 35.03 | 12.38 |
| `supervised_dro` | 0.75 | 1.00 | 60.44 | 124.60 | 80.2% | 66.7% | 38.04 | 12.89 |
| `supervised_dro` | 0.90 | 0.00 | 52.28 | 124.65 | 84.5% | 70.8% | 31.22 | 11.75 |
| `supervised_dro` | 0.90 | 0.25 | 56.91 | 132.05 | 81.8% | 70.8% | 35.21 | 12.41 |
| `supervised_dro` | 0.90 | 0.50 | 57.85 | 132.05 | 81.4% | 72.9% | 36.01 | 12.54 |
| `supervised_dro` | 0.90 | 0.75 | 57.81 | 124.65 | 81.3% | 70.8% | 35.95 | 12.53 |
| `supervised_dro` | 0.90 | 1.00 | 61.54 | 124.55 | 78.9% | 66.7% | 39.20 | 13.06 |
| `supervised_dro` | 0.95 | 0.00 | 52.28 | 124.65 | 84.5% | 70.8% | 31.22 | 11.75 |
| `supervised_dro` | 0.95 | 0.25 | 56.91 | 124.65 | 81.5% | 72.9% | 35.19 | 12.41 |
| `supervised_dro` | 0.95 | 0.50 | 55.99 | 117.30 | 81.8% | 72.9% | 34.39 | 12.28 |
| `supervised_dro` | 0.95 | 0.75 | 58.71 | 117.35 | 79.5% | 70.8% | 37.62 | 12.01 |
| `supervised_dro` | 0.95 | 1.00 | 63.76 | 117.25 | 75.8% | 64.6% | 42.46 | 12.41 |

## 10. Methodological Caveats & Limitations

1. **Observed Sales != Latent Demand:** Even when run on M5 data, observed unit sales are right-censored by historical stockouts; in `synthetic_demo` mode (`DEMO / SYNTHETIC`), demand is generated from a documented Poisson-Gamma weekly process with shared cross-store shocks.
2. **Synthetic Operating Economics:** Opening stock, warehouse supply caps, store capacities, and unit costs (`MU`) are explicitly synthetic (`data_provenance='synthetic'`).
3. **Explicit Ambiguity Set:** DRO policies optimize over the scenario-probability reweighting ambiguity set $\mathcal{P}_\alpha = \{ w \ge 0 : \sum_j w_j = 1, w_j \le p_j / (1-\alpha) \}$ (equivalent to Rockafellar–Uryasev $\text{CVaR}_\alpha$), **not** Wasserstein DRO.
