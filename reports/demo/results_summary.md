# Inventory Allocation Experiment Results Summary

- **Data Provenance / Mode:** DEMO / SYNTHETIC (`data_mode='synthetic_demo'`)
- **Run ID:** `demo_v1` (Global Seed: `42`, Git Commit: `89a083b`)
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
- **DRO Tail-Risk Protection:** `supervised_dro` achieved realized tail cost $\text{CVaR}_{0.90} = 132.05\text{ MU}$ and `dfl_dro` achieved $\text{CVaR}_{0.90} = 124.65\text{ MU}$ (under probability-reweighting ambiguity set $\mathcal{P}_\alpha$ with $\alpha=0.90, \text{risk\_weight}=0.50$).
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
  - `supervised_dro`: selected $\alpha^* = 0.90, \text{risk\_weight}^* = 0.25$.
  - `dfl_dro`: selected $\alpha^* = 0.90, \text{risk\_weight}^* = 0.25$.

## 6. Multi-Seed Stability Across Seeds `[42, 101, 202]` (Priority 4)

| Policy | Mean Cost Across Seeds (MU) | Seed SD (MU) | Mean Fill Rate (%) |
| :--- | ---: | ---: | ---: |
| `clairvoyant_oracle` | 23.73 | 0.00 | 93.2% |
| `dfl_dro` | 53.78 | 2.91 | 83.5% |
| `dfl_pto` | 51.32 | 0.00 | 85.4% |
| `equal_allocation` | 53.13 | 0.00 | 84.5% |
| `proportional_allocation` | 52.16 | 0.00 | 85.2% |
| `supervised_dro` | 55.61 | 2.11 | 82.5% |
| `supervised_pto` | 51.37 | 0.00 | 85.6% |

## 7. Rolling Inventory Carryover & Replenishment Lead-Time Evaluation (Priority 7)

| Lead Time (Weeks) | Policy | Rolling Mean Cost (MU) | Shortage Cost (MU) | Holding Cost (MU) | Fill Rate (%) | Ending Inventory (Units) |
| :---: | :--- | ---: | ---: | ---: | ---: | ---: |
| `0` | `equal_allocation` | 50.80 | 10.07 | 31.72 | 93.6% | 52 |
| `0` | `proportional_allocation` | 54.43 | 13.95 | 31.61 | 91.2% | 52 |
| `0` | `supervised_pto` | 45.08 | 9.26 | 27.03 | 94.1% | 45 |
| `0` | `supervised_dro` | 44.32 | 12.50 | 23.55 | 92.2% | 37 |
| `0` | `dfl_pto` | 43.83 | 9.26 | 25.91 | 94.1% | 42 |
| `0` | `dfl_dro` | 44.79 | 13.40 | 23.22 | 91.7% | 36 |
| `0` | `clairvoyant_oracle` | 23.72 | 16.90 | 0.00 | 89.2% | 0 |
| `1` | `equal_allocation` | 65.07 | 33.26 | 24.11 | 78.9% | 45 |
| `1` | `proportional_allocation` | 67.31 | 37.17 | 22.67 | 76.5% | 44 |
| `1` | `supervised_pto` | 69.06 | 38.45 | 23.03 | 75.5% | 47 |
| `1` | `supervised_dro` | 67.93 | 36.94 | 23.41 | 76.5% | 46 |
| `1` | `dfl_pto` | 69.71 | 39.97 | 22.28 | 74.5% | 46 |
| `1` | `dfl_dro` | 69.87 | 39.39 | 23.02 | 75.0% | 47 |
| `1` | `clairvoyant_oracle` | 78.93 | 61.83 | 11.73 | 60.8% | 20 |

## 8. Methodological Caveats & Limitations

1. **Observed Sales != Latent Demand:** Even when run on M5 data, observed unit sales are right-censored by historical stockouts; in `synthetic_demo` mode (`DEMO / SYNTHETIC`), demand is generated from a documented Poisson-Gamma weekly process with shared cross-store shocks.
2. **Synthetic Operating Economics:** Opening stock, warehouse supply caps, store capacities, and unit costs (`MU`) are explicitly synthetic (`data_provenance='synthetic'`).
3. **Explicit Ambiguity Set:** DRO policies optimize over the scenario-probability reweighting ambiguity set $\mathcal{P}_\alpha = \{ w \ge 0 : \sum_j w_j = 1, w_j \le p_j / (1-\alpha) \}$ (equivalent to Rockafellar–Uryasev $\text{CVaR}_\alpha$), **not** Wasserstein DRO.
