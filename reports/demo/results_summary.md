# Inventory Allocation Experiment Results Summary

- **Data Provenance / Mode:** DEMO / SYNTHETIC (`data_mode='synthetic_demo'`)
- **Run ID:** `demo_v1` (Global Seed: `42`, Git Commit: `uncommitted`)
- **Economic Units:** Generic Monetary Units (`MU`), synthetic inventory/operating constraints (`data_provenance='synthetic'`).
- **Held-out Test Origins:** `8` chronological weekly origins.
- **Lowest Mean Out-of-Sample Cost Policy (Non-Oracle):** `dfl_pto` (**50.48 MU**, Fill Rate **86.1%**, Mean Oracle Regret **26.75 MU**).

## 1. Primary 7-Policy Out-of-Sample Comparison

| Policy | Mean Cost (MU) | 95% Origin Block-Bootstrap CI (MU) | Realized CVaR_0.90 (MU) | Mean Oracle Regret (MU) | Fill Rate (%) | Mean Unmet Units | Diff vs Equal (MU / %) | Diff vs Supervised PTO (MU / %) | Solver Status |
| :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `equal_allocation` | 53.13 | [35.06, 75.01] | 117.15 | 29.41 | 84.5% | 5.12 | +0.00 (+0.0%) | +1.76 (+3.4%) | `optimal` |
| `proportional_allocation` | 52.16 | [34.48, 76.87] | 131.90 | 28.44 | 85.2% | 5.00 | -0.97 (-1.8%) | +0.79 (+1.5%) | `optimal` |
| `supervised_pto` | 51.37 | [36.11, 70.46] | 109.80 | 27.64 | 85.6% | 4.88 | -1.76 (-3.3%) | +0.00 (+0.0%) | `optimal` |
| `supervised_dro` | 55.98 | [40.02, 79.86] | 124.65 | 32.26 | 82.5% | 5.50 | +2.85 (+5.4%) | +4.61 (+9.0%) | `optimal` |
| `dfl_pto` | 50.48 | [35.20, 70.47] | 109.90 | 26.75 | 86.1% | 4.75 | -2.66 (-5.0%) | -0.89 (-1.7%) | `optimal` |
| `dfl_dro` | 62.33 | [44.74, 86.20] | 124.60 | 38.60 | 78.9% | 6.38 | +9.19 (+17.3%) | +10.96 (+21.3%) | `optimal` |
| `clairvoyant_oracle` | 23.73 | [8.04, 50.41] | 109.05 | 0.00 | 93.2% | 2.75 | -29.41 (-55.3%) | -27.64 (-53.8%) | `optimal` |

## 2. Scientific Verdict & Key Findings

- **Baseline `supervised_pto` achieved lower mean cost than `dfl_dro` on this cohort:** `dfl_dro` averaged **62.33 MU** vs. **51.37 MU** for `supervised_pto` (**+10.96 MU**), while `dfl_pto` differed from `supervised_pto` by **-0.89 MU**.
- **Heuristic Baselines:** `equal_allocation` averaged **53.13 MU** (fill rate **84.5%**) and `proportional_allocation` averaged **52.16 MU** (fill rate **85.2%**).
- **DFL vs. Supervised PTO:** `dfl_pto` averaged **50.48 MU** (regret **26.75 MU**) compared to `supervised_pto` at **51.37 MU** (regret **27.64 MU**).
- **DRO Tail-Risk Protection:** `supervised_dro` achieved realized tail cost $\text{CVaR}_{0.90} = 124.65\text{ MU}$ and `dfl_dro` achieved $\text{CVaR}_{0.90} = 124.60\text{ MU}$ (under probability-reweighting ambiguity set $\mathcal{P}_\alpha$ with $\alpha=0.90, \text{risk\_weight}=0.50$).
- **Statistical Caveat:** Computed via moving-block bootstrap (block_size=2, B=200) across 8 held-out forecast origins. Because 8 < 20 origins is a small time-series sample, intervals are descriptive uncertainty bands and should not be over-interpreted as asymptotic p-values.

## 3. Model Expected Cost vs. Model CVaR vs. Realized Historical Test Cost

Per `AGENTS.md` §7 & §10, model-implied scenario expectations/CVaR and realized held-out test costs are strictly separated:

| Policy | Model-Implied Expected Cost (MU) | Model-Implied CVaR_0.90 (MU) | Realized Held-Out Mean Cost (MU) | Realized Held-Out CVaR_0.90 (MU) |
| :--- | ---: | ---: | ---: | ---: |
| `equal_allocation` | 134.55 | 257.33 | 53.13 | 117.15 |
| `proportional_allocation` | 135.31 | 256.79 | 52.16 | 131.90 |
| `supervised_pto` | 134.37 | 255.96 | 51.37 | 109.80 |
| `supervised_dro` | 134.65 | 247.93 | 55.98 | 124.65 |
| `dfl_pto` | 115.26 | 230.63 | 50.48 | 109.90 |
| `dfl_dro` | 115.17 | 219.55 | 62.33 | 124.60 |
| `clairvoyant_oracle` | 167.55 | 308.25 | 23.73 | 109.05 |

## 4. Decision-Focused Learning (`lambda_DFL`) Validation Ablation

| SKU | `lambda_DFL` | Validation NLL | Validation MAE | Validation WAPE | Validation Decision Regret (MU) | Calibrated Temperature |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `HOUSEHOLD_DEMO_001` | 0.00 | 2.5208 | 2.365 | 0.424 | 10.838 | 2.10 |
| `HOUSEHOLD_DEMO_001` | 0.50 | 2.5918 | 2.282 | 0.409 | 10.825 | 1.20 |
| `HOUSEHOLD_DEMO_001` | 1.00 | 2.5341 | 2.021 | 0.362 | 10.838 | 1.15 |
| `HOUSEHOLD_DEMO_001` | 2.00 | 2.4910 | 1.975 | 0.354 | 9.969 | 1.15 |

## 5. Methodological Caveats & Limitations

1. **Observed Sales != Latent Demand:** Even when run on M5 data, observed unit sales are right-censored by historical stockouts; in `synthetic_demo` mode (`DEMO / SYNTHETIC`), demand is generated from a documented Poisson-Gamma weekly process with shared cross-store shocks.
2. **Synthetic Operating Economics:** Opening stock, warehouse supply caps, store capacities, and unit costs (`MU`) are explicitly synthetic (`data_provenance='synthetic'`).
3. **Explicit Ambiguity Set:** DRO policies optimize over the scenario-probability reweighting ambiguity set $\mathcal{P}_\alpha = \{ w \ge 0 : \sum_j w_j = 1, w_j \le p_j / (1-\alpha) \}$ (equivalent to Rockafellar–Uryasev $\text{CVaR}_\alpha$), **not** Wasserstein DRO.
