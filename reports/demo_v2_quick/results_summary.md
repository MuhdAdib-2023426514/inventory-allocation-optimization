# Inventory experiment results

Run: `demo_v2_quick` | Data: `synthetic_demo` | Economics: synthetic MU

Retrospective corrected rerun on previously inspected test periods; not a fresh confirmatory holdout

Computed via moving-block bootstrap (block_size=2, B=50) across 8 held-out forecast origins. Because 8 < 20 origins is a small time-series sample, intervals are descriptive uncertainty bands and should not be over-interpreted as asymptotic p-values.

Lowest recorded non-oracle mean cost: **supervised_pto**, **48.62 MU**.
This is an observed ranking on this sample; it does not establish general superiority.

## Held-out results

| Policy | Mean cost MU | CVaR90 MU | Fill % | Regret MU | Solve s |
|---|---:|---:|---:|---:|---:|
| equal_allocation | 53.13 | 117.15 | 84.5 | 29.41 | 0.0000 |
| proportional_allocation | 52.16 | 131.90 | 85.2 | 28.44 | 0.0000 |
| supervised_pto | 48.62 | 109.85 | 87.0 | 24.89 | 0.0042 |
| supervised_dro | 61.47 | 124.70 | 78.5 | 37.74 | 0.0060 |
| dfl_pto | 53.16 | 109.80 | 84.4 | 29.44 | 0.0041 |
| dfl_dro | 54.92 | 117.20 | 83.4 | 31.19 | 0.0058 |
| clairvoyant_oracle | 23.73 | 109.05 | 93.2 | 0.00 | 0.0007 |
| supervised_stochastic | 53.21 | 117.25 | 84.0 | 29.48 | 0.0047 |
| dfl_stochastic | 54.69 | 117.15 | 83.2 | 30.97 | 0.0048 |

## Paired differences and uncertainty

Negative differences favor the named policy. Intervals use paired chronological origin blocks.

| Policy | Difference vs supervised PTO MU | Paired 95% interval MU | Difference vs proportional MU | Paired 95% interval MU |
|---|---:|---|---:|---|
| equal_allocation | +4.51 | [-0.62, 13.00] | +0.97 | [-5.28, 4.49] |
| proportional_allocation | +3.54 | [-2.69, 12.76] | +0.00 | [0.00, 0.00] |
| supervised_pto | +0.00 | [0.00, 0.00] | -3.54 | [-12.76, 2.69] |
| supervised_dro | +12.85 | [8.50, 20.41] | +9.31 | [6.61, 13.57] |
| dfl_pto | +4.54 | [1.14, 8.63] | +1.00 | [-10.22, 6.64] |
| dfl_dro | +6.30 | [3.54, 11.17] | +2.76 | [-3.14, 9.07] |
| clairvoyant_oracle | -24.89 | [-29.33, -14.09] | -28.44 | [-36.50, -23.27] |
| supervised_stochastic | +4.59 | [2.03, 7.45] | +1.04 | [-6.66, 6.09] |
| dfl_stochastic | +6.07 | [3.30, 11.02] | +2.53 | [-1.64, 6.93] |

## Independent equal-budget seeds

Full numerical table: `artifacts/demo_v2_quick/multi_seed_summary.parquet` (27 rows).


## Carryover with dispatch/admission oracle

Full numerical table: `artifacts/demo_v2_quick/carryover_leadtime_metrics.parquet` (18 rows).

| Lead time | Policy | Mean cost MU | Regret MU |
|---|---|---:|---:|
| 0 | equal_allocation | 50.80 | 36.42 |
| 0 | proportional_allocation | 54.43 | 40.05 |
| 0 | supervised_pto | 45.51 | 31.13 |
| 0 | supervised_dro | 48.50 | 34.12 |
| 0 | dfl_pto | 43.67 | 29.29 |
| 0 | dfl_dro | 46.78 | 32.39 |
| 0 | clairvoyant_oracle | 14.38 | 0.00 |
| 0 | supervised_stochastic | 47.55 | 33.17 |
| 0 | dfl_stochastic | 44.89 | 30.51 |
| 1 | equal_allocation | 48.59 | 31.01 |
| 1 | proportional_allocation | 53.56 | 35.98 |
| 1 | supervised_pto | 52.38 | 34.81 |
| 1 | supervised_dro | 59.20 | 41.62 |
| 1 | dfl_pto | 50.65 | 33.08 |
| 1 | dfl_dro | 52.93 | 35.36 |
| 1 | clairvoyant_oracle | 17.57 | 0.00 |
| 1 | supervised_stochastic | 56.21 | 38.64 |
| 1 | dfl_stochastic | 54.98 | 37.41 |

## Frozen-model distribution sensitivity

Full numerical table: `artifacts/demo_v2_quick/distribution_sensitivity.parquet` (30 rows).


## Interpretation and limits

- Seven required policies plus two risk-neutral scenario ablations. Heuristic baselines are verified feasible, not certified cost-optimal.
- Supervised and DFL use matched architectures, initialization, batches, budgets and checkpoint rules. Only lambda differs in the controlled training ablation.
- PMF temperature is fitted on validation. Nominal and scenario methods share those marginals and the shifted-geometric tail; no scenario-only rescaling is applied.
- Model-implied risk uses common supervised reference scenarios at alpha 0.90 for comparison. Optimization uses each family's PMF/scenarios and validation-selected risk parameters.
- Reweighting ambiguity set: sum(w)=1, 0 <= w_j <= p_j/(1-alpha); its worst expected loss equals CVaR. This is not Wasserstein DRO.
- Sales are proxies for demand and may be censored. Inventory and operating economics are synthetic; no actual retailer savings are claimed.
- Short test horizons yield uncertain paired means and especially unstable realized tail estimates. Test frontiers are descriptive, not tuning evidence.
- Carryover replenishes the synthetic warehouse dispatch budget each week. Store admissions are explicit actions; the hindsight oracle can choose admissions with future knowledge and bounds all causal admissions.
- Terminal stock has no salvage value. Deferred stock incurs holding; scheduled pipeline stock has no holding charge. End-of-horizon orders that cannot arrive are zero.
- Lead-time feature projection is a causal approximation, not a separately trained lead-time forecasting model.
- Training duration: 9.56 s. Backtest duration including seed retraining/tuning: 31.46 s.
