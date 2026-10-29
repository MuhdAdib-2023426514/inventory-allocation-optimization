# Inventory experiment results

Run: `demo_v2` | Data: `synthetic_demo` | Economics: synthetic MU

Retrospective corrected rerun on previously inspected test periods; not a fresh confirmatory holdout

Computed via moving-block bootstrap (block_size=2, B=2000) across 8 held-out forecast origins. Because 8 < 20 origins is a small time-series sample, intervals are descriptive uncertainty bands and should not be over-interpreted as asymptotic p-values.

Lowest recorded non-oracle mean cost: **supervised_pto**, **48.62 MU**.
This is an observed ranking on this sample; it does not establish general superiority.

## Held-out results

| Policy | Mean cost MU | CVaR90 MU | Fill % | Regret MU | Solve s |
|---|---:|---:|---:|---:|---:|
| equal_allocation | 53.13 | 117.15 | 84.5 | 29.41 | 0.0000 |
| proportional_allocation | 52.16 | 131.90 | 85.2 | 28.44 | 0.0000 |
| supervised_pto | 48.62 | 109.85 | 87.0 | 24.89 | 0.0039 |
| supervised_dro | 57.77 | 117.20 | 80.9 | 34.04 | 0.0089 |
| dfl_pto | 53.16 | 109.80 | 84.4 | 29.44 | 0.0037 |
| dfl_dro | 53.58 | 117.10 | 83.9 | 29.86 | 0.0071 |
| clairvoyant_oracle | 23.73 | 109.05 | 93.2 | 0.00 | 0.0006 |
| supervised_stochastic | 53.12 | 117.20 | 84.0 | 29.39 | 0.0062 |
| dfl_stochastic | 51.77 | 117.10 | 84.8 | 28.04 | 0.0064 |

## Paired differences and uncertainty

Negative differences favor the named policy. Intervals use paired chronological origin blocks.

| Policy | Difference vs supervised PTO MU | Paired 95% interval MU | Difference vs proportional MU | Paired 95% interval MU |
|---|---:|---|---:|---|
| equal_allocation | +4.51 | [-1.68, 10.74] | +0.97 | [-4.55, 4.64] |
| proportional_allocation | +3.54 | [-1.97, 11.83] | +0.00 | [0.00, 0.00] |
| supervised_pto | +0.00 | [0.00, 0.00] | -3.54 | [-11.83, 1.97] |
| supervised_dro | +9.15 | [5.43, 13.79] | +5.61 | [-0.76, 12.89] |
| dfl_pto | +4.54 | [1.83, 8.03] | +1.00 | [-8.22, 6.54] |
| dfl_dro | +4.96 | [1.71, 9.16] | +1.42 | [-4.96, 7.32] |
| clairvoyant_oracle | -24.89 | [-30.64, -15.72] | -28.44 | [-34.64, -22.15] |
| supervised_stochastic | +4.50 | [1.82, 7.18] | +0.96 | [-6.39, 7.28] |
| dfl_stochastic | +3.15 | [0.38, 7.36] | -0.39 | [-5.45, 4.60] |

## Independent equal-budget seeds

Full numerical table: `artifacts/demo_v2/multi_seed_summary.parquet` (27 rows).


## Carryover with dispatch/admission oracle

Full numerical table: `artifacts/demo_v2/carryover_leadtime_metrics.parquet` (18 rows).

| Lead time | Policy | Mean cost MU | Regret MU |
|---|---|---:|---:|
| 0 | equal_allocation | 50.80 | 36.42 |
| 0 | proportional_allocation | 54.43 | 40.05 |
| 0 | supervised_pto | 45.51 | 31.13 |
| 0 | supervised_dro | 47.55 | 33.17 |
| 0 | dfl_pto | 43.67 | 29.29 |
| 0 | dfl_dro | 48.77 | 34.39 |
| 0 | clairvoyant_oracle | 14.38 | 0.00 |
| 0 | supervised_stochastic | 41.67 | 27.29 |
| 0 | dfl_stochastic | 44.63 | 30.25 |
| 1 | equal_allocation | 48.59 | 31.01 |
| 1 | proportional_allocation | 53.56 | 35.98 |
| 1 | supervised_pto | 52.38 | 34.81 |
| 1 | supervised_dro | 57.00 | 39.42 |
| 1 | dfl_pto | 50.65 | 33.08 |
| 1 | dfl_dro | 49.46 | 31.89 |
| 1 | clairvoyant_oracle | 17.57 | 0.00 |
| 1 | supervised_stochastic | 55.30 | 37.72 |
| 1 | dfl_stochastic | 50.76 | 33.18 |

## Frozen-model distribution sensitivity

Full numerical table: `artifacts/demo_v2/distribution_sensitivity.parquet` (30 rows).


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
- Training duration: 18.60 s. Backtest duration including seed retraining/tuning: 58.52 s.
