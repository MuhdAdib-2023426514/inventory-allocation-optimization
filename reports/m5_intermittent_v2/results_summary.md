# Inventory experiment results

Run: `m5_intermittent_v2` | Data: `m5_sales_proxy` | Economics: synthetic MU

Retrospective corrected rerun on previously inspected test periods; not a fresh confirmatory holdout

Computed via moving-block bootstrap (block_size=2, B=2000) across 12 held-out forecast origins. Because 12 < 20 origins is a small time-series sample, intervals are descriptive uncertainty bands and should not be over-interpreted as asymptotic p-values.

Lowest recorded non-oracle mean cost: **supervised_dro**, **168.89 MU**.
This is an observed ranking on this sample; it does not establish general superiority.

## Held-out results

| Policy | Mean cost MU | CVaR90 MU | Fill % | Regret MU | Solve s |
|---|---:|---:|---:|---:|---:|
| equal_allocation | 174.45 | 269.21 | 56.2 | 84.60 | 0.0001 |
| proportional_allocation | 176.94 | 274.42 | 55.3 | 87.09 | 0.0002 |
| supervised_pto | 169.94 | 261.35 | 57.3 | 80.09 | 0.0052 |
| supervised_dro | 168.89 | 268.03 | 56.4 | 79.04 | 0.1503 |
| dfl_pto | 179.67 | 266.57 | 53.6 | 89.82 | 0.0055 |
| dfl_dro | 179.11 | 265.06 | 53.3 | 89.26 | 0.0859 |
| clairvoyant_oracle | 89.85 | 160.78 | 73.7 | 0.00 | 0.0074 |
| supervised_stochastic | 169.15 | 248.45 | 57.5 | 79.30 | 0.0305 |
| dfl_stochastic | 180.29 | 262.45 | 52.9 | 90.44 | 0.0287 |

## Paired differences and uncertainty

Negative differences favor the named policy. Intervals use paired chronological origin blocks.

| Policy | Difference vs supervised PTO MU | Paired 95% interval MU | Difference vs proportional MU | Paired 95% interval MU |
|---|---:|---|---:|---|
| equal_allocation | +4.51 | [-0.55, 8.23] | -2.50 | [-7.25, 1.38] |
| proportional_allocation | +7.00 | [2.49, 11.43] | +0.00 | [0.00, 0.00] |
| supervised_pto | +0.00 | [0.00, 0.00] | -7.00 | [-11.43, -2.49] |
| supervised_dro | -1.05 | [-10.21, 7.16] | -8.05 | [-19.20, 2.07] |
| dfl_pto | +9.73 | [4.39, 14.78] | +2.73 | [-3.11, 9.51] |
| dfl_dro | +9.17 | [0.67, 16.05] | +2.17 | [-6.98, 11.00] |
| clairvoyant_oracle | -80.09 | [-88.93, -69.55] | -87.09 | [-98.45, -74.50] |
| supervised_stochastic | -0.78 | [-3.67, 2.49] | -7.79 | [-12.93, -3.13] |
| dfl_stochastic | +10.35 | [4.59, 14.95] | +3.35 | [-1.49, 8.43] |

## Independent equal-budget seeds

Full numerical table: `artifacts/m5_intermittent_v2/multi_seed_summary.parquet` (27 rows).


## Carryover with dispatch/admission oracle

Full numerical table: `artifacts/m5_intermittent_v2/carryover_leadtime_metrics.parquet` (18 rows).

| Lead time | Policy | Mean cost MU | Regret MU |
|---|---|---:|---:|
| 0 | equal_allocation | 241.20 | 165.14 |
| 0 | proportional_allocation | 240.44 | 164.38 |
| 0 | supervised_pto | 145.53 | 69.46 |
| 0 | supervised_dro | 151.10 | 75.04 |
| 0 | dfl_pto | 153.37 | 77.30 |
| 0 | dfl_dro | 158.72 | 82.66 |
| 0 | clairvoyant_oracle | 76.06 | 0.00 |
| 0 | supervised_stochastic | 146.19 | 70.12 |
| 0 | dfl_stochastic | 155.30 | 79.24 |
| 1 | equal_allocation | 235.85 | 149.00 |
| 1 | proportional_allocation | 242.07 | 155.21 |
| 1 | supervised_pto | 165.55 | 78.70 |
| 1 | supervised_dro | 173.27 | 86.42 |
| 1 | dfl_pto | 178.65 | 91.80 |
| 1 | dfl_dro | 186.64 | 99.78 |
| 1 | clairvoyant_oracle | 86.85 | 0.00 |
| 1 | supervised_stochastic | 169.24 | 82.39 |
| 1 | dfl_stochastic | 183.95 | 97.10 |

## Frozen-model distribution sensitivity

Full numerical table: `artifacts/m5_intermittent_v2/distribution_sensitivity.parquet` (30 rows).


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
- Training duration: 101.08 s. Backtest duration including seed retraining/tuning: 753.26 s.
