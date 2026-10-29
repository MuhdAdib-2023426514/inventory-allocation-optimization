# Inventory experiment results

Run: `m5_v2` | Data: `m5_sales_proxy` | Economics: synthetic MU

Retrospective corrected rerun on previously inspected test periods; not a fresh confirmatory holdout

Computed via moving-block bootstrap (block_size=2, B=2000) across 12 held-out forecast origins. Because 12 < 20 origins is a small time-series sample, intervals are descriptive uncertainty bands and should not be over-interpreted as asymptotic p-values.

Primary seed 42; mean across held-out origins. Lowest recorded non-oracle mean cost: **proportional_allocation**, **1298.91 MU**.
This is an observed ranking on this sample; it does not establish general superiority.

## Held-out results

| Policy | Mean cost MU | CVaR90 MU | Fill % | Regret MU | Solve s |
|---|---:|---:|---:|---:|---:|
| equal_allocation | 1432.35 | 1843.00 | 71.2 | 412.20 | 0.0003 |
| proportional_allocation | 1298.91 | 1711.69 | 73.9 | 278.76 | 0.0002 |
| supervised_pto | 1311.54 | 1688.00 | 73.7 | 291.39 | 0.0200 |
| supervised_dro | 1320.18 | 1698.36 | 73.5 | 300.03 | 0.1045 |
| dfl_pto | 1305.72 | 1674.74 | 73.8 | 285.57 | 0.0188 |
| dfl_dro | 1311.75 | 1684.01 | 73.7 | 291.60 | 0.1097 |
| clairvoyant_oracle | 1020.15 | 1536.59 | 79.2 | 0.00 | 0.0048 |
| supervised_stochastic | 1318.53 | 1688.55 | 73.5 | 298.38 | 0.0607 |
| dfl_stochastic | 1313.70 | 1686.89 | 73.6 | 293.55 | 0.0636 |

## Paired differences and uncertainty

Negative differences favor the named policy. Intervals use paired chronological origin blocks.

| Policy | Difference vs supervised PTO MU | Paired 95% interval MU | Difference vs proportional MU | Paired 95% interval MU |
|---|---:|---|---:|---|
| equal_allocation | +120.80 | [87.62, 147.76] | +133.43 | [113.76, 155.30] |
| proportional_allocation | -12.63 | [-48.66, 8.77] | +0.00 | [0.00, 0.00] |
| supervised_pto | +0.00 | [0.00, 0.00] | +12.63 | [-8.77, 48.66] |
| supervised_dro | +8.64 | [-0.34, 16.80] | +21.27 | [-6.72, 60.97] |
| dfl_pto | -5.82 | [-20.80, 5.32] | +6.80 | [-14.85, 44.78] |
| dfl_dro | +0.20 | [-15.71, 10.66] | +12.83 | [-16.65, 54.28] |
| clairvoyant_oracle | -291.39 | [-323.29, -277.00] | -278.76 | [-308.84, -260.34] |
| supervised_stochastic | +6.99 | [-1.10, 12.81] | +19.62 | [-6.66, 57.53] |
| dfl_stochastic | +2.15 | [-12.86, 13.43] | +14.78 | [-15.39, 56.13] |

## Independent equal-budget seeds

Full numerical table: `artifacts/m5_v2/multi_seed_summary.parquet` (27 rows).

Between-seed sample standard deviation describes initialization/scenario variation on the same test period, not independent time-series evidence.

| Policy | Mean across seeds MU | Between-seed SD MU | Minimum MU | Maximum MU |
|---|---:|---:|---:|---:|
| clairvoyant_oracle | 1020.15 | 0.00 | 1020.15 | 1020.15 |
| dfl_dro | 1318.10 | 5.99 | 1311.75 | 1323.65 |
| dfl_pto | 1309.27 | 3.89 | 1305.72 | 1313.43 |
| dfl_stochastic | 1320.71 | 6.08 | 1313.70 | 1324.42 |
| equal_allocation | 1432.35 | 0.00 | 1432.35 | 1432.35 |
| proportional_allocation | 1298.91 | 0.00 | 1298.91 | 1298.91 |
| supervised_dro | 1320.67 | 1.66 | 1319.31 | 1322.53 |
| supervised_pto | 1307.78 | 4.99 | 1302.12 | 1311.54 |
| supervised_stochastic | 1321.18 | 2.55 | 1318.53 | 1323.63 |

## Carryover with dispatch/admission oracle

Full numerical table: `artifacts/m5_v2/carryover_leadtime_metrics.parquet` (18 rows).

| Lead time | Policy | Mean cost MU | Regret MU |
|---|---|---:|---:|
| 0 | equal_allocation | 2080.03 | 1095.12 |
| 0 | proportional_allocation | 1620.85 | 635.94 |
| 0 | supervised_pto | 1214.79 | 229.89 |
| 0 | supervised_dro | 1218.40 | 233.50 |
| 0 | dfl_pto | 1208.89 | 223.98 |
| 0 | dfl_dro | 1209.43 | 224.52 |
| 0 | clairvoyant_oracle | 984.90 | 0.00 |
| 0 | supervised_stochastic | 1214.30 | 229.40 |
| 0 | dfl_stochastic | 1212.08 | 227.18 |
| 1 | equal_allocation | 1986.02 | 1010.67 |
| 1 | proportional_allocation | 1638.27 | 662.91 |
| 1 | supervised_pto | 1421.36 | 446.00 |
| 1 | supervised_dro | 1440.18 | 464.82 |
| 1 | dfl_pto | 1401.18 | 425.82 |
| 1 | dfl_dro | 1408.50 | 433.14 |
| 1 | clairvoyant_oracle | 975.35 | 0.00 |
| 1 | supervised_stochastic | 1435.68 | 460.33 |
| 1 | dfl_stochastic | 1402.57 | 427.21 |

## Frozen-model distribution sensitivity

Full numerical table: `artifacts/m5_v2/distribution_sensitivity.parquet` (30 rows).


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
- Training duration: 1062.08 s. Backtest duration including seed retraining/tuning: 2920.11 s.
