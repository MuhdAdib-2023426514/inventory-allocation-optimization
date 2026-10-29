# Catalog of Temporal & Target Leakage Vectors (`data-leakage-audit`)

Reference for [`AGENTS.md`](../../../AGENTS.md) §1.2, §4, §6, §7, §8, §9, and §14.

| Stage | Forbidden Leakage Pattern | Required Causal Rule |
| :--- | :--- | :--- |
| **1. Cohort Selection** | Selecting top-selling or low-variance SKUs/stores across all 1941 M5 days. | Filter `cat_id == "HOUSEHOLD"` and rank/select stores & SKUs using `split == "train"` dates only. |
| **2. Time Splitting** | Random row splitting (`shuffle=True`) or per-SKU differing split dates. | Split chronologically by whole `forecast_origin` blocks: all store–SKUs share the same split at origin $t_0$. |
| **3. Feature Lags** | Rolling means/quantiles that include `target_start` or days after `forecast_origin`. | Every historical feature window must end at $d \le \text{forecast\_origin} < \text{target\_start}$. |
| **4. Price Covariates** | Joining future unscheduled weekly `sell_price` from the holdout target week without verification. | Use lagged prices ($\le \text{forecast\_origin}$) unless scheduled prices are explicitly documented as known at origin. |
| **5. PMF Support $H$ & Caps $K_{sp}$** | Setting $H = \max(D_{\text{all}})$ or clipping test demand $D_{\text{test}} \leftarrow \min(D_{\text{test}}, H)$. | Choose $H$, tail distribution parameters, and $K_{sp}$ from `train` only; keep actual $D_{\text{test}}$ unclipped. |
| **6. Probability Calibration** | Fitting post-hoc calibration (e.g., temperature scaling) on `test` origins. | Fit calibration strictly on `val` origins; freeze parameters before evaluating on `test`. |
| **7. DRO Scenario Copula** | Estimating cross-store rank correlation or shock bootstrap from full dataset including `test`. | Fit empirical copula / rank-bootstrap strictly on `train` historical origins. |
| **8. Proportional Baseline** | Allocating proportionally to current target week demand $D_{sp}$. | Use trailing historical demand strictly prior to `forecast_origin` (e.g., trailing 8 weeks $\le t_0$). |
| **9. Clairvoyant Oracle** | Using oracle solutions as a selectable operational policy or feature input. | Restrict `clairvoyant_oracle` to computing hindsight regret $C(x_{\text{policy}}, D) - C(x_{\text{oracle}}, D)$. |

## Lineage Manifest JSON Schema (`--manifest`)

```json
{
  "records": [
    {
      "forecast_origin": "2016-01-03",
      "target_start": "2016-01-04",
      "target_end": "2016-01-10",
      "feature_max_date": "2016-01-03",
      "price_max_date": "2016-01-03",
      "split": "train",
      "store_id": "CA_1",
      "sku_id": "HOUSEHOLD_1_001"
    }
  ],
  "cohort_selection_end_date": "2016-02-28",
  "H_estimation_end_date": "2016-02-28",
  "copula_fit_end_date": "2016-02-28",
  "calibration_fit_splits": ["val"],
  "proportional_baseline_max_lag_date_offset": 0,
  "realized_test_demand_clipped": false
}
```
