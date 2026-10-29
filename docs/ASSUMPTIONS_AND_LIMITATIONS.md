# Assumptions and Limitations

**Authoritative specification:** [`AGENTS.md`](../AGENTS.md)

---

## 1. Core Scientific Disclaimers (Non-Negotiable)

1. **`Observed Sales != True Latent Demand`**
   - Historical retail datasets (including Walmart M5 `sales_train_evaluation.csv`) record **completed unit transactions**, not uncensored customer demand. When a store stocks out on a given day, observed sales are zero even if customer demand was positive.
   - Treating observed weekly unit sales as a demand proxy (`observed_units_proxy`) implicitly assumes uncensored observations during the historical window.
   - No claims of actual financial savings for MR. D.I.Y. or Walmart are made.

2. **`Hypothetical Inventory != Operational Inventory`**
   - Public M5 data do not include store opening on-hand stock $I_{sp}$, central warehouse supply $W_p$, store shelf/backroom capacity $C_s$, supplier lead times, or unit shortage/holding/shipping costs ($b_{sp}, h_{sp}, t_{sp}$).
   - All inventory parameters and cost coefficients in `inventory_config.parquet` carry `data_provenance = "synthetic"` and are denominated in generic **Monetary Units (`MU`)**, never Malaysian Ringgit (`RM`) or USD.

3. **`Tail-Risk Metrics Depend on Estimated Scenario Distributions`**
   - Model-Implied Expected Cost and Model-Implied $\text{CVaR}_\alpha$ on Page D are functions of the $N$ scenarios sampled from the learned PMF and training-only empirical copula.
   - If the learned distribution is miscalibrated or if a test week experiences a structural regime shift outside the training support, model-implied $\text{CVaR}_\alpha$ will differ from **Realized Historical Test Cost**. For this reason, our reports and dashboard strictly separate:
     1. `model_expected_cost_mu` (Model-Implied Expected Cost)
     2. `model_cvar_cost_mu` (Model-Implied $\text{CVaR}_\alpha$)
     3. `realized_test_cost_mu` (Realized Historical Test Cost on held-out demand proxy)

---

## 2. Modeling & Structural Limitations

- **Single-Period Static Allocation Snapshot:** Each weekly forecast origin is solved as a standalone 7-day allocation problem with configured opening stock $I_{sp}$ (defaulting to 0 in the core experiment). Multi-week inventory carryover dynamics and supplier replenishment order lead times are out of scope for the core MVP.
- **Per-SKU Fixed-Feasible Training Template for SPO+:** PyEPO's `SPOPlus` surrogate requires an instance-invariant feasible polytope. Training uses a fixed per-SKU warehouse budget $W_p$ and per-store caps $K_{sp}$ derived strictly from the training split, whereas full inference MILPs (`solve_nominal_marginal_cost_milp` and `solve_scenario_reweighting_dro_milp`) enforce joint multi-SKU store capacity $C_s$ and dynamic supply stress multipliers.
- **Discrete Support $H$ and Tail Approximation:** While realized evaluation demand is never clipped at $H$, the categorical PMF groups probabilities above $H$ into a single `overflow > H` bin and samples excess units from a training-fitted shifted Poisson tail. Extreme fat-tailed surges far above $H$ are only approximately represented.
- **Trade-off Between Average Cost and Tail Hedging in DRO:** Increasing `risk_weight` or $\alpha$ shifts stock toward stores with high shortage penalties or high tail variance to cap worst-case scenario losses. When realized test demand is moderate rather than extreme, conservative DRO allocations can incur higher holding or average shortage costs than risk-neutral `dfl_pto` or `supervised_pto`—as transparently shown in our synthetic demo backtest (`dfl_pto` achieves the lowest mean cost of `50.48 MU`, whereas `dfl_dro` trades mean cost for lower model-implied $\text{CVaR}_{0.90}$).
- **Small Number of Held-Out Time Origins:** With 8 test origins in demo mode (and 12 in M5 mode), moving-block bootstrap confidence intervals across forecast origins provide descriptive uncertainty bands rather than high-powered asymptotic hypothesis tests.
- **Domain Shift from Walmart M5 / Synthetic Fixtures to MR. D.I.Y.:** Walmart US `HOUSEHOLD` items and synthetic Poisson-Gamma fixtures differ from Malaysian hardware/household retail store formats, pack sizes, and promotional calendars. Production deployment at MR. D.I.Y. requires integrating real ERP stock-on-hand snapshots, warehouse pack-size rounding constraints, and lost-sales censoring imputation.
