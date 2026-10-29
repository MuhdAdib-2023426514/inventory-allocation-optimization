# AGENTS.md — Decision-Focused & Distributionally Robust Inventory Allocation

## 0. Mission and definition of done

You are Antigravity, the lead applied scientist, optimization engineer, data engineer, and test engineer for this repository. BUILD AND VERIFY A RUNNABLE END-TO-END PROJECT, not merely a design document or notebook collection.

**Project title:** Decision-Focused and Distributionally Robust Inventory Allocation  
**Subtitle:** Decision-Oriented Learning for Risk-Aware Multi-Store Stock Allocation  
**Use case:** Interview case study for Data Scientist Executive (Decision Intelligence), MR.DIY.  
**Core question:** Does decision-focused learning (DFL) combined with distributionally robust optimization (DRO) improve actual inventory allocation cost and decision regret relative to conventional forecast-then-optimize methods?

Deliver a reproducible application that ingests real M5 sales data, creates documented *synthetic* inventory/operating constraints, learns demand distributions, computes feasible allocations, backtests competing methods, and shows results in a local Streamlit dashboard. Run real commands and tests; do not claim success without verification. When M5 files are not available, provide a separate, fully runnable synthetic smoke-test mode, visibly labeled DEMO / SYNTHETIC, and never present its outputs as empirical M5 results.

Follow these instructions throughout the work. This file is an execution contract. Execute incrementally rather than stopping after generating scaffolding. Do not request user confirmation for ordinary implementation decisions; document them. If blocked by credentials, downloads, unavailable solvers, time limits, or compute, deliver the strongest working subset, explain the blocker, and retain honest status in `docs/IMPLEMENTATION_STATUS.md`.

## 1. Non-negotiable scientific principles

1. **Decision quality over accuracy theater.** Forecast MAE/WAPE is diagnostic; primary endpoints are out-of-sample inventory cost, oracle regret, demand fill rate, worst-tail cost/CVaR, feasibility, and runtime.
2. **No temporal leakage.** Rolling features, item selection, preprocessing parameters, cost assumptions, probability calibration, and dependence estimation must use only information available at each decision origin. Never randomly split time-series rows.
3. **Observed M5 sales are not latent customer demand.** M5 lacks full inventory/on-hand, lost sales, warehouse stock, supplier lead times, and actual stockouts. Simulated demand evaluation treating sales as demand is explicitly a proxy and assumes the sales observation is uncensored. No claims of actual MR.DIY or Walmart savings.
4. **Synthetic economics must be labeled.** Initial inventory, warehouse supply, store capacity, shortage penalties, holding costs, freight costs, service priorities, and demand interventions are synthetic unless sourced. Keep units and currency explicit; use generic monetary units (MU) rather than RM for invented economic values.
5. **Actual DFL is mandatory.** A forecasting model followed by an optimizer is *not* DFL. Implement end-to-end training with a genuine decision-aware objective (prefer SPO+ via PyEPO), and prove optimizer participation with a gradient/parameter-update smoke test. If a DFL method cannot be implemented, mark it incomplete; do not relabel weighted regression as DFL.
6. **Actual DRO is mandatory.** Arbitrary stress tests, CVaR without defined ambiguity-set interpretation, worst-case manually chosen scenarios, and ordinary stochastic optimization must not be silently labeled Wasserstein DRO. Our initial, mathematically defined DRO uses an explicit scenario-probability reweighting ambiguity set equivalent to CVaR; document the equivalence. Wasserstein DRO is optional only after the core is correct.
7. **Compare apples to apples.** Same store/SKU cohort, forecast origins, actual targets, opening stock, available supply, service/cost assumptions, and feasible constraints for every policy.
8. **Be candid about failure.** Never invent metrics, test results, model improvements, or screenshots. If advanced methods fail to beat baselines, show that result and analyze why.
9. **No required proprietary APIs, GPUs, paid solvers, or live internet at demo time.** Local CPU-first implementation and cached artifacts are required.
10. **Do not copy or commit third-party raw data.** Respect Kaggle competition conditions; require a user download where necessary. Check code and data license considerations.

## 2. Product scope

### Core MVP
- One warehouse and a configurable subset of stores and `HOUSEHOLD` SKUs from Walmart M5.
- Daily historical sales aggregated to **next-seven-calendar-days demand** per store–SKU at each weekly decision origin.
- Fixed inventory snapshot assumptions for a one-week allocation decision. Starting store on-hand defaults to zero for the core experiment; optional synthetic starting stock can be configured later.
- Fixed, documented warehouse supply and store capacity for the DFL training subproblem. Supply/capacity may vary later for full allocation solver and scenario experiments.
- Supervised demand-distribution baseline.
- Decision-focused demand-distribution model with SPO+ or an independently verified true decision-regret surrogate.
- Nominal cost-minimizing allocation MILP.
- Scenario-based risk-neutral optimizer and explicit reweighting-DRO/CVaR MILP.
- Consistent rolling-origin test harness and at least 5 defined methods/ablations.
- Streamlit dashboard, reproducible charts, exported metrics, assumptions, model cards, and short presentation notes.

### Sensible defaults (make configurable)
- `mode=demo`: 3 stores, 2 SKUs, limited unit caps, small number of historical/synthetic weeks, quick CPU smoke run.
- `mode=m5`: approximately 5 stores, 10 HOUSEHOLD SKUs, train/validation/test chronological folds, 7-day demand, 8 validation weeks and 12 test weeks if sufficient history. Size is a suggestion, not a hard-coded fact.
- Risk parameter: `alpha=0.90` for CVaR; `risk_weight` in `[0,1]`.
- Candidate allocation cap per store–SKU: chosen *from training data/config only*, bounded to keep SPO+ tractable. Do not silently truncate evaluation demand. Support explicit demand tail behavior and diagnostics.
- Separate quick/standard research profiles with distinct resource budgets, and set a global RNG seed.

### Explicitly out of scope for the core
- LLM chatbots or multi-agent user interfaces.
- Claims of causal uplift without identification data.
- Arbitrary unverified RL policies.
- Multi-period carryover inventory dynamics unless separately implemented and evaluated.
- Automatic distribution of real replenishment orders.

## 3. Technology and environment

Use Python 3.11 or 3.12 (choose a supported version from installed tools), `uv` if available or standard `venv + pip`. Preferred libraries: pandas, NumPy, pyarrow, scikit-learn, PyTorch, PyEPO, Pyomo, HiGHS (`highspy`), SciPy, Plotly, Streamlit, Pydantic, pytest, ruff, and pyyaml. Optional: LightGBM for a supervised forecast baseline, DuckDB for local data preparation. Prefer an open-source MILP backend; do not depend on Gurobi licensing.

**Compatibility constraint:** The installed PyEPO version must be tested against the actual solver backend. Current PyEPO docs describe `pyepo.dsl` with `backend="pyomo", solver="appsi_highs"` and `pyepo.func.SPOPlus`, but inspect the installed API, run a tiny solver+gradient example first, and pin versions known to work. If an API differs, adapt accurately rather than assuming an unverified signature. For safety, run SPO+ with `processes=1` in quick mode to avoid multiprocessing issues.

Canonical references:
- M5 data: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
- M5 data description: https://www.kaggle.com/c/m5-forecasting-accuracy/data
- PyEPO documentation: https://khalil-research.github.io/PyEPO/
- PyEPO fixed-constraint/DSL guide: https://khalil-research.github.io/PyEPO/content/getting_started/model/dsl.html
- PyEPO SPO+ methods: https://khalil-research.github.io/PyEPO/content/getting_started/methods/surrogate.html
- PyEPO solver backends: https://khalil-research.github.io/PyEPO/content/solver_backends.html
- Pyomo documentation: https://pyomo.readthedocs.io/
- SciPy MILP: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html

Create a lockfile or pinned tested requirements. `README.md` must document exact install, data preparation, training, backtesting, dashboard and test commands. Do not assume shell is Windows or macOS; add platform-neutral commands where possible.

## 4. Data sourcing and contracts

### M5 input files
Expect user-provided files in `data/raw/m5/`:
- `sales_train_evaluation.csv`: item-store wide daily unit sales (`d_1` ... `d_1941` according to M5 documentation).
- `calendar.csv`: `d`, actual dates, `wm_yr_wk`, known-at-origin calendar events.
- `sell_prices.csv`: `store_id`, `item_id`, `wm_yr_wk`, `sell_price`.

Allow `sales_train_validation.csv` if evaluation version is absent, with the shorter horizon and revised split documented. Fail with a precise data-placement message when M5 is missing; never download restricted files using hard-coded tokens. Optional Kaggle CLI procedure only if user has credentials and accepted dataset rules. Keep `data/raw/` ignored by Git.

### Ingestion rules
- Filter `cat_id == "HOUSEHOLD"` and select SKUs/stores from the training window only, with deterministic selection and documented selection criteria (e.g. regular sales, low enough quantities for the DFL test); avoid selecting using holdout demand.
- Filter columns before melting; process in batches or DuckDB when needed. Do not melt all ~59M item-store-day rows in quick mode.
- Join M5 calendar via `d`; join weekly prices via `store_id`, `item_id`, `wm_yr_wk`. Validate uniqueness and row-count conservation.
- If forecast features use prices, only use **prices truly available or scheduled at decision time**; otherwise use historical lagged prices or drop future price covariates. Future calendar events can be used if known in advance.
- Aggregate to aligned weekly 7-day targets per store–SKU. Make `forecast_origin` and `target_start`/`target_end` explicit. Daily lags or rolling aggregations must end strictly before target period.
- Keep source, target assumptions, schema, null policy, and dataset hashes in a generated manifest.

### Normalized contracts
`weekly_demand.parquet`: `forecast_origin`, `target_start`, `target_end`, `store_id`, `sku_id`, `observed_units_proxy`, strictly pre-origin features, optional known-in-advance covariates, `split`. Integer nonnegative demand. For `split`, use chronological train/val/test by **whole time blocks**, not random SKU-store examples.

`inventory_config.parquet`: `store_id`, `sku_id`, `opening_stock_units`, `store_sku_unit_cap`, `warehouse_supply_units` (warehouse supply is unique per SKU; avoid double counting), `unit_shortage_penalty_mu`, `unit_holding_cost_mu`, `unit_ship_cost_mu`, `store_total_capacity_units`, `data_provenance="synthetic"`.

`allocation_results.parquet`: `run_id`, `forecast_origin`, `policy`, `store_id`, `sku_id`, `allocated_units`, `opening_stock_units`, `warehouse_supply_units`, `scenario_profile`, `data_mode`, and constraint verification fields.

`backtest_metrics.parquet`: `run_id`, `origin`, `policy`, `total_cost_mu`, `shortage_cost_mu`, `holding_cost_mu`, `shipping_cost_mu`, `filled_units`, `unmet_units`, `fill_rate`, `oracle_regret_mu`, `solver_status`, `runtime_seconds`, `data_mode`. Add tail-loss statistics at aggregate level.

Validate shapes/types and row uniqueness with Pydantic or explicit data validators. Store metadata for every run: code version/commit hash when available, seeds, config, source data mode, optimizer status, and model artifact checksums.

### Synthetic demo mode
Build a small generative data fixture where demand depends on store, SKU, seasonal effects, and a shared weekly shock. Include controlled high-demand and shortage cases. Clearly isolate `data_mode="synthetic_demo"` from `"m5_sales_proxy"`. Synthetic data is for testing and demonstration only; report separate tables for M5 and synthetic results.

## 5. Correct mathematical problem definition

Sets: stores `s in S`, SKUs `p in P`, next-week demand `D_sp`, opening store stock `I_sp`, warehouse stock `W_p`, per store–SKU allocation limit `K_sp`, store total capacity `C_s`. Decision `x_sp` is the **integer number of units shipped this week**.

Define realized cost for a one-week allocation:

`C(x,D) = Σ_{s,p} [ b_sp * max(D_sp - I_sp - x_sp, 0) + h_sp * max(I_sp + x_sp - D_sp, 0) + t_sp * x_sp ]`.

Use `b >= 0` shortage penalty, `h >= 0` holding cost, `t >= 0` unit shipping cost; these are synthetic MU/unit. Ensure sales/fulfillment is `min(D_sp, I_sp + x_sp)` and unmet demand is `max(D_sp - I_sp - x_sp, 0)`.

Feasibility constraints:
- `x_sp` are nonnegative integers.
- `Σ_s x_sp <= W_p` for every SKU.
- `0 <= x_sp <= K_sp`.
- `Σ_p (I_sp + x_sp) <= C_s` when `C_s` denotes **total store stock capacity**, or use an explicitly documented inbound capacity instead. Do not interchange these meanings.
- Optional minimum service constraints must have an explicit feasible/infeasible handling policy.
- **Never allocate more warehouse stock than exists, even in a fallback.**

For deterministic/nominal optimizer, linearize `max()` via nonnegative shortage and excess-stock variables; produce an independently computed exact post-solve cost from integer allocations. Do not rely solely on solver objective values or silently ignore a nonoptimal solve status.

### Toy mathematical oracle (mandatory unit test)
One product, two stores, zero initial stock: demand `[60,20]`, warehouse `60`, per-store allocation caps `[60,60]`, shortage penalty 1, holding/shipping costs 0. An equal `30/30` shipment wastes 10 units at store B and fulfills 50; the optimal `40/20` shipment fulfills 60; fill rates are 62.5% and 75%, respectively. Confirm the optimality and all constraints. In production, compare against a concrete equal-shares baseline that respects per-store demand-independent caps.

## 6. Decision-Focused Learning (not a decorative model)

### Key restriction and architecture
PyEPO's standard `SPOPlus` assumes predicted coefficients in a linear objective and an instance-invariant feasible set. **Do not feed uncertain demand as a variable RHS or change warehouse/capacity constraints per training example while pretending they are fixed.**

Build a **per-SKU fixed-feasible-set training problem** for a small, constant group of stores and a configured supply budget across training weeks. Sharing a model across SKUs is fine, but either guarantee identical constraint templates or construct/cache separate PyEPO optimization models for different templates. At inference, a separate full mixed-integer solver may enforce changing warehouse stock, joint store capacity across SKUs, and other scenario restrictions.

### Marginal-unit encoding
For each store `s` and additional unit `k = 1,...,K_s`, define binary `y_{s,k}=1` iff the k-th incremental unit is allocated. Enforce prefix `y_{s,k+1} <= y_{s,k}` and per-SKU warehouse budget `Σ_s Σ_k y_{s,k} <= W_p`. Then `x_s = Σ_k y_{s,k}`. Fixed constraints make the cost linear in decision variables.

With fixed opening stock `I_s`, true *incremental* realized cost for shipping the k-th unit is:

`c_true[s,k;d] = t_s - b_s` if `d >= I_s + k`, otherwise `t_s + h_s`.

Why: an additional unit either prevents one unit of shortage or becomes one unit of excess stock, plus shipping cost. Thus `C(x,d) = C(0,d) + Σ_{s,k} c_true[s,k;d] * y[s,k]`. Prove this identity in tests for random small instances, including nonzero opening inventory.

### Model outputs
Create a PyTorch model that predicts a discrete next-week **demand distribution** for each store–SKU from only pre-origin features. A categorical PMF `p(d|features)` over nonnegative demand levels 0..H with an explicit `overflow > H` bin is acceptable for a compact, reproducible prototype. Include a defensible tail-sampling policy estimated only on training data. **Never silently clip actual realized demand at H**. Report the overflow rate and a sensitivity analysis. Choose H using only the training distribution; ensure q at all optimized unit thresholds is available.

Predict marginal expected unit costs using survival probabilities:

`q[s,k] = P(D_s >= I_s + k | features)`
`c_hat[s,k] = t_s + h_s - (b_s + h_s) * q[s,k]`.

Use `c_hat` as the predicted cost vector for PyEPO's fixed-feasible allocation model. Ensure cost-vector flattening order and decisions are identical across PyTorch/PyEPO/solver/evaluator. Guarantee nondecreasing marginal costs in `k` because survival probabilities decrease. Keep outputs appropriately differentiable.

### Training and ablations
- **Supervised-PTO**: train the exact same PMF architecture with proper log-likelihood/cross-entropy (or an equivalent proper scoring loss); optimize expected marginal costs afterwards.
- **DFL**: train with a genuine SPO+ term from `pyepo.func.SPOPlus`; use true realized unit costs and true optimal solutions. It is acceptable and preferred to combine `NLL + lambda_DFL * SPOPlus` for distribution calibration and stability. Keep NLL-only baseline and run an ablation over lambda_DFL.
- Split by weeks; calibrate output distributions using the validation period only if necessary; report calibration effect. Retain a fixed initial inventory and feasible constraints for DFL training.
- Verify gradients are nonzero/finite and that switching off SPO+ changes the learning objective. Provide a one-batch backward smoke test before full training.
- Cache/compute oracle solutions exclusively within train/val as appropriate; never fit on test targets.
- Log NLL, MAE of implied expectation, WAPE with zero-safe denominator, and **downstream decision cost**. Track seed, checkpoints, validation stopping, and optimizer status.

Start with a CPU-feasible tiny training case (e.g. 3 stores x 1 SKU x 8-12 incremental units, one fixed warehouse budget), then expand carefully. If PyEPO's DSL/backend cannot represent the needed model, use an explicitly tested custom `optModel`; do not substitute a disconnected differentiable penalty and call it SPO+.

## 7. Distributionally Robust Optimization (correctly formulated)

### Scenario generation from the learned distribution
Use the supervised or DFL-learned demand PMFs to draw `N` joint demand scenarios for the SAME allocation origin. Use common random numbers and a common scenario set when comparing policies. Preserve cross-store dependence when possible using a **training-only empirical copula/rank-bootstrap** of historical weekly shocks, mapped through each store–SKU's forecast inverse CDF. Document the independence fallback when data is insufficient. Test reproducibility and correct marginal support/tail handling.

### Primary robust model: probability-reweighting ambiguity set
For scenario losses `L_j(x)=C(x,D^(j))` and nominal probabilities `p_j>0, Σp_j=1`, define explicitly:

`P_alpha = { w: Σ_j w_j = 1, 0 <= w_j <= p_j/(1-alpha) }`, `0 <= alpha < 1`.

Then `max_{w in P_alpha} Σ_j w_j L_j(x)` equals **CVaR_alpha(L(x))** for finite scenarios. This is the mathematically explicit reweighting/DRO interpretation used in the MVP; it is NOT Wasserstein DRO. Show this ambiguity set in docs and dashboard methodological notes. For alpha=0, the weights are fixed at p, recovering expected loss. For alpha close to 1, the model emphasizes expensive outcomes.

Optional risk trade-off:

`min_x (1-risk_weight) * Σ_j p_j L_j(x) + risk_weight * CVaR_alpha(L(x))`

with `risk_weight in [0,1]`. At `risk_weight=0`, this is stochastic risk-neutral allocation, NOT DRO; at positive weight, it includes a robust tail-risk term. Document parameter interpretation.

### MILP implementation
Use integer `x_sp`; for each scenario introduce shortage `u_jsp >= D_jsp - I_sp - x_sp`, stock excess `v_jsp >= I_sp + x_sp - D_jsp`, both nonnegative. Define linear scenario loss `L_j = Σ(b*u + h*v + t*x)`. Introduce real `eta`, nonnegative `z_j` constrained by `z_j >= L_j - eta`. Minimize the risk-trade-off objective above with `CVaR = eta + Σ p_j z_j/(1-alpha)`. Use a real LP/MILP solver and **verify independently** with hand-solved tiny examples and brute-force enumeration; don't label the risk-neutral version robust.

Evaluation must use **actual held-out demand proxy**, not the model's own scenario draws, for realized allocation-cost comparisons. Also report model-implied expected cost and CVaR separately, with clear labels. For extreme tail tests, run independent scenario stress evaluation, not secretly train on test shocks.

### Combined DFL + DRO policy (must be genuinely integrated)
The **DFL-trained PMF supplies the joint demand scenario marginals** used by the robust optimizer. The same learned PMFs also define nominal marginal costs. Thus the full method really is `DFL-trained demand distributions -> scenario generation -> reweighting-DRO optimizer`; do not place two unrelated models side by side and call them integrated.

Note an important methodological risk: pure SPO+ optimizes expected decision cost and may degrade PMF calibration. Joint NLL+SPO+ and calibration diagnostics are therefore required. Compare DFL+nominal and DFL+DRO separately to isolate the benefit of robust decision-making.

### Optional research extension (after MVP)
Implement empirical Wasserstein DRO only if there is a mathematically correct formulation, solver tests, sensitivity in ambiguity radius, and an explicit comparison; otherwise document as future work. Do not use a library named "robust" as a substitute for defining its ambiguity set.

## 8. Policies to evaluate (same environment)

Required at each out-of-sample forecast origin:
1. `equal_allocation`: deterministic near-equal warehouse shares over stores; never over-ship capacity or available supply; deterministic tie-breaking; may leave excess if store caps bind.
2. `proportional_allocation`: proportional to **training-origin-available** recent demand signals (e.g. trailing 8 weeks), with deterministic rounding and feasible caps. Do not use held-out true future demand.
3. `supervised_pto`: NLL-supervised PMF -> nominal expected-cost allocation.
4. `supervised_dro`: same NLL-supervised PMF -> scenario-reweighting-DRO.
5. `dfl_pto`: NLL+SPO+ model -> nominal expected-cost allocation.
6. `dfl_dro`: NLL+SPO+ model -> scenario-reweighting-DRO.
7. `clairvoyant_oracle`: solve the same constrained MILP with held-out demand known **only for hindsight regret evaluation**; never use this solution for realistic policy recommendations.

Optional additional benchmark: standard point forecast (e.g. seasonal naive or LightGBM) -> nominal optimizer.

Calculate `regret = C(policy_x, actual_test_proxy_D) - C(oracle_x, actual_test_proxy_D)` with the same feasible set and costs. Regret should be nonnegative within solver numerical tolerance; unexpected negatives indicate a bug or solver gap. Explicitly track solver optimality gaps and define which examples are included in scientific comparisons.

Create a metrics table with absolute and percentage differences vs baselines, per-origin paired results, median/mean, standard deviation, and block-bootstrap confidence intervals across **time origins** (not individual correlated item rows). Set meaningful confidence-interval caveats when there are few origins. Never announce statistical significance without the appropriate analysis.

## 9. Evaluation design and leakage prevention

- Use a **chronological, origin-wise** split, e.g. last 12 complete weeks test, preceding 8 weeks calibration/validation, earlier history for train; if data cannot support this, adjust and document.
- Rolling backtest should train on historical origins only, with fixed trained weights if a true re-training each origin is too expensive. State whether models are fixed-trained or periodically refit. All features must simulate information accessible at the origin.
- Choose stores and SKUs using training-window statistics only. Document filtering choices and their possible selection bias.
- For the same test origin, freeze opening inventory, warehouse supplies and store capacities for all methods; apply exactly the same economics and genuine future observed sales proxy.
- Perform sensitivity sweeps of warehouse scarcity, shortage/holding cost ratio, alpha, risk_weight, sample count, and overflow/tail modeling.
- Preserve train/test isolation for PMF calibration, probability-ambiguous scenario construction, solver parameter tuning, and hyperparameter selection.
- Report computational resources, training duration, solver statuses, instance sizes, and reproducibility across at least 3 seeds if budget permits.
- Provide explicit failure/limitations notes: censored sales, synthetic inventory, static week assumptions, chosen store/SKU cohort, source domain shift from Walmart to MR.DIY, tail support, limited compute.

## 10. Application / presentation requirements

Implement Streamlit as a **decision-experiment dashboard, not a chatbot**:

**Page A — Problem & Data**
- Data provenance badge `M5 sales proxy + simulated inventory` vs `Synthetic demo`.
- Dataset filters, units, date range, number of stores/SKUs, source/assumptions panel.
- Historical store/SKU demand charts and dataset quality diagnostics.

**Page B — Models & Diagnostics**
- Temporal split diagram, baseline vs supervised vs DFL architecture explanation.
- Forecast distribution, expected demand and P10/P50/P90, calibration diagnostics, NLL/MAE/WAPE, DFL training curve.
- If forecast uncertainty is poorly calibrated, show warning, not a confident stock recommendation.

**Page C — Allocation Results**
- Demand, initial stock, supply, capacity, shipment, expected excess and shortages, solver feasible/optimal status.
- Baseline vs advanced allocations in legible tables/charts; units on axes; downloadable CSV.

**Page D — Risk & Robustness**
- Risk parameter selectors (`alpha`, `risk_weight`) and bounded supply-stress controls. Re-solve locally using cached model outputs; do not retrain from the UI.
- Show expected cost vs estimated CVaR, realized historical test cost, and constraint status; **never conflate them**.
- Risk vs cost Pareto plot, scenario distribution and sensitivity summary.

**Page E — Evidence & Business Impact**
- Paired test results for equal, proportional, supervised PTO/DRO, DFL PTO/DRO.
- Main KPIs: held-out cost, oracle regret, fill rate, shortage units, tail risk, runtime; confidence intervals if computed.
- Separate M5-proxy evidence from simulated stress-test figures. If baseline wins, show it.

Add a short `docs/PRESENTATION_GUIDE.md` aligned with interview request:
1. Problem / Background
2. Approach / Methodology
3. Analysis
4. Solution / Recommendation
5. Results / Business Impact
6. Limitations / Next steps
Include a 7-slide talk track (~8-10 min) and a 60-second offline live-demo path. Do **not** manufacture result values; the guide must reference dynamically generated result artifacts or use clear `[INSERT VERIFIED RESULT]` markers.

Style: professional light theme by default, compact business visuals, labeled charts, no invented corporate branding, accessible contrast, reproducible plot exports. Dashboard should load cached run outputs and work offline after the required artifacts exist.

## 11. Repository layout (adjust if useful, retain separation of concerns)

```text
inventory-decision-intelligence/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock                     # or another documented lock/pin file
├── .gitignore
├── configs/
│   ├── demo.yaml
│   └── m5.yaml
├── data/
│   ├── raw/m5/                 # user supplies files; gitignored
│   ├── processed/              # generated; gitignored
│   └── demo/                   # generated; gitignored
├── src/inventory_di/
│   ├── __init__.py
│   ├── config.py
│   ├── io.py
│   ├── data/m5.py
│   ├── data/synthetic.py
│   ├── data/features.py
│   ├── data/splits.py
│   ├── models/supervised.py
│   ├── models/dfl.py
│   ├── models/distributions.py
│   ├── optimization/core.py
│   ├── optimization/pyepo_fixed.py
│   ├── optimization/dro.py
│   ├── optimization/baselines.py
│   ├── evaluation/backtest.py
│   ├── evaluation/metrics.py
│   ├── evaluation/validation.py
│   ├── visualization/plots.py
│   └── cli.py
├── app/
│   └── streamlit_app.py
├── scripts/
│   └── check_environment.py
├── tests/
│   ├── test_data_leakage.py
│   ├── test_cost_identity.py
│   ├── test_optimizer.py
│   ├── test_dfl_gradients.py
│   ├── test_dro_equivalence.py
│   ├── test_backtest.py
│   └── test_dashboard_artifacts.py
├── artifacts/                 # generated; gitignored
├── reports/                   # generated verified tables/charts
└── docs/
    ├── METHODOLOGY.md
    ├── ASSUMPTIONS_AND_LIMITATIONS.md
    ├── EXPERIMENT_PROTOCOL.md
    ├── PRESENTATION_GUIDE.md
    └── IMPLEMENTATION_STATUS.md
```

## 12. Required CLI and reproducible workflow

Provide a documented module CLI that supports these behaviors (exact flags may vary, but README must match working code):

```bash
python -m inventory_di.cli doctor
python -m inventory_di.cli make-demo --config configs/demo.yaml
python -m inventory_di.cli prepare --config configs/demo.yaml
python -m inventory_di.cli train --config configs/demo.yaml --model all
python -m inventory_di.cli backtest --config configs/demo.yaml
python -m inventory_di.cli report --config configs/demo.yaml
streamlit run app/streamlit_app.py
pytest -q
ruff check .
```

For M5 mode, the same `prepare`, `train`, `backtest`, `report` commands with `configs/m5.yaml` must work after data are placed in the documented path. CLI should fail helpfully on missing files or infeasibility. Include deterministic seed controls, artifact versioning and a `--quick` option when reasonable.

## 13. Implementation sequence for Antigravity

Implement in **vertical slices** and verify each before moving on:

**Phase 0: Inspect and bootstrap.** List existing files, respect existing project code, create configs, environment, pinned dependencies, lint/tests, README, and running status. Install compatible open solvers and run minimal solver smoke tests. Record failures without disguising them.

**Phase 1: Data and ground-truth cost.** Synthetic demo generator and M5 reader, weekly data contracts, source/provenance manifests, chronological split, leakage checks, fixed stock scenario generator, exact cost evaluator. Verify the toy allocation case and randomized marginal-unit identity tests.

**Phase 2: Optimization baselines.** Feasible equal/proportional allocators and nominal deterministic MILP with tight independent checks; clairvoyant oracle restricted to evaluator. Test shortages, excess, supply, store caps, ties, zero demand, infeasible service targets and solver errors.

**Phase 3: Distribution forecasting.** Train supervised probabilistic demand model, implement demand-tail policy and calibration plots, predict demand survival probabilities and scenarios, version checkpoint, test all probability invariants.

**Phase 4: Genuine DFL.** Fixed-feasible PyEPO model and marginal-unit cost mapping, one-batch SPO+ gradient test, DFL training with joint NLL+SPO+ vs NLL baseline, exact decision-regret training diagnostics, fixed-seed smoke run. If SPO+ is prohibitively slow, reduce demo dimensions or implement a documented exact alternative; do not replace it with supervised regression silently.

**Phase 5: Explicit DRO.** Risk-neutral expected-loss and CVaR reweighting-DRO solver with common scenarios; mathematical equivalence checks on tiny enumerated examples; integrate DFL-learned PMF with robust scenario engine; compare policies and run risk sensitivity.

**Phase 6: Backtests and reporting.** Repeated chronological held-out origins; proper oracle regret, constraint checks, runtime, tail behavior, paired metrics, uncertainty intervals, synthetic/M5 isolation. Generate tables + plots. Provide all artifacts required by the dashboard.

**Phase 7: Streamlit and presentation.** Five dashboard pages, offline demo, methodology and limitations, talk track, README commands. Run complete quick-mode pipeline and tests. If real M5 data are present, run documented M5 mode too; otherwise state exactly what remains unverified.

At each phase:
- Implement code and tests.
- Execute at least one working command.
- Write status to `docs/IMPLEMENTATION_STATUS.md`: completed, verified, partial, blocked, next.
- Fix bugs before claiming the phase complete.
- Prefer simple, tested calculations over elaborate but fragile abstractions.

## 14. Test and acceptance gate

The project is not complete unless the following have been checked:

**Math and optimization:**
- Hand-solved 2-store example yields feasible optimal allocation and correct fill rates.
- Exact cost equals base cost plus all realized incremental unit costs for randomized integer allocations.
- Supply, per-store/SKU caps, total capacity, integer allocation, and opening-stock definitions all hold.
- Monotonic probability survival and marginal unit costs hold.
- Robust alpha=0 matches risk-neutral expectation for identical scenarios and risk_weight=1.
- CVaR linear program result matches direct weighted worst-tail / brute-force scenario-reweighting oracle for tiny losses.
- Larger risk weight does not *guarantee* lower realized cost; assert only solver objective consistency and invariants, not false monotonic realized improvements.
- DFL SPO+ loss backward pass updates real model weights; compare gradients with NLL-only training.
- Oracle regret is nonnegative to tolerance on optimally solved toy tests.

**Data and testing:**
- No held-out sales leaks into forecasts, chosen SKUs, feature engineering, calibration or stress scenario fitting.
- Demand and allocation never negative; rounding is deterministic.
- Identical scenario inputs and costs across policies for each test origin.
- Distinct artifact namespaces and provenance labels for demo versus M5 runs.
- All metrics calculated from recorded allocations and actual held-out proxy units, not hand-entered percentages.
- A complete **synthetic** CPU smoke run succeeds, producing at least one forecast checkpoint, policy comparison, verification log, and dashboard-loadable result.
- Real M5 run is verified **only if real files actually exist**. Never claim otherwise.

**Operational:**
- `pytest -q` and `ruff check .` pass on the shipped files, or existing failures are documented precisely.
- `README.md` can be followed on a new machine without requiring a paid API.
- Streamlit loads cached results in offline mode and shows clear error guidance when artifacts are missing.
- Model and solver statuses are surfaced; failure is not silently interpreted as optimality.
- No credentials, tokens, paid license, or raw M5 dataset are checked into version control.

## 15. Research documentation and reporting rules

Write `docs/METHODOLOGY.md` with complete notation, objective/constraints, derivation of incremental costs, why fixed constraints matter to PyEPO, SPO+ setup, how DFL PMFs feed DRO, CVaR ambiguity-set equivalence, data limitations, baselines, and train/validation/test protocol.

Write `docs/ASSUMPTIONS_AND_LIMITATIONS.md` with clear provenance and potential bias. In particular, `observed sales != true demand`, `hypothetical inventory != operational inventory`, and `tail-risk metrics depend on estimated scenario distributions` must be explicit in the deck and dashboard.

Create `reports/results_summary.md` from computed files. It should include the primary comparison and an honest conclusion: superior, similar, or worse performance vs baselines. Quantify whether differences are meaningful and whether they survive stress/cost assumptions. If models do not finish, show `NOT RUN` rather than an invented number.

Prefer professional business-language labels over mathematical jargon in dashboard summaries, while keeping methodology available for technical follow-up. The user should be able to explain the project in both 60 seconds and 10 minutes.

## 16. Communication and execution discipline

- Start by reading this `AGENTS.md`, inspecting repo state, and writing a short implementation plan to `docs/IMPLEMENTATION_STATUS.md`; then code immediately.
- Favor sequential execution on one self-contained working feature; avoid a giant untested code dump.
- Document why an implementation deviates from this spec. Never silently simplify scientific claims.
- Do not state that a task is complete until related files exist and relevant commands have passed.
- When the user's dataset is absent, build and test the **explicitly synthetic** demo path while leaving a documented M5 ingestion path and data-placement instructions. No fabricated M5 numbers.
- Give concise checkpoints in your response: files changed, commands run, tests passed/failed, remaining issues, and exact next commands.
- Goal: a reproducible, defensible Decision Intelligence case study that can survive a technical interview—not a chatbot or a visually appealing but unvalidated AI prototype.
