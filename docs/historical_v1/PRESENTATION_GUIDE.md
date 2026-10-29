# Interview Presentation Guide — Data Scientist Executive (Decision Intelligence), MR. D.I.Y.

**Project Title:** Decision-Focused and Distributionally Robust Inventory Allocation  
**Subtitle:** Decision-Oriented Learning for Risk-Aware Multi-Store Stock Allocation  
**Dynamic Results Reference:** [`reports/results_summary.md`](../reports/results_summary.md)

---

## 1. Interview Alignment (The 5 Required Sections + Limitations)

| Interview Prompt Section | Core Message for MR. D.I.Y. Panel | Dashboard Page |
| :--- | :--- | :--- |
| **1. Problem / Background** | When central warehouse supply $W_p$ is scarce relative to multi-store demand, equal or static proportional allocation rules waste stock at low-demand stores while starving high-demand stores. Minimizing forecast MAE/WAPE alone does not minimize downstream shortage and holding costs. | **Page A — Problem & Data** |
| **2. Approach / Methodology** | Formulate integer multi-store stock allocation $C(x, D)$ with exact marginal-unit binary encoding $y_{s,p,k}$ under shared store capacity $C_s$ (`JointMultiSKUMarginalAllocationModel`). Train a discrete demand PMF end-to-end with PyEPO `SPOPlus` ($\text{NLL} + \lambda_{\text{DFL}}\text{SPO}+$) and feed calibrated copula scenarios into an explicit probability-reweighting $\text{CVaR}_\alpha$ DRO MILP using sparse CSC matrices. | **Page B — Models & Diagnostics** |
| **3. Analysis** | Demonstrate the mandatory 2-store mathematical oracle ($D=[60,20], W=60$: equal `30/30` gives `62.5%` fill rate vs. optimal `40/20` giving `75.0%` fill rate), inspect survival probability curves $q[s,k]$, validation scenario calibration (Energy Score & bias), and $\lambda_{\text{DFL}}$ validation ablations. | **Page B & Page C** |
| **4. Solution / Recommendation** | Provide an offline-ready Decision Intelligence simulator that outputs feasible integer store shipments $x_{sp}$ and lets planners stress-test warehouse supply scarcity, opening stock overrides, pricing/display demand uplift (`%`), shortage penalty priority multipliers, and tail-risk weights ($\alpha, \text{risk\_weight}$) in $<50\text{ ms}$ without retraining models. | **Page C & Page D** |
| **5. Results / Business Impact** | Present the verified 7-policy out-of-sample comparisons across three cohorts: (1) **`DEMO / SYNTHETIC` (`demo_v1`, [`reports/demo/results_summary.md`](../reports/demo/results_summary.md)):** `dfl_pto` achieves the lowest mean cost (**51.32 MU**, `-3.4%` vs `equal_allocation` at **53.13 MU** and beating `supervised_pto` at **51.37 MU** and `proportional_allocation` at **52.16 MU**). (2) **Empirical Walmart M5 Regular Cohort (`m5_v1`, 5 stores $\times$ 10 regular `HOUSEHOLD` SKUs, `ADI <= 1.25`, [`reports/m5/results_summary.md`](../reports/m5/results_summary.md)):** `dfl_pto` (**1312.70 MU**) and `supervised_dro` (**1315.14 MU**) beat `supervised_pto` (**1322.76 MU**) and reduce cost by **-8.4% (-119.64 MU)** vs `equal_allocation` (**1432.35 MU**), while achieving the **lowest realized tail risk ($\text{CVaR}_{0.90} = 1677.40\text{ MU}$ and $1686.33\text{ MU}$ vs $1711.69\text{ MU}$ for `proportional_allocation`)**. Under multi-period rolling inventory carryover with strict mass conservation, **`dfl_dro` wins #1 at both $L=0$ (`1230.94 MU`, `-19.4%` vs `supervised_pto` at `1526.37 MU`, `-40.8%` vs `equal_allocation`) and $L=1$ (`1526.90 MU`, beating `supervised_dro` at `1565.34 MU`, `dfl_pto` at `1566.20 MU`, and `supervised_pto` at `1628.22 MU`)**! (3) **Empirical Walmart M5 Intermittent Cohort (`m5_intermittent_v1`, 10 lumpy SKUs, `ADI >= 1.35`, [`reports/m5_intermittent/results_summary.md`](../reports/m5_intermittent/results_summary.md)):** Trailing `proportional_allocation` degrades to **last place (`176.94 MU`)**, while **`supervised_dro` (`171.30 MU`) and `dfl_dro` (`172.59 MU`) win #1 and #2 overall**! | **Page E — Evidence & Business Impact** |
| **6. Limitations / Next Steps** | Explicitly state that observed sales $\ne$ uncensored demand, inventory/cost parameters are synthetic (`MU`), and our DRO uses the explicit scenario-probability-reweighting $\text{CVaR}_\alpha$ ambiguity set $\mathcal{P}_\alpha$, not Wasserstein DRO. | **Page A, D, E** |

---

## 2. Seven-Slide Talk Track (~8–10 Minutes)

### Slide 1 — Executive Summary & The Retail Allocation Problem (~1 min)
- **Hook:** "In multi-store retail like MR. D.I.Y., forecasting demand is only half the problem—the real financial impact comes from the **allocation decision** when warehouse stock is constrained."
- **Key Point:** Show how optimizing forecast accuracy (MAE/WAPE) treats over-forecasting and under-forecasting symmetrically, whereas retail economics are asymmetric (stockout penalty $b_{sp} \gg$ holding cost $h_{sp}$) and coupled across stores and SKUs by warehouse supply $W_p$ and shared store capacity $C_s$.

### Slide 2 — Mathematical Formulation & Toy Oracle Proof (~1.5 min)
- **Exact Cost Function:** $C(x, D) = \sum_{s,p} [b_{sp}\max(D_{sp}-I_{sp}-x_{sp},0) + h_{sp}\max(I_{sp}+x_{sp}-D_{sp},0) + t_{sp}x_{sp}]$.
- **Toy Oracle Intuition:** Walk through 1 SKU, 2 stores, $D=[60, 20], W=60$. Equal split `30/30` wastes 10 units at Store B and leaves 30 units unmet at Store A (`62.5%` fill rate, cost `30 MU`). Optimal allocation `40/20` fulfills all 60 available units (`75.0%` fill rate, cost `20 MU`).

### Slide 3 — Architecture: Why Standard Forecasting Fails & How Genuine Joint Multi-SKU DFL Works (~1.5 min)
- **Marginal-Unit Encoding:** Explain why we introduce binary variables $y_{s,p,k} \in \{0,1\}$ with prefix constraints $y_{s,p,k+1} \le y_{s,p,k}$, per-SKU warehouse budget $\sum_{s,k} y_{s,p,k} \le W_p$, and shared store capacity $\sum_{p,k} y_{s,p,k} \le C_s - \sum_p I_{sp}$, proving $C(x,d) = C(0,d) + \sum_{s,p,k} c_{\text{true}}[s,p,k;d]y_{s,p,k}$.
- **Differentiable Survival Mapping:** Our PyTorch PMF network predicts $p(d \mid \text{features})$ over $0 \dots H$ plus an `overflow > H` bin, computes survival probabilities $q[s,p,k] = P(D_{sp} \ge I_{sp} + k)$, and maps them to expected marginal costs $\hat{c}[s,p,k] = t_{sp} + h_{sp} - (b_{sp} + h_{sp})q[s,p,k]$.
- **End-to-End PyEPO `SPOPlus` Training:** Contrast `supervised_pto` ($\text{NLL}$ only) with `dfl_pto` ($\text{NLL} + \lambda_{\text{DFL}}\text{SPO}+$ solved via `JointMultiSKUMarginalAllocationModel` and HiGHS).

### Slide 4 — Risk-Aware Allocation via Probability-Reweighting DRO ($\text{CVaR}_\alpha$) (~1.5 min)
- **Joint Scenario Generation & Validation Calibration:** Use a training-only empirical rank-bootstrap copula to sample $N$ cross-store correlated demand scenarios from the learned PMFs, calibrated on `split == 'val'` to minimize scenario bias and multivariate Energy Score (`25.35 -> 21.90` on M5).
- **Explicit Ambiguity Set & Validation Tuning:** Define $\mathcal{P}_\alpha = \{w \ge 0 : \sum w_j = 1, w_j \le p_j/(1-\alpha)\}$ ($\equiv \text{CVaR}_\alpha(L(x))$ by Rockafellar–Uryasev duality) and select $(\alpha^*=0.90, \text{risk\_weight}^*=0.25)$ strictly on `split == 'val'`.
- **Three Distinct Metrics:** Emphasize why a Decision Intelligence system must never conflate *Model-Implied Expected Cost*, *Model-Implied $\text{CVaR}_\alpha$*, and *Realized Held-Out Test Cost*.

### Slide 5 — Live Decision-Experiment Walkthrough (Pages A–D) (~1.5 min)
- Switch to the live Streamlit dashboard running locally on your laptop (see 60-Second Demo Path below). Show the **Full-Horizon Cost–Risk–Service-Level Trade-Off Frontier** and the interactive **What-If Simulation Controls** (Warehouse Scarcity, Opening Stock Override, Pricing/Display Demand Uplift %, and Shortage Penalty Priority Multiplier).

### Slide 6 — Empirical Backtest Evidence Across Demand Regimes & Operational Horizons (~1.5 min)
- **1. Single-Period Regular M5 (`m5_v1`, 5 stores $\times$ 10 regular `HOUSEHOLD` SKUs, 12 test weeks, [`reports/m5/results_summary.md`](../reports/m5/results_summary.md)):**
  - `dfl_pto` (**1312.70 MU**, $\text{CVaR}_{0.90} = \text{1686.33 MU}$) and `supervised_dro` (**1315.14 MU**, $\text{CVaR}_{0.90} = \text{1677.40 MU}$) both beat `supervised_pto` (**1322.76 MU**, $\text{CVaR}_{0.90} = \text{1708.12 MU}$) and reduce cost by **-8.4% (-119.64 MU)** vs `equal_allocation` (**1432.35 MU**).
  - Across 3 independent neural network training seeds (`[42, 101, 202]`), **`dfl_pto` (`1305.74 ± 6.05 MU`) and `dfl_dro` (`1320.57 ± 6.98 MU`) beat both `supervised_pto` (`1322.71 ± 4.41 MU`) and `supervised_dro` (`1324.02 ± 12.67 MU`)**.
  - On steady single-period staples, trailing 8-week `proportional_allocation` has slightly lower mean cost (**1298.91 MU**), **but `supervised_dro` (`1677.40 MU`), `dfl_pto` (`1686.33 MU`), and `dfl_dro` (`1689.26 MU`) all beat `proportional_allocation` (`1711.69 MU`) on worst-week tail cost ($\text{CVaR}_{0.90}$)**.
- **2. Multi-Period Rolling Inventory Carryover & Lead Time (`L=0` and `L=1` weeks on M5):**
  - When leftover inventory carries over across weeks and orders face replenishment lead times (`InventoryStateTransitionSystem` with strict mass conservation `is_conserved = True`), static heuristics over-ship into already stocked stores:
    - At **`L=0` (Immediate Replenishment):** **`dfl_dro` wins #1 (`1230.94 MU`, `76.5%` fill rate, regret `242.18 MU`)**, followed by **`supervised_dro` #2 (`1233.16 MU`)** and **`dfl_pto` #3 (`1459.18 MU`)**, beating `supervised_pto` (`1526.37 MU`), `proportional_allocation` (`1620.85 MU`), and `equal_allocation` (`2080.03 MU`).
    - At **`L=1` (1-Week Lead Time, zero-leakage $\mathcal{F}_t$ feature projection):** **`dfl_dro` wins #1 (`1526.90 MU`, `70.2%` fill rate, regret `530.40 MU`)**, followed by **`supervised_dro` #2 (`1565.34 MU`)** and **`dfl_pto` #3 (`1566.20 MU`)**, beating `supervised_pto` (`1628.22 MU`), `proportional_allocation` (`1880.69 MU`), and `equal_allocation` (`2076.93 MU`).
- **3. Intermittent/Lumpy M5 Demand Regime (`m5_intermittent_v1`, 10 lumpy `HOUSEHOLD` SKUs, `ADI >= 1.35`, [`reports/m5_intermittent/results_summary.md`](../reports/m5_intermittent/results_summary.md)):**
  - On lumpy SKUs, trailing 8-week `proportional_allocation` collapses to **last place (`176.94 MU`, $\text{CVaR}_{0.90} = \text{274.42 MU}$)**, while **`supervised_dro` (`171.30 MU`) and `dfl_dro` (`172.59 MU`) win #1 and #2 overall**, and **`dfl_pto` (`253.09 MU`) achieves the lowest realized tail risk ($\text{CVaR}_{0.90}$)**!

### Slide 7 — Limitations & Production Roadmap for MR. D.I.Y. (~1 min)
- **Candor on Assumptions:** Censored sales proxy (`observed sales != latent demand`), synthetic inventory capacities (`MU`), and small-origin bootstrap intervals (`12 < 20` origins).
- **First 90 Days at MR. D.I.Y.:** Connect to actual ERP stock-on-hand + warehouse case-pack constraints, impute stockout censoring using intraday POS velocity, and deploy the Page D what-if allocation simulator for weekly replenishment planners.

---

## 3. 60-Second Offline Live-Demo Runbook

Before walking into the interview room, start the local server once (no internet required):
```bash
PYTHONPATH=src .venv/bin/streamlit run app/streamlit_app.py
```

1. **0:00–0:15 (`Page A — Problem & Data`):** Switch between `M5 Mode (M5 Sales Proxy)`, `M5 Intermittent Mode (Lumpy/Intermittent SKUs)`, and `Demo Mode (DEMO / SYNTHETIC)` in the sidebar. Point out the provenance badge, the weekly demand chart across chronological `train`/`val`/`test` splits, and the explicit `MU` synthetic inventory contract.
2. **0:15–0:30 (`Page B — Models & Diagnostics` & `Page C — Allocation Results`):** Show the joint multi-SKU DFL training curve, validation $\lambda_{\text{DFL}}$ ablation (`17.59 MU -> 13.04 MU` validation regret reduction on M5), and store-level integer allocations on Page C with verified `optimal` solver status and downloadable CSV.
3. **0:30–0:45 (`Page D — Risk & Robustness`):** Point out the validation-selected $(\alpha^*=0.90, \text{risk\_weight}^*=0.25)$ banner and the **Full-Horizon Cost–Risk–Service-Level Trade-Off Frontier**, then adjust the **What-If Simulation Controls** (`Warehouse Supply Scarcity Multiplier = 0.8`, `Pricing / Display Demand Uplift = +25%`, `Shortage Penalty Priority Multiplier = 1.5`) to re-solve the sparse MILP instantaneously (`<50 ms`) while keeping *Model Expected Cost*, *Model $\text{CVaR}_\alpha$*, and *Realized Historical Test Cost* strictly separated.
4. **0:45–1:00 (`Page E — Evidence & Business Impact`):** Highlight the 7-policy comparison table, the **Independent Training-Seed Stability (`[42, 101, 202]`)**, the **Rolling Inventory Carryover (`L=0` & `L=1` where `dfl_dro` wins #1 overall)**, and the **Regular vs. Intermittent M5 Demand Regime table**.
