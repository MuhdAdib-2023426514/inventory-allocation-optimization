# Interview Presentation Guide — Data Scientist Executive (Decision Intelligence), MR. D.I.Y.

**Project Title:** Decision-Focused and Distributionally Robust Inventory Allocation  
**Subtitle:** Decision-Oriented Learning for Risk-Aware Multi-Store Stock Allocation  
**Dynamic Results Reference:** [`reports/results_summary.md`](../reports/results_summary.md)

---

## 1. Interview Alignment (The 5 Required Sections + Limitations)

| Interview Prompt Section | Core Message for MR. D.I.Y. Panel | Dashboard Page |
| :--- | :--- | :--- |
| **1. Problem / Background** | When central warehouse supply $W_p$ is scarce relative to multi-store demand, equal or static proportional allocation rules waste stock at low-demand stores while starving high-demand stores. Minimizing forecast MAE/WAPE alone does not minimize downstream shortage and holding costs. | **Page A — Problem & Data** |
| **2. Approach / Methodology** | Formulate integer multi-store stock allocation $C(x, D)$ with exact marginal-unit binary encoding $y_{s,k}$. Train a discrete demand PMF end-to-end with PyEPO `SPOPlus` ($\text{NLL} + \lambda_{\text{DFL}}\text{SPO}+$) and feed learned distributions into an explicit probability-reweighting $\text{CVaR}_\alpha$ DRO MILP. | **Page B — Models & Diagnostics** |
| **3. Analysis** | Demonstrate the mandatory 2-store mathematical oracle ($D=[60,20], W=60$: equal `30/30` gives `62.5%` fill rate vs. optimal `40/20` giving `75.0%` fill rate), inspect survival probability curves $q[s,k]$, and analyze $\lambda_{\text{DFL}}$ validation ablations. | **Page B & Page C** |
| **4. Solution / Recommendation** | Provide an offline-ready Decision Intelligence simulator that outputs feasible integer store shipments $x_{sp}$ and lets planners stress-test warehouse supply scarcity and tail-risk weights ($\alpha, \text{risk\_weight}$) in $<50\text{ ms}$ without retraining models. | **Page C & Page D** |
| **5. Results / Business Impact** | Present the verified 7-policy out-of-sample comparisons across both modes: (1) In `DEMO / SYNTHETIC` (`demo_v1`, [`reports/demo/results_summary.md`](../reports/demo/results_summary.md)), `dfl_pto` achieves the lowest mean cost (**50.48 MU**, `-5.0%` vs `equal_allocation` at **53.13 MU** and `-1.7%` vs `supervised_pto` at **51.37 MU**) and highest non-oracle fill rate (**86.1%**). (2) In empirical Walmart M5 (`m5_v1`, 5 stores $\times$ 10 `HOUSEHOLD` SKUs, 12 test origins, [`reports/m5/results_summary.md`](../reports/m5/results_summary.md)), `dfl_pto` (**1315.41 MU**) beats `supervised_pto` (**1322.76 MU**) and reduces cost by **-8.2% (-116.94 MU)** vs `equal_allocation` (**1432.35 MU**), while achieving the **lowest realized tail risk ($\text{CVaR}_{0.90} = 1688.20\text{ MU}$)** of all non-oracle policies—even beating `proportional_allocation` (`1711.69 MU`) on worst-week tail cost. | **Page E — Evidence & Business Impact** |
| **6. Limitations / Next Steps** | Explicitly state that observed sales $\ne$ uncensored demand, inventory/cost parameters are synthetic (`MU`), and our DRO uses the explicit scenario-probability-reweighting $\text{CVaR}_\alpha$ ambiguity set $\mathcal{P}_\alpha$, not Wasserstein DRO. | **Page A, D, E** |

---

## 2. Seven-Slide Talk Track (~8–10 Minutes)

### Slide 1 — Executive Summary & The Retail Allocation Problem (~1 min)
- **Hook:** "In multi-store retail like MR. D.I.Y., forecasting demand is only half the problem—the real financial impact comes from the **allocation decision** when warehouse stock is constrained."
- **Key Point:** Show how optimizing forecast accuracy (MAE/WAPE) treats over-forecasting and under-forecasting symmetrically, whereas retail economics are asymmetric (stockout penalty $b_{sp} \gg$ holding cost $h_{sp}$) and coupled across stores by warehouse supply $W_p$ and store capacity $C_s$.

### Slide 2 — Mathematical Formulation & Toy Oracle Proof (~1.5 min)
- **Exact Cost Function:** $C(x, D) = \sum_{s,p} [b_{sp}\max(D_{sp}-I_{sp}-x_{sp},0) + h_{sp}\max(I_{sp}+x_{sp}-D_{sp},0) + t_{sp}x_{sp}]$.
- **Toy Oracle Intuition:** Walk through 1 SKU, 2 stores, $D=[60, 20], W=60$. Equal split `30/30` wastes 10 units at Store B and leaves 30 units unmet at Store A (`62.5%` fill rate, cost `30 MU`). Optimal allocation `40/20` fulfills all 60 available units (`75.0%` fill rate, cost `20 MU`).

### Slide 3 — Architecture: Why Standard Forecasting Fails & How Genuine DFL Works (~1.5 min)
- **Marginal-Unit Encoding:** Explain why we introduce binary variables $y_{s,k} \in \{0,1\}$ with prefix constraints $y_{s,k+1} \le y_{s,k}$ and prove $C(x,d) = C(0,d) + \sum_{s,k} c_{\text{true}}[s,k;d]y_{s,k}$.
- **Differentiable Survival Mapping:** Our PyTorch PMF network predicts $p(d \mid \text{features})$ over $0 \dots H$ plus an `overflow > H` bin, computes survival probabilities $q[s,k] = P(D_s \ge I_s + k)$, and maps them to expected marginal costs $\hat{c}[s,k] = t_s + h_s - (b_s + h_s)q[s,k]$.
- **End-to-End PyEPO `SPOPlus` Training:** Contrast `supervised_pto` ($\text{NLL}$ only) with `dfl_pto` ($\text{NLL} + \lambda_{\text{DFL}}\text{SPO}+$ solved via HiGHS).

### Slide 4 — Risk-Aware Allocation via Probability-Reweighting DRO ($\text{CVaR}_\alpha$) (~1.5 min)
- **Joint Scenario Generation:** Use a training-only empirical rank-bootstrap copula to sample $N$ cross-store correlated demand scenarios from the learned PMFs.
- **Explicit Ambiguity Set:** Define $\mathcal{P}_\alpha = \{w \ge 0 : \sum w_j = 1, w_j \le p_j/(1-\alpha)\}$ and show Rockafellar–Uryasev equivalence to $\text{CVaR}_\alpha(L(x))$.
- **Three Distinct Metrics:** Emphasize why a Decision Intelligence system must never conflate *Model-Implied Expected Cost*, *Model-Implied $\text{CVaR}_\alpha$*, and *Realized Held-Out Test Cost*.

### Slide 5 — Live Decision-Experiment Walkthrough (Pages A–D) (~1.5 min)
- Switch to the live Streamlit dashboard running locally on your laptop (see 60-Second Demo Path below).

### Slide 6 — Empirical Backtest Evidence & Candid Analysis (~1.5 min)
- Walk through the 7-policy table in [`reports/results_summary.md`](../reports/results_summary.md) (`DEMO / SYNTHETIC` run `demo_v1`):
  - `equal_allocation`: **53.13 MU** (`84.5%` fill rate, regret **29.41 MU**)
  - `proportional_allocation`: **52.16 MU** (`85.2%` fill rate, regret **28.44 MU**)
  - `supervised_pto`: **51.37 MU** (`85.6%` fill rate, regret **27.64 MU**)
  - `dfl_pto`: **50.48 MU** (`86.1%` fill rate, regret **26.75 MU** — best realized mean cost and fill rate)
  - `supervised_dro` (**55.98 MU**) & `dfl_dro` (**62.33 MU**): Candidly explain why setting high tail-risk aversion ($\alpha=0.90, \text{risk\_weight}=0.50$) reduces model-implied tail risk ($\text{CVaR}_{0.90}$ drops from `255.96 MU` to `219.55 MU`) but increases average realized cost on moderate test weeks.
- Walk through the empirical Walmart M5 (`m5_v1`, [`reports/m5/results_summary.md`](../reports/m5/results_summary.md)) and `DEMO / SYNTHETIC` (`demo_v1`, [`reports/demo/results_summary.md`](../reports/demo/results_summary.md)) results side by side:
  - **On Real Walmart M5 (`5 stores x 10 HOUSEHOLD SKUs, 12 test weeks`):**
    - `dfl_pto` (**1315.41 MU**) outperforms `supervised_pto` (**1322.76 MU**) and `equal_allocation` (**1432.35 MU**, `-8.2%` cost reduction), and `dfl_dro` (**1326.05 MU**) outperforms `supervised_dro` (**1339.99 MU**).
    - **Candid Scientific Insight:** On steady-volume M5 household staples, trailing 8-week `proportional_allocation` achieves slightly lower mean cost (**1298.91 MU**), **but `dfl_pto` achieves the lowest realized worst-tail cost ($\text{CVaR}_{0.90} = 1688.20\text{ MU}$ vs `1711.69 MU` for `proportional_allocation`)**, and on shock-heavy demand (`demo_v1`), `dfl_pto` wins outright on mean cost (**50.48 MU** vs **52.16 MU**).

### Slide 7 — Limitations & Production Roadmap for MR. D.I.Y. (~1 min)
- **Candor on Assumptions:** Censored sales proxy (`observed sales != latent demand`), synthetic inventory capacities (`MU`), and small-origin bootstrap intervals.
- **First 90 Days at MR. D.I.Y.:** Connect to actual ERP stock-on-hand + warehouse case-pack constraints, impute stockout censoring using intraday POS velocity, and deploy the Page D what-if allocation simulator for weekly replenishment planners.

---

## 3. 60-Second Offline Live-Demo Runbook

Before walking into the interview room, start the local server once (no internet required):
```bash
PYTHONPATH=src .venv/bin/streamlit run app/streamlit_app.py
```

1. **0:00–0:15 (`Page A — Problem & Data`):** Point out the `DEMO / SYNTHETIC` provenance badge, the 3-store $\times$ 2-SKU weekly demand chart across chronological `train`/`val`/`test` splits, and the explicit `MU` synthetic inventory contract.
2. **0:15–0:30 (`Page B — Models & Diagnostics` & `Page C — Allocation Results`):** Show the DFL training curve, validation $\lambda_{\text{DFL}}$ ablation, and store-level integer allocations on Page C with verified `optimal` solver status and downloadable CSV.
3. **0:30–0:45 (`Page D — Risk & Robustness`):** Drag the `Tail Risk Weight (risk_weight)` slider from `0.0` to `0.75` and `Warehouse Supply Scarcity Multiplier` to `0.8`. Show the MILP re-solving instantaneously offline while keeping *Model Expected Cost*, *Model $\text{CVaR}_\alpha$*, and *Realized Historical Test Cost* strictly separated.
4. **0:45–1:00 (`Page E — Evidence & Business Impact`):** Highlight the 7-policy comparison chart showing `dfl_pto` beating `supervised_pto`, `proportional_allocation`, and `equal_allocation` on realized cost and fill rate, plus the SHA-256 reproducibility manifest.
