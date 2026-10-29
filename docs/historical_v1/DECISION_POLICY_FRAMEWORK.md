# Retail Decision Intelligence Policy Framework
## Empirical Policy Selection Matrix for Multi-Store Inventory Allocation
**Target Role:** Data Scientist Executive (Decision Intelligence), MR.DIY  
**Framework Version:** 1.0 (Production & Interview Specification)  
**Methodological Scope:** Decision-Focused Learning (DFL), Distributionally Robust Optimization (DRO), and Benchmark Allocators  

---

## 1. Executive Summary & Retail Context

In large-scale discount retail environments such as MR.DIY (managing 1,200+ stores across diverse geographic clusters and tens of thousands of SKUs spanning Hardware, Household, Electrical, Furnishing, and Stationery), centralized warehouse inventory allocation represents a high-stakes, asymmetric operational challenge.

Conventional retail allocation relies on **Forecast-Then-Optimize (PTO)**: supervised time-series models (e.g., LightGBM, DeepAR, or Poisson regression) are trained independently to minimize symmetric statistical errors (Root Mean Squared Error, Mean Absolute Error, or Negative Log-Likelihood), and their point or distribution forecasts are subsequently passed into a downstream linear or mixed-integer programming (MILP) solver. 

While intuitive, two structural flaws degrade profitability under real-world conditions:
1. **Objective Mismatch:** Statistical training treats positive and negative prediction errors identically or symmetrically, whereas retail stockouts impose severe business costs (lost margin, lost customer goodwill, competitor substitution) compared to holding costs (inventory financing, shelf congestion).
2. **Distributional Tail Risk:** Point forecasts and uncalibrated scenario models ignore correlated demand surges across regional store clusters and underestimate extreme tail outcomes, causing systemic stockouts during promotional spikes or seasonal events.

To solve this, our framework evaluates four advanced paradigms alongside classical heuristics:
- **Equal Allocation (`equal_allocation`)**: Uniform warehouse distribution respecting store caps.
- **Proportional Allocation (`proportional_allocation`)**: Allocation proportional to trailing sales velocity.
- **Supervised PTO (`supervised_pto`)**: Maximum Likelihood Estimation (MLE) PMF $\rightarrow$ Nominal Expected-Cost MILP.
- **Decision-Focused Learning (`dfl_pto`)**: Smart Predict-then-Optimize (SPO+) trained PMF $\rightarrow$ Nominal Expected-Cost MILP.
- **Scenario-Reweighted DRO (`supervised_dro`)**: Supervised PMF $\rightarrow$ Probability-Reweighted $\text{CVaR}_\alpha$ Ambiguity Set MILP.
- **Integrated DFL + DRO (`dfl_dro`)**: Decision-trained PMF $\rightarrow$ Empirical Copula Scenarios $\rightarrow$ $\text{CVaR}_\alpha$ Ambiguity Set MILP.
- **Clairvoyant Oracle (`clairvoyant_oracle`)**: Theoretical upper-bound benchmark with perfect hindsight demand.

This document establishes the **empirical policy selection rules** derived from our rigorous experiments across 12-week held-out test windows on Walmart M5 Household categories (both regular and intermittent demand regimes) and verified multi-period carryover simulations.

---

## 2. Empirical Decision Policy Matrix

The following decision matrix provides category managers, inventory controllers, and decision systems engineers with unambiguous, rule-based dispatch criteria for store allocation.

| Demand Regime | Operating Horizon & Supply Condition | Shortage Penalty Ratio ($b / h$) | Target Fill Rate ($\beta^*$) | Recommended Policy | Primary Rationale & Mathematical Mechanism | Observed Benchmark Evidence (`m5_v1`, `m5_intermittent_v1`, `demo_v1`) | Optimization Overhead |
|---|---|---|---|---|---|---|---|
| **Regular / Fast-Moving (`m5_v1`, $\text{ADI} \le 1.25$)** | **Multi-Period Carryover ($L=0, L=1$)** | **High** ($\ge 8:1$) | $\ge 75\%$ | **`dfl_dro`** (Integrated DFL + DRO) | Under rolling inventory carryover, SPO+ shapes marginal survival probabilities while DRO ($\alpha^*=0.90, \lambda_{\text{risk}}^*=0.25$) prevents over-allocation into already stocked stores. | **#1 at $L=0$ (`1230.94 MU`)**, reducing cost by **-19.4% (-295.43 MU)** and oracle regret by **-55.0%** vs. `supervised_pto` (`1526.37 MU`); **#1 at $L=1$ (`1526.90 MU`)** vs. `supervised_pto` (`1628.22 MU`, **-6.2%**). | Moderate (Offline SPO+ training; <40ms online MILP solve). |
| **Regular / Fast-Moving (`m5_v1` & `demo_v1`)** | **Single-Period Constrained Snapshot ($I=0$)** | **Moderate / High** ($4:1 - 10:1$) | $73\% - 86\%$ | **`dfl_pto`** / **`supervised_dro`** | DFL directly minimizes expected downstream regret (`1312.70 MU` single-period, `1305.74 ± 6.05 MU` across 3 seeds), while `supervised_dro` achieves the lowest tail risk. | **`dfl_pto` (`1312.70 MU`)** cuts cost by **-8.4% (-119.64 MU)** vs. `equal_allocation` (`1432.35 MU`) and beats `supervised_pto` (`1322.76 MU`); **`supervised_dro` achieves lowest $\text{CVaR}_{0.90} = \text{1677.40 MU}$** (**-30.72 MU** vs. `supervised_pto`). | Low online (<10ms nominal LP / <38ms DRO MILP). |
| **Intermittent / Lumpy (`m5_intermittent_v1`, $\text{ADI} \ge 1.35$)** | **Single-Period & Multi-Period Carryover** | **High** ($\ge 8:1$) | $55\% - 67\%$ | **`supervised_dro`** & **`dfl_dro`** | Intermittent SKUs exhibit zero runs and sporadic spikes where trailing heuristics fail; calibrated PMFs + DRO scenario reweighting buffer lumpy tail surges. | **#1 (`171.30 MU`) and #2 (`172.59 MU`)** single-period mean cost vs. `proportional_allocation` (`176.94 MU`, last place), reducing $\text{CVaR}_{0.90}$ from `274.42 MU` to `256.92 MU` (**-17.50 MU**) and `254.54 MU` (**-19.88 MU**); **-38.0% cost reduction** at $L=0$ (`149.14 MU` vs. `240.44 MU`). | Low (<38ms scenario MILP solve). |
| **Regular Steady Staples (`m5_v1`)** | **Single-Period Static Snapshot ($I=0$), Risk-Neutral** | **Moderate** ($\le 5:1$) | Standard ($\approx 74\%$) | **`proportional_allocation`** (Trailing 8-Week Heuristic) | On steady single-period staples with zero opening stock and no carryover dynamics, trailing 8-week demand shares closely track mean store velocity without solver overhead. | Achieves **1298.91 MU** single-period mean cost on `m5_v1` (**-1.0% / -13.79 MU** lower mean cost than `dfl_pto` at `1312.70 MU`, though with higher tail $\text{CVaR}_{0.90} = \text{1711.69 MU}$ vs. `1677.40 MU` for `supervised_dro`). | Zero solver overhead (deterministic analytical rounding). |
| **Critical / Promotional** | Any Supply Level | **Severe** ($> 15:1$) | Hard constraint ($\ge 85\% - 95\%$) | **`service_level_milp`** (Service-Constrained Allocation) | When executive business rules enforce non-negotiable shelf availability, nominal economic optimization must be replaced by explicit expected fill-rate constraints with physical shortfall proof. | Formally proves whether warehouse supply is physically sufficient (`is_service_level_met`); calculates exact minimum unit shortfall (`shortfall_units`) via linear slack penalty. | Low (<42ms MILP solve with slack penalty). |

---

## 3. Deep-Dive Policy Analyses

### 3.1 Decision-Focused Learning (`dfl_pto` via PyEPO SPO+)

#### Theoretical Foundation
Conventional PTO minimizes Negative Log-Likelihood (NLL) over discrete demand levels $d \in \{0, \dots, H\}$:
$$\mathcal{L}_{\text{NLL}}(\theta) = -\sum_{i} \log p_\theta(d_i \mid x_i)$$
However, PyEPO's Smart Predict-then-Optimize (SPO+) surrogate loss optimizes the parameters $\theta$ directly against the downstream allocation regret:
$$\mathcal{L}_{\text{SPO+}}(c_\theta, c_{\text{true}}, w^*) = \max_{w \in \mathcal{S}} \left\{ (c_{\text{true}} - 2 c_\theta)^\top w \right\} + 2 c_\theta^\top w^* - c_{\text{true}}^\top w^*$$
where $w^*$ is the clairvoyant optimal allocation under true realized demand, and $c_\theta$ represents the predicted marginal unit cost vector:
$$c_{\theta}[s, k] = t_s + h_s - (b_s + h_s) \cdot P_\theta(D_s \ge I_s + k)$$

#### Why DFL Outperforms under Supply Rationing
When total warehouse inventory $W_p < \sum_s \mathbb{E}[D_{sp}]$, the allocation problem becomes a zero-sum competition across stores. Standard PTO distributes forecast errors uniformly across the distribution. In contrast, DFL concentrates model precision on the **marginal survival probability thresholds** ($q[s, k]$) that dictate which store receives the scarce $k$-th unit. Units that are guaranteed to sell or guaranteed to sit unsold receive zero optimization gradient, preventing model capacity from being wasted on decision-irrelevant regions of the demand distribution.

#### Hybrid Training Requirement: $\mathcal{L} = \mathcal{L}_{\text{NLL}} + \lambda_{\text{DFL}} \mathcal{L}_{\text{SPO+}}$
Pure SPO+ training can lead to uncalibrated probability tails because multiple degenerate PMFs can yield identical optimal discrete decisions. By combining NLL (proper scoring rule) with SPO+ ($\lambda_{\text{DFL}} \approx 1.0$), we ensure both well-calibrated scenario sampling and decision-optimal marginal ranking.

---

### 3.2 Distributionally Robust Optimization (`supervised_dro` & `dfl_dro`)

#### Ambiguity Set Formulation
Rather than relying on unverified Wasserstein heuristics, our framework implements an explicit **scenario probability-reweighting ambiguity set** $\mathcal{P}_\alpha$:
$$\mathcal{P}_\alpha = \left\{ w \in \mathbb{R}^N : \sum_{j=1}^N w_j = 1, \; 0 \le w_j \le \frac{p_j}{1 - \alpha} \right\}, \quad 0 \le \alpha < 1$$
The robust worst-case loss over this ambiguity set is mathematically equivalent to the **Conditional Value-at-Risk (CVaR)** at confidence level $\alpha$:
$$\max_{w \in \mathcal{P}_\alpha} \sum_{j=1}^N w_j L_j(x) = \text{CVaR}_\alpha(L(x)) = \min_{\eta \in \mathbb{R}} \left\{ \eta + \frac{1}{1 - \alpha} \sum_{j=1}^N p_j [L_j(x) - \eta]^+ \right\}$$

#### Practical Risk Balancing
The operational objective solved at each weekly replenishment origin is:
$$\min_{x \in \mathcal{X}} (1 - \lambda_{\text{risk}}) \sum_{j=1}^N p_j L_j(x) + \lambda_{\text{risk}} \cdot \text{CVaR}_\alpha(L(x))$$
- $\lambda_{\text{risk}} = 0.0$: Pure stochastic programming (risk-neutral, assumes scenarios are exactly true).
- $\lambda_{\text{risk}} > 0.0$ (e.g., $0.25$ or $0.50$): Robust hedging against adverse scenario realizations and correlated store shocks.
- Validation tuning on held-out validation folds (`split == 'val'`) selects $(\alpha^* = 0.90, \lambda_{\text{risk}}^* = 0.25)$ on `m5_v1` (and $\lambda_{\text{risk}}^* = 0.50$ for `supervised_dro` on `m5_intermittent_v1`) to balance mean expected cost and worst-case tail regret.

---

### 3.3 When Simple Heuristics Suffice (`proportional_allocation`)

A critical hallmark of mature Decision Intelligence is knowing **when NOT to deploy complex optimization models**.
In our empirical benchmarks:
1. **Single-Period Steady Staples with Zero Opening Stock (`m5_v1`):** On regular-selling staples ($I_{sp}=0$, no multi-week carryover), trailing 8-week `proportional_allocation` achieves `1298.91 MU` mean cost vs. `1312.70 MU` for `dfl_pto` (**-1.0% / -13.79 MU** lower mean cost), though with higher worst-week tail cost ($\text{CVaR}_{0.90} = \text{1711.69 MU}$ vs. `1677.40 MU` for `supervised_dro` and `1686.33 MU` for `dfl_pto`).
2. **Where Static Heuristics Break Down:** Under multi-period rolling inventory carryover ($L=0$ on `m5_v1`), `proportional_allocation` degrades to `1620.85 MU` (**+31.7% / +389.91 MU worse** than `dfl_dro` at `1230.94 MU`), and on intermittent/lumpy SKUs (`m5_intermittent_v1`), `proportional_allocation` finishes last at `176.94 MU` (**+3.3% worse** than `supervised_dro` at `171.30 MU` in single-period and **+61.2% worse** at $L=0$ carryover, `240.44 MU` vs. `149.14 MU`).
3. **Emergency Store-Level Fallback:** If network connectivity to the central optimization engine is interrupted, store systems can deterministically fall back to `proportional_allocation` using local 8-week sales history without violating warehouse supply or store storage limits.

---

## 4. Multi-Period Dynamics: Lead Times & Inventory Carryover

Real-world retail replenishment does not operate in an isolated single-period vacuum. Our evaluation incorporates the `InventoryStateTransitionSystem`, auditing strict mass conservation across rolling multi-period horizons:
$$I_{t+1, s, p} = I_{t, s, p} + a_{t, s, p} - y_{t, s, p}$$
$$\text{Pipeline}_{t+1} = \text{Pipeline}_t + x_t - a_t$$

### Key Multi-Period Findings
1. **Lead Time Penalty ($L=1$ vs. $L=0$ on `m5_v1`):** Introducing a 1-week replenishment transit lead time ($L=1$) increases rolling mean inventory cost from `1230.94 MU` to `1526.90 MU` (**+24.0% / +295.96 MU**) for `dfl_dro` and from `1526.37 MU` to `1628.22 MU` (**+6.7% / +101.85 MU**) for `supervised_pto`. Under non-zero lead times, orders placed at time $t$ cannot satisfy demand at time $t$; policies must project arrival-week opening stock and features strictly from $\mathcal{F}_t$ (`_project_features_one_week_ahead`).
2. **Pipeline-Aware vs. Myopic Decisions:** Myopic policies that ignore pipeline inventory ($\text{in-transit}$) repeatedly over-order during temporary stockouts, causing massive inventory bullwhip and store capacity breaches upon shipment arrival. All advanced policies (`supervised_pto`, `dfl_dro`, `service_level_milp`) explicitly subtract outstanding pipeline shipments from net inventory requirements:
$$\text{Net Inventory} = I_{\text{on-hand}} + \text{In-Transit} - \text{Backorders}$$

---

## 5. MR.DIY Production Implementation Roadmap

To deploy this Decision Intelligence framework across MR.DIY's supply chain ecosystem, we recommend a phased 4-stage operational roadmap:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MR.DIY DECISION ROADMAP                               │
├─────────────────────┬─────────────────────┬───────────────────┬─────────────┤
│      Phase 1        │      Phase 2        │      Phase 3      │   Phase 4   │
│ Shadow Deployment   │  Category Rollout   │  Joint Multi-SKU  │ Automated   │
│ & Nonanticipativity │  (High-Value FMCG)  │ Capacity Coupling │ Replenish   │
│     (Weeks 1-4)     │    (Weeks 5-10)     │   (Weeks 11-16)   │ (Weeks 17+) │
└─────────────────────┴─────────────────────┴───────────────────┴─────────────┘
```

### Stage 1: Shadow Deployment & Digital Twin Auditing (Weeks 1–4)
- Run `InventoryStateTransitionSystem` in passive shadow mode against actual daily/weekly MR.DIY DC shipment logs.
- Audit real-world data feeds for temporal leakage, verifying that all features and promotional flags are strictly lagged relative to order placement cutoffs.
- Validate solver latency on production hardware: ensure sparse HiGHS MILP solves complete within <50ms per store cluster (10 stores $\times$ 50 SKUs).

### Stage 2: Category Rollout on High-Asymmetry SKUs (Weeks 5–10)
- Deploy `dfl_dro` on top-revenue, high-margin Household and Electrical SKUs where $b/h \ge 8:1$ and supply shortages are recurrent.
- Utilize the Streamlit Decision Dashboard (Pages A–E) as a Category Manager Decision Cockpit, enabling planners to run what-if simulations before confirming weekly warehouse releases.
- Establish an automated A/B rollout across store regions (e.g., Central Region on `dfl_dro` vs. Northern Region on conventional PTO).

### Stage 3: Joint Multi-SKU Capacity Coupling & Fleet Routing (Weeks 11–16)
- Expand the linear objective to enforce joint store cubic-meter storage constraints ($\sum_{p} v_p (I_{sp} + x_{sp}) \le C_s$) and cross-SKU warehouse pallet limits.
- Incorporate store delivery schedules and multi-stop truckload minimums into the MILP constraints.

### Stage 4: Automated Closed-Loop Replenishment (Weeks 17+)
- Connect optimized shipment vectors $x_{sp}^*$ directly to MR.DIY's Warehouse Management System (WMS) via automated EDI/API purchase order creation.
- Establish real-time drift detection monitoring: trigger automatic model retraining whenever realized out-of-sample regret exceeds 1.5x historical validation thresholds.

---

## 6. Executive Talking Points for Data Scientist Executive Interview

When presenting this framework to MR.DIY leadership:

1. **On Business Value:**  
   *"We do not measure success by forecast accuracy (WAPE or RMSE). We measure success by bottom-line inventory cost, customer demand fill rate, and worst-case regret. On our 12-week Walmart M5 multi-period carryover benchmark (`m5_v1`, $L=0$), our integrated `dfl_dro` pipeline reduces mean inventory cost by **-19.4% (-295.43 MU)** and cuts oracle regret by **-55.0%** compared to conventional `supervised_pto` (`1230.94 MU` vs. `1526.37 MU`)."*

2. **On Technical Rigor & Leakage:**  
   *"This codebase verifies nonanticipativity across the full feature-to-allocation pipeline (`tests/test_simulation_and_nonanticipativity.py`). Every forecast origin uses strictly lagged covariates. Furthermore, our multi-period simulation enforces exact mass conservation—no inventory units are magically scaled, floored, or destroyed."*

3. **On Pragmatic Simplicity:**  
   *"We do not advocate deep neural networks or complex MILPs for every item. On single-period steady staples with zero opening stock (`m5_v1`), our empirical benchmark shows that trailing 8-week `proportional_allocation` achieves `1298.91 MU` mean cost (`-1.0%` below `dfl_pto` at `1312.70 MU`, though with higher tail $\text{CVaR}_{0.90}$ of `1711.69 MU` vs. `1677.40 MU` for `supervised_dro`), whereas under multi-period carryover or intermittent demand (`m5_intermittent_v1`), `proportional_allocation` degrades substantially and `dfl_dro` / `supervised_dro` win #1."*

4. **On Solvability & Production Readiness:**  
   *"The entire pipeline is open-source and CPU-first, utilizing `scipy.optimize.milp` and HiGHS. It requires zero commercial Gurobi licenses, executes in milliseconds, and runs completely offline with verified mathematical proofs."*
