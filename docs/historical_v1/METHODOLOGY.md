# Methodology: Decision-Focused & Distributionally Robust Inventory Allocation

**Authoritative specification:** [`AGENTS.md`](../AGENTS.md)

---

## 1. Problem Formulation & Exact Cost Accounting (`AGENTS.md` §5)

Let $S$ denote the set of retail stores and $P$ denote the set of SKUs (`cat_id == "HOUSEHOLD"`). At a weekly decision origin $t_0$:
- $D_{sp} \in \mathbb{Z}_{\ge 0}$: next-7-calendar-day customer demand proxy at store $s \in S$ for SKU $p \in P$.
- $I_{sp} \in \mathbb{Z}_{\ge 0}$: opening store on-hand stock at origin $t_0$.
- $W_p \in \mathbb{Z}_{\ge 0}$: central warehouse supply available to ship for SKU $p$.
- $K_{sp} \in \mathbb{Z}_{\ge 0}$: per-store–SKU weekly allocation cap.
- $C_s \in \mathbb{Z}_{\ge 0}$: total store stock capacity across all SKUs ($\sum_{p \in P} (I_{sp} + x_{sp}) \le C_s$).
- $b_{sp} \ge 0, h_{sp} \ge 0, t_{sp} \ge 0$: unit shortage penalty, holding cost, and shipping cost in generic Monetary Units (**MU/unit**).

### Decision Variables and Feasibility Set $\mathcal{X}$
The allocation decision $x_{sp} \in \mathbb{Z}_{\ge 0}$ is the integer number of units of SKU $p$ shipped from the warehouse to store $s$ at origin $t_0$:

$$
\mathcal{X} = \left\{ x \in \mathbb{Z}_{\ge 0}^{|S| \times |P|} \;\middle|\;
0 \le x_{sp} \le K_{sp} \;\forall s,p; \quad
\sum_{s \in S} x_{sp} \le W_p \;\forall p; \quad
\sum_{p \in P} (I_{sp} + x_{sp}) \le C_s \;\forall s
\right\}
$$

### Exact Realized One-Week Cost
For any feasible integer allocation $x \in \mathcal{X}$ and realized demand vector $D$, the one-week realized cost is:

$$
C(x, D) = \sum_{s \in S} \sum_{p \in P} \Big[
b_{sp} \max(D_{sp} - I_{sp} - x_{sp}, 0)
+ h_{sp} \max(I_{sp} + x_{sp} - D_{sp}, 0)
+ t_{sp} x_{sp}
\Big]
$$

Fulfilled demand is $\min(D_{sp}, I_{sp} + x_{sp})$ and unmet demand is $\max(D_{sp} - I_{sp} - x_{sp}, 0)$.

---

## 2. Marginal-Unit Binary Encoding & Incremental Cost Identity (`AGENTS.md` §6)

Standard `SPOPlus` in PyEPO requires uncertain parameters to appear **linearly in the objective function** over an **instance-invariant feasible set**. In the raw formulation $C(x, D)$, random demand $D_{sp}$ sits inside $\max(\cdot)$ terms (or on the right-hand side of linearization constraints $u_{sp} \ge D_{sp} - I_{sp} - x_{sp}$).

To make the objective strictly linear in the decision variables with fixed constraints for each SKU $p$:
1. Introduce binary marginal-unit variables $y_{s,k} \in \{0, 1\}$ for store $s \in S$ and incremental unit $k \in \{1, \dots, K_s\}$, where $y_{s,k} = 1$ iff at least $k$ units are shipped to store $s$.
2. Enforce prefix ordering constraints and the per-SKU warehouse supply budget:
   $$y_{s, k+1} \le y_{s, k} \quad \forall s \in S, \; k = 1, \dots, K_s - 1, \qquad \sum_{s \in S} \sum_{k=1}^{K_s} y_{s, k} \le W_p$$
   Under prefix ordering, $x_s = \sum_{k=1}^{K_s} y_{s, k}$.
3. Define the true incremental realized cost of the $k$-th unit shipped to store $s$ under realized demand $d_s$:
   $$
   c_{\text{true}}[s, k; d_s] =
   \begin{cases}
   t_s - b_s & \text{if } d_s \ge I_s + k \quad (\text{unit prevents 1 unit of shortage}) \\
   t_s + h_s & \text{if } d_s < I_s + k \quad (\text{unit becomes 1 unit of excess stock})
   \end{cases}
   $$
4. **Exact Decomposition Identity** (verified across randomized instances in [`tests/test_cost_identity.py`](../tests/test_cost_identity.py)):
   $$C(x, d) = C(0, d) + \sum_{s \in S} \sum_{k=1}^{K_s} c_{\text{true}}[s, k; d_s] \, y_{s, k}$$
   Because $C(0, d) = \sum_{s \in S} b_s \max(d_s - I_s, 0)$ does not depend on $x$, minimizing $\sum_{s,k} c_{\text{true}}[s, k; d_s] y_{s,k}$ is mathematically equivalent to minimizing $C(x, d)$.

---

## 3. Discrete Demand PMF, Survival Mapping & Decision-Focused Learning (`AGENTS.md` §6)

### Discrete PMF with Explicit Overflow Bin
For each SKU $p$, [`DiscreteDemandPMFNet`](../src/inventory_di/models/distributions.py) predicts a discrete categorical PMF over $\{0, 1, \dots, H\}$ plus an explicit `overflow > H` bin ($H + 2$ total outputs) from strictly pre-origin standardized features:

$$p_\theta(d \mid \text{features}_{s, t_0}), \quad d \in \{0, 1, \dots, H, \text{overflow} > H\}$$

The support bound $H$ and the mean excess above $H$ for tail sampling are estimated **strictly on `split == "train"`**. Realized held-out evaluation demand $D_{sp}$ is **never clipped at $H$**.

### Differentiable Survival-to-Marginal-Cost Mapping
For each store $s$ and unit threshold $k = 1, \dots, K_s$, the survival probability is:

$$q_\theta[s, k] = P_\theta(D_s \ge I_s + k \mid \text{features}) = \sum_{d = I_s + k}^{H} p_\theta(d \mid \text{features}) + p_\theta(\text{overflow} > H \mid \text{features})$$

Taking the expectation of $c_{\text{true}}[s, k; D_s]$ under $p_\theta$ yields the predicted marginal unit cost vector:

$$\hat{c}_\theta[s, k] = t_s + h_s - (b_s + h_s) \, q_\theta[s, k]$$

Because $q_\theta[s, k+1] \le q_\theta[s, k]$ (non-negative PMF probabilities), $\hat{c}_\theta[s, k]$ is guaranteed non-decreasing in $k$ (convex piecewise-linear expected cost in $x_s$).

### Supervised PTO vs. DFL (Joint NLL + PyEPO `SPOPlus`)
- **`supervised_pto` (`lambda_DFL = 0`):** Trained with proper categorical negative log-likelihood ($\text{NLL}$), temperature-calibrated on `split == "val"`, and passed into the nominal marginal-cost MILP at test time.
- **`dfl_pto` / `dfl_dro` (`lambda_DFL > 0`):** Trained end-to-end using PyEPO's `SPOPlus` surrogate loss over [`FixedFeasibleMarginalAllocationModel`](../src/inventory_di/optimization/pyepo_fixed.py) (`optOmoModel` with `appsi_highs`):
  $$\mathcal{L}(\theta) = \text{NLL}(\theta) + \lambda_{\text{DFL}} \cdot \frac{1}{M} \mathcal{L}_{\text{SPO}+}\big(\hat{c}_\theta, c_{\text{true}}, y^*, z^*\big)$$
  Combining $\text{NLL}$ with $\text{SPO}+$ preserves probability calibration (needed when sampling scenarios for DRO) while steering the PMF survival thresholds toward lower downstream allocation regret.

---

## 4. Distributionally Robust Optimization via Probability-Reweighting Ambiguity Set (`AGENTS.md` §7)

### Scenario Generation & Training-Only Empirical Copula
At each test origin $t_0$, we draw $N$ joint demand scenarios $D^{(1)}, \dots, D^{(N)} \in \mathbb{Z}_{\ge 0}^{|S| \times |P|}$ from either the supervised PMFs (`supervised_dro`) or the DFL-trained PMFs (`dfl_dro`). Cross-store and cross-SKU rank dependence is preserved via [`EmpiricalCopulaScenarioGenerator`](../src/inventory_di/models/distributions.py), which bootstraps historical weekly empirical rank vectors fitted **exclusively on `split == "train"`** using common random numbers across policies.

### Probability-Reweighting Ambiguity Set $\mathcal{P}_\alpha$ and $\text{CVaR}_\alpha$ Equivalence
Given scenario losses $L_j(x) = C(x, D^{(j)})$ and nominal uniform probabilities $p_j = 1/N$, we define the **probability-reweighting ambiguity set** for $\alpha \in [0, 1)$:

$$
\mathcal{P}_\alpha = \left\{ w \in \mathbb{R}_{\ge 0}^N \;\middle|\; \sum_{j=1}^N w_j = 1, \quad 0 \le w_j \le \frac{p_j}{1 - \alpha} \;\forall j = 1, \dots, N \right\}
$$

By Rockafellar–Uryasev duality, worst-case expected loss over $\mathcal{P}_\alpha$ is mathematically identical to Conditional Value-at-Risk ($\text{CVaR}_\alpha$):

$$
\max_{w \in \mathcal{P}_\alpha} \sum_{j=1}^N w_j L_j(x)
= \text{CVaR}_\alpha(L(x))
= \min_{\eta \in \mathbb{R}, \, z \in \mathbb{R}_{\ge 0}^N} \left\{ \eta + \frac{1}{1 - \alpha} \sum_{j=1}^N p_j z_j \;\middle|\; z_j \ge L_j(x) - \eta \;\forall j \right\}
$$

> **Important Distinction:** This formulation is explicit scenario-probability-reweighting / $\text{CVaR}_\alpha$ DRO. It is **not** Wasserstein DRO.

### Risk-Trade-Off MILP Objective
For $\text{risk\_weight} \in [0, 1]$, [`solve_scenario_reweighting_dro_milp`](../src/inventory_di/optimization/dro.py) solves the integer linear program:

$$
\min_{x \in \mathcal{X}, \, \eta, \, z \ge 0, \, u \ge 0, \, v \ge 0} \;
(1 - \text{risk\_weight}) \sum_{j=1}^N p_j L_j(x) + \text{risk\_weight} \left( \eta + \frac{1}{1 - \alpha} \sum_{j=1}^N p_j z_j \right)
$$

- When $\text{risk\_weight} = 0$ (or $\alpha = 0$), the problem reduces to risk-neutral stochastic expected-cost minimization.
- When $\text{risk\_weight} > 0$ and $\alpha > 0$, the allocation hedges against worst-tail demand surges across scenarios.

---

## 5. Seven-Policy Evaluation Protocol (`AGENTS.md` §8, §9)

At every held-out test origin $t_0$, all 7 policies are evaluated on the **exact same feasible set $\mathcal{X}$, opening stock $I_{sp}$, warehouse supply $W_p$, store capacities $C_s$, and held-out demand proxy $D_{\text{actual}}$**:
1. `equal_allocation`: Deterministic round-robin near-equal warehouse shares respecting $K_{sp}$ and $C_s$.
2. `proportional_allocation`: Proportional to trailing 8-week mean demand strictly prior to $t_0$, with largest-remainder rounding and feasible caps.
3. `supervised_pto`: NLL-trained PMF $\rightarrow$ nominal expected marginal-cost MILP.
4. `supervised_dro`: NLL-trained PMF $\rightarrow$ empirical copula scenarios $\rightarrow$ reweighting-DRO MILP.
5. `dfl_pto`: Joint NLL+SPO+ PMF $\rightarrow$ nominal expected marginal-cost MILP.
6. `dfl_dro`: Joint NLL+SPO+ PMF $\rightarrow$ empirical copula scenarios $\rightarrow$ reweighting-DRO MILP.
7. `clairvoyant_oracle`: Hindsight MILP solved with $D_{\text{actual}}$ known, used **solely** to compute non-negative hindsight decision regret:
   $$\text{Regret}(x_{\text{policy}}, D_{\text{actual}}) = C(x_{\text{policy}}, D_{\text{actual}}) - C(x_{\text{oracle}}, D_{\text{actual}}) \ge 0$$
