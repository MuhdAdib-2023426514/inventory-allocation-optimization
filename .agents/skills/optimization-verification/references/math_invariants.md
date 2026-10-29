# Mathematical Invariants Reference (`optimization-verification`)

All definitions strictly follow [`AGENTS.md`](../../../AGENTS.md) §5–§8 and §14.

## 1. Realized Cost & Fulfillment Accounting (`AGENTS.md` §5)

Given stores $s \in S$, SKUs $p \in P$, demand $D_{sp} \in \mathbb{Z}_{\ge 0}$, opening stock $I_{sp} \in \mathbb{Z}_{\ge 0}$, and integer allocation $x_{sp} \in \mathbb{Z}_{\ge 0}$:

- **Fulfilled units**: $\text{filled}_{sp} = \min(D_{sp}, I_{sp} + x_{sp})$
- **Unmet demand (shortage)**: $u_{sp} = \max(D_{sp} - I_{sp} - x_{sp}, 0)$
- **Excess stock (holding)**: $v_{sp} = \max(I_{sp} + x_{sp} - D_{sp}, 0)$
- **Fill rate**: $\frac{\sum_{s,p} \text{filled}_{sp}}{\sum_{s,p} D_{sp}}$ (defined as $1.0$ if $\sum_{s,p} D_{sp} = 0$)
- **Total realized cost (MU)**:
  $$C(x, D) = \sum_{s,p} \left( b_{sp} u_{sp} + h_{sp} v_{sp} + t_{sp} x_{sp} \right)$$

## 2. Feasibility Constraints (`AGENTS.md` §5)

1. **Integrality and non-negativity**: $x_{sp} \in \mathbb{Z}_{\ge 0}$
2. **Per-SKU warehouse supply**: $\sum_{s \in S} x_{sp} \le W_p \quad \forall p \in P$
3. **Per-store–SKU allocation cap**: $0 \le x_{sp} \le K_{sp} \quad \forall s \in S, p \in P$
4. **Total store stock capacity**: $\sum_{p \in P} (I_{sp} + x_{sp}) \le C_s \quad \forall s \in S$

## 3. Marginal-Unit Decomposition (`AGENTS.md` §6)

For a fixed SKU $p$, binary variables $y_{s,k} \in \{0, 1\}$ for $k \in \{1, \dots, K_s\}$ satisfy:
- $y_{s,k+1} \le y_{s,k}$ for $k = 1, \dots, K_s - 1$
- $x_s = \sum_{k=1}^{K_s} y_{s,k}$
- $\sum_{s \in S} \sum_{k=1}^{K_s} y_{s,k} \le W_p$
- Realized incremental cost of unit $k$:
  $$c_{\text{true}}[s, k; d_s] = \begin{cases} t_s - b_s & \text{if } d_s \ge I_s + k \\ t_s + h_s & \text{otherwise} \end{cases}$$
- Exact cost identity:
  $$C(x, d) = C(0, d) + \sum_{s \in S} \sum_{k=1}^{K_s} c_{\text{true}}[s, k; d_s] \, y_{s,k}$$

## 4. Reweighting DRO & CVaR Equivalence (`AGENTS.md` §7)

For scenario losses $L_j(x) = C(x, D^{(j)})$ and nominal probabilities $p_j > 0, \sum_{j=1}^N p_j = 1$, and $\alpha \in [0, 1)$:
- **Ambiguity set**: $\mathcal{P}_\alpha = \{ w \ge 0 : \sum_j w_j = 1, \; w_j \le p_j / (1 - \alpha) \}$
- **Greedy / exact sorting dual**: Sort scenarios so $L_{(1)} \ge L_{(2)} \ge \dots \ge L_{(N)}$ and assign maximum weight $w_{(j)} = \min\left(\frac{p_{(j)}}{1-\alpha}, \; 1 - \sum_{m < j} w_{(m)}\right)$.
- **LP / MILP representation**:
  $$\text{CVaR}_\alpha(L(x)) = \min_{\eta \in \mathbb{R}, \, z \ge 0} \left\{ \eta + \frac{1}{1-\alpha} \sum_{j=1}^N p_j z_j \; : \; z_j \ge L_j(x) - \eta \right\}$$
- **Risk trade-off objective**:
  $$\min_{x} \; (1 - \lambda_{\text{risk}}) \sum_{j=1}^N p_j L_j(x) + \lambda_{\text{risk}} \text{CVaR}_\alpha(L(x)), \quad \lambda_{\text{risk}} \in [0, 1]$$

## 5. Candidate JSON Schema for `--input`

```json
{
  "stores": ["S1", "S2"],
  "skus": ["P1"],
  "D": [[60], [20]],
  "I": [[0], [0]],
  "W": [60],
  "K": [[60], [60]],
  "C": [100, 100],
  "b": [[1.0], [1.0]],
  "h": [[0.0], [0.0]],
  "t": [[0.0], [0.0]],
  "x": [[40], [20]],
  "reported_cost": 20.0,
  "oracle_cost": 20.0,
  "solver_status": "optimal"
}
```
