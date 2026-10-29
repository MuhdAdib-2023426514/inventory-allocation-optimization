# Decision-Focused Learning Checklist & Reference (`decision-focused-learning`)

Aligned with [`AGENTS.md`](../../../AGENTS.md) §6, §7, and §14.

## 1. Why Fixed-Feasible-Set Encoding Is Required

Standard SPO+ (`pyepo.func.SPOPlus`) solves:
$$z^*(c) = \min_{y \in \mathcal{Y}} c^\top y$$
where the feasible polytope/lattice $\mathcal{Y}$ is **constant across training instances** and only the objective vector $c$ varies.

In our per-SKU training problem with stores $s \in S$ and incremental unit cap $K_s$:
- Decision variables: $y_{s,k} \in \{0, 1\}$ for $s \in S, \; k \in \{1, \dots, K_s\}$.
- Fixed prefix constraints: $y_{s,k+1} \le y_{s,k}$ for $k = 1, \dots, K_s - 1$.
- Fixed warehouse budget: $\sum_{s \in S} \sum_{k=1}^{K_s} y_{s,k} \le W_p$.

Because $\hat{c}[s, k]$ is non-decreasing in $k$, any optimal LP/MILP solution naturally respects the prefix ordering $y_{s,1} \ge y_{s,2} \ge \dots \ge y_{s,K_s}$, and $x_s = \sum_{k=1}^{K_s} y_{s,k}$.

## 2. Exact SPO+ Loss and Subgradient Formula

Given predicted cost vector $\hat{c}$, true realized incremental cost vector $c_{\text{true}}$, true optimal decision $y^*(c_{\text{true}}) \in \arg\min_{y \in \mathcal{Y}} c_{\text{true}}^\top y$, and true optimal value $z^*(c_{\text{true}}) = c_{\text{true}}^\top y^*(c_{\text{true}})$:

Define the SPO+ perturbed cost vector:
$$c_{\text{spo}} = 2 \hat{c} - c_{\text{true}}$$
and let:
$$y^*(2\hat{c} - c_{\text{true}}) \in \arg\min_{y \in \mathcal{Y}} (2\hat{c} - c_{\text{true}})^\top y$$

Then the convex SPO+ surrogate loss is:
$$\mathcal{L}_{\text{SPO}+}(\hat{c}, c_{\text{true}}) = \max_{y \in \mathcal{Y}} \left( c_{\text{true}}^\top y - 2 \hat{c}^\top y \right) + 2 \hat{c}^\top y^*(c_{\text{true}}) - z^*(c_{\text{true}})$$
$$= -(2\hat{c} - c_{\text{true}})^\top y^*(2\hat{c} - c_{\text{true}}) + 2 \hat{c}^\top y^*(c_{\text{true}}) - c_{\text{true}}^\top y^*(c_{\text{true}})$$

And its subgradient with respect to $\hat{c}$ is:
$$\nabla_{\hat{c}} \mathcal{L}_{\text{SPO}+}(\hat{c}, c_{\text{true}}) = 2 \left( y^*(c_{\text{true}}) - y^*(2\hat{c} - c_{\text{true}}) \right)$$

## 3. Input Audit JSON Schema (`--input`)

```json
{
  "H": 10,
  "K_s": [4, 4],
  "I_s": [0, 0],
  "W_p_instances": [5, 5],
  "b_s": [4.0, 3.0],
  "h_s": [0.5, 0.5],
  "t_s": [0.2, 0.2],
  "pmf": [
    [0.05, 0.10, 0.20, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02, 0.02, 0.01, 0.02],
    [0.10, 0.20, 0.25, 0.20, 0.10, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01]
  ],
  "d_true": [3, 2],
  "oracle_splits_used": ["train", "val"],
  "grad_norm_nll_only": 0.412,
  "grad_norm_dfl": 0.895,
  "param_diff_l2_vs_nll": 0.018
}
```
Note: `pmf[s]` has length $H + 2$, corresponding to bins $0, 1, \dots, H$ and index $H+1$ for `overflow > H`.
