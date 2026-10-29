"""Distributionally Robust Optimization (DRO) via probability-reweighting ambiguity set `P_alpha`
(mathematically equivalent to Rockafellar–Uryasev CVaR_alpha) (`AGENTS.md` §7).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import LinearConstraint, linprog, milp
from scipy.sparse import csc_matrix

from inventory_di.optimization.core import (
    AllocationInstance,
     check_allocation_feasibility,
    evaluate_exact_cost,
)


@dataclass(frozen=True)
class DROSolveResult:
    """Output of the scenario-reweighting DRO MILP solver."""

    x: np.ndarray  # shape (S, P), int
    solver_objective: float
    model_expected_cost_mu: float
    model_cvar_cost_mu: float
    eta_var_mu: float
    alpha: float
    risk_weight: float
    solver_status: str
    is_feasible: bool
    violations: list[str]
    runtime_seconds: float
    scenario_losses_mu: np.ndarray  # shape (J,)


def exact_ambiguity_set_worst_case_loss(
    losses: np.ndarray,
    probs: np.ndarray,
    alpha: float,
) -> tuple[float, np.ndarray]:
    """Compute exact worst-case expected loss over the probability-reweighting ambiguity set:
        P_alpha = { w in R^J_{>=0} : sum_j w_j = 1, 0 <= w_j <= p_j / (1 - alpha) }
    by sorting scenario losses in descending order and greedily assigning maximum weight `p_j / (1 - alpha)`
    until total weight reaches 1.0 (`AGENTS.md` §7, §14).
    """
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    L = np.asarray(losses, dtype=float)
    p = np.asarray(probs, dtype=float)
    p = p / float(np.sum(p))

    if alpha == 0.0:
        return float(np.dot(p, L)), p.copy()

    order = np.argsort(-L, kind="mergesort")
    w = np.zeros_like(p)
    rem = 1.0
    cap_mult = 1.0 / (1.0 - float(alpha))

    for idx in order:
        if rem <= 1e-12:
            break
        max_w = min(rem, float(p[idx]) * cap_mult)
        w[idx] = max_w
        rem -= max_w

    return float(np.dot(w, L)), w


def rockafellar_uryasev_cvar_lp(
    losses: np.ndarray,
    probs: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Solve the Rockafellar–Uryasev LP for fixed scenario losses `L_j`:
        min_{eta, z >= 0} eta + (1 / (1 - alpha)) * sum_j p_j z_j
        s.t. z_j >= L_j - eta  for all j = 1..J.
    Returns `(cvar_value, optimal_eta)`.
    """
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    L = np.asarray(losses, dtype=float)
    p = np.asarray(probs, dtype=float)
    p = p / float(np.sum(p))
    J = len(L)

    # Variables: [eta, z_0, ..., z_{J-1}]
    c = np.zeros(1 + J, dtype=float)
    c[0] = 1.0
    c[1:] = p / (1.0 - float(alpha))

    bounds = [(None, None)] + [(0.0, None) for _ in range(J)]
    # Constraints: -eta - z_j <= -L_j
    A_ub = np.zeros((J, 1 + J), dtype=float)
    A_ub[:, 0] = -1.0
    for j in range(J):
        A_ub[j, 1 + j] = -1.0
    b_ub = -L

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Rockafellar-Uryasev LP failed: {res.message}")
    return float(res.fun), float(res.x[0])


def evaluate_allocation_on_scenarios(
    x: np.ndarray,
    scenarios: np.ndarray,
    inst: AllocationInstance,
    alpha: float = 0.90,
    probs: np.ndarray | None = None,
) -> dict[str, float | np.ndarray]:
    """Compute scenario loss vector `L_j(x)`, expected cost, and CVaR_alpha for an allocation `x`."""
    J = scenarios.shape[0]
    if probs is None:
        p = np.full(J, 1.0 / float(J), dtype=float)
    else:
        p = np.asarray(probs, dtype=float)
        p = p / float(np.sum(p))

    losses = np.zeros(J, dtype=float)
    for j in range(J):
        losses[j] = evaluate_exact_cost(x, scenarios[j], inst).total_cost_mu

    exp_cost = float(np.dot(p, losses))
    cvar_cost, _ = exact_ambiguity_set_worst_case_loss(losses, p, alpha=alpha)
    return {
        "expected_cost_mu": exp_cost,
        "cvar_cost_mu": cvar_cost,
        "scenario_losses_mu": losses,
    }


def solve_scenario_reweighting_dro_milp(
    scenarios: np.ndarray,
    inst: AllocationInstance,
    alpha: float = 0.90,
    risk_weight: float = 0.50,
    probs: np.ndarray | None = None,
) -> DROSolveResult:
    """Solve the explicit scenario-reweighting DRO MILP (`AGENTS.md` §7):
        min_x (1 - risk_weight) * sum_j p_j L_j(x) + risk_weight * CVaR_alpha(L(x))
    where `CVaR_alpha(L(x)) = min_{eta, z >= 0} { eta + sum_j p_j z_j / (1 - alpha) : z_j >= L_j(x) - eta }`
    and `L_j(x) = sum_{s,p} (b_{sp} u_{jsp} + h_{sp} v_{jsp} + t_{sp} x_{sp})`.

    Note:
    - When `risk_weight == 0.0` (or `alpha == 0.0`), the objective reduces to risk-neutral
      expected scenario loss.
    - When `risk_weight > 0.0` and `alpha > 0.0`, optimizes the convex combination of
      expected loss and worst-case expected loss over the probability-reweighting ambiguity set `P_alpha`.
    """
    t0 = time.perf_counter()
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    if not (0.0 <= risk_weight <= 1.0):
        raise ValueError(f"risk_weight must be in [0, 1], got {risk_weight}")

    J, S, P = scenarios.shape
    if probs is None:
        p = np.full(J, 1.0 / float(J), dtype=float)
    else:
        p = np.asarray(probs, dtype=float)
        p = p / float(np.sum(p))

    N = S * P
    # Variable layout:
    # 0 .. N-1             : x_{s,p} (integer >= 0)
    # N                   : eta (real, unrestricted)
    # N+1 .. N+J          : z_j (real >= 0)
    # N+1+J .. N+1+J+J*N-1: u_{j,s,p} (shortage >= 0)
    # N+1+J+J*N .. end    : v_{j,s,p} (excess >= 0)
    idx_eta = N
    idx_z_start = N + 1
    idx_u_start = N + 1 + J
    idx_v_start = N + 1 + J + J * N
    total_vars = N + 1 + J + 2 * J * N

    c = np.zeros(total_vars, dtype=float)
    integrality = np.zeros(total_vars, dtype=int)
    lb = np.zeros(total_vars, dtype=float)
    ub = np.full(total_vars, np.inf, dtype=float)

    # eta is unrestricted in sign
    lb[idx_eta] = -1e6

    rw = float(risk_weight)
    one_minus_rw = 1.0 - rw

    # Shipping cost term appears directly on x_{sp} with weight 1.0 in expected cost
    # AND inside L_j for z_j >= L_j - eta
    for s in range(S):
        for p_idx in range(P):
            sp = s * P + p_idx
            c[sp] = one_minus_rw * float(inst.t[s, p_idx])
            integrality[sp] = 1
            ub[sp] = float(max(0, min(int(inst.K[s, p_idx]), int(inst.C[s]) - int(inst.I[s, p_idx]))))

    if rw > 0.0:
        c[idx_eta] = rw
        for j in range(J):
            c[idx_z_start + j] = rw * float(p[j]) / (1.0 - float(alpha))

    for j in range(J):
        pj = float(p[j])
        for s in range(S):
            for p_idx in range(P):
                sp = s * P + p_idx
                u_idx = idx_u_start + j * N + sp
                v_idx = idx_v_start + j * N + sp
                c[u_idx] = one_minus_rw * pj * float(inst.b[s, p_idx])
                h_val = float(inst.h[s, p_idx]) if inst.h[s, p_idx] > 0 else 1e-7
                c[v_idx] = one_minus_rw * pj * h_val

    # Build sparse CSC constraint matrix (Priority 6: sparse MILP matrices)
    num_rows = P + S + 2 * J * N + (J if rw > 0.0 else 0)
    row_ind: list[int] = []
    col_ind: list[int] = []
    data_val: list[float] = []
    b_l = np.full(num_rows, -np.inf, dtype=float)
    b_u = np.zeros(num_rows, dtype=float)
    r_idx = 0

    # 1. Warehouse supply per SKU: sum_s x_{sp} <= W_p
    for p_idx in range(P):
        for s in range(S):
            row_ind.append(r_idx)
            col_ind.append(s * P + p_idx)
            data_val.append(1.0)
        b_u[r_idx] = float(inst.W[p_idx])
        r_idx += 1

    # 2. Store total capacity: sum_p x_{sp} <= C_s - sum_p I_{sp}
    for s in range(S):
        for p_idx in range(P):
            row_ind.append(r_idx)
            col_ind.append(s * P + p_idx)
            data_val.append(1.0)
        b_u[r_idx] = float(inst.C[s] - np.sum(inst.I[s, :]))
        r_idx += 1

    # 3. Scenario shortage and excess constraints
    for j in range(J):
        for s in range(S):
            for p_idx in range(P):
                sp = s * P + p_idx
                u_idx = idx_u_start + j * N + sp
                v_idx = idx_v_start + j * N + sp
                net_d = float(scenarios[j, s, p_idx] - inst.I[s, p_idx])

                # -x_{sp} - u_{jsp} <= -net_d
                row_ind.extend([r_idx, r_idx])
                col_ind.extend([sp, u_idx])
                data_val.extend([-1.0, -1.0])
                b_u[r_idx] = -net_d
                r_idx += 1

                # x_{sp} - v_{jsp} <= net_d
                row_ind.extend([r_idx, r_idx])
                col_ind.extend([sp, v_idx])
                data_val.extend([1.0, -1.0])
                b_u[r_idx] = net_d
                r_idx += 1

    # 4. CVaR excess constraint: L_j(x) - eta - z_j <= 0
    if rw > 0.0:
        for j in range(J):
            row_ind.extend([r_idx, r_idx])
            col_ind.extend([idx_eta, idx_z_start + j])
            data_val.extend([-1.0, -1.0])
            for s in range(S):
                for p_idx in range(P):
                    sp = s * P + p_idx
                    u_idx = idx_u_start + j * N + sp
                    v_idx = idx_v_start + j * N + sp
                    row_ind.extend([r_idx, r_idx, r_idx])
                    col_ind.extend([sp, u_idx, v_idx])
                    data_val.extend(
                        [
                            float(inst.t[s, p_idx]),
                            float(inst.b[s, p_idx]),
                            float(inst.h[s, p_idx]),
                        ]
                    )
            b_u[r_idx] = 0.0
            r_idx += 1

    A = csc_matrix((data_val, (row_ind, col_ind)), shape=(num_rows, total_vars), dtype=float)

    res = milp(
        c=c,
        integrality=integrality,
        bounds=(lb, ub),
        constraints=LinearConstraint(A, b_l, b_u),
    )
    runtime = time.perf_counter() - t0
    if not res.success or res.x is None:
        x_zero = np.zeros((S, P), dtype=int)
        ok, viols = check_allocation_feasibility(x_zero, inst)
        eval_zero = evaluate_allocation_on_scenarios(x_zero, scenarios, inst, alpha=alpha, probs=p)
        return DROSolveResult(
            x=x_zero,
            solver_objective=float("nan"),
            model_expected_cost_mu=float(eval_zero["expected_cost_mu"]),
            model_cvar_cost_mu=float(eval_zero["cvar_cost_mu"]),
            eta_var_mu=0.0,
            alpha=float(alpha),
            risk_weight=float(risk_weight),
            solver_status=f"failed ({res.message})",
            is_feasible=ok,
            violations=viols,
            runtime_seconds=runtime,
            scenario_losses_mu=np.asarray(eval_zero["scenario_losses_mu"]),
        )

    x_sol = np.rint(res.x[:N]).astype(int).reshape(S, P)
    eta_val = float(res.x[idx_eta]) if rw > 0.0 else 0.0
    ok, viols = check_allocation_feasibility(x_sol, inst)
    scen_eval = evaluate_allocation_on_scenarios(x_sol, scenarios, inst, alpha=alpha, probs=p)
    exp_cost = float(scen_eval["expected_cost_mu"])
    cvar_cost = float(scen_eval["cvar_cost_mu"])
    recomputed_obj = (1.0 - rw) * exp_cost + rw * cvar_cost

    return DROSolveResult(
        x=x_sol,
        solver_objective=recomputed_obj,
        model_expected_cost_mu=exp_cost,
        model_cvar_cost_mu=cvar_cost,
        eta_var_mu=eta_val,
        alpha=float(alpha),
        risk_weight=float(risk_weight),
        solver_status="optimal",
        is_feasible=ok,
        violations=viols,
        runtime_seconds=runtime,
        scenario_losses_mu=np.asarray(scen_eval["scenario_losses_mu"]),
    )
