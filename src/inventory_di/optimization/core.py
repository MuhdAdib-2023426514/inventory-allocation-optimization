"""Core mathematical problem definition, exact post-solve cost evaluator, marginal-unit
encoding identities, and nominal MILP solvers (`AGENTS.md` §5, §6).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, milp


@dataclass(frozen=True)
class AllocationInstance:
    """Single-origin multi-store, multi-SKU allocation problem parameters."""

    stores: list[str]
    skus: list[str]
    I: np.ndarray  # shape (S, P), int >= 0, opening stock
    W: np.ndarray  # shape (P,), int >= 0, warehouse supply per SKU
    K: np.ndarray  # shape (S, P), int >= 0, per store-SKU allocation cap
    C: np.ndarray  # shape (S,), int >= 0, total store capacity (sum_p (I_sp + x_sp) <= C_s)
    b: np.ndarray  # shape (S, P), float >= 0, shortage penalty MU/unit
    h: np.ndarray  # shape (S, P), float >= 0, holding cost MU/unit
    t: np.ndarray  # shape (S, P), float >= 0, shipping cost MU/unit


@dataclass(frozen=True)
class CostBreakdown:
    """Exact post-solve cost breakdown and fulfillment statistics."""

    total_cost_mu: float
    shortage_cost_mu: float
    holding_cost_mu: float
    shipping_cost_mu: float
    filled_units: int
    unmet_units: int
    fill_rate: float


@dataclass(frozen=True)
class SolveResult:
    """Solver output with independent feasibility verification."""

    x: np.ndarray  # shape (S, P), int
    solver_objective: float
    solver_status: str
    is_feasible: bool
    violations: list[str]
    runtime_seconds: float


def check_allocation_feasibility(
    x: np.ndarray,
    inst: AllocationInstance,
    tol: float = 1e-6,
) -> tuple[bool, list[str]]:
    """Independently check all integer and capacity constraints (`AGENTS.md` §5)."""
    violations: list[str] = []
    S, P = len(inst.stores), len(inst.skus)
    if x.shape != (S, P):
        return False, [f"Invalid shape {x.shape}, expected {(S, P)}"]

    for s in range(S):
        for p in range(P):
            val = float(x[s, p])
            if abs(val - round(val)) > tol:
                violations.append(f"Non-integer x[{s},{p}]={val}")
            if val < -tol:
                violations.append(f"Negative x[{s},{p}]={val}")
            if val > float(inst.K[s, p]) + tol:
                violations.append(f"Cap violation x[{s},{p}]={val} > K[{s},{p}]={inst.K[s, p]}")

    for p in range(P):
        shipped = float(np.sum(x[:, p]))
        if shipped > float(inst.W[p]) + tol:
            violations.append(f"Warehouse supply violation SKU {p}: {shipped} > W={inst.W[p]}")

    for s in range(S):
        total_stock = float(np.sum(inst.I[s, :] + x[s, :]))
        if total_stock > float(inst.C[s]) + tol:
            violations.append(
                f"Store total capacity violation store {s}: {total_stock} > C={inst.C[s]}"
            )

    return len(violations) == 0, violations


def evaluate_exact_cost(
    x: np.ndarray,
    D: np.ndarray,
    inst: AllocationInstance,
) -> CostBreakdown:
    """Independently compute exact one-week allocation cost from integer allocations (`AGENTS.md` §5):
    C(x, D) = sum_{s,p} [ b_sp * max(D_sp - I_sp - x_sp, 0)
                        + h_sp * max(I_sp + x_sp - D_sp, 0)
                        + t_sp * x_sp ].
    """
    x_int = np.rint(x).astype(int)
    D_int = np.rint(D).astype(int)
    I_int = np.rint(inst.I).astype(int)

    avail = I_int + x_int
    shortage = np.maximum(D_int - avail, 0)
    excess = np.maximum(avail - D_int, 0)
    filled = np.minimum(D_int, avail)

    shortage_cost = float(np.sum(inst.b * shortage))
    holding_cost = float(np.sum(inst.h * excess))
    shipping_cost = float(np.sum(inst.t * x_int))
    total_cost = shortage_cost + holding_cost + shipping_cost

    filled_units = int(np.sum(filled))
    unmet_units = int(np.sum(shortage))
    total_demand = int(np.sum(D_int))
    fill_rate = float(filled_units / total_demand) if total_demand > 0 else 1.0

    return CostBreakdown(
        total_cost_mu=total_cost,
        shortage_cost_mu=shortage_cost,
        holding_cost_mu=holding_cost,
        shipping_cost_mu=shipping_cost,
        filled_units=filled_units,
        unmet_units=unmet_units,
        fill_rate=fill_rate,
    )


def compute_true_marginal_unit_costs(
    d_s: np.ndarray,
    I_s: np.ndarray,
    K_s: np.ndarray,
    b_s: np.ndarray,
    h_s: np.ndarray,
    t_s: np.ndarray,
) -> np.ndarray:
    """Compute true incremental realized cost vector c_true[s, k; d] for a single SKU (`AGENTS.md` §6):
    c_true[s, k; d] = t_s - b_s if d_s >= I_s + k else t_s + h_s, for k = 1, ..., K_s.
    Flattened in deterministic (store s, unit k=1..K_s) order.
    """
    costs: list[float] = []
    for s in range(len(K_s)):
        cap = int(K_s[s])
        for k in range(1, cap + 1):
            if int(d_s[s]) >= int(I_s[s]) + k:
                costs.append(float(t_s[s] - b_s[s]))
            else:
                costs.append(float(t_s[s] + h_s[s]))
    return np.array(costs, dtype=np.float64)


def allocation_to_marginal_bits(x_s: np.ndarray, K_s: np.ndarray) -> np.ndarray:
    """Encode integer allocations x_s into prefix binary vector y[s, k] in {0, 1}."""
    bits: list[float] = []
    for s, cap in enumerate(K_s):
        alloc = int(round(x_s[s]))
        for k in range(1, int(cap) + 1):
            bits.append(1.0 if k <= alloc else 0.0)
    return np.array(bits, dtype=np.float64)


def marginal_bits_to_allocation(y_flat: np.ndarray, K_s: np.ndarray) -> np.ndarray:
    """Decode flattened prefix binary vector y[s, k] to integer store allocations x_s."""
    x_s = np.zeros(len(K_s), dtype=int)
    idx = 0
    for s, cap in enumerate(K_s):
        c_int = int(cap)
        x_s[s] = int(round(float(np.sum(y_flat[idx : idx + c_int]))))
        idx += c_int
    return x_s


def verify_marginal_cost_identity(
    x_s: np.ndarray,
    d_s: np.ndarray,
    I_s: np.ndarray,
    K_s: np.ndarray,
    b_s: np.ndarray,
    h_s: np.ndarray,
    t_s: np.ndarray,
) -> dict[str, Any]:
    """Prove C(x, d) = C(0, d) + sum_{s,k} c_true[s,k;d] * y[s,k] for a single-SKU instance."""
    S = len(K_s)
    dummy_inst = AllocationInstance(
        stores=[f"S{s}" for s in range(S)],
        skus=["P0"],
        I=I_s.reshape(S, 1),
        W=np.array([int(np.sum(K_s))]),
        K=K_s.reshape(S, 1),
        C=(I_s + K_s).astype(int),
        b=b_s.reshape(S, 1),
        h=h_s.reshape(S, 1),
        t=t_s.reshape(S, 1),
    )
    c_zero = evaluate_exact_cost(np.zeros((S, 1), dtype=int), d_s.reshape(S, 1), dummy_inst).total_cost_mu
    c_exact = evaluate_exact_cost(x_s.reshape(S, 1), d_s.reshape(S, 1), dummy_inst).total_cost_mu

    c_true = compute_true_marginal_unit_costs(d_s, I_s, K_s, b_s, h_s, t_s)
    y_bits = allocation_to_marginal_bits(x_s, K_s)
    c_decomposed = float(c_zero + np.dot(c_true, y_bits))
    diff = abs(c_exact - c_decomposed)
    return {
        "c_zero": c_zero,
        "c_exact": c_exact,
        "c_decomposed": c_decomposed,
        "abs_diff": diff,
        "matches": diff <= 1e-9,
    }


def solve_deterministic_milp(
    D_target: np.ndarray,
    inst: AllocationInstance,
) -> SolveResult:
    """Solve nominal deterministic/oracle MILP with linearized shortage and excess (`AGENTS.md` §5).

    Variables per (s, p):
      - x_{s,p} in Z_{>=0}, 0 <= x_{s,p} <= K_{s,p}
      - u_{s,p} >= 0, u_{s,p} >= D_{s,p} - I_{s,p} - x_{s,p} (shortage)
      - v_{s,p} >= 0, v_{s,p} >= I_{s,p} + x_{s,p} - D_{s,p} (excess)

    When h_{s,p} == 0 and t_{s,p} == 0, adds a tiny secondary tie-breaking regularization
    on excess v_{s,p} (1e-6) so the solver never ships redundant units above demand D_{s,p}
    to one store while another store has unmet demand, and computed post-solve exact cost
    is returned without the tie-breaker.
    """
    t0 = time.perf_counter()
    S, P = len(inst.stores), len(inst.skus)
    N = S * P
    # Variable layout: [x (0..N-1), u (N..2N-1), v (2N..3N-1)]
    num_vars = 3 * N
    c = np.zeros(num_vars, dtype=float)
    integrality = np.zeros(num_vars, dtype=int)
    lb = np.zeros(num_vars, dtype=float)
    ub = np.full(num_vars, np.inf, dtype=float)

    for s in range(S):
        for p in range(P):
            idx = s * P + p
            c[idx] = float(inst.t[s, p])
            c[N + idx] = float(inst.b[s, p])
            # Tiny 1e-7 tie-breaker on excess v when h==0 and t==0 to avoid wasting units above D_sp
            c[2 * N + idx] = float(inst.h[s, p]) if inst.h[s, p] > 0 else 1e-7
            integrality[idx] = 1
            ub[idx] = float(max(0, min(int(inst.K[s, p]), int(inst.C[s]) - int(inst.I[s, p]))))

    rows: list[np.ndarray] = []
    b_u_list: list[float] = []

    # 1. Shortage linearization: -x_{sp} - u_{sp} <= -(D_{sp} - I_{sp})
    # 2. Excess linearization:    x_{sp} - v_{sp} <= D_{sp} - I_{sp}
    for s in range(S):
        for p in range(P):
            idx = s * P + p
            net_d = float(D_target[s, p] - inst.I[s, p])

            row_u = np.zeros(num_vars, dtype=float)
            row_u[idx] = -1.0
            row_u[N + idx] = -1.0
            rows.append(row_u)
            b_u_list.append(-net_d)

            row_v = np.zeros(num_vars, dtype=float)
            row_v[idx] = 1.0
            row_v[2 * N + idx] = -1.0
            rows.append(row_v)
            b_u_list.append(net_d)

    # 3. Warehouse supply per SKU: sum_s x_{sp} <= W_p
    for p in range(P):
        row_w = np.zeros(num_vars, dtype=float)
        for s in range(S):
            row_w[s * P + p] = 1.0
        rows.append(row_w)
        b_u_list.append(float(inst.W[p]))

    # 4. Store total capacity: sum_p x_{sp} <= C_s - sum_p I_{sp}
    for s in range(S):
        row_c = np.zeros(num_vars, dtype=float)
        for p in range(P):
            row_c[s * P + p] = 1.0
        rows.append(row_c)
        rem_cap = float(inst.C[s] - np.sum(inst.I[s, :]))
        b_u_list.append(rem_cap)

    A = np.vstack(rows)
    b_l = np.full(len(b_u_list), -np.inf, dtype=float)
    b_u = np.array(b_u_list, dtype=float)

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
        return SolveResult(
            x=x_zero,
            solver_objective=float("nan"),
            solver_status=f"failed ({res.message})",
            is_feasible=ok,
            violations=viols,
            runtime_seconds=runtime,
        )

    x_sol = np.rint(res.x[:N]).astype(int).reshape(S, P)
    ok, viols = check_allocation_feasibility(x_sol, inst)
    exact = evaluate_exact_cost(x_sol, D_target, inst)
    return SolveResult(
        x=x_sol,
        solver_objective=exact.total_cost_mu,
        solver_status="optimal",
        is_feasible=ok,
        violations=viols,
        runtime_seconds=runtime,
    )


def solve_nominal_marginal_cost_milp(
    marginal_costs_by_sku: dict[str, np.ndarray],
    inst: AllocationInstance,
) -> SolveResult:
    """Solve multi-SKU nominal expected-cost MILP using predicted marginal unit costs c_hat[s, k]
    subject to prefix constraints y_{s,p,k+1} <= y_{s,p,k}, warehouse supply sum_{s,k} y_{s,p,k} <= W_p,
    and joint store capacity sum_{p,k} y_{s,p,k} <= C_s - sum_p I_{s,p}.
    """
    t0 = time.perf_counter()
    S, P = len(inst.stores), len(inst.skus)

    # Map (s, p, k) for k=1..K_{s,p} to flat variable index
    var_map: dict[tuple[int, int, int], int] = {}
    c_list: list[float] = []
    for p, sku in enumerate(inst.skus):
        c_sku = marginal_costs_by_sku[sku]
        flat_idx = 0
        for s in range(S):
            cap = int(inst.K[s, p])
            for k in range(1, cap + 1):
                var_map[(s, p, k)] = len(c_list)
                c_list.append(float(c_sku[flat_idx]))
                flat_idx += 1

    num_vars = len(c_list)
    if num_vars == 0:
        x_zero = np.zeros((S, P), dtype=int)
        ok, viols = check_allocation_feasibility(x_zero, inst)
        return SolveResult(x_zero, 0.0, "optimal", ok, viols, time.perf_counter() - t0)

    c = np.array(c_list, dtype=float)
    integrality = np.ones(num_vars, dtype=int)
    lb = np.zeros(num_vars, dtype=float)
    ub = np.ones(num_vars, dtype=float)

    rows: list[np.ndarray] = []
    b_u_list: list[float] = []

    # 1. Prefix constraints: y_{s,p,k+1} - y_{s,p,k} <= 0
    for s in range(S):
        for p in range(P):
            cap = int(inst.K[s, p])
            for k in range(1, cap):
                row = np.zeros(num_vars, dtype=float)
                row[var_map[(s, p, k + 1)]] = 1.0
                row[var_map[(s, p, k)]] = -1.0
                rows.append(row)
                b_u_list.append(0.0)

    # 2. Warehouse supply per SKU: sum_{s,k} y_{s,p,k} <= W_p
    for p in range(P):
        row = np.zeros(num_vars, dtype=float)
        for s in range(S):
            for k in range(1, int(inst.K[s, p]) + 1):
                row[var_map[(s, p, k)]] = 1.0
        rows.append(row)
        b_u_list.append(float(inst.W[p]))

    # 3. Store total capacity: sum_{p,k} y_{s,p,k} <= C_s - sum_p I_{s,p}
    for s in range(S):
        row = np.zeros(num_vars, dtype=float)
        for p in range(P):
            for k in range(1, int(inst.K[s, p]) + 1):
                row[var_map[(s, p, k)]] = 1.0
        rows.append(row)
        b_u_list.append(float(inst.C[s] - np.sum(inst.I[s, :])))

    A = np.vstack(rows)
    b_l = np.full(len(b_u_list), -np.inf, dtype=float)
    b_u = np.array(b_u_list, dtype=float)

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
        return SolveResult(
            x=x_zero,
            solver_objective=float("nan"),
            solver_status=f"failed ({res.message})",
            is_feasible=ok,
            violations=viols,
            runtime_seconds=runtime,
        )

    x_sol = np.zeros((S, P), dtype=int)
    for s in range(S):
        for p in range(P):
            cap = int(inst.K[s, p])
            x_sol[s, p] = int(round(sum(res.x[var_map[(s, p, k)]] for k in range(1, cap + 1))))

    ok, viols = check_allocation_feasibility(x_sol, inst)
    return SolveResult(
        x=x_sol,
        solver_objective=float(res.fun),
        solver_status="optimal",
        is_feasible=ok,
        violations=viols,
        runtime_seconds=runtime,
    )
