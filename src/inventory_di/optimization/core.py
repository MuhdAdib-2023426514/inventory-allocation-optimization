"""Core mathematical problem definition, exact post-solve cost evaluator, marginal-unit
encoding identities, and nominal MILP solvers (`AGENTS.md` §5, §6).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import csc_matrix


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
    mip_gap: float = 0.0
    recomputed_objective: float | None = None


@dataclass(frozen=True)
class ServiceLevelSolveResult:
    """Output from service-level constrained allocation optimization (`AGENTS.md` §7, Phase 7)."""

    x: np.ndarray  # shape (S, P), int
    solver_objective: float
    solver_status: str
    is_feasible: bool
    violations: list[str]
    runtime_seconds: float
    target_fill_rate: float
    expected_fill_rate: float
    is_service_level_met: bool
    service_level_shortfall: float
    model_expected_cost_mu: float
    shortfall_units: float = 0.0
    maximum_achievable_fill_rate: float | None = None
    service_mode: str = "hard"

    @property
    def achieved_fill_rate(self) -> float:
        """Alias for expected_fill_rate used in dashboard UI."""
        return self.expected_fill_rate

    @property
    def expected_cost_mu(self) -> float:
        """Alias for model_expected_cost_mu used in dashboard UI."""
        return self.model_expected_cost_mu


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

    row_ind: list[int] = []
    col_ind: list[int] = []
    data_val: list[float] = []
    b_u_list: list[float] = []
    r_idx = 0

    # 1. Shortage linearization: -x_{sp} - u_{sp} <= -(D_{sp} - I_{sp})
    # 2. Excess linearization:    x_{sp} - v_{sp} <= D_{sp} - I_{sp}
    for s in range(S):
        for p in range(P):
            idx = s * P + p
            net_d = float(D_target[s, p] - inst.I[s, p])

            row_ind.extend([r_idx, r_idx])
            col_ind.extend([idx, N + idx])
            data_val.extend([-1.0, -1.0])
            b_u_list.append(-net_d)
            r_idx += 1

            row_ind.extend([r_idx, r_idx])
            col_ind.extend([idx, 2 * N + idx])
            data_val.extend([1.0, -1.0])
            b_u_list.append(net_d)
            r_idx += 1

    # 3. Warehouse supply per SKU: sum_s x_{sp} <= W_p
    for p in range(P):
        for s in range(S):
            row_ind.append(r_idx)
            col_ind.append(s * P + p)
            data_val.append(1.0)
        b_u_list.append(float(inst.W[p]))
        r_idx += 1

    # 4. Store total capacity: sum_p x_{sp} <= C_s - sum_p I_{sp}
    for s in range(S):
        for p in range(P):
            row_ind.append(r_idx)
            col_ind.append(s * P + p)
            data_val.append(1.0)
        rem_cap = float(inst.C[s] - np.sum(inst.I[s, :]))
        b_u_list.append(rem_cap)
        r_idx += 1

    A = csc_matrix((data_val, (row_ind, col_ind)), shape=(r_idx, num_vars), dtype=float)
    b_l = np.full(r_idx, -np.inf, dtype=float)
    b_u = np.array(b_u_list, dtype=float)

    res = milp(
        c=c,
        integrality=integrality,
        bounds=(lb, ub),
        constraints=LinearConstraint(A, b_l, b_u),
        options={"mip_rel_gap": 0.0},
    )
    runtime = time.perf_counter() - t0
    if res.status != 0 or res.x is None:
        raise RuntimeError(f"Allocation solve rejected: {res.message}")

    x_sol = np.rint(res.x[:N]).astype(int).reshape(S, P)
    ok, viols = check_allocation_feasibility(x_sol, inst)
    exact = evaluate_exact_cost(x_sol, D_target, inst)
    if not ok or not np.isclose(res.fun, exact.total_cost_mu, atol=1e-5):
        raise AssertionError("Deterministic solution failed independent validation")
    return SolveResult(
        x=x_sol,
        solver_objective=float(res.fun),
        mip_gap=float(res.mip_gap),
        recomputed_objective=exact.total_cost_mu,
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

    row_ind: list[int] = []
    col_ind: list[int] = []
    data_val: list[float] = []
    b_u_list: list[float] = []
    r_idx = 0

    # 1. Prefix constraints: y_{s,p,k+1} - y_{s,p,k} <= 0
    for s in range(S):
        for p in range(P):
            cap = int(inst.K[s, p])
            for k in range(1, cap):
                row_ind.extend([r_idx, r_idx])
                col_ind.extend([var_map[(s, p, k + 1)], var_map[(s, p, k)]])
                data_val.extend([1.0, -1.0])
                b_u_list.append(0.0)
                r_idx += 1

    # 2. Warehouse supply per SKU: sum_{s,k} y_{s,p,k} <= W_p
    for p in range(P):
        for s in range(S):
            for k in range(1, int(inst.K[s, p]) + 1):
                row_ind.append(r_idx)
                col_ind.append(var_map[(s, p, k)])
                data_val.append(1.0)
        b_u_list.append(float(inst.W[p]))
        r_idx += 1

    # 3. Store total capacity: sum_{p,k} y_{s,p,k} <= C_s - sum_p I_{s,p}
    for s in range(S):
        for p in range(P):
            for k in range(1, int(inst.K[s, p]) + 1):
                row_ind.append(r_idx)
                col_ind.append(var_map[(s, p, k)])
                data_val.append(1.0)
        b_u_list.append(float(inst.C[s] - np.sum(inst.I[s, :])))
        r_idx += 1

    A = csc_matrix((data_val, (row_ind, col_ind)), shape=(r_idx, num_vars), dtype=float)
    b_l = np.full(r_idx, -np.inf, dtype=float)
    b_u = np.array(b_u_list, dtype=float)

    res = milp(
        c=c,
        integrality=integrality,
        bounds=(lb, ub),
        constraints=LinearConstraint(A, b_l, b_u),
        options={"mip_rel_gap": 0.0},
    )
    runtime = time.perf_counter() - t0
    if res.status != 0 or res.x is None:
        raise RuntimeError(f"Allocation solve rejected: {res.message}")

    x_sol = np.zeros((S, P), dtype=int)
    for s in range(S):
        for p in range(P):
            cap = int(inst.K[s, p])
            x_sol[s, p] = int(round(sum(res.x[var_map[(s, p, k)]] for k in range(1, cap + 1))))

    ok, viols = check_allocation_feasibility(x_sol, inst)
    recomputed = sum(float(np.dot(marginal_costs_by_sku[sku], allocation_to_marginal_bits(x_sol[:, p], inst.K[:, p]))) for p, sku in enumerate(inst.skus))
    if not ok or not np.isclose(res.fun, recomputed, atol=1e-5):
        raise AssertionError("Nominal solution failed independent validation")
    return SolveResult(
        x=x_sol,
        mip_gap=float(res.mip_gap),
        recomputed_objective=recomputed,
        solver_objective=float(res.fun),
        solver_status="optimal",
        is_feasible=ok,
        violations=viols,
        runtime_seconds=runtime,
    )


def solve_multiperiod_carryover_oracle(
    D_trajectory: np.ndarray,
    base_inst: AllocationInstance,
    lead_time: int = 0,
    initial_pipeline: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Exact hindsight benchmark over the simulator's dispatch/admission action space."""
    from inventory_di.optimization.multiperiod import solve_horizon
    return solve_horizon(D_trajectory, base_inst, lead_time, initial_pipeline)


def solve_service_level_constrained_milp(
    scenarios: np.ndarray,
    inst: AllocationInstance,
    target_fill_rate: float = 0.85,
    shortfall_penalty_mu: float = 500.0,
    mode: str = "hard",
) -> ServiceLevelSolveResult:
    """Solve the service-level constrained allocation MILP (`AGENTS.md` §7, Phase 7).

    Minimizes expected inventory cost subject to an expected fill rate constraint:
        Expected Fulfilled Units / Expected Total Demand >= target_fill_rate - slack

    When warehouse inventory or store capacities make the target service level physically
    impossible, `slack > 0` flags the infeasibility, quantifies the exact shortfall,
    and penalizes it via `shortfall_penalty_mu`, providing the optimal graceful relaxation.
    """
    t_start = time.perf_counter()
    scenarios = np.asarray(scenarios, dtype=float)
    if scenarios.ndim != 3 or len(scenarios) == 0 or not np.isfinite(scenarios).all() or (scenarios < 0).any():
        raise ValueError("Scenarios must be nonempty finite nonnegative (N,S,P)")
    if not 0 <= target_fill_rate <= 1 or mode not in {"hard", "soft"} or shortfall_penalty_mu < 0:
        raise ValueError("Invalid service target, mode or penalty")
    N, S, P = scenarios.shape
    if (S, P) != inst.I.shape:
        raise ValueError("Scenario dimensions do not match inventory")
    maximum_fill = None
    M = S * P
    probs = np.full(N, 1.0 / N)
    exp_demand = float(np.mean(np.sum(scenarios, axis=(1, 2))))

    idx_slack = M + 2 * N * M
    total_vars = idx_slack + 1

    c = np.zeros(total_vars, dtype=float)
    integrality = np.zeros(total_vars, dtype=int)
    lb = np.zeros(total_vars, dtype=float)
    ub = np.full(total_vars, np.inf, dtype=float)

    # Shipping cost on x
    for s in range(S):
        for p in range(P):
            sp = s * P + p
            c[sp] = float(inst.t[s, p])
            integrality[sp] = 1
            ub[sp] = float(inst.K[s, p])

    # Scenario shortage and holding costs
    for j in range(N):
        pj = float(probs[j])
        for s in range(S):
            for p in range(P):
                sp = s * P + p
                idx_u = M + j * M + sp
                idx_v = M + N * M + j * M + sp
                c[idx_u] = pj * float(inst.b[s, p])
                c[idx_v] = pj * float(inst.h[s, p])

    c[idx_slack] = float(shortfall_penalty_mu) * exp_demand if mode == "soft" else 0.0
    ub[idx_slack] = 1.0 if mode == "soft" else 0.0

    row_ind: list[int] = []
    col_ind: list[int] = []
    data_val: list[float] = []
    b_l_list: list[float] = []
    b_u_list: list[float] = []
    r_idx = 0

    # 1. Warehouse supply per SKU: sum_s x_{s, p} <= W_p
    for p in range(P):
        for s in range(S):
            row_ind.append(r_idx)
            col_ind.append(s * P + p)
            data_val.append(1.0)
        b_l_list.append(-np.inf)
        b_u_list.append(float(inst.W[p]))
        r_idx += 1

    # 2. Total store stock capacity: sum_p x_{s, p} <= C_s - sum_p I_{s, p}
    for s in range(S):
        for p in range(P):
            row_ind.append(r_idx)
            col_ind.append(s * P + p)
            data_val.append(1.0)
        b_l_list.append(-np.inf)
        b_u_list.append(float(inst.C[s] - np.sum(inst.I[s, :])))
        r_idx += 1

    # 3. Scenario balance: v_{j, s, p} - u_{j, s, p} - x_{s, p} == I_{s, p} - D_{j, s, p}
    for j in range(N):
        for s in range(S):
            for p in range(P):
                sp = s * P + p
                idx_u = M + j * M + sp
                idx_v = M + N * M + j * M + sp
                row_ind.extend([r_idx, r_idx, r_idx])
                col_ind.extend([idx_v, idx_u, sp])
                data_val.extend([1.0, -1.0, -1.0])
                rhs = float(inst.I[s, p] - scenarios[j, s, p])
                b_l_list.append(rhs)
                b_u_list.append(rhs)
                r_idx += 1

    # 4. Target service-level constraint:
    #    sum_{j, s, p} p_j * u_{j, s, p} - slack * exp_demand <= (1 - target_fill_rate) * exp_demand
    for j in range(N):
        pj = float(probs[j])
        for s in range(S):
            for p in range(P):
                sp = s * P + p
                idx_u = M + j * M + sp
                row_ind.append(r_idx)
                col_ind.append(idx_u)
                data_val.append(pj)

    row_ind.append(r_idx)
    col_ind.append(idx_slack)
    data_val.append(-exp_demand)

    b_l_list.append(-np.inf)
    b_u_list.append(float((1.0 - target_fill_rate) * exp_demand))
    r_idx += 1

    A = csc_matrix((data_val, (row_ind, col_ind)), shape=(r_idx, total_vars), dtype=float)
    res = milp(
        c=c,
        integrality=integrality,
        bounds=(lb, ub),
        constraints=LinearConstraint(A, np.array(b_l_list, dtype=float), np.array(b_u_list, dtype=float)),
    )
    if mode == "hard" and res.status == 2:
        # Independently establish maximum expected service, then minimize cost
        # at that service optimum. No finite penalty can weaken this promise.
        ub[idx_slack] = 1.0
        fill_obj = np.zeros(total_vars)
        for j in range(N):
            fill_obj[M + j*M:M + (j+1)*M] = probs[j]
        max_service = milp(c=fill_obj, integrality=integrality, bounds=(lb, ub),
                           constraints=LinearConstraint(A, b_l_list, b_u_list),
                           options={"mip_rel_gap": 0.0})
        if max_service.status != 0:
            raise RuntimeError(f"Maximum service solve failed: {max_service.message}")
        min_unmet = float(max_service.fun)
        maximum_fill = 1.0 - min_unmet / exp_demand if exp_demand > 0 else 1.0
        from scipy.sparse import vstack
        A = vstack([A, csc_matrix(fill_obj.reshape(1, -1))], format="csc")
        b_l_list.append(-np.inf)
        b_u_list.append(min_unmet + 1e-7)
        res = milp(c=c, integrality=integrality, bounds=(lb, ub),
                   constraints=LinearConstraint(A, b_l_list, b_u_list),
                   options={"mip_rel_gap": 0.0})
    rt = time.perf_counter() - t_start

    if res.success and res.x is not None:
        x_sol = np.rint(res.x[:M]).astype(int).reshape(S, P)
        slack_val = float(res.x[idx_slack])
        exp_unmet = float(np.sum([
            float(probs[j]) * np.sum(np.maximum(0, scenarios[j] - inst.I - x_sol))
            for j in range(N)
        ]))
        achieved_fill = float(max(0.0, 1.0 - exp_unmet / exp_demand)) if exp_demand > 0 else 1.0
        is_met = achieved_fill >= target_fill_rate - 1e-6

        model_exp_cost = float(np.sum(inst.t * x_sol) + np.mean([
            np.sum(inst.b * np.maximum(0, scenarios[j] - inst.I - x_sol) +
                   inst.h * np.maximum(0, inst.I + x_sol - scenarios[j]))
            for j in range(N)
        ]))

        ok, viols = check_allocation_feasibility(x_sol, inst)
        status_msg = "optimal" if is_met else f"optimal_with_shortfall_{slack_val:.2%}"
        shortfall_frac = max(0.0, target_fill_rate - achieved_fill)
        return ServiceLevelSolveResult(
            x=x_sol,
            solver_objective=float(res.fun),
            solver_status=status_msg,
            is_feasible=ok,
            violations=viols,
            runtime_seconds=rt,
            target_fill_rate=float(target_fill_rate),
            expected_fill_rate=achieved_fill,
            is_service_level_met=is_met,
            service_level_shortfall=shortfall_frac,
            model_expected_cost_mu=model_exp_cost,
            shortfall_units=float(shortfall_frac * exp_demand),
            maximum_achievable_fill_rate=maximum_fill,
            service_mode=mode,
        )
    else:
        x_fallback = np.zeros((S, P), dtype=int)
        return ServiceLevelSolveResult(
            x=x_fallback,
            solver_objective=1e9,
            solver_status=f"infeasible ({res.message if hasattr(res, 'message') else 'failed'})",
            is_feasible=False,
            violations=["Solver failed to find integer feasible solution"],
            runtime_seconds=rt,
            target_fill_rate=float(target_fill_rate),
            expected_fill_rate=0.0,
            is_service_level_met=False,
            service_level_shortfall=float(target_fill_rate),
            model_expected_cost_mu=1e9,
            shortfall_units=float(target_fill_rate * exp_demand),
        )
