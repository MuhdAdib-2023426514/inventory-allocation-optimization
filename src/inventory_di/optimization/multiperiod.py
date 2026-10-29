"""Exact dispatch/admission horizon oracle, independently replayed in the simulator."""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def solve_horizon(D_trajectory, base_inst, lead_time=0, initial_pipeline=None):
    from inventory_di.simulation.simulator import InventoryStateTransitionSystem

    D = np.asarray(D_trajectory)
    S, P = len(base_inst.stores), len(base_inst.skus)
    if D.ndim != 3 or D.shape[1:] != (S, P) or len(D) == 0:
        raise ValueError("Demand trajectory must be nonempty (T, stores, SKUs)")
    T, N = len(D), S * P
    sim = InventoryStateTransitionSystem(base_inst, lead_time, initial_pipeline)
    # Validate all demand before building a model; no rounding of fractional input.
    for d in D:
        sim.validate_units(d, "demand")
    L = sim.lead_time
    pipe = sim.pipeline
    # Blocks: dispatch x, admitted a, closing store v, closing deferred q,
    # fulfilled f and binary z selecting which bound of min(D, stock) binds.
    n = 6 * T * N
    c = np.zeros(n)
    lb, ub = np.zeros(n), np.full(n, np.inf)
    integrality = np.ones(n, dtype=int)

    def idx(block, week, sp):
        return block * T * N + week * N + sp

    for week in range(T):
        for sp in range(N):
            s, p = divmod(sp, P)
            c[idx(0, week, sp)] = base_inst.t[s, p]
            c[idx(2, week, sp)] = base_inst.h[s, p]
            c[idx(3, week, sp)] = base_inst.h[s, p]
            c[idx(4, week, sp)] = -base_inst.b[s, p]
            ub[idx(0, week, sp)] = base_inst.K[s, p] if week + L < T else 0
            ub[idx(1, week, sp)] = base_inst.C[s]
            ub[idx(2, week, sp)] = base_inst.C[s]
            ub[idx(4, week, sp)] = D[week, s, p]
            ub[idx(5, week, sp)] = 1
    rows, cols, values, lower, upper = [], [], [], [], []

    def row(terms, lo=-np.inf, hi=np.inf):
        r = len(lower)
        for col, value in terms.items():
            rows.append(r)
            cols.append(col)
            values.append(value)
        lower.append(lo)
        upper.append(hi)

    for week in range(T):
        for p in range(P):
            row({idx(0, week, s * P + p): 1 for s in range(S)}, hi=base_inst.W[p])
        for s in range(S):
            terms = {idx(1, week, s * P + p): 1 for p in range(P)}
            if week:
                terms.update({idx(2, week - 1, s * P + p): 1 for p in range(P)})
            opening = 0 if week else base_inst.I[s].sum()
            row(terms, hi=base_inst.C[s] - opening)
        for sp in range(N):
            s, p = divmod(sp, P)
            initial = int(base_inst.I[s, p]) if week == 0 else 0
            # store balance v = previous v + a - f
            terms = {idx(2, week, sp): 1, idx(1, week, sp): -1, idx(4, week, sp): 1}
            if week:
                terms[idx(2, week - 1, sp)] = -1
            row(terms, initial, initial)
            # deferred balance q = previous q + due - a
            terms = {idx(3, week, sp): 1, idx(1, week, sp): 1}
            if week:
                terms[idx(3, week - 1, sp)] = -1
            due = int(pipe[week][s, p]) if week < L else 0
            if week >= L:
                terms[idx(0, week - L, sp)] = -1
            row(terms, due, due)
            # Exact min: v <= M*z, D-f <= M*(1-z).
            big_m = float(max(base_inst.C[s], D[week, s, p], 1))
            row({idx(2, week, sp): 1, idx(5, week, sp): -big_m}, hi=0)
            row({idx(4, week, sp): -1, idx(5, week, sp): big_m}, hi=big_m - D[week, s, p])
    matrix = coo_matrix((values, (rows, cols)), shape=(len(lower), n)).tocsc()
    res = milp(
        c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"mip_rel_gap": 0.0},
    )
    if res.status != 0 or res.x is None:
        raise RuntimeError(f"Horizon oracle did not solve optimally: {res.message}")
    orders = np.rint(res.x[: T * N]).astype(int).reshape(T, S, P)
    admissions = np.rint(res.x[T * N : 2 * T * N]).astype(int).reshape(T, S, P)
    for week in range(T):
        sim.step(orders[week], D[week], week, admitted_units=admissions[week])
    original = float(res.fun + np.sum(D * base_inst.b))
    if not np.isclose(original, sim.total_cost_mu_cumulative, atol=1e-5, rtol=1e-9):
        raise AssertionError(
            f"Oracle objective/replay mismatch: {original}, {sim.total_cost_mu_cumulative}"
        )
    summary = sim.get_summary()
    return dict(
        x_orders=orders,
        admissions=admissions,
        total_cost_mu=sim.total_cost_mu_cumulative,
        solver_objective_mu=original,
        mip_gap=float(res.mip_gap),
        mean_cost_mu=summary["mean_cost_mu"],
        mean_shortage_cost_mu=summary["mean_shortage_cost_mu"],
        mean_holding_cost_mu=summary["mean_holding_cost_mu"],
        fill_rate=summary["fill_rate"],
        ending_inventory_units=summary["ending_store_inventory_units"],
        ending_in_transit_units=summary["ending_in_transit_units"],
        is_conserved=summary["is_conserved"],
        solver_status="optimal",
    )
