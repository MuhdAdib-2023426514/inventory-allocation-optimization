"""Heuristic baselines (`equal_allocation`, `proportional_allocation`) and `clairvoyant_oracle` (`AGENTS.md` §8)."""

from __future__ import annotations

import time

import numpy as np

from inventory_di.optimization.core import (
    AllocationInstance,
    SolveResult,
    check_allocation_feasibility,
    solve_deterministic_milp,
)


def allocate_equal_shares(inst: AllocationInstance) -> SolveResult:
    """Deterministic near-equal warehouse share allocator (`equal_allocation`).

    Distributes available warehouse supply `W_p` as evenly as possible across stores `s`
    in deterministic round-robin order while strictly respecting per-store-SKU caps `K_{s,p}`
    and remaining store total capacity `C_s - sum_p (I_{s,p} + x_{s,p})`.
    Never over-ships warehouse supply or store capacity.
    """
    t0 = time.perf_counter()
    S, P = len(inst.stores), len(inst.skus)
    x = np.zeros((S, P), dtype=int)
    rem_store_cap = (inst.C - np.sum(inst.I, axis=1)).astype(int).copy()

    for p in range(P):
        rem_w = int(inst.W[p])
        while rem_w > 0:
            progressed = False
            for s in range(S):
                if rem_w <= 0:
                    break
                if x[s, p] < int(inst.K[s, p]) and rem_store_cap[s] > 0:
                    x[s, p] += 1
                    rem_w -= 1
                    rem_store_cap[s] -= 1
                    progressed = True
            if not progressed:
                break

    ok, viols = check_allocation_feasibility(x, inst)
    return SolveResult(
        x=x,
        solver_objective=0.0,
        solver_status="heuristic_feasible",
        is_feasible=ok,
        violations=viols,
        runtime_seconds=time.perf_counter() - t0,
    )


def allocate_proportional_shares(
    trailing_demand_signals: np.ndarray,
    inst: AllocationInstance,
) -> SolveResult:
    """Proportional allocator (`proportional_allocation`) using strictly pre-origin trailing demand.

    Allocates each SKU's warehouse supply `W_p` proportionally to `trailing_demand_signals[s, p]`
    (with largest-remainder deterministic tie-breaking) while enforcing `K_{s,p}` and `C_s`.
    """
    t0 = time.perf_counter()
    S, P = len(inst.stores), len(inst.skus)
    x = np.zeros((S, P), dtype=int)
    rem_store_cap = (inst.C - np.sum(inst.I, axis=1)).astype(int).copy()
    signals = np.maximum(np.asarray(trailing_demand_signals, dtype=float), 0.0)

    for p in range(P):
        rem_w = int(inst.W[p])
        weights = signals[:, p].copy()
        if float(np.sum(weights)) <= 1e-12:
            weights = np.ones(S, dtype=float)
        weights = weights / float(np.sum(weights))

        # Initial floor target bounded by caps
        raw_targets = weights * rem_w
        floors = np.floor(raw_targets).astype(int)
        for s in range(S):
            give = max(0, min(int(floors[s]), int(inst.K[s, p]), int(rem_store_cap[s]), rem_w))
            x[s, p] += give
            rem_w -= give
            rem_store_cap[s] -= give

        # Distribute remaining units by largest fractional remainder, then store index
        remainders = raw_targets - floors
        order = sorted(range(S), key=lambda s_idx: (-float(remainders[s_idx]), s_idx))
        while rem_w > 0:
            progressed = False
            for s in order:
                if rem_w <= 0:
                    break
                if x[s, p] < int(inst.K[s, p]) and rem_store_cap[s] > 0:
                    x[s, p] += 1
                    rem_w -= 1
                    rem_store_cap[s] -= 1
                    progressed = True
            if not progressed:
                break

    ok, viols = check_allocation_feasibility(x, inst)
    return SolveResult(
        x=x,
        solver_objective=0.0,
        solver_status="heuristic_feasible",
        is_feasible=ok,
        violations=viols,
        runtime_seconds=time.perf_counter() - t0,
    )


def solve_clairvoyant_oracle(
    realized_demand_proxy: np.ndarray,
    inst: AllocationInstance,
) -> SolveResult:
    """Solve hindsight optimal MILP (`clairvoyant_oracle`) using held-out demand proxy.
    Strictly restricted to hindsight regret calculation (`AGENTS.md` §8).
    """
    return solve_deterministic_milp(realized_demand_proxy, inst)
