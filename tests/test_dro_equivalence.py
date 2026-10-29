"""Tests for probability-reweighting DRO / CVaR LP equivalence, brute-force tiny enumeration,
and risk-neutral limiting cases (`AGENTS.md` §7, §14).
"""

from __future__ import annotations

import numpy as np

from inventory_di.optimization.core import AllocationInstance, evaluate_exact_cost
from inventory_di.optimization.dro import (
    exact_ambiguity_set_worst_case_loss,
    rockafellar_uryasev_cvar_lp,
    solve_scenario_reweighting_dro_milp,
)


def test_cvar_lp_matches_probability_reweighting_ambiguity_set() -> None:
    rng = np.random.default_rng(123)
    for _ in range(15):
        J = int(rng.integers(5, 25))
        losses = rng.uniform(5.0, 120.0, size=J)
        raw_p = rng.uniform(0.5, 2.0, size=J)
        probs = raw_p / raw_p.sum()
        for alpha in [0.0, 0.50, 0.75, 0.90, 0.95]:
            reweight_val, w = exact_ambiguity_set_worst_case_loss(losses, probs, alpha=alpha)
            lp_val, _ = rockafellar_uryasev_cvar_lp(losses, probs, alpha=alpha)
            assert abs(reweight_val - lp_val) < 1e-6
            assert abs(float(np.sum(w)) - 1.0) < 1e-7
            assert np.all(w >= -1e-9)
            assert np.all(w <= probs / (1.0 - alpha) + 1e-7)


def test_dro_milp_matches_brute_force_enumeration_and_limiting_cases() -> None:
    """Verify on a tiny 2-store, 1-SKU instance that:
    1. DRO MILP matches brute-force enumeration over all feasible integer allocations x.
    2. alpha=0 with risk_weight=1 matches risk_weight=0 (risk-neutral expected loss).
    3. CVaR >= Expected Cost for all alpha in [0, 1).
    """
    inst = AllocationInstance(
        stores=["S1", "S2"],
        skus=["P1"],
        I=np.array([[0], [0]], dtype=int),
        W=np.array([6], dtype=int),
        K=np.array([[5], [5]], dtype=int),
        C=np.array([5, 5], dtype=int),
        b=np.array([[8.0], [5.0]], dtype=float),
        h=np.array([[1.0], [1.0]], dtype=float),
        t=np.array([[0.2], [0.4]], dtype=float),
    )
    # 5 scenarios of shape (J=5, S=2, P=1)
    scenarios = np.array(
        [
            [[2], [2]],
            [[3], [2]],
            [[4], [3]],
            [[5], [1]],
            [[5], [5]],
        ],
        dtype=int,
    )
    J = scenarios.shape[0]
    probs = np.full(J, 1.0 / J)

    for alpha, rw in [(0.0, 1.0), (0.80, 0.0), (0.80, 0.50), (0.80, 1.0)]:
        dro_res = solve_scenario_reweighting_dro_milp(
            scenarios, inst, alpha=alpha, risk_weight=rw, probs=probs
        )
        assert dro_res.is_feasible
        assert dro_res.solver_status == "optimal"
        assert dro_res.model_cvar_cost_mu + 1e-7 >= dro_res.model_expected_cost_mu

        # Brute-force enumerate all feasible integer (x0, x1) in [0..5] x [0..5] with x0 + x1 <= 6
        best_bf_obj = float("inf")
        for x0 in range(6):
            for x1 in range(6):
                if x0 + x1 > 6:
                    continue
                x_cand = np.array([[x0], [x1]], dtype=int)
                losses = np.array(
                    [evaluate_exact_cost(x_cand, scenarios[j], inst).total_cost_mu for j in range(J)]
                )
                exp_c = float(np.dot(probs, losses))
                cvar_c, _ = exact_ambiguity_set_worst_case_loss(losses, probs, alpha=alpha)
                obj_c = (1.0 - rw) * exp_c + rw * cvar_c
                if obj_c < best_bf_obj:
                    best_bf_obj = obj_c

        assert abs(dro_res.solver_objective - best_bf_obj) < 1e-6

    # Verify alpha=0, risk_weight=1 matches risk_weight=0 for identical scenarios
    res_alpha0 = solve_scenario_reweighting_dro_milp(scenarios, inst, alpha=0.0, risk_weight=1.0, probs=probs)
    res_rw0 = solve_scenario_reweighting_dro_milp(scenarios, inst, alpha=0.90, risk_weight=0.0, probs=probs)
    assert abs(res_alpha0.solver_objective - res_rw0.solver_objective) < 1e-6
    assert abs(res_alpha0.model_cvar_cost_mu - res_alpha0.model_expected_cost_mu) < 1e-6
