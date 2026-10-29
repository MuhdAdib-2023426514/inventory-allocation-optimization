"""Tests for mandatory 2-store toy mathematical oracle, nominal MILP, and baseline allocators (`AGENTS.md` §5, §8, §14)."""

from __future__ import annotations

import numpy as np

from inventory_di.optimization.baselines import (
    allocate_equal_shares,
    allocate_proportional_shares,
    solve_clairvoyant_oracle,
)
from inventory_di.optimization.core import (
    AllocationInstance,
    check_allocation_feasibility,
    compute_true_marginal_unit_costs,
    evaluate_exact_cost,
    solve_deterministic_milp,
    solve_multiperiod_carryover_oracle,
    solve_nominal_marginal_cost_milp,
)


def test_mandatory_toy_mathematical_oracle() -> None:
    """AGENTS.md §5: One product, two stores, zero initial stock:
    demand [60,20], warehouse 60, per-store allocation caps [60,60],
    shortage penalty 1, holding/shipping costs 0.
    Equal 30/30 wastes 10 units at store B and fulfills 50 (fill rate 62.5%, cost 30).
    Optimal 40/20 fulfills 60 (fill rate 75%, cost 20).
    """
    inst = AllocationInstance(
        stores=["Store_A", "Store_B"],
        skus=["SKU_1"],
        I=np.array([[0], [0]], dtype=int),
        W=np.array([60], dtype=int),
        K=np.array([[60], [60]], dtype=int),
        C=np.array([60, 60], dtype=int),
        b=np.array([[1.0], [1.0]], dtype=float),
        h=np.array([[0.0], [0.0]], dtype=float),
        t=np.array([[0.0], [0.0]], dtype=float),
    )
    D = np.array([[60], [20]], dtype=int)

    # 1. Equal allocation 30/30
    eq_res = allocate_equal_shares(inst)
    assert eq_res.is_feasible
    np.testing.assert_array_equal(eq_res.x, np.array([[30], [30]]))
    eq_cost = evaluate_exact_cost(eq_res.x, D, inst)
    assert abs(eq_cost.total_cost_mu - 30.0) < 1e-9
    assert eq_cost.filled_units == 50
    assert eq_cost.unmet_units == 30
    assert abs(eq_cost.fill_rate - 0.625) < 1e-9

    # 2. Optimal allocation 40/20 and MILP solver optimality check
    opt_40_20_cost = evaluate_exact_cost(np.array([[40], [20]]), D, inst)
    assert abs(opt_40_20_cost.total_cost_mu - 20.0) < 1e-9
    assert opt_40_20_cost.filled_units == 60
    assert opt_40_20_cost.unmet_units == 20
    assert abs(opt_40_20_cost.fill_rate - 0.75) < 1e-9

    opt_res = solve_deterministic_milp(D, inst)
    assert opt_res.is_feasible
    assert opt_res.solver_status == "optimal"
    opt_cost = evaluate_exact_cost(opt_res.x, D, inst)
    assert abs(opt_cost.total_cost_mu - 20.0) < 1e-9
    assert opt_cost.filled_units == 60
    assert opt_cost.unmet_units == 20
    assert abs(opt_cost.fill_rate - 0.75) < 1e-9


def test_baselines_and_marginal_milp_feasibility_and_regret() -> None:
    inst = AllocationInstance(
        stores=["S1", "S2", "S3"],
        skus=["P1", "P2"],
        I=np.array([[1, 0], [0, 2], [1, 1]], dtype=int),
        W=np.array([10, 8], dtype=int),
        K=np.array([[6, 5], [6, 5], [6, 5]], dtype=int),
        C=np.array([10, 10, 10], dtype=int),
        b=np.array([[6.0, 5.5], [6.2, 5.8], [5.9, 6.1]], dtype=float),
        h=np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=float),
        t=np.array([[0.2, 0.2], [0.25, 0.25], [0.3, 0.3]], dtype=float),
    )
    D_realized = np.array([[7, 4], [5, 6], [2, 3]], dtype=int)
    trailing_signals = np.array([[6.0, 3.0], [4.5, 5.0], [2.0, 2.5]], dtype=float)

    eq_res = allocate_equal_shares(inst)
    prop_res = allocate_proportional_shares(trailing_signals, inst)
    oracle_res = solve_clairvoyant_oracle(D_realized, inst)

    # Also test marginal-cost MILP with true marginal costs (should match oracle)
    mc_by_sku = {
        sku: compute_true_marginal_unit_costs(
            D_realized[:, p],
            inst.I[:, p],
            inst.K[:, p],
            inst.b[:, p],
            inst.h[:, p],
            inst.t[:, p],
        )
        for p, sku in enumerate(inst.skus)
    }
    mc_res = solve_nominal_marginal_cost_milp(mc_by_sku, inst)

    oracle_cost = evaluate_exact_cost(oracle_res.x, D_realized, inst).total_cost_mu
    mc_cost = evaluate_exact_cost(mc_res.x, D_realized, inst).total_cost_mu
    assert abs(oracle_cost - mc_cost) < 1e-6

    for res in [eq_res, prop_res, oracle_res, mc_res]:
        ok, viols = check_allocation_feasibility(res.x, inst)
        assert ok, f"Feasibility violation: {viols}"
        c = evaluate_exact_cost(res.x, D_realized, inst).total_cost_mu
        regret = c - oracle_cost
        assert regret >= -1e-6, f"Negative oracle regret: {regret}"


def test_multiperiod_carryover_oracle_nonnegative_regret() -> None:
    """Verify solve_multiperiod_carryover_oracle achieves nonnegative rolling regret at L=0 and L=1."""
    inst = AllocationInstance(
        stores=["S1", "S2"],
        skus=["P1"],
        I=np.array([[1], [0]], dtype=int),
        W=np.array([10], dtype=int),
        K=np.array([[8], [8]], dtype=int),
        C=np.array([12, 12], dtype=int),
        b=np.array([[6.0], [5.5]], dtype=float),
        h=np.array([[1.0], [1.0]], dtype=float),
        t=np.array([[0.2], [0.3]], dtype=float),
    )
    D_horizon = np.array(
        [
            [[6], [4]],
            [[8], [3]],
            [[3], [7]],
            [[5], [5]],
        ],
        dtype=int,
    )
    T = D_horizon.shape[0]

    for lead_time in [0, 1]:
        init_in_transit = [allocate_equal_shares(inst).x.copy() for _ in range(lead_time)]
        oracle_mp = solve_multiperiod_carryover_oracle(
            D_trajectory=D_horizon,
            base_inst=inst,
            lead_time=lead_time,
            initial_pipeline=init_in_transit if lead_time > 0 else None,
        )
        assert oracle_mp["solver_status"] == "optimal"

        # Simulate equal_allocation and clairvoyant_oracle under identical state transitions
        for pol in ["equal_allocation", "clairvoyant_oracle"]:
            I_curr = inst.I.copy()
            pipe = [arr.copy() for arr in init_in_transit]
            tot_cost = 0.0
            for t_i in range(T):
                C_rem = np.maximum(inst.C - np.sum(I_curr, axis=1), 0)
                K_rem = np.minimum(inst.K, C_rem[:, None])
                inst_t = AllocationInstance(
                    stores=inst.stores,
                    skus=inst.skus,
                    I=I_curr.copy(),
                    W=inst.W.copy(),
                    K=K_rem,
                    C=inst.C.copy(),
                    b=inst.b.copy(),
                    h=inst.h.copy(),
                    t=inst.t.copy(),
                )
                if pol == "equal_allocation":
                    x_ord = allocate_equal_shares(inst_t).x
                else:
                    x_ord = oracle_mp["x_orders"][t_i]

                if lead_time == 0:
                    x_arr = x_ord
                else:
                    pipe.append(x_ord.copy())
                    x_arr = pipe.pop(0)

                c_t = evaluate_exact_cost(x_arr, D_horizon[t_i], inst_t)
                tot_cost += c_t.total_cost_mu
                I_curr = np.maximum(I_curr + x_arr - D_horizon[t_i], 0).astype(int)

            if pol == "clairvoyant_oracle":
                oracle_sim_cost = tot_cost
                assert abs(oracle_sim_cost - oracle_mp["total_cost_mu"]) < 1e-4
            else:
                eq_sim_cost = tot_cost

        assert eq_sim_cost - oracle_sim_cost >= -1e-6, (
            f"Negative rolling oracle regret at L={lead_time}: eq={eq_sim_cost}, oracle={oracle_sim_cost}"
        )
