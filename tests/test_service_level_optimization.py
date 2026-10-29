"""Tests for service-level constrained allocation optimization (`AGENTS.md` §7, Phase 7)."""

from __future__ import annotations

import numpy as np

from inventory_di.optimization.core import (
    AllocationInstance,
    solve_service_level_constrained_milp,
)


def test_service_level_constrained_milp_feasible_and_infeasible_slack() -> None:
    """Verify service-level constrained MILP achieves target fill rate when feasible,
    and returns graceful relaxation with shortfall slack when warehouse supply is scarce.
    """
    inst = AllocationInstance(
        stores=["S1", "S2"],
        skus=["P1"],
        I=np.array([[2], [2]], dtype=int),
        W=np.array([20], dtype=int),
        K=np.array([[15], [15]], dtype=int),
        C=np.array([20, 20], dtype=int),
        b=np.array([[6.0], [5.0]], dtype=float),
        h=np.array([[1.0], [1.0]], dtype=float),
        t=np.array([[0.2], [0.2]], dtype=float),
    )

    # 4 demand scenarios with average total demand = 16 units
    scenarios = np.array(
        [
            [[6], [6]],  # 12
            [[8], [8]],  # 16
            [[10], [10]],  # 20
            [[8], [8]],  # 16
        ],
        dtype=float,
    )

    # Case 1: Target fill rate 85% is easily feasible with W=20 + I=4 = 24 units vs expected demand 16
    res_feas = solve_service_level_constrained_milp(
        scenarios=scenarios,
        inst=inst,
        target_fill_rate=0.85,
    )
    assert res_feas.is_feasible
    assert res_feas.is_service_level_met
    assert res_feas.expected_fill_rate >= 0.85 - 1e-4
    assert res_feas.achieved_fill_rate == res_feas.expected_fill_rate
    assert res_feas.service_level_shortfall == 0.0
    assert res_feas.shortfall_units == 0.0
    assert res_feas.expected_cost_mu == res_feas.model_expected_cost_mu

    # Case 2: Severe warehouse shortage where W = 2 units (total supply 2 + 4 = 6 units vs expected demand 16)
    # Maximum achievable fill rate is 6 / 16 = 37.5%, so target 90% is physically impossible.
    inst_scarce = AllocationInstance(
        stores=["S1", "S2"],
        skus=["P1"],
        I=np.array([[2], [2]], dtype=int),
        W=np.array([2], dtype=int),  # only 2 units available
        K=np.array([[15], [15]], dtype=int),
        C=np.array([20, 20], dtype=int),
        b=np.array([[6.0], [5.0]], dtype=float),
        h=np.array([[1.0], [1.0]], dtype=float),
        t=np.array([[0.2], [0.2]], dtype=float),
    )
    res_infeas = solve_service_level_constrained_milp(
        scenarios=scenarios,
        inst=inst_scarce,
        target_fill_rate=0.90,
    )
    assert res_infeas.is_feasible  # Physical allocation (W, K, C) is feasible
    assert not res_infeas.is_service_level_met  # Service level has a shortfall
    assert res_infeas.service_level_shortfall > 0.40  # Shortfall is quantified
    assert res_infeas.shortfall_units > 6.0
    assert "shortfall" in res_infeas.solver_status
    assert int(np.sum(res_infeas.x)) <= 2  # Respects scarce supply
