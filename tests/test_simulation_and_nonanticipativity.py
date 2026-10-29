"""Tests for physical inventory conservation and nonanticipativity (`AGENTS.md` §2, §5; Phases 1 & 2)."""

from __future__ import annotations

import numpy as np
import pytest

from inventory_di.optimization.baselines import allocate_equal_shares
from inventory_di.optimization.core import AllocationInstance
from inventory_di.simulation.simulator import InventoryStateTransitionSystem


@pytest.fixture
def base_test_instance() -> AllocationInstance:
    return AllocationInstance(
        stores=["Store_1", "Store_2", "Store_3"],
        skus=["SKU_A", "SKU_B"],
        I=np.array([[2, 1], [0, 3], [1, 1]], dtype=int),
        W=np.array([20, 15], dtype=int),
        K=np.array([[10, 8], [10, 8], [10, 8]], dtype=int),
        C=np.array([12, 12, 12], dtype=int),
        b=np.array([[5.0, 4.0], [5.0, 4.0], [5.0, 4.0]], dtype=float),
        h=np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=float),
        t=np.array([[0.2, 0.2], [0.2, 0.2], [0.2, 0.2]], dtype=float),
    )


def test_inventory_conservation_under_random_trajectories(base_test_instance: AllocationInstance) -> None:
    """Invariant test: Opening System Stock + Receipts == Closing Stock + Fulfilled Sales on every step."""
    rng = np.random.default_rng(42)
    inst = base_test_instance
    S, P = len(inst.stores), len(inst.skus)
    T = 8

    for lead_time in [0, 1, 2]:
        init_pipeline = [rng.integers(0, 3, size=(S, P), dtype=int) for _ in range(lead_time)]
        sim = InventoryStateTransitionSystem(
            base_inst=inst,
            lead_time_weeks=lead_time,
            initial_pipeline=init_pipeline,
        )

        for t in range(T):
            order = rng.integers(0, 4, size=(S, P), dtype=int)
            # Enforce warehouse constraint on order
            for p in range(P):
                if np.sum(order[:, p]) > inst.W[p]:
                    order[:, p] = np.floor(order[:, p] * (inst.W[p] / np.sum(order[:, p]))).astype(int)

            demand = rng.integers(0, 8, size=(S, P), dtype=int)
            sim.step(order_x=order, realized_demand_D=demand, period_idx=t)
            is_conserved, error = sim.verify_conservation()
            assert is_conserved, f"Conservation failed at step {t} for lead_time {lead_time}: error={error}"

        summary = sim.get_summary()
        assert summary["is_conserved"]


def test_inventory_conservation_mandatory_edge_cases(base_test_instance: AllocationInstance) -> None:
    """Test all mandatory edge cases: zero demand, zero supply, store capacity full, delayed shipments."""
    inst = base_test_instance
    S, P = len(inst.stores), len(inst.skus)

    # Edge Case 1: Zero store demand across all periods
    sim_zero_d = InventoryStateTransitionSystem(base_inst=inst, lead_time_weeks=1)
    for t in range(4):
        order = np.ones((S, P), dtype=int)
        demand = np.zeros((S, P), dtype=int)
        sim_zero_d.step(order_x=order, realized_demand_D=demand, period_idx=t)
        assert sim_zero_d.verify_conservation()[0]
    assert sim_zero_d.fulfilled_sales_cumulative == 0

    # Edge Case 2: Zero warehouse stock / zero orders placed
    sim_zero_w = InventoryStateTransitionSystem(base_inst=inst, lead_time_weeks=1)
    for t in range(4):
        order = np.zeros((S, P), dtype=int)
        demand = np.full((S, P), 5, dtype=int)
        sim_zero_w.step(order_x=order, realized_demand_D=demand, period_idx=t)
        assert sim_zero_w.verify_conservation()[0]

    # Edge Case 3: Store capacity completely full before arrival (excess deferred, never destroyed)
    full_c_inst = AllocationInstance(
        stores=["S1"],
        skus=["P1"],
        I=np.array([[10]], dtype=int),
        W=np.array([20], dtype=int),
        K=np.array([[20]], dtype=int),
        C=np.array([10], dtype=int),  # store is already 100% full
        b=np.array([[5.0]], dtype=float),
        h=np.array([[1.0]], dtype=float),
        t=np.array([[0.2]], dtype=float),
    )
    sim_full = InventoryStateTransitionSystem(base_inst=full_c_inst, lead_time_weeks=0)
    # Attempt to deliver 8 units to a full store with 0 demand
    log1 = sim_full.step(order_x=np.array([[8]]), realized_demand_D=np.array([[0]]), period_idx=0)
    assert log1.accepted_arrivals[0, 0] == 0
    assert log1.deferred_arrivals_after[0, 0] == 8
    assert sim_full.verify_conservation()[0]
    # Next period, 6 units of demand occurs, freeing space
    log2 = sim_full.step(order_x=np.array([[0]]), realized_demand_D=np.array([[6]]), period_idx=1)
    assert log2.fulfilled_sales[0, 0] == 6  # 6 units sold from on-hand
    # Store closing on-hand is 10 - 6 = 4. Total system conserved:
    assert sim_full.verify_conservation()[0]

    # Edge Case 4: Demand strictly greater than available inventory (shortage occurs)
    sim_shortage = InventoryStateTransitionSystem(base_inst=full_c_inst, lead_time_weeks=0)
    log_sh = sim_shortage.step(order_x=np.array([[2]]), realized_demand_D=np.array([[50]]), period_idx=0)
    assert log_sh.fulfilled_sales[0, 0] == 10  # only on-hand was available
    assert log_sh.unmet_shortage[0, 0] == 40
    assert sim_shortage.verify_conservation()[0]


def test_nonanticipativity_future_demand_independence() -> None:
    """Strict nonanticipativity test: Changing future demand must NEVER alter earlier decisions."""
    inst = AllocationInstance(
        stores=["S1", "S2"],
        skus=["P1"],
        I=np.array([[2], [1]], dtype=int),
        W=np.array([10], dtype=int),
        K=np.array([[6], [6]], dtype=int),
        C=np.array([10, 10], dtype=int),
        b=np.array([[6.0], [5.0]], dtype=float),
        h=np.array([[1.0], [1.0]], dtype=float),
        t=np.array([[0.2], [0.3]], dtype=float),
    )

    # Trajectory A vs Trajectory B: identical demand at t=0, completely different demand at t=1, t=2
    D_traj_A = np.array([[[4], [3]], [[10], [15]], [[20], [25]]], dtype=int)
    D_traj_B = np.array([[[4], [3]], [[0], [1]], [[2], [0]]], dtype=int)

    # Both causal policy simulations at t=0 use only information available at t=0
    sim_A = InventoryStateTransitionSystem(base_inst=inst, lead_time_weeks=1)
    sim_B = InventoryStateTransitionSystem(base_inst=inst, lead_time_weeks=1)

    order_A_0 = allocate_equal_shares(inst).x
    order_B_0 = allocate_equal_shares(inst).x
    np.testing.assert_array_equal(order_A_0, order_B_0)

    log_A_0 = sim_A.step(order_x=order_A_0, realized_demand_D=D_traj_A[0], period_idx=0)
    log_B_0 = sim_B.step(order_x=order_B_0, realized_demand_D=D_traj_B[0], period_idx=0)
    np.testing.assert_array_equal(log_A_0.fulfilled_sales, log_B_0.fulfilled_sales)
    np.testing.assert_array_equal(log_A_0.closing_store_on_hand, log_B_0.closing_store_on_hand)

    # Future demand steps diverge, but step 0 was strictly invariant to future
    sim_A.step(order_x=np.zeros_like(order_A_0), realized_demand_D=D_traj_A[1], period_idx=1)
    sim_B.step(order_x=np.zeros_like(order_B_0), realized_demand_D=D_traj_B[1], period_idx=1)
    assert sim_A.verify_conservation()[0]
    assert sim_B.verify_conservation()[0]
