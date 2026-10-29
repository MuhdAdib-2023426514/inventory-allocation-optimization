"""Regression cases reproduced during the interview readiness review."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from inventory_di.config import load_config
from inventory_di.evaluation import backtest
from inventory_di.models.distributions import (
    EmpiricalCopulaScenarioGenerator,
    compute_survival_and_marginal_costs_torch,
)
from inventory_di.optimization.core import AllocationInstance, solve_multiperiod_carryover_oracle
from inventory_di.simulation.simulator import InventoryStateTransitionSystem


def inventory_frame(supply: int = 5) -> pd.DataFrame:
    return pd.DataFrame([dict(
        store_id="s", sku_id="p", opening_stock_units=0,
        warehouse_supply_units=supply, store_sku_unit_cap=5,
        store_total_capacity_units=10, unit_shortage_penalty_mu=6.0,
        unit_holding_cost_mu=1.0, unit_ship_cost_mu=0.25,
    )])


def test_zero_warehouse_supply_does_not_create_stock() -> None:
    inst = backtest.build_allocation_instance_from_config_df(inventory_frame(supply=0))
    assert inst.W.tolist() == [0]
    inst = backtest.build_allocation_instance_from_config_df(inventory_frame(), supply_multiplier=0)
    assert inst.W.tolist() == [0]


def test_carryover_nominal_policy_values_units_using_current_stock(monkeypatch) -> None:
    """A store already holding all five needed units must not order five more.

    Exercise the actual multi-period evaluator, with deterministic forecasts and
    two realized periods [0, 5]. First-period excess costs 6.25 MU including
    shipping; the second period should cost zero (mean 3.125, previously 6.25).
    """
    cfg = load_config("configs/demo.yaml")
    cfg.evaluation.seeds = []
    cfg.dro.alpha_grid = [0.9]
    cfg.dro.risk_weight_grid = [0.5]
    cfg.dro.num_scenarios = 2
    weekly = pd.DataFrame([
        dict(forecast_origin=o, target_start=o, store_id="s", sku_id="p",
             observed_units_proxy=d, split="test", lag_1w_units=5.0,
             lag_2w_units=5.0, rolling_4w_mean=5.0, rolling_8w_mean=5.0)
        for o, d in [("2020-01-01", 0), ("2020-01-08", 5)]
    ])
    pmf = torch.zeros(1, 1, 12)
    pmf[0, 0, 5] = 1

    def predict(artifact, origin_df, I_s, K_s, b_s, h_s, t_s):
        _, costs = compute_survival_and_marginal_costs_torch(pmf, I_s, K_s, b_s, h_s, t_s)
        return dict(pmf=pmf.numpy()[0], c_hat_flat=costs.numpy()[0], expected_demand=np.array([5.0]))

    monkeypatch.setattr(backtest, "predict_sku_origin_outputs", predict)
    arts = {"p": SimpleNamespace(support_h=10, tail_mean_excess=2.0)}
    copula = EmpiricalCopulaScenarioGenerator(["s"], ["p"], 10)
    _, carryover, _ = backtest.run_multi_seed_and_carryover_experiments(
        cfg, weekly, inventory_frame(), arts, arts, copula, copula,
        dict(alpha=0.9, risk_weight=0.5), dict(alpha=0.9, risk_weight=0.5),
    )
    nominal = carryover.query("lead_time_weeks == 0 and policy in ['supervised_pto', 'dfl_pto']")
    np.testing.assert_allclose(nominal["mean_cost_mu"], 3.125)
    assert nominal["is_conserved"].all()


def test_horizon_oracle_bounds_a_feasible_deferred_delivery_policy() -> None:
    inst = AllocationInstance(
        ["s"], ["p"], np.array([[0]]), np.array([3]), np.array([[3]]),
        np.array([5]), np.array([[6.0]]), np.array([[0.01]]), np.array([[0.0]]),
    )
    demand = np.array([[[0]], [[0]], [[5]], [[5]]])
    sim = InventoryStateTransitionSystem(inst)
    for period, quantity in enumerate([3, 3, 3, 1]):
        sim.step(np.array([[quantity]]), demand[period], period)
    assert sim.verify_conservation()[0]
    oracle = solve_multiperiod_carryover_oracle(demand, inst)
    assert oracle["solver_status"] == "optimal"
    assert oracle["total_cost_mu"] <= sim.total_cost_mu_cumulative + 1e-6
