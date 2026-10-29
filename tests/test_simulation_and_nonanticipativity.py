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


def test_end_to_end_trained_model_dfl_dro_nonanticipativity(demo_release_config) -> None:
    """Verify that the complete trained-model-to-allocation pipeline—including
    rolling feature generation (compute_weekly_features_strictly_pre_origin),
    1-week-ahead feature projection (_project_features_one_week_ahead),
    supervised and DFL neural PMF inference (predict_sku_origin_outputs),
    empirical copula scenario sampling (EmpiricalCopulaScenarioGenerator),
    and nominal/DRO MILP allocations (proportional_allocation, supervised_pto,
    dfl_pto, supervised_dro, dfl_dro)—produces bit-for-bit identical decisions
    at origin t_0 (under both L=0 and L=1) when realized demand at t_0 and
    future origins t > t_0 is perturbed.
    """
    from pathlib import Path

    import pandas as pd

    from inventory_di.config import load_config
    from inventory_di.data.features import (
        FEATURE_COLUMNS,
        compute_weekly_features_strictly_pre_origin,
    )
    from inventory_di.evaluation.backtest import (
        _project_features_one_week_ahead,
        build_allocation_instance_from_config_df,
        load_sku_forecast_artifacts,
    )
    from inventory_di.models.distributions import EmpiricalCopulaScenarioGenerator
    from inventory_di.models.supervised import predict_sku_origin_outputs
    from inventory_di.optimization.baselines import allocate_proportional_shares
    from inventory_di.optimization.core import solve_nominal_marginal_cost_milp
    from inventory_di.optimization.dro import solve_scenario_reweighting_dro_milp

    cfg = load_config(demo_release_config)
    proc_dir = Path(cfg.data.processed_dir)
    art_dir = Path(cfg.data.artifacts_dir)

    weekly_df = pd.read_parquet(proc_dir / "weekly_demand.parquet")
    inv_df = pd.read_parquet(proc_dir / "inventory_config.parquet")
    base_inst = build_allocation_instance_from_config_df(inv_df)
    S, P = len(base_inst.stores), len(base_inst.skus)
    store_idx = {s: i for i, s in enumerate(base_inst.stores)}
    sku_idx = {p: i for i, p in enumerate(base_inst.skus)}

    sup_artifacts = load_sku_forecast_artifacts(
        art_dir / "supervised_models.pt", hidden_dim=cfg.model.hidden_dim
    )
    dfl_artifacts = load_sku_forecast_artifacts(
        art_dir / "dfl_models.pt", hidden_dim=cfg.model.hidden_dim
    )
    first_art = next(iter(sup_artifacts.values()))

    # Fit empirical copulas strictly on split == 'train'
    train_df = weekly_df[weekly_df["split"] == "train"]
    copula_sup = EmpiricalCopulaScenarioGenerator(
        stores=base_inst.stores,
        skus=base_inst.skus,
        support_h=int(first_art.support_h),
        tail_mean_excess=float(first_art.tail_mean_excess),
    )
    copula_sup.fit_on_train(train_df)
    copula_dfl = EmpiricalCopulaScenarioGenerator(
        stores=base_inst.stores,
        skus=base_inst.skus,
        support_h=int(first_art.support_h),
        tail_mean_excess=float(first_art.tail_mean_excess),
    )
    copula_dfl.fit_on_train(train_df)

    test_origins = sorted(weekly_df[weekly_df["split"] == "test"]["forecast_origin"].unique())
    t0 = test_origins[1]  # Pick an interior test origin with prior history and future origins

    # Construct two raw weekly demand trajectories:
    # Trajectory A: unmodified historical + test trajectory
    # Trajectory B: identical for all origins < t0, but massively perturbs realized demand
    # at origin t0 (+50 units) and all future origins > t0 (+120 units).
    raw_cols = [
        "forecast_origin",
        "target_start",
        "target_end",
        "store_id",
        "sku_id",
        "observed_units_proxy",
        "event_flag_target_week",
        "price_at_origin_mu",
        "split",
    ]
    raw_A = weekly_df[raw_cols].copy()
    raw_B = weekly_df[raw_cols].copy()
    mask_t0 = raw_B["forecast_origin"] == t0
    mask_future = raw_B["forecast_origin"] > t0
    raw_B.loc[mask_t0, "observed_units_proxy"] = raw_B.loc[mask_t0, "observed_units_proxy"] + 50
    raw_B.loc[mask_future, "observed_units_proxy"] = raw_B.loc[mask_future, "observed_units_proxy"] + 120

    # Recompute rolling features from scratch on both trajectories
    feat_A = compute_weekly_features_strictly_pre_origin(raw_A)
    feat_B = compute_weekly_features_strictly_pre_origin(raw_B)

    origin_A = feat_A[feat_A["forecast_origin"] == t0].sort_values(["store_id", "sku_id"]).reset_index(drop=True)
    origin_B = feat_B[feat_B["forecast_origin"] == t0].sort_values(["store_id", "sku_id"]).reset_index(drop=True)

    # 1. Verify pre-origin features at t0 are strictly invariant to demand at t0 and > t0
    pd.testing.assert_frame_equal(origin_A[FEATURE_COLUMNS], origin_B[FEATURE_COLUMNS])

    # 2. Verify trailing demand signals for proportional_allocation are identical
    def _compute_trailing_signals(df_all: pd.DataFrame) -> np.ndarray:
        all_orig = sorted(df_all["forecast_origin"].unique())
        prior_orig = [o for o in all_orig if o < t0][-cfg.evaluation.trailing_weeks_proportional :]
        prior_sub = df_all[df_all["forecast_origin"].isin(prior_orig)]
        sig = np.zeros((S, P), dtype=float)
        for (s_id, p_id), grp in prior_sub.groupby(["store_id", "sku_id"]):
            sig[store_idx[str(s_id)], sku_idx[str(p_id)]] = float(grp["observed_units_proxy"].mean())
        return sig

    trail_A = _compute_trailing_signals(feat_A)
    trail_B = _compute_trailing_signals(feat_B)
    np.testing.assert_allclose(trail_A, trail_B, atol=0.0)

    # 3. Execute full model-to-allocation pipeline for both L=0 and L=1
    def _run_origin_decisions(origin_df: pd.DataFrame, trail_sig: np.ndarray, lead_time: int) -> dict[str, np.ndarray]:
        I_curr = np.array([[2, 1], [1, 2], [0, 1]], dtype=int)
        arriving_curr = np.array([[1, 1], [1, 0], [1, 1]], dtype=int)

        # Base F_{t0} predictions
        sup_pmfs: dict[str, np.ndarray] = {}
        sup_mc: dict[str, np.ndarray] = {}
        sup_exp = np.zeros((S, P), dtype=float)
        dfl_pmfs: dict[str, np.ndarray] = {}
        dfl_mc: dict[str, np.ndarray] = {}
        dfl_exp = np.zeros((S, P), dtype=float)
        for p_i, sku in enumerate(base_inst.skus):
            s_out = predict_sku_origin_outputs(
                sup_artifacts[sku],
                origin_df,
                I_s=I_curr[:, p_i],
                K_s=base_inst.K[:, p_i],
                b_s=base_inst.b[:, p_i],
                h_s=base_inst.h[:, p_i],
                t_s=base_inst.t[:, p_i],
            )
            sup_pmfs[sku] = s_out["pmf"]
            sup_mc[sku] = s_out["c_hat_flat"]
            sup_exp[:, p_i] = s_out["expected_demand"]

            d_out = predict_sku_origin_outputs(
                dfl_artifacts[sku],
                origin_df,
                I_s=I_curr[:, p_i],
                K_s=base_inst.K[:, p_i],
                b_s=base_inst.b[:, p_i],
                h_s=base_inst.h[:, p_i],
                t_s=base_inst.t[:, p_i],
            )
            dfl_pmfs[sku] = d_out["pmf"]
            dfl_mc[sku] = d_out["c_hat_flat"]
            dfl_exp[:, p_i] = d_out["expected_demand"]

        if lead_time == 0:
            I_alloc = I_curr.copy()
            sup_scen = copula_sup.sample_scenarios(sup_pmfs, num_scenarios=25, seed=12345)
            dfl_scen = copula_dfl.sample_scenarios(dfl_pmfs, num_scenarios=25, seed=12345)
            inst_alloc = AllocationInstance(
                stores=base_inst.stores,
                skus=base_inst.skus,
                I=I_alloc,
                W=base_inst.W,
                K=base_inst.K,
                C=base_inst.C,
                b=base_inst.b,
                h=base_inst.h,
                t=base_inst.t,
            )
            return {
                "sup_exp": sup_exp,
                "dfl_exp": dfl_exp,
                "sup_scen": sup_scen,
                "dfl_scen": dfl_scen,
                "proportional_allocation": allocate_proportional_shares(trail_sig, inst_alloc).x,
                "supervised_pto": solve_nominal_marginal_cost_milp(sup_mc, inst_alloc).x,
                "dfl_pto": solve_nominal_marginal_cost_milp(dfl_mc, inst_alloc).x,
                "supervised_dro": solve_scenario_reweighting_dro_milp(
                    sup_scen, inst_alloc, alpha=0.90, risk_weight=0.25
                ).x,
                "dfl_dro": solve_scenario_reweighting_dro_milp(
                    dfl_scen, inst_alloc, alpha=0.90, risk_weight=0.25
                ).x,
            }

        # Lead time L=1: project opening stock and arrival-week features strictly from F_{t0}
        I_proj_sup = np.maximum(0, I_curr + arriving_curr - sup_exp).astype(int)
        I_proj_dfl = np.maximum(0, I_curr + arriving_curr - dfl_exp).astype(int)
        feat_proj_sup = _project_features_one_week_ahead(
            origin_df, expected_demand=sup_exp, stores=base_inst.stores, skus=base_inst.skus
        )
        feat_proj_dfl = _project_features_one_week_ahead(
            origin_df, expected_demand=dfl_exp, stores=base_inst.stores, skus=base_inst.skus
        )
        mc_sup_l1 = {
            sku: predict_sku_origin_outputs(
                sup_artifacts[sku],
                feat_proj_sup,
                I_s=I_proj_sup[:, p_i],
                K_s=base_inst.K[:, p_i],
                b_s=base_inst.b[:, p_i],
                h_s=base_inst.h[:, p_i],
                t_s=base_inst.t[:, p_i],
            )["c_hat_flat"]
            for p_i, sku in enumerate(base_inst.skus)
        }
        pmf_sup_l1 = {
            sku: predict_sku_origin_outputs(
                sup_artifacts[sku],
                feat_proj_sup,
                I_s=I_proj_sup[:, p_i],
                K_s=base_inst.K[:, p_i],
                b_s=base_inst.b[:, p_i],
                h_s=base_inst.h[:, p_i],
                t_s=base_inst.t[:, p_i],
            )["pmf"]
            for p_i, sku in enumerate(base_inst.skus)
        }
        mc_dfl_l1 = {
            sku: predict_sku_origin_outputs(
                dfl_artifacts[sku],
                feat_proj_dfl,
                I_s=I_proj_dfl[:, p_i],
                K_s=base_inst.K[:, p_i],
                b_s=base_inst.b[:, p_i],
                h_s=base_inst.h[:, p_i],
                t_s=base_inst.t[:, p_i],
            )["c_hat_flat"]
            for p_i, sku in enumerate(base_inst.skus)
        }
        pmf_dfl_l1 = {
            sku: predict_sku_origin_outputs(
                dfl_artifacts[sku],
                feat_proj_dfl,
                I_s=I_proj_dfl[:, p_i],
                K_s=base_inst.K[:, p_i],
                b_s=base_inst.b[:, p_i],
                h_s=base_inst.h[:, p_i],
                t_s=base_inst.t[:, p_i],
            )["pmf"]
            for p_i, sku in enumerate(base_inst.skus)
        }
        sup_scen_l1 = copula_sup.sample_scenarios(pmf_sup_l1, num_scenarios=25, seed=54321)
        dfl_scen_l1 = copula_dfl.sample_scenarios(pmf_dfl_l1, num_scenarios=25, seed=54321)

        inst_sup_l1 = AllocationInstance(
            stores=base_inst.stores,
            skus=base_inst.skus,
            I=I_proj_sup,
            W=base_inst.W,
            K=base_inst.K,
            C=base_inst.C,
            b=base_inst.b,
            h=base_inst.h,
            t=base_inst.t,
        )
        inst_dfl_l1 = AllocationInstance(
            stores=base_inst.stores,
            skus=base_inst.skus,
            I=I_proj_dfl,
            W=base_inst.W,
            K=base_inst.K,
            C=base_inst.C,
            b=base_inst.b,
            h=base_inst.h,
            t=base_inst.t,
        )
        return {
            "sup_exp": sup_exp,
            "dfl_exp": dfl_exp,
            "sup_scen": sup_scen_l1,
            "dfl_scen": dfl_scen_l1,
            "proportional_allocation": allocate_proportional_shares(trail_sig, inst_sup_l1).x,
            "supervised_pto": solve_nominal_marginal_cost_milp(mc_sup_l1, inst_sup_l1).x,
            "dfl_pto": solve_nominal_marginal_cost_milp(mc_dfl_l1, inst_dfl_l1).x,
            "supervised_dro": solve_scenario_reweighting_dro_milp(
                sup_scen_l1, inst_sup_l1, alpha=0.90, risk_weight=0.25
            ).x,
            "dfl_dro": solve_scenario_reweighting_dro_milp(
                dfl_scen_l1, inst_dfl_l1, alpha=0.90, risk_weight=0.25
            ).x,
        }

    for lt in [0, 1]:
        res_A = _run_origin_decisions(origin_A, trail_A, lead_time=lt)
        res_B = _run_origin_decisions(origin_B, trail_B, lead_time=lt)
        for key in res_A:
            np.testing.assert_array_equal(
                res_A[key],
                res_B[key],
                err_msg=f"Nonanticipativity violated for '{key}' at lead_time={lt}",
            )
