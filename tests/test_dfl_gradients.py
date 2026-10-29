"""Tests for probabilistic PMF invariants, survival/marginal cost monotonicity, and genuine PyEPO SPO+ gradients (`AGENTS.md` §6, §14)."""

from __future__ import annotations

import numpy as np
import torch

from inventory_di.config import load_config
from inventory_di.data.synthetic import generate_synthetic_demo_dataset
from inventory_di.models.dfl import train_dfl_sku_model
from inventory_di.models.distributions import (
    DiscreteDemandPMFNet,
    compute_survival_and_marginal_costs_torch,
)
from inventory_di.models.supervised import (
    predict_sku_origin_outputs,
    train_supervised_sku_model,
)


def test_pmf_survival_and_marginal_cost_monotonicity() -> None:
    torch.manual_seed(7)
    model = DiscreteDemandPMFNet(num_stores=3, in_features=9, support_h=12, hidden_dim=16)
    x = torch.randn(2, 3, 9)
    pmf = model.predict_pmf(x)

    assert pmf.shape == (2, 3, 14)
    np.testing.assert_allclose(pmf.detach().numpy().sum(axis=-1), np.ones((2, 3)), atol=1e-6)
    assert (pmf.detach().numpy() >= 0.0).all()

    I_s = np.array([1, 0, 2])
    K_s = np.array([5, 6, 4])
    b_s = np.array([6.0, 5.5, 6.2])
    h_s = np.array([1.0, 1.0, 1.0])
    t_s = np.array([0.2, 0.25, 0.3])

    q_flat, c_hat_flat = compute_survival_and_marginal_costs_torch(pmf, I_s, K_s, b_s, h_s, t_s)
    q_np = q_flat.detach().numpy()
    c_np = c_hat_flat.detach().numpy()

    for b_idx in range(2):
        offset = 0
        for s_idx, cap in enumerate(K_s):
            q_s = q_np[b_idx, offset : offset + cap]
            c_s = c_np[b_idx, offset : offset + cap]
            # Survival probabilities decrease in k, marginal unit costs increase in k
            assert np.all(np.diff(q_s) <= 1e-6), f"Non-monotonic survival q_s={q_s}"
            assert np.all(np.diff(c_s) >= -1e-6), f"Non-monotonic marginal cost c_s={c_s}"
            offset += cap


def test_spoplus_gradient_smoke_and_dfl_vs_nll_training() -> None:
    cfg = load_config("configs/demo.yaml", quick=True)
    weekly_df, inv_df, lineage = generate_synthetic_demo_dataset(cfg)
    stores = sorted(inv_df["store_id"].unique())
    sku_id = sorted(inv_df["sku_id"].unique())[0]
    sku_inv = inv_df[inv_df["sku_id"] == sku_id].sort_values("store_id")

    I_s = sku_inv["opening_stock_units"].to_numpy(dtype=int)
    K_s = sku_inv["store_sku_unit_cap"].to_numpy(dtype=int)
    W_p = int(sku_inv["warehouse_supply_units"].iloc[0])
    b_s = sku_inv["unit_shortage_penalty_mu"].to_numpy(dtype=float)
    h_s = sku_inv["unit_holding_cost_mu"].to_numpy(dtype=float)
    t_s = sku_inv["unit_ship_cost_mu"].to_numpy(dtype=float)

    sup_art = train_supervised_sku_model(
        weekly_df=weekly_df,
        sku_id=sku_id,
        stores=stores,
        support_h=cfg.model.max_demand_support_h,
        tail_mean_excess=lineage["support_stats"]["tail_mean_excess_above_h"],
        hidden_dim=16,
        learning_rate=0.02,
        epochs=4,
        seed=42,
    )
    dfl_art = train_dfl_sku_model(
        weekly_df=weekly_df,
        sku_id=sku_id,
        stores=stores,
        I_s=I_s,
        K_s=K_s,
        W_p=W_p,
        b_s=b_s,
        h_s=h_s,
        t_s=t_s,
        support_h=cfg.model.max_demand_support_h,
        tail_mean_excess=lineage["support_stats"]["tail_mean_excess_above_h"],
        hidden_dim=16,
        learning_rate=0.02,
        epochs=4,
        batch_size=8,
        lambda_dfl=1.0,
        spoplus_processes=1,
        seed=42,
    )

    assert dfl_art.gradient_check["grad_norm_pure_spoplus"] > 0.0
    assert dfl_art.gradient_check["param_diff_l2_vs_nll"] > 1e-6

    test_origin = sorted(weekly_df[weekly_df["split"] == "test"]["forecast_origin"].unique())[0]
    origin_df = weekly_df[weekly_df["forecast_origin"] == test_origin]
    sup_pred = predict_sku_origin_outputs(sup_art, origin_df, I_s, K_s, b_s, h_s, t_s)
    dfl_pred = predict_sku_origin_outputs(dfl_art, origin_df, I_s, K_s, b_s, h_s, t_s)

    assert not np.allclose(sup_pred["c_hat_flat"], dfl_pred["c_hat_flat"])
