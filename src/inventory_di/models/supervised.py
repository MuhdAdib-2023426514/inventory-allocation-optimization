"""Supervised NLL-only probabilistic forecasting baseline (`supervised_pto` / `supervised_dro`) (`AGENTS.md` §6, §8)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from inventory_di.data.features import FEATURE_COLUMNS
from inventory_di.models.distributions import (
    DiscreteDemandPMFNet,
    calibrate_temperature_on_validation,
    compute_discrete_nll_loss,
    compute_survival_and_marginal_costs_torch,
    pmf_moments_and_quantiles,
)


@dataclass
class SKUForecastArtifact:
    """Trained model, feature standardization stats (from train only), and training logs for one SKU."""

    sku_id: str
    stores: list[str]
    model_type: str  # "supervised" or "dfl"
    lambda_dfl: float
    support_h: int
    tail_mean_excess: float
    feat_mean: np.ndarray  # shape (F,) fitted on train only
    feat_std: np.ndarray  # shape (F,) fitted on train only
    model: DiscreteDemandPMFNet
    history: list[dict[str, float]]
    calibration_info: dict[str, Any]
    gradient_check: dict[str, float]


def extract_sku_split_tensors(
    weekly_df: pd.DataFrame,
    sku_id: str,
    stores: list[str],
    split_name: str,
    feat_mean: np.ndarray | None = None,
    feat_std: np.ndarray | None = None,
) -> tuple[list[str], torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Extract aligned `(origins, X_tensor [T, S, F], D_tensor [T, S], feat_mean, feat_std)` for a SKU and split."""
    sub = weekly_df[(weekly_df["sku_id"] == sku_id) & (weekly_df["split"] == split_name)].copy()
    origins = sorted(sub["forecast_origin"].unique())
    S = len(stores)
    F_dim = len(FEATURE_COLUMNS)
    T = len(origins)

    X_np = np.zeros((T, S, F_dim), dtype=np.float32)
    D_np = np.zeros((T, S), dtype=np.int64)

    origin_map = {o: i for i, o in enumerate(origins)}
    store_map = {s: i for i, s in enumerate(stores)}

    for row in sub.itertuples(index=False):
        t_i = origin_map[str(row.forecast_origin)]
        s_i = store_map[str(row.store_id)]
        D_np[t_i, s_i] = int(row.observed_units_proxy)
        for f_i, col in enumerate(FEATURE_COLUMNS):
            X_np[t_i, s_i, f_i] = float(getattr(row, col))

    if feat_mean is None or feat_std is None:
        if split_name != "train":
            raise ValueError("Feature mean/std must only be fitted on split == 'train'!")
        flat = X_np.reshape(-1, F_dim)
        feat_mean = flat.mean(axis=0).astype(np.float32)
        feat_std = np.maximum(flat.std(axis=0), 1e-5).astype(np.float32)

    X_norm = (X_np - feat_mean.reshape(1, 1, F_dim)) / feat_std.reshape(1, 1, F_dim)
    return (
        origins,
        torch.tensor(X_norm, dtype=torch.float32),
        torch.tensor(D_np, dtype=torch.long),
        feat_mean,
        feat_std,
    )


def evaluate_pmf_forecast_diagnostics(
    model: DiscreteDemandPMFNet,
    X_tensor: torch.Tensor,
    D_tensor: torch.Tensor,
    tail_mean_excess: float,
) -> dict[str, float]:
    """Compute NLL, MAE of implied expectation, zero-safe WAPE, P10-P90 coverage, and overflow rate."""
    model.eval()
    with torch.no_grad():
        logits = model.forward_logits(X_tensor)
        temp = float(model.temperature.item())
        nll = float(compute_discrete_nll_loss(logits, D_tensor, model.support_h, temperature=temp).item())
        pmf = model.predict_pmf(X_tensor).cpu().numpy()

    D_np = D_tensor.cpu().numpy().astype(float)
    T, S, _ = pmf.shape
    exp_mat = np.zeros((T, S), dtype=float)
    p10_mat = np.zeros((T, S), dtype=float)
    p90_mat = np.zeros((T, S), dtype=float)
    overflow_prob_mat = np.zeros((T, S), dtype=float)

    for t_i in range(T):
        stats = pmf_moments_and_quantiles(pmf[t_i], tail_mean_excess=tail_mean_excess)
        exp_mat[t_i] = stats["expected"]
        p10_mat[t_i] = stats["p10"]
        p90_mat[t_i] = stats["p90"]
        overflow_prob_mat[t_i] = stats["overflow_prob"]

    abs_err = np.abs(exp_mat - D_np)
    mae = float(np.mean(abs_err))
    denom = max(1.0, float(np.sum(np.abs(D_np))))
    wape = float(np.sum(abs_err) / denom)
    coverage_80 = float(np.mean((D_np >= p10_mat) & (D_np <= p90_mat)))
    realized_overflow_rate = float(np.mean(D_np > model.support_h))
    predicted_overflow_prob = float(np.mean(overflow_prob_mat))

    return {
        "nll": nll,
        "mae": mae,
        "wape": wape,
        "coverage_p10_p90": coverage_80,
        "realized_overflow_rate": realized_overflow_rate,
        "predicted_overflow_prob": predicted_overflow_prob,
    }


def train_supervised_sku_model(
    weekly_df: pd.DataFrame,
    sku_id: str,
    stores: list[str],
    support_h: int,
    tail_mean_excess: float,
    hidden_dim: int,
    learning_rate: float,
    epochs: int,
    seed: int,
) -> SKUForecastArtifact:
    """Train the NLL-only supervised discrete PMF model for one SKU and calibrate on `val`."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    _, train_x, train_d, feat_mean, feat_std = extract_sku_split_tensors(
        weekly_df, sku_id, stores, "train"
    )
    _, val_x, val_d, _, _ = extract_sku_split_tensors(
        weekly_df, sku_id, stores, "val", feat_mean=feat_mean, feat_std=feat_std
    )

    model = DiscreteDemandPMFNet(
        num_stores=len(stores),
        in_features=len(FEATURE_COLUMNS),
        support_h=support_h,
        hidden_dim=hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # One-batch gradient check before training
    model.train()
    optimizer.zero_grad()
    logits_0 = model.forward_logits(train_x[:2])
    loss_0 = compute_discrete_nll_loss(logits_0, train_d[:2], support_h)
    loss_0.backward()
    grad_norm_0 = float(
        math.sqrt(sum(float(p.grad.norm().item() ** 2) for p in model.parameters() if p.grad is not None))
    )

    # Re-seed so initial weights match DFL starting point identically
    torch.manual_seed(seed)
    model = DiscreteDemandPMFNet(
        num_stores=len(stores),
        in_features=len(FEATURE_COLUMNS),
        support_h=support_h,
        hidden_dim=hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: list[dict[str, float]] = []
    best_val_nll = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model.forward_logits(train_x)
        train_nll = compute_discrete_nll_loss(logits, train_d, support_h)
        train_nll.backward()
        optimizer.step()

        val_diag = evaluate_pmf_forecast_diagnostics(model, val_x, val_d, tail_mean_excess)
        if val_diag["nll"] < best_val_nll:
            best_val_nll = val_diag["nll"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        history.append(
            {
                "epoch": float(epoch),
                "train_nll": float(train_nll.item()),
                "train_spoplus": 0.0,
                "train_total_loss": float(train_nll.item()),
                "val_nll": val_diag["nll"],
                "val_mae": val_diag["mae"],
                "val_wape": val_diag["wape"],
            }
        )

    model.load_state_dict(best_state)
    cal_info = calibrate_temperature_on_validation(model, val_x, val_d)
    val_final = evaluate_pmf_forecast_diagnostics(model, val_x, val_d, tail_mean_excess)
    cal_info.update(val_final)

    return SKUForecastArtifact(
        sku_id=sku_id,
        stores=list(stores),
        model_type="supervised",
        lambda_dfl=0.0,
        support_h=support_h,
        tail_mean_excess=tail_mean_excess,
        feat_mean=feat_mean,
        feat_std=feat_std,
        model=model,
        history=history,
        calibration_info=cal_info,
        gradient_check={"grad_norm_nll_only": grad_norm_0},
    )


def predict_sku_origin_outputs(
    artifact: SKUForecastArtifact,
    origin_df: pd.DataFrame,
    I_s: np.ndarray,
    K_s: np.ndarray,
    b_s: np.ndarray,
    h_s: np.ndarray,
    t_s: np.ndarray,
) -> dict[str, np.ndarray]:
    """Predict calibrated PMF `(S, H + 2)`, survival probabilities `q_flat`, and marginal costs `c_hat_flat`
    for a single forecast origin and SKU.
    """
    sub = origin_df[origin_df["sku_id"] == artifact.sku_id].copy()
    S = len(artifact.stores)
    F_dim = len(FEATURE_COLUMNS)
    X_np = np.zeros((1, S, F_dim), dtype=np.float32)
    store_map = {s: i for i, s in enumerate(artifact.stores)}

    for row in sub.itertuples(index=False):
        s_i = store_map[str(row.store_id)]
        for f_i, col in enumerate(FEATURE_COLUMNS):
            X_np[0, s_i, f_i] = float(getattr(row, col))

    X_norm = (X_np - artifact.feat_mean.reshape(1, 1, F_dim)) / artifact.feat_std.reshape(1, 1, F_dim)
    x_t = torch.tensor(X_norm, dtype=torch.float32)

    artifact.model.eval()
    with torch.no_grad():
        pmf_t = artifact.model.predict_pmf(x_t)  # (1, S, H + 2)
        q_t, c_hat_t = compute_survival_and_marginal_costs_torch(
            pmf_t, I_s=I_s, K_s=K_s, b_s=b_s, h_s=h_s, t_s=t_s,
            tail_mean_excess=artifact.tail_mean_excess
        )

    pmf_np = pmf_t[0].cpu().numpy()
    moments = pmf_moments_and_quantiles(pmf_np, tail_mean_excess=artifact.tail_mean_excess)
    return {
        "pmf": pmf_np,
        "q_flat": q_t[0].cpu().numpy(),
        "c_hat_flat": c_hat_t[0].cpu().numpy(),
        "expected_demand": moments["expected"],
        "p10_demand": moments["p10"],
        "p50_demand": moments["p50"],
        "p90_demand": moments["p90"],
        "overflow_prob": moments["overflow_prob"],
    }
