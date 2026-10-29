"""Genuine Decision-Focused Learning (`dfl_pto` / `dfl_dro`) using PyEPO `SPOPlus`
on the fixed-feasible marginal-unit allocation problem (`AGENTS.md` §6, §7, §14).
"""

from __future__ import annotations

import math
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
)
from inventory_di.models.supervised import (
    SKUForecastArtifact,
    evaluate_pmf_forecast_diagnostics,
    extract_sku_split_tensors,
)
from inventory_di.optimization.pyepo_fixed import (
    build_joint_multi_sku_spoplus_layer,
    build_spoplus_layer,
    precompute_joint_multi_sku_oracle_cache,
    precompute_sku_oracle_cache,
)


def verify_one_batch_spoplus_vs_nll_update(
    x_batch: torch.Tensor,
    d_batch: torch.Tensor,
    true_costs_batch: torch.Tensor,
    true_sols_batch: torch.Tensor,
    true_objs_batch: torch.Tensor,
    spoplus_layer: Any,
    num_stores: int,
    support_h: int,
    hidden_dim: int,
    I_s: np.ndarray,
    K_s: np.ndarray,
    b_s: np.ndarray,
    h_s: np.ndarray,
    t_s: np.ndarray,
    lambda_dfl: float,
    seed: int = 42,
) -> dict[str, float]:
    """Run a mandatory 1-batch backward smoke test verifying:
    1. Pure SPO+ gradients w.r.t. network weights are finite and non-zero.
    2. Joint `NLL + lambda_DFL * SPOPlus` produces measurably different parameter updates
       from `NLL`-only (`lambda_DFL = 0`) starting from identical initial weights (`AGENTS.md` §6, §14).
    """
    # 1. NLL-only step
    torch.manual_seed(seed)
    model_nll = DiscreteDemandPMFNet(num_stores, len(FEATURE_COLUMNS), support_h, hidden_dim)
    opt_nll = torch.optim.SGD(model_nll.parameters(), lr=0.05)
    opt_nll.zero_grad()
    logits_nll = model_nll.forward_logits(x_batch)
    loss_nll = compute_discrete_nll_loss(logits_nll, d_batch, support_h)
    loss_nll.backward()
    grad_norm_nll = float(
        math.sqrt(
            sum(float(p.grad.norm().item() ** 2) for p in model_nll.parameters() if p.grad is not None)
        )
    )
    opt_nll.step()

    # 2. Pure SPO+ gradient check
    torch.manual_seed(seed)
    model_spo = DiscreteDemandPMFNet(num_stores, len(FEATURE_COLUMNS), support_h, hidden_dim)
    model_spo.zero_grad()
    pmf_spo = model_spo.predict_pmf(x_batch)
    _, c_hat_spo = compute_survival_and_marginal_costs_torch(pmf_spo, I_s, K_s, b_s, h_s, t_s)
    loss_pure_spo = spoplus_layer(c_hat_spo, true_costs_batch, true_sols_batch, true_objs_batch)
    loss_pure_spo.backward()
    grad_norm_pure_spo = float(
        math.sqrt(
            sum(float(p.grad.norm().item() ** 2) for p in model_spo.parameters() if p.grad is not None)
        )
    )

    # 3. Joint NLL + lambda_DFL * SPOPlus step from identical initial weights
    torch.manual_seed(seed)
    model_dfl = DiscreteDemandPMFNet(num_stores, len(FEATURE_COLUMNS), support_h, hidden_dim)
    opt_dfl = torch.optim.SGD(model_dfl.parameters(), lr=0.05)
    opt_dfl.zero_grad()
    logits_dfl = model_dfl.forward_logits(x_batch)
    nll_part = compute_discrete_nll_loss(logits_dfl, d_batch, support_h)
    pmf_dfl = model_dfl.predict_pmf(x_batch)
    _, c_hat_dfl = compute_survival_and_marginal_costs_torch(pmf_dfl, I_s, K_s, b_s, h_s, t_s)
    spo_part = spoplus_layer(c_hat_dfl, true_costs_batch, true_sols_batch, true_objs_batch)
    total_dfl_loss = nll_part + float(lambda_dfl) * spo_part
    total_dfl_loss.backward()
    grad_norm_dfl = float(
        math.sqrt(
            sum(float(p.grad.norm().item() ** 2) for p in model_dfl.parameters() if p.grad is not None)
        )
    )
    opt_dfl.step()

    param_diff_sq = 0.0
    for p1, p2 in zip(model_nll.parameters(), model_dfl.parameters()):
        param_diff_sq += float(torch.sum((p1.detach() - p2.detach()) ** 2).item())
    param_diff_l2 = math.sqrt(param_diff_sq)

    if not np.isfinite(grad_norm_pure_spo) or grad_norm_pure_spo <= 0.0:
        raise RuntimeError(
            f"SPO+ backward smoke test failed: grad_norm_pure_spo={grad_norm_pure_spo}"
        )
    if float(lambda_dfl) > 0.0 and param_diff_l2 <= 1e-8:
        raise RuntimeError(
            f"DFL parameter update did not diverge from NLL-only: param_diff_l2={param_diff_l2}"
        )

    return {
        "grad_norm_nll_only": grad_norm_nll,
        "grad_norm_pure_spoplus": grad_norm_pure_spo,
        "grad_norm_dfl": grad_norm_dfl,
        "param_diff_l2_vs_nll": param_diff_l2,
        "spoplus_initial_loss": float(loss_pure_spo.item()),
    }


def evaluate_fixed_sku_regret_on_split(
    model: DiscreteDemandPMFNet,
    opt_model: Any,
    x_split: torch.Tensor,
    oracle_cache: dict[str, np.ndarray],
    I_s: np.ndarray,
    K_s: np.ndarray,
    b_s: np.ndarray,
    h_s: np.ndarray,
    t_s: np.ndarray,
) -> float:
    """Compute mean decision regret on the fixed-feasible per-SKU problem for validation monitoring."""
    model.eval()
    with torch.no_grad():
        pmf = model.predict_pmf(x_split)
        _, c_hat = compute_survival_and_marginal_costs_torch(pmf, I_s, K_s, b_s, h_s, t_s)
        c_hat_np = c_hat.cpu().numpy()

    true_costs = oracle_cache["true_costs"]
    true_objs = oracle_cache["true_objs"]
    regrets: list[float] = []
    for idx in range(len(c_hat_np)):
        opt_model.setObj(c_hat_np[idx])
        sol_pred, _ = opt_model.solve()
        realized_incremental = float(np.dot(true_costs[idx], sol_pred))
        regret = max(0.0, realized_incremental - float(true_objs[idx]))
        regrets.append(regret)
    return float(np.mean(regrets))


def train_dfl_sku_model(
    weekly_df: pd.DataFrame,
    sku_id: str,
    stores: list[str],
    I_s: np.ndarray,
    K_s: np.ndarray,
    W_p: int,
    b_s: np.ndarray,
    h_s: np.ndarray,
    t_s: np.ndarray,
    support_h: int,
    tail_mean_excess: float,
    hidden_dim: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    lambda_dfl: float,
    spoplus_processes: int,
    seed: int,
) -> SKUForecastArtifact:
    """Train genuine DFL demand-distribution model with `NLL + lambda_DFL * SPOPlus` (`AGENTS.md` §6)."""
    _, train_x, train_d, feat_mean, feat_std = extract_sku_split_tensors(
        weekly_df, sku_id, stores, "train"
    )
    _, val_x, val_d, _, _ = extract_sku_split_tensors(
        weekly_df, sku_id, stores, "val", feat_mean=feat_mean, feat_std=feat_std
    )

    opt_model, spoplus_layer = build_spoplus_layer(
        store_caps=K_s,
        warehouse_supply=W_p,
        processes=spoplus_processes,
        solver="appsi_highs",
    )

    train_oracle = precompute_sku_oracle_cache(
        opt_model,
        demands_matrix=train_d.cpu().numpy(),
        I_s=I_s,
        K_s=K_s,
        b_s=b_s,
        h_s=h_s,
        t_s=t_s,
        split_name="train",
    )
    val_oracle = precompute_sku_oracle_cache(
        opt_model,
        demands_matrix=val_d.cpu().numpy(),
        I_s=I_s,
        K_s=K_s,
        b_s=b_s,
        h_s=h_s,
        t_s=t_s,
        split_name="val",
    )

    train_tc = torch.tensor(train_oracle["true_costs"], dtype=torch.float32)
    train_ts = torch.tensor(train_oracle["true_sols"], dtype=torch.float32)
    train_to = torch.tensor(train_oracle["true_objs"], dtype=torch.float32)

    grad_check = verify_one_batch_spoplus_vs_nll_update(
        x_batch=train_x[: min(4, len(train_x))],
        d_batch=train_d[: min(4, len(train_d))],
        true_costs_batch=train_tc[: min(4, len(train_tc))],
        true_sols_batch=train_ts[: min(4, len(train_ts))],
        true_objs_batch=train_to[: min(4, len(train_to))],
        spoplus_layer=spoplus_layer,
        num_stores=len(stores),
        support_h=support_h,
        hidden_dim=hidden_dim,
        I_s=I_s,
        K_s=K_s,
        b_s=b_s,
        h_s=h_s,
        t_s=t_s,
        lambda_dfl=max(lambda_dfl, 0.5),
        seed=seed,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DiscreteDemandPMFNet(
        num_stores=len(stores),
        in_features=len(FEATURE_COLUMNS),
        support_h=support_h,
        hidden_dim=hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: list[dict[str, float]] = []
    best_val_score = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    T_train = len(train_x)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(T_train)
        epoch_nll = 0.0
        epoch_spo = 0.0
        num_batches = 0

        for start in range(0, T_train, batch_size):
            idx = perm[start : start + batch_size]
            bx = train_x[idx]
            bd = train_d[idx]
            btc = train_tc[idx]
            bts = train_ts[idx]
            bto = train_to[idx]

            optimizer.zero_grad()
            logits = model.forward_logits(bx)
            nll_loss = compute_discrete_nll_loss(logits, bd, support_h)
            pmf = model.predict_pmf(bx)
            _, c_hat = compute_survival_and_marginal_costs_torch(pmf, I_s, K_s, b_s, h_s, t_s)

            if lambda_dfl > 0.0:
                spo_loss = spoplus_layer(c_hat, btc, bts, bto)
                # Normalize SPO+ by number of cost terms so scale aligns smoothly with NLL
                spo_scaled = spo_loss / float(max(1, opt_model.num_cost))
                total_loss = nll_loss + float(lambda_dfl) * spo_scaled
                spo_val = float(spo_scaled.item())
            else:
                total_loss = nll_loss
                spo_val = 0.0

            total_loss.backward()
            optimizer.step()

            epoch_nll += float(nll_loss.item())
            epoch_spo += spo_val
            num_batches += 1

        epoch_nll /= max(1, num_batches)
        epoch_spo /= max(1, num_batches)

        val_diag = evaluate_pmf_forecast_diagnostics(model, val_x, val_d, tail_mean_excess)
        val_regret = evaluate_fixed_sku_regret_on_split(
            model, opt_model, val_x, val_oracle, I_s, K_s, b_s, h_s, t_s
        )
        # Validation stopping combines NLL and decision regret on validation split
        val_composite = val_diag["nll"] + 0.15 * val_regret
        if val_composite < best_val_score:
            best_val_score = val_composite
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        history.append(
            {
                "epoch": float(epoch),
                "train_nll": epoch_nll,
                "train_spoplus": epoch_spo,
                "train_total_loss": epoch_nll + float(lambda_dfl) * epoch_spo,
                "val_nll": val_diag["nll"],
                "val_mae": val_diag["mae"],
                "val_wape": val_diag["wape"],
                "val_decision_regret_mu": val_regret,
            }
        )

    model.load_state_dict(best_state)
    cal_info = calibrate_temperature_on_validation(model, val_x, val_d)
    val_final = evaluate_pmf_forecast_diagnostics(model, val_x, val_d, tail_mean_excess)
    val_final_regret = evaluate_fixed_sku_regret_on_split(
        model, opt_model, val_x, val_oracle, I_s, K_s, b_s, h_s, t_s
    )
    cal_info.update(val_final)
    cal_info["val_decision_regret_mu"] = val_final_regret

    return SKUForecastArtifact(
        sku_id=sku_id,
        stores=list(stores),
        model_type="dfl",
        lambda_dfl=float(lambda_dfl),
        support_h=support_h,
        tail_mean_excess=tail_mean_excess,
        feat_mean=feat_mean,
        feat_std=feat_std,
        model=model,
        history=history,
        calibration_info=cal_info,
        gradient_check=grad_check,
    )


def train_joint_multi_sku_dfl_models(
    weekly_df: pd.DataFrame,
    skus: list[str],
    stores: list[str],
    I_sp: np.ndarray,
    K_sp: np.ndarray,
    W_p: np.ndarray,
    C_s: np.ndarray,
    b_sp: np.ndarray,
    h_sp: np.ndarray,
    t_sp: np.ndarray,
    support_h: int,
    tail_mean_excess: float,
    hidden_dim: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    lambda_dfl: float,
    spoplus_processes: int,
    seed: int,
) -> dict[str, SKUForecastArtifact]:
    """Train all SKU DiscreteDemandPMFNets jointly with PyEPO `SPOPlus` over the exact
    multi-SKU store-capacity-constrained polytope (`JointMultiSKUMarginalAllocationModel`) (Priority 5).
    """
    P = len(skus)
    S = len(stores)

    train_x_list: list[torch.Tensor] = []
    train_d_list: list[torch.Tensor] = []
    val_x_list: list[torch.Tensor] = []
    val_d_list: list[torch.Tensor] = []
    feat_means: list[np.ndarray] = []
    feat_stds: list[np.ndarray] = []

    for p_idx, sku in enumerate(skus):
        _, tx, td, fm, fs = extract_sku_split_tensors(weekly_df, sku, stores, "train")
        _, vx, vd, _, _ = extract_sku_split_tensors(
            weekly_df, sku, stores, "val", feat_mean=fm, feat_std=fs
        )
        train_x_list.append(tx)
        train_d_list.append(td)
        val_x_list.append(vx)
        val_d_list.append(vd)
        feat_means.append(fm)
        feat_stds.append(fs)

    # Shape (T, S, P) for joint oracle precomputation
    train_d_tsp = np.stack([td.cpu().numpy() for td in train_d_list], axis=-1)
    val_d_tsp = np.stack([vd.cpu().numpy() for vd in val_d_list], axis=-1)

    joint_opt, joint_spoplus = build_joint_multi_sku_spoplus_layer(
        K_sp=K_sp,
        W_p=W_p,
        I_sp=I_sp,
        C_s=C_s,
        processes=spoplus_processes,
        solver="appsi_highs",
    )

    train_oracle = precompute_joint_multi_sku_oracle_cache(
        joint_opt, train_d_tsp, I_sp, K_sp, b_sp, h_sp, t_sp, split_name="train"
    )
    val_oracle = precompute_joint_multi_sku_oracle_cache(
        joint_opt, val_d_tsp, I_sp, K_sp, b_sp, h_sp, t_sp, split_name="val"
    )

    train_tc = torch.tensor(train_oracle["true_costs"], dtype=torch.float32)
    train_ts = torch.tensor(train_oracle["true_sols"], dtype=torch.float32)
    train_to = torch.tensor(train_oracle["true_objs"], dtype=torch.float32)

    models: list[DiscreteDemandPMFNet] = []
    for p_idx in range(P):
        sku_seed = seed + 100 * p_idx
        torch.manual_seed(sku_seed)
        np.random.seed(sku_seed)
        models.append(
            DiscreteDemandPMFNet(
                num_stores=S,
                in_features=len(FEATURE_COLUMNS),
                support_h=support_h,
                hidden_dim=hidden_dim,
            )
        )

    # 1-batch joint SPO+ backward gradient check across all SKU networks
    b_sz = min(4, len(train_x_list[0]))
    for m in models:
        m.zero_grad()
    c_hat_parts_0 = []
    for p_idx in range(P):
        pmf_0 = models[p_idx].predict_pmf(train_x_list[p_idx][:b_sz])
        _, ch_0 = compute_survival_and_marginal_costs_torch(
            pmf_0, I_sp[:, p_idx], K_sp[:, p_idx], b_sp[:, p_idx], h_sp[:, p_idx], t_sp[:, p_idx]
        )
        c_hat_parts_0.append(ch_0)
    c_hat_joint_0 = torch.cat(c_hat_parts_0, dim=-1)
    loss_spo_0 = joint_spoplus(
        c_hat_joint_0, train_tc[:b_sz], train_ts[:b_sz], train_to[:b_sz]
    )
    loss_spo_0.backward()
    joint_grad_norms = [
        float(
            math.sqrt(
                sum(float(p.grad.norm().item() ** 2) for p in m.parameters() if p.grad is not None)
            )
        )
        for m in models
    ]

    if not np.isfinite(joint_grad_norms).all() or sum(joint_grad_norms) <= 0:
        raise RuntimeError("Joint SPO+ gradient participation check failed")
    # A real matched SGD update smoke: NLL versus NLL+SPO+, same initial weights.
    spo_gradients = [[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                      for p in m.parameters()] for m in models]
    smoke_differences = []
    import copy
    for p_idx, m in enumerate(models):
        left, right = copy.deepcopy(m), copy.deepcopy(m)
        for candidate, add_spo in [(left, False), (right, True)]:
            candidate.zero_grad()
            loss = compute_discrete_nll_loss(candidate.forward_logits(train_x_list[p_idx][:b_sz]),
                                             train_d_list[p_idx][:b_sz], support_h)
            loss.backward()
            if add_spo:
                for param, gradient in zip(candidate.parameters(), spo_gradients[p_idx]):
                    param.grad += gradient / max(1, joint_opt.num_cost)
            torch.optim.SGD(candidate.parameters(), lr=learning_rate).step()
        smoke_differences.append(float(torch.linalg.vector_norm(torch.cat([
            (a-b).detach().flatten() for a,b in zip(left.parameters(), right.parameters())])).item()))
    if sum(smoke_differences) <= 0:
        raise RuntimeError("Joint SPO+ did not change matched parameter updates")

    # Re-initialize with exact seeds for clean training
    models = []
    all_params = []
    for p_idx in range(P):
        sku_seed = seed + 100 * p_idx
        torch.manual_seed(sku_seed)
        np.random.seed(sku_seed)
        m = DiscreteDemandPMFNet(
            num_stores=S,
            in_features=len(FEATURE_COLUMNS),
            support_h=support_h,
            hidden_dim=hidden_dim,
        )
        models.append(m)
        all_params.extend(list(m.parameters()))

    optimizer = torch.optim.Adam(all_params, lr=learning_rate)
    T_train = len(train_x_list[0])
    histories: list[list[dict[str, float]]] = [[] for _ in range(P)]
    best_val_score = float("inf")
    best_states = [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in models]

    best_epoch = 0
    for epoch in range(1, epochs + 1):
        for m in models:
            m.train()
        perm = torch.randperm(T_train)
        epoch_nll_per_sku = np.zeros(P, dtype=float)
        epoch_spo = 0.0
        num_batches = 0

        for start in range(0, T_train, batch_size):
            idx = perm[start : start + batch_size]
            btc = train_tc[idx]
            bts = train_ts[idx]
            bto = train_to[idx]

            optimizer.zero_grad()
            nll_losses = []
            c_hat_parts = []
            for p_idx in range(P):
                bx = train_x_list[p_idx][idx]
                bd = train_d_list[p_idx][idx]
                logits = models[p_idx].forward_logits(bx)
                nll_p = compute_discrete_nll_loss(logits, bd, support_h)
                nll_losses.append(nll_p)
                epoch_nll_per_sku[p_idx] += float(nll_p.item())

                pmf_p = models[p_idx].predict_pmf(bx)
                _, ch_p = compute_survival_and_marginal_costs_torch(
                    pmf_p,
                    I_sp[:, p_idx],
                    K_sp[:, p_idx],
                    b_sp[:, p_idx],
                    h_sp[:, p_idx],
                    t_sp[:, p_idx],
                )
                c_hat_parts.append(ch_p)

            mean_nll = torch.stack(nll_losses).mean()
            if lambda_dfl > 0.0:
                c_hat_joint = torch.cat(c_hat_parts, dim=-1)
                spo_loss = joint_spoplus(c_hat_joint, btc, bts, bto)
                spo_scaled = spo_loss / float(max(1, joint_opt.num_cost))
                total_loss = mean_nll + float(lambda_dfl) * spo_scaled
                spo_val = float(spo_scaled.item())
            else:
                total_loss = mean_nll
                spo_val = 0.0

            total_loss.backward()
            optimizer.step()
            epoch_spo += spo_val
            num_batches += 1

        epoch_nll_per_sku /= max(1, num_batches)
        epoch_spo /= max(1, num_batches)

        # Evaluate joint validation regret and per-SKU NLL
        for m in models:
            m.eval()
        with torch.no_grad():
            val_c_parts = []
            for p_idx in range(P):
                v_pmf = models[p_idx].predict_pmf(val_x_list[p_idx])
                _, v_ch = compute_survival_and_marginal_costs_torch(
                    v_pmf,
                    I_sp[:, p_idx],
                    K_sp[:, p_idx],
                    b_sp[:, p_idx],
                    h_sp[:, p_idx],
                    t_sp[:, p_idx],
                )
                val_c_parts.append(v_ch)
            val_c_joint = torch.cat(val_c_parts, dim=-1).cpu().numpy()

        val_tc = val_oracle["true_costs"]
        val_to = val_oracle["true_objs"]
        joint_regrets = []
        for t_i in range(len(val_c_joint)):
            joint_opt.setObj(val_c_joint[t_i])
            sol_pred, _ = joint_opt.solve()
            realized_inc = float(np.dot(val_tc[t_i], sol_pred))
            regret = realized_inc - float(val_to[t_i])
            if regret < -1e-5:
                raise AssertionError("Negative joint validation regret")
            joint_regrets.append(regret)
        mean_joint_val_regret = float(np.mean(joint_regrets)) / float(P)

        val_diags = [
            evaluate_pmf_forecast_diagnostics(models[p_idx], val_x_list[p_idx], val_d_list[p_idx], tail_mean_excess)
            for p_idx in range(P)
        ]
        mean_val_nll = float(np.mean([d["nll"] for d in val_diags]))
        val_composite = mean_val_nll + 0.15 * mean_joint_val_regret
        if val_composite < best_val_score:
            best_epoch = epoch
            best_val_score = val_composite
            best_states = [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in models]

        for p_idx in range(P):
            histories[p_idx].append(
                {
                    "epoch": float(epoch),
                    "train_nll": float(epoch_nll_per_sku[p_idx]),
                    "train_spoplus": epoch_spo,
                    "train_total_loss": float(epoch_nll_per_sku[p_idx]) + float(lambda_dfl) * epoch_spo,
                    "val_nll": val_diags[p_idx]["nll"],
                    "val_mae": val_diags[p_idx]["mae"],
                    "val_wape": val_diags[p_idx]["wape"],
                    "val_decision_regret_mu": mean_joint_val_regret,
                }
            )

    calibration_infos = []
    for p_idx in range(P):
        models[p_idx].load_state_dict(best_states[p_idx])
        calibration_infos.append(calibrate_temperature_on_validation(models[p_idx], val_x_list[p_idx], val_d_list[p_idx]))
    with torch.no_grad():
        final_costs = torch.cat([
            compute_survival_and_marginal_costs_torch(models[p].predict_pmf(val_x_list[p]),
                I_sp[:,p], K_sp[:,p], b_sp[:,p], h_sp[:,p], t_sp[:,p], tail_mean_excess)[1]
            for p in range(P)], dim=-1).numpy()
    final_regrets = []
    for t_idx, costs in enumerate(final_costs):
        joint_opt.setObj(costs)
        solution, _ = joint_opt.solve()
        final_regrets.append(float(np.dot(val_oracle["true_costs"][t_idx], solution)-val_oracle["true_objs"][t_idx]))
    if min(final_regrets) < -1e-5:
        raise AssertionError("Negative calibrated validation regret")
    final_regret = float(np.mean(final_regrets)) / P
    artifacts: dict[str, SKUForecastArtifact] = {}
    for p_idx, sku in enumerate(skus):
        cal_info = calibration_infos[p_idx]
        val_final = evaluate_pmf_forecast_diagnostics(
            models[p_idx], val_x_list[p_idx], val_d_list[p_idx], tail_mean_excess
        )
        cal_info.update(val_final)
        cal_info["val_decision_regret_mu"] = final_regret
        cal_info["best_epoch"] = best_epoch
        cal_info["selection_rule"] = "mean validation NLL + 0.15 * joint regret per SKU"
        cal_info["epochs_budget"] = epochs
        cal_info["batch_size"] = batch_size
        cal_info["joint_multi_sku_dfl_aligned"] = True

        artifacts[sku] = SKUForecastArtifact(
            sku_id=sku,
            stores=list(stores),
            model_type="dfl" if lambda_dfl > 0 else "supervised",
            lambda_dfl=float(lambda_dfl),
            support_h=support_h,
            tail_mean_excess=tail_mean_excess,
            feat_mean=feat_means[p_idx],
            feat_std=feat_stds[p_idx],
            model=models[p_idx],
            history=histories[p_idx],
            calibration_info=cal_info,
            gradient_check={
                "grad_norm_pure_spoplus": joint_grad_norms[p_idx],
                "joint_multi_sku_polytope": 1.0,
                "param_diff_l2_vs_nll": smoke_differences[p_idx],
            },
        )
    return artifacts
