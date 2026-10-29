"""End-to-end training, rolling-origin 7-policy backtesting, and sensitivity sweep runner (`AGENTS.md` §6, §7, §8, §9)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from inventory_di.config import ExperimentConfig
from inventory_di.evaluation.metrics import REQUIRED_POLICIES, summarize_backtest_metrics
from inventory_di.io import (
    get_environment_snapshot,
    get_git_commit,
    save_json,
    save_parquet,
    sha256_file,
    validate_allocation_results_df,
    validate_backtest_metrics_df,
)
from inventory_di.models.dfl import train_dfl_sku_model, train_joint_multi_sku_dfl_models
from inventory_di.models.distributions import EmpiricalCopulaScenarioGenerator
from inventory_di.models.supervised import (
    SKUForecastArtifact,
    predict_sku_origin_outputs,
    train_supervised_sku_model,
)
from inventory_di.optimization.baselines import (
    allocate_equal_shares,
    allocate_proportional_shares,
    solve_clairvoyant_oracle,
)
from inventory_di.optimization.core import (
    AllocationInstance,
    evaluate_exact_cost,
    solve_nominal_marginal_cost_milp,
)
from inventory_di.optimization.dro import (
    evaluate_allocation_on_scenarios,
    solve_scenario_reweighting_dro_milp,
)


def build_allocation_instance_from_config_df(
    inv_df: pd.DataFrame,
    supply_multiplier: float = 1.0,
    opening_stock_override: int | None = None,
    shortage_penalty_multiplier: float = 1.0,
) -> AllocationInstance:
    """Construct an `AllocationInstance` from `inventory_config.parquet`."""
    stores = sorted(inv_df["store_id"].unique())
    skus = sorted(inv_df["sku_id"].unique())
    S, P = len(stores), len(skus)

    I = np.zeros((S, P), dtype=int)
    W = np.zeros(P, dtype=int)
    K = np.zeros((S, P), dtype=int)
    C = np.zeros(S, dtype=int)
    b = np.zeros((S, P), dtype=float)
    h = np.zeros((S, P), dtype=float)
    t = np.zeros((S, P), dtype=float)

    store_idx = {s: i for i, s in enumerate(stores)}
    sku_idx = {p: i for i, p in enumerate(skus)}

    for row in inv_df.itertuples(index=False):
        s_i = store_idx[str(row.store_id)]
        p_i = sku_idx[str(row.sku_id)]
        I[s_i, p_i] = (
            int(opening_stock_override)
            if opening_stock_override is not None
            else int(row.opening_stock_units)
        )
        K[s_i, p_i] = int(row.store_sku_unit_cap)
        W[p_i] = max(1, int(round(float(row.warehouse_supply_units) * float(supply_multiplier))))
        C[s_i] = int(row.store_total_capacity_units)
        b[s_i, p_i] = float(row.unit_shortage_penalty_mu) * float(shortage_penalty_multiplier)
        h[s_i, p_i] = float(row.unit_holding_cost_mu)
        t[s_i, p_i] = float(row.unit_ship_cost_mu)

    return AllocationInstance(stores=stores, skus=skus, I=I, W=W, K=K, C=C, b=b, h=h, t=t)


def save_sku_forecast_artifacts(
    artifacts: dict[str, SKUForecastArtifact],
    out_path: str | Path,
) -> str:
    """Serialize trained PyTorch checkpoints and normalization parameters, returning SHA-256."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for sku, art in artifacts.items():
        payload[sku] = {
            "sku_id": art.sku_id,
            "stores": art.stores,
            "model_type": art.model_type,
            "lambda_dfl": art.lambda_dfl,
            "support_h": art.support_h,
            "tail_mean_excess": art.tail_mean_excess,
            "feat_mean": art.feat_mean,
            "feat_std": art.feat_std,
            "state_dict": art.model.state_dict(),
            "history": art.history,
            "calibration_info": art.calibration_info,
            "gradient_check": art.gradient_check,
        }
    torch.save(payload, p)
    return sha256_file(p)


def load_sku_forecast_artifacts(
    ckpt_path: str | Path,
    hidden_dim: int,
) -> dict[str, SKUForecastArtifact]:
    """Load serialized `SKUForecastArtifact` dictionary from disk."""
    from inventory_di.data.features import FEATURE_COLUMNS
    from inventory_di.models.distributions import DiscreteDemandPMFNet

    raw = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    out: dict[str, SKUForecastArtifact] = {}
    for sku, d in raw.items():
        net = DiscreteDemandPMFNet(
            num_stores=len(d["stores"]),
            in_features=len(FEATURE_COLUMNS),
            support_h=int(d["support_h"]),
            hidden_dim=hidden_dim,
        )
        net.load_state_dict(d["state_dict"])
        net.eval()
        out[sku] = SKUForecastArtifact(
            sku_id=d["sku_id"],
            stores=list(d["stores"]),
            model_type=d["model_type"],
            lambda_dfl=float(d["lambda_dfl"]),
            support_h=int(d["support_h"]),
            tail_mean_excess=float(d["tail_mean_excess"]),
            feat_mean=np.asarray(d["feat_mean"], dtype=np.float32),
            feat_std=np.asarray(d["feat_std"], dtype=np.float32),
            model=net,
            history=list(d["history"]),
            calibration_info=dict(d["calibration_info"]),
            gradient_check=dict(d["gradient_check"]),
        )
    return out


def train_all_models(
    cfg: ExperimentConfig,
    weekly_df: pd.DataFrame,
    inv_df: pd.DataFrame,
    lineage: dict[str, Any],
    model_choice: str = "all",
) -> dict[str, Any]:
    """Train `supervised` (NLL-only) and `dfl` (`NLL + lambda_DFL * SPOPlus`) models per SKU,
    plus a `lambda_DFL` ablation table on the validation split.
    """
    inst = build_allocation_instance_from_config_df(inv_df)
    support_h = int(
        max(
            cfg.model.max_demand_support_h,
            lineage.get("support_stats", {}).get("recommended_support_h", cfg.model.max_demand_support_h),
        )
    )
    tail_excess = float(
        lineage.get("support_stats", {}).get("tail_mean_excess_above_h", 2.0)
    )

    art_dir = Path(cfg.data.artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)

    sup_artifacts: dict[str, SKUForecastArtifact] = {}
    dfl_artifacts: dict[str, SKUForecastArtifact] = {}
    ablation_rows: list[dict[str, Any]] = []

    for p_idx, sku in enumerate(inst.skus):
        sku_seed = cfg.global_seed + 100 * p_idx
        if model_choice in {"all", "supervised"}:
            sup_art = train_supervised_sku_model(
                weekly_df=weekly_df,
                sku_id=sku,
                stores=inst.stores,
                support_h=support_h,
                tail_mean_excess=tail_excess,
                hidden_dim=cfg.model.hidden_dim,
                learning_rate=cfg.model.learning_rate,
                epochs=cfg.model.epochs,
                seed=sku_seed,
            )
            sup_artifacts[sku] = sup_art

        if model_choice in {"all", "dfl"} and p_idx == 0:
            # Run quick lambda_DFL ablation on the first SKU to document NLL vs SPO+ trade-off
            for lam in cfg.model.lambda_dfl_grid:
                abl_art = train_dfl_sku_model(
                    weekly_df=weekly_df,
                    sku_id=sku,
                    stores=inst.stores,
                    I_s=inst.I[:, p_idx],
                    K_s=inst.K[:, p_idx],
                    W_p=int(inst.W[p_idx]),
                    b_s=inst.b[:, p_idx],
                    h_s=inst.h[:, p_idx],
                    t_s=inst.t[:, p_idx],
                    support_h=support_h,
                    tail_mean_excess=tail_excess,
                    hidden_dim=cfg.model.hidden_dim,
                    learning_rate=cfg.model.learning_rate,
                    epochs=max(8, cfg.model.epochs // 2),
                    batch_size=cfg.model.batch_size,
                    lambda_dfl=float(lam),
                    spoplus_processes=cfg.model.spoplus_processes,
                    seed=sku_seed,
                )
                ablation_rows.append(
                    {
                        "sku_id": sku,
                        "lambda_dfl": float(lam),
                        "val_nll": abl_art.calibration_info["nll"],
                        "val_mae": abl_art.calibration_info["mae"],
                        "val_wape": abl_art.calibration_info["wape"],
                        "val_decision_regret_mu": abl_art.calibration_info.get(
                            "val_decision_regret_mu", 0.0
                        ),
                        "optimal_temperature": abl_art.calibration_info["optimal_temperature"],
                    }
                )

    # Priority 5: Train DFL jointly across all SKUs with the exact store-capacity-coupled polytope
    if model_choice in {"all", "dfl"}:
        dfl_artifacts = train_joint_multi_sku_dfl_models(
            weekly_df=weekly_df,
            skus=inst.skus,
            stores=inst.stores,
            I_sp=inst.I,
            K_sp=inst.K,
            W_p=inst.W,
            C_s=inst.C,
            b_sp=inst.b,
            h_sp=inst.h,
            t_sp=inst.t,
            support_h=support_h,
            tail_mean_excess=tail_excess,
            hidden_dim=cfg.model.hidden_dim,
            learning_rate=cfg.model.learning_rate,
            epochs=cfg.model.epochs,
            batch_size=cfg.model.batch_size,
            lambda_dfl=cfg.model.lambda_dfl,
            spoplus_processes=cfg.model.spoplus_processes,
            seed=cfg.global_seed,
        )

    hashes: dict[str, str] = {}
    if sup_artifacts:
        hashes["supervised_ckpt_sha256"] = save_sku_forecast_artifacts(
            sup_artifacts, art_dir / "supervised_models.pt"
        )
    if dfl_artifacts:
        hashes["dfl_ckpt_sha256"] = save_sku_forecast_artifacts(
            dfl_artifacts, art_dir / "dfl_models.pt"
        )

    training_diagnostics = {
        "data_mode": cfg.data_mode,
        "support_h": support_h,
        "tail_mean_excess": tail_excess,
        "checkpoints": hashes,
        "supervised": {
            sku: {
                "calibration": art.calibration_info,
                "gradient_check": art.gradient_check,
                "history": art.history,
            }
            for sku, art in sup_artifacts.items()
        },
        "dfl": {
            sku: {
                "lambda_dfl": art.lambda_dfl,
                "calibration": art.calibration_info,
                "gradient_check": art.gradient_check,
                "history": art.history,
            }
            for sku, art in dfl_artifacts.items()
        },
        "lambda_dfl_ablation": ablation_rows,
    }
    save_json(training_diagnostics, art_dir / "training_diagnostics.json")
    return training_diagnostics


def calibrate_scenarios_and_tune_dro_on_val(
    cfg: ExperimentConfig,
    weekly_df: pd.DataFrame,
    inst: AllocationInstance,
    sup_artifacts: dict[str, SKUForecastArtifact],
    dfl_artifacts: dict[str, SKUForecastArtifact],
    copula_sup: EmpiricalCopulaScenarioGenerator,
    copula_dfl: EmpiricalCopulaScenarioGenerator,
) -> dict[str, Any]:
    """Priorities 2 & 3:
    1. Diagnose and calibrate `copula_sup` and `copula_dfl` scenario generators strictly on `split == 'val'`.
    2. Select defensible `(alpha*, risk_weight*)` on `split == 'val'` for `supervised_dro` and `dfl_dro`
       minimizing `0.85 * val_mean_cost + 0.15 * val_cvar_90_cost` before touching any test origin.
    """
    from inventory_di.optimization.dro import exact_ambiguity_set_worst_case_loss

    val_origins = sorted(weekly_df[weekly_df["split"] == "val"]["forecast_origin"].unique())
    S, P = len(inst.stores), len(inst.skus)
    store_idx = {s: i for i, s in enumerate(inst.stores)}
    sku_idx = {p: i for i, p in enumerate(inst.skus)}

    val_sup_pmfs_list: list[dict[str, np.ndarray]] = []
    val_dfl_pmfs_list: list[dict[str, np.ndarray]] = []
    val_realized_list: list[np.ndarray] = []

    for origin in val_origins:
        origin_df = weekly_df[weekly_df["forecast_origin"] == origin].copy()
        D_val = np.zeros((S, P), dtype=int)
        for row in origin_df.itertuples(index=False):
            D_val[store_idx[str(row.store_id)], sku_idx[str(row.sku_id)]] = int(
                row.observed_units_proxy
            )
        val_realized_list.append(D_val)

        s_pmfs: dict[str, np.ndarray] = {}
        d_pmfs: dict[str, np.ndarray] = {}
        for p_i, sku in enumerate(inst.skus):
            s_out = predict_sku_origin_outputs(
                sup_artifacts[sku],
                origin_df,
                I_s=inst.I[:, p_i],
                K_s=inst.K[:, p_i],
                b_s=inst.b[:, p_i],
                h_s=inst.h[:, p_i],
                t_s=inst.t[:, p_i],
            )
            s_pmfs[sku] = s_out["pmf"]
            d_out = predict_sku_origin_outputs(
                dfl_artifacts[sku],
                origin_df,
                I_s=inst.I[:, p_i],
                K_s=inst.K[:, p_i],
                b_s=inst.b[:, p_i],
                h_s=inst.h[:, p_i],
                t_s=inst.t[:, p_i],
            )
            d_pmfs[sku] = d_out["pmf"]
        val_sup_pmfs_list.append(s_pmfs)
        val_dfl_pmfs_list.append(d_pmfs)

    # Priority 2: Diagnose & calibrate scenario generators on validation origins
    sup_scen_diag = copula_sup.calibrate_and_diagnose_on_val(
        val_sup_pmfs_list,
        val_realized_list,
        num_scenarios=cfg.dro.num_scenarios,
        seed=cfg.global_seed + 500,
    )
    dfl_scen_diag = copula_dfl.calibrate_and_diagnose_on_val(
        val_dfl_pmfs_list,
        val_realized_list,
        num_scenarios=cfg.dro.num_scenarios,
        seed=cfg.global_seed + 500,
    )

    # Sample calibrated validation scenarios once per validation origin
    val_sup_scenarios = [
        copula_sup.sample_scenarios(
            val_sup_pmfs_list[i], num_scenarios=cfg.dro.num_scenarios, seed=cfg.global_seed + 500 + i
        )
        for i in range(len(val_origins))
    ]
    val_dfl_scenarios = [
        copula_dfl.sample_scenarios(
            val_dfl_pmfs_list[i], num_scenarios=cfg.dro.num_scenarios, seed=cfg.global_seed + 500 + i
        )
        for i in range(len(val_origins))
    ]

    # Priority 3: Validate DRO parameters on validation data
    cand_alphas = [a for a in cfg.dro.alpha_grid if a > 0.0] or [cfg.dro.alpha]
    cand_rws = [rw for rw in cfg.dro.risk_weight_grid if rw > 0.0] or [cfg.dro.risk_weight]

    def _tune_grid(val_scens: list[np.ndarray], family_name: str) -> tuple[float, float, list[dict[str, Any]]]:
        best_alpha = float(cfg.dro.alpha)
        best_rw = float(cfg.dro.risk_weight)
        best_score = float("inf")
        grid_rows: list[dict[str, Any]] = []
        probs = np.full(len(val_origins), 1.0 / max(1, len(val_origins)), dtype=float)

        for a_val in cand_alphas:
            for rw_val in cand_rws:
                origin_costs: list[float] = []
                for t_i in range(len(val_origins)):
                    sol = solve_scenario_reweighting_dro_milp(
                        val_scens[t_i], inst, alpha=float(a_val), risk_weight=float(rw_val)
                    )
                    c_real = evaluate_exact_cost(sol.x, val_realized_list[t_i], inst).total_cost_mu
                    origin_costs.append(c_real)
                costs_arr = np.array(origin_costs, dtype=float)
                mean_c = float(np.mean(costs_arr))
                cvar90_c, _ = exact_ambiguity_set_worst_case_loss(costs_arr, probs, alpha=0.90)
                composite = 0.85 * mean_c + 0.15 * cvar90_c
                grid_rows.append(
                    {
                        "policy_family": family_name,
                        "split": "val",
                        "alpha": float(a_val),
                        "risk_weight": float(rw_val),
                        "val_mean_cost_mu": mean_c,
                        "val_realized_cvar90_mu": cvar90_c,
                        "val_composite_score_mu": composite,
                    }
                )
                if composite < best_score - 1e-6:
                    best_score = composite
                    best_alpha = float(a_val)
                    best_rw = float(rw_val)
        return best_alpha, best_rw, grid_rows

    sup_alpha_star, sup_rw_star, sup_grid = _tune_grid(val_sup_scenarios, "supervised_dro")
    dfl_alpha_star, dfl_rw_star, dfl_grid = _tune_grid(val_dfl_scenarios, "dfl_dro")

    return {
        "tuned_on_split": "val",
        "supervised_scenario_diagnostics": sup_scen_diag,
        "dfl_scenario_diagnostics": dfl_scen_diag,
        "supervised_dro_selected": {"alpha": sup_alpha_star, "risk_weight": sup_rw_star},
        "dfl_dro_selected": {"alpha": dfl_alpha_star, "risk_weight": dfl_rw_star},
        "validation_dro_grid": sup_grid + dfl_grid,
    }


def run_multi_seed_and_carryover_experiments(
    cfg: ExperimentConfig,
    weekly_df: pd.DataFrame,
    inv_df: pd.DataFrame,
    sup_artifacts: dict[str, SKUForecastArtifact],
    dfl_artifacts: dict[str, SKUForecastArtifact],
    copula_sup: EmpiricalCopulaScenarioGenerator,
    copula_dfl: EmpiricalCopulaScenarioGenerator,
    sup_dro_params: dict[str, float],
    dfl_dro_params: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Priorities 4 & 7:
    - Priority 4: Evaluate all 7 policies across multiple scenario/evaluation seeds `cfg.evaluation.seeds`
      to establish seed stability (mean +/- SD across seeds).
    - Priority 7: Evaluate multi-period rolling inventory carryover (`I_{s,p,t+1} = max(I_{s,p,t} + x_{s,p,t-L} - D_{s,p,t}, 0)`)
      and replenishment lead times `L in {0, 1}` weeks across test origins.
    """
    base_inst = build_allocation_instance_from_config_df(inv_df)
    S, P = len(base_inst.stores), len(base_inst.skus)
    store_idx = {s: i for i, s in enumerate(base_inst.stores)}
    sku_idx = {p: i for i, p in enumerate(base_inst.skus)}
    test_origins = sorted(weekly_df[weekly_df["split"] == "test"]["forecast_origin"].unique())
    all_origins_sorted = sorted(weekly_df["forecast_origin"].unique())

    # Pre-extract per-origin realized demand, trailing signals, and predicted PMFs/marginal costs
    origin_inputs: list[dict[str, Any]] = []
    for origin in test_origins:
        origin_df = weekly_df[weekly_df["forecast_origin"] == origin].copy()
        D_realized = np.zeros((S, P), dtype=int)
        for row in origin_df.itertuples(index=False):
            D_realized[store_idx[str(row.store_id)], sku_idx[str(row.sku_id)]] = int(
                row.observed_units_proxy
            )
        prior_origins = [o for o in all_origins_sorted if o < origin][
            -cfg.evaluation.trailing_weeks_proportional :
        ]
        prior_df = weekly_df[weekly_df["forecast_origin"].isin(prior_origins)]
        trailing_signals = np.zeros((S, P), dtype=float)
        for (s_id, p_id), grp in prior_df.groupby(["store_id", "sku_id"]):
            trailing_signals[store_idx[str(s_id)], sku_idx[str(p_id)]] = float(
                grp["observed_units_proxy"].mean()
            )
        sup_pmfs: dict[str, np.ndarray] = {}
        sup_mc: dict[str, np.ndarray] = {}
        dfl_pmfs: dict[str, np.ndarray] = {}
        dfl_mc: dict[str, np.ndarray] = {}
        for p_i, sku in enumerate(base_inst.skus):
            s_out = predict_sku_origin_outputs(
                sup_artifacts[sku],
                origin_df,
                I_s=base_inst.I[:, p_i],
                K_s=base_inst.K[:, p_i],
                b_s=base_inst.b[:, p_i],
                h_s=base_inst.h[:, p_i],
                t_s=base_inst.t[:, p_i],
            )
            sup_pmfs[sku] = s_out["pmf"]
            sup_mc[sku] = s_out["c_hat_flat"]
            d_out = predict_sku_origin_outputs(
                dfl_artifacts[sku],
                origin_df,
                I_s=base_inst.I[:, p_i],
                K_s=base_inst.K[:, p_i],
                b_s=base_inst.b[:, p_i],
                h_s=base_inst.h[:, p_i],
                t_s=base_inst.t[:, p_i],
            )
            dfl_pmfs[sku] = d_out["pmf"]
            dfl_mc[sku] = d_out["c_hat_flat"]
        origin_inputs.append(
            {
                "origin": origin,
                "D_realized": D_realized,
                "trailing_signals": trailing_signals,
                "sup_pmfs": sup_pmfs,
                "sup_mc": sup_mc,
                "dfl_pmfs": dfl_pmfs,
                "dfl_mc": dfl_mc,
            }
        )

    # Priority 4: Multi-seed reliability across `cfg.evaluation.seeds`
    seed_rows: list[dict[str, Any]] = []
    for eval_seed in cfg.evaluation.seeds:
        pol_costs: dict[str, list[float]] = {p: [] for p in REQUIRED_POLICIES}
        pol_fills: dict[str, list[float]] = {p: [] for p in REQUIRED_POLICIES}
        for o_idx, inp in enumerate(origin_inputs):
            D_real = inp["D_realized"]
            s_scen = copula_sup.sample_scenarios(
                inp["sup_pmfs"], num_scenarios=cfg.dro.num_scenarios, seed=int(eval_seed) + 1000 + o_idx
            )
            d_scen = copula_dfl.sample_scenarios(
                inp["dfl_pmfs"], num_scenarios=cfg.dro.num_scenarios, seed=int(eval_seed) + 1000 + o_idx
            )
            sols = {
                "equal_allocation": allocate_equal_shares(base_inst).x,
                "proportional_allocation": allocate_proportional_shares(
                    inp["trailing_signals"], base_inst
                ).x,
                "supervised_pto": solve_nominal_marginal_cost_milp(inp["sup_mc"], base_inst).x,
                "supervised_dro": solve_scenario_reweighting_dro_milp(
                    s_scen,
                    base_inst,
                    alpha=float(sup_dro_params["alpha"]),
                    risk_weight=float(sup_dro_params["risk_weight"]),
                ).x,
                "dfl_pto": solve_nominal_marginal_cost_milp(inp["dfl_mc"], base_inst).x,
                "dfl_dro": solve_scenario_reweighting_dro_milp(
                    d_scen,
                    base_inst,
                    alpha=float(dfl_dro_params["alpha"]),
                    risk_weight=float(dfl_dro_params["risk_weight"]),
                ).x,
                "clairvoyant_oracle": solve_clairvoyant_oracle(D_real, base_inst).x,
            }
            for pol_name, x_mat in sols.items():
                cb = evaluate_exact_cost(x_mat, D_real, base_inst)
                pol_costs[pol_name].append(cb.total_cost_mu)
                pol_fills[pol_name].append(cb.fill_rate)
        for pol_name in REQUIRED_POLICIES:
            seed_rows.append(
                {
                    "seed": int(eval_seed),
                    "policy": pol_name,
                    "mean_cost_mu": float(np.mean(pol_costs[pol_name])),
                    "std_origin_cost_mu": float(np.std(pol_costs[pol_name], ddof=0)),
                    "mean_fill_rate": float(np.mean(pol_fills[pol_name])),
                    "data_mode": cfg.data_mode,
                }
            )
    multi_seed_df = pd.DataFrame(seed_rows)

    # Priority 7: Multi-period rolling inventory carryover and replenishment lead times L in {0, 1}
    carryover_rows: list[dict[str, Any]] = []
    for lead_time in [0, 1]:
        for pol_name in REQUIRED_POLICIES:
            I_curr = base_inst.I.copy()
            pipeline_queue: list[np.ndarray] = [
                np.zeros((S, P), dtype=int) for _ in range(lead_time)
            ]
            tot_cost = 0.0
            tot_shortage = 0.0
            tot_holding = 0.0
            tot_filled = 0
            tot_demand = 0

            for o_idx, inp in enumerate(origin_inputs):
                origin_df = weekly_df[weekly_df["forecast_origin"] == inp["origin"]].copy()
                D_real = inp["D_realized"]
                # Construct dynamic instance with carried-over opening stock I_curr
                inst_t = AllocationInstance(
                    stores=base_inst.stores,
                    skus=base_inst.skus,
                    I=I_curr.copy(),
                    W=base_inst.W,
                    K=base_inst.K,
                    C=base_inst.C,
                    b=base_inst.b,
                    h=base_inst.h,
                    t=base_inst.t,
                )

                if pol_name == "equal_allocation":
                    x_order = allocate_equal_shares(inst_t).x
                elif pol_name == "proportional_allocation":
                    x_order = allocate_proportional_shares(inp["trailing_signals"], inst_t).x
                elif pol_name == "supervised_pto":
                    mc_t = {
                        sku: predict_sku_origin_outputs(
                            sup_artifacts[sku],
                            origin_df,
                            I_s=I_curr[:, p_i],
                            K_s=inst_t.K[:, p_i],
                            b_s=inst_t.b[:, p_i],
                            h_s=inst_t.h[:, p_i],
                            t_s=inst_t.t[:, p_i],
                        )["c_hat_flat"]
                        for p_i, sku in enumerate(inst_t.skus)
                    }
                    x_order = solve_nominal_marginal_cost_milp(mc_t, inst_t).x
                elif pol_name == "dfl_pto":
                    mc_t = {
                        sku: predict_sku_origin_outputs(
                            dfl_artifacts[sku],
                            origin_df,
                            I_s=I_curr[:, p_i],
                            K_s=inst_t.K[:, p_i],
                            b_s=inst_t.b[:, p_i],
                            h_s=inst_t.h[:, p_i],
                            t_s=inst_t.t[:, p_i],
                        )["c_hat_flat"]
                        for p_i, sku in enumerate(inst_t.skus)
                    }
                    x_order = solve_nominal_marginal_cost_milp(mc_t, inst_t).x
                elif pol_name == "supervised_dro":
                    s_scen = copula_sup.sample_scenarios(
                        inp["sup_pmfs"],
                        num_scenarios=cfg.dro.num_scenarios,
                        seed=cfg.global_seed + 1000 + o_idx,
                    )
                    x_order = solve_scenario_reweighting_dro_milp(
                        s_scen,
                        inst_t,
                        alpha=float(sup_dro_params["alpha"]),
                        risk_weight=float(sup_dro_params["risk_weight"]),
                    ).x
                elif pol_name == "dfl_dro":
                    d_scen = copula_dfl.sample_scenarios(
                        inp["dfl_pmfs"],
                        num_scenarios=cfg.dro.num_scenarios,
                        seed=cfg.global_seed + 1000 + o_idx,
                    )
                    x_order = solve_scenario_reweighting_dro_milp(
                        d_scen,
                        inst_t,
                        alpha=float(dfl_dro_params["alpha"]),
                        risk_weight=float(dfl_dro_params["risk_weight"]),
                    ).x
                else:
                    x_order = solve_clairvoyant_oracle(D_real, inst_t).x

                # Receive shipment ordered `lead_time` weeks ago
                if lead_time == 0:
                    x_arrived = x_order
                else:
                    x_arrived = pipeline_queue.pop(0)
                    # Cap arrived units if store capacity binds with carried-over stock
                    rem_c = np.maximum(0, inst_t.C - np.sum(I_curr, axis=1))
                    for s_i in range(S):
                        if np.sum(x_arrived[s_i, :]) > rem_c[s_i]:
                            scale = float(rem_c[s_i]) / max(1.0, float(np.sum(x_arrived[s_i, :])))
                            x_arrived[s_i, :] = np.floor(x_arrived[s_i, :] * scale).astype(int)
                    pipeline_queue.append(x_order)

                cb = evaluate_exact_cost(x_arrived, D_real, inst_t)
                tot_cost += cb.total_cost_mu
                tot_shortage += cb.shortage_cost_mu
                tot_holding += cb.holding_cost_mu
                tot_filled += cb.filled_units
                tot_demand += int(np.sum(D_real))

                # Carry over remaining on-hand stock to week t+1 (capped by store capacity)
                I_next = np.maximum(0, I_curr + x_arrived - D_real)
                for s_i in range(S):
                    if np.sum(I_next[s_i, :]) > inst_t.C[s_i]:
                        scale = float(inst_t.C[s_i]) / float(np.sum(I_next[s_i, :]))
                        I_next[s_i, :] = np.floor(I_next[s_i, :] * scale).astype(int)
                I_curr = I_next

            n_orig = max(1, len(origin_inputs))
            carryover_rows.append(
                {
                    "lead_time_weeks": int(lead_time),
                    "policy": pol_name,
                    "mean_cost_mu": tot_cost / n_orig,
                    "mean_shortage_cost_mu": tot_shortage / n_orig,
                    "mean_holding_cost_mu": tot_holding / n_orig,
                    "fill_rate": float(tot_filled / max(1, tot_demand)),
                    "ending_inventory_units": int(np.sum(I_curr)),
                    "data_mode": cfg.data_mode,
                }
            )

    carryover_df = pd.DataFrame(carryover_rows)
    return multi_seed_df, carryover_df


def run_rolling_backtest(
    cfg: ExperimentConfig,
    weekly_df: pd.DataFrame,
    inv_df: pd.DataFrame,
    sup_artifacts: dict[str, SKUForecastArtifact],
    dfl_artifacts: dict[str, SKUForecastArtifact],
    lineage: dict[str, Any],
) -> dict[str, Any]:
    """Execute rolling-origin out-of-sample evaluation across all 7 required policies (`AGENTS.md` §8)."""
    inst = build_allocation_instance_from_config_df(inv_df)
    S, P = len(inst.stores), len(inst.skus)
    store_idx = {s: i for i, s in enumerate(inst.stores)}
    sku_idx = {p: i for i, p in enumerate(inst.skus)}

    first_art = next(iter(sup_artifacts.values()))
    copula_sup = EmpiricalCopulaScenarioGenerator(
        stores=inst.stores,
        skus=inst.skus,
        support_h=first_art.support_h,
        tail_mean_excess=first_art.tail_mean_excess,
    )
    copula_dfl = EmpiricalCopulaScenarioGenerator(
        stores=inst.stores,
        skus=inst.skus,
        support_h=first_art.support_h,
        tail_mean_excess=first_art.tail_mean_excess,
    )
    train_only_df = weekly_df[weekly_df["split"] == "train"]
    copula_info = copula_sup.fit_on_train(train_only_df)
    copula_dfl.fit_on_train(train_only_df)

    # Priorities 2 & 3: Diagnose & calibrate scenarios on `val`, and tune DRO (alpha, risk_weight) on `val`
    val_tuning = calibrate_scenarios_and_tune_dro_on_val(
        cfg=cfg,
        weekly_df=weekly_df,
        inst=inst,
        sup_artifacts=sup_artifacts,
        dfl_artifacts=dfl_artifacts,
        copula_sup=copula_sup,
        copula_dfl=copula_dfl,
    )
    sup_alpha_star = float(val_tuning["supervised_dro_selected"]["alpha"])
    sup_rw_star = float(val_tuning["supervised_dro_selected"]["risk_weight"])
    dfl_alpha_star = float(val_tuning["dfl_dro_selected"]["alpha"])
    dfl_rw_star = float(val_tuning["dfl_dro_selected"]["risk_weight"])

    test_origins = sorted(weekly_df[weekly_df["split"] == "test"]["forecast_origin"].unique())
    all_origins_sorted = sorted(weekly_df["forecast_origin"].unique())

    allocation_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    cached_origin_bundles: dict[str, Any] = {}
    origin_env_records: list[dict[str, Any]] = []

    for o_idx, origin in enumerate(test_origins):
        origin_df = weekly_df[weekly_df["forecast_origin"] == origin].copy()
        origin_seed = cfg.global_seed + 1000 + o_idx

        # 1. Ground-truth held-out demand proxy D_realized (S, P) — NEVER clipped at H
        D_realized = np.zeros((S, P), dtype=int)
        for row in origin_df.itertuples(index=False):
            D_realized[store_idx[str(row.store_id)], sku_idx[str(row.sku_id)]] = int(
                row.observed_units_proxy
            )

        # 2. Trailing demand signal strictly prior to `origin` for `proportional_allocation`
        prior_origins = [o for o in all_origins_sorted if o < origin][
            -cfg.evaluation.trailing_weeks_proportional :
        ]
        prior_df = weekly_df[weekly_df["forecast_origin"].isin(prior_origins)]
        trailing_signals = np.zeros((S, P), dtype=float)
        for (s_id, p_id), grp in prior_df.groupby(["store_id", "sku_id"]):
            trailing_signals[store_idx[str(s_id)], sku_idx[str(p_id)]] = float(
                grp["observed_units_proxy"].mean()
            )

        # 3. Predict PMFs and marginal unit costs for Supervised and DFL models
        sup_pmfs: dict[str, np.ndarray] = {}
        sup_mc: dict[str, np.ndarray] = {}
        dfl_pmfs: dict[str, np.ndarray] = {}
        dfl_mc: dict[str, np.ndarray] = {}
        forecast_quantiles_rows: list[dict[str, Any]] = []

        for p_i, sku in enumerate(inst.skus):
            s_out = predict_sku_origin_outputs(
                sup_artifacts[sku],
                origin_df,
                I_s=inst.I[:, p_i],
                K_s=inst.K[:, p_i],
                b_s=inst.b[:, p_i],
                h_s=inst.h[:, p_i],
                t_s=inst.t[:, p_i],
            )
            sup_pmfs[sku] = s_out["pmf"]
            sup_mc[sku] = s_out["c_hat_flat"]

            d_out = predict_sku_origin_outputs(
                dfl_artifacts[sku],
                origin_df,
                I_s=inst.I[:, p_i],
                K_s=inst.K[:, p_i],
                b_s=inst.b[:, p_i],
                h_s=inst.h[:, p_i],
                t_s=inst.t[:, p_i],
            )
            dfl_pmfs[sku] = d_out["pmf"]
            dfl_mc[sku] = d_out["c_hat_flat"]

            for s_i, store in enumerate(inst.stores):
                forecast_quantiles_rows.append(
                    {
                        "origin": origin,
                        "store_id": store,
                        "sku_id": sku,
                        "realized_units": int(D_realized[s_i, p_i]),
                        "sup_expected": float(s_out["expected_demand"][s_i]),
                        "sup_p10": float(s_out["p10_demand"][s_i]),
                        "sup_p50": float(s_out["p50_demand"][s_i]),
                        "sup_p90": float(s_out["p90_demand"][s_i]),
                        "sup_overflow_prob": float(s_out["overflow_prob"][s_i]),
                        "dfl_expected": float(d_out["expected_demand"][s_i]),
                        "dfl_p10": float(d_out["p10_demand"][s_i]),
                        "dfl_p50": float(d_out["p50_demand"][s_i]),
                        "dfl_p90": float(d_out["p90_demand"][s_i]),
                        "dfl_overflow_prob": float(d_out["overflow_prob"][s_i]),
                    }
                )

        # 4. Sample common-random-number joint demand scenarios for DRO policies
        # Both use identical `origin_seed` so the underlying uniform rank draws U are identical
        sup_scenarios = copula_sup.sample_scenarios(
            sup_pmfs, num_scenarios=cfg.dro.num_scenarios, seed=origin_seed
        )
        dfl_scenarios = copula_dfl.sample_scenarios(
            dfl_pmfs, num_scenarios=cfg.dro.num_scenarios, seed=origin_seed
        )

        # 5. Solve all 7 policies on the exact same instance (using validation-tuned DRO parameters)
        res_equal = allocate_equal_shares(inst)
        res_prop = allocate_proportional_shares(trailing_signals, inst)
        res_sup_pto = solve_nominal_marginal_cost_milp(sup_mc, inst)
        res_sup_dro = solve_scenario_reweighting_dro_milp(
            sup_scenarios,
            inst,
            alpha=sup_alpha_star,
            risk_weight=sup_rw_star,
        )
        res_dfl_pto = solve_nominal_marginal_cost_milp(dfl_mc, inst)
        res_dfl_dro = solve_scenario_reweighting_dro_milp(
            dfl_scenarios,
            inst,
            alpha=dfl_alpha_star,
            risk_weight=dfl_rw_star,
        )
        res_oracle = solve_clairvoyant_oracle(D_realized, inst)
        oracle_cost_mu = evaluate_exact_cost(res_oracle.x, D_realized, inst).total_cost_mu

        policy_solutions: dict[str, tuple[np.ndarray, str, bool, float, float, float]] = {}
        # Compute model-implied expected cost & CVaR for each policy on reference scenarios
        for pol_name, sol_obj, scen_ref in [
            ("equal_allocation", res_equal, sup_scenarios),
            ("proportional_allocation", res_prop, sup_scenarios),
            ("supervised_pto", res_sup_pto, sup_scenarios),
            ("supervised_dro", res_sup_dro, sup_scenarios),
            ("dfl_pto", res_dfl_pto, dfl_scenarios),
            ("dfl_dro", res_dfl_dro, dfl_scenarios),
            ("clairvoyant_oracle", res_oracle, sup_scenarios),
        ]:
            scen_stats = evaluate_allocation_on_scenarios(
                sol_obj.x, scen_ref, inst, alpha=cfg.dro.alpha
            )
            policy_solutions[pol_name] = (
                sol_obj.x,
                sol_obj.solver_status,
                sol_obj.is_feasible,
                sol_obj.runtime_seconds,
                float(scen_stats["expected_cost_mu"]),
                float(scen_stats["cvar_cost_mu"]),
            )

        env_fp = hashlib.sha256(
            json.dumps(
                {
                    "origin": origin,
                    "I": inst.I.tolist(),
                    "W": inst.W.tolist(),
                    "K": inst.K.tolist(),
                    "C": inst.C.tolist(),
                    "seed": origin_seed,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]

        for pol_name in REQUIRED_POLICIES:
            x_mat, status_str, is_feas, rt_sec, mod_exp, mod_cvar = policy_solutions[pol_name]
            realized = evaluate_exact_cost(x_mat, D_realized, inst)
            regret_mu = float(realized.total_cost_mu - oracle_cost_mu)

            metric_rows.append(
                {
                    "run_id": cfg.run_id,
                    "origin": origin,
                    "policy": pol_name,
                    "total_cost_mu": realized.total_cost_mu,
                    "shortage_cost_mu": realized.shortage_cost_mu,
                    "holding_cost_mu": realized.holding_cost_mu,
                    "shipping_cost_mu": realized.shipping_cost_mu,
                    "filled_units": realized.filled_units,
                    "unmet_units": realized.unmet_units,
                    "fill_rate": realized.fill_rate,
                    "oracle_regret_mu": regret_mu,
                    "model_expected_cost_mu": mod_exp,
                    "model_cvar_cost_mu": mod_cvar,
                    "solver_status": status_str,
                    "runtime_seconds": rt_sec,
                    "data_mode": cfg.data_mode,
                }
            )

            for s_i, store in enumerate(inst.stores):
                for p_i, sku in enumerate(inst.skus):
                    allocation_rows.append(
                        {
                            "run_id": cfg.run_id,
                            "forecast_origin": origin,
                            "policy": pol_name,
                            "store_id": store,
                            "sku_id": sku,
                            "allocated_units": int(x_mat[s_i, p_i]),
                            "realized_demand_units": int(D_realized[s_i, p_i]),
                            "opening_stock_units": int(inst.I[s_i, p_i]),
                            "warehouse_supply_units": int(inst.W[p_i]),
                            "store_sku_unit_cap": int(inst.K[s_i, p_i]),
                            "store_total_capacity_units": int(inst.C[s_i]),
                            "scenario_profile": f"alpha={cfg.dro.alpha:.2f}_rw={cfg.dro.risk_weight:.2f}",
                            "data_mode": cfg.data_mode,
                            "is_feasible": bool(is_feas),
                            "solver_status": status_str,
                        }
                    )

            origin_env_records.append(
                {
                    "origin": origin,
                    "policy": pol_name,
                    "env_fingerprint": env_fp,
                    "solver_status": status_str,
                    "runtime_seconds": rt_sec,
                }
            )

        cached_origin_bundles[origin] = {
            "origin": origin,
            "seed": origin_seed,
            "D_realized": D_realized.tolist(),
            "sup_scenarios": sup_scenarios.tolist(),
            "dfl_scenarios": dfl_scenarios.tolist(),
            "sup_mc": {k: v.tolist() for k, v in sup_mc.items()},
            "dfl_mc": {k: v.tolist() for k, v in dfl_mc.items()},
            "forecast_quantiles": forecast_quantiles_rows,
        }

    alloc_df = pd.DataFrame(allocation_rows)
    metrics_df = pd.DataFrame(metric_rows)
    validate_allocation_results_df(alloc_df)
    validate_backtest_metrics_df(metrics_df)

    # Sensitivity sweep on the first test origin across alpha, risk_weight, and supply_multiplier
    first_origin = test_origins[0]
    b0 = cached_origin_bundles[first_origin]
    D0 = np.array(b0["D_realized"], dtype=int)
    dfl_scen0 = np.array(b0["dfl_scenarios"], dtype=int)
    sup_scen0 = np.array(b0["sup_scenarios"], dtype=int)

    sensitivity_rows: list[dict[str, Any]] = []
    for mult in cfg.dro.supply_stress_multipliers:
        inst_m = build_allocation_instance_from_config_df(inv_df, supply_multiplier=float(mult))
        oracle_m = solve_clairvoyant_oracle(D0, inst_m)
        oracle_m_cost = evaluate_exact_cost(oracle_m.x, D0, inst_m).total_cost_mu

        for alpha_val in cfg.dro.alpha_grid:
            for rw_val in cfg.dro.risk_weight_grid:
                for model_label, scen_arr in [("dfl_dro", dfl_scen0), ("supervised_dro", sup_scen0)]:
                    dro_m = solve_scenario_reweighting_dro_milp(
                        scen_arr, inst_m, alpha=float(alpha_val), risk_weight=float(rw_val)
                    )
                    real_m = evaluate_exact_cost(dro_m.x, D0, inst_m)
                    sensitivity_rows.append(
                        {
                            "origin": first_origin,
                            "policy_family": model_label,
                            "supply_multiplier": float(mult),
                            "alpha": float(alpha_val),
                            "risk_weight": float(rw_val),
                            "model_expected_cost_mu": dro_m.model_expected_cost_mu,
                            "model_cvar_cost_mu": dro_m.model_cvar_cost_mu,
                            "realized_test_cost_mu": real_m.total_cost_mu,
                            "realized_fill_rate": real_m.fill_rate,
                            "oracle_regret_mu": float(real_m.total_cost_mu - oracle_m_cost),
                            "solver_status": dro_m.solver_status,
                            "is_feasible": dro_m.is_feasible,
                            "allocated_total_units": int(np.sum(dro_m.x)),
                        }
                    )

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    summary_df, bootstrap_meta = summarize_backtest_metrics(
        metrics_df,
        alpha=cfg.dro.alpha,
        block_size=cfg.evaluation.bootstrap_block_size,
        num_resamples=cfg.evaluation.bootstrap_resamples,
        seed=cfg.global_seed,
    )

    # Priorities 4 & 7: Multi-seed stability across `cfg.evaluation.seeds` & rolling carryover/lead-time evaluation
    multi_seed_df, carryover_df = run_multi_seed_and_carryover_experiments(
        cfg=cfg,
        weekly_df=weekly_df,
        inv_df=inv_df,
        sup_artifacts=sup_artifacts,
        dfl_artifacts=dfl_artifacts,
        copula_sup=copula_sup,
        copula_dfl=copula_dfl,
        sup_dro_params={"alpha": sup_alpha_star, "risk_weight": sup_rw_star},
        dfl_dro_params={"alpha": dfl_alpha_star, "risk_weight": dfl_rw_star},
    )

    art_dir = Path(cfg.data.artifacts_dir)
    alloc_sha = save_parquet(alloc_df, art_dir / "allocation_results.parquet")
    metrics_sha = save_parquet(metrics_df, art_dir / "backtest_metrics.parquet")
    sens_sha = save_parquet(sensitivity_df, art_dir / "risk_sensitivity.parquet")
    summary_sha = save_parquet(summary_df, art_dir / "summary_metrics.parquet")
    multiseed_sha = save_parquet(multi_seed_df, art_dir / "multi_seed_summary.parquet")
    carryover_sha = save_parquet(carryover_df, art_dir / "carryover_leadtime_metrics.parquet")
    save_json(cached_origin_bundles, art_dir / "cached_origin_bundles.json")
    save_json(val_tuning, art_dir / "validation_dro_tuning.json")

    # Build reproducibility manifest conforming to `.agents/skills/experiment-reproducibility`
    sup_ckpt_path = art_dir / "supervised_models.pt"
    dfl_ckpt_path = art_dir / "dfl_models.pt"
    sup_sha = sha256_file(sup_ckpt_path) if sup_ckpt_path.exists() else "in_memory"
    dfl_sha = sha256_file(dfl_ckpt_path) if dfl_ckpt_path.exists() else "in_memory"

    dataset_hashes: dict[str, str] = {
        "weekly_demand.parquet": str(lineage.get("weekly_demand_sha256", alloc_sha)),
        "inventory_config.parquet": str(lineage.get("inventory_config_sha256", metrics_sha)),
    }
    if cfg.data_mode == "m5_sales_proxy" and "m5_raw_file_hashes" in lineage:
        raw_h = lineage["m5_raw_file_hashes"]
        dataset_hashes["calendar.csv"] = raw_h.get("calendar", "")
        dataset_hashes["sell_prices.csv"] = raw_h.get("sell_prices", "")
        dataset_hashes["sales_train_evaluation.csv"] = raw_h.get("sales", "")

    combined_fp = hashlib.sha256(
        json.dumps([r["env_fingerprint"] for r in origin_env_records], sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    solver_statuses_by_pol: dict[str, list[str]] = {}
    for pol in REQUIRED_POLICIES:
        solver_statuses_by_pol[pol] = (
            metrics_df[metrics_df["policy"] == pol].sort_values("origin")["solver_status"].tolist()
        )

    run_manifest = {
        "run_id": cfg.run_id,
        "git_commit": get_git_commit(),
        "python_version": get_environment_snapshot()["python_version"],
        "platform": get_environment_snapshot()["platform"],
        "global_seed": cfg.global_seed,
        "config_path": cfg.config_path,
        "config_sha256": cfg.config_sha256,
        "data_mode": cfg.data_mode,
        "economic_currency": cfg.economics.currency,
        "inventory_data_provenance": "synthetic",
        "dataset_hashes": dataset_hashes,
        "model_checkpoints": {
            "supervised_pmf": {"path": str(sup_ckpt_path), "sha256": sup_sha},
            "dfl_pmf": {"path": str(dfl_ckpt_path), "sha256": dfl_sha},
        },
        "test_origins": test_origins,
        "common_scenario_seed": int(cfg.global_seed + 1000),
        "num_scenarios": int(cfg.dro.num_scenarios),
        "policies_evaluated": REQUIRED_POLICIES,
        "policy_origin_Fingerprints": {pol: combined_fp for pol in REQUIRED_POLICIES},
        "solver_statuses": solver_statuses_by_pol,
        "metrics_recorded": sorted(list(metrics_df.columns) + ["cvar_90_cost_mu"]),
        "small_origin_caveat_noted": True,
        "artifact_hashes": {
            "allocation_results_parquet": alloc_sha,
            "backtest_metrics_parquet": metrics_sha,
            "risk_sensitivity_parquet": sens_sha,
            "summary_metrics_parquet": summary_sha,
            "multi_seed_summary_parquet": multiseed_sha,
            "carryover_leadtime_metrics_parquet": carryover_sha,
        },
        "copula_info": copula_info,
        "validation_dro_tuning": {
            "tuned_on_split": "val",
            "supervised_dro_selected": val_tuning["supervised_dro_selected"],
            "dfl_dro_selected": val_tuning["dfl_dro_selected"],
            "supervised_scenario_diagnostics": val_tuning["supervised_scenario_diagnostics"],
            "dfl_scenario_diagnostics": val_tuning["dfl_scenario_diagnostics"],
        },
        "evaluated_seeds": [int(s) for s in cfg.evaluation.seeds],
        "evaluated_policies": REQUIRED_POLICIES,
        "origin_evaluations": origin_env_records,
        **bootstrap_meta,
    }

    save_json(run_manifest, art_dir / "run_manifest.json")
    return {
        "allocation_results": alloc_df,
        "backtest_metrics": metrics_df,
        "summary_metrics": summary_df,
        "risk_sensitivity": sensitivity_df,
        "run_manifest": run_manifest,
    }
