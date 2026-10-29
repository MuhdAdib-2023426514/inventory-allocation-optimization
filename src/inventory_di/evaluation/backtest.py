"""End-to-end training, rolling-origin 7-policy backtesting, and sensitivity sweep runner (`AGENTS.md` §6, §7, §8, §9)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from inventory_di.config import ExperimentConfig
from inventory_di.evaluation.metrics import EVALUATED_POLICIES, summarize_backtest_metrics
from inventory_di.io import (
    get_environment_snapshot,
    get_git_commit,
    save_json,
    save_parquet,
    sha256_file,
    source_provenance,
    validate_allocation_results_df,
    validate_backtest_metrics_df,
)
from inventory_di.models.dfl import train_joint_multi_sku_dfl_models
from inventory_di.models.distributions import EmpiricalCopulaScenarioGenerator
from inventory_di.models.supervised import (
    SKUForecastArtifact,
    predict_sku_origin_outputs,
)
from inventory_di.optimization.baselines import (
    allocate_equal_shares,
    allocate_proportional_shares,
    solve_clairvoyant_oracle,
)
from inventory_di.optimization.core import (
    AllocationInstance,
    evaluate_exact_cost,
    solve_multiperiod_carryover_oracle,
    solve_nominal_marginal_cost_milp,
)
from inventory_di.optimization.dro import (
    evaluate_allocation_on_scenarios,
    solve_scenario_reweighting_dro_milp,
)
from inventory_di.simulation.simulator import InventoryStateTransitionSystem


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
        W[p_i] = max(0, int(round(float(row.warehouse_supply_units) * float(supply_multiplier))))
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
    excess = weekly_df.loc[(weekly_df.split == "train") & (weekly_df.observed_units_proxy > support_h), "observed_units_proxy"] - support_h
    tail_excess = float(excess.mean()) if len(excess) else 2.0

    art_dir = Path(cfg.data.artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    def train_joint(lam):
        return train_joint_multi_sku_dfl_models(
            weekly_df=weekly_df, skus=inst.skus, stores=inst.stores, I_sp=inst.I,
            K_sp=inst.K, W_p=inst.W, C_s=inst.C, b_sp=inst.b, h_sp=inst.h, t_sp=inst.t,
            support_h=support_h, tail_mean_excess=tail_excess, hidden_dim=cfg.model.hidden_dim,
            learning_rate=cfg.model.learning_rate, epochs=cfg.model.epochs,
            batch_size=cfg.model.batch_size, lambda_dfl=lam,
            spoplus_processes=cfg.model.spoplus_processes, seed=cfg.global_seed)
    sup_artifacts, dfl_artifacts = {}, {}
    ablation_rows = []
    if model_choice in {"all", "supervised"}:
        sup_artifacts = train_joint(0.0)
    if model_choice in {"all", "dfl"}:
        best_score = float("inf")
        for lam in sorted(set(cfg.model.lambda_dfl_grid + [cfg.model.lambda_dfl])):
            if lam == 0 and sup_artifacts:
                arts = sup_artifacts
            else:
                arts = train_joint(float(lam))
            for sku, art in arts.items():
                cal = art.calibration_info
                ablation_rows.append(dict(sku_id=sku, lambda_dfl=lam, val_nll=cal["nll"],
                    val_mae=cal["mae"], val_wape=cal["wape"],
                    val_decision_regret_mu=cal["val_decision_regret_mu"], epochs=cfg.model.epochs))
            score = float(np.mean([a.calibration_info["nll"] + 0.15*a.calibration_info["val_decision_regret_mu"] for a in arts.values()]))
            if lam > 0 and score < best_score:
                best_score, dfl_artifacts = score, arts
        if not dfl_artifacts:
            raise ValueError("DFL requires a positive lambda candidate")

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
        "training_seconds": time.perf_counter() - started,
        "effective_config": cfg.model_dump(),
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
    diag_path = art_dir / "training_diagnostics.json"
    if model_choice != "all" and diag_path.exists():
        old = json.loads(diag_path.read_text())
        other = "dfl" if model_choice == "supervised" else "supervised"
        training_diagnostics[other] = old.get(other, {})
        training_diagnostics["checkpoints"] = {**old.get("checkpoints", {}), **hashes}
    save_json(training_diagnostics, diag_path)
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


def _project_features_one_week_ahead(
    origin_df: pd.DataFrame,
    expected_demand: np.ndarray,
    stores: list[str],
    skus: list[str],
) -> pd.DataFrame:
    """Project weekly features strictly 1-week ahead using decision-time forecast expectations.

    Eliminates future-information leakage: at origin t, lag_1w for arrival week t+1
    is the model's own expected demand D_hat(t) derived strictly from F_t,
    NOT the ground-truth realization D(t).
    """
    proj = origin_df.copy()
    store_map = {s: i for i, s in enumerate(stores)}
    sku_map = {p: i for i, p in enumerate(skus)}

    for row_idx in range(len(proj)):
        s_id = str(proj.iat[row_idx, proj.columns.get_loc("store_id")])
        p_id = str(proj.iat[row_idx, proj.columns.get_loc("sku_id")])
        s_i = store_map[s_id]
        p_i = sku_map[p_id]
        d_hat = float(expected_demand[s_i, p_i])
        old_lag1 = float(proj.iat[row_idx, proj.columns.get_loc("lag_1w_units")])
        old_r4 = float(proj.iat[row_idx, proj.columns.get_loc("rolling_4w_mean")])
        old_r8 = float(proj.iat[row_idx, proj.columns.get_loc("rolling_8w_mean")])

        # At t+1, lag 2w is week t-1 demand (which was old lag 1w)
        proj.iat[row_idx, proj.columns.get_loc("lag_2w_units")] = old_lag1
        # At t+1, lag 1w is the projected expectation for week t
        proj.iat[row_idx, proj.columns.get_loc("lag_1w_units")] = d_hat
        # Roll moving averages forward with expectation
        proj.iat[row_idx, proj.columns.get_loc("rolling_4w_mean")] = 0.75 * old_r4 + 0.25 * d_hat
        proj.iat[row_idx, proj.columns.get_loc("rolling_8w_mean")] = (7.0 / 8.0) * old_r8 + (1.0 / 8.0) * d_hat

    # Advance calendar seasonality by 7 days
    dt = pd.to_datetime(proj["target_start"]) + pd.Timedelta(days=7)
    iso_w = dt.dt.isocalendar().week.astype(float)
    proj["week_of_year_sin"] = np.sin(2.0 * np.pi * iso_w / 52.0)
    proj["week_of_year_cos"] = np.cos(2.0 * np.pi * iso_w / 52.0)
    return proj


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute P1.1, P1.2 & P1.3:
    - P1.2: Independent training-seed + scenario-seed experiments across `cfg.evaluation.seeds`
      (training independent `supervised` and `dfl` neural networks per seed).
    - P1.1: Corrected multi-period rolling inventory carryover simulator (`L in {0, 1}`) with
      exact multi-period rolling-horizon `clairvoyant_oracle` MILP (`Rolling Regret >= 0`) and
      projected arrival-week opening stock for `L=1`.
    - P1.3: Full-test-horizon Cost–Risk–Service-Level trade-off frontier across `(alpha, risk_weight)`.
    """
    from inventory_di.optimization.dro import exact_ambiguity_set_worst_case_loss

    base_inst = build_allocation_instance_from_config_df(inv_df)
    S, P = len(base_inst.stores), len(base_inst.skus)
    store_idx = {s: i for i, s in enumerate(base_inst.stores)}
    sku_idx = {p: i for i, p in enumerate(base_inst.skus)}
    test_origins = sorted(weekly_df[weekly_df["split"] == "test"]["forecast_origin"].unique())
    all_origins_sorted = sorted(weekly_df["forecast_origin"].unique())
    first_art = next(iter(sup_artifacts.values()))
    support_h = int(first_art.support_h)
    tail_excess = float(first_art.tail_mean_excess)

    # Pre-extract per-origin realized demand, trailing signals, and primary-seed PMFs/marginal costs
    origin_inputs: list[dict[str, Any]] = []
    for o_idx, origin in enumerate(test_origins):
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
        sup_exp = np.zeros((S, P), dtype=float)
        dfl_pmfs: dict[str, np.ndarray] = {}
        dfl_mc: dict[str, np.ndarray] = {}
        dfl_exp = np.zeros((S, P), dtype=float)
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
            sup_exp[:, p_i] = s_out["expected_demand"]
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
            dfl_exp[:, p_i] = d_out["expected_demand"]
        origin_inputs.append(
            {
                "origin": origin,
                "origin_df": origin_df,
                "D_realized": D_realized,
                "trailing_signals": trailing_signals,
                "sup_pmfs": sup_pmfs,
                "sup_mc": sup_mc,
                "sup_exp": sup_exp,
                "dfl_pmfs": dfl_pmfs,
                "dfl_mc": dfl_mc,
                "dfl_exp": dfl_exp,
                "sup_scen": copula_sup.sample_scenarios(
                    sup_pmfs, num_scenarios=cfg.dro.num_scenarios, seed=cfg.global_seed + 1000 + o_idx
                ),
                "dfl_scen": copula_dfl.sample_scenarios(
                    dfl_pmfs, num_scenarios=cfg.dro.num_scenarios, seed=cfg.global_seed + 1000 + o_idx
                ),
            }
        )

    # P1.2: Independent training-seed + scenario-seed experiments across `cfg.evaluation.seeds`
    seed_rows: list[dict[str, Any]] = []
    for eval_seed in cfg.evaluation.seeds:
        seed_int = int(eval_seed)
        independent = bool(cfg.evaluation.independent_training_seeds)
        seed_cp_sup, seed_cp_dfl = copula_sup, copula_dfl
        seed_sup_params, seed_dfl_params = sup_dro_params, dfl_dro_params
        if seed_int == int(cfg.global_seed) or not independent:
            seed_sup_arts, seed_dfl_arts = sup_artifacts, dfl_artifacts
        else:
            seed_cfg = cfg.model_copy(deep=True)
            seed_cfg.global_seed = seed_int
            seed_cfg.run_id += f"_seed_{seed_int}"
            seed_dir = Path(cfg.data.artifacts_dir) / "seeds" / str(seed_int)
            seed_cfg.data.artifacts_dir = str(seed_dir)
            train_all_models(seed_cfg, weekly_df, inv_df, {"support_stats": {"recommended_support_h": support_h, "tail_mean_excess_above_h": tail_excess}}, "all")
            seed_sup_arts = load_sku_forecast_artifacts(seed_dir / "supervised_models.pt", cfg.model.hidden_dim)
            seed_dfl_arts = load_sku_forecast_artifacts(seed_dir / "dfl_models.pt", cfg.model.hidden_dim)
            seed_cp_sup = EmpiricalCopulaScenarioGenerator(base_inst.stores, base_inst.skus, support_h, tail_excess)
            seed_cp_dfl = EmpiricalCopulaScenarioGenerator(base_inst.stores, base_inst.skus, support_h, tail_excess)
            seed_cp_sup.fit_on_train(weekly_df[weekly_df.split == "train"])
            seed_cp_dfl.fit_on_train(weekly_df[weekly_df.split == "train"])
            tuning = calibrate_scenarios_and_tune_dro_on_val(seed_cfg, weekly_df, base_inst,
                seed_sup_arts, seed_dfl_arts, seed_cp_sup, seed_cp_dfl)
            seed_sup_params, seed_dfl_params = tuning["supervised_dro_selected"], tuning["dfl_dro_selected"]
            save_json(tuning, seed_dir / "validation_dro_tuning.json")

        pol_costs: dict[str, list[float]] = {p: [] for p in EVALUATED_POLICIES}
        pol_fills: dict[str, list[float]] = {p: [] for p in EVALUATED_POLICIES}
        for o_idx, inp in enumerate(origin_inputs):
            D_real = inp["D_realized"]
            origin_df = inp["origin_df"]
            s_pmfs_seed: dict[str, np.ndarray] = {}
            s_mc_seed: dict[str, np.ndarray] = {}
            d_pmfs_seed: dict[str, np.ndarray] = {}
            d_mc_seed: dict[str, np.ndarray] = {}
            for p_i, sku in enumerate(base_inst.skus):
                so = predict_sku_origin_outputs(
                    seed_sup_arts[sku],
                    origin_df,
                    I_s=base_inst.I[:, p_i],
                    K_s=base_inst.K[:, p_i],
                    b_s=base_inst.b[:, p_i],
                    h_s=base_inst.h[:, p_i],
                    t_s=base_inst.t[:, p_i],
                )
                s_pmfs_seed[sku] = so["pmf"]
                s_mc_seed[sku] = so["c_hat_flat"]
                do = predict_sku_origin_outputs(
                    seed_dfl_arts[sku],
                    origin_df,
                    I_s=base_inst.I[:, p_i],
                    K_s=base_inst.K[:, p_i],
                    b_s=base_inst.b[:, p_i],
                    h_s=base_inst.h[:, p_i],
                    t_s=base_inst.t[:, p_i],
                )
                d_pmfs_seed[sku] = do["pmf"]
                d_mc_seed[sku] = do["c_hat_flat"]

            s_scen = seed_cp_sup.sample_scenarios(
                s_pmfs_seed, num_scenarios=cfg.dro.num_scenarios, seed=seed_int + 1000 + o_idx
            )
            d_scen = seed_cp_dfl.sample_scenarios(
                d_pmfs_seed, num_scenarios=cfg.dro.num_scenarios, seed=seed_int + 1000 + o_idx
            )
            sols = {
                "equal_allocation": allocate_equal_shares(base_inst).x,
                "proportional_allocation": allocate_proportional_shares(
                    inp["trailing_signals"], base_inst
                ).x,
                "supervised_pto": solve_nominal_marginal_cost_milp(s_mc_seed, base_inst).x,
                "supervised_dro": solve_scenario_reweighting_dro_milp(
                    s_scen,
                    base_inst,
                    alpha=float(seed_sup_params["alpha"]),
                    risk_weight=float(seed_sup_params["risk_weight"]),
                ).x,
                "dfl_pto": solve_nominal_marginal_cost_milp(d_mc_seed, base_inst).x,
                "dfl_dro": solve_scenario_reweighting_dro_milp(
                    d_scen,
                    base_inst,
                    alpha=float(seed_dfl_params["alpha"]),
                    risk_weight=float(seed_dfl_params["risk_weight"]),
                ).x,
                "supervised_stochastic": solve_scenario_reweighting_dro_milp(s_scen, base_inst, risk_weight=0).x,
                "dfl_stochastic": solve_scenario_reweighting_dro_milp(d_scen, base_inst, risk_weight=0).x,
                "clairvoyant_oracle": solve_clairvoyant_oracle(D_real, base_inst).x,
            }
            for pol_name, x_mat in sols.items():
                cb = evaluate_exact_cost(x_mat, D_real, base_inst)
                pol_costs[pol_name].append(cb.total_cost_mu)
                pol_fills[pol_name].append(cb.fill_rate)
        oracle_seed_mean = float(np.mean(pol_costs["clairvoyant_oracle"]))
        for pol_name in EVALUATED_POLICIES:
            seed_rows.append(
                {
                    "seed": seed_int,
                    "training_seed_independent": independent,
                    "epochs_budget": cfg.model.epochs,
                    "batch_size": cfg.model.batch_size,
                    "policy": pol_name,
                    "mean_cost_mu": float(np.mean(pol_costs[pol_name])),
                    "mean_regret_mu": float(np.mean(pol_costs[pol_name]) - oracle_seed_mean),
                    "std_origin_cost_mu": float(np.std(pol_costs[pol_name], ddof=0)),
                    "mean_fill_rate": float(np.mean(pol_fills[pol_name])),
                    "data_mode": cfg.data_mode,
                }
            )
    multi_seed_df = pd.DataFrame(seed_rows)

    # P1.1: Corrected multi-period rolling inventory carryover and replenishment lead times L in {0, 1}
    D_traj = np.stack([inp["D_realized"] for inp in origin_inputs], axis=0)  # (T, S, P)
    # Common initial pipeline for L=1 (pre-test baseline order in transit at t=0)
    init_in_transit = allocate_proportional_shares(origin_inputs[0]["trailing_signals"], base_inst).x
    carryover_rows: list[dict[str, Any]] = []

    for lead_time in [0, 1]:
        init_pipe = [init_in_transit.copy() for _ in range(lead_time)]
        oracle_mp = solve_multiperiod_carryover_oracle(
            D_trajectory=D_traj,
            base_inst=base_inst,
            lead_time=lead_time,
            initial_pipeline=init_pipe,
        )
        oracle_mp_cost = float(oracle_mp["mean_cost_mu"])

        for pol_name in EVALUATED_POLICIES:
            if pol_name == "clairvoyant_oracle":
                carryover_rows.append(
                    {
                        "lead_time_weeks": int(lead_time),
                        "policy": "clairvoyant_oracle",
                        "mean_cost_mu": oracle_mp_cost,
                        "rolling_oracle_regret_mu": 0.0,
                        "mean_shortage_cost_mu": float(oracle_mp["mean_shortage_cost_mu"]),
                        "mean_holding_cost_mu": float(oracle_mp["mean_holding_cost_mu"]),
                        "fill_rate": float(oracle_mp["fill_rate"]),
                        "ending_inventory_units": int(oracle_mp["ending_inventory_units"]),
                        "ending_in_transit_units": int(oracle_mp["ending_in_transit_units"]),
                        "is_conserved": bool(oracle_mp["is_conserved"]),
                        "data_mode": cfg.data_mode,
                    }
                )
                continue

            sim = InventoryStateTransitionSystem(
                base_inst=base_inst,
                lead_time_weeks=lead_time,
                initial_pipeline=init_pipe,
            )

            for o_idx, inp in enumerate(origin_inputs):
                D_real = inp["D_realized"]
                I_curr = sim.store_on_hand.copy()

                if lead_time == 0:
                    I_order_alloc = I_curr.copy()
                    # Carryover changes the survival thresholds I_s + k. The cached
                    # zero-stock costs would value units already on the shelf again.
                    mc_sup = {
                        sku: predict_sku_origin_outputs(
                            sup_artifacts[sku], inp["origin_df"],
                            I_curr[:, p_i], base_inst.K[:, p_i],
                            base_inst.b[:, p_i], base_inst.h[:, p_i], base_inst.t[:, p_i],
                        )["c_hat_flat"]
                        for p_i, sku in enumerate(base_inst.skus)
                    }
                    mc_dfl = {
                        sku: predict_sku_origin_outputs(
                            dfl_artifacts[sku], inp["origin_df"],
                            I_curr[:, p_i], base_inst.K[:, p_i],
                            base_inst.b[:, p_i], base_inst.h[:, p_i], base_inst.t[:, p_i],
                        )["c_hat_flat"]
                        for p_i, sku in enumerate(base_inst.skus)
                    }
                    sup_scen_order = inp["sup_scen"]
                    dfl_scen_order = inp["dfl_scen"]
                else:
                    # Lead time 1: Order placed at o_idx arrives at o_idx + 1
                    # Expected demand at current week derived strictly from F_{o_idx}
                    exp_d_curr = (
                        inp["dfl_exp"]
                        if pol_name in {"dfl_pto", "dfl_dro", "dfl_stochastic"}
                        else (
                            inp["sup_exp"]
                            if pol_name in {"supervised_pto", "supervised_dro", "supervised_stochastic"}
                            else inp["trailing_signals"]
                        )
                    )
                    # Arriving shipment at current week (from pipeline or deferred queue)
                    arriving_curr = sim.deferred_arrivals.copy() + (
                        sim.pipeline[0].copy()
                        if len(sim.pipeline) > 0
                        else np.zeros((S, P), dtype=int)
                    )
                    # Projected opening stock at arrival week t+1 based strictly on F_{o_idx}
                    I_proj = np.maximum(0, I_curr + arriving_curr - exp_d_curr).astype(int)
                    for s_i in range(S):
                        if np.sum(I_proj[s_i, :]) > base_inst.C[s_i]:
                            scale = float(base_inst.C[s_i]) / float(np.sum(I_proj[s_i, :]))
                            I_proj[s_i, :] = np.floor(I_proj[s_i, :] * scale).astype(int)
                    I_order_alloc = I_proj.copy()

                    # Projected features for arrival week t+1 strictly from F_{o_idx} (zero future leakage)
                    features_proj = _project_features_one_week_ahead(
                        inp["origin_df"],
                        expected_demand=exp_d_curr,
                        stores=base_inst.stores,
                        skus=base_inst.skus,
                    )
                    mc_sup = {
                        sku: predict_sku_origin_outputs(
                            sup_artifacts[sku],
                            features_proj,
                            I_s=I_order_alloc[:, p_i],
                            K_s=base_inst.K[:, p_i],
                            b_s=base_inst.b[:, p_i],
                            h_s=base_inst.h[:, p_i],
                            t_s=base_inst.t[:, p_i],
                        )["c_hat_flat"]
                        for p_i, sku in enumerate(base_inst.skus)
                    }
                    mc_dfl = {
                        sku: predict_sku_origin_outputs(
                            dfl_artifacts[sku],
                            features_proj,
                            I_s=I_order_alloc[:, p_i],
                            K_s=base_inst.K[:, p_i],
                            b_s=base_inst.b[:, p_i],
                            h_s=base_inst.h[:, p_i],
                            t_s=base_inst.t[:, p_i],
                        )["c_hat_flat"]
                        for p_i, sku in enumerate(base_inst.skus)
                    }
                    sup_pmfs_proj = {
                        sku: predict_sku_origin_outputs(
                            sup_artifacts[sku],
                            features_proj,
                            I_order_alloc[:, p_i],
                            base_inst.K[:, p_i],
                            base_inst.b[:, p_i],
                            base_inst.h[:, p_i],
                            base_inst.t[:, p_i],
                        )["pmf"]
                        for p_i, sku in enumerate(base_inst.skus)
                    }
                    dfl_pmfs_proj = {
                        sku: predict_sku_origin_outputs(
                            dfl_artifacts[sku],
                            features_proj,
                            I_order_alloc[:, p_i],
                            base_inst.K[:, p_i],
                            base_inst.b[:, p_i],
                            base_inst.h[:, p_i],
                            base_inst.t[:, p_i],
                        )["pmf"]
                        for p_i, sku in enumerate(base_inst.skus)
                    }
                    sup_scen_order = copula_sup.sample_scenarios(
                        sup_pmfs_proj, num_scenarios=cfg.dro.num_scenarios, seed=cfg.global_seed + o_idx + 777
                    )
                    dfl_scen_order = copula_dfl.sample_scenarios(
                        dfl_pmfs_proj, num_scenarios=cfg.dro.num_scenarios, seed=cfg.global_seed + o_idx + 777
                    )

                inst_order = AllocationInstance(
                    stores=base_inst.stores,
                    skus=base_inst.skus,
                    I=I_order_alloc.copy(),
                    W=base_inst.W,
                    K=base_inst.K,
                    C=base_inst.C,
                    b=base_inst.b,
                    h=base_inst.h,
                    t=base_inst.t,
                )

                # Order decision
                if o_idx + lead_time >= len(origin_inputs):
                    x_order = np.zeros((S, P), dtype=int)
                elif pol_name == "equal_allocation":
                    x_order = allocate_equal_shares(inst_order).x
                elif pol_name == "proportional_allocation":
                    x_order = allocate_proportional_shares(inp["trailing_signals"], inst_order).x
                elif pol_name == "supervised_pto":
                    x_order = solve_nominal_marginal_cost_milp(mc_sup, inst_order).x
                elif pol_name == "dfl_pto":
                    x_order = solve_nominal_marginal_cost_milp(mc_dfl, inst_order).x
                elif pol_name in {"supervised_dro", "supervised_stochastic"}:
                    x_order = solve_scenario_reweighting_dro_milp(
                        sup_scen_order,
                        inst_order,
                        alpha=float(sup_dro_params["alpha"]),
                        risk_weight=0.0 if pol_name == "supervised_stochastic" else float(sup_dro_params["risk_weight"]),
                    ).x
                else:  # dfl_dro
                    x_order = solve_scenario_reweighting_dro_milp(
                        dfl_scen_order,
                        inst_order,
                        alpha=float(dfl_dro_params["alpha"]),
                        risk_weight=0.0 if pol_name == "dfl_stochastic" else float(dfl_dro_params["risk_weight"]),
                    ).x

                # Physical state transition with strict conservation verified
                sim.step(order_x=x_order, realized_demand_D=D_real, period_idx=o_idx)

            sim_summary = sim.get_summary()
            mean_pol_cost = float(sim_summary["mean_cost_mu"])
            raw_regret = float(mean_pol_cost - oracle_mp_cost)
            if raw_regret < -1e-5:
                raise AssertionError(f"Negative rolling regret: {raw_regret}")
            carryover_rows.append(
                {
                    "lead_time_weeks": int(lead_time),
                    "policy": pol_name,
                    "mean_cost_mu": mean_pol_cost,
                    "rolling_oracle_regret_mu": raw_regret,
                    "mean_shortage_cost_mu": float(sim_summary["mean_shortage_cost_mu"]),
                    "mean_holding_cost_mu": float(sim_summary["mean_holding_cost_mu"]),
                    "fill_rate": float(sim_summary["fill_rate"]),
                    "ending_inventory_units": int(sim_summary["ending_store_inventory_units"]),
                    "ending_in_transit_units": int(sim_summary["ending_in_transit_units"]),
                    "is_conserved": bool(sim_summary["is_conserved"]),
                    "data_mode": cfg.data_mode,
                }
            )

    carryover_df = pd.DataFrame(carryover_rows)

    # P1.3: Full-Horizon Cost–Risk–Service-Level Trade-Off Frontier across test origins
    frontier_rows: list[dict[str, Any]] = []
    probs_orig = np.full(len(origin_inputs), 1.0 / max(1, len(origin_inputs)), dtype=float)
    alpha_sweep = [a for a in cfg.dro.alpha_grid if a in {0.50, 0.75, 0.90, 0.95}] or [0.90]
    rw_sweep = cfg.dro.risk_weight_grid

    for fam_name, scen_key in [("dfl_dro", "dfl_scen"), ("supervised_dro", "sup_scen")]:
        for a_val in alpha_sweep:
            for rw_val in rw_sweep:
                orig_costs: list[float] = []
                orig_shortages: list[float] = []
                orig_holdings: list[float] = []
                orig_fills: list[float] = []
                orig_stockout_freq: list[float] = []
                orig_mod_exp: list[float] = []
                orig_mod_cvar: list[float] = []

                for inp in origin_inputs:
                    D_r = inp["D_realized"]
                    sc_arr = inp[scen_key]
                    dro_sol = solve_scenario_reweighting_dro_milp(
                        sc_arr, base_inst, alpha=float(a_val), risk_weight=float(rw_val)
                    )
                    cb = evaluate_exact_cost(dro_sol.x, D_r, base_inst)
                    orig_costs.append(cb.total_cost_mu)
                    orig_shortages.append(cb.shortage_cost_mu)
                    orig_holdings.append(cb.holding_cost_mu)
                    orig_fills.append(cb.fill_rate)
                    stockout_rate = float(np.mean((D_r - base_inst.I - dro_sol.x) > 0))
                    orig_stockout_freq.append(stockout_rate)
                    orig_mod_exp.append(dro_sol.model_expected_cost_mu)
                    orig_mod_cvar.append(dro_sol.model_cvar_cost_mu)

                c_arr = np.array(orig_costs, dtype=float)
                real_cvar90, _ = exact_ambiguity_set_worst_case_loss(c_arr, probs_orig, alpha=0.90)
                frontier_rows.append(
                    {
                        "policy_family": fam_name,
                        "alpha": float(a_val),
                        "risk_weight": float(rw_val),
                        "realized_mean_cost_mu": float(np.mean(c_arr)),
                        "realized_cvar90_cost_mu": float(real_cvar90),
                        "mean_fill_rate": float(np.mean(orig_fills)),
                        "service_level_no_stockout_rate": float(1.0 - np.mean(orig_stockout_freq)),
                        "mean_shortage_cost_mu": float(np.mean(orig_shortages)),
                        "mean_holding_cost_mu": float(np.mean(orig_holdings)),
                        "model_expected_cost_mean_mu": float(np.mean(orig_mod_exp)),
                        "model_cvar_cost_mean_mu": float(np.mean(orig_mod_cvar)),
                        "data_mode": cfg.data_mode,
                    }
                )

    frontier_df = pd.DataFrame(frontier_rows)
    return multi_seed_df, carryover_df, frontier_df


def run_rolling_backtest(
    cfg: ExperimentConfig,
    weekly_df: pd.DataFrame,
    inv_df: pd.DataFrame,
    sup_artifacts: dict[str, SKUForecastArtifact],
    dfl_artifacts: dict[str, SKUForecastArtifact],
    lineage: dict[str, Any],
) -> dict[str, Any]:
    """Execute rolling-origin out-of-sample evaluation across all 7 required policies (`AGENTS.md` §8)."""
    backtest_started = time.perf_counter()
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

    timing_rows = []
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

        inference_started = time.perf_counter()
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

        inference_seconds = time.perf_counter() - inference_started
        sample_started = time.perf_counter()
        # 4. Sample common-random-number joint demand scenarios for DRO policies
        # Both use identical `origin_seed` so the underlying uniform rank draws U are identical
        sup_scenarios = copula_sup.sample_scenarios(
            sup_pmfs, num_scenarios=cfg.dro.num_scenarios, seed=origin_seed
        )
        dfl_scenarios = copula_dfl.sample_scenarios(
            dfl_pmfs, num_scenarios=cfg.dro.num_scenarios, seed=origin_seed
        )

        sample_seconds = time.perf_counter() - sample_started
        solve_started = time.perf_counter()
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
        res_sup_stochastic = solve_scenario_reweighting_dro_milp(sup_scenarios, inst, risk_weight=0)
        res_dfl_stochastic = solve_scenario_reweighting_dro_milp(dfl_scenarios, inst, risk_weight=0)
        res_oracle = solve_clairvoyant_oracle(D_realized, inst)
        oracle_cost_mu = evaluate_exact_cost(res_oracle.x, D_realized, inst).total_cost_mu

        timing_rows.append(dict(origin=origin, inference_seconds=inference_seconds, scenario_seconds=sample_seconds, solve_group_seconds=time.perf_counter()-solve_started))
        solution_gaps = {}
        policy_solutions: dict[str, tuple[np.ndarray, str, bool, float, float, float]] = {}
        # Compute model-implied expected cost & CVaR for each policy on reference scenarios
        for pol_name, sol_obj, scen_ref, alpha_ref in [
            ("equal_allocation", res_equal, sup_scenarios, cfg.dro.alpha),
            ("proportional_allocation", res_prop, sup_scenarios, cfg.dro.alpha),
            ("supervised_pto", res_sup_pto, sup_scenarios, cfg.dro.alpha),
            ("supervised_dro", res_sup_dro, sup_scenarios, sup_alpha_star),
            ("dfl_pto", res_dfl_pto, dfl_scenarios, cfg.dro.alpha),
            ("dfl_dro", res_dfl_dro, dfl_scenarios, dfl_alpha_star),
            ("supervised_stochastic", res_sup_stochastic, sup_scenarios, cfg.dro.alpha),
            ("dfl_stochastic", res_dfl_stochastic, dfl_scenarios, cfg.dro.alpha),
            ("clairvoyant_oracle", res_oracle, sup_scenarios, cfg.dro.alpha),
        ]:
            if sol_obj.solver_status not in {"optimal", "heuristic_feasible"} or not sol_obj.is_feasible:
                raise RuntimeError(f"Rejected scientific policy solve: {pol_name}")
            solution_gaps[pol_name] = float(sol_obj.mip_gap)
            scen_stats = evaluate_allocation_on_scenarios(
                sol_obj.x, sup_scenarios, inst, alpha=cfg.dro.alpha
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

        for pol_name in EVALUATED_POLICIES:
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
                    "model_risk_alpha": cfg.dro.alpha,
                    "model_evaluation_distribution": "supervised_reference_common_to_all_policies",
                    "solver_mip_gap": solution_gaps[pol_name],
                    "model_expected_cost_mu": mod_exp,
                    "model_cvar_cost_mu": mod_cvar,
                    "solver_status": status_str,
                    "runtime_seconds": rt_sec,
                    "data_mode": cfg.data_mode,
                }
            )

            if pol_name == "supervised_dro":
                pol_scen_prof = f"alpha={sup_alpha_star:.2f}_rw={sup_rw_star:.2f}"
            elif pol_name == "dfl_dro":
                pol_scen_prof = f"alpha={dfl_alpha_star:.2f}_rw={dfl_rw_star:.2f}"
            else:
                pol_scen_prof = "nominal_or_heuristic"

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
                            "scenario_profile": pol_scen_prof,
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

    # Priorities 4 & 7: Multi-seed stability across `cfg.evaluation.seeds`, rolling carryover/lead-time, and cost-risk-service frontier
    multi_seed_df, carryover_df, frontier_df = run_multi_seed_and_carryover_experiments(
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

    from inventory_di.evaluation.sensitivity import distribution_sensitivity
    distribution_sens = distribution_sensitivity(cfg, weekly_df, inst, dfl_artifacts, {"alpha": dfl_alpha_star, "risk_weight": dfl_rw_star})
    art_dir = Path(cfg.data.artifacts_dir)
    save_parquet(distribution_sens, art_dir / "distribution_sensitivity.parquet")
    alloc_sha = save_parquet(alloc_df, art_dir / "allocation_results.parquet")
    metrics_sha = save_parquet(metrics_df, art_dir / "backtest_metrics.parquet")
    sens_sha = save_parquet(sensitivity_df, art_dir / "risk_sensitivity.parquet")
    summary_sha = save_parquet(summary_df, art_dir / "summary_metrics.parquet")
    multiseed_sha = save_parquet(multi_seed_df, art_dir / "multi_seed_summary.parquet")
    carryover_sha = save_parquet(carryover_df, art_dir / "carryover_leadtime_metrics.parquet")
    frontier_sha = save_parquet(frontier_df, art_dir / "cost_risk_service_frontier.parquet")
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
    for pol in EVALUATED_POLICIES:
        solver_statuses_by_pol[pol] = (
            metrics_df[metrics_df["policy"] == pol].sort_values("origin")["solver_status"].tolist()
        )

    run_manifest = {
        "run_id": cfg.run_id,
        "git_commit": get_git_commit(),
        "source": source_provenance(),
        "effective_config": cfg.model_dump(),
        "evaluation_disclosure": "Retrospective corrected rerun on previously inspected test periods; not a fresh confirmatory holdout",
        "python_version": get_environment_snapshot()["python_version"],
        "platform": get_environment_snapshot()["platform"],
        "global_seed": cfg.global_seed,
        "config_path": cfg.config_path,
        "config_sha256": cfg.config_sha256,
        "data_mode": cfg.data_mode,
        "demand_regime": getattr(cfg.data, "demand_regime", "regular"),
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
        "policies_evaluated": EVALUATED_POLICIES,
        "policy_origin_Fingerprints": {pol: combined_fp for pol in EVALUATED_POLICIES},
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
            "cost_risk_service_frontier_parquet": frontier_sha,
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
        "independent_training_seeds": bool(getattr(cfg.evaluation, "independent_training_seeds", True)),
        "evaluated_policies": EVALUATED_POLICIES,
        "origin_evaluations": origin_env_records,
        **bootstrap_meta,
    }

    run_manifest["timing"] = dict(total_backtest_seconds=time.perf_counter()-backtest_started, origins=timing_rows)
    save_json(run_manifest, art_dir / "run_manifest.json")
    return {
        "allocation_results": alloc_df,
        "backtest_metrics": metrics_df,
        "summary_metrics": summary_df,
        "risk_sensitivity": sensitivity_df,
        "cost_risk_service_frontier": frontier_df,
        "run_manifest": run_manifest,
    }
