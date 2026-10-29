"""Streamlit Decision-Experiment Dashboard (`app/streamlit_app.py`) (`AGENTS.md` §10).

Pages:
- Page A — Problem & Data
- Page B — Models & Diagnostics
- Page C — Allocation Results
- Page D — Risk & Robustness (interactive local re-solve using cached model outputs; no retraining in UI)
- Page E — Evidence & Business Impact
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from inventory_di.config import ExperimentConfig, load_config
from inventory_di.evaluation.backtest import build_allocation_instance_from_config_df
from inventory_di.optimization.baselines import solve_clairvoyant_oracle
from inventory_di.optimization.core import (
    evaluate_exact_cost,
    solve_service_level_constrained_milp,
)
from inventory_di.optimization.dro import solve_scenario_reweighting_dro_milp
from inventory_di.provenance import validate_stage
from inventory_di.visualization.plots import (
    plot_cost_risk_service_tradeoff,
    plot_forecast_quantiles,
    plot_historical_demand_by_split,
    plot_policy_cost_comparison,
    plot_risk_cost_pareto,
    plot_store_allocations_comparison,
    plot_training_curves,
)

REGISTERED_PAGES = [
    "Page A — Problem & Data",
    "Page B — Models & Diagnostics",
    "Page C — Allocation Results",
    "Page D — Risk & Robustness",
    "Page E — Evidence & Business Impact",
]


def load_cached_bundle(config_path: str):
    cfg = load_config(config_path)
    paths = [Path(config_path)] + list(Path(cfg.data.processed_dir).glob("*")) + list(Path(cfg.data.artifacts_dir).glob("*"))
    signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(paths) if p.is_file())
    return _load_cached_bundle(config_path, signature)


@st.cache_data(show_spinner=False)
def _load_cached_bundle(config_path: str, signature: tuple) -> dict[str, Any]:
    """Load cached Parquet and JSON artifacts offline without retraining."""
    cfg: ExperimentConfig = load_config(config_path)
    proc_dir = Path(cfg.data.processed_dir)
    art_dir = Path(cfg.data.artifacts_dir)

    required_files = [
        proc_dir / "weekly_demand.parquet",
        proc_dir / "inventory_config.parquet",
        proc_dir / "lineage_manifest.json",
        art_dir / "allocation_results.parquet",
        art_dir / "backtest_metrics.parquet",
        art_dir / "summary_metrics.parquet",
        art_dir / "risk_sensitivity.parquet",
        art_dir / "training_diagnostics.json",
        art_dir / "cached_origin_bundles.json",
        art_dir / "run_manifest.json",
    ]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        return {"status": "MISSING", "missing_files": missing, "cfg": cfg}

    try:
        validate_stage(cfg, "backtest", check_source=False)
    except (ValueError, KeyError) as exc:
        return {"status": "INVALID", "missing_files": [str(exc)], "cfg": cfg}
    return {
        "status": "OK",
        "cfg": cfg,
        "weekly_demand": pd.read_parquet(proc_dir / "weekly_demand.parquet"),
        "inventory_config": pd.read_parquet(proc_dir / "inventory_config.parquet"),
        "lineage": json.loads((proc_dir / "lineage_manifest.json").read_text(encoding="utf-8")),
        "allocation_results": pd.read_parquet(art_dir / "allocation_results.parquet"),
        "backtest_metrics": pd.read_parquet(art_dir / "backtest_metrics.parquet"),
        "summary_metrics": pd.read_parquet(art_dir / "summary_metrics.parquet"),
        "risk_sensitivity": pd.read_parquet(art_dir / "risk_sensitivity.parquet"),
        "training_diagnostics": json.loads(
            (art_dir / "training_diagnostics.json").read_text(encoding="utf-8")
        ),
        "cached_origin_bundles": json.loads(
            (art_dir / "cached_origin_bundles.json").read_text(encoding="utf-8")
        ),
        "run_manifest": json.loads((art_dir / "run_manifest.json").read_text(encoding="utf-8")),
        "multi_seed_summary": (
            pd.read_parquet(art_dir / "multi_seed_summary.parquet")
            if (art_dir / "multi_seed_summary.parquet").exists()
            else None
        ),
        "carryover_leadtime": (
            pd.read_parquet(art_dir / "carryover_leadtime_metrics.parquet")
            if (art_dir / "carryover_leadtime_metrics.parquet").exists()
            else None
        ),
        "cost_risk_service_frontier": (
            pd.read_parquet(art_dir / "cost_risk_service_frontier.parquet")
            if (art_dir / "cost_risk_service_frontier.parquet").exists()
            else None
        ),
    }


def render_provenance_banner(cfg: ExperimentConfig) -> None:
    """Render unmistakable provenance badge and economic currency notice."""
    if cfg.data_mode == "synthetic_demo":
        badge_text = "DEMO / SYNTHETIC"
        desc = (
            "Synthetic Demo Mode (`data_mode='synthetic_demo'`). Outputs are from a controlled generative "
            "fixture for engineering verification and demonstration — NEVER interpret as empirical Walmart M5 results."
        )
        st.warning(f"**Provenance Badge: `{badge_text}`** — {desc}")
    else:
        badge_text = "M5 sales proxy + simulated inventory"
        desc = (
            "M5 Sales Proxy Mode (`data_mode='m5_sales_proxy'`). Historical observed Walmart M5 HOUSEHOLD unit sales "
            "are used as an uncensored demand proxy combined with documented synthetic inventory/operating economics."
        )
        st.info(f"**Provenance Badge: `{badge_text}`** — {desc}")


def render_page_a(bundle: dict[str, Any]) -> None:
    st.header("Page A — Problem & Data")
    weekly_df: pd.DataFrame = bundle["weekly_demand"]
    inv_df: pd.DataFrame = bundle["inventory_config"]
    lineage: dict[str, Any] = bundle["lineage"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stores in Cohort", int(weekly_df["store_id"].nunique()))
    col2.metric("HOUSEHOLD SKUs", int(weekly_df["sku_id"].nunique()))
    col3.metric("Weekly Origins", int(weekly_df["forecast_origin"].nunique()))
    col4.metric("Economic Units", "MU (Synthetic)")

    st.markdown(
        """
        ### Multi-Store Constrained Stock Allocation Problem
        At each weekly `forecast_origin`, a central warehouse holds limited supply $W_p$ (units) for each SKU $p \\in P$
        and must allocate integer shipments $x_{sp} \\in \\mathbb{Z}_{\\ge 0}$ across stores $s \\in S$ subject to
        per-store–SKU caps $K_{sp}$ and total store capacity $C_s$. Realized one-week cost is measured in generic
        Monetary Units (**MU**):
        $$C(x, D) = \\sum_{s,p} \\left[ b_{sp}\\max(D_{sp} - I_{sp} - x_{sp}, 0) + h_{sp}\\max(I_{sp} + x_{sp} - D_{sp}, 0) + t_{sp} x_{sp} \\right]$$
        """
    )

    skus = sorted(weekly_df["sku_id"].unique())
    selected_sku = st.selectbox("Select SKU for Historical Demand & Split Inspection", skus, key="page_a_sku")
    st.plotly_chart(plot_historical_demand_by_split(weekly_df, selected_sku), width="stretch")

    st.subheader("Synthetic Inventory & Operating Economics Contract (`inventory_config.parquet`)")
    st.dataframe(inv_df, width="stretch")

    with st.expander("Data Quality, Temporal Split & Leakage-Audit Manifest"):
        st.json(lineage)


def render_page_b(bundle: dict[str, Any]) -> None:
    st.header("Page B — Models & Diagnostics")
    diag: dict[str, Any] = bundle["training_diagnostics"]
    origins_bundle: dict[str, Any] = bundle["cached_origin_bundles"]

    st.markdown(
        """
        ### Supervised PTO vs. Decision-Focused Learning (DFL via PyEPO `SPOPlus`)
        * **Chronological Whole-Origin Split:** `train` $\\rightarrow$ `val` (temperature calibration, checkpoint/lambda selection) $\\rightarrow$ `test` (out-of-sample evaluation).
        * **Discrete PMF with Overflow Tail:** Both models share the exact same neural architecture predicting $p(d \\mid \\text{features})$ over $d \\in \\{0, \\dots, H\\}$ plus an explicit `overflow > H` bin.
        * **Survival-to-Marginal-Cost Mapping:** Survival probabilities $q[s,k] = P(D_s \\ge I_s + k \\mid \\text{features})$ map differentiably to marginal unit costs $\\hat{c}[s,k] = t_s + h_s - (b_s + h_s)q[s,k]$.
        * **Genuine DFL Objective:** Trains end-to-end with $\\mathcal{L} = \\text{NLL} + \\lambda_{\\text{DFL}} \\cdot \\mathcal{L}_{\\text{SPO}+}(\\hat{c}, c_{\\text{true}}, y^*, z^*)$ over the joint multi-SKU fixed-feasible marginal-unit model. The SPO+ term is normalized by the number of marginal-unit coefficients.
        """
    )

    weekly = bundle["weekly_demand"]
    for column, split in zip(st.columns(3), ["train", "val", "test"]):
        block = weekly[weekly.split == split]
        column.metric(f"{split.capitalize()} origins", block.forecast_origin.nunique())
        column.caption(f"{block.target_start.min()} to {block.target_end.max()}")
    st.caption("Chronological flow: training → validation selection/calibration → retrospective test evaluation. All stores/SKUs at an origin share its split.")
    skus = sorted(diag["dfl"].keys())
    col_s, col_o = st.columns(2)
    selected_sku = col_s.selectbox("Select SKU", skus, key="page_b_sku")
    selected_origin = col_o.selectbox(
        "Select Test Forecast Origin", sorted(origins_bundle.keys()), key="page_b_origin"
    )

    sup_cal = diag["supervised"][selected_sku]["calibration"]
    dfl_cal = diag["dfl"][selected_sku]["calibration"]

    # Calibration warning gate (AGENTS.md §10 Page B)
    sup_cov = float(sup_cal.get("coverage_p10_p90", 0.80))
    dfl_cov = float(dfl_cal.get("coverage_p10_p90", 0.80))
    uniform_nll = float(np.log(diag["support_h"] + 2))
    poor_nll = any(float(cal["nll"]) > uniform_nll for cal in [sup_cal, dfl_cal])
    poor_tail = any(abs(float(cal["realized_overflow_rate"]) - float(cal["predicted_overflow_prob"])) > .15 for cal in [sup_cal, dfl_cal])
    if sup_cov < 0.60 or dfl_cov < 0.60 or poor_nll or poor_tail:
        st.warning(
            "**Diagnostic warning:** Coverage is below 60%, NLL is worse than a uniform-bin reference, or overflow probability differs from validation frequency by over 15 percentage points. Interpret forecasts and allocations cautiously."
        )
    else:
        st.success(
            f"**Validation Diagnostics:** Temperature scaling applied (`split='val'`). "
            f"P10–P90 coverage: Supervised = `{100.0 * sup_cov:.1f}%`, DFL = `{100.0 * dfl_cov:.1f}%`."
        )

    st.caption("These descriptive validation checks do not guarantee calibration on future demand.")
    diag_table = pd.DataFrame(
        [
            {
                "Model Family": "Supervised (NLL-only)",
                "Validation NLL": round(sup_cal["nll"], 4),
                "Validation MAE (Units)": round(sup_cal["mae"], 3),
                "Validation WAPE": round(sup_cal["wape"], 3),
                "P10-P90 Coverage": f"{100.0 * sup_cal['coverage_p10_p90']:.1f}%",
                "Realized Overflow Rate (>H)": f"{100.0 * sup_cal['realized_overflow_rate']:.2f}%",
                "Calibrated Temp": round(sup_cal["optimal_temperature"], 2),
                "SPO+ Grad Norm": 0.0,
            },
            {
                "Model Family": f"DFL (NLL + {diag['dfl'][selected_sku]['lambda_dfl']} * SPO+)",
                "Validation NLL": round(dfl_cal["nll"], 4),
                "Validation MAE (Units)": round(dfl_cal["mae"], 3),
                "Validation WAPE": round(dfl_cal["wape"], 3),
                "P10-P90 Coverage": f"{100.0 * dfl_cal['coverage_p10_p90']:.1f}%",
                "Realized Overflow Rate (>H)": f"{100.0 * dfl_cal['realized_overflow_rate']:.2f}%",
                "Calibrated Temp": round(dfl_cal["optimal_temperature"], 2),
                "SPO+ Grad Norm": round(
                    diag["dfl"][selected_sku]["gradient_check"]["grad_norm_pure_spoplus"], 4
                ),
            },
        ]
    )
    st.dataframe(diag_table, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_training_curves(diag["dfl"][selected_sku]["history"], selected_sku, "DFL (NLL + SPO+)"),
            width="stretch",
        )
    with c2:
        q_df = pd.DataFrame(origins_bundle[selected_origin]["forecast_quantiles"])
        st.plotly_chart(plot_forecast_quantiles(q_df, selected_sku), width="stretch")

    if diag.get("lambda_dfl_ablation"):
        st.subheader("Validation Ablation Across `lambda_DFL`")
        st.dataframe(pd.DataFrame(diag["lambda_dfl_ablation"]), width="stretch")


def render_page_c(bundle: dict[str, Any]) -> None:
    st.header("Page C — Allocation Results")
    alloc_df: pd.DataFrame = bundle["allocation_results"]
    metrics_df: pd.DataFrame = bundle["backtest_metrics"]

    origins = sorted(alloc_df["forecast_origin"].unique())
    skus = sorted(alloc_df["sku_id"].unique())
    c1, c2 = st.columns(2)
    sel_origin = c1.selectbox("Select Test Forecast Origin", origins, key="page_c_origin")
    sel_sku = c2.selectbox("Select SKU", skus, key="page_c_sku")

    sub_alloc = alloc_df[alloc_df["forecast_origin"] == sel_origin].copy()
    sub_metrics = metrics_df[metrics_df["origin"] == sel_origin].copy()

    all_feasible = bool(sub_alloc["is_feasible"].all())
    all_optimal = bool(sub_alloc["solver_status"].isin(["optimal", "heuristic_feasible"]).all())
    if all_feasible and all_optimal:
        st.success(
            f"**Allocation acceptance (`{sel_origin}`):** MILPs optimal; heuristic baselines feasible. Warehouse supply ($W_p$), "
            "per-store-SKU caps ($K_{sp}$), and total store capacities ($C_s$) verified."
        )
    else:
        st.error(f"**Solver Warning (`{sel_origin}`):** Non-optimal or infeasible allocation detected!")

    st.plotly_chart(plot_store_allocations_comparison(sub_alloc, sel_sku), width="stretch")

    st.subheader(f"Origin `{sel_origin}` Policy Cost & Fulfillment Summary (MU & Units)")
    st.dataframe(
        sub_metrics[
            [
                "policy",
                "total_cost_mu",
                "shortage_cost_mu",
                "holding_cost_mu",
                "shipping_cost_mu",
                "filled_units",
                "unmet_units",
                "fill_rate",
                "oracle_regret_mu",
                "solver_status",
            ]
        ],
        width="stretch",
    )

    st.subheader(f"Store-Level Shipment Table (`{sel_origin}`, `{sel_sku}`)")
    ob = bundle["cached_origin_bundles"][sel_origin]
    for family in sub_alloc.policy.unique():
        scenarios = np.asarray(ob["dfl_scenarios"] if family.startswith("dfl") else ob["sup_scenarios"])
        for si, store in enumerate(sorted(bundle["inventory_config"].store_id.unique())):
            for pi, sku in enumerate(sorted(bundle["inventory_config"].sku_id.unique())):
                mask = (sub_alloc.policy == family) & (sub_alloc.store_id == store) & (sub_alloc.sku_id == sku)
                row = sub_alloc.loc[mask].iloc[0]
                available = row.opening_stock_units + row.allocated_units
                sub_alloc.loc[mask, "model_expected_shortage_units"] = np.maximum(scenarios[:,si,pi]-available, 0).mean()
                sub_alloc.loc[mask, "model_expected_excess_units"] = np.maximum(available-scenarios[:,si,pi], 0).mean()
    st.caption("Expected excess/shortage use each family's cached scenarios; they differ from realized demand.")
    sku_table = sub_alloc[sub_alloc["sku_id"] == sel_sku][
        [
            "policy",
            "store_id",
            "sku_id",
            "opening_stock_units",
            "warehouse_supply_units",
            "store_sku_unit_cap",
            "store_total_capacity_units",
            "model_expected_shortage_units",
            "model_expected_excess_units",
            "allocated_units",
            "realized_demand_units",
            "is_feasible",
            "solver_status",
        ]
    ]
    st.dataframe(sku_table, width="stretch")

    csv_bytes = sub_alloc.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Allocation Results CSV for Selected Origin",
        data=csv_bytes,
        file_name=f"allocations_{sel_origin}.csv",
        mime="text/csv",
    )


def render_page_d(bundle: dict[str, Any]) -> None:
    st.header("Page D — Risk & Robustness")
    inv_df: pd.DataFrame = bundle["inventory_config"]
    sens_df: pd.DataFrame = bundle["risk_sensitivity"]
    origins_bundle: dict[str, Any] = bundle["cached_origin_bundles"]

    st.markdown(
        """
        ### Explicit Scenario-Probability Reweighting DRO ($\\text{CVaR}_\\alpha$ Ambiguity Set)
        For finite joint demand scenarios $D^{(1)}, \\dots, D^{(N)}$ with nominal probabilities $p_j = 1/N$
        and scenario losses $L_j(x) = C(x, D^{(j)})$, our DRO model optimizes against the explicit
        **probability-reweighting ambiguity set**:
        $$\\mathcal{P}_\\alpha = \\left\\{ w \\in \\mathbb{R}^N_{\\ge 0} : \\sum_{j=1}^N w_j = 1, \\; 0 \\le w_j \\le \\frac{p_j}{1 - \\alpha} \\right\\}, \\quad \\alpha \\in [0, 1)$$
        By Rockafellar–Uryasev duality, $\\max_{w \\in \\mathcal{P}_\\alpha} \\sum_{j=1}^N w_j L_j(x) = \\text{CVaR}_\\alpha(L(x))$.
        The risk trade-off objective minimizes:
        $$\\min_{x \\in \\mathcal{X}} \\; (1 - \\text{risk\\_weight}) \\sum_{j=1}^N p_j L_j(x) + \\text{risk\\_weight} \\cdot \\text{CVaR}_\\alpha(L(x))$$
        *(Note: This is probability-reweighting / CVaR DRO, **not** Wasserstein DRO. At `risk_weight = 0` or `alpha = 0`, it reduces to risk-neutral expected cost minimization. UI controls re-solve the MILP locally from cached scenarios without retraining any neural network.)*
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    sel_origin = c1.selectbox(
        "Test Origin", sorted(origins_bundle.keys()), index=0, key="page_d_origin"
    )
    sel_family = c2.selectbox(
        "Scenario Distribution Source",
        ["dfl_dro", "supervised_dro"],
        index=0,
        key="page_d_family",
    )
    selected = bundle["run_manifest"]["validation_dro_tuning"][sel_family + "_selected"]
    defaults_for = (bundle["cfg"].run_id, sel_family)
    if st.session_state.get("risk_defaults_for") != defaults_for:
        st.session_state["page_d_alpha"] = selected["alpha"]
        st.session_state["page_d_rw"] = selected["risk_weight"]
        st.session_state["risk_defaults_for"] = defaults_for
    st.caption("Defaults were selected on validation. Any control changes are exploratory counterfactuals, not new test evidence.")
    sel_alpha = c3.select_slider(
        "Tail Ambiguity Parameter (alpha)",
        options=[0.0, 0.50, 0.75, 0.90, 0.95],
        key="page_d_alpha",
    )
    sel_rw = c4.select_slider(
        "Tail Risk Weight (risk_weight)",
        options=[0.0, 0.25, 0.50, 0.75, 1.0],
        key="page_d_rw",
    )

    st.markdown("#### Operational What-If Simulation Controls (Stock, Display/Price Lift & Priority)")
    w1, w2, w3, w4 = st.columns(4)
    sel_supply_mult = w1.select_slider(
        "Warehouse Supply Scarcity Multiplier",
        options=[0.6, 0.8, 1.0, 1.2],
        value=1.0,
        key="page_d_supply",
    )
    sel_open_stock = w2.select_slider(
        "Opening Store Stock Override (I_sp units)",
        options=[0, 1, 2, 3, 5],
        value=0,
        key="page_d_open_stock",
    )
    sel_promo_lift = w3.select_slider(
        "Pricing / Display Demand Uplift (%)",
        options=[-20, -10, 0, 10, 25, 50],
        value=0,
        key="page_d_promo_lift",
    )
    sel_shortage_mult = w4.select_slider(
        "Shortage Penalty Priority Multiplier (b_sp)",
        options=[0.5, 1.0, 1.5, 2.0],
        value=1.0,
        key="page_d_shortage_mult",
    )

    # Re-solve locally using cached scenarios without model retraining.
    ob = origins_bundle[sel_origin]
    demand_scale = 1.0 + (float(sel_promo_lift) / 100.0)
    scen_base = np.array(
        ob["dfl_scenarios"] if sel_family == "dfl_dro" else ob["sup_scenarios"], dtype=float
    )
    scen_arr = np.maximum(0, np.rint(scen_base * demand_scale)).astype(int)
    D_real_base = np.array(ob["D_realized"], dtype=float)
    D_real = np.maximum(0, np.rint(D_real_base * demand_scale)).astype(int)
    inst_mod = build_allocation_instance_from_config_df(
        inv_df,
        supply_multiplier=float(sel_supply_mult),
        opening_stock_override=int(sel_open_stock),
        shortage_penalty_multiplier=float(sel_shortage_mult),
    )

    dro_live = solve_scenario_reweighting_dro_milp(
        scen_arr, inst_mod, alpha=float(sel_alpha), risk_weight=float(sel_rw)
    )
    real_live = evaluate_exact_cost(dro_live.x, D_real, inst_mod)
    oracle_live = solve_clairvoyant_oracle(D_real, inst_mod)
    oracle_live_cost = evaluate_exact_cost(oracle_live.x, D_real, inst_mod).total_cost_mu

    st.subheader("Strictly Separated Cost Metrics (Never Conflated)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "1. Model-Implied Expected Cost (MU)",
        f"{dro_live.model_expected_cost_mu:.2f} MU",
        help="Average cost across the N sampled demand scenarios under nominal weights p_j = 1/N.",
    )
    m2.metric(
        f"2. Model-Implied CVaR_{sel_alpha:.2f} (MU)",
        f"{dro_live.model_cvar_cost_mu:.2f} MU",
        help="Worst-case expected cost over the probability-reweighting ambiguity set P_alpha.",
    )
    m3.metric(
        ("3. Simulated Uplift Test Cost (MU)" if sel_promo_lift != 0
         else "3. Realized Historical Test Cost (MU)"),
        f"{real_live.total_cost_mu:.2f} MU",
        help="Cost on the held-out sales proxy; with demand uplift, the target is a synthetic intervention.",
    )
    m4.metric(
        "Solver & Constraint Status",
        f"{dro_live.solver_status} ({100.0 * real_live.fill_rate:.1f}% fill)",
        help=f"Feasible: {dro_live.is_feasible}, Oracle Regret: {real_live.total_cost_mu - oracle_live_cost:.2f} MU",
    )

    with st.expander("Inspect Live Simulated Store-by-SKU Shipment Plan & Download CSV", expanded=False):
        sim_rows = []
        for s_i, s_id in enumerate(inst_mod.stores):
            for p_i, p_id in enumerate(inst_mod.skus):
                sim_rows.append(
                    {
                        "store_id": s_id,
                        "sku_id": p_id,
                        "opening_stock_units": int(inst_mod.I[s_i, p_i]),
                        "warehouse_supply_units": int(inst_mod.W[p_i]),
                        "store_sku_unit_cap": int(inst_mod.K[s_i, p_i]),
                        "allocated_units": int(dro_live.x[s_i, p_i]),
                        "simulated_realized_demand_units": int(D_real[s_i, p_i]),
                        "unit_shortage_penalty_mu": float(inst_mod.b[s_i, p_i]),
                    }
                )
        sim_df = pd.DataFrame(sim_rows)
        st.dataframe(sim_df, width="stretch")
        st.download_button(
            label="Download What-If Simulated Shipment Plan (Excel-Compatible CSV)",
            data=sim_df.to_csv(index=False).encode("utf-8"),
            file_name=f"what_if_simulation_{sel_origin}.csv",
            mime="text/csv",
        )

    st.subheader("Service-Level Constrained Allocation (Retail Decision Tool)")
    st.markdown(
        "Evaluate required inventory allocations and physical stock sufficiency under hard service-level requirements. "
        "MR.DIY category managers can prescribe a target fill rate $\\beta^*$, and the MILP determines the cost-minimal "
        "allocation under sampled demand scenarios. Hard mode enforces the target when feasible and "
        "otherwise establishes maximum achievable service; soft mode permits an economic trade-off."
    )
    service_mode = st.selectbox("Service Target Mode", ["hard", "soft"], key="service_mode")
    c_sl1, c_sl2 = st.columns([2, 2])
    with c_sl1:
        sel_target_sl = st.slider(
            "Target Expected Fill Rate / Service Level",
            min_value=0.70,
            max_value=0.99,
            value=0.85,
            step=0.01,
            format="%.2f",
            help="Fraction of expected demand fulfilled: 0.85 means 85%. Hard mode enforces a feasible target; soft mode uses penalized slack.",
        )
    with c_sl2:
        shortfall_pen = st.number_input(
            "Shortfall Penalty Weight (MU/unit)",
            min_value=50.0,
            max_value=2000.0,
            value=500.0,
            step=50.0,
            help="Applies only in soft mode. A larger penalty favors service; it does not prove physical infeasibility.",
            disabled=service_mode == "hard",
        )

    sl_res = solve_service_level_constrained_milp(
        scenarios=scen_arr,
        inst=inst_mod,
        target_fill_rate=float(sel_target_sl),
        shortfall_penalty_mu=float(shortfall_pen),
        mode=service_mode,
    )

    sl1, sl2, sl3, sl4 = st.columns(4)
    sl1.metric("Target Fill Rate", f"{sl_res.target_fill_rate * 100:.1f}%")
    sl2.metric("Achieved Fill Rate", f"{sl_res.achieved_fill_rate * 100:.1f}%")
    sl3.metric("Expected Service Shortfall", f"{sl_res.shortfall_units:.1f} units")
    sl4.metric("Expected Total Cost (MU)", f"{sl_res.expected_cost_mu:.2f} MU")

    if not sl_res.is_feasible:
        st.error(f"Service-level solve failed: {sl_res.solver_status}")
    elif sl_res.shortfall_units > 1e-4:
        st.warning(
            f"**Service target missed:** This solution achieves {sl_res.achieved_fill_rate * 100:.1f}% "
            f"against a {sl_res.target_fill_rate * 100:.1f}% target, with "
            f"{sl_res.shortfall_units:.1f} expected unfilled units below target. "
            + ("This reflects supply, store capacity or allocation caps. " if service_mode == "hard"
             else "This may reflect physical constraints or the selected economic penalty. ")
            + (f"A separate solve establishes maximum achievable fill of {100*sl_res.maximum_achievable_fill_rate:.1f}%."
               if sl_res.maximum_achievable_fill_rate is not None else "Soft shortfall alone does not establish physical infeasibility.")
        )
    else:
        st.success(
            f"✅ **Target Achievable:** Target fill rate of {sl_res.target_fill_rate * 100:.1f}% is feasible within existing warehouse supply and store capacity."
        )

    with st.expander("Inspect Service-Level Shipment Plan & Download CSV", expanded=False):
        sl_rows = []
        for s_i, s_id in enumerate(inst_mod.stores):
            for p_i, p_id in enumerate(inst_mod.skus):
                sl_rows.append(
                    {
                        "store_id": s_id,
                        "sku_id": p_id,
                        "allocated_units": int(sl_res.x[s_i, p_i]),
                        "opening_stock_units": int(inst_mod.I[s_i, p_i]),
                        "warehouse_supply_units": int(inst_mod.W[p_i]),
                        "store_sku_unit_cap": int(inst_mod.K[s_i, p_i]),
                    }
                )
        sl_plan_df = pd.DataFrame(sl_rows)
        st.dataframe(sl_plan_df, width="stretch")
        st.download_button(
            label="Download Service-Level Allocation Plan (CSV)",
            data=sl_plan_df.to_csv(index=False).encode("utf-8"),
            file_name=f"service_level_plan_{sel_origin}_{int(sel_target_sl * 100)}pct.csv",
            mime="text/csv",
        )

    val_tuning = bundle.get("run_manifest", {}).get("validation_dro_tuning", {})
    if val_tuning:
        sup_sel = val_tuning.get("supervised_dro_selected", {})
        dfl_sel = val_tuning.get("dfl_dro_selected", {})
        st.info(
            f"**Validation-Selected DRO Hyperparameters (`split='val'`):** "
            f"`supervised_dro` $\\rightarrow (\\alpha^*={sup_sel.get('alpha', 0.90):.2f}, \\text{{risk\\_weight}}^*={sup_sel.get('risk_weight', 0.50):.2f})$ | "
            f"`dfl_dro` $\\rightarrow (\\alpha^*={dfl_sel.get('alpha', 0.90):.2f}, \\text{{risk\\_weight}}^*={dfl_sel.get('risk_weight', 0.50):.2f})$."
        )

    if bundle.get("cost_risk_service_frontier") is not None:
        st.subheader("Full-Horizon Cost–Risk–Service-Level Trade-Off Frontier Across Held-Out Test Origins")
        fr_df: pd.DataFrame = bundle["cost_risk_service_frontier"]
        st.plotly_chart(plot_cost_risk_service_tradeoff(fr_df), width="stretch")
        st.dataframe(fr_df, width="stretch")

    st.caption("The following precomputed sensitivity is for the first test origin and original stock, demand and costs; only its supply filter tracks the supply control. The full-horizon frontier above is descriptive and is not a hyperparameter selection set.")
    st.plotly_chart(
        plot_risk_cost_pareto(sens_df, supply_multiplier=float(sel_supply_mult)),
        width="stretch",
    )

    st.subheader("Precomputed Risk & Supply Sensitivity Sweep Table")
    st.dataframe(
        sens_df[
            (sens_df["supply_multiplier"] - float(sel_supply_mult)).abs() < 1e-6
        ].sort_values(["policy_family", "alpha", "risk_weight"]),
        width="stretch",
    )


def render_page_e(bundle: dict[str, Any]) -> None:
    st.header("Page E — Evidence & Business Impact")
    summary_df: pd.DataFrame = bundle["summary_metrics"]
    metrics_df: pd.DataFrame = bundle["backtest_metrics"]
    manifest: dict[str, Any] = bundle["run_manifest"]

    st.plotly_chart(plot_policy_cost_comparison(summary_df), width="stretch")

    st.subheader("Seven Required Policies + Two Risk-Neutral Scenario Ablations (MU)")
    st.caption(manifest.get("small_sample_caveat", ""))
    st.dataframe(
        summary_df[
            [
                "policy",
                "mean_cost_mu",
                "cost_ci95_low_mu",
                "cost_ci95_high_mu",
                "realized_cvar90_cost_mu",
                "mean_regret_mu",
                "mean_fill_rate",
                "mean_unmet_units",
                "diff_vs_equal_mu",
                "pct_vs_equal",
                "diff_vs_supervised_pto_mu",
                "pct_vs_supervised_pto",
                "mean_runtime_seconds",
                "all_optimal",
                "data_mode",
            ]
        ],
        width="stretch",
    )

    st.subheader("Per-Origin Paired Realized Total Cost (MU)")
    piv = metrics_df.pivot(index="origin", columns="policy", values="total_cost_mu")[
        summary_df["policy"].tolist()
    ]
    st.dataframe(piv, width="stretch")

    if bundle.get("multi_seed_summary") is not None:
        st.subheader("Three Matched Training Seeds (`[42, 101, 202]`; Descriptive Evidence)")
        st.caption("Each independent seed uses the same training budget, matched baseline/DFL updates and its own validation calibration and DRO selection.")
        ms_df: pd.DataFrame = bundle["multi_seed_summary"]
        st.dataframe(ms_df, width="stretch")

    if bundle.get("carryover_leadtime") is not None:
        st.subheader("Rolling Inventory Carryover & Lead-Time (`L=0` vs `L=1` Week) Evaluation")
        st.caption("Corrected carryover experiment: explicit admissions, deferred inventory and independently replayed hindsight oracle. Lead-time feature projection remains approximate; these are retrospective simulations with synthetic economics.")
        co_df: pd.DataFrame = bundle["carryover_leadtime"]
        st.dataframe(co_df, width="stretch")

    reg_p = Path("artifacts/m5_v2/summary_metrics.parquet")
    int_p = Path("artifacts/m5_intermittent_v2/summary_metrics.parquet")
    if reg_p.exists() and int_p.exists():
        st.subheader("Demand Regime Generalization: Regular vs. Intermittent M5 SKUs")
        reg_s = pd.read_parquet(reg_p)[["policy", "mean_cost_mu", "realized_cvar90_cost_mu", "mean_fill_rate"]]
        int_s = pd.read_parquet(int_p)[["policy", "mean_cost_mu", "realized_cvar90_cost_mu", "mean_fill_rate"]]
        merged_regime = reg_s.merge(int_s, on="policy", suffixes=("_regular", "_intermittent"))
        st.dataframe(merged_regime, width="stretch")

    with st.expander("Run Reproducibility Manifest, Validation DRO Tuning & Checksums"):
        st.json(manifest)


def main() -> None:
    st.set_page_config(
        page_title="Decision-Focused & Robust Inventory Allocation",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("Inventory Decision Intelligence")
    default_mode_idx = 1 if Path("artifacts/m5_v2/run_manifest.json").exists() else 0
    mode_label = st.sidebar.radio(
        "Dataset / Experiment Mode",
        options=[
            "Demo Mode (DEMO / SYNTHETIC)",
            "M5 Mode (M5 Sales Proxy)",
            "M5 Intermittent Mode (Lumpy/Intermittent SKUs)",
        ],
        index=default_mode_idx,
    )
    if "Demo" in mode_label:
        config_path = "configs/demo.yaml"
    elif "Intermittent" in mode_label:
        config_path = "configs/m5_intermittent.yaml"
    else:
        config_path = "configs/m5.yaml"

    page = st.sidebar.radio("Navigation (Pages A–E)", options=REGISTERED_PAGES, index=0)

    bundle = load_cached_bundle(config_path)
    render_provenance_banner(bundle["cfg"])

    if bundle["status"] != "OK":
        st.error(
            "### Cached Experiment Artifacts Missing or Inconsistent\n"
            "The dashboard runs strictly offline from verified precomputed artifacts. "
            "Missing files:\n"
            + "\n".join(f"- `{m}`" for m in bundle["missing_files"])
        )
        st.code(
            f"# Generate artifacts for {config_path}:\n"
            f"python -m inventory_di.cli prepare --config {config_path}\n"
            f"python -m inventory_di.cli train --config {config_path} --model all\n"
            f"python -m inventory_di.cli backtest --config {config_path}\n"
            f"python -m inventory_di.cli report --config {config_path}",
            language="bash",
        )
        return

    renderers = dict(zip(REGISTERED_PAGES, [render_page_a, render_page_b, render_page_c, render_page_d, render_page_e]))
    try:
        renderers[page](bundle)
    except (ValueError, RuntimeError) as exc:
        st.error(f"This experiment could not be evaluated: {exc}. Check the controls or regenerate the indicated artifacts.")


if __name__ == "__main__":
    main()
