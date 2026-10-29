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
from inventory_di.optimization.core import evaluate_exact_cost
from inventory_di.optimization.dro import solve_scenario_reweighting_dro_milp
from inventory_di.visualization.plots import (
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


@st.cache_data(show_spinner=False)
def load_cached_bundle(config_path: str) -> dict[str, Any]:
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
    st.plotly_chart(plot_historical_demand_by_split(weekly_df, selected_sku), use_container_width=True)

    st.subheader("Synthetic Inventory & Operating Economics Contract (`inventory_config.parquet`)")
    st.dataframe(inv_df, use_container_width=True)

    with st.expander("Data Quality, Temporal Split & Leakage-Audit Manifest"):
        st.json(lineage)


def render_page_b(bundle: dict[str, Any]) -> None:
    st.header("Page B — Models & Diagnostics")
    diag: dict[str, Any] = bundle["training_diagnostics"]
    origins_bundle: dict[str, Any] = bundle["cached_origin_bundles"]

    st.markdown(
        """
        ### Supervised PTO vs. Decision-Focused Learning (DFL via PyEPO `SPOPlus`)
        * **Chronological Whole-Origin Split:** `train` $\\rightarrow$ `val` (temperature calibration & early stopping) $\\rightarrow$ `test` (out-of-sample evaluation).
        * **Discrete PMF with Overflow Tail:** Both models share the exact same neural architecture predicting $p(d \\mid \\text{features})$ over $d \\in \\{0, \\dots, H\\}$ plus an explicit `overflow > H` bin.
        * **Survival-to-Marginal-Cost Mapping:** Survival probabilities $q[s,k] = P(D_s \\ge I_s + k \\mid \\text{features})$ map differentiably to marginal unit costs $\\hat{c}[s,k] = t_s + h_s - (b_s + h_s)q[s,k]$.
        * **Genuine DFL Objective:** Trains end-to-end with $\\mathcal{L} = \\text{NLL} + \\lambda_{\\text{DFL}} \\cdot \\mathcal{L}_{\\text{SPO}+}(\\hat{c}, c_{\\text{true}}, y^*, z^*)$ on the per-SKU fixed-feasible marginal-unit optimization model.
        """
    )

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
    if sup_cov < 0.60 or dfl_cov < 0.60 or float(dfl_cal.get("nll", 0.0)) > 3.2:
        st.warning(
            "**Calibration Warning:** Validation P10–P90 interval coverage or NLL indicates elevated "
            "distributional uncertainty. Inspect scenario stress bounds on Page D before committing stock."
        )
    else:
        st.success(
            f"**Calibration Diagnostics Check:** Validation temperature scaling applied (`split='val'`). "
            f"P10–P90 coverage: Supervised = `{100.0 * sup_cov:.1f}%`, DFL = `{100.0 * dfl_cov:.1f}%`."
        )

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
    st.dataframe(diag_table, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_training_curves(diag["dfl"][selected_sku]["history"], selected_sku, "DFL (NLL + SPO+)"),
            use_container_width=True,
        )
    with c2:
        q_df = pd.DataFrame(origins_bundle[selected_origin]["forecast_quantiles"])
        st.plotly_chart(plot_forecast_quantiles(q_df, selected_sku), use_container_width=True)

    if diag.get("lambda_dfl_ablation"):
        st.subheader("Validation Ablation Across `lambda_DFL`")
        st.dataframe(pd.DataFrame(diag["lambda_dfl_ablation"]), use_container_width=True)


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
    all_optimal = bool((sub_alloc["solver_status"] == "optimal").all())
    if all_feasible and all_optimal:
        st.success(
            f"**Solver & Feasibility Status (`{sel_origin}`):** `optimal` | All warehouse supply ($W_p$), "
            "per-store-SKU caps ($K_{sp}$), and total store capacities ($C_s$) verified."
        )
    else:
        st.error(f"**Solver Warning (`{sel_origin}`):** Non-optimal or infeasible allocation detected!")

    st.plotly_chart(plot_store_allocations_comparison(sub_alloc, sel_sku), use_container_width=True)

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
        use_container_width=True,
    )

    st.subheader(f"Store-Level Shipment Table (`{sel_origin}`, `{sel_sku}`)")
    sku_table = sub_alloc[sub_alloc["sku_id"] == sel_sku][
        [
            "policy",
            "store_id",
            "sku_id",
            "opening_stock_units",
            "warehouse_supply_units",
            "store_sku_unit_cap",
            "allocated_units",
            "realized_demand_units",
            "is_feasible",
            "solver_status",
        ]
    ]
    st.dataframe(sku_table, use_container_width=True)

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
    sel_alpha = c3.select_slider(
        "Tail Ambiguity Parameter (alpha)",
        options=[0.0, 0.50, 0.75, 0.90, 0.95],
        value=0.90,
        key="page_d_alpha",
    )
    sel_rw = c4.select_slider(
        "Tail Risk Weight (risk_weight)",
        options=[0.0, 0.25, 0.50, 0.75, 1.0],
        value=0.50,
        key="page_d_rw",
    )
    sel_supply_mult = st.select_slider(
        "Warehouse Supply Scarcity Multiplier",
        options=[0.6, 0.8, 1.0, 1.2],
        value=1.0,
        key="page_d_supply",
    )

    # Re-solve locally using cached scenarios (fast <50ms MILP solve; zero model retraining)
    ob = origins_bundle[sel_origin]
    scen_arr = np.array(
        ob["dfl_scenarios"] if sel_family == "dfl_dro" else ob["sup_scenarios"], dtype=int
    )
    D_real = np.array(ob["D_realized"], dtype=int)
    inst_mod = build_allocation_instance_from_config_df(inv_df, supply_multiplier=float(sel_supply_mult))

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
        "3. Realized Historical Test Cost (MU)",
        f"{real_live.total_cost_mu:.2f} MU",
        help="Actual out-of-sample cost evaluated on the held-out demand proxy D_realized.",
    )
    m4.metric(
        "Solver & Constraint Status",
        f"{dro_live.solver_status} ({100.0 * real_live.fill_rate:.1f}% fill)",
        help=f"Feasible: {dro_live.is_feasible}, Oracle Regret: {real_live.total_cost_mu - oracle_live_cost:.2f} MU",
    )

    st.plotly_chart(
        plot_risk_cost_pareto(sens_df, supply_multiplier=float(sel_supply_mult)),
        use_container_width=True,
    )

    st.subheader("Precomputed Risk & Supply Sensitivity Sweep Table")
    st.dataframe(
        sens_df[
            (sens_df["supply_multiplier"] - float(sel_supply_mult)).abs() < 1e-6
        ].sort_values(["policy_family", "alpha", "risk_weight"]),
        use_container_width=True,
    )


def render_page_e(bundle: dict[str, Any]) -> None:
    st.header("Page E — Evidence & Business Impact")
    summary_df: pd.DataFrame = bundle["summary_metrics"]
    metrics_df: pd.DataFrame = bundle["backtest_metrics"]
    manifest: dict[str, Any] = bundle["run_manifest"]

    st.plotly_chart(plot_policy_cost_comparison(summary_df), use_container_width=True)

    st.subheader("Apples-to-Apples 7-Policy Out-of-Sample Evaluation Table (MU)")
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
        use_container_width=True,
    )

    st.subheader("Per-Origin Paired Realized Total Cost (MU)")
    piv = metrics_df.pivot(index="origin", columns="policy", values="total_cost_mu")[
        summary_df["policy"].tolist()
    ]
    st.dataframe(piv, use_container_width=True)

    with st.expander("Run Reproducibility Manifest & Checksums"):
        st.json(manifest)


def main() -> None:
    st.set_page_config(
        page_title="Decision-Focused & Robust Inventory Allocation",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("Inventory Decision Intelligence")
    mode_label = st.sidebar.radio(
        "Dataset / Experiment Mode",
        options=["Demo Mode (DEMO / SYNTHETIC)", "M5 Mode (M5 Sales Proxy)"],
        index=0,
    )
    config_path = "configs/demo.yaml" if "Demo" in mode_label else "configs/m5.yaml"

    page = st.sidebar.radio("Navigation (Pages A–E)", options=REGISTERED_PAGES, index=0)

    bundle = load_cached_bundle(config_path)
    render_provenance_banner(bundle["cfg"])

    if bundle["status"] != "OK":
        st.error(
            "### Cached Experiment Artifacts Not Found\n"
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

    if page == "Page A — Problem & Data":
        render_page_a(bundle)
    elif page == "Page B — Models & Diagnostics":
        render_page_b(bundle)
    elif page == "Page C — Allocation Results":
        render_page_c(bundle)
    elif page == "Page D — Risk & Robustness":
        render_page_d(bundle)
    elif page == "Page E — Evidence & Business Impact":
        render_page_e(bundle)


if __name__ == "__main__":
    main()
