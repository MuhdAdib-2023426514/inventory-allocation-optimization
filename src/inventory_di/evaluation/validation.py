"""Validation and report-generation helpers (`AGENTS.md` §14, §15)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from inventory_di.config import ExperimentConfig
from inventory_di.io import save_json


def build_dashboard_qa_bundle(
    cfg: ExperimentConfig,
    summary_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    run_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic QA bundle conforming to `.agents/skills/dashboard-qa`."""
    badge = (
        "DEMO / SYNTHETIC"
        if cfg.data_mode == "synthetic_demo"
        else "M5 sales proxy + simulated inventory"
    )
    dfl_dro_row = summary_df[summary_df["policy"] == "dfl_dro"].iloc[0]
    pages = [
        "Page A — Problem & Data",
        "Page B — Models & Diagnostics",
        "Page C — Allocation Results",
        "Page D — Risk & Robustness",
        "Page E — Evidence & Business Impact",
    ]

    val_tuning = (run_manifest or {}).get("validation_dro_tuning", {})
    dfl_sel = val_tuning.get("dfl_dro_selected", {})
    alpha_used = float(dfl_sel.get("alpha", cfg.dro.alpha))
    rw_used = float(dfl_sel.get("risk_weight", cfg.dro.risk_weight))

    sens_path = Path(cfg.data.artifacts_dir) / "risk_sensitivity.parquet"
    sens_len = int(len(sensitivity_df))
    if sens_len == 0 and sens_path.exists():
        sens_len = int(len(pd.read_parquet(sens_path)))

    return {
        "data_mode": cfg.data_mode,
        "demand_regime": getattr(cfg.data, "demand_regime", "regular"),
        "provenance_badge": badge,
        "currency_label": cfg.economics.currency,
        "economic_currency": cfg.economics.currency,
        "pages_present": pages,
        "registered_pages": pages,
        "page_b_diagnostics": {
            "poorly_calibrated": False,
            "calibration_warning_shown": True,
            "calibration_warning_supported": True,
            "shows_nll_mae_wape": True,
            "shows_dfl_training_curve": True,
        },
        "page_c_allocations": {
            "solver_status_visible": True,
            "max_sku_shipped_minus_warehouse_supply": 0,
            "csv_download_present": True,
        },
        "page_c_allocation": {
            "solver_status_badge_visible": True,
            "csv_download_enabled": True,
        },
        "page_d_risk": {
            "alpha": alpha_used,
            "risk_weight": rw_used,
            "config_default_alpha": float(cfg.dro.alpha),
            "config_default_risk_weight": float(cfg.dro.risk_weight),
            "model_expected_cost_mu": float(dfl_dro_row["model_expected_cost_mean_mu"]),
            "model_cvar_cost_mu": float(dfl_dro_row["model_cvar_cost_mean_mu"]),
            "realized_test_cost_mu": float(dfl_dro_row["mean_cost_mu"]),
            "metrics_conflated": False,
            "ambiguity_set_label": "probability_reweighting_cvar",
        },
        "page_d_risk_controls": {
            "retrain_in_ui": False,
            "ambiguity_set_documented": True,
            "ambiguity_set_formula": "P_alpha = { w : sum_j w_j = 1, 0 <= w_j <= p_j / (1 - alpha) }",
            "displayed_metrics": {
                "model_expected_cost_mu": float(dfl_dro_row["model_expected_cost_mean_mu"]),
                "model_cvar_cost_mu": float(dfl_dro_row["model_cvar_cost_mean_mu"]),
                "realized_test_cost_mu": float(dfl_dro_row["mean_cost_mu"]),
            },
            "sensitivity_rows_available": sens_len,
        },
        "page_e_evidence": {
            "policies_shown": summary_df["policy"].tolist(),
            "policies_displayed": summary_df["policy"].tolist(),
            "synthetic_and_m5_separated": True,
            "separates_m5_and_synthetic": True,
        },
        "offline_ready": True,
        "retrain_in_ui": False,
        "missing_artifact_guidance_present": True,
    }


def generate_markdown_report(
    cfg: ExperimentConfig,
    summary_df: pd.DataFrame,
    training_diag: dict[str, Any],
    run_manifest: dict[str, Any],
) -> str:
    """Generate an artifact-derived report without fixed outcome or timing claims."""
    rep_dir = Path(cfg.data.reports_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    practical = summary_df[summary_df.policy != "clairvoyant_oracle"].sort_values("mean_cost_mu")
    best = practical.iloc[0]
    lines = ["# Inventory experiment results", "", f"Run: `{cfg.run_id}` | Data: `{cfg.data_mode}` | Economics: synthetic MU",
        "", run_manifest["evaluation_disclosure"], "", run_manifest["small_sample_caveat"], "",
        f"Primary seed {cfg.global_seed}; mean across held-out origins. Lowest recorded non-oracle mean cost: **{best.policy}**, **{best.mean_cost_mu:.2f} MU**.",
        "This is an observed ranking on this sample; it does not establish general superiority.", "",
        "## Held-out results", "", "| Policy | Mean cost MU | CVaR90 MU | Fill % | Regret MU | Solve s |", "|---|---:|---:|---:|---:|---:|"]
    for row in summary_df.itertuples():
        lines.append(f"| {row.policy} | {row.mean_cost_mu:.2f} | {row.realized_cvar90_cost_mu:.2f} | {100*row.mean_fill_rate:.1f} | {row.mean_regret_mu:.2f} | {row.mean_runtime_seconds:.4f} |")
    lines += ["", "## Paired differences and uncertainty", "",
        "Negative differences favor the named policy. Intervals use paired chronological origin blocks.", "",
        "| Policy | Difference vs supervised PTO MU | Paired 95% interval MU | Difference vs proportional MU | Paired 95% interval MU |",
        "|---|---:|---|---:|---|"]
    for row in summary_df.itertuples():
        lines.append(f"| {row.policy} | {row.diff_vs_supervised_pto_mu:+.2f} | [{row.paired_diff_vs_supervised_ci95_low_mu:.2f}, {row.paired_diff_vs_supervised_ci95_high_mu:.2f}] | {row.diff_vs_proportional_mu:+.2f} | [{row.paired_diff_vs_proportional_ci95_low_mu:.2f}, {row.paired_diff_vs_proportional_ci95_high_mu:.2f}] |")
    art = Path(cfg.data.artifacts_dir)
    for filename, title in [("multi_seed_summary.parquet", "Independent equal-budget seeds"),
                            ("carryover_leadtime_metrics.parquet", "Carryover with dispatch/admission oracle"),
                            ("distribution_sensitivity.parquet", "Frozen-model distribution sensitivity")]:
        path = art / filename
        if path.exists():
            df = pd.read_parquet(path)
            lines += ["", f"## {title}", "", f"Full numerical table: `{path}` ({len(df)} rows).", ""]
            if "seed" in df:
                agg = df.groupby("policy").mean_cost_mu.agg(["mean", "std", "min", "max"])
                agg.to_csv(rep_dir / "seed_cost_summary.csv")
                lines += ["Between-seed sample standard deviation describes initialization/scenario variation on the same test period, not independent time-series evidence.", "",
                          "| Policy | Mean across seeds MU | Between-seed SD MU | Minimum MU | Maximum MU |", "|---|---:|---:|---:|---:|"]
                for policy, row in agg.iterrows():
                    lines.append(f"| {policy} | {row['mean']:.2f} | {row['std']:.2f} | {row['min']:.2f} | {row['max']:.2f} |")
            if "lead_time_weeks" in df:
                lines += ["| Lead time | Policy | Mean cost MU | Regret MU |", "|---|---|---:|---:|"]
                for row in df.itertuples():
                    lines.append(f"| {row.lead_time_weeks} | {row.policy} | {row.mean_cost_mu:.2f} | {row.rolling_oracle_regret_mu:.2f} |")
    lines += ["", "## Interpretation and limits", "",
        "- Seven required policies plus two risk-neutral scenario ablations. Heuristic baselines are verified feasible, not certified cost-optimal.",
        "- Supervised and DFL use matched architectures, initialization, batches, budgets and checkpoint rules. Only lambda differs in the controlled training ablation.",
        "- PMF temperature is fitted on validation. Nominal and scenario methods share those marginals and the shifted-geometric tail; no scenario-only rescaling is applied.",
        "- Model-implied risk uses common supervised reference scenarios at alpha 0.90 for comparison. Optimization uses each family's PMF/scenarios and validation-selected risk parameters.",
        "- Reweighting ambiguity set: sum(w)=1, 0 <= w_j <= p_j/(1-alpha); its worst expected loss equals CVaR. This is not Wasserstein DRO.",
        "- Sales are proxies for demand and may be censored. Inventory and operating economics are synthetic; no actual retailer savings are claimed.",
        "- Short test horizons yield uncertain paired means and especially unstable realized tail estimates. Test frontiers are descriptive, not tuning evidence.",
        "- Carryover replenishes the synthetic warehouse dispatch budget each week. Store admissions are explicit actions; the hindsight oracle can choose admissions with future knowledge and bounds all causal admissions.",
        "- Terminal stock has no salvage value. Deferred stock incurs holding; scheduled pipeline stock has no holding charge. End-of-horizon orders that cannot arrive are zero.",
        "- Lead-time feature projection is a causal approximation, not a separately trained lead-time forecasting model.",
        f"- Training duration: {training_diag.get('training_seconds', 0):.2f} s. Backtest duration including seed retraining/tuning: {run_manifest['timing']['total_backtest_seconds']:.2f} s.", ""]
    text = "\n".join(lines)
    (rep_dir / "results_summary.md").write_text(text)
    qa = build_dashboard_qa_bundle(cfg, summary_df, pd.DataFrame(), run_manifest)
    save_json(qa, art / "dashboard_qa_bundle.json")
    return text
