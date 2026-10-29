#!/usr/bin/env python3
"""Deterministic Streamlit Dashboard QA & Playwright MCP readiness verifier.

Enforces AGENTS.md Sections 10, 14, and 15:
- All 5 required dashboard pages (Pages A-E)
- Visible provenance badge matching data_mode ("DEMO / SYNTHETIC" vs "M5 sales proxy + simulated inventory")
- Strict separation of Model-Implied Expected Cost, Model-Implied CVaR, and Realized Historical Test Cost
- Explicit probability-reweighting ambiguity set note (never mislabeled Wasserstein DRO)
- Offline demo readiness, solver status visibility, and missing-artifact error state handling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TOL = 1e-6

REQUIRED_PAGES = [
    "Page A — Problem & Data",
    "Page B — Models & Diagnostics",
    "Page C — Allocation Results",
    "Page D — Risk & Robustness",
    "Page E — Evidence & Business Impact",
]

REQUIRED_POLICIES_ON_PAGE_E = {
    "equal_allocation",
    "proportional_allocation",
    "supervised_pto",
    "supervised_dro",
    "dfl_pto",
    "dfl_dro",
}


def build_playwright_mcp_plan(url: str = "http://localhost:8501") -> list[dict[str, Any]]:
    """Return the deterministic sequence of Playwright MCP tool calls for live UI QA."""
    return [
        {
            "step": 1,
            "server": "playwright",
            "tool": "browser_navigate",
            "arguments": {"url": url},
            "verify": "Page loads without Streamlit exception; provenance badge visible on Page A.",
        },
        {
            "step": 2,
            "server": "playwright",
            "tool": "browser_snapshot",
            "arguments": {},
            "verify": "Confirm Page A store/SKU counts, MU currency, and synthetic inventory disclosures.",
        },
        {
            "step": 3,
            "server": "playwright",
            "tool": "browser_select_option",
            "arguments": {"page": "Page B — Models & Diagnostics"},
            "verify": "Confirm temporal split diagram, P10/P50/P90, NLL/MAE/WAPE, and calibration warning gate.",
        },
        {
            "step": 4,
            "server": "playwright",
            "tool": "browser_select_option",
            "arguments": {"page": "Page C — Allocation Results"},
            "verify": "Confirm feasible allocations (sum_s x_sp <= W_p), solver_status badge, and CSV export.",
        },
        {
            "step": 5,
            "server": "playwright",
            "tool": "browser_select_option",
            "arguments": {"page": "Page D — Risk & Robustness"},
            "verify": "Confirm alpha/risk_weight selectors, separate Expected Cost vs CVaR vs Realized Cost, and P_alpha note.",
        },
        {
            "step": 6,
            "server": "playwright",
            "tool": "browser_select_option",
            "arguments": {"page": "Page E — Evidence & Business Impact"},
            "verify": "Confirm paired 6-policy + oracle table, main KPIs, and synthetic vs M5 separation.",
        },
        {
            "step": 7,
            "server": "playwright",
            "tool": "browser_console_messages",
            "arguments": {},
            "verify": "Zero uncaught browser or Streamlit runtime errors.",
        },
        {
            "step": 8,
            "server": "playwright",
            "tool": "browser_network_requests",
            "arguments": {},
            "verify": "Offline readiness: no required paid/external API calls.",
        },
        {
            "step": 9,
            "server": "playwright",
            "tool": "browser_close",
            "arguments": {},
            "verify": "Clean browser session shutdown.",
        },
    ]


def verify_dashboard_bundle(bundle: dict[str, Any]) -> list[str]:
    """Validate a dashboard state/artifact QA bundle against AGENTS.md §10 & §14."""
    errors: list[str] = []
    required_top = [
        "data_mode",
        "provenance_badge",
        "currency_label",
        "pages_present",
        "page_b_diagnostics",
        "page_c_allocations",
        "page_d_risk",
        "page_e_evidence",
        "offline_ready",
        "retrain_in_ui",
        "missing_artifact_guidance_present",
    ]
    for key in required_top:
        if key not in bundle:
            errors.append(f"Missing required dashboard QA field '{key}'")
    if errors:
        return errors

    # 1. Provenance badge vs data_mode
    data_mode = str(bundle["data_mode"])
    badge = str(bundle["provenance_badge"])
    if data_mode == "synthetic_demo" and "DEMO" not in badge.upper() and "SYNTHETIC" not in badge.upper():
        errors.append(
            f"data_mode='synthetic_demo' must display visible DEMO / SYNTHETIC badge, got {badge!r}"
        )
    if data_mode == "m5_sales_proxy" and "M5" not in badge.upper():
        errors.append(
            f"data_mode='m5_sales_proxy' must display 'M5 sales proxy + simulated inventory' badge, got {badge!r}"
        )

    if str(bundle["currency_label"]) != "MU":
        errors.append(f"Dashboard currency_label must be 'MU', got {bundle['currency_label']!r}")

    # 2. All 5 pages present
    pages_present = set(bundle.get("pages_present", []))
    for req_page in REQUIRED_PAGES:
        if req_page not in pages_present:
            errors.append(f"Missing required dashboard page: {req_page!r}")

    # 3. Page B calibration warning gate
    page_b = bundle.get("page_b_diagnostics", {})
    if page_b.get("poorly_calibrated", False) and not page_b.get("calibration_warning_shown", False):
        errors.append(
            "Page B violation: forecast uncertainty is poorly calibrated but calibration_warning_shown is False"
        )

    # 4. Page C solver status & warehouse feasibility
    page_c = bundle.get("page_c_allocations", {})
    if not page_c.get("solver_status_visible", False):
        errors.append("Page C violation: solver_status_visible is False")
    if page_c.get("max_sku_shipped_minus_warehouse_supply", 0) > 0:
        errors.append("Page C violation: displayed allocation exceeds warehouse supply W_p")

    # 5. Page D strict separation of Expected Cost, CVaR, and Realized Historical Test Cost
    page_d = bundle.get("page_d_risk", {})
    for metric_key in ("model_expected_cost_mu", "model_cvar_cost_mu", "realized_test_cost_mu"):
        if metric_key not in page_d:
            errors.append(f"Page D missing distinct metric '{metric_key}'")
    if all(k in page_d for k in ("model_expected_cost_mu", "model_cvar_cost_mu", "realized_test_cost_mu")):
        exp_c = float(page_d["model_expected_cost_mu"])
        cvar_c = float(page_d["model_cvar_cost_mu"])
        if cvar_c + TOL < exp_c:
            errors.append(
                f"Page D mathematical inconsistency: model_cvar_cost_mu ({cvar_c}) < model_expected_cost_mu ({exp_c})"
            )
    if page_d.get("metrics_conflated", False):
        errors.append("Page D violation: expected cost, CVaR, and realized test cost are conflated")
    if str(page_d.get("ambiguity_set_label", "")) != "probability_reweighting_cvar":
        errors.append(
            f"Page D violation: ambiguity_set_label must be 'probability_reweighting_cvar' (not silently labeled Wasserstein), got {page_d.get('ambiguity_set_label')!r}"
        )

    # 6. Page E policies & synthetic/M5 separation
    page_e = bundle.get("page_e_evidence", {})
    shown_policies = set(page_e.get("policies_shown", []))
    missing_pols = REQUIRED_POLICIES_ON_PAGE_E - shown_policies
    if missing_pols:
        errors.append(f"Page E missing required policies: {sorted(missing_pols)}")
    if not page_e.get("synthetic_and_m5_separated", False):
        errors.append("Page E violation: synthetic stress figures and M5-proxy evidence are not separated")

    # 7. Operational & offline readiness
    if not bool(bundle.get("offline_ready")):
        errors.append("Dashboard is not offline_ready")
    if bool(bundle.get("retrain_in_ui")):
        errors.append("Dashboard triggers model retraining inside the UI (forbidden by AGENTS.md §10)")
    if not bool(bundle.get("missing_artifact_guidance_present")):
        errors.append("Dashboard lacks clear error guidance when cached artifacts are missing")

    return errors


def run_self_test() -> dict[str, Any]:
    """Run built-in positive and negative dashboard QA checks."""
    errors: list[str] = []

    valid_bundle = {
        "data_mode": "synthetic_demo",
        "provenance_badge": "DEMO / SYNTHETIC",
        "currency_label": "MU",
        "pages_present": list(REQUIRED_PAGES),
        "page_b_diagnostics": {
            "poorly_calibrated": True,
            "calibration_warning_shown": True,
        },
        "page_c_allocations": {
            "solver_status_visible": True,
            "max_sku_shipped_minus_warehouse_supply": 0,
            "csv_download_present": True,
        },
        "page_d_risk": {
            "alpha": 0.90,
            "risk_weight": 0.50,
            "model_expected_cost_mu": 120.5,
            "model_cvar_cost_mu": 165.0,
            "realized_test_cost_mu": 134.0,
            "metrics_conflated": False,
            "ambiguity_set_label": "probability_reweighting_cvar",
        },
        "page_e_evidence": {
            "policies_shown": sorted(REQUIRED_POLICIES_ON_PAGE_E | {"clairvoyant_oracle"}),
            "synthetic_and_m5_separated": True,
        },
        "offline_ready": True,
        "retrain_in_ui": False,
        "missing_artifact_guidance_present": True,
    }
    errors.extend(verify_dashboard_bundle(valid_bundle))

    # Negative check: synthetic demo labeled as M5, Wasserstein mislabeling, CVaR < Expected, retrain in UI
    bad_bundle = json.loads(json.dumps(valid_bundle))
    bad_bundle["provenance_badge"] = "Empirical Walmart M5 Savings"
    bad_bundle["page_d_risk"]["model_cvar_cost_mu"] = 90.0
    bad_bundle["page_d_risk"]["ambiguity_set_label"] = "wasserstein_dro"
    bad_bundle["retrain_in_ui"] = True
    bad_errs = verify_dashboard_bundle(bad_bundle)

    for expected in (
        "DEMO / SYNTHETIC",
        "mathematical inconsistency",
        "probability_reweighting_cvar",
        "retraining inside the UI",
    ):
        if not any(expected in e for e in bad_errs):
            errors.append(f"Self-test failed to detect dashboard violation: {expected}")

    app_path = Path("app/streamlit_app.py")
    return {
        "skill": "dashboard-qa",
        "status": "PASS" if not errors else "FAIL",
        "streamlit_app_exists": app_path.exists(),
        "checks_run": [
            "five_page_structure_and_provenance_badge",
            "page_b_calibration_warning_gate",
            "page_d_cost_cvar_realized_separation_and_ambiguity_label",
            "offline_and_no_ui_retrain_gates",
            "negative_mislabeling_and_conflation_detection",
        ],
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dashboard QA and Playwright MCP plan verifier.")
    parser.add_argument("--self-test", action="store_true", help="Run built-in QA self-test.")
    parser.add_argument("--bundle", type=Path, help="Path to dashboard QA bundle JSON.")
    parser.add_argument(
        "--emit-playwright-plan",
        action="store_true",
        help="Include step-by-step Playwright MCP tool execution plan in output.",
    )
    args = parser.parse_args(argv)

    if not (args.self_test or args.bundle or args.emit_playwright_plan):
        parser.error("Provide --self-test, --bundle <path>, or --emit-playwright-plan")

    if args.self_test:
        report = run_self_test()
    elif args.bundle:
        data = json.loads(args.bundle.read_text(encoding="utf-8"))
        errs = verify_dashboard_bundle(data)
        report = {
            "skill": "dashboard-qa",
            "bundle": str(args.bundle),
            "status": "PASS" if not errs else "FAIL",
            "errors": errs,
        }
    else:
        report = {"skill": "dashboard-qa", "status": "PASS", "errors": []}

    if args.emit_playwright_plan:
        report["playwright_mcp_plan"] = build_playwright_mcp_plan()

    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
