---
name: dashboard-qa
description: >-
  Validate the local Streamlit decision-experiment dashboard using deterministic
  artifact preflight checks and Playwright MCP browser automation. Verifies Pages
  A through E, scenario and risk controls, numerical consistency with backtest
  artifacts, strict separation of expected cost vs CVaR vs realized test cost,
  error states, and offline demonstration readiness. Activate when building,
  modifying, testing, or demonstrating app/streamlit_app.py or dashboard plots.
---

# Dashboard QA Skill (`dashboard-qa`)

This skill validates the Streamlit decision-experiment dashboard against [`AGENTS.md`](../../../AGENTS.md) (§0, §1.4, §1.8, §1.9, §7, §10, §14, §15), combining deterministic preflight checks with interactive browser verification via the `playwright` MCP server.

## Triggering Conditions

Invoke this skill automatically when:
- Implementing or modifying `app/streamlit_app.py` or `src/inventory_di/visualization/plots.py`.
- Adding or updating cached dashboard artifacts, risk sensitivity tables, or downloadable CSV exports.
- Verifying Pages A–E, provenance badges (`DEMO / SYNTHETIC` vs `M5 sales proxy + simulated inventory`), error states for missing artifacts, or 60-second offline demo readiness.

## Required Inputs

1. **Authoritative Contract**: [`AGENTS.md`](../../../AGENTS.md) §10 (Application / presentation requirements) and §14 (Operational acceptance gate).
2. **Playwright MCP Runbook**: [`references/playwright_mcp_runbook.md`](./references/playwright_mcp_runbook.md).
3. **Target Artifacts**: Dashboard bundle metadata JSON and/or `app/streamlit_app.py` source file.

## Execution Workflow

1. **Run Deterministic Dashboard Preflight & Self-Test**:
   ```bash
   .venv/bin/python .agents/skills/dashboard-qa/scripts/verify_dashboard_qa.py --self-test
   ```
2. **Audit Dashboard Artifact Bundle & Source Offline Readiness**:
   - Verify all 5 required pages are registered (`Page A — Problem & Data`, `Page B — Models & Diagnostics`, `Page C — Allocation Results`, `Page D — Risk & Robustness`, `Page E — Evidence & Business Impact`).
   - Verify the provenance badge matches `data_mode` (`"DEMO / SYNTHETIC"` for `synthetic_demo`; `"M5 sales proxy + simulated inventory"` for `m5_sales_proxy`).
   - Verify Page D displays **three distinct, non-conflated cost metrics**:
     1. `model_expected_cost_mu` (Model-Implied Expected Cost)
     2. `model_cvar_cost_mu` (Model-Implied CVaR / Tail Risk)
     3. `realized_test_cost_mu` (Realized Historical Test Cost on held-out demand proxy)
   - Verify UI interactions on Page D never trigger model retraining (`retrain_in_ui == False`) and that missing artifacts render a clear actionable message (`missing_artifact_guidance_present == True`).
3. **Interactive Verification via Playwright MCP (When Streamlit Server Is Running)**:
   - Follow [`references/playwright_mcp_runbook.md`](./references/playwright_mcp_runbook.md) to inspect the live app via `call_mcp_tool` with `ServerName: "playwright"`:
     1. Start `streamlit run app/streamlit_app.py --server.headless true --server.port 8501` as a background task once `app/streamlit_app.py` exists.
     2. Call `browser_navigate` to `http://localhost:8501`.
     3. Call `browser_snapshot` and `browser_console_messages` on each of Pages A–E to confirm zero uncaught exceptions, visible provenance badges, `MU` currency labels, and solver status badges.
     4. On Page D, interact with `alpha`, `risk_weight`, and warehouse supply-stress controls (`browser_select_option` / `browser_click`), then assert that CVaR $\ge$ Expected Cost and that the ambiguity-set formula $\mathcal{P}_\alpha = \{w : \sum w_j = 1, 0 \le w_j \le p_j/(1-\alpha)\}$ is documented.
     5. Call `browser_network_requests` to verify zero external API or CDN failures in offline mode, and `browser_close` when finished.

## Verification Procedures

Validate a dashboard QA bundle and emit the Playwright MCP test plan:
```bash
.venv/bin/python .agents/skills/dashboard-qa/scripts/verify_dashboard_qa.py --bundle path/to/dashboard_bundle.json --emit-playwright-plan
```

## Expected Outputs

- JSON QA report covering page completeness, provenance badge accuracy, cost-metric separation, calibration/solver warning gates, offline readiness, and the step-by-step Playwright MCP call specification.
- Exit code `0` when all deterministic gates pass; exit code `1` on any scientific or operational violation.

## Completion Criteria

- `verify_dashboard_qa.py --self-test` exits `0`.
- When `app/streamlit_app.py` is present, both deterministic preflight and Playwright MCP browser checks pass across Pages A–E with zero console errors and verified offline readiness.
