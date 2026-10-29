# Playwright MCP Runbook for Streamlit Dashboard QA (`dashboard-qa`)

This runbook defines the exact sequence of lazy-loaded `playwright` MCP tool calls (`call_mcp_tool` with `ServerName="playwright"`) to verify `app/streamlit_app.py` against [`AGENTS.md`](../../../AGENTS.md) §10 and §14.

## Prerequisites

1. Read the schema files in `<appDataDir>/mcp/playwright/` (`browser_navigate.json`, `browser_snapshot.json`, `browser_click.json`, `browser_select_option.json`, `browser_console_messages.json`, `browser_network_requests.json`, `browser_take_screenshot.json`, `browser_close.json`) before invoking `call_mcp_tool`.
2. Ensure `app/streamlit_app.py` and cached artifacts exist. If `app/streamlit_app.py` is not yet implemented, run `verify_dashboard_qa.py` in preflight mode and report dashboard UI status as `NOT_IMPLEMENTED` in `docs/IMPLEMENTATION_STATUS.md`.

## Step-by-Step Playwright MCP Execution Plan

### Step 1: Navigate & Verify Initial Load (Page A — Problem & Data)
- **Tool**: `browser_navigate` -> `{"url": "http://localhost:8501"}`
- **Tool**: `browser_snapshot` -> `{}`
- **Assertions**:
  - Provenance badge is visibly rendered (`DEMO / SYNTHETIC` or `M5 sales proxy + simulated inventory`).
  - Store count, SKU count, date range, and `MU` unit labels match the run manifest.
  - Synthetic economics assumption banner (`observed sales != true demand`, `hypothetical inventory != operational inventory`) is visible.

### Step 2: Inspect Page B — Models & Diagnostics
- **Tool**: `browser_click` / `browser_select_option` -> select `"Page B — Models & Diagnostics"`
- **Tool**: `browser_snapshot` -> `{}`
- **Assertions**:
  - Chronological split diagram (`train` / `val` / `test`) is displayed.
  - Forecast distribution shows expected demand, P10/P50/P90 quantiles, NLL, MAE, WAPE, and DFL training curve.
  - If calibration error exceeds threshold, a visible calibration warning is shown instead of an unqualified stock recommendation.

### Step 3: Inspect Page C — Allocation Results
- **Tool**: `browser_click` / `browser_select_option` -> select `"Page C — Allocation Results"`
- **Tool**: `browser_snapshot` -> `{}`
- **Assertions**:
  - Table and chart display demand, opening stock $I_{sp}$, warehouse supply $W_p$, capacity $C_s$, allocated shipment $x_{sp}$, expected excess, expected shortage, and explicit `solver_status`.
  - Total allocated units per SKU never exceed $W_p$.
  - Downloadable CSV control is present.

### Step 4: Inspect Page D — Risk & Robustness
- **Tool**: `browser_click` / `browser_select_option` -> select `"Page D — Risk & Robustness"`
- **Tool**: `browser_snapshot` -> `{}`
- **Assertions**:
  - Controls for $\alpha \in [0, 1)$, `risk_weight` $\in [0, 1]$, and bounded warehouse supply stress exist and resolve locally without triggering neural model training.
  - **Three distinct metrics** are labeled and never conflated:
    1. Model-Implied Expected Cost (`MU`)
    2. Model-Implied $\text{CVaR}_\alpha$ (`MU`)
    3. Realized Historical Test Cost (`MU`)
  - Methodological note explicitly states the probability-reweighting ambiguity set $\mathcal{P}_\alpha = \{w : \sum_j w_j = 1, 0 \le w_j \le p_j/(1-\alpha)\}$ and does not mislabel it as Wasserstein DRO.

### Step 5: Inspect Page E — Evidence & Business Impact
- **Tool**: `browser_click` / `browser_select_option` -> select `"Page E — Evidence & Business Impact"`
- **Tool**: `browser_snapshot` -> `{}`
- **Assertions**:
  - Paired test table displays `equal_allocation`, `proportional_allocation`, `supervised_pto`, `supervised_dro`, `dfl_pto`, `dfl_dro`, and `clairvoyant_oracle` reference.
  - Main KPIs (held-out cost `MU`, oracle regret `MU`, fill rate, shortage units, tail risk, runtime) match `backtest_metrics.parquet`.
  - M5-proxy tables are strictly separated from synthetic stress-test figures.

### Step 6: Console, Offline Network, Error State & Teardown
- **Tool**: `browser_console_messages` -> `{}` (assert zero uncaught JS/Streamlit runtime errors).
- **Tool**: `browser_network_requests` -> `{}` (assert zero failed required external API calls).
- **Tool**: `browser_take_screenshot` -> capture evidence screenshot to artifacts directory if requested.
- **Tool**: `browser_close` -> `{}`.
