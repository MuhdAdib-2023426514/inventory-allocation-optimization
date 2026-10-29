# Experiment Protocol & Reproducibility Contract

**Authoritative specification:** [`AGENTS.md`](../AGENTS.md)

---

## 1. Temporal Split & Leakage-Prevention Protocol

1. **Whole-Origin Chronological Partitioning (`src/inventory_di/data/splits.py`):**
   - Unique weekly `forecast_origin` dates are sorted chronologically and partitioned into contiguous time blocks:
     $$\max(\text{origin}_{\text{train}}) < \min(\text{origin}_{\text{val}}) \le \max(\text{origin}_{\text{val}}) < \min(\text{origin}_{\text{test}})$$
   - Default `configs/demo.yaml`: 20 train origins, 8 validation origins, 8 test origins (36 total weekly origins after 2 warm-up lag weeks).
   - Default `configs/m5.yaml`: 100 train origins, 8 validation origins, 12 test origins.

2. **Strictly Pre-Origin Feature Engineering (`src/inventory_di/data/features.py`):**
   - For every origin $t_0$ and 7-day target window $[t_{\text{start}}, t_{\text{end}}]$, `t_start = t_0 + 1 day` and `t_end = t_0 + 7 days`.
   - Lagged demand features (`lag_1w_units`, `lag_2w_units`, `rolling_4w_mean`, `rolling_8w_mean`, `rolling_4w_std`) and price covariates (`lag_price_mu`) are computed strictly from target windows ending $\le t_0$.

3. **Training-Only Parameter & Cohort Estimation:**
   - Store/SKU cohort filtering (`cat_id == "HOUSEHOLD"`), store–SKU allocation caps $K_{sp}$, baseline warehouse supply $W_p$, discrete PMF support $H$, tail excess mean, feature standardization (`feat_mean`, `feat_std`), and empirical copula rank matrices are fitted **exclusively on `split == "train"`**.
   - Post-hoc temperature scaling is fitted **exclusively on `split == "val"`**.
   - Oracle solution caching (`precompute_sku_oracle_cache`) is restricted to `train` and `val` splits.

---

## 2. Apples-to-Apples Policy Comparison Protocol

At each test origin $t_0 \in \text{test\_origins}$:
- All 7 policies receive the exact same `AllocationInstance` ($I_{sp}, W_p, K_{sp}, C_s, b_{sp}, h_{sp}, t_{sp}$) verified by a shared SHA-256 environment fingerprint in `run_manifest.json`.
- Both DRO policies (`supervised_dro` and `dfl_dro`) draw $N$ joint demand scenarios using the **same deterministic origin seed** (`global_seed + 1000 + origin_idx`) and the same training-fitted empirical rank-bootstrap copula, isolating the effect of the DFL-learned marginals vs. supervised marginals.
- Every solver output is checked by `check_allocation_feasibility` and evaluated on the unclipped held-out demand proxy $D_{\text{actual}}$ via `evaluate_exact_cost`.

---

## 3. Verification & Audit Commands

Run the complete test suite and all five deterministic Codex verification skills:

```bash
# 1. Unit & integration test suite
.venv/bin/pytest -q

# 2. Linter check
.venv/bin/ruff check .

# 3. Independent mathematical optimization oracle & identity self-test
.venv/bin/python .agents/skills/optimization-verification/scripts/verify_optimization.py --self-test

# 4. DFL contract & SPO+ gradient verification
.venv/bin/python .agents/skills/decision-focused-learning/scripts/verify_dfl_contract.py --self-test

# 5. Temporal & target leakage audit on generated lineage manifest and source tree
.venv/bin/python .agents/skills/data-leakage-audit/scripts/audit_leakage.py --manifest data/demo/lineage_manifest.json --scan-dir src

# 6. Reproducibility & 7-policy parity audit on generated run manifest
.venv/bin/python .agents/skills/experiment-reproducibility/scripts/verify_reproducibility.py --manifest artifacts/demo/run_manifest.json

# 7. Dashboard QA bundle & Playwright MCP verification plan
.venv/bin/python .agents/skills/dashboard-qa/scripts/verify_dashboard_qa.py --bundle artifacts/demo/dashboard_qa_bundle.json --emit-playwright-plan
```
