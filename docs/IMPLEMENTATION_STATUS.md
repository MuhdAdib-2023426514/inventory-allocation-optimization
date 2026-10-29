# Implementation Status

**Last updated:** 2026-10-29  
**Authoritative specification:** [`AGENTS.md`](../AGENTS.md)

---

## 1. Summary of Completed & Verified Phases (Phases 0–7)

| Phase | Milestone | Status | Verification Evidence |
| :--- | :--- | :---: | :--- |
| **Phase 0** | Environment Bootstrap, HiGHS MILP & PyEPO `SPOPlus` Smoke Test | **COMPLETED & VERIFIED** | Python `3.12.14` pinned in `.python-version` and `pyproject.toml`; `uv.lock` generated. `python -m inventory_di.cli doctor` and `scripts/check_environment.py` pass (`appsi_highs` + `SPOPlus` backward pass `weight_grad_norm = 7.7972`). |
| **Phase 1** | Data Contracts, Synthetic Demo Generator, M5 Reader & Exact Cost Accounting | **COMPLETED & VERIFIED** | Implemented `src/inventory_di/config.py`, `io.py`, `data/splits.py`, `data/features.py`, `data/synthetic.py`, `data/m5.py`, and `optimization/core.py`. Verified by `tests/test_data_leakage.py`, `tests/test_cost_identity.py` (30 randomized marginal-unit cost identity trials including $I_s > 0$), and `.agents/skills/data-leakage-audit/scripts/audit_leakage.py --manifest data/demo/lineage_manifest.json --scan-dir src`. |
| **Phase 2** | Optimization Baselines & Nominal MILP | **COMPLETED & VERIFIED** | Implemented `equal_allocation`, `proportional_allocation`, `solve_deterministic_milp`, `solve_nominal_marginal_cost_milp`, and `clairvoyant_oracle` in `src/inventory_di/optimization/baselines.py` and `core.py`. Mandatory 2-store toy oracle (`30/30` cost `30.0` / fill rate `62.5%` vs `40/20` cost `20.0` / fill rate `75.0%`) verified in `tests/test_optimizer.py`. |
| **Phase 3** | Probabilistic Demand Distribution Forecasting | **COMPLETED & VERIFIED** | Implemented `DiscreteDemandPMFNet` ($0 \dots H$ + `overflow > H` bin), validation-only temperature scaling, survival probability mapping $q[s,k]$, predicted marginal costs $\hat{c}[s,k]$, and training-only `EmpiricalCopulaScenarioGenerator` in `src/inventory_di/models/distributions.py` and `supervised.py`. |
| **Phase 4** | Genuine Decision-Focused Learning (PyEPO `SPOPlus`) | **COMPLETED & VERIFIED** | Implemented `FixedFeasibleMarginalAllocationModel` (`optOmoModel` + `appsi_highs`), `precompute_sku_oracle_cache` (`train`/`val` only), 1-batch SPO+ vs NLL parameter divergence check, and $\lambda_{\text{DFL}}$ validation ablation in `src/inventory_di/optimization/pyepo_fixed.py` and `models/dfl.py`. Verified in `tests/test_dfl_gradients.py`. |
| **Phase 5** | Explicit Scenario-Reweighting DRO ($\text{CVaR}_\alpha$) | **COMPLETED & VERIFIED** | Implemented `solve_scenario_reweighting_dro_milp`, `exact_ambiguity_set_worst_case_loss`, and `rockafellar_uryasev_cvar_lp` in `src/inventory_di/optimization/dro.py`. Verified LP $\equiv \mathcal{P}_\alpha$ reweighting duality, brute-force enumeration, and risk-neutral limiting cases ($\alpha=0$, `risk_weight=0`) in `tests/test_dro_equivalence.py`. |
| **Phase 6** | 7-Policy Rolling-Origin Backtest, Sensitivity Sweeps & Reporting | **COMPLETED & VERIFIED** | Implemented `src/inventory_di/evaluation/backtest.py`, `metrics.py`, `validation.py`, `visualization/plots.py`, and `cli.py`. Executed complete `DEMO / SYNTHETIC` pipeline producing `artifacts/demo/*` and `reports/results_summary.md`. Verified by `tests/test_backtest.py` and `verify_reproducibility.py --manifest artifacts/demo/run_manifest.json`. |
| **Phase 7** | Streamlit Decision-Experiment Dashboard & Research Documentation | **COMPLETED & VERIFIED** | Implemented `app/streamlit_app.py` (Pages A–E), `docs/METHODOLOGY.md`, `docs/ASSUMPTIONS_AND_LIMITATIONS.md`, `docs/EXPERIMENT_PROTOCOL.md`, `docs/PRESENTATION_GUIDE.md`, and `README.md`. Verified by `tests/test_dashboard_artifacts.py`, `verify_dashboard_qa.py --bundle artifacts/demo/dashboard_qa_bundle.json`, and live Playwright MCP browser automation across all 5 pages (`0` console errors). |

---

## 2. Empirical vs. Synthetic Status Disclosure

- **Synthetic Demo Mode (`configs/demo.yaml`, `data_mode="synthetic_demo"`, badge `DEMO / SYNTHETIC`):**
  - **Status:** Fully executed, verified, and exported to `data/demo/`, `artifacts/demo/`, `reports/demo/results_summary.md`, and `reports/results_summary.md`.
  - **Key Result (`demo_v1`, 8 held-out test origins):** `dfl_pto` achieved the lowest mean out-of-sample cost (**50.48 MU**, fill rate **86.1%**, mean oracle regret **26.75 MU**), outperforming `supervised_pto` (**51.37 MU**, fill rate **85.6%**), `proportional_allocation` (**52.16 MU**, fill rate **85.2%**), and `equal_allocation` (**53.13 MU**, fill rate **84.5%**). Conservative tail-risk weighting in `supervised_dro` (**55.98 MU**) and `dfl_dro` (**62.33 MU**) reduced model-implied worst-tail $\text{CVaR}_{0.90}$ (**247.93 MU** and **219.55 MU** vs. **255.96 MU** for `supervised_pto`), illustrating the explicit cost-of-robustness trade-off when held-out test weeks do not all realize extreme tail surges.
- **Walmart M5 Mode (`configs/m5.yaml`, `data_mode="m5_sales_proxy"`):**
  - **Status:** Code path (`src/inventory_di/data/m5.py` and CLI `--config configs/m5.yaml`) is implemented and tested for missing-file error handling, but **empirical M5 execution is `BLOCKED / NOT RUN`** because the user-licensed Kaggle M5 raw CSV files (`calendar.csv`, `sell_prices.csv`, `sales_train_evaluation.csv`) are not present in `data/raw/m5/`. No M5 numbers are claimed or fabricated.
