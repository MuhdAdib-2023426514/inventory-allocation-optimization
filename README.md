# Decision-Focused and Distributionally Robust Inventory Allocation

**Subtitle:** Decision-Oriented Learning for Risk-Aware Multi-Store Stock Allocation  
**Use Case:** Technical Case Study for *Data Scientist Executive (Decision Intelligence)*, MR. D.I.Y.  
**Authoritative Specification:** [`AGENTS.md`](./AGENTS.md)

---

## 1. Overview & Core Research Question

> **Core Question:** Does decision-focused learning (DFL) combined with distributionally robust optimization (DRO) improve actual inventory allocation cost and decision regret relative to conventional forecast-then-optimize methods?

This repository implements an end-to-end, CPU-first, reproducible Decision Intelligence pipeline and offline Streamlit experiment dashboard that:
1. Ingests either **Walmart M5** `HOUSEHOLD` daily unit sales (`mode=m5`, `data_mode="m5_sales_proxy"`) or a fully self-contained, visibly labeled **Synthetic Demo** fixture (`mode=demo`, `data_mode="synthetic_demo"`, badge `DEMO / SYNTHETIC`).
2. Enforces strict chronological whole-origin splits (`train` $\rightarrow$ `val` $\rightarrow$ `test`) with zero temporal leakage.
3. Trains a discrete demand distribution neural network ($0 \dots H$ plus explicit `overflow > H` bin) under both **Supervised NLL** (`supervised_pto`) and **Genuine Decision-Focused Learning** (`NLL + lambda_DFL * SPOPlus` via PyEPO and HiGHS over a fixed-feasible marginal-unit allocation model).
4. Maps learned demand distributions through a training-only empirical rank-bootstrap copula into an explicit **scenario-probability-reweighting DRO** ($\text{CVaR}_\alpha$ ambiguity set $\mathcal{P}_\alpha = \{w \ge 0 : \sum w_j = 1, w_j \le p_j/(1-\alpha)\}$) MILP solver.
5. Evaluates all **7 required policies** apples-to-apples across rolling out-of-sample test origins (`equal_allocation`, `proportional_allocation`, `supervised_pto`, `supervised_dro`, `dfl_pto`, `dfl_dro`, `clairvoyant_oracle`).

---

## 2. Environment Setup (Python 3.12 + Open-Source MILP Stack)

Using [`uv`](https://docs.astral.sh/uv/) (recommended) or standard `venv + pip`:

```bash
# Option A: Using uv (uses Python 3.12 and uv.lock)
uv sync

# Option B: Using standard venv + pip
python3.12 -m venv .venv
.venv/bin/pip install -e .
```

Verify the environment, HiGHS solver, mandatory 2-store toy mathematical oracle, and PyEPO `SPOPlus` backward gradient pass:

```bash
.venv/bin/python -m inventory_di.cli doctor
```

---

## 3. Reproducible CLI Workflow

### A. Synthetic Demo Mode (`DEMO / SYNTHETIC` — Runs Out-of-the-Box)

```bash
.venv/bin/python -m inventory_di.cli doctor
.venv/bin/python -m inventory_di.cli make-demo --config configs/demo.yaml
.venv/bin/python -m inventory_di.cli prepare --config configs/demo.yaml
.venv/bin/python -m inventory_di.cli train --config configs/demo.yaml --model all
.venv/bin/python -m inventory_di.cli backtest --config configs/demo.yaml
.venv/bin/python -m inventory_di.cli report --config configs/demo.yaml
```

*(Add `--quick` to `train` or `backtest` for a fast smoke budget.)*

### B. Walmart M5 Mode (`data_mode="m5_sales_proxy"`)

Per `AGENTS.md` §1.10 & §4, raw M5 competition files are not committed to Git. To run M5 mode:
1. Accept competition rules and download the dataset from [Kaggle M5 Forecasting — Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data).
2. Place `calendar.csv`, `sell_prices.csv`, and `sales_train_evaluation.csv` (or `sales_train_validation.csv`) into `data/raw/m5/`.
3. Execute the exact same pipeline with `configs/m5.yaml`:

```bash
.venv/bin/python -m inventory_di.cli prepare --config configs/m5.yaml
.venv/bin/python -m inventory_di.cli train --config configs/m5.yaml --model all
.venv/bin/python -m inventory_di.cli backtest --config configs/m5.yaml
.venv/bin/python -m inventory_di.cli report --config configs/m5.yaml
```

### C. Launch the Offline Streamlit Decision-Experiment Dashboard

```bash
PYTHONPATH=src .venv/bin/streamlit run app/streamlit_app.py
```

### D. Run Automated Tests, Linter & Codex Verification Skills

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .

# Run deterministic Codex skill verifiers
.venv/bin/python .agents/skills/optimization-verification/scripts/verify_optimization.py --self-test
.venv/bin/python .agents/skills/decision-focused-learning/scripts/verify_dfl_contract.py --self-test
.venv/bin/python .agents/skills/data-leakage-audit/scripts/audit_leakage.py --manifest data/demo/lineage_manifest.json --scan-dir src
.venv/bin/python .agents/skills/experiment-reproducibility/scripts/verify_reproducibility.py --manifest artifacts/demo/run_manifest.json
.venv/bin/python .agents/skills/dashboard-qa/scripts/verify_dashboard_qa.py --bundle artifacts/demo/dashboard_qa_bundle.json
```

---

## 4. Documentation & Artifacts Index

- [`docs/METHODOLOGY.md`](./docs/METHODOLOGY.md) — Complete mathematical formulation, marginal-unit cost derivation, PyEPO `SPOPlus` architecture, and $\mathcal{P}_\alpha \equiv \text{CVaR}_\alpha$ DRO duality.
- [`docs/ASSUMPTIONS_AND_LIMITATIONS.md`](./docs/ASSUMPTIONS_AND_LIMITATIONS.md) — Explicit provenance disclosures (`observed sales != true demand`, synthetic `MU` economics, tail-risk dependence on scenario distributions).
- [`docs/EXPERIMENT_PROTOCOL.md`](./docs/EXPERIMENT_PROTOCOL.md) — Chronological split rules, leakage prevention gates, and 7-policy evaluation protocol.
- [`docs/PRESENTATION_GUIDE.md`](./docs/PRESENTATION_GUIDE.md) — 7-slide interview talk track (~8–10 min) and 60-second offline live-demo runbook for the MR. D.I.Y. interview.
- [`docs/IMPLEMENTATION_STATUS.md`](./docs/IMPLEMENTATION_STATUS.md) — Current verified status across Phases 0–7.
- [`reports/results_summary.md`](./reports/results_summary.md) — Computed 7-policy out-of-sample results table and scientific verdict.
