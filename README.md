# Decision-Focused and Distributionally Robust Inventory Allocation

A CPU-based interview research prototype for decision intelligence: learn weekly demand distributions, allocate scarce warehouse units, and compare realized inventory cost against strong baselines. M5 sales are a demand proxy; inventory and operating costs are synthetic MU. No actual retailer savings or enterprise deployment readiness are claimed.

The corrected implementation and verification plan are in [docs/INTERVIEW_READINESS_IMPLEMENTATION_PLAN.md](docs/INTERVIEW_READINESS_IMPLEMENTATION_PLAN.md). Current evidence and release limitations are in [docs/INTERVIEW_READINESS_REPORT.md](docs/INTERVIEW_READINESS_REPORT.md). Historical v1 claims are archived separately.

## Install

Use Python 3.11 or 3.12; the verified local runtime and dependency versions are recorded in the release evidence. From the project root:

```bash
uv sync --locked --group presentation --group qa
uv run python -m inventory_di.cli doctor
```

Alternatively create/activate a Python virtual environment, then install the package. The locked uv workflow is the reproducible reference; a plain editable pip install resolves the version ranges in `pyproject.toml`.

```bash
python -m venv .venv
# Activate using your platform's virtual-environment command.
python -m pip install -e .
python -m inventory_di.cli doctor
```

After activation, the `python` and `streamlit` commands below work across platforms. Installation/preparation may need downloads; the prepared dashboard runs locally from cached artifacts without proprietary APIs, GPUs or commercial solver licenses.

## Synthetic demo

```bash
python -m inventory_di.cli make-demo --config configs/demo.yaml
python -m inventory_di.cli prepare --config configs/demo.yaml
python -m inventory_di.cli train --config configs/demo.yaml --model all
python -m inventory_di.cli backtest --config configs/demo.yaml
python -m inventory_di.cli report --config configs/demo.yaml
```

For a smaller smoke budget, add `--quick` consistently to prepare/train/backtest/report. Quick and standard profiles use different run IDs and directories. Standard configs target `demo_v2`, `m5_v2` and `m5_intermittent_v2`; old v1 outputs are historical.

## M5

Accept the [Kaggle M5 competition rules](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data), download the files yourself, and place `calendar.csv`, `sell_prices.csv`, and `sales_train_evaluation.csv` (or the shorter `sales_train_validation.csv`) in `data/raw/m5/`. Raw data are ignored by Git.

Run the same prepare/train/backtest/report commands with `configs/m5.yaml` or `configs/m5_intermittent.yaml`. Cohorts use training-only mean/nonzero-rate filters. Their actual daily ADI values and chosen IDs are in the lineage manifest; an ADI <= 1.25 threshold is not assumed. Both M5 modes include 5 stores, 10 selected HOUSEHOLD SKUs and chronological weekly origins.

To execute all corrected standard runs sequentially:

```bash
python scripts/run_release.py
```

Training includes matched joint lambda ablations and independent seeds with equal budgets. Runs can take substantial CPU time. Stage timings and logs are in `reports/release/`; no fixed few-second regeneration promise is made. Validated train/backtest stages are reused when their configuration, inputs, outputs and scientific-source hashes match. Use a new run ID for a deliberate independent rerun.

## Dashboard and presentation

```bash
streamlit run app/streamlit_app.py --server.port 8511 --server.headless true --browser.gatherUsageStats false
python scripts/generate_presentation_deck.py
```

Open `http://localhost:8511`. Pages A–E show provenance, models, allocations, risk/service controls and evidence. The UI does not retrain. Expected cost, model CVaR and realized test cost are separated; synthetic uplift is labeled accordingly. Bundle hashes/configuration are validated, and changed files invalidate the cache.

Use [docs/PRESENTATION_GUIDE.md](docs/PRESENTATION_GUIDE.md), [speaker notes](docs/PRESENTATION_SPEAKER_NOTES.md), and the [offline demo guide](docs/OFFLINE_DEMO_GUIDE.md). PowerPoint, HTML and Markdown share one artifact-derived content source. Browser QA also exports a local PDF fallback.

## Verification

```bash
pytest -q
ruff check .
python scripts/verify_release.py
python scripts/browser_release_qa.py --slides
```

Tests generate isolated integration fixtures rather than require existing ignored demo outputs. Independent release verification requires the three standard v2 pipelines to have completed. Browser QA requires the server on port 8511, the QA dependency group and an installed Chrome; it uses an isolated headless browser and blocks non-loopback HTTP requests. It does not close an existing user browser.

The mathematical suite covers the mandatory two-store oracle, marginal-unit identity, CVaR ambiguity equivalence, hard versus soft service, and the dispatch/admission horizon oracle against independent exhaustive enumeration. Run the project skill self-tests and actual bundle/lineage audits as described in their SKILL.md files; their results complement the tests and independent checks.

## Scientific interpretation

Seven required policies are compared with two risk-neutral scenario ablations. Heuristic baselines are marked feasible, not cost-optimal. A more complex method need not win: choose from validation evidence and report held-out wins, losses and uncertainty honestly.

The v2 reruns use previously inspected test periods and are retrospective. Short horizons, synthetic economics, sales censoring, selected cohorts, geometric tail/dependence assumptions and domain shift limit conclusions. Carryover uses explicitly renewed warehouse dispatch budgets and synthetic holding/terminal assumptions. Full methodology is in [docs/METHODOLOGY.md](docs/METHODOLOGY.md); limitations in [docs/ASSUMPTIONS_AND_LIMITATIONS.md](docs/ASSUMPTIONS_AND_LIMITATIONS.md).
