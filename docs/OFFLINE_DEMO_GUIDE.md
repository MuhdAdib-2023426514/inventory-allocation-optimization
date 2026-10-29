# Offline interview demo

Run from the repository root after generating the v2 artifacts. Installation and model preparation require available dependencies/data; the prepared dashboard demonstration runs locally without model training or external APIs.

```bash
uv sync --locked --group presentation --group qa
uv run python -m inventory_di.cli doctor
uv run streamlit run app/streamlit_app.py --server.port 8511 --server.headless true --browser.gatherUsageStats false
```

Open `http://localhost:8511`. Start before the interview. Use the completed release verification report to confirm which modes passed browser/offline checks.

The 60-second path:

1. Page A: select regular M5, show “M5 sales proxy + simulated inventory”, MU and the chronological split.
2. Page C: show feasible shipments and warehouse/store constraints. Download the selected-origin allocation CSV.
3. Page D: show three separate metrics: model expected cost, model CVaR and realized historical test cost. Defaults come from validation. Change supply to 0.8 or uplift to +25%; uplift is labeled simulated. Use hard service mode to inspect target feasibility. Return controls to defaults.
4. Page E: show the observed winner, the proportional baseline, paired uncertainty and the risk-neutral scenario ablations. Do not claim universal DFL+DRO superiority.

If the app cannot load, use `docs/presentation_slides.pdf` or the local HTML deck and `reports/release/browser/` screenshots. Do not attempt to rebuild models in the interview. A missing/stale-artifact warning requires preparation beforehand; regeneration can take minutes, not a guaranteed few seconds.

If port 8511 is occupied, choose 8512. After replacing artifacts, reload the page; cache signatures include file modification times and sizes. An inconsistent bundle is rejected with a regeneration message. A solver error is an error, not proof of a physical stock deficit.

This is a research demonstration using synthetic economics. No actual inventory order is sent. Candidate delivery rehearsal is a personal step; the prepared script is not evidence that a rehearsal occurred.
