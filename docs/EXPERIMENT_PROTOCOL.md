# Corrected experiment protocol

The machine-readable pre-run specification is `reports/release/experiment_protocol.json`. Current configs identify v2 runs; historical v1 configs and results remain separate. Quick profiles append `_quick` to run IDs and all generated directories. Standard and quick outputs cannot overwrite each other.

Run the synthetic quick profile first, then synthetic standard, regular M5 standard, and intermittent M5 standard. Standard budgets are fixed in YAML; seeds 42, 101 and 202 receive identical epochs, batch sizes and selection procedures. Positive lambda candidates define the DFL family; the zero-lambda model is retained as the supervised baseline. All selection uses validation. Fixed model weights are used through each test backtest; weekly feature updates use information available at that origin.

Main endpoint: paired held-out realized cost versus proportional allocation and supervised PTO. Secondary endpoints: oracle regret, fill, shortage, holding, shipping, origin-tail CVaR90, inference/sampling/solver runtime, and feasibility. The same realized target and feasible inventory/economic instance applies to every policy at an origin.

Use moving blocks of two origins and 2,000 bootstrap samples for standard paired intervals. Record small-origin and effective-tail-mass caveats. Preserve negative and inconclusive findings. These corrected runs reuse previously inspected test periods and must be labeled retrospective.

Risk settings minimize 0.85 validation mean cost plus 0.15 validation CVaR90. Matched checkpoints/lambda candidates minimize mean validation NLL plus 0.15 validation decision regret per SKU. Each seed repeats its own calibration and selection.

Predeclared diagnostics include full-origin risk grids, first-origin supply stress, and first-origin frozen-model scenario-count, tail-mean, support-rebin and shortage-ratio sweeps. They are not used to select policies on test data. No success criterion requires DFL or DRO to beat a baseline.

Stage manifests record effective configuration, input/output hashes, environment and source/worktree provenance. Missing, modified or incompatible artifacts fail acceptance. Scientific MILP results require optimal termination, zero relative MIP gap, independently checked feasibility and recomputed objectives. Heuristic solutions are marked feasible, not cost-optimal. Failed oracles and negative regrets stop reporting.

No raw M5 data are committed. Download from Kaggle after accepting its rules; put files in `data/raw/m5/`. Synthetic outputs are identified as `synthetic_demo`; M5 outputs as `m5_sales_proxy`, always with synthetic operating economics.

Execution note: after the verified synthetic runs, the two independent M5 cohorts may execute concurrently in separate directories to use available CPU. Their model budgets, splits and selection rules remain unchanged. Validated stages are reused rather than retrained when the sequential driver reaches an already completed cohort. Full source snapshots are recorded; changes to scientific modules invalidate cached models/results, while bookkeeping or presentation-only changes do not by themselves require new training.
