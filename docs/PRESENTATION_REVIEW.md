# Interview presentation review — 2026-10-29

**Verdict: a substantial, runnable research prototype; the existing full presentation is not ready to present unchanged.** The single-period experiment can support an honest technical case study. The carryover headline and blanket readiness certification cannot currently be defended. This review supersedes earlier claims that every scientific and operational gate passed.

The review inspected ingestion/features/splits, PMF training and calibration, SPO+ integration, nominal/DRO/baseline solvers, simulation, evaluation/provenance, dashboard, tests, and presentation sources. Existing user edits were retained. No full training or M5 experiment rerun was performed. Targeted corrections below do not refresh saved experiment artifacts.

## Evidence independently checked

- Initial suite: **23 passed in 7.05 seconds**. Ruff passed. Python 3.12.14, open-source solver and real PyEPO SPO+ backward smoke test passed (gradient norm 7.7972).
- All five skill verifier self-tests passed. Leakage manifests/static scan, reproducibility manifests, and dashboard QA bundles passed for demo, regular M5, and intermittent M5. These checks validate their particular assertions; they do not prove the absence of scientific defects.
- Independently recomputed allocation feasibility, exact realized cost components, fulfilled/unmet units, and fill rate for **224 saved policy/origin instances**: 56 demo, 84 regular M5, 84 intermittent M5. No discrepancies. Summary tables recompute exactly; recorded artifact, dataset, and checkpoint hashes match.
- Streamlit AppTest rendered all **five pages in all three modes** without runtime exceptions before changes. After corrections, **30 page/control checks** passed with zero runtime exceptions, including uplift, supply stress, opening stock, alpha zero and risk-weight zero. This exercises Python app behavior, not visual layout or browser JavaScript.
- Final test suite: **25 passed, 1 expected failure in 8.92 seconds**; Ruff passed. The strict expected failure records the unresolved R2 oracle counterexample and must not be described as a passing oracle acceptance gate.
- A local dashboard server started successfully on port 8511. Playwright MCP could not acquire its browser because the browser profile was already in use. **Fresh browser visual/console/offline-network verification remains outstanding.** Existing QA JSON booleans are not a substitute for that check.
- Aggregate verification evidence and paired bootstrap results: [presentation_review_evidence.json](../reports/presentation_review_evidence.json).

## Findings and repair trace

### R1 — Critical: nominal carryover policies valued stock already on hand again

**Location:** `src/inventory_di/evaluation/backtest.py`, `run_multi_seed_and_carryover_experiments`, `lead_time == 0` branch.

Cached `sup_mc`/`dfl_mc` were calculated at base opening inventory, usually zero. The rolling solver then used those coefficients with positive carryover stock. DRO scenario costs used the updated inventory. This disadvantaged nominal policies and confounds the reported 19.4% carryover advantage.

**Reproduction:** deterministic demand five, five units already on the shelf: stale coefficients ship five more and incur 6.25 MU; current-stock coefficients ship zero and incur zero cost.

**Correction made:** recalculate marginal coefficients using current stock. A regression test exercises the actual multi-period evaluator on two periods with realized demand `[0, 5]`; the correct mean nominal cost is 3.125 MU, previously 6.25 MU.

**Remaining:** rerun all three carryover experiments, rebuild reports/decks, and revise policy rankings. Existing saved carryover numbers are historical and must not be quoted as corrected evidence. Dashboard now warns explicitly.

### R2 — Critical: multi-period oracle is not a lower bound for the simulator

**Location:** `src/inventory_di/optimization/core.py::solve_multiperiod_carryover_oracle` and `simulation/simulator.py::InventoryStateTransitionSystem.step`.

The simulator can stage arrivals outside the store and deliver them later. The oracle enforces capacity on all arriving units and has no deferred queue. Its feasible set excludes some legal simulator trajectories. Conservation alone does not prove oracle equivalence.

**Reproduced counterexample:** one store/SKU, zero opening stock, weekly supply/cap 3, store capacity 5, demand `[0, 0, 5, 5]`, shortage penalty 6, holding 0.01, shipping zero. Orders `[3, 3, 3, 1]` obey dispatch limits and conserve stock, costing **0.13 MU** in the simulator. The oracle reports **12.02 MU**, giving **−11.89 MU regret** for a feasible policy.

**Status:** unresolved; the reproduction is an explicit strict expected-failure test. Align queue/capacity/holding/terminal rules and recheck against tiny enumerations. Either both systems permit the same staging mechanism or both prohibit it. Also stop comparison on a failed oracle solve: the current implementation can simulate zero orders after MILP failure.

### R3 — High: presentation claims exceed the verified evidence

**Locations:** `scripts/generate_presentation_deck.py`, slides in HTML/Markdown/PowerPoint, `PRESENTATION_GUIDE.md`, speaker notes, readiness report.

- Existing carryover headlines depend on R1/R2. Withdraw the 19.4% improvement and carryover policy recommendations until rerun.
- Measured regular-M5 mean runtimes are approximately **0.028 s supervised PTO, 0.030 s DFL PTO, 0.290 s supervised DRO, 0.156 s DFL DRO**. Claims of the entire comparison taking <50 ms or all MILPs <40 ms contradict saved artifacts. Runtime excludes forecasting, scenario generation, training, and UI reruns.
- The regular cohort's recorded training ADI spans **1.228–1.343**, so the repeated `ADI <= 1.25` description is false for most selected SKUs. The selection code uses nonzero-rate/mean filters, not that threshold. The intermittent cohort ADI is roughly **11.55–12.44**; clarify daily versus weekly ADI and avoid calling all intermittent series “lumpy” without variability diagnostics.
- The invented “8–15x” shortage-to-holding cost assertion in the slide generator is not established operational evidence. Configured synthetic ratios are around six. Remove unsupported company/network facts or source and date them.
- “Certified”, “fully rehearsed”, “proves stable”, and blanket “zero leakage” statements go beyond the checks performed. Use scoped, measured statements.

**Status:** warning added to the old readiness/status documents. Existing decks remain uncorrected and should not be used unchanged. Rebuild every presentation format from the same verified artifact source after scientific repairs; avoid separately hardcoding result values.

### R4 — High: DFL evidence logging could manufacture a positive gradient

**Location:** `models/dfl.py::train_joint_multi_sku_dfl_models`.

Reported joint gradient norms used `max(1e-4, measured_norm)`, hiding zero gradients. **Corrected:** logs now preserve the actual measured value. The standalone SPO+ gradient smoke test passes, but joint training should additionally enforce finite/nonzero measured participation and compare matched NLL-only versus joint SPO+ updates.

The joint model's final `val_decision_regret_mu` is also copied from the last training epoch, although the best checkpoint is restored and then temperature-calibrated. **Remaining:** recompute regret on the actual final calibrated model. Saved historical joint gradient/final-regret diagnostics are not independently repaired by changing this code.

### R5 — High: seed robustness and DFL attribution are overstated

**Locations:** `evaluation/backtest.py` seed loop, supervised/joint DFL training loops.

Primary seed uses the full epoch budget; additional seeds use 60% (at least 10 epochs). They reuse scenario scale factors and DRO settings calibrated for the primary model. `training_seed_independent` is written as true even when configuration requests model reuse. This is exploratory variation, not a controlled seed-only experiment.

Supervised training uses full-batch updates; joint DFL uses minibatches and a different validation selection score. A per-SKU lambda ablation is not a matched joint multi-SKU ablation. Isolate SPO+ effects with identical architecture, initialization, batches, update budget, calibration protocol and checkpoint criteria, changing only lambda. Retune/calibrate independently within validation for each seed. Record effective training/config details and checkpoints.

**Status:** dashboard caveat added; reruns remain necessary for stronger claims.

### R6 — High: soft service slack does not prove missing warehouse units

**Locations:** `optimization/core.py::solve_service_level_constrained_milp`, dashboard Page D.

A finite penalty permits an economically optimal service miss even when the target is physically feasible. Counterexample: one available unit, one demanded unit, shipping 100 MU, shortage/holding zero, slack penalty 50 MU/unit, target 90%. Solver ships zero and pays 45 MU penalty even though shipping one satisfies the target. Shortfall may also arise from store caps/capacity, not warehouse stock. The result is expected unfilled demand below target, not a required warehouse replenishment quantity.

**Corrections made:** dashboard describes a penalized service target and expected service shortfall, handles failed solves separately, and no longer claims maximum achievable service or proven physical deficit. The fractional slider now displays 0.85 clearly rather than rounding it to 1%.

**Remaining:** if a hard service promise is required, first solve with zero slack; on infeasibility separately maximize fulfillment or minimize shortfall, then minimize cost at the established service optimum. Validate ratio-of-expected-units semantics and low/zero-demand denominators.

### R7 — High: scenario adjustments confound the isolated value of DRO

**Location:** `models/distributions.py::EmpiricalCopulaScenarioGenerator`.

Validation multiplicative scaling changes scenario demand marginals after inverse-CDF sampling, while nominal PTO coefficients use the unscaled PMF. Therefore PTO-versus-DRO gains combine distribution correction and risk optimization. Label this transformation explicitly and add a stochastic risk-neutral policy using the same calibrated scenarios to isolate risk aversion.

Ties are assigned ranks using stable chronological order. In zero-heavy intermittent series, this can create artificial cross-series dependence from ordering tied zero observations. Use a reproducible tie treatment and verify dependence on independent tied-demand fixtures.

Overflow calculations also differ: marginal survival beyond H+1 decays geometrically with fixed factor 0.5; scenarios use a shifted Poisson tail fitted from training; overflow quantiles use a conditional mean as a placeholder. These are different distributions. Align survival, scenario inverse CDF, moments and quantiles; test high opening inventory and perform tail/sample-count sensitivity before making strong tail-risk claims.

### R8 — Medium: zero supply created one phantom warehouse unit

**Location:** `evaluation/backtest.py::build_allocation_instance_from_config_df`.

`max(1, ...)` raised configured zero supply or zero supply stress to one. **Corrected** to preserve zero; two zero-supply cases have a regression test. Existing default runs used positive supply, so their saved single-period values are unaffected by this correction.

### R9 — Medium: future M5 price filling and join contracts need stronger enforcement

**Location:** `data/m5.py` price fill/joins and `io.py` validators.

The combined price fill uses ungrouped `.bfill()` after grouped forward filling. Missing prices can draw on later observations or another series before lag construction. This is a potential leakage/cross-series path, not evidence that current cached complete-price rows are wrong. Use historical-only per-series filling, explicit missing indicators or a documented fixed fallback. Add missing-price and future-price perturbation tests.

Calendar and price joins should enforce unique keys, many-to-one merge validation and row-count conservation. Current validators mostly check column presence and a few properties; strengthen finite/integer/nonnegative checks, rectangular cohorts, nulls, date ordering, consistent capacities, and unique policy/origin rows.

### R10 — Medium: provenance and solver acceptance can silently under-check failures

**Locations:** `io.py::get_git_commit`, `evaluation/backtest.py` manifests, nominal/DRO solvers.

Manifests record a commit without dirty-worktree status or a source snapshot hash. Quick mode changes epochs/scenario counts under the same run ID/output paths and hashes only original YAML. A partial retrain can replace diagnostics for the other model family. Record effective configuration, source/worktree state, per-stage lineage and unique profile/run namespaces.

Solver failures return zero allocations and may flow into reports. Enforce optimal status and verified feasibility before scientific ranking, and record solver gaps. DRO currently returns a recomputed objective as `solver_objective`, losing the original solver objective rather than comparing the two independently. Compare and retain both.

### R11 — Medium: dashboard interpretation and demonstration scope

**Correction made:** changing demand uplift now labels the evaluated target cost **Simulated Uplift Test Cost**, not realized historical test cost.

Remaining: Page D defaults can differ from validation-selected risk settings; state clearly when exploring a new policy. Precomputed Pareto tables are based on their recorded origin/assumptions and do not track every live control. Avoid selecting hyperparameters on the displayed test frontier and claiming the same test data provide independent confirmation. Page C should expose total store capacity and expected excess/shortage alongside allocations. App cache keys currently do not include artifact modification times. Refresh/restart after artifact replacement.

### R12 — Medium: reproducibility/presentation packaging

`python-pptx` is imported by the presentation generator but absent from `pyproject.toml`/`uv.lock`; it happens to be installed locally. Add it to a pinned presentation dependency group. Test installation and `doctor` outside the repository: `doctor` imports the repository `scripts` package, which the wheel package list does not include. Tests currently depend on generated, ignored demo artifacts; document preparation before tests or provide isolated fixtures. Do not ship third-party raw M5 data to solve this packaging problem.

Browser visual review, PPTX text overflow review, actual offline-network verification, and rehearsal on the presentation laptop remain necessary. A precomputed QA bundle does not certify them.

## Defensible interview narrative now

“I built a CPU-based, end-to-end inventory decision experiment with genuine SPO+ training and mathematically defined CVaR probability-reweighting DRO. I evaluated it chronologically on M5 sales proxies with synthetic inventory economics. The best method depends on demand regime and objective; sophisticated methods do not consistently win.”

Regular M5 mean single-period cost: proportional **1298.91 MU**, DFL PTO **1312.70 MU**, supervised DRO **1315.14 MU**, supervised PTO **1322.76 MU**, DFL DRO **1328.63 MU**, equal **1432.35 MU**. Therefore DFL+DRO is not the mean-cost winner. DFL PTO's improvement over supervised PTO is about **0.76%**, with a descriptive paired moving-block 95% interval for the difference of **[−19.19, +0.48] MU** (12 origins, block size 2, 2,000 resamples). Do not claim established statistical superiority. Realized CVaR90 from 12 weeks reflects only about 1.2 weeks of tail mass and is especially unstable.

Intermittent M5: supervised DRO has the lowest non-oracle mean cost **171.30 MU**, followed by DFL DRO **172.59 MU**. DFL PTO loses to supervised PTO on each of the three saved seeds. Present that negative result explicitly. These are simulated economic outcomes on historical sales proxies, with no claim of realized MR.DIY savings.

Recommended 8–10 minute delivery: business decision and constraints; cost/toy example; chronological data/provenance; supervised versus genuine DFL and explicit DRO; one honest single-period results table; one dashboard what-if; limitations and a proposed monitored pilot. Keep full mathematical derivations, exploratory carryover and parameter sweeps as appendix material.

## Release gate

For the existing full presentation: resolve R2; rerun carryover after R1/R2; correct and synchronize every slide/note/report; complete browser/PPTX/offline review. For a narrower interview case study now: remove carryover savings, universal-win claims, physical-shortage proofs and unsupported timing/company claims; disclose the remaining experiment limitations. A strong candid research prototype is presentable. The existing blanket certification is not.
