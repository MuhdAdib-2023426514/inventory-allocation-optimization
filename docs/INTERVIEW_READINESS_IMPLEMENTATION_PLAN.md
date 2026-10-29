# Implementation plan: verified interview release

**Created:** 2026-10-29  
**Status:** Executed; technical release gates passed. Candidate personal rehearsal is prepared and remains a handoff step.
**Scope:** Close all R1–R12 findings in [PRESENTATION_REVIEW.md](PRESENTATION_REVIEW.md), regenerate the scientific evidence, and deliver a consistent, verified interview package. This plan does not certify the existing presentation.

## Definition of done

The release is ready when every required gate below has recorded evidence, no critical/high scientific defect remains, no known-defect expected failure remains, and all presentation claims can be traced to the frozen release artifacts. A method need not beat its baselines to pass. Honest negative or inconclusive findings are acceptable outcomes.

Preserve existing user edits and archive the current v1 results as historical. Use new run IDs and separate output directories for corrected experiments. Never overwrite old results and silently retain their claims. Keep M5 raw files local and ignored by Git.

## Execution order and dependencies

| Phase | Deliverable | Depends on | Findings closed |
|---|---|---|---|
| 0 | Reproducible release workspace and run protocol | — | R10, R12 foundations |
| 1 | Aligned simulator, horizon oracle, solver acceptance | 0 | R1, R2, R8, R10 |
| 2 | Leakage-safe ingestion and validated contracts | 0 | R9 |
| 3 | One coherent demand distribution and scenario engine | 2 | R7 |
| 4 | Matched supervised/DFL training and independent seed experiments | 1–3 | R4, R5 |
| 5 | Correct service-level optimization and diagnostics | 1, 3 | R6 |
| 6 | New end-to-end runs, paired evidence, sensitivity checks | 1–5 | Scientific reruns for R1–R10 |
| 7 | Dashboard aligned with new artifacts | 5–6 | R11; UI aspects of R3/R6/R7 |
| 8 | Artifact-derived reports, slides and interview narrative | 6–7 | R3, R12 presentation |
| 9 | Clean-install, browser, offline and presentation release gate | 0–8 | Final acceptance of R1–R12 |

Keep dependent work sequential. Run demo correctness checks before spending CPU on full M5 retraining. Use the project optimization, leakage, DFL, reproducibility and dashboard QA skills during the applicable phases. Fetch current Context7 documentation before library-specific API changes.

## Phase 0 — Freeze the starting point and establish reproducibility

**Implementation**

- Inventory current source changes, configs, dependencies, raw-data availability, historical artifacts and hashes. Preserve uncommitted user work.
- Define corrected run IDs such as `demo_v2`, `m5_v2`, and `m5_intermittent_v2`; isolate quick, standard, and per-seed outputs.
- Record the full effective config after overrides, seed, profile, code commit, dirty-worktree state, source snapshot hash, dependency versions and checksums. A YAML hash alone is insufficient.
- Make prepare/train/backtest/report stage manifests reference their actual inputs. Reject stale or mismatched data/checkpoint/config combinations with actionable errors.
- Keep partial model retraining from erasing the other family's diagnostics; merge by family and verify checkpoint hashes.
- Add a locked presentation dependency group including `python-pptx`. Package the environment doctor inside `inventory_di` so an installed wheel works outside the source checkout.
- Make unit tests independent of pre-existing ignored artifacts; reserve generated-artifact checks for documented integration fixtures.
- Define quick/standard budgets and profile the demo before estimating full-run duration. Log training and evaluation timings separately.

**Files:** `config.py`, `io.py`, `cli.py`, `pyproject.toml`, `uv.lock`, configs, environment scripts and artifact-dependent tests.

**Acceptance:** quick and standard runs cannot overwrite each other; changing effective config invalidates incompatible artifacts; a wheel installed in a temporary environment runs `doctor` outside the repository; the full run protocol is recorded before examining new test results.

## Phase 1 — Repair the inventory experiment and mathematical oracle

**Chosen design:** preserve deferred deliveries, but represent store admissions explicitly in the physical state/action contract. The simulator and hindsight optimizer must implement the same admissible inventory operations. Simply removing the expected-failure marker or changing the displayed regret is not a fix.

**Implementation**

- Specify event ordering: opening stock, due pipeline, deferred queue, store admissions, new dispatch, demand fulfillment, closing state and charged costs. Document replenishment of warehouse supply, per-period dispatch caps, admission capacity and horizon-end treatment.
- Extend the simulator action interface to accept validated admission quantities, with a documented deterministic admission rule for normal policies. Validate integer/nonnegative stock, demand, orders and admissions; dispatch supply/caps; pipeline length; available queued units; and store capacity.
- Extend the horizon MILP with deferred-stock balances and integer admissions. Include the same shipping, store holding, deferred holding, initial pipeline and terminal rules as the simulator.
- Ensure fulfillment is exactly `min(demand, available stock)`. Use an exact formulation where needed; do not allow shortage/excess slack to create stock or dispose of it implicitly.
- Replay both oracle dispatches and admissions through the simulator. Retain and compare raw MILP objective, replayed cost and conservation audit.
- Reject failed/nonoptimal oracle results before computing regret. Apply the same acceptance policy to every scientific policy comparison. Record termination and optimality gap; return an explicit excluded/error state rather than rank a fallback as a successful policy.
- Retain the current-stock nominal-cost correction and zero-supply correction. Audit L=1 projected stock, due/deferred inventory, admission constraints and feature projection for causality; identify projection approximations explicitly.
- Compare original solver objectives with independently recomputed objectives in nominal/DRO solvers. Preserve both values.

**Tests**

- Convert the existing deferred-delivery oracle counterexample into an ordinary passing test: a feasible 0.13 MU policy cannot have an oracle lower bound of 12.02 MU.
- Build an independent exhaustive enumerator for tiny horizons and compare its optimum, MILP optimum and simulator replay. Cover L=0/L=1, binding capacity, mixed SKUs, staging, initial pipeline, zero demand/supply, positive stock and horizon-end inventory.
- Check conservation per SKU, not only a grand total that could hide SKU substitution.
- Keep the two-period nominal carryover regression and zero-supply tests.
- Perturb future realized demand and confirm all earlier causal decisions are unchanged in the actual rolling evaluator.
- Inject solver failures and verify reports do not classify them as valid scientific results.

**Acceptance:** no expected-failure marker for R2; enumerated/MILP/replayed optimum agree within numerical tolerance; every feasible tested policy has nonnegative regret; policies and oracle obey the same state/action contract.

## Phase 2 — Close leakage and data-contract gaps

**Implementation**

- Replace future/cross-series price backfill with sorted historical-only per-series filling and a documented fixed fallback/missing indicator. Derive price availability dates from source records.
- Require unique calendar and price join keys, validated many-to-one joins, full calendar coverage and conserved row counts.
- Validate finite values, integer/nonnegative quantities, valid date windows, whole-origin chronological splits, complete store/SKU grids, consistent supply/capacity and unique policy/origin records.
- Record actual cohort filters and daily training ADI distributions. Keep the cohort selection rule explicit; remove unsupported `ADI <= 1.25` descriptions. Do not call a cohort lumpy without supporting variability diagnostics.
- Reconfirm all selection, inventory-cap, PMF-support and tail fitting use training only. Calibration and parameter selection use validation only.

**Acceptance:** duplicate joins and invalid contracts fail helpfully; perturbing later prices/demand cannot alter earlier features or training-derived cohort parameters; runtime lineage and source leakage checks pass on all three corrected datasets.

## Phase 3 — Use a consistent PMF, tail policy and dependence model

**Implementation**

- Adopt one explicit integer overflow distribution, initially a shifted geometric tail with training-estimated conditional mean excess. Use the same law for survival, inverse CDF sampling, expectation and quantiles at every inventory threshold.
- Correctly derive overflow quantiles from the conditional CDF; do not substitute the tail mean for every overflow quantile.
- Replace chronological tie ranking with reproducible randomized distributional ranks using independent tie randomization across dimensions. Test on independent zero-heavy series and known correlated fixtures.
- Remove post-sampling multiplicative demand scaling from the core comparison. Use the same validation-calibrated demand distribution for nominal and scenario policies; any retained scaling must be a separately labeled ablation with matching coefficient calculations.
- Maintain common uniform draws across policy families, while acknowledging that differing learned PMFs imply different demand scenarios. Evaluate competing allocations on a common reference scenario set when comparing model-implied risks.
- Add supervised and DFL scenario risk-neutral policies as diagnostic ablations alongside the seven required policies. Compare each with its robust counterpart on exactly the same scenarios.

**Acceptance:** sampled moments/quantiles/survival agree with analytic values within Monte Carlo tolerance, including thresholds beyond H+1; tied independent series do not acquire systematic artificial dependence; fixed seeds reproduce draws; alpha=0/risk_weight=0 recover the risk-neutral objective.

## Phase 4 — Make learning and seed comparisons controlled

**Implementation**

- Share a training driver for supervised and joint DFL variants: same PMF architecture, initial weights, minibatch order, optimizer, update budget, feature normalization, validation checkpoint rule and calibration procedure. Change only lambda for the controlled ablation.
- Keep the fixed feasible constraint template independent of each example's demand and invariant throughout SPO+ training.
- Measure pure joint SPO+ gradients before full training. Require finite/nonzero optimizer participation in a deliberately nondegenerate fixture and verify parameter-update divergence against lambda=0. Record raw measurements without floors.
- Recompute final validation decision regret after restoring the selected checkpoint and applying temperature calibration. Store the best epoch and final-model checksum.
- Run the joint lambda ablation under the same budget. Select on validation using a predeclared rule; retain NLL-only as the matched baseline.
- Train seeds 42, 101 and 202 with equal budgets. Fit/calibrate each seed's models and scenario diagnostics independently and retune DRO settings using validation only.
- Record reused-model experiments as scenario-seed experiments, not independent retraining. Save per-seed checkpoints, histories, calibration settings and outcomes.

**Acceptance:** changing lambda demonstrably changes the joint learning update; reported final regret matches independently recomputed calibrated-checkpoint regret; all seed comparisons differ only in recorded random seeds; no test target enters training or tuning.

## Phase 5 — Implement defensible hard service targets

**Implementation**

- Expose distinct hard-target and soft-penalty service modes with explicit meanings.
- For hard mode, minimize expected cost with zero service slack. If infeasible, separately maximize expected fulfillment under the physical constraints, then minimize cost at that maximum fulfillment. Report an unattainable target using the independently established maximum.
- Report service shortfall in expected unfilled units. Do not label it a required warehouse replenishment quantity or identify a warehouse bottleneck without a separate diagnostic solve.
- Define portfolio service as expected filled units divided by expected demand. Handle zero demand explicitly, and do not replace small positive expected demand with one in the denominator.
- Distinguish a failed solve from proved infeasibility. Retain the currently corrected dashboard labels until the hard-mode implementation passes verification.

**Acceptance:** the high-shipping-cost counterexample meets a feasible hard target; supply-, cap- and capacity-limited cases report verified maximum service; zero/low-demand cases are correct; brute-force tiny solutions match the solver.

## Phase 6 — Regenerate all scientific evidence

**Run order**

1. New isolated synthetic quick run: prepare → train → backtest → report, including carryover, service and failure-path checks.
2. New synthetic standard run.
3. New regular M5 and intermittent M5 standard runs, each with matched training and independent three-seed evaluation.
4. Predeclared sensitivity analyses: scarcity, shortage/holding ratio, alpha, risk weight, scenario count and tail support/parameters. Record which are validation selection, held-out diagnostics or independent synthetic stress tests.

**Evaluation requirements**

- Verify identical realized targets, inventory economics and feasible constraints across policies at every origin; keep the oracle solely a hindsight benchmark.
- Recompute allocations, costs, regret, fulfillment and solver status independently from saved data. Verify hashes and stage lineage.
- Export per-origin paired results, paired moving-block confidence intervals, mean/median/standard deviation, realized origin-tail CVaR and runtimes. Include comparisons against proportional allocation, not only equal allocation.
- Keep model-implied expected loss, model-implied CVaR and realized held-out loss separate. Record alpha for every risk statistic.
- Disclose small-origin/tail-mass limitations. Reusing the previously inspected test period makes the corrected rerun a retrospective validation, not a new untouched confirmatory test. Use a genuinely untouched later holdout only if available and declared in advance; otherwise explicitly retain that limitation.
- Time training, inference, scenario generation, individual solves and dashboard interactions separately.
- Report all wins, losses and inconclusive comparisons. Do not tune until DFL+DRO wins.

**Acceptance:** all required policies and seed runs complete with accepted solver outcomes; no unexplained negative regret or feasibility violation; reported metrics recompute; every result points to the new run ID; old carryover numbers disappear from current-facing outputs.

## Phase 7 — Finish dashboard behavior and user-facing interpretation

**Implementation**

- Load only internally consistent release bundles; include artifact signatures in cache invalidation and clearly reject incomplete/mixed runs.
- Initialize Page D from that family/run's validation-selected settings. Label changes as exploratory what-ifs.
- On Page C expose stock, supply, store/SKU cap, total store capacity, shipments, expected excess/shortage and solver acceptance status.
- Make live-control effects explicit. Recompute relevant plots or label precomputed charts with their actual origin/configuration; they must not appear to reflect unrelated live controls.
- Show corrected service modes and distinguish synthetic uplift cost from historical realized cost. Explain scenario risk-neutral ablations and any residual calibration differences.
- Keep M5 proxy and synthetic evidence visibly separate. Remove temporary historical warnings only after replacement artifacts pass their gates.
- Replace invented timing guarantees with measured timings and clear CPU/resource context. Resolve installed-version deprecations using current documentation.

**Acceptance:** AppTest covers all five pages/all three modes, risk-neutral boundaries, stress controls, service modes, CSV exports, stale/missing artifacts and solver errors; every displayed default metric matches its artifact.

## Phase 8 — Rebuild the interview package from verified artifacts

**Implementation**

- Build one presentation-content source from release manifests and computed summaries. Generate PowerPoint, HTML, Markdown, speaker notes and report tables from it. Add checks preventing stale run IDs or mismatched rounded numbers.
- Rewrite README, methodology, limitations, experiment protocol, policy framework, presentation guide, offline runbook, status and readiness report to match the corrected implementation.
- Remove unsupported company/network facts, invented operational cost ratios, universal improvement statements and production-scale promises. Use generic retail context unless an external fact is verified and sourced.
- Replace “certified”, “fully rehearsed” and absolute “zero leakage” claims with precisely scoped verification evidence.
- Structure seven slides for 8–10 minutes: decision problem; data/assumptions; methods; evaluation design; verified findings including strongest baseline; live decision example; limitations and proposed pilot.
- Keep detailed mathematics, per-seed results, sensitivity and carryover in an appendix unless they are essential and fully verified. Define MU, sales proxy, oracle, SPO+ and reweighting DRO in plain language.
- Prepare a 60-second dashboard path and static fallback with local HTML/PDF or screenshots. Replace unsupported controls in the existing runbook, such as a +30% uplift option that the UI does not offer.
- Write Q&A covering simple baselines, small sample size, sales censoring, synthetic costs, calibration, why DRO may lose, and what actual deployment would require. Distinguish proposed scale work from demonstrated scale.

**Acceptance:** every numeric claim traces to release artifacts, all formats agree, negative findings remain visible, and no measured experimental effect is described as actual MR.DIY savings. Rehearsal materials must say “rehearsal prepared” until an actual rehearsal is completed.

## Phase 9 — Execute the final release gate

**Automated verification**

- Run the full test suite and Ruff; require zero known-defect expected failures and investigate every skip relevant to release.
- Run all five skill self-tests plus actual lineage/reproducibility/dashboard bundle audits for each new run. Self-tests alone do not certify project correctness.
- Independently recalculate all final policy/origin allocations and metric tables, all checkpoint/data hashes, paired comparisons and presentation numbers.
- Install from the lockfile in a clean environment; run demo end to end and test the built wheel outside the repository.

**Interactive/offline verification**

- Obtain an isolated browser session without closing another user's active browser. Navigate Pages A–E in all modes, change risk/service controls, download CSVs and inspect console/runtime errors.
- Disable external network access while allowing loopback and confirm the dashboard completes the intended demo. Record required/optional network behavior rather than assume offline operation from cached artifacts.
- Render and inspect every PowerPoint slide and notes; check overflow, clipped formulas, table density and readable font sizes. Inspect HTML at the intended presentation resolution and test fallback files.
- Repeat the launch/demo path on the intended presentation machine. Measure cold/warm startup and interaction time; verify the guide's exact labels/options and fallback path.
- Prepare a timed rehearsal script and record actual rehearsals only when performed. Candidate delivery confidence remains a human check, separate from software acceptance.

**Final outputs**

- A release manifest identifying exact corrected datasets, models, results, source snapshot and presentation files.
- A readiness report with a pass/fail/evidence row for every gate, remaining limitations, verified commands and the demonstrated operating scope.
- A closure matrix linking every R1–R12 finding to code changes, tests and regenerated artifacts.
- Frozen presentation files, local dashboard bundle and static fallback. Freeze means evidence/artifact consistency; do not create a Git commit or publish anything without appropriate session authorization.

**Release decision:** declare “ready for an interview research-prototype presentation” only after the gates pass. Do not declare production readiness. If a gate fails, retain its explicit status and repair it; do not relabel it as passed because the demo happens to load.

## Tracking checklist

- [x] Phase 0: run isolation, provenance, packaging and clean test fixtures
- [x] Phase 1: simulator/oracle equivalence and solver acceptance
- [x] Phase 2: ingestion, missing prices and data contracts
- [x] Phase 3: consistent tails, calibration and dependence
- [x] Phase 4: matched learning, final diagnostics and independent seeds
- [x] Phase 5: hard service targets and shortfall diagnostics
- [x] Phase 6: all corrected experiments and independent evidence checks
- [x] Phase 7: complete dashboard and exports
- [x] Phase 8: synchronized presentations, documentation and fallback
- [x] Phase 9: clean-install/browser/offline/slide checks and final readiness report

No fixed completion-time promise is made before profiling the corrected CPU workload. Correctness gates precede expensive reruns; release readiness follows evidence, not a target metric or deadline.
