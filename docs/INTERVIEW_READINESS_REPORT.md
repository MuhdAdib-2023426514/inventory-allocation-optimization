# Interview readiness report — corrected v2 release

**Technical verdict: ready for an interview research-prototype presentation in the verified local environment.**
This does not establish production readiness, actual retailer savings, or universal model superiority.

- Automated suite: **47 passed**, no known-defect expected failures; Ruff passed.
- Independent numerical checks: **288 policy/origin instances** across three modes; costs, feasibility, summaries, hashes and regret checks passed.
- Full synthetic, regular-M5 and intermittent-M5 pipelines completed, with equal-budget independent seeds and validation selection.
- A clean installed package completed the synthetic pipeline outside the checkout; quick-run checkpoint hashes and decision metrics reproduced exactly (runtime excluded).
- Browser: all 15 page/mode combinations, risk controls and CSV downloads passed in an isolated browser with external HTTP blocked; no console or page errors.
- Offline: dashboard server ran under the recorded OS loopback-only policy; local health succeeded and an external IP connection was denied.
- Presentation: seven slides exported through local Microsoft PowerPoint; complete expected text, canvas bounds and footer separation checked; HTML slides checked and PDF fallback exported.

## Scope and limitations

Results are retrospective on previously inspected test periods, with M5 sales proxies, synthetic MU economics, small selected cohorts, short test horizons and three seeds. Primary result tables identify the predeclared seed; seed variation is retained in reports. No statistical or operational guarantee is implied.
Candidate delivery rehearsal is prepared, not claimed performed. Personally rehearse the nine-minute talk and sixty-second dashboard path before the interview. The software checks cannot certify speaking delivery or an untested future laptop setup.

## Finding closure

| Finding | Implementation and evidence |
|---|---|
| R1 | Current-stock marginal costs; rolling regression; new v2 carryover runs |
| R2 | Dispatch/admission/deferred oracle; independent enumeration; replay and nonnegative regret checks |
| R3 | Artifact-derived seven-slide decks, notes and measured results; unsupported claims removed |
| R4 | Raw SPO+ gradients, matched update check and recomputed calibrated-checkpoint validation regret |
| R5 | Matched joint learning, equal seed budgets and seed-specific validation/risk selection |
| R6 | Hard/soft service modes, separate maximum-fill solve and brute-force service verification |
| R7 | Common coherent tail/calibrated marginals, randomized ties and risk-neutral scenario ablations |
| R8 | Zero supply retained; regression tests |
| R9 | Historical-only prices, unique validated joins and stricter finite/integer/temporal contracts |
| R10 | Effective configs, source/input/output lineage, profile isolation, accepted solver statuses and objective checks |
| R11 | Cache invalidation, validation defaults, complete CSV metrics and offline browser/AppTest checks |
| R12 | Locked presentation/QA groups, installed-package doctor and clean pipeline, actual PowerPoint/HTML layout checks |

## Evidence

See `reports/release/`: pipeline/clean execution logs, independent verification, skill checks, browser QA/screenshots, offline probe, presentation content/source hashes and PowerPoint verification. The release manifest identifies the exact artifact files.

Use `PRESENTATION_GUIDE.md`, `PRESENTATION_SPEAKER_NOTES.md` and `OFFLINE_DEMO_GUIDE.md`. Historical v1 presentations are archived and must not be used as current evidence.
