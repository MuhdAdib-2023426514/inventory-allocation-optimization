# Slide 1: Inventory allocation: from forecasts to decisions

Interview research case study | Decision intelligence for multi-store retail

- Decision: how many units should a warehouse send to each store when stock and shelf space are limited?
- Objective: reduce shortage, holding and shipping cost while respecting integer inventory constraints.
- Deliverable: a CPU-based experiment pipeline, mathematical checks and an offline decision dashboard.
- Evidence: Walmart M5 sales proxies with synthetic operating economics; no claim of actual retailer savings.

# Slide 2: Data and assumptions define the experiment

Chronological evaluation and explicit provenance

- Regular and intermittent cohorts: 5 stores and 10 HOUSEHOLD SKUs each, selected using training history only.
- Each origin predicts the next seven calendar days. Main M5 split: 100 training, 8 validation and 12 test origins.
- Features use completed history; calendar events are known in advance. Future realized prices are excluded.
- Stock, supply, capacity and MU costs are synthetic. Observed sales can understate latent demand during stockouts.
- The v2 runs reuse previously inspected test periods: these are corrected retrospective experiments.

# Slide 3: Learning and optimization answer different questions

A distribution feeds the decision; SPO+ supplies training feedback

Pre-origin features → Calibrated demand PMF → Marginal costs / joint scenarios → Integer allocation

Training feedback: SPO+ updates the PMF through a fixed feasible optimization template.

- Matched learning: the same architecture, initialization, batches and budget; DFL adds normalized SPO+.
- Probability-reweighting DRO: sum(w)=1 and 0 <= w[j] <= p[j]/(1-alpha). Its worst expected loss equals CVaR.
- Risk-neutral scenario ablations isolate sampling from risk aversion; weight zero is expected-cost optimization.

# Slide 4: Test the decision, not only the forecast

Shared conditions and independent mathematical checks

- Seven required policies plus two diagnostic ablations use the same origin, stock, supply, capacity and realized target.
- Primary outcomes: held-out cost, hindsight regret and fill rate; forecast error is a diagnostic.
- Three independent seeds use equal budgets and their own validation calibration and risk selection.
- Paired moving-block intervals resample time origins. Small samples and few tail weeks limit certainty.
- Carryover extension: explicit dispatch, admissions and deferred inventory; the hindsight MILP is replayed and tested against exhaustive tiny cases.

# Slide 5: The strongest method depends on the setting

Primary seed 42: mean test-origin cost (synthetic MU; lower is better)

| Policy | Regular M5 | Intermittent M5 |
|---|---:|---:|
| Equal allocation | 1432.35 | 174.45 |
| Proportional allocation | 1298.91 | 176.94 |
| Supervised PTO | 1311.54 | 169.94 |
| Supervised DRO | 1320.18 | 168.89 |
| DFL PTO | 1305.72 | 179.67 |
| DFL + DRO | 1311.75 | 179.11 |

- Primary-seed non-oracle policies: regular winner proportional_allocation (1298.91 MU); intermittent winner supervised_dro (168.89 MU).
- Regular DFL+DRO minus supervised PTO: +0.20 MU; paired 95% band [-15.71, 10.66] MU.
- Primary-seed rankings vary across seeds and objectives. Full seed results are retained; universal superiority is not established.

# Slide 6: A decision tool makes the trade-off visible

60-second local demonstration; cached models, fresh optimization

- Page A: identify the M5-proxy badge, synthetic MU economics and chronological data split.
- Page C: inspect integer shipments, available supply, store capacity and expected shortage/excess.
- Page D: compare model expected cost, model CVaR and realized historical test cost as separate metrics.
- Change supply to 0.8 or demand uplift to +25%. Uplift is explicitly a synthetic intervention.
- Hard service mode meets a feasible target or separately establishes maximum achievable service. Soft mode permits a cost trade-off.
- Page E: show paired results and the baseline winner. Download CSVs; no training runs inside the UI.

# Slide 7: Recommendation: validate before an operational pilot

A defensible result includes its limitations

- Choose policy using the business objective and validation evidence; do not assume the most complex model wins.
- Limitations: sales censoring, synthetic economics, small selected cohorts, short test horizon and domain shift.
- Retrospective reruns do not create a fresh confirmatory holdout. Future evidence should include untouched periods.
- Next: obtain real stock and lead-time records, validate costs and case packs, then run a monitored planner-in-the-loop pilot.
- Demonstrated: a tested research workflow and offline prototype. Enterprise-wide scale and actual savings remain to be established.
