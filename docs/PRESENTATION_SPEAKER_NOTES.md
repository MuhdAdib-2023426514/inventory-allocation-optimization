# Speaker notes

Prepared talk track; candidate delivery rehearsal must be performed personally. Target: 9 minutes 0 seconds.

## Slide 1: Inventory allocation: from forecasts to decisions (65 seconds)

Open with the business decision rather than model complexity. Explain a scarce unit can be valuable at one store and excess at another. MU means synthetic monetary units. This is an interview research prototype, not an operational replenishment system.

## Slide 2: Data and assumptions define the experiment (75 seconds)

Show Page A provenance and actual filters. ADI is daily inverse nonzero-sales frequency; do not claim a threshold absent from the selection rule. Discuss Walmart-to-MR.DIY domain shift. Twelve test weeks support a small case study, not a causal savings estimate.

## Slide 3: Learning and optimization answer different questions (90 seconds)

Define PTO and DFL in plain language. The feasible set is fixed during SPO+ training; uncertain demand maps to objective coefficients. The optimizer participates in learning. This is probability-reweighting DRO, not Wasserstein DRO. Scenario dependence is estimated on training only; PMF temperature is selected on validation.

## Slide 4: Test the decision, not only the forecast (80 seconds)

The oracle knows future demand only for evaluation and never supplies a realistic recommendation. Explain why conserving total stock alone was insufficient and why matching the state/action contract matters. Show mathematical tests if asked. Negative or inconclusive model results are retained.

## Slide 5: The strongest method depends on the setting (90 seconds)

Read the table candidly, including heuristic wins and losses by advanced methods. The two stochastic ablations and oracle are available in the full report. Explain a negative paired difference favors DFL+DRO; an interval crossing zero is inconclusive on this sample. These costs are modelled outcomes, not revenue or proven operational savings.

## Slide 6: A decision tool makes the trade-off visible (70 seconds)

Leave the dashboard running before the interview. Use one controlled change, then return to defaults. Do not promise a latency threshold: actual solver and pipeline timings are in the artifacts. If the browser fails, use the local PDF or HTML and screenshots; do not rebuild models during the interview.

## Slide 7: Recommendation: validate before an operational pilot (70 seconds)

Close with a measured proposal. Describe the data needed from an ERP and warehouse system and how to compare against the incumbent policy. Keep humans responsible for orders. Invite questions about calibration, solver feasibility, simple baselines and why robust optimization can cost more.
