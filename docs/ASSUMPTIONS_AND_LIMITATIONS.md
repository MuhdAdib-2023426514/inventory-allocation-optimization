# Assumptions and limitations

- Observed M5 sales are an uncensored-demand proxy for this experiment. Historical stockouts may hide demand. No causal uplift or actual retailer savings are established.
- Stock, supply, capacity, costs and interventions are synthetic; monetary values use MU rather than RM. The model is a research prototype, not an automated ordering system.
- Cohorts are small, training-selected HOUSEHOLD subsets. Walmart-to-MR.DIY transfer and enterprise scale are unverified.
- Corrected v2 runs reuse previously inspected test periods. They are retrospective validation; future confirmation requires untouched data or a prospective pilot.
- Twelve M5 test origins and three training seeds are limited evidence. Paired bootstrap intervals are descriptive; CVaR90 at twelve origins uses only about 1.2 origins of tail mass.
- The geometric overflow shape is a pooled training assumption. Temperature scaling and rank dependence do not guarantee calibrated future demand or protection against unseen regimes.
- DRO protects against probability reweighting within the sampled scenario support. It does not guarantee service under every possible real-world demand outcome.
- Warehouse dispatch budgets are renewed externally each week in carryover simulations. Shelf and deferred inventory incur holding, scheduled pipeline does not; terminal inventory has no salvage value. These assumptions materially influence rankings.
- Normal admission rules are deterministic; the hindsight oracle can optimize admissions with future information. The resulting regret is a hindsight benchmark, not an achievable forecast.
- One-week-ahead features and residual inventory are causal approximations, not fully probabilistic lead-time forecasts.
- Service constraints refer to expected portfolio fulfillment across scenarios, not per-store or every-scenario guarantees. Expected service shortfall is not an exact warehouse replenishment requirement.
- Sensitivity plots on test data are descriptive. Selecting a policy from these plots would require a new evaluation set for an independent performance claim.
- Solver timing is hardware-, instance- and configuration-dependent. Deployment scaling requires separate benchmarks and real operational constraints such as case packs, volumes, lead times and transport.
- Automated demo checks do not substitute for the candidate's personal delivery rehearsal.
