# Decision policy framework

Use the corrected result tables in `reports/m5_v2/` and `reports/m5_intermittent_v2/`. Observed winners are specific to each cohort, objective and synthetic operating regime. This framework does not prescribe actual MR.DIY replenishment orders.

| Decision | Evidence to inspect | Action |
|---|---|---|
| Minimize average inventory cost | Paired differences against proportional and supervised PTO, with uncertainty | Retain the strongest validated policy, including a simple baseline when it wins |
| Reduce expensive tail outcomes | Same-scenario stochastic versus DRO comparison, realized tail diagnostics | Choose risk settings on validation; accept that protection can increase mean cost |
| Require expected service | Hard service solve and independently established maximum fill | Enforce a feasible target; investigate constraints when unattainable |
| Explore an economic trade-off | Soft penalty and resulting expected service | Treat shortfall as an economic outcome, not proof of physical shortage |
| Carry inventory through lead time | Conserved simulator and matched hindsight oracle | Treat rankings as conditional on replenishment, holding, admission and terminal assumptions |
| Scale or deploy | Real stock/cost/lead-time data and prospective monitoring | Run a planner-reviewed pilot and benchmark the actual constraints before operational use |

Report disagreements across seeds, regimes and objectives. Small retrospective tests support a research discussion, not a guarantee of production savings.
