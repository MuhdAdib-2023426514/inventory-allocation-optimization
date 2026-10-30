# MR.DIY Decision Intelligence — 30-minute Presenter Guide

**Format:** 25 minutes narrated slides + 5 minutes live offline dashboard; Q&A afterwards.
**Evidence:** Walmart M5 sales as a demand proxy; modeled stock and synthetic Monetary Units (MU).
**Sources:** https://github.com/MuhdAdib-2023426514/inventory-allocation-optimization and corrected reports/m5_v2, reports/m5_intermittent_v2.

## Running order

- **01. Which store gets the last unit?** — target 1:00, cumulative 1:00
- **02. Three findings in one slide** — target 1:15, cumulative 2:15
- **03. Same stock, different outcome: allocation is the decision** — target 1:25, cumulative 3:40
- **04. Prediction errors and inventory decisions have different costs** — target 1:20, cumulative 5:00
- **05. Data: what is observed, and what is modeled?** — target 1:30, cumulative 6:30
- **06. From history to shipments, with the optimiser in the training loop** — target 1:35, cumulative 8:05
- **07. DRO: balance expected cost against expensive demand scenarios** — target 1:40, cumulative 9:45
- **08. How I made the comparison fair** — target 1:20, cumulative 11:05
- **09. Regular demand: proportional allocation sets a strong benchmark** — target 1:25, cumulative 12:30
- **10. Intermittent demand: the primary-run winner is not stable** — target 1:25, cumulative 13:55
- **11. Independent retraining exposes fragile model rankings** — target 1:30, cumulative 15:25
- **12. When orders arrive later, today's decision affects next week** — target 1:35, cumulative 17:00
- **13. Which inventory policy would I recommend to the business?** — target 1:40, cumulative 18:40
- **14. Live demo: change the constraint, then explain the decision** — target 0:35, cumulative 19:15
- **15. Evidence supports a smarter baseline—not guaranteed advanced-model savings** — target 1:30, cumulative 20:45
- **16. Limits that matter—and how I would address them** — target 1:25, cumulative 22:10
- **17. A controlled path from prototype to retail pilot** — target 1:20, cumulative 23:30
- **18. Get the decision right first. Prove complexity before deployment.** — target 1:30, cumulative 25:00

## Slide-by-slide speaking notes

### Slide 1: Which store gets the last unit?
**Target:** 1:00

Good morning, and thank you for the opportunity. This case study asks a question that inventory planners face every day: if the warehouse cannot satisfy every store, which store receives the next unit? I built an end-to-end decision-support prototype, then compared sophisticated learning and robust optimization methods against straightforward rules. I'll present the business problem, data, mathematical approach, evidence, a policy recommendation, and a five-minute live demonstration. The experiment uses Walmart M5 public sales as a demand proxy with explicitly simulated inventory economics. This is a research prototype, not a claim of actual MR.DIY savings.

**Likely question:** Why did you pick this problem?
**Answer:** Allocation is where forecasting meets money. Two teams can have the same forecast accuracy and very different stockout costs. It is also a problem where an analyst can do harm by shipping complexity that doesn't pay, so it is a good test of judgement.

### Slide 2: Three findings in one slide
**Target:** 1:15

I will begin with the answer so you know what to listen for. Compared with an equal split, proportional allocation reduced the mean simulated regular-demand cost by about 133 MU per week, or 9.3 percent. The paired interval for that improvement is well away from zero. But the advanced methods did not show a conclusive mean-cost advantage over the proportional rule in that same cohort. For intermittent-demand products, a forecast-then-optimize method looked better in the primary training run, yet the advantage did not persist across independent random seeds. That distinction is central: a model is not ready for business simply because one run looks promising. I recommend choosing a practical baseline and asking each more complex challenger to earn its place on untouched data.

**Likely question:** So your advanced method failed?
**Answer:** On this sample it didn't beat a strong simple rule, and I'd rather show that than hide it. The pipeline, verification and evaluation design are what transfer to a real business: they stop you deploying a model because it looks clever. With real stock records and costs the answer may change, and the same harness would show it.

### Slide 3: Same stock, different outcome: allocation is the decision
**Target:** 1:25

Consider a warehouse with 60 units and two stores. Store A demands 60 units while Store B demands 20. With an equal allocation of 30 units per store, Store A is short of 30 units and Store B receives 10 units it cannot sell. The network fulfils only 50 out of 80 units demanded. A more sensible allocation is 40 to A and 20 to B. That fulfils 60 units, a 20 percent increase in fulfilled units and a fill-rate increase from 62.5 to 75 percent, without buying any additional inventory. In this simple illustration there are other equally good allocations; I am not claiming 40/20 is unique. The example assumes demand is already known. Actual planners face uncertainty, shared capacity, shipment limits and many competing SKUs. That turns the allocation task into a constrained optimization problem.

**Likely question:** Isn't this just ranking stores by expected demand?
**Answer:** With one item and no limits, nearly. With shared warehouse supply, shelf capacity across items and uncertain demand, it becomes an integer optimisation: the value of the next unit depends on how many that store already has and how likely it is to sell.

### Slide 4: Prediction errors and inventory decisions have different costs
**Target:** 1:20

Here is why forecasting accuracy is not the same as decision quality. Under my synthetic economics, failing to meet one unit of demand incurs a six-MU shortage penalty. Holding an unsold unit costs one MU, and sending any unit costs another quarter MU. The net effect of an extra shipment is therefore negative 5.75 MU if the unit prevents a shortage, because we avoid six MU but pay shipping. If it remains unsold, the same shipment adds 1.25 MU of cost. That asymmetry makes the value of the next unit depend on its probability of being needed. A conventional average-error measure does not directly account for warehouse competition or shipment economics. My outcome metric is the realized cost of the resulting allocation; I also compare it with a perfect-information hindsight benchmark. This is a conceptual distinction, not a claim that MAE is inherently wrong.

**Likely question:** Why not use quantile regression at the critical ratio?
**Answer:** That answers 'how much for one store with unlimited supply'. When stores share a limited warehouse and shelf space, the quantile alone can't decide who to cut. You still need the optimiser; the question is what to feed it.

### Slide 5: Data: what is observed, and what is modeled?
**Target:** 1:30

Before we discuss the algorithms, I want to separate real observations from the simulated operating environment. The public Walmart M5 dataset contains historical daily product sales, calendar features, store identifiers and product prices. I use five stores and ten household-category products in each of two demand cohorts. The model forecasts seven-day demand at weekly decision origins. Importantly, recorded sales are only a proxy for true customer demand: a zero may reflect lack of stock rather than lack of customer interest. M5 does not provide the complete warehouse balances, shelf capacities, freight rates and stockout economics required for this allocation study, so those inputs are explicitly simulated. I use 100 training, eight validation and twelve testing weekly origins. The corrected v2 report is a retrospective rerun after earlier test inspection, so the results should not be described as fresh, confirmatory performance. This boundary keeps the analysis useful without pretending it proves actual RM savings.

**Likely question:** What would make this conclusion change?
**Answer:** A longer, prospective evaluation using real stock balances, calibrated business economics and the incumbent operational policy.

### Slide 6: From history to shipments, with the optimiser in the training loop
**Target:** 1:35

The architecture separates prediction from prescription. At each historical decision origin, the forecast model sees only information available at that time and predicts a discrete probability distribution for demand next week. I convert those probabilities into expected marginal costs for shipping each additional unit. An integer optimizer then chooses the units per store while respecting warehouse supply, store shelf space and shipment caps. The ordinary predict-then-optimize baseline trains the forecaster to assign high probability to the observed demand. Decision-focused learning adds an optimization-aware training signal: the SPO-plus surrogate. It uses solutions of a structured optimization problem to penalize predicted coefficients that would induce costly choices. It is a surrogate, not literally the observed business regret in each gradient step. The controlled experiments match the two models' architecture and training budget so the additional training signal can be tested fairly. I also verify nonzero decision-focused gradients.

**Likely question:** How do you know the optimiser actually influences training?
**Answer:** I test it. There's an automated check that the decision-focused gradient is finite and non-zero, and that switching it off changes the parameter updates. Without that check, a lot of 'DFL' implementations are just regression with extra steps.

### Slide 7: DRO: balance expected cost against expensive demand scenarios
**Target:** 1:40

The second method is distributionally robust optimization, or DRO. Even when we forecast a full demand distribution, that estimate can be wrong. My implementation generates demand scenarios from the trained model and permits a bounded reweighting of their probabilities. The worst expectation over this defined ambiguity set equals CVaR, a measure of mean loss in the adverse tail. The actual optimization objective blends ordinary expected cost with the CVaR component. The confidence parameter alpha controls which tail is emphasized, while rho controls how much weight to give the tail relative to the average. These parameters are selected using the validation period, not by looking at final test results. Setting rho to zero removes the tail preference. It is also true that alpha equal to zero makes CVaR equal the scenario mean. DRO is useful only if risk reduction justifies the additional cost and the forecast represents the tail adequately.

**Likely question:** Why would robust optimisation ever cost more?
**Answer:** Because it deliberately buys insurance. If the forecast's uncertainty is too wide, or bad weeks don't arrive in the test period, you pay the premium without collecting. That is what happened here on average, which is why I report average and tail cost separately.

### Slide 8: How I made the comparison fair
**Target:** 1:20

A fair experiment matters as much as the optimization method. I use weekly chronological origins: 100 for training, eight for validation, and twelve for held-out evaluation. The final test segment represents a retrospective comparison, not a new independent dataset: the corrected v2 test period was inspected in an earlier version of the project. All candidate policies face the same realized demand proxy, the same warehouse supply and store constraints, and the same synthetic unit economics. I also compare against equal split and a strong proportional rule based on past sales, not just against a weak machine learning baseline. My independent verification scripts re-evaluate the feasibility and accounting of 288 policy-origin allocations. Finally, I compute paired moving-block bootstrap intervals because weekly results are serially related. With only twelve test weeks, these intervals are descriptive and especially weak for tail-risk claims.

**Likely question:** How do you handle stockouts hiding true demand in sales data?
**Answer:** M5 sales are a proxy: when a store ran out, true demand was higher than recorded sales. I state this as a limitation. In production I'd combine sales with on-hand stock records and estimate the hidden demand, for example with a censored-demand model.

### Slide 9: Regular demand: proportional allocation sets a strong benchmark
**Target:** 1:25

The first result is for regular-selling products. The chart uses hindsight regret rather than raw cost; the bars are each policy's mean regret and the whiskers show descriptive 95 percent block-bootstrap intervals. Lower values are better. The proportional rule achieved a mean simulated cost of 1,298.91 MU, compared with 1,432.35 MU for equal split. That is a difference of 133.44 MU, or 9.3 percent. This is our most persuasive result because the paired interval excludes zero. The advanced strategies had mean costs quite close to the proportional rule and their paired difference intervals include zero. I would call that an inconclusive difference, not proof of equality. The proportional rule remains 278.76 MU above a hindsight oracle. That gap expresses the value of perfect future knowledge under the model; it does not prove the gap is impossible to narrow with better data or policy design.

**Likely question:** Is a 12-week test enough?
**Answer:** It's enough to see big effects, like the 133 MU gain over equal split. It isn't enough to separate methods that differ by 5 to 10 MU. That's why I show intervals and say 'tie' instead of picking a winner.

### Slide 10: Intermittent demand: the primary-run winner is not stable
**Target:** 1:25

Now consider products with intermittent sales. Here the proportional rule performed worse than in the regular-demand cohort, with a mean cost of 176.94 MU. In the primary training run, supervised DRO had the lowest mean cost at 168.89 MU. But that numerical winner is not automatically an established improvement. The paired interval for DRO versus proportional allocation includes zero. Supervised forecast-then-optimize, at 169.94 MU, has a narrower paired difference of minus seven MU with an interval excluding zero for that particular trained model. We will see on the next slide why even that result needs caution: the learned models are not consistent across training seeds. The right business interpretation is to investigate forecasting-based policies for intermittent products, but not promote one as proven simply because it has the smallest primary-run number.

**Likely question:** What would make this conclusion change?
**Answer:** A longer, prospective evaluation using real stock balances, calibrated business economics and the incumbent operational policy.

### Slide 11: Independent retraining exposes fragile model rankings
**Target:** 1:30

This slide tests whether the observed ranking survives retraining. I independently retrained the learned models using three initialization seeds, while holding the data split and training budget fixed. For regular-demand products, the proportional baseline remained the lowest-cost non-oracle option in all three comparisons. Intermittent demand is more revealing. With seed 42, supervised DRO costs 168.89 MU and looks best. With seeds 101 and 202, its mean costs rise to approximately 184.5 and 180.1 MU, respectively. The simple equal allocation policy remains at 174.45 MU. Thus the policy that looked best in the first run no longer wins after retraining. Model-training randomness is different from demand uncertainty: the former comes from initialization and optimization of neural weights, while the latter concerns future sales. The experiment does not prove forecasting is useless. It shows that I should not present a one-seed improvement as stable business savings.

**Likely question:** Why not just report the best seed?
**Answer:** Because the business doesn't get to choose the seed. If performance depends on a random number, the honest expectation is the spread, and here the spread covers the simple rule.

### Slide 12: When orders arrive later, today's decision affects next week
**Target:** 1:35

A single-period model chooses shipments for one week and scores what happens in that week. Real retail operations are more complicated because unsold stock carries forward and orders can be delayed. In the multi-period version I track opening stock, dispatch, shipments in transit, store admissions, deferred arrivals, realized sales and closing stock. This avoids inventing or losing units during simulation. The corrected report evaluates immediate replenishment and a one-week lead time. For regular-selling items, DFL with nominal optimization has the smallest reported mean cost at both lead times: 1,208.89 MU for immediate replenishment and 1,401.18 for a one-week lead. Compared with supervised PTO at 1,421.36 MU, the delayed-case improvement is about 1.4 percent. DFL plus DRO is slightly more expensive here. In the separate intermittent-demand carryover experiment, supervised PTO performs best on mean cost, reinforcing that no method dominates across demand regimes. These exploratory numbers have not been validated in an untouched prospective horizon.

**Likely question:** What would make this conclusion change?
**Answer:** A longer, prospective evaluation using real stock balances, calibrated business economics and the incumbent operational policy.

### Slide 13: Which inventory policy would I recommend to the business?
**Target:** 1:40

The solution I would recommend is a policy-selection framework, not unconditional deployment of DFL plus DRO. For regular-demand single-period allocation, proportional allocation is a strong and simple benchmark, and it was the lowest-cost strategy in all three training-seed comparisons. For intermittent demand, the learned approaches changed rankings after retraining, so I would keep the incumbent or a simple baseline while running challengers in shadow mode. For repeated replenishment, DFL with nominal optimization is promising in the regular-demand simulation, but its gain is small and should be challenged on fresh data. If the business needs a minimum fill rate, that should become an explicit constraint or feasibility condition; minimizing modeled cost alone is not enough. When extreme shortages would be particularly damaging, DRO is a candidate, but its tail-risk benefit has to be validated against real penalties and calibrated scenarios. My overarching principle is to select the least complex policy that reliably meets the business objective.

**Likely question:** What would make this conclusion change?
**Answer:** A longer, prospective evaluation using real stock balances, calibrated business economics and the incumbent operational policy.

### Slide 14: Live demo: change the constraint, then explain the decision
**Target:** 0:35

I will now show the working decision-support tool. First I will open the data provenance page so that the business assumptions are transparent. Next, I will show the current allocation by store and SKU and confirm that supply and capacity constraints are respected. Then I will reduce warehouse availability to 60 percent of the original setting and rerun the optimization. The point is not simply that a number changes: the tool must now ration the remaining stock, and we can compare which stores and items lose units, what happens to predicted shortages, and how expected cost changes. I will also explain the difference between a scenario-model risk metric and realized test cost. If there is time, I will adjust the risk weight to show what a more conservative decision looks like. Finally I will return to the evidence page and the policy comparison. This demo lasts five minutes; model training is precomputed and it runs locally without external APIs.

**Likely question:** Could planners use this?
**Answer:** As a decision-support view, yes: it shows the allocation, why, and what changes if supply drops. It doesn't place orders; a planner stays responsible. For production it would need real stock feeds, case-pack rounding and an override log.

### Slide 15: Evidence supports a smarter baseline—not guaranteed advanced-model savings
**Target:** 1:30

What has the project demonstrated? First, a demand-aware proportional allocation reduces modelled cost materially relative to equal splitting in the regular-demand cohort. That comparison is about 133 MU per week or 9.3 percent. Second, the optimization layer supports explicitly constrained, integer allocation decisions and can be solved quickly at this research scale. Third, the verification and testing system surfaces unstable results instead of letting us present a lucky run as proof. Equally important is what I cannot claim. I do not have actual MR.DIY inventory or cost data. I cannot translate MU into Malaysian Ringgit without calibrated margin, holding and freight assumptions. The experiment has five stores and ten products per demand cohort, not a national-scale network. And the advanced methods did not show a robust mean-cost win over the strong regular-demand proportional benchmark. The business benefit today is a validated decision-support methodology and an evidence-based recommendation process, not a guaranteed deployed savings figure.

**Likely question:** How would you turn MU into Ringgit?
**Answer:** Replace the synthetic costs with real ones: lost gross margin for a stockout, cost of capital and space for holding, freight per unit. Then rerun the same harness on real stock and sales and multiply weekly regret differences by volume, with intervals.

### Slide 16: Limits that matter—and how I would address them
**Target:** 1:25

I want to be explicit about the experiment's limits because they determine the recommendation. Recorded sales can be censored by stockouts, so they may understate true demand. I would combine sales with stock-on-hand and stockout records, potentially using censored-demand estimation. Second, warehouse supply and economic costs are synthetic. Real gross margins, shipment costs, storage costs and contractual lead times might change policy rankings. Third, the twelve held-out weekly origins offer a narrow view of demand regimes; tail-risk estimates are particularly noisy at that sample size. Fourth, these corrected v2 results are retrospective: the test periods were inspected while repairing the previous implementation. A fresh prospective test is needed before claiming confirmatory superiority. Finally, multi-period experiments do exist, but they are separate exploratory evaluations, not an enterprise-scale inventory planner. Every limitation has a practical remedy, which forms the basis of the pilot roadmap.

**Likely question:** What would you change if you started again?
**Answer:** Lock the test period away and pre-register the comparison before building anything, and start from real stock records instead of sales alone.

### Slide 17: A controlled path from prototype to retail pilot
**Target:** 1:20

If I were implementing this with MR.DIY, I would use staged decision gates rather than deploying an optimization algorithm across every store immediately. The first phase is shadow mode: connect the real inventory and sales feeds, build transparent assumptions, and calculate recommendations without placing orders. We can validate data quality, feasibility, forecast calibration and modelled differences from planner allocations, but shadow mode alone does not measure realized counterfactual savings. The second phase is a controlled pilot on selected stores and products. Planners retain approval authority and all overrides are logged with reasons. We then evaluate actual service, shortage, excess stock and cost outcomes against an appropriate comparison group. The third phase is scaling only after the pilot meets agreed cost and service gates. Scaling must account for shared shelf capacity across SKUs. Independent SKU optimizations cannot simply be combined if their aggregate capacity is infeasible; sparse joint optimization or a tested decomposition is required.

**Likely question:** What is the biggest risk in rollout?
**Answer:** Planner trust and data quality, more than algorithms. Shadow mode, visible reasons for each allocation and an override log address both.

### Slide 18: Get the decision right first. Prove complexity before deployment.
**Target:** 1:30

To conclude, I developed a complete data-to-decision pipeline: probabilistic demand forecasting, integer inventory optimization, decision-focused and risk-aware variants, reproducible experiments, and an offline decision-support dashboard. I also built verification around the accounting, constraints and learning mechanisms because a compelling model is not sufficient if its decisions are wrong. The strongest finding is that even a simple change from equal to proportional stock allocation can materially improve simulated business outcomes. The more advanced models are useful research candidates, but they did not establish consistent superiority in these tests. I would therefore recommend beginning with a transparent, reliable allocation baseline, defining service and risk objectives with the business, and only promoting more complex models after forward-looking validation. That combination of model building, skeptical evaluation and practical recommendation is how I would approach Decision Intelligence at MR.DIY. Thank you; I would welcome your questions.

**Likely question:** Why should we hire you?
**Answer:** I can take a business decision from data to a working, tested tool, and I report what the evidence says even when it isn't the exciting answer. That protects the business from expensive mistakes.

## Five-minute live demonstration

1. 0:00–0:45 — Show data provenance, M5 sales-proxy and synthetic MU disclaimer (Page A).
2. 0:45–1:45 — Show baseline allocations and store / warehouse constraints (Page C).
3. 1:45–3:10 — Reduce supply to 0.60×, rerun optimizer, explain who loses stock and why (Page D).
4. 3:10–4:10 — Compare expected cost with modeled tail risk; reset controls to defaults (Page D).
5. 4:10–5:00 — Show evidence page and export, then transition to business recommendation (Page E).

## Pre-presentation checklist

- Open the revised HTML locally, full-screen; preload the Streamlit dashboard on port 8511.
- Make sure all data/model artifacts needed by the dashboard are present locally.
- Disable Wi-Fi as a rehearsal and verify no remote requests are required.
- Keep the PPTX version ready if your browser or HTML controls fail.
- Correctly describe all improvements as simulated MU; no measured MR.DIY savings.
- Do not infer equivalence merely because a confidence interval includes zero.
- Keep backup model equations and full results for technical questions.