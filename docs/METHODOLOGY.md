# Methodology — corrected v2 experiments

This is a retrospective inventory decision experiment using M5 observed sales as a demand proxy and synthetic operating economics. It is not evidence of actual savings at Walmart or MR.DIY.

## One-week decision

For integer shipment x, observed demand proxy D and opening stock I, realized cost is

`sum[b * max(D-I-x, 0) + h * max(I+x-D, 0) + t*x]`.

Constraints are warehouse supply per SKU, shipment caps per store/SKU, and total store stock capacity `sum_p(I+x) <= C`. Costs are synthetic MU/unit. Feasibility and cost are recomputed independently from allocations. Exact MIP solves require accepted optimal status and zero reported relative gap; a failed solve raises an error. Equal/proportional policies are labeled feasible heuristics rather than cost-optimal solutions.

## Chronology and contracts

Every row predicts seven calendar days after its forecast origin. Demand lags/rolling features use completed weeks; target-week calendar events are known in advance. Prices are forward-filled within sorted item/store histories with a fixed missing fallback and lagged before use. Calendar/price joins require unique keys and preserve row counts. Whole-origin train/validation/test blocks prevent random-row mixing.

Cohorts, inventory assumptions, support recommendations and tail parameters use training history. Feature normalization is fitted on train. Temperature, lambda selection and risk settings use validation. No test target is used for these fits. The test periods were inspected in v1; the corrected v2 evaluation is therefore retrospective, not a new confirmatory holdout.

Cohort filters and daily ADI values are in the lineage manifest. Regular selection uses mean/nonzero-rate filters, not a claimed ADI <= 1.25 threshold. Intermittent does not automatically mean lumpy.

## Matched demand learning

A per-SKU neural PMF predicts demand bins 0..H plus an explicit overflow bin. H covers the training-derived support recommendation, the configured support floor, and all fixed-training allocation thresholds. Actual targets are never clipped for economic evaluation. Overflow targets map to one categorical bin only for NLL indexing.

The conditional overflow excess is geometric on 1,2,... with training-estimated mean m (default two when no excess is observed). Thus `P(excess >= k) = (1-1/m)^(k-1)`. Survival, inverse CDF, expectation and quantiles use this same distribution. The overflow probability is learned; the conditional tail shape is a pooled training assumption.

Supervised and DFL models share architecture, initial weights, optimizer, minibatch order, epochs and checkpoint rule. Supervised uses NLL only. DFL uses `mean_NLL + lambda * SPOPlus / number_of_marginal_units`. Positive lambda candidates define the DFL family; lambda zero remains the matched baseline even if it wins.

The fixed joint multi-SKU PyEPO model uses prefix binary variables y[s,p,k], fixed warehouse budgets and shared store capacity. Realized marginal coefficients are `t-b` when `D >= I+k`, otherwise `t+h`. Therefore `C(x,D) = C(0,D) + dot(c_true,y)`. Predicted coefficients are `t+h-(b+h)*P(D >= I+k)`.

A true SPO+ backward pass and matched parameter-update smoke check verify optimizer participation. Raw gradient norms are retained without a positive floor. Validation checkpoints minimize mean NLL plus 0.15 times decision regret per SKU. After restoring the selected checkpoint, temperature is calibrated on validation and final decision regret is recomputed on the calibrated models.

## Scenarios and risk

Training-only rank-bootstrap uniforms represent cross-store/SKU dependence. Reproducible independent randomization of ties avoids assigning zero-demand ties the same chronological ordering across series. Inverse CDF maps uniforms into the calibrated PMF including its geometric tail. No scenario-only multiplicative scaling is applied. Common uniforms do not imply identical demand draws when forecast PMFs differ.

For nominal scenario probabilities p and losses L, the ambiguity set is

`P_alpha = {w: sum(w)=1, 0 <= w[j] <= p[j]/(1-alpha)}`.

Its worst expected loss is exactly finite-scenario CVaR_alpha. The robust objective is `(1-risk_weight)*E[L] + risk_weight*CVaR_alpha(L)`. Alpha zero or weight zero recovers risk-neutral expected cost. This is probability-reweighting DRO, not Wasserstein DRO.

Each family has a scenario risk-neutral diagnostic policy using the same scenarios as its DRO policy. These isolate the risk preference from finite-sampling effects. Main model-implied comparison metrics evaluate all allocations on common supervised reference scenarios at alpha 0.90. Optimization itself uses the appropriate model-family scenarios and validation-selected risk settings. Realized costs use held-out observed sales, never the model's sampled demand.

## Carryover state/action contract

Each week renews the synthetic warehouse dispatch budget W. This is an external replenishment assumption, not a model of finite central-warehouse stock depletion.

1. Opening store stock and due pipeline are known; deferred deliveries remain queued.
2. Dispatch is integer, nonnegative and bounded by W and per-store/SKU K. With L=0 it is due immediately; with L=1 it enters the next week's pipeline.
3. Admissions are explicit integer actions, cannot exceed queued units, and cannot exceed remaining store stock capacity.
4. Fulfillment is exactly `min(demand, stock after admissions)`. Closing store/deferred balances and the pipeline are conserved per SKU.
5. Store stock and deferred stock incur holding; dispatch incurs shipping. Scheduled pipeline has no holding charge. Terminal inventory has zero salvage value. Orders arriving after the horizon are zero.

Normal policies use a deterministic proportional admission rule. The clairvoyant horizon optimizer may choose admissions within the same action space using future demand. Its explicit store/deferred/pipeline balances and exact minimum-fulfillment formulation are independently replayed through the simulator. Tiny exhaustive dynamic programs verify its optimum and lower-bound property. Hindsight regret is never clipped to hide a negative value.

Nominal carryover coefficients are recomputed for current opening stock. For L=1, stock and arrival-week features are projected from currently available information, including due/deferred stock. Moving averages, carried calendar/event fields and future residual stock are approximate projections rather than a separately trained lead-time model.

## Service targets

Portfolio expected fill is expected fulfilled units divided by expected demand; zero demand has fill one. Hard mode first minimizes cost subject to the target with zero slack. If infeasible, a separate solve maximizes expected fulfillment; a third cost solve retains that maximum. Soft mode minimizes cost plus a chosen service shortfall penalty. Expected unfilled units below target are not a required warehouse replenishment quantity and do not identify a bottleneck by themselves.

## Evaluation

Seven required policies plus two scenario risk-neutral ablations share targets and operating assumptions. Three independent training seeds use equal budgets and their own validation calibration/risk selection. Paired moving-block bootstrap resamples chronological origins, not correlated store/SKU rows. Cost, regret, fill, tail cost, solver gaps and timings are recorded separately. Short horizons make intervals descriptive and tail estimates unstable.

Risk frontiers are descriptive test diagnostics, not hyperparameter selection evidence. First-origin sample-count, shortage-ratio, tail-mean and frozen-PMF support-rebin sensitivities do not establish general robustness. Full runs and source/input/checkpoint hashes are identified in the release manifest.
