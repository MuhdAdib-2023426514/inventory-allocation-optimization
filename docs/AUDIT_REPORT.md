# Technical Audit & Baseline Assessment

**Project:** Decision-Focused and Distributionally Robust Inventory Allocation  
**Role:** Principal Decision Scientist, Optimization Researcher & Senior AI Software Engineer  
**Date:** 2026-10-29  
**Target:** Data Scientist Executive (Decision Intelligence), MR.DIY  

---

## 1. Executive Summary

This audit assesses the codebase against the 12-phase technical specification, identifying mathematical, simulation, and experimental issues, verifying already resolved components, and establishing the remediation roadmap.

| Area | Status | Critical Findings |
| :--- | :---: | :--- |
| **Phase 1: Information Timing & Causality** | **DEFECT DETECTED** | In `run_multi_seed_and_carryover_experiments` (lines 737–740 of `backtest.py`), when lead time $L > 0$, the evaluator selected forecast inputs using `target_idx = min(len(origin_inputs) - 1, o_idx + lead_time)` and `target_df = origin_inputs[target_idx]["origin_df"]`. This exposes the decision at origin $t$ to `lag_1w_units` from week $t+L$, which is the realized demand of week $t$ (which has not occurred at decision time). |
| **Phase 2: Inventory Conservation** | **DEFECT DETECTED** | In `backtest.py` (lines 708–712), when arriving shipments exceed remaining store capacity, incoming stock was scaled and floored: `x_arrived = floor(x_arrived * scale)`. The remaining units were discarded from the system without accounting, violating physical inventory conservation. |
| **Phase 3: Multi-Period Oracle** | **PARTIALLY VALID** | `solve_multiperiod_carryover_oracle` formulates a multi-period horizon MILP over $t=0 \dots T-1$. However, because the simulator destroyed excess arrived units for causal policies, the comparison between oracle and heuristic policies did not operate under an identical accounting boundary. |
| **Phase 4: Decision-Focused Learning (DFL)** | **VERIFIED** | Genuine DFL via PyEPO SPO+ is implemented in `FixedFeasibleMarginalAllocationModel` and `JointMultiSKUMarginalAllocationModel` with survival-probability cost mapping $\hat{c}[s,k] = t_s + h_s - (b_s + h_s)q[s,k]$. Non-zero gradients and NLL divergence are confirmed. |
| **Phase 5: DRO & Ambiguity Formulation** | **VERIFIED** | Scenario-reweighting DRO with ambiguity set $\mathcal{P}_\alpha = \{w \ge 0 : \sum w_j = 1, w_j \le p_j / (1-\alpha)\}$ matches Rockafellar–Uryasev $\text{CVaR}_\alpha$. Limiting cases agree with risk-neutral optimization. Parameter tuning is conducted strictly on `split == 'val'`. |
| **Phase 6: Experiment Design & Regimes** | **EXPANDED** | Evaluates 7 required policies across 3 cohorts: Synthetic Demo, M5 Regular (`ADI <= 1.25`), and M5 Intermittent (`ADI >= 1.35`). Training seeds and scenario seeds are separated. |
| **Phase 7: Business Decision Intelligence** | **ACTION REQUIRED** | Lacks an explicit service-level-constrained MILP formulation ($\min \text{cost}$ subject to fill rate $\ge \beta$, with infeasibility slack diagnostics). Business decision table needs formalization. |
| **Phase 8: Streamlit Dashboard** | **ENHANCEMENT REQUIRED** | 5 pages exist and work offline, but need explicit service-level constraint controls, lead-time selector, and what-if simulation display for operational planners. |
| **Phase 9: Documentation & Provenance** | **ACTION REQUIRED** | Synchronize all reports, line items, and manifests to reflect the leak-free, conservation-verified simulator. |
| **Phase 10: Presentation & Interview Deck** | **ACTION REQUIRED** | Generate an editable 7-slide `.pptx` presentation deck with speaker notes, along with HTML/Markdown companion slides. |
| **Phase 11: Offline Demonstration Readiness** | **VERIFIED** | Dashboard runs locally on CPU without external APIs. Create formal offline demonstration runbook and troubleshooting checklist. |
| **Phase 12: Quality Gates & Verification** | **ACTION REQUIRED** | Implement property-based conservation tests, nonanticipativity tests, service-level tests, and compile `docs/INTERVIEW_READINESS_REPORT.md`. |

---

## 2. Detailed Technical Audit

### 2.1 Confirmed Defects

#### Defect 1: Future-Information Leakage in Multi-Period Carryover Evaluation (`Phase 1`)
- **Location:** `src/inventory_di/evaluation/backtest.py`, lines 737–740 and 760–800.
- **Code:**
  ```python
  target_idx = min(len(origin_inputs) - 1, o_idx + lead_time)
  target_inp = origin_inputs[target_idx]
  target_df = target_inp["origin_df"]
  ```
- **Analysis:** At weekly origin $t$, an order $x_t$ is placed that arrives at $t+L$. When $L=1$, the evaluator grabbed `origin_inputs[t+1]["origin_df"]`. In that dataframe, `lag_1w_units` corresponds to the sales realized during week $t$. At the decision timestamp (beginning of week $t$), week $t$ sales have not yet occurred. Passing `target_df` into `predict_sku_origin_outputs` leaks future sales into the neural network's feature inputs.
- **Correction:** The decision at time $t$ must be made using strictly $\mathcal{F}_t$ (the information available at the decision timestamp). To forecast demand for the arrival horizon $t+L$, features must be projected forward autoregressively using the model's own expected demand $\hat{D}_t$, never the ground-truth realization $D_t$.

#### Defect 2: Violation of Physical Inventory Conservation (`Phase 2`)
- **Location:** `src/inventory_di/evaluation/backtest.py`, lines 708–712.
- **Code:**
  ```python
  if np.sum(x_arrived[s_i, :]) > rem_c_arr[s_i]:
      scale = float(rem_c_arr[s_i]) / max(1.0, float(np.sum(x_arrived[s_i, :])))
      x_arrived[s_i, :] = np.floor(x_arrived[s_i, :] * scale).astype(int)
  ```
- **Analysis:** If arriving inventory exceeds store capacity, the excess units are simply scaled down and discarded. In physical supply chains, physical mass cannot vanish. Discarding units artificially deflates holding costs and distorts system accounting.
- **Correction:** Implement an explicit state-transition inventory simulator. When arrived inventory exceeds store capacity $C_s - \sum_p I_{s,p}$, the excess units are retained in transit (deferred delivery attempt in week $t+1$) or moved to a store overflow buffer. Every unit is tracked across warehouse stock, store on-hand stock, in-transit stock, and fulfilled demand, satisfying the conservation invariant:
  $$\text{Opening System Inventory} + \text{External Receipts} = \text{Closing System Inventory} + \text{Fulfilled Demand} + \text{Recorded Losses}$$

---

### 2.2 Already Resolved & Verified Components

1. **Mandatory 2-Store Mathematical Oracle (`AGENTS.md` §5):**
   - Verified in `tests/test_optimizer.py`. Equal split `30/30` yields $62.5\%$ fill rate and $30.0\text{ MU}$ cost; optimal split `40/20` yields $75.0\%$ fill rate and $20.0\text{ MU}$ cost.
2. **Marginal-Unit Survival Cost Identity (`AGENTS.md` §6):**
   - Verified across 30 random instances with $I_s \ge 0$:
     $$C(x, D) = C(0, D) + \sum_{s,p,k} c_{\text{true}}[s,p,k; D] \cdot y_{s,p,k}$$
3. **Rockafellar–Uryasev CVaR $\equiv$ Probability-Reweighting Duality (`AGENTS.md` §7):**
   - Verified in `tests/test_dro_equivalence.py` matching brute-force enumeration and LP duality on the ambiguity set $\mathcal{P}_\alpha = \{w \ge 0 : \sum w_j = 1, w_j \le p_j / (1-\alpha)\}$.
4. **Demand Regime Generalization (`P2.1`):**
   - Automated cohort selection based on training-only Average Demand Interval (`ADI`): Regular (`ADI <= 1.25`) vs. Intermittent (`ADI >= 1.35`). Evaluated across both regimes.
5. **Offline Dashboard & Python 3.12 Stack:**
   - 5 Streamlit pages run offline from cached Parquet/JSON artifacts with zero console errors.

---

## 3. Implementation Plan by Phase

1. **Phase 1 & Phase 2:**
   - Create `src/inventory_di/simulation/simulator.py` implementing `InventoryStateTransitionSystem`.
   - Maintain explicit state: warehouse inventory, store on-hand, pipeline in-transit, deferred arrivals, fulfilled demand, unmet demand.
   - Enforce system conservation invariant check on every single step.
   - Refactor `run_multi_seed_and_carryover_experiments` in `backtest.py` to use `InventoryStateTransitionSystem` and eliminate future-information leakage by projecting arrival-week features strictly using information available at decision origin $t$.
2. **Phase 3:**
   - Validate `solve_multiperiod_carryover_oracle` against the exact state-transition system and verify $\text{Rolling Regret} \ge 0$ across all policies.
3. **Phase 4 & Phase 5:**
   - Implement `solve_service_level_constrained_milp` in `src/inventory_di/optimization/core.py` to optimize cost under explicit minimum fill-rate / service-level constraints, with diagnostic slack relaxation when warehouse stock is severely constrained.
4. **Phase 6:**
   - Re-execute full pipeline on `configs/demo.yaml`, `configs/m5.yaml`, and `configs/m5_intermittent.yaml` with the leak-free, conservation-verified simulator.
5. **Phase 7 & Phase 8:**
   - Add the service-level optimization control and Pareto trade-off curve to the dashboard (`app/streamlit_app.py`) and formulate the retail decision policy framework.
6. **Phase 9, Phase 10 & Phase 11:**
   - Generate `docs/presentation_slides.pptx` (PowerPoint deck with notes), `docs/presentation_slides.html`, and update `docs/PRESENTATION_GUIDE.md`.
   - Compile `docs/OFFLINE_DEMO_GUIDE.md` and `docs/INTERVIEW_READINESS_REPORT.md`.
7. **Phase 12:**
   - Run complete test suite and verification skills.
