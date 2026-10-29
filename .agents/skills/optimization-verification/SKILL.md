---
name: optimization-verification
description: >-
  Independently verify mathematical optimization formulations, integer allocation
  feasibility, toy-instance optimality, marginal-unit cost identities, CVaR
  probability-reweighting equivalence, solver statuses, and risk-neutral limiting
  cases. Activate when implementing, modifying, debugging, or reviewing nominal
  MILP, PyEPO marginal-unit models, scenario-reweighting DRO, baseline allocators,
  or cost/regret evaluators.
---

# Optimization Verification Skill

This skill enforces independent mathematical verification for all inventory allocation solvers and cost evaluators defined in [`AGENTS.md`](../../../AGENTS.md) (§5, §6, §7, §8, §14). Never rely solely on a solver's reported objective value or a clean exit code as proof of mathematical correctness.

## Triggering Conditions

Invoke this skill automatically when:
- Implementing or modifying `src/inventory_di/optimization/` (`core.py`, `pyepo_fixed.py`, `dro.py`, `baselines.py`) or cost evaluation logic.
- Adding or changing constraints (warehouse supply $W_p$, store–SKU caps $K_{sp}$, total store capacity $C_s$, integer bounds, or optional minimum service constraints).
- Implementing or changing the marginal-unit binary encoding $y_{s,k}$ or incremental cost vector $c_{\text{true}}[s,k;d]$.
- Implementing or modifying the CVaR / scenario-probability-reweighting DRO MILP or risk-trade-off objective.
- Reviewing backtest outputs where `oracle_regret_mu` is negative or solver statuses are non-optimal.

## Required Inputs

1. **Authoritative Problem Contract**: [`AGENTS.md`](../../../AGENTS.md) §5 (Correct mathematical problem definition), §6 (Marginal-unit encoding), §7 (DRO formulation), and §14 (Math and optimization acceptance gate).
2. **Target Code or Candidate Solution JSON**: Either the optimization module under test or a serialized instance/solution JSON conforming to [`references/math_invariants.md`](./references/math_invariants.md).

## Execution Workflow

1. **Run the Independent Mathematical Oracle Self-Test**:
   Execute [`scripts/verify_optimization.py`](./scripts/verify_optimization.py) with `--self-test` to confirm all reference identities before testing project code:
   ```bash
   .venv/bin/python .agents/skills/optimization-verification/scripts/verify_optimization.py --self-test
   ```
2. **Verify the Mandatory Toy Mathematical Oracle (`AGENTS.md` §5)**:
   - 1 product, 2 stores, $I = [0, 0]$, $D = [60, 20]$, $W = 60$, $K = [60, 60]$, $b = [1, 1]$, $h = [0, 0]$, $t = [0, 0]$.
   - Confirm equal allocation $x = [30, 30]$ yields realized cost $30$, fulfilled units $50$, unmet units $30$, fill rate $62.5\%$.
   - Confirm optimal allocation $x^* = [40, 20]$ yields realized cost $20$, fulfilled units $60$, unmet units $20$, fill rate $75.0\%$.
3. **Verify Feasibility & Post-Solve Cost Recomputation**:
   - Check $x_{sp} \in \mathbb{Z}_{\ge 0}$, $\sum_s x_{sp} \le W_p$, $0 \le x_{sp} \le K_{sp}$, and $\sum_p (I_{sp} + x_{sp}) \le C_s$.
   - Recompute realized cost independently from integer allocations $x_{sp}$:
     $$C(x, D) = \sum_{s,p} \left[ b_{sp} \max(D_{sp} - I_{sp} - x_{sp}, 0) + h_{sp} \max(I_{sp} + x_{sp} - D_{sp}, 0) + t_{sp} x_{sp} \right]$$
   - Compare recomputed $C(x, D)$ against solver-reported objective within numerical tolerance ($10^{-6}$).
4. **Verify Marginal-Unit Encoding Identity (`AGENTS.md` §6)**:
   - For randomized instances including $I_s > 0$, verify prefix constraints $y_{s,k+1} \le y_{s,k}$, $x_s = \sum_{k=1}^{K_s} y_{s,k}$, and exact decomposition:
     $$C(x, d) = C(0, d) + \sum_{s, k} c_{\text{true}}[s, k; d] \, y_{s, k}$$
     where $c_{\text{true}}[s, k; d] = t_s - b_s$ if $d \ge I_s + k$, else $t_s + h_s$.
5. **Verify CVaR Ambiguity-Set Equivalence & Limiting Cases (`AGENTS.md` §7)**:
   - Confirm that the Rockafellar–Uryasev LP representation $\min_{\eta, z \ge 0} \left\{ \eta + \frac{1}{1-\alpha} \sum_j p_j z_j : z_j \ge L_j(x) - \eta \right\}$ matches the exact maximization over the probability-reweighting ambiguity set:
     $$\mathcal{P}_\alpha = \left\{ w \in \mathbb{R}^N_{\ge 0} : \sum_{j=1}^N w_j = 1, \; 0 \le w_j \le \frac{p_j}{1-\alpha} \right\}$$
   - Confirm limiting cases: $\alpha = 0$ (with `risk_weight = 1`) and `risk_weight = 0` (for any $\alpha \in [0, 1)$) both equal the risk-neutral expected loss $\sum_j p_j L_j(x)$.
6. **Verify Solver Status & Non-Negative Hindsight Regret (`AGENTS.md` §8)**:
   - Reject any solution whose solver status is not optimal (`optimal` / `ok`).
   - Confirm $\text{regret} = C(x_{\text{policy}}, D_{\text{actual}}) - C(x_{\text{oracle}}, D_{\text{actual}}) \ge -10^{-6}$ on identically constrained instances.

## Verification Procedures

To validate a specific optimization artifact or JSON payload:
```bash
.venv/bin/python .agents/skills/optimization-verification/scripts/verify_optimization.py --input path/to/instance_solution.json
```
Consult [`references/math_invariants.md`](./references/math_invariants.md) for the JSON payload format and exact mathematical assertions.

## Expected Outputs

- Structured JSON report containing `status` (`"PASS"` or `"FAIL"`), checked invariants, numerical discrepancies, and violated constraints.
- Non-zero exit code (`1`) on any mathematical identity failure, constraint violation, negative oracle regret, or non-optimal solver status.

## Completion Criteria

- `verify_optimization.py --self-test` exits with code `0` and all positive and negative (error-detection) checks pass.
- Any candidate solver implementation passes all 6 workflow checks without silent fallbacks that violate warehouse supply $W_p$ or store caps $K_{sp}, C_s$.
