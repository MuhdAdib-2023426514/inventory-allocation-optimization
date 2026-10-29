---
name: decision-focused-learning
description: >-
  Implement and validate genuine Decision-Focused Learning (DFL) for inventory
  allocation using PyEPO SPO+ surrogate loss, marginal-unit survival cost mapping,
  fixed-feasible-set templates, gradient/parameter-update verification, and
  controlled ablations against a supervised NLL-only PMF baseline. Activate when
  building, training, debugging, or auditing demand distribution models, PyEPO
  integration, or DFL+DRO pipelines.
---

# Decision-Focused Learning (DFL) Skill

This skill enforces the scientific and architectural requirements for Decision-Focused Learning defined in [`AGENTS.md`](../../../AGENTS.md) (§1.5, §6, §7, §8, §14). A forecasting model followed by an optimizer is **not** DFL, and weighted regression must never be relabeled as DFL.

## Triggering Conditions

Invoke this skill automatically when:
- Implementing or modifying `src/inventory_di/models/dfl.py`, `src/inventory_di/models/supervised.py`, `src/inventory_di/models/distributions.py`, or `src/inventory_di/optimization/pyepo_fixed.py`.
- Defining the discrete demand PMF $p(d \mid \text{features})$ over $0 \dots H$ plus the `overflow > H` bin, or computing survival probabilities $q[s, k]$ and predicted marginal costs $\hat{c}[s, k]$.
- Setting up PyEPO `optModel` / `SPOPlus` training loops or caching training/validation oracle solutions.
- Connecting DFL-learned PMFs to the scenario generator for `dfl_dro`.

## Required Inputs

1. **Authoritative Contract**: [`AGENTS.md`](../../../AGENTS.md) §1.5, §6, §7 (Combined DFL + DRO policy), and §14.
2. **DFL Verification Reference**: [`references/dfl_checklist.md`](./references/dfl_checklist.md).
3. **Model / Training Specification or Artifact**: PMF outputs, marginal cost vectors $\hat{c}$ and $c_{\text{true}}$, constraint template specs, and gradient diagnostic logs.

## Execution Workflow

1. **Run the Deterministic DFL Contract & SPO+ Surrogate Verifier**:
   ```bash
   .venv/bin/python .agents/skills/decision-focused-learning/scripts/verify_dfl_contract.py --self-test
   ```
2. **Validate Discrete PMF & Explicit Overflow Bin (`AGENTS.md` §6)**:
   - Model outputs a valid categorical PMF over $\{0, 1, \dots, H\}$ plus an explicit `overflow > H` probability bin ($p \ge 0, \sum_{d=0}^H p_d + p_{\text{overflow}} = 1$).
   - Choose $H$ and tail-sampling parameters strictly from training data (`split == "train"`). Never silently clip realized evaluation demand $D_{sp}$ at $H$.
3. **Validate Differentiable Survival & Marginal Cost Mapping (`AGENTS.md` §6)**:
   - Compute survival probabilities:
     $$q[s, k] = P(D_s \ge I_s + k \mid \text{features}) = \sum_{d = I_s + k}^{H} p(d \mid \text{features}) + p(\text{overflow} > H \mid \text{features})$$
     (for $I_s + k \le H + 1$).
   - Compute predicted marginal cost vector:
     $$\hat{c}[s, k] = t_s + h_s - (b_s + h_s) \, q[s, k]$$
   - Assert $q[s, k+1] \le q[s, k]$ and $\hat{c}[s, k+1] \ge \hat{c}[s, k]$ for all $k = 1, \dots, K_s - 1$.
   - Assert exact flattening order alignment `[store s, unit k]` across PyTorch, PyEPO, MILP solver, and evaluator.
4. **Enforce Fixed-Feasible-Set Invariance (`AGENTS.md` §6)**:
   - PyEPO `SPOPlus` requires uncertain parameters only in the linear objective and an **instance-invariant feasible set**.
   - Never pass uncertain demand as a right-hand side constraint or vary $W_p$, $K_s$, or $I_s$ across examples sharing the same PyEPO model template.
5. **Verify SPO+ Gradient Participation & NLL Baseline Ablation (`AGENTS.md` §6, §14)**:
   - Train `supervised_pto` with NLL only ($\lambda_{\text{DFL}} = 0$) and `dfl_pto` / `dfl_dro` with composite loss $\mathcal{L} = \text{NLL} + \lambda_{\text{DFL}} \, \mathcal{L}_{\text{SPO}+}(\hat{c}, c_{\text{true}}, y^*, z^*)$.
   - Run a one-batch backward pass test verifying that:
     1. Gradients with respect to model parameters are finite and non-zero.
     2. Parameter updates under $\lambda_{\text{DFL}} > 0$ differ measurably from $\lambda_{\text{DFL}} = 0$ on identical initial weights and batch inputs.
6. **Ensure Genuine Integration for `dfl_dro` (`AGENTS.md` §7)**:
   - Confirm that `dfl_dro` draws its scenario marginals directly from the DFL-trained PMF (plus training-only tail policy and copula), not from a separate supervised model.

## Verification Procedures

To validate a serialized DFL batch/diagnostic artifact:
```bash
.venv/bin/python .agents/skills/decision-focused-learning/scripts/verify_dfl_contract.py --input path/to/dfl_batch_audit.json
```

## Expected Outputs

- JSON verification report checking PMF normalization, survival monotonicity, marginal-cost monotonicity, fixed-feasible-set invariance, SPO+ subgradient/loss computation, and NLL vs NLL+SPO+ gradient divergence.
- Explicit environment probe indicating whether `torch` and `pyepo` are installed for live neural training vs contract-only verification.

## Completion Criteria

- `verify_dfl_contract.py --self-test` exits `0`.
- Before full training, a one-batch backward smoke test confirms non-zero finite gradients from `SPOPlus` and distinct parameter updates relative to `lambda_DFL=0`.
- Oracle cache generation uses only `train` and `val` splits (never `test`).
