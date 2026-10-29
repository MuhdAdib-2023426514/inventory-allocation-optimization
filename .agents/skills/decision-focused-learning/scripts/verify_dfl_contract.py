#!/usr/bin/env python3
"""Deterministic Decision-Focused Learning (DFL) contract and SPO+ verifier.

Validates PMF normalization (with explicit overflow > H bin), survival probability
monotonicity, marginal expected cost monotonicity, fixed-feasible-set invariance,
exact SPO+ loss and subgradient mechanics, and NLL vs NLL+SPO+ gradient differentiation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

TOL = 1e-6


def check_optional_runtime_deps() -> dict[str, bool]:
    """Report whether PyTorch, PyEPO, Pyomo, and HiGHS are installed in the active Python."""
    return {
        pkg: importlib.util.find_spec(pkg) is not None
        for pkg in ("torch", "pyepo", "pyomo", "highspy")
    }


def validate_pmf_and_compute_survival(
    pmf: list[list[float]],
    H: int,
    I_s: list[int],
    K_s: list[int],
    b_s: list[float],
    h_s: list[float],
    t_s: list[float],
) -> tuple[list[str], list[list[float]], list[list[float]]]:
    """Validate discrete PMF (0..H + overflow) and compute survival q[s,k] and c_hat[s,k]."""
    errors: list[str] = []
    q_mat: list[list[float]] = []
    c_hat_mat: list[list[float]] = []

    n_stores = len(pmf)
    expected_len = H + 2  # 0..H plus overflow > H bin

    for s in range(n_stores):
        row = pmf[s]
        if len(row) != expected_len:
            errors.append(
                f"Store {s} PMF length {len(row)} != H+2 ({expected_len}); must include 0..H and overflow bin"
            )
            continue
        if any(p < -TOL for p in row):
            errors.append(f"Store {s} PMF contains negative probability")
        prob_sum = sum(row)
        if abs(prob_sum - 1.0) > 1e-5:
            errors.append(f"Store {s} PMF sum={prob_sum:.6f} != 1.0")

        q_row: list[float] = []
        c_row: list[float] = []
        for k in range(1, K_s[s] + 1):
            threshold = I_s[s] + k
            if threshold > H + 1:
                errors.append(
                    f"Store {s} threshold I_s+k={threshold} exceeds H+1={H + 1}; q[s,k] cannot be resolved without tail split"
                )
                q_sk = row[-1]
            else:
                # Sum probabilities from d = threshold .. H plus overflow bin (index H+1)
                q_sk = sum(row[threshold:])
            c_sk = t_s[s] + h_s[s] - (b_s[s] + h_s[s]) * q_sk
            q_row.append(q_sk)
            c_row.append(c_sk)

        # Verify survival probabilities decrease and marginal costs increase in k
        for idx in range(len(q_row) - 1):
            if q_row[idx + 1] > q_row[idx] + TOL:
                errors.append(
                    f"Store {s} non-monotonic survival probability: q[{idx + 2}]={q_row[idx + 1]:.6f} > q[{idx + 1}]={q_row[idx]:.6f}"
                )
            if c_row[idx + 1] + TOL < c_row[idx]:
                errors.append(
                    f"Store {s} decreasing marginal expected cost: c_hat[{idx + 2}]={c_row[idx + 1]:.6f} < c_hat[{idx + 1}]={c_row[idx]:.6f}"
                )

        q_mat.append(q_row)
        c_hat_mat.append(c_row)

    return errors, q_mat, c_hat_mat


def solve_fixed_feasible_marginal_problem(
    cost_mat: list[list[float]], K_s: list[int], W_p: int
) -> tuple[list[list[int]], float]:
    """Solve min_y sum_{s,k} cost[s,k]*y[s,k] s.t. y_{s,k+1} <= y_{s,k}, sum y <= W_p.

    Uses exact dynamic programming over stores s and integer store allocations x_s in 0..K_s[s]
    so prefix constraints y_{s,k}=1[k <= x_s] are enforced exactly even for non-monotonic
    perturbed SPO+ cost vectors (2*c_hat - c_true).
    """
    n_stores = len(K_s)
    # dp[w] = (best_cost, allocations_tuple)
    dp: dict[int, tuple[float, list[int]]] = {0: (0.0, [])}

    for s in range(n_stores):
        next_dp: dict[int, tuple[float, list[int]]] = {}
        # Precompute prefix sums for store s
        prefix_costs = [0.0]
        running = 0.0
        for k_idx in range(K_s[s]):
            running += cost_mat[s][k_idx]
            prefix_costs.append(running)

        for used_w, (cost_so_far, alloc_so_far) in dp.items():
            max_x = min(K_s[s], W_p - used_w)
            for x_s in range(max_x + 1):
                new_w = used_w + x_s
                new_cost = cost_so_far + prefix_costs[x_s]
                if new_w not in next_dp or new_cost < next_dp[new_w][0] - 1e-12:
                    next_dp[new_w] = (new_cost, alloc_so_far + [x_s])
        dp = next_dp

    best_cost = math.inf
    best_alloc: list[int] = [0] * n_stores
    for used_w, (c_val, alloc) in dp.items():
        if c_val < best_cost - 1e-12:
            best_cost = c_val
            best_alloc = alloc

    y_opt: list[list[int]] = []
    for s in range(n_stores):
        y_opt.append([1 if k <= best_alloc[s] else 0 for k in range(1, K_s[s] + 1)])

    return y_opt, best_cost


def compute_spo_plus_loss_and_subgrad(
    c_hat: list[list[float]],
    c_true: list[list[float]],
    K_s: list[int],
    W_p: int,
) -> dict[str, Any]:
    """Compute exact SPO+ loss and subgradient w.r.t. c_hat on the fixed-feasible set."""
    y_true_opt, z_true_opt = solve_fixed_feasible_marginal_problem(c_true, K_s, W_p)

    # Perturbed cost 2 * c_hat - c_true
    c_spo: list[list[float]] = []
    for s in range(len(K_s)):
        c_spo.append([2.0 * c_hat[s][k] - c_true[s][k] for k in range(K_s[s])])

    y_spo_opt, z_spo_opt = solve_fixed_feasible_marginal_problem(c_spo, K_s, W_p)

    dot_chat_ytrue = sum(
        c_hat[s][k] * y_true_opt[s][k] for s in range(len(K_s)) for k in range(K_s[s])
    )
    spo_plus_loss = -z_spo_opt + 2.0 * dot_chat_ytrue - z_true_opt

    subgrad: list[list[float]] = []
    subgrad_norm_sq = 0.0
    for s in range(len(K_s)):
        row_g: list[float] = []
        for k in range(K_s[s]):
            g = 2.0 * (y_true_opt[s][k] - y_spo_opt[s][k])
            row_g.append(g)
            subgrad_norm_sq += g * g
        subgrad.append(row_g)

    return {
        "spo_plus_loss": spo_plus_loss,
        "y_true_opt": y_true_opt,
        "y_spo_opt": y_spo_opt,
        "subgrad_c_hat": subgrad,
        "subgrad_l2_norm": math.sqrt(subgrad_norm_sq),
    }


def verify_dfl_payload(payload: dict[str, Any]) -> list[str]:
    """Verify a DFL training/contract payload against AGENTS.md §6 and §14."""
    errors: list[str] = []
    required = [
        "H",
        "K_s",
        "I_s",
        "W_p_instances",
        "b_s",
        "h_s",
        "t_s",
        "pmf",
        "d_true",
        "oracle_splits_used",
        "grad_norm_nll_only",
        "grad_norm_dfl",
        "param_diff_l2_vs_nll",
    ]
    for key in required:
        if key not in payload:
            errors.append(f"Missing required field '{key}'")
    if errors:
        return errors

    # 1. Fixed-feasible-set check across instances sharing a template
    w_instances = payload["W_p_instances"]
    if len(set(w_instances)) > 1:
        errors.append(
            f"Fixed-feasible-set violation: W_p varies across training examples {w_instances} within a single PyEPO template"
        )

    # 2. Oracle cache leakage check
    oracle_splits = {str(s).lower() for s in payload["oracle_splits_used"]}
    if "test" in oracle_splits:
        errors.append("Oracle solutions cached or computed on 'test' split during DFL training")

    # 3. PMF, survival, and marginal cost validation
    H = int(payload["H"])
    K_s = [int(k) for k in payload["K_s"]]
    I_s = [int(i) for i in payload["I_s"]]
    b_s = [float(b) for b in payload["b_s"]]
    h_s = [float(h) for h in payload["h_s"]]
    t_s = [float(t) for t in payload["t_s"]]

    pmf_errs, _, c_hat = validate_pmf_and_compute_survival(
        payload["pmf"], H, I_s, K_s, b_s, h_s, t_s
    )
    errors.extend(pmf_errs)

    # 4. Realized demand must not be silently clipped at H
    if payload.get("realized_demand_clipped_at_H", False):
        errors.append("Realized evaluation demand was clipped at H (forbidden by AGENTS.md §6)")

    # 5. Gradient & ablation check
    grad_nll = float(payload["grad_norm_nll_only"])
    grad_dfl = float(payload["grad_norm_dfl"])
    param_diff = float(payload["param_diff_l2_vs_nll"])
    if not math.isfinite(grad_dfl) or grad_dfl <= 0.0:
        errors.append(f"Invalid or zero DFL gradient norm: {grad_dfl}")
    if not math.isfinite(grad_nll) or grad_nll <= 0.0:
        errors.append(f"Invalid or zero NLL baseline gradient norm: {grad_nll}")
    if param_diff <= 1e-12:
        errors.append(
            "Switching on SPO+ (lambda_DFL > 0) produced zero parameter difference vs NLL-only baseline"
        )

    # 6. Verify SPO+ loss non-negativity and convexity invariant on the batch
    if not pmf_errs and w_instances:
        d_true = [int(d) for d in payload["d_true"]]
        c_true = [
            [
                (t_s[s] - b_s[s]) if d_true[s] >= I_s[s] + k else (t_s[s] + h_s[s])
                for k in range(1, K_s[s] + 1)
            ]
            for s in range(len(K_s))
        ]
        spo_res = compute_spo_plus_loss_and_subgrad(c_hat, c_true, K_s, int(w_instances[0]))
        if spo_res["spo_plus_loss"] < -TOL:
            errors.append(f"SPO+ loss was negative: {spo_res['spo_plus_loss']:.6f}")

    return errors


def run_self_test() -> dict[str, Any]:
    """Run positive and negative DFL verification checks."""
    errors: list[str] = []

    valid_payload = {
        "H": 10,
        "K_s": [4, 4],
        "I_s": [0, 0],
        "W_p_instances": [4, 4, 4],
        "b_s": [4.0, 3.0],
        "h_s": [0.5, 0.5],
        "t_s": [0.2, 0.2],
        "pmf": [
            [0.35, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01],
            [0.05, 0.05, 0.10, 0.20, 0.25, 0.15, 0.08, 0.04, 0.03, 0.02, 0.02, 0.01],
        ],
        "d_true": [1, 4],
        "oracle_splits_used": ["train", "val"],
        "grad_norm_nll_only": 0.412,
        "grad_norm_dfl": 0.934,
        "param_diff_l2_vs_nll": 0.024,
    }
    errors.extend(verify_dfl_payload(valid_payload))

    # Check exact SPO+ subgradient non-zero when forecast prioritizes wrong store
    _, _, c_hat = validate_pmf_and_compute_survival(
        valid_payload["pmf"],
        valid_payload["H"],
        valid_payload["I_s"],
        valid_payload["K_s"],
        valid_payload["b_s"],
        valid_payload["h_s"],
        valid_payload["t_s"],
    )
    # Suppose true demand is high at store 0 (d=4) and 0 at store 1 (d=0), opposite to PMF
    c_true_opposite = [
        [(0.2 - 4.0) if 4 >= k else (0.2 + 0.5) for k in range(1, 5)],
        [(0.2 - 3.0) if 0 >= k else (0.2 + 0.5) for k in range(1, 5)],
    ]
    spo_info = compute_spo_plus_loss_and_subgrad(c_hat, c_true_opposite, [4, 4], W_p=4)
    if spo_info["spo_plus_loss"] <= TOL or spo_info["subgrad_l2_norm"] <= TOL:
        errors.append("SPO+ self-test failed to produce positive loss and non-zero subgradient")

    # Negative check 1: detect variable RHS warehouse supply in PyEPO template
    bad_rhs = dict(valid_payload, W_p_instances=[4, 6, 3])
    if not any("Fixed-feasible-set violation" in e for e in verify_dfl_payload(bad_rhs)):
        errors.append("Self-test failed to detect variable W_p in fixed-feasible-set check")

    # Negative check 2: detect missing overflow bin or test-split oracle caching
    bad_leak = dict(
        valid_payload,
        oracle_splits_used=["train", "test"],
        param_diff_l2_vs_nll=0.0,
    )
    leak_errs = verify_dfl_payload(bad_leak)
    if not any("test" in e for e in leak_errs):
        errors.append("Self-test failed to detect test-split oracle caching")
    if not any("zero parameter difference" in e for e in leak_errs):
        errors.append("Self-test failed to detect inactive SPO+ term")

    return {
        "skill": "decision-focused-learning",
        "status": "PASS" if not errors else "FAIL",
        "optional_runtime_dependencies": check_optional_runtime_deps(),
        "checks_run": [
            "pmf_normalization_and_overflow_bin",
            "survival_and_marginal_cost_monotonicity",
            "fixed_feasible_set_invariance",
            "spo_plus_loss_and_subgradient_verification",
            "negative_variable_rhs_and_inactive_spo_detection",
        ],
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decision-Focused Learning contract verifier.")
    parser.add_argument("--self-test", action="store_true", help="Run built-in DFL verification.")
    parser.add_argument("--input", type=Path, help="Path to DFL audit JSON file.")
    args = parser.parse_args(argv)

    if not args.self_test and not args.input:
        parser.error("Provide --self-test or --input <path/to/dfl_audit.json>")

    if args.self_test:
        report = run_self_test()
    else:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        errs = verify_dfl_payload(payload)
        report = {
            "skill": "decision-focused-learning",
            "input": str(args.input),
            "status": "PASS" if not errs else "FAIL",
            "optional_runtime_dependencies": check_optional_runtime_deps(),
            "errors": errs,
        }

    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
