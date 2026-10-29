#!/usr/bin/env python3
"""Independent mathematical oracle and optimization verifier for inventory allocation.

Enforces AGENTS.md Sections 5, 6, 7, 8, and 14 using pure Python standard library
calculations so that solver outputs are checked by an independent implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

TOL = 1e-6
VALID_SOLVER_STATUSES = {"optimal", "ok", "optimal_solution"}


def compute_exact_cost_and_metrics(
    x: list[list[int]],
    D: list[list[int]],
    I: list[list[int]],
    b: list[list[float]],
    h: list[list[float]],
    t: list[list[float]],
) -> dict[str, float]:
    """Recompute exact realized cost, shortage, holding, shipping, and fill rate (AGENTS.md §5)."""
    n_stores = len(x)
    n_skus = len(x[0]) if n_stores > 0 else 0

    shortage_cost = 0.0
    holding_cost = 0.0
    shipping_cost = 0.0
    filled_units = 0
    unmet_units = 0
    total_demand = 0

    for s in range(n_stores):
        for p in range(n_skus):
            alloc = int(x[s][p])
            dem = int(D[s][p])
            init_stock = int(I[s][p])
            avail = init_stock + alloc

            shortage = max(dem - avail, 0)
            excess = max(avail - dem, 0)
            filled = min(dem, avail)

            shortage_cost += float(b[s][p]) * shortage
            holding_cost += float(h[s][p]) * excess
            shipping_cost += float(t[s][p]) * alloc

            filled_units += filled
            unmet_units += shortage
            total_demand += dem

    total_cost = shortage_cost + holding_cost + shipping_cost
    fill_rate = (filled_units / total_demand) if total_demand > 0 else 1.0

    return {
        "total_cost_mu": total_cost,
        "shortage_cost_mu": shortage_cost,
        "holding_cost_mu": holding_cost,
        "shipping_cost_mu": shipping_cost,
        "filled_units": float(filled_units),
        "unmet_units": float(unmet_units),
        "fill_rate": fill_rate,
    }


def check_feasibility(
    x: list[list[int]],
    I: list[list[int]],
    W: list[int],
    K: list[list[int]],
    C: list[int],
) -> list[str]:
    """Validate integer non-negativity, warehouse supply, store-SKU cap, and store total capacity."""
    errors: list[str] = []
    n_stores = len(x)
    n_skus = len(x[0]) if n_stores > 0 else 0

    for s in range(n_stores):
        store_stock_sum = 0
        for p in range(n_skus):
            val = x[s][p]
            if not isinstance(val, int) or isinstance(val, bool):
                errors.append(f"Non-integer allocation x[{s}][{p}]={val!r}")
                continue
            if val < 0:
                errors.append(f"Negative allocation x[{s}][{p}]={val}")
            if val > K[s][p]:
                errors.append(f"Store-SKU cap violated at ({s},{p}): x={val} > K={K[s][p]}")
            store_stock_sum += I[s][p] + val
        if store_stock_sum > C[s]:
            errors.append(
                f"Store total capacity violated at store {s}: sum_p(I+x)={store_stock_sum} > C={C[s]}"
            )

    for p in range(n_skus):
        shipped_p = sum(int(x[s][p]) for s in range(n_stores))
        if shipped_p > W[p]:
            errors.append(f"Warehouse supply violated for SKU {p}: sum_s x={shipped_p} > W={W[p]}")

    return errors


def compute_marginal_unit_cost(
    x_s: list[int],
    d_s: list[int],
    I_s: list[int],
    K_s: list[int],
    b_s: list[float],
    h_s: list[float],
    t_s: list[float],
) -> tuple[float, list[list[int]], list[list[float]]]:
    """Compute C(x, d) via marginal-unit binary encoding C(0, d) + sum_{s,k} c_true[s,k;d]*y[s,k]."""
    n_stores = len(x_s)
    zero_alloc = [[0] for _ in range(n_stores)]
    D_col = [[d_s[s]] for s in range(n_stores)]
    I_col = [[I_s[s]] for s in range(n_stores)]
    b_col = [[b_s[s]] for s in range(n_stores)]
    h_col = [[h_s[s]] for s in range(n_stores)]
    t_col = [[t_s[s]] for s in range(n_stores)]

    base_cost = compute_exact_cost_and_metrics(zero_alloc, D_col, I_col, b_col, h_col, t_col)[
        "total_cost_mu"
    ]
    y_mat: list[list[int]] = []
    c_true_mat: list[list[float]] = []
    incr_sum = 0.0

    for s in range(n_stores):
        y_row: list[int] = []
        c_row: list[float] = []
        for k in range(1, K_s[s] + 1):
            y_sk = 1 if x_s[s] >= k else 0
            c_sk = (t_s[s] - b_s[s]) if d_s[s] >= I_s[s] + k else (t_s[s] + h_s[s])
            y_row.append(y_sk)
            c_row.append(c_sk)
            incr_sum += c_sk * y_sk
        # Verify prefix ordering y_{s,k+1} <= y_{s,k}
        for k_idx in range(len(y_row) - 1):
            if y_row[k_idx + 1] > y_row[k_idx]:
                raise ValueError(f"Prefix violation at store {s}, unit {k_idx + 1}")
        y_mat.append(y_row)
        c_true_mat.append(c_row)

    return base_cost + incr_sum, y_mat, c_true_mat


def cvar_reweighting_dual(losses: list[float], probs: list[float], alpha: float) -> float:
    """Exact maximization over P_alpha = {w: sum w_j = 1, 0 <= w_j <= p_j/(1-alpha)}."""
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    if len(losses) != len(probs) or not losses:
        raise ValueError("losses and probs must be non-empty and equal length")
    if abs(sum(probs) - 1.0) > 1e-6 or any(p < 0 for p in probs):
        raise ValueError("probs must be non-negative and sum to 1")

    if alpha == 0.0:
        return sum(p * l for p, l in zip(probs, losses))

    # Sort scenarios in descending order of loss
    paired = sorted(zip(losses, probs), key=lambda item: item[0], reverse=True)
    remaining_mass = 1.0
    total_weighted_loss = 0.0
    scale = 1.0 / (1.0 - alpha)

    for loss_val, p_val in paired:
        if remaining_mass <= 0.0:
            break
        w_j = min(p_val * scale, remaining_mass)
        total_weighted_loss += w_j * loss_val
        remaining_mass -= w_j

    return total_weighted_loss


def cvar_rockafellar_uryasev_lp(losses: list[float], probs: list[float], alpha: float) -> float:
    """Evaluate min_eta { eta + 1/(1-alpha) * sum_j p_j max(L_j - eta, 0) }.

    Since the piecewise-linear convex function in eta attains its minimum at one of the
    kinks eta in losses, enumerating all kinks is exact for finite scenarios.
    """
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    scale = 1.0 / (1.0 - alpha)
    candidate_etas = set(losses)
    best = math.inf
    for eta in candidate_etas:
        obj = eta + scale * sum(p * max(l - eta, 0.0) for l, p in zip(losses, probs))
        if obj < best:
            best = obj
    return best


def risk_tradeoff_objective(
    losses: list[float], probs: list[float], alpha: float, risk_weight: float
) -> float:
    """Compute (1 - risk_weight) * E_p[L] + risk_weight * CVaR_alpha(L)."""
    if not (0.0 <= risk_weight <= 1.0):
        raise ValueError(f"risk_weight must be in [0, 1], got {risk_weight}")
    expected_loss = sum(p * l for p, l in zip(probs, losses))
    cvar_val = cvar_rockafellar_uryasev_lp(losses, probs, alpha)
    return (1.0 - risk_weight) * expected_loss + risk_weight * cvar_val


def verify_toy_oracle() -> list[str]:
    """Verify the mandatory 2-store toy mathematical oracle from AGENTS.md §5."""
    errors: list[str] = []
    D = [[60], [20]]
    I = [[0], [0]]
    W = [60]
    K = [[60], [60]]
    C = [60, 60]
    b = [[1.0], [1.0]]
    h = [[0.0], [0.0]]
    t = [[0.0], [0.0]]

    # Equal allocation [30, 30]
    x_equal = [[30], [30]]
    errors.extend(check_feasibility(x_equal, I, W, K, C))
    m_equal = compute_exact_cost_and_metrics(x_equal, D, I, b, h, t)
    if abs(m_equal["total_cost_mu"] - 30.0) > TOL:
        errors.append(f"Toy equal cost expected 30.0, got {m_equal['total_cost_mu']}")
    if abs(m_equal["filled_units"] - 50.0) > TOL:
        errors.append(f"Toy equal filled_units expected 50, got {m_equal['filled_units']}")
    if abs(m_equal["fill_rate"] - 0.625) > TOL:
        errors.append(f"Toy equal fill_rate expected 0.625, got {m_equal['fill_rate']}")

    # Optimal allocation [40, 20]
    x_opt = [[40], [20]]
    errors.extend(check_feasibility(x_opt, I, W, K, C))
    m_opt = compute_exact_cost_and_metrics(x_opt, D, I, b, h, t)
    if abs(m_opt["total_cost_mu"] - 20.0) > TOL:
        errors.append(f"Toy optimal cost expected 20.0, got {m_opt['total_cost_mu']}")
    if abs(m_opt["filled_units"] - 60.0) > TOL:
        errors.append(f"Toy optimal filled_units expected 60, got {m_opt['filled_units']}")
    if abs(m_opt["fill_rate"] - 0.75) > TOL:
        errors.append(f"Toy optimal fill_rate expected 0.75, got {m_opt['fill_rate']}")

    # Brute-force check over all feasible integer splits (x1, x2) with x1+x2 <= 60
    best_cost = math.inf
    for x1 in range(61):
        for x2 in range(61 - x1):
            c_val = compute_exact_cost_and_metrics([[x1], [x2]], D, I, b, h, t)["total_cost_mu"]
            if c_val < best_cost:
                best_cost = c_val
    if abs(best_cost - m_opt["total_cost_mu"]) > TOL:
        errors.append(f"Brute-force minimum cost {best_cost} != [40,20] cost {m_opt['total_cost_mu']}")

    return errors


def verify_marginal_cost_identity_randomized(seed: int = 42, trials: int = 25) -> list[str]:
    """Prove C(x,d) = C(0,d) + sum_{s,k} c_true[s,k;d]*y[s,k] across random instances with I_s >= 0."""
    rng = random.Random(seed)
    errors: list[str] = []
    for trial in range(trials):
        n_stores = rng.randint(2, 4)
        K_s = [rng.randint(5, 12) for _ in range(n_stores)]
        I_s = [rng.randint(0, 6) for _ in range(n_stores)]
        d_s = [rng.randint(0, 18) for _ in range(n_stores)]
        x_s = [rng.randint(0, K_s[s]) for s in range(n_stores)]
        b_s = [round(rng.uniform(1.0, 8.0), 3) for _ in range(n_stores)]
        h_s = [round(rng.uniform(0.1, 2.5), 3) for _ in range(n_stores)]
        t_s = [round(rng.uniform(0.0, 1.5), 3) for _ in range(n_stores)]

        direct_cost = compute_exact_cost_and_metrics(
            [[x_s[s]] for s in range(n_stores)],
            [[d_s[s]] for s in range(n_stores)],
            [[I_s[s]] for s in range(n_stores)],
            [[b_s[s]] for s in range(n_stores)],
            [[h_s[s]] for s in range(n_stores)],
            [[t_s[s]] for s in range(n_stores)],
        )["total_cost_mu"]

        decomp_cost, _, _ = compute_marginal_unit_cost(x_s, d_s, I_s, K_s, b_s, h_s, t_s)
        if abs(direct_cost - decomp_cost) > TOL:
            errors.append(
                f"Trial {trial}: direct cost {direct_cost:.6f} != marginal identity cost {decomp_cost:.6f}"
            )
    return errors


def verify_cvar_and_limiting_cases() -> list[str]:
    """Verify CVaR LP == reweighting dual and risk-neutral limiting cases (AGENTS.md §7, §14)."""
    errors: list[str] = []
    rng = random.Random(101)
    for trial in range(15):
        n_scen = rng.randint(3, 12)
        losses = [round(rng.uniform(10.0, 250.0), 4) for _ in range(n_scen)]
        raw_w = [rng.uniform(0.1, 2.0) for _ in range(n_scen)]
        total_w = sum(raw_w)
        probs = [w / total_w for w in raw_w]

        expected_loss = sum(p * l for p, l in zip(probs, losses))

        # Limiting case 1: alpha = 0, risk_weight = 1 -> expected loss
        cvar_0_dual = cvar_reweighting_dual(losses, probs, alpha=0.0)
        cvar_0_lp = cvar_rockafellar_uryasev_lp(losses, probs, alpha=0.0)
        if abs(cvar_0_dual - expected_loss) > TOL or abs(cvar_0_lp - expected_loss) > TOL:
            errors.append(f"Trial {trial}: alpha=0 failed to recover expected loss")

        # Limiting case 2: risk_weight = 0 for arbitrary alpha -> expected loss
        rw_0 = risk_tradeoff_objective(losses, probs, alpha=0.90, risk_weight=0.0)
        if abs(rw_0 - expected_loss) > TOL:
            errors.append(f"Trial {trial}: risk_weight=0 failed to recover expected loss")

        # Equivalence across alphas
        for alpha in (0.50, 0.75, 0.90, 0.95):
            dual_val = cvar_reweighting_dual(losses, probs, alpha=alpha)
            lp_val = cvar_rockafellar_uryasev_lp(losses, probs, alpha=alpha)
            if abs(dual_val - lp_val) > TOL:
                errors.append(
                    f"Trial {trial} alpha={alpha}: reweighting dual {dual_val} != LP {lp_val}"
                )
            if dual_val + TOL < expected_loss:
                errors.append(
                    f"Trial {trial} alpha={alpha}: CVaR {dual_val} < expected loss {expected_loss}"
                )

    # Brute-force integer allocation check on a tiny 2-store, 3-scenario DRO instance
    D_scen = [[[10], [5]], [[20], [15]], [[35], [25]]]
    probs = [0.5, 0.3, 0.2]
    I = [[0], [0]]
    b = [[4.0], [3.0]]
    h = [[0.5], [0.5]]
    t = [[0.2], [0.2]]
    best_dro_obj = math.inf
    for x1 in range(26):
        for x2 in range(26 - x1):
            x_cand = [[x1], [x2]]
            scen_losses = [
                compute_exact_cost_and_metrics(x_cand, D_j, I, b, h, t)["total_cost_mu"]
                for D_j in D_scen
            ]
            dual_obj = cvar_reweighting_dual(scen_losses, probs, alpha=0.80)
            lp_obj = cvar_rockafellar_uryasev_lp(scen_losses, probs, alpha=0.80)
            if abs(dual_obj - lp_obj) > TOL:
                errors.append(f"2-store DRO candidate {x_cand}: dual {dual_obj} != lp {lp_obj}")
            best_dro_obj = min(best_dro_obj, lp_obj)
    if not math.isfinite(best_dro_obj):
        errors.append("Failed to enumerate tiny 2-store DRO instance")

    return errors


def verify_candidate_payload(payload: dict[str, Any]) -> list[str]:
    """Validate an optimization solution payload against all AGENTS.md mathematical gates."""
    errors: list[str] = []
    required = ["D", "I", "W", "K", "C", "b", "h", "t", "x", "reported_cost", "solver_status"]
    for key in required:
        if key not in payload:
            errors.append(f"Missing required field '{key}'")
    if errors:
        return errors

    status = str(payload["solver_status"]).strip().lower()
    if status not in VALID_SOLVER_STATUSES:
        errors.append(f"Invalid or non-optimal solver_status: {payload['solver_status']!r}")

    errors.extend(
        check_feasibility(payload["x"], payload["I"], payload["W"], payload["K"], payload["C"])
    )
    metrics = compute_exact_cost_and_metrics(
        payload["x"], payload["D"], payload["I"], payload["b"], payload["h"], payload["t"]
    )
    reported_cost = float(payload["reported_cost"])
    if abs(metrics["total_cost_mu"] - reported_cost) > TOL:
        errors.append(
            f"Reported cost {reported_cost:.6f} != recomputed exact cost {metrics['total_cost_mu']:.6f}"
        )

    if "oracle_cost" in payload and payload["oracle_cost"] is not None:
        oracle_cost = float(payload["oracle_cost"])
        regret = metrics["total_cost_mu"] - oracle_cost
        if regret < -TOL:
            errors.append(
                f"Negative hindsight regret detected: policy_cost={metrics['total_cost_mu']:.6f} "
                f"< oracle_cost={oracle_cost:.6f} (regret={regret:.6f})"
            )

    return errors


def run_self_test() -> dict[str, Any]:
    """Run all built-in positive and negative verification tests."""
    errors: list[str] = []
    errors.extend(verify_toy_oracle())
    errors.extend(verify_marginal_cost_identity_randomized())
    errors.extend(verify_cvar_and_limiting_cases())

    # Negative check 1: detect warehouse overshipment
    neg_errors_w = check_feasibility(
        x=[[40], [30]], I=[[0], [0]], W=[60], K=[[60], [60]], C=[100, 100]
    )
    if not any("Warehouse supply violated" in e for e in neg_errors_w):
        errors.append("Self-test failed to detect warehouse overshipment violation")

    # Negative check 2: detect negative oracle regret & bad solver status
    bad_payload = {
        "D": [[60], [20]],
        "I": [[0], [0]],
        "W": [60],
        "K": [[60], [60]],
        "C": [60, 60],
        "b": [[1.0], [1.0]],
        "h": [[0.0], [0.0]],
        "t": [[0.0], [0.0]],
        "x": [[40], [20]],
        "reported_cost": 20.0,
        "oracle_cost": 25.0,
        "solver_status": "infeasible",
    }
    neg_payload_errors = verify_candidate_payload(bad_payload)
    if not any("Negative hindsight regret" in e for e in neg_payload_errors):
        errors.append("Self-test failed to detect negative hindsight regret")
    if not any("solver_status" in e for e in neg_payload_errors):
        errors.append("Self-test failed to detect non-optimal solver_status")

    return {
        "skill": "optimization-verification",
        "status": "PASS" if not errors else "FAIL",
        "checks_run": [
            "toy_oracle_2_stores",
            "marginal_unit_cost_identity_25_random_instances",
            "cvar_reweighting_lp_equivalence_and_limiting_cases",
            "negative_warehouse_overshipment_detection",
            "negative_regret_and_solver_status_detection",
        ],
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optimization verification oracle.")
    parser.add_argument("--self-test", action="store_true", help="Run built-in verification suite.")
    parser.add_argument("--input", type=Path, help="Path to candidate JSON payload to verify.")
    args = parser.parse_args(argv)

    if not args.self_test and not args.input:
        parser.error("Provide --self-test or --input <path/to/payload.json>")

    if args.self_test:
        report = run_self_test()
    else:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        errs = verify_candidate_payload(payload)
        report = {
            "skill": "optimization-verification",
            "input": str(args.input),
            "status": "PASS" if not errs else "FAIL",
            "errors": errs,
        }

    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
