"""Aggregate metrics, origin-wise block bootstrap confidence intervals, and paired policy tables (`AGENTS.md` §8)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from inventory_di.optimization.dro import exact_ambiguity_set_worst_case_loss

REQUIRED_POLICIES = [
    "equal_allocation",
    "proportional_allocation",
    "supervised_pto",
    "supervised_dro",
    "dfl_pto",
    "dfl_dro",
    "clairvoyant_oracle",
]


def block_bootstrap_origin_ci(
    origin_values: np.ndarray,
    block_size: int = 2,
    num_resamples: int = 200,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute 95% confidence interval for the mean across time origins using moving-block bootstrap.
    Never resamples across correlated store-SKU rows (`AGENTS.md` §8).
    """
    vals = np.asarray(origin_values, dtype=float)
    T = len(vals)
    if T <= 1:
        m = float(vals[0]) if T == 1 else 0.0
        return m, m

    rng = np.random.default_rng(seed)
    b = max(1, min(int(block_size), T))
    num_blocks = int(np.ceil(T / float(b)))
    max_start = max(1, T - b + 1)

    means = np.zeros(num_resamples, dtype=float)
    for r in range(num_resamples):
        starts = rng.integers(0, max_start, size=num_blocks)
        sample_idx = np.concatenate([np.arange(s, s + b) for s in starts])[:T]
        means[r] = float(np.mean(vals[sample_idx]))

    low = float(np.percentile(means, 2.5))
    high = float(np.percentile(means, 97.5))
    return low, high


def summarize_backtest_metrics(
    metrics_df: pd.DataFrame,
    alpha: float = 0.90,
    block_size: int = 2,
    num_resamples: int = 200,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate per-origin `backtest_metrics.parquet` into a summary table with origin-wise block-bootstrap CIs
    and paired differences against `equal_allocation` and `supervised_pto`.
    """
    origins = sorted(metrics_df["origin"].unique())
    num_origins = len(origins)
    small_sample_caveat = (
        f"Computed via moving-block bootstrap (block_size={block_size}, B={num_resamples}) across "
        f"{num_origins} held-out forecast origins. Because {num_origins} < 20 origins is a small time-series "
        "sample, intervals are descriptive uncertainty bands and should not be over-interpreted as asymptotic p-values."
        if num_origins < 20
        else f"Computed via moving-block bootstrap (block_size={block_size}, B={num_resamples}) across {num_origins} origins."
    )

    eq_by_origin = (
        metrics_df[metrics_df["policy"] == "equal_allocation"]
        .set_index("origin")["total_cost_mu"]
        .to_dict()
    )
    sup_by_origin = (
        metrics_df[metrics_df["policy"] == "supervised_pto"]
        .set_index("origin")["total_cost_mu"]
        .to_dict()
    )

    rows: list[dict[str, Any]] = []
    for policy in REQUIRED_POLICIES:
        sub = metrics_df[metrics_df["policy"] == policy].sort_values("origin")
        if sub.empty:
            continue
        costs = sub["total_cost_mu"].to_numpy(dtype=float)
        regrets = sub["oracle_regret_mu"].to_numpy(dtype=float)
        fills = sub["fill_rate"].to_numpy(dtype=float)
        shortages = sub["shortage_cost_mu"].to_numpy(dtype=float)
        holdings = sub["holding_cost_mu"].to_numpy(dtype=float)
        shippings = sub["shipping_cost_mu"].to_numpy(dtype=float)
        unmets = sub["unmet_units"].to_numpy(dtype=float)
        runtimes = sub["runtime_seconds"].to_numpy(dtype=float)

        ci_low, ci_high = block_bootstrap_origin_ci(
            costs, block_size=block_size, num_resamples=num_resamples, seed=seed
        )
        reg_low, reg_high = block_bootstrap_origin_ci(
            regrets, block_size=block_size, num_resamples=num_resamples, seed=seed + 1
        )

        # Realized worst-tail cost across origins (CVaR_alpha of realized origin costs)
        probs = np.full(len(costs), 1.0 / len(costs))
        realized_cvar, _ = exact_ambiguity_set_worst_case_loss(costs, probs, alpha=alpha)
        realized_p90 = float(np.percentile(costs, 90))

        mean_cost = float(np.mean(costs))
        eq_costs = np.array([eq_by_origin.get(o, mean_cost) for o in sub["origin"]], dtype=float)
        sup_costs = np.array([sup_by_origin.get(o, mean_cost) for o in sub["origin"]], dtype=float)

        diff_vs_eq = float(np.mean(costs - eq_costs))
        pct_vs_eq = float(100.0 * diff_vs_eq / max(1e-6, float(np.mean(eq_costs))))
        diff_vs_sup = float(np.mean(costs - sup_costs))
        pct_vs_sup = float(100.0 * diff_vs_sup / max(1e-6, float(np.mean(sup_costs))))

        rows.append(
            {
                "policy": policy,
                "num_origins": len(costs),
                "mean_cost_mu": mean_cost,
                "std_cost_mu": float(np.std(costs, ddof=0)),
                "median_cost_mu": float(np.median(costs)),
                "cost_ci95_low_mu": ci_low,
                "cost_ci95_high_mu": ci_high,
                "realized_p90_cost_mu": realized_p90,
                "realized_cvar90_cost_mu": realized_cvar,
                "mean_regret_mu": float(np.mean(regrets)),
                "regret_ci95_low_mu": reg_low,
                "regret_ci95_high_mu": reg_high,
                "mean_fill_rate": float(np.mean(fills)),
                "mean_unmet_units": float(np.mean(unmets)),
                "mean_shortage_cost_mu": float(np.mean(shortages)),
                "mean_holding_cost_mu": float(np.mean(holdings)),
                "mean_shipping_cost_mu": float(np.mean(shippings)),
                "model_expected_cost_mean_mu": float(sub["model_expected_cost_mu"].mean()),
                "model_cvar_cost_mean_mu": float(sub["model_cvar_cost_mu"].mean()),
                "diff_vs_equal_mu": diff_vs_eq,
                "pct_vs_equal": pct_vs_eq,
                "diff_vs_supervised_pto_mu": diff_vs_sup,
                "pct_vs_supervised_pto": pct_vs_sup,
                "mean_runtime_seconds": float(np.mean(runtimes)),
                "all_optimal": bool((sub["solver_status"] == "optimal").all()),
                "data_mode": str(sub["data_mode"].iloc[0]),
            }
        )

    summary_df = pd.DataFrame(rows)
    meta = {
        "bootstrap_unit": "forecast_origin",
        "bootstrap_block_size": block_size,
        "bootstrap_resamples": num_resamples,
        "num_test_origins": num_origins,
        "small_sample_caveat": small_sample_caveat,
    }
    return summary_df, meta
