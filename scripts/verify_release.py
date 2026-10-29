"""Independently recompute release allocations, costs, hashes and paired tables."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from inventory_di.config import load_config
from inventory_di.evaluation.backtest import build_allocation_instance_from_config_df
from inventory_di.evaluation.metrics import EVALUATED_POLICIES, summarize_backtest_metrics
from inventory_di.io import sha256_file
from inventory_di.optimization.core import check_allocation_feasibility, evaluate_exact_cost
from inventory_di.provenance import validate_stage


def verify(mode):
    cfg = load_config(f"configs/{mode}.yaml")
    for stage in ["prepare", "train", "backtest"]:
        validate_stage(cfg, stage)
    art, proc = Path(cfg.data.artifacts_dir), Path(cfg.data.processed_dir)
    manifest = json.loads((art / "run_manifest.json").read_text())
    weekly = pd.read_parquet(proc / "weekly_demand.parquet")
    alloc = pd.read_parquet(art / "allocation_results.parquet")
    metrics = pd.read_parquet(art / "backtest_metrics.parquet")
    summary = pd.read_parquet(art / "summary_metrics.parquet")
    inst = build_allocation_instance_from_config_df(pd.read_parquet(proc / "inventory_config.parquet"))
    assert set(metrics.policy) == set(EVALUATED_POLICIES)
    assert not (metrics.solver_mip_gap > 1e-8).any()
    checked = 0
    for (origin, policy), rows in alloc.groupby(["forecast_origin", "policy"]):
        x = rows.pivot(index="store_id", columns="sku_id", values="allocated_units").loc[inst.stores, inst.skus].to_numpy()
        demand = weekly[weekly.forecast_origin == origin].pivot(index="store_id", columns="sku_id", values="observed_units_proxy").loc[inst.stores, inst.skus].to_numpy()
        assert check_allocation_feasibility(x, inst)[0]
        actual = evaluate_exact_cost(x, demand, inst)
        saved = metrics[(metrics.origin == origin) & (metrics.policy == policy)].iloc[0]
        for key in ["total_cost_mu", "shortage_cost_mu", "holding_cost_mu", "shipping_cost_mu", "filled_units", "unmet_units", "fill_rate"]:
            assert np.isclose(getattr(actual, key), saved[key]), (mode, origin, policy, key)
        oracle = metrics[(metrics.origin == origin) & (metrics.policy == "clairvoyant_oracle")].iloc[0]
        assert np.isclose(saved.oracle_regret_mu, actual.total_cost_mu-oracle.total_cost_mu)
        assert saved.oracle_regret_mu >= -1e-5
        checked += 1
    recomputed, _ = summarize_backtest_metrics(metrics, alpha=cfg.dro.alpha,
        block_size=cfg.evaluation.bootstrap_block_size, num_resamples=cfg.evaluation.bootstrap_resamples, seed=cfg.global_seed)
    pd.testing.assert_frame_equal(summary, recomputed)
    for key, digest in manifest["artifact_hashes"].items():
        assert sha256_file(art / (key.removesuffix("_parquet") + ".parquet")) == digest
    for checkpoint in manifest["model_checkpoints"].values():
        assert sha256_file(checkpoint["path"]) == checkpoint["sha256"]
    seeds = pd.read_parquet(art / "multi_seed_summary.parquet")
    assert set(seeds.seed) == set(cfg.evaluation.seeds)
    assert seeds.training_seed_independent.all()
    assert (seeds.epochs_budget == cfg.model.epochs).all()
    carry = pd.read_parquet(art / "carryover_leadtime_metrics.parquet")
    assert carry.is_conserved.all() and (carry.rolling_oracle_regret_mu >= -1e-5).all()
    diagnostics = json.loads((art / "training_diagnostics.json").read_text())
    assert all(np.isfinite(a["gradient_check"]["grad_norm_pure_spoplus"]) for a in diagnostics["dfl"].values())
    return dict(status="PASS", run_id=cfg.run_id, checked_policy_origins=checked,
                origins=metrics.origin.nunique(), seeds=cfg.evaluation.seeds,
                all_allocations_feasible=True, costs_recompute=True, summaries_recompute=True,
                hashes_match=True, carryover_conserved=True, min_regret_mu=float(metrics.oracle_regret_mu.min()),
                min_carryover_regret_mu=float(carry.rolling_oracle_regret_mu.min()))


if __name__ == "__main__":
    result = {mode: verify(mode) for mode in ["demo", "m5", "m5_intermittent"]}
    Path("reports/release/independent_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
