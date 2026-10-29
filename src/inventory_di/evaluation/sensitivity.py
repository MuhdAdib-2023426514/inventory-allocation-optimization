"""Predeclared frozen-model sensitivities, never used to select a test policy."""

from dataclasses import replace

import numpy as np
import pandas as pd

from inventory_di.models.distributions import EmpiricalCopulaScenarioGenerator
from inventory_di.models.supervised import predict_sku_origin_outputs
from inventory_di.optimization.core import evaluate_exact_cost
from inventory_di.optimization.dro import solve_scenario_reweighting_dro_milp


def distribution_sensitivity(cfg, weekly, inst, artifacts, selected):
    origin = sorted(weekly.loc[weekly.split == "test", "forecast_origin"].unique())[0]
    frame = weekly[weekly.forecast_origin == origin]
    train = weekly[weekly.split == "train"]
    actual = frame.pivot(index="store_id", columns="sku_id", values="observed_units_proxy").loc[inst.stores, inst.skus].to_numpy()
    pmfs = {sku: predict_sku_origin_outputs(artifacts[sku], frame, inst.I[:,p], inst.K[:,p],
             inst.b[:,p], inst.h[:,p], inst.t[:,p])["pmf"] for p,sku in enumerate(inst.skus)}
    first = next(iter(artifacts.values()))
    # Each row changes only its named factor, all on the first test origin.
    cases = [("sample_count", n, first.support_h, first.tail_mean_excess, 1.) for n in [25, 50, 100]]
    cases += [("tail_mean", cfg.dro.num_scenarios, first.support_h, tail, 1.) for tail in [1., 2., 4.]]
    cases += [("shortage_ratio", cfg.dro.num_scenarios, first.support_h, first.tail_mean_excess, ratio) for ratio in [.5, 1., 2.]]
    reduced_h = max(2, first.support_h-4)
    excess = train.loc[train.observed_units_proxy > reduced_h, "observed_units_proxy"] - reduced_h
    cases.append(("support_rebin_frozen_pmf", cfg.dro.num_scenarios, reduced_h, float(excess.mean()) if len(excess) else 2., 1.))
    rows = []
    for factor, count, support, tail, penalty in cases:
        local_pmfs = {}
        for sku, pmf in pmfs.items():
            local_pmfs[sku] = np.concatenate([pmf[:,:support+1], pmf[:,support+1:].sum(axis=1, keepdims=True)], axis=1)
        gen = EmpiricalCopulaScenarioGenerator(inst.stores, inst.skus, support, tail)
        gen.fit_on_train(train)
        local = replace(inst, b=inst.b*penalty)
        for seed in [42,101,202]:
            scenarios = gen.sample_scenarios(local_pmfs, count, seed)
            result = solve_scenario_reweighting_dro_milp(scenarios, local, **selected)
            actual_cost = evaluate_exact_cost(result.x, actual, local)
            rows.append(dict(factor=factor, origin=origin, num_scenarios=count, support_h=support,
                tail_mean_excess=tail, shortage_multiplier=penalty, scenario_seed=seed,
                realized_cost_mu=actual_cost.total_cost_mu, fill_rate=actual_cost.fill_rate,
                model_expected_cost_mu=result.model_expected_cost_mu, model_cvar_cost_mu=result.model_cvar_cost_mu,
                solver_status=result.solver_status, data_mode=cfg.data_mode,
                scope="First-origin frozen-model descriptive sensitivity; no policy selection"))
    return pd.DataFrame(rows)
