"""Independent mathematical and causality checks for the corrected release."""

from functools import lru_cache
from itertools import product

import numpy as np
import pandas as pd
import pytest
import torch

from inventory_di.config import load_config
from inventory_di.data.features import compute_weekly_features_strictly_pre_origin
from inventory_di.models.distributions import (
    EmpiricalCopulaScenarioGenerator,
    compute_survival_and_marginal_costs_torch,
    inverse_demand_cdf,
    pmf_moments_and_quantiles,
)
from inventory_di.optimization.core import (
    AllocationInstance,
    solve_deterministic_milp,
    solve_multiperiod_carryover_oracle,
    solve_service_level_constrained_milp,
)
from inventory_di.simulation.simulator import InventoryStateTransitionSystem


def enumerate_horizon(inst, demand, lead, pipeline):
    """Independent exhaustive DP over dispatch/admission actions; no project simulator."""
    T, S, P = demand.shape
    N = S * P
    flat_demand = demand.reshape(T, N)
    b, h, t = inst.b.ravel(), inst.h.ravel(), inst.t.ravel()

    @lru_cache(None)
    def value(week, stock, deferred, pipe):
        if week == T:
            return 0.0
        best = float("inf")
        for order in product(*[range(int(k) + 1) for k in inst.K.ravel()]):
            if any(sum(order[si * P + p] for si in range(S)) > inst.W[p] for p in range(P)) or (
                week + lead >= T and any(order)
            ):
                continue
            due = order if lead == 0 else pipe[0]
            queued = tuple(deferred[p] + due[p] for p in range(N))
            for admitted in product(*[range(q + 1) for q in queued]):
                if any(
                    sum(stock[si * P : (si + 1) * P]) + sum(admitted[si * P : (si + 1) * P])
                    > inst.C[si]
                    for si in range(S)
                ):
                    continue
                filled = tuple(min(flat_demand[week, p], stock[p] + admitted[p]) for p in range(N))
                closing = tuple(stock[p] + admitted[p] - filled[p] for p in range(N))
                waiting = tuple(queued[p] - admitted[p] for p in range(N))
                cost = sum(
                    b[p] * (flat_demand[week, p] - filled[p])
                    + h[p] * (closing[p] + waiting[p])
                    + t[p] * order[p]
                    for p in range(N)
                )
                next_pipe = pipe[1:] + (order,) if lead else ()
                best = min(best, cost + value(week + 1, closing, waiting, next_pipe))
        return best

    return value(0, tuple(inst.I.ravel()), (0,) * N, tuple(tuple(x.ravel()) for x in pipeline))


@pytest.mark.parametrize("lead", [0, 1])
@pytest.mark.parametrize("products", [1, 2])
@pytest.mark.parametrize("stores", [1, 2])
def test_horizon_oracle_matches_independent_enumeration(lead, products, stores):
    rng = np.random.default_rng(72 + products)
    for _ in range(4):
        shape = (stores, products)
        inst = AllocationInstance(
            [f"s{si}" for si in range(stores)],
            [f"p{p}" for p in range(products)],
            rng.integers(0, 2, size=shape),
            np.ones(products, dtype=int),
            np.ones(shape, dtype=int),
            np.full(stores, 2),
            np.full(shape, 6.0),
            np.full(shape, 0.2),
            np.full(shape, 0.1),
        )
        d = rng.integers(0, 3, size=(2 if stores * products > 2 else 3, stores, products))
        pipe = [np.ones(shape, dtype=int) for _ in range(lead)]
        result = solve_multiperiod_carryover_oracle(d, inst, lead, pipe)
        assert result["total_cost_mu"] == pytest.approx(enumerate_horizon(inst, d, lead, pipe))
        assert result["mip_gap"] == pytest.approx(0)


def test_tail_survival_moments_quantiles_and_sampling_agree():
    pmf = np.array([0.1, 0.2, 0.3, 0.4])  # H=2, 40% geometric excess with mean four
    uniforms = (np.arange(100000) + 0.5) / 100000
    draws = inverse_demand_cdf(pmf, uniforms, 4)
    moments = pmf_moments_and_quantiles(pmf[None], 4)
    assert draws.mean() == pytest.approx(moments["expected"][0], abs=0.002)
    assert moments["p90"][0] == np.quantile(draws, 0.9, method="inverted_cdf")
    q, _ = compute_survival_and_marginal_costs_torch(
        torch.tensor(pmf.reshape(1, 1, -1)), [0], [12], [6], [1], [0], 4
    )
    for threshold in range(1, 13):
        assert q[0, threshold - 1].item() == pytest.approx((draws >= threshold).mean(), abs=2e-5)


def test_zero_ties_do_not_create_chronological_dependence():
    rows = [
        dict(forecast_origin=str(t), store_id=s, sku_id="p", observed_units_proxy=0, split="train")
        for t in range(600)
        for s in ["a", "b"]
    ]
    gen = EmpiricalCopulaScenarioGenerator(["a", "b"], ["p"], 3)
    gen.fit_on_train(pd.DataFrame(rows))
    assert (
        abs(np.corrcoef(gen.rank_matrix_train[:, 0, 0], gen.rank_matrix_train[:, 1, 0])[0, 1])
        < 0.15
    )


def test_hard_service_does_not_trade_away_feasible_target():
    inst = AllocationInstance(
        ["s"],
        ["p"],
        np.array([[0]]),
        np.array([1]),
        np.array([[1]]),
        np.array([1]),
        np.array([[0.0]]),
        np.array([[0.0]]),
        np.array([[100.0]]),
    )
    scenarios = np.array([[[1]]])
    hard = solve_service_level_constrained_milp(scenarios, inst, 0.9, 50, mode="hard")
    soft = solve_service_level_constrained_milp(scenarios, inst, 0.9, 50, mode="soft")
    assert hard.x[0, 0] == 1 and hard.is_service_level_met
    assert soft.x[0, 0] == 0 and not soft.is_service_level_met
    # Demand averages below one unit: ratio denominator must not be rounded to one.
    rare = np.zeros((10, 1, 1))
    rare[0, 0, 0] = 1
    assert solve_service_level_constrained_milp(rare, inst, 0.9).expected_fill_rate == 1


@pytest.mark.parametrize("supply", [0, 2, 5])
def test_hard_service_matches_brute_force_cost_at_required_or_maximum_fill(supply):
    from inventory_di.optimization.core import check_allocation_feasibility, evaluate_exact_cost

    inst = AllocationInstance(
        ["a", "b"],
        ["p"],
        np.array([[1], [0]]),
        np.array([supply]),
        np.array([[3], [3]]),
        np.array([3, 3]),
        np.array([[6.0], [4.0]]),
        np.array([[0.5], [1.0]]),
        np.array([[0.2], [2.0]]),
    )
    scenarios = np.array([[[1], [3]], [[4], [1]], [[0], [0]]])
    choices = []
    for a, b in product(range(4), repeat=2):
        x = np.array([[a], [b]])
        if not check_allocation_feasibility(x, inst)[0]:
            continue
        costs = [evaluate_exact_cost(x, d, inst) for d in scenarios]
        fill = sum(c.filled_units for c in costs) / scenarios.sum()
        choices.append((fill, np.mean([c.total_cost_mu for c in costs])))
    target = 0.85
    maximum = max(fill for fill, _ in choices)
    required = min(maximum, target)
    expected = min(cost for fill, cost in choices if fill >= required - 1e-8)
    result = solve_service_level_constrained_milp(scenarios, inst, target)
    assert result.model_expected_cost_mu == pytest.approx(expected)
    if maximum < target:
        assert result.maximum_achievable_fill_rate == pytest.approx(maximum)
    else:
        assert result.is_service_level_met


def test_invalid_dispatch_and_failed_solver_rejected(monkeypatch):
    inst = AllocationInstance(
        ["s"],
        ["p"],
        np.array([[0]]),
        np.array([0]),
        np.array([[1]]),
        np.array([1]),
        np.array([[6.0]]),
        np.array([[1.0]]),
        np.array([[0.1]]),
    )
    sim = InventoryStateTransitionSystem(inst)
    with pytest.raises(ValueError, match="Dispatch"):
        sim.step(np.array([[1]]), np.array([[0]]), 0)
    from types import SimpleNamespace

    from inventory_di.optimization import core

    monkeypatch.setattr(
        core,
        "milp",
        lambda **kw: SimpleNamespace(status=2, success=False, x=None, message="injected"),
    )
    with pytest.raises(RuntimeError, match="rejected"):
        solve_deterministic_milp(np.array([[1]]), inst)


def test_quick_outputs_are_isolated():
    normal = load_config("configs/demo.yaml")
    quick = load_config("configs/demo.yaml", quick=True)
    assert normal.run_id != quick.run_id
    for name in ["processed_dir", "artifacts_dir", "reports_dir"]:
        assert getattr(normal.data, name) != getattr(quick.data, name)


def test_missing_prices_never_backfill_from_future():
    rows = []
    for week in range(6):
        start = pd.Timestamp("2020-01-06") + pd.Timedelta(weeks=week)
        rows.append(
            dict(
                forecast_origin=str((start - pd.Timedelta(days=1)).date()),
                target_start=str(start.date()),
                target_end=str((start + pd.Timedelta(days=6)).date()),
                store_id="s",
                sku_id="p",
                observed_units_proxy=1,
                price_at_origin_mu=np.nan if week < 4 else 7.0,
                event_flag_target_week=0.0,
            )
        )
    before = compute_weekly_features_strictly_pre_origin(pd.DataFrame(rows))
    rows[-1]["price_at_origin_mu"] = 999
    after = compute_weekly_features_strictly_pre_origin(pd.DataFrame(rows))
    np.testing.assert_array_equal(before.lag_price_mu, after.lag_price_mu)
    assert before.lag_price_mu.iloc[0] == 1


def test_m5_join_contracts_reject_duplicate_keys(tmp_path):
    from inventory_di.data.m5 import ingest_m5_dataset

    cfg = load_config("configs/m5.yaml")
    cfg.data.num_stores = cfg.data.num_skus = 1
    cfg.data.num_weeks, cfg.data.train_weeks, cfg.data.val_weeks, cfg.data.test_weeks = 8, 4, 2, 2
    cfg.data.raw_m5_dir = str(tmp_path)
    dates = pd.date_range("2020-01-01", periods=70)
    calendar = pd.DataFrame(
        dict(
            d=[f"d_{i + 1}" for i in range(70)],
            date=dates.strftime("%Y-%m-%d"),
            wm_yr_wk=np.ones(70, dtype=int),
            event_name_1=[None] * 70,
        )
    )
    sales = dict(
        id="id", item_id="p", dept_id="HOUSEHOLD_1", cat_id="HOUSEHOLD", store_id="s", state_id="CA"
    )
    sales.update({f"d_{i + 1}": 1 for i in range(70)})
    prices = pd.DataFrame([dict(store_id="s", item_id="p", wm_yr_wk=1, sell_price=2.0)])
    pd.DataFrame([sales]).to_csv(tmp_path / "sales_train_evaluation.csv", index=False)
    calendar.to_csv(tmp_path / "calendar.csv", index=False)
    prices.to_csv(tmp_path / "sell_prices.csv", index=False)
    weekly, _, _ = ingest_m5_dataset(cfg)
    assert len(weekly) == 8
    pd.concat([calendar, calendar.iloc[:1]]).to_csv(tmp_path / "calendar.csv", index=False)
    with pytest.raises(pd.errors.MergeError):
        ingest_m5_dataset(cfg)
    calendar.to_csv(tmp_path / "calendar.csv", index=False)
    pd.concat([prices, prices]).to_csv(tmp_path / "sell_prices.csv", index=False)
    with pytest.raises(pd.errors.MergeError):
        ingest_m5_dataset(cfg)


def test_joint_final_diagnostics_match_calibrated_checkpoint():
    from inventory_di.data.synthetic import generate_synthetic_demo_dataset
    from inventory_di.evaluation.backtest import build_allocation_instance_from_config_df
    from inventory_di.models.dfl import train_joint_multi_sku_dfl_models
    from inventory_di.models.supervised import predict_sku_origin_outputs
    from inventory_di.optimization.core import evaluate_exact_cost, solve_nominal_marginal_cost_milp

    cfg = load_config("configs/demo.yaml")
    weekly, inv, _ = generate_synthetic_demo_dataset(cfg)
    inst = build_allocation_instance_from_config_df(inv)
    kwargs = dict(
        weekly_df=weekly,
        stores=inst.stores,
        skus=inst.skus,
        I_sp=inst.I,
        K_sp=inst.K,
        W_p=inst.W,
        C_s=inst.C,
        b_sp=inst.b,
        h_sp=inst.h,
        t_sp=inst.t,
        support_h=20,
        tail_mean_excess=2.0,
        hidden_dim=16,
        learning_rate=0.02,
        epochs=3,
        batch_size=8,
        spoplus_processes=1,
        seed=42,
    )
    baseline = train_joint_multi_sku_dfl_models(**kwargs, lambda_dfl=0)
    dfl = train_joint_multi_sku_dfl_models(**kwargs, lambda_dfl=1)
    assert any(
        not torch.equal(a, b)
        for sku in inst.skus
        for a, b in zip(baseline[sku].model.parameters(), dfl[sku].model.parameters())
    )
    regrets = []
    for _, frame in weekly[weekly.split == "val"].groupby("forecast_origin"):
        costs = {
            sku: predict_sku_origin_outputs(
                dfl[sku],
                frame,
                inst.I[:, p],
                inst.K[:, p],
                inst.b[:, p],
                inst.h[:, p],
                inst.t[:, p],
            )["c_hat_flat"]
            for p, sku in enumerate(inst.skus)
        }
        demand = (
            frame.pivot(index="store_id", columns="sku_id", values="observed_units_proxy")
            .loc[inst.stores, inst.skus]
            .to_numpy()
        )
        x = solve_nominal_marginal_cost_milp(costs, inst).x
        oracle = solve_deterministic_milp(demand, inst)
        regrets.append(
            (
                evaluate_exact_cost(x, demand, inst).total_cost_mu
                - evaluate_exact_cost(oracle.x, demand, inst).total_cost_mu
            )
            / len(inst.skus)
        )
    for art in dfl.values():
        assert art.calibration_info["val_decision_regret_mu"] == pytest.approx(
            np.mean(regrets), abs=1e-4
        )
        assert art.gradient_check["param_diff_l2_vs_nll"] >= 0
        assert art.calibration_info["epochs_budget"] == 3


def test_stage_lineage_rejects_changed_inputs_config_and_scientific_source(tmp_path, monkeypatch):
    from inventory_di import provenance

    cfg = load_config("configs/demo.yaml")
    cfg.data.processed_dir = str(tmp_path)
    input_file, output_file = tmp_path / "input", tmp_path / "output"
    input_file.write_text("original")
    output_file.write_text("result")
    stage = provenance.record_stage(cfg, "prepare", [input_file], [output_file])
    assert stage["source"]["source_files"]
    provenance.validate_stage(cfg, "prepare")
    input_file.write_text("changed")
    with pytest.raises(ValueError, match="Stale"):
        provenance.validate_stage(cfg, "prepare")
    input_file.write_text("original")
    changed_cfg = cfg.model_copy(deep=True)
    changed_cfg.model.epochs += 1
    with pytest.raises(ValueError, match="config mismatch"):
        provenance.validate_stage(changed_cfg, "prepare")
    source = dict(stage["source"])
    source["source_files"] = dict(source["source_files"])
    key = next(
        k for k in source["source_files"] if "/models/" in k and not k.endswith("__init__.py")
    )
    source["source_files"][key] = "changed"
    monkeypatch.setattr(provenance, "source_provenance", lambda: source)
    with pytest.raises(ValueError, match="Scientific source changed"):
        provenance.validate_stage(cfg, "prepare")
