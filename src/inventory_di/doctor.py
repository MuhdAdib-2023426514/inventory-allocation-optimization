#!/usr/bin/env python3
"""Environment, solver, and PyEPO SPO+ smoke test script (Phase 0)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyomo.environ as pe
import torch
from pyepo.func import SPOPlus
from pyepo.model.omo import optOmoModel
from scipy.optimize import LinearConstraint, milp


class _SmokeMarginalAllocationModel(optOmoModel):
    """Tiny fixed-feasible marginal-unit allocation model for Phase 0 smoke testing."""

    def __init__(
        self,
        store_caps: tuple[int, ...],
        warehouse_supply: int,
        solver: str = "appsi_highs",
    ) -> None:
        self.store_caps = tuple(int(k) for k in store_caps)
        self.warehouse_supply = int(warehouse_supply)
        super().__init__(solver=solver)

    def _getModel(self):
        m = pe.ConcreteModel()
        indices = [
            (s, k)
            for s, cap in enumerate(self.store_caps)
            for k in range(1, cap + 1)
        ]
        m.indices = pe.Set(initialize=indices, dimen=2, ordered=True)
        m.x = pe.Var(m.indices, domain=pe.Binary)
        m.cons = pe.ConstraintList()
        for s, cap in enumerate(self.store_caps):
            for k in range(1, cap):
                m.cons.add(m.x[(s, k + 1)] <= m.x[(s, k)])
        m.cons.add(sum(m.x[idx] for idx in indices) <= self.warehouse_supply)
        return m, m.x


def run_environment_doctor() -> dict:
    """Run Phase 0 environment, MILP toy oracle, and PyEPO SPO+ gradient checks."""
    # 1. Toy MILP check via SciPy/HiGHS: D=[60,20], W=60, K=[60,60], b=1, h=0, t=0
    # Verify both equal [30, 30] -> cost=30.0, fill=0.625 and optimal [40, 20] -> cost=20.0, fill=0.75,
    # and confirm MILP solver achieves optimal objective 20.0 (75.0% fill rate).
    c = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
    integrality = np.array([1, 1, 0, 0])
    lb = np.array([0.0, 0.0, 0.0, 0.0])
    ub = np.array([60.0, 60.0, 100.0, 100.0])
    A = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0, -1.0],
        ]
    )
    b_l = np.array([-np.inf, -np.inf, -np.inf])
    b_u = np.array([60.0, -60.0, -20.0])
    res = milp(c=c, integrality=integrality, bounds=(lb, ub), constraints=LinearConstraint(A, b_l, b_u))
    x_opt = [int(round(res.x[0])), int(round(res.x[1]))]
    toy_cost = float(max(60 - x_opt[0], 0) + max(20 - x_opt[1], 0))
    toy_fill_rate = float((min(60, x_opt[0]) + min(20, x_opt[1])) / 80.0)
    equal_cost = float(max(60 - 30, 0) + max(20 - 30, 0))
    equal_fill_rate = float((min(60, 30) + min(20, 30)) / 80.0)
    opt_40_20_cost = float(max(60 - 40, 0) + max(20 - 20, 0))
    opt_40_20_fill_rate = float((min(60, 40) + min(20, 20)) / 80.0)
    if (
        abs(toy_cost - 20.0) > 1e-6
        or abs(toy_fill_rate - 0.75) > 1e-6
        or abs(equal_cost - 30.0) > 1e-6
        or abs(equal_fill_rate - 0.625) > 1e-6
        or abs(opt_40_20_cost - 20.0) > 1e-6
        or abs(opt_40_20_fill_rate - 0.75) > 1e-6
    ):
        raise RuntimeError(f"Toy MILP check failed: x={x_opt}, cost={toy_cost}, fill={toy_fill_rate}")

    # 2. PyEPO SPO+ forward & backward pass with appsi_highs
    torch.manual_seed(42)
    opt_model = _SmokeMarginalAllocationModel((4, 4), 5, solver="appsi_highs")
    spoplus = SPOPlus(opt_model, processes=1)
    c_true_np = np.array([[-5.0, -5.0, -5.0, 1.0, -5.0, -5.0, 1.0, 1.0]], dtype=np.float32)
    opt_model.setObj(c_true_np[0])
    sol_true, obj_true = opt_model.solve()

    linear = torch.nn.Linear(4, 8)
    x_feat = torch.randn(1, 4)
    c_pred = linear(x_feat)
    loss = spoplus(
        c_pred,
        torch.tensor(c_true_np),
        torch.tensor(np.array([sol_true], dtype=np.float32)),
        torch.tensor([obj_true], dtype=torch.float32),
    )
    loss.backward()
    grad_norm = float(linear.weight.grad.norm().item())
    if not np.isfinite(grad_norm) or grad_norm <= 0.0:
        raise RuntimeError(f"PyEPO SPO+ gradient check failed: grad_norm={grad_norm}")

    m5_dir = Path("data/raw/m5")
    m5_files = {
        "calendar.csv": (m5_dir / "calendar.csv").exists(),
        "sell_prices.csv": (m5_dir / "sell_prices.csv").exists(),
        "sales_train_evaluation.csv": (m5_dir / "sales_train_evaluation.csv").exists(),
        "sales_train_validation.csv": (m5_dir / "sales_train_validation.csv").exists(),
    }

    return {
        "status": "PASS",
        "python_version": sys.version.split()[0],
        "toy_milp_oracle": {
            "optimal_allocation": x_opt,
            "realized_cost_mu": toy_cost,
            "fill_rate": toy_fill_rate,
            "solver_status": "optimal" if res.success else "failed",
        },
        "pyepo_spoplus_smoke": {
            "solver": "appsi_highs",
            "true_objective": float(obj_true),
            "spoplus_loss": float(loss.item()),
            "weight_grad_norm": grad_norm,
        },
        "m5_raw_data_present": any(m5_files.values()),
        "m5_files": m5_files,
    }


def main() -> int:
    report = run_environment_doctor()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
