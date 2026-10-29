"""Fixed-feasible-set PyEPO optimization model (`optOmoModel` + `appsi_highs`) for genuine SPO+
decision-focused learning (`AGENTS.md` §6).
"""

from __future__ import annotations

import numpy as np
import pyomo.environ as pe
from pyepo.func import SPOPlus
from pyepo.model.omo import optOmoModel

from inventory_di.optimization.core import compute_true_marginal_unit_costs


class FixedFeasibleMarginalAllocationModel(optOmoModel):
    """Per-SKU fixed-feasible-set marginal-unit allocation model for PyEPO `SPOPlus`.

    Decision variables:
      `y_{s, k} in {0, 1}` for store `s = 0..S-1` and incremental unit `k = 1..K_s`.
    Fixed constraints across all training instances for a SKU:
      1. Prefix ordering: `y_{s, k+1} <= y_{s, k}` for `k = 1..K_s - 1`
      2. Fixed warehouse budget: `sum_{s, k} y_{s, k} <= W_p`
    """

    def __init__(
        self,
        store_caps: tuple[int, ...],
        warehouse_supply: int,
        solver: str = "appsi_highs",
    ) -> None:
        self.store_caps = tuple(int(k) for k in store_caps)
        self.warehouse_supply = int(warehouse_supply)
        super().__init__(solver=solver)

    def get_config(self) -> dict:
        return {
            "store_caps": self.store_caps,
            "warehouse_supply": self.warehouse_supply,
            "solver": self.solver,
        }

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


def build_spoplus_layer(
    store_caps: list[int] | np.ndarray,
    warehouse_supply: int,
    processes: int = 1,
    solver: str = "appsi_highs",
) -> tuple[FixedFeasibleMarginalAllocationModel, SPOPlus]:
    """Instantiate a FixedFeasibleMarginalAllocationModel and its PyEPO SPOPlus surrogate loss."""
    opt_model = FixedFeasibleMarginalAllocationModel(
        store_caps=tuple(int(k) for k in store_caps),
        warehouse_supply=int(warehouse_supply),
        solver=solver,
    )
    spoplus = SPOPlus(opt_model, processes=int(processes))
    return opt_model, spoplus


def precompute_sku_oracle_cache(
    opt_model: FixedFeasibleMarginalAllocationModel,
    demands_matrix: np.ndarray,
    I_s: np.ndarray,
    K_s: np.ndarray,
    b_s: np.ndarray,
    h_s: np.ndarray,
    t_s: np.ndarray,
    split_name: str,
) -> dict[str, np.ndarray]:
    """Precompute true marginal cost vectors `c_true`, optimal binary solutions `y*`,
    and optimal objective values `z*` for a split (`train` or `val` ONLY).
    """
    if split_name not in {"train", "val"}:
        raise ValueError(
            f"Oracle cache for SPO+ training/validation cannot be computed on split='{split_name}'"
        )

    num_origins = demands_matrix.shape[0]
    num_cost = opt_model.num_cost
    true_costs = np.zeros((num_origins, num_cost), dtype=np.float32)
    true_sols = np.zeros((num_origins, num_cost), dtype=np.float32)
    true_objs = np.zeros(num_origins, dtype=np.float32)

    for idx in range(num_origins):
        c_vec = compute_true_marginal_unit_costs(
            d_s=demands_matrix[idx],
            I_s=I_s,
            K_s=K_s,
            b_s=b_s,
            h_s=h_s,
            t_s=t_s,
        )
        opt_model.setObj(c_vec)
        sol, obj = opt_model.solve()
        true_costs[idx] = c_vec.astype(np.float32)
        true_sols[idx] = sol.astype(np.float32)
        true_objs[idx] = float(obj)

    return {
        "true_costs": true_costs,
        "true_sols": true_sols,
        "true_objs": true_objs,
        "split": np.array([split_name]),
    }
