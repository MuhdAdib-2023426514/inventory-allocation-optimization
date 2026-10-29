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


class JointMultiSKUMarginalAllocationModel(optOmoModel):
    """Joint multi-SKU fixed-feasible-set marginal-unit allocation model for PyEPO `SPOPlus` (Priority 5).

    Aligns DFL training with the exact multi-SKU nominal MILP polytope:
      - Variables: `y_{p, s, k} in {0, 1}` in `(sku p=0..P-1, store s=0..S-1, unit k=1..K_{s,p})` order.
      - Constraints:
        1. Prefix ordering: `y_{p, s, k+1} <= y_{p, s, k}`
        2. Per-SKU warehouse supply: `sum_{s, k} y_{p, s, k} <= W_p`
        3. Joint multi-SKU store capacity: `sum_{p, k} y_{p, s, k} <= C_s - sum_p I_{s,p}`
    """

    def __init__(
        self,
        store_sku_caps: np.ndarray,
        warehouse_supplies: np.ndarray,
        rem_store_capacities: np.ndarray,
        solver: str = "appsi_highs",
    ) -> None:
        self.K = np.asarray(store_sku_caps, dtype=int)  # (S, P)
        self.W = np.asarray(warehouse_supplies, dtype=int)  # (P,)
        self.rem_C = np.asarray(rem_store_capacities, dtype=int)  # (S,)
        super().__init__(solver=solver)

    def get_config(self) -> dict:
        return {
            "store_sku_caps": self.K.tolist(),
            "warehouse_supplies": self.W.tolist(),
            "rem_store_capacities": self.rem_C.tolist(),
            "solver": self.solver,
        }

    def _getModel(self):
        S, P = self.K.shape
        m = pe.ConcreteModel()
        # Ordered identically to concatenating each SKU's (s, k) vector in p=0..P-1 order
        indices = [
            (p, s, k)
            for p in range(P)
            for s in range(S)
            for k in range(1, int(self.K[s, p]) + 1)
        ]
        m.indices = pe.Set(initialize=indices, dimen=3, ordered=True)
        m.x = pe.Var(m.indices, domain=pe.Binary)
        m.cons = pe.ConstraintList()

        # 1. Prefix ordering
        for p in range(P):
            for s in range(S):
                cap = int(self.K[s, p])
                for k in range(1, cap):
                    m.cons.add(m.x[(p, s, k + 1)] <= m.x[(p, s, k)])

        # 2. Per-SKU warehouse supply
        for p in range(P):
            m.cons.add(
                sum(
                    m.x[(p, s, k)]
                    for s in range(S)
                    for k in range(1, int(self.K[s, p]) + 1)
                )
                <= int(self.W[p])
            )

        # 3. Joint multi-SKU store capacity
        for s in range(S):
            m.cons.add(
                sum(
                    m.x[(p, s, k)]
                    for p in range(P)
                    for k in range(1, int(self.K[s, p]) + 1)
                )
                <= max(0, int(self.rem_C[s]))
            )

        return m, m.x


def build_joint_multi_sku_spoplus_layer(
    K_sp: np.ndarray,
    W_p: np.ndarray,
    I_sp: np.ndarray,
    C_s: np.ndarray,
    processes: int = 1,
    solver: str = "appsi_highs",
) -> tuple[JointMultiSKUMarginalAllocationModel, SPOPlus]:
    """Instantiate a JointMultiSKUMarginalAllocationModel and its PyEPO SPOPlus surrogate loss."""
    rem_c = np.asarray(C_s, dtype=int) - np.sum(np.asarray(I_sp, dtype=int), axis=1)
    opt_model = JointMultiSKUMarginalAllocationModel(
        store_sku_caps=K_sp,
        warehouse_supplies=W_p,
        rem_store_capacities=rem_c,
        solver=solver,
    )
    spoplus = SPOPlus(opt_model, processes=int(processes))
    return opt_model, spoplus


def precompute_joint_multi_sku_oracle_cache(
    opt_model: JointMultiSKUMarginalAllocationModel,
    demands_tensor_t_s_p: np.ndarray,
    I_sp: np.ndarray,
    K_sp: np.ndarray,
    b_sp: np.ndarray,
    h_sp: np.ndarray,
    t_sp: np.ndarray,
    split_name: str,
) -> dict[str, np.ndarray]:
    """Precompute joint multi-SKU true marginal cost vectors, optimal solutions, and objectives
    on `train` or `val` ONLY.
    """
    if split_name not in {"train", "val"}:
        raise ValueError(
            f"Joint oracle cache for SPO+ cannot be computed on split='{split_name}'"
        )

    num_origins, _, P = demands_tensor_t_s_p.shape
    num_cost = opt_model.num_cost
    true_costs = np.zeros((num_origins, num_cost), dtype=np.float32)
    true_sols = np.zeros((num_origins, num_cost), dtype=np.float32)
    true_objs = np.zeros(num_origins, dtype=np.float32)

    for idx in range(num_origins):
        c_parts = [
            compute_true_marginal_unit_costs(
                d_s=demands_tensor_t_s_p[idx, :, p],
                I_s=I_sp[:, p],
                K_s=K_sp[:, p],
                b_s=b_sp[:, p],
                h_s=h_sp[:, p],
                t_s=t_sp[:, p],
            )
            for p in range(P)
        ]
        c_vec = np.concatenate(c_parts, axis=0)
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
