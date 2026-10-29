"""Explicit state-transition multi-period inventory simulator (`AGENTS.md` §2, §5).

Tracks physical inventory conservation across:
- Store on-hand inventory by SKU
- In-transit shipments in the replenishment pipeline
- Deferred arrivals (undeliverable shipments retained in transit when store capacity is full)
- Fulfilled demand (sales)
- Unmet demand (shortages)
- Cumulative external receipts / initial inventory

Verifies the system-wide conservation invariant at every single period:
    Opening System Inventory + External Receipts == Closing System Inventory + Fulfilled Sales + Recorded Disposals
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from inventory_di.optimization.core import AllocationInstance


@dataclass
class InventoryStepLog:
    """Detailed audit record for a single period state transition."""

    period: int
    opening_store_on_hand: np.ndarray  # (S, P)
    scheduled_arrivals: np.ndarray  # (S, P)
    deferred_arrivals_before: np.ndarray  # (S, P)
    accepted_arrivals: np.ndarray  # (S, P)
    deferred_arrivals_after: np.ndarray  # (S, P)
    store_stock_before_sales: np.ndarray  # (S, P)
    order_placed: np.ndarray  # (S, P)
    realized_demand: np.ndarray  # (S, P)
    fulfilled_sales: np.ndarray  # (S, P)
    unmet_shortage: np.ndarray  # (S, P)
    closing_store_on_hand: np.ndarray  # (S, P)
    in_transit_pipeline_after: list[np.ndarray]  # list of (S, P)
    period_cost_mu: float
    shortage_cost_mu: float
    holding_cost_mu: float
    shipping_cost_mu: float
    fill_rate: float
    conservation_error_units: int


class InventoryStateTransitionSystem:
    """Rigorous physical inventory state-transition system with strict conservation.

    Handling policy for arrived inventory when store capacity is exceeded:
      - Units up to available store capacity (C_s - sum_p I_{s,p}) are accepted on shelves.
      - Any excess units are retained in the deferred delivery queue (held in transit/dock)
        and attempt delivery in the next period when space frees up.
      - ZERO units are destroyed, lost, or silently discarded.
    """

    def __init__(
        self,
        base_inst: AllocationInstance,
        lead_time_weeks: int = 0,
        initial_pipeline: list[np.ndarray] | None = None,
    ) -> None:
        self.base_inst = base_inst
        self.S = len(base_inst.stores)
        self.P = len(base_inst.skus)
        if int(lead_time_weeks) != lead_time_weeks or lead_time_weeks < 0:
            raise ValueError("Lead time must be a nonnegative integer")
        self.lead_time = int(lead_time_weeks)
        self.validate_units(base_inst.I, "opening inventory")
        if np.any(base_inst.I.sum(axis=1) > base_inst.C):
            raise ValueError("Opening store stock exceeds capacity")
        if initial_pipeline is not None and len(initial_pipeline) != self.lead_time:
            raise ValueError("Initial pipeline length must equal lead time")
        for shipment in initial_pipeline or []:
            self.validate_units(shipment, "pipeline")

        # Store on-hand stock
        self.store_on_hand = np.array(base_inst.I, dtype=int, copy=True)
        # Deferred arrivals retained in transit (dock/truck) when store is full
        self.deferred_arrivals = np.zeros((self.S, self.P), dtype=int)

        # Pipeline of scheduled future arrivals: length == lead_time
        if initial_pipeline is not None:
            self.pipeline: list[np.ndarray] = [
                np.array(arr, dtype=int, copy=True) for arr in initial_pipeline
            ]
        else:
            self.pipeline = [np.zeros((self.S, self.P), dtype=int) for _ in range(self.lead_time)]

        # Cumulative counters
        self.fulfilled_sales_cumulative = 0
        self.unmet_demand_cumulative = 0
        self.total_cost_mu_cumulative = 0.0
        self.shortage_cost_mu_cumulative = 0.0
        self.holding_cost_mu_cumulative = 0.0
        self.shipping_cost_mu_cumulative = 0.0
        self.total_demand_cumulative = 0
        self.external_receipts_cumulative = 0
        self.disposed_units_cumulative = 0

        # Initial total system inventory: stores + deferred + pipeline
        self.initial_system_inventory = int(
            np.sum(self.store_on_hand)
            + np.sum(self.deferred_arrivals)
            + sum(int(np.sum(p)) for p in self.pipeline)
        )

        self.initial_by_sku = self.store_on_hand.sum(axis=0) + sum(
            (p.sum(axis=0) for p in self.pipeline), np.zeros(self.P, dtype=int))
        self.receipts_by_sku = np.zeros(self.P, dtype=int)
        self.sales_by_sku = np.zeros(self.P, dtype=int)
        self.step_history: list[InventoryStepLog] = []

    def validate_units(self, value, label):
        arr = np.asarray(value)
        if arr.shape != (self.S, self.P) or not np.isfinite(arr).all() or (arr < 0).any() or (arr != np.rint(arr)).any():
            raise ValueError(f"{label} must have shape {(self.S, self.P)} and finite nonnegative integer units")
        return arr.astype(int)

    def get_system_inventory_in_circulation(self) -> int:
        """Calculate total physical inventory currently in the system."""
        return int(
            np.sum(self.store_on_hand)
            + np.sum(self.deferred_arrivals)
            + sum(int(np.sum(p)) for p in self.pipeline)
        )

    def verify_conservation(self) -> tuple[bool, int]:
        """Verify: Opening System Inventory + Receipts == Current Inventory + Fulfilled Sales + Disposals."""
        current_in_circulation = self.get_system_inventory_in_circulation()
        expected = self.initial_system_inventory + self.external_receipts_cumulative
        actual = current_in_circulation + self.fulfilled_sales_cumulative + self.disposed_units_cumulative
        diff = actual - expected
        by_sku = self.store_on_hand.sum(axis=0) + self.deferred_arrivals.sum(axis=0) + sum(
            (p.sum(axis=0) for p in self.pipeline), np.zeros(self.P, dtype=int))
        sku_ok = np.array_equal(self.initial_by_sku + self.receipts_by_sku, by_sku + self.sales_by_sku)
        return diff == 0 and sku_ok, int(diff)

    def step(
        self,
        order_x: np.ndarray,
        realized_demand_D: np.ndarray,
        period_idx: int,
        admitted_units: np.ndarray | None = None,
    ) -> InventoryStepLog:
        """Execute one period state transition.

        Order of operations:
        1. Receive shipments scheduled to arrive + any previously deferred units.
        2. Accept units up to physical store capacity; retain excess in deferred queue.
        3. Dispatch new order `order_x` into pipeline (or immediate arrival if lead_time == 0).
        4. Realize demand D and fulfill sales up to available store stock.
        5. Update closing stock and compute period costs.
        6. Verify exact system conservation invariant.
        """
        order_x = self.validate_units(order_x, "order")
        realized_demand_D = self.validate_units(realized_demand_D, "demand")
        if (order_x > self.base_inst.K).any() or (order_x.sum(axis=0) > self.base_inst.W).any():
            raise ValueError("Dispatch exceeds SKU cap or warehouse supply")
        if admitted_units is not None:
            admitted_units = self.validate_units(admitted_units, "admissions")
            due = order_x if self.lead_time == 0 else self.pipeline[0]
            if (admitted_units > due + self.deferred_arrivals).any() or ((self.store_on_hand + admitted_units).sum(axis=1) > self.base_inst.C).any():
                raise ValueError("Admissions exceed queued stock or store capacity")
        self.receipts_by_sku += order_x.sum(axis=0)

        # 1. Opening store on-hand
        opening_store = self.store_on_hand.copy()
        deferred_before = self.deferred_arrivals.copy()

        # 2. Arriving shipments
        if self.lead_time > 0:
            # Pop arrival from pipeline and add any new order to end of pipeline
            scheduled_arrival = self.pipeline.pop(0)
            self.pipeline.append(order_x.copy())
            self.external_receipts_cumulative += int(np.sum(order_x))
        else:
            # Lead time 0: order arrives in the current period
            scheduled_arrival = order_x.copy()
            self.external_receipts_cumulative += int(np.sum(order_x))

        # Total units attempting to enter store floor: scheduled + deferred
        attempting_arrival = scheduled_arrival + deferred_before
        accepted_arrival = np.zeros((self.S, self.P), dtype=int)
        deferred_after = np.zeros((self.S, self.P), dtype=int)

        # Enforce store total capacity C_s without destroying units
        for s in range(self.S):
            current_s = int(np.sum(opening_store[s, :]))
            capacity_s = int(self.base_inst.C[s])
            available_space = max(0, capacity_s - current_s)
            total_incoming = int(np.sum(attempting_arrival[s, :]))

            if total_incoming <= available_space:
                accepted_arrival[s, :] = attempting_arrival[s, :]
                deferred_after[s, :] = 0
            else:
                # Store is full or partially full: accept up to available space
                # Proportional allocation across SKUs of incoming batch
                if total_incoming > 0 and available_space > 0:
                    scale = float(available_space) / float(total_incoming)
                    accepted_s = np.floor(attempting_arrival[s, :] * scale).astype(int)
                    # Allocate remaining fractional slots deterministically
                    rem_slots = available_space - int(np.sum(accepted_s))
                    if rem_slots > 0:
                        frac_parts = (attempting_arrival[s, :] * scale) - accepted_s
                        top_indices = np.argsort(-frac_parts)[:rem_slots]
                        for idx in top_indices:
                            accepted_s[idx] += 1
                    accepted_arrival[s, :] = accepted_s
                else:
                    accepted_arrival[s, :] = 0

                # Crucial: Undeliverable units are RETAINED in deferred queue, NEVER destroyed
                deferred_after[s, :] = attempting_arrival[s, :] - accepted_arrival[s, :]

        if admitted_units is not None:
            accepted_arrival = admitted_units.copy()
            deferred_after = attempting_arrival - accepted_arrival

        # Update deferred queue
        self.deferred_arrivals = deferred_after

        # Store stock available for sales in this period
        store_stock_before_sales = opening_store + accepted_arrival

        # 3. Realize demand and fulfill sales
        fulfilled_sales = np.minimum(realized_demand_D, store_stock_before_sales).astype(int)
        unmet_shortage = np.maximum(0, realized_demand_D - store_stock_before_sales).astype(int)
        closing_store_on_hand = (store_stock_before_sales - fulfilled_sales).astype(int)

        # Update on-hand stock
        self.store_on_hand = closing_store_on_hand

        # 4. Cost accounting
        period_shortage_cost = float(np.sum(self.base_inst.b * unmet_shortage))
        period_holding_cost = float(np.sum(self.base_inst.h * closing_store_on_hand))
        # Holding cost also applies to deferred stock retained in transit / dock
        period_holding_cost += float(np.sum(self.base_inst.h * deferred_after))
        period_shipping_cost = float(np.sum(self.base_inst.t * order_x))
        period_total_cost = period_shortage_cost + period_holding_cost + period_shipping_cost

        period_demand = int(np.sum(realized_demand_D))
        period_fulfilled = int(np.sum(fulfilled_sales))
        fill_rate = float(period_fulfilled / period_demand) if period_demand else 1.0

        # Accumulate metrics
        self.sales_by_sku += fulfilled_sales.sum(axis=0)
        self.fulfilled_sales_cumulative += period_fulfilled
        self.unmet_demand_cumulative += int(np.sum(unmet_shortage))
        self.total_demand_cumulative += period_demand
        self.total_cost_mu_cumulative += period_total_cost
        self.shortage_cost_mu_cumulative += period_shortage_cost
        self.holding_cost_mu_cumulative += period_holding_cost
        self.shipping_cost_mu_cumulative += period_shipping_cost

        # 5. Strict Conservation Invariant Check
        is_conserved, error_units = self.verify_conservation()
        if not is_conserved:
            raise AssertionError(
                f"Physical inventory conservation violated at period {period_idx}: "
                f"discrepancy of {error_units} units! (Expected {self.initial_system_inventory + self.external_receipts_cumulative}, "
                f"Actual {self.get_system_inventory_in_circulation() + self.fulfilled_sales_cumulative})"
            )

        step_log = InventoryStepLog(
            period=period_idx,
            opening_store_on_hand=opening_store,
            scheduled_arrivals=scheduled_arrival,
            deferred_arrivals_before=deferred_before,
            accepted_arrivals=accepted_arrival,
            deferred_arrivals_after=deferred_after,
            store_stock_before_sales=store_stock_before_sales,
            order_placed=order_x,
            realized_demand=realized_demand_D,
            fulfilled_sales=fulfilled_sales,
            unmet_shortage=unmet_shortage,
            closing_store_on_hand=closing_store_on_hand,
            in_transit_pipeline_after=[p.copy() for p in self.pipeline],
            period_cost_mu=period_total_cost,
            shortage_cost_mu=period_shortage_cost,
            holding_cost_mu=period_holding_cost,
            shipping_cost_mu=period_shipping_cost,
            fill_rate=fill_rate,
            conservation_error_units=error_units,
        )
        self.step_history.append(step_log)
        return step_log

    def get_summary(self) -> dict[str, Any]:
        """Return cumulative evaluation summary across all executed steps."""
        n_periods = max(1, len(self.step_history))
        return {
            "num_periods": len(self.step_history),
            "mean_cost_mu": self.total_cost_mu_cumulative / n_periods,
            "mean_shortage_cost_mu": self.shortage_cost_mu_cumulative / n_periods,
            "mean_holding_cost_mu": self.holding_cost_mu_cumulative / n_periods,
            "mean_shipping_cost_mu": self.shipping_cost_mu_cumulative / n_periods,
            "fill_rate": float(self.fulfilled_sales_cumulative / max(1, self.total_demand_cumulative)),
            "ending_store_inventory_units": int(np.sum(self.store_on_hand)),
            "ending_in_transit_units": sum(int(np.sum(p)) for p in self.pipeline) + int(np.sum(self.deferred_arrivals)),
            "ending_total_system_units": self.get_system_inventory_in_circulation(),
            "total_sales_units": self.fulfilled_sales_cumulative,
            "total_unmet_units": self.unmet_demand_cumulative,
            "is_conserved": self.verify_conservation()[0],
        }
