"""Tests for exact cost evaluation and randomized marginal-unit cost identity (`AGENTS.md` §5, §6, §14)."""

from __future__ import annotations

import numpy as np

from inventory_di.optimization.core import (
    allocation_to_marginal_bits,
    marginal_bits_to_allocation,
    verify_marginal_cost_identity,
)


def test_randomized_marginal_unit_cost_identity_with_nonzero_opening_stock() -> None:
    rng = np.random.default_rng(2026)
    for trial in range(30):
        num_stores = int(rng.integers(2, 6))
        K_s = rng.integers(3, 12, size=num_stores)
        # Include both zero and nonzero opening stock I_s
        I_s = np.zeros(num_stores, dtype=int) if trial % 3 == 0 else rng.integers(0, 6, size=num_stores)
        d_s = rng.integers(0, 20, size=num_stores)
        x_s = np.array([int(rng.integers(0, int(k) + 1)) for k in K_s], dtype=int)
        b_s = rng.uniform(2.0, 10.0, size=num_stores)
        h_s = rng.uniform(0.2, 2.5, size=num_stores)
        t_s = rng.uniform(0.05, 1.0, size=num_stores)

        res = verify_marginal_cost_identity(x_s, d_s, I_s, K_s, b_s, h_s, t_s)
        assert res["matches"], f"Trial {trial} failed marginal cost identity: {res}"

        bits = allocation_to_marginal_bits(x_s, K_s)
        decoded = marginal_bits_to_allocation(bits, K_s)
        np.testing.assert_array_equal(decoded, x_s)
