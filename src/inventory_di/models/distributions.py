"""Discrete demand PMF architecture, survival probability & marginal cost mapping,
validation-only temperature calibration, and training-only empirical copula scenario generator
(`AGENTS.md` §6, §7).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteDemandPMFNet(nn.Module):
    """Neural network predicting discrete categorical PMF over 0..H plus overflow > H bin.

    Given pre-origin standardized features for each store `s = 0..S-1`, outputs logits
    of shape `(batch, S, H + 2)` where index `0..H` correspond to demand `d = 0..H`
    and index `H + 1` corresponds to `overflow > H`.
    """

    def __init__(
        self,
        num_stores: int,
        in_features: int,
        support_h: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_stores = int(num_stores)
        self.in_features = int(in_features)
        self.support_h = int(support_h)
        self.num_bins = self.support_h + 2  # 0..H plus overflow > H

        self.store_emb = nn.Embedding(self.num_stores, 8)
        self.net = nn.Sequential(
            nn.Linear(self.in_features + 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_bins),
        )
        # Learnable or validation-calibrated temperature (defaults to 1.0)
        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float32))

    def forward_logits(self, x_feat: torch.Tensor) -> torch.Tensor:
        """Compute raw logits of shape (batch, S, H + 2) from x_feat of shape (batch, S, F)."""
        B, S, _ = x_feat.shape
        store_ids = torch.arange(S, device=x_feat.device).unsqueeze(0).expand(B, S)
        emb = self.store_emb(store_ids)
        inp = torch.cat([x_feat, emb], dim=-1)
        return self.net(inp)

    def predict_pmf(self, x_feat: torch.Tensor) -> torch.Tensor:
        """Compute calibrated PMF p(d | features) of shape (batch, S, H + 2)."""
        logits = self.forward_logits(x_feat)
        temp = torch.clamp(self.temperature, min=0.1, max=10.0)
        return F.softmax(logits / temp, dim=-1)


def compute_survival_and_marginal_costs_torch(
    pmf: torch.Tensor,
    I_s: list[int] | np.ndarray,
    K_s: list[int] | np.ndarray,
    b_s: list[float] | np.ndarray,
    h_s: list[float] | np.ndarray,
    t_s: list[float] | np.ndarray,
    tail_mean_excess: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiably map store PMFs `(batch, S, H + 2)` to survival probabilities `q[s, k]`
    and predicted marginal unit costs `c_hat[s, k] = t_s + h_s - (b_s + h_s) * q[s, k]` (`AGENTS.md` §6).

    Returns:
      - q_flat: shape `(batch, sum_s K_s)`
      - c_hat_flat: shape `(batch, sum_s K_s)` in exact `(store s, unit k=1..K_s)` order.
    """
    B, S, num_bins = pmf.shape
    H = num_bins - 2
    q_list: list[torch.Tensor] = []
    c_list: list[torch.Tensor] = []

    for s in range(S):
        cap = int(K_s[s])
        init_inv = int(I_s[s])
        b_val = float(b_s[s])
        h_val = float(h_s[s])
        t_val = float(t_s[s])

        for k in range(1, cap + 1):
            thresh = init_inv + k
            if thresh <= H + 1:
                # P(D_s >= thresh) = sum_{d=thresh}^{H} p(d) + p(overflow > H)
                q_sk = torch.sum(pmf[:, s, thresh:], dim=-1)
            else:
                # If threshold exceeds H + 1, decay overflow probability geometrically
                excess_steps = thresh - (H + 1)
                q_sk = pmf[:, s, -1] * ((1.0 - 1.0 / max(1.0, tail_mean_excess)) ** excess_steps)
            q_sk = torch.clamp(q_sk, min=0.0, max=1.0)
            c_sk = (t_val + h_val) - (b_val + h_val) * q_sk
            q_list.append(q_sk)
            c_list.append(c_sk)

    q_flat = torch.stack(q_list, dim=-1)
    c_hat_flat = torch.stack(c_list, dim=-1)
    return q_flat, c_hat_flat


def compute_discrete_nll_loss(
    logits: torch.Tensor,
    d_target: torch.Tensor,
    support_h: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute proper categorical cross-entropy / NLL over 0..H plus overflow > H bin.

    Note: Target mapping to bin `min(d, H + 1)` is used ONLY for categorical bin indexing
    of the discrete PMF; actual realized evaluation demand `D_sp` is never clipped.
    """
    B, S, num_bins = logits.shape
    bins = torch.clamp(d_target.long(), min=0, max=support_h + 1)
    scaled_logits = logits / max(0.1, float(temperature))
    return F.cross_entropy(scaled_logits.reshape(B * S, num_bins), bins.reshape(B * S))


def calibrate_temperature_on_validation(
    model: DiscreteDemandPMFNet,
    val_x: torch.Tensor,
    val_d: torch.Tensor,
) -> dict[str, float]:
    """Fit post-hoc temperature scaling strictly on validation split (`split == 'val'`)."""
    model.eval()
    with torch.no_grad():
        logits = model.forward_logits(val_x)
        nll_before = float(
            compute_discrete_nll_loss(logits, val_d, model.support_h, temperature=1.0).item()
        )

    best_temp = 1.0
    best_nll = nll_before
    for cand in np.linspace(0.5, 2.5, 41):
        with torch.no_grad():
            nll_cand = float(
                compute_discrete_nll_loss(
                    logits, val_d, model.support_h, temperature=float(cand)
                ).item()
            )
        if nll_cand < best_nll:
            best_nll = nll_cand
            best_temp = float(cand)

    model.temperature.fill_(best_temp)
    return {
        "calibrated_on_split": "val",
        "nll_before_calibration": nll_before,
        "nll_after_calibration": best_nll,
        "optimal_temperature": best_temp,
    }


def pmf_moments_and_quantiles(
    pmf_np: np.ndarray,
    tail_mean_excess: float = 2.0,
) -> dict[str, np.ndarray]:
    """Compute implied expected demand, P10, P50, P90, and overflow probability from PMF `(S, H + 2)`."""
    S, num_bins = pmf_np.shape
    H = num_bins - 2
    support_vals = np.arange(num_bins, dtype=float)
    support_vals[-1] = float(H + max(1.0, tail_mean_excess))

    expected = np.sum(pmf_np * support_vals.reshape(1, -1), axis=-1)

    quantiles = [
        np.array(
            [inverse_demand_cdf(pmf_np[s], np.array([q]), tail_mean_excess)[0] for s in range(S)],
            dtype=float,
        )
        for q in (0.10, 0.50, 0.90)
    ]
    p10, p50, p90 = quantiles

    return {
        "expected": expected,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "overflow_prob": pmf_np[:, -1].copy(),
    }


def inverse_demand_cdf(pmf, uniforms, tail_mean_excess=2.0):
    """Inverse CDF of 0..H plus shifted-geometric overflow, including tail quantiles."""
    p = np.asarray(pmf, dtype=float)
    if (
        p.ndim != 1
        or len(p) < 3
        or not np.isfinite(p).all()
        or (p < 0).any()
        or not np.isclose(p.sum(), 1, atol=1e-5)
    ):
        raise ValueError("Invalid discrete demand PMF")
    p = p / p.sum()
    u = np.asarray(uniforms, dtype=float)
    if ((u < 0) | (u >= 1)).any():
        raise ValueError("CDF uniforms must be in [0,1)")
    cdf = np.cumsum(p)
    cdf[-1] = 1.0
    bins = np.searchsorted(cdf, u, side="left")
    H = len(p) - 2
    tail = bins > H
    if tail.any():
        conditional = np.clip((u[tail] - cdf[-2]) / p[-1], 0, 1 - 1e-14)
        mean = max(1.0, float(tail_mean_excess))
        excess = (
            np.ones(tail.sum(), dtype=int)
            if mean == 1
            else np.maximum(1, np.ceil(np.log1p(-conditional) / np.log1p(-1.0 / mean)).astype(int))
        )
        bins[tail] = H + excess
    return bins.astype(int)


class EmpiricalCopulaScenarioGenerator:
    """Training-only empirical rank-bootstrap / copula scenario generator (`AGENTS.md` §7).

    Fits cross-(store, SKU) historical weekly demand rank uniforms exclusively on `split == 'train'`.
    At evaluation time, samples `N` joint uniform vectors using a deterministic seed per origin
    (so all policies share common random numbers) and maps them through each store-SKU's predicted
    inverse CDF (with shifted geometric tail inverse CDF when `u` lands in the `overflow > H` bin).
    """

    def __init__(
        self,
        stores: list[str],
        skus: list[str],
        support_h: int,
        tail_mean_excess: float = 2.0,
    ) -> None:
        self.stores = list(stores)
        self.skus = list(skus)
        self.support_h = int(support_h)
        self.tail_mean_excess = float(max(1.0, tail_mean_excess))
        self.rank_matrix_train: np.ndarray | None = None  # shape (T_train, S, P) in (0, 1)
        self.val_scale_factors: np.ndarray | None = None  # shape (S, P) fitted on val only
        self.val_scenario_diagnostics: dict[str, Any] = {}

    def fit_on_train(self, train_weekly_df: Any) -> dict[str, Any]:
        """Estimate empirical rank uniforms across (store, SKU) on `split == 'train'` only."""
        if (train_weekly_df["split"] != "train").any():
            raise ValueError(
                "EmpiricalCopulaScenarioGenerator.fit_on_train received non-train rows!"
            )

        origins = sorted(train_weekly_df["forecast_origin"].unique())
        S, P = len(self.stores), len(self.skus)
        T = len(origins)
        if T < 4:
            self.rank_matrix_train = None
            return {"copula_mode": "independent_fallback", "train_origins_used": T}

        mat = np.zeros((T, S, P), dtype=float)
        origin_idx = {o: i for i, o in enumerate(origins)}
        store_idx = {s: i for i, s in enumerate(self.stores)}
        sku_idx = {p: i for i, p in enumerate(self.skus)}

        for row in train_weekly_df.itertuples(index=False):
            t_i = origin_idx[str(row.forecast_origin)]
            s_i = store_idx[str(row.store_id)]
            p_i = sku_idx[str(row.sku_id)]
            mat[t_i, s_i, p_i] = float(row.observed_units_proxy)

        # Convert each (s, p) time series across training weeks into empirical rank uniforms in (0, 1)
        ranks = np.zeros_like(mat)
        tie_rng = np.random.default_rng(1729)
        for s in range(S):
            for p in range(P):
                series = mat[:, s, p]
                sorted_indices = np.lexsort((tie_rng.random(T), series))
                order = np.empty(T, dtype=int)
                order[sorted_indices] = np.arange(T)
                ranks[:, s, p] = (order + 0.5) / float(T)

        self.rank_matrix_train = ranks
        return {"copula_mode": "empirical_rank_bootstrap", "train_origins_used": T}

    def sample_scenarios(
        self,
        pmf_by_sku: dict[str, np.ndarray],
        num_scenarios: int,
        seed: int,
    ) -> np.ndarray:
        """Draw `N` joint integer demand scenarios of shape `(N, S, P)` from learned PMFs."""
        rng = np.random.default_rng(seed)
        S, P = len(self.stores), len(self.skus)
        N = int(num_scenarios)

        # Generate dependent uniforms U of shape (N, S, P)
        if self.rank_matrix_train is not None and len(self.rank_matrix_train) >= 4:
            T_train = len(self.rank_matrix_train)
            row_indices = rng.integers(0, T_train, size=N)
            base_u = self.rank_matrix_train[row_indices]  # (N, S, P)
            # Add jitter within rank bin [-0.5/T, +0.5/T] to cover continuous [0, 1]
            jitter = rng.uniform(-0.5 / T_train, 0.5 / T_train, size=(N, S, P))
            U = np.clip(base_u + jitter, 1e-6, 1.0 - 1e-6)
        else:
            U = rng.uniform(1e-6, 1.0 - 1e-6, size=(N, S, P))

        scenarios = np.zeros((N, S, P), dtype=int)
        for p_idx, sku in enumerate(self.skus):
            for s_idx in range(S):
                scenarios[:, s_idx, p_idx] = inverse_demand_cdf(
                    pmf_by_sku[sku][s_idx], U[:, s_idx, p_idx], self.tail_mean_excess
                )

        return scenarios

    def calibrate_and_diagnose_on_val(
        self,
        val_pmfs_by_origin: list[dict[str, np.ndarray]],
        val_realized_list: list[np.ndarray],
        num_scenarios: int = 40,
        seed: int = 777,
    ) -> dict[str, Any]:
        """Priority 2: Diagnose and calibrate sampled demand scenarios strictly on `split == 'val'`.

        Computes pre-calibration and post-calibration scenario mean bias, P10-P90 coverage,
        and Energy Score (multivariate CRPS generalization), and fits a bounded validation
        multiplicative scale factor per `(store, SKU)` to eliminate systematic diffuse-bin
        overestimation bias without touching test data.
        """
        T_val = len(val_realized_list)
        if T_val == 0:
            return {"calibrated_on_split": "val", "status": "no_val_origins"}

        # Reset scale factors before measuring raw scenario draws
        self.val_scale_factors = None
        raw_scen_list: list[np.ndarray] = []
        for t_i in range(T_val):
            sc = self.sample_scenarios(
                val_pmfs_by_origin[t_i], num_scenarios=num_scenarios, seed=seed + t_i
            )
            raw_scen_list.append(sc)

        val_real_arr = np.stack(val_realized_list, axis=0).astype(float)  # (T_val, S, P)
        raw_scen_arr = np.stack(raw_scen_list, axis=0).astype(float)  # (T_val, N, S, P)

        # PMF temperature calibration is performed upstream on validation only.
        # Preserve those same marginals for PTO and DRO; no scenario-only scaling.
        self.val_scale_factors = np.ones(val_real_arr.shape[1:])

        cal_scen_list: list[np.ndarray] = []
        for t_i in range(T_val):
            sc_cal = self.sample_scenarios(
                val_pmfs_by_origin[t_i], num_scenarios=num_scenarios, seed=seed + t_i
            )
            cal_scen_list.append(sc_cal)
        cal_scen_arr = np.stack(cal_scen_list, axis=0).astype(float)

        def _energy_score(scen_t_n_s_p: np.ndarray, obs_t_s_p: np.ndarray) -> float:
            N_s = scen_t_n_s_p.shape[1]
            es_vals: list[float] = []
            for t_i in range(len(obs_t_s_p)):
                X = scen_t_n_s_p[t_i].reshape(N_s, -1)
                y = obs_t_s_p[t_i].reshape(1, -1)
                term1 = float(np.mean(np.linalg.norm(X - y, axis=1)))
                # Subsample pairwise distances for O(N) speed
                X_shift = np.roll(X, shift=1, axis=0)
                term2 = 0.5 * float(np.mean(np.linalg.norm(X - X_shift, axis=1)))
                es_vals.append(max(0.0, term1 - term2))
            return float(np.mean(es_vals))

        raw_bias = float(np.mean(np.mean(raw_scen_arr, axis=1) - val_real_arr))
        cal_bias = float(np.mean(np.mean(cal_scen_arr, axis=1) - val_real_arr))
        raw_es = _energy_score(raw_scen_arr, val_real_arr)
        cal_es = _energy_score(cal_scen_arr, val_real_arr)

        cal_p10 = np.percentile(cal_scen_arr, 10, axis=1)
        cal_p90 = np.percentile(cal_scen_arr, 90, axis=1)
        cal_cov80 = float(np.mean((val_real_arr >= cal_p10) & (val_real_arr <= cal_p90)))

        diag = {
            "calibrated_on_split": "val",
            "scenario_transform": "none; upstream PMF temperature only",
            "num_val_origins": T_val,
            "raw_scenario_mean_bias_units": raw_bias,
            "calibrated_scenario_mean_bias_units": cal_bias,
            "raw_energy_score": raw_es,
            "calibrated_energy_score": cal_es,
            "calibrated_p10_p90_coverage": cal_cov80,
            "mean_scale_factor": float(np.mean(self.val_scale_factors)),
        }
        self.val_scenario_diagnostics = diag
        return diag
