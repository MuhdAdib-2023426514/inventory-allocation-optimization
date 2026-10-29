"""Plotly chart builders for the Streamlit decision-experiment dashboard and static report exports."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_historical_demand_by_split(weekly_df: pd.DataFrame, sku_id: str) -> go.Figure:
    """Plot weekly store-level demand across chronological train/val/test splits (Page A)."""
    sub = weekly_df[weekly_df["sku_id"] == sku_id].sort_values(["forecast_origin", "store_id"])
    fig = px.line(
        sub,
        x="forecast_origin",
        y="observed_units_proxy",
        color="store_id",
        symbol="split",
        markers=True,
        title=f"Weekly Demand Proxy by Store & Chronological Split ({sku_id})",
        labels={
            "forecast_origin": "Forecast Origin (Week)",
            "observed_units_proxy": "Weekly Demand (Units)",
            "store_id": "Store",
            "split": "Chronological Split",
        },
        template="plotly_white",
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_training_curves(history: list[dict[str, float]], sku_id: str, model_label: str) -> go.Figure:
    """Plot training NLL, SPO+ surrogate loss, and validation decision regret across epochs (Page B)."""
    df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["epoch"],
            y=df["train_nll"],
            mode="lines+markers",
            name="Train NLL",
        )
    )
    if "train_spoplus" in df.columns and (df["train_spoplus"] > 0).any():
        fig.add_trace(
            go.Scatter(
                x=df["epoch"],
                y=df["train_spoplus"],
                mode="lines+markers",
                name="Train Scaled SPO+ Surrogate",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=df["epoch"],
            y=df["val_nll"],
            mode="lines+markers",
            name="Validation NLL",
        )
    )
    if "val_decision_regret_mu" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["epoch"],
                y=df["val_decision_regret_mu"],
                mode="lines+markers",
                name="Validation Decision Regret (MU)",
            )
        )
    fig.update_layout(
        title=f"{model_label} Training & Validation Curve ({sku_id})",
        xaxis_title="Epoch",
        yaxis_title="Loss / Regret (MU)",
        template="plotly_white",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_forecast_quantiles(quantiles_df: pd.DataFrame, sku_id: str) -> go.Figure:
    """Compare Supervised vs DFL P10/P50/P90 and Expected Demand vs Realized Demand across stores (Page B)."""
    sub = quantiles_df[quantiles_df["sku_id"] == sku_id].sort_values("store_id")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=sub["store_id"],
            y=sub["realized_units"],
            name="Realized Demand Proxy (Units)",
            marker_color="#1f2937",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sub["store_id"],
            y=sub["sup_expected"],
            error_y=dict(
                type="data",
                symmetric=False,
                array=(sub["sup_p90"] - sub["sup_expected"]).tolist(),
                arrayminus=(sub["sup_expected"] - sub["sup_p10"]).tolist(),
            ),
            mode="markers",
            marker=dict(size=11, color="#2563eb"),
            name="Supervised Expected [P10, P90]",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sub["store_id"],
            y=sub["dfl_expected"],
            error_y=dict(
                type="data",
                symmetric=False,
                array=(sub["dfl_p90"] - sub["dfl_expected"]).tolist(),
                arrayminus=(sub["dfl_expected"] - sub["dfl_p10"]).tolist(),
            ),
            mode="markers",
            marker=dict(size=11, color="#dc2626", symbol="diamond"),
            name="DFL (NLL+SPO+) Expected [P10, P90]",
        )
    )
    fig.update_layout(
        title=f"Forecast Distribution Quantiles vs. Realized Demand ({sku_id})",
        xaxis_title="Store",
        yaxis_title="Demand (Units)",
        template="plotly_white",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_store_allocations_comparison(alloc_origin_df: pd.DataFrame, sku_id: str) -> go.Figure:
    """Compare store unit allocations across policies for a selected origin and SKU (Page C)."""
    sub = alloc_origin_df[alloc_origin_df["sku_id"] == sku_id].sort_values(["policy", "store_id"])
    fig = px.bar(
        sub,
        x="store_id",
        y="allocated_units",
        color="policy",
        barmode="group",
        title=f"Allocated Shipment Units by Store & Policy ({sku_id})",
        labels={
            "store_id": "Store",
            "allocated_units": "Allocated Shipment (Units)",
            "policy": "Policy",
        },
        template="plotly_white",
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_risk_cost_pareto(sensitivity_df: pd.DataFrame, supply_multiplier: float = 1.0) -> go.Figure:
    """Plot Model-Implied Expected Cost vs Model-Implied CVaR and Realized Cost across risk_weight (Page D)."""
    sub = sensitivity_df[
        (sensitivity_df["supply_multiplier"] - supply_multiplier).abs() < 1e-6
    ].copy()
    fig = px.scatter(
        sub,
        x="model_expected_cost_mu",
        y="model_cvar_cost_mu",
        color="policy_family",
        symbol="alpha",
        size="risk_weight",
        hover_data=["alpha", "risk_weight", "realized_test_cost_mu", "realized_fill_rate"],
        title=f"Risk vs. Expected Cost Frontier (Supply Multiplier = {supply_multiplier:.1f}x)",
        labels={
            "model_expected_cost_mu": "Model-Implied Expected Cost (MU)",
            "model_cvar_cost_mu": "Model-Implied Worst-Tail Cost / CVaR (MU)",
            "policy_family": "Policy Family",
        },
        template="plotly_white",
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_policy_cost_comparison(summary_df: pd.DataFrame) -> go.Figure:
    """Plot Mean Realized Cost (with 95% Origin Block-Bootstrap CI) vs Realized CVaR_0.90 across all 7 policies (Page E)."""
    df = summary_df.copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["policy"],
            y=df["mean_cost_mu"],
            name="Realized Mean Held-Out Cost (MU)",
            marker_color="#2563eb",
            error_y=dict(
                type="data",
                symmetric=False,
                array=(df["cost_ci95_high_mu"] - df["mean_cost_mu"]).tolist(),
                arrayminus=(df["mean_cost_mu"] - df["cost_ci95_low_mu"]).tolist(),
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["policy"],
            y=df["realized_cvar90_cost_mu"],
            name="Realized Worst-Tail Cost CVaR_0.90 (MU)",
            marker_color="#dc2626",
        )
    )
    fig.update_layout(
        title="Out-of-Sample Realized Mean Cost (with 95% Origin CI) & Tail Risk (CVaR_0.90)",
        xaxis_title="Allocation Policy",
        yaxis_title="Cost (MU)",
        barmode="group",
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
