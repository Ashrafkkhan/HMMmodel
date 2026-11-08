"""Plotly figures used by the dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from .clustering import ward_linkage_matrix


def _layout(fig, title=None, height=380):
    fig.update_layout(title=title, height=height, margin=dict(l=40, r=20, t=50, b=40))
    return fig


def variance_fig(explained: pd.DataFrame, n_selected: int, threshold: float):
    fig = go.Figure()
    fig.add_bar(x=explained["Component"], y=explained["ExplainedVariance"], name="Per component")
    fig.add_scatter(x=explained["Component"], y=explained["Cumulative"], name="Cumulative", mode="lines+markers")
    fig.add_hline(y=threshold, line_dash="dash", annotation_text=f"{threshold:.0%} threshold")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
    return _layout(fig, f"Explained variance ({n_selected} component(s) selected for modelling)")


def _cat(labels, mapping):
    return pd.Series(labels).map(mapping if mapping else (lambda x: f"Cluster {x}")).astype(str)


def scatter_fig(Z_viz: np.ndarray, labels, mapping=None, title="PC1-PC2 projection", n_model_components=2):
    df = pd.DataFrame({"PC1": Z_viz[:, 0], "PC2": Z_viz[:, 1], "Segment": _cat(labels, mapping)})
    fig = px.scatter(df, x="PC1", y="PC2", color="Segment", opacity=0.6,
                     category_orders={"Segment": sorted(df["Segment"].unique())})
    note = (f"2-D projection only - models were trained on {n_model_components} principal components"
            if n_model_components > 2 else "Models were trained on these 2 components")
    return _layout(fig, f"{title}<br><sup>{note}</sup>", 460)


def sweep_line(df: pd.DataFrame, y: str, title: str, x: str = "k"):
    fig = px.line(df, x=x, y=y, markers=True)
    fig.update_xaxes(dtick=1)
    return _layout(fig, title, 320)


def gmm_ic_fig(sweep: pd.DataFrame):
    fig = go.Figure()
    fig.add_scatter(x=sweep["k"], y=sweep["BIC"], name="BIC", mode="lines+markers")
    fig.add_scatter(x=sweep["k"], y=sweep["AIC"], name="AIC", mode="lines+markers")
    fig.update_xaxes(dtick=1, title="Components")
    return _layout(fig, "GMM information criteria (lower is better)", 320)


def size_bar(profile: pd.DataFrame, title="Cluster sizes"):
    d = profile.assign(Label=profile["Cluster"].astype(str) + ": " + profile["Segment"])
    fig = px.bar(d, x="Label", y="Customers", text="Customer %")
    fig.update_traces(texttemplate="%{text:.1f}%")
    return _layout(fig, title, 340)


def revenue_bar(profile: pd.DataFrame):
    d = profile.assign(Label=profile["Cluster"].astype(str) + ": " + profile["Segment"])
    fig = px.bar(d, x="Label", y="Revenue %")
    return _layout(fig, "Share of purchase revenue by cluster (%)", 340)


def heatmap_fig(rel: pd.DataFrame, mapping=None):
    rel = rel.copy()
    rel.index = [mapping.get(i, str(i)) if mapping else f"Cluster {i}" for i in rel.index]
    fig = px.imshow(rel, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", color_continuous_midpoint=1)
    return _layout(fig, "Cluster median / overall median (original features)", 380)


def spending_box(features: pd.DataFrame, labels, mapping=None):
    df = pd.DataFrame({"Monetary": features["Monetary"].values, "Segment": _cat(labels, mapping)})
    fig = px.box(df, x="Segment", y="Monetary", log_y=True)
    return _layout(fig, "Spending distribution by cluster (log scale)", 380)


def recency_monetary_scatter(features: pd.DataFrame, labels, mapping=None):
    df = features.assign(Segment=_cat(labels, mapping).values)
    fig = px.scatter(df, x="Recency", y="Monetary", color="Segment", log_y=True, opacity=0.6,
                     hover_data=["CustomerID", "Frequency"])
    return _layout(fig, "Recency vs monetary (original values, log y)", 420)


def feature_hist(features: pd.DataFrame, col: str, log_x: bool):
    fig = px.histogram(features, x=col, nbins=50, log_x=log_x)
    return _layout(fig, f"{col} distribution" + (" (log x)" if log_x else ""), 280)


def dendrogram_fig(Z_sample: np.ndarray):
    fig = ff.create_dendrogram(Z_sample, linkagefun=lambda x: ward_linkage_matrix(x))
    fig.update_xaxes(showticklabels=False)
    return _layout(fig, f"Ward dendrogram - sample of {len(Z_sample)} customers (visualization only)", 420)
