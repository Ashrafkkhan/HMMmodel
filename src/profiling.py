"""Segment profiles computed from ORIGINAL (untransformed) customer features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_engineering import EXTENDED_FEATURES


def profile_segments(features: pd.DataFrame, labels: np.ndarray, names: dict | None = None) -> pd.DataFrame:
    df = features.copy()
    df["Cluster"] = np.asarray(labels)
    total_rev = df["Monetary"].sum()
    g = df.groupby("Cluster")
    prof = pd.DataFrame({
        "Customers": g.size(),
        "Total revenue": g["Monetary"].sum(),
        "Mean monetary": g["Monetary"].mean(),
        "Median monetary": g["Monetary"].median(),
        "Median recency": g["Recency"].median(),
        "Median frequency": g["Frequency"].median(),
        "Median average order value": g["AverageOrderValue"].median(),
        "Median quantity": g["TotalQuantity"].median(),
        "Median unique products": g["UniqueProducts"].median(),
    })
    prof.insert(1, "Customer %", 100 * prof["Customers"] / len(df))
    prof.insert(3, "Revenue %", 100 * prof["Total revenue"] / total_rev)
    prof = prof.reset_index()
    auto = suggest_names(prof, features)
    names = names or {}
    prof.insert(1, "Segment", [names.get(int(c), auto[int(c)]) for c in prof["Cluster"]])
    return prof


def suggest_names(profile: pd.DataFrame, features: pd.DataFrame) -> dict[int, str]:
    """Descriptive names from how each cluster's medians compare with the overall medians.

    These are starting points derived from the actual profiles; edit them as needed.
    """
    om = {c: max(features[c].median(), 1e-9) for c in ("Recency", "Frequency", "Monetary")}
    raw = {}
    for _, r in profile.iterrows():
        hi_m = r["Median monetary"] >= 1.5 * om["Monetary"]
        lo_m = r["Median monetary"] <= 0.67 * om["Monetary"]
        hi_f = r["Median frequency"] >= 1.5 * om["Frequency"]
        lo_f = r["Median frequency"] <= 0.67 * om["Frequency"]
        old = r["Median recency"] >= 1.5 * om["Recency"]
        new = r["Median recency"] <= 0.67 * om["Recency"]
        if old and (hi_m or hi_f):
            n = "Previously valuable, less-recently active customers"
        elif hi_m and hi_f:
            n = "High-value, frequent customers"
        elif hi_m:
            n = "High-spending customers"
        elif hi_f:
            n = "Frequent customers"
        elif old and lo_m:
            n = "Less-recent, low-spending customers"
        elif old:
            n = "Less-recent customers"
        elif new and lo_f:
            n = "Recent, low-frequency customers"
        elif lo_m:
            n = "Low-spending customers"
        else:
            n = "Regular mid-range customers"
        raw[int(r["Cluster"])] = n
    counts = pd.Series(raw).value_counts()
    return {c: (n if counts[n] == 1 else f"{n} (group {c})") for c, n in raw.items()}


def label_names(profile: pd.DataFrame) -> dict[int, str]:
    return dict(zip(profile["Cluster"].astype(int), profile["Segment"]))


def display_labels(profile: pd.DataFrame) -> dict[int, str]:
    """'Cluster 2: name' - keeps the arbitrary numeric ID next to the readable label."""
    return {c: f"Cluster {c}: {n}" for c, n in label_names(profile).items()}


def heatmap_table(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Cluster median of each feature relative to the overall median (1.0 = typical)."""
    df = features[EXTENDED_FEATURES].copy()
    df["Cluster"] = np.asarray(labels)
    med = df.groupby("Cluster").median()
    return med / df[EXTENDED_FEATURES].median().replace(0, np.nan)


def customer_vs_segment(features: pd.DataFrame, labels: np.ndarray, customer_id: str) -> pd.DataFrame:
    """One customer's original features next to the medians of their segment."""
    df = features.copy()
    df["Cluster"] = np.asarray(labels)
    row = df[df["CustomerID"] == customer_id].iloc[0]
    seg = df[df["Cluster"] == row["Cluster"]][EXTENDED_FEATURES].median()
    out = pd.DataFrame({"Customer": row[EXTENDED_FEATURES].astype(float), "Segment median": seg})
    out["Ratio to segment median"] = out["Customer"] / out["Segment median"].replace(0, np.nan)
    return out
