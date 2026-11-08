"""Internal clustering metrics. There are no ground-truth labels, so no accuracy/F1."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

SEED = 42
SILHOUETTE_SAMPLE = 5000


def compute_metrics(Z: np.ndarray, labels: np.ndarray, seed: int = SEED) -> dict:
    """Silhouette / Davies-Bouldin / Calinski-Harabasz in the model feature space.

    Invalid cases (fewer than 2 or n or more clusters) give NaN plus a note.
    """
    labels = np.asarray(labels)
    n_clusters = len(np.unique(labels))
    out = {"Clusters found": n_clusters, "Silhouette": np.nan, "Davies-Bouldin": np.nan,
           "Calinski-Harabasz": np.nan, "Note": ""}
    if n_clusters < 2:
        out["Note"] = "Metrics skipped: fewer than 2 clusters were produced."
        return out
    if n_clusters >= len(labels):
        out["Note"] = "Metrics skipped: every customer is in its own cluster."
        return out
    sample = SILHOUETTE_SAMPLE if len(labels) > SILHOUETTE_SAMPLE else None
    out["Silhouette"] = float(silhouette_score(Z, labels, sample_size=sample, random_state=seed))
    out["Davies-Bouldin"] = float(davies_bouldin_score(Z, labels))
    out["Calinski-Harabasz"] = float(calinski_harabasz_score(Z, labels))
    if sample:
        out["Note"] = f"Silhouette estimated on a fixed-seed sample of {sample}."
    return out


def cluster_sizes(labels: np.ndarray) -> dict:
    vals, counts = np.unique(labels, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def comparison_table(Z: np.ndarray, models: dict[str, np.ndarray], scope: str, seed: int = SEED) -> pd.DataFrame:
    """Metrics for several label sets computed on the same customers and features."""
    rows = []
    for name, labels in models.items():
        m = compute_metrics(Z, labels, seed)
        rows.append({"Model": name, "Scope": scope, "Customers evaluated": len(labels), **m})
    cols = ["Model", "Scope", "Customers evaluated", "Clusters found", "Silhouette",
            "Davies-Bouldin", "Calinski-Harabasz", "Note"]
    return pd.DataFrame(rows)[cols]


def usefulness_checks(profile: pd.DataFrame, features: pd.DataFrame) -> list[str]:
    """Practical-usefulness warnings for one segmentation."""
    msgs = []
    if profile["Customer %"].min() < 2:
        msgs.append("At least one cluster holds under 2% of customers; it may be a small outlier group.")
    if profile["Customer %"].max() > 60:
        msgs.append("One cluster holds over 60% of customers; the segmentation is highly unbalanced.")
    med = profile["Median monetary"]
    if len(profile) > 1 and med.min() > 0 and med.max() / med.min() < 1.5:
        msgs.append("Median spend barely differs between clusters; profiles may not be meaningfully different.")
    top = features["Monetary"].sort_values(ascending=False)
    top_share = top.head(max(1, int(round(len(top) * 0.01)))).sum() / top.sum() * 100
    if top_share > 20:
        msgs.append(f"The top 1% of customers generate {top_share:.1f}% of revenue; unusual customers may dominate some clusters.")
    if not msgs:
        msgs.append("No obvious balance or separation problems detected (still check that segments are explainable).")
    return msgs
