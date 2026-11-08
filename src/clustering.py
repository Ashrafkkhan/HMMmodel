"""K-Means (primary), Gaussian mixture and Ward hierarchical clustering (comparison)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture

from .evaluation import SEED, cluster_sizes, compute_metrics


def candidate_range(n_customers: int, lo: int = 2, hi: int = 8) -> list[int]:
    """Candidate cluster counts, limited by how many customers exist."""
    return list(range(lo, min(hi, n_customers - 1) + 1))


# ---------------------------------------------------------------- K-Means
def fit_kmeans(Z: np.ndarray, k: int):
    model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Z)
    return model, model.labels_


def kmeans_sweep(Z: np.ndarray, ks: list[int]) -> pd.DataFrame:
    rows = []
    for k in ks:
        model, labels = fit_kmeans(Z, k)
        m = compute_metrics(Z, labels)
        sizes = cluster_sizes(labels)
        rows.append({
            "k": k, "Inertia": float(model.inertia_), "Silhouette": m["Silhouette"],
            "Davies-Bouldin": m["Davies-Bouldin"], "Calinski-Harabasz": m["Calinski-Harabasz"],
            "Smallest cluster %": 100 * min(sizes.values()) / len(labels),
            "Cluster sizes": ", ".join(str(sizes[c]) for c in sorted(sizes)),
        })
    return pd.DataFrame(rows)


def recommend_k(sweep: pd.DataFrame) -> int:
    """Highest valid silhouette; falls back to the smallest candidate if none is valid."""
    valid = sweep.dropna(subset=["Silhouette"])
    if valid.empty:
        return int(sweep["k"].min())
    return int(valid.loc[valid["Silhouette"].idxmax(), "k"])


# -------------------------------------------------------------------- GMM
def fit_gmm(Z: np.ndarray, k: int) -> dict:
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=5, reg_covar=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gmm.fit(Z)
    probs = gmm.predict_proba(Z)
    return {
        "model": gmm, "labels": probs.argmax(axis=1), "probs": probs,
        "max_prob": probs.max(axis=1), "converged": bool(gmm.converged_),
        "aic": float(gmm.aic(Z)), "bic": float(gmm.bic(Z)),
    }


def gmm_sweep(Z: np.ndarray, ks: list[int]) -> pd.DataFrame:
    rows = []
    for k in ks:
        r = fit_gmm(Z, k)
        rows.append({"k": k, "AIC": r["aic"], "BIC": r["bic"], "Converged": r["converged"]})
    return pd.DataFrame(rows)


def recommend_gmm_k(sweep: pd.DataFrame) -> int:
    return int(sweep.loc[sweep["BIC"].idxmin(), "k"])


# ----------------------------------------------------------- Hierarchical
def fit_hierarchical(Z: np.ndarray, k: int) -> np.ndarray:
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)


def reproducible_sample(n: int, size: int, seed: int = SEED) -> np.ndarray:
    """Sorted positional indices of a reproducible sample (all rows if n <= size)."""
    if n <= size:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=size, replace=False))


def ward_linkage_matrix(Z: np.ndarray) -> np.ndarray:
    return linkage(Z, method="ward")


def fit_comparison_models(Z: np.ndarray, k: int, gmm_k: int, cap: int) -> dict:
    """Fit K-Means, Ward and GMM(s) on the *same* customers.

    If more customers than `cap` exist, all models are refit on one reproducible
    sample; the returned `indices` say exactly which customers were used.
    """
    idx = reproducible_sample(len(Z), cap)
    Zc = Z[idx]
    _, km_labels = fit_kmeans(Zc, k)
    g_same = fit_gmm(Zc, k)
    g_bic = g_same if gmm_k == k else fit_gmm(Zc, gmm_k)
    return {
        "indices": idx, "Z": Zc, "sampled": len(idx) < len(Z),
        "labels": {
            f"K-Means (k={k})": km_labels,
            f"Hierarchical Ward (k={k})": fit_hierarchical(Zc, k),
            f"GMM (k={k}, same as K-Means)": g_same["labels"],
            f"GMM (k={gmm_k}, BIC-selected)": g_bic["labels"],
        },
        "gmm": {f"GMM (k={k}, same as K-Means)": g_same, f"GMM (k={gmm_k}, BIC-selected)": g_bic},
    }
