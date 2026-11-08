"""log1p -> StandardScaler -> PCA, kept together so fitted transforms can be reused."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .feature_engineering import EXTENDED_FEATURES

IDENTIFIER_COLUMNS = {"CustomerID", "InvoiceNo", "StockCode"}


def select_feature_matrix(customers: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    bad = IDENTIFIER_COLUMNS.intersection(feature_cols)
    if bad:
        raise ValueError(f"Identifier columns cannot be model features: {sorted(bad)}")
    unknown = [c for c in feature_cols if c not in EXTENDED_FEATURES]
    if unknown:
        raise ValueError(f"Unknown feature columns: {unknown}")
    return customers[list(feature_cols)].astype(float)


def log_transform(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if (arr < 0).any():
        raise ValueError("log1p requires non-negative features.")
    return np.log1p(arr)


def assert_finite(arr: np.ndarray, what: str = "model input") -> None:
    if not np.isfinite(arr).all():
        raise ValueError(f"{what} contains missing or infinite values.")


def fit_representation(
    customers: pd.DataFrame,
    feature_cols: list[str],
    variance_threshold: float = 0.90,
) -> dict:
    """Fit log1p + StandardScaler + PCA and pick the smallest component count
    reaching `variance_threshold` of the standardized variance."""
    if not 0 < variance_threshold <= 1:
        raise ValueError("variance_threshold must be in (0, 1].")
    if len(customers) < 3:
        raise ValueError("At least 3 customers are needed for clustering.")
    X = select_feature_matrix(customers, feature_cols)
    X_log = log_transform(X)
    scaler = StandardScaler().fit(X_log)
    X_std = scaler.transform(X_log)
    assert_finite(X_std, "standardized features")

    pca = PCA(svd_solver="full", random_state=42).fit(X_std)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n_comp = int(min(np.searchsorted(cum, variance_threshold - 1e-12) + 1, len(cum)))

    Z_all = pca.transform(X_std)
    Z = Z_all[:, :n_comp]
    assert_finite(Z, "PCA model input")
    pcs = [f"PC{i + 1}" for i in range(len(cum))]
    return {
        "feature_cols": list(feature_cols),
        "variance_threshold": float(variance_threshold),
        "scaler": scaler,
        "pca": pca,
        "n_components": n_comp,
        "Z": Z,                    # model input (all models cluster on this)
        "Z_viz": Z_all[:, :2],     # first two PCs, for plotting only
        "explained": pd.DataFrame(
            {"Component": pcs, "ExplainedVariance": pca.explained_variance_ratio_, "Cumulative": cum}
        ),
        "loadings": pd.DataFrame(pca.components_.T, index=list(feature_cols), columns=pcs),
    }


def transform_with_representation(rep: dict, customers: pd.DataFrame) -> np.ndarray:
    """Apply the already-fitted scaler/PCA (no refitting) and keep the selected components."""
    X = select_feature_matrix(customers, rep["feature_cols"])
    X_std = rep["scaler"].transform(log_transform(X))
    return rep["pca"].transform(X_std)[:, : rep["n_components"]]
