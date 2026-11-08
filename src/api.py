"""Turn a pipeline result into the JSON payload the web dashboard renders."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

from . import pipeline as pl
from .clustering import reproducible_sample, ward_linkage_matrix
from .evaluation import usefulness_checks
from .feature_engineering import EXTENDED_FEATURES, FEATURE_DEFINITIONS
from .profiling import heatmap_table

KM_KEY = "K-Means (full data)"


def to_jsonable(o):
    """Recursively convert numpy/pandas values; NaN/inf become null."""
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, pd.DataFrame):
        return to_jsonable(o.to_dict("records"))
    if isinstance(o, pd.Series):
        return to_jsonable(o.tolist())
    if isinstance(o, (pd.Timestamp,)):
        return o.strftime("%Y-%m-%d %H:%M")
    if o is pd.NA or o is pd.NaT or o is None:
        return None
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    return str(o)


def _box_stats(values: pd.Series, labels: np.ndarray) -> list[dict]:
    out = []
    for c in sorted(np.unique(labels)):
        v = values[labels == c].to_numpy(float)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        iqr = q3 - q1
        lo = v[v >= q1 - 1.5 * iqr].min()
        hi = v[v <= q3 + 1.5 * iqr].max()
        out.append({"cluster": int(c), "lo": lo, "q1": q1, "med": med, "q3": q3, "hi": hi,
                    "outliers": int(((v < lo) | (v > hi)).sum()), "n": len(v)})
    return out


def build_payload(result: dict, raw_preview: pd.DataFrame | None = None, dend_n: int = 300) -> dict:
    rep, cmp_ = result["rep"], result["comparison"]
    customers = result["customers"]

    ass = pl.assignments_table(result)
    cust = {c: ass[c].tolist() for c in ["CustomerID"] + EXTENDED_FEATURES}
    cust["KMeansCluster"] = result["kmeans_labels"].tolist()
    cust["PC1"], cust["PC2"] = rep["Z_viz"][:, 0].tolist(), rep["Z_viz"][:, 1].tolist()
    cust["GMMCluster"] = ass["GMMCluster"].astype(object).where(ass["GMMCluster"].notna(), None).tolist()
    cust["GMMProb"] = ass["GMMMaxMembershipProbability"].tolist()

    models = {}
    sources = {KM_KEY: (customers, result["kmeans_labels"], None)}
    for name, lab in cmp_["labels"].items():
        sources[name] = (result["comparison_features"], lab, cmp_["indices"])
    for name, (feats, labels, idx) in sources.items():
        prof = result["profiles"][name]
        models[name] = {
            "indices": None if idx is None else idx.tolist(),
            "labels": np.asarray(labels).tolist(),
            "profile": prof,
            "heatmap": {"features": EXTENDED_FEATURES, "rows": heatmap_table(feats, labels).to_dict("index")},
            "box": _box_stats(feats["Monetary"], np.asarray(labels)),
            "checks": usefulness_checks(prof, feats),
        }

    n_d = min(dend_n, len(cmp_["Z"]))
    sel = reproducible_sample(len(cmp_["Z"]), n_d)
    d = dendrogram(ward_linkage_matrix(cmp_["Z"][sel]), no_plot=True)

    rep_summary = {k: (v if not isinstance(v, pd.Timestamp) else v) for k, v in result["report"].items()}
    return to_jsonable({
        "meta": {
            "feature_set": result["feature_set"], "features": rep["feature_cols"],
            "variance_threshold": rep["variance_threshold"], "n_components": rep["n_components"],
            "k": result["k"], "gmm_k": result["gmm_k"], "k_recommended": result["k_recommended"],
            "gmm_k_bic": result["gmm_k_bic"], "ks": result["kmeans_sweep"]["k"].tolist(),
            "hier_cap": result["hier_cap"], "scope": result["comparison_scope"],
            "sampled": cmp_["sampled"], "reference_date": result["reference_date"],
            "n_customers": len(customers), "dend_n": n_d,
        },
        "cleaning": {"report": rep_summary, "steps": result["steps"], "warnings": result["warnings"]},
        "definitions": FEATURE_DEFINITIONS,
        "raw_preview": raw_preview,
        "customers": cust,
        "pca": {"explained": rep["explained"], "features": rep["feature_cols"],
                "pcs": rep["loadings"].columns.tolist(), "loadings": rep["loadings"].to_numpy()},
        "sweeps": {"kmeans": result["kmeans_sweep"], "gmm": result["gmm_sweep"]},
        "models": models,
        "comparison": {"table": result["comparison_table"], "gmm_ic": result["gmm_ic"]},
        "dendrogram": {"icoord": d["icoord"], "dcoord": d["dcoord"], "n": n_d},
    })
