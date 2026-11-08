"""End-to-end orchestration, saved artifacts and export tables."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .clustering import (candidate_range, fit_comparison_models, fit_kmeans, gmm_sweep, kmeans_sweep,
                         recommend_gmm_k, recommend_k, reproducible_sample)
from .dimensionality import fit_representation, transform_with_representation
from .evaluation import comparison_table
from .feature_engineering import FEATURE_SETS, build_customer_features
from .preprocessing import clean_transactions, data_quality_warnings
from .profiling import profile_segments

ARTIFACT_NAME = "segmentation_artifacts.joblib"


def prepare_customers(raw: pd.DataFrame) -> dict:
    """Raw transactions -> cleaned transactions, cleaning report, customer table."""
    clean, report, steps = clean_transactions(raw)
    customers, ref = build_customer_features(clean)
    return {"clean": clean, "report": report, "steps": steps, "customers": customers,
            "reference_date": ref, "warnings": data_quality_warnings(raw, clean, report)}


def run_sweeps(Z: np.ndarray) -> dict:
    ks = candidate_range(len(Z))
    km, gm = kmeans_sweep(Z, ks), gmm_sweep(Z, ks)
    return {"kmeans_sweep": km, "gmm_sweep": gm, "k_recommended": recommend_k(km),
            "gmm_k_bic": recommend_gmm_k(gm)}


def fit_models(customers: pd.DataFrame, rep: dict, k: int, gmm_k: int, hier_cap: int = 3000) -> dict:
    """Primary full-data K-Means plus same-customer comparison models."""
    Z = rep["Z"]
    km, km_labels = fit_kmeans(Z, k)
    cmp_ = fit_comparison_models(Z, k, gmm_k, hier_cap)
    scope = (f"Sample of {len(cmp_['indices'])} of {len(Z)} customers (all models refit on it)"
             if cmp_["sampled"] else f"All {len(Z)} customers")
    table = comparison_table(cmp_["Z"], cmp_["labels"], scope)
    feats_c = customers.iloc[cmp_["indices"]].reset_index(drop=True)
    return {
        "k": k, "gmm_k": gmm_k, "kmeans_model": km, "kmeans_labels": km_labels,
        "comparison": cmp_, "comparison_scope": scope, "comparison_table": table,
        "comparison_features": feats_c,
        "gmm_ic": pd.DataFrame([{"Model": n, "k": g["model"].n_components, "AIC": g["aic"], "BIC": g["bic"],
                                 "Converged": g["converged"]} for n, g in cmp_["gmm"].items()]),
        "profiles": {"K-Means (full data)": profile_segments(customers, km_labels),
                     **{n: profile_segments(feats_c, lab) for n, lab in cmp_["labels"].items()}},
    }


def run_pipeline(raw: pd.DataFrame, feature_set: str = "RFM", variance_threshold: float = 0.90,
                 k: int | None = None, gmm_k: int | None = None, hier_cap: int = 3000) -> dict:
    prep = prepare_customers(raw)
    rep = fit_representation(prep["customers"], FEATURE_SETS[feature_set], variance_threshold)
    sweeps = run_sweeps(rep["Z"])
    k = k or sweeps["k_recommended"]
    gmm_k = gmm_k or sweeps["gmm_k_bic"]
    models = fit_models(prep["customers"], rep, k, gmm_k, hier_cap)
    return {"feature_set": feature_set, "hier_cap": hier_cap, **prep, "rep": rep, **sweeps, **models}


# ------------------------------------------------------------- exports
def assignments_table(result: dict, include_gmm: bool = True) -> pd.DataFrame:
    """Customer features + K-Means cluster/segment (+ GMM columns for the comparison customers)."""
    df = result["customers"].copy()
    prof = result["profiles"]["K-Means (full data)"]
    df["KMeansCluster"] = result["kmeans_labels"]
    df["KMeansSegment"] = df["KMeansCluster"].map(dict(zip(prof["Cluster"], prof["Segment"])))
    if include_gmm:
        cmp_ = result["comparison"]
        name = f"GMM (k={result['k']}, same as K-Means)"
        g = cmp_["gmm"][name]
        df["InComparisonSet"] = False
        df.loc[df.index[cmp_["indices"]], "InComparisonSet"] = True
        df["GMMCluster"] = pd.array([pd.NA] * len(df), dtype="Int64")
        df["GMMMaxMembershipProbability"] = np.nan
        pos = df.index[cmp_["indices"]]
        df.loc[pos, "GMMCluster"] = g["labels"]
        df.loc[pos, "GMMMaxMembershipProbability"] = g["max_prob"]
    return df


def profiles_export(result: dict) -> pd.DataFrame:
    return pd.concat([p.assign(Model=n) for n, p in result["profiles"].items()], ignore_index=True)


def cleaning_summary_export(result: dict) -> pd.DataFrame:
    rep = pd.DataFrame({"Step": list(result["report"].keys()),
                        "Value": [str(v) for v in result["report"].values()]})
    return pd.concat([result["steps"].assign(Section="Cleaning steps"), rep.assign(Section="Summary")],
                     ignore_index=True)[["Section", "Step", "Rows removed", "Rows remaining", "Value"]]


# ----------------------------------------------------------- artifacts
def save_artifacts(result: dict, folder: str | Path) -> Path:
    """Persist fitted scaler/PCA, feature config, models, reference date and results.

    The cleaned transaction table is deliberately left out (large, not needed to display results).
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rep = result["rep"]
    bundle = {
        "scaler": rep["scaler"], "pca": rep["pca"], "n_components": rep["n_components"],
        "feature_config": {"feature_set": result["feature_set"], "feature_cols": rep["feature_cols"],
                           "variance_threshold": rep["variance_threshold"]},
        "kmeans_model": result["kmeans_model"],
        "gmm_models": {n: g["model"] for n, g in result["comparison"]["gmm"].items()},
        "reference_date": result["reference_date"],
        "result": {k: v for k, v in result.items() if k != "clean"},
    }
    path = folder / ARTIFACT_NAME
    joblib.dump(bundle, path)
    return path


def load_artifacts(folder: str | Path) -> dict:
    return joblib.load(Path(folder) / ARTIFACT_NAME)


def score_with_artifacts(bundle: dict, customers: pd.DataFrame) -> np.ndarray:
    """Assign customers to saved K-Means clusters using the saved (not refitted) transforms."""
    rep = {"feature_cols": bundle["feature_config"]["feature_cols"], "scaler": bundle["scaler"],
           "pca": bundle["pca"], "n_components": bundle["n_components"]}
    return bundle["kmeans_model"].predict(transform_with_representation(rep, customers))
