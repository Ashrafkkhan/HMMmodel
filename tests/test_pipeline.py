import numpy as np

from src import pipeline as pl
from src.clustering import reproducible_sample
from src.dimensionality import transform_with_representation


def test_pca_retention_and_viz_distinct(raw):
    r = pl.run_pipeline(raw, "Extended", 0.90)
    rep = r["rep"]
    assert rep["explained"]["Cumulative"].iloc[rep["n_components"] - 1] >= 0.90 - 1e-9
    assert rep["Z_viz"].shape[1] == 2 and rep["Z"].shape[1] == rep["n_components"]


def test_all_models_same_customers_and_profiles_use_original_values(raw):
    r = pl.run_pipeline(raw, "RFM", 0.9, hier_cap=60)
    cmp_ = r["comparison"]
    assert cmp_["sampled"] and len(cmp_["indices"]) == 60
    assert {len(v) for v in cmp_["labels"].values()} == {60}
    prof = r["profiles"]["K-Means (full data)"]
    assert prof["Customers"].sum() == len(r["customers"])
    assert np.isclose(prof["Total revenue"].sum(), r["report"]["Total purchase revenue"])
    assert np.isclose(prof["Revenue %"].sum(), 100)
    assert "Accuracy" not in " ".join(r["comparison_table"].columns)


def test_sample_reproducible():
    assert (reproducible_sample(1000, 100) == reproducible_sample(1000, 100)).all()


def test_k_override_and_exports(raw):
    r = pl.run_pipeline(raw, "RFM", 0.9, k=3)
    assert r["k"] == 3 and len(set(r["kmeans_labels"])) == 3
    a = pl.assignments_table(r)
    assert len(a) == len(r["customers"]) and a["CustomerID"].is_unique
    assert a["GMMMaxMembershipProbability"].dropna().between(0, 1).all()


def test_artifacts_roundtrip_reuses_fitted_transforms(raw, tmp_path):
    r = pl.run_pipeline(raw, "RFM", 0.9)
    pl.save_artifacts(r, tmp_path)
    b = pl.load_artifacts(tmp_path)
    assert b["reference_date"] == r["reference_date"] and "feature_config" in b
    rep = {"feature_cols": b["feature_config"]["feature_cols"], "scaler": b["scaler"], "pca": b["pca"],
           "n_components": b["n_components"]}
    assert np.allclose(transform_with_representation(rep, r["customers"]), r["rep"]["Z"])
    assert (pl.score_with_artifacts(b, r["customers"]) == r["kmeans_labels"]).all()


def test_web_payload_is_strict_json(raw):
    import json
    from src.api import build_payload
    r = pl.run_pipeline(raw, "RFM", 0.9, hier_cap=60)
    p = build_payload(r, raw.head(5))
    json.dumps(p, allow_nan=False)  # no NaN/inf, all values serialisable
    km = p["models"]["K-Means (full data)"]
    assert len(km["labels"]) == p["meta"]["n_customers"]
    assert set(km["heatmap"]["rows"]) == {str(c) for c in set(km["labels"])}
    assert len(p["customers"]["CustomerID"]) == p["meta"]["n_customers"]
