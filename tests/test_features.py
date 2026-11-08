import numpy as np
import pandas as pd

from src.dimensionality import fit_representation, select_feature_matrix
from src.feature_engineering import EXTENDED_FEATURES, RFM_FEATURES, build_customer_features
from src.preprocessing import clean_transactions
import pytest


def tiny():
    d = pd.DataFrame({
        "InvoiceNo": ["1", "1", "2", "3"], "StockCode": ["A", "B", "A", "C"], "Description": "x",
        "Quantity": [2, 1, 3, 5], "InvoiceDate": pd.to_datetime(["2011-01-01 09:00", "2011-01-01 09:00",
                                                                 "2011-01-05 23:30", "2011-01-10 08:00"]),
        "UnitPrice": [1.0, 2.0, 4.0, 1.0], "CustomerID": ["a", "a", "a", "b"], "Country": "UK"})
    d["LineRevenue"] = d["Quantity"] * d["UnitPrice"]
    return d


def test_frequency_counts_invoices_not_lines():
    cust, ref = build_customer_features(tiny())
    a = cust.set_index("CustomerID").loc["a"]
    assert a["Frequency"] == 2 and a["Monetary"] == 16.0 and a["TotalQuantity"] == 6
    assert a["UniqueProducts"] == 2 and a["AverageOrderValue"] == 8.0


def test_reference_date_and_recency_are_dataset_relative():
    cust, ref = build_customer_features(tiny())
    assert ref == pd.Timestamp("2011-01-11")
    c = cust.set_index("CustomerID")
    assert c.loc["b", "Recency"] == 1 and c.loc["a", "Recency"] == 6


def test_one_row_per_customer(raw):
    clean, _, _ = clean_transactions(raw)
    cust, _ = build_customer_features(clean)
    assert cust["CustomerID"].is_unique and len(cust) == clean["CustomerID"].nunique()


def test_identifiers_excluded_and_finite(raw):
    clean, _, _ = clean_transactions(raw)
    cust, _ = build_customer_features(clean)
    with pytest.raises(ValueError):
        select_feature_matrix(cust, ["CustomerID", "Recency"])
    for cols in (RFM_FEATURES, EXTENDED_FEATURES):
        rep = fit_representation(cust, cols, 0.9)
        assert np.isfinite(rep["Z"]).all()
        assert "CustomerID" not in rep["feature_cols"]
