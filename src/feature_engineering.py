"""Transaction-level data -> one row per customer."""
from __future__ import annotations

import pandas as pd

RFM_FEATURES = ["Recency", "Frequency", "Monetary"]
EXTENDED_FEATURES = RFM_FEATURES + ["TotalQuantity", "UniqueProducts", "AverageOrderValue"]
FEATURE_SETS = {"RFM": RFM_FEATURES, "Extended": EXTENDED_FEATURES}

FEATURE_DEFINITIONS = {
    "Recency": "Days between the reference date and the customer's latest purchase (lower = more recent)",
    "Frequency": "Number of distinct invoices for the customer",
    "Monetary": "Sum of retained transaction-line revenue (Quantity x UnitPrice)",
    "TotalQuantity": "Sum of purchased quantities",
    "UniqueProducts": "Number of distinct product codes purchased",
    "AverageOrderValue": "Monetary divided by Frequency",
}


def reference_date_for(clean_df: pd.DataFrame) -> pd.Timestamp:
    """Dataset-relative reference date (never today's date)."""
    return clean_df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)


def build_customer_features(clean_df: pd.DataFrame, reference_date: pd.Timestamp | None = None):
    """Return (customers, reference_date). CustomerID is kept as a column for joins only."""
    ref = reference_date if reference_date is not None else reference_date_for(clean_df)
    g = clean_df.groupby("CustomerID", sort=True)
    last = g["InvoiceDate"].max().dt.normalize()
    out = pd.DataFrame(
        {
            "Recency": (ref - last).dt.days,
            "Frequency": g["InvoiceNo"].nunique(),
            "Monetary": g["LineRevenue"].sum(),
            "TotalQuantity": g["Quantity"].sum(),
            "UniqueProducts": g["StockCode"].nunique(),
        }
    )
    out["AverageOrderValue"] = out["Monetary"] / out["Frequency"]
    out = out.reset_index()
    return out[["CustomerID"] + EXTENDED_FEATURES], ref
