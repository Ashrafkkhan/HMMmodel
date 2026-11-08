"""Reusable transaction cleaning with a sequential cleaning report."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import REQUIRED_COLUMNS, validate_schema


def _normalise_customer_id(s: pd.Series) -> pd.Series:
    """Return customer IDs as text identifiers (17850.0 -> '17850'); blanks -> NA."""
    num = pd.to_numeric(s, errors="coerce")
    out = s.astype("string").str.strip()
    out = out.mask(out == "")
    is_int = num.notna() & (num == num.round())
    return out.where(~is_int, num.round().astype("Int64").astype("string"))


def clean_transactions(df: pd.DataFrame):
    """Clean raw transactions.

    Returns (clean_df, report, steps) where `steps` is a DataFrame of sequential
    removal counts (a row removed by one step is never counted again) and
    `report` is a dict of headline numbers.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    validate_schema(df)
    df = df[REQUIRED_COLUMNS]
    original_rows = len(df)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["CustomerID"] = _normalise_customer_id(df["CustomerID"])
    df["InvoiceNo"] = df["InvoiceNo"].astype("string").str.strip()
    df["StockCode"] = df["StockCode"].astype("string").str.strip()

    steps: list[dict] = []
    cur = df

    def drop(keep: pd.Series, label: str) -> pd.DataFrame:
        keep = keep.fillna(False).astype(bool)
        removed = int((~keep).sum())
        out = cur[keep]
        steps.append({"Step": label, "Rows removed": removed, "Rows remaining": len(out)})
        return out

    cur = drop(cur["CustomerID"].notna(), "Missing customer ID")
    cur = drop(cur["InvoiceNo"].notna() & (cur["InvoiceNo"] != ""), "Missing invoice number")
    cur = drop(cur["InvoiceDate"].notna(), "Invalid date")
    cur = drop(
        np.isfinite(cur["Quantity"].astype(float)) & np.isfinite(cur["UnitPrice"].astype(float)),
        "Invalid quantity or price (non-numeric / non-finite)",
    )
    cur = drop(~cur["InvoiceNo"].str.upper().str.startswith("C"), "Cancelled invoices (InvoiceNo starts with 'C')")
    cur = drop((cur["Quantity"] > 0) & (cur["UnitPrice"] > 0), "Non-positive quantity or unit price")
    cur = drop(~cur.duplicated(keep="first"), "Exact duplicate rows")

    clean = cur.copy()
    clean["Quantity"] = clean["Quantity"].astype(float)
    clean["UnitPrice"] = clean["UnitPrice"].astype(float)
    clean["LineRevenue"] = clean["Quantity"] * clean["UnitPrice"]
    clean = clean.reset_index(drop=True)

    if clean.empty:
        raise ValueError("No valid purchase rows remain after cleaning; check the input data.")

    report = {
        "Original rows": original_rows,
        "Final rows": len(clean),
        "Removed rows": original_rows - len(clean),
        "Customers": int(clean["CustomerID"].nunique()),
        "Distinct invoices": int(clean["InvoiceNo"].nunique()),
        "First purchase": clean["InvoiceDate"].min(),
        "Last purchase": clean["InvoiceDate"].max(),
        "Total purchase revenue": float(clean["LineRevenue"].sum()),
    }
    return clean, report, pd.DataFrame(steps)


def data_quality_warnings(raw: pd.DataFrame | None, clean: pd.DataFrame, report: dict) -> list[str]:
    """Plain-language warnings about the dataset and its limits."""
    w = [
        "Revenue here is purchase revenue: cancelled invoices and returns are excluded, "
        "so it is NOT net revenue after refunds and this is not a profitability analysis."
    ]
    if report["Original rows"]:
        pct = 100 * report["Removed rows"] / report["Original rows"]
        if pct > 30:
            w.append(f"{pct:.1f}% of rows were removed during cleaning; check that the input is as expected.")
    if raw is not None and "CustomerID" in raw:
        miss = raw["CustomerID"].isna().mean() * 100
        if miss > 5:
            w.append(f"{miss:.1f}% of raw rows have no CustomerID (e.g. guest checkouts); they cannot be segmented.")
    if report["Customers"] < 30:
        w.append(f"Only {report['Customers']} customers remain; clustering results will be unstable.")
    per_cust = clean.groupby("CustomerID")["InvoiceNo"].nunique()
    one_time = (per_cust == 1).mean() * 100
    if one_time > 30:
        w.append(f"{one_time:.1f}% of customers made a single purchase, so frequency has limited spread.")
    w.append("The data covers a limited period; 'recency' is relative to the last date in the dataset, not today.")
    return w
