import pytest

from src.data_loader import SchemaError, validate_schema
from src.preprocessing import clean_transactions


def test_missing_columns_readable_error(raw):
    with pytest.raises(SchemaError) as e:
        validate_schema(raw.drop(columns=["Quantity", "Country"]))
    assert "Quantity" in str(e.value) and "Country" in str(e.value)


def test_invalid_rows_removed_and_counts_sequential(raw):
    clean, report, steps = clean_transactions(raw)
    assert (clean["Quantity"] > 0).all() and (clean["UnitPrice"] > 0).all()
    assert not clean["InvoiceNo"].str.upper().str.startswith("C").any()
    assert clean["CustomerID"].notna().all()
    assert not clean.duplicated(subset=["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"]).any()
    assert steps["Rows removed"].sum() == report["Removed rows"] == report["Original rows"] - report["Final rows"]
    removed = dict(zip(steps["Step"], steps["Rows removed"]))
    assert removed["Missing customer ID"] == 1
    assert removed["Invalid date"] == 1
    assert sum(v for s, v in removed.items() if s.startswith("Cancelled")) == 2  # C and c
    assert removed["Exact duplicate rows"] == 1


def test_line_revenue(raw):
    clean, _, _ = clean_transactions(raw)
    assert (clean["LineRevenue"] == clean["Quantity"] * clean["UnitPrice"]).all()
    assert clean["CustomerID"].str.fullmatch(r"\d+").all()  # identifiers as text, not 10000.0
