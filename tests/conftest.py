import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_raw(n_customers=120, seed=0):
    """Synthetic transactions with three behavioural groups plus deliberately bad rows."""
    rng = np.random.default_rng(seed)
    rows, inv = [], 1000
    for c in range(n_customers):
        grp = c % 3
        n_inv = [1, 4, 12][grp]
        scale = [5, 15, 40][grp]
        for _ in range(n_inv):
            inv += 1
            date = pd.Timestamp("2011-01-01") + pd.Timedelta(days=int(rng.integers(0, [300, 200, 360][grp])),
                                                              hours=int(rng.integers(8, 18)))
            for p in rng.choice(50, size=int(rng.integers(1, 4)), replace=False):
                rows.append([str(inv), f"P{p}", "item", int(rng.integers(1, 10)), date,
                             float(round(rng.uniform(1, 3) * scale / 5, 2)), 10000.0 + c, "UK"])
    df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
                                     "UnitPrice", "CustomerID", "Country"])
    bad = pd.DataFrame([
        ["C9001", "P1", "x", -1, pd.Timestamp("2011-05-01"), 2.0, 10000.0, "UK"],      # cancelled
        ["9002", "P1", "x", -2, pd.Timestamp("2011-05-01"), 2.0, 10000.0, "UK"],        # negative qty
        ["9003", "P1", "x", 2, pd.Timestamp("2011-05-01"), 0.0, 10000.0, "UK"],         # zero price
        ["9004", "P1", "x", 2, pd.Timestamp("2011-05-01"), 2.0, np.nan, "UK"],          # no customer
        ["c9005", "P1", "x", 1, pd.Timestamp("2011-05-01"), 2.0, 10001.0, "UK"],        # lowercase c
        ["9006", "P1", "x", 1, "not a date", 2.0, 10001.0, "UK"],                       # invalid date
    ], columns=df.columns)
    dup = df.iloc[[0]]
    return pd.concat([df, bad, dup], ignore_index=True)


@pytest.fixture(scope="session")
def raw():
    return make_raw()
