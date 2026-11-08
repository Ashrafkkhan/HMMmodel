"""Load and validate the UCI Online Retail transaction file."""
from __future__ import annotations

import io
import os
from typing import BinaryIO, Union

import pandas as pd

REQUIRED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

# Identifiers are read as text so they are never treated as measurements.
_ID_DTYPES = {"InvoiceNo": str, "StockCode": str}


class SchemaError(ValueError):
    """Raised when the input file does not have the expected columns."""


def validate_schema(df: pd.DataFrame) -> None:
    """Raise SchemaError listing every missing required column."""
    present = {str(c).strip() for c in df.columns}
    missing = [c for c in REQUIRED_COLUMNS if c not in present]
    if missing:
        raise SchemaError(
            "The dataset is missing required column(s): "
            + ", ".join(missing)
            + ". Columns found: "
            + (", ".join(map(str, df.columns)) or "(none)")
            + "."
        )


def load_with_cache(path: str | os.PathLike, cache_dir: str | os.PathLike) -> pd.DataFrame:
    """load_dataset for a path, with a pickle cache (Excel parsing is slow: minutes for the full file).

    The cache is reused only if it is newer than the source file.
    """
    path = os.fspath(path)
    cache = os.path.join(os.fspath(cache_dir), os.path.basename(path) + ".pkl")
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        return pd.read_pickle(cache)
    df = load_dataset(path)
    os.makedirs(os.fspath(cache_dir), exist_ok=True)
    df.to_pickle(cache)
    return df


def load_dataset(source: Union[str, os.PathLike, BinaryIO], filename: str | None = None) -> pd.DataFrame:
    """Read an Excel (.xlsx/.xls) or CSV file and validate its schema.

    `source` may be a path or a file-like object (e.g. a Streamlit upload); for
    file-like objects pass `filename` so the format can be detected.
    """
    if hasattr(source, "read"):
        name = filename or getattr(source, "name", "")
        data = source.read()
    else:
        name = filename or str(source)
        with open(source, "rb") as fh:
            data = fh.read()

    ext = os.path.splitext(str(name))[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(io.BytesIO(data), engine="openpyxl", dtype=_ID_DTYPES)
    elif ext in (".csv", ".txt"):
        try:
            df = pd.read_csv(io.BytesIO(data), dtype=_ID_DTYPES, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(data), dtype=_ID_DTYPES, encoding="latin-1")
    else:
        raise SchemaError(f"Unsupported file type '{ext}'. Upload an .xlsx or .csv file.")

    df.columns = [str(c).strip() for c in df.columns]
    validate_schema(df)
    return df
