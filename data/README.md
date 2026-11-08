# Data

Place the UCI Online Retail file here as `Online Retail.xlsx` (or a CSV named `Online Retail.csv`):
https://archive.ics.uci.edu/dataset/352/online+retail

Required columns: `InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country`.
You can also upload an .xlsx/.csv with the same columns from the dashboard sidebar.

The first Excel read takes a few minutes; the parsed table is then cached as a pickle in `artifacts/`.
