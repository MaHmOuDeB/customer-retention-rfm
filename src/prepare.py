"""Convert the UCI Online Retail II workbook (two sheets) into one Parquet file. Run once; slow (about 2-4 minutes)."""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
sheets = pd.read_excel(DATA / "online_retail_II.xlsx", sheet_name=None, dtype={"Invoice": str, "StockCode": str})
df = pd.concat(sheets.values(), ignore_index=True)
df.columns = ["invoice", "stock_code", "description", "quantity", "invoice_ts", "unit_price", "customer_id", "country"]
df["description"] = df["description"].astype("string")      # a few descriptions are numbers in the workbook
df["stock_code"] = df["stock_code"].astype("string")
df.to_parquet(DATA / "online_retail_ii.parquet", index=False)
print(f"{len(df):,} rows, {df.invoice_ts.min()} to {df.invoice_ts.max()}, sheets: {list(sheets)}")
