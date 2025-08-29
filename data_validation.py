# data_validation.py — reusable validation rules (now includes AMZ Order # in results)
from __future__ import annotations
import pandas as pd
import numpy as np

# Defaults; can be overridden via function params
DATE_HEADER = "Order Date"
REVENUE_HEADER = "Sale Price"
PURCHASE_HEADER = "Purchase Price"
UNITS_HEADER = "Units"
ASIN_HEADER = "ASIN"
ORDER_ID_HEADER = "AMZ Order #"

def build_validation_issues(
    work_df: pd.DataFrame,
    *,
    date_col: str = DATE_HEADER,
    revenue_col: str = REVENUE_HEADER,
    purchase_col: str = PURCHASE_HEADER,
    units_col: str = UNITS_HEADER,
    asin_col: str = ASIN_HEADER,
    order_col: str = ORDER_ID_HEADER,   # NEW: pass AMZ Order # column name
) -> pd.DataFrame:
    """
    Build a table of validation issues across all stores.
    Columns: Store, Row, Issue, Column, Value, AMZ Order #, Suggestion
    Row is 1-based index + 2 (to approximate spreadsheet row incl. header).
    """
    if work_df is None or work_df.empty:
        return pd.DataFrame(columns=["Store","Row","Issue","Column","Value","AMZ Order #","Suggestion"])

    df = work_df.copy()

    # Ensure required columns exist to avoid KeyError
    for c in [date_col, revenue_col, purchase_col, units_col, asin_col, order_col]:
        if c not in df.columns:
            df[c] = pd.NA
    if "Account" not in df.columns:
        df["Account"] = "Unknown"

    df["_row"] = df.index.astype(int) + 2  # header ~1, data ~2
    issues = []

    def add_issues(mask, issue, column, value_series, suggestion):
        if mask is None or not hasattr(mask, "any") or not mask.any():
            return
        tmp = df.loc[mask, ["Account", "_row", order_col]].copy()
        tmp["Issue"] = issue
        tmp["Column"] = column
        tmp["Value"] = value_series[mask].astype(str).str.slice(0, 300)
        tmp["Suggestion"] = suggestion
        tmp.rename(columns={
            "Account":"Store",
            "_row":"Row",
            order_col:"AMZ Order #",
        }, inplace=True)
        issues.append(tmp)

    # 1) Invalid / missing date
    date_series = pd.to_datetime(df[date_col], errors="coerce")
    add_issues(
        date_series.isna(),
        "Invalid or missing date",
        date_col,
        df.get(date_col, pd.Series(index=df.index, dtype="object")),
        "Fix the date format (e.g., 20-Aug-2025)."
    )

    # 2) Purchase checks
    pur_raw = df.get(purchase_col).astype(str)
    add_issues(pur_raw.str.strip().eq("").fillna(True),
               "Missing purchase price", purchase_col, pur_raw, "Enter numeric purchase.")
    pur_num = pd.to_numeric(df.get(purchase_col), errors="coerce")
    add_issues(pur_num.lt(0).fillna(False),
               "Negative purchase", purchase_col, pur_num, "Make non-negative.")
    add_issues(pur_num.gt(10000).fillna(False),
               "Unusually high purchase", purchase_col, pur_num, "Check currency/decimal; confirm cost.")

    # 3) Sale checks
    sale_raw = df.get(revenue_col).astype(str)
    add_issues(sale_raw.str.strip().eq("").fillna(True),
               "Missing sale price", revenue_col, sale_raw, "Enter numeric sale price.")
    sale_num = pd.to_numeric(df.get(revenue_col), errors="coerce")
    add_issues(sale_num.lt(0).fillna(False),
               "Negative sale price", revenue_col, sale_num, "Make non-negative.")

    # 4) Units checks
    units_num = pd.to_numeric(df.get(units_col), errors="coerce")
    add_issues(units_num.isna() | units_num.le(0),
               "Invalid units", units_col, units_num, "Use integer >= 1.")

    # 5) ASIN checks (present and len >= 9, per your rule)
    asin_series = df.get(asin_col).astype(str)
    asin_bad = asin_series.isna() | (asin_series.str.len() < 9) | asin_series.str.strip().eq("")
    add_issues(asin_bad.fillna(True),
               "Unassigned sale (ASIN missing/short)", asin_col, asin_series, "Fill ASIN (>= 9 chars).")

    # 6) Hunter mapping
    hunter_series = df.get("Hunter Name", pd.Series(index=df.index, dtype="object")).astype(str)
    add_issues(hunter_series.str.startswith("Unassigned"),
               "Unassigned hunter", "Hunter Name", hunter_series,
               "Add ASIN to Product DB (ListedItems) with Hunter Name.")

    # 7) Purchase > 2× Sale (swap/typo sanity)
    add_issues((pur_num > (2.0 * sale_num)).fillna(False),
               "Purchase exceeds 2× sale", purchase_col, pur_num,
               "Check sale/purchase values for swap/typo.")

    if not issues:
        return pd.DataFrame(columns=["Store","Row","Issue","Column","Value","AMZ Order #","Suggestion"])

    out = pd.concat(issues, ignore_index=True)
    return out.sort_values(["Store","Row","Issue"], kind="mergesort").reset_index(drop=True)
