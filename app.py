# app.py — Staff Progress (profit-only) + Validation with logging + robust typo detection
from __future__ import annotations

import os, io, re, time, datetime, unicodedata, zipfile
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, quote

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ===================== CONFIG / CONSTANTS =====================
APP_TITLE = "Staff Progress"

# Read-only sources (masked in UI)
DEFAULT_SHEET_ID = "1pIo7rvtYBJhnFW0sSrPorF-O-dHDEYL1Q0B_HsNh8r8"
DEFAULT_PRODUCT_PUBHTML = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vSQ0NTS2w_jkBtUy-"
    "h5W7Ml60fTd1H_tlblwCCt1-0MGLxRos5Y2HxrlRK5ONmpitIcxeznuwP67Y8d/"
    "pubhtml#gid=959982031"
)
FALLBACK_PRODUCT_SHEET_ID = DEFAULT_SHEET_ID
FALLBACK_PRODUCT_TAB_NAME = "ListedItems"  # renamed (no space)

# Canonical headers in Sales
DATE_HEADER = "Order Date"
REVENUE_HEADER = "Sale Price"
PURCHASE_HEADER = "Purchase Price"
UNITS_HEADER = "Units"
ASIN_HEADER = "ASIN"
ORDER_ID_HEADER = "AMZ Order #"

# Keep RAW text for validation (before coercion)
RAW_SALE_COL = "__raw_sale__"
RAW_PURCHASE_COL = "__raw_purchase__"

# Validation log (CSV persisted with app — may reset on rebuilds of free hosts)
VALIDATION_LOG_PATH = "validation_log.csv"

# Profit rule: reduce margin by 10% via purchase uplift
# Profit = 0.85 * Sales − 1.10 * Purchase
def adjusted_profit(sales, purchase):
    return 0.85 * sales - 1.06 * purchase


# ===================== LOW-LEVEL HELPERS =====================
_WS_RE = re.compile(r"\s+")
_GVIZ_CSV_TMPL = "https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet}"

def _normspace(s: str) -> str:
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", str(s or "")).strip())

def _normalize_columns_inplace(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    df.rename(
        columns={c: " ".join(str(c).replace("\n", " ").strip().split()) for c in df.columns},
        inplace=True,
    )

def _coerce_money(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace("[^0-9.\\-]", "", regex=True)
         .replace({"": np.nan, "-": np.nan})
         .astype(float)
         .fillna(0.0)
    )

def _ensure_date_column(df: pd.DataFrame, target_col: str = DATE_HEADER) -> pd.DataFrame:
    if df is None or df.empty:
        df = pd.DataFrame(columns=[target_col])
        df[target_col] = pd.NaT
        return df

    cols = list(df.columns)
    lower = [str(c).strip().lower() for c in cols]
    candidates: List[str] = []
    if target_col in df.columns:
        candidates.append(target_col)
    alt = {
        "order date",
        "order  date",
        "order_date",
        "orderdate",
        "fulfillment date",
        "order fullfill date",
        "date",
    }
    for c, n in zip(cols, lower):
        if c == target_col:
            continue
        if any(k in n for k in alt) or "date" in n:
            candidates.append(c)
    if not candidates and cols:
        candidates = [cols[0]]

    def try_parse(series: pd.Series) -> pd.Series:
        s = (
            series.astype(str)
            .str.replace("\u200b", "")
            .str.replace("\u200c", "")
            .str.replace("\ufeff", "")
            .str.replace("\xa0", " ")
            .str.strip()
        )
        out = pd.to_datetime(s, errors="coerce", dayfirst=True)
        ok = out.notna().sum()
        if ok < max(5, 0.2 * len(s)):
            for fmt in ("%d-%b-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"):
                out2 = pd.to_datetime(s, format=fmt, errors="coerce")
                if out2.notna().sum() > ok:
                    out, ok = out2, out2.notna().sum()
        if ok < max(5, 0.2 * len(s)):
            s_num = pd.to_numeric(series, errors="coerce")
            if s_num.notna().any() and (s_num > 20000).sum() > 0:
                out3 = pd.to_datetime(
                    s_num, unit="D", origin="1899-12-30", errors="coerce"
                )
                if out3.notna().sum() > ok:
                    out, ok = out3, out3.notna().sum()
        return out

    best = None
    best_ok = -1
    for c in candidates:
        parsed = try_parse(df[c])
        ok = parsed.notna().sum()
        if ok > best_ok:
            best = parsed
            best_ok = ok
    df[target_col] = best if best is not None else pd.NaT
    return df

def _strict_asin_from_column(series: pd.Series) -> pd.Series:
    """Use ONLY the ASIN column. Ignore blanks and values with length < 9."""
    s = series.astype(str).str.upper().str.strip()
    s = s.str.replace(r"[^A-Z0-9]", "", regex=True)
    s = s.where(s.str.len() >= 9, other=pd.NA)
    return s

def _canonicalize_asin_inplace_strict(df: pd.DataFrame) -> None:
    """Enforce ASIN solely from 'ASIN' column with len >= 9."""
    if df is None or df.empty:
        return
    if ASIN_HEADER not in df.columns:
        df[ASIN_HEADER] = pd.NA
    df[ASIN_HEADER] = _strict_asin_from_column(df[ASIN_HEADER])

def _gviz_csv_url(sheet_id: str, sheet_name: str) -> str:
    return _GVIZ_CSV_TMPL.format(sheet_id=sheet_id, sheet=quote(sheet_name))


# ===================== DATA LOADING =====================
@st.cache_data(show_spinner=False, ttl=600)
def _read_csv_url(url: str) -> pd.DataFrame:
    return pd.read_csv(url, dtype=str, keep_default_na=False, na_values=[""])

def _detect_money_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    sale_aliases = ["sale price", "sale", "sales", "total sale", REVENUE_HEADER.lower()]
    purchase_aliases = [
        "purchase price",
        "purchase",
        "cost",
        "buy price",
        "total purchase",
        PURCHASE_HEADER.lower(),
    ]
    sale_col = next((cols[a] for a in sale_aliases if a in cols), None)
    pur_col = next((cols[a] for a in purchase_aliases if a in cols), None)
    if sale_col is None or pur_col is None:
        for key, raw in cols.items():
            if sale_col is None and "sale" in key and "price" in key:
                sale_col = raw
            if pur_col is None and (("purchase" in key) or ("cost" in key)):
                pur_col = raw
    if sale_col is None or pur_col is None:
        sale_col = cols.get(REVENUE_HEADER.lower(), REVENUE_HEADER)
        pur_col = cols.get(PURCHASE_HEADER.lower(), PURCHASE_HEADER)
    return sale_col, pur_col

def _prepare_sales_frame(df: pd.DataFrame, account_name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    _normalize_columns_inplace(df)
    _ensure_date_column(df, DATE_HEADER)

    sale_col, pur_col = _detect_money_columns(df)
    if sale_col not in df.columns:
        df[sale_col] = 0
    if pur_col not in df.columns:
        df[pur_col] = 0

    # --- keep RAW text for validation (BEFORE coercion) ---
    df[RAW_SALE_COL] = df[sale_col].astype(str)
    df[RAW_PURCHASE_COL] = df[pur_col].astype(str)

    # numeric working columns used by the app
    df[REVENUE_HEADER] = _coerce_money(df[sale_col])
    df[PURCHASE_HEADER] = _coerce_money(df[pur_col])

    if UNITS_HEADER in df.columns:
        df[UNITS_HEADER] = pd.to_numeric(df[UNITS_HEADER], errors="coerce").fillna(0)
    else:
        df[UNITS_HEADER] = 1

    _canonicalize_asin_inplace_strict(df)
    df["Account"] = account_name
    return df

def load_store_names() -> List[str]:
    path = os.path.join(os.getcwd(), "StoreNames.txt")
    if not os.path.exists(path):
        return []
    names: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = _normspace(line)
            if name and not line.strip().startswith("#"):
                names.append(name)
    return names

def load_sales_by_tabs(sheet_id: str, tab_names: Iterable[str]):
    attempted, failures, frames = [], [], []
    sid = sheet_id or DEFAULT_SHEET_ID
    for tab in tab_names:
        try:
            url = _gviz_csv_url(sid, tab)
            attempted.append(url)
            raw = _read_csv_url(url)
            if raw is None or raw.empty:
                failures.append(f"{tab}: empty")
                continue
            prepped = _prepare_sales_frame(raw, account_name=tab)
            if not prepped.empty:
                frames.append(prepped)
            else:
                failures.append(f"{tab}: no rows after prep")
        except Exception as e:
            failures.append(f"{tab}: {e}")
    if not frames:
        return pd.DataFrame(), attempted, failures
    out = pd.concat(frames, ignore_index=True)
    out[REVENUE_HEADER] = pd.to_numeric(out[REVENUE_HEADER], errors="coerce").fillna(0.0)
    out[PURCHASE_HEADER] = pd.to_numeric(out[PURCHASE_HEADER], errors="coerce").fillna(0.0)
    out[UNITS_HEADER] = pd.to_numeric(out[UNITS_HEADER], errors="coerce").fillna(0)
    _ensure_date_column(out, DATE_HEADER)
    _canonicalize_asin_inplace_strict(out)
    return out, attempted, failures

def _pubhtml_to_csv_url(pubhtml_url: str) -> Optional[str]:
    """Convert /pubhtml to /pub CSV URL, preserving gid."""
    if not pubhtml_url or "/pubhtml" not in pubhtml_url:
        return None
    parsed = urlparse(pubhtml_url)
    gid = None
    if parsed.fragment:
        try:
            parts = parse_qs(parsed.fragment)
            gid = parts.get("gid", [None])[0]
        except Exception:
            gid = None
    if not gid:
        q = parse_qs(parsed.query)
        gid = q.get("gid", [None])[0]
    path = parsed.path.replace("/pubhtml", "/pub")
    params = {}
    if gid:
        params["gid"] = gid
        params["single"] = "true"
    params["output"] = "csv"
    query = urlencode(params)
    new_parsed = parsed._replace(path=path, query=query, fragment="")
    return urlunparse(new_parsed)

@st.cache_data(show_spinner=False, ttl=600)
def _safe_read_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(url, dtype=str, keep_default_na=False, na_values=[""])

def _detect_product_columns(df: pd.DataFrame):
    cols = {str(c).strip().lower(): c for c in df.columns}
    asin_col = None
    hunter_col = None
    for key, raw in cols.items():
        if "asin" in key:
            asin_col = raw
            break
    for alias in ["hunter name", "hunter", "listed by", "hunter_name", "huntername"]:
        if alias in cols:
            hunter_col = cols[alias]
            break
    if hunter_col is None:
        for key, raw in cols.items():
            if "hunter" in key and "name" in key:
                hunter_col = raw
                break
            if "hunter" in key:
                hunter_col = raw
                break
    return asin_col, hunter_col

@st.cache_data(show_spinner=False, ttl=600)
def load_product_db_from_pubhtml(
    pubhtml_url: str,
    fallback_sheet_id: Optional[str] = None,
    fallback_tab_name: Optional[str] = None,
) -> pd.DataFrame:
    last_error = None
    df = pd.DataFrame()
    try:
        csv_url = _pubhtml_to_csv_url(pubhtml_url) if pubhtml_url else None
        if csv_url:
            df = _safe_read_csv(csv_url)
    except Exception as e:
        last_error = e
        df = pd.DataFrame()
    if (df is None or df.empty) and fallback_sheet_id and fallback_tab_name:
        try:
            url = _GVIZ_CSV_TMPL.format(sheet_id=fallback_sheet_id, sheet=quote(fallback_tab_name))
            df = _read_csv_url(url)
        except Exception as e:
            last_error = e
            df = pd.DataFrame()
    if df is None or df.empty:
        raise RuntimeError(f"Unable to load Product DB. Last error: {last_error}")
    _normalize_columns_inplace(df)
    asin_col, hunter_col = _detect_product_columns(df)

    out = pd.DataFrame()
    if asin_col is not None and asin_col in df.columns:
        out[ASIN_HEADER] = _strict_asin_from_column(df[asin_col])
    else:
        out[ASIN_HEADER] = pd.NA
    out = out.dropna(subset=[ASIN_HEADER]).copy()
    out = out[out[ASIN_HEADER].str.len() >= 9].copy()
    out = out.drop_duplicates(subset=[ASIN_HEADER], keep="first")

    if hunter_col is not None and hunter_col in df.columns:
        out["Hunter Name"] = df[hunter_col].apply(_normspace)
    else:
        out["Hunter Name"] = pd.NA

    out = out.reset_index(drop=True)
    return out

def join_sales_with_hunters(sales_df: pd.DataFrame, product_df: pd.DataFrame) -> pd.DataFrame:
    if sales_df is None or sales_df.empty:
        return sales_df
    if product_df is None or product_df.empty:
        out = sales_df.copy()
        out["Hunter Name"] = "Unassigned"
        return out
    left = sales_df.copy()
    right = product_df.copy()
    _normalize_columns_inplace(left)
    _normalize_columns_inplace(right)
    _canonicalize_asin_inplace_strict(left)
    _canonicalize_asin_inplace_strict(right)

    merged = left.merge(
        right[[ASIN_HEADER, "Hunter Name"]].drop_duplicates(subset=[ASIN_HEADER]),
        on=ASIN_HEADER,
        how="left",
    )
    merged["Hunter Name"] = merged["Hunter Name"].fillna("Unassigned")
    return merged


# ===================== AGGREGATION (profit-only) =====================
def aggregate_by_hunter_profit_only(df: pd.DataFrame) -> pd.DataFrame:
    """Group by Hunter and compute Orders & Profit only. Sort by Profit desc."""
    if df.empty:
        return pd.DataFrame(columns=["Name", "Orders", "Profit"])

    d = df.copy()
    d["_units"] = pd.to_numeric(d.get(UNITS_HEADER, 1), errors="coerce").fillna(0)

    g = d.groupby("Hunter Name", as_index=False).agg(
        Orders=("_units", "sum"),
        Sales=(REVENUE_HEADER, "sum"),
        Purchase=(PURCHASE_HEADER, "sum"),
    )
    g["Profit"] = adjusted_profit(g["Sales"], g["Purchase"])
    g.rename(columns={"Hunter Name": "Name"}, inplace=True)
    g = (
        g[["Name", "Orders", "Profit"]]
        .sort_values("Profit", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    g["Orders"] = pd.to_numeric(g["Orders"], errors="coerce").fillna(0).astype(int)
    g["Profit"] = pd.to_numeric(g["Profit"], errors="coerce").fillna(0.0)
    return g

def kpis_for_hunter(df: pd.DataFrame) -> dict:
    """KPIs (profit-only) for a single hunter in a given period."""
    if df.empty:
        return dict(profit=0.0, orders=0)
    rev = float(df[REVENUE_HEADER].sum(skipna=True))
    pur = float(df[PURCHASE_HEADER].sum(skipna=True))
    profit = float(adjusted_profit(rev, pur))
    orders = int(pd.to_numeric(df.get(UNITS_HEADER, 1), errors="coerce").fillna(0).sum())
    return dict(profit=profit, orders=orders)


# ===================== VALIDATION (with AMZ Order #) =====================
def build_validation_issues(
    work_df: pd.DataFrame,
    *,
    date_col: str = DATE_HEADER,
    revenue_col: str = REVENUE_HEADER,
    purchase_col: str = PURCHASE_HEADER,
    units_col: str = UNITS_HEADER,
    asin_col: str = ASIN_HEADER,
    order_col: str = ORDER_ID_HEADER,
) -> pd.DataFrame:
    """
    Return: columns [Store, Row, Issue, Column, Value, AMZ Order #, Suggestion]
    Row ≈ spreadsheet row number (header ~1, data ~2+)
    """
    if work_df is None or work_df.empty:
        return pd.DataFrame(
            columns=["Store", "Row", "Issue", "Column", "Value", "AMZ Order #", "Suggestion"]
        )

    df = work_df.copy()
    # ensure columns exist
    for c in [date_col, revenue_col, purchase_col, units_col, asin_col, order_col]:
        if c not in df.columns:
            df[c] = pd.NA
    if "Account" not in df.columns:
        df["Account"] = "Unknown"

    df["_row"] = df.index.astype(int) + 2
    issues = []

    def add_issues(mask, issue, column, value_series, suggestion):
        if mask is None or not hasattr(mask, "any") or not mask.any():
            return
        tmp = df.loc[mask, ["Account", "_row", order_col]].copy()
        tmp["Issue"] = issue
        tmp["Column"] = column
        tmp["Value"] = value_series[mask].astype(str).str.slice(0, 300)
        tmp["Suggestion"] = suggestion
        tmp.rename(
            columns={"Account": "Store", "_row": "Row", order_col: "AMZ Order #"}, inplace=True
        )
        issues.append(tmp)

    # 1) Date invalid/missing
    date_series = pd.to_datetime(df[date_col], errors="coerce")
    add_issues(
        date_series.isna(),
        "Invalid or missing date",
        date_col,
        df.get(date_col, pd.Series(index=df.index, dtype="object")),
        "Fix date format (e.g., 20-Aug-2025).",
    )

    # 2) Purchase checks (RAW string typo checks first)
    pur_raw = (
        df.get(RAW_PURCHASE_COL, df.get(purchase_col))
        .astype(str)
    )

    pur_multiple_dots = pur_raw.str.contains(r"\.\.", regex=True, na=False)
    add_issues(
        pur_multiple_dots,
        "Purchase typo: multiple decimals",
        purchase_col,
        pur_raw,
        "Remove extra dot (e.g., 59..95 → 59.95).",
    )

    pur_comma_decimal = pur_raw.str.match(r"^\s*-?\d+[ ]*,[ ]*\d{1,3}\s*$", na=False)
    add_issues(
        pur_comma_decimal,
        "Purchase typo: comma used as decimal",
        purchase_col,
        pur_raw,
        "Use a dot as decimal (18,99 → 18.99).",
    )

    pur_illegal_chars = pur_raw.str.contains(r"[^\d\.\,\-\s]", regex=True, na=False)
    add_issues(
        pur_illegal_chars,
        "Purchase contains invalid characters",
        purchase_col,
        pur_raw,
        "Keep only digits, one dot, optional minus (remove letters/symbols).",
    )

    # Optional: invalid thousands separators (allow 1,234 and 12,345,678 or plain digits)
    def _bad_thousands(s: pd.Series) -> pd.Series:
        plain_digits = s.str.match(r"^\s*-?\d+(\.\d+)?\s*$", na=False)
        good_thousands = s.str.match(r"^\s*-?\d{1,3}(,\d{3})+(\.\d+)?\s*$", na=False)
        comma_decimal = s.str.match(r"^\s*-?\d+,\d{1,3}\s*$", na=False)
        has_comma = s.str.contains(",", regex=False, na=False)
        return has_comma & ~plain_digits & ~good_thousands & ~comma_decimal

    add_issues(
        _bad_thousands(pur_raw),
        "Purchase typo: invalid thousands separators",
        purchase_col,
        pur_raw,
        "Use 1,234 or 12,345,678 (or no commas).",
    )

    add_issues(
        pur_raw.str.strip().eq("").fillna(True),
        "Missing purchase price",
        purchase_col,
        pur_raw,
        "Enter numeric purchase.",
    )
    pur_num = pd.to_numeric(df.get(purchase_col), errors="coerce")
    add_issues(pur_num.lt(0).fillna(False), "Negative purchase", purchase_col, pur_num, "Make non-negative.")
    add_issues(
        pur_num.gt(10000).fillna(False),
        "Unusually high purchase",
        purchase_col,
        pur_num,
        "Check currency/decimal; confirm cost.",
    )

    # 3) Sale checks (RAW string typo checks first)
    sale_raw = (
        df.get(RAW_SALE_COL, df.get(revenue_col))
        .astype(str)
    )

    sale_multiple_dots = sale_raw.str.contains(r"\.\.", regex=True, na=False)
    add_issues(
        sale_multiple_dots,
        "Sale typo: multiple decimals",
        revenue_col,
        sale_raw,
        "Remove extra dot (e.g., 59..95 → 59.95).",
    )

    sale_comma_decimal = sale_raw.str.match(r"^\s*-?\d+[ ]*,[ ]*\d{1,3}\s*$", na=False)
    add_issues(
        sale_comma_decimal,
        "Sale typo: comma used as decimal",
        revenue_col,
        sale_raw,
        "Use a dot as decimal (18,99 → 18.99).",
    )

    sale_illegal_chars = sale_raw.str.contains(r"[^\d\.\,\-\s]", regex=True, na=False)
    add_issues(
        sale_illegal_chars,
        "Sale contains invalid characters",
        revenue_col,
        sale_raw,
        "Keep only digits, one dot, optional minus (remove letters/symbols).",
    )

    add_issues(
        _bad_thousands(sale_raw),
        "Sale typo: invalid thousands separators",
        revenue_col,
        sale_raw,
        "Use 1,234 or 12,345,678 (or no commas).",
    )

    add_issues(
        sale_raw.str.strip().eq("").fillna(True),
        "Missing sale price",
        revenue_col,
        sale_raw,
        "Enter numeric sale price.",
    )
    sale_num = pd.to_numeric(df.get(revenue_col), errors="coerce")
    add_issues(sale_num.lt(0).fillna(False), "Negative sale price", revenue_col, sale_num, "Make non-negative.")

    # 4) Units checks
    units_num = pd.to_numeric(df.get(units_col), errors="coerce")
    add_issues(units_num.isna() | units_num.le(0), "Invalid units", units_col, units_num, "Use integer ≥ 1.")

    # 5) ASIN checks
    asin_series = df.get(asin_col).astype(str)
    asin_bad = asin_series.isna() | (asin_series.str.len() < 9) | asin_series.str.strip().eq("")
    add_issues(
        asin_bad.fillna(True), "Unassigned sale (ASIN missing/short)", asin_col, asin_series, "Fill ASIN (≥ 9 chars)."
    )

    # 6) Hunter mapping
    hunter_series = df.get("Hunter Name", pd.Series(index=df.index, dtype="object")).astype(str)
    add_issues(
        hunter_series.str.startswith("Unassigned"),
        "Unassigned hunter",
        "Hunter Name",
        hunter_series,
        "Add ASIN to Product DB (ListedItems) with Hunter Name.",
    )

    # 7) Purchase > 2× Sale (only when Sale > 0)
    mask_exceeds = (sale_num > 0) & (pur_num > (2.0 * sale_num))
    add_issues(
        mask_exceeds.fillna(False),
        "Purchase exceeds 2× sale",
        purchase_col,
        pur_num,
        "Check sale/purchase for swap/typo.",
    )

    if not issues:
        return pd.DataFrame(
            columns=["Store", "Row", "Issue", "Column", "Value", "AMZ Order #", "Suggestion"]
        )
    out = pd.concat(issues, ignore_index=True)
    return out.sort_values(["Store", "Row", "Issue"], kind="mergesort").reset_index(drop=True)


# ===================== LOGGING HELPERS =====================
def _load_validation_log() -> pd.DataFrame:
    if not os.path.exists(VALIDATION_LOG_PATH):
        return pd.DataFrame(
            columns=[
                "Timestamp",
                "User",
                "Action",
                "Store",
                "Row",
                "Issue",
                "Column",
                "Value",
                "AMZ Order #",
                "Note",
            ]
        )
    try:
        df = pd.read_csv(VALIDATION_LOG_PATH, dtype=str)
        return df
    except Exception:
        return pd.DataFrame(
            columns=[
                "Timestamp",
                "User",
                "Action",
                "Store",
                "Row",
                "Issue",
                "Column",
                "Value",
                "AMZ Order #",
                "Note",
            ]
        )

def _append_validation_log(entries: pd.DataFrame):
    if entries is None or entries.empty:
        return
    file_exists = os.path.exists(VALIDATION_LOG_PATH)
    entries.to_csv(
        VALIDATION_LOG_PATH,
        mode="a",
        index=False,
        header=not file_exists,
        encoding="utf-8",
    )


# ===================== APP =====================
st.set_page_config(page_title=APP_TITLE, page_icon="👥", layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Data Source")
    _ = st.text_input("Sales Sheet ID", value=DEFAULT_SHEET_ID, type="password", disabled=True)
    _ = st.text_input("Product DB URL", value=DEFAULT_PRODUCT_PUBHTML, type="password", disabled=True)

    st.subheader("Stores")
    stores = load_store_names()
    st.write(stores if stores else "— none —")

    if st.button("Refresh", use_container_width=True):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.session_state["__do_refresh__"] = True
        st.rerun()

# Progress
show_progress = st.session_state.pop("__do_refresh__", False)
progress_bar = None
progress_text = None
def step(pct: int, text: str):
    global progress_bar, progress_text
    if progress_bar is None:
        progress_text = st.empty()
        progress_bar = st.progress(0, text=text)
    progress_text.text(text)
    progress_bar.progress(min(max(pct, 0), 100))

# Guard: stores
if not stores:
    st.error("Add store tab names in StoreNames.txt.")
    st.stop()

# Load sales
try:
    if show_progress:
        step(20, "Loading sales…")
    sales_df, attempted_urls, failures = load_sales_by_tabs(DEFAULT_SHEET_ID, stores)
    if show_progress:
        step(45, "Sales loaded.")
    if failures:
        with st.expander("Load Warnings"):
            st.write(failures)
except Exception as e:
    if show_progress:
        step(100, "Failed loading sales.")
    st.error(f"Failed to load sales: {e}")
    st.stop()

# Load product DB
product_df = None
try:
    if show_progress:
        step(60, "Loading products…")
    product_df = load_product_db_from_pubhtml(
        DEFAULT_PRODUCT_PUBHTML,
        fallback_sheet_id=FALLBACK_PRODUCT_SHEET_ID,
        fallback_tab_name=FALLBACK_PRODUCT_TAB_NAME,
    )
    if show_progress:
        step(70, "Products loaded.")
except Exception as e:
    if show_progress:
        step(70, "Products skipped.")
    st.warning(f"Product DB not loaded: {e}")

# Join → work
try:
    if show_progress:
        step(80, "Joining…")
    if product_df is None:
        work = sales_df.copy()
        work["Hunter Name"] = "Unassigned"
    else:
        work = join_sales_with_hunters(sales_df, product_df)
    if show_progress:
        step(90, "Done.")
except Exception as e:
    if show_progress:
        step(100, "Join failed.")
    st.error(f"Failed to map Hunter Name: {e}")
    st.stop()

if show_progress:
    step(100, "Refreshed")
    time.sleep(0.2)

if work.empty:
    st.info("No data.")
    st.stop()

# ===================== TABS =====================
tab_progress, tab_validation = st.tabs(["Progress", "Validation"])

with tab_progress:
    # Staff selector & trend window (30/60 only)
    hunters = sorted([h for h in work["Hunter Name"].dropna().unique() if h and h != "Unassigned"])
    if not hunters:
        st.error("No mapped hunters found.")
        st.stop()

    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        who = st.selectbox("Hunter", hunters, index=0, key="sel_hunter")
    with col_sel2:
        days = st.selectbox("Trend Window", ["30 days", "60 days"], index=1, key="sel_days")
    trend_days = int(days.split()[0])

    mine = work[work["Hunter Name"] == who].copy()

    # Determine month windows based on data
    max_date = pd.to_datetime(mine[DATE_HEADER].dropna().max())
    if pd.isna(max_date):
        st.error("No dates for this hunter.")
        st.stop()

    def _safe_month_bounds(d: pd.Timestamp):
        d = pd.to_datetime(d)
        start = d.to_period("M").start_time.normalize()
        end = (start + pd.offsets.MonthEnd(0)).normalize()
        return start, end

    cur_start, cur_end = _safe_month_bounds(max_date)
    last_start, last_end = _safe_month_bounds(cur_start - pd.Timedelta(days=1))
    prev2_start, prev2_end = _safe_month_bounds(last_start - pd.Timedelta(days=1))

    cur_scope = mine[(mine[DATE_HEADER] >= cur_start) & (mine[DATE_HEADER] <= cur_end)]
    last_scope = mine[(mine[DATE_HEADER] >= last_start) & (mine[DATE_HEADER] <= last_end)]

    kc = kpis_for_hunter(cur_scope)
    kl = kpis_for_hunter(last_scope)

    st.markdown(f"### {who}")
    (m1,) = st.columns(1)
    m1.metric("Profit", f"${kc['profit']:,.2f}", delta=f"${(kc['profit'] - kl['profit']):,.2f}")

    # Daily profit trend (last N days)
    since = max_date - pd.Timedelta(days=trend_days - 1)
    trend = mine[mine[DATE_HEADER] >= since].copy()
    daily = (
        trend.groupby(DATE_HEADER, as_index=False)
        .agg(Sales=(REVENUE_HEADER, "sum"), Purchase=(PURCHASE_HEADER, "sum"))
        .sort_values(DATE_HEADER)
    )
    daily["Profit"] = adjusted_profit(daily["Sales"], daily["Purchase"])
    fig1 = px.line(daily, x=DATE_HEADER, y=["Profit"], title="Daily Profit")
    st.plotly_chart(fig1, use_container_width=True)

    # Cumulative profit — current vs last 2 months
    def cumulative_profit(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
        frame = df[(df[DATE_HEADER] >= start) & (df[DATE_HEADER] <= end)].copy()
        g = frame.groupby(DATE_HEADER, as_index=False).agg(
            Sales=(REVENUE_HEADER, "sum"), Purchase=(PURCHASE_HEADER, "sum")
        )
        g = g.sort_values(DATE_HEADER)
        g["Cumulative Profit"] = adjusted_profit(g["Sales"], g["Purchase"]).cumsum()
        return g[[DATE_HEADER, "Cumulative Profit"]]

    cur_cum = cumulative_profit(mine, cur_start, cur_end)
    cur_cum["Period"] = cur_start.strftime("%b %Y")
    last_cum = cumulative_profit(mine, last_start, last_end)
    last_cum["Period"] = last_start.strftime("%b %Y")
    prev2_cum = cumulative_profit(mine, prev2_start, prev2_end)
    prev2_cum["Period"] = prev2_start.strftime("%b %Y")
    cum = pd.concat([cur_cum, last_cum, prev2_cum], ignore_index=True)
    fig2 = px.line(
        cum, x=DATE_HEADER, y="Cumulative Profit", color="Period", markers=True, title="Cumulative Profit"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Top ASINs this month by profit
    thism = mine[(mine[DATE_HEADER] >= cur_start) & (mine[DATE_HEADER] <= cur_end)].copy()
    asin_top = (
        thism.dropna(subset=[ASIN_HEADER])
        .groupby(ASIN_HEADER, as_index=False)
        .agg(Sales=(REVENUE_HEADER, "sum"), Purchase=(PURCHASE_HEADER, "sum"), Orders=(UNITS_HEADER, "sum"))
    )
    asin_top["Profit"] = adjusted_profit(asin_top["Sales"], asin_top["Purchase"])
    asin_top = asin_top.sort_values("Profit", ascending=False).head(15)
    fig3 = px.bar(asin_top, x=ASIN_HEADER, y="Profit", title="Top ASINs (Profit)")
    st.plotly_chart(fig3, use_container_width=True)

    # All hunters (current & last 2 months) — Profit only
    def slice_period(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return df[(df[DATE_HEADER] >= start) & (df[DATE_HEADER] <= end)]

    cur_agg_all = aggregate_by_hunter_profit_only(slice_period(work, cur_start, cur_end))
    last_agg_all = aggregate_by_hunter_profit_only(slice_period(work, last_start, last_end))
    prev2_agg_all = aggregate_by_hunter_profit_only(slice_period(work, prev2_start, prev2_end))

    fmt = {"Orders": "{:,.0f}", "Profit": "{:,.2f}"}
    st.subheader(f"All Hunters — {cur_start.strftime('%b %Y')}")
    st.dataframe(cur_agg_all.style.format(fmt), use_container_width=True)

    st.subheader(f"All Hunters — {last_start.strftime('%b %Y')}")
    st.dataframe(last_agg_all.style.format(fmt), use_container_width=True)

    st.subheader(f"All Hunters — {prev2_start.strftime('%b %Y')}")
    st.dataframe(prev2_agg_all.style.format(fmt), use_container_width=True)

    # Export three CSVs (profit-only tables)
    def _zip_reports(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"profit_{cur_start.strftime('%b_%Y').lower()}.csv", a.to_csv(index=False))
            z.writestr(f"profit_{last_start.strftime('%b_%Y').lower()}.csv", b.to_csv(index=False))
            z.writestr(f"profit_{prev2_start.strftime('%b_%Y').lower()}.csv", c.to_csv(index=False))
        buf.seek(0)
        return buf.getvalue()

    st.download_button(
        "Download Reports (ZIP)",
        data=_zip_reports(cur_agg_all, last_agg_all, prev2_agg_all),
        file_name="staff_profit_reports.zip",
        mime="application/zip",
        use_container_width=True,
    )

with tab_validation:
    st.subheader("Data Validation")

    # Build the full issues list (includes Row internally)
    issues_all = build_validation_issues(
        work,
        date_col=DATE_HEADER,
        revenue_col=REVENUE_HEADER,
        purchase_col=PURCHASE_HEADER,
        units_col=UNITS_HEADER,
        asin_col=ASIN_HEADER,
        order_col=ORDER_ID_HEADER,
    )

    # If no issues at all
    if issues_all.empty:
        st.success("No issues found.")
    else:
        # --- Suppress issues that were marked as Ignored previously ---
        log = _load_validation_log()
        issues = issues_all.copy()

        if not log.empty:
            ignored = log.loc[log["Action"] == "Ignore", ["Store", "Issue", "Column", "Value", "AMZ Order #"]].copy()
            if not ignored.empty:
                ignored["key"] = (
                    ignored["Store"].astype(str)
                    + "|" + ignored["Issue"].astype(str)
                    + "|" + ignored["Column"].astype(str)
                    + "|" + ignored["Value"].astype(str)
                    + "|" + ignored["AMZ Order #"].astype(str)
                )
                issues["key"] = (
                    issues["Store"].astype(str)
                    + "|" + issues["Issue"].astype(str)
                    + "|" + issues["Column"].astype(str)
                    + "|" + issues["Value"].astype(str)
                    + "|" + issues["AMZ Order #"].astype(str)
                )
                issues = issues[~issues["key"].isin(ignored["key"])]
                issues = issues.drop(columns=["key"], errors="ignore")

        # After filtering, show message if nothing remains
        if issues.empty:
            st.info("No new/unreviewed issues.")
        else:
            # ---------- Filters ----------
            cols = st.columns(4)
            with cols[0]:
                stores_unique = ["All"] + sorted(issues["Store"].dropna().unique().tolist())
                f_store = st.selectbox("Store", stores_unique, index=0, key="val_store")
            with cols[1]:
                types_unique = ["All"] + sorted(issues["Issue"].dropna().unique().tolist())
                f_issue = st.selectbox("Issue Type", types_unique, index=0, key="val_issue")
            with cols[2]:
                topn = st.selectbox("Limit", ["All", "100", "500", "1000"], index=1, key="val_topn")
            with cols[3]:
                reviewer = st.text_input("Your Name", value=st.session_state.get("reviewer", ""), key="reviewer_name")

            view = issues.copy()
            if f_store != "All":
                view = view[view["Store"] == f_store]
            if f_issue != "All":
                view = view[view["Issue"] == f_issue]
            if topn != "All":
                n = int(topn)
                view = view.head(n)

            st.caption("Issues table excludes rows previously marked as Ignored.")

            # ---------- Editable review table (nonce-based reset) ----------
            # Build display frame WITHOUT the Row column (hidden from user),
            # but keep a composite key to join back to get Row on submit.
            if "Action" not in view.columns:
                view["Action"] = ""
            if "Note" not in view.columns:
                view["Note"] = ""

            # Build join key (same as used for suppression)
            view["__key__"] = (
                view["Store"].astype(str)
                + "|" + view["Issue"].astype(str)
                + "|" + view["Column"].astype(str)
                + "|" + view["Value"].astype(str)
                + "|" + view["AMZ Order #"].astype(str)
            )

            display_cols = [c for c in view.columns if c not in ("Row", "__key__")]
            display = view[display_cols].copy()

            nonce = st.session_state.get("validation_editor_nonce", 0)
            disabled_cols = [c for c in display.columns if c not in ("Action", "Note")]
            edited = st.data_editor(
                display,
                use_container_width=True,
                height=420,
                column_config={
                    "Action": st.column_config.SelectboxColumn(
                        "Action", options=["", "Confirm", "Ignore"], required=False
                    ),
                    "Note": st.column_config.TextColumn("Note"),
                },
                disabled=disabled_cols,
                key=f"validation_editor_{nonce}",  # unique key per mount
            )

            left, right = st.columns([1, 1])
            with left:
                if st.button("Submit Selected Actions", use_container_width=True):
                    to_log_display = edited[edited["Action"].isin(["Confirm", "Ignore"])].copy()
                    if to_log_display.empty:
                        st.warning("No rows with an Action selected.")
                    else:
                        # Re-attach Row using the key (join edited back to original view)
                        to_log_display["__key__"] = (
                            to_log_display["Store"].astype(str)
                            + "|" + to_log_display["Issue"].astype(str)
                            + "|" + to_log_display["Column"].astype(str)
                            + "|" + to_log_display["Value"].astype(str)
                            + "|" + to_log_display["AMZ Order #"].astype(str)
                        )
                        merged_for_log = to_log_display.merge(
                            view[["__key__", "Row"]],
                            on="__key__",
                            how="left",
                        )

                        ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                        user = reviewer.strip() or "anonymous"

                        log_df = merged_for_log[
                            ["Store", "Row", "Issue", "Column", "Value", "AMZ Order #", "Action", "Note"]
                        ].copy()
                        log_df.insert(0, "Timestamp", ts)
                        log_df.insert(1, "User", user)

                        _append_validation_log(log_df)
                        st.success(f"Logged {len(log_df)} action(s).")

                        # Bump nonce to remount editor with a fresh key (clears edits)
                        st.session_state["validation_editor_nonce"] = nonce + 1
                        st.rerun()

            with right:
                st.download_button(
                    "Download All Issues (CSV)",
                    data=issues_all.to_csv(index=False).encode("utf-8"),
                    file_name="data_validation_issues.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with st.expander("Activity Log"):
                log = _load_validation_log()
                if log.empty:
                    st.info("No actions logged yet.")
                else:
                    st.dataframe(log, use_container_width=True, height=300)
