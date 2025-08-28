# app.py — Staff Progress Dashboard (Profit-only, simplified titles, 10% margin cut via purchase uplift)
from __future__ import annotations

import io, zipfile, os, re, unicodedata, time
from typing import List, Optional, Iterable, Tuple
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, quote

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ===================== CONFIG / CONSTANTS =====================
APP_TITLE = "Staff Progress"

# Your Sales Google Sheet ID (masked & read-only in UI)
DEFAULT_SHEET_ID = "1pIo7rvtYBJhnFW0sSrPorF-O-dHDEYL1Q0B_HsNh8r8"

# Published-to-web URL for Product DB “ListedItems” (masked & read-only in UI)
DEFAULT_PRODUCT_PUBHTML = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vSQ0NTS2w_jkBtUy-"
    "h5W7Ml60fTd1H_tlblwCCt1-0MGLxRos5Y2HxrlRK5ONmpitIcxeznuwP67Y8d/"
    "pubhtml#gid=959982031"
)

# Fallback if published link fails (try loading tab directly from same file)
FALLBACK_PRODUCT_SHEET_ID = DEFAULT_SHEET_ID
FALLBACK_PRODUCT_TAB_NAME = "ListedItems"  # renamed (no space)

# Canonical headers in Sales
DATE_HEADER = "Order Date"
REVENUE_HEADER = "Sale Price"
PURCHASE_HEADER = "Purchase Price"
UNITS_HEADER = "Units"
ASIN_HEADER = "ASIN"

# Global adjusted-profit rule: reduce margin by 10% by increasing purchase
# Profit_adj = 0.85 * Sales − 1.10 * Purchase
def adjusted_profit(sales: float | pd.Series, purchase: float | pd.Series) -> float | pd.Series:
    return 0.85 * sales - 1.05 * purchase

# ===================== HELPERS =====================
_WS_RE = re.compile(r"\s+")
_GVIZ_CSV_TMPL = "https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet}"

def _normspace(s: str) -> str:
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", str(s or "")).strip())

def _normalize_columns_inplace(df: pd.DataFrame) -> None:
    if df is None or df.empty: return
    df.rename(columns={c: " ".join(str(c).replace("\n"," ").strip().split()) for c in df.columns}, inplace=True)

def _coerce_money(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace("[^0-9.\-]", "", regex=True)
         .replace({"": np.nan, "-": np.nan})
         .astype(float)
         .fillna(0.0)
    )

def _ensure_date_column(df: pd.DataFrame, target_col: str = DATE_HEADER) -> pd.DataFrame:
    if df is None or df.empty:
        df = pd.DataFrame(columns=[target_col]); df[target_col] = pd.NaT; return df
    cols = list(df.columns)
    lower = [str(c).strip().lower() for c in cols]
    candidates: List[str] = []
    if target_col in df.columns: candidates.append(target_col)
    alt = {"order date","order  date","order_date","orderdate","fulfillment date","order fullfill date","date"}
    for c, n in zip(cols, lower):
        if c == target_col: continue
        if any(k in n for k in alt) or "date" in n:
            candidates.append(c)
    if not candidates and cols: candidates = [cols[0]]

    def try_parse(series: pd.Series) -> pd.Series:
        s = (series.astype(str).str.replace("\u200b","").str.replace("\u200c","")
                           .str.replace("\ufeff","").str.replace("\xa0"," ").str.strip())
        out = pd.to_datetime(s, errors="coerce", dayfirst=True)
        ok = out.notna().sum()
        if ok < max(5, 0.2*len(s)):
            for fmt in ("%d-%b-%Y","%d/%m/%Y","%Y-%m-%d","%d-%m-%Y","%m/%d/%Y"):
                out2 = pd.to_datetime(s, format=fmt, errors="coerce")
                if out2.notna().sum() > ok: out, ok = out2, out2.notna().sum()
        if ok < max(5, 0.2*len(s)):
            s_num = pd.to_numeric(series, errors="coerce")
            if s_num.notna().any() and (s_num > 20000).sum() > 0:
                out3 = pd.to_datetime(s_num, unit="D", origin="1899-12-30", errors="coerce")
                if out3.notna().sum() > ok: out, ok = out3.notna().sum()
        return out

    best = None; best_ok = -1
    for c in candidates:
        parsed = try_parse(df[c]); ok = parsed.notna().sum()
        if ok > best_ok: best = parsed; best_ok = ok
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
    if df is None or df.empty: return
    if ASIN_HEADER not in df.columns:
        df[ASIN_HEADER] = pd.NA
    df[ASIN_HEADER] = _strict_asin_from_column(df[ASIN_HEADER])

def _gviz_csv_url(sheet_id: str, sheet_name: str) -> str:
    return _GVIZ_CSV_TMPL.format(sheet_id=sheet_id, sheet=quote(sheet_name))

@st.cache_data(show_spinner=False, ttl=600)
def _read_csv_url(url: str) -> pd.DataFrame:
    return pd.read_csv(url, dtype=str, keep_default_na=False, na_values=[""])

def _detect_money_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    sale_aliases = ["sale price","sale","sales","total sale", REVENUE_HEADER.lower()]
    purchase_aliases = ["purchase price","purchase","cost","buy price","total purchase", PURCHASE_HEADER.lower()]
    sale_col = next((cols[a] for a in sale_aliases if a in cols), None)
    pur_col  = next((cols[a] for a in purchase_aliases if a in cols), None)
    if sale_col is None or pur_col is None:
        for key, raw in cols.items():
            if sale_col is None and "sale" in key and "price" in key: sale_col = raw
            if pur_col is None and (("purchase" in key) or ("cost" in key)): pur_col = raw
    if sale_col is None or pur_col is None:
        sale_col = cols.get(REVENUE_HEADER.lower(), REVENUE_HEADER)
        pur_col  = cols.get(PURCHASE_HEADER.lower(), PURCHASE_HEADER)
    return sale_col, pur_col

def _prepare_sales_frame(df: pd.DataFrame, account_name: str) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    _normalize_columns_inplace(df)
    _ensure_date_column(df, DATE_HEADER)

    sale_col, pur_col = _detect_money_columns(df)
    if sale_col not in df.columns: df[sale_col] = 0
    if pur_col  not in df.columns: df[pur_col]  = 0

    df[REVENUE_HEADER] = _coerce_money(df[sale_col])
    df[PURCHASE_HEADER] = _coerce_money(df[pur_col])

    if UNITS_HEADER in df.columns:
        df[UNITS_HEADER] = pd.to_numeric(df[UNITS_HEADER], errors="coerce").fillna(0)
    else:
        df[UNITS_HEADER] = 1

    # STRICT ASIN: only from ASIN column (ignore URLs), keep len >= 9
    _canonicalize_asin_inplace_strict(df)

    df["Account"] = account_name
    return df

def load_store_names() -> List[str]:
    path = os.path.join(os.getcwd(), "StoreNames.txt")
    if not os.path.exists(path): return []
    names: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = _normspace(line)
            if name and not name.startswith("#"): names.append(name)
    return names

def load_sales_by_tabs(sheet_id: str, tab_names: Iterable[str]):
    attempted, failures, frames = [], [], []
    sid = sheet_id or DEFAULT_SHEET_ID
    for tab in tab_names:
        try:
            url = _gviz_csv_url(sid, tab); attempted.append(url)
            raw = _read_csv_url(url)
            if raw is None or raw.empty:
                failures.append(f"{tab}: empty"); continue
            prepped = _prepare_sales_frame(raw, account_name=tab)
            if not prepped.empty: frames.append(prepped)
            else: failures.append(f"{tab}: no rows after prep")
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
    """Convert /pubhtml to CSV URL, preserving gid."""
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
    asin_col = None; hunter_col = None
    for key, raw in cols.items():
        if "asin" in key: asin_col = raw; break
    for alias in ["hunter name","hunter","listed by","hunter_name","huntername"]:
        if alias in cols: hunter_col = cols[alias]; break
    if hunter_col is None:
        for key, raw in cols.items():
            if "hunter" in key and "name" in key: hunter_col = raw; break
            if "hunter" in key: hunter_col = raw; break
    return asin_col, hunter_col

@st.cache_data(show_spinner=False, ttl=600)
def load_product_db_from_pubhtml(pubhtml_url: str, fallback_sheet_id: Optional[str]=None, fallback_tab_name: Optional[str]=None) -> pd.DataFrame:
    last_error = None; df = pd.DataFrame()
    try:
        csv_url = _pubhtml_to_csv_url(pubhtml_url) if pubhtml_url else None
        if csv_url: df = _safe_read_csv(csv_url)
    except Exception as e:
        last_error = e; df = pd.DataFrame()
    if (df is None or df.empty) and fallback_sheet_id and fallback_tab_name:
        try:
            url = _GVIZ_CSV_TMPL.format(sheet_id=fallback_sheet_id, sheet=quote(fallback_tab_name))
            df = _read_csv_url(url)
        except Exception as e:
            last_error = e; df = pd.DataFrame()
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
    if sales_df is None or sales_df.empty: return sales_df
    if product_df is None or product_df.empty:
        out = sales_df.copy(); out["Hunter Name"] = "Unassigned"; return out
    left = sales_df.copy(); right = product_df.copy()
    _normalize_columns_inplace(left); _normalize_columns_inplace(right)
    _canonicalize_asin_inplace_strict(left); _canonicalize_asin_inplace_strict(right)

    merged = left.merge(
        right[[ASIN_HEADER, "Hunter Name"]].drop_duplicates(subset=[ASIN_HEADER]),
        on=ASIN_HEADER, how="left"
    )
    merged["Hunter Name"] = merged["Hunter Name"].fillna("Unassigned")
    return merged

# ========== Aggregation (Profit-only, 10% margin cut via purchase uplift) ==========
def aggregate_by_hunter_profit_only(df: pd.DataFrame) -> pd.DataFrame:
    """Group by Hunter and compute Orders & Profit only. Sort by Profit desc."""
    if df.empty:
        return pd.DataFrame(columns=["Name","Orders","Profit"])

    d = df.copy()
    d["_units"] = pd.to_numeric(d.get(UNITS_HEADER, 1), errors="coerce").fillna(0)

    g = d.groupby("Hunter Name", as_index=False).agg(
        Orders=("_units", "sum"),
        Sales=(REVENUE_HEADER, "sum"),
        Purchase=(PURCHASE_HEADER, "sum"),
    )
    g["Profit"] = adjusted_profit(g["Sales"], g["Purchase"])
    g.rename(columns={"Hunter Name": "Name"}, inplace=True)
    g = g[["Name","Orders","Profit"]].sort_values("Profit", ascending=False, kind="mergesort").reset_index(drop=True)
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

# ===================== UI / APP =====================
st.set_page_config(page_title=APP_TITLE, page_icon="👥", layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Data Source")
    # Read-only & masked
    _ = st.text_input("Sales Sheet ID", value=DEFAULT_SHEET_ID, type="password", disabled=True)
    _ = st.text_input("Product DB URL", value=DEFAULT_PRODUCT_PUBHTML, type="password", disabled=True)

    st.subheader("Stores")
    stores = []
    try:
        if os.path.exists("StoreNames.txt"):
            with open("StoreNames.txt","r",encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        stores.append(line)
    except Exception:
        stores = []
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
    progress_bar.progress(min(max(pct,0),100))

# Load data
if not stores:
    st.error("Add store tab names in StoreNames.txt.")
    st.stop()

try:
    if show_progress: step(20, "Loading sales…")
    sales_df, attempted, failures = load_sales_by_tabs(DEFAULT_SHEET_ID, stores)
    if show_progress: step(45, "Sales loaded.")
    if failures:
        with st.expander("Load Warnings"):
            st.write(failures)
except Exception as e:
    if show_progress: step(100, "Failed loading sales.")
    st.error(f"Failed to load sales: {e}")
    st.stop()

product_df = None
try:
    if show_progress: step(60, "Loading products…")
    product_df = load_product_db_from_pubhtml(
        DEFAULT_PRODUCT_PUBHTML,
        fallback_sheet_id=FALLBACK_PRODUCT_SHEET_ID,
        fallback_tab_name=FALLBACK_PRODUCT_TAB_NAME,
    )
    if show_progress: step(70, "Products loaded.")
except Exception as e:
    if show_progress: step(70, "Products skipped.")
    st.warning(f"Product DB not loaded: {e}")

try:
    if show_progress: step(80, "Joining…")
    if product_df is None:
        work = sales_df.copy()
        work["Hunter Name"] = "Unassigned"
    else:
        work = join_sales_with_hunters(sales_df, product_df)
    if show_progress: step(90, "Done.")
except Exception as e:
    if show_progress: step(100, "Join failed.")
    st.error(f"Failed to map Hunter Name: {e}")
    st.stop()

if show_progress: step(100, "Refreshed"); time.sleep(0.2)

if work.empty:
    st.info("No data.")
    st.stop()

# -------------------- Staff selector & trend window --------------------
hunters = sorted([h for h in work["Hunter Name"].dropna().unique() if h and h != "Unassigned"])
if not hunters:
    st.error("No mapped hunters found.")
    st.stop()

col_sel1, col_sel2 = st.columns([2,1])
with col_sel1:
    who = st.selectbox("Hunter", hunters, index=0, key="sel_hunter")
with col_sel2:
    days = st.selectbox("Trend Window", ["30 days","60 days"], index=1, key="sel_days")
trend_days = int(days.split()[0])

mine = work[work["Hunter Name"] == who].copy()

# Month windows
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

cur_scope = mine[(mine[DATE_HEADER] >= cur_start) & (mine[DATE_HEADER] <= cur_end)]
last_scope = mine[(mine[DATE_HEADER] >= last_start) & (mine[DATE_HEADER] <= last_end)]

kc = kpis_for_hunter(cur_scope)
kl = kpis_for_hunter(last_scope)

st.markdown(f"### {who}")

# KPI: Profit only (delta vs last month)
m1, = st.columns(1)
m1.metric("Profit", f"${kc['profit']:,.2f}", delta=f"${(kc['profit']-kl['profit']):,.2f}")

# -------------------- Trends & breakdowns (Profit-only) --------------------
# 1) Daily profit trend (last N days)
since = max_date - pd.Timedelta(days=trend_days-1)
trend = mine[mine[DATE_HEADER] >= since].copy()
daily = (trend.groupby(DATE_HEADER, as_index=False)
         .agg(Sales=(REVENUE_HEADER,"sum"),
              Purchase=(PURCHASE_HEADER,"sum")))
daily["Profit"] = adjusted_profit(daily["Sales"], daily["Purchase"])
fig1 = px.line(daily, x=DATE_HEADER, y=["Profit"], title="Daily Profit")
st.plotly_chart(fig1, use_container_width=True)

# 2) Cumulative profit — current vs last month
def cumulative_profit(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    frame = df[(df[DATE_HEADER] >= start) & (df[DATE_HEADER] <= end)].copy()
    g = frame.groupby(DATE_HEADER, as_index=False).agg(Sales=(REVENUE_HEADER,"sum"), Purchase=(PURCHASE_HEADER,"sum"))
    g = g.sort_values(DATE_HEADER)
    g["Cumulative Profit"] = adjusted_profit(g["Sales"], g["Purchase"]).cumsum()
    return g[[DATE_HEADER, "Cumulative Profit"]]

cur_cum = cumulative_profit(mine, cur_start, cur_end); cur_cum["Period"] = cur_start.strftime("%b %Y")
last_cum = cumulative_profit(mine, last_start, last_end); last_cum["Period"] = last_start.strftime("%b %Y")
cum = pd.concat([cur_cum, last_cum], ignore_index=True)
fig2 = px.line(cum, x=DATE_HEADER, y="Cumulative Profit", color="Period", markers=True, title="Cumulative Profit")
st.plotly_chart(fig2, use_container_width=True)

# 3) Top ASINs this month by profit
thism = mine[(mine[DATE_HEADER] >= cur_start) & (mine[DATE_HEADER] <= cur_end)].copy()
asin_top = (thism.dropna(subset=[ASIN_HEADER])
                 .groupby(ASIN_HEADER, as_index=False)
                 .agg(Sales=(REVENUE_HEADER,"sum"),
                      Purchase=(PURCHASE_HEADER,"sum"),
                      Orders=(UNITS_HEADER,"sum")))
asin_top["Profit"] = adjusted_profit(asin_top["Sales"], asin_top["Purchase"])
asin_top = asin_top.sort_values("Profit", ascending=False).head(15)
fig3 = px.bar(asin_top, x=ASIN_HEADER, y="Profit", title="Top ASINs (Profit)")
st.plotly_chart(fig3, use_container_width=True)

# -------------------- All hunters (current & last) — Profit only --------------------
def slice_period(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df[DATE_HEADER] >= start) & (df[DATE_HEADER] <= end)]

cur_agg_all = aggregate_by_hunter_profit_only(slice_period(work, cur_start, cur_end))
last_agg_all = aggregate_by_hunter_profit_only(slice_period(work, last_start, last_end))

fmt = {"Orders":"{:,.0f}", "Profit":"{:,.2f}"}

st.subheader(f"All Hunters — {cur_start.strftime('%b %Y')}")
st.dataframe(cur_agg_all.style.format(fmt), use_container_width=True)

st.subheader(f"All Hunters — {last_start.strftime('%b %Y')}")
st.dataframe(last_agg_all.style.format(fmt), use_container_width=True)

# Export two CSVs (profit-only tables)
def _zip_reports(a: pd.DataFrame, b: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"profit_{cur_start.strftime('%b_%Y').lower()}.csv", a.to_csv(index=False))
        z.writestr(f"profit_{last_start.strftime('%b_%Y').lower()}.csv", b.to_csv(index=False))
    buf.seek(0); return buf.getvalue()

st.download_button(
    "Download Reports (ZIP)",
    data=_zip_reports(cur_agg_all, last_agg_all),
    file_name="staff_profit_reports.zip",
    mime="application/zip",
    use_container_width=True,
)
