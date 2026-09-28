"""
Markets Dashboard: combines the pipeline's DuckDB databases for equities, indices,
forex, crypto, commodities, bonds, ETFs, sectors and sentiment into one simple interface.

Run:
    pip install streamlit pandas plotly pyarrow duckdb
    streamlit run interface.py

Reads from the pipeline's data folder (set in the sidebar), read-only:

    Database/data/
      prices.duckdb       prices_daily (ticker, date, close, currency ...), instruments (name, asset class),
                          load_log (last run per step)
      companies.duckdb    company_info (company names), load_log
      raw.duckdb          load_log

Bond yield indices (^IRX, ^FVX, ^TNX, ^TYX) are split out into their own "Yields"
class so that their changes can be shown in basis points while the bond ETFs show
% returns. Prices quoted in minor units (London GBp, ...) are converted to the
major unit (GBP).

Connections are opened read-only, queried and closed at once; only the resulting
data is cached (st.cache_data), so an open app never blocks the nightly pipeline
run. While `python run.py` is writing, the databases are locked; the app then
shows a message instead of data (not cached, the next reload tries again).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))      # company_page / company_data next to this file
import company_page
import headquarters_page

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DEFAULT_FOLDER = Path(__file__).resolve().parents[2] / "Database" / "data"

# instruments.asset_class (the _tickers file name) -> display class ("Yields" is split out of bond)
CLASS_MAP = {
    "equities": "Equities",
    "indices": "Indices",
    "forex": "Forex",
    "crypto": "Crypto",
    "commodities": "Commodities",
    "bond": "Bonds",
    "etfs": "ETFs",
    "sectors": "Sectors",
    "sentiment": "Sentiment",
}
ASSET_CLASSES = ["Equities", "Indices", "Forex", "Crypto", "Commodities", "Bonds", "Yields",
                 "ETFs", "Sectors", "Sentiment"]
CLASS_DTYPE = pd.CategoricalDtype(ASSET_CLASSES)
# bond yield indices, quoted in % (= FRED_BOND_MAP keys in Database/config.py)
YIELD_TICKERS = {"^IRX", "^FVX", "^TNX", "^TYX"}
# quote units in 1/100 of the currency (= MINOR_UNITS in Database/config.py)
MINOR_UNITS = {"GBp": 100, "GBX": 100, "ZAc": 100, "ILA": 100, "USX": 100}
LOCKED = "locked"
EMPTY = pd.DataFrame(columns=["date", "symbol", "close", "asset_class"])


class DatabaseLocked(Exception):
    """The pipeline is writing; raised (not returned) so that st.cache_data doesn't cache it."""

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _connect(path: Path) -> duckdb.DuckDBPyConnection | None:
    """Read-only connection; None if the file is missing. Raises DatabaseLocked while the pipeline writes."""
    if not path.exists():
        return None
    try:
        return duckdb.connect(str(path), read_only=True)
    except duckdb.IOException as exc:
        raise DatabaseLocked(path.name) from exc


def _last_runs(con: duckdb.DuckDBPyConnection, database: str) -> pd.DataFrame:
    df = con.execute("""
        SELECT step, finished_at, status, rows_written, n_failed FROM load_log
        QUALIFY row_number() OVER (PARTITION BY step ORDER BY finished_at DESC) = 1""").df()
    return df.assign(database=database)


@st.cache_data(show_spinner="Loading database…")
def load_database(folder: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Return (long dataframe date | symbol | close | asset_class, status, ticker -> name).

    Every connection is closed before returning; only the data is cached. The status
    frame has one row per pipeline step (last run), or a single row with status
    'missing'. Raises DatabaseLocked while the pipeline writes to prices.duckdb.
    """
    root = Path(folder).expanduser()
    con = _connect(root / "prices.duckdb")
    if con is None:
        return EMPTY, pd.DataFrame([{"database": "prices.duckdb", "status": "missing"}]), {}

    units = pd.DataFrame(list(MINOR_UNITS.items()), columns=["unit", "divisor"])
    try:
        con.register("_units", units)
        raw = con.execute("""
            SELECT p.date, p.ticker AS symbol, p.close / coalesce(u.divisor, 1) AS close, i.asset_class
            FROM prices_daily p
            JOIN instruments i USING (ticker)
            LEFT JOIN _units u ON u.unit = p.currency
            WHERE NOT coalesce(i.is_helper, FALSE)                 -- FX pairs added only for USD conversion
              AND coalesce(i.status, 'active') <> 'removed'
              AND NOT ends_with(i.ticker, '_FRED')                -- FRED copies of the yield indices
              AND NOT coalesce(p.price_suspect, FALSE)            -- one-day price spikes (Yahoo glitches)
              AND p.close IS NOT NULL""").df()
        names = dict(con.execute("SELECT ticker, name FROM instruments WHERE name IS NOT NULL").fetchall())
        status = [_last_runs(con, "prices.duckdb")]
        ids = dict(con.execute("SELECT ticker, corporate_id FROM instruments WHERE corporate_id IS NOT NULL").fetchall())
    finally:
        con.close()

    # company names and the other databases' runs; skipped if they are locked or missing
    for db in ("companies.duckdb", "raw.duckdb"):
        try:
            other = _connect(root / db)
        except DatabaseLocked:
            status.append(pd.DataFrame([{"database": db, "status": LOCKED}]))
            continue
        if other is None:
            continue
        try:
            status.append(_last_runs(other, db))
            if db == "companies.duckdb":
                companies = dict(other.execute("SELECT corporate_id, name FROM company_info WHERE name IS NOT NULL").fetchall())
                names.update({t: companies[c] for t, c in ids.items() if c in companies})
        finally:
            other.close()

    df = pd.DataFrame({
        "date": pd.to_datetime(raw["date"]),
        "symbol": raw["symbol"].astype(str),
        "close": raw["close"].astype(float),
        "asset_class": pd.Series(raw["asset_class"].map(CLASS_MAP), dtype=CLASS_DTYPE),
    })
    df.loc[df["symbol"].isin(YIELD_TICKERS), "asset_class"] = "Yields"
    df = df.dropna(subset=["date", "close", "asset_class"])

    # A shared categorical keeps ~10M ticker strings from being stored as Python objects
    df["symbol"] = df["symbol"].astype(pd.CategoricalDtype(sorted(df["symbol"].unique())))
    return df.reset_index(drop=True), pd.concat(status, ignore_index=True), names


@st.cache_data
def demo_data() -> pd.DataFrame:
    """Random-walk sample data so the app can be tried without real files."""
    rng = np.random.default_rng(7)
    end = pd.Timestamp.today().normalize()
    spec = {
        "Equities": {"AAPL": 230, "MSFT": 510, "LLY": 900, "NVDA": 180, "SAP": 240},
        "Indices": {"S&P 500": 6600, "Nasdaq 100": 24000, "DAX": 23500, "Euro Stoxx 50": 5400, "Nikkei 225": 44000},
        "Forex": {"EURUSD": 1.17, "USDJPY": 148.0, "GBPUSD": 1.34, "USDCHF": 0.80},
        "Crypto": {"BTC": 110000, "ETH": 4200, "SOL": 210},
        "Commodities": {"Gold": 3700, "Brent": 67, "Copper": 4.6, "Silver": 44},
        "Yields": {"US 2Y": 3.6, "US 10Y": 4.1, "DE 10Y": 2.7, "UK 10Y": 4.7},
    }
    vol = {"Equities": 0.018, "Indices": 0.011, "Forex": 0.005, "Crypto": 0.035, "Commodities": 0.015, "Yields": 0.02}
    rows = []
    for cls, instruments in spec.items():
        freq = "D" if cls == "Crypto" else "B"
        dates = pd.date_range(end=end, periods=(3 * 365 if freq == "D" else 3 * 260), freq=freq)
        for sym, last in instruments.items():
            steps = rng.normal(0.0002, vol[cls], len(dates))
            path = np.exp(np.cumsum(steps))
            path = path / path[-1] * last
            rows.append(pd.DataFrame({"date": dates, "symbol": sym, "close": path, "asset_class": cls}))
    return pd.concat(rows, ignore_index=True)


def get_data(folder: str, use_demo: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    if use_demo:
        return demo_data(), pd.DataFrame(), {}
    try:
        return load_database(folder)
    except DatabaseLocked:                          # run.py is writing
        return EMPTY, pd.DataFrame([{"database": "prices.duckdb", "status": LOCKED}]), {}


# --------------------------------------------------------------------------
# Calculations
# --------------------------------------------------------------------------
PERIODS = {
    "1D": None, "1W": pd.DateOffset(weeks=1),
    "1M": pd.DateOffset(months=1),
    "YTD": "ytd", "1Y": pd.DateOffset(years=1)
}


def is_yield(cls: str, yields_mode: bool) -> bool:
    return cls == "Yields" and yields_mode


def snapshot(data: pd.DataFrame, cls: str, yields_mode: bool, names: dict[str, str]) -> pd.DataFrame:
    """Latest value and changes over standard periods for one asset class."""
    out = []
    for sym, g in data[data["asset_class"] == cls].groupby("symbol", observed=True):
        s = g.set_index("date")["close"].sort_index()
        if s.empty:
            continue
        last_date, last = s.index[-1], s.iloc[-1]
        row = {"Instrument": sym, "Name": names.get(sym, ""), "Last": last, "As of": last_date.date()}
        for label, off in PERIODS.items():
            if off is None:
                ref = s.iloc[-2] if len(s) > 1 else np.nan
            elif off == "ytd":
                prior = s[s.index < pd.Timestamp(last_date.year, 1, 1)]
                ref = prior.iloc[-1] if len(prior) else np.nan
            else:
                ref = s.asof(last_date - off) if last_date - off >= s.index[0] else np.nan
            if is_yield(cls, yields_mode):
                row[label] = (last - ref) * 100  # basis points
            else:
                row[label] = (last / ref - 1) * 100 if ref else np.nan
        out.append(row)
    return pd.DataFrame(out)


@st.cache_data(show_spinner="Computing snapshots…")
def get_snapshots(folder: str, use_demo: bool, yields_mode: bool) -> dict[str, pd.DataFrame]:
    data, _, names = get_data(folder, use_demo)
    return {cls: snapshot(data, cls, yields_mode, names) for cls in ASSET_CLASSES}


def colour(v):
    if pd.isna(v) or v == 0:
        return ""
    return "color: #1a9850" if v > 0 else "color: #d73027"


def show_snapshot(snap: pd.DataFrame, cls: str, yields_mode: bool, key: str | None = None):
    """Snapshot table; with a key, equities rows are clickable and open the company page."""
    unit = "bp" if is_yield(cls, yields_mode) else "%"
    fmt = "%+.1f bp" if unit == "bp" else "%+.2f%%"
    cfg = {"Last": st.column_config.NumberColumn(format="%.4f" if cls == "Forex" else "%,.2f")}
    cfg.update({p: st.column_config.NumberColumn(f"{p} ({unit})", format=fmt) for p in PERIODS})
    styled = snap.style.map(colour, subset=list(PERIODS))
    if key and cls == "Equities":
        event = st.dataframe(styled, column_config=cfg, hide_index=True, width="stretch", key=key,
                             on_select="rerun", selection_mode="single-row")
        st.caption("Click a row to open the company page.")
        if event.selection.rows:
            st.session_state["company"] = str(snap.iloc[event.selection.rows[0]]["Instrument"])
            st.switch_page(COMPANY)
    else:
        st.dataframe(styled, column_config=cfg, hide_index=True, width="stretch")


def select(data: pd.DataFrame, cls: str, symbols: list[str]) -> pd.DataFrame:
    """Rows for the given instruments of one asset class, with plain string columns."""
    sub = data[(data["asset_class"] == cls) & data["symbol"].isin(symbols)]
    return sub.astype({"symbol": str, "asset_class": str})


def window(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if df.empty or period == "All":
        return df
    end = df["date"].max()
    start = {"1M": end - pd.DateOffset(months=1), "3M": end - pd.DateOffset(months=3),
             "6M": end - pd.DateOffset(months=6), "YTD": pd.Timestamp(end.year, 1, 1),
             "1Y": end - pd.DateOffset(years=1), "5Y": end - pd.DateOffset(years=5)}[period]
    return df[df["date"] >= start]


def rebase(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    df["value"] = df.groupby(col)["close"].transform(lambda s: s / s.iloc[0] * 100)
    return df


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------
def markets_page() -> None:
    folder = st.session_state["folder"]
    with st.sidebar:
        use_demo = st.checkbox("Use demo data instead", value=False)
        yields_mode = st.checkbox("Show yield changes in bp", value=True)

    data, log, names = get_data(folder, use_demo)

    st.title("Markets Dashboard")

    if data.empty:
        if not log.empty and (log["status"] == LOCKED).any():
            st.info("The database is being updated right now (`python run.py` is writing to it). "
                    "Try **Reload data** in the sidebar in a few minutes.")
        else:
            st.warning(f"No data found in `{folder}`. Expected the pipeline's `prices.duckdb` there "
                       "(run `python run.py` in the Database folder), or tick **Use demo data** in the sidebar.")
        if not log.empty:
            st.dataframe(log, hide_index=True, width="stretch")
        st.stop()

    snaps = get_snapshots(folder, use_demo, yields_mode)
    classes = [c for c in ASSET_CLASSES if not snaps[c].empty]


    def label(sym: str) -> str:
        return f"{sym} · {names[sym]}" if names.get(sym) else sym


    # Summary tiles
    cols = st.columns(len(classes))
    for col, cls in zip(cols, classes):
        col.metric(f"{cls} · instruments", len(snaps[cls]))
        col.caption(f"Data to {max(snaps[cls]['As of']):%d %b %Y}")

    tabs = st.tabs(["Overview"] + classes + ["Compare", "Status"])

    # Overview: biggest movers across all non-yield instruments
    with tabs[0]:
        movers = pd.concat([snaps[c].assign(Class=c) for c in classes if not is_yield(c, yields_mode)],
                           ignore_index=True) if any(not is_yield(c, yields_mode) for c in classes) else pd.DataFrame()
        if not movers.empty:
            left, right = st.columns(2)
            show = ["Class", "Instrument", "Name", "Last", "1D", "1M", "YTD"]
            fmt = {p: st.column_config.NumberColumn(f"{p} (%)", format="%+.2f%%") for p in ["1D", "1M", "YTD"]}
            fmt["Last"] = st.column_config.NumberColumn(format="%,.2f")
            with left:
                st.subheader("Top gainers (1D)")
                st.dataframe(movers.nlargest(8, "1D")[show].style.map(colour, subset=["1D", "1M", "YTD"]),
                             column_config=fmt, hide_index=True, width="stretch")
            with right:
                st.subheader("Top losers (1D)")
                st.dataframe(movers.nsmallest(8, "1D")[show].style.map(colour, subset=["1D", "1M", "YTD"]),
                             column_config=fmt, hide_index=True, width="stretch")
        for cls in classes:
            st.subheader(cls)
            show_snapshot(snaps[cls], cls, yields_mode)

    # One tab per asset class
    for tab, cls in zip(tabs[1:1 + len(classes)], classes):
        with tab:
            show_snapshot(snaps[cls], cls, yields_mode, key=f"snap_{cls}")
            symbols = sorted(snaps[cls]["Instrument"])
            c1, c2, c3 = st.columns([3, 2, 1])
            picked = c1.multiselect("Instruments", symbols, default=symbols[:3], format_func=label, key=f"sel_{cls}")
            period = c2.radio("Period", ["1M", "3M", "6M", "YTD", "1Y", "5Y", "All"], index=4,
                              horizontal=True, key=f"per_{cls}")
            rb = c3.checkbox("Rebase to 100", value=len(picked) > 1 and not is_yield(cls, yields_mode), key=f"rb_{cls}")
            if picked:
                chart = window(select(data, cls, picked), period)
                chart = rebase(chart, "symbol") if rb else chart.assign(value=chart["close"])
                y_label = "Rebased (start = 100)" if rb else ("Yield (%)" if cls == "Yields" else "Price")
                fig = px.line(chart, x="date", y="value", color="symbol", labels={"value": y_label, "date": ""})
                fig.update_layout(height=450, legend_title_text="", hovermode="x unified", margin=dict(t=20))
                st.plotly_chart(fig, width="stretch")

    # Compare across asset classes
    with tabs[1 + len(classes)]:
        options = [f"{cls}: {sym}" for cls in classes for sym in sorted(snaps[cls]["Instrument"])]
        defaults = [f"{cls}: {sorted(snaps[cls]['Instrument'])[0]}" for cls in classes[:4]]
        picked = st.multiselect("Instruments from any asset class", options, default=defaults,
                                format_func=lambda o: f"{o.split(': ', 1)[0]}: {label(o.split(': ', 1)[1])}")
        period = st.radio("Period", ["1M", "3M", "6M", "YTD", "1Y", "5Y", "All"], index=4, horizontal=True, key="per_cmp")
        if picked:
            sel = pd.concat([select(data, cls, [sym]).assign(label=f"{cls}: {sym}")
                             for cls, sym in (p.split(": ", 1) for p in picked)], ignore_index=True)
            sel = window(sel, period)
            fig = px.line(rebase(sel, "label"), x="date", y="value", color="label",
                          labels={"value": "Rebased (start = 100)", "date": ""})
            fig.update_layout(height=450, legend_title_text="", hovermode="x unified", margin=dict(t=20))
            st.plotly_chart(fig, width="stretch")

            if len(picked) > 1:
                st.subheader("Correlation of daily changes")
                wide = sel.pivot_table(index="date", columns="label", values="close").sort_index().ffill()
                changes = pd.DataFrame({
                    c: wide[c].diff() if is_yield(c.split(": ")[0], yields_mode) else wide[c].pct_change()
                    for c in wide.columns})
                corr = changes.corr()
                heat = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto")
                heat.update_layout(height=420, margin=dict(t=20))
                st.plotly_chart(heat, width="stretch")
                st.caption("Uses days when every selected market has a price (crypto weekends are forward-filled).")

            st.download_button("Download selection (CSV)", sel.to_csv(index=False),
                               file_name="market_data_selection.csv", mime="text/csv")

    # Pipeline status: last run per step, instruments per class
    with tabs[-1]:
        if log.empty:
            st.info("Showing demo data.")
        else:
            st.subheader("Last pipeline run per step")
            st.dataframe(log, hide_index=True, width="stretch")
            st.subheader("Instruments per class")
            counts = (data.groupby("asset_class", observed=True)
                      .agg(rows=("close", "size"), instruments=("symbol", "nunique"), data_to=("date", "max"))
                      .reset_index())
            st.dataframe(counts, hide_index=True, width="stretch")


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
st.set_page_config(page_title="Markets Dashboard", layout="wide")

with st.sidebar:
    st.header("Data")
    st.session_state["folder"] = st.text_input("Database folder", value=str(DEFAULT_FOLDER), key="folder_input")
    if st.button("Reload data"):
        st.cache_data.clear()

MARKETS = st.Page(markets_page, title="Markets", icon=":material/show_chart:", default=True)
COMPANY = st.Page(company_page.render_page, title="Company", icon=":material/apartment:", url_path="company")
HEADQUARTERS = st.Page(headquarters_page.render_page, title="Headquarters", icon=":material/public:",
                       url_path="headquarters")
headquarters_page.COMPANY_PAGE = COMPANY           # a click on the globe opens the company page
st.navigation([MARKETS, COMPANY, HEADQUARTERS]).run()
