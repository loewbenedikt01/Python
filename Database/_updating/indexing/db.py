"""
db.py: create, fill and query the platform database (DuckDB).

    pip install duckdb pandas pyarrow

    python Database/_updating/indexing/db.py --init   # create _database/platform.duckdb with all tables
    python Database/_updating/indexing/db.py --demo   # build _database/demo.duckdb with made-up data and run every query

In your loaders:
    from Database._updating.indexing.db import connect, upsert
    con = connect()
    upsert(con, "prices_daily", df)          # df columns = table column names (any subset, any order)

In the Streamlit app:
    from Database._updating.indexing.db import connect, company_profile, financials, ...
    con = connect(read_only=True)

corporate_id is the random 5-digit integer from corporate_ids.csv; new companies
get one through generate_id.assign_corporate_ids(), never from this module.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from Database._schema import corporate, filings, clinical_trials

DATABASE_DIR = Path(__file__).resolve().parents[2] / "_database"
DEFAULT_DB   = DATABASE_DIR / "platform.duckdb"
INTRADAY_DIR = DATABASE_DIR / "prices_intraday"

# order matters: views in later modules may read tables from earlier ones
SCHEMAS = [corporate.SCHEMA_SQL, filings.SCHEMA_SQL, clinical_trials.SCHEMA_SQL]


# --------------------------------------------------------------------------
# Setup and writing
# --------------------------------------------------------------------------

def connect(path: str | Path = DEFAULT_DB, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        for sql in SCHEMAS:
            con.execute(sql)
    return con


def upsert(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame, source: str | None = None) -> int:
    """Insert rows; rows with an existing primary key are replaced. Re-running a load is safe."""
    if df is None or df.empty:
        return 0
    df = df.copy()
    # Fill columns the frame leaves out with the table's default (needed when they are part of the key)
    key_cols = con.execute("""SELECT unnest(constraint_column_names) FROM duckdb_constraints()
                              WHERE table_name = ? AND constraint_type = 'PRIMARY KEY'""", [table]).fetchall()
    key_cols = {k[0] for k in key_cols}
    for name, default in con.execute(
            "SELECT column_name, column_default FROM duckdb_columns() WHERE table_name = ? AND column_default IS NOT NULL",
            [table]).fetchall():
        if name not in df.columns and name in key_cols:
            df[name] = con.execute(f"SELECT {default}").fetchone()[0]
    con.register("_incoming", df)
    try:
        con.execute(f"INSERT OR REPLACE INTO {table} BY NAME SELECT * FROM _incoming")
    finally:
        con.unregister("_incoming")
    con.execute("INSERT INTO load_log (source, table_name, rows, status) VALUES (?, ?, ?, 'ok')",
                [source, table, len(df)])
    return len(df)


def resolve(con, id_type: str, id_value: str, on: date | None = None) -> int | None:
    """Find corporate_id from an outside ID, e.g. resolve(con, 'TICKER', 'LLY') or ('CIK', '0000059478')."""
    on = on or date.today()
    row = con.execute("""
        SELECT corporate_id FROM company_identifiers
        WHERE id_type = ? AND id_value = ? AND valid_from <= ? AND (valid_to IS NULL OR valid_to >= ?)
        ORDER BY valid_from DESC LIMIT 1""", [id_type.upper(), id_value, on, on]).fetchone()
    return row[0] if row else None


def write_intraday(df: pd.DataFrame, interval: str = "1m", asset_class: str = "equity") -> None:
    """Append intraday bars (columns: instrument_id, ts, open, high, low, close, volume) to Parquet."""
    df = df.copy()
    df["year"], df["month"] = df["ts"].dt.year, df["ts"].dt.month
    out = INTRADAY_DIR / f"interval={interval}" / f"asset_class={asset_class}"
    df.to_parquet(out, partition_cols=["year", "month"], index=False)


# --------------------------------------------------------------------------
# Reading: one function per screen / panel
# --------------------------------------------------------------------------

def company_profile(con, corporate_id: int) -> pd.DataFrame:
    return con.execute("SELECT * FROM v_company_overview WHERE corporate_id = ?", [corporate_id]).df()


def daily_prices(con, instrument_ids: list[str], start: date | None = None) -> pd.DataFrame:
    return con.execute("""
        SELECT p.instrument_id, i.symbol, i.asset_class, p.date, p.close, p.adj_close, p.volume
        FROM prices_daily p JOIN instruments i USING (instrument_id)
        WHERE p.instrument_id IN (SELECT unnest(?)) AND p.date >= coalesce(?, DATE '1900-01-01')
        ORDER BY p.date""", [instrument_ids, start]).df()


def intraday_prices(con, instrument_id: str, start: str, end: str, interval: str = "1m") -> pd.DataFrame:
    pattern = str(INTRADAY_DIR / f"interval={interval}" / "**" / "*.parquet")
    return con.execute(f"""
        SELECT * FROM read_parquet('{pattern}', hive_partitioning = true)
        WHERE instrument_id = ? AND ts BETWEEN ? AND ? ORDER BY ts""", [instrument_id, start, end]).df()


def financials(con, corporate_id: int, period: str = "FY", years: int = 10) -> pd.DataFrame:
    """Income statement / balance sheet / cash flow; period = 'FY' or 'Q' for quarters."""
    cond = "fiscal_period = 'FY'" if period == "FY" else "fiscal_period LIKE 'Q%'"
    return con.execute(f"""
        SELECT * FROM v_financials WHERE corporate_id = ? AND {cond}
        ORDER BY period_end DESC LIMIT ?""", [corporate_id, years * (1 if period == 'FY' else 4)]).df()


def fundamentals_as_of(con, as_of: date, item_codes: list[str]) -> pd.DataFrame:
    """What was known on a past date (for backtests: no look-ahead)."""
    return con.execute("""
        SELECT * EXCLUDE (rn) FROM (
            SELECT f.*, row_number() OVER (PARTITION BY corporate_id, item_code
                                           ORDER BY period_end DESC, filed_date DESC) AS rn
            FROM fundamentals f
            WHERE filed_date <= ? AND fiscal_period = 'FY' AND item_code IN (SELECT unnest(?)))
        WHERE rn = 1""", [as_of, item_codes]).df()


def top_holders(con, corporate_id: int, report_date: date | None = None, n: int = 20) -> pd.DataFrame:
    return con.execute("""
        WITH d AS (SELECT coalesce(?, max(report_date)) AS rd FROM institutional_holdings WHERE corporate_id = ?)
        SELECT h.name, h.holder_type, i.symbol AS ticker, i.isin, ih.shares, ih.pct_of_shares_out,
               ih.value, ih.currency, ih.report_date
        FROM institutional_holdings ih
        JOIN d ON ih.report_date = d.rd
        JOIN holders h ON h.holder_id = ih.holder_id
        JOIN instruments i ON i.instrument_id = ih.instrument_id
        WHERE ih.corporate_id = ?
        ORDER BY ih.shares DESC LIMIT ?""", [report_date, corporate_id, corporate_id, n]).df()


def holdings_changes(con, corporate_id: int) -> pd.DataFrame:
    """Change in shares per holder between the two latest quarters."""
    return con.execute("""
        WITH q AS (SELECT DISTINCT report_date FROM institutional_holdings WHERE corporate_id = ?
                   ORDER BY report_date DESC LIMIT 2),
             cur AS (SELECT * FROM institutional_holdings WHERE corporate_id = ? AND report_date = (SELECT max(report_date) FROM q)),
             prev AS (SELECT * FROM institutional_holdings WHERE corporate_id = ? AND report_date = (SELECT min(report_date) FROM q))
        SELECT h.name,
               coalesce(prev.shares, 0) AS shares_before, coalesce(cur.shares, 0) AS shares_now,
               coalesce(cur.shares, 0) - coalesce(prev.shares, 0) AS change,
               CASE WHEN prev.shares IS NULL THEN 'new position'
                    WHEN cur.shares IS NULL THEN 'sold out'
                    WHEN cur.shares > prev.shares THEN 'added'
                    WHEN cur.shares < prev.shares THEN 'reduced' ELSE 'unchanged' END AS action
        FROM cur FULL OUTER JOIN prev USING (holder_id)
        JOIN holders h USING (holder_id)
        ORDER BY abs(change) DESC""", [corporate_id] * 3).df()


def executives_by_year(con, corporate_id: int) -> pd.DataFrame:
    return con.execute("""
        SELECT e.fiscal_year, p.full_name, e.title, e.is_board_member, e.since_date, e.total_compensation, e.currency
        FROM executives e JOIN people p USING (person_id)
        WHERE e.corporate_id = ? ORDER BY e.fiscal_year DESC, e.is_board_member, e.total_compensation DESC NULLS LAST
        """, [corporate_id]).df()


def year_over_year_changes(con, table: str, corporate_id: int, key_sql: str, label_sql: str) -> pd.DataFrame:
    """Who/what joined or left between consecutive fiscal years (executives, subsidiaries)."""
    return con.execute(f"""
        WITH s AS (SELECT fiscal_year, {key_sql} AS k, {label_sql} AS label FROM {table} WHERE corporate_id = ?),
             y AS (SELECT DISTINCT fiscal_year FROM s)
        SELECT y.fiscal_year, 'added' AS change, s.label FROM y JOIN s ON s.fiscal_year = y.fiscal_year
            WHERE y.fiscal_year > (SELECT min(fiscal_year) FROM s)
              AND s.k NOT IN (SELECT k FROM s s2 WHERE s2.fiscal_year = y.fiscal_year - 1)
        UNION ALL
        SELECT y.fiscal_year, 'removed', s.label FROM y JOIN s ON s.fiscal_year = y.fiscal_year - 1
            WHERE s.k NOT IN (SELECT k FROM s s2 WHERE s2.fiscal_year = y.fiscal_year)
        ORDER BY 1 DESC, 2, 3""", [corporate_id]).df()


def executive_changes(con, corporate_id: int) -> pd.DataFrame:
    return year_over_year_changes(
        con, "executives e JOIN people p USING (person_id)", corporate_id,
        "person_id || '|' || title", "full_name || ' - ' || title")


def subsidiary_changes(con, corporate_id: int) -> pd.DataFrame:
    return year_over_year_changes(
        con, "subsidiaries", corporate_id,
        "subsidiary_name || '|' || jurisdiction", "subsidiary_name || ' (' || jurisdiction || ')'")


def upcoming_events(con, days: int = 30, corporate_ids: list[int] | None = None) -> pd.DataFrame:
    return con.execute("""
        SELECT e.event_date, e.event_time, c.name, e.event_type, e.fiscal_year, e.fiscal_period, e.status
        FROM events e JOIN companies c USING (corporate_id)
        WHERE e.event_date BETWEEN current_date AND current_date + ?::INTEGER
          AND (? IS NULL OR e.corporate_id IN (SELECT unnest(?)))
        ORDER BY e.event_date, c.name""", [days, corporate_ids, corporate_ids]).df()


def screener(con, where_sql: str = "TRUE", params: list | None = None) -> pd.DataFrame:
    """Latest annual numbers + valuation per company; filter with SQL, e.g.
    screener(con, "sector = ? AND market_cap > ?", ['Health Care', 50e9])"""
    return con.execute(f"""
        WITH fy AS (SELECT * FROM v_financials WHERE fiscal_period = 'FY'
                    QUALIFY row_number() OVER (PARTITION BY corporate_id ORDER BY fiscal_year DESC) = 1),
             prev AS (SELECT corporate_id, fiscal_year, revenue FROM v_financials WHERE fiscal_period = 'FY')
        SELECT o.corporate_id, o.name, o.sector, o.industry, o.continent, o.market_cap,
               fy.fiscal_year, fy.revenue, fy.net_income, fy.employees,
               fy.gross_profit / nullif(fy.revenue, 0) * 100 AS gross_margin_pct,
               (fy.revenue / nullif(prev.revenue, 0) - 1) * 100 AS revenue_growth_pct,
               o.market_cap / nullif(fy.net_income, 0) AS pe,
               s.n_subsidiaries
        FROM v_company_overview o
        LEFT JOIN fy USING (corporate_id)
        LEFT JOIN prev ON prev.corporate_id = o.corporate_id AND prev.fiscal_year = fy.fiscal_year - 1
        LEFT JOIN v_subsidiary_counts s ON s.corporate_id = o.corporate_id AND s.fiscal_year = fy.fiscal_year
        WHERE {where_sql}
        ORDER BY o.market_cap DESC NULLS LAST""", params or []).df()


# --------------------------------------------------------------------------
# Demo data (MADE-UP numbers, only to show how the tables fit together)
# --------------------------------------------------------------------------

def load_demo(con) -> None:
    upsert(con, "countries", pd.DataFrame({
        "country_code": ["US", "DK", "DE"], "country_name": ["United States", "Denmark", "Germany"],
        "continent": ["North America", "Europe", "Europe"]}))
    upsert(con, "currencies", pd.DataFrame({"currency_code": ["USD", "DKK", "EUR"],
                                            "currency_name": ["US Dollar", "Danish Krone", "Euro"]}))
    upsert(con, "companies", pd.DataFrame([
        dict(corporate_id=10001, name="Eli Lilly", country_code="US", reporting_currency="USD",
             fiscal_year_end="12-31", sector="Health Care", industry="Pharmaceuticals", sic_code="2834",
             hq_city="Indianapolis", hq_state="IN", hq_country_code="US", business_description="[demo]"),
        dict(corporate_id=10002, name="Novo Nordisk", country_code="DK", reporting_currency="DKK",
             fiscal_year_end="12-31", sector="Health Care", industry="Pharmaceuticals", sic_code="2834",
             hq_city="Bagsvaerd", hq_country_code="DK", business_description="[demo]")]))
    upsert(con, "company_identifiers", pd.DataFrame({
        "corporate_id": [10001, 10001, 10002], "id_type": ["TICKER", "CIK", "TICKER"],
        "id_value": ["LLY", "0000059478", "NVO"]}))
    upsert(con, "instruments", pd.DataFrame([
        dict(instrument_id="EQ:LLY:XNYS", asset_class="equity", symbol="LLY", exchange="XNYS", currency="USD",
             corporate_id=10001, isin="US5324571083", is_primary_listing=True),
        dict(instrument_id="EQ:NVO:XNYS", asset_class="equity", symbol="NVO", exchange="XNYS", currency="USD",
             corporate_id=10002, share_class="ADR", is_primary_listing=False),
        dict(instrument_id="FX:EURUSD", asset_class="forex", symbol="EURUSD", currency="USD", base_currency="EUR")]))
    days = pd.bdate_range(start=date.today() - pd.Timedelta(days=14), end=date.today())[-5:]
    upsert(con, "prices_daily", pd.DataFrame({"instrument_id": "EQ:LLY:XNYS", "date": days.date,
                                              "close": [900, 905, 898, 910, 905.4], "source": "demo"}))
    upsert(con, "valuation_daily", pd.DataFrame({"corporate_id": [10001, 10002], "date": [days[-1].date()] * 2,
                                                 "market_cap": [812e9, 268e9], "currency": ["USD", "USD"]}))
    items = [("revenue", "Revenue", "IS", "currency", "duration"),
             ("gross_profit", "Gross profit", "IS", "currency", "duration"),
             ("net_income", "Net income", "IS", "currency", "duration"),
             ("total_assets", "Total assets", "BS", "currency", "instant"),
             ("employees", "Employees", "OTHER", "persons", "instant")]
    upsert(con, "financial_items", pd.DataFrame(items, columns=["item_code", "label", "statement", "unit_type", "period_type"]))
    rows = []
    for fy, rev, gp, ni, ta, emp in [(2023, 34.1e9, 27.0e9, 5.2e9, 64e9, 43000),
                                     (2024, 45.0e9, 36.6e9, 10.6e9, 78e9, 47000),
                                     (2025, 61.0e9, 50.0e9, 18.0e9, 95e9, 50000)]:
        for code, v in [("revenue", rev), ("gross_profit", gp), ("net_income", ni), ("total_assets", ta), ("employees", emp)]:
            rows.append(dict(corporate_id=10001, fiscal_year=fy, fiscal_period="FY", period_end=date(fy, 12, 31),
                             item_code=code, value=v, currency=None if code == "employees" else "USD",
                             filed_date=date(fy + 1, 2, 15), source="demo"))
    # a restatement: FY2024 revenue revised a year later; views pick the newest
    rows.append(dict(corporate_id=10001, fiscal_year=2024, fiscal_period="FY", period_end=date(2024, 12, 31),
                     item_code="revenue", value=45.1e9, currency="USD", filed_date=date(2026, 2, 15), source="demo"))
    upsert(con, "fundamentals", pd.DataFrame(rows))
    upsert(con, "holders", pd.DataFrame({"holder_id": ["H1", "H2", "H3"], "holder_type": "asset_manager",
                                         "name": ["Holder A Asset Mgmt", "Holder B Advisors", "Holder C Capital"]}))
    upsert(con, "institutional_holdings", pd.DataFrame({
        "corporate_id": 10001, "instrument_id": "EQ:LLY:XNYS",
        "report_date": [date(2026, 3, 31)] * 2 + [date(2026, 6, 30)] * 3,
        "holder_id": ["H1", "H2", "H1", "H2", "H3"],
        "shares": [70e6, 50e6, 72e6, 45e6, 10e6], "pct_of_shares_out": [7.8, 5.6, 8.0, 5.0, 1.1]}))
    upsert(con, "people", pd.DataFrame({"person_id": ["P1", "P2", "P3"], "full_name": ["Person A", "Person B", "Person C"]}))
    upsert(con, "executives", pd.DataFrame({
        "corporate_id": 10001, "fiscal_year": [2024, 2024, 2025, 2025],
        "person_id": ["P1", "P2", "P1", "P3"],
        "title": ["Chief Executive Officer", "Chief Financial Officer"] * 2}))
    upsert(con, "subsidiaries", pd.DataFrame({
        "corporate_id": 10001, "fiscal_year": [2024, 2024, 2025, 2025, 2025],
        "subsidiary_name": ["Sub One", "Sub Two", "Sub One", "Sub Three", "Sub Four"],
        "jurisdiction": ["Delaware", "Ireland", "Delaware", "Germany", "Japan"]}))
    today = date.today()
    upsert(con, "events", pd.DataFrame({
        "event_id": ["10001|earnings_release|2026|Q3", "10002|earnings_release|2026|Q3"],
        "corporate_id": [10001, 10002], "event_type": "earnings_release",
        "event_date": [today + pd.Timedelta(days=10), today + pd.Timedelta(days=20)],
        "event_time": ["BMO", "BMO"], "fiscal_year": 2026, "fiscal_period": "Q3",
        "status": ["confirmed", "estimated"], "source": "demo"}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true", help="create platform.duckdb with all tables")
    ap.add_argument("--demo", action="store_true", help="build demo.duckdb with sample data and run the queries")
    args = ap.parse_args()
    if args.init:
        connect().close()
        print(f"Created {DEFAULT_DB}")
    if args.demo:
        path = DATABASE_DIR / "demo.duckdb"
        path.unlink(missing_ok=True)
        con = connect(path)
        load_demo(con)
        pd.set_option("display.width", 200, "display.max_columns", 20)
        checks = {
            "Company profile": company_profile(con, resolve(con, "TICKER", "LLY"))[["name", "sector", "continent", "market_cap"]],
            "Annual financials (restated FY2024 revenue = 45.1B)": financials(con, 10001)[["fiscal_year", "revenue", "net_income", "employees"]],
            "Known on 2025-06-01 (point in time)": fundamentals_as_of(con, date(2025, 6, 1), ["revenue"])[["fiscal_year", "value"]],
            "Top holders": top_holders(con, 10001)[["name", "ticker", "isin", "shares", "pct_of_shares_out"]],
            "Holding changes": holdings_changes(con, 10001),
            "Executive changes": executive_changes(con, 10001),
            "Subsidiary counts": con.execute("SELECT * FROM v_subsidiary_counts").df(),
            "Subsidiary changes": subsidiary_changes(con, 10001),
            "Upcoming events": upcoming_events(con, 30),
            "Screener: Health Care, mcap > $100B": screener(con, "sector = ? AND market_cap > ?", ["Health Care", 100e9])[
                ["name", "market_cap", "revenue_growth_pct", "gross_margin_pct", "pe", "n_subsidiaries"]],
        }
        for title, df in checks.items():
            print(f"\n=== {title} ===\n{df.to_string(index=False)}")