# Task: Build a data pipeline for my market-data platform (Python + DuckDB + Parquet)

## Context
I am building a financial data platform (similar to S&P Capital IQ). Restructure all my data storage into a clean, incremental pipeline controlled from **one entry script**.

You already read my files in the previous step (summary below). The open questions from that step are answered in **"Decisions"** at the end of this prompt. If anything is still unclear, ask before building. Implement in steps, test each step with a few tickers (`--tickers`), ask me before starting each step, and tell me afterwards what changed.

## Existing files (do not modify)
- `_tickers/`: one Python file per asset class. The **file name defines the asset class**. Each file holds dicts of the form `ticker: name` (Yahoo Finance ticker format):

  | File | Dicts (count) | Source |
  |---|---|---|
  | `bond.py` | `ticker_bonds` (16): 4 CBOE yield indices `^IRX ^FVX ^TNX ^TYX` + 12 bond ETFs (TLT, IEF, …) | yfinance (+ FRED, see below) |
  | `commodities.py` | `ticker_commodities` (93) | yfinance |
  | `crypto.py` | `ticker_crypto` (3) | yfinance |
  | `equities.py` | `ticker_us` 434, `ticker_de` 151, `ticker_asia` 499, `ticker_europe` 239, `ticker_rotw` 103 (1,426 total) | yfinance |
  | `forex.py` | `ticker_forex` (73) | yfinance |
  | `indices.py` | `ticker_indices` (81) | yfinance |
  | `macro.py` | `ticker_macro` (100 FRED series, e.g. `FEDFUNDS`) | FRED |
  | `sectors.py` | `ticker_sectors` (40 sector ETFs) | yfinance |
  | `sentiment.py` | `ticker_sentiment` (24, e.g. `^VIX`) | yfinance |

- `api_keys.txt`: `FRED_API_KEY`, `ALPACA_KEY_ID`, `ALPACA_SECRET`, `FINNHUB`, `NAME`, `EMAIL` (NAME + EMAIL = SEC EDGAR User-Agent). SEC EDGAR and ClinicalTrials.gov need no key. Read keys only from this file. **Never hardcode or print keys.**
- `Database/_database/corporate_ids.csv`: existing 5-digit corporate IDs (see Decisions).
- `Platform/interface/interface.py`: my Streamlit app. It still reads the old parquet files. **Don't change it.** At the end, list what has to change so it reads the new databases.
- The Thesis project has its own `_database` and must not be touched.

## Project location
Build everything inside `Database/`: `Database/run.py`, `config.py`, `pipeline/*.py`, `README.md`, `requirements.txt`; all data in `Database/data/`. The old folders `_updating/`, `_schema/`, `_controlling/` and `_database/` stay until the new pipeline works. **Ask me before deleting anything.**

## Company ID: `corporate_id`
- `corporate_id` is a **random 5-digit number** per company (e.g. `48213`). It connects all files.
- Stored in **`Database/data/corporate_ids.csv`**, the single source of truth (columns: corporate_id, company_name, cik, tickers, assigned_at). Once assigned, an ID **never changes and is never reused**. New companies get a new random, unused 5-digit ID.
- **One ID per company:** tickers of the same company (e.g. GOOGL + GOOG) share one ID. Group by SEC CIK where the company files with the SEC, otherwise by company name.
- Keep the existing IDs from the old `corporate_ids.csv` for companies that already have one (see Decisions for merged share classes).

## Target storage (all in `Database/data/`)

### 1. `prices.duckdb`: daily prices, all asset classes
- **`instruments`**: one row per ticker: ticker (PK), name/description (from my dict), asset_class (file name), group_name (dict name, e.g. `ticker_us`), currency, isin, corporate_id (equities only, else NULL), source, first_date, last_date.
- **`prices_daily`**: one row per ticker per day. Primary key (ticker, date). Columns:
  - date, **open, high, low, close, volume, adj_close**, dividends, stock_splits
  - **asset_class**
  - currency, **fx_to_usd** (rate used), **open_usd, high_usd, low_usd, close_usd, adj_close_usd**
  - source, updated_at
- Start date: **1995-01-01** (config setting). Source: **yfinance** for all asset classes: equities (US and non-US), bonds, commodities, crypto, forex, indices, sectors, sentiment.
- **USD prices:** for every instrument not quoted in USD, convert using the daily FX rate from the forex data (yfinance, e.g. `EURUSD=X`, `JPY=X`). If a needed currency pair is missing from `forex.py`, download it automatically as a helper series (flagged as helper in `instruments`). Handle **minor units**: London prices in GBp (pence) ÷ 100, ZAc ÷ 100, ILA ÷ 100. Use the last available FX rate if a date is missing (forward fill, max 5 days). When FX history is re-downloaded, recalculate the affected USD columns.
- **Bonds:** download all 16 bond tickers from yfinance. **Also** check FRED for every bond ticker (mapping in `config.py`, e.g. `^IRX`→`DTB3`, `^FVX`→`DGS5`, `^TNX`→`DGS10`, `^TYX`→`DGS30`). Where a FRED series exists, store it as its **own instrument** with ticker `<ticker>_FRED` (e.g. `^TNX_FRED`), with the FRED series id in the description (e.g. "US 10-Year Treasury Yield (FRED: DGS10)"). Only close is filled; open/high/low/volume stay empty. Bond ETFs without a FRED equivalent are skipped silently. Only `^` yield tickers without a mapping are logged.
- **`macro_observations`** (series_id, date, value, PK (series_id, date)) + **`macro_series`** (series_id, title, frequency, units, seasonal_adjustment, last_updated): the 100 series from `macro.py`, from FRED, updated incrementally.
- **`market_cap_daily`**: one row per company per day: corporate_id, date, shares_outstanding, market_cap (local currency), market_cap_usd, shares_source. PK (corporate_id, date).
  - Market cap = total shares outstanding (all share classes) × close of the company's primary listing.
  - Shares: from SEC filings (cover page `dei:EntityCommonStockSharesOutstanding`, carried forward until the next filing) for SEC filers. Otherwise from yfinance (`sharesOutstanding`, current value only; mark `shares_source = 'yfinance_current'`).
- Technical indicators are **not** part of this task (later).

### 2. `companies.duckdb`: company data, **wide design** (one column per item)
Only tickers from `equities.py`. **All 1,426** get a row in `company_info`. Only **US companies** (`ticker_us`) get full quarterly + yearly fundamentals. For **non-US companies**, check whether they exist in SEC EDGAR (20-F / 40-F filers). If they do, fill **only the yearly table** from their annual report; quarterly stays empty.

- **`company_info`**: one row per company. Columns: corporate_id (PK), primary_ticker, tickers (list), name, legal_name, cik, **isin** (from yfinance; empty if missing), cusip (derived from US ISINs, characters 3–11), sec_filer (bool), sec_form_type (10-K / 20-F / 40-F), exchange, country, continent (derived from country), trading_currency, reporting_currency, sector, industry, sic_code, sic_description, business_description, hq_street, hq_city, hq_state, hq_postal_code, hq_country, website, fiscal_year_end, employees (latest), updated_at.
  - Sources: SEC EDGAR submissions (legal name, CIK, SIC, address, fiscal year end), yfinance `.info` (ISIN, sector, industry, description, country, currency, employees, website), Finnhub company profile as a fallback.
- **`fundamentals_quarterly`**: one row per company per fiscal quarter, PK (corporate_id, fiscal_year, fiscal_quarter).
- **`fundamentals_yearly`**: one row per company per fiscal year, PK (corporate_id, fiscal_year).

**Financial columns (same ~40 items in both tables)** from SEC EDGAR XBRL company facts (us-gaap tags; ifrs-full tags for 20-F filers). Each item gets a list of alternative XBRL tags, tried in order:
- Income statement: revenue, cost_of_revenue, gross_profit, rnd_expense, sga_expense, operating_expenses, operating_income, interest_expense, pretax_income, income_tax, net_income, eps_basic, eps_diluted, shares_basic_wavg, shares_diluted_wavg, depreciation_amortization, ebitda (derived)
- Balance sheet: cash, short_term_investments, receivables, inventory, current_assets, ppe_net, goodwill, intangibles, total_assets, accounts_payable, current_liabilities, short_term_debt, long_term_debt, total_debt (derived), net_debt (derived), total_liabilities, total_equity, retained_earnings
- Cash flow: operating_cash_flow, capex, free_cash_flow (derived), investing_cash_flow, financing_cash_flow, dividends_paid, share_buybacks, stock_based_comp
- Other: shares_outstanding (cover page, sum of all classes), dividends_per_share, employees
- Margins (derived): gross_margin, operating_margin, net_margin

Also in each row: period_start, period_end, filed_date, form_type, accession_no, currency, updated_at.

Rules:
- US companies: 3 quarterly reports (10-Q) + 1 annual report (10-K). **Q4 = full year − Q1–Q3** for income statement and cash flow. Q4 balance sheet = 10-K. Leave per-share and weighted-share items empty for derived Q4. Column `q4_derived` (bool).
- 10-Q cash flows are often year-to-date: convert to single-quarter values (YTD − previous YTD).
- Values **as originally reported** in each filing (point-in-time). Ignore amendments unless the original is missing.
- History: as far back as XBRL exists (~2009).

**Extra columns** (DuckDB list/struct columns where there are several entries):
- **Earnings calendar:** `earnings_date` (past: SEC 8-K item 2.02; upcoming: Finnhub earnings calendar) and `earnings_date_status` (estimated / confirmed / reported). Create a row for the **next** quarter as soon as its date is known; its financial columns stay empty until the report is filed. Yearly: `annual_report_date` (10-K filing date).
- **Top holders** (quarterly and yearly): list of {holder_name, holder_cik, shares, value_usd, pct_of_shares_out, change_shares_vs_prev_quarter}, top 20. Source: SEC Form 13F quarterly bulk data sets. Match companies via **CUSIP derived from the yfinance ISIN**; fallback: issuer-name matching. Log unmatched companies. Map the 13F report date to the nearest fiscal quarter. Yearly = holders at fiscal year end. Default 4 quarters on the first run.
- **Executives** (yearly): list of {name, cik, title, is_officer, is_director}. Source: SEC insider transactions data sets (Form 3/4/5 bulk files): officers and directors who filed during the fiscal year. Default: last 5 years.
- **Subsidiaries** (yearly): list of {name, jurisdiction} + `n_subsidiaries`. Source: 10-K Exhibit 21 (20-F: Exhibit 8). Parse HTML tables and plain-text lists. Default: last 5 years. Changes are visible by comparing yearly rows.

### 3. `raw.duckdb`: raw downloads (keep this part small; more tables later)
- **`clinical_trials_raw`**: columns in this order: **nct_id, raw_json (JSON), last_change_date**, then version, downloaded_at, json_hash, is_latest, change_flag, changed_sections. PK (nct_id, version).
- When a trial changes (newer last_change_date or different hash), **keep the old version**, insert a new version, set `change_flag = TRUE`, and record which top-level sections changed.
- Source: ClinicalTrials.gov API v2. For now: refresh stored trials, plus an optional list of sponsor names / NCT IDs in the config (empty by default).

### 4. `intraday/`: Parquet, US stocks only
- Only the 434 tickers in `ticker_us`.
- **Download 1-minute bars from Alpaca** (try the `sip` feed for data older than 15 minutes; fall back to `iex` if refused). **Calculate 5m, 10m and 1h from the 1m bars.**
- One folder per timeframe: `intraday/1m/`, `5m/`, `10m/`, `1h/`. One file per ticker per year (`ticker=AAPL/2026.parquet`), readable with DuckDB `read_parquet(..., hive_partitioning=true)`.
- Columns: ticker, ts (UTC), open, high, low, close, volume, vwap, trade_count.
- History: **1 year** on the first run (configurable). Regular trading hours only by default (configurable). Split-adjusted; if a split is detected, rebuild that ticker's intraday history.

## Incremental updates
Every run checks what is stored and downloads only what is new:
- **Per ticker, not per table:** last stored date (prices) or timestamp (intraday) per ticker. A newly added ticker gets its full history.
- **Adjusted close:** re-download a ~5-day overlap. If stored adj_close differs (dividend/split), re-download that ticker's full history and recalculate its USD columns.
- Macro: per series, from the last stored date (FRED revises values, so re-fetch a 60-day overlap).
- Fundamentals: reprocess a company only when SEC has a new 10-Q / 10-K / 20-F / 8-K since the last stored filing date. Then recalculate `market_cap_daily` for that company from the new filing date.
- Holders / executives: only bulk data sets not processed yet (cached in `data/cache/`).
- Clinical trials: only trials changed since the latest `last_change_date`.
- Company info: refresh when older than 30 days (config).
- All writes are **upserts on the primary key**: re-running never creates duplicates.

## Control: one entry script
- `python run.py`: all steps in order: sync tickers + corporate IDs → company info → daily prices (incl. FX and USD columns) → FRED bonds + macro → fundamentals → market cap → earnings calendar → holders → executives → subsidiaries → intraday → clinical trials
- `python run.py prices intraday`: only the named steps
- `python run.py --tickers AAPL MSFT`: limit to some tickers (testing)
- `python run.py status`: per database/table: row count, latest date, last successful run, number of failed tickers
- `config.py`: all settings in one place: paths, start dates, history lengths, intraday intervals, regular-hours switch, FRED bond mapping, top-N holders, refresh days, trial sponsors, rate limits.
- A `load_log` table in each database: step, start/end time, rows written, status, error message.

## Technical requirements
- Python 3.11+, `duckdb`, `pandas`, `pyarrow`, `requests`, `yfinance`; `requirements.txt`.
- API limits: SEC max 10 requests/s with User-Agent "NAME EMAIL"; Alpaca 200/min; Finnhub 60/min; FRED limits; retries with backoff; batch yfinance downloads.
- One failing ticker or company must not stop the run: log it and continue. Print a summary per step at the end (updated / skipped / failed).
- Modules: `pipeline/common.py`, `schema.py`, `ids.py`, `prices.py`, `fx.py`, `fred.py`, `marketcap.py`, `intraday.py`, `companies.py`, `fundamentals.py`, `calendar.py`, `holders.py`, `executives.py`, `subsidiaries.py`, `trials.py`.
- `README.md`: how to run, what each table contains, example DuckDB queries (incl. joining databases via `ATTACH` on `corporate_id`).
- Tests with sample data for: XBRL → wide columns, Q4 and YTD derivation, USD conversion incl. GBp, corporate-ID stability (re-running never changes IDs), resampling, trial versioning.

## Expected first-run size (OK)
- Intraday: ~40M rows, ~20–30 min.
- 13F: ~50–100 MB per quarter × 4. Insider data: 5 years of quarterly files. All cached in `data/cache/`.

## Decisions (answers to your questions)
1. Location: inside `Database/` as you suggested; ask before deleting old folders.
2. corporate_id: random 5-digit, stored in `data/corporate_ids.csv`, never changes. One ID per company (grouped by CIK, else name).
3. Macro via FRED into `macro_observations`; sectors and sentiment are normal asset classes in `prices.duckdb`.
4. Market cap: calculated (`market_cap_daily`). USD prices: stored as columns in `prices_daily`. Indicators: later.
5. Daily start date: 1995-01-01.
6. Bonds: all from yfinance; FRED equivalents stored as extra `<ticker>_FRED` instruments.
7. ISIN from yfinance, empty if missing; CUSIP derived from it; name fallback for holders.
8. First-run size: OK.
