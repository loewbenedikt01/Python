# Market-data database

Python + DuckDB + Parquet. One entry script (`run.py`) downloads and updates everything incrementally:
daily prices, FX, FRED, SEC fundamentals, market cap, earnings calendar, 13F holders, executives, insider
transactions, subsidiaries, 1-minute intraday bars and clinical trials.

Why things are done the way they are (split rule, spin-offs, FX, matching, ...): **[DECISIONS.md](DECISIONS.md)**.

## Setup

```bash
pip install -r requirements.txt
```

API keys live in `../api_keys.txt` (one level above `Database/`, git-ignored), one `NAME=value` per line:

| Key | Used for |
|---|---|
| `FRED_API_KEY` | FRED (bond yields, macro, FX helper series) |
| `ALPACA_KEY_ID`, `ALPACA_SECRET` | intraday bars |
| `FINNHUB` | upcoming earnings dates |
| `NAME`, `EMAIL` | SEC EDGAR User-Agent (SEC requires it) |

SEC EDGAR, yfinance and ClinicalTrials.gov need no key. Keys are read only from this file and are never
printed (log and error messages redact them).

## Running

```bash
python run.py                          # all steps, in order
python run.py prices intraday          # only these steps
python run.py --tickers AAPL MSFT      # limit to some tickers (testing)
python run.py trials --nct NCT04368728 # add trials to the clinical-trials step
python run.py status                   # rows, latest date, last successful run, failures per table
python -m pytest tests                 # tests (no network; a few also read the built databases)
```

A step that fails doesn't stop the run; every step writes a row to `load_log` in its database. At the end
a summary table is printed. Logs: `data/logs/`.

**Daily run:** Windows Task Scheduler task `MarketDataPipeline` runs `run_daily.bat` Monday-Saturday at
23:30 (output: `data/logs/daily_<date>_<time>.log`). If the computer was off, it runs as soon as possible
afterwards. Only while you are logged on. On Saturdays `backup_weekly.ps1` then copies `raw.duckdb` and
`corporate_ids.csv` to `~/Backups/database_weekly/` (last 8 kept). Definition: `run_daily_task.xml`; re-register with
`schtasks /Create /TN "MarketDataPipeline" /XML "run_daily_task.xml" /F`, remove with
`schtasks /Delete /TN "MarketDataPipeline"`.

### Steps (in this order)

| Step | What it does | Writes |
|---|---|---|
| `ids` | syncs `_tickers/` with `data/corporate_ids.csv` (5-digit company IDs) | corporate_ids.csv, `instruments`, `company_info` |
| `companies` | company info (name, sector, country, SEC CIK, employees, yfinance shares) | `company_info` |
| `prices` | daily OHLCV + dividends / splits for all tickers, FX rates, USD columns | `prices_daily`, `fx_rates_daily`, `instruments` |
| `fred` | FRED bond yields and macro series | `macro_series`, `macro_observations` |
| `fundamentals` | SEC XBRL company facts -> quarterly / yearly financials, filings | `fundamentals_*`, `sec_filings` |
| `marketcap` | daily market cap per company; splits vs spin-offs | `market_cap_daily`, `corporate_actions` |
| `calendar` | earnings dates / times (8-K 2.02 + Finnhub) | `fundamentals_*` |
| `holders` | 13F top holders, institutional ownership | `fundamentals_*` |
| `executives` | executives per fiscal year, insider transactions | `fundamentals_yearly`, `insider_transactions_raw` |
| `subsidiaries` | Exhibit 21 subsidiaries per fiscal year | `fundamentals_yearly` |
| `intraday` | Alpaca 1-minute bars, resampled to 5m / 10m / 1h | `data/intraday/`, `intraday_state` |
| `trials` | ClinicalTrials.gov, versioned | `clinical_trials_raw` |

## Data

```
data/
  prices.duckdb        instruments, prices_daily, fx_rates_daily, macro_series, macro_observations,
                       market_cap_daily, corporate_actions, intraday_state, load_log
  companies.duckdb     company_info, fundamentals_quarterly, fundamentals_yearly, sec_filings,
                       processed_datasets, fundamentals_state, load_log
  raw.duckdb           clinical_trials_raw, insider_transactions_raw, load_log
  intraday/<1m|5m|10m|1h>/ticker=<T>/<year>.parquet
  corporate_ids.csv    source of truth for corporate_id (never renumbered)
  cache/               downloaded SEC / 13F / Form 3-4-5 data sets (safe to delete, re-downloaded)
  review/              csv files to check by hand (see below)
  logs/
```

Keys: `corporate_id` (5 digits) links companies across all tables; prices are per `ticker`. Reading from
another program, e.g.:

```python
import duckdb
con = duckdb.connect('data/prices.duckdb', read_only=True)
con.execute("ATTACH 'data/companies.duckdb' AS c (READ_ONLY)")
con.sql("""SELECT m.date, m.market_cap_usd, f.revenue
           FROM market_cap_daily m JOIN c.fundamentals_quarterly f USING (corporate_id)
           WHERE m.primary_ticker = 'AAPL' LIMIT 5""")
con.sql("SELECT * FROM read_parquet('data/intraday/1h/ticker=AAPL/*.parquet')")
```

Only one process can open a DuckDB file for writing. Readers should open with `read_only=True`, query and
close right away (like the Streamlit app); if a run finds a database locked, it waits and retries every
30 s for up to 10 minutes and logs it.

## Incremental updates

- Prices: from the last stored date with a 5-day overlap; if a re-downloaded `adj_close` differs (new
  dividend / split), that ticker's history is reloaded. Tickers without data for 3 runs become `no_data`
  and are retried every 30 days.
- FRED: skipped when the series' `last_updated` is unchanged; daily series re-download 60 days.
- Fundamentals: only companies with a new SEC filing since the last build.
- Holders / executives: only SEC bulk data sets not processed yet (cached in `data/cache/`).
- Intraday: from each ticker's last stored minute; full reload after a real split.
- Clinical trials: only trials changed since the latest `last_change_date`; a changed trial gets a new
  version, old versions are kept.
- Company info: refreshed after 30 days. Market cap: rebuilt completely (seconds).
- All writes are upserts on the primary key, so re-running never creates duplicates.

## Maintenance

- **Tickers:** edit the files in `_tickers/`, then run `python run.py`. New companies get new IDs;
  corrections (renames, company changes, removals) go into `data/review/equities_changes.csv` (see
  DECISIONS.md, "Company IDs").
- **Settings:** everything is in `config.py` (paths, start dates, history lengths, intraday intervals,
  FRED bond mapping, top-N holders, refresh days, trial sponsors / NCT IDs, rate limits, overrides).
- **Review files** in `data/review/` (rewritten on each run of their step):

| File | What to look at |
|---|---|
| `corporate_actions_review.csv` | Yahoo split factors that couldn't be classified as split or spin-off; decide in `CORPORATE_ACTION_OVERRIDES` |
| `marketcap_checks.csv` | latest share count > 20 % off yfinance; day-to-day market cap changes > 30 % without a corporate action |
| `fundamentals_checks.csv` | implausible financials (balance sheet doesn't add up, jumps, ...) |
| `holders_checks.csv`, `holders_matching.csv` | 13F ownership < 5 % or > 110 %, how each company was matched |
| `intraday_checks.csv` | intraday high / low > 2 % off daily, or volume ratio > 15 % off the ticker's median |
| `subsidiaries_unparsed.csv` | Exhibit 21 documents that couldn't be read |
| `ticker_name_check.csv`, `name_groups.csv`, `sec_matches.csv` | ticker / name / SEC CIK mismatches |

- Overrides in `config.py`: `SEC_CIK_OVERRIDES`, `PRIMARY_TICKER_OVERRIDES`, `CUSIP_OVERRIDES`,
  `CORPORATE_ACTION_OVERRIDES`, `CURRENCY_OVERRIDES`, `MARKETCAP_YFINANCE_ONLY`.

## Code

```
run.py              entry point (steps, --tickers, --nct, status)
config.py           all settings
pipeline/
  common.py         keys, rate limits, HTTP with retries, DuckDB connect / upsert, logging
  schema.py         all tables + migrations
  ids.py  companies.py  prices.py  fx.py  fred.py  fundamentals.py  marketcap.py
  calendar.py  holders.py  executives.py  subsidiaries.py  intraday.py  trials.py
tests/              pytest, one file per module
```
