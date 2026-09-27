# Decisions

Design decisions for the data pipeline in `Database/`. The spec is `platform_prompt.md`; this file records
what was decided on top of it (or where the spec had to be changed), with the reason. Newest additions are
at the end of each section. Keep adding to it.

## Company IDs (`data/corporate_ids.csv`)

- `corporate_id` is a random 5-digit number, assigned once, never changed, never reused. The CSV is the
  single source of truth and is tracked in git (everything else in `data/` is ignored).
- One ID per company: tickers are grouped by SEC CIK (US), otherwise by exact company name.
- Share classes that had two IDs before (GOOGL/GOOG, UAA/UA, NWSA/NWS, FOXA/FOX): the Class A ticker's ID
  survives; the other ID stays in the CSV with `merged_into`. Both tickers keep their own prices.
- The primary listing (used for market cap) is: `PRIMARY_TICKER_OVERRIDES` > Class A / first SEC ticker
  for SEC filers > home-country listing (first in `_tickers` if several) for non-SEC companies.
- Tickers missing from SEC's ticker list are looked up by name (AVB, EA, EQR). EQR is registered as
  "VIVMARK RESIDENTIAL" (same CIK); kept as is.
- **The one exception to "an ID never changes":** a ticker that was grouped into a company by mistake
  (same name in the ticker file, e.g. BEZ.DE Berentzen listed as "Beiersdorf AG") gets a new ID when it is
  split off after the name is corrected. The company keeps its ID: the group that holds the first ticker
  of the registry row. `data/review/name_groups.csv` lists non-US name groups whose yfinance name,
  country or currency differ (dual listings like China A/H shares legitimately differ in currency).
- **Corrections list `data/review/equities_changes.csv`** (ticker, change, old_name, new_name, note) is
  applied once by the ids step (recorded by file hash in `processed_datasets`, so a re-run does not hand
  out new IDs again):
  - `rename` (same company, new name) -> keeps its ID; the registry name follows.
  - `a` / `a+add` (the ticker is a different company than its old name said) -> **new ID**; an old ID with
    no ticker left becomes `reassigned` and is never reused.
  - `replaced by X` (same company, new ticker) -> X inherits the ID; the old ticker is `removed`.
  - `added` -> new ID; `removed` -> status `removed` (price history kept, no more updates).
- `corporate_ids.csv.status` / `company_info.status`: `active`, `removed` (no ticker in the files any more,
  also the ETFs moved to etfs.py), `reassigned`, `merged`. Only active companies are refreshed and get a
  market cap; `instruments.status = 'removed'` for tickers no longer in any file (prices step skips them).

## SEC matching

- Only US companies (`ticker_us`) are matched to SEC. `MATCH_NON_US_TO_SEC = False`: name matching gave
  wrong hits (MRK.DE -> Merck & Co., 9434.T -> SoftBank Group, APA.AX <-> APA Corp).
- `SEC_PREDECESSOR_CIKS`: a company that changed CIK (ExxonMobil Holdings Corp 2115436 <- Exxon Mobil Corp
  34088, 2026) is read under all its CIKs; each filing (accession number) is counted once.
- A company that files 10-Qs but no 10-K yet (new holding company) counts as a 10-K filer.

## Instruments and tickers

- The `_tickers` files are never changed by the pipeline.
- A ticker listed in several files gets one asset class by `ASSET_CLASS_PRIORITY`
  (equities > bond > crypto > forex > commodities > indices > sectors > etfs > sentiment).
- The 9 iShares ETFs that were in `ticker_de` moved to `_tickers/etfs.py` (asset class `etfs`); they keep
  their row in `corporate_ids.csv` (IDs are never reused), but only equities are linked to a corporate_id.
- `data/review/ticker_name_check.csv`: every ticker's name vs yfinance's longName / shortName with a
  similarity score (legal forms removed), worst first, for correcting the ticker files. For equities below 0.5 it
  suggests (a) the correct name for the ticker as it is and (b) the ticker yfinance finds for the named
  company (same exchange preferred); the user picks one per row.
- A ticker without price data for `NO_DATA_AFTER_FAILURES` (3) runs in a row gets `status = 'no_data'`,
  is skipped, and is re-checked every 30 days. An empty download does not count as a failure if the
  ticker has data from the last 14 days (holiday weeks). Same rule for FRED series.
- yfinance ISINs are only kept if they pass the ISIN checksum and their country prefix fits the listing's
  exchange country or the company's country (GOOGL returned a Canadian CDR ISIN). CUSIP = characters 3-11
  of a US ISIN.
- Finnhub's free plan only works for US tickers; it is only used as a fallback for them.

## Daily prices (`prices_daily`)

- yfinance with `auto_adjust=False, actions=True`: `close` is split-adjusted only, `adj_close` is split +
  dividend adjusted; dividends and splits are stored.
- **`close` is split-adjusted, not the price that was actually traded.** Anything that combines `close`
  with unadjusted numbers (SEC share counts, per-share values from filings) must first undo the splits,
  see Market cap.
- Dates are the exchange's local trading dates, never converted to UTC.
- Incremental per ticker: re-download from the last stored date minus 5 trading days. This overwrites a
  partial row stored while the market was open. If `adj_close` in the overlap differs by more than 0.01 %
  (relative) on a date before the last stored one, the ticker's full history is re-downloaded.
- `currency` is the quote unit as Yahoo reports it (`GBp`, `USX` = cents); `fx_to_usd` is USD per 1 of
  that unit (GBp: GBPUSD / 100). Minor units: GBp, GBX, ZAc, ILA, USX (all / 100).
- `CURRENCY_OVERRIDES`: ^MERV = ARS, JKM=F = USD (Yahoo reports none).
- No USD conversion for values that are not money: yields (^IRX ^FVX ^TNX ^TYX and the `_FRED` copies),
  volatility indices (name contains "Volatility Index"), ^SKEW, ^NYHL (`NO_USD_*` in config).

### Price glitches (`price_corrected`, `price_suspect`; `pipeline/price_checks.py`, run by the prices step)

- **Unit switches, GBp / GBX tickers:** a day-to-day jump of ~100x or ~1/100x (+-10 %) is a switch between
  pence and pounds. The stretch in the wrong unit is scaled to the unit of the latest data (OHLC and
  adj_close), `price_corrected = TRUE` (kept on later runs). None in the data today; a safeguard.
- **One-day spikes:** a move of more than 30 % that comes back within 1-3 trading days (within 15 % of the
  level before; the days in between stay > 30 % away) -> `price_suspect = TRUE` on the spike days. Rows are
  kept, but left out of `market_cap_daily` and the dashboard. Recomputed on every run.
  - Only equities, ETFs, sectors and indices, and **not for US listings**: all 21 US cases were real
    (C / BAC 2009-01-21, AIG / RF / HBAN 2008-09, BIIB 2020-11-04 aducanumab). Not for volatility indices,
    yields, commodities or crypto either, where such moves happen.
  - 406 days in 63 tickers: e.g. ULVR.L x51 and back (1995), a zero-volume 4,793p row among ~1,460p
    (1999-05-03), TATASTEEL.NS x10.5, NOVO-B.CO x2, ^J203.JO 99,324 -> 179.8 -> 99,971, and CHIF (a
    sector ETF priced at $0.02-0.05, 154 days).
- The adj_close change check of the prices step ignores corrected rows (no reload loop).

## FX (`fx_rates_daily`)

- FX direction comes from the Yahoo ticker: `EURUSD=X` = USD per EUR, `JPY=X` / `USDJPY=X` = JPY per USD.
- Per currency one Yahoo pair (CCYUSD=X, then USDCCY=X, then CCY=X); missing pairs are added as helper
  instruments.
- Where Yahoo has no rate (before its history starts, or gaps longer than 5 days), sources in this order:
  pegs (`FX_PEGS`), then FRED daily H.10 rates, then FRED monthly OECD averages (IDR, ILS, PLN, ARS),
  applied to every business day of the month and marked `... monthly` in `source`.
- EUR before 1999: FRED has no daily DEM series, so the monthly `EXGEUS` (DEM per USD) / 1.95583 is used.
- ARS: pegged 1:1 to USD until 2002-01-06 (convertibility law), source `peg:ARS=USD ...`.
- An FX rate is carried forward at most 5 calendar days; after that the USD columns stay empty.

## FRED

- Bond yields with a FRED series (`FRED_BOND_MAP`) get an extra instrument `<ticker>_FRED`
  (e.g. `^TNX_FRED` from DGS10); only `close` is filled.
- Update rule per series: FRED `last_updated` unchanged -> skip; daily series -> re-download the last 60
  days; weekly / monthly / quarterly -> full re-download (revisions can go back years).
- `macro_series.category`: `macro` (the user's `macro.py`), `bond_yield`, `fx_helper`. A series in several
  groups keeps the first of that order; a helper category never overwrites `macro`.
- ICE BofA spread series on FRED only go back ~3 years; use DBAA / BAA10Y for long history.
- **Macro dates are FRED observation dates, not publication dates.** A monthly value dated 2026-08-01 is
  published weeks later and may be revised afterwards. For backtests, lag macro data by its publication
  delay (or use ALFRED vintages); do not treat the observation date as the date it was known.
- 2026-09-27: 8 invalid IDs in `macro.py` replaced: COMPAPER3M -> DCPN3M, AWHAREMAN -> AWHMAN,
  A067RX1A020SBEA -> DSPIC96, IPBUSED -> IPBUSEQ, DGS3M -> DGS3MO, NONREVOLSL -> NONREVSL,
  TWEXBPAE01 -> DTWEXBGS (nominal, daily); BAMLC0A1AALH dropped (DBAA was already in the file). Series no
  longer in `macro.py` get `status = 'removed'` in `macro_series` (kept, not requested, not listed in status).

## Fundamentals (`fundamentals_quarterly`, `fundamentals_yearly`)

- Source: SEC XBRL company facts (us-gaap; ifrs-full tags are in the lists for 20-F filers, which are
  switched off). History from fiscal year 2009 — the largest filers only; XBRL was phased in, so most
  smaller companies start in fiscal 2010 or 2011 (later for later IPOs). No data before XBRL.
- **Point-in-time:** each period comes from the filing that first reported it; an amendment (`/A`) only
  if there is no original.
- **Only facts whose period end equals the filing's report period** (EDGAR `reportDate`, +-7 days) are
  used. 10-Qs carry the prior-year comparatives with the same fy/fp; those are ignored.
- **XBRL tag chosen per period**, not per company: for every item the first tag of its list with a value in
  that filing (AAPL revenue: `SalesRevenueNet` to FY2017, `Revenues` FY2018,
  `RevenueFromContractWithCustomerExcludingAssessedTax` from FY2019).
- **Revenue: the largest of the main revenue tags** in that period (`RevenueFromContract...Excluding
  AssessedTax`, `Revenues`, `SalesRevenueNet`, `SalesRevenueGoodsNet`, ...). Companies sometimes put a
  partial number under a higher-ranked tag (TKR 2013 `Revenues` = $80M vs goods revenue $4.3bn; HSY, PG;
  REITs tag only non-lease revenue as `RevenueFromContract...`; `OperatingLeaseLeaseIncome` is in the group
  for them, e.g. CPT). `...IncludingAssessedTax` (incl. sales taxes) is only a fallback.
- **`total_equity` = equity attributable to the parent** (`StockholdersEquity`). Noncontrolling interest
  is its own column `noncontrolling_interest` (`MinorityInterest`); if only equity incl. NCI is tagged,
  the missing part is derived from it. Balance identity:
  **assets = liabilities + redeemable_equity + noncontrolling_interest + total_equity.**
- **`redeemable_equity`** = mezzanine / temporary equity (redeemable NCI, redeemable preferred, the equity
  component of convertible notes). It is neither a liability nor equity. Several tag variants (older
  `TemporaryEquityCarryingAmount`, fair-value and common / preferred variants) are in its list.
- **Revenue is net of excise taxes; `revenue_incl_excise` holds the gross figure** when it can be
  identified: (a) both `...ExcludingAssessedTax` and `...IncludingAssessedTax` tags -> the smaller is net
  (companies swap the labels, e.g. TAP); (b) one revenue tag and revenue - cost_of_revenue - reported
  gross profit = the excise amount (`ExciseAndSalesTaxes`, or MO's `OtherCostOfOperatingRevenue`) -> it
  was gross (MO); = 0 -> it was net, gross = net + `ExciseAndSalesTaxes` (PM). The identity check makes
  sure excise is only used when it is confirmed. Otherwise revenue is left unchanged.
- **`gross_profit`**: the reported `GrossProfit`; only if not reported, revenue - cost_of_revenue with
  `gross_profit_derived = TRUE`. The gross-profit check only looks at reported values.
- **`net_income_to_common`** (`NetIncomeLossAvailableToCommonStockholders...`): after preferred dividends
  and participating securities; the basis of EPS and of the EPS check.
- **Derived Q4 flows are empty unless the nine-month YTD is known**: either reported in the Q3 10-Q or
  chained from complete Q1-Q3. A reported nine-month YTD makes Q4 = FY - 9M exact even if the Q1 / Q2 rows
  are missing (first XBRL year); a missing quarter without a reported YTD leaves Q4 empty (tested).
- **`employees`**: company facts practically never contain it. On every company_info refresh the yfinance
  number is written into the latest fiscal year's row of `fundamentals_yearly` if that is empty
  (`employees_source = 'yfinance <date>'`), so a history builds up. The fundamentals step never writes it.
- EPS check: net income to common / diluted weighted shares vs `eps_diluted`, flagged if more than 5 % *and*
  more than one cent off (EPS is rounded to cents).
- **Share counts tagged in thousands / millions are corrected:** if net income (to common) / weighted
  shares is ~1,000x or ~1,000,000x the reported EPS (within 5 %), the weighted share count is multiplied by
  that factor and `shares_scale_fixed = TRUE`. Other EPS mismatches are only flagged.
- **fiscal_year / fiscal_quarter are the company's own**, from the filing's DocumentFiscalYearFocus /
  DocumentFiscalPeriodFocus (WMT fiscal 2026 = Feb 2025 - Jan 2026).
- Period lengths: quarter 80-100 days (13 or 14 weeks), half year 170-195, nine months 260-290, year
  350-380 days (52 or 53 weeks).
- Flows (income statement, cash flow): the reported quarter if there is one, else YTD - previous YTD
  (10-Q cash flows are usually only YTD). **Q4 = full year - nine-month YTD**, `q4_derived = TRUE`; its
  balance sheet is the 10-K's. Per-share items and weighted share counts are never derived (empty for Q4).
- **shares_outstanding / shares_as_of:** the cover page (dei `EntityCommonStockSharesOutstanding`) and its
  date, summed over share classes. Multi-class companies (e.g. GOOGL) tag the cover page per class, which
  company facts leave out; for them the balance-sheet `CommonStockSharesOutstanding` at period end is used
  and `shares_as_of` is that date.
- `employees` stays empty: dei `EntityNumberOfEmployees` is practically never in company facts;
  `company_info.employees` (yfinance) has the current number.
- **Sign convention (cash flow):** `capex`, `dividends_paid`, `share_buybacks` are stored as the positive
  amounts of the XBRL "Payments..." tags (cash out = positive). `operating_cash_flow`,
  `investing_cash_flow`, `financing_cash_flow` keep their sign (outflow negative).
  So **`free_cash_flow = operating_cash_flow - capex`**.
- **Derived values** (`ebitda = operating_income + depreciation_amortization`,
  `total_debt = short_term_debt + long_term_debt`, `net_debt = total_debt - cash`, `free_cash_flow`,
  margins = item / revenue) **stay empty if any input is missing, never 0**. Margins are empty when
  revenue is 0. `gross_profit` is only stored when reported (not derived), so the check below is real.
- Checks after every run -> `data/review/fundamentals_checks.csv`: assets vs liabilities + equity (> 1 %),
  gross profit vs revenue - cost of revenue (> 1 %), yearly revenue change > 50 %. Flags are for review;
  real growth (AAPL 2010-11) is flagged too.
- Companies that changed CIK are read under all CIKs (`SEC_PREDECESSOR_CIKS`: XOM, GOOGL <- Google Inc.).
- **Mislabelled filings:** an 'FY' filing whose period end is more than a month away from the company's
  usual fiscal-year-end month is a quarter and is dropped (SNOW, QRVO, DELL); a fiscal year far from the
  period end (STX's 10-K for June 2025 tagged 2027) is replaced by the period-end year plus the company's
  usual offset. On a rebuild, rows whose key no longer exists are deleted (rows without a period end,
  created by the calendar / holders steps before the 10-Q, stay).
- Incremental: a company is rebuilt only when SEC has a new 10-K / 10-Q / 20-F / 40-F / 8-K, when
  `BUILD_VERSION` (code) or its CIK list changed, or **while a 10-Q / 10-K from the last 400 days is not in
  its rows yet**: SEC's company facts API lags behind EDGAR (in Sept 2026, 58 of 421 filings since July
  were not in company facts yet, e.g. ABT's Q2 10-Q). It is always rebuilt from the full filing history stored
  in `sec_filings`.

## Fundamentals: when a company is rebuilt

- A company is rebuilt when SEC has a new 10-Q / 10-K / 20-F / 8-K since its last build, or the build code or
  CIK list changed. A recent 10-Q / 10-K whose facts are not in the fundamentals yet ("pending", SEC's company
  facts lag EDGAR) keeps it on the list; it is then only rebuilt when its **company facts changed**:
  fingerprint = number of facts used + newest `filed` date (`fundamentals_state.facts_fingerprint`). SEC's
  company-facts endpoint has no Last-Modified / ETag, so the facts are still downloaded for those companies.
  (64 companies had such filings that never map to a stored row; they were rebuilt on every run before.)

## Market cap (`market_cap_daily`)

- **Split rule:** yfinance `close` is split-adjusted, SEC share counts are not. Both are put on today's
  share basis before multiplying: `shares x product of all splits after the shares' date`, times the
  split-adjusted close. This equals the real historical price x the real share count, also when a split
  falls between the shares' date and the price date. `shares_outstanding` is stored as the real count on
  that day. Never multiply split-adjusted `close` by an unadjusted historical share count.
- Tested: AAPL continuous across 2020-08-31 (4:1), 7203.T across 2021-09-29 (5:1), on synthetic and real data.
- Market cap = all share classes x the primary listing's close. Non-primary tickers keep their own prices.
  `market_cap` is in the major currency (GBp prices -> GBP), `market_cap_usd` via `fx_to_usd`.
- **Shares, SEC filers:** the fundamentals' share count (cover page, or balance sheet for multi-class
  companies), effective from its as-of date and carried forward, **at most `SHARES_STALE_DAYS` (400)**;
  after that the current yfinance count (Berkshire stopped tagging its cover page in 2011).
- **Before the first SEC count** the earliest count is extended backwards in split-adjusted terms
  (`shares_source = 'sec_backfilled'`), so US companies also go back to 1995. This is an approximation:
  buybacks and issuance before the first XBRL filing are not reflected.
- **Share issuance through a deal:** if two consecutive SEC counts differ by more than 20 % and an 8-K
  with item 2.01 (completion of an acquisition) lies between them, the new count applies from the 8-K
  date (`shares_source = 'sec_after_deal'`), not only from the next cover page (DVN: 621M -> 1.1bn shares
  on 2026-05-07; the next cover page was 2026-07-22).
- **Glitches in SEC counts are dropped** (the previous count carries on): counts below 10,000 (FOXA's
  1-share placeholder from before its listing), counts more than 100x off the company's median, points
  more than 2x off the median of their 5 neighbours (cover pages tagged in thousands), and a single count
  that jumps > 30 % while the next one is back within 10 % of the previous (WFC 2023-07-21: 3.75bn ->
  1.82bn -> 3.63bn). Real jumps such as AIG 2011 (x13, recapitalisation) or mergers stay.
- Days with `price_suspect` (one-day price spikes) are left out of `market_cap_daily`.
- **Shares, everyone else:** yfinance `impliedSharesOutstanding` (all share classes, e.g. BYD H + A shares),
  falling back to `sharesOutstanding`; current value only, applied to the whole price history
  (`shares_source = 'yfinance_current'`). Refreshed with company_info.
- `MARKETCAP_YFINANCE_ONLY = {'BRK-B'}`: Berkshire's SEC counts are Class A shares, its primary listing
  is Class B.
- **Spin-offs vs splits (`corporate_actions`, prices.duckdb):** Yahoo records a spin-off as a fractional
  `stock_splits` (HON 1.061, BDX 1.272, SPGI 1.057). That factor is right for the price history but must not
  scale share counts. Each Yahoo factor is split into `split_factor x price_only_factor`:
  - SEC companies: **real split factor = SEC shares after / before the event, rounded to a clean ratio**
    (p:q with p, q <= 10, or n:1 / 1:n up to 50, incl. 1:1) **within +-3 %** (`SPLIT_RATIO_TOLERANCE`).
    The rest of Yahoo's factor is price-only. HON 2026 (0.9535) = 1:2 reverse split x 1.907 spin-off.
  - No SEC count after the event yet: the current yfinance count, if fetched after the event (SPGI 2026).
  - **No clean ratio, or counts more than 120 days from the event** (`SPLIT_SEC_MAX_DAYS`) ->
    `data/review/corporate_actions_review.csv`, **price-only until reviewed**. Exception: far-away counts whose
    ratio rounds to the same clean ratio as Yahoo's own factor confirm the split (CSX 2011 3:1, V 2015 4:1,
    SMCI 2024 10:1). Reviewed events go into `CORPORATE_ACTION_OVERRIDES` (ticker, date) -> split factor.
  - **Split basis of an SEC count:** cover-page counts are real at their as-of date. Balance-sheet counts are
    restated for splits up to the filing date (SEC rules): a 10-Q filed after a split shows the period-end
    count post-split (GOOGL 2014-03-31 = 674.5M in the 10-Q of 2014-04-24, real 337M). So a balance-sheet
    count is on the split basis of its filing date. Classification compares the last count still on the
    pre-event basis with the first one on the post-event basis; the distance to the event uses the as-of or
    filing date, whichever is nearer. Without this, 10 real splits looked like spin-offs (TJX, NFLX 7:1,
    SHW 3:1, GOOGL / GOOG 20:1 2022, ...).
  - Overrides so far (reviewed): MA 2014 10:1, CME 2012 5:1, NKE 2015 2:1, AOS 2016 2:1, DUK 2012 1:3 (same
    day as the Progress Energy merger), GOOGL 2014 and WBD 2014 (Class C share dividends, 2:1; WBD confirmed
    by the convertible preferred's conversion ratio going from 1 to 2 on 2014-08-06), BALL 2011 2:1. Left
    price-only: the spin-offs (COP 2012, TT 2013, SPG 2014, YUM 2016, DOV 2018, LH 2023, BDX 2026) and the
    unclear HST 2009, SPG 2009, GOOG 2015, DELL 2018.
  - Events before the company's first SEC count, and all non-SEC companies: Yahoo's factor is the split (as
    before). Pre-coverage events with a non-clean factor (5 % stock dividends in the 1990s) are listed in the
    review file for information only.
  - Market cap = close x (Yahoo factors after the day / split factors after the day) x shares on today's
    split basis. Share counts only use split factors.
  - Intraday reloads its history only after a real split (split_factor != 1), not after a spin-off (Alpaca
    doesn't adjust spin-offs).
- The table (and `corporate_actions`) is rebuilt completely on every run. Checks in
  `data/review/marketcap_checks.csv` (column `check`):
  - `shares_vs_yfinance`: latest share count more than 20 % off yfinance's (BX: Yahoo also counts
    partnership units).
  - `daily_jump`: market cap changes by more than 30 % from one trading day to the next, except on days with
    a corporate action: a split / spin-off of the primary ticker, or a completed deal (8-K item 2.01, the
    `sec_after_deal` date). `driver` = `price` (share count unchanged) or `shares` (share count changed).
    First run 1,224 days; after the price-glitch flags, the share-count blip rule and the deal exclusion:
    806 (US: 372 price, mostly real crashes 1998-2003 / 2008-10 / 2020, and 16 share-count jumps; non-US:
    418 price, e.g. FMG.AX month-end prints in the 1990s).

## Running, locks and backups

- **Locked database:** DuckDB allows one writing process per file. If another process holds a database (the
  Streamlit app, a notebook), the pipeline logs it and retries every 30 s for up to 10 minutes
  (`DB_LOCK_RETRY_SECONDS`, `DB_LOCK_MAX_WAIT_SECONDS`), then the step fails. Other IO errors are not retried.
  The Streamlit app opens read-only, queries and closes at once, and caches only the data, so it doesn't block.
- **Daily run:** Task Scheduler `MarketDataPipeline`, Mon-Sat 23:30, `run_daily.bat`, log in `data/logs/`;
  missed runs start as soon as possible.
- **Weekly backup (Saturday run):** `raw.duckdb` (the clinical-trial version history can't be downloaded
  again) and `corporate_ids.csv` -> `~/Backups/database_weekly/<name>_<date>`, the last 8 of each kept
  (`backup_weekly.ps1`).

## Earnings calendar (`earnings_date`, `earnings_time`, `earnings_date_status`)

- **Past earnings date = the first 8-K with item 2.02 after the quarter's period end, within 90 days**
  (not any 2.02). Status `reported`.
- **Time of day from the EDGAR acceptance time** (UTC, converted to New York time): before 09:30 `bmo`,
  from 16:00 `amc`, otherwise `during`. The date is the New York date of acceptance, not EDGAR's filing
  date (filings accepted after 17:30 are dated the next business day).
- A 2.02 8-K after the latest quarter with fundamentals (10-Q not filed yet) creates the next fiscal
  quarter's row with status `reported`; its financial columns fill in when the 10-Q arrives.
- **Upcoming dates: US companies only** (Finnhub's free plan has no others). The earliest Finnhub date from
  today is assigned to **the fiscal quarter after the latest reported one**; Finnhub's own fiscal quarter
  labels are not used. Time from Finnhub's `hour` (bmo / amc / dmh = during).
- Status `confirmed` only if the source says so. Finnhub's calendar has no confirmation field, so all
  upcoming dates are `estimated`. Estimates are cleared and rewritten on every run.
- Yearly rows: earnings date / time / status of fiscal Q4; `annual_report_date` = the 10-K filing date.

## Institutional holders (13F: `top_holders`, `inst_ownership_pct`, `n_institutional_holders`)

- Source: SEC Form 13F data sets (3-month filing windows). **Grouped by PERIODOFREPORT**, because late
  filings and amendments of a quarter arrive in later windows. Each window is downloaded once and kept as
  Parquet in `data/cache/13f/` (only the columns used, SH positions).
- **5 quarters are loaded, 4 are shown**: the oldest only provides the change vs the previous quarter.
- Only SH positions; no PRN, no puts / calls.
- Amendments per filer and quarter: the latest original or **RESTATEMENT is the base** (a restatement
  replaces the original); **NEW HOLDINGS** amendments filed after it are **added**.
- VALUE is in thousands for filings before 2023-01-03, in dollars from then on; both are converted to USD.
- **Matching (US companies only)**, in this order; a CUSIP claimed by an exact CUSIP match can't be taken
  by a name match:
  1. `cusip`: CUSIPs of the company's US ISINs, widened to all common-stock CUSIPs of the same issuer
     (first 6 characters) = all share classes (GOOGL + GOOG).
  2. `name`: the CUSIP with the most holder rows (>= 20) whose most common 13F issuer name equals the
     company name.
  3. `name_abbrev`: every company word abbreviated by a 13F word, any order ("COSTCO WHSL", "DISNEY WALT",
     "TEXAS INSTRS"; >= 3 letters, or a 2-letter prefix), at most one harmless extra word ("IRELAND",
     cut-off "CORPORATI").
  4. `name_fuzzy`: same first word, similarity >= 0.85.
  Name matches are widened only to share classes with the same issuer code and name and >= 5 % of the
  main CUSIP's holder rows. Common stock = valid check digit, numeric issue code below 90 (letters = debt,
  90-99 = options), no preferred / note / warrant / ETN / option title.
  `data/review/holders_matching.csv` shows the method and CUSIPs per company.
  0. `override`: `CUSIP_OVERRIDES` in config (WY, SJM: 13F names 'WEYERHAEUSER CO MTN BE', 'SMUCKER J M').
  A CUSIP that gives < 5 % institutional ownership is outdated (new CUSIP after a re-domiciling, STX):
  the company is matched again by name without it, taking the CUSIP with the most holders in the latest
  quarter.
- Per company, filer and quarter all share classes are added up; top 20 holders by shares. Each holder
  has a `holder_group` (config `HOLDER_GROUPS`: Vanguard, BlackRock, Fidelity, State Street, ...), so
  related filers can be added up (AAPL: 3 Vanguard filers = 9.4 %).
- **change_shares_vs_prev_quarter** is empty when the filer's previous report is missing or incomplete
  (fewer than 10 % of its current positions; Norges Bank files full reports only for Q2 / Q4). A complete
  previous report without the stock = a new position (change = all shares).
- `pct_of_shares_out` and `inst_ownership_pct` use the shares outstanding (all classes) from
  `market_cap_daily` on the 13F date. The 13F quarter goes to the company's fiscal quarter with the
  nearest period end (max 45 days); the yearly row gets the fiscal Q4 holders. If the 10-Q of that quarter is
  not filed yet, the fiscal quarter is counted on from the latest row, with quarter ends derived from
  `company_info.fiscal_year_end`, and the row is created (like the calendar step does).
- Check: `data/review/holders_checks.csv` lists companies whose latest institutional ownership is below
  5 % (almost always a wrong match) or above 110 % (known 13F effect for heavily shorted stocks: lent
  shares are reported by lender and buyer).

## Executives (`fundamentals_yearly.executives`) and insider transactions

- Source: SEC insider transactions data sets (Forms 3 / 4 / 5), quarterly, last 5 years (+ one year for
  fiscal years that start earlier). Kept as Parquet in `data/cache/form345/`.
- **Executives of a fiscal year = every person who filed a Form 3 / 4 / 5 for the company with a period of
  report inside that fiscal year as officer or director.** Companies / funds / trusts (by name) and pure
  10 % owners are left out. Predecessor CIKs are included (GOOGL <- Google Inc., XOM).
  Someone who left during the year is still listed for that year (AAPL FY2025: both CFOs, Maestri and
  Parekh); changes between years are visible by comparing the yearly rows.
- Per person: `name` (readable, 'Timothy D. Cook'), `sec_name` ('COOK TIMOTHY D'), CIK, the latest `title`
  of the year, `role`, `is_officer`, `is_director`.
- **`role`** from the title, first match wins: CEO, CFO, COO, President, Chair, General Counsel; else
  Director (director only) or Other. CEO / CFO / COO / President only count company-wide: if other words
  follow the role it is a business-line head ('CEO CCB', 'Co-CEO CIB', 'President, Global Banking' ->
  Other); 'Chairman & CEO', 'President and CEO', 'CEO and Director' stay CEO.
- Only completed fiscal years (rows in `fundamentals_yearly`) get executives.
- **`insider_transactions_raw`** (raw.duckdb): all non-derivative transactions of the covered companies, one
  row per transaction and reporting owner (joint filings have several owners), incl. 10 % owners and
  entities, with date, code (P purchase, S sale, A award, M exercise, F tax, G gift ...), shares, price,
  holdings afterwards, role. Kept exactly as filed: a few filings have typos in the date (year 24, 2031).
  **`date_suspect = TRUE`** when the transaction date is after the filing date or in the future; the rows are
  kept.

## Subsidiaries (`fundamentals_yearly.subsidiaries`, `n_subsidiaries`)

- Source: Exhibit 21 of the 10-K for the last 5 fiscal years (the 10-K stored in the yearly row; a 10-K/A
  for the same period if the 10-K has none). Found via the filing's EDGAR index page (document type EX-21).
- Parsed from HTML tables (name | jurisdiction | [ownership %]) or text lines ('Name (Jurisdiction)',
  'Name ..... Jurisdiction', 'Name, a Delaware corporation', plain names). Header rows ('Name of
  subsidiary', 'Organized under the laws of', dates) and footnotes are skipped; periods in names are kept
  ('S.A. de C.V.').
- **A 10-K without its own Exhibit 21** usually incorporates it by reference: the exhibit index links the
  earlier filing's document, which is loaded (17 of 38 cases). Without a link, the previous fiscal year's
  list is carried forward with `subsidiaries_carried_forward = TRUE` (ALB 2022, MTB 2023). Still empty:
  companies that never file an Exhibit 21 in the window (ODFL, TXG) or list subsidiaries in the 10-K body
  (CINF: "contained in Part I, Item 1"), and the first year of GL / MTB.
- An Exhibit 21 that can't be read (one block of text without any separators: APA, ABT 2022) stays empty,
  listed in `data/review/subsidiaries_unparsed.csv`. Numbering and bullets ('1.', '•') are stripped.
- **`subsidiaries_scope`**: `all` (no restriction), `omits_insignificant` (the standard Item 601(b)(21)(ii)
  clause: subsidiaries that together would not be significant are left out; used by near-complete lists
  like UNH 2024 with 2,327 entries as well as by AAPL with 19), `significant_only` (the list says it lists
  significant subsidiaries: UNH 2025, JPM). `subsidiaries_significant_only` = scope `significant_only`.
  Policy switches are visible: UNH 2024 -> 2025, NVDA 2023 (`all`) -> 2024 (`omits_insignificant`). Companies list
  only *significant* subsidiaries (Alphabet: 3), so counts are not comparable in absolute terms; changes
  between years are.
- Each filing is processed once per parser version (`processed_datasets 'ex21:v<N>:<accession>'`); the
  documents are cached in `data/cache/ex21/`, so a new parser version re-parses without downloading.

## Intraday (`data/intraday/<interval>/ticker=<T>/<year>.parquet`)

- 1-minute bars from Alpaca for the US tickers, **SIP feed** (all exchanges; IEX only as fallback, it has
  ~2-3 % of the volume). The last 16 minutes are never requested (SIP is delayed 15 min on the free plan).
- **Split-adjusted only, not for dividends** (Alpaca `adjustment=split`). If a split happens after a ticker's
  history was downloaded, its whole intraday history is downloaded again (`intraday_state.split_basis_date`).
- **Session: 09:30 up to and including the closing-auction minute** (16:00; 13:00 on early-close days: 3 July,
  the day after Thanksgiving, 24 December when weekdays). The 16:00 bar holds the closing cross (10-20 % of
  the day's volume) and the official close; without it summed volume was only 70-80 % of the daily volume,
  with it 95-99 %. Pre- and after-market bars are dropped. Early-close days are shorter days, not gaps.
- **5m / 10m / 1h are resampled from the 1-minute bars, anchored at the open**: 1h bars start 09:30, 10:30,
  ..., 15:30 (the last one is 15:30-16:00). The closing-auction minute is folded into the session's last bar
  (no extra bar at 16:00; the last bar closes at the official close).
- **No empty bars**: only minutes with trades exist, resampling uses only those. Volume and trade_count are
  summed, VWAP is volume-weighted, open / close = first / last, high / low = max / min.
- Timestamps are UTC, start of the bar. Incremental per ticker from the last stored minute; first run
  `INTRADAY_HISTORY_DAYS` (365).
- **Check vs prices_daily** per ticker and day -> `data/review/intraday_checks.csv`: intraday high / low more
  than 2 % off, or the day's volume ratio (session volume / daily volume) more than 15 % off **the ticker's
  own median ratio**. Yahoo's daily volume includes pre- and after-market trading, so a normal ratio is below
  1 and differs per ticker (AAPL 0.93, FMC 0.99, BRK-B 0.89); comparing with the median means normal
  extended-hours volume doesn't cause flags. No separate pre / post volume table. The check always covers
  all US tickers, also on a `--tickers` run.
- Alpaca symbols use a dot for share classes (`BF-B` -> `BF.B`); Yahoo's dash returns HTTP 400.
- The feed fallback (SIP -> IEX) happens only when SIP is refused (HTTP 403 / 422). Network errors (DNS,
  timeouts) fail the ticker for this run instead, so it isn't silently stored on the thin IEX feed.
- Known check differences: (a) spin-offs: Yahoo adjusts the price history for a spin-off like a split,
  Alpaca doesn't, so price flags before the spin-off date are expected (HON, FDX, SPGI, BDX, DD);
  (b) index-event days (quadruple witching, MSCI / Russell rebalances): part of the closing volume is
  reported after the 16:00 minute, so session volume looks 15-35 % low.
- **Possible extension (not built):** extended-hours 1-minute bars (04:00-20:00) with a `session` column
  (`pre` / `regular` / `post`), e.g. for earnings reactions (most earnings are released before the open or
  after the close). The resampled 5m / 10m / 1h bars would stay regular-session only.

## Clinical trials (`raw.duckdb` `clinical_trials_raw`)

- Source: ClinicalTrials.gov API v2, no key. Trials: the stored ones + `config.TRIAL_NCT_IDS` + the trials of
  `config.TRIAL_SPONSORS` (lead sponsor) + `run.py trials --nct NCT...`. Both config lists stay empty for now.
- One row per trial and version, PK (nct_id, version). `last_change_date` =
  `protocolSection.statusModule.lastUpdatePostDateStruct.date` ('YYYY-MM' -> first of the month).
- **Change = different content hash**: sha256 of the normalised JSON (sorted keys, no whitespace), so identical
  content never counts as a change. `derivedSection.miscInfoModule.versionHolder` is left out of the hash: it is
  the date of the ClinicalTrials.gov snapshot and changes every day for every trial. It stays in `raw_json`.
- A new version keeps all old ones; `is_latest` moves to it, `change_flag = TRUE`, `changed_sections` = the
  modules that differ (`statusModule`, `designModule`, ...; `hasResults` for the top-level flag; a new
  `resultsSection` lists its modules). Version 1: `change_flag = FALSE`, `changed_sections = []`.
- Incremental: stored trials and known sponsors are only requested with lastUpdatePostDate on or after the
  latest stored `last_change_date` (inclusive, the hash drops re-downloads without changes). New NCT IDs and new
  sponsors are requested in full. MeSH terms in `derivedSection` (conditionBrowseModule, interventionBrowseModule)
  are maintained by the NLM and can change without the sponsor updating the trial; they count as changes.
