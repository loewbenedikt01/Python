"""
Core schema: reference tables, companies, identifiers, market data, calendar,
macro and the load log. Run once per connection by db.connect(); every
statement is CREATE ... IF NOT EXISTS / CREATE OR REPLACE VIEW, so re-running is safe.
"""

SCHEMA_SQL = r"""
-- ==========================================================================
-- Market Intelligence Platform: storage schema (DuckDB)
--
-- Every table about a company carries corporate_id, your own permanent ID
-- (random 5-digit integer from corporate_ids.csv). External IDs (CIK, ticker, ISIN, LEI) live in
-- company_identifiers, because tickers change and one company can have
-- several listings.
--
-- Conventions
--   * No foreign-key constraints: DuckDB blocks upserts on referenced rows.
--     Relationships are documented in comments ("-> table.column").
--   * Primary keys define what one row means; loaders upsert on them, so
--     re-running a download never creates duplicates.
--   * Money is stored in the currency given in the row's currency column.
--   * "Changes over years" tables store one snapshot per fiscal_year;
--     differences between years are computed by query, not stored.
-- ==========================================================================

-- --------------------------------------------------------------------------
-- 0. Reference tables
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS countries (
    country_code   VARCHAR PRIMARY KEY,          -- ISO 3166-1 alpha-2, e.g. 'US'
    country_name   VARCHAR NOT NULL,
    continent      VARCHAR NOT NULL,             -- 'North America', 'Europe', ...
    region         VARCHAR                       -- optional finer grouping
);

CREATE TABLE IF NOT EXISTS currencies (
    currency_code  VARCHAR PRIMARY KEY,          -- ISO 4217, e.g. 'USD'
    currency_name  VARCHAR NOT NULL
);

-- --------------------------------------------------------------------------
-- 1. Companies (mostly static) - your "table 2"
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS companies (
    corporate_id          INTEGER PRIMARY KEY,   -- 5-digit id from corporate_ids.csv, never reused
    name                  VARCHAR NOT NULL,      -- current short name
    legal_name            VARCHAR,
    country_code          VARCHAR,               -- -> countries (continent comes from there)
    reporting_currency    VARCHAR,               -- -> currencies; currency of the financial statements
    fiscal_year_end       VARCHAR,               -- 'MM-DD', e.g. '12-31'
    sector                VARCHAR,
    industry              VARCHAR,
    sic_code              VARCHAR,
    business_description  VARCHAR,
    hq_street             VARCHAR,
    hq_city               VARCHAR,
    hq_state              VARCHAR,
    hq_postal_code        VARCHAR,
    hq_country_code       VARCHAR,               -- can differ from country of incorporation
    website               VARCHAR,
    ipo_date              DATE,
    status                VARCHAR DEFAULT 'active',  -- active / delisted / acquired / bankrupt
    status_date           DATE,
    updated_at            TIMESTAMP DEFAULT current_timestamp
);

-- "Static" data does change (renames, HQ moves, reclassification).
-- companies holds the current value; this table keeps the history.
CREATE TABLE IF NOT EXISTS company_history (
    corporate_id   INTEGER NOT NULL,
    attribute      VARCHAR NOT NULL,             -- 'name', 'sector', 'hq_city', ...
    value          VARCHAR,
    valid_from     DATE NOT NULL,
    valid_to       DATE,                         -- NULL = still valid
    source         VARCHAR,
    PRIMARY KEY (corporate_id, attribute, valid_from)
);

-- Maps outside IDs to your corporate_id. Look-ups always go through here.
CREATE TABLE IF NOT EXISTS company_identifiers (
    corporate_id   INTEGER NOT NULL,
    id_type        VARCHAR NOT NULL,             -- 'CIK', 'LEI', 'ISIN', 'FIGI', 'TICKER'
    id_value       VARCHAR NOT NULL,
    valid_from     DATE NOT NULL DEFAULT DATE '1900-01-01',
    valid_to       DATE,                         -- ticker changes end here
    is_primary     BOOLEAN DEFAULT TRUE,
    PRIMARY KEY (id_type, id_value, valid_from)
);

-- --------------------------------------------------------------------------
-- 2. Instruments and daily market data - your "table 1"
--    One instruments table for ALL asset classes: stocks, indices, FX,
--    crypto, commodities, bonds. Stocks link to a company via corporate_id.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS instruments (
    instrument_id       VARCHAR PRIMARY KEY,     -- your ID, e.g. 'EQ:LLY:XNYS', 'FX:EURUSD', 'BD:US10Y'
    asset_class         VARCHAR NOT NULL,        -- equity / index / forex / crypto / commodity / bond / etf
    symbol              VARCHAR NOT NULL,
    name                VARCHAR,
    exchange            VARCHAR,                 -- MIC code, e.g. 'XNYS'
    currency            VARCHAR,                 -- trading / quote currency
    corporate_id        INTEGER,                 -- -> companies (equities only)
    isin                VARCHAR,
    figi                VARCHAR,
    share_class         VARCHAR,                 -- 'A', 'C', 'ADR', ...
    is_primary_listing  BOOLEAN,
    base_currency       VARCHAR,                 -- FX / crypto pairs
    maturity_date       DATE,                    -- bonds
    coupon_rate         DOUBLE,                  -- bonds
    status              VARCHAR DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS prices_daily (
    instrument_id  VARCHAR NOT NULL,             -- -> instruments
    date           DATE NOT NULL,
    open           DOUBLE,
    high           DOUBLE,
    low            DOUBLE,
    close          DOUBLE NOT NULL,              -- raw close (bonds: yield in %)
    adj_close      DOUBLE,                       -- split/dividend adjusted
    volume         DOUBLE,
    source         VARCHAR,
    PRIMARY KEY (instrument_id, date)
);
-- Intraday bars are NOT stored here. They go to Parquet files:
--   prices_intraday/interval=1m/asset_class=equity/year=2026/month=09/part.parquet
-- and are queried with read_parquet(...) (see db.py).

CREATE TABLE IF NOT EXISTS corporate_actions (
    instrument_id  VARCHAR NOT NULL,             -- -> instruments
    action_type    VARCHAR NOT NULL,             -- split / dividend / spinoff / ticker_change / delisting
    ex_date        DATE NOT NULL,
    pay_date       DATE,
    ratio          DOUBLE,                       -- splits: 2.0 = 2-for-1
    amount         DOUBLE,                       -- dividends per share
    currency       VARCHAR,
    details        VARCHAR,
    PRIMARY KEY (instrument_id, action_type, ex_date)
);

-- Computed daily: shares outstanding (carried forward) x price.
CREATE TABLE IF NOT EXISTS valuation_daily (
    corporate_id        INTEGER NOT NULL,
    date                DATE NOT NULL,
    shares_outstanding  DOUBLE,                  -- all share classes combined
    market_cap          DOUBLE,
    enterprise_value    DOUBLE,                  -- market cap + debt - cash (latest balance sheet)
    currency            VARCHAR,
    PRIMARY KEY (corporate_id, date)
);

-- --------------------------------------------------------------------------
-- 7. Calendar
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS events (
    event_id       VARCHAR PRIMARY KEY,          -- corporate_id|event_type|fiscal_year|fiscal_period
    corporate_id   INTEGER NOT NULL,
    event_type     VARCHAR NOT NULL,             -- earnings_release / annual_report / quarterly_report /
                                                 -- dividend_ex / agm / investor_day / split
    event_date     DATE NOT NULL,
    event_time     VARCHAR,                      -- BMO (before open) / AMC (after close) / unknown
    fiscal_year    INTEGER,
    fiscal_period  VARCHAR,
    status         VARCHAR NOT NULL,             -- estimated / confirmed / completed
    source         VARCHAR,
    accession_no   VARCHAR,                      -- filled once the filing exists
    notes          VARCHAR
);

-- --------------------------------------------------------------------------
-- 8. Data modules (add more of these as you go)
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS macro_series (
    series_id   VARCHAR PRIMARY KEY,             -- FRED id, e.g. 'DGS10'
    title       VARCHAR,
    frequency   VARCHAR,
    units       VARCHAR,
    source      VARCHAR
);

CREATE TABLE IF NOT EXISTS macro_observations (
    series_id   VARCHAR NOT NULL,
    date        DATE NOT NULL,
    value       DOUBLE,
    PRIMARY KEY (series_id, date)
);

-- Every loader run writes one row here, so you can see what is stale or broken.
CREATE TABLE IF NOT EXISTS load_log (
    run_at      TIMESTAMP DEFAULT current_timestamp,
    source      VARCHAR,
    table_name  VARCHAR,
    rows        INTEGER,
    status      VARCHAR,
    message     VARCHAR
);

-- --------------------------------------------------------------------------
-- Views: what the app reads
-- --------------------------------------------------------------------------

-- Company header info incl. continent and latest valuation.
CREATE OR REPLACE VIEW v_company_overview AS
SELECT c.*, co.country_name, co.continent,
       v.date AS valuation_date, v.market_cap, v.enterprise_value, v.currency AS valuation_currency
FROM companies c
LEFT JOIN countries co ON co.country_code = c.country_code
LEFT JOIN (
    SELECT * FROM valuation_daily
    QUALIFY row_number() OVER (PARTITION BY corporate_id ORDER BY date DESC) = 1
) v ON v.corporate_id = c.corporate_id;
"""
