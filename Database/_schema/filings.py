"""
Filing-based schema (SEC EDGAR): filings, fundamentals, ratios, institutional
holdings (13F), executives (DEF 14A) and subsidiaries (10-K Exhibit 21).
Loaded after corporate.SCHEMA_SQL.
"""

SCHEMA_SQL = r"""
-- --------------------------------------------------------------------------
-- 3. Filings and fundamentals (yearly / quarterly) - your "table 3", split up
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS filings (
    accession_no   VARCHAR PRIMARY KEY,          -- SEC accession no. (or your own ID for non-SEC docs)
    corporate_id   INTEGER NOT NULL,
    form_type      VARCHAR NOT NULL,             -- 10-K, 10-Q, 20-F, 8-K, DEF 14A, 13F-HR ...
    filed_date     DATE NOT NULL,
    period_end     DATE,
    fiscal_year    INTEGER,
    fiscal_period  VARCHAR,                      -- FY, Q1, Q2, Q3, Q4
    url            VARCHAR
);

-- Dictionary of every line item you store; keeps names consistent across companies.
CREATE TABLE IF NOT EXISTS financial_items (
    item_code      VARCHAR PRIMARY KEY,          -- 'revenue', 'net_income', 'total_assets', 'employees' ...
    label          VARCHAR NOT NULL,
    statement      VARCHAR NOT NULL,             -- IS / BS / CF / OTHER
    unit_type      VARCHAR NOT NULL,             -- currency / shares / persons / per_share / ratio
    period_type    VARCHAR NOT NULL,             -- duration (IS, CF) / instant (BS, shares)
    xbrl_tags      VARCHAR,                      -- comma-separated us-gaap tags that map to it
    sort_order     INTEGER
);

-- LONG format: one row per company x period x line item.
-- Stores restatements too (same period, later filed_date).
CREATE TABLE IF NOT EXISTS fundamentals (
    corporate_id   INTEGER NOT NULL,
    fiscal_year    INTEGER NOT NULL,
    fiscal_period  VARCHAR NOT NULL,             -- FY, Q1..Q4
    period_end     DATE NOT NULL,
    item_code      VARCHAR NOT NULL,             -- -> financial_items
    value          DOUBLE,
    currency       VARCHAR,                      -- NULL for shares / persons
    filed_date     DATE NOT NULL,                -- when the number became public
    accession_no   VARCHAR,                      -- -> filings
    source         VARCHAR,
    PRIMARY KEY (corporate_id, fiscal_year, fiscal_period, item_code, filed_date)
);

-- Ratios computed by your scripts (P/E, margins, ROIC, debt/equity, growth ...).
CREATE TABLE IF NOT EXISTS ratios (
    corporate_id   INTEGER NOT NULL,
    fiscal_year    INTEGER NOT NULL,
    fiscal_period  VARCHAR NOT NULL,
    ratio_code     VARCHAR NOT NULL,             -- 'gross_margin', 'pe', 'roic' ...
    value          DOUBLE,
    calc_date      DATE NOT NULL,                -- price-based ratios depend on this date
    PRIMARY KEY (corporate_id, fiscal_year, fiscal_period, ratio_code)
);

-- --------------------------------------------------------------------------
-- 4. Ownership: institutional holders
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS holders (
    holder_id      VARCHAR PRIMARY KEY,          -- your ID, e.g. 'H:0001364742' (filer CIK)
    name           VARCHAR NOT NULL,
    holder_type    VARCHAR,                      -- asset_manager / bank / pension / hedge_fund / insider ...
    cik            VARCHAR,
    lei            VARCHAR,
    country_code   VARCHAR,
    ticker         VARCHAR,                      -- the holder's own ticker, if it is listed
    isin           VARCHAR                       -- the holder's own ISIN, if it is listed
);

CREATE TABLE IF NOT EXISTS institutional_holdings (
    corporate_id       INTEGER NOT NULL,         -- company that is held
    instrument_id      VARCHAR NOT NULL,         -- which share class is held (its ticker / ISIN via instruments)
    report_date        DATE NOT NULL,            -- quarter end of the 13F
    holder_id          VARCHAR NOT NULL,         -- -> holders
    shares             DOUBLE NOT NULL,
    value              DOUBLE,
    currency           VARCHAR DEFAULT 'USD',
    pct_of_shares_out  DOUBLE,                   -- shares / shares outstanding x 100
    accession_no       VARCHAR,
    PRIMARY KEY (instrument_id, report_date, holder_id)
);

-- --------------------------------------------------------------------------
-- 5. People: executives and board, one snapshot per fiscal year
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS people (
    person_id      VARCHAR PRIMARY KEY,          -- your ID
    full_name      VARCHAR NOT NULL,
    birth_year     INTEGER
);

CREATE TABLE IF NOT EXISTS executives (
    corporate_id        INTEGER NOT NULL,
    fiscal_year         INTEGER NOT NULL,
    person_id           VARCHAR NOT NULL,        -- -> people
    title               VARCHAR NOT NULL,        -- 'Chief Executive Officer', 'Director' ...
    is_executive        BOOLEAN DEFAULT TRUE,
    is_board_member     BOOLEAN DEFAULT FALSE,
    since_date          DATE,                    -- in this role since
    total_compensation  DOUBLE,
    currency            VARCHAR,
    accession_no        VARCHAR,                 -- proxy statement (DEF 14A)
    PRIMARY KEY (corporate_id, fiscal_year, person_id, title)
);

-- --------------------------------------------------------------------------
-- 6. Subsidiaries, one snapshot per fiscal year (10-K Exhibit 21)
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS subsidiaries (
    corporate_id       INTEGER NOT NULL,
    fiscal_year        INTEGER NOT NULL,
    subsidiary_name    VARCHAR NOT NULL,
    jurisdiction       VARCHAR NOT NULL DEFAULT '',  -- country / state of incorporation
    ownership_pct      DOUBLE,
    accession_no       VARCHAR,
    PRIMARY KEY (corporate_id, fiscal_year, subsidiary_name, jurisdiction)
);

-- --------------------------------------------------------------------------
-- Views: what the app reads
-- --------------------------------------------------------------------------

-- Latest known value per period (ignores superseded, restated numbers).
CREATE OR REPLACE VIEW v_fundamentals_latest AS
SELECT * EXCLUDE (rn) FROM (
    SELECT f.*, row_number() OVER (
        PARTITION BY corporate_id, fiscal_year, fiscal_period, item_code
        ORDER BY filed_date DESC) AS rn
    FROM fundamentals f)
WHERE rn = 1;

-- Wide statement view for the company page (add columns as you add items).
CREATE OR REPLACE VIEW v_financials AS
SELECT corporate_id, fiscal_year, fiscal_period, max(period_end) AS period_end, any_value(currency) AS currency,
    max(value) FILTER (WHERE item_code = 'revenue')             AS revenue,
    max(value) FILTER (WHERE item_code = 'gross_profit')        AS gross_profit,
    max(value) FILTER (WHERE item_code = 'operating_income')    AS operating_income,
    max(value) FILTER (WHERE item_code = 'net_income')          AS net_income,
    max(value) FILTER (WHERE item_code = 'eps_diluted')         AS eps_diluted,
    max(value) FILTER (WHERE item_code = 'total_assets')        AS total_assets,
    max(value) FILTER (WHERE item_code = 'total_liabilities')   AS total_liabilities,
    max(value) FILTER (WHERE item_code = 'total_equity')        AS total_equity,
    max(value) FILTER (WHERE item_code = 'cash')                AS cash,
    max(value) FILTER (WHERE item_code = 'total_debt')          AS total_debt,
    max(value) FILTER (WHERE item_code = 'operating_cash_flow') AS operating_cash_flow,
    max(value) FILTER (WHERE item_code = 'capex')               AS capex,
    max(value) FILTER (WHERE item_code = 'shares_outstanding')  AS shares_outstanding,
    max(value) FILTER (WHERE item_code = 'employees')           AS employees
FROM v_fundamentals_latest
GROUP BY corporate_id, fiscal_year, fiscal_period;


CREATE OR REPLACE VIEW v_subsidiary_counts AS
SELECT corporate_id, fiscal_year, count(*) AS n_subsidiaries,
       count(DISTINCT jurisdiction) AS n_jurisdictions
FROM subsidiaries
GROUP BY corporate_id, fiscal_year;
"""
