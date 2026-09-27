"""
Table definitions for the three DuckDB files. create(con, db) runs on every
write connection; all statements are CREATE ... IF NOT EXISTS, so it is safe
to repeat. Add new columns with ALTER TABLE ... ADD COLUMN IF NOT EXISTS.
"""

from __future__ import annotations


# ----
# SHARED
# ----

LOAD_LOG = """
CREATE TABLE IF NOT EXISTS load_log (
    run_id        VARCHAR,
    step          VARCHAR,
    started_at    TIMESTAMP,
    finished_at   TIMESTAMP,
    rows_written  BIGINT,
    n_updated     INTEGER,
    n_skipped     INTEGER,
    n_failed      INTEGER,
    failed_items  VARCHAR[],          -- first 500 failed tickers / companies
    status        VARCHAR,            -- ok / partial / failed
    message       VARCHAR
);
"""


# ----
# prices.duckdb
# ----

PRICES = """
CREATE TABLE IF NOT EXISTS instruments (
    ticker        VARCHAR PRIMARY KEY,   -- Yahoo format; FRED copies of bond yields: '<ticker>_FRED'
    name          VARCHAR,               -- from the _tickers dict (FRED: incl. series id)
    asset_class   VARCHAR,               -- _tickers file name: bond, commodities, equities, ...
    group_name    VARCHAR,               -- dict name, e.g. ticker_us
    currency      VARCHAR,               -- quote currency as reported (GBp = pence)
    isin          VARCHAR,
    corporate_id  INTEGER,               -- equities only
    is_helper     BOOLEAN DEFAULT FALSE, -- FX pairs added only for USD conversion
    source        VARCHAR,               -- yfinance / fred
    first_date    DATE,
    last_date     DATE,
    updated_at    TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prices_daily (
    ticker         VARCHAR NOT NULL,
    date           DATE NOT NULL,
    open           DOUBLE,
    high           DOUBLE,
    low            DOUBLE,
    close          DOUBLE,               -- split-adjusted, not dividend-adjusted
    volume         DOUBLE,
    adj_close      DOUBLE,               -- split + dividend adjusted
    dividends      DOUBLE,
    stock_splits   DOUBLE,
    asset_class    VARCHAR,
    currency       VARCHAR,              -- quote currency as Yahoo reports it (GBp = pence)
    fx_to_usd      DOUBLE,               -- USD per 1 unit of `currency` (GBp: GBPUSD / 100)
    open_usd       DOUBLE,
    high_usd       DOUBLE,
    low_usd        DOUBLE,
    close_usd      DOUBLE,
    adj_close_usd  DOUBLE,
    source         VARCHAR,
    updated_at     TIMESTAMP,
    PRIMARY KEY (ticker, date)
);

-- daily USD rate per currency, built by the prices step from Yahoo FX pairs,
-- with FRED filling the years before Yahoo's history starts
CREATE TABLE IF NOT EXISTS fx_rates_daily (
    currency      VARCHAR NOT NULL,      -- ISO major currency, e.g. EUR
    date          DATE NOT NULL,
    usd_per_unit  DOUBLE,
    source        VARCHAR,               -- e.g. 'yahoo:EURUSD=X', 'fred:DEXUSEU', 'fred:DEXGEUS/DEM'
    PRIMARY KEY (currency, date)
);

CREATE TABLE IF NOT EXISTS macro_series (
    series_id            VARCHAR PRIMARY KEY,
    title                VARCHAR,
    frequency            VARCHAR,
    units                VARCHAR,
    seasonal_adjustment  VARCHAR,
    last_updated         TIMESTAMP
);

CREATE TABLE IF NOT EXISTS macro_observations (
    series_id  VARCHAR NOT NULL,
    date       DATE NOT NULL,
    value      DOUBLE,
    PRIMARY KEY (series_id, date)
);

-- intraday bookkeeping per ticker: the day its history was downloaded split-adjusted (a later split
-- forces a full reload), the Alpaca feed used and the last stored minute
CREATE TABLE IF NOT EXISTS intraday_state (
    ticker            VARCHAR PRIMARY KEY,
    split_basis_date  DATE,
    feed              VARCHAR,
    last_ts           TIMESTAMP,          -- UTC
    updated_at        TIMESTAMP
);

CREATE TABLE IF NOT EXISTS market_cap_daily (
    corporate_id        INTEGER NOT NULL,
    date                DATE NOT NULL,
    primary_ticker      VARCHAR,          -- listing whose close is used
    shares_outstanding  DOUBLE,           -- all share classes
    market_cap          DOUBLE,           -- in `currency`
    currency            VARCHAR,
    market_cap_usd      DOUBLE,
    shares_source       VARCHAR,          -- sec_cover_page / yfinance_current
    PRIMARY KEY (corporate_id, date)
);

-- Yahoo's stock_splits split into the real share split and a price-only part (spin-offs);
-- rebuilt by the marketcap step. yahoo_factor = split_factor x price_only_factor
CREATE TABLE IF NOT EXISTS corporate_actions (
    ticker              VARCHAR NOT NULL,
    date                DATE NOT NULL,        -- ex-date
    corporate_id        INTEGER,
    yahoo_factor        DOUBLE,               -- prices_daily.stock_splits
    split_factor        DOUBLE,               -- applied to share counts (2 = 2:1, 0.5 = 1:2, 1 = none)
    price_only_factor   DOUBLE,               -- spin-off part: adjusts prices only
    method              VARCHAR,              -- sec_counts / yfinance_after / price_only_review / override /
                                              -- no_sec_data (Yahoo factor = split, as for non-SEC companies)
    sec_before_date     DATE,
    sec_before_shares   DOUBLE,
    sec_after_date      DATE,
    sec_after_shares    DOUBLE,
    note                VARCHAR,
    PRIMARY KEY (ticker, date)
);
"""


# ----
# companies.duckdb
# ----

# wide financial columns, same in the quarterly and yearly table
FINANCIAL_COLUMNS = [
    # income statement
    'revenue', 'revenue_incl_excise', 'cost_of_revenue', 'gross_profit', 'rnd_expense', 'sga_expense',
    'operating_expenses', 'operating_income', 'interest_expense', 'pretax_income',
    'income_tax', 'net_income', 'net_income_to_common', 'eps_basic', 'eps_diluted', 'shares_basic_wavg',
    'shares_diluted_wavg', 'depreciation_amortization', 'ebitda',
    # balance sheet
    'cash', 'short_term_investments', 'receivables', 'inventory', 'current_assets',
    'ppe_net', 'goodwill', 'intangibles', 'total_assets', 'accounts_payable',
    'current_liabilities', 'short_term_debt', 'long_term_debt', 'total_debt',
    'net_debt', 'total_liabilities', 'total_equity', 'noncontrolling_interest', 'redeemable_equity',
    'retained_earnings',
    # cash flow
    'operating_cash_flow', 'capex', 'free_cash_flow', 'investing_cash_flow',
    'financing_cash_flow', 'dividends_paid', 'share_buybacks', 'stock_based_comp',
    # other
    'shares_outstanding', 'dividends_per_share', 'employees',
    # margins
    'gross_margin', 'operating_margin', 'net_margin',
]

HOLDER_STRUCT    = ('STRUCT(holder_name VARCHAR, holder_cik VARCHAR, holder_group VARCHAR, shares DOUBLE, '
                    'value_usd DOUBLE, pct_of_shares_out DOUBLE, change_shares_vs_prev_quarter DOUBLE)')
EXECUTIVE_STRUCT = ('STRUCT(name VARCHAR, sec_name VARCHAR, cik VARCHAR, title VARCHAR, role VARCHAR, '
                    'is_officer BOOLEAN, is_director BOOLEAN)')
SUBSIDIARY_STRUCT = 'STRUCT(name VARCHAR, jurisdiction VARCHAR)'

_FIN_COLS_SQL = ',\n    '.join(f'{c} DOUBLE' for c in FINANCIAL_COLUMNS)

_PERIOD_COLS = """
    period_start          DATE,
    period_end            DATE,
    filed_date            DATE,
    form_type             VARCHAR,
    accession_no          VARCHAR,
    currency              VARCHAR,"""

COMPANIES = f"""
CREATE TABLE IF NOT EXISTS company_info (
    corporate_id          INTEGER PRIMARY KEY,
    primary_ticker        VARCHAR,
    tickers               VARCHAR[],
    name                  VARCHAR,
    legal_name            VARCHAR,
    cik                   INTEGER,
    isin                  VARCHAR,
    cusip                 VARCHAR,          -- characters 3-11 of a US ISIN
    sec_filer             BOOLEAN,
    sec_form_type         VARCHAR,          -- 10-K / 20-F / 40-F
    exchange              VARCHAR,
    country               VARCHAR,
    continent             VARCHAR,
    trading_currency      VARCHAR,
    reporting_currency    VARCHAR,
    sector                VARCHAR,
    industry              VARCHAR,
    sic_code              VARCHAR,
    sic_description       VARCHAR,
    business_description  VARCHAR,
    hq_street             VARCHAR,
    hq_city               VARCHAR,
    hq_state              VARCHAR,
    hq_postal_code        VARCHAR,
    hq_country            VARCHAR,
    website               VARCHAR,
    fiscal_year_end       VARCHAR,          -- 'MMDD' as reported by SEC
    employees             BIGINT,
    updated_at            TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fundamentals_quarterly (
    corporate_id          INTEGER NOT NULL,
    fiscal_year           INTEGER NOT NULL,
    fiscal_quarter        INTEGER NOT NULL, -- 1..4
    {_PERIOD_COLS.strip()}
    q4_derived            BOOLEAN,          -- Q4 = full year - Q1..Q3
    {_FIN_COLS_SQL},
    earnings_date         DATE,
    earnings_date_status  VARCHAR,          -- estimated / confirmed / reported
    top_holders           {HOLDER_STRUCT}[],
    updated_at            TIMESTAMP,
    PRIMARY KEY (corporate_id, fiscal_year, fiscal_quarter)
);

CREATE TABLE IF NOT EXISTS fundamentals_yearly (
    corporate_id          INTEGER NOT NULL,
    fiscal_year           INTEGER NOT NULL,
    {_PERIOD_COLS.strip()}
    {_FIN_COLS_SQL},
    annual_report_date    DATE,             -- 10-K / 20-F filing date
    earnings_date         DATE,             -- Q4 / full-year earnings release
    earnings_date_status  VARCHAR,
    top_holders           {HOLDER_STRUCT}[],
    executives            {EXECUTIVE_STRUCT}[],
    subsidiaries          {SUBSIDIARY_STRUCT}[],
    n_subsidiaries        INTEGER,
    updated_at            TIMESTAMP,
    PRIMARY KEY (corporate_id, fiscal_year)
);

-- bookkeeping: every SEC filing seen per company (drives incremental updates)
CREATE TABLE IF NOT EXISTS sec_filings (
    accession_no   VARCHAR PRIMARY KEY,
    corporate_id   INTEGER,
    cik            INTEGER,
    form_type      VARCHAR,
    filed_date     DATE,
    report_date    DATE,
    items          VARCHAR,                 -- 8-K item numbers, e.g. '2.02,9.01'
    primary_doc    VARCHAR
);

-- bookkeeping: bulk data sets (13F, insider) already processed
CREATE TABLE IF NOT EXISTS processed_datasets (
    dataset        VARCHAR PRIMARY KEY,     -- e.g. '13f:2026q2', 'insider:2026q2'
    processed_at   TIMESTAMP,
    rows_used      BIGINT
);
"""


# ----
# raw.duckdb
# ----

RAW = """
CREATE TABLE IF NOT EXISTS clinical_trials_raw (
    nct_id            VARCHAR NOT NULL,
    raw_json          JSON,
    last_change_date  DATE,
    version           INTEGER NOT NULL,
    downloaded_at     TIMESTAMP,
    json_hash         VARCHAR,
    is_latest         BOOLEAN,
    change_flag       BOOLEAN,
    changed_sections  VARCHAR[],
    PRIMARY KEY (nct_id, version)
);

-- SEC Forms 3 / 4 / 5: non-derivative transactions of the covered US companies,
-- one row per transaction and reporting owner (joint filings have several owners)
CREATE TABLE IF NOT EXISTS insider_transactions_raw (
    accession_no         VARCHAR NOT NULL,
    trans_sk             VARCHAR NOT NULL,     -- SEC row id within the data set
    owner_cik            VARCHAR NOT NULL,
    corporate_id         INTEGER,
    form_type            VARCHAR,              -- 3 / 4 / 5 (and /A)
    filing_date          DATE,
    trans_date           DATE,
    trans_code           VARCHAR,              -- P purchase, S sale, A award, M option exercise, F tax, G gift, ...
    acquired_disposed    VARCHAR,              -- A / D
    shares               DOUBLE,
    price                DOUBLE,
    shares_owned_after   DOUBLE,
    direct_indirect      VARCHAR,              -- D / I
    security_title       VARCHAR,
    owner_name           VARCHAR,              -- as filed ('COOK TIMOTHY D')
    owner_readable_name  VARCHAR,              -- 'Timothy D. Cook'
    relationship         VARCHAR,              -- Director / Officer / TenPercentOwner / Other (comma-separated)
    title                VARCHAR,
    role                 VARCHAR,              -- CEO / CFO / COO / President / Chair / General Counsel / Director / Other
    is_officer           BOOLEAN,
    is_director          BOOLEAN,
    is_ten_pct_owner     BOOLEAN,
    downloaded_at        TIMESTAMP,
    PRIMARY KEY (accession_no, trans_sk, owner_cik)
);
"""


# columns added after the first release (existing databases get them on connect)
PRICES_MIGRATIONS = """
ALTER TABLE instruments ADD COLUMN IF NOT EXISTS status VARCHAR DEFAULT 'active';   -- active / no_data
ALTER TABLE instruments ADD COLUMN IF NOT EXISTS consecutive_failures INTEGER DEFAULT 0;
ALTER TABLE instruments ADD COLUMN IF NOT EXISTS last_checked DATE;               -- last download attempt
ALTER TABLE macro_series ADD COLUMN IF NOT EXISTS category VARCHAR;               -- macro / bond_yield / fx_helper
ALTER TABLE macro_series ADD COLUMN IF NOT EXISTS status VARCHAR DEFAULT 'active';  -- active / no_data
ALTER TABLE macro_series ADD COLUMN IF NOT EXISTS consecutive_failures INTEGER DEFAULT 0;
ALTER TABLE macro_series ADD COLUMN IF NOT EXISTS last_checked DATE;
"""

COMPANIES_MIGRATIONS = """
-- company status from corporate_ids.csv: active / removed / reassigned / merged
ALTER TABLE company_info ADD COLUMN IF NOT EXISTS status VARCHAR DEFAULT 'active';
-- yfinance share count (all classes: impliedSharesOutstanding) for market cap of non-SEC companies
ALTER TABLE company_info ADD COLUMN IF NOT EXISTS yf_shares_outstanding DOUBLE;
ALTER TABLE company_info ADD COLUMN IF NOT EXISTS yf_shares_date DATE;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS shares_as_of DATE;   -- date of shares_outstanding
ALTER TABLE fundamentals_yearly    ADD COLUMN IF NOT EXISTS shares_as_of DATE;
-- mezzanine equity (redeemable noncontrolling interests): assets = liabilities + redeemable + equity
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS redeemable_equity DOUBLE;
ALTER TABLE fundamentals_yearly    ADD COLUMN IF NOT EXISTS redeemable_equity DOUBLE;
-- total_equity = equity attributable to the parent; NCI separately
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS noncontrolling_interest DOUBLE;
ALTER TABLE fundamentals_yearly    ADD COLUMN IF NOT EXISTS noncontrolling_interest DOUBLE;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS revenue_incl_excise DOUBLE;
ALTER TABLE fundamentals_yearly    ADD COLUMN IF NOT EXISTS revenue_incl_excise DOUBLE;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS gross_profit_derived BOOLEAN;   -- revenue - cost_of_revenue
ALTER TABLE fundamentals_yearly    ADD COLUMN IF NOT EXISTS gross_profit_derived BOOLEAN;
ALTER TABLE fundamentals_yearly    ADD COLUMN IF NOT EXISTS employees_source VARCHAR;
-- Exhibit 21: list says it contains only significant subsidiaries / list carried from the previous year
ALTER TABLE fundamentals_yearly    ADD COLUMN IF NOT EXISTS subsidiaries_significant_only BOOLEAN;
ALTER TABLE fundamentals_yearly    ADD COLUMN IF NOT EXISTS subsidiaries_carried_forward BOOLEAN;
-- all / omits_insignificant (Item 601(b)(21)(ii) clause) / significant_only (list says so explicitly)
ALTER TABLE fundamentals_yearly    ADD COLUMN IF NOT EXISTS subsidiaries_scope VARCHAR;
-- net income available to common shareholders (after preferred dividends etc.; basis of EPS)
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS net_income_to_common DOUBLE;
ALTER TABLE fundamentals_yearly    ADD COLUMN IF NOT EXISTS net_income_to_common DOUBLE;
-- weighted share counts tagged in thousands / millions, corrected via EPS
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS shares_scale_fixed BOOLEAN;
ALTER TABLE fundamentals_yearly    ADD COLUMN IF NOT EXISTS shares_scale_fixed BOOLEAN;
-- earnings calendar: time of day (bmo / during / amc) and the EDGAR acceptance time (UTC) of filings
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS earnings_time VARCHAR;
ALTER TABLE fundamentals_yearly    ADD COLUMN IF NOT EXISTS earnings_time VARCHAR;
ALTER TABLE sec_filings            ADD COLUMN IF NOT EXISTS acceptance_time TIMESTAMP;
-- EDGAR acceptance time of the earnings 8-K in New York time (source of earnings_time)
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS earnings_accepted_at TIMESTAMP;
ALTER TABLE fundamentals_yearly    ADD COLUMN IF NOT EXISTS earnings_accepted_at TIMESTAMP;
-- 13F: sum of all institutions' shares / shares outstanding (%), number of filers, 13F report date used
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS inst_ownership_pct DOUBLE;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS n_institutional_holders INTEGER;
ALTER TABLE fundamentals_quarterly ADD COLUMN IF NOT EXISTS holders_report_date DATE;       -- e.g. 'yfinance 2026-09-26'

-- bookkeeping: newest SEC filing seen per company when its fundamentals were last built
CREATE TABLE IF NOT EXISTS fundamentals_state (
    corporate_id       INTEGER PRIMARY KEY,
    last_filing_date   DATE,
    processed_at       TIMESTAMP
);
ALTER TABLE fundamentals_state ADD COLUMN IF NOT EXISTS build_key VARCHAR;       -- code version + CIKs used
-- '<number of facts>|<newest filed date>' of the company facts used in the last build
ALTER TABLE fundamentals_state ADD COLUMN IF NOT EXISTS facts_fingerprint VARCHAR;
"""

RAW_MIGRATIONS = """
-- transaction date after the filing date or in the future (typos in the SEC data); rows are kept
ALTER TABLE insider_transactions_raw ADD COLUMN IF NOT EXISTS date_suspect BOOLEAN;
"""

SCHEMAS = {
    'prices':    PRICES + LOAD_LOG + PRICES_MIGRATIONS,
    'companies': COMPANIES + LOAD_LOG + COMPANIES_MIGRATIONS,
    'raw':       RAW + LOAD_LOG + RAW_MIGRATIONS,
}

# table -> (date column for "latest", step that writes it); used by `run.py status`
TABLE_INFO = {
    'prices': {
        'instruments':         ('last_date', 'prices'),
        'prices_daily':        ('date', 'prices'),
        'fx_rates_daily':      ('date', 'prices'),
        'macro_series':        ('last_updated', 'fred'),
        'macro_observations':  ('date', 'fred'),
        'market_cap_daily':    ('date', 'marketcap'),
        'corporate_actions':   ('date', 'marketcap'),
    },
    'companies': {
        'company_info':           ('updated_at', 'companies'),
        'fundamentals_quarterly': ('period_end', 'fundamentals'),
        'fundamentals_yearly':    ('period_end', 'fundamentals'),
        'sec_filings':            ('filed_date', 'fundamentals'),
        'processed_datasets':     ('processed_at', 'holders'),
    },
    'raw': {
        'clinical_trials_raw': ('last_change_date', 'trials'),
        'insider_transactions_raw': ('trans_date', 'executives'),
    },
}

# step -> database whose load_log records it
STEP_DB = {
    'ids': 'companies', 'companies': 'companies', 'prices': 'prices', 'fred': 'prices',
    'fundamentals': 'companies', 'marketcap': 'prices', 'calendar': 'companies',
    'holders': 'companies', 'executives': 'companies', 'subsidiaries': 'companies',
    'intraday': 'prices', 'trials': 'raw',
}


# list-of-struct columns whose struct gained fields: recreated (empty) when the stored type differs;
# the steps that fill them rebuild them on their next run
STRUCT_COLUMNS = {
    'companies': [('fundamentals_quarterly', 'top_holders', f'{HOLDER_STRUCT}[]'),
                  ('fundamentals_yearly', 'top_holders', f'{HOLDER_STRUCT}[]'),
                  ('fundamentals_yearly', 'executives', f'{EXECUTIVE_STRUCT}[]')],
}


def _norm_type(t: str) -> str:
    return ''.join(t.upper().split()).replace('"', '')


def create(con, db: str) -> None:
    con.execute(SCHEMAS[db])
    for table, column, typ in STRUCT_COLUMNS.get(db, []):
        row = con.execute("""SELECT data_type FROM duckdb_columns()
                             WHERE table_name = ? AND column_name = ?""", [table, column]).fetchone()
        if row and _norm_type(row[0]) != _norm_type(typ):
            con.execute(f'ALTER TABLE {table} DROP COLUMN {column}')
            con.execute(f'ALTER TABLE {table} ADD COLUMN {column} {typ}')
