"""
All pipeline settings in one place. Change values here, not in the pipeline modules.
"""

from pathlib import Path


# ----
# PATHS
# ----

BASE_DIR      = Path(__file__).resolve().parent            # .../Database
PROJECT_DIR   = BASE_DIR.parent                            # .../Python
TICKERS_DIR   = BASE_DIR / '_tickers'
API_KEYS_FILE = PROJECT_DIR / 'api_keys.txt'

DATA_DIR      = BASE_DIR / 'data'
CACHE_DIR     = DATA_DIR / 'cache'                         # downloaded zip / json files
INTRADAY_DIR  = DATA_DIR / 'intraday'
REVIEW_DIR    = DATA_DIR / 'review'                        # csv files for manual checks
LOG_DIR       = DATA_DIR / 'logs'

PRICES_DB     = DATA_DIR / 'prices.duckdb'
COMPANIES_DB  = DATA_DIR / 'companies.duckdb'
RAW_DB        = DATA_DIR / 'raw.duckdb'

CORPORATE_IDS_CSV     = DATA_DIR / 'corporate_ids.csv'                 # single source of truth
# corrections to the ticker files (ticker, change, old_name, new_name, note), applied once by the ids step
TICKER_CHANGES_CSV    = DATA_DIR / 'review' / 'equities_changes.csv'


# ----
# TICKER FILES
# ----

EQUITIES_ASSET_CLASS = 'equities'
US_GROUP             = 'ticker_us'          # dict in equities.py with the US companies
FRED_ASSET_CLASSES   = {'macro'}            # _tickers files that hold FRED series, not Yahoo tickers
# a ticker listed in several files (e.g. ADM in equities + commodities, ^TNX in bond + indices)
# gets the first asset class of this list
ASSET_CLASS_PRIORITY = ['equities', 'bond', 'crypto', 'forex', 'commodities', 'indices', 'sectors', 'etfs',
                        'sentiment']


# ----
# DAILY PRICES
# ----

START_DATE          = '1995-01-01'
PRICE_OVERLAP_DAYS  = 5                     # re-download window to detect adj_close changes
ADJ_CLOSE_TOLERANCE = 1e-4                  # relative difference that counts as "changed"
YF_BATCH_SIZE       = 50                    # tickers per yfinance download call

NO_DATA_AFTER_FAILURES = 3                  # consecutive failed runs -> status 'no_data', skipped
NO_DATA_RECHECK_DAYS   = 30                 # 'no_data' tickers are tried again after this many days
RECENT_DATA_DAYS       = 14                 # empty download for a ticker with data this recent is not a failure (holidays)

# no USD conversion (USD columns stay empty): yields, volatility and other non-money levels
NO_USD_TICKERS       = {'^IRX', '^FVX', '^TNX', '^TYX', '^SKEW', '^NYHL'}
NO_USD_SUFFIXES      = ('_FRED',)              # FRED yield copies
NO_USD_NAME_PATTERN  = r'(?i)volatility index'  # ^VIX, ^VXN, ^OVX, ^V2TX, ...

# quote currency for tickers where Yahoo reports none
CURRENCY_OVERRIDES = {
    '^MERV': 'ARS',                         # S&P Merval, Buenos Aires
    'JKM=F': 'USD',                         # NYMEX JKM LNG future
}

FX_FFILL_LIMIT      = 5                     # max calendar days an FX rate is carried forward
# FRED daily FX series used where Yahoo's FX history starts too late or has gaps > FX_FFILL_LIMIT,
# in order of use. (FRED has no daily DEM series; EXGEUS is a monthly average.)
# 'usd_per' = USD per 1 unit of the currency, 'per_usd' = currency units per 1 USD.
FRED_FX_SERIES = {
    'EUR': [('DEXUSEU', 'usd_per'), ('EXGEUS', 'dem_per_usd_monthly')],   # before 1999: monthly DEM avg / 1.95583
    'GBP': [('DEXUSUK', 'usd_per')], 'AUD': [('DEXUSAL', 'usd_per')], 'NZD': [('DEXUSNZ', 'usd_per')],
    'JPY': [('DEXJPUS', 'per_usd')], 'CAD': [('DEXCAUS', 'per_usd')], 'CHF': [('DEXSZUS', 'per_usd')],
    'CNY': [('DEXCHUS', 'per_usd')], 'HKD': [('DEXHKUS', 'per_usd')], 'INR': [('DEXINUS', 'per_usd')],
    'KRW': [('DEXKOUS', 'per_usd')], 'SGD': [('DEXSIUS', 'per_usd')], 'TWD': [('DEXTAUS', 'per_usd')],
    'THB': [('DEXTHUS', 'per_usd')], 'MYR': [('DEXMAUS', 'per_usd')], 'SEK': [('DEXSDUS', 'per_usd')],
    'DKK': [('DEXDNUS', 'per_usd')], 'NOK': [('DEXNOUS', 'per_usd')], 'BRL': [('DEXBZUS', 'per_usd')],
    'MXN': [('DEXMXUS', 'per_usd')], 'ZAR': [('DEXSFUS', 'per_usd')],
    # no daily FRED series: OECD monthly average of daily rates (currency per USD)
    'IDR': [('CCUSMA02IDM618N', 'per_usd_monthly')],
    'ILS': [('CCUSMA02ILM618N', 'per_usd_monthly')],
    'PLN': [('CCUSMA02PLM618N', 'per_usd_monthly')],
    'ARS': [('ARGCCUSMA02STM', 'per_usd_monthly')],       # after the peg (FX_PEGS), e.g. 2002-2003 Yahoo gaps
}
# fixed pegs used where no market rate exists (before the FRED series above):
# currency -> (USD per unit, last day of the peg, note)
FX_PEGS = {
    'ARS': (1.0, '2002-01-06', 'peg:ARS=USD 1:1 convertibility law'),
}
EUR_PER_DEM = 1 / 1.95583
# quote unit -> (currency, divisor): London GBp in pence, etc.
MINOR_UNITS = {
    'GBp': ('GBP', 100),
    'GBX': ('GBP', 100),
    'ZAc': ('ZAR', 100),
    'ILA': ('ILS', 100),
    'USX': ('USD', 100),                    # US cents, e.g. grain / softs futures (ZL=F, KC=F)
}


# ----
# FRED
# ----

# bond ticker -> FRED series; stored as extra instrument '<ticker>_FRED'
FRED_BOND_MAP = {
    '^IRX': 'DTB3',
    '^FVX': 'DGS5',
    '^TNX': 'DGS10',
    '^TYX': 'DGS30',
}
MACRO_OVERLAP_DAYS = 60                     # FRED revises recent values


# ----
# COMPANIES
# ----

COMPANY_REFRESH_DAYS  = 30
YF_WORKERS            = 8                   # parallel yfinance .info calls
MATCH_NON_US_TO_SEC   = False               # True: look for 20-F / 40-F filers among non-US companies
SEC_CIK_OVERRIDES     = {}                  # ticker -> CIK (int) or None to force "not an SEC filer"
# current CIK -> older CIKs of the same company (reorganisations); the fundamentals step
# reads all of them and keeps each filing (accession_no) only once
SEC_PREDECESSOR_CIKS  = {
    2115436: [34088],                       # ExxonMobil Holdings Corp <- Exxon Mobil Corp (2026)
    1652044: [1288776],                     # Alphabet Inc. <- Google Inc. (2015)
}
PRIMARY_TICKER_OVERRIDES = {}               # corporate_id -> ticker used for market cap
# market cap: an SEC share count is used for at most this many days after its date; after that
# (e.g. a company that stopped tagging its cover page) the current yfinance count is used
SHARES_STALE_DAYS = 400
# primary tickers whose SEC share counts do not fit the listing's price (share classes) -> yfinance only
MARKETCAP_YFINANCE_ONLY = {'BRK-B'}          # SEC counts are Class A shares, primary listing is Class B
# completed deals whose 8-K didn't use item 2.01: (primary ticker, completion date) -> note. The new SEC share
# count applies from that date (shares_source 'sec_after_deal') and the day is not a market cap check flag.
KNOWN_DEALS = {
    ('PNR', '2012-09-28'): 'Tyco Flow Control merger; 8-K of 2012-10-01 reports it under item 8.01',
}
# Yahoo records spin-offs as fractional stock_splits. For SEC companies the real split factor is
# SEC shares after / before the event, rounded to a clean ratio (within SPLIT_RATIO_TOLERANCE);
# the rest of Yahoo's factor is price-only. Counts more than SPLIT_SEC_MAX_DAYS from the event, or
# no clean ratio -> data/review/corporate_actions_review.csv, price-only until reviewed.
SPLIT_RATIO_TOLERANCE = 0.03
SPLIT_SEC_MAX_DAYS    = 120
# reviewed events: (ticker, 'YYYY-MM-DD') -> real split factor (1.0 = spin-off only, 2.0 = 2:1, 0.5 = 1:2)
CORPORATE_ACTION_OVERRIDES = {
    ('MA',    '2014-01-22'): 10.0,              # 10:1 split; SEC counts 2010 / 2026 only
    ('CME',   '2012-07-23'): 5.0,               # 5:1 split
    ('NKE',   '2015-12-24'): 2.0,               # 2:1 split
    ('AOS',   '2016-10-06'): 2.0,               # 2:1 split
    ('DUK',   '2012-07-03'): 1 / 3,             # 1:3 reverse split, same day as the Progress Energy merger
    ('GOOGL', '2014-04-03'): 2.0,               # Class C share dividend (2:1); SEC counts then Class A only
    ('BALL',  '2011-02-16'): 2.0,               # 2:1 split; SEC ratio 1.916 (split + ~4 % buybacks in 5 months)
    ('WBD',   '2014-08-07'): 2.0,               # Discovery Class C share dividend (2:1): convertible preferred
                                                # converts into 2 instead of 1 common share from 2014-08-06
}


# Yahoo ticker suffix -> country of that exchange (no suffix = United States).
# Used to find a non-SEC company's home listing (primary ticker).
EXCHANGE_SUFFIXES = {
    '.T': 'Japan', '.HK': 'Hong Kong', '.SS': 'China', '.SZ': 'China', '.NS': 'India',
    '.BO': 'India', '.TW': 'Taiwan', '.TWO': 'Taiwan', '.KS': 'South Korea', '.KQ': 'South Korea',
    '.SI': 'Singapore', '.JK': 'Indonesia', '.KL': 'Malaysia', '.BK': 'Thailand',
    '.DE': 'Germany', '.F': 'Germany', '.PA': 'France', '.AS': 'Netherlands', '.MI': 'Italy',
    '.MC': 'Spain', '.ST': 'Sweden', '.CO': 'Denmark', '.HE': 'Finland', '.BR': 'Belgium',
    '.VI': 'Austria', '.LS': 'Portugal', '.WA': 'Poland', '.L': 'United Kingdom',
    '.SW': 'Switzerland', '.OL': 'Norway', '.IR': 'Ireland', '.TO': 'Canada', '.V': 'Canada',
    '.SA': 'Brazil', '.MX': 'Mexico', '.AX': 'Australia', '.NZ': 'New Zealand',
    '.JO': 'South Africa', '.TA': 'Israel', '.SR': 'Saudi Arabia', '.IS': 'Turkey',
}

# country name (as yfinance writes it) -> (ISO 3166 alpha-2, continent)
COUNTRIES = {
    'United States': ('US', 'North America'), 'Canada': ('CA', 'North America'),
    'Mexico': ('MX', 'North America'), 'Bermuda': ('BM', 'North America'),
    'Cayman Islands': ('KY', 'North America'), 'Bahamas': ('BS', 'North America'),
    'Panama': ('PA', 'North America'), 'Puerto Rico': ('PR', 'North America'),
    'Brazil': ('BR', 'South America'), 'Argentina': ('AR', 'South America'),
    'Chile': ('CL', 'South America'), 'Colombia': ('CO', 'South America'),
    'Peru': ('PE', 'South America'), 'Uruguay': ('UY', 'South America'),
    'United Kingdom': ('GB', 'Europe'), 'Ireland': ('IE', 'Europe'), 'Germany': ('DE', 'Europe'),
    'France': ('FR', 'Europe'), 'Netherlands': ('NL', 'Europe'), 'Belgium': ('BE', 'Europe'),
    'Luxembourg': ('LU', 'Europe'), 'Switzerland': ('CH', 'Europe'), 'Austria': ('AT', 'Europe'),
    'Italy': ('IT', 'Europe'), 'Spain': ('ES', 'Europe'), 'Portugal': ('PT', 'Europe'),
    'Sweden': ('SE', 'Europe'), 'Denmark': ('DK', 'Europe'), 'Norway': ('NO', 'Europe'),
    'Finland': ('FI', 'Europe'), 'Iceland': ('IS', 'Europe'), 'Poland': ('PL', 'Europe'),
    'Czech Republic': ('CZ', 'Europe'), 'Hungary': ('HU', 'Europe'), 'Greece': ('GR', 'Europe'),
    'Jersey': ('JE', 'Europe'), 'Guernsey': ('GG', 'Europe'), 'Isle of Man': ('IM', 'Europe'),
    'Monaco': ('MC', 'Europe'), 'Malta': ('MT', 'Europe'), 'Cyprus': ('CY', 'Europe'),
    'Liechtenstein': ('LI', 'Europe'), 'Gibraltar': ('GI', 'Europe'), 'Turkey': ('TR', 'Europe'),
    'Russia': ('RU', 'Europe'), 'Romania': ('RO', 'Europe'), 'Slovenia': ('SI', 'Europe'),
    'Estonia': ('EE', 'Europe'), 'Lithuania': ('LT', 'Europe'), 'Latvia': ('LV', 'Europe'),
    'Japan': ('JP', 'Asia'), 'China': ('CN', 'Asia'), 'Hong Kong': ('HK', 'Asia'),
    'Macau': ('MO', 'Asia'), 'Taiwan': ('TW', 'Asia'), 'South Korea': ('KR', 'Asia'),
    'India': ('IN', 'Asia'), 'Singapore': ('SG', 'Asia'), 'Indonesia': ('ID', 'Asia'),
    'Malaysia': ('MY', 'Asia'), 'Thailand': ('TH', 'Asia'), 'Philippines': ('PH', 'Asia'),
    'Vietnam': ('VN', 'Asia'), 'Israel': ('IL', 'Asia'), 'Saudi Arabia': ('SA', 'Asia'),
    'United Arab Emirates': ('AE', 'Asia'), 'Qatar': ('QA', 'Asia'), 'Kazakhstan': ('KZ', 'Asia'),
    'Australia': ('AU', 'Oceania'), 'New Zealand': ('NZ', 'Oceania'),
    'Papua New Guinea': ('PG', 'Oceania'),
    'South Africa': ('ZA', 'Africa'), 'Nigeria': ('NG', 'Africa'), 'Egypt': ('EG', 'Africa'),
    'Morocco': ('MA', 'Africa'), 'Kenya': ('KE', 'Africa'), 'Zambia': ('ZM', 'Africa'),
}


# ----
# FUNDAMENTALS / FILINGS
# ----

FUNDAMENTALS_START_YEAR = 2009
HOLDERS_TOP_N           = 20
# ticker -> 13F CUSIPs, when automatic matching fails (13F issuer names 'WEYERHAEUSER CO MTN BE', 'SMUCKER J M CO')
CUSIP_OVERRIDES = {
    'WY':  ['962166104'],
    'SJM': ['832696405'],
}
# holder_group: related 13F filers (regex on the filer name, first match wins)
HOLDER_GROUPS = [
    ('Vanguard',        r'VANGUARD'),
    ('BlackRock',       r'BLACKROCK'),
    ('Fidelity',        r'\bFMR\b|FIDELITY'),
    ('State Street',    r'STATE STREET|SSGA'),
    ('Geode',           r'GEODE'),
    ('Capital Group',   r'CAPITAL RESEARCH|CAPITAL INTERNATIONAL|CAPITAL WORLD|CAPITAL GROUP'),
    ('Morgan Stanley',  r'MORGAN STANLEY'),
    ('JPMorgan',        r'JPMORGAN|J\.P\. MORGAN'),
    ('Goldman Sachs',   r'GOLDMAN SACHS'),
    ('Bank of America', r'BANK OF AMERICA|MERRILL LYNCH'),
    ('Wells Fargo',     r'WELLS FARGO'),
    ('UBS',             r'\bUBS\b'),
    ('Northern Trust',  r'NORTHERN TRUST'),
    ('T. Rowe Price',   r'T\. ?ROWE PRICE|PRICE T ROWE'),
    ('Invesco',         r'INVESCO'),
    ('Charles Schwab',  r'SCHWAB'),
    ('Dimensional',     r'DIMENSIONAL FUND'),
    ('Wellington',      r'WELLINGTON MANAGEMENT'),
    ('Norges Bank',     r'NORGES BANK'),
    ('Berkshire Hathaway', r'BERKSHIRE HATHAWAY'),
    ('Legal & General', r'LEGAL & GENERAL|LEGAL AND GENERAL'),
    ('Nuveen / TIAA',   r'NUVEEN|TIAA|TEACHERS ADVISORS'),
    ('Franklin',        r'FRANKLIN RESOURCES|FRANKLIN TEMPLETON'),
    ('Amundi',          r'AMUNDI'),
    ('Deutsche Bank / DWS', r'DEUTSCHE BANK|\bDWS\b'),
]
HOLDERS_FIRST_RUN_QUARTERS = 4
EXECUTIVES_YEARS        = 5
SUBSIDIARIES_YEARS      = 5


# ----
# INTRADAY
# ----

INTRADAY_HISTORY_DAYS   = 365
INTRADAY_BASE_INTERVAL  = '1m'              # downloaded from Alpaca
INTRADAY_DERIVED        = {'5m': '5min', '10m': '10min', '1h': '1h'}   # folder -> pandas resample rule
REGULAR_HOURS_ONLY      = True
ALPACA_FEEDS            = ['sip', 'iex']    # tried in this order


# ----
# CLINICAL TRIALS
# ----

TRIAL_SPONSORS = []                         # extra sponsor searches, e.g. ['Eli Lilly and Company']
TRIAL_NCT_IDS  = []                         # e.g. ['NCT01234567']
TRIAL_SECTORS  = ['Healthcare']             # companies whose trials are searched (company_info.sector)
TRIAL_MAX_TRIALS = 60_000                   # at most this many trials stored (new searches share the budget)
TRIAL_START_FROM = '2015-01-01'             # company searches: only trials starting on / after this date
# primary ticker -> sponsor search texts (query.spons), replacing the company name, where trials are
# registered under another name. Also used to match sponsors to the company.
TRIAL_SPONSOR_ALIASES = {
    'MRK':   ['Merck Sharp & Dohme'],                 # Merck & Co. files as MSD; 'Merck' alone finds Merck KGaA too
    'MRK.DE': ['Merck KGaA', 'EMD Serono'],
    'MRNA':  ['ModernaTX'],
    'JNJ':   ['Johnson & Johnson', 'Janssen'],
    'AZN.L': ['AstraZeneca', 'MedImmune', 'Alexion'],
    'PFE':   ['Pfizer', 'Seagen', 'Wyeth'],
    'SAN.PA': ['Sanofi', 'Genzyme'],
    'BMY':   ['Bristol-Myers Squibb', 'Celgene'],
    'ABBV':  ['AbbVie', 'Allergan'],
    'GSK.L': ['GlaxoSmithKline', 'GSK', 'ViiV Healthcare'],
    '4578.T': ['Otsuka'],                             # yfinance name 'Ootsuka Holdings'
    'PHIA.AS': ['Philips'],
    '6160.HK': ['BeiGene', 'BeOne'],
    'CSL.AX': ['CSL Behring', 'Seqirus'],
    'COO':   ['CooperVision', 'CooperSurgical'],      # 'Cooper' alone is too broad
    'ORNBV.HE': ['Orion Corporation', 'Orion Pharma'],
    'WAT':   ['Waters Corporation'],
    'SUNPHARMA.NS': ['Sun Pharmaceutical', 'Sun Pharma'],
    'SHL.DE': ['Siemens Healthineers', 'Siemens Healthcare'],
    'GEHC':  ['GE Healthcare'],
    'FRE.DE': ['Fresenius Kabi'],
    '1177.HK': ['Sino Biopharmaceutical', 'Chia Tai Tianqing'],
    '207940.KS': ['Samsung Biologics', 'Samsung Bioepis'],
    '4502.T': ['Takeda'],                             # sponsor name is just 'Takeda' / 'Takeda Development ...'
}
# primary ticker -> extra sponsor names for matching only (not searched)
TRIAL_SPONSOR_MATCH = {
    'MRK.DE': ['Merck KGaA, Darmstadt, Germany'],
    'GILD':  ['Kite, A Gilead Company'],
    'UCB.BR': ['UCB Biopharma', 'UCB Pharma'],
    '1093.HK': ['CSPC'],                             # 'CSPC Ouyi / ZhongQi / Zhongnuo Pharmaceutical ...'
}


# ----
# NETWORK
# ----

# service -> (max calls, per seconds)
RATE_LIMITS = {
    'sec':     (10, 1),
    'alpaca':  (200, 60),
    'finnhub': (60, 60),
    'fred':    (120, 60),
    'ctgov':   (50, 60),
    'geonames': (10, 60),
    'nominatim': (1, 1.1),                  # OpenStreetMap usage policy: max 1 request / second
    'yfinance': (240, 60),
}
# a database file held by another process (Streamlit app, notebook): wait and retry, logged
DB_LOCK_RETRY_SECONDS    = 30
DB_LOCK_MAX_WAIT_SECONDS = 600
HTTP_RETRIES = 4
HTTP_BACKOFF = 2.0                          # seconds, doubled per retry
HTTP_TIMEOUT = 30


# ----
# RUN ORDER
# ----

# step name -> module in pipeline/ that has run(ctx)
STEPS = {
    'ids':          'ids',
    'companies':    'companies',
    'prices':       'prices',
    'fred':         'fred',
    'fundamentals': 'fundamentals',
    'marketcap':    'marketcap',
    'calendar':     'calendar',
    'holders':      'holders',
    'executives':   'executives',
    'subsidiaries': 'subsidiaries',
    'intraday':     'intraday',
    'trials':       'trials',
}
