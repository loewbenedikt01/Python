"""
Step 'companies': fill company_info (one row per corporate_id) and the
currency / isin columns of instruments.

Sources, in order of preference per field:
  SEC EDGAR submissions (SEC filers): legal name, SIC, HQ address, fiscal year end, form type
  yfinance .info (every ticker):       ISIN, sector, industry, description, country, currency,
                                        employees, website, exchange
  Finnhub profile2 (US tickers only;   country, currency, industry, website, exchange
  the free plan refuses non-US):        when yfinance has no data
Companies are refreshed when older than config.COMPANY_REFRESH_DAYS; with
--tickers the selected companies are always refreshed.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

import config
from pipeline import common
from pipeline.ids import load_registry


log = common.get_logger('pipeline.companies')

SUBMISSIONS_URL  = 'https://data.sec.gov/submissions/CIK{cik:010d}.json'
SUBMISSIONS_DIR  = config.CACHE_DIR / 'sec_submissions'
UNKNOWN_COUNTRIES_CSV = config.REVIEW_DIR / 'unknown_countries.csv'
ANNUAL_FORMS     = ('10-K', '20-F', '40-F')

YF_FIELDS = ['longName', 'sector', 'industry', 'country', 'currency', 'financialCurrency',
             'fullTimeEmployees', 'website', 'exchange', 'fullExchangeName', 'address1', 'city',
             'state', 'zip', 'lastFiscalYearEnd', 'longBusinessSummary', 'quoteType',
             'impliedSharesOutstanding', 'sharesOutstanding']


# ----
# ISIN / CUSIP
# ----

def valid_isin(isin) -> bool:
    """
    ISO 6166 format + Luhn check digit (yfinance sometimes returns '-' or junk).
    """
    if not isinstance(isin, str) or len(isin) != 12 or not isin[:2].isalpha() or not isin.isalnum():
        return False
    digits = ''.join(str(int(c, 36)) for c in isin.upper())
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d) * (2 if i % 2 else 1)
        total += n - 9 if n > 9 else n
    return total % 10 == 0


def isin_plausible(ticker: str, isin, company_country: str | None) -> bool:
    """
    yfinance's ISIN lookup returns the first search hit, sometimes another
    listing (GOOGL -> the Canadian CDR 'CA...'). Accept an ISIN only if its
    country prefix is the listing's exchange country or the company's country.
    """
    if not valid_isin(isin):
        return False
    allowed = {config.COUNTRIES.get(c, (None,))[0] for c in (exchange_country(ticker), company_country)}
    return isin[:2] in allowed


def cusip_from_isin(isin) -> str | None:
    """
    US ISIN = 'US' + 9-character CUSIP + check digit.
    """
    return isin[2:11] if valid_isin(isin) and isin.startswith('US') else None


# ----
# DOWNLOADS
# ----

def fetch_yfinance(ticker: str, with_isin: bool) -> dict:
    common.limiter('yfinance').wait()
    tk = yf.Ticker(ticker)
    info = tk.info or {}
    out = {k: info.get(k) for k in YF_FIELDS}
    if with_isin:
        common.limiter('yfinance').wait()
        try:
            isin = tk.isin
        except Exception:
            isin = None
        out['isin'] = isin if valid_isin(isin) else None
    return out


def fetch_finnhub(ticker: str) -> dict:
    r = common.http_get('https://finnhub.io/api/v1/stock/profile2', 'finnhub',
                        params={'symbol': ticker, 'token': common.get_key('FINNHUB')})
    return r.json() or {}


def fetch_submissions(cik: int) -> dict:
    """
    SEC submissions JSON; also cached to data/cache/sec_submissions/ for the filing steps.
    """
    data = common.http_get(SUBMISSIONS_URL.format(cik=cik), 'sec').json()
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    (SUBMISSIONS_DIR / f'CIK{cik:010d}.json').write_text(json.dumps(data), encoding='utf-8')
    return data


def sec_fields(sub: dict) -> dict:
    addr = (sub.get('addresses') or {}).get('business') or {}
    street = ', '.join(filter(None, [addr.get('street1'), addr.get('street2')])) or None
    foreign = bool(addr.get('isForeignLocation'))
    forms = (sub.get('filings') or {}).get('recent', {}).get('form', [])
    annual = next((f for f in forms if f in ANNUAL_FORMS), None)        # newest annual form
    if annual is None and '10-Q' in forms:
        annual = '10-K'                    # e.g. a new holding company before its first 10-K (XOM 2026)
    return {
        'legal_name':      sub.get('name'),
        'sic_code':        sub.get('sic') or None,
        'sic_description': sub.get('sicDescription') or None,
        'fiscal_year_end': sub.get('fiscalYearEnd') or None,
        'website':         sub.get('website') or None,
        'hq_street':       street,
        'hq_city':         addr.get('city'),
        'hq_state':        None if foreign else addr.get('stateOrCountry'),
        'hq_postal_code':  addr.get('zipCode'),
        'hq_country':      addr.get('stateOrCountryDescription') if foreign else ('United States' if addr else None),
        'sec_form_type':   annual,
        'sec_filer':       annual is not None,
    }


def _parallel(fn, items, workers):
    """
    {item: result or Exception}, keeping going when single items fail.
    """
    out = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fn, item): item for item in items}
        for i, f in enumerate(cf.as_completed(futures), 1):
            item = futures[f]
            try:
                out[item] = f.result()
            except Exception as e:
                out[item] = e
            if i % 200 == 0:
                log.info(f'  ...{i}/{len(items)}')
    return out


# ----
# COMPANY ROWS
# ----

def exchange_country(ticker: str) -> str:
    dot = ticker.rfind('.')
    return config.EXCHANGE_SUFFIXES.get(ticker[dot:], 'Unknown') if dot > 0 else 'United States'


def choose_primary(corporate_id: int, tickers: list[str], country: str | None, is_sec: bool) -> str:
    """
    Override > first ticker for SEC filers (Class A / SEC order) > home-country
    listing > first ticker in the registry.
    """
    if corporate_id in config.PRIMARY_TICKER_OVERRIDES:
        return config.PRIMARY_TICKER_OVERRIDES[corporate_id]
    if not is_sec and country:
        home = [t for t in tickers if exchange_country(t) == country]
        if home:
            return home[0]
    return tickers[0]


def major_currency(currency: str | None) -> str | None:
    if currency in config.MINOR_UNITS:
        return config.MINOR_UNITS[currency][0]
    return currency.upper() if isinstance(currency, str) else None


def _fye_from_epoch(epoch) -> str | None:
    try:
        return datetime.fromtimestamp(int(epoch), tz=timezone.utc).strftime('%m%d')
    except (TypeError, ValueError, OSError):
        return None


def build_company(cid: int, name: str, cik, tickers: list[str], yf_info: dict[str, dict],
                  sub: dict | None, finnhub: dict | None) -> dict:
    infos = [yf_info[t] for t in tickers if isinstance(yf_info.get(t), dict) and yf_info[t].get('quoteType')]
    base = infos[0] if infos else {}
    fh = finnhub or {}
    sec = sec_fields(sub) if sub else {}

    fh_country = next((n for n, (iso, _) in config.COUNTRIES.items() if iso == fh.get('country')), None)
    country = base.get('country') or fh_country or sec.get('hq_country')
    primary = choose_primary(cid, tickers, country, bool(sec.get('sec_filer')))
    p = yf_info.get(primary) if isinstance(yf_info.get(primary), dict) and yf_info[primary].get('quoteType') else base

    isin = p.get('isin') or next((i.get('isin') for i in infos if i.get('isin')), None)
    employees = p.get('fullTimeEmployees')
    yf_shares = p.get('impliedSharesOutstanding') or p.get('sharesOutstanding')
    return {
        'corporate_id':         cid,
        'primary_ticker':       primary,
        'tickers':              tickers,
        'name':                 name,
        'legal_name':           sec.get('legal_name') or p.get('longName') or fh.get('name'),
        'cik':                  int(cik) if pd.notna(cik) else None,
        'isin':                 isin,
        'cusip':                cusip_from_isin(isin),
        'sec_filer':            bool(sec.get('sec_filer')),
        'sec_form_type':        sec.get('sec_form_type'),
        'exchange':             p.get('fullExchangeName') or fh.get('exchange'),
        'country':              country,
        'continent':            config.COUNTRIES.get(country, (None, None))[1],
        'trading_currency':     major_currency(p.get('currency') or fh.get('currency')),
        'reporting_currency':   major_currency(p.get('financialCurrency')),
        'sector':               p.get('sector') or fh.get('finnhubIndustry'),
        'industry':             p.get('industry') or fh.get('finnhubIndustry'),
        'sic_code':             sec.get('sic_code'),
        'sic_description':      sec.get('sic_description'),
        'business_description': p.get('longBusinessSummary'),
        'hq_street':            sec.get('hq_street') or p.get('address1'),
        'hq_city':              sec.get('hq_city') or p.get('city'),
        'hq_state':             sec.get('hq_state') if sec else p.get('state'),
        'hq_postal_code':       sec.get('hq_postal_code') or p.get('zip'),
        'hq_country':           sec.get('hq_country') or country,
        'website':              p.get('website') or sec.get('website') or fh.get('weburl'),
        'fiscal_year_end':      sec.get('fiscal_year_end') or _fye_from_epoch(p.get('lastFiscalYearEnd')),
        'employees':            int(employees) if employees else None,
        'yf_shares_outstanding': float(yf_shares) if yf_shares else None,
        'yf_shares_date':       datetime.now().date() if yf_shares else None,
        'updated_at':           datetime.now(),
    }


def sync_employees(con, corporate_ids: list[int]) -> int:
    """
    Write company_info.employees (yfinance, current) into the latest fiscal
    year's row of fundamentals_yearly if that row has no employee count, with
    employees_source = 'yfinance <date>'. Run on every company_info refresh, so
    a history builds up over time. Returns the number of rows written.
    """
    if not corporate_ids:
        return 0
    return con.execute("""
        UPDATE fundamentals_yearly AS f
        SET employees = c.employees, employees_source = 'yfinance ' || strftime(c.updated_at, '%Y-%m-%d')
        FROM company_info c,
             (SELECT corporate_id, max(fiscal_year) AS fy FROM fundamentals_yearly GROUP BY 1) AS latest
        WHERE f.corporate_id = c.corporate_id AND latest.corporate_id = f.corporate_id
          AND f.fiscal_year = latest.fy AND f.employees IS NULL AND c.employees IS NOT NULL
          AND f.corporate_id IN (SELECT unnest(?))""", [list(map(int, corporate_ids))]).fetchone()[0]


NAME_GROUPS_CSV = config.REVIEW_DIR / 'name_groups.csv'


def review_name_groups() -> pd.DataFrame:
    """
    Non-US tickers that share a company name (and therefore a corporate_id):
    data/review/name_groups.csv lists every group where yfinance's longName,
    country or currency differ between the tickers. Dual listings on two
    exchanges (BYD .HK / .SZ) legitimately differ in currency.
    """
    files = common.load_ticker_files()
    eq = files[(files['asset_class'] == config.EQUITIES_ASSET_CLASS) & (files['group_name'] != config.US_GROUP)]
    eq = eq.drop_duplicates('ticker')
    grouped = eq[eq.duplicated('name', keep=False)]
    info = _parallel(lambda t: fetch_yfinance(t, with_isin=False), grouped['ticker'].tolist(), config.YF_WORKERS)
    rows = []
    for t, name in zip(grouped['ticker'], grouped['name']):
        v = info.get(t) if isinstance(info.get(t), dict) else {}
        rows.append({'name_in_ticker_file': name, 'ticker': t, 'yf_long_name': v.get('longName'),
                     'yf_country': v.get('country'), 'yf_currency': v.get('currency')})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    diff = []
    for name, g in df.groupby('name_in_ticker_file'):
        fields = [f for f in ('yf_long_name', 'yf_country', 'yf_currency') if g[f].nunique(dropna=False) > 1]
        if fields:
            diff.append(g.assign(differs=', '.join(f[3:] for f in fields)))
    out = pd.concat(diff) if diff else pd.DataFrame(columns=list(df.columns) + ['differs'])
    out.to_csv(NAME_GROUPS_CSV, index=False)
    return out


TICKER_NAME_CSV = config.REVIEW_DIR / 'ticker_name_check.csv'

_LEGAL = set("""inc incorporated corp corporation co company cos ltd limited plc llc lp ag se sa nv spa ab asa oyj
    kgaa gmbh holding holdings group the class and of aktiengesellschaft publ bhd berhad tbk pcl public adr ads
    reit trust sab cv de la le les del y et ord shs common stock limitada kk kabushiki kaisha
    new companies tr""".split())                  # 13F: 'NEWS CORP NEW', 'WILLIAMS COS', 'ESSEX PPTY TR'


def _norm_name(name) -> list[str]:
    import unicodedata
    if not isinstance(name, str):
        return []
    s = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower().replace('&', ' and ')
    s = ''.join(ch if ch.isalnum() else ' ' for ch in s)
    words = [w for w in s.split() if w not in _LEGAL]
    # single letters join the next word ("O'Reilly" -> 'oreilly', 'S&P' -> 'sp', 'V.F.' -> 'vf')
    kept, prefix = [], ''
    for w in words:
        if len(w) == 1:
            prefix += w
        else:
            kept.append(prefix + w)
            prefix = ''
    if prefix and (len(prefix) > 1 or not kept):
        kept.append(prefix)
    return kept


def name_similarity(a, b) -> float | None:
    """
    0..1: the higher of character similarity and word overlap (shared words /
    words of the shorter name) after removing legal forms and share classes.
    """
    import difflib
    ta, tb = _norm_name(a), _norm_name(b)
    if not ta or not tb:
        return None
    seq = difflib.SequenceMatcher(None, ' '.join(ta), ' '.join(tb)).ratio()
    overlap = len(set(ta) & set(tb)) / min(len(set(ta)), len(set(tb)))
    return round(max(seq, overlap), 3)


def ticker_name_check(asset_classes: set[str] | None = None) -> pd.DataFrame:
    """
    Compare each ticker's name in the _tickers files with yfinance's longName /
    shortName -> data/review/ticker_name_check.csv, worst matches first.
    """
    files = common.load_ticker_files()
    files = files[~files['asset_class'].isin(config.FRED_ASSET_CLASSES)]
    if asset_classes:
        files = files[files['asset_class'].isin(asset_classes)]
    files = files.drop_duplicates('ticker')
    info = _parallel(_yf_names, files['ticker'].tolist(), config.YF_WORKERS)
    rows = []
    for t, name, ac, grp in zip(files['ticker'], files['name'], files['asset_class'], files['group_name']):
        v = info.get(t) if isinstance(info.get(t), dict) else {}
        long_, short = v.get('longName'), v.get('shortName')
        scores = [x for x in (name_similarity(name, long_), name_similarity(name, short)) if x is not None]
        rows.append({'ticker': t, 'asset_class': ac, 'group_name': grp, 'my_name': name, 'yf_long_name': long_,
                     'yf_short_name': short, 'yf_quote_type': v.get('quoteType'),
                     'similarity': max(scores) if scores else None})
    df = pd.DataFrame(rows).sort_values('similarity', na_position='first')
    df.to_csv(TICKER_NAME_CSV, index=False)
    return df


def _search_ticker(name: str, ticker: str) -> tuple[str | None, str | None]:
    """
    yfinance search for the company named in the ticker file: the first equity
    on the same exchange (same suffix) as `ticker`, else the first equity.
    """
    import time
    suffix = ticker[ticker.rfind('.'):] if '.' in ticker else ''
    for attempt in range(3):
        common.limiter('yfinance').wait()
        try:
            quotes = yf.Search(name, max_results=10).quotes or []
            break
        except Exception:
            quotes = []
            time.sleep(2 * (attempt + 1))
    equities = [q for q in quotes if q.get('quoteType') == 'EQUITY' and q.get('symbol')]
    same = [q for q in equities if (q['symbol'][q['symbol'].rfind('.'):] if '.' in q['symbol'] else '') == suffix]
    best = (same or equities or [None])[0]
    if best is None:
        return None, None
    return best['symbol'], best.get('longname') or best.get('shortname')


def refresh_ticker_name_check(threshold: float = 0.5) -> pd.DataFrame:
    """
    Update data/review/ticker_name_check.csv: fetch names again for tickers
    without a yfinance name (rate limits), and for equities below `threshold`
    add two suggestions: (a) the correct name for the ticker as it is,
    (b) the ticker yfinance finds for the company named in the file.
    """
    df = pd.read_csv(TICKER_NAME_CSV)
    missing = df.loc[df['yf_long_name'].isna() & df['yf_short_name'].isna(), 'ticker'].tolist()
    if missing:
        info = _parallel(_yf_names, missing, 3)
        for t, v in info.items():
            if isinstance(v, dict):
                i = df.index[df['ticker'] == t][0]
                df.loc[i, ['yf_long_name', 'yf_short_name', 'yf_quote_type']] = [
                    v.get('longName'), v.get('shortName'), v.get('quoteType')]
    df['similarity'] = [max([x for x in (name_similarity(m, l), name_similarity(m, sh)) if x is not None], default=None)
                        for m, l, sh in zip(df['my_name'], df['yf_long_name'], df['yf_short_name'])]
    low = df['asset_class'].eq(config.EQUITIES_ASSET_CLASS) & (df['similarity'].isna() | (df['similarity'] < threshold))
    df['suggest_a_name_for_ticker'] = df['yf_long_name'].fillna(df['yf_short_name']).where(low)
    found = _parallel(lambda i: _search_ticker(df.at[i, 'my_name'], df.at[i, 'ticker']),
                      df.index[low].tolist(), 3)
    for i, v in found.items():
        if isinstance(v, tuple):
            df.loc[i, ['suggest_b_ticker_for_name', 'suggest_b_found_name']] = list(v)
    df = df.sort_values('similarity', na_position='first')
    df.to_csv(TICKER_NAME_CSV, index=False)
    return df


def _yf_names(ticker: str) -> dict:
    # Yahoo rate-limits bursts silently (empty info): up to 3 attempts with a pause
    import time
    for attempt in range(3):
        common.limiter('yfinance').wait()
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}
        if info.get('longName') or info.get('shortName'):
            break
        time.sleep(2 * (attempt + 1))
    return {k: info.get(k) for k in ('longName', 'shortName', 'quoteType')}


# ----
# STEP
# ----

def run(ctx: common.RunContext) -> common.StepResult:
    result = common.StepResult()
    registry = load_registry()
    registry = registry[(registry['status'] == 'active') & (registry['tickers'] != '')].copy()
    registry['ticker_list'] = registry['tickers'].str.split(';')

    con = common.connect('companies')
    pcon = common.connect('prices')
    try:
        # which companies are due
        stored = dict(con.execute('SELECT corporate_id, updated_at FROM company_info').fetchall())
        cutoff = datetime.now() - timedelta(days=config.COMPANY_REFRESH_DAYS)
        if ctx.tickers:
            wanted = set(ctx.selected([t for ts in registry['ticker_list'] for t in ts]))
            due = registry[registry['ticker_list'].map(lambda ts: bool(wanted & set(ts)))]
        else:
            due = registry[registry['corporate_id'].map(lambda c: c not in stored or stored[c] < cutoff)]
        result.skipped = len(registry) - len(due) if not ctx.tickers else 0
        log.info(f'{len(due)} companies to refresh ({result.skipped} still fresh)')

        # downloads; tickers without data (no_data) or removed are skipped
        skip = {t for (t,) in pcon.execute(
            "SELECT ticker FROM instruments WHERE status IN ('no_data', 'removed')").fetchall()}
        eq_tickers = [t for ts in due['ticker_list'] for t in ts if t not in skip]
        yf_info = _parallel(lambda t: fetch_yfinance(t, with_isin=True), eq_tickers, config.YF_WORKERS)
        for t, v in yf_info.items():
            if isinstance(v, dict) and v.get('isin') and not isin_plausible(t, v['isin'], v.get('country')):
                log.info(f'{t}: ISIN {v["isin"]} from yfinance does not fit the listing, dropped')
                v['isin'] = None
        ciks = [int(c) for c in due['cik'].dropna().unique()]
        subs = _parallel(fetch_submissions, ciks, 4)

        us = set(common.load_ticker_files().query('group_name == @config.US_GROUP')['ticker'])
        need_fh = [ts[0] for ts in due['ticker_list']
                   if ts[0] in us and not (isinstance(yf_info.get(ts[0]), dict) and yf_info[ts[0]].get('country'))]
        fh = _parallel(fetch_finnhub, need_fh, 2) if need_fh else {}

        # company rows
        rows, unknown = [], []
        for cid, name, cik, tickers in zip(due['corporate_id'], due['company_name'], due['cik'], due['ticker_list']):
            for t in tickers:
                if isinstance(yf_info.get(t), Exception):
                    result.fail(t, yf_info[t], log)
            sub = subs.get(int(cik)) if pd.notna(cik) else None
            if isinstance(sub, Exception):
                result.fail(f'CIK {int(cik)}', sub, log)
                sub = None
            f = fh.get(tickers[0])
            row = build_company(int(cid), name, cik, tickers, yf_info, sub, f if isinstance(f, dict) else None)
            if row['country'] and not row['continent']:
                unknown.append({'corporate_id': cid, 'name': name, 'country': row['country']})
            rows.append(row)

        companies = pd.DataFrame(rows)
        result.rows += common.upsert(con, 'company_info', companies)
        result.updated = len(companies)
        n = sync_employees(con, companies['corporate_id'].tolist()) if not companies.empty else 0
        if n:
            log.info(f'employees written into {n} fundamentals_yearly rows')
        if not ctx.tickers:
            groups = review_name_groups()
            log.info(f'{groups["name_in_ticker_file"].nunique() if not groups.empty else 0} name groups with '
                     f'differing yfinance data -> {NAME_GROUPS_CSV.name}')

        # instruments: currency + isin for the refreshed equities
        inst = pd.DataFrame([
            {'ticker': t, 'currency': v.get('currency'), 'isin': v.get('isin')}
            for t, v in yf_info.items() if isinstance(v, dict) and v.get('quoteType')])
        result.rows += common.upsert(pcon, 'instruments', inst)

        # instruments: currency for non-equities that don't have one yet
        missing = [t for (t,) in pcon.execute("""
            SELECT ticker FROM instruments
            WHERE asset_class <> ? AND currency IS NULL AND NOT ends_with(ticker, '_FRED')
              AND coalesce(status, 'active') NOT IN ('no_data', 'removed')""",
            [config.EQUITIES_ASSET_CLASS]).fetchall()]
        missing = ctx.selected(missing)
        if missing:
            log.info(f'{len(missing)} non-equity instruments without currency')
            other = _parallel(lambda t: fetch_yfinance(t, with_isin=False), missing, config.YF_WORKERS)
            ok = pd.DataFrame([{'ticker': t, 'currency': v.get('currency')}
                               for t, v in other.items() if isinstance(v, dict) and v.get('currency')])
            for t, v in other.items():
                if not (isinstance(v, dict) and v.get('currency')):
                    result.fail(t, v if isinstance(v, Exception) else 'no currency from yfinance', log)
            result.rows += common.upsert(pcon, 'instruments', ok)

        if unknown:
            pd.DataFrame(unknown).to_csv(UNKNOWN_COUNTRIES_CSV, index=False)
            log.warning(f'{len(unknown)} companies with a country missing in config.COUNTRIES '
                        f'-> {UNKNOWN_COUNTRIES_CSV.name}')
    finally:
        con.close()
        pcon.close()
    return result
