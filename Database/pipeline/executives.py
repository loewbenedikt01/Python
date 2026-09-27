"""
Step 'executives': officers and directors per fiscal year from the SEC insider
transactions data sets (Forms 3 / 4 / 5), plus the raw insider transactions.

- Quarterly data sets for the last EXECUTIVES_YEARS years are downloaded once
  and kept as Parquet in data/cache/form345/ (processed_datasets).
- fundamentals_yearly.executives: every person who filed a Form 3 / 4 / 5 for
  the company with a period of report inside the fiscal year, as officer or
  director. Companies / funds and pure 10 % owners are left out. Per person:
  readable name ('Timothy D. Cook') and SEC name ('COOK TIMOTHY D'), CIK, the
  latest title in the year, a normalized role, officer / director flags.
  Predecessor CIKs (GOOGL, XOM) are included.
- raw.duckdb insider_transactions_raw: the non-derivative transactions (buys,
  sells, awards, ...) of the covered companies, one row per transaction and
  reporting owner.
"""

from __future__ import annotations

import re
import zipfile
from datetime import date, datetime

import duckdb
import pandas as pd

import config
from pipeline import common


log = common.get_logger('pipeline.executives')

LIST_URL = 'https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets'
CACHE    = config.CACHE_DIR / 'form345'

# role from the title, first match wins ('Chairman and CEO' -> CEO)
ROLES = [
    ('CEO',             r'CHIEF EXECUTIVE|\bCEO\b|PRINCIPAL EXECUTIVE'),
    ('CFO',             r'CHIEF FINANCIAL|\bCFO\b|PRINCIPAL FINANCIAL'),
    ('COO',             r'CHIEF OPERATING|\bCOO\b'),
    ('President',       r'(?<!VICE )(?<!VICE-)\bPRESIDENT\b'),
    ('Chair',           r'\bCHAIR'),
    ('General Counsel', r'GENERAL COUNSEL|CHIEF LEGAL|\bGC\b'),
]
ROLE_ORDER = ['CEO', 'CFO', 'COO', 'President', 'Chair', 'General Counsel', 'Director', 'Other']
ENTITY = re.compile(r'\b(LLC|L\.?L\.?C|L\.?P|LP|INC|CORP|CORPORATION|FUND|FUNDS|TRUST|HOLDINGS?|CAPITAL|PARTNERS|'
                    r'MANAGEMENT|GROUP|LTD|LIMITED|PLC|FOUNDATION|BANK|ADVISORS?|INVESTMENTS?|VENTURES|N\.?A)\b\.?$|'
                    r'\b(LLC|L\.P\.|FUND|TRUST|CAPITAL|PARTNERS|HOLDINGS|MANAGEMENT)\b')
SUFFIXES = {'JR', 'SR', 'II', 'III', 'IV', 'V', 'MD', 'PHD', 'ESQ'}
PARTICLES = {'VAN', 'VON', 'DE', 'DER', 'DEN', 'TER', 'TEN', 'DI', 'DA', 'DEL', 'DELLA', 'LA', 'LE', 'ST', 'DU', 'DOS', 'DAS'}


# ----
# NAMES / ROLES
# ----

def _cap(word: str) -> str:
    w = word.capitalize()
    for pre in ("Mc", "O'", "D'"):
        if w.startswith(pre) and len(w) > len(pre):
            w = pre + w[len(pre)].upper() + w[len(pre) + 1:]
    return '-'.join(p[:1].upper() + p[1:] for p in w.split('-'))


def readable_name(sec_name: str) -> str:
    """
    SEC 'LAST FIRST MIDDLE [SUFFIX]' -> 'First M. Last [Suffix]' ('COOK TIMOTHY D' -> 'Timothy D. Cook').
    """
    if not isinstance(sec_name, str) or not sec_name.strip():
        return sec_name
    tokens = sec_name.replace(',', ' ').replace('.', ' ').split()
    suffix = [t for t in tokens[1:] if t.upper() in SUFFIXES]
    tokens = [t for t in tokens if t not in suffix]
    if len(tokens) == 1:
        return _cap(tokens[0])
    n_last = 1                                   # surname = leading particles + one word ('VAN DER BERG')
    while n_last < len(tokens) - 1 and tokens[n_last - 1].upper() in PARTICLES:
        n_last += 1
    last, given = tokens[:n_last], tokens[n_last:]
    given = [t.upper() + '.' if len(t) == 1 else _cap(t) for t in given]
    last = [t.lower() if t.upper() in PARTICLES and i < len(last) - 1 else _cap(t) for i, t in enumerate(last)]
    out = ' '.join(given + last)
    if suffix:
        out += ' ' + ' '.join(s.capitalize() if len(s) > 3 else s.upper() if s.upper() in {'II', 'III', 'IV', 'V'}
                              else s.capitalize() + '.' for s in suffix)
    return out


# words that may follow a company-wide role ('Chairman & CEO', 'CEO and Director'); anything else
# after it means a business-line head ('CEO CCB', 'Co-CEO CIB', 'President, Global Banking')
_TAIL_OK = {'AND', 'DIRECTOR', 'CHAIRMAN', 'CHAIR', 'CHAIRWOMAN', 'PRESIDENT', 'OF', 'THE', 'BOARD', 'COMPANY',
            'SECRETARY', 'TREASURER', 'EXECUTIVE', 'OFFICER', 'CHIEF', 'CEO', 'CFO', 'COO', 'INTERIM', 'ACTING'}
_COMPANY_WIDE = {'CEO', 'CFO', 'COO', 'President'}


def role_of(title, is_officer: bool, is_director: bool) -> str:
    t = (title or '').upper()
    for role, pattern in ROLES:
        m = None
        for m in re.finditer(pattern, t):
            pass
        if m is None:
            continue
        if role in _COMPANY_WIDE:
            tail = re.sub(r'[^A-Z ]', ' ', t[m.end():]).split()
            if any(w not in _TAIL_OK for w in tail) or re.search(r'\bCO-?\s?' + pattern.split('|')[0], t):
                continue                                  # business-line head / co-head of a unit
        return role
    if is_officer:
        return 'Other'
    return 'Director' if is_director else 'Other'


def is_entity(name) -> bool:
    return isinstance(name, str) and bool(ENTITY.search(name.upper()))


# ----
# DATA SETS
# ----

def list_datasets() -> pd.DataFrame:
    html = common.http_get(LIST_URL, 'sec').text
    rows = []
    for href in set(re.findall(r'href="([^"]+form345[^"]*\.zip)"', html, re.I)):
        name = href.rsplit('/', 1)[1]
        m = re.match(r'(\d{4})q(\d)_form345\.zip', name, re.I)
        if not m:
            continue
        y, q = int(m.group(1)), int(m.group(2))
        rows.append({'name': name, 'url': href if href.startswith('http') else 'https://www.sec.gov' + href,
                     'start': date(y, 3 * q - 2, 1)})
    return pd.DataFrame(rows).sort_values('start').reset_index(drop=True)


def ensure_dataset(con, row) -> None:
    folder = CACHE / row['name'].replace('.zip', '')
    if (folder / 'trans.parquet').exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    log.info(f"downloading {row['name']}")
    zpath = CACHE / row['name']
    r = common.http_get(row['url'], 'sec', stream=True)
    with open(zpath, 'wb') as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    with zipfile.ZipFile(zpath) as z:
        for member in ('SUBMISSION.tsv', 'REPORTINGOWNER.tsv', 'NONDERIV_TRANS.tsv'):
            name = next(n for n in z.namelist() if n.rsplit('/', 1)[-1].upper() == member.upper())
            with z.open(name) as src, open(folder / member, 'wb') as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
    q = lambda p: str(p).replace('\\', '/')
    tsv = lambda n: f"read_csv('{q(folder / n)}', delim='\\t', header=true, all_varchar=true, quote='', ignore_errors=true)"
    d = lambda col: f"try_strptime({col}, '%d-%b-%Y')::DATE"
    duck = duckdb.connect()
    duck.execute(f"""COPY (SELECT ACCESSION_NUMBER AS accession, {d('FILING_DATE')} AS filing_date,
                            {d('PERIOD_OF_REPORT')} AS period, DOCUMENT_TYPE AS form_type,
                            ltrim(ISSUERCIK, '0') AS issuer_cik, ISSUERNAME AS issuer_name,
                            ISSUERTRADINGSYMBOL AS issuer_symbol
                     FROM {tsv('SUBMISSION.tsv')}) TO '{q(folder / 'submission.parquet')}' (FORMAT parquet)""")
    duck.execute(f"""COPY (SELECT ACCESSION_NUMBER AS accession, ltrim(RPTOWNERCIK, '0') AS owner_cik,
                            RPTOWNERNAME AS owner_name, RPTOWNER_RELATIONSHIP AS relationship,
                            RPTOWNER_TITLE AS title, RPTOWNER_TXT AS other_text
                     FROM {tsv('REPORTINGOWNER.tsv')}) TO '{q(folder / 'owner.parquet')}' (FORMAT parquet)""")
    duck.execute(f"""COPY (SELECT ACCESSION_NUMBER AS accession, NONDERIV_TRANS_SK AS trans_sk,
                            SECURITY_TITLE AS security_title, {d('TRANS_DATE')} AS trans_date,
                            TRANS_CODE AS trans_code, TRANS_ACQUIRED_DISP_CD AS acquired_disposed,
                            TRY_CAST(TRANS_SHARES AS DOUBLE) AS shares,
                            TRY_CAST(TRANS_PRICEPERSHARE AS DOUBLE) AS price,
                            TRY_CAST(SHRS_OWND_FOLWNG_TRANS AS DOUBLE) AS shares_owned_after,
                            DIRECT_INDIRECT_OWNERSHIP AS direct_indirect
                     FROM {tsv('NONDERIV_TRANS.tsv')}) TO '{q(folder / 'trans.parquet')}' (FORMAT parquet)""")
    duck.close()
    for member in ('SUBMISSION.tsv', 'REPORTINGOWNER.tsv', 'NONDERIV_TRANS.tsv'):
        (folder / member).unlink()
    zpath.unlink()
    common.upsert(con, 'processed_datasets', pd.DataFrame([{
        'dataset': f"insider:{row['name']}", 'processed_at': datetime.now(), 'rows_used': None}]))


# ----
# STEP
# ----

def run(ctx: common.RunContext) -> common.StepResult:
    result = common.StepResult()
    con = common.connect('companies')
    try:
        datasets = list_datasets()
        first = date(date.today().year - config.EXECUTIVES_YEARS - 1, 1, 1)   # + the year before, for fiscal years
        needed = datasets[datasets['start'] >= first]
        for row in needed.to_dict('records'):
            try:
                ensure_dataset(con, row)
            except Exception as e:
                result.fail(row['name'], common.format_error(e), log)
        folders = [CACHE / n.replace('.zip', '') for n in needed['name']
                   if (CACHE / n.replace('.zip', '') / 'trans.parquet').exists()]
        files = lambda n: '[' + ', '.join("'" + str(f / n).replace('\\', '/') + "'" for f in folders) + ']'

        # companies and their CIKs (incl. predecessors)
        us = set(common.load_ticker_files().query('group_name == @config.US_GROUP')['ticker'])
        comp = con.execute("""SELECT corporate_id, primary_ticker, tickers, cik FROM company_info
                              WHERE sec_filer AND cik IS NOT NULL""").df()
        comp = comp[comp['tickers'].map(lambda ts: bool(us & set(ts)))]
        if ctx.tickers:
            wanted = {t.upper() for t in ctx.tickers}
            comp = comp[comp['tickers'].map(lambda ts: bool(wanted & {t.upper() for t in ts}))]
        cik_map = pd.DataFrame([(str(k), int(c.corporate_id)) for c in comp.itertuples(index=False)
                                for k in [int(c.cik)] + config.SEC_PREDECESSOR_CIKS.get(int(c.cik), [])],
                               columns=['issuer_cik', 'corporate_id'])

        duck = duckdb.connect()
        duck.register('_ciks', cik_map)
        filings = duck.execute(f"""
            SELECT s.accession, s.filing_date, s.period, s.form_type, k.corporate_id,
                   o.owner_cik, o.owner_name, o.relationship, o.title
            FROM (SELECT DISTINCT * FROM read_parquet({files('submission.parquet')})) s
            JOIN _ciks k USING (issuer_cik)
            JOIN (SELECT DISTINCT * FROM read_parquet({files('owner.parquet')})) o USING (accession)""").df()
        trans = duck.execute(f"""
            SELECT t.* FROM (SELECT DISTINCT * FROM read_parquet({files('trans.parquet')})) t
            WHERE t.accession IN (SELECT accession FROM read_parquet({files('submission.parquet')}) s
                                  JOIN _ciks k USING (issuer_cik))""").df()
        duck.close()
        filings['is_officer'] = filings['relationship'].fillna('').str.contains('Officer')
        filings['is_director'] = filings['relationship'].fillna('').str.contains('Director')
        filings['is_ten_pct_owner'] = filings['relationship'].fillna('').str.contains('TenPercentOwner')
        filings['name'] = filings['owner_name'].map(readable_name)
        filings['role'] = [role_of(t, o, d) for t, o, d in
                           zip(filings['title'], filings['is_officer'], filings['is_director'])]
        filings['period'] = pd.to_datetime(filings['period'])

        # executives per fiscal year: officers / directors, no entities, no pure 10 % owners
        people = filings[(filings['is_officer'] | filings['is_director']) & ~filings['owner_name'].map(is_entity)]
        years = con.execute("""SELECT corporate_id, fiscal_year, period_start, period_end FROM fundamentals_yearly
                               WHERE period_start IS NOT NULL AND period_end IS NOT NULL""").df()
        years = years[years['corporate_id'].isin(comp['corporate_id'])]
        years = years[years['period_end'] >= pd.Timestamp(first)]
        rows = []
        for (cid, fy, start, end) in years.itertuples(index=False):
            p = people[(people['corporate_id'] == cid) & (people['period'] >= pd.Timestamp(start))
                       & (people['period'] <= pd.Timestamp(end))]
            if p.empty:
                continue
            per = []
            for owner, g in p.sort_values('filing_date').groupby('owner_cik'):
                last = g.iloc[-1]
                titled = g[g['title'].notna() & (g['title'].str.strip() != '')]
                title = titled.iloc[-1]['title'] if not titled.empty else None
                officer, director = bool(g['is_officer'].any()), bool(g['is_director'].any())
                per.append({'name': last['name'], 'sec_name': last['owner_name'], 'cik': owner, 'title': title,
                            'role': role_of(title, officer, director),
                            'is_officer': officer, 'is_director': director})
            per.sort(key=lambda x: (ROLE_ORDER.index(x['role']), x['name'] or ''))
            rows.append({'corporate_id': int(cid), 'fiscal_year': int(fy), 'executives': per})
        out = pd.DataFrame(rows)
        if not out.empty:
            result.rows += common.upsert(con, 'fundamentals_yearly', out)
        result.updated = out['corporate_id'].nunique() if not out.empty else 0
    finally:
        con.close()

    # raw insider transactions (all reporting owners, incl. 10 % owners and entities)
    raw = trans.merge(filings[['accession', 'filing_date', 'form_type', 'corporate_id', 'owner_cik', 'owner_name',
                               'name', 'relationship', 'title', 'role', 'is_officer', 'is_director',
                               'is_ten_pct_owner']], on='accession')
    raw = raw.rename(columns={'accession': 'accession_no', 'name': 'owner_readable_name'})
    raw['downloaded_at'] = datetime.now()
    td, fd = pd.to_datetime(raw['trans_date']), pd.to_datetime(raw['filing_date'])
    raw['date_suspect'] = (td > fd) | (td > pd.Timestamp.now().normalize())
    rcon = common.connect('raw')
    try:
        result.rows += common.upsert(rcon, 'insider_transactions_raw', raw)
    finally:
        rcon.close()
    result.message = f'{len(out)} company-years, {len(raw)} insider transactions'
    log.info(result.message)
    return result
