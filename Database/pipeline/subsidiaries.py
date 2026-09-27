"""
Step 'subsidiaries': subsidiaries per fiscal year from Exhibit 21 of the 10-K
(fundamentals_yearly.subsidiaries = list of {name, jurisdiction}, n_subsidiaries).

- The 10-K of each of the last SUBSIDIARIES_YEARS fiscal years (the accession
  stored in fundamentals_yearly); if it has no EX-21, an amendment (10-K/A)
  filed within a year for the same period is tried.
- Exhibit 21 comes as an HTML table (name | jurisdiction | [ownership]) or as
  text lines: 'Name (Jurisdiction)', 'Name ..... Jurisdiction',
  'Name, a Delaware corporation', or a plain name.
- A 10-K without its own Exhibit 21 usually incorporates it by reference:
  the exhibit index links the earlier filing's document, which is loaded.
  Without such a link the previous fiscal year's list is carried forward
  (subsidiaries_carried_forward = TRUE). An exhibit that can't be parsed
  (text without any separators) stays empty, listed in
  data/review/subsidiaries_unparsed.csv.
- subsidiaries_scope: 'all', 'omits_insignificant' (the standard Item
  601(b)(21)(ii) clause: subsidiaries that together would not be significant
  are left out) or 'significant_only' (the list says it lists significant
  subsidiaries); subsidiaries_significant_only = scope 'significant_only'.
  Policy switches become visible (UNH 2025, NVDA 2024).
- Each filing is read once (processed_datasets 'ex21:v<PARSER_VERSION>:<acc>');
  the document is cached in data/cache/ex21/ so a new parser version re-parses
  without downloading again. Only US SEC filers (20-F Exhibit 8 when
  MATCH_NON_US_TO_SEC is switched on is not implemented).
"""

from __future__ import annotations

import html as htmllib
import re
from datetime import datetime

import pandas as pd

import config
from pipeline import common


log = common.get_logger('pipeline.subsidiaries')

CACHE = config.CACHE_DIR / 'ex21'
PARSER_VERSION = 6
UNPARSED_CSV = config.REVIEW_DIR / 'subsidiaries_unparsed.csv'

HEADER = re.compile(r'(?i)^(name(\s+of\s+(the\s+)?(subsidiary|entity|company|corporation))?|subsidiar(y|ies)'
                    r'|jurisdiction.*|state\s+or\s+(other\s+)?(country|jurisdiction).*|country|'
                    r'(place|state|country)\s+of\s+(incorporation|organization|formation).*|ownership.*|%.*|'
                    r'percent.*|entity\s+name|legal\s+name)$')
JURISDICTION_HEADER = re.compile(r'(?i)organized under|jurisdiction|incorporat|state or|country of|domicile|'
                                 r'place of (formation|organization)|ownership|percent')
NOISE = re.compile(r'(?i)(subsidiaries of |exhibit\s*21|list of (significant )?subsidiaries|^page\s*\d+|'
                   r'^\*|pursuant to|omitted|as of (december|january|february|march|april|may|june|july|'
                   r'august|september|october|november)|^\(?\d+\)?$|^the following|^\s*$)')
ENTITY_SUFFIX = re.compile(r'(?i)\b(inc|corp|corporation|llc|l\.l\.c|ltd|limited|lp|l\.p|plc|gmbh|ag|sa|s\.a|'
                           r'b\.v|bv|n\.v|nv|pte|pty|co|company|s\.r\.l|srl|kk|k\.k|ab|oy|as|a/s|spa|s\.p\.a|sas|'
                           r'sarl|s\.a\.r\.l|trust|partnership|holdings?|group|bank|fund|de c\.v)\b\.?')


# ----
# PARSING
# ----

def _clean(text: str) -> str:
    text = htmllib.unescape(re.sub(r'<[^>]+>', ' ', text))
    text = text.replace(' ', ' ').replace('​', '')
    return re.sub(r'\s+', ' ', text).strip(' ;:,\t')


def _pct(cell: str) -> bool:
    return bool(re.fullmatch(r'[\d.,]+\s*%?|%', cell.strip()))


def parse_table(doc: str) -> list[dict]:
    out = []
    for row in re.findall(r'(?is)<tr[^>]*>(.*?)</tr>', doc):
        cells = [_clean(c) for c in re.findall(r'(?is)<t[dh][^>]*>(.*?)</t[dh]>', row)]
        cells = [c for c in cells if c and c not in ('*', '-', '—')]
        while cells and re.fullmatch(r'\(?\d{1,4}[.)]?|[•·▪\-–]|[a-zA-Z][.)]', cells[0]):
            cells = cells[1:]                               # numbering / bullets: '1.', '(12)', '•'
        if not cells:
            continue
        name = cells[0]
        if HEADER.match(name) or NOISE.search(name) or len(name) > 200 or _pct(name):
            continue
        if any(JURISDICTION_HEADER.search(c) for c in cells[1:]):     # 'Name | Organized under the laws of'
            continue
        rest = [c for c in cells[1:] if not _pct(c) and not HEADER.match(c)]
        jurisdiction = rest[0] if rest else ''
        out.append({'name': name, 'jurisdiction': jurisdiction})
    return out


def parse_text(doc: str) -> list[dict]:
    text = re.sub(r'(?i)<br\s*/?>|</p>|</div>|</li>', '\n', doc)
    text = htmllib.unescape(re.sub(r'<[^>]+>', ' ', text)).replace(' ', ' ')
    out = []
    for raw in text.split('\n'):
        raw = re.sub(r'^\s*(?:[•·▪\-–*]|\(?\d{1,4}[.)]?)\s+', '', raw)       # bullets / numbering
        line = re.sub(r'[ \t]+', ' ', raw).strip(' ;:,\t')
        if len(line) < 3 or NOISE.search(line) or HEADER.match(line):
            continue
        m = re.match(r'^(.*?\S)\s*\(([^()]{2,60})\)$', line)                       # Name (Jurisdiction)
        if m and ENTITY_SUFFIX.search(m.group(1)):
            out.append({'name': m.group(1), 'jurisdiction': m.group(2)})
            continue
        parts = [p.strip() for p in re.split(r'\.{3,}|\s{2,}|\t|_{3,}', raw.strip()) if p.strip(' .')]
        if parts and HEADER.match(_clean(parts[0])):                                # 'Name of Subsidiary   Jurisdiction'
            continue
        if len(parts) >= 2:                                                         # Name ..... Jurisdiction
            out.append({'name': _clean(parts[0]), 'jurisdiction': _clean(parts[-1])})
            continue
        m = re.match(r'^(.*?),\s+an?\s+(.+?)\s+(corporation|limited liability company|limited partnership|'
                     r'partnership|company|trust|bank|limited company)\b', line, re.I)   # Name, a Delaware corporation
        if m:
            out.append({'name': m.group(1), 'jurisdiction': m.group(2)})
            continue
        if ENTITY_SUFFIX.search(line) and len(line) <= 150:                         # plain name
            out.append({'name': line, 'jurisdiction': ''})
    return out


# the list explicitly consists of significant subsidiaries ('Significant Subsidiaries of the Company',
# 'listed below are the significant subsidiaries', JPM 'significant legal entity subsidiaries')
SIGNIFICANT_LIST = re.compile(r'(?i)(?:listed below are|following (?:is a list of|are)|list of)\s+(?:the\s+)?'
                              r'(?:\w+\W{0,2}s\s+)?significant\s+(?:legal\s+entity\s+)?subsidiar|'
                              r'significant\s+(?:legal\s+entity\s+)?subsidiaries\s+of\s+(?:the|[A-Z])|'
                              r'significant\s+legal\s+entity\s+subsidiar')
# standard clause: subsidiaries left out because together they would not be a significant subsidiary
OMITS_CLAUSE = re.compile(r'(?i)would\s+not,?\s*(?:in\s+the\s+aggregate,?\s*)?(?:constitute|be)\s+(?:a\s+)?'
                          r'\W?significant\s+subsidiar|601\s*\(b\)\s*\(21\)\s*\(ii\)')


def scope_of(doc: str) -> str:
    """
    'significant_only' (the list says it lists significant subsidiaries), 'omits_insignificant'
    (the standard Item 601(b)(21)(ii) clause) or 'all'.
    """
    text = _clean(doc)
    if SIGNIFICANT_LIST.search(text):
        return 'significant_only'
    return 'omits_insignificant' if OMITS_CLAUSE.search(text) else 'all'


def is_significant_only(doc: str) -> bool:
    return scope_of(doc) == 'significant_only'


def parse_ex21(doc: str) -> list[dict]:
    """
    Subsidiaries from an Exhibit 21 document (HTML table or text), de-duplicated in order.
    """
    rows = parse_table(doc) if re.search(r'(?i)<table', doc) else []
    if not rows:
        rows = parse_text(doc)
    seen, out = set(), []
    for r in rows:
        key = (r['name'].lower(), r['jurisdiction'].lower())
        if key not in seen and len(r['name']) >= 3:
            seen.add(key)
            out.append(r)
    return out


# ----
# DOWNLOADS
# ----

def ex21_url(cik: int, accession: str) -> str | None:
    """
    Link to the EX-21 document of a filing (from its EDGAR index page), or None.
    """
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{accession}-index.htm"
    page = common.http_get(url, 'sec').text
    for row in re.findall(r'(?is)<tr[^>]*>(.*?)</tr>', page):
        cells = [_clean(c) for c in re.findall(r'(?is)<td[^>]*>(.*?)</td>', row)]
        if any(c.upper().startswith('EX-21') for c in cells):
            links = re.findall(r'href="([^"]+)"', row)
            if links:
                return 'https://www.sec.gov' + links[0] if links[0].startswith('/') else links[0]
    return None


def referenced_ex21(cik: int, accession: str, primary_doc: str | None) -> str | None:
    """
    A 10-K without its own Exhibit 21 usually incorporates it by reference; since 2017 the exhibit
    index links the earlier filing's document. Returns that document (cached), or None.
    """
    if not primary_doc:
        return None
    path = CACHE / f'{accession}.ref.htm'
    none = CACHE / f'{accession}.ref.none'
    if path.exists():
        return path.read_text(encoding='utf-8', errors='replace')
    if none.exists():
        return None
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"
    page = common.http_get(base + primary_doc, 'sec').text
    link = None
    for row in re.findall(r'(?is)<tr[^>]*>(.*?)</tr>', page):
        text = _clean(row)
        if re.search(r'(?i)subsidiar', text) and re.search(r'\b21(\.\d+)?\b', text):
            hrefs = [h for h in re.findall(r'(?i)href="([^"]+)"', row) if re.search(r'(?i)\.(htm|html|txt)$', h)]
            if hrefs:
                link = hrefs[0]
                break
    if link is None:
        none.write_text('')
        return None
    url = link if link.startswith('http') else ('https://www.sec.gov' + link if link.startswith('/') else base + link)
    doc = common.http_get(url, 'sec').text
    path.write_text(doc, encoding='utf-8')
    return doc


def fetch_ex21(cik: int, accession: str) -> str | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f'{accession}.htm'
    none = CACHE / f'{accession}.none'
    if path.exists():
        return path.read_text(encoding='utf-8', errors='replace')
    if none.exists():
        return None
    url = ex21_url(cik, accession)
    if url is None:
        none.write_text('')
        return None
    doc = common.http_get(url, 'sec').text
    path.write_text(doc, encoding='utf-8')
    return doc


# ----
# STEP
# ----

def run(ctx: common.RunContext) -> common.StepResult:
    result = common.StepResult()
    con = common.connect('companies')
    try:
        us = set(common.load_ticker_files().query('group_name == @config.US_GROUP')['ticker'])
        comp = con.execute("""SELECT corporate_id, primary_ticker, tickers, cik FROM company_info
                              WHERE sec_filer AND cik IS NOT NULL AND coalesce(status, 'active') = 'active'""").df()
        comp = comp[comp['tickers'].map(lambda ts: bool(us & set(ts)))]
        if ctx.tickers:
            wanted = {t.upper() for t in ctx.tickers}
            comp = comp[comp['tickers'].map(lambda ts: bool(wanted & {t.upper() for t in ts}))]
        years = con.execute("""
            SELECT y.corporate_id, y.fiscal_year, y.accession_no, y.period_end, f.cik
            FROM fundamentals_yearly y LEFT JOIN sec_filings f USING (accession_no)
            WHERE y.accession_no IS NOT NULL
            QUALIFY y.fiscal_year > max(y.fiscal_year) OVER (PARTITION BY y.corporate_id) - ?""",
            [config.SUBSIDIARIES_YEARS]).df()
        years = years[years['corporate_id'].isin(comp['corporate_id'])]
        done = {r[0] for r in con.execute("SELECT dataset FROM processed_datasets WHERE dataset LIKE ?",
                                          [f'ex21:v{PARSER_VERSION}:%']).fetchall()}
        cik_of = dict(zip(comp['corporate_id'], comp['cik']))
        amendments = con.execute("""SELECT corporate_id, accession_no, cik, report_date, filed_date FROM sec_filings
                                    WHERE form_type = '10-K/A'""").df()
        primary = dict(con.execute('SELECT accession_no, primary_doc FROM sec_filings').fetchall())
        log.info(f'{len(years)} company-years, {sum(f"ex21:v{PARSER_VERSION}:{a}" in done for a in years["accession_no"])} done before')

        rows, marks, unparsed = [], [], []
        for i, y in enumerate(years.itertuples(index=False), 1):
            key = f'ex21:v{PARSER_VERSION}:{y.accession_no}'
            if key in done:
                result.skipped += 1
                continue
            cik = int(y.cik) if pd.notna(y.cik) else int(cik_of[y.corporate_id])
            try:
                doc = fetch_ex21(cik, y.accession_no)
                if doc is None:                                     # try a 10-K/A for the same period
                    a = amendments[(amendments['corporate_id'] == y.corporate_id)
                                   & (pd.to_datetime(amendments['report_date']) == pd.Timestamp(y.period_end))]
                    for acc, acik in zip(a['accession_no'], a['cik']):
                        doc = fetch_ex21(int(acik), acc)
                        if doc:
                            break
                if doc is None:                                     # incorporated by reference -> linked filing
                    doc = referenced_ex21(cik, y.accession_no, primary.get(y.accession_no))
                subs = parse_ex21(doc) if doc else None
                if doc and not subs:                                # exhibit exists but can't be read
                    unparsed.append({'corporate_id': int(y.corporate_id), 'fiscal_year': int(y.fiscal_year),
                                     'accession_no': y.accession_no})
                    subs = None
                rows.append({'corporate_id': int(y.corporate_id), 'fiscal_year': int(y.fiscal_year),
                             'subsidiaries': subs, 'n_subsidiaries': len(subs) if subs else None,
                             'subsidiaries_significant_only': is_significant_only(doc) if subs else None,
                             'subsidiaries_scope': scope_of(doc) if subs else None,
                             'subsidiaries_carried_forward': False if subs else None})
                marks.append({'dataset': key, 'processed_at': datetime.now(),
                              'rows_used': len(subs) if subs else (0 if doc else -1)})
            except Exception as e:
                result.fail(f'{y.accession_no}', common.format_error(e), log)
            if i % 250 == 0:
                log.info(f'  ...{i}/{len(years)}')
                result.rows += _write(con, rows, marks)
                rows, marks = [], []
        result.rows += _write(con, rows, marks)
        result.rows += carry_forward(con, comp['corporate_id'].tolist())
        if unparsed:
            u = pd.DataFrame(unparsed).merge(comp[['corporate_id', 'primary_ticker']], on='corporate_id', how='left')
            u.to_csv(UNPARSED_CSV, index=False)
            log.info(f'{len(u)} filings with an Exhibit 21 that could not be parsed -> {UNPARSED_CSV.name}')
        result.updated = int(con.execute(f"""SELECT count(*) FROM processed_datasets WHERE dataset LIKE 'ex21:v{PARSER_VERSION}:%'
                                             AND rows_used > 0""").fetchone()[0])
        n_none = con.execute(f"""SELECT count(*) FROM processed_datasets WHERE dataset LIKE 'ex21:v{PARSER_VERSION}:%'
                                 AND rows_used = -1""").fetchone()[0]
        result.message = f'{result.updated} filings with subsidiaries, {n_none} without an Exhibit 21'
        log.info(result.message)
    finally:
        con.close()
    return result


def carry_forward(con, ids: list) -> int:
    """
    Fiscal years whose 10-K has no Exhibit 21 at all (not even a linked one) get the previous
    year's list, with subsidiaries_carried_forward = TRUE. Unreadable exhibits stay empty.
    """
    no_ex21 = {r[0].split(':', 2)[2] for r in con.execute(
        f"SELECT dataset FROM processed_datasets WHERE dataset LIKE 'ex21:v{PARSER_VERSION}:%' AND rows_used = -1").fetchall()}
    df = con.execute("""SELECT corporate_id, fiscal_year, accession_no, subsidiaries, n_subsidiaries,
                               subsidiaries_significant_only, subsidiaries_scope, subsidiaries_carried_forward
                        FROM fundamentals_yearly WHERE corporate_id IN (SELECT unnest(?))
                        ORDER BY corporate_id, fiscal_year""", [ids]).df()
    out = []
    for cid, g in df.groupby('corporate_id'):
        last = None
        for r in g.itertuples(index=False):
            has = r.n_subsidiaries is not None and not pd.isna(r.n_subsidiaries)
            if has:
                last = r
            elif r.accession_no in no_ex21 and last is not None:
                out.append({'corporate_id': int(cid), 'fiscal_year': int(r.fiscal_year),
                            'subsidiaries': last.subsidiaries, 'n_subsidiaries': int(last.n_subsidiaries),
                            'subsidiaries_significant_only': last.subsidiaries_significant_only,
                            'subsidiaries_scope': last.subsidiaries_scope,
                            'subsidiaries_carried_forward': True})
                last = last._replace(fiscal_year=r.fiscal_year)
    if out:
        common.upsert(con, 'fundamentals_yearly', pd.DataFrame(out))
        log.info(f'{len(out)} fiscal years without an Exhibit 21: previous list carried forward')
    return len(out)


def _write(con, rows, marks) -> int:
    if not rows:
        return 0
    n = common.upsert(con, 'fundamentals_yearly', pd.DataFrame(rows))
    common.upsert(con, 'processed_datasets', pd.DataFrame(marks))
    return n
