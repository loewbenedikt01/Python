"""
Step 'ids': sync the _tickers files with data/corporate_ids.csv and the
instruments table.

corporate_ids.csv is the single source of truth for company IDs:
    corporate_id, company_name, cik, tickers, assigned_at, merged_into, status
- status: active / merged / removed (no ticker in the ticker files any more)
  / reassigned (its tickers turned out to belong to another company)
- one row per company; tickers separated by ';', primary ticker first
- an ID never changes and is never reused; an ID retired by a merge keeps
  its row with merged_into = the surviving ID and no tickers
- tickers are grouped into companies by SEC CIK (US companies; non-US only
  when config.MATCH_NON_US_TO_SEC), otherwise by exact company name
- the one exception to "IDs never change": a ticker that was grouped into a
  company by mistake (same name in the ticker file, e.g. BEZ.DE Berentzen
  under 'Beiersdorf AG') and is split off after the name is corrected gets a
  new ID; the company keeps its ID (the group holding the registry row's
  first ticker)
- corrections from config.TICKER_CHANGES_CSV are applied once per row (ticker + change):
  'rename' keeps the ID; 'a' / 'a+add' (the ticker is a different company than
  its old name said) -> new ID, the old ID becomes 'reassigned' if it has no
  tickers left; 'replaced by X' -> X inherits the ID; 'removed' / 'added'
  follow from the ticker files
"""

from __future__ import annotations

import json
import random
import re
import time
from datetime import date, datetime

import pandas as pd

import config
from pipeline import common


log = common.get_logger('pipeline.ids')

ID_MIN, ID_MAX = 10000, 99999
REGISTRY_COLS  = ['corporate_id', 'company_name', 'cik', 'tickers', 'assigned_at', 'merged_into', 'status']
SEC_TICKERS_URL = 'https://www.sec.gov/files/company_tickers.json'
SEC_TICKERS_CACHE = config.CACHE_DIR / 'sec_company_tickers.json'
SEC_MATCHES_CSV = config.REVIEW_DIR / 'sec_matches.csv'


# ----
# REGISTRY FILE
# ----

def load_registry(path=config.CORPORATE_IDS_CSV) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=REGISTRY_COLS)
    df = pd.read_csv(path, dtype={'company_name': str, 'tickers': str, 'assigned_at': str})
    df['corporate_id'] = df['corporate_id'].astype(int)
    df['cik'] = df['cik'].astype('Int64')
    df['merged_into'] = df['merged_into'].astype('Int64')
    df['tickers'] = df['tickers'].fillna('')
    if 'status' not in df.columns:
        df['status'] = ['merged' if pd.notna(m) else 'active' for m in df['merged_into']]
    return df[REGISTRY_COLS]


def save_registry(df: pd.DataFrame, path=config.CORPORATE_IDS_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # active companies A-Z, then removed / reassigned / merged IDs
    order = {'active': 0, 'removed': 1, 'reassigned': 2, 'merged': 3}
    df = df.assign(_o=df['status'].map(order).fillna(9)).sort_values(['_o', 'company_name'])
    df[REGISTRY_COLS].to_csv(path, index=False)


def load_old_ids(path=config.OLD_CORPORATE_IDS_CSV) -> dict[str, int]:
    """
    Ticker -> ID from the previous registry (_database/corporate_ids.csv).
    """
    if not path.exists():
        return {}
    old = pd.read_csv(path, dtype={'Ticker': str})
    return dict(zip(old['Ticker'], old['corporate_id'].astype(int)))


def apply_ticker_changes(registry: pd.DataFrame, changes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply a corrections list (columns ticker, change) to the registry before
    assign_ids(). Pure function, see the module docstring for the rules.
    """
    reg = registry.copy()
    reg['tickers'] = reg['tickers'].fillna('')
    counts = {'reassigned_tickers': 0, 'replaced': 0, 'retired_ids': 0}
    for t, change in zip(changes['ticker'], changes['change'].fillna('')):
        hit = reg.index[reg['tickers'].map(lambda s: t in s.split(';'))]
        if hit.empty:
            continue
        i = hit[0]
        ts = reg.at[i, 'tickers'].split(';')
        if change in ('a', 'a+add'):
            ts = [x for x in ts if x != t]
            counts['reassigned_tickers'] += 1
            if not ts:
                reg.at[i, 'status'] = 'reassigned'
                counts['retired_ids'] += 1
            log.info(f"{t}: different company than '{reg.at[i, 'company_name']}' -> new ID "
                     f"(old ID {reg.at[i, 'corporate_id']}{' reassigned' if not ts else ''})")
        elif change.startswith('replaced by'):
            new = change.split('replaced by', 1)[1].strip()
            ts = [new] + [x for x in ts if x != t] + [t]          # new ticker first, old one kept listed
            counts['replaced'] += 1
            log.info(f'{t} replaced by {new}: keeps corporate_id {reg.at[i, "corporate_id"]}')
        else:
            continue
        reg.at[i, 'tickers'] = ';'.join(dict.fromkeys(ts))
    return reg, counts


def ticker_map(registry: pd.DataFrame) -> dict[str, int]:
    """
    Ticker -> corporate_id for active (not merged) companies.
    """
    out = {}
    for cid, tickers in zip(registry['corporate_id'], registry['tickers']):
        for t in filter(None, str(tickers).split(';')):
            out[t] = int(cid)
    return out


# ----
# SEC CIK LOOKUP
# ----

def sec_tickers(max_age_days: int = 1) -> pd.DataFrame:
    """
    SEC company_tickers.json as rows (cik, ticker, title, rank); cached for a day.
    rank = position in SEC's list, used to find a company's first (Class A) ticker.
    """
    fresh = SEC_TICKERS_CACHE.exists() and time.time() - SEC_TICKERS_CACHE.stat().st_mtime < max_age_days * 86400
    if not fresh:
        data = common.http_get(SEC_TICKERS_URL, 'sec').json()
        SEC_TICKERS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        SEC_TICKERS_CACHE.write_text(json.dumps(data), encoding='utf-8')
    data = json.loads(SEC_TICKERS_CACHE.read_text(encoding='utf-8'))
    rows = [{'cik': int(v['cik_str']), 'ticker': v['ticker'].upper(), 'title': v['title'], 'rank': int(k)}
            for k, v in data.items()]
    return pd.DataFrame(rows)


def yahoo_to_sec(ticker: str) -> str:
    return ticker.upper().replace('.', '-')


def lookup_cik_by_name(name: str) -> tuple[int | None, str | None]:
    """
    EDGAR company search for 10-K filers. Accepted only if exactly one company matches.
    """
    query = re.sub(r'[,.]|\b(Inc|Corp|Corporation|Co|Ltd|plc|Class [A-Z])\b', ' ', name).strip()
    query = re.sub(r'\s+', ' ', query)
    r = common.http_get('https://www.sec.gov/cgi-bin/browse-edgar', 'sec', params={
        'action': 'getcompany', 'company': query, 'type': '10-K', 'owner': 'include',
        'count': 10, 'output': 'atom'})
    if 'atom' not in r.headers.get('content-type', ''):
        return None, None                   # several companies matched -> HTML list
    ciks = set(re.findall(r'<cik>(\d+)</cik>', r.text))
    names = re.findall(r'<conformed-name>([^<]+)</conformed-name>', r.text)
    if len(ciks) != 1:
        return None, None
    return int(ciks.pop()), (names[0] if names else None)


def resolve_ciks(equities: pd.DataFrame, sec: pd.DataFrame) -> tuple[dict[str, int], dict[str, int], list[dict]]:
    """
    Ticker -> CIK for tickers that should be matched to SEC, ticker -> SEC rank,
    and review rows describing how each ticker was matched.
    """
    by_ticker = sec.drop_duplicates('ticker').set_index('ticker')
    cik_map, rank_map, review = {}, {}, []

    for t, name, group in zip(equities['ticker'], equities['name'], equities['group_name']):
        if t in config.SEC_CIK_OVERRIDES:
            cik = config.SEC_CIK_OVERRIDES[t]
            if cik:
                cik_map[t] = int(cik)
            review.append({'ticker': t, 'name': name, 'cik': cik, 'sec_name': None, 'method': 'override'})
            continue
        if group != config.US_GROUP and not config.MATCH_NON_US_TO_SEC:
            continue

        key = yahoo_to_sec(t)
        if key in by_ticker.index:
            row = by_ticker.loc[key]
            cik_map[t], rank_map[t] = int(row['cik']), int(row['rank'])
            review.append({'ticker': t, 'name': name, 'cik': int(row['cik']), 'sec_name': row['title'], 'method': 'ticker'})
            continue

        try:
            cik, sec_name = lookup_cik_by_name(name)
        except Exception as e:
            cik, sec_name = None, f'lookup failed: {common.format_error(e)}'
        if cik:
            cik_map[t] = cik
            log.info(f'{t}: not in SEC ticker list, matched by name to CIK {cik} ({sec_name})')
        else:
            log.warning(f'{t}: no SEC CIK found by ticker or name — no fundamentals')
        review.append({'ticker': t, 'name': name, 'cik': cik, 'sec_name': sec_name,
                       'method': 'name' if cik else 'not found'})
    return cik_map, rank_map, review


# ----
# ID ASSIGNMENT
# ----

def _company_name(names: list[str]) -> str:
    """
    Company name from its tickers' names, without share-class suffixes.
    """
    name = names[0]
    return re.sub(r'\s+(Class|Series)\s+[A-Z]\b.*$', '', name).strip()


def _order_tickers(tickers: list[str], names: dict[str, str], rank_map: dict[str, int],
                   file_order: dict[str, int]) -> list[str]:
    """
    Primary first: 'Class A' in the name, then SEC's order, then _tickers file order.
    """
    def key(t):
        is_class_a = bool(re.search(r'\bClass A\b', names.get(t, '')))
        return (not is_class_a, rank_map.get(t, 10**9), file_order.get(t, 10**9))
    return sorted(tickers, key=key)


def assign_ids(equities: pd.DataFrame, cik_map: dict[str, int], rank_map: dict[str, int],
               registry: pd.DataFrame, old_ids: dict[str, int],
               rng: random.Random | None = None, today: str | None = None,
               old_ids_date: str | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Return the updated registry and counts. Pure function (no I/O) so it can be tested.

    Rules: existing IDs never change; a company whose tickers carry several
    IDs keeps the primary ticker's ID and the others get merged_into; new
    companies get a random unused 5-digit ID. Companies no longer in the
    ticker files keep their row unchanged.
    """
    rng   = rng or random.Random()
    today = today or date.today().isoformat()
    registry = registry.copy()
    if 'status' not in registry.columns:
        registry['status'] = ['merged' if pd.notna(m) else 'active' for m in registry['merged_into']]
    current  = ticker_map(registry[registry['status'] != 'reassigned'])
    names      = dict(zip(equities['ticker'], equities['name']))
    file_order = {t: i for i, t in enumerate(equities['ticker'])}
    used = set(registry['corporate_id'].astype(int)) | set(old_ids.values())

    # group tickers into companies
    groups: dict[tuple, list[str]] = {}
    for t in equities['ticker']:
        key = ('cik', cik_map[t]) if t in cik_map else ('name', names[t].strip())
        groups.setdefault(key, []).append(t)

    counts = {'new': 0, 'kept': 0, 'merged': 0, 'split': 0}
    rows = {int(cid): row for cid, row in zip(registry['corporate_id'], registry.to_dict('records'))}

    # owner of each existing ID: the group that holds the first ticker of its registry row
    group_of = {t: key for key, ts in groups.items() for t in ts}
    id_owner = {}
    for cid, tickers in zip(registry['corporate_id'], registry['tickers']):
        first = next((t for t in str(tickers).split(';') if t in group_of), None)
        if first is not None:
            id_owner[int(cid)] = group_of[first]

    for key, tickers in groups.items():
        tickers = _order_tickers(tickers, names, rank_map, file_order)
        ids = []
        for t in tickers:
            cid = current.get(t, old_ids.get(t))
            if cid is not None and cid not in ids:
                if id_owner.get(cid, key) != key:          # ID belongs to the company this ticker was split from
                    counts['split'] += 1
                    log.info(f'{t}: split from corporate_id {cid} ({names[t]}), gets a new ID')
                    continue
                ids.append(cid)

        if not ids:
            cid = rng.randint(ID_MIN, ID_MAX)
            while cid in used:
                cid = rng.randint(ID_MIN, ID_MAX)
            used.add(cid)
            counts['new'] += 1
        else:
            cid = ids[0]                       # the primary ticker's ID wins
            counts['kept'] += 1
            for other in ids[1:]:
                owner = next(t for t in tickers if current.get(t, old_ids.get(t)) == other)
                r = rows.get(other) or {'corporate_id': other, 'company_name': names[owner],
                                        'cik': None, 'assigned_at': old_ids_date or today}
                r.update({'tickers': '', 'merged_into': cid, 'status': 'merged'})
                rows[other] = r
                counts['merged'] += 1
                log.info(f'corporate_id {other} merged into {cid} ({", ".join(tickers)})')

        existing = rows.get(cid, {})
        old_tickers = [t for t in str(existing.get('tickers') or '').split(';') if t]
        # tickers removed from the ticker files stay listed (after the current ones)
        all_tickers = tickers + [t for t in old_tickers if t not in names]
        seeded = not existing and cid in old_ids.values()
        rows[cid] = {
            'corporate_id': cid,
            'company_name': _company_name([names[t] for t in tickers]),      # follows renames

            'cik':          key[1] if key[0] == 'cik' else existing.get('cik'),
            'tickers':      ';'.join(dict.fromkeys(all_tickers)),
            'assigned_at':  existing.get('assigned_at') or (old_ids_date if seeded and old_ids_date else today),
            'merged_into':  None,
            'status':       'active',
        }

    # companies without any ticker in the files: removed (unless merged / reassigned)
    active = {cid for cid, r in rows.items() if r.get('status') == 'active'
              and any(t in names for t in str(r.get('tickers') or '').split(';'))}
    for cid, r in rows.items():
        if r.get('status') not in ('merged', 'reassigned') and cid not in active:
            r['status'] = 'removed'
    out = pd.DataFrame(list(rows.values()), columns=REGISTRY_COLS)
    out['corporate_id'] = out['corporate_id'].astype(int)
    out['cik'] = out['cik'].astype('Int64')
    out['merged_into'] = out['merged_into'].astype('Int64')
    return out, counts


# ----
# STEP
# ----

def run(ctx: common.RunContext) -> common.StepResult:
    result  = common.StepResult()
    tickers = common.load_ticker_files()
    tickers = tickers[~tickers['asset_class'].isin(config.FRED_ASSET_CLASSES)]
    equities = tickers[tickers['asset_class'] == config.EQUITIES_ASSET_CLASS].drop_duplicates('ticker')

    # IDs are always assigned for all equities (not only --tickers), so grouping stays consistent
    sec = sec_tickers()
    cik_map, rank_map, review = resolve_ciks(equities, sec)
    registry = load_registry()
    first_run = registry.empty
    old_ids = load_old_ids() if first_run else {}

    # corrections list, applied once per file version
    change_counts, change_key = {}, None
    if config.TICKER_CHANGES_CSV.exists():
        # every row is applied once (key = ticker + change): extending the file later only applies the new
        # rows; re-applying an 'a' row would hand out yet another new ID
        import hashlib
        changes = pd.read_csv(config.TICKER_CHANGES_CSV, dtype=str)
        changes['key'] = 'ticker_change:' + changes['ticker'] + '|' + changes['change'].fillna('')
        file_key = 'ticker_changes:' + hashlib.sha1(config.TICKER_CHANGES_CSV.read_bytes()).hexdigest()[:12]
        ccon = common.connect('companies')
        try:
            applied = {r[0] for r in ccon.execute(
                "SELECT dataset FROM processed_datasets WHERE dataset LIKE 'ticker_change%'").fetchall()}
        finally:
            ccon.close()
        if file_key in applied:                        # applied as a whole file before (first version of this rule)
            applied |= set(changes['key'])
        pending = changes[~changes['key'].isin(applied)]
        change_key = changes['key'].tolist()           # recorded after saving (incl. rows done as a whole file)
        if not pending.empty:
            registry, change_counts = apply_ticker_changes(registry, pending)
            log.info(f'{config.TICKER_CHANGES_CSV.name}: {len(pending)} new rows applied: {change_counts}')

    new_registry, counts = assign_ids(equities, cik_map, rank_map, registry, old_ids,
                                      old_ids_date=_old_file_date() if first_run else None)
    save_registry(new_registry)
    pd.DataFrame(review).to_csv(SEC_MATCHES_CSV, index=False)
    status = new_registry['status'].value_counts().to_dict()
    log.info(f"corporate_ids.csv: {counts['kept']} kept, {counts['new']} new, {counts['merged']} merged, "
             f"{counts['split']} split; status {status}")
    if change_key:
        ccon = common.connect('companies')
        try:
            common.upsert(ccon, 'processed_datasets', pd.DataFrame(
                {'dataset': change_key, 'processed_at': datetime.now(), 'rows_used': None}))
        finally:
            ccon.close()

    # company_info.status follows the registry
    ccon = common.connect('companies')
    try:
        common.upsert(ccon, 'company_info', new_registry.loc[
            new_registry['corporate_id'].isin([r[0] for r in ccon.execute('SELECT corporate_id FROM company_info').fetchall()]),
            ['corporate_id', 'status']])
    finally:
        ccon.close()

    # instruments: identity columns only; other columns are filled by later steps
    id_map = ticker_map(new_registry)
    selected = tickers[tickers['ticker'].isin(ctx.selected(tickers['ticker']))]
    priority = {c: i for i, c in enumerate(config.ASSET_CLASS_PRIORITY)}
    selected = selected.sort_values('asset_class', key=lambda s: s.map(priority).fillna(len(priority)), kind='stable')
    inst = selected.drop_duplicates('ticker', keep='first').assign(
        # only equities belong to a company (a ticker moved out of equities keeps its registry row, not the link)
        corporate_id=lambda d: d['ticker'].map(id_map).where(d['asset_class'] == config.EQUITIES_ASSET_CLASS).astype('Int64'),
        source='yfinance',
        updated_at=datetime.now(),
    )[['ticker', 'name', 'asset_class', 'group_name', 'corporate_id', 'source', 'updated_at']]
    con = common.connect('prices')
    try:
        result.rows = common.upsert(con, 'instruments', inst)
        # tickers no longer in any ticker file -> 'removed' (history kept, no updates); back -> 'active'
        con.register('_files', tickers[['ticker']].drop_duplicates())
        con.execute("""UPDATE instruments SET status = 'removed'
                       WHERE ticker NOT IN (SELECT ticker FROM _files) AND NOT ends_with(ticker, '_FRED')
                         AND coalesce(group_name, '') <> 'fx_helper' AND coalesce(status, '') <> 'removed'""")
        con.execute("""UPDATE instruments SET status = 'active', consecutive_failures = 0
                       WHERE ticker IN (SELECT ticker FROM _files) AND status = 'removed'""")
        con.unregister('_files')
    finally:
        con.close()

    result.message = f'changes: {change_counts}' if change_counts else ''
    result.updated = counts['new'] + counts['merged']
    result.skipped = counts['kept']
    result.failed  = [r['ticker'] for r in review if r['method'] == 'not found']
    return result


def _old_file_date() -> str:
    p = config.OLD_CORPORATE_IDS_CSV
    return date.fromtimestamp(p.stat().st_mtime).isoformat() if p.exists() else date.today().isoformat()
