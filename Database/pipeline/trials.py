"""
Step 'trials': ClinicalTrials.gov API v2 -> trial versions (Parquet) and two
company tables.

Storage (DECISIONS.md): every version is appended to
data/trials/versions/<run>.parquet (zstd; DuckDB doesn't compress long JSON
strings: ~100k trials would be ~10 GB as a table, ~0.5 GB as Parquet).
raw.duckdb clinical_trials_raw is a VIEW over those files with the specified
columns: nct_id, raw_json, last_change_date, version, downloaded_at,
json_hash, is_latest, change_flag, changed_sections.

Which trials:
- the stored ones, config.TRIAL_NCT_IDS, `run.py trials --nct NCT...`;
- the trials of the healthcare companies (config.TRIAL_SECTORS): searched by
  sponsor / collaborator text (query.spons) with the company name without its
  legal form, or the names in config.TRIAL_SPONSOR_ALIASES; plus
  config.TRIAL_SPONSORS. At most config.TRIAL_MAX_TRIALS trials are stored:
  the budget is shared between the searches (small ones get all their trials,
  large ones share the rest), newest start date first. A search that didn't
  get all its trials is continued in a later run if budget is left.
- Incremental: stored trials and completed searches only request trials whose
  lastUpdatePostDate is on or after the latest stored last_change_date.

Versioning (unchanged): json_hash = sha256 of the normalised JSON (sorted
keys) without derivedSection.miscInfoModule.versionHolder (ClinicalTrials.gov
snapshot date). Same hash as the latest version -> nothing is written;
otherwise a new version with change_flag = TRUE and changed_sections = the
modules that differ. Version 1: change_flag = FALSE.

Company tables (companies.duckdb, rebuilt from the latest versions each run):
- clinical_trials: one row per trial; phase_groups = the phases 1-4 it counts
  in (Phase 1/2 -> [1, 2], Phase 2/3 -> [2, 3], Early Phase 1 -> [1],
  N/A -> []); start_year from the start date (actual or planned), else the
  first-posted date.
- clinical_trial_sponsors: lead sponsor and collaborators, one row each, with
  the matched corporate_id (a trial counts for every matched company).
  Matching on the normalised name: config.TRIAL_SPONSOR_ALIASES / _MATCH, the
  company name, a company's Exhibit 21 subsidiary names; else the parent named
  in the sponsor name ("Stiefel, a GSK Company", "... subsidiary of Pfizer");
  else, for company sponsors only (not foundations), the company name followed
  by more words ("Novartis Pharmaceuticals"). Names that fit several companies
  are left unmatched. Unmatched industry sponsors with their trial counts ->
  data/review/trial_sponsors_unmatched.csv.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from datetime import datetime

import duckdb
import pandas as pd

import config
from pipeline import common


log = common.get_logger('pipeline.trials')

API = 'https://clinicaltrials.gov/api/v2/studies'
PAGE_SIZE = 1000
IDS_PER_REQUEST = 200             # NCT IDs per filter.ids request (keeps the URL short)
BUILD_BATCH = 4000                # trials per batch when the company tables are built (memory)
VERSIONS_DIR = config.DATA_DIR / 'trials' / 'versions'
UNMATCHED_CSV = config.REVIEW_DIR / 'trial_sponsors_unmatched.csv'
RAW_COLUMNS = ['nct_id', 'raw_json', 'last_change_date', 'version', 'downloaded_at', 'json_hash',
               'is_latest', 'change_flag', 'changed_sections']
PHASE_GROUPS = {'EARLY_PHASE1': 1, 'PHASE1': 1, 'PHASE2': 2, 'PHASE3': 3, 'PHASE4': 4}


# ----
# VERSIONING (pure functions)
# ----

def _without_snapshot_date(study: dict) -> dict:
    s = copy.deepcopy(study)
    s.get('derivedSection', {}).get('miscInfoModule', {}).pop('versionHolder', None)
    return s


def normalised(study: dict) -> str:
    return json.dumps(_without_snapshot_date(study), sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def json_hash(study: dict) -> str:
    return hashlib.sha256(normalised(study).encode('utf-8')).hexdigest()


def modules(study: dict) -> dict:
    """
    {module name: content} over all sections; top-level values that aren't sections (hasResults) by their key.
    """
    s = _without_snapshot_date(study)
    out = {}
    for key, val in s.items():
        if isinstance(val, dict) and key.endswith('Section'):
            out.update(val)
        else:
            out[key] = val
    return out


def changed_sections(old: dict, new: dict) -> list[str]:
    a, b = modules(old), modules(new)
    return sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))


def last_change_date(study: dict):
    d = study.get('protocolSection', {}).get('statusModule', {}).get('lastUpdatePostDateStruct', {}).get('date')
    if not d:
        return None
    return pd.Timestamp(d if len(d) > 7 else f'{d}-01').date()        # 'YYYY-MM' -> first of the month


def nct_id(study: dict) -> str:
    return study['protocolSection']['identificationModule']['nctId']


def new_version(latest: dict | None, study: dict, now: datetime) -> dict | None:
    """
    The row to insert for a downloaded study, or None if its content equals the latest stored version.
    `latest`: {'version', 'json_hash', 'raw_json'} of the stored latest version, or None.
    """
    h = json_hash(study)
    if latest is not None and latest['json_hash'] == h:
        return None
    first = latest is None
    return {
        'nct_id': nct_id(study),
        'raw_json': json.dumps(study, ensure_ascii=False),
        'last_change_date': last_change_date(study),
        'version': 1 if first else latest['version'] + 1,
        'downloaded_at': now,
        'json_hash': h,
        'is_latest': True,
        'change_flag': not first,
        'changed_sections': [] if first else changed_sections(json.loads(latest['raw_json']), study),
    }


# ----
# NAMES AND PHASES (pure functions)
# ----

LEGAL_WORDS = {
    'inc', 'incorporated', 'corp', 'corporation', 'co', 'company', 'companies', 'ltd', 'limited', 'llc', 'lp',
    'plc', 'ag', 'se', 'sa', 'nv', 'bv', 'as', 'ab', 'publ', 'oyj', 'spa', 'gmbh', 'kgaa', 'kk', 'berhad',
    'bhd', 'pcl', 'public', 'the', 'holding', 'holdings', 'group', 'cayman', 'sarl', 'srl', 'sas', 'pty',
}


def norm_name(name: str | None) -> str:
    """
    Lower case, accents / punctuation removed (apostrophes dropped, not split: "Reddy's" -> "reddys"),
    '&' and 'and' removed, legal forms removed. 'Merck Sharp & Dohme LLC' -> 'merck sharp dohme'.
    """
    if not name:
        return ''
    s = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r"['’`]", '', s)
    s = re.sub(r'\b([a-z])/([a-z])\b', r'\1\2', s)        # "A/S" -> "as" (then dropped as legal form)
    s = re.sub(r'\b(?:[a-z]\.){2,}', lambda m: m.group(0).replace('.', ''), s)   # "S.A.", "S.p.A." -> "sa", "spa"
    s = re.sub(r'\(.*?\)', ' ', s)                        # "(Cayman)", "(publ)"
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    words = [w for w in s.split() if w not in LEGAL_WORDS and w != 'and']
    return ' '.join(words)


def search_term(company_name: str) -> str:
    """
    Text for query.spons: the company name without legal form, in its own spelling ("Dr. Reddy's Laboratories").
    """
    s = re.sub(r'\(.*?\)', ' ', company_name)
    words = [w for w in re.split(r'[\s,]+', s) if w and re.sub(r'[^a-z]', '', w.lower()) not in LEGAL_WORDS]
    while words and words[-1].lower() in ('and', '&'):
        words.pop()
    return ' '.join(words).strip()


def as_list(v) -> list:
    """
    A DuckDB list value as a Python list: None / NA -> [], numpy array -> list.
    """
    if v is None or (not hasattr(v, '__len__') and pd.isna(v)):
        return []
    return list(v)


def phase_groups(phases) -> list[int]:
    """
    ['PHASE1', 'PHASE2'] -> [1, 2]; ['EARLY_PHASE1'] -> [1]; ['NA'] / None -> [].
    """
    return sorted({PHASE_GROUPS[p] for p in as_list(phases) if p in PHASE_GROUPS})


def allocate(counts: dict[str, int], budget: int) -> dict[str, int]:
    """
    Share `budget` trials between searches: each gets min(its count, an equal share); what small searches
    don't use goes to the larger ones (water filling).
    """
    out = {k: 0 for k in counts}
    left = {k: v for k, v in counts.items() if v > 0}
    while left and budget > 0:
        share = max(budget // len(left), 1)
        for k in sorted(left, key=lambda k: left[k]):
            if budget <= 0:
                break
            take = min(left[k] - out[k], share, budget)
            out[k] += take
            budget -= take
        left = {k: v for k, v in left.items() if out[k] < v}
    return out


PARENT_PATTERNS = [
    re.compile(r'[,\-(]\s*(?:a|an)\s+(.+?)\s+(?:company|group company|affiliate)\)?\s*$', re.I),   # "Stiefel, a GSK Company"
    re.compile(r'(?:subsidiary|affiliate|part|division)\s+of\s+(.+?)\)?\s*$', re.I),         # "... subsidiary of Pfizer"
]
PREFIX_CLASSES = {'INDUSTRY', 'UNKNOWN', None}


def match_sponsor(name: str, exact: dict[str, set], prefixes: dict[str, set],
                  sponsor_class: str | None = 'INDUSTRY') -> tuple[int | None, str | None]:
    """
    corporate_id and method for a sponsor name. exact: normalised name -> {(corporate_id, method)};
    prefixes: normalised company name -> {corporate_id}. Ambiguous names -> (None, 'ambiguous').
    Order: exact name, the parent named in the name ("Stiefel, a GSK Company"), company name at the start
    (companies only: not for foundations or other sponsor classes).
    """
    n = norm_name(name)
    if not n:
        return None, None
    hits = exact.get(n)
    if hits:
        ids = {cid for cid, _ in hits}
        if len(ids) == 1:
            order = ['alias', 'name', 'subsidiary']
            method = min((m for _, m in hits), key=order.index)
            return next(iter(ids)), method
        return None, 'ambiguous'
    for pat in PARENT_PATTERNS:
        m = pat.search(name)
        if m:
            cid, method = match_sponsor(m.group(1), exact, prefixes, sponsor_class)
            if cid is not None:
                return cid, 'parent'
    if sponsor_class not in PREFIX_CLASSES or 'foundation' in n.split():
        return None, None
    words = n.split()
    for k in range(len(words) - 1, 0, -1):                 # longest company name at the start of the sponsor name
        ids = prefixes.get(' '.join(words[:k]))
        if ids:
            return (next(iter(ids)), 'name_prefix') if len(ids) == 1 else (None, 'ambiguous')
    return None, None


# ----
# DOWNLOAD
# ----

def fetch(params: dict, limit: int | None = None) -> list[dict]:
    """
    All studies for a query (follows nextPageToken), at most `limit`.
    """
    out, token = [], None
    while True:
        size = PAGE_SIZE if limit is None else min(PAGE_SIZE, limit - len(out))
        if size <= 0:
            return out
        p = {**params, 'pageSize': size, 'format': 'json'}
        if token:
            p['pageToken'] = token
        j = common.http_get(API, 'ctgov', params=p).json()
        out += j.get('studies', [])
        token = j.get('nextPageToken')
        if not token:
            return out


def count(params: dict) -> int:
    j = common.http_get(API, 'ctgov', params={**params, 'pageSize': 1, 'countTotal': 'true', 'format': 'json'}).json()
    return int(j.get('totalCount') or 0)


def since_filter(since) -> dict:
    return {'filter.advanced': f'AREA[LastUpdatePostDate]RANGE[{since:%Y-%m-%d},MAX]'} if since else {}


def search_filter(since=None) -> dict:
    """
    Company searches: trials starting on / after TRIAL_START_FROM, and (incremental) updated since `since`.
    """
    parts = []
    if config.TRIAL_START_FROM:
        parts.append(f'AREA[StartDate]RANGE[{config.TRIAL_START_FROM},MAX]')
    if since:
        parts.append(f'AREA[LastUpdatePostDate]RANGE[{since:%Y-%m-%d},MAX]')
    return {'filter.advanced': ' AND '.join(parts)} if parts else {}


# ----
# STORAGE: Parquet versions + view
# ----

def _pattern() -> str:
    return (VERSIONS_DIR / '*.parquet').as_posix()


def ensure_view(rcon) -> bool:
    """
    (Re)create the clinical_trials_raw view over the Parquet files. Moves rows of an old
    clinical_trials_raw table into a Parquet file first. False if no versions exist yet.
    """
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    kind = rcon.execute("""SELECT table_type FROM information_schema.tables
                           WHERE table_name = 'clinical_trials_raw'""").fetchone()
    if kind and kind[0] == 'BASE TABLE':
        old = rcon.execute('SELECT * FROM clinical_trials_raw').df()
        if not old.empty:
            write_versions(old)
            log.info(f'{len(old)} rows of the clinical_trials_raw table moved to {VERSIONS_DIR}')
        rcon.execute('DROP TABLE clinical_trials_raw')
    if not any(VERSIONS_DIR.glob('*.parquet')):
        return False
    rcon.execute(f"""
        CREATE OR REPLACE VIEW clinical_trials_raw AS
        SELECT nct_id, CAST(raw_json AS JSON) AS raw_json, last_change_date, version, downloaded_at, json_hash,
               version = max(version) OVER (PARTITION BY nct_id) AS is_latest, change_flag, changed_sections
        FROM read_parquet('{_pattern()}')""")
    return True


def write_versions(df: pd.DataFrame) -> None:
    """
    Append versions as a new Parquet file (zstd). is_latest is not stored (the view computes it).
    """
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = df.drop(columns=['is_latest'], errors='ignore').copy()
    out['raw_json'] = out['raw_json'].astype(str)
    out['changed_sections'] = out['changed_sections'].map(lambda v: [str(x) for x in v] if v is not None else [])
    path = VERSIONS_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}.parquet"
    con = duckdb.connect()
    try:
        con.register('_v', out)
        # explicit types: every file has the same schema (the view reads them together)
        con.execute(f"""
            COPY (SELECT CAST(nct_id AS VARCHAR) AS nct_id, CAST(raw_json AS VARCHAR) AS raw_json,
                         CAST(last_change_date AS DATE) AS last_change_date, CAST(version AS INTEGER) AS version,
                         CAST(downloaded_at AS TIMESTAMP) AS downloaded_at, CAST(json_hash AS VARCHAR) AS json_hash,
                         CAST(change_flag AS BOOLEAN) AS change_flag,
                         CAST(changed_sections AS VARCHAR[]) AS changed_sections FROM _v)
            TO '{path.as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
    finally:
        con.close()


# ----
# STEP
# ----

def company_searches(ccon) -> pd.DataFrame:
    """
    corporate_id, ticker, term: the sponsor searches for the healthcare companies (+ TRIAL_SPONSORS).
    """
    comp = ccon.execute("""
        SELECT corporate_id, primary_ticker, name FROM company_info
        WHERE coalesce(status, 'active') = 'active' AND sector IN (SELECT unnest(?))""",
        [list(config.TRIAL_SECTORS)]).df()
    rows = []
    for c in comp.itertuples(index=False):
        for term in config.TRIAL_SPONSOR_ALIASES.get(c.primary_ticker) or [search_term(c.name)]:
            rows.append((int(c.corporate_id), c.primary_ticker, term))
    rows += [(None, None, s) for s in config.TRIAL_SPONSORS]
    return pd.DataFrame(rows, columns=['corporate_id', 'ticker', 'term']).drop_duplicates('term')


def run(ctx: common.RunContext) -> common.StepResult:
    result = common.StepResult()
    now = datetime.now()
    rcon = common.connect('raw')
    ccon = common.connect('companies', read_only=True)
    try:
        has_view = ensure_view(rcon)
        latest = {r[0]: {'version': r[1], 'json_hash': r[2]} for r in rcon.execute("""
            SELECT nct_id, version, json_hash FROM clinical_trials_raw WHERE is_latest""").fetchall()} if has_view else {}
        since = rcon.execute('SELECT max(last_change_date) FROM clinical_trials_raw').fetchone()[0] if has_view else None
        # a search made with another TRIAL_START_FROM counts as not searched
        searched = {t: (n, done, f or 0) for t, n, done, f, sf in rcon.execute(
            'SELECT term, n_total, complete, n_fetched, start_from FROM trial_searches').fetchall()
            if (sf or None) == (config.TRIAL_START_FROM or None)}
        searches = company_searches(ccon) if not ctx.nct_ids else pd.DataFrame(columns=['corporate_id', 'ticker', 'term'])
    finally:
        ccon.close()

    studies: dict[str, dict] = {}

    def add(label, params, limit=None):
        try:
            got = fetch(params, limit)
            for s in got:
                studies[nct_id(s)] = s
            return got
        except Exception as e:
            result.fail(label, common.format_error(e), log)
            return None

    # 1. stored trials: only the changed ones
    stored = sorted(latest)
    for i in range(0, len(stored), IDS_PER_REQUEST):
        chunk = stored[i:i + IDS_PER_REQUEST]
        add(f'stored {chunk[0]}..', {'filter.ids': ','.join(chunk), **since_filter(since)})

    # 2. explicit NCT IDs: in full
    wanted = {i.upper() for i in [*config.TRIAL_NCT_IDS, *(ctx.nct_ids or [])]}
    new_ids = sorted(wanted - latest.keys())
    for i in range(0, len(new_ids), IDS_PER_REQUEST):
        chunk = new_ids[i:i + IDS_PER_REQUEST]
        add(f'new {chunk[0]}..', {'filter.ids': ','.join(chunk)})
        for m in sorted(set(chunk) - studies.keys()):
            result.fail(m, 'not found on ClinicalTrials.gov', log)

    # 3. company searches: completed ones incremental, the others share the budget of new trials
    search_rows = []
    budget = max(config.TRIAL_MAX_TRIALS - len(latest) - len(new_ids), 0)
    todo = {}
    for s in searches.itertuples(index=False):
        n, done, fetched = searched.get(s.term, (None, False, 0))
        if done:
            add(f'search {s.term}', {'query.spons': s.term, **search_filter(since)})
            search_rows.append({'term': s.term, 'ticker': s.ticker, 'n_total': n, 'complete': True, 'last_run': now,
                                'n_fetched': fetched, 'start_from': config.TRIAL_START_FROM})
        else:
            try:
                todo[s.term] = (count({'query.spons': s.term, **search_filter()}), s.ticker, fetched)
            except Exception as e:
                result.fail(f'search {s.term}', common.format_error(e), log)
    if todo:
        before = set(studies) | set(latest)
        # budget per search for trials it hasn't fetched yet; it fetches its first n_fetched + alloc trials
        # (newest start first) again: the ones fetched before are unchanged (hash) and cost no budget
        alloc = allocate({t: max(n - f, 0) for t, (n, _, f) in todo.items()}, budget)
        log.info(f'{len(todo)} searches not complete: {sum(n for n, _, _ in todo.values()):,} trials found, '
                 f'budget {budget:,} (TRIAL_MAX_TRIALS {config.TRIAL_MAX_TRIALS:,}, {len(latest):,} stored)')
        for term, (n, ticker, fetched) in todo.items():
            if not alloc[term]:
                continue
            got = add(f'search {term}', {'query.spons': term, 'sort': 'StartDate:desc', **search_filter()},
                      fetched + alloc[term])
            if got is None:
                continue
            search_rows.append({'term': term, 'ticker': ticker, 'n_total': n, 'complete': len(got) >= n,
                                'last_run': now, 'n_fetched': len(got), 'start_from': config.TRIAL_START_FROM})
        # overlapping searches can bring more new trials than the budget: keep the budget
        new = [k for k in studies if k not in before]
        for k in new[budget:]:
            studies.pop(k)

    # 4. versions
    rows, changed = [], []
    for nid, s in studies.items():
        prev = latest.get(nid)
        if prev is not None:
            h = json_hash(s)
            if h == prev['json_hash']:
                result.skipped += 1
                continue
            old = rcon.execute('SELECT raw_json FROM clinical_trials_raw WHERE nct_id = ? AND is_latest', [nid]).fetchone()
            prev = {**prev, 'raw_json': old[0]}
        rows.append(new_version(prev, s, now))
    try:
        if rows:
            df = pd.DataFrame(rows)
            write_versions(df)
            n_changed = int((df['version'] > 1).sum())
            result.rows = len(df)
            result.updated = len(df)
            log.info(f'{len(df) - n_changed} new trials, {n_changed} new versions of stored trials')
        if search_rows:
            common.upsert(rcon, 'trial_searches', pd.DataFrame(search_rows))
        if ensure_view(rcon):
            build_company_tables(rcon)
    finally:
        rcon.close()
    return result


# ----
# COMPANY TABLES
# ----

def build_company_tables(rcon) -> None:
    """
    clinical_trials and clinical_trial_sponsors (companies.duckdb) from the latest versions.
    """
    # latest version per trial first (no JSON read), then the fields in batches: reading all versions' JSON at once
    # with a window over it needed > 25 GB for 31k trials
    rcon.execute(f"""CREATE OR REPLACE TEMP TABLE _latest AS
                     SELECT nct_id, max(version) AS version FROM read_parquet('{_pattern()}') GROUP BY 1""")
    ids = [r[0] for r in rcon.execute('SELECT nct_id FROM _latest ORDER BY 1').fetchall()]
    fields = """
        SELECT nct_id, version,
               raw_json->>'$.protocolSection.identificationModule.briefTitle' AS title,
               raw_json->>'$.protocolSection.statusModule.overallStatus' AS overall_status,
               raw_json->>'$.protocolSection.designModule.studyType' AS study_type,
               CAST(json_extract(raw_json, '$.protocolSection.designModule.phases') AS VARCHAR[]) AS phases,
               raw_json->>'$.protocolSection.statusModule.startDateStruct.date' AS start_date,
               raw_json->>'$.protocolSection.statusModule.startDateStruct.type' AS start_date_type,
               raw_json->>'$.protocolSection.statusModule.primaryCompletionDateStruct.date' AS primary_completion_date,
               raw_json->>'$.protocolSection.statusModule.completionDateStruct.date' AS completion_date,
               raw_json->>'$.protocolSection.statusModule.studyFirstPostDateStruct.date' AS first_posted,
               last_change_date,
               TRY_CAST(raw_json->>'$.protocolSection.designModule.enrollmentInfo.count' AS INTEGER) AS enrollment,
               CAST(json_extract(raw_json, '$.protocolSection.conditionsModule.conditions') AS VARCHAR[]) AS conditions,
               json_extract_string(raw_json, '$.protocolSection.armsInterventionsModule.interventions[*].name') AS interventions,
               coalesce(TRY_CAST(raw_json->>'$.hasResults' AS BOOLEAN), FALSE) AS has_results,
               raw_json->>'$.protocolSection.sponsorCollaboratorsModule.leadSponsor.name' AS lead_sponsor,
               raw_json->>'$.protocolSection.sponsorCollaboratorsModule.leadSponsor.class' AS lead_class,
               json_extract_string(raw_json, '$.protocolSection.sponsorCollaboratorsModule.collaborators[*].name') AS collab_names,
               json_extract_string(raw_json, '$.protocolSection.sponsorCollaboratorsModule.collaborators[*].class') AS collab_classes
        FROM (SELECT p.nct_id, p.version, p.last_change_date, CAST(p.raw_json AS JSON) AS raw_json
              FROM read_parquet('{pattern}') p JOIN _latest l USING (nct_id, version)
              WHERE p.nct_id IN (SELECT unnest(?)))"""
    parts = []
    for i in range(0, len(ids), BUILD_BATCH):
        parts.append(rcon.execute(fields.format(pattern=_pattern()), [ids[i:i + BUILD_BATCH]]).df())
    trials = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    rcon.execute('DROP TABLE IF EXISTS _latest')

    def to_date(s):
        return pd.to_datetime(s.map(lambda d: d if not isinstance(d, str) or len(d) > 7 else f'{d}-01'),
                              errors='coerce').dt.date
    for c in ('start_date', 'primary_completion_date', 'completion_date', 'first_posted'):
        trials[c] = to_date(trials[c])
    trials['phase_groups'] = trials['phases'].map(phase_groups)
    year = pd.to_datetime(trials['start_date']).dt.year.fillna(pd.to_datetime(trials['first_posted']).dt.year)
    trials['start_year'] = year.astype('Int64')
    trials['interventions'] = trials['interventions'].map(lambda v: list(dict.fromkeys(as_list(v))))
    trials['conditions'] = trials['conditions'].map(as_list)
    trials['phases'] = trials['phases'].map(as_list)

    # sponsors: lead + collaborators
    sp = []
    for t in trials.itertuples(index=False):
        if t.lead_sponsor:
            sp.append((t.nct_id, t.lead_sponsor, 'lead', t.lead_class))
        for n, c in zip(as_list(t.collab_names), as_list(t.collab_classes)):
            sp.append((t.nct_id, n, 'collaborator', c))
    sponsors = pd.DataFrame(sp, columns=['nct_id', 'sponsor_name', 'role', 'sponsor_class']) \
        .drop_duplicates(['nct_id', 'sponsor_name'])

    ccon = common.connect('companies')
    try:
        exact, prefixes = _name_index(ccon)
        keys = sponsors[['sponsor_name', 'sponsor_class']].drop_duplicates()
        matched = {(n, c): match_sponsor(n, exact, prefixes, c) for n, c in keys.itertuples(index=False)}
        res = [matched[(n, c)] for n, c in zip(sponsors['sponsor_name'], sponsors['sponsor_class'])]
        sponsors['corporate_id'] = pd.array([r[0] for r in res], dtype='Int64')
        sponsors['match_method'] = [r[1] for r in res]

        cols = ['nct_id', 'version', 'title', 'overall_status', 'study_type', 'phases', 'phase_groups', 'start_date',
                'start_date_type', 'start_year', 'primary_completion_date', 'completion_date', 'first_posted',
                'last_change_date', 'enrollment', 'conditions', 'interventions', 'has_results', 'lead_sponsor']
        ccon.execute('BEGIN')
        ccon.execute('DELETE FROM clinical_trials')
        ccon.execute('DELETE FROM clinical_trial_sponsors')
        common.upsert(ccon, 'clinical_trials', trials[cols])
        common.upsert(ccon, 'clinical_trial_sponsors', sponsors)
        ccon.execute('COMMIT')
    finally:
        ccon.close()

    ind = sponsors[sponsors['sponsor_class'] == 'INDUSTRY']
    unmatched = (ind[ind['corporate_id'].isna()].groupby(['sponsor_name', 'match_method'], dropna=False)
                 .size().rename('trials').reset_index().sort_values('trials', ascending=False))
    unmatched.to_csv(UNMATCHED_CSV, index=False)
    m = sponsors['corporate_id'].notna()
    log.info(f"{len(trials):,} trials, {len(sponsors):,} sponsor rows: {int(m.sum()):,} matched to "
             f"{sponsors.loc[m, 'corporate_id'].nunique()} companies "
             f"({', '.join(f'{k}: {v}' for k, v in sponsors.loc[m, 'match_method'].value_counts().items())}); "
             f"{len(unmatched)} unmatched industry sponsors -> {UNMATCHED_CSV.name}")


def _name_index(ccon) -> tuple[dict[str, set], dict[str, set]]:
    """
    normalised name -> {(corporate_id, method)} from aliases, company names and Exhibit 21 subsidiaries;
    normalised company name -> {corporate_id} for the prefix rule (names of >= 5 letters).
    """
    comp = ccon.execute("""SELECT corporate_id, primary_ticker, name, legal_name FROM company_info
                           WHERE coalesce(status, 'active') = 'active'""").df()
    exact: dict[str, set] = {}
    prefixes: dict[str, set] = {}
    by_ticker = dict(zip(comp['primary_ticker'], comp['corporate_id']))
    name_owner: dict[str, set] = {}
    for c in comp.itertuples(index=False):
        for n in {norm_name(c.name), norm_name(c.legal_name)} - {''}:
            name_owner.setdefault(n, set()).add(int(c.corporate_id))
    for ticker in set(config.TRIAL_SPONSOR_ALIASES) | set(config.TRIAL_SPONSOR_MATCH):
        if ticker not in by_ticker:
            continue
        cid = int(by_ticker[ticker])
        for a in config.TRIAL_SPONSOR_ALIASES.get(ticker, []) + config.TRIAL_SPONSOR_MATCH.get(ticker, []):
            n = norm_name(a)
            if not n or name_owner.get(n, {cid}) != {cid}:   # 'Merck KGaA' -> 'merck' is also Merck & Co.
                continue
            exact.setdefault(n, set()).add((cid, 'alias'))
            if len(n.replace(' ', '')) >= 4:                # aliases are chosen by hand: 'CSPC' is enough
                prefixes.setdefault(n, set()).add(cid)
    for c in comp.itertuples(index=False):
        for n in {norm_name(c.name), norm_name(c.legal_name)} - {''}:
            exact.setdefault(n, set()).add((int(c.corporate_id), 'name'))
            if len(n.replace(' ', '')) >= 5:
                prefixes.setdefault(n, set()).add(int(c.corporate_id))
    subs = ccon.execute("""
        SELECT DISTINCT corporate_id, s.name FROM (
            SELECT corporate_id, unnest(subsidiaries) AS s FROM fundamentals_yearly
            WHERE subsidiaries IS NOT NULL
            QUALIFY fiscal_year = max(fiscal_year) OVER (PARTITION BY corporate_id))""").fetchall()
    for cid, name in subs:
        n = norm_name(name)
        if len(n.replace(' ', '')) >= 5:
            exact.setdefault(n, set()).add((int(cid), 'subsidiary'))
    # an alias decides over other hits of the same name
    for n, hits in exact.items():
        if any(m == 'alias' for _, m in hits):
            exact[n] = {h for h in hits if h[1] == 'alias'}
    return exact, prefixes


def yearly_phase_counts(ccon, corporate_id: int) -> pd.DataFrame:
    """
    start_year, phase (1-4), trials: trials of a company per start year and phase. A trial counts in each of
    its phases (Phase 1/2 in 1 and 2) and for every matched company (lead sponsor or collaborator).
    """
    return ccon.execute("""
        SELECT start_year, phase, count(DISTINCT nct_id) AS trials
        FROM (SELECT nct_id, start_year, unnest(phase_groups) AS phase FROM clinical_trials
              WHERE nct_id IN (SELECT nct_id FROM clinical_trial_sponsors WHERE corporate_id = ?)
                AND start_year IS NOT NULL)
        GROUP BY ALL ORDER BY 1, 2""", [corporate_id]).df()
