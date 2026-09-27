"""
Step 'trials': ClinicalTrials.gov API v2 -> raw.duckdb clinical_trials_raw,
one row per trial and version (PK nct_id, version).

- Which trials: the stored ones, plus config.TRIAL_NCT_IDS, plus the trials of
  config.TRIAL_SPONSORS (lead sponsor; both lists empty by default), plus
  `run.py trials --nct NCT...`.
- Incremental: stored trials and known sponsors are only requested when their
  lastUpdatePostDate is on or after the latest stored last_change_date. New
  NCT IDs and new sponsors (no stored trial with that lead sponsor) are
  requested in full.
- last_change_date = protocolSection.statusModule.lastUpdatePostDateStruct.date.
- Versioning: json_hash = sha256 of the normalised JSON (sorted keys, no
  whitespace) without derivedSection.miscInfoModule.versionHolder (the date of
  the ClinicalTrials.gov snapshot, it changes every day for every trial). The
  same hash as the latest version -> nothing is written. A different hash ->
  a new version (old versions are kept, only is_latest moves), change_flag =
  TRUE, changed_sections = the modules that differ (e.g. statusModule,
  designModule; 'hasResults' for the top-level flag). Version 1 has
  change_flag = FALSE and no changed_sections.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime

import pandas as pd

import config
from pipeline import common


log = common.get_logger('pipeline.trials')

API = 'https://clinicaltrials.gov/api/v2/studies'
PAGE_SIZE = 1000
IDS_PER_REQUEST = 200             # NCT IDs per filter.ids request (keeps the URL short)


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
# DOWNLOAD
# ----

def fetch(params: dict) -> list[dict]:
    """
    All studies for a query (follows nextPageToken).
    """
    out, token = [], None
    while True:
        p = {**params, 'pageSize': PAGE_SIZE, 'format': 'json'}
        if token:
            p['pageToken'] = token
        j = common.http_get(API, 'ctgov', params=p).json()
        out += j.get('studies', [])
        token = j.get('nextPageToken')
        if not token:
            return out


def since_filter(since) -> dict:
    return {'filter.advanced': f'AREA[LastUpdatePostDate]RANGE[{since:%Y-%m-%d},MAX]'} if since else {}


# ----
# STEP
# ----

def run(ctx: common.RunContext) -> common.StepResult:
    result = common.StepResult()
    now = datetime.now()
    con = common.connect('raw')
    try:
        latest = {r[0]: {'version': r[1], 'json_hash': r[2], 'raw_json': r[3]} for r in con.execute("""
            SELECT nct_id, version, json_hash, raw_json FROM clinical_trials_raw WHERE is_latest""").fetchall()}
        since = con.execute('SELECT max(last_change_date) FROM clinical_trials_raw').fetchone()[0]
        known_sponsors = {r[0] for r in con.execute("""
            SELECT DISTINCT raw_json->>'$.protocolSection.sponsorCollaboratorsModule.leadSponsor.name'
            FROM clinical_trials_raw WHERE is_latest""").fetchall()}
    finally:
        con.close()

    wanted = {i.upper() for i in [*config.TRIAL_NCT_IDS, *(ctx.nct_ids or [])]}
    new_ids = sorted(wanted - latest.keys())
    stored = sorted(latest)

    studies = {}

    def add(label, params):
        try:
            for s in fetch(params):
                studies[nct_id(s)] = s
        except Exception as e:
            result.fail(label, common.format_error(e), log)

    for i in range(0, len(stored), IDS_PER_REQUEST):          # stored trials: only the changed ones
        chunk = stored[i:i + IDS_PER_REQUEST]
        add(f'stored {chunk[0]}..', {'filter.ids': ','.join(chunk), **since_filter(since)})
    for i in range(0, len(new_ids), IDS_PER_REQUEST):         # new NCT IDs: in full
        chunk = new_ids[i:i + IDS_PER_REQUEST]
        add(f'new {chunk[0]}..', {'filter.ids': ','.join(chunk)})
        missing = set(chunk) - studies.keys()
        for m in sorted(missing):
            result.fail(m, 'not found on ClinicalTrials.gov', log)
    for sponsor in config.TRIAL_SPONSORS:                     # new sponsors in full, known ones incremental
        add(f'sponsor {sponsor}', {'query.spons': sponsor,
                                   **(since_filter(since) if sponsor in known_sponsors else {})})
    log.info(f'{len(stored)} stored trials, {len(new_ids)} new NCT IDs, {len(config.TRIAL_SPONSORS)} sponsors '
             f'-> {len(studies)} studies downloaded (changed since {since})')

    rows = []
    for nid, s in studies.items():
        prev = latest.get(nid)
        row = new_version(prev, s, now)
        if row is None:
            result.skipped += 1
        else:
            rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        con = common.connect('raw')
        try:
            con.execute('BEGIN')
            changed = df.loc[df['version'] > 1, 'nct_id'].tolist()
            con.execute('UPDATE clinical_trials_raw SET is_latest = FALSE WHERE is_latest AND nct_id IN (SELECT unnest(?))',
                        [changed])
            result.rows = common.upsert(con, 'clinical_trials_raw', df)
            con.execute('COMMIT')
        except Exception:
            con.execute('ROLLBACK')
            raise
        finally:
            con.close()
        n_changed = int((df['version'] > 1).sum())
        result.updated = len(df)
        log.info(f'{len(df) - n_changed} new trials, {n_changed} new versions of stored trials')
    return result
