"""
Clinical-trial versioning: hash, changed sections, version numbers. No network.
"""

import copy
import json
from datetime import date, datetime

from pipeline.trials import changed_sections, json_hash, last_change_date, new_version

NOW = datetime(2026, 9, 27, 12, 0)


def study(status='RECRUITING', updated='2026-03-25', snapshot='2026-09-25', enrollment=100):
    return {
        'protocolSection': {
            'identificationModule': {'nctId': 'NCT00000001', 'briefTitle': 'A trial'},
            'statusModule': {'overallStatus': status, 'lastUpdatePostDateStruct': {'date': updated, 'type': 'ACTUAL'}},
            'designModule': {'phases': ['PHASE3'], 'enrollmentInfo': {'count': enrollment}},
        },
        'derivedSection': {'miscInfoModule': {'versionHolder': snapshot}},
        'hasResults': False,
    }


def stored(row):
    return {'version': row['version'], 'json_hash': row['json_hash'], 'raw_json': row['raw_json']}


def test_hash_ignores_key_order_and_snapshot_date():
    a = study()
    b = json.loads(json.dumps(a, sort_keys=True))                   # same content, other key order
    b['derivedSection']['miscInfoModule']['versionHolder'] = '2026-09-27'
    assert json_hash(a) == json_hash(b)
    assert json_hash(a) != json_hash(study(status='COMPLETED'))


def test_first_version():
    row = new_version(None, study(), NOW)
    assert (row['version'], row['change_flag'], row['changed_sections'], row['is_latest']) == (1, False, [], True)
    assert row['last_change_date'] == date(2026, 3, 25) and row['nct_id'] == 'NCT00000001'


def test_identical_content_is_not_a_new_version():
    v1 = new_version(None, study(), NOW)
    assert new_version(stored(v1), study(snapshot='2026-09-27'), NOW) is None


def test_version_change_records_changed_modules():
    v1 = new_version(None, study(), NOW)
    changed = study(status='ACTIVE_NOT_RECRUITING', updated='2026-09-20', enrollment=120)
    changed['hasResults'] = True
    v2 = new_version(stored(v1), changed, NOW)
    assert v2['version'] == 2 and v2['change_flag'] is True
    assert v2['changed_sections'] == ['designModule', 'hasResults', 'statusModule']
    assert v2['last_change_date'] == date(2026, 9, 20)
    v3 = new_version(stored(v2), copy.deepcopy(changed), NOW)
    assert v3 is None


def test_new_section_counts_its_modules():
    old = study()
    new = copy.deepcopy(old)
    new['resultsSection'] = {'participantFlowModule': {'groups': []}, 'adverseEventsModule': {}}
    assert changed_sections(old, new) == ['adverseEventsModule', 'participantFlowModule']


def test_last_change_date_month_only():
    assert last_change_date(study(updated='2021-11')) == date(2021, 11, 1)
