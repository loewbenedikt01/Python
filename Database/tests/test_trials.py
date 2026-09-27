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


# ----
# names, phases, budget, sponsor matching
# ----

from pipeline.trials import allocate, match_sponsor, norm_name, phase_groups, search_term


def test_norm_name():
    assert norm_name('Merck Sharp & Dohme LLC') == 'merck sharp dohme'
    assert norm_name("Dr. Reddy's Laboratories Limited") == 'dr reddys laboratories'
    assert norm_name('Novo Nordisk A/S') == 'novo nordisk'
    assert norm_name('WuXi Biologics (Cayman) Inc.') == 'wuxi biologics'
    assert norm_name('BioMérieux S.A.') == 'biomerieux'


def test_search_term_keeps_spelling_without_legal_form():
    assert search_term('Merck & Co., Inc.') == 'Merck'
    assert search_term("Dr. Reddy's Laboratories Limited") == "Dr. Reddy's Laboratories"
    assert search_term('Johnson & Johnson') == 'Johnson & Johnson'
    assert search_term('Bangkok Dusit Medical Services Public Company Limited') == 'Bangkok Dusit Medical Services'


def test_phase_groups_combined_phases_count_in_both():
    assert phase_groups(['PHASE1', 'PHASE2']) == [1, 2]
    assert phase_groups(['PHASE2', 'PHASE3']) == [2, 3]
    assert phase_groups(['EARLY_PHASE1']) == [1]
    assert phase_groups(['PHASE4']) == [4]
    assert phase_groups(['NA']) == [] and phase_groups(None) == []


def test_allocate_budget_small_searches_get_all():
    out = allocate({'big1': 6000, 'big2': 5000, 'small': 90, 'none': 0}, 10_000)
    assert out['small'] == 90 and out['none'] == 0
    assert out['big1'] + out['big2'] == 10_000 - 90 and abs(out['big1'] - out['big2']) <= 1
    assert allocate({'a': 10, 'b': 20}, 1000) == {'a': 10, 'b': 20}


def test_match_sponsor_exact_prefix_ambiguous():
    exact = {'merck sharp dohme': {(1, 'alias')}, 'pfizer': {(2, 'name')},
             'janssen research development': {(3, 'subsidiary')}, 'merck': {(1, 'name'), (4, 'name')}}
    prefixes = {'pfizer': {2}, 'janssen': {3}, 'merck': {1, 4}}
    assert match_sponsor('Merck Sharp & Dohme LLC', exact, prefixes) == (1, 'alias')
    assert match_sponsor('Pfizer', exact, prefixes) == (2, 'name')
    assert match_sponsor('Janssen Research & Development, LLC', exact, prefixes) == (3, 'subsidiary')
    assert match_sponsor('Pfizer Pharmaceuticals Ltd', exact, prefixes) == (2, 'name_prefix')
    assert match_sponsor('Merck KGaA, Darmstadt, Germany', exact, prefixes) == (None, 'ambiguous')
    assert match_sponsor('University of Oxford', exact, prefixes) == (None, None)


def test_match_sponsor_parent_and_foundations():
    exact = {'gsk': {(5, 'name')}, 'pfizer': {(2, 'name')}, 'merck darmstadt germany': {(4, 'alias')}}
    prefixes = {'pfizer': {2}, 'resmed': {6}}
    assert match_sponsor('Stiefel, a GSK Company', exact, prefixes) == (5, 'parent')
    assert match_sponsor('Sierra Oncology LLC - a GSK company', exact, prefixes) == (5, 'parent')
    assert match_sponsor('Wyeth is now a wholly owned subsidiary of Pfizer', exact, prefixes) == (2, 'parent')
    assert match_sponsor('Merck Healthcare KGaA, Darmstadt, Germany, an affiliate of Merck KGaA, Darmstadt, Germany',
                         exact, prefixes) == (4, 'parent')
    assert match_sponsor('ResMed Foundation', exact, prefixes, 'OTHER') == (None, None)
    assert match_sponsor('ResMed Foundation', exact, prefixes, 'INDUSTRY') == (None, None)       # foundation
    assert match_sponsor('Pfizer University Hospital', exact, prefixes, 'OTHER') == (None, None)
