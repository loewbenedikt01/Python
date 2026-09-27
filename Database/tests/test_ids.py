"""
corporate_id rules: stable across runs, merges keep the Class A ID, retired
IDs are never reused. Uses made-up tickers/CIKs, no network.
"""

import random

import duckdb
import pandas as pd

from pipeline import common
from pipeline.ids import assign_ids, load_registry, save_registry, ticker_map, REGISTRY_COLS

EMPTY = pd.DataFrame(columns=REGISTRY_COLS)


def equities(rows):
    return pd.DataFrame(rows, columns=['ticker', 'name', 'group_name'])


BASE = equities([
    ('AAA',   'Alpha Inc. Class A', 'ticker_us'),
    ('AAA.C', 'Alpha Inc. Class C', 'ticker_us'),
    ('BBB',   'Beta Corp',          'ticker_us'),
    ('X.DE',  'Gamma AG',           'ticker_de'),
    ('X2.DE', 'Gamma AG',           'ticker_de'),     # second listing, same name
])
CIKS  = {'AAA': 1, 'AAA.C': 1, 'BBB': 2}
RANKS = {'AAA': 0, 'AAA.C': 1, 'BBB': 2}
OLD   = {'AAA': 11111, 'AAA.C': 22222, 'BBB': 33333, 'X.DE': 44444, 'X2.DE': 44444}


def test_first_run_keeps_old_ids_and_merges_share_classes():
    reg, counts = assign_ids(BASE, CIKS, RANKS, EMPTY, OLD, rng=random.Random(0))
    ids = ticker_map(reg)
    assert ids == {'AAA': 11111, 'AAA.C': 11111, 'BBB': 33333, 'X.DE': 44444, 'X2.DE': 44444}
    retired = reg.set_index('corporate_id').loc[22222]
    assert retired['merged_into'] == 11111 and retired['tickers'] == ''
    alpha = reg.set_index('corporate_id').loc[11111]
    assert alpha['tickers'] == 'AAA;AAA.C' and alpha['company_name'] == 'Alpha Inc.' and alpha['cik'] == 1
    assert counts == {'new': 0, 'kept': 3, 'merged': 1, 'split': 0}
    assert set(reg['status']) == {'active', 'merged'}


def test_rerun_never_changes_ids(tmp_path):
    reg1, _ = assign_ids(BASE, CIKS, RANKS, EMPTY, OLD, rng=random.Random(0))
    path = tmp_path / 'ids.csv'
    save_registry(reg1, path)
    for seed in range(3):                       # different random state must not matter
        reg2, counts = assign_ids(BASE, CIKS, RANKS, load_registry(path), {}, rng=random.Random(seed))
        assert ticker_map(reg2) == ticker_map(reg1)
        assert counts['new'] == 0 and counts['merged'] == 0
        save_registry(reg2, path)
    assert load_registry(path).sort_values('corporate_id').reset_index(drop=True).equals(
        reg1.sort_values('corporate_id').reset_index(drop=True))


def test_new_company_gets_unused_id_and_retired_ids_are_not_reused():
    reg, _ = assign_ids(BASE, CIKS, RANKS, EMPTY, OLD)
    taken = set(reg['corporate_id'])
    more = pd.concat([BASE, equities([(f'N{i}', f'New {i}', 'ticker_us') for i in range(200)])])
    reg2, counts = assign_ids(more, CIKS, RANKS, reg, {}, rng=random.Random(1))
    new_ids = set(reg2['corporate_id']) - taken
    assert counts['new'] == 200 and len(new_ids) == 200
    assert all(10000 <= i <= 99999 for i in new_ids)
    assert 22222 not in new_ids                 # retired by the merge


def test_new_share_class_joins_existing_company():
    reg, _ = assign_ids(BASE, CIKS, RANKS, EMPTY, OLD)
    more = pd.concat([BASE, equities([('AAA.B', 'Alpha Inc. Class B', 'ticker_us')])])
    reg2, counts = assign_ids(more, {**CIKS, 'AAA.B': 1}, RANKS, reg, {})
    assert ticker_map(reg2)['AAA.B'] == 11111 and counts['new'] == 0


def test_removed_ticker_keeps_its_company_row():
    reg, _ = assign_ids(BASE, CIKS, RANKS, EMPTY, OLD)
    reg2, _ = assign_ids(BASE[BASE['ticker'] != 'BBB'], CIKS, RANKS, reg, {})
    assert ticker_map(reg2)['BBB'] == 33333


def test_upsert_updates_only_given_columns():
    con = duckdb.connect()
    con.execute('CREATE TABLE t (k VARCHAR PRIMARY KEY, a INTEGER, b INTEGER)')
    common.upsert(con, 't', pd.DataFrame({'k': ['x'], 'a': [1], 'b': [2]}))
    common.upsert(con, 't', pd.DataFrame({'k': ['x', 'y', 'y'], 'a': [10, 5, 6]}))
    assert con.execute('SELECT * FROM t ORDER BY k').fetchall() == [('x', 10, 2), ('y', 6, None)]


def test_wrongly_grouped_ticker_is_split_off():
    """
    X2.DE was listed under the same name as X.DE; after the name is corrected
    it gets a new ID, X.DE keeps 44444.
    """
    reg, _ = assign_ids(BASE, CIKS, RANKS, EMPTY, OLD)
    fixed = BASE.copy()
    fixed.loc[fixed['ticker'] == 'X2.DE', 'name'] = 'Other AG'
    reg2, counts = assign_ids(fixed, CIKS, RANKS, reg, {}, rng=random.Random(3))
    ids = ticker_map(reg2)
    assert ids['X.DE'] == 44444 and ids['X2.DE'] not in (44444, 11111, 22222, 33333)
    assert counts['split'] == 1 and counts['new'] == 1
    assert reg2.set_index('corporate_id').loc[44444, 'tickers'] == 'X.DE'


def test_ticker_changes_from_corrections_list():
    """
    rename keeps the ID; 'a' (different company) gets a new ID and the old one is reassigned;
    'replaced by' hands the ID to the new ticker; a removed ticker's company becomes 'removed'.
    """
    from pipeline.ids import apply_ticker_changes
    base = equities([('R.DE', 'Old Name AG', 'ticker_de'), ('W.DE', 'Wrong AG', 'ticker_de'),
                     ('OLD.HK', 'Same Co', 'ticker_asia'), ('GONE.DE', 'Gone AG', 'ticker_de')])
    old = {'R.DE': 50001, 'W.DE': 50002, 'OLD.HK': 50003, 'GONE.DE': 50004}
    reg, _ = assign_ids(base, {}, {}, EMPTY, old)
    changes = pd.DataFrame({'ticker': ['R.DE', 'W.DE', 'OLD.HK', 'GONE.DE'],
                            'change': ['rename', 'a', 'replaced by NEW.SS', 'removed']})
    fixed = equities([('R.DE', 'New Name AG', 'ticker_de'), ('W.DE', 'Right AG', 'ticker_de'),
                      ('NEW.SS', 'Same Co', 'ticker_asia')])
    reg2, counts = apply_ticker_changes(reg, changes)
    reg3, _ = assign_ids(fixed, {}, {}, reg2, {}, rng=random.Random(5))
    ids = ticker_map(reg3[reg3['status'] != 'reassigned'])
    by_id = reg3.set_index('corporate_id')
    assert ids['R.DE'] == 50001 and by_id.loc[50001, 'company_name'] == 'New Name AG'
    assert ids['W.DE'] not in old.values() and by_id.loc[50002, 'status'] == 'reassigned'
    assert ids['NEW.SS'] == 50003
    assert by_id.loc[50004, 'status'] == 'removed'
    assert counts == {'reassigned_tickers': 1, 'replaced': 1, 'retired_ids': 1}
