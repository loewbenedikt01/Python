"""
Headquarters geocoding: name matching against a small GeoNames-like index. No network.
"""

import pandas as pd

from pipeline.geocode import build_indexes, match, match_world, plain, variants

PLACES = pd.DataFrame([
    # name, asciiname, alternatenames, lat, lon, country, admin1, population
    ('Springfield', 'Springfield', '', 39.80, -89.64, 'US', 'IL', 114_000),
    ('Springfield', 'Springfield', '', 42.10, -72.59, 'US', 'MA', 155_000),
    ('Bagsværd', 'Bagsvaerd', '', 55.76, 12.45, 'DK', '17', 20_000),
    ('Weinstadt-Endersbach', 'Weinstadt-Endersbach', '', 48.81, 9.37, 'DE', '01', 26_000),
    ('Rapperswil', 'Rapperswil', '', 47.23, 8.82, 'CH', 'SG', 35_000),
    ('Jakarta', 'Jakarta', 'Djakarta', -6.21, 106.85, 'ID', '04', 8_500_000),
    ('London', 'London', '', 51.51, -0.13, 'GB', 'ENG', 8_900_000),
    ('A Coruña', 'A Coruna', 'Corunna,La Coruna', 43.37, -8.40, 'ES', '58', 245_000),
], columns=['name', 'asciiname', 'alternatenames', 'lat', 'lon', 'country', 'admin1', 'population'])
IDX, NAMES, WORLD = build_indexes(PLACES)


def test_plain_and_variants():
    assert plain("Marcy-l'Étoile") == 'marcy letoile'
    assert plain('Bagsværd') == 'bagsvaerd' and plain('Søborg') == 'soborg'
    assert 'jakarta' in variants('Jakarta Selatan')
    assert 'dublin' in variants('CO. DUBLIN') and 'london' in variants('LONDON ENGLAND')


def test_state_decides_between_same_names():
    assert match('SPRINGFIELD', 'MA', 'US', IDX, NAMES)[:2] == (42.10, -72.59)
    assert match('Springfield', None, 'US', IDX, NAMES)[:2] == (42.10, -72.59)       # largest without state


def test_translit_prefix_words_alternate_names():
    assert match('Bagsvaerd', None, 'DK', IDX, NAMES)[2] == 'city'
    assert match('Weinstadt', None, 'DE', IDX, NAMES)[:2] == (48.81, 9.37)             # 'Weinstadt-Endersbach'
    assert match('Rapperswil-Jona', None, 'CH', IDX, NAMES)[:2] == (47.23, 8.82)       # single word
    assert match('Corunna', None, 'ES', IDX, NAMES)[:2] == (43.37, -8.40)              # alternate name
    assert match('Jakarta Selatan', None, 'ID', IDX, NAMES)[:2] == (-6.21, 106.85)     # district word dropped


def test_foreign_city_of_us_listing_matched_worldwide():
    assert match('LONDON ENGLAND', 'X0', 'US', IDX, NAMES) == (None, None, None)
    assert match_world('LONDON ENGLAND', WORLD) == (51.51, -0.13, 'city_world')
    assert match_world('Rapperswil', WORLD) == (None, None, None)                      # < 100,000 inhabitants
