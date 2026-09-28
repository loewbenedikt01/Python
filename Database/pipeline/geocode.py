"""
Headquarters coordinates (company_info.hq_lat / hq_lon / hq_geo_source) from
the HQ address, run by the companies step for companies whose address changed
(hq_geo_key = city|state|country when geocoded).

Offline first: GeoNames cities1000 (all places with more than 1,000
inhabitants; downloaded once to data/cache/geonames/, no key). In the
company's HQ country (config.COUNTRIES -> ISO code):
1. the city equals a place's name, ASCII name or alternate name (case, accents,
   punctuation ignored; ø -> o, æ -> ae, ß -> ss); also its comma parts and
   without district words ('Jakarta Selatan' -> 'Jakarta', 'CO. DUBLIN');
   several places with that name -> the one in the HQ state (US), else the
   largest -> 'city';
2. a place whose name starts with it ('Weinstadt' -> 'Weinstadt-Endersbach'),
   then the single words of the name ('Rapperswil-Jona' -> 'Rapperswil') -> 'city'.
Then:
3. anywhere in the world, places with > 100,000 inhabitants only: US-listed
   companies based abroad have HQ country 'United States' and a foreign city
   ('LONDON ENGLAND', 'ZURICH', 'DUBLIN 2') -> 'city_world';
4. OpenStreetMap Nominatim (street, city, state, country; then city, country;
   max. 1 request per second, only the few companies left) -> 'nominatim';
5. the country's largest city -> 'country_fallback'.
Everything not matched in step 1-2 -> data/review/hq_geocode_unmatched.csv.
"""

from __future__ import annotations

import io
import re
import unicodedata
import zipfile

import pandas as pd

import config
from pipeline import common


log = common.get_logger('pipeline.geocode')

GEONAMES_DIR = config.CACHE_DIR / 'geonames'
CITIES_URL = 'https://download.geonames.org/export/dump/cities1000.zip'
NOMINATIM_URL = 'https://nominatim.openstreetmap.org/search'
UNMATCHED_CSV = config.REVIEW_DIR / 'hq_geocode_unmatched.csv'
WORLD_MIN_POPULATION = 100_000
DROP_WORDS = {'cedex', 'ku', 'shi', 'city', 'district', 'gun', 'si', 'dong', 'town', 'municipality', 'co',
              'selatan', 'pusat', 'utara', 'barat', 'timur', 'england', 'surrey'}
TRANSLIT = str.maketrans({'ø': 'o', 'Ø': 'O', 'æ': 'ae', 'Æ': 'AE', 'ß': 'ss', 'đ': 'd', 'Đ': 'D', 'ł': 'l',
                          'Ł': 'L', 'œ': 'oe', 'Œ': 'OE', 'þ': 'th', 'ı': 'i'})


# ----
# MATCHING (pure functions)
# ----

def plain(text) -> str:
    """
    Lower-case ASCII, punctuation -> spaces, apostrophes dropped: "Marcy-l'Étoile" -> 'marcy letoile'.
    """
    s = str(text or '').translate(TRANSLIT)
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r"['’`]", '', s)
    return re.sub(r'[^a-z0-9]+', ' ', s).strip()


def variants(city: str) -> list[str]:
    """
    Names to try for a city, most specific first: the whole name, its comma parts, without district words.
    """
    key = plain(city)
    parts = [plain(p) for p in re.split(r'[,/()]', str(city)) if plain(p)]
    out = [key] + parts
    out += [' '.join(w for w in v.split() if w not in DROP_WORDS) for v in [key] + parts]
    return list(dict.fromkeys(v for v in out if v))


def _best(cands: list, state: str | None):
    if state:
        s = [c for c in cands if str(c[1]).upper() == str(state).upper()]
        if s:
            return max(s, key=lambda c: c[0])
    return max(cands, key=lambda c: c[0])


def match(city: str | None, state: str | None, iso: str | None, idx: dict,
          names_by_country: dict[str, list]) -> tuple[float | None, float | None, str | None]:
    """
    (lat, lon, 'city') for one HQ in its country, or (None, None, None).
    idx: (country, plain name) -> [(population, admin1, lat, lon)]; names_by_country: country -> [plain names].
    """
    if not iso or not city:
        return None, None, None
    vs = variants(city)
    for v in vs:                                             # 1. exact name
        if (iso, v) in idx:
            b = _best(idx[(iso, v)], state)
            return b[2], b[3], 'city'
    names = names_by_country.get(iso, [])
    for v in vs:                                             # 2a. a place whose name starts with it
        cands = [c for n in names if n.startswith(v + ' ') for c in idx[(iso, n)]]
        if cands:
            b = _best(cands, state)
            return b[2], b[3], 'city'
    for v in vs:                                             # 2b. single words of the name
        for w in v.split():
            if len(w) >= 4 and w not in DROP_WORDS and (iso, w) in idx:
                b = _best(idx[(iso, w)], state)
                return b[2], b[3], 'city'
    return None, None, None


def match_world(city: str | None, world: dict) -> tuple[float | None, float | None, str | None]:
    """
    The city anywhere: world = plain name -> (population, lat, lon) of the largest place (> 100,000 inhabitants).
    """
    for v in variants(city or ''):
        if v in world:
            return world[v][1], world[v][2], 'city_world'
    return None, None, None


# ----
# DATA
# ----

def load_places() -> pd.DataFrame:
    """
    GeoNames places: name, asciiname, alternatenames, lat, lon, country, admin1, population.
    """
    GEONAMES_DIR.mkdir(parents=True, exist_ok=True)
    path = GEONAMES_DIR / 'cities1000.zip'
    if not path.exists():
        log.info('downloading GeoNames cities1000')
        path.write_bytes(common.http_get(CITIES_URL, 'geonames').content)
    with zipfile.ZipFile(path) as z:
        raw = z.read('cities1000.txt').decode('utf-8')
    cols = ['geonameid', 'name', 'asciiname', 'alternatenames', 'lat', 'lon', 'fclass', 'fcode', 'country',
            'cc2', 'admin1', 'admin2', 'admin3', 'admin4', 'population', 'elevation', 'dem', 'tz', 'modified']
    df = pd.read_csv(io.StringIO(raw), sep='\t', header=None, names=cols, dtype=str, quoting=3,
                     usecols=['name', 'asciiname', 'alternatenames', 'lat', 'lon', 'country', 'admin1', 'population'])
    df['lat'] = df['lat'].astype(float)
    df['lon'] = df['lon'].astype(float)
    df['population'] = pd.to_numeric(df['population'], errors='coerce').fillna(0)
    return df


def build_indexes(places: pd.DataFrame) -> tuple[dict, dict, dict]:
    """
    idx: (country, plain name) -> [(population, admin1, lat, lon)] over names, ASCII and alternate names;
    names_by_country: country -> [plain names]; world: plain name -> (population, lat, lon) of places
    with > WORLD_MIN_POPULATION inhabitants (largest per name).
    """
    idx: dict[tuple, list] = {}
    world: dict[str, tuple] = {}
    for r in places.itertuples(index=False):
        names = {r.name, r.asciiname}
        if isinstance(r.alternatenames, str):
            names |= set(r.alternatenames.split(','))
        for n in names:
            k = plain(n)
            if not k:
                continue
            idx.setdefault((r.country, k), []).append((r.population, r.admin1, r.lat, r.lon))
            if r.population > WORLD_MIN_POPULATION and n in (r.name, r.asciiname) \
                    and (k not in world or world[k][0] < r.population):
                world[k] = (r.population, r.lat, r.lon)
    names_by_country: dict[str, list] = {}
    for cc, n in idx:
        names_by_country.setdefault(cc, []).append(n)
    return idx, names_by_country, world


def nominatim(street, city, state, country) -> tuple[float | None, float | None]:
    """
    OpenStreetMap Nominatim: the full address, then city + country. (None, None) if nothing is found.
    """
    ua = f"market-data-pipeline ({common.load_keys().get('EMAIL', 'no email')})"
    for q in (', '.join(x for x in (street, city, state, country) if isinstance(x, str) and x),
              ', '.join(x for x in (city, country) if isinstance(x, str) and x)):
        if not q:
            continue
        try:
            r = common.http_get(NOMINATIM_URL, 'nominatim', params={'q': q, 'format': 'json', 'limit': 1},
                                headers={'User-Agent': ua}).json()
        except Exception as e:
            log.warning(f'nominatim {q!r}: {common.format_error(e)}')
            continue
        if r:
            return float(r[0]['lat']), float(r[0]['lon'])
    return None, None


# ----
# STEP PART (called by the companies step)
# ----

def geocode(ccon, force: bool = False) -> dict:
    """
    Fill hq_lat / hq_lon / hq_geo_source / hq_geo_key for companies whose address changed (all if force).
    """
    comp = ccon.execute("""
        SELECT corporate_id, primary_ticker, name, hq_street, hq_city, hq_state, hq_country, country, hq_geo_key
        FROM company_info WHERE coalesce(status, 'active') = 'active'""").df()
    comp['key'] = comp['hq_city'].fillna('') + '|' + comp['hq_state'].fillna('') + '|' + \
        comp['hq_country'].fillna(comp['country']).fillna('')
    todo = comp if force else comp[comp['key'] != comp['hq_geo_key'].fillna('')]
    if todo.empty:
        return {'geocoded': 0, 'by_city': 0}
    places = load_places()
    idx, names_by_country, world = build_indexes(places)
    largest = places.sort_values('population', ascending=False).drop_duplicates('country').set_index('country')

    rows, review = [], []
    for r in todo.itertuples(index=False):
        country = r.hq_country if isinstance(r.hq_country, str) else r.country
        iso = config.COUNTRIES.get(country, (None,))[0] if isinstance(country, str) else None
        lat, lon, src = match(r.hq_city, r.hq_state, iso, idx, names_by_country)
        if lat is None:
            lat, lon, src = match_world(r.hq_city, world)
        if lat is None and isinstance(r.hq_city, str):
            lat, lon = nominatim(r.hq_street, r.hq_city, r.hq_state, country)
            src = 'nominatim' if lat is not None else None
        if lat is None and iso in largest.index:
            lat, lon, src = float(largest.at[iso, 'lat']), float(largest.at[iso, 'lon']), 'country_fallback'
        if src != 'city':
            review.append({'primary_ticker': r.primary_ticker, 'name': r.name, 'hq_street': r.hq_street,
                           'hq_city': r.hq_city, 'hq_state': r.hq_state, 'hq_country': country, 'result': src,
                           'lat': lat, 'lon': lon})
        rows.append({'corporate_id': int(r.corporate_id), 'hq_lat': lat, 'hq_lon': lon, 'hq_geo_source': src,
                     'hq_geo_key': r.key})
    common.upsert(ccon, 'company_info', pd.DataFrame(rows))
    if force or review:
        pd.DataFrame(review, columns=['primary_ticker', 'name', 'hq_street', 'hq_city', 'hq_state', 'hq_country',
                                      'result', 'lat', 'lon']).to_csv(UNMATCHED_CSV, index=False)
    counts = pd.Series([r['hq_geo_source'] for r in rows]).value_counts(dropna=False).to_dict()
    log.info(f'HQ coordinates for {len(rows)} companies: {counts} (not by city -> {UNMATCHED_CSV.name})')
    return {'geocoded': len(rows), 'by_city': counts.get('city', 0)}
