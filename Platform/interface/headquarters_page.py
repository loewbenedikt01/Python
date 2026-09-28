"""
Headquarters page: a 3D globe (orthographic projection, drag to rotate) with a dot per company
headquarters, filters above it, and click-through to the company page. Coordinates come from the
pipeline (company_info.hq_lat / hq_lon, geocoded from the HQ city with GeoNames / OpenStreetMap).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import company_data as cd

COMPANY_PAGE = None            # set by interface.py (the st.Page to open on a click)

# categorical slots valid for dot maps (all pairs distinguishable, also with colour-vision deficiency):
# only three, everything else folds into a neutral "Other"
SLOTS = {'light': ['#2a78d6', '#eb6834', '#1baf7a'], 'dark': ['#3987e5', '#d95926', '#199e70']}
OTHER = '#898781'
THEME = {
    'light': dict(surface='#fcfcfb', land='#ecebe6', ocean='#f4f7fb', border='#c3c2b7', ink='#0b0b0b',
                  muted='#52514e'),
    'dark': dict(surface='#1a1a19', land='#2c2c2a', ocean='#141413', border='#383835', ink='#ffffff',
                 muted='#c3c2b7'),
}
REGION_VIEW = {'World': (15, 10), 'North America': (-95, 38), 'Europe': (10, 50), 'Asia': (110, 30),
               'Oceania': (140, -25), 'South America': (-60, -15)}
MCAP_MIN = {'All': 0, '> 1bn': 1e9, '> 10bn': 1e10, '> 50bn': 5e10, '> 100bn': 1e11, '> 500bn': 5e11}


@st.cache_data(ttl=900, show_spinner=False)
def load_points(folder: str) -> pd.DataFrame:
    return cd.hq_points(folder)


def _theme() -> str:
    try:
        return 'dark' if st.context.theme.type == 'dark' else 'light'
    except Exception:
        return 'light'


def spread(df: pd.DataFrame) -> pd.DataFrame:
    """
    Companies at the same point (39 in one city) are spread on a small spiral around it, so every dot can be
    hovered and clicked. Largest company in the middle.
    """
    df = df.sort_values('market_cap_usd', ascending=False).copy()
    k = df.groupby(['hq_lat', 'hq_lon']).cumcount().to_numpy()
    r = 0.09 * np.sqrt(k)                                 # degrees
    a = k * 2.39996                                       # golden angle
    df['lat'] = df['hq_lat'] + r * np.sin(a)
    df['lon'] = df['hq_lon'] + r * np.cos(a) / np.cos(np.radians(df['hq_lat'].clip(-80, 80)))
    return df


def dot_size(mcap: pd.Series) -> np.ndarray:
    """
    Market cap (USD) -> marker diameter 8..26 px, area ~ log of the size (a few giants shouldn't hide the rest).
    """
    v = np.log10(mcap.fillna(1e8).clip(lower=1e8))
    lo, hi = 8.0, 26.0
    return lo + (v - 8.0) / (12.8 - 8.0) * (hi - lo)


def fmt_usd(v) -> str:
    if v is None or pd.isna(v):
        return '–'
    for div, suf in ((1e12, 'T'), (1e9, 'bn'), (1e6, 'M')):
        if abs(v) >= div:
            return f'${v / div:,.1f}{suf}'
    return f'${v:,.0f}'


def render_page() -> None:
    folder = st.session_state.get('folder')
    st.title('Headquarters')
    try:
        pts = load_points(folder)
    except cd.DatabaseLocked:
        st.info('The database is being updated right now (`python run.py` is writing to it). Try again in a few minutes.')
        return
    except Exception as exc:
        st.warning(f'Could not read the databases in `{folder}`: {exc}')
        return

    # ---- filters: one row above the globe
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1.2])
    sectors = f1.multiselect('Sector', sorted(pts['sector'].unique()), key='hq_sector', placeholder='All sectors')
    ind_pool = pts[pts['sector'].isin(sectors)] if sectors else pts
    industries = f2.multiselect('Industry', sorted(ind_pool['industry'].unique()), key='hq_industry',
                                placeholder='All industries')
    regions = f3.multiselect('Region', sorted(pts['continent'].unique()), key='hq_region', placeholder='All regions')
    mcap = f4.selectbox('Market cap (USD)', list(MCAP_MIN), key='hq_mcap')
    g1, g2, g3, g4 = st.columns([2, 2, 2, 1.2])
    c_pool = pts[pts['continent'].isin(regions)] if regions else pts
    countries = g1.multiselect('HQ country', sorted(c_pool['hq_country'].dropna().unique()), key='hq_country',
                               placeholder='All countries')
    q = g2.text_input('Name or ticker', key='hq_q', placeholder='e.g. Pfizer')
    colour = g3.radio('Colour by', ['Region', 'Sector', 'None'], horizontal=True, key='hq_colour')
    view = g4.selectbox('Centre on', list(REGION_VIEW), key='hq_view')

    v = pts
    if sectors:
        v = v[v['sector'].isin(sectors)]
    if industries:
        v = v[v['industry'].isin(industries)]
    if regions:
        v = v[v['continent'].isin(regions)]
    if countries:
        v = v[v['hq_country'].isin(countries)]
    v = v[v['market_cap_usd'].fillna(0) >= MCAP_MIN[mcap]]
    if q.strip():
        hits = cd.search_companies(v, q, limit=len(v))
        v = v[v['primary_ticker'].isin(hits['primary_ticker'])]

    m1, m2, m3 = st.columns(3)
    m1.metric('Companies', f'{len(v):,}')
    m2.metric('Countries', f"{v['hq_country'].nunique():,}")
    m3.metric('Market cap (USD)', fmt_usd(v['market_cap_usd'].sum()))
    if v.empty:
        st.info('No company matches the filters.')
        return

    # ---- colour groups (at most three hues + Other)
    mode = _theme()
    th, slots = THEME[mode], SLOTS[mode]
    if colour == 'Sector' and v['sector'].nunique() > 3:
        st.caption('Colour by sector needs at most 3 sectors (more colours can\'t be told apart on a map): '
                   'choose up to 3 in the Sector filter. Coloured by region meanwhile.')
        colour = 'Region'
    if colour == 'None':
        groups = pd.Series('Headquarters', index=v.index)
    else:
        col = 'continent' if colour == 'Region' else 'sector'
        top = v[col].value_counts().index[:3].tolist()
        groups = v[col].where(v[col].isin(top), 'Other')
    v = spread(v.assign(group=groups))
    order = [g for g in v['group'].value_counts().index if g != 'Other'] + (['Other'] if (v['group'] == 'Other').any() else [])
    colours = {g: (OTHER if g == 'Other' else slots[i]) for i, g in enumerate(order)}

    fig = go.Figure()
    for g in order:
        d = v[v['group'] == g]
        fig.add_trace(go.Scattergeo(
            lat=d['lat'], lon=d['lon'], mode='markers', name=f'{g} ({len(d):,})',
            marker=dict(size=dot_size(d['market_cap_usd']), color=colours[g], opacity=0.9,
                        line=dict(width=1.5, color=th['surface'])),
            customdata=np.stack([d['primary_ticker'], d['name'], d['hq_city'].fillna(''), d['hq_country'].fillna(''),
                                 d['sector'], d['market_cap_usd'].map(fmt_usd)], axis=-1),
            hovertemplate='<b>%{customdata[1]}</b> (%{customdata[0]})<br>%{customdata[2]}, %{customdata[3]}<br>'
                          '%{customdata[4]} · %{customdata[5]}<extra>Click to open</extra>'))
    lon0, lat0 = REGION_VIEW[view]
    fig.update_geos(projection_type='orthographic', projection_rotation=dict(lon=lon0, lat=lat0),
                    showland=True, landcolor=th['land'], showocean=True, oceancolor=th['ocean'],
                    showcountries=True, countrycolor=th['border'], countrywidth=0.5,
                    showcoastlines=True, coastlinecolor=th['border'], coastlinewidth=0.5,
                    showframe=False, bgcolor='rgba(0,0,0,0)', lataxis_showgrid=False, lonaxis_showgrid=False)
    fig.update_layout(height=680, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)',
                      legend=dict(orientation='h', y=1.02, x=0, font=dict(color=th['muted']),
                                  title_text=f'{colour}:' if colour != 'None' else ''),
                      showlegend=colour != 'None', dragmode='pan', clickmode='event+select',
                      hoverlabel=dict(bgcolor=th['surface'], font_color=th['ink']))
    event = st.plotly_chart(fig, width='stretch', key='hq_globe', on_select='rerun', selection_mode='points',
                            config={'scrollZoom': True, 'displaylogo': False})
    st.caption('Drag to rotate, scroll to zoom. Dot size: market cap. Click a dot to open the company. Companies in '
               'the same city are spread slightly around it.')
    picked = _picked_ticker(event)
    if picked:
        _open(picked)

    # ---- the same companies as a table (also clickable)
    with st.expander(f'Companies on the globe ({len(v):,})'):
        t = v.sort_values('market_cap_usd', ascending=False)[
            ['primary_ticker', 'name', 'sector', 'industry', 'hq_city', 'hq_country', 'market_cap_usd']]
        ev = st.dataframe(t, hide_index=True, width='stretch', height=400, key='hq_table', on_select='rerun',
                          selection_mode='single-row',
                          column_config={'primary_ticker': 'Ticker', 'name': 'Company', 'sector': 'Sector',
                                         'industry': 'Industry', 'hq_city': 'City', 'hq_country': 'Country',
                                         'market_cap_usd': st.column_config.NumberColumn('Market cap (USD)',
                                                                                         format='compact')})
        if ev.selection.rows:
            _open(t.iloc[ev.selection.rows[0]]['primary_ticker'])


def _picked_ticker(event) -> str | None:
    try:
        points = event.selection.points
    except AttributeError:
        points = (event or {}).get('selection', {}).get('points', [])
    for p in points or []:
        cd_ = p.get('customdata') if isinstance(p, dict) else None
        if cd_:
            return cd_[0]
    return None


def _open(ticker: str) -> None:
    st.session_state['company'] = ticker
    if COMPANY_PAGE is not None:
        st.switch_page(COMPANY_PAGE)
