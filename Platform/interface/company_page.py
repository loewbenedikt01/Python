"""
Company page: header (name, ticker, sector, country, price, market cap) and tabs Overview,
Financials, Holders, Executives, Subsidiaries, Calendar, Clinical Trials. All data from the
pipeline's DuckDB files via company_data (read-only, closed after each query, results cached).
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import company_data as cd

PHASE_COLOURS = {1: '#9ecae1', 2: '#4292c6', 3: '#08519c', 4: '#f16913'}
RANGES = ['1M', '3M', '6M', 'YTD', '1Y', '5Y', 'Max']
INTRADAY_WINDOWS = {'1m': [1, 2, 5], '5m': [1, 5, 10, 30], '10m': [5, 10, 30, 60], '1h': [30, 90, 180, 365]}
LABELS = {
    'revenue': 'Revenue', 'cost_of_revenue': 'Cost of revenue', 'gross_profit': 'Gross profit',
    'rnd_expense': 'R&D', 'sga_expense': 'SG&A', 'operating_expenses': 'Operating expenses',
    'operating_income': 'Operating income', 'interest_expense': 'Interest expense', 'pretax_income': 'Pre-tax income',
    'income_tax': 'Income tax', 'net_income': 'Net income', 'net_income_to_common': 'Net income to common',
    'ebitda': 'EBITDA', 'depreciation_amortization': 'D&A', 'eps_basic': 'EPS basic', 'eps_diluted': 'EPS diluted',
    'shares_basic_wavg': 'Shares basic (avg)', 'shares_diluted_wavg': 'Shares diluted (avg)',
    'gross_margin': 'Gross margin', 'operating_margin': 'Operating margin', 'net_margin': 'Net margin',
    'cash': 'Cash', 'short_term_investments': 'Short-term investments', 'receivables': 'Receivables',
    'inventory': 'Inventory', 'current_assets': 'Current assets', 'ppe_net': 'PP&E (net)', 'goodwill': 'Goodwill',
    'intangibles': 'Intangibles', 'total_assets': 'Total assets', 'accounts_payable': 'Accounts payable',
    'current_liabilities': 'Current liabilities', 'short_term_debt': 'Short-term debt',
    'long_term_debt': 'Long-term debt', 'total_debt': 'Total debt', 'net_debt': 'Net debt',
    'total_liabilities': 'Total liabilities', 'redeemable_equity': 'Redeemable equity',
    'noncontrolling_interest': 'Non-controlling interest', 'total_equity': 'Equity (parent)',
    'retained_earnings': 'Retained earnings', 'shares_outstanding': 'Shares outstanding',
    'operating_cash_flow': 'Operating cash flow', 'capex': 'Capex', 'free_cash_flow': 'Free cash flow',
    'investing_cash_flow': 'Investing cash flow', 'financing_cash_flow': 'Financing cash flow',
    'dividends_paid': 'Dividends paid', 'share_buybacks': 'Share buybacks', 'stock_based_comp': 'Stock-based comp',
    'dividends_per_share': 'Dividend per share',
}
TRANS_CODES = {'P': 'Purchase', 'S': 'Sale', 'A': 'Award', 'M': 'Option exercise', 'F': 'Tax withholding',
               'G': 'Gift', 'D': 'Disposition to issuer', 'C': 'Conversion', 'J': 'Other', 'X': 'Option exercise'}


# ----
# CACHED DATA (the functions close their connections; only the results are cached)
# ----

@st.cache_data(ttl=900, show_spinner=False)
def _cached(fn_name: str, *args):
    return getattr(cd, fn_name)(*args)


def load(fn_name: str, *args):
    return _cached(fn_name, *args)


# ----
# FORMATTING
# ----

def money(v, cur: str | None = '', digits: int = 2) -> str:
    if v is None or pd.isna(v):
        return '–'
    a = abs(v)
    for div, suf in ((1e12, 'T'), (1e9, 'bn'), (1e6, 'M'), (1e3, 'k')):
        if a >= div:
            return f"{v / div:,.{digits}f}{suf} {cur or ''}".strip()
    return f"{v:,.{digits}f} {cur or ''}".strip()


def num(v, digits: int = 2) -> str:
    return '–' if v is None or pd.isna(v) else f'{v:,.{digits}f}'


def pct(v, digits: int = 1) -> str:
    return '–' if v is None or pd.isna(v) else f'{v * 100:.{digits}f}%'


def fmt_item(item: str, v, cur: str | None) -> str:
    if v is None or pd.isna(v):
        return ''
    if item in cd.RATIOS:
        return f'{v * 100:.1f}%'
    if item in cd.PER_SHARE:
        return f'{v:,.2f}'
    if item in cd.COUNTS:
        return money(v, '', 1)
    return money(v, '', 1)


def date_str(d) -> str:
    return '–' if d is None or pd.isna(d) else pd.Timestamp(d).strftime('%Y-%m-%d')


# ----
# PAGE
# ----

def render_page() -> None:
    folder = st.session_state.get('folder')
    try:
        comps = load('companies', folder)
    except cd.DatabaseLocked:
        st.info('The database is being updated right now (`python run.py` is writing to it). Try again in a few minutes.')
        return
    except Exception as exc:                                 # missing files
        st.warning(f'Could not read the databases in `{folder}`: {exc}')
        return

    tickers = comps['primary_ticker'].tolist()
    names = dict(zip(comps['primary_ticker'], comps['name']))
    # any listing -> the company's primary ticker (GOOG -> GOOGL)
    primary = {t: p for p, ts in zip(comps['primary_ticker'], comps['tickers']) for t in (ts if ts is not None else [])}
    primary.update({t: t for t in tickers})
    clicked = st.session_state.pop('company', None)                 # row clicked on the Markets page
    if clicked or 'company_current' not in st.session_state:
        want = clicked or st.query_params.get('company', 'AAPL')
        st.session_state['company_current'] = primary.get(want, tickers[0])

    def pick(t: str) -> None:                                    # button callback: open it, clear the search
        st.session_state['company_current'] = t
        st.session_state['company_q'] = ''

    q = st.text_input('Search company', key='company_q', placeholder='Name or ticker, e.g. Moderna or MRNA')
    if q.strip():
        hits = cd.search_companies(comps, q)
        if hits.empty:
            st.caption(f'No company matches "{q}".')
        elif len(hits) == 1:
            st.session_state['company_current'] = hits.iloc[0]['primary_ticker']
        else:
            cols = st.columns(4)
            for i, r in enumerate(hits.itertuples(index=False)):
                cols[i % 4].button(f'{r.primary_ticker} · {r.name}', key=f'hit_{r.primary_ticker}',
                                   on_click=pick, args=(r.primary_ticker,), width='stretch')
    ticker = st.session_state['company_current']
    if st.query_params.get('company') != ticker:
        st.query_params['company'] = ticker
    row = comps[comps['primary_ticker'] == ticker].iloc[0]
    cid = int(row['corporate_id'])

    try:
        h = load('header', folder, cid)
        render_header(h)
        sec = bool(h.get('sec_filer')) and pd.notna(h.get('cik'))
        tabs = st.tabs(['Overview', 'Financials', 'Holders', 'Executives', 'Subsidiaries', 'Calendar',
                        'Clinical Trials'])
        with tabs[0]:
            tab_overview(folder, cid, ticker, h, sec)
        with tabs[1]:
            tab_financials(folder, cid, sec) if sec else no_sec()
        with tabs[2]:
            tab_holders(folder, cid) if sec else no_sec()
        with tabs[3]:
            tab_executives(folder, cid) if sec else no_sec()
        with tabs[4]:
            tab_subsidiaries(folder, cid) if sec else no_sec()
        with tabs[5]:
            tab_calendar(folder, cid, ticker, h)
        with tabs[6]:
            tab_trials(folder, cid)
    except cd.DatabaseLocked:
        st.info('The database is being updated right now (`python run.py` is writing to it). Try again in a few minutes.')


def no_sec() -> None:
    st.info('No SEC data for this company (non-US or not an SEC filer): financials, holders, executives and '
            'subsidiaries come from SEC filings and are only available for the US companies.')


def render_header(h: dict) -> None:
    ticker, cur = h.get('primary_ticker'), h.get('price_currency') or h.get('trading_currency')
    st.markdown(f"## {h.get('name', ticker)}  \n"
                f"**{ticker}**" + (f" · {h['exchange']}" if h.get('exchange') else '')
                + (f" · other listings: {', '.join(t for t in h['tickers'] if t != ticker)}"
                   if h.get('tickers') is not None and len(h['tickers']) > 1 else ''))
    parts = [' › '.join(x for x in (h.get('sector'), h.get('industry')) if x), h.get('country')]
    if h.get('fiscal_year_end'):
        parts.append(f"FY ends {h['fiscal_year_end']}")
    if h.get('employees') and pd.notna(h.get('employees')):
        parts.append(f"{int(h['employees']):,} employees")
    st.caption(' · '.join(p for p in parts if p))

    c = st.columns(5)
    price, prev = h.get('price'), h.get('prev_close')
    delta = f'{(price / prev - 1) * 100:+.2f}% (1D)' if price and prev else None
    c[0].metric('Price', f'{num(price)} {cur or ""}', delta)
    mc_usd = h.get('market_cap_usd')
    c[1].metric('Market cap', money(h.get('market_cap'), h.get('mcap_currency')),
                f'{money(mc_usd, "USD")}' if mc_usd and h.get('mcap_currency') not in (None, 'USD') else None,
                delta_color='off')
    src = {'sec_cover_page': 'SEC cover page', 'sec_balance_sheet': 'SEC balance sheet',
           'sec_after_deal': 'SEC, after deal', 'sec_backfilled': 'SEC', 'yfinance_current': 'yfinance'}
    c[2].metric('Shares', money(h.get('shares'), '', 2),
                f"{src.get(h.get('shares_source'), h.get('shares_source') or '')}"
                + (f", {date_str(h.get('shares_as_of'))}" if str(h.get('shares_source', '')).startswith('sec') else ''),
                delta_color='off')
    nxt = h.get('next_earnings')
    c[3].metric('Next earnings', date_str(nxt) if nxt else '–',
                ' '.join(x for x in (h.get('next_earnings_time'), h.get('next_earnings_status')) if x) or None,
                delta_color='off')
    c[4].metric('52-week range', f"{num(h.get('low_52w'))} – {num(h.get('high_52w'))}",
                f"data to {date_str(h.get('price_date'))}", delta_color='off')


# ----
# OVERVIEW
# ----

def tab_overview(folder, cid, ticker, h, sec) -> None:
    left, right = st.columns([3, 1])
    intraday_ok = cd.has_intraday(folder, ticker)
    with right:
        mode = st.radio('Chart', ['Daily', 'Intraday'] if intraday_ok else ['Daily'], horizontal=True,
                        key='ov_mode', help=None if intraday_ok else 'Intraday bars exist for US stocks only.')
        if mode == 'Daily':
            rng = st.radio('Range', RANGES, index=4, horizontal=True, key='ov_range')
            show = st.radio('Show', ['Price', 'Market cap'], horizontal=True, key='ov_show')
            marks = st.checkbox('Earnings dates', value=True, key='ov_marks')
        else:
            interval = st.radio('Bars', cd.INTRADAY_INTERVALS, index=2, horizontal=True, key='ov_interval')
            days = st.select_slider('Days', INTRADAY_WINDOWS[interval], value=INTRADAY_WINDOWS[interval][-1],
                                    key=f'ov_days_{interval}')
    with left:
        if mode == 'Daily':
            daily_chart(folder, cid, ticker, rng, show, marks, h)
        else:
            intraday_chart(folder, ticker, interval, days)

    kf = load('key_figures', folder, cid) if sec else pd.DataFrame()
    a, b = st.columns([2, 3])
    with a:
        st.subheader('Key figures')
        if kf.empty:
            st.caption('No SEC financials.')
        else:
            cur = kf.attrs.get('currency')
            tbl = pd.DataFrame({
                '': [LABELS.get(i, i) for i in kf['item']],
                f"FY{kf.attrs.get('fiscal_year') or ''}": [fmt_item(i, v, cur) for i, v in zip(kf['item'], kf['last_fy'])],
                f"TTM to {date_str(kf.attrs.get('ttm_end'))}" if kf.attrs.get('ttm_end') else 'TTM':
                    [fmt_item(i, v, cur) for i, v in zip(kf['item'], kf['ttm'])],
            })
            st.dataframe(tbl, hide_index=True, width='stretch')
            st.caption(f'Amounts in {cur}. TTM = sum of the last 4 quarters; debt and equity from the latest quarter.')
    with b:
        st.subheader('About')
        loc = ', '.join(x for x in (h.get('hq_city'), h.get('hq_state'), h.get('hq_country')) if x)
        st.markdown(' · '.join(x for x in (
            f"**HQ** {loc}" if loc else None,
            f"[{h['website']}]({h['website']})" if h.get('website') else None,
            f"**CIK** {int(h['cik'])}" if pd.notna(h.get('cik')) else None,
            f"**ISIN** {h['isin']}" if h.get('isin') else None,
            f"**Reporting currency** {h['reporting_currency']}" if h.get('reporting_currency') else None) if x))
        desc = h.get('business_description')
        if desc:
            st.write(desc)
        if sec:
            f = load('filings', folder, cid, 8)
            if not f.empty:
                st.markdown('**Latest SEC filings**')
                f = f.assign(link=f.apply(cd.filing_url, axis=1))
                st.dataframe(f[['filed_date', 'form_type', 'items', 'link']], hide_index=True, width='stretch',
                             column_config={'link': st.column_config.LinkColumn('Document', display_text='open'),
                                            'filed_date': 'Filed', 'form_type': 'Form', 'items': '8-K items'})


def window(df: pd.DataFrame, rng: str, col: str = 'date') -> pd.DataFrame:
    if df.empty or rng == 'Max':
        return df
    end = df[col].max()
    start = {'1M': end - pd.DateOffset(months=1), '3M': end - pd.DateOffset(months=3),
             '6M': end - pd.DateOffset(months=6), 'YTD': pd.Timestamp(end.year, 1, 1),
             '1Y': end - pd.DateOffset(years=1), '5Y': end - pd.DateOffset(years=5)}[rng]
    return df[df[col] >= start]


def daily_chart(folder, cid, ticker, rng, show, marks, h) -> None:
    df = window(load('daily_prices', folder, ticker, cid), rng)
    if df.empty:
        st.info('No prices.')
        return
    y = 'close' if show == 'Price' else 'market_cap'
    unit = h.get('price_currency') if show == 'Price' else h.get('mcap_currency')
    fig = go.Figure(go.Scatter(x=df['date'], y=df[y], mode='lines', name=show, line=dict(width=1.6)))
    if marks:
        e = load('earnings_history', folder, cid, ticker, 60)
        e = e[(pd.to_datetime(e['earnings_date']) >= df['date'].min())
              & (pd.to_datetime(e['earnings_date']) <= df['date'].max() + pd.Timedelta(days=120))] if not e.empty else e
        for r in e.itertuples(index=False):
            fig.add_vline(x=pd.Timestamp(r.earnings_date), line_width=1, line_dash='dot',
                          line_color='#d62728' if r.earnings_date_status == 'estimated' else '#999')
    fig.update_layout(height=420, margin=dict(t=10, b=10), yaxis_title=f'{show} ({unit or ""})',
                      hovermode='x unified', showlegend=False)
    st.plotly_chart(fig, width='stretch')
    if marks:
        st.caption('Dotted lines: earnings releases (grey reported, red upcoming / estimated).')


def intraday_chart(folder, ticker, interval, days) -> None:
    df = load('intraday_prices', folder, ticker, interval, days)
    if df.empty:
        st.info('No intraday bars.')
        return
    fig = go.Figure(go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                                   name=ticker))
    fig.update_layout(height=420, margin=dict(t=10, b=10), xaxis_rangeslider_visible=False,
                      yaxis_title='Price (USD)', hovermode='x unified')
    # no gaps for nights and weekends (New York time)
    fig.update_xaxes(rangebreaks=[dict(bounds=['sat', 'mon']), dict(bounds=[16, 9.5], pattern='hour')])
    st.plotly_chart(fig, width='stretch')
    st.caption(f'{interval} bars, New York time, regular session, split-adjusted. Source: Alpaca (SIP).')


# ----
# FINANCIALS
# ----

def tab_financials(folder, cid, sec) -> None:
    a, b, c = st.columns([1, 1, 2])
    period = a.radio('Period', ['Yearly', 'Quarterly'], horizontal=True, key='fin_period')
    n = b.select_slider('Periods', [4, 8, 12, 16, 20], value=10 if period == 'Yearly' else 8, key=f'fin_n_{period}') \
        if period == 'Quarterly' else b.select_slider('Periods', [5, 10, 15], value=10, key='fin_n_y')
    fin = load('financials', folder, cid, 'Y' if period == 'Yearly' else 'Q', n)
    if fin.empty:
        st.info('No financials.')
        return
    cur = fin['currency'].dropna().iloc[0] if fin['currency'].notna().any() else ''
    all_items = [i for cols in cd.STATEMENTS.values() for i in cols if i in fin.columns and fin[i].notna().any()]
    item = c.selectbox('Chart', all_items, index=all_items.index('revenue') if 'revenue' in all_items else 0,
                       format_func=lambda i: LABELS.get(i, i), key='fin_item')
    ch = fin[['label', 'period_end', item]].sort_values('period_end')
    fig = px.bar(ch, x='label', y=item, labels={'label': '', item: f'{LABELS.get(item, item)} ({cur})'})
    fig.update_layout(height=260, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width='stretch')

    for statement in cd.STATEMENTS:
        t = cd.statement_table(fin, statement)
        if t.empty:
            continue
        show = pd.DataFrame({col: [fmt_item(i, t.at[i, col], cur) for i in t.index] for col in t.columns},
                            index=[LABELS.get(i, i) for i in t.index])
        st.markdown(f'**{statement}**')
        st.dataframe(show, width='stretch', height=min(38 + 35 * len(show), 760))
    notes = [f'Amounts in {cur}; shares in millions / billions.', 'Newest period left.']
    if fin['q4_derived'].any():
        notes.append('Q4 derived = full year minus Q1-Q3 (no Q4 filing).')
    if fin['gross_profit_derived'].fillna(False).any():
        notes.append('Some gross profit values derived as revenue - cost of revenue.')
    st.caption(' '.join(notes) + ' Source: SEC XBRL company facts.')


# ----
# HOLDERS
# ----

def tab_holders(folder, cid) -> None:
    hq = load('holders', folder, cid)
    if hq.empty:
        st.info('No 13F holder data.')
        return
    hist = hq.dropna(subset=['inst_ownership_pct']).sort_values('period_end')
    if not hist.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist['label'], y=hist['inst_ownership_pct'] * 100, name='Institutional ownership %',
                                 mode='lines+markers'))
        fig.add_trace(go.Bar(x=hist['label'], y=hist['n_institutional_holders'], name='Holders', yaxis='y2',
                             opacity=0.3))
        fig.update_layout(height=260, margin=dict(t=10, b=10), yaxis=dict(title='% of shares'),
                          yaxis2=dict(title='Holders', overlaying='y', side='right', showgrid=False),
                          legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig, width='stretch')

    with_top = hq[hq['top_holders'].map(lambda v: v is not None and len(v) > 0)]
    if with_top.empty:
        return
    a, b = st.columns([1, 1])
    q = a.selectbox('Quarter', with_top['label'].tolist(), key='hold_q')
    group = b.checkbox('Add up related filers (holder group)', key='hold_group')
    r = with_top[with_top['label'] == q].iloc[0]
    top = pd.DataFrame(list(r['top_holders']))
    if group:
        top['holder'] = top['holder_group'].fillna(top['holder_name'])
        top = top.groupby('holder', as_index=False)[['shares', 'value_usd', 'pct_of_shares_out',
                                                     'change_shares_vs_prev_quarter']].sum(min_count=1)
        top = top.sort_values('shares', ascending=False)
    else:
        top = top.rename(columns={'holder_name': 'holder'})
    top['pct_of_shares_out'] = top['pct_of_shares_out'] * 100
    st.dataframe(top[['holder'] + (['holder_group'] if not group else []) + ['shares', 'value_usd', 'pct_of_shares_out',
                                                                           'change_shares_vs_prev_quarter']],
                 hide_index=True, width='stretch',
                 column_config={'holder': 'Holder', 'holder_group': 'Group',
                                'shares': st.column_config.NumberColumn('Shares', format='%,.0f'),
                                'value_usd': st.column_config.NumberColumn('Value (USD)', format='%,.0f'),
                                'pct_of_shares_out': st.column_config.NumberColumn('% of shares', format='%.2f%%'),
                                'change_shares_vs_prev_quarter': st.column_config.NumberColumn(
                                    'Change vs prev. quarter', format='%+,.0f')})
    st.caption(f"13F report date {date_str(r['holders_report_date'])}. Institutional ownership "
               f"{pct(r['inst_ownership_pct'])} from {num(r['n_institutional_holders'], 0)} filers. Change is empty "
               'when the filer\'s previous report was missing or incomplete.')


# ----
# EXECUTIVES
# ----

def tab_executives(folder, cid) -> None:
    ex = load('executives', folder, cid)
    if ex.empty:
        st.info('No executives (Forms 3 / 4 / 5).')
    else:
        years = ex['fiscal_year'].astype(int).tolist()
        y = st.selectbox('Fiscal year', years, key='exec_year')
        people = pd.DataFrame(list(ex[ex['fiscal_year'] == y].iloc[0]['executives']))
        a, b = st.columns([3, 1])
        a.dataframe(people[['name', 'title', 'role', 'is_officer', 'is_director']], hide_index=True, width='stretch',
                    column_config={'name': 'Name', 'title': 'Title', 'role': 'Role', 'is_officer': 'Officer',
                                   'is_director': 'Director'})
        added, removed = cd.year_changes(ex, 'executives', 'name', y)
        with b:
            st.markdown(f'**Changes vs FY{y - 1}**')
            if not added and not removed:
                st.caption('none' if y - 1 in years else f'no data for FY{y - 1}')
            for n in added:
                st.markdown(f':green[+ {n}]')
            for n in removed:
                st.markdown(f':red[− {n}]')
        st.caption('Everyone who filed a Form 3 / 4 / 5 as officer or director during the fiscal year.')

    st.subheader('Insider transactions')
    tr = load('insider_transactions', folder, cid, 50)
    if tr.empty:
        st.caption('None.')
        return
    tr['type'] = tr['trans_code'].map(TRANS_CODES).fillna(tr['trans_code'])
    tr['date'] = tr['trans_date'].astype(str) + tr['date_suspect'].map({True: ' ⚠', False: ''})
    st.dataframe(tr[['date', 'filing_date', 'name', 'role', 'type', 'acquired_disposed', 'shares', 'price', 'value',
                     'shares_owned_after']], hide_index=True, width='stretch',
                 column_config={'date': 'Trade date', 'filing_date': 'Filed', 'name': 'Name', 'role': 'Role',
                                'type': 'Type', 'acquired_disposed': 'A/D',
                                'shares': st.column_config.NumberColumn('Shares', format='%,.0f'),
                                'price': st.column_config.NumberColumn('Price', format='%.2f'),
                                'value': st.column_config.NumberColumn('Value', format='%,.0f'),
                                'shares_owned_after': st.column_config.NumberColumn('Owned after', format='%,.0f')})
    st.caption('Last 50 non-derivative transactions (Form 4). ⚠ = trade date after the filing date or in the future '
               '(typo in the filing).')


# ----
# SUBSIDIARIES
# ----

def tab_subsidiaries(folder, cid) -> None:
    sub = load('subsidiaries', folder, cid)
    if sub.empty:
        st.info('No Exhibit 21 subsidiary lists.')
        return
    years = sub['fiscal_year'].astype(int).tolist()
    a, b, c, d = st.columns([1, 1, 1, 2])
    y = a.selectbox('Fiscal year', years, key='sub_year')
    r = sub[sub['fiscal_year'] == y].iloc[0]
    lst = pd.DataFrame(list(r['subsidiaries']) if r['subsidiaries'] is not None else [], columns=['name', 'jurisdiction'])
    b.metric('Subsidiaries', f'{len(lst):,}')
    c.metric('Jurisdictions', f"{lst['jurisdiction'].nunique():,}")
    added, removed = cd.year_changes(sub, 'subsidiaries', 'name', y)
    d.metric(f'Changes vs FY{y - 1}', f'+{len(added)} / −{len(removed)}' if y - 1 in years else '–', delta_color='off')
    notes = []
    if r['carried_forward']:
        notes.append('List carried forward from the previous year (no Exhibit 21 filed).')
    if r['subsidiaries_scope'] == 'significant_only':
        notes.append('The company lists only its significant subsidiaries.')
    elif r['subsidiaries_scope'] == 'omits_insignificant':
        notes.append('Subsidiaries that together are not significant are left out (Item 601(b)(21)).')
    if notes:
        st.caption(' '.join(notes))

    left, right = st.columns([2, 3])
    with left:
        top = lst['jurisdiction'].fillna('(not given)').value_counts().head(15).sort_values()
        fig = px.bar(x=top.values, y=top.index, orientation='h', labels={'x': 'Subsidiaries', 'y': ''})
        fig.update_layout(height=420, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width='stretch')
    with right:
        q = st.text_input('Search', key='sub_search', placeholder='name or country')
        view = lst if not q else lst[lst['name'].str.contains(q, case=False, na=False)
                                     | lst['jurisdiction'].str.contains(q, case=False, na=False)]
        st.dataframe(view, hide_index=True, width='stretch', height=380,
                     column_config={'name': 'Name', 'jurisdiction': 'Jurisdiction'})
    if added or removed:
        with st.expander(f'Added ({len(added)}) / removed ({len(removed)}) vs FY{y - 1}'):
            x, z = st.columns(2)
            x.dataframe(pd.DataFrame({'Added': added}), hide_index=True, width='stretch')
            z.dataframe(pd.DataFrame({'Removed': removed}), hide_index=True, width='stretch')


# ----
# CALENDAR
# ----

def tab_calendar(folder, cid, ticker, h) -> None:
    e = load('earnings_history', folder, cid, ticker, 12)
    a, b = st.columns([3, 2])
    with a:
        st.subheader('Earnings')
        if e.empty:
            st.caption('No earnings dates (SEC 8-K / Finnhub: US companies only).')
        else:
            e = e.assign(reaction=e['reaction'] * 100)
            st.dataframe(e[['earnings_date', 'earnings_time', 'earnings_date_status', 'quarter', 'reaction']],
                         hide_index=True, width='stretch',
                         column_config={'earnings_date': 'Date', 'earnings_time': 'Time',
                                        'earnings_date_status': 'Status', 'quarter': 'Fiscal quarter',
                                        'reaction': st.column_config.NumberColumn('Price reaction', format='%+.1f%%')})
            st.caption('bmo = before the open, amc = after the close. Price reaction: bmo previous close -> close of '
                       'the day, otherwise close of the day -> next close.')
        ar = load('annual_reports', folder, cid, 6)
        if not ar.empty:
            st.markdown('**Annual reports (10-K)**')
            st.dataframe(ar, hide_index=True, width='stretch',
                         column_config={'fiscal_year': 'Fiscal year', 'period_end': 'Period end',
                                        'annual_report_date': 'Filed'})
    with b:
        div, ca = load('dividends_splits', folder, ticker, 12)
        st.subheader('Dividends')
        if div.empty:
            st.caption('None.')
        else:
            st.dataframe(div, hide_index=True, width='stretch',
                         column_config={'date': 'Ex-date', 'dividends': st.column_config.NumberColumn('Amount', format='%.4f'),
                                        'currency': 'Currency'})
        st.subheader('Splits and spin-offs')
        if ca.empty:
            st.caption('None.')
        else:
            ca = ca.assign(kind=ca.apply(lambda r: 'split' if abs(r['price_only_factor'] - 1) < 1e-6 else (
                'spin-off' if abs(r['split_factor'] - 1) < 1e-6 else 'split + spin-off'), axis=1))
            st.dataframe(ca[['date', 'kind', 'split_factor', 'price_only_factor', 'yahoo_factor']], hide_index=True,
                         width='stretch',
                         column_config={'date': 'Ex-date', 'kind': 'Type',
                                        'split_factor': st.column_config.NumberColumn('Split', format='%.4g'),
                                        'price_only_factor': st.column_config.NumberColumn('Spin-off factor', format='%.4g'),
                                        'yahoo_factor': st.column_config.NumberColumn('Yahoo factor', format='%.4g')})


# ----
# CLINICAL TRIALS
# ----

def tab_trials(folder, cid) -> None:
    t = load('trials', folder, cid)
    if t.empty:
        st.info('No clinical trials found for this company (lead sponsor or collaborator). Trials are searched for '
                'the healthcare companies only.')
        return
    counts = load('trial_phase_counts', folder, cid)
    a, b, c, d = st.columns(4)
    a.metric('Trials', f'{len(t):,}')
    b.metric('As lead sponsor', f"{int((t['role'] == 'lead').sum()):,}")
    c.metric('Recruiting / active', f"{int(t['overall_status'].isin(['RECRUITING', 'ACTIVE_NOT_RECRUITING', 'NOT_YET_RECRUITING', 'ENROLLING_BY_INVITATION']).sum()):,}")
    d.metric('With results', f"{int(t['has_results'].sum()):,}")

    if not counts.empty:
        years = sorted(counts['start_year'].unique())
        yr = st.select_slider('Start years', options=years, value=(max(years[0], years[-1] - 15), years[-1]),
                              key='trial_years') if len(years) > 1 else (years[0], years[0])
        cc = counts[(counts['start_year'] >= yr[0]) & (counts['start_year'] <= yr[1])]
        # total per year = distinct trials with a phase (a Phase 1/2 trial is one trial but two segments)
        phased = t[t['phase_groups'].map(lambda v: v is not None and len(v) > 0) & t['start_year'].notna()]
        totals = phased.groupby('start_year')['nct_id'].nunique()
        totals = totals[(totals.index >= yr[0]) & (totals.index <= yr[1])]
        fig = go.Figure()
        for ph in (1, 2, 3, 4):
            s = cc[cc['phase'] == ph]
            fig.add_trace(go.Bar(x=s['start_year'], y=s['trials'], name=f'Phase {ph}', marker_color=PHASE_COLOURS[ph]))
        fig.add_trace(go.Scatter(x=totals.index, y=cc.groupby('start_year')['trials'].sum().reindex(totals.index),
                                 text=totals.values, mode='text', textposition='top center', showlegend=False,
                                 hoverinfo='skip'))
        fig.update_layout(barmode='stack', height=380, margin=dict(t=30, b=10), xaxis=dict(dtick=1, title='Start year'),
                          yaxis_title='Trials', legend=dict(orientation='h', y=1.12))
        st.plotly_chart(fig, width='stretch')
        st.caption('Trials by start year (planned starts included) and phase; the number on top is the total of '
                   'trials. A Phase 1/2 trial counts in both phases, so the segments can add up to more than the '
                   'total. Trials without a phase (devices, observational) are not in the chart.')

    f1, f2, f3 = st.columns(3)
    phases = f1.multiselect('Phase', [1, 2, 3, 4], key='trial_phase', format_func=lambda p: f'Phase {p}')
    statuses = f2.multiselect('Status', sorted(t['overall_status'].dropna().unique()), key='trial_status')
    role = f3.radio('Role', ['All', 'Lead sponsor', 'Collaborator'], horizontal=True, key='trial_role')
    v = t
    if phases:
        v = v[v['phase_groups'].map(lambda g: g is not None and bool(set(g) & set(phases)))]
    if statuses:
        v = v[v['overall_status'].isin(statuses)]
    if role != 'All':
        v = v[v['role'] == ('lead' if role == 'Lead sponsor' else 'collaborator')]
    v = v.assign(link='https://clinicaltrials.gov/study/' + v['nct_id'],
                 phase=v['phases'].map(lambda p: ', '.join(x.replace('PHASE', 'P').replace('EARLY_P', 'Early P')
                                                           for x in p) if p is not None else ''),
                 conditions=v['conditions'].map(lambda c: ', '.join(c[:3]) if c is not None else ''))
    st.dataframe(v[['link', 'title', 'phase', 'overall_status', 'start_date', 'primary_completion_date', 'enrollment',
                    'conditions', 'role', 'lead_sponsor']], hide_index=True, width='stretch', height=420,
                 column_config={'link': st.column_config.LinkColumn('NCT ID', display_text=r'https://clinicaltrials\.gov/study/(.*)'),
                                'title': 'Title', 'phase': 'Phase', 'overall_status': 'Status', 'start_date': 'Start',
                                'primary_completion_date': 'Primary completion',
                                'enrollment': st.column_config.NumberColumn('Enrollment', format='%,d'),
                                'conditions': 'Conditions', 'role': 'Role', 'lead_sponsor': 'Lead sponsor'})
    with st.expander('Version history of a trial'):
        nid = st.selectbox('Trial', v['nct_id'].tolist(), key='trial_hist') if not v.empty else None
        if nid:
            vh = load('trial_versions', folder, nid)
            vh = vh.assign(changed_sections=vh['changed_sections'].map(lambda c: ', '.join(c) if c is not None else ''))
            st.dataframe(vh, hide_index=True, width='stretch',
                         column_config={'version': 'Version', 'downloaded_at': 'Downloaded',
                                        'last_change_date': 'Changed on ClinicalTrials.gov', 'change_flag': 'Changed',
                                        'changed_sections': 'Changed modules'})
