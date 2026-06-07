import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

DATABASE_DIR = r'c:\Users\benel\Coding\Python\Database'

START_DATA = '2000-01-01'
START_DATE = '2001-01-01'
END_DATE = '2025-12-31'

# Load Data
df_macro        = pd.read_parquet(os.path.join(DATABASE_DIR, 'macro.parquet'))
df_us           = pd.read_parquet(os.path.join(DATABASE_DIR, 'US_equities.parquet'))
df_de           = pd.read_parquet(os.path.join(DATABASE_DIR, 'DE_equities.parquet'))
df_eu           = pd.read_parquet(os.path.join(DATABASE_DIR, 'EU_equities.parquet'))
df_asia         = pd.read_parquet(os.path.join(DATABASE_DIR, 'ASIA_equities.parquet'))
df_rotw         = pd.read_parquet(os.path.join(DATABASE_DIR, 'ROTW_equities.parquet'))
df_bonds        = pd.read_parquet(os.path.join(DATABASE_DIR, 'bonds.parquet'))
df_forex        = pd.read_parquet(os.path.join(DATABASE_DIR, 'forex.parquet'))
df_commodities  = pd.read_parquet(os.path.join(DATABASE_DIR, 'commodities.parquet'))
df_crypto       = pd.read_parquet(os.path.join(DATABASE_DIR, 'crypto.parquet'))
df_sentiment    = pd.read_parquet(os.path.join(DATABASE_DIR, 'sentiment.parquet'))
df_sectors      = pd.read_parquet(os.path.join(DATABASE_DIR, 'sectors.parquet'))
df_indices      = pd.read_parquet(os.path.join(DATABASE_DIR, 'indices.parquet'))

# Process Data
df_equities = pd.concat([df_us, df_de, df_eu, df_asia, df_rotw], axis=0)
df_equities.sort_index(inplace=True)
df_equities = df_equities.loc[pd.IndexSlice[START_DATA:END_DATE, :], :]

# print(df_equities.head(10))


""" Single Asset Class Regimes """
# Equity Regime

# Market-cap-based weights per continent for the weighted global breadth score
REGION_WEIGHTS = {
    'north_america': 0.63,
    'europe':        0.17,
    'asia':          0.13,
    'south_america': 0.04,
    'oceania':       0.03,
}

# Hysteresis: asymmetric entry/exit thresholds prevent rapid flip-flopping
BULL_THRESHOLD = 0.57   # breadth must reach 57% to enter a bull regime
BEAR_THRESHOLD = 0.53   # breadth must drop to 53% to exit a bull regime

# Numeric codes used for the rolling confirmation window
_REGIME_TO_NUM = {
    'Expanding Bull':    0,
    'Deteriorating Bull':1,
    'Recovering Bear':   2,
    'Confirmed Bear':    3,
    'Unknown':          -1,
}
_NUM_TO_REGIME = {v: k for k, v in _REGIME_TO_NUM.items()}


def _apply_hysteresis(smooth: pd.Series, rising: pd.Series) -> list:
    """Stateful regime assignment with asymmetric bull/bear thresholds."""
    regimes = []
    in_bull = None   # None = not yet initialised

    for val, up in zip(smooth, rising):
        if pd.isna(val):
            regimes.append('Unknown')
            continue

        if in_bull is None:
            in_bull = val >= (BULL_THRESHOLD + BEAR_THRESHOLD) / 2 

        if in_bull:
            # already in bull — exit only if breadth drops below BEAR_THRESHOLD
            if val < BEAR_THRESHOLD:
                in_bull = False
        else:
            # already in bear — enter bull only if breadth rises above BULL_THRESHOLD
            if val >= BULL_THRESHOLD:
                in_bull = True

        if in_bull:
            regimes.append('Expanding Bull' if up else 'Deteriorating Bull')
        else:
            regimes.append('Recovering Bear' if up else 'Confirmed Bear')

    return regimes


def compute_equity_regime(df):
    """
    1. Computes daily % of tickers above their 200-day MA per continent.
    2. Builds a weighted global breadth score.
    3. Smooths breadth and classifies into 4 regime states (level x direction).
    4. Measures regional dispersion for regime conviction.
    5. Prints a current snapshot and returns the full time-series DataFrame.
    """
    close        = df['Close'].unstack('Ticker')
    continent_map = df['Continent'].groupby(level='Ticker').last()

    # Compute MA200 per ticker on its own trading calendar.
    # dropna() is critical: the combined parquet retains some NaN-Close rows for non-trading
    # days (where the stack did not fully drop the row). Rolling min_periods=200 requires 200
    # *non-NaN* values in the window; even one NaN row in the window makes the result NaN.
    # Dropping NaN rows first ensures the rolling only sees actual trading-day prices.
    close_valid = df['Close'].dropna()
    ma200 = (
        close_valid
        .groupby(level='Ticker')
        .transform(lambda s: s.rolling(200, min_periods=200).mean())
        .reindex(df.index)          # restore full index (NaN for dropped rows)
        .unstack('Ticker')
    )
    valid = ma200.notna()
    above = (close > ma200) & valid          # True only where MA200 exists and price is above it

    # --- daily breadth by continent ---
    # MIN_SAMPLE_ABS: minimum tickers with valid MA200 needed to trust a continent's
    # breadth reading. Blocks 1-4 stock spikes on European/Asian holidays (e.g. Europe
    # at 0% on Easter Monday because only 1 stock traded) while still passing through
    # legitimate thin days like NA on Presidents Day where ~9 Mexican stocks trade.
    # 5% of the universe was too aggressive — it masked NA on US-only holidays.
    MIN_SAMPLE_ABS = 5

    regime_df  = pd.DataFrame(index=close.index)
    region_cols = []

    for cont in sorted(continent_map.unique()):
        col     = 'breadth_' + cont.lower().replace(' ', '_')
        tickers = continent_map[continent_map == cont].index.intersection(close.columns)
        if tickers.empty:
            continue
        n_above = above[tickers].sum(axis=1)
        n_valid = valid[tickers].sum(axis=1)
        regime_df[col] = (n_above / n_valid.replace(0, np.nan)).where(n_valid >= MIN_SAMPLE_ABS)
        region_cols.append(col)

    # --- weighted global breadth ---
    # Only include continents that have valid breadth on each day (non-NaN).
    # Filling with 0 caused artificial dips on market holidays / non-trading days.
    weighted     = pd.Series(0.0, index=regime_df.index)
    weight_used  = pd.Series(0.0, index=regime_df.index)
    for cont, w in REGION_WEIGHTS.items():
        col = 'breadth_' + cont
        if col not in regime_df.columns:
            continue
        mask          = regime_df[col].notna()
        weighted     += regime_df[col].fillna(0) * w * mask
        weight_used  += w * mask
    regime_df['breadth_weighted'] = weighted / weight_used.replace(0, np.nan)

    # --- smoothing (trading days only) ---
    # regime_df index includes every date across all markets (including weekends/holidays).
    # breadth_weighted is NaN on non-trading days. Rolling with default min_periods=window
    # needs all window rows to be valid — with 2 weekends in a 10-row window we only get 8
    # valid values, so the output is NaN almost everywhere. Fix: compute rolling on the
    # trading-day-only subset, then reindex back to the full calendar.
    bw_td  = regime_df['breadth_weighted'].dropna()
    smooth = bw_td.rolling(10, min_periods=5).mean()
    trend  = bw_td.rolling(63, min_periods=30).mean()
    delta  = smooth.diff(10)
    rising = delta > 0

    regime_df['breadth_smooth']  = smooth.reindex(regime_df.index)
    regime_df['breadth_trend']   = trend.reindex(regime_df.index)
    regime_df['breadth_delta']   = delta.reindex(regime_df.index)
    regime_df['breadth_rising']  = rising.reindex(regime_df.index)

    # --- 4-state regime with hysteresis + confirmation window ---
    # Step 1: stateful hysteresis (asymmetric entry/exit thresholds)
    raw_regime = _apply_hysteresis(regime_df['breadth_smooth'], regime_df['breadth_rising'])
    regime_df['equity_regime_raw'] = raw_regime

    # Step 2: require 4 of the last 5 *trading* days to agree before confirming a switch.
    # Compute on the trading-day-only index so weekend rows (Unknown/-1) don't pollute
    # the 5-row window, then ffill through weekends.
    raw_series    = pd.Series(raw_regime, index=regime_df.index)
    trading_idx   = regime_df['breadth_smooth'].dropna().index
    regime_num_td = raw_series.loc[trading_idx].map(_REGIME_TO_NUM).astype(float)
    confirmed_td  = (
        regime_num_td
        .rolling(5)
        .apply(lambda x: x[-1] if (x == x[-1]).sum() >= 4 else np.nan, raw=True)
        .ffill()
    )
    regime_df['equity_regime'] = (
        confirmed_td
        .reindex(regime_df.index)
        .ffill()
        .map(_NUM_TO_REGIME)
        .fillna('Unknown')
    )

    # --- conviction: how aligned are the regions? ---
    regime_df['regional_dispersion'] = regime_df[region_cols].std(axis=1)
    regime_df['regime_conviction']   = np.select(
        [regime_df['regional_dispersion'] < 0.08,
         regime_df['regional_dispersion'] < 0.15],
        ['High', 'Medium'], default='Low',
    )

    # --- cross-regional divergence signals ---
    if 'breadth_north_america' in regime_df.columns and 'breadth_europe' in regime_df.columns:
        regime_df['us_eu_divergence'] = regime_df['breadth_north_america'] - regime_df['breadth_europe']

    # --- print snapshot ---
    valid_rows = regime_df.dropna(subset=['breadth_smooth'])
    latest     = valid_rows.iloc[-1]
    date_str   = valid_rows.index[-1].strftime('%Y-%m-%d')

    print(f'\n=== Equity Regime Snapshot ({date_str}) ===\n')
    print(f'  {"Regime:":<28} {latest["equity_regime"]}')
    print(f'  {"Conviction:":<28} {latest["regime_conviction"]}')
    print(f'  {"Weighted Breadth:":<28} {latest["breadth_weighted"]:.1%}')
    print(f'  {"Breadth Trend (63d avg):":<28} {latest["breadth_trend"]:.1%}')
    print(f'  {"Direction:":<28} {"Rising" if latest["breadth_rising"] else "Falling"}')

    print('\n  --- Breadth by Continent (% above 200-MA) ---')
    for col in sorted(region_cols):
        cont_name = col.replace('breadth_', '').replace('_', ' ').title()
        val = latest[col] if col in latest.index else np.nan
        if pd.isna(val):
            continue
        bar = '#' * int(val * 20)
        print(f'  {cont_name:<22} {val:>5.1%}  {bar}')

    if 'us_eu_divergence' in latest.index and not pd.isna(latest['us_eu_divergence']):
        print(f'\n  US - EU Divergence:  {latest["us_eu_divergence"]:+.1%}')
    print()

    return regime_df


REGIME_COLORS = {
    'Expanding Bull':    '#2ecc71',   # green
    'Deteriorating Bull':'#ffc04d',   # light orange
    'Recovering Bear':   '#3498db',   # blue
    'Confirmed Bear':    '#e74c3c',   # red
    'Unknown':           '#cccccc',   # grey
}


def plot_equity_regime(regime_df, region_cols):
    data = regime_df.dropna(subset=['breadth_smooth']).copy()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 9),
        gridspec_kw={'height_ratios': [2, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor('#1a1a2e')
    for ax in (ax1, ax2):
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='#cccccc')
        ax.xaxis.label.set_color('#cccccc')
        ax.yaxis.label.set_color('#cccccc')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

    # ── top panel: shade regime periods + breadth lines ──────────────────────
    prev_regime = None
    start       = data.index[0]

    for date, row in data.iterrows():
        regime = row['equity_regime']
        if regime != prev_regime:
            if prev_regime is not None:
                ax1.axvspan(start, date,
                            color=REGIME_COLORS.get(prev_regime, '#cccccc'),
                            alpha=0.25, linewidth=0)
            start       = date
            prev_regime = regime
    # close last block
    ax1.axvspan(start, data.index[-1],
                color=REGIME_COLORS.get(prev_regime, '#cccccc'),
                alpha=0.25, linewidth=0)

    ax1.plot(data.index, data['breadth_weighted'] * 100,
             color='#ffffff', linewidth=0.8, alpha=0.5, label='Daily breadth')
    ax1.plot(data.index, data['breadth_smooth'] * 100,
             color='#f1c40f', linewidth=1.8, label='Smooth (10d)')
    ax1.plot(data.index, data['breadth_trend'] * 100,
             color='#e74c3c', linewidth=1.5, linestyle='--', label='Trend (63d)')

    ax1.axhline(55, color='#7f8c8d', linewidth=1, linestyle=':')
    ax1.axhline(45, color='#7f8c8d', linewidth=1, linestyle=':')
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('% Above 200-MA', color='#cccccc')
    ax1.set_title('Global Equity Regime  —  Breadth Above 200-Day MA',
                  color='#ecf0f1', fontsize=13, pad=10)

    # legend with regime colour swatches
    from matplotlib.patches import Patch
    handles = [ax1.get_legend_handles_labels()[0][i]
               for i in range(3)]
    handles += [Patch(facecolor=REGIME_COLORS[r], alpha=0.6, label=r)
                for r in ['Expanding Bull', 'Deteriorating Bull',
                          'Recovering Bear', 'Confirmed Bear']]
    ax1.legend(handles=handles,
               labels=['Daily breadth', 'Smooth (10d)', 'Trend (63d)',
                       'Expanding Bull', 'Deteriorating Bull',
                       'Recovering Bear', 'Confirmed Bear'],
               loc='upper left', framealpha=0.3,
               labelcolor='#ecf0f1', facecolor='#1a1a2e')

    # ── bottom panel: per-continent breadth ───────────────────────────────────
    cont_colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    for col, color in zip(sorted(region_cols), cont_colors):
        if col not in data.columns:
            continue
        label = col.replace('breadth_', '').replace('_', ' ').title()
        # Roll on non-NaN rows only (trading days for this continent) so that
        # a single holiday NaN doesn't propagate across the following 9 rows.
        td = data[col].dropna()
        series = td.rolling(10, min_periods=5).mean().reindex(data.index) * 100
        ax2.plot(data.index, series, linewidth=1.2, color=color, label=label)

    ax2.axhline(55, color='#7f8c8d', linewidth=1, linestyle=':')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('% Above 200-MA', color='#cccccc')
    ax2.set_xlabel('Date', color='#cccccc')
    ax2.legend(loc='upper left', framealpha=0.3,
               labelcolor='#ecf0f1', facecolor='#1a1a2e',
               ncol=3, fontsize=8)

    plt.tight_layout()
    plt.show()


equity_regime_df = compute_equity_regime(df_equities)
equity_regime_df = equity_regime_df.loc[START_DATE:END_DATE]

_EXCLUDE = {'breadth_weighted', 'breadth_smooth', 'breadth_trend', 'breadth_rising', 'breadth_delta'}
plot_equity_regime(equity_regime_df, [
    c for c in equity_regime_df.columns
    if c.startswith('breadth_') and c not in _EXCLUDE
])

# Bond Regime


# Forex Regime


# Commodity Regime


# Crypto Regime


""" Multi-Asset Class Regimes """
# Global Growth Regime


# Inflation Regime


# Liquidity Regime


# Risk Appetite Regime


""" Major Regime """


""" Hidden Markov Model """

