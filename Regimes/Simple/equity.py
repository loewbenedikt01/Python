import pandas as pd
import numpy as np
from config import (
    EQUITY_REGION_WEIGHTS, EQUITY_BULL_THRESHOLD, EQUITY_BEAR_THRESHOLD,
    EQUITY_MIN_STOCKS, EQUITY_MA_WINDOW,
    EQUITY_SMOOTH_WINDOW, EQUITY_TREND_WINDOW,
    EQUITY_CONFIRM_WINDOW, EQUITY_CONFIRM_MIN,
)

_REGIME_TO_NUM = {'Bull': 2, 'Neutral': 1, 'Bear': 0, 'Unknown': -1}
_NUM_TO_REGIME = {v: k for k, v in _REGIME_TO_NUM.items()}


def _apply_hysteresis(smooth: pd.Series) -> list:
    regimes = []
    state   = None
    mid     = (EQUITY_BULL_THRESHOLD + EQUITY_BEAR_THRESHOLD) / 2

    for val in smooth:
        if pd.isna(val):
            regimes.append('Unknown')
            continue

        if state is None:
            state = 'Bull' if val >= mid else 'Bear'

        elif state == 'Bull':
            if   val <  EQUITY_BEAR_THRESHOLD: state = 'Bear'
            elif val <  EQUITY_BULL_THRESHOLD: state = 'Neutral'

        elif state == 'Bear':
            if   val >= EQUITY_BULL_THRESHOLD: state = 'Bull'
            elif val >= EQUITY_BEAR_THRESHOLD: state = 'Neutral'

        else:  # Neutral
            if   val >= EQUITY_BULL_THRESHOLD: state = 'Bull'
            elif val <  EQUITY_BEAR_THRESHOLD: state = 'Bear'

        regimes.append(state)

    return regimes


def compute_equity_regime(df: pd.DataFrame) -> pd.DataFrame:
    close         = df['Close'].unstack('Ticker')
    continent_map = df['Continent'].groupby(level='Ticker').last()

    # dropna before groupby: combined parquet has NaN-Close rows for cross-market
    # non-trading days; one NaN inside the window makes the whole rolling result NaN.
    close_valid = df['Close'].dropna()
    ma200 = (
        close_valid
        .groupby(level='Ticker')
        .transform(lambda s: s.rolling(EQUITY_MA_WINDOW, min_periods=EQUITY_MA_WINDOW).mean())
        .reindex(df.index)
        .unstack('Ticker')
    )
    valid = ma200.notna()
    above = (close > ma200) & valid

    regime_df   = pd.DataFrame(index=close.index)
    region_cols = []

    for cont in sorted(continent_map.unique()):
        col     = 'breadth_' + cont.lower().replace(' ', '_')
        tickers = continent_map[continent_map == cont].index.intersection(close.columns)
        if tickers.empty:
            continue
        n_above = above[tickers].sum(axis=1)
        n_valid = valid[tickers].sum(axis=1)
        # EQUITY_MIN_STOCKS guards against holiday spikes (1-4 stocks trading → spurious 0% or 100%)
        regime_df[col] = (n_above / n_valid.replace(0, np.nan)).where(n_valid >= EQUITY_MIN_STOCKS)
        region_cols.append(col)

    # ffill(limit=3): a region dipping below EQUITY_MIN_STOCKS for 1-3 days (public holiday)
    # produces a NaN spike that the rolling window treats as a genuine data point.
    # Filling forward up to 3 days bridges the gap without masking real regime changes.
    for col in region_cols:
        regime_df[col] = regime_df[col].ffill(limit=3)

    # Adaptive smoothing: small-universe regions (South America, Oceania) swing
    # 10pp per extra stock crossing its MA. Apply a longer window for regions with
    # fewer than 50 tickers to reduce noise without masking real trend changes.
    for col in region_cols:
        cont_name = col.replace('breadth_', '').replace('_', ' ').title()
        n = len(continent_map[continent_map == cont_name].index)
        window = 10 if n >= 50 else 21
        regime_df[col] = (
            regime_df[col].dropna()
            .rolling(window, min_periods=max(3, window // 3))
            .mean()
            .reindex(regime_df.index)
        )

    weighted    = pd.Series(0.0, index=regime_df.index)
    weight_used = pd.Series(0.0, index=regime_df.index)
    for cont, w in EQUITY_REGION_WEIGHTS.items():
        col = 'breadth_' + cont
        if col not in regime_df.columns:
            continue
        mask         = regime_df[col].notna()
        weighted    += regime_df[col].fillna(0) * w * mask
        weight_used += w * mask
    regime_df['breadth_weighted'] = weighted / weight_used.replace(0, np.nan)

    # Roll on trading-day-only subset: NaN weekends inside a 10-row window would
    # prevent min_periods from ever being met on the full calendar index.
    bw_td  = regime_df['breadth_weighted'].dropna()
    smooth = bw_td.rolling(EQUITY_SMOOTH_WINDOW, min_periods=EQUITY_SMOOTH_WINDOW // 2).mean()
    trend  = bw_td.rolling(EQUITY_TREND_WINDOW,  min_periods=EQUITY_TREND_WINDOW  // 2).mean()

    # 31-day delta (half of 63d trend window) with a 1pp deadband: avoids the sign
    # flipping constantly near peaks/troughs where a 10-day diff ticks ± every session.
    delta_long = smooth.diff(EQUITY_TREND_WINDOW // 2)

    regime_df['breadth_smooth'] = smooth.reindex(regime_df.index)
    regime_df['breadth_trend']  = trend.reindex(regime_df.index)
    regime_df['breadth_delta']  = delta_long.reindex(regime_df.index)

    # Direction kept as metadata — monthly slope of the 63d trend MA.
    # Shown in the plot as an indicator but does NOT drive regime state.
    trend_slope = regime_df['breadth_trend'].diff(21)
    regime_df['breadth_direction'] = np.where(
        trend_slope > 0, 'Rising', 'Falling'
    )

    raw_regime = _apply_hysteresis(regime_df['breadth_smooth'])
    regime_df['equity_regime_raw'] = raw_regime

    # Confirmation window on trading-day index so weekends don't dilute the 5-row window.
    # Order matters: ffill on trading days first (fills intra-week gaps in the confirmed
    # series), then reindex to the calendar index, then ffill only to bridge weekends.
    # Mapping to regime labels happens after reindex so the ffill works on integers not strings.
    raw_series    = pd.Series(raw_regime, index=regime_df.index)
    trading_idx   = regime_df['breadth_smooth'].dropna().index
    regime_num_td = raw_series.loc[trading_idx].map(_REGIME_TO_NUM).astype(float)
    confirmed_td  = (
        regime_num_td
        .rolling(EQUITY_CONFIRM_WINDOW)
        .apply(lambda x: x[-1] if (x == x[-1]).sum() >= EQUITY_CONFIRM_MIN else np.nan, raw=True)
        .ffill()                          # fill on trading days only
    )
    regime_df['equity_regime'] = (
        confirmed_td
        .map(_NUM_TO_REGIME)
        .reindex(regime_df.index)         # extend to calendar index (weekends → NaN)
        .ffill()                          # fill only those weekend gaps
        .fillna('Unknown')
    )

    regime_df['regional_dispersion'] = regime_df[region_cols].std(axis=1)
    regime_df['regime_conviction']   = np.select(
        [regime_df['regional_dispersion'] < 0.08,
         regime_df['regional_dispersion'] < 0.15],
        ['High', 'Medium'], default='Low',
    )

    if 'breadth_north_america' in regime_df.columns and 'breadth_europe' in regime_df.columns:
        regime_df['us_eu_divergence'] = (
            regime_df['breadth_north_america'] - regime_df['breadth_europe']
        )

    # ── snapshot print ────────────────────────────────────────────────────────
    valid_rows = regime_df.dropna(subset=['breadth_smooth'])
    latest     = valid_rows.iloc[-1]
    date_str   = valid_rows.index[-1].strftime('%Y-%m-%d')

    print(f'\n=== Equity Regime Snapshot ({date_str}) ===\n')
    print(f'  {"Regime:":<28} {latest["equity_regime"]}')
    print(f'  {"Conviction:":<28} {latest["regime_conviction"]}')
    print(f'  {"Weighted Breadth:":<28} {latest["breadth_weighted"]:.1%}')
    print(f'  {"Breadth Trend (63d avg):":<28} {latest["breadth_trend"]:.1%}')
    print(f'  {"Direction (metadata):":<28} {latest.get("breadth_direction", "?")}')

    print('\n  --- Breadth by Continent (% above 200-MA) ---')
    for col in sorted(region_cols):
        val = latest.get(col, np.nan)
        if pd.isna(val):
            continue
        print(f'  {col.replace("breadth_","").replace("_"," ").title():<22} {val:>5.1%}  {"#"*int(val*20)}')

    if 'us_eu_divergence' in latest.index and not pd.isna(latest['us_eu_divergence']):
        print(f'\n  US - EU Divergence:  {latest["us_eu_divergence"]:+.1%}')

    print('\n  --- Regime Distribution (full history) ---')
    confirmed = valid_rows[valid_rows['equity_regime'] != 'Unknown']['equity_regime']
    dist      = confirmed.value_counts(normalize=True)
    for r in ['Bull', 'Neutral', 'Bear']:
        pct = dist.get(r, 0.0)
        print(f'  {r:<22} {pct:>5.1%}  {"#"*int(pct*40)}')
    print()

    return regime_df
