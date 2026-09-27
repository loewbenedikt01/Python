import pandas as pd
import numpy as np
from Finance.Regimes.config import (
    BOND_BULL_ENTRY, BOND_BULL_EXIT, BOND_BEAR_ENTRY, BOND_BEAR_EXIT,
    BOND_CURVE_STEEP_THRESHOLD, BOND_CURVE_FLAT_THRESHOLD,
    BOND_MOMENTUM_THRESHOLD, BOND_MOMENTUM_WINDOW,
    BOND_REAL_YIELD_HIGH, BOND_REAL_YIELD_LOW,
    BOND_CURVE_SMOOTH, BOND_MOMENTUM_SMOOTH, BOND_REAL_YIELD_SMOOTH,
    BOND_REAL_YIELD_NORM_WINDOW, BOND_TLT_SMOOTH,
    BOND_MA50_WINDOW, BOND_MA200_WINDOW, BOND_SHY_REL_WINDOW,
    BOND_SCORE_SMOOTH, BOND_CONFIRM_WINDOW, BOND_CONFIRM_MIN,
)

_BOND_REGIME_TO_NUM = {'Bond Bull': 2, 'Bond Neutral': 1, 'Bond Bear': 0, 'Unknown': -1}
_BOND_NUM_TO_REGIME = {v: k for k, v in _BOND_REGIME_TO_NUM.items()}


def _smooth(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    mp = min_periods or max(1, window // 2)
    return series.dropna().rolling(window, min_periods=mp).mean().reindex(series.index)


def _apply_bond_hysteresis(score_series: pd.Series) -> list:
    regimes = []
    state   = None

    for score in score_series:
        if pd.isna(score):
            regimes.append('Unknown')
            continue

        if state is None:
            if   score >= BOND_BULL_ENTRY: state = 'Bond Bull'
            elif score <= BOND_BEAR_ENTRY: state = 'Bond Bear'
            else:                          state = 'Bond Neutral'

        elif state == 'Bond Bull':
            # Exit Bull only when score drops to BOND_BULL_EXIT (1 unit below entry)
            # prevents exiting on small dips that don't represent a real regime change
            if   score <= BOND_BEAR_ENTRY: state = 'Bond Bear'
            elif score <  BOND_BULL_EXIT:  state = 'Bond Neutral'

        elif state == 'Bond Bear':
            # Exit Bear only when score rises to BOND_BEAR_EXIT (1 unit above entry)
            if   score >= BOND_BULL_ENTRY: state = 'Bond Bull'
            elif score >  BOND_BEAR_EXIT:  state = 'Bond Neutral'

        else:  # Bond Neutral — still requires full entry threshold to commit
            if   score >= BOND_BULL_ENTRY: state = 'Bond Bull'
            elif score <= BOND_BEAR_ENTRY: state = 'Bond Bear'

        regimes.append(state)

    return regimes


def compute_bond_regime(df_bonds: pd.DataFrame) -> pd.DataFrame:
    close   = df_bonds['Close'].unstack('Ticker')
    bond_df = pd.DataFrame(index=close.index)

    irx = close['^IRX']
    fvx = close['^FVX']
    tnx = close['^TNX']
    tyx = close['^TYX']

    bond_df['curve_2s10s']        = tnx - irx
    bond_df['curve_5s30s']        = tyx - fvx
    bond_df['curve_level']        = tnx
    bond_df['curve_2s10s_smooth'] = _smooth(bond_df['curve_2s10s'], window=BOND_CURVE_SMOOTH)

    bond_df['curve_regime'] = np.select([
        bond_df['curve_2s10s_smooth'] >  BOND_CURVE_STEEP_THRESHOLD,
        bond_df['curve_2s10s_smooth'] >  0,
        bond_df['curve_2s10s_smooth'] >  BOND_CURVE_FLAT_THRESHOLD,
    ], ['Steep', 'Normal', 'Flat'], default='Inverted')

    bond_df['rate_ma50']  = _smooth(tnx, window=BOND_MA50_WINDOW)
    bond_df['rate_ma200'] = _smooth(tnx, window=BOND_MA200_WINDOW)

    bond_df['rate_momentum']        = tnx.dropna().diff(BOND_MOMENTUM_WINDOW).reindex(close.index)
    bond_df['rate_momentum_smooth'] = _smooth(bond_df['rate_momentum'], window=BOND_MOMENTUM_SMOOTH)

    bond_df['rate_trend'] = np.where(tnx > bond_df['rate_ma200'], 'Rising', 'Falling')
    bond_df['policy_direction'] = np.select([
        bond_df['rate_momentum_smooth'] >  BOND_MOMENTUM_THRESHOLD,
        bond_df['rate_momentum_smooth'] < -BOND_MOMENTUM_THRESHOLD,
    ], ['Tightening', 'Easing'], default='Neutral')

    tip      = close['TIP'].dropna()
    ief      = close['IEF'].dropna()
    tip_norm = (tip / tip.rolling(BOND_REAL_YIELD_NORM_WINDOW, min_periods=BOND_REAL_YIELD_NORM_WINDOW // 2).mean()).reindex(close.index)
    ief_norm = (ief / ief.rolling(BOND_REAL_YIELD_NORM_WINDOW, min_periods=BOND_REAL_YIELD_NORM_WINDOW // 2).mean()).reindex(close.index)

    bond_df['real_yield_proxy']        = tip_norm / ief_norm
    bond_df['real_yield_proxy_smooth'] = _smooth(bond_df['real_yield_proxy'], window=BOND_REAL_YIELD_SMOOTH)
    bond_df['real_yield_regime'] = np.where(
        bond_df['real_yield_proxy_smooth'] > BOND_REAL_YIELD_HIGH, 'Falling Real',
        np.where(bond_df['real_yield_proxy_smooth'] < BOND_REAL_YIELD_LOW, 'Rising Real', 'Neutral')
    )

    tlt = close['TLT'].dropna()
    shy = close['SHY'].dropna()

    bond_df['tlt_ma200'] = _smooth(tlt, window=BOND_MA200_WINDOW)
    tlt_smooth = _smooth(tlt, window=BOND_TLT_SMOOTH).reindex(close.index)
    bond_df['tlt_trend'] = np.where(tlt_smooth > bond_df['tlt_ma200'], 'Bull', 'Bear')

    shy_rel = (shy / _smooth(shy, window=BOND_SHY_REL_WINDOW)).reindex(close.index)
    tlt_rel = (tlt / _smooth(tlt, window=BOND_SHY_REL_WINDOW)).reindex(close.index)
    bond_df['duration_spread'] = shy_rel - tlt_rel

    bond_df['sig_curve_ok']       = (bond_df['curve_2s10s_smooth']     >  0                 ).astype(float)
    bond_df['sig_yields_falling'] = (bond_df['rate_momentum_smooth']    <  0                 ).astype(float)
    bond_df['sig_tlt_uptrend']    = (tlt_smooth                         >  bond_df['tlt_ma200']).astype(float)
    bond_df['sig_real_falling']   = (bond_df['real_yield_proxy_smooth'] >  BOND_REAL_YIELD_HIGH).astype(float)

    sig_cols = ['sig_curve_ok', 'sig_yields_falling', 'sig_tlt_uptrend', 'sig_real_falling']
    # fillna(0): sig_real_falling is NaN before ~2003 (TIP/IEF data); treat as abstaining
    bond_df['bond_bull_score']        = bond_df[sig_cols].fillna(0).sum(axis=1)
    bond_df['bond_bull_score_smooth'] = _smooth(bond_df['bond_bull_score'], window=BOND_SCORE_SMOOTH)

    # Feed smoothed continuous score — prevents single-signal integer flips from
    # instantly crossing the bull/bear thresholds every day
    raw_regime = _apply_bond_hysteresis(bond_df['bond_bull_score_smooth'])
    bond_df['bond_regime_raw'] = raw_regime

    regime_num    = bond_df['bond_regime_raw'].map(_BOND_REGIME_TO_NUM).astype(float)
    confirmed_num = (
        regime_num
        .rolling(BOND_CONFIRM_WINDOW)
        .apply(lambda x: x[-1] if (x == x[-1]).sum() >= BOND_CONFIRM_MIN else np.nan, raw=True)
        .bfill()
        .ffill()
    )
    bond_df['bond_regime'] = confirmed_num.map(_BOND_NUM_TO_REGIME).fillna('Unknown')

    # ── snapshot print ────────────────────────────────────────────────────────
    valid    = bond_df.dropna(subset=['curve_2s10s_smooth', 'rate_ma200'])
    latest   = valid.iloc[-1]
    date_str = valid.index[-1].strftime('%Y-%m-%d')

    print(f'\n=== Bond Regime Snapshot ({date_str}) ===\n')
    print(f'  {"Bond Regime:":<28} {latest["bond_regime"]}')
    print(f'  {"Curve Regime:":<28} {latest["curve_regime"]}')
    print(f'  {"2s10s Spread (smooth):":<28} {latest["curve_2s10s_smooth"]:+.2f}%')
    print(f'  {"10yr Yield:":<28} {latest["curve_level"]:.2f}%')
    print(f'  {"Rate Trend (vs MA200):":<28} {latest["rate_trend"]}')
    print(f'  {"Policy Direction:":<28} {latest["policy_direction"]}')
    print(f'  {"Real Yield Regime:":<28} {latest["real_yield_regime"]}')
    print(f'  {"Duration Trend (TLT):":<28} {latest["tlt_trend"]}')
    print(f'  {"Bull Score (raw):":<28} {int(latest["bond_bull_score"])}/4')
    print(f'  {"Bull Score (smooth):":<28} {latest["bond_bull_score_smooth"]:.2f}/4')

    print('\n  --- Signal Breakdown ---')
    for col, label in [
        ('sig_curve_ok',       'Curve Not Inverted'),
        ('sig_yields_falling', 'Yields Falling (63d smooth)'),
        ('sig_tlt_uptrend',    'TLT Above MA200'),
        ('sig_real_falling',   'Real Yields Falling'),
    ]:
        print(f'    {"YES" if latest[col] == 1.0 else "NO ":<4} {label}')

    print('\n  --- Bond Regime Distribution (full history) ---')
    for regime in ['Bond Bull', 'Bond Neutral', 'Bond Bear']:
        pct = (bond_df['bond_regime'] == regime).mean()
        print(f'  {regime:<16} {pct:>5.1%}  {"#"*int(pct*40)}')
    print()

    return bond_df
