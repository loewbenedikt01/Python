import pandas as pd
import numpy as np
from config import (
    COMMODITY_MA_WINDOW, COMMODITY_SMOOTH_WINDOW, COMMODITY_MOMENTUM_WINDOW,
    COMMODITY_BASKET_WINDOW, COMMODITY_SCORE_SMOOTH,
    COMMODITY_BULL_ENTRY, COMMODITY_BULL_EXIT,
    COMMODITY_BEAR_ENTRY, COMMODITY_BEAR_EXIT,
    COMMODITY_CONFIRM_WINDOW, COMMODITY_CONFIRM_MIN,
)

_REGIME_TO_NUM = {'Commodity Bull': 2, 'Commodity Neutral': 1, 'Commodity Bear': 0, 'Unknown': -1}
_NUM_TO_REGIME = {v: k for k, v in _REGIME_TO_NUM.items()}


def _smooth(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    mp = min_periods or max(1, window // 2)
    return series.dropna().rolling(window, min_periods=mp).mean().reindex(series.index)


def _apply_hysteresis(score_series: pd.Series) -> list:
    regimes = []
    state   = None
    for score in score_series:
        if pd.isna(score):
            regimes.append('Unknown')
            continue
        if state is None:
            if   score >= COMMODITY_BULL_ENTRY: state = 'Commodity Bull'
            elif score <= COMMODITY_BEAR_ENTRY: state = 'Commodity Bear'
            else:                               state = 'Commodity Neutral'
        elif state == 'Commodity Bull':
            if   score <= COMMODITY_BEAR_ENTRY: state = 'Commodity Bear'
            elif score <  COMMODITY_BULL_EXIT:  state = 'Commodity Neutral'
        elif state == 'Commodity Bear':
            if   score >= COMMODITY_BULL_ENTRY: state = 'Commodity Bull'
            elif score >  COMMODITY_BEAR_EXIT:  state = 'Commodity Neutral'
        else:  # Neutral
            if   score >= COMMODITY_BULL_ENTRY: state = 'Commodity Bull'
            elif score <= COMMODITY_BEAR_ENTRY: state = 'Commodity Bear'
        regimes.append(state)
    return regimes


def _confirm(raw: pd.Series) -> pd.Series:
    num = raw.map(_REGIME_TO_NUM).astype(float)
    confirmed = (
        num
        .rolling(COMMODITY_CONFIRM_WINDOW, min_periods=COMMODITY_CONFIRM_MIN)
        .apply(lambda x: x[-1] if (x == x[-1]).sum() >= COMMODITY_CONFIRM_MIN else np.nan, raw=True)
        .bfill()
        .ffill()
    )
    return confirmed.map(_NUM_TO_REGIME).fillna('Unknown')


# ── 1. Broad commodity trend (CRB proxy) ──────────────────────────────────────

def compute_commodity_trend(close: pd.DataFrame) -> pd.Series:
    """Average 63d return across a liquid futures basket — overall commodity momentum."""
    basket  = ['CL=F', 'HG=F', 'GC=F', 'ZC=F', 'ZW=F', 'NG=F']
    returns = {t: close[t].dropna().pct_change(COMMODITY_BASKET_WINDOW).reindex(close.index)
               for t in basket if t in close.columns}
    if not returns:
        return pd.Series(np.nan, index=close.index)
    return _smooth(pd.DataFrame(returns).mean(axis=1), window=COMMODITY_SMOOTH_WINDOW)


# ── 2. Copper / Gold ratio (growth vs fear) ───────────────────────────────────

def compute_copper_gold_ratio(close: pd.DataFrame):
    """Rising ratio = growth > fear = risk-on. Falling = recession / safe-haven demand."""
    if 'HG=F' not in close.columns or 'GC=F' not in close.columns:
        nan = pd.Series(np.nan, index=close.index)
        return nan, nan, nan, nan
    ratio        = (close['HG=F'].dropna() / close['GC=F'].dropna()).reindex(close.index)
    ratio_ma200  = _smooth(ratio, window=COMMODITY_MA_WINDOW)
    ratio_smooth = _smooth(ratio, window=COMMODITY_SMOOTH_WINDOW)
    sig          = (ratio_smooth > ratio_ma200).astype(float)
    return ratio, ratio_smooth, ratio_ma200, sig


# ── 3. Energy regime (inflation signal) ───────────────────────────────────────

def compute_energy_regime(close: pd.DataFrame) -> pd.Series:
    """WTI trend, momentum, Brent confirmation → score 0-3."""
    if 'CL=F' not in close.columns:
        return pd.Series(np.nan, index=close.index)
    # Floor at $1: April 2020 WTI went negative (-$37) due to storage constraints —
    # a physically impossible settlement that corrupts MA and momentum calculations
    wti       = close['CL=F'].dropna().clip(lower=1.0)
    wti_ma200 = _smooth(wti, window=COMMODITY_MA_WINDOW)
    wti_sm    = _smooth(wti, window=COMMODITY_SMOOTH_WINDOW)
    wti_mom   = _smooth(
        wti.pct_change(COMMODITY_MOMENTUM_WINDOW).reindex(close.index), window=10
    )
    sig_trend = (wti_sm    > wti_ma200).astype(float)
    sig_mom   = (wti_mom   > 0        ).astype(float)
    brent     = close.get('BZ=F')
    if brent is not None:
        brent_ma200 = _smooth(brent, window=COMMODITY_MA_WINDOW)
        sig_brent   = (_smooth(brent, window=COMMODITY_SMOOTH_WINDOW) > brent_ma200).astype(float)
    else:
        sig_brent = sig_trend  # fallback — WTI counts twice
    return (sig_trend + sig_mom + sig_brent).reindex(close.index)


# ── 4. Gold regime (real yield / safe-haven signal) ───────────────────────────

def compute_gold_regime(close: pd.DataFrame) -> pd.Series:
    """Gold trend, momentum, silver confirmation → score 0-3."""
    if 'GC=F' not in close.columns:
        return pd.Series(np.nan, index=close.index)
    gold       = close['GC=F'].dropna()
    gold_ma200 = _smooth(gold, window=COMMODITY_MA_WINDOW)
    gold_sm    = _smooth(gold, window=COMMODITY_SMOOTH_WINDOW)
    gold_mom   = _smooth(
        gold.pct_change(COMMODITY_MOMENTUM_WINDOW).reindex(close.index), window=10
    )
    sig_trend  = (gold_sm  > gold_ma200).astype(float)
    sig_mom    = (gold_mom > 0         ).astype(float)
    silver     = close.get('SI=F')
    if silver is not None:
        silver_ma200 = _smooth(silver, window=COMMODITY_MA_WINDOW)
        sig_silver   = (_smooth(silver, window=COMMODITY_SMOOTH_WINDOW) > silver_ma200).astype(float)
    else:
        sig_silver = sig_trend
    return (sig_trend + sig_mom + sig_silver).reindex(close.index)


# ── 5. Agriculture regime (food inflation / supply shock) ─────────────────────

def compute_agriculture_regime(close: pd.DataFrame) -> pd.Series:
    """Fraction of corn / wheat / soybeans above their MA200 — breadth-style."""
    ags  = ['ZC=F', 'ZW=F', 'ZS=F']
    sigs = []
    for ticker in ags:
        if ticker in close.columns:
            s     = close[ticker].dropna()
            ma200 = _smooth(s, window=COMMODITY_MA_WINDOW)
            sm    = _smooth(s, window=COMMODITY_SMOOTH_WINDOW)
            sigs.append((sm > ma200).astype(float).reindex(close.index))
    if not sigs:
        return pd.Series(np.nan, index=close.index)
    return pd.concat(sigs, axis=1).mean(axis=1)


# ── Master commodity regime ────────────────────────────────────────────────────

def compute_commodity_regime(df_commodities: pd.DataFrame) -> pd.DataFrame:
    """
    4-group master regime: Industrial (Cu/Au) + Energy + Precious + Agriculture.
    Each group casts one vote → score 0-4 → smooth → hysteresis → confirmation.

    Regime states: Commodity Bull / Commodity Neutral / Commodity Bear
    """
    close = df_commodities['Close'].unstack('Ticker')

    ratio, ratio_smooth, ratio_ma200, sig_copper_gold = compute_copper_gold_ratio(close)
    energy_score = compute_energy_regime(close)
    gold_score   = compute_gold_regime(close)
    ag_signal    = compute_agriculture_regime(close)
    broad_trend  = compute_commodity_trend(close)

    commodity_df = pd.DataFrame(index=close.index)

    # ── Store raw prices and sub-signals for the plot ─────────────────────────
    if 'CL=F' in close.columns:
        wti = close['CL=F'].dropna().clip(lower=1.0)
        commodity_df['oil']        = wti.reindex(close.index)
        commodity_df['oil_ma200']  = _smooth(wti, window=COMMODITY_MA_WINDOW)
        commodity_df['oil_smooth'] = _smooth(wti, window=COMMODITY_SMOOTH_WINDOW)
    if 'GC=F' in close.columns:
        gold = close['GC=F'].dropna()
        commodity_df['gold']       = gold.reindex(close.index)
        commodity_df['gold_ma200'] = _smooth(gold, window=COMMODITY_MA_WINDOW)

    commodity_df['copper_gold_ratio']        = ratio
    commodity_df['copper_gold_ratio_smooth'] = ratio_smooth
    commodity_df['copper_gold_ratio_ma200']  = ratio_ma200
    commodity_df['energy_score']             = energy_score
    commodity_df['gold_score']               = gold_score
    commodity_df['ag_signal']                = ag_signal
    commodity_df['broad_trend']              = broad_trend

    # ── 4-vote master score ───────────────────────────────────────────────────
    commodity_df['sig_industrial'] = sig_copper_gold.reindex(close.index)
    commodity_df['sig_energy']     = (energy_score >= 2).astype(float).where(energy_score.notna())
    commodity_df['sig_gold']       = (gold_score   >= 2).astype(float).where(gold_score.notna())
    commodity_df['sig_ag']         = (ag_signal    > 0.5).astype(float).where(ag_signal.notna())

    sig_cols = ['sig_industrial', 'sig_energy', 'sig_gold', 'sig_ag']
    commodity_df['bull_score']        = commodity_df[sig_cols].fillna(0).sum(axis=1)
    commodity_df['bull_score_smooth'] = _smooth(commodity_df['bull_score'], window=COMMODITY_SCORE_SMOOTH)

    raw_regime = pd.Series(
        _apply_hysteresis(commodity_df['bull_score_smooth']),
        index=commodity_df.index,
    )
    commodity_df['commodity_regime_raw'] = raw_regime
    commodity_df['commodity_regime']     = _confirm(raw_regime)

    # ── snapshot print ────────────────────────────────────────────────────────
    valid    = commodity_df.dropna(subset=['bull_score'])
    latest   = valid.iloc[-1]
    date_str = valid.index[-1].strftime('%Y-%m-%d')

    print(f'\n=== Commodity Regime Snapshot ({date_str}) ===\n')
    print(f'  {"Commodity Regime:":<28} {latest["commodity_regime"]}')
    print(f'  {"Bull Score (raw):":<28} {int(latest["bull_score"])}/4')
    print(f'  {"Bull Score (smooth):":<28} {latest["bull_score_smooth"]:.2f}/4')
    print(f'  {"Broad Trend (63d avg ret):":<28} {latest["broad_trend"]:+.2%}')
    if not pd.isna(latest.get('copper_gold_ratio_smooth', np.nan)):
        dev = latest['copper_gold_ratio_smooth'] / latest['copper_gold_ratio_ma200'] - 1
        print(f'  {"Cu/Au Ratio vs MA200:":<28} {dev:+.2%}')

    print('\n  --- Signal Breakdown ---')
    for col, label in [
        ('sig_industrial', 'Cu/Au Ratio Above MA200'),
        ('sig_energy',     'Energy Score ≥ 2/3'),
        ('sig_gold',       'Gold Score ≥ 2/3'),
        ('sig_ag',         'Agriculture Breadth > 50%'),
    ]:
        val = latest.get(col, np.nan)
        print(f'    {"YES" if val == 1.0 else "NO ":<4} {label}')

    print('\n  --- Commodity Regime Distribution (full history) ---')
    for regime in ['Commodity Bull', 'Commodity Neutral', 'Commodity Bear']:
        pct = (commodity_df['commodity_regime'] == regime).mean()
        print(f'  {regime:<22} {pct:>5.1%}  {"#"*int(pct*40)}')
    print()

    return commodity_df
