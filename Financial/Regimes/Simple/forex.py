import pandas as pd
import numpy as np
from Finance.Regimes.config import (
    FOREX_MA50_WINDOW, FOREX_MA200_WINDOW, FOREX_MOMENTUM_WINDOW,
    FOREX_STRENGTH_WINDOW, FOREX_SCORE_SMOOTH, FOREX_EM_NORM_WINDOW,
    FOREX_USD_BULL_ENTRY, FOREX_USD_BEAR_ENTRY,
    FOREX_CARRY_BULL_ENTRY, FOREX_CARRY_BEAR_ENTRY,
    FOREX_JPY_BULL_ENTRY, FOREX_JPY_BEAR_ENTRY,
    FOREX_EUROPE_BULL_ENTRY, FOREX_EUROPE_BEAR_ENTRY,
    FOREX_EM_STRESS_THRESHOLD, FOREX_EM_RELIEF_THRESHOLD,
    FOREX_CONFIRM_WINDOW, FOREX_CONFIRM_MIN,
    FOREX_MASTER_CONFIRM_WINDOW, FOREX_MASTER_CONFIRM_MIN, FOREX_MIN_PAIRS,
)

# ── Currency pair definitions ─────────────────────────────────────────────────
# 'positive': ticker rising → currency stronger
# 'negative': ticker rising → currency weaker (sign is flipped in strength calc)
CURRENCY_PAIRS = {
    'USD': {
        'positive': ['USDJPY=X', 'USDCAD=X', 'USDCHF=X', 'USDMXN=X',
                     'USDINR=X', 'USDCNY=X', 'USDZAR=X', 'USDTRY=X'],
        'negative': ['EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'NZDUSD=X'],
        'index':    'DX-Y.NYB',
    },
    'EUR': {
        'positive': ['EURUSD=X', 'EURGBP=X', 'EURJPY=X', 'EURCHF=X',
                     'EURAUD=X', 'EURCAD=X'],
        'negative': [],
    },
    'GBP': {
        'positive': ['GBPUSD=X', 'GBPJPY=X', 'GBPCHF=X', 'GBPAUD=X'],
        'negative': ['EURGBP=X'],
    },
    'JPY': {
        'positive': [],
        'negative': ['USDJPY=X', 'EURJPY=X', 'GBPJPY=X', 'AUDJPY=X',
                     'NZDJPY=X', 'CADJPY=X', 'CHFJPY=X'],
    },
    'AUD': {
        'positive': ['AUDUSD=X', 'AUDJPY=X', 'AUDNZD=X', 'AUDCAD=X'],
        'negative': ['EURAUD=X', 'GBPAUD=X'],
    },
    # USDCHF/EURCHF/GBPCHF rising = CHF weaker → negative for CHF strength
    'CHF': {
        'positive': ['CHFJPY=X'],
        'negative': ['USDCHF=X', 'EURCHF=X', 'GBPCHF=X'],
    },
}

# ── Internal helpers ──────────────────────────────────────────────────────────

def _smooth(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    mp = min_periods or max(1, window // 2)
    return series.dropna().rolling(window, min_periods=mp).mean().reindex(series.index)


def _apply_hysteresis(score: pd.Series, bull_entry: float, bear_entry: float,
                      bull_label: str, neutral_label: str, bear_label: str) -> list:
    regimes = []
    state   = None
    for val in score:
        if pd.isna(val):
            regimes.append('Unknown')
            continue
        if state is None:
            if   val >= bull_entry: state = bull_label
            elif val <= bear_entry: state = bear_label
            else:                   state = neutral_label
        elif state == bull_label:
            if   val <= bear_entry: state = bear_label
            elif val <  bull_entry: state = neutral_label
        elif state == bear_label:
            if   val >= bull_entry: state = bull_label
            elif val >  bear_entry: state = neutral_label
        else:  # neutral
            if   val >= bull_entry: state = bull_label
            elif val <= bear_entry: state = bear_label
        regimes.append(state)
    return regimes


def _confirm(regime_series: pd.Series, num_map: dict, window: int, min_agree: int) -> pd.Series:
    rev_map = {v: k for k, v in num_map.items()}
    num = regime_series.map(num_map).astype(float)
    confirmed = (
        num
        .rolling(window, min_periods=min_agree)
        .apply(lambda x: x[-1] if (x == x[-1]).sum() >= min_agree else np.nan, raw=True)
        .bfill()
        .ffill()
    )
    return confirmed.map(rev_map).fillna('Unknown')


def _score_to_regime(score_raw: pd.Series, bull_entry: float, bear_entry: float,
                     bull_label: str, neutral_label: str, bear_label: str) -> pd.Series:
    """Smooth score → hysteresis → confirmation → confirmed regime Series."""
    score_smooth = _smooth(score_raw, window=FOREX_SCORE_SMOOTH)
    raw = pd.Series(
        _apply_hysteresis(score_smooth, bull_entry, bear_entry,
                          bull_label, neutral_label, bear_label),
        index=score_raw.index,
    )
    num_map = {bull_label: 2, neutral_label: 1, bear_label: 0, 'Unknown': -1}
    return _confirm(raw, num_map, FOREX_CONFIRM_WINDOW, FOREX_CONFIRM_MIN)


# ── 1. Currency strength index ────────────────────────────────────────────────

def compute_currency_strength(close: pd.DataFrame) -> pd.DataFrame:
    """Normalised N-day pct change averaged across all pairs, with sign adjustment."""
    returns = close.pct_change(FOREX_STRENGTH_WINDOW)
    strength = {}
    for currency, pairs in CURRENCY_PAIRS.items():
        scores = []
        for t in pairs['positive']:
            if t in returns.columns:
                scores.append(returns[t])
        for t in pairs['negative']:
            if t in returns.columns:
                scores.append(-returns[t])
        if scores:
            strength[f'strength_{currency.lower()}'] = pd.concat(scores, axis=1).mean(axis=1)
    return pd.DataFrame(strength, index=close.index)


# ── 2. USD regime ─────────────────────────────────────────────────────────────

def compute_usd_regime(close: pd.DataFrame) -> pd.DataFrame:
    """
    3 signals → score 0-3 → USD Bull / USD Neutral / USD Bear.
    USD Bull = dollar dominant (risk-off for EM, commodities headwind).
    """
    out = pd.DataFrame(index=close.index)

    if 'DX-Y.NYB' not in close.columns:
        out['usd_regime'] = 'Unknown'
        return out

    dxy = close['DX-Y.NYB']
    dxy_s                       = _smooth(dxy, window=5)
    out['dxy']                  = dxy
    out['dxy_ma50']             = _smooth(dxy, window=FOREX_MA50_WINDOW)
    out['dxy_ma200']            = _smooth(dxy, window=FOREX_MA200_WINDOW)
    out['dxy_momentum']         = dxy.dropna().pct_change(FOREX_MOMENTUM_WINDOW).reindex(close.index)
    out['dxy_momentum_smooth']  = _smooth(out['dxy_momentum'], window=FOREX_SCORE_SMOOTH)

    # Signal 1: DXY above long-term trend
    out['sig_usd_above_ma200']  = (dxy_s > out['dxy_ma200']).astype(float)
    # Signal 2: 63d momentum — independent of level
    out['sig_usd_momentum']     = (out['dxy_momentum_smooth'] > 0).astype(float)
    # Signal 3: DXY making higher 126d highs vs 21 days ago — trend acceleration,
    # uncorrelated with whether price is above/below MA
    roll_max                    = dxy.dropna().rolling(126).max().reindex(close.index)
    out['sig_usd_trend_accel']  = (roll_max > roll_max.shift(21)).astype(float)

    score = out[['sig_usd_above_ma200', 'sig_usd_momentum', 'sig_usd_trend_accel']].sum(axis=1)
    out['usd_bull_score']  = score
    out['usd_regime'] = _score_to_regime(
        score, FOREX_USD_BULL_ENTRY, FOREX_USD_BEAR_ENTRY,
        'USD Bull', 'USD Neutral', 'USD Bear',
    )
    return out


# ── 3. Carry regime ───────────────────────────────────────────────────────────

def compute_carry_regime(close: pd.DataFrame) -> pd.DataFrame:
    """
    AUD/JPY is the single best carry proxy — captures high-yield vs safe-haven in one pair.
    3 signals → Carry Bull (risk-on) / Neutral / Carry Bear (risk-off).
    """
    out = pd.DataFrame(index=close.index)

    audjpy = close.get('AUDJPY=X')
    if audjpy is None:
        out['carry_regime'] = 'Unknown'
        return out

    audjpy    = audjpy.dropna()
    audjpy_s  = _smooth(audjpy, window=10)
    out['audjpy']          = audjpy.reindex(close.index)
    out['audjpy_ma50']     = _smooth(audjpy, window=FOREX_MA50_WINDOW).reindex(close.index)
    out['audjpy_ma200']    = _smooth(audjpy, window=FOREX_MA200_WINDOW).reindex(close.index)
    out['audjpy_momentum'] = audjpy.pct_change(FOREX_MOMENTUM_WINDOW).reindex(close.index)

    # Signal 1: level vs MA200 (slow trend)
    out['sig_audjpy_vs_ma200'] = (
        audjpy_s.reindex(close.index) > out['audjpy_ma200']
    ).astype(float)

    # Signal 2: 63d momentum — independent of level; price can be above MA200
    # yet momentum decelerating, giving an early warning
    out['sig_audjpy_momentum'] = (
        _smooth(out['audjpy_momentum'], window=15) > 0
    ).astype(float)

    # Signal 3: NZD/JPY as independent carry confirmation — different high-yield
    # currency (NZD) vs safe-haven (JPY), so it's not driven by AUD-specific factors
    nzdjpy = close.get('NZDJPY=X')
    if nzdjpy is not None:
        nzdjpy_ma200               = _smooth(nzdjpy, window=FOREX_MA200_WINDOW)
        out['sig_nzdjpy_vs_ma200'] = (
            _smooth(nzdjpy, window=10).reindex(close.index) > nzdjpy_ma200.reindex(close.index)
        ).astype(float)
    else:
        out['sig_nzdjpy_vs_ma200'] = out['sig_audjpy_vs_ma200']  # fallback

    # Level signal weighted 2× — it's the primary driver; momentum and NZD/JPY are confirmations.
    # Score range: 0-4 (matches FOREX_CARRY_BULL_ENTRY=3.0, FOREX_CARRY_BEAR_ENTRY=1.0)
    score = (
        out['sig_audjpy_vs_ma200'] * 2 +
        out['sig_audjpy_momentum'] * 1 +
        out['sig_nzdjpy_vs_ma200'] * 1
    )
    out['carry_bull_score'] = score
    out['carry_regime'] = _score_to_regime(
        score, FOREX_CARRY_BULL_ENTRY, FOREX_CARRY_BEAR_ENTRY,
        'Carry Bull', 'Carry Neutral', 'Carry Bear',
    )
    return out


# ── 4. EM stress ──────────────────────────────────────────────────────────────

def compute_dm_em_regime(close: pd.DataFrame) -> pd.DataFrame:
    """
    Average z-score of USD vs EM currency pairs.
    High z → USD dominant → EM stress. Low z → EM relief.
    """
    out = pd.DataFrame(index=close.index)

    em_pairs = ['USDMXN=X', 'USDINR=X', 'USDCNY=X', 'USDZAR=X',
                'USDTRY=X', 'USDBRL=X', 'USDKRW=X']
    scores = []
    for pair in em_pairs:
        if pair not in close.columns:
            continue
        s = close[pair].dropna()
        roll_mean = s.rolling(FOREX_EM_NORM_WINDOW, min_periods=FOREX_EM_NORM_WINDOW // 2).mean()
        roll_std  = s.rolling(FOREX_EM_NORM_WINDOW, min_periods=FOREX_EM_NORM_WINDOW // 2).std()
        scores.append(((s - roll_mean) / roll_std.replace(0, np.nan)).reindex(close.index))

    if not scores:
        out['em_regime'] = 'Unknown'
        return out

    out['em_stress']        = pd.concat(scores, axis=1).mean(axis=1)
    out['em_stress_smooth'] = _smooth(out['em_stress'], window=FOREX_SCORE_SMOOTH)

    # Asymmetric thresholds: stress spikes sharply, relief builds gradually
    out['em_regime'] = np.select(
        [out['em_stress_smooth'] >  FOREX_EM_STRESS_THRESHOLD,
         out['em_stress_smooth'] <  FOREX_EM_RELIEF_THRESHOLD],
        ['EM Stress', 'EM Relief'],
        default='EM Neutral',
    )
    return out


# ── 5. JPY safe-haven ─────────────────────────────────────────────────────────

def compute_jpy_regime(close: pd.DataFrame) -> pd.DataFrame:
    """
    JPY Bull = risk-off (safe-haven demand). 3 signals:
    USDJPY below MA200, USDJPY momentum falling, carry (AUD/JPY) falling.
    """
    out = pd.DataFrame(index=close.index)

    usdjpy = close.get('USDJPY=X')
    audjpy = close.get('AUDJPY=X')
    if usdjpy is None:
        out['jpy_regime'] = 'Unknown'
        return out

    out['usdjpy_ma200']     = _smooth(usdjpy, window=FOREX_MA200_WINDOW)
    usdjpy_mom              = usdjpy.dropna().pct_change(21).reindex(close.index)
    out['usdjpy_momentum']  = _smooth(usdjpy_mom, window=10)

    out['sig_jpy_vs_ma']    = (usdjpy < out['usdjpy_ma200']).astype(float)  # inverted — JPY strong
    out['sig_jpy_rising']   = (out['usdjpy_momentum'] < 0).astype(float)    # USDJPY falling = JPY rising

    if audjpy is not None:
        audjpy_mom = audjpy.dropna().pct_change(21).reindex(close.index)
        out['audjpy_momentum_jpy'] = _smooth(audjpy_mom, window=10)
        out['sig_carry_falling']   = (out['audjpy_momentum_jpy'] < 0).astype(float)
    else:
        out['sig_carry_falling'] = np.nan

    sig_cols = [c for c in ['sig_jpy_vs_ma', 'sig_jpy_rising', 'sig_carry_falling']
                if out[c].notna().any()]
    score = out[sig_cols].sum(axis=1)
    out['jpy_bull_score'] = score
    out['jpy_regime'] = _score_to_regime(
        score, FOREX_JPY_BULL_ENTRY, FOREX_JPY_BEAR_ENTRY,
        'JPY Bull', 'JPY Neutral', 'JPY Bear',
    )
    return out


# ── 6. European stress ────────────────────────────────────────────────────────

def compute_europe_regime(close: pd.DataFrame, strength_df: pd.DataFrame) -> pd.DataFrame:
    """
    EURCHF is the best EU stress indicator — falls during EU crises as capital flees to CHF.
    3 signals → Europe Stable / Europe Neutral / Europe Stress.
    """
    out = pd.DataFrame(index=close.index)

    eurchf = close.get('EURCHF=X')
    eurusd = close.get('EURUSD=X')
    if eurchf is None and eurusd is None:
        out['europe_regime'] = 'Unknown'
        return out

    sigs = {}
    if eurchf is not None:
        out['eurchf_ma200']     = _smooth(eurchf, window=FOREX_MA200_WINDOW)
        sigs['sig_eurchf_ok']   = (eurchf > out['eurchf_ma200']).astype(float)

    if eurusd is not None:
        out['eurusd_ma200']     = _smooth(eurusd, window=FOREX_MA200_WINDOW)
        sigs['sig_eurusd_trend'] = (eurusd > out['eurusd_ma200']).astype(float)

    eur_str = strength_df.get('strength_eur', pd.Series(dtype=float))
    if eur_str.notna().any():
        sigs['sig_eur_strong'] = (eur_str.reindex(close.index) > 0).astype(float)

    if not sigs:
        out['europe_regime'] = 'Unknown'
        return out

    for col, s in sigs.items():
        out[col] = s

    score = pd.concat(list(sigs.values()), axis=1).sum(axis=1)
    out['europe_bull_score'] = score
    out['europe_regime'] = _score_to_regime(
        score, FOREX_EUROPE_BULL_ENTRY, FOREX_EUROPE_BEAR_ENTRY,
        'Europe Stable', 'Europe Neutral', 'Europe Stress',
    )
    return out


# ── Master forex regime ───────────────────────────────────────────────────────

def compute_forex_regime(df_forex: pd.DataFrame) -> pd.DataFrame:
    """
    Master regime driven by pure price action on two instruments: DXY and AUD/JPY.
    Sub-regimes (USD, carry, EM, JPY, Europe) are computed but kept as metadata only
    — they feed the multi-asset regimes later rather than driving the master label.

    Master regime states:
      Risk On — carry working and dollar not dominant (carry_bull & ~usd_bull)
      Stress  — dollar surging and carry off       (usd_bull & carry_bear)
      Neutral — everything else
    """
    close = df_forex['Close'].unstack('Ticker')

    strength_df  = compute_currency_strength(close)
    usd_df       = compute_usd_regime(close)
    carry_df     = compute_carry_regime(close)
    em_df        = compute_dm_em_regime(close)
    jpy_df       = compute_jpy_regime(close)
    europe_df    = compute_europe_regime(close, strength_df)

    forex_df = pd.concat(
        [strength_df, usd_df, carry_df, em_df, jpy_df, europe_df],
        axis=1,
    )

    # ── Price-action master regime — DXY × AUD/JPY ───────────────────────────
    # Score-then-hysteresis fights forex's zero-sum nature (USD up = EUR/AUD down
    # simultaneously, so signals always partially cancel). Two instruments,
    # two MA200s, percentage buffers as hysteresis — simpler and more stable.
    dxy    = close.get('DX-Y.NYB')
    audjpy = close.get('AUDJPY=X')

    if dxy is None or audjpy is None:
        forex_df['forex_regime'] = 'Unknown'
    else:
        dxy_ma200    = dxy.rolling(FOREX_MA200_WINDOW, min_periods=FOREX_MA200_WINDOW // 2).mean()
        audjpy_ma200 = audjpy.rolling(FOREX_MA200_WINDOW, min_periods=FOREX_MA200_WINDOW // 2).mean()

        # 20-day price smooth — eliminates daily MA crossings without a second parameter
        dxy_sm    = dxy.rolling(20, min_periods=10).mean()
        audjpy_sm = audjpy.rolling(20, min_periods=10).mean()

        forex_df['dxy_smooth']    = dxy_sm
        forex_df['audjpy_smooth'] = audjpy_sm

        # 2% buffer on DXY, 1% on AUD/JPY (FX pairs have tighter percentage ranges)
        usd_bull   = dxy_sm    > dxy_ma200    * 1.02
        carry_bull = audjpy_sm > audjpy_ma200 * 1.01
        carry_bear = audjpy_sm < audjpy_ma200 * 0.99

        # Risk On fires whenever carry is working and dollar is not dominant —
        # this covers both USD Neutral + carry bull and USD Bear + carry bull.
        master_raw = pd.Series(
            np.select(
                [carry_bull & ~usd_bull,
                 usd_bull   &  carry_bear],
                ['Risk On', 'Stress'],
                default='Neutral',
            ),
            index=close.index,
        )

        coverage   = close.notna().sum(axis=1)
        master_raw = master_raw.where(coverage >= FOREX_MIN_PAIRS, 'Unknown')

        forex_df['forex_regime'] = _confirm(
            master_raw,
            {'Risk On': 2, 'Neutral': 1, 'Stress': 0, 'Unknown': -1},
            FOREX_MASTER_CONFIRM_WINDOW, FOREX_MASTER_CONFIRM_MIN,
        )

    # ── snapshot print ────────────────────────────────────────────────────────
    valid    = forex_df.dropna(subset=['usd_bull_score'])
    latest   = valid.iloc[-1]
    date_str = valid.index[-1].strftime('%Y-%m-%d')

    print(f'\n=== Forex Regime Snapshot ({date_str}) ===\n')
    print(f'  {"Forex Regime:":<28} {latest["forex_regime"]}')
    print(f'  {"USD Regime:":<28} {latest["usd_regime"]}  (score {latest["usd_bull_score"]:.1f}/3)')
    print(f'  {"Carry Regime:":<28} {latest["carry_regime"]}  (score {latest["carry_bull_score"]:.1f}/3)')
    print(f'  {"EM Regime:":<28} {latest["em_regime"]}')
    print(f'  {"JPY Regime:":<28} {latest["jpy_regime"]}')
    print(f'  {"Europe Regime:":<28} {latest["europe_regime"]}')

    print('\n  --- Forex Regime Distribution (full history) ---')
    for state in ['Risk On', 'Neutral', 'Stress', 'Unknown']:
        pct = (forex_df['forex_regime'] == state).mean()
        print(f'  {state:<14} {pct:>5.1%}  {"#"*int(pct*40)}')
    print()

    return forex_df
