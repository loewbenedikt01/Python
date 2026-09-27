import pandas as pd
import numpy as np
from Finance.Regimes.config import (
    CRYPTO_BULL_ENTRY, CRYPTO_BULL_EXIT,
    CRYPTO_BEAR_ENTRY, CRYPTO_BEAR_EXIT,
    CRYPTO_MA_WINDOW, CRYPTO_BTC_SMOOTH, CRYPTO_ETH_SMOOTH,
    CRYPTO_MOMENTUM_WINDOW, CRYPTO_MOMENTUM_SMOOTH,
    CRYPTO_ETH_BTC_MA, CRYPTO_SCORE_SMOOTH,
    CRYPTO_CONFIRM_WINDOW, CRYPTO_CONFIRM_MIN,
)

_BTC = 'BTC-USD'
_ETH = 'ETH-USD'

_CRYPTO_REGIME_TO_NUM = {'Crypto Bull': 2, 'Crypto Neutral': 1, 'Crypto Bear': 0, 'Unknown': -1}
_CRYPTO_NUM_TO_REGIME = {v: k for k, v in _CRYPTO_REGIME_TO_NUM.items()}


def _smooth(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    mp = min_periods or max(1, window // 2)
    return series.dropna().rolling(window, min_periods=mp).mean().reindex(series.index)


def _apply_crypto_hysteresis(score_series: pd.Series) -> list:
    regimes = []
    state   = None

    for score in score_series:
        if pd.isna(score):
            regimes.append('Unknown')
            continue

        if state is None:
            if   score >= CRYPTO_BULL_ENTRY: state = 'Crypto Bull'
            elif score <= CRYPTO_BEAR_ENTRY: state = 'Crypto Bear'
            else:                            state = 'Crypto Neutral'

        elif state == 'Crypto Bull':
            # Must drop below BULL_EXIT to leave bull — lands in Neutral, never jumps to Bear.
            # Prevents a single-candle BTC crash from skipping straight to Bear.
            if score < CRYPTO_BULL_EXIT:
                state = 'Crypto Neutral'

        elif state == 'Crypto Bear':
            # Mirror: must rise above BEAR_EXIT to leave bear — lands in Neutral.
            if score > CRYPTO_BEAR_EXIT:
                state = 'Crypto Neutral'

        else:  # Crypto Neutral
            if   score >= CRYPTO_BULL_ENTRY: state = 'Crypto Bull'
            elif score <= CRYPTO_BEAR_ENTRY: state = 'Crypto Bear'

        regimes.append(state)

    return regimes


def compute_crypto_regime(df_crypto: pd.DataFrame) -> pd.DataFrame:
    close      = df_crypto['Close'].unstack('Ticker')
    crypto_df  = pd.DataFrame(index=close.index)

    btc = close[_BTC]
    eth = close[_ETH]

    # ── 1. BTC trend: smoothed price vs 200-day MA ────────────────────────────
    # Smooth before comparing to avoid triggering on daily noise near the MA line.
    btc_ma200        = _smooth(btc, window=CRYPTO_MA_WINDOW)
    btc_smooth       = _smooth(btc, window=CRYPTO_BTC_SMOOTH).reindex(close.index)
    crypto_df['btc_ma200']  = btc_ma200
    crypto_df['btc_smooth'] = btc_smooth
    crypto_df['sig_btc_uptrend'] = (btc_smooth > btc_ma200).astype(float)

    # ── 2. BTC momentum: 63-day return, smoothed ─────────────────────────────
    # Captures the direction of the trend, not just its level. A falling BTC that's
    # still above MA200 (early-cycle deterioration) gets flagged here before sig 1 flips.
    btc_return = btc.dropna().pct_change(CRYPTO_MOMENTUM_WINDOW).reindex(close.index)
    btc_return_smooth = _smooth(btc_return, window=CRYPTO_MOMENTUM_SMOOTH)
    crypto_df['btc_return_smooth'] = btc_return_smooth
    crypto_df['sig_btc_momentum']  = (btc_return_smooth > 0).astype(float)

    # ── 3. ETH uptrend: confirmation that the #2 asset agrees ────────────────
    eth_ma200  = _smooth(eth, window=CRYPTO_MA_WINDOW)
    eth_smooth = _smooth(eth, window=CRYPTO_ETH_SMOOTH).reindex(close.index)
    crypto_df['eth_ma200']       = eth_ma200
    crypto_df['sig_eth_uptrend'] = (eth_smooth > eth_ma200).astype(float)

    # ── 4. ETH/BTC ratio trend: alt-season / risk-on signal ──────────────────
    # ETH outperforming BTC (ratio rising above its MA) signals capital rotating
    # into higher-beta assets — characteristic of late-bull / alt-season.
    # ETH underperforming (ratio falling) signals risk-off rotation back to BTC.
    eth_btc_ratio       = (eth / btc).reindex(close.index)
    eth_btc_ma          = _smooth(eth_btc_ratio, window=CRYPTO_ETH_BTC_MA)
    crypto_df['eth_btc_ratio']    = eth_btc_ratio
    crypto_df['eth_btc_ma']       = eth_btc_ma
    crypto_df['sig_eth_leading']  = (eth_btc_ratio > eth_btc_ma).astype(float)

    # ── 5. Bull score (0–4), smoothed, then regime ───────────────────────────
    sig_cols = ['sig_btc_uptrend', 'sig_btc_momentum', 'sig_eth_uptrend', 'sig_eth_leading']
    crypto_df['crypto_bull_score'] = crypto_df[sig_cols].sum(axis=1)

    # Smooth the integer score before hysteresis — crypto flips binary signals far
    # faster than equities; a 10-day mean score creates gradual transitions instead
    # of step-changes at every BTC candle that crosses the MA200 line.
    crypto_df['crypto_score_smooth'] = _smooth(
        crypto_df['crypto_bull_score'], window=CRYPTO_SCORE_SMOOTH
    )

    raw_regime = _apply_crypto_hysteresis(crypto_df['crypto_score_smooth'])
    crypto_df['crypto_regime_raw'] = raw_regime

    # Confirmation window: require CRYPTO_CONFIRM_MIN of the last CRYPTO_CONFIRM_WINDOW
    # days to agree before the regime is considered confirmed.
    # Crypto trades 24/7 so there are no NaN weekend rows — no trading-day subset needed.
    regime_num    = crypto_df['crypto_regime_raw'].map(_CRYPTO_REGIME_TO_NUM).astype(float)
    confirmed_num = (
        regime_num
        .rolling(CRYPTO_CONFIRM_WINDOW, min_periods=CRYPTO_CONFIRM_MIN)
        .apply(lambda x: x[-1] if (x == x[-1]).sum() >= CRYPTO_CONFIRM_MIN else np.nan, raw=True)
        .bfill()   # fill leading NaNs at series start before history is long enough
        .ffill()   # carry last confirmed regime forward through any remaining gaps
    )
    crypto_df['crypto_regime'] = confirmed_num.map(_CRYPTO_NUM_TO_REGIME).fillna('Unknown')

    # ── 6. Snapshot print ────────────────────────────────────────────────────
    valid    = crypto_df.dropna(subset=['btc_ma200', 'eth_btc_ma'])
    latest   = valid.iloc[-1]
    date_str = valid.index[-1].strftime('%Y-%m-%d')

    print(f'\n=== Crypto Regime Snapshot ({date_str}) ===\n')
    print(f'  {"Crypto Regime:":<28} {latest["crypto_regime"]}')
    print(f'  {"Bull Score (raw):":<28} {int(latest["crypto_bull_score"])}/4')
    print(f'  {"Bull Score (smooth):":<28} {latest["crypto_score_smooth"]:.2f}/4')
    print(f'  {"BTC vs MA200:":<28} {"Above" if latest["sig_btc_uptrend"] else "Below"} '
          f'(smooth {latest["btc_smooth"]:,.0f} vs MA {latest["btc_ma200"]:,.0f})')
    print(f'  {"BTC 63d Momentum:":<28} {"Positive" if latest["sig_btc_momentum"] else "Negative"} '
          f'({latest["btc_return_smooth"]:+.1%})')
    print(f'  {"ETH vs MA200:":<28} {"Above" if latest["sig_eth_uptrend"] else "Below"}')
    print(f'  {"ETH/BTC Ratio:":<28} {"ETH Leading" if latest["sig_eth_leading"] else "BTC Dominant"}')

    print('\n  --- Signal Breakdown ---')
    for col, label in [
        ('sig_btc_uptrend',  'BTC Above MA200'),
        ('sig_btc_momentum', 'BTC 63d Return Positive'),
        ('sig_eth_uptrend',  'ETH Above MA200'),
        ('sig_eth_leading',  'ETH/BTC Above MA50'),
    ]:
        print(f'    {"YES" if latest[col] == 1.0 else "NO ":<4} {label}')

    print('\n  --- Crypto Regime Distribution (full history) ---')
    for regime in ['Crypto Bull', 'Crypto Neutral', 'Crypto Bear']:
        pct = (crypto_df['crypto_regime'] == regime).mean()
        print(f'  {regime:<16} {pct:>5.1%}  {"#"*int(pct*40)}')
    print()

    return crypto_df
