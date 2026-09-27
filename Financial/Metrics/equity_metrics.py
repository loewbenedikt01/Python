import numpy as np
import pandas as pd

SMA_50   = 50       # Simple Moving Average
SMA_200  = 200      # Simple Moving Average
EMA      = 20       # Exponential Moving Average
WMA      = 20       # Weighted Moving Average
ATR      = 14       # Average True Range
VOL      = 20
UL       = 14       # Ulcer Index (volatility indicator)
RSI      = 14       # Relative Strength Index
ROC      = 14       # Rate of Change (Momentum)
MACD_FAST     = 12
MACD_SLOW     = 26
MACD_SIGNAL   = 9
CCI      = 20       # Commodity Channel Index (trend direction and strength in stocks)
PSAR     = 0.2      # Parabolic SAR (Stop and Reverse) AF = 0.02 (increasing by 0.02, max = 0.20)
K        = 14       # %K = fast stochastic oscillator
D        = 3        # %D = slow stochastic oscillator (apply a 3 day smooting average to the fast %K)
R        = 14        # %R = reverse %K
AO       = 34       # Awesome Oscillator
BOLL_WINDOW   = 20
BOLL_STD      = 2
RANGE_WINDOW  = 252
MFI      = 14       # Money Flow Index




def _by_ticker(series):
    return series.groupby(level='Ticker', group_keys=False)


def _roll_mean(series, window, min_periods=None):
    return _by_ticker(series).transform(
        lambda s: s.rolling(window, min_periods=min_periods or window).mean()
    )


def _roll_std(series, window, min_periods=None):
    return _by_ticker(series).transform(
        lambda s: s.rolling(window, min_periods=min_periods or window).std()
    )


def _roll_sum(series, window, min_periods=None):
    return _by_ticker(series).transform(
        lambda s: s.rolling(window, min_periods=min_periods or window).sum()
    )


def _roll_max(series, window, min_periods=1):
    return _by_ticker(series).transform(lambda s: s.rolling(window, min_periods=min_periods).max())


def _roll_min(series, window, min_periods=1):
    return _by_ticker(series).transform(lambda s: s.rolling(window, min_periods=min_periods).min())


def _ewm(series, span):
    return _by_ticker(series).transform(lambda s: s.ewm(span=span, adjust=False).mean())


def _wilder_ewm(series, window):
    return _by_ticker(series).transform(
        lambda s: s.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    )


def _pct_change(series):
    return _by_ticker(series).transform(lambda s: s.pct_change(fill_method=None))


def _log_return(series):
    return _by_ticker(series).transform(lambda s: np.log(s / s.shift(1)))


def _shift(series, periods=1):
    return _by_ticker(series).transform(lambda s: s.shift(periods))


def _cumsum(series):
    return _by_ticker(series).transform(lambda s: s.cumsum())


def _rolling_weighted(series, window, weights):
    weights = np.asarray(weights, dtype=float)
    denom = weights.sum()
    return _by_ticker(series).transform(
        lambda s: s.rolling(window, min_periods=window).apply(lambda x: np.dot(x, weights) / denom, raw=True)
    )


def _wma(series, window):
    return _rolling_weighted(series, window, np.arange(1, window + 1))


def _swma(series):
    # Symmetric (palindromic) weights are what make a WMA "symmetric" — fixed
    # 4-bar [1,2,2,1] shape regardless of window length.
    return _rolling_weighted(series, 4, [1, 2, 2, 1])


def _rsi(close, window=RSI):
    def per_ticker(s):
        delta    = s.diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    return _by_ticker(close).transform(per_ticker)


def _true_range(high, low, close):
    prev_close = _shift(close, 1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _atr(high, low, close, window):
    return _wilder_ewm(_true_range(high, low, close), window)


def _ulcer_index(close, window):
    def per_ticker(s):
        roll_max = s.rolling(window, min_periods=window).max()
        drawdown_pct = 100 * (s - roll_max) / roll_max
        return np.sqrt((drawdown_pct ** 2).rolling(window, min_periods=window).mean())
    return _by_ticker(close).transform(per_ticker)


def _roc(close, period):
    return _by_ticker(close).transform(lambda s: 100 * (s - s.shift(period)) / s.shift(period))


def _cci(typical_price, window):
    def per_ticker(s):
        sma = s.rolling(window, min_periods=window).mean()
        mad = s.rolling(window, min_periods=window).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        return (s - sma) / (0.015 * mad.replace(0, np.nan))
    return _by_ticker(typical_price).transform(per_ticker)


def _psar(high, low, af_step=0.02, af_max=PSAR):
    def per_ticker(group):
        h = group['High'].to_numpy()
        l = group['Low'].to_numpy()
        n = len(h)
        psar = np.empty(n)
        if n == 0:
            return pd.Series(psar, index=group.index)

        psar[0] = l[0]
        bull = True
        af = af_step
        ep = h[0]

        for i in range(1, n):
            psar[i] = psar[i - 1] + af * (ep - psar[i - 1])
            prior_low = l[i - 2] if i > 1 else l[i - 1]
            prior_high = h[i - 2] if i > 1 else h[i - 1]

            if bull:
                psar[i] = min(psar[i], l[i - 1], prior_low)
                if l[i] < psar[i]:
                    bull, psar[i], ep, af = False, ep, l[i], af_step
                elif h[i] > ep:
                    ep, af = h[i], min(af + af_step, af_max)
            else:
                psar[i] = max(psar[i], h[i - 1], prior_high)
                if h[i] > psar[i]:
                    bull, psar[i], ep, af = True, ep, h[i], af_step
                elif l[i] < ep:
                    ep, af = l[i], min(af + af_step, af_max)

        return pd.Series(psar, index=group.index)

    hl = pd.DataFrame({'High': high, 'Low': low})
    return hl.groupby(level='Ticker', group_keys=False).apply(per_ticker)


def _stochastic(high, low, close, k_window, d_window):
    roll_low  = _roll_min(low, k_window, min_periods=k_window)
    roll_high = _roll_max(high, k_window, min_periods=k_window)
    percent_k = 100 * (close - roll_low) / (roll_high - roll_low)
    percent_d = _roll_mean(percent_k, d_window)
    return percent_k, percent_d


def _williams_r(high, low, close, window):
    roll_high = _roll_max(high, window, min_periods=window)
    roll_low  = _roll_min(low, window, min_periods=window)
    return -100 * (roll_high - close) / (roll_high - roll_low)


def _awesome_oscillator(high, low, slow_window, fast_window=5):
    median_price = (high + low) / 2
    return _roll_mean(median_price, fast_window) - _roll_mean(median_price, slow_window)


def _pvt(close, volume):
    return _cumsum(_pct_change(close) * volume)


def _obv(close, volume):
    direction = _by_ticker(close).transform(lambda s: np.sign(s.diff()))
    return _cumsum((direction * volume).fillna(0))


def _mfi(typical_price, volume, window):
    raw_flow   = typical_price * volume
    price_diff = _by_ticker(typical_price).transform(lambda s: s.diff())
    pos_flow   = raw_flow.where(price_diff > 0, 0.0)
    neg_flow   = raw_flow.where(price_diff < 0, 0.0)
    pos_sum    = _roll_sum(pos_flow, window)
    neg_sum    = _roll_sum(neg_flow, window)
    money_ratio = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + money_ratio))


def add_all_indicators(df, price_col='Close'):
    """Add technical indicator columns to a (Date, Ticker)-indexed equities
    DataFrame. Every indicator is computed per ticker along its own trading
    calendar (via groupby transform) — never against a shared/unstacked date
    index, since different markets' holiday calendars would otherwise inject
    phantom NaN rows into each other's rolling windows."""
    close  = df[price_col]
    high   = df['High']
    low    = df['Low']
    volume = df['Volume']
    typical_price = (high + low + close) / 3

    ret            = _pct_change(close)
    log_ret        = _log_return(close)
    sma50          = _roll_mean(close, SMA_50)
    sma200         = _roll_mean(close, SMA_200)
    above_sma200   = (close > sma200).where(sma200.notna())
    ema20          = _ewm(close, EMA)
    wma20          = _wma(close, WMA)
    swma           = _swma(close)
    vol20          = _roll_std(ret, VOL) * np.sqrt(252)
    rsi14          = _rsi(close, RSI)
    atr14          = _atr(high, low, close, ATR)
    ulcer          = _ulcer_index(close, UL)
    roc            = _roc(close, ROC)
    cci            = _cci(typical_price, CCI)
    psar           = _psar(high, low)
    percent_k, percent_d = _stochastic(high, low, close, K, D)
    percent_r      = _williams_r(high, low, close, R)
    ao             = _awesome_oscillator(high, low, AO)
    pvt            = _pvt(close, volume)
    obv            = _obv(close, volume)
    mfi14          = _mfi(typical_price, volume, MFI)

    ema_fast    = _ewm(close, MACD_FAST)
    ema_slow    = _ewm(close, MACD_SLOW)
    macd        = ema_fast - ema_slow
    macd_signal = _ewm(macd, MACD_SIGNAL)
    macd_hist   = macd - macd_signal

    boll_mid   = _roll_mean(close, BOLL_WINDOW)
    boll_std   = _roll_std(close, BOLL_WINDOW)
    boll_upper = boll_mid + BOLL_STD * boll_std
    boll_lower = boll_mid - BOLL_STD * boll_std

    high_252 = _roll_max(close, RANGE_WINDOW)
    low_252  = _roll_min(close, RANGE_WINDOW)
    drawdown = close / high_252 - 1

    df = df.copy()
    df['Return_1D']       = ret
    df['Log_Return']      = log_ret
    df['SMA_50']          = sma50
    df['SMA_200']         = sma200
    df['Above_SMA200']    = above_sma200
    df['EMA_20']          = ema20
    df['WMA_20']          = wma20
    df['SWMA']            = swma
    df['Volatility_20D']  = vol20
    df['RSI_14']          = rsi14
    df['ATR_14']          = atr14
    df['Ulcer_Index']     = ulcer
    df['ROC']             = roc
    df['CCI']             = cci
    df['PSAR']            = psar
    df['Stoch_K']         = percent_k
    df['Stoch_D']         = percent_d
    df['Williams_R']      = percent_r
    df['Awesome_Osc']     = ao
    df['PVT']             = pvt
    df['OBV']             = obv
    df['MFI_14']          = mfi14
    df['MACD']            = macd
    df['MACD_Signal']     = macd_signal
    df['MACD_Hist']       = macd_hist
    df['Bollinger_Upper'] = boll_upper
    df['Bollinger_Lower'] = boll_lower
    df['High_252D']       = high_252
    df['Low_252D']        = low_252
    df['Drawdown']        = drawdown
    return df
