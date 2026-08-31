"""
Feature definitions for the ML portfolio models (xgb / rf / lstm).

`features(db, asof)` returns one row per ticker of 56 stock-level features,
each ranked cross-sectionally on the as-of date (rank in [0, 1]; a missing
value maps to 0.5, the neutral rank).

Everything is derived from the OHLCV panel in ``database.parquet`` (plus the
``^GSPC`` series, used only as the market factor for beta / idiosyncratic
features), so there is no look-ahead beyond `asof`: the caller passes the
rebalance date and the frame is sliced to ``:asof`` first.

Feature groups
--------------
momentum / trend .... multi-horizon returns, 12-1 momentum, acceleration,
                      distance to 52w high/low, moving-average ratios, OLS
                      trend t-stat / R^2, information discreteness
reversal ............ 1w / 1m reversal, Bollinger position, MA z-score, RSI,
                      overnight vs intraday decomposition
volatility .......... close-to-close vol (1/3/6/12m), Parkinson & Garman-Klass
                      range vol, downside vol, realised skew / kurtosis,
                      MAX / MIN daily return, vol-of-vol
market risk ......... CAPM beta, downside beta, idiosyncratic vol, residual
                      (idiosyncratic) momentum, average pairwise correlation
liquidity ........... Amihud illiquidity (1/6m), dollar volume level & trend,
                      volume shock, price-volume correlation
tail / drawdown ..... historical VaR / CVaR (95%), max drawdown, Ulcer index,
                      days since the trailing 52w high
"""

import warnings

import numpy as np
import pandas as pd

try:
    from config import DATABASE_PATH
except ImportError:                       # allow standalone import
    DATABASE_PATH = None

# ----
# Constants
# ----
TRADING_DAYS = 252
ANN          = np.sqrt(TRADING_DAYS)
MKT_TICKER   = "^GSPC"
VIX_TICKER   = "^VIX"
MIN_BARS     = 300                        # trailing bars the longest window needs
EPS          = 1e-9


# ----
# Loading / plumbing
# ----

def load_db(database_path=DATABASE_PATH) -> pd.DataFrame:
    """Full OHLCV panel, wide with a (field, ticker) column MultiIndex."""
    db = pd.read_parquet(database_path).sort_index()
    db.index = pd.to_datetime(db.index)
    return db


def _adjust_ohlc(db: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """
    Split/dividend-consistent OHLC: scale raw O/H/L by the Adj Close / Close
    ratio and use Adj Close itself for the close.
    """
    close  = db["Close"]
    adj    = db["Adj Close"]
    factor = (adj / close).where(close > 0)
    return db["Open"] * factor, db["High"] * factor, db["Low"] * factor, adj


# ----
# Feature groups  (each returns dict[str, pd.Series] indexed by ticker)
# ----

def _momentum(c: pd.DataFrame, ret: pd.DataFrame) -> dict:
    f: dict = {}
    last = c.iloc[-1]

    for name, n in [("mom_1w", 5), ("mom_2w", 10), ("mom_1m", 21),
                    ("mom_3m", 63), ("mom_6m", 126), ("mom_9m", 189)]:
        f[name] = c.pct_change(n).iloc[-1]

    f["mom_12_1"]     = c.iloc[-22] / c.iloc[-253] - 1
    f["mom_12_1_lag"] = c.iloc[-43] / c.iloc[-274] - 1
    f["mom_chg_12_1"] = f["mom_12_1"] - f["mom_12_1_lag"]

    cr = ret.fillna(0.0).cumsum()
    f["mom_accel_3m"] = (cr.iloc[-1] - cr.iloc[-64]) - (cr.iloc[-64] - cr.iloc[-127])

    w = c.tail(TRADING_DAYS)
    f["dist_52w_high"] = last / w.max() - 1
    f["dist_52w_low"]  = last / w.min() - 1

    f["ma_ratio_50_200"] = c.tail(50).mean()  / c.tail(200).mean() - 1
    f["ma_ratio_20_100"] = c.tail(20).mean()  / c.tail(100).mean() - 1
    f["px_vs_200dma"]    = last / c.tail(200).mean() - 1

    # OLS of log price on time over the last 126 sessions -> trend strength
    y   = np.log(c.tail(126).to_numpy())
    T   = y.shape[0]
    x   = np.arange(T, dtype=float)
    xc  = x - x.mean()
    Sxx = float((xc ** 2).sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yc     = y - np.nanmean(y, axis=0)
        slope  = np.nansum(xc[:, None] * yc, axis=0) / Sxx
        resid  = yc - slope[None, :] * xc[:, None]
        dof    = np.maximum((~np.isnan(y)).sum(axis=0) - 2, 1)
        sigma  = np.sqrt(np.nansum(resid ** 2, axis=0) / dof)
        ss_res = np.nansum(resid ** 2, axis=0)
        ss_tot = np.nansum(yc ** 2, axis=0)
    f["trend_tstat_6m"] = pd.Series(slope / (sigma / np.sqrt(Sxx) + EPS), index=c.columns)
    f["trend_r2_6m"]    = pd.Series(1.0 - ss_res / (ss_tot + EPS), index=c.columns)

    # information discreteness (Da, Gurun & Warachka): smooth momentum persists
    r    = ret.tail(TRADING_DAYS)
    pos  = (r > 0).sum()
    neg  = (r < 0).sum()
    f["info_discreteness"] = np.sign(f["mom_12_1"]) * ((neg - pos) / (pos + neg + EPS))
    return f


def _reversal(c: pd.DataFrame, ret: pd.DataFrame, o: pd.DataFrame) -> dict:
    f: dict = {}
    f["rev_1w"] = -c.pct_change(5).iloc[-1]
    f["rev_1m"] = -c.pct_change(21).iloc[-1]

    p20 = c.tail(20)
    f["bb_position"] = (c.iloc[-1] - p20.mean()) / (2 * p20.std() + EPS)
    p50 = c.tail(50)
    f["ma50_zscore"] = (c.iloc[-1] - p50.mean()) / (p50.std() + EPS)

    d    = ret.tail(TRADING_DAYS)
    gain = d.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean().iloc[-1]
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean().iloc[-1]
    f["rsi_14"] = 100 - 100 / (1 + gain / (loss + EPS))

    overnight = (o / c.shift(1) - 1).tail(63)
    intraday  = (c / o - 1).tail(63)
    on_c, in_c = overnight.sum(), intraday.sum()
    f["overnight_ret_3m"] = on_c
    f["intraday_ret_3m"]  = in_c
    f["overnight_share"]  = on_c / (on_c.abs() + in_c.abs() + EPS)
    return f


def _volatility(ret: pd.DataFrame, o: pd.DataFrame, h: pd.DataFrame,
                l: pd.DataFrame, c: pd.DataFrame) -> dict:
    f: dict = {}
    for name, n in [("vol_1m", 21), ("vol_3m", 63), ("vol_6m", 126), ("vol_12m", 252)]:
        f[name] = ret.tail(n).std() * ANN
    f["vol_ratio"] = ret.tail(21).std() / (ret.tail(252).std() + EPS)

    hl = np.log(h / l)
    f["parkinson_1m"] = np.sqrt((hl.tail(21) ** 2).mean() / (4 * np.log(2))) * ANN
    f["parkinson_3m"] = np.sqrt((hl.tail(63) ** 2).mean() / (4 * np.log(2))) * ANN

    co = np.log(c / o)
    gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
    f["garman_klass_1m"] = np.sqrt(gk.tail(21).mean().clip(lower=0)) * ANN

    r6 = ret.tail(126)
    f["downside_vol_6m"] = r6.where(r6 < 0).std() * ANN
    f["ret_skew_6m"]     = r6.skew()
    f["ret_kurt_6m"]     = r6.kurt()

    f["max_ret_1m"] = ret.tail(21).max()          # Bali MAX effect
    f["min_ret_1m"] = ret.tail(21).min()

    f["vol_of_vol_6m"] = ret.rolling(21).std().tail(126).std()
    return f


def _market_risk(logret: pd.DataFrame, mkt_ret: pd.Series, ret: pd.DataFrame) -> dict:
    f: dict = {}
    idx = logret.columns

    Y = logret.tail(TRADING_DAYS).to_numpy()
    X = mkt_ret.tail(TRADING_DAYS).to_numpy()
    good = ~np.isnan(X)
    Y, X = Y[good], X[good]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xc      = X - X.mean()
        Sxx     = float((Xc ** 2).sum())
        beta    = np.nansum(Xc[:, None] * (Y - np.nanmean(Y, 0)), 0) / (Sxx + EPS)
        resid   = Y - (np.nanmean(Y, 0) + beta[None, :] * Xc[:, None])
        idio    = np.sqrt(np.nanmean(resid ** 2, 0)) * ANN
        # residual (idiosyncratic) momentum: cumulative residual up to t-21
        cum_res = np.nancumsum(resid, 0)
        resid_mom = cum_res[-22] if cum_res.shape[0] > 22 else np.full(Y.shape[1], np.nan)
        # downside beta: regression restricted to down-market days
        dn      = X < 0
        Xd      = X[dn] - X[dn].mean()
        beta_dn = np.nansum(Xd[:, None] * (Y[dn] - np.nanmean(Y[dn], 0)), 0) / ((Xd ** 2).sum() + EPS)

    f["beta_12m"]          = pd.Series(beta, idx)
    f["downside_beta_12m"] = pd.Series(beta_dn, idx)
    f["idio_vol_12m"]      = pd.Series(idio, idx)
    f["resid_mom_12_1"]    = pd.Series(resid_mom, idx)

    cc = ret.tail(60).corr()
    np.fill_diagonal(cc.values, np.nan)
    f["avg_corr_60d"] = cc.mean()
    return f


def _liquidity(ret: pd.DataFrame, c: pd.DataFrame, vol: pd.DataFrame) -> dict:
    f: dict = {}
    dollar = (c * vol).replace(0.0, np.nan)
    f["amihud_1m"] = (ret.abs().tail(21)  / dollar.tail(21)).mean()
    f["amihud_6m"] = (ret.abs().tail(126) / dollar.tail(126)).mean()
    f["dollar_vol_level"] = np.log(dollar.tail(21).mean())
    f["dollar_vol_trend"] = dollar.tail(21).mean() / (dollar.tail(126).mean() + EPS)
    f["volume_shock"]     = vol.tail(5).mean() / (vol.tail(60).mean() + EPS)
    f["pv_corr_1m"]       = ret.tail(21).corrwith(vol.pct_change().tail(21))
    return f


def _tail(ret: pd.DataFrame, c: pd.DataFrame) -> dict:
    f: dict = {}
    r = ret.tail(TRADING_DAYS)
    q05 = r.quantile(0.05)
    f["var_95_12m"]  = q05
    f["cvar_95_12m"] = r.where(r.le(q05)).mean()

    w  = c.tail(TRADING_DAYS)
    dd = w / w.cummax() - 1
    f["max_dd_12m"] = dd.min()
    f["ulcer_12m"]  = np.sqrt((dd ** 2).mean())
    f["days_since_high"] = len(w) - 1 - w.reset_index(drop=True).idxmax()
    return f


# ----
# Public API
# ----

def features(db: pd.DataFrame,
             asof=None,
             min_history: int = TRADING_DAYS,
             min_coverage: float = 0.9) -> pd.DataFrame:
    """
    One row of cross-sectionally ranked features per ticker, as of `asof`
    (default: last row of `db`).

    `db` is the wide (field, ticker) OHLCV panel from `load_db`.  Tickers with
    less than `min_coverage` of `min_history` recent closes are dropped; every
    feature is rank-transformed to [0, 1] across the surviving names, with
    missing values filled at the neutral rank 0.5.
    """
    if asof is not None:
        db = db.loc[:pd.Timestamp(asof)]
    if len(db) < MIN_BARS:
        raise ValueError(f"need >= {MIN_BARS} trailing rows, got {len(db)}")

    o_all, h_all, l_all, c_all = _adjust_ohlc(db)
    vol_all = db["Volume"]

    tickers = [t for t in c_all.columns if t not in (MKT_TICKER, VIX_TICKER)]
    recent  = c_all[tickers].tail(min_history)
    keep    = [t for t in tickers
               if recent[t].notna().sum() >= min_history * min_coverage]
    if len(keep) < 2:
        raise ValueError("fewer than two tickers with sufficient history")

    c   = c_all[keep]
    o   = o_all[keep]
    h   = h_all[keep]
    l   = l_all[keep]
    vol = vol_all[keep]

    ret    = c.pct_change()
    logret = np.log(c / c.shift(1))
    mkt_ret = np.log(db[("Adj Close", MKT_TICKER)] / db[("Adj Close", MKT_TICKER)].shift(1))

    cols: dict = {}
    cols.update(_momentum(c, ret))
    cols.update(_reversal(c, ret, o))
    cols.update(_volatility(ret, o, h, l, c))
    cols.update(_market_risk(logret, mkt_ret, ret))
    cols.update(_liquidity(ret, c, vol))
    cols.update(_tail(ret, c))

    feat = pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan)
    feat = feat.rank(pct=True).reindex(index=keep).fillna(0.5)
    feat.index.name = "ticker"
    return feat


def features_panel(db: pd.DataFrame, dates) -> pd.DataFrame:
    """
    Stack `features` over many as-of dates into a (date, ticker) MultiIndex
    frame -- the training matrix for the ML models.  Dates without enough
    history are skipped.
    """
    frames: dict = {}
    for d in pd.DatetimeIndex(dates):
        try:
            frames[d] = features(db, asof=d)
        except ValueError:
            continue
    if not frames:
        raise ValueError("no date had enough history")
    return pd.concat(frames, names=["date", "ticker"])


if __name__ == "__main__":
    _db = load_db()
    _f  = features(_db)
    print(f"{_f.shape[0]} tickers x {_f.shape[1]} features  (as of {_db.index[-1].date()})")
    print(_f.columns.tolist())
    print(_f.round(3).to_string())
