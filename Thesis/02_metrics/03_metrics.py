
"""
Definition of Crisis Periods

Used for Metrics calculation and timewindow definition
of different crisis periods.
6 Main-Crises
3 Sub-Crises of GFC
"""


TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR       = 12

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _annualized_factor(freq: str) -> int:
    if freq == 'D':
        return TRADING_DAYS_PER_YEAR
    elif freq == 'M':
        return MONTHS_PER_YEAR
    raise ValueError(f"freq must be 'D' or 'M', got '{freq}'")


# ─────────────────────────────────────────────────────────────────────────────
# B. RETURN METRICS
# ─────────────────────────────────────────────────────────────────────────────
def cumulative_return(log_returns: pd.Series) -> float:
    """Total cumulative return: exp(sum(log_returns)) - 1"""
    return float(np.exp(log_returns.sum()) - 1)


def annualized_return(log_returns: pd.Series, freq: str = 'D') -> float:
    """Geometric annualised return accounting for compounding."""
    n     = len(log_returns)
    ann_f = _annualized_factor(freq)
    return float(np.exp((log_returns.sum() / n) * ann_f) - 1)


# ─────────────────────────────────────────────────────────────────────────────
# C. RISK METRICS
# ─────────────────────────────────────────────────────────────────────────────
def annualized_volatility(log_returns: pd.Series, freq: str = 'D') -> float:
    """Annualised standard deviation of log returns (ddof=1)."""
    return float(log_returns.std(ddof=1) * np.sqrt(_annualized_factor(freq)))


def maximum_drawdown(price_series: pd.Series) -> float:
    """Worst peak-to-trough decline in the price series. Returns a negative number."""
    rolling_max = price_series.cummax()
    return float(((price_series / rolling_max) - 1).min())


def drawdown_series(price_series: pd.Series) -> pd.Series:
    """Full time series of drawdown at each date (always <= 0)."""
    rolling_max = price_series.cummax()
    return (price_series / rolling_max) - 1


def max_drawdown_duration(price_series: pd.Series) -> int:
    """
    Longest consecutive run of trading days spent below a prior peak.
    Measures how long an investor stays underwater at its worst stretch.
    """
    rolling_max       = price_series.cummax()
    is_at_peak        = price_series >= rolling_max
    underwater_groups = is_at_peak.cumsum()
    durations         = price_series.groupby(underwater_groups).cumcount()
    return int(durations.max())


def recovery_duration(price_series: pd.Series) -> int:
    """
    Trading days from the global MDD trough back to the pre-trough peak level.
    If the series never recovers, returns remaining length from trough.
    """
    rolling_max  = price_series.cummax()
    drawdown     = (price_series / rolling_max) - 1
    mdd_date     = drawdown.idxmin()
    peak_at_mdd  = rolling_max.loc[mdd_date]
    recovery_ser = price_series.loc[mdd_date:]
    hits         = recovery_ser[recovery_ser >= peak_at_mdd]
    if hits.empty:
        return len(recovery_ser)
    return int(len(price_series.loc[mdd_date:hits.index[0]]) - 1)


def value_at_risk(log_returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Historical VaR at the given confidence level.
    Returns the daily loss threshold exceeded only (1 - confidence_level)% of days.
    """
    return float(log_returns.quantile(1 - confidence_level))


def conditional_value_at_risk(log_returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Expected Shortfall (CVaR / ES) at the given confidence level.
    Average return on the worst (1 - confidence_level)% of days.
    """
    var = value_at_risk(log_returns, confidence_level)
    return float(log_returns[log_returns <= var].mean())


def ulcer_index(price_series: pd.Series) -> float:
    """
    Ulcer Index = sqrt(mean(drawdown²)).
    Measures both depth and duration of being underwater.
    All drawdowns computed from the running peak (standard definition).
    Higher values = more painful drawdown profile.
    """
    rolling_max = price_series.cummax()
    dd          = (price_series / rolling_max) - 1
    return float(np.sqrt((dd ** 2).mean()))


# ─────────────────────────────────────────────────────────────────────────────
# D. RISK-ADJUSTED RATIOS
# ─────────────────────────────────────────────────────────────────────────────
def sharpe_ratio(log_returns: pd.Series, rfr: float = 0.0,
                 freq: str = 'D') -> float:
    """
    Annualised Sharpe Ratio.
    (annualized_return - rfr) / annualized_volatility.
    Uses full return distribution (not downside-only).
    """
    ann_ret = annualized_return(log_returns, freq)
    ann_vol = annualized_volatility(log_returns, freq)
    return float((ann_ret - rfr) / ann_vol) if ann_vol != 0 else np.nan


def sortino_ratio(log_returns: pd.Series, rfr: float = 0.0,
                  freq: str = 'D') -> float:
    """
    Annualised Sortino Ratio.
    (annualized_return - rfr) / downside_deviation.
    Downside deviation = sqrt(mean(min(r, 0)²)) * sqrt(ann_factor).
    Penalises only negative returns, not upside volatility.
    """
    ann_ret      = annualized_return(log_returns, freq)
    ann_f        = _annualized_factor(freq)
    downside_vol = np.sqrt((np.minimum(log_returns, 0) ** 2).mean()) * np.sqrt(ann_f)
    return float((ann_ret - rfr) / downside_vol) if downside_vol != 0 else np.nan


def calmar_ratio(log_returns: pd.Series, price_series: pd.Series,
                 freq: str = 'D') -> float:
    """
    Calmar Ratio = annualized_return / |maximum_drawdown|.
    Reward per unit of the worst observed drawdown.
    """
    ann_ret = annualized_return(log_returns, freq)
    mdd     = abs(maximum_drawdown(price_series))
    return float(ann_ret / mdd) if mdd != 0 else np.nan


def omega_ratio(log_returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Omega Ratio = sum(gains above threshold) / sum(losses below threshold).
    Probability-weighted ratio of gains to losses. Value > 1 = net positive.
    """
    gains  = log_returns[log_returns >  threshold].sum()
    losses = abs(log_returns[log_returns < threshold].sum())
    return float(gains / losses) if losses != 0 else np.inf


# ─────────────────────────────────────────────────────────────────────────────
# E. BENCHMARK-RELATIVE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Beta — sensitivity of portfolio returns to benchmark returns.
    Formula: cov(p, b) / var(b).
    Beta > 1: amplifies market moves. Beta < 1: dampens them.
    """
    aligned_p, aligned_b = portfolio_returns.align(benchmark_returns, join='inner')
    cov = np.cov(aligned_p.values, aligned_b.values)
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else np.nan


def alpha(portfolio_returns: pd.Series, benchmark_returns: pd.Series,
          rfr: float = 0.0, freq: str = 'D') -> float:
    """
    Jensen's Alpha — annualised excess return vs CAPM expectation.
    Formula: ann_port - (rfr + beta * (ann_bench - rfr)).
    Positive alpha = portfolio outperforms its risk-adjusted benchmark expectation.
    """
    b      = beta(portfolio_returns, benchmark_returns)
    ann_p  = annualized_return(portfolio_returns, freq)
    ann_bm = annualized_return(benchmark_returns, freq)
    return float(ann_p - (rfr + b * (ann_bm - rfr)))
