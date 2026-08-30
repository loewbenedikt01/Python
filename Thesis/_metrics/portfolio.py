
"""
Portfolio construction — the shared mechanic every model feeds into.


"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    DATABASE_PATH,
    MAX_HOLDINGS,
    TRANSACTION_COST_BPS,
    START_DATE,
    END_DATE,
)
from universe import tickers as UNIVERSE

_UNI_YEARS = sorted(UNIVERSE)


@dataclass
class PortfolioResult:
    log_returns:       pd.Series      # daily portfolio log returns, net of costs
    weights:           pd.DataFrame   # daily effective (drifted) weights, date x ticker
    turnover:          pd.Series      # per rebalance date
    transaction_costs: pd.Series      # per rebalance date
    target_weights:    pd.DataFrame   # per rebalance date, after universe filter + renorm
    rebalance_status:  pd.Series      # per rebalance date: "ok" | "carried_forward"


# ----
# Inputs
# ----
def load_prices(database_path=DATABASE_PATH) -> pd.DataFrame:
    """
    Adjusted close, wide (date x ticker).
    """
    px = pd.read_parquet(database_path)["Adj Close"].sort_index()
    px.index = pd.to_datetime(px.index)
    return px


def _universe_for(year: int) -> set[str]:
    """
    The 20 tickers investable during calendar 'year' = universe of year-1.
    """
    uni_year = min(max(year - 1, _UNI_YEARS[0]), _UNI_YEARS[-1])
    return {t for t, _ in UNIVERSE[uni_year]}


def universe_for(year: int) -> list[str]:
    """
    Sorted list of the tickers investable during calendar 'year' — the end-of-
    (year-1) market-cap top 20 from universe.py.  Models use this to build their
    target-weight matrix.
    """
    return sorted(_universe_for(year))


def _month_first_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    First trading day of each month present in 'index'.
    """
    s = index.to_series()
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).first().to_numpy())


# Which calendar months carry a rebalance, anchored at the January start
# (matching the universe's end-of-(year-1) timing).
REBALANCE_MONTHS = {
    "Monthly":   set(range(1, 13)),
    "Quarterly": {1, 4, 7, 10},
    "Yearly":    {1},
}

# If the month's first trading day has thin price coverage, step forward up to
# this many trading days (within the same month) to a better-covered day.
REBALANCE_FALLBACK_DAYS = 3


def _rebalance_dates(
    index: pd.DatetimeIndex,
    frequency: str,
    valid_price: pd.DataFrame | None = None,
) -> pd.DatetimeIndex:
    """
    First trading day of each qualifying month for 'frequency'.  When
    `valid_price` is given, a candidate day with fewer priced tickers than a day
    within the next REBALANCE_FALLBACK_DAYS trading days is bumped to that
    better-covered day (ties keep the earlier date).
    """
    firsts = _month_first_dates(index)
    firsts = firsts[firsts.month.isin(REBALANCE_MONTHS[frequency])]
    if valid_price is None:
        return firsts

    coverage = valid_price.sum(axis=1)
    out = []
    for d in firsts:
        pos = index.get_loc(d)
        window = [
            index[pos + k]
            for k in range(REBALANCE_FALLBACK_DAYS + 1)
            if pos + k < len(index) and index[pos + k].month == d.month
        ]
        out.append(max(window, key=lambda x: (coverage.get(x, 0), -index.get_loc(x))))
    return pd.DatetimeIndex(out)


# ----
# Target weights per rebalance date
# ----
def _resolve_targets(
    target_weights: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    valid_price: pd.DataFrame,
) -> pd.DataFrame:
    tw = target_weights.copy()
    tw.index = pd.to_datetime(tw.index)
    tw = tw.sort_index()

    resolved: list[pd.Series] = []
    status: dict[pd.Timestamp, str] = {}
    prev: pd.Series | None = None
    for d in rebalance_dates:
        pos = tw.index.searchsorted(d, side="right") - 1
        row = pd.Series(dtype=float)
        if pos >= 0:
            uni = _universe_for(d.year)
            priced = valid_price.loc[d]
            row = tw.iloc[pos].dropna()
            row = row[[c for c in row.index if c in uni]]
            row = row[row.abs() > 0]
            row = row[[t for t in row.index if bool(priced.get(t, False))]]
            if len(row) > MAX_HOLDINGS:
                row = row[row.abs().sort_values(ascending=False).index[:MAX_HOLDINGS]]

        usable = bool(len(row)) and row.sum() != 0
        if usable:
            prev = row / row.sum()
        if prev is None:
            continue
        resolved.append(prev.rename(d))
        status[d] = "ok" if usable else "carried_forward"
        if not usable:
            print(f"  [portfolio] ERROR {d.date()}: no usable target weights "
                  f"(universe / price / precomputation missing) - holding previous allocation")

    tgt = pd.DataFrame(resolved).fillna(0.0)
    tgt = tgt.loc[:, tgt.abs().sum() > 0]
    return tgt, pd.Series(status, name="rebalance_status").sort_index()


# ----
# Simulation
# ----
def build_portfolio(
    target_weights: pd.DataFrame,
    *,
    frequency: str,
    prices: pd.DataFrame | None = None,
    start=None,
    end=None,
    transaction_cost_bps: float = TRANSACTION_COST_BPS,
) -> PortfolioResult:
    """
    Parameters
    ----------
    target_weights
        date x ticker DataFrame of desired weights at each rebalance.  NaN /
        absent entry = pre-computation missing for that name that month.
    frequency
        "Monthly" / "Quarterly" / "Yearly" — the rebalance calendar.
    prices
        Adjusted-close panel (date x ticker); defaults to load_prices().
    start, end
        Optional clamp on the simulation window.

    Returns
    -------
    PortfolioResult
    """
    px = load_prices() if prices is None else prices.copy()
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    asset_ret = px.pct_change()

    first_model = pd.to_datetime(target_weights.index).min()
    start = pd.Timestamp(start) if start is not None else max(first_model, pd.Timestamp(START_DATE))
    end = pd.Timestamp(end) if end is not None else min(px.index.max(), pd.Timestamp(END_DATE))

    cal = px.loc[start:end].index
    rebs = _rebalance_dates(cal, frequency, px.notna())
    rebs = rebs[(rebs >= start) & (rebs <= end)]

    tgt, rebalance_status = _resolve_targets(target_weights, rebs, px.notna())
    reb_list = list(tgt.index)
    if len(reb_list) < 2:
        raise ValueError("need at least two usable rebalance dates")

    cost_rate = transaction_cost_bps / 1e4
    ret_parts:  list[pd.Series] = []
    w_parts:    list[pd.DataFrame] = []
    turnover:   dict[pd.Timestamp, float] = {}
    txn_cost:   dict[pd.Timestamp, float] = {}

    # first rebalance: deploy from cash on d0's close
    d0 = reb_list[0]
    held = tgt.loc[d0].copy()
    turnover[d0] = float(held.abs().sum())                 # == 1.0
    txn_cost[d0] = turnover[d0] * cost_rate
    ret_parts.append(pd.Series([-txn_cost[d0]], index=[d0]))

    for j, d_j in enumerate(reb_list):
        d_next = reb_list[j + 1] if j + 1 < len(reb_list) else cal[-1]
        block = cal[(cal > d_j) & (cal <= d_next)]
        if len(block) == 0:
            continue

        cols = held.index[held.abs() > 0]
        sub = asset_ret.loc[block, cols].fillna(0.0)
        contrib = (1.0 + sub).cumprod().mul(held[cols], axis=1)   # value per name, base = weight
        pv = contrib.sum(axis=1)                                  # book value, base 1.0 at d_j
        port_r = pv / pv.shift(1).fillna(1.0) - 1.0
        w_daily = contrib.div(pv, axis=0)                         # effective weights each day

        drifted = pd.Series(0.0, index=tgt.columns)
        drifted.loc[cols] = w_daily.iloc[-1].to_numpy()

        if j + 1 < len(reb_list):                                 # rebalance at d_next close
            w_new = tgt.loc[d_next]
            to = float((w_new - drifted).abs().sum())
            turnover[d_next] = to
            txn_cost[d_next] = to * cost_rate
            port_r.loc[d_next] -= txn_cost[d_next]
            held = w_new.copy()
        else:
            held = drifted

        ret_parts.append(port_r)
        w_parts.append(w_daily.reindex(columns=tgt.columns).fillna(0.0))

    port_simple = pd.concat(ret_parts).sort_index()
    port_simple = port_simple[~port_simple.index.duplicated()]

    weights = pd.concat(w_parts).sort_index()
    weights = weights[~weights.index.duplicated()].reindex(columns=tgt.columns).fillna(0.0)
    for d in reb_list:                                            # show post-trade book on rebalance days
        weights.loc[d] = tgt.loc[d]
    weights = weights.sort_index()

    return PortfolioResult(
        log_returns=np.log1p(port_simple).rename("portfolio"),
        weights=weights,
        turnover=pd.Series(turnover).sort_index().rename("turnover"),
        transaction_costs=pd.Series(txn_cost).sort_index().rename("transaction_costs"),
        target_weights=tgt,
        rebalance_status=rebalance_status,
    )


# ----
# Smoke test — equal-weight over each year's universe
# ----
if __name__ == "__main__":
    all_tickers = sorted({t for lst in UNIVERSE.values() for t, _ in lst})
    rows = {}
    for year in range(pd.Timestamp(START_DATE).year, pd.Timestamp(END_DATE).year + 1):
        uni = universe_for(year)
        rows[pd.Timestamp(year, 1, 1)] = pd.Series(1.0 / len(uni), index=uni)
    equal_weight = pd.DataFrame(rows).T.reindex(columns=all_tickers)

    for frequency in REBALANCE_MONTHS:
        res = build_portfolio(equal_weight, frequency=frequency)
        print(f"{frequency:10s} {len(res.log_returns):5d} days  "
              f"cum {np.expm1(res.log_returns.sum()):8.2%}  "
              f"avg turnover {res.turnover.iloc[1:].mean():.3f}  "
              f"carried_forward {int(res.rebalance_status.eq('carried_forward').sum())}")
