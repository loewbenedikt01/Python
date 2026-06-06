import backtrader as bt
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
PARQUET     = r'C:\Users\benel\Coding\Python\Database\US_equities.parquet'
_GITHUB_API = 'https://api.github.com/repos/fja05680/sp500/contents/'
TOP_N       = 10
CASH        = 100_000.0
START_YEAR  = 2001   # first year we trade (requires 1999 & 2000 as baseline)
END_YEAR    = 2025


# ── Pre-computation (outside backtrader, fast pandas) ─────────────────────────
def load_year_end_constituents():
    """Point-in-time S&P 500 membership at each Dec-31."""
    resp = requests.get(_GITHUB_API)
    resp.raise_for_status()
    files = resp.json()
    csv = next(f for f in files if f['name'].startswith('S&P 500') and f['name'].endswith('.csv'))
    df = pd.read_csv(csv['download_url'])
    df['date'] = pd.to_datetime(df['date'])

    out = {}
    for year in range(START_YEAR - 2, END_YEAR + 1):
        rows = df[df['date'].dt.year == year]
        if rows.empty:
            continue
        tickers = {t.strip().replace('.', '-') for t in rows.iloc[-1]['tickers'].split(',')}
        out[year] = tickers
    return out


def compute_year_end_prices(parquet_path):
    """Last available close per (year, ticker) — the Dec-31 snapshot."""
    df = pd.read_parquet(parquet_path, columns=['Date', 'Ticker', 'Close'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    idx = df.groupby(['Year', 'Ticker'])['Date'].idxmax()
    snap = df.loc[idx].pivot(index='Year', columns='Ticker', values='Close')
    return snap


def compute_selections(constituents, snap, top_n):
    """
    For year Y: rank all tickers in the Dec-31 (Y-1) S&P 500 by their
    return over year Y-1 (Dec-31 Y-2 -> Dec-31 Y-1). Pick top N.
    No lookahead: only data available at Dec-31 Y-1 is used.
    """
    selections = {}
    for year in range(START_YEAR, END_YEAR + 1):
        prev, prev2 = year - 1, year - 2
        if prev not in constituents or prev not in snap.index or prev2 not in snap.index:
            continue

        universe = list(constituents[prev])
        avail = [t for t in universe if t in snap.columns]

        p1 = snap.loc[prev, avail]
        p0 = snap.loc[prev2, avail]
        rets = ((p1 - p0) / p0).dropna()

        top = rets.nlargest(top_n).index.tolist()
        if top:
            selections[year] = top
    return selections


# ── Backtrader strategy ───────────────────────────────────────────────────────
class AnnualMomentumStrategy(bt.Strategy):
    params = dict(selections=None, top_n=10)

    def __init__(self):
        self.data_map         = {d._name: d for d in self.datas}
        self.current_year     = None
        self.pending_year     = None
        self.closing_count    = 0
        self.last_rebal_value = None  # portfolio value at the previous rebalancing
        self.entry_prices     = {}    # ticker -> avg cost basis for the current holding

    def log(self, msg):
        print(f"{self.datas[0].datetime.date(0)} | {msg}")

    def _portfolio_value(self):
        """Cash + mark-to-market of positions, treating NaN prices as 0."""
        val = self.broker.getcash()
        for d in self.datas:
            size = self.getposition(d).size
            if size:
                price = d.close[0]
                val += size * (price if not np.isnan(price) else 0)
        return val

    def notify_order(self, order):
        if order.status == order.Completed:
            side = 'BUY ' if order.isbuy() else 'SELL'
            pct_str = ''
            if order.issell():
                entry = self.entry_prices.get(order.data._name)
                if entry and entry > 0:
                    pct = (order.executed.price / entry - 1) * 100
                    pct_str = f"  ({pct:+.1f}%)"
            self.log(f"  {side} {order.data._name:>8s}  "
                     f"{order.executed.size:>6.0f} shares @ ${order.executed.price:>8.2f}{pct_str}")

        if order.issell() and order.status in [order.Completed, order.Cancelled,
                                                order.Rejected, order.Expired]:
            self.closing_count = max(0, self.closing_count - 1)

    def next(self):
        year = self.datas[0].datetime.date(0).year

        # Year rolled over: Phase 1 — close all current positions
        if year != self.current_year and year in self.p.selections:
            current_val = self._portfolio_value()
            if self.last_rebal_value is not None:
                yoy = (current_val / self.last_rebal_value - 1) * 100
                self.log(f"=== Rebalancing {year} | YoY: {yoy:+.1f}% | "
                         f"target: {self.p.selections[year]} ===")
            else:
                self.log(f"=== Rebalancing {year} | target: {self.p.selections[year]} ===")
            self.last_rebal_value = current_val

            self.current_year = year
            self.pending_year = year

            for d in self.datas:
                pos = self.getposition(d).size
                if not pos:
                    continue
                # Only submit a close when the current price is valid; NaN-price
                # positions would queue a ghost order that executes a year later.
                if not np.isnan(d.close[0]) and d.close[0] > 0:
                    self.entry_prices[d._name] = self.getposition(d).price
                    self.close(data=d)
                    self.closing_count += 1
                # Else: position is stranded (delisted / no data); skip it.

        # Phase 2 — all close orders settled, now buy
        elif self.pending_year and self.closing_count == 0:
            self._buy(self.pending_year)
            self.pending_year = None

    def _buy(self, year):
        target = self.p.selections.get(year, [])
        available = [
            t for t in target
            if t in self.data_map
            and not np.isnan(self.data_map[t].close[0])
            and self.data_map[t].close[0] > 0
        ]
        if not available:
            self.log(f"  No tradeable tickers available for {year}")
            return

        alloc = self.broker.getcash() * 0.10
        for ticker in available:
            d = self.data_map[ticker]
            size = int(alloc / d.close[0])
            if size > 0:
                self.buy(data=d, size=size)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Loading S&P 500 year-end constituents...')
    constituents = load_year_end_constituents()

    print('Computing year-end price snapshots...')
    snap = compute_year_end_prices(PARQUET)

    print('Selecting top-N momentum picks per year...')
    selections = compute_selections(constituents, snap, TOP_N)

    print('\nYear-by-year selections (prior-year return in brackets):')
    for y, picks in sorted(selections.items()):
        prev, prev2 = y - 1, y - 2
        parts = []
        for t in picks:
            if t in snap.columns and prev in snap.index and prev2 in snap.index:
                p1, p0 = snap.loc[prev, t], snap.loc[prev2, t]
                if pd.notna(p0) and p0 > 0:
                    parts.append(f"{t}({(p1-p0)/p0:+.0%})")
        print(f'  {y}: {", ".join(parts)}')

    all_tickers = sorted({t for picks in selections.values() for t in picks})
    print(f'\nLoading {len(all_tickers)} unique tickers into backtrader...')

    df = pd.read_parquet(PARQUET)
    df['Date'] = pd.to_datetime(df['Date'])

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(CASH)
    cerebro.broker.setcommission(commission=0.001)

    fromdate = datetime(START_YEAR, 1, 1)
    todate   = datetime(END_YEAR, 12, 31)

    # Use actual trading days from the parquet as the shared timeline.
    # bdate_range includes weekday holidays (e.g. Jan 2 observed New Year) as NaN
    # bars, which causes close orders to silently fail and Phase 2 to never fire.
    master_idx = (df[(df['Date'] >= fromdate) & (df['Date'] <= todate)]
                   ['Date'].drop_duplicates().sort_values())

    n_loaded = 0
    for ticker in all_tickers:
        tdf = (df[df['Ticker'] == ticker]
               [['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
               .set_index('Date')
               .sort_index())
        if len(tdf) < 50:
            continue
        # Align to trading-day calendar; pre-IPO rows are NaN (filtered in _buy)
        tdf = tdf.reindex(master_idx)
        feed = bt.feeds.PandasData(
            dataname=tdf, fromdate=fromdate, todate=todate, openinterest=-1)
        cerebro.adddata(feed, name=ticker)
        n_loaded += 1

    print(f'Loaded {n_loaded} feeds.\n')
    cerebro.addstrategy(AnnualMomentumStrategy, selections=selections, top_n=TOP_N)

    start_val = cerebro.broker.getvalue()
    print(f'Starting Portfolio Value: ${start_val:>12,.2f}')
    strat = cerebro.run()[0]
    # Use NaN-safe valuation: stranded delisted positions counted at $0
    end_val = strat._portfolio_value()
    print(f'Final Portfolio Value:    ${end_val:>12,.2f}')
    print(f'Total Return:             {(end_val / start_val - 1):+.1%}')
