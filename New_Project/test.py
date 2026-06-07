import pandas as pd
import numpy as np
import vectorbt as vbt

# ── Config ────────────────────────────────────────────────────────────────────
PARQUET    = r'C:\Users\benel\Coding\Python\Database\US_equities.parquet'
START_YEAR = 2001
END_YEAR   = 2025

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_parquet(PARQUET, columns=['Date', 'Ticker', 'Close', 'Adj Close'])
df['Date'] = pd.to_datetime(df['Date'])
df = df[(df['Date'].dt.year >= START_YEAR) & (df['Date'].dt.year <= END_YEAR)]

# Use Adj Close — consistently split+dividend adjusted throughout history.
# Raw Close mixes adjusted (pre-update) and unadjusted (post-update) prices
# which makes old shares look artificially cheap and inflates returns.
close = df.pivot_table(index='Date', columns='Ticker', values='Adj Close')
close = close.sort_index()

# ── Strategy 1: Moving-average crossover ──────────────────────────────────────
fast     = close.rolling(10).mean()
slow     = close.rolling(50).mean()
entries1 = fast > slow
exits1   = fast < slow

# ── Strategy 2: RSI mean-reversion ───────────────────────────────────────────
rsi      = vbt.RSI.run(close, window=14)
entries2 = rsi.rsi_below(30)
exits2   = rsi.rsi_above(70)

# -- Strategy 3: Bollinger Bands -----
bb       = vbt.BBANDS.run(close, window=20, alpha=2)
# bb.lower / bb.upper have a MultiIndex column (ticker + params); use .values
# to compare shapes directly and re-attach the original labels.
entries3 = pd.DataFrame(close.values < bb.lower.values,
                         index=close.index, columns=close.columns)
exits3   = pd.DataFrame(close.values > bb.upper.values,
                         index=close.index, columns=close.columns)

# ── Backtest ──────────────────────────────────────────────────────────────────
# group_by=True  → treat all tickers as one combined portfolio
# freq='D'       → daily frequency so Sharpe/Sortino/Calmar are calculated
# fees=0.001     → 0.1% commission per trade
pf1 = vbt.Portfolio.from_signals(
    close, entries1, exits1,
    init_cash=100_000, fees=0.00,
    group_by=True, cash_sharing=True,
    freq='D'
)
pf2 = vbt.Portfolio.from_signals(
    close, entries2, exits2,
    init_cash=100_000, fees=0.00,
    group_by=True, cash_sharing=True,
    freq='D'
)
pf3 = vbt.Portfolio.from_signals(
    close, entries3, exits3,
    init_cash=100_000, fees=0.00,
    group_by=True, cash_sharing=True,
    freq='D'
)

print("=== Strategy 1: MA Crossover ===")
print(pf1.stats())

print("\n=== Strategy 2: RSI Mean-Reversion ===")
print(pf2.stats())

print("\n=== Strategy 3: Bollinger Bands ===")
print(pf3.stats())
