"""
main.py
=======
Master entry point. Controls what gets computed and plotted.
Toggle the RUN_* flags to enable/disable each regime.
"""

from concurrent.futures import ThreadPoolExecutor

from config import START_DATE, END_DATE
from data   import load_all

from Simple.equity    import compute_equity_regime
from Simple.bond      import compute_bond_regime
from Simple.forex     import compute_forex_regime
from Simple.commodity import compute_commodity_regime
from Simple.crypto    import compute_crypto_regime

from Multi_Model.growth        import compute_growth_regime
from Multi_Model.inflation     import compute_inflation_regime
from Multi_Model.liquidity     import compute_liquidity_regime
from Multi_Model.risk_appetite import compute_risk_appetite_regime

from Hidden_Markov_Model.hmm import compute_hmm_regime

from plots import (
    plot_equity_regime,
    plot_bond_regime,
    plot_crypto_regime,
    plot_forex_regime,
)

# ── Toggle what to run ────────────────────────────────────────────────────────
RUN_EQUITY    = False
RUN_BOND      = False
RUN_FOREX     = True
RUN_COMMODITY = False
RUN_CRYPTO    = False

RUN_GROWTH    = False
RUN_INFLATION = False
RUN_LIQUIDITY = False
RUN_RISK      = False

RUN_HMM       = False

PLOT          = True   # set False for headless / batch runs

# ── Load all data once ────────────────────────────────────────────────────────
data = load_all()

# ── Single asset regimes (parallel) ──────────────────────────────────────────
futures = {}
with ThreadPoolExecutor() as pool:
    if RUN_EQUITY:    futures['equity']    = pool.submit(compute_equity_regime,    data['equities'])
    if RUN_BOND:      futures['bond']      = pool.submit(compute_bond_regime,      data['bonds'])
    if RUN_FOREX:     futures['forex']     = pool.submit(compute_forex_regime,     data['forex'])
    if RUN_COMMODITY: futures['commodity'] = pool.submit(compute_commodity_regime, data['commodities'])
    if RUN_CRYPTO:    futures['crypto']    = pool.submit(compute_crypto_regime,    data['crypto'])

equity_df    = futures['equity'].result()    if 'equity'    in futures else None
bond_df      = futures['bond'].result()      if 'bond'      in futures else None
forex_df     = futures['forex'].result()     if 'forex'     in futures else None
commodity_df = futures['commodity'].result() if 'commodity' in futures else None
crypto_df    = futures['crypto'].result()    if 'crypto'    in futures else None

# ── Clip all single regimes to analysis window ────────────────────────────────
def _clip(df):
    return df.loc[START_DATE:END_DATE] if df is not None and not df.empty else df

equity_df    = _clip(equity_df)
bond_df      = _clip(bond_df)
forex_df     = _clip(forex_df)
commodity_df = _clip(commodity_df)
crypto_df    = _clip(crypto_df)

# ── Multi asset regimes (sequential — depend on single regime outputs) ────────
growth_df    = compute_growth_regime(equity_df, bond_df, commodity_df, data['macro']) \
               if RUN_GROWTH    else None
inflation_df = compute_inflation_regime(commodity_df, bond_df, data['macro']) \
               if RUN_INFLATION else None
liquidity_df = compute_liquidity_regime(crypto_df, bond_df, forex_df) \
               if RUN_LIQUIDITY else None
risk_df      = compute_risk_appetite_regime(equity_df, bond_df, forex_df,
                                             commodity_df, crypto_df, data['sentiment']) \
               if RUN_RISK      else None

# ── HMM ───────────────────────────────────────────────────────────────────────
hmm_df = compute_hmm_regime(equity_df, bond_df, forex_df, commodity_df, crypto_df) \
         if RUN_HMM else None

# ── Plots ─────────────────────────────────────────────────────────────────────
if PLOT:
    if equity_df is not None and not equity_df.empty:
        plot_equity_regime(equity_df)
    if bond_df is not None and not bond_df.empty:
        plot_bond_regime(bond_df)
    if crypto_df is not None and not crypto_df.empty:
        plot_crypto_regime(crypto_df)
    if forex_df is not None and not forex_df.empty:
        plot_forex_regime(forex_df)
