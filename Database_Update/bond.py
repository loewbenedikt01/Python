import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import os

ticker_bonds = {
    # --- Pure CBOE Yield Indices ---
    '^IRX': 'CBOE 13-Week Treasury Bill Yield',
    '^FVX': 'CBOE 5-Year Treasury Note Yield',
    '^TNX': 'CBOE 10-Year Treasury Note Yield',
    '^TYX': 'CBOE 30-Year Treasury Bond Yield',

    # --- Liquid Duration ETFs (Execution Proxies) ---
    'BIL': 'SPDR Bloomberg 1-3 Month T-Bill ETF',
    'SHV': 'iShares Short Treasury Bond ETF',
    'SHY': 'iShares 1-3 Year Treasury Bond ETF',
    'IEI': 'iShares 3-7 Year Treasury Bond ETF',
    'IEF': 'iShares 7-10 Year Treasury Bond ETF',
    'TLH': 'iShares 10-20 Year Treasury Bond ETF',
    'TLT': 'iShares 20+ Year Treasury Bond ETF',
    'GOVT': 'iShares Core U.S. Treasury ETF',
    'TIP': 'iShares TIPS Bond ETF',
    'STIP': 'iShares Short-Term TIPS ETF',
    'EDV': 'Vanguard Extended Duration Treasury ETF',
    'TMF': 'Direxion Daily 20+ Year Treasury Bull 3X Shares',
    'TMV': 'Direxion Daily 20+ Year Treasury Bear 3X Shares'
}