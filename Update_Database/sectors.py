import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import os

ticker_sectors = {
    # --- Global Sector ETFs ---
    'IXC': 'iShares Global Energy ETF',
    'IXG': 'iShares Global Financials ETF',
    'IXN': 'iShares Global Technology ETF',
    'IXP': 'iShares Global Communication Services ETF',
    'EXI': 'iShares Global Industrials ETF',
    'MXI': 'iShares Global Materials ETF',
    'KXI': 'iShares Global Consumer Staples ETF',
    'RXI': 'iShares Global Consumer Discretionary ETF',
    'IXJ': 'iShares Global Healthcare ETF',
    'JXI': 'iShares Global Utilities ETF',
    'REET': 'iShares Global Real Estate ETF',

    # --- European Sector ETFs (STOXX Europe 600) ---
    'EXH1.DE': 'iShares STOXX Europe 600 Energy UCITS ETF',
    'EXH3.DE': 'iShares STOXX Europe 600 Banks UCITS ETF',
    'EXH7.DE': 'iShares STOXX Europe 600 Financial Services UCITS ETF',
    'EXH8.DE': 'iShares STOXX Europe 600 Technology UCITS ETF',
    'EXH9.DE': 'iShares STOXX Europe 600 Telecommunications UCITS ETF',
    'EXHA.DE': 'iShares STOXX Europe 600 Industrial Goods & Services UCITS ETF',
    'EXH2.DE': 'iShares STOXX Europe 600 Chemicals UCITS ETF',
    'EXHC.DE': 'iShares STOXX Europe 600 Food & Beverage UCITS ETF',
    'EXHF.DE': 'iShares STOXX Europe 600 Retail UCITS ETF',
    'EXH4.DE': 'iShares STOXX Europe 600 Health Care UCITS ETF',
    'EXH5.DE': 'iShares STOXX Europe 600 Utilities UCITS ETF',

    # --- Canada Sector ETFs (S&P/TSX Capped) ---
    'XEG.TO': 'iShares S&P/TSX Capped Energy Index ETF',
    'XFN.TO': 'iShares S&P/TSX Capped Financials Index ETF',
    'XIT.TO': 'iShares S&P/TSX Capped Information Technology Index ETF',
    'XMA.TO': 'iShares S&P/TSX Capped Materials Index ETF',
    'XRE.TO': 'iShares S&P/TSX Capped REIT Index ETF',
    'XUT.TO': 'iShares S&P/TSX Capped Utilities Index ETF',
    'XST.TO': 'iShares S&P/TSX Capped Consumer Staples Index ETF',
    'XCD.TO': 'iShares S&P/TSX Capped Consumer Discretionary Index ETF',
    'ZIN.TO': 'BMO Equal Weight Industrials Index ETF',

    # --- APAC Sectors ---
    'OZF.AX': 'iShares Core S&P/ASX 200 Financials Sector ETF',
    'OZR.AX': 'iShares Core S&P/ASX 200 Resources Sector ETF',
    'MVR.AX': 'VanEck Vectors Australian Resources ETF',
    'SLF.AX': 'SPDR S&P/ASX 200 Listed Property Fund',
    'CHIK': 'Global X MSCI China Information Technology ETF',
    'CHIF': 'Global X MSCI China Financials ETF',
    'CHIE': 'Global X MSCI China Energy ETF',
    'CHIS': 'Global X MSCI China Consumer Staples ETF',
    'DXJ': 'WisdomTree Japan Hedged Equity ETF'
}