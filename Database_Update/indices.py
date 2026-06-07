import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import os

ticker_indices = {
    # === I. UNITED STATES BROAD MARKETS & STYLE MATRIX ===
    '^GSPC': 'S&P 500 Index',
    '^IXIC': 'NASDAQ Composite Index',
    '^NDX': 'NASDAQ-100 Index',
    '^DJI': 'Dow Jones Industrial Average',
    '^RUT': 'Russell 2000 Index',
    '^RUA': 'Russell 3000 Index',
    '^RUI': 'Russell 1000 Index',
    '^MID': 'S&P MidCap 400 Index',
    '^OEX': 'S&P 100 Index',
    '^NYA': 'NYSE Composite Index',
    '^RUJ': 'Russell 1000 Growth Index',
    '^RUO': 'Russell 1000 Value Index',
    '^RLG': 'Russell 1000 Growth Index (Alternative)',
    '^RLV': 'Russell 1000 Value Index (Alternative)',
    '^XAX': 'NYSE American Composite Index',

    # === II. GERMANY (DOMESTIC MATRIX) ===
    '^GDAXI': 'DAX Performance Index',
    '^MDAXI': 'MDAX Index',
    '^SDAXI': 'SDAX Index',
    '^TECDAX': 'TecDAX Index',
    '^GDAXIP': 'DAX Price Index',

    # === III. WESTERN & NORTHERN EUROPE ===
    '^STOXX50E': 'EURO STOXX 50 Index',
    '^FTSE': 'FTSE 100 Index',
    '^FTMC': 'FTSE 250 Index',
    '^FCHI': 'CAC 40 Index',
    '^AEX': 'AEX Index',
    '^IBEX': 'IBEX 35 Index',
    'FTSEMIB.MI': 'FTSE MIB Index',
    '^SSMI': 'Swiss Market Index (SMI)',
    '^ATX': 'Austrian Traded Index (ATX)',
    '^BFX': 'BEL 20 Index',
    '^OMXS30': 'OMX Stockholm 30 Index',
    '^OMXH25': 'OMX Helsinki 25 Index',
    '^OMXC20': 'OMX Copenhagen 20 Index',
    '^OSEAX': 'Oslo Børs All-Share Index',
    '^PSI20': 'PSI 20 Index',
    '^WIG20': 'WIG20 Index',
    '^ISEQ': 'ISEQ Overall Index',
    'BCEX.BR': 'Brussels Stock Exchange Index',

    # === IV. AMERICAS (EX-UNITED STATES) ===
    '^GSPTSE': 'S&P/TSX Composite Index',
    '^BVSP': 'IBOVESPA Index',
    '^MXX': 'S&P/BMV IPC Index',
    '^MERV': 'S&P Merval Index',
    '^IPSA': 'S&P IPSA Index',
    '^COLCAP': 'MSCI Colcap Index',
    '^GSPCVP': 'S&P 500 Volatility Index (Proxy)',

    # === V. ASIA-PACIFIC CORE & REGIONAL BENCHMARKS ===
    '^N225': 'Nikkei 225',
    '^TPX': 'TOPIX Index',
    '000001.SS': 'Shanghai Composite Index',
    '399001.SZ': 'Shenzhen Component Index',
    '000300.SS': 'CSI 300 Index',
    '^HSI': 'Hang Seng Index',
    '^HSCE': 'Hang Seng China Enterprises Index',
    '^TWII': 'Taiwan Capitalization Weighted Stock Index',
    '^TWOTCI': 'Taiwan OTC Exchange Index',
    '^KS11': 'KOSPI Composite Index',
    '^KS200': 'KOSPI 200 Index',
    '^BSESN': 'S&P BSE SENSEX',
    '^NSEI': 'NIFTY 50 Index',
    '^AXJO': 'S&P/ASX 200',
    '^AORD': 'All Ordinaries Index',
    '^NZ50': 'S&P/NZX 50 Index',
    '^STI': 'Straits Times Index',
    '^JKSE': 'IDX Composite',
    '^KLSE': 'FTSE Bursa Malaysia KLCI',
    '^SET.BK': 'SET Index',
    '^PSEI.MN': 'PSEi Index',
    '^VNINDEX.VN': 'VN-Index',
    '^CSE': 'CSE All-Share Index',
    '^KSE': 'KSE 100 Index',

    # === VI. MIDDLE EAST & AFRICA ===
    '^J203.JO': 'FTSE/JSE Africa All Share Index',
    '^TASI.SR': 'Tadawul All Share Index (TASI)',
    '^TA125.TA': 'TA-125 Index',
    '^QE': 'QE General Index',
    '^DFMGI': 'DFM General Index',
    '^ADI': 'FTSE ADX General Index',
    '^EGX30': 'EGX 30 Index',
    '^KWSE': 'All-Share Index (Boursa Kuwait)',

    # === VII. SYSTEMIC RISK & VOLATILITY GAUGES ===
    '^VIX': 'CBOE Volatility Index (VIX)',
    '^VXN': 'CBOE Nasdaq-100 Volatility Index',
    '^RVX': 'CBOE Russell 2000 Volatility Index',
    '^VVIX': 'CBOE VIX Volatility Index',
    '^VXEEM': 'CBOE Emerging Markets ETF Volatility Index',
    '^VXD': 'CBOE DJIA Volatility Index',

    # === VIII. CROSS-SECTIONAL INDUSTRY SECTORS ===
    '^SOX': 'PHLX Semiconductor Sector Index',
    '^XAU': 'PHLX Gold/Silver Sector Index',
    '^HGX': 'PHLX Housing Sector Index',
    '^OSX': 'PHLX Oil Service Sector Index',
    '^UTY': 'PHLX Utility Sector Index',
    '^TRAN': 'Dow Jones Transportation Average',
    '^XBD': 'NYSE Arca Securities Broker/Dealer Index',
    '^IXBK': 'NASDAQ Bank Index',
    '^IXFN': 'NASDAQ Financial-100 Index',
    '^IXTR': 'NASDAQ Transportation Index',
    '^IXHC': 'NASDAQ Healthcare Index',
    '^IXCO': 'NASDAQ Computer Index',
    '^DRG': 'NYSE Arca Pharmaceutical Index',

    # === IX. MACRO INTEREST RATE & FIXED INCOME YIELDS ===
    '^IRX': 'CBOE 13-Week Treasury Bill Yield',
    '^FVX': 'CBOE 5-Year Treasury Note Yield',
    '^TNX': 'CBOE 10-Year Treasury Note Yield',
    '^TYX': 'CBOE 30-Year Treasury Bond Yield'
}