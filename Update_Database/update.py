import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta



import EU_equities
import US_equities




ticker_eu = [
    "ADS.DE", "AIR.PA", "ALV.DE", "BAS.DE",
    "BAYN.DE", "BEI.DE", "BMW.DE", "BNR.DE",
    "CBK.DE", "CON.DE", "DTG.DE", "DBK.DE",
    "DB1.DE", "DHL.DE", "DTE.DE", "EOAN.DE",
    "FRE.DE", "FME.DE", "G1A.DE", "HNR1.DE",
    "HEI.DE", "HEN3.DE", "IFX.DE", "MBG.DE",
    "MRK.DE", "MTX.DE", "MUV2.DE", "PAH3.DE",
    "QIA.DE", "RHM.DE", "RWE.DE", "SAP.DE",
    "G24.DE", "SIE.DE", "ENR.DE", "SHL.DE",
    "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE",
]

ticker_us = [

]

ticker_commodities = [
    'CL=F', 
    'BZ=F', 
    'GC=F', 
    'NG=F', 
    'ZC=F', 
    'ZS=F', 
    'HG=F', 
    'SI=F', 
    'LE=F', 
    'KC=F'
]

ticker_crypto = [
    'BTC-USD',  # Bitcoin
    'ETH-USD',  # Ethereum
    'USDT-USD'  # Tether (Tracking stablecoin supply volume highlights dollar-liquidity shifts)
]

ticker_indices = [
    '^GSPC',     # S&P 500 Index (US Large Cap Baseline)
    '^IXIC',     # NASDAQ Composite (Global Tech Proxy)
    '^RUT',      # Russell 2000 (US Small Cap / Domestic Economic Health)
    '^N225',     # Nikkei 225 (Japan Benchmark)
    '^HSI',      # Hang Seng Index (Hong Kong / China Gateway Proxy)
    'IMOEX.ME',  # MOEX Russia Index
    '^BSESN',    # BSE SENSEX (India Benchmark)
    '^DJI',      # Dow Jones Industrial Average (US Blue-Chip Proxy)
    '^FTSE',     # FTSE 100 (UK Benchmark)
    '^GDAXI',    # DAX (Germany Benchmark)
]

ticker_ir = [
    '^IRX',  # 13-Week Treasury Bill Yield
    '^FVX',  # 5-Year Treasury Bond Yield
    '^TNX',  # 10-Year Treasury Bond Yield
    '^TYX'   # 30-Year Treasury Bond Yield
]

ticker_macro = [
    'FEDFUNDS',   # Effective Federal Reserve Interest Rate
    'CPIAUCSL',   # Consumer Price Index (Inflation Measure)
    'UNRATE',     # US Unemployment Rate
    'GDPC1',      # US Real Gross Domestic Product (Quarterly)
    'T10Y2Y',     # 10-Year Treasury Constant Maturity Minus 2-Year Treasury (Yield Curve Inversion Indicator)
    'UMCSENT',    # University of Michigan Consumer Sentiment Index
    'M2SL'        # M2 Money Supply (Liquidity Measure)
]



ticker_fx = [
    'EURUSD=X',  # Euro / US Dollar
    'GBPUSD=X',  # British Pound / US Dollar
    'DX-Y.NYB',  # US Dollar Index (DXY) - measures USD against a basket of currencies
    'USDJPY=X',  # US Dollar / Japanese Yen
    'EURGBP=X'   # Euro / British Pound
]

ticker_sentiment = [
    '^VIX',   # CBOE Volatility Index (S&P 500 Fear Gauge)
    '^VVIX',  # Volatility of the VIX (Measure of tail-risk/uncertainty)
    '^V2TX',  # Euro Stoxx 50 Volatility Index (European equivalent of VIX)
    '^MOVE'   # ICE BofA US Bond Market Volatility Index (Bond market fear gauge)
]

ticker_sectors = [
    'XLE',  # Energy (Exxon, Chevron)
    'XLF',  # Financials (JPMorgan, Goldman Sachs)
    'XLK',  # Technology (Apple, Microsoft, Nvidia)
    'XLY',  # Consumer Discretionary (Amazon, Tesla)
    'XLP',  # Consumer Staples (Walmart, Procter & Gamble - Defensive)
    'XLV',  # Healthcare (Johnson & Johnson, UnitedHealth)
    'XLI',  # Industrials (Caterpillar, General Electric)
    'XLB',  # Materials (Linde, Newmont)
    'XLRE', # Real Estate
    'XLU',  # Utilities (Defensive/Yield-sensitive)
    'XLC'   # Communication Services (Meta, Alphabet)
]


