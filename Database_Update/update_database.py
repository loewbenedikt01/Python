
import sys
import os
import time
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

# --------------
# PARAMS
# --------------

START_DATE = '2000-01-01'
END_DATE   = datetime.now().strftime('%Y-%m-%d')
DATABASE_DIR = os.path.join(_ROOT, 'Database')

# --------------
# API KEYS
# --------------

FRED_API_KEY = os.getenv('FRED_API_KEY', '8ce0ff451f8503c5fcc7aa27b9ecb3e0')


os.makedirs(DATABASE_DIR, exist_ok=True)



# --------------
# TICKERS
# --------------

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
    'TMF': 'Direxion Daily 20+ Year Treasury Bull 3X Shares',
    'TMV': 'Direxion Daily 20+ Year Treasury Bear 3X Shares'
}


ticker_commodities = {
    # === I. ENERGY & ENVIRONMENTAL MARKETS ===
    'CL=F': 'Crude Oil Futures',
    'BZ=F': 'Brent Crude Oil Futures',
    'NG=F': 'Natural Gas Futures',
    'RB=F': 'RBOB Gasoline Futures',
    'HO=F': 'Heating Oil Futures',
    'TTF=F': 'Dutch TTF Natural Gas Futures',
    'JKM=F': 'Japan/Korea LNG Futures',
    'MTF=F': 'Methanol Futures',
    'KRBN': 'KraneShares Global Carbon Strategy ETF',
    'URA': 'Global X Uranium ETF',
    'XLE': 'Energy Select Sector SPDR Fund',
    'UNG': 'United States Natural Gas Fund',
    'USO': 'United States Oil Fund',
    'BNO': 'United States Brent Oil Fund',
    'KMLM': 'KFA Mount Lucas Index Strategy ETF',

    # === II. PRECIOUS METALS ===
    'GC=F': 'Gold Futures',
    'SI=F': 'Silver Futures',
    'PL=F': 'Platinum Futures',
    'PA=F': 'Palladium Futures',
    'GLD': 'SPDR Gold Shares',
    'SLV': 'iShares Silver Trust',
    'PPLT': 'abrdn Physical Platinum Shares ETF',
    'PALL': 'abrdn Physical Palladium Shares ETF',
    'IAU': 'iShares Gold Trust',
    'PSLV': 'Sprott Physical Silver Trust',

    # === III. BASE, INDUSTRIAL & TRANSITION METALS ===
    'HG=F': 'Copper Futures',
    'ALI=F': 'Aluminum Futures',
    'DBB': 'Invesco DB Base Metals Fund',
    'LIT': 'Global X Lithium & Battery Tech ETF',
    'REMX': 'VanEck Rare Earth/Strategic Metals ETF',
    'SLX': 'VanEck Steel ETF',
    'COPX': 'Global X Copper Miners ETF',
    'PICK': 'iShares MSCI Global Metals & Mining Producers ETF',
    'VALE': 'Vale S.A.',
    'RIO': 'Rio Tinto Group',
    'BHP': 'BHP Group Limited',
    'FCX': 'Freeport-McMoRan Inc.',
    'AA': 'Alcoa Corporation',
    'NUE': 'Nucor Corporation',
    'STLD': 'Steel Dynamics, Inc.',
    'XBM.TO': 'iShares S&P/TSX Global Base Metals Index ETF',
    'JJCTF': 'iPath Series B Bloomberg Copper Subindex Total Return ETN',
    'CPER': 'United States Copper Index Fund',
    'GLNCY': 'Glencore plc',
    'ANFGF': 'Anglo American plc',
    'HBM': 'Hudbay Minerals Inc.',
    'TECK': 'Teck Resources Limited',

    # === IV. AGRICULTURAL GRAINS & OILSEEDS ===
    'ZC=F': 'Corn Futures',
    'ZW=F': 'Wheat Futures',
    'ZS=F': 'Soybean Futures',
    'ZM=F': 'Soybean Meal Futures',
    'ZL=F': 'Soybean Oil Futures',
    'ZR=F': 'Rough Rice Futures',
    'ZO=F': 'Oat Futures',
    'KE=F': 'KC Hard Red Winter Wheat Futures',
    'CORN': 'Teucrium Corn Fund',
    'WEAT': 'Teucrium Wheat Fund',
    'SOYB': 'Teucrium Soybean Fund',
    'DBA': 'Invesco DB Agriculture Fund',
    'TAGS': 'Teucrium Agricultural Fund',
    'MOO': 'VanEck Agribusiness ETF',
    'BG': 'Bunge Global SA',
    'ADM': 'Archer-Daniels-Midland Company',
    'ANDE': 'The Andersons, Inc.',
    'NTR': 'Nutrien Ltd.',
    'MOS': 'The Mosaic Company',

    # === V. SOFTS, FIBERS & FOREST PRODUCTS ===
    'KC=F': 'Coffee C Futures',
    'CC=F': 'Cocoa Futures',
    'SB=F': 'Sugar No. 11 Futures',
    'CT=F': 'Cotton No. 2 Futures',
    'OJ=F': 'Orange Juice Futures',
    'LBS=F': 'Lumber Futures',
    'JO': 'iPath Series B Bloomberg Coffee Subindex Total Return ETN',
    'NIB': 'iPath Series B Bloomberg Cocoa Subindex Total Return ETN',
    'SGG': 'iPath Series B Bloomberg Sugar Subindex Total Return ETN',
    'BAL': 'iPath Series B Bloomberg Cotton Subindex Total Return ETN',
    'WOOD': 'iShares Global Timber & Forestry ETF',
    'CUT': 'Invesco MSCI Global Timber ETF',
    'WY': 'Weyerhaeuser Company',
    'IP': 'International Paper Company',
    'CTVA': 'Corteva, Inc.',
    'FMC': 'FMC Corporation',
    'SMG': 'The Scotts Miracle-Gro Company',
    'SUZ': 'Suzano S.A.',

    # === VI. LIVESTOCK, POULTRY & DAIRY ===
    'LE=F': 'Live Cattle Futures',
    'GF=F': 'Feeder Cattle Futures',
    'HE=F': 'Lean Hog Futures',
    'DC=F': 'Class III Milk Futures',
    'GNF=F': 'General/Other Livestock Proxy',
    'COW': 'iPath Series B Bloomberg Livestock Subindex Total Return ETN',
    'TSN': 'Tyson Foods, Inc.',
    'PPC': 'Pilgrim\'s Pride Corporation',
    'HRL': 'Hormel Foods Corporation',
}


ticker_crypto = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'USDT-USD': 'Tether'
}


ticker_forex = {
    # Major Pairs
    'EURUSD=X': 'Euro / US Dollar',
    'GBPUSD=X': 'British Pound / US Dollar',
    'USDJPY=X': 'US Dollar / Japanese Yen',
    'AUDUSD=X': 'Australian Dollar / US Dollar',
    'USDCAD=X': 'US Dollar / Canadian Dollar',
    'USDCHF=X': 'US Dollar / Swiss Franc',
    'NZDUSD=X': 'New Zealand Dollar / US Dollar',

    # Cross Pairs
    'EURGBP=X': 'Euro / British Pound',
    'EURJPY=X': 'Euro / Japanese Yen',
    'EURCHF=X': 'Euro / Swiss Franc',
    'EURAUD=X': 'Euro / Australian Dollar',
    'EURCAD=X': 'Euro / Canadian Dollar',
    'EURNZD=X': 'Euro / New Zealand Dollar',
    'EURSEK=X': 'Euro / Swedish Krona',
    'EURNOK=X': 'Euro / Norwegian Krone',
    'GBPJPY=X': 'British Pound / Japanese Yen',
    'GBPCHF=X': 'British Pound / Swiss Franc',
    'GBPAUD=X': 'British Pound / Australian Dollar',
    'GBPCAD=X': 'British Pound / Canadian Dollar',
    'AUDJPY=X': 'Australian Dollar / Japanese Yen',
    'NZDJPY=X': 'New Zealand Dollar / Japanese Yen',
    'CADJPY=X': 'Canadian Dollar / Japanese Yen',
    'CHFJPY=X': 'Swiss Franc / Japanese Yen',
    'AUDNZD=X': 'Australian Dollar / New Zealand Dollar',
    'AUDCAD=X': 'Australian Dollar / Canadian Dollar',
    'AUDCHF=X': 'Australian Dollar / Swiss Franc',
    'CADAUD=X': 'Canadian Dollar / Australian Dollar',
    'CADCHF=X': 'Canadian Dollar / Swiss Franc',
    'NZDCAD=X': 'New Zealand Dollar / Canadian Dollar',

    # Latin American Emerging Markets
    'USDMXN=X': 'US Dollar / Mexican Peso',
    'USDBRL=X': 'US Dollar / Brazilian Real',
    'USDCLP=X': 'US Dollar / Chilean Peso',
    'USDCOP=X': 'US Dollar / Colombian Peso',
    'USDPEN=X': 'US Dollar / Peruvian Sol',
    'USDARS=X': 'US Dollar / Argentine Peso',

    # Asian Emerging Markets
    'USDCNY=X': 'US Dollar / Chinese Yuan (Offshore/Onshore)',
    'USDCNH=X': 'US Dollar / Chinese Yuan (Offshore)',
    'USDHKD=X': 'US Dollar / Hong Kong Dollar',
    'USDINR=X': 'US Dollar / Indian Rupee',
    'USDKRW=X': 'US Dollar / South Korean Won',
    'USDSGD=X': 'US Dollar / Singapore Dollar',
    'USDTWD=X': 'US Dollar / New Taiwan Dollar',
    'USDTHB=X': 'US Dollar / Thai Baht',
    'USDMYR=X': 'US Dollar / Malaysian Ringgit',
    'USDIDR=X': 'US Dollar / Indonesian Rupiah',
    'USDPHP=X': 'US Dollar / Philippine Peso',

    # EMEA Emerging Markets
    'USDZAR=X': 'US Dollar / South African Rand',
    'USDTRY=X': 'US Dollar / Turkish Lira',
    'USDPLN=X': 'US Dollar / Polish Zloty',
    'USDHUF=X': 'US Dollar / Hungarian Forint',
    'USDCZK=X': 'US Dollar / Czech Koruna',
    'USDILS=X': 'US Dollar / Israeli New Shekel',
    'USDSAR=X': 'US Dollar / Saudi Riyal',
    'USDAED=X': 'US Dollar / UAE Dirham',
    'USDEGP=X': 'US Dollar / Egyptian Pound',
    'USDQAR=X': 'US Dollar / Qatari Riyal',

    # Northern European / G10
    'USDSEK=X': 'US Dollar / Swedish Krona',
    'USDNOK=X': 'US Dollar / Norwegian Krone',
    'USDDKK=X': 'US Dollar / Danish Krone',
    'GBPSEK=X': 'British Pound / Swedish Krona',
    'GBPNOK=X': 'British Pound / Norwegian Krone',

    # Exotic Crosses
    'EURMXN=X': 'Euro / Mexican Peso',
    'EURZAR=X': 'Euro / South African Rand',
    'EURCNH=X': 'Euro / Chinese Yuan (Offshore)',
    'EURINR=X': 'Euro / Indian Rupee',
    'GBPMXN=X': 'British Pound / Mexican Peso',
    'GBPZAR=X': 'British Pound / South African Rand',
    'AUDCNH=X': 'Australian Dollar / Chinese Yuan (Offshore)',
    'AUDZAR=X': 'Australian Dollar / South African Rand',
    'JPYZAR=X': 'Japanese Yen / South African Rand',

    # Indices and ETFs
    'DX-Y.NYB': 'ICE US Dollar Index (DXY)',
    'UUP': 'Invesco DB US Dollar Index Bullish Fund',
    'UDN': 'Invesco DB US Dollar Index Bearish Fund'
}




# --------------
# HELPERS
# --------------

def fetch_yfinance(
        tickers,
        start,
        end
):
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns.names = ['Field', 'Ticker']
    else:
        ticker = tickers[0] if isinstance(tickers, list) else tickers
        raw.columns = pd.MultiIndex.from_tuples(
            [(col, ticker) for col in raw.columns], names=['Field', 'Ticker']
        )

    try:
        df = raw.stack(level='Ticker', future_stack=True)
    except TypeError:
        df = raw.stack(level='Ticker')

    df.index.names = ['Date', 'Ticker']
    df.columns.name = None
    df = df[[c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]]
    return df.sort_index()


def get_last_date(path):
    if not os.path.exists(path):
        return None
    df  = pd.read_parquet(path, columns=[])
    idx = df.index.get_level_values('Date')
    return pd.Timestamp(idx.max())


def fill_short_gaps(df, limit=2):
    """Forward-fill NaNs per ticker, but only across gaps of up to `limit` rows."""
    return (
        df.sort_index()
          .groupby(level='Ticker', group_keys=False)
          .apply(lambda g: g.ffill(limit=limit))
    )


def split_yield_column(df, yield_prefix='^'):
    """Move Close into a Yield column for tickers starting with `yield_prefix`
    (CBOE yield indices), zeroing their Close. ETF tickers get Yield = 0."""
    df = df.copy()
    is_yield = df.index.get_level_values('Ticker').str.startswith(yield_prefix)
    df['Yield'] = 0.0
    df.loc[is_yield, 'Yield'] = df.loc[is_yield, 'Close']
    df.loc[is_yield, 'Close'] = 0.0
    return df


def add_name_column(df, name_map):
    """Add a Name column looked up from a {ticker: name} dict."""
    df = df.copy()
    df['Name'] = df.index.get_level_values('Ticker').map(name_map)
    return df


def update_yfinance_group(tickers, filename, yield_prefix=None):
    ticker_list = list(tickers.keys()) if isinstance(tickers, dict) else list(tickers)
    name_map    = tickers if isinstance(tickers, dict) else None
    path        = os.path.join(DATABASE_DIR, filename)

    last  = get_last_date(path)
    start = (last + timedelta(days=1)).strftime('%Y-%m-%d') if last is not None else START_DATE

    if start > END_DATE:
        print(f'  {filename}: already up to date (last = {last.date()}).')
        return

    print(f'  {filename}: fetching {len(ticker_list)} tickers from {start} to {END_DATE} ...')
    new_df = fetch_yfinance(ticker_list, start, END_DATE)
    if new_df.empty:
        print('  No new data returned.')
        return

    if yield_prefix is not None:
        new_df = split_yield_column(new_df, yield_prefix)

    if name_map is not None:
        new_df = add_name_column(new_df, name_map)

    if os.path.exists(path):
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    else:
        combined = new_df.sort_index()

    combined = fill_short_gaps(combined, limit=2)

    before = len(combined)
    combined = combined.dropna(how='any')
    dropped = before - len(combined)

    combined.to_parquet(path)
    print(f'  +{len(new_df)} rows -> {filename} (total {len(combined)} rows, dropped {dropped} NaN rows)')


# --------------
# MAIN
# --------------

def main():
    print(f'=== Database Update — {END_DATE} ===')
    update_yfinance_group(ticker_bonds, 'bonds.parquet', yield_prefix='^')
    update_yfinance_group(ticker_commodities, 'commodities.parquet')
    update_yfinance_group(ticker_crypto, 'crypto.parquet')
    update_yfinance_group(ticker_forex, 'forex.parquet')
    print('=== Done ===')


if __name__ == '__main__':
    main()






