import yfinance as yf
import pandas as pd
import requests
from io import StringIO
from datetime import datetime, timedelta
import os

START_DATE = '2000-01-01'
UNIVERSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Database', 'EU_equities.parquet')

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

_STOXX50_FALLBACK = [
    'ADS.DE', 'ADYEN.AS', 'AD.AS', 'AI.PA', 'AIR.PA', 'ALV.DE', 'ABI.BR', 'ARGX.BR',
    'ASML.AS', 'CS.PA', 'BAS.DE', 'BAYN.DE', 'BBVA.MC', 'BMW.DE', 'BNP.PA', 'BN.PA',
    'DBK.DE', 'DB1.DE', 'DHL.DE', 'DTE.DE', 'ENEL.MI', 'ENI.MI', 'EL.PA', 'RACE.MI',
    'RMS.PA', 'IBE.MC', 'ITX.MC', 'IFX.DE', 'INGA.AS', 'ISP.MI', 'OR.PA', 'MC.PA',
    'MBG.DE', 'MUV2.DE', 'NDA-FI.HE', 'PRX.AS', 'RHM.DE', 'SAF.PA', 'SGO.PA', 'SAN.PA',
    'SAN.MC', 'SAP.DE', 'SU.PA', 'SIE.DE', 'ENR.DE', 'TTE.PA', 'UCG.MI', 'DG.PA',
    'VOW.DE', 'WKL.AS',
]


def _read_html(url):
    resp = requests.get(url, headers=_HEADERS)
    resp.raise_for_status()
    return pd.read_html(StringIO(resp.text), flavor='lxml')


def _find_table(tables, required_cols, min_rows=10):
    for t in tables:
        if all(c in t.columns for c in required_cols) and len(t) >= min_rows:
            return t
    return None


def get_european_tickers():
    universe = {}

    # --- 1. DAX 40 (Germany) ---
    try:
        tables = _read_html("https://en.wikipedia.org/wiki/DAX")
        dax_df = _find_table(tables, ['Ticker'], min_rows=30)
        if dax_df is None:
            raise ValueError("DAX components table not found")
        universe['DAX 40'] = [str(t).strip().replace('.', '-') + ".DE" for t in dax_df['Ticker'].tolist()]
    except Exception as e:
        print(f"Error fetching DAX: {e}")
        universe['DAX 40'] = []

    # --- 2. EURO STOXX 50 (Eurozone) ---
    try:
        tables = _read_html("https://en.wikipedia.org/wiki/Euro_Stoxx_50")
        stoxx_df = _find_table(tables, ['Ticker', 'Exchange'], min_rows=40)
        if stoxx_df is None:
            raise ValueError("Euro Stoxx 50 components table not found")
        exchange_suffix = {
            'frankfurt': '.DE', 'paris': '.PA', 'amsterdam': '.AS',
            'milano': '.MI', 'milan': '.MI', 'madrid': '.MC',
            'brussels': '.BR', 'dublin': '.IR', 'helsinki': '.HE',
        }
        tickers = []
        for _, row in stoxx_df.iterrows():
            ticker = str(row['Ticker']).strip().replace('.', '-')
            exchange = str(row['Exchange']).lower()
            suffix = next((s for k, s in exchange_suffix.items() if k in exchange), '')
            tickers.append(f"{ticker}{suffix}")
        universe['EURO STOXX 50'] = tickers
    except Exception as e:
        print(f"Euro Stoxx 50 scrape failed ({e}), using hardcoded fallback.")
        universe['EURO STOXX 50'] = _STOXX50_FALLBACK

    # --- 3. FTSE 100 (United Kingdom) ---
    try:
        tables = _read_html("https://en.wikipedia.org/wiki/FTSE_100_Index")
        ftse_df = _find_table(tables, ['Ticker'], min_rows=90)
        if ftse_df is None:
            raise ValueError("FTSE 100 components table not found")
        universe['FTSE 100'] = [str(t).strip().replace('.', '-') + ".L" for t in ftse_df['Ticker'].tolist()]
    except Exception as e:
        print(f"Error fetching FTSE 100: {e}")
        universe['FTSE 100'] = []

    all_tickers = []
    for index_name, tickers in universe.items():
        print(f"  {index_name}: {len(tickers)} tickers")
        all_tickers.extend(tickers)

    unique = sorted(set(all_tickers))
    print(f"  Total unique: {len(unique)}")
    return unique


def update_universe(tickers=None, start_date=START_DATE, universe_file=UNIVERSE_FILE):
    if tickers is None:
        print("Fetching European tickers...")
        tickers = get_european_tickers()

    today = datetime.today().date()
    yf_end = (today + timedelta(days=1)).strftime('%Y-%m-%d')

    if os.path.exists(universe_file):
        existing = pd.read_parquet(universe_file)
        last_date = pd.to_datetime(existing['Date']).max().date()

        if last_date >= today:
            print(f"Already up to date (last date: {last_date}).")
            return existing

        fetch_start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"Updating: last date {last_date} -> fetching {fetch_start} to {today}...")
    else:
        existing = None
        fetch_start = start_date
        print(f"No file found. Downloading full history from {fetch_start} to {today}...")

    raw = yf.download(tickers, start=fetch_start, end=yf_end, auto_adjust=True, progress=True)

    if raw.empty:
        print("No new data found.")
        return existing

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns.names = ['Field', 'Ticker']
        new_rows = raw.stack(level='Ticker').reset_index()
    else:
        new_rows = raw.reset_index()
        new_rows['Ticker'] = tickers[0]

    if existing is not None:
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=['Date', 'Ticker'], keep='last')
    else:
        combined = new_rows

    combined = combined.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    combined.to_parquet(universe_file, index=False)
    print(f"Saved {len(combined):,} rows to {universe_file}.")
    return combined


if __name__ == '__main__':
    df = update_universe()
    print(df.tail(20))
