import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import os

UNIVERSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Database', 'US_equities.parquet')
_GITHUB_API = "https://api.github.com/repos/fja05680/sp500/contents/"
_START_YEAR = 1999


def _fetch_sp500_tickers():
    # Find the CSV filename dynamically so it works when the repo updates the file
    resp = requests.get(_GITHUB_API)
    resp.raise_for_status()
    files = resp.json()
    csv_file = next(f for f in files if f['name'].startswith('S&P 500') and f['name'].endswith('.csv'))
    print(f"Using: {csv_file['name']}")

    df = pd.read_csv(csv_file['download_url'])
    df['date'] = pd.to_datetime(df['date'])

    # Collect year-end constituents from _START_YEAR to current year
    all_tickers = set()
    current_year = datetime.today().year
    for year in range(_START_YEAR, current_year + 1):
        year_rows = df[df['date'].dt.year == year]
        if year_rows.empty:
            continue
        last_row = year_rows.iloc[-1]
        tickers = [t.strip().replace('.', '-') for t in last_row['tickers'].split(',')]
        all_tickers.update(tickers)

    unique = sorted(all_tickers)
    print(f"Unique tickers across year-end snapshots ({_START_YEAR}-{current_year}): {len(unique)}")
    return unique


def update_universe(universe_file=UNIVERSE_FILE):
    print("Fetching S&P 500 historical constituents...")
    tickers = _fetch_sp500_tickers()

    today = datetime.today().date()
    yf_end = (today + timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = f'{_START_YEAR}-01-01'

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
