from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Define paths
BASE_DIR = Path(__file__).resolve().parents[2] / "_database"
DATABASE_PATH = BASE_DIR / "database.parquet"
OUTPUT_PATH = BASE_DIR / "changepoint.parquet"


def process_and_save_log_returns(
    tickers: list[str],
    output_path: Path | str = OUTPUT_PATH,
    database_path: Path | str = DATABASE_PATH,
) -> pd.DataFrame:

    parquet_file = pq.ParquetFile(database_path)
    all_columns = parquet_file.schema.names
    target_columns = [
        col for col in all_columns if "Adj Close" in col and any(t in col for t in tickers)
    ]

    if not target_columns:
        target_columns = [col for col in all_columns if any(t in col for t in tickers)]
    px = pd.read_parquet(database_path, columns=target_columns)

    if isinstance(px.columns, pd.MultiIndex):
        # Drop 'Adj Close' level if present
        if "Adj Close" in px.columns.get_level_values(0):
            px = px["Adj Close"]
        else:
            px.columns = px.columns.droplevel(0)
    else:
        px.columns = [next((t for t in tickers if t in col), col) for col in px.columns]

    px.index = pd.to_datetime(px.index)
    px = px.sort_index()

    log_returns = np.log(px / px.shift(1)).dropna()
    log_returns.to_parquet(output_path)

    return log_returns

df_log_returns = process_and_save_log_returns(
    tickers=["^VIX", "^GSPC"], output_path=OUTPUT_PATH
)