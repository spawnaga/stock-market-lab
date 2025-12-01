"""
Market Data Loader
==================
Utility functions for loading market data from various sources.
"""

import os
import glob
import logging
from typing import Optional, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_market_data(
    symbol: str = "AAPL",
    data_path: str = "F:/Market Data/Extracted",
    data_type: str = "stock",
    timeframe: str = "1min"
) -> pd.DataFrame:
    """
    Load market data for a given symbol from parquet or CSV files.

    Args:
        symbol: Stock/asset symbol (e.g., "AAPL", "MSFT")
        data_path: Base path to market data
        data_type: Type of data ("stock", "etf", "crypto", "fx", "index", "futures")
        timeframe: Data timeframe ("1min", "5min", "1h", "1d")

    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Loading {symbol} data from {data_path}")

    # Try different folder patterns based on data type
    folder_patterns = [
        f"{data_type}_full_{timeframe}_adjsplitdiv",
        f"{data_type}_full_{timeframe}",
        f"{data_type}_{timeframe}",
        data_type
    ]

    for folder in folder_patterns:
        folder_path = os.path.join(data_path, folder)
        if os.path.exists(folder_path):
            # Try parquet first
            parquet_pattern = os.path.join(folder_path, f"{symbol}*.parquet")
            parquet_files = glob.glob(parquet_pattern)
            if parquet_files:
                logger.info(f"Found parquet file: {parquet_files[0]}")
                return pd.read_parquet(parquet_files[0])

            # Try CSV
            csv_pattern = os.path.join(folder_path, f"{symbol}*.csv")
            csv_files = glob.glob(csv_pattern)
            if csv_files:
                logger.info(f"Found CSV file: {csv_files[0]}")
                return pd.read_csv(csv_files[0])

    # Recursive search
    logger.info(f"Searching recursively for {symbol}...")

    # Parquet
    parquet_files = glob.glob(os.path.join(data_path, "**", f"*{symbol}*.parquet"), recursive=True)
    if parquet_files:
        logger.info(f"Found parquet file: {parquet_files[0]}")
        return pd.read_parquet(parquet_files[0])

    # CSV
    csv_files = glob.glob(os.path.join(data_path, "**", f"*{symbol}*.csv"), recursive=True)
    if csv_files:
        logger.info(f"Found CSV file: {csv_files[0]}")
        return pd.read_csv(csv_files[0])

    raise FileNotFoundError(f"No data found for symbol {symbol} in {data_path}")


def list_available_symbols(
    data_path: str = "F:/Market Data/Extracted",
    data_type: str = "stock"
) -> List[str]:
    """List all available symbols in the data directory."""
    symbols = set()

    # Search for parquet and CSV files
    for pattern in ["*.parquet", "*.csv"]:
        files = glob.glob(os.path.join(data_path, "**", pattern), recursive=True)
        for f in files:
            # Extract symbol from filename
            filename = os.path.basename(f)
            symbol = filename.split('_')[0].split('.')[0]
            if symbol.isalpha() and len(symbol) <= 5:
                symbols.add(symbol.upper())

    return sorted(list(symbols))


def generate_synthetic_data(
    symbol: str = "SYNTH",
    num_samples: int = 5000,
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(seed)

    dates = pd.date_range(start='2020-01-01', periods=num_samples, freq='1min')
    price = 100.0
    prices = [price]

    for _ in range(num_samples - 1):
        change = np.random.normal(0.0001, 0.005)
        price *= (1 + change)
        prices.append(price)

    prices = np.array(prices)

    return pd.DataFrame({
        'datetime': dates,
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, num_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.003, num_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.003, num_samples))),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, num_samples),
        'symbol': symbol
    })


if __name__ == "__main__":
    # Test loading
    import sys

    if len(sys.argv) > 1:
        symbol = sys.argv[1]
        data_path = sys.argv[2] if len(sys.argv) > 2 else "F:/Market Data/Extracted"

        try:
            df = load_market_data(symbol, data_path)
            print(f"Loaded {len(df)} rows for {symbol}")
            print(df.head())
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Generating synthetic data instead...")
            df = generate_synthetic_data(symbol)
            print(df.head())
    else:
        print("Usage: python market_data_loader.py <symbol> [data_path]")