#!/usr/bin/env python3
"""
Data Loader for Stock Market Lab
Loads CSV market data into PostgreSQL database.

Usage:
    python data_loader.py --symbol AAPL
    python data_loader.py --all --limit 10
    python data_loader.py --popular
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = os.environ.get(
    'DATABASE_URL',
    'postgresql://stock_user:stock_password@localhost:5432/stock_market'
)

# Market data path from environment (set in docker-compose.yml or .env)
MARKET_DATA_PATH = os.environ.get('MARKET_DATA_PATH', '/data/market')

# Data paths - supports both Docker and local
# Priority order: env var, Docker mount, Windows local, WSL
DATA_PATHS = [
    MARKET_DATA_PATH,  # From environment variable
    os.path.join(MARKET_DATA_PATH, 'stock_full_1min_adjsplitdiv'),  # Subdirectory
    '/data/market',  # Docker mount root
    '/data/market/stock_full_1min_adjsplitdiv',  # Docker mount subdirectory
    'F:/Market Data/Extracted/stock_full_1min_adjsplitdiv',  # Windows local
    '/mnt/f/Market Data/Extracted/stock_full_1min_adjsplitdiv',  # WSL
]

# Supported file extensions for market data
SUPPORTED_EXTENSIONS = ['.csv', '.txt']

# Popular symbols to load first
POPULAR_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
    'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'DIS', 'PYPL', 'NFLX',
    'ADBE', 'CRM', 'INTC', 'VZ', 'T', 'PFE', 'KO', 'PEP', 'MRK', 'ABT'
]

BATCH_SIZE = 10000  # Number of rows to insert at once


def get_data_path():
    """Find the data directory containing CSV or TXT market data files."""
    for path in DATA_PATHS:
        if os.path.exists(path) and os.path.isdir(path):
            # Check if directory contains any supported data files
            for f in os.listdir(path):
                if any(f.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    logger.info(f"Found data directory with market files: {path}")
                    return path
    raise FileNotFoundError(
        f"Data directory not found or contains no CSV/TXT files. "
        f"Tried: {DATA_PATHS}. "
        f"Set MARKET_DATA_PATH environment variable to your data directory."
    )


def get_db_connection():
    """Create database connection."""
    return psycopg2.connect(DATABASE_URL)


def ensure_table_exists(conn):
    """Ensure the OHLCV table exists."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS market_data_ohlcv (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(10) NOT NULL,
                datetime TIMESTAMP NOT NULL,
                open NUMERIC(12, 4) NOT NULL,
                high NUMERIC(12, 4) NOT NULL,
                low NUMERIC(12, 4) NOT NULL,
                close NUMERIC(12, 4) NOT NULL,
                volume BIGINT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, datetime)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_datetime
            ON market_data_ohlcv(symbol, datetime)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol
            ON market_data_ohlcv(symbol)
        """)
        conn.commit()
    logger.info("Table market_data_ohlcv is ready")


def count_existing_rows(conn, symbol):
    """Count existing rows for a symbol."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM market_data_ohlcv WHERE symbol = %s",
            (symbol,)
        )
        return cur.fetchone()[0]


def load_csv_file(filepath, symbol):
    """
    Load CSV file and yield batches of data.
    CSV format: datetime,open,high,low,close,volume (no header)
    """
    batch = []
    line_count = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                parts = line.split(',')
                if len(parts) != 6:
                    continue

                dt_str, open_p, high_p, low_p, close_p, volume = parts

                # Parse datetime
                dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')

                batch.append((
                    symbol,
                    dt,
                    float(open_p),
                    float(high_p),
                    float(low_p),
                    float(close_p),
                    int(volume)
                ))

                line_count += 1

                if len(batch) >= BATCH_SIZE:
                    yield batch
                    batch = []

            except (ValueError, IndexError) as e:
                logger.debug(f"Skipping invalid line: {line[:50]}... Error: {e}")
                continue

    if batch:
        yield batch

    logger.info(f"Processed {line_count:,} lines from {filepath}")


def insert_batch(conn, batch):
    """Insert a batch of data using execute_values for performance."""
    if not batch:
        return 0

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO market_data_ohlcv
            (symbol, datetime, open, high, low, close, volume)
            VALUES %s
            ON CONFLICT (symbol, datetime) DO NOTHING
            """,
            batch,
            template="(%s, %s, %s, %s, %s, %s, %s)"
        )
        inserted = cur.rowcount
        conn.commit()
        return inserted


def find_symbol_file(data_path, symbol):
    """Find the data file for a symbol (supports .csv and .txt extensions)."""
    for ext in SUPPORTED_EXTENSIONS:
        filepath = os.path.join(data_path, f"{symbol}{ext}")
        if os.path.exists(filepath):
            return filepath
        # Also try uppercase/lowercase variants
        filepath_upper = os.path.join(data_path, f"{symbol.upper()}{ext}")
        if os.path.exists(filepath_upper):
            return filepath_upper
        filepath_lower = os.path.join(data_path, f"{symbol.lower()}{ext}")
        if os.path.exists(filepath_lower):
            return filepath_lower
    return None


def load_symbol(conn, data_path, symbol, skip_existing=True):
    """Load data for a single symbol."""
    filepath = find_symbol_file(data_path, symbol)

    if not filepath:
        logger.warning(f"File not found for symbol {symbol} in {data_path}")
        return 0

    if skip_existing:
        existing = count_existing_rows(conn, symbol)
        if existing > 0:
            logger.info(f"Skipping {symbol}: {existing:,} rows already exist")
            return existing

    logger.info(f"Loading {symbol}...")

    total_inserted = 0
    batch_num = 0

    for batch in load_csv_file(filepath, symbol):
        inserted = insert_batch(conn, batch)
        total_inserted += inserted
        batch_num += 1

        if batch_num % 10 == 0:
            logger.info(f"  {symbol}: {total_inserted:,} rows inserted...")

    logger.info(f"Completed {symbol}: {total_inserted:,} rows inserted")
    return total_inserted


def get_available_symbols(data_path):
    """Get list of available symbol files (CSV or TXT)."""
    symbols = set()  # Use set to avoid duplicates
    for f in os.listdir(data_path):
        f_lower = f.lower()
        for ext in SUPPORTED_EXTENSIONS:
            if f_lower.endswith(ext):
                # Remove extension to get symbol name
                symbol = f[:-(len(ext))].upper()
                symbols.add(symbol)
                break
    return sorted(list(symbols))


def load_popular_symbols(conn, data_path):
    """Load popular symbols first."""
    logger.info(f"Loading {len(POPULAR_SYMBOLS)} popular symbols...")

    total = 0
    for symbol in POPULAR_SYMBOLS:
        rows = load_symbol(conn, data_path, symbol)
        total += rows

    logger.info(f"Completed loading popular symbols: {total:,} total rows")
    return total


def load_all_symbols(conn, data_path, limit=None):
    """Load all available symbols."""
    symbols = get_available_symbols(data_path)

    if limit:
        symbols = symbols[:limit]

    logger.info(f"Loading {len(symbols)} symbols...")

    total = 0
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] Processing {symbol}")
        rows = load_symbol(conn, data_path, symbol)
        total += rows

    logger.info(f"Completed loading all symbols: {total:,} total rows")
    return total


def get_loaded_symbols(conn):
    """Get list of symbols already in database."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT symbol, COUNT(*) as rows,
                   MIN(datetime) as start_date,
                   MAX(datetime) as end_date
            FROM market_data_ohlcv
            GROUP BY symbol
            ORDER BY symbol
        """)
        return cur.fetchall()


def main():
    parser = argparse.ArgumentParser(description='Load market data into PostgreSQL')
    parser.add_argument('--symbol', '-s', type=str, help='Load specific symbol')
    parser.add_argument('--popular', '-p', action='store_true', help='Load popular symbols')
    parser.add_argument('--all', '-a', action='store_true', help='Load all symbols')
    parser.add_argument('--limit', '-l', type=int, help='Limit number of symbols to load')
    parser.add_argument('--list', action='store_true', help='List loaded symbols')
    parser.add_argument('--force', '-f', action='store_true', help='Force reload existing data')

    args = parser.parse_args()

    try:
        data_path = get_data_path()
        logger.info(f"Using data path: {data_path}")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    conn = get_db_connection()
    ensure_table_exists(conn)

    try:
        if args.list:
            symbols = get_loaded_symbols(conn)
            print(f"\nLoaded symbols ({len(symbols)}):")
            print("-" * 70)
            for symbol, rows, start, end in symbols:
                print(f"  {symbol:6} | {rows:>12,} rows | {start} to {end}")
            return

        if args.symbol:
            load_symbol(conn, data_path, args.symbol.upper(), skip_existing=not args.force)
        elif args.popular:
            load_popular_symbols(conn, data_path)
        elif args.all:
            load_all_symbols(conn, data_path, limit=args.limit)
        else:
            # Default: load popular symbols
            logger.info("No option specified, loading popular symbols by default")
            load_popular_symbols(conn, data_path)

    finally:
        conn.close()


if __name__ == '__main__':
    main()
