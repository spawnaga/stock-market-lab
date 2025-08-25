#!/usr/bin/env python3
"""
Data Loading Script for 17+ Years of 1-Minute OHLCV Data

This script loads your historical OHLCV data into the system's Redis database
for use by the LSTM agent and other components.

Usage:
    python data_loading_script.py --data-file <path_to_your_data.csv> --symbol <ticker_symbol>
"""

import argparse
import csv
import json
import redis
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def connect_to_redis():
    """Connect to Redis database."""
    try:
        r = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)
        # Test connection
        r.ping()
        logger.info("Connected to Redis successfully")
        return r
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return None

def load_csv_data(file_path, symbol):
    """
    Load CSV data assuming standard OHLCV format:
    Date,Open,High,Low,Close,Volume
    
    For 1-minute data, you might have:
    Datetime,Open,High,Low,Close,Volume
    """
    data_points = []
    
    try:
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row_num, row in enumerate(reader, 1):
                try:
                    # Handle different timestamp formats
                    timestamp = None
                    if 'datetime' in row:
                        # Assume datetime format like "2020-01-01 09:30:00"
                        timestamp_str = row['datetime']
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S').timestamp()
                    elif 'date' in row:
                        # Assume date format like "2020-01-01"
                        date_str = row['date']
                        timestamp = datetime.strptime(date_str, '%Y-%m-%d').timestamp()
                    else:
                        # If no timestamp field, skip this row
                        continue
                        
                    # Create data point
                    data_point = {
                        'timestamp': timestamp,
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': int(row['volume']),
                        'symbol': symbol
                    }
                    
                    data_points.append(data_point)
                    
                    # Log progress every 10,000 records
                    if row_num % 10000 == 0:
                        logger.info(f"Processed {row_num} data points...")
                        
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid row {row_num}: {e}")
                    continue
                    
        logger.info(f"Successfully loaded {len(data_points)} data points from {file_path}")
        return data_points
        
    except FileNotFoundError:
        logger.error(f"Data file not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading data file: {e}")
        return []

def load_data_to_redis(redis_client, data_points, symbol):
    """
    Load data points into Redis database.
    Uses Redis lists for storing historical data.
    """
    if not data_points:
        logger.warning("No data points to load")
        return 0
        
    try:
        # Clear existing data for this symbol (optional)
        # redis_client.delete(f"historical_prices_{symbol}")
        
        # Load data into Redis list (using LPUSH to add to the front)
        for data_point in reversed(data_points):  # Reverse to maintain chronological order
            # Convert to JSON string for storage
            json_data = json.dumps(data_point)
            redis_client.lpush(f"historical_prices_{symbol}", json_data)
            
        # Also store as a generic list for the LSTM agent
        for data_point in reversed(data_points):
            json_data = json.dumps(data_point)
            redis_client.lpush("historical_prices", json_data)
            
        logger.info(f"Successfully loaded {len(data_points)} data points into Redis")
        return len(data_points)
        
    except Exception as e:
        logger.error(f"Error loading data to Redis: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description='Load OHLCV data into Redis for the stock market lab')
    parser.add_argument('--data-file', required=True, help='Path to CSV file containing OHLCV data')
    parser.add_argument('--symbol', required=True, help='Stock symbol (e.g., AAPL, MSFT)')
    parser.add_argument('--redis-host', default='redis', help='Redis host (default: redis)')
    parser.add_argument('--redis-port', default=6379, type=int, help='Redis port (default: 6379)')
    
    args = parser.parse_args()
    
    logger.info(f"Loading data for symbol: {args.symbol}")
    logger.info(f"Data file: {args.data_file}")
    
    # Connect to Redis
    redis_client = redis.Redis(host=args.redis_host, port=args.redis_port, db=0, decode_responses=True)
    
    try:
        # Test connection
        redis_client.ping()
        logger.info("Connected to Redis successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return 1
    
    # Load data from CSV
    data_points = load_csv_data(args.data_file, args.symbol)
    
    if not data_points:
        logger.error("No data loaded. Exiting.")
        return 1
    
    # Load data to Redis
    loaded_count = load_data_to_redis(redis_client, data_points, args.symbol)
    
    if loaded_count > 0:
        logger.info(f"Successfully loaded {loaded_count} data points for {args.symbol}")
        logger.info("Data is now available for the LSTM agent and other components")
        return 0
    else:
        logger.error("Failed to load data")
        return 1

if __name__ == "__main__":
    exit(main())