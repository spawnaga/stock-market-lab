#!/usr/bin/env python3
"""
NQ Futures Trading Strategy Training Pipeline
Loads NQ data, trains LSTM model, and finds profitable trading strategies
"""

import pandas as pd
import redis
import json
import time
import requests
from datetime import datetime, timedelta
import numpy as np

# Configuration
NQ_DATA_PATH = r"F:\Market Data\Extracted\Futures_Original_full_1min_continuous_adjusted\NQ.csv"
REDIS_HOST = "localhost"
REDIS_PORT = 7379  # Note: Your Redis is on port 7379
SYMBOL = "NQ"
API_BASE = "http://localhost:5000"

def load_nq_data(file_path, sample_size=None):
    """Load NQ futures data from CSV."""
    print(f"📊 Loading NQ data from {file_path}...")
    
    df = pd.read_csv(file_path, header=None, 
                     names=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    
    # Convert datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['timestamp'] = df['datetime'].astype(np.int64) // 10**9
    
    print(f"   Total records: {len(df):,}")
    print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Sample if needed (for faster testing)
    if sample_size and len(df) > sample_size:
        print(f"   Sampling {sample_size:,} most recent records...")
        df = df.tail(sample_size)
    
    return df

def load_to_redis(df, symbol):
    """Load data into Redis."""
    print(f"\n🔄 Loading data to Redis...")
    
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
        r.ping()
        print("   ✅ Connected to Redis")
    except Exception as e:
        print(f"   ❌ Redis connection failed: {e}")
        return False
    
    # Clear existing data
    r.delete(f"historical_prices_{symbol}")
    r.delete("historical_prices")
    
    # Load data
    count = 0
    for _, row in df.iterrows():
        data_point = {
            'timestamp': float(row['timestamp']),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': int(row['volume']),
            'symbol': symbol
        }
        
        json_data = json.dumps(data_point)
        r.lpush(f"historical_prices_{symbol}", json_data)
        r.lpush("historical_prices", json_data)
        
        count += 1
        if count % 10000 == 0:
            print(f"   Loaded {count:,} records...")
    
    print(f"   ✅ Loaded {count:,} records to Redis")
    return True

def get_auth_token():
    """Get authentication token."""
    try:
        response = requests.post(f"{API_BASE}/login", json={
            "username": "trader",
            "password": "password123"
        })
        if response.status_code == 200:
            return response.json()['token']
    except:
        pass
    return None

def train_lstm_model(token):
    """Train LSTM model on NQ data."""
    print(f"\n🧠 Training LSTM model on NQ data...")
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{API_BASE}/lstm/retrain", headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Training completed!")
        print(f"   Data points used: {result.get('data_points_used', 'N/A')}")
        return True
    else:
        print(f"   ❌ Training failed: {response.text}")
        return False

def run_strategy_backtest(token, strategy_name, days=30):
    """Run backtest for a strategy."""
    print(f"\n🔬 Backtesting: {strategy_name}")
    
    headers = {"Authorization": f"Bearer {token}"}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    response = requests.post(f"{API_BASE}/backtest/run", headers=headers, json={
        "strategy_name": strategy_name,
        "symbol": SYMBOL,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "initial_capital": 50000.0  # NQ futures require more capital
    })
    
    if response.status_code == 200:
        result = response.json()
        print(f"   Initial: ${result['initial_capital']:,.2f}")
        print(f"   Final: ${result['final_capital']:,.2f}")
        print(f"   Return: {result['total_return']:.2f}%")
        print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
        print(f"   Max DD: {result['max_drawdown']:.2f}%")
        print(f"   Trades: {result['trade_count']} (W:{result['win_count']} L:{result['loss_count']})")
        return result
    else:
        print(f"   ❌ Failed: {response.text}")
        return None

def compare_all_strategies(token, days=30):
    """Compare all available strategies."""
    print(f"\n📊 Comparing All Strategies on NQ...")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Get available strategies
    response = requests.get(f"{API_BASE}/backtest/strategies", headers=headers)
    if response.status_code != 200:
        print("   ❌ Could not get strategies")
        return None
    
    strategies = response.json()['strategies']
    strategy_names = [s['name'] for s in strategies]
    
    print(f"   Testing {len(strategy_names)} strategies...")
    
    # Compare
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    response = requests.post(f"{API_BASE}/backtest/compare", headers=headers, json={
        "strategy_names": strategy_names,
        "symbol": SYMBOL,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "initial_capital": 50000.0
    })
    
    if response.status_code == 200:
        results = response.json()['comparison']
        
        print(f"\n{'Strategy':<30} {'Return %':<12} {'Sharpe':<10} {'Trades':<10}")
        print("=" * 65)
        
        sorted_results = sorted(
            [(name, res) for name, res in results.items() if 'error' not in res],
            key=lambda x: x[1]['total_return'],
            reverse=True
        )
        
        for name, result in sorted_results:
            print(f"{name:<30} {result['total_return']:>10.2f}% {result['sharpe_ratio']:>9.2f} {result['trade_count']:>9}")
        
        return sorted_results
    else:
        print(f"   ❌ Comparison failed")
        return None

def main():
    print("=" * 70)
    print("🚀 NQ Futures Trading Strategy Development")
    print("=" * 70)
    
    # Step 1: Load NQ data
    print("\n[1/5] Loading NQ futures data...")
    df = load_nq_data(NQ_DATA_PATH, sample_size=100000)  # Use 100k records for faster testing
    
    # Step 2: Load to Redis
    print("\n[2/5] Loading data to Redis...")
    if not load_to_redis(df, SYMBOL):
        print("❌ Failed to load data to Redis")
        return
    
    # Step 3: Authenticate
    print("\n[3/5] Authenticating...")
    token = get_auth_token()
    if not token:
        print("❌ Authentication failed")
        return
    print("   ✅ Authenticated")
    
    # Step 4: Train LSTM model
    print("\n[4/5] Training LSTM model...")
    if train_lstm_model(token):
        print("   Waiting for model to stabilize...")
        time.sleep(3)
    
    # Step 5: Test strategies
    print("\n[5/5] Testing trading strategies...")
    results = compare_all_strategies(token, days=30)
    
    if results:
        best_strategy = results[0]
        print(f"\n🏆 BEST STRATEGY FOR NQ:")
        print(f"   Name: {best_strategy[0]}")
        print(f"   Return: {best_strategy[1]['total_return']:.2f}%")
        print(f"   Sharpe Ratio: {best_strategy[1]['sharpe_ratio']:.2f}")
        print(f"   Win Rate: {best_strategy[1]['win_count']}/{best_strategy[1]['trade_count']}")
    
    print("\n" + "=" * 70)
    print("✅ NQ Strategy Development Complete!")
    print("=" * 70)
    print("\nNext Steps:")
    print("1. Review the best strategy above")
    print("2. Run longer backtests: Modify 'days=30' to 'days=365'")
    print("3. Load full dataset: Remove 'sample_size' parameter")
    print("4. Monitor live: docker logs -f stock-market-lab-python-agents-1")

if __name__ == "__main__":
    main()
