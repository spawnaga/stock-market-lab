#!/usr/bin/env python3
"""
Complete Training and Trading Pipeline
This script trains models and creates trading strategies.
"""

import requests
import json
import time
import sys
from datetime import datetime, timedelta

BASE_URL = "http://localhost:5000"

def login():
    """Get authentication token."""
    response = requests.post(f"{BASE_URL}/login", json={
        "username": "trader",
        "password": "password123"
    })
    if response.status_code == 200:
        token = response.json()['token']
        print("✅ Authenticated successfully")
        return token
    else:
        print("❌ Authentication failed")
        sys.exit(1)

def check_health(token):
    """Check system health."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/health", headers=headers)
    if response.status_code == 200:
        health = response.json()
        print(f"\n📊 System Health:")
        print(f"   Agents Running: {health['agents']}")
        print(f"   Memory: {health['system']['memory_mb']:.2f} MB")
        print(f"   CPU: {health['system']['cpu_percent']:.1f}%")
        return True
    return False

def check_lstm_status(token):
    """Check LSTM model status."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/lstm/status", headers=headers)
    if response.status_code == 200:
        status = response.json()
        print(f"\n🧠 LSTM Model Status:")
        print(f"   Trained: {status['model_trained']}")
        print(f"   Device: {status['model_device']}")
        print(f"   Sequence Length: {status['sequence_length']}")
        return status['model_trained']
    return False

def train_lstm_model(token):
    """Trigger LSTM model training."""
    headers = {"Authorization": f"Bearer {token}"}
    print("\n🔄 Training LSTM model...")
    response = requests.post(f"{BASE_URL}/lstm/retrain", headers=headers)
    if response.status_code == 200:
        result = response.json()
        print(f"✅ LSTM training completed!")
        print(f"   Data points used: {result.get('data_points_used', 'N/A')}")
        return True
    else:
        print(f"❌ Training failed: {response.text}")
        return False

def get_available_strategies(token):
    """Get list of available backtesting strategies."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/backtest/strategies", headers=headers)
    if response.status_code == 200:
        strategies = response.json()['strategies']
        print(f"\n📋 Available Strategies:")
        for i, strategy in enumerate(strategies, 1):
            print(f"   {i}. {strategy['name']} ({strategy['type']})")
        return strategies
    return []

def run_backtest(token, strategy_name, symbol="AAPL", days=30):
    """Run a backtest for a strategy."""
    headers = {"Authorization": f"Bearer {token}"}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"\n🔬 Running backtest for {strategy_name}...")
    print(f"   Symbol: {symbol}")
    print(f"   Period: {start_date.date()} to {end_date.date()}")
    
    response = requests.post(f"{BASE_URL}/backtest/run", headers=headers, json={
        "strategy_name": strategy_name,
        "symbol": symbol,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "initial_capital": 10000.0
    })
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Backtest Results for {strategy_name}:")
        print(f"   Initial Capital: ${result['initial_capital']:,.2f}")
        print(f"   Final Capital: ${result['final_capital']:,.2f}")
        print(f"   Total Return: {result['total_return']:.2f}%")
        print(f"   Annualized Return: {result['annualized_return']:.2f}%")
        print(f"   Max Drawdown: {result['max_drawdown']:.2f}%")
        print(f"   Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"   Trades: {result['trade_count']} (Wins: {result['win_count']}, Losses: {result['loss_count']})")
        return result
    else:
        print(f"❌ Backtest failed: {response.text}")
        return None

def compare_strategies(token, strategy_names, symbol="AAPL", days=30):
    """Compare multiple strategies."""
    headers = {"Authorization": f"Bearer {token}"}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"\n📊 Comparing {len(strategy_names)} strategies...")
    
    response = requests.post(f"{BASE_URL}/backtest/compare", headers=headers, json={
        "strategy_names": strategy_names,
        "symbol": symbol,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "initial_capital": 10000.0
    })
    
    if response.status_code == 200:
        results = response.json()['comparison']
        print(f"\n✅ Strategy Comparison Results:")
        print(f"\n{'Strategy':<30} {'Return':<12} {'Sharpe':<10} {'Trades':<10}")
        print("-" * 65)
        for name, result in results.items():
            if 'error' not in result:
                print(f"{name:<30} {result['total_return']:>10.2f}% {result['sharpe_ratio']:>9.2f} {result['trade_count']:>9}")
        return results
    else:
        print(f"❌ Comparison failed: {response.text}")
        return None

def create_custom_strategy(token, name, description):
    """Create a custom trading strategy."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{BASE_URL}/strategies", headers=headers, json={
        "name": name,
        "description": description,
        "parameters": {
            "risk_tolerance": "medium",
            "max_position_size": 0.1,
            "stop_loss": 0.05
        }
    })
    
    if response.status_code == 201:
        strategy = response.json()
        print(f"\n✅ Created strategy: {strategy['name']}")
        return strategy
    else:
        print(f"❌ Strategy creation failed: {response.text}")
        return None

def main():
    print("=" * 70)
    print("🚀 AI-Driven Stock Market Lab - Training & Trading Pipeline")
    print("=" * 70)
    
    # Step 1: Authenticate
    print("\n[1/6] Authenticating...")
    token = login()
    
    # Step 2: Check system health
    print("\n[2/6] Checking system health...")
    if not check_health(token):
        print("❌ System not healthy. Please check services.")
        sys.exit(1)
    
    # Step 3: Check LSTM status and train if needed
    print("\n[3/6] Checking LSTM model...")
    is_trained = check_lstm_status(token)
    
    if not is_trained:
        print("   Model not trained. Starting training...")
        if train_lstm_model(token):
            print("   Waiting for training to complete...")
            time.sleep(5)
            check_lstm_status(token)
    else:
        print("   ✅ Model already trained")
    
    # Step 4: Get available strategies
    print("\n[4/6] Loading available strategies...")
    strategies = get_available_strategies(token)
    
    # Step 5: Run backtests
    print("\n[5/6] Running backtests...")
    if strategies:
        # Test first 3 strategies
        for strategy in strategies[:3]:
            run_backtest(token, strategy['name'], symbol="AAPL", days=30)
            time.sleep(1)
    
    # Step 6: Compare strategies
    print("\n[6/6] Comparing strategies...")
    if len(strategies) >= 2:
        strategy_names = [s['name'] for s in strategies[:3]]
        compare_strategies(token, strategy_names, symbol="AAPL", days=30)
    
    # Create a custom strategy
    print("\n[BONUS] Creating custom strategy...")
    create_custom_strategy(
        token,
        "My LSTM Strategy",
        "Custom strategy using LSTM predictions with risk management"
    )
    
    print("\n" + "=" * 70)
    print("✅ Training and trading pipeline completed!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Load your 17+ years of data: python data_loading_script.py --data-file <path> --symbol AAPL")
    print("2. Retrain with more data: curl -X POST http://localhost:5000/lstm/retrain")
    print("3. Monitor performance: curl http://localhost:5000/metrics")
    print("4. View dashboard: http://localhost:3001")

if __name__ == "__main__":
    main()
