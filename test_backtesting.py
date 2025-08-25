#!/usr/bin/env python3
"""
Test script for the backtesting framework.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agents-python'))

from backtesting_framework import BacktestingEngine, SimpleMovingAverageStrategy
import json
import redis
from datetime import datetime, timedelta

def test_backtesting_framework():
    """Test the backtesting framework functionality."""
    print("Testing Backtesting Framework...")
    
    # Create a mock Redis client for testing
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    
    # Initialize backtesting engine
    engine = BacktestingEngine(redis_client)
    
    # Create a simple strategy
    strategy = SimpleMovingAverageStrategy("Test_SMA", 10000.0, 5, 20)
    
    # Test data generation
    test_data = [
        {
            'timestamp': '2023-01-01T00:00:00Z',
            'open': 100.0,
            'high': 102.0,
            'low': 98.0,
            'close': 101.0,
            'volume': 1000000
        },
        {
            'timestamp': '2023-01-02T00:00:00Z',
            'open': 101.0,
            'high': 103.0,
            'low': 99.0,
            'close': 102.0,
            'volume': 1200000
        },
        {
            'timestamp': '2023-01-03T00:00:00Z',
            'open': 102.0,
            'high': 104.0,
            'low': 100.0,
            'close': 103.0,
            'volume': 1100000
        }
    ]
    
    # Test strategy logic
    print("Testing strategy logic...")
    print(f"Strategy name: {strategy.name}")
    print(f"Initial capital: {strategy.initial_capital}")
    
    # Test SMA calculations
    for data_point in test_data:
        if 'close' in data_point:
            strategy.update_moving_averages(data_point['close'])
            print(f"Close: {data_point['close']}, Short MA: {strategy.short_ma[-1] if strategy.short_ma else 'N/A'}, Long MA: {strategy.long_ma[-1] if strategy.long_ma else 'N/A'}")
    
    # Test should_buy/should_sell
    print(f"Should buy: {strategy.should_buy(test_data[-1])}")
    print(f"Should sell: {strategy.should_sell(test_data[-1])}")
    
    print("Backtesting framework test completed successfully!")
    return True

if __name__ == "__main__":
    test_backtesting_framework()