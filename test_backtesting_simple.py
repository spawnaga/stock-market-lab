#!/usr/bin/env python3
"""
Simple test script for the backtesting framework.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agents-python'))

def test_imports():
    """Test that we can import the backtesting framework."""
    try:
        from backtesting_framework import BacktestingEngine, SimpleMovingAverageStrategy, BacktestResult
        print("✓ Successfully imported backtesting framework components")
        return True
    except Exception as e:
        print(f"✗ Failed to import backtesting framework: {e}")
        return False

def test_classes_exist():
    """Test that key classes exist."""
    try:
        from backtesting_framework import BacktestingEngine, SimpleMovingAverageStrategy, BacktestResult
        
        # Test that classes can be instantiated
        engine = BacktestingEngine(None)
        strategy = SimpleMovingAverageStrategy("Test", 10000.0)
        result = BacktestResult()
        
        print("✓ All key classes can be instantiated")
        return True
    except Exception as e:
        print(f"✗ Failed to instantiate classes: {e}")
        return False

if __name__ == "__main__":
    print("Testing Backtesting Framework Imports...")
    
    success = True
    success &= test_imports()
    success &= test_classes_exist()
    
    if success:
        print("\n✓ All tests passed! Backtesting framework is ready.")
    else:
        print("\n✗ Some tests failed.")
        sys.exit(1)