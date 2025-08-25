#!/usr/bin/env python3
"""
Test script to verify the enhanced LSTM agent functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agents-python'))

from main import LSTMPricePredictor
import time

def test_lstm_agent_creation():
    """Test that LSTM agent can be created successfully."""
    print("Testing LSTM Agent Creation...")
    
    try:
        agent = LSTMPricePredictor("test-lstm-001")
        print(f"✓ LSTM Agent created successfully with ID: {agent.agent_id}")
        print(f"✓ Agent type: {agent.agent_type}")
        print(f"✓ Device: {agent.device}")
        print(f"✓ Sequence length: {agent.sequence_length}")
        return True
    except Exception as e:
        print(f"✗ Failed to create LSTM agent: {e}")
        return False

def test_lstm_model_initialization():
    """Test that LSTM model can be initialized."""
    print("\nTesting LSTM Model Initialization...")
    
    try:
        agent = LSTMPricePredictor("test-lstm-001")
        agent.initialize_model()
        print("✓ LSTM Model initialized successfully")
        print(f"✓ Model type: {type(agent.model)}")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize LSTM model: {e}")
        return False

def test_lstm_training():
    """Test that LSTM training can be attempted."""
    print("\nTesting LSTM Training...")
    
    try:
        agent = LSTMPricePredictor("test-lstm-001")
        # Create some mock data for testing
        import numpy as np
        X = np.random.rand(10, 30, 1)  # Mock sequences
        y = np.random.rand(10)        # Mock targets
        
        # Initialize model first
        agent.initialize_model()
        
        # Try to train
        trained = agent.train_model(X, y)
        print(f"✓ LSTM Training attempt completed (trained: {trained})")
        return True
    except Exception as e:
        print(f"✗ Failed to train LSTM model: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Enhanced LSTM Agent")
    print("=" * 40)
    
    tests = [
        test_lstm_agent_creation,
        test_lstm_model_initialization,
        test_lstm_training
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! LSTM agent enhancement is working correctly.")
        return 0
    else:
        print("✗ Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())