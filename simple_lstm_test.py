#!/usr/bin/env python3
"""
Simple test to verify LSTM agent structure and methods exist.
"""

import inspect
import sys
import os

def test_lstm_structure():
    """Test that LSTM agent has the expected structure."""
    print("Testing LSTM Agent Structure...")
    
    # Import the module in a way that avoids Flask dependencies
    try:
        # Read the file and check for key components
        with open('agents-python/main.py', 'r') as f:
            content = f.read()
        
        # Check for key components
        checks = [
            ('LSTMModel class', 'class LSTMModel'),
            ('LSTMPricePredictor class', 'class LSTMPricePredictor'),
            ('prepare_data method', 'def prepare_data'),
            ('train_model method', 'def train_model'),
            ('_predict_price method', 'def _predict_price'),
            ('model_trained attribute', 'self.model_trained = False')
        ]
        
        all_passed = True
        for name, pattern in checks:
            if pattern in content:
                print(f"✓ Found {name}")
            else:
                print(f"✗ Missing {name}")
                all_passed = False
                
        return all_passed
        
    except Exception as e:
        print(f"✗ Error testing LSTM structure: {e}")
        return False

def test_lstm_methods():
    """Test that LSTM agent has expected methods."""
    print("\nTesting LSTM Agent Methods...")
    
    try:
        # Read the file and extract method names
        with open('agents-python/main.py', 'r') as f:
            content = f.read()
        
        # Look for method definitions in LSTMPricePredictor class
        lines = content.split('\n')
        in_lstm_class = False
        methods_found = []
        
        for line in lines:
            if 'class LSTMPricePredictor' in line:
                in_lstm_class = True
                continue
            elif in_lstm_class and line.strip() == '' and 'class ' in lines[lines.index(line)+1]:
                # End of LSTM class
                break
            elif in_lstm_class and line.strip().startswith('def '):
                method_name = line.strip().split('(')[0].replace('def ', '')
                methods_found.append(method_name)
                print(f"✓ Found method: {method_name}")
        
        expected_methods = ['__init__', 'prepare_data', 'initialize_model', 'train_model', 'run', '_predict_price']
        missing_methods = [m for m in expected_methods if m not in methods_found]
        
        if not missing_methods:
            print("✓ All expected methods found")
            return True
        else:
            print(f"✗ Missing methods: {missing_methods}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing LSTM methods: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Enhanced LSTM Agent Structure")
    print("=" * 40)
    
    tests = [
        test_lstm_structure,
        test_lstm_methods
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All structure tests passed! LSTM agent enhancement looks good.")
        return 0
    else:
        print("✗ Some structure tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())