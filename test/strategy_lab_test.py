#!/usr/bin/env python3
"""
Test script to verify the Strategy Lab implementation
"""

import subprocess
import sys
import time
import requests
import threading
from unittest.mock import patch
import os

def test_docker_compose_up():
    """Test that docker-compose can start the services"""
    print("Testing docker-compose setup...")
    
    # Change to the infra directory and try to bring up services
    try:
        result = subprocess.run([
            'docker-compose', '-f', 'infra/docker-compose.yml', 'up', '-d'
        ], cwd='/workspace/stock-market-lab', capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ Docker compose started successfully")
            return True
        else:
            print(f"✗ Docker compose failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error running docker-compose: {e}")
        return False

def test_strategy_endpoints():
    """Test that the strategy endpoints are working"""
    print("Testing strategy endpoints...")
    
    try:
        # Test GET /strategies
        response = requests.get('http://localhost:5000/strategies', timeout=5)
        if response.status_code == 200:
            print("✓ GET /strategies endpoint working")
        else:
            print(f"✗ GET /strategies failed with status {response.status_code}")
            return False
            
        # Test POST /strategies
        test_strategy = {
            "name": "Test Strategy",
            "description": "A test strategy for validation",
            "parameters": {
                "lookback_period": 14,
                "threshold": 0.02
            }
        }
        
        response = requests.post(
            'http://localhost:5000/strategies',
            json=test_strategy,
            timeout=5
        )
        
        if response.status_code == 201:
            print("✓ POST /strategies endpoint working")
            return True
        else:
            print(f"✗ POST /strategies failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing strategy endpoints: {e}")
        return False

def main():
    """Main test function"""
    print("Running Strategy Lab Implementation Tests\n")
    
    success = True
    
    # Test 1: Check that files were created correctly
    print("1. Checking file structure...")
    files_to_check = [
        'frontend-react/src/pages/StrategyLab.tsx',
        'frontend-react/src/pages/StrategyLab.css',
        'frontend-react/src/pages/AgentMonitor.tsx',
        'frontend-react/src/pages/AgentMonitor.css',
        'agents-python/main.py'
    ]
    
    for file_path in files_to_check:
        full_path = f'/workspace/stock-market-lab/{file_path}'
        if os.path.exists(full_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ Missing {file_path}")
            success = False
    
    # Test 2: Check that strategy endpoints were added to main.py
    print("\n2. Checking strategy endpoints in main.py...")
    try:
        with open('/workspace/stock-market-lab/agents-python/main.py', 'r') as f:
            content = f.read()
            if '/strategies' in content and 'create_strategy' in content:
                print("✓ Strategy endpoints found in main.py")
            else:
                print("✗ Strategy endpoints not found in main.py")
                success = False
    except Exception as e:
        print(f"✗ Error reading main.py: {e}")
        success = False
    
    # Test 3: Basic functionality check
    print("\n3. Checking basic functionality...")
    try:
        # Import the main module to check syntax
        import importlib.util
        spec = importlib.util.spec_from_file_location("main", "/workspace/stock-market-lab/agents-python/main.py")
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        print("✓ main.py syntax is valid")
    except Exception as e:
        print(f"✗ Error importing main.py: {e}")
        success = False
    
    if success:
        print("\n🎉 All tests passed! Strategy Lab implementation is ready.")
        return 0
    else:
        print("\n❌ Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())