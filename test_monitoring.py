#!/usr/bin/env python3
"""
Test script to verify the enhanced monitoring features.
"""

import requests
import json
import time
from datetime import datetime

# Test endpoints
BASE_URL = "http://localhost:5000"
TEST_TOKEN = "test-token"  # In real scenario, this would be a valid JWT token

def test_health_endpoint():
    """Test the enhanced health check endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✓ Health check successful")
            print(f"  Status: {data['status']}")
            print(f"  System Uptime: {data['system']['uptime_seconds']:.2f}s")
            print(f"  Memory Usage: {data['system']['memory_mb']} MB")
            print(f"  CPU Usage: {data['system']['cpu_percent']}%")
            print(f"  Active Threads: {data['system']['active_threads']}")
            print(f"  Connected Clients: {data['system']['connected_clients']}")
            print(f"  Requests/Second: {data['metrics']['requests_per_second']:.2f}")
            return True
        else:
            print(f"✗ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_metrics_endpoint():
    """Test the metrics endpoint."""
    print("\nTesting metrics endpoint...")
    try:
        headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
        response = requests.get(f"{BASE_URL}/metrics", headers=headers)
        if response.status_code == 200:
            data = response.json()
            print("✓ Metrics retrieval successful")
            print(f"  Timestamp: {datetime.fromtimestamp(data['timestamp'])}")
            print(f"  Connected Clients: {data['connected_clients']}")
            print(f"  System Memory: {data['system']['memory_mb']} MB")
            print(f"  System CPU: {data['system']['cpu_percent']}%")
            print(f"  Total Requests: {data['requests']['total']}")
            print(f"  Error Count: {data['requests']['errors']}")
            print(f"  Process Threads: {data['process']['threads']}")
            return True
        else:
            print(f"✗ Metrics retrieval failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Metrics endpoint error: {e}")
        return False

def test_performance_optimization():
    """Test the performance optimization endpoint."""
    print("\nTesting performance optimization endpoint...")
    try:
        headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
        response = requests.get(f"{BASE_URL}/performance/optimization", headers=headers)
        if response.status_code == 200:
            data = response.json()
            print("✓ Performance optimization check successful")
            print(f"  System Health: {data['system_health']}")
            print(f"  Current Memory: {data['current_metrics']['memory_mb']} MB")
            print(f"  Current CPU: {data['current_metrics']['cpu_percent']}%")
            print(f"  Active Threads: {data['current_metrics']['active_threads']}")
            print(f"  Avg Request Time: {data['current_metrics']['avg_request_time']}s")
            if data['recommendations']:
                print(f"  Recommendations: {len(data['recommendations'])}")
                for rec in data['recommendations'][:2]:  # Show first 2 recommendations
                    print(f"    - {rec['recommendation']}")
            else:
                print("  No recommendations (system healthy)")
            return True
        else:
            print(f"✗ Performance optimization check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Performance optimization endpoint error: {e}")
        return False

def main():
    """Run all monitoring tests."""
    print("Running Enhanced Monitoring Tests")
    print("=" * 40)
    
    tests = [
        test_health_endpoint,
        test_metrics_endpoint,
        test_performance_optimization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All monitoring tests passed!")
        return 0
    else:
        print("✗ Some tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())