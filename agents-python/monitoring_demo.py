#!/usr/bin/env python3
"""
Demonstration script for the enhanced monitoring features in the stock market lab.
This script shows how to use the new monitoring endpoints.
"""

import requests
import json
import time
from datetime import datetime

# Base URL for the agents service
BASE_URL = "http://localhost:5000"

def demo_health_check():
    """Demonstrate the health check endpoint."""
    print("=== Health Check Demo ===")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"System Status: {data['status']}")
            print(f"Agents Running: {data['agents']}")
            print(f"Data Streaming: {data['data_streaming']}")
            print(f"Memory Usage: {data['system']['memory_mb']} MB")
            print(f"CPU Usage: {data['system']['cpu_percent']}%")
            print(f"Uptime: {data['system']['uptime_seconds']:.2f} seconds")
            print(f"Total Requests: {data['metrics']['total_requests']}")
            print(f"Error Count: {data['metrics']['error_count']}")
        else:
            print(f"Health check failed with status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Error calling health check: {e}")

def demo_metrics():
    """Demonstrate the metrics endpoint."""
    print("\n=== Metrics Demo ===")
    try:
        # First get a login token
        login_data = {
            "username": "demo_user",
            "password": "demo_password"
        }
        login_response = requests.post(f"{BASE_URL}/login", json=login_data)
        
        if login_response.status_code == 200:
            token = login_response.json()['token']
            headers = {"Authorization": f"Bearer {token}"}
            
            # Get metrics
            metrics_response = requests.get(f"{BASE_URL}/metrics", headers=headers)
            if metrics_response.status_code == 200:
                data = metrics_response.json()
                print(f"Timestamp: {datetime.fromtimestamp(data['timestamp'])}")
                print(f"Connected Clients: {data['connected_clients']}")
                print(f"System Memory: {data['system']['memory_mb']} MB")
                print(f"System CPU: {data['system']['cpu_percent']}%")
                print(f"Requests per Second: {data['system']['requests_per_second']:.2f}")
                print(f"Average Request Time: {data['system']['avg_request_time']} seconds")
                print(f"Total Requests: {data['requests']['total']}")
                print(f"Error Count: {data['requests']['errors']}")
                
                # Show agent metrics
                print("\nAgent Metrics:")
                for agent_type, metrics in data['agents'].items():
                    print(f"  {agent_type}:")
                    print(f"    Executions: {metrics['executions']}")
                    print(f"    Errors: {metrics['errors']}")
                    print(f"    Avg Execution Time: {metrics['avg_execution_time']} seconds")
            else:
                print(f"Metrics request failed with status {metrics_response.status_code}: {metrics_response.text}")
        else:
            print(f"Login failed with status {login_response.status_code}: {login_response.text}")
    except Exception as e:
        print(f"Error calling metrics: {e}")

def demo_debug_agents():
    """Demonstrate the debug agents endpoint."""
    print("\n=== Debug Agents Demo ===")
    try:
        # First get a login token
        login_data = {
            "username": "demo_user",
            "password": "demo_password"
        }
        login_response = requests.post(f"{BASE_URL}/login", json=login_data)
        
        if login_response.status_code == 200:
            token = login_response.json()['token']
            headers = {"Authorization": f"Bearer {token}"}
            
            # Get debug info
            debug_response = requests.get(f"{BASE_URL}/debug/agents", headers=headers)
            if debug_response.status_code == 200:
                data = debug_response.json()
                print("Agent Debug Information:")
                for agent_name, agent_info in data['agents'].items():
                    print(f"\n{agent_name}:")
                    print(f"  ID: {agent_info['id']}")
                    print(f"  Type: {agent_info['type']}")
                    print(f"  Running: {agent_info['running']}")
                    print(f"  Guardrails Enabled: {agent_info['guardrails_enabled']}")
                    print(f"  Executions: {agent_info['metrics']['executions']}")
                    print(f"  Errors: {agent_info['metrics']['errors']}")
                    print(f"  Last Execution: {datetime.fromtimestamp(agent_info['metrics']['last_execution'])}")
            else:
                print(f"Debug agents request failed with status {debug_response.status_code}: {debug_response.text}")
        else:
            print(f"Login failed with status {login_response.status_code}: {login_response.text}")
    except Exception as e:
        print(f"Error calling debug agents: {e}")

def main():
    """Run all demos."""
    print("Stock Market Lab Monitoring Demo")
    print("=" * 40)
    
    # Wait a moment for the service to be ready
    time.sleep(1)
    
    demo_health_check()
    demo_metrics()
    demo_debug_agents()
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()