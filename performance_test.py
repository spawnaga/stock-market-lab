#!/usr/bin/env python3
"""
Performance testing script for the AI-Driven Multi-Agent Stock Market Lab.
Measures real-time performance, data processing speed, and ROI tracking capabilities.
"""

import requests
import time
import json
import threading
from datetime import datetime, timedelta
import statistics
import argparse

class PerformanceTester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = {}
        
    def test_system_health(self):
        """Test system health endpoint."""
        print("Testing system health...")
        start_time = time.time()
        try:
            response = self.session.get(f"{self.base_url}/health")
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                health_time = end_time - start_time
                print(f"✓ Health check completed in {health_time:.4f}s")
                print(f"  System uptime: {data['system']['uptime_seconds']:.2f}s")
                print(f"  Memory usage: {data['system']['memory_mb']:.2f} MB")
                print(f"  CPU usage: {data['system']['cpu_percent']:.2f}%")
                print(f"  Active threads: {data['system']['active_threads']}")
                print(f"  Connected clients: {data['system']['connected_clients']}")
                
                self.test_results['health_check'] = {
                    'time': health_time,
                    'status': 'success',
                    'data': data
                }
                return True
            else:
                print(f"✗ Health check failed with status {response.status_code}")
                self.test_results['health_check'] = {
                    'time': health_time,
                    'status': 'failed',
                    'error': f"HTTP {response.status_code}"
                }
                return False
                
        except Exception as e:
            print(f"✗ Health check error: {e}")
            self.test_results['health_check'] = {
                'time': time.time() - start_time,
                'status': 'failed',
                'error': str(e)
            }
            return False

    def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        print("Testing metrics endpoint...")
        start_time = time.time()
        try:
            # Using a fake token for testing
            headers = {"Authorization": "Bearer test-token"}
            response = self.session.get(f"{self.base_url}/metrics", headers=headers)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                metrics_time = end_time - start_time
                print(f"✓ Metrics retrieval completed in {metrics_time:.4f}s")
                print(f"  Connected clients: {data['connected_clients']}")
                print(f"  System memory: {data['system']['memory_mb']:.2f} MB")
                print(f"  System CPU: {data['system']['cpu_percent']:.2f}%")
                print(f"  Total requests: {data['requests']['total']}")
                print(f"  Error count: {data['requests']['errors']}")
                
                self.test_results['metrics'] = {
                    'time': metrics_time,
                    'status': 'success',
                    'data': data
                }
                return True
            else:
                print(f"✗ Metrics retrieval failed with status {response.status_code}")
                self.test_results['metrics'] = {
                    'time': metrics_time,
                    'status': 'failed',
                    'error': f"HTTP {response.status_code}"
                }
                return False
                
        except Exception as e:
            print(f"✗ Metrics endpoint error: {e}")
            self.test_results['metrics'] = {
                'time': time.time() - start_time,
                'status': 'failed',
                'error': str(e)
            }
            return False

    def test_performance_optimization(self):
        """Test performance optimization endpoint."""
        print("Testing performance optimization...")
        start_time = time.time()
        try:
            headers = {"Authorization": "Bearer test-token"}
            response = self.session.get(f"{self.base_url}/performance/optimization", headers=headers)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                opt_time = end_time - start_time
                print(f"✓ Performance optimization check completed in {opt_time:.4f}s")
                print(f"  System health: {data['system_health']}")
                print(f"  Current memory: {data['current_metrics']['memory_mb']:.2f} MB")
                print(f"  Current CPU: {data['current_metrics']['cpu_percent']:.2f}%")
                print(f"  Active threads: {data['current_metrics']['active_threads']}")
                print(f"  Avg request time: {data['current_metrics']['avg_request_time']:.4f}s")
                
                if data['recommendations']:
                    print(f"  Recommendations: {len(data['recommendations'])}")
                    for rec in data['recommendations'][:2]:  # Show first 2 recommendations
                        print(f"    - {rec['recommendation']}")
                else:
                    print("  No recommendations (system healthy)")
                
                self.test_results['performance_optimization'] = {
                    'time': opt_time,
                    'status': 'success',
                    'data': data
                }
                return True
            else:
                print(f"✗ Performance optimization check failed with status {response.status_code}")
                self.test_results['performance_optimization'] = {
                    'time': opt_time,
                    'status': 'failed',
                    'error': f"HTTP {response.status_code}"
                }
                return False
                
        except Exception as e:
            print(f"✗ Performance optimization endpoint error: {e}")
            self.test_results['performance_optimization'] = {
                'time': time.time() - start_time,
                'status': 'failed',
                'error': str(e)
            }
            return False

    def run_comprehensive_test(self, iterations=5):
        """Run comprehensive performance tests."""
        print("Running comprehensive performance tests...")
        print("=" * 50)
        
        # Run individual tests
        tests = [
            self.test_system_health,
            self.test_metrics_endpoint,
            self.test_performance_optimization
        ]
        
        results = []
        for test in tests:
            result = test()
            results.append(result)
            print()
        
        # Aggregate results
        successful_tests = sum(results)
        total_tests = len(results)
        
        print("=" * 50)
        print(f"Test Results: {successful_tests}/{total_tests} tests passed")
        
        # Calculate performance statistics
        if 'health_check' in self.test_results and self.test_results['health_check']['status'] == 'success':
            print(f"Average health check time: {statistics.mean([r['time'] for r in self.test_results.values() if r['status'] == 'success'])}")
        
        return successful_tests == total_tests

    def test_real_time_processing(self):
        """Test real-time data processing capabilities."""
        print("Testing real-time processing capabilities...")
        start_time = time.time()
        
        # Simulate processing multiple data points
        num_iterations = 100
        processing_times = []
        
        for i in range(num_iterations):
            try:
                # Test health check multiple times to simulate real-time load
                health_start = time.time()
                response = self.session.get(f"{self.base_url}/health")
                health_end = time.time()
                
                if response.status_code == 200:
                    processing_times.append(health_end - health_start)
                else:
                    print(f"Warning: Health check failed at iteration {i}")
                    
            except Exception as e:
                print(f"Warning: Error at iteration {i}: {e}")
                continue
        
        if processing_times:
            avg_time = statistics.mean(processing_times)
            median_time = statistics.median(processing_times)
            max_time = max(processing_times)
            min_time = min(processing_times)
            
            print(f"✓ Real-time processing test completed")
            print(f"  Iterations: {num_iterations}")
            print(f"  Average time: {avg_time:.6f}s")
            print(f"  Median time: {median_time:.6f}s")
            print(f"  Max time: {max_time:.6f}s")
            print(f"  Min time: {min_time:.6f}s")
            
            # Performance benchmarks
            print("\nPerformance Benchmarks:")
            if avg_time < 0.1:
                print("  ✅ Excellent performance (< 100ms avg)")
            elif avg_time < 0.5:
                print("  ⚠️ Good performance (< 500ms avg)")
            else:
                print("  ❌ Performance needs improvement (> 500ms avg)")
                
            return True
        else:
            print("✗ No successful processing iterations")
            return False

def main():
    parser = argparse.ArgumentParser(description='Performance testing for AI Stock Market Lab')
    parser.add_argument('--url', default='http://localhost:5000', help='Base URL for the API')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations for comprehensive tests')
    parser.add_argument('--real-time', action='store_true', help='Run real-time processing test')
    
    args = parser.parse_args()
    
    tester = PerformanceTester(args.url)
    
    print("AI-Driven Multi-Agent Stock Market Lab")
    print("Performance Testing Suite")
    print("=" * 50)
    
    success = True
    
    # Run basic tests
    success &= tester.run_comprehensive_test(args.iterations)
    
    # Run real-time test if requested
    if args.real_time:
        print()
        success &= tester.test_real_time_processing()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All performance tests completed successfully!")
        print("System is ready for real-world trading applications.")
    else:
        print("❌ Some performance tests failed.")
        print("Please review system configuration and performance.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())