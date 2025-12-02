"""
Performance Benchmarking Suite
Sprint 10 Phase 2 - Issue 005: SEC-105
Measures system performance under various load conditions
"""

import pytest
import asyncio
import time
import statistics
import concurrent.futures
import random
import json
import psutil
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import queue

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


@dataclass
class PerformanceMetrics:
    """Performance measurement results"""
    operation: str
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput_rps: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    total_requests: int
    successful_requests: int
    failed_requests: int


@dataclass
class LoadTestResult:
    """Load test execution result"""
    test_name: str
    duration_seconds: float
    metrics: PerformanceMetrics
    resource_usage: Dict[str, Any]
    errors: List[str] = field(default_factory=list)


class ResourceMonitor:
    """Monitor system resources during testing"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return aggregated metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        if not self.metrics:
            return {"cpu_avg": 0, "memory_avg": 0, "peak_memory": 0}
        
        cpu_values = [m["cpu"] for m in self.metrics]
        memory_values = [m["memory"] for m in self.metrics]
        
        return {
            "cpu_avg": statistics.mean(cpu_values),
            "cpu_max": max(cpu_values),
            "memory_avg": statistics.mean(memory_values),
            "memory_max": max(memory_values),
            "peak_memory": max(memory_values),
            "samples": len(self.metrics)
        }
    
    def _monitor_loop(self):
        """Resource monitoring loop"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                self.metrics.append({
                    "timestamp": time.time(),
                    "cpu": cpu_percent,
                    "memory": memory_mb
                })
                
                time.sleep(0.1)  # Monitor every 100ms
            except Exception as e:
                # Continue monitoring even if there's an error
                pass


class TestSecurityPerformance:
    """Test ID: SEC-105-A - Security Feature Performance"""
    
    def test_rbac_authentication_performance(self):
        """Benchmark RBAC authentication performance"""
        from src.auth.rbac import RBACMiddleware, CapabilityManager
        
        middleware = RBACMiddleware()
        cap_manager = CapabilityManager()
        
        # Create test token
        token = cap_manager.create_token(
            user_id="perf_user",
            role="writer",
            capabilities=["store_context", "retrieve_context"]
        )
        
        # Benchmark authentication
        num_requests = 1000
        start_time = time.time()
        successful = 0
        failed = 0
        response_times = []
        
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        for _ in range(num_requests):
            req_start = time.time()
            try:
                result = middleware.authorize_tool("store_context", token)
                if result.authorized:
                    successful += 1
                else:
                    failed += 1
            except Exception:
                failed += 1
            
            response_times.append(time.time() - req_start)
        
        total_time = time.time() - start_time
        resource_metrics = monitor.stop_monitoring()
        
        # Calculate metrics
        throughput = num_requests / total_time
        avg_response = statistics.mean(response_times)
        p95_response = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        
        # Performance assertions
        assert avg_response < 0.01, f"RBAC auth too slow: {avg_response:.4f}s avg"
        assert throughput > 1000, f"RBAC throughput too low: {throughput:.1f} RPS"
        assert successful >= num_requests * 0.99, f"Too many auth failures: {failed}/{num_requests}"
        
        print(f"RBAC Performance: {throughput:.1f} RPS, {avg_response*1000:.2f}ms avg")
    
    def test_waf_filtering_performance(self):
        """Benchmark WAF filtering performance"""
        from src.security.waf import WAFFilter
        
        waf = WAFFilter()
        
        # Test payloads (mix of legitimate and malicious)
        test_payloads = [
            {"query": "SELECT * FROM contexts WHERE id = 1"},  # Legitimate
            {"query": "' OR 1=1--"},  # SQL injection
            {"input": "<script>alert('xss')</script>"},  # XSS
            {"command": "; ls -la"},  # Command injection
            {"content": "Normal user content"},  # Legitimate
        ]
        
        num_iterations = 1000
        start_time = time.time()
        blocked_count = 0
        allowed_count = 0
        response_times = []
        
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        for i in range(num_iterations):
            payload = test_payloads[i % len(test_payloads)]
            
            req_start = time.time()
            result = waf.check_request(payload)
            response_times.append(time.time() - req_start)
            
            if result.blocked:
                blocked_count += 1
            else:
                allowed_count += 1
        
        total_time = time.time() - start_time
        resource_metrics = monitor.stop_monitoring()
        
        # Calculate metrics
        throughput = num_iterations / total_time
        avg_response = statistics.mean(response_times)
        
        # Performance assertions
        assert avg_response < 0.005, f"WAF filtering too slow: {avg_response:.4f}s avg"
        assert throughput > 2000, f"WAF throughput too low: {throughput:.1f} RPS"
        assert blocked_count > 0, "WAF should block some malicious requests"
        
        print(f"WAF Performance: {throughput:.1f} RPS, {avg_response*1000:.2f}ms avg, {blocked_count} blocked")
    
    def test_rate_limiting_performance(self):
        """Benchmark rate limiting performance"""
        from src.security.waf import WAFRateLimiter
        
        limiter = WAFRateLimiter(requests_per_minute=60000)  # High limit for testing
        
        num_clients = 100
        requests_per_client = 100
        total_requests = num_clients * requests_per_client
        
        response_times = []
        allowed_count = 0
        blocked_count = 0
        
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        # Simulate multiple clients
        for client_id in range(num_clients):
            client_ip = f"192.168.{client_id // 256}.{client_id % 256}"
            
            for _ in range(requests_per_client):
                req_start = time.time()
                result = limiter.check_rate_limit(client_ip)
                response_times.append(time.time() - req_start)
                
                if result.allowed:
                    allowed_count += 1
                else:
                    blocked_count += 1
        
        total_time = time.time() - start_time
        resource_metrics = monitor.stop_monitoring()
        
        # Calculate metrics
        throughput = total_requests / total_time
        avg_response = statistics.mean(response_times)
        
        # Performance assertions
        assert avg_response < 0.001, f"Rate limiting too slow: {avg_response:.4f}s avg"
        assert throughput > 5000, f"Rate limit throughput too low: {throughput:.1f} RPS"
        
        print(f"Rate Limiting Performance: {throughput:.1f} RPS, {avg_response*1000:.2f}ms avg")


class TestConcurrentLoad:
    """Test ID: SEC-105-B - Concurrent Load Testing"""
    
    def test_concurrent_authentication(self):
        """Test concurrent authentication requests"""
        from src.auth.rbac import CapabilityManager
        
        cap_manager = CapabilityManager()
        
        def auth_worker(worker_id: int, num_requests: int) -> Tuple[int, int, float]:
            """Worker function for authentication"""
            successful = 0
            failed = 0
            total_time = 0
            
            for i in range(num_requests):
                start = time.time()
                try:
                    token = cap_manager.create_token(
                        user_id=f"user_{worker_id}_{i}",
                        role="reader",
                        capabilities=["retrieve_context"]
                    )
                    if token:
                        successful += 1
                except Exception:
                    failed += 1
                total_time += time.time() - start
            
            return successful, failed, total_time
        
        # Run concurrent authentication
        num_workers = 50
        requests_per_worker = 20
        
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for worker_id in range(num_workers):
                futures.append(
                    executor.submit(auth_worker, worker_id, requests_per_worker)
                )
            
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        resource_metrics = monitor.stop_monitoring()
        
        # Aggregate results
        total_successful = sum(r[0] for r in results)
        total_failed = sum(r[1] for r in results)
        total_requests = total_successful + total_failed
        
        # Performance assertions
        success_rate = total_successful / total_requests
        throughput = total_requests / total_time
        
        assert success_rate > 0.95, f"High failure rate under load: {success_rate:.2%}"
        assert throughput > 100, f"Concurrent throughput too low: {throughput:.1f} RPS"
        
        print(f"Concurrent Auth: {throughput:.1f} RPS, {success_rate:.2%} success rate")
    
    def test_mixed_security_operations_load(self):
        """Test mixed security operations under load"""
        from src.auth.rbac import RBACMiddleware, CapabilityManager
        from src.security.waf import WAFFilter
        from src.security.input_validator import InputValidator
        
        # Initialize components
        rbac = RBACMiddleware()
        cap_manager = CapabilityManager()
        waf = WAFFilter()
        validator = InputValidator()
        
        # Create test token
        token = cap_manager.create_token(
            user_id="load_test_user",
            role="writer",
            capabilities=["store_context", "retrieve_context"]
        )
        
        def mixed_operations_worker(num_operations: int) -> Dict[str, int]:
            """Worker performing mixed security operations"""
            results = {"rbac": 0, "waf": 0, "validation": 0, "errors": 0}
            
            operations = ["rbac", "waf", "validation"]
            
            for _ in range(num_operations):
                op = random.choice(operations)
                
                try:
                    if op == "rbac":
                        result = rbac.authorize_tool("store_context", token)
                        if result.authorized:
                            results["rbac"] += 1
                    
                    elif op == "waf":
                        test_data = {"query": f"test_query_{random.randint(1, 1000)}"}
                        waf.check_request(test_data)
                        results["waf"] += 1
                    
                    elif op == "validation":
                        test_input = f"test_input_{'x' * random.randint(10, 100)}"
                        result = validator.validate_input(test_input)
                        if result.valid:
                            results["validation"] += 1
                
                except Exception:
                    results["errors"] += 1
            
            return results
        
        # Run mixed load test
        num_workers = 30
        operations_per_worker = 50
        
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for _ in range(num_workers):
                futures.append(
                    executor.submit(mixed_operations_worker, operations_per_worker)
                )
            
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        resource_metrics = monitor.stop_monitoring()
        
        # Aggregate results
        total_ops = sum(sum(r.values()) for r in results)
        total_errors = sum(r["errors"] for r in results)
        
        error_rate = total_errors / total_ops if total_ops > 0 else 1
        throughput = total_ops / total_time
        
        # Performance assertions
        assert error_rate < 0.05, f"High error rate under mixed load: {error_rate:.2%}"
        assert throughput > 200, f"Mixed operations throughput too low: {throughput:.1f} OPS"
        
        print(f"Mixed Load: {throughput:.1f} OPS, {error_rate:.2%} error rate")
    
    def test_memory_usage_under_load(self):
        """Test memory usage doesn't grow excessively under load"""
        from src.security.waf import WAFFilter
        
        waf = WAFFilter()
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate load
        num_requests = 10000
        large_payloads = [
            {"data": "x" * 1000},  # 1KB payloads
            {"query": "SELECT " + ", ".join([f"col_{i}" for i in range(100)])},
            {"content": json.dumps({"items": list(range(1000))})},
        ]
        
        for i in range(num_requests):
            payload = large_payloads[i % len(large_payloads)]
            waf.check_request(payload)
            
            # Check memory every 1000 requests
            if i % 1000 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                
                # Should not grow more than 100MB
                assert memory_growth < 100, \
                    f"Excessive memory growth: {memory_growth:.1f}MB after {i} requests"
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory
        
        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{total_growth:.1f}MB)")


class TestPerformanceRegression:
    """Test ID: SEC-105-C - Performance Regression Testing"""
    
    def test_baseline_performance_metrics(self):
        """Establish baseline performance metrics"""
        from src.auth.rbac import RBACMiddleware, CapabilityManager
        from src.security.waf import WAFFilter
        
        # Test configuration
        num_iterations = 1000
        
        results = {}
        
        # Baseline RBAC performance
        middleware = RBACMiddleware()
        cap_manager = CapabilityManager()
        token = cap_manager.create_token("test_user", "writer", ["store_context"])
        
        start_time = time.time()
        for _ in range(num_iterations):
            middleware.authorize_tool("store_context", token)
        rbac_time = (time.time() - start_time) / num_iterations
        results["rbac_avg_ms"] = rbac_time * 1000
        
        # Baseline WAF performance
        waf = WAFFilter()
        test_request = {"query": "SELECT * FROM contexts WHERE id = 1"}
        
        start_time = time.time()
        for _ in range(num_iterations):
            waf.check_request(test_request)
        waf_time = (time.time() - start_time) / num_iterations
        results["waf_avg_ms"] = waf_time * 1000
        
        # Store baseline metrics (in real implementation, would save to file/database)
        baseline_metrics = {
            "rbac_avg_ms": 5.0,  # 5ms baseline
            "waf_avg_ms": 2.0,   # 2ms baseline
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check for regression (more than 50% slower than baseline)
        for metric, current_value in results.items():
            if metric in baseline_metrics:
                baseline_value = baseline_metrics[metric]
                regression_threshold = baseline_value * 1.5  # 50% regression threshold
                
                assert current_value < regression_threshold, \
                    f"Performance regression in {metric}: {current_value:.2f}ms > {regression_threshold:.2f}ms"
        
        print(f"Performance Baseline: RBAC={results['rbac_avg_ms']:.2f}ms, WAF={results['waf_avg_ms']:.2f}ms")
    
    def test_scalability_limits(self):
        """Test performance at scalability limits"""
        from src.security.waf import WAFRateLimiter
        
        limiter = WAFRateLimiter()
        
        # Test different client loads
        client_loads = [10, 50, 100, 500, 1000]
        results = {}
        
        for num_clients in client_loads:
            start_time = time.time()
            
            # Simulate clients
            for client_id in range(num_clients):
                client_ip = f"10.0.{client_id // 256}.{client_id % 256}"
                for _ in range(10):  # 10 requests per client
                    limiter.check_rate_limit(client_ip)
            
            total_time = time.time() - start_time
            throughput = (num_clients * 10) / total_time
            results[num_clients] = throughput
        
        # Verify scalability
        # Should maintain reasonable performance even at high client counts
        for clients, throughput in results.items():
            if clients <= 100:
                assert throughput > 1000, f"Poor performance at {clients} clients: {throughput:.1f} RPS"
            elif clients <= 500:
                assert throughput > 500, f"Poor scalability at {clients} clients: {throughput:.1f} RPS"
        
        print(f"Scalability test results: {results}")


class TestStressConditions:
    """Test ID: SEC-105-D - Stress Testing"""
    
    def test_extreme_load_stability(self):
        """Test system stability under extreme load"""
        from src.security.waf import WAFFilter
        
        waf = WAFFilter()
        
        # Extreme load parameters
        duration_seconds = 30
        target_rps = 5000
        
        # Tracking
        successful_requests = 0
        failed_requests = 0
        response_times = []
        
        def stress_worker(stop_event: threading.Event):
            nonlocal successful_requests, failed_requests
            
            while not stop_event.is_set():
                start = time.time()
                try:
                    # Vary request types
                    if random.random() < 0.1:
                        # 10% malicious requests
                        payload = {"query": "' OR 1=1--"}
                    else:
                        # 90% legitimate requests
                        payload = {"query": f"SELECT * FROM table WHERE id = {random.randint(1, 1000)}"}
                    
                    result = waf.check_request(payload)
                    successful_requests += 1
                    response_times.append(time.time() - start)
                    
                except Exception:
                    failed_requests += 1
                
                # Small delay to control rate
                time.sleep(0.0001)  # 0.1ms delay
        
        # Start stress test
        num_workers = 20
        stop_event = threading.Event()
        
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        threads = []
        for _ in range(num_workers):
            thread = threading.Thread(target=stress_worker, args=(stop_event,))
            thread.start()
            threads.append(thread)
        
        # Run for specified duration
        time.sleep(duration_seconds)
        stop_event.set()
        
        # Wait for threads to finish
        for thread in threads:
            thread.join()
        
        resource_metrics = monitor.stop_monitoring()
        
        # Calculate results
        total_requests = successful_requests + failed_requests
        error_rate = failed_requests / total_requests if total_requests > 0 else 1
        actual_rps = total_requests / duration_seconds
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]
        else:
            avg_response_time = 0
            p95_response_time = 0
        
        # Stress test assertions
        assert error_rate < 0.1, f"High error rate under stress: {error_rate:.2%}"
        assert avg_response_time < 0.1, f"Response time too high under stress: {avg_response_time:.3f}s"
        assert actual_rps > target_rps * 0.5, f"Throughput too low under stress: {actual_rps:.1f} RPS"
        
        print(f"Stress Test: {actual_rps:.1f} RPS, {error_rate:.2%} errors, {avg_response_time*1000:.1f}ms avg")
    
    def test_resource_exhaustion_resistance(self):
        """Test resistance to resource exhaustion attacks"""
        from src.security.input_validator import InputValidator
        
        validator = InputValidator()
        
        # Try to exhaust resources with large inputs
        large_inputs = []
        for size in [1000, 5000, 10000, 50000, 100000]:  # Various sizes up to 100KB
            large_inputs.append("x" * size)
        
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_time = time.time()
        
        processed = 0
        rejected = 0
        
        # Process large inputs
        for large_input in large_inputs * 10:  # Process each size 10 times
            try:
                result = validator.validate_input(large_input, input_type="content")
                if result.valid:
                    processed += 1
                else:
                    rejected += 1
            except Exception:
                rejected += 1
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        memory_growth = end_memory - start_memory
        processing_time = end_time - start_time
        
        # Resource exhaustion resistance assertions
        assert memory_growth < 200, f"Excessive memory growth: {memory_growth:.1f}MB"
        assert processing_time < 10, f"Processing took too long: {processing_time:.1f}s"
        assert rejected > 0, "Should reject some oversized inputs"
        
        print(f"Resource test: {memory_growth:.1f}MB growth, {processing_time:.1f}s duration, {rejected} rejected")


def create_performance_report(test_results: List[LoadTestResult]) -> Dict[str, Any]:
    """Create comprehensive performance report"""
    
    report = {
        "summary": {
            "test_date": datetime.utcnow().isoformat(),
            "total_tests": len(test_results),
            "passed_tests": sum(1 for r in test_results if not r.errors),
            "failed_tests": sum(1 for r in test_results if r.errors)
        },
        "performance_metrics": {},
        "resource_usage": {},
        "recommendations": []
    }
    
    # Aggregate metrics by test type
    for result in test_results:
        if result.test_name not in report["performance_metrics"]:
            report["performance_metrics"][result.test_name] = {
                "avg_response_time": result.metrics.avg_response_time,
                "throughput_rps": result.metrics.throughput_rps,
                "error_rate": result.metrics.error_rate,
                "memory_usage_mb": result.metrics.memory_usage_mb
            }
    
    # Generate recommendations
    for test_name, metrics in report["performance_metrics"].items():
        if metrics["error_rate"] > 0.05:
            report["recommendations"].append(f"High error rate in {test_name}: investigate stability")
        
        if metrics["avg_response_time"] > 0.1:
            report["recommendations"].append(f"Slow response time in {test_name}: optimize performance")
        
        if metrics["memory_usage_mb"] > 500:
            report["recommendations"].append(f"High memory usage in {test_name}: check for memory leaks")
    
    if not report["recommendations"]:
        report["recommendations"].append("Performance metrics are within acceptable ranges")
    
    return report


if __name__ == "__main__":
    # Run performance benchmarks
    pytest.main([__file__, "-v", "-s"])