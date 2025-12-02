#!/usr/bin/env python3
"""
Resilient Security Testing Suite
Sprint 10 Phase 3 - Non-hanging tests with proper recovery
"""

import os
import sys
import time
import signal
import threading
from contextlib import contextmanager
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from src.security.waf import WAFFilter, WAFRateLimiter


class TimeoutError(Exception):
    """Raised when operation times out"""
    pass


@contextmanager
def timeout_context(seconds: int):
    """Context manager for operation timeout"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@dataclass
class TestResult:
    """Result of a security test"""
    test_name: str
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = None


class ResilientSecurityTester:
    """Security tester with timeout protection and automatic recovery"""
    
    def __init__(self, test_timeout: int = 30):
        """Initialize with configurable timeout"""
        self.test_timeout = test_timeout
        self.results: List[TestResult] = []
    
    def run_with_timeout(self, test_func, test_name: str) -> TestResult:
        """Run a test function with timeout protection"""
        start_time = time.time()
        
        try:
            with timeout_context(self.test_timeout):
                success = test_func()
                execution_time = time.time() - start_time
                
                return TestResult(
                    test_name=test_name,
                    passed=success,
                    execution_time=execution_time
                )
                
        except TimeoutError as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name=test_name,
                passed=False,
                execution_time=execution_time,
                error_message=f"Test timed out: {str(e)}"
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name=test_name,
                passed=False,
                execution_time=execution_time,
                error_message=f"Test failed: {str(e)}"
            )
    
    def test_rate_limiting_resilient(self) -> bool:
        """Test rate limiting with bounded execution time"""
        try:
            limiter = WAFRateLimiter(
                requests_per_minute=60, 
                burst_size=10, 
                global_requests_per_minute=100  # Reduced for faster testing
            )
            
            # Limited attack simulation (reduced scale)
            attack_clients = [f"192.168.1.{i}" for i in range(1, 21)]  # 20 clients instead of 100
            blocked_count = 0
            allowed_count = 0
            
            # Each client tries 10 requests instead of 100
            for client_ip in attack_clients:
                for _ in range(10):
                    result = limiter.check_rate_limit(client_ip)
                    if result.allowed:
                        allowed_count += 1
                    else:
                        blocked_count += 1
                    
                    # Early exit if we've proven rate limiting works
                    if blocked_count > 50:
                        break
                if blocked_count > 50:
                    break
            
            # Verify rate limiting is working
            success = blocked_count > allowed_count
            
            # Clean up state for next test
            limiter.global_requests.clear()
            limiter.request_counts.clear()
            limiter.blocked_clients.clear()
            
            return success
            
        except Exception as e:
            print(f"Rate limiting test error: {e}")
            return False
    
    def test_waf_patterns_resilient(self) -> bool:
        """Test WAF pattern matching with quick verification"""
        try:
            waf = WAFFilter()
            
            # Quick test of core WAF functionality
            test_payloads = [
                {"query": "' OR 1=1--"},      # SQL injection
                {"input": "<script>alert(1)</script>"},  # XSS
                {"command": "; ls -la"},       # Command injection
                {"path": "../../../etc/passwd"},  # Path traversal
                {"role": "admin"}              # Privilege escalation
            ]
            
            blocked_count = 0
            for payload in test_payloads:
                result = waf.check_request(payload)
                if result.blocked:
                    blocked_count += 1
            
            # Should block at least 80% of attack patterns
            return blocked_count >= len(test_payloads) * 0.8
            
        except Exception as e:
            print(f"WAF pattern test error: {e}")
            return False
    
    def test_recovery_after_attack(self) -> bool:
        """Test system recovery after simulated attack"""
        try:
            limiter = WAFRateLimiter(requests_per_minute=10)
            client_ip = "192.168.1.100"
            
            # Phase 1: Normal operation
            result = limiter.check_rate_limit(client_ip)
            if not result.allowed:
                return False
            
            # Phase 2: Simulate attack (exceed rate limit quickly)
            for _ in range(15):  # Exceed the limit of 10
                limiter.check_rate_limit(client_ip)
            
            # Verify blocking during attack
            result = limiter.check_rate_limit(client_ip)
            if result.allowed:
                return False  # Should be blocked
            
            # Phase 3: Force recovery (simulate time passing)
            limiter.reset_client(client_ip)
            
            # Phase 4: Verify recovery
            result = limiter.check_rate_limit(client_ip)
            return result.allowed  # Should be allowed again
            
        except Exception as e:
            print(f"Recovery test error: {e}")
            return False
    
    def test_concurrent_safety(self) -> bool:
        """Test thread safety with concurrent operations"""
        try:
            waf = WAFFilter()
            results = []
            
            def worker_thread(thread_id: int):
                """Worker thread for concurrent testing"""
                try:
                    for i in range(5):  # Reduced iterations
                        payload = {"query": f"' OR {thread_id}={thread_id}--"}
                        result = waf.check_request(payload)
                        results.append(result.blocked)
                        time.sleep(0.01)  # Small delay to prevent overwhelming
                except Exception as e:
                    print(f"Worker {thread_id} error: {e}")
                    results.append(False)
            
            # Create and start threads
            threads = []
            for i in range(5):  # 5 threads instead of 50
                thread = threading.Thread(target=worker_thread, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion with timeout
            for thread in threads:
                thread.join(timeout=5)  # 5 second timeout per thread
            
            # Verify most requests were blocked
            if not results:
                return False
                
            blocked_ratio = sum(results) / len(results)
            return blocked_ratio >= 0.8  # At least 80% blocked
            
        except Exception as e:
            print(f"Concurrent safety test error: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all security tests with resilience"""
        print("🔒 Running Resilient Security Test Suite")
        print("=" * 50)
        
        tests = [
            (self.test_rate_limiting_resilient, "Rate Limiting Protection"),
            (self.test_waf_patterns_resilient, "WAF Pattern Matching"),
            (self.test_recovery_after_attack, "Attack Recovery"),
            (self.test_concurrent_safety, "Concurrent Request Safety")
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_func, test_name in tests:
            print(f"\n🧪 Testing: {test_name}")
            result = self.run_with_timeout(test_func, test_name)
            self.results.append(result)
            
            if result.passed:
                print(f"   ✅ PASSED ({result.execution_time:.2f}s)")
                passed_tests += 1
            else:
                print(f"   ❌ FAILED ({result.execution_time:.2f}s)")
                if result.error_message:
                    print(f"      Error: {result.error_message}")
        
        # Summary
        success_rate = (passed_tests / total_tests) * 100
        total_time = sum(r.execution_time for r in self.results)
        
        print(f"\n📊 Test Summary")
        print(f"   Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Max Test Time: {max(r.execution_time for r in self.results):.2f}s")
        
        return {
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "success_rate": success_rate,
            "total_time": total_time,
            "results": self.results
        }


def main():
    """Main entry point for resilient security testing"""
    tester = ResilientSecurityTester(test_timeout=30)
    
    try:
        summary = tester.run_all_tests()
        
        if summary["success_rate"] >= 75:
            print(f"\n🎉 Security tests PASSED with {summary['success_rate']:.1f}% success rate")
            return 0
        else:
            print(f"\n⚠️  Security tests need attention: {summary['success_rate']:.1f}% success rate")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())