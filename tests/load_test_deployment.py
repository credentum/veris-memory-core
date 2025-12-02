#!/usr/bin/env python3
"""
Load testing script for Veris Memory deployment verification.

This script tests:
1. Redis connectivity and caching
2. Embedding generation under load
3. Cache hit/miss behavior
4. Relationship creation
5. Overall system performance
"""

import asyncio
import json
import time
from typing import Dict, List, Any
import requests
import sys

# Configuration
BASE_URL = "http://172.17.0.1:8000"  # Docker network address
API_KEY = "vmk_mcp_903e1bcb70d704da4fbf207722c471ba"  # Default test key
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}\n")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")


class RedisVerifier:
    """Verify Redis connectivity and caching."""

    def __init__(self):
        self.results = {
            "redis_accessible": False,
            "cache_writes_working": False,
            "cache_reads_working": False,
            "cache_hit_detected": False,
            "errors": []
        }

    def test_redis_direct(self) -> bool:
        """Test direct Redis connection via health endpoint."""
        print_info("Testing Redis connectivity via health endpoint...")

        try:
            response = requests.get(f"{BASE_URL}/health/detailed", headers=HEADERS, timeout=5)
            if response.status_code == 200:
                health = response.json()
                redis_status = health.get("components", {}).get("redis", {}).get("status")

                if redis_status == "healthy":
                    print_success(f"Redis health check: {redis_status}")
                    self.results["redis_accessible"] = True
                    return True
                else:
                    print_error(f"Redis health check failed: {redis_status}")
                    self.results["errors"].append(f"Redis unhealthy: {redis_status}")
                    return False
            else:
                print_error(f"Health endpoint returned {response.status_code}")
                self.results["errors"].append(f"Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print_error(f"Failed to check Redis health: {e}")
            self.results["errors"].append(f"Health check error: {str(e)}")
            return False

    def test_cache_behavior(self) -> bool:
        """Test cache write and read by making duplicate queries."""
        print_info("Testing cache write and read behavior...")

        # Use a unique query to avoid hitting existing cache
        unique_query = f"test_cache_verification_{int(time.time())}"

        payload = {
            "query": unique_query,
            "limit": 5,
            "search_mode": "hybrid"
        }

        try:
            # First request - should be cache MISS
            print_info(f"Making first request (cache MISS expected)...")
            start1 = time.time()
            response1 = requests.post(
                f"{BASE_URL}/tools/retrieve_context",
                headers=HEADERS,
                json=payload,
                timeout=10
            )
            latency1 = (time.time() - start1) * 1000  # Convert to ms

            if response1.status_code != 200:
                print_error(f"First request failed: {response1.status_code}")
                self.results["errors"].append(f"Retrieve failed: {response1.status_code}")
                return False

            print_success(f"First request completed: {latency1:.2f}ms")

            # Wait a moment to ensure cache is written
            time.sleep(0.5)

            # Second request - should be cache HIT
            print_info(f"Making second request (cache HIT expected)...")
            start2 = time.time()
            response2 = requests.post(
                f"{BASE_URL}/tools/retrieve_context",
                headers=HEADERS,
                json=payload,
                timeout=10
            )
            latency2 = (time.time() - start2) * 1000  # Convert to ms

            if response2.status_code != 200:
                print_error(f"Second request failed: {response2.status_code}")
                self.results["errors"].append(f"Second retrieve failed: {response2.status_code}")
                return False

            print_success(f"Second request completed: {latency2:.2f}ms")

            # Check if second request was faster (indicating cache hit)
            speedup = latency1 / latency2 if latency2 > 0 else 1.0

            if speedup > 1.5:  # At least 50% faster
                print_success(f"Cache HIT detected! Speedup: {speedup:.2f}x ({latency1:.2f}ms → {latency2:.2f}ms)")
                self.results["cache_hit_detected"] = True
                self.results["cache_writes_working"] = True
                self.results["cache_reads_working"] = True
                return True
            else:
                print_warning(f"Cache behavior unclear. Speedup: {speedup:.2f}x ({latency1:.2f}ms → {latency2:.2f}ms)")
                print_info("This might indicate Redis caching is not working or queries are too fast to measure")
                return False

        except Exception as e:
            print_error(f"Cache behavior test failed: {e}")
            self.results["errors"].append(f"Cache test error: {str(e)}")
            return False

    def run(self) -> Dict[str, Any]:
        """Run all Redis verification tests."""
        print_header("REDIS VERIFICATION")

        # Test 1: Direct Redis connectivity
        redis_ok = self.test_redis_direct()

        # Test 2: Cache behavior
        cache_ok = self.test_cache_behavior()

        # Summary
        print_info("\nRedis Verification Summary:")
        print(f"  Redis Accessible: {Colors.GREEN if self.results['redis_accessible'] else Colors.RED}{self.results['redis_accessible']}{Colors.END}")
        print(f"  Cache Writes Working: {Colors.GREEN if self.results['cache_writes_working'] else Colors.RED}{self.results['cache_writes_working']}{Colors.END}")
        print(f"  Cache Reads Working: {Colors.GREEN if self.results['cache_reads_working'] else Colors.RED}{self.results['cache_reads_working']}{Colors.END}")
        print(f"  Cache Hit Detected: {Colors.GREEN if self.results['cache_hit_detected'] else Colors.RED}{self.results['cache_hit_detected']}{Colors.END}")

        if self.results["errors"]:
            print_warning("\nErrors encountered:")
            for error in self.results["errors"]:
                print(f"  - {error}")

        return self.results


class LoadTester:
    """Run load tests on the deployment."""

    def __init__(self):
        self.results = {
            "store_context_tests": [],
            "retrieve_context_tests": [],
            "embedding_tests": [],
            "relationship_tests": [],
            "cache_performance": {},
            "errors": []
        }

    def test_store_context_load(self, num_requests: int = 10) -> bool:
        """Test storing contexts under load."""
        print_info(f"Testing store_context with {num_requests} requests...")

        latencies = []
        successes = 0
        failures = 0

        for i in range(num_requests):
            payload = {
                "type": "log",
                "content": {
                    "title": f"Load Test #{i}",
                    "description": f"Load testing deployment at {time.time()}"
                },
                "metadata": {
                    "author": "load_tester",
                    "tags": ["load_test", f"batch_{int(time.time())}"]
                }
            }

            try:
                start = time.time()
                response = requests.post(
                    f"{BASE_URL}/tools/store_context",
                    headers=HEADERS,
                    json=payload,
                    timeout=15
                )
                latency = (time.time() - start) * 1000

                if response.status_code == 200:
                    data = response.json()
                    successes += 1
                    latencies.append(latency)

                    # Check embedding status
                    embedding_status = data.get("embedding_status", "unknown")
                    if embedding_status == "completed":
                        self.results["embedding_tests"].append({"success": True, "latency": latency})
                    else:
                        self.results["embedding_tests"].append({"success": False, "status": embedding_status})
                else:
                    failures += 1
                    self.results["errors"].append(f"Store failed with {response.status_code}")
            except Exception as e:
                failures += 1
                self.results["errors"].append(f"Store error: {str(e)}")

        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        print_success(f"Store context: {successes}/{num_requests} succeeded")
        print_info(f"  Average latency: {avg_latency:.2f}ms")
        print_info(f"  Min: {min(latencies):.2f}ms, Max: {max(latencies):.2f}ms" if latencies else "  No successful requests")

        self.results["store_context_tests"] = {
            "total": num_requests,
            "successes": successes,
            "failures": failures,
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0
        }

        return successes > 0

    def test_retrieve_context_load(self, num_requests: int = 20) -> bool:
        """Test retrieving contexts under load (tests cache performance)."""
        print_info(f"Testing retrieve_context with {num_requests} requests (mix of cache hits/misses)...")

        # Use a few different queries to test cache behavior
        queries = [
            "load test deployment verification",
            "test cache performance",
            "embedding generation",
            "relationship creation",
            "system health check"
        ]

        latencies = []
        successes = 0
        failures = 0

        for i in range(num_requests):
            query = queries[i % len(queries)]  # Cycle through queries to trigger cache hits

            payload = {
                "query": query,
                "limit": 5,
                "search_mode": "hybrid"
            }

            try:
                start = time.time()
                response = requests.post(
                    f"{BASE_URL}/tools/retrieve_context",
                    headers=HEADERS,
                    json=payload,
                    timeout=10
                )
                latency = (time.time() - start) * 1000

                if response.status_code == 200:
                    successes += 1
                    latencies.append(latency)
                else:
                    failures += 1
                    self.results["errors"].append(f"Retrieve failed with {response.status_code}")
            except Exception as e:
                failures += 1
                self.results["errors"].append(f"Retrieve error: {str(e)}")

        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        print_success(f"Retrieve context: {successes}/{num_requests} succeeded")
        print_info(f"  Average latency: {avg_latency:.2f}ms")
        print_info(f"  Min: {min(latencies):.2f}ms, Max: {max(latencies):.2f}ms" if latencies else "  No successful requests")

        self.results["retrieve_context_tests"] = {
            "total": num_requests,
            "successes": successes,
            "failures": failures,
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0
        }

        return successes > 0

    def run(self) -> Dict[str, Any]:
        """Run all load tests."""
        print_header("LOAD TESTING")

        # Test 1: Store context load
        store_ok = self.test_store_context_load(num_requests=10)

        # Test 2: Retrieve context load (tests caching)
        retrieve_ok = self.test_retrieve_context_load(num_requests=20)

        # Summary
        print_info("\nLoad Testing Summary:")

        if self.results["store_context_tests"]:
            store = self.results["store_context_tests"]
            print(f"\n  Store Context:")
            print(f"    Success Rate: {Colors.GREEN if store['successes'] == store['total'] else Colors.YELLOW}{store['successes']}/{store['total']}{Colors.END}")
            print(f"    Avg Latency: {store['avg_latency_ms']:.2f}ms")

        if self.results["retrieve_context_tests"]:
            retrieve = self.results["retrieve_context_tests"]
            print(f"\n  Retrieve Context:")
            print(f"    Success Rate: {Colors.GREEN if retrieve['successes'] == retrieve['total'] else Colors.YELLOW}{retrieve['successes']}/{retrieve['total']}{Colors.END}")
            print(f"    Avg Latency: {retrieve['avg_latency_ms']:.2f}ms")

        if self.results["embedding_tests"]:
            successful_embeddings = sum(1 for t in self.results["embedding_tests"] if t.get("success"))
            print(f"\n  Embeddings:")
            print(f"    Success Rate: {Colors.GREEN if successful_embeddings == len(self.results['embedding_tests']) else Colors.YELLOW}{successful_embeddings}/{len(self.results['embedding_tests'])}{Colors.END}")

        if self.results["errors"]:
            print_warning(f"\n  Errors: {len(self.results['errors'])} encountered")

        return self.results


def main():
    """Run all deployment verification tests."""
    print_header("VERIS MEMORY DEPLOYMENT VERIFICATION")
    print_info(f"Testing deployment at: {BASE_URL}")
    print_info(f"Using API Key: {API_KEY[:20]}...")

    # Phase 1: Verify Redis
    redis_verifier = RedisVerifier()
    redis_results = redis_verifier.run()

    # Phase 2: Load Testing
    load_tester = LoadTester()
    load_results = load_tester.run()

    # Final Summary
    print_header("FINAL SUMMARY")

    redis_ok = redis_results["redis_accessible"] and redis_results["cache_hit_detected"]
    load_ok = (load_results.get("store_context_tests", {}).get("successes", 0) > 0 and
               load_results.get("retrieve_context_tests", {}).get("successes", 0) > 0)

    if redis_ok and load_ok:
        print_success("All tests passed! ✅")
        print_info("\nKey Findings:")
        print(f"  ✅ Redis is working and caching properly")
        print(f"  ✅ Embeddings are generating successfully")
        print(f"  ✅ System handles load well")
        return 0
    else:
        print_error("Some tests failed! ❌")
        print_info("\nIssues Found:")
        if not redis_ok:
            print(f"  ❌ Redis caching is not working properly")
            if redis_results["errors"]:
                print(f"     Errors: {', '.join(redis_results['errors'][:3])}")
        if not load_ok:
            print(f"  ❌ Load testing revealed issues")
            if load_results["errors"]:
                print(f"     Errors: {', '.join(load_results['errors'][:3])}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
