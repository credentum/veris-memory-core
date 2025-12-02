#!/usr/bin/env python3
"""
Smoke test runner for CI/CD pipeline.
Quick sanity checks to verify deployment.
"""

import sys
import time
import json
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import requests
from colorama import init, Fore, Style


# Initialize colorama for cross-platform color support
init(autoreset=True)


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration_ms: float
    message: Optional[str] = None
    details: Optional[Dict] = None


class SmokeTestRunner:
    """Runner for smoke tests in CI."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.results: List[TestResult] = []
        
        # Service URLs
        self.api_url = self.config.get("api_url", "http://localhost:8000")
        self.qdrant_url = self.config.get("qdrant_url", "http://localhost:6333")
        self.neo4j_url = self.config.get("neo4j_url", "http://localhost:7474")
        
        # Test configuration
        self.timeout = self.config.get("timeout", 10)
        self.fail_fast = self.config.get("fail_fast", False)
        
    def test_api_health(self) -> TestResult:
        """Test API health endpoint."""
        start = time.time()
        
        try:
            response = requests.get(
                f"{self.api_url}/health",
                timeout=self.timeout
            )
            duration = (time.time() - start) * 1000
            
            if response.status_code == 200:
                return TestResult(
                    name="API Health",
                    passed=True,
                    duration_ms=duration,
                    message="API is healthy"
                )
            else:
                return TestResult(
                    name="API Health",
                    passed=False,
                    duration_ms=duration,
                    message=f"Unexpected status: {response.status_code}"
                )
                
        except Exception as e:
            return TestResult(
                name="API Health",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                message=str(e)
            )
    
    def test_qdrant_health(self) -> TestResult:
        """Test Qdrant health."""
        start = time.time()
        
        try:
            # Qdrant uses root endpoint for health, not /health
            response = requests.get(
                f"{self.qdrant_url}/",
                timeout=self.timeout
            )
            duration = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                # Check for expected Qdrant response structure
                if "title" in data and "qdrant" in data.get("title", "").lower():
                    return TestResult(
                        name="Qdrant Health",
                        passed=True,
                        duration_ms=duration,
                        message=f"Qdrant is healthy (v{data.get('version', 'unknown')})"
                    )
                else:
                    return TestResult(
                        name="Qdrant Health",
                        passed=False,
                        duration_ms=duration,
                        message="Qdrant response missing expected fields"
                    )
            else:
                return TestResult(
                    name="Qdrant Health",
                    passed=False,
                    duration_ms=duration,
                    message=f"Unexpected status: {response.status_code}"
                )
                
        except Exception as e:
            return TestResult(
                name="Qdrant Health",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                message=str(e)
            )
    
    def test_qdrant_collection(self) -> TestResult:
        """Test Qdrant collection exists with correct config."""
        start = time.time()
        
        try:
            response = requests.get(
                f"{self.qdrant_url}/collections/context_embeddings",
                timeout=self.timeout
            )
            duration = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                result = data.get("result", {})
                config = result.get("config", {})
                params = config.get("params", {})
                vectors = params.get("vectors", {})
                
                # Check dimensions
                size = vectors.get("size", 0)
                distance = vectors.get("distance", "")
                
                if size == 384 and distance.lower() == "cosine":
                    return TestResult(
                        name="Qdrant Collection",
                        passed=True,
                        duration_ms=duration,
                        message="Collection configured correctly",
                        details={
                            "dimensions": size,
                            "distance": distance,
                            "vectors_count": result.get("vectors_count", 0)
                        }
                    )
                else:
                    return TestResult(
                        name="Qdrant Collection",
                        passed=False,
                        duration_ms=duration,
                        message=f"Wrong config: dim={size}, distance={distance}",
                        details={"dimensions": size, "distance": distance}
                    )
            else:
                return TestResult(
                    name="Qdrant Collection",
                    passed=False,
                    duration_ms=duration,
                    message=f"Collection not found: {response.status_code}"
                )
                
        except Exception as e:
            return TestResult(
                name="Qdrant Collection",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                message=str(e)
            )
    
    def test_neo4j_connectivity(self) -> TestResult:
        """Test Neo4j connectivity."""
        start = time.time()
        
        try:
            # Neo4j REST API endpoint
            response = requests.get(
                f"{self.neo4j_url}/db/data/",
                timeout=self.timeout,
                auth=("neo4j", "password")  # Default auth for testing
            )
            duration = (time.time() - start) * 1000
            
            if response.status_code in [200, 401]:  # 401 means server is up but auth failed
                return TestResult(
                    name="Neo4j Connectivity",
                    passed=True,
                    duration_ms=duration,
                    message="Neo4j is reachable"
                )
            else:
                return TestResult(
                    name="Neo4j Connectivity",
                    passed=False,
                    duration_ms=duration,
                    message=f"Unexpected status: {response.status_code}"
                )
                
        except Exception as e:
            return TestResult(
                name="Neo4j Connectivity",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                message=str(e)
            )
    
    def test_api_search_endpoint(self) -> TestResult:
        """Test API search endpoint."""
        start = time.time()
        
        try:
            # Use the correct MCP tool endpoint for retrieval
            payload = {
                "query": "test query",
                "limit": 5,
                "type": "all"
            }
            
            response = requests.post(
                f"{self.api_url}/tools/retrieve_context",
                json=payload,
                timeout=self.timeout
            )
            duration = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                
                # Check response structure
                if "results" in data:
                    return TestResult(
                        name="API Search",
                        passed=True,
                        duration_ms=duration,
                        message="Search endpoint working",
                        details={"result_count": len(data.get("results", []))}
                    )
                else:
                    return TestResult(
                        name="API Search",
                        passed=False,
                        duration_ms=duration,
                        message="Invalid response structure"
                    )
            elif response.status_code == 404:
                # Endpoint might not exist yet
                return TestResult(
                    name="API Search",
                    passed=False,
                    duration_ms=duration,
                    message="Search endpoint not found (404)"
                )
            else:
                return TestResult(
                    name="API Search",
                    passed=False,
                    duration_ms=duration,
                    message=f"Unexpected status: {response.status_code}"
                )
                
        except Exception as e:
            return TestResult(
                name="API Search",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                message=str(e)
            )
    
    def test_dimension_consistency(self) -> TestResult:
        """Test dimension consistency across configs."""
        start = time.time()

        try:
            # Check if production config exists
            import yaml
            from pathlib import Path

            config_path = Path("config/production_locked_config.yaml")
            
            if not config_path.exists():
                return TestResult(
                    name="Dimension Consistency",
                    passed=False,
                    duration_ms=(time.time() - start) * 1000,
                    message="Config file not found"
                )
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract dimensions
            embedding_config = config.get("embedding", {})
            dimensions = embedding_config.get("dimensions")
            
            if dimensions == 384:
                return TestResult(
                    name="Dimension Consistency",
                    passed=True,
                    duration_ms=(time.time() - start) * 1000,
                    message="Dimensions correctly set to 384",
                    details={"dimensions": dimensions}
                )
            else:
                return TestResult(
                    name="Dimension Consistency",
                    passed=False,
                    duration_ms=(time.time() - start) * 1000,
                    message=f"Wrong dimensions: {dimensions}",
                    details={"dimensions": dimensions}
                )
                
        except Exception as e:
            return TestResult(
                name="Dimension Consistency",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                message=str(e)
            )
    
    def run_all_tests(self) -> Tuple[bool, List[TestResult]]:
        """
        Run all smoke tests.
        
        Returns:
            Tuple of (all_passed, results)
        """
        tests = [
            self.test_api_health,
            self.test_qdrant_health,
            self.test_qdrant_collection,
            self.test_neo4j_connectivity,
            self.test_api_search_endpoint,
            self.test_dimension_consistency
        ]
        
        self.results = []
        all_passed = True
        
        for test_func in tests:
            try:
                result = test_func()
                self.results.append(result)
                
                if not result.passed:
                    all_passed = False
                    
                    if self.fail_fast:
                        break
                        
            except Exception as e:
                # Catch any unexpected errors
                result = TestResult(
                    name=test_func.__name__,
                    passed=False,
                    duration_ms=0,
                    message=f"Unexpected error: {e}"
                )
                self.results.append(result)
                all_passed = False
                
                if self.fail_fast:
                    break
        
        return all_passed, self.results
    
    def print_results(self, results: List[TestResult], verbose: bool = False):
        """Print test results with colors."""
        print("\n" + "=" * 60)
        print("SMOKE TEST RESULTS")
        print("=" * 60 + "\n")
        
        for result in results:
            if result.passed:
                status = f"{Fore.GREEN}✓ PASS{Style.RESET_ALL}"
            else:
                status = f"{Fore.RED}✗ FAIL{Style.RESET_ALL}"
            
            print(f"{status} {result.name:<25} ({result.duration_ms:.0f}ms)")
            
            if result.message:
                print(f"       {result.message}")
            
            if verbose and result.details:
                print(f"       Details: {json.dumps(result.details, indent=2)}")
        
        # Summary
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        total_time = sum(r.duration_ms for r in results)
        
        print("\n" + "-" * 60)
        print(f"Total: {len(results)} tests")
        print(f"Passed: {Fore.GREEN}{passed}{Style.RESET_ALL}")
        print(f"Failed: {Fore.RED}{failed}{Style.RESET_ALL}")
        print(f"Duration: {total_time:.0f}ms")
        print("-" * 60)
        
        if failed == 0:
            print(f"\n{Fore.GREEN}✓ All smoke tests passed!{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}✗ {failed} smoke test(s) failed!{Style.RESET_ALL}")
    
    def export_results(self, results: List[TestResult], format: str = "json") -> str:
        """Export results in specified format."""
        if format == "json":
            data = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "summary": {
                    "total": len(results),
                    "passed": sum(1 for r in results if r.passed),
                    "failed": sum(1 for r in results if not r.passed),
                    "duration_ms": sum(r.duration_ms for r in results)
                },
                "tests": [asdict(r) for r in results]
            }
            return json.dumps(data, indent=2)
        
        elif format == "junit":
            # JUnit XML format for CI integration
            xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>']
            xml_lines.append('<testsuites>')
            xml_lines.append('  <testsuite name="Smoke Tests" tests="{}" failures="{}">'.format(
                len(results),
                sum(1 for r in results if not r.passed)
            ))
            
            for result in results:
                xml_lines.append(f'    <testcase name="{result.name}" time="{result.duration_ms/1000:.3f}">')
                if not result.passed:
                    xml_lines.append(f'      <failure message="{result.message or "Test failed"}"/>')
                xml_lines.append('    </testcase>')
            
            xml_lines.append('  </testsuite>')
            xml_lines.append('</testsuites>')
            
            return '\n'.join(xml_lines)
        
        else:
            raise ValueError(f"Unsupported format: {format}")


def main():
    """CLI for smoke test runner."""
    parser = argparse.ArgumentParser(description='Run smoke tests for CI/CD')
    parser.add_argument('--api-url', default='http://localhost:8000',
                        help='API URL')
    parser.add_argument('--qdrant-url', default='http://localhost:6333',
                        help='Qdrant URL')
    parser.add_argument('--neo4j-url', default='http://localhost:7474',
                        help='Neo4j URL')
    parser.add_argument('--fail-fast', action='store_true',
                        help='Stop on first failure')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--export', choices=['json', 'junit'],
                        help='Export results to format')
    parser.add_argument('--output', help='Output file for export')
    parser.add_argument('--timeout', type=int, default=10,
                        help='Request timeout in seconds')
    
    args = parser.parse_args()
    
    # Create config
    config = {
        'api_url': args.api_url,
        'qdrant_url': args.qdrant_url,
        'neo4j_url': args.neo4j_url,
        'timeout': args.timeout,
        'fail_fast': args.fail_fast
    }
    
    # Run tests
    runner = SmokeTestRunner(config)
    all_passed, results = runner.run_all_tests()
    
    # Print results
    runner.print_results(results, verbose=args.verbose)
    
    # Export if requested
    if args.export:
        output = runner.export_results(results, args.export)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"\nResults exported to: {args.output}")
        else:
            print(f"\n{output}")
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()