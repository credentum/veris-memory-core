#!/usr/bin/env python3
"""
Split health endpoints for Kubernetes-style health checks.
Provides separate liveness and readiness probes.
"""

from typing import Dict, List, Optional, Tuple
import asyncio
import time
import json
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import requests

# SPRINT 11: Import dimension validation
try:
    from ..core.config import Config
    from ..core.error_handler import handle_v1_dimension_mismatch
except ImportError:
    from src.core.config import Config
    from src.core.error_handler import handle_v1_dimension_mismatch


class HealthStatus(Enum):
    """Health check status enum."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus
    latency_ms: float
    message: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class HealthResponse:
    """Overall health response."""
    status: HealthStatus
    timestamp: str
    components: List[ComponentHealth]
    total_latency_ms: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "latency_ms": c.latency_ms,
                    "message": c.message,
                    "metadata": c.metadata
                } for c in self.components
            ],
            "total_latency_ms": self.total_latency_ms
        }


class HealthChecker:
    """Health checker for all system components."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.startup_time = time.time()

        # Component URLs - Use Docker service names by default, allow override via config
        # Read from environment variables first (set in docker-compose.yml)
        import os
        self.qdrant_url = self.config.get("qdrant_url",
                                          os.getenv("QDRANT_URL", "http://qdrant:6333"))
        self.neo4j_url = self.config.get("neo4j_url",
                                         os.getenv("NEO4J_HTTP_URL", "http://neo4j:7474"))
        self.api_url = self.config.get("api_url",
                                       os.getenv("API_BASE_URL", "http://context-store:8000"))

        # Qdrant collection name - Read from env var for consistency
        self.qdrant_collection = os.getenv("QDRANT_COLLECTION_NAME", "context_embeddings")

        # Timeouts
        self.liveness_timeout = self.config.get("liveness_timeout", 5)
        self.readiness_timeout = self.config.get("readiness_timeout", 10)
        
    async def check_qdrant_async(self) -> ComponentHealth:
        """Check Qdrant health asynchronously."""
        start = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.qdrant_url}/",  # Qdrant doesn't have /health, use root
                    timeout=aiohttp.ClientTimeout(total=self.readiness_timeout)
                ) as response:
                    latency = (time.time() - start) * 1000
                    
                    if response.status == 200:
                        # Check collection status
                        async with session.get(
                            f"{self.qdrant_url}/collections/{self.qdrant_collection}"
                        ) as coll_response:
                            if coll_response.status == 200:
                                data = await coll_response.json()
                                result = data.get("result", {})
                                return ComponentHealth(
                                    name="qdrant",
                                    status=HealthStatus.HEALTHY,
                                    latency_ms=latency,
                                    metadata={
                                        "vectors_count": result.get("vectors_count", 0),
                                        "indexed_vectors_count": result.get("indexed_vectors_count", 0)
                                    }
                                )
                            else:
                                return ComponentHealth(
                                    name="qdrant",
                                    status=HealthStatus.DEGRADED,
                                    latency_ms=latency,
                                    message="Collection not ready"
                                )
                    else:
                        return ComponentHealth(
                            name="qdrant",
                            status=HealthStatus.UNHEALTHY,
                            latency_ms=latency,
                            message=f"HTTP {response.status}"
                        )
                        
        except asyncio.TimeoutError:
            return ComponentHealth(
                name="qdrant",
                status=HealthStatus.UNHEALTHY,
                latency_ms=self.readiness_timeout * 1000,
                message="Timeout"
            )
        except Exception as e:
            return ComponentHealth(
                name="qdrant",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=str(e)
            )
    
    async def check_neo4j_async(self) -> ComponentHealth:
        """Check Neo4j health asynchronously."""
        start = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                # Neo4j REST API health check - just check if server responds
                # We expect 401 (unauthorized) which still means the server is running
                async with session.get(
                    f"{self.neo4j_url}/db/data/",
                    timeout=aiohttp.ClientTimeout(total=self.readiness_timeout)
                ) as response:
                    latency = (time.time() - start) * 1000
                    
                    # 200 = authenticated, 401 = unauthorized but server is up
                    if response.status in [200, 401]:
                        return ComponentHealth(
                            name="neo4j",
                            status=HealthStatus.HEALTHY,
                            latency_ms=latency
                        )
                    else:
                        return ComponentHealth(
                            name="neo4j",
                            status=HealthStatus.UNHEALTHY,
                            latency_ms=latency,
                            message=f"HTTP {response.status}"
                        )
                        
        except asyncio.TimeoutError:
            return ComponentHealth(
                name="neo4j",
                status=HealthStatus.UNHEALTHY,
                latency_ms=self.readiness_timeout * 1000,
                message="Timeout"
            )
        except Exception as e:
            return ComponentHealth(
                name="neo4j",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=str(e)
            )
    
    async def check_api_async(self) -> ComponentHealth:
        """Check API health asynchronously."""
        start = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/health",  # Use health endpoint for context-store
                    timeout=aiohttp.ClientTimeout(total=self.readiness_timeout)
                ) as response:
                    latency = (time.time() - start) * 1000
                    
                    if response.status == 200:
                        return ComponentHealth(
                            name="api",
                            status=HealthStatus.HEALTHY,
                            latency_ms=latency
                        )
                    else:
                        return ComponentHealth(
                            name="api",
                            status=HealthStatus.UNHEALTHY,
                            latency_ms=latency,
                            message=f"HTTP {response.status}"
                        )
                        
        except asyncio.TimeoutError:
            return ComponentHealth(
                name="api",
                status=HealthStatus.UNHEALTHY,
                latency_ms=self.readiness_timeout * 1000,
                message="Timeout"
            )
        except Exception as e:
            return ComponentHealth(
                name="api",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=str(e)
            )

    async def check_dimension_integrity(self) -> ComponentHealth:
        """SPRINT 11: Check vector dimension integrity for v1.0 compliance."""
        start_time = time.time()
        try:
            # Validate configuration dimensions
            if Config.EMBEDDING_DIMENSIONS != 384:
                latency_ms = (time.time() - start_time) * 1000
                return ComponentHealth(
                    name="dimensions",
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=latency_ms,
                    message=f"Dimension drift: configured {Config.EMBEDDING_DIMENSIONS}, required 384",
                    metadata={
                        "configured_dimensions": Config.EMBEDDING_DIMENSIONS,
                        "required_dimensions": 384,
                        "compliance_status": "FAILED"
                    }
                )
            
            latency_ms = (time.time() - start_time) * 1000
            return ComponentHealth(
                name="dimensions",
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                message="Dimension integrity verified: 384 dimensions",
                metadata={
                    "configured_dimensions": 384,
                    "required_dimensions": 384,
                    "compliance_status": "PASSED"
                }
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return ComponentHealth(
                name="dimensions",
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message=f"Dimension check failed: {e}"
            )
    
    async def liveness_check(self) -> Tuple[HealthStatus, Dict]:
        """
        Liveness probe - checks if the service is alive.
        Returns quickly, only checks basic process health.
        """
        start = time.time()
        
        # Basic liveness: process is running and responsive
        uptime = time.time() - self.startup_time
        
        response = {
            "status": "alive",
            "uptime_seconds": uptime,
            "latency_ms": (time.time() - start) * 1000
        }
        
        return HealthStatus.HEALTHY, response
    
    async def readiness_check(self) -> Tuple[HealthStatus, HealthResponse]:
        """
        Readiness probe - checks if service is ready to accept traffic.
        Checks all dependencies and returns detailed status.
        """
        start = time.time()
        
        # Check all components in parallel (including SPRINT 11 dimension check)
        tasks = [
            self.check_qdrant_async(),
            self.check_neo4j_async(),
            self.check_api_async(),
            self.check_dimension_integrity()  # SPRINT 11: Critical v1.0 compliance check
        ]
        
        components = await asyncio.gather(*tasks)
        
        # Determine overall status
        unhealthy_count = sum(1 for c in components if c.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for c in components if c.status == HealthStatus.DEGRADED)
        
        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        response = HealthResponse(
            status=overall_status,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            components=components,
            total_latency_ms=(time.time() - start) * 1000
        )
        
        return overall_status, response
    
    def liveness_check_sync(self) -> Tuple[int, Dict]:
        """Synchronous wrapper for liveness check."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            status, response = loop.run_until_complete(self.liveness_check())
            http_code = 200 if status == HealthStatus.HEALTHY else 503
            return http_code, response
        finally:
            loop.close()
    
    def readiness_check_sync(self) -> Tuple[int, Dict]:
        """Synchronous wrapper for readiness check."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            status, response = loop.run_until_complete(self.readiness_check())
            http_code = 200 if status == HealthStatus.HEALTHY else 503
            return http_code, response.to_dict()
        finally:
            loop.close()


# FastAPI integration
def create_health_routes(app):
    """Create health check routes for FastAPI application."""
    from fastapi import FastAPI, Response
    
    checker = HealthChecker()
    
    @app.get("/health/live")
    async def liveness():
        """Liveness probe endpoint."""
        status, response = await checker.liveness_check()
        http_code = 200 if status == HealthStatus.HEALTHY else 503
        return Response(
            content=json.dumps(response),
            status_code=http_code,
            media_type="application/json"
        )
    
    @app.get("/health/ready")
    async def readiness():
        """Readiness probe endpoint."""
        status, response = await checker.readiness_check()
        http_code = 200 if status == HealthStatus.HEALTHY else 503
        return Response(
            content=json.dumps(response.to_dict()),
            status_code=http_code,
            media_type="application/json"
        )

    # NOTE: Do NOT override /health endpoint
    # The main.py already has a lightweight /health endpoint for Docker health checks
    # that doesn't query backends. Overriding it breaks container health checks during startup.


# Flask integration
def create_flask_health_blueprint():
    """Create health check blueprint for Flask application."""
    from flask import Blueprint, jsonify
    
    health_bp = Blueprint('health', __name__)
    checker = HealthChecker()
    
    @health_bp.route('/health/live')
    def liveness():
        """Liveness probe endpoint."""
        http_code, response = checker.liveness_check_sync()
        return jsonify(response), http_code
    
    @health_bp.route('/health/ready')
    def readiness():
        """Readiness probe endpoint."""
        http_code, response = checker.readiness_check_sync()
        return jsonify(response), http_code
    
    @health_bp.route('/health')
    def health():
        """Combined health endpoint (for backward compatibility)."""
        return readiness()
    
    return health_bp


# CLI usage
def main():
    """CLI for health checks."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Health check utility')
    parser.add_argument('--check', choices=['live', 'ready', 'both'],
                        default='both', help='Type of health check')
    parser.add_argument('--format', choices=['json', 'text'],
                        default='text', help='Output format')
    parser.add_argument('--qdrant-url', default='http://localhost:6333',
                        help='Qdrant URL')
    parser.add_argument('--neo4j-url', default='http://localhost:7474',
                        help='Neo4j URL')
    parser.add_argument('--api-url', default='http://localhost:8000',
                        help='API URL')
    
    args = parser.parse_args()
    
    config = {
        'qdrant_url': args.qdrant_url,
        'neo4j_url': args.neo4j_url,
        'api_url': args.api_url
    }
    
    checker = HealthChecker(config)
    
    if args.check in ['live', 'both']:
        http_code, response = checker.liveness_check_sync()
        
        if args.format == 'json':
            print(json.dumps(response, indent=2))
        else:
            print(f"Liveness: {'✅ HEALTHY' if http_code == 200 else '❌ UNHEALTHY'}")
            print(f"  Uptime: {response.get('uptime_seconds', 0):.1f}s")
        
        if args.check == 'live':
            exit(0 if http_code == 200 else 1)
    
    if args.check in ['ready', 'both']:
        http_code, response = checker.readiness_check_sync()
        
        if args.format == 'json':
            print(json.dumps(response, indent=2))
        else:
            status = response['status']
            print(f"\nReadiness: {'✅ HEALTHY' if status == 'healthy' else '⚠️ DEGRADED' if status == 'degraded' else '❌ UNHEALTHY'}")
            print(f"  Total latency: {response['total_latency_ms']:.1f}ms")
            print("\n  Components:")
            for comp in response['components']:
                emoji = '✅' if comp['status'] == 'healthy' else '⚠️' if comp['status'] == 'degraded' else '❌'
                print(f"    {emoji} {comp['name']}: {comp['status']} ({comp['latency_ms']:.1f}ms)")
                if comp.get('message'):
                    print(f"       Message: {comp['message']}")
        
        exit(0 if http_code == 200 else 1)


if __name__ == '__main__':
    main()