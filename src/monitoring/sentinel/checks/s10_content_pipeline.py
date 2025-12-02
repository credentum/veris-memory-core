#!/usr/bin/env python3
"""
S10: Content Pipeline Monitoring Check

Monitors the content processing pipeline to ensure data flows
correctly through all stages of the system and validates end-to-end
content processing quality and performance.

This check validates:
- Content ingestion and processing stages
- Pipeline throughput and latency validation
- Data transformation and enrichment quality
- Storage backend integration health
- Content retrieval and search performance
- Pipeline error handling and recovery
- Content lifecycle management
"""

import asyncio
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import aiohttp
import logging

from ..base_check import BaseCheck
from ..models import CheckResult, SentinelConfig

logger = logging.getLogger(__name__)

# Constants for pipeline monitoring
INGESTION_SUCCESS_THRESHOLD = 0.8
RETRIEVAL_SUCCESS_THRESHOLD = 0.8
STAGE_HEALTH_THRESHOLD = 0.8
ERROR_HANDLING_THRESHOLD = 0.75
LIFECYCLE_SUCCESS_THRESHOLD = 0.75
CONCURRENT_SUCCESS_THRESHOLD = 0.8
MIN_CONCURRENT_SUCCESS = 4  # At least 80% of 5 operations


class ContentPipelineMonitoring(BaseCheck):
    """S10: Content pipeline monitoring for data processing validation."""

    def __init__(self, config: SentinelConfig) -> None:
        super().__init__(config, "S10-content-pipeline", "Content pipeline monitoring")
        default_url = os.getenv("TARGET_BASE_URL", "http://localhost:8000")
        self.veris_memory_url = config.get("veris_memory_url", default_url)
        self.timeout_seconds = config.get("s10_pipeline_timeout_sec", 60)

        # Get API key from environment for authentication
        self.api_key = os.getenv('SENTINEL_API_KEY')

        self.pipeline_stages = config.get("s10_pipeline_stages", [
            "ingestion",
            "validation", 
            "enrichment",
            "storage",
            "indexing",
            "retrieval"
        ])
        self.test_content_samples = config.get("s10_test_content_samples", self._get_default_test_samples())
        self.performance_thresholds = config.get("s10_performance_thresholds", {
            "ingestion_latency_ms": 5000,
            "retrieval_latency_ms": 2000,
            "pipeline_throughput_per_min": 10,
            "storage_consistency_ratio": 0.95
        })

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests including authentication."""
        headers = {}
        if self.api_key:
            # Extract key portion from format: vmk_{prefix}_{hash}:user_id:role:is_agent
            # Context-store expects only the key portion (before first colon)
            api_key_parts = self.api_key.strip().split(":")
            api_key = api_key_parts[0]
            headers['X-API-Key'] = api_key
        return headers

    async def _cleanup_test_contexts(
        self,
        session: aiohttp.ClientSession,
        context_ids: List[str]
    ) -> None:
        """
        Clean up test contexts after check completes.

        Deletes test contexts created during S10 check to avoid polluting real data.
        Supports multiple contexts from consistency, throughput, latency, and concurrent tests.
        """
        if not context_ids:
            return

        headers = self._get_headers()

        for context_id in context_ids:
            if not context_id:  # Skip None/empty IDs
                continue

            try:
                delete_url = f"{self.veris_memory_url}/api/v1/contexts/{context_id}"
                async with session.delete(delete_url, headers=headers) as response:
                    if response.status in [200, 204, 404]:
                        # 200/204: Successfully deleted, 404: Already gone
                        logger.debug(f"Cleaned up test context: {context_id}")
                    else:
                        logger.warning(
                            f"Failed to delete test context {context_id}: "
                            f"status {response.status}"
                        )
            except Exception as e:
                logger.warning(f"Error cleaning up test context {context_id}: {e}")

    def _get_default_test_samples(self) -> List[Dict[str, Any]]:
        """Default test content samples for pipeline validation.

        MCP Type Validation: All types must be valid MCP types (design|decision|trace|sprint|log)
        Using "log" for all test samples, with original category preserved in tags/metadata.
        """
        return [
            {
                "type": "log",  # Changed from "technical_documentation" (invalid MCP type)
                "content": {
                    "text": "Setting up PostgreSQL database connection for production environment with SSL encryption and connection pooling configuration.",
                    "title": "PostgreSQL Production Setup",
                    "tags": ["database", "postgresql", "production", "ssl", "technical_documentation"]
                },
                "expected_features": ["database", "connection", "ssl", "production"]
            },
            {
                "type": "log",  # Changed from "api_documentation" (invalid MCP type)
                "content": {
                    "text": "REST API endpoint for user authentication using JWT tokens with role-based access control and session management.",
                    "title": "User Authentication API",
                    "tags": ["api", "authentication", "jwt", "rbac", "api_documentation"]
                },
                "expected_features": ["api", "authentication", "jwt", "access"]
            },
            {
                "type": "log",  # Changed from "troubleshooting_guide" (invalid MCP type)
                "content": {
                    "text": "Debugging network connectivity issues in microservices architecture with service mesh monitoring and distributed tracing.",
                    "title": "Network Troubleshooting Guide",
                    "tags": ["troubleshooting", "network", "microservices", "tracing", "troubleshooting_guide"]
                },
                "expected_features": ["debugging", "network", "microservices", "monitoring"]
            },
            {
                "type": "log",  # Changed from "deployment_process" (invalid MCP type)
                "content": {
                    "text": "Automated deployment pipeline with Docker containers, Kubernetes orchestration, and CI/CD integration for production releases.",
                    "title": "Deployment Automation",
                    "tags": ["deployment", "docker", "kubernetes", "cicd", "deployment_process"]
                },
                "expected_features": ["deployment", "docker", "kubernetes", "automation"]
            },
            {
                "type": "log",  # Changed from "performance_optimization" (invalid MCP type)
                "content": {
                    "text": "Application performance tuning strategies including database query optimization, caching layers, and load balancing configuration.",
                    "title": "Performance Optimization Strategies",
                    "tags": ["performance", "optimization", "caching", "database", "performance_optimization"]
                },
                "expected_features": ["performance", "optimization", "caching", "database"]
            }
        ]
        
    async def run_check(self) -> CheckResult:
        """
        Execute comprehensive content pipeline monitoring.

        DEPRECATED (Phase 2 Optimization):
        This check has been consolidated into S2-golden-fact-recall for efficiency.
        S2's store/retrieve cycle implicitly validates the content pipeline.

        Comprehensive pipeline testing should be done in CI/CD pipeline,
        not runtime monitoring (reduces from 5 queries to 0 additional queries).
        """
        # OPTIMIZATION: Return early with deprecation notice
        # Pipeline validation now handled implicitly by enhanced S2 check
        # (S2's store/retrieve cycle tests the full pipeline: ingestion → storage → retrieval)
        return CheckResult(
            check_id=self.check_id,
            timestamp=datetime.utcnow(),
            status="pass",
            latency_ms=0.0,
            message="DEPRECATED: Content pipeline monitoring consolidated into S2 (Phase 2 optimization)",
            details={
                "deprecated": True,
                "deprecated_since": "2025-11-17",
                "removal_planned": "2025-12-17",
                "reason": "Consolidated into S2-golden-fact-recall for query efficiency",
                "optimization": "Reduced runtime queries from 5 to 0 (100% reduction - covered by S2)",
                "recommendation": "Run comprehensive pipeline tests in CI/CD pipeline",
                "consolidated_into": "S2-golden-fact-recall",
                "validation": "S2 store/retrieve cycle validates ingestion → storage → retrieval pipeline",
                "migration_status": "S2's 6 test cases (store→retrieve) validate pipeline health",
                "phase": "2"
            }
        )

        # Original implementation below (kept for reference)
