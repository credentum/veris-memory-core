#!/usr/bin/env python3
"""
S9: Graph Intent Validation Check

Validates that graph queries and relationships are correctly
interpreted and produce expected semantic intent and graph traversal results.

This check validates:
- Graph relationship accuracy and consistency
- Semantic intent preservation in graph queries
- Context connectivity validation
- Graph traversal result quality
- Relationship inference correctness
- Graph query optimization effectiveness
- Knowledge graph coherence testing
"""

import asyncio
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import aiohttp
import logging

from ..base_check import BaseCheck
from ..models import CheckResult, SentinelConfig

logger = logging.getLogger(__name__)

# Constants for graph analysis
# PR #300: Lowered thresholds to reflect realistic graph relationship quality
# with sparse/development data. Original values were too strict and caused
# false positives when database had limited semantically related contexts.
ACCURACY_THRESHOLD = 0.5  # Lowered from 0.7
CONNECTIVITY_THRESHOLD = 0.4  # Lowered from 0.6
TRAVERSAL_THRESHOLD = 0.4  # Lowered from 0.6
CLUSTERING_THRESHOLD = 0.4  # Lowered from 0.6
INFERENCE_THRESHOLD = 0.4  # Lowered from 0.5
COHERENCE_THRESHOLD = 0.4  # Lowered from 0.6
PRESERVATION_THRESHOLD = 0.5  # Lowered from 0.6
EXPECTED_RELATIONSHIPS_THRESHOLD = 0.5  # Lowered from 0.6
SEMANTIC_COHERENCE_THRESHOLD = 0.4  # Lowered from 0.5
CLUSTER_COHERENCE_THRESHOLD = 0.3  # Lowered from 0.4
PATH_QUALITY_THRESHOLD = 0.2  # Lowered from 0.3
INFERENCE_QUALITY_THRESHOLD = 0.3  # Lowered from 0.4
CROSS_DOMAIN_COHERENCE_THRESHOLD = 0.2  # Lowered from 0.3
INTENT_PRESERVATION_THRESHOLD = 0.5  # Lowered from 0.7


class GraphIntentValidation(BaseCheck):
    """S9: Graph intent validation for query correctness and relationship accuracy."""

    def __init__(self, config: SentinelConfig) -> None:
        super().__init__(config, "S9-graph-intent", "Graph intent validation")
        default_url = os.getenv("TARGET_BASE_URL", "http://localhost:8000")
        self.veris_memory_url = config.get("veris_memory_url", default_url)
        self.timeout_seconds = config.get("s9_graph_timeout_sec", 45)
        self.max_traversal_depth = config.get("s9_max_traversal_depth", 3)
        self.graph_sample_size = config.get("s9_graph_sample_size", 15)

        # Get API key from environment for authentication
        self.api_key = os.getenv('SENTINEL_API_KEY')

        # Graph intent test scenarios
        self.intent_scenarios = config.get("s9_intent_scenarios", self._get_default_intent_scenarios())

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

    def _get_default_intent_scenarios(self) -> List[Dict[str, Any]]:
        """Default graph intent test scenarios."""
        return [
            {
                "name": "context_relationship_accuracy",
                "description": "Test relationship accuracy between related contexts",
                "contexts": [
                    "Database configuration for PostgreSQL connection",
                    "Setting up PostgreSQL for production environment",
                    "PostgreSQL performance tuning and optimization"
                ],
                "expected_relationships": ["configuration", "database", "postgresql", "performance"]
            },
            {
                "name": "semantic_clustering_validation",
                "description": "Validate semantic clustering in graph relationships",
                "contexts": [
                    "User authentication implementation",
                    "JWT token validation process",
                    "Session management and security",
                    "OAuth2 integration setup"
                ],
                "expected_relationships": ["authentication", "security", "token", "session"]
            },
            {
                "name": "technical_concept_connectivity",
                "description": "Test connectivity between technical concepts",
                "contexts": [
                    "API rate limiting implementation",
                    "Redis caching for performance",
                    "Load balancing configuration",
                    "Performance monitoring setup"
                ],
                "expected_relationships": ["performance", "api", "caching", "monitoring"]
            },
            {
                "name": "workflow_sequence_validation",
                "description": "Validate workflow and process sequences",
                "contexts": [
                    "Initial project setup and configuration",
                    "Development environment setup",
                    "Testing framework implementation",
                    "Production deployment process"
                ],
                "expected_relationships": ["setup", "development", "testing", "deployment"]
            },
            {
                "name": "error_troubleshooting_connections",
                "description": "Test error handling and troubleshooting relationships",
                "contexts": [
                    "Database connection timeout errors",
                    "Network connectivity troubleshooting",
                    "Application error logging setup",
                    "System monitoring and alerting"
                ],
                "expected_relationships": ["error", "troubleshooting", "monitoring", "logging"]
            }
        ]
        
    async def run_check(self) -> CheckResult:
        """
        Execute comprehensive graph intent validation.

        DEPRECATED (Phase 2 Optimization):
        This check has been consolidated into S2-golden-fact-recall for efficiency.
        S2 now includes core graph relationship validation.

        Comprehensive graph intent testing should be done in CI/CD pipeline,
        not runtime monitoring (reduces from 8 queries to 2 queries).
        """
        # OPTIMIZATION: Return early with deprecation notice
        # Core graph relationship testing now handled by enhanced S2 check
        return CheckResult(
            check_id=self.check_id,
            timestamp=datetime.utcnow(),
            status="pass",
            latency_ms=0.0,
            message="DEPRECATED: Graph intent validation consolidated into S2 (Phase 2 optimization)",
            details={
                "deprecated": True,
                "deprecated_since": "2025-11-17",
                "removal_planned": "2025-12-17",
                "reason": "Consolidated into S2-golden-fact-recall for query efficiency",
                "optimization": "S9 queries reduced from 8 to 0 (functionality moved to S2 which uses 6 queries for graph validation)",
                "recommendation": "Run comprehensive graph intent tests in CI/CD pipeline",
                "consolidated_into": "S2-golden-fact-recall",
                "migration_status": "S2 now includes 3 graph relationship test cases (6 queries total)",
                "phase": "2"
            }
        )
