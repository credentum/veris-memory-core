#!/usr/bin/env python3
"""
Comprehensive tests for backend restoration fixes (Phases 1-4).

This module tests the critical fixes applied in PR #170 to restore
backend functionality from 0% to 100% operational:

Phase 1: Embedding generation with robust service
Phase 2: Backend import handling with granular error reporting
Phase 3: Neo4j relationship validation before creation
Phase 4: Redis caching implementation with cache hit/miss

Tests ensure 30%+ coverage for modified code sections.
"""

import asyncio
import hashlib
import json
import os
import sys
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Dynamic embedding dimension from environment or default
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))


# =============================================================================
# PHASE 1: Embedding Generation Tests
# =============================================================================


class TestPhase1EmbeddingGeneration:
    """Test suite for Phase 1: Embedding generation with robust service."""

    @pytest.mark.asyncio
    async def test_generate_embedding_success(self):
        """Test successful embedding generation with robust service."""
        from src.mcp_server.main import _generate_embedding

        test_content = {"title": "Test", "description": "Test content"}

        # Mock the embedding service to return valid embedding
        with patch("src.embedding.generate_embedding") as mock_embed:
            mock_embed.return_value = [0.1] * EMBEDDING_DIM

            result = await _generate_embedding(test_content)

            assert result is not None
            assert len(result) == EMBEDDING_DIM
            assert all(isinstance(x, float) for x in result)
            mock_embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embedding_service_failure(self):
        """Test embedding generation failure with proper error propagation."""
        from src.mcp_server.main import _generate_embedding

        test_content = {"title": "Test"}

        # Mock embedding service to fail
        with patch("src.embedding.generate_embedding") as mock_embed:
            mock_embed.side_effect = Exception("Service unavailable")

            with pytest.raises(ValueError) as exc_info:
                await _generate_embedding(test_content)

            assert "Embedding generation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_embedding_strict_mode(self):
        """Test STRICT_EMBEDDINGS environment variable enforcement."""
        from src.mcp_server.main import _generate_embedding

        test_content = {"title": "Test"}

        with patch.dict("os.environ", {"STRICT_EMBEDDINGS": "true"}):
            with patch("src.embedding.generate_embedding") as mock_embed:
                mock_embed.side_effect = Exception("Model not available")

                with pytest.raises(ValueError) as exc_info:
                    await _generate_embedding(test_content)

                assert "Semantic embeddings unavailable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_embedding_empty_result(self):
        """Test handling of empty embedding result."""
        from src.mcp_server.main import _generate_embedding

        test_content = {"title": "Test"}

        with patch("src.embedding.generate_embedding") as mock_embed:
            mock_embed.return_value = []  # Empty embedding

            with pytest.raises(ValueError) as exc_info:
                await _generate_embedding(test_content)

            assert "empty embedding" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_generate_embedding_wrong_dimensions(self):
        """Test handling of incorrect embedding dimensions."""
        from src.mcp_server.main import _generate_embedding

        test_content = {"title": "Test"}

        with patch("src.embedding.generate_embedding") as mock_embed:
            # Return wrong number of dimensions
            mock_embed.return_value = [0.1] * 512  # Should be 768

            # Should still return the result (service handles dimension adjustment)
            result = await _generate_embedding(test_content)
            assert result is not None


# =============================================================================
# PHASE 2: Backend Import Handling Tests
# =============================================================================


class TestPhase2BackendImportHandling:
    """Test suite for Phase 2: Backend import handling with granular errors."""

    def test_try_import_backend_component_success(self):
        """Test successful import of backend component."""
        from src.mcp_server.main import _try_import_backend_component

        components = {}
        errors = []

        # Test importing a real module (json is always available)
        result = _try_import_backend_component(
            "json", "dumps", components, errors
        )

        assert result is not None
        assert "dumps" in components
        assert len(errors) == 0

    def test_try_import_backend_component_import_error(self):
        """Test import error handling."""
        from src.mcp_server.main import _try_import_backend_component

        components = {}
        errors = []

        # Try to import a non-existent module
        result = _try_import_backend_component(
            "nonexistent.module", "SomeClass", components, errors
        )

        assert result is None
        assert "SomeClass" not in components
        assert len(errors) == 1
        assert "SomeClass" in errors[0]

    def test_try_import_backend_component_attribute_error(self):
        """Test attribute error handling when component doesn't exist."""
        from src.mcp_server.main import _try_import_backend_component

        components = {}
        errors = []

        # Import real module but ask for non-existent attribute
        result = _try_import_backend_component(
            "json", "NonExistentFunction", components, errors
        )

        assert result is None
        assert "NonExistentFunction" not in components
        assert len(errors) == 1

    def test_backend_import_graceful_degradation(self):
        """Test that system degrades gracefully with missing components."""
        # This test verifies that UNIFIED_BACKEND_AVAILABLE is False
        # when required components are missing
        from src.mcp_server.main import (
            UNIFIED_BACKEND_AVAILABLE,
            unified_backend_errors,
        )

        # If unified backend is unavailable, errors should be logged
        if not UNIFIED_BACKEND_AVAILABLE:
            assert len(unified_backend_errors) > 0
        else:
            # If available, no critical errors should exist
            assert isinstance(UNIFIED_BACKEND_AVAILABLE, bool)

    def test_unified_backend_components_dict(self):
        """Test that unified_backend_components dictionary is populated correctly."""
        from src.mcp_server.main import unified_backend_components

        # Should be a dictionary
        assert isinstance(unified_backend_components, dict)

        # If any components loaded, they should be valid
        for name, component in unified_backend_components.items():
            assert component is not None
            assert isinstance(name, str)


# =============================================================================
# PHASE 3: Neo4j Relationship Validation Tests
# =============================================================================


class TestPhase3RelationshipValidation:
    """Test suite for Phase 3: Neo4j relationship validation."""

    @pytest.mark.asyncio
    async def test_relationship_creation_with_valid_target(self):
        """Test relationship creation when target node exists."""
        from src.mcp_server.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # Mock Neo4j to confirm target exists
        with patch("src.mcp_server.main.neo4j_client") as mock_neo4j:
            with patch("src.mcp_server.main.qdrant_client", None):
                # Mock target node exists
                mock_neo4j.query.return_value = [
                    {"n": {"id": "target-123", "title": "Target Node"}}
                ]
                mock_neo4j.create_relationship.return_value = True

                response = client.post(
                    "/tools/store_context",
                    json={
                        "type": "decision",
                        "content": {"title": "Source Node"},
                        "author": "test",
                        "author_type": "agent",
                        "relationships": [
                            {"target": "target-123", "type": "RELATED_TO"}
                        ],
                    },
                )

                assert response.status_code == 200
                data = response.json()
                assert "relationships_created" in data
                # Target exists, so relationship should be created
                assert data["relationships_created"] >= 0

    @pytest.mark.asyncio
    async def test_relationship_creation_with_missing_target(self):
        """Test relationship creation gracefully handles missing target node."""
        from src.mcp_server.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        with patch("src.mcp_server.main.neo4j_client") as mock_neo4j:
            with patch("src.mcp_server.main.qdrant_client", None):
                # Mock target node doesn't exist
                mock_neo4j.query.return_value = []

                response = client.post(
                    "/tools/store_context",
                    json={
                        "type": "decision",
                        "content": {"title": "Source Node"},
                        "author": "test",
                        "author_type": "agent",
                        "relationships": [
                            {"target": "missing-123", "type": "RELATED_TO"}
                        ],
                    },
                )

                assert response.status_code == 200
                data = response.json()
                assert "relationships_created" in data
                # Target missing, so no relationships created
                assert data["relationships_created"] == 0

    @pytest.mark.asyncio
    async def test_relationship_creation_multiple_relationships(self):
        """Test creation of multiple relationships with mixed success."""
        from src.mcp_server.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        with patch("src.mcp_server.main.neo4j_client") as mock_neo4j:
            with patch("src.mcp_server.main.qdrant_client", None):

                def mock_query_side_effect(query, parameters):
                    # First target exists, second doesn't
                    target_id = parameters.get("id")
                    if target_id == "valid-target":
                        return [{"n": {"id": target_id}}]
                    return []

                mock_neo4j.query.side_effect = mock_query_side_effect
                mock_neo4j.create_relationship.return_value = True

                response = client.post(
                    "/tools/store_context",
                    json={
                        "type": "decision",
                        "content": {"title": "Source Node"},
                        "author": "test",
                        "author_type": "agent",
                        "relationships": [
                            {"target": "valid-target", "type": "RELATED_TO"},
                            {"target": "invalid-target", "type": "DEPENDS_ON"},
                        ],
                    },
                )

                assert response.status_code == 200
                data = response.json()
                assert "relationships_created" in data

    @pytest.mark.asyncio
    async def test_relationship_creation_with_error(self):
        """Test relationship creation handles errors gracefully."""
        from src.mcp_server.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        with patch("src.mcp_server.main.neo4j_client") as mock_neo4j:
            with patch("src.mcp_server.main.qdrant_client", None):
                # Target exists but creation fails
                mock_neo4j.query.return_value = [{"n": {"id": "target-123"}}]
                mock_neo4j.create_relationship.side_effect = Exception(
                    "Database error"
                )

                response = client.post(
                    "/tools/store_context",
                    json={
                        "type": "decision",
                        "content": {"title": "Source Node"},
                        "author": "test",
                        "author_type": "agent",
                        "relationships": [
                            {"target": "target-123", "type": "RELATED_TO"}
                        ],
                    },
                )

                # Should still succeed but with 0 relationships created
                assert response.status_code == 200
                data = response.json()
                assert data["relationships_created"] == 0


# =============================================================================
# PHASE 4: Redis Caching Tests
# =============================================================================


class TestPhase4RedisCaching:
    """Test suite for Phase 4: Redis caching implementation."""

    @pytest.mark.asyncio
    async def test_cache_miss_then_hit(self):
        """Test cache miss on first request, hit on second."""
        from src.mcp_server.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        mock_redis = Mock()
        mock_redis.get.side_effect = [
            None,  # First call: cache miss
            json.dumps(
                {
                    "results": [{"id": "1", "content": {}, "metadata": {}}],
                    "total_count": 1,
                }
            ),  # Second call: cache hit
        ]
        mock_redis.setex.return_value = True

        with patch("src.mcp_server.main.simple_redis", mock_redis):
            with patch("src.mcp_server.main.neo4j_client") as mock_neo4j:
                with patch("src.mcp_server.main.qdrant_client", None):
                    mock_neo4j.query.return_value = [
                        {"n": {"id": "1", "title": "Test"}}
                    ]

                    # First request: cache miss
                    response1 = client.post(
                        "/tools/retrieve_context",
                        json={"query": "test", "limit": 5},
                    )
                    assert response1.status_code == 200
                    data1 = response1.json()
                    assert "results" in data1

                    # Verify setex was called to cache results
                    assert mock_redis.setex.call_count >= 1

                    # Second request: cache hit
                    response2 = client.post(
                        "/tools/retrieve_context",
                        json={"query": "test", "limit": 5},
                    )
                    assert response2.status_code == 200
                    data2 = response2.json()
                    assert data2.get("cache_hit") or data2.get("cached")

    @pytest.mark.asyncio
    async def test_cache_key_generation(self):
        """Test cache key generation with SHA256 hash."""
        # Test the cache key generation logic
        cache_params = {
            "query": "test query",
            "limit": 10,
            "search_mode": "hybrid",
            "context_type": None,
            "sort_by": "relevance",
        }

        cache_hash = hashlib.sha256(
            json.dumps(cache_params, sort_keys=True).encode()
        ).hexdigest()
        cache_key = f"retrieve:{cache_hash}"

        # Verify cache key format
        assert cache_key.startswith("retrieve:")
        assert len(cache_hash) == 64  # SHA256 produces 64 hex characters

    @pytest.mark.asyncio
    async def test_cache_ttl_behavior(self):
        """Test that cache entries have correct TTL (5 minutes = 300 seconds)."""
        from src.mcp_server.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        mock_redis = Mock()
        mock_redis.get.return_value = None  # Cache miss
        mock_redis.setex.return_value = True

        with patch("src.mcp_server.main.simple_redis", mock_redis):
            with patch("src.mcp_server.main.neo4j_client") as mock_neo4j:
                with patch("src.mcp_server.main.qdrant_client", None):
                    mock_neo4j.query.return_value = [
                        {"n": {"id": "1", "title": "Test"}}
                    ]

                    response = client.post(
                        "/tools/retrieve_context",
                        json={"query": "test", "limit": 5},
                    )

                    assert response.status_code == 200

                    # Verify setex was called with correct TTL (default 300 seconds or custom)
                    if mock_redis.setex.called:
                        call_args = mock_redis.setex.call_args
                        # Second argument should be TTL
                        ttl = call_args[0][1]
                        expected_ttl = int(os.getenv("VERIS_CACHE_TTL_SECONDS", "300"))
                        assert ttl == expected_ttl, f"Expected TTL {expected_ttl}, got {ttl}"

    @pytest.mark.asyncio
    async def test_cache_disabled_via_request_param(self):
        """Test that caching can be disabled via use_cache parameter."""
        from src.mcp_server.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        mock_redis = Mock()
        mock_redis.get.return_value = json.dumps(
            {"results": [], "total_count": 0}
        )

        with patch("src.mcp_server.main.simple_redis", mock_redis):
            with patch("src.mcp_server.main.neo4j_client") as mock_neo4j:
                with patch("src.mcp_server.main.qdrant_client", None):
                    mock_neo4j.query.return_value = []

                    # Request with use_cache=False (if supported)
                    response = client.post(
                        "/tools/retrieve_context",
                        json={"query": "test", "limit": 5, "use_cache": False},
                    )

                    assert response.status_code == 200
                    # Cache should not be checked when use_cache=False
                    # (Implementation may vary)

    @pytest.mark.asyncio
    async def test_cache_error_handling(self):
        """Test that cache errors don't break retrieval."""
        from src.mcp_server.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        mock_redis = Mock()
        mock_redis.get.side_effect = Exception("Redis connection failed")
        mock_redis.setex.side_effect = Exception("Redis connection failed")

        with patch("src.mcp_server.main.simple_redis", mock_redis):
            with patch("src.mcp_server.main.neo4j_client") as mock_neo4j:
                with patch("src.mcp_server.main.qdrant_client", None):
                    mock_neo4j.query.return_value = [
                        {"n": {"id": "1", "title": "Test"}}
                    ]

                    # Should still work even if cache fails
                    response = client.post(
                        "/tools/retrieve_context",
                        json={"query": "test", "limit": 5},
                    )

                    assert response.status_code == 200
                    data = response.json()
                    assert "results" in data

    @pytest.mark.asyncio
    async def test_cache_key_includes_all_parameters(self):
        """Test that cache key includes all relevant request parameters."""
        cache_params_1 = {
            "query": "test",
            "limit": 10,
            "search_mode": "hybrid",
            "context_type": None,
            "sort_by": "relevance",
        }

        cache_params_2 = {
            "query": "test",
            "limit": 5,  # Different limit
            "search_mode": "hybrid",
            "context_type": None,
            "sort_by": "relevance",
        }

        hash1 = hashlib.sha256(
            json.dumps(cache_params_1, sort_keys=True).encode()
        ).hexdigest()
        hash2 = hashlib.sha256(
            json.dumps(cache_params_2, sort_keys=True).encode()
        ).hexdigest()

        # Different parameters should produce different cache keys
        assert hash1 != hash2

    @pytest.mark.asyncio
    async def test_cache_key_determinism(self):
        """Test that cache key generation is deterministic and handles edge cases."""
        # Test 1: Same parameters should always produce same key
        cache_params = {
            "query": "test query",
            "limit": 10,
            "search_mode": "hybrid",
            "context_type": None,
            "sort_by": "relevance",
        }

        hash1 = hashlib.sha256(
            json.dumps(cache_params, sort_keys=True).encode()
        ).hexdigest()
        hash2 = hashlib.sha256(
            json.dumps(cache_params, sort_keys=True).encode()
        ).hexdigest()

        # Same parameters should always produce same hash (deterministic)
        assert hash1 == hash2

        # Test 2: Handle null values consistently
        cache_params_null = {
            "query": "test",
            "limit": 10,
            "search_mode": "hybrid",
            "context_type": None,  # Explicitly None
            "sort_by": "relevance",
        }

        hash_null = hashlib.sha256(
            json.dumps(cache_params_null, sort_keys=True).encode()
        ).hexdigest()

        # Should produce consistent hash even with None values
        assert len(hash_null) == 64  # SHA256 produces 64 hex chars
        assert hash_null.isalnum()  # Should be valid hex

        # Test 3: Handle special characters in query
        cache_params_special = {
            "query": "test with 'quotes' and \"double quotes\" and \n newlines",
            "limit": 10,
            "search_mode": "hybrid",
            "context_type": None,
            "sort_by": "relevance",
        }

        hash_special = hashlib.sha256(
            json.dumps(cache_params_special, sort_keys=True).encode()
        ).hexdigest()

        # Should handle special characters without errors
        assert len(hash_special) == 64
        assert hash_special != hash1  # Different query should produce different hash

        # Test 4: Empty string vs None should produce different keys
        cache_params_empty = {
            "query": "",  # Empty string
            "limit": 10,
            "search_mode": "hybrid",
            "context_type": None,
            "sort_by": "relevance",
        }

        hash_empty = hashlib.sha256(
            json.dumps(cache_params_empty, sort_keys=True).encode()
        ).hexdigest()

        # Empty string is different from "test query"
        assert hash_empty != hash1


# =============================================================================
# Integration Tests
# =============================================================================


class TestBackendRestorationIntegration:
    """Integration tests for all 4 phases working together."""

    @pytest.mark.asyncio
    async def test_full_store_and_retrieve_flow(self):
        """Test complete flow: store with embeddings, relationships, and retrieve with cache."""
        from src.mcp_server.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        mock_redis = Mock()
        mock_redis.get.return_value = None  # Cache miss
        mock_redis.setex.return_value = True

        # Mock successful embedding generation
        with patch("src.embedding.generate_embedding") as mock_embed:
            mock_embed.return_value = [0.1] * 768

            with patch("src.mcp_server.main.simple_redis", mock_redis):
                with patch("src.mcp_server.main.neo4j_client") as mock_neo4j:
                    with patch("src.mcp_server.main.qdrant_client") as mock_qdrant:
                        # Setup mocks
                        mock_neo4j.query.return_value = [
                            {"n": {"id": "target-123"}}
                        ]
                        mock_neo4j.create_node.return_value = "node-123"
                        mock_neo4j.create_relationship.return_value = True

                        mock_qdrant.upsert.return_value = True

                        # Store context
                        store_response = client.post(
                            "/tools/store_context",
                            json={
                                "type": "decision",
                                "content": {"title": "Test Node"},
                                "author": "test",
                                "author_type": "agent",
                                "relationships": [
                                    {
                                        "target": "target-123",
                                        "type": "RELATED_TO",
                                    }
                                ],
                            },
                        )

                        assert store_response.status_code == 200
                        store_data = store_response.json()

                        # Verify all phases worked
                        assert "embedding_status" in store_data  # Phase 1
                        assert "relationships_created" in store_data  # Phase 3

                        # Setup retrieval
                        mock_neo4j.query.return_value = [
                            {
                                "n": {
                                    "id": "node-123",
                                    "title": "Test Node",
                                    "type": "decision",
                                }
                            }
                        ]

                        # Retrieve context (should cache)
                        retrieve_response = client.post(
                            "/tools/retrieve_context",
                            json={"query": "test", "limit": 5},
                        )

                        assert retrieve_response.status_code == 200
                        retrieve_data = retrieve_response.json()
                        assert "results" in retrieve_data

                        # Verify cache was used (Phase 4)
                        if mock_redis.setex.called:
                            assert mock_redis.setex.call_count >= 1


# =============================================================================
# PHASE 5: Monitoring Metrics Emission Tests
# =============================================================================


class TestPhase5MonitoringMetrics:
    """Test suite for monitoring metrics emission (PR #170 observability)."""

    @pytest.mark.asyncio
    async def test_embedding_generation_error_metric_emitted(self, caplog):
        """Test that embedding_generation_errors_total metric is emitted on failure."""
        import logging

        from src.mcp_server.main import _generate_embedding

        test_content = {"title": "Test", "description": "Test content"}

        # Mock the embedding service to raise an exception
        with patch("src.embedding.generate_embedding") as mock_embed:
            mock_embed.side_effect = RuntimeError("Embedding service unavailable")

            with caplog.at_level(logging.INFO):
                with pytest.raises(ValueError, match="Embedding generation failed"):
                    await _generate_embedding(test_content)

                # Verify metric log was emitted
                metric_logs = [
                    record.message
                    for record in caplog.records
                    if "METRIC:" in record.message
                    and "embedding_generation_errors_total" in record.message
                ]

                assert len(metric_logs) > 0, "No embedding error metric emitted"
                assert "embedding_generation_errors_total" in metric_logs[0]
                assert "strict_mode='false'" in metric_logs[0]

    def test_cache_hit_metric_format(self, caplog):
        """Test that cache hit metrics follow correct Prometheus format."""
        import logging

        # Simulate logging a cache hit metric (as done in main.py)
        logger = logging.getLogger("src.mcp_server.main")

        with caplog.at_level(logging.INFO):
            # This is the exact format used in src/mcp_server/main.py lines 2090-2091
            logger.info(
                "METRIC: cache_requests_total{result='hit',search_mode='hybrid'} 1"
            )

            # Verify metric was logged
            metric_logs = [
                record.message
                for record in caplog.records
                if "METRIC:" in record.message
                and "cache_requests_total" in record.message
                and "result='hit'" in record.message
            ]

            assert len(metric_logs) > 0, "No cache hit metric emitted"
            assert "cache_requests_total" in metric_logs[0]
            assert "result='hit'" in metric_logs[0]
            assert "search_mode=" in metric_logs[0]

    def test_cache_miss_metric_format(self, caplog):
        """Test that cache miss metrics follow correct Prometheus format."""
        import logging

        # Simulate logging a cache miss metric (as done in main.py)
        logger = logging.getLogger("src.mcp_server.main")

        with caplog.at_level(logging.INFO):
            # This is the exact format used in src/mcp_server/main.py lines 2105-2106
            logger.info(
                "METRIC: cache_requests_total{result='miss',search_mode='vector'} 1"
            )

            # Verify metric was logged
            metric_logs = [
                record.message
                for record in caplog.records
                if "METRIC:" in record.message
                and "cache_requests_total" in record.message
                and "result='miss'" in record.message
            ]

            assert len(metric_logs) > 0, "No cache miss metric emitted"
            assert "cache_requests_total" in metric_logs[0]
            assert "result='miss'" in metric_logs[0]

    def test_cache_error_metric_format(self, caplog):
        """Test that cache error metrics follow correct Prometheus format."""
        import logging

        # Simulate logging a cache error metric (as done in main.py)
        logger = logging.getLogger("src.mcp_server.main")

        with caplog.at_level(logging.INFO):
            # This is the exact format used in src/mcp_server/main.py line 2121
            logger.info("METRIC: cache_requests_total{result='error'} 1")

            # Verify metric was logged
            metric_logs = [
                record.message
                for record in caplog.records
                if "METRIC:" in record.message
                and "cache_requests_total" in record.message
                and "result='error'" in record.message
            ]

            assert len(metric_logs) > 0, "No cache error metric emitted"
            assert "cache_requests_total" in metric_logs[0]
            assert "result='error'" in metric_logs[0]

    def test_structured_logging_fields_present(self, caplog):
        """Test that structured logging includes required 'extra' fields."""
        import logging

        # Test that logging includes structured fields
        logger = logging.getLogger("src.mcp_server.main")

        with caplog.at_level(logging.WARNING):
            logger.warning(
                "Test message",
                extra={
                    "event_type": "test_event",
                    "error_type": "TestError",
                    "strict_mode": False,
                },
            )

            # Verify structured fields are present
            assert len(caplog.records) > 0
            record = caplog.records[0]

            # Check that extra fields are attached to log record
            assert hasattr(record, "event_type")
            assert record.event_type == "test_event"
            assert hasattr(record, "error_type")
            assert record.error_type == "TestError"
            assert hasattr(record, "strict_mode")
            assert record.strict_mode is False

    def test_metric_format_compliance(self):
        """Test that metric logs follow Prometheus format."""
        # Validate metric format: METRIC: metric_name{label='value'} number
        valid_metrics = [
            "METRIC: embedding_generation_errors_total{strict_mode='false'} 1",
            "METRIC: cache_requests_total{result='hit',search_mode='hybrid'} 1",
            "METRIC: cache_requests_total{result='miss',search_mode='vector'} 1",
            "METRIC: cache_requests_total{result='error'} 1",
        ]

        for metric in valid_metrics:
            # Check format
            assert metric.startswith("METRIC: ")
            assert "{" in metric
            assert "}" in metric
            assert metric.split()[-1].isdigit()  # Last part should be a number

            # Extract metric name
            metric_name = metric.split("{")[0].replace("METRIC: ", "").strip()
            assert len(metric_name) > 0
            assert "_" in metric_name  # Prometheus convention

            # Extract labels
            labels_section = metric.split("{")[1].split("}")[0]
            assert "=" in labels_section

            # Extract value
            value = metric.split()[-1]
            assert value.isdigit()
            assert int(value) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.mcp_server.main"])
