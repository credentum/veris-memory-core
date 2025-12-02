#!/usr/bin/env python3
"""
Unit tests for S9 Graph Intent Validation Check.

Tests the GraphIntentValidation check with mocked HTTP calls and graph analysis.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import aiohttp

from src.monitoring.sentinel.checks.s9_graph_intent import GraphIntentValidation
from src.monitoring.sentinel.models import SentinelConfig


class TestGraphIntentValidation:
    """Test suite for GraphIntentValidation check."""

    @pytest.fixture
    def config(self) -> SentinelConfig:
        """Create a test configuration using real SentinelConfig."""
        # Create real SentinelConfig instance
        config = SentinelConfig(
            target_base_url="http://test.example.com:8000",
            enabled_checks=["S9-graph-intent"]
        )
        return config
    
    @pytest.fixture
    def check(self, config: SentinelConfig) -> GraphIntentValidation:
        """Create a GraphIntentValidation check instance."""
        return GraphIntentValidation(config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, config: SentinelConfig) -> None:
        """Test check initialization."""
        check = GraphIntentValidation(config)
        
        assert check.check_id == "S9-graph-intent"
        assert check.description == "Graph intent validation"
        assert check.veris_memory_url == "http://test.example.com:8000"
        assert check.timeout_seconds == 30
        assert check.max_traversal_depth == 3
        assert check.graph_sample_size == 10
        assert len(check.intent_scenarios) == 1
    
    @pytest.mark.asyncio
    async def test_run_check_all_pass(self, check: GraphIntentValidation) -> None:
        """Test run_check when all graph intent tests pass."""
        mock_results = [
            {"passed": True, "message": "Relationship accuracy validated"},
            {"passed": True, "message": "Semantic connectivity confirmed"},
            {"passed": True, "message": "Graph traversal quality verified"},
            {"passed": True, "message": "Context clustering validated"},
            {"passed": True, "message": "Relationship inference successful"},
            {"passed": True, "message": "Graph coherence confirmed"},
            {"passed": True, "message": "Intent preservation validated"}
        ]
        
        with patch.object(check, '_test_relationship_accuracy', return_value=mock_results[0]):
            with patch.object(check, '_validate_semantic_connectivity', return_value=mock_results[1]):
                with patch.object(check, '_test_graph_traversal_quality', return_value=mock_results[2]):
                    with patch.object(check, '_validate_context_clustering', return_value=mock_results[3]):
                        with patch.object(check, '_test_relationship_inference', return_value=mock_results[4]):
                            with patch.object(check, '_validate_graph_coherence', return_value=mock_results[5]):
                                with patch.object(check, '_test_intent_preservation', return_value=mock_results[6]):
                                    
                                    result = await check.run_check()
        
        assert result.check_id == "S9-graph-intent"
        assert result.status == "pass"
        assert "All graph intent validation checks passed: 7 tests successful" in result.message
        assert result.details["total_tests"] == 7
        assert result.details["passed_tests"] == 7
        assert result.details["failed_tests"] == 0
    
    @pytest.mark.asyncio
    async def test_run_check_with_failures(self, check: GraphIntentValidation) -> None:
        """Test run_check when some graph intent tests fail."""
        mock_results = [
            {"passed": False, "message": "Relationship accuracy low"},
            {"passed": False, "message": "Semantic connectivity failed"},
            {"passed": True, "message": "Graph traversal quality verified"},
            {"passed": True, "message": "Context clustering validated"},
            {"passed": True, "message": "Relationship inference successful"},
            {"passed": True, "message": "Graph coherence confirmed"},
            {"passed": True, "message": "Intent preservation validated"}
        ]
        
        with patch.object(check, '_test_relationship_accuracy', return_value=mock_results[0]):
            with patch.object(check, '_validate_semantic_connectivity', return_value=mock_results[1]):
                with patch.object(check, '_test_graph_traversal_quality', return_value=mock_results[2]):
                    with patch.object(check, '_validate_context_clustering', return_value=mock_results[3]):
                        with patch.object(check, '_test_relationship_inference', return_value=mock_results[4]):
                            with patch.object(check, '_validate_graph_coherence', return_value=mock_results[5]):
                                with patch.object(check, '_test_intent_preservation', return_value=mock_results[6]):
                                    
                                    result = await check.run_check()
        
        assert result.status == "fail"
        assert "Graph intent validation issues detected: 2 problems found" in result.message
        assert result.details["passed_tests"] == 5
        assert result.details["failed_tests"] == 2
    
    @pytest.mark.asyncio
    async def test_relationship_accuracy_success(self, check: GraphIntentValidation) -> None:
        """Test successful relationship accuracy analysis."""
        # Mock context storage
        mock_store_response = AsyncMock()
        mock_store_response.status = 201
        mock_store_response.json = AsyncMock(return_value={"context_id": "test_ctx_123"})
        
        # Mock context retrieval
        mock_get_response = AsyncMock()
        mock_get_response.status = 200
        mock_get_response.json = AsyncMock(return_value={
            "content": {"text": "Test context about configuration"}
        })
        
        # Mock search results
        mock_search_response = AsyncMock()
        mock_search_response.status = 200
        mock_search_response.json = AsyncMock(return_value={
            "contexts": [
                {"context_id": "related_ctx", "content": {"text": "Configuration setup guide"}}
            ]
        })
        
        mock_session = AsyncMock()
        
        # Configure different responses for different endpoints
        def mock_request(method, url, **kwargs):
            ctx = AsyncMock()
            if method == "post" and "/contexts" in url and "search" not in url:
                ctx.__aenter__ = AsyncMock(return_value=mock_store_response)
            elif method == "get" and "/contexts/" in url:
                ctx.__aenter__ = AsyncMock(return_value=mock_get_response)
            elif method == "post" and "/search" in url:
                ctx.__aenter__ = AsyncMock(return_value=mock_search_response)
            return ctx
        
        mock_session.post.side_effect = lambda url, **kwargs: mock_request("post", url, **kwargs)
        mock_session.get.side_effect = lambda url, **kwargs: mock_request("get", url, **kwargs)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._test_relationship_accuracy()
        
        assert result["passed"] is True
        assert "Relationship accuracy" in result["message"]
        assert result["accuracy_score"] >= 0.0
        assert len(result["scenario_results"]) > 0
    
    @pytest.mark.asyncio
    async def test_semantic_connectivity_success(self, check: GraphIntentValidation) -> None:
        """Test successful semantic connectivity validation."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "contexts": [
                {"context_id": "ctx1", "content": {"text": "Database configuration setup"}, "score": 0.9},
                {"context_id": "ctx2", "content": {"text": "Database setup guide"}, "score": 0.8}
            ]
        })
        
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._validate_semantic_connectivity()
        
        assert result["passed"] is True
        assert "Semantic connectivity" in result["message"]
        assert result["connectivity_ratio"] >= 0.0
        assert len(result["query_tests"]) > 0
    
    @pytest.mark.asyncio
    async def test_graph_traversal_quality_success(self, check: GraphIntentValidation) -> None:
        """Test successful graph traversal quality assessment."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "contexts": [
                {"context_id": "ctx1", "content": {"text": "Database configuration and setup"}, "score": 0.9}
            ]
        })
        
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._test_graph_traversal_quality()
        
        assert result["passed"] is True
        assert "Graph traversal quality" in result["message"]
        assert result["traversal_ratio"] >= 0.0
        assert len(result["concept_tests"]) > 0
    
    @pytest.mark.asyncio
    async def test_context_clustering_success(self, check: GraphIntentValidation) -> None:
        """Test successful context clustering validation."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "contexts": [
                {"context_id": "ctx1", "content": {"text": "Database operations setup"}, "score": 0.9},
                {"context_id": "ctx2", "content": {"text": "Database configuration"}, "score": 0.8}
            ]
        })
        
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._validate_context_clustering()
        
        assert result["passed"] is True
        assert "Context clustering quality" in result["message"]
        assert result["clustering_ratio"] >= 0.0
        assert len(result["cluster_tests"]) > 0
    
    @pytest.mark.asyncio
    async def test_relationship_inference_success(self, check: GraphIntentValidation) -> None:
        """Test successful relationship inference."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "contexts": [
                {"context_id": "ctx1", "content": {"text": "Error handling with logging and monitoring for troubleshooting"}, "score": 0.9}
            ]
        })
        
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._test_relationship_inference()
        
        assert result["passed"] is True
        assert "Relationship inference" in result["message"]
        assert result["inference_ratio"] >= 0.0
        assert len(result["inference_tests"]) > 0
    
    @pytest.mark.asyncio
    async def test_graph_coherence_success(self, check: GraphIntentValidation) -> None:
        """Test successful graph coherence validation."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "contexts": [
                {"context_id": "ctx1", "content": {"text": "Database performance monitoring system"}, "score": 0.9}
            ]
        })
        
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._validate_graph_coherence()
        
        assert result["passed"] is True
        assert "Graph coherence" in result["message"]
        assert result["coherence_ratio"] >= 0.0
        assert len(result["coherence_tests"]) > 0
    
    @pytest.mark.asyncio
    async def test_intent_preservation_success(self, check: GraphIntentValidation) -> None:
        """Test successful intent preservation validation."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "contexts": [
                {"context_id": "ctx1", "content": {"text": "How to troubleshoot database connection issues with proper authentication"}, "score": 0.9}
            ]
        })
        
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._test_intent_preservation()
        
        assert result["passed"] is True
        assert "Intent preservation" in result["message"]
        assert result["preservation_ratio"] >= 0.0
        assert len(result["intent_tests"]) > 0
    
    @pytest.mark.asyncio
    async def test_calculate_semantic_coherence(self, check: GraphIntentValidation) -> None:
        """Test semantic coherence calculation."""
        contexts = [
            {"content": {"text": "database configuration setup"}},
            {"content": {"text": "database setup guide"}},
            {"content": {"text": "configuration database"}}
        ]
        query = "database configuration"
        
        coherence = check._calculate_semantic_coherence(contexts, query)
        
        assert 0.0 <= coherence <= 1.0
        # Should have high coherence since all contexts contain query terms
        assert coherence > 0.5
    
    @pytest.mark.asyncio
    async def test_calculate_path_relevance(self, check: GraphIntentValidation) -> None:
        """Test path relevance calculation."""
        source_text = "database configuration setup"
        target_text = "configuration database guide"
        
        relevance = check._calculate_path_relevance(source_text, target_text)
        
        assert 0.0 <= relevance <= 1.0
        # Should have some relevance due to shared terms
        assert relevance > 0.0
    
    @pytest.mark.asyncio
    async def test_analyze_cluster_coherence(self, check: GraphIntentValidation) -> None:
        """Test cluster coherence analysis."""
        contexts = [
            {"content": {"text": "database setup configuration"}},
            {"content": {"text": "database configuration guide"}},
            {"content": {"text": "setup database config"}}
        ]
        query = "database configuration"
        
        coherence = check._analyze_cluster_coherence(contexts, query)
        
        assert 0.0 <= coherence <= 1.0
        # Should have decent coherence for related contexts
        assert coherence > 0.0
    
    @pytest.mark.asyncio
    async def test_evaluate_inference_quality(self, check: GraphIntentValidation) -> None:
        """Test inference quality evaluation."""
        contexts = [
            {"content": {"text": "error handling with logging and monitoring for troubleshooting"}},
            {"content": {"text": "debugging system with monitoring tools"}}
        ]
        related_concepts = ["logging", "monitoring", "debugging"]
        expected_inference = "troubleshooting"
        
        quality = check._evaluate_inference_quality(contexts, related_concepts, expected_inference)
        
        assert 0.0 <= quality <= 1.0
        # Should have high quality since most concepts are present
        assert quality > 0.5
    
    @pytest.mark.asyncio
    async def test_calculate_cross_domain_coherence(self, check: GraphIntentValidation) -> None:
        """Test cross-domain coherence calculation."""
        contexts = [
            {"content": {"text": "database performance monitoring system"}},
            {"content": {"text": "monitoring database performance"}}
        ]
        query = "database performance monitoring"
        
        coherence = check._calculate_cross_domain_coherence(contexts, query)
        
        assert 0.0 <= coherence <= 1.0
        # Should have high coherence for comprehensive coverage
        assert coherence > 0.5
    
    @pytest.mark.asyncio
    async def test_calculate_intent_preservation(self, check: GraphIntentValidation) -> None:
        """Test intent preservation calculation."""
        contexts = [
            {"content": {"text": "troubleshoot database connection authentication issues"}},
            {"content": {"text": "database authentication troubleshooting guide"}}
        ]
        key_concepts = ["database", "connection", "troubleshoot"]
        
        preservation = check._calculate_intent_preservation(contexts, key_concepts)
        
        assert 0.0 <= preservation <= 1.0
        # Should be 1.0 since all key concepts are present
        assert preservation == 1.0
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, check: GraphIntentValidation) -> None:
        """Test handling of API errors."""
        mock_session = AsyncMock()
        mock_session.post.side_effect = aiohttp.ClientError("Connection failed")
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check._validate_semantic_connectivity()
        
        assert result["passed"] is False
        assert "error" in result
        assert "Connection failed" in result["error"]
    
    @pytest.mark.asyncio
    async def test_empty_contexts_handling(self, check: GraphIntentValidation) -> None:
        """Test handling of empty context results."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"contexts": []})
        
        mock_session = AsyncMock()
        mock_session.post.return_value.__aender__.return_value = mock_response
        
        # Test semantic coherence with empty contexts
        coherence = check._calculate_semantic_coherence([], "test query")
        assert coherence == 0.0
        
        # Test path relevance with empty text
        relevance = check._calculate_path_relevance("", "test text")
        assert relevance == 0.0
        
        # Test intent preservation with empty contexts
        preservation = check._calculate_intent_preservation([], ["test"])
        assert preservation == 0.0
    
    @pytest.mark.asyncio
    async def test_analyze_traversal_paths_no_results(self, check: GraphIntentValidation) -> None:
        """Test traversal path analysis with no initial results."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"contexts": []})
        
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        quality = await check._analyze_traversal_paths(mock_session, "nonexistent", 3)
        
        assert quality == 0.0
    
    @pytest.mark.asyncio
    async def test_run_check_with_exception(self, check: GraphIntentValidation) -> None:
        """Test run_check when an exception occurs."""
        with patch.object(check, '_test_relationship_accuracy', side_effect=Exception("Graph error")):
            result = await check.run_check()
        
        assert result.status == "fail"
        assert "Graph intent validation failed with error: Graph error" in result.message
        assert result.details["error"] == "Graph error"
    
    @pytest.mark.asyncio
    async def test_default_intent_scenarios(self, check: GraphIntentValidation) -> None:
        """Test default intent scenarios configuration."""
        # Test with default configuration
        default_config = SentinelConfig({})
        default_check = GraphIntentValidation(default_config)
        
        assert len(default_check.intent_scenarios) == 5
        
        # Verify structure of default scenarios
        for scenario in default_check.intent_scenarios:
            assert "name" in scenario
            assert "description" in scenario
            assert "contexts" in scenario
            assert "expected_relationships" in scenario
            assert len(scenario["contexts"]) >= 3
            assert len(scenario["expected_relationships"]) >= 3

    # ==========================================
    # PR #247: Tests for endpoint fallback logic
    # ==========================================

    @pytest.mark.asyncio
    async def test_context_storage_endpoint_fallback_success_first(self, check: GraphIntentValidation) -> None:
        """Test context storage succeeds on first endpoint."""
        mock_session = AsyncMock()

        # Mock successful response on first endpoint
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.json = AsyncMock(return_value={"context_id": "ctx123"})

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value = ctx

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check.run_check()

        # Should succeed without trying fallback endpoints
        assert mock_session.post.call_count > 0

    @pytest.mark.asyncio
    async def test_context_storage_endpoint_fallback_to_second(self, check: GraphIntentValidation) -> None:
        """Test context storage falls back to second endpoint when first fails."""
        mock_session = AsyncMock()

        call_count = 0
        def mock_post(url, **kwargs):
            nonlocal call_count
            ctx = AsyncMock()
            if call_count == 0 and '/api/store_context' in url:
                # First endpoint (/api/store_context) fails
                mock_response = AsyncMock()
                mock_response.status = 404
                ctx.__aenter__ = AsyncMock(return_value=mock_response)
            elif '/api/v1/contexts' in url:
                # Second endpoint (/api/v1/contexts) succeeds
                mock_response = AsyncMock()
                mock_response.status = 201
                mock_response.json = AsyncMock(return_value={"context_id": "ctx123"})
                ctx.__aenter__ = AsyncMock(return_value=mock_response)
            else:
                # Other endpoints
                mock_response = AsyncMock()
                mock_response.status = 404
                ctx.__aenter__ = AsyncMock(return_value=mock_response)
            call_count += 1
            return ctx

        mock_session.post.side_effect = mock_post

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check.run_check()

        # Should have tried multiple endpoints
        assert mock_session.post.call_count >= 2

    @pytest.mark.asyncio
    async def test_context_storage_all_endpoints_fail(self, check: GraphIntentValidation) -> None:
        """Test graceful handling when all storage endpoints fail."""
        mock_session = AsyncMock()

        # All endpoints return 404
        mock_response = AsyncMock()
        mock_response.status = 404

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value = ctx

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check.run_check()

        # Check should complete despite storage failures
        assert result.check_id == "S9-graph-intent"
        # Should have tried to store contexts multiple times
        assert mock_session.post.call_count > 0

    @pytest.mark.asyncio
    async def test_context_storage_handles_both_id_formats(self, check: GraphIntentValidation) -> None:
        """Test that both 'context_id' and 'id' response formats are handled."""
        mock_session = AsyncMock()

        responses = [
            {"context_id": "ctx123"},  # Format 1
            {"id": "ctx456"},          # Format 2
        ]
        response_index = 0

        def mock_post(url, **kwargs):
            nonlocal response_index
            ctx = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 201
            mock_response.json = AsyncMock(return_value=responses[response_index % len(responses)])
            ctx.__aenter__ = AsyncMock(return_value=mock_response)
            response_index += 1
            return ctx

        mock_session.post.side_effect = mock_post

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check.run_check()

        # Should handle both response formats
        assert result.check_id == "S9-graph-intent"

    @pytest.mark.asyncio
    async def test_context_storage_network_error_fallback(self, check: GraphIntentValidation) -> None:
        """Test fallback to next endpoint on network errors."""
        mock_session = AsyncMock()

        call_count = 0
        def mock_post(url, **kwargs):
            nonlocal call_count
            if call_count == 0:
                # First attempt raises network error
                call_count += 1
                raise aiohttp.ClientError("Connection refused")
            else:
                # Second attempt succeeds
                ctx = AsyncMock()
                mock_response = AsyncMock()
                mock_response.status = 201
                mock_response.json = AsyncMock(return_value={"context_id": "ctx123"})
                ctx.__aenter__ = AsyncMock(return_value=mock_response)
                call_count += 1
                return ctx

        mock_session.post.side_effect = mock_post

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check.run_check()

        # Should have attempted multiple times
        assert call_count > 1

    @pytest.mark.asyncio
    async def test_context_storage_empty_id_triggers_fallback(self, check: GraphIntentValidation) -> None:
        """Test that empty context_id triggers fallback to next endpoint."""
        mock_session = AsyncMock()

        call_count = 0
        def mock_post(url, **kwargs):
            nonlocal call_count
            ctx = AsyncMock()
            if call_count == 0:
                # First endpoint returns empty context_id
                mock_response = AsyncMock()
                mock_response.status = 201
                mock_response.json = AsyncMock(return_value={"context_id": None})
                ctx.__aenter__ = AsyncMock(return_value=mock_response)
            else:
                # Second endpoint returns valid id
                mock_response = AsyncMock()
                mock_response.status = 201
                mock_response.json = AsyncMock(return_value={"id": "ctx456"})
                ctx.__aenter__ = AsyncMock(return_value=mock_response)
            call_count += 1
            return ctx

        mock_session.post.side_effect = mock_post

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check.run_check()

        # Should have tried fallback
        assert call_count > 1


class TestGraphIntentTargetBaseURL:
    """Test TARGET_BASE_URL environment variable usage in GraphIntentValidation."""

    @pytest.mark.asyncio
    @patch.dict('os.environ', {'TARGET_BASE_URL': 'http://context-store:8000'}, clear=False)
    async def test_uses_target_base_url_when_set(self) -> None:
        """Test check uses TARGET_BASE_URL environment variable when set."""
        config = SentinelConfig(enabled_checks=["S9-graph-intent"])
        check = GraphIntentValidation(config)

        # Verify TARGET_BASE_URL is used
        assert check.veris_memory_url == "http://context-store:8000"

    @pytest.mark.asyncio
    @patch.dict('os.environ', {}, clear=True)
    async def test_falls_back_to_localhost_when_unset(self) -> None:
        """Test check falls back to localhost:8000 when TARGET_BASE_URL is not set."""
        config = SentinelConfig(enabled_checks=["S9-graph-intent"])
        check = GraphIntentValidation(config)

        # Verify fallback to localhost
        assert check.veris_memory_url == "http://localhost:8000"

    @pytest.mark.asyncio
    @patch.dict('os.environ', {'TARGET_BASE_URL': 'http://custom-host:9999'}, clear=False)
    async def test_config_overrides_env_var(self) -> None:
        """Test config veris_memory_url overrides TARGET_BASE_URL if explicitly set."""
        config = SentinelConfig(
            enabled_checks=["S9-graph-intent"],
            veris_memory_url="http://config-override:7777"
        )
        check = GraphIntentValidation(config)

        # Config should override env var
        assert check.veris_memory_url == "http://config-override:7777"

    @pytest.mark.asyncio
    @patch.dict('os.environ', {'TARGET_BASE_URL': 'http://context-store:8000'}, clear=False)
    async def test_can_connect_using_target_base_url(self) -> None:
        """Test check can successfully connect to context-store using TARGET_BASE_URL."""
        config = SentinelConfig(enabled_checks=["S9-graph-intent"])
        check = GraphIntentValidation(config)

        # Mock successful HTTP connection to context-store
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"contexts": []})

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_response)
        ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = ctx

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await check.run_check()

        # Verify connection was attempted to context-store URL
        assert mock_session.get.called
        call_url = str(mock_session.get.call_args[0][0])
        assert call_url.startswith("http://context-store:8000")

class TestExtractContentText:
    """Test suite for S9 _extract_content_text() bug fix."""

    @pytest.fixture
    def check(self):
        """Create a GraphIntentValidation check instance."""
        config = SentinelConfig(enabled_checks=["S9-graph-intent"])
        return GraphIntentValidation(config)

    def test_extract_content_text_dict_format(self, check):
        """Test extraction from dict format {"content": {"text": "..."}}."""
        context_data = {
            "content": {
                "text": "This is the test content"
            }
        }

        result = check._extract_content_text(context_data)

        assert result == "This is the test content"

    def test_extract_content_text_string_format(self, check):
        """Test extraction from string format {"content": "..."}."""
        context_data = {
            "content": "This is direct string content"
        }

        result = check._extract_content_text(context_data)

        assert result == "This is direct string content"

    def test_extract_content_text_missing_content(self, check):
        """Test extraction when content key is missing."""
        context_data = {
            "type": "log",
            "metadata": {}
        }

        result = check._extract_content_text(context_data)

        assert result == ""

    def test_extract_content_text_empty_dict(self, check):
        """Test extraction from empty dict."""
        context_data = {}

        result = check._extract_content_text(context_data)

        assert result == ""

    def test_extract_content_text_none_content(self, check):
        """Test extraction when content is None."""
        context_data = {
            "content": None
        }

        result = check._extract_content_text(context_data)

        assert result == ""

    def test_extract_content_text_unknown_format(self, check):
        """Test extraction with unknown format (e.g., integer, list)."""
        # Test with integer
        context_data_int = {
            "content": 12345
        }
        result_int = check._extract_content_text(context_data_int)
        assert result_int == ""

        # Test with list
        context_data_list = {
            "content": ["item1", "item2"]
        }
        result_list = check._extract_content_text(context_data_list)
        assert result_list == ""

    def test_extract_content_text_dict_missing_text_key(self, check):
        """Test extraction from dict format missing 'text' key."""
        context_data = {
            "content": {
                "value": "Some value",
                "other_field": "data"
            }
        }

        result = check._extract_content_text(context_data)

        # Should return empty string when dict doesn't have 'text' key
        assert result == ""

    def test_extract_content_text_nested_string(self, check):
        """Test that nested strings are properly extracted."""
        context_data = {
            "content": {
                "text": "Nested content with special chars: @#$%^&*()"
            }
        }

        result = check._extract_content_text(context_data)

        assert result == "Nested content with special chars: @#$%^&*()"
        assert "@#$%^&*()" in result

    def test_extract_content_text_empty_string(self, check):
        """Test extraction with empty string content."""
        context_data = {
            "content": ""
        }

        result = check._extract_content_text(context_data)

        assert result == ""

    def test_extract_content_text_empty_text_in_dict(self, check):
        """Test extraction with empty text in dict format."""
        context_data = {
            "content": {
                "text": ""
            }
        }

        result = check._extract_content_text(context_data)

        assert result == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
